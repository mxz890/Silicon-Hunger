#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I: final gated local subgraph on the 2-obstacle +100 baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from growth_2obs_direct import GatedPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
CFG = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
GROW = 50
ROUNDS = 3
THRESH = 0.0001
SEEDS = [62100, 62101, 62102]


class GatedLocalPolicy(GatedPolicy):
    def __init__(self, n, head_dim):
        super().__init__(n)
        self.head_dim = head_dim
        self.head = nn.Linear(head_dim, 2)

    def forward(self, obs):
        h = self.forward_full(obs)
        return torch.tanh(self.head(h[:, 2:2 + self.head_dim]))


def env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def expert(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55, kt=1.0):
    p, v, t = state[:2], state[2:4], state[4:6]
    f = kp * (t - p) - kd * v
    for off in (6, 8):
        rel = state[off:off + 2]; d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            f = f + eta * (1 / d - 1 / d0) * (-rel / d)
            perp = np.array([-rel[1], rel[0]]) / d
            cross = rel[0] * (t - p)[1] - rel[1] * (t - p)[0]
            f = f + kt * (1 / d - 1 / d0) * (perp if cross > 0 else -perp)
    return np.clip(f, -1, 1)


def collect(seed, episodes=100):
    e = env(seed); X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert(s); X.append(s.copy()); Y.append(a.copy())
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, seed=62200):
    e = env(seed); succ = 0; steps = []
    for _ in range(100):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / 100, float(np.mean(steps))


def load_base(seed):
    state = torch.load(ROOT / "mzx_2obs_plus100.pt", weights_only=False)
    model = GatedLocalPolicy(1000, 998)
    model.src = state["src"].long(); model.dst = state["dst"].long()
    model.w = nn.Parameter(state["w"].float())
    model.load_state_dict(state, strict=False)
    return model


def grow_local(model, X, Y, grow):
    old_n = model.n; new_n = old_n + grow
    with torch.no_grad():
        h = model.forward_full(X)
        err = (model.head(h[:, 2:model.head_dim + 2]) - torch.tensor(Y)).abs().mean(1)
    top_samples = torch.topk(err, k=max(1, len(X) // 10)).indices
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n):
        idx = int(top_samples[(j - old_n) % len(top_samples)])
        top = torch.topk(h[idx, 2:old_n - 1].abs(), k=4).indices + 2
        for u in top.tolist():
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        # new node feeds old hidden target; no new output
        target = int(torch.topk(h[idx, 2:old_n - 1].abs(), k=1).indices[0] + 2)
        src.append(j); dst.append(target); ws.append(0.02)
    new = GatedLocalPolicy(new_n, model.head_dim)
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(grow)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(grow, 64) * 0.02]))
    new.alive.data[:old_n] = model.alive.data; new.alive.data[old_n:] = 1.0
    new.gate.data[:old_n] = model.gate.data; new.gate.data[old_n:] = 0.5
    new.last_used = torch.cat([model.last_used, torch.full((grow,), 1.0)])
    with torch.no_grad():
        new.head.weight.copy_(model.head.weight)
        new.head.bias.copy_(model.head.bias)
    return new, list(range(old_n, new_n))


def train_new(model, X, Y, new_ids, epochs=2):
    new_mask = torch.zeros(model.n, dtype=torch.bool); new_mask[new_ids] = True
    edge_mask = new_mask[model.dst] | new_mask[model.src]
    for p in model.parameters():
        p.requires_grad_(False)
    model.bias.requires_grad_(True); model.input_gain.requires_grad_(True)
    model.w.requires_grad_(True); model.gate.requires_grad_(True)
    opt = torch.optim.SGD([model.bias, model.input_gain, model.w, model.gate], lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            pred = model(X[idx]); loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                model.bias.grad[~new_mask] = 0
                model.input_gain.grad[~new_mask] = 0
                model.w.grad[~edge_mask] = 0
                model.gate.grad[~new_mask] = 0
            opt.step()
    return model


def contribution(model, new_ids):
    with torch.no_grad():
        h = model.forward_full(torch.randn(128, 12))
    mask = torch.isin(model.src, torch.tensor(new_ids))
    vals = model.w[mask].abs() * h[:, new_ids].abs().mean()
    return float(vals.mean().detach()) if len(vals) else 0.0


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        base = load_base(seed)
        X, Y = collect(seed + 1)
        Xt, Yt = torch.tensor(X), torch.tensor(Y)
        base_s, base_steps = evaluate(base, seed + 2)
        model = base
        hist = []
        for r in range(ROUNDS):
            model, new_ids = grow_local(model, Xt, Yt, GROW)
            train_new(model, Xt, Yt, new_ids)
            c = contribution(model, new_ids)
            if c < THRESH:
                model = base; c = 0.0
            s, st = evaluate(model, seed + 300 + r)
            hist.append({"round": r, "nodes": model.n, "success": s, "contrib": round(c, 6), "accepted": c >= THRESH})
            print("seed", seed, "round", r, hist[-1], flush=True)
        rows.append({"seed": seed, "base": base_s, "base_steps": base_steps, "history": hist})
        torch.save(model.state_dict(), ROOT / f"mzx_gated_local_seed{seed}.pt")
    payload = {"seeds": rows, "threshold": THRESH, "grow": GROW, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "gated_local_final_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
