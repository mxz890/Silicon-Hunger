#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Growth-count scale sweep on 3 fixed obstacles, then zero-shot MuJoCo 2-obstacle test.

Only growth-per-round changes; base BC, data, gate, threshold and rounds are fixed.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
FIXED3 = ((0.2, 0.2), (-0.3, 0.3), (0.3, -0.3))
CFG3 = dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.0, target_bias=(0.5, 0.0), fixed_obstacles=FIXED3)
CFG2 = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
COUNTS = [0, 20, 50, 100, 200, 400]
ROUNDS = 5
THRESH = 0.0001


class GatedPolicy(LifeGraphPolicy):
    def __init__(self, n, obs_dim=12):
        super().__init__(n, obs_dim=obs_dim)
        self.gate = nn.Parameter(torch.ones(n))

    def forward_full(self, obs):
        return super().forward_full(obs) * self.gate.unsqueeze(0)


def env(cfg, seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)


def expert(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55, kt=1.0):
    p, v, t = state[:2], state[2:4], state[4:6]
    f = kp * (t - p) - kd * v
    for off in (6, 8, 10):
        rel = state[off:off + 2]; d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            f = f + eta * (1 / d - 1 / d0) * (-rel / d)
            perp = np.array([-rel[1], rel[0]]) / d
            cross = rel[0] * (t - p)[1] - rel[1] * (t - p)[0]
            f = f + kt * (1 / d - 1 / d0) * (perp if cross > 0 else -perp)
    return np.clip(f, -1, 1)


def collect(seed, episodes=100):
    e = env(CFG3, seed); X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert(s); X.append(s.copy()); Y.append(a.copy())
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, cfg, seed=59000):
    e = env(cfg, seed); succ = 0; steps = []
    for _ in range(100):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / 100, float(np.mean(steps))


def grow_high_error(model, X, Y, grow):
    old_n = model.n; new_n = old_n + grow; new_out = new_n - 1
    with torch.no_grad():
        h = model.forward_full(X)
        err = (model.head(h[:, 2:]) - Y).abs().mean(1)
    top_samples = torch.topk(err, k=max(1, len(X) // 10)).indices
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n - 1):
        idx = int(top_samples[(j - old_n) % len(top_samples)])
        top = torch.topk(h[idx, :old_n].abs(), k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        src.append(j); dst.append(new_out); ws.append(0.02)
    src.append(old_n - 1); dst.append(new_out); ws.append(0.02)

    new = GatedPolicy(new_n, obs_dim=12)
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(grow)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(grow, 64) * 0.02]))
    new.alive.data[:old_n] = model.alive.data; new.alive.data[old_n:] = 1.0
    new.gate.data[:old_n] = model.gate.data; new.gate.data[old_n:] = 0.5
    new.last_used = torch.cat([model.last_used, torch.full((grow,), 1.0)])
    with torch.no_grad():
        new.head.weight[:, :old_n - 2] = model.head.weight
        new.head.weight[:, old_n - 2:] = 0.02
        new.head.bias.copy_(model.head.bias)
    return new, list(range(old_n, new_n))


def train_new(model, X, Y, new_ids, epochs=2):
    new_mask = torch.zeros(model.n, dtype=torch.bool); new_mask[new_ids] = True
    edge_mask = new_mask[model.dst] | new_mask[model.src]
    for p in model.parameters():
        p.requires_grad_(False)
    model.bias.requires_grad_(True); model.input_gain.requires_grad_(True)
    model.w.requires_grad_(True); model.gate.requires_grad_(True); model.head.weight.requires_grad_(True)
    opt = torch.optim.SGD([model.bias, model.input_gain, model.w, model.gate, model.head.weight], lr=1e-3)
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
                model.head.weight.grad[:, :model.n - 2 - len(new_ids)] = 0
            opt.step()
    return model


def contribution(model, new_ids):
    with torch.no_grad():
        h = model.forward_full(torch.randn(128, 12))
    feats = torch.tensor([i - 2 for i in new_ids])
    vals = model.head.weight[:, feats].abs() * h[:, new_ids].abs().mean()
    return float(vals.mean().detach())


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(59000); np.random.seed(59000)
    base = GatedPolicy(500, obs_dim=12)
    state = torch.load(ROOT / "mzx_bc_3obs.pt", weights_only=False)
    base.load_state_dict(state, strict=False)
    base.gate.data[:] = 1.0
    X, Y = collect(59001)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    rows = []
    for grow in COUNTS:
        model = copy.deepcopy(base)
        hist = []
        for r in range(ROUNDS):
            if grow > 0:
                model, new_ids = grow_high_error(model, Xt, Yt, grow)
                train_new(model, Xt, Yt, new_ids)
                c = contribution(model, new_ids)
            else:
                c = 0.0
            hist.append({"round": r, "nodes": model.n, "contrib": round(c, 6)})
            print("grow", grow, "round", r, "nodes", model.n, "contrib", round(c, 6), flush=True)
        s3, st3 = evaluate(model, CFG3, seed=59200 + grow)
        s2, st2 = evaluate(model, CFG2, seed=59300 + grow)
        print("RESULT", grow, "3obs", s3, "2obs", s2, flush=True)
        rows.append({"grow_per_round": grow, "total_grown": grow * ROUNDS, "nodes": model.n, "three_fixed": s3, "three_steps": st3, "two_random": s2, "two_steps": st2, "history": hist})
    payload = {"rows": rows, "rounds": ROUNDS, "threshold": THRESH, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "growth_scale_mujoco_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
