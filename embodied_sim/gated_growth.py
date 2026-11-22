#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1-G4: gated output growth, local subgraph growth, gated local growth, 3 fixed obstacles."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lifecycle_embodied_v2 import LifeGraphPolicy
from mzx_lifecycle import train_epoch
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
SEEDS = [55000, 55001, 55002]
FIXED3 = ((0.2, 0.2), (-0.3, 0.3), (0.3, -0.3))
CFG = dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.0, target_bias=(0.5, 0.0), fixed_obstacles=FIXED3)
GROW = 20
ROUNDS = 3
BC_EPOCHS = 10


class GatedPolicy(LifeGraphPolicy):
    def __init__(self, n, obs_dim=12, head_dim=None):
        super().__init__(n, obs_dim=obs_dim)
        self.head_dim = n - 2 if head_dim is None else head_dim
        self.head = nn.Linear(self.head_dim, 2)
        self.gate = nn.Parameter(torch.ones(n))

    def forward_full(self, obs):
        h = super().forward_full(obs)
        return h * self.gate.unsqueeze(0)

    def forward(self, obs):
        h = self.forward_full(obs)
        return torch.tanh(self.head(h[:, 2:2 + self.head_dim]))


def env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    for off in (6, 8, 10):
        rel = state[off:off + 2]; d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def collect(seed, episodes=80):
    e = env(seed); X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert_action(s)
            X.append(s.copy()); Y.append(a.copy())
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, seed=56000):
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


def dagger(model, seed):
    e = env(seed); X, Y = [], []
    for _ in range(20):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(s.copy()); Y.append(expert_action(s))
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def grow_local(model, X, Y, now, gate_value):
    old_n = model.n; new_n = old_n + GROW
    with torch.no_grad():
        h = model.forward_full(X)
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n):
        act = h[:, :old_n].abs().mean(0)
        top = torch.topk(act, k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        # new node feeds a few old hidden targets; no new output
        target = int(torch.topk(act[2:old_n - 1], k=4).indices[0] + 2)
        src.append(j); dst.append(target); ws.append(0.02)

    new = GatedPolicy(new_n, obs_dim=12, head_dim=getattr(model, "head_dim", model.n - 2))
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(GROW)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(GROW, 64) * 0.02]))
    new.alive.data[:old_n] = model.alive.data
    new.alive.data[old_n:] = 1.0
    new.gate.data[:old_n] = model.gate.data
    new.gate.data[old_n:] = gate_value
    new.last_used = torch.cat([model.last_used, torch.full((GROW,), float(now))])
    with torch.no_grad():
        new.head.weight.copy_(model.head.weight)
        new.head.bias.copy_(model.head.bias)
    return new, list(range(old_n, new_n))


def grow_output(model, X, Y, now, gate_value):
    old_n = model.n; new_n = old_n + GROW; new_out = new_n - 1
    with torch.no_grad():
        h = model.forward_full(X)
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n - 1):
        act = h[:, :old_n].abs().mean(0)
        top = torch.topk(act, k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        src.append(j); dst.append(new_out); ws.append(0.02)
    src.append(old_n - 1); dst.append(new_out); ws.append(0.02)

    new = GatedPolicy(new_n, obs_dim=12, head_dim=new_n - 2)
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(GROW)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(GROW, 64) * 0.02]))
    new.alive.data[:old_n] = model.alive.data
    new.alive.data[old_n:] = 1.0
    new.gate.data[:old_n] = model.gate.data
    new.gate.data[old_n:] = gate_value
    new.last_used = torch.cat([model.last_used, torch.full((GROW,), float(now))])
    with torch.no_grad():
        new.head.weight[:, :old_n - 2] = model.head.weight
        new.head.weight[:, old_n - 2:] = 0.02
        new.head.bias.copy_(model.head.bias)
    return new, list(range(old_n, new_n))


def contribution(model, new_ids, local):
    with torch.no_grad():
        h = model.forward_full(torch.randn(128, 12))
    if local:
        mask = torch.isin(model.src, torch.tensor(new_ids))
        vals = model.w[mask].abs() * h[:, new_ids].abs().mean()
        return float(vals.mean().detach()) if len(vals) else 0.0
    feats = torch.tensor([i - 2 for i in new_ids])
    vals = model.head.weight[:, feats].abs() * h[:, new_ids].abs().mean()
    return float(vals.mean())


def train_new(model, X, Y, new_ids, local, epochs=2):
    new_mask = torch.zeros(model.n, dtype=torch.bool); new_mask[new_ids] = True
    edge_mask = new_mask[model.dst] | new_mask[model.src]
    for p in model.parameters():
        p.requires_grad_(False)
    model.bias.requires_grad_(True); model.input_gain.requires_grad_(True)
    model.w.requires_grad_(True); model.gate.requires_grad_(True)
    if not local:
        model.head.weight.requires_grad_(True)
    params = [model.bias, model.input_gain, model.w, model.gate]
    if not local:
        params.append(model.head.weight)
    opt = torch.optim.SGD(params, lr=1e-3)
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
                if not local:
                    model.head.weight.grad[:, :model.n - 2 - len(new_ids)] = 0
            opt.step()
    return model


def run_condition(base, X, Y, name, local, gate_value, seed):
    model = copy.deepcopy(base)
    model.gate = nn.Parameter(torch.ones(model.n))
    history = []
    for r in range(ROUNDS):
        before, _ = evaluate(model, seed + 100 + r)
        snap = copy.deepcopy(model)
        fn = grow_local if local else grow_output
        model, new_ids = fn(model, torch.tensor(X), torch.tensor(Y), (r + 1) / ROUNDS, gate_value)
        train_new(model, torch.tensor(X), torch.tensor(Y), new_ids, local, epochs=2)
        contrib = contribution(model, new_ids, local)
        after, steps = evaluate(model, seed + 200 + r)
        accepted = contrib > 0.001 and after >= before - 0.01
        if not accepted:
            model = snap
            after, steps = evaluate(model, seed + 200 + r)
        history.append({"round": r, "nodes": model.n, "alive": int(model.alive.sum()), "before": before, "after": after, "contrib": round(contrib, 5), "accepted": accepted})
        print(name, "seed", seed, history[-1], flush=True)
    return model, history


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        X, Y = collect(seed + 1)
        base = LifeGraphPolicy(500, obs_dim=12)
        train_epoch(base, torch.tensor(X), torch.tensor(Y), epochs=BC_EPOCHS)
        Xd, Yd = dagger(base, seed + 2)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        train_epoch(base, torch.tensor(X), torch.tensor(Y), epochs=3)
        s0, _ = evaluate(base, seed + 3)
        print("seed", seed, "base", s0, flush=True)
        row = {"seed": seed, "base": s0, "conditions": {}}
        for name, local, gate_value in [("G1_output", False, 0.1), ("G2_local", True, 1.0), ("G3_gated_local", True, 0.1)]:
            model, hist = run_condition(base, X, Y, name, local, gate_value, seed)
            row["conditions"][name] = hist
        rows.append(row)
    payload = {"seeds": rows, "rounds": ROUNDS, "grow": GROW, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "gated_growth_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
