#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embodied lifecycle: train -> grow -> backprop -> forget/prune.

500-node graph policy on MuJoCo point mass.
Round time is measured and normalized to short round units.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from point_mass_env import PointMassEnv, pd_action
from graph_policy import GraphPolicy, build_graph, collect_pd_data

ROOT = Path(__file__).resolve().parent
DIM = 64
GROW = 20
CYCLES = 5
TIME_UNIT = 0.10      # seconds per round-unit
TAU = 2.0             # round-units, short scale
USAGE_EPS = 0.0015
RETENTION_MIN = 0.35


class LifeGraphPolicy(GraphPolicy):
    def __init__(self, n=500):
        super().__init__(n)
        self.register_buffer("alive", torch.ones(n))
        self.last_used = torch.zeros(n)

    def forward_full(self, obs):
        b = obs.shape[0]
        x = torch.zeros(b, self.in_dim)
        x[:, :self.obs_dim] = obs
        h = torch.zeros(b, self.n)
        h[:, :self.obs_dim] = obs
        term = x @ self.input_gain.T
        for _ in range(self.k):
            msg = torch.zeros(b, self.n)
            msg.index_add_(1, self.dst, self.w.unsqueeze(0) * h[:, self.src])
            h = torch.tanh(self.bias.unsqueeze(0) + term + msg)
        return h * self.alive.unsqueeze(0)

    def forward(self, obs):
        return torch.tanh(self.head(self.forward_full(obs)[:, 2:]))


def grow(model, X, Y, now):
    old_n = model.n; old_out = old_n - 1; new_n = old_n + GROW; new_out = new_n - 1
    with torch.no_grad():
        h = model.forward_full(X)
        err = (model.head(h[:, 2:]) - Y).abs().view(-1)
    worst = torch.argsort(err, descending=True)[:max(1, GROW // 2)]
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n - 1):
        s = int(worst[(j - old_n) % len(worst)]) % h.shape[0]
        top = torch.topk(h[s, :old_n].abs(), k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        src.append(j); dst.append(new_out); ws.append(0.05)
    src.append(old_out); dst.append(new_out); ws.append(0.05)

    new = LifeGraphPolicy(new_n)
    new.src = torch.tensor(src, dtype=torch.long); new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(GROW)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(GROW, DIM) * 0.02]))
    new.alive.data[:old_n] = model.alive.data
    new.alive.data[old_n:] = 1.0
    new.last_used = torch.cat([model.last_used, torch.full((GROW,), float(now))])
    old_feat = old_n - 2
    new.head = nn.Linear(new_n - 2, 2)
    with torch.no_grad():
        new.head.weight[:, :old_feat] = model.head.weight
        new.head.weight[:, old_feat:] = 0.0
        new.head.bias[:] = model.head.bias
    return new


def prune(model, X, now):
    model.eval()
    with torch.no_grad():
        h = model.forward_full(X)
    usage = h.abs().mean(0)
    retention = torch.exp(-(now - model.last_used) / TAU)
    removed = []
    for i in range(2, model.n - 1):
        if model.alive[i] > 0 and usage[i] < USAGE_EPS and retention[i] < RETENTION_MIN:
            model.alive[i] = 0.0; removed.append(i)
    model.last_used[usage > USAGE_EPS] = now
    return removed, usage, retention


def train_epoch(model, X, Y, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 64):
        idx = perm[i:i + 64]
        pred = model(X[idx])
        loss = torch.mean((pred - Y[idx]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def evaluate(model, episodes=20, seed=9001):
    env = PointMassEnv(seed=seed)
    success = 0; steps = []
    for _ in range(episodes):
        obs = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(obs, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); obs = r.obs
            if r.done:
                success += int(r.success); steps.append(env.steps); break
    return success / episodes, float(np.mean(steps))


def main() -> None:
    torch.manual_seed(9000); np.random.seed(9000)
    X, Y, _ = collect_pd_data(episodes=80, seed=9002)
    Xtr, Ytr = torch.tensor(X), torch.tensor(Y)
    model = LifeGraphPolicy(500)
    now = 0.0; history = []
    t0 = time.perf_counter()

    for cyc in range(CYCLES + 1):
        print("CYCLE", cyc, "nodes", model.n, "alive", int(model.alive.sum()), flush=True)
        t_round = time.perf_counter()
        model = train_epoch(model, Xtr, Ytr)
        if cyc == CYCLES:
            s, st = evaluate(model)
            history.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "success": s, "steps": st, "phase": "final"})
            break
        model = grow(model, Xtr, Ytr, now)
        model = train_epoch(model, Xtr, Ytr)
        sec = time.perf_counter() - t_round
        dt = max(sec, 1e-3) / TIME_UNIT
        now += dt
        removed, usage, retention = prune(model, Xtr[:256], now)
        s, st = evaluate(model)
        history.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "success": s, "steps": st, "round_seconds": round(sec, 4), "dt_units": round(dt, 3), "pruned": len(removed), "retention_mean": round(float(retention.mean()), 4)})
        print("HISTORY", history[-1], flush=True)

    payload = {"mode": "embodied train->grow->backprop->forget/prune", "grow_per_cycle": GROW, "time_unit_s": TIME_UNIT, "tau_round_units": TAU, "history": history, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "lifecycle_embodied_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
