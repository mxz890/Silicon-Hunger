#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F/G/H minimal advanced ablation on 500-node 2-obstacle task.

F: normalized-time input
G: time only as schedule / no input
H: pruning with half-life 0.05, 0.10, 0.25
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
CFG = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
MAX_STEPS = 250
EPOCHS = 20


class TimePolicy(LifeGraphPolicy):
    def __init__(self, n=500):
        super().__init__(n, obs_dim=13)


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


def collect(seed, episodes=120):
    e = env(seed); X, Y, T = [], [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert(s)
            X.append(s.copy()); Y.append(a.copy()); T.append(e.steps / MAX_STEPS)
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(T, dtype=np.float32)


def train(model, X, Y, T=None, epochs=EPOCHS, seed=61000):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    Tt = None if T is None else torch.tensor(T)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt)); total = 0.0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            inp = Xt[idx] if Tt is None else torch.cat([Xt[idx], Tt[idx].view(-1, 1)], 1)
            pred = model(inp); loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()) * len(idx)
        if ep % 10 == 0:
            print("epoch", ep, round(total / len(Xt), 6), flush=True)
    return model


def evaluate(model, timed=False, seed=61100):
    e = env(seed); succ = 0; steps = []
    for _ in range(100):
        _, s = e.reset()
        while True:
            inp = s if not timed else np.concatenate([s, [e.steps / MAX_STEPS]]).astype(np.float32)
            with torch.no_grad():
                a = model(torch.tensor(inp, dtype=torch.float32).view(1, -1))[0].numpy()
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / 100, float(np.mean(steps))


def prune_round(model, X, t, half_life):
    with torch.no_grad():
        h = model.forward_full(X)
    usage = h.abs().mean(0)
    wnorm = torch.bincount(model.dst, weights=model.w.detach().abs(), minlength=model.n).float()
    last = getattr(model, "last_used", torch.zeros(model.n))
    hist = getattr(model, "hist_importance", torch.zeros(model.n))
    lam = np.log(2.0) / half_life
    imp = 0.6 * usage + 0.4 * hist * torch.exp(-lam * (t - last)) + 0.01 * wnorm
    model.hist_importance = 0.8 * hist + 0.2 * usage
    model.last_used = last
    model.last_used[usage > 0.02] = t
    candidates = [i for i in range(2, model.n - 1) if model.alive[i] > 0]
    candidates.sort(key=lambda i: float(imp[i]))
    removed = candidates[:max(1, len(candidates) // 20)]
    for i in removed:
        model.alive[i] = 0.0
    return removed


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(61000); np.random.seed(61000)
    X, Y, T = collect(61001)
    print("samples", len(X), flush=True)

    g = LifeGraphPolicy(500, obs_dim=12)
    train(g, X, Y, None, seed=61002)
    gs, _ = evaluate(g, False, 61100)
    print("G schedule", gs, flush=True)

    f = TimePolicy(500)
    train(f, X, Y, T, seed=61003)
    fs, _ = evaluate(f, True, 61100)
    print("F time input", fs, flush=True)

    h_rows = {}
    for hl in (0.05, 0.10, 0.25):
        m = copy.deepcopy(g)
        m.hist_importance = torch.zeros(m.n)
        m.last_used = torch.zeros(m.n)
        hist = []
        for r in range(5):
            removed = prune_round(m, torch.tensor(X[:512]), (r + 1) / 5, hl)
            s, _ = evaluate(m, False, 61200 + r)
            hist.append({"round": r, "alive": int(m.alive.sum()), "success": s, "pruned": len(removed)})
        h_rows[str(hl)] = hist
        print("H", hl, hist[-1], flush=True)

    payload = {"samples": len(X), "F_time_input": fs, "G_schedule": gs, "H": h_rows, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "advanced_fgh_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
