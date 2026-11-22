#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core ablation A-E on MZX-500, 2 obstacles, 3 seeds.

A fixed topology
B grow only
C prune only
D grow+prune, no time
E grow+prune, normalized-time decay
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from mzx_lifecycle import grow, train_new_nodes, train_epoch
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
SEEDS = [54000, 54001, 54002]
CFG = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
GROW = 20
ROUNDS = 4
BC_EPOCHS = 10


def env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    for off in (6, 8):
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


def evaluate(model, seed=55000):
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


def prune_importance(model, X, t, use_time):
    with torch.no_grad():
        h = model.forward_full(X)
    usage = h.abs().mean(0)
    wnorm = torch.bincount(model.dst, weights=model.w.detach().abs(), minlength=model.n).float()
    hist = getattr(model, "hist_importance", torch.zeros(model.n))
    if use_time:
        lam = np.log(2.0) / 0.25
        decay = torch.exp(-lam * (t - getattr(model, "last_used", torch.zeros(model.n))))
        importance = 0.6 * usage + 0.4 * hist * decay + 0.01 * wnorm
    else:
        importance = usage + 0.01 * wnorm
    model.hist_importance = 0.8 * hist + 0.2 * usage
    candidates = [i for i in range(2, model.n - 1) if model.alive[i] > 0]
    candidates.sort(key=lambda i: float(importance[i]))
    removed = candidates[:max(1, len(candidates) // 50)]
    for i in removed:
        model.alive[i] = 0.0
    model.last_used = getattr(model, "last_used", torch.zeros(model.n))
    model.last_used[usage > 0.02] = t
    return removed, usage


def run_condition(base, X, Y, name, seed):
    model = copy.deepcopy(base)
    model.hist_importance = torch.zeros(model.n)
    model.last_used = torch.zeros(model.n)
    history = []
    for r in range(ROUNDS):
        t = (r + 1) / ROUNDS
        if name in ("B", "D", "E"):
            old_n = model.n
            old_hist = getattr(model, "hist_importance", torch.zeros(old_n))
            model, new_ids = grow(model, torch.tensor(X), torch.tensor(Y), t, grow=GROW)
            model.hist_importance = torch.cat([old_hist, torch.zeros(model.n - old_n)])
            train_new_nodes(model, torch.tensor(X), torch.tensor(Y), new_ids, epochs=2)
        if name in ("C", "D", "E"):
            removed, _ = prune_importance(model, torch.tensor(X[:512]), t, use_time=(name == "E"))
        else:
            removed = []
        s, st = evaluate(model, seed=seed + 100 + r)
        history.append({"round": r, "nodes": model.n, "alive": int(model.alive.sum()), "success": s, "steps": st, "pruned": len(removed)})
        print(name, "seed", seed, history[-1], flush=True)
    return model, history


def main() -> None:
    t0 = time.perf_counter()
    all_rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        X, Y = collect(seed + 1)
        base = LifeGraphPolicy(500, obs_dim=12)
        train_epoch(base, torch.tensor(X), torch.tensor(Y), epochs=BC_EPOCHS)
        bc_s, _ = evaluate(base, seed=seed + 2)
        print("seed", seed, "BC", bc_s, flush=True)
        Xd, Yd = dagger(base, seed + 3)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        train_epoch(base, torch.tensor(X), torch.tensor(Y), epochs=3)
        base_s, _ = evaluate(base, seed=seed + 4)
        print("seed", seed, "base after DAgger", base_s, flush=True)
        seed_row = {"seed": seed, "bc": bc_s, "base": base_s, "conditions": {}}
        for name in "ABCDE":
            model, hist = run_condition(base, X, Y, name, seed)
            seed_row["conditions"][name] = hist
        all_rows.append(seed_row)
    payload = {"seeds": all_rows, "config": CFG, "rounds": ROUNDS, "grow": GROW, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "core_ablation_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
