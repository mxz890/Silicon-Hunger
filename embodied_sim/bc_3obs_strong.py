#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get 3-obstacle BC above 80% before any growth/gate work."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from mzx_lifecycle import train_epoch
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
FIXED3 = ((0.2, 0.2), (-0.3, 0.3), (0.3, -0.3))
CFG = dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.0, target_bias=(0.5, 0.0), fixed_obstacles=FIXED3)


def env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55, kt=1.0):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    goal_dir = t - p
    for off in (6, 8, 10):
        rel = state[off:off + 2]; d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
            perp = np.array([-rel[1], rel[0]]) / d
            cross = rel[0] * goal_dir[1] - rel[1] * goal_dir[0]
            force = force + kt * (1.0 / d - 1.0 / d0) * (perp if cross > 0 else -perp)
    return np.clip(force, -1.0, 1.0)


def collect(seed, episodes=200):
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


def evaluate(model, seed=57000):
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


def dagger(model, seed, episodes=30):
    e = env(seed); X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(s.copy()); Y.append(expert_action(s))
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(57000); np.random.seed(57000)
    e = env(57002); ex_succ = 0
    for _ in range(100):
        _, s = e.reset()
        while True:
            r = e.step(expert_action(s)); s = r.state
            if r.done:
                ex_succ += int(r.success); break
    ex_s = ex_succ / 100
    print("expert", ex_s, flush=True)

    X, Y = collect(57003)
    model = LifeGraphPolicy(500, obs_dim=12)
    train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=40)
    bc_s, bc_steps = evaluate(model)
    print("BC initial", bc_s, bc_steps, flush=True)

    for r in range(2):
        Xd, Yd = dagger(model, 57100 + r)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=10)
        s, st = evaluate(model, 57200 + r)
        print("DAgger", r, s, st, flush=True)

    s, st = evaluate(model, 57300)
    torch.save(model.state_dict(), ROOT / "mzx_bc_3obs.pt")
    payload = {"expert": ex_s, "samples": len(X), "bc_initial": bc_s, "bc_steps": bc_steps, "final": s, "final_steps": st, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_3obs_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
