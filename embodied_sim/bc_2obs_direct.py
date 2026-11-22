#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2-obstacle BC trained directly, no zero-shot from 3 obstacles."""

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
CFG = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))


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


def collect(seed, episodes=200):
    e = env(seed); X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert(s); X.append(s.copy()); Y.append(a.copy())
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, seed=60000):
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
            X.append(s.copy()); Y.append(expert(s))
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(60000); np.random.seed(60000)
    X, Y = collect(60001)
    model = LifeGraphPolicy(500, obs_dim=12)
    train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=40)
    bc_s, bc_steps = evaluate(model)
    print("BC", bc_s, bc_steps, flush=True)
    for r in range(2):
        Xd, Yd = dagger(model, 60100 + r)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=10)
        s, st = evaluate(model, 60200 + r)
        print("DAgger", r, s, st, flush=True)
    s, st = evaluate(model, 60300)
    torch.save(model.state_dict(), ROOT / "mzx_bc_2obs_direct.pt")
    payload = {"samples": len(X), "bc": bc_s, "bc_steps": bc_steps, "final": s, "final_steps": st, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_2obs_direct_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
