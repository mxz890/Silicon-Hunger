#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the two main +100 growth baselines: 2-obstacle and 3-obstacle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from growth_2obs_direct import GatedPolicy, grow_high_error, train_new, evaluate
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
CFG2 = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
FIXED3 = ((0.2, 0.2), (-0.3, 0.3), (0.3, -0.3))
CFG3 = dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.0, target_bias=(0.5, 0.0), fixed_obstacles=FIXED3)


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


def collect(cfg, seed, episodes=100):
    e = VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)
    X, Y = [], []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = expert(s)
            X.append(s.copy()); Y.append(a.copy())
            r = e.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build(seed, ckpt, cfg, out_name):
    torch.manual_seed(seed); np.random.seed(seed)
    base = GatedPolicy(500, obs_dim=12)
    base.load_state_dict(torch.load(ROOT / ckpt, weights_only=False), strict=False)
    base.gate.data[:] = 1.0
    X, Y = collect(cfg, seed + 1)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    model = base
    for r in range(5):
        model, new_ids = grow_high_error(model, Xt, Yt, 100)
        train_new(model, Xt, Yt, new_ids)
        print(out_name, "round", r, "nodes", model.n, flush=True)
    s, st = evaluate(model, cfg, seed=seed + 100)
    torch.save(model.state_dict(), ROOT / out_name)
    print(out_name, "success", s, st, flush=True)
    return {"checkpoint": out_name, "success": s, "steps": st, "nodes": model.n}


def main() -> None:
    t0 = time.perf_counter()
    two = build(60200, "mzx_bc_2obs_direct.pt", CFG2, "mzx_2obs_plus100.pt")
    three = build(60300, "mzx_bc_3obs.pt", CFG3, "mzx_3obs_plus100.pt")
    payload = {"two_obstacle": two, "three_obstacle": three, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "main_baselines_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
