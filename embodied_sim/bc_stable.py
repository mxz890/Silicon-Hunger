#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BC stability: fixed graph initialization, 5 seeds, 1 fixed obstacle.

Validation is 50 episodes; train/val split and best checkpoint are used before
any PPO.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from point_mass_env import pd_action
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
SEEDS = [33000, 33001, 33002, 33003, 33004]
CFG = dict(n_obstacles=1, target_range=0.0, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))
EPOCHS = 12
BATCH = 64
LR = 1e-3


def make_env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def collect(episodes=100, seed=33100):
    env = make_env(seed)
    X, Y = [], []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            a = pd_action(state[:6])
            X.append(state.copy()); Y.append(a.copy())
            r = env.step(a); state = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, episodes=50, seed=33200):
    env = make_env(seed)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def train_one(base, Xtr, Ytr, Xval, Yval, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = copy.deepcopy(base)
    model.alive.fill_(1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, _ = evaluate(model, episodes=30, seed=33300 + seed % 100)
    best_state = copy.deepcopy(model.state_dict())
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xval_t, Yval_t = torch.tensor(Xval), torch.tensor(Yval)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), BATCH):
            idx = perm[i:i + BATCH]
            pred = model(Xtr_t[idx])
            loss = torch.mean((pred - Ytr_t[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val_s, _ = evaluate(model, episodes=30, seed=33400 + seed % 100 + ep)
        if val_s > best_val:
            best_val = val_s; best_state = copy.deepcopy(model.state_dict())
        print("seed", seed, "epoch", ep, "val", val_s, "best", best_val, flush=True)
    model.load_state_dict(best_state)
    return model, best_val


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(33000)
    # Fixed graph + fixed initial weights for all BC runs.
    base = LifeGraphPolicy(500, obs_dim=12)
    X, Y = collect()
    n = len(X)
    rng = np.random.default_rng(33500)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    tr, va = perm[:split], perm[split:]
    Xtr, Ytr = X[tr], Y[tr]
    Xval, Yval = X[va], Y[va]
    results = []
    for seed in SEEDS:
        model, val = train_one(base, Xtr, Ytr, Xval, Yval, seed)
        s, st = evaluate(model, episodes=50, seed=33600 + seed % 100)
        results.append({"seed": seed, "val": val, "test_success": s, "test_steps": st})
        print("RESULT", results[-1], flush=True)
        torch.save(model.state_dict(), ROOT / f"mzx_bc_stable_seed{seed}.pt")
    mean = float(np.mean([r["test_success"] for r in results]))
    std = float(np.std([r["test_success"] for r in results]))
    payload = {"config": CFG, "samples": int(n), "train": int(len(tr)), "val": int(len(va)), "seeds": results, "mean_success": round(mean, 4), "std_success": round(std, 4), "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_stable_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
