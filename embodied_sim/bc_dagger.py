#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PD expert with obstacle repulsion + DAgger for MZX.

Checks the expert first, then BC 40 epochs, then 2 DAgger rounds.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
CFG = dict(n_obstacles=1, target_range=0.0, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))
EPOCHS = 40
BATCH = 64
LR = 1e-3


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    rel = state[6:8]
    d = float(np.linalg.norm(rel))
    if 1e-6 < d < d0:
        force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def make_env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def eval_expert(episodes=100, seed=34000):
    env = make_env(seed)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            r = env.step(expert_action(state)); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def collect(episodes=100, seed=34100):
    env = make_env(seed)
    X, Y = [], []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            a = expert_action(state)
            X.append(state.copy()); Y.append(a.copy())
            r = env.step(a); state = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, episodes=100, seed=34200):
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


def train(model, X, Y, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            pred = model(Xt[idx])
            loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()) * len(idx)
        if ep % 5 == 0:
            print("epoch", ep, "mse", round(total / len(Xt), 6), flush=True)
    return model


def dagger_rollout(model, episodes=30, seed=34300):
    env = make_env(seed)
    X, Y = [], []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(state.copy()); Y.append(expert_action(state))
            r = env.step(a); state = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(34000); np.random.seed(34000)
    expert_s, expert_steps = eval_expert()
    print("EXPERT", expert_s, expert_steps, flush=True)

    X, Y = collect()
    print("demo", X.shape, flush=True)
    model = LifeGraphPolicy(500, obs_dim=12)
    train(model, X, Y, EPOCHS, 34001)
    bc_s, bc_steps = evaluate(model)
    print("BC", bc_s, bc_steps, flush=True)
    history = [{"stage": "BC", "success": bc_s, "steps": bc_steps, "samples": len(X)}]

    for d in range(1, 3):
        Xd, Yd = dagger_rollout(model, episodes=30, seed=34400 + d)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        train(model, X, Y, 10, 34500 + d)
        s, st = evaluate(model)
        print("DAgger", d, s, st, flush=True)
        history.append({"stage": f"DAgger{d}", "success": s, "steps": st, "samples": len(X)})

    torch.save(model.state_dict(), ROOT / "mzx_bc_dagger.pt")
    payload = {"expert_success": expert_s, "expert_steps": expert_steps, "history": history, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_dagger_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
