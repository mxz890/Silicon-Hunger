#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triggered DAgger + dynamic safety shield for MZX-500."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
CFG = dict(n_obstacles=1, target_range=0.3, start_range=0.5, obstacle_range=0.5, target_bias=(0.5, 0.0))


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    rel = state[6:8]; d = float(np.linalg.norm(rel))
    if 1e-6 < d < d0:
        force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def env(seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **CFG)


def action(model, state):
    with torch.no_grad():
        return model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()


def evaluate(model, episodes=100, seed=55000, shield=False):
    e = env(seed); succ = 0; steps = []
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = action(model, s)
            expert = expert_action(s)
            dist_obs = np.linalg.norm(s[6:8])
            triggered = shield and dist_obs < 0.45 and np.linalg.norm(a - expert) > 0.15
            r = e.step(expert if triggered else a)
            s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / episodes, float(np.mean(steps))


def trigger_dagger(model, episodes=30, seed=55100):
    e = env(seed); X, Y = [], []
    triggers = 0
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            a = action(model, s); expert = expert_action(s)
            dist_obs = np.linalg.norm(s[6:8])
            triggered = dist_obs < 0.4 or np.linalg.norm(a - expert) > 0.25
            if triggered:
                triggers += 1
            X.append(s.copy()); Y.append(expert)
            r = e.step(expert if triggered else a)
            s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), triggers


def train(model, X, Y, seed, epochs=8):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt)); total = 0.0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            pred = model(Xt[idx]); loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()) * len(idx)
        print("epoch", ep, round(total / len(Xt), 6), flush=True)
    return model


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(55000); np.random.seed(55000)
    model = LifeGraphPolicy(500, obs_dim=12)
    model.load_state_dict(torch.load(ROOT / "mzx_bc_dagger.pt", weights_only=False))
    base_s, base_steps = evaluate(model, seed=55100)
    print("BC baseline on new task", base_s, base_steps, flush=True)

    X, Y, triggers = trigger_dagger(model)
    train(model, X, Y, 55200, epochs=8)
    td_s, td_steps = evaluate(model, seed=55300)
    shield_s, shield_steps = evaluate(model, seed=55400, shield=True)
    torch.save(model.state_dict(), ROOT / "mzx_triggered_dagger.pt")
    payload = {
        "baseline_success": base_s, "baseline_steps": base_steps,
        "trigger_samples": len(X), "triggers": triggers,
        "triggered_dagger_success": td_s, "triggered_dagger_steps": td_steps,
        "shield_success": shield_s, "shield_steps": shield_steps,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "triggered_dagger_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
