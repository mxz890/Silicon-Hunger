#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node growth/prune with 3 obstacles, 5 seeds, rollback, node observation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import LifeGraphPolicy
from mzx_lifecycle import train_epoch, grow, prune, snapshot
from mzx_2000_sim2sim import ppo, evaluate
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
SEEDS = [52000, 52001, 52002, 52003, 52004]
CFG = dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
TGT = dict(n_obstacles=3, target_range=0.8, start_range=0.7, obstacle_range=0.8, target_bias=(0.5, 0.0))
CYCLES = 4
GROW = 20
BC_EPOCHS = 8


def make_env(cfg, seed, domain="source"):
    return VisionPointMassEnv(seed=seed, render=False, size=8, domain=domain, obs_noise=(0.05 if domain == "target" else 0.0), **cfg)


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    for off in (6, 8, 10):
        rel = state[off:off + 2]
        d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def collect(episodes=80, seed=53000):
    env = make_env(CFG, seed)
    X, Y = [], []
    for _ in range(episodes):
        _, s = env.reset()
        while True:
            a = expert_action(s)
            X.append(s.copy()); Y.append(a.copy())
            r = env.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def eval_model(model, cfg=CFG, domain="source", seed=53100):
    env = make_env(cfg, seed, domain)
    succ = 0; steps = []
    for _ in range(100):
        _, s = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / 100, float(np.mean(steps))


def train_bc(model, X, Y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=BC_EPOCHS)
    return model


def dagger(model, seed):
    X, Y = [], []
    env = make_env(CFG, seed)
    for _ in range(20):
        _, s = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(s.copy()); Y.append(expert_action(s))
            r = env.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = collect(seed=seed + 1)
    model = LifeGraphPolicy(2000, obs_dim=12)
    train_bc(model, X, Y, seed + 2)
    s0, _ = eval_model(model, seed=seed + 3)
    print("seed", seed, "BC", s0, flush=True)
    Xd, Yd = dagger(model, seed + 4)
    X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
    train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=3)
    s1, _ = eval_model(model, seed=seed + 5)
    print("seed", seed, "dagger", s1, flush=True)
    history = []
    now = 0.0
    for cyc in range(CYCLES):
        before, _ = eval_model(model, seed=seed + 100 + cyc)
        snap = snapshot(model)
        t0 = time.perf_counter()
        train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=2)
        model, new_ids = grow(model, torch.tensor(X), torch.tensor(Y), now, grow=GROW)
        train_epoch(model, torch.tensor(X), torch.tensor(Y), epochs=2)
        sec = time.perf_counter() - t0
        now += max(sec, 1e-3) / 0.1
        removed, usage, retention = prune(model, torch.tensor(X[:512]), now)
        after, steps = eval_model(model, seed=seed + 200 + cyc)
        accepted = after >= before - 0.01
        if not accepted:
            model = snap
            after, steps = eval_model(model, seed=seed + 200 + cyc)
        history.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "before": before, "after": after, "accepted": accepted, "grew": GROW, "pruned": len(removed), "new_usage": round(float(usage[new_ids].mean()), 5) if new_ids else 0.0, "round_s": round(sec, 2)})
        print("CYCLE", seed, history[-1], flush=True)
    bc_life, _ = eval_model(model, seed=seed + 6)
    if bc_life >= 0.80:
        model, ppo_stats = ppo(model, CFG, seed + 7)
    else:
        ppo_stats = {"skipped": True, "bc": bc_life}
    src, src_steps = eval_model(model, seed=seed + 500)
    tgt, tgt_steps = eval_model(model, TGT, domain="target", seed=seed + 600)
    torch.save(model.state_dict(), ROOT / f"mzx_growth3_seed{seed}.pt")
    return {"seed": seed, "bc": s0, "dagger": s1, "after_lifecycle": bc_life, "cycles": history, "ppo": ppo_stats, "source": src, "source_steps": src_steps, "target": tgt, "target_steps": tgt_steps}


def main() -> None:
    t0 = time.perf_counter()
    rows = [run_seed(s) for s in SEEDS]
    payload = {"nodes": 2000, "seeds": rows, "cycles": CYCLES, "grow": GROW, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "mzx_growth3_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
