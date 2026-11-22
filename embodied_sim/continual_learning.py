#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continual MZX: carry the previous stage checkpoint into the next context.

Stages:
  0. no obstacle, small range
  1. one fixed obstacle
  2. two randomized obstacles

Compare sequential warm-start vs training each stage from scratch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lifecycle_embodied_v2 import LifeGraphPolicy, train_epoch
from point_mass_env import pd_action
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("stage0_no_obstacle", 0, 0.4, 0.3, 0.0),
    ("stage1_one_obstacle", 1, 0.8, 0.6, 0.0),
    ("stage2_two_obstacles", 2, 1.0, 0.8, 1.0),
]


def make_env(stage, seed):
    _, n_obs, target, start, obs_range = stage
    return VisionPointMassEnv(seed=seed, render=False, size=8, n_obstacles=n_obs, target_range=target, start_range=start, obstacle_range=obs_range)


def collect(stage, episodes=60, seed=10000):
    env = make_env(stage, seed)
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


def evaluate(model, stage, episodes=20, seed=11000):
    env = make_env(stage, seed)
    success = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); state = r.state
            if r.done:
                success += int(r.success); steps.append(env.steps); break
    return success / episodes, float(np.mean(steps))


def save_checkpoint(model, path):
    torch.save({
        "n": model.n, "obs_dim": model.obs_dim,
        "src": model.src.cpu(), "dst": model.dst.cpu(), "w": model.w.detach().cpu(),
        "bias": model.bias.detach().cpu(), "input_gain": model.input_gain.detach().cpu(),
        "head": model.head.state_dict(), "alive": model.alive.cpu(), "last_used": model.last_used.cpu(),
    }, path)


def load_checkpoint(path):
    ck = torch.load(path, weights_only=False)
    model = LifeGraphPolicy(int(ck["n"]), obs_dim=int(ck["obs_dim"]))
    model.src = ck["src"]; model.dst = ck["dst"]
    model.w = nn.Parameter(ck["w"]); model.bias = nn.Parameter(ck["bias"]); model.input_gain = nn.Parameter(ck["input_gain"])
    model.head.load_state_dict(ck["head"])
    model.alive.data = ck["alive"].clone(); model.last_used = ck["last_used"].clone()
    return model


def train_stage(model, stage, seed):
    X, Y = collect(stage, seed=seed)
    return train_epoch(model, torch.tensor(X), torch.tensor(Y))


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(11000); np.random.seed(11000)
    # sequential: carry checkpoint
    seq = LifeGraphPolicy(500, obs_dim=12)
    seq_hist = []
    for name, n_obs, target, start, obs_range in STAGES:
        seq = train_stage(seq, (name, n_obs, target, start, obs_range), seed=13000 + len(seq_hist))
        s, st = evaluate(seq, (name, n_obs, target, start, obs_range), seed=14000 + len(seq_hist))
        seq_hist.append({"stage": name, "success": s, "steps": st, "nodes": seq.n})
        save_checkpoint(seq, ROOT / f"mzx_{name}.pt")
        seq = load_checkpoint(ROOT / f"mzx_{name}.pt")
        print("SEQ", seq_hist[-1], flush=True)

    # scratch: new model each stage
    scratch_hist = []
    for i, (name, n_obs, target, start, obs_range) in enumerate(STAGES):
        m = LifeGraphPolicy(500, obs_dim=12)
        m = train_stage(m, (name, n_obs, target, start, obs_range), seed=15000 + i)
        s, st = evaluate(m, (name, n_obs, target, start, obs_range), seed=16000 + i)
        scratch_hist.append({"stage": name, "success": s, "steps": st})
        print("SCRATCH", scratch_hist[-1], flush=True)

    # forgetting: final model on all previous stages
    forget = []
    for i, (name, n_obs, target, start, obs_range) in enumerate(STAGES):
        s, st = evaluate(seq, (name, n_obs, target, start, obs_range), seed=17000 + i)
        forget.append({"stage": name, "success": s, "steps": st})
        print("FINAL-ON", forget[-1], flush=True)

    payload = {"sequential": seq_hist, "scratch": scratch_hist, "final_model_on_all_stages": forget, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "continual_learning_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
