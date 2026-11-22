#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Obstacle-count scaling test: 1, 2, 3 obstacles with the I model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gated_local_final import GatedLocalPolicy, expert
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent


def load_model():
    state = torch.load(ROOT / "mzx_gated_local_seed62100.pt", weights_only=False)
    model = GatedLocalPolicy(1150, 998)
    model.src = state["src"].long()
    model.dst = state["dst"].long()
    model.w = nn.Parameter(state["w"].float())
    model.load_state_dict(state, strict=False)
    return model


def make_env(n, seed):
    cfg = dict(n_obstacles=n, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
    return VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)


def evaluate(model, n, shield_threshold=None, episodes=100, seed=64000):
    e = make_env(n, seed); succ = 0; steps = []; triggers = 0
    for _ in range(episodes):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            if shield_threshold is not None:
                ds = [np.linalg.norm(s[6:8]), np.linalg.norm(s[8:10]), np.linalg.norm(s[10:12])]
                if min(ds) < shield_threshold:
                    a = expert(s); triggers += 1
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / episodes, float(np.mean(steps)), triggers


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(64000); np.random.seed(64000)
    model = load_model()
    rows = []
    for n in (1, 2, 3):
        base_s, base_steps, _ = evaluate(model, n, None, seed=64001)
        sh_s, sh_steps, tr = evaluate(model, n, 0.30, seed=64002)
        rows.append({"obstacles": n, "baseline": base_s, "baseline_steps": base_steps, "shield": sh_s, "shield_steps": sh_steps, "triggers": tr})
        print("n", n, "base", base_s, "shield", sh_s, flush=True)
    payload = {"rows": rows, "seconds": round(time.perf_counter() - t0, 3), "note": "obs is 12-D; only 3 obstacle slots are visible"}
    (ROOT / "obstacle_scaling_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
