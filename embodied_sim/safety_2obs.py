#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J safety layer: evaluate 2-obstacle +100 policy with/without expert shield."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from growth_2obs_direct import GatedPolicy
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


def evaluate(model, shield=False, seed=62000):
    e = env(seed); succ = 0; steps = []; triggers = 0
    for _ in range(100):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            if shield:
                d_obs = min(np.linalg.norm(s[6:8]), np.linalg.norm(s[8:10]))
                if d_obs < 0.45:
                    a = expert(s); triggers += 1
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / 100, float(np.mean(steps)), triggers


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(62000); np.random.seed(62000)
    state = torch.load(ROOT / "mzx_2obs_plus100.pt", weights_only=False)
    model = GatedPolicy(1000)
    model.src = state["src"].long()
    model.dst = state["dst"].long()
    model.w = torch.nn.Parameter(state["w"].float())
    model.load_state_dict(state, strict=False)
    base_s, base_steps, _ = evaluate(model, False, 62001)
    sh_s, sh_steps, triggers = evaluate(model, True, 62002)
    torch.save(model.state_dict(), ROOT / "mzx_2obs_safety.pt")
    payload = {"baseline": base_s, "baseline_steps": base_steps, "shield": sh_s, "shield_steps": sh_steps, "triggers": triggers, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "safety_2obs_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
