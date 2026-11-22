#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PD baseline on the MuJoCo point-mass task."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from point_mass_env import PointMassEnv, pd_action

ROOT = Path(__file__).resolve().parent


def run_episode(env, record=False):
    obs = env.reset()
    path = [obs[:2].copy()]
    total = 0.0
    steps = 0
    success = False
    while True:
        result = env.step(pd_action(obs))
        obs = result.obs
        path.append(obs[:2].copy())
        total += result.reward
        steps += 1
        if result.done:
            success = result.success
            break
    return {"return": total, "steps": steps, "success": success, "path": np.array(path), "target": obs[4:6].copy()}


def main() -> None:
    t0 = time.perf_counter()
    env = PointMassEnv(seed=42)
    episodes = [run_episode(env, record=i < 4) for i in range(100)]
    success = float(np.mean([e["success"] for e in episodes]))
    steps = float(np.mean([e["steps"] for e in episodes]))
    ret = float(np.mean([e["return"] for e in episodes]))
    print("success", success, "steps", steps, "return", ret)

    fig, ax = plt.subplots(figsize=(6, 6))
    for e in episodes[:4]:
        ax.plot(e["path"][:, 0], e["path"][:, 1], label="trajectory")
        ax.scatter(*e["target"], marker="*", s=140)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_title("MuJoCo point-mass PD baseline")
    fig.tight_layout(); fig.savefig(ROOT / "point_mass_pd.png", dpi=160)

    payload = {"env": "MuJoCo point_mass", "policy": "PD", "episodes": 100, "success_rate": round(success, 4), "mean_steps": round(steps, 2), "mean_return": round(ret, 4), "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "point_mass_pd_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
