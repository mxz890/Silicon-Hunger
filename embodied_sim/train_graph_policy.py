#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a graph policy for MuJoCo point-mass via PD imitation + DAgger."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from point_mass_env import PointMassEnv, pd_action
from graph_policy import GraphPolicy, collect_pd_data, train, evaluate

ROOT = Path(__file__).resolve().parent


def collect_policy_data(policy, episodes=20, seed=3010):
    env = PointMassEnv(seed=seed)
    X, Y = [], []
    for _ in range(episodes):
        obs = env.reset()
        while True:
            with torch.no_grad():
                a = policy(torch.tensor(obs, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(obs.copy())
            Y.append(pd_action(obs))
            r = env.step(a)
            obs = r.obs
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> None:
    torch.manual_seed(3005)
    X, Y, env = collect_pd_data(episodes=80, seed=3006)
    print("data", X.shape, Y.shape, flush=True)
    policy = GraphPolicy(500)
    train(policy, X, Y, epochs=12)
    succ, steps, _ = evaluate(policy, episodes=30, seed=3007)
    print("BC success", succ, "steps", steps, flush=True)
    history = [{"stage": "BC", "success": succ, "steps": steps, "samples": len(X)}]

    for dagger in range(1, 3):
        Xd, Yd = collect_policy_data(policy, episodes=20, seed=3010 + dagger)
        X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
        print("dagger", dagger, "samples", len(X), flush=True)
        train(policy, X, Y, epochs=6)
        succ, steps, paths = evaluate(policy, episodes=30, seed=3020 + dagger)
        print("DAgger", dagger, "success", succ, "steps", steps, flush=True)
        history.append({"stage": f"DAgger{dagger}", "success": succ, "steps": steps, "samples": len(X)})

    # final plot
    succ, steps, paths = evaluate(policy, episodes=50, seed=3030)
    fig, ax = plt.subplots(figsize=(6, 6))
    for path, target in paths:
        ax.plot(path[:, 0], path[:, 1])
        ax.scatter(*target, marker="*", s=140)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    ax.grid(alpha=.25); ax.set_title("Graph policy — MuJoCo point mass")
    fig.tight_layout(); fig.savefig(ROOT / "graph_policy.png", dpi=160)

    payload = {"graph_nodes": 500, "K": 6, "history": history, "final_success": succ, "final_steps": steps, "samples": len(X)}
    (ROOT / "graph_policy_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
