#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-variable: add graph growth before Stage 1 PPO.

Everything else follows curriculum_vision_v2:
20k BC demos, 30 epochs, 64x64 vision, one fixed obstacle,
4 envs, per-env GAE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from curriculum_vision_v2 import collect_demos, train_bc, ppo_finetune, eval_policy
from vision_graph_ppo import VisionGraphAC, prep, OBS_DIM
from graph_policy import build_graph

ROOT = Path(__file__).resolve().parent
GROW = 200


def grow_model(model, data, n_new=GROW):
    old_n = model.n
    new_n = old_n + n_new
    new_output = new_n - 1
    old_output = old_n - 1

    sample = data[:64]
    images = torch.stack([prep(x[0], x[1])[0][0] for x in sample])
    states = torch.stack([prep(x[0], x[1])[1][0] for x in sample])
    with torch.no_grad():
        h = model.graph_features(images, states).mean(0).abs()

    old_src = model.src.cpu().numpy().tolist()
    old_dst = model.dst.cpu().numpy().tolist()
    old_w = model.w.detach().cpu().numpy().tolist()
    existing = set(zip(old_src, old_dst))
    src, dst, ws = list(old_src), list(old_dst), list(old_w)

    for j in range(old_n, new_n - 1):
        top = torch.topk(h[:old_n], k=8).indices.tolist()
        added = 0
        for u in top:
            if added >= 4:
                break
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02)
                existing.add((u, j)); added += 1
        src.append(j); dst.append(new_output); ws.append(0.05)
    src.append(old_output); dst.append(new_output); ws.append(0.05)

    new = VisionGraphAC(new_n)
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(n_new)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(n_new, 64) * 0.02]))
    old_feat = old_n - 2 + OBS_DIM
    new_feat = new_n - 2 + OBS_DIM
    new.actor = nn.Linear(new_feat, 2)
    new.critic = nn.Linear(new_feat, 1)
    with torch.no_grad():
        new.actor.weight[:, :old_feat] = model.actor.weight
        new.actor.weight[:, old_feat:] = 0
        new.actor.bias[:] = model.actor.bias
        new.critic.weight[:, :old_feat] = model.critic.weight
        new.critic.weight[:, old_feat:] = 0
        new.critic.bias[:] = model.critic.bias
    new.log_std = nn.Parameter(model.log_std.detach().clone())
    print("grow", old_n, "->", new_n, "edges", len(ws), flush=True)
    return new


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(8200); np.random.seed(8200)
    model = VisionGraphAC(2000)
    data = collect_demos()
    print("demos", len(data), flush=True)
    train_bc(model, data)
    s0, st0 = eval_policy(model, n_obstacles=0, target_range=0.4, start_range=0.3, episodes=30, seed=8300)
    print("STAGE0", s0, st0, flush=True)
    model = grow_model(model, data)
    s0g, st0g = eval_policy(model, n_obstacles=0, target_range=0.4, start_range=0.3, episodes=30, seed=8301)
    print("STAGE0 GROWN", s0g, st0g, flush=True)
    ppo_finetune(model)
    s1, st1 = eval_policy(model, n_obstacles=1, target_range=0.8, start_range=0.6, episodes=30, seed=8400)
    s2, st2 = eval_policy(model, n_obstacles=1, target_range=1.0, start_range=0.7, domain="target", episodes=30, seed=8500)
    payload = {
        "growth_nodes": GROW, "final_nodes": model.n,
        "stage0_before_growth": s0, "stage0_after_growth": s0g,
        "stage1_success": s1, "stage1_steps": st1,
        "target_success": s2, "target_steps": st2,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "growth_curriculum_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
