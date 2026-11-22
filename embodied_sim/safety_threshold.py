#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J: safety layer on the I gated-local model + threshold ablation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gated_local_final import GatedLocalPolicy, env, expert, ROOT

CFG = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))


def load_model():
    state = torch.load(ROOT / "mzx_gated_local_seed62100.pt", weights_only=False)
    model = GatedLocalPolicy(1150, 998)
    model.src = state["src"].long()
    model.dst = state["dst"].long()
    model.w = nn.Parameter(state["w"].float())
    model.load_state_dict(state, strict=False)
    return model


def evaluate(model, threshold=None, seed=63000):
    e = env(seed); succ = 0; steps = []; triggers = 0
    for _ in range(100):
        _, s = e.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            if threshold is not None:
                d_obs = min(np.linalg.norm(s[6:8]), np.linalg.norm(s[8:10]))
                if d_obs < threshold:
                    a = expert(s); triggers += 1
            r = e.step(a); s = r.state
            if r.done:
                succ += int(r.success); steps.append(e.steps); break
    return succ / 100, float(np.mean(steps)), triggers


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(63000); np.random.seed(63000)
    model = load_model()
    base_s, base_steps, _ = evaluate(model, None, 63002)
    rows = []
    for th in (0.0, 0.10, 0.20, 0.30, 0.45, 0.60):
        s, st, tr = evaluate(model, th, 63002)
        rows.append({"threshold": th, "success": s, "steps": st, "triggers": tr})
        print("threshold", th, s, st, "triggers", tr, flush=True)
    payload = {"baseline": base_s, "baseline_steps": base_steps, "rows": rows, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "safety_threshold_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
