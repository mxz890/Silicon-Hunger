#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node MZX with the lifecycle growth/prune core.

Training remains: expert -> BC -> DAgger -> lifecycle growth/prune -> PPO.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lifecycle_embodied_v2 import grow, prune, train_epoch, CYCLES, TIME_UNIT, TAU
from mzx_2000_sim2sim import (
    SRC, TGT, N_NODES, expert_action, collect, evaluate, train_bc, dagger, ppo,
)
from lifecycle_embodied_v2 import LifeGraphPolicy

ROOT = Path(__file__).resolve().parent
SEED = 50000


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(SEED); np.random.seed(SEED)
    X, Y = collect(SRC, seed=SEED + 1)
    model = LifeGraphPolicy(N_NODES, obs_dim=12)
    train_bc(model, X, Y, SEED + 2)
    bc0, bc0_steps = evaluate(model, SRC, seed=SEED + 3)
    print("BC before lifecycle", bc0, bc0_steps, flush=True)

    Xd, Yd = dagger(model, SRC, SEED + 4)
    X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
    train_bc(model, X, Y, SEED + 5)
    dg0, dg0_steps = evaluate(model, SRC, seed=SEED + 6)
    print("DAgger before lifecycle", dg0, dg0_steps, flush=True)

    Xtr, Ytr = torch.tensor(X), torch.tensor(Y)
    now = 0.0
    cycle_log = []
    for cyc in range(CYCLES):
        t_round = time.perf_counter()
        train_epoch(model, Xtr, Ytr)
        model = grow(model, Xtr, Ytr, now)
        train_epoch(model, Xtr, Ytr)
        sec = time.perf_counter() - t_round
        dt = max(sec, 1e-3) / TIME_UNIT
        now += dt
        removed, usage, retention = prune(model, Xtr[:512], now)
        s, st = evaluate(model, SRC, seed=SEED + 100 + cyc)
        cycle_log.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "success": s, "steps": st, "grew": model.n - (N_NODES + cyc * 10), "pruned": len(removed), "dt": round(dt, 3)})
        print("CYCLE", cycle_log[-1], flush=True)

    bc_life, bc_life_steps = evaluate(model, SRC, seed=SEED + 7)
    print("BC/DAgger after lifecycle", bc_life, bc_life_steps, flush=True)
    ppo_model, ppo_stats = ppo(model, SRC, SEED + 8) if bc_life >= 0.80 else (model, {"skipped": True, "bc": bc_life})
    src_s, src_steps = evaluate(ppo_model, SRC, seed=SEED + 500)
    tgt_s, tgt_steps = evaluate(ppo_model, TGT, domain="target", seed=SEED + 600)
    torch.save(ppo_model.state_dict(), ROOT / "mzx_2000_growth.pt")
    payload = {
        "nodes": N_NODES, "cycles": CYCLES, "bc_before": bc0, "dagger_before": dg0,
        "after_lifecycle": bc_life, "cycle_log": cycle_log, "ppo": ppo_stats,
        "source": src_s, "source_steps": src_steps, "target": tgt_s, "target_steps": tgt_steps,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "mzx_2000_growth_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
