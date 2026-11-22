#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forgetting-curve growth/prune loop on the 2000-node math reservoir.

Time is measured per round (seconds), then normalized to a short round-unit
scale.  The forgetting curve is exponential with tau ~ 1.5 round units,
NOT biological days/years.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from grow_train_loop_v3 import Brain, train_inner, evaluate, K
from large_math_2000 import build_graph
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
DIM = 64
GROW = 20
CYCLES = 6
TIME_UNIT = 0.10       # seconds per normalized round-unit
TAU = 1.5              # round-units, deliberately short
USAGE_EPS = 0.002
RETENTION_MIN = 0.35


class LifeBrain(Brain):
    def __init__(self, n, eu, ev, ew):
        super().__init__(n, eu, ev, ew)
        self.register_buffer("alive", torch.ones(n))

    def forward_full(self, x):
        h = super().forward_full(x)
        return h * self.alive.unsqueeze(0)


def grow_life(model, X_val, y_val, errors, now):
    old_n = model.n
    old_output = model.output
    new_n = old_n + GROW
    new_output = new_n - 1
    with torch.no_grad():
        h = model.forward_full(X_val)
    worst = torch.argsort(errors, descending=True)[:max(1, GROW // 2)]
    src = model.src.cpu().numpy().tolist()
    dst = model.dst.cpu().numpy().tolist()
    ws = model.w.detach().cpu().numpy().tolist()
    existing = set(zip(src, dst))
    for j in range(old_n, new_n - 1):
        sample = int(worst[(j - old_n) % len(worst)])
        top = torch.topk(h[sample].abs()[:old_n], k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        src.append(j); dst.append(new_output); ws.append(0.05)
    src.append(old_output); dst.append(new_output); ws.append(0.05)

    new = LifeBrain(new_n, np.array(src), np.array(dst), np.array(ws))
    new.bias.data[:old_n] = model.bias.data
    new.input_gain.data[:old_n] = model.input_gain.data
    new.bias.data[old_n:] = 0.0
    new.input_gain.data[old_n:] = torch.randn(GROW, DIM) * 0.02
    new.w.data[:len(model.w)] = model.w.data
    new.w.data[len(model.w):] = 0.02
    new.alive.data[:old_n] = model.alive.data
    new.alive.data[old_n:] = 1.0
    new.last_used = torch.cat([model.last_used, torch.full((GROW,), float(now))])
    old_feat = old_n - 2
    new_feat = new_n - 2
    new.head.weight.data[:, :old_feat] = model.head.weight.data
    new.head.weight.data[:, old_feat:] = 0.0
    new.head.bias.data[:] = model.head.bias.data
    return new


def prune_life(model, X, now):
    model.eval()
    with torch.no_grad():
        h = model.forward_full(X)
    usage = h.abs().mean(0)
    retention = torch.exp(-(now - model.last_used) / TAU)
    candidates = []
    for i in range(2, model.n - 1):
        if model.alive[i] > 0 and usage[i] < USAGE_EPS and retention[i] < RETENTION_MIN:
            candidates.append(i)
    for i in candidates:
        model.alive[i] = 0.0
    used = usage > USAGE_EPS
    model.last_used[used] = now
    return candidates, usage, retention


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(2300)
    math_x, math_y = task_sum_sq(900)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    X = torch.tensor(math_x); Y_raw = torch.tensor(math_y).view(-1, 1)
    ym, ys = float(Y_raw.mean()), float(Y_raw.std() + 1e-6)
    Y = (Y_raw - ym) / ys
    Xtr, Ytr = X[:700], Y[:700]
    Xval, Yval = X[700:800], Y[700:800]
    Xte, Yte = X[800:], Y[800:]
    _, _, _, eu, ev, ew = build_graph()
    model = LifeBrain(2000, eu, ev, ew)
    model.last_used = torch.zeros(model.n)
    now = 0.0
    history = []

    for cyc in range(CYCLES + 1):
        print("CYCLE", cyc, "nodes", model.n, "alive", int(model.alive.sum()), flush=True)
        t_round = time.perf_counter()
        model = train_inner(model, Xtr, Ytr, epochs=1)
        val = evaluate(model, Xval, Yval) * ys
        test = evaluate(model, Xte, Yte) * ys
        if cyc == CYCLES:
            history.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "val_mae": round(val, 5), "test_mae": round(test, 5), "round_seconds": 0.0, "pruned": 0})
            break
        with torch.no_grad():
            errs = torch.abs(model.head(model(Xval)) - Yval).view(-1)
        model = grow_life(model, Xval, Yval, errs, now)
        model = train_inner(model, Xtr, Ytr, epochs=1)
        sec = time.perf_counter() - t_round
        dt = max(sec, 1e-3) / TIME_UNIT
        now += dt
        pruned, usage, retention = prune_life(model, Xtr[:128], now)
        val = evaluate(model, Xval, Yval) * ys
        test = evaluate(model, Xte, Yte) * ys
        history.append({"cycle": cyc, "nodes": model.n, "alive": int(model.alive.sum()), "val_mae": round(val, 5), "test_mae": round(test, 5), "round_seconds": round(sec, 4), "dt_units": round(dt, 3), "pruned": len(pruned), "retention_mean": round(float(retention.mean()), 4)})
        print("HISTORY", history[-1], flush=True)

    payload = {"mode": "grow + backprop + forgetting prune", "K": K, "grow_per_cycle": GROW, "time_unit_s": TIME_UNIT, "tau_round_units": TAU, "history": history, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "forgetting_growth_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
