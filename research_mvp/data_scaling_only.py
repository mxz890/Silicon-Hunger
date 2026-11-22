#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-only scaling: fixed 2000-node graph, K=8, no growth."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from grow_train_loop_v3 import Brain, train_inner, evaluate, K, EPOCHS, BATCH, LR
from large_math_2000 import build_graph
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
DIM = 64
SEED = 2180
SIZES = [500, 1000, 2000, 5000, 10000, 20000]


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(22500)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    X = torch.tensor(math_x, dtype=torch.float32)
    Y_raw = torch.tensor(math_y, dtype=torch.float32).view(-1, 1)
    y_mean = float(Y_raw.mean())
    y_std = float(Y_raw.std() + 1e-6)
    Y = (Y_raw - y_mean) / y_std

    Xval, Yval = X[20000:20500], Y[20000:20500]
    Xte, Yte = X[20500:21500], Y[20500:21500]
    _, incoming, edges, eu, ev, ew = build_graph()

    results = []
    for size in SIZES:
        torch.manual_seed(SEED)
        model = Brain(2000, eu, ev, ew)
        t1 = time.perf_counter()
        model = train_inner(model, X[:size], Y[:size], epochs=EPOCHS)
        val = evaluate(model, Xval, Yval) * y_std
        test = evaluate(model, Xte, Yte) * y_std
        sec = time.perf_counter() - t1
        results.append({"train_size": size, "val_mae": round(val, 5), "test_mae": round(test, 5), "seconds": round(sec, 3)})
        print("SCALE", results[-1], flush=True)

    payload = {
        "mode": "data-only scaling, fixed graph, no growth",
        "nodes": 2000, "K": K, "epochs": EPOCHS, "batch": BATCH,
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/data_scaling_only_result.json", "report": "research_mvp/DATA_SCALING_ONLY_REPORT.md"},
    }
    (ROOT / "data_scaling_only_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "data_scaling_only_result.json")


if __name__ == "__main__":
    main()
