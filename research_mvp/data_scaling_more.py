#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-only scaling to 50k/100k, fixed 2000 graph, K=8."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from grow_train_loop_v3 import Brain, train_inner, evaluate, K, EPOCHS, BATCH
from large_math_2000 import build_graph
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
DIM = 64
SEED = 2181
SIZES = [20000, 50000, 100000]


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(120000)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    X = torch.tensor(math_x, dtype=torch.float32)
    Y_raw = torch.tensor(math_y, dtype=torch.float32).view(-1, 1)
    y_mean = float(Y_raw.mean())
    y_std = float(Y_raw.std() + 1e-6)
    Y = (Y_raw - y_mean) / y_std

    Xval, Yval = X[100000:100500], Y[100000:100500]
    Xte, Yte = X[119000:120000], Y[119000:120000]
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
        "mode": "data-only scaling to 100k, fixed 2000 graph, no growth",
        "nodes": 2000, "K": K, "epochs": EPOCHS, "batch": BATCH,
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/data_scaling_more_result.json", "report": "research_mvp/DATA_SCALING_MORE_REPORT.md"},
    }
    (ROOT / "data_scaling_more_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "data_scaling_more_result.json")


if __name__ == "__main__":
    main()
