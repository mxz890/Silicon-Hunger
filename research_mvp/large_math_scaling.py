#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does more readout data fix the weak tasks?  Linear and arithmetic scaling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from large_math_2000 import N, build_graph, chemical_field
from large_math_multi import forward_multi, task_arithmetic, task_linear

ROOT = Path(__file__).resolve().parent
OUTPUT = 1999
rng = np.random.default_rng(2028)


def main() -> None:
    pos, incoming, edges, eu, ev, ew = build_graph()
    C, gain = chemical_field(eu, ev, ew)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 3)).astype(np.float32)

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_multi(x, incoming, bias, input_gain, gain)[2:]
        return X

    n_total = 3500
    n_train_max = 3000
    n_test = 500
    tasks = {"linear": task_linear, "arithmetic": task_arithmetic}
    results = {}
    t0 = time.perf_counter()
    for name, generator in tasks.items():
        x, y = generator(n_total)
        X = features(x)
        X_train, X_test = X[:n_train_max], X[n_train_max:]
        y_train, y_test = y[:n_train_max], y[n_train_max:]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-6)
        rows = []
        for n in (500, 1000, 2000, 3000):
            coef, *_ = np.linalg.lstsq(
                X_train[:n], (y_train[:n] - y_mean) / y_std, rcond=1e-3
            )
            pred = X_test @ coef * y_std + y_mean
            mae = float(np.mean(np.abs(pred - y_test)))
            rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            rel = mae / (float(np.mean(np.abs(y_test))) + 1e-9)
            rows.append({"train": n, "mae": round(mae, 5), "rmse": round(rmse, 5), "relative_mae": round(rel, 5)})
            print(name, "train", n, "mae", round(mae, 4), "rel", round(rel, 4))
        results[name] = rows
    payload = {
        "graph": {"nodes": N, "edges": len(edges)},
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/large_math_scaling_result.json"},
    }
    (ROOT / "large_math_scaling_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", ROOT / "large_math_scaling_result.json")


if __name__ == "__main__":
    main()
