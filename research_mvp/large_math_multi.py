#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node spatial reservoir, six chemical fields, multiple junior-high tasks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from large_math_2000 import (
    N,
    CHEMICALS,
    build_graph,
    chemical_field,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = 1999
rng = np.random.default_rng(2027)


def forward_multi(x, incoming, bias, input_gain, gain):
    act = np.zeros(N, dtype=np.float32)
    act[0], act[1] = x[0], x[1]
    for j in range(2, N):
        pre = bias[j]
        for k in range(input_gain.shape[1]):
            pre += input_gain[j, k] * x[k]
        for u, w in incoming[j]:
            pre += w * act[u]
        act[j] = np.tanh(gain[j] * pre)
    return act


def task_sum_sq(n):
    S = rng.integers(3, 21, n)
    P = rng.integers(1, np.maximum(2, S * S // 4) + 1, n)
    x = np.column_stack([S / 20.0, P / 100.0, np.zeros(n)])
    y = S * S - 2 * P
    return x.astype(np.float32), y.astype(np.float32)


def task_sum_cube(n):
    S = rng.integers(3, 16, n)
    P = rng.integers(1, np.maximum(2, S * S // 4) + 1, n)
    x = np.column_stack([S / 20.0, P / 100.0, np.zeros(n)])
    y = S ** 3 - 3 * P * S
    return x.astype(np.float32), y.astype(np.float32)


def task_reciprocal(n):
    S = rng.integers(2, 11, n)
    x = np.column_stack([S / 10.0, np.zeros(n), np.zeros(n)])
    y = S * S - 2
    return x.astype(np.float32), y.astype(np.float32)


def task_hypotenuse(n):
    a = rng.integers(1, 16, n)
    b = rng.integers(1, 16, n)
    x = np.column_stack([a / 15.0, b / 15.0, np.zeros(n)])
    y = np.sqrt(a * a + b * b)
    return x.astype(np.float32), y.astype(np.float32)


def task_linear(n):
    a = rng.integers(1, 10, n)
    b = rng.integers(1, 21, n)
    c = rng.integers(1, 31, n)
    x = np.column_stack([a / 10.0, b / 30.0, c / 30.0])
    y = (c - b) / a
    return x.astype(np.float32), y.astype(np.float32)


def task_arithmetic(n):
    a1 = rng.integers(1, 11, n)
    d = rng.integers(1, 6, n)
    m = rng.integers(3, 16, n)
    x = np.column_stack([a1 / 10.0, d / 5.0, m / 15.0])
    y = a1 + (m - 1) * d
    return x.astype(np.float32), y.astype(np.float32)


TASKS = [
    ("sum_sq", "a+b=S, ab=P -> S^2-2P", task_sum_sq),
    ("sum_cube", "a+b=S, ab=P -> S^3-3PS", task_sum_cube),
    ("reciprocal", "x+1/x=S -> S^2-2", task_reciprocal),
    ("hypotenuse", "right triangle a,b -> sqrt(a^2+b^2)", task_hypotenuse),
    ("linear", "ax+b=c -> x=(c-b)/a", task_linear),
    ("arithmetic", "a1,d,n -> a1+(n-1)d", task_arithmetic),
]


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges, eu, ev, ew = build_graph()
    C, gain = chemical_field(eu, ev, ew)
    build_sec = time.perf_counter() - t0
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 3)).astype(np.float32)

    max_edge_distance = max(
        float(np.linalg.norm(pos[u] - pos[v])) for u, v, _ in edges
    ) if edges else 0.0
    print("GRAPH", "nodes", N, "edges", len(edges), "max_edge_distance", round(max_edge_distance, 3))
    print("CHEM_MEAN", {name: round(float(C[k].mean()), 5) for k, name in enumerate(CHEMICALS)})

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_multi(x, incoming, bias, input_gain, gain)[2:]
        return X

    split = 500
    train_features = {}
    test_features = {}
    targets = {}
    t1 = time.perf_counter()
    for name, label, generator in TASKS:
        x, y = generator(split + 150)
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = features(x_train)
        X_test = features(x_test)
        train_features[name] = (X_train, y_train)
        test_features[name] = (X_test, y_test)
        targets[name] = label
        print("TASK", name, "train", split, "test", 150, "feature_sec", round(time.perf_counter() - t1, 2))
    feature_sec = time.perf_counter() - t1

    all_train = np.vstack([v[0] for v in train_features.values()])
    mean = all_train.mean(axis=0)
    std = all_train.std(axis=0) + 1e-6

    results = {}
    t2 = time.perf_counter()
    for name, label, _ in TASKS:
        X_train, y_train = train_features[name]
        X_test, y_test = test_features[name]
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-6)
        coef, *_ = np.linalg.lstsq(X_train, (y_train - y_mean) / y_std, rcond=1e-3)
        pred = X_test @ coef * y_std + y_mean
        mae = float(np.mean(np.abs(pred - y_test)))
        rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
        rel = mae / (float(np.mean(np.abs(y_test))) + 1e-9)
        results[name] = {
            "label": label,
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
            "relative_mae": round(rel, 5),
        }
        print("RESULT", name, "mae", round(mae, 4), "rmse", round(rmse, 4), "rel", round(rel, 4))
    fit_sec = time.perf_counter() - t2

    # Held-out classic problem for sum_sq.
    x_problem = np.array([[7 / 20.0, 12 / 100.0, 0.0]], dtype=np.float32)
    feat = (features(x_problem) - mean) / std
    X_train, y_train = train_features["sum_sq"]
    X_train = (X_train - mean) / std
    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-6)
    coef, *_ = np.linalg.lstsq(X_train, (y_train - y_mean) / y_std, rcond=1e-3)
    problem_pred = float((feat @ coef * y_std + y_mean)[0])
    print("PROBLEM a+b=7, ab=12 target 25 ->", round(problem_pred, 4))

    names = [t[0] for t in TASKS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, [results[n]["mae"] for n in names], color="#2f7d5b")
    ax.set_ylabel("test MAE")
    ax.set_title("2000-node reservoir: multi-task test error")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(ROOT / "large_math_multi.png", dpi=180)

    payload = {
        "graph": {"nodes": N, "edges": len(edges), "max_edge_distance": round(max_edge_distance, 4)},
        "chemicals": {
            "names": CHEMICALS,
            "means": {name: round(float(C[k].mean()), 6) for k, name in enumerate(CHEMICALS)},
            "gain_min": round(float(gain.min()), 5),
            "gain_max": round(float(gain.max()), 5),
        },
        "tasks": results,
        "competition_problem": {
            "statement": "a+b=7, ab=12, find a^2+b^2",
            "target": 25,
            "prediction": round(problem_pred, 5),
        },
        "timing": {
            "build_seconds": round(build_sec, 4),
            "feature_seconds": round(feature_sec, 4),
            "fit_seconds": round(fit_sec, 4),
        },
        "output_files": {
            "result_json": "research_mvp/large_math_multi_result.json",
            "plot": "research_mvp/large_math_multi.png",
            "report": "research_mvp/LARGE_MATH_MULTI_REPORT.md",
        },
    }
    (ROOT / "large_math_multi_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", ROOT / "large_math_multi_result.json")


if __name__ == "__main__":
    main()
