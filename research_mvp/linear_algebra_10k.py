#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard linear-algebra probes on the fresh 10k reservoir."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scale_10k_fresh import build_fresh, chemical_gain, forward_sample, N, DIM

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(2100)


def fit_multi(X, Y):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    ym = Y.mean(axis=0)
    ys = Y.std(axis=0) + 1e-6
    coef, *_ = np.linalg.lstsq(Xs, (Y - ym) / ys, rcond=1e-3)
    return {"mean": mean, "std": std, "ym": ym, "ys": ys, "coef": coef}


def predict_multi(model, X):
    return ((X - model["mean"]) / model["std"]) @ model["coef"] * model["ys"] + model["ym"]


def det2(n):
    A = rng.integers(-5, 6, (n, 4)).astype(np.float32)
    x = A / 5.0
    y = A[:, 0] * A[:, 3] - A[:, 1] * A[:, 2]
    return x, y[:, None]


def det3(n):
    A = rng.integers(-3, 4, (n, 9)).astype(np.float32)
    x = A / 3.0
    a, b, c, d, e, f, g, h, i = A.T
    y = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return x, y[:, None]


def matvec2(n):
    A = rng.integers(-5, 6, (n, 4)).astype(np.float32)
    v = rng.integers(-5, 6, (n, 2)).astype(np.float32)
    x = np.concatenate([A / 5.0, v / 5.0], axis=1)
    y = np.stack([A[:, 0] * v[:, 0] + A[:, 1] * v[:, 1], A[:, 2] * v[:, 0] + A[:, 3] * v[:, 1]], axis=1)
    return x, y / 10.0


def solve2(n):
    rows = []
    ys = []
    while len(rows) < n:
        A = rng.integers(-5, 6, 4).astype(np.float32)
        det = A[0] * A[3] - A[1] * A[2]
        if det == 0:
            continue
        e, f = rng.integers(-5, 6, 2).astype(np.float32)
        x = np.concatenate([A / 5.0, np.array([e, f], dtype=np.float32) / 5.0])
        rows.append(x)
        ys.append([(e * A[3] - A[1] * f) / det, (A[0] * f - e * A[2]) / det])
    return np.array(rows, dtype=np.float32), np.array(ys, dtype=np.float32) / 5.0


def eigen2(n):
    A = rng.integers(-5, 6, (n, 3)).astype(np.float32)  # a,b,c -> [[a,b],[b,c]]
    a, b, c = A.T
    disc = np.sqrt((a - c) ** 2 + 4 * b * b)
    y = np.stack([(a + c + disc) / 2.0, (a + c - disc) / 2.0], axis=1)
    return A / 5.0, y / 5.0


TASKS = [
    ("det2", "2x2 determinant ad-bc", 4, det2),
    ("det3", "3x3 determinant", 9, det3),
    ("matvec2", "2x2 matrix-vector product", 6, matvec2),
    ("solve2", "2x2 linear system", 6, solve2),
    ("eigen2", "2x2 symmetric eigenvalues", 3, eigen2),
]


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            xp = np.pad(x, (0, DIM - len(x)))
            X[i] = forward_sample(xp, incoming, bias, input_gain, gain)[2:]
        return X

    print("GRAPH", N, len(edges))
    results = []
    for name, label, input_dim, generator in TASKS:
        x, y = generator(400)
        X_train, X_test = features(x[:300]), features(x[300:])
        model = fit_multi(X_train, y[:300])
        pred = predict_multi(model, X_test)
        mae = float(np.mean(np.abs(pred - y[300:])))
        rmse = float(np.sqrt(np.mean((pred - y[300:]) ** 2)))
        denom = float(np.mean(np.abs(y[300:]))) + 1e-9
        result = {
            "task": name, "label": label,
            "input_dim": input_dim, "output_dim": int(y.shape[1]),
            "mae": round(mae, 5), "rmse": round(rmse, 5),
            "relative_mae": round(mae / denom, 5),
            "sample": {
                "target": [round(float(v), 4) for v in y[300]],
                "prediction": [round(float(v), 4) for v in pred[0]],
            },
        }
        results.append(result)
        print("RESULT", result)

    payload = {
        "graph": {"nodes": N, "edges": len(edges)},
        "tasks": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/linear_algebra_10k_result.json", "report": "research_mvp/LINEAR_ALGEBRA_10K_REPORT.md"},
    }
    (ROOT / "linear_algebra_10k_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "linear_algebra_10k_result.json")


if __name__ == "__main__":
    main()
