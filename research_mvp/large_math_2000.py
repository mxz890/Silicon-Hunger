#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node spatial reservoir + six chemical fields + a junior-high algebra task.

Task family:
  given a+b=S and ab=P, compute a^2+b^2 = S^2 - 2P.

Held-out competition-style problem:
  a+b=7, ab=12  ->  a^2+b^2 = 25
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N = 2000
INPUTS = (0, 1)
OUTPUT = 1999
HIDDEN = list(range(2, OUTPUT))
CHEMICALS = ["DA", "5HT", "NE", "ACh", "GABA", "Glu"]
DECAY = np.array([0.12, 0.06, 0.16, 0.20, 0.10, 0.16])
DIFFUSION = np.array([0.10, 0.06, 0.08, 0.12, 0.09, 0.10])
rng = np.random.default_rng(2026)


def build_graph():
    pos = np.zeros((N, 3), dtype=float)
    pos[0] = [50.0, 50.0, 50.0]
    for i in range(1, N):
        pos[i] = np.clip(pos[i - 1] + rng.normal(0, 2.0, 3), 0, 100)

    incoming = [[] for _ in range(N)]
    edges = []
    for j in range(2, N):
        candidates = list(range(max(0, j - 6), j))
        scored = []
        for u in candidates:
            d = float(np.linalg.norm(pos[u] - pos[j]))
            if d <= 30.0:
                scored.append((d, u))
        scored.sort()
        # 4 nearest predecessors, plus one random local skip edge.
        chosen = scored[:4]
        if scored and len(scored) > 4 and rng.random() < 0.5:
            chosen.append(scored[int(rng.integers(0, len(scored)))])
        for _, u in chosen:
            w = float(rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))
        if not chosen:
            u = j - 1
            w = float(rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))

    # Output node: connect the last local nodes.
    for u in range(max(2, N - 25), N - 1):
        if float(np.linalg.norm(pos[u] - pos[OUTPUT])) <= 30.0:
            w = float(rng.uniform(0.2, 0.8))
            incoming[OUTPUT].append((u, w))
            edges.append((u, OUTPUT, w))
    # Fallback: nearest predecessor to output.
    if not incoming[OUTPUT]:
        u = OUTPUT - 1
        w = 0.5
        incoming[OUTPUT].append((u, w))
        edges.append((u, OUTPUT, w))

    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    edge_w = np.array([e[2] for e in edges], dtype=np.float32)
    return pos, incoming, edges, edge_u, edge_v, edge_w


def chemical_field(edge_u, edge_v, edge_w):
    C = np.full((6, N), 0.05, dtype=np.float32)
    sources = np.zeros((6, N), dtype=np.float32)
    sources[0, 0] = 0.22       # DA at input
    sources[1, 1] = 0.12       # 5HT
    sources[2, 0] = 0.18       # NE at input
    sources[3, 1] = 0.14       # ACh
    sources[4, OUTPUT] = 0.16  # GABA at output
    sources[5, OUTPUT] = 0.16  # Glu at output
    for _ in range(20):
        for k in range(6):
            incoming = np.bincount(edge_v, weights=edge_w * C[k, edge_u], minlength=N)
            degree = np.bincount(edge_v, weights=edge_w, minlength=N)
            flux = incoming - degree * C[k]
            C[k] = C[k] + 0.08 * (DIFFUSION[k] * flux - DECAY[k] * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
    gain = np.clip(gain, 0.4, 1.8)
    return C, gain


def forward_sample(x, incoming, bias, input_gain, gain):
    act = np.zeros(N, dtype=np.float32)
    act[0], act[1] = x[0], x[1]
    for j in range(2, N):
        pre = bias[j] + input_gain[j, 0] * x[0] + input_gain[j, 1] * x[1]
        for u, w in incoming[j]:
            pre += w * act[u]
        act[j] = np.tanh(gain[j] * pre)
    return act


def make_data(n_train=900, n_test=300):
    rows = []
    for _ in range(n_train + n_test):
        S = int(rng.integers(3, 21))
        P = int(rng.integers(1, max(2, S * S // 4) + 1))
        target = S * S - 2 * P
        rows.append((S, P, target))
    return rows[:n_train], rows[n_train:]


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges, eu, ev, ew = build_graph()
    build_seconds = time.perf_counter() - t0
    max_edge_distance = max(
        float(np.linalg.norm(pos[u] - pos[v])) for u, v, _ in edges
    ) if edges else 0.0

    C, gain = chemical_field(eu, ev, ew)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 2)).astype(np.float32)

    train, test = make_data()
    print("GRAPH", "nodes", N, "edges", len(edges), "build_sec", round(build_seconds, 3))
    print("MAX_EDGE_DISTANCE", round(max_edge_distance, 3))
    print("CHEM_MEAN", {name: round(float(C[k].mean()), 4) for k, name in enumerate(CHEMICALS)})
    print("GAIN_RANGE", round(float(gain.min()), 3), round(float(gain.max()), 3))

    def features(rows):
        X = np.empty((len(rows), N - 2), dtype=np.float32)
        y = np.empty(len(rows), dtype=np.float32)
        for i, (S, P, target) in enumerate(rows):
            x = np.array([S / 20.0, P / 100.0], dtype=np.float32)
            act = forward_sample(x, incoming, bias, input_gain, gain)
            X[i] = act[2:]
            y[i] = target
        return X, y

    t1 = time.perf_counter()
    X_train, y_train = features(train)
    X_test, y_test = features(test)
    feature_seconds = time.perf_counter() - t1

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    y_train_n = (y_train - y_mean) / y_std

    t2 = time.perf_counter()
    coef, *_ = np.linalg.lstsq(X_train, y_train_n, rcond=1e-3)
    fit_seconds = time.perf_counter() - t2

    pred_test = X_test @ coef * y_std + y_mean
    mae = float(np.mean(np.abs(pred_test - y_test)))
    rmse = float(np.sqrt(np.mean((pred_test - y_test) ** 2)))

    # The held-out competition-style problem: a+b=7, ab=12 -> 25.
    x_problem = np.array([7 / 20.0, 12 / 100.0], dtype=np.float32)
    act_problem = forward_sample(x_problem, incoming, bias, input_gain, gain)
    feat_problem = (act_problem[2:] - mean) / std
    y_problem = float(feat_problem @ coef * y_std + y_mean)

    print("DATA", "train", len(train), "test", len(test), "features", X_train.shape[1])
    print("MAE", round(mae, 4), "RMSE", round(rmse, 4))
    print("PROBLEM a+b=7, ab=12 -> target 25, prediction", round(y_problem, 4))
    print("TIME build", round(build_seconds, 3), "features", round(feature_seconds, 3), "fit", round(fit_seconds, 3))

    result = {
        "task": "a+b=S, ab=P -> a^2+b^2",
        "competition_problem": {
            "statement": "a+b=7, ab=12, find a^2+b^2",
            "target": 25,
            "prediction": round(y_problem, 5),
        },
        "graph": {
            "nodes": N,
            "edges": len(edges),
            "max_edge_distance": round(max_edge_distance, 4),
            "build_seconds": round(build_seconds, 4),
        },
        "chemicals": {
            "names": CHEMICALS,
            "means": {name: round(float(C[k].mean()), 6) for k, name in enumerate(CHEMICALS)},
            "gain_min": round(float(gain.min()), 5),
            "gain_max": round(float(gain.max()), 5),
        },
        "dataset": {"train": len(train), "test": len(test)},
        "metrics": {"mae": round(mae, 5), "rmse": round(rmse, 5)},
        "timing": {
            "feature_seconds": round(feature_seconds, 4),
            "fit_seconds": round(fit_seconds, 4),
        },
        "output_files": {
            "result_json": "research_mvp/large_math_result.json",
            "report": "research_mvp/LARGE_MATH_REPORT.md",
        },
    }
    (ROOT / "large_math_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", ROOT / "large_math_result.json")


if __name__ == "__main__":
    main()
