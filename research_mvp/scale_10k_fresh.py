#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh 10,000-node graph, not grown from the 2000 seed.

Test math + real Chinese next-character prediction on a new independent graph.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from large_math_multi import task_sum_sq
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes, predict_classes

ROOT = Path(__file__).resolve().parent
N = 10000
DIM = 64
OUTPUT = N - 1
rng = np.random.default_rng(2060)


def build_fresh():
    pos = np.empty((N, 3), dtype=np.float32)
    pos[0] = [50, 50, 50]
    for i in range(1, N):
        pos[i] = np.clip(pos[i - 1] + rng.normal(0, 2.0, 3), 0, 100)
    incoming = [[] for _ in range(N)]
    edges = []
    for j in range(2, OUTPUT):
        scored = []
        for u in range(max(0, j - 8), j):
            d = float(np.linalg.norm(pos[u] - pos[j]))
            if d <= 30.0:
                scored.append((d, u))
        scored.sort()
        chosen = scored[:4]
        if not chosen:
            chosen = [(float(np.linalg.norm(pos[j - 1] - pos[j])), j - 1)]
        for _, u in chosen:
            w = float(rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))
    # Independent output: receive from the last 120 nodes.
    for j in range(max(2, OUTPUT - 120), OUTPUT):
        w = float(rng.uniform(0.2, 0.7))
        incoming[OUTPUT].append((j, w))
        edges.append((j, OUTPUT, w))
    return pos, incoming, edges


def chemical_gain(edge_u, edge_v, edge_w):
    C = np.full((6, N), 0.05, dtype=np.float32)
    sources = np.zeros((6, N), dtype=np.float32)
    sources[0, 0] = 0.22
    sources[1, 1] = 0.12
    sources[2, 0] = 0.18
    sources[3, 1] = 0.14
    sources[4, OUTPUT] = 0.16
    sources[5, OUTPUT] = 0.16
    for _ in range(20):
        for k in range(6):
            incoming = np.bincount(edge_v, weights=edge_w * C[k, edge_u], minlength=N)
            degree = np.bincount(edge_v, weights=edge_w, minlength=N)
            flux = incoming - degree * C[k]
            C[k] = C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
    return C, np.clip(gain, 0.4, 1.8)


def forward_sample(x, incoming, bias, input_gain, gain):
    act = np.zeros(N, dtype=np.float32)
    act[0], act[1] = x[0], x[1]
    input_term = input_gain @ x
    for j in range(2, N):
        pre = bias[j] + input_term[j]
        for u, w in incoming[j]:
            pre += w * act[u]
        act[j] = np.tanh(gain[j] * pre)
    return act


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)

    math_x, math_y = task_sum_sq(900)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    train_cn, test_cn, vocab = make_chinese()
    cn_train = train_cn[:800]
    cn_test = test_cn[:300]

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    print("GRAPH", "nodes", N, "edges", len(edges), "max_edge_distance", round(max(float(np.linalg.norm(pos[u] - pos[v])) for u, v, _ in edges), 3))
    print("OUTPUT", OUTPUT, "indegree", len(incoming[OUTPUT]))

    Xm, Xmv = features(math_x[:400]), features(math_x[400:600])
    mm = fit_ridge(Xm, math_y[:400])
    mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[400:600])))
    rmse = float(np.sqrt(np.mean((predict(mm, Xmv) - math_y[400:600]) ** 2)))

    Xc = features(np.array([r[0] for r in cn_train[:600]], dtype=np.float32))
    Xcv = features(np.array([r[0] for r in cn_test[:300]], dtype=np.float32))
    cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train[:600]], dtype=np.int64), len(vocab))
    cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test[:300]], dtype=np.int64)))

    x_problem = np.array([7 / 20.0, 12 / 100.0] + [0.0] * (DIM - 2), dtype=np.float32)
    act = forward_sample(x_problem, incoming, bias, input_gain, gain)
    problem_pred = float(predict(mm, act[2:].reshape(1, -1))[0])

    payload = {
        "mode": "fresh 10000-node graph (not grown from 2000 seed)",
        "nodes": N,
        "edges": len(edges),
        "output_indegree": len(incoming[OUTPUT]),
        "math": {"test_mae": round(mae, 5), "test_rmse": round(rmse, 5), "prediction": round(problem_pred, 5)},
        "chinese": {"task": "real next-character prediction (Lu Xun corpus)", "test_accuracy": round(cacc, 4), "vocab_size": len(vocab)},
        "chemicals": {"gain_min": round(float(gain.min()), 5), "gain_max": round(float(gain.max()), 5)},
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/scale_10k_fresh_result.json", "report": "research_mvp/SCALE_10K_FRESH_REPORT.md"},
    }
    (ROOT / "scale_10k_fresh_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", payload)
    print("SAVED", ROOT / "scale_10k_fresh_result.json")


if __name__ == "__main__":
    main()
