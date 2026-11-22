#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair 10k comparison: seed-grown vs fresh, with matched output degree and edge budget.

Both graphs:
  - 10000 nodes
  - 56212 edges
  - output indegree ~8003
  - same math/Chinese datasets
  - same readout hyperparameters
  - same model RNG seed for bias/input_gain
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from large_math_2000 import build_graph as build_base
from large_math_multi import task_sum_sq
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes, predict_classes

ROOT = Path(__file__).resolve().parent
N = 10000
OUTPUT = 9999
DIM = 64
EDGE_TARGET = 56212
OUT_DEGREE_TARGET = 8003
graph_rng = np.random.default_rng(2070)
model_rng = np.random.default_rng(777)


def build_seedgrown():
    pos0, incoming, edges, _, _, _ = build_base()
    pos = np.empty((N, 3), dtype=np.float32)
    pos[:len(pos0)] = pos0
    for j in range(2000, N - 1):
        pos[j] = np.clip(pos[j - 1] + graph_rng.normal(0, 2.0, 3), 0, 100)
        incoming.append([])
        scored = []
        for u in range(max(0, j - 8), j):
            d = float(np.linalg.norm(pos[u] - pos[j]))
            if d <= 30.0:
                scored.append((d, u))
        scored.sort()
        for _, u in scored[:4]:
            w = float(graph_rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))
    incoming.append([])  # output node
    pos[OUTPUT] = np.clip(pos[OUTPUT - 1] + graph_rng.normal(0, 2, 3), 0, 100)
    # All new nodes -> output.
    for j in range(2000, OUTPUT):
        w = float(graph_rng.uniform(0.2, 0.7))
        incoming[OUTPUT].append((j, w))
        edges.append((j, OUTPUT, w))
    for j in (1999, 0, 1):
        w = float(graph_rng.uniform(0.2, 0.7))
        incoming[OUTPUT].append((j, w))
        edges.append((j, OUTPUT, w))
    return pos, incoming, edges


def build_fresh():
    pos = np.empty((N, 3), dtype=np.float32)
    pos[0] = [50, 50, 50]
    for i in range(1, N):
        pos[i] = np.clip(pos[i - 1] + graph_rng.normal(0, 2.0, 3), 0, 100)
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
            w = float(graph_rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))
    # Match output indegree: nodes 2000..9998 plus 1999/1998/1997.
    for j in list(range(2000, OUTPUT)) + [1999, 1998, 1997]:
        w = float(graph_rng.uniform(0.2, 0.7))
        incoming[OUTPUT].append((j, w))
        edges.append((j, OUTPUT, w))
    return pos, incoming, edges


def enforce_budget(pos, incoming, edges):
    existing = {(u, v) for u, v, _ in edges}
    while len(edges) < EDGE_TARGET:
        j = int(graph_rng.integers(2, OUTPUT))
        candidates = [u for u in range(max(0, j - 12), j) if (u, j) not in existing]
        if not candidates:
            continue
        u = int(graph_rng.choice(candidates))
        if float(np.linalg.norm(pos[u] - pos[j])) > 30.0:
            continue
        w = float(graph_rng.uniform(0.1, 0.6))
        incoming[j].append((u, w))
        edges.append((u, j, w))
        existing.add((u, j))
    return edges


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
            inc = np.bincount(edge_v, weights=edge_w * C[k, edge_u], minlength=N)
            deg = np.bincount(edge_v, weights=edge_w, minlength=N)
            flux = inc - deg * C[k]
            C[k] = C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
    return C, np.clip(gain, 0.4, 1.8)


def forward_sample(x, incoming, bias, input_gain, gain):
    act = np.zeros(N, dtype=np.float32)
    act[0], act[1] = x[0], x[1]
    term = input_gain @ x
    for j in range(2, N):
        pre = bias[j] + term[j]
        for u, w in incoming[j]:
            pre += w * act[u]
        act[j] = np.tanh(gain[j] * pre)
    return act


def evaluate_graph(name, pos, incoming, edges, math_x, math_y, cn_train, cn_test, vocab):
    global model_rng
    model_rng = np.random.default_rng(777)
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    bias = model_rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = model_rng.normal(0, 0.7, (N, DIM)).astype(np.float32)

    def feats(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    Xm, Xmv = feats(math_x[:400]), feats(math_x[400:600])
    mm = fit_ridge(Xm, math_y[:400])
    mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[400:600])))
    rmse = float(np.sqrt(np.mean((predict(mm, Xmv) - math_y[400:600]) ** 2)))
    Xc = feats(np.array([r[0] for r in cn_train[:600]], dtype=np.float32))
    Xcv = feats(np.array([r[0] for r in cn_test[:300]], dtype=np.float32))
    cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train[:600]], dtype=np.int64), len(vocab))
    cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test[:300]], dtype=np.int64)))
    result = {
        "name": name,
        "nodes": N,
        "edges": len(edges),
        "output_indegree": len(incoming[OUTPUT]),
        "max_edge_distance": round(max(float(np.linalg.norm(pos[u] - pos[v])) for u, v, _ in edges), 3),
        "math_mae": round(mae, 5),
        "math_rmse": round(rmse, 5),
        "chinese_accuracy": round(cacc, 4),
    }
    print("RESULT", result)
    return result


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(900)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    cn_train, cn_test, vocab = make_chinese()
    cn_train = cn_train[:800]
    cn_test = cn_test[:300]

    results = []
    for name, builder in (("seed-grown", build_seedgrown), ("fresh", build_fresh)):
        pos, incoming, edges = builder()
        if len(edges) > EDGE_TARGET:
            print("WARNING edges exceed target", name, len(edges))
        if len(edges) < EDGE_TARGET:
            edges = enforce_budget(pos, incoming, edges)
        print(name, "edges", len(edges), "output_indegree", len(incoming[OUTPUT]))
        results.append(evaluate_graph(name, pos, incoming, edges, math_x, math_y, cn_train, cn_test, vocab))

    payload = {
        "task": "fair 10k comparison",
        "edge_target": EDGE_TARGET,
        "output_indegree_target": OUT_DEGREE_TARGET,
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/fair_10k_result.json", "report": "research_mvp/FAIR_10K_REPORT.md"},
    }
    (ROOT / "fair_10k_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "fair_10k_result.json")


if __name__ == "__main__":
    main()
