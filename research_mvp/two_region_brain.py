#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two 2000-node regions vs one 4000-node graph.

Both regions are independent fixed reservoirs.  Only readouts are trained.
This tests whether modular concatenation helps before adding cross-region edges.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from large_math_multi import task_sum_sq
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes, predict_classes

ROOT = Path(__file__).resolve().parent
DIM = 64


def build_graph(n: int, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.empty((n, 3), dtype=np.float32)
    pos[0] = [50, 50, 50]
    for i in range(1, n):
        pos[i] = np.clip(pos[i - 1] + rng.normal(0, 2.0, 3), 0, 100)
    incoming = [[] for _ in range(n)]
    edges = []
    output = n - 1
    for j in range(2, output):
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
    for j in range(max(2, output - 120), output):
        w = float(rng.uniform(0.2, 0.7))
        incoming[output].append((j, w))
        edges.append((j, output, w))
    return pos, incoming, edges


def prepare_graph(n, seed):
    pos, incoming, edges = build_graph(n, seed)
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C = np.full((6, n), 0.05, dtype=np.float32)
    sources = np.zeros((6, n), dtype=np.float32)
    sources[0, 0] = 0.22; sources[1, 1] = 0.12
    sources[2, 0] = 0.18; sources[3, 1] = 0.14
    sources[4, n - 1] = 0.16; sources[5, n - 1] = 0.16
    for _ in range(20):
        for k in range(6):
            inc = np.bincount(ev, weights=ew * C[k, eu], minlength=n)
            deg = np.bincount(ev, weights=ew, minlength=n)
            flux = inc - deg * C[k]
            C[k] = np.clip(C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k]), 0.0, 3.0)
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = np.clip(1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh), 0.4, 1.8)
    return {"n": n, "incoming": incoming, "eu": eu, "ev": ev, "ew": ew, "gain": gain}


def features(xs, g, bias, input_gain):
    n = g["n"]
    incoming = g["incoming"]
    X = np.empty((len(xs), n - 2), dtype=np.float32)
    for i, x in enumerate(xs):
        h = np.zeros(n, dtype=np.float32)
        h[0], h[1] = x[0], x[1]
        term = input_gain @ x
        for j in range(2, n):
            pre = bias[j] + term[j]
            for u, w in incoming[j]:
                pre += w * h[u]
            h[j] = np.tanh(g["gain"][j] * pre)
        X[i] = h[2:]
    return X


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(1000)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    cn_train, cn_test, vocab = make_chinese()
    cn_train, cn_test = cn_train[:700], cn_test[:300]
    cn_x = np.array([r[0] for r in cn_train] + [r[0] for r in cn_test], dtype=np.float32)
    cn_y = np.array([r[1] for r in cn_train] + [r[1] for r in cn_test], dtype=np.int64)

    rng = np.random.default_rng(2160)
    single = prepare_graph(4000, 3001)
    bias_s = rng.normal(0, 0.15, 4000).astype(np.float32)
    ig_s = rng.normal(0, 0.7, (4000, DIM)).astype(np.float32)
    t1 = time.perf_counter()
    Xm_s = features(math_x, single, bias_s, ig_s)
    Xc_s = features(cn_x, single, bias_s, ig_s)
    single_feat_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    cn_graph = prepare_graph(2000, 3002)
    math_graph = prepare_graph(2000, 3003)
    bias_cn = rng.normal(0, 0.15, 2000).astype(np.float32)
    ig_cn = rng.normal(0, 0.7, (2000, DIM)).astype(np.float32)
    bias_math = rng.normal(0, 0.15, 2000).astype(np.float32)
    ig_math = rng.normal(0, 0.7, (2000, DIM)).astype(np.float32)
    Xm_cn = features(math_x, cn_graph, bias_cn, ig_cn)
    Xm_math = features(math_x, math_graph, bias_math, ig_math)
    Xc_cn = features(cn_x, cn_graph, bias_cn, ig_cn)
    Xc_math = features(cn_x, math_graph, bias_math, ig_math)
    dual_feat_sec = time.perf_counter() - t2

    def math_eval(X, name):
        m = fit_ridge(X[:700], math_y[:700])
        pred = predict(m, X[700:])
        mae = float(np.mean(np.abs(pred - math_y[700:])))
        print("MATH", name, round(mae, 5))
        return mae

    def cn_eval(X, name):
        m, _ = fit_classes(X[:700], cn_y[:700], len(vocab))
        pred = predict_classes(m, X[700:])
        acc = float(np.mean(pred == cn_y[700:]))
        print("CN", name, round(acc, 4))
        return acc

    single_math = math_eval(Xm_s, "single4000")
    single_cn = cn_eval(Xc_s, "single4000")
    dual_math = math_eval(np.hstack([Xm_cn, Xm_math]), "two-region")
    dual_cn = cn_eval(np.hstack([Xc_cn, Xc_math]), "two-region")

    within = {
        "cn_region_math": round(math_eval(Xm_cn, "cn-region-math"), 5),
        "math_region_math": round(math_eval(Xm_math, "math-region-math"), 5),
        "cn_region_chinese": round(cn_eval(Xc_cn, "cn-region-chinese"), 4),
        "math_region_chinese": round(cn_eval(Xc_math, "math-region-chinese"), 4),
    }

    payload = {
        "single_4000": {"nodes": 4000, "edges": len(single["eu"]), "math_mae": round(single_math, 5), "chinese_acc": round(single_cn, 4)},
        "two_region": {"region_nodes": 2000, "math_mae": round(dual_math, 5), "chinese_acc": round(dual_cn, 4)},
        "within_region": within,
        "efficiency": {
            "single_feature_seconds": round(single_feat_sec, 3),
            "dual_feature_seconds": round(dual_feat_sec, 3),
            "total_seconds": round(time.perf_counter() - t0, 3),
        },
        "output_files": {"result_json": "research_mvp/two_region_result.json", "report": "research_mvp/TWO_REGION_REPORT.md"},
    }
    (ROOT / "two_region_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", payload)
    print("SAVED", ROOT / "two_region_result.json")


if __name__ == "__main__":
    main()
