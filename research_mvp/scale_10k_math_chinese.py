#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grow a 2000-node seed to 10,000 nodes, then test math + Chinese.

No time / recurrence.  The graph is a static DAG; growth happens in stages.
Math: a+b=S, ab=P -> a^2+b^2.
Chinese: synthetic sentiment classification from Chinese characters.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from large_math_2000 import N as SEED_N, build_graph as build_seed, chemical_field as ignore_chem
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
TARGET_N = 10000
DIM = 64
STAGES = [2000, 4000, 6000, 8000, 10000]
OUTPUT = 1999
rng = np.random.default_rng(2040)


def grow_graph(pos, incoming, edges, target_n):
    old_n = len(pos)
    if target_n <= old_n:
        return pos, incoming, edges
    new_pos = np.empty((target_n, 3), dtype=np.float32)
    new_pos[:old_n] = pos
    for j in range(old_n, target_n):
        new_pos[j] = np.clip(new_pos[j - 1] + rng.normal(0, 2.0, 3), 0, 100)
        incoming.append([])
        scored = []
        for u in range(max(0, j - 8), j):
            d = float(np.linalg.norm(new_pos[u] - new_pos[j]))
            if d <= 30.0:
                scored.append((d, u))
        scored.sort()
        chosen = scored[:4]
        if not chosen:
            chosen = [(float(np.linalg.norm(new_pos[j - 1] - new_pos[j])), j - 1)]
        for _, u in chosen:
            w = float(rng.uniform(0.15, 0.75))
            incoming[j].append((u, w))
            edges.append((u, j, w))
    return new_pos, incoming, edges


def chemical_gain_for(edge_u, edge_v, edge_w, n):
    C = np.full((6, n), 0.05, dtype=np.float32)
    sources = np.zeros((6, n), dtype=np.float32)
    sources[0, 0] = 0.22
    sources[1, 1] = 0.12
    sources[2, 0] = 0.18
    sources[3, 1] = 0.14
    sources[4, OUTPUT] = 0.16
    sources[5, OUTPUT] = 0.16
    for _ in range(20):
        for k in range(6):
            incoming = np.bincount(edge_v, weights=edge_w * C[k, edge_u], minlength=n)
            degree = np.bincount(edge_v, weights=edge_w, minlength=n)
            flux = incoming - degree * C[k]
            C[k] = C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
    return C, np.clip(gain, 0.4, 1.8)


def forward_sample(x, incoming, bias, input_gain, gain):
    n = len(incoming)
    act = np.zeros(n, dtype=np.float32)
    act[0], act[1] = x[0], x[1]
    input_term = input_gain @ x
    for j in range(2, n):
        pre = bias[j] + input_term[j]
        for u, w in incoming[j]:
            pre += w * act[u]
        act[j] = np.tanh(gain[j] * pre)
    return act


def fit_ridge(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    coef, *_ = np.linalg.lstsq(Xs, (y - ym) / ys, rcond=1e-3)
    return {"mean": mean, "std": std, "ym": ym, "ys": ys, "coef": coef}


def predict(model, X):
    return ((X - model["mean"]) / model["std"]) @ model["coef"] * model["ys"] + model["ym"]


def encode_chinese(text):
    vec = np.zeros(DIM, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    for i in range(len(text) - 1):
        code = (ord(text[i]) * 131 + ord(text[i + 1])) % DIM
        vec[code] += 0.5
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def make_chinese(n=700):
    pos_adj = ["很好", "精彩", "满意", "开心", "优秀", "顺利", "漂亮", "温暖"]
    neg_adj = ["很差", "无聊", "失望", "难过", "糟糕", "失败", "丑陋", "冷漠"]
    pos_feel = ["很开心", "很满意", "很惊喜", "很舒服"]
    neg_feel = ["很难过", "很失望", "很生气", "很痛苦"]
    objs = ["今天的天气", "这部电影", "这个产品", "这家餐厅", "这次旅行", "这本书", "这次考试", "这次服务"]
    prefixes = ["", "说实话，", "总体来看，", "我个人觉得，"]
    rows = []
    for _ in range(n):
        label = int(rng.integers(0, 2))
        obj = str(rng.choice(objs))
        prefix = str(rng.choice(prefixes))
        if label:
            text = f"{prefix}{obj}{rng.choice(pos_adj)}，我{rng.choice(pos_feel)}。"
        else:
            text = f"{prefix}{obj}{rng.choice(neg_adj)}，我{rng.choice(neg_feel)}。"
        rows.append((encode_chinese(text), float(label), text))
    return rows


def main() -> None:
    t0 = time.perf_counter()
    base_pos, base_incoming, base_edges, _, _, _ = build_seed()
    pos, incoming, edges = base_pos, base_incoming, base_edges
    print("SEED", "nodes", len(pos), "edges", len(edges))

    # Fixed development batches for stage checks.
    math_rows = task_sum_sq(800)
    chinese_rows = make_chinese(800)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_rows[0]], dtype=np.float32)
    math_y = math_rows[1].astype(np.float32)
    chin_x = np.vstack([r[0] for r in chinese_rows])
    chin_y = np.array([r[1] for r in chinese_rows], dtype=np.float32)

    stage_results = []
    current = 0
    for target in STAGES:
        if target > len(pos):
            pos, incoming, edges = grow_graph(pos, incoming, edges, target)
        n = len(pos)
        eu = np.array([e[0] for e in edges], dtype=np.int32)
        ev = np.array([e[1] for e in edges], dtype=np.int32)
        ew = np.array([e[2] for e in edges], dtype=np.float32)
        C, gain = chemical_gain_for(eu, ev, ew, n)
        bias = rng.normal(0, 0.15, n).astype(np.float32)
        input_gain = rng.normal(0, 0.7, (n, DIM)).astype(np.float32)

        def features(xs):
            X = np.empty((len(xs), n - 2), dtype=np.float32)
            for i, x in enumerate(xs):
                X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
            return X

        # Stage check: 120 train / 60 test per task.
        Xm_tr, Xm_te = features(math_x[:120]), features(math_x[120:180])
        mm = fit_ridge(Xm_tr, math_y[:120])
        mae = float(np.mean(np.abs(predict(mm, Xm_te) - math_y[120:180])))
        Xc_tr, Xc_te = features(chin_x[:120]), features(chin_x[120:180])
        cm = fit_ridge(Xc_tr, chin_y[:120])
        cacc = float(np.mean((predict(cm, Xc_te) >= 0.5) == (chin_y[120:180] >= 0.5)))
        stage_results.append({"nodes": n, "edges": len(edges), "math_mae": round(mae, 5), "chinese_acc": round(cacc, 4)})
        print("STAGE", stage_results[-1])
        current = n

    # Final test: 400 train / 200 test per task.
    n = len(pos)
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain_for(eu, ev, ew, n)
    bias = rng.normal(0, 0.15, n).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (n, DIM)).astype(np.float32)

    def features(xs):
        X = np.empty((len(xs), n - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    print("FINAL_FEATURE_EXTRACT", n)
    Xm_tr, Xm_te = features(math_x[:400]), features(math_x[400:600])
    mm = fit_ridge(Xm_tr, math_y[:400])
    mae = float(np.mean(np.abs(predict(mm, Xm_te) - math_y[400:600])))
    rmse = float(np.sqrt(np.mean((predict(mm, Xm_te) - math_y[400:600]) ** 2)))

    Xc_tr, Xc_te = features(chin_x[:400]), features(chin_x[400:600])
    cm = fit_ridge(Xc_tr, chin_y[:400])
    cpred = predict(cm, Xc_te)
    cacc = float(np.mean((cpred >= 0.5) == (chin_y[400:600] >= 0.5)))

    # Exact held-out math problem.
    x_problem = np.array([7 / 20.0, 12 / 100.0] + [0.0] * (DIM - 2), dtype=np.float32)
    act = forward_sample(x_problem, incoming, bias, input_gain, gain)
    problem_pred = float(predict(mm, act[2:].reshape(1, -1))[0])

    payload = {
        "seed_nodes": 2000,
        "final_nodes": n,
        "final_edges": len(edges),
        "stages": stage_results,
        "math": {"test_mae": round(mae, 5), "test_rmse": round(rmse, 5), "problem": "a+b=7,ab=12 -> 25", "prediction": round(problem_pred, 5)},
        "chinese": {"test_accuracy": round(cacc, 4), "task": "synthetic sentiment classification"},
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/scale_10k_result.json", "report": "research_mvp/SCALE_10K_REPORT.md"},
    }
    (ROOT / "scale_10k_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", payload)
    print("SAVED", ROOT / "scale_10k_result.json")


if __name__ == "__main__":
    main()
