#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic chemical gating of edges + reward-modulated transmission weights.

No time dimension.  "Dynamic" means one update per training round, not per tick.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from large_math_2000 import build_graph
from large_math_multi import task_sum_sq
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes, predict_classes

ROOT = Path(__file__).resolve().parent
N = 2000
OUTPUT = N - 1
DIM = 64
ROUNDS = 10
rng = np.random.default_rng(2080)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def chemical_means(C):
    return {name: round(float(C[k].mean()), 5) for k, name in enumerate(["DA", "5HT", "NE", "ACh", "GABA", "Glu"])}


def diffuse(C, sources, steps=3):
    # C: [6, N]; directed edge diffusion.
    u, v, w = edges_u, edges_v, edges_w
    for _ in range(steps):
        for k in range(6):
            inc = np.bincount(v, weights=w * C[k, u], minlength=N)
            deg = np.bincount(v, weights=w, minlength=N)
            flux = inc - deg * C[k]
            C[k] = C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    return C


def gates(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    return sigmoid(2.5 * (exc - inh - 0.35))


def effective_edges(w, gate, u, v):
    g = gate[u] * gate[v]
    we = w * (0.2 + 1.4 * g)
    active = we > 0.08
    return we, active


def build_incoming(u, v, we):
    inc = [[] for _ in range(N)]
    for a, b, weight in zip(u, v, we):
        if weight > 0.08:
            inc[b].append((int(a), float(weight)))
    return inc


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


def chemical_gain(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    return np.clip(1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh), 0.4, 1.8)


def main() -> None:
    global edges_u, edges_v, edges_w
    t0 = time.perf_counter()
    pos, incoming0, edges, eu, ev, ew = build_graph()
    edges_u, edges_v, edges_w = eu.astype(np.int32), ev.astype(np.int32), ew.astype(np.float32)
    w0 = edges_w.copy()
    w = w0.copy()

    C = np.full((6, N), 0.05, dtype=np.float32)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)

    math_x, math_y = task_sum_sq(700)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    cn_train, cn_test, vocab = make_chinese()
    cn_train, cn_test = cn_train[:400], cn_test[:200]

    history = []

    def features(xs, incoming):
        out = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            out[i] = forward_sample(x, incoming, bias, input_gain, chemical_gain(C))[2:]
        return out

    for round_no in range(0, ROUNDS + 1):
        gate = gates(C)
        we, active = effective_edges(w, gate, edges_u, edges_v)
        incoming = build_incoming(edges_u, edges_v, we)
        Xm, Xmv = features(math_x[:400], incoming), features(math_x[400:600], incoming)
        mm = fit_ridge(Xm, math_y[:400])
        mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[400:600])))
        Xc = features(np.array([r[0] for r in cn_train], dtype=np.float32), incoming)
        Xcv = features(np.array([r[0] for r in cn_test], dtype=np.float32), incoming)
        cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train], dtype=np.int64), len(vocab))
        cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test], dtype=np.int64)))
        history.append({
            "round": round_no, "math_mae": round(mae, 5), "chinese_acc": round(cacc, 4),
            "active_edges": int(active.sum()), "mean_weight": round(float(we.mean()), 5),
            "chem": chemical_means(C), "gate_mean": round(float(gate.mean()), 5),
        })
        print("ROUND", history[-1])

        if round_no == ROUNDS:
            break

        # Collect feedback from a small training subset.
        node_fb = np.zeros(N, dtype=np.float32)
        edge_delta = np.zeros(len(edges_u), dtype=np.float32)
        feedback_count = 0

        math_acts = []
        math_errs = []
        for i in range(120):
            x = math_x[i]
            act = forward_sample(x, incoming, bias, input_gain, chemical_gain(C))
            pred = float(predict(mm, act[2:].reshape(1, -1))[0])
            math_acts.append(act)
            math_errs.append(abs(pred - float(math_y[i])))
        median_err = float(np.median(math_errs))
        for act, err in zip(math_acts, math_errs):
            fb = 1.0 if err <= median_err else -1.0
            node_fb += fb * np.abs(act)
            edge_delta += fb * act[edges_u] * act[edges_v]
            feedback_count += 1

        for i in range(120):
            x, label = cn_train[i]
            act = forward_sample(np.asarray(x, dtype=np.float32), incoming, bias, input_gain, chemical_gain(C))
            pred = int(np.argmax(predict(cm, act[2:].reshape(1, -1))[0]))
            fb = 1.0 if pred == int(label) else -1.0
            node_fb += fb * np.abs(act)
            edge_delta += fb * act[edges_u] * act[edges_v]
            feedback_count += 1

        node_fb /= max(1, feedback_count)
        edge_delta /= max(1, feedback_count)

        # Chemical sources: positive feedback -> DA/Glu, negative -> NE/GABA.
        sources = np.zeros((6, N), dtype=np.float32)
        sources[0] = np.maximum(node_fb, 0.0) * 0.35
        sources[5] = np.maximum(node_fb, 0.0) * 0.25
        sources[2] = np.maximum(-node_fb, 0.0) * 0.35
        sources[4] = np.maximum(-node_fb, 0.0) * 0.30
        C = diffuse(C, sources, steps=3)

        # Reward-modulated edge weight update.
        w = w + 0.020 * edge_delta - 0.010 * (w - w0)
        w = np.clip(w, 0.02, 1.5)

    payload = {
        "nodes": N, "edges": len(edges_u), "rounds": ROUNDS,
        "history": history,
        "final": history[-1],
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/dynamic_chem_edges_result.json", "report": "research_mvp/DYNAMIC_CHEM_EDGES_REPORT.md"},
    }
    (ROOT / "dynamic_chem_edges_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "dynamic_chem_edges_result.json")


if __name__ == "__main__":
    main()
