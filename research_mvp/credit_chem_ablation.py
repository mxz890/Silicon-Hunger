#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation: no chemistry vs broadcast chemistry vs credit-based chemistry.

2000-node fixed graph; 8 static training rounds; math + real Chinese.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from large_math_2000 import build_graph
from large_math_multi import task_sum_sq
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes

ROOT = Path(__file__).resolve().parent
N = 2000
OUTPUT = N - 1
DIM = 64
ROUNDS = 8
rng = np.random.default_rng(2110)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def features(xs, incoming, bias, input_gain, gain):
    out = np.empty((len(xs), N - 2), dtype=np.float32)
    for i, x in enumerate(xs):
        act = np.zeros(N, dtype=np.float32)
        act[0], act[1] = x[0], x[1]
        term = input_gain @ x
        for j in range(2, N):
            pre = bias[j] + term[j]
            for u, w in incoming[j]:
                pre += w * act[u]
            act[j] = np.tanh(gain[j] * pre)
        out[i] = act[2:]
    return out


def math_credit(act, model, target):
    z = (act[2:] - model["mean"]) / model["std"]
    pred = float((z @ model["coef"] * model["ys"] + model["ym"]).ravel()[0])
    err = pred - float(target)
    return (-err * (model["coef"].ravel() * z)).astype(np.float32), pred


def chinese_credit(act, model, label):
    z = (act[2:] - model["mean"]) / model["std"]
    logits = (z @ model["coef"] * model["ys"] + model["ym"]).reshape(1, -1)
    p = softmax(logits)[0]
    y = np.zeros_like(p)
    y[int(label)] = 1.0
    grad = (p - y)  # dL/dlogit for cross entropy
    credit = ((model["coef"] * grad[None, :]).sum(axis=1) * z).astype(np.float32)
    return credit, int(np.argmax(p))


def init_state(mode):
    _, incoming, edges, eu, ev, ew = build_graph()
    eu = eu.astype(np.int32)
    ev = ev.astype(np.int32)
    ew = ew.astype(np.float32)
    C = np.full((6, N), 0.05, dtype=np.float32)
    if mode == "no_chem":
        incoming_eff = incoming
        gain = np.ones(N, dtype=np.float32)
        active = np.ones(len(eu), dtype=bool)
    else:
        gain = chemical_gain(C)
        gate = gates(C)
        we = effective(ew, gate, eu, ev)
        incoming_eff = build_incoming(eu, ev, we)
        active = we > 0.08
    return incoming, eu, ev, ew, C, gain, incoming_eff, active


def gates(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    return 1.0 / (1.0 + np.exp(-np.clip(2.5 * (exc - inh - 0.35), -30, 30)))


def effective(w, gate, u, v):
    return w * (0.2 + 1.4 * gate[u] * gate[v])


def build_incoming(u, v, we):
    inc = [[] for _ in range(N)]
    for a, b, w in zip(u, v, we):
        if w > 0.08:
            inc[b].append((int(a), float(w)))
    return inc


def chemical_gain(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    return np.clip(1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh), 0.4, 1.8)


def diffuse(C, sources, edge_u, edge_v, edge_w):
    for _ in range(2):
        for k in range(6):
            inc = np.bincount(edge_v, weights=edge_w * C[k, edge_u], minlength=N)
            deg = np.bincount(edge_v, weights=edge_w, minlength=N)
            flux = inc - deg * C[k]
            C[k] = np.clip(C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k]), 0.0, 3.0)
    return C


def run_mode(mode, math_x, math_y, cn_train, cn_test, vocab):
    base_incoming, edge_u, edge_v, edge_w, C, gain, incoming, active = init_state(mode)
    w = edge_w.copy()
    w0 = w.copy()
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)
    history = []

    def refit():
        nonlocal gain, incoming, active
        if mode == "no_chem":
            gain = np.ones(N, dtype=np.float32)
            incoming = base_incoming
            active = np.ones(len(edge_u), dtype=bool)
        else:
            gate = gates(C)
            we = effective(w, gate, edge_u, edge_v)
            incoming = build_incoming(edge_u, edge_v, we)
            active = we > 0.08
            gain = chemical_gain(C)

    for round_no in range(ROUNDS + 1):
        refit()
        Xm, Xmv = features(math_x[:300], incoming, bias, input_gain, gain), features(math_x[300:450], incoming, bias, input_gain, gain)
        mm = fit_ridge(Xm, math_y[:300])
        mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[300:450])))
        Xc = features(np.array([r[0] for r in cn_train[:300]], dtype=np.float32), incoming, bias, input_gain, gain)
        Xcv = features(np.array([r[0] for r in cn_test[:150]], dtype=np.float32), incoming, bias, input_gain, gain)
        cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train[:300]], dtype=np.int64), len(vocab))
        pred_c = np.argmax(predict(cm, Xcv), axis=1)
        cacc = float(np.mean(pred_c == np.array([r[1] for r in cn_test[:150]], dtype=np.int64)))
        history.append({"round": round_no, "math_mae": round(mae, 5), "chinese_acc": round(cacc, 4), "active_edges": int(active.sum())})
        print(mode, history[-1])
        if round_no == ROUNDS:
            break

        node_credit = np.zeros(N, dtype=np.float32)
        edge_credit = np.zeros(len(edge_u), dtype=np.float32)
        count = 0
        for i in range(80):
            act = np.zeros(N, dtype=np.float32)
            act[0], act[1] = math_x[i][0], math_x[i][1]
            term = input_gain @ math_x[i]
            for j in range(2, N):
                pre = bias[j] + term[j]
                for u, ww in incoming[j]:
                    pre += ww * act[u]
                act[j] = np.tanh(gain[j] * pre)
            if mode == "broadcast":
                pred = float(predict(mm, act[2:].reshape(1, -1))[0])
                fb = 1.0 if abs(pred - math_y[i]) <= 1.0 else -1.0
                node_credit += fb * np.abs(act)
                edge_credit += fb * act[edge_u] * act[edge_v]
            else:
                credit, _ = math_credit(act, mm, math_y[i])
                cc = np.zeros(N, dtype=np.float32)
                cc[2:] = credit
                node_credit += cc
                edge_credit += cc[edge_v] * act[edge_u]
            count += 1

        for i in range(80):
            x, label = cn_train[i]
            x = np.asarray(x, dtype=np.float32)
            act = np.zeros(N, dtype=np.float32)
            act[0], act[1] = x[0], x[1]
            term = input_gain @ x
            for j in range(2, N):
                pre = bias[j] + term[j]
                for u, ww in incoming[j]:
                    pre += ww * act[u]
                act[j] = np.tanh(gain[j] * pre)
            if mode == "broadcast":
                pred = int(np.argmax(predict(cm, act[2:].reshape(1, -1))[0]))
                fb = 1.0 if pred == int(label) else -1.0
                node_credit += fb * np.abs(act)
                edge_credit += fb * act[edge_u] * act[edge_v]
            else:
                credit, _ = chinese_credit(act, cm, label)
                cc = np.zeros(N, dtype=np.float32)
                cc[2:] = credit
                node_credit += cc
                edge_credit += cc[edge_v] * act[edge_u]
            count += 1

        node_credit /= max(1, count)
        edge_credit /= max(1, count)

        # Sparse contribution allocation.
        sources = np.zeros((6, N), dtype=np.float32)
        if mode == "broadcast":
            sources[0] = np.maximum(node_credit, 0) * 0.35
            sources[5] = np.maximum(node_credit, 0) * 0.25
            sources[2] = np.maximum(-node_credit, 0) * 0.35
            sources[4] = np.maximum(-node_credit, 0) * 0.30
        elif mode == "credit":
            pos = np.argsort(np.maximum(node_credit, 0))[-20:]
            neg = np.argsort(np.maximum(-node_credit, 0))[-20:]
            for idx in pos:
                if node_credit[idx] > 0:
                    sources[0, idx] += 0.6
                    sources[5, idx] += 0.4
                    sources[3, idx] += 0.2
            for idx in neg:
                if node_credit[idx] < 0:
                    sources[4, idx] += 0.5
                    sources[2, idx] += 0.3
                    sources[1, idx] += 0.2
        if mode != "no_chem":
            C = diffuse(C, sources, edge_u, edge_v, edge_w)
            w = np.clip(w + 0.02 * edge_credit - 0.01 * (w - w0), 0.02, 1.5)
    return {"mode": mode, "history": history}


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(700)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    cn_train, cn_test, vocab = make_chinese()
    cn_train, cn_test = cn_train[:300], cn_test[:150]

    results = []
    for mode in ("no_chem", "broadcast", "credit"):
        results.append(run_mode(mode, math_x, math_y, cn_train, cn_test, vocab))

    payload = {"graph": {"nodes": N, "edges": 9022}, "results": results, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "credit_chem_ablation_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "credit_chem_ablation_result.json")


if __name__ == "__main__":
    main()
