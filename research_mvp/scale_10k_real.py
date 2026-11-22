#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000 -> 10,000 with new nodes entering the output path, 10 growth rounds,
and a real Chinese next-character task from public-domain Lu Xun stories.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from large_math_2000 import build_graph as build_seed
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "luxun_corpus.txt"
TARGET_N = 10000
DIM = 64
rng = np.random.default_rng(2050)


def build_initial():
    pos, incoming, edges, _, _, _ = build_seed()
    # Base output is 1999.  It becomes a hidden node; output is always last.
    return pos.astype(np.float32), incoming, edges


def grow_stage(pos, incoming, edges, start_n, end_n):
    # Preallocate positions in the caller; here pos already has length end_n.
    for j in range(start_n, end_n):
        pos[j] = np.clip(pos[j - 1] + rng.normal(0, 2.0, 3), 0, 100)
        incoming.append([])
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
    output = end_n - 1
    # Explicit output path for every newly added node.
    for j in range(start_n, output):
        w = float(rng.uniform(0.2, 0.7))
        incoming[output].append((j, w))
        edges.append((j, output, w))
    return pos, incoming, edges


def chemical_gain(edge_u, edge_v, edge_w, n):
    output = n - 1
    C = np.full((6, n), 0.05, dtype=np.float32)
    sources = np.zeros((6, n), dtype=np.float32)
    sources[0, 0] = 0.22
    sources[1, 1] = 0.12
    sources[2, 0] = 0.18
    sources[3, 1] = 0.14
    sources[4, output] = 0.16
    sources[5, output] = 0.16
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


def fit_classes(X, labels, n_classes):
    Y = np.zeros((len(labels), n_classes), dtype=np.float32)
    Y[np.arange(len(labels)), labels] = 1.0
    return fit_ridge(X, Y), None


def predict_classes(model, X):
    return np.argmax(predict(model, X), axis=1)


def encode_context(text):
    vec = np.zeros(DIM, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def make_chinese(data_path=None):
    raw = Path(data_path or DATA).read_text(encoding="utf-8")
    text = re.sub(r"===.*?===", "", raw, flags=re.S)
    text = re.sub(r"\s+", "", text)
    cutoff = int(len(text) * 0.8)
    train_text = text[:cutoff]
    vocab_chars = [ch for ch, _ in Counter(train_text).most_common(64)]
    vocab = {ch: i for i, ch in enumerate(vocab_chars)}
    rows = []
    window = 8
    for i in range(len(text) - window - 1):
        target = text[i + window]
        if target not in vocab:
            continue
        rows.append((i, encode_context(text[i:i + window]), vocab[target]))
    train = [(x, y) for i, x, y in rows if i + window < cutoff]
    test = [(x, y) for i, x, y in rows if i + window >= cutoff]
    return train, test, vocab_chars


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges = build_initial()
    pos_full = np.empty((TARGET_N, 3), dtype=np.float32)
    pos_full[:len(pos)] = pos
    pos = pos_full

    math_x, math_y = task_sum_sq(900)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    train_cn, test_cn, vocab = make_chinese()
    cn_train = train_cn[:800]
    cn_test = test_cn[:300]

    stages = []
    start_n = 2000
    for round_no in range(1, 11):
        end_n = 2000 + 800 * round_no
        pos, incoming, edges = grow_stage(pos, incoming, edges, start_n, end_n)
        start_n = end_n
        n = end_n
        if n > TARGET_N:
            break
        eu = np.array([e[0] for e in edges], dtype=np.int32)
        ev = np.array([e[1] for e in edges], dtype=np.int32)
        ew = np.array([e[2] for e in edges], dtype=np.float32)
        C, gain = chemical_gain(eu, ev, ew, n)
        bias = rng.normal(0, 0.15, n).astype(np.float32)
        input_gain = rng.normal(0, 0.7, (n, DIM)).astype(np.float32)

        def feats(xs):
            X = np.empty((len(xs), n - 2), dtype=np.float32)
            for i, x in enumerate(xs):
                X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
            return X

        # Small stage probes: 60 train / 30 test per task.
        Xm, Xmv = feats(math_x[:60]), feats(math_x[60:90])
        mm = fit_ridge(Xm, math_y[:60])
        mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[60:90])))
        Xc = feats(np.array([r[0] for r in cn_train[:60]], dtype=np.float32))
        Xcv = feats(np.array([r[0] for r in cn_test[:30]], dtype=np.float32))
        cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train[:60]], dtype=np.int64), len(vocab))
        cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test[:30]], dtype=np.int64)))
        out_edges = sum(1 for u, v, w in edges if v == n - 1)
        stages.append({"round": round_no, "nodes": n, "edges": len(edges), "output_indegree": out_edges, "math_mae": round(mae, 5), "chinese_acc": round(cacc, 4)})
        print("ROUND", stages[-1])

    # Final full test at 10k.  Every new node gets a direct edge to the
    # final output, so all 8000 new nodes are on the output path.
    n = TARGET_N
    output = n - 1
    existing = {(u, v) for u, v, _ in edges if v == output}
    for j in range(2000, output):
        if (j, output) not in existing:
            w = float(rng.uniform(0.2, 0.7))
            incoming[output].append((j, w))
            edges.append((j, output, w))
            existing.add((j, output))
    print("OUTPUT_EDGES", output, "indegree", sum(1 for u, v, w in edges if v == output))
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew, n)
    bias = rng.normal(0, 0.15, n).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (n, DIM)).astype(np.float32)

    def feats(xs):
        X = np.empty((len(xs), n - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    print("FINAL_EXTRACT", n)
    Xm, Xmv = feats(math_x[:400]), feats(math_x[400:600])
    mm = fit_ridge(Xm, math_y[:400])
    mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[400:600])))
    rmse = float(np.sqrt(np.mean((predict(mm, Xmv) - math_y[400:600]) ** 2)))

    Xc = feats(np.array([r[0] for r in cn_train[:600]], dtype=np.float32))
    Xcv = feats(np.array([r[0] for r in cn_test[:300]], dtype=np.float32))
    cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train[:600]], dtype=np.int64), len(vocab))
    cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test[:300]], dtype=np.int64)))

    x_problem = np.array([7 / 20.0, 12 / 100.0] + [0.0] * (DIM - 2), dtype=np.float32)
    act = forward_sample(x_problem, incoming, bias, input_gain, gain)
    problem_pred = float(predict(mm, act[2:].reshape(1, -1))[0])

    payload = {
        "seed_nodes": 2000,
        "final_nodes": n,
        "final_edges": len(edges),
        "new_nodes_enter_output": True,
        "output_node": n - 1,
        "output_indegree": int(sum(1 for u, v, w in edges if v == n - 1)),
        "stages": stages,
        "math": {"test_mae": round(mae, 5), "test_rmse": round(rmse, 5), "prediction": round(problem_pred, 5)},
        "chinese": {"task": "real next-character prediction (Lu Xun corpus)", "test_accuracy": round(cacc, 4), "vocab_size": len(vocab)},
        "data": {"source": "research_mvp/data/luxun_corpus.txt", "stories": ["狂人日记", "孔乙己", "故乡", "药"]},
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/scale_10k_real_result.json", "report": "research_mvp/SCALE_10K_REAL_REPORT.md"},
    }
    (ROOT / "scale_10k_real_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", payload)
    print("SAVED", ROOT / "scale_10k_real_result.json")


if __name__ == "__main__":
    main()
