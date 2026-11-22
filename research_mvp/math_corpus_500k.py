#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh 10k base + ~500k-character Chinese math corpus.

Five task families, 2000 problems each.  Text is encoded with character
hashing; numeric features are concatenated.  Each task has its own ridge
readout on the same 10k reservoir.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from scale_10k_fresh import build_fresh, chemical_gain, forward_sample, N, DIM
from scale_10k_real import fit_ridge, predict

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(2130)
PER_TASK = 2000
CORPUS = ROOT / "data" / "math_corpus_500k.txt"


def text_hash(text):
    vec = np.zeros(16, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % 16] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def to_input(num, text):
    x = np.zeros(DIM, dtype=np.float32)
    x[:min(8, len(num))] = num[:8]
    x[8:24] = text_hash(text)
    return x


def make_sum_sq():
    S = int(rng.integers(3, 31))
    P = int(rng.integers(1, max(2, S * S // 4) + 1))
    target = S * S - 2 * P
    text = f"小明在数学竞赛中遇到一道代数题：已知两个数的和为{S}，积为{P}，请你求出这两个数的平方和。请写出计算过程，并给出最终答案。"
    return to_input([S / 30, P / 225], text), float(target), text


def make_linear():
    a = int(rng.integers(1, 10)); b = int(rng.integers(1, 21)); c = int(rng.integers(1, 31))
    target = (c - b) / a
    text = f"小华在课后作业中遇到一个一元一次方程：{a}x+{b}={c}，请你解出未知数x的值。请写出移项和化简过程，并给出最终答案。"
    return to_input([a / 10, b / 30, c / 30], text), float(target), text


def make_rect():
    l = int(rng.integers(1, 31)); w = int(rng.integers(1, 31))
    target = l * w
    text = f"学校要计算一块长方形操场的面积，已知操场的长为{l}米，宽为{w}米，请计算这块操场的面积。请写出公式和计算过程。"
    return to_input([l / 30, w / 30], text), float(target), text


def make_arith():
    a1 = int(rng.integers(1, 21)); d = int(rng.integers(1, 11)); n = int(rng.integers(3, 21))
    target = a1 + (n - 1) * d
    text = f"在数列兴趣小组中，一个等差数列的首项是{a1}，公差是{d}，请你求出这个数列的第{n}项。请写出通项公式并代入计算。"
    return to_input([a1 / 20, d / 10, n / 20], text), float(target), text


def make_hyp():
    a = int(rng.integers(1, 31)); b = int(rng.integers(1, 31))
    target = math.sqrt(a * a + b * b)
    text = f"一个直角三角形的两条直角边分别是{a}米和{b}米，请你求出斜边的长度。请写出勾股定理并给出最终结果。"
    return to_input([a / 30, b / 30], text), float(target), text


TASKS = [
    ("sum_sq", make_sum_sq), ("linear", make_linear), ("rectangle", make_rect),
    ("arithmetic", make_arith), ("hypotenuse", make_hyp),
]


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    rng_model = np.random.default_rng(2131)
    bias = rng_model.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng_model.normal(0, 0.7, (N, DIM)).astype(np.float32)

    all_text = []
    task_data = {}
    for name, gen in TASKS:
        rows = [gen() for _ in range(PER_TASK)]
        task_data[name] = rows
        all_text.extend(row[2] for row in rows)
    CORPUS.write_text("\n".join(all_text), encoding="utf-8")
    print("CORPUS", "lines", len(all_text), "chars", len("\n".join(all_text)))

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    results = []
    for name, _ in TASKS:
        rows = task_data[name]
        x = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows], dtype=np.float32)
        t1 = time.perf_counter()
        X = features(x)
        train, test = 1600, 400
        model = fit_ridge(X[:train], y[:train])
        pred = predict(model, X[train:])
        mae = float(np.mean(np.abs(pred - y[train:])))
        rmse = float(np.sqrt(np.mean((pred - y[train:]) ** 2)))
        rel = mae / (float(np.mean(np.abs(y[train:]))) + 1e-9)
        result = {"task": name, "train": train, "test": test, "mae": round(mae, 5), "rmse": round(rmse, 5), "relative_mae": round(rel, 5), "seconds": round(time.perf_counter() - t1, 2)}
        results.append(result)
        print("RESULT", result)

    payload = {
        "graph": {"nodes": N, "edges": len(edges), "mode": "fresh 10k base"},
        "corpus": {"path": "research_mvp/data/math_corpus_500k.txt", "chars": len("\n".join(all_text)), "problems": len(all_text)},
        "tasks": results,
        "total_seconds": round(time.perf_counter() - t0, 3),
        "output_files": {"result_json": "research_mvp/math_500k_result.json", "report": "research_mvp/MATH_500K_REPORT.md"},
    }
    (ROOT / "math_500k_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["total_seconds"])
    print("SAVED", ROOT / "math_500k_result.json")


if __name__ == "__main__":
    main()
