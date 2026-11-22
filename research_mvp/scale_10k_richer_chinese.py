#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh 10k + richer modern Chinese prose corpus; report accuracy and efficiency."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scale_10k_fresh import build_fresh, chemical_gain, forward_sample, N, DIM
from scale_10k_real import make_chinese, fit_ridge, predict, fit_classes, predict_classes
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "modern_chinese_corpus.txt"


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    rng = np.random.default_rng(2090)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)

    math_x, math_y = task_sum_sq(900)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    cn_train, cn_test, vocab = make_chinese(CORPUS)
    cn_train, cn_test = cn_train[:1000], cn_test[:400]
    print("CORPUS", CORPUS, "train", len(cn_train), "test", len(cn_test), "vocab", len(vocab))

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_sample(x, incoming, bias, input_gain, gain)[2:]
        return X

    t1 = time.perf_counter()
    Xm, Xmv = features(math_x[:400]), features(math_x[400:600])
    Xc = features(np.array([r[0] for r in cn_train], dtype=np.float32))
    Xcv = features(np.array([r[0] for r in cn_test], dtype=np.float32))
    feature_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    mm = fit_ridge(Xm, math_y[:400])
    mae = float(np.mean(np.abs(predict(mm, Xmv) - math_y[400:600])))
    rmse = float(np.sqrt(np.mean((predict(mm, Xmv) - math_y[400:600]) ** 2)))
    cm, _ = fit_classes(Xc, np.array([r[1] for r in cn_train], dtype=np.int64), len(vocab))
    cacc = float(np.mean(predict_classes(cm, Xcv) == np.array([r[1] for r in cn_test], dtype=np.int64)))
    fit_sec = time.perf_counter() - t2

    x_problem = np.array([7 / 20.0, 12 / 100.0] + [0.0] * (DIM - 2), dtype=np.float32)
    act = forward_sample(x_problem, incoming, bias, input_gain, gain)
    problem_pred = float(predict(mm, act[2:].reshape(1, -1))[0])

    total = time.perf_counter() - t0
    n_samples = 600 + len(cn_train) + len(cn_test)
    payload = {
        "task": "fresh 10k, richer modern Chinese corpus",
        "nodes": N,
        "edges": len(edges),
        "corpus": "research_mvp/data/modern_chinese_corpus.txt",
        "math": {"test_mae": round(mae, 5), "test_rmse": round(rmse, 5), "prediction": round(problem_pred, 5)},
        "chinese": {"task": "real next-character prediction, modern prose", "train": len(cn_train), "test": len(cn_test), "vocab_size": len(vocab), "accuracy": round(cacc, 4)},
        "efficiency": {
            "build_and_chem_seconds": round(time.perf_counter() - t0 - feature_sec - fit_sec, 3),
            "feature_seconds": round(feature_sec, 3),
            "fit_seconds": round(fit_sec, 3),
            "total_seconds": round(total, 3),
            "samples": n_samples,
            "ms_per_sample": round(1000.0 * total / n_samples, 3),
        },
        "output_files": {"result_json": "research_mvp/scale_10k_richer_result.json", "report": "research_mvp/SCALE_10K_RICHER_REPORT.md"},
    }
    (ROOT / "scale_10k_richer_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", payload)
    print("SAVED", ROOT / "scale_10k_richer_result.json")


if __name__ == "__main__":
    main()
