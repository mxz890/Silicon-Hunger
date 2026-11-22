#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error-driven, readout-only growth on the 2000-node reservoir.

Each new node is a nonlinear residual feature attached only to the readout.
Old graph activations never change.  A candidate is accepted only if it
improves the validation MAE.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from large_math_2000 import N, build_graph, chemical_field
from large_math_multi import forward_multi, task_arithmetic, task_linear

ROOT = Path(__file__).resolve().parent


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return mean, std


def ridge_fit(X, y, rcond=1e-3):
    mean, std = standardize_fit(X)
    Xs = (X - mean) / std
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    coef, *_ = np.linalg.lstsq(Xs, (y - ym) / ys, rcond=rcond)
    return {"mean": mean, "std": std, "ym": ym, "ys": ys, "coef": coef}


def predict(model, X):
    Xs = (X - model["mean"]) / model["std"]
    return Xs @ model["coef"] * model["ys"] + model["ym"]


def metrics(y, pred):
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rel = mae / (float(np.mean(np.abs(y))) + 1e-9)
    return {"mae": round(mae, 5), "rmse": round(rmse, 5), "relative_mae": round(rel, 5)}


def grow_task(name, generator, features, max_rounds=12):
    x, y = generator(2000)
    X_all = features(x)
    X_train, X_val, X_test = X_all[:1000], X_all[1000:1300], X_all[1300:2000]
    y_train, y_val, y_test = y[:1000], y[1000:1300], y[1300:2000]

    model = ridge_fit(X_train, y_train)
    base_val = metrics(y_val, predict(model, X_val))
    base_test = metrics(y_test, predict(model, X_test))
    best_val = base_val["mae"]
    print(f"TASK {name} baseline val={base_val['mae']:.4f} test={base_test['mae']:.4f}")

    X_train_aug = X_train.copy()
    X_val_aug = X_val.copy()
    X_test_aug = X_test.copy()
    history = []
    accepted = 0

    for round_no in range(1, max_rounds + 1):
        pred_train = predict(model, X_train_aug)
        residual = y_train - pred_train
        # Error direction in the current feature space.
        direction, *_ = np.linalg.lstsq(X_train_aug, residual, rcond=1e-3)
        z_train = X_train_aug @ direction
        z_val = X_val_aug @ direction
        z_test = X_test_aug @ direction
        z_mean, z_std = float(z_train.mean()), float(z_train.std() + 1e-6)

        best = None
        for gamma in (0.5, 1.0, 2.0):
            h_train = np.tanh(gamma * (z_train - z_mean) / z_std).reshape(-1, 1)
            h_val = np.tanh(gamma * (z_val - z_mean) / z_std).reshape(-1, 1)
            h_test = np.tanh(gamma * (z_test - z_mean) / z_std).reshape(-1, 1)
            candidate = ridge_fit(np.hstack([X_train_aug, h_train]), y_train)
            val = metrics(y_val, predict(candidate, np.hstack([X_val_aug, h_val])))
            if best is None or val["mae"] < best["val"]["mae"]:
                best = {
                    "gamma": gamma, "model": candidate, "val": val,
                    "h_train": h_train, "h_val": h_val, "h_test": h_test,
                }

        if best is not None and best["val"]["mae"] < best_val - 1e-5:
            best_val = best["val"]["mae"]
            accepted += 1
            model = best["model"]
            X_train_aug = np.hstack([X_train_aug, best["h_train"]])
            X_val_aug = np.hstack([X_val_aug, best["h_val"]])
            X_test_aug = np.hstack([X_test_aug, best["h_test"]])
            test = metrics(y_test, predict(model, X_test_aug))
            history.append(
                {
                    "round": round_no, "accepted": True, "gamma": best["gamma"],
                    "val": best["val"], "test": test, "features": X_train_aug.shape[1],
                }
            )
            print(
                f"  grow {accepted:02d} round={round_no:02d} gamma={best['gamma']} "
                f"val={best['val']['mae']:.4f} test={test['mae']:.4f}"
            )
        else:
            history.append({"round": round_no, "accepted": False, "best_val": best["val"] if best else None})
            print(f"  reject round={round_no:02d}: no validation improvement")
            break

    final_val = metrics(y_val, predict(model, X_val_aug))
    final_test = metrics(y_test, predict(model, X_test_aug))
    return {
        "task": name,
        "baseline": {"val": base_val, "test": base_test, "features": X_train.shape[1]},
        "grown": {
            "val": final_val, "test": final_test,
            "features": X_train_aug.shape[1], "new_nodes": accepted,
        },
        "history": history,
    }


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges, eu, ev, ew = build_graph()
    C, gain = chemical_field(eu, ev, ew)
    # Use a deterministic reservoir RNG separate from task data.
    rng = np.random.default_rng(2030)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 3)).astype(np.float32)

    def features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_multi(x, incoming, bias, input_gain, gain)[2:]
        return X

    results = []
    for name, gen in (("linear", task_linear), ("arithmetic", task_arithmetic)):
        results.append(grow_task(name, gen, features))

    payload = {
        "graph": {"nodes": N, "edges": len(edges)},
        "chemicals": {
            "gain_min": round(float(gain.min()), 5),
            "gain_max": round(float(gain.max()), 5),
        },
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {
            "result_json": "research_mvp/readout_growth_result.json",
            "report": "research_mvp/READOUT_GROWTH_REPORT.md",
        },
    }
    (ROOT / "readout_growth_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TOTAL", round(payload["seconds"], 3), "seconds")
    print("SAVED", ROOT / "readout_growth_result.json")


if __name__ == "__main__":
    main()
