#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node baseline + static iterative growth + a thinking-chain visualization.

No recurrence and no time axis: each iteration is a new static model.
Growth nodes are readout-only RBF features attached to the output.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from large_math_2000 import N, build_graph, chemical_field
from large_math_multi import forward_multi, task_linear

ROOT = Path(__file__).resolve().parent
VIS = ROOT / "visualizations"
OUTPUT = 1999
rng = np.random.default_rng(2032)


def fit_ridge(X, y, rcond=1e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    coef, *_ = np.linalg.lstsq(Xs, (y - ym) / ys, rcond=rcond)
    return {"mean": mean, "std": std, "ym": ym, "ys": ys, "coef": coef}


def predict(model, X):
    return ((X - model["mean"]) / model["std"]) @ model["coef"] * model["ys"] + model["ym"]


def metrics(y, pred):
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {"mae": round(mae, 5), "rmse": round(rmse, 5), "relative_mae": round(mae / (float(np.mean(np.abs(y))) + 1e-9), 5)}


def rbf(xs, center, sigma):
    return np.exp(-np.sum((xs - center) ** 2, axis=1) / (2.0 * sigma * sigma)).astype(np.float32)


def greedy_chain(graph, activation, start=OUTPUT, max_len=90):
    chain = [start]
    current = start
    visited = {start}
    while len(chain) < max_len:
        preds = list(graph.predecessors(current))
        if not preds:
            break
        nxt = max(preds, key=lambda u: abs(activation[u] * graph[u][current]["weight"]))
        if nxt in visited:
            break
        chain.append(nxt)
        visited.add(nxt)
        current = nxt
        if current in (0, 1):
            break
    return list(reversed(chain))


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges, eu, ev, ew = build_graph()
    C, gain = chemical_field(eu, ev, ew)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 3)).astype(np.float32)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    def base_features(xs):
        X = np.empty((len(xs), N - 2), dtype=np.float32)
        for i, x in enumerate(xs):
            X[i] = forward_multi(x, incoming, bias, input_gain, gain)[2:]
        return X

    x_all, y_all = task_linear(2000)
    X_train, X_val, X_test = base_features(x_all[:1000]), base_features(x_all[1000:1300]), base_features(x_all[1300:])
    y_train, y_val, y_test = y_all[:1000], y_all[1000:1300], y_all[1300:]

    X_train_aug, X_val_aug, X_test_aug = X_train.copy(), X_val.copy(), X_test.copy()
    model = fit_ridge(X_train_aug, y_train)
    base_val = metrics(y_val, predict(model, X_val_aug))
    base_test = metrics(y_test, predict(model, X_test_aug))
    print("BASELINE", base_val, base_test)

    history = []
    centers = []
    features_train = []
    features_val = []
    features_test = []
    best_val = base_val["mae"]
    max_rounds = 8

    for iteration in range(1, max_rounds + 1):
        pred_train = predict(model, X_train_aug)
        residual = y_train - pred_train
        high = np.argsort(np.abs(residual))[-40:]
        candidate_indices = rng.choice(high, size=min(6, len(high)), replace=False)
        best = None
        for idx in candidate_indices:
            center = x_all[idx]
            for sigma in (0.08, 0.15, 0.25):
                h_train = rbf(x_all[:1000], center, sigma).reshape(-1, 1)
                h_val = rbf(x_all[1000:1300], center, sigma).reshape(-1, 1)
                candidate_X = np.hstack([X_train_aug, h_train])
                candidate_model = fit_ridge(candidate_X, y_train)
                val = metrics(y_val, predict(candidate_model, np.hstack([X_val_aug, h_val])))
                if best is None or val["mae"] < best["val"]["mae"]:
                    best = {
                        "val": val, "model": candidate_model, "center": center,
                        "sigma": sigma, "h_train": h_train, "h_val": h_val,
                        "h_test": rbf(x_all[1300:], center, sigma).reshape(-1, 1),
                    }
        if best is not None and best["val"]["mae"] < best_val - 1e-5:
            best_val = best["val"]["mae"]
            model = best["model"]
            X_train_aug = np.hstack([X_train_aug, best["h_train"]])
            X_val_aug = np.hstack([X_val_aug, best["h_val"]])
            X_test_aug = np.hstack([X_test_aug, best["h_test"]])
            centers.append(best["center"])
            features_train.append(best["h_train"])
            features_val.append(best["h_val"])
            features_test.append(best["h_test"])
            test = metrics(y_test, predict(model, X_test_aug))
            history.append(
                {
                    "iteration": iteration, "accepted": True,
                    "sigma": best["sigma"], "val": best["val"], "test": test,
                    "center": [round(float(v), 4) for v in best["center"]],
                }
            )
            print("GROW", iteration, "sigma", best["sigma"], "val", best["val"], "test", test)
        else:
            history.append({"iteration": iteration, "accepted": False, "best_val": best["val"] if best else None})
            print("REJECT", iteration)
            break

    final_val = metrics(y_val, predict(model, X_val_aug))
    final_test = metrics(y_test, predict(model, X_test_aug))

    # A held-out linear equation: 3x + 5 = 20 -> x = 5.
    x_problem = np.array([[3 / 10.0, 5 / 30.0, 20 / 30.0]], dtype=np.float32)
    act_problem = forward_multi(x_problem[0], incoming, bias, input_gain, gain)
    feature_problem = np.concatenate([act_problem[2:], *[rbf(x_problem, c, s).reshape(-1) for c, s in zip(centers, [h["sigma"] for h in history if h["accepted"]])]])
    pred_problem = float(predict(model, feature_problem.reshape(1, -1))[0])

    z = (feature_problem - model["mean"]) / model["std"]
    contribution = z * model["coef"]
    top_base = np.argsort(np.abs(contribution[: N - 2]))[-15:][::-1]
    top_nodes = [(int(k + 2), float(contribution[k])) for k in top_base]

    chain = greedy_chain(graph, act_problem)
    selected = set(chain) | {0, 1, OUTPUT} | {n for n, _ in top_nodes}
    sub_edges = [(u, v) for u, v in graph.edges if u in selected and v in selected]
    chain_edges = set(zip(chain, chain[1:]))

    # Visualization: selected actual graph + readout-only growth nodes.
    vis_dir = VIS
    vis_dir.mkdir(exist_ok=True)
    plotly_js = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"
    if plotly_js.exists() and not (vis_dir / "plotly.min.js").exists():
        shutil.copyfile(plotly_js, vis_dir / "plotly.min.js")

    xs = [pos[i][0] for i in selected]
    ys = [pos[i][1] for i in selected]
    zs = [pos[i][2] for i in selected]
    colors = [float(act_problem[i]) for i in selected]
    labels = [str(i) for i in selected]
    sizes = [8 + 10 * abs(act_problem[i]) for i in selected]

    fig = go.Figure()
    # Base edges
    ex, ey, ez = [], [], []
    for u, v in sub_edges:
        if (u, v) in chain_edges:
            continue
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
        ez += [pos[u][2], pos[v][2], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#4b5d54", width=2), hoverinfo="skip"))
    # Chain edges
    cx, cy, cz = [], [], []
    for u, v in chain_edges:
        cx += [pos[u][0], pos[v][0], None]
        cy += [pos[u][1], pos[v][1], None]
        cz += [pos[u][2], pos[v][2], None]
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", line=dict(color="#f1c40f", width=6), hoverinfo="skip"))
    # Base nodes
    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers+text", text=labels, textposition="top center",
            textfont=dict(size=8, color="#d8e7de"),
            hovertext=[f"node {i}<br>act {act_problem[i]:.4f}" for i in selected], hoverinfo="text",
            marker=dict(size=sizes, color=colors, colorscale="Viridis", line=dict(width=1, color="#173d2c")),
            showlegend=False,
        )
    )
    # Growth / readout-only nodes
    if centers:
        gx, gy, gz = [], [], []
        for k, center in enumerate(centers):
            gx.append(50 + 18 * math.cos(k * 1.7))
            gy.append(50 + 18 * math.sin(k * 1.7))
            gz.append(50 + 12 * (k + 1))
        fig.add_trace(
            go.Scatter3d(
                x=gx, y=gy, z=gz, mode="markers+text",
                text=[f"g{i+1}" for i in range(len(centers))], textposition="top center",
                textfont=dict(color="#e74c3c"),
                marker=dict(size=12, color="#e74c3c", symbol="diamond"),
                hovertext=[f"readout growth node {i+1}<br>center {c}" for i, c in enumerate(centers)],
                hoverinfo="text", showlegend=False,
            )
        )
    fig.update_layout(
        title=dict(text=f"Thinking chain — 3x+5=20 -> {pred_problem:.3f}", font=dict(color="#d8e7de", size=15)),
        paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        scene=dict(
            bgcolor="#0b0f14", aspectmode="cube", dragmode="orbit",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.4)),
        ),
        margin=dict(l=0, r=0, t=45, b=0), showlegend=False, font=dict(color="#d8e7de"),
    )

    hud = f"""
    <div style="position:fixed;left:16px;top:14px;z-index:10;background:rgba(11,15,20,.82);
    border:1px solid #2f7d5b;border-radius:10px;padding:12px 16px;color:#d8e7de;
    font-family:system-ui;max-width:380px;font-size:13px;line-height:1.65">
    <b>Thinking chain / static iteration</b><br>
    problem: 3x + 5 = 20 → target 5<br>
    prediction: <b>{pred_problem:.4f}</b><br><br>
    baseline test MAE: {base_test['mae']}<br>
    grown test MAE: {final_test['mae']}<br>
    growth nodes: {len(centers)}<br><br>
    <span style="color:#f1c40f">yellow path</span> = greedy activation chain<br>
    <span style="color:#e74c3c">red diamonds</span> = readout-only growth<br>
    <span style="color:#8aa99a">grey</span> = selected contributing edges
    </div>
    """
    fig_html = fig.to_html(
        include_plotlyjs="directory", full_html=False, div_id="graph",
        config={"displayModeBar": False, "responsive": True, "scrollZoom": True, "doubleClick": "reset"},
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <title>Thinking chain</title><style>html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#0b0f14}}</style>
    </head><body>{hud}{fig_html}</body></html>"""
    chain_path = vis_dir / "thought_chain.html"
    chain_path.write_text(page, encoding="utf-8")

    payload = {
        "task": "linear: ax+b=c -> x=(c-b)/a",
        "baseline": {"val": base_val, "test": base_test, "features": 1998},
        "grown": {"val": final_val, "test": final_test, "features": int(X_train_aug.shape[1]), "new_nodes": len(centers)},
        "history": history,
        "problem": {"statement": "3x+5=20", "target": 5, "prediction": round(pred_problem, 5)},
        "top_contributors": top_nodes,
        "thought_chain": [int(i) for i in chain],
        "output_files": {
            "result_json": "research_mvp/iterative_growth_2000.json",
            "thought_chain_html": "research_mvp/visualizations/thought_chain.html",
            "report": "research_mvp/ITERATIVE_GROWTH_REPORT.md",
        },
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "iterative_growth_2000.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", final_val, final_test, "problem", pred_problem, "growth", len(centers))
    print("SAVED", chain_path)


if __name__ == "__main__":
    main()
