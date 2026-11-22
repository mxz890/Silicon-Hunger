#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal 2000-node reservoir: leaky recurrence + slow chemical field.

Tasks:
  - temporal parity over 12 bits
  - delayed match: cue at t0, probe at t11, target cue == probe

No BPTT; fixed recurrent reservoir + readout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from large_math_2000 import build_graph

ROOT = Path(__file__).resolve().parent
VIS = ROOT / "visualizations"
N = 2000
OUTPUT = N - 1
T = 12
ALPHA = 0.22
CHEM_EVERY = 2
rng = np.random.default_rng(2120)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_recurrent_graph():
    pos, incoming, edges, eu, ev, ew = build_graph()
    eu = eu.astype(np.int32)
    ev = ev.astype(np.int32)
    ew = ew.astype(np.float32)
    existing = {(int(a), int(b)) for a, b in zip(eu, ev)}
    added = 0
    while added < 2000:
        u = int(rng.integers(2, OUTPUT))
        v = int(rng.integers(2, OUTPUT))
        if u == v or (u, v) in existing:
            continue
        w = float(rng.uniform(0.02, 0.09))
        incoming[v].append((u, w))
        edges.append((u, v, w))
        eu = np.append(eu, u)
        ev = np.append(ev, v)
        ew = np.append(ew, w)
        existing.add((u, v))
        added += 1
    return pos, incoming, edges, eu, ev, ew


def chemical_gain(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    return np.clip(1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh), 0.4, 1.8)


def chem_step(C, h, x, eu, ev, ew):
    sources = np.zeros((6, N), dtype=np.float32)
    activity = float(np.mean(np.abs(h)))
    sources[0, 0] = 0.08 * abs(float(x))
    sources[2, 1] = 0.06 * abs(float(x))
    sources[5, OUTPUT] = 0.03 * activity
    sources[4, OUTPUT] = 0.03 * (1.0 - activity)
    for k in range(6):
        inc = np.bincount(ev, weights=ew * C[k, eu], minlength=N)
        deg = np.bincount(ev, weights=ew, minlength=N)
        flux = inc - deg * C[k]
        C[k] = np.clip(C[k] + 0.08 * (0.10 * flux - 0.12 * C[k] + sources[k]), 0.0, 3.0)
    return C


def run_sequence(bits, incoming_edges, bias, input_gain, C0):
    h = np.zeros(N, dtype=np.float32)
    C = C0.copy()
    trace = []
    for t, bit in enumerate(bits):
        gain = chemical_gain(C)
        incoming = np.bincount(incoming_edges[0], weights=incoming_edges[2] * h[incoming_edges[1]], minlength=N)
        pre = bias + input_gain[:, 0] * float(bit) + incoming
        h = (1.0 - ALPHA) * h + ALPHA * np.tanh(gain * pre)
        if t % CHEM_EVERY == 0:
            C = chem_step(C, h, bit, incoming_edges[0], incoming_edges[1], incoming_edges[2])
        trace.append(
            {
                "t": t,
                "bit": int(bit),
                "mean_abs_h": round(float(np.mean(np.abs(h))), 5),
                "gain_mean": round(float(gain.mean()), 5),
                "chem": {name: round(float(C[k].mean()), 5) for k, name in enumerate(["DA", "5HT", "NE", "ACh", "GABA", "Glu"])},
            }
        )
    return h[2:], trace


def make_parity(n):
    rows = []
    for _ in range(n):
        bits = rng.integers(0, 2, T).astype(np.float32)
        rows.append((bits, float(int(bits.sum()) % 2)))
    return rows


def make_delayed(n):
    rows = []
    for _ in range(n):
        cue = int(rng.integers(0, 2))
        probe = int(rng.integers(0, 2))
        bits = rng.integers(0, 2, T).astype(np.float32)
        bits[0], bits[-1] = cue, probe
        rows.append((bits, float(cue == probe)))
    return rows


def fit_readout(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    coef, *_ = np.linalg.lstsq(Xs, (y - ym) / ys, rcond=1e-3)
    return {"mean": mean, "std": std, "ym": ym, "ys": ys, "coef": coef}


def predict(model, X):
    return ((X - model["mean"]) / model["std"]) @ model["coef"] * model["ys"] + model["ym"]


def run_task(name, rows, features_fn, trace_holder):
    X = np.empty((len(rows), N - 2), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.float32)
    for i, (bits, target) in enumerate(rows):
        feat, trace = features_fn(bits)
        X[i] = feat
        y[i] = target
        if i == 0:
            trace_holder.append(trace)
    model = fit_readout(X[:700], y[:700])
    val = predict(model, X[700:900])
    test = predict(model, X[900:])
    val_acc = float(np.mean((val >= 0.5) == (y[700:900] >= 0.5)))
    test_acc = float(np.mean((test >= 0.5) == (y[900:] >= 0.5)))
    return {"task": name, "val_acc": round(val_acc, 4), "test_acc": round(test_acc, 4), "train": 700, "val": 200, "test": 300}, model, X, y


def main() -> None:
    t0 = time.perf_counter()
    pos, incoming, edges, eu, ev, ew = build_recurrent_graph()
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, 1)).astype(np.float32)
    C0 = np.full((6, N), 0.05, dtype=np.float32)
    incoming_edges = (eu.astype(np.int32), ev.astype(np.int32), ew.astype(np.float32))

    print("GRAPH", "nodes", N, "edges", len(edges), "T", T, "alpha", ALPHA)
    traces = []
    results = []
    for name, rows, fn in [
        ("temporal_parity", make_parity(1200), lambda b: run_sequence(b, incoming_edges, bias, input_gain, C0)),
        ("delayed_match", make_delayed(1200), lambda b: run_sequence(b, incoming_edges, bias, input_gain, C0)),
    ]:
        task_trace = []
        result, model, X, y = run_task(name, rows, fn, task_trace)
        results.append(result)
        print("RESULT", result)
        if task_trace:
            traces.append((name, task_trace[0], model, X, y))

    # Timeline for the first delayed-match sequence.
    if traces:
        name, trace, model, X, y = traces[-1]
        fig = make_subplots(rows=2, cols=1, subplot_titles=("input bits", "mean |h| and chemical means"))
        t = [x["t"] for x in trace]
        fig.add_trace(go.Scatter(x=t, y=[x["bit"] for x in trace], mode="lines+markers", name="bit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=[x["mean_abs_h"] for x in trace], mode="lines+markers", name="mean |h|"), row=2, col=1)
        for chem in ("DA", "GABA", "Glu"):
            fig.add_trace(go.Scatter(x=t, y=[x["chem"][chem] for x in trace], mode="lines", name=chem), row=2, col=1)
        fig.update_layout(
            paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font=dict(color="#d8e7de"),
            height=520, margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_xaxes(gridcolor="#26352d")
        fig.update_yaxes(gridcolor="#26352d")
        VIS.mkdir(exist_ok=True)
        html = fig.to_html(include_plotlyjs="directory", full_html=False, div_id="chart", config={"responsive": True})
        page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>temporal timeline</title>
<style>body{{background:#0b0f14;color:#d8e7de;font-family:system-ui;margin:24px}}h1{{font-size:22px}}p{{color:#8aa99a}}</style></head>
<body><h1>Temporal reservoir timeline</h1><p>示例序列：cue / noise / probe。看 h 和化学场随 tick 的变化。</p>{html}</body></html>"""
        (VIS / "temporal_timeline.html").write_text(page, encoding="utf-8")

    payload = {
        "nodes": N,
        "edges": len(edges),
        "T": T,
        "alpha": ALPHA,
        "chem_every": CHEM_EVERY,
        "results": results,
        "seconds": round(time.perf_counter() - t0, 3),
        "output_files": {
            "result_json": "research_mvp/temporal_reservoir_result.json",
            "timeline_html": "research_mvp/visualizations/temporal_timeline.html",
            "report": "research_mvp/TEMPORAL_RESERVOIR_REPORT.md",
        },
    }
    (ROOT / "temporal_reservoir_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", ROOT / "temporal_reservoir_result.json")


if __name__ == "__main__":
    main()
