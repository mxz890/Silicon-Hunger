#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grow -> train (backprop) -> validate -> grow loop.

Prototype on 2000 nodes with K-step differentiable message passing.
The graph topology is fixed inside each inner training phase; growth is an
outer discrete step.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from large_math_2000 import build_graph
from large_math_multi import task_sum_sq

ROOT = Path(__file__).resolve().parent
DIM = 64
K = 8
GROW = 50
ROUNDS = 6
PATIENCE = 4
EPOCHS = 8
BATCH = 64
LR = 8e-4
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2171
rng = np.random.default_rng(SEED - 1)
torch.manual_seed(SEED)


class Brain(nn.Module):
    def __init__(self, n, eu, ev, ew):
        super().__init__()
        self.n = n
        self.output = n - 1
        self.register_buffer("src", torch.tensor(eu, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(ev, dtype=torch.long))
        self.w = nn.Parameter(torch.tensor(ew, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n))
        self.input_gain = nn.Parameter(torch.randn(n, DIM) * 0.05)
        self.head = nn.Linear(n - 2, 1)

    def forward_full(self, x):
        b = x.shape[0]
        h = torch.zeros(b, self.n)
        h[:, 0] = x[:, 0]
        h[:, 1] = x[:, 1]
        term = x @ self.input_gain.T
        for _ in range(K):
            msg = torch.zeros(b, self.n)
            msg.index_add_(1, self.dst, self.w.unsqueeze(0) * h[:, self.src])
            h = torch.tanh(self.bias.unsqueeze(0) + term + msg)
        return h

    def forward(self, x):
        return self.forward_full(x)[:, 2:]


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model.head(model(X))
    return float(torch.mean(torch.abs(pred - y)).item())


def train_inner(model, X, y, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pred = model.head(model(X[idx]))
            loss = torch.mean((pred - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        print("  epoch", epoch, "train_mse", round(total / n, 5), flush=True)
    return model


def grow(model, X_val, y_val, errors):
    old_n = model.n
    old_output = model.output
    new_n = old_n + GROW
    new_output = new_n - 1
    # Build a no-grad activation for the worst validation samples.
    with torch.no_grad():
        h = model.forward_full(X_val)
    worst = torch.argsort(errors, descending=True)[:max(1, GROW // 2)]
    new_src, new_dst, new_w = [], [], []
    existing = set(zip(model.src.tolist(), model.dst.tolist()))

    for j in range(old_n, new_n - 1):
        sample = int(worst[(j - old_n) % len(worst)])
        act = h[sample].abs()
        top = torch.topk(act[:old_n], k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                new_src.append(u); new_dst.append(j); new_w.append(0.02)
                existing.add((u, j))
        # every new node projects to the new output
        new_src.append(j); new_dst.append(new_output); new_w.append(0.05)
        existing.add((j, new_output))
    # old output becomes a hidden relay to the new output
    new_src.append(old_output); new_dst.append(new_output); new_w.append(0.05)
    existing.add((old_output, new_output))

    eu = torch.cat([model.src, torch.tensor(new_src, dtype=torch.long)])
    ev = torch.cat([model.dst, torch.tensor(new_dst, dtype=torch.long)])
    ew = torch.cat([model.w.detach(), torch.tensor(new_w, dtype=torch.float32)])

    new = Brain(new_n, eu.numpy(), ev.numpy(), ew.numpy()).to(X_val.device)
    new.bias.data[:old_n] = model.bias.data
    new.input_gain.data[:old_n] = model.input_gain.data
    new.bias.data[old_n:] = torch.randn(GROW) * 0.01
    new.input_gain.data[old_n:] = torch.randn(GROW, DIM) * 0.02
    new.w.data[:len(model.w)] = model.w.data
    new.w.data[len(model.w):] = torch.tensor(new_w) * 0.5
    # preserve old readout, new columns zero
    new.head.weight.data[:, :old_n - 2] = model.head.weight.data
    new.head.weight.data[:, old_n - 2:] = 0.0
    new.head.bias.data[:] = model.head.bias.data
    print("grew", old_n, "->", new_n, "edges", len(eu), flush=True)
    return new


def main() -> None:
    t0 = time.perf_counter()
    math_x, math_y = task_sum_sq(3000)
    math_x = np.array([np.pad(x, (0, DIM - len(x))) for x in math_x], dtype=np.float32)
    X = torch.tensor(math_x, dtype=torch.float32)
    Y_raw = torch.tensor(math_y, dtype=torch.float32).view(-1, 1)
    y_mean = float(Y_raw.mean())
    y_std = float(Y_raw.std() + 1e-6)
    Y = (Y_raw - y_mean) / y_std
    Xtr, Ytr = X[:2000], Y[:2000]
    Xval, Yval = X[2000:2500], Y[2000:2500]
    Xte, Yte = X[2500:], Y[2500:]

    _, incoming, edges, eu, ev, ew = build_graph()
    model = Brain(2000, eu, ev, ew)
    best = float("inf")
    best_test = float("inf")
    best_model = None
    no_improve = 0
    history = []
    early_stopped = None

    for rnd in range(ROUNDS + 1):
        print("ROUND", rnd, "nodes", model.n, "edges", len(model.w), flush=True)
        model = train_inner(model, Xtr, Ytr)
        val = evaluate(model, Xval, Yval)
        test = evaluate(model, Xte, Yte)
        improved = val < best
        history.append({"round": rnd, "nodes": model.n, "edges": int(len(model.w)), "val_mae": round(val * y_std, 5), "test_mae": round(test * y_std, 5), "best": improved})
        print("  VAL", round(val * y_std, 5), "TEST", round(test * y_std, 5), "BEST" if improved else "", flush=True)
        if improved:
            best = val
            best_test = test
            best_model = copy.deepcopy(model).cpu()
            no_improve = 0
            print("  ACCEPT best", round(best * y_std, 5), flush=True)
        else:
            no_improve += 1
            print("  NO IMPROVEMENT", no_improve, flush=True)
        if rnd == ROUNDS:
            break
        if no_improve >= PATIENCE:
            early_stopped = rnd
            print("  EARLY STOP at round", rnd, flush=True)
            break
        with torch.no_grad():
            pred = model.head(model(Xval))
            errs = torch.abs(pred - Yval).view(-1)
        model = grow(model, Xval, Yval, errs)

    final_test = evaluate(best_model, Xte, Yte) if best_model is not None else float("inf")
    payload = {
        "mode": "grow -> backprop train -> validate -> grow, best checkpoint",
        "K_steps": K, "grow_per_round": GROW, "rounds": ROUNDS, "patience": PATIENCE,
        "history": history,
        "best_val_mae": round(best * y_std, 5),
        "best_test_mae": round(final_test * y_std, 5),
        "early_stopped_round": early_stopped,
        "seconds": round(time.perf_counter() - t0, 3),
        "seed": SEED, "output_files": {"result_json": f"research_mvp/grow_train_loop_v3_result_seed{SEED}.json", "report": "research_mvp/GROW_TRAIN_LOOP_V3_REPORT.md"},
    }
    out = ROOT / f"grow_train_loop_v3_result_seed{SEED}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", out)


if __name__ == "__main__":
    main()
