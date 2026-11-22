#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable MZX lifecycle: grow, train, prune, rollback."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from lifecycle_embodied_v2 import LifeGraphPolicy

DIM = 64
TIME_UNIT = 0.1
TAU = 8.0
USAGE_EPS = 0.02
RETENTION_MIN = 0.50


def train_epoch(model, X, Y, lr=1e-3, epochs=2, batch=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            pred = model(X[idx])
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def grow(model, X, Y, now, grow=20):
    old_n = model.n; old_out = old_n - 1; new_n = old_n + grow; new_out = new_n - 1
    with torch.no_grad():
        h = model.forward_full(X)
        err = (model.head(h[:, 2:]) - Y).abs().view(-1)
    worst = torch.argsort(err, descending=True)[:max(1, grow // 2)]
    src = model.src.tolist(); dst = model.dst.tolist(); ws = model.w.detach().tolist()
    existing = set(zip(src, dst))
    new_ids = list(range(old_n, new_n - 1))
    for j in new_ids:
        s = int(worst[(j - old_n) % len(worst)]) % h.shape[0]
        top = torch.topk(h[s, :old_n].abs(), k=4).indices.tolist()
        for u in top:
            if (u, j) not in existing:
                src.append(int(u)); dst.append(j); ws.append(0.02); existing.add((u, j))
        src.append(j); dst.append(new_out); ws.append(0.05)
    src.append(old_out); dst.append(new_out); ws.append(0.05)

    new = LifeGraphPolicy(new_n, obs_dim=model.obs_dim)
    new.src = torch.tensor(src, dtype=torch.long)
    new.dst = torch.tensor(dst, dtype=torch.long)
    new.w = nn.Parameter(torch.tensor(ws, dtype=torch.float32))
    new.bias = nn.Parameter(torch.cat([model.bias.detach(), torch.zeros(grow)]))
    new.input_gain = nn.Parameter(torch.cat([model.input_gain.detach(), torch.randn(grow, DIM) * 0.02]))
    new.alive.data[:old_n] = model.alive.data
    new.alive.data[old_n:] = 1.0
    new.last_used = torch.cat([model.last_used, torch.full((grow,), float(now))])
    old_feat = old_n - 2; new_feat = new_n - 2
    new.head = nn.Linear(new_feat, 2)
    with torch.no_grad():
        new.head.weight[:, :old_feat] = model.head.weight
        new.head.weight[:, old_feat:] = 0.0
        new.head.bias[:] = model.head.bias
    return new, new_ids


def prune(model, X, now):
    model.eval()
    with torch.no_grad():
        h = model.forward_full(X)
    usage = h.abs().mean(0)
    retention = torch.exp(-(now - model.last_used) / TAU)
    removed = []
    for i in range(2, model.n - 1):
        if model.alive[i] > 0 and usage[i] < USAGE_EPS and retention[i] < RETENTION_MIN:
            model.alive[i] = 0.0; removed.append(i)
    model.last_used[usage > USAGE_EPS] = now
    return removed, usage, retention


def snapshot(model):
    return copy.deepcopy(model)


def train_new_nodes(model, X, Y, new_ids, epochs=2, lr=1e-3, batch=64):
    """Train only new node bias/input_gain, new edges, and new head columns."""
    n = model.n
    new_mask = torch.zeros(n, dtype=torch.bool)
    new_mask[new_ids] = True
    edge_mask = new_mask[model.dst] | new_mask[model.src]
    feat_mask = torch.zeros(n - 2, dtype=torch.bool)
    feat_mask[torch.tensor(new_ids) - 2] = True
    params = [model.bias, model.input_gain, model.w, model.head.weight, model.head.bias]
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            pred = model(X[idx])
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                model.bias.grad[~new_mask] = 0
                model.input_gain.grad[~new_mask] = 0
                model.w.grad[~edge_mask] = 0
                model.head.weight.grad[:, ~feat_mask] = 0
            opt.step()
    return model
