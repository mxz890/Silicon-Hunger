#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A small differentiable graph policy for the MuJoCo point-mass task."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from point_mass_env import PointMassEnv


def build_graph(n: int = 500, seed: int = 3001):
    rng = np.random.default_rng(seed)
    pos = np.empty((n, 3), dtype=np.float32)
    pos[0] = [50, 50, 50]
    for i in range(1, n):
        pos[i] = np.clip(pos[i - 1] + rng.normal(0, 2.0, 3), 0, 100)
    incoming = [[] for _ in range(n)]
    edges = []
    output = n - 1
    for j in range(2, output):
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
    for j in range(max(2, output - 80), output):
        w = float(rng.uniform(0.2, 0.7))
        incoming[output].append((j, w))
        edges.append((j, output, w))
    return pos, incoming, edges


class GraphPolicy(nn.Module):
    def __init__(self, n: int = 500, obs_dim: int = 6, act_dim: int = 2):
        super().__init__()
        pos, incoming, edges = build_graph(n)
        self.n = n
        self.in_dim = 64
        self.register_buffer("src", torch.tensor([e[0] for e in edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([e[1] for e in edges], dtype=torch.long))
        self.w = nn.Parameter(torch.tensor([e[2] for e in edges], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n))
        self.input_gain = nn.Parameter(torch.randn(n, self.in_dim) * 0.05)
        self.head = nn.Linear(n - 2, act_dim)
        self.obs_dim = obs_dim
        self.k = 6

    def forward(self, obs):
        b = obs.shape[0]
        x = torch.zeros(b, self.in_dim, device=obs.device)
        x[:, :self.obs_dim] = obs
        h = torch.zeros(b, self.n, device=obs.device)
        h[:, :self.obs_dim] = obs
        term = x @ self.input_gain.T
        for _ in range(self.k):
            msg = torch.zeros(b, self.n, device=obs.device)
            msg.index_add_(1, self.dst, self.w.unsqueeze(0) * h[:, self.src])
            h = torch.tanh(self.bias.unsqueeze(0) + term + msg)
        return torch.tanh(self.head(h[:, 2:]))


def collect_pd_data(episodes=80, seed=3002):
    env = PointMassEnv(seed=seed)
    X, Y = [], []
    for _ in range(episodes):
        obs = env.reset()
        while True:
            from point_mass_env import pd_action
            a = pd_action(obs)
            X.append(obs.copy()); Y.append(a.copy())
            r = env.step(a)
            obs = r.obs
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), env


def train(policy, X, Y, epochs=12):
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor(X); Y = torch.tensor(Y)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            pred = policy(X[idx])
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        print("epoch", epoch, "mse", round(total / len(X), 6), flush=True)
    return policy


def evaluate(policy, episodes=50, seed=3003):
    env = PointMassEnv(seed=seed)
    success = 0
    steps = []
    paths = []
    for ep in range(episodes):
        obs = env.reset(); path = [obs[:2].copy()]
        while True:
            with torch.no_grad():
                a = policy(torch.tensor(obs, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a)
            obs = r.obs; path.append(obs[:2].copy())
            if r.done:
                success += int(r.success); steps.append(env.steps)
                if ep < 4: paths.append((np.array(path), obs[4:6].copy()))
                break
    return success / episodes, float(np.mean(steps)), paths
