#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual PPO with a 2000-node graph actor-critic + sim-to-sim evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from graph_policy import build_graph
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
OBS_DIM = 12
ACT_DIM = 2
K = 6
UPDATES = 20
ROLLOUT = 1024
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 3e-4


class VisionGraphAC(nn.Module):
    def __init__(self, n=2000):
        super().__init__()
        _, _, edges = build_graph(n)
        self.n = n
        self.register_buffer("src", torch.tensor([e[0] for e in edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([e[1] for e in edges], dtype=torch.long))
        self.w = nn.Parameter(torch.tensor([e[2] for e in edges], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n))
        self.input_gain = nn.Parameter(torch.randn(n, 64) * 0.05)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(n - 2 + OBS_DIM, ACT_DIM)
        self.critic = nn.Linear(n - 2 + OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

    def graph_features(self, image, state):
        latent = self.encoder(image)
        b = latent.shape[0]
        term = latent @ self.input_gain.T
        h = torch.zeros(b, self.n, device=image.device)
        h[:, :64] = latent
        for _ in range(K):
            msg = torch.zeros(b, self.n, device=image.device)
            msg.index_add_(1, self.dst, self.w.unsqueeze(0) * h[:, self.src])
            h = torch.tanh(self.bias.unsqueeze(0) + term + msg)
        return h

    def forward(self, image, state):
        h = self.graph_features(image, state)
        feat = torch.cat([h[:, 2:], state], dim=1)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, image, state):
        mean, value = self.forward(image, state)
        std = torch.exp(self.log_std).clamp(0.02, 1.0)
        dist = Normal(mean, std)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        return torch.tanh(raw), raw, logp, value


def prep(img, state):
    x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return x, s


def evaluate(model, domain, episodes=10, seed=5000):
    env = VisionPointMassEnv(seed=seed, domain=domain)
    succ = 0; steps = []
    for _ in range(episodes):
        img, state = env.reset()
        while True:
            with torch.no_grad():
                action, _, _, _ = model.act(*prep(img, state))
            r = env.step(action[0].numpy())
            img, state = r.image, r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def main() -> None:
    torch.manual_seed(5001)
    model = VisionGraphAC(2000)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = VisionPointMassEnv(seed=5002, domain="source")
    img, state = env.reset()
    t0 = time.perf_counter()

    for update in range(UPDATES):
        I, S, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(ROLLOUT):
            it, st = prep(img, state)
            with torch.no_grad():
                action, raw, logp, value = model.act(it, st)
            r = env.step(action[0].numpy())
            I.append(it[0]); S.append(st[0]); A.append(raw[0]); LP.append(logp[0]);
            R.append(r.reward); D.append(float(r.done)); V.append(value[0])
            img, state = r.image, r.state
            if r.done:
                img, state = env.reset()
        I = torch.stack(I); S = torch.stack(S); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.tensor(R); D = torch.tensor(D); V = torch.stack(V)
        with torch.no_grad():
            _, nv = model(*prep(img, state)); nv = nv[0]
        adv = torch.zeros_like(R); last = 0.0
        for t in reversed(range(len(R))):
            nonterm = 1.0 - D[t]
            nextv = nv if t == len(R) - 1 else V[t + 1]
            delta = R[t] + GAMMA * nextv * nonterm - V[t]
            last = delta + GAMMA * LAM * nonterm * last
            adv[t] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(R))
            for i in range(0, len(R), MINIBATCH):
                idx = perm[i:i + MINIBATCH]
                mean, value = model(I[idx], S[idx])
                std = torch.exp(model.log_std).clamp(0.02, 1.0)
                dist = Normal(mean, std)
                logp = dist.log_prob(A[idx]).sum(-1)
                ratio = torch.exp(logp - LP[idx])
                surr = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[idx])
                loss = -surr.mean() + 0.5 * torch.mean((value - ret[idx]) ** 2) - 0.01 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        if update % 5 == 0:
            s_src, _ = evaluate(model, "source", 5, 5100 + update)
            s_tgt, _ = evaluate(model, "target", 5, 5200 + update)
            print("update", update, "source", s_src, "target", s_tgt, flush=True)

    src, src_steps = evaluate(model, "source", 20, 5300)
    tgt, tgt_steps = evaluate(model, "target", 20, 5400)
    print("FINAL source", src, src_steps, "target", tgt, tgt_steps, flush=True)
    payload = {
        "graph_nodes": 2000, "K": K, "updates": UPDATES,
        "source_success": src, "source_steps": src_steps,
        "target_success": tgt, "target_steps": tgt_steps,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "vision_graph_ppo_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
