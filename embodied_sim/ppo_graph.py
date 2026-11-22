#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO with a 500-node graph actor-critic on point-mass + obstacles.

No PD expert.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from point_mass_obstacles import PointMassObstacleEnv
from graph_policy import build_graph

ROOT = Path(__file__).resolve().parent
OBS_DIM = 12
ACT_DIM = 2
K = 6
UPDATES = 40
ROLLOUT = 2048
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCH = 64
LR = 3e-4
ENT_COEF = 0.01


class GraphAC(nn.Module):
    def __init__(self, n=500):
        super().__init__()
        _, _, edges = build_graph(n)
        self.n = n
        self.in_dim = 64
        self.register_buffer("src", torch.tensor([e[0] for e in edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([e[1] for e in edges], dtype=torch.long))
        self.w = nn.Parameter(torch.tensor([e[2] for e in edges], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n))
        self.input_gain = nn.Parameter(torch.randn(n, self.in_dim) * 0.05)
        self.actor = nn.Linear(n - 2, ACT_DIM)
        self.critic = nn.Linear(n - 2, 1)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

    def forward(self, obs):
        b = obs.shape[0]
        x = torch.zeros(b, self.in_dim, device=obs.device)
        x[:, :OBS_DIM] = obs
        h = torch.zeros(b, self.n, device=obs.device)
        h[:, :OBS_DIM] = obs
        term = x @ self.input_gain.T
        for _ in range(K):
            msg = torch.zeros(b, self.n, device=obs.device)
            msg.index_add_(1, self.dst, self.w.unsqueeze(0) * h[:, self.src])
            h = torch.tanh(self.bias.unsqueeze(0) + term + msg)
        feat = h[:, 2:]
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).clamp(0.02, 1.0)
        dist = Normal(mean, std)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw)
        return action, raw, logp, value


def ppo_update(model, opt, obs, raw, logp_old, ret, adv):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, MINIBATCH):
            idx = perm[i:i + MINIBATCH]
            mean, value = model(obs[idx])
            std = torch.exp(model.log_std).clamp(0.02, 1.0)
            dist = Normal(mean, std)
            logp = dist.log_prob(raw[idx]).sum(-1)
            ratio = torch.exp(logp - logp_old[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.mean((value - ret[idx]) ** 2)
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


def evaluate(model, episodes=30, seed=4000):
    env = PointMassObstacleEnv(seed=seed)
    success = 0
    steps = []
    paths = []
    collisions = 0
    for ep in range(episodes):
        obs = env.reset(); path = [obs[:2].copy()]
        while True:
            with torch.no_grad():
                action, _, _, _ = model.act(torch.tensor(obs, dtype=torch.float32).view(1, -1))
            r = env.step(action[0].numpy())
            obs = r.obs; path.append(obs[:2].copy()); collisions += int(r.collision)
            if r.done:
                success += int(r.success); steps.append(env.steps)
                if ep < 4: paths.append((np.array(path), obs[4:6].copy()))
                break
    return success / episodes, float(np.mean(steps)), collisions, paths


def main() -> None:
    torch.manual_seed(4001)
    model = GraphAC(500)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = PointMassObstacleEnv(seed=4002)
    obs = env.reset()
    history = []
    t0 = time.perf_counter()

    for update in range(UPDATES):
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            o = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                action, raw, logp, value = model.act(o)
            r = env.step(action[0].numpy())
            O.append(o[0]); A.append(raw[0]); LP.append(logp[0]); R.append(r.reward); D.append(float(r.done)); V.append(value[0])
            obs = r.obs
            if r.done:
                obs = env.reset()
        O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.tensor(R, dtype=torch.float32); D = torch.tensor(D)
        V = torch.stack(V)
        with torch.no_grad():
            _, next_v = model(torch.tensor(obs, dtype=torch.float32).view(1, -1))
            next_v = next_v[0]
        adv = torch.zeros_like(R); last = 0.0
        for t in reversed(range(len(R))):
            nonterminal = 1.0 - D[t]
            nv = next_v if t == len(R) - 1 else V[t + 1]
            delta = R[t] + GAMMA * nv * nonterminal - V[t]
            last = delta + GAMMA * LAM * nonterminal * last
            adv[t] = last
        ret = adv + V
        ppo_update(model, opt, O, A, LP, ret, adv)
        if update % 5 == 0:
            succ, steps, col, _ = evaluate(model, episodes=10, seed=4100 + update)
            print("update", update, "eval_success", succ, "steps", steps, "collisions", col, flush=True)
        history.append({"update": update, "reward": round(float(R.mean()), 4)})

    succ, steps, col, paths = evaluate(model, episodes=50, seed=4200)
    print("FINAL", succ, steps, col, flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for path, target in paths:
        ax.plot(path[:, 0], path[:, 1])
        ax.scatter(*target, marker="*", s=140)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_title("PPO graph policy — obstacles")
    fig.tight_layout(); fig.savefig(ROOT / "ppo_graph_obstacles.png", dpi=160)

    payload = {"graph_nodes": 500, "K": K, "obs_dim": OBS_DIM, "updates": UPDATES, "final_success": succ, "final_steps": steps, "collisions": col, "seconds": round(time.perf_counter() - t0, 3), "history": history}
    (ROOT / "ppo_graph_obstacles_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
