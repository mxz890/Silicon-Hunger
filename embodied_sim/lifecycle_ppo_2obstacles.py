#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle v2 -> PPO fine-tune with 3 randomized obstacles.

Uses the same 500/550-node lifecycle policy, then adds two more obstacles
(3 total), stronger target randomization, and PPO fine-tuning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from lifecycle_embodied_v2 import (
    LifeGraphPolicy, collect_stage1, train_epoch, grow, prune,
    CYCLES, GROW, TIME_UNIT, TAU,
)
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
N_ENVS = 4
UPDATES = 20
STEPS = 128
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 1e-4


class PolicyAC(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.critic = nn.Linear(policy.n - 2, 1)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        h = self.policy.forward_full(obs)
        feat = h[:, 2:]
        return torch.tanh(self.policy.head(feat)), self.critic(feat).squeeze(-1)

    def act(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).clamp(0.02, 1.0)
        dist = Normal(mean, std)
        raw = dist.sample()
        return torch.tanh(raw), raw, dist.log_prob(raw).sum(-1), value


class VecState:
    def __init__(self, n_envs, seed, **kwargs):
        self.envs = [VisionPointMassEnv(seed=seed + i, render=False, **kwargs) for i in range(n_envs)]
        self.n = n_envs

    def reset(self):
        return [e.reset()[1] for e in self.envs]

    def step(self, actions):
        states, rewards, dones, successes = [], [], [], []
        for e, a in zip(self.envs, actions):
            r = e.step(a)
            if r.done:
                state = e.reset()[1]
            else:
                state = r.state
            states.append(state); rewards.append(r.reward); dones.append(r.done); successes.append(r.success)
        return states, rewards, dones, successes


def evaluate(model, n_obstacles, target_range, start_range, episodes=20, seed=9200):
    env = VisionPointMassEnv(seed=seed, render=False, size=8, n_obstacles=n_obstacles, target_range=target_range, start_range=start_range, obstacle_range=1.0)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a, _, _, _ = model.act(torch.tensor(state, dtype=torch.float32).view(1, -1))
            r = env.step(a[0].numpy()); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(9200); np.random.seed(9200)

    # 1. Lifecycle v2: one fixed obstacle
    X, Y = collect_stage1(episodes=80, seed=9202)
    Xtr, Ytr = torch.tensor(X), torch.tensor(Y)
    policy = LifeGraphPolicy(500, obs_dim=12)
    now = 0.0
    for cyc in range(CYCLES):
        policy = train_epoch(policy, Xtr, Ytr)
        policy = grow(policy, Xtr, Ytr, now)
        policy = train_epoch(policy, Xtr, Ytr)
        sec = 0.1
        now += sec / TIME_UNIT
        remove, usage, retention = prune(policy, Xtr[:256], now)
        print("lifecycle", cyc, "alive", int(policy.alive.sum()), "pruned", len(remove), flush=True)

    model = PolicyAC(policy)
    base1, base1_steps = evaluate(model, 1, 0.8, 0.6, seed=9300)
    two_before, two_before_steps = evaluate(model, 2, 1.0, 0.8, seed=9350)
    print("lifecycle stage1", base1, base1_steps, "two before", two_before, two_before_steps, flush=True)

    # 2. PPO with 2 obstacles + target randomization
    venv = VecState(N_ENVS, seed=9400, size=8, n_obstacles=2, target_range=1.0, start_range=0.8, obstacle_range=1.0)
    states = venv.reset()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for upd in range(UPDATES):
        buffers = [{"S": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(N_ENVS)]
        for _ in range(STEPS):
            acts = []
            for i in range(N_ENVS):
                s = torch.tensor(states[i], dtype=torch.float32).view(1, -1)
                with torch.no_grad():
                    a, raw, logp, value = model.act(s)
                buffers[i]["S"].append(s[0]); buffers[i]["A"].append(raw[0]); buffers[i]["LP"].append(logp[0]); buffers[i]["V"].append(value[0])
                acts.append(a[0].numpy())
            states, rewards, dones, _ = venv.step(acts)
            for i in range(N_ENVS):
                buffers[i]["R"].append(rewards[i]); buffers[i]["D"].append(float(dones[i]))

        all_S, all_A, all_LP, all_R, all_V, all_adv, all_ret = [], [], [], [], [], [], []
        for i in range(N_ENVS):
            b = buffers[i]
            R = torch.tensor(b["R"]); D = torch.tensor(b["D"]); V = torch.stack(b["V"])
            with torch.no_grad():
                _, nv = model(torch.tensor(states[i], dtype=torch.float32).view(1, -1)); nv = nv[0]
            adv = torch.zeros_like(R); last = 0.0
            for t in reversed(range(len(R))):
                nonterm = 1.0 - D[t]
                nextv = nv if t == len(R) - 1 else V[t + 1]
                delta = R[t] + GAMMA * nextv * nonterm - V[t]
                last = delta + GAMMA * LAM * nonterm * last
                adv[t] = last
            all_S.extend(b["S"]); all_A.extend(b["A"]); all_LP.extend(b["LP"])
            all_R.append(R); all_V.append(V); all_adv.append(adv); all_ret.append(adv + V)
        all_S = torch.stack(all_S); all_A = torch.stack(all_A); all_LP = torch.stack(all_LP)
        all_R = torch.cat(all_R); all_adv = torch.cat(all_adv); all_ret = torch.cat(all_ret)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(all_R))
            for j in range(0, len(all_R), MINIBATCH):
                idx = perm[j:j + MINIBATCH]
                mean, value = model(all_S[idx])
                std = torch.exp(model.log_std).clamp(0.02, 1.0)
                dist = Normal(mean, std)
                logp = dist.log_prob(all_A[idx]).sum(-1)
                ratio = torch.exp(logp - all_LP[idx])
                surr = torch.min(ratio * all_adv[idx], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * all_adv[idx])
                loss = -surr.mean() + 0.5 * torch.mean((value - all_ret[idx]) ** 2) - 0.01 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        if upd % 5 == 0:
            s2, _ = evaluate(model, 2, 1.0, 0.8, episodes=10, seed=9500 + upd)
            print("PPO", upd, "2obs", s2, flush=True)

    s2, st2 = evaluate(model, 2, 1.0, 0.8, episodes=20, seed=9600)
    s_target, st_target = evaluate(model, 2, 1.2, 1.0, episodes=20, seed=9700)
    payload = {
        "lifecycle_nodes": policy.n, "alive": int(policy.alive.sum()),
        "stage1_one_obstacle": base1, "stage1_steps": base1_steps,
        "two_obstacles_before": two_before, "two_obstacles_before_steps": two_before_steps,
        "two_obstacles_after": s2, "two_obstacles_after_steps": st2,
        "target_success": s_target, "target_steps": st_target,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "lifecycle_ppo_2obstacles_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
