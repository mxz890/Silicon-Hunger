#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BC 90% checkpoint -> PPO fine-tune with KL/BC anchor.

Deterministic evaluation.  No PPO from scratch.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from continual_learning import load_checkpoint
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
OBS_DIM = 12
ACT_DIM = 2
N_ENVS = 8
UPDATES = 20
STEPS = 256
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 1e-4
KL_START = 0.20
KL_END = 0.02
BC_COEF = 0.30
PATIENCE = 2
VAL_EPISODES = 30


class PolicyAC(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.critic = nn.Linear(policy.n - 2, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.8))

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
    def __init__(self, cfg, seed):
        self.envs = [VisionPointMassEnv(seed=seed + i, render=False, size=8, **cfg) for i in range(N_ENVS)]

    def reset(self):
        return [e.reset()[1] for e in self.envs]

    def step(self, actions):
        states, rewards, dones = [], [], []
        for e, a in zip(self.envs, actions):
            r = e.step(a)
            states.append(e.reset()[1] if r.done else r.state)
            rewards.append(r.reward); dones.append(r.done)
        return states, rewards, dones


def deterministic_action(model, state):
    with torch.no_grad():
        mean, _ = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
    return torch.tanh(mean)[0].numpy()


def evaluate(model, cfg, episodes=30, seed=18000):
    env = VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            a = deterministic_action(model, state)
            r = env.step(a); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def bc_mean(bc_policy, obs):
    h = bc_policy.forward_full(obs)
    return torch.tanh(bc_policy.head(h[:, 2:]))


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(18000); np.random.seed(18000)
    cfg = dict(n_obstacles=1, target_range=0.0, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))

    policy = load_checkpoint(ROOT / "mzx_stage2_two_obstacles.pt")
    model = PolicyAC(policy)
    bc_policy = copy.deepcopy(policy).eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)

    bc_s, bc_steps = evaluate(model, cfg, seed=18100)
    print("BC deterministic", bc_s, bc_steps, flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    venv = VecState(cfg, seed=18200)
    states = venv.reset()
    replay_s = torch.zeros((0, OBS_DIM))
    best_success = bc_s
    best_update = -1
    best_state = copy.deepcopy(model.state_dict())
    drops = 0

    for upd in range(UPDATES):
        kl_coef = KL_END + (KL_START - KL_END) * max(0.0, 1.0 - upd / max(1, UPDATES - 1))
        bufs = [{"S": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(N_ENVS)]
        for _ in range(STEPS):
            acts = []
            for i in range(N_ENVS):
                s = torch.tensor(states[i], dtype=torch.float32).view(1, -1)
                with torch.no_grad():
                    a, raw, logp, value = model.act(s)
                bufs[i]["S"].append(s[0]); bufs[i]["A"].append(raw[0]); bufs[i]["LP"].append(logp[0]); bufs[i]["V"].append(value[0])
                acts.append(a[0].numpy())
            states, rews, dones = venv.step(acts)
            for i in range(N_ENVS):
                bufs[i]["R"].append(rews[i]); bufs[i]["D"].append(float(dones[i]))

        all_S, all_A, all_LP, all_R, all_V, all_adv, all_ret = [], [], [], [], [], [], []
        for i in range(N_ENVS):
            b = bufs[i]
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
        replay_s = torch.cat([replay_s, all_S.detach()])[-4000:]

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

                # KL/BC anchor to the BC checkpoint.
                ridx = torch.randint(0, len(replay_s), (64,))
                bcm = bc_mean(bc_policy, replay_s[ridx])
                newm, _ = model(replay_s[ridx])
                newstd = std
                kl = (torch.log(newstd / 0.5) + (0.25 + (bcm - newm) ** 2) / (2 * newstd ** 2) - 0.5).sum(-1).mean()
                bcloss = torch.mean((newm - bcm) ** 2)
                loss = loss + kl_coef * kl + BC_COEF * bcloss

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()

        if upd % 5 == 0:
            s, st = evaluate(model, cfg, episodes=VAL_EPISODES, seed=18300 + upd)
            if s > best_success:
                best_success = s
                best_update = upd
                best_state = copy.deepcopy(model.state_dict())
                drops = 0
            else:
                drops += 1
            print("PPO", upd, "det", s, st, "best", best_success, "drops", drops, flush=True)
            if drops >= PATIENCE:
                print("EARLY STOP", upd, flush=True)
                break

    last_s, last_steps = evaluate(model, cfg, seed=18400)
    if best_state is not None:
        model.load_state_dict(best_state)
    ppo_s, ppo_steps = evaluate(model, cfg, seed=18400)
    strong_cfg = dict(n_obstacles=2, target_range=1.2, start_range=1.0, obstacle_range=1.0)
    strong_s, strong_steps = evaluate(model, strong_cfg, seed=18500)
    payload = {
        "bc_success": bc_s, "bc_steps": bc_steps,
        "ppo_last_success": last_s, "ppo_last_steps": last_steps,
        "ppo_best_success": ppo_s, "ppo_best_steps": ppo_steps,
        "best_update": best_update,
        "strong_target_success": strong_s, "strong_target_steps": strong_steps,
        "updates": UPDATES, "kl_start": KL_START, "kl_end": KL_END, "bc_coef": BC_COEF,
        "n_envs": N_ENVS, "steps": STEPS, "target": "fixed (0.5,0.0)", "obstacles": 1,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "bc_to_ppo_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
