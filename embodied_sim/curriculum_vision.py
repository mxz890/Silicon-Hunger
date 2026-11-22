#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curriculum: no obstacles/small range -> PD vision BC -> PPO with 2 obstacles.

Includes a small vectorized environment wrapper (4 envs).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from point_mass_env import pd_action
from vision_env import VisionPointMassEnv
from vision_graph_ppo import VisionGraphAC, prep

ROOT = Path(__file__).resolve().parent
OBS_DIM = 12
ACT_DIM = 2
N_ENVS = 4
UPDATES = 12
ROLLOUT_PER_ENV = 64
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 2e-4


class VecVisionEnv:
    def __init__(self, n_envs=4, **kwargs):
        self.envs = [VisionPointMassEnv(seed=7000 + i, **kwargs) for i in range(n_envs)]
        self.n = n_envs

    def reset(self):
        outs = [e.reset() for e in self.envs]
        return [o[0] for o in outs], [o[1] for o in outs]

    def step(self, actions):
        imgs, states, rewards, dones, successes = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            r = e.step(a)
            if r.done:
                img, state = e.reset()
            else:
                img, state = r.image, r.state
            imgs.append(img); states.append(state)
            rewards.append(r.reward); dones.append(r.done); successes.append(r.success)
        return imgs, states, rewards, dones, successes


def eval_policy(model, n_obstacles=0, target_range=0.6, start_range=0.5, domain="source", episodes=20, seed=7100):
    env = VisionPointMassEnv(seed=seed, size=32, n_obstacles=n_obstacles, target_range=target_range, start_range=start_range, domain=domain)
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


def collect_pd(episodes=40):
    venv = VecVisionEnv(N_ENVS, size=32, n_obstacles=0, target_range=0.6, start_range=0.5)
    imgs, states = venv.reset()
    data = []
    total = 0
    while total < episodes * 80:
        actions = [pd_action(s[:6]) for s in states]
        next_imgs, next_states, rewards, dones, _ = venv.step(actions)
        for img, state, a in zip(imgs, states, actions):
            data.append((img, state, a))
        total += len(imgs)
        imgs, states = next_imgs, next_states
    return data


def train_bc(model, data, epochs=8):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = np.random.permutation(len(data))
        total = 0.0
        for i in range(0, len(data), 64):
            batch = [data[j] for j in perm[i:i + 64]]
            imgs = torch.stack([prep(b[0], b[1])[0][0] for b in batch])
            states = torch.stack([prep(b[0], b[1])[1][0] for b in batch])
            targets = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            mean, _ = model(imgs, states)
            loss = torch.mean((mean - targets) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += float(loss.item()) * len(batch)
        print("BC epoch", ep, "mse", round(total / len(data), 6), flush=True)
    return model


def ppo_finetune(model):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    venv = VecVisionEnv(N_ENVS, size=32, n_obstacles=2, target_range=0.8, start_range=0.6)
    imgs, states = venv.reset()
    for upd in range(UPDATES):
        I, S, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(ROLLOUT_PER_ENV):
            acts = []
            for i in range(N_ENVS):
                it, st = prep(imgs[i], states[i])
                with torch.no_grad():
                    action, raw, logp, value = model.act(it, st)
                I.append(it[0]); S.append(st[0]); A.append(raw[0]); LP.append(logp[0]); V.append(value[0])
                acts.append(action[0].numpy())
            imgs, states, rewards, dones, _ = venv.step(acts)
            R.extend(rewards); D.extend([float(x) for x in dones])
        I = torch.stack(I); S = torch.stack(S); A = torch.stack(A); LP = torch.stack(LP)
        R = torch.tensor(R); D = torch.tensor(D); V = torch.stack(V)
        adv = torch.zeros_like(R); last = 0.0
        for t in reversed(range(len(R))):
            nonterm = 1.0 - D[t]
            nextv = 0.0 if t == len(R) - 1 else V[t + 1]
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
        if upd % 3 == 0:
            s0, _ = eval_policy(model, n_obstacles=0, target_range=0.6, start_range=0.5, episodes=5)
            s1, _ = eval_policy(model, n_obstacles=2, target_range=0.8, start_range=0.6, episodes=5)
            print("PPO", upd, "stage0", s0, "stage1", s1, flush=True)


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(7200)
    np.random.seed(7200)
    model = VisionGraphAC(2000)
    data = collect_pd(episodes=40)
    print("demo samples", len(data), flush=True)
    train_bc(model, data, epochs=8)
    s0, st0 = eval_policy(model, n_obstacles=0, target_range=0.6, start_range=0.5, episodes=20, seed=7300)
    print("BC stage0", s0, st0, flush=True)
    ppo_finetune(model)
    s1, st1 = eval_policy(model, n_obstacles=2, target_range=0.8, start_range=0.6, episodes=20, seed=7400)
    s2, st2 = eval_policy(model, n_obstacles=2, target_range=1.0, start_range=0.7, domain="target", episodes=20, seed=7500)
    payload = {
        "graph_nodes": 2000, "image_size": 32, "n_envs": N_ENVS,
        "demo_samples": len(data), "stage0_success": s0, "stage1_success": s1, "target_success": s2,
        "stage1_steps": st1, "target_steps": st2,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "curriculum_vision_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
