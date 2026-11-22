#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curriculum v2: 20k vision demos -> BC 30 epochs -> per-env GAE PPO.

Stage 0: no obstacle, small range, 64x64 vision.
Stage 1: one fixed obstacle at the origin.
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
N_ENVS = 4
BC_SAMPLES = 20000
BC_EPOCHS = 30
PPO_UPDATES = 10
PPO_STEPS_PER_ENV = 128
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 2e-4


class VecVisionEnv:
    def __init__(self, n_envs, **kwargs):
        self.envs = [VisionPointMassEnv(seed=8000 + i, **kwargs) for i in range(n_envs)]
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


def eval_policy(model, n_obstacles=0, target_range=0.4, start_range=0.3, domain="source", episodes=30, seed=8100):
    env = VisionPointMassEnv(seed=seed, size=64, n_obstacles=n_obstacles, target_range=target_range, start_range=start_range, obstacle_range=0.0, domain=domain)
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


def collect_demos():
    venv = VecVisionEnv(N_ENVS, size=64, n_obstacles=0, target_range=0.4, start_range=0.3, obstacle_range=0.0)
    imgs, states = venv.reset()
    data = []
    while len(data) < BC_SAMPLES:
        actions = [pd_action(s[:6]) for s in states]
        for img, state, a in zip(imgs, states, actions):
            data.append((img, state, a))
        imgs, states, _, _, _ = venv.step(actions)
    return data[:BC_SAMPLES]


def train_bc(model, data):
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for ep in range(BC_EPOCHS):
        perm = np.random.permutation(len(data))
        total = 0.0
        for i in range(0, len(data), 128):
            batch = [data[j] for j in perm[i:i + 128]]
            images = torch.stack([prep(b[0], b[1])[0][0] for b in batch])
            states = torch.stack([prep(b[0], b[1])[1][0] for b in batch])
            targets = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            mean, _ = model(images, states)
            loss = torch.mean((mean - targets) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += float(loss.item()) * len(batch)
        print("BC epoch", ep, "mse", round(total / len(data), 6), flush=True)
    return model


def ppo_finetune(model):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    venv = VecVisionEnv(N_ENVS, size=64, n_obstacles=1, target_range=0.8, start_range=0.6, obstacle_range=0.0)
    images, states = venv.reset()

    for upd in range(PPO_UPDATES):
        buffers = [{"I": [], "S": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(N_ENVS)]
        for _ in range(PPO_STEPS_PER_ENV):
            acts = []
            for i in range(N_ENVS):
                it, st = prep(images[i], states[i])
                with torch.no_grad():
                    action, raw, logp, value = model.act(it, st)
                buffers[i]["I"].append(it[0]); buffers[i]["S"].append(st[0])
                buffers[i]["A"].append(raw[0]); buffers[i]["LP"].append(logp[0]); buffers[i]["V"].append(value[0])
                acts.append(action[0].numpy())
            images, states, rewards, dones, _ = venv.step(acts)
            for i in range(N_ENVS):
                buffers[i]["R"].append(rewards[i]); buffers[i]["D"].append(float(dones[i]))

        all_I, all_S, all_A, all_LP, all_R, all_D, all_V = [], [], [], [], [], [], []
        for i in range(N_ENVS):
            b = buffers[i]
            R = torch.tensor(b["R"]); D = torch.tensor(b["D"]); V = torch.stack(b["V"])
            with torch.no_grad():
                _, nv = model(*prep(images[i], states[i])); nv = nv[0]
            adv = torch.zeros_like(R); last = 0.0
            for t in reversed(range(len(R))):
                nonterm = 1.0 - D[t]
                nextv = nv if t == len(R) - 1 else V[t + 1]
                delta = R[t] + GAMMA * nextv * nonterm - V[t]
                last = delta + GAMMA * LAM * nonterm * last
                adv[t] = last
            all_R.append(R); all_D.append(D); all_V.append(V); all_I.extend(b["I"]); all_S.extend(b["S"]); all_A.extend(b["A"]); all_LP.extend(b["LP"])
            # store per-env adv/ret temporarily
            b["adv"] = adv; b["ret"] = adv + V
        all_R = torch.cat(all_R); all_I = torch.stack(all_I); all_S = torch.stack(all_S)
        all_A = torch.stack(all_A); all_LP = torch.stack(all_LP)
        adv = torch.cat([b["adv"] for b in buffers]); ret = torch.cat([b["ret"] for b in buffers])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(all_R))
            for j in range(0, len(all_R), MINIBATCH):
                idx = perm[j:j + MINIBATCH]
                mean, value = model(all_I[idx], all_S[idx])
                std = torch.exp(model.log_std).clamp(0.02, 1.0)
                dist = Normal(mean, std)
                logp = dist.log_prob(all_A[idx]).sum(-1)
                ratio = torch.exp(logp - all_LP[idx])
                surr = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[idx])
                loss = -surr.mean() + 0.5 * torch.mean((value - ret[idx]) ** 2) - 0.01 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        if upd % 3 == 0:
            s1, _ = eval_policy(model, n_obstacles=1, target_range=0.8, start_range=0.6, episodes=10)
            print("PPO", upd, "stage1", s1, flush=True)


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(8200); np.random.seed(8200)
    model = VisionGraphAC(2000)
    data = collect_demos()
    print("demos", len(data), flush=True)
    train_bc(model, data)
    s0, st0 = eval_policy(model, n_obstacles=0, target_range=0.4, start_range=0.3, episodes=30, seed=8300)
    print("STAGE0", s0, st0, flush=True)
    ppo_finetune(model)
    s1, st1 = eval_policy(model, n_obstacles=1, target_range=0.8, start_range=0.6, episodes=30, seed=8400)
    s2, st2 = eval_policy(model, n_obstacles=1, target_range=1.0, start_range=0.7, domain="target", episodes=30, seed=8500)
    payload = {
        "graph_nodes": 2000, "image_size": 64, "bc_samples": len(data), "bc_epochs": BC_EPOCHS,
        "stage0_success": s0, "stage0_steps": st0, "stage1_success": s1, "stage1_steps": st1,
        "target_success": s2, "target_steps": st2, "seconds": round(time.perf_counter() - t0, 3),
    }
    (ROOT / "curriculum_vision_v2_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
