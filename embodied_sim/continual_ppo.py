#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continual PPO with replay + KL-to-old-policy + optimizer checkpoints.

Six stages with conflicting target regions:
  right -> left -> up -> center -> left -> three obstacles

No PD expert.
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

from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
OBS_DIM = 12
ACT_DIM = 2
N_ENVS = 4
UPDATES = 15
STEPS = 128
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINIBATCH = 64
LR = 1e-4
KL_COEF = 0.05
BC_COEF = 0.05
REPLAY_MAX = 4000


class PolicyAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = LifeGraphPolicy(500, obs_dim=OBS_DIM)
        self.critic = nn.Linear(self.policy.n - 2, 1)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

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


STAGES = [
    ("stage0_right", dict(n_obstacles=0, target_range=0.35, start_range=0.30, obstacle_range=0.0, target_bias=(0.6, 0.0))),
    ("stage1_left", dict(n_obstacles=1, target_range=0.40, start_range=0.40, obstacle_range=0.8, target_bias=(-0.7, 0.0))),
    ("stage2_up", dict(n_obstacles=1, target_range=0.40, start_range=0.40, obstacle_range=0.8, target_bias=(0.0, 0.7))),
    ("stage3_center", dict(n_obstacles=2, target_range=0.50, start_range=0.50, obstacle_range=1.0, target_bias=(0.0, 0.0))),
    ("stage4_left2", dict(n_obstacles=2, target_range=0.50, start_range=0.50, obstacle_range=1.0, target_bias=(-0.6, 0.0))),
    ("stage5_three", dict(n_obstacles=3, target_range=0.55, start_range=0.60, obstacle_range=1.0, target_bias=(0.0, 0.0))),
]


def make_env(cfg, seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)


class VecState:
    def __init__(self, cfg, seed):
        self.envs = [make_env(cfg, seed + i) for i in range(N_ENVS)]

    def reset(self):
        return [e.reset()[1] for e in self.envs]

    def step(self, actions):
        states, rewards, dones = [], [], []
        for e, a in zip(self.envs, actions):
            r = e.step(a)
            states.append(e.reset()[1] if r.done else r.state)
            rewards.append(r.reward); dones.append(r.done)
        return states, rewards, dones


def evaluate(model, cfg, episodes=20, seed=12000):
    env = make_env(cfg, seed)
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


def save_ckpt(model, opt, old_model, replay_s, replay_a, idx):
    path = ROOT / f"mzx_ppo_stage{idx}.pt"
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "old_model": old_model.state_dict() if old_model is not None else None,
        "replay_s": replay_s, "replay_a": replay_a, "stage": idx,
    }, path)
    return path


def load_ckpt(path):
    ck = torch.load(path, weights_only=False)
    model = PolicyAC(); model.load_state_dict(ck["model"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt.load_state_dict(ck["optimizer"])
    old_model = None
    if ck["old_model"] is not None:
        old_model = PolicyAC(); old_model.load_state_dict(ck["old_model"])
    return model, opt, old_model, ck["replay_s"], ck["replay_a"]


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(12000); np.random.seed(12000)
    model = PolicyAC()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    replay_s = torch.zeros((0, OBS_DIM))
    replay_a = torch.zeros((0, ACT_DIM))
    old_model = None
    history = []
    final_eval = []

    for idx, (name, cfg) in enumerate(STAGES):
        if idx > 0:
            model, opt, old_model, replay_s, replay_a = load_ckpt(ROOT / f"mzx_ppo_stage{idx-1}.pt")
            print("loaded stage", idx - 1, flush=True)
        old_model = copy.deepcopy(model).eval()
        for p in old_model.parameters():
            p.requires_grad_(False)

        venv = VecState(cfg, seed=13000 + idx)
        states = venv.reset()
        for upd in range(UPDATES):
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

            # Replay: keep a bounded buffer of states/actions from this and past stages.
            replay_s = torch.cat([replay_s, all_S.detach()])[-REPLAY_MAX:]
            replay_a = torch.cat([replay_a, torch.tanh(all_A).detach()])[-REPLAY_MAX:]

            for _ in range(EPOCHS):
                perm = torch.randperm(len(all_R))
                for j in range(0, len(all_R), MINIBATCH):
                    idxs = perm[j:j + MINIBATCH]
                    mean, value = model(all_S[idxs])
                    std = torch.exp(model.log_std).clamp(0.02, 1.0)
                    dist = Normal(mean, std)
                    logp = dist.log_prob(all_A[idxs]).sum(-1)
                    ratio = torch.exp(logp - all_LP[idxs])
                    surr = torch.min(ratio * all_adv[idxs], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * all_adv[idxs])
                    loss = -surr.mean() + 0.5 * torch.mean((value - all_ret[idxs]) ** 2) - 0.01 * dist.entropy().sum(-1).mean()

                    # Replay KL + BC to old policy.
                    if len(replay_s) >= 64 and old_model is not None:
                        ridx = torch.randint(0, len(replay_s), (64,))
                        with torch.no_grad():
                            old_mean = old_model(replay_s[ridx])[0]
                            old_std = torch.exp(old_model.log_std).clamp(0.02, 1.0)
                        new_mean, _ = model(replay_s[ridx])
                        new_std = std
                        kl = (torch.log(new_std / old_std) + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2) - 0.5).sum(-1).mean()
                        bc = torch.mean((new_mean - replay_a[ridx]) ** 2)
                        loss = loss + KL_COEF * kl + BC_COEF * bc

                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            if upd % 5 == 0:
                s, _ = evaluate(model, cfg, episodes=10, seed=14000 + idx * 100 + upd)
                print("stage", idx, "upd", upd, "success", s, flush=True)

        s, st = evaluate(model, cfg, episodes=20, seed=15000 + idx)
        history.append({"stage": name, "success": s, "steps": st, "replay": len(replay_s)})
        print("STAGE", history[-1], flush=True)
        ck = save_ckpt(model, opt, old_model, replay_s, replay_a, idx)
        print("saved", ck, flush=True)

    # Evaluate final model on all prior stages (forgetting).
    model, _, _, _, _ = load_ckpt(ROOT / f"mzx_ppo_stage{len(STAGES)-1}.pt")
    for idx, (name, cfg) in enumerate(STAGES):
        s, st = evaluate(model, cfg, episodes=20, seed=16000 + idx)
        final_eval.append({"stage": name, "success": s, "steps": st})
        print("FINAL-ON", final_eval[-1], flush=True)

    payload = {"stages": history, "final_eval": final_eval, "seconds": round(time.perf_counter() - t0, 3), "replay_max": REPLAY_MAX, "kl_coef": KL_COEF, "bc_coef": BC_COEF}
    (ROOT / "continual_ppo_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
