#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-variable curriculum: fixed obstacle + random target, then random obstacle."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from bc_to_ppo import PolicyAC, VecState, evaluate as ppo_evaluate, bc_mean, N_ENVS, STEPS, GAMMA, LAM, CLIP, EPOCHS, MINIBATCH, LR, KL_START, KL_END, BC_COEF, PATIENCE
from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
CFG_A = dict(n_obstacles=1, target_range=0.3, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))
CFG_B = dict(n_obstacles=1, target_range=0.3, start_range=0.5, obstacle_range=0.5, target_bias=(0.5, 0.0))
BC_EPOCHS = 40
PPO_UPDATES = 20
VAL = 50


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    rel = state[6:8]
    d = float(np.linalg.norm(rel))
    if 1e-6 < d < d0:
        force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def make_env(cfg, seed):
    return VisionPointMassEnv(seed=seed, render=False, size=8, **cfg)


def eval_expert(cfg, episodes=100, seed=35000):
    env = make_env(cfg, seed)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            r = env.step(expert_action(state)); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def collect(cfg, episodes=100, seed=35100):
    env = make_env(cfg, seed)
    X, Y = [], []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            a = expert_action(state)
            X.append(state.copy()); Y.append(a.copy())
            r = env.step(a); state = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, cfg, episodes=100, seed=35200):
    env = make_env(cfg, seed)
    succ = 0; steps = []
    for _ in range(episodes):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); state = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def train_bc(model, X, Y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for ep in range(BC_EPOCHS):
        perm = torch.randperm(len(Xt)); total = 0.0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            pred = model(Xt[idx]); loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()) * len(idx)
        if ep % 10 == 0:
            print("BC epoch", ep, "mse", round(total / len(Xt), 6), flush=True)
    return model


def dagger(model, cfg, seed):
    X, Y = [], []
    env = make_env(cfg, seed)
    for _ in range(30):
        _, state = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(state, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(state.copy()); Y.append(expert_action(state))
            r = env.step(a); state = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def ppo_finetune(model, cfg, seed):
    bc_policy = copy.deepcopy(model).eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)
    ac = PolicyAC(model)
    opt_venv = VecState(cfg, seed=seed + 100)
    states = opt_venv.reset()
    replay_s = torch.zeros((0, 12))
    opt = torch.optim.Adam(ac.parameters(), lr=LR)
    bc_s, bc_steps = evaluate(model, cfg, seed=seed + 200)
    best_s = bc_s; best_state = copy.deepcopy(ac.state_dict()); drops = 0
    print("PPO start", bc_s, bc_steps, flush=True)
    for upd in range(PPO_UPDATES):
        kl_coef = KL_END + (KL_START - KL_END) * max(0.0, 1.0 - upd / max(1, PPO_UPDATES - 1))
        bufs = [{"S": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(N_ENVS)]
        for _ in range(STEPS):
            acts = []
            for i in range(N_ENVS):
                s = torch.tensor(states[i], dtype=torch.float32).view(1, -1)
                with torch.no_grad():
                    a, raw, logp, value = ac.act(s)
                bufs[i]["S"].append(s[0]); bufs[i]["A"].append(raw[0]); bufs[i]["LP"].append(logp[0]); bufs[i]["V"].append(value[0])
                acts.append(a[0].numpy())
            states, rews, dones = opt_venv.step(acts)
            for i in range(N_ENVS):
                bufs[i]["R"].append(rews[i]); bufs[i]["D"].append(float(dones[i]))
        all_S, all_A, all_LP, all_R, all_V, all_adv, all_ret = [], [], [], [], [], [], []
        for i in range(N_ENVS):
            b = bufs[i]; R = torch.tensor(b["R"]); D = torch.tensor(b["D"]); V = torch.stack(b["V"])
            with torch.no_grad():
                _, nv = ac(torch.tensor(states[i], dtype=torch.float32).view(1, -1)); nv = nv[0]
            adv = torch.zeros_like(R); last = 0.0
            for t in reversed(range(len(R))):
                nonterm = 1.0 - D[t]; nextv = nv if t == len(R) - 1 else V[t + 1]
                delta = R[t] + GAMMA * nextv * nonterm - V[t]
                last = delta + GAMMA * LAM * nonterm * last; adv[t] = last
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
                mean, value = ac(all_S[idx]); std = torch.exp(ac.log_std).clamp(0.02, 1.0)
                dist = Normal(mean, std); logp = dist.log_prob(all_A[idx]).sum(-1)
                ratio = torch.exp(logp - all_LP[idx])
                surr = torch.min(ratio * all_adv[idx], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * all_adv[idx])
                loss = -surr.mean() + 0.5 * torch.mean((value - all_ret[idx]) ** 2) - 0.01 * dist.entropy().sum(-1).mean()
                ridx = torch.randint(0, len(replay_s), (64,))
                bcm = bc_mean(bc_policy, replay_s[ridx]); newm, _ = ac(replay_s[ridx])
                kl = (torch.log(std / 0.5) + (0.25 + (bcm - newm) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                loss = loss + kl_coef * kl + BC_COEF * torch.mean((newm - bcm) ** 2)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        if upd % 5 == 0:
            s, st = evaluate(ac.policy, cfg, episodes=VAL, seed=seed + 300 + upd)
            if s > best_s: best_s = s; best_state = copy.deepcopy(ac.state_dict()); drops = 0
            else: drops += 1
            print("PPO", upd, s, st, "best", best_s, flush=True)
            if drops >= PATIENCE: break
    ac.load_state_dict(best_state)
    s, st = evaluate(ac.policy, cfg, episodes=100, seed=seed + 400)
    return ac.policy, {"bc_success": bc_s, "best_success": best_s, "final_success": s, "final_steps": st}


def run_task(cfg, seed):
    expert_s, expert_steps = eval_expert(cfg, seed=seed)
    X, Y = collect(cfg, seed=seed + 1)
    model = LifeGraphPolicy(500, obs_dim=12)
    train_bc(model, X, Y, seed + 2)
    bc_s, bc_steps = evaluate(model, cfg, seed=seed + 3)
    print("BC", expert_s, bc_s, bc_steps, flush=True)
    Xd, Yd = dagger(model, cfg, seed + 4)
    train_bc(model, np.concatenate([X, Xd]), np.concatenate([Y, Yd]), seed + 5)
    dg_s, dg_steps = evaluate(model, cfg, seed=seed + 6)
    print("DAgger", dg_s, dg_steps, flush=True)
    model, ppo_stats = ppo_finetune(model, cfg, seed + 7)
    return {"expert": expert_s, "expert_steps": expert_steps, "bc": bc_s, "bc_steps": bc_steps, "dagger": dg_s, "dagger_steps": dg_steps, "ppo": ppo_stats}


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(35000); np.random.seed(35000)
    a = run_task(CFG_A, 35000)
    print("TASK_A", a, flush=True)
    b = run_task(CFG_B, 36000)
    print("TASK_B", b, flush=True)
    payload = {"task_a_fixed_obstacle_random_target": a, "task_b_random_obstacle": b, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "target_obstacle_curriculum_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
