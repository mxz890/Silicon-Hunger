#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2000-node MZX, 2 obstacles, wider target, 2 seeds, sim-to-sim.

Expert: PD + two-obstacle repulsion.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from bc_to_ppo import PolicyAC, VecState, bc_mean, N_ENVS, STEPS, GAMMA, LAM, CLIP, EPOCHS, MINIBATCH, LR, KL_START, KL_END, BC_COEF, PATIENCE
from lifecycle_embodied_v2 import LifeGraphPolicy
from vision_env import VisionPointMassEnv

ROOT = Path(__file__).resolve().parent
N_NODES = 2000
SEEDS = [40000, 40001]
BC_EPOCHS = 25
PPO_UPDATES = 10
VAL = 50
SRC = dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))
TGT = dict(n_obstacles=2, target_range=0.8, start_range=0.7, obstacle_range=0.8, target_bias=(0.5, 0.0))


def expert_action(state, kp=4.0, kd=1.5, eta=1.8, d0=0.55):
    p, v, t = state[:2], state[2:4], state[4:6]
    force = kp * (t - p) - kd * v
    for off in (6, 8):
        rel = state[off:off + 2]
        d = float(np.linalg.norm(rel))
        if 1e-6 < d < d0:
            force = force + eta * (1.0 / d - 1.0 / d0) * (-rel / d)
    return np.clip(force, -1.0, 1.0)


def make_env(cfg, seed, domain="source"):
    return VisionPointMassEnv(seed=seed, render=False, size=8, domain=domain, **cfg)


def eval_expert(cfg, domain="source", episodes=100, seed=41000):
    env = make_env(cfg, seed, domain)
    succ = 0; steps = []
    for _ in range(episodes):
        _, s = env.reset()
        while True:
            r = env.step(expert_action(s)); s = r.state
            if r.done:
                succ += int(r.success); steps.append(env.steps); break
    return succ / episodes, float(np.mean(steps))


def collect(cfg, seed, episodes=100):
    env = make_env(cfg, seed)
    X, Y = [], []
    for _ in range(episodes):
        _, s = env.reset()
        while True:
            a = expert_action(s)
            X.append(s.copy()); Y.append(a.copy())
            r = env.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(model, cfg, domain="source", episodes=100, seed=42000):
    env = make_env(cfg, seed, domain)
    succ = 0; steps = []
    for _ in range(episodes):
        _, s = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            r = env.step(a); s = r.state
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
        if ep % 5 == 0:
            print("BC", seed, ep, round(total / len(Xt), 6), flush=True)
    return model


def dagger(model, cfg, seed):
    X, Y = [], []
    env = make_env(cfg, seed)
    for _ in range(25):
        _, s = env.reset()
        while True:
            with torch.no_grad():
                a = model(torch.tensor(s, dtype=torch.float32).view(1, -1))[0].numpy()
            X.append(s.copy()); Y.append(expert_action(s))
            r = env.step(a); s = r.state
            if r.done:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def ppo(model, cfg, seed):
    bc_policy = copy.deepcopy(model).eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)
    ac = PolicyAC(model)
    venv = VecState(cfg, seed=seed + 100)
    states = venv.reset()
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
            states, rews, dones = venv.step(acts)
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
            print("PPO", seed, upd, s, "best", best_s, flush=True)
            if drops >= PATIENCE: break
    ac.load_state_dict(best_state)
    s, st = evaluate(ac.policy, cfg, episodes=100, seed=seed + 400)
    return ac.policy, {"bc": bc_s, "best": best_s, "final": s, "final_steps": st}


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ex_s, ex_steps = eval_expert(SRC, episodes=100, seed=seed + 1)
    X, Y = collect(SRC, seed + 2)
    model = LifeGraphPolicy(N_NODES, obs_dim=12)
    train_bc(model, X, Y, seed + 3)
    bc_s, bc_steps = evaluate(model, SRC, seed=seed + 4)
    print("seed", seed, "expert", ex_s, "bc", bc_s, flush=True)
    Xd, Yd = dagger(model, SRC, seed + 5)
    train_bc(model, np.concatenate([X, Xd]), np.concatenate([Y, Yd]), seed + 6)
    dg_s, dg_steps = evaluate(model, SRC, seed=seed + 7)
    print("seed", seed, "dagger", dg_s, flush=True)
    model, p = ppo(model, SRC, seed + 8)
    src_s, src_steps = evaluate(model, SRC, episodes=100, seed=seed + 500)
    tgt_s, tgt_steps = evaluate(model, TGT, domain="target", episodes=100, seed=seed + 600)
    torch.save(model.state_dict(), ROOT / f"mzx_2000_seed{seed}.pt")
    return {"seed": seed, "expert": ex_s, "bc": bc_s, "dagger": dg_s, "ppo_best": p["best"], "ppo_final": p["final"], "source_success": src_s, "source_steps": src_steps, "target_success": tgt_s, "target_steps": tgt_steps}


def main() -> None:
    t0 = time.perf_counter()
    rows = [run_seed(s) for s in SEEDS]
    payload = {"nodes": N_NODES, "seeds": rows, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "mzx_2000_sim2sim_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
