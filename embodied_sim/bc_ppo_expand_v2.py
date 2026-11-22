#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand curriculum to medium target and 1->2->3 obstacles, 3 seeds."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from bc_to_ppo import PolicyAC, VecState, evaluate, bc_mean, N_ENVS, STEPS, GAMMA, LAM, CLIP, EPOCHS, MINIBATCH, LR, KL_START, KL_END, BC_COEF, PATIENCE
from continual_learning import load_checkpoint

ROOT = Path(__file__).resolve().parent
UPDATES = 20
VAL_EPISODES = 50
SEEDS = [32000, 32001, 32002, 32003, 32004]
STAGES = [
    ("fixed_1obs", dict(n_obstacles=1, target_range=0.5, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))),
    ("small_1obs", dict(n_obstacles=1, target_range=0.5, start_range=0.5, obstacle_range=0.3, target_bias=(0.5, 0.0))),
    ("medium_1obs", dict(n_obstacles=1, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))),
    ("medium_2obs", dict(n_obstacles=2, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))),
    ("medium_3obs", dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))),
    ("three_extra", dict(n_obstacles=3, target_range=0.6, start_range=0.6, obstacle_range=0.8, target_bias=(0.5, 0.0))),
]


def run_stage(model, bc_policy, cfg, seed, name):
    start_s, start_steps = evaluate(model, cfg, episodes=VAL_EPISODES, seed=seed + 100)
    print(name, "start", start_s, start_steps, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    venv = VecState(cfg, seed=seed + 200)
    states = venv.reset()
    replay_s = torch.zeros((0, 12))
    best_success = start_s
    best_state = copy.deepcopy(model.state_dict())
    drops = 0
    history = []

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
                ridx = torch.randint(0, len(replay_s), (64,))
                bcm = bc_mean(bc_policy, replay_s[ridx])
                newm, _ = model(replay_s[ridx])
                kl = (torch.log(std / 0.5) + (0.25 + (bcm - newm) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                loss = loss + kl_coef * kl + BC_COEF * torch.mean((newm - bcm) ** 2)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()

        if upd % 5 == 0 or upd == UPDATES - 1:
            s, st = evaluate(model, cfg, episodes=VAL_EPISODES, seed=seed + 300 + upd)
            if s > best_success:
                best_success = s; best_state = copy.deepcopy(model.state_dict()); drops = 0
            else:
                drops += 1
            history.append({"update": upd, "success": s, "steps": st, "best": best_success, "kl": round(kl_coef, 4)})
            print(name, "upd", upd, "s", s, "best", best_success, "drops", drops, flush=True)
            if drops >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final_s, final_steps = evaluate(model, cfg, episodes=VAL_EPISODES, seed=seed + 500)
    return model, {"stage": name, "start_success": start_s, "final_success": final_s, "final_steps": final_steps, "best_success": best_success, "history": history}


def main() -> None:
    t0 = time.perf_counter()
    all_results = []
    base_policy = load_checkpoint(ROOT / "mzx_stage2_two_obstacles.pt")
    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = PolicyAC(base_policy)
        bc_policy = copy.deepcopy(base_policy).eval()
        for p in bc_policy.parameters():
            p.requires_grad_(False)
        seed_results = []
        for j, (name, cfg) in enumerate(STAGES):
            model, r = run_stage(model, bc_policy, cfg, seed + j * 1000, name)
            seed_results.append(r)
            print("RESULT", seed, name, r["final_success"], r["best_success"], flush=True)
        all_results.append({"seed": seed, "stages": seed_results})

    payload = {"seeds": all_results, "updates": UPDATES, "n_envs": N_ENVS, "steps": STEPS, "val_episodes": VAL_EPISODES, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_ppo_expand_v2_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
