#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BC -> PPO curriculum: fixed target/obstacle -> small random target/obstacle."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from bc_to_ppo import (
    PolicyAC, VecState, evaluate, bc_mean, N_ENVS, STEPS, UPDATES,
    GAMMA, LAM, CLIP, EPOCHS, MINIBATCH, LR, KL_START, KL_END,
    BC_COEF, PATIENCE, VAL_EPISODES,
)
from continual_learning import load_checkpoint

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("fixed_target_fixed_obstacle", dict(n_obstacles=1, target_range=0.0, start_range=0.5, obstacle_range=0.0, target_bias=(0.5, 0.0))),
    ("small_random_target_random_obstacle", dict(n_obstacles=1, target_range=0.3, start_range=0.5, obstacle_range=0.5, target_bias=(0.5, 0.0))),
]


def run_stage(model, bc_policy, cfg, seed, stage_name):
    bc_s, bc_steps = evaluate(model, cfg, seed=seed)
    print(stage_name, "BC", bc_s, bc_steps, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    venv = VecState(cfg, seed=seed + 100)
    states = venv.reset()
    replay_s = torch.zeros((0, 12))
    best_success = bc_s
    best_update = -1
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
                bcloss = torch.mean((newm - bcm) ** 2)
                loss = loss + kl_coef * kl + BC_COEF * bcloss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()

        if upd % 5 == 0:
            s, st = evaluate(model, cfg, episodes=VAL_EPISODES, seed=seed + 200 + upd)
            if s > best_success:
                best_success = s; best_update = upd; best_state = copy.deepcopy(model.state_dict()); drops = 0
            else:
                drops += 1
            history.append({"stage": stage_name, "update": upd, "success": s, "steps": st, "best": best_success, "kl": round(kl_coef, 4)})
            print(stage_name, "upd", upd, "s", s, "best", best_success, "drops", drops, "kl", round(kl_coef, 3), flush=True)
            if drops >= PATIENCE:
                print("EARLY STOP", stage_name, upd, flush=True)
                break

    model.load_state_dict(best_state)
    final_s, final_steps = evaluate(model, cfg, seed=seed + 400)
    torch.save(model.state_dict(), ROOT / f"mzx_ppo_{stage_name}.pt")
    return model, {"stage": stage_name, "bc_success": bc_s, "bc_steps": bc_steps, "best_update": best_update, "best_success": best_success, "final_success": final_s, "final_steps": final_steps, "history": history}


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(20000); np.random.seed(20000)
    policy = load_checkpoint(ROOT / "mzx_stage2_two_obstacles.pt")
    model = PolicyAC(policy)
    bc_policy = copy.deepcopy(policy).eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)

    results = []
    for i, (name, cfg) in enumerate(STAGES):
        model, r = run_stage(model, bc_policy, cfg, seed=21000 + i * 1000, stage_name=name)
        results.append(r)
        print("RESULT", r, flush=True)

    strong = dict(n_obstacles=2, target_range=1.0, start_range=0.8, obstacle_range=1.0)
    strong_s, strong_steps = evaluate(model, strong, episodes=30, seed=23000)
    payload = {"stages": results, "strong_random_success": strong_s, "strong_random_steps": strong_steps, "seconds": round(time.perf_counter() - t0, 3)}
    (ROOT / "bc_ppo_curriculum_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
