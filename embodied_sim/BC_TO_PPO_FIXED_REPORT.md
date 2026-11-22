# BC → PPO：KL 退火 + 固定目标小范围

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
任务：MuJoCo 点质量 + 1 个固定障碍 + 固定目标 (0.5, 0.0)

---

## 1. 设置

- BC checkpoint：mzx_stage2_two_obstacles.pt
- PPO：8 envs × 256 steps
- KL：0.20 → 0.02 退火
- BC loss：0.30
- 验证：30 episodes
- 连续 2 次验证下降 → 早停
- best checkpoint

---

## 2. 结果

| 阶段 | success | steps |
| --- | ---: | ---: |
| BC 在固定目标 | 0.5333 | 161.07 |
| PPO update 0 | 0.4000 | 178.43 |
| PPO update 5 | **0.7333** | 138.23 |
| PPO update 10 | 0.5000 | 166.80 |
| PPO update 15 | 0.4667 | 166.33 |
| best checkpoint 回测 | **0.7667** | 135.90 |
| 强目标回测 | 0.1667 | 226.00 |

早停：update 15。

---

## 3. 结论

> 固定目标小范围下，PPO 确实比 BC 提升了。

- BC：53.3%
- PPO best：76.7%
- 提升约 23 个百分点
- 但强目标随机化下只有 16.7%

所以：

> PPO 在“固定目标小范围”有效，但不能直接迁移到强随机目标。

---

## 4. 为什么 BC 只有 53%

BC checkpoint 原来训练的是：

- 2 个随机障碍
- 目标范围 ±1.0

现在评估：

- 1 个固定障碍
- 固定目标 (0.5, 0.0)

这是分布变化。

---

## 5. 这次说明了什么

能说明：

- KL 退火 + 固定目标 + 更多数据，PPO 可以超过 BC
- best checkpoint + early stop 能保住最好结果
- 强随机目标仍然很难

不能说明：

- PPO 已经解决随机障碍
- 可以直接上真机

---

## 6. 下一步

1. 固定目标 → 小范围随机目标
2. 1 障碍固定 → 1 障碍随机
3. 保持 KL 退火
4. 保持 8 envs × 256 steps
5. 继续 best checkpoint + early stop

---

## 7. 一句话

> 固定目标小范围下，PPO 从 BC 的 53% 提升到 77%；但强目标随机化仍然只有 17%，下一步应该逐步扩大目标范围。

---

## 8. 文件

- bc_to_ppo.py
- bc_to_ppo_result.json
- BC_TO_PPO_FIXED_REPORT.md
