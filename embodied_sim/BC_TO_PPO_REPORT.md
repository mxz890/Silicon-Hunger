# BC 93% → PPO 微调：仍会坍缩，但 best checkpoint 能救回来

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
任务：MuJoCo 点质量 + 2 个随机障碍  
方法：BC checkpoint → PPO + KL/BC anchor  
评估：确定性动作

---

## 1. 设置

- BC checkpoint：mzx_stage2_two_obstacles.pt
- BC 确定性评估：0.9333
- PPO：20 updates
- 4 envs
- KL 到 BC：0.05
- BC loss：0.10
- 目标随机化：±1.0
- 评估：确定性 tanh(mean)，不采样

---

## 2. 结果

| 模型 | success | steps |
| --- | ---: | ---: |
| BC checkpoint | **0.9333** | 118.70 |
| PPO 最后 | **0.5000** | 204.23 |
| PPO best checkpoint | **0.9000** | 124.83 |
| best 在强目标 | 0.9000 | 118.33 |

best update：5

---

## 3. 结论

> 从 93% 的 BC 开始 PPO，仍然会坍缩。

- 最后 20 updates：93% → 50%
- 中间 update 5：一度 100%
- 用 best checkpoint 后：90%
- 强目标下：90%

所以：

> PPO 会先短暂变好，然后过训练坍缩；best checkpoint 是必须的。

---

## 4. 为什么有 BC 还会坍缩

1. PPO 的 critic 是新初始化的
2. 价值估计早期不准
3. KL/BC 系数不够强
4. 目标随机化较大
5. 10–20 updates 内 policy 会漂移
6. 评估用 10 episodes 选 best，噪声大
7. 没有 early stopping

---

## 5. 这次说明了什么

能说明：

- PPO 不是“从 93 自然涨到更高”
- 它会先探索，再漂移
- best checkpoint 能保住最好结果

不能说明：

- PPO 一定比 BC 差
- KL/BC anchor 无效

只说明：

> 当前 PPO 配置不稳定，需要 early stopping + 更强 BC anchor。

---

## 6. 下一步

1. 保存 best checkpoint（已做）
2. 早停：连续 2 次验证下降就停
3. KL 系数从 0.05 提到 0.2
4. BC 系数从 0.10 提到 0.3
5. 增加验证 episodes 到 30–50
6. 先固定目标，再随机
7. 100+ updates 但必须早停

---

## 7. 一句话

> 从 BC 93% 开始 PPO，最后会掉到 50%；但 update 5 的 best checkpoint 能到 90%。  
> 所以 PPO 微调必须配 best checkpoint / early stopping，不能拿最后一轮。

---

## 8. 文件

- bc_to_ppo.py
- bc_to_ppo_result.json
- BC_TO_PPO_REPORT.md
