# MZX 连续 PPO：replay + KL + optimizer checkpoint

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
任务：MuJoCo 点质量  
阶段：6 个，目标区域冲突  
方法：PPO，无 PD 专家

---

## 1. 已实现

- PPO 替代 BC
- replay buffer（上限 4000）
- KL 到旧策略（系数 0.05）
- BC 正则到 replay 动作（系数 0.05）
- optimizer state 保存/加载
- 6 个阶段：
  - right
  - left
  - up
  - center
  - left2
  - three obstacles

每阶段：

~~~text
4 envs
15 updates × 128 steps
~~~

---

## 2. 结果

| stage | success | steps |
| --- | ---: | ---: |
| 0 right | 0.05 | 244.35 |
| 1 left | 0.35 | 213.50 |
| 2 up | 0.00 | 250.00 |
| 3 center | 0.00 | 250.00 |
| 4 left2 | 0.05 | 243.15 |
| 5 three | 0.05 | 244.35 |

最终模型回测：

| stage | success |
| --- | ---: |
| right | 0.00 |
| left | 0.00 |
| up | 0.10 |
| center | 0.10 |
| left2 | 0.00 |
| three | 0.05 |

总耗时：67.622 s

---

## 3. 结论

> 流程全部跑通，但 PPO 从零连续训练失败。

- stage0 只有 5%
- stage1 最高 35%
- 后面阶段基本 0–5%
- 最终模型在所有旧阶段都没有保留能力

---

## 4. 为什么

1. PPO 从零开始，stage0 都没学会
2. 奖励稀疏：距离 + 碰撞
3. 15 updates 太少
4. 目标区域跨度大（right → left → up）
5. replay + KL 只能防止漂移，不能教会新任务
6. 没有 BC 初始化，500 节点随机图很难探索
7. 任务冲突太强

---

## 5. 重要说明

这次失败不是“replay/KL 没用”的证明。

而是：

> 在 PPO 还没学会第一个任务时，replay 和 KL 没有东西可保护。

---

## 6. 下一步

1. 先用 BC 或示范把 stage0 做到 80%+
2. 再开 PPO + replay + KL
3. 课程从 right 开始，不要立刻 left/up
4. 增加 updates 到 100+
5. 奖励塑形
6. 再测任务冲突

---

## 7. 一句话

> 连续 PPO 的 replay、KL、optimizer checkpoint 都已实现；但 PPO 从零在 6 个冲突阶段上失败，最高只有 35%，说明先要有 BC 初始化，不能直接连续 PPO。

---

## 8. 文件

- continual_ppo.py
- continual_ppo_result.json
- mzx_ppo_stage0.pt ... mzx_ppo_stage5.pt
- CONTINUAL_PPO_REPORT.md
