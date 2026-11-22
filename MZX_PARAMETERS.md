# MZX-0.1 参数总表

日期：2026-11-22  
范围：当前具身主线；历史 7D/10k 分支只在最后列关键参数。

---

## 1. 模型结构

| 参数 | 值 |
| --- | ---: |
| 节点数 | 500 |
| 边数 | 2065 |
| 图消息传递步数 K | 6 |
| 输入维度 | 64 |
| 状态观察维度 | 12 |
| 图输出特征 | 498 |
| 动作维度 | 2 |
| actor 输出 | 2 |
| critic 输出 | 1 |
| log_std | 2 |

参数量：

| 部分 | 参数 |
| --- | ---: |
| 边权 | 2,065 |
| 节点 bias | 500 |
| input_gain (500×64) | 32,000 |
| actor | 998 |
| critic | 499 |
| log_std | 2 |
| **总计** | **36,064** |

---

## 2. MuJoCo 环境

| 参数 | 值 |
| --- | ---: |
| 物理引擎 | MuJoCo 3.2.7 |
| 世界 | 2D 点质量 |
| 自由度 | 2 |
| 关节 | jx, jy |
| 时间步 | 0.01 s |
| 电机 gear | 10 |
| 关节 damping | 1.2 |
| agent 半径 | 0.10 |
| target 半径 | 0.08 |
| obstacle 半径 | 0.14 |
| 成功距离 | < 0.14 |
| 最大步数 | 250 |
| z | 0.3 固定 |

当前 PPO 固定任务：

| 参数 | 值 |
| --- | ---: |
| 障碍数 | 1 |
| 障碍范围 | 0（固定原点） |
| 目标范围 | 0（固定） |
| 目标位置 | (0.5, 0.0) |
| 起点范围 | ±0.5 |

强目标评估：

| 参数 | 值 |
| --- | ---: |
| 障碍数 | 2 |
| 目标范围 | ±1.2 |
| 起点范围 | ±1.0 |
| 障碍范围 | ±1.0 |

---

## 3. 生命周期参数

| 参数 | 值 |
| --- | ---: |
| grow per cycle | 10 |
| cycles | 5 |
| epochs per phase | 3 |
| time_unit | 0.1 s |
| tau | 8.0 round-units |
| usage threshold | 0.005 |
| retention threshold | 0.50 |
| 初节点 | 500 |
| 终节点 | 550 |

---

## 4. BC 参数

| 参数 | 值 |
| --- | ---: |
| BC checkpoint | mzx_stage2_two_obstacles.pt |
| PD episodes / stage | 60 |
| BC epochs / stage | 3 |
| optimizer | AdamW |
| LR | 1e-3 |
| weight decay | 1e-4 |
| batch | 64 |

---

## 5. 当前 PPO 参数

| 参数 | 值 |
| --- | ---: |
| 并行环境 | 8 |
| updates | 20 |
| steps / env / update | 256 |
| gamma | 0.99 |
| lambda | 0.95 |
| clip | 0.2 |
| PPO epochs | 3 |
| minibatch | 64 |
| LR | 1e-4 |
| KL start | 0.20 |
| KL end | 0.02 |
| BC loss coef | 0.30 |
| replay max | 4000 |
| validation episodes | 30 |
| early-stop patience | 2 |
| best checkpoint | 是 |

---

## 6. 当前结果

| 指标 | 值 |
| --- | ---: |
| BC 在固定目标 | 0.5333 |
| PPO best | 0.7667 |
| PPO 最后 | 0.4667 |
| 强目标 | 0.1667 |
| best update | 5 |
| 早停 | update 15 |

---

## 7. Checkpoint 文件

- mzx_stage0_no_obstacle.pt
- mzx_stage1_one_obstacle.pt
- mzx_stage2_two_obstacles.pt
- mzx_ppo_stage0.pt ... mzx_ppo_stage5.pt

---

## 8. 归档分支的关键参数

| 分支 | 参数 |
| --- | --- |
| 7D 状态 | E/A/L/C/B/S/P |
| 10k 图 | 10000 节点 / 40105 边 |
| 10k 图（seed-grown） | 10000 节点 / 56212 边 |
| 2000 节点视觉 | 2000 节点 / 64×64 |
| 37 节点 parity | 37 节点 / 6 递质 |

---

## 9. 说明

这里列的是结构、训练、环境和当前实验参数。  
具体权重张量在 checkpoint 文件中，不逐个列出数值。
