# 课程 v2：20k 演示、30 epoch BC、64 视觉、单障碍、per-env GAE

日期：2026-11-22  
模型：2000 节点图 + CNN  
视觉：64×64 RGB  
并行：4 envs

---

## 1. 改动

- BC 数据：3200 → **20,000**
- BC epoch：8 → **30**
- 视觉：32 → **64**
- Stage 0：无障碍、小范围
- Stage 1：**1 个固定障碍物**，位于原点
- PPO：改为 **per-env GAE**

---

## 2. Stage 0 BC

| epoch | MSE |
| ---: | ---: |
| 0 | 0.2799 |
| 10 | 0.0321 |
| 20 | 0.0148 |
| 29 | **0.00935** |

评估：

~~~text
stage0 success = 0.9667
stage0 steps   = 96.63
~~~

结论：

> Stage 0 已经超过 90% 目标。

---

## 3. Stage 1 PPO

从 BC 权重开始，单固定障碍物。

| update | stage1 success |
| ---: | ---: |
| 0 | 0.0 |
| 3 | 0.1 |
| 6 | 0.0 |
| 9 | 0.1 |

最终：

~~~text
stage1 success = 0.0333
stage1 steps   = 245.1
~~~

Target domain：

~~~text
success = 0.00
~~~

---

## 4. 结论

> Stage 0 成功了；Stage 1 PPO 仍然失败。

这说明：

- 20k 演示 + 30 epoch BC 足够学会无障碍小范围
- 但从 BC 到单障碍 PPO，10 updates 不够
- 固定障碍物仍然需要避障策略
- per-env GAE 修正了训练信号，但没有解决样本量

---

## 5. 效率

~~~text
总耗时：201.255 s
~~~

---

## 6. 下一步

1. Stage 1 PPO 增加到 100+ updates
2. 先固定目标，不随机
3. 障碍物从中心开始，逐步随机
4. 加避障奖励
5. 并行环境增加到 8–16
6. 用 BC 数据做 DAgger

---

## 7. 一句话

> 20k 演示 + 30 epoch BC 把 Stage 0 做到 96.7%；但单障碍 Stage 1 的 PPO 只有 3.3%，还需要更多训练和课程。

---

## 8. 文件

- curriculum_vision_v2.py
- curriculum_vision_v2_result.json
- CURRICULUM_VISION_V2_REPORT.md
