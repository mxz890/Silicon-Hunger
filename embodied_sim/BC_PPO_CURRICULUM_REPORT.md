# BC → PPO 课程：固定目标 → 小范围随机

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
方法：BC checkpoint → PPO 课程  
PPO：8 envs × 256 steps，KL 0.20 → 0.02  
保护：best checkpoint + early stop

---

## 1. 课程

### Stage A

~~~text
1 个固定障碍
目标固定 (0.5, 0.0)
起点 ±0.5
~~~

### Stage B

~~~text
1 个随机障碍，范围 ±0.5
目标 (0.5, 0.0) 附近，范围 ±0.3
起点 ±0.5
~~~

---

## 2. 结果

### Stage A：固定目标

| 模型 | success | steps |
| --- | ---: | ---: |
| BC | 0.5333 | 160.13 |
| PPO best | 0.7333 | 131.90 |
| final best checkpoint | **0.7667** | 133.27 |

### Stage B：小范围随机

| 模型 | success | steps |
| --- | ---: | ---: |
| 进入时 BC | 0.4333 | 177.97 |
| PPO best | **0.9667** | 104.80 |
| final best checkpoint | 0.8333 | 125.00 |

### 强随机目标（2 障碍，±1.0）

~~~text
success = 0.4333
steps   = 197.17
~~~

总耗时：58.437 s

---

## 3. 结论

> 课程学习有效。

固定目标：

~~~text
BC 53% → PPO 77%
~~~

小范围随机：

~~~text
BC 43% → PPO best 97%
~~~

说明：

- PPO 在固定小范围上能超过 BC
- 从固定到小范围随机，迁移成功
- best checkpoint + early stop 保住了最好结果
- 强随机目标仍有明显下降

---

## 4. 为什么提升

1. BC 已经会接近目标
2. 固定目标让 PPO 先修正控制
3. 小范围随机只扩大一点分布
4. KL 退火允许后期更多探索
5. 8 envs × 256 steps 提供了更多数据
6. early stop 防止过训练

---

## 5. 限制

- 每个阶段只有 20 updates
- 验证 30 episodes，噪声仍存在
- 1 障碍，不是 2–3 障碍
- 强随机目标只有 43%
- 没有多 seed

---

## 6. 下一步

1. 小范围随机 → 中等范围随机
2. 1 障碍 → 2 障碍
3. 多 seed 复现
4. 验证 episodes 提到 50
5. 再扩到 3 障碍

---

## 7. 一句话

> 固定目标 BC 53% → PPO 77%；小范围随机 BC 43% → PPO best 97%。  
> 这次课程和 PPO 微调都真正起作用了。

---

## 8. 文件

- bc_ppo_curriculum.py
- bc_ppo_curriculum_result.json
- BC_PPO_CURRICULUM_REPORT.md
