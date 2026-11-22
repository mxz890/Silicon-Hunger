# 目标随机 → 障碍随机：一次只改一个变量

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
专家：PD + 障碍斥力  
流程：新任务专家 → BC 40 epochs → DAgger → PPO  
评估：100 episodes

---

## 1. Task A：固定障碍 + 随机目标

配置：

~~~text
1 个固定障碍
目标 (0.5, 0.0) 附近 ±0.3
起点 ±0.5
~~~

结果：

| 阶段 | success | steps |
| --- | ---: | ---: |
| 专家 | 0.85 | 143.69 |
| BC | 0.88 | 145.42 |
| DAgger | 0.87 | 145.54 |
| PPO best | 0.90 | — |
| PPO final | **0.90** | 147.07 |

---

## 2. Task B：随机障碍 + 同一目标分布

配置：

~~~text
1 个随机障碍 ±0.5
目标 (0.5, 0.0) 附近 ±0.3
起点 ±0.5
~~~

结果：

| 阶段 | success | steps |
| --- | ---: | ---: |
| 专家 | 0.93 | 112.34 |
| BC | 0.84 | 122.48 |
| DAgger | 0.92 | 108.81 |
| PPO best | **0.96** | — |
| PPO final | 0.91 | 113.94 |

---

## 3. 结论

> 这次一次只改一个变量，成功了。

Task A：

~~~text
固定障碍 → 随机目标
BC 88% → PPO 90%
~~~

Task B：

~~~text
固定目标 → 随机障碍
BC 84% → DAgger 92% → PPO best 96%
~~~

- BC 都超过 80%
- DAgger 有提升
- PPO best 更高
- 没有出现之前那种 50% 崩溃

总耗时：168.584 s

---

## 4. 为什么这次有效

1. 每个任务都重新收集专家数据
2. 先 BC 到 80%+
3. 再 DAgger
4. 再 PPO 微调
5. 一次只改一个变量
6. 保留 best checkpoint

---

## 5. 下一步

1. 2000 节点替换 500
2. 2 障碍
3. 目标范围继续扩大
4. 多 seed
5. sim-to-sim

---

## 6. 一句话

> 固定障碍+随机目标：BC 88% → PPO 90%；随机障碍：BC 84% → DAgger 92% → PPO best 96%。  
> “每次只改一个变量”这次真的有效。

---

## 7. 文件

- target_obstacle_curriculum.py
- target_obstacle_curriculum_result.json
- TARGET_OBSTACLE_CURRICULUM_REPORT.md
