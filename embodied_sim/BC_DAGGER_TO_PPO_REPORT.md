# BC 99% → PPO：小范围随机目标 + 1 随机障碍

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
BC checkpoint：mzx_bc_dagger.pt（固定目标 99%）  
PPO：8 envs × 256 steps，20 updates  
KL：0.20 → 0.02，BC coef 0.30  
评估：100 episodes，确定性动作

---

## 1. 结果

| 模型 | success | steps |
| --- | ---: | ---: |
| BC 在固定任务 | 0.99 | 134.21 |
| BC 在小范围随机任务 | **0.53** | 179.67 |
| PPO 最后 | 0.51 | 178.68 |
| PPO best | **0.47** | 191.42 |
| best 在强目标 | 0.03 | 246.48 |

best update：0  
早停：update 10

总耗时：37.938 s

---

## 2. 结论

> BC 99% 一进入小范围随机目标任务，就掉到 53%。

PPO 没有把它拉回来：

~~~text
BC 53%
PPO best 47%
PPO last 51%
强目标 3%
~~~

---

## 3. 为什么

1. BC checkpoint 只见过固定目标/固定障碍
2. 小范围随机目标 + 随机障碍是分布偏移
3. PPO 从 BC 起点开始，但 KL/BC anchor 把策略锁在旧分布
4. 20 updates 不够
5. 没有在新任务上重新收集专家数据
6. 强目标差距更大

---

## 4. 关键教训

> BC 99% 不是“通用避障能力”，只是“固定任务 99%”。

跨任务直接 PPO 微调不够。

---

## 5. 下一步

1. 在新任务上重新收集 PD 专家数据
2. 先 BC/DAgger 到 80%+
3. 再 PPO 微调
4. 或先固定障碍，只随机目标
5. 再随机障碍
6. 每次只改一个变量

---

## 6. 一句话

> BC 在固定任务 99%，但换到小范围随机目标 + 随机障碍后只有 53%；PPO 直接微调没有救回来，必须先在新任务上补专家数据。

---

## 7. 文件

- bc_dagger_to_ppo.py
- bc_dagger_to_ppo_result.json
- mzx_ppo_dagger_best.pt
- BC_DAGGER_TO_PPO_REPORT.md
