# BC → PPO：小范围 → 中等范围，1 → 2 → 3 障碍，多 seed

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
PPO：8 envs × 256 steps，10 updates / stage  
验证：50 episodes  
保护：KL 0.20 → 0.02、best checkpoint、early stop  
Seeds：31000 / 31001 / 31002

---

## 1. 结果

### 每个 seed

| seed | stage | start | final | best |
| ---: | --- | ---: | ---: | ---: |
| 31000 | small 1obs | 0.86 | 0.90 | 0.94 |
| 31000 | medium 1obs | 0.98 | 0.90 | 0.98 |
| 31000 | medium 2obs | 0.92 | 0.84 | 0.92 |
| 31000 | medium 3obs | 0.86 | 0.70 | 0.86 |
| 31001 | small 1obs | 0.88 | 0.90 | 0.88 |
| 31001 | medium 1obs | 0.94 | 0.98 | 0.96 |
| 31001 | medium 2obs | 0.88 | 0.82 | 0.88 |
| 31001 | medium 3obs | 0.78 | 0.70 | 0.80 |
| 31002 | small 1obs | 0.80 | 0.92 | 0.96 |
| 31002 | medium 1obs | 0.92 | 0.96 | 0.92 |
| 31002 | medium 2obs | 0.90 | 0.80 | 0.90 |
| 31002 | medium 3obs | 0.86 | 0.78 | 0.86 |

### 均值

| stage | final mean | best mean |
| --- | ---: | ---: |
| small 1obs | 0.907 | 0.927 |
| medium 1obs | 0.947 | 0.953 |
| medium 2obs | 0.820 | 0.900 |
| medium 3obs | 0.727 | 0.840 |

总耗时：195.642 s

---

## 2. 结论

> 课程扩展到 2/3 障碍后仍然可训练，但 best 与 final 差距变大。

- small 1obs：best 0.927
- medium 1obs：best 0.953
- medium 2obs：best 0.900
- medium 3obs：best 0.840

final 下降：

~~~text
2obs：0.820
3obs：0.727
~~~

说明：

- best checkpoint 很重要
- 后期 PPO 仍会漂移
- 3 障碍仍可到 84% best
- 但最后一轮不稳定

---

## 3. 关键观察

1. 小范围→中等范围，迁移成功
2. 1→2 障碍，best 从 95% 掉到 90%
3. 2→3 障碍，best 从 90% 掉到 84%
4. 每增加一个障碍，性能下降约 5–6 个百分点
5. final 比 best 低约 8–12 个百分点

---

## 4. 下一步

1. 增加 updates 到 20，但保留 early stop
2. 验证 episodes 50 已够，下一步多 seed 5 个
3. 在 3 障碍上单独再跑一轮
4. 加障碍物课程：固定位置 → 小随机 → 中随机
5. 把 best checkpoint 作为下一阶段起点，而不是 final

---

## 5. 一句话

> 3 seeds 下，2 障碍 best 90%，3 障碍 best 84%；课程能扩展，但 final 不稳定，必须用 best checkpoint。

---

## 6. 文件

- bc_ppo_expand.py
- bc_ppo_expand_result.json
- BC_PPO_EXPAND_REPORT.md
