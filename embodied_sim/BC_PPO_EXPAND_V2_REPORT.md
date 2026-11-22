# BC → PPO 扩展 v2：5 seeds、20 updates、障碍课程

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
设置：8 envs × 256 steps，20 updates，验证 50 episodes  
保护：KL 0.20 → 0.02，best checkpoint，early stop  
Seeds：32000–32004

---

## 1. 课程

~~~text
fixed_1obs
→ small_1obs
→ medium_1obs
→ medium_2obs
→ medium_3obs
→ three_extra
~~~

每阶段以上一阶段 best checkpoint 为起点。

---

## 2. 结果

| seed | 3obs best | 3obs final | three_extra best |
| ---: | ---: | ---: | ---: |
| 32000 | 0.80 | 0.76 | 0.84 |
| 32001 | 0.62 | 0.52 | 0.54 |
| 32002 | 0.34 | 0.28 | 0.32 |
| 32003 | 0.28 | 0.28 | 0.34 |
| 32004 | 0.24 | 0.14 | 0.20 |

均值：

~~~text
3obs best mean  ≈ 0.456
3obs final mean ≈ 0.396
extra best mean ≈ 0.448
~~~

总耗时：722.227 s

---

## 3. 结论

> 5 seeds 下方差极大，只有 seed 32000 表现好。

seed 32000：

~~~text
3obs best 0.80
extra best 0.84
~~~

其他 seed：

~~~text
0.24–0.62
~~~

说明：

- BC checkpoint 本身对方差很敏感
- PPO 在差初始化上无法恢复
- best checkpoint 只能保住当前 seed 的最好结果
- 20 updates 没有解决 seed 依赖

---

## 4. 关键问题

### 4.1 BC 起点不稳定

在同一固定任务上，不同 seed 的 BC 初始 success：

~~~text
0.78 / 0.70 / 0.26 / 0.18 / 0.20
~~~

差异巨大。

### 4.2 PPO 无法从差起点恢复

差 seed 进入 medium 后继续下降。

### 4.3 early stop 会过早停

差起点上验证不提升，很快早停。

---

## 5. 这说明什么

不能说明：

- 课程无效
- PPO 无效

只能说明：

> 当前 BC checkpoint 和 5 seed 训练对初始化太敏感，需要先稳定 BC。

---

## 6. 下一步

1. 先让 BC 在多 seed 下稳定 >80%
2. 固定 BC 初始化，减少 seed 方差
3. 先做 1 障碍 5 seeds，不要直接扩 3
4. 增加验证 episodes
5. PPO 前先做 BC 验证筛选

---

## 7. 一句话

> 5 seeds 下只有 seed 32000 在 3 障碍上到 80–84%，其余只有 24–62%；当前瓶颈不是课程，而是 BC 初始化方差。

---

## 8. 文件

- bc_ppo_expand_v2.py
- bc_ppo_expand_v2_result.json
- BC_PPO_EXPAND_V2_REPORT.md
