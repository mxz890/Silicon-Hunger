# J 初版：2 障碍安全层（Safety Shield）

日期：2026-11-22  
模型：mzx_2obs_plus100.pt（2 障碍 +100 基线）  
任务：2 随机障碍，目标 ±0.6  
评估：100 episodes

---

## 1. 设置

当最近障碍距离：

~~~text
d_obs < 0.45
~~~

直接使用专家动作：

~~~text
PD + 斥力 + 切向力
~~~

不是纯 PPO，是 PPO 策略 + 安全层。

---

## 2. 结果

| 模式 | success | steps |
| --- | ---: | ---: |
| baseline | 0.82 | 137.47 |
| safety shield | **0.91** | 132.44 |

触发次数：

~~~text
3,077
~~~

总耗时：7.735 s

---

## 3. 结论

> 安全层把 2 障碍从 82% 提到 91%。

- baseline：82%
- shield：91%
- 提升 9 个百分点

---

## 4. 说明

这不是 Offline RL。

它是：

~~~text
PPO/BC 策略
+ 距离触发的专家安全层
~~~

Offline RL 还没实现。

---

## 5. 下一步

1. I：gated 局部子图结构收口
2. J：真正 Offline RL（IQL / TD3+BC）或安全 PPO 训练
3. 安全层阈值做消融

---

## 6. 一句话

> 2 障碍 +100 基线 82%，安全层 91%；这是 J 的安全层初版，不是 Offline RL。

---

## 7. 文件

- safety_2obs.py
- safety_2obs_result.json
- mzx_2obs_safety.pt
- SAFETY_2OBS_REPORT.md
