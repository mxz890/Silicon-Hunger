# MZX 2000 节点：2 障碍 + sim-to-sim（2 seeds）

日期：2026-11-22  
模型：MZX-0.1 / 2000 节点  
任务：2 个障碍，目标 ±0.6，起点 ±0.6  
方法：PD 避障专家 → BC → DAgger → PPO  
评估：100 episodes  
Seeds：40000 / 40001

---

## 1. 结果

| seed | expert | BC | DAgger | PPO best | PPO final | source | target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40000 | 0.96 | 0.85 | 0.83 | 0.88 | 0.79 | 0.77 | 0.81 |
| 40001 | 0.91 | 0.89 | 0.91 | 0.90 | 0.83 | 0.79 | 0.78 |

均值：

~~~text
source ≈ 0.78
target ≈ 0.80
~~~

总耗时：276.076 s

---

## 2. 结论

> 2000 节点没有崩，2 障碍下 source/target 都在 78%–81%。

- BC 85% / 89%
- DAgger 83% / 91%
- PPO best 88% / 90%
- PPO final 79% / 83%
- sim-to-sim target 81% / 78%

所以：

> 2000 节点 + 2 障碍可以训练；sim-to-sim 没有明显掉点。

---

## 3. 重要说明

- 只有 2 seeds，不是 5 seeds
- 2 障碍，不是 3
- target 是 ±0.6
- target domain 改颜色/阻尼，不是完全不同物理
- source/target 都 100 episodes

---

## 4. 下一步

1. 补到 5 seeds
2. 3 障碍
3. target 范围继续扩大
4. 更强 sim-to-sim 差异
5. 2000 节点本地推理延迟测试

---

## 5. 一句话

> 2000 节点、2 障碍：source 78%，target 80%；sim-to-sim 没有明显掉点，但只有 2 seeds。

---

## 6. 文件

- mzx_2000_sim2sim.py
- mzx_2000_sim2sim_result.json
- mzx_2000_seed40000.pt
- mzx_2000_seed40001.pt
- MZX_2000_SIM2SIM_REPORT.md
