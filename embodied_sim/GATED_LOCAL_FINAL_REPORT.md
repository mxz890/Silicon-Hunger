# I 收口：gated local subgraph

日期：2026-11-22  
基座：mzx_2obs_plus100.pt（2 障碍 +100）  
任务：2 随机障碍，目标 ±0.6  
结构：新节点只接旧 hidden，不扩 head  
gate：0.5  
贡献阈值：0.0001  
生长：3 轮 × 50  
Seeds：62100–62102

---

## 1. 固定结构

~~~text
主图 1000 节点
新节点只接旧 hidden 节点
不接新 output，不扩 head
gate 初始 0.5
只训练新节点
贡献 < 0.0001 拒绝
~~~

---

## 2. 结果

| seed | base | +50 r0 | +50 r1 | +50 r2 | best |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 62100 | 0.83 | 0.87 | 0.90 | 0.89 | **0.90** |
| 62101 | 0.84 | 0.90 | 0.89 | 0.85 | **0.90** |
| 62102 | 0.90 | 0.89 | 0.85 | 0.89 | 0.89 |

均值：

~~~text
base ≈ 0.8567
best ≈ 0.8967
final ≈ 0.8767
~~~

总耗时：67.323 s

---

## 3. 结论

> gated local subgraph 是有效的。

- base 85.67%
- best 89.67%
- final 87.67%

说明：

- 新节点接旧 hidden，而不是新 output，可提升
- 只训练新节点也能工作
- 贡献验收全部通过

---

## 4. 固定 I 结构

~~~text
local subgraph
gate 0.5
threshold 0.0001
grow 50 / round
3 rounds
old hidden target, no head expansion
~~~

这个结构可以冻结，作为 J 的输入。

---

## 5. J 当前

安全层：

~~~text
baseline 0.82
shield 0.91
~~~

Offline RL 还没实现。

---

## 6. 下一步

1. 用 I 的 gated local 模型做 J 安全层
2. J Offline RL（IQL / TD3+BC）
3. 安全层阈值消融

---

## 7. 一句话

> I 收口成功：gated local subgraph 从 85.67% 提升到 89.67% best；结构固定为 local + gate 0.5 + threshold 0.0001 + 50/round。

---

## 8. 文件

- gated_local_final.py
- gated_local_final_result.json
- mzx_gated_local_seed62100.pt
- GATED_LOCAL_FINAL_REPORT.md
