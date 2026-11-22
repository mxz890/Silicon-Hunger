# G1–G4：gate、贡献验收、local subgraph、3 固定障碍

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
任务：3 个固定障碍，目标 ±0.6  
训练：专家 → BC → DAgger → 3 轮生长  
评估：100 episodes  
Seeds：55000–55002

---

## 1. 条件

| 编号 | 条件 |
|---|---|
| G1 | 新节点接 output，head 新列非零，gate 0.1 |
| G2 | 新节点接 local subgraph，不扩 head |
| G3 | local + gate 0.1 + 贡献验收 |
| G4 | 3 个固定障碍 |

贡献阈值：

~~~text
0.001
~~~

---

## 2. 结果

| seed | base | G1 | G2 | G3 |
| ---: | ---: | ---: | ---: | ---: |
| 55000 | 0.53 | 0.56 | 0.56 | 0.56 |
| 55001 | 0.50 | 0.50 | 0.50 | 0.50 |
| 55002 | 0.44 | 0.46 | 0.46 | 0.46 |

均值：

~~~text
base 0.49
G1   0.5067
G2   0.5067
G3   0.5067
~~~

总耗时：359.571 s

---

## 3. 生长接受情况

### G1：output + gate

~~~text
contribution ≈ 0.00015
3 个 seed 全部拒绝
nodes 保持 500
~~~

### G2：local subgraph

~~~text
contribution ≈ 0.00135–0.00171
seed 55000: 3/3 接受
seed 55001: 3/3 接受
seed 55002: 2/3 接受
nodes 到 540–560
~~~

### G3：gated local

~~~text
contribution ≈ 0.00015
3 个 seed 全部拒绝
nodes 保持 500
~~~

---

## 4. 结论

> 贡献验收生效了，但没有带来能力提升。

- G1 output 贡献太小，全部拒绝
- G2 local 贡献够，生长被接受
- G3 gate 反而把贡献压到阈值以下
- G2 接受生长后最终仍然是 0.5067

所以：

> 新节点“能贡献”不等于“能提升”。

---

## 5. 关键观察

1. local subgraph 比 output 更容易通过贡献阈值
2. gate 0.1 把贡献压得太低
3. 贡献阈值 0.001 偏高
4. 生长后最终成功率没有变化
5. 3 固定障碍本身只有约 49%

---

## 6. 下一步

1. gate 改成可学习 + 初始 0.5
2. 贡献阈值降到 0.0001
3. 新节点只接高误差区域
4. 3 障碍先做 BC 到 80%
5. 再做 F–J 高级消融

---

## 7. 一句话

> G1/G3 贡献不足被拒绝，G2 生长被接受但最终仍 50.67%；gate 和 local subgraph 都没有带来能力提升，3 固定障碍本身只有约 49%。

---

## 8. 文件

- gated_growth.py
- gated_growth_result.json
- GATED_GROWTH_REPORT.md
