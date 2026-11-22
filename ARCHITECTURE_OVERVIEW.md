# Silicon Hunger — Architecture Overview

> 一页纸版本，用于和别人讨论。
> 当前正式基线：**10k fresh graph**；7D/LLM 与图网络是两条线，Graph→LLM 只做到最小桥接。

---

## 1. 一句话

**一个固定的 3D 稀疏图 reservoir，用 6 种神经递质调制节点增益，输出高维激活，再由每个任务独立的线性 readout 解码；另有 7D 状态机用于调制 LLM 行为。**

不是 Transformer，不是标准 GNN，也不是 LLM。

---

## 2. 两条主线

```text
A. 7D 状态机 → LLM
   E/A/L/C/B/S/P → GRU → prompt / sampling / activation steering

B. 10k 图网络 → readout
   input → 3D DAG → chemical gain → h[2:] → ridge readout → answer
```

目前两者只在最小桥上连接：

```text
10k graph → 128-d state → 8 soft tokens → Qwen2.5-3B
```

这个桥能跑，但还没有证明图状态带来稳定增量。

---

## 3. 主结构：10k graph

```text
x (64-d)
  │
  ▼
input_gain [10000 × 64]
  │
  ▼
┌──────────────────────────────────────┐
│ 10000-node 3D sparse DAG             │
│                                      │
│ h_i = tanh(                          │
│   g_i * (                            │
│     b_i                              │
│     + u_i · x                        │
│     + Σ_j w_ji h_j                   │
│   )                                  │
│ )                                    │
│                                      │
│ 6 chemical fields:                   │
│ DA / 5HT / NE / ACh / GABA / Glu     │
│ → diffusion → g_i                    │
└──────────────────────────────────────┘
  │
  ▼
h[2:] = 9998-d features
  │
  ▼
per-task ridge readout
  │
  ▼
answer
```

### 基线参数

| 项 | 值 |
|---|---:|
| nodes | 10000 |
| edges | 40105 |
| input | 0, 1 |
| output | 9999 |
| output indegree | 120 |
| input dim | 64 |
| feature dim | 9998 |
| time | 无 |
| chemical | 6，静态扩散后固定 |
| training | 每任务独立 ridge readout |

---

## 4. 节点和边

### 节点

```text
h_i = tanh(g_i * (b_i + u_i·x + Σ_j w_ji h_j))
```

- `b_i`：节点 bias
- `u_i`：64 维 input gain
- `w_ji`：入边权重
- `g_i`：化学增益

### 边

- 新节点连接最近的 4 个前驱
- 3D 距离约束通常 ≤ 30
- output 接收全局汇聚边

### 化学

```text
exc = 0.8*Glu + 0.6*DA + 0.4*NE + 0.3*ACh
inh = 0.9*GABA + 0.4*5HT

g_i = 1 + 0.8*exc/(1+exc) - 0.45*inh/(1+inh)
```

---

## 5. 时间版本

目前只在 2000 节点上验证：

```text
h_i(t+1) =
    (1-α) h_i(t)
    + α tanh(
        g_i(t) * (
            b_i
            + u_i x(t)
            + Σ_j w_ji h_j(t)
        )
    )
```

```text
α = 0.22
T = 12
chemical update every 2 ticks
2000 base edges + 2000 recurrent edges = 11022
```

结果：

| 任务 | test acc |
|---|---:|
| delayed match | 0.9733 |
| temporal parity | 0.510 |

---

## 6. 7D 状态机

```text
E  energy
A  anxiety
L  load
C  curiosity
B  boredom
S  entropy
P  pain
```

```text
state_t → 2-layer GRU → Δstate
```

Version A：

```text
7D → prompt telemetry / sampling / refusal filter
model: gpt-oss:20b
```

Version B：

```text
7D → soft prompt or layer-12 activation steering
model: Qwen2.5-3B base
```

结论：

> 7D 能改变长度/采样/拒绝；不能稳定改变语义人格。

---

## 7. Graph → LLM 桥

```text
prompt
→ char hash 64
→ 10k graph
→ h[2:] 9998
→ fixed random projection 128
→ 8 soft tokens
→ Qwen2.5-3B inputs_embeds
```

两组实验：

1. 随机 adapter：能注入，但输出不稳定，会退化。
2. 训练 adapter：8 训练 / 4 测试
   - graph test loss 2.3574
   - constant test loss 2.2729
   - graph 没有优势

结论：

> 桥能通，但没有证明图状态给 LLM 带来增量。

---

## 8. 当前结果

| 实验 | 结果 |
|---|---|
| 10k fresh 数学 | MAE 0.31333 |
| 10k richer 数学 | MAE 0.20834 |
| 10k fresh 中文 | 7.0% |
| 10k richer 中文 | 7.75% |
| 557k 字数学 | 4/5 题型相对 MAE < 0.2%；linear 4.6% |
| 高难线代 | 基本失败；eigen2 11.8% |
| 37 节点 parity | baseline 0.50 / chem 0.75（单次） |
| 动态化学边 | 数学 0.244 → 0.104；中文 0.105 → 0.080 |
| credit vs broadcast | broadcast 更好 |
| 时间 2000 | delayed 0.9733 / parity 0.510 |

---

## 9. 关键限制

- 图权重基本固定，只训练 readout
- 没有全图 BPTT
- 没有真正的语言预训练
- 中文 next-character 仍很弱
- 高难线性代数失败
- 化学场不是主因
- 图与 LLM 只在最小桥上连接
- 时间版本还没扩到 10k

---

## 10. 讨论时最值得问的问题

1. **固定随机 reservoir + 线性 readout，理论上还能走多远？**
2. **图状态到底有没有比 constant soft prompt 更多的信息？**
3. **生长/修剪应该由什么驱动：活动、奖励、预测误差，还是冲突？**
4. **化学场应该全局广播，还是按贡献度局部释放？**
5. **要不要放弃 graph-only，把图作为 LLM 的外部状态/记忆？**
6. **具身智能是否是更合适的落点：500–2000 节点本地控制，而不是 10k 通用模型？**

---

## 11. 文件

- 结果索引：`research_mvp/RESULTS_INDEX.md`
- 10k 基线：`research_mvp/10K_BASELINE.md`
- 数学 557k：`research_mvp/MATH_500K_REPORT.md`
- 时间版本：`research_mvp/TEMPORAL_RESERVOIR_REPORT.md`
- Graph → LLM：`GRAPH_TO_LLM_REPORT.md`
- 训练 adapter：`GRAPH_LLM_TRAINED_REPORT.md`
- 动态化学：`research_mvp/DYNAMIC_CHEM_EDGES_REPORT.md`
- credit 对照：`research_mvp/CREDIT_CHEM_ABLATION_REPORT.md`

---

## 12. 一句话定位

> **轻量、可解释、可本地部署的神经调质图 reservoir 原型；不是新的大模型架构，也不是已经验证的通用智能。**
