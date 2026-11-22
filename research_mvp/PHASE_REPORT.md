# Silicon Hunger — 灾难 / 功能冲突 / 拓扑突变实验报告

日期：2026-11-20  
范围：3D 空间约束 + Catastrophe + Functional Conflict Split + secrets 突变 + 复杂任务测试。

---

## 0. 三个补丁

| 补丁 | 实现 |
| --- | --- |
| 环境剧变 | 第 50 轮切换极端 7D 向量，并把 prune_after 临时降到 3 |
| 功能冲突分裂 | 不再使用 degree > 3；A/B 同向反馈 + 入边权方差过大才分裂 |
| 量子随机突变 | 每 10 轮 secrets 选节点，切断主边或加跨区长边 |

运行：

~~~powershell
.\silicon_env\Scripts\python.exe research_mvp\phase0_visualization.py --cycles 100 --render-every 10 --seed 11
.\silicon_env\Scripts\python.exe research_mvp\phase1_development.py --cycles 200 --seed 11
.\silicon_env\Scripts\python.exe research_mvp\phase2_learning.py
.\silicon_env\Scripts\python.exe research_mvp\phase2_tasks.py
~~~

---

## 1. 三维空间约束

每个节点：

~~~text
pos = [x, y, z]
0 <= x, y, z <= 100
~~~

所有 `_add_edge`：

~~~text
distance > 30.0 -> reject
~~~

被拒绝边记录在 `blocked_edges`。

### 近邻重组

`prune` 会按被删节点的 3D 位置排序存活节点，把原连接优先转给最近的 3 个存活节点。

空间检查：

~~~text
seed=3
pos_min=18.534
pos_max=81.512
max_edge_distance_before=29.733
max_edge_distance_after=29.733
rewires=5
~~~

---

## 2. Catastrophe：第 50 轮环境剧变

实现：

~~~text
step 50:
  env = [0, 1, 0, 1, 0, 1, 0]
  prune_after = 3
  activation_threshold = 0.68

step 70:
  prune_after = 20
  activation_threshold = 0.50
~~~

Phase 0 日志：

~~~text
CATASTROPHE_START step=50 env=[0,1,0,1,0,1,0]
step=050 nodes=20 edges=72 max_degree=18 hub=13 hub_degree=9 deaths=5
step=060 nodes=15 edges=54 max_degree=14 hub=16 hub_degree=9 deaths=13
CATASTROPHE_END step=70
step=070 nodes=17 edges=63 max_degree=16 hub=16 hub_degree=9 deaths=13
step=100 nodes=25 edges=93 max_degree=24 hub=22 hub_degree=16 deaths=13
~~~

Phase 0 最终：

~~~text
nodes=25
edges=93
hub=22 hub_degree=16
deaths=13
births=28
rewires=12
mutations=6
blocked_edges=3
~~~

Phase 1 日志：

~~~text
CATASTROPHE_START step=50
step=040 nodes=18 edges=66 hub=6 hub_degree=8
step=060 nodes=16 edges=59 hub=22 hub_degree=11
CATASTROPHE_END step=70
step=080 nodes=20 edges=76 hub=22 hub_degree=15
step=100 nodes=25 edges=94 hub=22 hub_degree=19
step=200 nodes=25 edges=89 hub=22 hub_degree=18
~~~

Phase 1 最终：

~~~text
nodes=25
edges=89
max_degree=21
hub=22 hub_degree=18
births=27
deaths=12
rewires=22
blocked_edges=5
mutations=14
~~~

结论：

> 灾难后节点从 18 降到 16，然后重新长回 25；死亡时钟删掉了旧结构，近邻重组重新接回了 22 条边。

---

## 3. Functional Conflict Split

不再使用：

~~~text
degree > 3
~~~

改为检测两个冲突环境：

~~~text
A = [0,1,0,1,0,1,0]
B = [1,0,1,0,1,0,1]
~~~

对每个 hidden 节点：

~~~text
act_a = activation(A)
act_b = activation(B)
feedback_a = 2*act_a - 1
feedback_b = 2*act_b - 1
~~~

触发条件：

1. A/B 的 feedback 同向
2. 两个方向都足够强
3. 入边权重方差 > 0.015
4. 节点对输出的数值敏感度 > 1e-4

触发后新增节点：

~~~text
parent -> child
input -> child
child -> output
~~~

并且 child 位置在 parent 附近，仍受 30 距离限制。

Phase 1 中出现了多个 `functional_conflict` 分裂。例子：

~~~text
node 11 score=0.0361 act_a=0.724 act_b=0.623
node 22 score=0.1036 act_a=0.753 act_b=0.831
~~~

这说明分裂不再由“连接多”触发，而是由“功能方向打架 + 权重方差”触发。

---

## 4. secrets 拓扑突变

每 10 轮：

~~~text
node = secrets.choice(alive_hidden)
action = secrets.choice(["cut", "long"])
~~~

两种突变：

- cut：切断该节点权重最大的 incident edge
- long：连接一个距离 > 30 的远端节点，故意绕过空间限制

Phase 1 的突变例子：

~~~text
step 40 cut 0 -> 5 weight=0.7755
step 100 long 5 -> 7 distance=31.619
step 130 long 3 -> 34 distance=32.647
~~~

最终：

~~~text
mutations=14
~~~

注意：long edge 是 mutation 特权；普通 `_add_edge` 仍然必须 <= 30。

---

## 5. 三维可视化

Phase 0 输出两套：

- 2D pyvis：`visualizations/step_010.html ... step_100.html`
- 3D Plotly：`visualizations/step_3d_010.html ... step_3d_100.html`

索引：

~~~text
visualizations/index.html
~~~

3D 使用 Plotly：

- 节点 `x/y/z` 直接来自 pos
- Plotly 3D Scatter3d
- 本地 `plotly.min.js`，不需要 CDN
- `dragmode=orbit`：左键拖动自由旋转
- `scrollZoom=true`：滚轮自由缩放
- `doubleClick=reset`：双击重置
- 自动旋转已关闭，视图完全由用户控制
- 新节点绿色
- 被剪节点红色 ghost
- 普通节点蓝色

> pyvis / vis-network 本身是 2D，所以 3D 改用 Plotly；pyvis 2D 保留为对照。

---

## 6. Phase 2：原有任务

从 Phase 1 的 25 节点 / 89 边图开始。

XOR：

~~~text
final_loss=0.019896
final_accuracy=1.0
local growth=1
final nodes=26
final edges=92
~~~

说明：

> 灾难后的图仍然可以学 XOR；这次需要一次输出附近的局部生长。

---

## 7. 复杂任务测试

代码：phase2_tasks.py

### XOR 基线

~~~text
loss=0.019913
acc=1.0
nodes=26
~~~

### 3 位奇偶校验

- 输入：节点 0, 1, 3
- 8 个 3-bit 样本
- 目标：a XOR b XOR c
- 节点 2 已在灾难中被剪掉，所以第三个输入使用节点 3（仍有到 output 的路径）

结果：

~~~text
loss=0.124998
acc=0.625
nodes=27
local growth=2
verdict=FAILED
~~~

### 简单序列记忆 / delayed match

- 输入：节点 0, 1
- 任务：两个输入相同则输出 1，不同则输出 0（XNOR）
- 这是最小的工作记忆 / 延迟匹配测试

结果：

~~~text
loss=0.019949
acc=1.0
nodes=26
local growth=1
verdict=PASSED
~~~

结论：

> 现有拓扑可以学 2-bit 延迟匹配，但不能稳定学 3-bit parity；局部生长不足以补出第三个输入的专用子结构。

---

## 8. 总结

| 项目 | 结果 |
| --- | --- |
| 3D 坐标 | 0–100，存入 pos |
| 空间限制 | 30.0 |
| 普通边越界 | 拒绝 |
| mutation long edge | 允许，记录 bypass_spatial |
| Catastrophe | 第 50 轮切换极端输入 + 激进修剪 |
| 灾难后死亡 | Phase 1 共 12 个 hidden 被删 |
| 近邻重组 | Phase 1 产生 22 条转移边 |
| 功能冲突分裂 | 出现，reason=functional_conflict |
| secrets 突变 | 14 次 |
| 3D HTML | step_3d_010 ... step_3d_100 |
| XOR | 通过（局部生长 1 次） |
| 3-bit parity | 失败（acc=0.625） |
| delayed match | 通过（acc=1.0） |

一句话：

> 灾难把旧结构打碎，近邻重组和功能冲突分裂在废墟上重建了 25 个节点；3D 空间约束与 secrets 突变同时生效。

---

## 9. 产物

- graph_core.py
- phase0_visualization.py
- phase1_development.py
- phase2_learning.py
- phase2_tasks.py
- spatial_checks.py
- phase0_log.json
- phase1_log.json
- phase1_graph.json
- phase2_result.json
- phase2_tasks_result.json
- spatial_check.json
- visualizations/index.html
- visualizations/plotly.min.js
- visualizations/step_3d_*.html
