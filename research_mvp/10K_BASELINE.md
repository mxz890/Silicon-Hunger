# 10K Research Baseline

日期：2026-11-20  
状态：正式研究基线。  
原则：以后默认用 10000 节点；2000 / 37 节点只作为历史和消融。

---

## 1. 基线定义

| 项 | 值 |
| --- | --- |
| 节点 | 10000 |
| 边 | 40105 |
| 输入 | 节点 0, 1 |
| 输出 | 9999 |
| output indegree | 120 |
| 输入维度 | 64 |
| 特征维度 | 9998 |
| 图类型 | 3D 稀疏 DAG |
| 时间 | 无 |
| 化学场 | 6 个，静态扩散后固定 |
| 训练 | 每任务独立 ridge readout |

代码：

- scale_10k_fresh.py
- scale_10k_richer_chinese.py

---

## 2. 基线结果

Lu Xun 语料：

| 任务 | 指标 | 结果 |
| --- | --- | ---: |
| 数学 | MAE | 0.31333 |
| 数学竞赛题 | 预测 | 25.44105 |
| 中文 | next-char acc | 0.0700 |

更丰富白话文语料：

| 任务 | 指标 | 结果 |
| --- | --- | ---: |
| 数学 | MAE | 0.20834 |
| 数学竞赛题 | 预测 | 25.14509 |
| 中文 | next-char acc | 0.0775 |
| 总耗时 | — | 91.061 s |
| 单样本 | — | 45.531 ms |

---

## 3. 基线文件

- scale_10k_fresh.py
- scale_10k_fresh_result.json
- SCALE_10K_FRESH_REPORT.md
- scale_10k_richer_chinese.py
- scale_10k_richer_result.json
- SCALE_10K_RICHER_REPORT.md
- data/modern_chinese_corpus.txt
- data/SOURCES.md

---

## 3.5 大规模数学语料

- 语料：557,698 字 / 10,000 题
- 图：fresh 10k / 40105 边
- 每任务：1600 训练 / 400 测试
- 结果：

| 任务 | 相对 MAE |
| --- | ---: |
| sum_sq | 0.097% |
| linear | 4.613% |
| rectangle | 0.060% |
| arithmetic | 0.144% |
| hypotenuse | 0.032% |

- 耗时：431.165 s
- 平均：43.1 ms / 样本

文件：MATH_500K_REPORT.md

---

## 4. 对照，不是基线

| 版本 | 用途 |
| --- | --- |
| seed-grown 10k | 看“2000 长到 10k”是否更好 |
| fair 10k | 控制 output 入度/边数后的公平对比 |
| 2000 节点 | 历史 PoC / 快速消融 |
| 37 节点 parity | 神经递质/化学分支 |
| temporal 2000 | 时间版本 PoC，尚未扩到 10k |

---

## 5. 基线限制

- 无时间/循环
- 图权重固定
- 只训练 readout
- 中文 next-character 仍接近随机水平
- 高难线性代数基本失败
- 化学场不是主要性能来源
- 10k 前向约 40–45 ms/样本

---

## 6. 当前研究优先级

1. 以 10k fresh 为默认基线
2. 只在 10k 上做新实验
3. 2000 只用于快速验证
4. 时间版本先保留在 2000，确认有效后再扩 10k
5. 所有新结果写入 RESULTS_INDEX.md

---

## 7. 复现

~~~powershell
cd "C:\Users\MENG XAINGZE\Desktop\DEMO"

.\silicon_env\Scripts\python.exe research_mvp\scale_10k_fresh.py
.\silicon_env\Scripts\python.exe research_mvp\scale_10k_richer_chinese.py
~~~

---

## 8. 一句话

> 10k fresh 是当前研究基线；2000 是 PoC，37 是分支，fair/seed-grown 是对照。
