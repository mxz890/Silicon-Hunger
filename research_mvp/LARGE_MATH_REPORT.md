# 2000 节点空间图 + 初中数学竞赛题测试

日期：2026-11-20  
任务：把图扩大到 2000 个节点，并测试一个初中代数竞赛题。

---

## 1. 题目

训练任务族：

> 已知 a + b = S，ab = P，求 a² + b²。

由恒等式：

~~~text
a² + b² = (a + b)² - 2ab = S² - 2P
~~~

最终测试题：

> **已知 a + b = 7，ab = 12，求 a² + b²。**

答案：

~~~text
7² - 2×12 = 49 - 24 = 25
~~~

---

## 2. 图结构

- 节点：2000
- 输入：节点 0, 1
- 输出：节点 1999
- 隐藏节点：2–1998
- 边：9022
- 3D 坐标：沿随机游走生成
- 连接：每个节点连接最近的 4 个前驱，并偶尔增加一条局部跳边
- 最大边距离：16.288
- 构建耗时：0.043 秒

图不是全连接；它是一个稀疏的 3D 发育图。

---

## 3. 化学场

使用 6 个化学场：

| 递质 | 功能 |
| --- | --- |
| DA | 奖励 / 动机 |
| 5HT | 稳定 / 抑制 |
| NE | 唤醒 / 误差 |
| ACh | 注意 / 可塑性 |
| GABA | 抑制 |
| Glu | 兴奋 |

扩散方程：

~~~text
dc_i/dt = D * sum_j w_ij(c_j - c_i) + source_i - decay_i c_i
~~~

化学场均值：

~~~text
DA   = 0.0414
5HT  = 0.0455
NE   = 0.0388
ACh  = 0.0363
GABA = 0.0426
Glu  = 0.0387
~~~

最终 gain 范围：

~~~text
1.037 .. 1.195
~~~

---

## 4. 训练方式

这是一个 reservoir + 线性 readout：

1. 每个节点有自己的输入增益、bias、化学 gain
2. 输入 (S, P) 注入节点 0, 1
3. 信号沿 3D 稀疏图前向传播
4. 取 1998 个隐藏节点激活作为特征
5. 用 ridge regression 训练输出层

为什么不用全图反向传播：

- 2000 节点 / 9022 边全图 BPTT 在纯 Python 里会很慢
- 这个测试先验证：**2000 节点的空间图是否能形成可用的非线性特征**
- readout 仍然是可训练的，只是训练放在输出层

---

## 5. 运行日志

命令：

~~~powershell
cd "C:\Users\MENG XAINGZE\Desktop\DEMO"

.\silicon_env\Scripts\python.exe research_mvp\large_math_2000.py
~~~

日志：

~~~text
GRAPH nodes 2000 edges 9022 build_sec 0.043
MAX_EDGE_DISTANCE 16.288
CHEM_MEAN {'DA': 0.0414, '5HT': 0.0455, 'NE': 0.0388, 'ACh': 0.0363, 'GABA': 0.0426, 'Glu': 0.0387}
GAIN_RANGE 1.037 1.195
DATA train 900 test 300 features 1998
MAE 0.2396 RMSE 0.3491
PROBLEM a+b=7, ab=12 -> target 25, prediction 24.6395
TIME build 0.043 features 10.896 fit 0.956
~~~

---

## 6. 结果

| 指标 | 结果 |
| --- | ---: |
| 测试集 MAE | 0.2396 |
| 测试集 RMSE | 0.3491 |
| 竞赛题目标 | 25 |
| 竞赛题预测 | 24.6395 |
| 绝对误差 | 0.3605 |
| 相对误差 | 约 1.44% |

结论：

> 2000 节点的稀疏 3D 图 + 化学 gain，可以在这个代数题族上形成有效特征；  
> 最终竞赛题的预测是 24.6395，距离正确答案 25 约 1.44%。

---

## 7. 耗时

| 阶段 | 时间 |
| --- | ---: |
| 图构建 | 0.043 s |
| 1200 个样本前向特征 | 10.896 s |
| ridge 拟合 | 0.956 s |
| 总耗时 | 约 11.9 s |

这是纯 Python + NumPy，没有 GPU，没有 PyTorch。

---

## 8. 文件

- [large_math_2000.py](large_math_2000.py)
- [large_math_result.json](large_math_result.json)
- [LARGE_MATH_REPORT.md](LARGE_MATH_REPORT.md)

---

## 9. 说明

这次只测试了一个初中代数竞赛题族，没有跑其他任务。

结果不是“网络理解了代数”，而是：

> 2000 个节点的非线性空间图提供了足够丰富的随机特征；  
> 线性 readout 在这些特征上拟合出了 S² - 2P 这个关系。
