# 化学扩散 GPU 基准测试

日期：2026-11-20  
设备：NVIDIA GeForce RTX 5070 Ti 16GB  
环境：torch 2.11.0+cu128

---

## 1. 测试内容

- 图：2000 节点 / 9022 边
- 化学场：6 个
- 扩散步数：20
- dt：0.08
- CPU：NumPy bincount
- GPU：torch index_add_，C、边、source 常驻显存
- GPU 只把最终 gain 拷回 CPU

代码：

~~~powershell
.\silicon_env\Scripts\python.exe research_mvp\gpu_chem_benchmark.py
~~~

---

## 2. 结果

| 路径 | 中位时间 | 相对 CPU |
| --- | ---: | ---: |
| CPU | 5.976 ms | 1.00x |
| GPU persistent | 3.862 ms | 1.55x |
| GPU cold（含传输） | 3.992 ms | 1.50x |
| GPU 每步 synchronize | 4.055 ms | 1.47x |

原始时间：

~~~text
CPU: [86.561, 6.187, 5.658, 5.858, 5.976] ms
GPU persistent: [208.683, 4.576, 3.418, 3.804, 3.862] ms
GPU cold: [4.205, 3.667, 3.759, 3.991, 4.251] ms
GPU sync each: [4.646, 3.915, 4.055] ms
~~~

第一项是首次运行/预热，中位数排除了它。

数值一致性：

~~~text
max_concentration_diff = 0.0
max_gain_diff = 0.0
~~~

CPU 和 GPU 的结果完全一致。

---

## 3. 怎么解释

GPU 只快了约 1.5 倍，不是 10 倍。原因：

- 2000 节点 / 9022 边太小
- 20 步 × 6 个场，计算量很小
- GPU 时间主要花在 kernel launch 和同步
- 即使 C 常驻，仍然有 Python/torch 调度开销

结论：

> 当前规模下，把化学扩散搬到 GPU 有收益，但收益只有约 1.5 倍，不值得为此重构整个训练循环。

---

## 4. 功耗要单独看

这次没有用功率计，只测了延迟。  
RTX 5070 Ti 空闲时 nvidia-smi 显示约 56 W；负载功耗会更高。

所以：

> GPU 更快 ≠ GPU 更省电。

在 4 ms 的任务上，GPU 的固定功耗和唤醒成本可能吃掉收益。  
要判断能效，需要实际测 wall power，不能只看 ms。

---

## 5. 什么时候值得上 GPU

值得：

- 节点数到 1 万 / 10 万
- 边数到几十万
- 扩散步数很多
- 同时批量跑很多图
- 整个 reservoir + readout 都在 GPU 上

不值得：

- 单个 2000 节点图
- 只算 20 步扩散
- 每步都 CPU/GPU 同步
- 只把化学场放 GPU，其他还在 CPU

---

## 6. 正确集成方式

不要：

~~~text
CPU graph -> GPU diffusion -> CPU gain -> CPU reservoir
~~~

应该：

~~~text
GPU: C + adjacency + edge weights
GPU: 20 步 diffusion
GPU: gain
只把最终 gain 或最终结果拷回 CPU
~~~

更彻底：

~~~text
整个 2000 节点 reservoir 前向也在 GPU 上
~~~

这样才不用每步跨系统。

---

## 7. 建议

当前 2000 节点版本：

> 保持 CPU 化学扩散。

理由：

- 6 ms 已经足够快
- GPU 只有 1.5x
- 重构成本和同步复杂度不划算

等图规模到 10k+，或者开始批量训练/批量推理时，再切 GPU。

---

## 8. 文件

- gpu_chem_benchmark.py
- gpu_chem_result.json
- GPU_CHEM_REPORT.md

---

## 9. 一句话

> RTX 5070 Ti 上，2000 节点 / 9022 边 / 6 化学场 / 20 步扩散，GPU 从 5.98 ms 降到 3.86 ms，约 1.55 倍；结果完全一致，但当前规模还不值得全面迁移。
