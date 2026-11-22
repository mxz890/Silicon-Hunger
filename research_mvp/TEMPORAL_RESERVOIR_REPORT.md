# 时间版本：2000 节点 leaky recurrent reservoir

日期：2026-11-20  
范围：给 2000 节点图加入时间、循环状态和慢化学场；先做 temporal parity 与 delayed match。

---

## 1. 结构

- 基础图：2000 节点 / 9022 边
- 额外随机 recurrent edges：2000 条
- 最终边：11022
- 时间步：T = 12
- leak rate：alpha = 0.22
- 化学更新：每 2 个 tick

节点更新：

~~~text
h_i(t+1) =
    (1 - alpha) * h_i(t)
    + alpha * tanh(
        gain_i(t) * (
            bias_i
            + input_gain_i * x(t)
            + sum_j w_ij h_j(t)
        )
    )
~~~

化学场不是每步全扩散，而是：

~~~text
if t % 2 == 0:
    chemical_step()
~~~

---

## 2. 任务

### temporal parity

- 12 个 bit
- 目标：所有 bit 的 XOR
- 训练 700 / 验证 200 / 测试 300

### delayed match

- t0 是 cue
- 中间是噪声
- t11 是 probe
- 目标：cue == probe

---

## 3. 结果

| 任务 | Val acc | Test acc |
| --- | ---: | ---: |
| temporal_parity | 0.535 | **0.510** |
| delayed_match | 0.985 | **0.9733** |

结论：

> 时间 reservoir 可以保留 cue，所以 delayed match 成功；  
> 但 12-bit 全序列 parity 基本失败，接近随机 0.5。

---

## 4. 可视化

时间轴：

~~~text
visualizations/temporal_timeline.html
~~~

展示：

- 每个 tick 的输入 bit
- 平均 |h|
- DA / GABA / Glu 化学均值

打开：

~~~text
http://127.0.0.1:8000/temporal_timeline.html
~~~

---

## 5. 效率

~~~text
两个任务：1200 sequences × 12 ticks
总耗时：9.662 s
~~~

因为只保留最后一个状态训练 readout，没有 BPTT。

---

## 6. 这版“完整”在哪里

现在第一次同时具备：

- 3D 图
- 循环状态
- 时间步
- leaky integration
- 动态化学场
- 慢化学时钟
- 序列任务
- readout 训练
- 时间轴可视化

仍然没有：

- BPTT
- 结构生长与时间同步
- 长序列
- 真正语言级序列建模

---

## 7. 关键结论

> 加入时间后，系统从“单次静态前向”变成了“有状态的动态系统”。  
> 它能做 delayed match，但不能做 12-bit parity。

这说明：

- 记忆存在
- 但容量/非线性/训练方式仍然有限
- 时间不是自动带来推理

---

## 8. 文件

- temporal_reservoir.py
- temporal_reservoir_result.json
- visualizations/temporal_timeline.html
- TEMPORAL_RESERVOIR_REPORT.md
