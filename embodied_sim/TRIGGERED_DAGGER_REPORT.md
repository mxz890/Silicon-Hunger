# 触发式 DAgger + 动态安全盾

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
新任务：1 随机障碍 + 目标 ±0.3  
专家：PD + 障碍斥力

---

## 1. 触发式 DAgger

不是固定轮次随机跑，而是：

~~~text
模型自己跑
→ 动作与专家差异 > 0.25
   或 离障碍物 < 0.4
→ 专家接管
→ 把纠错状态加入训练集
~~~

结果：

| 阶段 | success | steps |
| --- | ---: | ---: |
| BC baseline | 0.48 | 185.89 |
| Triggered DAgger | **0.88** | 123.86 |

样本：

~~~text
3,044
~~~

触发：

~~~text
2,810
~~~

总耗时：11.421 s

---

## 2. 动态安全盾

评估时：

~~~text
离障碍物 < 0.45
且 模型动作与专家差异 > 0.15
→ 使用专家动作
~~~

结果：

~~~text
shield success = 0.84
shield steps   = 125.20
~~~

---

## 3. 结论

> 触发式 DAgger 把新任务从 48% 拉到 88%。

- 静态 BC：48%
- 触发式 DAgger：88%
- 安全盾：84%

说明：

- “迷茫时专家介入”有效
- 纠错数据确实把模型从死胡同拉回来
- 安全盾略低，但更保守

---

## 4. 限制

- 只有 1 个 seed
- 1 个障碍
- 触发阈值是手调的
- 没有不确定性估计
- 没有离线 RL
- 没有空间特异性生长

---

## 5. 下一步

1. 触发阈值用 ensemble disagreement
2. 动态安全阈值按障碍距离分档
3. Offline RL：IQL / TD3+BC
4. 重要性评分生长/剪枝
5. 空间 gated 子网

---

## 6. 一句话

> 触发式 DAgger 把新任务 BC 48% 拉到 88%，安全盾 84%；第一板斧有效，第二板斧初版可用。

---

## 7. 文件

- triggered_dagger.py
- triggered_dagger_result.json
- mzx_triggered_dagger.pt
- TRIGGERED_DAGGER_REPORT.md
