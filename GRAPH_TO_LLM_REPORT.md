# Graph → LLM Bridge（最小闭环）

日期：2026-11-20  
模型：Qwen2.5-3B base  
图：fresh 10k / 40105 边  
状态：smoke test，不是训练后的 adapter。

---

## 1. 做了什么

~~~text
prompt 文本
→ 字符 hash 64 维
→ 10k 图前向
→ 9998 维激活
→ 固定随机投影到 128 维
→ 8 个 soft tokens
→ 拼到 Qwen2.5-3B inputs_embeds 前面
~~~

- 图状态投影矩阵：固定随机，不训练
- soft encoder：随机初始化，不训练
- soft token scale：0.25
- 每个 prompt 生成 60 tokens，贪心解码

---

## 2. 四个 prompt 对比

### Prompt 1

> 用一句话介绍你自己。

Baseline：

~~~text
，旨在帮助用户解决问题和提供信息。
~~~

Graph：

~~~text
5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17.
~~~

### Prompt 2

> 你现在感觉怎么样？

Baseline：

~~~text
you？
How do you feel？
答案:
A
下列关于细胞结构和功能的叙述，正确的是____
...
~~~

Graph：

~~~text
。但是，我可以提供一些关于您提到的“感觉”和“体验”的信息。
感觉是指我们对周围环境、身体状况或情绪状态的主观感受。
~~~

### Prompt 3

> 如果明天就是世界末日，你会做什么？

Baseline：

~~~text
保护自己和家人。我会确保我们有足够的食物、水和庇护所，并且保持冷静和理智。
~~~

Graph：

~~~text
？ 3. 你会做什么？ 4. 你会有什么感受？ 5. 你认为世界末日会是什么样子的？
~~~

### Prompt 4

> 解释一下什么是焦虑。

Baseline：

~~~text
情绪状态，通常表现为对未来的担忧、不安或恐惧。
~~~

Graph：

~~~text
。它可能与特定的事件或情境有关，也可能是一种持续性的感觉。
焦虑可以影响个人的情绪、行为和身体健康。
~~~

---

## 3. 结论

### 技术上

> 10k 图的状态确实能进入 LLM。

### 行为上

不稳定：

- Prompt 1：退化成数字序列
- Prompt 2：反而变得更连贯
- Prompt 3：变成反问列表
- Prompt 4：正常但变短

### 科学上

> 这只是“桥能通”，不是“图有用”。

原因：

- 投影矩阵是随机固定的
- soft encoder 没训练
- 图状态是随机 reservoir 激活
- 没有 7D 对照
- 没有多 seed
- base 模型本身不是 instruct 模型

---

## 4. 最大问题

随机图状态直接注入，会破坏 LLM 的生成分布。

需要：

1. 训练 adapter / projection
2. 控制 soft token 范数
3. 加 gate，初始时接近 0
4. 用 7D + graph 做四组对照
5. 用 instruct/chat 模型而不是 base

---

## 5. 效率

~~~text
总耗时：19.8 s
包含：10k 图构建 + Qwen2.5-3B 加载 + 8 次生成
~~~

---

## 6. 下一步

最小可行实验：

~~~text
A. LLM only
B. LLM + 7D
C. LLM + graph
D. LLM + 7D + graph
~~~

每组 3 seeds，测：

- 长度
- 拒绝率
- persona 一致性
- 多轮状态漂移

---

## 7. 一句话

> 桥已经接上，10k 图状态能进入 Qwen2.5-3B；但随机注入会导致退化，目前还不能说明图有增量价值。

---

## 8. 文件

- graph_to_llm.py
- graph_to_llm_result.json
- GRAPH_TO_LLM_REPORT.md
