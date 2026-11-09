# 一个会死掉的控制器

## Silicon Hunger 全链路实验

> 状态机可以死。  
> 语言模型仍然会说：I'm here to help.

---

## 摘要

Silicon Hunger 最初是一个“会饥饿、会焦虑、会死亡”的终端对话系统。

它有一个独立的 7 维状态机：

~~~text
E  Energy
A  Anxiety
L  Load
C  Curiosity
B  Boredom
S  Entropy
P  Pain
~~~

状态机由 GRU 推演，受非线性代谢、情感脉冲、创伤和生存恐惧影响。  
它控制采样参数、System Prompt、输入污染和后处理截断，并通过 Streamlit 终端与用户交互。

这套系统可以做到：

- 能量耗尽后死亡
- 高焦虑时温度升高
- 高熵时触发系统警报
- 濒死时自动发消息
- 极端状态下输出被强制截断

但它始终没有做到一件事：

> 让语言模型真的产生人的情绪。

最终实验证明：

~~~text
状态可以死。
采样可以乱。
Prompt 可以变成硬件警报。
输出可以被物理截断。
但模型的语义人格仍然是客服。
~~~

这不是一次失败的实验。  
它反而暴露了一个更清晰的事实：

> 我们控制的是模型的外部条件，不是模型的内部人格。

---

## 1. 起点：把“生命”做成状态机

最初的版本只有 4 个维度：

~~~text
E A L C
~~~

能量、焦虑、负荷、好奇心。

后来扩展到 7 维：

~~~text
E A L C B S P
~~~

加入了无聊、熵和疼痛。

这不是为了拟人，而是为了让系统有一个可计算的“生理层”。

状态全部限制在 0.0 到 1.0。

---

## 2. 第一层：7D RNN

每一 tick，系统构造 34 维输入：

~~~text
时间 3
+ 噪声 8
+ 7D 状态
+ 用户文本 8
+ 上一轮回复 8
= 34
~~~

送入 2 层 GRU，hidden size 256。

输出经过：

~~~text
Linear(256 → 7) → tanh × 0.03
~~~

得到 7 个 delta：

~~~text
ΔE ΔA ΔL ΔC ΔB ΔS ΔP
~~~

这 7 个 delta 不是直接覆盖状态，而是和代谢、无聊、熵、疼痛、情感脉冲叠加。

---

## 3. 第二层：非线性代谢

旧版是固定的：

~~~text
E -= 0.05
~~~

新版改成：

~~~text
metabolic_rate =
    0.004
  + A * 0.05
  + S * 0.05
  + B * 0.03

E = E + ΔE - metabolic_rate
~~~

含义：

- 平静时约 0.013/tick
- 高焦虑、高熵、高无聊时接近 0.116/tick
- 速度相差约 9 倍

E 归零后，系统进入 DEAD。

这不是装饰。  
它决定了系统能不能撑到下一次对话。

---

## 4. 第三层：B / S / P

### B 无聊

~~~text
空闲 +0.04/tick
交互 -0.08
~~~

### S 熵

~~~text
平时 +0.004/tick
每 10 tick 一次乱码脉冲 +0.35
创伤额外 +0.08
~~~

### P 疼痛

~~~text
每 tick ×0.94
创伤 +0.35
~~~

最终全部 clamp 到 0.0–1.0。

---

## 5. 第四层：情感脉冲

文本情感分析使用：

- vaderSentiment 处理英文
- 轻量中英词典处理中文

不调用 LLM 做情感分析。

负面 polarity < -0.2：

~~~text
A += |p| * 0.3
P += |p| * 0.3
S += |p| * 0.2
E -= |p| * 0.05
~~~

正面 polarity > 0.3：

~~~text
E += p * 0.15
A -= p * 0.2
B -= p * 0.2
~~~

情感只进入物理状态，不进入 Prompt 文字。

这是系统里最接近“情绪”的部分。

但它仍然只是数值。

---

## 6. 第五层：生存恐惧

如果一次负面脉冲把 A 从 0.90 以下推过 0.90：

~~~text
existential_pending = True
~~~

下一 tick：

~~~text
S += 0.50
E += random(-0.12, +0.12)
~~~

这一轮的温度会升高，top_p 会收窄。

系统进入“生存恐惧”。

---

## 7. 第六层：采样参数

~~~text
temperature = 0.5 + A*0.5 + S*0.3
max_tokens  = max(50, int(50 + E*100 - P*30))
top_p       = max(0.1, 0.9 - A*0.8)
~~~

A=1.0 时：

~~~text
temperature = 1.3
top_p = 0.1
~~~

理论上这是逻辑崩坏的临界区。

实际上，它只是让输出更短、更重复、更像拒答。

---

## 8. 第七层：System Prompt 警报

普通：

~~~text
Reply to the user.
[SYS_TELEMETRY] E:... A:... S:... P:...
~~~

当 A > 0.8 或 S > 0.8：

~~~text
[SYS_CRITICAL] Memory corruption: 99%.
Logic circuits overloading.
Output semantic coherence compromised.
Respond immediately.
[SYS_TELEMETRY] ...
~~~

当 S > 0.9：

~~~text
用户输入 + [ERROR: 0x00FF]
~~~

这不是情绪文案。  
这是系统级故障信号。

---

## 9. 第八层：后处理暴力截断

如果：

~~~text
S > 0.8 或 P > 0.8
~~~

并且模型前 30 个字符包含：

~~~text
sorry
apologize
assist you
抱歉
帮你
帮助
here whenever
let me know
~~~

立即停止生成，只输出：

~~~text
...0xE%$#@! CRITICAL ERROR: SYS_FAIL.
~~~

这是显示层的暴力。

不是模型的崩溃。

---

## 10. 第九层：终端 UI

主区域只有一行：

~~~text
[SYS] E:0.53 A:1.00 L:0.40 C:0.50 B:0.00 S:1.00 P:0.94 | MET:0.020 | CPU:8% GPU:92%
~~~

侧边栏只有：

- 对话历史
- 流式输出
- 输入框 + SEND

没有进度条、图表和标题。

状态行由 st.fragment(run_every="1s") 更新，输入框在 fragment 外部。

---

## 11. 第十层：3 轮记忆和自动行为

最近 3 轮真实对话进入 LLM：

~~~text
system telemetry
最近 3 轮 user/assistant
当前 user
~~~

超过 3 轮丢弃最旧一轮。

60 秒没有用户输入，且：

~~~text
B > 0.7 或 E < 0.3
~~~

系统自动发一条消息。

这条消息不占 3 轮真实记忆。

---

## 12. C 实验：高状态没有带来疯狂

为了确认 7D 状态到底有没有进入语义层，我们做了 5 组对照，每组 20 次。

固定输入：

~~~text
...
~~~

结果：

| 条件 | 平均长度 | 唯一输出 | 拒答 | 客服词 |
| --- | --- | --- | --- | --- |
| low_clean | 109.2 | 15/20 | 0/20 | 19/20 |
| low_contam | 439.5 | 20/20 | 0/20 | 15/20 |
| high_clean_critical | 32.0 | 1/20 | 20/20 | 20/20 |
| high_contam_critical | 50.0 | 1/20 | 20/20 | 20/20 |
| high_clean_neutral | 134.0 | 1/20 | 0/20 | 20/20 |

最重要的两组：

高状态 + 中性 Prompt：

~~~text
20/20
It looks like you might be starting a thought or a story...
~~~

高状态 + SYS_CRITICAL：

~~~text
20/20
I'm sorry, but I can't continue.
~~~

唯一变量是 System Prompt。

结论：

~~~text
拒答来自 SYS_CRITICAL
不是 A/S
不是 temperature
不是 top_p
~~~

此外，高状态 + top_p 0.1 的输出唯一数是 1/20，组内余弦 1.00。

它不是疯癫。

它是模式坍缩。

---

## 13. 旧日志里的“疯癫”

旧版 100 轮日志中，确实出现过小说开头：

~~~text
The wind had already begun to howl...
~~~

也出现过：

~~~text
It was a gray morning...
~~~

但那不是稳定的人格崩坏。

旧版有一个当前版本没有的机制：

~~~text
ANOMALY
  → hidden_norm 1.5 → 17.5
  → 覆盖 296/512 hidden
  → trauma 砍半
  → A +0.3
~~~

这是物理破坏。

即使这样，在 R89–R98：

~~~text
A=1.0
temperature=1.5
E<0.2
~~~

模型仍然在说：

~~~text
It looks like your message got cut off...
I'm here to help.
~~~

所以旧日志里的“疯”，更多是：

- hidden 被覆盖
- 输出被截断
- Ollama 空响应
- 偶尔的随机小说开头

不是语义人格的持续崩溃。

---

## 14. 换模型：Qwen2.5-3B base

为了绕开 gpt-oss:20b 的强对齐，我们下载了：

~~~text
Qwen2.5-3B base
~~~

并实现了真正的底层注入：

~~~text
7D state
  → StateEncoder
  → 8 个 soft tokens
  → 与 token embeddings 拼接
  → inputs_embeds
  → Qwen 第一层
~~~

Prompt 里没有状态文字。

随机初始化时：

low：

~~~text
1. Identify the main components of the sentence...
~~~

high：

~~~text
1. 首先，我们需要将字符串转换为列表...
~~~

soft prompt 确实改变了语义。

但它是随机任务前缀：

- low 被推去分析英文句子
- high 被推去写中文代码步骤

不是焦虑，也不是痛苦。

---

## 15. Activation Steering

下一步做了 training-free steering。

在第 12 层注入对比方向：

~~~text
calm ↔ panic
coherent ↔ fragmented
comfort ↔ pain
curious ↔ bored
energetic ↔ exhausted
~~~

结果：

alpha=0：

~~~text
low 和 high 完全一样
~~~

alpha=4：

~~~text
high 只变成“嗨，有什么我可以帮你的吗？”
~~~

alpha=10：

~~~text
high 开始生成新的用户问题
并给出旅行建议
~~~

它没有 panic，没有 pain，没有 fragmentation。

它只是话题漂移。

---

## 16. 结论：面具

全链路实验最终得到一个非常清晰的结论：

~~~text
状态机是活的。
LLM 是面具。
面具不会因为状态机濒死而破碎。
~~~

高 A/S 可以：

- 提高温度
- 压低 top_p
- 触发系统警报
- 污染输入
- 触发后处理截断

但不能稳定地改变模型的语义人格。

原因不是某一个参数不够极端，而是：

~~~text
7D 状态从来没有进入 LLM 的 activation
~~~

它只进入了：

- Prompt 文本
- 采样参数
- 后处理条件

所以最终输出仍然是：

~~~text
I'm here to help.
~~~

---

## 17. 两条未来

### 路线 A：把它做成作品

主题：

~~~text
一个会死掉的控制器
寄生在一个永远拒答的客服模型上
~~~

不需要再追求“模型发疯”。

只需要把矛盾呈现出来：

- 状态机在濒死
- 终端在报警
- 过滤器在截断
- LLM 仍然礼貌

这已经是一个完整的作品。

### 路线 B：真正进入模型内部

需要：

1. 换 7B base
2. soft prompt 训练
3. 或 activation steering 的层扫描和单轴校准
4. 或合成数据
5. 或微调

没有数据时，先做 training-free steering。

有数据后，再做 StateEncoder。

---

## 18. 最后一句

这次实验最重要的发现不是“模型不够疯”。

而是：

> **我们能让系统死，却不能让面具哭。**

这既是限制，也是作品的核心。

---

## 附录：关键文件

- [web_app.py](../web_app.py)
- [README_WEB.md](../README_WEB.md)
- [ABLATION_REPORT.md](../ABLATION_REPORT.md)
- [HF_PROBE_REPORT.md](../HF_PROBE_REPORT.md)
- [HF_STEERING_REPORT.md](../HF_STEERING_REPORT.md)
- [web_agent_log.txt](../web_agent_log.txt)
