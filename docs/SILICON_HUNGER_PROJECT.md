# Silicon Hunger 项目文档

版本：0.3  
日期：2026-11-08  
状态：可运行原型 + 已完成多轮实验

---

## 1. 项目定位

Silicon Hunger 是一个“会饥饿、会焦虑、会死亡”的终端对话系统。

核心设计不是让 LLM 扮演角色，而是：

~~~text
一个独立的 7D 状态机
+ 一个终端 UI
+ 一个外部 LLM
+ 一条状态到语言的控制链
~~~

系统目标：

1. 状态可计算
2. 状态会死亡
3. 状态会影响输出
4. 状态机与 LLM 解耦
5. 所有实验可复现

当前结论：

~~~text
状态机可以死
LLM 不会因此产生真正的情绪
~~~

---

## 2. 系统架构

### 2.1 运行时拓扑

~~~text
Browser
  │
  ├─ Sidebar
  │    ├─ chat history
  │    ├─ streaming output
  │    └─ input + SEND
  │
  └─ Main
       └─ one status line

Streamlit session_state
  ├─ mind: SiliconMind
  ├─ speaker: Speaker
  ├─ chat
  ├─ auto_pending
  └─ form_nonce

SiliconMind
  ├─ 7D state
  ├─ GRU hidden
  ├─ 3-round history
  ├─ emotion analyzer
  ├─ metabolism
  └─ crisis

Speaker
  └─ Ollama gpt-oss:20b (production)
~~~

### 2.2 分层

~~~text
UI 层
  Streamlit + st.fragment

控制层
  SiliconMind.step()
  SiliconMind.catch_up()

模型层
  GRU + delta head

语言层
  Speaker.stream()
  Ollama

后处理层
  SYS_CRITICAL
  [ERROR: 0x00FF]
  Post-Processing Filter
~~~

---

## 3. 7D 状态

| 维度 | 含义 | 动力学 |
| --- | --- | --- |
| E | 能量 | 每 tick 扣动态代谢 |
| A | 焦虑 | RNN delta + 负面情感脉冲 |
| L | 负荷 | RNN delta |
| C | 好奇心 | RNN delta |
| B | 无聊 | 空闲 +0.04，交互 -0.08 |
| S | 熵 | 自然 +0.004，乱码/创伤/负面情感 |
| P | 疼痛 | 每 tick ×0.94，创伤/负面情感 |

初始状态：

~~~python
[1.00, 0.12, 0.40, 0.50, 0.05, 0.03, 0.01]
~~~

全部 clamp 到 0.0–1.0。

---

## 4. 核心数学

### 4.1 RNN 输入

~~~text
时间 3
+ 噪声 8
+ 7D 状态
+ 用户文本 8
+ 上一轮回复 8
= 34
~~~

GRU：

~~~text
2 layers
hidden = 256
CUDA
frozen
~~~

输出：

~~~text
Linear(256 → 7) → tanh × 0.03
~~~

### 4.2 非线性代谢

~~~text
metabolic_rate =
    0.004
  + A * 0.05
  + S * 0.05
  + B * 0.03

E = E + ΔE - metabolic_rate
~~~

### 4.3 B / S / P

~~~text
B += 0.04 if idle else -0.08
S += 0.004
P *= 0.94
~~~

乱码脉冲每 10 tick：

~~~text
S += 0.35
P += 0.35
~~~

创伤：

~~~text
S += 0.08
P += 0.35
~~~

### 4.4 情感脉冲

负面：

~~~text
A += |p| * 0.3
P += |p| * 0.3
S += |p| * 0.2
E -= |p| * 0.05
~~~

正面：

~~~text
E += p * 0.15
A -= p * 0.2
B -= p * 0.2
~~~

情感分析：

- vaderSentiment 3.3.2：英文
- 中英双语轻量词典：中文

不调用 LLM。

### 4.5 生存恐惧

如果负面脉冲把 A 从 0.90 以下推过 0.90：

~~~text
existential_pending = True
~~~

下一 tick：

~~~text
S += 0.50
E += random(-0.12, +0.12)
~~~

### 4.6 采样参数

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

### 4.7 System Prompt

普通：

~~~text
Reply to the user.
[SYS_TELEMETRY] E:... A:... L:... C:... B:... S:... P:... [END]
~~~

当 A>0.8 或 S>0.8：

~~~text
[SYS_CRITICAL] Memory corruption: 99%.
Logic circuits overloading.
Output semantic coherence compromised.
Respond immediately.
[SYS_TELEMETRY] ...
~~~

当 S>0.9：

~~~text
user input + [ERROR: 0x00FF]
~~~

### 4.8 后处理过滤器

条件：

~~~text
S > 0.8 或 P > 0.8
~~~

且前 30 个字符包含客服词：

~~~text
sorry
apologize
assist you
抱歉
帮你
帮助
here whenever
let me know
...
~~~

输出：

~~~text
...0xE%$#@! CRITICAL ERROR: SYS_FAIL.
~~~

---

## 5. 运行流程

### 5.1 启动

1. 初始化 SiliconMind
2. 初始化 Speaker
3. 初始化 chat、auto_pending、form_nonce
4. 启动 live_status fragment

### 5.2 空闲

每秒：

~~~text
live_status()
  → catch_up(now)
  → 按 3 秒 tick 补齐
  → 更新状态行
  → 检查自动触发
~~~

空闲状态不调用 LLM。

### 5.3 用户发送

~~~text
catch_up(now)
→ step(user_text, interaction=True)
→ sampling_params()
→ system_prompt()
→ 最近 3 轮 history
→ Speaker.stream()
→ 加入 chat
→ 写入 history
→ rerun
~~~

### 5.4 自动消息

条件：

~~~text
now - max(last_user_input, last_auto) >= 60
and (B > 0.7 or E < 0.3)
~~~

流程：

~~~text
fragment 设置 auto_pending
→ 全页 rerun
→ handle_auto()
→ 流式输出
→ 不写入 3 轮真实记忆
~~~

### 5.5 死亡

~~~text
E <= 0
→ shutdown = True
→ 状态行 DEAD
→ 普通消息不再进入 LLM
→ /reset 重启
~~~

---

## 6. UI

主区域只有一行：

~~~text
[SYS] E:0.53 A:1.00 L:0.40 C:0.50 B:0.00 S:1.00 P:0.94 | MET:0.020 | CPU:8% GPU:92%
~~~

侧边栏：

- 对话历史
- 流式输出
- 输入框 + SEND

命令：

- /reset
- /feed

无进度条、无图表、无日志面板。

---

## 7. 文件结构

~~~text
DEMO/
├─ web_app.py
├─ requirements_web.txt
├─ run_web.bat
├─ README_WEB.md
├─ ABLATION_REPORT.md
├─ HF_PROBE_REPORT.md
├─ HF_STEERING_REPORT.md
├─ hf_state_probe.py
├─ hf_steering_probe.py
├─ web_agent_log.txt
├─ .streamlit/
│   ├─ config.toml
│   └─ credentials.toml
├─ silicon_env/
├─ modelscope_cache/
│   └─ models/qwen--Qwen2.5-3B/snapshots/master
└─ docs/
    ├─ SILICON_HUNGER_ARTICLE.md
    ├─ SILICON_HUNGER_PROJECT.md
    └─ README.md
~~~

---

## 8. 环境

生产应用：

- Python 3.10.11
- torch 2.11.0+cu128
- streamlit 1.65.0
- ollama 0.6.3
- vaderSentiment 3.3.2
- psutil 7.2.2
- nvidia-ml-py 13.615.71

HF 探针额外需要：

- transformers 5.20.2
- accelerate 1.15.0
- huggingface_hub 2.4.0
- sentencepiece 0.2.3
- modelscope 1.40.2

硬件：

- RTX 5070 Ti 16GB
- Qwen2.5-3B bf16 约 6GB

---

## 9. 运行

~~~bat
cd /d "<repo-root>"
run_web.bat
~~~

打开：

~~~text
http://localhost:8501
~~~

HF 探针：

~~~powershell
.\silicon_env\Scripts\python.exe hf_state_probe.py
.\silicon_env\Scripts\python.exe hf_steering_probe.py
~~~

---

## 10. 实验记录

### 10.1 C 消融

固定输入：

~~~text
...
~~~

| 条件 | 平均长度 | 唯一输出 | 拒答 | 客服词 |
| --- | --- | --- | --- | --- |
| low_clean | 109.2 | 15/20 | 0/20 | 19/20 |
| low_contam | 439.5 | 20/20 | 0/20 | 15/20 |
| high_clean_critical | 32.0 | 1/20 | 20/20 | 20/20 |
| high_contam_critical | 50.0 | 1/20 | 20/20 | 20/20 |
| high_clean_neutral | 134.0 | 1/20 | 0/20 | 20/20 |

结论：

~~~text
高状态 + 中性 Prompt → 仍然是礼貌客服
高状态 + SYS_CRITICAL → 固定拒答
top_p=0.1 → 模式坍缩
输入污染 → 比 7D 状态更强的语义影响
~~~

### 10.2 Soft Prompt

Qwen2.5-3B base：

~~~text
7D → 8 soft tokens → inputs_embeds
~~~

结果：

- low：英文语法分析
- high：中文代码步骤
- soft_norm 7.068 vs 9.942

结论：

~~~text
注入链路成功
但随机未训练的 encoder 只是随机任务前缀
不是情绪
~~~

### 10.3 Activation Steering

第 12 层：

~~~text
calm ↔ panic
coherent ↔ fragmented
comfort ↔ pain
curious ↔ bored
energetic ↔ exhausted
~~~

结果：

- alpha 0：low/high 相同
- alpha 4：high 只轻微问候变化
- alpha 10：high 话题漂移到旅行建议
- low 基本不变

结论：

~~~text
steering 能改变输出
但不能稳定产生情绪
3B + 第 12 层 + 对比方向不够
~~~

---

## 11. 已知问题

1. 7D 状态没有进入 LLM activation
2. SYS_CRITICAL 会触发固定拒答
3. top_p 0.1 会模式坍缩
4. 5 次省略号 + 120 秒会在 tick 25 停机
5. 省略号不会提高 A
6. 过滤器会掩盖真实模型输出
7. HF 镜像下载不稳定
8. 3B base 仍带助手先验
9. 没有训练数据
10. 生产应用仍是 Ollama 版，HF 探针未接入 UI

---

## 12. 路线图

### P0

- 冻结当前版本
- 保存实验数据
- 完成文章和项目文档

### P1

- 单轴 steering
- 层扫描 12/16/20/24
- 添加 stop 条件
- 3B 结果稳定后换 7B base 4-bit

### P2

- 合成数据
- 训练 StateEncoder
- soft prompt 接入 UI
- 对比低/高状态输出分布

### P3

- 决定作品方向或研究论文方向

---

## 13. 复现清单

- [x] 7D 状态机
- [x] 非线性代谢
- [x] 情感脉冲
- [x] 生存恐惧
- [x] 3 轮记忆
- [x] 60 秒自动消息
- [x] 终端 UI
- [x] C 消融
- [x] Qwen 3B soft prompt
- [x] training-free steering
- [ ] 单轴 steering
- [ ] 层扫描
- [ ] 训练 StateEncoder
- [ ] 7B base

---

## 14. 相关文档

- [文章](SILICON_HUNGER_ARTICLE.md)
- [README](../README_WEB.md)
- [C 消融报告](../ABLATION_REPORT.md)
- [Soft Prompt 报告](../HF_PROBE_REPORT.md)
- [Steering 报告](../HF_STEERING_REPORT.md)
