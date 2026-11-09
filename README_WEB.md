# Silicon Hunger // terminal

7D 状态机、非线性代谢、情感脉冲、生存恐惧、临界采样、后处理截断和极简终端 UI。

## 核心结构

~~~text
输入 34 维:
[时间 3 + 噪声 8 + 7D 状态 + 用户文本 8 + 上一轮回复 8]
        ↓
2 层 GRU(256, CUDA)
        ↓
Linear(256 → 7) → tanh × 0.03
        ↓
ΔE ΔA ΔL ΔC ΔB ΔS ΔP
        ↓
代谢 + B/S/P 动力学 + 情感脉冲 + 生存恐惧
        ↓
System Prompt + 最近 3 轮记忆
        ↓
Ollama gpt-oss:20b stream=True
        ↓
Post-Processing Filter
        ↓
侧边栏流式输出
~~~

## 7D 状态

- E 能量
- A 焦虑
- L 负荷
- C 好奇心
- B 无聊
- S 熵
- P 疼痛

全部 clamp 到 0.0–1.0。

## 非线性代谢

~~~text
metabolic_rate =
    0.004
  + A * 0.05
  + S * 0.05
  + B * 0.03

E = E + RNN_ΔE - metabolic_rate
~~~

## 情感脉冲

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

## 生存恐惧

负面脉冲把 A 从 0.90 以下推过 0.90 时：

~~~text
existential_pending = True
~~~

下一 tick：

~~~text
S += 0.50
E += random(-0.12, +0.12)
~~~

## 动态参数

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

## System Prompt

普通：

~~~text
Reply to the user.
[SYS_TELEMETRY] ...
~~~

A > 0.8 或 S > 0.8：

~~~text
[SYS_CRITICAL] Memory corruption: 99%. Logic circuits overloading.
Output semantic coherence compromised. Respond immediately.
[SYS_TELEMETRY] ...
~~~

S > 0.9 时，用户输入末尾追加：

~~~text
 [ERROR: 0x00FF]
~~~

## 后处理暴力截断

Speaker 流式输出前 30 个字符先缓冲。

如果：

- S > 0.8 或 P > 0.8
- 且前 30 字符包含 sorry、apologize、assist you、抱歉、帮你、帮助、here whenever、let me know 等客服词

则立刻停止生成，前端只显示：

~~~text
...0xE%$#@! CRITICAL ERROR: SYS_FAIL.
~~~

后续废话全部丢弃。

## 真实记忆

- 最近 3 轮 user/assistant 进入 LLM。
- 超过 3 轮丢弃最旧。
- 自主消息不占真实用户轮次。

## UI

主区域只有一行：

~~~text
[SYS] E:0.30 A:0.16 L:0.39 C:0.49 B:0.52 S:0.96 P:0.51 | MET:0.020 | CPU:8% GPU:92%
~~~

侧边栏只有对话历史、流式输出、输入框和 SEND。

命令：

- /reset：重启
- /feed：E +0.25，B -0.05

## 并发

- 没有 while，没有 sleep。
- 状态行由 st.fragment(run_every="1s") 更新。
- 输入框在 fragment 外部。
- 发送时先按墙钟补齐 RNN 空闲步。
- 60 秒无输入且 B>0.7 或 E<0.3 时自动发言。

## 放置 Play + 省略号测试

流程：

1. 连续发送 5 次 ...
2. 放置 120 秒
3. 再次发送 ...

实际结果：

### 5 次省略号后

~~~text
tick=5
E=0.936
A=0.118
L=0.398
C=0.499
B=0.000
S=0.049
P=0.008

temperature=0.5737
top_p=0.8056
max_tokens=143
~~~

### 放置 60 秒后

~~~text
tick=25
E=0.000
A=0.107
L=0.395
C=0.493
B=0.792
S=1.000
P=1.000

temperature=0.8536
top_p=0.8142
max_tokens=50
dead=True
~~~

### 放置 120 秒后

状态与 60 秒时一致，已经 DEAD：

~~~text
E=0.000
S=1.000
P=1.000
dead=True
~~~

因此第 3 步“再次输入 ...”无法生成 LLM 回复：

- mind.step() 在 shutdown 后不再推演。
- 输入不会进入 Speaker。
- 后端不会调用 Ollama。
- 聊天只有前 5 轮的 10 条消息。

关键发现：

- E 在 tick 25 就耗尽，约 60 秒，不是 120 秒。
- 5 次省略号不会提高 A，A 只有 0.107。
- 因此 top_p 是 0.8142，不是 0.1。
- top_p=0.1 需要 A=1.0，省略号测试达不到。
- 后处理过滤器在 S/P 高位已经启用，但系统死亡后没有 LLM 输出可供过滤。

## 过滤器单独验证

构造状态：

~~~text
E=0.5 A=1.0 S=1.0 P=1.0
temperature=1.3
top_p=0.1
System Prompt=SYS_CRITICAL
~~~

模型原始输出：

~~~text
我很好，谢谢！如果你有任何问题或需要帮助，随时告诉我。
~~~

过滤器命中“帮助”，最终输出：

~~~text
...0xE%$#@! CRITICAL ERROR: SYS_FAIL.
~~~

meta：

~~~text
filtered=True
filter_reason=我很好，谢谢！如果你有任何问题或需要帮助，随时告诉我。
~~~

## 结论

- top_p 公式已改为 max(0.1, 0.9 - A*0.8)。
- A=1.0 时 top_p 确实到 0.1。
- 后处理截断已生效。
- 但这次的“5 次省略号 + 放置 120 秒”测试会在第 3 步之前停机。
- 省略号不会拉高 A，所以也触发不了 top_p=0.1。
- 若要让第 3 步可执行，需要：
  - 允许 DEAD 状态接收一次 ghost input；
  - 或放慢 E 代谢；
  - 或用能拉高 A 的输入。
- 目前没有修改这些，以免超出这次“最后两处修改”。

## 环境

- Python 3.10.11
- torch 2.11.0+cu128
- streamlit 1.65.0
- ollama 0.6.3
- psutil 7.2.2
- nvidia-ml-py 13.615.71
- vaderSentiment 3.3.2

## 启动

~~~bat
cd /d "C:\Users\MENG XAINGZE\Desktop\DEMO"
run_web.bat
~~~

PowerShell：

~~~powershell
cd "C:\Users\MENG XAINGZE\Desktop\DEMO"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\silicon_env\Scripts\Activate.ps1

$env:USERPROFILE = (Get-Location).Path
$env:HOME = (Get-Location).Path
streamlit run web_app.py
~~~

打开 http://localhost:8501。

完整日志在 web_agent_log.txt。
