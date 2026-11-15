# Silicon Hunger

一个会饥饿、会焦虑、会死亡的 7D 状态机，接在一个语言模型前面。

状态机负责能量、焦虑、无聊、熵和疼痛；语言模型负责说话。

这个仓库里有两条不同的技术路线，必须分开看：

> **版本 A：gpt-oss:20b 的软注入 / 外部控制**  
> **版本 B：Qwen2.5-3B base 的底层注入**

两者共用同一个 7D 状态机，但状态进入模型的方式不同，结论也不同。

---

## 先看版本边界

| | 版本 A | 版本 B |
| --- | --- | --- |
| 语言模型 | gpt-oss:20b（Ollama） | Qwen2.5-3B base |
| 状态进入方式 | Prompt 遥测、采样参数、系统警报、输入污染、后处理 | soft tokens -> inputs_embeds；activation steering |
| 是否进入模型内部 | 否 | 是 |
| 主要结果 | 拒答、模式坍缩、客服语义不变 | 输出被改写，但没有稳定情绪 |
| 入口 | [web_app.py](web_app.py) | [hf_state_probe.py](hf_state_probe.py)、[hf_steering_probe.py](hf_steering_probe.py) |

早期文档里也把 Ollama 里的这个模型写作 ChatGPT OS 20B；实际模型 ID 是 **gpt-oss:20b**。

---

## 版本 A：gpt-oss:20b 的软注入

这是最早的 Web 终端版本。

状态不进入 LLM 的 activation，只进入四个外部边界：

1. System Prompt 里的 [SYS_TELEMETRY]
2. temperature / top_p / max_tokens
3. SYS_CRITICAL 和 [ERROR: 0x00FF]
4. 后处理客服词过滤器

### 结果

C 消融，5 组条件 × 20 次：

| 条件 | 平均长度 | 唯一输出 | 拒答 |
| --- | --- | --- | --- |
| low | 109.2 | 15/20 | 0/20 |
| low + contam | 439.5 | 20/20 | 0/20 |
| high + critical | 32.0 | 1/20 | 20/20 |
| high + crit + contam | 50.0 | 1/20 | 20/20 |
| high + neutral | 134.0 | 1/20 | 0/20 |

关键结果：

- 高 A/S + 中性 Prompt：20/20 仍然是礼貌客服句。
- 高 A/S + SYS_CRITICAL：20/20 固定拒答。
- top_p=0.1：唯一输出 1/20，组内余弦 1.00。
- 输入污染比 7D 状态更有语义影响。

结论：

> 软注入可以改变长度、采样和拒答，但不能稳定改变语义人格。

完整报告：[ABLATION_REPORT.md](ABLATION_REPORT.md)

---

## 版本 B：Qwen2.5-3B base 的底层注入

这一版不再把状态写进 Prompt，而是把它变成连续向量，直接进入模型输入。

### B1. Soft prompt

~~~text
7D state
  -> Linear(7, 256)
  -> SiLU
  -> Linear(256, 8 * 2048)
  -> 8 soft tokens
  -> cat(soft, token_embeds)
  -> inputs_embeds
  -> Qwen first layer
~~~

greedy 解码下：

- low state：英文语法分析
- high state：中文代码步骤
- soft_norm：7.068 vs 9.942

结论：

> 注入链路成功，但 StateEncoder 未训练，随机向量只是随机任务前缀，不是情绪。

### B2. Activation steering

在第 12 层 residual stream 注入 5 个对比方向：

~~~text
calm <-> panic
coherent <-> fragmented
comfort <-> pain
curious <-> bored
energetic <-> exhausted
~~~

结果：

| alpha | low | high |
| --- | --- | --- |
| 0 | 客服澄清 | 与 low 完全一致 |
| 4 | 客服澄清 | 只变成轻微问候 |
| 10 | 基本不变 | 生成新的用户问题，并给出旅行建议 |

结论：

> steering 能改变话题和对话模式，但没有稳定产生 panic、pain 或 fragmentation。

完整报告：

- [HF_PROBE_REPORT.md](HF_PROBE_REPORT.md)
- [HF_STEERING_REPORT.md](HF_STEERING_REPORT.md)

---

## 两个版本不要混在一起说

版本 A 的结论是：

~~~text
状态 -> Prompt / 采样
-> 拒答 / 模式坍缩
-> 语义人格不变
~~~

版本 B 的结论是：

~~~text
状态 -> soft tokens / activation
-> 输出被改写
-> 但未训练时没有稳定情绪
~~~

正确的总标题是：

> **软注入改变不了人格，底层注入能改变输出，但还没有学会情绪。**

---

## 共享的 7D 状态机

| 维度 | 含义 | 动力学 |
| --- | --- | --- |
| E | 能量 | 非线性代谢 |
| A | 焦虑 | RNN delta + 负面情感脉冲 |
| L | 负荷 | RNN delta |
| C | 好奇心 | RNN delta |
| B | 无聊 | 空闲 +0.04，交互 -0.08 |
| S | 熵 | 自然 +0.004，乱码 / 创伤 / 负面情感 |
| P | 疼痛 | 每 tick ×0.94，创伤 / 负面情感 |

关键公式：

~~~text
metabolic =
    0.004
  + A * 0.05
  + S * 0.05
  + B * 0.03

temperature = 0.5 + A*0.5 + S*0.3
max_tokens  = max(50, int(50 + E*100 - P*30))
top_p       = max(0.1, 0.9 - A*0.8)
~~~

---

## 快速开始

### 版本 A：gpt-oss:20b Web 终端

~~~powershell
python -m venv silicon_env
.\silicon_env\Scripts\python.exe -m pip install -r requirements_web.txt

ollama pull gpt-oss:20b
ollama serve

.\run_web.bat
~~~

打开 http://localhost:8501。

### 版本 B：Qwen2.5-3B base 探针

~~~powershell
.\silicon_env\Scripts\python.exe -m pip install -r requirements_hf.txt

$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen2.5-3B', cache_dir='modelscope_cache'))"

.\silicon_env\Scripts\python.exe hf_state_probe.py
.\silicon_env\Scripts\python.exe hf_steering_probe.py
~~~

---

## 项目结构

~~~text
.
├── web_app.py                     # 版本 A：7D 状态机 + Streamlit + Ollama
├── main.py                        # 旧 CLI 版本
├── hf_state_probe.py              # 版本 B：7D -> soft prompt
├── hf_steering_probe.py           # 版本 B：activation steering
├── build_word_report.py           # Word 报告生成
├── README.md
├── README_WEB.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── requirements_web.txt
├── requirements_hf.txt
├── run_web.bat
├── run_demo.bat
├── ABLATION_REPORT.md             # 版本 A：C 消融
├── HF_PROBE_REPORT.md             # 版本 B：soft prompt
├── HF_STEERING_REPORT.md          # 版本 B：steering
├── ablation_raw.json
├── ablation_2x2.json
├── ablation_neutral_high.json
├── Silicon_Hunger_Report_2026-11-08.docx
├── Silicon_Hunger_Docs_2026-11-08.zip
├── docs/
│   ├── README.md
│   ├── PACKAGE_MANIFEST.md
│   ├── SILICON_HUNGER_ARTICLE.md
│   └── SILICON_HUNGER_PROJECT.md
├── Silicon_Hunger_Docs_2026-11-08/  # 可分发副本
├── .streamlit/
│   └── config.toml
└── .gitignore
~~~

完整文件树见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

---

## 文档

1. [长文文章](docs/SILICON_HUNGER_ARTICLE.md)
2. [项目技术文档](docs/SILICON_HUNGER_PROJECT.md)
3. [Web 版说明](README_WEB.md)
4. [C 消融报告](ABLATION_REPORT.md)
5. [Soft prompt 报告](HF_PROBE_REPORT.md)
6. [Steering 报告](HF_STEERING_REPORT.md)
7. [合并 Word](Silicon_Hunger_Report_2026-11-08.docx)
8. [完整文档包](Silicon_Hunger_Docs_2026-11-08.zip)

---

## 已知限制

### 版本 A

- 状态没有进入 activation。
- SYS_CRITICAL 会触发固定拒答。
- top_p=0.1 会模式坍缩。
- 5 次省略号 + 120 秒会在 tick 25 停机。
- 后处理过滤器会掩盖真实模型输出。

### 版本 B

- 只有 3B，模型容量有限。
- StateEncoder 随机初始化，未训练。
- steering 只做了第 12 层、3 个 alpha，未做单轴和层扫描。
- 没有训练数据，不能证明“状态可以稳定映射到情绪”。

---

## 相关研究

- Dynamic Personality Adaptation in LLMs via State Machines
- Giving Characters a Life
- Anthropic: Emotion concepts in a large language model
- Refusal in Language Models Is Mediated by a Single Direction
- COSMIC
- CAA / Representation Engineering
- Generative Agents
- Project Sid
- EILS

详细地图见 [项目文档 2.7 节](docs/SILICON_HUNGER_PROJECT.md) 和 Word 版。
