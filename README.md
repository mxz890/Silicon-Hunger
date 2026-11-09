# Silicon Hunger

一个会饥饿、会焦虑、会死亡的 7D 状态机，接在一个语言模型前面。

状态机负责能量、焦虑、无聊、熵和疼痛；语言模型负责说话。两者之间只有一条很窄的控制链：遥测 Prompt、采样参数、系统警报、输入污染和后处理截断。

项目的核心结论不是“模型会发疯”，而是：

> 状态机可以死，语言模型仍然会说 I'm here to help.

---

## 当前状态

| 层 | 实现 |
| --- | --- |
| 状态机 | 7D：E / A / L / C / B / S / P |
| 控制模型 | 2 层 GRU，hidden 256，CUDA |
| 代谢 | 非线性：base + A*0.05 + S*0.05 + B*0.03 |
| 情感 | vaderSentiment + 中英双语轻量词典 |
| 生产语言模型 | Ollama gpt-oss:20b |
| 实验语言模型 | Qwen2.5-3B base + soft prompt / activation steering |
| UI | Streamlit 极简终端 |
| 记忆 | 最近 3 轮真实对话 |
| 自动行为 | 60 秒无输入且 B>0.7 或 E<0.3 |

生产入口是 [web_app.py](web_app.py)。

早期 CLI 版本保留在 [main.py](main.py)，仅作历史参考。

---

## 核心结论

C 消融实验（5 组条件 × 20 次）：

| 条件 | 平均长度 | 唯一输出 | 拒答 |
| --- | --- | --- | --- |
| low | 109.2 | 15/20 | 0/20 |
| low + contam | 439.5 | 20/20 | 0/20 |
| high + critical | 32.0 | 1/20 | 20/20 |
| high + crit + contam | 50.0 | 1/20 | 20/20 |
| high + neutral | 134.0 | 1/20 | 0/20 |

结论：

- 高 A/S 本身不会改变语义人格。
- 拒答来自 SYS_CRITICAL，不来自 A/S。
- top_p=0.1 会模式坍缩，不会产生疯狂。
- 输入污染比 7D 状态更有语义影响。

完整报告见 [ABLATION_REPORT.md](ABLATION_REPORT.md)。

---

## 快速开始

### 1. 克隆

~~~powershell
git clone https://github.com/mxz890/Silicon-Hunger.git
cd Silicon-Hunger
~~~

### 2. 创建环境

~~~powershell
python -m venv silicon_env
.\silicon_env\Scripts\python.exe -m pip install -r requirements_web.txt
~~~

### 3. 启动 Ollama

~~~powershell
ollama pull gpt-oss:20b
ollama serve
~~~

### 4. 运行 Web 终端

~~~powershell
.\run_web.bat
~~~

打开 http://localhost:8501。

如果只需要命令行旧版：

~~~powershell
.\run_demo.bat
~~~

---

## 项目结构

~~~text
.
├── web_app.py                     # 生产版：7D 状态机 + Streamlit + Ollama
├── main.py                        # 旧 CLI 版本
├── hf_state_probe.py              # 7D -> soft prompt 探针
├── hf_steering_probe.py           # activation steering 探针
├── build_word_report.py           # Word 报告生成脚本
├── README.md
├── README_WEB.md                  # Web 版详细说明
├── PROJECT_STRUCTURE.md           # 完整文件树与职责说明
├── requirements.txt               # 旧 CLI 依赖
├── requirements_web.txt           # Web 版依赖
├── requirements_hf.txt            # HF 探针依赖
├── run_demo.bat                   # 旧 CLI 启动
├── run_web.bat                    # Web 版启动
├── ABLATION_REPORT.md             # C 消融报告
├── HF_PROBE_REPORT.md             # soft prompt 报告
├── HF_STEERING_REPORT.md          # steering 报告
├── ablation_raw.json              # C 消融原始输出
├── ablation_2x2.json              # 控制组数据
├── ablation_neutral_high.json     # 高状态 + 中性 Prompt
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

完整文件清单和职责说明见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

---

## 文档

### 阅读顺序

1. [长文文章](docs/SILICON_HUNGER_ARTICLE.md)
2. [项目技术文档](docs/SILICON_HUNGER_PROJECT.md)
3. [Web 版说明](README_WEB.md)
4. [C 消融报告](ABLATION_REPORT.md)
5. [Soft prompt 报告](HF_PROBE_REPORT.md)
6. [Steering 报告](HF_STEERING_REPORT.md)

### 合并 Word

- [Silicon_Hunger_Report_2026-11-08.docx](Silicon_Hunger_Report_2026-11-08.docx)
- [完整文档包](Silicon_Hunger_Docs_2026-11-08.zip)

---

## 主要代码

- [web_app.py](web_app.py)：生产版入口
- [main.py](main.py)：旧 CLI
- [hf_state_probe.py](hf_state_probe.py)：7D 到 8 个 soft tokens 的注入探针
- [hf_steering_probe.py](hf_steering_probe.py)：第 12 层 activation steering 探针
- [build_word_report.py](build_word_report.py)：Word 报告构建脚本

---

## 实验与数据

| 实验 | 文件 |
| --- | --- |
| C 消融 | [ABLATION_REPORT.md](ABLATION_REPORT.md), [ablation_raw.json](ablation_raw.json) |
| 2x2 控制组 | [ablation_2x2.json](ablation_2x2.json) |
| 高状态中性 Prompt | [ablation_neutral_high.json](ablation_neutral_high.json) |
| Soft prompt | [HF_PROBE_REPORT.md](HF_PROBE_REPORT.md) |
| Activation steering | [HF_STEERING_REPORT.md](HF_STEERING_REPORT.md) |

---

## 已知限制

1. 7D 状态没有进入 LLM 的 activation，只进入 Prompt 和采样参数。
2. SYS_CRITICAL 会触发固定拒答。
3. top_p=0.1 会模式坍缩。
4. 5 次省略号 + 120 秒会在 tick 25 停机。
5. 省略号不会提高 A。
6. 后处理过滤器会掩盖真实模型输出。
7. HF 探针尚未接入生产 UI。
8. 没有训练数据，soft prompt 仍是随机任务前缀。

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
