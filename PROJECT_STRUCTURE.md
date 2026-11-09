# Project Structure

本文由当前仓库的 tracked files 整理，生成时间：2026-11-09。

本地虚拟环境、模型权重和 Hugging Face 缓存不进入 Git：

~~~text
silicon_env/
modelscope_cache/
hf_cache/
__pycache__/
AppData/
~~~

---

## 完整结构

~~~text
.
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── README_WEB.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── requirements_web.txt
├── requirements_hf.txt
├── run_demo.bat
├── run_web.bat
├── web_app.py
├── main.py
├── hf_state_probe.py
├── hf_steering_probe.py
├── build_word_report.py
├── ABLATION_REPORT.md
├── HF_PROBE_REPORT.md
├── HF_STEERING_REPORT.md
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
├── Silicon_Hunger_Docs_2026-11-08/
│   ├── README_WEB.md
│   ├── ABLATION_REPORT.md
│   ├── HF_PROBE_REPORT.md
│   ├── HF_STEERING_REPORT.md
│   ├── requirements_hf.txt
│   ├── requirements_web.txt
│   ├── web_app.py
│   ├── hf_state_probe.py
│   ├── hf_steering_probe.py
│   ├── Silicon_Hunger_Report_2026-11-08.docx
│   ├── web_agent_log.txt
│   ├── ablation_raw.json
│   ├── ablation_2x2.json
│   ├── ablation_neutral_high.json
│   └── docs/
│       ├── README.md
│       ├── PACKAGE_MANIFEST.md
│       ├── SILICON_HUNGER_ARTICLE.md
│       └── SILICON_HUNGER_PROJECT.md
├── agent_log.txt
├── log.txt
└── web_agent_log.txt
~~~

---

## 关键文件职责

| 路径 | 职责 |
| --- | --- |
| [web_app.py](web_app.py) | 生产版入口：7D 状态机、Streamlit UI、Ollama 流式对话 |
| [main.py](main.py) | 早期 CLI 版本，保留作历史参考 |
| [hf_state_probe.py](hf_state_probe.py) | 7D 状态到 8 个 soft tokens 的注入探针 |
| [hf_steering_probe.py](hf_steering_probe.py) | 第 12 层 activation steering 探针 |
| [build_word_report.py](build_word_report.py) | Word 报告生成脚本 |
| [README_WEB.md](README_WEB.md) | Web 版详细说明 |
| [ABLATION_REPORT.md](ABLATION_REPORT.md) | C 消融完整报告 |
| [HF_PROBE_REPORT.md](HF_PROBE_REPORT.md) | Soft prompt 报告 |
| [HF_STEERING_REPORT.md](HF_STEERING_REPORT.md) | Activation steering 报告 |
| [docs/SILICON_HUNGER_ARTICLE.md](docs/SILICON_HUNGER_ARTICLE.md) | 长文文章 |
| [docs/SILICON_HUNGER_PROJECT.md](docs/SILICON_HUNGER_PROJECT.md) | 项目技术文档 |
| [Silicon_Hunger_Report_2026-11-08.docx](Silicon_Hunger_Report_2026-11-08.docx) | 合并 Word 总结 |
| [Silicon_Hunger_Docs_2026-11-08.zip](Silicon_Hunger_Docs_2026-11-08.zip) | 完整文档包 |
| [ablation_raw.json](ablation_raw.json) | C 消融原始输出 |
| [ablation_2x2.json](ablation_2x2.json) | 低状态污染 / 高状态中性 Prompt 控制组 |
| [ablation_neutral_high.json](ablation_neutral_high.json) | 高状态 + 中性 Prompt 20 次 |
| [web_agent_log.txt](web_agent_log.txt) | Web 版运行日志 |
| [agent_log.txt](agent_log.txt) | 旧 CLI 运行日志 |
| [log.txt](log.txt) | 旧 100 轮 4D 日志 |

---

## 说明

- Silicon_Hunger_Docs_2026-11-08/ 是文档包的可分发副本。
- Silicon_Hunger_Docs_2026-11-08.zip 是同一份内容的压缩包。
- 两者保留是为了让访问者无需解压即可直接阅读 Markdown。
- 模型权重和虚拟环境不进入仓库。
