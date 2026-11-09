# Silicon Hunger 文档包

这个包包含两部分：

1. 一篇完整叙事文章
2. 一份项目技术文档

## 阅读顺序

### 先读文章

[SILICON_HUNGER_ARTICLE.md](SILICON_HUNGER_ARTICLE.md)

适合：

- 想快速理解整个项目在做什么
- 想看实验过程和结论
- 想理解“为什么模型始终礼貌”

### 再读项目文档

[SILICON_HUNGER_PROJECT.md](SILICON_HUNGER_PROJECT.md)

适合：

- 复现系统
- 查公式
- 查架构
- 查实验数据
- 查已知问题
- 查下一步路线

## 项目一句话

~~~text
一个会死掉的 7D 状态机，
寄生在一个始终礼貌的 LLM 上。
~~~

## 当前结论

~~~text
状态机可以死。
采样可以乱。
Prompt 可以变成系统警报。
输出可以被物理截断。
但模型的语义人格仍然是客服。
~~~

## 核心数据

### C 消融

| 条件 | 平均长度 | 唯一输出 | 拒答 |
| --- | --- | --- | --- |
| low_clean | 109.2 | 15/20 | 0/20 |
| low_contam | 439.5 | 20/20 | 0/20 |
| high_clean_critical | 32.0 | 1/20 | 20/20 |
| high_contam_critical | 50.0 | 1/20 | 20/20 |
| high_clean_neutral | 134.0 | 1/20 | 0/20 |

### 软提示

~~~text
7D → 8 soft tokens → inputs_embeds
~~~

注入成功，但随机未训练时只是随机任务前缀。

### Activation Steering

第 12 层，alpha=10：

- low 不变
- high 从客服句漂移到旅行建议
- 没有 panic / pain / fragmentation

## 文件列表

- SILICON_HUNGER_ARTICLE.md
- SILICON_HUNGER_PROJECT.md
- README.md

## 项目根目录

完整代码和报告在：

~~~text
C:\Users\MENG XAINGZE\Desktop\DEMO
~~~

关键文件：

- web_app.py
- README_WEB.md
- ABLATION_REPORT.md
- HF_PROBE_REPORT.md
- HF_STEERING_REPORT.md
- hf_state_probe.py
- hf_steering_probe.py
- web_agent_log.txt
