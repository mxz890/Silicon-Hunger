# Silicon Hunger

一个本地运行的“硅基生命”状态机 Demo。CPU 侧维护能量、焦虑和 RNN 隐状态，GPU 侧按需调用 Ollama 的 `gpt-oss:20b` 生成第一人称独白；终端由 `rich` 实时渲染。

## 文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 完整状态机、Ollama 调用、输入子线程、日志和 UI |
| `requirements.txt` | Python 依赖及固定版本 |
| `run_demo.bat` | Windows 一键激活虚拟环境并启动 |
| `agent_log.txt` | 运行日志（首次运行后生成） |
| `silicon_env/` | Python 3.10 虚拟环境 |

## 环境准备

当前目录下已经创建好 `silicon_env` 并装好依赖。若换一台机器重建：

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
& $py -m venv silicon_env
.\silicon_env\Scripts\python.exe -m pip install -r requirements.txt
```

确认 Ollama 服务和模型：

```powershell
ollama list
# 如果没有模型：
ollama pull gpt-oss:20b
# 如果服务没启动：
ollama serve
```

## 运行

PowerShell（推荐，绕开脚本执行策略）：

```powershell
cd "C:\Users\MENG XAINGZE\Desktop\DEMO"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\silicon_env\Scripts\Activate.ps1
python main.py
```

或者直接使用 venv 里的解释器，不必激活：

```powershell
.\silicon_env\Scripts\python.exe main.py
```

也可以双击 / 在 CMD 中运行：

```bat
run_demo.bat
```

## 交互

- 输入 `feed` 回车：energy +30（上限 100），焦虑下降，立即显示一句满足语。
- 输入 `quit` 或 `exit`：退出。
- 程序在 500 轮或 `energy <= 0` 时自动退出。

## 行为参数

- 初始 energy = 100，anxiety = 0。
- 每个循环 energy 按 RNN 隐状态轻微消耗，anxiety 向低能量对应的高值靠近。
- `energy < 30` 时按概率触发饥饿/求生独白（带冷却）。
- 每 10 轮向 Prompt 注入恰好 15 字符随机乱码，并要求模型强行解释。
- 每次 LLM 调用只带最近 5 轮历史，`num_ctx=4096`。
- API 调用失败时不会崩溃，会显示本地降级独白继续运行。

## 自检命令

```powershell
# 不调用模型，验证循环、UI、异常注入
.\silicon_env\Scripts\python.exe main.py --cycles 12 --delay 0.1 --no-input --no-llm

# 只用真实模型跑 1 轮认知
.\silicon_env\Scripts\python.exe main.py --cycles 1 --delay 0 --no-input --no-ui --cognition-every 1

# 测试 Ollama 不可达时的 fallback
.\silicon_env\Scripts\python.exe main.py --cycles 1 --no-input --no-ui --cognition-every 1 --host http://127.0.0.1:1 --timeout 5
```

日志位于 `agent_log.txt`，包含每轮状态、异常注入、完整 Prompt、模型回复、feed 事件和 API 异常堆栈。
