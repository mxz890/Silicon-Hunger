# Project Path — From Silicon Hunger to Embodied

日期：2026-11-22  
状态：7D/LLM 与 10k graph 已归档；具身仿真为 active。

---

## 1. 总路径

```text
Silicon Hunger
│
├── A. 7D 状态机 + LLM                    [ARCHIVED]
│   ├── A1 gpt-oss:20b 软注入
│   ├── A2 Qwen2.5-3B soft prompt
│   ├── A3 Qwen2.5-3B activation steering
│   └── A4 C 消融 / 拒绝率 / persona 探针
│
├── B. 图 reservoir / 数学 / 中文          [ARCHIVED]
│   ├── B1 2000 节点 math
│   ├── B2 2000 → 10000 生长
│   ├── B3 10k fresh / fair 对照
│   ├── B4 真实中文 next-character
│   ├── B5 高难线性代数
│   ├── B6 动态化学 / 边门控
│   ├── B7 生长 + 反传 + 遗忘剪枝
│   └── B8 Graph → LLM adapter
│
└── C. 具身智能仿真                        [ACTIVE]
    ├── C1 MuJoCo 安装与点质量环境
    ├── C2 500 节点图 BC + DAgger
    ├── C3 2000 节点 + 64×64 视觉
    ├── C4 课程学习 Stage 0 / Stage 1
    ├── C5 PPO / per-env GAE
    └── C6 sim-to-sim
```

---

## 2. 当前 active 闭环

```text
MuJoCo 2D point mass
  ↓
64×64 RGB camera
  ↓
CNN → 64-d latent
  ↓
2000-node graph (K=6)
  ↓
actor / critic
  ↓
2-D action
  ↓
MuJoCo
```

训练：

```text
Stage 0: 无障碍 + 小范围
  PD 演示 20k
  BC 30 epochs
  success 0.9667

Stage 1: 1 个固定障碍
  PPO 10 updates
  per-env GAE
  success 0.0333
```

---

## 3. 当前结果

| 实验 | 成功率 |
|---|---:|
| PD 基线 | 1.00 |
| 500 节点图 BC + DAgger | 1.00 |
| 2000 节点视觉 BC（Stage 0） | 0.9667 |
| 2000 节点视觉 PPO（Stage 1） | 0.0333 |
| sim-to-sim target | 0.00–0.0333 |
| 生长 200 节点后 Stage 0 | 0.10 |

---

## 4. 当前唯一目标

> Stage 1（单固定障碍）> 80%。

在达到之前，不再加：

- 新化学
- 新视觉模块
- 10k 节点
- 新任务
- 真机

---

## 5. 目录路径

```text
DEMO/
├── embodied_sim/                 ACTIVE
│   ├── point_mass_env.py
│   ├── point_mass_obstacles.py
│   ├── vision_env.py
│   ├── graph_policy.py
│   ├── vision_graph_ppo.py
│   ├── curriculum_vision_v2.py
│   └── ...
│
├── research_mvp/                 ARCHIVED
│   ├── 10k graph
│   ├── math / Chinese
│   ├── chemistry
│   ├── growth / forgetting
│   └── visualizations
│
├── archive/
│   └── README.md
├── CURRENT_FOCUS.md
├── PROJECT_PATH.md
└── ...
```

---

## 6. 下一步

```text
Stage 1 > 80%
→ 8–16 parallel envs
→ sim-to-sim target > 50%
→ 2000-node stable training
→ real robot only after that
```

---

## 7. 一句话

> 路径已经从“7D 状态 + LLM”转到“图 reservoir”，再转到“具身仿真”；  
> 现在只保留一条 active 主线：MuJoCo 点质量 + 视觉 + 图控制器，先把单障碍 Stage 1 做上去。
