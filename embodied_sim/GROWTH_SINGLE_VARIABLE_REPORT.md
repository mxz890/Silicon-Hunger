# 单一变量：加入节点生长

日期：2026-11-22  
基线：curriculum v2（20k BC、30 epochs、64×64 视觉、单障碍、4 envs、per-env GAE）  
唯一改动：BC 后、Stage 1 PPO 前，图生长 200 节点（2000 → 2200）。

---

## 1. 结果

| 阶段 | 成功率 | 步数 |
| --- | ---: | ---: |
| Stage 0 BC（生长前） | **0.9667** | 96.63 |
| Stage 0（生长后，未再训） | **0.1000** | 233.07 |
| Stage 1 PPO（生长后） | **0.0333** | 249.97 |
| Target domain | 0.0333 | 246.77 |

固定 2000 基线 Stage 1：

~~~text
success = 0.0333
~~~

所以：

> 生长没有提升 Stage 1；同时把 Stage 0 从 96.7% 打到 10%。

---

## 2. 生长方式

- 新增 200 节点
- 每个新节点连接当前激活最高的 4 个节点
- 新节点接新 output
- 旧 output 变成 hidden relay
- 旧 actor/critic 权重保留
- 新输入列零初始化

---

## 3. 结论

> 这次生长是破坏性的。

原因：

1. 生长后没有 warm-up
2. 新节点直接接 output，改变了输出路径
3. 旧 policy head 的特征语义被破坏
4. 新节点权重太小，但 output 聚合变了
5. PPO 只有 10 updates，无法恢复

---

## 4. 这说明什么

不能说明“生长一定有害”。

只能说明：

> 在当前“BC 后直接生长，再 PPO”流程里，生长没有带来收益，反而破坏已有策略。

---

## 5. 下一步

1. 生长后先冻结旧图，只训练新节点/新 head
2. 用 gate 控制新节点输出，初始 gate=0
3. 生长前后做 stage0 验证，掉了就回滚
4. 生长后先 DAgger / BC 恢复，再 PPO
5. 生长量从 200 降到 20–50

---

## 6. 一句话

> 单一变量实验失败：生长 200 节点后，Stage 0 从 96.7% 掉到 10%，Stage 1 仍是 3.3%；当前生长方式是破坏性的。

---

## 7. 文件

- growth_curriculum.py
- growth_curriculum_result.json
- GROWTH_SINGLE_VARIABLE_REPORT.md
