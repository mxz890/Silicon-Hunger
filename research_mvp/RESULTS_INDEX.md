# Silicon Hunger — Results Index

> **ARCHIVED INDEX.**  
> Active focus is now embodied simulation: `CURRENT_FOCUS.md` / `embodied_sim/`.  
> The 10k graph / 7D / chemistry results are archived, not deleted.

日期：2026-11-20  
规则：从今天起，**10000 节点版本作为研究基线**；2000 / 37 节点版本保留为历史与消融，不与 10k 主结果混用。

---

## 1. 研究基线

### 10k fresh baseline

- 节点：10000
- 边：40105
- output：9999
- output indegree：120
- 输入维度：64
- 特征：9998
- 结构：固定 3D DAG + 静态化学场
- 训练：每任务独立 ridge readout
- 时间：无

对应代码：

- scale_10k_fresh.py
- scale_10k_richer_chinese.py

基线结果：

| 数据 | 数学 MAE | 中文 next-char acc |
| --- | ---: | ---: |
| Lu Xun only | 0.31333 | 0.0700 |
| richer modern prose | 0.20834 | 0.0775 |

> 10k fresh 是当前的“干净基线”。  
> 10k fair / seed-grown 是结构对照，不是默认基线。

---

## 2. 10k 结果

| 实验 | 节点 | 边 | 关键结果 | 文件 |
| --- | ---: | ---: | --- | --- |
| seed-grown 10k | 10000 | 56212 | math MAE 0.10894; 竞赛题 25.03728; Chinese 0.0667 | SCALE_10K_REAL_REPORT.md |
| fresh 10k | 10000 | 40105 | math MAE 0.31333; Chinese 0.0700 | SCALE_10K_FRESH_REPORT.md |
| fair: seed-grown | 10000 | 56212 | math MAE 0.87411; Chinese 0.0700 | FAIR_10K_REPORT.md |
| fair: fresh | 10000 | 56212 | math MAE 0.21827; Chinese 0.0867 | FAIR_10K_REPORT.md |
| richer Chinese | 10000 | 40105 | math MAE 0.20834; Chinese 0.0775 | SCALE_10K_RICHER_REPORT.md |
| 557k 字数学语料 | 10000 | 40105 | sum_sq 0.097%; linear 4.61%; rectangle 0.060%; arithmetic 0.144%; hypotenuse 0.032% | MATH_500K_REPORT.md |
| 双脑区 2×2000 | 4000 | 16105 | single 4000 math 0.08498 / Chinese 0.09；two-region 0.15116 / 0.07 | TWO_REGION_REPORT.md |
| 生长 + 反传循环 | 2000→2060 | 9022→9310 | K=4；val 21.97→5.99；test 14.84→6.16；机制跑通但精度差 | GROW_TRAIN_LOOP_REPORT.md |
| 生长 + 反传 v2 | 2000→2300 | 9022→10498 | K=8；test 5.76→0.48755；曲线不稳定，需多 seed | GROW_TRAIN_LOOP_V2_REPORT.md |
| 遗忘曲线生长/剪枝 | 2000→2120 | 9022→? | 6 轮；只剪 5 节点；MAE 未改善 | FORGETTING_GROWTH_REPORT.md |
| Grow v2 多 seed | 2000→2300 | 9022→10498 | 3 seeds final 0.49 / 1.28 / 3.55；均值约 1.77；0.49 是幸运 seed | GROW_V2_MULTISEED_REPORT.md |
| Grow v3 best checkpoint | 2000→2300 | 9022→10498 | 3 seeds best 0.49 / 1.28 / 2.81；均值约 1.53；早停未触发 | GROW_TRAIN_LOOP_V3_REPORT.md |
| Data-only scaling | 2000 | 9022 | 500→20k samples：test 6.46→2.20；非单调；单 seed | DATA_SCALING_ONLY_REPORT.md |
| Data scaling to 100k | 2000 | 9022 | 20k/50k/100k：test 0.504/0.226/0.243；50k 后饱和 | DATA_SCALING_MORE_REPORT.md |
| hard linear algebra | 10000 | 40105 | det2 42%; det3 57%; matvec2 73%; solve2 169%; eigen2 11.8% | LINEAR_ALGEBRA_10K_REPORT.md |

---

## 3. 化学 / 边权结果

| 实验 | 节点 | 结果 | 文件 |
| --- | ---: | --- | --- |
| 37 节点 parity | 37 | baseline 0.50; neurochem 0.75（单次） | NEUROCHEM_PARITY_REPORT.md |
| 动态化学门控 | 2000 | math 0.24422 → 0.10360; Chinese 0.105 → 0.080; active edges 7747 → 8687 | DYNAMIC_CHEM_EDGES_REPORT.md |
| credit / broadcast / no_chem | 2000 | broadcast 最优；credit 数学 0.23742，broadcast 0.09272 | CREDIT_CHEM_ABLATION_REPORT.md |
| GPU 化学扩散 | 2000 | CPU 5.976 ms; GPU 3.862 ms; 1.55x | GPU_CHEM_REPORT.md |

---

## 4. 图生长 / 空间结果

| 实验 | 节点 | 结果 | 文件 |
| --- | ---: | --- | --- |
| 10 节点生长 | 10 | 度分布 9 / 6 / 2；出现 hub | graph_growth.py |
| Phase 0 / Catastrophe | 25 | 100 轮；灾难 + 重组 + 突变 | PHASE_REPORT.md |
| Phase 1 / 25 节点 | 25 | 灾难后重建；XOR 通过；parity 失败 | PHASE_REPORT.md |
| 2000 → 10000 | 10000 | 新节点进入 output；10 轮 | SCALE_10K_REAL_REPORT.md |
| 静态迭代生长 | 2000+8 | math MAE 0.45286 → 0.28769 | ITERATIVE_GROWTH_REPORT.md |
| readout growth | 2000 | 全局残差方向失败；linear 无改善 | READOUT_GROWTH_REPORT.md |

---

## 4.5 具身控制

| 实验 | 节点 | 结果 | 文件 |
| --- | ---: | --- | --- |
| MuJoCo 点质量 PD | — | success 1.00 / 94.64 steps | embodied_sim/README.md |
| 图控制 500 节点 | 500 | imitation + DAgger：success 1.00 / 95.68 steps | embodied_sim/GRAPH_CONTROL_REPORT.md |
| PPO + 障碍物 | 500 | 从零 PPO：success 0.04 / 247 steps；失败 | embodied_sim/PPO_OBSTACLES_REPORT.md |
| 视觉 + 2000 节点 + sim2sim | 2000 | source/target 均 0.00；流程跑通但 PPO 失败 | embodied_sim/VISION_2000_SIM2SIM_REPORT.md |
| 课程 + 视觉 BC + PPO | 2000 | 32×32；BC 0.40；stage1 0.05；target 0.00 | embodied_sim/CURRICULUM_VISION_REPORT.md |
| 课程 v2：20k BC + 64 视觉 | 2000 | Stage0 0.9667；单障碍 PPO Stage1 0.0333；target 0.00 | embodied_sim/CURRICULUM_VISION_V2_REPORT.md |
| 生长单一变量 | 2000→2200 | Stage0 0.9667→0.10；Stage1 0.0333；生长破坏策略 | embodied_sim/GROWTH_SINGLE_VARIABLE_REPORT.md |
| 具身生命周期 | 500→600 | 训练→生长→反传→遗忘剪枝；success 1.00；只剪 1 节点 | embodied_sim/LIFECYCLE_EMBODIED_REPORT.md |
| 具身生命周期 v2 | 500→550 | Stage1 单障碍；success 0.65–0.70；生长 50，剪 34 | embodied_sim/LIFECYCLE_EMBODIED_V2_REPORT.md |
| 生命周期 + PPO + 3 障碍 | 550 | 1 随机障碍 0.20；3 随机障碍 0.10；target 0.25 | embodied_sim/LIFECYCLE_PPO_OBSTACLES_REPORT.md |
| 生命周期 + PPO + 2 障碍 | 550 | 2 随机障碍 0.25→0.30；target 0.20 | embodied_sim/LIFECYCLE_PPO_2OBSTACLES_REPORT.md |
| MZX 连续训练 checkpoint | 500 | 连续 1.00/0.90/0.95 vs 从零 1.00/0.65/0.90 | embodied_sim/CONTINUAL_LEARNING_REPORT.md |
| MZX 连续 PPO + replay/KL | 500 | 6 冲突阶段；stage0 0.05，stage1 0.35，后面 0–5% | embodied_sim/CONTINUAL_PPO_REPORT.md |
| BC 93% → PPO 微调 | 500 | BC 0.9333；PPO last 0.50；best 0.90 | embodied_sim/BC_TO_PPO_REPORT.md |
| BC 93% → PPO 微调 v2 | 500 | KL .2/BC .3/早停；PPO 追平 0.9333，最终保留 BC | embodied_sim/BC_TO_PPO_V2_REPORT.md |
| BC → PPO 固定目标 | 500 | BC 0.5333；PPO best 0.7667；强目标 0.1667 | embodied_sim/BC_TO_PPO_FIXED_REPORT.md |
| BC → PPO 固定→小范围课程 | 500 | 固定 0.53→0.77；小范围 0.43→0.97；强随机 0.43 | embodied_sim/BC_PPO_CURRICULUM_REPORT.md |
| BC → PPO 扩展 1→2→3 障碍 | 500 | 3 seeds best：1obs 0.953；2obs 0.900；3obs 0.840 | embodied_sim/BC_PPO_EXPAND_REPORT.md |
| BC → PPO 扩展 v2 5 seeds | 500 | 20 updates；3obs best 均值 0.456；仅 seed 32000 到 0.80 | embodied_sim/BC_PPO_EXPAND_V2_REPORT.md |
| BC 稳定性 5 seeds | 500 | 固定初始化；test 均值 0.508，std 0.047；未达 80% | embodied_sim/BC_STABLE_REPORT.md |
| PD 避障专家 + DAgger | 500 | 专家 0.96；BC 0.99；DAgger 0.98 | embodied_sim/BC_DAGGER_REPORT.md |
| BC 99% → PPO 小范围随机 | 500 | BC 固定 0.99；小范围随机 0.53；PPO best 0.47；强目标 0.03 | embodied_sim/BC_DAGGER_TO_PPO_REPORT.md |
| 目标随机→障碍随机 | 500 | Task A PPO 0.90；Task B DAgger 0.92 / PPO best 0.96 | embodied_sim/TARGET_OBSTACLE_CURRICULUM_REPORT.md |
| MZX 2000 + 2 障碍 + sim2sim | 2000 | 2 seeds；source ≈0.78，target ≈0.80 | embodied_sim/MZX_2000_SIM2SIM_REPORT.md |
| MZX 2000 + 3 障碍 + 5 seeds | 2000 | source ≈0.596，target ≈0.548；延迟 0.2234 ms | embodied_sim/MZX_2000_FINAL_REPORT.md |
| MZX 2000 + 2 障碍 + 5 seeds | 2000 | source 0.824；target 0.762；fixed3 0.474；延迟 0.2191ms | embodied_sim/MZX_2OBS_5SEEDS_REPORT.md |
| MZX 2000 + 生长逻辑 | 2000→2050 | BC 0.88；生命周期 0.83；PPO 0.88；target 0.76 | embodied_sim/MZX_2000_GROWTH_REPORT.md |
| MZX 生长 v3：3 障碍 5 seeds | 2000→2080 | BC 58–76%；source≈0.686；target≈0.632；PPO 全跳过 | embodied_sim/MZX_GROWTH3_REPORT.md |
| MZX 生长 v2：2 障碍 5 seeds | 2000→2080 | source≈0.778；target≈0.760；fixed3≈0.424 | embodied_sim/MZX_GROWTH2_REPORT.md |
| 触发式 DAgger + 安全盾 | 500 | BC 0.48 → DAgger 0.88；shield 0.84 | embodied_sim/TRIGGERED_DAGGER_REPORT.md |
| 核心消融 A–E | 500 | A/B/D/E 0.8067；C 0.7600；生长无贡献 | embodied_sim/CORE_ABLATION_REPORT.md |
| G1–G4 gate/local/3固定障碍 | 500 | G1/G3 拒绝；G2 接受但 0.5067；base 0.49 | embodied_sim/GATED_GROWTH_REPORT.md |
| 3 障碍 BC + 生长数 sweep | 500 | BC 0.86；5/10/20/40 无提升；基线 0.85–0.89 | embodied_sim/GROWTH_SWEEP_3OBS_REPORT.md |
| 生长数量级 + MuJoCo 2 障碍 | 500→2500 | +100/250 最佳：2随机 0.48；+2000 崩 | embodied_sim/GROWTH_SCALE_MUJOCO_REPORT.md |
| 3 seeds 细扫 | 500→1750 | 3固定 +50/+100 0.86；2随机 +150 0.4233 | embodied_sim/GROWTH_SCALE_3SEEDS_REPORT.md |
| 2 障碍单独训练 | 500 | BC 0.84；DAgger 0.85 | embodied_sim/BC_2OBS_DIRECT_REPORT.md |
| 2 障碍生长 3 seeds | 500→1250 | +100 最优 0.8733；baseline 0.8567 | embodied_sim/GROWTH_2OBS_DIRECT_REPORT.md |
| 主基线 2/3 障碍 +100 | 500→1000 | 2障碍 0.84；3障碍 0.93 | embodied_sim/MAIN_BASELINES_REPORT.md |
| F/G/H 高级消融 | 500 | F=G=0.72；H 0.05/0.10/0.25 全 0.36 | embodied_sim/ADVANCED_FGH_REPORT.md |
| J 初版：2 障碍安全层 | 1000 | baseline 0.82；shield 0.91 | embodied_sim/SAFETY_2OBS_REPORT.md |
| I 收口：gated local | 1000→1150 | base 0.8567；best 0.8967；3 seeds | embodied_sim/GATED_LOCAL_FINAL_REPORT.md |
| J：I 模型安全层阈值 | 1150 | baseline 0.89；threshold .30 → 0.92 | embodied_sim/SAFETY_THRESHOLD_I_REPORT.md |
| 增加障碍测试 1/2/3 | 1150 | 1:0.88/1.00；2:0.89/0.95；3:0.74/0.81 | embodied_sim/OBSTACLE_SCALING_REPORT.md |

## 5. 时间版本

| 实验 | 节点 | 结果 | 文件 |
| --- | ---: | --- | --- |
| temporal parity | 2000 | test acc 0.510 | TEMPORAL_RESERVOIR_REPORT.md |
| delayed match | 2000 | test acc 0.9733 | TEMPORAL_RESERVOIR_REPORT.md |

说明：

> 时间版本目前只在 2000 节点上验证；尚未扩到 10k。

---

## 6. 7D / LLM 结果

| 内容 | 结果 | 文件 |
| --- | --- | --- |
| 7D 状态机 | E/A/L/C/B/S/P + GRU 跑通 | web_app.py |
| Version A 软注入 | C 消融；状态改变长度/采样/拒绝 | ABLATION_REPORT.md |
| Version B soft prompt | 能进入 inputs_embeds，无稳定情绪语义 | HF_PROBE_REPORT.md |
| Version B steering | 第 12 层探针；改变话题，不产生稳定 panic/pain | HF_STEERING_REPORT.md |
| Graph → LLM bridge | 10k 图 → 8 soft tokens → Qwen2.5-3B；能注入，随机情况下不稳定 | GRAPH_TO_LLM_REPORT.md |
| Graph adapter 训练 | 8/4 小样本；graph test loss 2.3574，constant 2.2729；graph 无优势 | GRAPH_LLM_TRAINED_REPORT.md |

---

## 7. 关键结论

1. 7D 状态能调制 LLM 行为，不能稳定改变语义人格。
2. 软注入改变不了人格；底层注入能改变输出，但还没学会情绪。
3. 图生长能形成 hub、灾难、重组、突变；拓扑变化不等于能力。
4. 10k 规模能跑通，但中文 next-character 只有 7%–8%。
5. 高难线性代数基本失败；瓶颈是特征表达，不是节点数。
6. 化学场有调制作用，但不是当前主因；credit 分配这次不如 broadcast。
7. 时间版本有记忆：delayed match 0.9733，但 12-bit parity 0.510。

---

## 8. 可视化入口

- fair_10k_dashboard.html
- thought_chain.html
- temporal_timeline.html
- index.html

---

## 9. 文件命名规则

- REPORT.md：实验报告
- _result.json：原始结果
- .png：图
- .py：可复现实验脚本
- data/：中文语料与来源

---

## 10. 备注

> 2000 节点实验保留为消融和 PoC；  
> 37 节点 parity 保留为神经递质分支；  
> 10k fresh 作为当前研究基线；  
> 10k fair 作为结构对照。
