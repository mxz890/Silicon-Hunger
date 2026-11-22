# BC 93% → PPO 微调 v2：KL 0.2 / BC 0.3 / early stop

日期：2026-11-22  
模型：MZX-0.1 / 500 节点  
任务：MuJoCo 点质量 + 2 个随机障碍  
设置：BC 基线 → PPO 微调

---

## 1. 设置

- BC checkpoint：0.9333
- PPO：20 updates
- KL 到 BC：**0.20**
- BC loss：**0.30**
- 验证：30 episodes
- 连续 2 次验证不提升 → 早停
- 保存 best checkpoint

---

## 2. 结果

| 阶段 | success | steps |
| --- | ---: | ---: |
| BC 基线 | **0.9333** | 118.70 |
| PPO update 0 | 0.8667 | 118.80 |
| PPO update 5 | 0.9333 | 125.50 |
| 早停 | — | — |
| best checkpoint 回测 | 0.9000 | 119.67 |
| 强目标回测 | 0.9333 | 112.87 |

best_update = -1，表示最终保留的是 **BC 模型**，不是 PPO 模型。

---

## 3. 结论

> KL/BC 加强后，PPO 不再坍缩，但也没有超过 BC。

- update 0：0.8667，下降
- update 5：0.9333，追平 BC
- 连续 2 次不提升 → 早停
- 最终 best 仍是 BC

所以：

> 强 BC anchor 解决了“坍缩”，但也把 PPO 锁在 BC 附近。

---

## 4. 这次说明了什么

能说明：

- 强 KL/BC 可以防止 PPO 漂移
- early stop 可以避免最后一轮变差
- best checkpoint 可以保住 BC 基线

不能说明：

- PPO 不能提升
- BC 一定最好

只说明：

> 当前 PPO 的探索空间被 KL/BC 限制住了，没有找到比 BC 更好的策略。

---

## 5. 下一步

1. 降低 KL 到 0.10，观察是否能小幅提升
2. 增加 PPO 数据量，而不是增加 updates
3. 用固定目标先做小范围 PPO
4. 用 KL 退火：早期强，后期弱
5. 保留 best checkpoint 和 early stop

---

## 6. 一句话

> KL 0.2 + BC 0.3 + 早停让 PPO 不再坍缩，但也只追平 BC 93%，没有超过；当前 best 仍是 BC。

---

## 7. 文件

- bc_to_ppo.py
- bc_to_ppo_result.json
- BC_TO_PPO_V2_REPORT.md
