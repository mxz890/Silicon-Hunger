# Graph Adapter 训练：graph vs constant

日期：2026-11-20  
模型：Qwen2.5-3B base（冻结）  
图：fresh 10k / 40105 边  
状态：小样本 smoke test，不是正式 benchmark。

---

## 1. 设置

- 训练样本：8 条指令
- 测试样本：4 条未见指令
- 只训练 adapter，LLM 全部冻结
- 训练 24 step，batch=1
- 两种 adapter：
  - graph：10k 图状态 → 128 维 → 8 soft tokens
  - constant：8 个可学习 soft tokens，不看图

目标：

> graph adapter 是否比 constant adapter 更能利用图状态？

---

## 2. 训练 loss

| step | graph | constant |
| ---: | ---: | ---: |
| 0 | 4.5889 | 2.8802 |
| 4 | 2.2894 | 2.2701 |
| 8 | 1.7017 | 2.0398 |
| 12 | 1.3562 | 1.4908 |
| 16 | 1.0318 | 1.1905 |
| 20 | 1.0475 | 0.8144 |

两边都能拟合训练集。

---

## 3. 测试 loss

| adapter | test loss |
| --- | ---: |
| graph | 2.3574 |
| constant | **2.2729** |

结论：

> constant adapter 反而更好。  
> 没有看到 graph 状态带来的泛化优势。

---

## 4. 生成结果

测试 prompt 上：

- baseline 能正常生成
- graph adapter 多数输出为空
- constant adapter 也多数为空

说明 adapter 训练后把模型推向了 EOS/短输出。

这是当前小样本 soft prompt 训练的失败模式，不是图本身的结论。

---

## 5. 解释

可能原因：

1. 只有 8 条训练样本，严重过拟合
2. constant adapter 更灵活，graph adapter 被 128 维随机投影限制
3. 图状态是固定随机 reservoir，没有任务相关结构
4. soft token 没有正则/长度控制
5. 没有验证集早停

---

## 6. 结论

> 这次训练没有证明 graph adapter 有优势；constant adapter 的测试 loss 更低。

更准确地说：

- adapter 能训练
- graph 状态能进入 LLM
- 但在 8 条样本上，图没有提供可泛化的增量

---

## 7. 下一步

不要再增加训练步数。  
应该改实验设计：

1. 至少 200–1000 条指令
2. 用 instruct/chat 模型
3. 加验证集和早停
4. soft token 长度/范数正则
5. 先训练 constant adapter，再训练 graph adapter
6. 用 held-out loss + 生成质量双指标

---

## 8. 一句话

> graph adapter 能训练，但这次没有超过 constant adapter；当前证据仍然不支持“图状态给 LLM 带来增量”。

---

## 9. 文件

- graph_llm_train.py
- graph_llm_trained_result.json
- GRAPH_LLM_TRAINED_REPORT.md
