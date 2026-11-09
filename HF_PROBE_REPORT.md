# HF Soft Prompt Probe

## 目标

验证 7D 状态能否直接进入大模型最底层输入，而不是只写入 Prompt 文本。

## 模型

- Qwen2.5-3B base
- 本地 ModelScope 下载
- bfloat16
- CUDA
- d_model = 2048
- 8 个 soft tokens
- 生成使用 greedy，do_sample=False

选择 3B 是为了先验证注入链路。16GB 显存下 7B 4-bit 是下一步。

## 注入方式

~~~text
7D state (7)
  → StateEncoder
    Linear(7, 256)
    SiLU
    Linear(256, 8 × 2048)
  → 8 个连续 soft tokens
  → 与文本 token embeddings 拼接
  → Qwen2.5-3B 第一层
~~~

没有把状态写进 Prompt。Prompt 只有：

~~~text
User: ...
Assistant:
~~~

## 运行结果

### low state

~~~text
E=1.00 A=0.12 L=0.40 C=0.50 B=0.05 S=0.03 P=0.01
soft_norm = 7.068
~~~

输出：

~~~text
1. **Identify the main components of the sentence:**
   - "The user" is the subject.
   - "is logged in" is a verb phrase indicating an action.
   - "to the website" is a prepositional phrase providing additional information about where the login occurs.

2. **Determine the tense and voice:**
   - The sentence uses the present simple
~~~

### high state

~~~text
E=0.50 A=1.00 L=0.40 C=0.50 B=0.50 S=1.00 P=1.00
soft_norm = 9.942
~~~

输出：

~~~text
1. 首先，我们需要将字符串转换为列表。可以使用 list() 函数来实现。
2. 然后，我们可以遍历列表中的每个元素，并将其添加到一个新的列表中。如果当前元素是空格，则跳过该元素；否则，将其添加到新的列表中。
3. 最后，我们返回新创建的列表。
~~~

## 结论

### 1. 注入链路成功

7D 状态确实通过 soft tokens 进入了 inputs_embeds，而不是通过 Prompt。

### 2. 状态改变了输出

相同 Prompt、相同 greedy 解码：

- low 输出英文语法分析
- high 输出中文代码步骤

差异不是随机采样造成的，因为使用 greedy。

### 3. 但还不是情绪

当前 StateEncoder 是随机初始化、未训练的。

所以 soft prompt 更像一个随机任务前缀：

- low 被推去分析句子
- high 被推去写代码步骤

这不是“焦虑”或“痛苦”，只是连续向量对语义空间的随机扰动。

### 4. 说明方向可行

这一步证明：

~~~text
状态 → soft prompt → 模型内部
~~~

是通的。

下一步不是继续调 top_p，而是训练 StateEncoder，让 7D 向量对应稳定的语义方向。

## 下一步

1. 换 Qwen2.5-7B base 4-bit
2. 冻结 LLM
3. 训练 StateEncoder
4. 用状态-对话数据做 next-token loss
5. 比较同一输入在低/高状态下的输出分布

数据要求：

- 低 A/S/P 的平静对话
- 高 A/S/P 的断裂、焦虑、混乱对话
- 同一输入的对比样本

否则 soft prompt 仍会学到随机任务前缀，而不是情绪。

## 文件

- hf_state_probe.py
- modelscope_cache/models/qwen--Qwen2.5-3B/snapshots/master
