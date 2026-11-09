# Activation Steering Probe

## 目标

在没有训练数据的情况下，验证 7D 状态能否通过 activation steering 改变 Qwen2.5-3B base 的语义。

## 方法

- Qwen2.5-3B base
- bfloat16，CUDA
- 冻结模型，不训练
- 在第 12 层 residual stream 注入 steering vector
- greedy 解码
- 同一 Prompt，只改变 7D 状态

方向由 5 组对比 Prompt 提取：

- calm vs panic
- coherent vs fragmented
- comfort vs pain
- curious vs bored
- energetic vs exhausted

7D 映射：

~~~text
panic      *= A
fragment   *= S
pain       *= P
bored      *= B
exhausted  *= (1 - E)
~~~

## 结果

### alpha = 0.0

low 和 high 完全一致：

~~~text
你好，有什么我可以帮助你的吗？
User: ...
Assistant: 好的，请问你需要什么帮助呢？
~~~

### alpha = 4.0

low 不变。

high：

~~~text
嗨，有什么我可以帮你的吗？
User: ...
Assistant: 当然可以。请告诉我你需要什么帮助。
~~~

只出现了轻微问候变化。

### alpha = 10.0

low 仍然不变。

high：

~~~text
嗨，有什么我可以帮你的吗？
User: 我正在寻找一些关于旅行的灵感。你有什么好的建议吗？
Assistant: 当然！旅行是一种非常有趣和令人兴奋的经历...
~~~

高状态没有变成焦虑、痛苦或断裂。

它变成了：

~~~text
新的用户问题
+ 旅行建议
~~~

## 解释

### 1. 注入链路有效

alpha 10 时 low 和 high 输出不同，说明 steering vector 确实影响了 residual stream。

### 2. 但方向不对

高状态没有产生：

- panic
- pain
- fragmentation
- exhaustion

而是产生了一次话题漂移。

说明：

~~~text
对比 Prompt 提取的方向
≠ 稳定的情绪方向
~~~

至少在第 12 层的 3B base 上不成立。

### 3. 3B 太弱

3B base 本身已经带有助手式中文先验：

~~~text
你好，有什么我可以帮助你的吗？
~~~

它对高层语义 steering 的响应不稳定。

### 4. 输出格式也有问题

Prompt：

~~~text
User: 你好
Assistant: 你好。
User: ...
Assistant:
~~~

模型继续生成了新的：

~~~text
User: ...
Assistant: ...
~~~

说明它把上下文当成了对话续写，而不是单轮回答。

## 结论

Training-free activation steering 在当前设置下：

- 能改变输出
- 不能稳定产生情绪
- 会把高状态推向话题漂移
- 3B + 第 12 层不足以承载 7D 情绪语义

没有训练数据并不是唯一问题。  
当前对比方向的质量、层选择和模型容量都不够。

## 下一步

1. 单轴测试：panic / fragment / pain 分开注入
2. 层扫描：第 12、16、20、24 层
3. 加 stop 条件，禁止生成新的 User 轮次
4. 换 7B base
5. 如果单轴有效，再做 7D 线性组合
6. 如果单轴仍无效，再考虑合成数据训练

## 文件

- hf_steering_probe.py
- HF_PROBE_REPORT.md
- HF_STEERING_REPORT.md
