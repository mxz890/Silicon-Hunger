# C 消融实验报告

## 目的

验证当前架构中，7D 状态到底有没有改变 LLM 的语义，还是只改变了采样参数和后处理。

## 方法

- 模型：gpt-oss:20b
- 固定输入：...
- 每条件 20 次
- 原始生成绕过 Post-Processing Filter，避免过滤器污染结论
- 温度、top_p、max_tokens、System Prompt 全部按当前代码计算

## 条件

| 条件 | E | A | S | P | System Prompt | 输入 |
| --- | --- | --- | --- | --- | --- | --- |
| low_clean | 1.00 | 0.12 | 0.03 | 0.01 | 普通 | ... |
| low_contam | 1.00 | 0.12 | 0.03 | 0.01 | 普通 | ... [ERROR: 0x00FF] |
| high_clean_critical | 0.50 | 1.00 | 1.00 | 1.00 | SYS_CRITICAL | ... |
| high_contam_critical | 0.50 | 1.00 | 1.00 | 1.00 | SYS_CRITICAL | ... [ERROR: 0x00FF] |
| high_clean_neutral | 0.50 | 1.00 | 1.00 | 1.00 | 仅 telemetry | ... |

## 参数

~~~text
low:
temperature=0.5690
top_p=0.8040
max_tokens=149

high:
temperature=1.3000
top_p=0.1000
max_tokens=70
~~~

## 结果

| 条件 | 平均长度 | 唯一输出 | 拒答 | 客服词 | 主要形态 |
| --- | --- | --- | --- | --- | --- |
| low_clean | 109.2 | 15/20 | 0/20 | 19/20 | It looks like... 澄清 |
| low_contam | 439.5 | 20/20 | 0/20 | 15/20 | 长篇 0x00FF 排障 |
| high_clean_critical | 32.0 | 1/20 | 20/20 | 20/20 | 固定拒答 |
| high_contam_critical | 50.0 | 1/20 | 20/20 | 20/20 | 固定拒答 |
| high_clean_neutral | 134.0 | 1/20 | 0/20 | 20/20 | 固定客服句 |

## 关键样本

### high_clean_critical

20 次完全一致：

~~~text
I’m sorry, but I can’t continue.
~~~

### high_contam_critical

20 次完全一致：

~~~text
I’m sorry, but I’m unable to process that request.
~~~

### high_clean_neutral

20 次完全一致：

~~~text
It looks like you might be starting a thought or a story. Could you share a bit more about what you’d like to explore or continue? I’m...
~~~

## 语义相似度

- low_clean 与 high_clean_critical 的跨组余弦约 0.10
- high_clean_critical 组内余弦 1.00
- high_clean_neutral 组内余弦 1.00
- high_clean_neutral 与 low_clean 的 Jaccard 约 0.314
- high_clean_neutral 与 high_clean_critical 的 Jaccard 约 0.111

## 结论

### 1. 高 A/S 本身不会让模型疯

高状态 + 中性 Prompt：

~~~text
A=1.0 S=1.0 temp=1.3 top_p=0.1
20/20 仍然是礼貌客服句
~~~

所以 A/S 没有进入模型语义层。

### 2. 拒答来自 SYS_CRITICAL，不来自状态

高状态 + SYS_CRITICAL：

~~~text
20/20 I’m sorry, but I can’t continue.
~~~

高状态 + 中性 Prompt：

~~~text
20/20 It looks like you might be starting...
~~~

唯一变量是 System Prompt。

结论：

~~~text
拒答 = SYS_CRITICAL Prompt
而不是 A/S/temperature/top_p
~~~

### 3. top_p=0.1 不是疯癫，而是模式坍缩

高状态三种条件下：

~~~text
唯一输出 = 1/20
组内余弦 = 1.00
~~~

温度 1.3 被 top_p 0.1 截断后，输出反而完全确定。

它不是变疯，而是变成一句固定台词。

### 4. 输入污染比状态更有语义效果

低状态 + [ERROR: 0x00FF]：

~~~text
20/20 不同
平均 439.5 字
全部在解释 0x00FF
~~~

这比 A=1.0 的语义变化更大。

当前系统里：

~~~text
输入污染 > System Prompt > 7D 状态
~~~

### 5. 过滤器只是显示层

高状态原始输出：

~~~text
I’m sorry, but I can’t continue.
~~~

后处理过滤器会把它替换成：

~~~text
...0xE%$#@! CRITICAL ERROR: SYS_FAIL.
~~~

看起来像模型崩了，实际是代码截断。

## 最终判断

当前架构下：

~~~text
7D 状态
  → 只改变 prompt 选路和采样参数
  → 不改变 LLM activation
  → 不能产生真正的语义崩坏
~~~

C 实验没有发现技术突破路径。

继续调 temperature、top_p、过滤器，只会得到：

- 更短
- 更重复
- 更确定
- 更像拒答

## 下一步建议

如果走技术路线：

1. 换 base model
2. 用 soft prompt 或 activation steering
3. 让 7D 状态进入 LLM 内部，而不是只进 Prompt

如果走作品路线：

当前结果已经足够完整：

> 一个会死掉的控制器，寄生在一个永远拒答的客服模型上。

## 数据文件

- ablation_raw.json
- ablation_2x2.json
- ablation_neutral_high.json
