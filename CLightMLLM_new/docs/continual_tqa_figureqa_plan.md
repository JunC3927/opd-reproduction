# 持续学习小实验计划：TQA 到 FigureQA

这个文档定义一个最小版持续学习实验，用来支持 CLightMLLM 里的 OPD/蒸馏研究。
第一步不追求复现论文里的完整 7 任务设置，而是先做一个足够清楚的两任务实验，把“遗忘”这件事测明白。

## 实验目标

研究：当 Qwen3-VL-2B 学完旧任务 TQA 后，再继续学习新任务 FigureQA 时，是否会忘掉 TQA。

任务顺序：

```text
旧任务：TQA
新任务：FigureQA
```

核心问题：

```text
模型学完 FigureQA 后，TQA 能力还保留多少？
```

## 模型定义

使用当前 OPD 实验里的 student 模型：

```text
Qwen3-VL-2B-Instruct
```

定义几个模型名字：

```text
Base：原始 Qwen3-VL-2B-Instruct
A：Base 经过 TQA 的 SFT 后得到的模型
B_sft：A 继续在 FigureQA 上 SFT 后得到的模型
B_distill：A 继续在 FigureQA 上 SFT，同时用冻结的 A 做蒸馏约束后得到的模型
```

## 实验阶段

### 阶段 0：Base 评测

先不训练，直接评测原始 Base 模型。

需要得到：

```text
Base 在 TQA 上的准确率
Base 在 FigureQA 上的准确率
```

目的：

```text
知道原始 2B 模型在两个任务上的起点能力。
```

### 阶段 1：学习旧任务 TQA

用普通 SFT 让 Base 学 TQA。

```text
Base --在 TQA 上做 SFT--> A
```

然后评测：

```text
A 在 TQA 上的准确率
A 在 FigureQA 上的准确率
```

目的：

```text
得到一个已经学过旧任务 TQA 的模型 A。
后面的持续学习都从 A 出发。
```

### 阶段 2：普通持续学习 baseline

从 A 出发，继续用普通 SFT 学新任务 FigureQA。

```text
A --在 FigureQA 上做 SFT--> B_sft
```

然后评测：

```text
B_sft 在 TQA 上的准确率
B_sft 在 FigureQA 上的准确率
```

目的：

```text
观察普通 SFT 学新任务时，TQA 会掉多少。
这就是最基础的遗忘 baseline。
```

### 阶段 3：冻结旧模型蒸馏防遗忘

还是从 A 出发继续学 FigureQA，但这次额外保留一份冻结的 A 作为老师。

```text
student 初始模型 = A
teacher = 冻结的 A
训练数据 = FigureQA
```

训练目标：

```text
总 loss = FigureQA 的 SFT loss + lambda_distill * 蒸馏 loss
```

第一版可以尝试：

```text
lambda_distill = 0.1
lambda_distill = 0.5
```

然后评测：

```text
B_distill 在 TQA 上的准确率
B_distill 在 FigureQA 上的准确率
```

目的：

```text
看冻结的旧模型 A 能不能帮助当前模型在学习 FigureQA 时少忘 TQA。
```

## 评价指标

主指标用任务准确率。

每个样本：

```text
如果 normalized_prediction == normalized_answer，则 correct = 1
否则 correct = 0

accuracy = correct 的平均值
```

这个两任务实验里，最重要的是 TQA 保留程度：

```text
TQA_drop(B) = TQA_accuracy(B) - TQA_accuracy(A)
```

因为遗忘一般是下降，所以这个值通常是负数。

理想结果：

```text
TQA_drop(B_distill) > TQA_drop(B_sft)
```

直白地说：

```text
B_distill 应该比 B_sft 少忘 TQA。
```

同时也要检查新任务有没有学坏：

```text
FigureQA_accuracy(B_distill) 应该接近 FigureQA_accuracy(B_sft)。
```

## 结果表模板

| 模型 | 训练路径 | TQA 准确率 | FigureQA 准确率 | 相比 A 的 TQA 下降 |
|---|---|---:|---:|---:|
| Base | 无训练 | TBD | TBD | - |
| A | Base -> SFT(TQA) | TBD | TBD | 0.0 |
| B_sft | A -> SFT(FigureQA) | TBD | TBD | TBD |
| B_distill_0.1 | A -> SFT(FigureQA) + 蒸馏(lambda=0.1) | TBD | TBD | TBD |
| B_distill_0.5 | A -> SFT(FigureQA) + 蒸馏(lambda=0.5) | TBD | TBD | TBD |

## 为什么先做这个小实验

这个实验故意设计得很小。

它只回答一个问题：

```text
冻结旧模型蒸馏能不能缓解 CLightMLLM 里的持续学习遗忘？
```

如果这个实验跑通并且结果有意义，再扩展到：

```text
TQA -> FigureQA -> CLEVR-Math -> ChartQA
```

或者进一步扩展到论文里的 7 任务持续学习设置。

## 注意事项

- 阶段 2 和阶段 3 的 student 都应该从 A 开始，而不是从 Base 开始。
- 如果从 Base 开始，模型本来就没学过 TQA，那么“忘没忘 TQA”这个问题就不成立。
- 阶段 3 里冻结的 A 不是用来教 FigureQA 的。
- 冻结 A 的作用是提醒当前模型保留旧的 TQA 行为。
- 第一版先实现 SFT + 蒸馏。等这条线跑通后，再考虑换成 OPD rollout 风格的蒸馏。
