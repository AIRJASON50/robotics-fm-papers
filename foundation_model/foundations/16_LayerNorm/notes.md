# Layer Normalization -- 学习笔记
> 一句话: 将归一化维度从 batch 维改为 feature 维 (对单个样本的所有 hidden units 做归一化), 彻底消除对 batch size 的依赖, 成为 Transformer 的标准组件.
> 论文: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (U of Toronto), 2016, arXiv 1607.06450

## 这篇论文解决了什么问题
BatchNorm 依赖 mini-batch 统计量, 存在三个问题: (1) batch size 小时统计量估计不准; (2) RNN 中每个 time step 的激活分布不同, 需要为每步维护独立统计量, 且测试序列可能比训练时更长导致统计量不匹配; (3) 无法用于 online learning (batch size = 1). 这些限制使 BatchNorm 难以应用于 RNN 和 Transformer 等序列模型.

## 核心想法 (用直觉解释)
BatchNorm 是 "跨样本, 单特征" 做归一化; LayerNorm 反过来, "单样本, 跨特征" 做归一化. 对于一层的所有 hidden units, 计算它们在当前样本上的 mean 和 variance, 然后标准化. 每个样本独立处理, 不依赖 batch 中的其他样本. 训练和推理行为完全一致, 无需维护 running stats.

## 关键设计决策
- **归一化维度**: mu^l = (1/H) * sum(a_i^l), sigma^l = sqrt((1/H) * sum((a_i^l - mu^l)^2)). 所有 hidden units 共享归一化统计量, 但每个训练样本有自己的 mu 和 sigma
- **可学习 gain (g) 和 bias (b)**: 与 BatchNorm 的 gamma/beta 对应, 恢复网络表达能力. 一组 g 和 b 在所有 time step 共享, 适合 RNN
- **对 RNN 的天然适配**: RNN 在不同 time step 共享权重, BatchNorm 要为每步存统计量; LayerNorm 的统计量只依赖当前步的激活, 完全不受序列长度限制
- **不同的 invariance 性质**: LayerNorm 对整个 weight matrix 的 scaling 和 shift 不变 (BN 只对单个权重向量的 scaling 不变), 但 LayerNorm 对单个 training case 的 re-scaling 不变 (BN 不具备). 见 Table 1

## 这篇论文之后发生了什么
Vaswani (2017) 在 Transformer 中采用 LayerNorm (Add & Norm), 从此 LayerNorm 成为 NLP/LLM 的标准组件. Pre-LN (GPT-2 style, LN 放在 attention/FFN 之前) 比 Post-LN (原始 Transformer, LN 放在之后) 训练更稳定, 成为主流. RMSNorm (Zhang 2019) 去掉 mean centering 只保留 variance scaling, 进一步简化计算, 被 LLaMA 等模型采用. Group Normalization (Wu 2018) 介于 BN 和 LN 之间, 在 CV 中流行.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | RL 的 actor/critic MLP 应用 LayerNorm 而非 BatchNorm | RL 的 batch 内数据分布不稳定 (on-policy 持续变化), BatchNorm 统计量不可靠. LayerNorm 每样本独立, 更适合 |
| 2 | Pre-LN vs Post-LN 对训练稳定性影响大 | Pre-LN (LN 在 residual 分支内部) 缓解梯度爆炸, 是 GPT-2/3 和大多数 VLA (RT-2, Octo) 的选择 |
| 3 | LayerNorm 隐式降低有效学习率 | 权重 norm 越大, sigma 越大, 梯度越小 -- 类似 early stopping 效果. 这让训练更稳定但也意味着大模型可能需要更大初始学习率 |
| 4 | LayerNorm + Adam + Residual = Transformer 可训练的三板斧 | 理解这三者的协同作用 (LN 稳定分布, Adam 自适应步长, Residual 保梯度) 是理解所有基于 Transformer 的 FM 的基础 |
