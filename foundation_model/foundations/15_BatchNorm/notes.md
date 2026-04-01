# Batch Normalization: Accelerating Deep Network Training -- 学习笔记
> 一句话: 在每层激活值上用 mini-batch 统计量做归一化, 解决 internal covariate shift, 让深网能用大学习率快速稳定训练.
> 论文: Sergey Ioffe, Christian Szegedy (Google), 2015, ICML 2015

## 这篇论文解决了什么问题
深度网络训练中, 每层输入的分布随着前面层参数更新而不断变化 -- 论文称之为 internal covariate shift (ICS, 内部协变量偏移). 这迫使使用小学习率, 精心初始化, 且 saturating nonlinearity (如 sigmoid) 几乎不可用. 训练深层网络非常困难, 需要大量 tricks.

## 核心想法 (用直觉解释)
既然每层输入分布不稳定是问题根源, 那就强制稳定它: 对每个 mini-batch 计算 activation 的 mean 和 variance, 标准化到零均值单位方差, 然后用可学习的 gamma (scale) 和 beta (shift) 恢复表达能力. 这样每层总是看到分布稳定的输入, 梯度流动更顺畅, 可以放心用大学习率. 训练时用 mini-batch 统计量, 推理时用训练阶段累积的 running mean/variance.

## 关键设计决策
- **Per-dimension 归一化**: 对每个 scalar feature 独立做归一化 (不做 whitening), 计算简单且效果好. 完整 whitening 需要协方差矩阵逆, 计算量太大
- **可学习 gamma 和 beta**: 如果 gamma=sqrt(Var), beta=E[x], 就恢复了原始激活. 这保证 BN 至少不比没有 BN 差, 网络可以 "选择" 要不要归一化
- **放在 nonlinearity 之前**: 归一化 Wu+b 而非 g(Wu+b), 因为仿射变换的输出更可能是对称 Gaussian-like 分布
- **CNN 中 per-channel 归一化**: 同一 feature map 的所有空间位置共享 gamma/beta (effective batch = m*p*q), 保留卷积的平移不变性

## 这篇论文之后发生了什么
ResNet (2015) 和几乎所有 CNN backbone 都采用 BatchNorm. 但其缺点暴露: (1) 依赖 batch size, 小 batch 性能差; (2) train/eval 行为不一致 (running stats); (3) 不适合 RNN/Transformer (变长序列). Layer Normalization (Ba 2016) 解决了这些问题并成为 Transformer 标配. Group Normalization (Wu 2018) 介于两者之间. 后续研究 (Santurkar 2018) 质疑 ICS 才是 BN 奏效的真正原因, 提出 BN 的核心作用是 smoothing loss landscape.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | RL 的 policy network 通常不用 BatchNorm | RL 的 on-policy 数据分布持续变化, batch 统计量不可靠, 应用 LayerNorm 或不加 normalization |
| 2 | 用 pre-trained CNN 做 visual encoder 时必须设 eval() mode | 冻结 ResNet 时如果忘记 model.eval(), running stats 会被 RL 数据污染, sim2real 性能崩溃 |
| 3 | BN 使训练对初始化和学习率更鲁棒 | BN(aW) = BN(W) -- 权重 scale 不影响输出, 这给了你更大的超参搜索空间. 但在 RL fine-tune 时要注意冻结 BN 层 |
| 4 | BN 的正则化效果可替代 Dropout | 在 batch-normalized 网络中, 每个样本的归一化依赖同 batch 其他样本, 引入噪声起到正则化作用 |
