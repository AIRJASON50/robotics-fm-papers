# Batch Normalization (Ioffe & Szegedy, 2015) -- Takeaway Notes

> 一句话: 在每层对 mini-batch 的激活值做归一化, 解决 internal covariate shift, 让深网训练更快更稳.

## 核心贡献
- 在网络每一层插入 normalize -> scale (gamma) -> shift (beta) 操作, 参数可学习
- 训练时用 mini-batch 统计量, 推理时用 running mean/var -- 引入了 train/eval 模式差异
- 实验证明: 可以用更大 learning rate, 减少对初始化的敏感度, 有轻微正则化效果

## 为什么重要
BatchNorm 是让 CNN 从"难训练"变成"开箱即用"的关键技术. ResNet, EfficientNet 等
vision backbone 都依赖 BatchNorm. 它也暴露了一个基本 tradeoff: batch 统计量让训练
更稳定, 但在 batch size 小或序列模型中效果差, 这直接催生了 LayerNorm.

## 对你 (RL->FM) 的 Takeaway
- RL 的 policy network 通常不用 BatchNorm (batch size 不稳定, on-policy 数据分布
  持续变化), 而是用 LayerNorm 或不加 normalization. 理解 BatchNorm 的局限才能理解
  为什么 RL 社区做了不同选择.
- 如果你用 pre-trained CNN (如 ResNet) 做 visual encoder, 冻结时务必设 eval() mode,
  否则 running stats 会被 RL 数据污染, 导致 sim2real 性能崩溃.

## 与知识库其他内容的关联
- 16_LayerNorm: 解决 BatchNorm 在序列/小 batch 场景的不足, Transformer 标配
- 15_Adam: BatchNorm + Adam 共同让深网训练变得实际可行
- 15_DQN: DQN 时期的网络还没有用 normalization, 训练不稳定是常态
