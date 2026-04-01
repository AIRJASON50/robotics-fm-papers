# Deep Residual Learning for Image Recognition (He et al., 2015) -- Takeaway Notes

> 一句话: Residual connection 让网络深度从 20 层跳到 152 层, 确立了 "ImageNet pre-train + downstream fine-tune" 的 foundation model 原型范式。

## 核心贡献

1. **Residual connection (shortcut / skip connection)**: 学 F(x) = H(x) - x 而非 H(x), 让梯度可以直通跳层, 解决了深度网络的 degradation problem (不是 vanishing gradient, 而是更深的网络 train loss 反而更高)。
2. **Bottleneck block**: 1x1 conv 降维 -> 3x3 conv -> 1x1 conv 升维, 让 152 层网络的计算量可控。
3. **ImageNet pre-train paradigm**: ResNet 成为 CV 领域第一个被广泛当作通用 backbone 使用的模型 -- 一次 ImageNet 预训练, 迁移到检测/分割/姿态估计等所有下游任务。

## 为什么重要

- **"越深越好"终于成立**: 在 ResNet 之前, 网络加深到一定程度性能反而下降。Residual connection 打破了这个瓶颈, 开启了 depth scaling 的时代。
- **Foundation model 思想的原型**: 虽然 2015 年没有 "foundation model" 这个词, 但 "一个 backbone, 多个任务" 的 pre-train + fine-tune 范式就是从 ResNet 开始的。
- **影响力穿透至今**: Transformer 中的 residual connection (Add & Norm) 直接来自 ResNet; ViT、DiT、VLA 全都在用。

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动关联 |
|---|----------|---------|
| 1 | **Residual = identity mapping 是深度网络训练的充要条件**。PPO 中用的 MLP policy 通常只有 2-3 层, 不需要 residual; 但 VLA 的 backbone 动辄 24-48 层, 没有 residual 根本训不动。 | 理解 VLA backbone 为什么能 scale |
| 2 | **Pre-train + fine-tune 范式**: RT-1 用 EfficientNet (ResNet 的后继), RT-2 用 ViT -- 都是 ImageNet pre-train 的 backbone。机器人视觉 backbone 的选型逻辑直接继承自 ResNet 时代。 | 理解 robot vision backbone 选型 |
| 3 | **深度 vs 宽度 tradeoff**: ResNet 证明深度比宽度更重要 (ResNet-152 > 宽但浅的 VGG-19)。同样的 insight 后来被 Transformer scaling law 验证。 | Foundation model scaling 的设计直觉 |

## 与知识库其他内容的关联

- **ViT (CV/0_backbone)**: ViT 的 Transformer block 中 residual connection + LayerNorm 直接来自 ResNet + BatchNorm 的设计 pattern
- **DiT (CV/1_generation)**: GR00T N1 的 DiT action head 中每个 block 都有 residual -- 没有 ResNet 就没有 DiT
- **PPO (foundations/17_PPO)**: RL policy network 通常太浅不需要 residual, 但当 policy 变成 VLA (百层级别), residual 重新成为必需
- **CV_技术演进图谱**: ResNet 位于 "Backbone 演进线" 的起点, 是 CNN 时代 foundation model 思想的开端
