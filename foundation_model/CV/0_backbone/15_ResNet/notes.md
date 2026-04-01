# Deep Residual Learning for Image Recognition -- 学习笔记

> 一句话: 通过 shortcut connection 让网络学习残差映射 F(x) 而非完整映射 H(x), 使上百层深度网络的训练成为可能。
> 论文: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research), 2015, CVPR 2016

## 这篇论文解决了什么问题

深度网络的 **degradation problem (退化问题)**: 当 plain network 层数增加时, training error 反而上升。注意这不是 overfitting -- 56 层 plain net 的 training error 比 20 层的还高。这说明更深的网络更难优化, 而非容量不够。

理论上, 一个更深的网络至少可以通过让多出来的层学成 identity mapping 来达到和浅网络一样的性能。但实际上 SGD 优化器找不到这个解。问题的根源是: 让一组非线性层逼近 identity mapping 非常困难。

## 核心想法 (用直觉解释)

**与其让网络直接学习目标映射 H(x), 不如让它学习残差 F(x) = H(x) - x。**

直觉: 如果最优映射接近 identity, 那么把残差推向零比从头学一个 identity 容易得多。网络只需要学"在 identity 基础上做多少微调", 而不是"从零学完整变换"。

实现极其简单: 在每隔几层的输出上加一条 shortcut connection, 直接把输入 x 加到输出上:
```
y = F(x, {W_i}) + x
```
这条 shortcut 不引入额外参数, 不增加计算量, 不需要修改优化器。

## 关键设计决策

**1. Identity shortcut vs. projection shortcut**

三种方案: (A) 维度不匹配时 zero-padding; (B) 维度不匹配时用 1x1 conv projection, 其余用 identity; (C) 全部用 projection。实验表明 B 略好于 A, C 比 B 好一点但引入更多参数。论文选择 B -- identity shortcut 是核心, projection 只在必要时使用。设计哲学: **免费的 information highway 比学出来的 gating 更有效** (对比 Highway Network 用 learned gate, 效果反而不如 ResNet)。

**2. Bottleneck block (1x1 -> 3x3 -> 1x1)**

ResNet-50/101/152 使用 bottleneck: 先 1x1 conv 降维, 再 3x3 conv, 再 1x1 conv 升维。关键: identity shortcut 连接高维两端, 如果换成 projection shortcut, 时间复杂度和模型大小都翻倍。Identity shortcut 对 bottleneck 的效率至关重要。152 层 ResNet (11.3B FLOPs) 仍低于 VGG-19 (19.6B FLOPs)。

**3. BN everywhere + no dropout**

BN 放在每个 conv 之后、activation 之前。不使用 dropout, 靠深度本身提供正则化。这个模式后来成为 CNN 时代的标配。

**4. 层响应分析验证核心假设**

ResNet 各层的输出响应标准差普遍小于对应 plain net, 且越深的 ResNet 响应越小 (Fig. 7)。验证了: residual function 确实接近零, 网络在做"对 identity 的微小扰动"而非学习全新的变换。

**5. 实验结果亮点**

- ResNet-34 training error 低于 plain-34, 逆转了 degradation (Fig. 4)
- 152 层 ResNet ensemble: ImageNet top-5 error 3.57%, ILSVRC 2015 冠军
- COCO detection: ResNet-101 替换 VGG-16 后 mAP@[.5,.95] 从 21.2 提升到 27.2 (+28% 相对提升)
- CIFAR-10: 1202 层 ResNet 可以训练 (training error <0.1%), 但 test error 不如 110 层 (overfitting)

## 这篇论文之后发生了什么

- **成为 CV 默认 backbone (2016-2020)**: 检测 (Faster R-CNN)、分割 (Mask R-CNN, DeepLab)、生成任务几乎都用 ResNet
- **ResNet 变体爆发**: ResNeXt (分组卷积), Wide ResNet, DenseNet (dense connection), SE-Net (channel attention)
- **Pre-trained ResNet 定义了 transfer learning**: ImageNet pre-trained ResNet-50 是 CV 下游任务的默认初始化
- **ViT (2020) 开始挑战 ResNet**, 但 ResNet-50 至今仍是 SSL (DINO, MoCo, MAE) 的标准对比 baseline
- **Residual connection 泛化到 Transformer**: 每个 sublayer 都用 x + Sublayer(x), 直接继承 ResNet 思想

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|----------|
| 1 | **Residual connection 是深度网络的基础设施** -- 没有它就没有 Transformer, 没有 Transformer 就没有 foundation model | 你用的每个 FM (GPT, CLIP, DINO) 内部都有 residual connection, 它是梯度流过上百层的根本保障 |
| 2 | **让网络学"修正量"而非"完整映射"是强大的 inductive bias** | RL 中 residual policy (在 base policy 上学残差) 比从零学更容易; Diffusion Policy 的 denoising 也是在学残差 |
| 3 | **Identity shortcut 比 learned gating 更有效** -- Highway Network 用 learned gate 反而不如无参数 identity shortcut | 设计 robot policy 网络时, 简单的 skip connection 往往比复杂 gating 更可靠 |
| 4 | **Pre-trained backbone 范式从 ResNet 开始** -- "ImageNet pre-train + downstream fine-tune" | 这就是 robotics FM 的前身: 预训练视觉 encoder (R3M, VIP, DINO) 然后在 robot task 上微调 |
| 5 | **深度比宽度重要, 但有上限** -- ResNet-152 远超 ResNet-18, 但 1202 层开始 overfit | Scaling law 的早期信号: robot FM 也需要找到 depth/width 的 sweet spot |
