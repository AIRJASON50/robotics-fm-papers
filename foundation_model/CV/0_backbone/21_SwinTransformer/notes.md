# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows -- 学习笔记 (选读)
> 一句话: 通过 hierarchical feature maps + shifted window attention 让 ViT 具备线性计算复杂度和多尺度能力, 成为真正的通用视觉 backbone.
> 论文: Ze Liu, Yutong Lin, Yue Cao, Han Hu, et al. (Microsoft Research Asia), ICCV 2021 Best Paper

## 核心想法
ViT 的两大问题阻碍其成为通用 backbone: (1) 全局 self-attention 是 O(n^2), 高分辨率图像计算不可行; (2) 单一分辨率 feature map 无法支持检测/分割等密集预测任务. Swin Transformer 的解法: 在局部 **non-overlapping windows** (默认 7x7 patches) 内做 self-attention (线性复杂度), 通过相邻层间 **shifted window** 引入跨窗口连接. 同时用 **patch merging** 逐层降低分辨率 (4x->8x->16x->32x), 产出类似 CNN 的多尺度 hierarchical feature maps, 可直接替换 ResNet 接入 FPN/U-Net 等下游框架. 核心工程技巧: 用 cyclic shift + masking 实现 shifted window 的高效 batch 计算.

## 与主线论文的关系
- **ViT 的工程化落地**: ViT 证明了 Transformer 可以做视觉, Swin 让它在实际任务中可用 (检测/分割/高分辨率)
- **为后续工作提供 backbone**: DINOv2、SAM、VideoMAE 等都支持 Swin 作为 backbone 选项
- **CNN 归纳偏置的回归**: 局部窗口 attention + 层级结构本质上是在 Transformer 中注入 CNN 的 locality + hierarchy 先验

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 多尺度 hierarchical feature map 对密集预测至关重要 -- 这是 ViT 欠缺的 | 机器人需要精确的空间理解 (抓取点/接触面), Swin 的多尺度结构比 ViT 更适合提供像素级特征 |
| 2 | Shifted window = 局部计算 + 跨区域连接, 平衡了效率与建模能力 | 机器人处理高分辨率输入 (如灵巧手的触觉图) 时, 局部 attention 是必须的工程选择 |
| 3 | Relative position bias 优于 absolute position embedding, 支持不同分辨率 fine-tuning | 机器人视觉输入尺寸经常变化 (不同相机), relative position bias 更实用 |
| 4 | 好的 vision backbone 设计: 输入端高分辨率细粒度, 输出端低分辨率高语义 | 这正是机器人需要的: 既要看到手指接触的细节, 又要理解整个场景的语义 |
