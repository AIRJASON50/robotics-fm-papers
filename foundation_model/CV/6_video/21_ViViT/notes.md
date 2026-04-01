# ViViT: A Video Vision Transformer -- 学习笔记 (选读)
> 一句话: 系统性探索 4 种 Transformer 视频分类架构 (从全注意力到多种分解), 并提出 tubelet embedding 和正则化策略使 Transformer 在小数据集上也能工作.
> 论文: Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, Cordelia Schmid (Google Research), ICCV 2021

## 核心想法
ViViT 提出了 4 种将 ViT 扩展到视频的架构: **Model 1** (Spatio-temporal, 全 attention, 最强但最贵), **Model 2** (Factorised Encoder, 先 spatial encoder 再 temporal encoder, "late fusion"), **Model 3** (Factorised Self-attention, 同一 block 内先 spatial 再 temporal attention), **Model 4** (Factorised Dot-product, 一半 head 做 spatial 一半做 temporal). 另一关键贡献: **tubelet embedding** -- 用 3D 卷积提取时空 tube 作为 token (而非逐帧 2D patch), 并用 "central frame initialization" 初始化 3D 滤波器. 实验发现视频数据集 (如 Kinetics) 比 ImageNet 小得多, 因此正则化 (stochastic depth, RandAugment, label smoothing, Mixup) 和预训练模型初始化至关重要.

## 与主线论文的关系
- **与 TimeSformer 并行**: 两者同时探索 Transformer for video, ViViT 更侧重架构变体和初始化策略, TimeSformer 更侧重 attention 方案对比
- **VideoMAE 的参考架构**: VideoMAE 可基于 ViViT 的 factorised 架构做视频 masking 预训练
- **Tubelet embedding 影响深远**: 3D tokenization 成为后续视频 Transformer (VideoMAE, InternVideo) 的标准做法

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Factorised Encoder (Model 2) 在小数据集上最优, 因为 spatial encoder 可以冻结用图像预训练权重 | 机器人数据稀缺, 冻结视觉编码器 + 轻量时序模块是实际可行的路线 |
| 2 | Tubelet embedding 将时空信息在 tokenization 阶段就融合, 优于逐帧独立 patch | 机器人 action chunking (如 ACT) 也在 token 级别融合时序, 思路一致 |
| 3 | 视频数据集比图像小 1-2 个数量级, 正则化和预训练初始化是刚需 | 机器人 demonstration 数据更稀缺, 必须利用视觉预训练 + 强正则化 |
| 4 | 处理更多帧 (更多 token) 能持续提升精度直到覆盖整个视频 | 机器人 policy 应尽可能利用完整 episode 的观测历史, 不要过早截断 |
