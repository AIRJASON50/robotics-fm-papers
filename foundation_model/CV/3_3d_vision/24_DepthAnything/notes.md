# Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data -- 学习笔记 (选读)
> 一句话: 通过 62M 无标注图像的 self-training + 强数据增强 + DINOv2 语义对齐, 构建了零样本泛化能力最强的单目深度估计 foundation model.
> 论文: Lihe Yang, Bingyi Kang, Zilong Huang, et al. (HKU + TikTok), CVPR 2024

## 核心想法
Depth Anything 的核心不是新模块而是 **数据工程**: (1) 用 1.5M 有标注深度图像训练 teacher model, 对 62M 无标注图像生成伪标签; (2) 关键发现 -- 直接 self-training 不涨点, 必须对 student 施加**强扰动** (color jitter + Gaussian blur + CutMix), 迫使它学到比 teacher 更鲁棒的表征; (3) 用 DINOv2 的 feature alignment loss 作为辅助监督, 继承其丰富的语义先验. 使用 DINOv2 作为 encoder 初始化, affine-invariant loss 处理不同数据集间的深度 scale/shift 差异. 最终模型在 6 个公开数据集上零样本超越 MiDaS, fine-tune 后超越 ZoeDepth.

## 与主线论文的关系
- **DINOv2 的下游应用**: 直接用 DINOv2 初始化 encoder 并对齐其语义特征, 证明 DINOv2 是极强的视觉 foundation encoder
- **数据 scaling 的又一胜利**: 与 CLIP (400M), SAM (1B masks), DINOv2 (142M) 一样, Depth Anything 再次证明数据规模 >> 模型创新
- **NeRF/3DGS 的单目替代**: 不需要多视角, 从单张图直接估计深度, 与 NeRF/3DGS 形成 mono vs multi-view 3D 感知的互补

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Self-training + 强扰动是利用无标注数据的关键 -- 朴素伪标签训练不涨点 | 机器人也有大量无标注 demo 视频, 可以用同样的 "teacher 打标 + student 强增强" 策略 |
| 2 | 从预训练 FM (DINOv2) 继承语义先验比训练 auxiliary task (如语义分割) 更有效 | 机器人 visual encoder 应直接用 DINOv2 初始化, 而不是从头训 |
| 3 | 62M 廉价无标注图像 >> 1.5M 昂贵有标注图像, 数据覆盖度决定泛化能力 | 机器人数据采集应优先扩大场景多样性 (unlabeled), 而非精细标注少量 demo |
| 4 | Affine-invariant loss 解决了多数据集 scale/shift 不一致问题 | 机器人多源数据 (不同相机/场景/机器人) 融合时, invariant loss 是刚需设计 |
