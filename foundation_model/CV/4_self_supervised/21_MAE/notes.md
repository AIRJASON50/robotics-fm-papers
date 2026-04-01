# Masked Autoencoders Are Scalable Vision Learners (He et al., 2021) -- Takeaway Notes

> 一句话: 随机 mask 75% 的 image patches 然后重建像素, 用最简单的 pretext task 实现了 scalable 的视觉自监督预训练, 是 CV 领域的 "BERT moment"。

## 核心贡献

1. **极高 mask ratio (75%)**: 图像信息冗余远高于语言 (相邻 patch 高度相关), 所以 mask ratio 要远高于 BERT 的 15%。75% mask 迫使模型理解全局语义而非插值邻居。
2. **非对称 encoder-decoder**: encoder 只处理 visible patches (25%), decoder 轻量且只用于预训练 -- 这让 encoder 计算量降到全量的 25%, 大幅提升训练效率。
3. **直接预测像素**: 不需要 tokenizer (BEiT 需要 dVAE), 不需要 momentum encoder (DINO 需要 EMA teacher) -- 最简单的目标函数, 最好 scale。

## 为什么重要

- **自监督视觉预训练的 scaling 突破**: 之前的对比学习 (MoCo, SimCLR) 在 ViT-L/H 上 scale 困难; MAE 可以轻松训 ViT-Huge, 证明 MIM (Masked Image Modeling) 是更 scalable 的自监督路线。
- **效率**: encoder 只看 25% token, 训练速度 3x+ 提升, 使大规模视觉预训练变得经济可行。
- **揭示视觉信息的冗余性**: 75% 的 patch 被 mask 仍能重建, 说明图像中大部分信息是冗余的 -- 这对理解 robot visual observation 的信息瓶颈有重要启示。

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动关联 |
|---|----------|---------|
| 1 | **视觉 75% 冗余**: 如果 robot camera 的图像 75% 可以从剩余 25% 重建, 那 visual observation 的实际信息量远小于 pixel 数。这暗示 robot visual encoder 可以更激进地压缩。 | 理解 VLA 中 visual token 数量的设计空间 |
| 2 | **Fine-tune >> frozen feature (MAE 的局限)**: MAE 的 frozen feature 在 linear probe 上弱于 DINO/DINOv2, 必须 fine-tune 才强。这说明 MAE 学到的是 "可适配的通用表征" 而非 "即插即用的特征"。对 robot 来说, DINOv2 frozen feature 更实用。 | MAE vs DINOv2 选型: robot 场景优先 DINOv2 |
| 3 | **MIM -> VideoMAE -> robot 视频预训练**: MAE 的 mask-and-predict 范式从图像扩展到视频 (VideoMAE, mask 90%), 再扩展到 robot 视频 -- 这是一条清晰的技术迁移路径。 | 理解 robot video pre-training 的方法来源 |

## 与知识库其他内容的关联

- **DINO/DINOv2 (CV/4_self_supervised)**: MAE 和 DINO 是自监督两条路线 -- MAE 是 MIM (重建), DINO 是 self-distillation (判别); DINOv2 融合了两者
- **BEiT (CV/4_self_supervised)**: BEiT 也做 MIM 但需要 dVAE tokenizer; MAE 证明直接预测像素就够, 更简单也更 scalable
- **VideoMAE (CV/6_video)**: MAE 的视频版本, mask ratio 提高到 90-95%, 是 robot 视频表征预训练的基础
- **Transformer (foundations/17_Transformer)**: MAE 的成功依赖 ViT 的 patch tokenization -- 有了 patch, 才能 mask patch
