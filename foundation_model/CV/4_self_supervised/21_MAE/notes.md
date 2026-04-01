# Masked Autoencoders Are Scalable Vision Learners -- 学习笔记
> 一句话: 随机 mask 75% image patches 再重建像素, 用非对称 encoder-decoder 把视觉自监督预训练变得高效且 scalable, 是 CV 的 "BERT moment"。
> 论文: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick (FAIR), CVPR 2022
> 引用量级: ~7000+

## 这篇论文解决了什么问题
NLP 中 masked autoencoding (BERT) 早已成功, 但在 CV 中落后。作者指出三个根本原因: (1) 架构不同 -- CNN 不好插 mask token, 直到 ViT 出现才解决; (2) 信息密度不同 -- 图像有大量空间冗余, mask 15% 的 patch 太简单, 从邻居就能插值恢复, 学不到高层语义; (3) decoder 角色不同 -- NLP 的 decoder 预测 word (高语义), 而 CV 的 decoder 重建 pixel (低语义), 所以 decoder 设计必须不同。之前的 BEiT 需要额外的 dVAE tokenizer, 对比学习 (MoCo/SimCLR) 在 ViT-L/H 上 scale 困难。MAE 要用最简单的方法解决所有这些问题。

## 核心想法 (用直觉解释)
想象你拿到一张拼图, 但 75% 的碎片被拿走了。如果只缺几块, 你看看周围的颜色就能猜出来 (低层插值)。但如果缺了 3/4, 你必须理解 "这是一只狗" "这是天空" 才能补全 (高层语义理解)。MAE 就是这个思路: 故意 mask 掉绝大部分, 逼模型理解整体结构。同时, encoder 只处理剩下 25% 的可见 patches (不放 mask token), 大幅省算力; 一个很浅的 decoder 负责从 latent + mask tokens 重建像素。预训练结束后 decoder 扔掉, 只保留 encoder 用于下游任务。

## 关键设计决策
- **75% random masking**: 最优 mask ratio 远高于 BERT 的 15%, 因为图像空间冗余极高。uniform random 采样防止 center bias, 也比 block/grid masking 效果更好
- **非对称 encoder-decoder**: encoder (ViT-L, 24 blocks) 只处理 visible patches, decoder (8 blocks, 512-d) 处理全部 tokens (visible + mask)。decoder 计算量 < encoder 的 10%, 且可以独立设计深度/宽度。这是效率关键: encoder FLOPs 降到 1/4, 训练速度 3-4x 提升
- **重建 normalized pixels**: 目标是每个 patch 归一化后的像素值 (per-patch mean/std normalization), 比原始像素效果更好。不需要 dVAE tokenizer (BEiT), 比 BEiT 更简单且更准
- **几乎不需要 data augmentation**: random crop 就够, 不需要 color jittering/flipping -- 因为 random masking 本身就是最强的 augmentation

## 这篇论文之后发生了什么
- **VideoMAE (2022)**: 扩展到视频, mask ratio 推到 90-95%, tube masking 处理时间冗余
- **DINOv2 (2023)**: 融合 MAE 的 masked prediction 和 DINO 的 self-distillation, 产出最强 frozen features
- **SAM (2023)**: 用 MAE pre-trained ViT-H 作为 image encoder, 构建分割 foundation model
- **MAE 的局限被暴露**: frozen features (linear probing) 弱于 contrastive methods; 需要 fine-tune 才能发挥全部实力, 不如 DINOv2 的 "即插即用"

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 视觉信号 75% 冗余, 实际信息量远小于 pixel 数量 | Robot camera observation 也可以激进压缩; VLA 中 visual token 数量有很大削减空间 |
| 2 | MIM (Masked Image Modeling) 比 contrastive learning 更 scalable | MAE 轻松训到 ViT-Huge, 而 MoCo/SimCLR 在大模型上不稳定; robot visual pre-training 选路线时考虑 scalability |
| 3 | MAE frozen feature 弱, 必须 fine-tune; DINOv2 frozen feature 强, 即插即用 | Robot 部署优先选 DINOv2; MAE 的价值在于预训练范式而非直接当 backbone |
| 4 | Mask-and-predict 范式从 NLP->image->video->robot, 是一条清晰的技术迁移链 | VideoMAE 用于 robot video pre-training 的方法论基础就来自 MAE |
