# BEiT: BERT Pre-Training of Image Transformers -- 学习笔记 (选读)
> 一句话: 把 BERT 的 masked language modeling 迁移到视觉: mask 掉 image patch, 预测离散 visual token (由 dVAE 生成), 开创 Masked Image Modeling (MIM) 范式.
> 论文: Hangbo Bao, Li Dong, Songhao Piao, Furu Wei (HIT + Microsoft Research), ICLR 2022

## 核心想法
BEiT 将图像看成两种视图: (1) image patches (连续像素, 作为输入) 和 (2) visual tokens (离散编码, 作为预测目标). 预训练时随机 mask 约 40% 的 patches (blockwise masking), 让 ViT encoder 从可见 patches 预测被 mask 位置的 visual token (由预训练的 dVAE tokenizer 提供, 词表大小 8192). 关键洞察: 直接回归原始像素会让模型聚焦于低级纹理和高频细节, 而预测离散 visual token 迫使模型学习高层语义抽象. Ablation 证明: 去掉 visual token (回归像素) 导致 ImageNet -2.2%, ADE20K -7.3%.

## 与主线论文的关系
- **MAE 的直接竞争者**: BEiT 用 tokenizer 生成离散目标, MAE 直接回归像素 -- 两者共同定义了 MIM 范式的两条路线
- **DINO/DINOv2 的互补**: BEiT 走 generative (重建) 路线, DINO 走 discriminative (对比/蒸馏) 路线, DINOv2 融合了两者
- **从 BERT 到视觉**: 证明 NLP 的 "mask-then-predict" 范式可以迁移到视觉, 是 CV 模仿 NLP 的重要一步

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 预测目标的抽象层级决定了学到的表征质量 -- 离散 token > 原始像素 | 机器人 world model 设计时, 预测抽象 latent (如 JEPA) 优于预测原始观测 (如逐像素重建) |
| 2 | Blockwise masking 比随机 masking 更好, 因为它迫使模型建模更远距离的依赖 | 机器人 visual encoder 的 masking 策略应考虑空间结构, 不能随机撒点 |
| 3 | Self-attention 在无监督预训练后自动学会了分割语义区域和物体边界 | 无标注预训练就能获得 semantic-aware 表征, 对机器人物体理解很有价值 |
| 4 | BEiT + intermediate fine-tuning 的两阶段范式 (SSL pretrain -> supervised finetune -> task finetune) 效果最好 | 机器人 FM 也可以走三阶段: SSL 预训练 -> 大规模 manipulation 微调 -> 特定任务微调 |
