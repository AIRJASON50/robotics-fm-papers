# High-Resolution Image Synthesis with Latent Diffusion Models -- 学习笔记
> 一句话: 把 diffusion process 从 pixel space 搬到 VAE 的 latent space, 计算量骤降且质量不损, 并用 cross-attention 注入任意条件 -- 催生 Stable Diffusion, 直接启发 Diffusion Policy。
> 论文: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer (LMU Munich, Runway ML), CVPR 2022
> 引用量级: ~15000+

## 这篇论文解决了什么问题
Diffusion model (DDPM) 已经证明能生成高质量图像, 但在 pixel space 直接做 diffusion 计算量极大 -- 训练最强的 pixel-space DM 需要 150-1000 V100 GPU days, 生成 50k 样本需要单卡 5 天。这把 DM 限制在了少数有大量计算资源的实验室中。核心矛盾: 图像 pixel space 中大部分 bits 对应人眼不可感知的高频细节 (perceptual compression), 但 DM 必须在所有 pixel 上计算 loss/gradient, 浪费了大量算力在 "不重要" 的信息上。

## 核心想法 (用直觉解释)
分两步走: 第一步, 训一个 autoencoder 把 256x256x3 的图像压缩到 32x32x4 的 latent (去掉高频细节, 保留语义信息); 第二步, 在这个小得多的 latent space 里做 diffusion (加噪 -> 去噪)。这就像画家先用铅笔画草稿 (latent, 语义构图), 再细化上色 (decoder, 像素恢复)。因为 latent 比 pixel 小 64 倍, diffusion 的计算量骤降。要生成特定内容 (text/layout/class), 用 cross-attention 把条件信息注入 U-Net 的每一层 -- 这是一个通用的条件注入接口。

## 关键设计决策
- **两阶段解耦**: Stage 1 (perceptual compression) -- 训练 autoencoder (encoder E + decoder D), 用 perceptual loss + adversarial loss 确保 latent space 是 perceptually equivalent 的。用 KL-reg 或 VQ-reg 避免 latent variance 过大。Stage 2 (semantic compression) -- 在固定的 latent space 上训练 diffusion model, 目标是 L_LDM = E[||epsilon - epsilon_theta(z_t, t)||^2]。两阶段各训一次, autoencoder 可复用于多个 DM
- **Downsampling factor f 的选择**: f=1 是 pixel-space DM, f=32 压缩太狠丢信息。实验表明 f=4 和 f=8 最优 -- 在效率和质量间取得平衡。f=4 用于复杂数据集 (ImageNet), f=8 用于简单数据集
- **Cross-attention conditioning**: 条件 y (text/semantic map/image) 经过 domain-specific encoder tau_theta 映射到中间表示, 再通过 cross-attention 注入 U-Net: Q = W_Q * phi(z_t), K = W_K * tau(y), V = W_V * tau(y)。这让同一个 DM 架构适配任意条件类型
- **Classifier-free guidance**: 推理时混合有条件和无条件生成, 大幅提升 text-to-image 质量 (FID 从 12.6 降到 3.60 on ImageNet)

## 这篇论文之后发生了什么
- **Stable Diffusion**: LDM 的开源实现, 在 LAION-5B 上训练, 引爆 AI 生成图像的产业化
- **DiT (2023)**: 把 LDM 的 U-Net 换成 Transformer, 进一步提升 scalability -- GR00T N1 的 action head 基于 DiT
- **Diffusion Policy (2023)**: 继承 "latent space + conditional denoising" 的核心思路, 把生成目标从图像换成 action trajectory
- **Flow Matching (2023+)**: pi0 等用 flow matching 替代 diffusion noise schedule, 但 "latent conditional generation" 的框架不变

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | "先压缩再生成" 是 scalable generation 的核心范式 | Diffusion Policy 也可以先学 action latent space 再做 diffusion; 直接在高维 joint space 做 diffusion 可能效率低 |
| 2 | Cross-attention = 通用条件注入接口: LDM 注入 text, Diffusion Policy 注入 observation | 理解 Diffusion Policy 中 observation conditioning 的架构来源 |
| 3 | Perceptual vs semantic 分离: VAE 处理感知压缩, DM 处理语义生成 | VLA 两阶段架构的设计直觉: visual encoder 做感知 (像素->特征), policy head 做语义 (特征->动作) |
| 4 | Autoencoder 训一次可复用: 同一个 latent space 可支撑多种生成任务 | Robot visual encoder (DINOv2) 训一次, 可同时支撑 manipulation/navigation/grasping 等不同 policy |
