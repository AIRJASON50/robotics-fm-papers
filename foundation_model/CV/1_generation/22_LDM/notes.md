# High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022) -- Takeaway Notes

> 一句话: 把 diffusion process 从 pixel space 搬到 VAE 的 latent space, 计算量降 10-50x 且不损失质量, 催生 Stable Diffusion 并直接启发了 Diffusion Policy。

## 核心贡献

1. **两阶段架构**: Stage 1 -- 训练 VAE (或 VQ-VAE) 把图像压缩到低维 latent space; Stage 2 -- 在 latent space 做 diffusion 的前向加噪 + 反向去噪。pixel space 的 256x256x3 压到 latent space 的 32x32x4, 计算量骤降。
2. **Cross-attention conditioning**: 用 cross-attention 将条件信息 (text, layout, semantic map) 注入 U-Net 的中间层, 实现灵活的条件生成, 不需要为每种条件重新设计架构。
3. **Perceptual compression vs semantic compression 的分离**: VAE 负责 perceptual compression (去掉高频细节), diffusion 负责 semantic compression (学概念级生成) -- 两者各司其职。

## 为什么重要

- **Diffusion 从学术走向产品**: DDPM (2020) 证明 diffusion 能生成高质量图像, 但在 pixel space 计算太贵。LDM 降了维度, 直接催生 Stable Diffusion -- 开源社区爆发。
- **"先压缩再生成" 成为默认范式**: 之后的 DiT、Sora、甚至 pi_0 都遵循这个思路 -- 先把数据压到 latent, 再用生成模型处理 latent。
- **Cross-attention conditioning 成为通用接口**: text-to-image 用 CLIP text embedding 做条件; robot 领域 Diffusion Policy 用 observation embedding 做条件 -- 同一个 cross-attention 机制。

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动关联 |
|---|----------|---------|
| 1 | **Latent space diffusion = scalable generation**: pixel space 做 diffusion 太贵, latent space 做 diffusion 才 practical。类比: 直接在 joint space (高维) 做 action diffusion 可能也太贵, 先学一个 action latent space 再做 diffusion 可能更好。 | 理解 Diffusion Policy 为什么能 work |
| 2 | **Cross-attention = 通用条件注入**: LDM 用 cross-attention 注入 text; Diffusion Policy 用同样的机制注入 visual observation。这是 "条件生成" 的标准做法。 | Diffusion Policy 架构中 observation conditioning 的来源 |
| 3 | **Perceptual vs semantic 分离**: VAE 去噪高频, diffusion 学语义。在 robot 场景, visual encoder 做 perceptual (从像素到特征), policy head 做 semantic (从特征到动作) -- 同样的分层思想。 | VLA 两阶段架构的设计直觉 |

## 与知识库其他内容的关联

- **VAE (CV/1_generation/14_VAE)**: LDM 的 Stage 1 就是 VAE; 没有 VAE 的 latent space, LDM 不存在
- **DDPM (CV/1_generation)**: LDM 的 Stage 2 用 DDPM 的 forward/reverse process, 只是改在 latent space 做
- **DiT (CV/1_generation)**: DiT 把 LDM 的 U-Net backbone 换成 Transformer -- GR00T N1 的 action head 就是 DiT
- **Diffusion Policy (robotics/policy_learning)**: 直接继承 LDM 的 "conditional denoising in latent space" 思路, 只是生成目标从图像变成 action trajectory
- **Flow Matching (CV/1_generation)**: pi_0 用 flow matching 替代 diffusion, 但 "latent space + conditional generation" 的框架不变
