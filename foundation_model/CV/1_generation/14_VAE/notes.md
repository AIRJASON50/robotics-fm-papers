# Auto-Encoding Variational Bayes (Kingma & Welling, 2014) -- Takeaway Notes

> 一句话: 用 reparameterization trick 让 variational inference 可以反向传播, 开创了 continuous latent space 生成模型, 是 LDM/Diffusion Policy/ACT 的数学根基。

## 核心贡献

1. **Reparameterization trick**: 将采样操作 z ~ q(z|x) 改写为 z = mu + sigma * epsilon (epsilon ~ N(0,1)), 让梯度可以穿过随机采样节点, 解决了 variational inference 无法端到端训练的核心问题。
2. **ELBO (Evidence Lower Bound) 目标函数**: L = E[log p(x|z)] - KL(q(z|x) || p(z)), 第一项是 reconstruction, 第二项是 latent regularization -- 两项的平衡决定了 latent space 的结构化程度。
3. **连续结构化的 latent space**: 不同于 GAN 的隐式 latent space, VAE 的 latent space 是显式的、连续的、有概率语义的 -- 可以插值、采样、操控。

## 为什么重要

- **生成模型的第二范式**: GAN 靠对抗, VAE 靠变分推断 -- 两条路线分别发展, 但 VAE 的 latent space 思想影响远超 GAN。
- **Latent Diffusion 的前提**: LDM (Stable Diffusion) = VAE encoder 压缩到 latent space + diffusion 在 latent space 做生成。没有 VAE, 就没有 "latent" diffusion。
- **连续 action space 的建模启发**: VAE 证明连续分布可以用 encoder-decoder + 正则化 latent space 来建模, 这直接启发了机器人动作生成中的 latent action space 设计。

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动关联 |
|---|----------|---------|
| 1 | **Reparameterization trick 是所有 "梯度穿过随机性" 的模板**。PPO 的 policy gradient 用 log-probability trick; VAE 用 reparameterization trick -- 两者本质都是解决 "如何对随机采样求导" 的不同方案。 | 理解 stochastic policy 的数学基础 |
| 2 | **Latent space 压缩是 scalable 生成的关键**: pixel space 太大 (256x256x3=196K 维), latent space 压缩到 32x32x4=4K 维 -- LDM 就是靠 VAE 降了 50x 维度才能用 diffusion 做高分辨率生成。Diffusion Policy 同理。 | 理解为什么 robot action generation 也走 latent space |
| 3 | **KL regularization = latent space 的结构化**: KL 项强制 latent 接近标准正态, 使得 latent space 平滑且可插值。ACT (Action Chunking with Transformers) 的 CVAE 结构直接用了这个设计。 | ACT policy 中 CVAE 的设计来源 |

## 与知识库其他内容的关联

- **LDM (CV/1_generation/22_LDM)**: LDM = VAE (pixel->latent) + Diffusion (latent space 生成), VAE 是 LDM 的前半部分
- **ACT (robotics/policy_learning)**: ACT 使用 CVAE 架构, encoder 将 observation+action 编码到 latent, decoder 从 latent 解码 action chunk
- **Diffusion Policy (robotics/policy_learning)**: 虽然不直接用 VAE, 但 "在 latent space 做生成" 的思路源于 VAE
- **GAN (foundations/14_GAN)**: VAE vs GAN 是生成模型两条路线; VAE 生成模糊但可控, GAN 生成清晰但不稳定; diffusion 最终统一了两者的优势
