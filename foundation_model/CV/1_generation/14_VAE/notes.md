# Auto-Encoding Variational Bayes -- 学习笔记

> 一句话: 通过 reparameterization trick 让变分推断可以用标准梯度下降训练, 奠定了深度生成模型的概率基础。
> 论文: Diederik P. Kingma, Max Welling (Universiteit van Amsterdam), 2013, ICLR 2014

## 这篇论文解决了什么问题

对于带有连续 latent variable 的概率生成模型 (z -> x), 我们想做三件事:
1. 学模型参数 theta -- 生成逼真的数据
2. 推断后验 p(z|x) -- 给定观察, 找到 latent code
3. 估计边际似然 p(x) -- 评估模型好坏

传统方法的困难: 当 likelihood p(x|z) 是神经网络时, 边际似然 p(x) = integral of p(x|z)p(z)dz intractable (不可解析计算), EM 算法和 mean-field variational inference 都失效。MCMC 采样太慢, 不适合大数据集。

核心矛盾: **需要一种既能处理 intractable posterior, 又能 scale 到大数据的推断方法。**

## 核心想法 (用直觉解释)

**用一个神经网络 (encoder) 直接学习近似后验 q_phi(z|x), 和 decoder p_theta(x|z) 一起端到端训练。**

关键障碍: 从 q_phi(z|x) 采样的操作不可微分 -- 不能对"从分布中采样"求梯度。

**Reparameterization trick** 解决了这个问题:
```
epsilon ~ N(0, I)          # sample noise from fixed distribution
z = mu + sigma * epsilon   # deterministic transformation of encoder output
```
z 关于 phi 变成可微的 (mu 和 sigma 是 encoder 的输出), 梯度可以流过去。

训练目标 (ELBO, Evidence Lower Bound):
```
L = -D_KL(q_phi(z|x) || p(z)) + E_q[log p_theta(x|z)]
     ^-- regularizer              ^-- reconstruction
     后验别离先验太远               能还原出输入 x
```

## 关键设计决策

**1. Reparameterization trick -- 论文的核心贡献**

把"对随机变量期望的梯度"转化为"对确定性函数期望的梯度"。适用于任何可以表示为 z = g(epsilon, x) 的分布 (Gaussian, Logistic, Laplace 等 location-scale 族)。打通了概率推断和深度学习的桥梁。

**2. KL divergence 的解析计算**

当 prior p(z) = N(0, I) 且 q(z|x) = N(mu, sigma^2 I) 时, KL 有闭合解:
```
D_KL = -1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```
只有 reconstruction term 需要采样估计。论文发现 L=1 (一个采样) 在 minibatch M=100 时就够了。

**3. ELBO 的双重角色: 目标函数 + 正则化器**

KL 项鼓励后验接近标准正态先验, 起到正则化作用。实验发现增加 latent 维度不会 overfit -- 多余维度的 KL 自动被压到零。这是比传统 autoencoder 更优雅的正则化方式。

**4. 与 autoencoder 的联系**

VAE 的目标 = autoencoder 重建误差 + KL 正则项。但这不是 ad hoc 的设计, 而是从变分推断的 ELBO 自然推导出来的。给了 "encoder-decoder + bottleneck" 结构一个概率理论基础。

**5. 实验结果**

在 MNIST 和 Frey Face 上对比 wake-sleep algorithm 和 Monte Carlo EM。AEVB 收敛更快, 在各种 latent 维度 (3-200) 上都更好。更多 latent 变量不导致 overfitting, 归功于 variational bound 的正则化效果。

## 这篇论文之后发生了什么

- **VAE 成为生成模型三大范式之一** (与 GAN, Flow 并列)
- **VQ-VAE (2017)**: discrete latent code, 后来演化为 DALL-E 的核心组件
- **Latent Diffusion (Stable Diffusion)**: VAE 把图像压到 latent space, diffusion 在 latent space 做生成 -- VAE 是 LDM 的前级
- **beta-VAE**: 调节 KL 权重控制 disentanglement, 学习可解释 representation
- **Reparameterization trick 广泛使用**: Gumbel-Softmax (discrete), normalizing flow, diffusion forward process 都源于此

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|----------|
| 1 | **Reparameterization trick 是让"采样"可微的通用技术** | SAC 用 reparameterization 训练连续 policy, PPO 用 log-prob trick -- 两条路解决同一个问题; Diffusion Policy 也依赖可微采样 |
| 2 | **Encoder-decoder + bottleneck + regularization 是学习 latent representation 的基本范式** | ACT 的 CVAE, latent plan 的 VQ-VAE, robot policy 的 latent action space 都继承这个结构 |
| 3 | **KL regularization 防止 latent space 退化** -- 没有它, autoencoder 的 latent space 无结构, 不能采样 | 训练 robot policy 的 latent space 时需要类似正则化保证 latent code 可泛化 |
| 4 | **VAE 是 Stable Diffusion / LDM 的前级压缩器** -- 没有 VAE 把 512x512 压到 64x64, diffusion 算不起 | 理解 image generation pipeline 对理解 world model (UniSim, DreamGen) 至关重要 |
| 5 | **概率生成模型自然表达不确定性** -- latent space 是分布而非点 | Robot 的多模态动作分布 (同一 observation 有多种合理动作) 天然需要概率建模 |
