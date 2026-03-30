# DDPM: Denoising Diffusion Probabilistic Models -- 阅读笔记

Ho, Jain, Abbeel (UC Berkeley), NeurIPS 2020

---

## 1. Core Problem

DDPM 解决的核心问题：**如何用 diffusion model 生成高质量样本**。

Sohl-Dickstein et al. (2015) 最早提出了 diffusion probabilistic model 的框架，但当时生成质量远不如 GAN。DDPM 首次证明 diffusion model 可以达到甚至超越 GAN 的样本质量（CIFAR10 FID 3.17），同时保留了似然模型的优势（可计算 log-likelihood，训练稳定，无需 adversarial training）。

直觉上的理解：模型学习「如何一步步从纯噪声中恢复出干净数据」。每一步只需要做微小的 denoising，单步任务简单，但 1000 步累积起来可以从 N(0,I) 生成复杂图像。

---

## 2. Method

### 2.1 Forward Process (加噪)

给定干净数据 x_0，forward process 按照固定的 variance schedule {beta_t} 逐步加噪：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

关键性质：可以跳过中间步骤，直接从 x_0 采样任意时刻 x_t（reparameterization trick）：

```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

其中 alpha_t = 1 - beta_t, alpha_bar_t = prod(alpha_s, s=1..t)。

实现中 beta 从 1e-4 线性增长到 0.02，共 T=1000 步。注意 forward process 不是简单叠加噪声 -- 每步都乘以 sqrt(1-beta_t) 对信号进行衰减，确保方差不会发散。

### 2.2 Reverse Process (去噪)

反向过程用神经网络参数化：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

论文提出了两种参数化方式，最终选择 **epsilon-prediction**：
- 网络 epsilon_theta(x_t, t) 预测加入的噪声 epsilon
- 从噪声预测反推出 x_0，再计算 posterior mean

反推公式：x_0_pred = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)

方差 sigma_t^2 固定为 beta_t（fixedlarge）或 posterior variance beta_tilde_t（fixedsmall），不做学习。实验表明两种选择效果相近。

### 2.3 Loss Function

完整的 variational bound 包含 T 个 KL divergence 项。但论文的核心发现是 **simplified loss L_simple 效果更好**：

```
L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon, t) ||^2 ]
```

即：随机采一个 timestep t，加噪，让网络预测噪声，MSE loss。

这个简化做了两件事：(1) 去掉了每个 timestep 前面的权重系数 beta_t^2 / (2 * sigma_t^2 * alpha_t * (1-alpha_bar_t))；(2) 效果上 down-weight 了小 t（低噪声）的 loss，让网络更关注高噪声情况。实验证明这提升了样本质量，但略微损害了 log-likelihood。

### 2.4 与 Score Matching 的联系

论文揭示了一个重要等价关系：

- epsilon-prediction 参数化下的 sampling 过程等价于 annealed Langevin dynamics
- L_simple 等价于 denoising score matching over multiple noise levels（NCSN, Song & Ermon 2019）

因为 score function = -epsilon / sqrt(1-alpha_bar_t)，所以预测噪声 epsilon 本质上就是在估计数据分布的 score（对数密度的梯度）。这个联系是后来 score-based 和 diffusion 两条线合流的理论基础。

---

## 3. Key Designs

### 3.1 Epsilon-Prediction Parameterization

网络不直接预测 x_{t-1} 或 x_0，而是预测添加的噪声 epsilon。这看似只是变量替换，但有深层意义：

- 与 score matching 建立了直接联系
- Loss 的 reweighting 自然产生了类似 SNR weighting 的效果
- 实验中比预测 mu_tilde（posterior mean）效果更好（Table 2: FID 3.17 vs 13.22）

### 3.2 简化的训练目标 L_simple

丢弃 variational bound 中的权重项，改用 unweighted MSE。从信息论角度看，这放弃了对 log-likelihood 的优化，转而优化样本质量。论文对此的解释是：小 t 对应的去噪任务太简单（噪声极小），这些项主导了 VLB 但对样本质量贡献很小；L_simple 相当于让网络把能力集中在困难的大 t 去噪任务上。

### 3.3 渐进式生成的 Coarse-to-Fine 结构

实验揭示 diffusion 的 reverse process 自然实现了 coarse-to-fine 生成：早期步骤（t 大）决定全局结构（姿态、构图），后期步骤（t 小）补充细节（纹理、边缘）。这一性质对 Diffusion Policy 很重要 -- 动作序列的生成也遵循类似规律：先确定大致轨迹，再微调精细动作。

---

## 4. Experiments

### 主要结果

| Dataset | FID | IS | NLL (bits/dim) |
|---------|-----|----|----------------|
| CIFAR10 (L_simple) | **3.17** | 9.46 | <= 3.75 |
| CIFAR10 (VLB) | 13.51 | 7.67 | **<= 3.70** |
| LSUN Bedroom 256 | 4.90 | - | - |
| LSUN Church 256 | 7.89 | - | - |
| CelebA-HQ 256 | 未报 FID | - | - |

- FID 3.17 在当时超越了所有 unconditional 模型（包括 BigGAN 14.73, StyleGAN2+ADA 2.67 是 conditional 的）
- Log-likelihood 不如 autoregressive models，但样本质量远超 likelihood-based 方法
- 率失真分析：大部分 lossless codelength 用于编码人类不可感知的细节

### Ablation 要点（Table 2）

- epsilon-prediction + L_simple: FID **3.17** (best)
- epsilon-prediction + VLB (L): FID 13.51
- mu-prediction + VLB (L): FID 13.22 (仅在用 VLB 时可训)
- 学习方差 Sigma: 训练不稳定

### 训练细节

- U-Net backbone (类 PixelCNN++), 35.7M params (CIFAR10), 114M (256x256)
- Group normalization, Swish activation
- Timestep t 通过 Transformer sinusoidal positional embedding 注入每个 residual block
- Self-attention 仅在 16x16 分辨率
- Adam, lr=2e-4, EMA decay 0.9999
- CIFAR10: batch 128, ~800K steps (~10.6 hrs on TPU v3-8)

---

## 5. 对 Robotics 的意义

### DDPM -> Diffusion Policy

Diffusion Policy (Chi et al., 2024) 将 DDPM 的 denoising 框架从图像生成迁移到动作生成：

| DDPM (图像) | Diffusion Policy (动作) |
|-------------|------------------------|
| x_0 = clean image | x_0 = action sequence (a_{t:t+H}) |
| 加噪 / 去噪在像素空间 | 加噪 / 去噪在动作空间 |
| 无条件生成 | 以观测 o_t 为条件 |
| U-Net on 2D image | 1D temporal U-Net 或 Transformer |

核心洞察：DDPM 的 iterative denoising 本质上是一个 **iterative refinement** 过程，与数据的具体含义无关。把 "pixel" 换成 "action"，整个数学框架完全成立。DDPM 处理多模态分布的能力（一张噪声图可以去噪成不同的人脸）正好解决了行为克隆中的 multi-modality 问题（同一观测下可能有多种合理动作）。

### DDPM -> Flow Matching -> pi_0

Flow Matching (Lipman et al., 2023) 可以理解为 DDPM 的简化变体：

| | DDPM | Flow Matching |
|--|------|---------------|
| Forward 过程 | 固定 Markov chain, 逐步加噪 | 直线插值 x_t = (1-t)*x_0 + t*epsilon |
| 网络预测 | 噪声 epsilon | velocity field v_theta(x_t, t) |
| 采样 | 1000 步 discrete reverse chain | ODE 积分 (可用少量步数) |
| Loss | MSE on epsilon | MSE on velocity |
| 理论基础 | ELBO / score matching | Continuous normalizing flow |

Flow Matching 的优势：(1) 训练更简单（无需 noise schedule 的精心设计）; (2) 采样可用 adaptive ODE solver，通常 10-50 步就够; (3) 理论更清晰。pi_0 (Physical Intelligence, 2024) 正是基于 Flow Matching 来生成动作。

从 DDPM 到 Flow Matching 的核心思路演变：从「离散 Markov chain + ELBO」简化为「连续 ODE + regression loss」，但本质都是学习一个从噪声到数据的确定性/随机映射。

---

## 6. Paper vs Code 差异

基于对 `/home/l/ws/doc/paper/foundation_model/methods/20_DDPM/diffusion/` 的代码审查：

### 6.1 代码中存在两套 diffusion 实现

- `diffusion_utils.py`: `GaussianDiffusion` -- 早期版本，仅支持 noise prediction (`loss_type='noisepred'`)，方差固定
- `diffusion_utils_2.py`: `GaussianDiffusion2` -- 完整版本，支持三种 mean type (`xprev/xstart/eps`) 和三种 var type (`learned/fixedsmall/fixedlarge`)

实际训练脚本 (`run_cifar.py`) 使用的是 `GaussianDiffusion2`，默认配置为 `model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse'`。

### 6.2 论文未强调的实现细节

- **Gradient clipping**: 代码中使用 `grad_clip=1.0`，论文未提及
- **Learning rate warmup**: 前 5000 步线性 warmup，论文未提及
- **Dropout**: CIFAR10 使用 dropout=0.1，其他数据集为 0（论文在 Appendix B 简要提及）
- **Random flip**: CIFAR10 训练时使用随机水平翻转
- **Variance clipping**: posterior log variance 在 t=0 处 clip 为 posterior_variance[1] 而非 0，防止 log(0)
- **初始化**: 最后一层 conv 和 residual block 最后一层用 `init_scale=0.`（近零初始化），这意味着 residual block 初始时接近 identity，attention block 初始时输出接近 0。这是训练稳定性的关键 trick

### 6.3 Architecture 细节

代码中 U-Net 的具体配置（论文仅在 Appendix B 简述）：
- CIFAR10: `ch=128, ch_mult=(1,2,2,2), num_res_blocks=2, attn_resolutions=(16,)` -> 35.7M params
- Activation: Swish (非 ReLU)
- Normalization: Group Norm (非 Batch Norm)
- Timestep embedding: sinusoidal -> 2-layer MLP -> additive injection into each ResBlock
- Self-attention 实现为 spatial attention (QKV via 1x1 conv, einsum-based dot product)

### 6.4 代码是 TensorFlow 1.x

原始实现基于 TF1 + TPU。后续社区的 PyTorch 复现 (如 lucidrains/denoising-diffusion-pytorch) 更广泛使用，也是 Diffusion Policy 等下游工作的参考基础。
