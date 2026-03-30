# DiT: Scalable Diffusion Models with Transformers -- 阅读笔记

**Paper**: William Peebles (UC Berkeley), Saining Xie (NYU), ICCV 2023 (arXiv 2212.09748)
**Code**: `/home/l/ws/doc/paper/foundation_model/methods/23_DiT/DiT/`

---

## 1. Core Problem

DiT 解决的核心问题: **diffusion model 的 backbone 能否从 U-Net 替换为 transformer, 并继承 transformer 在 NLP/CV 中展现的优秀 scaling 特性?**

背景:
- 自 DDPM (Ho et al., 2020) 以来, 所有主流 diffusion model (ADM, GLIDE, DALL-E 2, Stable Diffusion) 的 backbone 均为卷积 U-Net, 其架构沿袭自 PixelCNN++, 仅做了少量修改 (加入 self-attention block, 使用 adaptive normalization)
- Transformer 已在 NLP 和视觉识别 (ViT) 中展现了远超 CNN 的 scaling behavior, 但在图像生成领域一直是 U-Net 的天下
- 关键问题: U-Net 的归纳偏置 (多尺度特征、skip connection) 是否对 diffusion model 至关重要? 还是可以用更标准的 transformer 替代?

DiT 的回答: **U-Net 的归纳偏置对 diffusion model 并非关键**。用标准 ViT 架构操作 latent patches, 通过增加 Gflops (增大模型或减小 patch size) 就能持续提升生成质量, 最终超越所有 U-Net-based diffusion model。

---

## 2. Method Overview

### 2.1 整体 Pipeline

DiT 在 Latent Diffusion Model (LDM) 框架下工作:

```
Training:
Image (256x256x3) --> [Frozen VAE Encoder] --> Latent (32x32x4)
                                                    |
                                              Add noise (t)
                                                    |
Noised Latent (32x32x4) --> [Patchify] --> Tokens (T, d) --> [N x DiT Block] --> [FinalLayer] --> [Unpatchify] --> Noise + Sigma prediction

Sampling:
Pure Noise (32x32x4) --> [250-step DDPM reverse] --> Clean Latent --> [Frozen VAE Decoder] --> Image (256x256x3)
```

### 2.2 DiT 架构详解

**Patchify**: 将 latent representation (32x32x4) 切分为 patches, 线性投影为 token 序列。Token 数量 T = (I/p)^2, 其中 p 为 patch size。

- p=2: T=256 tokens (最大 Gflops, 最好性能)
- p=4: T=64 tokens
- p=8: T=16 tokens

加上 fixed 2D sinusoidal positional embedding (来自 MAE)。

**DiT Block (adaLN-Zero)**: 核心 building block, 包含:

1. LayerNorm (不带可学习 affine 参数, `elementwise_affine=False`)
2. Multi-Head Self-Attention (来自 timm 的标准实现)
3. Pointwise Feedforward (GELU approximate="tanh", MLP ratio=4)
4. adaLN-Zero conditioning: 从条件向量 c 回归 6 个参数 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

数学表达:

```
c = t_embed(t) + y_embed(y)                      # timestep + class label embedding 相加
(shift1, scale1, gate1, shift2, scale2, gate2) = MLP(SiLU(c))   # 回归 6 个调制参数

x = x + gate1 * Attn(modulate(LN(x), shift1, scale1))   # attention 分支
x = x + gate2 * MLP(modulate(LN(x), shift2, scale2))    # FFN 分支

where modulate(x, shift, scale) = x * (1 + scale) + shift
```

**FinalLayer**: 最后一层也使用 adaLN (2 参数: shift, scale), 然后线性投影到 p*p*2C 维 (noise prediction + learned sigma)。

**Transformer Decoder**: unpatchify 操作将 token 序列重排回空间布局。

### 2.3 关键超参数

| 维度 | 选项 |
|------|------|
| Patch size p | 2, 4, 8 |
| Model size | S (12L/384d/6h), B (12L/768d/12h), L (24L/1024d/16h), XL (28L/1152d/16h) |
| Conditioning | In-context, Cross-attention, adaLN, **adaLN-Zero** (best) |

### 2.4 训练配置

- Optimizer: AdamW, lr=1e-4, no weight decay, no warmup
- Batch size: 256 (global)
- Data augmentation: 仅 horizontal flip
- EMA decay: 0.9999
- VAE: Stable Diffusion 的 pretrained VAE (downsample 8x), latent 空间维度 32x32x4
- Diffusion: T=1000, linear beta schedule (1e-4 to 2e-2)
- 训练 7M steps for SOTA 结果

---

## 3. Key Designs

### 3.1 adaLN-Zero: 条件注入机制

这是 DiT 最重要的设计贡献。论文探索了 4 种将 timestep t 和 class label y 注入 transformer block 的方式:

| 方式 | 机制 | Gflops (XL/2) | 400K步 FID |
|------|------|---------------|-----------|
| In-context | t, y 作为额外 token 拼入序列 | 119.4 | ~35.2 |
| Cross-attention | 额外 cross-attention 层 | 137.6 | ~26.1 |
| adaLN | 替换 LN 的 affine 参数, 回归 gamma, beta | 118.6 | ~25.2 |
| **adaLN-Zero** | adaLN + zero-initialized scale gate alpha | **118.6** | **~19.5** |

adaLN-Zero 的关键 insight 来自 ResNet 训练的经验:

1. **Zero initialization**: adaLN 的 MLP 输出层权重和 bias 全部初始化为 0, 使得初始时 gate alpha = 0, shift = 0, scale = 0
2. **效果**: 每个 DiT block 在训练初期等价于 identity function (因为 residual connection + gate=0), 整个网络初始化为 identity mapping
3. **优势**: 训练更稳定, 性能显著优于 vanilla adaLN (FID ~19.5 vs ~25.2 at 400K), 与训练 U-Net 中 zero-initialize 最后一层 conv 的做法一脉相承

对机器人学的意义: GR00T N1 的 DiT action head 直接采用了 adaLN-Zero 机制, 用 timestep + VLM embedding 作为条件信号。这种 "conditioning via normalization" 的范式比 cross-attention 更计算高效, 特别适合需要高频推理的实时控制场景。

### 3.2 Transformer 替代 U-Net 的 Scaling 优势

DiT 通过系统实验证明了 transformer backbone 对 diffusion model 的 scaling 优势:

**Gflops 与 FID 强相关 (correlation -0.93)**: 不同 DiT config 如果 Gflops 相近 (如 DiT-S/2 和 DiT-B/4), 其 FID 也相近。这意味着 **计算量 (而非参数量) 是决定性能的关键因素**。

**两种独立的 scaling 维度**:
1. 增大模型 (S -> B -> L -> XL): 更深更宽的 transformer
2. 减小 patch size (8 -> 4 -> 2): 增加 token 数量, 增加自注意力的计算量

两种方式都能持续降低 FID, 且效果可叠加。

**为什么比 U-Net 更适合 scaling**:

| U-Net | DiT (Transformer) |
|-------|-------------------|
| 多尺度特征图 + skip connection, 架构复杂 | 单尺度 token 序列, 架构统一 |
| 每层 channel 数不同, 难以系统性 scale | depth/width 独立 scale, 遵循 ViT 规律 |
| Self-attention 仅在低分辨率 (16x16) | 全局 self-attention 覆盖所有 token |
| 难以直接利用 NLP 领域的训练优化技术 | 直接复用 ViT/LLM 的训练 recipe |
| 参数量 != 实际计算量 (conv 层参数效率高但 Gflops 大) | Gflops 与参数量关系更清晰 |

对机器人学的意义: 这一发现为后续 diffusion-based 动作生成 (Diffusion Policy, pi_0, GR00T N1) 从 U-Net 迁移到 transformer 提供了理论和实验依据。Transformer backbone 让 diffusion model 可以自然融入 VLM 架构 (共享 attention 机制), 实现 vision-language-action 的统一 pipeline。

### 3.3 Latent Space Diffusion 与计算效率

DiT 在 VAE 的 latent space (32x32x4) 而非 pixel space (256x256x3) 进行 diffusion, 这带来了巨大的计算优势:

| 方法 | 空间 | Gflops |
|------|------|--------|
| ADM (pixel space) | 256x256 | 1120 |
| ADM-U (pixel + upsampler) | 256x256 | 742 |
| LDM-4 (latent, U-Net) | 32x32 | 103.6 |
| **DiT-XL/2 (latent, transformer)** | **32x32** | **118.6** |

DiT-XL/2 与 LDM-4 Gflops 相近, 但 FID 显著更低 (2.27 vs 3.60)。相比 pixel space 的 ADM, DiT 计算量降低约 10 倍。

---

## 4. Experiments

### 4.1 主要结果 -- 256x256 ImageNet

| Model | FID | sFID | IS | Precision | Recall |
|-------|-----|------|----|-----------|--------|
| BigGAN-deep | 6.95 | 7.36 | 171.4 | 0.87 | 0.28 |
| StyleGAN-XL | 2.30 | 4.02 | 265.12 | 0.78 | 0.53 |
| ADM | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| ADM-G, ADM-U | 3.94 | 6.14 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | - | 247.67 | 0.87 | 0.48 |
| **DiT-XL/2-G (cfg=1.50)** | **2.27** | **4.60** | **278.24** | **0.83** | **0.57** |

DiT-XL/2 在所有 diffusion model 中达到最低 FID (2.27), 同时 Recall (0.57) 显著高于 LDM (0.48), 表明生成多样性更好。

### 4.2 512x512 ImageNet

| Model | FID | IS |
|-------|-----|----|
| ADM | 23.24 | 58.06 |
| ADM-G, ADM-U | 3.85 | 221.72 |
| **DiT-XL/2-G (cfg=1.50)** | **3.04** | **240.82** |

512x512 时 DiT 处理 1024 tokens (64x64 latent, p=2), Gflops 为 524.6, 仍远低于 ADM 的 1983 Gflops。

### 4.3 Ablation 发现

**Conditioning 机制**: adaLN-Zero >> adaLN > cross-attention > in-context。adaLN-Zero 在 400K 步时 FID 几乎是 in-context 的一半。

**Scaling behavior**:
- 固定 patch size, 增大模型: FID 持续下降
- 固定模型, 减小 patch size: FID 持续下降
- 参数量不能唯一决定性能 -- 相同参数量下减小 patch 能显著提升 (因为增加了 Gflops)
- 大模型更 compute-efficient: XL/2 训练 10^10 Gflops 后优于 XL/4 训练相同计算量

**Sampling compute 不能替代 model compute**: 小模型用 1000 步采样也不如大模型用 128 步。模型能力是根本。

### 4.4 训练稳定性

论文特别指出: DiT 训练极其稳定, 所有 config 都没有观察到 loss spike, 不需要 learning rate warmup 或 regularization。这与 ViT 训练常见的不稳定形成对比, 可能得益于 adaLN-Zero 的 identity initialization。

---

## 5. Related Work Analysis

### 5.1 发展脉络

```
DDPM (2020) -- U-Net backbone, 首次证明 diffusion 可达 GAN 质量
  |
  v
IDDPM (2021) -- 学习方差, cosine schedule
  |
  v
ADM/Guided Diffusion (2021) -- classifier guidance, 超越 GAN
  |
  v
LDM/Stable Diffusion (2022) -- latent space diffusion, 大幅降低计算量
  |
  v
DiT (2023) -- [本文] 用 transformer 替代 U-Net, 证明 scaling law
  |
  v
应用扩展:
  |-- Diffusion Policy (2024) -- 将 diffusion 用于机器人动作生成
  |-- pi_0 (2024) -- flow matching + transformer 做 VLA
  |-- GR00T N1 (2025) -- DiT 作为 System 1 action head, 120Hz 控制
```

### 5.2 DiT 的独特性

1. **首次系统性证明 transformer 在 diffusion 中的 scaling law**: 之前虽有 concurrent work (如 GenViT, Hourglass Diffusion Transformer), 但 DiT 提供了最全面的 Gflops vs FID 分析
2. **极简设计哲学**: 尽可能保持标准 ViT 架构, 仅修改 normalization 层, 不引入任何 vision-specific 归纳偏置
3. **奠定了 latent transformer diffusion 范式**: 后续几乎所有大规模 diffusion model (Sora, SD3, Flux) 都采用了 DiT 或其变体

---

## 6. Limitations & Future Directions

### 论文明确指出

1. **仅验证了 class-conditional 生成**: 未测试 text-conditioned 生成, 但论文建议 DiT 可作为 DALL-E 2 / Stable Diffusion 的 drop-in replacement
2. **未验证 pixel space**: 所有实验在 latent space, pixel space 的 scaling behavior 可能不同

### 从代码推断

3. **Fixed positional embedding**: 代码使用 fixed 2D sinusoidal pos embed (`requires_grad=False`), 不支持可变分辨率。后续工作 (如 SD3 的 RoPE) 解决了这一问题

4. **DDPM 采样效率低**: 代码默认 250 步 DDPM 采样, 未集成 DDIM/DPM-Solver 等加速采样器。对实时控制场景 (如机器人) 不可接受 -- GR00T N1 通过 flow matching + 4 步采样解决了这个问题

5. **单一条件模态**: 代码仅支持 class label conditioning, 扩展到 text/image/state conditioning 需要修改架构 (加入 cross-attention 或修改 adaLN 的输入)

6. **Classifier-free guidance 只应用于前 3 个 channel**: `forward_with_cfg` 中 `eps, rest = model_out[:, :3], model_out[:, 3:]` -- 仅对前 3 个 channel 做 CFG, 论文在 appendix 解释这是为了 reproducibility, 但标准做法应该对所有 channel 做 (代码中有注释但被 comment 掉)

7. **无 gradient checkpointing / FlashAttention**: 开源代码未做任何内存优化, 限制了在单卡上训练大模型的能力

---

## 7. Paper vs Code Discrepancies

### 7.1 Diffusion 代码完全复用 OpenAI

DiT 的 `diffusion/` 目录直接来自 OpenAI 的三个 repo (GLIDE, ADM, IDDPM), 包含大量 DiT 论文未使用的功能:

| 功能 | 代码支持 | 论文使用 |
|------|---------|---------|
| 三种 mean type (PREVIOUS_X, START_X, EPSILON) | 是 | 仅 EPSILON |
| 四种 var type (LEARNED, FIXED_SMALL, FIXED_LARGE, LEARNED_RANGE) | 是 | LEARNED_RANGE |
| KL loss, RESCALED_KL, RESCALED_MSE | 是 | MSE |
| Timestep respacing (减少采样步数) | 是 | 是 (250 步) |
| DDIM 采样 | 是 (ddimN 格式) | 未使用 (用标准 DDPM) |
| Warmup beta schedule, quad schedule 等 | 是 | 仅 linear |

### 7.2 论文未提及的代码实现细节

1. **TF32 加速**: `torch.backends.cuda.matmul.allow_tf32 = True`, 显著加速 A100 训练, 论文未提及

2. **VAE scaling factor**: 训练时 latent 乘以 `0.18215` (`train.py` line 203), 采样时除以同一因子 (`sample.py` line 65)。这是 Stable Diffusion VAE 的标准 scaling 但论文未说明

3. **Weight initialization 细节** (`models.py`):
   - 所有 Linear 层: Xavier uniform
   - PatchEmbed 的 Conv2d: 当作 Linear 做 Xavier uniform (flatten weight)
   - Label embedding: normal(std=0.02)
   - Timestep MLP: normal(std=0.02)
   - adaLN MLP 最后一层: 全零初始化 (这是 adaLN-Zero 的关键)
   - FinalLayer 的 linear: 全零初始化

4. **GELU 近似**: 代码使用 `nn.GELU(approximate="tanh")` 而非精确 GELU, 论文 appendix 提到参考了 [Hendrycks & Gimpel 2016]

5. **Learned sigma 的实现**: 模型输出 `out_channels = in_channels * 2 = 8`, 其中 4 channel 是 noise prediction, 4 channel 是 learned variance (LEARNED_RANGE, 即在 fixedsmall 和 fixedlarge 之间插值)。论文提到 follow Nichol & Dhariwal 的做法但未展开

6. **Label dropout 实现**: `LabelEmbedder` 中 `dropout_prob=0.1`, 训练时以 10% 概率将 class label 替换为 `num_classes` (即第 1001 个 embedding, 代表 "null" class), 这是 classifier-free guidance 的基础

7. **EMA 应用于 frozen pos_embed**: `update_ema` 代码中有 TODO 注释: "Consider applying only to params that require_grad to avoid small numerical changes of pos_embed" -- EMA 会微调 frozen 参数的值, 虽然变化极小

8. **Paper 用 JAX/TPU, 代码用 PyTorch/GPU**: 论文明确说 "implement all models in JAX and train using TPU-v3 pods", 但开源代码是 PyTorch + DDP 实现。这是完全独立的重写

---

## 8. Cross-Paper Comparison

### 8.1 DiT vs DDPM

| 维度 | DDPM (Ho et al., 2020) | DiT (Peebles & Xie, 2023) |
|------|------------------------|---------------------------|
| **核心贡献** | 证明 diffusion 可匹配 GAN | 证明 transformer 可替代 U-Net |
| **Backbone** | U-Net (来自 PixelCNN++) | Transformer (来自 ViT) |
| **操作空间** | Pixel space | **Latent space** (借助 LDM) |
| **条件注入** | Timestep via sinusoidal embed + additive | Timestep + class via **adaLN-Zero** |
| **参数量** | 35.7M (CIFAR10), 114M (256x256) | 33M (DiT-S) ~ 675M (DiT-XL) |
| **Variance** | Fixed (两种选择) | **Learned range** (IDDPM 方式) |
| **Loss** | L_simple (unweighted MSE on epsilon) | MSE + KL for learned variance |
| **训练稳定性** | 未报告 | 极其稳定, 无 loss spike |
| **CIFAR10 FID** | 3.17 (unconditional) | - (未测 CIFAR10) |
| **ImageNet 256 FID** | - (未测) | **2.27** (conditional, w/ guidance) |
| **Scaling analysis** | 无 | Gflops vs FID 的系统分析 |

关键演进: DDPM 证明了 diffusion 的可行性, DiT 将其从 "U-Net + pixel space" 升级为 "Transformer + latent space", 使 diffusion model 进入了可 scaling 的时代。DiT 的 diffusion 代码 (`gaussian_diffusion.py`) 实际上直接复用了 IDDPM (改进版 DDPM) 的实现。

### 8.2 DiT vs ViT

| 维度 | ViT (Dosovitskiy et al., 2020) | DiT (Peebles & Xie, 2023) |
|------|--------------------------------|---------------------------|
| **任务** | 图像分类 (discriminative) | 图像生成 (generative) |
| **输入** | Image patches -> tokens | **Latent** patches -> tokens |
| **Patch size** | 16, 32 | 2, 4, 8 (在 32x32 latent 上) |
| **位置编码** | Learnable 1D | **Fixed 2D sinusoidal** (来自 MAE) |
| **特殊 token** | [CLS] token 做分类 | 无 CLS token |
| **输出** | Class prediction (单个 token) | **空间重建** (所有 token unpatchify) |
| **Normalization** | Standard LayerNorm | **adaLN-Zero** (条件 LN) |
| **Model configs** | S/B/L (同名) | S/B/L/**XL** (新增 XL) |
| **训练数据依赖** | 需要大数据才能超越 CNN | 中等规模 ImageNet 即可 |
| **Training recipe** | 需要 warmup, augmentation, regularization | **无需** warmup/regularization |

DiT 的模型配置直接沿用 ViT 的命名 (S/B/L), 并新增了 XL config (28 层, 1152 维)。代码中 `PatchEmbed`, `Attention`, `Mlp` 三个核心组件直接从 `timm.models.vision_transformer` 导入。DiT 本质上是 **将 ViT 从判别式任务迁移到生成式任务的桥梁**, 最关键的修改就是用 adaLN-Zero 替换标准 LayerNorm 来注入时间步和条件信息。

### 8.3 DiT vs GR00T N1 (System 1 Action Head)

GR00T N1 的 System 1 (action generation) 本质上就是一个 DiT, 从图像生成迁移到动作生成:

| 维度 | DiT (图像生成) | GR00T N1 System 1 (动作生成) |
|------|---------------|------------------------------|
| **生成目标** | Noise prediction (denoise latent -> image) | Action chunk (denoise noise -> 16-step actions) |
| **输入 token** | Patchified noised latent (32x32x4) | Noised action tokens + state encoding |
| **条件信号** | Class label (via adaLN-Zero) | **VLM embedding** (via cross-attention + adaLN-Zero) |
| **DiT layers** | 28 (XL) | 16 (N1) / **32** (N1.6) |
| **Conditioning** | adaLN-Zero only | adaLN-Zero **+ cross-attention** (交替 attend image/text) |
| **Diffusion 框架** | DDPM, 250-1000 步采样 | **Flow matching**, **4 步**采样 |
| **推理频率** | 离线生成 (不关心延迟) | **120Hz** 实时控制 |
| **输出维度** | 32x32x8 (noise + sigma) | Action dim x chunk length (如 29x16) |
| **训练数据** | ImageNet (1.3M images) | Data pyramid (video + sim + teleop) |
| **Cross-embodiment** | N/A | Embodiment-specific projectors |

GR00T N1 对 DiT 的关键改进:

1. **Flow matching 替代 DDPM**: 将 DiT 论文的 250 步 DDPM 降为 4 步 forward Euler, 推理延迟从秒级降到毫秒级 (63.9ms/chunk on L40)。这是从图像生成 (不关心延迟) 到机器人控制 (实时性要求) 必须做的改进。

2. **Cross-attention 引入 VLM tokens**: DiT 原文结论是 adaLN-Zero 优于 cross-attention, 但 GR00T N1 **同时使用两者** -- adaLN-Zero 注入 timestep, cross-attention 注入 VLM 的 vision-language tokens。N1.6 的 `AlternateVLDiT` 进一步区分 image/text token 的 attend 频率。

3. **从 fixed class embedding 到 dynamic VLM embedding**: DiT 的条件是 1000 类中的离散 label, GR00T N1 的条件是 VLM 生成的连续 embedding, 包含丰富的视觉-语言语义。这让 action 生成能够 ground 到当前观测和语言指令。

4. **Embodiment-specific projectors**: DiT 只处理固定维度的 latent, GR00T N1 通过 per-embodiment 的 encoder/decoder 适配不同机器人的 state/action 维度 (从 7-DoF Franka 到 full-body humanoid)。

### 8.4 DiT 在 robotics pipeline 中的位置

```
Perception & Reasoning          Action Generation             Execution
[VLM / World Model]  ------>   [DiT (diffusion/flow)]  ------>  [Robot]

ViT:  提供 visual feature        DiT: 将 condition 映射到           物理执行
CLIP: 对齐 vision-language        连续 action trajectory
LLM:  language reasoning         (本质是条件去噪过程)

Examples:
- Diffusion Policy:  ResNet encoder --> U-Net diffusion --> action
- pi_0:              VLM --> flow matching (transformer) --> action
- GR00T N1:          Eagle-2 VLM --> DiT (flow matching) --> action
```

DiT 的核心价值在于: 它证明了 **transformer 可以做高质量的条件去噪生成**, 且性能随计算量平滑 scale。这个发现直接催生了将 diffusion-based action generation 从 U-Net 迁移到 transformer 的趋势, 使得 vision-language-action 统一架构成为可能。
