# ViT: An Image is Worth 16x16 Words -- Transformers for Image Recognition at Scale

Paper: Dosovitskiy et al., Google Brain, ICLR 2021
Code: `/home/l/ws/doc/paper/foundation_model/methods/20_ViT/vision_transformer/`

---

## 1. Core Problem

ViT 要回答一个根本性问题: **纯 Transformer 架构 (不含任何卷积) 能否在视觉任务上达到甚至超越 CNN (Convolutional Neural Network, 卷积神经网络)?**

2020 年的背景:
- NLP 领域 Transformer 已成为标准架构 (BERT, GPT), 模型规模突破 100B 参数且性能未饱和
- CV (Computer Vision, 计算机视觉) 领域 CNN 仍占统治地位 (ResNet, EfficientNet), 虽有多种尝试将 self-attention 引入视觉 (local attention, sparse attention, axial attention), 但无一能替代 CNN
- 核心矛盾: 像素级 self-attention 的计算量是 O(N^2), 对 224x224 图像意味着 50176 个 token, 完全不可行

ViT 的 insight 极其简洁: **不要在像素级做 attention, 把图像切成 16x16 的 patch, 每个 patch 当作一个 "word"**。对 224x224 的图像, 这意味着只有 196 个 token -- 与 NLP 中的序列长度相当, 标准 Transformer 可以直接处理。

这个 idea 并非 ViT 首创 (Cordonnier et al. 2020 使用 2x2 patch), 但 ViT 首次在大规模数据上验证了: **当数据足够多时, 纯 Transformer 不仅能匹配 CNN, 而且在计算效率上更优**。

---

## 2. Method Overview

### 2.1 Architecture

```
Image (H x W x C)
  |
  v
Patch Extraction: 切成 N = HW/P^2 个 patch, 每个 patch 大小 P x P x C
  |
  v
Linear Projection: 每个 patch flatten 后通过可训练线性层映射到 D 维
  |  (实现: 一个 kernel_size=stride=P 的 Conv2D, 等价于 flatten + Dense)
  v
Prepend [CLS] token: 添加一个可学习的 class token (Classification token, 分类标记 -- 一个特殊的 D 维可学习向量, 不对应任何 patch, 通过 attention 聚合所有 patch 的信息, 最终用于分类输出)
  |
  v
Add Position Embeddings: 加上可学习的 1D position embedding (shape: (N+1) x D)
  |
  v
Transformer Encoder: L 层标准 encoder (Pre-LN 变体, 即 LayerNorm 在 attention/FFN 之前)
  |
  v
[CLS] token output -> MLP Head -> Classification
```

### 2.2 关键公式

**Patch Embedding (Eq. 1)**:
```
z_0 = [x_class; x_1_p * E; x_2_p * E; ...; x_N_p * E] + E_pos
```
其中 E in R^{(P^2*C) x D}, E_pos in R^{(N+1) x D}

**Transformer Encoder (Eq. 2-3)**, 使用 Pre-LN (LayerNorm 在 attention/MLP 之前):
```
z'_l = MSA(LN(z_{l-1})) + z_{l-1}    (MSA = Multi-head Self-Attention, 多头自注意力 + residual)
z_l  = MLP(LN(z'_l)) + z'_l           (MLP = Multi-Layer Perceptron, 即 FFN, 前馈网络 + residual)
```

**分类输出 (Eq. 4)**:
```
y = LN(z_L^0)    (取 [CLS] token 的最终输出)
```

### 2.3 Model Variants

| 变体 | Layers | Hidden D | MLP dim | Heads | Params |
|------|--------|----------|---------|-------|--------|
| ViT-B/16 (Base) | 12 | 768 | 3072 | 12 | ~86M |
| ViT-L/16 (Large) | 24 | 1024 | 4096 | 16 | ~307M |
| ViT-H/14 (Huge) | 32 | 1280 | 5120 | 16 | ~632M |

命名规则: ViT-{Size}/{Patch size}。Patch size 越小, 序列越长, 计算越贵。

### 2.4 训练 Pipeline

**Pre-training**: Adam (beta1=0.9, beta2=0.999), batch size 4096, weight decay 0.1, linear warmup 10k steps + cosine/linear decay, 训练分辨率 224

**Fine-tuning**: SGD (Stochastic Gradient Descent, 随机梯度下降) with momentum 0.9, batch size 512, 分辨率通常提升到 384 (甚至 512/518), position embedding 做 2D 双线性插值适应新分辨率

**关键点**: Fine-tuning 时移除整个 pre-training head (包括 pre_logits MLP), 替换为 zero-initialized 的单层线性分类器。

---

## 3. Key Designs

### 3.1 Patch Embedding -- 为什么 "图像 = patch 序列" 可以 work

这是 ViT 最核心的设计, 也是对后续所有 vision foundation model 影响最深的 idea。

**直觉解释**: CNN 通过 locality (局部性 -- 相邻像素更相关) 和 translation equivariance (平移等变性 -- 物体在图像中移动不影响识别) 这两个 inductive bias (归纳偏置 -- 模型架构中内置的先验假设) 来处理图像。ViT 的 patch embedding 只保留了最弱的 locality 假设 (patch 内部的像素相关), 完全放弃了平移不变性和层级结构。代价是需要更多数据来学习这些关系; 收益是模型容量不再被 CNN 的固有结构限制, 可以学到更灵活的表征。

**为什么对 robotics 很重要**: 后续所有 VLA 模型 (CLIP, RT-2, pi_0, GR00T N1) 的 vision encoder 都基于 ViT 的 patch embedding。这个设计有几个对机器人至关重要的属性:
- **输出是一组 spatial tokens (空间标记)** (不是单个向量): 每个 patch token 保留了空间位置信息, 下游模型 (如 GR00T N1 的 DiT, Diffusion Transformer) 可以通过 cross-attention (交叉注意力, 一组 token 关注另一组 token) 选择性地关注图像的不同区域
- **分辨率可变**: 通过 position embedding 插值, 同一模型可以处理不同分辨率的输入, 适应不同相机配置
- **与文本 token 格式统一**: 图像变成 token 序列后, 可以与文本 token 无缝拼接, 这是 VLM/VLA 的架构基础

### 3.2 Scaling 特性 -- 数据量 vs Inductive Bias 的 Tradeoff

ViT 论文最重要的实验发现:

| 数据规模 | ViT vs CNN |
|---------|------------|
| ImageNet (1.3M) | ViT 明显弱于 ResNet (缺少 inductive bias 导致过拟合) |
| ImageNet-21k (14M) | ViT 和 ResNet 持平 |
| JFT-300M (303M) | ViT 显著优于 ResNet, 且使用 2-4x 更少的计算量 |

**核心 insight**: "Large scale training trumps inductive bias." 当数据足够多时, 模型不需要人工注入的先验知识 (locality, equivariance), 可以直接从数据中学到等价甚至更好的表征。

这个发现对 robotics foundation model 的启示是双重的:
1. **正面**: 预训练在大规模视觉数据上的 ViT 可以提供强大的 visual representation, 这是 CLIP/SigLIP 预训练的理论基础
2. **警告**: ViT 在小数据上表现差, 这解释了为什么直接在少量 robot data 上从头训练 ViT 效果不好 (Octo 只有 93M 参数就是为了适应有限数据), 而 pi_0/GR00T N1 必须依赖 VLM 预训练

**Scaling 的具体表现** (Appendix D.2 的 ablation):
- Depth 的 scaling 效果最显著 (8 层到 64 层持续提升, 16 层后收益递减)
- Width 的 scaling 效果最小
- 减小 patch size (增加序列长度) 提升 robust 且不增加参数量
- 结论: **compute 比 parameters 更能预测性能**, scaling 应优先加深度

### 3.3 Position Embedding -- 1D 足矣

论文做了详尽的 ablation (Appendix D.4):

| Position Embedding 方式 | ImageNet 5-shot |
|------------------------|-----------------|
| 无 position embedding | 显著更差 |
| 1D 学习型 (默认) | 基线 |
| 2D 学习型 | 几乎无差异 |
| Relative attention | 几乎无差异 |

**为什么 1D 就够了**: ViT 在 patch 级别操作 (14x14 grid), 空间分辨率很低。在这个分辨率下, 模型可以很容易地从 1D 索引中学到 2D 空间关系。可视化证实 learned 1D position embedding 自动学到了行列结构和距离编码 (Figure 7 center)。

**但这不意味着 2D 完全无用**: 后续工作 (如 CLIP, SigLIP) 在更高分辨率 / 更多 token 的场景下会改用 2D position embedding 或 RoPE, 因为 token 数量增加后 1D 的表达力可能不足。

---

## 4. Experiments

### 4.1 SOTA 对比 (Table 2, 在 JFT-300M 上预训练)

| Model | ImageNet | CIFAR-100 | VTAB (19 tasks) | TPUv3-core-days |
|-------|----------|-----------|-----------------|-----------------|
| BiT-L (ResNet152x4) | 87.54% | 87.31% | 76.3% | 9,900 |
| Noisy Student (EfficientNet-L2) | 88.4% | -- | -- | 12,300 |
| ViT-H/14 | **88.55%** | **94.55%** | **77.63%** | **2,500** |
| ViT-L/16 | 87.76% | 93.44% | 76.28% | 2,500 |

关键发现:
- ViT-H/14 在所有 benchmark 上达到或超过 SOTA
- **计算效率**: ViT 达到同等性能只需 ResNet 的约 1/4 计算量 (2500 vs 9900 TPUv3-core-days)
- ViT-L/16 在 ImageNet-21k (公开数据) 上预训练也表现出色, 8 核 TPUv3 约 30 天可训完

### 4.2 Attention 距离分析 (Figure 7 right)

- 低层: 部分 head 关注全局 (大 attention distance), 部分 head 关注局部 (小 distance) -- 低层的局部 attention 起到了类似 CNN 早期卷积层的作用
- 高层: 几乎所有 head 都是全局 attention
- **对 robotics 的启示**: ViT 自然实现了 local-to-global 的信息聚合, 这对理解场景结构 (近处物体细节 + 远处场景布局) 很重要

### 4.3 Hybrid Architecture

R50+ViT (Hybrid Architecture, 混合架构 -- 用 ResNet 做 stem (前端特征提取), 输出 feature map (特征图) 作为 ViT 的输入):
- 小模型时 hybrid 优于 pure ViT (ResNet stem 提供 low-level feature extraction 的 inductive bias)
- 大模型时差异消失 (ViT 自己学到了等价的 low-level features)

### 4.4 Self-supervised Pre-training (初步探索)

Masked patch prediction (类似 BERT):
- ViT-B/16 self-supervised: 79.9% (比 from scratch 高 2%, 但比 supervised pre-training 低 4%)
- 50% mask rate (比 BERT 的 15% 更高)
- 这个初步结果预示了后来 MAE (2021) 的成功

---

## 5. Related Work Analysis

### 5.1 发展脉络

```
CNN 时代 (2012-2020):
  AlexNet -> VGG -> ResNet -> EfficientNet
  |
  尝试引入 attention:
  Non-local Networks (2018), Stand-alone self-attention (2019),
  Axial Attention (2019), Image Transformer (2018)
  -> 这些方法要么只做局部 attention, 要么需要特殊硬件优化, 无法真正替代 CNN

ViT (2020): 不做任何妥协, 直接用标准 Transformer + patch embedding
  |
  v
后续演化:
  DeiT (2021): 知识蒸馏让 ViT 在 ImageNet alone 上也能训好
  MAE (2021): 自监督预训练 ViT (类似 BERT 的 masked modeling)
  SigLIP/SigLIP-2: CLIP 变体, 用 ViT 做 image encoder
  DiT (2023): 用 ViT 替代 U-Net 做 diffusion backbone
```

### 5.2 ViT 的独特贡献

ViT 不是提出新的技术组件, 而是提出一个 **反直觉的实验发现**: 不需要任何视觉特定的设计, 标准 NLP Transformer 加上最简单的 patch tokenization 就够了。这个发现的意义在于:
1. **统一了 NLP 和 CV 的架构**: 同一个 Transformer 可以同时处理文本和图像, 这是 VLM (如 PaLI, PaliGemma) 的架构前提
2. **解锁了 NLP 的 scaling recipe**: ViT 可以直接复用 NLP 社区成熟的分布式训练、优化器设置 (Adam, large batch, linear warmup)
3. **打通了多模态学习**: 图像和文本变成同质的 token 序列后, CLIP 的 contrastive learning 和 VLA 的 action token 才成为可能

---

## 6. Limitations & Future Directions

### 论文明确指出

1. **Detection 和 Segmentation 未验证**: ViT 论文只做了 classification。Detection (DETR 系列) 和 segmentation (SegFormer) 后来证明了 ViT 在这些任务上也 work。
2. **Self-supervised pre-training 有很大提升空间**: Supervised pre-training 仍比 self-supervised 高 4%, MAE (2021) 后来大幅缩小了这个差距。
3. **Further scaling 可能带来更多收益**: 论文观察到 ViT 性能未饱和, 后来 ViT-22B 等工作证实了这一点。

### 从代码推断的局限

4. **Position embedding 不支持任意分辨率**: 代码中 `interpolate_posembed()` 使用 `scipy.ndimage.zoom` 做双线性插值, 只支持方形 grid (`gs = int(sqrt(N))`), 不支持非方形输入。后续 SigLIP-2 等改进支持 flexible resolution。
5. **Patch size 固定**: 训练和推理必须使用相同的 patch size (代码中 `nn.Conv` 的 kernel_size 和 strides 都等于 `patches.size`), 不支持 multi-scale patch。
6. **[CLS] token vs GAP 的 learning rate 敏感性**: Appendix D.3 揭示 GAP 和 CLS token 效果相当, 但需要不同的 learning rate。这个 sensitivity 在下游使用中容易被忽视。

### 对 robotics 的局限

7. **只输出全局特征或 CLS token**: 原版 ViT 的分类 head 只取 CLS token, 丢弃了所有 spatial information (空间信息)。Robotics 需要 spatial features (物体位置、抓取点), 后续工作通过取消 CLS token 或使用 `unpooled` (未池化, 保留所有 patch token) 输出解决。代码中已有 `classifier='unpooled'` 和 `classifier='token_unpooled'` 选项。
8. **无时序建模**: ViT 处理单帧图像, 没有视频理解能力。Robot 操作需要理解运动和变化, 这需要额外的 temporal modeling (如 GR00T N1 的 VLM 处理 history frames)。

---

## 7. Paper vs Code Discrepancies

基于对 `/home/l/ws/doc/paper/foundation_model/methods/20_ViT/vision_transformer/vit_jax/` 的代码审查。

### 7.1 Patch Embedding 实际用 Conv 而非 Flatten+Dense

论文公式 (Eq. 1) 描述的是 "flatten patches then linear project": `x_p * E, E in R^{(P^2*C) x D}`

代码实际实现 (`models_vit.py:263-270`):
```python
x = nn.Conv(
    features=self.hidden_size,
    kernel_size=self.patches.size,
    strides=self.patches.size,
    padding='VALID',
    name='embedding')(x)
```
数学上完全等价 (stride=kernel_size 的 Conv2D = 不重叠 patch 提取 + 线性投影), 但 Conv 实现更高效, 不需要显式 reshape。代码注释也写了: "We can merge s2d+emb into a single conv; it's the same."

### 7.2 MLP 初始化: 极小的 bias_init

代码中 MLP (`MlpBlock`) 的 bias 初始化使用 `nn.initializers.normal(stddev=1e-6)`, 即近似为零的极小随机数。论文未提及这个初始化细节。这是一个影响训练稳定性的实践 trick。

### 7.3 [CLS] token 初始化为全零

代码 (`models_vit.py:281`):
```python
cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
```
论文未指定 CLS token 的初始化方式。零初始化意味着训练开始时 CLS token 不包含任何信息, 完全依靠 attention 从 patch tokens 聚合信息。

### 7.4 Head 初始化: kernel=zeros, bias=constant

代码 (`models_vit.py:303-307`):
```python
x = nn.Dense(
    features=self.num_classes,
    name='head',
    kernel_init=nn.initializers.zeros,
    bias_init=nn.initializers.constant(self.head_bias_init))
```
分类 head 的 kernel 初始化为 0, bias 可配置 (默认 0)。这意味着模型初始时对所有类别输出相同的 logits (均匀分布)。论文只简要提到 "zero-initialized feedforward layer", 但代码细节更清晰。

### 7.5 Fine-tuning 用 SGD + Gradient Clipping, 而非 Adam

代码中 fine-tuning (`train.py:138-145`) 使用:
```python
tx = optax.chain(
    optax.clip_by_global_norm(config.grad_norm_clip),  # gradient clipping
    optax.sgd(learning_rate=lr_fn, momentum=0.9, accumulator_dtype='bfloat16'),
)
```
论文提到 fine-tuning 用 SGD with momentum, 但 **gradient clipping at global norm 1** 和 **bfloat16 accumulator** 是代码中才能发现的训练 trick。特别是 gradient clipping 在 Appendix B.1 Table 3 中提到对 ImageNet 有益, 但论文正文没有强调。

### 7.6 Gradient Accumulation 实现

代码 (`utils.py:99-119`) 实现了 gradient accumulation:
```python
def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
```
当 `accum_steps > 1` 时, 将 batch 切分后逐块计算梯度再累加。论文未讨论这一工程细节, 但对在有限 TPU 内存下训练大 batch size 至关重要。

### 7.7 Hybrid Model (ResNet+ViT) 中 ResNet 的配置

代码 (`configs/models.py:276-282`) 中 R50+ViT-B/16 的 ResNet50 配置:
```python
config.resnet.num_layers = (3, 4, 9)  # NOT the standard (3, 4, 6, 3)
```
注释说明: 使用 (3, 4, 9) 而非标准 ResNet50 的 (3, 4, 6, 3), 这样 downsample factor 为 2^(1+3)=16, 与 ViT 的 /16 patch size 匹配。论文只说 "modified ResNet50" 但没有解释这个具体配置。

### 7.8 ResNet 使用 Weight Standardization + GroupNorm

代码 (`models_resnet.py:23-27, 30-39`) 中 hybrid model 的 ResNet 使用:
- `StdConv`: 对卷积 kernel 做 weight standardization (减均值除标准差)
- `GroupNorm`: 替代 BatchNorm

论文提到这是 BiT (Big Transfer) 的改进, 有利于 transfer learning, 但没有详细说明 weight standardization 的具体实现。

### 7.9 代码中存在 Ti/S 变体但论文未涉及

代码 (`configs/models.py`) 注册了 ViT-Ti/16 (hidden=192, 3 heads) 和 ViT-S/16 (hidden=384, 6 heads) 两个小变体, 以及多种 GAP 分类器变体 (`ViT-S_16-gap-norep` 等)。这些是后续 "How to train your ViT" (AugReg) 论文引入的, 不在原始 ViT 论文中。

### 7.10 图像预处理: 值域映射到 [-1, 1]

代码 (`preprocess.py:171-172`):
```python
def _value_range(self, image):
    image = tf.cast(image, tf.float32) / 255
    return -1 + image * 2
```
图像值域映射到 [-1, 1] 而非常见的 [0, 1] 或 ImageNet normalization。论文未指明预处理方式, 但这个细节对复现很重要。

---

## 8. Cross-Paper Comparison: ViT 作为 VLA 模型的 Vision Backbone

### 8.1 ViT -> CLIP -> VLA 的演化链

ViT 为后续模型奠定了 "图像 = patch token 序列" 的范式。具体传承:

| 阶段 | 模型 | ViT 的角色 | 关键改进 |
|------|------|----------|---------|
| 视觉分类 | ViT (2020) | 完整模型 | Patch embedding + standard Transformer |
| 视觉-语言对齐 | CLIP (2021) | Image encoder | Contrastive pre-training (对比预训练), 获得 language-aligned visual features |
| 视觉-语言理解 | PaliGemma / Eagle-2 | VLM (Vision-Language Model, 视觉语言模型) 的 vision tower | 与 LLM decoder 联合训练, 获得 grounded visual reasoning |
| 机器人控制 | pi_0 / GR00T N1 | VLA (Vision-Language-Action, 视觉语言动作模型) 的 vision encoder | 视觉特征驱动动作生成 |

### 8.2 ViT vs CLIP vs DDPM -- 三种范式的核心差异

| 维度 | ViT | CLIP | DDPM |
|------|-----|------|------|
| **任务** | 图像分类 | 图像-文本匹配 | 图像生成 |
| **核心创新** | Patch tokenization (图像切块为 token) | Contrastive alignment (对比对齐) | Iterative denoising (迭代去噪) |
| **对 VLA 的贡献** | 视觉 token 化范式 | Language-aligned vision backbone (语言对齐的视觉主干) | Action generation 框架 (动作生成) |
| **训练数据** | 标注分类数据 (ImageNet-21k/JFT) | 互联网 image-text pairs (400M) | 无标注图像 |
| **输出** | 单个 class label | Image/text embedding pair | 生成的图像 |
| **Scaling 特性** | 数据量决定 ViT vs CNN 优劣 | Compute-performance log-log linear | FID (Frechet Inception Distance, 生成质量指标, 越低越好) 随模型增大单调下降 |

### 8.3 ViT 在 pi_0 和 GR00T N1 中的具体使用方式

**pi_0 中的 ViT**:
- Vision encoder: SigLIP ViT (PaliGemma 的 vision tower), 来自 CLIP 思路的 contrastive pre-trained ViT
- 图像经 ViT 编码后得到 spatial tokens, 与 language tokens 拼接, 送入 Gemma 2B LLM 做联合推理
- Action expert (300M) 通过 shared self-attention 与 VLM tokens 交互
- ViT 提供的是 **language-aligned visual representation**: 模型可以理解 "pick up the red cup" 中 "red cup" 对应图像的哪个区域

**GR00T N1 中的 ViT**:
- Vision encoder: SigLIP-2 ViT (Eagle-2 VLM 的 vision tower), 224x224 输入, pixel shuffle 压缩后 64 tokens/frame
- 使用 **中间层** (第 12 层) 而非最终层的 VLM embeddings (实验证明中间层更好)
- ViT 输出通过 cross-attention 传递给 DiT (Diffusion Transformer) 做动作生成
- N1.6 升级: VLM 换用 Cosmos-Reason-2B, 支持 flexible resolution (无需 padding)

### 8.4 ViT 的 Patch Embedding 为什么适合 VLA

| 属性 | 对 VLA 的意义 |
|------|-------------|
| **每个 patch token 保留空间信息** | 机器人需要定位物体 (哪个 patch 包含目标物体), 而非只要 "全局类别" |
| **Token 序列格式与文本统一** | 图像 token 和语言 token 可以在同一个 Transformer 中处理, 实现 vision-language 融合 |
| **支持多图像输入** | 多相机观测只需拼接更多 patch tokens, 架构不变 (pi_0 支持 3 个相机, GR00T N1 支持多 embodiment 的不同相机配置) |
| **计算量可控** | 通过调节 patch size 或 pixel shuffle 控制 token 数量 (GR00T N1 用 pixel shuffle 从 196 tokens 压缩到 64) |
| **Pre-trained features 丰富** | CLIP/SigLIP 预训练的 ViT 已经理解了丰富的视觉概念, fine-tune 到 robot task 时 data efficiency 极高 (GR00T N1 用 10% 数据超过 Diffusion Policy 用全量数据) |

### 8.5 Scaling 特性对比

| 模型 | Scaling 维度 | 规律 |
|------|------------|------|
| ViT | 模型大小 + 数据量 | 大数据时 ViT 超越 CNN; depth > width; 性能与 compute log-log linear |
| CLIP | Compute | 性能与 compute 呈 log-log linear (类似 GPT scaling laws) |
| DDPM | 模型大小 | FID 随模型增大单调改善; U-Net 是瓶颈 (后被 DiT 替代) |
| pi_0 | VLM 规模 + 数据量 | PaliGemma 3B 已足够 (比 RT-2 的 55B 小很多), 数据质量/多样性更关键 |
| GR00T N1 | 数据金字塔 + DiT depth | N1->N1.6: DiT 从 16 层加到 32 层; 数据从 88h real 扩增到 827h neural |

**关键 takeaway**: ViT 的 scaling 发现 ("data trumps inductive bias") 是整个 foundation model for robotics 路线的理论基石。没有这个发现, 就不会有 "在 web-scale 数据上预训练 ViT, 再迁移到 robot task" 的范式。

---

## Summary for Robotics Researchers

ViT 论文的核心贡献不是具体的技术创新 (patch embedding, position embedding 都很简单), 而是一个被大规模实验验证的 paradigm shift: **纯 Transformer + patch tokenization + 大规模预训练 = 比 CNN 更好的视觉表征**。

对于 robotics foundation model pipeline 的理解:

1. **ViT 提供了统一的视觉表征格式** (spatial tokens), 使得 vision 可以与 language, action 在同一个 Transformer 框架内交互
2. **CLIP/SigLIP 将 ViT 的表征与语言对齐**, 使得机器人可以通过自然语言指令理解视觉场景
3. **DDPM/Flow Matching 提供了动作生成的框架**, 将 ViT+CLIP 得到的 visual-language features 转化为连续动作
4. **pi_0/GR00T N1 将以上三者整合**: ViT-based VLM (理解场景+指令) + Diffusion/Flow Matching (生成动作) = VLA

ViT 是这条 pipeline 的第一块基石。
