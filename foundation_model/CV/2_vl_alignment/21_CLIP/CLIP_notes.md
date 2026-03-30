# CLIP: Learning Transferable Visual Models From Natural Language Supervision

Paper: Radford et al., OpenAI, 2021 (ICML)
Code: https://github.com/OpenAI/CLIP

---

## 1. Core Problem

传统视觉模型依赖固定类别标签训练 (如 ImageNet 的 1000 类)，导致两大局限:
- **泛化性差**: 模型只能识别训练集中出现过的类别，无法 zero-shot 迁移到新任务
- **扩展性差**: 每增加一个新概念就需要人工标注数据

CLIP 提出: 能否用互联网上天然存在的 (image, text) pairs 作为监督信号，学到一个通用的视觉表征，使其能通过自然语言 zero-shot 迁移到任意视觉任务?

---

## 2. Method Overview

### 2.1 Architecture

双塔结构 (dual-encoder):
- **Image Encoder**: ResNet (ModifiedResNet) 或 Vision Transformer (ViT)
- **Text Encoder**: Transformer (causal masked self-attention, 12 layers, 512 width, 8 heads, 63M params)
- **Projection**: 两个 encoder 的输出各自通过线性投影 (W_i, W_t) 映射到共享的 multi-modal embedding space

### 2.2 Training Objective -- Contrastive Learning

给定 batch 中 N 个 (image, text) pairs，最大化 N 个正确配对的 cosine similarity，同时最小化 N^2 - N 个错误配对的 similarity。具体 loss:

```
I_f = image_encoder(I)          # [N, d_i]
T_f = text_encoder(T)           # [N, d_t]
I_e = l2_normalize(I_f @ W_i)   # [N, d_e]
T_e = l2_normalize(T_f @ W_t)   # [N, d_e]
logits = I_e @ T_e.T * exp(t)   # [N, N], t is learnable temperature
labels = arange(N)
loss = (CE(logits, labels, axis=0) + CE(logits, labels, axis=1)) / 2
```

这个 symmetric cross-entropy loss 本质是 multi-class N-pair loss (InfoNCE loss)。

### 2.3 Zero-Shot Transfer

推理时不需要任何训练样本:
1. 将目标数据集的所有类别名转成文本: "A photo of a {label}."
2. 用 text encoder 编码得到所有类别的 text embeddings (可缓存复用)
3. 对新图片用 image encoder 得到 image embedding
4. 计算 cosine similarity, 取最高分的类别作为预测

直觉: text encoder 是一个 hypernetwork, 根据文本动态生成 classifier weights。

---

## 3. Key Designs

### 3.1 Contrastive Objective 而非 Predictive Objective

论文对比了三种 pre-training 方法:
- Transformer language model (predict exact caption): 效率最低
- Bag-of-Words prediction: 中等
- **Contrastive (CLIP)**: 效率最高, 比 predictive 方法快 4x

原因: 预测确切文本太难 (同一张图可以有无数种描述)，而 contrastive 只需判断 "哪个文本和哪个图像配对"，是一个更简单的 proxy task。这个效率差异是 CLIP 能扩展到 400M pairs 的关键。

### 3.2 Linear Projection (而非 Non-linear)

论文明确指出: 不使用 MLP projection head (如 SimCLR 中常用的)，只用单层线性投影。作者推测 non-linear projection 在 self-supervised methods 中可能导致 image encoder "co-adapt" 到当前 batch 的细节，不利于通用表征。这是一个与主流 contrastive learning 做法 (SimCLR, MoCo) 不同的设计选择。

### 3.3 Learnable Temperature Parameter

Temperature tau 不是固定超参数，而是训练时直接优化的 log-parameterized scalar。初始化为 0.07 的等价值，并 clip 到不超过 100 (防止 logits 过大导致训练不稳定)。这消除了一个需要调参的超参数。

---

## 4. Experiments

### 4.1 Zero-Shot Performance
- ImageNet zero-shot: **76.2%** top-1 (匹配原始 ResNet-50 的 supervised 性能)
- 对比 Visual N-Grams (此前唯一做 zero-shot 的方法): ImageNet 11.5% -> 76.2%
- 在 30+ 数据集上测试，16/27 个数据集上 zero-shot CLIP 超过 supervised linear probe on ResNet-50

### 4.2 Representation Learning
- Linear probe: 最佳 CLIP (ViT-L/14@336px) 在 27 个数据集平均 outperform Noisy Student EfficientNet-L2 2.6%
- ViT 比 ResNet 计算效率高约 3x

### 4.3 Distribution Shift Robustness
- 这是最重要的发现之一: zero-shot CLIP 在 ImageNet distribution shift 上远超 supervised models
- ImageNet-R: 37.7% (ResNet-101) vs 88.9% (CLIP), 差距 51.2%
- ImageNet Sketch: 25.2% vs 60.2%, 差距 35%
- Robustness gap 缩小高达 75%

### 4.4 Scaling
- Performance 随 compute 增长呈 log-log linear trend (类似 GPT scaling laws)
- 最大模型训练: ResNet-50x64 在 592 V100 上 18 天; ViT-L/14 在 256 V100 上 12 天

### 4.5 Limitations
- 对抽象/专业任务 (卫星图像 EuroSAT, 肿瘤检测 PatchCamelyon, 计数 CLEVRCounts) 效果差
- Fine-grained classification (花种, 飞机型号) 表现不一
- MNIST 手写数字只有 88% (因为 pre-training 数据中几乎没有手写体)

---

## 5. Impact on Robotics (VLA Models)

CLIP 对 robotics foundation models 的影响是深远的:

### 5.1 Vision Backbone for VLA
- **pi_0**: 使用 PaliGemma 作为 VLM backbone, 而 PaliGemma 的 vision encoder (SigLIP ViT) 就是 CLIP 思路的直接延续 -- contrastive image-text pre-training
- **GR00T N1**: 使用 Eagle-2 VLM, 其 vision encoder 同样源自 CLIP/SigLIP 系列
- RT-2 使用 PaLI-X (ViT-22B) 和 PaLM-E, 其 vision tower 也是 contrastive pre-trained

### 5.2 Open-Vocabulary Understanding
CLIP 赋予 robot 理解任意自然语言指令的能力。在 RT-2 之前, CLIPort 等工作直接用 CLIP features 做 manipulation -- 证明了 CLIP embedding space 对 robotics 的通用性。

### 5.3 Robustness for Sim-to-Real
CLIP 的 distribution shift robustness 对 sim-to-real transfer 至关重要。Robot 部署环境的光照、背景、物体变化都是 distribution shift, CLIP 在这方面远优于 supervised features。

### 5.4 Text as Interface
CLIP 确立了 "language as the universal task specification" 的范式。后续所有 VLA 模型 (RT-2, pi_0, GR00T N1) 都沿用这一思路: 用自然语言描述任务，而非固定类别编码。

---

## 6. Paper vs Code Discrepancies

通过对比论文描述和 `/home/l/ws/doc/paper/foundation_model/methods/21_CLIP/CLIP/clip/model.py` 代码，发现以下差异:

### 6.1 Activation Function: QuickGELU vs Standard GELU
代码使用 `QuickGELU`: `x * sigmoid(1.702 * x)`, 这是 GELU 的近似版本。论文没有提及这一选择。标准 GELU 使用 erf 函数，QuickGELU 用 sigmoid 近似更快但精度略有差异。后续很多使用 CLIP 的工作 (如 OpenCLIP) 改回了标准 GELU。

### 6.2 Text Encoder 使用 Causal Mask
代码中 text transformer 使用 `build_attention_mask()` 生成 causal (autoregressive) mask。论文只简单说 "masked self-attention" 是为了兼容未来用预训练 LM 初始化或添加 language modeling 辅助 loss。但实际代码并没有实现这些功能。这意味着 text encoder 实际上不是 bidirectional 的 (不同于 BERT)，而是 GPT-style 的 causal attention。

### 6.3 Text Feature 取的是 EOT Token 而非 [CLS]
代码: `x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection`
用 `argmax` 找到 EOT (end of text) token 的位置, 取该位置的 hidden state 作为文本表示。这不同于 BERT-style 的 [CLS] token pooling, 也不同于 mean pooling。论文提到了 "activations of the highest layer of the transformer at the [EOS] token" 但没有详细解释 argmax 的技巧。

### 6.4 ResNet Stem: 3 Conv 替代 1 Conv
代码中 `ModifiedResNet` 使用 3 个 3x3 conv 代替标准 ResNet 的 1 个 7x7 conv, 且用 AvgPool 代替 MaxPool。论文说采用了 "ResNet-D improvements" 但没有列出具体改动。

### 6.5 Attention Pooling 替代 Global Average Pooling
ResNet 的最后一层不是 GAP, 而是 `AttentionPool2d` -- 一个单层 QKV attention。query 是 mean-pooled feature, key/value 是所有空间位置。这个设计论文简要提及但代码实现比描述更清晰。

### 6.6 训练中的工程细节未在论文中体现
- `convert_weights()` 函数: 代码显式将模型参数转为 fp16, 论文只说 "mixed precision"
- `logit_scale` 初始化为 `log(1/0.07) = 2.659`, 论文只说 "equivalent to 0.07"
- BPE vocab 大小 49152, context length 77 -- 这些具体数字散落在代码但论文不易找到
- Image preprocessing: BICUBIC interpolation + specific normalization constants (0.481, 0.458, 0.408) -- 这些在复现时很重要但论文没有列出

### 6.7 ViT 中的 LN Position
代码在 patch embedding + positional embedding 之后、transformer 之前有一个 `ln_pre` (LayerNorm), 在 transformer 之后取 [CLS] token 后有 `ln_post`。这两层 LayerNorm 论文只提到 "an additional layer normalization" 但代码有 pre 和 post 两个。

---

## Summary for Robotics Researchers

CLIP 的核心贡献是证明了: 用 contrastive learning 对齐 image-text 表征, 可以学到一个 zero-shot 可迁移、distribution shift robust 的 vision backbone。这个 insight 直接催生了 VLA 模型的技术路线 -- 用大规模 web data 预训练 vision-language model, 再 fine-tune 到 robot control。理解 CLIP 是理解 RT-2, pi_0, GR00T N1 等现代 VLA 模型的基础。
