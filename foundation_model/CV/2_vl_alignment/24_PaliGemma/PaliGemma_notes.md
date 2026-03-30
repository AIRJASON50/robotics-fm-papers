# PaliGemma: A versatile 3B VLM for transfer

Paper: Beyer et al., Google DeepMind, July 2024 (arXiv:2407.07726)
Model: https://huggingface.co/google/paligemma

---

## 1. Core Problem

现有 VLM 存在两极化问题:
- **大模型 (PaLI-X 55B, PaLM-E 562B)**: 性能优秀但体积巨大，fine-tuning 成本极高，无法作为下游任务的实用 backbone
- **小模型 (LLaVA, Mini-Gemini 等)**: 依赖 instruction tuning 和 GPT-4 生成的数据，优化方向偏向 "user-friendly" 而非 "transfer-friendly"

PaliGemma 要解决的核心问题: **能否用不到 3B 参数构建一个 base VLM，使其在 40+ 种视觉-语言任务上都能通过简单 fine-tuning 达到 SOTA 水平？**

关键定位: PaliGemma 不是 instruction-tuned 的 chatbot，而是一个为 transfer learning 优化的 base model -- 预训练学"技能"，fine-tuning 学"格式"。

---

## 2. Method Overview

### 2.1 Architecture

三组件结构:

```
Image --> SigLIP ViT-So400m (400M) --> Linear Projection --> [image tokens]
                                                                  |
Text  --> SentencePiece Tokenizer  --> Gemma Embedding  --> [text tokens]
                                                                  |
                                                       Concatenate: [img..., BOS, prefix..., SEP, suffix..., EOS, PAD...]
                                                                  |
                                                       Gemma 2B Decoder (prefix-LM mode)
                                                                  |
                                                            Output text
```

| Component | Source | Parameters | Role |
|-----------|--------|-----------|------|
| SigLIP ViT-So400m | Publicly available checkpoint | ~400M | Contrastive vision encoder (sigmoid loss) |
| Linear Projection | Zero-initialized, trained from scratch | ~2M | Map SigLIP output dim to Gemma vocab dim |
| Gemma 2B v1.0 | Publicly available pretrained checkpoint | ~2B | Autoregressive language decoder |
| Total | -- | ~2.8B | -- |

### 2.2 Token Sequence Structure

```
tokens = [image tokens...,
          BOS, prefix tokens..., SEP,
          suffix tokens..., EOS, PAD...]
```

- Image tokens 固定在序列开头，数量由分辨率决定 (224px: 256, 448px: 1024, 896px: 4096)
- `\n` 作为 SEP token，单独 tokenize 避免与 prefix/suffix 合并
- Prefix = task prompt (input)，Suffix = expected output

### 2.3 Prefix-LM Attention Masking

这是 PaliGemma 区别于标准 causal LM 的核心设计:

```
         img1 img2 img3 [bos] inp1 inp2 [sep] out1 out2 [eos] [pad]
img1      Y    Y    Y    Y     Y    Y    N     N    N    N     N
img2      Y    Y    Y    Y     Y    Y    N     N    N    N     N
img3      Y    Y    Y    Y     Y    Y    N     N    N    N     N
[bos]     Y    Y    Y    Y     Y    Y    N     N    N    N     N
inp1      Y    Y    Y    Y     Y    Y    N     N    N    N     N
inp2      Y    Y    Y    Y     Y    Y    N     N    N    N     N
[sep]     Y    Y    Y    Y     Y    Y    Y     N    N    N     N
out1      Y    Y    Y    Y     Y    Y    Y     Y    N    N     N
out2      Y    Y    Y    Y     Y    Y    Y     Y    Y    N     N
[eos]     Y    Y    Y    Y     Y    Y    Y     Y    Y    Y     N
[pad]     Y    Y    Y    Y     Y    Y    Y     Y    Y    Y     Y
```

- **Prefix (image + prompt)**: Full bidirectional attention -- 所有 prefix tokens 互相可见
- **Suffix (output)**: Causal autoregressive attention -- 只能看到自己和前面的 tokens
- **Loss**: 只在 suffix tokens 上计算 next-token-prediction loss

直觉: image tokens 可以 "lookahead" 看到 prompt，理解任务要求后再编码视觉信息，比纯 causal masking 更高效地利用模型容量。

### 2.4 Pretraining Stages

| Stage | Resolution | Image Tokens | Text Length | Data | Focus |
|-------|-----------|-------------|-------------|------|-------|
| Stage0 | -- | -- | -- | Off-the-shelf | Unimodal pretrain (SigLIP + Gemma) |
| Stage1 | 224px | 256 | 128 | 1B examples | Multimodal knowledge, broad skills |
| Stage2 | 448/896px | 1024/4096 | 512 | 50M/10M | High-resolution adaptation |
| Stage3 | Task-specific | Task-specific | Task-specific | Task data | Fine-tuning to specific task |

Stage1 的关键决策:
- **不冻结 image encoder** (与 PaLI 传统做法不同): 受 CapPa/LocCa 启发，captioning 等任务可以修复 contrastive encoder 在空间关系理解上的盲区
- **Slow linear warm-up for SigLIP learning rate**: 避免 LLM 梯度一开始就破坏 vision encoder 质量
- **1B examples long training**: 比 LLaVA (~1M) 长 1000x，确保广泛的知识覆盖

### 2.5 Pretraining Task Mixture

| Task | Format | Data Source |
|------|--------|------------|
| caption {lang} | Plain captioning in 100+ languages | WebLI, CC3M-35L |
| ocr | Full text transcription (raster order) | Public OCR system |
| answer en {question} | VQA in 35 languages | CC3M-35L, OpenImages |
| detect {thing} ; {thing} ; ... | Multi-object detection with location tokens | Pix2Seq-style pseudo-labels |
| segment {thing} ; {thing} ; ... | Instance segmentation with VQVAE mask tokens | OWL-ViTv2, SAM pseudo-labels |
| caption \<ymin\>\<xmin\>\<ymax\>\<xmax\> | Grounded captioning | LocCa-style |

每个任务有唯一的 prefix 标识，避免跨任务学习信号冲突。

### 2.6 Special Tokens

| Token Type | Count | Format | Purpose |
|------------|-------|--------|---------|
| Location tokens | 1024 | \<loc0000\> - \<loc1023\> | Binned normalized image coordinates for detection/grounding |
| Segmentation tokens | 128 | \<seg000\> - \<seg127\> | VQVAE tokenized single-object masks |

New token initialization: 使用 small Gaussian noise (sigma=0.02) 而非 AvgEmb 策略 -- 虽然初始 loss 更高，但最终 transfer 效果更好。

---

## 3. Key Designs

### 3.1 Prefix-LM: Maximizing Capacity of Small Models

传统 VLM 用纯 causal attention，image tokens 只能看到前面的 image tokens。Prefix-LM 让所有 prefix tokens (image + prompt) 互相可见:

**为什么重要**: 对于只有 2B 参数的小模型，每一层的表达能力都很宝贵。Prefix-LM 让 image tokens 能看到 query prompt，提前理解 "我需要关注图像的哪些方面"。Ablation 证实 prefix-LM 在 36 个 transfer tasks 平均分上超过纯 causal masking ~2 分。

**与 pi_0 的联系**: pi_0 的 blockwise causal attention 就是 prefix-LM 的进一步扩展 -- [images, language] 块内双向 attention，action 块 causal attention。

### 3.2 Linear Projection: Simplicity Wins

将 SigLIP output 投影到 Gemma embedding space 时，论文对比了:
- Linear projection (单层矩阵乘法)
- MLP (1 hidden layer + GeLU)

结果: 两者在 full fine-tuning 下性能几乎相同 (77.2 vs 77.1)。在 frozen 场景下，linear 甚至略优于 MLP (70.7 vs 69.7)。

直觉: SigLIP 的输出已经是高质量的语义表征，不需要复杂的非线性变换。简单的线性投影减少参数、降低过拟合风险，且 zero-initialization 确保训练初期 image tokens 不会给 LM 注入噪声。

**对 robotics 的意义**: 这个发现支持了 pi_0 的设计 -- 不需要复杂的 vision-language connector，简单连接即可，把计算预算留给 action expert。

### 3.3 Unfrozen Vision Encoder with Slow Warmup

PaLI 系列传统做法是 Stage1 冻结 image encoder，但 PaliGemma 打破了这个传统:

- **不冻结**: 让 captioning 等任务的梯度流回 SigLIP，修复 contrastive training 在空间关系/定位上的盲区 (CapPa/LocCa 的发现)
- **Slow linear warmup**: SigLIP learning rate 从 0 缓慢上升，避免初期 LLM 的随机梯度破坏 vision encoder

Ablation 结果:
- 冻结 ViT: 在 transfer 后没有差异 (说明 fine-tuning 能补回来)
- 但冻结 ViT 时, pretraining 阶段的 captioning/detection perplexity 显著更差
- 说明 unfrozen ViT 在 pretraining 阶段学到了更好的空间理解能力

---

## 4. Experiments

### 4.1 Main Results (Table 1)

PaliGemma 在 30+ academic benchmarks 上 fine-tuning 后的表现 (全部未在预训练数据中出现):

| Category | Representative Tasks | 224px | 448px | 896px |
|----------|---------------------|-------|-------|-------|
| Image Captioning | COCOcap | 141.9 | 144.6 | -- |
| | TextCaps | 127.5 | 153.9 | -- |
| VQA | VQAv2 | 83.2 | 85.6 | -- |
| | ScienceQA | 95.4 | 95.9 | -- |
| | DocVQA | 43.7 | 78.0 | 84.8 |
| | InfoVQA | 28.5 | 40.5 | 47.8 |
| | ChartQA (human) | 40.0 | 54.2 | -- |
| Segmentation | RefCOCO (testA) | 75.7 | 77.9 | 78.7 |
| Video | ActivityNet-QA | 50.8 | -- | -- |
| | MSRVTT-QA | 50.1 | -- | -- |

关键观察:
- **不到 3B 参数匹配 10x-100x 更大的 PaLI-X/PaLM-E**
- Resolution-sensitive 任务 (DocVQA, OCR, ChartQA) 从高分辨率受益巨大
- Video 任务通过将 16 帧编码为 16x256=4096 image tokens 处理

### 4.2 Ablation Findings

#### 4.2.1 Pretraining Duration (Figure 4)
- 完整 1B examples 最佳
- 100M examples (10x shorter) 是合理的 ablation trade-off -- 不显著损害任何单一任务
- 完全跳过 Stage1 是最差选择

#### 4.2.2 Prefix-LM vs Causal Masking (Figure 5)
- Prefix-LM (loss only on suffix) > Causal (loss on suffix + prefix) > Causal (loss on suffix + prefix + image)
- 在 prefix 上也加 loss 反而降低性能 -- 强制模型 "猜问题" 分散了学习信号

#### 4.2.3 Freeze vs Unfreeze (Figure 7)
- 全部 unfreeze (TT) 最佳
- 冻结 LM 或 reset 任何组件都显著降低性能
- 关键: Stage0 pretrained components 是成功的基础

#### 4.2.4 Resolution (Figures 9-10)
- 分辨率提升的收益来自两方面: (1) 更高的图像信息量 (2) 更长的序列给模型更大容量
- 两方面贡献大致相等
- 需要 resolution-specific checkpoints (448 checkpoint 直接用在 224 比用 224 checkpoint 更差)

#### 4.2.5 Transfer with Limited Examples (Figure 12)
- 64 examples: 约 40-60% full-data score
- 256 examples: 约 80% full-data score
- 4k examples: 约 90% full-data score
- 说明 PaliGemma 预训练学到的技能确实可以快速 transfer

### 4.3 Transferability (Section 6)

| Aspect | Finding |
|--------|---------|
| Repeatability | 5 runs std < 0.5 for most tasks |
| Hyper-parameter sensitivity | Single default recipe (lr=1e-5, bs=256, no dropout/label-smooth) works for 37/43 tasks within 2.5% of best |
| Few-shot capability | 10k examples sufficient for most tasks |

### 4.4 Notable Findings (Section 7)

- **Simple resize for segmentation**: 直接 resize 到正方形效果等同于 aspect-ratio-preserving augmentations
- **RoPE interpolation unnecessary for upscaling**: Stage2 不需要 RoPE 插值来保持位置编码语义
- **Zero-shot generalization to 3D renders**: PaliGemma 在 Objaverse 上 zero-shot 效果出人意料地好
- **MMVP SOTA by large margin**: 224px PaliGemma 在 MMVP 上 47.3%，GPT4-V 38.7%, Gemini 40.7%

---

## 5. Related Work Analysis

### 发展脉络

```
First Generation (Contrastive):
  CLIP (2021) / ALIGN / SigLIP
    --> Image-text alignment via contrastive learning
    --> Open-vocabulary understanding

Second Generation (Generative):
  PaLI (2022) --> PaLI-X (2023) --> PaLI-3 (2023)
    --> Encoder-decoder VLMs with ViT + LM
    --> SigLIP replaces CLIP as vision encoder in PaLI-3

Third Generation (Instruction-tuned):
  LLaVA (2023) / InstructBLIP / Flamingo
    --> Focus on user-friendliness
    --> Rely on GPT-4 generated instruction data

PaliGemma (2024):
    --> Takes PaLI-3's recipe (SigLIP + decoder-only LM)
    --> Shrinks to 3B, open-source, transfer-optimized base model
    --> NOT instruction-tuned (base model for further research)
```

### PaliGemma 的独特定位

| Dimension | LLaVA | PaLI-3 | Flamingo | PaliGemma |
|-----------|-------|--------|----------|-----------|
| Parameters | 7-13B | 5B | 80B | 2.8B |
| Open weights | Yes | No | No | Yes |
| Vision encoder | CLIP ViT-L | SigLIP ViT-G | NFNet-F6 | SigLIP ViT-So400m |
| LM backbone | LLaMA/Vicuna | UL2 3B | Chinchilla 70B | Gemma 2B |
| Training data | GPT-4 generated | Web-scale multimodal | Web-scale | Web-scale multimodal (no GPT-4 data) |
| Design goal | Instruction following | Transfer | Few-shot | Transfer |
| Pretraining scale | ~1M | ~1B | ~1B | ~1B |

PaliGemma 的独特性: **最小的开源 transfer-optimized VLM**，不依赖任何商业 VLM 生成的数据。

---

## 6. Limitations & Future Directions

### 论文明确提到

- 不是 instruction-tuned model: 直接 zero-shot 使用效果不如 LLaVA 等 (需要 fine-tuning)
- 只提供 base model，用户需要自行 fine-tune 到具体任务
- Resolution-specific checkpoints 增加了使用复杂度 (3 个 checkpoint)

### 从设计和实验推断

- **Decoder-only 架构的局限**: 论文 Section 5.6 的 Fuyu-style 消融说明去掉 vision encoder 直接处理 raw patches 可行但效率低 (~15 分差距)，暗示当前 ViT+LM 管道仍是必要的 overhead
- **视频理解能力有限**: 16 帧 x 256 tokens = 4096 tokens 的粗暴拼接方式，无法处理长视频或时序关系复杂的任务
- **缺乏交互能力**: 无 dialogue history，不支持多轮对话或 in-context learning
- **Frozen-encoder transfer 未深入探索**: 虽然 unfrozen 预训练更好，但 transfer 后差异消失 -- 这暗示更高效的 transfer 方法 (如 adapter, LoRA) 可能同样有效
- **无 3D 理解**: 虽然 zero-shot 在 Objaverse 上表现不错，但没有显式的 3D/depth reasoning

### 对 robotics 领域的未来意义

- PaliGemma 的 "base model for transfer" 定位天然适合 VLA 场景: robot control 恰好需要对 base model fine-tuning，而非 instruction following
- 2.8B 参数量是 edge deployment 的上限 (RTX 4090 可实时推理)
- SigLIP 的空间理解 + Gemma 的推理能力 = 理想的 VLA vision-language backbone

---

## 7. Paper vs Code Discrepancies

PaliGemma 在 HuggingFace 上以 `google/paligemma` 系列发布，包括多种分辨率和 fine-tuned 变体。

### HuggingFace 实现 vs 论文

| Aspect | Paper | HuggingFace (google/paligemma) |
|--------|-------|-------------------------------|
| Model variants | 224px, 448px, 896px | paligemma-3b-pt-224, paligemma-3b-pt-448, paligemma-3b-pt-896 (pt=pretrained) |
| Fine-tuned variants | 各 benchmark 独立 fine-tune | paligemma-3b-ft-* (DocVQA, RefCOCO, etc.) |
| Mix variant | 多任务同时 fine-tune | paligemma-3b-mix-224/448 |
| Tokenizer | Gemma SentencePiece (256k vocab) | 与论文一致，扩展了 loc/seg tokens |
| Image preprocessing | Resize to square, random JPEG/resize augmentation | 标准 resize + normalize，无训练时增强 |
| Training framework | JAX + big_vision on TPUv5e | HuggingFace Transformers (PyTorch) |
| Attention implementation | Custom prefix-LM mask | 通过 `token_type_ids` 控制 prefix-LM mask |

### 论文未详述但 HuggingFace 代码体现的细节

- **Image normalization**: SigLIP 使用特定的均值/方差归一化 (与 CLIP 不同)
- **Token type IDs**: HuggingFace 用 `token_type_ids=0` 表示 prefix (full attention)，`token_type_ids=1` 表示 suffix (causal attention)，这是 prefix-LM 的工程实现
- **Vocab size 扩展**: 实际 vocab 包含 256k base tokens + 1024 loc tokens + 128 seg tokens = 257,152 tokens
- **PaliGemma 2**: 后续发布了 PaliGemma 2 (基于 Gemma 2)，提供 3B, 10B, 28B 三种规模

---

## 8. Cross-Paper Comparison

### 8.1 PaliGemma vs CLIP (methods/21_CLIP)

| Dimension | CLIP (2021) | PaliGemma (2024) |
|-----------|-------------|------------------|
| Architecture | Dual-encoder (image + text encoders, separate) | Single decoder with vision encoder feeding into LM |
| Vision encoder | ViT-L/14 (CLIP-trained) | SigLIP ViT-So400m (sigmoid loss, not softmax) |
| Text model | 12-layer Transformer (63M, for embedding only) | Gemma 2B (full generative LM) |
| Training objective | Contrastive (InfoNCE, softmax over batch) | Generative (next-token prediction on suffix) |
| Output | Embedding vector (for similarity matching) | Text string (any format: caption, coordinates, labels) |
| Zero-shot capability | Strong (prompt engineering with class names) | Weak without fine-tuning (designed for transfer) |
| Task flexibility | Classification, retrieval (embedding-based) | Any vision-language task (generative) |
| Parameters | ~400M (ViT-L/14) | ~2.8B |

**SigLIP 与 CLIP 的核心区别**:

SigLIP 是 CLIP 的直接改进，替换了 CLIP 的 softmax-based contrastive loss 为 sigmoid-based loss:

```
CLIP:  loss = -log(exp(sim(i,t)/tau) / sum(exp(sim(i,t_j)/tau)))   # softmax over batch
SigLIP: loss = -log(sigmoid(z * (sim(i,t)/tau - b)))                # per-pair sigmoid
```

Sigmoid loss 消除了 batch 内负样本的归一化，使得 loss 可以在更小的 batch size 下训练，同时允许 "shape optimized" 的模型架构 (ViT-So400m = Shape Optimized 400M)。

**CLIP -> SigLIP -> PaliGemma 的演进逻辑**:
1. CLIP 证明了 contrastive image-text pretraining 可以学到高质量 visual representations
2. SigLIP 改进了 contrastive loss (sigmoid vs softmax)，并搜索了最优模型形状 (So400m)
3. PaliGemma 将 SigLIP 作为 vision encoder，接入 Gemma LM，从 "embedding model" 升级为 "generative model"

### 8.2 PaliGemma vs pi_0 (methods/24_pi0)

| Dimension | PaliGemma | pi_0 |
|-----------|-----------|------|
| Role | Vision-Language Model (VLM) | Vision-Language-Action Model (VLA) |
| Architecture | SigLIP + Gemma 2B | PaliGemma + Action Expert (300M) |
| Parameters | 2.8B | 3.3B (2.8B VLM + 300M action expert) |
| Input | Image + text prompt | Image + text prompt + proprioception + noisy actions |
| Output | Text string | Continuous action chunk (H=50 steps) |
| Action generation | None | Flow matching (10-step Euler integration) |
| Attention | Prefix-LM (bidirectional prefix, causal suffix) | Blockwise causal (bidirectional VLM block, causal action block) |
| Training | Web-scale multimodal data (1B examples) | 10,000 hours robot teleoperation data |
| Inference | Text generation (autoregressive) | Action generation (denoising, 73ms on RTX 4090) |

### 8.3 Why pi_0 Chose PaliGemma as VLM Backbone

pi_0 选择 PaliGemma 而非其他 VLM (LLaVA-7B, Qwen-VL-7B, InternVL 等) 有以下技术原因:

**1. Size-Performance Sweet Spot**

PaliGemma 论文的核心贡献就是证明 2.8B 参数可以匹配 55B PaLI-X 的 transfer 性能。对于 robotics:
- 7B+ VLM 在 RTX 4090 上推理延迟过高 (>200ms)，无法满足 50Hz 控制
- pi_0 需要在 VLM 之上再加 300M action expert，总量控制在 3.3B
- PaliGemma 的 2.8B 是唯一在此规模下经过充分验证的 VLM

**2. SigLIP Vision Encoder: Spatial Understanding**

SigLIP ViT-So400m 相对于 CLIP ViT 的优势:
- Sigmoid loss 训练更稳定 (不依赖 batch 内负样本归一化)
- Shape-optimized architecture 在 400M 参数下性能最优
- PaliGemma Stage1 **unfrozen ViT training** 修复了 contrastive encoder 在空间关系上的盲区 -- 这对 robot manipulation (需要精确的空间理解) 至关重要

**3. Transfer-Optimized Base Model**

PaliGemma 是专为 transfer 设计的，不是 instruction-tuned model:
- pi_0 需要的不是 "聊天能力"，而是 "将视觉信息编码为语义丰富的 token sequence" 的能力
- PaliGemma 预训练了 detection/segmentation/grounding 等空间任务，这些 "skills" 直接有益于 robot manipulation
- 论文证明 256 examples 就能 fine-tune 到 80% full-data performance，说明预训练的 skill 确实可以快速 transfer

**4. Prefix-LM Architecture Compatibility**

PaliGemma 的 prefix-LM attention mask 与 pi_0 的 blockwise causal mask 天然兼容:
- PaliGemma: [image + prompt] = full attention (prefix), [output] = causal (suffix)
- pi_0: [image + language] = full attention (block 1), [proprioception] = block 2, [actions] = causal (block 3)
- pi_0 本质上是在 PaliGemma 的 suffix 位置插入了 action expert，而不是破坏原有的 VLM 结构

**5. Open Weights + Gemma Ecosystem**

- PaliGemma 是完全开源的 (Apache 2.0 / Gemma terms)
- Gemma 2B 作为 LM backbone，与 Google 的 JAX/TPU 生态兼容 (pi_0 使用 JAX 训练)
- 不依赖 GPT-4 生成的数据，避免潜在的 license 风险

### 8.4 Architecture Evolution: CLIP -> PaliGemma -> pi_0

```
CLIP (2021):
  Vision Encoder (ViT) --> [CLS embedding]
  Text Encoder          --> [EOS embedding]
  Contrastive Loss: align embeddings
  Output: similarity score

PaliGemma (2024):
  SigLIP (improved CLIP) --> [image token sequence] --> Linear Projection
                                                              |
  Text Tokenizer --> Gemma Embedding  -->  [text token sequence]
                                                              |
                                                    Gemma 2B (prefix-LM)
                                                              |
                                                        Output text

pi_0 (2024):
  PaliGemma VLM --> [VLM hidden states]     (shared attention)     Action Expert
        |                    |                      ^                    |
  [image + language]    VLM weights ---------> joint QKV attention <--- action weights
                                                    |
                                            [action tokens]
                                                    |
                                          Flow Matching Denoising
                                                    |
                                         Continuous Action Chunk
```

从 CLIP 到 pi_0 的演进逻辑:
1. **CLIP**: 对齐视觉和语言的 embedding space (理解 "what")
2. **SigLIP**: 改进 contrastive loss + 架构搜索 (更好的 "what")
3. **PaliGemma**: 从 embedding 升级到 generation，加入 spatial tasks (理解 "what" + "where")
4. **pi_0**: 从 language generation 扩展到 action generation (理解 "what" + "where" + "how to act")

每一步都保留了前一步的核心能力，同时扩展了输出空间的维度。

---

## Summary for Robotics Researchers

PaliGemma 是当前 VLA 模型 (pi_0, pi_0.5) 的 vision-language backbone，其核心价值在于:

1. **Compact but capable**: 2.8B 参数在 40+ 任务上匹配 55B 模型，是 real-time robotics inference 的理想规模
2. **Spatial-aware vision encoder**: SigLIP + unfrozen pretraining 提供了 contrastive+generative 双重训练的视觉表征，比纯 CLIP 更擅长空间理解
3. **Transfer-optimized**: 不是 chatbot，而是为下游 fine-tuning 设计的 base model -- 正是 VLA 场景需要的
4. **Prefix-LM architecture**: 与 pi_0 的 blockwise causal attention 天然兼容，无需架构改造即可扩展为 VLA

理解 PaliGemma 的设计选择 (为什么用 SigLIP 而非 CLIP、为什么 linear projection 而非 MLP、为什么 prefix-LM 而非 causal) 是理解 pi_0 等现代 VLA 模型设计动机的关键。
