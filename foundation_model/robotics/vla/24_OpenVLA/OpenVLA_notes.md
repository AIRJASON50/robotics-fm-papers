# OpenVLA: An Open-Source Vision-Language-Action Model -- 综合分析

Paper: Kim, Pertsch, Karamcheti et al., Stanford / UC Berkeley / TRI / Google DeepMind, 2024
Code: `openvla/` (PyTorch, 基于 Prismatic VLM codebase)

---

## 1. Core Problem

现有 VLA (Vision-Language-Action) 模型存在两个关键瓶颈阻碍了广泛应用:

1. **封闭性**: RT-2-X 等 SOTA VLA 模型不开源，模型架构、训练数据、训练流程均不透明，社区无法复现或改进
2. **缺乏 fine-tuning 方案**: 已有工作仅关注 out-of-the-box 泛化，没有探索如何在消费级 GPU 上高效适配 VLA 到新任务/新机器人

OpenVLA 的目标: 构建一个 **7B 参数的开源 VLA 基线模型**，在 970k 真实机器人 demonstration 上训练，提供从预训练到 fine-tuning 到部署的完整开源工具链，并在性能上超越 55B 参数的闭源 RT-2-X。

---

## 2. Method Overview

### 2.1 Architecture

OpenVLA 的架构直接继承自 Prismatic VLM，由三部分组成:

```
Image (224x224) --> [DINOv2 ViT-L + SigLIP ViT-SO400M] --> concat patches --> [2-layer MLP Projector] --> [Llama 2 7B] --> Action Tokens
                    ~600M params (vision)                                                                    ~7B params (LLM)
```

| Component | Model | Params | Role |
|-----------|-------|--------|------|
| Vision Encoder (alpha) | DINOv2 ViT-L/14 | ~300M | Spatial/low-level features |
| Vision Encoder (beta) | SigLIP ViT-SO400M/14 | ~400M | Semantic/high-level features |
| Projector | 2-layer MLP (GELU) | ~Few M | Vision->LLM embedding space mapping |
| LLM Backbone | Llama 2 7B | ~7B | Action sequence generation |

Fused vision backbone 的关键: DINOv2 提供 fine-grained spatial features (物体位置、几何)，SigLIP 提供 language-aligned semantic features (物体类别、语义匹配)。两个 encoder 分别处理输入图像，输出 patch embeddings 在 channel 维度拼接，再经 MLP projector 映射到 LLM 的 embedding space。

代码中 fused projector 使用 `FusedMLPProjector` (3-layer MLP: `Linear -> GELU -> Linear -> GELU -> Linear`)，注意这比标准 `MLPProjector` (2-layer) 多一层，这是因为 fused vision backbone 的 embedding 维度更大 (DINOv2 embed_dim + SigLIP embed_dim)。

### 2.2 Action Tokenization

核心设计 -- 将连续 robot action 离散化为 LLM vocabulary 中的 token:

1. **Normalization**: 每个 action 维度用训练数据的 1st/99th percentile (q01/q99) 归一化到 [-1, 1]
   - 使用 quantile 而非 min/max，避免 outlier action 破坏 bin resolution
   - 公式: `a_norm = 2 * (a - q01) / (q99 - q01) - 1`，然后 clip 到 [-1, 1]

2. **Discretization**: 每个归一化后的 action 维度均匀离散化为 **256 bins**
   - `np.linspace(-1, 1, 256)` 创建 bin boundaries
   - `np.digitize(action, bins)` 获取 bin index

3. **Token Mapping**: 将 bin index 映射到 Llama tokenizer vocabulary 的 **最后 256 个 token** (least-used tokens)
   - Token ID = `vocab_size - bin_index`
   - 不增加新 token，而是覆盖 tokenizer 中最低频的 token

4. **Decoding**: inference 时从 token ID 反向映射回 bin center，再 un-normalize
   - `a_continuous = bin_centers[vocab_size - token_id - 1]`
   - `a_real = 0.5 * (a_norm + 1) * (q99 - q01) + q01`

对于 7-DoF action (6D end-effector delta + gripper)，模型 autoregressively 生成 7 个 action token。

### 2.3 Training

- **Loss**: Standard next-token prediction (cross-entropy)，**只对 action tokens 计算 loss** (其他位置的 label 为 IGNORE_INDEX = -100)
- **数据**: Open X-Embodiment 数据集中筛选的 970k trajectories，含多种机器人形态
- **训练配置**: 64 A100 GPUs, batch size 2048, lr=2e-5 (constant), 27 epochs, ~14 days (21,500 A100-hours)
- **Vision encoder**: 训练中 **unfrozen** (与 VLM 预训练相反，论文发现 VLA 训练中冻结 vision encoder 效果差)

### 2.4 Prompt Template

不同版本使用不同 prompt format:

- **v01** (Vicuna-style): `"A chat between a curious user... USER: What action should the robot take to {instruction}? ASSISTANT:"`
- **v0.2+** (Pure prompt): `"In: What action should the robot take to {instruction}?\nOut:"`

### 2.5 Inference Pipeline

```
Image + Instruction
  --> Vision Encoder (DinoV2 + SigLIP) --> patch features
  --> MLP Projector --> projected embeddings
  --> Prepend to tokenized instruction
  --> Llama 2 autoregressive generate (max_new_tokens = action_dim)
  --> Decode action tokens --> un-normalize --> continuous action
```

Inference speed: ~6 Hz on RTX 4090 (bfloat16)

---

## 3. Key Designs

### 3.1 Fused DINOv2 + SigLIP Vision Backbone (最重要的架构选择)

OpenVLA 选择 Prismatic VLM 作为 backbone 的核心原因是其 dual vision encoder 设计:

- **DINOv2**: 自监督训练，擅长 spatial reasoning、物体边界检测、几何关系理解
- **SigLIP**: 对比学习 (image-text)，擅长 semantic understanding、物体识别、语言-视觉对齐

这对机器人控制意义重大: 精确操控需要 fine-grained spatial information (抓取点位置、物体朝向)，同时语言指令遵循需要 semantic understanding (区分"红色杯子"和"蓝色杯子")。

Ablation 证据:
- Prismatic (DinoSigLIP) 比 LLaVA (CLIP-only) 高 ~10% absolute success rate
- LLaVA 比 IDEFICS-1 在 language grounding 任务高 35%
- 在 224px vs 384px 分辨率对比中，**未发现分辨率提升带来收益** (与 VLM 文献相反)，因此选择 224px 降低计算成本

### 3.2 Q99 Action Normalization + Least-Used Token Mapping

两个看似简单但关键的工程决策:

**Q99 Normalization**: 使用 1st/99th percentile 而非 min/max 来定义 action 范围。在真实机器人数据中，action distribution 通常有 long tail (如碰撞导致的极端 action)，min/max 会被 outlier 拉扯，导致大部分 action 集中在少数 bin 中，有效分辨率大幅下降。

**Least-Used Token Mapping**: Llama tokenizer 只保留 100 个 special token 位置，不够放 256 个 action token。解决方案: 覆盖 vocabulary 末尾 256 个最低频 token (主要是罕见 Unicode 字符)。这避免了修改 tokenizer 或增加 vocabulary size 的复杂性。

代码实现 (`action_tokenizer.py`):
```python
# Token mapping: vocab_size - bin_index
self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))
```

### 3.3 LoRA Fine-Tuning 方案 (实用性贡献)

OpenVLA 首次系统性探索了 VLA 的 parameter-efficient fine-tuning:

| Fine-Tuning Method | Success Rate | Trainable Params | GPU Memory |
|-------------------|-------------|-----------------|------------|
| Full fine-tuning | Best | 100% | 8x A100 |
| Last layer only | Poor | ~0.1% | Low |
| Frozen vision | Poor | ~93% | Medium |
| Sandwich (vision + last layer) | Medium | ~7% | Medium |
| **LoRA (r=32, all-linear)** | **Match full** | **1.4%** | **1x A100** |

LoRA 配置细节 (from `finetune.py`):
- `target_modules="all-linear"` -- 应用于所有 linear layer (包括 vision encoder + projector + LLM)
- `lora_alpha = min(r, 16)` -- alpha 不超过 16
- `init_lora_weights="gaussian"` -- Gaussian 初始化
- 10-15 小时 on 单张 A100 即可完成 fine-tuning

---

## 4. Experiments

### 4.1 Out-of-the-Box Evaluation

| Model | Params | BridgeV2 (17 tasks) | Google Robot (12 tasks) |
|-------|--------|--------------------|-----------------------|
| RT-1-X | 35M | Low | Low |
| Octo | 93M | Low | Low |
| RT-2-X | 55B | Moderate | Moderate |
| **OpenVLA** | **7B** | **Highest** (+16.5% vs RT-2-X) | **Comparable** to RT-2-X |

关键发现:
- OpenVLA 以 **7x 更少参数** 在 BridgeV2 上超越 RT-2-X 16.5% absolute success rate
- RT-2-X 仅在 semantic generalization 上优于 OpenVLA (预期内，因为 RT-2-X co-fine-tune 保留更多 web knowledge)
- RT-1-X 和 Octo 在 language grounding 任务上严重不足，有时机器人"手臂乱挥"

### 4.2 Fine-Tuning to New Robots

在 Franka robot 上 7 个任务的 fine-tuning 对比:

| Method | Single-Task | Multi-Task (Language Grounding) | Overall |
|--------|-------------|-------------------------------|---------|
| Diffusion Policy | Strong | Weak | Moderate |
| Diffusion Policy (matched) | Medium | Weak | Lower |
| Octo (fine-tuned) | Weak | Medium | Lower |
| **OpenVLA (fine-tuned)** | **Good** | **Strong** | **Highest** |
| OpenVLA (scratch) | Lower | Lower | Lower than OpenVLA |

关键发现:
- **Diffusion Policy 在窄任务上更强** (单指令、高精度)，但在需要 language grounding 的多物体场景中失败
- **OpenVLA 是唯一在所有测试任务上 >50% success rate 的方法**
- OpenX pretraining 的价值: OpenVLA (fine-tuned) >> OpenVLA (scratch)，证明大规模 robot 预训练有效

### 4.3 Quantization

| Precision | BridgeV2 Success | VRAM |
|-----------|-----------------|------|
| bfloat16 | 71.3% | 16.8 GB |
| int8 | 58.1% (degraded) | 10.2 GB |
| **int4** | **71.9%** | **7.0 GB** |

int8 性能下降的原因不是精度损失，而是 inference 速度变慢 (A5000 上仅 1.2Hz)，导致与 5Hz 数据采集频率不匹配。int4 在更少内存下实现接近 bf16 的 throughput。

---

## 5. Related Work Analysis

### 5.1 VLA 发展脉络

```
RT-1 (2022)              -- Transformer policy from scratch, 专用 action head
  |
RT-2 / RT-2-X (2023)     -- VLM co-fine-tuning, action-as-language-token, 闭源
  |
Octo (2023)              -- 小模型 (93M), 开源, 支持异构输入, 但容量不足
  |
OpenVLA (2024)           -- 7B 开源 VLA, 强 VLM backbone, LoRA fine-tuning
  |
pi_0 (2024)              -- Flow matching, action chunking, 灵巧操作
```

### 5.2 OpenVLA 的独特定位

OpenVLA 不追求方法论上的突破，而是将已有技术 (VLM + action tokenization) 组合成一个**高质量开源基线**:

1. **开源生态**: 模型权重 + 训练代码 + fine-tuning notebook + HuggingFace 集成
2. **更强的 VLM backbone**: Prismatic (DINOv2 + SigLIP) vs RT-2 的 PaLI-X
3. **更大更干净的数据**: 970k trajectories (vs RT-2-X 的 350k)，更细致的数据清洗
4. **首次系统性探索 VLA fine-tuning**: full/LoRA/quantized 各种方案

---

## 6. Limitations & Future Directions

### 6.1 论文明确指出的局限

| Limitation | Description |
|-----------|-------------|
| 单图输入 | 不支持多相机、observation history、proprioception |
| 低频控制 | ~6 Hz, 无法满足 ALOHA (50Hz) 等高频灵巧操作需求 |
| 无 action chunking | 每次只预测单步 action, 轨迹不平滑 |
| 可靠性不足 | 最好的任务也通常 <90% success rate |
| 设计空间未充分探索 | VLM 规模、co-training、视觉特征选择等问题尚未回答 |

### 6.2 从代码推断的局限

1. **硬编码 Llama tokenizer**: `predict_action()` 中硬编码检查 `LlamaTokenizerFast` 和 magic token id `29871` (Llama empty token)，不支持其他 LLM backbone
2. **固定 256 bins**: bin 数量写死在代码中，没有 adaptive binning 机制
3. **RLDS 数据格式强依赖**: 必须将数据转换为 RLDS/TFDS 格式，对新数据集不友好 (虽然提供了 `DummyDataset` 模板)
4. **单 action space**: 不支持 multi-modal action distribution (离散化天然是 unimodal)
5. **Gradient accumulation 不支持 VLA 训练**: 代码中 `assert self.grad_accumulation_steps == 1`

---

## 7. Paper vs Code Discrepancies

### 7.1 论文未提及但代码实现的

| Item | Detail | Code Location |
|------|--------|--------------|
| FusedMLPProjector 是 3 层 MLP | 论文说 "2-layer MLP projector"，但 fused backbone 使用的是 `FusedMLPProjector` (3-layer: `[vision_dim*4, llm_dim, llm_dim]`) | `prismatic/util/nn_utils.py` |
| Image augmentation | Fine-tuning 时默认开启 (`image_aug=True`)，包含 random resized crop, brightness, contrast, saturation, hue | `prismatic/vla/datasets/datasets.py` L122-136 |
| 第二到最后一层 features | Vision encoder 不取最后一层 patch features，而是取 **倒数第二层** (`n={len(blocks) - 2}`) | `prismatic/models/backbones/vision/dinosiglip_vit.py` L63-68 |
| Gripper action binarization | 代码中对 gripper action 做了 binarization 处理 (连续值转为 0/1)，论文未提及 | `prismatic/vla/datasets/rlds/utils/data_utils.py` L106+ |
| Per-dataset metric tracking | 训练中分别追踪每个数据集的 action accuracy 和 L1 loss，用于诊断哪些数据集学得好/差 | `prismatic/training/strategies/base_strategy.py` L335-340 |
| DROID 数据集中途移除 | 论文简要提到，但代码配置中明确体现: `oxe_magic_soup_plus` (含 DROID) -> `oxe_magic_soup_plus_minus` (不含 DROID)，在训练 70% 处切换 | `prismatic/conf/vla.py` L131-132 |
| Unused action dimension 置零 | 对于 min==max 的 action 维度 (即该机器人不使用的自由度)，代码显式将其归一化结果置为 0 | `prismatic/vla/datasets/rlds/utils/data_utils.py` L96-99 |
| Action mask in un-normalization | Inference 时 un-normalize 使用 `mask` 跳过不需要 un-normalize 的维度 | `prismatic/models/vlas/openvla.py` L95-101 |
| Deploy server | 提供完整的 FastAPI inference server，支持远程调用 | `vla-scripts/deploy.py` |
| Multiple freeze strategies | 代码实现了 5 种不同的 weight freezing 策略 (align / finetune / full-finetune / last-layer / sandwich)，论文只讨论了部分 | `prismatic/models/vlms/prismatic.py` L129-234 |
| LoRA merge during training | LoRA fine-tuning 中，每次保存 checkpoint 时都会执行 LoRA weight merge (加载 base model + merge adapter)，这非常耗时 | `vla-scripts/finetune.py` L337-361 |

### 7.2 Prompt format version 差异

论文没有详细区分 v01 和后续版本的 prompt 格式差异。代码中:
- v01 使用 `VicunaV15ChatPromptBuilder` (Vicuna chat format)
- v0.2+ 使用 `PurePromptBuilder` (简洁的 `"In: ... \nOut:"` 格式)

Deploy 代码硬编码了这个逻辑:
```python
if "v01" in openvla_path:
    return f"{SYSTEM_PROMPT} USER: ... ASSISTANT:"
else:
    return f"In: ... \nOut:"
```

---

## 8. Cross-Paper Comparison

### 8.1 OpenVLA vs RT-2 vs pi_0 vs Octo

| Dimension | OpenVLA | RT-2 / RT-2-X | pi_0 | Octo |
|-----------|---------|---------------|------|------|
| Params | 7B | 5B / 55B | 3.3B | 93M |
| Open-source | Yes | No | No | Yes |
| VLM Backbone | Prismatic (DINOv2+SigLIP+Llama2) | PaLI-X / PaLM-E | PaliGemma (SigLIP+Gemma) | N/A (from scratch) |
| Action Representation | Discrete tokens (256 bins) | Discrete tokens (256 bins) | **Continuous (flow matching)** | Continuous (diffusion) |
| Action Chunking | No (single step) | No (single step) | **Yes (H=50)** | Yes |
| Training Data | 970k trajs (OpenX) | 130k-350k trajs | Cross-embodiment + DROID | 800k trajs (OpenX) |
| Control Frequency | ~6 Hz | 1-5 Hz | **50 Hz** | ~10 Hz |
| Multi-modal Action | No | No | **Yes** | Yes |
| Observation | Single image | Single image | Multi-image + proprio | Multi-image + proprio |
| Fine-tuning | LoRA (1.4% params) | Not supported | Task-specific fine-tuning | Flexible fine-tuning |
| Dexterity | Low-moderate | Low-moderate | **High (cloth folding etc.)** | Moderate |

### 8.2 Discrete Action Tokens (OpenVLA/RT-2) vs Continuous Flow Matching (pi_0) -- Tradeoff 分析

这是 VLA 领域最关键的设计分歧之一:

**Discrete Action Tokens 的优势:**

| Advantage | Explanation |
|-----------|-------------|
| 架构统一性 | Actions 直接复用 LLM tokenizer/vocabulary，无需额外 action head |
| Web knowledge transfer | 所有 weights 在 language/action 任务间共享，VLM 预训练知识可直接影响 action 生成 |
| 生态兼容性 | 可直接利用 LLM 训练基础设施 (FSDP, FlashAttention, quantization, LoRA) |
| Language grounding | 因为 action 和 language 在同一空间，模型天然更擅长理解语言指令与动作的对应关系 |

**Discrete Action Tokens 的劣势:**

| Disadvantage | Explanation |
|-------------|-------------|
| 精度上限 | 256 bins 的分辨率约为 1/128 的 action range (~0.008), 对精细操作可能不够 |
| Unimodal limitation | 离散化后的 autoregressive generation 本质上预测单峰分布，无法表达多模态 action distribution (如"左绕"或"右绕"都可行) |
| 无 action chunking | 自回归逐 token 生成，难以有效实现 action chunking, inference 速度受限 |
| 非平滑轨迹 | 每步独立预测，缺乏时间一致性，轨迹可能抖动 |

**Flow Matching (pi_0) 的优势:**

| Advantage | Explanation |
|-----------|-------------|
| 连续动作空间 | 天然支持高精度连续 action，无量化误差 |
| 多模态分布 | Flow matching 可以学习任意复杂的 action distribution，处理多模态性 |
| Action chunking | 一次预测 50 步 action chunk, 实现 50 Hz 高频控制 |
| 平滑轨迹 | Chunk-based prediction 天然产生时间一致的平滑轨迹 |

**Flow Matching 的劣势:**

| Disadvantage | Explanation |
|-------------|-------------|
| 额外参数 | 需要独立的 Action Expert (300M params)，不能完全复用 LLM backbone |
| 训练复杂度 | Flow matching loss + 噪声调度 + 多步 denoising, 训练配置更复杂 |
| Inference 成本 | 每步 action 需要 10 步 Euler integration, 计算量更高 (虽然可以 amortize over chunk) |
| 生态兼容性稍差 | 不能直接用 LLM 标准 fine-tuning 工具 (如 LoRA for language head)，需要定制化 |

### 8.3 什么时候选哪种方案

| Scenario | Recommended Approach | Reason |
|----------|---------------------|--------|
| 通用 manipulation baseline | OpenVLA | 开源、易用、LoRA fine-tuning 门槛低 |
| 需要强 language grounding | OpenVLA / RT-2 | Action-as-token 天然利用 VLM 语义知识 |
| 高频灵巧操作 (叠衣服、倒水) | pi_0 | Action chunking + flow matching 支持 50Hz |
| Multi-modal action distribution | pi_0 / Octo (diffusion) | 离散化方案天然无法处理多峰 |
| 计算受限 (消费级 GPU) | OpenVLA (LoRA + int4) | 7B model + 4-bit quantization 仅需 7GB VRAM |
| 异构 sensor 输入 | Octo / pi_0 | OpenVLA/RT-2 仅支持单图 |
| 快速原型验证 | OpenVLA | HuggingFace AutoClass 集成，几行代码即可运行 |

### 8.4 趋势判断

从 RT-2 -> OpenVLA -> pi_0 的发展可以看出:
1. **VLM backbone 越来越重要**: 从 Octo 的 from-scratch 到 VLA 的 pretrained VLM，性能差距巨大
2. **Action representation 是核心分歧**: discrete token vs continuous flow matching 各有适用场景，尚未统一
3. **开源推动进步**: OpenVLA 的最大贡献不是方法论创新，而是提供了一个 reproducible 的强基线
4. **Fine-tuning > pretraining**: 对于具体部署，fine-tuning 能力比 out-of-box 泛化更重要
5. **pi_0 的 flow matching 方向可能是下一代主流**: 同时解决了精度、多模态、action chunking 问题，但需要更多开源工作来验证
