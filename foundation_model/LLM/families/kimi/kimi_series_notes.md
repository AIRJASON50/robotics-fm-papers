# Kimi 系列 -- Moonshot AI 的 LLM 全栈技术演进

> **阅读视角**: 本笔记的出发点是 **从 LLM 领域学习做机器人基础模型**。
> 关注的不是 Kimi 产品本身, 而是: LLM 领域的哪些技术分岔和经验可以迁移到 robotics foundation model?

**覆盖项目**:
- **Kimi k1.5**: "Scaling Reinforcement Learning with LLMs", arXiv:2501.12599, 2025.01
- **MoBA**: "Mixture of Block Attention for Long-Context LLMs", arXiv:2502.13189, 2025.02
- **Moonlight**: "Muon is Scalable for LLM Training", arXiv:2502.16982, 2025.02
- **Kimi-Audio**: "Kimi-Audio Technical Report", arXiv:2504.18425, 2025.04
- **Kimi K2**: "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534, 2025.07
- **Kimi K2.5**: "Kimi K2.5: Visual Agentic Intelligence", arXiv:2602.02276, 2026.01
- **Kimi K2.6**: 无独立 arxiv (基于 K2.5 架构), 2026.04.21 (preview 04.13), **300 sub-agent swarm × 4000 步 orchestration, SWE-Bench Pro 58.6 开源最佳**
- **Kimi CLI**: Terminal coding agent (工程项目, 无论文), 2025.10

---

## 1. Moonshot AI 与 LLM 发展的交织

### 1.1 创始人的学术基因

**杨植麟 (Yang Zhilin)**, 清华/CMU, 核心贡献:
- **Transformer-XL** (2019): 引入 recurrence 机制解决 Transformer 固定长度限制, 是长上下文 LLM 的思想起源
- **XLNet** (2019): 将 Transformer-XL 与 permutation language modeling 结合, 在 BERT 时代提出了 autoregressive 的替代路线

这两篇工作直接决定了 Moonshot AI 的技术 DNA: **长上下文是公司的基因**。Kimi Chat (2024.03) 是国内首个支持 200K+ 上下文的 AI 助手, 不是偶然而是创始人的学术积累的自然延伸。

### 1.2 Kimi 在 LLM 发展时间线中的位置

```
=== 全球 LLM 主线 ===

2017  Transformer (Google)
2018  GPT-1 (OpenAI) / BERT (Google)
2019  GPT-2 / Transformer-XL + XLNet (杨植麟) / T5
2020  GPT-3 / Scaling Laws (Kaplan)
2021  Codex / Chinchilla 前期
2022  InstructGPT -> ChatGPT (OpenAI) / Chinchilla (DeepMind)
2023  GPT-4 / Llama 1&2 (Meta) / Qwen (Alibaba) / Moonshot AI 成立
2024  Llama 3 / Qwen2 / DeepSeek-V2 (MLA+MoE) / Kimi Chat 200K 上线
                                                   ^^^^^^^^^^^^^^^
                                            Moonshot 的第一个公开产品

=== Kimi 技术线 (2025-2026) ===

2025.01  Kimi k1.5 -------- RL 推理 (对标 o1)
2025.02  MoBA ------------- 长上下文注意力 (生产部署)
2025.02  Moonlight --------- Muon 优化器 (2x 训练效率)
2025.04  Kimi-Audio -------- 音频模态 (7B)
2025.07  Kimi K2 ----------- 1T MoE 开源 (对标 Claude 4 / GPT-4.1)
2025.10  Kimi CLI ---------- 终端编码智能体 (对标 Claude Code)
2026.01  Kimi K2.5 --------- 多模态 agentic (对标 GPT-5.2)
2026.04  Kimi K2.6 --------- 300-agent swarm × 4000 步, SWE-Bench Pro 58.6 (preview→GA 8 天)
```

### 1.3 Kimi 的技术借鉴图谱

Kimi 不是从零开始, 而是站在整个 LLM 生态的肩膀上。以下是明确的技术借鉴关系:

| 借鉴的组件 | 来源 | 用在 Kimi 哪里 | 为什么借鉴 |
|-----------|------|--------------|-----------|
| MLA (Multi-head Latent Attention) | DeepSeek-V2/V3 | K2, K2.5 | 低秩压缩 KV cache, 长上下文显存瓶颈的最佳方案 |
| MoE 架构模式 | DeepSeek-V3 | K2, K2.5, Moonlight | 稀疏激活: 1T 总参但只激活 32B, 训练和推理成本可控 |
| DeepSeek-V3-small 架构 | DeepSeek-V3 | Moonlight | 直接复用验证过的架构, 专注优化器创新 |
| Qwen2.5 7B | Alibaba | Kimi-Audio | 已训好的文本 LLM 作为音频模型 backbone |
| SigLIP-SO-400M | Google | K2.5 MoonViT | 视觉编码器初始化, 不从零训练 |
| Whisper large-v3 | OpenAI | Kimi-Audio | 声学特征提取 (冻结或微调) |
| GLM-4-Voice tokenizer | Zhipu AI (智谱) | Kimi-Audio | 语义 token 编码器 |
| Muon 优化器 | KellerJordan (开源) | Moonlight -> K2 -> K2.5 | 2x 训练效率, 逐步改进为 MuonClip |
| NaViT packing | Google | K2.5 MoonViT | 原生分辨率视觉处理 |
| YaRN | 开源社区 | K2, K2.5 | 上下文长度从 8K 扩展到 128K/256K |
| Seed-VC framework | ByteDance | Kimi-Audio | 语音转换 |
| PPO/online mirror descent | RL 社区 | k1.5, K2, K2.5 | 策略优化基础 |

**Kimi 的原创贡献** (非借鉴, 是 Moonshot 自己的创新):

| 创新 | 项目 | 意义 |
|------|------|------|
| MoBA (块稀疏注意力) | MoBA | MoE 思想用于 attention, 1M 上下文 |
| MuonClip | K2 | Muon + weight decay + QK-Clip, 15.5T tokens 零 loss spike |
| Partial rollouts | k1.5 | 长上下文 RL (128K) 的关键效率技巧 |
| Long2Short | k1.5 | 长 CoT 知识蒸馏到短 CoT |
| Sparsity scaling law | K2 | 固定激活参数, 增加专家数持续降低 loss |
| Zero-vision SFT | K2.5 | 纯文本 SFT 即可激活视觉推理 (不需视觉 SFT 数据) |
| Agent Swarm / PARL | K2.5 | 多智能体并行执行, 延迟降低 4.5x |
| MoonViT-3D | K2.5 | NaViT 的 3D 扩展, 视频帧时间维度压缩 |

---

## 2. 技术演进: 四个阶段

### Phase 1: 基础设施 -- 长上下文 + 训练效率 (2025 Q1)

在做大模型之前, Moonshot 先解决两个基础问题: **长上下文推理太慢** 和 **训练太贵**。

#### 2.1 MoBA -- 将 MoE 思想用于 Attention

**问题**: 标准 attention 计算量是 O(n^2), 128K 上下文下推理极慢。

**核心思想**: MoE 中每个 token 只选几个 expert 计算; 同理, attention 中每个 query 也不需要关注所有 KV -- 只选最相关的几个 KV block 即可。

```
Standard attention:  每个 query 关注所有 n 个 KV tokens  -> O(n^2)
MoBA:               把 KV 分成 blocks, top-k 选最相关的  -> O(n * k * block_size)
```

**关键设计**:
- 门控是无参数的: query 与每个 block 的 mean-pooled key 做内积, top-k 选择
- 当前 block 必须关注 (类似 MoE 中的 shared expert)
- 因果性保证: 不关注未来 block
- 混合训练: 90% tokens 用 MoBA + 10% 用 full attention, loss 几乎不降

**与其他长上下文方案对比**:

| 方案 | 策略 | 灵活性 | 兼容性 |
|------|------|--------|--------|
| Sliding window | 只看最近 tokens | 固定 pattern | 好 |
| Sink attention | 开头 + 最近 | 固定 pattern | 好 |
| Mamba/RWKV/RetNet | 线性注意力/RNN | 全新架构 | **需从头训练** |
| **MoBA** | **模型自学关注哪些 blocks** | **动态, 数据驱动** | 需继续训练但不需从头训 |

**已部署在 Kimi 生产环境**处理长上下文请求。1M 上下文 Needle-in-a-Haystack 测试通过。

#### 2.2 Moonlight -- Muon 优化器 Scaling 验证

**问题**: AdamW 是 LLM 训练标准优化器, 但不一定是最优的。Muon (矩阵正交化优化器) 在小模型上有优势, 但没人证明能 scale 到大模型。

**核心贡献**: 识别两个 scaling 技巧:
1. **添加权重衰减** (标准 AdamW 做法)
2. **精心匹配每参数更新尺度** (Muon 的 update RMS 对齐到 AdamW 水平 ~0.2-0.4)

**结果**: Scaling law 实验证明 Muon 比 AdamW **~2x compute-efficient** -- 用同样算力, Muon 训练的模型等效于 AdamW 用 2x 算力的结果。

**Moonlight-16B-A3B** (DeepSeek-V3-small 架构, 5.7T tokens):

| Benchmark | Llama3.2-3B | Qwen2.5-3B | **Moonlight** |
|-----------|-------------|------------|---------------|
| MMLU | 54.75 | 65.6 | **70.0** |
| HumanEval | 28.0 | 42.1 | **48.1** |
| MATH | 8.5 | 42.6 | **45.3** |

**与 Scaling Laws 的关系**: Kaplan (2020) 和 Chinchilla (2022) 研究的是 "模型大小 vs 数据量" 的最优比例, 前提是优化器固定为 AdamW。Moonlight 开辟了新维度: **改进优化器本身也能获得等效的 scaling 提升**。2x 效率意味着在 Kaplan 的 power law 上, Muon 相当于把 compute 翻倍, 而实际硬件成本不变。

**技术传承**: Muon -> 在 K2 中进化为 **MuonClip** (加 QK-Clip 防止 attention logit 爆炸) -> K2.5 继续使用。

### Phase 2: 推理能力突破 (2025 Q1)

#### 2.3 Kimi k1.5 -- RL 驱动的 o1 级推理

**背景**: OpenAI o1 (2024.09) 展示了 "在推理时花更多 compute 来思考" 的路线, 但方法细节完全未公开。社区普遍认为 o1 依赖复杂的 MCTS + process reward model + value function。

**k1.5 的反论**: 这些复杂技术都不需要, 两个简单技巧就够:

1. **长上下文 RL**: 将 RL 的 context window 从常规 8K/32K 扩展到 **128K**。模型有更多空间 "思考", 自然涌现 planning/reflection/correction 能力。

2. **Partial rollouts**: 128K 上下文的 RL 采样极其昂贵。解法: 复用已有 trajectory 的前缀, 只重新生成后半段。

```
传统 RL:        [===== 完整生成 =====]  -- 每 episode 从头开始, 128K 太贵
Partial rollout: [旧前缀 (复用)] + [新后缀 (生成)]  -- 只付后缀的生成成本
```

**结果** (Short-CoT / Long-CoT):

| Benchmark | GPT-4o | Claude 3.5 Sonnet | o1 | **k1.5 Short** | **k1.5 Long** |
|-----------|--------|-------------------|----|----------------|---------------|
| AIME | 9.3 | 16.0 | 83.3 | 77.5 | 匹配 o1 |
| MATH-500 | 74.6 | 78.3 | 96.4 | 97.3 | 匹配 o1 |

**与 GPT 系列的对比**: InstructGPT 用 RLHF 解决 "alignment" (让模型听话); k1.5 用 RL 解决 "reasoning" (让模型思考)。两者都是 post-training 阶段的 RL, 但优化目标不同:
- InstructGPT: reward = 人类偏好 (主观)
- k1.5: reward = 答案正确性 (客观, 可验证)

### Phase 3: 多模态扩展 (2025 Q2)

#### 2.4 Kimi-Audio -- 音频基础模型

**架构** (借鉴多方, 整合为统一框架):

```
Audio input
  -> Audio Tokenizer:
     语义 tokens (VQ @ 12.5Hz, 来自 GLM-4-Voice)
     声学特征 (Whisper large-v3 encoder, 50Hz -> 12.5Hz via adapter)
  -> Audio LLM (7B, 从 Qwen2.5 7B 初始化):
     共享 Transformer 层处理 text + audio tokens
     并行头: text head (文本生成) + audio head (语义 token 生成)
  -> Audio Detokenizer:
     flow-matching (语义 token -> mel-spectrogram)
     BigVGAN vocoder (mel -> 24kHz waveform)
     chunk-wise streaming (低延迟)
```

**训练数据**: 1300 万小时原始音频, 经 speech enhancement (BSRNN) + diarization (PyAnnote) + transcription (Whisper/Paraformer) 处理。预训练: 585B audio tokens + 585B text tokens。

**与 GPT-4o 的对比**: GPT-4o (2024.05) 是原生 omni-modal (text + image + audio 统一 token), 但方法完全未公开。Kimi-Audio 是公开技术细节的音频基础模型, 代码和权重均开源。

### Phase 4: Agentic Intelligence (2025 Q3 - 2026)

#### 2.5 Kimi K2 -- 1T MoE 开源 Agentic 模型

这是 Moonshot 的旗舰模型, 汇集了前述所有基础设施创新:

| 组件 | 来源 | 说明 |
|------|------|------|
| 架构 | DeepSeek-V3 (MLA + MoE) | 1T 总参 / 32B 激活, 384 experts |
| 优化器 | Moonlight -> MuonClip | Muon + weight decay + QK-Clip |
| 长上下文 | YaRN + MoBA 经验 | 128K context |
| RL | k1.5 经验 | Agentic RL (tool-use reward) |

**K2 架构 vs DeepSeek-V3**:

| | DeepSeek-V3 | Kimi K2 |
|---|---|---|
| 总参数 | 671B | **1.04T** |
| 激活参数 | 37B | 32B |
| 专家数 | 256 | **384** |
| 注意力头 | 128 | **64** (减半, 降低长上下文推理成本) |
| 注意力 | MLA | MLA (同) |
| 训练 tokens | 14.8T | **15.5T** |
| 优化器 | AdamW | **MuonClip** |
| Loss spike | 有, 需手动 rollback | **零** (15.5T tokens 全程稳定) |

**MuonClip 解决了什么**: DeepSeek-V3 训练中出现过 loss spike, 需要手动回滚 checkpoint。K2 通过 QK-Clip (当 QK attention logit 超过阈值 tau=100 时, 按 per-head 缩放) 在 15.5T tokens 全程实现 **零 loss spike** -- 这是工程上的重大进步。

**Sparsity Scaling Law** (K2 的原创发现): 固定激活参数 (32B), 只增加总专家数 (从 8 到 384), loss 持续下降。Sparsity 48 (=384/8) 比 Sparsity 8 减少 1.69x FLOPs。这补充了 Kaplan/Chinchilla 的 scaling law: **除了模型大小和数据量, 稀疏度也是一个独立的 scaling 维度**。

**Agentic 能力**: K2 的 post-training 特别强调 tool-use:
- 20,000+ 合成工具 + 3,000+ 真实 MCP 工具
- 多 agent 轨迹生成 + LLM-based quality judge
- 结果: 在 agentic benchmarks 上对标 Claude 4 Opus/Sonnet 和 GPT-4.1

**开源策略**: Apache 2.0, 类似 Meta Llama。在 GPT-4 关闭技术细节 (2023.03) 后, 开源大模型成为与 OpenAI 竞争的核心策略 -- Llama, DeepSeek, Qwen, Kimi K2 都走了这条路。

#### 2.6 Kimi K2.5 -- 多模态 Agentic + Agent Swarm

在 K2 基础上增加视觉和多 agent 协作:

**MoonViT-3D**: 从 SigLIP-SO-400M (Google) 初始化, NaViT 原生分辨率处理, 3D 扩展支持视频 (4 帧分组, 时间维度 4x 压缩)。

**关键发现**:
1. **Zero-vision SFT**: 纯文本 SFT 即可激活视觉推理能力, 不需要专门的视觉 SFT 数据。添加人工设计的视觉轨迹反而损害泛化。
2. **视觉 RL 提升文本性能**: 视觉 RL 训练后, MMLU-Pro 84.7% -> 86.4%, GPQA-Diamond 84.3% -> 86.4% -- 跨模态 transfer。
3. **Agent Swarm**: 单 agent 顺序执行太慢。Agent Swarm 将复杂任务分解为并行子任务, 由动态实例化的领域专用 agent 执行。延迟降低最高 4.5x。

**PARL (Parallel Agent RL)**: 训练 orchestrator (可训练) + sub-agents (冻结)。Sub-agent 输出作为环境 observation 而非梯度传播信号。三种 reward: 实例化奖励 + 子 agent 完成率 + 任务级结果。

#### 2.7 Kimi CLI -- 产品落地

类似 Claude Code 的终端编码智能体, 支持 ACP (Agent Client Protocol) 和 MCP (Model Context Protocol)。是 K2/K2.5 agentic 能力的产品化形态。

#### 2.8 Kimi K2.6 -- 300-agent Swarm + Long-horizon Coding (2026.04.21)

**定位**: K2.5 架构骨架上的**能力扩展**, 不是新架构换代。

**关键新能力**:
- **Long-Horizon Coding**: 12 小时自主 coding session, 跨 Rust/Go/Python, 跨 front-end/DevOps/性能优化
- **300 个子 agent 并行 × 4000 步 orchestration** (PARL 的极致放大)
- **Preview → GA 仅 8 天** (2026.04.13 → 04.21), 行业最快迭代节奏
- 保留 K2 架构: MLA + MuonClip + INT4 native + 256K context + MoonViT 400M

**benchmark 亮点 (开源最佳)**:
- SWE-Bench Pro **58.6** (超过 GPT-5.4 的 57.7)
- AIME 2026: 96.4, GPQA-Diamond: 90.5
- BrowseComp w/ Agent Swarm: **86.3**

**意义**: K2 → K2.5 (架构+视觉) → K2.6 (能力+agent swarm), 证明一个好的 MoE + MLA + MuonClip base 可以持续做 post-training 扩展, 不需要每代都换架构。这条路径对机器人 FM 的启示: **稳定 VLA backbone 上做 skill-level fine-tune** 是可行演进路线。详见 `26_KimiK26/kimik26_notes.md`。

---

## 3. GPT 为起点的技术分岔与交织

GPT 是 LLM 的开创者。后续所有工作 (包括 Kimi) 都是从 GPT 定义的技术原点出发, 在不同方向上分岔。

### 3.1 技术分岔树: 从 GPT 到全行业

```
GPT-1/2 (2018-2019): 定义技术原点
  |  Transformer decoder + autoregressive + next-token prediction
  |  全行业从此继承, 至今未变
  |
  +=====================================================+
  |              分岔 1: 怎么做大 (Scale)                |
  +=====================================================+
  |
  +---> [GPT 路线] 暴力 Dense scale
  |     GPT-3 (175B, 2020) -> PaLM (540B, 2022)
  |     问题: Dense 模型推理太贵, 175B 每 token 都要跑全部参数
  |
  +---> [Google/Mistral] MoE 稀疏激活
  |     GShard (2020) -> Switch Transformer (2021) -> Mixtral (2023)
  |     idea: 不是每个 token 都需要全部参数, 只激活一部分专家
  |         |
  |         +---> [DeepSeek] MoE + MLA
  |         |     DeepSeek-V2 (2024): MLA 压缩 KV cache + MoE 稀疏激活
  |         |     = 同时解决 "推理显存大" 和 "推理计算大" 两个问题
  |         |         |
  |         |         +---> Kimi K2 (2025): 直接采用 MLA + MoE, 384 experts
  |         |         +---> Qwen2-MoE (2024): 采用 MoE 但用 GQA (不用 MLA)
  |         |
  |         +---> [Qwen3.5] 混合注意力 (线性 + softmax)
  |               Gated DeltaNet 替代部分 softmax attention -> 推理加速 3.5-7.2x
  |               (2026, 另一条效率路线: 不稀疏激活, 而是简化注意力本身)
  |
  +=====================================================+
  |          分岔 2: 怎么对齐 (Alignment)                |
  +=====================================================+
  |
  +---> [GPT 路线] RLHF for alignment
  |     RLHF-Summarize (2020) -> InstructGPT (2022) -> ChatGPT (2022)
  |     目标: 让模型听话 (helpful, harmless, honest)
  |     reward = 人类偏好 (主观)
  |
  +---> [o1/k1.5 路线] RL for reasoning
  |     o1 (2024, 闭源) / DeepSeek-R1 (2025) / Kimi k1.5 (2025) / QwQ (2025)
  |     目标: 让模型会思考 (数学, 代码, 逻辑推理)
  |     reward = 答案正确性 (客观, 可验证)
  |         |
  |         +---> k1.5 独特贡献: partial rollouts (128K RL 的效率关键)
  |         +---> Qwen3 独特贡献: thinking/non-thinking 统一 (同一模型两种模式)
  |         +---> R1 独特贡献: 大规模 RL 蒸馏 (大模型 RL -> 小模型 SFT)
  |
  +---> [DPO 路线] 去掉 RL, 简化对齐
        Rafailov (2023) -> Qwen2.5+/Llama 3 广泛采用
        直接从偏好数据训练, 不需要单独的 reward model
  |
  +=====================================================+
  |       分岔 3: 怎么做长上下文 (Long Context)          |
  +=====================================================+
  |
  +---> [位置编码扩展] RoPE interpolation -> NTK-aware -> YaRN
  |     最简单: 不改架构, 只改位置编码的频率基, 把 8K 扩展到 128K
  |     Qwen/Llama/K2 都用此方法
  |
  +---> [注意力稀疏化] 减少 attention 计算量
  |     Sliding window (Mistral) / Sink attention / ...
  |         |
  |         +---> MoBA (Kimi, 2025): 将 MoE routing 引入 attention
  |               最灵活: 模型自己学习该关注什么, 而非人工定义 pattern
  |
  +---> [线性注意力] 替换 softmax attention
        Mamba (2023) / RWKV / RetNet / Gated DeltaNet
            |
            +---> Qwen3.5 (2026): 混合架构 (75% 线性 + 25% softmax)
                  不完全替换, 而是混合使用
  |
  +=====================================================+
  |       分岔 4: 怎么做多模态 (Multimodal)              |
  +=====================================================+
  |
  +---> [后接 VLM] 文本 LLM + 视觉编码器拼接
  |     GPT-4V (2023) -> Qwen2-VL (M-RoPE) -> K2.5 (MoonViT-3D)
  |     文本模型先训好, 再接视觉编码器 fine-tune
  |
  +---> [原生 Omni] 预训练阶段就混合所有模态
  |     GPT-4o (2024) -> Qwen2.5-Omni (Thinker-Talker) -> K2.5
  |
  +---> [专项模态] 单独训练音频/代码/数学模型
        Codex (代码, 2021) / Kimi-Audio (音频, 2025) / Qwen2.5-Math (数学, 2024)
        在通用 base model 上用领域数据 fine-tune
  |
  +=====================================================+
  |       分岔 5: 训练效率 (Training Efficiency)         |
  +=====================================================+
  |
  +---> [数据效率] 用更少数据达到同等性能
  |     Chinchilla (2022): 纠正 Kaplan, 数据和模型应等比增长
  |     Qwen: self-improvement (用上一代生成下一代训练数据)
  |
  +---> [优化器效率] 用更少 compute 达到同等性能
  |     标准: AdamW (全行业默认)
  |         |
  |         +---> Moonlight/Muon (Kimi, 2025): ~2x 效率, 传承到 K2 (MuonClip)
  |               这是一个被低估的方向: 改优化器等效于免费翻倍 compute
  |
  +---> [计算精度] FP8/FP4 训练和推理
        DeepSeek-V3: FP8 训练
        gpt-oss: MXFP4 推理量化
```

*(Robotics 启示已独立为 Section 4)*

---

## 4. 对 Robotics Foundation Model 的启示

**为什么要看这些 LLM 分岔?** 因为 robotics 正在重走 LLM 的路:

| LLM 分岔 | Robotics 对应 | 当前状态 |
|----------|--------------|---------|
| Dense vs MoE | 单策略 vs 多专家策略 | GR00T N1 的 dual-system (VLM+DiT) 是雏形 |
| RLHF (alignment) | Reward shaping for robot policy | 机器人 reward 设计远比 LLM 难 (连续动作空间) |
| RL for reasoning | RL for motor planning | DeepMimic/PHC 已有, 但远未达到 o1/k1.5 的规模 |
| 长上下文 | 长 trajectory 处理 | 机器人 episode 通常很短, 还不需要 MoBA 级别方案 |
| 多模态融合 | 视觉+力觉+本体感觉融合 | pi_0/GR00T 在做, 但不如 LLM 成熟 |
| 优化器效率 | 更高效的 policy 训练 | **Muon 在 RL 中的应用尚未被探索** |
| Scaling Laws | Robot Scaling Laws | 25_RobotScalingLaws: 存在但数据不足以精确量化 |

### 4.1 Robotics Takeaway 总表

| # | Takeaway | 原理 | 对你的行动项 |
|---|---------|------|------------|
| 1 | **Muon 优化器: 2x 训练效率, 在 robot RL 中未验证** | Muon 通过矩阵正交化让梯度更新方向更优, 在 LLM pretraining 中实现 ~2x compute-efficiency (Moonlight)。但尚无人在 RL policy 训练中测试过。 | 低成本高回报实验: 在 IsaacLab 中将 PPO 的 AdamW 替换为 Muon, 对比 locomotion/manipulation 任务的 sample efficiency。只需改优化器, 不改架构。 |
| 2 | **MoBA: 长机器人视频的块稀疏注意力** | MoBA 将 KV 分块, 每个 query 只关注 top-k 最相关的块, 将 O(n^2) 注意力降为 O(n * k * block_size)。已部署在 Kimi 生产环境处理 1M 上下文。 | 机器人视频理解 (如 long-horizon task demonstration) 的 Transformer 处理瓶颈在 attention。当 VLA 模型需要处理长视频 episode (>1000 帧) 时, MoBA 是比 sliding window 更灵活的方案 -- 模型自学关注哪些帧, 而非人工定义固定窗口。 |
| 3 | **Partial Rollouts: 直接适用于长 horizon robot RL** | k1.5 的 128K 上下文 RL 通过复用已有 trajectory 前缀 + 只重新生成后缀来降低采样成本。 | 直接迁移到 long-horizon robot RL (如 mobile manipulation): 复用已有 episode 前半段 (navigation), 只重新采样后半段 (manipulation)。在 sim 中零成本实现, 可大幅减少长 episode 的 RL 采样开销。 |
| 4 | **Agent Swarm (PARL): 多机器人协调框架** | K2.5 将复杂任务分解为并行子任务, 由可训练 orchestrator 调度 + 冻结 sub-agents 执行, 延迟降低 4.5x。 | 多机器人协作场景 (如仓库多臂协同): 可训练一个 orchestrator policy 负责任务分解, 各 robot 运行冻结的专用 policy。PARL 的三种 reward (实例化 + 完成率 + 任务级) 可直接适配为多机器人 reward 设计模板。 |
| 5 | **Zero-vision SFT: 纯文本 SFT 激活视觉推理** | K2.5 发现纯文本 SFT 即可激活模型的视觉推理能力, 添加人工视觉轨迹反而损害泛化。 | 对 VLA 训练的暗示: 不需要大量视觉标注数据。先用文本描述操作步骤做 SFT (成本低), 再用少量视觉数据做 RL 微调。这可能大幅降低 robot VLA 的数据获取成本。 |
| 6 | **Sparsity Scaling Law: MoE 是 multi-task policy 的天然架构** | K2 证明固定激活参数 (32B), 只增加专家数 (8->384), loss 持续下降。稀疏度是独立于模型大小和数据量的第三个 scaling 维度。 | multi-task robot policy 不需要一个巨大的 dense policy, 可以用 MoE 让不同 expert 负责不同技能 (抓取/放置/旋转), 共享 expert 处理通用视觉+运动基元。参考 DeepSeekMoE 的 fine-grained segmentation。 |

### 4.2 最可迁移的经验 (详述)

1. **Muon 优化器 (Moonlight)**: 2x 训练效率, 目前只在 LLM 验证。如果 Muon 在 RL policy 训练中也有效, 等于免费翻倍 robot 训练 compute。值得实验。

2. **MoBA (MoBA paper)**: 机器人视频处理的 attention 瓶颈与 LLM 长上下文完全同构。当 VLA 模型接收长视频输入 (>1000 帧, 每帧多 patch), token 数轻松破万, MoBA 的块稀疏注意力可以大幅降低计算量且保留关键帧信息。

3. **Partial Rollouts (k1.5)**: 长 trajectory 的 RL 训练复用已有 trajectory 前缀。对机器人的长 horizon 任务 (如 mobile manipulation) 可能直接适用。

4. **Agent Swarm (K2.5)**: 多 agent 并行执行复杂任务。机器人领域的 multi-robot coordination 可以借鉴 PARL 框架 (可训练 orchestrator + 冻结 sub-agents)。

5. **Zero-vision SFT (K2.5)**: 纯文本 SFT 就能激活视觉推理。暗示: 机器人的 vision-language 能力可能不需要大量视觉标注数据, 文本描述 + 少量视觉数据就够。

6. **Sparsity Scaling Law (K2)**: 固定激活参数, 增加专家数持续降低 loss。这对 multi-task robot policy 有直接指导意义 -- 不需要一个巨大的 dense policy, 可以用 MoE 让不同 expert 负责不同技能。

7. **Self-improvement (Qwen)**: 用上一代模型生成下一代训练数据。Robot policy 也可以用 "trained policy 生成 demo -> filter -> 训练下一代 policy" 的方式扩充数据。

---

## 5. Kimi 内部的技术传承链

Kimi 的 7 个项目不是独立的, 而是有清晰的技术传承:

```
Moonlight (2025.02): Muon 优化器验证
  |
  | Muon -> MuonClip (加 QK-Clip)
  v
Kimi K2 (2025.07): 用 MuonClip 训练 15.5T tokens, 零 loss spike
  |
  | K2-Base 作为 foundation
  v
Kimi K2.5 (2026.01): 在 K2-Base 上持续预训练 ~15T 视觉-文本 tokens

---

MoBA (2025.02): 块稀疏注意力
  |
  | 部署到 Kimi 生产环境
  v
Kimi Chat 长上下文请求 / K2 训练和推理中的长上下文处理

---

Kimi k1.5 (2025.01): RL for reasoning 方法论
  |
  | RL 框架 + partial rollouts 经验
  v
K2 post-training: agentic RL (tool-use reward)
  |
  v
K2.5: visual RL + Agent Swarm (PARL)

---

Kimi-Audio (2025.04): 音频模态
  |
  | 文本数据来自 Moonlight
  | 独立产品线, 尚未整合到 K2/K2.5
  v
(Future: omni-modal Kimi?)

---

Kimi CLI (2025.10): 产品化
  |
  | 使用 K2/K2.5 作为后端模型
  v
开发者工具入口
```

---

## 6. 阅读建议

| 目标 | 推荐阅读顺序 |
|------|-----------|
| 理解 Kimi 架构 | K2 paper Section 2 (MLA+MoE) -> 对比 DeepSeek-V3 paper |
| 理解训练效率创新 | Moonlight paper (Muon) -> K2 paper Section 2.3 (MuonClip) |
| 理解长上下文 | MoBA paper (注意力) -> K2 paper (YaRN + MLA 的 KV 压缩) |
| 理解 RL for reasoning | k1.5 paper (partial rollouts) -> 对比 OpenAI o1 blog |
| 理解多模态 | Kimi-Audio paper -> K2.5 paper (MoonViT + zero-vision SFT) |
| 理解 agentic | K2 paper Section 3.2 (tool synthesis) -> K2.5 Section 5 (Agent Swarm) |
| 理解行业格局 | GPT_series_notes.md Section 9 (GPT 商业逻辑) -> 本文 Section 3 (交织借鉴) |
| Robotics 迁移 | 本文 Section 4 (Takeaway 总表) -> DeepSeek notes Section 4+9 (对比两家的 robotics 启示) |

---

## 7. 与其他家族笔记的关联

| 家族 | 笔记位置 | 与 Kimi 的关系 |
|------|---------|--------------|
| GPT | `../GPT_Series/GPT_series_notes.md` | 技术原点: Kimi 所有分岔都从 GPT 定义的 Transformer decoder + autoregressive 出发 |
| DeepSeek | `../deepseek/deepseek_series_notes.md` | 架构供给方: K2/K2.5 直接采用 MLA+MoE; Moonlight 用 V3-small 架构; GRPO 思路影响 k1.5 |
| Qwen | `../qwen/qwen_series_notes.md` | Base model 供给方: Kimi-Audio 用 Qwen2.5 7B 做 backbone; Qwen 的 self-improvement 可借鉴 |
| Llama | `../llama/llama_series_notes.md` | 开源策略参照: K2 走 Apache 2.0 开源路线, 与 Llama 竞争开源生态 |
| Google RT | `../../robotics/families/Google_RT_Series/RT_family_notes.md` | Kimi 的 MoE+RL 经验可迁移到 VLA; RT-2 式架构可用 MoBA 处理长视频 |
| PI | `../../robotics/families/pi_Series/pi_family_notes.md` | pi_0 的 Flow Matching policy + Kimi 的 Muon 优化器 = 潜在加速组合 |
| GR00T | `../../robotics/families/GR00T_Series/GR00T_family_notes.md` | GR00T N1 的 dual-system 是 MoE 的雏形; Agent Swarm 可指导多机器人协调 |
