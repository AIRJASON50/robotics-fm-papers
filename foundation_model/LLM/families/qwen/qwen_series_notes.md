# Qwen (通义千问) 系列 -- Alibaba 的开源 LLM 全家桶演进

> **阅读视角**: 本笔记的出发点是 **从 LLM 领域学习做机器人基础模型**。
> 关注: Qwen 的哪些技术路线和经验可以迁移到 robotics foundation model?

**覆盖论文** (主线):
- **Qwen**: "Qwen Technical Report", arXiv:2309.16609, 2023.09
- **Qwen2**: "Qwen2 Technical Report", arXiv:2407.10671, 2024.07
- **Qwen2-VL**: "Qwen2-VL: Enhancing VLM's Perception at Any Resolution", arXiv:2409.12191, 2024.09
- **Qwen2.5**: "Qwen2.5 Technical Report", arXiv:2412.15115, 2024.12
- **Qwen2.5-VL**: "Qwen2.5-VL Technical Report", arXiv:2502.13923, 2025.02
- **Qwen3**: "Qwen3 Technical Report", arXiv:2505.09388, 2025.05

**完整 arXiv 列表** (含专项):

| 模型 | arXiv | 年份 |
|------|-------|------|
| Qwen | 2309.16609 | 2023 |
| Qwen-VL | 2308.12966 | 2023 |
| Qwen-Audio | 2311.07919 | 2023 |
| Qwen2 | 2407.10671 | 2024 |
| Qwen2-Audio | 2407.10759 | 2024 |
| Qwen2-VL | 2409.12191 | 2024 |
| Qwen2.5-Math | 2409.12122 | 2024 |
| Qwen2.5-Coder | 2409.12186 | 2024 |
| Qwen2.5 | 2412.15115 | 2024 |
| Qwen2.5-VL | 2502.13923 | 2025 |
| Qwen2.5-Omni | 2503.20215 | 2025 |
| Qwen3 | 2505.09388 | 2025 |
| Qwen3-Omni | 2509.17765 | 2025 |
| Qwen3-VL | 2511.21631 | 2025 |
| Qwen3-Coder-Next | 2603.00729 | 2026 |

---

## 1. Alibaba / Qwen 与 LLM 发展的交织

### 1.1 Qwen 在全球 LLM 时间线中的位置

```
=== 全球 LLM 主线 ===

2017  Transformer (Google)
2018  GPT-1 (OpenAI) / BERT (Google)
2019  GPT-2 / XLNet (杨植麟)
2020  GPT-3 / Scaling Laws (Kaplan)
2022  InstructGPT -> ChatGPT (OpenAI) / Chinchilla (DeepMind)
2023.02  LLaMA (Meta) -- 开源大模型元年
2023.03  GPT-4 (OpenAI)
2023.07  Llama 2 (Meta, 开源商用)
2023.08  *** Qwen (Alibaba, 首次发布) ***  <- 中国开源大模型入场
2023.08  Qwen-VL (多模态扩展)
2024.02  Qwen1.5 (过渡版)
2024.04  Llama 3 (Meta)
2024.06  *** Qwen2 (架构升级: GQA + MoE) ***
2024.09  *** Qwen2.5 (18T tokens, 性能大跃进) ***
2024.09  o1 (OpenAI, 推理模型)
2025.01  DeepSeek-V3 / R1 / Kimi k1.5
2025.03  QwQ-32B (推理模型, 32B 匹配 DeepSeek-R1 671B)
2025.04  *** Qwen3 (36T tokens, thinking/non-thinking 统一) ***
2025.07  Kimi K2 / Qwen3-Coder
2026.02  *** Qwen3.5 (混合注意力, 201 种语言) ***
```

### 1.2 Qwen 的战略定位: 中国的 Llama

Qwen 的角色在中国 LLM 生态中类似 Meta 的 Llama 在全球生态中的角色:

| 维度 | Llama (Meta) | Qwen (Alibaba) |
|------|-------------|----------------|
| 定位 | 全球最广泛使用的开源 LLM | 中国最广泛使用的开源 LLM |
| 策略 | 开源换生态, 对抗 OpenAI 闭源垄断 | 开源换生态, 同时服务阿里云商业 |
| 许可证演进 | 自定义 -> Llama 3 许可 (准开放) | 自定义 -> Apache 2.0 (完全开放) |
| 被谁用作基座 | 大量微调/衍生模型 (Mistral, Vicuna...) | 大量中国模型用 Qwen 做 base (含 Kimi-Audio) |
| 模态覆盖 | 文本 + 视觉 | **文本 + 视觉 + 音频 + 代码 + 数学 + Omni** |

注意: **Qwen 在许可证上比 Llama 更开放** (Apache 2.0 vs Llama 自定义许可), 在模态覆盖上也更全面。这使得 Qwen 成为其他公司的首选基座模型 -- 比如 Moonshot 的 Kimi-Audio 就直接用 Qwen2.5-7B 做 backbone。

### 1.3 Qwen 的技术借鉴图谱

Qwen 系列是整个 LLM 社区技术积累的集大成者, 每一代都大量吸收最新进展:

| 借鉴的技术 | 来源 | 用在 Qwen 哪里 | 意义 |
|-----------|------|--------------|------|
| Transformer decoder-only | GPT-1 (OpenAI, 2018) | 所有 Qwen 模型 | 架构基础 |
| SwiGLU activation | PaLM (Google, 2022) | 所有 Qwen 模型 | 比 GELU 更好 |
| RMSNorm (pre-norm) | GPT-2/LLaMA | 所有 Qwen 模型 | 训练稳定性 |
| RoPE | Su et al. (2021) | 所有 Qwen 模型 | 相对位置编码 |
| GQA (Grouped Query Attention) | Llama 2 (Meta, 2023) | Qwen2+ | 推理效率 |
| MoE (Mixture of Experts) | GShard/Switch (Google) | Qwen2 MoE+ | 稀疏激活 |
| BPE tokenizer (tiktoken) | OpenAI | Qwen1 | 高效 tokenization |
| SFT + RLHF (PPO) | InstructGPT (OpenAI) | Qwen-Chat 系列 | Alignment |
| NTK-aware RoPE interpolation | 开源社区 (Reddit) | Qwen1 (32K) | 上下文扩展 |
| YaRN | 开源社区 | Qwen2+ (128K) | 长上下文 |
| DPO | Rafailov et al. (2023) | Qwen2.5+ | 比 PPO 更简单的 alignment |
| Flash Attention | Dao et al. (2022) | 所有版本 | 训练/推理加速 |

**Qwen 的原创贡献**:

| 创新 | 版本 | 意义 |
|------|------|------|
| **M-RoPE** (Multimodal RoPE) | Qwen2-VL | 将 RoPE 扩展为 3D (文本位置 + 图像高宽 + 视频时间) |
| **TMRoPE** (Time-aligned M-RoPE) | Qwen2.5-Omni | 音频-视频时间戳同步 |
| **Naive Dynamic Resolution** | Qwen2-VL | 不同分辨率图像映射到不同数量 tokens, 无需 resize |
| **Thinker-Talker** 架构 | Qwen2.5-Omni | Thinker 生成文本, Talker 流式合成语音, 解耦思考和表达 |
| **Thinking/Non-thinking 统一** | Qwen3 | 单一模型内切换深度推理和快速响应, thinking budget 控制 |
| **Self-improvement pipeline** | Qwen2.5-Math | 用上一代模型生成训练数据给下一代, 数据自举 |
| **QK-Norm** | Qwen3 | 替代 QKV-bias, 大模型训练更稳定 |
| 全面 Apache 2.0 | Qwen3 | 第一个将最大模型 (235B) 完全开放的主流 LLM 系列 |

---

## 2. 技术演进: 五个阶段

```
=== Phase 1: 入场 (2023) ===

Qwen (2023.08): 首次发布, 追赶 Llama
  |  1.8B ~ 72B, 3T tokens, RoPE + SwiGLU + RMSNorm
  |  同时发布 VL 和 Audio 多模态版本
  |  (意义: 中国首个完整开源 LLM 生态)

=== Phase 2: 架构升级 (2024 H1) ===

Qwen1.5 (2024.02): 过渡版, 统一 32K 上下文
  v
Qwen2 (2024.06): 关键架构升级
  |  全面引入 GQA (推理效率)
  |  首次引入 MoE (57B-A14B)
  |  RoPE base 从 10K -> 1M (长上下文基础)
  |  训练数据: 3T -> 7T
  |  支持 ~30 种语言

=== Phase 3: 规模飞跃 (2024 H2) ===

Qwen2.5 (2024.09): 数据量大跃进
  |  训练数据: 7T -> 18T (2.5x)
  |  100 万+ SFT 样本 + 多阶段 RL
  |  72B-Instruct 匹配 Llama-3-405B (参数量 1/5)
  |  同时发布 Math / Coder / VL 专项版本

Qwen2-VL (2024.09): 多模态创新
  |  M-RoPE: 3D 位置编码 (文本 + 图像 + 视频)
  |  Naive Dynamic Resolution: 不同分辨率 -> 不同 token 数
  |  72B 匹配 GPT-4o

=== Phase 4: 推理 + 全模态 (2025) ===

QwQ-32B (2025.03): 推理模型
  |  outcome-based RL, 32B 匹配 DeepSeek-R1 (671B)
  v
Qwen3 (2025.04): thinking/non-thinking 统一
  |  训练数据: 18T -> 36T
  |  119 种语言
  |  Dense (0.6B~32B) + MoE (30B-A3B, 235B-A22B)
  |  核心创新: 单一模型同时支持深度推理和快速响应
  |  四阶段后训练: CoT Cold Start -> RL -> Thinking Fusion -> General RL
  |  全部 Apache 2.0

Qwen2.5-Omni (2025.03): 全模态
  |  Thinker-Talker 架构
  |  文本/图像/音频/视频输入 -> 文本/语音输出
  |  TMRoPE 时间对齐

=== Phase 5: 效率革命 (2026) ===

Qwen3.5 (2026.02): 混合注意力
  |  线性注意力 (Gated Delta Networks) + 稀疏 MoE
  |  397B-A17B: 3.5x~7.2x 解码加速
  |  201 种语言
  |  原生多模态 (早期融合, 非后接)
```

---

## 3. 各版本核心技术要点

### 3.1 架构演进一览

| 特性 | Qwen (2023) | Qwen2 (2024) | Qwen2.5 (2024) | Qwen3 (2025) | Qwen3.5 (2026) |
|------|-------------|--------------|-----------------|--------------|----------------|
| 注意力 | MHA -> GQA | 全面 GQA | GQA | GQA + QK-Norm | **混合注意力** (线性+稀疏) |
| 位置编码 | RoPE | RoPE + QKV-bias | RoPE + QKV-bias | RoPE (去 QKV-bias) | RoPE + Gated Delta |
| MoE | 无 | 首次引入 | Turbo 系列 | 扩大规模, 去共享专家 | 稀疏 MoE + 线性注意力 |
| 训练数据 | **3T** | **7T** | **18T** | **36T** | 未公开 |
| 语言数 | ~中英 | ~30 | 29 | **119** | **201** |
| 上下文 | 8K/32K | 128K | 128K | 128K | 128K+ |
| 许可证 | 自定义 | 部分 Apache 2.0 | 多数 Apache 2.0 | **全 Apache 2.0** | Apache 2.0 |

### 3.2 训练数据策略: Qwen 的核心竞争力

Qwen 系列最显著的特征是 **数据规模的持续爆炸式增长**: 3T → 7T → 18T → 36T。

数据来源三路并行:
1. **网页爬取**: Common Crawl + 自有爬虫, 多层质量过滤
2. **PDF/文档提取**: 用 Qwen2.5-VL 从 PDF 中提取结构化文本 (self-improvement 闭环)
3. **合成数据**: 用上一代模型生成数学/代码训练数据给下一代 (bootstrapping)

这种 self-improvement 策略尤其值得注意:
```
Qwen2.5 (18T) 训练完成
  -> 用 Qwen2.5 生成大量数学/代码 synthetic data
  -> 加入 Qwen3 的 36T 训练集
  -> Qwen3 性能提升
  -> 再用 Qwen3 生成更好的 synthetic data...
```

### 3.3 M-RoPE: Qwen 最具影响力的原创技术

标准 RoPE 只编码 1D 序列位置。Qwen2-VL 将其扩展为 3D:

```
Text tokens:  RoPE(position_in_sequence)           -- 1D, 标准
Image tokens: RoPE(height, width)                  -- 2D, 空间位置
Video tokens: RoPE(time, height, width)             -- 3D, 时空位置
```

三个维度的 RoPE 通过拼接注入到 attention:
```
query/key embedding = concat(RoPE_temporal, RoPE_height, RoPE_width)
```

这使得模型能感知图像的空间结构和视频的时间结构, 而不是把所有 token 当作 1D 序列。后续 TMRoPE (Qwen2.5-Omni) 进一步加入音频时间戳对齐。

### 3.4 Qwen3 的 Thinking/Non-thinking 统一

这是 Qwen3 最核心的创新, 解决了一个实际问题: **推理模型 (如 o1, QwQ) 什么时候都很慢, 即使简单问题也要长篇大论地 "思考"**。

Qwen3 的方案: 在单一模型中同时支持两种模式:
- **Thinking mode**: 生成 `<think>...</think>` 内部推理链, 适合数学/代码/复杂推理
- **Non-thinking mode**: 直接回答, 适合简单查询/聊天

**四阶段后训练** 实现这一统一:

```
Stage 1: Long-CoT Cold Start
  用少量高质量 CoT 数据做 SFT, 给模型一个 "思考" 的起点

Stage 2: RL with Reasoning
  用 outcome-based reward 做 RL, 强化推理能力
  (类似 Kimi k1.5 的方法, 但更大规模)

Stage 3: Thinking Mode Fusion
  将 thinking 和 non-thinking 数据混合训练
  模型学会根据问题难度选择是否 "思考"

Stage 4: General RL
  在通用任务上做 RL, 确保日常对话能力不退化
```

**thinking budget**: 用户可以控制模型分配多少 "思考时间":
- budget = 0: 完全不思考, 最快
- budget = 高: 深度推理, 最准
- budget = 中: 平衡

### 3.5 MoE 演进

**Problem**: MoE (Mixture of Experts, 混合专家) 最初使用共享专家 (shared expert) 作为通用"兜底"层, 所有 token 都经过共享专家 + 若干路由专家。但共享专家会稀释路由专家的专精化。

**Insight**: Qwen3 去掉了共享专家, 改用 global-batch load balancing loss 促进专家专精化 -- 每个专家负责更窄的领域, 整体效率更高。

**Takeaway**: 去共享专家与 Kimi K2 (保留 1 个共享专家) 形成有趣对比, 说明 MoE routing 的最优设计仍未收敛, 是活跃的研究方向。从 Qwen2 到 Qwen3.5, 趋势是 **更多专家、更低激活比、更强专精化**。

---

## 4. 三大中国 LLM 路线对比: Qwen vs Kimi vs DeepSeek

| 维度 | Qwen (Alibaba) | Kimi (Moonshot) | DeepSeek |
|------|----------------|-----------------|----------|
| 创始人背景 | 阿里达摩院团队 | 杨植麟 (Transformer-XL) | 梁文锋 (量化基金) |
| 核心 DNA | **全家桶 + 数据规模** | **长上下文 + 效率** | **架构创新 + 低成本** |
| 架构路线 | 标准 Transformer → MoE | MLA + MoE (借鉴 DeepSeek) | **MLA + MoE (原创)** |
| 特色注意力 | M-RoPE (多模态位置编码) | MoBA (块稀疏注意力) | **MLA** (低秩 KV 压缩) |
| 优化器 | AdamW (标准) | **Muon** (2x 效率) | AdamW + FP8 训练 |
| 训练数据 | **36T tokens** (最大) | 15.5T (K2) | 14.8T (V3) |
| 开源策略 | **全 Apache 2.0** | Apache 2.0 (K2) | MIT |
| 模态覆盖 | 文本/视觉/音频/Omni/代码/数学 | 文本/视觉/音频 | 文本/视觉/代码 |
| 推理模型 | QwQ → Qwen3 thinking | Kimi k1.5 | **DeepSeek-R1** |
| 商业依托 | 阿里云 | Kimi Chat 产品 | API 服务 |
| 被引用情况 | 被 Kimi-Audio 用作基座 | -- | 被 Kimi K2/Moonlight 借鉴架构 |

**三者的互相影响**:

```
DeepSeek-V2 (2024.05): 发明 MLA + MoE
  |
  +---> Kimi K2 (2025.07): 直接采用 MLA + MoE
  +---> Qwen2 (2024.06): 采用 MoE (但用 GQA 而非 MLA)
  |
DeepSeek-R1 (2025.01): 开源推理模型
  |
  +---> QwQ-32B (2025.03): 同样的 outcome-based RL 思路
  +---> Kimi k1.5 (2025.01): 同时期独立发展, partial rollouts 是独特贡献
  |
Qwen2.5-7B (2024.09): 高质量开源基座
  |
  +---> Kimi-Audio (2025.04): 用 Qwen2.5-7B 做 audio LLM backbone
  +---> 大量社区微调模型
```

---

## 5. Qwen vs GPT: 路线对比

| 维度 | OpenAI (GPT) | Alibaba (Qwen) |
|------|-------------|----------------|
| 起步 | 2018 (GPT-1, 开创者) | 2023 (Qwen, 后发追赶) |
| Pre-training 创新 | 定义了范式 (autoregressive + scaling) | 跟随范式, 在数据规模上做到极致 |
| Post-training 创新 | **RLHF 的发明者** (InstructGPT) | 跟随 RLHF, 在 thinking 模式上有创新 |
| 多模态 | GPT-4V/4o (先发, 闭源) | M-RoPE, Omni (后发, 开源, 技术公开) |
| 推理模型 | o1 (先发, 完全未公开) | QwQ/Qwen3 (后发, thinking budget 有创新) |
| Scaling Laws | Kaplan (奠基) | 遵循, 用 36T tokens 验证 data scaling |
| 代码能力 | Codex → Copilot (先发) | Qwen3-Coder (后发, 480B MoE, 开源) |
| 开源程度 | GPT-2 → 闭源 → gpt-oss (2025) | 持续开源, **Apache 2.0** |
| 商业模式 | ChatGPT 订阅 + API | 阿里云集成 + 开源生态 |

**Qwen 对 GPT 路线的继承和发展**:

1. **Pre-training**: 完全继承 GPT 路线 (decoder-only, next-token prediction, scaling), 但在数据处理上有独到之处 (PDF 提取, self-improvement)
2. **Post-training**: 继承 InstructGPT 的 SFT+RLHF, 但在 Qwen3 发展出 thinking/non-thinking 统一方案
3. **多模态**: GPT-4V 证明了可行性, Qwen 在技术细节上更透明 (M-RoPE 论文公开)
4. **开源策略**: GPT 从开放走向封闭, Qwen 从封闭走向开放 -- 路线完全相反

---

## 6. 时间线与阅读建议

### 6.1 完整发布时间线

| 时间 | 发布 | 意义 |
|------|------|------|
| 2023.08 | Qwen + Qwen-VL | 首发, 中国开源 LLM 入场 |
| 2023.11 | Qwen-Audio | 音频模态扩展 |
| 2024.02 | Qwen1.5 | 过渡版, 统一 32K 上下文 |
| 2024.06 | **Qwen2** | **架构升级: GQA + MoE + 7T tokens** |
| 2024.09 | **Qwen2.5 + Qwen2-VL + Math + Coder** | **数据飞跃: 18T tokens, M-RoPE** |
| 2024.11 | QwQ-32B-Preview | 推理模型探索 |
| 2025.01 | Qwen2.5-1M | 1M token 上下文 |
| 2025.02 | Qwen2.5-VL | 强化视觉理解 |
| 2025.03 | QwQ-32B + Qwen2.5-Omni | 推理 + 全模态 |
| 2025.04 | **Qwen3** | **36T tokens, thinking/non-thinking 统一, 全 Apache 2.0** |
| 2025.07 | Qwen3-Coder | 480B MoE 代码模型 |
| 2025.09 | Qwen3-Omni | 全模态 MoE |
| 2025.11 | Qwen3-VL | 视觉 MoE |
| 2026.02 | **Qwen3.5** | **混合注意力 (线性+稀疏), 201 种语言** |
| 2026.03 | Qwen3-Coder-Next | 混合注意力代码模型 |

### 6.2 阅读建议

| 目标 | 推荐阅读 |
|------|---------|
| 理解 Qwen 基础 | Qwen1 report → Qwen2 report (架构升级) |
| 理解多模态 | Qwen2-VL paper (M-RoPE) → Qwen2.5-Omni (Thinker-Talker) |
| 理解推理模型 | QwQ blog → Qwen3 report Section 4 (四阶段后训练) |
| 理解 MoE | Qwen2 report → Qwen3 report (去共享专家) |
| 理解数据策略 | Qwen2.5 report (18T 数据处理 pipeline) |
| 与 GPT 对比 | GPT_series_notes.md (本库) → 本文 Section 5 |
| 与 Kimi/DeepSeek 对比 | kimi_series_notes.md (本库) → 本文 Section 4 |

---

## 7. Qwen 的商业逻辑 (参考 GPT 系列分析)

### 7.1 与 GPT 的商业逻辑对比

GPT 的路线: 学术突破 → Scaling Laws 论证 → 融资 → 产品 (ChatGPT) → 闭源壁垒

Qwen 的路线完全不同:

```
Phase 1 -- 跟随入场 (2023): 阿里内部孵化, 无需外部融资
  Qwen1: 追赶 Llama/GPT 的性能, 建立技术团队
  成本: 阿里云内部资源
  目标: 不被淘汰

Phase 2 -- 开源抢生态 (2024): 用开源换用户和影响力
  Qwen2/2.5: 密集发布, 覆盖所有模态/场景
  策略: 比 Llama 更开放 (Apache 2.0), 比 GPT 更全面 (全家桶)
  目标: 成为中国乃至全球的默认基座模型

Phase 3 -- 商业闭环 (2025+): 开源 + 阿里云 = 飞轮
  Qwen3: 开源最强模型吸引开发者
  开发者用 Qwen → 部署在阿里云 → 阿里云收入增长 → 投入更多资源 → 更好的 Qwen
  (类似 Meta: 开源 Llama → 开发者生态 → Meta 平台数据 → 更好的模型)
```

### 7.2 为什么 Qwen 能在资源远少于 OpenAI 的情况下追赶

1. **后发优势**: GPT 系列花了 5 年 (2018-2023) 探索出正确路线, Qwen 直接继承结论
2. **开源社区**: 站在 Llama/GPT-2/Flash Attention 等开源工作的肩膀上
3. **数据优势**: 中文互联网数据是独占资源, OpenAI 和 Meta 都不擅长
4. **Self-improvement**: 用自己的模型生成训练数据, 形成数据飞轮
5. **阿里云基础设施**: 不需要像 OpenAI 那样依赖 Microsoft, 自有 GPU 集群

### 7.3 Qwen 的风险

1. **创新跟随者**: 核心架构创新 (MLA, MoE routing) 主要来自 DeepSeek, Qwen 在基础研究上贡献较少
2. **数据天花板**: 36T tokens 已经接近公开可用数据的上限, 未来增长靠合成数据, 质量存疑
3. **计算成本**: 训练 36T tokens 的 235B MoE 模型需要巨大算力, 阿里的 GPU 供应受地缘政治影响
4. **生态碎片化**: 中国 LLM 市场高度碎片化 (Qwen, DeepSeek, Kimi, GLM, Baichuan...), 难以形成垄断

---

## 8. 对 Robotics Foundation Model 的影响与启示

### 8.1 Qwen 技术的直接影响

| Qwen 技术 | Robotics 应用 | 具体案例 |
|-----------|-------------|---------|
| **M-RoPE** (3D 位置编码) | 视觉-动作序列的时空位置编码 | Robot VLA 需要编码 image (2D) + trajectory (1D) 的混合位置信息, M-RoPE 的扩展思路直接适用 |
| **Qwen2.5-7B** (开源 base) | 机器人 VLM backbone | Kimi-Audio 用 Qwen2.5-7B 初始化; OpenVLA 用 Llama 初始化; **选开源 base model 做 robot VLM 是通用策略** |
| **Thinking/Non-thinking 统一** | 机器人快慢系统 | GR00T N1 的 VLM (10Hz, 慢思考) + DiT (120Hz, 快反应) 是硬件分离的双系统; Qwen3 的方案在单模型内切换, 更轻量 |
| **Self-improvement** 数据飞轮 | 机器人数据扩充 | 用已训练的 policy 生成 demo → 质量过滤 → 训练下一代 policy; Qwen 在 LLM 上验证了此策略的可行性 |
| **全模态 (Omni)** | 机器人多模态感知 | Thinker-Talker 架构 (思考和表达解耦) 可能影响未来机器人的 "决策模块 + 执行模块" 分离设计 |

### 8.2 从 LLM 学到的通用经验

**GPT 系列确立了 LLM 的技术原点, Qwen 展示了后发者如何高效追赶。** 对 robotics 的启示:

1. **架构不用发明, 跟随即可**: Qwen 从未发明过核心架构 (Transformer/MoE/RoPE 都是别人的), 但通过数据和工程追到了前沿。Robotics foundation model 同理 -- 不需要发明新架构, 用 Transformer + diffusion/flow matching 即可, 精力应放在 **数据和训练策略** 上。

2. **数据规模的 power law**: Qwen 从 3T → 7T → 18T → 36T, 每次翻倍都带来可测量的性能提升。Robotics 的 data scaling (25_RobotScalingLaws) 也遵循 power law, 但 exponent 更大 (同等资源下收益更高), 意味着 **当前 robotics 处于 scaling 曲线的早期, 增加数据的边际收益仍然很大**。

3. **Self-improvement 数据飞轮是关键**: Qwen 用自己的模型生成训练数据给下一代。这对 robotics 极其重要 -- **遥操作数据太贵, 但 simulation + learned policy 可以自动生成大量数据** (GR00T N1 的 DexMimicGen 就是这个思路)。

4. **开源 base model 是生态基础**: Qwen Apache 2.0 使它成为下游应用的默认选择 (Kimi-Audio, 各种中文微调模型)。Robotics 领域的 Octo/OpenVLA 走的是同样的路 -- 开源 base policy 让社区在上面构建应用。

### 8.3 Qwen3.5 的混合注意力: 对 Robotics 的潜在影响

Qwen3.5 的 Gated DeltaNet (线性注意力) + softmax (标准注意力) 混合架构, 推理加速 3.5-7.2x。这对 robotics 有直接意义:

- **机器人端计算受限**: 机载 GPU 通常是 Jetson 级别, 3.5-7.2x 加速可能意味着 policy 从 "只能在云端跑" 变成 "可以在机器人上实时跑"
- **线性注意力处理长 trajectory**: 标准 attention 是 O(n^2), 处理长 demo 很慢; 线性注意力是 O(n), 可能使 long-horizon task 的 policy 训练更高效
- **但需要验证**: 线性注意力在 NLP 中的 quality 损失是否在 robotics (连续控制) 中更严重或更轻微, 尚未研究

### 8.4 Takeaway 汇总

| # | Takeaway | 原理 | 对你的行动项 |
|---|----------|------|-------------|
| 1 | **M-RoPE 可直接迁移到 robot VLA** | 3D 位置编码 (空间+时间) 天然适配视觉-动作序列 | 在设计 robot VLA 时, 用 M-RoPE 编码 image (2D) + trajectory (1D), 不要把所有 token 压成 1D |
| 2 | **选开源 base model 初始化 robot VLM** | Qwen2.5-7B 被 Kimi-Audio 等大量下游使用, 证明好 base = 好初始化 | 做 robot VLM 时, 从 Qwen/Llama 等开源 base 初始化, 不要从头训练 |
| 3 | **Self-improvement 数据飞轮是关键** | Qwen 用上一代模型生成 synthetic data 给下一代, 36T 中大量是合成数据 | 用已训练 policy 在 sim 中生成 demo -> 过滤 -> 训练下一代, 替代昂贵的遥操作 |
| 4 | **架构跟随, 精力放在数据和训练策略** | Qwen 从未发明核心架构, 靠数据规模追到前沿 | 不要造新架构, 用 Transformer + Diffusion/Flow Matching, 精力投在数据 pipeline |
| 5 | **Thinking/Non-thinking 统一 = 机器人快慢系统** | 单一模型内切换深度推理和快速响应, 比硬件分离更轻量 | 探索单一 policy 内的 "thinking budget" 控制, 替代 VLM+DiT 双系统硬分离 |
| 6 | **混合注意力 (线性+稀疏) 解锁端侧部署** | Qwen3.5 的 3.5-7.2x 推理加速 | 关注线性注意力在连续控制中的 quality 损失, 这可能是 policy 从云端迁移到机载 GPU 的关键 |

---

## 9. Cross-References

本笔记与以下系列笔记互为参照:

| 系列 | 路径 | 与 Qwen 的关系 |
|------|------|---------------|
| GPT Series | `families/GPT_Series/GPT_series_notes.md` | Qwen 的技术原点; 架构/RLHF/Scaling Laws 全部继承自 GPT 路线 |
| DeepSeek Series | `families/deepseek/deepseek_series_notes.md` | MLA + MoE 架构创新者; Qwen 采用 MoE 但选择 GQA 而非 MLA |
| Kimi Series | `families/kimi/kimi_series_notes.md` | Kimi-Audio 用 Qwen2.5-7B 做 backbone; K2 保留共享专家 vs Qwen3 去掉共享专家 |
| Llama Series | `families/llama/llama_series_notes.md` | Qwen 在中国生态中的角色 = Llama 在全球生态中的角色; GQA 技术来源 |
