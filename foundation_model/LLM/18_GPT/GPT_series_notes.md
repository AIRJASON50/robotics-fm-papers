# GPT 系列 -- 从 Generative Pre-Training 到 GPT-4 的大语言模型演进

**覆盖论文**:
- **GPT-1**: Radford et al., "Improving Language Understanding by Generative Pre-Training", OpenAI, 2018
- **GPT-2**: Radford et al., "Language Models are Unsupervised Multitask Learners", OpenAI, 2019
- **GPT-3**: Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020, arXiv:2005.14165
- **GPT-4**: OpenAI, "GPT-4 Technical Report", arXiv:2303.08774, 2023

**代码仓库**: [openai/gpt-2](https://github.com/openai/gpt-2) (GPT-2, TensorFlow, archived)

---

## 1. Core Problem

### GPT 系列要解决的根本问题

NLP 领域长期面临一个核心矛盾: **人类可以从极少的示例中学会新的语言任务, 但 NLP 系统需要大量标注数据和 task-specific fine-tuning**。GPT 系列论文逐步攻克这一问题:

| 论文 | 核心问题 | 提出的解法 |
|------|---------|---------|
| GPT-1 (2018) | 有标注数据稀缺, task-specific 架构难以迁移 | Generative pre-training + discriminative fine-tuning |
| GPT-2 (2019) | Fine-tuning 仍需 task-specific 数据 | 更大模型 + 更多数据 = unsupervised multitask learning (zero-shot) |
| GPT-3 (2020) | Zero-shot 能力仍不够强 | 175B 参数 + in-context learning (few-shot, 不更新权重) |
| GPT-4 (2023) | 纯文本 LLM 的能力天花板 | Multimodal (文本+图像) + RLHF alignment + predictable scaling |

### 四篇论文的递进逻辑

```
GPT-1: "pre-train + fine-tune" 范式的确立
  |  (insight: 去掉 fine-tuning 也能工作)
  v
GPT-2: "pre-train only" -- zero-shot task transfer
  |  (insight: 放大模型和数据, zero-shot 持续提升)
  v
GPT-3: "in-context learning" -- few-shot without weight updates
  |  (insight: 需要对齐人类意图, 不仅仅是 next-token prediction)
  v
GPT-4: multimodal + RLHF alignment + predictable scaling
```

---

## 2. Method Overview

### 2.1 GPT-1: Pre-train + Fine-tune 范式

**架构**: 12-layer Transformer decoder, 768 dim, 12 heads (~117M params)

**两阶段训练**:

Stage 1 -- Unsupervised pre-training (BooksCorpus, ~7000 books):
```
L_1(U) = sum_i log P(u_i | u_{i-k}, ..., u_{i-1}; Theta)
```
标准 autoregressive language modeling objective。

Stage 2 -- Supervised fine-tuning (task-specific labeled data):
```
L_2(C) = sum_{(x,y)} log P(y | x^1, ..., x^m)
L_3(C) = L_2(C) + lambda * L_1(C)    // auxiliary LM objective
```
关键创新: **task-specific input transformations** -- 将不同任务 (分类、蕴含、相似度、问答) 的输入统一转化为 token 序列, 通过 delimiter token 分隔, 使同一个 pre-trained model 可以处理所有任务, 无需修改架构。

**关键发现**:
- 在 12 个任务中的 9 个达到 SOTA
- 每一层 Transformer 都包含有用的功能 (transfer 更多层 = 更好性能)
- Transformer >> LSTM (差 5.6 分), 证明 attention 对 long-range transfer 至关重要
- 去掉 pre-training 会导致 14.8% 性能下降

### 2.2 GPT-2: Unsupervised Multitask Learning

**架构**: 与 GPT-1 相同的 Transformer decoder, 但放大到 4 个尺寸

| 模型 | 参数量 | 层数 | d_model |
|------|--------|------|---------|
| GPT-2 Small | 124M | 12 | 768 |
| GPT-2 Medium | 355M | 24 | 1024 |
| GPT-2 Large | 774M | 36 | 1280 |
| GPT-2 XL | 1.5B | 48 | 1600 |

**关键改进** (相对 GPT-1):
- Layer normalization 移到每个 sub-block 的输入前 (pre-norm)
- 增加一个 layer norm 在最后的 self-attention block 之后
- 残差连接的初始化按 1/sqrt(N) 缩放 (N = residual layers)
- Context length: 512 -> 1024 tokens
- 训练数据: BooksCorpus -> WebText (40GB, 8M web pages, Reddit outbound links with >= 3 karma)

**核心思想**: `P(output | input, task)` -- 语言模型本质上是所有 NLP 任务的 unsupervised multitask learner。不需要显式任务定义, 只需要足够大的模型和足够多的数据。

**代码仓库分析** (`gpt-2/src/model.py`):
- 架构极简: 整个模型 175 行 TensorFlow 代码
- Pre-norm 结构: `block()` 函数先 `norm(x, 'ln_1')` 再 `attn()`, 先 `norm(x, 'ln_2')` 再 `mlp()`
- GELU activation: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`
- KV cache 支持: `past` 参数实现 autoregressive generation 加速
- Weight tying: 输出 logits 直接用 `wte` (token embedding) 转置相乘

### 2.3 GPT-3: In-Context Learning

**架构**: 与 GPT-2 相同, 增加 alternating dense + locally banded sparse attention (类似 Sparse Transformer)

| 模型 | 参数量 | 层数 | d_model | n_heads | d_head | Context |
|------|--------|------|---------|---------|--------|---------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 2048 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 2048 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 2048 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2048 |
| GPT-3 175B | 175B | 96 | 12288 | 96 | 128 | 2048 |

**训练数据** (300B tokens):

| 数据集 | Token 量 | 训练占比 | Epochs |
|--------|---------|---------|--------|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

注意: 高质量数据集 (WebText2, Wikipedia) 的采样权重远高于其在数据集中的占比, 即模型在这些数据上重复训练多次。

**核心贡献: 定义 in-context learning 的三种模式**:

```
Fine-tuning:  gradient updates on task-specific data (传统方式)
Few-shot:     K examples in context, no gradient updates (K = 10~100)
One-shot:     1 example in context
Zero-shot:    only task description, no examples
```

**Meta-learning 视角**: Pre-training 是 outer loop (在大量任务/pattern 上学习), in-context learning 是 inner loop (在推理时快速适配特定任务)。模型在 pre-training 过程中隐式学会了如何从 context 中提取 task specification。

### 2.4 GPT-4: Multimodal + Alignment + Predictable Scaling

**架构**: Transformer-based, 但 OpenAI 未公开以下信息:
- 模型大小 (参数量)
- 训练数据构成
- 训练方法细节
- 硬件配置

**已公开的关键信息**:
- **Multimodal**: 接受文本 + 图像输入, 输出文本
- **RLHF alignment**: Pre-training + RLHF fine-tuning
- **Predictable scaling**: 用 1/1000th ~ 1/10000th compute 的小模型准确预测 GPT-4 的性能

**Predictable scaling 的两个层面**:
1. **Loss prediction**: `L(C) = a * C^b + c` (power law + irreducible loss), 小模型 fit 的曲线准确预测了 GPT-4 的 final loss
2. **Capability prediction**: HumanEval pass rate 也遵循 power law, 可以从小模型外推

**Safety pipeline**:
- Rule-Based Reward Models (RBRMs): 用 zero-shot GPT-4 classifier 作为 RLHF 的额外 reward signal
- 50+ domain experts 进行 adversarial red-teaming
- 结果: 有害内容生成率比 GPT-3.5 降低 82%

---

## 3. Key Designs

### 3.1 从 Fine-tuning 到 In-Context Learning 的范式转变

这是 GPT 系列最重要的技术贡献。四篇论文展示了一条清晰的路径:

| 范式 | 需要什么 | 泛化方式 | 代表 |
|------|---------|---------|------|
| **Supervised learning** | 每个任务大量标注数据 + task-specific 架构 | 不泛化 | Pre-GPT |
| **Pre-train + fine-tune** | Pre-trained weights + 少量标注数据 | 参数级别 transfer | GPT-1 |
| **Zero-shot transfer** | 更大的 pre-trained model, 无标注数据 | 知识已在权重中 | GPT-2 |
| **In-context learning** | Pre-trained model + context 中的 examples | 无参数更新, 纯 forward pass | GPT-3 |
| **Aligned generation** | Pre-trained model + RLHF + prompting | 对齐人类意图 | GPT-4 |

**关键洞察**: GPT-3 论文中最深刻的图是 Figure 1.2 -- in-context learning 能力随模型规模增长而急剧提升。175B 模型的 few-shot learning curve 斜率远大于 1.3B 模型, 暗示 in-context learning 是一种 emergent capability。

### 3.2 Scaling Law 与 Predictable Scaling

GPT-3 验证了 Kaplan et al. (2020) 的 scaling law: 语言模型的 cross-entropy loss 与 compute 之间存在 power-law 关系, 跨 3 个数量级稳定成立。

GPT-4 将其提升到实用层面: 不仅预测 loss, 还预测 **具体 capability metric** (如 HumanEval pass rate)。这意味着可以在训练前评估模型的预期能力, 对 safety 和 deployment 决策至关重要。

```
GPT-3 验证:  L(C) = 2.57 * C^(-0.048)  (cross-entropy loss vs compute)
GPT-4 扩展:  -E[log(pass_rate(C))] = alpha * C^(-k)  (capability vs compute)
```

### 3.3 Data Contamination 的系统性分析

GPT-3 首次对 internet-scale pre-training 的 data contamination 问题做了系统分析:
- 用 13-gram overlap 检测 benchmark contamination
- 构建 "clean" subset 重新评估
- 发现: 大多数 benchmark 上 contamination 对性能影响极小, 但 PIQA (-3pp) 和 Winograd (-2.6pp) 有可测量影响

这套方法论后来成为所有 LLM 论文的标准做法。

### 3.4 GPT-2 代码中的架构决策

从 `gpt-2/src/model.py` 可以验证的关键设计:

1. **Pre-norm (not post-norm)**: `block()` 中先 `norm` 再 `attn/mlp`, 这比原始 Transformer 的 post-norm 更稳定, 是后续所有 GPT 变体的标准做法

2. **1D convolution 实现 linear projection**: 用 `conv1d` (实质是 `matmul`) 替代 `nn.Linear`, 这是 OpenAI 的代码风格

3. **Causal attention mask**: `attention_mask()` 函数生成下三角矩阵, 确保 token 只能 attend 到前面的 token

4. **KV cache**: `past` 参数存储历史的 key/value, 在 generation 时避免重复计算

---

## 4. Experiments

### 4.1 GPT-1: Fine-tuning 性能

| 任务 | GPT-1 | 之前 SOTA | 提升 |
|------|-------|----------|------|
| MNLI-m | 82.1 | 80.6 | +1.5 |
| SNLI | 89.9 | 89.3 | +0.6 |
| Story Cloze | 86.5 | 77.6 | +8.9 |
| RACE | 59.0 | 53.3 | +5.7 |
| GLUE | 72.8 | 68.9 | +3.9 |

**Ablation**: 去掉 pre-training 导致 14.8% 下降; Transformer vs LSTM 差 5.6 分。

### 4.2 GPT-3: In-Context Learning 性能

**Few-shot 接近或超越 fine-tuned SOTA**:

| 任务 | GPT-3 Few-Shot | Fine-tuned SOTA | 评价 |
|------|---------------|-----------------|------|
| LAMBADA (acc) | 86.4 | 68.0 (zero-shot) | 超越 18pp |
| TriviaQA | 71.2 | 68.0 (RAG) | 超越 fine-tuned + retrieval |
| PIQA | 82.8 | 79.4 | 超越 |
| Winogrande | 77.7 | 84.6 | 仍低于 |
| SuperGLUE | 71.8 | 89.0 | 仍有差距 |

**GPT-3 弱项**: NLI (ANLI), 句子比较 (WiC), 阅读理解 (QuAC, RACE)。论文推测这与 autoregressive 架构缺乏 bidirectionality 有关。

**Scaling 趋势**: 42 个 benchmark 的 aggregate 性能随 model size 和 example count 双重增长, few-shot 增长速度快于 zero-shot (Figure 1.3)。

### 4.3 GPT-4: Professional Exam 性能

| 考试 | GPT-4 | GPT-3.5 | 人类百分位 |
|------|-------|---------|----------|
| Uniform Bar Exam | 298/400 | 213/400 | Top 10% |
| LSAT | ~163 | ~149 | ~88th |
| GRE Quantitative | 163/170 | 157/170 | ~80th |
| AP Biology | 5/5 | 4/5 | ~85th |

**NLP Benchmark** (pre-trained base model, few-shot):

| Benchmark | GPT-4 | GPT-3.5 | 之前 SOTA |
|-----------|-------|---------|----------|
| MMLU | 86.4% | 70.0% | 75.2% |
| HellaSwag | 95.3% | 85.5% | 85.6% |

**多语言**: GPT-4 在 24/26 种语言上超过 GPT-3.5 的英语性能 (MMLU 翻译版)。

**Safety**: factuality evaluation 比 GPT-3.5 高 19pp; toxic generation 从 6.48% 降至 0.73%。

---

## 5. Related Work Analysis

### 5.1 GPT 系列在 LLM 发展谱系中的位置

```
Word Embeddings era (2013-2017):
  Word2Vec (2013) -> GloVe (2014) -> ELMo (2018, contextualized)

Pre-train + Fine-tune era (2018-2019):
  GPT-1 (2018) -> BERT (2018) -> XLNet (2019) -> RoBERTa (2019) -> T5 (2019)
      |              |
      v              v
  Autoregressive   Masked LM (bidirectional)
  (GPT 路线)        (BERT 路线)

Scale era (2019-2020):
  GPT-2 (2019, 1.5B) -> Megatron-LM (2019, 8.3B) -> T5-11B (2020)
      |
      v
  GPT-3 (2020, 175B)  <-- 参数量跨越两个数量级

Alignment era (2022-2023):
  InstructGPT (2022) -> ChatGPT (2022.11) -> GPT-4 (2023.03)
      |
      RLHF 范式确立
```

### 5.2 GPT vs BERT: 两条路线的分野

| 维度 | GPT 路线 (autoregressive) | BERT 路线 (masked LM) |
|------|------------------------|---------------------|
| 训练目标 | Next-token prediction | Masked token prediction |
| 方向性 | 单向 (left-to-right) | 双向 |
| 生成能力 | 天然支持 text generation | 不直接支持 generation |
| 下游适配 | Zero/few-shot, 或 fine-tune | 必须 fine-tune |
| Scaling 表现 | 涌现 in-context learning | 未展现类似涌现 |
| 代表性后续 | GPT-3/4, PaLM, LLaMA, Claude | RoBERTa, DeBERTa, ELECTRA |

GPT-3 论文在 Section 5 明确讨论: autoregressive 架构在 bidirectional 任务 (WiC, ANLI, 阅读理解) 上确实弱于 BERT, 但 autoregressive 范式的 scaling behavior 远优于 bidirectional 范式, 最终通过规模优势压倒了架构劣势。

### 5.3 GPT-3 的 Broader Impact 分析开创性

GPT-3 是第一篇在 NLP 论文中系统性分析以下问题的工作:
- **Misuse 风险**: misinformation, phishing, fraud (Section 6.1)
- **Bias 量化**: gender (83% occupation 偏向男性), race (sentiment 分析), religion (Islam 与 terrorism 的关联) (Section 6.2)
- **Energy 消耗**: 数千 petaflop/s-days 的训练成本 (Section 6.3)

这套分析框架后来成为所有大型 LLM 论文的标准组成部分。

---

## 6. Limitations & Future Directions

### 6.1 各代 GPT 的局限

| 版本 | 核心局限 | 后续如何解决 |
|------|---------|-----------|
| GPT-1 | 仍需 task-specific fine-tuning | GPT-2/3 消除了 fine-tuning 的必要性 |
| GPT-2 | Zero-shot 能力有限, 很多任务不如 fine-tuned baseline | GPT-3 通过 1000x 规模提升 + few-shot |
| GPT-3 | 不可靠 (hallucination), 缺乏 bidirectionality, 推理成本高, 训练数据 sample efficiency 低 | GPT-4 通过 RLHF 改善 hallucination; 推理成本仍在 |
| GPT-4 | 仍有 hallucination, knowledge cutoff, 不从经验学习, RLHF 后 calibration 变差 | 仍是 open problem |

### 6.2 GPT-3 论文提出的核心未来方向 (2020 视角)

| 方向 | GPT-3 的预测 | 2023+ 的实际进展 |
|------|-----------|--------------|
| Bidirectional + autoregressive 结合 | "best of both worlds" | PaLM, LLaMA 系列仍用 autoregressive; 规模压倒了架构差异 |
| Fine-tuning with RL | "learning objective from humans" | InstructGPT/RLHF 精确实现了这一方向 |
| Multimodal grounding | "images to provide grounding" | GPT-4V 实现了 vision-language |
| Model distillation | "aggressive distillation" | Llama, Mistral 等开源模型验证了小模型也有强能力 |
| Sample efficiency | 核心瓶颈 | Chinchilla 证明 data scaling 比 model scaling 更重要 |

### 6.3 GPT-4 报告中的透明度问题

GPT-4 Technical Report 是 GPT 系列中信息量最少的一篇:
- 未公开模型大小、架构细节、训练数据
- 仅公开 evaluation 结果和 safety 分析
- 标志着 LLM 从 open research 转向 competitive product 的转折点

这一转变引发了 open-source LLM 运动 (LLaMA, Mistral, Qwen 等)。

---

## 7. Paper vs Code Discrepancies

### GPT-2 Code 分析 (`gpt-2/src/model.py`)

GPT-2 是 GPT 系列中唯一完整开源代码的版本, 可以验证论文中的技术描述:

| 维度 | 论文描述 | 代码实现 | 差异 |
|------|---------|---------|------|
| Normalization | "modified initialization, pre-normalization" | `block()`: `norm(x, 'ln_1')` -> `attn()` -> residual -> `norm(x, 'ln_2')` -> `mlp()` -> residual + 最终 `norm(h, 'ln_f')` | 一致 |
| Activation | GELU | `gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` | 一致, 使用 tanh approximation |
| Linear layer | 标准 linear | `conv1d` with kernel_size=1 (实质是 matmul) | 代码风格不同但数学等价 |
| Attention | Standard masked self-attention | `attention_mask` 用 `i >= j - ns + nd` 构造 causal mask | 一致 |
| Context length | 1024 | `hparams.n_ctx = 1024` | 一致 |
| Weight tying | 未在论文中明确说明 | 输出 `logits = tf.matmul(h_flat, wte, transpose_b=True)` 直接复用 embedding | 代码确认 weight tying |
| Sparse attention | GPT-3 论文提及使用 sparse attention | GPT-2 代码中无 sparse attention | GPT-2 是 dense attention, sparse 是 GPT-3 的改进 |

### 论文系列中的未公开信息

| 版本 | 公开信息 | 未公开信息 |
|------|---------|----------|
| GPT-1 | 架构、数据、训练细节、代码 (部分) | -- |
| GPT-2 | 架构、数据描述、完整代码、模型权重 (分阶段释放) | 训练细节 (optimizer schedule 等) |
| GPT-3 | 架构、数据构成、8 个模型的完整 hyperparameters、详细 benchmark | 训练代码、模型权重 |
| GPT-4 | Benchmark 结果、safety 分析 | 架构、模型大小、数据、训练方法、硬件 -- **几乎所有技术细节** |

---

## 8. Cross-Paper Comparison

### 8.1 GPT 系列的规模演进

| 维度 | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) | GPT-4 (2023) |
|------|-------------|-------------|-------------|-------------|
| 参数量 | 117M | 1.5B | 175B | 未公开 (传闻 ~1.8T MoE) |
| 层数 | 12 | 48 | 96 | 未公开 |
| Context | 512 | 1024 | 2048 | 8K / 32K |
| 训练数据 | BooksCorpus (~1B words) | WebText (40GB, ~10B tokens) | 300B tokens (filtered from ~45TB) | 未公开 |
| 训练 compute | ~8 GPU-days (估算) | ~256 GPU-days (估算) | ~3640 petaflop/s-days | 未公开 |
| 使用范式 | Fine-tune | Zero-shot | Few-shot / In-context | Chat / RLHF / Multimodal |
| Modality | Text only | Text only | Text only | Text + Image |
| Alignment | None | None | None (base model) | RLHF + RBRM |
| 开源程度 | 部分代码 | 完整代码+权重 | API only | API only |

### 8.2 与本 paper library 中其他工作的关联

| 本库论文 | 与 GPT 系列的关系 |
|---------|----------------|
| **FMRobotics** (23) | Survey 的核心分析对象: GPT-3/4 作为 LLM for task planning 的代表 (SayCan, Code-as-Policies 等) |
| **GeneralPurposeRobots** (23) | GPT-4 是 meta-analysis 中 "LLM as planner" 类别最常用的 base model |
| **24_AwesomeSurvey** | Repo 中 Task Planning 和 Code Generation 两个板块的核心驱动力就是 GPT-3/4 |
| **DiffusionPolicy** (24) | 与 GPT 路线形成对比: DP 用 diffusion model 做 continuous action, GPT 系列做 discrete token prediction |
| **GR00T N1** (25) | GR00T N1 的 System 2 (VLM) 直接继承了 GPT-4 确立的 VLM 范式; tokenization 策略受 GPT 影响 |
| **pi_0** (24) | pi_0 的 PaliGemma VLM 部分继承了 GPT-4 的 vision-language 范式, 但 action 生成用 flow matching 而非 autoregressive |
| **DecisionTransformer** (21) | DT 直接将 GPT 的 autoregressive sequence modeling 应用于 RL trajectory, 是 "GPT for RL" |
| **DreamerV3** (23) | 与 GPT 路线正交: DreamerV3 是 model-based RL (world model), GPT 是 model-free sequence prediction |
| **25_ScalingLaws** | GPT-3 验证的 scaling law 是本方向的直接前驱 |

### 8.3 GPT 系列对 Robotics Foundation Model 的影响路径

```
GPT-3 (2020): In-context learning + few-shot
    |
    +---> SayCan (2022): GPT 做 task planning, 用 affordance grounding 连接 robot skills
    +---> Code-as-Policies (2022): GPT 直接生成 robot control code
    +---> Voyager (2023): GPT-4 做 open-ended agent (code generation + skill library)
    |
GPT-4 (2023): Multimodal + RLHF
    |
    +---> RT-2 (2023): VLM co-fine-tune for action generation (GPT 式 token prediction)
    +---> GR00T N1 (2025): VLM (System 2) 继承 GPT-4 范式, DiT (System 1) 做 action
    +---> pi_0 (2024): VLM backbone 继承 GPT 式架构, flow matching 做 action
```

**核心影响**: GPT 系列确立了两个被 robotics 直接继承的范式:
1. **Autoregressive sequence modeling**: 从文本推广到 trajectory (Decision Transformer), 再到 action token (RT-2, Octo)
2. **Vision-Language Foundation Model**: GPT-4V 证明了 multimodal pre-training 的可行性, 被 GR00T N1 和 pi_0 的 VLM 组件直接继承

### 8.4 阅读建议

| 目标 | 推荐阅读顺序 |
|------|-----------|
| 理解 LLM 基础 | GPT-1 (框架) -> GPT-2 code (实现) -> GPT-3 Section 2 (scaling) |
| 理解 in-context learning | GPT-3 Section 1 (Figure 1.1-1.3) + Section 3.9 (synthetic tasks) |
| 理解 scaling law | GPT-3 Section 3 (Figure 3.1) -> GPT-4 Section 3 -> 25_ScalingLaws |
| 理解 LLM safety/alignment | GPT-3 Section 6 (bias analysis) -> GPT-4 Section 6 (RBRM + red-teaming) |
| 理解 GPT 对 robotics 的影响 | GPT-4 -> FMRobotics_notes.md (survey 分析) -> GR00T_N1_notes.md (实际应用) |
