# GPT 系列 -- 从 Generative Pre-Training 到 ChatGPT/GPT-4 的完整演进

**覆盖论文**:
- **GPT-1**: Radford et al., "Improving Language Understanding by Generative Pre-Training", OpenAI, 2018
- **GPT-2**: Radford et al., "Language Models are Unsupervised Multitask Learners", OpenAI, 2019
- **Scaling Laws**: Kaplan et al., "Scaling Laws for Neural Language Models", arXiv:2001.08361, 2020
- **GPT-3**: Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020, arXiv:2005.14165
- **RLHF-Summarize**: Stiennon et al., "Learning to Summarize from Human Feedback", NeurIPS 2020, arXiv:2009.01325
- **Codex**: Chen et al., "Evaluating Large Language Models Trained on Code", arXiv:2107.03374, 2021
- **WebGPT**: Nakano et al., "WebGPT: Browser-assisted Question-answering with Human Feedback", arXiv:2112.09332, 2021
- **InstructGPT**: Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback", NeurIPS 2022, arXiv:2203.02155
- **ChatGPT**: OpenAI blog, 2022.11 (no formal paper; built on InstructGPT methodology)
- **GPT-4**: OpenAI, "GPT-4 Technical Report", arXiv:2303.08774, 2023

**代码仓库**: [openai/gpt-2](https://github.com/openai/gpt-2) (GPT-2, TensorFlow, archived) | [openai/human-eval](https://github.com/openai/human-eval) (Codex benchmark)

---

## 1. Core Problem

### GPT 系列要解决的根本问题

NLP (Natural Language Processing, 自然语言处理) 领域长期面临一个核心矛盾: **人类可以从极少的示例中学会新的语言任务, 但 NLP 系统需要大量标注数据和 task-specific fine-tuning**。GPT 系列论文逐步攻克这一问题。

这 10 篇工作分为两个阶段: **Pre-training 定型 (架构+训练方式的探索)** 和 **Post-training 探索 (在已定型的 base model 上寻找最佳微调方式)**。Pre-training 的架构和训练目标在 GPT-2 (2019) 基本收敛, 之后的创新几乎全在 post-training。

**阶段一: Pre-training -- 架构与 scaling 的定型 (2018-2020)**

| 论文 | 做了什么 | 定型了什么 |
|------|---------|----------|
| GPT-1 (2018) | 确立 pre-train + fine-tune 范式 | Transformer decoder + autoregressive next-token prediction |
| GPT-2 (2019) | 去掉 fine-tune, 验证 zero-shot | 架构定型 (pre-norm, GELU, KV cache), 后续只需放大 |
| Scaling Laws (2020.01) | 量化 scaling 规律 | 理论定型: loss 与 compute 的 power-law 关系 |
| GPT-3 (2020.05) | 放大到 175B, in-context learning 涌现 | Scaling 验证完毕, pre-training 的故事到此结束 |

> **术语说明**:
> - **LM** (Language Model, 语言模型) / **LLM** (Large Language Model, 大语言模型, 通常指 >1B 参数)
> - **GELU** (Gaussian Error Linear Unit, 高斯误差线性单元) -- 比 ReLU 更平滑的激活函数, 现代 LLM 的标准选择
> - **KV cache** (Key-Value cache, 键值缓存) -- 推理时缓存已计算的 attention Key 和 Value, 避免重复计算

GPT-2 之后, pre-training 的架构 (Transformer decoder)、训练目标 (next-token prediction)、训练方式 (大数据 + 大模型) 基本不变。GPT-3 只是把 GPT-2 放大了 117 倍, 核心设计完全一样。

**阶段二: Post-training -- fine-tune 方式的探索 (2020-2023)**

Pre-training 定型后, 问题变成: **同一个 base model, 用什么方式 fine-tune 最有效?**

| 论文 | 探索了什么 fine-tune 方式 | 核心发现 |
|------|-------------------------|---------|
| RLHF-Summarize (2020.09) | RLHF (RL + 人类偏好) | 人类"判断好坏"比"写标准答案"便宜, 且效果更好 |
| Codex (2021.07) | 领域 SFT (代码数据续训) | 同一个 base model + 领域数据 = 专业能力涌现 |
| WebGPT (2021.12) | BC + RLHF (工具使用) | LLM 可以学会使用外部工具 |
| InstructGPT (2022.03) | SFT + RM + PPO (三步法) | Alignment > Scale: 对齐后的 1.3B > 未对齐的 175B |
| ChatGPT (2022.11) | InstructGPT 方法 + 对话数据 | 三步法可产品化, 2 个月 1 亿用户 |
| GPT-4 (2023.03) | 多模态 + RLHF + RBRM | 扩展到图像输入, 加入基于规则的安全对齐 |

> **术语说明** (按首次出现顺序):
> - **RLHF** (Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习) -- 不用人写"标准答案", 而是让人比较两个回答哪个更好, 用这个偏好信号训练模型
> - **SFT** (Supervised Fine-Tuning, 监督微调) -- 用人工编写的高质量回答直接微调模型, 给 RLHF 一个合格的起点
> - **RM** (Reward Model, 奖励模型) -- 从人类比较数据中训练, 输入 (prompt, response) 输出标量分数, 代替人类在 PPO 循环中反复打分
> - **PPO** (Proximal Policy Optimization, 近端策略优化) -- 一种 RL 算法, 用 RM 的分数作为 reward 信号优化模型
> - **BC** (Behavior Cloning, 行为克隆) -- 用监督学习直接模仿人类操作轨迹, 最简单的 imitation learning
> - **RBRM** (Rule-Based Reward Model, 基于规则的奖励模型) -- GPT-4 引入, 用规则而非人类偏好定义安全边界

注意: 阶段二的所有工作都不改变 pre-training 的架构或训练目标, 它们都是在 GPT-3 这个已训好的 base model 上做不同方式的 fine-tune。本质上都是**利用 base model 已有的能力, 通过不同的训练信号 (人类示范 / 人类偏好 / 领域数据) 将能力导向特定方向**。这个 "一个 base model, 多种 post-training" 的模式后来被机器人领域直接继承 (PaliGemma base model -> pi_0 通过机器人数据 fine-tune 获得动作生成能力)。

### 完整演进脉络

```
========= 阶段一: Pre-training 定型 (2018-2020) =========
  目标: 找到正确的架构、训练目标、scaling 策略

GPT-1 (2018.06): 确立 Transformer decoder + next-token prediction
  |  (insight: pre-train 提供通用能力, fine-tune 特化到任务)
  v
GPT-2 (2019.02): 架构定型 (pre-norm, GELU, KV cache)
  |  (insight: 架构不用改, 放大就行; zero-shot 能力随规模增长)
  v
Scaling Laws (2020.01): 理论定型 (power-law, compute-optimal)
  |  (结论: 性能可预测, 给了 GPT-3 放大到 175B 的信心)
  v
GPT-3 (2020.05): scaling 验证完毕, in-context learning 涌现
  |
  |  *** Pre-training 的故事到此结束 ***
  |  *** 架构和训练目标不再改变, 之后全是 post-training ***
  |
  |  (新问题: 模型能力够了, 但不听话、不可靠、有毒性)

========= 阶段二: Post-training 探索 (2020-2023) =========
  目标: base model 已定, 探索最佳 fine-tune 方式
  本质: 所有工作都是在 GPT-3 上做不同方式的 fine-tune,
        利用 base model 已有能力, 用不同训练信号将能力导向特定方向

RLHF-Summarize (2020.09): 验证 RLHF 可行
  |  (发现: 人类"判断好坏"比"写标准答案"便宜且效果更好)
  |  (但只在摘要单一任务上验证)
  |
  +---> Codex (2021.07): 领域 SFT 验证
  |       (发现: base model + 领域数据 fine-tune = 专业能力)
  |
  +---> WebGPT (2021.12): BC + RLHF + 工具使用验证
  |       (发现: LLM 可以学会使用外部工具)
  v
InstructGPT (2022.03): post-training 范式定型 (SFT -> RM -> PPO)
  |  (发现: alignment > scale, 对齐后 1.3B > 未对齐 175B)
  |  (意义: 将 RLHF 从单任务扩展到通用指令遵循)
  v
ChatGPT (2022.11): InstructGPT 产品化 -> 2 个月 1 亿用户
  v
GPT-4 (2023.03): 扩展到多模态输入 + 更好的 post-training
  v
GPT-4o / o1 (2024): omni-modal + inference-time reasoning
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
- GELU (Gaussian Error Linear Unit, 高斯误差线性单元 -- 比 ReLU 更平滑的激活函数) activation: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`
- KV cache (Key-Value cache, 键值缓存 -- 推理时缓存已计算的 attention Key 和 Value, 避免重复计算) 支持: `past` 参数实现 autoregressive generation 加速
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

### 2.4 Scaling Laws (2020): 为 GPT-3 提供理论基础

**核心发现**: LM 的 cross-entropy loss 与三个因素呈 power-law 关系:

```
L(N) = (N_c / N)^alpha_N     ; alpha_N ~ 0.076, N_c ~ 8.8e13 (params)
L(D) = (D_c / D)^alpha_D     ; alpha_D ~ 0.095, D_c ~ 5.4e13 (tokens)
L(C) = (C_c / C)^alpha_C     ; alpha_C ~ 0.050, C_c ~ 3.1e8  (PF-days)
```

**关键结论**:
- **Performance depends strongly on scale, weakly on shape**: 模型宽度 vs 深度影响很小, 总参数量才是关键
- **Larger models are more sample-efficient**: 大模型用更少的数据就能达到同样性能
- **Compute-efficient training = 训练大模型但不到收敛**: 固定 compute budget 下, 最优策略是训练非常大的模型, 但在收敛前停止 (N ~ C^0.73, D ~ C^0.27)
- **Overfitting 可预测**: 每增大模型 8x, 只需增大数据 ~5x 即可避免过拟合 (N^0.74/D)

**对 GPT-3 的直接影响**: Scaling Laws 论文 (2020.01) 发表 4 个月后 GPT-3 (2020.05) 出现, GPT-3 的设计正是遵循了 "优先扩大模型" 的结论 -- 从 GPT-2 的 1.5B 直接跳到 175B (117x)。

**后续修正 (Chinchilla, 2022)**: Hoffmann et al. 发现 Kaplan 的结论低估了数据的重要性。Chinchilla 法则认为模型参数和训练 token 应 1:1 等比增长 (N ~ 20D), 而非 Kaplan 建议的优先扩大模型。这解释了为什么 Chinchilla-70B (1.4T tokens) 优于 Gopher-280B (300B tokens)。

### 2.5 RLHF-Summarize (2020): RLHF 方法的奠基

**问题**: 训练目标 (MLE, Maximum Likelihood Estimation, 最大似然估计 -- 让模型尽可能拟合训练数据分布) 和真正目标 (generate good summaries) 之间存在 misalignment:
- MLE 无法区分重要错误 (编造事实) 和不重要错误 (用词不精确)
- Beam search 等策略可以提升 quality 但引入 repetition

**三步法** (后来被 InstructGPT 完整继承):

```
Step 1: Collect human feedback
  - 对同一 Reddit post, 从多个 policy 采样摘要
  - 人类标注者对比两两摘要, 选出更好的
  - 数据集: 64,832 comparisons on TL;DR

Step 2: Train reward model (RM)
  - 输入: (post, summary) -> 标量 reward
  - 训练目标: predict log-odds of human preference
  - loss = log(sigma(r_j - r_k))  // j preferred over k
  - 基于 GPT-3 (1.3B / 6.7B) 去掉最后一层 + 加 scalar head

Step 3: Optimize policy with PPO
  - Reward = RM_score - beta * KL(policy || reference_policy)
  - KL penalty 防止 policy 过度偏离 pre-trained model (reward hacking)
  - 用 PPO 算法 (Schulman et al., 2017) 优化
```

**核心发现**:
- **Human feedback 训练的 6.7B 模型 > SFT 训练的 12.9B 模型** (人类偏好率更高)
- Reward model 泛化到新领域: Reddit 上训练的 RM 在 CNN/DM 新闻上同样有效
- Optimizing reward model > optimizing ROUGE, 但过度优化 RM 会导致 quality 下降 (reward hacking 的早期观察)

**历史意义**: 这是第一篇证明 "RLHF 在大规模 LM 上有效" 的论文。方法论直接被 InstructGPT/ChatGPT 继承, 论文一作 Stiennon 和二作 Ouyang 后来分别是 RLHF-Summarize 和 InstructGPT 的一作。

### 2.6 Codex (2021): GPT-3 的代码专精版

**方法**: 在 GPT-3 基础上, 用 GitHub 公开代码 fine-tune:
- 训练数据: 54M public repos, 159GB unique Python files (过滤后)
- 模型: GPT-3 的多个尺寸 (300M ~ 12B params), 基于 GPT-3 tokenizer + 额外的 whitespace tokens (节省 30% token)
- 训练: 100B tokens, cosine LR decay

**HumanEval benchmark** (164 hand-written problems):
- 引入 **pass@k metric**: 生成 k 个样本, 任一通过 unit test 即算解决

| 模型 | pass@1 | pass@100 |
|------|--------|----------|
| GPT-3 12B | 0% | -- |
| GPT-J 6B | 11.4% | -- |
| Codex 300M | 13.2% | -- |
| Codex 12B | 28.8% | 72.3% |
| Codex-S 12B (SFT on standalone functions) | 37.7% | 77.5% |

**关键发现**:
- **Scaling law 在代码领域也成立**: test loss 随参数量呈 power-law 下降, 指数 -0.13
- **Repeated sampling 极其有效**: pass@100 远高于 pass@1, 说明模型 "知道怎么做" 但单次不一定做对
- **从 GPT-3 fine-tune 比从 scratch 训练收敛更快** (但最终性能类似), 因为 GPT-3 已有自然语言理解能力
- **Docstring → code 比 code → code 难**: 模型理解自然语言描述并转化为代码是核心挑战

**产品影响**: Codex 的生产版本成为 GitHub Copilot (2021.10 发布), 是第一个大规模商用的 AI 编程助手。

### 2.7 WebGPT (2021): 教模型使用工具

**创新点**: 不是让模型凭记忆回答问题, 而是让它 **学会使用搜索引擎**:

```
Environment: text-based web browser
Actions: Search <query>, Click on link <id>, Quote <text>,
         Scroll down/up, Find in page, Back, End: Answer
Task: answer ELI5 questions with references
```

**训练方法** (四种, 效果递增):
1. **Behavior Cloning (BC)**: 监督学习模仿人类浏览行为
2. **Reward Modeling (RM)**: 对比人类偏好, 训练 reward model
3. **Reinforcement Learning (RL)**: PPO 优化 policy against RM
4. **Rejection Sampling (best-of-n)**: 从 BC model 采样 n 个答案, 用 RM 选最好的

最佳模型: 175B best-of-64 (BC + rejection sampling)

**核心发现**:
- 模型答案被偏好超过人类 demonstrator 56% of the time
- 对比 Reddit 最高赞答案, 被偏好 69% of the time
- **关键**: 使用 rejection sampling + RM 选择, 比直接用 RL 训练效果更好 (且更简单)
- **References 机制**: 模型在浏览时 quote 网页段落作为引用来源, 大幅提升了 factual accuracy 的可验证性

**意义**: WebGPT 证明 LLM 可以学会使用 **外部工具**, 这是后来 tool-use (function calling) 的前身。同时, "检索 + 生成" 的模式直接影响了 RAG (Retrieval-Augmented Generation) 的流行。

### 2.8 InstructGPT (2022): ChatGPT 的直接前身

**核心问题**: Making language models bigger does NOT inherently make them better at following user intent. GPT-3 175B 能力很强但 misaligned -- 不遵循指令、编造事实、生成有害内容。

**方法**: 继承 RLHF-Summarize 的三步法, 但目标从 "好摘要" 扩展到 "遵循所有类型的指令":

```
Step 1: Supervised Fine-Tuning (SFT)
  - 数据: 40 contractors 编写的 ~13K demonstration prompts
  - 来源: OpenAI API 用户提交的 prompt + labeler 自创的 prompt
  - 训练: 在 GPT-3 上 fine-tune 16 epochs (故意过拟合, 因为后续 RL 会调整)

Step 2: Reward Model (RM) Training
  - 数据: ~33K prompts, 每个 prompt 4~9 个模型输出, 人类排序
  - 模型: 6B params (不用 175B, 因为大 RM 训练不稳定)
  - Loss: pairwise ranking loss -- 对所有 (K choose 2) 对比, 求和
  - loss = -1/(K choose 2) * sum_{(i,j)} log(sigma(r_j - r_i))

Step 3: PPO Optimization
  - Reward = RM(prompt, response) - beta * KL(policy || SFT_model)
  - 变体 PPO-ptx: 混入 pre-training gradient 防止在公共 NLP 数据集上退化
  - objective = E[RM(x,y)] - beta*KL + gamma*E[log P_pretrain(x)]
```

**核心发现**:

| 对比 | 人类偏好率 |
|------|----------|
| 175B InstructGPT vs 175B GPT-3 | 85% ± 3% |
| **1.3B InstructGPT** vs **175B GPT-3** | **> 50%** (fewer params, better aligned) |
| InstructGPT vs FLAN/T0 (instruction-tuned) | 73.4% vs 26.8% / 29.8% |

- **Truthfulness**: InstructGPT 生成 truthful+informative 答案的频率是 GPT-3 的 2 倍
- **Toxicity**: 毒性输出减少 25%
- **Hallucination**: closed-domain 任务上 hallucination rate 21% vs GPT-3 的 41%
- **Alignment tax**: RLHF 导致在 SQuAD/HellaSwag 等 NLP benchmark 上性能轻微下降, 但 PPO-ptx 可以缓解
- **泛化到 unseen tasks**: InstructGPT 能遵循中文指令、写代码, 尽管训练数据几乎全是英文非代码任务

**从 InstructGPT 到 ChatGPT**: ChatGPT (2022.11) 没有独立论文。根据 OpenAI 博客, ChatGPT 使用了与 InstructGPT 相同的方法论, 主要区别是:
1. 对话数据代替单轮 instruction 数据
2. 更大的基础模型 (GPT-3.5 而非 GPT-3)
3. 产品层面的优化 (多轮对话管理、安全过滤等)

### 2.9 GPT-4: Multimodal + Alignment + Predictable Scaling

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
| **Domain fine-tune** | Pre-trained model + 领域数据 | 专业能力涌现 | Codex |
| **RLHF alignment** | Pre-trained model + human feedback + PPO | 对齐人类偏好 | InstructGPT/ChatGPT |
| **Tool-augmented** | Aligned model + 外部工具 (搜索引擎等) | 突破知识截止限制 | WebGPT |
| **Multimodal aligned** | RLHF + 多模态输入 | 跨模态理解 | GPT-4 |

**关键洞察**: GPT-3 论文中最深刻的图是 Figure 1.2 -- in-context learning 能力随模型规模增长而急剧提升。175B 模型的 few-shot learning curve 斜率远大于 1.3B 模型, 暗示 in-context learning 是一种 emergent capability。

### 3.2 GPT-3: 一个 Base Model, 多种 Fine-tune = 多种能力

GPT-3 之后、GPT-4 之前的工作有一个共同模式: **全部基于 GPT-3 做不同方向的 fine-tune**。同一个 base model, 通过不同数据和方法, 获得完全不同的能力:

```
GPT-3 (175B base model, 2020.05)
  |
  +---> Codex (2021.07)          fine-tune on 代码数据
  +---> WebGPT (2021.12)         fine-tune on 浏览器交互数据
  +---> InstructGPT (2022.03)    fine-tune on 指令遵循数据
  +---> ChatGPT (2022.11)        fine-tune on 对话数据 (基于 GPT-3.5)
```

| 工作 | Base Model | Fine-tune 方法 | Fine-tune 数据 | 获得的能力 |
|------|-----------|---------------|---------------|----------|
| Codex | GPT-3 | Supervised (续训 100B tokens) | 159GB GitHub Python 代码 | 代码生成 (pass@1: 28.8%) |
| WebGPT | GPT-3 | BC + RLHF | 人类浏览器操作轨迹 + 偏好比较 | 搜索引擎使用 + 带引用的回答 |
| InstructGPT | GPT-3 | SFT + RLHF (3 步) | 13K 指令演示 + 33K 偏好比较 | 遵循指令 + 安全对齐 |
| ChatGPT | GPT-3.5 | SFT + RLHF | 对话数据 + 偏好比较 | 多轮对话 |

**这个模式对机器人的直接启示**: pi_0 做了完全相同的事 -- PaliGemma 3B (VLM base model) 本身不会控制机器人, 通过机器人操作数据 fine-tune 后获得动作生成能力。领域 fine-tune 的效果远大于模型 scale: GPT-3 175B 不会写代码, 但 Codex 12B (小 14x) 经过代码 fine-tune 就能写。

### 3.3 Scaling Law 与 Predictable Scaling

GPT-3 验证了 Kaplan et al. (2020) 的 scaling law: 语言模型的 cross-entropy loss 与 compute 之间存在 power-law 关系, 跨 3 个数量级稳定成立。

GPT-4 将其提升到实用层面: 不仅预测 loss, 还预测 **具体 capability metric** (如 HumanEval pass rate)。这意味着可以在训练前评估模型的预期能力, 对 safety 和 deployment 决策至关重要。

```
GPT-3 验证:  L(C) = 2.57 * C^(-0.048)  (cross-entropy loss vs compute)
GPT-4 扩展:  -E[log(pass_rate(C))] = alpha * C^(-k)  (capability vs compute)
```

### 3.4 Data Contamination 的系统性分析

GPT-3 首次对 internet-scale pre-training 的 data contamination 问题做了系统分析:
- 用 13-gram overlap 检测 benchmark contamination
- 构建 "clean" subset 重新评估
- 发现: 大多数 benchmark 上 contamination 对性能影响极小, 但 PIQA (-3pp) 和 Winograd (-2.6pp) 有可测量影响

这套方法论后来成为所有 LLM 论文的标准做法。

### 3.5 GPT-2 代码中的架构决策

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
  Scaling Laws (2020.01) -- 理论指导: 该怎么扩大
      |
  GPT-2 (2019, 1.5B) -> Megatron-LM (2019, 8.3B) -> T5-11B (2020)
      |
      v
  GPT-3 (2020, 175B)  <-- 参数量跨越两个数量级

RLHF + Domain specialization era (2020-2022):
  RLHF-Summarize (2020) -- 奠基: 证明 human feedback > supervised
      |
      +---> Codex (2021) ------> GitHub Copilot (产品)
      +---> WebGPT (2021) -----> 工具使用 + 引用 (factuality)
      v
  InstructGPT (2022) -- 集大成: SFT + RM + PPO, 1.3B > 175B
      v
  ChatGPT (2022.11) -- 产品爆发, 改变行业

Multimodal + Reasoning era (2023+):
  GPT-4 (2023.03) -> GPT-4o (2024.05) -> o1 (2024.09)
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
| **ACT** (23) | Action chunking 用 Transformer decoder 做 action prediction, 继承 GPT 的 autoregressive 思想但输出连续动作 |
| **DiT** (23) | Transformer 替代 U-Net 做 diffusion backbone, 证明 GPT 式 Transformer scaling 在生成模型中同样成立 |
| **FlowMatching** (22) | pi_0 用 flow matching 替代 GPT 式 discrete token prediction 做 action generation, 两种范式的分水岭 |
| **OpenXEmbodiment** (23) | 跨机器人数据集, RT-1-X/RT-2-X 直接继承 GPT 式 tokenization 做 action prediction |
| **Octo** (24) | Berkeley 开源 generalist robot policy, readout token 机制是 GPT in-context learning 在机器人中的变体 |
| **OpenVLA** (24) | 开源 VLA, 用 GPT 式 discrete action token prediction (256-bin), 是 GPT->RT-2 路线的开源实现 |
| **PaliGemma** (24) | pi_0 的 VLM backbone, prefix-LM 架构继承了 GPT 的 causal attention 但在 image tokens 上允许双向 |
| **25_RobotScalingLaws** | GPT-3 验证的 scaling law 在机器人领域的验证, 证明 robot performance 也遵循 power-law |

### 8.3 GPT 系列对 Robotics Foundation Model 的影响路径

```
GPT-1/2/3 (2018-2020): Pre-train + scaling + in-context learning
    |
    +---> Decision Transformer (2021): RL trajectory = token sequence
    +---> SayCan (2022): GPT 做 task planning + affordance grounding
    +---> RT-1 (2022): Robotics Transformer, 动作离散化为 token
    |
Scaling Laws (2020) + Chinchilla (2022): 理论指导
    |
    +---> 25_RobotScalingLaws (2025): 验证机器人也遵循 scaling law
    |
GPT-4 (2023): Multimodal + RLHF
    |
    +--- 路线 A: Discrete action tokens (GPT 式 autoregressive) ---+
    |    RT-2 (2023) -> OpenVLA (2024)                              |
    |                                                               |
    +--- 路线 B: Continuous actions (diffusion/flow) ---------------+
    |    ACT (2023) -> Diffusion Policy (2023) -> Octo (2024)       |
    |                                                               |
    +--- 融合: VLM backbone + continuous action head ---------------+
         pi_0 (2024): PaliGemma + flow matching action expert
         GR00T N1 (2025): Eagle VLM (10Hz) + DiT action head (120Hz)
```

**核心影响**: GPT 系列确立了三个被 robotics 直接继承的范式:
1. **Autoregressive sequence modeling**: 文本 -> trajectory (DT) -> action tokens (RT-2, OpenVLA)
2. **Vision-Language Foundation Model**: GPT-4V -> PaliGemma/Eagle -> pi_0/GR00T N1 的 VLM 组件
3. **Scaling law 思维**: 先小规模验证 power-law, 再决定投入多少 compute -- 被 robot data scaling 直接继承 (Open X-Embodiment)

### 8.4 阅读建议

| 目标 | 推荐阅读顺序 |
|------|-----------|
| 理解 LLM 基础 | GPT-1 (框架) -> GPT-2 code (实现) -> GPT-3 Section 2 (scaling) |
| 理解 in-context learning | GPT-3 Section 1 (Figure 1.1-1.3) + Section 3.9 (synthetic tasks) |
| 理解 scaling law | Scaling Laws (Kaplan) -> GPT-3 Section 3 -> Chinchilla (修正) -> GPT-4 Section 3 |
| **理解 RLHF/alignment** | **RLHF-Summarize (方法奠基) -> InstructGPT (完整方法) -> ChatGPT (产品)** |
| **理解 tool-use** | **WebGPT (原始工具使用) -> ChatGPT Plugins (2023) -> function calling** |
| **理解 code generation** | **Codex (HumanEval) -> GitHub Copilot -> GPT-4 (代码能力大幅提升)** |
| 理解 LLM safety/alignment | GPT-3 Section 6 (bias) -> InstructGPT Section 4 (alignment tax) -> GPT-4 Section 6 (RBRM) |
| 理解 GPT 对 robotics 的影响 | GPT-4 -> FMRobotics_notes.md (survey 分析) -> GR00T_N1_notes.md (实际应用) |

---

## 9. GPT Business Logic -- 从论文时间线推测 OpenAI 的战略决策

### 9.1 前史: 为什么是 OpenAI, 为什么是 autoregressive

**2015.12 -- OpenAI 成立 (非营利)**

Elon Musk, Sam Altman 等人成立 OpenAI, 使命是 "ensure AGI benefits all of humanity"。初始捐赠 ~$1B。此时 AI 领域主流是 CNN (视觉) + LSTM (NLP), 没有人知道通往 AGI 的路径。

**2017.06 -- Google 发表 "Attention Is All You Need"**

Transformer 架构诞生。但 Google 自己把它当作 machine translation 的改进工具, 没有看到 scaling 的潜力。OpenAI 的 Alec Radford 注意到了这个架构的通用性。

**2018.06 -- GPT-1: 一次关键的技术赌注**

同一时期, Google 在做 BERT (2018.10), 用 masked language model (双向)。GPT-1 选择了 autoregressive (单向)。
从当时的 benchmark 看, BERT 更强 -- 双向注意力在理解任务上有天然优势。

**为什么 OpenAI 坚持 autoregressive?** 从论文能推测的原因:
- Autoregressive 天然支持 **generation** (BERT 不行)
- GPT-1 论文就发现: zero-shot 性能随模型增大而提升 (Section 5), 暗示了 scaling 的潜力
- 如果目标是 AGI (通用智能), generation >> understanding -- 你需要模型能 "做事", 不只是 "理解"

这是一个在当时看起来 "输了" 的赌注 (BERT 横扫所有 NLP benchmark), 但从 AGI 视角看是正确的方向选择。

### 9.2 Scaling Law 作为 GPT-3 的商业论证

**2019.02 -- GPT-2: "大力出奇迹" 的初步验证**

GPT-2 (1.5B) 证明了放大模型 + 数据后 zero-shot 能力显著提升。但这里有一个关键问题:

```
GPT-1 (117M) → GPT-2 (1.5B): 花费 ~$50K, 效果明显提升
GPT-2 (1.5B) → GPT-3 (?): 需要花费数百万美元

凭什么相信继续放大还会有效?
万一 1.5B 已经是 "够大了" 呢?
```

**2019.03 -- OpenAI 从非营利转为 "capped-profit"**

这不是巧合。要训练更大的模型, 需要更多的钱。非营利结构无法支撑。OpenAI 需要向投资者证明: 继续投入 compute **一定会**带来可预测的回报。

**2019.07 -- Microsoft 投资 $1B**

Microsoft 的钱不是白给的。OpenAI 需要一个 "科学论据" 来证明:
1. 更大的模型确实更好 (不是靠信仰)
2. 性能提升是 **可预测的** (不是赌博)
3. 投入和产出之间有 **确定性关系** (power law)

**2020.01 -- Scaling Laws 论文: 正是这个论据**

Kaplan (物理学家) 的论文完美解决了这个问题:
- 证明 loss 与 compute 之间存在精确的 power law
- 证明回报是可预测的, 不是随机的
- 给出了具体的最优分配策略: 优先扩大模型

**从商业角度看, Scaling Laws 论文本质上是 GPT-3 的 feasibility study / 投资论证书。**
它告诉 Microsoft: "给我 X 倍的 compute, 我能给你 Y 的性能提升, 这是数学保证。"

**2020.05 -- GPT-3: 执行 Scaling Laws 的结论**

GPT-3 的设计完全遵循了 Scaling Laws 的指导:
- 参数量: 1.5B → 175B (跳了两个数量级, 因为 power law 说应该优先扩大模型)
- 训练数据: 40GB → 570GB (增长相对保守, 因为 Kaplan 说数据不那么重要)
- 训练没跑到收敛 (因为 Scaling Laws 说 "train large, stop early")

结果验证了预测: in-context learning 涌现, few-shot 接近 fine-tuned SOTA。

### 9.3 从 "能做" 到 "能卖": Alignment 阶段的商业逻辑

**2020.06 -- GPT-3 API 上线: 第一次商业化尝试**

GPT-3 通过 API 向开发者收费。但问题很快暴露:
- 模型不遵循指令 (用户说 "翻译这句话", 模型继续写一篇文章)
- 生成有害内容 (有毒性, 有偏见)
- 不可靠 (hallucination)

**API 收入远低于训练成本。** GPT-3 是一个技术奇迹, 但不是一个好产品。

**2020.09 -- RLHF-Summarize: 解决 "不听话" 问题的第一步**

OpenAI Alignment team (Stiennon, Ouyang, Christiano 等) 开始系统性研究如何让模型 "听话"。
选择了摘要任务作为试验场 -- 因为:
- 任务明确 (好摘要 vs 坏摘要, 人类容易判断)
- 可以量化 (人类偏好率)
- 方法可推广 (从摘要到所有指令)

核心发现: **6.7B + human feedback > 12.9B + supervised learning**。
这意味着 alignment 不需要等模型变得更大, 可以在现有模型上直接改善。

**2021.07 -- Codex → GitHub Copilot: 第一个 "killer app"**

在 alignment 研究还在进行的同时, OpenAI 找到了一个不需要完美 alignment 的商业场景: **代码补全**。
- 代码有明确的正确性标准 (通过 unit test)
- 用户是程序员, 能自己判断和修正输出
- Microsoft 通过 GitHub 有现成的分发渠道

Copilot (2021.10) 成为第一个大规模商用的 LLM 产品, 验证了 "LLM + 垂直场景" 的商业模式。

**2021.12 -- WebGPT: 为 ChatGPT 铺路**

WebGPT 看似是一个研究项目, 但从产品角度它验证了两个关键能力:
- 模型可以学会使用 **工具** (搜索引擎) -- 后来成为 ChatGPT Plugins / function calling
- Human feedback 可以提升 **事实准确性** -- 直接对应 ChatGPT 的 "减少 hallucination"

**2022.03 -- InstructGPT: 产品化的临门一脚**

InstructGPT 的核心发现 "1.3B InstructGPT > 175B GPT-3" 有两层含义:
- **技术层面**: alignment 比 scale 更重要
- **商业层面**: 不需要等到训练出更大的模型, 现在就可以做出好产品

InstructGPT 论文的 Figure 1 (本目录 22_InstructGPT.pdf) 就是产品化的信号:
PPO-ptx 在所有尺寸上都大幅超越 GPT baseline, 意味着 RLHF pipeline 已经成熟。

### 9.4 ChatGPT: 时机、执行与引爆

**2022.11.30 -- ChatGPT 发布**

ChatGPT 没有独立论文, 因为它不是技术突破 -- 它是 **工程整合**:
- 基础模型: GPT-3.5 (GPT-3 的改进版, 已有 code 训练)
- Alignment: InstructGPT 的 SFT + RLHF pipeline
- 产品形态: 多轮对话界面 (而非 API)

**为什么这个时机爆发?**

```
技术条件 (2022 年全部就位):
  [x] 足够强的 base model (GPT-3.5)
  [x] 成熟的 alignment 方法 (InstructGPT)
  [x] 代码能力 (Codex 训练数据的继承)
  [x] 工具使用能力 (WebGPT 的经验)

产品条件:
  [x] 对话界面 (比 API 低一个数量级的使用门槛)
  [x] 免费试用 (降低获客成本)
  [x] 足够安全 (RLHF 降低了有害输出, 使公开发布成为可能)
```

2 个月达到 1 亿用户。这个速度说明需求一直存在, 只是之前缺少一个 "足够好 + 足够安全 + 足够易用" 的产品。

### 9.5 GPT-4: 从 open research 到 competitive moat

**2023.03 -- GPT-4 Technical Report: 信息量最少的一篇**

GPT-4 论文不公开架构、参数量、训练数据、训练方法。这不是偶然:

```
GPT-1 (2018): 完全公开 -- 需要学术声誉, 团队还小
GPT-2 (2019): 完全公开 (含代码+权重) -- "太危险" 的 PR 策略
GPT-3 (2020): 公开论文但不开源 -- API 商业化开始
GPT-4 (2023): 几乎不公开 -- 竞争壁垒

开放程度与商业价值成反比。
```

GPT-4 报告唯一详细的部分是 **safety** 和 **predictable scaling**:
- Safety: "我们是负责任的" (回应社会压力)
- Predictable scaling: "我们能预测下一代模型的能力" (向投资者展示确定性)

### 9.6 总结: 论文时间线背后的战略逻辑

```
Phase 1 -- 技术探索 (2018-2019): 花别人的钱做研究
  GPT-1/2: 确立 autoregressive 路线, 发现 scale 有效
  成本: 数十万美元级
  收入: 零

Phase 2 -- 理论论证 + 融资 (2019-2020): 用科学说服资本
  Scaling Laws: 证明投入与回报有确定性关系
  OpenAI 转为 capped-profit, Microsoft $1B 投资
  成本: 数百万美元级 (GPT-3 训练)
  收入: API 收入 (有限)

Phase 3 -- 产品探索 (2020-2022): 找到 PMF (Product-Market Fit)
  Codex/Copilot: 验证垂直场景商业模式
  RLHF/InstructGPT: 解决 "好用" 问题
  成本: 数千万美元级
  收入: Copilot 订阅 ($10/month/user)

Phase 4 -- 产品爆发 (2022-2023): 技术积累的商业变现
  ChatGPT: 所有技术整合为消费级产品
  GPT-4: 关闭技术细节, 建立竞争壁垒
  Microsoft 追加 $10B 投资
  成本: 数亿美元级
  收入: ChatGPT Plus ($20/month) + API + Enterprise

Phase 5 -- 生态竞争 (2024+): 平台化
  GPT-4o, o1: 持续迭代
  GPT Store, Plugins, function calling: 平台生态
  gpt-oss (2025): 开源小模型, 扩大生态 (应对 Llama/Qwen 竞争)
  成本: 数十亿美元级
  收入: 年化 ~$5B+ (2024 估算)
```

每一篇论文都不是孤立的学术研究, 而是整体战略的一个环节:
- **Scaling Laws** = 投资论证
- **GPT-3** = 技术验证
- **RLHF-Summarize** = 产品化基础研究
- **Codex** = 第一个商业落地
- **WebGPT** = 工具能力储备
- **InstructGPT** = 产品可用性突破
- **ChatGPT** = 商业引爆点
- **GPT-4** = 竞争壁垒
