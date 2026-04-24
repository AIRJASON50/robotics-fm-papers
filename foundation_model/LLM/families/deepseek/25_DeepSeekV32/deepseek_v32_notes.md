# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

> arXiv:2512.02556 (2025.12.02), DeepSeek-AI
> 实验前身: DeepSeek-V3.2-Exp (2025.09.29 发布)
> 基础模型: DeepSeek-V3.1-Terminus (在 V3.1 基础上 128K context 扩展)
> 最强变体: DeepSeek-V3.2-Speciale (推理专用, IMO/IOI/ICPC/CMO 2025 全部金牌)

---

## 1. Core Problem

闭源 (GPT-5, Sonnet 4.5, Gemini 3.0) 和开源模型差距正在**拉大**而不是缩小。论文 identify 三个开源模型的关键短板:

1. **架构**: 仍依赖 vanilla attention, 长序列效率低 → 部署和 post-training 都被卡住
2. **资源分配**: post-training 计算投入不足, 难任务上不去
3. **Agent 能力**: 泛化和指令跟随明显弱于闭源

V3.2 的目标不是再发明一个新架构, 而是**针对这三点逐一解决**:
- 架构: DSA (DeepSeek Sparse Attention) 解决长上下文
- 资源: post-training 算力 > 10% 预训练, 设计稳定的 RL recipe
- Agent: 大规模 agentic task 合成 (1800 个环境, 85K prompt)

---

## 2. Method Overview

V3.2 = V3.1-Terminus + 新加 DSA + 重写 post-training

**没改的**:
- MLA, DeepSeekMoE, MTP, FP8 训练 — 全部继承 V3 / V3.1
- 模型 size 671B / 37B activated 没变

**改的**:
- 唯一架构改动: 用 continued training 的方式, 在 MLA 上面套一层 **DSA (DeepSeek Sparse Attention)**
- Post-training: specialist distillation + 大规模 GRPO + agentic synthesis

---

## 3. Key Designs

### 3.1 DSA: Lightning Indexer + Top-k Token Selection (核心创新)

**动机**: vanilla attention O(L²), MLA 已经压了 KV cache 但 attention compute 仍是 quadratic。要做真正的稀疏注意力, 必须有一个**便宜的索引器**告诉每个 query 该看哪些 key。

**Lightning Indexer**:
```
对每个 query token h_t 和它前面的 token h_s, 计算 index score:
  I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)

设计要点:
  - ReLU 而不是 softmax → throughput 高
  - indexer head 数量很小 → 可以全程 FP8
  - 整体算力远低于 MLA 主注意力
```

**Top-k Token Selection**:
```
对每个 query, 选 index score top-k 的 KV entries 做完整 attention:
  u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})

实际配置: k = 2048 个 KV token / query
```

**与 MLA 集成**: DSA 基于 MLA 的 **MQA mode** 实现 (不是 MHA mode), 这样每个 latent vector 在 query 所有 head 之间共享, kernel 层效率才能起飞。

**复杂度**: 主 attention 从 O(L²) 降到 O(Lk), indexer 仍 O(L²) 但常数极小, 长上下文端到端显著加速。

### 3.2 两阶段 continued pre-training (训 DSA 而不破坏旧能力)

从 V3.1-Terminus 的 128K base checkpoint 出发:

**Stage 1: Dense Warm-up (只训 indexer)**
- 冻结全部模型参数, 保持 dense attention
- 只训 lightning indexer
- 目标: 把 indexer 输出对齐到主 attention 的真实分布
  - 把所有 attention head score 加和 → L1 归一化得目标分布 p_t
  - Loss = KL(p_t || softmax(I_t))
- 学习率 1e-3, 只训 1000 步, 16 序列 × 128K = **2.1B tokens**

**Stage 2: Sparse Training (全开)**
- 引入 top-k selector, 全模型参数都训
- Indexer 仍用 KL 对齐, 但只在选中的 token 集上对齐
- **关键**: indexer 输入从计算图 detach — indexer 只受自己的 KL loss 监督, 主模型只受 LM loss 监督
- 学习率 7.3e-6, 训 15000 步, 480 序列 × 128K = **943.7B tokens**

### 3.3 Post-training: Specialist Distillation + Mixed RL

**Specialist Distillation**:
- 6 个专业方向 + 写作 + 通用 QA: 数学/编程/通用逻辑/通用 agent/agentic coding/agentic search
- 每个方向先单独训一个 specialist (从同一个 V3.2 base, 大规模 RL)
- 然后 specialist 生成数据回去喂给最终统一模型
- thinking 和 non-thinking 用不同模型生成数据
- 蒸馏后离 specialist 只差一点, 后续 RL 把差距抹平

**Mixed RL (GRPO)**:
- reasoning + agent + alignment 合并到**一个 RL 阶段** (不是多阶段, 避免灾难性遗忘)
- reasoning/agent 用 rule-based outcome reward + length penalty + language consistency reward
- general 用 generative reward model (每个 prompt 自带 rubric)

### 3.4 Agentic Synthesis Pipeline

为了把 reasoning 能力嫁接到 tool-use 上:

- **Cold-start**: 用 V3 方法把 reasoning 和 tool-use 拼成单条 trajectory, 系统 prompt 显式要求 "用 `<think>` 标签先推理再调工具"
- **Large-Scale**: 合成 **1800 个环境, 85,000 复杂 prompt**
  - Search agent: 多 agent pipeline 抽长尾实体 → 探索 → 生成 QA → 验证
  - Code agent: 从 GitHub 挖 issue/PR 对, 自动建可执行环境 (F2P 测试通过, P2F 测试不变), 覆盖 Python/Java/JS/TS/C/C++/Go/PHP
  - Code interpreter: Jupyter Notebook 解数学/逻辑/数据科学问题
- 同一个 RL 阶段内, agent reasoning 和普通 reasoning 一起训

### 3.5 Thinking 在 tool-call 中的保留机制

新设计: 当只追加工具调用结果而不是新用户消息时, **历史 reasoning 内容保留**, 只有新 user message 才清空 reasoning trace。这样 long-horizon agent 任务能保留思考连贯性。

注意: Roo Code / Terminus 这类把 tool 模拟成 user message 的框架不能享受这个增益, 推荐用 non-thinking 模式。

---

## 4. Experiments

### 4.1 V3.2 vs V3.1-Terminus (parity)

- Standard benchmarks (2025.09 评测): 短上下文和长上下文均无显著退化
- Human preference (ChatbotArena, 2025.11): Elo 持平
- 长上下文新基准: AA-LCR reasoning mode 比 V3.1-Terminus 高 4 分; Fiction.liveBench 多个指标超越

### 4.2 推理成本 (H800 集群, $2/GPU·h)

DSA 在 prefilling 和 decoding 上都显著降低 token cost, 长上下文增益最明显。论文给出曲线显示 token cost 随位置增加, V3.2 比 V3.1-Terminus 增长慢得多。

短序列 prefilling 单独实现了一个 **masked MHA mode 模拟 DSA**, 这种场景下效率甚至更高。

### 4.3 V3.2-Speciale (推理巅峰)

放宽 length penalty, 只训 reasoning 数据, 加入 DeepSeekMath-V2 的数据集和 reward:
- **IMO 2025 金牌**
- **IOI 2025 金牌**
- **ICPC World Final 2025 金牌**
- **CMO 2025 金牌**
- 推理表现与 Gemini-3.0-Pro 同档

### 4.4 V3.2 (生产版)

- 与 Kimi-K2-Thinking, GPT-5 在多个 reasoning benchmark 上接近
- Agent 任务 (mcpmark, mcpuniverse) 上显著缩小开源和闭源差距, 成本远低

---

## 5. Related Work Analysis

**与 NSA (Native Sparse Attention, Yuan et al. 2025) 的关系**: DSA 借鉴了 "key-value entry must be shared across multiple queries for kernel efficiency" 的设计原则 — 所以 DSA 必须基于 MLA 的 MQA mode 而不是 MHA mode。

**与传统 sparse attention 的区别**:
- 传统 sparse: 固定 pattern (window, stripe, BigBird)
- DSA: **数据驱动**, 由 lightning indexer 学到该看哪里
- 与 NSA 的区别: DSA 把 indexer 做得更轻 (ReLU + FP8 + 少量 head), 单独用 KL loss 对齐

**与 MoBA (Kimi) 的对比**: 都做"动态决定看哪个 chunk", MoBA 是 chunk 级 + top-k chunk, DSA 是 token 级 + top-k token, 粒度更细。

---

## 6. Limitations & Future Directions

论文/Speciale 的 caveat:
- 推理顶尖性能仍只在 Speciale 这个长 thinking 变体上达成, 生产模型为了延迟把 length penalty 加得更重
- DSA 仍依赖 lightning indexer 这个 O(L²) 步骤 (虽然常数小), 真正的 O(L) attention 还没出现
- 1800 个 agent 环境是大幅扩展, 但相比 production 部署的真实多样性还有差距

未明说但合理推论:
- DSA 的 top-k=2048 是固定的, 自适应 top-k 是潜在改进
- Indexer 用 KL 对齐 main attention, 长上下文区域里 main attention 本身就不准的时候, indexer 也学到错的目标

---

## 7. Paper vs Code Discrepancies

代码已开源:
- HuggingFace: deepseek-ai/DeepSeek-V3.2-Exp (Exp 版本) 和 deepseek-ai/DeepSeek-V3.2
- GitHub: deepseek-ai/DeepSeek-V3.2-Exp
- vLLM 集成: 2025.09.29 vLLM blog 有 fine-grained sparse attention 的 fused kernel 实现

**Exp vs 正式版**: 架构完全一样, 区别在 post-training 投入 (Exp 是早期实验, V3.2 是 GPT-5 级别 RL recipe 的产物)。

---

## 8. Cross-Paper Comparison

### V3.2 在 DeepSeek 自家系列中的位置

| 模型 | 关键创新 | 其他变化 |
| --- | --- | --- |
| V2 (24.05) | **MLA** + DeepSeekMoE | KV cache 压缩 93.3% |
| V3 (24.12) | FP8, MTP, aux-loss-free, $5.5M | 671B/37B |
| R1 (25.01) | Pure RL reasoning, R1-Zero | GRPO 在 LLM 上首战 |
| V3.1-Terminus | 128K context 扩展 | 中间产物 |
| **V3.2 (25.12)** | **DSA (lightning indexer + top-k)** | **post-training 算力 > 10% pretrain** |
| V3.2-Speciale | RL 推理特化, IMO/IOI/ICPC/CMO 金牌 | 同架构, 不同 RL recipe |

### V3.2 vs 其他长上下文 sparse attention

| 方案 | 团队 | 粒度 | 选择机制 | 复杂度 |
| --- | --- | --- | --- | --- |
| Vanilla attention | 任何人 | token | 全选 | O(L²) |
| MoBA | Kimi (24.10) | chunk | top-k chunk | O(L · √L) |
| NSA | DeepSeek (25) | block | learned router | O(L · k) |
| **DSA (V3.2)** | **DeepSeek** | **token** | **lightning indexer + top-k** | **O(L · k) attention + O(L²) 小常数 indexer** |

### 对机器人 FM 的启示

1. **Long-context attention 优化迁移**: 机器人长 horizon 视频/状态序列同样受 O(L²) 之苦, lightning indexer 模式可移植
2. **Post-training 算力分配**: 闭源 vs 开源差距很大程度来自后训, 机器人 FM 也需要类似规模的后训投入
3. **统一 RL 阶段**: 不要做多阶段 (灾难性遗忘), 用一个混合 RL + 不同 reward 类型

---

## 附: 核心数字速查

| 项 | 值 |
| --- | --- |
| 模型 size | 671B / 37B activated (继承 V3) |
| Context | 128K (continued from V3.1-Terminus) |
| DSA top-k | 2048 KV token / query |
| Lightning indexer 头数 | 少 (具体数论文未给), FP8 |
| Indexer activation | ReLU (非 softmax, 为 throughput) |
| Indexer 训练 loss | KL(main attention || softmax(indexer)) |
| Stage 1 warm-up | 2.1B tokens, lr 1e-3 |
| Stage 2 sparse train | 943.7B tokens, lr 7.3e-6 |
| Post-training 算力 | > 10% 预训练 |
| Agent 合成环境 | 1,800 |
| Agent 合成 prompt | 85,000 |
| RL 算法 | GRPO (多任务混合, 单阶段) |
| Speciale 成绩 | IMO/IOI/ICPC/CMO 2025 全部金牌 |
| 部署 GPU 价格 | $2/H800·h (论文用) |
