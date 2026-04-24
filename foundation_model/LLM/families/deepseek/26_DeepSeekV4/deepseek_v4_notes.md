# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

> 技术报告 PDF (无 arxiv): `DeepSeek_V4.pdf`, DeepSeek-AI, 2026.04.24 发布
> Preview 版本: V4-Pro / V4-Flash (双 size, 同时开源)
> License: MIT (代码 + 权重)
> HuggingFace: deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Flash

---

## 1. Core Problem

reasoning model 范式 (test-time scaling) 让性能持续提升, 但被 vanilla attention 的 O(L²) 卡死。同时 long-horizon agentic workflow 和大规模跨文档分析对 ultra-long context 的需求暴增。

V3.2 的 DSA 已经是 sparse attention 突破, 但还不够 — V4 的目标是把 **1M token context 推到"日常可用"**, 让 long-horizon task 和 test-time scaling 真的能 routinely 跑。

主要技术目标:
1. 1M context 下 inference FLOPs 和 KV cache 都要数量级下降
2. 训练稳定性能撑得住 mHC 这种新结构
3. RL post-training 能扩展到 million-token context

---

## 2. Method Overview

V4 = V3 架构骨架 + 4 个核心升级:

| # | 升级 | 解决什么 | 替代什么 |
| --- | --- | --- | --- |
| 1 | **mHC** (Manifold-Constrained Hyper-Connections) | 残差连接表达力和稳定性 | 普通 residual connection |
| 2 | **Hybrid Attention (CSA + HCA)** | 长上下文 attention 效率 | V3.2 的 DSA |
| 3 | **Muon Optimizer** | 收敛速度和训练稳定性 | AdamW (大部分模块) |
| 4 | **FP4 quantization-aware training** | 训练算力和显存 | FP8 (部分模块) |

继承 V3 的: DeepSeekMoE, MTP, MLA 思路 (但 attention 全换), Transformer 主干。

**两个 size**:
- **DeepSeek-V4-Pro**: 1.6T total / 49B activated
- **DeepSeek-V4-Flash**: 284B total / 13B activated

两个都原生支持 1M context。

---

## 3. Key Designs

### 3.1 Hybrid Attention: CSA + HCA (核心创新)

V4 把 attention 层分成两类, 交错堆叠 (interleaved hybrid):

#### CSA (Compressed Sparse Attention)
**思路**: 先压缩 + 再稀疏 (V3.2 DSA 的升级版)。

- **Token-Level Compressor**: 每 m 个 KV token 压缩成 1 个 entry
  - 计算 `C^a = H · W^aKV`, `C^b = H · W^bKV`, 同时学压缩权重 `Z^a, Z^b`
  - 每个压缩 entry 由 2m 个原 entry 组成, 但相邻压缩块之间有重叠 (实际压缩比 1/m)
- **Lightning Indexer + Top-k**: 在压缩后的 KV entry 上跑 DSA
  - Indexer query 走 low-rank: `c^Q_t = h_t · W^DQ`, `q^I = c^Q · W^IUQ`
  - Indexer score 仍是 ReLU 形式: `I_{t,s} = Σ w · ReLU(q^I · K^IComp)`
  - top-k 选择压缩后的 KV entry
- **Shared Key-Value MQA**: 选中的压缩 entry 同时充当 key 和 value, MQA 模式跑 core attention
- **Sliding Window Branch**: 额外保留最近 n_win 个未压缩 token, 增强局部依赖
- **Grouped Output Projection**: 输出 head 分 g 组, 先各自降维到 d_g 再拼接, 避免 cn_h → d 的巨大 projection

#### HCA (Heavily Compressed Attention)
**思路**: 极端压缩, 但保持 dense attention (不做稀疏选择)。

- 压缩比 m' ≫ m (远大于 CSA 的压缩比)
- 不做 top-k 选择, 直接对所有压缩 entry 做 attention
- 同样有 sliding window 分支保留局部信息
- 共享同一套 MQA + grouped output projection 机制

#### 为什么要混合?
- CSA: **细粒度 + 稀疏选择** → 适合需要精准定位远端 token 的层
- HCA: **极致压缩 + dense** → 适合粗粒度全局聚合的层
- 交错使用让模型既有"放大镜"又有"广角镜", 长上下文 attention 总成本断崖下降

#### 其他细节
- **Partial RoPE**: 只在最后 64 维加 RoPE; 因为 KV entry 既是 K 也是 V, naive 输出会带 absolute pos embedding, 所以 output 上也加 RoPE with position −i 来对消
- **Attention Sink**: 加可学的 sink logit z' (类似 OpenAI / StreamingLLM trick), 让每 head 的 attention 总和可以不为 1
- **Mixed Precision**: KV entry 中 RoPE 维度 BF16, 其余 FP8; **lightning indexer 全程 FP4** → KV cache 比纯 BF16 减半

#### 1M context 效率数字
| 项 | V3.2 baseline | V4-Pro | V4-Flash |
| --- | --- | --- | --- |
| Single-token FLOPs | 1.0 | **27%** | 10% |
| KV cache size | 1.0 | **10%** | 7% |
| 相对 BF16 GQA8 baseline 的 KV cache | — | ~2% | ~2% |

### 3.2 mHC: Manifold-Constrained Hyper-Connections

**HC (Hyper-Connections)**: Xie 等 2026 提出, 把残差流宽度从 d 扩到 n_hc × d, 用三个映射 A_l (输入), B_l (残差变换), C_l (输出) 控制信号传递:
```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

**HC 的问题**: 多层堆叠时数值不稳, scaling 上不去。

**mHC 的核心创新**: 把残差变换矩阵 B_l 约束在 **doubly stochastic matrices 流形 (Birkhoff polytope) M** 上:
```
B_l ∈ M := {M ∈ R^{n×n} | M·1_n = 1_n, 1_n^T·M = 1_n^T, M ≥ 0}
```

**为什么这能稳定训练**:
- B_l 是 doubly stochastic → 谱范数 ‖B_l‖_2 ≤ 1 → 残差变换 non-expansive → 前向/反向都稳
- 集合 M 在矩阵乘法下封闭 → 多层堆叠仍稳
- A_l, C_l 用 sigmoid 约束 non-negative + bounded, 防止信号互消

**Dynamic Parameterization**:
- A_l, B_l, C_l 都拆成 dynamic (input-dependent) 和 static (input-independent) 两部分
- 输入先 RMSNorm + flatten 成 1×n_hc·d, 走线性映射得到原始参数
- 用 **Sinkhorn-Knopp 算法** (20 次迭代) 把 B_l 投影到 doubly stochastic manifold

**计算开销**: n_hc 远小于 hidden size d, 所以 mHC 引入的额外开销很小。

### 3.3 Muon Optimizer

**为什么换 Muon**:
- 收敛更快 (Liu et al. 2025 验证)
- 训练更稳

**实现细节**:
- **AdamW 仍保留**给: embedding, prediction head, mHC 的 static bias 和 gating, 所有 RMSNorm 模块
- **其余全换 Muon**
- 用 **Hybrid Newton-Schulz iteration** 做正交化 (10 步, 分两阶段):
  - 前 8 步: (a, b, c) = (3.4445, -4.7750, 2.0315) — 快速收敛, 把奇异值拉到 1 附近
  - 后 2 步: (a, b, c) = (2, -1.5, 0.5) — 稳住奇异值在 1
- 不用 QK-Clip — V4 的 attention 架构允许直接对 query 和 KV entry 做 RMSNorm, 自然防 logit 爆炸

### 3.4 FP4 Quantization-Aware Training

- **MoE expert 权重**: FP4
- **Lightning indexer QK 路径**: FP4
- **大部分其他参数**: FP8 (继承 V3)
- 在当前硬件上 FP4 × FP8 的 peak FLOPs 和 FP8 × FP8 一样, 但**未来硬件理论上能再快 1/3**

### 3.5 DeepSeekMoE 改动

- **Affinity score 激活函数**: 从 Sigmoid → **Sqrt(Softplus)**
- 仍用 auxiliary-loss-free balancing + 序列级轻 balance loss
- **去掉 routing target node 数量限制**, 重新设计 parallelism 策略
- **前几层 dense FFN 替换为 Hash Routing MoE** (Roller et al. 2021): 按 token ID 走预定义 hash 函数决定专家, deterministic 无路由开销

### 3.6 MTP

继承 V3, 不变。

### 3.7 训练 Infrastructure

- 单 fused kernel for MoE: 完全 overlap compute / communication / memory access
- **TileLang DSL**: 平衡开发效率和 kernel 性能
- **Batch-invariant + deterministic kernel libraries**: train-inference bitwise reproducibility
- **FP4 quantization-aware training** 全栈支持
- **Hybrid ZeRO** for Muon optimizer
- **Cost-effective mHC** via recomputation + fused kernels
- **Two-stage contextual parallelism** for compressed attention

### 3.8 Inference Infrastructure

- **Heterogeneous KV cache structure**: 支持高效 shared-prefix 复用
- **On-disk KV cache storage**: 长 prefix 落盘

---

## 4. Experiments

### 4.1 Pre-training scale
- V4-Flash 训 32T tokens
- V4-Pro 训 33T tokens
- 内部评测: V4-Flash-Base 在多数 benchmark 超过 V3.2-Base; V4-Pro-Base 进一步刷新, 在推理/编程/长上下文/世界知识全面领先

### 4.2 Post-training Pipeline (两阶段)

**Stage 1: Specialist Training**
- 每个目标域 (数学/编程/agent/instruction following) 独立训一个 expert
- Base model 先 SFT 获得领域基础能力
- 然后用 GRPO + 任务特定 reward model 做 RL

**Stage 2: On-Policy Distillation**
- 单一统一模型作为 student, 用 reverse KL loss 蒸馏所有 specialist
- 得到一个能在所有领域都工作的统一 V4

### 4.3 三种推理模式

| 模式 | 特点 | 适用场景 |
| --- | --- | --- |
| **Non-Think** | 快速直觉 | routine task, 低风险决策 |
| **Think High** | 主动逻辑分析 | 复杂问题求解, 规划 |
| **Think Max** | 最深推理 | 能力边界探索, 需要 ≥ 384K context |

### 4.4 Base 模型 benchmark (V4-Pro-Base)

| Benchmark | Score |
| --- | --- |
| MMLU-Pro (EM) | **73.5** |
| HumanEval (Pass@1) | **76.8** |
| GSM8K (EM) | **92.6** |
| MATH (EM) | **64.5** |
| LongBench-V2 (EM) | **51.5** |
| TriviaQA (EM) | **85.6** |

### 4.5 Instruct (V4-Pro-Max) vs frontier 模型

| Benchmark | V4-Pro-Max | 注 |
| --- | --- | --- |
| MMLU-Pro | 87.5 | 接近 Gemini-3.1-Pro (91.0) |
| LiveCodeBench | **93.5** | 对比模型中最佳 |
| Codeforces Rating | **3206** | 编程最强 |
| IMOAnswerBench | 89.8 | frontier 同档 |
| SWE Verified | 80.6 | 与 Gemini-3.1-Pro 持平 |
| MRCR 1M (MMR) | 83.5 | 1M 长上下文 |
| CorpusQA 1M (Acc) | 62.0 | 1M 长上下文 |

### 4.6 与 Closed-Source 的关系

- **Knowledge**: V4-Pro-Max 在 SimpleQA / Chinese-SimpleQA 显著超过领先开源, 在教育型 (MMLU-Pro/HLE/GPQA) 仅小幅领先开源, 显著缩小与 Gemini-3.1-Pro 的差距
- **Reasoning**: 标准 reasoning 上超 GPT-5.2 和 Gemini-3.0-Pro, 但落后 GPT-5.4 和 Gemini-3.1-Pro 约 3-6 个月发展轨迹
- **Agent**: 公开 benchmark 与 Kimi-K2.6 / GLM-5.1 持平, 内部评测 outperforms Claude Sonnet 4.5, 接近 Opus 4.5 水平
- **Long Context**: V4-Pro-Max 在 1M context 学术 benchmark 上**超过 Gemini-3.1-Pro**

### 4.7 V4-Pro vs V4-Flash
- Knowledge: Flash 因参数少明显弱
- Reasoning: 给 Flash 更多 thinking budget 可以追平
- Agent: 简单任务 Flash 持平, 复杂高难度任务 Flash 仍不如 Pro

---

## 5. Related Work Analysis

### 与 V3.2 DSA 的关系
- DSA (V3.2): 在原始 KV 上做 lightning indexer + top-k 选择, attention 复杂度 O(L·k)
- **CSA (V4)**: 先把 m 个 KV 压成 1 个, 再在压缩后的 KV 上做 DSA → **二级稀疏**
- **HCA (V4)**: 极致压缩 + dense attention, 与 CSA 互补

### 与其他 sparse / linear attention
- **Mamba / Linear attention**: 完全 O(L), 但表达力弱; V4 选择"重压缩 + 高效稀疏"路线保留 attention 表达力
- **NSA (DeepSeek 25)**: native sparse, 但 V4 已经超越它走到 hybrid compressed
- **MoBA (Kimi)**: chunk 级 top-k, V4 的 CSA 是 token 级, 粒度更细

### mHC 的来源
- HC (Zhu et al. 2025): 提出残差宽度扩展但不稳
- mHC (Xie et al. 2026): doubly stochastic 约束让 HC 真正能 scale; **DeepSeek 共同作者 (Liang Wenfeng 亲自上传)**

### Muon Optimizer
- Jordan et al. 2024 提出; Liu et al. 2025 (Kimi K2 论文) 在 LLM 上验证。V4 第一次在 trillion 级模型上用 Muon。

---

## 6. Limitations & Future Directions

论文 §6 写的:
- V4-Pro-Max reasoning 仍落后最新 frontier 3-6 个月
- Knowledge benchmark 上还差 Gemini-3.1-Pro
- Preview 版本: 后续会有更完整的 release

可推论但论文未明说:
- 1M context 已经实现, 但 RL 中真实在 1M 上跑还是稀少 (RL infra 章节有 Million-Token RL Framework)
- mHC 的 Sinkhorn-Knopp 20 次迭代是经验值, 自适应迭代次数可探索
- Muon × 1.6T 模型是首次, 长期稳定性还需观察

---

## 7. Paper vs Code Discrepancies

V4 是 **Preview 版本**, 论文标题就写 "We present a preview version of DeepSeek-V4 series"。

代码和权重均已开源 (MIT):
- HF: deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Flash
- 推理实现: HF repo 内 `inference/` 目录
- 注意: **没提供 Jinja chat template**, 必须用 `encoding/` 目录的 Python 脚本编码 message

**版本对应**:
- V4-Pro-Base / V4-Flash-Base: 仅预训练
- V4-Pro / V4-Flash: 加 post-training
- V4-Pro-Max / V4-Flash-Max: Think Max 模式 (Pro 需 ≥384K context)

---

## 8. Cross-Paper Comparison

### V4 在 DeepSeek 自家系列中的位置

| 模型 | Total / Active | 关键架构 | 关键算法 | 长上下文 |
| --- | --- | --- | --- | --- |
| V2 (24.05) | 236B / 21B | MLA, DeepSeekMoE | — | 128K |
| V3 (24.12) | 671B / 37B | + FP8, MTP, aux-loss-free | — | 128K |
| R1 (25.01) | 671B / 37B | 同 V3 | **Pure RL reasoning, GRPO** | 128K |
| V3.2 (25.12) | 671B / 37B | + **DSA** | + agentic synthesis | 128K |
| **V4-Flash (26.04)** | **284B / 13B** | **+ CSA/HCA, mHC, Muon, FP4** | **+ specialist + on-policy distill** | **1M** |
| **V4-Pro (26.04)** | **1.6T / 49B** | 同上 | 同上 | **1M** |

**关键转折点**:
- V2: KV cache 革命 (MLA)
- V3: 训练效率革命 (FP8, $5.5M)
- R1: 推理范式革命 (RL reasoning)
- V3.2: 长上下文计算革命 (DSA)
- **V4: 长上下文存储革命 (CSA/HCA, KV cache 降到 baseline 的 ~2%)**

### V4 vs Kimi K2 / 其他开源
- Kimi K2: 用了 DeepSeek 的 MLA + Muon, 但没有 mHC 也没有 hybrid attention
- GLM-5.1, MiniMax-M2: 都还在 V3.2 量级的架构基础上
- V4 是开源里第一个走完 **MoE + sparse attention + 压缩 attention + mHC + Muon + FP4** 全栈升级

### 对机器人 FM 的启示

| V4 创新 | 是否可移植到 robot FM | 怎么用 |
| --- | --- | --- |
| Hybrid CSA + HCA | **可以** | robot 长 horizon 视频/状态序列同样面临 O(L²), 用 CSA 做精确定位, HCA 做粗粒度全局摘要 |
| mHC | **可以** | 大模型残差稳定性是通用问题, doubly stochastic 约束可移植到 VLA backbone |
| Muon | **可以** | LLM 已验证, robot FM 可直接换 |
| FP4 QAT | **可以** | 边缘部署 (Jetson) 显存吃紧, FP4 是天然解 |
| Specialist + on-policy distill | **可以** | 类似 pi_0.7 的 "蒸馏 specialist 回 generalist" 思路 |
| Two-mode reasoning (think / non-think) | **更有意义** | 机器人有"反应式"和"规划式"两种行为, 自然对应两种推理模式 |

---

## 附: 核心数字速查

| 项 | V4-Flash | V4-Pro |
| --- | --- | --- |
| Total params | 284B | **1.6T** |
| Activated params | 13B | **49B** |
| Context | 1M | 1M |
| Pre-train tokens | 32T | 33T |
| KV cache vs V3.2 (1M) | **7%** | **10%** |
| Single-token FLOPs vs V3.2 (1M) | **10%** | **27%** |
| KV cache vs BF16 GQA8 baseline (1M) | ~2% | ~2% |
| Precision | FP4 (MoE expert + indexer) + FP8 (大部分) + BF16 (RoPE 维度) |
| Optimizer | Muon (主) + AdamW (embedding/MTP/RMSNorm/mHC bias) |
| Newton-Schulz iter | 10 (8 fast + 2 stable) |
| MoE balancing | aux-loss-free + sequence-wise light balance loss |
| MoE 前几层 | **Hash Routing** (deterministic by token ID) |
| MoE affinity activation | **Sqrt(Softplus)** (从 V3 的 Sigmoid 改) |
| mHC Sinkhorn-Knopp iter | 20 |
| Post-training | Specialist (SFT+GRPO) → On-Policy Distillation |
| Three modes | Non-Think / Think High / Think Max |
| License | MIT |
