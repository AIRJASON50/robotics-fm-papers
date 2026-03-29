# Chinchilla: Training Compute-Optimal Large Language Models

**Paper**: Hoffmann et al., DeepMind, 2022 (NeurIPS), arXiv:2203.15556

---

## 1. Core Problem

Kaplan et al. (2020) 的 scaling law 建议: 在固定 compute budget 下, 应优先增大模型参数量,
数据量的增长可以相对滞后 (roughly N^{0.74} scaling for parameters vs D^{0.54} for data).
这导致了 GPT-3 (175B params, 300B tokens) 和 Gopher (280B params, 300B tokens) 等模型
**严重 undertrained** -- 参数量极大但训练数据相对不足.

Chinchilla 的核心问题: **给定固定的 training compute budget C, model size N 和 training
tokens D 应如何最优分配?**

## 2. Method Overview

作者通过三种互补的方法估计 compute-optimal scaling:

- **Approach 1 -- Fix model, vary tokens**: 训练超过 400 个不同大小的模型 (70M - 16B params),
  每个在 4 种不同的 token 数量下训练, 拟合 loss 曲线找最优点
- **Approach 2 -- IsoFLOP profiles**: 固定 9 个不同的 FLOP budget (从 6e18 到 3e21),
  在每个 budget 下扫描不同的 model size, 找到最优 N
- **Approach 3 -- Parametric loss fitting**: 直接拟合 L(N, D) = E + A/N^alpha + B/D^beta
  的参数化公式, 联合估计所有参数

三种方法得到一致结论: **model size 和 training tokens 应等比例增长**.

## 3. Key Designs

### 3.1 Corrected Scaling Law

Chinchilla 的核心发现:

| | Kaplan et al. (2020) | Chinchilla (2022) |
|--|----------------------|-------------------|
| N 与 D 的 scaling 关系 | N 增长更快, D 增长慢 | **N 和 D 应等比例增长** |
| 对 Gopher (280B) 的建议 | 用 300B tokens 即可 | 应用 ~5.6T tokens |

具体地, 对于 compute budget C, 最优配置满足:
- N_opt proportional to C^{0.50}
- D_opt proportional to C^{0.50}

即 compute 翻倍时, model size 和 data 各增长 ~sqrt(2) 倍.

### 3.2 Chinchilla 模型 (70B, 1.4T tokens)

为验证 scaling law, 作者训练了 Chinchilla:
- **70B parameters** (Gopher 的 1/4)
- **1.4T training tokens** (Gopher 的 ~4.7x)
- 与 Gopher 使用**相同的 compute budget**

关键: 更小的模型 + 更多的数据 = 更好的性能.

### 3.3 Loss Parametric Form

提出的 loss 公式 L(N, D) = E + A/N^alpha + B/D^beta 中:
- E 代表 irreducible loss (数据本身的 entropy)
- A/N^alpha 代表 model size 不足带来的欠拟合
- B/D^beta 代表 data 不足带来的欠拟合
- 三种方法拟合出的 alpha ~ 0.34, beta ~ 0.28

## 4. Experiments

### 4.1 Language Modeling Benchmarks

Chinchilla (70B) vs Gopher (280B), 使用相同 compute:

| Benchmark | Gopher 280B | Chinchilla 70B |
|-----------|-------------|----------------|
| MMLU (5-shot) | 60.0% | **67.6%** |
| BIG-bench (avg) | 较低 | **显著更高** |
| HellaSwag | 79.2% | **80.8%** |
| LAMBADA | 74.5% | **77.4%** |

Chinchilla 在几乎所有 benchmark 上都超过了 4 倍参数量的 Gopher.

### 4.2 Downstream Tasks

- 在 MMLU 上首次超过人类 average performance (67.6% vs ~65%)
- 在 reading comprehension, common sense reasoning 等任务上全面超越 Gopher, GPT-3, Megatron-Turing NLG 等

### 4.3 Inference Efficiency

Chinchilla 因为模型更小 (70B vs 280B):
- Inference 速度显著更快
- 部署成本大幅降低
- Fine-tuning 更容易

## 5. Impact

### 5.1 对 LLM 训练范式的根本改变

Chinchilla 之前, 社区共识是 "bigger model is better" (GPT-3 -> PaLM -> etc.),
普遍用 ~300B tokens 训练. Chinchilla 后:

- **LLaMA (Meta, 2023)**: 7B model 用 1T tokens, 65B model 用 1.4T tokens,
  直接引用 Chinchilla scaling law
- **LLaMA 2**: 进一步增加到 2T tokens
- 几乎所有后续 LLM 都遵循 "equal scaling" 原则, 大幅增加训练数据量

### 5.2 数据成为瓶颈

Chinchilla 间接推动了:
- 高质量 training data 的收集 (RedPajama, FineWeb, etc.)
- Data curation / filtering 技术的发展
- 关于 "is data scaling hitting a wall?" 的讨论

### 5.3 Compute-Optimal 思想的扩展

Chinchilla scaling law 的思路被扩展到:
- Vision models (如 ViT scaling)
- Multimodal models
- 后续关于 over-training 的研究 (故意超过 Chinchilla-optimal 训练, 换取更小更快的 inference 模型)
