# DeepSeek 系列 -- 从量化基金到 LLM 架构创新引擎

> **阅读视角**: 本笔记的出发点是 **从 LLM 领域学习做机器人基础模型**。
> 关注: DeepSeek 的哪些架构创新和工程实践可以迁移到 robotics foundation model?

**覆盖论文** (主线):
- **DeepSeek LLM**: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism", arXiv:2401.02954, 2024.01
- **DeepSeekMoE**: "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models", arXiv:2401.06066, 2024.01
- **DeepSeek-V2**: "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model", arXiv:2405.04434, 2024.05
- **DeepSeek-Coder-V2**: "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence", arXiv:2406.11931, 2024.06
- **DeepSeek-V3**: "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.12
- **DeepSeek-R1**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948, 2025.01

**关键方法论文** (被 DeepSeek 系列依赖):
- **DeepSeek-Math**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, 2024.02 (GRPO 的原始来源)

**完整 arXiv 列表**:

| 模型 | arXiv | 年份 | 关键创新 |
|------|-------|------|---------|
| DeepSeek LLM (7B/67B) | 2401.02954 | 2024.01 | Dense baseline, scaling law 验证 |
| DeepSeekMoE (16B/145B) | 2401.06066 | 2024.01 | Fine-grained experts + shared expert isolation |
| DeepSeek-Math | 2402.03300 | 2024.02 | **GRPO** (Group Relative Policy Optimization) |
| DeepSeek-V2 (236B/21B) | 2405.04434 | 2024.05 | **MLA** + DeepSeekMoE |
| DeepSeek-Coder-V2 | 2406.11931 | 2024.06 | MoE 代码模型 |
| DeepSeek-V3 (671B/37B) | 2412.19437 | 2024.12 | FP8, MTP, aux-loss-free, $5.5M (~$5.576M) 训练 |
| DeepSeek-R1 (671B/37B) | 2501.12948 | 2025.01 | Pure RL reasoning, R1-Zero |

---

## 1. DeepSeek 与 LLM 发展的交织

### 1.1 创始人与公司背景

**梁文锋 (Liang Wenfeng)**, 浙江大学电子信息工程, 核心经历:
- **幻方量化 (High-Flyer)**: 2015 年创立, 中国头部量化对冲基金, 管理规模超百亿
- 幻方量化的核心优势: **GPU 集群**。量化交易需要大规模并行计算, 幻方很早就积累了数千张 A100/H800 的 GPU 资源
- **DeepSeek**: 2023 年从幻方内部孵化, 独立运营。创立逻辑: 已有 GPU 资源 + AI 是量化之后的下一个计算密集型机会

与其他中国 LLM 公司创始人的对比:

| 创始人 | 学术背景 | 技术 DNA | 公司优势 |
|--------|---------|---------|---------|
| 杨植麟 (Moonshot/Kimi) | 清华/CMU, Transformer-XL | 长上下文 | 学术积累, 注意力机制 |
| 阿里 Qwen 团队 | 达摩院 | 全家桶覆盖 | 阿里云基础设施 + 数据 |
| **梁文锋 (DeepSeek)** | 量化金融 | **效率至上, 工程优先** | **GPU 集群 + 低成本工程文化** |

梁文锋的量化金融背景深刻影响了 DeepSeek 的技术路线: **用最少的计算资源达到最好的效果**。这解释了为什么 DeepSeek 的每一代模型都在极致追求训练和推理效率 -- MLA 压缩 KV cache, MoE 稀疏激活, FP8 低精度训练, 都是 "用更少资源做更多事" 的体现。

### 1.2 DeepSeek 在全球 LLM 时间线中的位置

```
=== 全球 LLM 主线 ===

2017  Transformer (Google)
2018  GPT-1 (OpenAI)
2019  GPT-2 / Transformer-XL (杨植麟)
2020  GPT-3 / Scaling Laws (Kaplan) / GShard MoE (Google)
2021  Switch Transformer (Google) / Codex (OpenAI)
2022  InstructGPT -> ChatGPT (OpenAI) / Chinchilla (DeepMind)
2023.02  LLaMA (Meta) -- 开源大模型元年
2023.03  GPT-4 (OpenAI)
2023.07  Llama 2 (Meta)
2023.08  Qwen (Alibaba) 首发
2023     *** DeepSeek 成立, 开始内部开发 ***
2024.01  DeepSeek LLM 7B/67B + DeepSeekMoE  <-- Phase 1: Dense 基线 + MoE 探索
2024.02  DeepSeek-Math (GRPO 诞生)
2024.05  *** DeepSeek-V2 (MLA + MoE) ***    <-- Phase 2: 架构创新爆发
2024.06  DeepSeek-Coder-V2
2024.09  o1 (OpenAI, 推理模型)
2024.12  *** DeepSeek-V3 (FP8, $5.5M) ***   <-- Phase 2 巅峰
2025.01  *** DeepSeek-R1 (pure RL reasoning) ***  <-- Phase 3: 推理 + 开源基础设施
2025.02  开源 FlashMLA / DeepGEMM / 3FS / DualPipe / DeepEP / EPLB
```

**DeepSeek 的速度**: 从公司成立 (2023.11) 到发布 V3+R1 (2025.01) 只用了约 14 个月。相比之下, OpenAI 从 GPT-1 到 ChatGPT 用了 4 年多。这种速度来自两个因素: (1) 站在全行业的技术积累之上 (后发优势); (2) 幻方量化的 GPU 资源和工程文化。

### 1.3 DeepSeek 的技术借鉴图谱

DeepSeek 不是从零开始, 但其借鉴的比例远低于 Qwen 和 Kimi -- DeepSeek 的核心竞争力在于 **原创架构创新**:

| 借鉴的技术 | 来源 | 用在 DeepSeek 哪里 | 说明 |
|-----------|------|----------------|------|
| Transformer decoder-only | GPT-1 (OpenAI, 2018) | 所有模型 | 架构基础 |
| SwiGLU activation | PaLM (Google, 2022) | 所有模型 | FFN 激活函数 |
| RMSNorm (pre-norm) | GPT-2/LLaMA | 所有模型 | 训练稳定性 |
| RoPE | Su et al. (2021) | 所有模型 | 位置编码 (但 MLA 中需特殊处理) |
| MoE 基本思想 | GShard/Switch Transformer (Google) | DeepSeekMoE+ | 稀疏激活基础 |
| PPO 框架 | Schulman et al. (2017) | GRPO 的改进基础 | RL 策略优化 |
| YaRN | 开源社区 | V2/V3 长上下文 | 上下文扩展 |
| Flash Attention | Dao et al. (2022) | 训练加速 | 注意力计算优化 |

**DeepSeek 的原创贡献** (非借鉴, 是 DeepSeek 自己的发明):

| 创新 | 项目 | 意义 |
|------|------|------|
| **MLA** (Multi-head Latent Attention) | V2 (2024.05) | KV cache 压缩 93.3%, 吞吐量 5.76x, **被 Kimi K2 直接采用** |
| **Fine-grained expert segmentation** | DeepSeekMoE (2024.01) | 将 N 个专家拆成 mN 个更细粒度专家, 更灵活组合 |
| **Shared expert isolation** | DeepSeekMoE (2024.01) | 隔离共享专家捕获公共知识, 减少路由专家冗余 |
| **GRPO** (Group Relative Policy Optimization) | DeepSeek-Math (2024.02) | 去掉 critic model, 降低 RL 成本, **被 R1/V3 广泛使用** |
| **Auxiliary-loss-free load balancing** | V3 (2024.12) | 用 bias term 动态调整路由, 不加 auxiliary loss |
| **MTP** (Multi-Token Prediction) | V3 (2024.12) | 训练时预测多个未来 token, 提升数据效率 |
| **FP8 混合精度训练** (671B 规模) | V3 (2024.12) | 首次在超大模型上验证 FP8 训练可行性 |
| **DualPipe** | V3 (2024.12) | pipeline 并行中计算-通信重叠, 几乎零 bubble |
| **R1-Zero** (pure RL, no SFT) | R1 (2025.01) | 纯 RL 训练涌现推理能力, 无需人类示范 |
| **大规模 RL 蒸馏** | R1 (2025.01) | 671B RL 模型蒸馏到 1.5B-70B 小模型 |

**关键观察**: DeepSeek 的原创贡献集中在两个方向 -- **推理效率** (MLA, MoE, FP8) 和 **训练方法** (GRPO, MTP, R1-Zero)。这两个方向恰好是 robotics foundation model 最需要的: 机器人端推理需要极致效率, 机器人训练数据稀缺需要高效训练方法。

---

## 2. 技术演进: 三个阶段

```
=== Phase 1: Dense 基线 + MoE 探索 (2024 Q1) ===

DeepSeek LLM (2024.01): Dense 基线
  |  7B / 67B, 2T tokens
  |  验证 scaling law, 建立训练基础设施
  |  (意义: 不急于创新, 先把 dense baseline 做扎实)
  v
DeepSeekMoE (2024.01): MoE 架构创新
  |  16B (2.8B 激活) / 145B
  |  Fine-grained expert segmentation + shared expert isolation
  |  16B 匹配 LLaMA2 7B (只用 40% 计算)
  |  145B 匹配 DeepSeek 67B (只用 28.5% 计算)
  v
DeepSeek-Math (2024.02): GRPO 诞生
  |  在数学领域验证 GRPO 的有效性
  |  (GRPO 后来成为 V2/V3/R1 的标配 RL 算法)

=== Phase 2: MoE + MLA 架构创新 (2024 H1-H2) ===

DeepSeek-V2 (2024.05): *** DeepSeek 的里程碑 ***
  |  236B 总参 / 21B 激活, 128K 上下文
  |  MLA: KV cache 压缩 93.3%, 吞吐量 5.76x
  |  DeepSeekMoE: 细粒度专家 + 共享专家
  |  训练成本比 DeepSeek 67B 节省 42.5%
  v
DeepSeek-Coder-V2 (2024.06): 代码专项
  |  MoE 架构, 开源代码模型 SOTA
  v
DeepSeek-V3 (2024.12): *** 效率奇迹 ***
  |  671B 总参 / 37B 激活, 128K 上下文
  |  新增: FP8 训练 + MTP + aux-loss-free load balancing
  |  14.8T tokens, 2048 H800 GPUs
  |  总训练成本: 2.788M H800 GPU hours = $5.576M
  |  性能匹配 GPT-4o 和 Claude-3.5-Sonnet
  |  全程零 loss spike, 零 rollback

=== Phase 3: 推理能力 + 开源基础设施 (2025) ===

DeepSeek-R1 (2025.01): *** 推理模型 ***
  |  基于 V3 架构 (671B/37B)
  |  R1-Zero: 纯 RL, 无 SFT, 推理能力自发涌现
  |  R1: 多阶段训练 (冷启动 SFT -> RL -> rejection sampling -> RL)
  |  蒸馏: R1 -> 1.5B/7B/8B/14B/32B/70B 小模型
  v
开源基础设施 (2025.02): *** 全栈开源 ***
  |  FlashMLA: MLA 的高效 GPU kernel
  |  DeepGEMM: FP8 矩阵乘法库
  |  3FS (Fire-Flyer File System): 分布式文件系统
  |  DualPipe: pipeline 并行框架
  |  DeepEP: expert parallelism 通信库
  |  EPLB: expert-parallel load balancer
```

---

## 3. 核心架构创新 (技术深度)

### 3.1 MLA (Multi-head Latent Attention) -- DeepSeek 最具影响力的创新

**问题**: 标准 Multi-Head Attention (MHA) 在推理时需要缓存所有 token 的 Key 和 Value, 即 KV cache。对于长上下文和大模型, KV cache 是显存瓶颈:

```
MHA KV cache 大小 = 2 * n_h * d_h * seq_len * num_layers
                  (K 和 V 各一份, 每个头 d_h 维, 所有头 n_h 个)

DeepSeek 67B (MHA): 128 heads * 128 dim = 16384 per token per layer (K+V)
```

**已有方案及其问题**:

| 方案 | 做法 | KV cache 压缩 | 代价 |
|------|------|-------------|------|
| MQA (Multi-Query Attention) | 所有 head 共享 1 组 KV | 最大 | 性能下降明显 |
| GQA (Grouped-Query Attention) | 多个 head 共享 1 组 KV (分组) | 中等 | 性能略有下降 |
| **MLA** | **低秩联合压缩 KV 到潜在向量** | **93.3%** | **性能不降反升** |

**MLA 的核心思想: 低秩 KV 联合压缩**

标准 MHA 中, 每个 token 需要缓存完整的 K 和 V 向量 (维度 = n_h * d_h)。MLA 的关键观察: K 和 V 中存在大量冗余, 可以用一个低维潜在向量 (latent vector) 来联合表示:

```
=== 标准 MHA ===
h_t -> W^K -> k_t (n_h * d_h 维)    缓存 k_t
h_t -> W^V -> v_t (n_h * d_h 维)    缓存 v_t
KV cache per token: 2 * n_h * d_h

=== MLA ===
h_t -> W^DKV -> c_t^KV (d_c 维, d_c << n_h * d_h)    只缓存 c_t^KV
推理时:
c_t^KV -> W^UK -> k_t^C (n_h * d_h 维)   (可被吸收到 W^Q 中, 不需要实际计算)
c_t^KV -> W^UV -> v_t^C (n_h * d_h 维)   (可被吸收到 W^O 中, 不需要实际计算)
KV cache per token: d_c  (+ d_h^R for decoupled RoPE key)
```

**数学形式**:

```
Down-projection (压缩):
  c_t^KV = W^DKV * h_t          # h_t in R^d -> c_t^KV in R^d_c,  d_c << d

Up-projection (解压缩, 可在推理时被吸收):
  k_t^C = W^UK * c_t^KV         # c_t^KV in R^d_c -> k_t^C in R^(n_h * d_h)
  v_t^C = W^UV * c_t^KV         # c_t^KV in R^d_c -> v_t^C in R^(n_h * d_h)

Query 同样做低秩压缩 (减少训练时的 activation memory):
  c_t^Q = W^DQ * h_t            # h_t in R^d -> c_t^Q in R^d_c'
  q_t^C = W^UQ * c_t^Q          # c_t^Q in R^d_c' -> q_t^C in R^(n_h * d_h)
```

**为什么 W^UK 可以被 "吸收"**: 在注意力计算中, Q 和 K 的交互是 `q_t^T * k_j`。如果 k_j = W^UK * c_j^KV, 那么 `q_t^T * W^UK * c_j^KV = (W^UK^T * q_t)^T * c_j^KV`。因此我们可以预计算 `W^UK^T * q_t` (将 W^UK 吸收到 query 变换中), 推理时只需要缓存 c_j^KV 而不需要解压缩回完整的 K。

**RoPE 兼容性问题与解决**: RoPE (Rotary Position Embedding) 是位置敏感的, 会在 Q 和 K 之间插入一个位置相关的旋转矩阵。这打破了上述 "吸收" 的条件 (矩阵乘法不满足交换律)。DeepSeek 的解决方案是 **Decoupled RoPE**: 额外分配一组小的 query 和 shared key 专门携带 RoPE, 与 MLA 的低秩压缩部分拼接:

```
Decoupled RoPE:
  k_t^R = RoPE(W^KR * h_t)      # 额外的小维度 key 携带位置信息
  q_t^R = RoPE(W^QR * c_t^Q)    # 额外的小维度 query 携带位置信息

最终:
  q_t = [q_t^C; q_t^R]          # concatenate
  k_t = [k_t^C; k_t^R]          # concatenate

KV cache = c_t^KV (d_c 维) + k_t^R (d_h^R 维)
```

**DeepSeek-V2 的实际数字**:

| | MHA (DeepSeek 67B) | MLA (DeepSeek-V2) | 压缩比 |
|---|---|---|---|
| KV cache per token | 2 * 128 * 128 = 32768 | 512 + 64 = 576 | **~98.2% 压缩** |
| 最大生成吞吐量 | 基准 | **5.76x** | - |
| MMLU 性能 | 参考 | **更好** | 不降反升 |

> **Note**: 93.3% 是论文报告的 V2 完整模型配置下的压缩比 (考虑了多头数量、RoPE 维度等全局因素)。上表的 per-token 简化计算仅为示意, 实际压缩比以论文数据为准。

**对 Robotics 的意义**: VLA (Vision-Language-Action) 模型在机器人上做推理时, KV cache 是关键瓶颈。以 OpenVLA 为例, 7B 模型在 Jetson 上推理需要大量显存用于 KV cache。MLA 可以将 KV cache 压缩到原来的 ~7%, 直接使得大模型在机器人端部署成为可能。

### 3.2 DeepSeekMoE -- 细粒度专家分割 + 共享专家隔离

**问题**: 传统 MoE (如 GShard, Switch Transformer) 使用 8-16 个粗粒度专家, 存在两个问题:
1. **Knowledge Hybridity (知识混杂)**: 少量专家意味着每个专家需要处理多种类型的知识, 难以专精
2. **Knowledge Redundancy (知识冗余)**: 不同专家独立学习, 共性知识被重复存储

**DeepSeekMoE 的双重解法**:

```
=== 传统 MoE (GShard) ===
N=8 experts, top-K=2 activated
每个 expert: FFN intermediate_dim = 4096
组合数: C(8,2) = 28 种

=== DeepSeekMoE ===
策略 1 -- Fine-grained segmentation: 拆成 mN 个更小 expert, 激活 mK 个
  m=4: 8 experts -> 32 experts, top-2 -> top-8
  每个 expert: FFN intermediate_dim = 4096/4 = 1024
  总计算量不变, 但组合数: C(32,8) = 10,518,300 种 -- 灵活性爆炸式增长

策略 2 -- Shared expert isolation: 固定 K_s 个 expert 始终激活, 捕获公共知识
  共享 expert 负责所有 token 都需要的通用知识 (如语法, 常识)
  路由 expert 只需学习特化知识, 减少冗余
```

**DeepSeek-V2/V3 中的 MoE 配置**:

| | DeepSeek-V2 | DeepSeek-V3 |
|---|---|---|
| 总参数 | 236B | **671B** |
| 激活参数 | 21B | **37B** |
| 路由专家数 N_r | 160 | **256** |
| 共享专家数 N_s | 2 | 1 |
| 每 token 激活路由专家 K_r | 6 | **8** |
| 路由函数 | Softmax | **Sigmoid** (V3 改用) |

**对 Robotics 的意义**: Multi-task robot policy 天然适合 MoE 架构 -- 不同的操作技能 (抓取、放置、旋转) 可以由不同专家负责, 共享专家处理通用的视觉理解和运动基元。DeepSeekMoE 的 fine-grained segmentation 使得技能组合更灵活, 适合机器人面对的组合爆炸式任务空间。

### 3.3 GRPO (Group Relative Policy Optimization) -- 去掉 Critic 的 RL

**问题**: PPO (Proximal Policy Optimization) 是 LLM 后训练 (RLHF) 的标准 RL 算法, 但它需要一个 critic model (value function) 来估计 baseline。对于 LLM 级别的模型, 这个 critic model 本身就是一个大模型, 训练和推理成本高。

**GRPO 的核心思想**: 不用 critic model, 而是用 **组内相对排名** 作为 baseline:

```
=== PPO ===
对于每个 (question, answer) pair:
  advantage = reward - V(state)    # V 是 critic model 的输出
  需要训练一个与 policy 同等大小的 critic model

=== GRPO ===
对于每个 question q:
  采样一组回答 {o_1, o_2, ..., o_G}
  计算每个回答的 reward {r_1, r_2, ..., r_G}
  advantage_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)  # 组内标准化
  不需要 critic model!
```

**GRPO 目标函数**:

```
J_GRPO(theta) = E[q ~ P(Q), {o_i} ~ pi_old(O|q)]
  (1/G) * sum_i [
    min(
      (pi_theta(o_i|q) / pi_old(o_i|q)) * A_i,
      clip(pi_theta(o_i|q) / pi_old(o_i|q), 1-eps, 1+eps) * A_i
    )
    - beta * D_KL(pi_theta || pi_ref)
  ]

where A_i = (r_i - mean(r)) / std(r)  # group relative advantage
```

**为什么 GRPO 有效**: 关键 insight 是, 对于 LLM 推理任务 (数学、代码), reward 是 **确定性的** (答案对不对可以自动验证)。在同一个问题上采样多个回答, 对的得高分、错的得低分, 组内排名就能提供有效的 advantage 信号。不需要 critic model 去估计 "这个状态有多好"。

**GRPO 的影响力**: 首次出现在 DeepSeek-Math (2024.02), 随后被用于:
- DeepSeek-V2 的 RL alignment
- DeepSeek-V3 的 post-training
- DeepSeek-R1/R1-Zero 的推理训练
- Kimi k1.5 也独立使用了类似思路 (outcome-based RL, 但 k1.5 用的不是 GRPO 而是 PPO 变体)
- QwQ (Qwen) 的推理训练

**对 Robotics 的意义**: 机器人 RL 中, 训练 critic/value function 同样昂贵。GRPO 的 "无 critic" 思路可以直接迁移: 在 sim 中对同一任务采样多条轨迹, 用任务成功率作为 reward, 组内排名作为 advantage。这省去了训练 value function 的成本, 对 robot policy RL 训练可能是一个重要加速。

### 3.4 FP8 混合精度训练 (DeepSeek-V3)

**问题**: 大模型训练通常使用 BF16/FP16 精度。671B 模型在 2048 张 H800 上训练, 显存和计算都是极限挑战。

**DeepSeek-V3 的 FP8 方案**:

```
=== 混合精度策略 ===
                     前向             反向
Linear layers:       FP8 compute      FP8 compute
                     FP8 storage      FP8 storage
Attention:           BF16/FP32        BF16/FP32
Embedding:           BF16             BF16
Output head:         BF16             BF16
Optimizer states:    FP32             FP32
Master weights:      FP32             FP32

核心: GEMM (矩阵乘法) 用 FP8, 其他用 BF16/FP32
```

**关键技术细节**:
1. **Fine-grained quantization**: 不是整个 tensor 用一个 scale, 而是 tile-wise (128x128 块) 量化, 减少量化误差
2. **提升精度的策略**: FP8 乘法 -> FP32 累加 (累加器精度不降), 部分关键层保持高精度
3. **通信也用低精度**: activation 和 gradient 的 all-to-all 通信使用 FP8, 减少通信带宽

**效果**: 671B 模型全训练 (pre-training + post-training) 只需 2.788M H800 GPU hours, 按 $2/GPU hour 计算 = **$5.576M**。对比: Llama 3 405B 的训练成本估计在 $60-100M 量级。DeepSeek-V3 用约 1/10 的成本达到了同级别性能。

**对 Robotics 的意义**: Robot policy 训练数据来自 sim 或 demo, 获取成本高。FP8 训练如果能直接应用于 VLA 等机器人模型, 可以在相同 GPU 预算下训练更大模型或更多步数。DeepSeek 开源的 DeepGEMM 库提供了 FP8 矩阵乘法的现成实现。

### 3.5 Auxiliary-Loss-Free Load Balancing (DeepSeek-V3)

**问题**: MoE 模型的经典难题 -- 路由不均衡 (routing collapse)。某些专家被过度使用, 某些被闲置。传统方案是加 auxiliary loss 鼓励均衡, 但 auxiliary loss 与主 loss 冲突, 损害模型性能。

**DeepSeek-V3 的方案**: 用动态 bias 调整路由, 不加额外 loss:

```
传统方案: loss = loss_main + alpha * loss_balance
  问题: alpha 太大损害性能, alpha 太小不均衡

DeepSeek-V3: 不改 loss, 改路由分数
  对每个 expert i, 维护一个 bias term b_i
  路由决策: top-K 从 {s_{i,t} + b_i} 中选择
  gating value: 仍然用原始 s_{i,t} (不含 bias)

  每个训练 step 结束后更新 bias:
    if expert_i overloaded: b_i -= gamma
    if expert_i underloaded: b_i += gamma
  gamma 是 bias update speed 超参数
```

**精巧之处**: bias 只影响路由选择 (哪些 expert 被激活), 不影响 gating value (expert 输出的加权系数)。这样不会改变梯度信号, 模型仍然按照最优方向训练, 只是通过 bias 把 "看哪些 expert" 调得更均衡。

### 3.6 Multi-Token Prediction (MTP)

**问题**: 标准 next-token prediction 每个位置只预测下一个 token, 训练信号密度低。

**DeepSeek-V3 的 MTP**:

```
标准:  position i -> 预测 t_{i+1}                (1 个训练信号)
MTP:   position i -> 预测 t_{i+1}, t_{i+2}, ..., t_{i+D}  (D 个训练信号)

实现: D 个顺序 MTP module, 每个包含:
  - 共享 embedding layer (与主模型共享)
  - 共享 output head (与主模型共享)
  - 独立 Transformer block
  - 独立 projection matrix

第 k 个 MTP module:
  input = concat(h_i^{k-1}, Emb(t_{i+k}))  # 上层输出 + 下个 token embedding
  -> Linear projection -> Transformer block -> output head -> predict t_{i+k+1}

训练时 MTP 提供额外 loss, 推理时 MTP module 可丢弃 (或用于 speculative decoding)
```

**MTP 的两个好处**:
1. **训练信号更密集**: 每个 position 提供 D+1 个训练信号而非 1 个, 提升数据效率
2. **Pre-planning**: 模型学会 "提前规划" 后续 token, 提升表征质量
3. **推理加速 (可选)**: MTP module 可用于 speculative decoding, 加速生成

### 3.7 R1-Zero: 纯 RL 涌现推理能力

**问题**: 传统思路认为, 让 LLM 学会推理需要: (1) 先用人类写的 CoT (Chain-of-Thought) 数据做 SFT; (2) 再用 RL 强化。人类写的 CoT 成本高, 且可能限制模型发展出更好的推理策略。

**R1-Zero 的激进做法**: 完全跳过 SFT, 直接从 V3 base model 开始做 RL:

```
=== R1-Zero 训练流程 ===

Base model: DeepSeek-V3-Base (671B/37B, 仅做过 pre-training, 未做任何 SFT/RLHF)

Prompt template:
  "A conversation between User and Assistant. The assistant first thinks about
   the reasoning process in the mind and then provides the user with the answer.
   The reasoning process and answer are enclosed within <think>...</think>
   and <answer>...</answer> tags."

RL algorithm: GRPO
Reward: rule-based
  - accuracy reward: 答案是否正确 (数学答案可自动验证)
  - format reward: 是否使用了 <think> 和 <answer> tag

关键: 没有任何人类推理示范, 没有 SFT, 没有 process reward model
```

**R1-Zero 训练中涌现的行为**:

| 训练阶段 | 涌现行为 | 说明 |
|---------|--------|------|
| 早期 (0-2K steps) | 尝试直接给答案 | reward 低, 被 RL 抑制 |
| 中期 (2K-8K steps) | 开始生成推理步骤, 长度递增 | RL 发现 "多想" 能提高正确率 |
| **"Aha Moment"** (~8K steps) | **自发出现 self-verification** | 模型学会 "Wait, let me re-evaluate..." |
| 后期 (8K+ steps) | 长推理链 + 多次验证 + 尝试替代方案 | 推理质量稳定提升 |

**R1-Zero 的 "Aha Moment"** (论文 Table 2): 模型在训练过程中出现了标志性的 self-reflection -- 在解题中途突然停下来说 "Wait, wait. Wait. That's an aha moment I can flag here.", 然后重新开始用不同方法验证。这种行为完全不是人类教的, 是 RL 奖励信号自然驱动涌现的。

**R1-Zero 的局限**: 虽然推理能力很强, 但存在可读性问题 (语言混杂) 和通用性问题 (只擅长可验证的推理任务)。因此 DeepSeek 又训练了 R1:

```
=== R1 训练流程 (完整版) ===

Stage 1: Cold-start SFT
  用少量高质量 CoT 数据做 SFT, 给模型一个推理格式的起点
  (注: 这些数据来自 R1-Zero 的好样本, 不是人工写的)

Stage 2: RL (reasoning-focused)
  GRPO + rule-based reward
  在数学/代码/科学推理任务上强化

Stage 3: Rejection Sampling + SFT
  用 Stage 2 的模型生成大量推理轨迹
  过滤出高质量的 -> 与通用 SFT 数据混合 -> 再做一轮 SFT
  (目的: 在保持推理能力的同时恢复通用能力)

Stage 4: RL (general)
  在通用任务上做 RL, 确保 helpfulness + safety
```

**R1 蒸馏**: R1 (671B) 的推理能力可以蒸馏到小模型:
- 用 R1 生成大量 (question, CoT, answer) 三元组
- 用这些数据对小模型 (1.5B ~ 70B) 做 SFT
- 结果: DeepSeek-R1-Distill-Qwen-32B 在 AIME 上达到 72.6%, 超过 OpenAI o1-mini

**对 Robotics 的意义**: R1-Zero 证明了 **纯 RL 可以涌现复杂策略, 不需要人类示范**。这对机器人极其重要 -- 机器人遥操作数据获取成本极高。如果 robot policy 也能通过纯 RL (在 simulation 中) 涌现出复杂操作策略, 就不再需要大量人类示范数据。R1-Zero 的 "aha moment" 也暗示: **给 RL 足够的探索空间 (长上下文/长 episode) + 正确的 reward 信号, 复杂行为可以自发涌现**。

---

## 4. 对 Robotics Foundation Model 的影响

### 4.1 直接可迁移的技术

| DeepSeek 技术 | Robotics 应用场景 | 预期收益 | 迁移难度 |
|-------------|----------------|--------|---------|
| **MLA** | VLA 模型推理时的 KV cache 压缩 | 显存降 ~93%, 支持在机器人端 (Jetson) 部署大模型 | 中 (需改 attention 实现) |
| **DeepSeekMoE** | Multi-task robot policy | 不同 expert 负责不同技能, 稀疏激活降低推理成本 | 中 (MoE routing 需适配连续动作空间) |
| **GRPO** | Robot policy RL 训练 | 去掉 value function 训练成本, 简化 RL pipeline | **低** (直接可用, 最易迁移) |
| **FP8 训练** | VLA/Diffusion Policy 训练加速 | ~2x 训练效率 (同等 GPU, 训更多步或更大模型) | 低 (DeepGEMM 已开源) |
| **R1-Zero (pure RL)** | Sim-to-real policy 训练 | 不依赖人类示范, 纯 sim RL 涌现操作技能 | 高 (需要好的 sim + reward) |
| **Aux-loss-free balancing** | MoE robot policy 的 expert 均衡 | 避免 auxiliary loss 干扰 policy 训练 | 低 |
| **MTP** | Robot action prediction | 一次预测多步动作 (类似 action chunking in ACT/DiffusionPolicy) | 中 |

### 4.2 具体迁移路径分析

**MLA for VLA inference memory**:
- OpenVLA (7B) 在 Jetson Orin 上推理时, KV cache 占显存约 2-4 GB (取决于上下文长度)
- 如果 VLA 模型采用 MLA, KV cache 可压缩到 ~200 MB
- 这意味着: 同样的硬件上可以跑更大的模型, 或者支持更长的观测历史
- **实际路径**: 下一代 VLA 模型 (如 pi_0 后续版本) 如果采用 MLA, 机器人端部署的显存瓶颈将大幅缓解

**GRPO for robot RL**:
- 当前 robot RL 的标准做法: PPO + value function (e.g., IsaacGym + PPO)
- GRPO 的替代方案: 对同一 task 采样 G 条轨迹, 用 task success rate 作为 reward, 组内排名作为 advantage
- **优势**: 不需要训练 value network, 对于高维观测空间 (image-based) 的 robot policy 尤其有价值 (value function 在高维空间中本身就很难训好)
- **限制**: GRPO 需要同一 question 多次采样, 在 real robot 上不现实; 但在 sim 中完全可行

**MTP 与 Action Chunking 的对应关系**:
- LLM 的 MTP: 每个 position 预测未来 D 个 token
- Robot policy 的 action chunking (ACT, Diffusion Policy): 每个 timestep 预测未来 H 步 action
- 两者的 insight 相同: **预测多步可以提供更密集的训练信号, 且使模型学会 "提前规划"**
- DeepSeek-V3 的 MTP 实现 (顺序预测, 保持因果链) 可能比 ACT 的并行预测更合理

### 4.3 开源基础设施对 Robotics 社区的价值

DeepSeek 2025.02 开源的基础设施工具, 对机器人 AI 训练有直接价值:

| 工具 | 用途 | 对 Robotics 的意义 |
|------|------|----------------|
| **FlashMLA** | MLA 的高效 CUDA kernel | 如果 VLA 采用 MLA, 这是现成的推理加速库 |
| **DeepGEMM** | FP8 矩阵乘法 | 机器人模型训练直接可用, 提升 GPU 利用率 |
| **3FS** | 分布式文件系统 | 大规模 robot demo 数据存储 (Open X-Embodiment 级别) |
| **DualPipe** | Pipeline 并行训练框架 | 多卡训练大型 VLA 模型 |
| **DeepEP** | Expert parallelism 通信 | MoE robot policy 的分布式训练 |
| **EPLB** | Expert-parallel load balancer | MoE robot policy 的 expert 均衡 |

---

## 5. 借鉴图谱: 谁借鉴了 DeepSeek

DeepSeek 是中国 LLM 生态中 **被借鉴最多** 的公司。与 Qwen 的 "被用作 base model" 不同, DeepSeek 的影响主要是 **架构和方法层面**:

```
DeepSeek-V2 (2024.05): 发明 MLA + DeepSeekMoE
  |
  +---> Kimi K2 (2025.07): 直接采用 MLA + MoE, 1T 总参 / 32B 激活
  |       (K2 paper 明确致谢 DeepSeek-V2/V3 架构)
  |
  +---> Moonlight (2025.02): 使用 DeepSeek-V3-small 架构做 Muon 优化器验证
  |       (Moonshot AI 选择 V3 架构, 说明其在 MoE+MLA 上的认可)
  |
  +---> Qwen2 (2024.06): 采用 MoE (但用 GQA 而非 MLA)
          (Qwen 在 MoE 上跟进, 但注意力机制走了不同路线)

DeepSeek-Math (2024.02): 发明 GRPO
  |
  +---> DeepSeek-R1 (2025.01): GRPO 作为核心 RL 算法
  +---> Kimi k1.5 (2025.01): 使用类似的 outcome-based RL (非直接 GRPO 但思路一致)
  +---> QwQ (2025.03): outcome-based RL for reasoning (GRPO 的思想影响)
  +---> 开源社区: 大量 GRPO 实现用于各种 reasoning 模型训练

DeepSeek-R1 (2025.01): 发明 R1-Zero + 蒸馏范式
  |
  +---> 社区涌现大量 "蒸馏 R1" 的小模型 (用 R1 数据 SFT 小模型)
  +---> DeepSeek-V3 post-training: 蒸馏 R1 的推理能力到 V3

DeepSeek-V3 (2024.12): FP8 训练 + aux-loss-free balancing + MTP
  |
  +---> 影响全行业对低精度训练的态度: 671B 都能 FP8 训, 还有什么不能?
  +---> Kimi K2 (2025.07): 受 aux-loss-free 启发 (K2 用了类似的 load balancing 策略)
```

**DeepSeek 在技术影响力上的独特地位**: Qwen 是 "被用作 base model" (Kimi-Audio 用 Qwen2.5-7B), Llama 是 "被 fine-tune" (大量衍生模型), 而 DeepSeek 是 **"被学习架构设计"**。MLA 和 GRPO 已经成为 LLM 社区的公共知识, 被多家公司采纳。

---

## 6. 商业逻辑: 效率至上的开源策略

### 6.1 从幻方量化到 DeepSeek

```
Phase 0 -- GPU 积累 (2015-2022): 幻方量化
  量化交易需要大规模 GPU 并行计算
  幻方积累了数千张 A100 的 GPU 集群
  (当时没有 AI 公司有这么多 GPU)

Phase 1 -- 试水入场 (2023-2024 Q1): Dense 基线
  DeepSeek LLM 7B/67B: 验证 "我们也能训大模型"
  DeepSeekMoE: 开始探索效率路线
  成本: 幻方的 GPU 资源, 边际成本极低

Phase 2 -- 架构创新爆发 (2024 H1-H2): MLA + MoE + GRPO
  DeepSeek-V2: 用 MLA 一举解决推理效率问题
  DeepSeek-V3: $5.5M 训练成本的效率奇迹
  (对比: Llama 3 训练成本 ~$60-100M)
  策略: 不拼数据规模 (Qwen 36T), 拼架构和工程效率

Phase 3 -- 推理 + 全栈开源 (2025): R1 + 基础设施
  R1: 推理模型, 引爆全球关注
  全栈开源: FlashMLA, DeepGEMM, 3FS, DualPipe, DeepEP, EPLB
  策略: 不只开源模型, 还开源训练基础设施
```

### 6.2 $5.5M 训练 V3 的效率奇迹

DeepSeek-V3 的训练成本拆解:

| 阶段 | H800 GPU hours | 成本 ($2/hr) |
|------|---------------|-------------|
| Pre-training (14.8T tokens) | 2,664K | $5.328M |
| Context extension (32K -> 128K) | 119K | $0.238M |
| Post-training (SFT + RL) | 5K | $0.01M |
| **Total** | **2,788K** | **$5.576M** |

**关键效率来源**:

1. **MoE 稀疏激活**: 671B 总参但每个 token 只激活 37B (~5.5%), 训练 FLOPs 远低于同等大小的 dense 模型
2. **FP8 训练**: ~2x 计算效率 vs BF16
3. **DualPipe**: pipeline 并行中几乎零 bubble, GPU 利用率极高
4. **No token dropping**: 由于 aux-loss-free 均衡策略, 不需要丢弃任何 token, 数据利用率 100%
5. **零 loss spike**: 训练全程稳定, 不需要 checkpoint rollback (这在大模型训练中很罕见)

**与 Llama 3 的成本对比**:

| | DeepSeek-V3 | Llama 3 405B | GPT-4 (估计) |
|---|---|---|---|
| 总参数 | 671B | 405B | ~1.8T (MoE, 传言) |
| 激活参数 | 37B | 405B (dense) | ~200B (传言) |
| 训练 tokens | 14.8T | 15T | ~13T (传言) |
| GPU hours | 2.788M (H800) | ~30.8M (H100, 估计) | 未知 |
| 训练成本 | **~$5.5M** | **~$60-100M** | **>$100M** |
| 性能 | 匹配 GPT-4o | 低于 GPT-4o | 基准 |

DeepSeek 用约 1/10 到 1/20 的成本, 训练出了性能相当的模型。这不是偶然, 而是 MLA + MoE + FP8 + DualPipe 四重效率优化的叠加效果。

### 6.3 开源策略: 为什么全部开源

DeepSeek 的开源程度在行业中最高:

| 开源内容 | DeepSeek | Qwen | Kimi | OpenAI | Meta (Llama) |
|---------|----------|------|------|--------|-------------|
| 模型权重 | MIT | Apache 2.0 | Apache 2.0 (K2) | 不开源 | Llama 许可 |
| 训练代码 | 部分开源 | 不开源 | 不开源 | 不开源 | 不开源 |
| 基础设施 | **全栈开源** | 不开源 | 不开源 | 不开源 | 不开源 |
| 论文细节 | **极其详细** | 详细 | 详细 | 越来越少 | 中等 |

**为什么 DeepSeek 要全栈开源**: 不同于 Qwen (阿里云商业闭环) 或 Kimi (产品导向), DeepSeek 的核心竞争力在于 **持续创新能力**, 而非某一个模型。开源当前最好的模型和基础设施:
1. 吸引人才 (技术实力展示)
2. 建立社区影响力 (GRPO, MLA 成为行业标准)
3. 推动硬件厂商适配 (FP8 生态)
4. 竞争对手即使复制也追不上 (DeepSeek 已在做下一代)

### 6.4 DeepSeek 的风险

1. **GPU 供应**: 受地缘政治影响, H800 (已限制) 和后续 GPU 获取可能受限
2. **人才竞争**: 核心架构创新依赖少数关键人才, 被挖角风险高
3. **商业化不足**: API 服务收入有限, 没有像 Qwen 那样的阿里云闭环或像 Kimi 那样的 C 端产品
4. **算力天花板**: 幻方的 GPU 资源虽然多, 但与 Microsoft (给 OpenAI) 或 Meta 的资源相比仍有量级差距

---

## 7. 三大中国 LLM 路线对比: DeepSeek vs Qwen vs Kimi

| 维度 | DeepSeek | Qwen (Alibaba) | Kimi (Moonshot) |
|------|----------|----------------|-----------------|
| 创始人背景 | 梁文锋 (量化基金) | 阿里达摩院 | 杨植麟 (Transformer-XL) |
| 核心 DNA | **架构创新 + 极致效率** | **全家桶 + 数据规模** | **长上下文 + 训练效率** |
| 架构路线 | **MLA + MoE (原创)** | GQA + MoE (跟随) | MLA + MoE (借鉴 DeepSeek) |
| 注意力机制 | **MLA (原创)** | GQA / M-RoPE (原创) | MLA (借鉴) + MoBA (原创) |
| MoE 设计 | **Fine-grained + Shared experts (原创)** | 标准 MoE | DeepSeek 架构 |
| RL 算法 | **GRPO (原创)** | Outcome-based RL | PPO 变体 + Partial rollouts (原创) |
| 优化器 | AdamW + FP8 | AdamW | **Muon/MuonClip (原创)** |
| 训练效率创新 | **FP8 + DualPipe (原创)** | 数据 self-improvement | **Muon 2x 效率 (原创)** |
| 训练数据量 | 14.8T (V3) | **36T** (Qwen3, 最大) | 15.5T (K2) |
| 训练成本 | **$5.5M (V3, 最低)** | 未公开, 估计 $50M+ | 未公开 |
| 推理模型 | **R1/R1-Zero (纯 RL, 原创)** | QwQ -> Qwen3 thinking | k1.5 (partial rollouts, 原创) |
| 开源程度 | **MIT + 全栈基础设施 (最高)** | Apache 2.0 | Apache 2.0 (K2) |
| 模态覆盖 | 文本/代码/数学 | 文本/视觉/音频/Omni/代码/数学 **(最全)** | 文本/视觉/音频 |
| 商业依托 | API + 幻方 GPU | 阿里云 | Kimi Chat 产品 |

**三者的分工互补**:
- **DeepSeek**: 架构创新引擎, 发明了被其他公司广泛采用的 MLA, MoE, GRPO
- **Qwen**: 全家桶覆盖, 数据规模最大, 许可最开放, 是社区的默认 base model
- **Kimi**: 注意力效率 (MoBA) 和优化器 (Muon) 的创新者, 在 agentic 方向最激进

---

## 8. GPT 为起点的技术分岔: DeepSeek 的位置

```
GPT-1/2 (2018-2019): 定义技术原点
  |  Transformer decoder + autoregressive + next-token prediction
  |
  +=====================================================+
  |              分岔 1: 怎么做大 (Scale)                |
  +=====================================================+
  |
  +---> [GPT 路线] 暴力 Dense scale
  |     GPT-3 (175B) -> Llama (7B-405B) -> Qwen (0.5B-72B Dense)
  |
  +---> [MoE 路线] 稀疏激活
        GShard (2020) -> Switch Transformer (2021) -> Mixtral (2023)
            |
            +---> *** DeepSeekMoE (2024.01): fine-grained experts ***
            |         |
            |         +---> DeepSeek-V2/V3: MoE 256 experts
            |         +---> Kimi K2: 384 experts (借鉴)
            |         +---> Qwen3 MoE: 128 experts (跟进)
            |
            +---> [Qwen 路线] 标准 MoE + 大数据
                  Qwen2-MoE -> Qwen3-MoE (235B-A22B)

  +=====================================================+
  |         分岔 2: 怎么降推理成本 (Inference)           |
  +=====================================================+
  |
  +---> [GQA 路线] 减少 KV head 数量
  |     MQA (Shazeer) -> GQA (Llama 2) -> Qwen 全系列
  |     简单有效, 但压缩比有限
  |
  +---> *** [MLA 路线] 低秩联合压缩 KV ***
  |     DeepSeek-V2 (2024.05): 发明 MLA, 93.3% KV 压缩
  |         +---> DeepSeek-V3: 继续使用
  |         +---> Kimi K2 / Moonlight: 直接采用
  |
  +---> [线性注意力] 替换 softmax
        Mamba / RWKV / Gated DeltaNet
            +---> Qwen3.5 (2026): 混合 (75% 线性 + 25% softmax)

  +=====================================================+
  |         分岔 3: 怎么做 RL (Post-training)            |
  +=====================================================+
  |
  +---> [RLHF 路线] PPO + critic model
  |     InstructGPT (2022) -> ChatGPT -> 全行业标准
  |     成本: 需要训练 critic model
  |
  +---> *** [GRPO 路线] 无 critic, 组内排名 ***
  |     DeepSeek-Math (2024.02): 发明 GRPO
  |         +---> R1/R1-Zero: GRPO 用于推理训练
  |         +---> V2/V3: GRPO 用于 alignment
  |         +---> 社区广泛采用
  |
  +---> [DPO 路线] 去掉 RL, 直接优化偏好
        Rafailov (2023) -> Qwen2.5+/Llama 3

  +=====================================================+
  |         分岔 4: 推理能力 (Reasoning)                 |
  +=====================================================+
  |
  +---> [o1 路线] 推理模型 (方法未公开)
  |     OpenAI o1 (2024.09): 闭源, 方法不明
  |
  +---> *** [R1 路线] 纯 RL 涌现推理 ***
  |     R1-Zero: 无 SFT, 纯 GRPO, 推理自发涌现
  |     R1: 冷启动 SFT + RL + 蒸馏
  |         +---> R1 蒸馏到 Qwen/Llama 小模型
  |
  +---> [k1.5 路线] 长上下文 RL
        Kimi k1.5: 128K 上下文 RL + partial rollouts

  +=====================================================+
  |         分岔 5: 训练效率 (Training Efficiency)       |
  +=====================================================+
  |
  +---> [数据效率] Chinchilla -> Qwen self-improvement
  |
  +---> [优化器效率] Muon/MuonClip (Kimi, 2x)
  |
  +---> *** [计算精度] FP8 训练 ***
        DeepSeek-V3: 首次在 671B 规模验证 FP8
        (+ DualPipe pipeline 并行, 近零 bubble)
```

---

## 9. 对 Robotics Foundation Model 的综合启示

### 9.1 DeepSeek 技术在 Robotics 中的应用优先级

按 "预期收益 / 迁移难度" 排序:

| 优先级 | 技术 | 预期收益 | 迁移难度 | 建议 |
|--------|------|--------|--------|------|
| **1 (最高)** | GRPO | 去掉 value function, 简化 robot RL | 低 | sim 中直接替换 PPO 试验 |
| **2** | FP8 训练 (DeepGEMM) | 训练效率 ~2x | 低 | VLA 训练直接集成 |
| **3** | MLA | 推理显存降 93% | 中 | 下一代 VLA 架构考虑采用 |
| **4** | MoE for multi-task | 不同 expert 负责不同技能 | 中 | 多任务 policy 训练试验 |
| **5** | MTP / action chunking | 多步预测提升规划能力 | 中 | 已在 DiffusionPolicy 中验证 |
| **6** | R1-Zero (pure RL) | 不依赖人类 demo | 高 | 需要好的 sim + reward 设计 |

### 9.2 DeepSeek 给 Robotics 的最大启示

**效率至上**: DeepSeek 用 $5.5M 训练出匹配 GPT-4o 的模型, 核心在于 "四重效率叠加" (MoE + MLA + FP8 + DualPipe)。对 Robotics 的启示:

1. **不是 "更大的模型 + 更多的数据" 就能赢**: DeepSeek-V3 (14.8T tokens) 胜过 Qwen2.5 (18T tokens), 证明架构效率比数据规模更重要。对 robot policy 同理 -- 与其收集更多 demo 数据, 不如先优化模型架构和训练效率。

2. **稀疏 > 密集**: MoE 用 671B 参数但只激活 37B, 性能匹配 dense 405B。Robot policy 也应该走 MoE 路线 -- 不同操作技能用不同 expert, 而不是用一个巨大的 dense policy 处理所有任务。

3. **RL 可以涌现复杂行为**: R1-Zero 从未见过人类推理示范, 却自发涌现了 self-verification 和 reflection。这对 robot manipulation 的暗示: **sim 中的纯 RL (无人类 demo) 可能涌现出人类未想到的操作策略**, 只要 reward 设计得当且训练时间足够长。

---

## 10. 阅读建议

| 目标 | 推荐阅读顺序 |
|------|-----------|
| 理解 MLA | V2 paper Section 2.1 (MLA 公式推导) -> V3 paper Section 2.1.1 (简化版) |
| 理解 MoE | DeepSeekMoE paper (fine-grained + shared) -> V3 paper Section 2.1.2 (aux-loss-free) |
| 理解 GRPO | DeepSeek-Math paper Section 3 -> R1 paper Section 2.1 |
| 理解 R1-Zero | R1 paper Section 2 (重点看 Table 2 "Aha Moment") |
| 理解训练效率 | V3 paper Section 3 (FP8 + DualPipe + 通信优化) |
| 理解商业逻辑 | V3 paper Table 1 (成本拆解) -> 与 Llama 3 对比 |
| 理解 MTP | V3 paper Section 2.2 (对比 Gloeckle et al. 2024 的原始 MTP) |
| 理解蒸馏 | R1 paper Section 3 (multi-stage pipeline) |
| 对比行业格局 | 本文 Section 7 -> kimi_series_notes.md -> qwen_series_notes.md |
| Robotics 迁移 | 本文 Section 4 + Section 9 -> pi0_notes.md -> GR00T_N1_notes.md |
