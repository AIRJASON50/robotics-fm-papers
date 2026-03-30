# Kimi-K2 MoE 精简笔记 -- 面向机器人研究者

**目的**: 只提取 MoE 架构相关内容, 帮助理解 pi_0 的 action expert 设计。不是完整论文分析。

---

## 1. MoE 核心概念

Mixture-of-Experts 的本质: **大模型里只激活一小部分参数做计算**。

```
Dense model (GPT-3):    175B 参数, 每个 token 都经过全部 175B 参数
MoE model (Kimi-K2):   1040B 参数, 但每个 token 只经过 32B 参数 (3%)
```

### 结构

标准 Transformer 的每一层: `Attention -> FFN (Feed-Forward Network)`

MoE 替换 FFN 部分:
```
标准:    Attention -> 1 个 FFN
MoE:     Attention -> Router -> 从 384 个 Expert FFN 中选 8 个激活
```

每个 Expert 就是一个独立的 FFN (2-layer MLP), 结构完全相同但**权重不同**。

### Router (门控网络)

Router 决定每个 token 分配给哪些 expert:
```
router_logits = Linear(hidden_state)  # [batch, num_experts]
top_k_experts = topk(router_logits, k=8)  # 选 8 个
weights = softmax(top_k_experts)  # 归一化权重
output = sum(weight_i * expert_i(input))  # 加权求和
```

### Kimi-K2 的具体配置

| 参数 | 值 | 说明 |
|------|------|------|
| Total params | 1040B | 总参数量 |
| Activated params | 32B | 每个 token 实际计算量 |
| Total experts | 384 | FFN expert 数量 |
| Activated experts | 8 | 每个 token 选中的 expert 数 |
| Shared experts | 1 | 所有 token 都经过的公共 expert |
| Sparsity | 48x | 384 / 8 = 48 |
| Hidden dim | 7168 | Transformer hidden size |
| Expert hidden dim | 2048 | 每个 Expert FFN 的 hidden size |
| Attention | MLA (Multi-head Latent Attention) | 类似 DeepSeek-V3 |

### Sparsity Scaling Law

Kimi-K2 的关键发现: **固定激活参数量 (FLOPs 不变), 增加总 expert 数量 (提高 sparsity) -> 性能持续提升**。

```
Sparsity  8  (8/64):   Loss 高
Sparsity 16  (8/128):  Loss 降
Sparsity 32  (8/256):  Loss 继续降
Sparsity 48  (8/384):  Loss 最低 (Kimi-K2 选择)
```

直觉: 更多 expert = 更细粒度的功能分工。不同 token 可以被路由到不同的"专家", 每个专家专注于更窄的功能。

---

## 2. 与 pi_0 Action Expert 的关联

pi_0 的架构本质上是一种 **hardcoded 2-expert MoE**:

```
pi_0 架构:
  VLM (PaliGemma 3B)        = Expert 1: 处理 vision + language tokens
  Action Expert (300M)       = Expert 2: 处理 action tokens

  Routing: 不用 learned router, 而是按 token 类型硬编码分配
    - image/text tokens -> VLM expert
    - action tokens     -> Action Expert
    - 两者通过 cross-attention 交互 (blockwise causal attention)
```

对比:

| 维度 | Kimi-K2 MoE | pi_0 dual-expert |
|------|-------------|-----------------|
| Expert 数量 | 384 个同构 FFN | 2 个异构模型 (VLM + Action) |
| Routing | Learned router (top-8) | Hardcoded by token type |
| Expert 大小 | 每个 ~2B | VLM 3B + Action 300M |
| 稀疏性 | 48x (只用 3% 参数) | 约 10x (action expert 是 VLM 的 1/10) |
| 训练 | 端到端, 所有 expert 一起训 | VLM frozen, 只训 action expert |
| 核心思想 | 同一类任务内的功能分工 | 跨模态的功能分离 (理解 vs 执行) |

**共同思想**: 不是所有参数都需要参与所有计算。让不同的参数组 (expert) 专注于不同的功能, 可以在不增加推理成本的前提下扩大模型容量。

### GR00T N1 的双系统也是 MoE 思想

```
GR00T N1:
  System 2 (VLM, 10Hz)   = "慢思考" expert: 理解指令和场景
  System 1 (DiT, 120Hz)  = "快执行" expert: 生成动作

  两者异步运行, System 2 提供条件, System 1 高频生成动作
```

---

## 3. 对机器人研究的启示

1. **Expert 分工降低推理成本**: pi_0 的 action expert 只有 300M 参数, 可以在 50Hz 实时运行。如果让整个 3B VLM 做 action generation, 推理太慢。MoE 思想让"理解"和"执行"解耦。

2. **Sparsity scaling law 可能适用于 robot VLA**: 如果 VLA 也能用更多 expert (比如不同 embodiment 或 task 用不同 expert), 可能在不增加推理成本的前提下提升泛化能力。这是 Open X-Embodiment 数据集带来的可能方向。

3. **Hardcoded routing vs learned routing**: pi_0 用 hardcoded routing (按 token 类型), 但未来可能会用 learned routing 让模型自动决定哪些 token 需要深度处理。这已经在 Octo 的 readout token 机制中有所体现。
