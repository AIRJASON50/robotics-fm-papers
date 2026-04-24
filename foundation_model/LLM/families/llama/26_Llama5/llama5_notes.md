# Llama 5 (2026.04.08)

> **状态**: 仅有 Meta blog + Zuckerberg 视频公告, **尚无 arxiv 技术报告**
> **License**: 开源, 开放权重 (具体许可待 Meta 公布)
> **意义**: Meta 跳过 Llama 4.5, 直接发布 Llama 5, 押注开源对抗 frontier 闭源

---

## 1. 公告关键信息

### 1.1 模型规模
- **总参数: 600B+** (dense, **没采用 MoE**)
- 这与 Llama 4 (Scout 109B-A17B / Maverick 400B+-A17B) 的 MoE 路线相反, **回到 dense**
- 推断动机: dense 训练更稳, RL 后训更可控

### 1.2 上下文窗口
- **5M tokens** (5 million)
- 远超同期 frontier (DeepSeek V4 1M, Gemini 1M, Claude 200K)
- 主打企业级 long-context 应用 (整套代码库, 跨年合同, 长 video transcript)

### 1.3 训练基础设施
- **500,000+ NVIDIA Blackwell B200 GPU**
- Meta 自家 MTIA silicon 用于 inference
- **$48B/年**的 AI 基础设施投入 (2025-2026)
- 这是公开报道里**目前最大的训练集群**

### 1.4 核心新能力 (Meta 强调)

**System 2 Thinking**:
- "slow, deliberate reasoning"
- 多步问题求解, 类似 GPT-5 thinking / o3 / DeepSeek R1 范式
- Meta 第一次正式进入 reasoning 模型赛道

**Recursive Self-Improvement (RSI)**:
- "model can refine its own internal logic and generate high-quality synthetic data"
- 模型生成训练数据 → 自己再训 → 性能提升 → 重复
- 这是 **Meta 公开认可的 RSI 范式**, 与 Qwen 的 self-improvement (用上一代生成数据) 同思想

**Agentic Optimization**:
- "first model specifically optimized for autonomous agents"
- 不只是 tool use, 而是从 base 训练就为 agentic 优化
- 与 Anthropic Claude (Sonnet 4.5+) / DeepSeek V3.2 / Kimi K2.6 同方向

### 1.5 安全
- **Llama Guard 4** 同步发布, 过滤有害输出
- 比 Llama 3 时代的 Llama Guard 3 升级一代

---

## 2. 性能 (Meta 自报, 无第三方独立验证)

- 匹配或超过 GPT-5 / Gemini 2.0 (注: Meta 提的是 Gemini 2.0, 已经过时, 实际对手是 Gemini 3.x)
- 没有公布具体 benchmark 数字

---

## 3. 与 Llama 历史路线的对比

| 维度 | Llama 1-3 | Llama 4 (2025.04) | **Llama 5 (2026.04)** |
| --- | --- | --- | --- |
| 架构 | dense | **首次 MoE** (Scout 109B-A17B, Maverick 400B+-A17B) | **回到 dense (600B+)** |
| 上下文 | 4K → 8K → 128K | 1M (Scout) / 10M (Scout 实验) | **5M** |
| 推理范式 | 标准 next-token | iRoPE (interleaved RoPE) | **System 2 Thinking + RSI** |
| 训练规模 | 1.4T → 2T → 15T tokens | 22T+ tokens | 未公布 (推测 50T+) |
| GPU 集群 | 16K H100 (Llama 3) | 24K H100 | **500K Blackwell B200** |
| Agentic | 通用 | 工具使用 | **专为 agent 优化** |
| License | 自定义 | 自定义 | 开源 (具体许可待公布) |

**Llama 4 → Llama 5 的反向转变**:
- **MoE → dense**: Llama 4 Behemoth (288B+ dense → MoE) 的 MoE 路线遇到 训练不稳定问题, Llama 5 回退到 dense 但放大到 600B
- **加 reasoning**: Meta 此前完全缺席 o1/R1/V3.2 推理模型潮, Llama 5 终于补上
- **加 RSI**: 类似 Qwen 的 self-improvement, 用模型自己生成训练数据

---

## 4. 与同期 frontier 模型对比 (2026.04)

| 模型 | 总参 | 激活/有效 | Context | 推理范式 | License |
| --- | --- | --- | --- | --- | --- |
| **Llama 5** | 600B+ | 600B+ (dense) | **5M** | System 2 + RSI | 开源 |
| DeepSeek V4-Pro | 1.6T | 49B (MoE) | 1M | Specialist + 三模式 | MIT |
| Qwen3.5 | 397B | 17B (MoE+GDN) | 128K+ | Thinking/Non-thinking | Apache 2.0 |
| GPT-5.5 | 闭源 | 闭源 (router) | 闭源 | 6 个 variant + adaptive | 闭源 |
| Gemini 3.x | 闭源 | 闭源 | 1M+ | Thinking modes | 闭源 |
| Kimi K2.6 | 1T | ~32B (MoE+MLA) | 长 | Agentic + thinking | 开源 |

**Llama 5 的差异化**:
- **唯一的 dense 600B+** (其他都是 MoE 或闭源)
- **最长 context (5M)**
- **官方 RSI 表态** — 公开承认用模型自训自练
- 但**最少技术细节** (没 paper, 没 system card, 没架构图)

---

## 5. 资料缺口与待补

| 资料 | 状态 |
| --- | --- |
| arxiv paper | ❌ 无 |
| Whitepaper / Technical report | ❌ 无 |
| HuggingFace 模型卡 | ⏳ 等 Meta release |
| 权重 | ⏳ 等开源放出 |
| 第三方 benchmark | ❌ 太新, 还没人测 |
| 架构细节 (RoPE 类型? attention? activation?) | ❌ Meta 未披露 |
| RSI 具体机制 | ❌ Meta 仅口头公告 |

待 Meta 后续放出更多资料时补充。

---

## 6. 对 Robotics 的启示

1. **dense 600B 还能 scale**: 在 MoE 一统天下的 2026, Llama 5 用 dense 600B 走另一条路。这暗示 dense 在足够大算力下仍有空间。对 robot FM, dense 模型部署更简单 (无 routing, 无负载均衡), 边缘场景值得保留 dense 选项
2. **5M context 拓宽了 long-horizon 上限**: robot 长任务的观测+动作历史完全可以装进 5M token, 这从根本上消除了 "context 不够" 的工程问题
3. **RSI 范式可移植**: 用上一代 robot 模型生成 sim 数据 → 训下一代 → 重复, 这就是机器人版 RSI。pi_0.5 的"反向合成数据"和 GR00T DreamGen 都是这个思想的早期实践
4. **dense 模型 + reasoning = robot reasoning policy 的可能架构**: o1/R1 都是 dense 路线 (R1 是 MoE 但 reasoning 部分本质 dense), Llama 5 也是 dense + reasoning。robot reasoning policy (规划 + 执行混合) 用 dense 可能比 MoE 更稳

---

## 7. 文件索引

```
26_Llama5/
└── llama5_notes.md   <- 本文件 (无 paper, 资料源自公告 + 媒体报道)
```

待 Meta 补技术报告后再补文件。
