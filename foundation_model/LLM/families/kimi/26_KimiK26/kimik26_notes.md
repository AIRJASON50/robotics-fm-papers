# Kimi K2.6 (2026.04.21)

> **状态**: 模型 + 权重已开源, 基于 K2.5 paper (arXiv:2602.02276), **无 K2.6 独立 arxiv**
> **License**: Modified MIT (代码 + 权重)
> **Preview**: 2026.04.13 以 "K2.6 Code Preview" beta 上线, **8 天后** (2026.04.21) preview 标签移除正式 GA
> Blog: https://www.kimi.com/blog/kimi-k2-6.html
> HuggingFace: moonshotai/Kimi-K2.6

---

## 1. 与 K2 / K2.5 的关系

K2.6 不是架构换代, 是 K2.5 的**能力侧扩展**:

| 版本 | 发布 | 核心点 |
| --- | --- | --- |
| K2 (2025.07) | arXiv:2507.20534 | 1T/32B MoE, MLA, MuonClip, 15.5T tokens 预训练 |
| K2-Thinking | 2025.11 | 加 long CoT reasoning, INT4 native 量化 |
| K2.5 (2026.02) | arXiv:2602.02276 "Visual Agentic Intelligence" | 加 vision encoder (MoonViT 400M), 视觉 agent |
| **K2.6 (2026.04)** | 无 paper, 沿用 K2.5 | **Long-horizon coding + 300-subagent swarm + 4000-step orchestration** |

→ K2.6 是 K2.5 架构骨架上的**后训练 + agentic 扩展**, 不是新架构。

---

## 2. 架构速查 (继承 K2.5)

| 项 | 值 |
| --- | --- |
| 总参数 | 1T |
| 激活参数 | 32B |
| 层数 | 61 (含 1 个 dense) |
| Attention | **MLA** (Multi-head Latent Attention, 借自 DeepSeek V2) |
| 注意力头数 | 64 (K2 从 DeepSeek V3 的 128 砍到 64 省推理) |
| Hidden dim | 7168 |
| MoE expert hidden | 2048 |
| Expert 总数 | **384** (K2 放大自 DeepSeek V3 的 256) |
| 每 token 激活 expert | 8 routed + 1 shared |
| 词表 | 160K |
| Context | **256K** |
| Vision encoder | **MoonViT 400M** (K2.5 引入) |
| Activation | SwiGLU |
| 优化器 | Muon + QK-Clip = MuonClip |
| 量化 | **INT4 native** (K2-Thinking 引入) |

---

## 3. K2.6 的核心新能力 (vs K2.5)

### 3.1 Long-Horizon Coding (主打)
- 跨多种语言 (Rust / Go / Python) 的 end-to-end 编程
- 跨多个领域 (front-end / DevOps / performance optimization)
- 支持 **12 小时自主 coding session**
- 对比 K2.5 主要在单文件/单 PR 级别

### 3.2 Coding-Driven Design
- 把 prompt + 视觉输入转换为 production-ready interface
- 生成 structured layout + interactive elements
- 可以直接输出 full-stack 工作流

### 3.3 Elevated Agent Swarm (最显眼)
- **300 个子 agent 并行**
- **4000 步 orchestration** 的动态任务分解
- 适合 BrowseComp 类复杂 web 任务

### 3.4 Proactive & Open Orchestration
- 24/7 background agent, 不需要人随时触发
- 支持 interleaved thinking: 多步 tool call 之间保持 reasoning 连贯

### 3.5 Thinking vs Instant 两种模式
- Thinking: `temperature=1.0, top_p=1.0`, 深度推理
- Instant: `temperature=0.6, top_p=0.95`, 快速响应

---

## 4. 实测结果 (vs frontier)

### 4.1 编程 (主战场)
| Benchmark | K2.6 | 对照 |
| --- | --- | --- |
| **SWE-Bench Pro** | **58.6** | GPT-5.4 = 57.7 |
| SWE-Bench Verified | 80.2 | — |
| SWE-Bench Multilingual | 76.7 | — |
| Terminal-Bench 2.0 | 66.7 | Terminus-2 framework |
| LiveCodeBench v6 | 89.6 | V4-Pro-Max = 93.5 |
| OJBench (Python) | 60.6 | — |

### 4.2 推理 & 知识
| Benchmark | K2.6 |
| --- | --- |
| AIME 2026 | **96.4** |
| HMMT 2026 (Feb) | 92.7 |
| GPQA-Diamond | 90.5 |
| IMO-AnswerBench | 86.0 |

### 4.3 Vision & Multimodal
| Benchmark | K2.6 | K2.6 + Python |
| --- | --- | --- |
| MMMU-Pro | 79.4 | 80.1 |
| CharXiv (RQ) | 80.4 | 86.7 |
| MathVision | 87.4 | 93.2 |
| BabyVision | 39.8 | 68.5 |

### 4.4 Agentic (K2.6 主打)
| Benchmark | K2.6 |
| --- | --- |
| HLE-Full (w/ tools) | 54.0 |
| BrowseComp | 83.2 |
| **BrowseComp (Agent Swarm)** | **86.3** |
| DeepSearchQA (f1) | 92.5 |
| Claw Eval (pass@3) | 80.9 |
| OSWorld-Verified | 73.1 |

Moonshot 官方说法: **持平或超过 GPT-5.4 / Claude Opus 4.6 / Gemini 3.1 Pro**。

---

## 5. 与同期 frontier (2026.04) 的横向对比

| 模型 | 规模 | Context | 主战场 | License |
| --- | --- | --- | --- | --- |
| DeepSeek V4-Pro | 1.6T / 49B | 1M | 全面, coding 93.5 LiveCodeBench | MIT |
| Kimi K2.6 | 1T / 32B | 256K | **Agent swarm + long-horizon coding** | **Modified MIT** |
| Qwen3.5-Omni | 397B / 17B | 256K | Omni (text/image/audio/video), ARIA TTS | Apache 2.0 |
| GPT-5.5 | 闭源 | 闭源 | Agentic coding + computer use | 闭源 |
| Gemini Robotics ER 1.6 | 闭源 | — | 机器人 embodied reasoning | 闭源 |
| Llama 5 | 600B dense | **5M** | System 2 + RSI | 开源 |

**K2.6 的差异化**:
- 最强 **agent swarm 规模** (300 子 agent × 4000 步)
- SWE-Bench Pro 开源最佳 (58.6 > GPT-5.4 57.7)
- 保留 K2 架构优势 (MLA + MuonClip + INT4 native)
- **6 周 K2.5 → K2.6**, 与 DeepSeek V3.2 → V4 的迭代速度对标

---

## 6. 6 周 K2.5 → K2.6 的 preview-to-GA 模式

**新模式**: K2.6 Code Preview (2026.04.13 beta) → K2.6 GA (2026.04.21), 仅隔 **8 天**。
- 对比 K2 → K2.5 隔 7 个月, K2.5 → K2.6 只隔 2.5 个月
- **下一代 K3 可能会在同样的 preview→GA 短周期内发布**
- K3 目标: 与美国 frontier 模型 parameter scale 对齐, 可能 **3-4T 总参数**

这种节奏反映了 **2026 Q2 开源 frontier 进入"周级迭代"**, 而不再是季度级。

---

## 7. 部署

### 推理引擎
- **vLLM** (推荐)
- **SGLang**
- **KTransformers**

### 版本要求
- Transformers: `>=4.57.1, <5.0.0`
- API 兼容: OpenAI / Anthropic 接口

### 特性
- Thinking mode (可选 `preserve_thinking` 跨 tool call 保留)
- Interleaved thinking (多步 tool call 中保持 reasoning)
- Visual input: 图像 + 视频 (video 当前仅官方 API)
- Instant mode (关掉 thinking 的快模式)

---

## 8. 对 Robotics 的启示

1. **Agent swarm 模式可借鉴到机器人**: 300 子 agent × 4000 步的 orchestration 是 multi-robot 协作或单机器人长 horizon 任务分解的自然参考
2. **Preview→GA 短周期**: 对应到 robot FM, 说明快速迭代比大版本号跳跃更有价值 (pi_0.5 → pi_0.6 → pi_0.7 也是类似节奏)
3. **K2 → K2.5 → K2.6 = 架构不变, 后训练扩展**: 说明一个好 base (K2 的 MLA + MuonClip) 可以持续做后训练创新。对 robot FM, 稳定的 VLA backbone 上做 skill-level fine-tune 是可行路线
4. **INT4 native 量化**: 边缘部署 robot FM 时, INT4 是从 Jetson 级设备跑大 VLA 的 enabling technology

---

## 9. 文件索引

```
26_KimiK26/
└── kimik26_notes.md   <- 本文件 (无独立 paper, 基于 HF 模型卡 + Moonshot blog)
```

资料源:
- HuggingFace: https://huggingface.co/moonshotai/Kimi-K2.6
- Blog: https://www.kimi.com/blog/kimi-k2-6.html
- K2.5 paper (架构来源): arXiv:2602.02276
- K2 paper (架构原始): arXiv:2507.20534
