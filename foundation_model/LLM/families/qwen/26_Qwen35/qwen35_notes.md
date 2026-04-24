# Qwen3.5 / Qwen3.5-Omni notes

> Qwen3.5 base 模型: 2026.02.16 发布, 首发 Qwen3.5-397B-A17B (MoE), Qwen3.5-9B/4B 后续补齐
> **Qwen3.5-Omni Technical Report**: arXiv:2604.15804v2 (2026.04.21)
> Qwen3.6-Plus 已发布 (2026.04, 闭源 API only)

---

## 1. Core Innovation: Hybrid Attention MoE 架构

### 1.1 架构核心 — Gated DeltaNet + Sparse MoE 双层稀疏

**60 层 Transformer**, 每层都是 MoE FFN, 重复模式:
```
3 层 Gated DeltaNet (GDN, 线性注意力)  →  1 层 Gated Attention (标准 softmax + GQA + RoPE)
```

→ 相当于 **75% 层用线性注意力, 25% 层用标准 softmax**

**MoE 配置 (Qwen3.5-397B-A17B)**:
- 397B 总参 / 17B 激活
- 512 个 expert (10 routed + 1 shared activated per token)

**为什么这样混**:
- 完全线性 (Mamba/RWKV) 的弱点: 表达力不够, 长程检索差
- 完全 softmax 的弱点: O(L²), 长上下文吃不消
- Qwen3.5 的赌注: **每 4 层一个 full attention 层来锚住表达力**, 其余 3 层用 GDN 加速 + 降 KV cache I/O
- 解码速度比 Qwen3-Max **快 8.6x ~ 19x**, 长上下文增益更大

### 1.2 与 DeepSeek V4 / Kimi 的对比 (同期 attention 路线)

| 方案 | 团队 | 长上下文压力转移到哪里 |
| --- | --- | --- |
| MLA + DSA (V3.2) | DeepSeek | 压缩 KV cache + sparse top-k token |
| **Hybrid CSA + HCA** (V4) | DeepSeek | **二级稀疏 + 极致压缩** |
| MoBA | Kimi | chunk 级 top-k |
| **Gated DeltaNet + sparse softmax** (Qwen3.5) | **Alibaba** | **75% 层换成线性注意力 (Mamba 同族)** |

**根本差异**: DeepSeek 走 "保留 softmax 但稀疏化 + 压缩",Qwen3.5 走 "大部分层换成线性, 少数层保留 softmax"。两条路在 1M 级上下文上都成立, 是同一问题的不同赌注。

---

## 2. Qwen3.5-Omni: 全模态 (text/image/audio/video → text/speech)

### 2.1 Thinker-Talker 架构 (继承 Qwen2.5-Omni)
- **Thinker**: 处理 text + audio + image + video, 输出 text 和 high-level representation
- **Talker**: 接收 Thinker 的高层表征 + text token, 自回归输出 streaming RVQ codec token
- **Code2Wav renderer**: 增量合成波形, 帧级 streaming

两端都基于 Qwen3.5 的 Hybrid Attention MoE 主干。

### 2.2 5 个核心升级 (vs Qwen3-Omni)
1. Thinker 和 Talker 都换成 Hybrid Attention MoE (GDN + sparse softmax)
2. **256K 上下文**, 支持 10 小时音频 / 400 秒 720P 视频 (1 FPS)
3. Talker 用 multi-codebook codec representation, single-frame 即时合成
4. **ARIA (Adaptive Rate Interleave Alignment)** — 解决 streaming 语音合成的不稳定
5. 多语言扩到 113 种 (ASR) + 36 种 (TTS)

### 2.3 ARIA — Streaming TTS 的核心创新

**问题**: text token 和 speech token 的 tokenization rate 不匹配 → streaming 时频繁出现 skip 词、错读、数字读得不清楚。

**ARIA 的做法**:
- 把双通道 (text 一路 + speech 一路) 统一成**单通道交错 stream**
- 自适应 rate constraint: 任意 prefix, 累积 speech-to-text token 比不能超过段落级全局比
- 支持任意 text prefix → 后续 speech continuation 仍然连贯
- 不依赖 MFA-derived alignment 或固定交错率

### 2.4 新涌现能力 — Audio-Visual Vibe Coding

> "an emergent capability wherein the model directly generates executable code from audio-visual instructions"

直接根据音视频指令写可执行代码, 不需要外部 orchestrator 把指令转成文字再喂给 coding 模型 — 是 Omni 模型独有的 emergent 能力。

### 2.5 实测结果

- **Qwen3.5-Omni-Plus** 在 215 个 audio + audio-visual 子任务上 SOTA
- 通用音频理解/推理/识别/翻译/对话: **超过 Gemini-3.1 Pro**
- 综合 audio-visual 理解: **匹配 Gemini-3.1 Pro**
- 没有牺牲 text/vision 性能 (与同 size 单模态 Qwen 模型持平)

---

## 3. 对 Robotics 的启示

1. **Hybrid Attention 是长上下文的另一种解**: 你做长 horizon 视频/状态序列, GDN + sparse softmax 这条路 (Qwen3.5) 与 DSA/CSA (DeepSeek) 是两个互补的 design space, 都值得跟踪
2. **Thinker-Talker 范式可移植到 VLA**: Thinker 输出 high-level representation, Talker 输出连续 action token — 这与 pi_0 的 "VLM + Action Expert" 结构同构, 但 Qwen3.5-Omni 的工程化程度更高 (256K context, streaming, low TTFT)
3. **ARIA 的对齐思想**: token rate mismatch 在机器人也存在 (语言 token 和动作 token 的 rate 不一样), ARIA 的 adaptive rate constraint 思想可借鉴

---

## 4. 文件索引

```
26_Qwen35/
├── README.md                              <- 早期占位说明
├── qwen35_notes.md                        <- 本文件
└── Qwen3.5-Omni_Technical_Report.md      <- arXiv:2604.15804v2 markdown
```

base Qwen3.5 (text-only) 仍无独立 arxiv 论文, 依赖 GitHub README + HuggingFace blog: https://github.com/QwenLM/Qwen3.5
