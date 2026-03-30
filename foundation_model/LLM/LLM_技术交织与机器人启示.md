# LLM 技术交织图谱 -- GPT / Llama / DeepSeek / Qwen / Kimi 的分岔与汇聚

**目的**: 从机器人基础模型的视角, 梳理 LLM 领域的技术演进脉络, 识别哪些技术决策已被迁移到 robotics, 哪些正在迁移, 哪些值得关注。

---

## 1. 一张图看清全部技术交织

```
2017 ──── Transformer (Google) ──────────────────────────────────────────────
          |
2018 ──── GPT-1 ──────── BERT ──────────────────────────────────────────────
          |  decoder-only   |  encoder-only
          |  autoregressive  |  bidirectional
          |                  |
2019 ──── GPT-2 ──────── XLNet (杨植麟) ────────────────────────────────────
          |  pre-norm        |  permutation LM + Transformer-XL recurrence
          |  GELU            |  (后来成为 Kimi 长上下文的基因)
          |  架构定型         |
          |                  |
2020 ──── Scaling Laws ── GPT-3 ────────── RLHF-Summarize ─────────────────
          |  (Kaplan)        |  175B          |  人类偏好 > SFT
          |  power-law       |  in-context    |
          |                  |  learning      |
          |                  |                |
2021 ──── ViT ── CLIP ── Codex ── WebGPT ── Decision Transformer ──────────
          |       |       代码能力   工具使用    |  RL trajectory = token seq
          |       |                             |  *** 第一次迁移到 robotics ***
          |       |                             |
2022 ──── InstructGPT ── ChatGPT ── Flow Matching ── Chinchilla ───────────
          |  SFT+RM+PPO     |            |              data=model
          |  post-training   |            |
          |  范式定型         |            |
          |                  |            |
2023 ──── GPT-4 ── Llama 1/2 ── Qwen ── Diffusion Policy ── RT-2 ─────────
     ╔════╩════╗   开源元年     入场     |                    VLM->action
     ║ 闭源转折 ║                        |  *** diffusion 迁移到 robotics ***
     ╚════╤════╝                        |
          |                              |
2024 ──── GPT-4o ── o1 ── Llama 3 ── DeepSeek-V2 ── Qwen2/2.5 ── pi_0 ── GR00T N1
          omni     推理    405B dense  |  MLA+MoE      GQA+MoE      VLM+flow  VLM+DiT
                   time    RoPE+GQA    |  (原创架构)    M-RoPE       *** VLA 时代 ***
                           SwiGLU      |
                           3.2 1B/3B   |
                           (端侧LLM)   |
2025 ──── Llama 4 ── Kimi k1.5 ── MoBA ── Moonlight ── DeepSeek-R1 ── QwQ ── Qwen3
          MoE转型    RL推理    长ctx   Muon优化器    开源推理      推理   thinking统一
          iRoPE
          10M ctx
          |                |
          |                |
     ──── Kimi K2 ──── Kimi-Audio ──── Qwen3-Coder ───────────────────────
          1T MoE        Qwen2.5做base   480B MoE
          MLA+MoE
          (借鉴DeepSeek)

2026 ──── Kimi K2.5 ── Qwen3.5 ───────────────────────────────────────────
          Agent Swarm   混合注意力 (Linear + Sparse MoE)
```

---

## 2. 六次关键分岔与汇聚

### 分岔 1: Autoregressive vs Bidirectional (2018)

```
GPT-1 (decoder, 单向) ←→ BERT (encoder, 双向)
```

**当时**: BERT 赢了所有 benchmark。GPT 看起来是错误路线。
**后来**: Autoregressive 支持生成, scale 后涌现 in-context learning。BERT 路线逐渐边缘化。
**对 robotics 的影响**: 机器人 VLA 全部选择了 autoregressive 或 decoder-based 架构 (RT-2, pi_0, GR00T N1)。BERT 式的 bidirectional encoder 只活在了 CLIP 和 ViT 里做视觉编码。

**启示**: 当目标是"生成动作"而非"理解输入"时, decoder-only 是正确选择。机器人需要生成 action sequence, 和语言生成本质相同。

### 分岔 2: Scale vs Architecture (2020-2022)

```
Kaplan: 优先扩大模型 (架构不重要)
  → GPT-3: 直接从 1.5B 跳到 175B, 架构不变

Chinchilla (2022): 修正为模型和数据等比例扩大
  → 影响了后续所有人: Qwen 从 3T→7T→18T→36T 猛堆数据

DeepSeek (2024): 第三条路 -- 改架构也有用
  → MLA: 低秩 KV 压缩, 长上下文显存降 5x
  → MoE: 1T 总参但只激活 32B
  → 证明: 不只是 scale, 架构创新也能带来数量级效率提升
```

各家的选择:
| | 对 scale vs architecture 的态度 |
|---|---|
| **GPT/Qwen** | Scale 优先, 架构跟随社区共识 (GQA 等) |
| **Llama** | 标准制定者, 自己不发明但定义行业标准配方 RoPE+GQA+SwiGLU+RMSNorm |
| **DeepSeek** | 架构创新优先 (MLA 是原创), scale 为辅 |
| **Kimi** | 借鉴 DeepSeek 架构 + 优化器创新 (Muon) |

**对 robotics 的影响**: 机器人面临的是 **data scarcity** (不是 compute scarcity)。Scaling Laws 告诉我们增加数据有用, 但 robotics 的数据极贵。因此:
- DeepSeek 的 "用架构创新换效率" 思路对 robotics 更有价值
- Kimi 的 "改优化器等效 2x compute" 同样有价值
- 纯堆数据 (Qwen 路线) 在 robotics 不可行

### 分岔 3: Dense vs MoE (2024)

```
Dense 路线:  Llama 3 (405B dense, Meta)
MoE 路线:   DeepSeek-V2 (236B-A21B) → Qwen2-MoE → Kimi K2 (1T-A32B)
转型:       Llama 4 (2025) 从 dense 转向 MoE + iRoPE (10M context), Meta 也认可 MoE 路线
```

MoE 的本质: 用参数量换计算量。1T 参数存储知识, 但每次推理只激活 32B, 推理成本和 32B dense 模型相当。

**在 robotics 中的对应**: 机器人有类似需求 -- 需要大量"知识"(物体属性、任务语义、物理常识) 但推理必须实时 (100Hz+)。GR00T N1 的双系统架构是一种变体:
```
System 2 (VLM, 10Hz): 大模型, 慢推理, 理解语义
System 1 (DiT, 120Hz): 小模型, 快推理, 生成动作
```
这和 MoE "大存储 + 小激活" 的思路异曲同工, 只是分离更彻底。

### 分岔 4: Post-training 路线 -- Alignment vs Reasoning (2022-2025)

GPT 系列定义了 post-training 的范式后, 出现了两个方向:

```
方向 A: Alignment (让模型听话)
  InstructGPT (2022) → ChatGPT → GPT-4 RLHF
  reward = 人类偏好 (主观, 不可验证)
  方法: SFT → RM → PPO

方向 B: Reasoning (让模型思考)
  o1 (2024) → DeepSeek-R1 (2025) → Kimi k1.5 → QwQ → Qwen3
  reward = 答案正确性 (客观, 可验证)
  方法: outcome-based RL + long CoT
```

**交织点**: Qwen3 将两者统一 -- thinking mode (reasoning) + non-thinking mode (alignment) 在同一模型中, 通过 thinking budget 切换。

**对 robotics 的直接影响**:

| LLM post-training | Robotics 对应 | 论文 |
|---|---|---|
| SFT (模仿人类示范) | **Behavior Cloning** (模仿遥操作) | ACT, Diffusion Policy |
| RLHF (人类偏好优化) | **RLHF for robots** (人类反馈优化策略) | 尚在探索 |
| Outcome-based RL (答案正确性) | **Success-based RL** (任务成功率) | PPO in sim (GR00T N1) |
| Long CoT (推理链) | **Planning chain** (任务规划) | SayCan, Code-as-Policies |
| Thinking budget | **Compute allocation** (简单动作快推理, 复杂任务慢规划) | GR00T N1 双系统 |

Qwen3 的 thinking budget 概念尤其值得 robotics 关注: 不是所有机器人动作都需要深度推理。抓杯子不需要"思考", 但整理桌面需要规划。动态分配推理计算量是未来方向。

### 分岔 5: 多模态路线 -- 后接 vs 原生 (2023-2026)

```
后接路线 (主流):
  预训练好文本 LLM → 接上视觉编码器 → 视觉-文本对齐训练
  代表: GPT-4V, Qwen-VL, Qwen2-VL, Kimi K2.5, pi_0 (PaliGemma)

原生路线 (新兴):
  从预训练开始就混合多模态数据
  代表: GPT-4o, Qwen3.5 (早期融合), Qwen2.5-Omni (Thinker-Talker)
```

**对 robotics 的影响**: 当前所有 VLA (pi_0, GR00T N1, RT-2, OpenVLA) 都用后接路线 -- VLM backbone + action head。但原生多模态可能是更好的方向:
- 机器人的"语言"是连续动作, 不是离散 token
- 后接式需要 tokenize 动作或用 flow matching 桥接, 有信息损失
- 原生融合 (视觉 + 本体感觉 + 语言 + 动作从预训练就混在一起) 可能效果更好

### 分岔 6: 长上下文技术 -- 对 robotics 的特殊意义

```
RoPE 扩展:    NTK-RoPE (Qwen1, 32K) → YaRN (Qwen2+, 128K) → 1M (Qwen2.5-1M)
稀疏注意力:   Sparse Transformer → MoBA (Kimi, block-sparse, 1M)
线性注意力:   Mamba/RWKV → Qwen3.5 (Gated Delta Networks 混合)
KV 压缩:     GQA (Llama2+) → MLA (DeepSeek, 低秩压缩)
```

**这对 robotics 至关重要, 但原因不是"上下文更长"**:

机器人 VLA 面临的上下文问题和 LLM 不同:

| LLM 的上下文需求 | Robotics 的上下文需求 |
|---|---|
| 长文档理解 (128K+ tokens) | 视频观察序列 (几百帧 ≈ 几千 tokens) |
| 一次性处理 | **流式处理** (每帧追加) |
| 文本 token (离散) | 图像 token (高维, 密集) |
| 注意力 O(n^2) 主要影响 latency | 注意力 O(n^2) 直接影响**控制频率** |

具体来说:
- **MoBA 的块稀疏注意力** → 可用于 VLA: 每帧只关注最相关的历史帧, 而非全部
- **MLA 的 KV 压缩** → 直接减少 VLA 推理显存, 允许处理更长视频历史
- **线性注意力 (Qwen3.5)** → 理论上 O(n) 复杂度, 对实时控制最友好

GR00T N1 用 120Hz DiT 做动作生成, 正是因为标准 Transformer attention 在高频下太慢。如果 MoBA/MLA/线性注意力能降低 VLM 的推理成本, 可能不需要双系统分离。

---

## 3. LLM 技术迁移到 Robotics 的完整图谱

### 3.1 已完成的迁移

| LLM 技术 | 谁发明的 | Robotics 中的形态 | 代表论文 |
|---------|---------|-----------------|---------|
| Transformer decoder | GPT-1 (2018) | VLA 的 backbone | RT-2, pi_0, GR00T N1 |
| Next-token prediction | GPT-1 (2018) | 离散动作 token 预测 | RT-2, OpenVLA |
| Scaling Laws | Kaplan (2020) | Robot scaling laws | 25_RobotScalingLaws |
| In-context learning | GPT-3 (2020) | Few-shot task specification | Prompt-based robot planning |
| SFT (模仿人类示范) | InstructGPT (2022) | Behavior Cloning | ACT, Diffusion Policy |
| RLHF / PPO | InstructGPT (2022) | RL fine-tune in sim | GR00T N1 |
| VLM backbone | GPT-4V (2023) | VLA 的视觉-语言部分 | pi_0 (PaliGemma), GR00T N1 (Eagle) |
| Tool use | WebGPT (2021) | Robot + tool interaction | SayCan, Voyager |
| Code generation | Codex (2021) | Code-as-Policy | Code-as-Policies (2022) |
| 开源 LLM backbone | Llama (2023) | VLA 的语言模型骨干 | OpenVLA (Llama 2 7B) |

### 3.2 正在迁移的技术

| LLM 技术 | 来源 | Robotics 中的潜在应用 | 当前状态 |
|---------|------|-------------------|---------|
| MoE 稀疏激活 | DeepSeek-V2 | 大知识库 + 小推理成本的 VLA | 初步探索 (Octo 的 readout tokens 类似) |
| MLA (KV 压缩) | DeepSeek-V2 | 降低 VLA 视频处理的显存 | 尚未有 robotics 论文采用 |
| M-RoPE (3D 位置编码) | Qwen2-VL | 编码视频帧的空间+时间位置 | 概念匹配但尚未直接迁移 |
| Thinking budget | Qwen3 | 简单动作快推理, 复杂任务慢规划 | GR00T N1 双系统是粗粒度版本 |
| Outcome-based RL | DeepSeek-R1 / k1.5 | 任务成功率做 reward 的 RL | 已有但不是从 LLM 迁移来的 (robotics RL 更早) |
| GRPO (无 critic RL) | DeepSeek-Math (2024) | Robot RL 无需训练 value function | 尚未有 robotics 论文采用 |
| Edge LLM (1B/3B) | Llama 3.2 (2024) | 机器人端侧推理 | 初步探索 |

### 3.3 值得关注但尚未迁移的技术

| LLM 技术 | 来源 | 为什么值得关注 |
|---------|------|-------------|
| **Muon 优化器** (Moonlight) | Kimi | 2x 训练效率, robotics 训练数据少, 每个 sample 更"值钱" |
| **MoBA 块稀疏注意力** (Kimi) | Kimi | 可让 VLA 在长视频上高效推理, 只关注相关帧 |
| **线性注意力** (Qwen3.5) | Qwen | O(n) 复杂度, 可能解决 VLA 高频推理瓶颈 |
| **Self-improvement** (Qwen2.5) | Qwen | 用已训模型生成 synthetic demonstrations, 扩充 robot 数据 |
| **Agent Swarm** (Kimi K2.5) | Kimi | 多机器人协作的 planning 框架 |
| **Partial rollouts** (Kimi k1.5) | Kimi | 长 trajectory RL 的效率技巧, 直接可用于 robot RL |
| **iRoPE (10M context)** | Llama 4 (2025) | 超长 trajectory 处理, 交替 RoPE/NoPE |

---

## 4. 对你 (从 LLM 学习做机器人基础模型) 的建议

### 4.1 LLM 中哪些 "定论" 可以直接继承

这些是 LLM 领域已经探索完毕、机器人可以直接拿来用的结论:

| 定论 | GPT 什么时候确立的 | 在 robotics 中怎么用 |
|------|-----------------|-------------------|
| Transformer decoder 是通用序列模型 | GPT-1 (2018) | VLA backbone, action sequence modeling |
| Pre-train + fine-tune 范式 | GPT-1 (2018) | Web data pre-train → robot data fine-tune |
| 架构不重要, scale 重要 | Scaling Laws (2020) | **部分成立**: robot data 太少, 架构效率更重要 |
| Post-training 比 pre-training 更灵活 | InstructGPT (2022) | 一个 base VLM, 多种 robot fine-tune |
| RLHF/RL 可以改善 base model 行为 | InstructGPT (2022) | RL in sim 改善 BC policy |
| 开源 LLM 作为基座 backbone | Llama 2 (2023) | VLA 直接用开源 LLM 做 language backbone (OpenVLA) |
| MoE 可以用大参数换小计算 | DeepSeek-V2 (2024) | 需要大知识但实时推理的 VLA |

### 4.2 LLM 中哪些问题 robotics 也面临

| LLM 的问题 | Robotics 的对应问题 | LLM 的解法 | Robotics 能否照搬 |
|-----------|-------------------|---------|-----------------|
| 数据获取 | 遥操作数据太少太贵 | 爬 Internet | **不能**, 需要 sim + synthetic + cross-embodiment |
| Hallucination | 动作不物理合理 | RLHF | **部分可以**, reward = 物理模拟器的成功率 |
| 推理太慢 | 控制频率不够 | MoE/MLA/线性注意力 | **可以**, 值得迁移 |
| 安全对齐 | 安全约束 | RLHF + guardrails | **不同**: robot 安全更靠物理约束而非语言约束 |
| 长上下文 | 长视频/长任务 | MoBA/YaRN/MLA | **可以**, 但视频 token 密度远高于文本 |
| 多模态融合 | 视觉+力觉+本体感觉 | M-RoPE / 原生融合 | **值得借鉴**, M-RoPE 的 3D 位置编码思想可直接用 |

### 4.3 本库尚缺但重要的 LLM→Robotics 桥梁论文

| 论文 | 年份 | 为什么重要 | 本库位置 |
|------|------|-----------|---------|
| **SayCan** (Google) | 2022 | GPT planning + CLIP affordance → robot execution, "LLM 指挥 robot" 的第一个 milestone | `methods/2_bridges/22_SayCan/` |
| **Code-as-Policies** (Google) | 2022 | GPT 直接生成 robot control code, 不训练 policy | `methods/2_bridges/22_CodeAsPolicies/` |
| **Inner Monologue** (Google) | 2022 | LLM 作为 robot 内心独白, 闭环语言反馈 | `methods/2_bridges/22_InnerMonologue/` |
| **LLaVA** (UW-Madison) | 2023 | 开源 VLM 训练范式 (visual instruction tuning), OpenVLA 的灵感来源 | `methods/3_vla_perception/23_LLaVA/` |
| **Voyager** (NVIDIA) | 2023 | GPT-4 做 open-ended agent: code generation + skill library + self-verification | `methods/3_vla_perception/23_Voyager/` |

这些论文代表 **LLM-as-planner 路线** (直接用 LLM 做 robot 大脑), 与本库已有的 **VLA 路线** (LLM 架构迁移到 robot) 互补:

```
路线 1 (VLA): LLM 架构 → robot policy (端到端)
  RT-1 → RT-2 → Octo → OpenVLA → pi_0 → GR00T N1

路线 2 (LLM-as-planner): LLM 能力 → robot planning (模块化)
  SayCan → Code-as-Policies → Voyager → Inner Monologue

当前趋势: 融合
  GR00T N1 = VLM 做 high-level planning (路线 2) + DiT 做 low-level action (路线 1)
```

### 4.4 优先阅读和实验建议

**如果你的目标是做机器人基础模型, 按这个顺序学习 LLM**:

```
=== 第一层: 理解范式 (必读) ===
1. GPT-1 → GPT-3: 理解 "pre-train + scale + in-context learning"
2. InstructGPT: 理解 "SFT + RM + PPO" 三步法 (=机器人的 BC + reward + RL)
3. Scaling Laws: 理解 power-law, 对 robot data scaling 有预期

=== 第二层: 理解效率创新 (对 robotics 最有价值) ===
4. MoBA (Kimi): 块稀疏注意力 → 可用于 VLA 长视频处理
5. Moonlight (Kimi): Muon 优化器 → 2x 训练效率对数据稀少的 robotics 极有价值
6. MLA (DeepSeek-V2): KV 压缩 → 直接降低 VLA 推理显存

=== 第三层: 理解多模态 (直接关联 VLA) ===
7. Qwen2-VL (M-RoPE): 3D 位置编码 → 视频理解的基础
8. Kimi K2.5 (MoonViT + Agent Swarm): 多模态 + 多智能体

=== 第四层: 理解 RL for reasoning (前沿方向) ===
9. Kimi k1.5 (partial rollouts): 长 trajectory RL 的效率技巧
10. Qwen3 (thinking budget): 动态计算分配 → 未来机器人的 System 1/2 统一
```

### 4.5 一句话总结

**LLM 用 8 年 (2018-2026) 走完了从 "pre-train 范式探索" 到 "post-training 方法成熟" 到 "效率优化" 的全过程。Robotics 正处于 2020 年的 GPT-3 时刻 -- 知道 scaling 有用, 但还没找到自己的 Chinchilla (数据-模型最优比例), 也还没找到自己的 InstructGPT (最佳 post-training 方法)。从 LLM 学习, 不是照搬技术, 而是借鉴 "提问题的方式" 和 "解决问题的框架"。**
