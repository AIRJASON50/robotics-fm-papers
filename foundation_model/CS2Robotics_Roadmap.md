# CS to Robotics: Foundation Model 学习路线图

> **你的背景**: PPO sim2real 灵巧手操作, 有 RL 和机器人实践经验
> **你的目标**: 理解 CS (LLM+CV) 的发展如何塑造了现代 robotics FM, 学习方法论和设计思想
> **不做什么**: 不复现代码, 不成为 LLM/CV 专家
> **核心洞察**: Robotics 正在重走 LLM/CV 的路, 学习这条路 = 看到 robotics 的未来

---

## 2026 Q2 重大更新汇总 (本次知识库集中追平)

> 本路线图的核心结构 (Level 0-4, 4 条迁移路径, VLM backbone 范式) 仍然成立, 但 2025.10 - 2026.04 期间出现了若干**改变路线图判断**的新发现, 已在对应章节标记 `*NEW*`。一句话速览:

| 维度 | 老路线图判断 | 2026 Q2 新发现 / 修正 |
| --- | --- | --- |
| **Robot 在 LLM 时间线的位置** | InstructGPT 到 ChatGPT 之间 (单一判断) | **拆成三轴看**: 能力接近 GPT-3 到 GPT-3.5 (pi_0.7 有 compositional generalization 的"strong signs", 不是 full emergence); 产品化仍在 pre-ChatGPT; 研究收敛度还在 BERT-vs-GPT 之争的早期。展望方向仍指向 GPT-4/o1 级 |
| **迁移路径** | 4 条 (token / 视觉 / 生成 / 世界模型) | **+4 条** (CoT→action thinking, 长上下文 attention, AI feedback→robot reward, 人类 video→robot policy) |
| **Robot family 数量** | 3 家 (RT / PI / GR00T) | **4 家** — 加 **Gemini Robotics** (复用 Gemini frontier + Embodied Thinking + 90% MuJoCo 评估) |
| **VLA RL 的角色** | RL 从 training 退到 post-training | **VLA 层 RL 也在退潮** — pi_0.7 用 metadata-conditioned BC 替代 pi*0.6 RECAP, 效果同档但简单 10 倍 |
| **数据采集范式** | 遥操作为主, 仿真合成补充 | **+2 种新范式**: GR00T N1.7 用 20K h 人类 ego video (首条 dexterity scaling law); pi_0.7 用 verbal coaching 替代遥操作 |
| **跨 embodiment 桥梁** | Open X 同构型混训 | **架构层接口**: GR00T relative EEF delta (人/机器人共享) + Gemini Motion Transfer recipe |
| **长 horizon 瓶颈** | context 不够, MEM 缓解 | **LLM 侧 1M-5M context 已普及** (DeepSeek V4 / Llama 5), 三条不同 attention 路线 (DSA/CSA+HCA, GDN, dense), 待迁移到 robot |
| **推理范式** | 老 roadmap 没单独讨论 | **新独立主题**: Embodied Thinking (Gemini Robotics 1.5), Metadata + CFG (pi_0.7), RL Tokens (GR00T RLT) |
| **LLM 家族覆盖** | GPT/Kimi/Qwen/DeepSeek/Llama (5 家) | **+1 家 Anthropic Claude** — Constitutional AI / Computer Use / alignment 研究 对 robot 安全和虚拟 VLA 路线有方法论价值 |
| **你的 RL 经验定位** | 在 pi*0.6 路线最有价值 | **适用范围收窄**: VLA 层被 metadata 替代; **WBC + 在线精调 (RLT)** 仍是你的核心阵地 |

**本次集中追平涉及的家族 notes 更新** (按提交顺序):
- `pi_Series/` → 加 pi_0.7 + Phase 7
- `GR00T_Series/` → 加 N1.7 (Cosmos-Reason2-2B + EgoScale + dexterity scaling law)
- `deepseek/` → 加 V3.2 (DSA) + V4 (CSA+HCA + mHC + Muon + FP4 + 1M context)
- `Google_RT_Series/` → 加 Gemini Robotics 子家族 (GR 1.0/1.5 + ER 1.6) + Phase 4
- `GPT_Series/` → 补 GPT-4o → o1 → o3 → 4.5 → 5.x (Phase 6 完整覆盖)
- `llama/` → 加 Llama 5 (600B dense, 5M context, RSI)
- `qwen/` → 加 Qwen3.5 (Hybrid GDN + sparse softmax) + Qwen3.5-Omni
- `kimi/` → 加 K2.6 (300-agent swarm, SWE-Bench Pro 58.6 开源最佳)
- `Anthropic_Claude/` → **新建** Constitutional AI + Claude 1→4.7 全演进

详细判断细节散落在下面各章节带 `*NEW*` 标记的部分。

### 本路线图在知识库中的位置

```
你的学习路径:

manip/ (你的起点)                     "问题是什么"
  manip_landscape.md                    灵巧操作 5 个主题的全景
  → 发现: per-task RL 不可扩展
  
humanoid/ (扩展视野)                   "规模化怎么做"
  humanoid_landscape.md                 人形控制 4 个主题的全景
  → 发现: motion tracking 可以统一所有运动

foundation_model/ (方法论来源)          "为什么可以这样做"
  >>> 本文件: CS2Robotics_Roadmap.md    Level 0-4 的学习路线 <<<
  LLM_技术交织与机器人启示.md             LLM 全景 + 技术分岔
  CV_技术演进与机器人启示.md              CV 全景 + 5 条演进线
```

如果你还没看过 manip_landscape.md 和 humanoid_landscape.md, **建议先看它们**——它们是你从自身经验出发理解 "为什么需要 FM" 的上下文。本文件假设你已经理解了灵巧操作的 5 个主题和人形控制的 4 个方向。

---

## 1. Robotics 在 LLM 时间线上的位置

这是理解整个路线图的关键：

```
LLM 时间线:                              Robotics 对应:

2018  GPT-1 (pre-train+fine-tune)        2022  RT-1 (robot Transformer)
2019  GPT-2 (scale → zero-shot)          2023  Open-X (跨机器人数据集)
2020  GPT-3 (in-context learning)        2023  RT-2 (VLM→动作, web 知识迁移)
2020  Scaling Laws (Kaplan/Chinchilla)   2026  GR00T N1.7 dexterity scaling law *NEW*
                                               (人类 video 1k→20k h = 2x task completion)
2022  InstructGPT (RLHF post-training)   2025  pi*0.6 (offline RL fine-tune VLA)
2022  ChatGPT (产品化)                    2026  Opus 4.7 Computer Use / Gemini Robotics 1.5
                                               (接近产品化, agentic 场景)
2022  Constitutional AI (AI feedback)    2026  pi_0.7 metadata conditioning *NEW*
                                               (metadata 标注替代 critic, 与 CAI 同思想)
2024  o1 (推理时计算, CoT)                2025  Gemini Robotics 1.5 Embodied Thinking *NEW*
                                               (action 层 CoT, 不只在高层规划)
2024  GPT-4o (多模态融合)                 2026  Qwen3.5-Omni / pi_0.7 (subgoal image) *NEW*
2025  R1-Zero (pure RL reasoning 开源)    2026  Audio-Visual Vibe Coding (新 emergent)
2025  GPT-5 router (system over model)   待观察  (robot 侧的 "router + specialist" 架构)
2026  DeepSeek V4 / Kimi K2.6 / Llama 5   2026  DreamZero (想象再行动, N2 核心)
      (million-token context, 300-agent
       swarm, recursive self-improvement)

>>> 能力轴: Robotics 2026.Q2 ≈ LLM 的 GPT-3 到 GPT-3.5 之间 (保守判断) <<<
>>> 产品化轴: 仍在 pre-ChatGPT (lab-scale, 无消费级产品爆发) <<<
>>> 研究收敛轴: 4 家打 4 种赌注, 像 BERT vs GPT 之争的 2018-2019 年 <<<
>>> 展望方向 (可乐观): 6-12 个月内能力轴有望触达 GPT-4/o1 级 <<<
```

**三轴错位的理由**:

| 轴 | 当前位置 (保守判断) | 证据 |
| --- | --- | --- |
| 能力 | GPT-3 到 GPT-3.5 之间 | pi_0.7 compositional generalization 是 "strong signs" 不是 full emergence (未见任务 60-80% vs 训练任务 >90%); Embodied Thinking 只有 Gemini 一家做; Relative EEF 刚起步 |
| 产品化 | pre-ChatGPT | 无 1 亿用户级产品, 部署仍限于实验室, 推理成本高 (pi_0.7 依赖云端 API, N1.7 训练需 800 GPU × 120 h) |
| 研究收敛 | 更早期 (BERT-vs-GPT 时代) | 4 家主线架构完全不同 (PI prompt / NVIDIA 数据 / Google frontier / Kimi agent swarm), 没有"Transformer 配方"这种全行业共识 |

**关键时间对应的更新 (2026.04) — 保守表述**:
- 老 roadmap 说 "Robotics 今天 ≈ InstructGPT 到 ChatGPT 之间" — 这个单一判断合并了多个轴, 不够精确
- **pi_0.7 (2026.04) 有 compositional generalization 的迹象**, 但未见任务成功率仍 60-80% — 只是接近 GPT-3 的 in-context learning 起步, 不是完整涌现
- **GR00T N1.7 (2026.04) 给出第一条明确的 dexterity scaling law** — 这是 robot 版的 Kaplan/Chinchilla, 理论层面的正式坐标
- **Gemini Robotics 1.5 (2025.10) 把 thinking/CoT 下沉到 action 层** — 范式的第一次出现, 但仅限 Google 一家, 没形成行业共识

**展望方向 (可以乐观)**: 6-12 个月内, 若 pi_0.7 metadata 范式 / GR00T 人类 video scaling / Gemini Embodied Thinking 三条路线之一出现**全行业跟进**, 能力轴有望跨入 GPT-4 级。但产品化轴仍受限于硬件成本 + 安全监管 + 用例匮乏, ChatGPT 式爆发可能仍需 2-3 年。

---

## 2. 迁移的迁移路径 (速查)

| 路径 | CS 原技术 | Robotics 迁移 | 代表 |
|------|---------|-------------|------|
| A: Token→动作 | GPT autoregressive | 动作离散化为 token | RT-2, OpenVLA |
| B: 视觉→感知 | ViT, CLIP, DINOv2 | VLM backbone | pi_0, GR00T |
| C: 生成→动作 | DDPM, Flow Matching | 连续动作生成 | Diffusion Policy, pi_0 |
| D: 世界模型→规划 | Video prediction | 想象未来再行动 | DreamZero |
| **E: CoT→action 层 thinking** *NEW* | o1/R1 chain-of-thought | action 之间穿插 reasoning trace | **Gemini Robotics 1.5** "Embodied Thinking", pi_0.7 metadata + CFG |
| **F: 长上下文 attention→robot memory** *NEW* | DSA / CSA+HCA / Gated DeltaNet | long-horizon 视频/状态序列 | **DeepSeek V3.2/V4** ↔ MEM (pi_0.6+MEM), 待迁移到 robot |
| **G: AI feedback→robot reward** *NEW* | Constitutional AI / RLAIF | 用 LLM 评判替代 critic | **pi_0.7 metadata** (轻量 advantage), pi*0.6 RECAP, **类 CAI 的 robot constitution 待探索** |
| **H: 人类 video→robot policy** *NEW* | Ego4D / EgoScale | dexterity scaling law | **GR00T N1.7** (20K h human video = 2x task completion) |

### 核心范式: VLM backbone 直接复用

四条路径的交汇点是一个被反复验证的核心范式: **直接复用 CV 预训练的 VLM 作为机器人的视觉理解 backbone, 只需接一个 action head 教它"怎么动"。**

VLM 在互联网图文数据上学到的空间理解 ("红色杯子在桌子左边")、动作语义 ("手正在拿起杯子")、物体状态 ("杯子是倒着的") 对机器人直接有用。action head 不需要理解语言和视觉, 只需要把 VLM 输出的表征映射成关节动作。

验证链条:

| 时间 | 工作 | 验证了什么 |
|------|------|---------|
| 2021 | CLIP | 视觉特征可以跨任务 zero-shot 迁移 |
| 2023 | RT-2 | VLM 的语言理解能力可以转移到机器人指令理解 |
| 2024 | Octo (反例) | 不用 VLM backbone, 从头训 93M → 有效但上限低 |
| 2024 | OpenVLA | VLM backbone fine-tune, 7B 超过 55B RT-2 |
| 2024 | pi_0 | 3B VLM 支撑灵巧操作级别的 VLA |
| 2025 | GR00T N1 | VLM backbone 带来量级级别 data efficiency (10% 数据 > Diffusion Policy 全量) |
| **2025** | **Gemini Robotics 1.0/1.5** *NEW* | **直接复用 frontier Gemini (而非训 robotics-specific VLM), 一个 checkpoint 控制 ALOHA + Franka + Apollo humanoid** |
| **2026.04** | **pi_0.7** *NEW* | **首次出现真正 compositional generalization** — 没采过 air fryer/toaster 数据, 仅靠 verbal coaching 就能完成长 horizon 任务 |
| **2026.04** | **GR00T N1.7** *NEW* | **首条明确的 dexterity scaling law**: 1k → 20k 小时人类 ego video = 任务完成率 2 倍, **relative EEF delta 让人和机器人共享同一动作表征** |

**不用 VLM backbone 的方案 (如 OmniReset 的 ResNet-18 distillation) 反而是需要论证的那个。** OmniReset 在 80K 单任务数据下试过 DINO 和 pi_0.5, 没有显著提升 — 因为数据量不足以 fine-tune 大 backbone (Chinchilla insight)。但多任务大数据场景下, VLM backbone 是必需的。

### 2026 Q2 新发现 — VLM backbone 已经分化为三种用法

| 用法 | 代表 | 特点 |
| --- | --- | --- |
| **训 robotics-specific VLM** | RT-2, pi_0, GR00T N1-N1.7 | 自己训一个适配 robot 的 VLM |
| **复用 frontier VLM, 加 action data 微调** *NEW* | **Gemini Robotics 1.0/1.5** | 不再训独立 VLM, 直接 piggyback Gemini frontier |
| **VLM 当 prompt 生成器, 不当 policy** *NEW* | **pi_0.7 + BAGEL world model** | 14B BAGEL 生成 subgoal image → 喂给 5B VLA, 解耦"想象"和"执行" |

**这三种用法不互斥**, 你可以根据需要选: 自己有大算力训自己的 VLM (NVIDIA 路线); 或 piggyback 现有 frontier (Google 路线); 或用 frontier 做某个特定任务 (PI 用 BAGEL 的路线)。

---

## 3. 渐进式学习路线

**升级考试**: 每个 Level 完成后, 做 [`note/level_exams.md`](note/level_exams.md) 中对应考试。
**考试形式**: 不是"解释 X 论文的方法", 而是"X 的思想怎么改变你的灵巧手/人形工作"。

### Level 0: 什么模式可以从 CS 迁移到 Robotics? (已通过, 90 分)

**要回答的问题**: 为什么好的表征 = 好的 AI? Attention 为什么是通用的?

| # | 内容 | 位置 | 阅读重点 | 时间 |
|---|------|------|---------|------|
| 0.1 | Representation Learning (Bengio) | `foundations/12_RepresentationLearning/` | 好表示的先验列表 | 1h |
| 0.2 | Transformer | `foundations/17_Transformer/` | self-attention 为什么统一了序列建模 | 2h |

**迁移 takeaway**: Transformer + 好的表征 = robotics FM 的两个根基。你后面读到的每一篇论文都建立在这两个概念上。

---

### Level 1: 怎么做大规模预训练?

**要回答的问题**: LLM/CV 怎么从"一个任务一个模型"变成"一个模型所有任务"? 这个模式怎么迁移到 robot?

**范式映射**:
```
LLM: pre-train on text → fine-tune on task → scale → 涌现能力
CV:  pre-train on images (自监督) → fine-tune on task → scale → 通用视觉表征
Robot: pre-train on demo → fine-tune on task → scale (Open-X) → ??? (正在发生)
```

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 1.1 | **LLM 技术交织与机器人启示** | `LLM/LLM_技术交织与机器人启示.md` | 先看地图再走路: LLM 全景 + 哪些技术已迁移到 robot | 1h | 建立"为什么学这些"的动机 |
| 1.2 | **GPT 系列笔记** (Section 1-2, 9) | `LLM/families/GPT_Series/GPT_series_notes.md` | pre-train+fine-tune 范式 + scaling + 商业逻辑 | 2h | robot FM 的核心范式来自 GPT |
| 1.3 | **Scaling Laws + Chinchilla** | `LLM/NLP_foundations/22_Chinchilla/` + GPT notes Section 3 | power-law, 数据和模型等比缩放 | 1.5h | robot data scaling 的理论指导 |
| 1.4 | **MAE** | `CV/4_self_supervised/21_MAE/` | 75% mask 为什么 work; 不需要标注的预训练 | 1.5h | robot 视觉数据没有标注, 必须自监督 |
| 1.5 | **DINOv2** | `CV/4_self_supervised/23_DINOv2/` | 不需要文本也能学强视觉表征; 对比 CLIP | 1h | robot visual representation 研究的重要基础 (R3M/VIP 用它, 但 VLA 更多用 VLM) |
| 1.6 | **CV 技术演进与机器人启示** | `CV/CV_技术演进与机器人启示.md` | CV 全景 + 五条技术线 + robotics 迁移 | 1h | 视觉是 robot 最重要的模态 |

**Level 1 Takeaway**: 大规模预训练的三个要素 — 统一的训练目标 (next-token / masked prediction / motion tracking), 大量数据, power-law scaling。Robot 的 "ImageNet moment" 是 Open-X, "next-token prediction" 是 motion tracking (SONIC)。

---

### Level 2: 怎么从 RL 走向生成式 Policy?

**要回答的问题**: 为什么你的 PPO pipeline 的默认用法不能直接 scale (但 OmniReset 证明改变 reset 分布后 PPO 仍有 scaling 潜力)? 生成模型 (diffusion/flow) 为什么比 RL 更适合做通用 policy?

**范式映射**:
```
你现在做的:  observation → PPO policy (Gaussian) → 单步 action
  问题: 单模态分布, 不能表达"同一场景多种有效动作"
  问题: reward engineering 不可扩展

新范式:      observation → 生成模型 (diffusion/flow) → action chunk
  优势: 多模态分布, 一次生成多步动作, 不需要 reward
  这就是 Diffusion Policy 和 pi_0 在做的事
```

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 2.1 | **CLIP** | `CV/2_vl_alignment/21_CLIP/` | 对比学习做视觉-语言对齐; zero-shot transfer | 2h | 语言指令怎么 ground 到视觉 = robot task specification |
| 2.2 | **DDPM** (notes) | `CV/1_generation/diffusion_family/20_DDPM/` | 前向加噪 + 反向去噪; simplified loss | 2h | Diffusion Policy 的数学基础 |
| 2.3 | **Flow Matching** (notes) | `CV/1_generation/diffusion_family/22_FlowMatching/` | ODE vs SDE; 直线比弯曲更快 | 1.5h | pi_0 和 GR00T 都选了 flow matching |
| 2.4 | **Diffusion Policy** (notes) | `robotics/policy_learning/24_DiffusionPolicy/` | 动作去噪 + action chunk + receding horizon | 2h | **核心**: 从 RL 到生成式 policy 的范式转换 |
| 2.5 | **ACT** (notes) | `robotics/policy_learning/23_ACT/` | CVAE + action chunking 概念 | 1.5h | 多步动作预测减少 compounding error |
| 2.6 | **Decision Transformer** (notes) | `robotics/policy_learning/21_DecisionTransformer/` | RL trajectory = token sequence | 1.5h | 桥接你的 RL 经验: "RL 可以被重新表述为序列建模" |
| 2.7 | **R3M + VIP** (notes) | `robotics/visual_repr/22_R3M/` + `23_VIP/` | 人类视频→robot 视觉表征; 视频=reward | 2h | 不需要 reward engineering, 视频本身就是 reward |

**Level 2 Takeaway**: 你的 PPO 训练 policy 用 Gaussian 输出单步 action → 新范式用 diffusion/flow 输出 action chunk, 天然处理多模态。RL 在 VLA 层面从 "training" 退到 "post-training fine-tuning" (pi*0.6), 但在 WBC/locomotion 层面 RL 仍是主训练范式 (SONIC 全程 PPO) (见 Level 3 的 pi\*0.6)。

**2026 Q2 重要更新 — RL 也在退潮?** pi_0.7 (2026.04) 用更轻量的 **metadata-conditioned BC** 取得了 pi\*0.6 RECAP 同等效果, 不再需要训 critic 和估 advantage:
- 给每条 episode 标 `{quality, mistake, speed}` → 当作 prompt 输入
- 训练时模型学条件分布 `p(action | observation, quality)`
- 部署时永远 prompt `quality=5, mistake=false` + classifier-free guidance
- → 等价于 advantage conditioning, **但不需要 RL pipeline**

这条路线对你 (RL 出身) 的启示有两层:
1. 你的 PPO 经验在 WBC/locomotion 仍然必要 (SONIC), 也仍然适用于 robot post-training (pi\*0.6)
2. 但如果你只是想做 "specialist 蒸馏回 generalist", **metadata 标注 + CFG 可能比 RL 简单 10 倍**, 而且效果同样好。这是 2026 年的新发现, 之前的 roadmap 没体现。

---

### Level 2→3 衔接: VLA 全局概览

在进入具体项目之前, 先建立全局图。Level 0-2 学的是**组件**, Level 3 看的是**组件如何组装成完整系统**。

#### VLA 的核心公式 (所有主流工作共享)

```
图像 + 语言指令 → VLM backbone (理解场景+指令) → Action Head (生成动作) → 执行器

VLM backbone: 来自 CS (CLIP→SigLIP→PaliGemma), 直接复用预训练权重
Action Head:  来自 CS (DDPM→Flow Matching→DiT), 换数据从图像生成变动作生成
训练范式:     来自 LLM (pre-train+fine-tune, scaling law, post-training)
```

两种几何视角 (Level 2 学到的):
- VLM backbone 做**跨空间映射** (图像→latent 表征, 找流形坐标)
- Action Head 做**同空间映射** (噪声→动作, 把点推到流形上)

#### 范式收敛的时间线

```
2022 前: 各方法各自为政 (RL / BC / LLM-planner)
2022:   RT-1 证明 Transformer 在机器人上 work
2023:   RT-2 证明 VLM 知识可迁移到机器人 ← 范式确立
2024:   pi_0 / OpenVLA / Octo 同时验证 ← 范式广泛接受
2025:   GR00T N1 扩展到人形机器人

三个独立技术在 2023 年交汇:
  1. VLM 预训练 backbone 远好于从头训 (CLIP→SigLIP)
  2. 生成模型解决多模态动作分布 (DDPM→Flow Matching)
  3. Transformer 统一所有模态 (image+text+action 同一架构)
```

#### 当前核心问题

| 问题 | 具体表现 | 正在探索的方向 (含 2026 Q2 新进展) |
|------|---------|-----------|
| 精度不够 | VLA 做粗操作好, 亚毫米装配/灵巧操控差 | pi\*0.6 RL post-training; **pi_0.7 用 metadata 替代 RL**; **GR00T RLT 在线 15 分钟微调** |
| 数据太贵 | pi_0 用 10k 小时遥操作 (GPT 用免费互联网数据) | AutoRT 自主采集; DreamZero 想象生成; 仿真合成; **GR00T N1.7 用 20K h EgoScale 人类 video 替代 teleop**; **pi_0.7 verbal coaching (用语言教而非遥操作)** |
| 推理太慢 | VLM ~14Hz, 灵巧手需要 50-120Hz | action chunking (低频生成高频执行); 大脑小脑分离; **pi_0.7 RTC 模拟 0-12 步推理延迟训练** |
| 泛化未知 | 新物体 OK, 新任务/新机器人未知 | 跨 embodiment (Open X); zero-pad action space; **pi_0.7 在 UR5e 折 T 恤匹配人类专家零样本水平 (85.6% vs 80.6%)**; **GR00T N1.7 relative EEF delta 统一人/机器人** |
| 长 horizon | 长任务 context 不够 / 失败检测难 | MEM (pi_0.6+MEM); **DSA / CSA+HCA 长上下文 attention (待迁移)**; **Llama 5 5M context** |
| 推理范式缺失 | VLA 不会"想想再做" | **Gemini Robotics 1.5 Embodied Thinking** (action 层 CoT); o1/R1/V3.2 范式可移植 |
| 安全黑箱 | 不知道模型"为什么"这样动 | **Anthropic Constitutional AI / Sleeper Agents / Alignment Faking 研究框架可借鉴**; mechanistic interpretability 方向待探索 |

#### 你在 VLA 世界的位置

```
你有的 → 对应 VLA 的:
  PPO + reward design         pi*0.6 的 RL post-training (精度校准)
  sim2real (DR, sysid)        GR00T SONIC 的 sim2real pipeline
  motion tracking             SONIC 的 universal tracker
  灵巧手 20 DoF               VLA 精度前沿 (大多数 VLA 只做粗操作)

你缺的:
  VLM 使用经验                需要跑 openpi
  大规模数据工程               需要理解 Open X
  生成模型训练经验             需要看 flow matching 代码
```

---

### Level 3: 完整的 Robot FM 长什么样?

**要回答的问题**: 工业级 robot FM 的架构、训练、部署是怎么设计的? 三个顶级团队各走了什么路?

**阅读方式**: 以 Family Notes 为主, 原论文按需参考。

#### 3.1 RT Family (已完成讨论)

**RT 系列验证了两个核心假设** (从 pattern 压缩视角):

```
RT-1 (2022): robot 数据中有可学的 pattern
  → 130k 遥操作轨迹, 纯 BC
  → 动机不是"知道有 pattern", 而是"NLP/CV 都 work 了, robot 大概也行"
  → 结果: 97% 成功率 + 组合泛化 (模型从数据中压缩出了动作 pattern)

RT-2 (2023): 互联网 pattern 可以迁移到 robot
  → 整个 VLM (55B) fine-tune, 动作当文字 token 输出
  → 核心限制: 动作和语言共享输出头 → 无法冻结 VLM → 太大太慢
  → 但验证了 VLM backbone 复用这个核心范式

Open X (2023): 跨机器人数据集
  → 收益主要在视觉语言层 (所有机器人都看到"杯子")
  → 动作层只在同构型内有效 (轮式和机械臂的动作 pattern 不通用)
  → 下游 (pi_0, Octo) 主要用同构型子集, 不是全部混用

RT → PI 的传承: 同一批人, 同样的 VLM→action 思想, 但解决了 RT-2 的两个问题:
  太大 → 紧凑 VLM (3.3B)
  离散不精确 → Flow Matching (连续动作)
```

**文件**: `robotics/families/Google_RT_Series/RT_family_notes.md`

#### 3.2 PI Family

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 3.2a | **PI Family Notes** | `robotics/families/pi_Series/pi_family_notes.md` | pi_0→FAST→pi_0.5→pi\*0.6→MEM 完整演进 | 2h | Knowledge Insulation + offline RL = 你的 RL 经验直接适用 |
| 3.2b | **pi_0 论文** (精读) | `robotics/families/pi_Series/24_pi0/` | VLM + Flow Matching action expert 的完整设计 | 2h | 当前 VLA 的最佳架构参考 |

#### 3.3 GR00T Family

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 3.3a | **GR00T Family Notes** | `robotics/families/GR00T_Series/GR00T_family_notes.md` | Isaac-GR00T (大脑) + SONIC (小脑) + DreamZero (想象) | 2h | 分层解耦 > 端到端; motion tracking = 人形的统一目标 |
| 3.3b | **SONIC 论文** (精读) | `robotics/families/GR00T_Series/vla_wbc/SONIC/` | 100M 帧 motion tracking + Universal Token Space | 2h | **直接连接你的灵巧手 motion tracking 经验** |
| 3.3c | **GR00T N1 论文** (精读) | `robotics/families/GR00T_Series/vla_wbc/Isaac-GR00T/25_N1/` | 双系统 VLA + data pyramid | 2h | 人形机器人 FM 的工程参考 |
| 3.3d | **GR00T N1.7 报告** *NEW* | `robotics/families/GR00T_Series/vla_wbc/Isaac-GR00T/26_N17/` | **首条 dexterity scaling law** + relative EEF + 20K h EgoScale | 1h | **数据规模 vs prompt 工程的对照** (vs pi_0.7) |

#### 3.4 Gemini Robotics Family *NEW (2025-2026)*

> **位置**: `robotics/families/Google_RT_Series/Gemini_Robotics/Gemini_Robotics_subfamily_notes.md`

Google 在 RT 团队出走后**重新入局**, 不再训独立 robotics-specific VLM, 直接复用 Gemini frontier:

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 3.4a | **Gemini Robotics 子家族 notes** | `Google_RT_Series/Gemini_Robotics/` | GR 1.0 → GR 1.5 → ER 1.6 演进 | 1h | 第三家 robotics FM 路线 |
| 3.4b | **Gemini Robotics 1.5 论文** | `Gemini_Robotics/25_GR15/` (PDF) | **Embodied Thinking + Motion Transfer + 90% MuJoCo 评估** | 2h | **action 层 CoT 是新范式**; **MuJoCo 评估闭环对你直接相关** |

**Level 3 Takeaway** (2026 Q2 更新): **现在是四个团队, 四种哲学**:
- **Google RT (2022-2024)**: 定义了 VLA 范式并验证核心假设, 然后团队出走 (人 > 架构)
- **PI (2024-2026)**: 在 RT-2 基础上解决架构问题 (紧凑 VLM + 连续动作); pi*0.6 用 RL 做 post-training; **pi_0.7 用 metadata 替代 RL, 实现 compositional generalization** (你的 RL 在 pi*0.6 路线有用, 但 pi_0.7 暗示 RL 不再是必经之路)
- **NVIDIA GR00T (2025-2026)**: 分层系统 (VLA + WBC + World Model), 开源全栈; **N1.7 给出 dexterity scaling law: 20K h 人类 video = 2x 任务完成率** (你的 sim2real 在 SONIC 层面有用)
- **Google Gemini Robotics (2025-2026)** *NEW*: 复用 Gemini frontier + Embodied Thinking + agentic orchestrator; 90%+ MuJoCo 评估闭环 (你刚好在用 MuJoCo, 这条路线对你工程实践最直接)

**四家的三个赌注差异 (2026 Q2)**:

| 维度 | PI | NVIDIA | Google Gemini Robotics |
| --- | --- | --- | --- |
| 数据 vs prompt | **prompt 多样性 (metadata + subgoal)** | **数据规模 (20K h human video)** | 复用 Gemini 互联网知识 + 真机 + 90% sim |
| 跨 embodiment | subgoal image (BAGEL 14B 生成) | **Relative EEF delta (人机统一表征)** | **Motion Transfer training recipe** |
| 推理 | metadata + CFG | 标准 VLA | **Embodied Thinking (两层 CoT)** |
| 开源 | 论文公开, 权重闭源 | **完全开源 (Apache 2.0)** | **完全闭源** (API only) |
| 部署 | 云端 API + RTC | 边缘 (Jetson 友好) | Gemini API + Google AI Studio |

---

### Level 4: 下一步和开放问题

**要回答的问题**: Robotics 接下来往哪走? 你的 RL + sim2real 经验在哪里最有价值?

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 4.1 | **DreamZero** (GR00T N2 核心) | `robotics/families/GR00T_Series/world_model/26_DreamZero/` | VLA→WAM: 想象未来再行动, 泛化 >2x VLA | 2h | 下一代架构: 世界模型 = policy |
| 4.2 | **Robot Scaling Laws** | `surveys/robotics/25_RobotScalingLaws/` | Robotics 的 power-law 验证 + 数据瓶颈 | 1.5h | robot 的 scaling 比 LLM 更高效但数据更贵 |
| 4.3 | **FM in Robotics Survey** (IJRR) | `surveys/robotics/23_FMRobotics/` | perception / decision / control 三层分类 | 2h | 全局视野: FM 在 robot 中的应用全景 |
| 4.4 | **World Models Survey** | `surveys/robotics/25_AwesomeWorldModels/` | 世界模型分类学 | 1.5h | DreamZero 的上下文: 这个方向有多大 |
| 4.5 | **pi_0.7 论文 + family Phase 7** *NEW* | `robotics/families/pi_Series/26_pi07/` + family notes Phase 7 | **Diverse Prompting + compositional generalization 的 strong signs + verbal coaching** | 2h | **2026 最值得关注的 robotics 进展之一** — 改变你对 robot RL 的判断 (但注意: 未见任务成功率 60-80%, 仍是雏形) |
| 4.6 | **Gemini Robotics 1.5 论文** *NEW* | `Google_RT_Series/Gemini_Robotics/25_GR15/` | **Embodied Thinking + Motion Transfer + 90% MuJoCo 评估** | 2h | action 层 CoT + 你能直接用的 MuJoCo 评估范式 |
| 4.7 | **GR00T N1.7 报告** *NEW* | `robotics/families/GR00T_Series/vla_wbc/Isaac-GR00T/26_N17/` | **Dexterity scaling law (1k→20k h = 2x)** | 1h | 第一条机器人 scaling law, 改变数据采集策略 |
| **选读** | **DeepSeek V3.2 + V4** *MAJOR UPDATE* | `LLM/families/deepseek/{25_DeepSeekV32,26_DeepSeekV4}/` | **DSA (V3.2) + CSA+HCA + mHC + Muon + FP4 (V4) + 1M context** | 2h | **长上下文 attention 路线对 robot long-horizon 直接迁移** |
| 选读 | **Anthropic Claude family** *NEW* | `LLM/families/Anthropic_Claude/claude_series_notes.md` | Constitutional AI + Sleeper Agents + Computer Use | 1h | **Computer Use = LLM 的"虚拟 VLA"**, alignment 研究框架对 robot 安全有用 |
| 选读 | Llama 5 / Qwen3.5 / Kimi K2.6 | 各家 family notes | dense vs MoE, GDN, agent swarm | 各 30min | 长上下文三种赌注的对比 |
| 选读 | **其他 surveys** | `surveys/CV/` + `surveys/robotics/` | 按需查阅 | 各 1h | 查缺补漏 |

**Level 4 Takeaway** (2026 Q2 重大更新): Robotics 的下一步**出现了几条路线分岔的早期信号** (保守说法: 尚未形成共识, 但方向明确):

**当前保守判断 (现在看到的)** + **方向乐观展望 (可能走向的)** 分开列:

**1. WAM 替代 VLA** (DreamZero: 想象→做, 而非 看→做)
- 现在: 仍是中长期方向, 只有 GR00T N2 一家押注
- 展望 (可乐观): 世界模型训练成本下降 + 视频生成质量持续提升, 1-2 年内可能成为第二主流架构

**2. 数据飞轮** — 两条新范式**雏形**已出现, 但未被广泛复制
- 老观点: 自主采集 > 人工遥操作
- **NEW**: 人类 video > robot teleop (GR00T N1.7 dexterity scaling law, **但只有 NVIDIA 一家验证**)
- **NEW**: verbal coaching > teleop (pi_0.7 air fryer/toaster 任务, **但只有 PI 一家在做**)
- 展望 (可乐观): 若两条路线在 6-12 个月内被其他团队独立复现, data collection 成本可能降一个数量级

**3. RL 的角色再次变化** — 判断需要非常小心
- 老观点: RL 从 training 退到 post-training
- **观察到的信号**: pi_0.7 用 metadata-conditioned BC 在几个任务上匹配了 pi*0.6 RECAP 的效果
- **但不等于**: RL 被替代 — 范围有限, 证据单一 (PI 自己对比), 在 WBC/locomotion 层 RL 仍是唯一可行训法
- 展望 (可乐观): 你的 RL + sim2real 经验**仍然是核心竞争力**, 适用范围是 **WBC + 在线精调 + sim2real 闭环**, 这些是 metadata conditioning 替代不了的

**4. NEW: Embodied Thinking 是值得关注的新方向** (而不是"新范式已确立")
- Gemini Robotics 1.5 第一次做 action 层 CoT
- pi_0.7 的 metadata + CFG 和 GR00T 的 RL Token 是另外两种不同实现
- 现在: 3 家 3 种方案, 没有行业共识
- 展望 (可乐观): 6-12 个月内可能看到一种方案跑出来, 成为类似 LLM o1 的全行业范式

**5. NEW: 长上下文 attention 革命** (LLM 侧)
- 现在: DeepSeek V3.2/V4 + Qwen3.5 + Llama 5 已经把 1M-5M context 做成 LLM 标配
- 但 robot 侧**还没开始迁移** — robot long-horizon 的 attention 工程瓶颈仍在
- 展望 (可乐观): 这条路迁移到 robot 是**低挂果实** — 三条路线 (DSA / GDN / dense + 5M) 已经被 LLM 验证, robot VLA backbone 直接 plugin 理论上应该能 work

**6. NEW: Computer Use 是 LLM-to-VLA 的中间形态** (Claude 3.5+)
- 现在: Anthropic Opus 4.7 的 Computer Use 可以自主点鼠标+输入键盘完成 UI 任务
- 这本质是**虚拟环境上的 VLA**, 动作空间是 click/type 而不是关节角
- 展望 (可乐观): 真实 VLA 可以先在虚拟环境训练 (便宜 × 可规模化), 再迁移到物理机器人 — 完整路径待验证

---

### 本路线图的局限性 (基于 survey 交叉验证)

以下是本路线图**有意识的简化和遗漏**, 基于 IJRR FM Survey、General-Purpose Robots Survey、Robot Scaling Laws Survey 的交叉验证:

**1. VLA 不是唯一路线**

本路线图重点覆盖端到端 VLA (RT→pi_0→GR00T), 但 IJRR survey 给予 **LLM-as-planner** (SayCan, Code-as-Policies, ProgPrompt) 和 **open-vocabulary perception** (Grounding DINO, CLIP-Fields) 同等权重。这些模块化方案在工业部署中可能比端到端 VLA 更实用 — VLA 的推理频率 (1-10Hz) 远不能满足全身控制 (需 500Hz)。RT family notes Phase 1 覆盖了 LLM-as-planner, 但路线图主线未纳入。

**2. Safety / Uncertainty 完全未覆盖**

两份 survey 都将 safety 和 uncertainty quantification 列为**核心挑战/最薄弱环节**。本路线图没有任何 safety 相关内容。如果你的工作涉及真机部署, 这是一个必须单独学习的方向。

**3. Scaling 的统计置信度有限**

Robot Scaling Laws survey 的结论基于 327 篇论文的 meta-analysis, 但大多数 scaling study 仅有 **2-3 个数据点**, compute scaling 维度 **几乎空白** (327 篇中仅 1 篇)。Power-law 趋势存在但不如 LLM 的 Kaplan/Chinchilla 那样可靠。且 survey 强调 **data diversity > data quantity** — seen task 的 scaling 效率是 unseen 的 2.5 倍。

**4. VLA 的已知瓶颈**

General-Purpose Robots survey 指出 VLA 在 unseen task 上性能下降 **21-31%**, 且缺乏统一 benchmark (不同论文的 success rate 定义不同)。本路线图呈现了 VLA 的能力但未充分警告其局限。

**5. 数据增广是被低估的方向**

IJRR survey 系统讨论了多种 data augmentation 方案 (ROSIE 语义增广、GenAug 生成式增广、CACTI 上下文增广), 这些在数据稀缺的 robotics 中极其重要。你的知识库中 GR00T DreamGen 和 UltraDexGrasp 的合成数据 pipeline 属于这个方向, 但路线图未将数据增广作为独立主题。

**数据增广的范式 (2026 Q2 更新, 现在是四种)**:

| 范式 | 方法 | 代表 | 与你的关联 |
|------|------|------|---------|
| **Domain Randomization** | 在 sim 中随机化视觉/物理参数 | SONIC 的 DR、你的 sim2real 工作 | 你已经在用, 但可以更系统化 |
| **生成式增广** | 用生成模型 (diffusion/video) 创建新场景/视角 | GR00T DreamGen (Cosmos 世界模型)、ROSIE、GenAug、**pi_0.7 BAGEL subgoal generation** | DreamGen 证明 11h→6500h 等效, 数据飞轮的核心 |
| **合成数据 Pipeline** | 自动化生成大规模标注数据 | UltraDexGrasp (BODex 20M 帧)、AutoRT 思路 (VLM 自动分配任务, 论文已移除) | 完全绕过人工数据采集瓶颈 |
| **NEW: 人类 video 直接迁移** *2026* | 把 human ego video 当 robot demo 用 | **GR00T N1.7** (20K h EgoScale, dexterity scaling law) | **relative EEF delta 是关键**: 人和机器人共享同一动作表征就能直接 co-train |
| **NEW: Verbal Coaching** *2026* | 人口头一步步指导, 不操控机器人 | **pi_0.7** (air fryer/toaster 任务) | 比遥操作便宜 10 倍以上, coaching 数据反向训 high-level 策略 |

> **对你的启示** (2026 Q2 更新): 你的 sim2real 经验 (Domain Randomization) 是数据增广的第一种范式。**2026 上半年新出现了两个范式**:
> - **GR00T N1.7 的人类 video 路线**告诉你: 与其采更多 robot teleop, 不如收集人类 ego video
> - **pi_0.7 的 verbal coaching** 告诉你: 长 horizon 任务的数据采集可以从遥操作转向 "人口头说"
> 这两个方向都值得关注, 因为它们从根本上改变了你对 "robot 数据成本" 的理解。

**6. NEW: Robot 推理范式被低估**

老 roadmap 没有"推理"这个独立主题。但 2026 出现了三条 robot 推理路线:
- **Embodied Thinking** (Gemini Robotics 1.5): 在 action 之前/之间穿插自然语言 reasoning trace
- **Metadata + CFG** (pi_0.7): 用 quality/mistake 标签 + classifier-free guidance, 等价于 condition on advantage
- **RL Tokens** (GR00T RLT): VLA 输出 action token + RL token, 后者由轻量 actor-critic 做 15 分钟在线 RL

这三条路线共同回答 "VLA 怎么学会想清楚再做" 的问题。值得作为独立主题学习。

**7. NEW: 推理时计算 (test-time scaling) 在 robot 侧的对应**

LLM 侧 o1/R1/V3.2 Speciale 已经验证: 推理时多花算力 = 性能更好。Robot 侧的对应是什么?
- pi_0.7 的多步 thinking + tool 编排
- Gemini Robotics 1.5 的 orchestrator 多次推理 + subgoal 刷新
- DreamZero 的 imagine N frames 再 extract action
- 这是一条没有明确名字的路线, 值得关注

---

## 4. 你的知识库全景

```
foundation_model/
├── foundations/              # 通用 ML 基础 (9 篇)
│   ├── 10_TransferLearning, 12_RepresentationLearning, 14_GAN
│   ├── 15_Adam, 15_BatchNorm, 16_LayerNorm
│   ├── 17_PPO, 17_Transformer, 18_SAC
├── LLM/                     # LLM 知识体系
│   ├── NLP_foundations/     #   Word2Vec→BERT→Chinchilla (5)
│   ├── families/            #   GPT, Kimi, Qwen, DeepSeek, Llama (5 家族)
│   └── LLM_技术交织与机器人启示.md
├── CV/                      # CV 知识体系 (24 篇)
│   ├── 0_backbone/          #   ResNet, ViT, Swin, TransferFeatures
│   ├── 1_generation/        #   VAE, DDPM, LDM, FlowMatching, DiT
│   ├── 2_vl_alignment/      #   CLIP, BLIP-2, LLaVA, PaliGemma
│   ├── 3_3d_vision/         #   NeRF, 3DGS, DepthAnything
│   ├── 4_self_supervised/   #   MoCo, SimCLR, DINO, BEiT, MAE, DINOv2
│   ├── 5_detection_seg/     #   DETR, SAM
│   ├── 6_video/             #   TimeSformer, ViViT, VideoMAE, Ego4D
│   └── CV_技术演进与机器人启示.md
├── robotics/                # Robotics 应用
│   ├── families/            #   Google_RT_Series, pi_Series, GR00T_Series
│   ├── policy_learning/     #   DT, ACT, DiffusionPolicy, DROID
│   ├── vla/                 #   Octo, OpenVLA
│   ├── visual_repr/         #   R3M, VIP
│   └── world_model/         #   DreamerV3, UniSim
└── surveys/                 # CV (7) + Robotics (7)
```

---

## 5. 一句话总结每篇的迁移意义

| 论文 | 对你 (RL→FM) 的迁移意义 |
|------|----------------------|
| Transformer (2017) | 所有 FM 的骨架, 从 NLP 到 CV 到 robot policy |
| GPT 系列 (2018-2023) | 定义了 pre-train+fine-tune+scale 三步走, robot FM 完全继承 |
| Scaling Laws (2020) | 性能可预测, 投入产出有数学保证 |
| MAE (2021) | 不用标注的视觉预训练, robot 数据没有标注所以必须自监督 |
| DINOv2 (2023) | 当前最强自监督视觉 backbone, 不需要文本 |
| CLIP (2021) | 语言指令怎么 ground 到视觉, robot task specification 的基础 |
| DDPM (2020) | 连续分布生成, Diffusion Policy 的数学基础 |
| Flow Matching (2022) | 比 diffusion 更快更稳, pi_0 和 GR00T 都选了它 |
| Decision Transformer (2021) | "RL = 序列建模" — 桥接你的 PPO 经验到 Transformer 世界 |
| Diffusion Policy (2023) | **范式转换**: 从 RL 的 Gaussian policy 到 diffusion 的多模态 action |
| ACT (2023) | action chunking 减少 compounding error |
| R3M + VIP (2022-23) | 人类视频→robot 表征→自动 reward, 不需要 reward engineering |
| RT Family (2022-24) | VLA 起源: 从 LLM-as-planner 到端到端, 再到团队出走做 PI |
| PI Family (2024-26) | VLA 纵深: tokenizer→泛化→offline RL→记忆→**Diverse Prompting (pi_0.7)**, RL 退潮 |
| GR00T Family (2025-26) | 分层系统: VLA (大脑) + SONIC (小脑) + DreamZero (想象), **N1.7 给出 dexterity scaling law**, 全栈开源 |
| **Gemini Robotics (2025-26)** *NEW* | **第四个 robotics FM 主线**: 复用 Gemini frontier + Embodied Thinking + agentic orchestrator + 90% MuJoCo 评估 |
| SONIC (2025) | **你最该精读的**: motion tracking at scale = 人形的统一可扩展目标 |
| **pi_0.7 (2026.04)** *NEW* | **2026 最值得关注的 robotics 进展之一**: Diverse Prompting → compositional generalization 的 strong signs (未见任务 60-80% 仍明显落后 in-distribution >90%) + 跨 embodiment 匹配人类专家零样本 + verbal coaching 替代遥操作的范式雏形 |
| **GR00T N1.7 (2026.04)** *NEW* | **第一条机器人 dexterity scaling law**: 1k → 20k h 人类 video = 2x 任务完成率 + relative EEF delta 统一人/机器人表征 |
| **Gemini Robotics 1.5 (2025.10)** *NEW* | **Embodied Thinking 范式**: action 层 chain-of-thought + Motion Transfer 让一个 ckpt 控制 ALOHA/Franka/Apollo + 90%+ 评估在 MuJoCo |
| **DeepSeek V3.2 + V4 (2025.12-2026.04)** *NEW* | **长上下文 attention 革命**: DSA + CSA+HCA + mHC + Muon + FP4 + 1M context, 直接对 robot long-horizon 有迁移价值 |
| **Anthropic Claude Opus 4.7 + CAI (2022-2026)** *NEW* | **Computer Use = 虚拟 VLA**, Constitutional AI 是 robot reward design 的方法论参考, alignment 研究框架对 robot 安全有用 |
| DreamZero (2026) | 下一代: 世界模型 = policy, 想象→做 替代 看→做 |

---

## 6. 学者指引

| 学者 | 与你的关联 |
|------|---------|
| **Sergey Levine** (Berkeley/PI) | RL→Robot FM 转型的标杆人物, pi_0 联合创始人 |
| **Chelsea Finn** (Stanford/PI) | Meta-learning→VLA, 和你一样从 RL 出发 |
| **Karol Hausman** (PI CEO) | SayCan→RT-2→pi_0, VLA 定义者 |
| **罗正宜 Zhengyi Luo** (NVIDIA) | PHC→SONIC, motion tracking 从小规模到 FM 级 |
| **Jim Fan** (NVIDIA) | GR00T 系列, Foundation Agent 倡导者 |
| **Yuke Zhu** (NVIDIA / UT Austin) | GR00T GEAR lab 共同负责人, 灵巧操作经典 (Robosuite, MimicGen) |
| **Shuran Song** (Stanford) | Diffusion Policy 指导教授 (一作 Cheng Chi) |
| **Kaiming He** (MIT) | ResNet→MoCo→MAE, 视觉自监督的定义者 |
| **梁文锋 Liang Wenfeng** (DeepSeek) *NEW* | 量化基金转 LLM, MLA / MoE / GRPO / FP8 / mHC 的源头, V4 1M context 推动者 |
| **杨植麟 Yang Zhilin** (Moonshot/Kimi) *NEW* | Transformer-XL 一作, 长上下文 + Muon 优化器路线, K2.6 agent swarm 范式 |
| **Dario Amodei + Daniela Amodei** (Anthropic) *NEW* | OpenAI 出走, Constitutional AI / Sleeper Agents / Computer Use 路线; 对 robot RL post-training 安全性研究有方法论价值 |
| **Gemini Robotics Team** (Google DeepMind) *NEW* | 第四家 robotics FM 主线, Embodied Thinking + Motion Transfer + 90% MuJoCo 评估范式 |
