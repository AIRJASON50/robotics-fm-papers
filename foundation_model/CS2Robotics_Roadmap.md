# CS to Robotics: Foundation Model 学习路线图

> **你的背景**: PPO sim2real 灵巧手操作, 有 RL 和机器人实践经验
> **你的目标**: 理解 CS (LLM+CV) 的发展如何塑造了现代 robotics FM, 学习方法论和设计思想
> **不做什么**: 不复现代码, 不成为 LLM/CV 专家
> **核心洞察**: Robotics 正在重走 LLM/CV 的路, 学习这条路 = 看到 robotics 的未来

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
2020  Scaling Laws (power-law)           2025  Robot Scaling Laws (验证中)
2022  InstructGPT (RLHF post-training)   2025  pi*0.6 (offline RL fine-tune VLA)
2022  ChatGPT (产品化)                    ????  还没到
2024  o1 (推理时计算)                     2026  DreamZero (想象再行动)

>>> Robotics 今天 ≈ LLM 的 InstructGPT 到 ChatGPT 之间 <<<
>>> 知道 scaling 有用, 正在探索 post-training, 还没有产品化时刻 <<<
```

---

## 2. 迁移的四条路径 (速查)

| 路径 | CS 原技术 | Robotics 迁移 | 代表 |
|------|---------|-------------|------|
| A: Token→动作 | GPT autoregressive | 动作离散化为 token | RT-2, OpenVLA |
| B: 视觉→感知 | ViT, CLIP, DINOv2 | VLM backbone | pi_0, GR00T |
| C: 生成→动作 | DDPM, Flow Matching | 连续动作生成 | Diffusion Policy, pi_0 |
| D: 世界模型→规划 | Video prediction | 想象未来再行动 | DreamZero |

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
| 2.2 | **DDPM** (notes) | `CV/1_generation/20_DDPM/` | 前向加噪 + 反向去噪; simplified loss | 2h | Diffusion Policy 的数学基础 |
| 2.3 | **Flow Matching** (notes) | `CV/1_generation/22_FlowMatching/` | ODE vs SDE; 直线比弯曲更快 | 1.5h | pi_0 和 GR00T 都选了 flow matching |
| 2.4 | **Diffusion Policy** (notes) | `robotics/policy_learning/24_DiffusionPolicy/` | 动作去噪 + action chunk + receding horizon | 2h | **核心**: 从 RL 到生成式 policy 的范式转换 |
| 2.5 | **ACT** (notes) | `robotics/policy_learning/23_ACT/` | CVAE + action chunking 概念 | 1.5h | 多步动作预测减少 compounding error |
| 2.6 | **Decision Transformer** (notes) | `robotics/policy_learning/21_DecisionTransformer/` | RL trajectory = token sequence | 1.5h | 桥接你的 RL 经验: "RL 可以被重新表述为序列建模" |
| 2.7 | **R3M + VIP** (notes) | `robotics/visual_repr/22_R3M/` + `23_VIP/` | 人类视频→robot 视觉表征; 视频=reward | 2h | 不需要 reward engineering, 视频本身就是 reward |

**Level 2 Takeaway**: 你的 PPO 训练 policy 用 Gaussian 输出单步 action → 新范式用 diffusion/flow 输出 action chunk, 天然处理多模态。RL 在 VLA 层面从 "training" 退到 "post-training fine-tuning" (pi*0.6), 但在 WBC/locomotion 层面 RL 仍是主训练范式 (SONIC 全程 PPO) (见 Level 3 的 pi\*0.6)。

---

### Level 3: 完整的 Robot FM 长什么样?

**要回答的问题**: 工业级 robot FM 的架构、训练、部署是怎么设计的? 三个顶级团队各走了什么路?

**范式映射**:
```
LLM:  GPT (基础) → InstructGPT (对齐) → ChatGPT (产品)
Robot: RT-1 (基础) → pi_0/GR00T (对齐+部署) → ??? (产品化)

你在 Level 3 要读的: 这三个阶段的完整故事
```

**阅读方式**: 以 Family Notes 为主, 原论文按需参考。Family Notes 已经是 takeaway 驱动的思维提炼, 所有论文都有基于原文的自包含 notes, 只看 notes 即可理解核心想法。

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 3.1 | **RT Family Notes** | `robotics/families/Google_RT_Series/RT_family_notes.md` | VLA 起源: LLM-as-planner → RT-1 → RT-2 → 团队出走 | 2h | VLA 的核心假设: web 知识可以迁移到 robot |
| 3.2 | **PI Family Notes** | `robotics/families/pi_Series/pi_family_notes.md` | pi_0→FAST→pi_0.5→pi\*0.6→MEM 完整演进 | 2h | Knowledge Insulation + offline RL = 你的 RL 经验直接适用 |
| 3.3 | **GR00T Family Notes** | `robotics/families/GR00T_Series/GR00T_family_notes.md` | Isaac-GR00T (大脑) + SONIC (小脑) + DreamZero (想象) | 2h | 分层解耦 > 端到端; motion tracking = 人形的统一目标 |
| 3.4 | **pi_0 论文** (精读) | `robotics/families/pi_Series/24_pi0/` | VLM + Flow Matching action expert 的完整设计 | 2h | 当前 VLA 的最佳架构参考 |
| 3.5 | **SONIC 论文** (精读) | `robotics/families/GR00T_Series/vla_wbc/SONIC/` | 100M 帧 motion tracking + Universal Token Space | 2h | **直接连接你的灵巧手 motion tracking 经验** |
| 3.6 | **GR00T N1 论文** (精读) | `robotics/families/GR00T_Series/vla_wbc/Isaac-GR00T/25_N1/` | 双系统 VLA + data pyramid | 2h | 人形机器人 FM 的工程参考 |

**Level 3 Takeaway**: 三个团队, 三种哲学:
- Google RT: 定义了 VLA, 然后团队出走 (人 > 架构)
- PI: 一条 VLA 路线走到底, RL 做 post-training (你的 RL 在这里有用)
- NVIDIA: 分层系统 (VLA + WBC + World Model), 开源全栈 (你的 sim2real 在这里有用)

---

### Level 4: 下一步和开放问题

**要回答的问题**: Robotics 接下来往哪走? 你的 RL + sim2real 经验在哪里最有价值?

| # | 内容 | 位置 | 阅读重点 | 时间 | 迁移点 |
|---|------|------|---------|------|--------|
| 4.1 | **DreamZero** (GR00T N2 核心) | `robotics/families/GR00T_Series/world_model/26_DreamZero/` | VLA→WAM: 想象未来再行动, 泛化 >2x VLA | 2h | 下一代架构: 世界模型 = policy |
| 4.2 | **Robot Scaling Laws** | `surveys/robotics/25_RobotScalingLaws/` | Robotics 的 power-law 验证 + 数据瓶颈 | 1.5h | robot 的 scaling 比 LLM 更高效但数据更贵 |
| 4.3 | **FM in Robotics Survey** (IJRR) | `surveys/robotics/23_FMRobotics/` | perception / decision / control 三层分类 | 2h | 全局视野: FM 在 robot 中的应用全景 |
| 4.4 | **World Models Survey** | `surveys/robotics/25_AwesomeWorldModels/` | 世界模型分类学 | 1.5h | DreamZero 的上下文: 这个方向有多大 |
| 选读 | **DeepSeek 系列笔记** | `LLM/families/deepseek/deepseek_series_notes.md` | MLA + MoE + GRPO 的效率创新 | 1h | 架构效率思路可迁移到 robot FM |
| 选读 | **其他 surveys** | `surveys/CV/` + `surveys/robotics/` | 按需查阅 | 各 1h | 查缺补漏 |

**Level 4 Takeaway**: Robotics 的下一步可能是:
1. **WAM 替代 VLA** (DreamZero: 想象→做, 而非 看→做)
2. **数据飞轮** (自主采集 > 人工遥操作)
3. **RL 回归** (不是做 training, 而是做 post-training fine-tuning)
4. **你的差异化**: RL + sim2real 经验在 post-training 阶段最有价值 (pi\*0.6 路线)

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
| PI Family (2024-26) | VLA 纵深: tokenizer→泛化→offline RL→记忆, RL 作为 post-training |
| GR00T Family (2025-26) | 分层系统: VLA (大脑) + SONIC (小脑) + DreamZero (想象), 全栈开源 |
| SONIC (2025) | **你最该精读的**: motion tracking at scale = 人形的统一可扩展目标 |
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
| **Shuran Song** (Stanford) | Diffusion Policy 指导教授 (一作 Cheng Chi) |
| **Kaiming He** (MIT) | ResNet→MoCo→MAE, 视觉自监督的定义者 |
