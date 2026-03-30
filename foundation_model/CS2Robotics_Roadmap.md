# CS to Robotics: Foundation Model 迁移路线图

本文档梳理 CS 领域 (NLP/CV) 的核心技术如何逐步迁移到机器人控制，提供渐进式学习路线。

---

## 1. 迁移主线: 从语言到动作

```
Phase 1: 表示学习的觉醒 (2012-2017)
  Bengio 表示学习综述 (2012) -> Word2Vec (2013) -> Transformer (2017)
  核心突破: "好的表示 = 好的 AI"; attention 机制统一了序列建模

Phase 2: 语言预训练革命 (2018-2020)
  GPT-1 (2018) -> BERT (2018) -> GPT-2 (2019) -> GPT-3 (2020)
  核心突破: pre-train + fine-tune 范式确立; scaling 带来涌现能力

Phase 3: 视觉-语言对齐 + 生成模型 (2020-2022)
  ViT (2020) + DDPM (2020) -> CLIP (2021) -> Flow Matching (2022)
  核心突破: Transformer 统一 CV/NLP; diffusion 实现连续分布建模

Phase 4: Scaling Laws 理论化 (2020-2022)
  Kaplan Scaling Laws (2020) -> Chinchilla (2022)
  核心突破: 模型-数据等比缩放; 数据量和模型量同等重要

Phase 5: 第一次迁移 -- RL as Sequence Modeling (2021-2023)
  Decision Transformer (2021) -> DreamerV3 (2023)
  核心突破: Transformer 编码 trajectory 和 world dynamics

Phase 6: 第二次迁移 -- Diffusion for Actions (2023)
  Diffusion Policy (2023)
  核心突破: 图像 diffusion 直接用于机器人连续动作生成

Phase 7: 第三次迁移 -- VLA 统一模型 (2022-2025)
  RT-1 (2022) -> RT-2 (2023) -> pi_0 (2024) -> GR00T N1 (2025)
  核心突破: VLM 直接输出机器人动作, 继承互联网规模知识
```

---

## 2. 技术迁移的四条路径

### 路径 A: 语言编码 -> 动作编码 (Token 化)

| 阶段 | CS 原技术 | 机器人迁移 | 关键论文 |
|------|---------|---------|--------|
| 文本 tokenization | BPE / SentencePiece | 动作离散化为 token | RT-2, OpenVLA |
| Autoregressive generation | GPT next-token prediction | 逐步预测动作 token | Decision Transformer |
| Action chunking | -- | 一次预测多步动作 | ACT, pi_0-FAST |
| 连续化改进 | Flow matching (图像生成) | Flow matching for actions | pi_0 |

### 路径 B: 视觉理解 -> 场景感知

| 阶段 | CS 原技术 | 机器人迁移 | 关键论文 |
|------|---------|---------|--------|
| Image classification | ViT (2020) | 视觉特征提取 | -- |
| Vision-Language alignment | CLIP (2021) | 语言指令理解 + 视觉 grounding | CLIPort |
| VLM | PaLM-E, PaliGemma | VLA 的视觉-语言 backbone | pi_0, GR00T N1 |
| Open-vocabulary detection | OWL-ViT, Grounding DINO | 零样本物体检测 | surveys/robotics/23_FMRobotics |

### 路径 C: 生成模型 -> 动作生成

| 阶段 | CS 原技术 | 机器人迁移 | 关键论文 |
|------|---------|---------|--------|
| DDPM | 图像去噪生成 | 动作去噪生成 | Diffusion Policy |
| Conditional generation | Classifier-free guidance | 观测条件下的动作生成 | Diffusion Policy |
| Flow matching | Rectified flow (图像) | Rectified flow for actions | pi_0 |
| DiT | Diffusion Transformer (图像) | Diffusion Transformer for actions | GR00T N1 |

### 路径 D: 世界模型 -> 仿真+规划

| 阶段 | CS 原技术 | 机器人迁移 | 关键论文 |
|------|---------|---------|--------|
| Latent dynamics | VAE + RNN | 潜空间世界模型 | DreamerV3 |
| Video prediction | Sora, video diffusion | 物理场景预测 | surveys/robotics/25_AwesomeWorldModels |
| Imagination-based RL | 在世界模型中训练 policy | 减少真实交互需求 | DreamerV3 |

---

## 3. 渐进式阅读推荐

**升级考试**: 每个 Level 完成后，做 [`note/level_exams.md`](note/level_exams.md) 中对应考试 (10 题)，80% 进入下一级。

### Level 0: 表示与编码基础 (已通过, 90分)

**目标**: 理解"为什么编码/表示方法如此重要"，建立 Transformer + autoregressive 直觉。
**考试**: Level 0 -> 1 入门考试 | **讨论笔记**: [`note/Level0_discussion_notes.md`](note/Level0_discussion_notes.md)

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 0.1 | **Representation Learning** (Bengio 2012) | `foundations/12_RepresentationLearning/` | Section 1-2, 5: 为什么好的表示 = 好的 AI; 好表示的先验列表 | 1h |
| 0.2 | **Attention Is All You Need** (2017) | `foundations/17_Transformer/` | Section 3: self-attention, multi-head, positional encoding -- 所有后续工作的根基 | 2h |

### Level 1: LLM -- 从 GPT 到现代大模型

**目标**: 理解 pre-train + fine-tune 范式、scaling law、MoE 架构。

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 1.1 | **GPT-1** (2018) | `LLM/families/GPT_Series/18_GPT1/` | Section 1-3: pre-train + fine-tune 范式确立 | 1h |
| 1.2 | **GPT-2 代码** | `LLM/families/GPT_Series/19_GPT2/code/src/model.py` | 175 行代码理解完整 Transformer decoder | 2h |
| 1.3 | **Scaling Laws** (2020) | `LLM/families/GPT_Series/20_ScalingLaws/` | power-law 关系, 为 GPT-3 的 175B 提供理论指导 | 1h |
| 1.4 | **GPT-3** (选读) | `LLM/families/GPT_Series/20_GPT3/` | Section 1 (Figure 1.1-1.3): in-context learning 涌现 | 2h |
| 1.5 | **GPT 系列笔记** | `LLM/families/GPT_Series/GPT_series_notes.md` | 全系列分析 + 商业逻辑 | 1h |
| 1.6 | **Chinchilla** (2022) | `LLM/NLP_foundations/22_Chinchilla/` | 模型和数据应等比缩放 | 1h |
| 1.7 | **LLM→Robotics 技术路线** | `LLM/LLM_技术交织与机器人启示.md` | LLM 技术分岔全景 + robotics distill 路径 | 1h |
| 1.8 | **Kimi/Qwen/DeepSeek** (选读) | `LLM/families/kimi/`, `qwen/`, `deepseek/` | MoE (K2), Muon 优化器 (Moonlight), 数据飞轮 (Qwen) | 2h |

### Level 2: 视觉-语言 + 生成模型

**目标**: 理解 CLIP 如何对齐视觉和语言、diffusion 如何生成连续分布。

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 2.1 | **ViT** (2020) | `CV/0_backbone/20_ViT/` | patch embedding, position embedding, cls token | 2h |
| 2.2 | **ViT 代码** | `CV/0_backbone/20_ViT/vision_transformer/` | 模型实现, 与 ResNet 对比 | 1h |
| 2.3 | **CLIP** (2021) | `CV/2_vl_alignment/21_CLIP/` | contrastive learning, dual encoder, zero-shot transfer | 2h |
| 2.4 | **CLIP 代码** | `CV/2_vl_alignment/21_CLIP/CLIP/` | model.py: ViT + text encoder + contrastive loss | 1h |
| 2.5 | **DDPM** (2020) | `CV/1_generation/20_DDPM/` | forward noise, reverse denoising, simplified loss | 2h |
| 2.6 | **DDPM 代码** | `CV/1_generation/20_DDPM/diffusion/` | U-Net + timestep embedding + 训练循环 | 1h |
| 2.7 | **Flow Matching** (2022) | `CV/1_generation/22_FlowMatching/` | ODE-based 生成, pi_0 的核心技术 | 2h |
| 2.8 | **Flow Matching 代码** | `CV/1_generation/22_FlowMatching/flow_matching/` | Meta 官方库 | 1h |
| 2.9 | **DiT** (2023) | `CV/1_generation/23_DiT/` | Transformer 做 diffusion backbone, GR00T N1 的基础 | 2h |
| 2.10 | **DiT 代码** | `CV/1_generation/23_DiT/DiT/` | adaLN-Zero 条件注入 | 1h |

### Level 3: 第一次迁移 -- RL/Robotics meets Transformer

**目标**: 理解 Transformer 如何从语言迁移到 RL 和机器人。

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 3.1 | **Decision Transformer** (2021) | `robotics/policy_learning/21_DecisionTransformer/` | RL = sequence modeling, reward conditioning | 2h |
| 3.2 | **DreamerV3** (2023) | `robotics/world_model/23_DreamerV3/` | 世界模型: 在"想象"中训练 | 3h |
| 3.3 | **ACT** (2023) | `robotics/policy_learning/23_ACT/` | CVAE + action chunking | 2h |
| 3.4 | **ACT 代码** | `robotics/policy_learning/23_ACT/act/` | 简洁的 imitation learning pipeline | 1h |
| 3.5 | **Diffusion Policy** (2023) | `robotics/generative_policy/24_DiffusionPolicy/` | 条件去噪 + action chunk + receding horizon | 3h |
| 3.6 | **Diffusion Policy 代码** | `robotics/generative_policy/24_DiffusionPolicy/diffusion_policy/` | 完整 policy 实现 | 2h |

### Level 4: VLA 统一模型 (当前前沿)

**目标**: 理解 VLM + action generation 的完整架构。

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 4.1 | **RT-1** (2022) | `robotics/policy_learning/22_RT1/` | 第一个大规模 robot Transformer | 1.5h |
| 4.2 | **RT-2** (2023) | `robotics/policy_learning/23_RT2/` | 第一个 VLA: VLM 直接输出离散动作 token | 2h |
| 4.3 | **Open X-Embodiment** (2023) | `robotics/policy_learning/23_OpenXEmbodiment/` | 跨机器人数据集, 所有 VLA 的训练数据基础 | 1.5h |
| 4.4 | **PaliGemma** (2024) | `CV/2_vl_alignment/24_PaliGemma/` | SigLIP + Gemma 2B, pi_0 的 VLM backbone | 1.5h |
| 4.5 | **Octo** (2024) | `robotics/vla/24_Octo/` | 开源 generalist robot policy, pi_0 的前身 | 2h |
| 4.6 | **OpenVLA** (2024) | `robotics/vla/24_OpenVLA/` | 开源 VLA baseline, LoRA fine-tuning | 2h |
| 4.7 | **pi_0** (2024) | `robotics/generative_policy/24_pi0/` | action expert + flow matching | 3h |
| 4.8 | **pi_0 代码** | `robotics/generative_policy/24_pi0/openpi/` | dual-expert attention, flow matching loss | 3h |
| 4.9 | **GR00T N1** (2025) | `robotics/generative_policy/25_GR00T_N1/` | 双系统: VLM (10Hz) + DiT (120Hz) | 2h |
| 4.10 | **GR00T N1 代码** | `robotics/generative_policy/25_GR00T_N1/Isaac-GR00T/` | 训练、评估、部署全流程 | 2h |

### Level 5: 综述 (建立全局视野, 按需阅读)

**目标**: 系统性理解领域全貌, 查缺补漏。

| 顺序 | 论文 | 在本库位置 | 阅读重点 | 预计时间 |
|------|------|---------|---------|---------|
| 5.1 | **Foundation Models in Robotics** | `surveys/robotics/23_FMRobotics/` | 三层分类: perception / decision / control | 2h |
| 5.2 | **Bridging Language and Action** | `surveys/robotics/23_LangCondManip/` | 语言在控制回路中的四种角色 | 2h |
| 5.3 | **Scaling Laws in Robotics** | `surveys/robotics/25_RobotScalingLaws/` | 机器人性能是否遵循 scaling law | 1.5h |
| 5.4 | **World Models Survey** | `surveys/robotics/25_AwesomeWorldModels/` | 世界模型分类学 | 1.5h |
| 5.5 | **Language Grounding** | `surveys/robotics/24_LanguageGrounding/` | 符号 vs 嵌入的 tradeoff | 1h |
| 5.6 | **Learned Dynamics Models** | `surveys/robotics/25_DynamicsModels/` | 状态表示对学习的影响 | 1h |
| 5.7 | **General-Purpose Robots** | `surveys/robotics/23_GeneralPurposeRobots/` | 五大挑战的 meta-analysis | 2h |
| 5.8 | **CV Surveys** (按需) | `surveys/CV/` | ViT(TPAMI), SSL(TPAMI), VLM(TPAMI), Depth, MIM(IJCV), NeRF+3DGS, Video | 各1h |

---

## 4. 已收录论文完整索引

### 必读 (直接影响 VLA 理解)

| 论文 | 年份 | 位置 |
|------|------|------|
| Transfer Learning Survey | 2010 | `foundations/10_TransferLearning/` |
| Representation Learning | 2013 | `foundations/12_RepresentationLearning/` |
| Transformer | 2017 | `foundations/17_Transformer/` |
| GPT-1/2/3/4 + RLHF + Codex | 2018-2023 | `LLM/families/GPT_Series/` |
| CLIP | 2021 | `CV/2_vl_alignment/21_CLIP/` |
| DDPM | 2020 | `CV/1_generation/20_DDPM/` |
| Flow Matching | 2022 | `CV/1_generation/22_FlowMatching/` |
| RT-1 / RT-2 | 2022-2023 | `robotics/policy_learning/` |
| Chinchilla | 2022 | `LLM/NLP_foundations/22_Chinchilla/` |
| PaliGemma | 2024 | `CV/2_vl_alignment/24_PaliGemma/` |

### 推荐 (加深理解)

| 论文 | 年份 | 位置 | 状态 |
|------|------|------|:---:|
| ResNet | 2015 | `CV/0_backbone/15_ResNet/` | 已有 |
| ViT | 2020 | `CV/0_backbone/20_ViT/` | 已有 |
| BERT | 2018 | `LLM/NLP_foundations/18_BERT/` | 已有 |
| DiT | 2023 | `CV/1_generation/23_DiT/` | 已有 |
| ACT | 2023 | `robotics/policy_learning/23_ACT/` | 已有 |
| OpenVLA | 2024 | `robotics/vla/24_OpenVLA/` | 已有 |
| SayCan | 2022 | `robotics/llm_planning/22_SayCan/` | 已有 |
| Code-as-Policies | 2022 | `robotics/llm_planning/22_CodeAsPolicies/` | 已有 |
| Inner Monologue | 2022 | `robotics/llm_planning/22_InnerMonologue/` | 已有 |
| LLaVA | 2023 | `CV/2_vl_alignment/23_LLaVA/` | 已有 |
| Voyager | 2023 | `robotics/llm_planning/23_Voyager/` | 已有 |
| MAE | 2021 | `CV/4_self_supervised/21_MAE/` | 已有 |
| DINOv2 | 2023 | `CV/4_self_supervised/23_DINOv2/` | 已有 |
| SAM | 2023 | `CV/5_detection_seg/23_SAM/` | 已有 |
| PaLME | 2023 | `robotics/vla/23_PaLME/` | 已有 |
| NeRF | 2020 | `CV/3_3d_vision/20_NeRF/` | 已有 |
| 3D Gaussian Splatting | 2023 | `CV/3_3d_vision/23_3DGS/` | 已有 |
| Depth Anything | 2024 | `CV/3_3d_vision/24_DepthAnything/` | 已有 |

---

## 5. 知名学者指引

| 学者 | 机构 | 核心贡献 | 与你的关联 |
|------|------|---------|---------|
| **Yoshua Bengio** | U. Montreal / CIFAR | 深度学习三巨头, 表示学习 | foundations/12_RepresentationLearning |
| **Ashish Vaswani** | Google (2017) | Transformer 原作者 | foundations/17_Transformer |
| **Alec Radford** | OpenAI | GPT-1/2, CLIP | LLM/families/GPT_Series + CV/21_CLIP |
| **Ilya Sutskever** | OpenAI/SSI | GPT-3 首席科学家 | Scaling 思想 |
| **Kaiming He** | Meta/MIT | ResNet, MAE | CV/0_backbone + CV/4_self_supervised |
| **Yann LeCun** | Meta | CNN 先驱, JEPA 世界模型 | 世界模型理论 |
| **Sergey Levine** | UC Berkeley / PI | RL for Robotics, pi_0 | robotics/generative_policy/24_pi0 |
| **Chelsea Finn** | Stanford / PI | Meta-learning, pi_0 | robotics/generative_policy/24_pi0 |
| **Pieter Abbeel** | UC Berkeley | RL, mjlab, Playground | RL 框架 |
| **Shuran Song** | Stanford (原 Columbia) | Diffusion Policy | robotics/generative_policy/24_DiffusionPolicy |
| **Cheng Chi** | Stanford (原 Columbia) | Diffusion Policy 一作 | robotics/generative_policy/24_DiffusionPolicy |
| **Dieter Fox** | NVIDIA/UW | GR00T N1 | robotics/generative_policy/25_GR00T_N1 |
| **Jim Fan** | NVIDIA | GR00T N1, Foundation Agent | VLA 工业化 |

---

## 6. 当前库的目录结构

```
foundation_model/
├── foundations/                   # 通用 ML 基础 (3 篇)
│   ├── 10_TransferLearning/
│   ├── 12_RepresentationLearning/
│   └── 17_Transformer/
├── LLM/                          # LLM 知识体系
│   ├── NLP_foundations/          # NLP 基础 (Word2Vec→BERT→Chinchilla)
│   ├── families/                 # 模型家族
│   │   ├── GPT_Series/          # GPT 全系列 + notes
│   │   ├── kimi/                # Kimi/Moonshot 全系列 + notes
│   │   ├── qwen/               # Qwen/Alibaba 全系列 + notes
│   │   ├── deepseek/           # DeepSeek 全系列 + notes
│   │   └── llama/              # Llama/Meta 全系列 + notes
│   └── LLM_技术交织与机器人启示.md
├── CV/                           # CV 知识体系 (按技术线)
│   ├── 0_backbone/              # ResNet, ViT, TransferFeatures
│   ├── 1_generation/            # VAE, DDPM, FlowMatching, DiT
│   ├── 2_vl_alignment/          # CLIP, LLaVA, PaliGemma
│   ├── 3_3d_vision/             # NeRF, 3DGS, DepthAnything
│   ├── 4_self_supervised/       # MAE, DINOv2
│   ├── 5_detection_seg/         # SAM
│   └── 6_video/                 # (待填)
├── robotics/                     # Robotics 应用 (按技术路线)
│   ├── llm_planning/            # SayCan, CodeAsPolicies, InnerMonologue, Voyager
│   ├── policy_learning/         # DT, RT-1, RT-2, ACT, OpenXEmbodiment
│   ├── vla/                     # PaLME, Octo, OpenVLA
│   ├── generative_policy/       # DiffusionPolicy, pi_0, GR00T N1
│   └── world_model/             # DreamerV3
├── surveys/
│   ├── CV/                      # 7 篇 CV surveys (TPAMI/IJCV/TCSVT)
│   └── robotics/               # 7 篇 Robotics surveys (IJRR + others)
├── note/                         # 学习笔记与考试
└── CS2Robotics_Roadmap.md        # 本文档
```

---

## 7. 一句话总结每篇的迁移意义

| 论文 | 迁移意义 |
|------|---------|
| Transfer Learning (2010) | pre-train + fine-tune 的理论根基, 所有 FM 的基本范式 |
| Bengio 表示学习 (2012) | 确立了"好的表示 = 好的 AI"信念, 预言了 foundation model 的成功 |
| Transformer (2017) | 去除循环依赖, attention 成为通用序列建模工具, 所有后续工作的根基 |
| GPT-1/2/3/4 (2018-2023) | 确立 "pre-train + fine-tune/prompt" 范式, 被机器人直接继承 |
| ViT (2020) | Transformer 从 NLP 迁移到 CV, patch embedding 成为所有 VLA 的视觉 backbone 基础 |
| DDPM (2020) | 扩散模型用于连续分布生成, 为 Diffusion Policy 和 flow matching 提供理论基础 |
| CLIP (2021) | 视觉-语言对齐, 使机器人能零样本理解语言指令 + 视觉场景 |
| MAE (2021) | 掩码自编码器, 自监督视觉预训练, 不需要标注数据就能学好的视觉表示 |
| Decision Transformer (2021) | 证明 Transformer 可处理 RL trajectory, 不需要 Bellman equation |
| Chinchilla (2022) | 修正 scaling law: 数据和模型应等比缩放, 影响所有大模型训练策略 |
| Flow Matching (2022) | 基于 ODE 的生成模型, 比 diffusion 更简洁高效, pi_0 的 action generation 核心 |
| SayCan (2022) | LLM task planning + CLIP affordance → robot execution, LLM-as-planner 路线开创者 |
| RT-1 (2022) | 第一个大规模 robot Transformer, 证明 130k 真实数据可以训练通用 robot policy |
| ACT (2023) | action chunking 概念的来源: CVAE 一次预测多步动作, 减少 compounding error |
| DiT (2023) | Transformer 替换 U-Net 做 diffusion backbone, GR00T N1 的 System 1 action head |
| DINOv2 (2023) | 自监督视觉基础模型, 不需要文本也能学通用视觉特征, robot 视觉 backbone 候选 |
| Open X-Embodiment (2023) | 22 个机器人, 527 种技能的跨机体数据集, 所有 VLA 的训练基础 |
| RT-2 (2023) | 第一个 VLA, VLM 直接输出离散动作 token, 继承互联网知识到机器人 |
| SAM (2023) | 通用图像分割, robot 场景理解的视觉基础 |
| DreamerV3 (2023) | 学习的世界模型可替代物理引擎做 planning, 150+ 任务单一算法 |
| Diffusion Policy (2023) | 图像 diffusion 直接用于机器人连续动作生成, 处理多模态动作分布 |
| Depth Anything (2024) | 单目深度估计基础模型, robot 3D 场景理解的关键工具 |
| Octo (2024) | 第一个开源通用 robot policy, diffusion action head + readout tokens, pi_0 的前身 |
| OpenVLA (2024) | 开源 VLA baseline, 7B 参数击败 55B RT-2-X, LoRA 高效微调 |
| PaliGemma (2024) | SigLIP + Gemma 2B 的 3B VLM, pi_0 选择的 backbone (小而能力足) |
| pi_0 (2024) | VLM + flow matching action expert = 第一个大规模灵巧操作 VLA (10k 小时数据) |
| GR00T N1 (2025) | 双系统 VLA: 慢思考 (VLM 10Hz) + 快执行 (DiT 120Hz), 面向人形机器人 |
| Robot Scaling Laws (2025) | 量化验证: 机器人性能也遵循 scaling law, 且比语言任务 scale 得更快 |
