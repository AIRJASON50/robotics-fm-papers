# paper/ - Robotics Foundation Model 知识库

> **用户背景**: RL 实践者 (PPO sim2real 灵巧手), 目标是理解 robotics FM 的设计思想
> **不复现代码**, 只要思想、方法和 takeaway
> **核心假设**: Robotics 正在重走 LLM/CV 的路

## 目录结构

```
paper/
├── foundation_model/              # CS → Robotics FM 知识库
│   ├── foundations/               #   通用 ML 基础 (9 篇)
│   │   ├── 10_TransferLearning    #     pre-train+fine-tune 的理论根基
│   │   ├── 12_RepresentationLearning  # 好表示 = 好 AI
│   │   ├── 14_GAN                 #     生成对抗网络
│   │   ├── 15_Adam                #     默认优化器
│   │   ├── 15_BatchNorm           #     CNN 训练稳定性
│   │   ├── 16_LayerNorm           #     Transformer 标配
│   │   ├── 17_PPO                 #     RLHF + robot RL 的核心算法
│   │   ├── 17_Transformer         #     一切的根基
│   │   └── 18_SAC                 #     连续控制 RL 标准
│   ├── LLM/                       #   LLM 知识体系
│   │   ├── NLP_foundations/       #     Word2Vec→Seq2Seq→Attention→BERT→Chinchilla
│   │   ├── families/             #     GPT, Kimi, Qwen, DeepSeek, Llama
│   │   └── LLM_技术交织与机器人启示.md
│   ├── CV/                        #   CV 知识体系 (24 篇, 7 个方向)
│   │   ├── 0_backbone/           #     ResNet, ViT, Swin, TransferFeatures
│   │   ├── 1_generation/         #     VAE, DDPM, LDM, FlowMatching, DiT
│   │   ├── 2_vl_alignment/       #     CLIP, BLIP-2, LLaVA, PaliGemma
│   │   ├── 3_3d_vision/          #     NeRF, 3DGS, DepthAnything
│   │   ├── 4_self_supervised/    #     MoCo, SimCLR, DINO, BEiT, MAE, DINOv2
│   │   ├── 5_detection_seg/      #     DETR, SAM
│   │   ├── 6_video/              #     TimeSformer, ViViT, VideoMAE, Ego4D
│   │   └── CV_技术演进与机器人启示.md
│   ├── robotics/                  #   Robotics 应用
│   │   ├── families/             #     Google_RT_Series, pi_Series, GR00T_Series
│   │   ├── policy_learning/      #     DT, ACT, DiffusionPolicy, DROID
│   │   ├── vla/                  #     Octo, OpenVLA
│   │   ├── visual_repr/          #     R3M, VIP
│   │   └── world_model/          #     DreamerV3, UniSim
│   ├── surveys/                   #   CV/ (7) + robotics/ (7)
│   ├── CS2Robotics_Roadmap.md     #   主学习路线 (Level 0-4, 按问题分级)
│   └── note/                      #   学习笔记 + 升级考试
├── humanoid/                      # 人形机器人 (12 项目)
├── manip/                         # 灵巧手操作 (16+ 项目)
│   └── dataset/                   #   手部数据集库 (hand_object/robot_hand/hand_only)
├── html2aitext_convert/           # arxiv → markdown 工具 (DO NOT modify)
├── papers.yaml                    # 全库清单
├── scripts/setup.sh               # 部署脚本 (默认拉论文, --repos 拉代码仓库)
├── README.md                      # 项目说明
└── CLAUDE.md                      # 本文件
```

## 论文获取流程

### 1. 拉取 arxiv 论文

```bash
bash html2aitext_convert/arxiv2md.sh <arxiv_id>
# 输出: html2aitext_convert/output/<title>.md
# 如果 404 (老论文无 HTML): curl -sL -o <name>.pdf https://arxiv.org/pdf/<id>
```

### 2. 放置位置决策

| 内容 | 放哪里 |
|------|--------|
| ML 通用基础 (Transformer, PPO, Adam...) | `foundations/` |
| LLM 模型家族 | `LLM/families/<name>/` |
| NLP 专属基础 | `LLM/NLP_foundations/` |
| CV 论文 (按技术方向) | `CV/<方向>/` |
| Robotics 家族项目 (RT, PI, GR00T) | `robotics/families/<name>/` |
| Robotics 单篇方法 | `robotics/policy_learning/` 或 `vla/` 或 `world_model/` |
| 综述 | `surveys/CV/` 或 `surveys/robotics/` |

### 3. 命名规范

- 文件夹: `<YY>_<ShortName>` (如 `25_SONIC`, `24_pi0`)
- 家族: `<Name>_Series` (如 `GR00T_Series`, `pi_Series`)
- 笔记: `<name>_notes.md` 或 `<name>_family_notes.md`

### 4. 笔记写作标准

**单篇论文笔记** (`*_notes.md`):

8 个标准 section:
1. Core Problem
2. Method Overview
3. Key Designs (2-3 个最重要贡献)
4. Experiments
5. Related Work Analysis
6. Limitations & Future Directions
7. Paper vs Code Discrepancies
8. Cross-Paper Comparison

**家族笔记** (`*_family_notes.md`):

以 takeaway 为核心, 参考 DeepSeek notes 格式:
1. 战略定位 / 背景
2. 演进脉络 (每个阶段: 问题→洞察→解法→takeaway)
3. 核心 takeaway 表 (# / Takeaway / 原理 / 对你的行动项)
4. 与其他家族的交叉引用
5. 文件索引

**格式规则**:
- 中文写作, 技术术语保留英文
- 不用 emoji
- 首次缩写注解: `SFT (Supervised Fine-Tuning, 监督微调)`
- markdown 表格做结构化对比

## Paper Index

### foundations/ (5 个第一原理目录 + RL 子目录)

| Folder | Paper | Year |
|--------|-------|------|
| NeuralNetwork_Origins | **Backpropagation+MLP (Rumelhart+Hinton), Universal Approximation (Cybenko), MoE (Jacobs+Hinton)** | 1986-1991 |
| RepresentationLearning | Representation Learning (Bengio, IEEE TPAMI) | 2013 |
| TransferLearning_Origins | **Thrun (终身学习), Caruana (多任务共享表征), Yosinski (特征可迁移性), Pan&Yang (分类体系)** | 1996-2014 |
| 17_Transformer | Attention Is All You Need (Vaswani, NeurIPS) | 2017 |
| RL/ | Bellman→Q-Learning→REINFORCE→DQN→PPO→SAC→DR→Dactyl→RMA (9 篇) | 1952-2021 |

### LLM/ (5 NLP 基础 + 5 模型家族)

| 内容 | Notes |
|------|-------|
| NLP_foundations: Word2Vec, Seq2Seq, Attention, BERT, Chinchilla | -- |
| GPT Series: GPT-1/2/3/4 + RLHF + Codex + WebGPT + InstructGPT | GPT_series_notes.md |
| Kimi: k1.5, MoBA, Moonlight, Audio, K2, K2.5 | kimi_series_notes.md |
| Qwen: 1/2/2.5/3/3.5, VL, Audio, Omni | qwen_series_notes.md |
| DeepSeek: MoE, V2 (MLA), V3, R1, V3.2 (DSA), V4 (CSA+HCA, mHC, Muon, FP4) | deepseek_series_notes.md |
| Llama: 1/2/3/4 | llama_series_notes.md |

### CV/ (24 篇, 7 方向)

| 方向 | 内容 |
|------|------|
| 0_backbone | ResNet, ViT, SwinTransformer, TransferFeatures |
| 1_generation | VAE, DDPM, LDM, FlowMatching, DiT |
| 2_vl_alignment | CLIP, BLIP-2, LLaVA, PaliGemma |
| 3_3d_vision | NeRF, 3DGS, DepthAnything |
| 4_self_supervised | MoCo, SimCLR, DINO, BEiT, MAE, DINOv2 |
| 5_detection_seg | DETR, SAM |
| 6_video | TimeSformer, ViViT, VideoMAE, Ego4D |

### robotics/ (3 家族 + 方法论文)

| 内容 | 位置 |
|------|------|
| **Google RT Series** (3 篇): RT-1→RT-2→OpenXEmbodiment | families/Google_RT_Series/ |
| **PI Series** (4 核心 + 4 支撑): pi_0→pi_0.5→pi\*0.6→pi_0.7 (DROID/FAST/HiRobot/MEM 蒸馏至 family notes) | families/pi_Series/ |
| **GR00T Series** (7 篇): N1→N1.5→SONIC→DreamGen→N1.6→DreamZero→N1.7 | families/GR00T_Series/ |
| 方法论文: ACT, DiffusionPolicy, SpatialForcing | policy_learning/ |
| 世界模型: DreamerV3, UniSim | world_model/ |

### surveys/ (14 篇)

| 方向 | 内容 |
|------|------|
| CV (7) | ViT(TPAMI), SSL(TPAMI), VLM(TPAMI), Depth, MIM(IJCV), Video(TCSVT), NeRF+3DGS |
| Robotics (7) | FMRobotics(IJRR), GeneralPurpose, LangCondManip, LanguageGrounding, WorldModels, DynamicsModels, RobotScalingLaws |

### humanoid/ (13 项目, 5 主题)

详见 `humanoid/humanoid_landscape.md`

| 主题 | 内容 |
|------|------|
| motion_tracking/ | DeepMimic → PHC → BeyondMimic → SONIC (→GR00T) |
| teleoperation/ | H2O, FPO, OmniRetarget, TWIST2 |
| retargeting/ | GMR (非均匀局部缩放 + 两阶段 diff-IK, 17+ 机器人) |
| sim2real/ | ASAP |
| video_world_model/ | HDMI, RWM |

### manip/ (13+ 项目, 5 主题)

详见 `manip/manip_landscape.md` (含 humanoid→manip 桥接分析)

| 主题 | 内容 |
|------|------|
| traditional_rl/ | ArtiGrasp, PhysHOI, ObjDexEnvs |
| human2robot/ | BiDexHD, DexMachina, DexTrack, HumDex |
| scaling_rl/ | OmniReset |
| sim2real/ | SimToolReal, Dex4D |
| fm_manip/ | RLToken (PI), DexGraspVLA, DexLatent (XL-VLA), UltraDexGrasp, UniDex, PAM |
| QiHaoZhi/ | 研究者 family (齐浩之) |
| dataset/ | 手部数据集库 (hand_object / robot_hand / hand_only) |
