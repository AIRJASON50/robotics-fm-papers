# A Comprehensive Survey on World Models for Embodied AI -- GitHub Survey 资源库分析

**类型**: GitHub awesome-list + arXiv survey 论文配套
**论文**: Li et al., "A Comprehensive Survey on World Models for Embodied AI", arXiv:2510.16732
**作者**: Nankai University, Tianjin University of Technology, UESTC, A*STAR, SUTD
**GitHub**: https://github.com/Li-Zn-H/AwesomeWorldModels

---

## 1. Core Problem

### World Model 在 Embodied AI 中的核心角色

World model 是环境动力学的内部模拟器, 使 agent 能够在不与真实环境交互的情况下进行 forward rollout 和 counterfactual reasoning。本 survey 针对的核心困境:

| 困境 | 具体表现 |
|------|---------|
| **术语混乱** | "World model" 在不同子社区 (robotics / autonomous driving / video generation) 含义不一致 |
| **分类缺失** | 现有 survey 按功能 (understanding vs prediction) 或按应用 (自动驾驶) 分类, 缺乏统一的多维度 taxonomy |
| **评估碎片化** | pixel fidelity vs physical consistency vs task performance, 不同社区用不同 metric |
| **架构多样** | RSSM, Transformer, Diffusion, 3DGS, NeRF 等多种架构并存, 缺乏系统比较 |

### 与 Awesome-Robotics-FM (24_AwesomeSurvey) 的互补

24_AwesomeSurvey 覆盖 FM 在 robotics 全栈的应用, 其中 Predictive Models 部分仅列 7 篇。本 repo 专注 world model, 覆盖 200+ 篇论文, 是 Predictive Models 方向的大幅扩展。

---

## 2. Method Overview: 三轴分类框架

### 2.1 核心创新: 三维 Taxonomy

本 survey 提出了 world model 领域第一个系统化的三轴分类框架:

```
Axis 1: Functionality (功能)
    +-- Decision-Coupled: 针对特定 decision task 优化的 world model
    +-- General-Purpose: 通用环境模拟器, 可泛化到多种下游任务

Axis 2: Temporal Modeling (时间建模)
    +-- Sequential Simulation and Inference: 逐步自回归展开 (如 RSSM, autoregressive Transformer)
    +-- Global Difference Prediction: 直接并行预测完整未来状态 (如 diffusion model)

Axis 3: Spatial Representation (空间表征)
    +-- Global Latent Vector: 紧凑向量表征 (RSSM latent state)
    +-- Token Feature Sequence: token 序列 (VQ-VAE token, LLM token)
    +-- Spatial Latent Grid: BEV feature / voxel grid (3D occupancy)
    +-- Decomposed Rendering: 3DGS / NeRF 等可微渲染 (显式 3D)
```

### 2.2 分类矩阵

三个轴组合形成 2 x 2 x 4 = 16 个理论类别, 实际 repo 覆盖了 12 个:

| | Sequential + Latent Vector | Sequential + Token | Sequential + Grid | Sequential + Rendering | Global + Token | Global + Grid | Global + Rendering |
|---|---|---|---|---|---|---|---|
| **Decision-Coupled** | Dreamer 系列, PlaNet, DayDreamer | IRIS, TWM, STORM, MineDreamer | OccWorld, DriveDreamer | ManiGaussian, DreMa | (少量) | OccWorld 变体 | -- |
| **General-Purpose** | -- | Sora, Cosmos, GAIA-1 | OccLLaMA, UniWorld | InfiniCube | Video generation 类 | BEV 预测类 | GS-based rendering |

**关键观察**:
- **Decision-Coupled + Sequential + Latent Vector**: 最成熟的类别 (Dreamer 系列, 2019-2025), 覆盖最全
- **General-Purpose + Sequential + Token**: 增长最快的类别 (Sora/Cosmos 驱动), 2024-2025 爆发
- **Decomposed Rendering**: 最新兴的类别, 3DGS 用于 world model 始于 2024

### 2.3 数学形式化

所有 world model 统一建模为 POMDP:

```
Dynamics Prior:     p_theta(z_t | z_{t-1}, a_{t-1})
Filtered Posterior: q_phi(z_t | z_{t-1}, a_{t-1}, o_t)
Reconstruction:     p_theta(o_t | z_t)
```

训练目标: ELBO = reconstruction loss + KL regularization。这个形式统一了 Dreamer (RSSM), IRIS (Transformer), ManiGaussian (3DGS) 等看似不同的架构。

---

## 3. Key Designs: 三条技术路线

### 3.1 Dreamer 系列: Decision-Coupled 的基石

RSSM (Recurrent State-Space Model) 是 Decision-Coupled world model 的核心架构, 从 PlaNet 到 DreamerV3 形成了清晰的技术演进:

```
PlaNet (2019): RSSM + CEM planning --> 基础范式
    |
DreamerV1 (2020): + actor-critic in imagination --> 从 planning 到 policy learning
    |
DreamerV2 (2021): + discrete latent + KL balancing --> Atari SOTA
    |
DreamerV3 (2023/2025): + symlog + fixed hyperparams --> 150+ tasks, Nature 发表
```

后续工作在此基础上做各种改进:
- **去解码器**: DreamerPro (prototype), Dreaming (contrastive), HRSSM (dual-branch)
- **跨域迁移**: PreLAR (learnable action), DisWM (disentangled semantic), ReDRAW (residual dynamics)
- **安全性**: VL-SAFE (VLM safety score), SR-AIF (self-revision)

### 3.2 Token-based World Model: LLM 范式的迁移

将 world state 离散化为 token, 复用 LLM/VQ-VAE 架构:

| 方法 | Tokenizer | 模型 | 优势 | 局限 |
|------|----------|------|------|------|
| IRIS | VQ-VAE | Transformer | 100k 交互即可学会 | 仅 Atari |
| STORM | VQ-VAE | Transformer | 更高效的 token 预测 | -- |
| MineWorld | 视觉 token | Autoregressive | 实时交互式 world model | Minecraft 专用 |
| WorldVLA | Action token | Autoregressive VLA | 统一 action 和 world prediction | 尚为预印本 |
| Cosmos (NVIDIA) | 多模态 token | Foundation model | 通用环境模拟 | 非开源核心 |

**关键趋势**: Token-based world model 正与 VLA (Vision-Language-Action) model 融合。WorldVLA 将 action prediction 和 world prediction 统一在同一个 autoregressive 框架中。

### 3.3 3DGS/NeRF-based World Model: 显式 3D 理解

2024 年兴起的新方向, 用可微渲染做 world model:

- **ManiGaussian** (ECCV'24): 3DGS 做 multi-task manipulation world model, 能预测物体变形和移动
- **DreMa** (ICLR'25): 组合式 world model, 用 imagination 增强 imitation learning
- **PIN-WM** (RSS'25): Physics-informed world model, 融合物理先验和 3DGS
- **GAF** (2025): 4D Gaussian Action Field, 将动作嵌入高斯表征

**与本库的关联**: ManiGaussian 系列与 `manip/` 目录下的灵巧操作工作 (BiDexHD, ManipTrans) 解决类似问题, 但从 world model 而非 RL policy 角度出发。

---

## 4. Experiments: 覆盖统计

### 4.1 论文规模与分布

README_clean.md 共 1177 行, 涵盖 200+ 篇论文。

按年份分布:

| 年份 | 论文数 (约) | 标志性工作 |
|------|-----------|-----------|
| 2018-2020 | ~10 | World Models (Ha), PlaNet, DreamerV1 |
| 2021-2022 | ~20 | DreamerV2, DayDreamer, TransDreamer |
| 2023 | ~25 | DreamerV3, IRIS, TWM, Sora(预告) |
| 2024 | ~60 | ManiGaussian, OccWorld, Cosmos, GenAD |
| 2025 | ~90+ | WorldVLA, DreMa, PIN-WM, 大量新工作 |

**2025 年占比接近一半**, 反映 world model 是当前 AI 最活跃的方向之一。

### 4.2 按应用领域分布

| 领域 | 标注 | 论文占比 (约) |
|------|------|-------------|
| Robotics (Manipulation + Locomotion) | :robot: | ~50% |
| Autonomous Driving | :car: | ~30% |
| Navigation | :compass: | ~10% |
| Video Generation (通用) | :clapper: | ~10% |

### 4.3 按三轴分类分布

| 功能维度 | 论文占比 |
|---------|---------|
| Decision-Coupled | ~55% |
| General-Purpose | ~45% |

| 时间建模 | 论文占比 |
|---------|---------|
| Sequential | ~70% |
| Global | ~30% |

| 空间表征 | 论文占比 |
|---------|---------|
| Latent Vector | ~25% |
| Token Sequence | ~35% |
| Spatial Grid | ~25% |
| Decomposed Rendering | ~15% |

---

## 5. Related Work Analysis

### 5.1 Survey 论文的定位

| 维度 | 本 survey | Ding et al. (2024) | Guan et al. (2024) | Zhu et al. (Sora, 2024) |
|------|---------|-------------------|-------------------|----------------------|
| 范围 | Embodied AI 全域 | Understanding + prediction | 仅自动驾驶 | Video generation 为主 |
| 分类方法 | 三轴 (功能 x 时间 x 空间) | 双功能 (理解 vs 预测) | 应用导向 | 能力导向 |
| 覆盖深度 | 200+ 论文, 定量比较 | ~100 论文 | ~50 论文 | ~30 论文 |
| 独特贡献 | 统一 taxonomy + 跨域覆盖 | 首个系统性 world model survey | 自动驾驶专深 | Sora 视角下的 world model |

### 5.2 与本库 DreamerV3 笔记的关系

DreamerV3 (`23_DreamerV3/DreamerV3_notes.md`) 是本 repo Decision-Coupled / Sequential / Latent Vector 类别中最核心的工作。DreamerV3 笔记深入分析了 RSSM 架构、symlog/symexp、KL balancing 等细节, 是理解本 repo 中 ~25% 论文 (所有 RSSM 变体) 的基础。

---

## 6. Limitations & Future Directions

### 6.1 Survey 论文提出的开放挑战

| 挑战 | 描述 | 与本库工作的关联 |
|------|------|----------------|
| **统一数据集缺乏** | 不同域 (robotics, driving, video) 用不同数据集, 无法跨域比较 | GR00T N1 的 data pyramid 是 robotics 侧的尝试 |
| **评估标准不统一** | Pixel fidelity (FVD/SSIM) vs physical consistency vs task performance | DreamerV3 用 task performance, video model 用 FVD, 需要统一 |
| **长时域一致性** | Autoregressive rollout 的 error accumulation | DreamerV3 的 KL balancing 和 symlog 部分缓解 |
| **实时性** | General-purpose world model (Sora-like) 无法 real-time | Decision-Coupled model (Dreamer) 可以, 但泛化差 |
| **Physical fidelity vs pixel fidelity** | Video model 像素完美但物理不一致 | Contact-rich manipulation 需要物理精度, 非像素精度 |

### 6.2 Repo 的覆盖盲区

| 盲区 | 缺失内容 |
|------|---------|
| **Humanoid locomotion world model** | DWL 和 WMR 有收录, 但缺少与 RL policy 的结合分析 |
| **Dexterous manipulation world model** | DexSim2Real^2 有收录, 但该领域仍极度稀缺 |
| **Multi-agent world model** | 几乎未涉及 |
| **Tactile/force prediction** | 完全缺失 |
| **Foundation world model + VLA 融合** | WorldVLA 是唯一一篇, 这将成为主要趋势 |

### 6.3 未来重要方向

1. **World Model + VLA 融合**: 将 world model 预测能力嵌入 VLA 的 action generation, 使 policy 具备 "想象力"
2. **Physics-informed World Model**: 结合物理先验 (如 PIN-WM), 解决纯 data-driven model 的物理不一致问题
3. **Cross-Embodiment World Model**: 类似 GR00T N1 的 embodiment-specific projector, 但用于 world model
4. **Real-Time Foundation World Model**: 缩小 General-Purpose model (秒级) 和 Decision-Coupled model (ms 级) 之间的差距

---

## 7. Paper vs Code Discrepancies

### Repo vs Survey 论文的差异

| 维度 | Survey 论文 (arXiv) | GitHub Repo |
|------|-------------------|-------------|
| **分类呈现** | 三轴 taxonomy + 详细分析 + 定量比较表 | 仅按三轴组合分 12 个 section, 无分析 |
| **数学框架** | 统一 POMDP 形式化, ELBO 推导 | 无 |
| **评估对比** | Table I/II 定量比较代表性方法 | 无性能对比 |
| **更新状态** | 固定版本 (arXiv 2510.16732) | 持续更新, 已包含 2025 年 ICCV/ICML/RSS 新工作 |

### Repo 较论文新增的重要工作

Repo 持续接收 PR, 已包含论文截止后的工作:
- **WorldVLA** (2025): Autoregressive Action World Model, 统一 VLA 和 world model
- **PIN-WM** (RSS'25): Physics-informed, 融合物理先验
- **ManiGaussian++** (IROS'25): Bimanual manipulation with 3DGS world model
- **EnerVerse** (2025): Embodied future space 可视化
- **FOUNDER** (ICML'25): Foundation model grounding in world model state space

---

## 8. Cross-Paper Comparison

### 8.1 与本库其他工作的关联图

```
本 Repo (AwesomeWorldModels)
    |
    +-- DreamerV3_notes.md
    |   (Decision-Coupled / Sequential / Latent Vector 的核心代表)
    |   (本 repo 中 ~25% 论文是 Dreamer 系列变体)
    |
    +-- 24_AwesomeSurvey (Awesome-Robotics-FM)
    |   (本 repo 的 Predictive Models 部分 = AwesomeSurvey 中 7 篇论文的 30x 扩展)
    |
    +-- GR00T_N1_notes.md
    |   (GR00T N1 的 DiT 可视为 General-Purpose / Sequential / Token 类型)
    |   (NVIDIA Cosmos world model 在本 repo 中有收录)
    |
    +-- pi0_notes.md
    |   (pi_0 的 flow matching 可对应 Global Difference Prediction 范式)
    |   (但 pi_0 本身不是 world model, 而是 action generation model)
    |
    +-- DiffusionPolicy_notes.md
        (Diffusion Policy 在本 repo 的 taxonomy 中归属模糊:
         既非典型 world model, 也非 Sequential/Global 明确归类)
```

### 8.2 World Model vs Policy Model 的边界

本 repo 的分类框架凸显了一个重要问题: **world model 和 policy model 的边界在模糊化**。

| 模型类型 | 代表工作 | 预测什么 | 在本 repo 中 |
|---------|---------|---------|-------------|
| 纯 World Model | DreamerV3, GAIA-1 | 下一状态 z_{t+1} | 核心收录 |
| World Model + Policy | Dreamer (imagination + actor-critic) | 状态 + 动作 | 核心收录 |
| Policy with World Knowledge | GR00T N1 (DiT + VLM) | 动作 (隐含状态理解) | 未收录 |
| Diffusion Policy | DP, pi_0 | 动作序列 (action chunk) | 未收录 |
| Action World Model | WorldVLA | 状态 + 动作 (统一) | 收录 (新增) |

**趋势**: WorldVLA 代表了 world model 和 policy model 融合的方向 -- 一个模型同时预测环境变化和应该采取的动作。这与 GR00T N1 的 "System 1 + System 2" 架构殊途同归。

### 8.3 阅读建议

| 目标 | 推荐路径 |
|------|---------|
| 理解 world model 基础理论 | Survey 论文 Section II (POMDP + ELBO) --> DreamerV3_notes.md (具体实现) |
| 了解 RSSM 系列全貌 | 本 repo "Decision-Coupled / Sequential / Latent Vector" section --> DreamerV3_notes.md |
| 探索 3DGS world model | 本 repo "Decomposed Rendering" sections --> ManiGaussian 系列论文 |
| 理解 world model + VLA 融合趋势 | WorldVLA (本 repo) + GR00T_N1_notes.md + pi0_notes.md |
| 对比 world model vs foundation model for robotics | 本 repo vs 24_AwesomeSurvey vs FMRobotics_notes.md |
