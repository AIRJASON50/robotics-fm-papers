# Awesome-Robotics-Foundation-Models -- GitHub Survey 资源库分析

**类型**: GitHub awesome-list / survey 配套资源库
**关联论文**: Firoozi et al., "Foundation Models in Robotics: Applications, Challenges, and the Future", IJRR 2024
**论文笔记**: 详见 `../23_FMRobotics/FMRobotics_notes.md` (本笔记不重复论文内容分析)
**GitHub**: https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models
**License**: Apache-2.0

---

## 1. Core Problem

### 这个资源库要解决什么问题

Foundation model 在 robotics 中的应用论文散布在 arxiv、各大会议中, 缺乏按功能维度组织的统一索引。研究者需要一个 structured reference, 快速定位特定子方向的代表性工作。

本 repo 是 Firoozi et al. survey 的配套资源库, 按论文的 Figure 1 分类框架组织文献。与纯 paper list 不同, 它提供了:

- 按 autonomy stack 层级的分类组织 (不是按发表时间/会议)
- 每篇论文附 Paper / Project / Code 链接
- 与正式 survey 论文的一一对应关系

### 定位: 资源索引而非分析文献

| 维度 | 本 repo | 关联论文 (FMRobotics) | GeneralPurposeRobots survey |
|------|---------|---------------------|---------------------------|
| 形式 | GitHub paper list | IJRR 正式 survey | arXiv survey + meta-analysis |
| 分析深度 | 仅列名 + 链接 | 深入讨论挑战与机遇 | 定量 meta-analysis |
| 更新频率 | 接受 PR, 持续更新 | 固定版本 | Living document (v2.1) |
| 主要用途 | 查文献找代码 | 理解领域全景与趋势 | 定量比较与分析 |

---

## 2. Method Overview: 分类框架

### 2.1 整体组织结构

Repo 的分类体系直接映射自论文 Figure 1 (survey_tree.png), 分为三大板块:

```
Foundation Models in Robotics
|
+-- Robotics (直接应用于机器人)                      ~62 papers
|   +-- Neural Scaling Laws                         (1)
|   +-- Robot Policy Learning                       (8)
|   |   +-- Language-Conditioned Imitation Learning  (4)
|   |   +-- Language-Assisted RL                     (4)
|   +-- Language-Image Goal-Conditioned Value Learning (7)
|   +-- Robot Task Planning Using LLMs              (10)
|   +-- LLM-Based Code Generation                   (6)
|   +-- Robot Transformers                          (12)
|   +-- In-context Learning for Decision-Making     (5)
|   +-- Open-Vocabulary Navigation & Manipulation   (13)
|
+-- Relevant to Robotics: Perception                ~33 papers
|   +-- Open-Vocabulary Object Detection & 3D       (7)
|   +-- Open-Vocabulary Semantic Segmentation       (6)
|   +-- Open-Vocabulary 3D Scene Representations    (4)
|   +-- Object Representations                      (7)
|   +-- Affordance Information                      (2)
|   +-- Predictive Models                           (7)
|
+-- Relevant to Robotics: Embodied AI               ~24 papers
    +-- Embodied Agents                             (12)
    +-- Generalist AI                               (4)
    +-- Simulators                                  (8)
```

**总计约 119 篇论文**, 覆盖 2020-2024 年的代表性工作。

### 2.2 分类维度分析

Repo 的组织维度是 **功能角色** (FM 在 robot system 中承担什么角色), 而非:
- 模型架构 (Transformer vs Diffusion vs SSM)
- 训练范式 (supervised vs RL vs self-supervised)
- 任务类型 (manipulation vs navigation vs locomotion)

这意味着同一篇论文可能出现在多个类别中 (如 NLMap 同时出现在 Task Planning 和 3D Scene 中; MineDreamer 出现在 Value Learning、Predictive Models 和 Embodied AI 三处)。

---

## 3. Key Designs: 分类框架的关键洞察

### 3.1 Robotics 板块: 从 perception 到 action 的完整链路

Repo 将 FM 在 robotics 中的直接应用组织为一条从高层到底层的 pipeline:

```
Task Specification (自然语言 / 图像目标)
    |
    v
Task Planning (LLM as Planner / Code Generation)    <-- 高层, plug-and-play
    |
    v
Value Learning (CLIP/VLM 提供 reward signal)         <-- 中层, fine-tune
    |
    v
Robot Policy (Robot Transformer / IL / RL)           <-- 底层, train from scratch
    |
    v
Open-Vocabulary Execution (场景理解 + manipulation)   <-- 跨层
```

这条链路暗含一个关键 trade-off (在 FMRobotics_notes.md 中有详细分析):
- **越靠近 task specification**: FM 越容易 plug-and-play (GPT-4 直接做 planning)
- **越靠近 low-level control**: FM 越需要 domain-specific 训练 (RT-1/2 需要海量 robot data)

### 3.2 Perception 板块: 2D -> 3D 的语义提升

Perception 部分的组织体现了 open-vocabulary 感知的分层架构:

| 层级 | 类别 | 代表工作 | 输出 |
|------|------|---------|------|
| 2D 检测 | Object Detection | OWL-ViT, Grounding DINO | 2D bounding box |
| 2D 分割 | Semantic Segmentation | SAM, LSeg, FastSAM | pixel mask |
| 3D 分类 | 3D Classification | PointCLIP, ULIP | 点云类别 |
| 3D 场景 | 3D Scene Representations | LERF, CLIP-Fields | 语义 3D field |
| 物体表征 | Object Representations | FoundationPose, NDF | 6-DoF pose / feature |
| 功能理解 | Affordance | Affordance Diffusion, VRB | 交互区域 / 方式 |

这条路线从 2D pixel-level 理解逐步扩展到 3D semantic field, 最终到 affordance (物体功能性理解)。每一层都依赖 CLIP/DINO 等 VFM 的 open-vocabulary 能力。

### 3.3 Embodied AI 板块: Minecraft 作为测试场

Repo 中 Embodied AI 部分有一个有趣的特征: **大量工作以 Minecraft 为测试环境** (MineDojo, VPT, Voyager, JARVIS-1, GROOT, MP5, MC-Planner, MineDreamer)。

Minecraft 之所以成为 embodied AI 的标准测试场:
- 开放世界, 无预定义任务
- 视觉丰富但物理简化
- 已有大量人类 gameplay 数据 (YouTube)
- 支持 mod 扩展, 易于定义新任务

但 Minecraft 的物理引擎远不如 MuJoCo/IsaacSim, 导致这些工作对 real-world contact-rich manipulation 的参考价值有限。

---

## 4. Experiments: 覆盖度统计分析

### 4.1 论文分布

按年份统计 (基于 README 中的论文):

| 年份 | 论文数 (约) | 代表性工作 |
|------|-----------|-----------|
| 2019-2021 | ~15 | Gibson, Habitat, Play-LMP, VPT |
| 2022 | ~25 | CLIPort, SayCan, Code-as-Policies, CLIP-Fields, SAM |
| 2023 | ~55 | RT-1/2, Voyager, JARVIS-1, Grounding DINO, PerAct |
| 2024 | ~20 | MineDreamer, ICRT, MAGIC-VFM, FoundationPose |

**2023 年是爆发期**, 占总量近一半, 与 GPT-4/ChatGPT 引爆 LLM 热潮的时间吻合。

### 4.2 按功能角色分布

| 功能角色 | 论文占比 | 分析 |
|---------|---------|------|
| Robotics (直接控制/规划) | 52% | 核心板块, 覆盖最全 |
| Perception | 28% | 以 open-vocabulary 方法为主 |
| Embodied AI + Simulators | 20% | Minecraft 生态占主导 |

### 4.3 代码可获取性

| 状态 | 比例 (估算) |
|------|-----------|
| 有 Code 链接 | ~55% |
| 有 Project Page | ~60% |
| 仅有 Paper 链接 | ~25% |
| Paper + Project + Code 齐全 | ~40% |

**观察**: Robot Transformer 和 Embodied AI 板块的代码开放率最高, 而 Task Planning 板块较多仅有论文。

---

## 5. Related Work Analysis: 与同类资源库的对比

### 5.1 与本库其他 survey repo 的对比

| 维度 | 本 repo (Awesome-Robotics-FM) | 25_AwesomeWorldModels | GeneralPurposeRobots (23) |
|------|------------------------------|----------------------|--------------------------|
| 主题 | FM 在 robotics 全栈的应用 | Embodied AI 的 world model | 通用机器人的 FM 应用 |
| 组织方式 | 按功能角色 (planning / perception / control) | 按模型架构 (latent vector / video / state-space) | 按任务类型 + meta-analysis |
| 论文数量 | ~119 | ~200+ (持续更新, 含 2025 新工作) | ~300+ (living document) |
| 覆盖时间 | 2020-2024 | 2015-2025 | 2020-2024 |
| 交叉关系 | 本 repo 的 Predictive Models 是 AwesomeWorldModels 的子集 | 更专注, 不覆盖 planning/perception | 更广泛, 含定量分析 |

### 5.2 与本库已有论文的交叉引用

Repo 中收录的论文, 与本 paper library 中已有详细笔记的工作的交叉:

| 本库论文 | 在 repo 中的位置 | 关系 |
|---------|-----------------|------|
| **DiffusionPolicy** (24) | 未收录 | Repo 截止时间早于 DP 发表; DP 属于 Robot Policy Learning 但用 diffusion 而非 transformer |
| **DecisionTransformer** (21) | 未收录 | DT 是 offline RL, 与 repo 的 Robot Transformer 类别相关但未列入 |
| **DreamerV3** (23) | 未收录 | DreamerV3 是 task-specific world model, 非 foundation model, 故未列入 |
| **GR00T N1** (25) | 未收录 | 发表于 repo 活跃更新期之后 |
| **pi_0** (24) | 未收录 | 发表晚于 repo 主要更新期 |

**关键发现**: 本 paper library 中的技术论文大多未被该 repo 收录, 原因主要是:
1. 时间差: pi_0, GR00T N1 发表较晚
2. 范畴差: DreamerV3, DecisionTransformer 不属于 "foundation model" 范畴 (task-specific training)
3. 方法差: DiffusionPolicy 用 diffusion model 做 action generation, repo 主要覆盖 LLM/VLM-based 方法

---

## 6. Limitations & Future Directions

### 6.1 Repo 的覆盖盲区

| 盲区 | 缺失内容 | 影响 |
|------|---------|------|
| **Dexterous Manipulation** | 无灵巧操作专区, 所有 manipulation 均为 gripper-based | 遗漏 DexMimicGen, ManipTrans, BiDexHD 等重要工作 |
| **Locomotion** | 仅在 Embodied AI 中零散提及, 无专门分类 | 遗漏 humanoid control 全领域 (PHC, SONIC, ASAP 等) |
| **Diffusion-based Policy** | Predictive Models 中有 Diffuser, 但未将 diffusion policy 作为独立类别 | 遗漏 Diffusion Policy, DDPO, 3D Diffuser Actor 等 |
| **VLA (Vision-Language-Action) Model** | Robot Transformers 中有 RT-2, 但未建立 VLA 专区 | 遗漏 pi_0, GR00T N1, OpenVLA, Octo 等 2024-2025 的核心工作 |
| **Scaling Laws** | 仅 1 条引用 | 未覆盖 robotics scaling 的系统性研究 |
| **Sim-to-Real Transfer** | 无专门分类 | 作为 robotics FM 的核心瓶颈却未被组织 |
| **Tactile / Force Sensing** | 完全缺失 | 触觉 foundation model 是新兴方向 |

### 6.2 组织结构的局限

1. **单一分类维度**: 仅按功能角色分类, 缺乏按任务类型 (manipulation / navigation / locomotion) 或按训练范式 (IL / RL / self-supervised) 的交叉索引
2. **重复收录无标注**: MineDreamer 等出现在 3 个类别中, 但无交叉引用标注
3. **无元信息**: 不标注论文的会议/期刊、年份、引用量, 难以判断影响力
4. **无更新日志**: 难以追踪 repo 的更新历史和新增内容

### 6.3 时效性问题

Repo 最后实质性更新约在 2024 年中, 此后几乎停止维护。2024 下半年至 2025 年的重要进展未被收录:

| 遗漏的重要工作 | 年份 | 应属类别 |
|--------------|------|---------|
| pi_0 (Physical Intelligence) | 2024 | Robot Policy / VLA |
| GR00T N1 (NVIDIA) | 2025 | Robot Transformer / VLA |
| OpenVLA (Stanford) | 2024 | Robot Transformer / VLA |
| Octo (Berkeley) | 2024 | Robot Transformer |
| Scaling Laws for Robotics (Google) | 2024 | Neural Scaling Laws |
| ICRT (Berkeley) | 2024 | In-context Learning (已收录) |

---

## 7. Paper vs Code Discrepancies

### Repo 与关联论文 (FMRobotics) 的差异

| 维度 | 论文 (Firoozi et al.) | GitHub Repo |
|------|---------------------|-------------|
| **分类粒度** | 更细: 区分 plug-and-play / fine-tune / build-new FM | 更粗: 仅按功能分类, 不区分 FM 集成方式 |
| **挑战讨论** | 8 大挑战 (data scarcity, safety, UQ 等) 深入分析 | 无挑战讨论, 纯列表 |
| **Perception 覆盖** | 作为 autonomy stack 一层讨论 | 独立大板块, 覆盖更全 |
| **Embodied AI 覆盖** | 简要提及 | 独立大板块, Minecraft 生态详尽 |
| **后续更新** | 固定版本 (IJRR 2024) | 接受 PR, 新增了 ICRT, MineDreamer, MAGIC-VFM, FoundationPose 等 |

### Repo 新增但论文未覆盖的工作

以下工作出现在 repo 但不在原始论文中, 系后续社区贡献:

- **ICRT** (In-Context Robot Transformer, 2024): In-context learning 的新范式, 直接在 demo 上 condition
- **MineDreamer** (2024): Chain-of-Imagination, 结合 video generation 和 embodied control
- **MAGIC-VFM** (2024): Meta-learning + visual foundation model, 用于地面交互控制
- **FoundationPose** (2023): 6-DoF pose estimation, 填补了 Object Representations 的空白
- **BundleSDF** (2023): Neural 6-DoF tracking + 3D reconstruction
- **LOTUS** (2023): Continual imitation learning via skill discovery

---

## 8. Cross-Paper Comparison

### 8.1 本 repo 在本 paper library 中的定位

```
阅读路径建议:

[入门概览]
    FMRobotics_notes.md  (挑战与机遇的深度分析)
        |
        v
    本 repo (AwesomeSurvey)  (快速查找具体论文/代码)
        |
        v
    GeneralPurposeRobots_notes.md  (定量 meta-analysis)

[按方向深入]
    Robot Policy Learning -----> DiffusionPolicy_notes.md, pi0_notes.md, GR00T_N1_notes.md
    World Model ------------> DreamerV3_notes.md, 25_AwesomeWorldModels/
    Language Grounding -----> LanguageGrounding_notes.md, LangCondManip_notes.md
    Scaling Laws -----------> ScalingLaws_notes.md
    Decision Making --------> DecisionTransformer_notes.md
```

### 8.2 三个 Survey 资源的互补关系

| 需求 | 推荐资源 | 原因 |
|------|---------|------|
| 想快速找到某方向的论文和代码 | **本 repo** | 链接齐全, 按功能分类 |
| 想理解 FM for robotics 的全局挑战 | **FMRobotics** | 8 大挑战 + data scarcity 策略分析 |
| 想做定量比较或了解数据偏差 | **GeneralPurposeRobots** | 唯一有 meta-analysis 表格和 Open-X 统计的 survey |
| 想了解 world model 最新进展 | **25_AwesomeWorldModels** | 覆盖到 2025, 含 DreamerV3, GAIA, Cosmos 等 |
| 想了解 language grounding 细节 | **LangCondManip** | 语言 grounding 分析最深入 |

### 8.3 本 repo 分类框架与后续工作 (2024-2025) 的映射

Repo 的分类框架在 2024-2025 年的新工作面前暴露了结构性不足:

| Repo 分类 | 2024-2025 新趋势 | 框架适配性 |
|----------|-----------------|----------|
| Robot Transformers | VLA model 崛起 (pi_0, GR00T N1, OpenVLA) | **需要新建 VLA 子类**: 这些模型不仅是 transformer, 还融合了 diffusion/flow matching |
| Predictive Models | Foundation world model 快速发展 (Cosmos, UniSim) | **需要拆分**: 从 "预测模型" 细化为 video prediction / dynamics model / world model |
| Language-Conditioned IL | Action chunking + diffusion 成为主流 | **需要重组**: diffusion policy 横跨 IL 和 generative model 两个范畴 |
| Robot Task Planning | Reasoning + planning 融合 (CoT, chain-of-thought) | **基本适配**: LLM planning 框架仍然有效 |
| Perception | 3D foundation model 爆发 (3D-LLM, Point-E) | **需要扩展**: 从 open-vocabulary 扩展到 generative 3D |

### 8.4 总结

Awesome-Robotics-Foundation-Models 作为 Firoozi et al. survey 的配套资源库, 在 2023-2024 年初提供了该领域最系统的文献索引。其按 autonomy stack 功能角色的分类框架是合理的, 且被后续 survey (包括 GeneralPurposeRobots) 所参考。

主要价值:
1. **快速查找**: 按功能分类 + Paper/Project/Code 三链接, 适合快速定位资源
2. **分类参考**: 其 taxonomy 可作为组织 FM for robotics 文献的参考框架
3. **历史快照**: 记录了 2023 年 LLM 爆发期 robotics 社区的关注焦点

主要局限:
1. **停止更新**: 2024 下半年后基本停滞, 遗漏 VLA 浪潮的核心工作
2. **覆盖偏差**: 重 LLM planning / perception, 轻 dexterous manipulation / locomotion / tactile
3. **无分析**: 纯列表形式, 不提供方法论分析或性能比较

建议与 FMRobotics_notes.md (深度分析) 和 GeneralPurposeRobots_notes.md (定量比较) 配合使用。
