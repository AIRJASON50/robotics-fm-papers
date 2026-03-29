# Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis

**作者**: Yafei Hu, Quanting Xie, Vidhi Jain, Jonathan Francis, et al. (CMU, Bosch, Field AI, Georgia Tech, Meta, UC San Diego, Google DeepMind)
**会议/期刊**: arXiv preprint, v2.1-2024.09 (living document, 持续更新)
**论文链接**: https://robotics-fm-survey.github.io/

---

## 1. Core Problem

### 通用机器人的愿景

构建能在任意环境、操作任意物体、使用多种技能完成多样任务的 general-purpose robot，是 AI 的长期目标。但当前机器人系统面临根本性瓶颈:

- **特定化陷阱**: 为特定任务设计、在特定数据集上训练、部署到特定环境
- **泛化失败**: 面对分布偏移 (distribution shift) 时性能急剧下降
- **标注依赖**: 需要大量精确标注数据和 task-specific model

### Foundation Model 带来的机遇

NLP 和 CV 领域的 foundation model (LLM, VFM, VLM, diffusion model) 展现了强大的 open-set 和 zero-shot/few-shot 泛化能力。这一成功激发了两个研究方向:

1. **借用现有 foundation model**: 将 NLP/CV 的 LLM、VLM 等直接应用到 robotics (zero-shot/in-context learning)
2. **构建 robotics-specific foundation model**: 用机器人数据训练专属的 Robotics Foundation Model (RFM)

### 五大核心挑战

论文将通用机器人面临的、可能被 foundation model 缓解的挑战归纳为五类:

| 挑战 | 具体内容 | FM 解决程度 |
|------|---------|------------|
| **Generalization** | 感知、规划、控制在不同任务/环境/形态间的泛化 | 部分解决 (感知和任务规划较好，跨 morphology 控制仍难) |
| **Data Scarcity** | 机器人数据采集昂贵，缺乏 internet-scale 的 data flywheel | FM 可做数据增强/生成，但核心问题未解 |
| **Model Requirements** | 传统方法依赖精确的世界模型、地图、动力学模型 | RFM 的 model-free 方法可绕过，但 world model 仍是前沿 |
| **Task Specification** | 如何自然、无歧义地向机器人传达任务意图 | VLM 支持语言/图像/视频等多模态任务规范，效果较好 |
| **Uncertainty & Safety** | 部署安全性、不确定性量化、鲁棒性 | 严重不足，是最薄弱的环节 |

---

## 2. Method Overview: 分类框架与 Meta-Analysis 方法

### 统一问题建模

论文将 foundation model for robotics 统一建模为:

$$f(\mathbf{x_{t,k}}, \mathbf{c_k}) \rightarrow \mathbf{y_{t,k}} \quad \forall k \in N, \forall t \in T$$

- $\mathbf{x}$: sensory input (视觉、文本、场景图、音频、触觉等)
- $\mathbf{c}$: context (任务规范、embodiment 描述)
- $\mathbf{y}$: output (目标位姿、任务计划、下一状态、reward、控制输出)

### 核心分类二分法

论文提出的最重要分类维度是将现有工作分为两大阵营:

```
Foundation Models for Robotics
  |
  +-- (A) Foundation Models used in Robotics  --- zero-shot, modular, 无额外 fine-tune
  |       +-- VFM/VLM for Perception (open-set 物体/场景识别, 3D 语义地图, 状态估计)
  |       +-- LLM/VLM for Task Planning (SayCan, Code as Policy, PDDL)
  |       +-- LLM/VLM for Action Generation (reward synthesis, 评估 frontier)
  |       +-- Action Grounding (skill library, code interface, 3D value map)
  |       +-- Data Generation (LLM/VGM 生成训练数据/轨迹)
  |       +-- Prompting Enhancement (CoT, ToT, MCTS, PDDL)
  |
  +-- (B) Robotics Foundation Models (RFM)  --- 用机器人数据训练, end-to-end
          +-- Single-purpose RFM
          |     +-- Action Generation (RT-1/2, RoboCat, RT-X; IL/RL at scale; V/L pre-training)
          |     +-- Motion Planning (GNM, ViNT)
          +-- General-purpose RFM (Gato, PaLM-E, PACT)
```

**关键观察**: 方向 (A) 通常采用 modular 策略，FM 作为即插即用模块，各模块间无梯度流动。方向 (B) 采用 end-to-end 可微分范式，模糊了传统 perception-planning-control 的模块边界。

### Meta-Analysis 方法

论文对所收录的论文进行系统化的 meta-analysis，按 6 个任务类别建立详细分析表:

| 类别 | 对应表 |
|------|--------|
| Tabletop Manipulation | Table 2 |
| Dexterous Manipulation | Table 3 |
| Mobile Manipulation | Table 4 |
| Locomotion | Table 5 |
| Navigation | Table 6 |
| Multi-Tasks | Table 7 |

每个条目记录: 使用的 base FM、训练数据来源、机器人平台、评估指标等。

---

## 3. Key Designs: 最重要的分类维度与发现

### 3.1 Robotics 从 CV/LLM 借鉴了什么

这是本文的核心贡献之一。按功能模块拆解:

**感知层面 (Perception)**:
- VLM (CLIP, DINOv2, SAM) 提供 open-set 物体识别和场景理解
- 将 2D VLM feature 蒸馏到 3D 空间: F3RM, GNFactor 结合 NeRF，GeFF 做特征场
- CLIP feature 用于 SLAM 和定位: AnyLoc 实现了跨域 place recognition

**规划层面 (Planning)**:
- LLM 作为 task planner: SayCan 将高层任务分解为可执行子任务
- Code 比自然语言更好: Code as Policy, ProgPrompt 用代码做规划接口，能精确描述空间位置和参数
- 外部记忆增强: SayPlan 用 3D scene graph 管理大环境，打破 context length 限制

**动作生成层面 (Action)**:
- LLM 直接输出 low-level action 的困难: 动作通常不具语义可组合性
- Reward synthesis 是更通用的方案: Eureka 用 LLM 生成 RL reward function，学到了人类难以设计的技能 (如笔旋转)
- Tokenization 方案: RT-2 将 end-effector 空间以文本形式表达; Gato 直接将不同模态 (文本、关节角、按键) tokenize 为统一序列

### 3.2 Tokenization 与数据问题

**Tokenization 困境**:

论文揭示了一个核心张力: 语言 token 天然具有语义组合性，而机器人动作 token 不具备这一特性。不同工作采用了不同的接口方案:

| 接口类型 | 代表工作 | 优势 | 局限 |
|---------|---------|------|------|
| 自然语言 (skill name) | SayCan, OK-Robot | 易于 LLM 理解 | 粒度受限，无法描述精细动作 |
| 代码 (Python API) | Code as Policy, ProgPrompt | 可精确参数化，可组合 | 依赖预定义 primitive API |
| End-effector 文本编码 | RT-2 | VLM 原生兼容 | 仅限 7-DoF 末端执行器 |
| 统一 token 序列 | Gato | 跨任务统一 | 需要大量多任务数据 |
| 3D value map | VoxPoser | 可加安全约束 | 仅验证在简单任务 |
| Reward function | Eureka | 与 embodiment 无关 | 需要 RL 训练，sim-to-real gap |

**数据问题的深入分析** (基于 Open-X Embodiment Dataset):

论文对 Open-X 数据集的分析揭示了三个关键偏差:

1. **Morphology 偏差**: 73 个数据集中有 55 个是 single-arm manipulation; 仅 1 个四足、1 个双臂
2. **场景偏差**: 以桌面场景为主，多用玩具厨房物品 (隐含刚性、轻重假设)
3. **采集方法偏差**: 高度依赖人类专家 (VR/haptic device)，RT-1 数据集耗时 17 个月

这直接解释了为什么当前 foundation model 在 tabletop pick-place 上效果最好，而在 dexterous manipulation 和 locomotion 上进展有限。

### 3.3 Action Grounding 的频谱观

论文提出了一个有洞察力的分析框架: grounding-to-action 是一个频谱:

- **频谱一端**: Pre-trained skill library -- 高精度、高灵巧度，但任务多样性差
- **频谱另一端**: Map/constraint-based grounding -- 任务灵活性高，但仅验证在简单 2D gripper 任务

理想接口应同时兼顾 task diversity 和 task complexity。目前没有一个方案能统一这两端。

---

## 4. Experiments: Meta-Analysis 统计结果

### 4.1 Base Foundation Model 使用频率

根据 Figure 7 的统计:

| 排名 | 模型 | 使用原因 |
|------|------|---------|
| 1 | GPT-4 / GPT-3 | Few-shot promptable, API 可达性 |
| 2 | CLIP | 桥接图像和文本表征的标准选择 |
| 3 | ViLD | 开放词汇检测 |
| 4 | T5 family | 文本编码 |
| 5 | PaLM / PaLM-E | 机器人规划 |
| 6 | RT-1 | 新兴 base model，其他操作模型基于其构建 |

### 4.2 关键定量发现

**泛化性能下降**:
- Tabletop manipulation 中使用 FM 在 unseen tasks 上性能下降 21%-31%
- 面对扰动时性能下降 14%-18%

**控制频率瓶颈**:
- 当前方法推理速度通常在 1-10 Hz，多数为 open-loop
- 对比: 人形机器人 locomotion 需要约 500 Hz 的闭环控制
- 这是 FM 直接用于 low-level control 的硬性障碍

**任务覆盖严重不均**:
- 大量工作集中在 tabletop gripper manipulation (pick-place 及其变体)
- Dexterous manipulation 和 locomotion 的直接 low-level action output 探索极少
- 直接输出关节角的模型仅限 7-DoF 末端执行器任务

**缺乏统一 benchmark**:
- 各工作使用不同仿真环境、embodiment 和任务
- Success rate 作为主要指标不够，未考虑 inference latency
- 论文提到 Compute Aware Success Rate (CASR) 作为更合理指标

---

## 5. Related Work Analysis: 与其他 Foundation Model Survey 的定位差异

| 维度 | 本文 (GeneralPurposeRobots) | Firoozi et al. (FMRobotics) | Yang et al. | Wang et al. | Lin et al. |
|------|--------------------------|---------------------------|------------|------------|-----------|
| **范围** | 物理机器人 + 高保真仿真 | FM 如何提升机器人能力 | 广义自主 agent (非仅物理机器人) | 广义自主 agent | 仅 LLM for navigation |
| **核心贡献** | 分类法 + meta-analysis | 机遇与挑战分析 | 决策问题框架 | Agent 综述 | 领域综述 |
| **分类维度** | FM used in Robotics vs. RFM 的二分法 | FM 对 robot capability 的贡献 | 问题-方法-机遇 | 基于 LLM 的 agent | 导航专属 |
| **数据分析** | 对 Open-X 做了详细统计分析 | 无 | 无 | 无 | 无 |
| **Meta-analysis** | 有 (6 类任务，详细表格) | 无 | 无 | 无 | 无 |
| **模块化 vs E2E** | 明确讨论，提出非二元对立观点 | 有讨论 | 未涉及 | 未涉及 | 未涉及 |

**本文独特价值**: 在所有同期 survey 中，唯一提供系统化 meta-analysis (带统计表格) 的工作，也是唯一明确区分 "借用现有 FM" 和 "构建 Robotics FM" 两条路线的分类体系。

---

## 6. Limitations & Future Directions

### 论文自述的局限

1. **文献截止日期**: 2024.09.01，遗漏了之后的快速进展
2. **可能有不准确之处**: 由于文献体量大，承认可能存在错误

### 未来方向 (论文讨论)

| 方向 | 核心问题 | 当前状态 |
|------|---------|---------|
| **Embodiment Grounding** | 需要有效媒介桥接概念和动作; 自然语言/代码接口有限; 需多模态感知 grounding; 需考虑 embodiment 差异 | grounding 方案碎片化，无统一解 |
| **Safety & Uncertainty** | FM 缺乏原生不确定性推理; 需形式化安全保证 | 严重不足，是最大缺口 |
| **End-to-End vs Modular** | 两者非对立而是正交 (架构 vs 优化); brain analogy: 统一训练可能产生功能模块化 | 论文倡导 functional approach |
| **Embodiment Adaptability** | 同一模型适应不同 morphology、工具使用、故障恢复 | 初步结果 (RT-X, ViNT) 但远未解决 |
| **World Model** | 精确 world model + 经典 model-based 方法 vs data-scaled learning; LeCun 主张 world model 是关键 | Video generation 使 foundation world model 变得可行 |
| **Novel Platforms & Sensors** | 当前限于 gripper-based 单臂; 需要灵巧手、触觉、多传感器 | 硬件成本高、数据难采集 |
| **Continual Learning** | 持续适应动态环境，避免 catastrophic forgetting | 在 FM 上几乎未探索 |
| **Sim vs Real Data** | 大规模真实数据 vs 仿真数据+sim-to-real | 两条路线并行发展 |
| **Edge Deployment** | 工业环境需 model quantization, distillation, 边缘计算 | 性能与模型大小的 trade-off |

### 个人分析的补充局限

- **Transformer 偏向**: 论文覆盖的工作几乎全部基于 Transformer 架构，缺乏对 diffusion policy、state-space model 等新兴架构的系统讨论
- **Scaling law 未涉及**: 未讨论 robotics FM 的 scaling behavior (对比 NLP 的 scaling law 研究)
- **Multi-agent 场景缺失**: 几乎未涉及多机器人协作场景下的 FM 应用
- **Sim-to-real 讨论深度不足**: 作为连接仿真与真实世界的关键问题，论文仅浅尝辄止

---

## 7. Paper vs Code Discrepancies

N/A -- 本文为 survey 论文，无配套代码实现。

论文维护了一个 living GitHub repository (https://robotics-fm-survey.github.io/)，用于持续更新文献列表。

---

## 8. Cross-Paper Comparison

### 与本库其他 survey/综述的互补

| 维度 | 本文 (GeneralPurposeRobots) | Awesome-Robotics-FM (24_AwesomeSurvey) | FMRobotics (23_FMRobotics) | LangCondManip (23_LangCondManip) |
|------|--------------------------|---------------------------------------|---------------------------|-------------------------------|
| **形式** | 正式学术 survey + meta-analysis | GitHub awesome-list (资源索引) | 正式 survey | 正式 survey |
| **聚焦点** | 通用机器人 (全任务类型) | 全面资源收集 | FM 提升机器人能力 | 语言条件化操作 |
| **任务覆盖** | Manipulation + Navigation + Locomotion + Multi-task | 全覆盖 (仅列表) | 全覆盖 | 仅 manipulation |
| **分析深度** | 方法论分析 + 定量 meta-analysis | 无分析 | 挑战与机遇讨论 | 方法论深入分析 |
| **独特贡献** | 唯一的 meta-analysis 表格; FM used in vs. RFM 二分法; Open-X 数据分析 | 最全的文献/项目列表 | safety, sim-to-real 讨论更深 | 语言 grounding 最详细 |
| **时效性** | 2024.09 截止 | 持续更新 | 2023 | 2023 |

### 与本库技术论文的关联

| 本库论文 | 与本 survey 的关系 |
|---------|------------------|
| **DiffusionPolicy** (24) | 本 survey 未充分覆盖 diffusion-based action generation，DiffusionPolicy 代表了 survey 截止后的重要进展 |
| **DecisionTransformer** (21) | 本 survey 涵盖的 offline RL at scale 路线的前身，tokenize trajectory 的早期范式 |
| **DreamerV3** (23) | 本 survey 讨论的 world model 方向的具体实例，但 survey 未深入分析 dreamer 系列 |
| **GR00T N1** (25) | survey 截止后出现的典型 Robotics Foundation Model，验证了 survey 对 general-purpose RFM 方向的预测 |

### 阅读建议

1. **入门路径**: 本 survey --> Awesome-Robotics-FM (查资源) --> 具体技术论文
2. **互补阅读**: 本文提供全局视角和定量分析; FMRobotics 补充 safety/sim-to-real 讨论; LangCondManip 深入 language grounding 细节
3. **时效补充**: 本文截止于 2024.09，之后的重要进展包括 OpenVLA, GR00T N1, 以及各种 VLA 模型的涌现，需要参考更新的文献
