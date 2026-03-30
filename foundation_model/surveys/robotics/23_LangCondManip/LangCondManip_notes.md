# Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation

**Paper**: Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation
**Authors**: Xiangtong Yao, Hongkuan Zhou, Oier Mees, Yuan Meng, Ted Xiao, Yonatan Bisk, Jean Oh, Edward Johns, Mohit Shridhar, Dhruv Shah, Jesse Thomason, Kai Huang, Joyce Chai, Zhenshan Bing, Alois Knoll
**Affiliations**: TU Munich, Bosch, UC Berkeley, Microsoft, Google DeepMind, CMU, Imperial College London, Princeton, USC, Sun Yat-sen University, U Michigan, U Stuttgart, Nanjing University
**arXiv**: 2312.10807v6, 2024 (updated 2026-03-26)

---

## 1. Core Problem

Language-conditioned robot manipulation 的核心挑战是: **如何将自然语言指令转化为机器人的物理操作动作**。这个问题的难点在于需要同时解决三个紧耦合的子问题:

1. **Language Understanding**: 理解自然语言指令的语义，包括组合性 (compositional) 指令、模糊表述、负约束 ("stay away from the vase") 和安全修饰语 ("slowly")
2. **Visual Perception**: 将语言中描述的概念 grounding 到实际视觉观察中，识别 "the red cup" 对应的具体物体
3. **Action Generation**: 将高层语义目标转化为低层可执行的机器人控制信号 (joint torques / end-effector poses / skill sequences)

传统方法 (专用编程、遥操作、reward engineering) 不具备泛化性且需要专家知识。Language-conditioned manipulation 的核心愿景是让非专家通过自然语言 "zero-learning" 地指挥机器人，实现通用目的的人机协作。

**论文的关键视角**: 不按模型类型 (LLM/VLM/VLA) 或替代的模块 (perception/planning/control) 分类，而是按 **语言在操作控制回路中扮演的功能角色** 进行分类，提供正交于传统分类法的新维度。

---

## 2. Method Overview

### 2.1 系统架构: Language-Perception-Control 三模块

论文将 language-conditioned manipulation pipeline 分解为三个交互模块:

| 模块 | 功能 | 典型技术 |
|------|------|--------|
| Language Module | 指令理解、任务表征 | NLMs, PLMs, LLMs |
| Perception Module | 环境状态估计、语言 grounding | Object Detection, Semantic Segmentation, 3D Reconstruction, VLMs |
| Control Module | 将任务规范转化为可执行动作 | RL, IL (BC/GCIL/IRL), Diffusion Policy, Motion Planning |

两个关键回路:
- **Interactive Loop** (左侧): 人-机器人语言交互
- **Control Loop** (右侧): agent 与环境的 action-observation 交互

### 2.2 核心分类框架: 语言的四种功能角色

这是论文最核心的贡献——按语言进入控制回路的方式分为四类:

| 类别 | 语言角色 | 核心问题 | 对应 Section |
|------|--------|---------|-------------|
| Language for State Evaluation | 将语言转化为 reward / cost function，量化任务进度 | 如何用语言评估 "做得怎么样"？ | Sec. 4 |
| Language as a Policy Condition | 语言作为策略的直接条件输入，指导行为生成 | 如何用语言条件化 "怎么做"？ | Sec. 5 |
| Language for Cognitive Planning & Reasoning | 语言作为内部推理媒介，分解任务、制定策略 | 机器人如何用语言 "思考"？ | Sec. 6 |
| Language in Unified VLA Models | 语言嵌入端到端 VLA 模型中，与视觉和动作统一建模 | 如何将视觉、语言、动作统一到单一模型中？ | Sec. 7 |

---

## 3. Key Designs

### 3.1 Language for State Evaluation (Sec. 4)

核心思想: 将语言指令翻译为可量化的评分函数。

**Reward Functions** (用于 RL):

| 方法演进 | 代表工作 | 机制 |
|--------|---------|------|
| Sparse Reward Design | ZSRM (Mahmoudieh 2022) | CLIP 计算 camera image 与 goal text 的相似度作为 reward |
| Dense Reward Design | PixL2R (Goyal 2021) | 学习 trajectory-language relatedness model 生成连续 shaping reward |
| Reward Learning | LOReL (Nair 2022) | 训练 binary classifier 判断 trajectory 是否满足 language instruction |
| FM-driven Reward Code Gen | Text2Reward (Xie 2024), EUREKA (Ma 2024) | LLM 直接生成 Pythonic reward function 代码，evolutionary refinement |
| VLM-driven Reward Learning | Video-Language Critic, ReWiND (Zhang 2025c) | VLM 从 pixels+language 直接学 dense reward，绕过 simulator state |

关键洞见: Reward 设计从手工 -> 数据驱动 -> FM 自动生成的演进。FM-driven 方法的瓶颈在于生成代码的 weight tuning，Reward-Self-Align 和 R* 通过 preference alignment 解决。

**Cost Functions** (用于 motion planning):

| 方法 | 机制 |
|------|------|
| VoxPoser (Huang 2023b) | GPT-4 写 Python 查询 OWL-ViT 构建 3D value map |
| ReKep (Huang 2025b) | VLM 生成 Python code 定义 3D keypoint 间的 arithmetic constraints |
| IMPACT (Ling 2025) | GPT-4o 评估 "acceptable contact" 生成 3D cost map，集成 RRT* |

### 3.2 Language as a Policy Condition (Sec. 5)

核心思想: 语言从 "goal specifier" 转变为 "behavior specifier"，直接作为 policy 的输入条件。

**在 RL 中的应用**:

| 方法 | 关键贡献 |
|------|--------|
| LanCon-Learn (Silva 2021) | Language embedding 通过 attention router gate 多个 skill modules |
| MILLION (Bing 2023a) | Memory-based meta-learning with Gated Transformer-XL，读-做解耦 |
| FLaRe (Hu 2025a) | Fine-tune 大型预训练 BC policy with sparse linguistic rewards，比 dense-reward baseline 快 15x |

**在 BC 中的应用**:

| 方法 | 关键贡献 |
|------|--------|
| MCIL (Lynch & Sermanet 2021) | 仅 1% language-annotated data 即可有效学习 |
| PerAct (Shridhar 2023) | Voxelize RGB-D + 6-DoF action space，强 3D inductive bias，few-demo learning |
| HULC (Mees 2022a) | Contrastive learning 构建 robust language-conditioned representations，long-horizon |
| ACT (Zhao 2023a) | Action chunking 缓解 compounding error |

**在 Diffusion Policy 中的应用**:

| 方法 | 关键贡献 |
|------|--------|
| StructDiffusion (Liu 2023c) | 语言作为 high-level constraint 引导 goal pose sampling |
| ChainedDiffuser (Xian 2023) | Global language transformer + local trajectory diffuser，long-horizon |
| PoCo (Wang 2024c) | Policy composition: 多个 pre-trained diffusion policies 通过语言组合 |
| Scaling&Distilling (Ha 2023) | LLM 提出 textual subtasks，在 simulation 执行后 distill 成 compact diffusion policy |

### 3.3 Language for Cognitive Planning & Reasoning (Sec. 6)

核心思想: 语言作为推理的内部工具，机器人在 "语言空间" 中思考。

**Classic Neuro-symbolic**:
- Learning for reasoning: 神经网络做 perception，符号系统做 planning (DANLI, HiTUT)
- Reasoning for learning: 符号知识 (STRIPS, verb clauses) 约束神经学习
- Learning-reasoning: 符号与神经紧密耦合互相增强 (Chai 2018 Interactive Task Learning)

局限: KG 构建成本高、ontology drift、缺乏 commonsense

**Empowered by LLMs**:

| 子类 | 代表工作 | 机制 |
|------|--------|------|
| Open-loop Planning | SayCan (Ahn 2022) | LLM + affordance functions，不接收执行反馈 |
| Closed-loop Planning | SayPlan (Rana 2023) | 3D scene graph 持续提供 textual feedback 给 LLM |
| Summarization | Tidybot (Wu 2023a) | Few-shot prompting 归纳整理策略 |
| Code Generation | Code as Policies (Liang 2023) | LLM 生成 Pythonic policy code，API composition |
| Iterative Reasoning | Inner Monologue (Huang 2022b) | Closed-loop self-reflection，success detector 反馈 |
| LLMs + PDDL | IALP (Wang 2025a) | Grounding mechanisms 将 PDDL predicates 连接物理现实 |
| LLMs + Behavior Trees | BETR-XP-LLM (Styrud 2025) | LLM 作为 "repair agent" 提出 minimal precondition + matching subtree |

**Empowered by VLMs**:
- Contrastive: CLIP/CLIPORT 做 language-visual alignment
- Generative (text): PaLM-E (Driess 2023) 562B 参数单模型直接生成 robot-executable plans
- Generative (image): SuSIE (Black 2024) 生成 subgoal images 解耦语义理解与低层控制
- World models: 生成未来 video frames 用于 planning 和 data augmentation

### 3.4 Language in Unified VLA Models (Sec. 7)

核心思想: 语言不再是外部条件，而是在统一的 VLA backbone 中与 vision/action 共同建模。

论文提出 VLA 的四维优化分类: Perception -> Reasoning -> Action -> Learning & Adapting

**Perception 优化**:

| 方向 | 代表工作 | 方法 |
|------|--------|------|
| Data Sources | EgoVLA | Egocentric human video pre-train + robot fine-tune |
| 3D Scene | SpatialVLA, PointVLA, BridgeVLA | Ego3D Position Encoding / 3D point cloud injection / 2D orthographic projection |
| Multimodal Sensing | VTLA, Tactile-VLA, ForceVLA | Tactile tokenization, force-aware MoE |

**Reasoning 优化**:

| 方向 | 代表工作 | 方法 |
|------|--------|------|
| Long-horizon | LoHoVLA, DexVLA | Hierarchical decomposition, "think before act" |
| World Models | Seer, CoT-VLA, WorldVLA, DreamVLA | Visual foresight, bidirectional prediction, compact "world embedding" |
| Knowledge Preserving | ChatVLA, Knowledge Insulation (Driess 2025) | MoE 隔离 task-specific parameters / gradient stopping |

**Action 优化**:

| 方向 | 代表工作 | 方法 |
|------|--------|------|
| Continuous Actions | pi_0 (Black 2025b) | Flow matching action expert 生成 50Hz continuous actions |
| Action Tokenization | pi_Fast (Pertsch 2025) | FAST tokenizer (DCT) 压缩 action 冗余 |
| Hybrid | pi_0.5 (Black 2025a) | 联合学习 discrete + continuous action heads |
| Discrete Diffusion | Discrete Diffusion VLA (Liang 2025) | 统一 cross-entropy objective 内做 discrete diffusion |

**Learning & Adapting**:

| 方向 | 代表工作 | 方法 |
|------|--------|------|
| Efficient Fine-tuning | OpenVLA-OFT (Kim 2025b) | Parallel decoding + action chunking + continuous L1 regression |
| Few-shot Adaptation | ControlVLA (Li 2025e) | Object-centric mask + zero-initialized projection，10-20 demos |
| RL Integration | ConRFT (Chen 2025h), RIPT-VLA (Tan 2025) | Reinforced fine-tuning pipeline |

---

## 4. Experiments: Benchmarks and Evaluation

### 4.1 主要仿真 Benchmarks

| Benchmark | 仿真引擎 | 机器人 | 数据量 | 特点 |
|-----------|---------|--------|-------|------|
| CALVIN | PyBullet | Franka Panda | 2400k | Long-horizon，MTLC + LH-MTLC 指标 |
| Meta-World | MuJoCo | Sawyer | - | 50 tasks，ML10/ML45 |
| RLBench | CoppeliaSim | Franka Panda | - | 100 tasks，多传感器 |
| VIMAbench | PyBullet | UR5 | 650K | 17 task templates，4-level evaluation |
| LoHoRavens | PyBullet | UR5 | 15k | 10 long-horizon tasks |
| ARNOLD | NVIDIA Omniverse | Franka Panda | 10k | 40 objects, 20 scenes，高视觉真实感 |
| LIBERO | MuJoCo | Franka Panda | 6.5k | Lifelong learning，4 task suites |
| RoboGen | PyBullet | Multiple | - | 自动任务/场景/监督生成 |

### 4.2 主要真实世界 Datasets

| Dataset | 数据量 | 机器人 | 特点 |
|---------|-------|--------|------|
| Open X-Embodiment | 2M+ trajectories | 22 robots | 527 skills, 160K+ tasks，跨机构协作 |
| DROID | 76k trajectories | Franka Panda | 564 scenes, 52 buildings, crowd-sourced language |
| Galaxea Open-World | 100k trajectories | Galaxea R1 Lite | 150 tasks, 50 scenes, bimanual |

### 4.3 Comparative Analysis 五维度

论文在 Section 8 提出五个正交分析维度:

| 分析维度 | 关键发现 |
|--------|--------|
| **Action Granularity** | Skill-level (SayCan) vs. Trajectory-level (HULC, PerAct) vs. Low-level torque (ManiFoundation, TA-VLA)。Skill-level 与语言天然匹配但受 skill library 限制；Trajectory-level 表达性与可控性平衡最好；Torque-level 用于接触丰富任务但 sim-to-real 困难 |
| **Data & Supervision** | Expert demos (高质量但昂贵) vs. Play data (规模大但需 post-hoc labeling) vs. Web-scale data (提供 broad priors)。三种 supervision: target labels, outcome evaluations, auxiliary (attention/reconstruction/prediction) |
| **System Cost & Latency** | VLA policy 7B params ~3-16Hz on RTX4090; Diffusion policy ~200M params ~1-60Hz; LLM planner (540B PaLM) seconds-scale latency。Real-time performance 是被忽视但关键的评估维度 |
| **Environments & Evaluations** | Sim-only 结果可能隐藏 calibration drift 和 sensor noise。缺乏统一的 real-world benchmark 评估语言 grounding 质量 |
| **Cross-modal Task Spec** | Language 提供 compositional generalization 和 negative constraints；Image/Video 提供更精确的 geometric grounding 和 kinematic priors。趋势: 语言指定 "what"，视觉指定 "how" 的多模态融合框架 |

### 4.4 Inference Cost 对比 (Table 6)

| 系统 | 模型规模 | 硬件 | 频率 | Cloud? |
|------|---------|------|------|--------|
| RT-1 | 35M VLA | - | ~3Hz | No |
| RT-2 55B | 55B VLA | TPU | 1-3Hz | Yes |
| OpenVLA | 7B VLA | 1x RTX4090 | 3-6Hz | No |
| PaLM-SayCan | 540B planner | TPU | - | Yes |
| PaLM-E | 562B VLM | TPU | - | Yes |
| Diffusion Policy | ~200M | - | ~1Hz | No |
| ManiCM | ~200M | 1x RTX4090 | ~50-60Hz | No |

---

## 5. Related Work Analysis: 领域发展脉络

### 5.1 技术演进时间线

**Phase 1: Pre-LLM 时代 (2015-2021)**
- Temporal logic / formal specification languages 用于语言-动作映射
- Text embedding (GloVe, BERT) + CNN/RNN 做 language-conditioned policy
- Goal-conditioned IL 框架适配语言作为 goal representation
- 代表: MCIL (Lynch 2021), LanCon-Learn (Silva 2021), CLIPort (Shridhar 2022)

**Phase 2: LLM 作为 planner (2022-2023)**
- SayCan (Ahn 2022): LLM + affordance scoring 开创性工作
- Code as Policies (Liang 2023): LLM 生成可执行 code
- SayPlan (Rana 2023): Closed-loop planning with 3D scene graph feedback
- Inner Monologue (Huang 2022b): Iterative reasoning with feedback
- PaLM-E (Driess 2023): 562B multimodal model 做 embodied reasoning

**Phase 3: VLA 统一模型 (2023-2025)**
- RT-2 (Zitkovich 2023): Co-fine-tune VLM + robot data
- OpenVLA (Kim 2025c): 7B open-source VLA
- pi_0 (Black 2025b): Flow matching for continuous action generation
- pi_0.5 (Black 2025a): Hybrid discrete+continuous heads with FAST tokenizer

**Phase 4: 当前前沿 (2025-2026)**
- 3D 感知融入 VLA (SpatialVLA, PointVLA, BridgeVLA)
- Tactile/Force sensing 融合 (VTLA, Tactile-VLA, ForceVLA)
- World model 集成 (WorldVLA, DreamVLA, CoT-VLA)
- Knowledge preservation (ChatVLA, Knowledge Insulation)
- Efficient adaptation (ControlVLA, OpenVLA-OFT, ConRFT)

### 5.2 与现有 Survey 的区别

| Survey | 组织维度 | 本文独特之处 |
|--------|--------|-----------|
| Tellex et al. (2020) | "Lexically grounded" vs "learning methods" | Pre-LLM 时代的 foundational survey |
| Hu et al. (2023), Li et al. (2024a) | 按 FM 类型 (LLM/VLM) 分类 | 关注 FM 对 robotics 各模块的增强 |
| Firoozi et al. (2025) | 按 capability ("Perception"/"Decision-making"/"Control") 分类 | 按 FM 替换的 robotic module 组织 |
| **本文** | **按语言在控制回路中的功能角色** | 允许跨算法范式 (RL/IL/Planning) 的 fine-grained 分析 |

---

## 6. Limitations & Future Directions

### 6.1 三大核心辩论 (Section 9)

**Debate 1: VLA 是否是正确方向？**
- 支持: Scaling laws 在 NLP/CV 中成功，VLA 统一建模减少模块间信息损失
- 质疑: Robot data 稀少且昂贵 (多数 VLA <7B params, few million trajectories); 物理世界中微小错误级联导致 total failure; 实证表明 task complexity/diversity 扩展比 model size 增长更重要
- 论文立场: "genuine scaling lies in expanding the task manifold"——任务复杂度、多样性和交互结构的扩展比单纯增大参数量更关键

**Debate 2: World Models 是否是正确方向？**
- 优势: Sample efficiency (imagination-based learning); safety filtering (offline what-if evaluation); 共享 dynamics priors 促进跨任务泛化
- 挑战: Model bias + distributional brittleness (小误差在 rollout 中累积); grounding imagined rollouts in real perception; 实时控制下的计算开销
- 开放问题: (i) 注入结构化先验 (physics, geometry, commonsense); (ii) 量化并传播模型不确定性; (iii) compute-efficient models 满足实时控制

**Debate 3: Scaling 是否在实时约束下有用？**
- Scaling 提升表示质量和指令理解范围，但物理机器人有固定 control cycle time budget
- 大模型推理延迟可能导致 control loop miss deadline -> jitter/instability，尤其在 contact-rich tasks
- 实践策略: model compression/distillation, pruning input tokens, multi-view 2D encoders (替代 heavy 3D volumetric), split architecture (lightweight onboard + heavy offboard async)
- 关键观点: Latency 应成为 first-class evaluation metric (目前多数论文仅在 footnotes 中提及)

### 6.2 未来方向 (Section 10)

**Generalization Capability**:
1. **Data**: 大规模、多样化、多模态对齐的 manipulation dataset
2. **Lifelong Learning**: 持续适应新任务+避免 catastrophic forgetting
3. **Cross-embodiment Alignment**: 学习共享的 semantic-to-control representations 跨不同机器人形态
4. **Zero-shot Capability**: 当前 zero-shot 在语义层/planning 层强，但在 policy/execution 层弱且脆弱

**Real-world Safety**:
1. **Ambiguity in Language**: "Remove the chemicals from the table" 可能有危险解释。需要 clarification dialogue 和 feedback loops
2. **Recovering from Failures**: LLM hallucination 产生不安全 plan; 硬件限制 (motor overheating, sensor interference)。需要 closed-loop self-verification 和 human-in-the-loop supervision
3. **Real-time Performance**: 大模型推理延迟 + cloud-based inference 的网络不确定性。需要 model compression, hybrid local/cloud architecture, 安全通信协议

---

## 7. Paper vs Code Discrepancies

N/A (survey paper, 无代码实现)

---

## 8. Cross-Paper Comparison

### 8.1 与 Awesome-Robotics-Foundation-Models Survey 对比

| 维度 | Awesome-Robotics-FMs (GitHub survey, 2024) | 本文 (Bridging Language and Action) |
|------|------------------------------------------|--------------------------------------|
| 组织维度 | 按 FM 类型 (LLM, VLM, VLA) + 应用领域 (navigation, manipulation, locomotion) | 按语言在 manipulation 控制回路中的功能角色 |
| 覆盖范围 | 广泛覆盖 robotics 所有子领域 | 深度聚焦 manipulation |
| 分析深度 | 文献列表+简要分类，侧重 "什么模型做了什么" | 系统性分析每类方法的演进逻辑、优劣势和 trade-offs |
| 正交分析 | 无 | 五维 comparative analysis (action granularity, data/supervision, cost/latency, environments, cross-modal task spec) |
| 讨论深度 | 列表式，无 debates | 三个核心 debates (VLA scaling, world models, real-time constraints) |
| 更新程度 | 持续更新的 GitHub repo | 截至 2025 年中 (v6 updated 2026-03) |
| 互补价值 | 作为全领域的 reference index | 作为 manipulation 子领域的深度技术分析 |

### 8.2 与 Foundation Models 相关 Survey 对比

| 维度 | Firoozi et al. (2025) FM in Robotics | 本文 |
|------|--------------------------------------|------|
| 视角 | FM 增强 robotics 的 capabilities (Perception / Decision-making / Control) | 语言在 manipulation pipeline 中的功能角色 |
| 分类 | 按被替换/增强的 robotic module | 按语言进入控制回路的方式 |
| 关键差异 | 同一技术 (如 RL agent 用 language reward) 归入 "Decision-making" | 区分 language 用于 reward (State Evaluation) vs. 直接条件化 policy (Policy Condition) -- 功能完全不同 |
| 优势 | 与传统 robotics pipeline 对应清晰 | 揭示 language 在不同角色下的设计 trade-offs |

### 8.3 与本库其他论文的关联

| 本库论文 | 与本 survey 的关系 |
|--------|----------------|
| Diffusion Policy (Chi 2023) | 属于 Sec. 5.3 "Language in Diffusion-based Policy Learning" 的基础方法。Survey 指出 Diffusion Policy 通过生成式建模解决了 BC 的 multi-modal action averaging 问题，但 inference latency 是瓶颈 |
| Decision Transformer (2021) | 属于 Sec. 5.1 "Language in RL" 的相关方法范式——将 RL 转化为 sequence modeling |
| DreamerV3 (2023) | 属于 Sec. 9.2 "Are World Models the Right Path Forward?" 的核心讨论对象。Survey 引用 Dreamer 家族作为 imagination-based learning 的代表 |
| GR00T N1 (2025) | 属于 Sec. 7 "Unified VLA Models" 的最新代表。作为 humanoid 通用 FM，与 survey 讨论的 cross-embodiment alignment 高度相关 |

### 8.4 对 Manipulation 研究的独特价值

相比其他 survey，本文对 manipulation 研究者最直接的价值在于:

1. **Reward 设计指南**: Section 4 系统梳理了从手工 reward -> learned reward -> FM-generated reward 的完整技术栈，对 RL-based manipulation 极为实用
2. **BC/Diffusion Policy 的语言条件化方法选择**: Section 5 清晰对比了 RL/BC/Diffusion 三种范式在语言条件化下的优劣势
3. **LLM planning 的 grounding 问题**: Section 6 深入分析了 open-loop vs. closed-loop planning 的 trade-offs，以及 PDDL/BT 等结构化方法与 LLM 的结合方式
4. **VLA 模型的系统性分类**: Section 7 按 Perception/Reasoning/Action/Learning 四维度整理了 30+ VLA 方法，是目前对 VLA 最全面的分类
5. **Cross-modal task specification**: Section 8.5 分析了 language vs. image/video conditioning 各自擅长的场景，指导多模态系统设计
