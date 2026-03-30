# Foundation Models in Robotics: Applications, Challenges, and the Future

**Paper**: Firoozi et al., IJRR 2024 (arXiv 2312.07843, Dec 2023)
**Authors**: Stanford, Princeton, UT Austin, NVIDIA, Scaled Foundations, Google DeepMind, TU Berlin, SJTU
**GitHub**: [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)

---

## 1. Core Problem

这篇 survey 聚焦一个根本性的时代问题: **Foundation model 的成功能否从 CV/NLP 迁移到 robotics?**

传统 robot learning 在小规模、任务特定的数据集上训练, 导致适应性差、泛化弱。Foundation model (FM) 在 internet-scale 数据上预训练, 展现出 zero-shot 泛化和 cross-task transfer 能力, 这恰好是 robotics 所缺乏的。但 robotics 面临独特挑战:

| 挑战维度 | 具体问题 | CV/NLP 中的对应 |
|----------|---------|----------------|
| **Data Scarcity** | 机器人 manipulation/locomotion/navigation 数据无法从 Internet 获取, 需要 teleoperation 或 sim 生成 | NLP 有 Common Crawl (TB 级文本), CV 有 LAION-5B |
| **High Variability** | 硬件平台、物理环境、物体多样性导致 embodiment gap 巨大 | CV 中不同 camera 的差异远小于不同 robot 的差异 |
| **Uncertainty Quantification** | LLM hallucination 在 robotics 中可能导致物理危险, 需要严格的 UQ | 聊天场景可容忍一定幻觉, 自动驾驶/机器人不行 |
| **Safety Evaluation** | 部署前/部署中/更新后的全生命周期安全测试 | NLP 中以 red-teaming 为主 |
| **Real-Time Performance** | Foundation model 推理延迟高 (GPT-4 秒级), 机器人需要 ms 级响应 | 文本生成可接受延迟, 控制回路不行 |

论文的核心观点是: **Foundation model 对 robotics 的影响不是简单替换, 而是重塑整个 autonomy stack** -- 从 perception 到 decision-making 到 control, 每一层都可以被 FM 增强或替代。

---

## 2. Method Overview

论文提出了一个三层分类框架, 将 FM 在 robotics 中的应用映射到 autonomy stack 的不同层级:

### 2.1 分类框架总览 (Figure 1)

```
Foundation Models in Robotics
|
+-- Robotics (直接应用于机器人的工作)
|   +-- Robot Policy Learning (Decision Making & Control)
|   |   +-- Language-Conditioned Imitation Learning
|   |   +-- Language-Assisted Reinforcement Learning
|   +-- Language-Image Goal-Conditioned Value Learning
|   +-- Robot Task Planning (LLM-based)
|   |   +-- Language Instructions for Task Specification
|   |   +-- Code Generation for Task Planning
|   +-- In-context Learning for Decision-Making
|   +-- Robot Transformers
|   +-- Open-Vocabulary Navigation and Manipulation
|
+-- Perception (与 robotics 相关的感知)
|   +-- Open-Vocabulary Object Detection & 3D Classification
|   +-- Open-Vocabulary Semantic Segmentation
|   +-- Open-Vocabulary 3D Scene Representations
|   +-- Learned Affordances
|   +-- Predictive Models (World Models)
|
+-- Embodied AI
    +-- Generalist AI
    +-- Simulators
```

### 2.2 论文覆盖的 Foundation Model 类型

| FM 类型 | 代表模型 | 参数量 | 在 Robotics 中的角色 |
|--------|---------|--------|---------------------|
| LLM | GPT-3/4, PaLM | 175B+ | Task planning, code generation, reward shaping |
| VLM | CLIP, BLIP | 0.3B~55B | Visual grounding, zero-shot detection, reward function |
| Embodied Multimodal LM | PaLM-E | 562B | End-to-end perception-to-action |
| Vision Transformer | ViT, DINOv2, SAM | 1.1B~4B | Visual representation, segmentation |
| Diffusion Model | DALL-E 2, Stable Diffusion | 3.5B~12B | Data augmentation, scene editing, goal generation |

### 2.3 论文将工作分为三类

1. **Background Papers**: 不直接涉及 robotics, 但对理解 FM 不可或缺 (如 BERT, GPT, CLIP)
2. **Robotics Papers**: 直接将 FM 集成进 robot system (plug-and-play 或 fine-tune 或新建 FM)
3. **Robotics-Adjacent Papers**: 方法来自 CV/embodied AI, 但有明确的 robotics 应用路径

---

## 3. Key Designs

### 3.1 FM 在 Robotics Autonomy Stack 中的角色分工

这是论文最重要的分析维度 -- FM 如何嵌入 robot autonomy stack 的每一层:

| Autonomy Layer | FM 的作用 | 代表工作 | Plug-and-Play vs Fine-tune |
|----------------|---------|---------|---------------------------|
| **Task Specification** | LLM 将自然语言转为 executable plan | SayCan, ProgPrompt, Code-as-Policies | Plug-and-Play (直接调用 GPT/PaLM API) |
| **Task Planning** | LLM 做 high-level plan, 分解 long-horizon task | Inner Monologue, Text2Motion, NL2TL | Plug-and-Play |
| **Perception** | VLM 做 open-vocabulary detection/segmentation/scene understanding | OWL-ViT, GLIP, SAM, LERF, CLIP-Fields | Plug-and-Play (冻结权重) |
| **Value Learning** | VLM 提供 goal-conditioned reward signal | R3M, VIP, LIV, VoxPoser | Fine-tune visual encoder |
| **Policy Learning** | Transformer/VLA 端到端输出 action | RT-1, RT-2, RT-X, PACT, RoboCat | Train from scratch / Fine-tune |
| **Data Augmentation** | Diffusion model 生成多样训练数据 | ROSIE, DALL-E-Bot, GenAug, CACTI | Plug-and-Play |

**关键洞察**: 越靠近底层 (控制), FM 越需要 fine-tune 或从头训练; 越靠近顶层 (规划), plug-and-play 越有效。这反映了 language 和 action 之间的 semantic gap -- LLM 的知识可以直接用于规划, 但转化为精确的物理动作需要额外的 grounding。

### 3.2 Language-Conditioned Imitation Learning 的演进

论文梳理了 language-conditioned manipulation 的完整脉络:

```
Play-LMP (2020): teleoperated play data --> latent plan --> goal-conditioned policy
    |
    v
CLIPort (2022): CLIP semantic + Transporter spatial --> 2D manipulation
    |
    v
PerAct (2023): 3D voxel + CLIP language encoder --> 6-DoF voxel action
    |
    v
CACTI (2022): Stable Diffusion augmentation + visual representation + multi-task policy
    |
    v
RT-1 (2023): 130k real demos, 700+ tasks, 13 robots --> generalize to new tasks
    |
    v
RT-2 (2023): VLM co-fine-tune --> action tokenization --> chain-of-thought prompting
    |
    v
RT-X (2023): 22 robots, 21 institutions, 527 skills --> cross-embodiment transfer
```

这条路线的核心趋势是: **数据规模扩大 + 模型规模扩大 + 跨 embodiment 统一**, 本质上是在走 NLP 从 word2vec 到 GPT-4 的路。

### 3.3 LLM 作为 Task Planner 的两条路线

论文清晰区分了 LLM 做 planning 的两种范式:

| 路线 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| **Language Specification** | SayCan (affordance grounding), Text2Motion (geometric feasibility), VoxPoser (3D value maps) | 可利用 LLM 的 commonsense reasoning | 依赖 predefined skill primitives, 难以覆盖 novel skills |
| **Code Generation** | Code-as-Policies, ProgPrompt, ChatGPT-Robotics, Voyager | LLM 可直接生成可执行代码, 组合性强 | 需要 well-defined API, 代码 bug 可能导致危险 |

Code-as-Policies 的核心 insight 尤为重要: 把 robot policy 表达为代码 (而非 neural network), 可以直接用 LLM 生成。代码天然具有组合性 (调用函数) 和精确性 (数值参数), 比自然语言 plan 更适合 robot execution。

### 3.4 Open-Vocabulary Perception 的分层架构

论文展示了 FM 如何逐层增强 robot perception:

| 层级 | 任务 | 核心 FM | 关键限制 |
|------|------|--------|---------|
| 2D Detection | Open-vocabulary object detection | OWL-ViT, GLIP, Grounding DINO | 仅 2D, 无深度 |
| 2D Segmentation | Promptable segmentation | SAM, LSeg, FastSAM, MobileSAM | SAM 无法 real-time |
| 3D Classification | Zero-shot 3D object recognition | PointCLIP, PointBERT, ULIP | 需要 point cloud, 依赖 CLIP 对齐 |
| 3D Scene | Language-grounded 3D representation | LERF, CLIP-Fields, CLIP-NeRF, 3D-LLM | 3D data scarcity, NeRF 计算成本高 |
| Affordance | 物体功能性理解 | Affordance Diffusion, VRB | 依赖人类交互视频, 难以泛化到 novel objects |

### 3.5 Predictive Models / World Models

论文将 video prediction 和 dynamics model 归为 predictive models, 这是一个被低估但重要的方向:

- **GAIA-1**: 从 4700 小时驾驶数据学习 world model, 预测未来 video frames conditioned on action
- **Diffuser**: 用 diffusion model 做 trajectory planning, 将 planning 转化为 denoising 过程
- **UniPi**: 从 text-conditioned video generation 中提取 universal policy
- **Video Language Planning**: 用 VLM 生成 plan, 再用 video model 验证 feasibility

这个方向与 DreamerV3 的 world model 思想一致, 但用 foundation model 替代了 task-specific RSSM。

---

## 4. Experiments

作为 survey, 论文没有自己的实验, 但系统整理了各领域的 benchmark 和结果:

### 4.1 Robot Transformer 性能对比 (Table II)

| Model | Backbone | Size | Task | Inference | Hardware |
|-------|----------|------|------|-----------|----------|
| RT-1 | EfficientNet + TokenLearner + decoder | 35M | Real-world manipulation | 3Hz | -- |
| RT-2 | PaLI-X | 55B | Real-world manipulation | 1-3Hz | Multi-TPU cloud |
| RT-X | UL2 | 55B | Real-world manipulation | 1-3Hz | Multi-TPU cloud |
| RoboCat | Decoder-only transformer | 1.18B | Manipulation | 10-20Hz | 16x16 TPU v3 slice |
| Gato | Decoder-only transformer | 1.2B | 604 tasks (generalist) | 20Hz | -- |
| ViNT | EfficientNet + decoder | 31M | Navigation | 4Hz | Various GPUs |
| VPT | ResNet 62 + attention | 0.5B | Minecraft agent | 20Hz | 720 V100 GPUs, 9 days |
| PACT | Decoder-only | 12M | Forward dynamics + action prediction | 10Hz (edge) / 50Hz | Xavier NX / V100 |

**关键观察**: 模型规模和推理速度存在明显 trade-off。RT-2 (55B) 只能跑 1-3Hz, 远低于 real-time control 所需的 50-100Hz。轻量模型如 PACT (12M) 可以在 edge device 上跑 50Hz。

### 4.2 Navigation: Classical vs End-to-End vs Modular

论文引用 semantic navigation 的对比研究发现:
- **Modular 方案** (classical + FM perception) 在 real-world 表现最好
- **End-to-end 方案** 面临 sim-to-real gap
- 启示: FM 更适合作为 modular system 的组件, 而非完全替代 classical stack

### 4.3 Manipulation 领域的关键发现

- **CLIPort -> PerAct**: 从 2D 到 3D, 从单视角到多视角, 任务成功率显著提升
- **RT-1 -> RT-2**: VLM co-fine-tune 使 generalization 大幅提升 (新物体, 新指令)
- **RT-X**: 跨 embodiment 训练带来 positive transfer, 证明 cross-embodiment foundation model 是可行的

---

## 5. Related Work Analysis

### 5.1 与同期 Survey 的定位对比

| Survey | 侧重 | 与本文的互补关系 |
|--------|------|----------------|
| **Firoozi et al. (本文)** | FM 在 robotics 全栈的应用, 强调安全和风险 | -- |
| Bommasani et al. (2021) "On the Opportunities and Risks of FM" | FM 的宏观影响, robotics 只是子话题之一 | 本文是 robotics 领域的 deep dive |
| Yang et al. (2023) "FM for Decision Making" | 聚焦 decision-making, 包括非 robotics 场景 | 本文覆盖 perception + embodied AI |
| 同期 arxiv survey (Xuan et al.) | 强调 algorithms 和 architectures 的对比 | 本文更强调 challenges 和 safety/risk |

### 5.2 领域全景: FM 重塑 Robotics 的三个浪潮

从论文和 GitHub repo 的引用可以梳理出三个浪潮:

| 浪潮 | 时间 | 核心思想 | 代表工作 |
|------|------|---------|---------|
| **第一浪潮: Representation Transfer** (2020-2022) | 2020-2022 | 用 pretrained visual encoder 做 robot representation | R3M, MVP, CLIPort |
| **第二浪潮: LLM as Planner** (2022-2023) | 2022-2023 | LLM 直接做 task planning / code generation | SayCan, Code-as-Policies, Voyager |
| **第三浪潮: VLA (Vision-Language-Action)** (2023-) | 2023+ | 端到端 VLM -> action, 统一 perception-decision-control | RT-2, PaLM-E, Gato |

论文发表时 (2023.12) 正处于第二到第三浪潮的过渡期。从今天 (2026) 的视角看, 第三浪潮已经产生了 GR00T N1, pi_0, Helix 等实际部署级的系统。

---

## 6. Limitations & Future Directions

### 6.1 论文提出的 8 大开放挑战

| 挑战 | 问题描述 | 当前状态 (2026 视角) |
|------|---------|---------------------|
| **VI-A: Data Scarcity** | Robot data 远少于 internet data | 部分缓解: GR00T N1 的 data pyramid, RT-X 的 cross-institution 数据集, DexMimicGen 等 sim 扩增 |
| **VI-B: Real-Time Performance** | FM 推理延迟过高 (秒级) | 显著改善: GR00T N1 用 4-step flow matching 达到 63.9ms/chunk; 模型蒸馏和量化持续推进 |
| **VI-C: Multimodal Representation** | 不同模态的 tokenization 和对齐 | 仍是开放问题: 3D point cloud 与 text 的对齐仍不成熟 |
| **VI-D: Uncertainty Quantification** | LLM hallucination + distribution shift | 有进展: KnowNo (conformal prediction) 提供了理论框架, 但实际部署仍缺乏标准 |
| **VI-E: Safety Evaluation** | 全生命周期安全测试 | 仍是短板: red-teaming for robotics 尚无成熟方法论 |
| **VI-F: Plug-and-Play vs Build New** | 直接用现成 FM 还是训练 robotics-specific FM | 趋势明确: 高层用 plug-and-play (LLM planning), 底层 build new (VLA for control) |
| **VI-G: High Variability** | 硬件/环境/任务差异大 | Cross-embodiment training (RT-X, GR00T N1) 正在缓解 |
| **VI-H: Benchmarking** | 缺乏统一 benchmark 和 reproducibility | 部分改善: LIBERO, RoboCasa, BEHAVIOR-1K 等 benchmark 涌现 |

### 6.2 Data Scarcity 的 6 种应对策略

论文详细分析了 6 种策略, 这是全文最有实践价值的部分:

| 策略 | 方法 | 代表工作 | 优劣分析 |
|------|------|---------|---------|
| Unstructured Play Data | 用 teleoperated play data 替代 expert demo | Play-LMP, MimicPlay | 便宜但标注少, 适合 goal-conditioned 方法 |
| Inpainting Augmentation | Text-to-image 模型编辑训练图像 | ROSIE, GenAug, CACTI | 可增加视觉多样性, 但物理合理性难保证 |
| 3D Data Generation | 利用 2D FM 生成 3D data | FeatureNeRF, 3D-LLM | 3D data + language description 极度稀缺 |
| High-Fidelity Simulation | 仿真生成大量数据 | TartanAir, DexMimicGen | Sim-to-real gap 仍是核心问题 |
| VLM Augmentation | VLM 自动标注/重标注数据 | DIAL | 低成本但可能引入 label noise |
| Human Video | 从人类活动视频学习 skill prior | VIP, LIV, R3M | 跨 embodiment gap (人手 vs 机械手) |

### 6.3 论文未充分讨论但重要的方向

1. **Scaling Law for Robotics**: 论文仅在 GitHub repo 中列了一条引用 (Neural Scaling Laws for Embodied AI), 但未深入分析 robotics 的 scaling behavior 是否遵循 Chinchilla-like law
2. **Sim-to-Real 的根本性**: 论文在 Benchmarking 部分提到 sim-to-real gap, 但未将其作为独立挑战。实际上这可能是 robotics FM 最核心的瓶颈
3. **Contact-Rich Manipulation**: 论文几乎未讨论 dexterous manipulation 和 contact physics, 而这是 FM 最难解决的物理交互问题
4. **Locomotion-Manipulation Co-optimization**: 论文将 navigation 和 manipulation 分开讨论, 未涉及 loco-manipulation 的统一控制问题

---

## 7. Paper vs Code Discrepancies

N/A (Survey 论文无代码)

GitHub repo ([Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)) 是配套的 paper list, 组织结构与论文一致。值得注意的是 repo 持续更新, 已包含论文未覆盖的新工作 (如 ICRT, MineDreamer, MAGIC-VFM 等)。

---

## 8. Cross-Paper Comparison

### 8.1 与 GR00T N1 (NVIDIA, 2025) 的关系

本 survey 是 GR00T N1 的 **理论前传**。Survey 提出的分类框架精确预测了 GR00T N1 的设计选择:

| Survey 的预测/分析 | GR00T N1 的实际做法 |
|---------------------|---------------------|
| "Plug-and-Play vs Build New FM" 的 trade-off | GR00T N1 选择 build new: VLM fine-tune + DiT from scratch, 而非直接 plug-and-play GPT-4 |
| Data scarcity 需要 play data + sim + augmentation | Data pyramid: human video (play data) + DexMimicGen (sim) + WAN2.1 neural trajectory (augmentation) |
| Real-time performance 是瓶颈 | 4-step flow matching, 63.9ms/chunk, 15.6Hz -- 比 RT-2 的 1-3Hz 快一个数量级 |
| Cross-embodiment 需要统一 action space | Embodiment-specific projectors: 共享 VLM+DiT 权重, 每个 embodiment 独立 encoder/decoder |
| Robot Transformers 是未来方向 | GR00T N1 的 DiT (Diffusion Transformer) 是 survey 中 Robot Transformer 方向的自然延伸 |
| UQ 需要 conformal prediction | GR00T N1 未解决 UQ -- survey 的这个预测仍是 open problem |

### 8.2 与 Diffusion Policy (Chi et al., 2024) 的关系

Survey 讨论的 language-conditioned imitation learning 和 robot transformers 是 Diffusion Policy 的上下文:

| 维度 | Survey 的分析 | Diffusion Policy 的定位 |
|------|-------------|----------------------|
| Action generation | 论文讨论了 discretized token (RT-1/2) vs continuous action | DP 选择 continuous action + DDPM denoising |
| Language conditioning | 论文强调 language-conditioned IL 的重要性 | DP 不支持 language -- 是 task-specific 方法, 非 FM |
| Data efficiency | 论文指出 data scarcity 是核心挑战 | DP 需要中等量 demo (100-200), 无 pre-training |
| Real-time | 论文指出推理延迟瓶颈 | DP 100-step denoising 较慢, GR00T N1 用 flow matching 将其压缩到 4 步 |

Diffusion Policy 在 survey 的框架中属于 "Robot Policy Learning" 类别, 但由于发表时间 (2024 IJRR), survey 未能收录。DP 代表了 survey 未预见的一个趋势: **diffusion model 不仅可以做 data augmentation (survey 的分析), 还可以直接做 action generation**。

### 8.3 与 DreamerV3 (Hafner et al., 2023) 的关系

Survey 在 Predictive Models 部分讨论了 world model, 与 DreamerV3 的思路有交叉:

| 维度 | Survey 的 Predictive Models | DreamerV3 |
|------|---------------------------|-----------|
| 范式 | 用 FM (video model) 做 prediction | 用 task-specific RSSM 做 dynamics prediction |
| 数据 | Internet-scale video | 单环境 rollout |
| 泛化 | Zero-shot transfer (理论上) | 需要重新训练 |
| Physics | 隐式 (视频像素级) | 显式 latent dynamics |

Survey 的分析暗示: **未来 world model 可能从 DreamerV3 的 task-specific RSSM 演变为 FM-based universal world model** (如 GAIA-1, Cosmos)。GR00T N1 的 NVIDIA 路线图中 Cosmos World Model 正是这个方向的实践。

### 8.4 "Robotics 重走 CV/LLM 路线" 的分析

这是 survey 的隐含主线, 值得独立分析:

| 阶段 | CV/NLP 的路径 | Robotics 的对应 | 时间差 |
|------|-------------|---------------|--------|
| Feature Engineering | SIFT, HOG, TF-IDF | 手工设计 reward, state representation | -- |
| Supervised Pre-training | ImageNet pre-training, Word2Vec | R3M, MVP (visual pre-training for robotics) | ~8 年 (2012 -> 2020) |
| Self-supervised Pre-training | BERT (MLM), MAE, DINO | VPT (Minecraft), PACT (self-supervised robot data) | ~4 年 (2018 -> 2022) |
| Foundation Model | GPT-3, CLIP, DALL-E | RT-2, PaLM-E, GR00T N1 | ~3 年 (2020 -> 2023) |
| Scaling to Generalist | GPT-4, Gemini | GR00T N1/N1.6, pi_0, Helix | ~2 年 (2023 -> 2025) |

**关键差异**: Robotics 不太可能完全复制 CV/NLP 的 scaling 路线, 原因在于:

1. **数据瓶颈无法靠 Internet 解决**: NLP 的 Common Crawl 有 PB 级文本, robotics 没有等价的 "Internet of Actions"。GR00T N1 的 data pyramid 是一种绕路方案 (video + sim + real), 但成本仍然极高。

2. **物理世界不可微分**: LLM 的 token prediction 是纯软件优化, robotics 需要与物理环境交互, sim-to-real gap 是根本性的。

3. **安全要求截然不同**: LLM 输出错误文本最多造成误导, robot 执行错误动作可能造成物理伤害。这意味着 robotics FM 的部署标准必须远高于 NLP FM。

4. **Embodiment 多样性远超 "设备差异"**: 不同 robot platform 的差异 (自由度、传感器、控制模式) 远大于不同 GPU/手机的差异。Cross-embodiment 统一本身就是一个 open research problem。

### 8.5 对 Manipulation 领域的特别分析

Survey 对 manipulation 的覆盖集中在以下几个方面, 与本 paper library 中的其他工作密切相关:

| 方向 | Survey 覆盖的工作 | 后续发展 (2024-2026) |
|------|------------------|---------------------|
| Language-conditioned grasping | CLIPort, PerAct, CACTI | GR00T N1 实现了真正的 language-conditioned manipulation |
| Open-vocabulary manipulation | VIMA, MOO, StructDiffusion | 仍是 active area, 但 VLA 方案开始主导 |
| Affordance for manipulation | Affordance Diffusion, VRB | VRB 的 affordance -> 多种 robot learning 范式 的 pipeline 被验证有效 |
| Dexterous manipulation | 几乎未覆盖 | DexMimicGen, ManipTrans, BiDexHD 等 (本 library 中有大量工作) |

Survey 的一个显著缺失是 **dexterous manipulation with foundation models** -- 论文几乎所有 manipulation 示例都是 gripper-based pick-and-place, 未涉及 multi-finger dexterous grasping。这在 2023 年可以理解 (FM + dexterous manipulation 的交叉研究刚起步), 但从 2026 年回看, 这已成为一个快速发展的方向。

---

## Summary

Firoozi et al. 的 survey 在 2023 年底提供了 foundation model 在 robotics 应用的全景式梳理。其核心贡献是:

1. **系统化分类**: 将 FM 在 robotics 中的应用映射到 autonomy stack (perception -> decision -> control), 并区分 plug-and-play vs fine-tune vs build-new 三种集成范式
2. **挑战识别**: 提出 data scarcity, real-time performance, uncertainty quantification, safety evaluation, high variability 五大核心挑战, 这些挑战在 2026 年仍然高度相关
3. **Robotics-specific 视角**: 相比泛 AI survey, 本文深入讨论了 sim-to-real, cross-embodiment, contact physics 等 robotics 独有问题

**局限**: Survey 写于 VLA (Vision-Language-Action) 范式刚起步之际, 未能充分预见 GR00T N1 / pi_0 这类 end-to-end VLA 系统的快速崛起。对 dexterous manipulation, locomotion-manipulation 统一控制、scaling law for robotics 等方向覆盖不足。

**历史定位**: 这篇 survey 是 "foundation model for robotics" 领域的奠基文献之一, 其分类框架和挑战分析为后续工作 (包括 GR00T N1 的 data pyramid 设计) 提供了思考框架。
