# A Survey of Robotic Language Grounding: Tradeoffs between Symbols and Embeddings

**Paper**: Vanya Cohen, Jason Xinyu Liu, Raymond Mooney, Stefanie Tellex, David Watkins (UT Austin / Brown / The AI Institute), 2024
**Type**: Survey paper

---

## 1. Core Problem

语言 grounding 到机器人行为的核心挑战在于: 如何将人类自然语言指令映射到机器人可执行的行为。这个问题由 Harnad (1990) 定义为 **symbol grounding problem** -- 将符号系统中的 symbol 映射到物理世界的 sensorimotor substrates。

LLM 的出现使问题变得更复杂也更有机遇:
- LLM 在语义层面表征概念, 不依赖显式的高阶符号, 推动了 end-to-end 方法的兴起
- 但许多最成功的系统实际上将 LLM 与 formal symbolic representations 结合使用 (如 Code as Policies 生成带预定义 API 的 Python, SayCan 将语言映射到离散 skill)

本文的核心观察: 现有方法可以沿一个 **spectrum** 分布, 两极分别为:
1. **Formal/Symbolic 端**: 语言 -> 人工定义的 formal representation (逻辑、PDDL、代码、预定义技能)
2. **Embedding/End-to-end 端**: 语言 -> 高维向量空间 -> 直接低层机器人策略

两极各有 tradeoff, 大多数实际系统混合使用两者。

---

## 2. Method Overview: Symbols vs Embeddings 的 Spectrum

### 2.1 Spectrum 总览

从 formal 到 end-to-end, 论文将方法排列如下:

```
[More Formal / Structured]                              [More End-to-End / Flexible]
    |                                                            |
  Logics --> PDDL --> Code --> Predefined Skills --> Image/Lang Subgoals --> EE/Joint Goals
  (LTL,FOL)                                                      (RT-1/2, VIMA, Octo)
```

### 2.2 两极特征对比

| 维度 | Formal/Symbolic | Embedding/End-to-End |
|------|----------------|---------------------|
| 表示形式 | LTL, FOL, PDDL, Python code, 离散 skill | 高维连续向量, joint states, EE poses |
| 数据需求 | 较少 (结构化 bias 减小搜索空间) | 较多 (参数更多, 约束更少) |
| 泛化方式 | 通过 formal representation 跨域迁移 | 通过大规模数据 + 大模型泛化 |
| 可解释性 | 天然可解释, 支持形式化验证 | 难以解释, 缺乏安全保证路径 |
| 安全保证 | 支持 model checking, correct-by-construction | 目前无清晰路径满足 ISO 61508 |
| 灵活性 | 受限于 formal language 的表达能力 | 理论上不受限, 但需要足够数据 |
| 执行方式 | planner/code interpreter + low-level controller | 直接输出 low-level control |

---

## 3. Key Designs: 分类维度和代表性方法

### 3.1 Formal 端: Language -> Formal Representation

#### 3.1.1 Logics (最 formal 的一端)

Goal-based representation, 描述世界应达到的状态而非具体动作。

| 方法 | 逻辑类型 | 核心思路 | 任务域 |
|------|---------|---------|--------|
| Lang2LTL (Liu et al., 2023) | LTL | 模块化系统, LLM 将导航指令 grounding 到 LTL + semantic map | 室内外导航 |
| Pan et al. (2023) | LTL | LLM paraphrase 生成多样化训练数据 | 导航 |
| LEFT (Hsu et al., 2023) | FOL | LLM 翻译语言为 FOL, differentiable executor 执行 | 2D/3D 场景理解 |
| AutoTAMP (Chen et al., 2024) | STL | LLM 生成 STL 公式 + STL planner 生成轨迹 | 2D 带几何/时序约束 |

优势: 可以形式化保证任务满足性, 自然表示 temporal 任务 (如 "avoid the red room")。
局限: 难以覆盖所有自然语言表达; 不同逻辑间转换代价不一。

#### 3.1.2 PDDL

Goal-based, 但也编码了 actions 和 effects, 可兼做 imperative。

| 方法 | 核心思路 | 关键发现 |
|------|---------|---------|
| Xie et al. (2023) | LLM 翻译自然语言目标为 PDDL goal | LLM 可处理明确目标, 但数值/空间推理困难 |
| Collins et al. (2022) | LLM -> PDDL goal -> symbolic planner | 优于直接用 LLM 做 planner |
| Guan et al. (2023), Liu et al. (2023a) | LLM -> 完整 PDDL problem definition | 提供正确性保证 |
| Silver et al. (2024) | LLM 合成 Python program 作为 generalized planner | 自动检测规划错误并 re-prompt |

关键发现: LLM 单独做 planner 的成功率很低 (Valmeekam et al., 2023), 但配合 symbolic planner 可显著提升。

#### 3.1.3 Code

灵活性最高的 formal representation, 可 goal-based 也可 action-based。

| 方法 | 核心思路 | 任务域 |
|------|---------|--------|
| **Code as Policies** (Liang et al., 2022) | LLM 直接生成可执行 Python, 调用预定义 perception/control API | manipulation, mobile manipulation |
| ProgPrompt (Singh et al., 2023) | 类似 + assertion 语句做错误恢复 | 仿真 |
| Varley et al. (2024) | 模块化双臂系统, LLM -> API calls -> VLM + control | tabletop bimanual manipulation |
| Socratic Models (Zeng et al., 2023) | LLM 代码生成 + 预训练感知模块 | pick-and-place |
| Voyager (Wang et al., 2023) | LLM 持续探索, 构建可执行代码 skill library | Minecraft |

**对 manipulation 特别重要**: Code as Policies 和 Varley et al. 的模块化方法在 manipulation 中展现了代码方法的优势 -- 模块化提供 safety 可审计性、failure 可解释性、以及逐模块改进的能力。

#### 3.1.4 Predefined Skills

Action-based representation, LLM 作为 planner 将语言映射到技能序列。

| 方法 | 核心思路 | 任务域 |
|------|---------|--------|
| **SayCan** (Ahn et al., 2022) | LLM 排序预训练 skills (verb phrases), 基于 value function 评估可行性 | mobile manipulation (真机) |
| Huang et al. (2022a) | LLM 分解高层任务为动作描述, 用 sentence similarity 匹配可用动作 | 仿真 |
| CAPE (Raman et al., 2024) | LLM planner + precondition 检查 + corrective feedback | - |
| Inner Monologue (Huang et al., 2022b) | LLM + 感知模型提供语言反馈, 检测 skill 执行成功 | - |

论文的重要观察: SayCan 虽然常被视为 end-to-end 方法, 但它的 predefined skill set 实质上是一种 symbolic representation。**skill 的粒度和覆盖范围对系统成功至关重要**, 但在论文中常被忽略。

### 3.2 End-to-End 端: Language -> High-Dimensional Vectors -> Actions

#### 3.2.1 Image and Language Subgoals

| 方法 | 核心思路 | 任务域 |
|------|---------|--------|
| Black et al. (2023) | text-guided image editing 生成视觉 subgoal, low-level policy 执行 | pick-and-place |
| VLP (Du et al., 2023) | 语言+图像 -> language/image subgoal 序列 -> tree search | tabletop arrangement |
| UniSim (Yang et al., 2023) | generative video model 预测 action 结果 | autonomous driving |

#### 3.2.2 End-Effector and Joint-State Goals (最 end-to-end 的一端)

**这是对 manipulation 任务最直接相关的类别。**

| 方法 | 模型规模 | Action Space | 数据来源 | 关键特征 |
|------|---------|-------------|---------|---------|
| **RT-1** (Brohan et al., 2023a) | - | 离散化 EE pose + base + gripper | 大规模真机示范 | 第一个大规模 language-conditioned manipulation |
| **RT-2** (Brohan et al., 2023b) | - | 同 RT-1 | RT-1 数据 + web 数据 | VLM -> VLA, web knowledge transfer |
| **RT-X / Open-X Embodiment** (O'Neill et al., 2024) | 35M-55B | EE positions | 22 个平台, 多机构 | 跨 embodiment positive transfer |
| **Octo** (Octo Model Team, 2023) | - | EE + joint-state | Open-X Embodiment | 同时支持两种 action space |
| **PaLM-E** (Driess et al., 2023) | 562B | 有限动作词汇表 | multimodal data | 多模态推理 + manipulation |
| **VIMA** (Jiang et al., 2023) | - | EE controls (low-level) | 仿真自动生成 | multimodal prompt, tabletop tasks |
| **ALOHA** (Zhao et al., 2023) | - | joint-state | 低成本硬件采集 | 解决数据采集瓶颈 |
| GATO (Reed et al., 2022) | 1.2B | joint states | 多任务 | **未展示** positive transfer |

关键对比:
- RT-X/Open-X 用 EE pose + inverse kinematics, GATO 直接用 joint states
- Open-X 展示了跨 embodiment 的 positive transfer, GATO 未能展示
- VIMA 通过仿真数据大量生成绕过真机数据不足, 但学到的策略 platform-specific

---

## 4. Experiments: 各方法的性能对比

论文作为 survey 未提供统一的 benchmark 对比, 但提供了以下定性和定量观察:

### 4.1 Formal vs End-to-End 的任务表现

| 对比维度 | Formal 方法表现 | End-to-End 方法表现 |
|----------|---------------|-------------------|
| 长 horizon 任务 | 优势 -- LTL/PDDL 自然表示时序约束 | 劣势 -- chaining policies 困难 |
| 数据效率 | 优势 -- few-shot / zero-shot 可用 | 劣势 -- 需大量数据 |
| 跨域泛化 | 优势 -- formal rep 可 port 到新 domain | 需重新收集数据 |
| 灵活性/表达力 | 劣势 -- 受限于预定义的 formal language | 优势 -- 更少约束, 更高上限 |
| 实时性 | 通常满足 | 控制频率 1-3 Hz, 无法响应快速语言修正 |

### 4.2 LLM 作为 Planner 的关键发现

- LLM 单独做 planner: 多个 domain 上成功率低 (Valmeekam et al., 2023)
- LLM + symbolic planner (如 PDDL): 显著优于单独使用任一方法
- AutoTAMP (LLM -> STL -> planner): 在有几何/时序约束的任务上优于纯 LLM planner

### 4.3 控制频率问题

论文指出一个被忽视的实际问题: 当前方法控制频率在 1-3 Hz, 无法支持 "move left... okay stop" 这类需要实时语言反馈的交互场景。

---

## 5. Related Work Analysis: 领域发展脉络

### 5.1 历史脉络

```
Pre-LLM era:
  Tellex et al. (2020) -- 传统 robot-language grounding (概率图模型, 语义解析)
                |
LLM 时代:
  SayCan (2022) -- LLM + predefined skills (开创性工作)
  Code as Policies (2022) -- LLM 生成可执行代码
  RT-1 (2023) -- 大规模 language-conditioned 真机训练
                |
  RT-2 (2023) -- VLM -> VLA, web knowledge transfer
  PaLM-E (2023) -- 超大规模多模态模型
  VIMA (2023) -- multimodal prompt + 仿真数据
                |
  Open-X Embodiment (2024) -- 跨平台数据集 + 跨 embodiment transfer
  Octo (2024) -- 开源 generalist robot policy
```

### 5.2 关键转折点

1. **LLM 使 formal 方法复兴**: LLM 的 few-shot 能力使得无需大量平行语料即可将自然语言翻译为 formal representation (LTL, PDDL, code)
2. **Scaling 推动 end-to-end 方法**: 从 RT-1 到 RT-X, 数据规模和模型规模的增长持续提升泛化能力
3. **混合方法成为主流**: 即使是 "end-to-end" 方法 (如 SayCan, RT-*) 也大量依赖 symbolic 中间表示

### 5.3 与其他 survey 的关系

| Survey | 焦点 | 与本文互补之处 |
|--------|------|---------------|
| Tellex et al. (2020) | Pre-LLM 时代的 robot-language grounding | 历史基础, 本文是其 LLM 时代的续篇 |
| Zhang et al. (2023) | LLM 在 HRI 中的广泛应用 | 更广 (含 QA, social), 本文更深 (聚焦 command understanding) |
| Zeng et al. (2023b) | LLM 应用于机器人 (广泛) | 不涉及 formal-to-embedding spectrum 分析 |
| Wang et al. (2024) | LLM 在机器人中的应用 | 更广的任务范围, 但不做 spectrum 分析 |

---

## 6. Limitations & Future Directions

### 6.1 论文识别的局限性

1. **Formal 方法的表达力瓶颈**: 不存在一种 formal language 能精确捕获所有自然语言的语义; 不同逻辑 (LTL, FOL, STL) 各有侧重但无法统一
2. **End-to-end 方法的数据饥渴**: 机器人多模态数据 (视频 + 传感器 + 关节状态) 维度远高于纯文本, 需要的数据量可能远超训练 LLM 的规模
3. **控制频率限制**: 1-3 Hz 的推理速度不足以支持实时交互
4. **自然语言本身的局限**: 英语对空间关系的描述 inherently ambiguous ("to the left" 有无穷多位置); 人类语言为生物体进化, 不一定适合描述机器人行为
5. **安全与可解释性的鸿沟**: End-to-end 方法目前无法满足 ISO 61508 要求的数学证明级安全保证
6. **跨 embodiment 泛化**: End-effector/joint-state 层面学到的策略通常 platform-specific

### 6.2 论文提出的未来方向

1. **Best of both worlds**: 结合 formal 的可解释性/安全保证与 end-to-end 的灵活性/泛化能力
2. **自动学习可扩展的 symbol set**: 如 Konidaris et al. (2018) 的方向 -- 机器人自主学习 low-level skill 对应的 symbol
3. **多模态整合**: 文本、音频、RGB-D、视频、关节轨迹的统一表示
4. **扩大物理能力**: 当前系统受限于机器人硬件本身的能力, 未来需要更通用的硬件平台
5. **形式化方法与 deep learning 的交叉**: 如 Yang et al. (2024) 结合 LLM + LTL 实现安全且灵活的指令跟随
6. **多语言支持**: 当前几乎所有工作聚焦英语

### 6.3 个人推断的方向

- **Hierarchical abstraction at different frame rates**: 论文提到但未展开 -- 语言可以谈论机器人系统的任何层级, 需要在不同时间尺度上构建层级化的 abstraction
- **Bitter lesson 的适用性**: 论文承认大规模数据+大模型最终可能胜出, 但前提是持续增长的数据供给, 这在机器人领域不像 NLP/CV 那样容易获取

---

## 7. Paper vs Code Discrepancies

N/A -- 本文为 survey, 不涉及具体代码实现。

---

## 8. Cross-Paper Comparison

### 8.1 与 Foundation Models in Robotics (Firoozi et al., 2024) 的互补关系

| 维度 | 本文 (Language Grounding Survey) | Foundation Models in Robotics |
|------|--------------------------------|------------------------------|
| 焦点 | Language -> Robot Behavior 的 grounding spectrum | Foundation model 在机器人中的广泛应用 |
| 分析框架 | Symbols vs Embeddings spectrum | 按应用分类 (planning, navigation, manipulation 等) |
| 深度 | 深入分析 command understanding 的表示形式和 tradeoff | 广覆盖, 列举大量工作但每篇分析较浅 |
| 安全/可解释性 | 重点讨论, 作为 formal 方法的核心优势 | 提及但不深入 |
| 互补性 | 提供理解"为什么某种方法用某种表示"的框架 | 提供"什么 foundation model 被用在什么任务上"的全景图 |

### 8.2 与本 library 中其他论文的关系

| 本 library 论文 | 在本 survey 中的位置 | 关联 |
|----------------|---------------------|------|
| Diffusion Policy (2024) | 未直接讨论, 但属于 end-to-end 端的 joint-state/EE goal 方法 | Diffusion Policy 输出 EE/joint action, 不涉及语言 conditioning, 但可作为语言 grounding 系统的 low-level policy |
| GR00T N1 (2025) | 后于本 survey, 但属于 RT-X/Octo 同一脉络 | GR00T N1 的 VLA 架构 = VLM (language understanding) + DiT (action generation), 是 end-to-end 端的进化 |
| Decision Transformer (2021) | 未讨论, 但提供了 sequence modeling 思路 | 将 RL 转化为 sequence prediction, 影响了后续 VLA 的 token-based 设计 |
| DreamerV3 (2023) | 未讨论, 但 world model 与 UniSim/GAIA-1 思路相近 | World model 方法可以作为 language grounding 的 imagination module |

### 8.3 对 Manipulation 任务的综合视角

本 survey 对 manipulation 中 language grounding 的关键洞察:

1. **Code 方法最适合 manipulation 的模块化需求**: Varley et al. (2024) 展示了 LLM -> API call -> VLM + control 的模块化架构在 bimanual manipulation 中的优势 -- safety 可审计、failure 可追溯
2. **Predefined skills 是当前 manipulation 的主流中间层**: SayCan 的 verb phrase skills 虽粗糙, 但 skill 粒度和覆盖范围决定系统上限
3. **End-to-end 方法 (RT-*, VIMA) 在 manipulation 数据充足时潜力更大**: 但 platform-specific 问题严重
4. **数据采集是瓶颈**: ALOHA 通过低成本硬件、VIMA 通过仿真分别试图解决这个问题, 但真机+多样场景的数据仍然稀缺
