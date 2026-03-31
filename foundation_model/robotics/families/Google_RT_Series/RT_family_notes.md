# Google RT Series -- VLA 的起源故事

> **目的**: 理解 VLA 是怎么从 "LLM 做规划" 一步步演化到 "VLM 直接出动作" 的。
> RT 系列是整个 robotics FM 的源头 -- PI 和 GR00T 都是它的后续。

---

## 1. 背景: 为什么是 Google

Google DeepMind Robotics 在 2022-2023 年同时拥有三样东西: 最强的语言模型、最强的视觉编码器、以及部署在真实办公室里的数百台机器人。这个组合让他们成为第一个把大模型和真实机器人对接的团队。

更关键的是**人**: Karol Hausman, Brian Ichter, Pete Florence -- 后来创办 PI 的核心人物 -- 当时全在这个团队里。RT 系列的思想密度和后续影响力, 本质上是这批人的产出。

---

## 2. 演进脉络: 三个阶段

### Phase 1: LLM-as-Planner (2022) -- "LLM 不直接控制, 只做高层规划"

**核心问题**: LLM 懂语言、懂常识, 但它不知道机器人能做什么。怎么连接?

**三次尝试, 三种回答**:

- **SayCan**: LLM 做任务分解 ("拿苹果 = 走过去 + 抓"), 再用视觉模型判断哪些步骤在当前场景中可执行, 最后调用预写的低层技能。第一次证明 LLM 可以做 robot task planning。
- **Code as Policies**: 不输出步骤列表, 而是直接让 LLM 写控制代码。比 SayCan 更灵活 (代码能组合任意行为), 但仍然依赖预定义的 API 函数库。
- **Inner Monologue**: 执行失败后, 用视觉模型描述当前状态, LLM 据此重新规划。第一次做 LLM-robot 的闭环反馈, 而非一次性规划。

**Phase 1 的核心洞察**:

LLM 能做规划, 但规划和执行是分离的 -- 低层执行器要么手写, 要么依赖预定义接口, 不可扩展。能不能让一个模型端到端搞定?

### Phase 2: Robot Transformer (2022-2023) -- "不做规划, 直接出动作"

**核心问题**: 能否训练一个 Transformer 直接从图像到动作, 跳过 "先理解再规划再执行" 的流程?

**三次尝试, 从专用到通用**:

- **RT-1**: 第一个大规模 robot Transformer。纯端到端: 图像输入, 离散动作 token 输出。证明 Transformer 在机器人领域同样有效, 不只是 NLP 的专利。但它只用了自己采集的数据, 没有继承互联网上的视觉-语言知识。
- **RT-2 (第一个 VLA)**: 核心思想极其简单 -- 既然 VLM 已经在互联网数据上学会了"苹果是什么", 为什么不直接让它输出动作? 将 VLM 与机器人数据混合微调, 动作被编码为离散 token 混在语言输出里。第一次证明 **web 知识可以迁移到 robot** -- 训练时没见过的物体也能抓。
- **PaLM-E**: 把这个思路推到极致 -- 用当时最大的语言模型, 把图像、语言、机器人状态全部当 token 输入。证明 VLM 越大, robot 能力越强 (scaling 在 robot 上也 work), 但模型太大完全无法部署。

**Phase 2 的核心洞察**:

VLM 可以做 robot policy, 且能继承 web 知识 -- 这就是 VLA 的核心假设。但当时的 VLM 太大太慢, 离散 token 精度也不够。需要: 更小的 VLM + 更高效的动作生成方式。

### Phase 3: 数据与规模化 (2023-2024) -- "模型架构够了, 下一步是数据和部署"

**核心问题**: 单一实验室的数据永远不够, 怎么让机器人数据像 Common Crawl 一样规模化?

- **Open X-Embodiment**: 汇聚 22 个机器人平台的数据, 建立第一个大规模跨 embodiment 数据集。证明不同机器人的数据混合训练, 比只用自己的数据更好。这是 robotics 的 "Common Crawl"。
- **AutoRT**: 用 VLM 自动给机器人分配任务, 机器人自主探索, VLM 评估成功/失败, 形成数据飞轮。第一次做到不靠人类遥操作的大规模数据采集。
- **RT-H**: 在高层 VLM 和低层控制器之间插入自然语言作为中间表示 ("move hand left 5cm")。语言接口既可解释又可纠正, 和后来 PI 的 Hi Robot 思路一脉相承。

**Phase 3 的核心洞察**:

单一团队的数据采集不可扩展, 跨平台数据共享和自主采集才是出路。Open X 定义了数据标准, AutoRT 定义了采集范式。

---

## 3. 关键设计决策演进

| 问题 | SayCan (2022) | RT-1 (2022) | RT-2 (2023) | pi_0 (2024, PI) |
|------|-------------|------------|------------|----------------|
| 大模型角色 | 做高层规划 | 不用大模型 | VLM 做端到端骨干 | VLM 做端到端骨干 |
| 动作生成方式 | 调用预写 skill | 离散 token | 离散 token | **连续 (Flow Matching)** |
| web 知识利用 | 通过 LLM 语言理解 | 不利用 | **VLM 混合微调迁移** | VLM 预训练迁移 |
| 模型策略 | 大模型规划 + 小 skill | 专用小模型 | 通用大模型 | **紧凑通用模型** |
| 新物体泛化 | 靠 LLM 语言理解 | 不行 | **可以** (知识迁移) | 可以 |

**核心演化**: LLM 做规划 (SayCan) --> Transformer 做 policy (RT-1) --> VLM 做 policy + 继承 web 知识 (RT-2) --> 更小更快的 VLM + 连续动作生成 (pi_0)

---

## 4. RT Series --> PI --> GR00T 的传承关系

这是 RT 系列最深远的影响 -- 它不是一个终结的项目, 而是整个 robotics FM 的源流。

### 4.1 人的传承: RT --> PI

2024 年, RT 系列的核心作者集体离开 Google 创办 PI。这不是巧合, 而是 RT 系列的逻辑终点: Google 内部的资源条件 (大 VLM + 真机) 造就了 VLA 思想, 但 Google 的体制不适合把它产品化。

PI 的 pi_0 本质是 RT-2 的精神续作, 解决了 RT-2 留下的两个核心问题:
- **模型太大**: 用紧凑 VLM 替代庞大 VLM, 推理速度提升一个数量级
- **离散 token 精度不够**: 用 Flow Matching 生成连续动作, 精度质变

同样的思想 (VLM --> robot action), 更好的工程实现。

### 4.2 数据的传承: Open X --> 所有 VLA

Open X-Embodiment 成为所有后续通用 robot policy 的数据基础。Octo, OpenVLA, pi_0, GR00T 都在 Open X 上训练或评估。这是 RT 系列对领域最具基础设施意义的贡献。

### 4.3 架构的传承: RT-2 --> GR00T

GR00T N1 继承了 RT-2 "VLM + action head" 的基本结构, 但在两个方向上大幅扩展:
- **低层控制**: RT 系列没有真正解决精细运动控制, GR00T 用 SONIC 补上了这块
- **世界模型**: RT 系列没有预测能力, GR00T 用 DreamZero 加入了 imagination

### 4.4 思想传承图

```
Google RT Series (2022-2024)
  |
  |-- [人 + 思想] --> PI (pi_Series)
  |   SayCan 的 "LLM 理解任务" + RT-2 的 "VLM->action"
  |   改进: 大模型->紧凑模型, 离散->Flow Matching, 加 RL
  |
  |-- [数据标准] --> 所有 VLA
  |   Open X-Embodiment --> Octo, OpenVLA, pi_0, GR00T
  |
  |-- [架构范式] --> GR00T
  |   RT-2 的 "VLM + action head" --> GR00T N1 的 "VLM + DiT"
  |   GR00T 补上了 RT 缺失的低层控制 (SONIC) 和世界模型 (DreamZero)
  |
  +-- [未解问题] --> 领域开放挑战
      离散 token 精度? --> Flow Matching (pi_0)
      太大不能部署?   --> 紧凑 VLM (pi_0, GR00T)
      没有低层控制?   --> SONIC (GR00T)
      没有世界模型?   --> DreamZero (GR00T)
```

---

## 5. Takeaway: 思维层面的原则

| # | 原则 | 为什么重要 | 对 robotics FM 研究的启示 |
|---|------|-----------|--------------------------|
| 1 | **web 预训练知识可以迁移到 robot, 这是 VLA 的核心假设** | RT-2 第一次验证: VLM 在互联网上学到的物体理解, 直接提升了机器人的零样本泛化能力 | 不要从头训 robot 模型, 要站在 VLM 的肩膀上 |
| 2 | **规划和执行的分离是瓶颈, 端到端是趋势** | SayCan 证明了 LLM 能规划, 但预写 skill 不可扩展; RT-2 把规划和执行合并到一个模型里 | 但层级方案 (RT-H, Hi Robot) 说明完全端到端不一定是唯一解 -- 关键是接口要可学习, 不能手写 |
| 3 | **跨 embodiment 数据共享的收益大于 domain gap 的损失** | Open X 证明: 混合不同机器人的数据, 即使形态不同, 也比只用自己的数据好 | 不要只用自己的机器人数据。数据多样性 > 数据纯净度 |
| 4 | **scaling 在 robot 上也 work, 但部署约束会反过来定义模型大小** | PaLM-E 证明越大越好, 但太大不能用; pi_0 证明紧凑模型 + 好数据可以逼近大模型 | 模型大小不是越大越好, 而是在部署约束下的最优解 |
| 5 | **人工遥操作不可扩展, 自主数据采集才是数据飞轮的正确形态** | AutoRT 用 VLM 分配任务 + 评估, 机器人自主探集, 效率远超人工 | 长期看, robot 数据不能靠人标注, 要靠 robot 自己生成 |
| 6 | **核心团队 = 核心竞争力, RT 团队出走后 Google robotics 明显减速** | 思想在人脑里, 不在代码库里。PI 快速崛起的速度验证了这一点 | 关注人, 不只关注架构 |

---

## 6. 文件索引

```
Google_RT_Series/
+-- RT_family_notes.md              <-- this file
+-- 22_SayCan/                      # LLM + affordance grounding
+-- 22_CodeAsPolicies/              # LLM generates control code
+-- 22_InnerMonologue/              # LLM closed-loop feedback
+-- 22_RT1/                         # first robot Transformer
+-- 23_RT2/                         # first VLA
+-- 23_PaLME/                       # giant embodied LM
+-- 23_OpenXEmbodiment/             # cross-embodiment dataset
+-- 24_AutoRT/                      # autonomous data collection
+-- 24_RTH/                         # hierarchical action + language interface
```
