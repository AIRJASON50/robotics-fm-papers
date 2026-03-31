# Google RT Series -- VLA 的起源故事

> **目的**: 理解 VLA 是怎么从 "LLM 做规划" 一步步演化到 "VLM 直接出动作" 的。
> RT 系列是整个 robotics FM 的源头——PI 和 GR00T 都是它的后续。

---

## 1. 背景: 为什么是 Google

Google DeepMind Robotics 在 2022-2023 年拥有独一无二的条件:
- **PaLM/PaLM-2**: 当时最强的 LLM (GPT-4 之前)
- **ViT/SigLIP**: 当时最强的视觉编码器
- **数十个真实机器人**: Google 办公室里部署了数百台 Everyday Robot
- **Karol Hausman, Brian Ichter, Pete Florence**: 后来创办 PI 的核心人物

这些条件让 Google 成为第一个把 LLM/VLM 和真实机器人结合的团队。

---

## 2. 演进脉络: 三个阶段

```
=== Phase 1: LLM-as-Planner (2022) ===
  "LLM 不直接控制机器人, 而是做高层规划"

22_SayCan: LLM 规划 + CLIP affordance grounding
  LLM: "要拿苹果, 步骤是: 1.走到桌子 2.抓苹果 3...."
  CLIP: "这些步骤中, 哪些在当前场景中可执行?"
  机器人: 用预训练的低层 skill 执行可行的步骤

  贡献: 第一次证明 LLM 可以做 robot task planning
  限制: 低层 skill 是手写的, 不可扩展

22_CodeAsPolicies: LLM 生成控制代码
  不是"规划步骤", 而是"直接写 Python 代码控制机器人"
  LLM 输出: move_to(apple); grasp(); move_to(plate); release()

  贡献: 比 SayCan 更灵活 (代码能组合任意行为)
  限制: 需要预定义 API (move_to, grasp 等)

22_InnerMonologue: LLM 做闭环反馈
  执行失败 → 用 VLM 描述当前状态 → LLM 重新规划
  "没抓到苹果" → "苹果滑到左边了, 重新尝试从左侧抓取"

  贡献: 第一次做 LLM-robot 的闭环 (不是 open-loop 一次规划)
  限制: 反馈全靠语言, 不够精确

Phase 1 的总结:
  LLM 能做规划, 但规划和执行是分离的
  低层执行器 (skill) 需要单独训练/手写
  → 能不能让一个模型端到端搞定?

=== Phase 2: Robot Transformer (2022-2023) ===
  "不用 LLM 做规划, 用 Transformer 直接做端到端 policy"

22_RT1: 第一个大规模 robot Transformer
  架构: EfficientNet (视觉) + TokenLearner + Transformer (决策)
  训练: 130K 真实 episodes, 700+ 任务
  动作: 离散化为 token (每个 DOF 256 bin)

  贡献: 证明 Transformer 可以做 robot policy (不只是 NLP)
         130K 数据 = 当时最大的 robot 数据集
  限制: 只用了自己的数据, 没有继承 web 知识

23_RT2: 第一个 VLA (Vision-Language-Action Model)
  核心思想: 为什么不用 VLM 替代 RT-1 的视觉编码器?
  架构: PaLI-X (55B VLM) → 直接输出离散动作 token

  VLM 在 web 数据上预训练 → 已经理解 "苹果是什么"
  co-fine-tuning: web VQA 数据 + robot demo 混合训练
  → VLM 的视觉-语言知识迁移到了 robot

  贡献: 第一次证明 "web 知识可以迁移到 robot"
         新物体零样本泛化 (训练没见过的物体也能抓)
  限制: 55B 太大, 推理太慢 (~3Hz)

23_PaLME: 走到极端 — 562B embodied LM
  输入: 图像 + 语言 + robot 状态 → 全部当 token
  输出: 语言回答 / 动作 token

  把 PaLM (540B) 变成多模态 embodied model
  证明: 更大的 VLM → 更好的 robot 能力 (scaling works)
  限制: 562B 不可能部署在真机上

Phase 2 的总结:
  RT-1: Transformer 可以做 robot policy
  RT-2: VLM 可以做 robot policy, 且继承 web 知识
  PaLM-E: 越大越好, 但太大不能用
  → 需要: 更小的 VLM + 更高效的动作生成

=== Phase 3: 数据与规模化 (2023-2024) ===
  "模型架构够了, 下一步是数据和部署"

23_OpenXEmbodiment: 跨机器人数据集
  22 个机器人平台, 527 种技能, 160K+ 轨迹
  第一个大规模跨 embodiment 数据集
  RT-1-X / RT-2-X: 在 Open X 上训练的 cross-embodiment 模型

  贡献: robotics 的 "Common Crawl"
         证明跨 embodiment 数据共享有效

24_AutoRT: 大规模机器人数据采集
  用 VLM 自动给机器人分配任务
  机器人自主探索 → VLM 评估成功/失败 → 数据飞轮
  20+ 机器人同时在 Google 办公室运行, 收集 77K+ 轨迹

  贡献: 第一次做 "机器人自主数据采集" (不靠人类遥操作)

24_RTH: 层级动作 + 语言中间表示
  高层: VLM 输出自然语言描述的动作 ("move hand left 5cm")
  低层: 语言描述 → 具体关节角

  贡献: 用语言作为高低层之间的接口 (可解释 + 可纠正)
  (和 PI 的 Hi Robot 思路类似, 但 RT-H 早 2 个月)

=== Phase 4: 核心团队离开 → PI 成立 (2024) ===

2024 年, RT 系列的核心作者集体离开 Google 创办 PI:
  Karol Hausman (SayCan, RT-1/2 核心作者) → PI CEO
  Brian Ichter (SayCan, Inner Monologue) → PI 联合创始人
  Sergey Levine (RT-2 顾问, Berkeley) → PI 联合创始人
  Chelsea Finn (RT-2 顾问, Stanford) → PI 联合创始人

PI 的 pi_0 (2024.10) 本质是 RT-2 的精神续作:
  RT-2: PaLI-X (55B) + 离散动作 token → 太大太慢
  pi_0: PaliGemma (3B) + Flow Matching → 小而快

  同样的思想 (VLM → robot action), 更好的工程实现
```

---

## 3. 关键设计决策演进

| 问题 | SayCan (2022) | RT-1 (2022) | RT-2 (2023) | pi_0 (2024, PI) |
|------|-------------|------------|------------|----------------|
| 用 LLM 吗 | 是 (做规划) | 否 (纯 Transformer) | 是 (VLM 做骨干) | 是 (PaliGemma) |
| 动作怎么出 | 调用预写 skill | 离散 token (256 bin) | 离散 token (256 bin) | **Flow Matching (连续)** |
| 继承 web 知识 | 通过 LLM | 不继承 | **VLM co-fine-tune** | VLM pre-train |
| 模型大小 | PaLM 540B + small skills | ~35M | 55B | **3B** |
| 推理频率 | ~1 Hz | ~3 Hz | ~3 Hz | **~50 Hz** |
| 新物体泛化 | 靠 LLM 语言理解 | 不行 | **可以** (VLM 知识迁移) | 可以 |

**核心演化**: LLM 做规划 (SayCan) → Transformer 做 policy (RT-1) → VLM 做 policy + 继承 web 知识 (RT-2) → 更小更快的 VLM + 连续动作生成 (pi_0)

---

## 4. RT 系列对整个领域的定义性贡献

| 贡献 | 论文 | 后续影响 |
|------|------|---------|
| **证明 LLM 可以做 robot planning** | SayCan | 催生了整个 LLM-for-robotics 方向 |
| **定义了 VLA 架构** | RT-2 | pi_0, GR00T, OpenVLA 都是 VLA |
| **证明 web 知识可以迁移到 robot** | RT-2 | 所有 VLA 的核心假设 |
| **建立了跨 embodiment 数据标准** | Open X-Embodiment | 所有通用 robot policy 的数据基础 |
| **证明离散动作 token 可行** | RT-1/RT-2 | OpenVLA 继承; FAST 改进 |
| **证明更大 VLM = 更好 robot** | PaLM-E | 支持 scaling 方向 |

**但 RT 系列也留下了未解问题**:
- 离散 token 精度不够 → pi_0 用 Flow Matching 解决
- 55B 太大 → pi_0 用 3B PaliGemma 解决
- 没有低层控制 → GR00T 的 SONIC 解决
- 没有世界模型 → GR00T 的 DreamZero 解决

---

## 5. Takeaway

| # | Takeaway | 原理 | 对你的启示 |
|---|----------|------|-----------|
| 1 | **VLA 起源于 RT-2: VLM 直接出动作 token** | web 预训练的 VLM 理解物体 → 零样本泛化 | VLA 的核心价值是知识迁移, 不只是端到端 |
| 2 | **LLM-as-planner 是 VLA 的前身, 不是替代品** | SayCan 证明了 LLM 理解任务, RT-2 把它端到端化 | 两者可以组合 (Hi Robot / RT-H 的层级方案) |
| 3 | **跨 embodiment 数据共享有效 (Open X)** | 不同机器人的数据混合训练 → 比单一数据更好 | 不要只用自己的机器人数据 |
| 4 | **离散 token 是第一步, 连续生成是下一步** | RT-1/2 用 256-bin 可行但粗糙, Flow Matching 更精确 | 新项目直接用 Flow Matching |
| 5 | **核心团队 = 核心竞争力** | RT 团队离开 → Google robotics 进展放缓, PI 快速崛起 | 人比架构重要 |
| 6 | **AutoRT: 数据飞轮的正确打开方式** | VLM 自动分配任务 + 评估 → 机器人自主采集数据 | 人工遥操作不可扩展, 自主采集才是 |

---

## 6. RT Series → PI → GR00T 的传承关系

```
Google RT Series (2022-2024)
  │
  ├── 思想传承 → PI (pi_Series)
  │   SayCan 的 "LLM 理解任务" + RT-2 的 "VLM→action" → pi_0
  │   核心作者: Hausman, Ichter, Levine, Finn
  │   改进: 55B→3B, 离散→Flow Matching, 加 RL
  │
  ├── 数据传承 → 所有 VLA
  │   Open X-Embodiment → Octo, OpenVLA, pi_0, GR00T 的训练数据
  │
  └── 架构传承 → GR00T
      RT-2 的 "VLM + action head" → GR00T N1 的 "Eagle VLM + DiT"
      但 GR00T 加了 SONIC (RT 没有的低层控制)
      和 DreamZero (RT 没有的世界模型)
```

---

## 7. 文件索引

```
Google_RT_Series/
├── RT_family_notes.md              ← 本文件
├── 22_SayCan/                      # LLM + affordance grounding
├── 22_CodeAsPolicies/              # LLM 生成控制代码
├── 22_InnerMonologue/              # LLM 闭环反馈
├── 22_RT1/                         # 第一个 robot Transformer
├── 23_RT2/                         # 第一个 VLA
├── 23_PaLME/                       # 562B embodied LM
├── 23_OpenXEmbodiment/             # 跨机器人数据集
├── 24_AutoRT/                      # 自主数据采集
└── 24_RTH/                         # 层级动作 + 语言中间表示
```
