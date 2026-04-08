# Robotics Foundation Model Paper Library

从灵巧手 RL 到 Robotics Foundation Model 的完整学习路径。

## 这个库的故事

```
第一步: 做灵巧手操作 (manip/)
  用 PPO sim2real 做灵巧手, 积累了 RL + reward design + sim2real 的实践经验。
  但发现: 每个任务单独训练 RL, 不可扩展。

第二步: 看人形机器人 (humanoid/)
  发现 SONIC 用 motion tracking 统一了所有人形动作, 不需要 per-task reward。
  类比: motion tracking 之于 humanoid = next-token prediction 之于 LLM。

第三步: 追溯到 CS (foundation_model/)
  发现 robotics 正在重走 LLM/CV 的路:
    LLM: task-specific → pre-train+finetune → scale → RLHF → product
    Robot: task-specific RL → pre-train+finetune (RT-1) → scale (Open-X) → RL finetune (pi*0.6) → ???

  Robotics 今天 ≈ LLM 的 InstructGPT→ChatGPT 之间。
  学习 LLM/CV 走过的路 = 看到 robotics 的未来。

目标: 理解 CS→Robotics 的技术迁移, 学习方法论, 为构建 robotics FM 做准备。
```

## 三块知识库的关系

```
manip/                          你的起点 -- 灵巧手 RL 实践
  ├── 5 个主题: traditional_rl → human2robot → scaling_rl → sim2real → fm_manip
  └── manip_landscape.md        全景 + humanoid→manip 桥接 + FM 启示

humanoid/                       扩展视野 -- 全身控制
  ├── 4 个主题: motion_tracking → teleoperation → sim2real → video_world_model
  └── humanoid_landscape.md     全景 + 研究者脉络

foundation_model/               方法论来源 -- CS 怎么做 FM
  ├── foundations/ (5 第一原理 + RL)  NN Origins, Representation, Transfer, Transformer, RL
  ├── LLM/ (5 families)        LLM 如何 scale + align + 产品化
  ├── CV/ (24 papers)           视觉表征 + 生成模型 + 视频理解
  ├── robotics/ (3 families)    RT/PI/GR00T 三大 robotics FM 家族
  └── CS2Robotics_Roadmap.md    主学习路线 (Level 0-4, 按问题分级)
```

**manip 教你"问题是什么"**, **humanoid 教你"规模化怎么做"**, **foundation_model 教你"方法论从哪来"**。

## 导读: 从哪开始

| 你想做什么 | 看什么 |
|-----------|--------|
| 理解整体学习路径 | [`foundation_model/CS2Robotics_Roadmap.md`](foundation_model/CS2Robotics_Roadmap.md) |
| 理解灵巧操作全景 + 从 humanoid 到 manip 的演进 | [`manip/manip_landscape.md`](manip/manip_landscape.md) |
| 理解人形机器人全景 + 研究者脉络 | [`humanoid/humanoid_landscape.md`](humanoid/humanoid_landscape.md) |
| 理解 LLM 怎么影响 robotics | [`foundation_model/LLM/LLM_技术交织与机器人启示.md`](foundation_model/LLM/LLM_技术交织与机器人启示.md) |
| 理解 CV 怎么影响 robotics | [`foundation_model/CV/CV_技术演进与机器人启示.md`](foundation_model/CV/CV_技术演进与机器人启示.md) |
| 理解三大 robotics FM 家族 | RT / PI / GR00T 各自的 `family_notes.md` |

## 学习路线 (Level 0-4)

详见 [`CS2Robotics_Roadmap.md`](foundation_model/CS2Robotics_Roadmap.md):

```
Level 0: 什么模式可以从 CS 迁移到 Robotics? (已通过)
Level 1: 怎么做大规模预训练?
Level 2: 怎么从 RL 走向生成式 Policy?
Level 3: 完整的 Robot FM 长什么样? (三大家族 family notes)
Level 4: 下一步和开放问题
```

每个 Level 回答一个问题, 混合 LLM+CV+RL 内容。

## 三大 Robotics FM 家族

| 家族 | 路线 | 与你的关联 |
|------|------|---------|
| **Google RT** → PI | VLA 起源→纵深 | pi\*0.6 的 offline RL = 你的 RL 经验直接适用 |
| **GR00T** (NVIDIA) | VLA + SONIC + DreamZero | SONIC 的 motion tracking = 你的灵巧手追踪的放大版 |
| **PI** | VLA + FAST + offline RL | Knowledge Insulation + RL Token = FM 时代的 RL 角色 |

## 结构

```
paper/
├── foundation_model/          # CS → Robotics FM 知识库
│   ├── foundations/           #   5 第一原理: NN Origins, Representation, Transfer, Transformer + RL/
│   ├── LLM/ (5 families)     #   GPT, Kimi, Qwen, DeepSeek, Llama
│   ├── CV/ (24 papers)        #   backbone, generation, VL, 3D, SSL, video
│   ├── robotics/ (3 families) #   RT, PI, GR00T + 方法论文
│   └── surveys/ (14)
├── humanoid/ (4 themes)       # motion_tracking, teleoperation, sim2real, video_wm
├── manip/ (5 themes)          # traditional_rl, human2robot, scaling_rl, sim2real, fm_manip
├── papers.yaml                # 全库清单
└── CLAUDE.md                  # AI 协作指南
```

## Git 跟踪

| 跟踪 | 不跟踪 |
|------|--------|
| 笔记 / family notes / landscape | PDF / 代码仓库 / 缓存 |
