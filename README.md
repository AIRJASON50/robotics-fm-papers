# Robotics Foundation Model Paper Library

从 RL 实践者到 Robotics FM 研究者的知识库。

## 我是谁

- **背景**: PPO sim2real, 灵巧手操作 (dexterous manipulation)
- **目标**: 理解 CS (LLM+CV) 如何塑造了现代 robotics foundation model, 学习方法论和设计思想
- **核心洞察**: Robotics 正在重走 LLM/CV 的路 — 学习这条路 = 看到 robotics 的未来

## 结构

```
paper/
├── foundation_model/          # CS → Robotics 知识库
│   ├── foundations/           #   通用 ML 基础 (9): Transformer, PPO, SAC, Adam, ...
│   ├── LLM/                  #   LLM 知识体系: NLP 基础 + 5 个模型家族
│   ├── CV/                   #   CV 知识体系: 7 个技术方向 (24 篇)
│   ├── robotics/             #   3 大家族 (RT/PI/GR00T) + 方法论文 + surveys
│   ├── surveys/              #   14 篇 survey (CV 7 + Robotics 7)
│   └── CS2Robotics_Roadmap.md    # 主学习路线 (按问题分级, 非按领域)
├── humanoid/                  # 人形机器人 (4 主题: motion_tracking, teleoperation, sim2real, video_world_model)
├── manip/                     # 灵巧手操作 (5 主题: traditional_rl, human2robot, scaling_rl, sim2real, fm_manip)
├── html2aitext_convert/       # arxiv → markdown 工具
├── papers.yaml                # 全库清单 (arxiv IDs + repo URLs)
└── scripts/setup.sh           # 一键 clone 所有代码仓库
```

## 学习路线

详见 [`foundation_model/CS2Robotics_Roadmap.md`](foundation_model/CS2Robotics_Roadmap.md):

```
Level 0: 什么模式可以从 CS 迁移到 Robotics? (已通过)
Level 1: 怎么做大规模预训练? (GPT 范式 + 自监督视觉 + Scaling Laws)
Level 2: 怎么从 RL 走向生成式 Policy? (Diffusion/Flow + RL frontier)
Level 3: 完整的 Robot FM 长什么样? (RT/PI/GR00T 三大家族 notes)
Level 4: 下一步和开放问题 (DreamZero/WAM + surveys)
```

每个 Level 回答一个问题, 混合 LLM+CV+RL 内容, 不按领域分。

## 三大 Robotics FM 家族

| 家族 | 团队 | 路线 | Family Notes |
|------|------|------|-------------|
| **Google RT** | DeepMind → PI | VLA 起源: SayCan → RT-1 → RT-2 | `robotics/families/Google_RT_Series/` |
| **PI** | Physical Intelligence | VLA 纵深: pi_0 → FAST → pi_0.5 → pi\*0.6 | `robotics/families/pi_Series/` |
| **GR00T** | NVIDIA | 全栈: Isaac-GR00T + SONIC + DreamZero | `robotics/families/GR00T_Series/` |

## Git 跟踪什么

| 跟踪 | 不跟踪 (gitignore) |
|------|-------------------|
| 分析笔记 / family notes | PDF 文件 |
| arxiv markdown | 代码仓库 clone |
| 学习路线图 + 考试 | HTML 缓存 / 大数据集 |
| papers.yaml 清单 | |

## 快速开始

```bash
git clone <url> && cd paper
./scripts/setup.sh                           # clone 所有代码仓库
cat foundation_model/CS2Robotics_Roadmap.md  # 开始学习
```
