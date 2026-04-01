# 学习导读: 从灵巧手 RL 到 Robotics Foundation Model

> **你**: PPO sim2real 灵巧手操作实践者
> **目标**: 理解 CS 基础模型技术如何重塑机器人领域, 以及你的 RL 经验如何融入
> **本文件**: 打开这个 repo 后第一个要读的。指向所有其他内容。

## 你的学习路径 (全景)

```
Phase 1                      Phase 2                         Phase 3                    Phase 4
你的领域                      CS 方法论                        三大家族                    回到你的工作
~~~~~~~~~~~                  ~~~~~~~~~~~~~~                  ~~~~~~~~~~~~~~             ~~~~~~~~~~~~
manip_landscape.md           CS2Robotics_Roadmap.md          RT_family_notes.md         把学到的应用到
  灵巧操作 5 个主题             Level 0: 表征与 Transformer       "VLA 起源故事"           你的 PPO sim2real
humanoid_landscape.md          Level 1: 预训练 + 规模化       pi_family_notes.md         pipeline 上
  人形机器人 4 个主题            Level 2: 生成式 Policy          "VLA + 离线 RL"
                               Level 3: 完整的 Robot FM       GR00T_family_notes.md
                               Level 4: 前沿方向                "分层 + 全身控制"

~2h                          ~30h (自定节奏)                  ~6h                        持续
```

## Phase 1: 理解你的领域 (manip + humanoid)

**为什么先看这个**: 你需要先搞清楚 "FM 该解决什么问题", 再去学 "FM 怎么做"。

| 文件 | 路径 | 核心 Takeaway | 时间 |
|------|------|-------------|------|
| 灵巧操作全景 | `manip/manip_landscape.md` | 5 个主题 (traditional_rl→human2robot→scaling_rl→sim2real→fm_manip) 展示了 per-task PPO 在接触多样性和物体泛化上碰壁 | 1h |
| 人形机器人全景 | `humanoid/humanoid_landscape.md` | Motion tracking 是人形的 "ImageNet moment"; SONIC 证明 PPO + scale 可以在 FM 级别 work | 1h |

**Phase 1 Takeaway**: 你的 PPO sim2real 技能是起点, 不是终点。领域正在从 "每个任务一个策略" 走向 "一个模型, 多种任务"。Phase 2 解释怎么做到的。

## Phase 2: 理解方法论来源 (foundation_model)

**主文档**: `foundation_model/CS2Robotics_Roadmap.md` — Level 0-4 渐进式路线。

| Level | 核心问题 | 读什么 | 时间 |
|-------|---------|--------|------|
| 0 | 什么模式可以从 CS 迁移到 Robotics? | 表征学习 (Bengio) + Transformer | 3h |
| 1 | 怎么做大规模预训练? | GPT 系列 + Scaling Laws + MAE + DINOv2 + CV 演进图谱 | 8h |
| 2 | 怎么从 RL 走向生成式 Policy? | CLIP + DDPM + Flow Matching + Diffusion Policy + ACT + DT + R3M/VIP | 12h |
| 3 | 完整的 Robot FM 长什么样? | RT/PI/GR00T 三份 family notes + pi_0/SONIC/GR00T N1 精读 | 12h |
| 4 | 下一步是什么? | DreamZero + Robot Scaling Laws + surveys + 局限性声明 | 7h |

**升级考试**: 每个 Level 完成后做 `foundation_model/note/level_exams.md` 中的考试。80 分进入下一级。

## Phase 3: 深入三大家族 (RT / PI / GR00T)

读 `foundation_model/robotics/families/` 下的 family notes。每个家族教你不同的东西:

| 家族 | Notes 路径 | 教你什么 |
|------|-----------|---------|
| Google RT | `Google_RT_Series/RT_family_notes.md` | VLA 是怎么诞生的: 从 LLM 做规划 (SayCan) 到端到端 (RT-2)。核心人物为什么出走做 PI。教训: web 知识可以迁移到机器人, 但离散 token 有精度上限。 |
| PI | `pi_Series/pi_family_notes.md` | 完整的 VLA 演进: flow matching + knowledge insulation + 跨 embodiment 预训练。pi\*0.6 展示离线 RL 做 post-training — 你的 PPO 经验在这里直接适用。 |
| GR00T | `GR00T_Series/GR00T_family_notes.md` | 分层架构: VLA 大脑 (10Hz) + WBC 小脑 (120Hz) + 世界模型想象 (DreamZero)。你的 sim2real 和 motion tracking 技能对应小脑层。 |

## Phase 4: 回到你的工作

学完后怎么用到你的 PPO sim2real 灵巧手工作上:

1. **短期**: 你的 PPO + sim2real pipeline 在 WBC / 低层控制层仍然有效 (SONIC 验证了这一点)。继续打磨。
2. **中期**: 学 Diffusion Policy 或 Flow Matching 做高层策略, PPO 保留在低层做追踪控制。这是 GR00T N1 的架构模式。
3. **长期**: 在预训练的 VLA (如 openpi) 上用你的灵巧手 demo 做 fine-tune。用 RL 做 post-training (pi\*0.6 路线)。
4. **数据策略**: 搭建遥操作 pipeline 采集 demo 数据。瓶颈是数据, 不是算法。
5. **重点精读 pi\*0.6**: 它展示了离线 RL 怎么 fine-tune VLA — 正好是 "你的 RL 技能" 和 "FM 新范式" 的交叉点。

## 快速查阅索引

| 我想了解... | 去看 |
|-----------|------|
| 为什么 per-task RL 不能 scale | `manip/manip_landscape.md` Section 0-1 |
| motion tracking 怎么统一人形控制 | `humanoid/humanoid_landscape.md` Theme A |
| CS→Robotics 完整迁移路线 | `foundation_model/CS2Robotics_Roadmap.md` |
| GPT 的 pre-train+fine-tune 怎么映射到 robot | `CS2Robotics_Roadmap.md` Level 1 |
| 为什么 diffusion/flow 替代了 Gaussian policy | `CS2Robotics_Roadmap.md` Level 2 |
| pi_0 的架构和设计思路 | `robotics/families/pi_Series/pi_family_notes.md` |
| GR00T 的分层 VLA+WBC 设计 | `robotics/families/GR00T_Series/GR00T_family_notes.md` |
| RL 怎么做 VLA 的 post-training | PI family notes, 搜索 "pi\*0.6" |
| VLA 之后是什么 (世界模型) | `CS2Robotics_Roadmap.md` Level 4 + DreamZero notes |
| 考试题测试我的理解 | `foundation_model/note/level_exams.md` |
| 这个 repo 所有论文的清单 | `papers.yaml` 或 `CLAUDE.md` |
