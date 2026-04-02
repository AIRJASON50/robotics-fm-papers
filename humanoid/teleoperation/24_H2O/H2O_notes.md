# H2O / OmniH2O: Human-to-Humanoid Teleoperation -- 学习笔记
> 一句话: 首个基于 RL 的全身 humanoid 实时遥操系统, 仅需 RGB camera; OmniH2O 通过 teacher-student + history 去掉 MoCap 依赖
> 论文: Tairan He, Zhengyi Luo et al. H2O (ICRA 2024) + OmniH2O (CoRL 2024)

## 这篇论文解决了什么问题
如何用低成本设备 (RGB camera / VR headset) 实时控制全尺寸 humanoid 做全身动作?
- **Embodiment gap**: 人与 H1 的关节结构/动力学差异巨大, 动作不能直接映射
- **输入稀疏+有噪**: RGB pose estimation 有延迟和误差; VR 只给 3 个点 (head + 2 hands)
- **H2O 残留问题**: 部署时需要 MoCap 提供 global linear velocity -- OmniH2O 用 25-step history 替代

## 核心想法 (用直觉解释)
**H2O**: AMASS 动捕数据 -> gradient-based retargeting 到 H1 -> privileged imitator 过滤不可行动作 -> RL motion imitation + sim-to-real DR -> 零样本部署

**OmniH2O 的关键升级**: teacher-student distillation
- Teacher 看完整 rigid body state, 容易训练
- Student 只看 25-step proprioception history + sparse keypoint diff, 不需要 MoCap
- DAgger 蒸馏: 让 student 模仿 teacher 的 action

直觉: 连续 25 帧 joint pos/vel + angular vel 包含足够的动力学信息, policy 能从中隐式估计 root velocity。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| Sim-to-data filtering | Privileged imitator 先跑, 删除失败序列 | 不可行动作会"浪费" RL 学习资源, 过滤后 77% -> 95%+ |
| Standing data augmentation | 固定 root + lower body, 只保留上肢动作 | AMASS 以移动为主, 但操作任务需要站稳; bias 数据分布解决 |
| Max feet height reward | 惩罚每步最大抬脚高度不够 | 解决原地小步乱踩 (stomping): 要走就抬够高, 不走就别抬 |
| 上下半身分权 reward | Upper sigma=0.03 (严), Lower sigma=0.5 (松), lower action_rate 惩罚 6x | 优先保上半身精确, 给下半身自由度保持平衡 |
| Motion package loss DR | 冻结 motion ref 1-10 steps 模拟通信丢包 | 论文未提, 但对真实遥操至关重要 |

## 这篇论文之后发生了什么
- **OmniH2O-6 dataset**: 首个 humanoid whole-body control 真机数据集, 6 任务 ~40 min
- 启发 OmniRetarget (Amazon, 2025) 在 retargeting 质量上改进
- Teacher-student + history 范式被 ASAP, TWIST2 等后续工作广泛采用
- GPT-4o 集成展示了 vision-language -> motion primitive -> humanoid 的完整闭环

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | History 替代 privileged state 是 sim-to-real 的通用模式 | 类比 LLM: context window 替代显式 state -- 让模型从序列中推断隐藏状态 |
| 2 | Data filtering (sim-to-data) 比 data augmentation 更优先 | 数据质量 > 数量, 和 LLM pre-training 数据清洗同理 |
| 3 | 上下半身分权 = 分层 attention 的物理版 | Robotics FM 需要知道哪些自由度对任务更重要, 类似 attention 的功能 |
| 4 | Paper vs code gap 巨大 (10+ obs versions, 5+ curriculum, package loss) | 真正的 know-how 在工程细节里, 不在论文公式中 |
