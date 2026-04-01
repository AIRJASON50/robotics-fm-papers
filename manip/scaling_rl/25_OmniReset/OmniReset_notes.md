# UWLab / OmniReset 分析笔记

- **论文**: Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning
- **作者**: Patrick Yin*, Tyler Westenbroek* 等 (University of Washington, NVIDIA, Microsoft Research)
- **年份**: 2026 (arXiv: 2603.15789v2)
- **代码**: UWLab (基于 Isaac Lab 扩展的框架)

---

## 1. Core Problem

灵巧操作中 RL 面临的核心瓶颈是 **exploration saturation**: 标准 on-policy RL (如 PPO) 在大规模并行仿真中训练时, agent 反复采样狭窄的 state-action 分布, 陷入 local minima. 具体表现为:

- **长 horizon 任务失败**: 对于需要 pick -> reorient -> insert 等多阶段行为的任务, agent 无法发现从 reaching 到 goal 的完整路径
- **计算量无法转化为性能**: 增加 parallel environments 后性能快速饱和, 因为所有环境都在探索同一片状态空间
- **过度依赖人工设计**: 现有方法需要 per-task reward shaping, curriculum design, 或 human demonstrations 来缓解探索困难

这与 LLM 领域形成鲜明对比 -- 在 NLP 中, 简单的 RL 算法 (如 RLHF/GRPO) 配合 scale 就能产生涌现能力. 本文试图回答: **机器人操作能否也通过简单算法 + scale 达到类似效果?**

---

## 2. Method Overview

### 2.1 整体架构

OmniReset 是一个两阶段 pipeline:

```
Stage 1: State-based RL (OmniReset)
  - Diverse reset states 生成 (离线预计算)
  - PPO + gSDE + Asymmetric Actor-Critic
  - Task-agnostic reward function
  - 大规模并行环境 (4096+ envs)

Stage 2: Sim-to-Real Transfer
  - Student-teacher distillation (80K trajectories)
  - RGB visuomotor policy (ResNet-18 + MLP)
  - Domain randomization (视觉 + 物理)
  - 可选: Finetune with sysid curriculum
```

### 2.2 训练流程

1. **离线阶段**: 对目标物体采样 1000 个可行 grasp points, 生成 partial assembly offsets, 构建 4 类 reset 数据集
2. **RL 训练**: 从 4 类 reset 分布均匀采样初始状态, 用标准 PPO 训练 state-based policy
3. **Finetune** (sim2real): 在 system-identified dynamics 上用 curriculum 逐步增加 randomization
4. **Distillation**: 收集 80K expert rollouts, 训练 RGB student policy
5. **部署**: Zero-shot transfer 到 UR7e 真机

### 2.3 关键设计选择

| 设计选择 | 具体实现 | 原因 |
|---------|---------|------|
| 算法 | PPO (on-policy) | 稳定性好, 易于并行化 |
| 探索噪声 | gSDE (Generalized State-Dependent Exploration) | 不同状态区域使用不同探索策略 |
| 网络架构 | Asymmetric Actor-Critic | Actor 仅用 compact observations, Critic 用 privileged info |
| 动作空间 | Relative Cartesian (OSC) | 小动作增量, 稳定训练 |
| 控制器 | Operational Space Control (torque-level) | 比 joint-space PD 更适合 contact-rich 任务 |
| Observation history | 5 frames (actor) / 1 frame (critic) | 提供时序信息, 同时保持 critic 稳定 |

---

## 3. Key Designs

### 3.1 Diverse Resets -- 解决 Exploration 的核心创新

OmniReset 的核心洞察: **与其让 agent 自己探索到关键中间状态, 不如直接通过 simulator reset 把 agent "放到"这些状态上**. 这在两个维度上扩展覆盖:

**维度 1: 物体路径覆盖** -- 在物体从桌面到目标的所有可能路径上采样物体位姿.

**维度 2: 机器人-物体交互覆盖** -- 在机器人与物体的所有可能交互模式上采样.

4 类 Reset 分布:

| Reset 类型 | 论文名称 | 代码名称 | 内容 |
|-----------|---------|---------|------|
| Reaching | S^R | `ObjectAnywhereEEAnywhere` | 物体在桌面随机位姿, gripper 在工作空间随机位置 |
| Near-Object | S^NO | `ObjectRestingEEGrasped` | 物体在桌面, EE 在物体附近的 grasp point (+offset), gripper 随机开/关 |
| Stable Grasp | S^G | `ObjectAnywhereEEGrasped` | 物体在空中随机位姿, EE 在 grasp point, 已形成稳定抓取 |
| Near-Goal | S^NG | `ObjectPartiallyAssembledEEGrasped` | 物体在 goal 附近(partial assembly offsets), EE 在 grasp point |

关键特点:
- **不构建图结构**: 4 类状态之间没有预定义的连接关系, 路径完全由 RL 自主发现
- **不编码动态行为**: 翻转、旋入等行为完全 emergent
- **Emergent curriculum**: 训练过程中自然出现 backward learning -- 先学会 near-goal, 再逐步学会从更远的状态成功

代码实现路径: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py` 中的 `MultiResetManager` 类. 核心逻辑:
- 预加载 4 个 `.pt` 数据集文件到 GPU
- 每次 reset 按概率 [0.25, 0.25, 0.25, 0.25] 采样数据集
- 从选中的数据集中随机采样一个状态, 写入仿真器

### 3.2 Large-Scale RL 的工程实践

**并行环境数量是关键**: Ablation 显示 4096 envs 远优于 512/1024 envs. 小 batch 能学会 near-goal 但无法 scale 到完整 long-horizon 任务. 直觉上, 大 batch 防止 catastrophic forgetting -- 当大部分 reset states 导致失败时, 少数成功样本不会被淹没.

**训练规模参考**:
- Peg Insertion: ~8h on 4x L40S GPU
- Leg Twisting: ~16h on 4x L40S GPU
- Drawer Assembly: ~24h on 4x L40S GPU
- Finetune (sysid curriculum): ~8-24h on 1-4x L40S GPU

**gSDE 的重要性**: 与标准 Gaussian noise 不同, gSDE 通过一个额外的 prediction head 生成 state-dependent 的 temporally-correlated 探索噪声. 这对多阶段任务至关重要 -- reaching 阶段需要大范围探索, insertion 阶段需要精细微调.

### 3.3 Task-Agnostic Reward

所有任务共享同一 reward function, 权重固定不变:

```
r = r_success + r_dist + r_reach + r_smooth + r_term

r_reach  = 0.1 * (1 - tanh(||p_ee - p_obj|| / sigma))       # EE 靠近物体
r_dist   = 0.1 * mean(exp(-||x_err||/sigma), exp(-||theta_err||/sigma))  # 物体靠近目标
r_success = 1.0 * 1[position_aligned & orientation_aligned]  # 稀疏成功奖励
r_smooth = -1e-4 * ||a||^2 - 1e-3 * ||a-a_prev||^2 - 1e-2 * ||dq||^2  # 平滑惩罚
r_term   = -100 * 1[abnormal_state]                          # 异常状态惩罚
```

关键洞察: **性能主要取决于 initial state distribution 的覆盖度, 而非 reward shaping**. 这是一个非常强的 claim -- dense resets 使得 sparse reward signals 可以顺畅传播.

---

## 4. Experiments

### 4.1 任务列表

| 任务 | 难度变体 | 描述 |
|------|---------|------|
| Leg Twisting | Easy/Hard | 螺旋拧入桌腿 (来自 FurnitureBench) |
| Drawer Insertion | Easy/Hard | 将抽屉底板插入抽屉框 |
| Peg Insertion | Easy/Hard | 圆柱插入孔 |
| Cube Stacking | Hard | 堆叠两个方块 |
| Wall Slide | Hard | 非 prehensile 推块 |
| Cupcake Placement | Hard | 放杯子蛋糕到盘子上 |
| Four Leg Assembly | -- | 4 个独立 policy + scripting 组合 |

Hard variant: 物体 xy 范围 [-0.2, 0.2] x [-0.15, 0.15], 随机 goal
Easy variant: 物体 xy 范围 [0.1, 0.12] x [0.1, 0.12], 固定 goal

### 4.2 Baseline 对比

所有 baseline 额外提供 100 条 expert demonstrations, OmniReset 不使用 demo:

| 方法 | 核心思想 | 表现 |
|------|---------|------|
| **OmniReset** | Diverse resets, no demo | 所有任务一致高成功率 |
| BC-PPO | BC loss + PPO, 标准 reset | Easy 部分成功, Hard 失败 |
| DeepMimic | Demo 作为 reward augmentation + reset from demo | 类似 BC-PPO |
| Demo Curriculum | 基于成功率的 autocurriculum + demo resets | 最佳 baseline, 但 Hard variant 仍大幅落后 |

关键发现:
- Baselines 可以从 near-goal states 成功, 但无法 scale 到完整 long-horizon (从 reaching states 开始)
- OmniReset 即使在没有 demo 的情况下也大幅优于使用 demo 的 baselines

### 4.3 Ablation 结果

1. **并行环境数量**: 512 -> 1024 -> 2048 -> 4096, 性能持续提升; 小规模训练只能解决 near-goal 子任务
2. **Grasp 采样范围**: Broad > Moderate > Narrow, 更广泛的 grasp 覆盖带来更好的 sample efficiency
3. **鲁棒性**: 对初始条件扰动, OmniReset 成功率几乎不受影响, baselines 快速下降

### 4.4 Sim-to-Real 结果

| 任务 | State-based (sim) | RGB distilled (sim) | RGB (real, zero-shot) | Diffusion Policy BC (100 demos) |
|------|-------------------|---------------------|-----------------------|-------------------------------|
| Peg Insertion | ~high | ~50% | 85.37% | ~2% |
| Leg Twisting | ~high | ~50% | 56.36% | ~2% |
| Drawer Insertion | ~high | ~50% | 15.38% | ~2% |

- RGB policy 在 sim 中成功率远低于 state-based expert, 说明 distillation 是主要瓶颈
- 但 real-world 表现大幅优于 Diffusion Policy BC baseline (100 demos)
- Policy 展现出 retrying behavior -- 失败后重新尝试

---

## 5. Related Work Analysis

### 5.1 与 OpenAI Rubik's Cube 的对比

| 维度 | OpenAI Rubik's Cube (2019) | OmniReset (2026) |
|------|---------------------------|------------------|
| 机器人平台 | Shadow Hand (多指灵巧手) | UR7e + Robotiq 2F-85 (parallel gripper) |
| 任务类型 | In-hand manipulation | Tabletop manipulation + insertion |
| 探索策略 | ADR (Automatic Domain Randomization) + LSTM | Diverse Resets + gSDE |
| Reset 策略 | 标准 reset | 4 类 diverse resets |
| Curriculum | Automatic (基于性能调整 DR) | 无 curriculum (仅 sim2real finetune 有) |
| Reward | Task-specific | Task-agnostic |
| Demo | 无 | 无 |
| Scale | 数千 CPU workers | 4096 GPU parallel envs |
| 核心贡献 | 展示 sim2real + massive scale 可以解决复杂任务 | 展示 diverse resets 可以替代 reward engineering |

两者的共同理念: **通过 scale 和系统设计而非算法复杂度来解决难题**. 但 OmniReset 更进一步, 证明了 task-agnostic 的方法也可以 work.

### 5.2 与 DexMachina 的对比

| 维度 | DexMachina (2024) | OmniReset (2026) |
|------|-------------------|------------------|
| 灵巧手类型 | LEAP Hand / Allegro Hand | Parallel Gripper |
| 方法类型 | 人类遥操 + RL | Pure RL (无 demo) |
| Demo 依赖 | 需要人类手部 demo | 完全不需要 |
| Reset 策略 | 从 demo 状态 reset | 程序化生成 diverse resets |
| 任务范围 | In-hand reorientation | Tabletop pick-place-insert |
| Sim2Real | Teacher-student | Teacher-student (类似) |

关键区别: DexMachina 依赖人类 demonstration 作为探索的引导, OmniReset 完全用程序化生成的 reset states 替代了这一需求.

### 5.3 与其他灵巧操作方法的对比

| 方法 | Demo 需求 | Curriculum | Reward 设计 | 长 Horizon 能力 |
|------|----------|-----------|------------|----------------|
| OmniReset | 无 | 无 (sim 训练) | Task-agnostic | 强 |
| DemoStart (Bauza 2025) | 需要 | Auto-curriculum | Task-specific | 中 |
| DeepMimic (Peng 2018) | 需要 | 无 | Demo-augmented | 弱 |
| Reverse Curriculum (Florensa 2017) | 无 | Goal-backward | Task-specific | 中 |
| DextrAH (Handa 2023) | 无 | 无 | Heavily shaped | 弱-中 |

---

## 6. Limitations & Future Directions

### 作者明确提出的局限:

1. **Grasp sampler 依赖**: 对高度非凸物体可能无法生成多样化的 grasps
2. **Bimanual / Dexterous hands**: 预计算稳定 grasps 对双臂或多指手更困难, 方法是否能 scale 到这些场景存疑
3. **Dynamics randomization 不足**: 相比现有 sim2real 方法, DR 范围较保守
4. **RGB distillation 瓶颈**: RGB policy 成功率远低于 state-based expert (~50% vs near-100%)
5. **Seed 敏感性**: 不同随机种子产生的行为差异大, 需要 seed selection (基于 noise robustness)
6. **Contact fidelity**: 依赖精细接触动力学的行为 (如推物体靠墙重定向) 难以 transfer

### 从代码推断的额外局限:

7. **Single rigid body 假设**: 框架假设操作单个 rigid body 到目标配置, 不支持 deformable objects 或多物体交互
8. **预计算 overhead**: 4 类 reset datasets 需要离线预计算 (grasp sampling + collision checking + simulator stepping), 增加了 pipeline 复杂度
9. **UR 系列特定优化**: 代码中大量针对 UR5e/UR7e + Robotiq 的具体实现 (sysid, controller gains 等), 推广到其他 platform 需要重新工程

### Future Directions:

- 结合 DAgger 或 image-based RL 提升 RGB policy 性能
- 研究 distillation 的 scaling laws (80K trajectories 仍在提升)
- 扩展到 dexterous hand manipulation
- 改进 contact modeling (多物理后端, 更精确的 SDF)
- 自动化 seed selection / behavior filtering

---

## 7. Paper vs Code Discrepancies

这是本笔记最关键的部分. 通过对比论文描述和 UWLab 代码实现, 发现以下差异:

### 7.1 Two-Stage Training Pipeline (论文未强调)

论文将 OmniReset 描述为一个"简单"的框架, 但代码揭示了一个 **两阶段训练 pipeline**:

- **Stage 1 (Train)**: 使用 implicit actuator (理想化动力学, 零摩擦/armature/delay), 4 类 diverse resets, 无 curriculum
  - 配置: `Ur5eRobotiq2f85RelCartesianOSCTrainCfg` (使用 `TrainEventCfg`)
- **Stage 2 (Finetune)**: 切换到 explicit actuator (system-identified dynamics), 用 ADR curriculum 逐步增加 sysid randomization 和 OSC gain randomization
  - 配置: `Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg` (使用 `FinetuneEventCfg` + `FinetuneCurriculumsCfg`)

论文 Appendix A.3.9 提到了这一点, 但主文将其描述为"无 curriculum"的方法. **实际上 "no curriculum" 仅指 Stage 1; sim2real 需要 Stage 2 的 curriculum**.

代码路径: `source/uwlab_tasks/.../omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py`

### 7.2 Reset 数据集命名差异

| 论文名称 | 代码名称 | 说明 |
|---------|---------|------|
| Reaching Resets (S^R) | `ObjectAnywhereEEAnywhere` | 一致 |
| Near-Object Resets (S^NO) | `ObjectRestingEEGrasped` | **名称差异**: 代码名暗示 "EE已抓取", 论文描述为 "EE在物体附近有offset" |
| Stable Grasp Resets (S^G) | `ObjectAnywhereEEGrasped` | 基本一致 |
| Near-Goal Resets (S^NG) | `ObjectPartiallyAssembledEEGrasped` | **名称差异**: 代码用 "partially assembled" 代替 "near-goal" |

### 7.3 额外的 Reward Terms (论文未提及)

代码中包含论文公式未列出的 reward terms:

- `collision_free`: 碰撞检测 reward (使用 OBB overlap check)
  - 代码路径: `source/uwlab_tasks/.../omnireset/mdp/rewards.py`, `collision_analyzer.py`
- `abnormal_robot_state`: 异常状态惩罚 (weight = -100), 同时作为 termination condition
- `progress_context`: 一个"监控型" reward term (weight=0.1), 实际不返回 reward 值(返回 zeros), 但负责计算 success/failure 状态供其他 reward 和 curriculum 使用

### 7.4 Extensive Domain Randomization (论文低估了复杂度)

代码中的 `BaseEventCfg` 包含大量 startup-time randomization:

- Robot, insertive object, receptive object, table 各自的 friction/mass randomization
- Gripper actuator gains randomization (reset-time)
- ARM sysid randomization (friction, armature, delay) -- 在 Finetune stage
- OSC controller gains randomization -- 在 Finetune stage

论文说"relatively modest levels of dynamics randomization", 但代码实现的 DR 范围其实相当广泛.

### 7.5 DAgger / Behavior Cloning 支持 (论文未提及)

代码中 `rsl_rl_cfg.py` 包含完整的 `Base_DAggerRunnerCfg`, 支持在线 behavior cloning (DAgger):
- `OffPolicyAlgorithmCfg` + `BehaviorCloningCfg`
- 可加载 expert JIT model, 配合 RL 训练
- 论文完全没有提到这一功能, 可能是实验未使用或效果不佳

代码路径: `source/uwlab_tasks/.../omnireset/config/ur5e_robotiq_2f85/agents/rsl_rl_cfg.py`

### 7.6 Gravity Control Event (论文未提及)

代码中有 `global_physics_control_event` 类, 可以在 episode 开头关闭重力 (用于 stable reset positioning), 然后在指定时间步后重新开启, 并可施加随机 force/torque 扰动. 论文没有讨论这一工程细节.

代码路径: `source/uwlab_tasks/.../omnireset/mdp/events.py`, line 460

### 7.7 Collision Analysis (论文未提及)

代码实现了完整的 OBB (Oriented Bounding Box) 碰撞检测系统 (`collision_analyzer.py`, `terminations.py`), 用于 reward 计算和 termination 判断. 论文仅简短提到了 "unsafe states" 的惩罚.

### 7.8 Success Monitor 系统 (论文未提及)

代码中有精细的 success monitoring:
- `SuccessMonitor` 跟踪每个 reset type 的历史成功率 (sliding window = 100)
- `MultiResetManager` 内部记录 per-task success rates 并通过 `extras["log"]` 输出
- `ProgressContext` 同时跟踪 position/orientation aligned 状态和 continuous success counter

这些监控机制是实现 ADR curriculum 的基础, 但论文中没有详细描述.

### 7.9 真机部署差异

- **UR7e vs UR5e**: 论文用 UR7e, 代码仓库的配置全部基于 UR5e. 这可能是代码发布时的简化版本, 或两者混用
- **Action space curriculum**: 代码中 `FinetuneCurriculumsCfg.action_scale` 从 [0.02,...] 逐步减小到 [0.01, 0.01, 0.002, ...], 论文 Appendix 描述了这一点但主文声称 "no curriculum"

---

## 8. Cross-Paper Comparison

### 8.1 与 Diffusion Policy 的对比

| 维度 | Diffusion Policy (Chi et al. 2024) | OmniReset (2026) |
|------|-----------------------------------|------------------|
| 方法范式 | Imitation Learning (action diffusion) | RL + distillation |
| 数据来源 | Human demonstrations | Sim-generated trajectories (80K) |
| 策略表示 | DDPM over action chunks | MLP (Gaussian) |
| 多模态行为 | 天然支持 (diffusion 的优势) | 通过 diverse resets 间接获得 |
| Contact-rich 任务 | 不擅长 (需要高频反应, action chunking 有 lag) | 强项 (RL 天然适合 reactive control) |
| 数据效率 | 100 demos 可 work | 需要大规模 sim 数据 |
| Sim-to-Real | 不需要 sim | 需要高保真 sim |

OmniReset 论文直接用 Diffusion Policy 作为 baseline, 在相同任务上 Diffusion Policy BC (100 real demos) 仅 ~2% 成功率, 而 OmniReset distilled policy 达 15-85%. 但这并非公平比较 -- OmniReset 使用了 80K sim trajectories + sim2real 工程.

OmniReset 作者也尝试了 diffusion-based distillation policy 但发现效果不如简单 MLP, 原因可能是 RL expert 的高频 reactive 行为与 action chunking 不兼容.

### 8.2 与 pi_0 (GR00T N1) 等 Foundation Model 方法的对比

| 维度 | Foundation Model 方法 (pi_0 等) | OmniReset |
|------|-------------------------------|-----------|
| 通用性 | 跨任务/跨机器人泛化 | 单任务训练 (但 reward 通用) |
| 训练数据 | 大规模真实数据 + 互联网数据 | 纯仿真数据 |
| 架构 | VLM backbone + action head | MLP (512-256-128-64) |
| 对新任务的适应 | Few-shot / zero-shot | 需要重新定义 reset 分布并重训 |
| Contact-rich 能力 | 弱 (大多依赖 waypoint 式控制) | 强 |
| Sim 依赖 | 低 | 高 |

OmniReset 和 Foundation Model 方法代表了两种互补的技术路径. OmniReset 在 contact-rich long-horizon 任务上更强, 但缺乏泛化能力. Foundation Model 方法泛化性好但在精细操作上还不够. 论文作者尝试用 Pi-0.5 做 distillation 但没有显著提升, 暗示当前 VLM backbone 对 manipulation 的 fine-grained control 帮助有限.

### 8.3 与 DexCanvas / HandelBot 的对比

| 维度 | DexCanvas (2025) | HandelBot (2026) | OmniReset (2026) |
|------|-----------------|-----------------|------------------|
| 末端执行器 | 灵巧手 | 灵巧手 (钢琴弹奏) | Parallel gripper |
| 方法核心 | Motion tracking + RL | RL + motion planning | Diverse resets + RL |
| Demo 需求 | 人手 demo | MIDI 乐谱 | 无 |
| Reset 策略 | 标准 | 标准 | Diverse (核心创新) |
| Task complexity | 中 (单步) | 高 (长序列, 但结构化) | 高 (contact-rich, 非结构化) |

OmniReset 的 diverse resets 思想原则上可以推广到灵巧手场景, 但 grasp sampling 会更复杂 (多指 grasp 空间远大于 parallel gripper).

---

## 附录: 代码结构概览

```
UWLab/source/
├── uwlab/                  # Core framework extensions
│   ├── controllers/        # Differential IK controller
│   ├── sim/                # Mesh converter, materials
│   ├── managers/           # Data manager
│   └── devices/            # Teleop, Rokoko glove
├── uwlab_assets/           # Robot URDF/USD definitions
├── uwlab_rl/               # RL algorithm extensions
│   ├── rsl_rl/             # RSL-RL config (gSDE, DAgger support)
│   ├── skrl/               # SKRL extensions
│   └── wrappers/           # Diffusion policy wrapper
└── uwlab_tasks/            # Task definitions
    └── manager_based/
        └── manipulation/
            └── omnireset/  # <-- OmniReset 核心实现
                ├── assembly_keypoints.py    # Offset 计算
                ├── config/ur5e_robotiq_2f85/
                │   ├── rl_state_cfg.py      # 主训练配置 (scene, obs, reward, events)
                │   ├── reset_states_cfg.py  # Reset state 生成配置
                │   ├── actions.py           # OSC 动作空间
                │   ├── agents/rsl_rl_cfg.py # PPO + gSDE 超参
                │   └── data_collection_rgb_cfg.py  # RGB 数据收集
                └── mdp/
                    ├── events.py            # MultiResetManager (核心: diverse resets)
                    ├── rewards.py           # Task-agnostic rewards
                    ├── observations.py      # Actor/Critic observations
                    ├── terminations.py      # OBB collision + abnormal state
                    ├── commands.py          # TaskCommand
                    ├── collision_analyzer.py # 碰撞分析
                    └── success_monitor.py   # 成功率监控
```
