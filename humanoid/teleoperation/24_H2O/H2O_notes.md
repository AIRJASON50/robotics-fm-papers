# H2O / OmniH2O 论文分析笔记

> H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation (arxiv 2403.04436, 2024.03)
> OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning (arxiv 2406.08858, 2024.06)
> 代码: `/home/l/ws/doc/paper/humanoid/24_H2O/human2humanoid/`

---

## 1. Core Problem

H2O 和 OmniH2O 解决的核心问题是:**如何用低成本输入设备实现全尺寸 humanoid 的实时全身遥操作 (whole-body teleoperation)**。

具体挑战:
- **embodiment gap**: 人类与 humanoid 的关节结构、自由度、动力学差异巨大，人类动作无法直接映射
- **sim-to-real gap**: 仿真训练的策略需要 zero-shot 迁移到真实机器人
- **输入信号的稀疏性与噪声**: RGB camera 的 3D pose estimation 有延迟和误差；VR headset 只提供 3 个点 (head + 2 hands) 的稀疏信号
- **全身协调**: locomotion 和 upper body manipulation 需要耦合控制，不能简单分解

H2O (2024.03) 首次实现了基于 RL 的实时全身遥操作框架，但**依赖 MoCap 获取 global linear velocity**，且只能做简单的移动任务。OmniH2O (2024.06) 是其直接后继，解决了上述限制:
- 去除了对 MoCap 的依赖 (用 history 替代 global linear velocity)
- 增加了灵巧手控制
- 提出 teacher-student distillation 架构
- 支持 VR / RGB / language / GPT-4o 等多种输入源
- 发布了首个 humanoid whole-body control dataset (OmniH2O-6)

---

## 2. Method Overview

### 2.1 整体 Pipeline

两篇论文共享同一个基本流程:

```
Human Motion Data (AMASS)
        |
        v
[Retargeting] SMPL -> H1 joint space (gradient-based optimization)
        |
        v
[Sim-to-Data] privileged imitator 过滤不可行动作
        |
        v
[RL Training] teacher policy (privileged obs) -> student policy (real-world obs)
        |
        v
[Sim-to-Real] zero-shot transfer to Unitree H1
```

### 2.2 H2O Pipeline

1. **Motion Retargeting**: 用 gradient-based optimization 将 SMPL 动作映射到 H1 关节空间。优化目标是 11 个关键点位置误差最小化 (pelvis, knees, ankles, shoulders, elbows, hands)
2. **Sim-to-Data**: 训练 privileged imitator 在仿真中过滤不可行动作，生成 cleaned dataset
3. **Policy Training**: 单阶段 RL (PPO) 训练 motion imitation policy，直接在 sim-to-real 条件下训练
4. **实时遥操作**: RGB camera -> HybrIK pose estimation -> motion goal -> policy -> H1

观测空间 (H2O):
- proprioception: dof_pos (19), dof_vel (19), base_lin_vel (3), base_ang_vel (3), projected_gravity (3)
- task: delta_base_pos (2), delta_heading (1), ref_dof_pos (19), ref_dof_vel (19), ref_base_vel (3), ref_base_ang_vel (3), ref_base_gravity (3)
- last_action (19)
- **需要 MoCap 提供 base_lin_vel**

### 2.3 OmniH2O Pipeline

关键改进 -- teacher-student distillation:

**Teacher Policy** (privileged):
- 输入全部 rigid body position, orientation, velocity, angular velocity
- Goal state 包含 reference 和 current state 之间的 one-frame difference
- 训练目标: 大规模 motion tracking (AMASS retargeted + augmented)

**Student Policy** (real-world可部署):
- proprioception: 25-step history 的 dof_pos, dof_vel, root_angular_vel, gravity, last_action
- task: keypoint position difference in base frame (6-8 keypoints), ref velocity
- **不需要 global linear velocity** -- 通过 history 隐式学习
- 使用 DAgger framework distill from teacher

**Distillation Loss:**

$$\mathcal{L} = \| a_t^{privileged} - a_t \|_2^2$$

### 2.4 Key Formulas

**Reward Function** (motion imitation, exponential kernel):

$$r_{pos} = \exp\left(-\frac{\| p_{ref} - p_{sim} \|^2}{\sigma_{pos}}\right)$$

$$r_{rot} = \exp\left(-\frac{\| \theta_{ref} - \theta_{sim} \|^2}{\sigma_{rot}}\right)$$

$$r_{vel} = \exp\left(-\frac{\| v_{ref} - v_{sim} \|^2}{\sigma_{vel}}\right)$$

**Max Feet Height Reward** (OmniH2O 的关键创新):

$$r_{feet} = \text{clamp\_min}(h_{desired} - h_{max\_this\_step}, 0)$$

当脚从空中着地时给予惩罚，如果这次抬脚的最大高度没达到阈值 (默认 0.25m)。这鼓励 "要么站稳不动，要么大步迈出"，解决了 H2O 中 humanoid 原地小步乱踩的问题。

---

## 3. Key Designs

### 3.1 Motion Data Augmentation -- Standing/Squatting Stable Variants

OmniH2O 发现 AMASS 数据集中大部分动作是移动的，导致策略**不擅长静止站立**。解决方案: 对每个动作序列生成一个 "stable" 版本 -- 固定 root position 和 lower body 为站立或蹲下姿态，只保留上半身动作。

直觉: manipulation 任务中，机器人大部分时间需要站稳或蹲下来操作，lower body 要稳定。通过 bias 训练数据分布，让策略充分学习静止平衡。

代码实现: retargeting 阶段直接在数据处理层面完成，见 `scripts/data_process/grad_fit_h1.py`。

### 3.2 Teacher-Student Distillation with History (OmniH2O)

这是从 H2O 到 OmniH2O 最关键的架构升级。

**为什么需要 teacher-student:**
- real-world 只能获得有限传感器信息 (joint encoder, IMU, VR 3 点)
- 直接用稀疏输入训练 RL 很难收敛
- privileged teacher 有完整 rigid body 状态，更容易找到好的策略

**为什么 history 能替代 global linear velocity:**
- 连续 25 帧的 joint position + joint velocity + angular velocity 包含了足够的动力学信息
- 策略可以从 history 中隐式估计出 root velocity (类似 state estimation)
- 这消除了对 MoCap 的依赖，使 in-the-wild 部署成为可能

代码中默认 obs version 是 `v-teleop-extend-max` (见 `legged_gym/legged_gym/cfg/motion/motion_teleop.yaml`):
```
obs = [dof_pos(19), dof_vel(19), base_vel(3), base_ang_vel(3), base_gravity(3),
       task_obs(keypoint_diff + ref_vel), last_action(19)]
```

但 student distillation 版本 (`v-teleop-extend-max_no_vel`) 去掉了 base_vel，加入了 short history。

### 3.3 Max Feet Height per Step Reward

这是 OmniH2O reward design 的核心创新，解决了 humanoid 原地踏步的 (feet stomping) 问题。

之前的做法 (H2O 以及其他工作如 Expressive WBC) 用 feet air time reward 鼓励大步走:
```python
rew_airTime = (feet_air_time - 0.25) * first_contact
```

问题: 这种 reward 在静止站立时仍然鼓励抬脚，导致 stomping。

OmniH2O 的 max feet height reward:
```python
# From legged_robot.py:3985
feet_air_max_height = max(feet_air_max_height, feet_height)
rew = clamp_min(desired_height - feet_air_max_height, 0) * from_air_to_contact
```

直觉: **惩罚每步的最大抬脚高度不够**。如果机器人要走，就要抬足够高 (>= 0.25m)；如果不需要走，就别抬脚。配合 curriculum 使用效果最佳。

---

## 4. Experiments

### 4.1 Simulation Motion Tracking

| Method | Success Rate | Global MPJPE (mm) | MPJPE (mm) |
|--------|-------------|-------------------|------------|
| H2O Privileged | 95.1% | - | - |
| H2O (full) | 77.3% | - | - |
| H2O w/o sim-to-data | 68.0% | - | - |
| H2O reduced state | 64.3% | - | - |

OmniH2O 在 14k 序列的 AMASS 上评测 (Table 1 in OmniH2O paper):
- OmniH2O teacher > H2O baseline, 尤其在 success rate 和 global MPJPE 上
- Teacher-student distillation 后性能略有下降但仍可部署

### 4.2 关键 Ablation 发现

**数据集大小**: 即使只用 0.1% 数据 (H2O Table IV)，由于充分的 domain randomization，仍能达到令人惊讶的高 success rate。但更大数据集持续提升性能。

**Sim-to-Data 过滤**: 过滤不可行动作显著提升 success rate (从 68% 到 77.3%)，即使训练数据量更少。不可行动作会 "浪费" RL 的学习资源。

**Observation Design**: H2O 中比较了 full vs reduced state，发现包含更多物理信息的 motion goal (velocity, angular velocity) 显著帮助泛化。

**Standing Data Augmentation** (OmniH2O): 加入 standing/squatting variants 后，静止时的稳定性大幅提升。

**History 替代 Linear Velocity**: OmniH2O 证明 25-step history 可以完全替代 MoCap 提供的 global linear velocity，且鲁棒性更好。

### 4.3 Real-world Results

H2O: 在 CMU Wean Hall 1334 实现了 walking, back jumping, kicking, turning, waving, pushing, boxing 等动态全身遥操作。

OmniH2O: 扩展到灵巧操作场景 -- 拍球、浇花、写毛笔字、蹲下拾取、拳击、递篮子等。在室外 (草地、斜坡、碎石路) 证明了鲁棒性。

### 4.4 Autonomy (OmniH2O)

- **GPT-4o 集成**: 头部摄像头 -> GPT-4o 选择 motion primitive -> OmniH2O 执行
- **OmniH2O-6 数据集**: 6 个任务，约 40 分钟 real-world demonstration，训练 diffusion policy 实现自主执行

---

## 5. Related Work Analysis

H2O/OmniH2O 在领域发展中的定位:

| 方向 | 代表工作 | H2O/OmniH2O 的改进 |
|------|----------|-------------------|
| Humanoid Locomotion | Radosavovic 2024, Li 2024 | 从仅 locomotion 扩展到 whole-body control |
| Motion Imitation (graphics) | PHC, PULSE, DeepMimic | 从仿真角色扩展到真实机器人 |
| VR-based Control | QuestSim, QuestEnvSim | 从动画角色扩展到 real robot manipulation |
| Humanoid Teleoperation (传统) | Exoskeleton, MoCap-based | 从昂贵设备简化到 RGB camera / VR headset |
| Expressive WBC | Cheng 2024 | 从 velocity tracking 扩展到 motion imitation |

**独特贡献**: H2O/OmniH2O 是第一个将 learning-based motion imitation 与 sim-to-real 和 real-time teleoperation 三者结合的系统。PHC 只在仿真中追踪动作；Expressive WBC 只做 velocity tracking。

---

## 6. Limitations & Future Directions

### 论文自述的局限

1. **Representation Gap**: motion goal 的表达力与 RL 的 sample efficiency 之间存在 trade-off
2. **Embodiment Gap**: 缺乏系统性的算法来识别哪些动作对特定机器人是可行的
3. **Sim-to-Real Gap**: 过度 regularization/randomization 会抑制动作学习，最优平衡点未知
4. **延迟**: RGB pose estimation 的延迟和误差不可避免
5. **OmniH2O**: VR 3 点输入有歧义性 (同样的 head+hands 位置可对应不同的 elbow/lower body 姿态)

### 从代码推断的局限

6. **单一机器人**: 代码完全针对 Unitree H1 硬编码 (19 DOF, 特定 PD gains, 特定关节限位)
7. **Teacher 需要完整 rigid body state**: 这意味着 teacher 训练依赖 IsaacGym 的 privileged 信息，无法用于 real-world online adaptation
8. **手部控制是独立的 IK**: dexterous hand 控制通过 VR 的 hand pose + 独立 IK 实现，并没有纳入 RL 策略训练
9. **没有 fall recovery**: policy 在跌倒后没有 get-up 能力 (虽然 phc 模块中有 `humanoid_im_getup.py`，但未在 OmniH2O student 中使用)
10. **Motion data 预处理 pipeline 复杂**: AMASS -> SMPL -> gradient optimization -> filter -> augment，中间有很多手动步骤

---

## 7. Paper vs Code Discrepancies

这是最关键的部分 -- 论文没有提到但代码实际实现的内容。

### 7.1 Observation Version 多样性

论文只描述了一种 observation design，但代码中实现了 **10+ 种 obs version** (见 `legged_robot.py:610-1200`):

| obs version | 特点 | num_obs |
|-------------|------|---------|
| `v1` | full self_obs + task_obs | 大 |
| `v-min` | 最小: dof + ref + heading | 小 |
| `v-min2` | v-min 去掉 ref_vel/ref_ang_vel | 更小 |
| `v-teleop` | + keypoint pos diff | 87 |
| `v-teleop-clean` | 去掉 delta_base_pos/heading | 84 |
| `v-teleop-superclean` | 只有 dof + task | 75 |
| `v-teleop-extend` | + extended hand keypoints | 90 |
| `v-teleop-extend-max` | **默认**: + ref velocity in obs | 138 |
| `v-teleop-extend-max-nolinvel` | 去掉 base_lin_vel | 135 |
| `v-teleop-extend-max_no_vel` | 可加 short history | variable |
| `v-teleop-extend-vr-max` | VR 3-point specific | variable |

默认使用 `v-teleop-extend-max`，这在论文中没有详细区分。

### 7.2 Extended Body (Virtual Keypoints)

代码中大量使用 "extend body" 概念 -- 在 elbow link 末端虚拟延伸 0.3m 生成 "hand link"，在 pelvis 上方虚拟延伸 0.75m 生成 "head link" (见 `torch_h1_humanoid_batch.py:42-66`)。这些虚拟关键点在论文中没有详细说明，但在代码中是 reward 计算和 observation 的核心组成部分。

### 7.3 Domain Randomization 细节远超论文描述

代码中的 domain randomization 非常丰富 (见 `cfg/domain_rand/domain_rand_teleop.yaml`):

| Randomization | Range | 论文提及? |
|---------------|-------|----------|
| friction | [-0.6, 1.2] | 是 |
| link mass | [0.7, 1.3]x | 是 |
| base CoM | +/-0.1m xyz | 否 |
| PD gain (kp, kd) | [0.75, 1.25]x | 否 |
| torque RFI (Random Force Injection) | 0.1 | 否 |
| control delay | [0-3] steps (0-60ms) | 否 |
| motion ref xyz noise | +/-0.02m xy, +/-0.1m z | 否 |
| motion package loss | 1-10 steps freeze | 否 |
| push robots | 1.0 m/s, every 5s | 部分 |
| born offset (初始位置偏移) | curriculum | 否 |
| born heading randomization | curriculum | 否 |

特别值得注意的是 **motion package loss** -- 模拟通信丢包，在一段时间内 freeze motion reference 不更新，这对真实遥操作场景非常重要但论文没提。

### 7.4 Reward Curriculum

代码实现了多维度的 curriculum learning:

1. **sigma curriculum** (`rewards.sigma_curriculum`): 逐渐收紧 body position tracking 的 sigma，从 1.0 降到 0.02
2. **penalty curriculum** (`rewards.penalty_curriculum`): 逐渐增大 regularization penalty 的权重，从 0.25 到 1.0
3. **motion curriculum**: episode length 达标则升级到更难的动作
4. **born offset curriculum**: 初始位置偏移逐渐增大
5. **born heading curriculum**: 初始朝向偏移逐渐增大

论文只简要提到 "carefully designed curriculum"，但没有给出这些细节。

### 7.5 上下半身分离的 Reward 权重

代码中 reward 对上下半身有不同的权重和 sigma:
- upper body position sigma: 0.03 (很严格)
- lower body position sigma: 0.5 (宽松)
- upper action_rate penalty: -0.625
- lower action_rate penalty: -3.0 (6x 惩罚!)
- hip_pitch joint weight: 2.0, 其他 lower body: 0.5

这意味着策略**优先保证上半身精确跟踪**，对下半身给予更大的自由度来保持平衡。这个设计在论文中没有明确说明。

### 7.6 VR 3-Point Specific Rewards

代码中有专门的 VR 3-point reward (`teleop_body_position_vr_3keypoints`，scale=50)，对 head 和 two hands 使用更小的 sigma (0.03) 进行精确跟踪。这是 VR 遥操作场景的特化设计。

### 7.7 ActorCriticPULSE (VAE-based Policy)

代码中包含 `ActorCriticPULSE` -- 一个带 VAE encoder-decoder 结构的策略网络 (见 `rsl_rl/rsl_rl/modules/actor_critic_pulse.py`)。这个模块包含:
- Encoder: obs -> mu, logvar
- Decoder: z -> action
- Prior: self_obs -> mu, logvar

这在两篇论文中都没有提到，但代码中已经实现并可配置使用 (通过 `ppo_pulse.yaml`)。这可能用于后续的 latent space motion planning。

### 7.8 Velocity Estimator

代码中包含 `VelocityEstimator` 和 `VelocityEstimatorGRU` 模块 (见 `rsl_rl/rsl_rl/modules/`)，用于从 proprioception history 估计 root linear velocity。这在 `legged_robot.py:74-86` 中有训练逻辑。论文提到用 history 替代 velocity，但没有提到专门训练过 velocity estimator。

### 7.9 Motion Package Loss (通信丢包模拟)

见 `domain_rand_teleop.yaml:45-47`:
```yaml
motion_package_loss: False  # 默认关闭
package_loss_range: [1, 10]  # freeze 1-10 steps
package_loss_interval_s: 2   # 每 2 秒触发一次
```

当 package loss 触发时，motion reference 会 freeze (不更新)，模拟真实通信中的丢包场景。代码中有完整的实现 (`legged_robot.py:629-642`, `4118-4122`)，但论文未提及。

### 7.10 Zero-Out-Far 策略

当 robot 离 reference motion 太远时 (> `close_distance`=0.25m)，代码会:
1. 将 upper body keypoints 的 reference 设为当前位置 (不再追踪)
2. 只保留 root 的方向引导
3. 当距离 > `far_distance`=5m 时，将目标缩放到最大 5m

这是一个非常实际的 fail-safe 机制，防止策略在迷失时做出危险动作。论文中没有提到。

---

## 8. Cross-Paper Comparison

### 8.1 与 SONIC 的比较

| 维度 | H2O/OmniH2O | SONIC |
|------|-------------|-------|
| 控制目标 | Motion imitation (全身 pose tracking) | Multi-modal command (语言+视觉) |
| 输入 | Kinematic pose (VR/RGB/language) | Language instruction + visual context |
| 训练范式 | RL + distillation | RL + foundation model |
| Retargeting | SMPL -> H1 gradient optimization | 不涉及 human motion |
| Dexterous Hand | 独立 IK 控制 (未入 RL) | 未涉及 |
| 数据集 | AMASS retargeted | Task-specific |
| 部署方式 | Zero-shot sim-to-real | Zero-shot sim-to-real |

关键差异: H2O/OmniH2O 是 motion-centric (跟踪人类动作)，SONIC 是 task-centric (完成语言指定任务)。OmniH2O 通过 GPT-4o 集成可以 bridge 两者。

### 8.2 与 HDMI 的比较

| 维度 | OmniH2O | HDMI |
|------|---------|------|
| Teacher-Student | DAgger distillation | Teacher-student (详细未知) |
| Motion Data | AMASS + standing augmentation | Large-scale motion data |
| Observation | History-based (25 steps) | 可能用 state estimation |
| Hand Control | 独立 IK | 可能集成 |
| Real-world | Unitree H1 | Specific humanoid |

### 8.3 与 OmniRetarget 的比较

| 维度 | OmniH2O | OmniRetarget |
|------|---------|--------------|
| Retargeting 方法 | Gradient-based fixed mapping (SMPL -> H1 11 关键点) | Universal retargeting (多种 embodiment) |
| 支持的机器人 | 只有 H1 | 多种 humanoid |
| DoF 处理 | 单轴旋转约束 (H1_ROTATION_AXIS) | 更通用的关节约束 |
| Data Pipeline | 离线 batch 处理 | 可能支持 online |

OmniRetarget 在 retargeting 泛化性上优于 H2O，但 H2O 的端到端 teleoperation pipeline 更完整。

### 8.4 与 PHC 的比较

PHC (Perpetual Humanoid Control) 是 H2O 的直接前身 (同一团队 Zhengyi Luo):

| 维度 | OmniH2O | PHC |
|------|---------|-----|
| 目标 | Real robot teleoperation | Simulated avatar control |
| 机器人 | Unitree H1 (19 DOF) | SMPL humanoid |
| Sim-to-Real | 完整 pipeline | 仅 simulation |
| Motion Library | `MotionLibH1` | `MotionLibSMPL` |
| Domain Rand | 大量 (10+ types) | 少量 |
| Recovery | 无 get-up | 有 get-up (perpetual) |

代码中直接复用了 PHC 的代码结构 (`phc/` 目录)，特别是 `HumanoidIm`, `HumanoidAMP`, `MotionLibBase` 等核心类。H2O/OmniH2O 本质上是 "PHC for real robots"。

### 8.5 与 bh_motion_track 的比较

| 维度 | OmniH2O | bh_motion_track |
|------|---------|-----------------|
| 框架 | IsaacGym + rsl_rl | 待确认 |
| Teacher-Student | DAgger | 待确认 |
| Retargeting | SMPL -> H1 (gradient opt) | 待确认 |
| Reward Design | 上下半身分权, max feet height | 待确认 |
| Obs Design | 多版本 (10+) | 待确认 |
| Domain Rand | 极其丰富 | 待确认 |

### 8.6 技术要素对比总表

| 技术要素 | H2O | OmniH2O | SONIC | PHC | OmniRetarget |
|----------|-----|---------|-------|-----|--------------|
| Whole-body Control | Yes | Yes | Task-specific | Yes (sim) | Yes (retarget only) |
| Teacher-Student | No | DAgger | - | No | - |
| History Obs | No | 25-step | - | No | - |
| Need MoCap at deploy | Yes | No | No | N/A | N/A |
| Dexterous Hand | No | IK-based | - | No | - |
| Standing Augmentation | No | Yes | - | No | - |
| Max Feet Height Reward | No | Yes | - | No | - |
| Sigma/Penalty Curriculum | No | Yes | - | No | - |
| VR Support | No | Yes | No | No | - |
| Autonomy (LfD) | No | Yes (diffusion) | Yes | No | - |
| Open Dataset | No | OmniH2O-6 | - | No | - |

---

## Code Structure Summary

代码库主要包含三个核心模块:

```
human2humanoid/
├── legged_gym/         # OmniH2O student policy 训练 (主体)
│   ├── envs/base/
│   │   ├── legged_robot.py       # 核心 env (4100+ 行, 10+ obs versions, 40+ rewards)
│   │   └── legged_robot_config.py # 基础配置
│   ├── envs/h1/
│   │   └── h1_teleop_config.py   # H1 specific config (PD gains, joint limits)
│   ├── cfg/                       # Hydra config files
│   │   ├── rewards/rewards_teleop_omnih2o_teacher.yaml  # OmniH2O teacher reward
│   │   ├── domain_rand/domain_rand_teleop.yaml          # Domain randomization
│   │   ├── train/ppo_teleop.yaml                        # PPO + distill config
│   │   └── motion/motion_teleop.yaml                    # Motion config
│   └── scripts/
│       ├── train_hydra.py        # Training entry point
│       └── play_hydra.py         # Evaluation entry point
├── phc/                # PHC-based teacher policy (from PHC project)
│   ├── env/tasks/
│   │   ├── humanoid_im.py        # Motion imitation task (teacher)
│   │   └── humanoid_amp.py       # AMP-based humanoid
│   ├── utils/
│   │   ├── motion_lib_h1.py      # H1 motion library (FK, interpolation)
│   │   └── torch_h1_humanoid_batch.py  # H1 FK batch computation
│   └── learning/                  # RL training modules (from rl_games)
├── rsl_rl/             # RSL RL library (modified)
│   ├── modules/
│   │   ├── actor_critic.py        # Standard MLP policy
│   │   ├── actor_critic_pulse.py  # VAE-based policy (未在论文中提及)
│   │   └── velocity_estimator.py  # Velocity estimation from history
│   ├── runners/on_policy_runner.py # PPO + DAgger runner
│   └── algorithms/ppo.py
├── scripts/
│   └── data_process/
│       └── grad_fit_h1.py         # SMPL -> H1 retargeting (gradient optimization)
└── hardware_code/                 # Real robot deployment code
```
