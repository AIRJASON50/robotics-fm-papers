# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

> arXiv: 2602.16863 | Code: github.com/tylerlum/simtoolreal
> Authors: Kushal Kedia (Cornell), Tyler Ga Wei Lum (Stanford), Jeannette Bohg, C. Karen Liu (Stanford)

---

## 1. 论文概述

### 核心问题

灵巧手的工具使用 (tool manipulation) 需要组合多种技能：从平面抓取细长物体、手内重定向 (in-hand reorientation)、在力交互中保持稳定抓取。现有方法要么需要为每个工具/任务做大量工程化 (per-task reward engineering)，要么依赖遥操作采集 (人-机对应鸿沟导致数据质量差)。

### 核心 Insight

**所有工具使用任务都可以统一为"将工具依次移动到一系列目标 6D 位姿"。** 只需训练一个通用的 goal-conditioned pose-reaching 策略，无需 per-task reward。

### 三大贡献

1. **SimToolReal 框架**: 在程序化生成的 primitive 物体上训练单一策略，zero-shot 部署到未见工具
2. **Object-Centric 感知**: 基于 SAM 3D + FoundationPose，从人类视频提取目标位姿序列
3. **DexToolBench**: 24 任务 / 12 物体 / 6 工具类别的真实世界 benchmark

---

## 2. 系统架构

### 硬件

| 组件 | 型号 | 参数 |
|------|------|------|
| 机械臂 | KUKA iiwa14 | 7 DOF |
| 灵巧手 | SharPa (左手) | 22 DOF, 5 指 |
| 相机 | ZED 1 立体相机 | RGB-D, 30Hz |
| 总 DOF | — | 29 (7 arm + 22 hand) |

### SharPa 手指关节分布

```
Thumb:  CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP     (5 DOF)
Index:  MCP_FE, MCP_AA, PIP, DIP                (4 DOF)
Middle: MCP_FE, MCP_AA, PIP, DIP                (4 DOF)
Ring:   MCP_FE, MCP_AA, PIP, DIP                (4 DOF)
Pinky:  CMC, MCP_FE, MCP_AA, PIP, DIP           (5 DOF)
```

---

## 3. 核心方法

### 3.1 Object-Centric Policy 设计

策略形式: `a_t = pi(s_t, o_t, phi, g)`

| 符号 | 含义 | 维度 |
|------|------|------|
| `s_t` | 本体感受 (joint pos/vel/prev_target + palm pose + fingertips) | ~80 |
| `o_t` | 物体位姿 (4 keypoints relative to palm) | 16 |
| `phi` | 物体描述符 (grasp bounding box scale) | 3 |
| `g` | 目标 (keypoint-to-goal error) | 12 |

**关键设计**: 不使用绝对坐标，全部使用相对表示 (relative to palm)，提高跨物体/跨坐标系泛化。

### 3.2 观测空间 (140D Actor / 扩展 Critic)

| 观测项 | 维度 | 说明 |
|--------|------|------|
| joint_pos (normalized) | 29 | 关节角 unscale 到 [-1,1] |
| joint_vel | 29 | 关节角速度 |
| prev_action_targets | 29 | 上一步目标 |
| palm_pos | 3 | 手掌位置 (world) |
| palm_rot | 4 | 手掌朝向 (quat xyzw) |
| object_rot | 4 | 物体朝向 (quat xyzw) |
| fingertip_pos_rel_palm | 15 | 5 指尖相对手掌 (5x3) |
| keypoints_rel_palm | 12 | 4 物体关键点相对手掌 (4x3) |
| keypoints_rel_goal | 12 | 4 关键点到目标误差 (4x3) |
| object_scales | 3 | 物体尺寸 (x,y,z) |

**Asymmetric Critic 额外特权信息**: palm_vel(6), object_vel(6), closest_fingertip_dist(5), lifted_object(1), progress(1), successes(1), reward(1)

### 3.3 动作空间 (29D)

| 范围 | 维度 | 控制方式 |
|------|------|----------|
| Arm (0:7) | 7 | **Delta 增量**: `target = prev + k_arm * action`, k_arm=0.025 |
| Hand (7:29) | 22 | **Absolute**: action 仿射映射到关节极限范围 |

两者都做 EMA 平滑: `alpha = 0.1` (强平滑)

### 3.4 Keypoint 表示

物体用 4 个关键点 (bounding box 的 4 个对角顶点) 编码 6D 位姿 + 尺寸:
- 固定 keypoint scale: `[0.141, 0.03025, 0.0271]` (m)
- 奖励专用 keypoint scale: `[0.14, 0.03025, 0.0271]` (x 方向大，对 pitch/yaw 更敏感)
- 距离度量: 4 个关键点的 **max distance**

### 3.5 奖励函数 (三阶段)

```
r = r_smooth + r_grasp + I_grasped * r_goal
```

| 阶段 | 奖励项 | Scale | 触发条件 |
|------|--------|-------|----------|
| 平滑 | arm_actions_penalty | -0.03 | 始终 |
| 平滑 | hand_actions_penalty | -0.003 | 始终 |
| 抓取 | fingertip_delta_rew | 50 | 物体未举起 |
| 抓取 | lifting_rew | 20 | 物体未举起 |
| 抓取 | lift_bonus | 300 | 物体超过阈值(0.15m)时一次性 |
| 目标 | keypoint_rew (progress) | 200 | 物体已举起 |
| 目标 | bonus_rew (success) | 1000 | keypoint dist < tolerance |

**Tolerance Curriculum**: 0.075 → 0.01，衰减系数 0.9

### 3.6 训练配置

| 参数 | 值 |
|------|-----|
| 仿真器 | IsaacGym (GPU 并行) |
| 并行环境数 | 24,576 |
| RL 算法 | PPO + SAPG (6 blocks) |
| 网络 | LSTM(1024) + MLP[1024,1024,512,512] |
| Actor-Critic | Asymmetric (Critic 有特权信息) |
| gamma / tau | 0.99 / 0.95 |
| lr | 1e-4 (adaptive) |
| horizon | 16 steps |
| mixed_precision | True |

### 3.7 程序化物体生成

每个工具 = **handle (柄)** + **head (头)**, 每部分随机选 cuboid 或 capsule:

| 部分 | 尺寸范围 | 密度 |
|------|---------|------|
| Handle | L:5-30cm, W/H:1-4cm | 300-600 kg/m^3 |
| Head | L:1-15cm, W/H:0.5-12cm | 300-2000 kg/m^3 |

### 3.8 Domain Randomization

| 类型 | 详情 |
|------|------|
| 物理参数 | DOF damping/stiffness/friction ×[0.7,1.3], 质量 ×[0.7,1.3] |
| 观测延迟 | obs max 3 steps, action max 3 steps |
| 物体状态噪声 | 位置 std=0.01m, 旋转 5 deg, 额外 max 10 steps 延迟 |
| 外力扰动 | force=20N, torque=2Nm, 随机脉冲 (仅举起后) |
| 重力 | additive Gaussian std=0.3 |
| 桌面高度 | 随机化 |

---

## 4. 代码仓库结构

```
simtoolreal/
├── isaacgymenvs/                    # 核心 RL 训练
│   ├── cfg/task/SimToolReal.yaml    # 主任务配置
│   ├── cfg/train/SimToolRealLSTMAsymmetricPPO.yaml  # 训练超参
│   ├── tasks/simtoolreal/env.py     # 环境定义 (5314 行, 核心)
│   ├── utils/observation_action_utils_sharpa.py  # 观测/动作定义
│   └── launch_training.py           # 训练启动
├── deployment/                      # 部署 (ROS 节点)
│   ├── rl_policy_node.py            # 策略推理节点 (60Hz)
│   ├── goal_pose_node.py            # 目标位姿管理
│   ├── sharpa_node.py               # SharPa 手控制
│   └── rl_player.py                 # 策略加载器
├── dextoolbench/                    # Benchmark
│   ├── metadata.py                  # 6类/12物体/24任务定义
│   ├── objects.py                   # 物体 URDF + scale
│   ├── eval.py                      # 评估脚本
│   └── trajectories/               # 目标位姿序列 JSON
├── assets/urdf/                     # KUKA + SharPa URDF
├── baselines/                       # Kinematic retargeting / Fixed grasp
├── rl_games/                        # 自定义 rl_games (含 SAPG)
└── recorded_data/                   # 录制数据接口
```

---

## 5. 实验结果

### Zero-Shot 真实世界 (120 rollouts)

- **Eraser 最佳**: 主要靠平移，无需复杂 in-hand rotation
- **Screwdriver 最难**: 需要功能性重定向 + 持续旋转

### 失败模式分析

| 原因 | 占比 |
|------|------|
| 位姿追踪丢失 (遮挡/对称/低对比度) | 43.7% |
| 物体掉落 | 34.5% |
| In-hand rotation 不完整 | 18.2% |
| 抓取失败 | 3.6% |

### 对比实验

| 方法 | Task Progress |
|------|--------------|
| SimToolReal | **最高** (比其他高 37%) |
| Fixed Grasp | 中等 (open-loop 无法纠错) |
| Kinematic Retargeting | 完全失败 (无法建立稳定抓取) |

### vs Specialist 策略 (仿真)

- 在训练配置上 SimToolReal **匹配** specialist
- 在 unseen object/trajectory 上 SimToolReal **远超** specialist (specialist 严重过拟合)

---

## 6. DexToolBench

| 类别 | 物体 | 任务 | 关键技能 |
|------|------|------|----------|
| Hammer | Claw / Mallet | Swing Down/Side | 抓取 + 90° 旋转 + 挥击 |
| Marker | Sharpie / Staples | Draw Smile / Write C | 抓取细物 + 书写 |
| Eraser | Flat / Handle | Wipe Smile / Wipe C | 抓取 + 擦除运动 |
| Brush | Blue / Red | Sweep Forward/Right | 抓取 + 90° 旋转 + 扫 |
| Spatula | Spoon / Flat | Serve Plate / Flip Over | 铲/翻 (Flip 需 180°) |
| Screwdriver | Long / Short | Spin V/H | 90° 旋转 + 360° 自旋 |

评估指标: **Task Progress** = 成功达到的 waypoint 占比 (闭环, 容差 2cm)

---

## 7. 局限性

1. **不保证功能性完成**: 跟踪位姿 ≠ 完成功能 (如锤击力度)
2. **环境盲**: 不感知周围障碍物
3. **刚体假设**: 不适用于铰接/柔性工具
4. **固定目标序列**: 不根据执行情况动态重规划
5. **位姿追踪瓶颈**: 遮挡/旋转对称/低对比度 → 43.7% 失败

---

## 8. 对我的工作的启发与帮助

### 8.1 背景对照

| 维度 | SimToolReal | 我的工作 (WujiHand + DexCanvas) |
|------|-------------|-------------------------------|
| 灵巧手 | SharPa 22-DOF (5指) | WujiHand 20-DOF (5指) |
| 手臂 | KUKA iiwa14 7-DOF | 6-DOF 虚拟 wrist (3 slide + 3 hinge) |
| 仿真器 | IsaacGym (GPU) | MuJoCo/MJX (GPU) |
| 数据源 | 人类 RGB-D 视频 → 6D 位姿轨迹 | DexCanvas MANO → retarget |
| 物体 | 工具 (hammer, brush, etc.) | Boob Cube (2-body, hinge) |
| 任务 | 工具使用 (tool manipulation) | In-hand manipulation / contact-rich |
| 控制 | PD position (arm delta + hand absolute) | PD position (wrist delta + finger residual) |

### 8.2 直接可借鉴的技术

#### (1) Object-Centric 观测设计

SimToolReal 的核心创新 — 用 **4 个 keypoint 相对手掌的位置** 代替物体绝对坐标。

**对你的价值**: 你当前的 obs (130D) 已包含 object_pose、fingertip 等，但如果要泛化到不同物体 (比如不同大小的 cube、不同形状的物体)，可以参考他们的做法:
- 将 cube 位姿编码为 keypoints relative to palm
- 将 face_hinge 误差编码为 relative 表示
- 加入 `object_scale` 参数使策略对物体尺寸具备感知

#### (2) Asymmetric Actor-Critic

他们的 Critic 额外拥有:
- 无噪声的物体速度 (ground truth)
- 累积进度 / 成功次数
- 即时奖励值

**对你的价值**: 你的 motion tracking RL 同样是部分可观测的 (real-world 无法获得精确物体速度)。给 Critic 喂 GT 物体速度和累积进度指标，可以让价值估计更准确，加速训练。

#### (3) 分阶段奖励 + Tolerance Curriculum

```
r_approach → r_lift → r_goal (with progress tracking)
tolerance: 0.075 → 0.01 (decay 0.9)
```

**对你的价值**: 你当前奖励设计中已有 finger_tracking、wrist_tracking、face_hinge_tracking 等项。可以借鉴:
- **Progress-based reward** (只奖励距离减小，不是距离本身) — 比 L2 距离更利于训练
- **Tolerance curriculum** — 前期宽松容忍度加速探索，后期收紧提高精度
- **有状态 d* 追踪** (episode 中历史最小距离) — 防止策略反复震荡

#### (4) 动作空间的混合控制

SimToolReal: arm 用 delta 增量 + hand 用 absolute + 强 EMA 平滑 (alpha=0.1)

**对你的价值**: 你当前的架构已经是 wrist delta + finger residual。对照点:
- 他们的 arm EMA=0.1 (很激进的平滑)，你的 finger EMA=0.9 (温和)，wrist 不平滑
- **考虑对 wrist 也做 EMA 平滑** — 可能缓解 6-DOF wrist 的高频振荡
- delta speed scale (k_arm=0.025) 约束了每步最大位移，类似你的 arm velocity limits

#### (5) Domain Randomization 策略

他们的 DR 非常全面:
- **物体状态额外延迟 (max 10 steps)** — 模拟视觉系统延时，这个你可能没有
- **外力扰动 (仅在 grasped 后)** — 条件性扰动比无差别扰动更高效
- **观测/动作 FIFO 延迟队列** — 比简单噪声更接近真实延时特性

#### (6) SAPG 替代 PPO

标准 PPO 在大规模并行 (24k+ envs) 中探索饱和。SAPG 通过维护多策略种群促进探索多样性。

**对你的价值**: 如果你在 MJX 上 scale 到大量并行 env (>8192)，训练出现 plateau，值得尝试 SAPG 或类似的 population-based 探索方法。

### 8.3 架构级启发

#### (7) 从 Motion Tracking → Goal-Conditioned Manipulation

SimToolReal 证明: **在随机目标位姿上训练的 motion tracking 能力，可以直接泛化为 manipulation 能力**。

**对你的意义**: 你当前做的 DexCanvas motion tracking (finger + wrist + object) 本质上就是在训练一种通用的 goal-conditioned 运动能力。SimToolReal 的实验 D 明确验证了 "training reward 与 downstream task progress 高度相关"。
- 你不一定需要为每个 DexCanvas 轨迹设计专门奖励
- 一个好的 motion tracking 策略 + 新的目标轨迹 = 新任务

#### (8) 程序化物体多样性训练 → 泛化

他们不在真实工具 mesh 上训练，而用 cuboid/capsule 的随机组合覆盖工具空间。

**对你的价值**: 如果未来要将 WujiHand 泛化到不同物体:
- 不需要收集每个物体的精确 mesh
- 用 primitive 几何体 + 随机物理参数训练
- 只需提供物体的 bounding box scale 作为 obs 即可

### 8.4 具体可复用的代码/技术

| 技术 | SimToolReal 文件 | 你的对应位置 |
|------|-----------------|-------------|
| Keypoint 表示 | `env.py: compute_kuka_reward()` | 可用于 boob_cube 的 face/base tracking |
| Progress reward | `env.py: keypoint_rew` (d* tracking) | 替换/增强你的 L2 距离奖励 |
| Tolerance curriculum | `env.py: tolerance_curriculum_update()` | 可集成到你的训练 loop |
| Obs/Action delay queue | `env.py: FIFO delay buffer` | 增强 sim-to-real robustness |
| Conditional force perturbation | `env.py: random_force()` | 仅在 contact 后施加 |
| Asymmetric critic | `cfg/train/...PPO.yaml: central_value_config` | 给 critic 喂 GT 速度 |

### 8.5 不直接适用的部分

1. **IsaacGym 环境代码**: 你用 MuJoCo/MJX，不能直接复用 IsaacGym env 代码，但设计思路可以迁移
2. **SharPa 手模型**: 22-DOF vs 你的 20-DOF (WujiHand)，关节拓扑不同
3. **FoundationPose 感知管线**: 你用 MANO joints 而非视觉位姿估计
4. **KUKA 臂控制**: 你用虚拟 6-DOF wrist 而非真实机械臂

### 8.6 总结: 优先级建议

| 优先级 | 可借鉴项 | 预期收益 |
|--------|---------|---------|
| **高** | Progress-based reward (d* tracking) | 训练稳定性和收敛速度 |
| **高** | Asymmetric Actor-Critic | 部分可观测下的价值估计质量 |
| **中** | Tolerance curriculum | 前期快速学习 + 后期精细控制 |
| **中** | Wrist EMA smoothing | 减少 6-DOF wrist 高频振荡 |
| **中** | Object-centric keypoint obs | 为未来物体泛化做准备 |
| **低** | SAPG 探索 | 仅当 scale 到大量并行 env 且遇到 plateau |
| **低** | 程序化物体多样性 | 仅当需要泛化到新物体时 |
