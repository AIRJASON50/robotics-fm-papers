# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation -- 研究笔记

> 论文: Kushal Kedia\*, Tyler Ga Wei Lum\* et al. Cornell + Stanford, 2025
> 一句话: 在程序化生成的 primitive 物体上训练单一 goal-conditioned policy, zero-shot 部署到 12 种真实工具 / 24 个任务

---

## 1. Core Problem

灵巧手工具操作需要组合三种基础技能: (1) 从桌面抓取细长物体, (2) in-hand reorientation (手内重定向) 到功能姿态, (3) 在力交互中保持抓握稳定性。现有方法面临三个可扩展性瓶颈:

| 瓶颈 | 具体表现 |
|------|---------|
| Per-task reward engineering | 每增加一个工具/任务就要重新设计 reward, 无法扩展 |
| 遥操数据质量差 | 人手-机器手结构不同 (embodiment gap), 高自由度手的遥操难以产生高质量接触数据 |
| 只能解子问题 | 现有 sim2real 方法只做 grasping 或 reorientation 或 spinning, 不能端到端执行完整工具使用流程 |

核心 insight: **所有工具使用任务都可以统一为 "将工具依次移动到一系列目标 6D 位姿"**。这个抽象消除了 per-task reward 的需求 -- 只需要一个 universal 的 goal-pose reaching objective。

---

## 2. Method Overview

### 2.1 整体架构

```
Training (Simulation)                     Deployment (Real World)
========================                  ==========================
Procedural Primitives                     Human RGB-D Video
(handle + head, cuboid/capsule)           |
        |                                 v
        v                                 SAM 3D --> Object Mesh + Grasp BBox
Goal-Conditioned RL Policy                FoundationPose --> 6D Pose Trajectory
(SAPG + Asymmetric Critic)                |
        |                                 v
        v                                 Policy (zero-shot)
Single LSTM Policy                        --> 29-DoF Joint Targets
(obs: proprioception + object pose        (KUKA iiwa14 arm + Sharpa 22-DoF hand)
 + grasp bbox + goal pose)
```

### 2.2 问题形式化

给定当前物体位姿 $o_t \in SE(3)$, 机器人本体感知 $s_t$, 粗略物体描述符 $\phi$ (bounding box scale), 和目标位姿 $g \in SE(3)$:

$$a_t = \pi_\theta(s_t, o_t, \phi, g)$$

策略输出 29-DoF 关节位置目标 (7 arm + 22 hand)。任务执行 = 逐个 reach 目标位姿序列 $\{g^k\}_{k=1}^K$, 当 $d(o_t, g^k) < \epsilon$ 时切换到下一个目标。

### 2.3 Reward 结构

$$r = r_{\text{smooth}} + r_{\text{grasp}} + \mathbb{I}_{\text{grasped}} \cdot r_{\text{goal}}$$

reward 分三阶段自然 curriculum:

| 阶段 | 主导 reward | 作用 |
|------|-----------|------|
| 1. Approach | $r_{\text{grasp}}$: fingertip delta reward | 指尖接近物体, 正向增量奖励 |
| 2. Lift | lifting reward + bonus | 将物体提离桌面, 超过阈值后停止 lifting reward |
| 3. Goal reaching | $r_{\text{goal}}$: keypoint delta reward | 只奖励 keypoint 距离减小 (progress-based), 不惩罚绝对距离 |

关键设计: **progress-based reward** -- 只在距离减小时给正奖励 ($d^* - d$, clamp at 0), 不惩罚距离本身。这防止策略在目标附近震荡。

### 2.4 Tolerance Curriculum

成功判定阈值从 $\epsilon_0 = 0.075m$ 逐步收紧到 $\epsilon_{target} = 0.01m$:
- 每 3000 步 (across all agents) 检查一次
- 若 mean successes per episode >= 3, 则 $\epsilon \leftarrow \epsilon \times 0.9$
- 精度逐步收紧, 避免一开始就要求过高精度导致探索困难

---

## 3. Key Designs

### 3.1 Object-Centric Representation (核心贡献)

**设计**: 4 个 keypoints (相对于 palm) + bounding box scale (3D) 作为物体表示。

| 组成 | 维度 | 含义 |
|------|------|------|
| keypoints_rel_palm | 4 x 3 = 12 | 物体 4 个 keypoint 相对于 palm center 的位移 |
| keypoints_rel_goal | 4 x 3 = 12 | 物体 keypoint 与目标 keypoint 之间的位移 |
| object_rot | 4 (quaternion) | 物体旋转四元数 |
| object_scales | 3 | bounding box [x, y, z] 尺寸 |

**为什么这样设计**:
1. **相对坐标**: keypoint 相对于 palm 表示, 而非世界坐标 -- 对坐标系变换不变, 直接跨场景泛化
2. **Keypoint 天然处理对称性**: 4 个点足以编码 6D pose 信息, 且比 quaternion 更适合 RL 学习
3. **Bounding box scale 而非精确几何**: 部署时只需粗略尺寸, 不需要精确 mesh -- 大幅降低 sim-to-real 感知要求

**Actor 完整观测空间 (140 维)**:

| 观测名 | 维度 | 说明 |
|--------|------|------|
| joint_pos (unscaled) | 29 | 归一化关节位置 |
| joint_vel | 29 | 关节速度 |
| prev_action_targets | 29 | 上一步目标关节位置 |
| palm_pos | 3 | 掌心位置 |
| palm_rot | 4 | 掌心旋转 |
| object_rot | 4 | 物体旋转 |
| fingertip_pos_rel_palm | 5 x 3 = 15 | 5 个指尖相对掌心的位移 |
| keypoints_rel_palm | 4 x 3 = 12 | 物体 keypoint 相对掌心 |
| keypoints_rel_goal | 4 x 3 = 12 | 物体 keypoint 与目标差 |
| object_scales | 3 | bounding box 尺寸 |
| **Total** | **140** | |

**Critic 额外观测 (Asymmetric)**: palm_vel, object_vel, closest_fingertip_dist, lifted_object flag, progress, successes, reward 等特权信息。

### 3.2 Procedural Object Generation (数据多样性)

**核心思想**: 不用真实工具 mesh, 用 primitive 几何体 (cuboid + capsule) 的随机组合模拟工具多样性。

每个工具 = **handle** + **head** (可选), 两者各自从 cuboid 或 cylinder 中采样:

| 工具类别 | Handle 尺寸范围 (m) | Head 尺寸范围 (m) | 密度策略 |
|---------|-------------------|------------------|---------|
| Hammer | [0.15-0.3] x [0.015-0.04] | [0.02-0.06] x [0.05-0.12] | handle 低密度 (300-600), head 高密度 (800-2000) |
| Screwdriver | [0.07-0.12] x [0.025-0.04] | [0.07-0.15] x [0.01-0.015] | handle 低密度, head 高密度 |
| Marker | [0.075-0.15] x cylinder | [0.01-0.03] x [0.005-0.01] | 均为低密度 |
| Spatula | [0.1-0.2] x [0.006-0.025] | [0.05-0.15] x [0.03-0.07] | 均为低密度 |
| Eraser | [0.07-0.15] x [0.02-0.07] | 无 head | 仅低密度 |
| Brush | [0.05-0.2] x [0.01-0.04] | [0.05-0.12] x [0.03-0.12] | 均为低密度 |

代码中共定义 12 个 `ObjectSizeDistribution` (每个类别 box + cylinder 各一, brush 有额外 head 变体)。8192 个并行环境, 每个环境在 reset 时从分布中独立采样生成新物体。

**质量分布的精心设计**: hammer 的 head 密度 (800-2000 kg/m^3) 远高于 handle (300-600 kg/m^3), 模拟真实工具的质心偏移。代码使用 parallel axis theorem 精确计算组合惯性张量。

### 3.3 SAPG (Sample-Efficient Actor-Population-Guided) + Action Smoothing

**SAPG**: 在 massively parallel 仿真中, 标准 PPO (Proximal Policy Optimization, 近端策略优化) 会遇到 exploration saturation -- 大量并行环境都在探索同一片状态空间。SAPG 通过维护多个策略 "block" 来促进探索多样性:

- 将 8192 个环境分成多个 block (由 `expl_coef_block_size` 控制, 默认 4096)
- 每个 block 维护不同的 exploration coefficient embedding (通过 sinusoidal encoding)
- Leader policy 使用所有 block 的经验进行更新 (importance sampling)

**Action smoothing (EMA)**: 动作通过极强的 EMA (Exponential Moving Average, 指数移动平均) 平滑:
- $target_t = \alpha \cdot action_t + (1-\alpha) \cdot target_{t-1}$
- hand_moving_average = arm_moving_average = **0.1** (即 alpha=0.1, 极强平滑)
- Action space: arm 输出 delta 关节位置, hand 输出 absolute 关节位置
- 效果: 消除高频振荡, 平滑指尖运动 -- 对 sim-to-real 迁移至关重要

**网络架构**: LSTM(1024, 1 layer, layer norm) -> MLP[1024, 1024, 512, 512], ELU 激活

---

## 4. Experiments

### 4.1 实验设置

| 项目 | 规格 |
|------|------|
| 机器人 | KUKA iiwa14 7-DoF arm + Sharpa 22-DoF 五指左手 |
| 仿真 | IsaacGym, 8192 并行环境, 60 Hz 控制频率 |
| 训练 | SAPG, ~数十亿 env steps |
| Benchmark | DexToolBench: 6 类工具, 12 物体实例, 24 任务, 120 次真机 rollout |
| 评估指标 | Task Progress: 成功 track 的 goal pose 百分比 ($\epsilon = 2cm$) |

### 4.2 主要结果

**A. Zero-shot 真机泛化 (Fig. 4)**:

| 工具类别 | 关键发现 |
|---------|---------|
| Eraser | 最高 Task Progress, 仅需平移, 无需 in-hand rotation |
| Marker | 不需旋转, 但太细导致抓握不稳 + 追踪易丢失 |
| Hammer/Brush/Spatula/Screwdriver | 需要大幅度 in-hand rotation, 性能随工具厚度和重量下降 |

**B. Baseline 对比 (Fig. 5)**:

| 方法 | Sweep Forward (无旋转) | Sweep Forward (需 90 度旋转) |
|------|----------------------|---------------------------|
| SimToolReal | 高 Task Progress | 高 Task Progress (执行 in-hand rotation) |
| Fixed Grasp | 可完成 (arm motion 够用) | 失败 (arm 碰桌子, 无法绕过 kinematic 限制) |
| Kinematic Retargeting | 失败 (无法建立稳定接触) | 失败 |

SimToolReal 比 retargeting 和 fixed grasp 高 **37%** Task Progress。

**C. Specialist 对比 (Fig. 6, 仿真)**:

| 设置 | Specialist | SimToolReal |
|------|-----------|-------------|
| 训练物体 + 训练轨迹 (Obj A / Traj A) | ~Match | ~Match |
| 训练物体 + 新轨迹 (Obj A / Traj B) | 显著下降 (只能 track 前几个 lift goals) | 保持高性能 |
| 新物体 + 训练轨迹 (Obj B / Traj A) | 最大下降 | 保持高性能 |

结论: Specialist 严重 overfit 到训练配置, SimToolReal 在所有变体上 zero-shot 匹配或超越 specialist。

**D. 训练目标预测泛化 (Fig. 7)**: Training reward (primitive goal-pose reaching) 与 DexToolBench Task Progress 之间呈强正相关 -- 验证了 "universal goal-pose reaching = universal tool manipulation" 的核心假说。

**E. Ablation (Fig. 8)**:

| Ablation | 效果 |
|----------|------|
| SAPG -> PPO | 性能显著下降 (exploration saturation) |
| 去掉 Asymmetric Critic | 性能显著下降 (partial observability 下 value estimation 不稳定) |

### 4.3 失败分析

| 失败原因 | 占比 |
|---------|------|
| 位姿追踪丢失 (遮挡/对称/低对比度) | **43.7%** |
| 物体掉落 | 34.5% |
| In-hand rotation 不完整 | 18.2% |
| 抓取失败 | 3.6% |

关键发现: **Perception (位姿追踪) 是 sim-to-real 的真正瓶颈, 不是 control policy 本身**。

---

## 5. Related Work Analysis

| 方向 | 代表工作 | SimToolReal 的定位 |
|------|---------|-------------------|
| IL from Teleoperation | DIME, Holo-Dex, AnyTeleop | SimToolReal 完全绕过遥操, 用 sim RL 替代; 遥操的 embodiment gap 和缺乏触觉反馈使其不适合工具操作 |
| Human Video Retargeting | DexFunc, DexTrack, Functional Retarget | SimToolReal 只提取 object 轨迹 (而非 hand motion), 策略泛化到新工具无需重训; 对比之下 DexTrack 需要 per-trajectory RL |
| Sim-to-Real (per-task) | Dactyl (Rubik's Cube), RMA | SimToolReal 用 procedural generation + universal reward 替代 per-task engineering |
| Locomotion scaling | Embodiment Scaling Laws | 灵感来源: locomotion 领域已证明 procedural generation + unified reward 可跨 embodiment 泛化, SimToolReal 将此思路迁移到 manipulation |

SimToolReal 在 related work 谱系中的独特位置: 它是第一个将 **procedural object diversity + universal goal-conditioned policy** 思路应用于灵巧手工具操作并实现真机 zero-shot 迁移的工作。

---

## 6. Limitations & Future Directions

### 论文明确提出的局限

| 局限 | 影响 | 未来方向 |
|------|------|---------|
| 不保证 functional task completion | 只 track 位姿序列, 不考虑力交互效果 (如锤子是否真的敲到钉子) | Force-aware goal conditioning |
| Environment-blind | 策略不感知环境障碍物, cluttered scenes 中会碰撞 | 加入 scene representation |
| 仅支持刚体工具 | 剪刀等铰接/柔性工具无法用单一 6D pose 描述 | Articulated pose representation |
| Goal sequence 固定 | 不能动态 replanning, 无法应对意外偏差 | Closed-loop goal adaptation |

### 分析中发现的隐含局限

| 局限 | 来源 |
|------|------|
| Perception 瓶颈 (43.7% 失败) | FoundationPose 在遮挡、对称物体、低对比度下追踪不稳 |
| 重工具性能下降 | 331g mallet 比 36g claw hammer 差很多 -- 训练时密度范围可能不够 |
| 薄工具抓握不稳 | ~1cm 厚的 flat spatula 比 ~3cm 的 spoon spatula 差 -- primitive 可能未充分覆盖极薄物体 |
| Sharpa 手特异性 | 整套系统绑定 KUKA + Sharpa, 跨 embodiment 泛化未验证 |
| Human video 作为 goal source 的局限 | 需要 RGB-D 相机 + 已知背景, 且 SAM 3D mesh 质量直接影响后续追踪 |

---

## 7. Paper vs Code Discrepancies

### 7.1 Reward 结构比论文更复杂

论文 Eq.(1) 给出简洁的三项 reward: $r = r_{\text{smooth}} + r_{\text{grasp}} + \mathbb{I}_{\text{grasped}} r_{\text{goal}}$

代码实际 reward 包含更多项:

| 代码中的 reward 项 | 论文对应 | 备注 |
|-------------------|---------|------|
| `fingertip_delta_rew` | $r_{\text{grasp}}$ | 指尖逐步接近物体的增量奖励, 物体提起后停止 |
| `lifting_rew` | 未单独提及 | 线性 z-lift reward, clamp 在 [0, 0.5] |
| `lift_bonus_rew` | 未单独提及 | 物体超过 0.15m 高度时一次性 bonus = **300** |
| `keypoint_rew` | $r_{\text{goal}}$ | progress-based keypoint 距离减小奖励 |
| `bonus_rew` (near goal) | 包含在 $r_{\text{goal}}$ 中 | 到达目标位姿后持续给 bonus = 1000/successSteps |
| `kuka_actions_penalty` | $r_{\text{smooth}}$ | arm 速度惩罚, scale = 0.03 |
| `hand_actions_penalty` | $r_{\text{smooth}}$ | hand 速度惩罚, scale = 0.003 |

论文还提到 object velocity penalty (`objectLinVelPenaltyScale`, `objectAngVelPenaltyScale`), 但代码中默认为 **0.0** (禁用)。`hand_delta_penalty` 也被乘以 0 禁用。

reward scale 配置:

| 项 | Scale 值 |
|----|---------|
| distanceDeltaRewScale | 50.0 |
| liftingRewScale | 20.0 |
| liftingBonus | 300.0 |
| keypointRewScale | **200.0** |
| reachGoalBonus | **1000.0** |

keypoint reward 和 goal bonus 的 scale 远大于其他项 -- goal reaching 是绝对主导 reward。

### 7.2 Keypoint reward 使用 fixed size (论文未提)

代码中有 `fixedSizeKeypointReward: True` 和 `fixedSize: [0.141, 0.03025, 0.0271]`。启用时, keypoint 距离使用固定物体尺寸计算, 而非实际物体尺寸。这使得不同大小物体的 reward scale 一致。论文未提及此设计选择。

### 7.3 Domain Randomization 比论文描述更全面

代码中的完整 DR (Domain Randomization, 域随机化) 列表:

| 随机化项 | 范围 | 论文是否提及 |
|---------|------|-------------|
| Observation delay | max 3 steps | 提及 |
| Action delay | max 3 steps | 提及 |
| Object state delay | max **10** steps | 提及 (但未说具体值) |
| Object xyz noise | std = 0.01m | 提及 |
| Object rotation noise | 5 degrees | 提及 |
| Random force perturbation | 20.0 N (only when lifted) | 提及 |
| Random torque perturbation | 2.0 Nm (only when lifted) | 提及 |
| Robot mass scaling | [0.7, 1.3] | 未详细提及 |
| Robot friction scaling | [0.7, 1.3] | 未详细提及 |
| Robot stiffness/damping scaling | [0.7, 1.3] loguniform | 未详细提及 |
| Object mass scaling | [0.7, 1.3] | 未详细提及 |
| Object friction scaling | [0.7, 1.3] | 未详细提及 |
| Gravity perturbation | additive gaussian, range [0, 0.3] | 未提及 |
| Joint velocity obs noise | std = 0.1 | 未提及 |

### 7.4 SAPG 实现细节

论文描述 SAPG 维护 "a population of policies with 6 blocks"。代码中的实现通过 `expl_coef_block_size = 4096` 和 `expl_type = 'mixed_expl'` 实现:
- 8192 envs / 4096 block_size = 2 blocks (而非论文说的 6 blocks)
- 默认配置 (`SimToolRealPPO.yaml`) 中 `expl_type: 'none'` 和 `use_others_experience: 'none'`, 这实际上是标准 PPO
- SAPG 需要通过命令行或实验配置覆盖这些参数才能启用

这说明默认开源代码配置并未直接启用 SAPG, 需要用户手动配置。

### 7.5 2-Link URDF 被弃用

`generate_handle_head_urdf_variable_density_2_links()` 函数存在但被注释掉, 代码注释说 "For some reason, the 2-link approach is not working well, causing physics instability"。实际使用的是单 link + 手动计算合成惯性张量的方案 (`generate_handle_head_urdf_variable_density`)。

---

## 8. Cross-Paper Comparison

### 8.1 SimToolReal vs OmniReset

| 维度 | SimToolReal | OmniReset |
|------|------------|-----------|
| **问题定义** | 工具操作 (grasp + reorient + force interaction) | 通用物体操作 (pick + reorient + insert/place) |
| **泛化维度** | 跨物体几何 + 跨任务轨迹 | 主要跨 reset 状态, 不强调跨物体泛化 |
| **exploration 解决方案** | SAPG (多 policy 种群) | OmniReset (多样 reset states) |
| **核心 insight** | 训练目标 = goal-pose reaching on diverse primitives | 数据覆盖 = diverse reset states 覆盖状态空间 |
| **Reward 设计** | Per-stage reward (approach -> lift -> goal), 有 bonus | Task-agnostic (r_reach + r_dist + r_success + r_smooth) |
| **控制器** | 关节 PD (arm delta + hand absolute + EMA) | OSC (Operational Space Control, 笛卡尔空间末端力矩控制) |
| **灵巧手** | Sharpa 22-DoF 五指手 | UR7e 夹爪 (非灵巧手) |
| **Sim-to-real** | One-stage: train once, deploy zero-shot | Two-stage: train (ideal) -> finetune (sysid + ADR) |
| **Perception 瓶颈** | FoundationPose 追踪 (43.7% 失败) | RGB distillation (state expert ~100% vs RGB ~50%) |

**核心对比**: SimToolReal 和 OmniReset 解决 exploration 问题的思路互补。SimToolReal 用算法级别的解法 (SAPG 多策略种群), OmniReset 用环境级别的解法 (diverse reset states)。两者都意识到 massively parallel RL 的 exploration saturation 问题, 但攻击角度不同。

### 8.2 SimToolReal vs ArtiGrasp

| 维度 | SimToolReal | ArtiGrasp |
|------|------------|-----------|
| **物体类型** | 刚体工具 (handle + head primitive) | 铰接物体 (scissors, pliers, drawers) |
| **手类型** | 机器手 (Sharpa 22-DoF) | 人手模型 (MANO 51-DOF), 双手 |
| **Sim-to-real** | 有 (zero-shot 真机部署) | 无 (纯仿真) |
| **目标表示** | 6D object pose trajectory | 人手参考轨迹 (motion tracking) |
| **训练策略** | 单一通用策略 | 单一策略, 但含 curriculum (先固定物体单手 -> 自由物体双手) |
| **物体表示** | Bounding box scale | 暴露完整物体状态 (关节角度等) |

**核心对比**: ArtiGrasp 处理铰接物体但不做 sim-to-real; SimToolReal 处理刚体工具但实现了 zero-shot 真机迁移。两者的 gap 指向一个 open problem: 如何对铰接/柔性工具做 zero-shot sim-to-real?

### 8.3 SimToolReal vs DexTrack

| 维度 | SimToolReal | DexTrack |
|------|------------|---------|
| **泛化粒度** | 一个策略覆盖所有物体+任务 (zero-shot) | 一个策略覆盖 3585 条轨迹 (但需要 data flywheel 迭代训练) |
| **目标来源** | Object pose trajectory (from human video) | Hand+object kinematic reference (from MoCap/ARCTIC/TACO) |
| **训练数据** | 无需人类数据 (procedural generation) | 需要人手数据集 (GRAB + TACO) |
| **Action space** | Joint PD (absolute hand + delta arm) + EMA | Double integration (residual acceleration) |
| **网络规模** | LSTM(1024) + MLP[1024,1024,512,512] | MLP[8192,4096,2048,1024,512,256,128] (巨型 MLP) |
| **物体表示** | 4 keypoints + bbox scale (31 dims) | PointNet 256D feature |
| **训练算法** | SAPG | PPO + IL (0.05% weight) hybrid |
| **接触力处理** | 随机扰动力 (DR) | 忽略 (只做运动学追踪) |

**核心对比**: SimToolReal 追求"简洁的 task abstraction + zero-shot 泛化", DexTrack 追求"通用 tracking + data scaling"。SimToolReal 不需要人类数据但只能做刚体; DexTrack 依赖人类数据但能处理更复杂的手-物交互。从 FM (Foundation Model, 基础模型) 视角看, SimToolReal 的 universal objective 更像 next-token prediction (一个目标覆盖一切), DexTrack 的 data flywheel 更像 web-scale pretraining (靠数据量产生能力)。

### 8.4 SimToolReal vs Dex4D

| 维度 | SimToolReal | Dex4D |
|------|------------|-------|
| **目标表示** | 6D object pose (位姿) | 3D point tracks (表面点时间轨迹) |
| **Goal source** | RGB-D human video + FoundationPose | Video generation model + CoTracker3 |
| **Sim-to-real gap** | 通过 minimal obs (pose + bbox) 绕过视觉 gap | 通过 point tracks 的 domain-invariance 绕过 |
| **物体对称性** | Keypoint 编码隐式处理 | Paired point encoding 显式保持 correspondence |
| **训练阶段** | 一阶段, 单一策略 | 三阶段 (category-specific -> all-category -> DAgger distillation) |
| **开源状态** | 完整代码 + benchmark | 代码部分开源, 硬件部分未开源 |

**核心对比**: 两篇论文共享 "task-agnostic goal-conditioned policy" 的核心理念, 但目标表示选择不同。Dex4D 的 3D point tracks 更 robust (不依赖精确 mesh 追踪), 但需要视频生成模型作为高层规划器。SimToolReal 的 6D pose 更简洁直接, 但受限于 FoundationPose 的追踪质量。长期来看, point tracks 可能是更好的 manipulation 接口 -- 不需要物体模型, 天然 domain-agnostic。

### 8.5 与其他 sim2real 方法的横向对比

| 方法 | 物体表示 | 任务表示 | 泛化范围 | 真机验证 |
|------|---------|---------|---------|---------|
| SimToolReal | bbox + 4 keypoints | 6D pose sequence | 12 物体 / 24 任务 zero-shot | 120 rollouts |
| Dex4D | 64 point tracks | 3D point trajectories | ~10 物体, 多任务 | 有限 |
| DexTrack | PointNet 256D | Kinematic reference | 3585 轨迹 (但需重训) | 无 |
| OmniReset | Full state | r_dist + r_success | 单物体多配置 | 有 (sysid required) |
| BiDexHD | Full state | Teacher action | 141 任务 (但需 per-task teacher) | 无 |

---

## Takeaway for RL -> FM

| # | Takeaway | 原理 | 行动项 |
|---|----------|------|--------|
| 1 | "Goal-conditioned pose reaching" 是 manipulation 的 universal objective | 类比 next-token prediction: 一个足够通用的目标函数自然覆盖所有下游技能 | 评估自己的 RL 任务是否可以统一为某种 universal goal (而非 per-task reward) |
| 2 | Procedural primitive diversity > accurate mesh modeling | 与 LLM 用 web-scale 粗糙数据同理: 多样性 > 精度; bounding box 足以作为跨物体的 transfer medium | 在自己的训练中尝试 primitive object generation 替代精确 URDF |
| 3 | Progress-based reward 优于 absolute distance reward | $\max(d^* - d, 0)$: 只奖励进步, 不惩罚距离, 防止策略在目标附近震荡 | 直接可用的 reward engineering 技巧 |
| 4 | Perception 是 sim-to-real 的真正瓶颈 | 43.7% 失败来自追踪丢失, 而非 policy failure; Robotics FM 的 vision backbone 质量决定系统天花板 | sim-to-real 项目中优先投资感知模块 (比优化 policy 更有效) |
| 5 | EMA alpha=0.1 是有效的 sim-to-real action smoothing | 极强平滑消除 policy 高频振荡, 是 zero-shot transfer 的关键工程选择 | 在自己的 sim2real pipeline 中实验不同的 EMA 值 |
