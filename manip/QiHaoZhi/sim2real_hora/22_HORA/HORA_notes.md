# HORA - 论文笔记

> **修订说明**: 本笔记最初基于代码分析撰写 (当时无法获取 arxiv 全文)。现已对照论文全文 (HORA_paper.md) 进行校正，修正了训练物体类型、奖励函数公式差异、实验结果等内容。主要修改标记为 [paper-revised]。

**论文**: In-Hand Object Rotation via Rapid Motor Adaptation
**作者**: Haozhi Qi*, Ashish Kumar*, Roberto Calandra, Yi Ma, Jitendra Malik (UC Berkeley, Meta)
**发表**: CoRL 2022; arXiv:2210.04887
**项目**: https://haozhi.io/hora
**代码**: https://github.com/haozhiqi/hora

---

## 一句话总结

基于 Rapid Motor Adaptation (RMA) 的两阶段 teacher-student 框架，在 Allegro Hand 上实现多种物体的手内连续旋转 (z 轴)，通过 privileged information 训练 oracle 策略再用 proprioceptive history 蒸馏 adaptation module，实现 sim-to-real 零样本迁移。

---

## 核心问题

灵巧手手内物体旋转 (In-Hand Object Rotation) 是灵巧操控的基础技能，但面临多重挑战：

1. **高维动作空间**: Allegro Hand 16 DoF，接触模式复杂
2. **Sim-to-Real Gap**: 物体质量、摩擦、质心位移等物理参数在真实世界中未知且多变
3. **感知限制**: 真实部署时无法获取物体精确位置、物理属性等信息

核心问题可概括为：如何在训练时利用仿真器的 privileged information (物体位置、质量、摩擦、尺寸、质心)，部署时仅用 proprioceptive 关节历史就实现鲁棒的手内旋转。

---

## 方法概述

采用经典的 RMA (Rapid Motor Adaptation) 两阶段框架 [Kumar et al., 2021]:

### Stage 1: Oracle Policy (privileged information)

- **训练算法**: PPO
- **输入**: 当前关节观测 (96D) + privileged information (9D)
  - 关节观测: 最近 3 帧的 [scaled joint pos (16D), joint target (16D)]，共 96D
  - Privileged info: obj_position (3D) + obj_scale (1D) + obj_mass (1D) + obj_friction (1D) + obj_com (3D) = 9D
- **Privileged info 编码**: MLP [256, 128, 8] 将 9D 映射为 8D latent，经 tanh 后与观测拼接送入 actor
- **Actor 网络**: MLP [512, 256, 128] + ELU 激活
- **Critic**: **共享 actor 的 MLP trunk** (actor_mlp)，仅输出头不同 (`Linear(128,1)` vs `Linear(128,16)`)。无独立 critic 网络——后续 PenSpin/DexScrew 改为独立 critic
- **动作**: 16D 关节位置增量 (delta position)，`target = prev_target + action / 24`

### Stage 2: Proprioceptive Adaptation

- **冻结 Stage 1 的所有参数**，只训练 adaptation module (`adapt_tconv`)
- **Adaptation Module**: `ProprioAdaptTConv`，将 30 步 proprioceptive history 映射为 8D latent (替代 Stage 1 的 env_mlp 输出)
  - Channel Transform: Linear(32, 32) + ReLU -> Linear(32, 32) + ReLU
  - Temporal Aggregation: 3 层 Conv1d (kernel 9/5/5, stride 2/1/1) + ReLU
  - Low-dim Projection: Linear(32*3, 8)
- **训练方式**: On-policy rollout + MSE loss: `L = ||e_adapt - e_oracle.detach()||^2`
  - 用 student (adapt_tconv 的输出) 替代 teacher (env_mlp 的输出) 做推理，但 loss 用 teacher 的 detach 输出作为 target
- **额外归一化**: 对 proprio_hist 有独立的 `RunningMeanStd` (`sa_mean_std`)
- **训练步数**: 最多 1e9 步 (远多于 Stage 1)

```
Stage 1: obs --+-- [actor_mlp] --> mu, sigma, value
               |
               +-- env_mlp(priv_info) --> tanh --> 8D latent (concatenate with obs)

Stage 2: obs --+-- [actor_mlp] --> mu (frozen, use adapt output)
               |
               +-- adapt_tconv(proprio_hist) --> tanh --> 8D latent (concatenate with obs)
               |
               +-- env_mlp(priv_info) --> tanh --> 8D latent_gt (supervision target, detach)
```

---

## 关键设计

### 1. Reward 设计

从代码 (`allegro_hand_hora.py` L321-L352) 提取的完整 reward:

| 奖励项 | 公式 | Scale | 说明 |
|--------|------|-------|------|
| Rotation reward | `clip(dot(object_angvel, rot_axis), min, max)` | +1.0 | 鼓励绕 -z 轴旋转 |
| Object linvel penalty | `norm(object_linvel, p=1)` | -0.3 | 惩罚物体线速度 (保持位置稳定) |
| Pose diff penalty | `sum((dof_pos - init_pose)^2)` | -0.3 | 惩罚关节偏离初始抓取姿态 |
| Torque penalty | `sum(torques^2)` | -0.1 | 惩罚大力矩 |
| Work penalty | `(sum(torques * dof_vel))^2` | -2.0 | 惩罚机械功 (鼓励高效运动) [paper-revised: 论文公式为 $-\tau^T \dot{q}$ 无平方，代码实现有平方] |

**论文与代码的奖励公式差异** [paper-revised]:
- 论文 $r_{\text{linvel}} = -\|\mathbf{v}\|_2^2$ (L2 范数的平方)，代码实现为 `norm(object_linvel, p=1)` (L1 范数)
- 论文 $r_{\text{work}} = -\boldsymbol{\tau}^T \dot{\mathbf{q}}$ (线性)，代码实现为 `(sum(torques * dof_vel))^2` (平方)
- 论文未给出具体 lambda 值，代码中的 scale 值 (-0.3, -0.3, -0.1, -2.0) 是从 config 提取的

**关键细节**:
- `rot_axis_buf = [0, 0, -1]`，即鼓励绕 z 轴**逆时针**旋转 (从上方看)
- Angular velocity 通过四元数差分手动计算 (`quat_to_axis_angle`)，**不使用仿真器的 angvel** (v0.0.2 changelog 明确指出之前用仿真频率读 angvel 导致振荡)
- Angular velocity clip: `[-0.5, 0.5]` (后更新为 `[-0.5, 0.4]`)
- PPO play_steps 中有全局缩放: `shaped_rewards = 0.01 * rewards`

### 2. Domain Randomization

从 config (`AllegroHandHora.yaml` L50-72) 提取:

| 参数 | 范围 | 说明 |
|------|------|------|
| Mass | [0.01, 0.25] kg | 物体质量 |
| COM | [-0.01, 0.01] m | 质心偏移 (x, y, z 各自独立) |
| Friction | [0.3, 3.0] | 手-物摩擦系数 (手和物体同一值) |
| Scale | {0.7, 0.72, 0.74, ..., 0.86} | **离散列表** (非连续范围) |
| P gain | [2.9, 3.1] | PD 控制器 P 增益 |
| D gain | [0.09, 0.11] | PD 控制器 D 增益 |
| Joint noise | 0.02 | 关节位置噪声 (训练时) |

**值得注意**:
- Scale 使用**离散列表** + 小范围扰动 (`[-0.025, +0.025]`)，而非连续均匀分布。每个 env 按 `env_id % num_scales` 固定分配 scale 桶
- 每个 scale 对应一个预计算的 grasp cache (`cache/internal_allegro_grasp_50k_s{scale}.npy`)
- PD gain 随机化范围极小 (约 +/- 3%)，而非大范围扰动
- 质心随机化范围也很小 (+/- 1cm)

### 3. 初始抓取姿态生成 (Grasp Cache)

HORA 使用预计算的 grasp cache 而非在线随机初始化:

1. `gen_grasp.py` 在 CPU pipeline 下运行 20000 envs
2. 每个 env 从 canonical_pose (硬编码 16D) + 随机扰动初始化手
3. 物体置于手上方自由落下
4. Episode 结束时检查 3 个条件:
   - 所有指尖与物体距离 < 0.1m
   - 至少 2 根手指与物体有接触
   - 物体高度 > reset_z_threshold
5. 满足条件的状态 (16D hand dof_pos + 7D object pose) 存入 cache
6. 每个 scale 收集 50k 个有效抓取

**训练时**: 按 env_id 分桶，每个桶从对应 scale 的 50k grasps 中随机采样

### 4. Torque Control with Software PD

代码 (`allegro_hand_hora.py` L436-L448) 实现了软件级 PD 控制:

```python
# torque_control = True (default)
torques = p_gain * (cur_targets - dof_pos) - d_gain * dof_vel
torques = torch.clip(torques, -0.5, 0.5)
```

- 关节速度用**有限差分** (`dof_pos_t - dof_pos_{t-1}) / dt` 计算
- 力矩 clip 到 [-0.5, 0.5] Nm
- 仿真频率 120Hz (`dt = 0.00833`)，控制频率 20Hz (`controlFrequencyInv = 6`)
- 每个控制步内执行 6 次 physics step

### 5. Observation 设计

观测空间 96D = 3 帧 x 32D:
- 每帧 32D = scaled_joint_pos (16D) + joint_target (16D)
- `scaled_joint_pos = unscale(noisy_dof_pos, lower_limits, upper_limits)` 映射到 [-1, 1]
- 使用滑动窗口 (`obs_buf_lag_history`) 维护 80 帧历史，取最近 3 帧
- 同时维护 `proprio_hist_buf`: 最近 30 帧 x 32D，供 Stage 2 的 adaptation module

### 6. 终止条件

极其简单:
- 物体 z 坐标 < reset_z_threshold (内部版 0.645，公开版 0.625)
- 或 episode 步数 >= 400

无其他终止条件 (无手指距离检查、无超大角速度检查等)。

---

## 实验

[paper-revised] 以下结合论文全文和代码分析:

### 实验设置

- **平台**: Allegro Hand (内部版 + 公开版 v4)
- **仿真器**: IsaacGym [35]
- **训练规模**: 16384 envs 并行 (论文明确)
- **训练步数**: Stage 1 最大 1.5B steps，Stage 2 最大 1B steps (代码配置)
- **物体**: 训练**仅用圆柱体** (不同长宽比和质量，论文原文: "trained entirely in simulation on only cylindrical objects")
- **仿真频率**: 120 Hz; **控制频率**: 20 Hz
- **回合长度**: 400 控制步 = 20 秒 (仿真); 真实世界评估 30 秒
- **评估**: 训练分布内 + 分布外 (OOD: 更大物理随机化范围 + 20% 球体和立方体)

### 基线方法 [paper-revised]

1. **DR (Domain Randomization)**: 相同奖励函数但无 privileged information，学习对所有变化鲁棒 (非自适应) 的策略
2. **SysID (System Identification)**: adaptation module 预测精确系统参数 $\mathbf{e}_t$ 而非 extrinsics $\mathbf{z}_t$
3. **NoAdapt**: 仅在第一个时间步估计 extrinsics 后冻结，不做连续在线自适应
4. **Periodic (Action Replay)**: 回放 Expert 的参考轨迹
5. **Expert** (上界): 直接使用 privileged information

### 评估指标 [paper-revised]

| 指标 | 含义 | 适用范围 |
|------|------|----------|
| TTF (Time-to-Fall) | 物体掉落前的归一化回合长度 (仿真 /20s, 真实 /30s) | 仿真+真实 |
| RotR (Rotation Reward) | 未裁剪的平均旋转奖励 $\omega \cdot \hat{k}$ (非训练用的 clipped 版本) | 仅仿真 |
| Rotations | 物体净旋转弧度 (沿世界 z 轴) | 仅真实 |
| ObjVel | 物体线速度大小 x 100 | 仅仿真 |
| Torque | 每步指令力矩的平均 L1 范数 | 仿真+真实 |

### 仿真实验结果 (Table 1) [paper-revised]

500K 回合评估, 5 种子平均:

| 方法 | RotR (↑) in-dist | TTF (↑) in-dist | RotR (↑) OOD | TTF (↑) OOD |
|------|------|------|------|------|
| Expert | 233.71±25.24 | 0.85±0.01 | 165.07±15.63 | 0.71±0.04 |
| Periodic | 43.62±2.52 | 0.44±0.12 | 22.45±0.59 | 0.34±0.08 |
| NoAdapt | 90.89±4.85 | 0.65±0.07 | 54.50±3.91 | 0.51±0.06 |
| DR | 176.12±26.47 | 0.81±0.02 | 140.80±17.51 | 0.63±0.02 |
| SysID | 174.42±23.31 | 0.81±0.02 | 132.56±17.42 | 0.62±0.08 |
| **Ours** | **222.27±21.20** | **0.82±0.02** | **160.60±10.22** | **0.68±0.07** |

关键发现:
- HORA 在所有指标上接近 Expert 上界，远超所有基线
- DR 基线 TTF 尚可但 ObjVel/Torque 差: 无法感知物体属性，学到单一保守步态
- SysID 不如 HORA: 学习精确物理参数既困难又不必要，低维嵌入更优
- NoAdapt 真实世界完全失败: 证明连续在线自适应的必要性
- Periodic 最差: 盲目回放无法泛化

### 真实世界结果 [paper-revised]

**重物体组** (>100g: 棒球、水果、蔬菜、杯子, 20 次 x 6 物体):

| 方法 | Rotations (↑) | TTF (↑) | Torque (↓) |
|------|---------------|---------|------------|
| DR | 9.67±4.33 | 0.72±0.34 | 2.03±0.36 |
| SysID | 10.36±2.32 | 0.61±0.33 | 1.88±0.38 |
| NoAdapt | N.A. | 0.35±0.20 | N.A. |
| **Ours** | **23.96±3.16** | **0.98±0.08** | **1.84±0.24** |

**不规则物体组** (移动质心容器、凹面物体、猕猴桃、羽毛球、带孔玩具、立方体玩具):

| 方法 | Rotations (↑) | TTF (↑) | Torque (↓) |
|------|---------------|---------|------------|
| DR | 6.59±3.71 | 0.66±0.41 | 1.85±0.37 |
| SysID | 8.16±3.39 | 0.46±0.36 | 1.70±0.40 |
| NoAdapt | N.A. | 0.12±0.05 | N.A. |
| **Ours** | **19.22±4.88** | **0.78±0.27** | **1.48±0.30** |

真实世界总结: 方法可成功旋转 30+ 种物体 (直径 4.5-7.5 cm, 质量 5-200 g), 包括可变形和多孔物体。

### 分析与理解 [paper-revised]

1. **Extrinsics 可解释性**: 8D 中 $z_{t,0}$ 响应物体直径变化 (小直径低值, 大直径高值), $z_{t,2}$ 响应质量变化 (轻物体高值, 重物体低值)
2. **t-SNE 聚类**: 不同尺寸和重量的物体在 extrinsics 空间中占据不同区域; 不规则形状物体的 extrinsics 更分散
3. **手指步态涌现**: 使用圆柱体训练对涌现稳定高间隙步态很重要; 纯球体训练导致动态步态，在球上好用但无法泛化到复杂物体
4. **多轴旋转**: 附录探索了 +/- z 轴双向策略的可能性

### 训练脚本揭示的关键参数

从 `scripts/train_s1.sh` 可知实际训练参数覆盖了 config 默认值:
- `forceScale=2, randomForceProbScalar=0.25`: **训练时施加随机外力** (config 默认为 0)
- `object.type=cylinder_default`: 使用 9 种圆柱体
- `priv_info=True, proprio_adapt=False`: Stage 1 标志

从 `scripts/eval_s1.sh` 可知评估时:
- `jointNoiseScale=0.005` (训练时 0.02，评估降低 4x)
- `reset_height_threshold=0.6` (训练时 0.645，评估更宽松)
- 保持所有 domain randomization

### 公开版 vs 内部版 Allegro

| 属性 | 内部版 | 公开版 |
|------|--------|--------|
| URDF | `allegro_internal.urdf` | `allegro.urdf` |
| reset_z_threshold | 0.645 | 0.625 |
| rotateRewardScale | 1.0 | 1.25 |
| grasp_cache_name | `internal_allegro` | `public_allegro` |

公开版需要更高的 rotation reward scale 来补偿手部结构差异。

---

## 相关工作分析

HORA 的定位可从以下方面理解:

### 方法论来源

1. **RMA (Kumar et al., 2021)**: HORA 的核心框架直接借鉴 RMA (Ashish Kumar 是共同一作)。RMA 最初用于 legged locomotion，HORA 将其扩展到 dexterous manipulation。关键思想相同: privileged teacher → proprioceptive adaptation → sim-to-real
2. **IsaacGymEnvs / ShadowHand**: 代码基础来自 NVIDIA IsaacGymEnvs 中的 ShadowHand rotation task。HORA 替换了手部模型 (Shadow → Allegro) 并简化了 reward/observation

### 与 OpenAI Rubik's Cube (2019) 的比较

- OpenAI 使用 Shadow Hand (24 DoF)，HORA 使用 Allegro (16 DoF, 无腕部自由度)
- OpenAI 需要大量域随机化 (ADR) + 指尖触觉 + 视觉，HORA 仅用 proprioception + 相对温和的域随机化
- OpenAI 训练数千年仿真时间，HORA 训练量级更小
- HORA 的 RMA 框架提供了更结构化的 privileged info 利用方式

### 创新点

1. 首次在 Allegro Hand 上实现连续物体旋转的 sim-to-real
2. 将 RMA 从 locomotion 迁移到 manipulation，验证其通用性
3. Grasp cache 系统: 工程化的初始状态管理
4. [paper-revised] **核心洞察 (Key Insight)**: 手内旋转任务中，指尖感知的重要物理属性 (局部形状、质量、大小) 可以被压缩到紧凑的低维表示中。这种局部性 (locality) 是泛化到未见形状的关键 -- 真实世界中看似不同的物体在 extrinsics 空间中可能相似
5. [paper-revised] **手指步态自然涌现**: 不强制任何启发式手指接触约束 (对比 [28] 显式要求 >= 3 指接触)，稳定步态从能量约束和姿态偏差惩罚中自然涌现

---

## 局限性与未来方向

### 局限性

[paper-revised] 对照论文 Section 6 修正:

1. **仅 z 轴旋转**: `rot_axis_buf[:, -1] = -1`，只鼓励绕 z 轴旋转，是 SO(3) 问题的简化。论文指出三个主轴策略理论上可覆盖任意姿态，但这是未来工作
2. **需要预置物体**: 依赖 grasp cache 提供稳定初始抓取，无法从桌面拾取
3. **无视觉/触觉**: 部署时仅用 proprioception (关节位置)，大部分真实世界失败源于不正确的接触点导致不稳定力闭合
4. **有限物体类型**: 训练**仅用圆柱体** (非球体)。对直径 < 4.0 cm 的小物体失败 (手指频繁碰撞)；极端形状也较难操控
5. **固定手基座**: 无手臂运动，手掌朝上固定
6. **未利用真实世界数据**: 论文明确提出用 meta-learning 结合真实数据改进策略是有意义的未来方向

### 未来方向 (从后续工作已实现)

| 方向 | 后续实现 |
|------|----------|
| 扩展到笔形物体 | PenSpin (2024) |
| 双手协作旋转 | TwistingLids (2024) |
| 集成触觉感知 | NeuralFeels (2024) 使用 HORA 策略做视触觉实验 |
| 更复杂操控任务 | DexScrew (2025) 继承 HORA 代码架构 |

---

## 论文与代码差异

**注意**: 以下基于代码 v0.0.2 分析，与论文原始版本 (v0.0.1) 可能有差异。README 明确指出 v0.0.2 的实验数值与论文不一致。

### 论文公式与代码实现的已确认差异 [paper-revised]

| 项目 | 论文公式 | 代码实现 | 影响 |
|------|----------|----------|------|
| Linear velocity penalty | $r_{\text{linvel}} = -\|\mathbf{v}\|_2^2$ (L2 范数平方) | `norm(object_linvel, p=1)` (L1 范数) | 惩罚形式不同 |
| Work penalty | $r_{\text{work}} = -\boldsymbol{\tau}^T \dot{\mathbf{q}}$ (线性) | `(sum(torques * dof_vel))^2` (平方) | 代码对大功率的惩罚更强 |
| Torque metric | 论文评估用 $\ell_1$ norm | 代码训练用 `sum(torques^2)` (L2 范数平方) | 训练奖励和评估指标用不同范数 |
| 线速度计算 | 论文未说明 | 代码用位置差分 `(pos_t - pos_{t-1}) / dt` 而非仿真器的 linvel | 论文未提及 |
| 角速度计算 | 论文未说明 | 代码用四元数差分 → axis-angle 而非仿真器 angvel | v0.0.2 changelog 说明了原因 |

### 代码中的重要细节 (论文未充分描述)

| 项目 | 代码实现 | 文件位置 |
|------|----------|----------|
| Reward 全局缩放 | `shaped_rewards = 0.01 * rewards` | `hora/algo/ppo/ppo.py` L332 |
| 角速度计算 | 四元数差分 → axis-angle，**不用仿真器 angvel** | `hora/tasks/allegro_hand_hora.py` L329 |
| Action scale | `1/24` (即 `prev_target + action/24`) | `hora/tasks/allegro_hand_hora.py` L405 |
| Scale 随机化 | **离散列表** `[0.7, 0.72, ..., 0.86]` | `configs/task/AllegroHandHora.yaml` L64 |
| 训练外力 | config 默认 0，**训练脚本覆盖** `forceScale=2, prob=0.25` | `scripts/train_s1.sh` L15 |
| 物体类型 | config 默认 `block`，**训练脚本覆盖** `cylinder_default` | `scripts/train_s1.sh` L17 |
| 关节速度 | 有限差分 `(pos_t - pos_{t-1}) / dt` | `hora/tasks/allegro_hand_hora.py` L441 |
| Stage 2 学习率 | 固定 3e-4 (无 scheduler) | `hora/algo/padapt/padapt.py` L76 |
| Stage 2 训练 | 每步 on-policy rollout + 单次 MSE 更新 (无 replay) | `hora/algo/padapt/padapt.py` L108-118 |
| Value bootstrap | timeout 时 bootstrap value | `hora/algo/ppo/ppo.py` L333-334 |
| Sigma 初始化 | `nn.Parameter(zeros)`, state-independent | `hora/algo/models/models.py` L77 |
| Obs lag buffer | 维护 80 帧历史，取最近 3 帧作为 obs | `hora/tasks/base/vec_task.py` L190-192 |
| Adaptation input | 30 帧 x 32D = 30 帧 x (joint_pos + target) | `hora/tasks/allegro_hand_hora.py` L553 |
| PPO 超参 | lr=5e-3, e_clip=0.2, gamma=0.99, tau=0.95, horizon=8, mini_epochs=5 | `configs/train/AllegroHandHora.yaml` |
| Bounds loss | `soft_bound=1.1`, 惩罚 mu 超出 [-1.1, 1.1] | `hora/algo/ppo/ppo.py` L280-283 |
| Deploy 频率 | 20Hz (与仿真控制频率一致) | `hora/algo/deploy/deploy.py` L80 |

### Changelog 记录的 Bug 修复 (v0.0.1 → v0.0.2)

来自 `docs/changelog.md`:
1. **角速度读取频率**: 之前在仿真频率 (120Hz) 读取，现在在控制频率 (20Hz) 读取。之前的版本导致策略学到利用高频振荡
2. **移除 privileged info 的 hand-crafted normalization**: 之前对 mass/friction 等有手动设定的 lower/upper 用于归一化，现在移除
3. **移除 online mass randomization**: 创建仿真后动态修改质量无效，这是一个 Bug
4. **Angular velocity max clip**: 从 0.5 降低到 0.4，补偿修复后更高的旋转速度

### 关节顺序映射

部署代码 (`hora/algo/deploy/deploy.py`) 包含关键的关节顺序重映射:
- `_obs_allegro2hora()`: Allegro 物理排列 [index, middle, ring, thumb] → HORA 策略排列 [index, thumb, middle, ring]
- `_action_hora2allegro()`: HORA 策略输出 → Allegro 物理排列
- 这意味着 HORA 策略空间中的手指排列与 Allegro 物理 URDF 不一致，**如果映射错误会导致完全不可调试的失败**

### 网络架构完整规格

```
ActorCritic:
  env_mlp (Stage 1, privileged):
    Linear(9 -> 256) -> ELU
    Linear(256 -> 128) -> ELU
    Linear(128 -> 8) -> ELU
    -> tanh (output: 8D latent)

  adapt_tconv (Stage 2, proprioceptive):
    channel_transform:
      Linear(32 -> 32) -> ReLU
      Linear(32 -> 32) -> ReLU
    temporal_aggregation:
      Conv1d(32, 32, kernel=9, stride=2) -> ReLU  # (N, 32, 30) -> (N, 32, 11)
      Conv1d(32, 32, kernel=5, stride=1) -> ReLU  # -> (N, 32, 7)
      Conv1d(32, 32, kernel=5, stride=1) -> ReLU  # -> (N, 32, 3)
      # 注: 源码注释写 "(N, 50, 32)" 但实际 propHistoryLen=30, 输出 T=3 匹配 Linear(32*3, 8)
    low_dim_proj:
      Linear(32*3=96, 8)                            # flatten -> 8D latent
    -> tanh

  actor_mlp:
    Linear(96+8 -> 512) -> ELU                     # 96D obs + 8D latent
    Linear(512 -> 256) -> ELU
    Linear(256 -> 128) -> ELU

  mu: Linear(128 -> 16)
  value: Linear(128 -> 1)
  sigma: nn.Parameter(zeros(16))                    # state-independent
```

**初始化**: Linear bias 全零，Conv1d 使用 fan_out normal init (`sqrt(2/fan_out)`)，sigma 初始化为 0。

### 文件路径索引

| 功能 | 文件路径 |
|------|----------|
| 主入口 | `code/train.py` |
| 任务环境 | `code/hora/tasks/allegro_hand_hora.py` |
| 抓取生成 | `code/hora/tasks/allegro_hand_grasp.py` |
| 网络定义 | `code/hora/algo/models/models.py` |
| PPO 训练 | `code/hora/algo/ppo/ppo.py` |
| Stage 2 训练 | `code/hora/algo/padapt/padapt.py` |
| 经验缓存 | `code/hora/algo/ppo/experience.py` |
| 归一化 | `code/hora/algo/models/running_mean_std.py` |
| 部署 | `code/hora/algo/deploy/deploy.py` |
| 真机接口 | `code/hora/algo/deploy/robots/allegro.py` |
| 基类 | `code/hora/tasks/base/vec_task.py` |
| Task config | `code/configs/task/AllegroHandHora.yaml` |
| Train config | `code/configs/train/AllegroHandHora.yaml` |
| 训练脚本 | `code/scripts/train_s1.sh`, `code/scripts/train_s2.sh` |

---

## 跨论文比较

HORA 是 Haozhi Qi 灵巧操控系列的起点作品。以下分析其与后续工作的关系。

### HORA 在系列中的位置

```
HORA (2022, CoRL)
  首个 Allegro 手内旋转 sim-to-real
  RMA 框架, 仅圆柱体训练, z 轴旋转
       |
       |--- PenSpin (2024): 笔状物体, 三阶段 pipeline
       |--- TwistingLids (2024): 双手拧盖, Brake Link
       |--- NeuralFeels (2024): 借用 HORA 策略做视触觉感知
       |
       └--- DexScrew (2025): 螺母/螺丝刀, 简化仿真 + 技能原语
```

### 详细对比

| 维度 | HORA (2022) | PenSpin (2024) | TwistingLids (2024) | DexScrew (2025) |
|------|-------------|----------------|---------------------|-----------------|
| **任务** | 圆柱体旋转 (z 轴) [paper-revised: 训练仅用圆柱体] | 笔旋转 (z 轴) | 双手拧瓶盖 | 螺母紧固/螺丝刀 |
| **手部** | Allegro (16 DoF) | Allegro (16 DoF) | 2x Allegro (32 DoF) | XHand (12 DoF) |
| **Sim-to-Real 范式** | 直接迁移 (RMA) | Oracle → BC → 开环微调 | 直接迁移 | 简化仿真 → 技能原语 → 触觉 BC |
| **蒸馏方法** | Proprioceptive adaptation (online MSE) | Oracle-rollout BC | 不蒸馏 | DAgger |
| **特权信息** | pos/scale/mass/friction/com (9D) | 触觉 + 点云 + 物理属性 | Critic 特权 (非对称 AC) | pos/scale/mass/friction/com |
| **Adaptation 架构** | Conv1d (30帧 x 32D) | Temporal Transformer | N/A | Conv1d (继承 HORA) |
| **物体初始化** | Grasp cache (落体筛选) | 人工设计的 canonical grasps | 固定初始 | Grasp cache |
| **外力扰动** | 有 (forceScale=2) | 有 | 默认关闭 | 有 |
| **控制方式** | 软件 PD (torque mode) | 软件 PD | 位置目标 + EMA 平滑 | 软件 PD |
| **控制频率** | 20 Hz | 20 Hz | 10 Hz | 20 Hz |
| **训练 envs** | 16384 | -- | -- | 8192 |
| **Reward 全局缩放** | 0.01 | 0.01 | 0.001 | 0.01 |
| **部署感知** | 仅 proprioception | 仅 proprioception | 2 点视觉 | Proprioception + 触觉 |

### 代码继承关系

**PenSpin**: 直接 fork HORA 代码库，扩展了:
- 笔形物体的 PointNet 编码
- 多个 canonical grasp 的初始状态设计
- 三阶段 pipeline (Oracle → BC → 开环微调)
- Student 观测噪声加倍

**DexScrew**: 继承 HORA 架构 (`notes_overview.md` 明确记录)，共享:
- 相同的 PPO 实现 + ExperienceBuffer + GAE
- 相同的 ActorCritic 网络框架
- 相同的 RunningMeanStd 归一化
- 相同的 ProprioAdaptTConv / Temporal 选择
- 相同的 `shaped_rewards = 0.01 * rewards`

**TwistingLids**: 使用独立的 minimal PPO (~750 行)，但 reward 思路类似 (旋转奖励 + 正则化)。

**NeuralFeels**: 使用 HORA 训练的旋转策略作为 manipulation policy，在其上构建视触觉感知模块。HORA 策略提供了稳定的手内旋转行为，NeuralFeels 关注的是如何从视觉+触觉重建物体的 neural field 表示。

### HORA 的开创性贡献 (对后续工作的影响)

1. **Grasp Cache 系统**: 几乎所有后续工作都沿用了预计算稳定抓取的范式 (PenSpin 扩展为多组 canonical grasps，DexScrew 直接继承)
2. **软件 PD + 有限差分速度**: 不信任仿真器 dof_vel 的做法成为后续工作的标准实践
3. **RMA-style adaptation**: 虽然 RMA 不是 HORA 原创，但 HORA 验证了这一框架在 dexterous manipulation 中的有效性，为 PenSpin/DexScrew 的 adaptation 设计奠定基础
4. **关节重映射工具**: `_obs_allegro2hora` / `_action_hora2allegro` 的模式在后续每个项目中重复 (不同手有不同映射)
5. **最小化感知的部署哲学**: HORA 证明仅用 proprioception (无视觉无触觉) 就能实现手内旋转。后续工作在此基础上逐步引入更多感知模态 (TwistingLids: 2 点视觉; NeuralFeels: 视触觉; DexScrew: 触觉)

### HORA 相对于后续工作的不足

1. **单一旋转轴**: 后续 PenSpin 仍是 z 轴，但 TwistingLids 实现了关节轴旋转
2. **简单蒸馏**: Online MSE loss 是最朴素的蒸馏方式。PenSpin 发现 DAgger 在动态任务上失败需要 Oracle-rollout BC; DexScrew 证明 DAgger 在旋拧任务上可行
3. **无初始状态设计**: HORA 用随机落体筛选，PenSpin 发现人类启发的 canonical grasps 对 finger gaiting 涌现至关重要
4. **无面向部署的奖励设计**: PenSpin 的 $r_z$ (保持笔水平) 虽然对仿真指标影响小，但对真实世界开环回放至关重要。HORA 的 reward 设计更朴素
