# ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation

## 论文概述

ArtiGrasp 是 ETH Zurich 提出的一种基于强化学习的方法，用于合成双手（bi-manual）灵巧抓取与铰接物体操作的动态序列。核心目标是：给定铰接物体的初始姿态和目标铰接角度，加上静态手部姿态参考（hand pose reference），生成物理可信的双手交互运动。

**核心贡献:**
1. 提出统一策略（single policy）同时处理抓取和铰接操作，无需针对不同物体或任务重新训练
2. 设计了通用的奖励函数，将手部姿态追踪和任务目标（铰接角度）统一到一个框架中
3. 提出渐进式课程学习（curriculum）：先固定物体底座 + 单手分别训练 -> 再解放物体 + 双手协作微调
4. 引入铰接特征（articulation features）增强策略对物体结构的感知

**关键信息:**
- 仿真引擎: RaiSim（非 MuJoCo）
- RL 算法: PPO
- 手部模型: MANO（51 DOF = 6 DOF 全局 + 45 DOF 局部关节）
- 数据集: ARCTIC（双手铰接物体交互数据集）
- 训练硬件: 单卡 RTX 6000 + 128 CPU cores，训练约 3 天

---

## 方法细节

### 问题建模

**手部模型:**
- 使用 MANO 手部模型，均值形状（mean shape）
- 每只手 51 DOF: 全局位姿 T (6 DOF) + 局部关节 q (45 DOF)
- 手部被转换为 URDF 格式在 RaiSim 中作为 ArticulatedSystem 加载
- 控制模式: PD + Feedforward Torque (`PD_PLUS_FEEDFORWARD_TORQUE`)
- PD 增益: 平移 P=50, D=0.1; 关节 P=50, D=0.2

**物体表示:**
- 铰接物体由两部分组成: `bottom`（基座）和 `top`（铰接部分），通过 hinge joint 连接
- 物体姿态 Omega = {6D 基座位姿, 1D 铰接角度}，共 8 DOF
- 取自 ARCTIC 数据集（notebook, box, laptop, waffleiron, mixer, microwave, ketchup）

**动作空间:**
- 动作维度 = gcDim（广义坐标维度 = 51）
- 动作为残差（residual）: `pTarget = action * actionStd + actionMean`
- 手指动作标准差: `finger_action_std = 0.015`
- 旋转动作标准差: `rot_action_std = 0.01`
- 平移动作标准差: `0.001`（root guided 模式下）
- 动作经过关节限位裁剪后作为 PD 控制的位置目标
- 最小动作标准差: `0.2`（通过 `enforce_minimum_std` 实现）

**观测空间:**
- 右手观测维度 `obDim_r = 276`，包含:
  - `gc_r.tail(45)`: 关节角度（去掉前 6 DOF 全局位姿）
  - `bodyLinearVel` (3): 腕部线速度（物体坐标系）
  - `bodyAngularVel` (3): 腕部角速度（物体坐标系）
  - `gv_r.tail(45)`: 关节角速度
  - `rel_body_pos` (63): 21 个关节位置与目标的差值（腕部坐标系）
  - `rel_pose` (48): 当前姿态与目标姿态的角度差
  - `rel_objpalm_pos` (3): 物体与腕部的相对位置（腕部坐标系）
  - `rel_obj_vel` (3): 物体在腕部坐标系中的速度
  - `rel_obj_qvel` (3): 物体在腕部坐标系中的角速度
  - `final_contact_array` (16): 目标接触数组
  - `impulses` (16): 接触冲量
  - `rel_contacts` (16): 当前达成的目标接触
  - `arm_in_wrist` (3): 手臂方向在腕部坐标系
  - `arm_norm` (1): 手臂长度
  - `m_base` (1), `m_top` (1): 物体两部分质量
  - `rotation_axis_h` (3): 铰接轴在腕部坐标系
  - `obj_angle` (1): 当前铰接角度
  - `obj_avel` (1): 铰接角速度
  - `rel_obj_angle` (1): 目标与当前铰接角度差
- 左手观测: 在预训练阶段 `obDim_l = 52`（右手的位置+姿态+关节角，用于辅助）
- 双手阶段两手观测维度均为 276，另有全局状态 `gsDim = 110`

**关键设计: 坐标系转换**
- 所有特征都转换到手部局部坐标系（wrist-relative）或物体坐标系（object-relative），避免策略依赖全局状态
- 3D 关节位置目标在物体坐标系中表示，再转换到腕部坐标系作为观测

---

### 奖励函数设计

总奖励由模仿奖励和任务奖励组成: `r = r_im + r_task`

模仿奖励: `r_im = r_p + r_c + r_reg`

#### 1. 手部姿态追踪奖励

**关节位置奖励 (pos_reward):**
```
pos_reward = -||rel_body_pos * finger_weights||^2
```
- `rel_body_pos`: 21 个关节的当前位置与目标位置之差（腕部坐标系下）
- `finger_weights`: 指尖（index 16-20）权重为 4，其余为 1，归一化后缩放
- 代码中: `pos_reward_r_ = -rel_body_pos_r_.cwiseProduct(finger_weights_).squaredNorm()`
- 权重系数 (cfg): `coeff: 3.0`
- 裁剪: `max(-10.0, pos_reward)`

**关节角度奖励 (pose_reward):**
```
pose_reward = -||rel_pose||
```
- `rel_pose`: 目标关节角与当前关节角之差（48 DOF，前 3 DOF 是物体坐标系中的腕部朝向差）
- 注意使用 L1 范数（norm），不是 squared norm
- 权重系数 (cfg): `coeff: 0.2`

#### 2. 接触奖励

**接触匹配奖励 (contact_reward):**
```
contact_reward = k_contact * sum(rel_contacts)
```
- `rel_contacts = final_contact_array * contacts`（element-wise，目标接触与实际接触的交集）
- `k_contact = 1.0 / num_active_contacts`（归一化到 [0,1]）
- 16 个 body parts 的接触状态（二值）
- 接触检测: 遍历手部与物体的碰撞对，跳过非动态物体的接触
- 权重系数 (cfg): `coeff: 1.5`（multi_obj_arti）/ `coeff: 2.0`（left_fixed）

**接触冲量奖励 (impulse_reward):**
```
impulse_reward = sum(final_contact_array * impulses)
```
- `impulses`: 每个 body part 的接触冲量范数
- 裁剪: `min(impulse_reward, obj_weight * 5)`（上限与物体质量成正比）
- 权重系数 (cfg): `coeff: 1.5`

#### 3. 正则化奖励 (负系数，惩罚项)

**手部线速度惩罚 (body_vel_reward):**
```
body_vel_reward = ||bodyLinearVel||^2
```
- 权重系数: `coeff: -0.5`

**手部角速度惩罚 (body_qvel_reward):**
```
body_qvel_reward = ||bodyAngularVel||^2
```
- 权重系数: `coeff: -0.2`

**物体相对速度惩罚 (rel_obj_vel_reward):**
```
rel_obj_vel_reward = ||rel_obj_vel||^2
```
- `rel_obj_vel`: 物体与腕部之间的相对速度（腕部坐标系）
- 权重系数: `coeff: -0.5`（general_two 配置中）

**力矩惩罚 (torque):**
```
torque = ||hand_torque||^2 + 4 * ||wrist_torque||^2
```
- 腕部力矩额外加权 4 倍
- 权重系数: `coeff: -0.0`（实际关闭）

#### 4. 任务奖励

**铰接角度奖励 (obj_angle_reward):**
```
obj_angle_reward = -||final_obj_angle - obj_angle||
```
- 使用 L1 范数
- 权重系数: `coeff: 1.5`（multi_obj_arti 和 general_two）

**铰接角速度惩罚 (obj_avel_reward):**
```
obj_avel_reward = ||obj_avel||^2
```
- 权重系数: `coeff: -0.5`

**物体位置保持 (obj_pos_reward, general_two 阶段):**
```
obj_pos_reward = -||rel_obj_goal_pos||^2
```
- 鼓励物体不要偏离初始位置
- 权重系数: `coeff: 0.2`

**物体线速度惩罚 (obj_vel_reward, general_two 阶段):**
```
obj_vel_reward = ||Obj_linvel||^2
```
- 权重系数: `coeff: -0.5`

**物体角速度惩罚 (obj_qvel_reward, general_two 阶段):**
```
obj_qvel_reward = ||Obj_qvel||^2
```
- 权重系数: `coeff: -0.3`

#### 5. 奖励的阶段性控制

在 `general_two` 环境中，根据 `right_kind_idx` / `left_kind_idx` 的值来控制奖励:
- `kind == 7`: 手处于空闲/过渡状态，所有奖励设为 0
- `kind == 8`: 铰接操作阶段，启用全部奖励（含 obj_angle_reward）
- `kind == 9`: 抓取/搬运阶段，启用手部追踪和物体位置奖励，但关闭 obj_angle_reward
- 其他 kind: 全部奖励设为 0

这实现了根据任务阶段动态切换奖励的机制，确保不同阶段的策略专注于不同目标。

#### 奖励权重总结表

| Reward Term | multi_obj_arti | left_fixed | general_two |
|---|---|---|---|
| pos_reward | 3.0 | 3.0 | 3.0 |
| pose_reward | 0.2 | 0.2 | 0.2 |
| contact_reward | 1.5 | 2.0 | 1.5 |
| impulse_reward | 1.5 | 2.0 | 1.5 |
| body_vel_reward_ | -0.5 | -0.5 | -0.5 |
| body_qvel_reward_ | -0.2 | -0.2 | -0.2 |
| obj_angle_reward_ | 1.5 | 0.0 | 1.5 |
| obj_avel_reward_ | -0.5 | 0.0 | -0.5 |
| rel_obj_vel_reward_ | N/A | N/A | -0.5 |
| obj_pos_reward_ | N/A | N/A | 0.2 |
| obj_vel_reward_ | N/A | N/A | -0.5 |
| obj_qvel_reward_ | N/A | N/A | -0.3 |
| torque | -0.0 | -0.0 | (enabled) |

---

### 训练策略

#### 课程学习（两阶段）

**Phase 1: 预训练（固定基座 + 单手）**
- `multi_obj_arti`: 右手策略预训练，物体基座固定在桌面上
- `left_fixed`: 左手策略预训练，同样物体基座固定
- 目标: 学习精细的手指控制和铰接操作
- 优势: 减少物理碰撞点数量，仿真速度更快（RaiSim 速度与碰撞对数量平方成反比）
- 迭代次数: ~50000 iterations
- 训练时左右手分别在独立的物理环境中

**Phase 2: 微调（自由基座 + 双手协作）**
- `general_two`: 加载预训练好的左右手策略权重
- 物体基座不再固定，可以自由移动
- 两只手在同一个物理环境中
- 目标: 学习双手协作（一只手抓住物体不让它动，另一只手执行铰接操作）
- 迭代次数: ~9000 iterations
- 新增物体位置保持奖励（obj_pos_reward）和物体速度惩罚

#### Reference State Initialization (RSI)

- 初始手部姿态从预处理的参考数据中加载
- 添加噪声扰动: 位置噪声 [-0.02, 0.02]m (xy), [0.01, 0.01]m (z)
- 关节角噪声: [-0.05, 0.05]
- 铰接目标角度随机化: [0.5, 1.5] rad

#### Wrist Guidance（腕部引导）

这是 ArtiGrasp 的关键创新之一:
- `root_guided = True`: 启用腕部引导模式
- 将目标手部3D关节位置从物体坐标系转换到世界坐标系
- 用当前物体位姿重新计算腕部目标位置
- 腕部动作的 actionMean 被设置为引导目标（而非上一帧姿态）
- 这使得腕部能够随着物体移动而自动调整

代码中的实现:
```cpp
// Convert final root hand translation from (current) object into world frame
raisim::matvecmul(Obj_orientation_temp_t, final_ee_pos_r_.head(3), Fpos_world_r);
raisim::vecadd(Obj_Position_t, Fpos_world_r);
// Compute distance of current root to initial root in world frame
raisim::vecsub(Fpos_world_r, init_root_r_, act_pos_r);
// Rotate into hand's origin frame
raisim::matvecmul(init_or_r_, act_pos_r, act_or_pose_r);
actionMean_r_.head(3) = act_or_pose_r.e();
```

#### PPO 超参数

| Parameter | Value |
|---|---|
| gamma | 0.996 |
| lambda (GAE) | 0.95 |
| num_learning_epochs | 4 |
| num_mini_batches | 4 |
| learning_rate | 5e-4 |
| clip_param | 0.2 |
| desired_kl | 0.01 |
| lr_schedule | adaptive |
| shuffle_batch | False |
| network | [128, 128] MLP, LeakyReLU |
| output activation | Tanh |

#### Early Termination

- 当观测值出现 NaN 时终止 episode
- 没有基于物体掉落等条件的显式 early termination
- Episode 长度固定: `max_time = 4.0s`, `control_dt = 0.01s`, `simulation_dt = 0.0025s`
- 预抓取步数: 100 步（1 秒），追踪步数: 200 步（2 秒），总共 300 步/episode

#### 仿真参数

- 仿真步长: 0.0025s
- 控制步长: 0.01s（每个控制步执行 4 次仿真积分）
- 摩擦系数: 手指-物体 0.8, 物体-物体 0.8, 默认材料 3.0
- ERP (Error Reduction Parameter): 0.0

---

### 关键设计决策

1. **物体坐标系下的特征表示**: 所有关节位置目标都在物体坐标系中表示和追踪，这使策略能自然适应物体的运动。当物体移动时，手部也相应跟随。

2. **指尖高权重**: 指尖（5 个 tip joints）的位置追踪权重是其他关节的 4 倍。这很关键，因为铰接操作需要指尖精确定位。

3. **接触奖励的归一化**: 接触匹配奖励除以目标接触数 `k_contact = 1.0 / num_active_contacts`，确保不同接触模式之间的可比性。

4. **冲量奖励上限**: 接触冲量有上限 `obj_weight * 5`，防止策略学会施加过大力量。

5. **铰接特征 I_art**: 提供铰接轴方向、到铰接轴的距离、物体各部分质量等信息，显著提升铰接操作成功率。

6. **kind-based 奖励切换**: 在 general_two 阶段，根据当前手的任务角色（空闲/抓取/铰接）动态启用或关闭不同奖励项，避免冲突。

7. **两阶段课程的必要性**: 消融实验表明，不使用课程直接训练双手+自由物体，抓取略好但铰接显著退化。

---

## 代码实现要点

### 代码结构

```
artigrasp/
  raisimGymTorch/raisimGymTorch/env/envs/
    multi_obj_arti/   # Phase 1: right hand pre-training (fixed-base)
    left_fixed/       # Phase 1: left hand pre-training (fixed-base)
    general_two/      # Phase 2: two-hand cooperation (free-base)
    compose_eval/     # Evaluation: Dynamic Object Grasping and Articulation
    fixed_arti_evaluation/    # Evaluation: fixed-base articulation
    floating_evaluation/      # Evaluation: free-base articulation
  raisimGymTorch/raisimGymTorch/algo/ppo/  # PPO implementation
  raisimGymTorch/raisimGymTorch/helper/    # Data processing, label generation
  rsc/
    mano_double/      # MANO hand URDF models (left + right)
    arctic/           # Articulated object URDFs
    meshes_simplified/ # Simplified object meshes
```

### 环境实现 (C++)

- 核心文件: `Environment.hpp`（每个环境目录下）
- 奖励使用 RaiSim 内置的 `Reward` 类，通过 `initializeFromConfigurationFile` 加载配置
- 每步奖励: 先裁剪到 [-10, +max]，然后按权重加和
- 左右手各维护独立的奖励记录器 `rewards_r_` / `rewards_l_`
- 返回两个标量 `rewards_sum_[0]` (right) 和 `rewards_sum_[1]` (left)

### 关键实现细节

**actionMean 的动态更新:**
```cpp
// After step, set next action mean to current pose
actionMean_r_ = gc_r_;
actionMean_l_ = gc_l_;
```
这意味着策略输出的是相对于上一帧姿态的残差动作。

**接触检测逻辑:**
```cpp
for(auto& contact: mano_r_->getContacts()) {
    if (contact.skip() || contact.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
    if (contact.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
    // For articulation (kind==8), only count contacts with the "top" part
    if(right_kind_idx == 8){
        if (contact_list_obj[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() != top_id) continue;
    }
    contacts_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = 1;
    impulses_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = contact.getImpulse().norm();
}
```

**碰撞掩码设计:**
- 手部: `COLLISION(0)` with mask `COLLISION(0)|COLLISION(2)|COLLISION(63)`
- 桌子: `COLLISION(1)`
- 物体: `COLLISION(2)` with mask `COLLISION(0)|COLLISION(1)|COLLISION(2)|COLLISION(63)`
- 这确保手-物体、物体-桌子有碰撞，但手-桌子无碰撞

---

## 与 bh_motion_track 项目的关联

bh_motion_track 项目的特点:
- 双手 WujiHand (5指20DOF) + Boob Cube 操作的 MuJoCo MJX 任务
- Gaussian kernel 奖励 `exp(-(e/sigma)^2)`
- tips-in-object-frame 追踪
- 3-term contact reward (touch + match - FP)
- 乘法组合物体奖励
- weld-based contact guidance curriculum
- Boob Cube 是铰接物体（两半通过 hinge joint 连接）

### 1. 可直接借鉴的技术

**铰接物体的建模方式:**
- ArtiGrasp 将铰接物体拆分为 `bottom` 和 `top` 两个 body，通过 hinge joint 连接，这与 Boob Cube 的设计完全一致
- 左手追踪 bottom 部分，右手追踪 top 部分 -- 这种手-部件对应关系值得借鉴
- 物体姿态用 8 DOF 表示 (3 trans + 4 quat + 1 joint angle)

**物体坐标系下的追踪:**
- ArtiGrasp 的 tips-in-object-frame 思路与 bh_motion_track 完全一致
- 关节位置目标在物体坐标系中定义，随物体旋转自动适应
- 代码中: 先计算 `Position - Obj_Position` 得到世界坐标系偏移，再通过 `matvecmul(Obj_orientation_inv, ...)` 转到物体坐标系

**腕部引导 (Wrist Guidance):**
- 在 bh_motion_track 中可以借鉴这个思路: 将手腕的 actionMean 设为基于当前物体位姿计算的引导目标
- 这对于操作自由浮动物体时保持手-物体相对关系特别有用
- 等价于一种隐式的 impedance control / task-space guidance

**kind-based 奖励切换:**
- 对于 Boob Cube 这种需要先抓取再铰接的任务，可以学习 ArtiGrasp 的 kind 机制
- 根据当前操作阶段动态切换奖励权重，避免抓取和铰接奖励的冲突

**指尖高权重:**
- ArtiGrasp 给指尖 4x 权重，bh_motion_track 可以参考这个设计
- 特别是对于需要精确指尖接触的铰接操作

### 2. 奖励设计上的差异和改进空间

**奖励形式:**
- ArtiGrasp 使用**线性奖励** (`-||e||` 或 `-||e||^2`)，bh_motion_track 使用 **Gaussian kernel** (`exp(-(e/sigma)^2)`)
- Gaussian kernel 的优势: 自动提供远距离时的梯度（不会完全饱和），近距离时的高精度奖励
- 建议: 对于铰接角度追踪，Gaussian kernel 可能比 `-||e||` 更好，因为它在接近目标时提供更强的精细化信号

**接触奖励:**
- ArtiGrasp: 2 项（接触匹配 + 冲量），加权求和
- bh_motion_track: 3 项（touch + match - FP），乘法组合
- ArtiGrasp 缺少对**假阳性接触**（FP）的惩罚，可能导致不必要的接触
- ArtiGrasp 的冲量奖励是一个好的补充 -- 不仅鼓励接触存在，还鼓励足够的接触力
- 改进方向: 可以在 bh_motion_track 中加入类似的 impulse reward，鼓励稳定抓握

**物体奖励组合:**
- ArtiGrasp: 所有奖励项加法组合（线性加权求和）
- bh_motion_track: 物体奖励使用乘法组合
- 乘法组合的优势: 任何一项为零则整体为零，确保所有条件同时满足
- 加法组合的优势: 更稳定的梯度，不会因一项为零而完全失去信号
- 建议: 保持 bh_motion_track 的乘法组合物体奖励，但参考 ArtiGrasp 的单独铰接角度奖励项

**铰接角度奖励:**
- ArtiGrasp 直接将铰接角度误差作为独立奖励项（L1 范数），系数 1.5
- bh_motion_track 可以将铰接角度追踪从物体奖励中分离出来，给予独立且较高的权重

### 3. 训练策略上的参考

**两阶段课程:**
- ArtiGrasp 的课程设计与 bh_motion_track 的 weld-based contact guidance 有相似思想
- ArtiGrasp: 固定物体 -> 自由物体（通过物理约束降低难度）
- bh_motion_track: weld 约束引导 -> 移除约束（通过显式约束降低难度）
- 两者可以结合: 先 weld 约束 + 固定物体 -> 只 weld -> 自由
- ArtiGrasp 的消融实验证实: 不用课程，铰接成功率显著下降

**多物体/多任务训练:**
- ArtiGrasp 在同一策略中同时训练 7 种铰接物体的抓取和操作
- 通过 label 系统管理不同物体和不同任务（抓取/铰接）的训练数据
- 每个 episode 随机分配物体和任务类型
- 这表明单一策略可以泛化到多种铰接物体

**初始状态随机化:**
- ArtiGrasp 对位置和关节角加小范围噪声（2cm 位置，0.05 rad 角度）
- 目标铰接角度在 [0.5, 1.5] rad 范围内随机化
- 这种适度的随机化可以提高鲁棒性，但范围不大

**独立左右手策略:**
- ArtiGrasp 为左右手训练独立的策略网络，而非单一策略控制双手
- 这简化了训练但增加了推理时的协调难度
- bh_motion_track 可以考虑是否采用类似的独立策略设计

**无 Domain Randomization:**
- ArtiGrasp 没有使用显著的 domain randomization（物体质量、摩擦等参数固定）
- 这是因为目标是 sim-only 的动画生成而非 sim-to-real 迁移
- bh_motion_track 如果目标是 sim-to-real，需要额外的 DR
