# Object-Centric Dexterous Manipulation from Human Motion Data

> 论文: Yuanpei Chen, Chen Wang, Yaodong Yang, C. Karen Liu (Stanford / PKU)
> arXiv: 2411.04005
> 代码: ObjDexEnvs (基于 Isaac Gym + rl_games)

## 论文概述

本文提出了一个层次化策略学习框架,利用人手动捕数据训练双手灵巧机器人执行 **物体中心(object-centric)** 的操作任务。核心思想是将问题拆分为:

1. **High-level planner**: Transformer 生成模型,从 ARCTIC 数据集中学习人手腕部运动,条件输入为期望的物体目标轨迹,输出双手腕部 6-DoF 动作序列。
2. **Low-level controller**: PPO 训练的指尖控制器,在高层规划器给出的腕部动作引导下,通过 RL 探索学习手指精细动作,同时学习腕部残差修正。

关键贡献:
- 利用人手腕部运动(而非手指运动)来跨越 embodiment gap -- 腕部运动对手型差异不敏感
- Data Augmentation Loop (DAL): 将 RL 训练中成功的轨迹回馈给高层规划器,迭代提升泛化能力
- 在 10 类日常物体上验证,支持 4 种不同机器手(Shadow, Allegro, Schunk, Ability),并完成了 sim-to-real 转移

## 方法细节

### 问题建模

**任务定义**: 物体中心操作 = 让机器人物理操作物体,使其跟随一条参考 SE(3) 轨迹。

- 物体轨迹: $G = (g_1, g_2, \dots, g_T)$,每个 $g_i = (g_i^R, g_i^T, g_i^J)$,分别为 3D 旋转、3D 平移、关节角度(铰接物体)
- MDP 建模: $\mathcal{M} = (S, A, \pi, \mathcal{T}, R, \gamma, \rho, G)$

**手部模型**: UR10e 机械臂 + 灵巧手(支持 Shadow/Allegro/Schunk/Ability),每只手臂6DoF + 手指若干DoF。双手共 60 维动作空间(单智能体模式)。

**动作空间 (60-dim)**:
- 前 6 维: 右手臂(腕部残差 + 高层规划器输出,通过 IK 转换为关节目标)
- 7-30 维: 右手手指关节目标位置
- 31-36 维: 左手臂
- 37-60 维: 左手手指关节目标位置

**观测空间 (571-dim, full_state 模式)**:
- 双手关节位置 (归一化): 0-83
- 双手腕部线速度/角速度: 84-95
- 双手腕部位置/姿态: 96-109
- 物体 pose + 速度: 110-122
- 物体与目标位置差: 123-125
- 物体底部/顶部 pose + 关节角: 126-141
- 未来 10 帧参考轨迹 (每帧 22 维: obj_pos[3] + obj_rot[4] + left_wrist_pos[3] + left_wrist_rot[4] + right_wrist_pos[3] + right_wrist_rot[4] + obj_joint[1]): 144-363
- 物体关节误差: 364
- 双手指尖状态 (10 个指尖 x 13 维): 365-494
- 全身接触信息: 495+

### 奖励函数设计

论文中的奖励公式 (Appendix B):

$$r_t = \exp\left(-(\lambda_1 \|g_t^T - \hat{g}_t^T\|_2 + \lambda_2 \|g_t^R - \hat{g}_t^R\|_2 + \lambda_3 \|g_t^J - \hat{g}_t^J\|_2)\right)$$

论文给出的参数: $\lambda_1 = 20$ (position), $\lambda_2 = 1$ (rotation), $\lambda_3 = 5$ (joint)。

但代码实现中有一些不同:

#### 物体追踪奖励 (核心)

```python
# code: compute_hand_reward()
# clamp with tolerance before computing reward
object_pos_dist = clamp(||obj_pos - ref_pos|| - 0.05, 0)
object_rot_dist = clamp(2*arcsin(||quat_diff[:, 0:3]||) - 0.1, 0)
object_joint_dist = clamp(|obj_dof - ref_dof| - tol, 0)
# tol = 0.05 for espressomachine, 0.1 for others

object_reward = exp(-2*object_rot_dist - 20*object_pos_dist - 2*object_joint_dist)
```

注意代码中的权重是 `(rot=2, pos=20, joint=2)`,与论文的 `(pos=20, rot=1, joint=5)` 有差异。旋转距离使用四元数差的 arcsin 计算。每个误差项都有一个容差阈值(dead zone),低于阈值的误差被 clamp 为 0。

#### 手部追踪奖励 (条件性)

```python
# with tolerance dead zone
left_hand_pos_dist = clamp(||left_pos - ref_left_pos|| - 0.15, 0)
left_hand_rot_dist = clamp(2*arcsin(||quat_diff[:, 0:3]||) - 0.5, 0)

left_hand_reward = exp(-1*left_hand_rot_dist - 20*left_hand_pos_dist)
right_hand_reward = exp(-1*right_hand_rot_dist - 20*right_hand_pos_dist)
```

手部追踪有更大的容差: 位置 0.15m, 旋转 0.5 rad。这些奖励**仅在 `use_hierarchy=True` 时以乘法形式组合**:

```python
if use_hierarchy:
    reward *= right_hand_reward * left_hand_reward
```

#### 指尖追踪奖励 (消融实验,默认关闭)

```python
if use_fingertip_reward:
    # left hand: average distance of 5 fingertips
    reward *= exp(-20 * avg_fingertip_dist_left)
    # right hand: same
    reward *= exp(-20 * avg_fingertip_dist_right)
```

以**乘法形式**与物体奖励组合。论文实验表明 fingertip reward 反而降低性能,验证了 embodiment gap 使得人类手指运动不适合直接用于机器人。

#### 能量惩罚

```python
jittering_penalty = 0.003 * sum(actions^2)  # defined but NOT used in final reward
energy_penalty = -0.000001 * (right_energy + left_energy)
# energy = (torque * velocity).sum()^2

reward = object_reward + energy_penalty
```

能量惩罚极小(`1e-6`量级),几乎可忽略。动作抖动惩罚定义了但未加入最终奖励。

#### Early Termination 条件

```python
# object falls below table
resets = where(object_pos[:, 2] <= -10.15, 1, reset_buf)
# object position error too large
resets = where(object_pos_dist >= 0.05, 1, resets)
# object rotation error too large
resets = where(object_rot_dist >= 0.5, 1, resets)
# object joint error too large
resets = where(object_joint_dist >= object_joint_reset, 1, resets)  # 0.2 or 0.5
# arm collision detection (left/right contacts with forbidden bodies)
resets = where(is_left_contact, 1, resets)
resets = where(is_right_contact, 1, resets)
# episode length
resets = where(progress_buf >= end_step_buf, 1, resets)  # end_step = init_step + 500
```

接触检测用于终止 episode: 如果手臂的非手指部位(wrist, shoulder, upper_arm, forearm等)与物体发生接触,则重置。

### 训练策略

**PPO 参数**:
- gamma=0.99, tau=0.95, lr=3e-4 (adaptive)
- horizon_length=8, minibatch=4096, mini_epochs=5
- clip=0.2, entropy_coef=0, grad_norm=1
- MLP 网络: [1024, 512, 256], ELU 激活
- max_epochs=50000, normalize_value=True, normalize_input=False

**动作空间处理**:
- 手指: policy 输出 [-1,1] 通过 `scale()` 映射到关节范围,并使用 EMA 平滑 (`actionsMovingAverage=1.0`)
- 手臂: policy 输出腕部残差,叠加到高层规划器的目标上,通过 damped least squares IK 转为关节增量
  - 残差范围: 位置 ±0.02m, 旋转 ±0.1 rad (非层次模式)
  - 若 `use_hierarchy=True`: 位置 ±0.04m, 旋转 ±0.5 rad (无高层引导)

**Domain Randomization** (线性调度, 30000-40000 步内渐增):
- 观测噪声: Gaussian [0, 0.002]
- 动作噪声: Gaussian [0, 0.05]
- 重力扰动: [0, 0.4]
- 腱/关节阻尼/刚度: log-uniform [0.3-3.0]倍
- 刚体质量: uniform [0.5-1.5]倍
- 摩擦系数: uniform [0.7-1.3]倍, 250 buckets
- 物体尺度: uniform [0.95-1.05]倍

**Data Augmentation Loop (DAL)**:
- 物体尺寸随机化: 宽/长/高各 [0.9, 1.1] 倍
- 初始位姿随机化: xy ±2cm, z轴旋转 0-30度
- 目标轨迹扰动: xyz各 ±2cm
- 训练出的成功轨迹回馈给高层规划器微调

**Sim-to-Real 蒸馏**: 使用 DAgger 算法,去除速度等难以真实估计的观测,使用 EMA 低通滤波减少抖动。

### 关键设计决策

1. **腕部/手指解耦**: 最关键的设计 -- 人类腕部运动跨 embodiment 可迁移,手指运动不行
2. **残差腕部动作**: RL 策略在高层规划器基础上学习小幅残差修正,而非从零学习全部腕部动作
3. **IK 转换**: 腕部目标通过 damped least squares IK 转为关节增量,避免直接在高维关节空间学习
4. **容差死区**: 奖励中的 clamp 操作为误差提供了容差区间,避免过度惩罚微小偏差
5. **乘法奖励组合**: 手部追踪和指尖追踪以乘法形式组合,保证了各项同时满足
6. **早停机制**: 物体偏离过大时立即终止,提高训练效率
7. **未来轨迹预览**: 观测中包含未来 10 帧参考轨迹,让策略能预判动作

## 代码实现要点

### 关键文件

| 文件 | 功能 |
|------|------|
| `tasks/dexterous_hand_arctic.py` | 主环境,包含 obs/reward/action/reset 全部逻辑 |
| `cfg/dexterous_hand_arctic.yaml` | 环境参数和 domain randomization 配置 |
| `cfg/arctic/arctic.yaml` | PPO 超参数和网络结构 |
| `high_level_planner/data_utils.py` | ARCTIC 数据加载和预处理 |

### 物体数据来源

使用 ARCTIC 数据集,支持 11 类物体: box, scissors, microwave, laptop, capsulemachine, ketchup, mixer, notebook, phone, waffleiron, espressomachine。每个物体有多条动捕序列(use/grab 两种交互模式)。

### 接触传感

接触传感器安装在手臂非手指部位(wrist_2_link, wrist_1_link, shoulder_link, upper_arm_link, forearm_link),用于检测不期望的碰撞并触发 early termination,而非用于奖励计算。

### 物体姿态表示

物体参考轨迹存储为 7 维向量 `obj_params[:, t, :]`:
- `[0]`: 关节角度 (铰接物体的 DOF)
- `[4:7]`: 位置 (x, y, z)
- 旋转单独存储为四元数 `obj_rot_quat[:, t, :]`

## 与 bh_motion_track 项目的关联

### 1. 可直接借鉴的技术

**奖励设计框架**: 本论文的核心奖励结构 `exp(-w_1*pos_dist - w_2*rot_dist - w_3*joint_dist)` 与 bh_motion_track 使用的 Gaussian kernel `exp(-(e/sigma)^2)` 高度相似。都是指数型奖励,将误差映射到 [0, 1] 范围。

**容差死区 (tolerance clamp)**: 本论文在计算误差时先 clamp 掉一个容差值 (pos: 5cm, rot: 0.1 rad, hand_pos: 15cm),这个技巧值得 bh_motion_track 参考 -- 避免策略浪费精力追求不必要的精度。

**乘法奖励组合**: 本论文中 `use_hierarchy` 模式下 `reward *= hand_reward` 的乘法组合方式,与 bh_motion_track 的乘法组合物体奖励一致。这验证了乘法组合在多项奖励中的有效性。

**未来轨迹预览**: 观测中包含未来 10 帧参考轨迹信息,给策略提供"预见性"。bh_motion_track 如果尚未包含此设计,可以考虑加入。

**Early Termination 策略**: 基于物体位置/旋转/关节偏离阈值的早停机制简单有效。

### 2. 奖励设计差异和改进空间

**接触奖励**: 本论文**没有**显式的接触奖励 -- 接触信息仅用于碰撞检测和 early termination。而 bh_motion_track 使用 3-term contact reward (touch + match - FP),这是一个更精细的设计。本论文的简化可能是因为双手+物体场景中,只要物体追踪到位,接触自然形成。

**指尖追踪**: 本论文的指尖追踪作为消融实验存在(`use_fingertip_reward`),但实验结论是"加入反而降低性能"。这与 bh_motion_track 使用的 tips-in-object-frame 追踪形成对比。关键差异在于:
- 本论文使用的是**全局坐标系**下的指尖位置匹配
- bh_motion_track 使用的是**物体坐标系**下的指尖位置匹配
- 物体坐标系下的追踪更鲁棒 -- 即使物体偏移,指尖相对位置仍然正确

**weld-based contact guidance**: bh_motion_track 的 weld 课程学习在本论文中没有对应概念。本论文通过高层规划器间接提供引导,而非直接约束手指位置。

**奖励权重**: 本论文的位置权重 20 远大于旋转权重 2,说明位置追踪优先级更高。bh_motion_track 可以参考这种非对称权重设计。

### 3. 训练策略上可参考的点

**残差动作设计**: 本论文的腕部残差(在高层目标基础上学习小幅修正)是一个很好的设计模式。bh_motion_track 如果有类似的参考轨迹引导,可以考虑让策略只学习残差。

**Data Augmentation Loop**: 将 RL 训练成功轨迹回馈给上游模型微调的闭环设计非常巧妙,可以提升泛化性。

**Domain Randomization 调度**: 线性调度 (30000 步内从 0 渐增到最大值) 的方式比一开始就加满随机化更稳定。

**Teacher-Student 蒸馏**: 先训练使用完整 state 的教师策略,再蒸馏为使用有限观测的学生策略,是 sim-to-real 的标准流程。

**Horizon Length=8**: 极短的 horizon 配合高频控制 (60Hz),说明策略更侧重反应式控制而非长期规划。
