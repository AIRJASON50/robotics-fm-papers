# DexTrack 笔记

DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References
Xueyi Liu et al. (Tsinghua / Qi Zhi / Shanghai AI Lab / UCSD), 2025.02

## 1. Core Problem

训练一个**通用 tracking controller**，从人类运动学参考驱动灵巧手操控多种物体（GRAB 日常物体 + TACO 工具使用，共 3585 条轨迹）。核心挑战：
- 接触动力学复杂，RL 样本效率低
- 需要跨物体、跨技能泛化
- 运动学参考有噪声（穿透、不可达状态）

与 per-task RL / model-based TO 不同，DexTrack 分离**高层运动规划** (kinematic reference) + **底层 tracking controller**。

## 2. Method Overview

### 2.1 整体架构: Data Flywheel

三阶段交替迭代:
1. RL 单轨迹 tracking → 初始 demo 集
2. RL+IL 训练 generalist controller + homotopy optimization 扩充 demo
3. 重复，使用更强 controller + learned homotopy generator 获取更多 demo

### 2.2 RL+IL 混合训练

**Residual action space (double integration)**:

论文说的是 residual，但代码实现是 **double integration**：
```python
delta_delta = speed_scale * dt * raw_action          # network output → acceleration-like
cur_delta = prev_delta + delta_delta                  # accumulate → velocity-like residual
target = kinematic_bias + cur_delta                   # add to reference
```
reset 时 `prev_delta` 和 `cur_delta` 归零，policy 从参考轨迹出发。这比简单 residual 更平滑。

**速度缩放不对称**:
- 全局平移 (前 3 DOF): `glb_trans_vel_scale × dt × action` (scale 通常设 0.1)
- 全局旋转 (3 DOF): `glb_rot_vel_scale × dt × action` (scale 通常设 0.1)
- 手指 (16 DOF): `dofSpeedScale(=20) × dt × action`

手指的有效 action magnitude 比腕部大 200 倍——手指需要快速精细调整，腕部动作要慢且稳。

**IL loss**: MSE(policy_mu, teacher_action)，但有两个关键细节：
1. **系数极小**: `supervised_loss_coef = 0.0005`，比 RL loss 小 2000 倍。IL 是 gentle regularizer，不是主驱动力
2. **Success-weighted**: 只有 teacher 成功 track 的样本才参与 IL loss (gt_succ_flag 过滤)

### 2.3 Homotopy Optimization

解决"RL 单轨迹 tracking 对复杂操作失败率高"问题。

链式推理 (类 chain-of-thought):
- 难轨迹 T_0 直接解不了
- 找路径 T_K(易) → T_{K-1} → ... → T_0
- 每步用前一步的 tracking 结果做下一步的 baseline
- K=3 (最多 3 跳)

Homotopy path 自动发现:
1. Brute-force search: 按轨迹相似度找 K_nei=10 个邻居，迭代转移
2. Learned generator: conditional diffusion model，输入当前轨迹，输出 parent 轨迹分布

## 3. Key Designs (代码揭示的细节)

### 3.1 Action Space — Double Integration (论文未明确)

论文 Eq.2 写的是 `a_n = s_n^b + Σ(Δa_k)`，但代码实现中 `delta_delta_targets` 才是网络输出，`cur_delta_targets` 是其累积和。这意味着：
- 网络学的是 **加速度级别的修正**
- 自动获得平滑特性（需要持续输出才能积累大偏移）
- reset 时清零，天然与参考轨迹对齐

### 3.2 巨型 MLP (论文未提)

默认网络 v4: **[8192, 4096, 2048, 1024, 512, 256, 128]** — 7 层 MLP。这是非常罕见的大规模 RL policy:
- v3: [4096, 2048, 1024, 512, 256, 128]
- v2: [2048, 1024, 512, 256, 128]
- v1: [1024, 512, 256, 128]

ELU 激活, shared actor-critic backbone, mixed precision 训练。

### 3.3 Object Feature: PointNet 256D

预训练 PointNet autoencoder，所有物体的 latent embedding 预计算存为 `obj_type_to_obj_feat.npy`。推理时不需要点云——直接查表拼入 obs。这是跨物体泛化的关键。

### 3.4 RSI 只在首次 Reset

`random_time=True` 仅在 **第一次 reset** 时随机初始化 progress_buf，之后所有 reset 从 frame 0 开始。不同于 DeepMimic 的标准 RSI (每次 reset 都随机)。

### 3.5 物体初始位置随机化 + 轨迹插值 (论文未提)

随机化物体 xy 初始位置后，不是直接跳到参考轨迹，而是**在前 100 帧内线性插值**从随机位置过渡到原始轨迹。避免跳变。

### 3.6 Reward Shaping Scale

所有 reward 乘以 `scale_value=0.01` 进入 value function — 100x 缩放。

### 3.7 关节索引重排 (论文未提)

retargeting 约定和 IsaacGym DOF 顺序不一致，代码中有显式重排:
```python
joint_idxes_ordering = [0..9] + [14..21] + [10, 11, 12, 13]  # thumb → end
```

### 3.8 不同数据集用不同 reward

GRAB 用 `compute_hand_reward_tracking`，TACO 用 `compute_hand_reward_tracking_taco`（更长序列，最大 1000 帧 vs GRAB 的 150/300 帧）。

## 4. 完整 I/O 规格

### 4.1 观测 (469D, 不含 history/future)

| Offset | Dim | Content | Note |
|--------|-----|---------|------|
| 0 | 22 | hand_dof_pos (normalized [-1,1]) | unscale by joint limits |
| 22 | 22 | hand_dof_vel × 0.2 | |
| 44 | 52 | 4 fingertip states (pos[3]+quat[4]+linvel[3]+angvel[3]) | |
| 96 | 6 | palm pose (pos[3] + euler_xyz[3]) | |
| 102 | 22 | prev_actions | |
| 124 | 13 | object state (pos[3]+quat[4]+linvel[3]+angvel[3]×0.2) | |
| 137 | 7 | goal object pose (next-step ref, pos[3]+quat[4]) | |
| 144 | 22 | ref hand qpos (next-step kinematic reference) | |
| 166 | 7 | should_achieve_goal_pose (pos[3]+quat[4]) | |
| ~173 | 22 | cur_delta_targets (accumulated residual) | |
| ~195 | ~18 | additional conditional features | varies |
| ~213 | 256 | object PointNet latent feature | pre-computed |

+ 可选 future obs: 5帧 × 29D = 145D

### 4.2 动作 (22D)

| DOF | Content | Speed Scale | Effective Rate |
|-----|---------|-------------|----------------|
| 0-2 | global translation (tx,ty,tz) | 0.1 × dt | ~0.0017 m/step |
| 3-5 | global rotation (rx,ry,rz) | 0.1 × dt | ~0.0017 rad/step |
| 6-21 | 16 finger joints | 20 × dt | ~0.33 rad/step |

PD control: finger kp=20, kd=1, effort=0.7

### 4.3 Reward

```
r = -0.5 × hand_pose_delta                    # track reference hand pose
    -0.5 × (finger_obj_dist + 2×palm_obj_dist) # approach object
    + goal_hand_rew                            # obj position+orientation tracking (when in contact)
    + bonus                                    # success bonus (goal_dist < 0.05)
```
All × 0.01 reward scale

## 5. Experiments

| | GRAB | TACO |
|---|---|---|
| 训练/测试 | 1072/197 | 1565/751 (4-level) |
| 手 | Allegro 22 DOF (sim), LEAP (real) |  |
| 平台 | Isaac Gym, 8192 envs |  |
| 最佳 success rate | +10% over PPO baseline |  |
| Real transfer | LEAP + Franka + FoundationPose |  |

## 6. Limitations

- Demo 获取计算量大（homotopy 搜索 + 单轨迹 RL）
- 代码质量: 14000+ 行单文件 env，大量条件分支
- 只支持单手 (Allegro 16+6=22 DOF)
- RSI 只在首次 reset，不如标准 RSI 鲁棒
- 速度 reward 缺失（Appendix A.4 明确说运动学参考的有限差分速度不准，故不用）

## 7. Paper vs Code Discrepancies

| 论文描述 | 代码实际 |
|---------|---------|
| "residual action space" | Double integration — 网络输出是加速度级别 |
| "MLP policy" | 巨型 7 层 MLP [8192..128]，非标准小 MLP |
| IL loss "bias the policy's predictions" | coef=0.0005，几乎是 homeopathic dose |
| "RSI" (implied standard) | 只在第一次 reset 随机，之后固定 frame 0 |
| "object feature (256D)" | 预计算 PointNet latent 查表，不是在线编码 |
| reward 公式 (Eq.10-14) | 实际有 hand-object distance penalty + bonus + contact gate |
| 未提 | 物体初始位置随机化 + 100帧轨迹插值 |
| 未提 | 关节索引重排 (thumb → end) |
| 未提 | reward × 0.01 scale |
| 未提 | GRAB 和 TACO 用不同 reward 函数 |

## 8. Cross-Paper Comparison

| | DexTrack | ManipTrans | DexMachina | bh_motion_track |
|---|---|---|---|---|
| 手 | Allegro 22D (单手) | Shadow 24D | 多种 | WujiHand 52D (双手) |
| 物体泛化 | 多物体 (PointNet 256D) | 跨场景 | 单任务 | 单物体 |
| Action | Double-integration residual | Absolute | Residual | Wrist residual + Finger absolute |
| IL | teacher supervision (0.0005 coef) | 无 | Demo augment | 无 |
| 核心创新 | Homotopy bootstrapping | Manipulation transfer | Demo-driven | Contact guidance curriculum |
| Network | 7-layer MLP [8192..128] | 标准 MLP | 标准 MLP | 3-layer [1024,512,256] |
| Real | LEAP+Franka+FoundationPose | 无 | 无 | WujiHand (WIP) |

### 与 bh_motion_track 可借鉴点

1. **Double-integration action**: 比直接 residual 更平滑，可能减少 action rate penalty 的需要
2. **不对称速度缩放**: wrist 用小 scale (慢且稳)，finger 用大 scale (快速精细)，与我们的 wrist_pos_scale/wrist_rot_scale 设计类似但更极端
3. **预计算 object feature**: 如果未来泛化到多物体，PointNet 查表方案已验证有效
4. **轨迹插值平滑**: 随机化初始状态后做 100 帧插值过渡，而非硬跳。我们的 RSI 可以参考
5. **IL 系数极小**: 0.0005 的 IL loss 基本只起正则化作用。如果要给 bh_motion_track 加 IL，可以从极小系数开始

### 代码文件索引

| 文件 | 行数 | 内容 |
|------|------|------|
| `tasks/allegro_hand_tracking_generalist.py` | ~14000 | 核心 env (obs, reward, reset, action) |
| `learning/a2c_supervised.py` | ~3200 | RL+IL 训练算法 |
| `train_pool_2.py` | ~3000 | 多 GPU 训练编排 (data flywheel) |
| `cfg/task/AllegroHandTrackingGeneralist.yaml` | - | Env 配置 |
| `cfg/train/HumanoidPPOSupervised.yaml` | - | 训练超参 |
