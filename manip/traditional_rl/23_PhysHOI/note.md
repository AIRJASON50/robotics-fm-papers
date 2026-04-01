# PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction

## 论文概述

PhysHOI 是第一个基于物理仿真的全身(含手指)人-物交互(HOI)模仿学习框架。核心目标是：给定一段运动学 HOI 演示（人体 SMPL-X 动作 + 物体轨迹），训练策略控制仿真人形机器人重现该交互，**无需设计任务特定的奖励函数**。

**关键挑战**：物体是被动的，只能通过控制人形体间接操纵；全身 52 个身体部件 + 51x3 DOF 驱动器（其中 30x3 DOF 用于双手）；纯运动学奖励容易陷入局部最优（如避免接触物体、用错误部位触球）。

**核心贡献**：
1. 提出 Contact Graph (CG) 及对应的 Contact Graph Reward (CGR)，显式建模身体部件与物体间的接触关系
2. 所有奖励项采用**乘法组合**，确保任一项不能过小
3. 引入 BallPlay 数据集（8 种篮球技能的全身交互演示）
4. 在 GRAB 抓取和 BallPlay 篮球技能上验证有效性，成功率显著优于 DeepMimic、AMP、Zhang et al.

**平台**：Isaac Gym (GPU pipeline)，基于 ASE 代码库开发。

---

## 方法细节

### 问题建模

**人形模型**：遵循 SMPL-X 运动学树，52 个身体部件，51x3=153 DOF 驱动器。身体简化为胶囊/盒子几何体用于碰撞检测。

**物体表示**：刚体，简化网格做碰撞检测。篮球用球体近似，一般物体用凸分解。

**动作空间**：`a_t in R^{153}`，为 PD 控制器的目标关节旋转。PD 控制器输出力矩驱动关节。

**观测空间**（policy 输入）= 仿真 HOI 状态 `g_t` + 参考 HOI 状态 `h_{t+1}`：
- 仿真状态 `g_t`：人体观测 `o^{sbj}` (root height, local body pos/rot/vel/ang_vel) + 接触力 `o^f` (指尖 contact forces) + 物体观测 `o^{obj}` (local pos/rot/vel/ang_vel)
- 参考状态 `h_{t+1}`：下一帧的运动学 HOI 数据

**策略网络**：Actor-Critic, MLP [1024, 512] + ReLU，输出高斯分布均值，固定方差 (sigma_init = -2.9)。

### 奖励函数设计

总奖励采用**纯乘法组合**：

```
r_t = r_b * r_o * r_ig * r_cg
```

每个子奖励均为 `exp(-lambda * MSE(...))` 形式（Gaussian kernel with linear exponent）。

#### 1. Body Motion Reward `r_b`

```
r_b = r_p * r_r * r_pv * r_rv
```

| reward term | formula | weight (lambda) | note |
|---|---|---|---|
| `r_p` (body position) | `exp(-lambda_p * MSE(key_pos, ref_key_pos))` | 50.0 | key bodies: Head, knees, elbows, ankles, fingertips (17 bodies) + root |
| `r_r` (body rotation) | `exp(-lambda_r * MSE(body_rot, ref_body_rot))` | 50.0 | root_quat + 51*3 dof_pos concatenated |
| `r_pv` (pos velocity) | `exp(-lambda_pv * MSE(...))` | 0.0 | **disabled** in config (weight=0, code sets error=0) |
| `r_rv` (rot velocity) | `exp(-lambda_rv * MSE(dof_vel, ref_dof_vel))` | 0.0 | **disabled** in config |

实际上 body reward 只使用位置和旋转追踪，速度项被禁用。

#### 2. Object Motion Reward `r_o`

```
r_o = r_op * r_or * r_opv * r_orv
```

| reward term | formula | weight (lambda) | note |
|---|---|---|---|
| `r_op` (obj position) | `exp(-lambda_op * MSE(obj_pos, ref_obj_pos))` | 1.0 | |
| `r_or` (obj rotation) | `exp(-lambda_or * MSE(...))` | 0.0 | **disabled** (BallPlay no rotation data) |
| `r_opv` (obj pos vel) | `exp(-lambda_opv * MSE(obj_vel, ref_obj_vel))` | 0.0 | **disabled** |
| `r_orv` (obj rot vel) | `exp(-lambda_orv * MSE(...))` | 0.0 | **disabled** |

实际上 object reward **仅使用位置追踪**。物体权重 (lambda_op=1.0) 远小于人体权重 (lambda_p=50.0)。

#### 3. Interaction Graph Reward `r_ig`

```
r_ig = exp(-lambda_ig * MSE(ig, ref_ig))
```

- `lambda_ig = 20.0`
- IG 定义：从物体到各 key body 的位置向量差，即 `key_pos[i] - obj_pos`
- 本质上是一种**相对位置**约束，强化手-物之间的空间关系

#### 4. Contact Graph Reward `r_cg` (core contribution)

代码中实现为**简化版 CG reward**，使用力检测近似接触检测（Isaac Gym GPU pipeline 不提供碰撞检测 API）：

```python
# body contact: check non-hand body parts have NO contact
# body ids: pelvis, hips, knees, torso, spine, chest, neck, head, shoulders, elbows
contact_body_ids = [0,1,2,5,6,9,10,11,12,13,14,15,16,33,34,35]
body_contact = all(|force| < 0.1)  # =1 when NO contact on these bodies

# object contact: check if object HAS contact (force on x,y axes > 0.1)
obj_contact = any(|tar_force[x,y]| > 0.1)  # =1 when object IS in contact

# CG reward = penalize wrong body contact * penalize wrong object contact
rcg1 = exp(-ecg1 * w['cg1'])  # w['cg1'] = 5.0
rcg2 = exp(-ecg2 * w['cg2'])  # w['cg2'] = 5.0
# ecg1 = |body_contact - 1.0|  (penalize if non-hand body touches anything)
# ecg2 = |obj_contact - ref_contact|  (match reference contact label)
rcg = rcg1 * rcg2
```

**关键设计**：
- **cg1 (body contact)**：惩罚非手部身体部件产生接触。参考标签始终为 1（"no body contact"），即**永远不允许**躯干/头/肘等与物体接触。
- **cg2 (object contact)**：匹配物体的接触状态与参考标签（数据中标注的 hand-ball 接触时间）。
- 两个 CG 项也是**乘法组合**。
- BallPlay 中：`cg1`=5.0, `cg2`=5.0。Fingerspin 特例 `cg2`=0.01（因为聚合 CG 对手指级操作不够精细）。

**CG 的作用**（论文 ablation 验证）：
- 没有 CGR 时，策略会学到：用头顶球、用手腕碰球而非手指、不敢抓物体、撑桌子保持平衡等局部最优
- CGR 有效引导正确的接触行为

#### 奖励权重汇总 (BallPlay default config)

| parameter | value | description |
|---|---|---|
| `p` | 50.0 | body key position tracking |
| `r` | 50.0 | body rotation tracking |
| `pv` | 0.0 | body pos velocity (disabled) |
| `rv` | 0.0 | body rot velocity (disabled) |
| `op` | 1.0 | object position tracking |
| `or` | 0.0 | object rotation (disabled, no data) |
| `opv` | 0.0 | object pos velocity (disabled) |
| `orv` | 0.0 | object rot velocity (disabled) |
| `ig` | 20.0 | interaction graph |
| `cg1` | 5.0 | body contact penalty |
| `cg2` | 5.0 | object contact matching |

### 训练策略

**状态初始化 (State Init)**：
- 默认使用 `Start`（从第一帧初始化），不使用 Reference State Initialization (RSI)
- 原因：HOI 数据可能有严重碰撞导致物体被弹飞
- 可选 `Random`（随机帧初始化）、`Hybrid`（混合）

**Early Termination**：
- 人体 root 高度低于 `terminationHeight=0.15m` 时终止
- 论文附录提到还有物体偏离阈值 0.5m、人体偏离阈值的终止条件（代码中只实现了高度终止）
- `progress_buf > 1` 才会触发（避免初始帧误终止）

**控制频率**：
- 仿真运行 60 Hz
- 策略采样 30 Hz (`controlFrequencyInv=2`)
- 数据帧率 25 FPS，通过 `dataFramesScale=1.2` 线性插值到 30 FPS

**PPO 超参数**：
- `gamma=0.99`, `tau=0.95` (GAE)
- `lr=2e-5`, constant schedule
- `horizon_length=32`, `mini_epochs=6`
- `minibatch_size=16384`, `e_clip=0.2`
- `entropy_coef=0.0` (no entropy bonus)
- `bounds_loss_coef=10` (action bounds regularization)
- 2048 parallel environments
- GRAB: 5000 epochs, BallPlay: 15000 epochs
- Episode length = motion sequence length (~40 frames config, actual = data length)

**无 Domain Randomization**：没有对物理参数做随机化，但推理时发现对不同球大小有一定泛化能力。

### 关键设计决策

1. **乘法组合奖励**：这是 work 的核心之一。乘法确保所有子项都不能太小，避免"顾此失彼"。任何一项接近 0 都会使总奖励趋近 0。
2. **Contact Graph Reward**：解决了纯运动学奖励的局部最优问题。两个关键子项：(a) 惩罚错误身体部位接触物体，(b) 匹配参考的接触时序。
3. **Aggregated CG**：将 52 个身体部件聚合为 2-3 个节点（hands / rest body / object），大幅降低复杂度且减少标注噪声影响。
4. **Interaction Graph**：key body 到物体的相对位置向量，提供了比独立追踪人体和物体更强的空间约束。
5. **力检测近似接触**：由于 Isaac Gym GPU pipeline 不支持碰撞检测 API，用 net contact force 阈值 (0.1) 近似判断接触状态。

---

## 代码实现要点

### 奖励计算（`compute_humanoid_reward` 函数）

位置: `/home/l/ws/doc/paper/manip/PhysHOI/PhysHOI/physhoi/env/tasks/physhoi.py` L1316-1454

```python
# Total reward: multiplicative combination
reward = rb * ro * rig * rcg

# Body reward: multiplicative
rb = rp * rr * rpv * rrv
# where rp = exp(-MSE(key_pos, ref_key_pos) * 50.0)
#       rr = exp(-MSE(body_rot, ref_body_rot) * 50.0)

# Object reward: multiplicative
ro = rop * ror * ropv * rorv
# where rop = exp(-MSE(obj_pos, ref_obj_pos) * 1.0)

# IG reward
rig = exp(-MSE(ig, ref_ig) * 20.0)
# ig = key_pos[i] - obj_pos (relative position vectors)

# CG reward (simplified, force-based contact detection)
# body_contact: 1 if no non-hand body part has force > 0.1
# obj_contact: 1 if object has force > 0.1 on x or y axis
rcg = rcg1 * rcg2
# rcg1 = exp(-|body_contact - 1.0| * 5.0)  -- penalize body contact
# rcg2 = exp(-|obj_contact - ref_contact| * 5.0)  -- match ref contact
```

### 观测空间结构

policy 的 obs_buf = humanoid_obs + task_obs (object in local frame) + ref_hoi_data (next frame)

**humanoid_obs** (from `compute_humanoid_observations_max`):
- root height (1)
- local body positions (51*3 = 153, excluding root)
- local body rotations (52*6 = 312, tan_norm format)
- local body velocities (52*3 = 156)
- local body angular velocities (52*3 = 156)
- contact body forces (10*3 = 30, fingertip forces)

**task_obs** (from `compute_obj_observations`):
- local object position (3)
- local object rotation (6, tan_norm)
- local object velocity (3)
- local object angular velocity (3)

**ref_obs**: raw HOI data of next frame (324 + 17*3 = 375 dim)

### HOI 数据格式

每帧 HOI 数据: `[root_pos(3), root_rot(3, exp_map), dof_pos(153), body_pos(52*3=156), obj_pos(3), obj_rot(3), contact(variable)]`

### 关键 body IDs

- `keyBodies` (17): Head, knees, elbows, ankles, 10 fingertips (L/R Index3, Middle3, Pinky3, Ring3, Thumb3)
- `contactBodies` (10): 10 fingertips (用于 force observation)
- CG body contact check (16): pelvis, hips, knees, torso, spine, chest, neck, head, shoulders, elbows

### Early Termination

```python
# Only root height check in code
body_fall = rigid_body_pos[:, 0, 2] < termination_heights  # 0.15m
terminated = body_fall & (progress_buf > 1)
reset = (progress_buf >= max_episode_length - 1) | terminated
```

---

## 与 bh_motion_track 项目的关联

bh_motion_track 项目: 双手 (WujiHand, 5指20DOF) + Boob Cube 操作的 MuJoCo MJX 任务。

### 1. 可直接借鉴的技术

**Contact Graph 思想**：PhysHOI 的 CG 设计理念完全适用。bh_motion_track 已有 3-term contact reward (touch + match - FP)，这本质上就是一种聚合 CG：
- touch reward ~ PhysHOI 的 `rcg2`（鼓励正确接触）
- match reward ~ 更精细的接触体匹配
- FP penalty ~ PhysHOI 的 `rcg1`（惩罚错误身体部位接触）

**Interaction Graph**：从物体到 fingertip 的相对位置向量，与 bh_motion_track 的 `tips-in-object-frame` 追踪理念一致。PhysHOI 证明这种相对位置约束（而非独立追踪人体和物体）对 HOI 任务至关重要。

**乘法组合**：bh_motion_track 已经在物体奖励上使用乘法组合，与 PhysHOI 的设计理念一致。

### 2. 奖励设计差异与改进空间

| 方面 | PhysHOI | bh_motion_track | 分析 |
|------|---------|-----------------|------|
| **kernel 形式** | `exp(-lambda * MSE)` (linear exponent) | `exp(-(e/sigma)^2)` (Gaussian kernel, quadratic) | Gaussian kernel 对小误差更宽容，对大误差惩罚更重。PhysHOI 的线性 exponent 更简单但梯度方向更一致 |
| **物体奖励组合** | 乘法 (r_op * r_or * r_opv * r_orv) | 乘法 | 一致 |
| **总奖励组合** | 纯乘法 (r_b * r_o * r_ig * r_cg) | 部分加法 + 部分乘法 | PhysHOI 的纯乘法更激进，任一项失败则全 0。可考虑对关键项做乘法、辅助项做加法的混合策略 |
| **接触奖励** | 二值接触匹配 + 力阈值 | 3-term (touch + match - FP) | bh_motion_track 更精细，但 PhysHOI 的设计更简洁。**差异在于 PhysHOI 不区分哪个指尖接触，只看聚合节点** |
| **速度追踪** | 代码中实际禁用 (weight=0) | 可能有速度项 | 值得注意: PhysHOI 发现纯位置+旋转追踪就够了，速度追踪对 HOI 场景不那么重要 |
| **物体权重** | lambda_op=1.0 远小于 lambda_p=50.0 | -- | PhysHOI 物体追踪权重很低，说明物体控制主要靠 IG reward 和 CG reward 间接实现 |

**改进空间**：
- **更精细的 CG 节点**：PhysHOI 的聚合 CG 对手指级操作不够（作者自己在 fingerspin 中降低了 cg2 权重）。bh_motion_track 可以为每个手指定义独立的 CG 节点，提供更精细的接触引导。
- **CG 的课程学习**：PhysHOI 没有做 CG 权重的课程学习，bh_motion_track 的 weld-based contact guidance curriculum 是一个改进方向——先用强约束引导接触，再逐渐放松让策略自主学习。
- **IG reward 的借鉴**：bh_motion_track 的 tips-in-object-frame 已经类似 IG，但可以考虑将其也乘法纳入总奖励（而非加法），强制空间关系不能太差。

### 3. 训练策略的参考

**Reference State Initialization (RSI)**：PhysHOI 默认不用 RSI，因为 HOI 数据有碰撞问题。但对 walkpick 建议使用 Random init。bh_motion_track 如果数据质量好，可以继续使用 RSI 来提升训练效率。

**Early Termination**：PhysHOI 只做了 root height 检查（代码实现），论文提到还有物体偏离检查。bh_motion_track 可以加入物体偏离的 early termination 来提升样本效率。

**控制频率与数据帧率对齐**：PhysHOI 通过线性插值将 25fps 数据对齐到 30Hz 控制频率。如果 bh_motion_track 有不同帧率的参考数据，可以参考这个做法。

**纯 PPO 无 entropy bonus**：PhysHOI 使用 `entropy_coef=0.0`，不鼓励探索，完全依赖 imitation reward 引导。这对单一动作模仿是合理的，但对需要更多探索的任务可能需要调整。

**低学习率**：PhysHOI 使用非常低的学习率 (`2e-5`)，constant schedule。对复杂 HOI 任务可能有助于训练稳定性。
