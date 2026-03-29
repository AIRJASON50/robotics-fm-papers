# Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation

**Paper**: Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki, Shubham Tulsiani (Carnegie Mellon University)
**ArXiv**: 2602.15828
**Project**: https://dex4d.github.io/

---

## 1. 核心问题与动机

### 1.1 灵巧操控的根本困境

灵巧手操控面临一个三角矛盾:

- **真机遥操采集**: 数据昂贵且难以规模化, 尤其灵巧手的高自由度使得遥操本身就非常困难
- **仿真学习**: 可规模化, 但需要为每个任务单独设计环境和奖励函数, 工程成本极高
- **泛化能力**: 多数方法只能处理单一任务/单一物体, 缺乏跨任务迁移能力

### 1.2 Dex4D 的切入点

Dex4D 的核心洞察: **与其为每个任务设计特定的 RL 环境, 不如在仿真中学习一个 task-agnostic 的通用操控技能, 然后在部署时通过 "指令" 来复用**。

这个 "指令" 的形式是 **4D point tracks** -- 物体表面关键点在 3D 空间中随时间变化的轨迹。任何操控任务 (翻转、提起、旋转、放置) 都可以被统一表达为 "把物体上的这些点从当前位置移到目标位置"。

### 1.3 与已有工作的关键区别

| 方法 | 目标表示 | 训练方式 | 部署 |
|------|----------|----------|------|
| Task-specific RL | 任务奖励函数 | 每任务独立训练 | 需微调 |
| DexCap (遥操) | 人手动作 | 人类演示 | 昂贵 |
| PointFlowMatch | 2D flow | 扩散模型 | 缺深度信息 |
| **Dex4D** | **3D point tracks** | **AnyPose-to-AnyPose 仿真** | **Zero-shot** |

Dex4D 与 PointFlowMatch 的关键区别在于使用 **3D** 而非 2D 的 point tracks, 这使得策略的 observation space 与视角无关 (domain-agnostic), 大幅简化了 sim-to-real transfer。

---

## 2. 方法论

### 2.1 总体 Pipeline

```
[Video Generation Model]       [Real-Time Camera]
        |                              |
        v                              v
  生成视频 (goal)              在线 RGBD 流
        |                              |
        v                              v
  Offline Point Tracking        Online Point Tracking
  (CoTracker3 + Depth)         (CoTracker3 Online + RealSense Depth)
        |                              |
        v                              v
  Goal 3D Point Tracks        Current 3D Point Tracks
        \                            /
         \                          /
          v                        v
     Sim-Trained Policy (AnyPose-to-AnyPose)
                    |
                    v
              Robot Actions (22 DOF)
```

整体流程分为三大阶段:

1. **Simulation Training** (离线): 在 Isaac Gym 中训练 AnyPose-to-AnyPose (AP2AP) 策略
2. **Vision Pipeline** (部署时): 使用视频生成模型 + 4D point tracking 提取目标轨迹
3. **Hardware Deployment** (部署时): 将策略 zero-shot 迁移到真机

### 2.2 AnyPose-to-AnyPose (AP2AP) 策略

#### 2.2.1 核心思想

AP2AP 将所有操控任务统一为一个问题: **给定物体当前 keypoints 位置和目标 keypoints 位置, 输出关节动作使物体从当前 pose 移动到目标 pose**。

关键点从物体 mesh 表面采样 (论文中使用 64-128 个点), 并在仿真中通过刚体变换计算其当前和目标的世界坐标。

#### 2.2.2 Observation Space

策略的观测由以下部分组成 (以 student policy 为例):

| 组件 | 维度 | 说明 |
|------|------|------|
| Robot DOF position (unscaled) | 22 | 6 DOF 臂 + 16 DOF LeapHand |
| Robot DOF velocity | 22 | |
| Previous action | 22 | |
| Object keypoints (3D) | N x 3 | N=64, 物体当前 keypoint 位置 |
| Goal keypoints (3D) | N x 3 | N=64, 目标 keypoint 位置 |

总维度: 66 + N x 6 (例如 N=64 时为 450 维)

对于 teacher policy (full state), 额外包含:
- DOF torques (22)
- Fingertip states (4 x 13 = 52)
- Fingertip force/torques (4 x 6 = 24)
- Right hand state (13)
- Object full state (pos/rot/vel, 13)
- Goal-object delta (pos/rot, 7)
- Visual feature (64)
- Fingertip-to-object vectors (4 x 3 = 12)

#### 2.2.3 Action Space

- **22 维**: 6 DOF xArm6 关节角 + 16 DOF LEAP Hand 关节角
- **控制模式**: 位置控制, actions 通过 `actionsMovingAverage` 平滑后设为关节目标位置
- **频率**: 30 Hz (controlFrequencyInv=2) 或 5 Hz (Stage 3, controlFrequencyInv=12)

#### 2.2.4 Z-axis Reposing

一个重要的工程设计: **repose_z** -- 将物体的 z 轴旋转归一化掉。

```python
# z_theta = 物体相对于参考抓取的 z 轴旋转角
unpose_z_theta_quat = quat_from_euler_xyz(0, 0, -z_theta)
# 所有观测 (keypoints, hand pos) 在 "unposed" 坐标系下表达
```

这确保了策略不需要学习围绕 z 轴的对称性, 大幅提高样本效率。在部署时同样需要做这个变换。

### 2.3 三阶段训练

#### Stage 1-2: 单类别 Teacher 训练

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: `ActorCriticPointNet` -- Actor 和 Critic 共享一个 PointNet backbone 处理 keypoints, 然后接各自的 MLP
- **物体**: 先在单一类别 (如 `bottle`) 上训练
- **Policy architecture**: [1024, 1024, 512, 512] MLP + PointNet backbone
- **环境数**: 4096 并行
- **Episode length**: 400 steps
- **关键超参**: lr=3e-4, gamma=0.96, clip=0.2, init_noise_std=0.8

#### Stage 3: 全类别 Teacher 训练

- 从 Stage 1-2 checkpoint 继续训练
- 扩展到所有物体类别 (`object_cls_list: null`)
- 降低控制频率到 5 Hz (controlFrequencyInv=12)
- 调整奖励权重: 关闭 obj_finger/obj_hand, 增大 finger_curl_reg (10x) 和 action_penalty (5x)
- 使用 UniDexGrasp 数据集中上千种物体

#### DAgger Distillation: Student 训练

- **Algorithm**: DAgger (Dataset Aggregation)
- **Student**: `ActorPointNetTransformer` -- 基于 Transformer 的架构, 将不同 observation 分量 tokenize 后通过 4-layer Transformer 处理
- **Teacher**: Stage 3 训练的 `ActorCriticPointNet` (frozen)
- **Loss**: L1 behavioral cloning loss + 可选的 future state prediction auxiliary loss
- **Masking**: Student 只看到带噪声的部分 keypoints (模拟真实视觉追踪的遮挡/噪声)

Student 的 Transformer 架构:

```
Input tokens:
  - robot_qpos (22 -> 128)  via Linear
  - robot_qvel (22 -> 128)  via Linear
  - action     (22 -> 128)  via Linear
  - kp_6d      (N*6 -> 128) via Simple6DPointNetBackbone

4-layer Transformer (token_dim=128)
  |
  v
Output: action token -> 22-dim actions
        (+ optional future qpos prediction)
```

### 2.4 奖励函数设计

奖励函数是分阶段条件激活的 (flag-based):

```python
flag = (obj_finger_dist <= 0.48).int() + (obj_hand_dist <= 0.12).int()
# flag=0: 手远离物体
# flag=1: 手接近但手指未到位
# flag=2: 手和手指都接近物体
```

各奖励项:

| 奖励项 | 权重 | 公式 | 说明 |
|--------|------|------|------|
| obj_finger | 1.0 | $-0.5 \cdot d_{finger}$ | 手指到物体距离 |
| obj_hand | 1.0 | $-0.5 \cdot d_{hand}$ | 手掌到物体距离 |
| goal_obj | 1.0 | $1.4 - 3 d_{goal}$ (flag=2) | 物体到目标 keypoint 距离 |
| success_bonus | 1.0 | $5 / (1 + 10 d_{goal})$ (flag=2, $d_{goal} \leq \theta$) | 接近成功的奖励 |
| terminal_bonus | 1.0 | 10.0 (持续成功) | 连续达标奖励 |
| finger_curl_reg | 1.0/10.0 | $-0.001 \|q_{hand} - q_{curl}\|^2$ | 手指蜷曲正则化 |
| table_collision | 1.0 | $-0.01$ per body below threshold | 碰桌子惩罚 |
| action_penalty | 1.0/5.0 | $-0.01 \|a\|^2$ | 动作平滑 |
| dof_vel | 0 | $-10^{-3} \|\.{q}\|^2$ | 关节速度惩罚 (默认关闭) |
| dof_acc | 0 | $-10^{-8} \|\"{q}\|^2$ | 关节加速度惩罚 (默认关闭) |

核心距离度量:

$$d_{goal} = \frac{1}{N} \sum_{i=1}^{N} \| p_i^{obj} - p_i^{goal} \|_2$$

其中 $p_i^{obj}$ 和 $p_i^{goal}$ 分别是物体和目标的第 $i$ 个 keypoint 的世界坐标。

### 2.5 Domain Randomization

仿真中采用广泛的 domain randomization:

| 随机化项 | 范围 | 调度 |
|----------|------|------|
| 观测噪声 (white) | [0, 0.002] | linear, 40k steps |
| 观测噪声 (correlated) | [0, 0.001] | linear, 40k steps |
| 动作噪声 | [0, 0.05] | linear, 40k steps |
| 重力扰动 | [0, 0.4] m/s^2 | linear, 40k steps |
| 关节阻尼 | [0.9, 1.1]x | linear, 30k steps |
| 关节刚度 | [0.9, 1.1]x | linear, 30k steps |
| 刚体质量 | [0.5, 1.5]x | linear, 30k steps |
| 摩擦力 | [0.7, 1.3]x | linear, 30k steps |

还包含 **物体推动** (push_objects): 每 4 秒随机施加线性/角速度扰动。

---

## 3. 系统架构详解

### 3.1 Vision 模块 (Dex4D-Vision)

#### 3.1.1 功能概览

Vision 模块负责将 RGB-D 视频转化为 3D point tracks。包含以下子模块:

1. **RGBD 采集** (`realsense.py`): RealSense 相机采集 RGB-D 帧
2. **分割** (`segmentation.py`): SAM2 交互式分割, 标注目标物体 mask
3. **视频深度估计** (`run.py`): Video-Depth-Anything 估计单目视频的逐帧深度
4. **离线追踪** (`cotracker3.py`): CoTracker3 离线模式追踪 2D 关键点
5. **在线追踪** (`real_time_tracking.py`): CoTracker3 在线模式实时追踪
6. **4D 可视化** (`visualize_track_4d.py`): 使用 viser 可视化 3D point tracks

#### 3.1.2 4D Point Track 生成流程

**离线 Pipeline** (用于提取 goal tracks):

```
RGBD 首帧 -> SAM2 分割 -> 物体 mask
           -> Video Gen Model -> 生成的操控视频
                                     |
                                     v
                              Video-Depth-Anything -> 逐帧深度图
                              CoTracker3 (offline) -> 2D tracks (T, N, 2)
                                     |
                                     v
                              2D tracks + 深度 + 相机内参 -> 3D tracks
                              AprilTag 标定 -> 世界坐标系
                              异常值过滤 -> 最终 4D point tracks
```

**在线 Pipeline** (用于实时感知):

```
RealSense RGBD 流 -> 首帧 SAM2 分割 -> mask
                  -> CoTracker3 Online -> 实时 2D tracks
                  -> RealSense 深度 -> 实时 3D tracks
                  -> AprilTag 标定 -> 世界坐标系
```

#### 3.1.3 关键实现细节

**CoTrackerOnlineSparsePredictor**: 对 Facebook 的 CoTracker3 做了封装, 支持基于 mask 的稀疏点追踪:

```python
class CoTrackerOnlineSparsePredictor(torch.nn.Module):
    def set_masked_queries(self, video, segm_mask, grid_size):
        # 在 mask 区域内均匀采样 grid 点作为追踪 query
        grid_pts = get_points_on_a_grid(grid_size, self.interp_shape)
        segm_mask = F.interpolate(segm_mask, self.interp_shape, mode="nearest")
        point_mask = segm_mask[0, 0][grid_pts.y, grid_pts.x].bool()
        grid_pts = grid_pts[:, point_mask]  # filter by mask
```

**2D -> 3D 提升**: 使用相机内参将 2D track + depth 转换为 3D:

$$x_{3D} = \frac{(u - c_x) \cdot z}{f_x}, \quad y_{3D} = \frac{(v - c_y) \cdot z}{f_y}, \quad z_{3D} = z$$

**异常值过滤**: 使用 Median Absolute Deviation (MAD) 去除深度错误导致的离群点:

```python
def outlier_inlier_mask(points, z_thresh=3.5):
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    mad = np.median(np.abs(distances - np.median(distances)))
    robust_z = 0.6745 * (distances - np.median(distances)) / mad
    return np.abs(robust_z) <= z_thresh
```

### 3.2 Simulation 模块 (Dex4D-Simulation)

#### 3.2.1 环境设计

基于 NVIDIA Isaac Gym 构建, 继承自 UniDexGrasp 的 `BaseTask`:

- **机器人**: xArm6 (6 DOF) + LEAP Hand (16 DOF) = 22 DOF
- **物体**: UniDexGrasp 数据集 (上千种物体, 包含 mesh 和 keypoint 采样)
- **场景**: 物体放在桌面上, 机器人从上方抓取操控
- **物理**: PhysX (TGS solver, 2 substeps, 8 position iterations)

#### 3.2.2 Keypoint 系统

物体 keypoints 从 mesh 表面通过 FPS (Farthest Point Sampling) 采样:

```python
# 从 mesh 采样 128 个 keypoints, 下采样比 2 -> 实际使用 64 个
numKeypoints: 128
kpDownsampleRatio: 2
```

这些 keypoints 在物体局部坐标系下固定, 运行时通过刚体变换转到世界坐标:

```python
object_keypoints = keypoint_local_to_world(
    self.object_keypoint_buf,  # (N, 3) local coords
    self.object_pos,            # (batch, 3) world position
    self.object_rot,            # (batch, 4) world quaternion
)
```

#### 3.2.3 Asymmetric Actor-Critic 与 Masking

为了桥接 sim 和 real 的 observation gap, 使用非对称 AC:

- **Critic (Teacher)**: 看到完整的 state (包括所有 keypoints, 完整物理状态)
- **Actor (Student)**: 只看到被 mask 过的 keypoints (模拟真实相机的遮挡和追踪失败)

Masking 策略模拟真实场景中的遮挡:

```python
# 根据高度随机 mask keypoints (模拟桌面遮挡)
mask_object_goal_keypoints_random_height_test_time(
    object_kp, goal_kp,
    max_mask_height=0.8,       # 随机高度阈值
    above_plane_mask_prob=0.9,  # 高于平面的 mask 概率
    below_plane_mask_prob=0.05, # 低于平面的 mask 概率
    noise_std=0.005,            # 高斯噪声
)
```

#### 3.2.4 Student 网络 (ActorPointNetTransformer)

这是最终部署到真机的 student 网络, 使用 Transformer 架构:

```
观测分量 -> Tokenization:
  robot_qpos (22) -> Linear -> token (128-d)
  robot_qvel (22) -> Linear -> token (128-d)
  action     (22) -> Linear -> token (128-d)
  kp_6d  (N*6)    -> Simple6DPointNetBackbone -> token (128-d)

4 tokens -> 4-layer Transformer -> 输出 action token
                                -> 可选: future qpos prediction
```

**Simple6DPointNetBackbone**: 自定义的 TorchScript-friendly PointNet 变体, 接收 6D 输入 (当前 xyz + 目标 xyz), 使用:
1. 减均值中心化 (subtract mean)
2. Conv1d MLP: 12 -> 128 -> 256
3. Max/Mean 混合聚合
4. Global MLP: 256 -> 256 -> feature_dim

**Future State Prediction**: 作为辅助损失, 使用 Transformer 中间 token 的 latent 来预测下一步的 robot_qpos 和 robot_qvel, 帮助学习物理动力学。

### 3.3 Hardware 模块 (Dex4D-Hardware)

**当前状态**: 代码仓库标记为 "Coming Soon", 仅包含 README.md, 尚未开源。

根据论文描述, 硬件部署的关键组件:

- **机器人**: xArm6 + LEAP Hand (与仿真一致)
- **相机**: Intel RealSense D455 (RGBD 相机)
- **标定**: AprilTag 外参标定
- **控制频率**: 5-30 Hz (与仿真 controlFrequencyInv 对应)
- **策略推理**: PyTorch JIT 导出后实时推理

部署时的 observation 构建:
1. 从 RealSense 获取 RGBD
2. CoTracker3 Online 在线追踪 2D points
3. 使用深度图 + 相机内参提升到 3D
4. 通过 AprilTag 标定变换到世界坐标系
5. 与 robot proprioception (关节角/速度) 拼接
6. 喂入 student policy 得到 22-dim action

---

## 4. 实验设计与结果

### 4.1 仿真实验

#### 4.1.1 实验设置

- **训练物体**: UniDexGrasp 数据集, 按类别划分 seen/unseen
- **评估指标**: 成功率 (keypoint 距离 < 阈值)
- **Baseline**: DexCap, PointFlowMatch (PFM), 以及不同变体的消融

#### 4.1.2 主要结果

仿真中的 AP2AP 成功率:

| 方法 | Seen 物体 | Unseen 物体 |
|------|-----------|-------------|
| Ours (Teacher) | ~60-70% | ~50-60% |
| Ours (Student) | ~55-65% | ~45-55% |
| PointFlowMatch | ~30-40% | ~20-30% |

(具体数值因任务和物体类别而异)

### 4.2 真机实验

#### 4.2.1 任务设置

真机实验覆盖多种操控任务:
- **Reorientation**: 旋转/翻转物体
- **Pick-and-place**: 抓取放置
- **Long-horizon tasks**: 多步骤任务 (通过视频生成模型生成多段轨迹)

#### 4.2.2 Real-World 成功率

论文报告在多种物体 (瓶子、杯子、罐头等) 上的 zero-shot 部署成功率约为 50-80%, 显著优于 baseline 方法。

#### 4.2.3 泛化能力

Dex4D 展现了对以下方面的泛化:
- **Novel objects**: 训练时未见过的物体
- **Scene layouts**: 不同的桌面布局
- **Backgrounds**: 不同背景
- **Trajectories**: 不同的操控路径

### 4.3 消融研究

关键消融发现:
1. **3D vs 2D point tracks**: 3D 显著优于 2D, 因为避免了视角依赖性
2. **Keypoint 数量**: 64 个 keypoints 已足够, 更多提升有限
3. **Masking 策略**: Random height masking 对 sim-to-real 至关重要
4. **Z-axis reposing**: 对 z 旋转不变性的提升非常大
5. **Multi-object training**: 使用大量物体训练提升了泛化能力

---

## 5. 对灵巧手操控的启示

### 5.1 关于 Sim-to-Real Transfer

Dex4D 的 sim-to-real 策略值得借鉴:

1. **Domain-agnostic representation**: 使用 3D point tracks 而非图像/2D 特征作为观测, 从根本上减少了 sim-real domain gap。不需要视觉 domain randomization (纹理、光照等), 因为 point tracking 天然对这些不变。

2. **Asymmetric AC + DAgger distillation**: Teacher 用完整仿真 state 训练, Student 用带噪声/遮挡的部分观测, 通过 DAgger 蒸馏。这比直接在部分观测上做 RL 效果好得多。

3. **Observation masking**: 随机 mask 掉部分 keypoints 来模拟真实追踪的失败和遮挡, 而不是尝试让追踪器完美工作。

### 5.2 关于训练效率

1. **分阶段训练**: 先单类别后全类别, 先高频后低频。这种 curriculum 策略使得策略先学习基本的接触动力学, 再泛化到多样物体。

2. **Keypoint-based reward**: 使用 keypoint 距离而非 6D pose 距离作为奖励, 天然处理了物体对称性问题。

3. **Flag-based reward staging**: 将奖励条件化为 "手是否接近物体" 和 "手指是否接近物体" 两个 flag, 形成自然的 curriculum -- 先学接近, 再学操控。

### 5.3 与我们项目 (WujiHand) 的关联

| 方面 | Dex4D | WujiHand |
|------|-------|----------|
| 物理引擎 | Isaac Gym (PhysX) | MuJoCo MJX (GPU) |
| 手 | LEAP Hand (16 DOF) | WujiHand (20 DOF) |
| 任务表示 | 3D point tracks | 四元数 + 位置 |
| 训练算法 | PPO + DAgger | PPO (Brax) |
| Observation | Keypoints + proprioception | 关节角 + 物体状态 |
| Sim-to-Real | Point tracking | 直接关节控制 |

**可借鉴的点**:
- **Keypoint-based reward**: 我们的 bh_motion_track 任务已经使用了类似的关键点跟踪, 但 Dex4D 的 random masking 策略值得考虑用于提升鲁棒性
- **分阶段训练**: 先单类别后多类别的策略可用于我们的泛化训练
- **Transformer student**: 如果需要处理变长/部分缺失的 observation, Transformer tokenization 是好的方案

---

## 6. 代码库成熟度与可用性评估

### 6.1 Dex4D-Simulation

**成熟度: 中高**

- **优点**:
  - 完整的训练 pipeline (PPO Stage 1/2/3 + DAgger)
  - 提供了预训练 checkpoint (`example_models/`)
  - 详细的 YAML 配置文件和训练脚本
  - WandB 集成用于实验追踪
  - 支持 JIT 导出

- **不足**:
  - 基于 Isaac Gym (已停止维护, 被 Isaac Lab 取代)
  - Python 3.8 + CUDA 11.8 的旧版本依赖
  - 代码中有大量注释掉的旧代码和 debug 语句 (如 `import pdb; pdb.set_trace()`, 未使用的 import)
  - 缺少自动化的物体数据集准备脚本, 需要手动从 UniDexGrasp 设置
  - 硬编码的物理偏移量 (fingertip position adjustment) 使得换手困难

- **可运行性**: 需要 Isaac Gym 许可证, 依赖较重但理论上可复现

### 6.2 Dex4D-Vision

**成熟度: 中等**

- **优点**:
  - Pipeline 清晰 (RGBD -> 分割 -> 追踪 -> 深度 -> 3D)
  - 支持 offline 和 online 两种追踪模式
  - 使用 viser 的 4D 可视化工具很实用
  - AprilTag 标定代码完整

- **不足**:
  - 部分硬编码的相机参数和过滤条件 (`z < 0.602`, `x < -0.5`)
  - Video-Depth-Anything 的权重需要单独下载
  - SAM2 交互式分割需要 GUI 环境
  - 尚未打包为 pip 可安装包 (README 中提到即将更新)

- **可运行性**: 需要 RealSense 相机才能运行在线模式; 离线模式可独立运行

### 6.3 Dex4D-Hardware

**成熟度: 未发布**

- 仅有 README 占位, 标记 "Coming Soon"
- 无任何实际代码
- 这是最大的可用性瓶颈 -- 缺少完整的 sim-to-real 部署代码

### 6.4 总体评估

这是一个典型的 "论文驱动" 的代码库, 足以复现论文实验但距离生产可用有较大差距:
- 三个模块分散在三个仓库, 缺少端到端的 pipeline 脚本
- Hardware 部分缺失使得 sim-to-real 完整流程无法复现
- 依赖过时的 Isaac Gym, 迁移到 Isaac Lab 或其他引擎需要额外工作

---

## 7. 局限性与未来方向

### 7.1 论文承认的局限性

1. **单视角限制**: 当前使用单个相机, 严重遮挡时 point tracking 失败
2. **抓取能力**: AP2AP 策略在需要精确抓取的任务上表现不如 task-specific 方法
3. **Contact-rich 任务**: 对于需要复杂接触模式的任务 (如工具使用), 仅靠 point track 信息不够
4. **视频生成质量**: 依赖视频生成模型的质量, 不合理的生成视频会导致失败

### 7.2 潜在改进方向

1. **多视角融合**: 使用多个相机覆盖更全面的 point tracks, 减少遮挡
2. **触觉反馈**: 融合触觉信息弥补视觉遮挡的不足
3. **自适应控制频率**: 根据任务复杂度动态调整控制频率
4. **MuJoCo MJX 迁移**: 将训练从 Isaac Gym 迁移到 MuJoCo MJX, 利用 JAX 的 JIT 加速
5. **Bi-level planning**: 结合 high-level planner (如 LLM/VLM) 和 low-level AP2AP policy, 处理更复杂的 long-horizon 任务
6. **Real-time depth**: 用 RealSense 原生深度替代 Video-Depth-Anything, 减少延迟

### 7.3 从方法论角度看的开放问题

- **Point track 与 pose 表示的等价性**: 在什么条件下 3D point tracks 严格等价于 6D pose tracking? 对于可变形物体或铰接体, 两者的区别是什么?
- **最优 keypoint 数量**: 不同几何复杂度的物体, 理论上需要多少个 keypoints 才能充分表达 pose?
- **Closed-loop vs Open-loop tracking**: 在线 point tracking 的延迟和误差如何影响闭环控制性能? 是否有理论保证?

---

## 8. 关键公式总结

**Keypoint 距离奖励**:

$$r_{goal} = \frac{1}{N} \sum_{i=1}^{N} \| T(p_i^{local}, q_{obj}, t_{obj}) - T(p_i^{local}, q_{goal}, t_{goal}) \|_2$$

其中 $T(p, q, t) = R(q) \cdot p + t$ 是刚体变换。

**DAgger 损失**:

$$\mathcal{L}_{DAgger} = \mathcal{L}_{BC} + \sum_{k \in \mathcal{K}} w_k \cdot \mathcal{L}_{future}^k$$

$$\mathcal{L}_{BC} = \| \pi_{student}(o_t) - \pi_{teacher}(s_t) \|_1$$

$$\mathcal{L}_{future}^k = \| \hat{s}_{t+1}^k - s_{t+1}^k \|_1$$

**Z-axis Reposing**:

$$\tilde{p} = R_z(-\theta_z) \cdot p, \quad \tilde{q} = q_z(-\theta_z) \otimes q$$

其中 $\theta_z$ 是物体在 z 轴上的旋转角。

**2D -> 3D 提升**:

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} (u - c_x) \cdot d / f_x \\ (v - c_y) \cdot d / f_y \\ d \end{bmatrix}$$

其中 $(u, v)$ 是像素坐标, $d$ 是深度值, $(f_x, f_y, c_x, c_y)$ 是相机内参。

---

## 9. 参考资料

- **Paper**: https://arxiv.org/abs/2602.15828
- **Project Page**: https://dex4d.github.io/
- **Dex4D-Simulation**: https://github.com/Dex4D/Dex4D-Simulation
- **Dex4D-Vision**: https://github.com/Dex4D/Dex4D-Vision
- **Dex4D-Hardware**: https://github.com/Dex4D/Dex4D-Hardware
- **UniDexGrasp**: https://github.com/PKU-EPIC/UniDexGrasp
- **CoTracker3**: https://github.com/facebookresearch/co-tracker
- **Video-Depth-Anything**: https://github.com/DepthAnything/Video-Depth-Anything
- **SAM2**: https://github.com/facebookresearch/sam2
