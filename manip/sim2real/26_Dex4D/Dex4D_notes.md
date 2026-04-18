# Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation

> Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki, Shubham Tulsiani. CMU, 2026.02

---

## 1. Core Problem

灵巧手操作领域面临三重困境:

1. **真机数据瓶颈**: 高自由度灵巧手的遥操作困难且昂贵，数据采集难以规模化
2. **仿真工程成本**: 为每个任务设计专用环境、编写 reward、调 RL pipeline 的工程量不可接受
3. **泛化性不足**: 大多数方法只能处理单任务/单物体，无法跨任务迁移

Dex4D 的核心命题: **能否用仿真学习一个 task-agnostic 的基础操作技能，在部署时通过 video generation model 零样本组合出任意下游任务?**

这与 SimToolReal 的思路高度一致 (都是 task-agnostic goal-conditioned policy)，但在**目标表示**上做出了关键分化: Dex4D 用 3D point tracks 替代 6D pose，获得了更好的 domain robustness 和 correspondence 保持。

---

## 2. Method Overview

Dex4D 将灵巧手操作解耦为两层:

| 层级 | 模块 | 功能 |
|------|------|------|
| 高层规划 | Video Generation + 4D Reconstruction | 语言指令 -> 生成视频 -> CoTracker3 追踪 2D 点 -> Video Depth Anything 提升到 3D -> object-centric point tracks |
| 低层控制 | AP2AP (Anypose-to-Anypose) Policy | 仿真训练的 task-agnostic 策略，接收 paired point encoding (current + target 3D points)，输出 22-DoF 动作 |

**训练流程** (三阶段 teacher-student):

1. **Teacher RL** (PPO, 3 阶段 curriculum):
   - Stage 1-2: 单类别物体 (bottle)，30Hz 控制频率，低 reset 阈值，高 arm speed limit
   - Stage 3: 全 3200 物体，5Hz 控制频率，更大 reset 阈值，更保守学习参数
   - 使用 privileged states + 128 个完整 keypoints

2. **Student DAgger**: 用 DAgger 蒸馏到 student policy
   - 输入: robot proprioception (44D) + last action (22D) + masked paired points (64 个 keypoints x 6D)
   - 输出: 22-DoF 动作 + predicted future joint states (auxiliary)
   - Random plane-height masking 模拟真实世界遮挡

3. **部署**: Wan2.6 生成视频 -> SAM2 分割 -> CoTracker3 追踪 -> Video Depth Anything 估深度 -> 3D point tracks -> closed-loop policy execution

**硬件**: xArm6 (6-DoF) + LEAP Hand (16-DoF) = 22-DoF，单目 RealSense D435 RGBD 相机。

---

## 3. Key Designs

### 3.1 Paired Point Encoding (PPE) -- 通过配对保持 correspondence

**问题**: 如何编码当前物体点和目标物体点，使 policy 能有效区分不同 pose?

**洞察**: 球体旋转时形状不变，只有 correspondence (哪个点要去哪) 能区分不同 pose。独立编码当前/目标点会丢失这一关键信息。

**方案**: 将每个 keypoint 的当前位置和目标位置 concatenate 成 6D pair:

$$q_t^i = [p_t^i; \bar{p}_t^i] \in \mathbb{R}^6$$

N 个 paired points $\{q_t^i\}_{i=1}^N \in \mathbb{R}^{N \times 6}$ 送入 PointNet-style encoder (shared MLP + mean-max mixed pooling)。

**代码实现细节** (来自 `Simple6DPointNetBackbone`):
- 对 current xyz 和 goal xyz 分别做 mean-subtraction (去中心化)
- 将 centered coords + mean coords concatenate 成 12D per point (不是论文描述的 6D)
- 使用 Conv1d MLP [12 -> 128 -> 256] + BN + ReLU
- Max/Mean mixed pooling (前半通道取 max，后半取 mean)
- 全局 MLP [256 -> 256 -> feature_dim]

**Ablation 验证**:

| 编码方式 | Student SR | 说明 |
|----------|-----------|------|
| MLP Point Encoding | 5.7% | 直接 MLP 编码，无 correspondence 无 permutation invariance |
| Decoupled PointNet | 低 | 分离编码，丢失 correspondence |
| **Paired Point Encoding** | **最高** | 保持 correspondence + permutation invariance |

### 3.2 Transformer-based Action World Model -- 联合预测动作和动力学

**问题**: Student policy 在部分观测下如何有效学习?

**方案**: 用 4-layer Transformer encoder 替代简单 MLP:
- 4 个 token: `robot_qpos (22D)`, `robot_qvel (22D)`, `action (22D)`, `object_and_goal_kp (64x6D)`
- 各 token 通过独立 tokenizer 映射到 128D
- Self-attention 层捕捉不同输入模态间的关系
- **Auxiliary world modeling loss**: 从 `robot_qpos` 和 `robot_qvel` 对应的 token latent 解码预测 next-step joint angles 和 velocities

$$\mathcal{L} = \|a_t^{stu} - a_t^{tea}\|_1 + \|\hat{\theta}_{t+1} - \theta_{t+1}\|_1 + \|\hat{\dot{\theta}}_{t+1} - \dot{\theta}_{t+1}\|_1$$

**代码实现细节**:
- Transformer 使用 BERT-style pre-LN 架构: `LN -> MHA -> LN -> FFN`
- FFN 扩展比 4x，GELU 激活，dropout=0.1
- **不使用 positional encoding** (`use_positional_encoding=False`)
- Action 从 `action` token 位置解码 (index=2)
- Future state decoder: 简单 Linear(128, 22)，初始化权重 0.01x

**Ablation**:

| 架构变体 | 效果 |
|----------|------|
| w/o Self-Attention (MLP concat) | SR 下降 |
| w/o World Modeling | SR 下降 |
| **Full model** | **最优** |

### 3.3 Z-axis Reposing -- 消除 z 轴旋转对称性

**问题**: 物体绕 z 轴旋转的对称性增加了 RL 学习难度。

**方案** (`repose_z=True`): 在观测空间中归一化掉物体 z 轴旋转。所有坐标通过 `unpose_z_theta_quat` 旋转到标准坐标系。policy 不需要学习如何应对 z 轴旋转的变化，大幅提高 sample efficiency。

---

## 4. Experiments

### 4.1 仿真实验

**6 个任务**: Apple2Plate, Pour, Hammer, StackCup, RotateBox, Sponge2Bowl

| 方法 | 平均 SR | 平均 TP | 说明 |
|------|---------|---------|------|
| NovaFlow (open-loop) | 低 | 低 | 视频 -> 3D tracks -> Kabsch pose estimation -> motion planning |
| NovaFlow-CL (closed-loop) | +9.2% SR vs NovaFlow | +16% TP | 加入闭环反馈 |
| **Dex4D** | **+16.3% SR vs NovaFlow-CL** | **+10.4% TP** | 完全学习的闭环策略 |

**关键发现**: NovaFlow-CL 的失败主要来自 (1) 手指缺乏反应性导致物体掉落 (2) motion planning 解空间受限导致遮挡 (3) 少量可见点时 Kabsch 算法不稳定。

### 4.2 真机实验

**4 个任务**: LiftToy, Broccoli2Plate, Meat2Bowl, Pour (所有物体均 unseen)

| 方法 | 平均 SR |
|------|---------|
| NovaFlow-CL | 基线 |
| **Dex4D** | **+22.5%** |

NovaFlow-CL 在 Pour 任务上 SR=0% (手指遮挡严重，Kabsch 无法估计正确旋转)。Dex4D 即使只剩 <10 个可见点仍能 robust 工作。

**主要失败模式**: CoTracker3 在 (1) 大幅物体移动 (2) 相似纹理邻域 (3) 初始追踪点被遮挡时丢失追踪。

### 4.3 泛化性测试

虽然仅在单物体场景的仿真中训练，Dex4D 在真机上泛化到:
- 未见物体类型和 pose
- 不同背景和相机视角
- 不同任务轨迹
- 外部干扰

### 4.4 训练资源

| 阶段 | 训练步数 | 时间 | 硬件 |
|------|---------|------|------|
| Teacher Stage 1-2 | 15k + 10k | ~2-3 天 | 1x RTX A6000 |
| Teacher Stage 3 | 25k | (包含在上述) | 1x RTX A6000 |
| Student DAgger | 25k | ~20 小时 | 1x RTX A6000 |

---

## 5. Related Work Analysis

### 5.1 Video-Based Robot Learning

Dex4D 在 video-based manipulation 中的定位:

| 方法 | 视频用途 | 动作映射 | 闭环 | 灵巧手 |
|------|---------|---------|------|--------|
| Gen2Act / Track2Act | human video -> affordance | heuristic retargeting | 否 | 否 |
| NovaFlow | generated video -> 3D flow | Kabsch + motion planning | 可选 | 适配 |
| Dream2Flow | generated video -> 3D flow | motion planning | 否 | 否 |
| **Dex4D** | generated video -> 3D point tracks | **learned sim-to-real policy** | **是** | **是** |

Dex4D 的核心优势: 不依赖 pose estimation + motion planning 的 pipeline，而是端到端学习从 noisy point tracks 到 action 的映射。

### 5.2 3D Policy Learning

| 方法 | 3D 表示 | 用途 | Dex4D 的区别 |
|------|---------|------|-------------|
| PointCloudMatters | point cloud | 场景观测 | Dex4D 用 3D 点作为 goal condition |
| 3D Diffusion Policy | point cloud | 策略输入 | Dex4D 用 paired encoding 保持 correspondence |
| 3D Diffuser Actor | 3D tokens | 扩散策略 | Dex4D 用 RL + DAgger 而非 diffusion |
| GNFactor | neural field | 场景理解 | Dex4D 用稀疏点而非 dense field |

### 5.3 Generalizable Dexterous Manipulation

Dex4D 区别于: (1) 基于优化的方法 (DexGraspNet, 无闭环反馈), (2) 遥操模仿学习 (DexCap, 不能泛化), (3) task-specific RL (in-hand rotation, 无高层规划)。Dex4D 结合了 video generation 高层规划 + task-agnostic RL 低层控制。

---

## 6. Limitations & Future Directions

### 6.1 论文承认的局限

| 局限 | 影响 | 可能方向 |
|------|------|---------|
| 未利用 HOI (Hand-Object Interaction) 数据集 | 缺少人类抓取先验，functional grasp 能力受限 | 用更拟人的灵巧手 (LEAP 只有 4 根粗手指) + 大规模 HOI 数据 |
| 仅支持单物体操作 | 无法处理 bimanual 或 multi-object 场景 | 扩展 AP2AP 到 multi-object |
| 无触觉反馈 | 缺少力觉信息，对精细力控任务不利 | 集成 tactile sensing |
| CoTracker3 在线追踪不稳定 | 主要失败来源: 大位移 / 纹理混淆 / 遮挡 | 更快更 robust 的在线 tracker |

### 6.2 额外观察到的局限

| 局限 | 详情 |
|------|------|
| Video generation 质量依赖 | 不合理的生成视频 (物理违规) 直接导致不可能完成的目标 point tracks |
| 深度估计误差传播 | relative depth estimation + median calibration 是启发式的，几何精度有限 |
| 控制频率低 | 5Hz 部署频率 (12x sim substep)，对高动态任务可能不够 |
| 物体多样性限制 | 训练集来自 UniDexGrasp (3200 objects)，但多为刚性小物体 |
| 无 articulated object 支持 | 关节物体 (抽屉、剪刀) 的 point tracks 描述更复杂 |

---

## 7. Paper vs Code Discrepancies

通过对比论文描述和代码实现 (`Dex4D-Simulation`, `Dex4D-Vision`, `Dex4D-Hardware`)，发现以下差异:

### 7.1 PointNet 编码细节

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Paired point 维度 | 6D (current_xyz + target_xyz) | `Simple6DPointNetBackbone` 实际构造 12D 输入: [mean_curr_xyz, centered_curr_xyz, mean_goal_xyz, centered_goal_xyz]，即加了 mean-subtraction trick |
| PointNet 类型 | "PointNet-style encoder" | Teacher 用 `SimplePointNetBackbone` (纯 MLP + mean pooling)；Student 用 `Simple6DPointNetBackbone` (Conv1d + BN + max/mean mixed pooling)，两者架构不同 |
| Pooling 方式 | "mean-max mixed pooling" | 代码中 teacher 的 `SimplePointNetBackbone` 只用 mean pooling (无 max)；student 才用 max/mean mixed |

### 7.2 Teacher vs Student 架构不一致

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Teacher 网络 | "RL teacher with PPO" | `ActorCriticPointNet`: 用 `SimplePointNetBackbone` (简单 2-layer MLP + mean pooling) 编码 paired points，不是 Conv1d + BN 架构 |
| Teacher keypoint start | -- | 代码硬编码 `kp_start = 197`，student 硬编码 `kp_start = 66`，说明 teacher 有更多 privileged state |

### 7.3 训练细节

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Curriculum 边界 | "Stage 1: 15k, Stage 2: 10k, Stage 3: 25k" | Stage 1-2 共用一个 config (单类别 bottle, 30Hz)，然后手动切换到 Stage 3 config (全类别, 5Hz)。不是自动 curriculum |
| Teacher keypoints | 128 | Config 确认 `numKeypoints: 128`，但 `kpDownsampleRatio: 2`，实际使用 64 个点 |
| Student future state prediction | "predict next-state joint angle and velocity" | DAgger config 中 `future_loss_weights: {robot_qpos: 1, robot_qvel: 1}`，且 `object_and_goal_kp` 被注释掉了 (未预测未来 keypoints) |
| Student `forward()` | 应支持 TorchScript export | `forward()` 方法硬编码了维度 (22, 22, 22)，与 `tokenize()` 方法冗余，是为了 `torch.jit.script` 兼容 |

### 7.4 Hardware 代码

**Dex4D-Hardware 标记 "Coming Soon"**, 仅有 README.md，无实际代码。完整 sim-to-real 部署流程 (action -> motor control, calibration, safety checks) 无法复现。

### 7.5 Vision Pipeline

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| 深度校准 | "scale based on ratio between median depth of frame and initial observation" | `run.py` 中: 如果有 RealSense depth，用 `scale_factor = realsense_median / estimated_median`；否则用默认值 (metric=550.0, relative=0.75) |
| 视频生成 | Wan2.6 with Chinese prompts | 代码不包含视频生成部分，只处理已生成的视频文件 |
| Relative vs Metric depth | "relative depth estimation yields better results" | 代码默认不带 `--metric` flag，使用 relative depth，需要 `1/(depth + 1e-6)` 转换 |

---

## 8. Cross-Paper Comparison

### 8.1 Dex4D vs SimToolReal

两篇论文共享相同的核心 insight: task-agnostic goal-conditioned policy for dexterous manipulation。

| 维度 | SimToolReal | Dex4D |
|------|------------|-------|
| **目标表示** | 6D pose (4 keypoints relative to palm + bounding box scale) | 3D point tracks (64 paired points) |
| **高层规划** | Human video -> SAM 3D + FoundationPose | Video generation -> CoTracker3 + Depth estimation |
| **训练物体** | 程序化生成的 primitive (cuboid + capsule) | UniDexGrasp 3200 real meshes |
| **RL 算法** | SAPG (Symmetric + Asymmetric PPO) + LSTM | PPO + teacher-student DAgger + Transformer |
| **动作空间** | Arm delta + Hand absolute + 强 EMA (alpha=0.1) | Arm delta + Hand absolute (无 EMA) |
| **控制频率** | 10Hz | 5Hz (Stage 3) / 30Hz (Stage 1-2) |
| **目标推进** | 6D pose waypoint 逐个完成 | Point distance threshold 触发下一 waypoint |
| **部署** | FoundationPose 实时追踪 6D pose | CoTracker3 在线追踪 2D 点 + depth backprojection |
| **Perception 依赖** | 需要 object mesh (FoundationPose) | 不需要 mesh，只需初始分割 + 追踪 |
| **真机任务** | 12 种工具 / 24 个任务 | 4 个任务 (规模更小) |
| **核心优势** | 工具使用场景更广，evaluation 更系统 | 无需 mesh，domain-agnostic 表示更 robust |

**关键差异分析**:
- SimToolReal 的 6D pose 需要 object mesh 做 pose estimation，这在真实场景中是强假设
- Dex4D 的 point tracks 是 mesh-free 的，但依赖在线 point tracker 的质量
- SimToolReal 的 perception 瓶颈在 pose tracking 丢失 (43.7% 失败)；Dex4D 的瓶颈在 CoTracker 追踪丢失
- 两者的"失败模式"殊途同归: perception (而非 control) 是 sim-to-real 的真正瓶颈

### 8.2 Dex4D vs DexTrack

| 维度 | DexTrack | Dex4D |
|------|----------|-------|
| **目标** | 从人类运动学参考 tracking control | Task-agnostic pose-to-pose manipulation |
| **参考来源** | MoCap 数据集 (GRAB, TACO) | Video generation + 4D reconstruction |
| **动作空间** | Double integration (加速度级修正) | Direct joint targets (position-level) |
| **网络规模** | 巨型 MLP [8192, 4096, ..., 128] | Transformer 4-layer, 128D tokens |
| **训练策略** | Data flywheel (RL+IL 交替 + homotopy) | 3-stage curriculum + DAgger |
| **物体表示** | PointNet 256D (物体点云特征) | Paired Point Encoding (paired 6D points) |
| **Sim-to-real** | 未验证 | 核心贡献之一 |
| **泛化性** | 3585 条轨迹，跨物体跨技能 | 3200 物体，zero-shot 跨任务 |

**关键差异**: DexTrack 是 tracking 问题 (给定参考轨迹，跟踪执行)，Dex4D 是 goal-conditioned planning 问题 (给定目标 pose，自主规划)。Dex4D 通过 video generation 获得参考，比 DexTrack 的 MoCap 数据更易获取。

### 8.3 Dex4D vs ASAP

| 维度 | ASAP | Dex4D |
|------|------|-------|
| **场景** | Humanoid whole-body loco-manipulation | Single-arm dexterous manipulation |
| **Sim-to-real 方法** | Simulation parameter alignment (物理参数优化) | Domain randomization (observation/action noise, PD gains, friction, etc.) |
| **核心 insight** | 对齐仿真物理参数到真实世界 | 通过 domain-agnostic 表示 (point tracks) 减小 sim-real gap |
| **训练规模** | 大规模全身运动 | 3200 物体的手部操作 |
| **物理精度** | 高 (通过 sysid 精确匹配) | 中 (通过 randomization 覆盖) |

**关键差异**: ASAP 从"让仿真更真实"的角度缩小 sim-real gap，Dex4D 从"让表示更 domain-agnostic"的角度缩小 gap。两种思路互补: ASAP 的 sysid 可以和 Dex4D 的 point track 表示结合使用。

### 8.4 综合对比表

| 方法 | Task-Agnostic | 不需 Object Mesh | Video Planner | 闭环控制 | 灵巧手 | 真机验证 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| SimToolReal | Yes | No (需要 mesh) | No (human video) | Yes | Yes | Yes (24 tasks) |
| DexTrack | No (per-trajectory) | Yes | No (MoCap) | Yes | Yes | No |
| ASAP | No (task-specific) | N/A | No | Yes | No (humanoid) | Yes |
| NovaFlow | Yes | No (需要 mesh) | Yes | Optional | Adapted | Yes |
| **Dex4D** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes (4 tasks)** |

Dex4D 是目前唯一同时满足所有六个条件的方法。但真机任务规模 (4 tasks) 远小于 SimToolReal (24 tasks)，且 hardware 代码未开源。

---

## Takeaway for RL -> FM Practitioner

| # | Takeaway | 原理 | 行动项 |
|---|----------|------|--------|
| 1 | 3D point tracks 是比 6D pose 更好的 manipulation 接口 | 视角无关 + 保持 correspondence + 无需 object mesh，天然适合 sim-to-real | 在灵巧手项目中优先考虑 point-based 表示，而非 pose-based |
| 2 | Paired encoding 保持 correspondence 是关键 | 对称物体的旋转只有 correspondence 能区分 (球的不同旋转 pose shape 完全相同) | 在 goal-conditioned 学习中，总是将 current 和 target 配对编码 |
| 3 | Video generation + point tracking = 可扩展的高层规划器 | Foundation model 已经 encode 了丰富的物体交互先验，只需提取结构化表示 | 关注 video generation model 的进展，这将直接提升 manipulation 系统上限 |
| 4 | Auxiliary world modeling 加速 policy learning | 预测 next-step joint state 迫使 student 学习物理动力学，不只是 action mimicking | 在 DAgger/BC 中加入 future state prediction 作为 auxiliary loss |
| 5 | Perception 是 sim-to-real 的真正瓶颈 | SimToolReal 和 Dex4D 的主要失败模式都来自 tracking 丢失，而非 control failure | 投资更好的 online tracker / pose estimator 比优化 RL policy 收益更大 |
