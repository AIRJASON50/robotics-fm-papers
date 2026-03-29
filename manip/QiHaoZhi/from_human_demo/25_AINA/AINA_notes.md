# AINA - 论文笔记

**论文**: Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations
**作者**: Irmak Guzey (NYU), Haozhi Qi, Julen Urain, Changhao Wang, Jessica Yin, Krishna Bodduluri, Mike Lambeta, Lerrel Pinto (NYU), Akshara Rai, Jitendra Malik, Tingfan Wu, Akash Sharma, Homanga Bharadhwaj (Meta)
**发表**: arXiv:2511.16661, 2025
**项目**: https://aina-robot.github.io
**代码**: https://github.com/facebookresearch/AINA.git

---

## 一句话总结

首个仅用人类佩戴智能眼镜拍摄的 in-the-wild 视频（不使用任何机器人数据，包括遥操作、在线纠正、RL 或仿真）训练多指灵巧手闭环策略的框架，通过 Aria Gen 2 眼镜的立体深度估计和手部姿态提取将人类视频提升到近似 4D，在 3D point cloud 空间学习策略以消除 human-robot embodiment gap。

---

## 核心问题

从人类日常视频学习多指灵巧操控一直是 robotics 社区的长期目标，但面临两大瓶颈：

1. **Embodiment gap**: 人手与机器人手的运动学差异导致人类演示无法直接部署
2. **Annotation scarcity**: in-the-wild 人类视频缺乏可靠的 3D 手部姿态和场景深度标注，使得提取精确的策略学习信号极为困难

此前的工作要么需要在机器人部署环境中采集数据（不可扩展），要么依赖网络视频但仅限于夹爪（无法提供多指手所需的 3D 手部姿态）。AINA 利用 Aria Gen 2 智能眼镜的丰富传感能力（RGB + stereo SLAM 相机 + 实时手部姿态估计），在 **不使用任何机器人数据** 的前提下实现多指灵巧操控。

---

## 方法概述

AINA 框架包含三个阶段：

### 阶段 1: 数据采集

- **In-the-wild 演示**: 人类佩戴 Aria Gen 2 眼镜在任意环境、任意表面上进行操作，录制约 50 条演示/任务，10 Hz
- **In-scene 演示**: 在机器人环境中用两个 RealSense RGB-D 相机录制单条人类演示，用于将 in-the-wild 数据对齐到机器人坐标系

### 阶段 2: 数据处理与对齐

**2D 物体分割与追踪**:
- Grounded-SAM 语言 prompt 分割初始帧物体
- CoTracker 跨帧追踪 2D 物体关键点

**深度估计 (in-the-wild)**:
- Aria 眼镜的两个前置 SLAM 相机提供立体图像
- FoundationStereo 估计 disparity map，结合已知基线 $B$ 和焦距 $f$ 恢复深度：$Z = fB/d$

**手部姿态**:
- In-the-wild: Aria Gen 2 内置 hand tracking (Nimble)，提供 21 个关键点 in device frame
- In-scene: Hamer 从两个相机视角估计 2D 手部姿态，三角化得到 3D

**Domain Alignment (核心对齐模块)**:
1. 计算 in-scene 与 in-the-wild 首帧物体点云质心偏移：$\Delta\mathcal{O} = \mathcal{O}^0_s - \mathcal{O}^0_w$
2. 利用 Kabsch 算法对初始手部姿态计算刚体变换，提取 z 轴旋转 $R_z$
3. 对所有 in-the-wild 演示执行统一变换：

$$\hat{\mathcal{O}}^t_w = R_z \cdot \mathcal{O}^t_w + \Delta\mathcal{O}$$
$$\hat{\mathcal{F}}^t_w = R_z \cdot \mathcal{F}^t_w + \Delta\mathcal{O}$$

### 阶段 3: 策略学习与部署

**Policy 架构**: 基于 Point-Policy 的 transformer 策略

- **Encoder**: Vector Neuron MLP (VN-MLP) -- SO(3) 等变激活层，将 obs_horizon=10 步的 3D 点历史编码为单个向量。每个 3D 点用 3 个 perceptron 表示而非 1 个
- **Backbone**: nanoGPT transformer encoder，4 层 2 头，hidden dim 512。输入 token = 500 个物体点 + 21 个手部关键点（代码中实际使用 5 个 fingertip + 500 个物体点 = 505 tokens，当 `return_fingertips=True` 时）
- **Head**: 2 层 MLP (hidden dim 256)，输出 pred_horizon=30 步的 fingertip 未来位置 (5 x 3 x 30)
- **Position Encoding**: 仅对 fingertip token 使用 learned positional encoding（5 个 embedding），物体点无 positional encoding

**训练**:
- MSE loss on predicted fingertip trajectories
- 3D augmentation: translation [-30cm, 30cm], rotation [-60, 60] around z-axis, scale [0.8, 1.2]
- Gaussian noise on input fingertips [-2cm, 2cm]
- AdamW optimizer, lr=1e-4, weight_decay=1e-4
- 约 2 小时/任务（论文说 2000 epochs，代码默认 1001 epochs）

**部署**:
- 自定义全臂-手 IK 模块将预测的 fingertip 位置转换为 Kinova (7 DoF) + Ability Hand (6 DoF) 的 13 个关节角
- 抓取阈值: 拇指与其他手指距离 < 5cm 时，手指自动靠拢以模拟抓取力

---

## 关键设计

### 1. 3D Point Cloud 作为 embodiment-agnostic 表示

最核心的设计选择是将人类和机器人的观测统一到 3D 点云空间。这消除了：
- 背景差异（不同环境的 RGB 图像差异巨大）
- 视角差异（人类头部运动 vs 固定机器人相机）
- 外观差异（人手 vs 机器人手）

论文通过对比实验证明，RGB-based baseline（Masked BAKU）即使使用相同数据集也表现很差，因为 head motion 导致 in-the-wild 视频的视角与部署时固定相机的视角严重不一致。

### 2. Vector Neuron MLP 编码器

VN-MLP 的关键优势在于 SO(3) 等变性：对输入点云的旋转不会改变编码结果的语义。这对于 in-the-wild 数据特别重要，因为不同演示中 Aria 眼镜的初始朝向是随机的（世界坐标系的 z 轴对齐重力，但水平朝向随机）。

实现细节：VN-MLP 中的 VNLinear 层是 bias-free 的线性变换，VNLeakyReLU 通过投影到一个学习的方向向量来实现等变的非线性激活。

### 3. 最小化假设的 Domain Alignment

只需要一条 in-scene 演示（不到 1 分钟采集）即可将所有 in-the-wild 数据对齐到机器人坐标系。对齐仅用：
- 首帧物体质心的平移
- 首帧手部姿态的 z 轴旋转（Kabsch 算法）

不需要 ArUco marker（EgoZero 需要）、不需要已知物体距离、不限制手部运动方式。

---

## 实验

### 任务覆盖

9 个日常任务：Toaster Press, Toy Picking, Oven Opening, Drawer Opening, Wiping, Planar Reorientation, Cup Pouring, Stowing (长序列), Knob Rotating

### 主要结果 (Table I -- 不同数据配方对比)

| 方法 | 描述 | 表现 |
|------|------|------|
| In-Scene Only | 单条 in-scene 演示训练 | 物体在演示位置附近时可成功，空间泛化差 |
| In-The-Wild Only | 仅 in-the-wild 数据 | 动作严重错位，策略不稳定 |
| In-Scene Transform + ITW | 不用 in-scene 训练但用于对齐 | 误差累积导致 OOD |
| In-Scene Train + ITW | 不用 in-scene 对齐但加入训练 | 旋转不稳定 |
| **AINA (完整)** | in-scene 对齐 + 联合训练 | **最佳空间泛化和成功率** |

### RGB baseline 对比 (Table II)

| 方法 | 任务 1 成功率 | 任务 2 成功率 |
|------|-------------|-------------|
| Masked BAKU | ~47% | 低 |
| Masked BAKU + History | 极低 | 极低 |
| **AINA** | **显著更高** | **显著更高** |

Masked BAKU with History 表现最差，因为 in-the-wild 视频中人头不断运动导致 RGB 历史与部署时固定相机严重不一致。

### 高度泛化 (Table III)

在 3 个不同高度的平台上测试（3.5cm 增量），仅需重新采集一条 in-scene 演示即可适应新高度。大部分任务保持鲁棒。

### 物体泛化

相似形状物体（新烤面包机、白色橡皮擦）可零样本泛化；形状/重量差异大的物体（爆米花包装 vs 玩具）泛化困难。

---

## 相关工作分析

AINA 在灵巧操控学习的光谱中占据一个独特位置：

| 维度 | 之前的工作 | AINA |
|------|-----------|------|
| 数据源 | Lab-constrained (Point-Policy, HuDOR) 或 web video (Track2Act, ZeroMimic) | Smart glass in-the-wild |
| Robot data | 需要遥操作/在线纠正/RL | 零机器人数据 |
| Robot embodiment | 大多只验证夹爪 | 多指灵巧手 |
| 3D 信息 | 有的无 3D，有的需要已知深度 | Stereo 深度估计 + 内置手部追踪 |
| 部署 | 部分开环/需要微调 | 闭环直接部署 |

与最接近的工作对比：
- **EgoZero**: 也用智能眼镜但仅限夹爪，且需要 ArUco marker
- **HuDOR**: 在机器人环境采集人类视频 + 在线 RL 纠正
- **DexCap**: 需要自定义多相机穿戴设备 + 在线纠正
- **Point-Policy**: AINA 的策略架构基础，但需要 in-domain 数据

AINA 的独特贡献是将 smart glass 的丰富传感能力（hand tracking + stereo depth + egocentric view）与 point-based 策略结合，首次实现 **纯人类 in-the-wild 数据 -> 多指手闭环策略** 的完整流程。

---

## 局限性与未来方向

### 作者声明的局限

1. **无力反馈**: 手部姿态估计无法捕捉力信息，这对精确灵巧操控至关重要。可通过 EMG 传感器或力估计手套解决
2. **SLAM/RGB 快门时间差**: Aria Gen 2 的 RGB 和 SLAM 相机快门不完全同步，快速头部运动导致深度-像素错位。当前通过要求数据采集者避免快速头部运动来缓解
3. **部署时使用 RealSense 而非 Aria**: 部署时用 RealSense 相机而非 Aria 流式输入，因为 FoundationStereo 尚无法实时运行

### 从代码推断的局限

4. **单手操作**: 代码仅使用右手 (`hand_poses_in_world["right"]`)，不支持双手任务
5. **固定物体点数**: 所有任务使用固定 500 个物体点 (`num_object_points: 500`)，可能限制复杂场景
6. **归一化策略粗糙**: `calculate_stats` 中使用 mean +/- 1 的 min-max 范围，而非数据驱动的边界，可能导致极端值被截断
7. **无学习率调度器**: 代码中无 learning rate scheduler，固定 lr=1e-4

---

## 论文与代码差异

### 1. 训练 epoch 数不一致

论文称训练 2000 epochs，但代码默认配置为 `train_epochs: 1001`。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/cfgs/train.yaml`

### 2. 实际使用 fingertip 而非 21 关键点

论文公式 (4) 描述输入为 $\mathcal{F}^{t-T_o:t} \in \mathbb{R}^{5 \times 3}$（5 个 fingertip），但代码中 dataset 配置 `return_fingertips: True` 时从 21 个手部关键点中提取 5 个 fingertip (index=8, middle=12, ring=16, pinky=20, thumb=4)。GPT block size 设为 `num_object_points + 21 = 521`，但实际输入 token 数为 500 + 5 = 505 当 `return_fingertips=True` 时。block_size 值是论文/代码的历史遗留。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/dataset/sequential_points.py` (L169-183)

### 3. 损失函数可选分布预测模式

论文只描述了 MSE loss (公式 5)，但代码实现了两种模式：
- `predict_distribution: False` -- 直接 MSE 回归（默认配置）
- `predict_distribution: True` -- 预测 Normal 分布，使用 negative log-likelihood loss，stddev 可按 schedule 衰减

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/learning/networks/policy_head.py` (L42-78)

### 4. 选择性监督物体点

论文未提及，但代码通过 `supervise_object_points` flag 控制是否也对物体点的未来位置施加监督。默认 `False`，即只监督 fingertip 预测。配置中 `return_next_object_points: True` 表示输出包含未来物体点，但通过 mask 机制将物体点的 loss 权重置零。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/learning/point_policy/pp_learner.py` (L214-216)

### 5. 非因果 Transformer

代码中 `GPTConfig.causal` 默认为 `False`，attention mask 为全 1 矩阵而非下三角矩阵。这意味着所有 token（物体点 + fingertip）之间进行全连接 self-attention，而非因果 attention。虽然文件名为 "gpt.py"，但实际是一个标准的 transformer encoder。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/learning/networks/gpt.py` (L161)

### 6. 数据增强细节

论文提到 translation [-30cm, 30cm]、rotation [-60, 60]、scale [0.8, 1.2]，但代码中 z 轴 translation 被 clip 到 [-5cm, 5cm]（比 x/y 的 [-30cm, 30cm] 小得多），且 scale 默认为 `False`（仅对输出做 scale=1.0）。z_rotation_type 设为 `centered`（绕质心旋转），而论文未明确说明旋转是绕原点还是质心。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/dataset/sequential_points.py` (L114-155)
配置文件: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/cfgs/dataset/sequential_points.yaml` (L16: `scale: False`)

### 7. Gaussian noise 实现

论文说对 fingertip 加 [-2cm, 2cm] 高斯噪声，代码实际为 0.015 标准差的噪声（约 1.5cm），且对所有手指施加相同的随机偏移（`noise = torch.randn(points.shape[0], 1, 3) * 0.015`，然后 repeat 到所有手指），这是一个 correlated noise 而非 per-finger 独立噪声。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/dataset/sequential_points.py` (L157-167, L193-194)

### 8. 手关键点顺序重映射

代码中有一个 Aria -> On-scene 的手部关键点顺序转换 (`convert_hand_points_order`)，Aria 的关键点顺序与标准 MANO/HaMer 不同。这个映射逻辑在论文中未提及。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/preprocessing/domain_aligner.py` (L37-65)

### 9. 归一化策略

代码中使用 mean-std normalization（`mean_std_norm: True`），而非论文中未详细描述的范围。统计量从所有演示的 object + hand points 联合计算。当使用 min-max 模式时，范围设为 `mean +/- 1`，这是一个较保守的选择。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/aina/utils/vector_ops.py` (L33-70)

### 10. 只提供单任务配置

代码仅提供 `stewing` 任务配置（bowl + toaster oven），论文中 9 个任务的配置未公开。每个任务需要不同的 `text_prompts` 和 `action_dim`（当 `return_fingertips=True` 时固定为 15 = 5*3）。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_AINA/code/cfgs/task/stewing.yaml`

---

## 跨论文比较

### 与同一作者 (Haozhi Qi) 的工作比较

| 维度 | PenSpin (2024) | TwistingLids (2024) | DexScrew (2025) | AINA (2025) |
|------|---------------|-------------------|-----------------|-------------|
| 核心范式 | Sim RL + BC 微调 | Sim RL + 零样本迁移 | Sim RL + 遥操 BC | 纯人类演示 BC |
| 机器人数据 | 45 条开环回放 | 无 (零样本) | 50-72 条遥操轨迹 | **无** |
| 仿真使用 | 训练 oracle + BC 预训练 | 训练策略 | 训练手指运动原语 | **不使用仿真** |
| 传感器 | Proprioception | 视觉 (2 个 3D 点) | Proprioception + 触觉 | 视觉 (3D 点云) |
| 手部类型 | LEAP Hand (16 DoF) | 2x Allegro (32 DoF) | XHand (12 DoF) | Ability Hand (6 DoF) |
| 任务复杂度 | 单技能 (旋转) | 单技能 (拧盖) | 多步 (拧螺丝) | 多任务 (9 个日常任务) |
| Sim-to-real gap | 通过 BC 微调绕过 | 域随机化 | 通过真实数据 BC 绕过 | **无 sim-to-real gap** |
| 关键创新 | 初始状态设计 | Brake-based 摩擦建模 | 简化仿真+技能辅助遥操 | Smart glass + point-based policy |

**趋势分析**: Haozhi Qi 的系列工作呈现一个清晰的演进路线：从依赖仿真 + 少量真实数据（PenSpin），到零样本仿真迁移（TwistingLids），到利用仿真作为遥操作原语（DexScrew），再到 AINA 完全抛弃仿真。这反映了对 sim-to-real gap 问题的不同应对策略。AINA 选择了最激进的路线——完全不使用仿真，但代价是任务复杂度相对较低（日常操控 vs 精密旋转/拧转）。

### 与同批次论文比较

| 维度 | MinBC (2025) | HandelBot (2026) | AINA (2025) |
|------|-------------|-----------------|-------------|
| 机器人平台 | 人形机器人全身 | 双手弹钢琴 | 单臂 + 多指手 |
| 数据来源 | 人类遥操作 | Sim RL + 真实 RL | Smart glass 人类视频 |
| 策略架构 | Choice Policy (多候选+打分) | Residual RL | VN-MLP + Transformer |
| 关键挑战 | 多模态行为建模 | 毫米级精度 sim-to-real | Human-robot embodiment gap |
| 任务类型 | 长序列操控+行走 | 精密按键序列 | 日常抓取/擦拭/倒水 |
| 创新本质 | 推理效率 (单次前向传播 vs Diffusion 多步) | 两阶段真实世界适应 | 数据采集范式 (in-the-wild) |
| 视觉输入 | RGB + 深度 + 本体感觉 | 无视觉 (仅本体感觉) | 3D 点云 |

**关键区分**: AINA 的核心贡献是**数据管线**（如何从 in-the-wild 人类视频提取可用的策略学习信号），而非策略架构本身（基于已有的 Point-Policy）。MinBC 的贡献在**策略学习算法** (Choice Policy)，HandelBot 的贡献在**sim-to-real adaptation**。三者解决的是灵巧操控中不同维度的瓶颈。
