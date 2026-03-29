# NeuralFeels - 论文笔记

**论文**: Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation
**作者**: Sudharshan Suresh*, Haozhi Qi, Tingfan Wu, Taosha Fan, Luis Pineda, Mike Lambeta, Jitendra Malik, Mrinal Kalakrishnan, Roberto Calandra, Michael Kaess, Joseph Ortiz, Mustafa Mukadam (CMU, FAIR/Meta, UC Berkeley, TU Dresden)
**发表**: Science Robotics 9, eadl0628 (2024); arXiv:2312.13469
**项目**: https://suddhu.github.io/neural-feels
**代码**: https://github.com/facebookresearch/neuralfeels.git

---

## 一句话总结

NeuralFeels 将视觉、触觉和本体感知统一融合到一个在线学习的 neural signed distance field (SDF) 中，在多指灵巧手的 in-hand rotation 过程中实现了对未知物体的同时定位与建图 (SLAM)，在重度遮挡场景下相比纯视觉方法实现了最高 94% 的位姿追踪改善。

---

## 核心问题

灵巧手 in-hand manipulation 中的物体感知面临三个根本挑战：

1. **物体先验知识缺失**: 现有方法大多假设已知物体 CAD 模型，限制了对未知物体的泛化能力
2. **视觉遮挡不可避免**: 多指手在操控过程中必然遮挡物体的大部分表面，纯视觉方法在遮挡严重时失效
3. **多模态融合缺乏统一框架**: 视觉、触觉、本体感知的数据格式和尺度差异大，如何统一表征是一个开放问题

NeuralFeels 的核心贡献在于：将触觉视为"局部视觉" (touch is vision, albeit local)，通过 neural field 作为统一表征，同时解决位姿追踪 (tracking) 和形状重建 (mapping) 的 chicken-and-egg 问题。

---

## 方法概述

### 整体架构

```
Input Stream                    Frontend                       Backend
[RGB-D camera]     -->  SAM segmentation + depth  -->  ┐
[4x DIGIT tactile] -->  Tactile transformer       -->  ├→ Shape Optimizer (neural SDF θ)
[16D joint angles]  -->  Forward kinematics        -->  ├→ Pose Optimizer (pose graph x_t)
                                                       └→ Online alternating optimization
```

NeuralFeels 采用经典 SLAM 的 frontend-backend 架构：

### Frontend: 传感器数据到深度图

1. **视觉分割**: 使用 Segment Anything Model (SAM ViT-L, 308M params) 进行 zero-shot 物体分割。关键技巧是利用机器人运动学计算 grasp center 作为 SAM 的 point prompt (embodied prompting)，无需人工标注
2. **触觉深度估计**: 预训练的 tactile transformer (基于 DPT/ViT, 21.7M params) 将 DIGIT 触觉 RGB 图像转换为接触深度图。在 TACTO 仿真器中用 40 个 YCB 物体生成 10K 随机交互的训练数据，通过 LED 光照随机化实现 sim-to-real 迁移
3. **接触掩码**: 基于背景模板的差分阈值法，过滤非接触区域

### Backend: 交替优化 shape 和 pose

1. **Object model**: 基于 instant-NGP 的 neural SDF `F_{x_t}^θ(p): R^3 -> R`，使用 multi-resolution hash encoding (19 levels, 2 features/level, hashmap 2^23) + 3 层 MLP (hidden_dim=64)
2. **Shape optimizer**: 在固定 pose 下，用 AdamW (lr=2e-4) 梯度下降优化 SDF 权重 θ。维护一个 keyframe bank，每次迭代从中采样 10 帧/传感器。采样混合 surface 点和 free-space 点，使用 truncated SDF loss
3. **Pose optimizer**: 在固定 SDF 下，构建 sliding window pose graph (window_size=3)，使用 Theseus 库的 Levenberg-Marquardt (LM) 二阶优化器求解。包含三个 cost function:
   - SDF loss: surface 点的 SDF 值最小化
   - Pose regularizer: 相邻 keyframe 间的 relative pose 约束
   - ICP loss: 相邻帧 point cloud 的 frame-to-frame 配准

### Visuotactile 融合

触觉与视觉在 backend 中的关键差异：

| 属性 | 视觉 (Realsense) | 触觉 (DIGIT) |
|------|-------------------|--------------|
| n_rays (mapping) | 400 | 5 |
| n_rays (pose tracking) | 300 | 25 |
| free_space_ratio | 0.7 | 0.0 |
| depth_range | [0.3, 1.0] m | [-0.01, 0.05] m |
| loss_ratio | 1.0 | 0.1 (sim) / 0.01 (real) |
| noise_std | 2e-3 (sim), 1e-3 (real) | 2e-3 (sim), 5e-3 (real) |

---

## 关键设计

### 设计 1: Posed neural field -- 翻转 SLAM 范式

传统 neural SLAM (如 iSDF, NICE-SLAM) 估计相机位姿相对于固定 neural field 的变换。NeuralFeels 翻转了这一范式：**传感器位姿已知** (通过机器人运动学和相机标定)，要估计的是**物体在 neural field 中的位姿**。这避免了多传感器之间的相对位姿估计问题。

在实现中，`SDFModel.transform()` 将采样点从世界坐标系变换到物体坐标系 (`neuralfeels/modules/model.py`)，使用 `torchlie.functional.SE3.juntransform` 同时计算变换后的点和位姿 Jacobian。

### 设计 2: 触觉作为局部视觉的统一处理

NeuralFeels 的核心 insight 是将触觉传感器建模为一个具有极小视场角和厘米级深度范围的虚拟相机。通过 TACTO 仿真器的透视相机模型，触觉和视觉数据可以用完全相同的 ray sampling 和 SDF supervision pipeline 处理。

为了适配差异，代码中做了以下调整：
- 触觉只采样 surface 点 (free_space_ratio=0.0)，不进行 free-space carving
- 触觉使用更小的 surface_samples_offset (1e-3 vs 1e-3，但 depth_range 差两个数量级)
- 触觉的 loss_ratio 更低 (0.1 for sim, 0.01 for real)，防止小视场的局部信号主导优化

### 设计 3: 自定义 analytic Jacobian 加速 pose optimization

代码中实现了 TSDF cost function 的 analytic Jacobian (`TSDFCostFunction` in `pose_optimizer.py`)，相比 PyTorch autograd 快 4x。Jacobian 的计算分两步：
1. SDF 对 3D 点的梯度 (通过 grid interpolation 或 neural network backprop)
2. 3D 点对 SE(3) pose 的 Jacobian (通过 torchlie 的 `juntransform`)

两者通过链式法则组合，避免了全图自动微分的开销。

### 设计 4: Embodied prompting for SAM segmentation

SAM 的分割质量高度依赖 prompt 质量。NeuralFeels 利用机器人运动学计算所有指尖位置，将它们投影到图像平面上作为 point prompts。关键技巧：
- 正 prompt: grasp center 的投影 (物体大概率在此处)
- 负 prompt: 各指尖位置的投影 (手指不是物体)
- 从 SAM 的多个输出 mask 中选择面积最接近预设 optimal_mask_size 的那个

### 设计 5: Heightmap temporal blending

触觉深度预测存在噪声，代码中实现了 exponentially weighted moving average 对 heightmap 进行时序平滑 (`blend_heightmaps` in `tactile_depth.py`)。这在论文中未提及但对实际效果很重要。

---

## 实验

### 实验设置

- **硬件**: Allegro hand (16 DOF, left-hand) + 4x DIGIT 传感器 + Intel D435 RGB-D 相机 + Franka Panda 手臂
- **仿真**: IsaacGym (物理) + TACTO (触觉渲染)
- **操控策略**: HORA (proprioception-driven in-hand rotation policy, 20Hz)
- **数据集 (FeelSight)**: 70 experiments, 14 objects, 30s/trial, 5 seeds/trial = 350 trials total

| 类别 | Sim 物体数 | Real 物体数 | 来源 |
|------|-----------|-----------|------|
| Simulation | 8 | - | YCB + ContactDB |
| Real-world | - | 6 | 3D 扫描 (Revopoint, 0.05mm) |

### 主要结果

#### SLAM (unknown objects)

| 指标 | Vision-only (Sim) | VisuoTactile (Sim) | Vision-only (Real) | VisuoTactile (Real) |
|------|-------------------|--------------------|--------------------|---------------------|
| F-score (tau=5mm) | ~69% | ~84% (+15.3%) | ~68% | ~78% (+14.6%) |
| Pose drift (ADD-S) | ~6.0mm | ~4.7mm (-21.3%) | ~6.5mm | ~4.7mm (-26.6%) |
| Median recon error | - | 2.1mm (sim) | - | 3.9mm (real) |

#### Tracking (known objects)

| 条件 | Vision-only | VisuoTactile | 改善 |
|------|-------------|--------------|------|
| Sim avg. | - | 2.3mm | -22.29% |
| Real avg. | - | 2.3mm | -3.9% |

#### Occlusion analysis (Rubik's cube, 200 viewpoints)

- 平均改善: 21.2%
- 最大改善: 94.1% (重度遮挡视角)
- 关键发现: 低遮挡时触觉起 refinement 作用，高遮挡时触觉起 disambiguation 作用

#### Depth noise analysis

在 depth noise factor D 从 0 增加到 50 的过程中，visuo-tactile fusion 始终优于 vision-only，且差距随噪声增加而扩大。

### 失败模式

论文 Table 1 报告了追踪失败次数 (ADD-S > 10mm)。Vision-only 在 sim/real 中分别有更多失败案例，加入触觉后失败次数显著减少。

---

## 相关工作分析

NeuralFeels 处于多个研究领域的交叉点：

### Neural SLAM for robotics

| 方法 | 传感器 | 场景 | 在线? | 位姿估计 |
|------|--------|------|-------|----------|
| iSDF (2022) | RGB-D | 室内场景 | Yes | 已知 |
| NICE-SLAM (2022) | RGB-D | 室内场景 | Yes | Joint |
| iMAP (2021) | RGB-D | 室内场景 | Yes | Joint |
| FingerSLAM (2023) | Touch + RGB-D | 固定物体 | Yes | Joint |
| **NeuralFeels** | **Touch + RGB-D + Proprio** | **In-hand** | **Yes** | **Joint** |

NeuralFeels 是首个在多指 in-hand manipulation 场景中实现 full multimodal SLAM 的方法。

### Tactile perception

与之前的触觉感知工作相比，NeuralFeels 的独特之处在于：
- 不需要物体先验 (vs. tactile localization methods)
- 支持多指同时触觉 (vs. single-finger methods)
- 在线增量式学习 (vs. batch reconstruction)

### 模块化 vs. 端到端

NeuralFeels 采用完全模块化设计：pre-trained foundation models (SAM, DPT) + classical SLAM optimization。这提供了：
- 可解释性：每个模块的输入输出都有明确的物理含义
- 可扩展性：可替换任何单独模块 (如其他触觉传感器、其他场景表征)
- 数据效率：不需要端到端训练数据

---

## 局限性与未来方向

### 作者指出的局限

1. **缺乏 3D 先验**: 初始几秒 SDF 不完整时追踪容易失败。可利用 pre-trained 3D foundation models (如 Zero-1-to-3) 提供初始 shape prior
2. **Sim-to-real gap**: 触觉在真实世界中改善幅度不如仿真 (real: +14.6% vs sim: +15.3% for F-score)。原因包括 DIGIT 弹性体灵敏度降低、RL 策略不稳定导致物体快速运动、本体感知噪声
3. **固定相机设置**: 当前需要外部固定相机，egocentric 或 hand-eye 标定方案可放松此限制
4. **非实时运行**: 当前 1-5Hz，需要在 pose optimizer 和 frontend 上做效率优化

### 从代码推断的额外局限

5. **初始位姿依赖**: SLAM 模式下初始位姿用 ground-truth 初始化 (`init_first_pose` in `trainer.py` L305-307)，虽然论文声明"this is not necessary otherwise"，但对评估公平性有影响
6. **loss_ratio 的手动调参**: 触觉的 loss_ratio 在 real-world 中进一步降低 10x (`sensor.py` L442-443)，说明 sim-to-real 中触觉信号的可靠性显著降低
7. **单物体假设**: 当前只处理手中的单个物体，不支持多物体或变形物体
8. **ICP loss 的脆弱性**: 代码中有大量对 ICP 结果的过滤逻辑 (fitness threshold, inlier RMSE, rotation/translation bounds)，说明 frame-to-frame ICP 在实际中经常给出错误结果

---

## 论文与代码差异

### 1. 触觉 loss_ratio 的二次衰减

论文中未提及触觉的 loss weight 在 real-world 中会进一步降低。但代码中 `DigitSensor.__init__()` (`neuralfeels/modules/sensor.py` L442-443):
```python
if cfg_sensor.tactile_depth.use_real_data:
    self.loss_ratio *= 0.1
```
即 real-world 中触觉 loss_ratio = 0.1 * 0.1 = 0.01，仅为仿真中的 1/10。这解释了为何 real-world 中触觉改善幅度小于仿真。

### 2. Heightmap temporal blending

论文未提及触觉深度预测使用了指数加权时序平滑。代码中 `TactileDepth.blend_heightmaps()` (`neuralfeels/contrib/tactile_transformer/tactile_depth.py` L114-146) 实现了 exponentially weighted moving average，使用可配置的 window size。

### 3. Pose optimization 中的 analytic vs numerical Jacobian 选项

论文只描述了 analytic Jacobian，但代码支持三种模式 (`pose/default.yaml` L24): `analytic`, `numerical`, `autodiff`。默认使用 `analytic`，但保留了对比选项。

### 4. Grid-based SDF interpolation for pose tracking

论文描述 pose tracking (known objects) 使用预计算的 SDF。代码中实现了 `SDFInterp` (`neuralfeels/modules/model.py` L179+) -- 一个 regular grid trilinear interpolator，而非 neural network forward pass。在 SLAM 模式中，每次 pose optimization 前会将 neural SDF 查询成 grid (`step_pose` in `trainer.py` L1926-1931) 传给 `frozen_sdf_map`。这是一个重要的效率优化，论文中仅隐含提及。

### 5. Keyframe selection 的 loss-weighted sampling

论文提到 keyframe replay 使用"weighted random sampling of past keyframes based on average rendering loss"。代码实现 (`select_keyframes` in `trainer.py` L758-801) 更复杂：
- 最新 2 帧必选
- 剩余帧按 loss distribution 加权采样
- 过滤掉超过 2*window_size 之前的 outlier keyframes (80th percentile)

### 6. Map initialization 的额外迭代

代码中首次 mapping 步骤使用 500 次迭代 (`map_init_iters: 500` in `train/default.yaml`)，之后每步仅 1 次迭代。论文未明确提及这一 warmup 策略。

### 7. 注释掉的 vision downweighting

`step_map()` (`trainer.py` L1862-1869) 中有一段被注释掉的代码，用于根据视觉点与触觉点的距离来降低视觉点的权重。这暗示作者尝试过更复杂的多模态权重方案，但最终未采用。

### 8. HORA 到 NeuralFeels 坐标系转换

代码中 `Allegro._hora_to_neural()` (`neuralfeels/modules/allegro.py` L54-68) 包含一个硬编码的 4x4 变换矩阵，用于将 DIGIT URDF 参考系 (传感器底部) 转换到 neural SLAM 坐标系。这个标定细节在论文中未提及。

### 9. 仿真中的深度噪声注入

代码中 `RealsenseSensor` 有 `sim_noise_iters` 参数 (`sensor/realsense.yaml` L23)，对仿真深度图施加 5 次噪声迭代来模拟真实 depth sensor 的行为。Real-world 数据则跳过此步骤 (`sensor.py` L592-594)。

---

## 跨论文比较

### 与 HORA 的关系

| 维度 | HORA | NeuralFeels |
|------|------|-------------|
| **目标** | 学习 in-hand rotation 策略 | 在 rotation 过程中估计物体 pose + shape |
| **贡献类型** | RL policy (控制) | Perception backbone (感知) |
| **关系** | 提供底层运动策略 | 消费 HORA 策略产生的交互数据 |
| **传感器** | Proprioception only | Vision + Touch + Proprioception |
| **物体假设** | 已知物体 (仿真训练) | 未知物体 (在线学习) |
| **手部** | Allegro (left/right) | Allegro (left) + 4x DIGIT |
| **频率** | 20Hz 控制 | 1-5Hz 感知 |
| **Sim-to-Real** | 直接迁移 (域随机化) | 不需要迁移 (在线学习) |

HORA 在 NeuralFeels 中的角色：HORA 的 proprioception-based rotation policy 在 20Hz 下运行，产生稳定的 in-hand rotation 运动。NeuralFeels 以 1-5Hz 的较低频率处理由此产生的多模态感知数据。两者形成"控制-感知"分离的管线。代码中通过 `allegro.py` 的 FK 计算和 `_hora_to_neural()` 坐标转换将两个系统连接起来。

### 与 HATO 的比较

| 维度 | HATO | NeuralFeels |
|------|------|-------------|
| **任务** | 5 种双手操控任务 | In-hand rotation (单手) |
| **触觉用途** | 策略学习的输入 (BC/Diffusion Policy) | SLAM 的观测 (深度重建) |
| **触觉传感器** | XELA (分布式压力) | DIGIT (视觉触觉) |
| **表征** | 触觉嵌入向量 | 3D SDF + SE(3) pose |
| **学习范式** | 端到端模仿学习 | 模块化在线优化 |
| **目标** | 输出控制动作 | 输出物体状态估计 |
| **手部** | 2x Psyonic Ability | 1x Allegro + DIGIT |

核心区别：HATO 将触觉作为 policy learning 的输入特征，NeuralFeels 将触觉作为 3D 重建的几何约束。前者是端到端的"触觉->动作"映射，后者是显式的"触觉->几何->状态"推理。

### 与 DexScrew 的比较

| 维度 | DexScrew | NeuralFeels |
|------|----------|-------------|
| **触觉传感器** | XHand (三轴力, 5x120) | DIGIT (视觉触觉, 4x 240x320) |
| **触觉用途** | BC 策略的输入特征 | Neural SDF 的训练信号 |
| **Sim-to-Real** | 简化仿真 RL + 真实数据 BC | Pre-trained frontend + 在线 backend |
| **物体模型** | 不需要 | 在线重建 |
| **核心贡献** | 不完美仿真也能诱导正确运动原语 | 触觉 disambiguate 视觉估计 |

共同点：两者都证明了触觉在 real-world 中的重要性，且都面临触觉 sim-to-real 的挑战。DexScrew 绕过问题 (只从真实数据学触觉)，NeuralFeels 正面解决 (在仿真中训练 tactile transformer)。

### 与 HOP 的比较

| 维度 | HOP | NeuralFeels |
|------|-----|-------------|
| **3D 表征** | Point cloud (PointNet) | Neural SDF (instant-NGP) |
| **表征目的** | Policy 的观测输入 | 物体状态估计的输出 |
| **在线/离线** | 离线预训练 + 在线微调 | 完全在线学习 |
| **传感模态** | 视觉 (point cloud) + 本体感知 | 视觉 + 触觉 + 本体感知 |
| **Sim-to-Real** | RL 微调不用域随机化 | 不涉及 policy 迁移 |
| **手部** | LEAP Hand | Allegro + DIGIT |

有趣的互补关系：HOP 需要物体的 point cloud 观测来做 goal-conditioned manipulation。NeuralFeels 的 neural SDF 可以在线生成高质量 point cloud。论文 Discussion 中提到了这种集成的可能性——将 NeuralFeels 作为 perception backbone 驱动 goal-conditioned planning (如 HOP 或 DexPBT)。

### 总结对比表

| 特征 | NeuralFeels | HORA | HATO | DexScrew | HOP |
|------|-------------|------|------|----------|-----|
| 感知 vs 控制 | **Perception** | Control | Control | Control | Control |
| 触觉角色 | 几何约束 | N/A | 策略输入 | 策略输入 | N/A |
| 物体模型 | 在线重建 (SDF) | 已知 | 不需要 | 不需要 | 点云观测 |
| 在线学习 | Yes | No | No | No | No |
| 模块化 | High | Medium | Low (E2E) | Medium | Medium |
| Sim-to-Real 方式 | Frontend预训练 | RL域随机化 | 直接Real | 分阶段 | RL微调 |
| 独特价值 | 填补灵巧手SLAM空白 | 纯本体感知旋转 | 低成本双手触觉 | 不完美仿真哲学 | 视频预训练先验 |
