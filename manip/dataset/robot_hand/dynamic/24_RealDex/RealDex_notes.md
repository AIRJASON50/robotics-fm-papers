# RealDex Notes

> RealDex: Towards Human-like Grasping for Robotic Dexterous Hand
> IJCAI 2024 | ShanghaiTech University, HKU, Texas A&M
> arXiv: 2402.13853

---

## 1. Core Problem

现有灵巧手抓取数据集存在三个核心缺陷:

1. **合成数据与真实世界的鸿沟**: 现有数据集如 DexGraspNet、MultiDex、DVGG 均为仿真优化生成的静态抓取姿态，依赖人工定义的能量函数，与真实机器人执行时的物理约束不匹配
2. **缺乏人类行为先验**: RL (Reinforcement Learning, 强化学习) 训练的抓取策略仅追求物理可行性，无法建模人类抓取习惯 -- 例如人会抓杯柄而非把拇指伸进杯内
3. **仅有静态 pose 无动态序列**: 以往数据集只包含最终抓取姿态，不包含从初始位置到接触物体的完整运动轨迹

RealDex 通过遥操作系统采集真实 ShadowHand 的 human-like 抓取运动序列，同时提供多视角多模态视觉数据，填补了这三个缺口。

---

## 2. Method Overview

整体框架分为**数据集**和**方法**两个部分:

### 数据集部分

- **硬件系统**: UR10e (6 DoF) + Shadow Right Hand (20 motors, 24 DoF) + 4x Azure Kinect RGB-D (15Hz) + 遥操作手套
- **采集流程**: 训练有素的操作员佩戴遥操作手套，实时控制 Shadow Hand 执行抓取
- **标定与同步**: Camera-camera (AprilTag + ICP)、Robot-camera (hand-eye calibration + ICP)、时间同步 (point cloud-mesh alignment)
- **数据规模**: 52 objects, 2.6K sequences, ~955K frames

### 方法部分 (Grasping Motion Generation Framework)

两阶段流水线:

| 阶段 | 模块 | 输入 | 输出 |
|------|------|------|------|
| Stage 1: Grasp Pose Generation | cVAE (conditional Variational Autoencoder, 条件变分自编码器) + MLLM (Multimodal Large Language Model, 多模态大语言模型) Selection | 物体点云 P^o | K 个 human-like 目标抓取姿态 {phi^k} |
| Stage 2: Motion Synthesis | MotionNet (autoregressive, 自回归) | 起始姿态 phi_0 + 目标姿态 phi_tgt | 完整运动序列 {phi_t} |

### 手部表示

- Joint angle: theta in R^22
- Global 6D pose: eta in R^6 (translation 3D + orientation angle-axis 3D)
- 总维度: phi = (theta, eta) in R^28

---

## 3. Key Designs

### 3.1 MLLM Selection Module -- 用 Gemini 做 human-like 过滤

**问题**: cVAE 生成的候选姿态满足物理约束 (contact map alignment)，但不考虑人类抓取偏好。

**解法**: 将每个候选抓取姿态渲染为 hand-object 交互图像，用 Gemini 从 naturalness、physical plausibility、human-likeness、preference 四个维度打分，选出最符合人类经验的目标姿态。

**关键洞察**: MLLM 内化了大量关于"人如何抓取物体"的世界知识，可以作为 human prior 的代理。这绕开了显式建模人类行为模式的困难。

**代码对应**: 论文中描述了 prompt 设计，但代码仓库中 **未包含 MLLM selection 的实现** -- 仓库仅提供 ContactMapNet + AffordanceCVAE 的训练代码。

### 3.2 Joint Self-Attention in MotionNet

**问题**: 不同于 MANO (通过 skinning 估计关节位置) 的参数化人手模型，机器人手各关节之间是固定铰链关系，关节间的空间依赖更强。

**解法**: 对关节坐标做 sinusoidal positional encoding 后，通过 self-attention 建模关节间的空间关系:

```
F^J_t = Attn(Q=J^PE_t, K=J^PE_t, V=J^PE_t)
```

**输入设计** (Eq. 2): 综合过去 5 帧关节特征 + pose、当前帧手部点云 + 速度、目标手部全局特征 + 当前到目标的位移，预测未来 10 帧的 pose 变化量。

**核心改进**: 通过 self-attention 显式建模 joint interdependence，而非像 GOAL 等方法那样将所有关节视为独立通道。

### 3.3 Contact-aware cVAE for Pose Generation

**架构** (代码分析):

| 模块 | 网络 | 输入 | 输出 |
|------|------|------|------|
| ContactMapNet | PointNet (object) + PointNet (hand) -> conv1d 融合 | 物体点云 + 手部点云 | 物体表面接触图 (连续 or 10-bin 离散化) |
| AffordanceCVAE | PointNet encoder -> VAE -> HandModel 重建 | 物体点云 + 手部 surface points | translation (3) + rotation (3) + qpos (22) |

**Loss 设计** (代码分析):
- qpos_loss + transl_loss + rotation_loss: MSE 监督抓取参数
- verts_loss: Chamfer Distance 监督手部表面点
- KLD: VAE 正则化
- cmap_loss: ContactMapNet 预测的接触图 vs 基于手-物距离的 pseudo contact map
- penetr_loss: 手-物穿透惩罚 (仅对穿透距离 > 0 的点求和)

---

## 4. Experiments

### 4.1 Grasping Motion Generation (Table 2)

- **Baseline**: SAGA、GOAL (从 human grasp generation 迁移到 robot hand)
- **数据集**: GRAB (human hand) + RealDex (robot hand)
- **指标**: 40 人 user study (rank-based scoring: rank 1=3pts, rank 2=2pts, rank 3=1pt)
- **结论**: RealDex 方法在两个数据集上均显著优于 baseline

### 4.2 Grasp Pose Generation (Table 3)

| 指标 | 含义 | 方向 |
|------|------|------|
| s.i.vol. (self-intersection volume) | 手部自穿透体积 | 越小越好 |
| p.dist. (penetration distance) | 手-物穿透距离 | 越小越好 |
| sim.disp. (simulation displacement) | 重力下物体位移 | 越小越好 (稳定性) |
| user score | 人工评分 | 越高越好 |

- Baseline: UniDexGrasp (conditional normalizing flows)
- 评估数据集: GRAB, DexGraspNet, RealDex
- Ablation: w/o MLLM 版本验证了 MLLM selection module 的有效性

### 4.3 Motion Synthesis (Table 4)

| 指标 | 含义 |
|------|------|
| MPJPE (Mean Per-Joint Positional Error) | 逐关节平均位置误差 |
| AVE (Average Variance Error) | 预测关节位置方差误差 |
| vertex offset | 最终姿态手部网格顶点偏移 |
| min distance | 手-物最小距离 |

- 输入 GT goal pose，测试纯运动合成质量
- 结论: Joint self-attention 带来的 spatial relationship 建模使运动合成优于 baseline

### 4.4 Real Robot Test

- 模型训练在 RealDex 上，直接部署到真实 Shadow Hand
- 生成的运动序列先在仿真器中安全测试，再编码为 stamped trajectory 发送到真实手
- 执行时间约 20s

---

## 5. Related Work Analysis

### 数据集对比 (Table 1)

| 对比维度 | DVGG | MultiDex | DexGraspNet | DDGdata | RealDex |
|---------|------|----------|-------------|---------|---------|
| 数据来源 | MuJoCo 仿真 | 优化合成 | 可微力闭合 | Planner | 真实遥操作 |
| 动态序列 | 否 | 否 | 否 | 否 | **是** |
| 人类行为 | 否 | 否 | 否 | 否 | **是** |
| 真实视觉 | 否 | 否 | 否 | 否 | **是** |
| 手类型 | 多种 | 5 种 | ShadowHand | 多种 | ShadowHand |

### 方法定位

- 区别于 RL 方法 (不需要 reward function design)
- 属于 supervised learning + generative model 范式
- 借鉴人手生成方法 (GOAL, SAGA) 但适配了 robot hand 的关节结构

---

## 6. Limitations & Future Directions

### 论文未明确讨论但可推断的局限

1. **规模有限**: 52 objects / 2.6K sequences 远小于合成数据集 (DexGraspNet 1.32M grasps, 5355 objects)。遥操作采集成本高，难以大规模扩展
2. **单一机器人手**: 仅 ShadowHand，无法直接泛化到 Allegro、LEAP、Inspire 等其他灵巧手
3. **MLLM 依赖**: Gemini 作为 selection module 引入不可控的 API 成本和延迟，且 MLLM 的评分可靠性缺乏定量分析
4. **无 MANO 表示**: 数据直接以 ShadowHand joint angle 存储，不提供 MANO 参数，限制了与 human hand 数据集的互操作性
5. **MotionNet 代码缺失**: 仓库仅提供 grasp pose generation 部分 (ContactMapNet + AffordanceCVAE)，MotionNet 和 MLLM selection 均未开源
6. **无接触力数据**: 尽管 ShadowHand 配备 100+ 传感器 (1kHz)，但数据集未包含任何接触力标注
7. **物体 6D pose 追踪方法未详述**: 论文提到有 ground truth object pose 但未详细说明追踪算法

### 可能的研究方向

- 将 human-like prior 引入 RL reward design (RealDex 作为 preference data)
- 多手型 retargeting (ShadowHand -> 其他灵巧手)
- 扩展到 in-hand manipulation 和 tool use 任务

---

## 7. Paper vs Code Discrepancies

| 维度 | 论文描述 | 代码实现 | 差异分析 |
|------|---------|---------|---------|
| **MotionNet** | 详细描述了 joint self-attention + autoregressive 架构 (Sec 4.3) | **仓库中未找到 MotionNet 代码** | 仅开源了 Stage 1 (pose generation)，Stage 2 (motion synthesis) 完全缺失 |
| **MLLM Selection** | 使用 Gemini 对渲染图像打分 (Sec 4.2) | **仓库中未找到 MLLM 相关代码** | 论文核心创新点之一未开源 |
| **ContactMapNet 加载** | 论文暗示 ContactMapNet 预训练后冻结 | 代码中 ContactMapNet 的 checkpoint 加载被**注释掉** (affordance_network.py L136-151) | 不清楚实际训练流程是否端到端 |
| **数据标准化** | 未在论文中提及 | 代码中 `pose_mean_std.pt` 归一化被**注释掉** (recon = recon * pose_std + pose_mean) | 推测是实验过程中发现不使用标准化效果更好 |
| **VAE inference** | 从学习的分布中采样 | 代码中 inference 使用 `z = torch.randn(...) * 0` 即**零向量** | 推理时直接用均值 decode，不做随机采样，严重限制了多样性 |
| **训练脚本** | 提供 train.sh | 仅包含 cm_net + cvae 两个 config 的训练 | 缺少 MotionNet 的训练配置 |
| **数据划分** | 论文: 40 obj train / 6 obj val / 6 obj test | 代码 main.py L489: 硬编码了 val/test 物体名 | 一致 |

---

## 8. Cross-Paper Comparison

### vs DexGraspNet

| 维度 | DexGraspNet (2022) | RealDex (2024) |
|------|-------------------|----------------|
| 数据类型 | 合成静态抓取姿态 | 真实动态抓取运动序列 |
| 规模 | 1.32M grasps, 5355 objects | 2.6K sequences, 52 objects |
| 手类型 | ShadowHand (仿真) | ShadowHand (真实) |
| 生成方法 | 可微力闭合优化 | 遥操作采集 |
| 人类行为 | 无 | 有 (遥操作隐式编码) |
| Sim2Real gap | 大 (纯仿真) | 小 (真实数据) |
| 视觉数据 | 无 | 4 视角 RGB-D |
| RealDex 直接使用 DexGraspNet 作为 eval benchmark | - | Table 3 对比 |

### vs DexCap

| 维度 | DexCap (2024) | RealDex (2024) |
|------|--------------|----------------|
| 数据采集 | 人手 MoCap + LEAP hand retarget | 遥操作手套 + ShadowHand |
| 手部表示 | 人手关节 -> LEAP joints | ShadowHand 22D joints |
| 任务 | Wiping + packaging (日常操作) | Grasping (抓取) |
| 视觉 | SLAM point cloud | 4x Azure Kinect RGB-D |
| 端到端部署 | 是 (采集->训练->部署 pipeline) | 是 (sim safety test -> real deployment) |
| 核心差异 | 强调可扩展性 (backpack 式采集系统) | 强调 human-like quality |
| MANO 支持 | 无 | 无 |
| 物体资产 | 无标准格式 | 3D 扫描 mesh (EinScan-Pro+) |

### vs EgoDex

| 维度 | EgoDex (2025) | RealDex (2024) |
|------|--------------|----------------|
| 规模 | 829h, 338K episodes, 194 tasks | ~数小时, 2.6K seq, 抓取任务 |
| 视角 | 第一人称 (egocentric) | 第三人称 (4 外部相机) |
| 手部表示 | ARKit skeleton | ShadowHand joints |
| 物体信息 | 无 (hand-only) | 有 (52 objects + 3D mesh + 6D pose) |
| 核心差异 | 大规模 egocentric dexterous manipulation | 小规模高精度 robot hand grasping |
| 数据质量 | 中 (ARKit 估计) | 高 (真实机器人关节 + 3D 扫描) |

### vs GigaHands

| 维度 | GigaHands (2025) | RealDex (2024) |
|------|-----------------|----------------|
| 规模 | 34h, 14K clips, 417 objects, 56 subjects | ~数小时, 2.6K seq, 52 objects |
| 手部格式 | MANO | ShadowHand joints (非 MANO) |
| 物体追踪 | Differentiable rendering (18.7% 成功率) | 3D 扫描 + 追踪 (高精度) |
| 双手 | 是 | 否 |
| 文本标注 | 84K 条 | 无 |
| 核心差异 | 大规模人手 MANO 数据 + 文本 | 唯一真实机器人手数据 + 高精度 GT |
| 对 RL 的直接价值 | 需要 retarget (MANO -> robot hand) | 直接可用 (ShadowHand joints) |

### 关键对比总结

| 特性 | DexGraspNet | DexCap | EgoDex | GigaHands | **RealDex** |
|------|-------------|--------|--------|-----------|-------------|
| 真实机器人数据 | 否 | 部分 (LEAP) | 否 | 否 | **是** |
| 动态序列 | 否 | 是 | 是 | 是 | **是** |
| 人类行为先验 | 否 | 隐式 | 隐式 | 隐式 | **显式 (遥操作)** |
| MANO 格式 | 否 | 否 | 否 | 是 | 否 |
| 物体 3D 资产 | 是 (5355) | 否 | 否 | 是 (417) | 是 (52) |
| 可直接部署 | 否 | 是 (LEAP) | 否 | 否 | **是** (ShadowHand) |
| 接触力 | 否 | 否 | 否 | 否 | 否 |

**RealDex 的独特定位**: 唯一提供真实机器人手 + 人类行为先验 + 多视角视觉 + 可直接部署的数据集。但规模是最大的短板。
