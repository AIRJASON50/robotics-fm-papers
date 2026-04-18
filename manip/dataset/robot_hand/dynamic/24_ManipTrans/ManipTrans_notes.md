# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

**Paper**: CVPR 2025  
**Authors**: Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, Siyuan Huang (BIGAI)  
**Code**: [github.com/ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans)  
**Dataset**: [huggingface.co/datasets/LiKailin/DexManipNet](https://huggingface.co/datasets/LiKailin/DexManipNet)

---

## 1. Core Problem

核心问题: 如何高效地将人类双手操作技能迁移到仿真中的灵巧机器人手上?

现有方法的三个瓶颈:

| 方法类别 | 限制 |
|----------|------|
| RL (Reinforcement Learning, 强化学习) 从零探索 | 需要手工设计 task-specific reward, 不可扩展; 高维动作空间探索低效 |
| 遥操作 (Teleoperation) | 人力密集、成本高; 缺乏触觉反馈导致动作生硬; 只能获得特定 embodiment 的数据 |
| 直接 Retargeting | 人手与机器人手的形态学差异导致 pose 映射不精确; MoCap 误差在物理仿真中放大 |

双手操作 (Bimanual Manipulation) 引入了额外难度: 两只手的协调需要在已经很高维的动作空间中进一步同步, 大多数先前工作停留在单手抓取/提升任务, 复杂的双手任务 (如旋盖、套笔帽) 基本未被探索。

---

## 2. Method Overview

ManipTrans 的核心洞察: 将 human-to-robot 迁移解耦为两个阶段 -- (1) 先学好手指动作模仿, (2) 再通过残差学习适配物体交互的物理约束。

### Stage 1: Hand Trajectory Imitating (手部轨迹模仿)

- 目标: 训练一个 generalist imitator model $\mathcal{I}$, 仅关注 hand motion (无物体)
- 输入 state: 目标人手轨迹 $\tau_h^t$ (wrist 6-DoF pose + 速度 + 手指关节位置/速度) + 机器人手本体感知 $s_{\text{prop}}$ (关节角、wrist pose 及其速度)
- 输出 action: 关节 PD 控制目标位置 $a_q \in \mathbb{R}^K$ + wrist 6-DoF 力 $a_w \in \mathbb{R}^6$
- 训练数据: 大规模 hand-only 数据集 (OakInk-V2, FAVOR, GRAB 等 + 插值生成的合成数据), 左右手镜像增强
- 训练策略: RSI (Reference State Initialization, 参考状态初始化) + Early Termination + Curriculum Learning (逐步收紧 $\epsilon_{\text{finger}}$ 阈值, 从 6cm 降到 4cm)
- Reward 由三部分组成: $r_{\text{wrist}}$ (wrist tracking) + $r_{\text{finger}}$ (finger keypoint 匹配, 带加权) + $r_{\text{smooth}}$ (平滑性惩罚)

### Stage 2: Residual Learning for Interaction (交互残差学习)

- 冻结预训练的 $\pi_{\mathcal{I}}$, 新增残差模块 $\mathcal{R}$ 学习修正量 $\Delta a$
- 最终动作: $a^t = a_{\mathcal{I}}^t + \Delta a_{\mathcal{R}}^t$, element-wise 相加后 clip 到关节限位
- 扩展的 state space: 在 $s_{\mathcal{I}}$ 基础上增加物体相关信息
  - 物体 pose/速度/质心/重力向量
  - BPS (Basis Point Set) 编码物体形状 (128 个点)
  - 手-物距离 $D(j_d, p_{\hat{o}})$
  - 指尖接触力 $C$ (从仿真器获取)
- 新增 Reward: $r_{\text{object}}$ (物体轨迹跟踪) + $r_{\text{contact}}$ (在 MoCap 指示接触时鼓励实际接触力)
- 训练策略: 物理约束渐进式放松
  - 初始重力设为 0, 摩擦系数设为高值 (init_value=3, 逐渐衰减到 1)
  - Curriculum: 物体偏差阈值 $\epsilon_{\text{object}}$ 从 90deg/6cm 收紧到 30deg/2cm
  - 接触终止条件: MoCap 显示紧握但仿真中无接触则终止

### MDP (Markov Decision Process, 马尔可夫决策过程) 设定

- 优化算法: PPO (Proximal Policy Optimization, 近端策略优化)
- Horizon: 32 frames, mini-batch 1024, $\gamma=0.99$
- 仿真: Isaac Gym, 4096 并行环境, 1/60s 时间步, RTX 4090 + i9-13900KF

---

## 3. Key Designs

### 3.1 两阶段解耦: Hand Imitation + Residual Refinement

这是本文最核心的设计。解耦的三重优势:

1. **Stage 1 不需要操作数据**: 手部模仿模型可以用大量 hand-only 数据训练, 无需稀缺的手-物交互数据
2. **降低动作空间复杂度**: 先解决形态学差异 (手指映射), 再解决物理交互约束, 避免同时学习两者
3. **Residual 初始化接近零**: 因为 Stage 1 已经提供了合理的粗动作, residual module 只需学习小的修正量, 避免 model collapse、加速收敛

从代码实现看, residual network 的 forward pass 流程为:
```
base_obs -> frozen imitator -> base_action (sample from Gaussian)
[full_obs, base_action] -> residual MLP -> delta_action
final_action = base_action + delta_action
```

值得注意: 代码中 base model 始终处于 `eval()` 模式且参数被 `recurse_freeze` 冻结, 只训练 residual 部分。

### 3.2 物理约束渐进放松策略 (Quasi-Physical Relaxation)

不同于 QuasiSim 使用自定义仿真器, ManipTrans 直接在 Isaac Gym 中调节物理参数:

| 参数 | 初始值 | 最终值 | Schedule |
|------|--------|--------|----------|
| Gravity ($\mathcal{G}$) | 0 | -9.81 | linear_decay over 1920 steps |
| Object friction ($\mathcal{F}$) | 3x (init_scale) | 1x | linear_decay over 1920 steps |
| Early termination threshold | 松 (scale_factor 大) | 紧 (scale_factor=0.7) | exp_decay over 3200 steps |

此外, 代码中手指的摩擦系数设为 2.0 (硬编码), 注释说明这是为了补偿仿真中缺失的皮肤变形摩擦。

### 3.3 触觉信息的三重利用

触觉信息 $C$ (contact force) 在 residual 阶段的三种使用方式:

1. **作为 observation**: 指尖接触力直接作为 policy 输入, 提供实时触觉反馈
2. **作为 reward component**: $r_{\text{contact}}$ 鼓励在 MoCap 指示接触区域内产生接触力
3. **作为 early termination 条件**: 当 MoCap 数据显示手紧握物体但仿真中无接触时, 触发终止

代码中的接触力判断: 维护一个 `tips_contact_history` 滑窗, 检查历史接触状态。若指尖距离物体 < 0.5cm 但接触历史全为零, 则终止 episode。

---

## 4. Experiments

### 4.1 数据集与指标

- **评估数据集**: OakInk-V2 验证集, 约 80 episodes (一半是双手任务), 4-20秒, 60fps
- **定性评估**: 额外使用 GRAB, FAVOR, ARCTIC

**指标体系** (比 QuasiSim 更严格):

| 指标 | 含义 | 成功阈值 |
|------|------|----------|
| $E_r$ | 每帧平均物体旋转误差 (deg) | < 30deg |
| $E_t$ | 每帧平均物体平移误差 (cm) | < 3cm |
| $E_j$ | 平均关节位置误差 (cm) | < 8cm |
| $E_{ft}$ | 平均指尖位置误差 (cm) | < 6cm |
| $SR$ | 成功率 (所有指标同时满足) | 双手任务更严格: 任一手失败即失败 |

### 4.2 主要对比结果

**vs RL-Combined Methods** (Table 1):
- Retarget-Only: 基本不可用 (动作空间复杂 + 误差累积)
- RL-Only (从零探索): 探索耗时长, 精度低
- Retarget + Residual: 比 ManipTrans 差, 因为 retargeting 经常在接触密集场景产生碰撞, 导致 residual 训练不稳定
- **ManipTrans**: 全面最优

**vs QuasiSim** (定性对比):
- 60 帧的 "rotating a mouse" 任务: ManipTrans 约 15 分钟, QuasiSim 约 40 小时
- ManipTrans 接触更稳定、运动更平滑

### 4.3 Cross-Embodiment 验证

在 4 种灵巧手上测试, 无需修改网络超参或 reward 权重:

| 灵巧手 | DoF (自由度) | 特点 |
|---------|------|------|
| Shadow Hand | 22 | 高 DoF, 业界标准 |
| Articulated MANO | 22 | 参数化人手模型 |
| Inspire Hand | 12 | 高性价比, 论文主平台 |
| Allegro Hand | 16 | 四指, 缺少小指 |

### 4.4 Real-World Deployment

- 硬件: 2x Realman 7-DoF 臂 + 2x Inspire Hand (升级版, 带触觉传感器)
- 12-DoF 仿真到 6-DoF 实物的映射: 通过 fitting-based 方法最小化指尖位置误差 + 时序平滑
- 手臂 wrist 跟踪: IK 求解
- 实现了开牙膏盖等精细双手操作

### 4.5 Ablation Studies

| 消融项 | 影响 |
|--------|------|
| w/o contact force as observation | 收敛速度下降 |
| w/o contact force as reward | 成功率下降 |
| w/o contact termination | 初期看似更好, 但最终收敛速度慢 |
| w/o relax-gravity | 收敛速度下降, SR 降低 |
| w/o increased friction | 同上 |
| w/o relax-thresholds | 网络可能完全不收敛 |

### 4.6 DexManipNet 数据集

| 属性 | 值 |
|------|------|
| 来源数据集 | OakInk-V2, FAVOR |
| 任务数 | 61 |
| Episodes | 3.3K |
| 总帧数 | 1.34M |
| 双手序列 | ~600 |
| 物体数 | 1.2K+ |
| 主平台 | Inspire Hand (12-DoF) |

Imitation learning benchmark: IBC, BET, Diffusion Policy (UNet/Transformer) 在 bottle rearrangement 上均表现欠佳, 说明灵巧手操作任务的 action space 复杂度对当前 IL 方法仍是重大挑战。

---

## 5. Related Work Analysis

### 与本文方法的关系

| 方向 | 代表工作 | 与 ManipTrans 的差异 |
|------|----------|---------------------|
| RL 探索 | DexPBT, ArtiGrasp | 需要 task-specific reward, 不可扩展 |
| 遥操作 | HATO, AnyTeleop | 人力密集、只适用特定 embodiment |
| 运动重定向 | dex-retargeting | 纯几何映射, 不保证物理可行性 |
| Residual RL | DexH2R, GraspGF | 各有特定 base policy 设计; ManipTrans 的 base 更通用 (大规模预训练 imitator) |
| 优化方法 | QuasiSim | 自定义仿真器, 极其耗时 (~40h vs ~15min) |
| 双手操作 | BiDexHD | 需要预定义操作阶段, ManipTrans 无需 |

### 关键引用链

- **MANO** -> 提供人手参数化模型, 作为人手轨迹的标准表示
- **OakInk-V2, FAVOR** -> 提供 MoCap 数据, 是 DexManipNet 的上游数据源
- **Isaac Gym** -> 仿真平台, 提供 4096 并行环境 + 物理参数调节
- **BPS (Basis Point Set)** -> 物体形状编码, 用于 residual module 的 state space
- **Transic** -> 训练 pipeline 的基础 (代码致谢)

---

## 6. Limitations & Future Directions

### 论文自述的限制

1. **MoCap 数据噪声**: 交互 pose 噪声过大时迁移失败
2. **物体模型精度不足**: 特别是铰接物体, 仿真用的 convex hull 近似可能导致物理交互不准确

### 笔记补充的限制

| 限制 | 分析 |
|------|------|
| 每个 episode 独立训练 residual | 不是 generalist residual policy, 每条新轨迹都需要 ~15min 重新训练 |
| 无视觉输入 | 完全依赖 privileged state (物体 pose/力), 无法直接用于 real-world closed-loop 控制 |
| 仿真到真实的 gap | 12-DoF 仿真 -> 6-DoF 实物需要额外 fitting, 且是 open-loop replay, 非真正的闭环策略 |
| 物体凸包近似 | COACD 生成的凸分解是对真实物体的简化, 影响接触精度 |
| 数据规模 | 3.3K episodes 相比遥操作数据量大, 但对 FM 训练可能仍不足 |

### 未来方向

1. **提升鲁棒性**: 对抗 MoCap 噪声的去噪/过滤机制
2. **物体模型生成**: 自动从视觉/扫描生成高精度仿真用物体模型
3. **Generalist residual policy**: 跨任务/跨物体的通用残差策略, 避免每条轨迹独立训练
4. **Vision-based policy**: 利用 DexManipNet 训练视觉驱动的操作策略
5. **与 FM 结合**: 作为数据引擎为 Robotics Foundation Model 提供大规模灵巧手操作数据

---

## 7. Paper vs Code Discrepancies

| 项目 | 论文描述 | 代码实现 |
|------|----------|----------|
| 手指摩擦系数 | 论文未详述具体值 | 代码硬编码为 2.0, 注释说明补偿仿真中缺失的皮肤变形摩擦 (dexhandmanip_sh.py:597) |
| Object friction 初始值 | 论文说 "high value" | 代码中 init_value=3, upper_bound=6.0, lower_bound=1.0 (ResDexHand.yaml:133-134) |
| Gravity relaxation schedule | 论文说 "gradually restore" | 代码中 linear_decay over 1920 steps, init_value=0 (ResDexHand.yaml:118-122) |
| BPS 编码 | 论文说 BPS representation | 代码中使用 grid_sphere, n_bps_points=128, radius=0.2 (dexhandmanip_sh.py:181-183) |
| Residual 初始化 | 论文说 "zero-mean Gaussian + warm-up" | 代码中 MLP 权重用标准初始化, mu 层用专门的 mu_init; 未见显式 warm-up schedule |
| Reward 权重 | 论文说 "parameters in Appx" | 代码中硬编码: obj_pos 权重 5.0 (最高), thumb_tip 0.9, index 0.8, middle 0.75, ring/pinky 0.6 (dexhandmanip_sh.py:1544-1563) |
| Finger imitation weights | 论文 Eq.1 用 $w_f, \lambda_f$ | 代码中用 exp(-100*thumb, -90*index, -80*middle, -60*ring/pinky) 的衰减系数 (dexhandmanip_sh.py:1464-1468) |
| PID 控制模式 | 论文主方法用 6D-Force wrist control | 代码支持两种模式: 6D-Force (主方法) 和 PID-controlled (base.py:25-31, 注释说明更稳定但需 careful tuning) |
| Contact termination | 论文说 "if MoCap indicates firm grasp but no contact" | 代码中检查 tips_distance < 0.005 且 tip_contact_history 全为零 (dexhandmanip_sh.py:1539) |
| Early termination thresholds | 论文说用 scale_factor | 代码中指尖阈值除以 0.7, 物体阈值除以 0.343 并用 scale_factor^3 (dexhandmanip_sh.py:1530-1538), 非线性关系论文未提及 |

### 代码中未在论文中提及的实现细节

1. **soft_clamp 函数**: 使用 sigmoid 做软裁剪而非硬 clip, 用于 action 输出 (dexhandimitator.py:43-44)
2. **action moving average**: `actionsMovingAverage=0.4` 参数, 对动作做时序平滑, 论文未提及
3. **DexHand 抽象类**: 提供了清晰的接口设计 (hand2dex_mapping, contact_body_names, weight_idx), 使得添加新灵巧手只需实现一个配置类
4. **rl_games 框架**: 底层 RL 训练基于 rl_games 库的 A2CBuilder, 论文只说 PPO

---

## 8. Cross-Paper Comparison

### ManipTrans vs BiDexHD vs DexLatent vs UniDex

| 维度 | ManipTrans | BiDexHD | DexLatent | UniDex |
|------|------------|---------|-----------|--------|
| **论文** | CVPR 2025 | RSS 2024 | -- | -- |
| **核心目标** | 人手 MoCap -> 机器人手迁移 (数据生成) | 人手视频 -> 双手灵巧操作策略 | 跨 embodiment 灵巧抓取 | 通用灵巧手操作 |
| **输入** | MANO 参数化人手轨迹 + 物体 mesh | RGB-D 人手视频 | 各种 embodiment 的抓取 pose | Task specification |
| **输出** | 机器人手仿真中的操作轨迹 | 双手 Shadow Hand 动作 | 跨手型的抓取策略 | 通用操作策略 |
| **是否 task-specific reward** | 否, 通用 reward | 是, 分阶段 reward | -- | -- |
| **是否需要预定义操作阶段** | 否 | 是 (approach-grasp-manipulate) | -- | -- |
| **Cross-embodiment** | 支持 (Shadow, Inspire, Allegro, ArtiMANO) | 仅 Shadow Hand | 支持多种手型 | 支持多种手型 |
| **双手支持** | 支持, 核心贡献 | 支持, 核心贡献 | 主要单手 | 主要单手 |
| **训练效率** | ~15min/episode (RTX 4090) | 较长 (复杂分阶段训练) | -- | -- |
| **数据集产物** | DexManipNet (3.3K eps, 61 tasks) | 无大规模数据集 | -- | -- |
| **Real-world** | Open-loop replay (非闭环) | 有 sim2real 尝试 | -- | -- |
| **仿真平台** | Isaac Gym | Isaac Gym | -- | -- |

### 方法论对比的关键差异

| 方面 | ManipTrans | 其他方法 |
|------|------------|----------|
| **Embodiment gap 处理** | 两阶段解耦: 先模仿手动作 (跨 embodiment), 再残差适配物理交互 | BiDexHD: retargeting + 分阶段 RL; DexLatent: latent space 对齐 |
| **数据需求** | Stage 1 只需 hand-only 数据 (大量可得), Stage 2 需 hand-object 交互数据 | 多数方法全程需要 hand-object 交互数据 |
| **通用性实现路径** | 通用 imitator + per-episode residual (非 generalist policy) | BiDexHD: per-task policy; UniDex: universal policy |
| **物理约束处理** | 渐进式放松 (gravity, friction curriculum) | QuasiSim: 自定义准物理仿真; BiDexHD: 标准物理 |

### 对 Robotics FM 的启示

1. **Data Engine 路线**: ManipTrans 本质上是一个 data engine -- 将人类 MoCap 数据转化为机器人可用的操作数据。这与 Robotics FM 的数据瓶颈直接相关: 高质量灵巧手操作数据稀缺, ManipTrans 提供了一种可扩展的生成路径。
2. **两阶段解耦的启示**: "先学通用表示, 再 task-specific 微调" 的范式与 FM 的 pre-train + fine-tune 一脉相承。区别在于 ManipTrans 的 Stage 1 是 RL pre-training (不是 self-supervised), Stage 2 是 per-episode residual (不是 few-shot fine-tune)。
3. **Generalist policy 的缺失**: ManipTrans 没有训练跨任务/跨物体的 generalist policy, 每条新轨迹需要独立训练 residual。这是未来需要解决的关键问题 -- 如何将 per-episode residual 升级为 generalist residual policy, 使系统能够泛化到新任务。
4. **与 SONIC/DreamGen 的互补性**: SONIC 解决 humanoid motion tracking, ManipTrans 解决 dexterous manipulation transfer, 两者可以互补: humanoid 全身 + 灵巧手操作 = 完整的 embodied agent 数据生成 pipeline。
