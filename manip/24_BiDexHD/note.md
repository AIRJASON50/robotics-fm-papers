# BiDexHD: Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations

## 论文概述

BiDexHD (Bimanual Dexterous from Human Demonstrations) 是北京大学提出的一个从人类演示中学习多样化双手灵巧操作技能的统一框架。核心思路是：将人类双手操作数据集（如 TACO）自动转化为仿真任务，然后用统一的两阶段奖励函数进行多任务强化学习（teacher），再蒸馏为基于视觉的策略（student）。

**核心贡献**:
1. 将人类双手操作数据集自动构建为 Dec-POMDP 仿真任务（IsaacGym），无需手动设计任务
2. 设计了通用的两阶段奖励函数，适用于所有双手工具使用任务
3. Teacher-Student 框架：IPPO 训练 state-based teacher，DAgger 蒸馏为 vision-based student
4. 在 TACO 数据集 141 个任务（6 类）上验证，训练任务完成率 74.59%，未见任务 51.07%

---

## 方法细节

### 问题建模

**任务形式化**: Dec-POMDP（去中心化部分可观测马尔可夫决策过程），双手被建模为 N=2 个独立 agent。

**手部模型**: LEAP Hand（4 指，16 DOF）+ 6-DOF RealMan 机械臂，每个 agent 输出 22 个关节角度（归一化到 [-1, 1]），位置控制。两臂间距 0.68m，桌面高度 0.7m。

**物体表示**: 工具 (tool) 和目标物体 (object) 各有 6D 位姿（位置 + 四元数）、线速度、角速度、唯一标识符 (one-hot ID)。

**观测空间**: 每侧 agent 的观测包含：
- 机器人本体感知：手臂-手部关节角度和速度 (j, v)
- 腕部位姿：(x, q)^{side, w}
- 指尖位置：x^{side, ft}
- 物体信息：(x, q, v, w, id)^{obj}

**动作空间**: 22 维关节角度，[-1, 1] 归一化，position control。

**数据来源**: TACO 数据集的人类双手操作轨迹，包含 MANO 参数（手部姿态 alpha in R^48、手部形状 beta in R^10、腕部位置 x^w in R^3），以及工具/物体的 6D 位姿序列。

**约定**: 默认右利手 -- 右手持工具 (tool)，左手持目标物体 (object)。

### 奖励函数设计

BiDexHD 采用统一的两阶段奖励函数，这是其核心技术贡献之一。

#### Stage 1: Simulation Alignment（仿真对齐）

目标：将仿真器状态从初始零位姿对齐到人类演示轨迹的第一帧参考位姿。包含三个奖励项：

**1. 接近奖励 r_appro（负距离惩罚）**

鼓励双手接近各自物体的抓取中心 (grasping center)。

```
r_appro^{side} = -||x_t^{side,w} - x_gc^{obj}||_2
                 - w_r * sum_m ||x_t^{side,ft} - x_gc^{obj}||_2
```

其中抓取中心的计算方式值得注意——不使用物体几何中心，而是基于人类演示的功能性抓取中心：
- 以演示参考时刻的腕部和指尖位置作为锚点
- 从物体表面 mesh 均匀采样 1024 个点
- 计算离锚点均值最近的 L=50 个点的平均位置作为抓取中心

```
x_gc^{obj} = (1/L) * sum NN(P, L, (x_0^{side,w} + sum_m x_0^{side,ft}) / (m+1))
```

这个设计对于薄片状、扁平或有手柄的物体尤为重要（消融实验证实）。

**2. 举升奖励 r_lift（条件触发）**

只有在手部足够接近抓取中心时才触发（门控机制）：

```
r_lift^{side} = {
  r_pos^{side} + w_q * r_quat^{side},  if ||x_t^{w} - x_gc|| <= lambda_w
                                         AND sum_m ||x_t^{ft} - x_gc|| <= lambda_ft
  0,                                     otherwise
}
```

位置奖励（非负线性）：
```
r_pos^{side} = max(1 - ||x_t^{obj} - x_hat_0^{obj}||_2 / ||x_0^{obj} - x_hat_0^{obj}||_2, 0)
```
即物体当前位置到目标位置的距离 / 初始位置到目标位置的距离，越接近目标越大，下限为 0。

姿态奖励（负四元数距离）：
```
r_quat^{side} = -D_quat(q_t^{obj}, q_hat_0^{obj})
```

**3. 成功奖励 r_bonus（阈值触发）**

当物体到达参考位姿附近时给予正奖励，信号阶段过渡：

```
r_bonus^{side} = {
  1 / (1 + ||x_t^{obj} - x_hat_0^{obj}||_2),  if ||x_t^{obj} - x_hat_0^{obj}||_2 <= eps_succ
  0,                                             otherwise
}
```

Stage 1 成功条件：左右两侧的 r_bonus 均为正，持续至少 u 步。

**Stage 1 总奖励（线性加权）**:
```
r_align^{side} = w1 * r_appro^{side} + w2 * r_lift^{side} + w3 * r_bonus^{side}
```

#### Stage 2: Trajectory Tracking（轨迹追踪）

在 Stage 1 成功后，双手需保持抓握并跟踪演示轨迹：

```
r_track^{side} = {
  exp(-w_t * ||x_{t_i}^{obj} - x_hat_i^{obj}||_2),  if stage 1 succeeds
  0,                                                    otherwise
}
```

使用指数型衰减奖励（与 Gaussian kernel 类似但不同——这里是 exp(-w * d) 而非 exp(-(d/sigma)^2)）。

轨迹追踪引入频率常数 f：每 f 个仿真步对应人类演示的 1 步，即 i = ceil(t_i / f)。

**总奖励**:
```
r_total^{side} = r_align^{side} + w4 * r_track^{side}
```

两阶段共存，r_track 在 Stage 1 未成功时自动为 0，无需额外切换逻辑。

#### 奖励设计特点总结

| 特点 | 描述 |
|------|------|
| 统一性 | 所有 141 个任务共享同一奖励函数，无需 per-task 设计 |
| 两阶段自动切换 | 通过条件判断 (gate) 实现，无需手动 curriculum |
| 功能性抓取中心 | 基于人类演示计算，而非几何中心 |
| 线性 + 指数混合 | Stage 1 用线性/负距离奖励，Stage 2 用指数奖励 |
| 加法组合 | 所有奖励项通过加权求和组合（非乘法） |
| 无接触奖励 | 没有显式的接触/碰触奖励项 |

### 训练策略

**RL 算法**: IPPO (Independent PPO)，左右手各自独立的 actor-critic，比集中式 PPO 效果好。原因：独立学习在更小的观测/动作空间中更高效，更易泛化到组合任务。

**网络结构**:
- State-based teacher: 5 层 MLP，隐藏层 [1024, 1024, 512, 512]，ELU 激活
- Vision-based student: 简化版 PointNet 处理点云（2 个 1D 卷积层 + max/avg pooling + 2 MLP 层，输出 128 维），actor 和 critic 共享 backbone

**策略蒸馏 (DAgger)**:
- 在线蒸馏，5% 概率使用 teacher 动作（混合采样加速早期训练）
- Student 观测用点云替代物体精确位姿（4096 点预采样，运行时子采样 + 高斯噪声）
- 移除 one-hot 物体标识以提升泛化
- 可选 K-step 未来位置条件输入（K=5 效果最佳，K=0 也可接受，差距 ~3%）

**参考时刻选择**: 不直接取第一帧，而是基于工具-物体距离的首次突变点（跳过抓取前的准备动作）。

**坐标对齐**: 人类腕部坐标对齐到机器人掌基坐标；z 轴平移偏移使所有物体初始高度统一。

**任务验证**: 通过 retargeting optimizer + IK 回放所有轨迹，剔除无效任务。

**计算资源**: 单 sub-task 的 state-based IPPO 训练约 2 天（单 A100 40G），每类 vision-based 策略蒸馏约 1 天。

### 关键设计决策

1. **功能性抓取中心 vs 几何中心**: 消融实验显示使用几何中心的策略无法正确抓握刷子手柄或平底锅，而功能性抓取中心引导了类人抓握行为。这是 Stage 1 成功率的关键。

2. **两阶段奖励 vs 纯追踪**: 移除 Stage 1 对齐奖励后，仅 30.5% 的简单任务能获得正的追踪率，其余完全失败。从静态零位姿直接学习动态技能不可行。

3. **r_bonus 的作用**: 移除后 r_2 下降，说明 bonus 有效地信号了阶段过渡，增强了策略对任务进展的感知。

4. **IPPO > PPO**: 独立策略比集中式策略在大规模多任务中更高效，更易泛化。

5. **去除 one-hot ID**: State-based teacher 使用 one-hot 物体标识可提升训练性能，但会降低泛化（新标签干扰决策）。Vision-based student 去掉 ID 后，对新物体的泛化大幅提升。

---

## 代码实现要点

代码仓库在笔记撰写时仍在克隆中（仅有 .git 目录），因此无法提取代码实现细节。根据论文描述，代码基于 UniDexGrasp++ 构建，使用 IsaacGym 作为仿真器。

已知的实现关键点（来自论文附录）：
- 基于 UniDexGrasp++ 代码库
- IsaacGym 并行仿真
- MANO 手部模型用于数据预处理（提取腕部/指尖位姿）
- Dex-retargeting 用于人手到机器人手的运动映射
- 评估阈值 eps_succ = eps_track = 0.1（约 10cm）

---

## 与 bh_motion_track 项目的关联

bh_motion_track 项目使用 WujiHand (5指20DOF) + Boob Cube 在 MuJoCo MJX 中进行双手操作任务，与 BiDexHD 有许多相似之处但也存在关键差异。

### 1. 可直接借鉴的技术

**功能性抓取中心计算**: BiDexHD 从人类演示中计算功能性抓取中心而非使用几何中心的做法值得借鉴。对于 Boob Cube 这种规则物体差异不大，但如果扩展到不规则物体，这个设计会很有价值。

**两阶段奖励框架**: 将任务分为"仿真对齐"（抓取到位）和"轨迹追踪"两个阶段，通过条件门控自动切换，这个整体框架可以参考。bh_motion_track 的 weld-based contact guidance curriculum 本质上也是类似的分阶段思想。

**IPPO 独立策略**: 左右手独立训练比集中式训练更高效的发现，对双手任务有普遍意义。bh_motion_track 如果遇到双手协调困难，可以考虑独立策略。

**移除物体标识以提升泛化**: 在蒸馏/迁移阶段去掉 task-specific 的标识信息。

### 2. 奖励设计上的差异和改进空间

| 方面 | BiDexHD | bh_motion_track | 分析 |
|------|---------|-----------------|------|
| **核心奖励形式** | 负距离惩罚 + 线性 + exp(-w*d) | Gaussian kernel exp(-(e/sigma)^2) | Gaussian kernel 在零误差附近更陡峭，提供更精细的位姿追踪信号 |
| **指尖追踪** | 指尖到抓取中心的距离（全局坐标） | tips-in-object-frame（物体坐标系下追踪） | 物体坐标系下追踪更鲁棒，不受物体全局位姿变化影响，bh_motion_track 的设计更优 |
| **接触奖励** | 无显式接触奖励 | 3-term contact reward (touch + match - FP) | bh_motion_track 的接触奖励更精细，能引导正确的指尖-物体接触模式 |
| **奖励组合方式** | 加法加权求和 | 乘法组合物体奖励 | 乘法组合要求各项同时满足，避免了单项奖励 hack；加法允许部分奖励驱动优化，训练更稳定但可能出现奖励 exploit |
| **课程学习** | 两阶段条件门控（硬切换） | weld-based contact guidance curriculum | weld-based 提供更平滑的过渡，BiDexHD 的硬门控更简单但可能导致阶段间不连续 |
| **物体姿态奖励** | 四元数距离（仅 Stage 1） | 乘法组合的物体位置+姿态 | BiDexHD Stage 2 只追踪位置不追踪姿态，这对精细操作可能不够 |

**改进空间**:

- BiDexHD 的 Stage 2 只追踪物体位置，不追踪姿态，对于需要精确旋转控制的任务（如 Boob Cube）这是不够的。bh_motion_track 同时追踪位置和姿态更合理。
- BiDexHD 没有接触奖励，纯粹依赖距离引导。bh_motion_track 的 3-term contact reward 提供了更直接的抓握质量信号，在灵巧操作中非常有价值。
- BiDexHD 的 exp(-w*d) 追踪奖励在大误差时衰减过快趋近于 0，可能导致梯度消失。Gaussian kernel exp(-(d/sigma)^2) 在中等误差范围内有更好的梯度特性。
- bh_motion_track 的 tips-in-object-frame 设计在物体运动时提供了更稳定的追踪信号。

### 3. 训练策略上的参考

**Reference State Initialization (RSI)**: BiDexHD 没有使用 RSI，而是从固定零位姿开始训练（Stage 1 对齐到参考位姿）。bh_motion_track 如果使用了 RSI，则在训练效率上有优势。BiDexHD 的两阶段设计实际上是在奖励层面解决了 RSI 的问题——先学抓取再学操作。

**Early Termination**: 论文未提及显式的 early termination 条件。BiDexHD 使用 u-step 持续成功作为阶段过渡条件，而非 early termination。

**Domain Randomization**: 论文未提及 domain randomization。多样化物体本身提供了一定的泛化训练效果。bh_motion_track 如需部署到实物，应保留 domain randomization。

**多阶段训练**: BiDexHD 的关键洞察是将复杂的双手操作拆分为抓取对齐和轨迹追踪两个阶段，但两个阶段在同一个奖励函数中同时存在，通过条件门控实现自动切换。这比手动切换训练阶段更优雅。bh_motion_track 的 weld-based curriculum 是另一种实现方式——通过逐步释放约束来引导学习。

**数据规模与泛化**: BiDexHD 在 141 个任务上训练，展示了约 51% 的零样本泛化能力。这提示大规模多任务训练对泛化的重要性。
