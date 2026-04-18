# DexMachina: Functional Grasp Synthesis -- 研究笔记

Zhao Mandi et al. (Stanford / NVIDIA), arXiv 2505.xxxxx
代码: Genesis-based RL, 6 种灵巧手, 5 种铰接物体

---

## 1. Core Problem

**Functional Retargeting (功能性重定向)**: 给定人手-物体交互的 tracked 演示, 学习灵巧手 RL 策略使物体沿演示轨迹运动 (位置、旋转、铰接角度)。与 kinematic retargeting (运动学重定向, 仅匹配手部姿态但不保证操作可行性) 不同, functional retargeting 以物体状态跟踪为核心目标。

聚焦场景: **long-horizon bimanual articulated object manipulation** -- 双手操作铰接物体的长时序任务 (最长 300 帧)。

核心挑战:

| 挑战 | 说明 |
|------|------|
| 高维动作空间 | 双手各 6-DOF 腕部 + N-DOF 手指, 探索困难 |
| 时空接触不连续性 | 操作铰接物体需精确的接触序列转换 (如单手支撑切换到双手打开盖子) |
| Embodiment Gap (形态差距) | 人手与机器人手运动学/动力学差异大, 直接 retarget 后的运动无法操作物体 |
| 早期灾难性失败 | 长序列中一次失误 (如物体掉落) 导致后续无法恢复 |

---

## 2. Method Overview

DexMachina 是一种 **pure RL** 方法 (无 BC pretraining, 无 teacher-student), 核心创新是 Virtual Object Controller (VOC, 虚拟物体控制器) curriculum。

**整体流程**:

```
ARCTIC 数据集 (MANO + 物体轨迹)
        |
        v
  [数据预处理]
  1. AnyTeleop 运动学重定向 -> 机器人关节值 Q
  2. Object-aware 碰撞后处理 -> 消除穿透的 Q' 和 keypoint X
  3. 距离近似接触 -> contact C, mask M
        |
        v
  [RL 训练 (PPO, rl-games)]
  观测: state-based (手关节 + 物体状态 + 接触力)
  动作: Hybrid (腕部残差 + 手指绝对)
  奖励: r_task * r_imi + r_bc + r_con
  课程: VOC gains 从高到零的自动衰减
        |
        v
  策略输出: 双手关节目标 -> PD 控制器
```

**关键设计决策一览**:

| 方面 | 选择 | 理由 |
|------|------|------|
| BC pretraining | 无 | 纯 RL from scratch |
| Teacher-student | 无 | 单策略直出 |
| 腕部控制 | Residual (紧约束: +/-4cm, +/-0.5rad) | retarget 质量好, 约束搜索空间 |
| 手指控制 | Absolute (关节限位归一化) | retarget 质量差 (穿透严重), 关节限位本身已约束空间 |
| 探索辅助 | VOC curriculum | 物体沿 demo 轨迹运动, 策略逐步接管 |
| 仿真器 | Genesis (非 IsaacGym) | 更稳定的接触建模, 支持 12,000 并行环境 |
| RL 算法 | PPO (rl-games) | 标准选择 |

---

## 3. Key Designs

### 3.1 Virtual Object Controller (VOC) Curriculum -- 核心贡献

**问题**: 长时序铰接物体任务中, 朴素 RL 面临灾难性早期失败。例如双手抬起盒子后, 策略无法预判一只手需要重新定位以打开盖子, 导致物体掉落并终止 episode。

**方案**: 给物体添加虚拟 PD actuator (6-DOF root + 1-DOF 铰接关节), 控制目标 = demo 轨迹的下一时间步状态。

| 参数 | 初始值 | 最终值 |
|------|--------|--------|
| kp | 1000.0 | 0 |
| kv | 10.0 | 0 |
| force_range | 50.0 | 0 |

**工作机制**:
1. 训练初期: 高增益使物体自行跟随 demo 轨迹, 策略无需担心物体掉落, 专注学习手部运动和接触模式
2. 训练中期: 增益指数衰减, 策略逐步从"搭便车"过渡到"主动控制"
3. 训练末期: 增益归零, 策略完全通过手-物体接触控制物体

**衰减条件** (代码 `curriculum.py` 中的 `determine_decay`):
- 超过 `wait_epochs` (默认 2000)
- task reward 均值超过阈值 (默认 0.5)
- 奖励梯度稳定 (|grad| < 0.0001)
- episode 长度接近最大值 (策略能完成整个序列)
- 距上次衰减至少 40 epochs

**Dialback 机制**: 若衰减后 episode 长度显著下降, 自动回退到之前的增益值 (乘以 0.98 的 ratio 而非完全恢复, 避免震荡)。

**与 ManipTrans curriculum 的核心区别**:

| | DexMachina VOC | ManipTrans |
|---|---|---|
| 衰减对象 | 物体上的虚拟控制器增益 | 物理参数 (重力, 摩擦力, 误差阈值) |
| 效果 | 物体被"引导"沿轨迹运动 | 物理约束放松, 降低任务难度 |
| 长时序表现 | 稳定, 逐步接管 | 初期高 reward, 后期崩溃无法恢复 |
| 直觉 | "先帮你把物体放好, 你学怎么接手" | "先让物理规则变简单" |

ManipTrans 的问题在于: 衰减物理参数改变了任务本身的动力学, 策略在简化物理下学到的控制策略可能在真实物理下不可行。VOC 则保持物理规则不变, 仅提供物体轨迹上的额外"托举力"。

### 3.2 Hybrid Action Space (混合动作空间)

**腕部 (6-DOF): 残差控制**
```
wrist_trans_target = retarget_qpos[t, :3] + 0.04 * a_wrist[:3]     # +/-4cm
wrist_rot_target  = retarget_qpos[t, 3:6] + 0.5  * a_wrist[3:6]   # +/-0.5rad
```

**手指 (remaining DOFs): 绝对控制**
```
finger_target = lower_limit + (upper_limit - lower_limit) * (a_finger + 1) / 2
```

**为什么不用统一的残差或绝对?** 消融实验 (Figure 8) 验证:

| Action Mode | 腕部 | 手指 | 结果 |
|-------------|------|------|------|
| Absolute | 绝对 | 绝对 | 最差 (腕部搜索空间太大) |
| Residual (loose) | 残差 (宽范围) | 残差 | 中等 |
| **Hybrid (tight)** | **残差 (紧约束)** | **绝对** | **最优** |

核心洞察: **腕部的紧约束比手指是否用参考更重要**。这是因为:
- 腕部 retarget 质量高 (大尺度运动容易匹配), 残差修正即可
- 手指 retarget 质量差 (指尖频繁穿透物体, 形态差异导致不可行姿态), 给绝对自由度反而更好
- 手指关节限位本身已约束了搜索空间

### 3.3 Object-Aware Retarget Post-processing (物体感知的重定向后处理)

**问题**: AnyTeleop 纯运动学 retarget 只匹配指尖位置, 忽略物体碰撞, 导致机器人手指频繁穿透物体 mesh。这破坏了:
1. 腕部残差控制的 base action (基于穿透的参考无意义)
2. imitation reward 的 keypoint 目标 (不可行的参考位置)

**解决方案**: 基于物理的碰撞解析

对每个 demo 时间步 t:
1. 固定物体在 demo 状态 (pose + 铰接角度) -- 物体不可移动
2. 设置机器人手关节目标为运动学 retarget 值
3. 运行一步物理仿真 -- PD 控制器驱动手向目标运动, 碰撞检测阻止穿透
4. 记录达到的关节值和 keypoint 位置

代码实现 (`parallel_retarget.py`): 该过程可在仿真中对所有时间步并行处理, 效率高。

效果显著: 消除穿透后的 retarget 结果使 ObjDex reimplementation 在 Ketchup-100 上从原始论文的 41.2% 提升至 >90% 成功率。

---

## 4. Experiments

### 4.1 实验设置

| 项目 | 配置 |
|------|------|
| 数据集 | ARCTIC (5 种铰接物体: box, notebook, mixer, waffleiron, ketchup) |
| 演示 | 7 个片段, 短时序 (100 帧) 和长时序 (170-300 帧) |
| 灵巧手 | 6 种: Inspire, Allegro, XHand, Schunk, Ability, DexRobot |
| 仿真器 | Genesis, 12,000 并行环境 |
| RL | PPO (rl-games), L40s 或 H100 GPU |
| 评估 | 5 seeds x 20 episodes, ADD-AUC 指标 |

### 4.2 评估指标: ADD-AUC

借鉴 6D pose estimation 领域 (FoundationPose, PoseCNN 等) 的 ADD (Average Distance of model points, 模型点平均距离) 指标:
- 对每个物体部件分别计算 ADD (适应铰接物体)
- 在计算 AUC (Area Under Curve, 曲线下面积) 前平均各部件 ADD
- 优势: 比阈值成功率更稳健 (不依赖 3 个阈值的选择), 比报告 3 个跟踪误差更简洁

### 4.3 主要结果

**vs 基线**:
- DexMachina 在所有手和任务上持续提升性能, 尤其在长时序任务上优势显著
- 仅任务奖励 (ObjDex reimpl) 在短时序上可接受, 长时序上崩溃
- 任务+辅助奖励 (无 curriculum) 有改善但不一致
- ManipTrans curriculum 训练不稳定, 初期高 reward 后期崩溃

**关于 ObjDex reimplementation**: DexMachina 的 reimpl 显著优于原始论文, 原因推测:
- ObjDex 使用两级框架 (高层腕部规划器 + 低层 RL), 规划器在小数据集上不可靠
- DexMachina 直接用 retarget 结果作为腕部 base action, 更简单更有效
- Genesis 仿真器比 IsaacGym 更稳定, 支持更多并行环境 (12k vs 2k)

**手部形态分析** (Section 5.3):

| 发现 | 说明 |
|------|------|
| 大手 + 全驱动 > 小手 + 欠驱动 | Allegro (不拟人但手指长) 表现出乎意料地好 |
| DOF 比尺寸更重要 | Schunk (驱动指尖+可折叠手掌) > Inspire/Ability (尺寸相似但欠驱动) |
| 策略适应硬件约束 | XHand 用左手托物体右手合盖; Inspire 双手协同合盖 (不同策略同一任务) |
| 欠驱动手策略偏离人类引导更多 | 形态差距迫使策略发现替代操作策略 |

### 4.4 消融实验

| 消融项 | 结论 |
|--------|------|
| Action mode | Hybrid (紧约束腕部残差 + 手指绝对) 最优 |
| Curriculum | VOC >> ManipTrans curriculum >> 无 curriculum |
| 辅助奖励 | 任务+辅助奖励 > 仅任务奖励, 但改善不一致; curriculum 才是关键 |
| Object-aware post-processing | 显著提升 retarget 质量和下游 RL 性能 |

---

## 5. Related Work Analysis

DexMachina 在 human2robot dexterous manipulation 领域中的定位:

| 方向 | 代表工作 | DexMachina 的区别 |
|------|----------|-------------------|
| RL for dexterous manipulation | DeXtreme, Dactyl, MyoDex | DexMachina 处理双手铰接物体长时序, 非单手简单任务 |
| IL for dexterous manipulation | DexCap, ACE, Open-TeleVision | DexMachina 不需要定制遥操系统, 用人手 demo 引导 RL |
| Human-to-robot retargeting | AnyTeleop, ObjDex | DexMachina 在运动学 retarget 上做物理碰撞后处理, 效果更好 |
| Curriculum learning | ManipTrans | DexMachina 的 VOC 比 ManipTrans 的物理参数衰减更适合长时序铰接任务 |
| Physics-based hand motion | ArtiGrasp, PhysHOI | ArtiGrasp 生成 MANO 运动 (graphics 目标); DexMachina 训练机器人手策略 (robotics 目标) |

关键定位: DexMachina 是目前唯一一个 **在多种商用灵巧手上训练双手铰接物体长时序操作策略** 的工作, 同时提供了一个跨手部设计的评估基准。

---

## 6. Limitations & Future Directions

### 论文明确指出的局限

| 局限 | 说明 | 论文建议 |
|------|------|----------|
| State-based 输入 | 依赖仿真器特权信息 (物体精确位姿, 关节角, 接触力) | Teacher-student distillation 到 vision policy |
| 高质量 demo 依赖 | ARCTIC 数据集需要动捕系统 + 密集标注, 采集昂贵 | 3D 生成模型和重建方法 |
| 仿真保真度 | 开源 URDF 的物理属性 (质量/惯性/碰撞) 是估计值 | 制造商提供精确仿真模型 |
| 无 sim2real | 缺乏硬件, 未在真实灵巧手上验证 | 作为 teacher policy 蒸馏 + sim2real transfer |

### 个人分析的潜在方向

1. **Sim2Real 是最大缺口**: 论文所有结果仅在仿真中, 无任何真实世界验证。VOC curriculum 依赖仿真中的虚拟力施加, sim2real gap 可能很大
2. **泛化能力未验证**: 每个 (物体, 手, demo) 组合需要独立训练策略, 无跨物体/跨 demo 泛化
3. **接触奖励计算效率**: 基于 mesh 顶点最近邻距离的接触近似比较粗糙, 且 contact_links 数据预计算增加了存储和预处理成本
4. **依赖 ARCTIC 格式**: 系统强耦合于 ARCTIC 数据集格式, 迁移到其他数据源 (如 DexYCB, HOI4D) 需要额外工作
5. **可扩展方向**: VOC 思想可推广到其他需要 curriculum 的 RL 任务 (如全身 locomotion + manipulation), 不局限于灵巧手

---

## 7. Paper vs Code Discrepancies

通过对比论文内容和代码仓库 (`repo/dexmachina/`), 发现以下差异和代码中未在论文中充分说明的细节:

| # | 项目 | 论文描述 | 代码实际 |
|---|------|----------|----------|
| 1 | Curriculum wait_epochs | 论文未明确 | 代码默认 2000 (而非论文 Algorithm 1 暗示的 500) |
| 2 | r_bc 命名 | 论文称 "behavior cloning reward" | 实际是关节级 imitation reward, 非 BC 意义上的 supervised cloning |
| 3 | Curriculum schedule | 论文仅描述 "exponential decay" | 代码支持 3 种: `fixed`, `exp`, `uniform` (per-env 采样) |
| 4 | Dialback 机制 | 论文 Algorithm 1 一笔带过 | 代码实现了复杂的 dialback 逻辑: 检测 episode 长度下降后回退增益, 乘以 0.98 ratio |
| 5 | 任务奖励公式 | 论文用乘法: `r_pos * r_rot * r_angle` | 代码支持乘法和加权和两种模式 (`multiply_task_rew` flag), 默认乘法 |
| 6 | zero_epoch 硬停 | 论文未提及 | 代码有 `zero_epoch=30000` 的硬停: 超过此 epoch 后强制将所有增益置零 |
| 7 | 重力补偿 | 论文未提及 | 代码中机器人手有 `gravity_compensation=0.8` 参数 |
| 8 | Contact reward weight | 论文列为重要组件 | 代码默认 `contact_rew_weight=0.0`, `imi_rew_weight=0.0`, 需要命令行显式开启 |
| 9 | 动作平滑 EMA | note.md 中提到 | 代码通过 `action_moving_avg` 参数控制, 默认值 1.0 (无平滑) |
| 10 | ManipTrans 实现 | 论文称 "faithful reimplementation" | 代码 `maniptrans_curr.py` 中对多个参数做了猜测 (原论文未公开衰减计划), 且修改了 Genesis 刚体求解器以支持在训练中修改重力 |
| 11 | DexRobot Hand | 论文评估 6 种手 | 代码 `hand_cfgs/dexrobot.py` 存在, 但论文主结果图中仅用 4 种 + 2 种额外评估 |

---

## 8. Cross-Paper Comparison

### 8.1 方法架构对比

| 维度 | DexMachina | BiDexHD | DexTrack | HumDex | ArtiGrasp |
|------|------------|---------|----------|--------|-----------|
| **年份** | 2025 | 2024 (ICLR 2025) | 2025 | 2026 | 2024 (3DV) |
| **目标** | 双手铰接物体功能性重定向 | 多任务双手灵巧操作 | 通用 tracking controller | 人形灵巧操作 pipeline | 双手抓取+铰接运动合成 |
| **手部** | 6 种商用灵巧手 (浮动手) | LEAP Hand + RM65 臂 | Shadow Hand (单手) | Unitree G1 全身 + Inspire Hand | MANO 模型 (graphics 手) |
| **物体** | 5 种铰接 (ARCTIC) | 141 任务 6 类 (TACO) | 3585 轨迹 (GRAB+TACO) | 真实工具 | 铰接物体 |
| **BC/IL** | 无 (pure RL) | Teacher-student distillation (DAgger) | RL+IL 混合 (IL 系数极小) | ACT pretrain + fine-tune | 无 (pure RL) |
| **动作空间** | Hybrid (腕部残差+手指绝对) | Position control (22D/手) | Double integration residual | Position control | 全局+局部 PD |
| **Curriculum** | VOC (虚拟物体控制器) | 无 (两阶段奖励自动切换) | Homotopy path (链式跳转) | 无 | 两阶段 (固定物体->自由物体) |
| **仿真器** | Genesis | IsaacGym | IsaacGym | IsaacGym | MuJoCo |
| **Sim2Real** | 无 | 有 (LEAP Hand) | 无 | 有 (Unitree G1) | 无 |
| **泛化** | 每组合独立训练 | 多任务共享策略 | 跨物体泛化 | 跨任务迁移 | Per-object 训练 |

### 8.2 关键技术路线差异

**探索辅助策略对比**:

| 方法 | DexMachina VOC | BiDexHD 两阶段奖励 | DexTrack Homotopy | ArtiGrasp 两阶段课程 |
|------|----------------|--------------------|-------------------|---------------------|
| 思路 | 物体被虚拟力"带着走", 策略逐步接管 | 先对齐再跟踪, 奖励自动切换 | 从简单轨迹逐步过渡到难轨迹 | 先固定物体单手学, 再自由物体双手学 |
| 衰减什么 | 虚拟控制器增益 | 无衰减 (条件门控) | 轨迹难度 | 环境约束 (固定->自由) |
| 是否改变物理 | 否 (添加虚拟力, 物理规则不变) | 否 | 否 | 否 (但改变物体约束) |
| 适用场景 | 长时序轨迹跟踪 | 多任务标准操作 | 跨物体泛化 | 铰接物体抓取+操作 |

**Retarget 策略对比**:

| 方法 | Retarget 方案 | 碰撞处理 | 腕部控制 |
|------|---------------|----------|----------|
| DexMachina | AnyTeleop + object-aware 物理后处理 | 仿真碰撞解析 (并行化) | Residual (紧约束 +/-4cm) |
| BiDexHD | dex-retargeting + IK 回放验证 | 未显式处理 | Position control (绝对) |
| DexTrack | kinematic reference + double integration | 网络隐式学习 | Double integration residual |
| HumDex | 优化式/学习式 MLP 重定向 | 自适应 alpha 混合 | Position control (绝对) |
| ArtiGrasp | 静态 grasp pose 参考 | 仿真碰撞 | RL 控制的全局手腕策略 |

### 8.3 对灵巧操作方法论的启示

1. **VOC 是当前最强的 long-horizon manipulation curriculum**: 相比 ManipTrans (物理参数衰减), ArtiGrasp (环境约束衰减), DexTrack (轨迹难度衰减), VOC 的 "保持物理不变, 仅辅助物体运动" 思路在长时序任务上优势最为明显

2. **Hybrid action space 可能是灵巧手的通解**: DexMachina 的"腕部紧约束残差 + 手指绝对"模式值得作为默认选择。DexTrack 的 double integration 也是一种 residual 变体, 但增加了平滑性。BiDexHD 和 HumDex 用绝对控制可能因为任务更短/有 teacher 数据

3. **Object-aware retarget 后处理是被低估的贡献**: 消除穿透后 ObjDex 从 41% 到 90%, 说明 retarget 质量对下游 RL 性能影响巨大。此方法简单、并行、不需要学习, 可直接迁移到其他 retarget pipeline

4. **Sim2Real 是 DexMachina 相对 BiDexHD/HumDex 的最大短板**: BiDexHD 和 HumDex 都有真实机器人验证, DexMachina 仅在仿真中评估。VOC 概念的 sim2real 可行性未知

5. **泛化能力差异**: DexTrack 追求跨物体泛化 (3585 轨迹), BiDexHD 追求多任务泛化 (141 任务), DexMachina 追求跨手部泛化 (6 种手)。三者优化方向不同, 互补而非替代

---
