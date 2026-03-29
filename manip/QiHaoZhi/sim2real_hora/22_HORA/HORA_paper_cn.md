# 通过快速运动自适应实现手内物体旋转

**齐昊之\*,1,2, Ashish Kumar\*,1, Roberto Calandra2, 马毅1, Jitendra Malik1,2**

1 UC Berkeley, 2 Meta AI

\* 同等贡献。

第六届机器人学习会议 (CoRL 2022)，新西兰奥克兰。

https://haozhi.io/hora/

---

**图 1**: **左:** 我们的控制器仅在仿真环境中使用不同大小和重量的简单圆柱形物体进行训练。**右:** 无需任何真实世界微调，该控制器即可部署到真实机器人上，仅使用本体感觉 (proprioceptive) 信息，就能操作具有不同形状、大小和重量的多种物体 (图中标注了物体质量和沿指尖方向的最短/最长直径轴长度)。**网站**: 在学习到的控制策略中可以观察到自然稳定的手指步态的涌现。

---

**摘要:** 通用化的手内操控长期以来一直是机器人技术中一个未解决的挑战。作为迈向这一宏伟目标的一小步，我们展示了如何设计和学习一个简单的自适应控制器，仅使用指尖实现手内物体旋转。该控制器完全在仿真环境中仅使用圆柱形物体进行训练，然后——无需任何微调——可以直接部署到真实的机器人手上，围绕 z 轴旋转数十种具有不同大小、形状和重量的物体。这是通过仅使用本体感觉历史对机器人控制器进行快速在线自适应 (rapid online adaptation) 来适应物体属性而实现的。此外，通过强化学习 (Reinforcement Learning) 训练控制策略，自然且稳定的手指步态 (finger gaits) 会自动涌现。代码和更多视频可在我们的网站上获取。

**关键词:** 手内操控 (In-Hand Manipulation)，物体旋转 (Object Rotation)，强化学习 (Reinforcement Learning)

---

## 1 引言

人类非常擅长手内物体操控——他们甚至可以毫不费力地适应不同形状、大小、质量和材料的新物体。虽然已有多项工作展示了使用真实世界多指手对单个或少量物体进行手内旋转 [1, 2, 3, 4]，但真正通用化的手内操控仍然是机器人技术中一个未解决的挑战。

在本文中，我们证明了可以训练一个自适应控制器，使其能够用多指机器人手的指尖围绕 z 轴旋转多种物体 (图 1)。该任务是通用手内重定向任务的简化，但对于机器人来说仍然相当具有挑战性，因为手指需要始终保持对物体的动态或静态力闭合 (force closure) 以防止其掉落 (因为它不能利用任何其他支撑面，如手掌)。

我们的方法受到最近使用强化学习在腿式运动 (legged locomotion) 方面取得的进展 [5, 6] 的启发。这些工作的核心是学习用于行走的不同地形属性的压缩表示 (称为外部参数 *extrinsics*)，该表示与控制策略联合训练。在部署期间，外部参数在线估计，控制器可以对其进行快速自适应。我们的关键洞察 (key insight) 是，尽管真实世界物体种类繁多，但对于手内物体旋转任务，*指尖*感知到的重要物理属性 (如局部形状、质量和大小) 可以被压缩到一个紧凑的表示中。一旦学习到不同物体的压缩表示 (外部参数 *extrinsics*)，控制器就可以从本体感觉历史在线估计它，并使用它来自适应地操控多种物体。

具体来说，我们将物体的内在属性 (intrinsic properties，如质量和大小) 编码为一个外部参数向量 (extrinsics vector)，并以此作为输入训练自适应策略。学习到的策略可以在仿真环境中鲁棒且高效地旋转不同物体。然而，当我们在真实世界中部署策略时，无法获得外部参数。为了解决这个问题，我们使用快速运动自适应 (rapid motor adaptation) [6] 来学习一个自适应模块 (adaptation module)，该模块利用观察到的本体感觉历史与指令动作之间的差异来估计外部参数向量。该自适应模块也可以完全在仿真中通过监督学习进行训练。使用本体感觉历史来估计物理属性的概念已在运动控制 [5, 6, 7] 中广泛使用，但尚未在手内操控中被探索。

在多指 Allegro Hand [8] 上的实验结果表明，我们的方法可以在真实世界中成功旋转超过 30 种具有不同大小 (从 4.5 cm 到 7.5 cm)、质量 (从 5 g 到 200 g) 和其他物理属性 (例如可变形或柔软的物体) 的物体。我们还观察到，从学习过程中涌现出了自适应且平滑的手指步态。我们的方法展示了仅使用本体感觉传感信号进行不同物体自适应的惊人有效性，甚至无需使用视觉和触觉感知。为了进一步理解我们方法的底层机制，我们研究了在操控不同物体时估计的外部参数。我们发现了与质量和尺度变化相关的可解释外部参数值，并且嵌入的低维结构确实存在，这两者对我们的泛化能力至关重要。

## 2 相关工作

**手内操控的经典控制方法。** 灵巧手内操控 (dexterous in-hand manipulation) 已经是数十年来一个活跃的研究领域 [9]。经典控制方法通常需要物体和机器人几何形状的解析模型来进行物体操控的运动规划。例如，[10, 11] 依赖此类模型来规划手指运动以旋转物体。[12] 假设物体是分段光滑的，并使用手指跟踪来旋转物体。[13, 14] 通过使用优化生成轨迹，在仿真中演示了不同物体的重定向。也有尝试在真实世界中部署系统的工作。例如，[15] 计算精确的接触位置来规划一系列接触位置用于旋转物体。[16] 在一组预定义的抓取策略上进行规划，以使用两只多指手实现物体重定向。[17, 18] 使用投掷或外力来扰动空中的物体并重新抓取。最近，[19, 20] 等工作在不断开接触的情况下进行抓取内操控。[4] 通过利用柔顺性和精确的位姿跟踪器，使用非拟人手展示了复杂的物体重定向技能。由于物理世界的内在复杂性，它们能够操控的物体多样性仍然有限。与传统控制方法可能使用启发式规则或简化模型来解决这个任务不同，我们使用无模型强化学习 (model-free reinforcement learning) 来训练自适应策略，并使用自适应来实现泛化。

**手内操控的强化学习。** 为了绕过对精确物体模型和物理属性测量的需求，在过去几年中，人们对直接在真实世界中使用强化学习进行灵巧手内操控的兴趣日益增长。[21] 学习了圆柱形物体的简单抓取内滚动。[22, 23] 学习动力学模型并在其上进行规划以旋转手掌上的物体。[24, 25] 使用人类演示来加速学习过程。然而，由于强化学习的样本效率非常低，学习到的技能相当简单或物体多样性有限。虽然可以在仿真中获得复杂技能，如重定向多种物体 [26, 27, 28] 和工具使用 [29, 30]，但将结果转移到真实世界仍然具有挑战性。我们的方法不是直接在真实世界中训练策略，而是完全在仿真器中学习策略，并旨在直接转移到真实世界。

**通过域随机化 (Domain Randomization) 实现 Sim-to-Real 迁移。** 若干工作旨在使用仿真器训练强化学习策略并直接部署到真实世界系统中。域随机化 [31] 在训练期间改变仿真参数，使策略暴露于多种仿真环境中，以便可以鲁棒地部署到真实世界中。代表性的例子是 [1] 和 [2]。他们利用大量计算资源和大规模强化学习方法来学习灵活的物体重定向技能，并用单个机器人手解决魔方。然而，它们仍然只关注操控有限数量的物体。[3] 高效地学习了手指步态行为，并在手朝下时转移到真实机器人，但他们不仅使用指尖，而且考虑的物体都是立方体。我们的方法关注对多种物体的泛化，并且可以在几个小时内完成训练。

**通过自适应实现 Sim-to-Real 迁移。** [32] 不依赖于对当前环境参数不可知的域随机化，而是通过初始校准进行系统辨识 (system identification)，或通过在线自适应控制来估计 Sim-to-Real 迁移的系统参数。然而，由于物理仿真的固有不精确性，学习精确的物理值以及仿真与真实世界之间的对齐可能是次优的。另一种方式是学习一个低维嵌入 (low-dimensional embedding) 来编码环境参数 [5, 6]，然后控制策略使用该嵌入来采取动作。这种范式已经实现了鲁棒且自适应的运动策略。然而，将其直接应用于手内操控任务并不简单。我们的方法展示了如何设计奖励和训练环境，以实现一个可以迁移到真实世界的自然且稳定的控制器。

## 3 用于手内物体旋转的快速运动自适应

我们方法的概述如图 2 所示。在部署期间 (图 2，下方)，我们的策略从本体感觉和动作历史推断物体属性 (如大小和质量) 的低维嵌入，然后由我们的基础策略用于旋转物体。我们首先描述如何使用仿真器提供的物体属性训练*基础策略* (base policy)，然后讨论如何训练能够推断这些属性的*自适应模块* (adaptation module)。

### 3.1 基础策略训练

**特权信息 (Privileged Information)。** 在本文中，特权信息是指物体的属性，如位置、大小、质量、摩擦系数、物体质心。该信息在时间步 $t$ 表示为一个 9 维向量 $\mathbf{e}_t \in \mathbb{R}^9$，可以在仿真中准确测量。我们将其作为策略的输入，但不直接使用 $\mathbf{e}_t$，而是使用一个 8 维嵌入 (在 [6] 中称为外部参数 extrinsics) $\mathbf{z}_t = \mu(\mathbf{e}_t)$，如我们在第 5 节中所示，这提供了更好的泛化行为。

**基础策略。** 我们的控制策略 $\pi$ 以当前机器人关节位置 $\mathbf{q}_t \in \mathbb{R}^{16}$、上一时间步的预测动作 $\mathbf{a}_{t-1} \in \mathbb{R}^{16}$ 以及外部参数向量 $\mathbf{z}_t \in \mathbb{R}^8$ 作为输入，输出 PD 控制器的目标 (记为 $\mathbf{a}_t$)。我们还增强了观测以包含两个额外的时间步，从而获得速度和加速度信息。形式化地，基础策略输出 $\mathbf{a}_t = \pi(\mathbf{o}_t, \mathbf{z}_t)$，其中 $\mathbf{o}_t = (\mathbf{q}_{t-2:t}, \mathbf{a}_{t-3:t-1}) \in \mathbb{R}^{96}$。

**奖励函数。** 我们使用 PPO [33] 联合优化策略 $\pi$ 和嵌入 $\mu$。奖励函数依赖于以下几个量：$\boldsymbol{\omega}$ 是物体的角速度。$\hat{\mathbf{k}}$ 是期望的旋转轴 (我们使用手部坐标系中的 z 轴)。$\mathbf{q}_{\text{init}}$ 是初始机器人配置。$\boldsymbol{\tau}$ 是每个时间步的指令力矩。$\mathbf{v}$ 是物体的线速度。我们要最大化的奖励函数 $r$ (为简洁起见省略下标 $t$) 为

$$r \doteq r_{\text{rot}} + \lambda_{\text{pose}} r_{\text{pose}} + \lambda_{\text{linvel}} r_{\text{linvel}} + \lambda_{\text{work}} r_{\text{work}} + \lambda_{\text{torque}} r_{\text{torque}} \tag{1}$$

其中 $r_{\text{rot}} \doteq \max(\min(\boldsymbol{\omega} \cdot \hat{\mathbf{k}}, r_{\text{max}}), r_{\text{min}})$ 是旋转奖励，$r_{\text{pose}} \doteq -\|\mathbf{q} - \mathbf{q}_{\text{init}}\|_2^2$ 是手部姿态偏差惩罚，$r_{\text{torque}} \doteq -\|\boldsymbol{\tau}\|_2^2$ 是力矩惩罚，$r_{\text{work}} \doteq -\boldsymbol{\tau}^T \dot{\mathbf{q}}$ 是能量消耗惩罚，$r_{\text{linvel}} \doteq -\|\mathbf{v}\|_2^2$ 是物体线速度惩罚。注意，与 [28] 显式鼓励至少三个指尖始终与物体保持接触不同，我们不强制任何启发式的手指步态行为。相反，稳定的手指步态行为从能量约束和初始姿态偏差惩罚中自然涌现。

**物体初始化和动力学随机化。** 良好的训练环境必须在仿真中提供足够的多样性以实现在真实世界中的泛化。在本工作中，我们发现使用不同长宽比和质量的圆柱体可以提供这种多样性。我们均匀采样圆柱体的不同直径和侧面长度。

我们将物体和手指初始化为稳定的精密抓取 (precision grasp)。与 [28] 中构造指尖位置不同，我们简单地围绕一个标准抓取随机采样物体位置、姿态和机器人关节位置，直到实现稳定抓取。我们还随机化这些物体的质量、质心和摩擦力 (详见附录)。

### 3.2 自适应模块训练

我们不能直接将学习到的策略 $\pi$ 部署到真实世界，因为我们无法直接观测向量 $\mathbf{e}_t$，因此无法计算外部参数 $\mathbf{z}_t$。相反，我们通过自适应模块 $\phi$ 从本体感觉历史和指令动作历史之间的差异来估计外部参数向量 $\hat{\mathbf{z}}_t$。这个想法受到最近运动控制领域工作 [5, 6] 的启发，其中本体感觉历史被用来估计地形属性。我们表明这些信息也可以用来估计物体属性。

为了训练这个网络，我们首先通过执行策略 $\pi(\mathbf{o}_t, \hat{\mathbf{z}}_t)$ 来收集轨迹和特权信息，其中预测的外部参数向量 $\hat{\mathbf{z}}_t = \phi(\mathbf{q}_{t-k:t}, \mathbf{a}_{t-k-1:t-1})$。同时我们也存储真实的外部参数向量 $\mathbf{z}_t$ 并构建训练集

$$\mathcal{B} = \{(\mathbf{q}_{t-k:t}^{(i)}, \mathbf{a}_{t-k-1:t-1}^{(i)}, \mathbf{z}_t^{(i)}, \hat{\mathbf{z}}_t^{(i)})\}_{i=1}^N.$$

然后我们使用 Adam [34] 通过最小化 $\mathbf{z}_t$ 和 $\hat{\mathbf{z}}_t$ 之间的 $\ell_2$ 距离来优化 $\phi$。该过程迭代进行直到损失收敛。我们使用与上述章节相同的物体初始化和动力学随机化设置。

## 4 实验设置和实现细节

**硬件设置。** 我们使用 Wonik Robotics 的 Allegro Hand [8]。Allegro Hand 是一种灵巧的拟人机器人手，具有四根手指，每根手指有四个自由度。这 16 个关节使用位置控制，频率为 20 Hz。目标位置指令通过 PD 控制器 ($K_p = 3.0, K_d = 0.1$) 以 300 Hz 转换为力矩。

**仿真设置。** 我们使用 IsaacGym 仿真器 [35]。在训练期间，我们使用 16384 个并行环境来收集训练智能体的样本。每个环境包含一个仿真的 Allegro Hand 和一个具有不同形状和物理属性的圆柱形物体 (确切参数在补充材料中)。仿真频率为 120 Hz，控制频率为 20 Hz。每个回合持续 400 个控制步 (相当于 20 秒)。

**基线方法。** 我们将我们的方法与以下列出的基线方法进行比较。我们还与具有特权信息访问权限的策略 (*Expert*) 进行比较，作为我们方法的上界 (图 2，上排)。

1. *使用域随机化训练的鲁棒策略 (DR):* 该基线使用相同的奖励函数训练，但不使用特权信息。这给出了一个对所有形状和物理属性变化鲁棒的策略，而不是自适应的策略 [1, 3, 2, 36]。
2. *在线显式系统辨识 (SysID):* 该基线在训练自适应模块时预测精确的系统参数 $\mathbf{e}_t$，而不是外部参数向量 $\mathbf{z}_t$。
3. *无在线自适应 (NoAdapt):* 在部署期间，外部参数向量 $\hat{\mathbf{z}}_t$ 在第一个时间步估计，并在剩余运行期间保持冻结。这是为了研究自适应模块 $\phi$ 所实现的在线自适应的重要性。
4. *动作回放 (Periodic):* 我们从具有特权信息的专家策略记录参考轨迹，并盲目运行它。这是为了表明我们的策略可以适应不同的物体和干扰，而不是周期性地执行相同的动作序列。

**评估指标。** 我们使用以下指标来比较我们方法与基线方法的性能。

1. *坠落时间 (Time-to-Fall, TTF)。* 物体从手中掉落之前的平均回合长度。该值由最大回合长度归一化 (仿真实验中为 20 秒，真实世界实验中为 30 秒)。
2. *旋转奖励 (Rotation Reward, RotR)。* 这是仿真中回合的平均旋转奖励 ($\boldsymbol{\omega} \cdot \hat{\mathbf{k}}$)。注意我们不使用这个奖励进行训练。相反，我们在训练期间使用这个奖励的裁剪版本。
3. *回合内旋转弧度 (Rotations)。* 由于物体角速度在真实世界中难以精确测量，我们改为测量策略相对于世界 z 轴实现的物体净旋转 (以弧度为单位)。该指标仅在真实世界实验中使用。
4. *物体线速度 (ObjVel)。* 我们测量物体线速度的大小以衡量物体的稳定性。该值被缩放 100 倍。该指标仅在仿真中测量。
5. *力矩惩罚 (Torque)。* 我们测量执行过程中每个时间步指令力矩的平均 $\ell_1$ 范数，以衡量能量效率。

**表 1:** 我们在仿真中将我们的方法与若干基线方法在两种设置下进行比较: 1) *训练分布内*; 2) *分布外*。我们的在线连续自适应方法与所有基线方法相比取得了最佳性能，紧密模拟了以特权信息作为输入的 Expert 的性能。

| 方法 | 训练分布内 | | | | 分布外 | | | |
|--------|---|---|---|---|---|---|---|---|
| | RotR (↑) | TTF (↑) | ObjVel (↓) | Torque (↓) | RotR (↑) | TTF (↑) | ObjVel (↓) | Torque (↓) |
| *Expert* | 233.71±25.24 | 0.85±0.01 | 0.28±0.08 | 1.24±0.19 | 165.07±15.63 | 0.71±0.04 | 0.42±0.06 | 1.24±0.16 |
| Periodic | 43.62±2.52 | 0.44±0.12 | 0.72±0.21 | 1.77±0.49 | 22.45±0.59 | 0.34±0.08 | 1.11±0.19 | 1.41±0.54 |
| NoAdapt | 90.89±4.85 | 0.65±0.07 | 0.44±0.11 | 1.34±0.12 | 54.50±3.91 | 0.51±0.06 | 0.63±0.13 | 1.34±0.11 |
| DR | 176.12±26.47 | 0.81±0.02 | 0.34±0.05 | 1.42±0.06 | 140.80±17.51 | 0.63±0.02 | 0.64±0.06 | 1.48±0.17 |
| SysID | 174.42±23.31 | 0.81±0.02 | 0.32±0.03 | 1.29±0.72 | 132.56±17.42 | 0.62±0.08 | 0.50±0.09 | 1.26±0.17 |
| **Ours** | **222.27±21.20** | **0.82±0.02** | **0.29±0.05** | **1.20±0.19** | **160.60±10.22** | **0.68±0.07** | **0.47±0.07** | **1.20±0.17** |

## 5 结果与分析

在本节中，我们将我们方法的性能与仿真和真实世界部署中的若干基线方法进行比较。我们还分析了自适应模块学到了什么，以及它在策略执行过程中和物体变化时如何变化。最后，我们专注于训练一个策略用于围绕相对于世界坐标系的负 z 轴旋转物体。我们还在附录中探索了训练多轴策略 (+/- z 轴) 的可能性。

### 5.1 通过自适应实现泛化

**仿真中的比较。** 我们首先在仿真中将我们的方法与第 4 节中提到的基线方法进行比较。我们在两种设置下评估所有方法: 1) 在*训练分布内*设置中，我们使用与强化学习训练相同的物体集和随机化设置; 2) 在*分布外*设置中，我们使用具有更大物理随机化范围的物体。我们还将 20% 的物体更改为球体和立方体。我们计算 500K 个具有不同参数随机化和初始化条件的回合的平均性能。我们报告使用不同种子训练的五个模型的平均值和标准差。

表 1 中的结果表明，我们的在线自适应方法与所有基线方法相比取得了最佳性能。我们看到，对物体形状和动力学的自适应不仅在训练中实现了更好的性能，而且与所有基线方法相比，对分布外物体参数的泛化效果也更好。*Periodic* 基线 (即简单回放专家策略) 没有给出合理的性能。虽然它可以用相同的初始抓取和动力学参数旋转完全相同的物体，但它无法泛化到这个非常狭窄的设置之外。该基线帮助我们理解问题的难度。*NoAdapt* 基线的性能也比我们使用连续在线自适应的方法差。该基线较弱的性能可以解释为它在回合期间不更新外部参数。这表明了连续在线自适应的重要性。*DR* 基线虽然在 *RotR* 和 *TTF* 方面大致匹配或优于其他基线，但在与物体稳定性和能量效率相关的其他指标上表现更差。这是因为 *DR* 基线不了解底层物体属性，需要为所有可能的物体学习单一步态，而不是自适应步态。*SysID* 基线在两种评估分布中的性能也比我们的方法差。这是因为学习形状和动力学参数的精确值既困难又不必要。这个比较表明了学习低维紧凑表示的好处，该表示对不同物理属性具有粗略的相对激活，而不是精确值 (参见图 5 和图 6)。

**真实世界比较。** 接下来，我们展示我们的策略与基线方法在图 3 和图 4 所示的两组具有挑战性的物体上的真实世界比较。我们排除了 *Periodic* 基线，因为它即使在仿真中也效果不佳，以及 *Expert* 因为我们无法在真实世界中获取特权信息。我们使用 20 个不同的初始抓取位置和每组 6 个不同的物体来评估每种方法。最大回合长度为 30 秒。

我们首先研究我们的策略和不同基线方法在一组重物体 (超过 100 g) 上的行为，包括棒球、不同的水果、蔬菜和杯子 (图 3)。我们的方法在所有指标上表现显著更好。我们的方法几乎在所有试验中都能实现一致的旋转而不掉落，平均旋转弧度为 23.96 (等效旋转速度 0.8 rad s$^{-1}$)。我们发现 *DR* 基线在所有试验中都非常缓慢和保守，因为它不了解物体属性，需要为所有物体学习单一的步态行为。因此，它的旋转弧度最低，尽管 TTF 高于 *SysID* 基线。我们还发现 *DR* 在较大尺寸的物体上表现尚可，因为它们更容易旋转，并且构成了训练分布的合理部分。*SysID* 基线学习了更具动态性和灵活性的行为，具有更好的旋转指标，但 TTF 较低，表明系统参数的精确估计既不必要也很困难。我们发现它特别难以泛化到小杯子和略微柔软的番茄。最后，我们观察到与 [6] 中类似的行为，*NoAdapt* 基线无法成功转移到真实世界，表明连续在线自适应对于成功的真实世界部署的重要性。

我们在一组不规则物体 (图 4) 上进行了相同的比较。它包含一个具有移动质心的容器、具有凹面的物体、圆柱形猕猴桃、羽毛球、带孔的玩具和立方体玩具。旋转立方体特别困难，因为我们仅依赖指尖的使用。虽然这些变化对于该任务具有挑战性，并且超出了训练期间所见的范围，但我们的方法仍然表现合理，优于所有基线方法。我们看到 *DR* 基线可以对容器和羽毛球执行稳定但缓慢的手内旋转，但对其他物体基本失败，表明其在形状泛化方面的困难。对于 *SysID* 基线，尽管具有更高的角速度，但其稳定性显著低于 *DR* 基线和我们的方法。*NoAdapt* 基线的表现与我们在图 3 中观察到的类似。

**图 3:** 在多种重物体 (左) 上的定量评估。我们使用自适应的方法在总旋转角度 (弧度)、坠落时间 (TTF) 和能量效率 (Torque) 方面表现最佳。*DR* 基线有一个保守的策略，导致较慢的角速度。*SysID* 有一个更具动态性和灵活性的策略，但非常不稳定，可以从比 *DR* 和我们的方法更低的 TTF 看出。*NoAdapt* 基线在该任务上失败，表明连续在线自适应的重要性。

| 方法 | Rotations (↑) | TTF (↑) | Torque (↓) |
|--------|---------------|---------|------------|
| DR | 9.67±4.33 | 0.72±0.34 | 2.03±0.36 |
| SysID | 10.36±2.32 | 0.61±0.33 | 1.88±0.38 |
| NoAdapt | N.A. | 0.35±0.20 | N.A. |
| **Ours** | **23.96±3.16** | **0.98±0.08** | **1.84±0.24** |

**图 4:** 在多种不规则物体 (左) 上的定量评估。我们的方法可以成功泛化到旋转多种物体，包括带孔的物体、柔软和可变形的物体 (这些都未包含在训练中)。我们的方法在所有指标上优于基线方法。*DR* 基线具有第二高的 TTF 但旋转弧度较低，因为它输出非常保守和缓慢的轨迹。*SysID* 实现了略快但非常不稳定的策略。我们的方法优于基线方法的表现表明了通过低维外部参数估计实现自适应对于该任务泛化的重要性。

| 方法 | Rotations (↑) | TTF (↑) | Torque (↓) |
|--------|---------------|---------|------------|
| DR | 6.59±3.71 | 0.66±0.41 | 1.85±0.37 |
| SysID | 8.16±3.39 | 0.46±0.36 | 1.70±0.40 |
| NoAdapt | N.A. | 0.12±0.05 | N.A. |
| **Ours** | **19.22±4.88** | **0.78±0.27** | **1.48±0.30** |

### 5.2 理解与分析

**外部参数随时间的变化。** 我们在真实世界中运行一个连续的评估回合，其中我们每 30 秒更换手中的物体，共 6 个物体。注意在训练期间，我们从未在一个回合内随机化物体。在整个运行过程中，我们记录估计的外部参数，并在图 5 中绘制 8 维外部参数向量中的 2 个维度。上方的图显示外部参数值 $z_{t,0}$ 对物体直径变化的响应。较小直径时值较低，较大直径时值较高。下方的图显示外部参数值 $z_{t,2}$ 对物体质量变化的响应。较轻的物体获得较高的值，较重的物体获得较低的值。

**外部参数聚类。** 理解估计的外部参数 $\hat{z}$ 的另一种方式是对旋转不同物体时估计的外部参数向量进行聚类。在图 6 中，我们使用 t-SNE 可视化旋转 6 个不同物体各 5 秒时估计的外部参数向量。我们发现不同大小和不同重量的物体倾向于占据不同的区域。例如，对应于较小物体的 $\hat{z}$ 倾向于聚集在左下方。ID 为 1 的物体 (在上图中左下方显示) 产生分散的 $\hat{z}$，因为其不规则的形状导致了物体等效尺度的变化。这种局部性导致外部参数在 t-SNE 图中的分散，而这种局部性也是使我们获得泛化能力的原因，如表 1、图 3 和图 4 中所评估的。

**涌现的手指步态。** 我们发现使用圆柱形物体对于涌现稳定且高间隙的步态很重要。在我们的网站上，我们比较了使用圆柱形物体和纯球形物体训练时学习到的手指步态。后一种训练方案导致了一种动态步态的策略，该策略在球上效果很好，但无法泛化到更复杂的物体。

### 5.3 真实世界定性结果

在我们的项目网站上，我们定性地将我们的方法与基线方法在 5 种不同物体上进行比较，并进一步在 30 种具有不同形状和物理属性的物体上评估我们的方法，包括多孔物体和非刚性物体。直径范围从 4.5 cm 到 7.5 cm，质量范围从 5 g 到 200 g。注意我们在训练期间仅使用不同大小和长宽比的圆柱形物体，并在真实世界中通过快速自适应展示形状泛化。允许这种泛化的关键洞察是，手指尖感知到的物体形状可以被压缩到低维空间中。我们从本体感觉历史估计这个低维空间，这使我们能够泛化到在真实世界中看似不同但在外部参数空间中可能看起来相似的物体。

## 6 讨论与局限性

通用灵巧手内操控存在不同难度级别。本文考虑的任务 (围绕 z 轴的手内物体旋转) 是一般 SO(3) 重定向问题的简化。然而，这不是一个限制性的简化。通过三个策略分别对应三个主轴的旋转，我们可以实现将物体旋转到任意目标姿态。我们将此任务视为我们工作的一个重要未来扩展。

我们展示了纯本体感觉手内物体旋转对于多种物体的可行性。我们发现真实世界实验中的大多数失败案例是由于不正确的接触点导致不稳定的力闭合。由于我们的方法仅依赖于本体感觉感知，它无法感知物体与指尖之间的精确接触位置。另一个失败案例是当要旋转的物体很小 (直径小于 4.0 cm) 时。在这种情况下，手指会频繁相互碰撞，使机器人无法保持物体的抓取平衡。更极端和复杂的形状也更难操控。对于那些更具挑战性的任务，可能需要结合触觉或视觉反馈。

我们旨在通过自适应研究向真实世界的泛化，因此我们不利用真实世界的经验来改进我们的策略。将真实世界数据纳入以改进我们的策略 (例如通过使用元学习 meta-learning) 将是一个有趣且有意义的下一步。

## 致谢

本研究得到了与 Meta 合作的 BAIR 开放研究共同项目的支持。此外，在 UC Berkeley 的学术角色中，Haozhi、Ashish 和 Jitendra 得到了 DARPA Machine Common Sense (MCS) 的部分支持，Haozhi 和 Yi 得到了 ONR (N00014-20-1-2002 和 N00014-22-1-2102) 的支持。我们感谢 Tingfan Wu 和 Mike Lambeta 在硬件设置上的慷慨帮助，感谢 Xinru Yang 帮助录制真实世界视频。我们还感谢 Mike Lambeta、Yu Sun、Tingfan Wu、Huazhe Xu 和 Xinru Yang 对本项目早期版本提供的反馈。

## 参考文献

[1] OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba. Learning Dexterous In-Hand Manipulation. *The International Journal of Robotics Research (IJRR)*, 2019.

[2] OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang. Solving Rubik's Cube with a Robot Hand. *arXiv preprint arXiv:1910.07113*, 2019.

[3] L. Sievers, J. Pitz, and B. Bauml. Learning Purely Tactile In-Hand Manipulation with a Torque-Controlled Hand. *International Conference on Robotics and Automation (ICRA)*, 2022.

[4] A. Morgan, K. Hang, B. Wen, K. E. Bekris, and A. Dollar. Complex In-Hand Manipulation via Compliance-Enabled Finger Gaiting and Multi-Modal Planning. *IEEE Robotics and Automation Letters (RA-L)*, 2022.

[5] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter. Learning Quadrupedal Locomotion over Challenging Terrain. *Science Robotics*, 2020.

[6] A. Kumar, Z. Fu, D. Pathak, and J. Malik. RMA: Rapid Motor Adaptation for Legged Robots. In *Robotics: Science and Systems (RSS)*, 2021.

[7] Z. Fu, A. Kumar, A. Agarwal, H. Qi, J. Malik, and D. Pathak. Coupling Vision and Proprioception for Navigation of Legged Robots. In *Computer Vision and Pattern Recognition (CVPR)*, 2022.

[8] WonikRobotics. AllegroHand. https://www.wonikrobotics.com/, 2013.

[9] A. M. Okamura, N. Smaby, and M. R. Cutkosky. An Overview of Dexterous Manipulation. In *International Conference on Robotics and Automation (ICRA)*, 2000.

[10] L. Han and J. C. Trinkle. Dextrous Manipulation by Rolling and Finger Gaiting. In *International Conference on Robotics and Automation (ICRA)*, 1998.

[11] J.-P. Saut, A. Sahbani, S. El-Khoury, and V. Perdereau. Dexterous Manipulation Planning Using Probabilistic Roadmaps in Continuous Grasp Subspaces. In *International Conference on Intelligent Robots and Systems (IROS)*, 2007.

[12] D. Rus. In-Hand Dexterous Manipulation of Piecewise-Smooth 3-D Objects. *The International Journal of Robotics Research (IJRR)*, 1999.

[13] Y. Bai and C. K. Liu. Dexterous Manipulation Using Both Palm and Fingers. In *International Conference on Robotics and Automation (ICRA)*, 2014.

[14] I. Mordatch, Z. Popovic, and E. Todorov. Contact-Invariant Optimization for Hand Manipulation. In *Eurographics*, 2012.

[15] R. Fearing. Implementing a Force Strategy for Object Re-orientation. In *International Conference on Robotics and Automation (ICRA)*, 1986.

[16] R. Platt, A. H. Fagg, and R. A. Grupen. Manipulation Gaits: Sequences of Grasp Control Tasks. In *International Conference on Robotics and Automation (ICRA)*, 2004.

[17] N. Furukawa, A. Namiki, S. Taku, and M. Ishikawa. Dynamic Regrasping Using a High-speed Multifingered Hand and a High-speed Vision System. In *International Conference on Robotics and Automation (ICRA)*, 2006.

[18] N. C. Dafle, A. Rodriguez, R. Paolini, B. Tang, S. S. Srinivasa, M. Erdmann, M. T. Mason, I. Lundberg, H. Staab, and T. Fuhlbrigge. Extrinsic Dexterity: In-Hand Manipulation with External Forces. In *International Conference on Robotics and Automation (ICRA)*, 2014.

[19] C. Teeple, B. Aktas, M. C.-S. Yuen, G. Kim, R. D. Howe, and R. Wood. Controlling Palm-Object Interactions Via Friction for Enhanced In-Hand Manipulation. *IEEE Robotics and Automation Letters (RA-L)*, 2022.

[20] B. Sundaralingam and T. Hermans. Relaxed-Rigidity Constraints: Kinematic Trajectory Optimization and Collision Avoidance for In-Grasp Manipulation. *Autonomous Robots*, 2019.

[21] H. Van Hoof, T. Hermans, G. Neumann, and J. Peters. Learning Robot In-Hand Manipulation with Tactile Features. In *International Conference on Humanoid Robots (Humanoids)*, 2015.

[22] V. Kumar, E. Todorov, and S. Levine. Optimal Control with Learned Local Models: Application to Dexterous Manipulation. In *International Conference on Robotics and Automation (ICRA)*, 2016.

[23] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar. Deep Dynamics Models for Learning Dexterous Manipulation. In *Conference on Robot Learning (CoRL)*, 2019.

[24] A. Gupta, C. Eppner, S. Levine, and P. Abbeel. Learning Dexterous Manipulation for a Soft Robotic Hand from Human Demonstration. In *International Conference on Intelligent Robots and Systems (IROS)*, 2016.

[25] V. Kumar, A. Gupta, E. Todorov, and S. Levine. Learning Dexterous Manipulation Policies from Experience and Imitation. *arXiv*, 2016.

[26] W. Huang, I. Mordatch, P. Abbeel, and D. Pathak. Generalization in Dexterous Manipulation via Geometry-Aware Multi-Task Learning. *arXiv*, 2021.

[27] T. Chen, J. Xu, and P. Agrawal. A System for General In-Hand Object Re-Orientation. In *Conference on Robot Learning (CoRL)*, 2022.

[28] G. Khandate, M. Haas-Heger, and M. Ciocarlie. On the Feasibility of Learning Finger-gaiting In-hand Manipulation with Intrinsic Sensing. In *International Conference on Robotics and Automation (ICRA)*, 2022.

[29] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. In *Robotics: Science and Systems (RSS)*, 2018.

[30] I. Radosavovic, X. Wang, L. Pinto, and J. Malik. State-Only Imitation Learning for Dexterous Manipulation. In *International Conference on Intelligent Robots and Systems (IROS)*, 2021.

[31] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. In *International conference on intelligent robots and systems (IROS)*, 2017.

[32] Y. Karayiannidis, C. Smith, D. Kragic, et al. Adaptive Control for Pivoting with Visual and Tactile Feedback. In *International Conference on Robotics and Automation (ICRA)*, 2016.

[33] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimization Algorithms. *arXiv*, 2017.

[34] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *International Conference on Learning Representations (ICLR)*, 2015.

[35] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning. *arXiv*, 2021.

[36] A. Allshire, M. Mittal, V. Lodaya, V. Makoviychuk, D. Makoviichuk, F. Widmaier, M. Wuthrich, S. Bauer, A. Handa, and A. Garg. Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World Trifinger. *arXiv*, 2021.

[37] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and Accurate Deep Learning by Exponential Linear Units (ELUs). In *International Conference on Learning Representations (ICLR)*, 2016.
