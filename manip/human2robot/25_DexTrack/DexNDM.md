# DexNDM: 通过 Joint-wise Neural Dynamics Model 弥合灵巧手内旋转的 Reality Gap

Xueyi Liu1,3, He Wang2,4, Li Yi1,3
1Tsinghua University 2Peking University 3Shanghai Qi Zhi Institute 4Galbot
项目网站: [meowuu7.github.io/DexNDM](https://meowuu7.github.io/DexNDM/)

## 摘要

实现通用的手内物体旋转(in-hand object rotation)仍然是机器人领域的重大挑战,主要原因在于将 policy 从仿真迁移到真实世界的困难。灵巧操作中复杂的、接触密集的动力学(contact-rich dynamics)造成了"reality gap"(仿真与真实之间的差距),这使得先前的工作局限于受限的场景,例如简单的几何形状、有限的物体尺寸和长宽比、受约束的手腕姿态或定制化的机械手。我们通过一个新颖的框架来解决这一 sim-to-real 挑战,使得一个在仿真中训练的 policy 能够泛化到真实世界中各种各样的物体和条件。我们方法的核心是一个 joint-wise dynamics model(按关节建模的动力学模型),它通过有效拟合有限的真实世界采集数据来学习弥合 reality gap,然后相应地调整仿真 policy 的动作。该模型具有高度的数据效率(data-efficient)和跨不同全手交互分布的泛化能力,这得益于将动力学按关节分解(factorizing dynamics across joints)、将系统级影响压缩为低维变量(low-dimensional variables)、以及从每个关节自身的动态特征(dynamic profile)学习其演化过程,从而隐式捕获这些净效应(net effects)。我们将此与一种完全自主的数据收集策略相结合,该策略以最少的人工干预收集多样化的真实世界交互数据。我们的完整流水线展示了前所未有的通用性:单一 policy 成功旋转具有复杂形状(例如动物模型)、高长宽比(最高5.33)和小尺寸的挑战性物体,同时处理多样的手腕朝向和旋转轴。全面的真实世界评估和远程操控(teleoperation)应用验证了我们方法的有效性和鲁棒性。

*图 1: 我们提出了 [DexNDM](https://meowuu7.github.io/DexNDM/), 一种 sim-to-real 方法,使得真实世界中前所未有的手内旋转成为可能。我们掌握了广泛的物体分布,包括 (A) 挑战性几何形状和 (B) 复杂造型,跨越 (C) 丰富的手腕朝向。(D) 远程操控应用。视频请见[项目网站](https://meowuu7.github.io/DexNDM/)。*

## 1 引言

推进灵巧操作(dexterous manipulation)对于实现高度能干的具身智能(embodied intelligence)至关重要。该领域中一项基础且具有挑战性的技能是手内物体旋转(in-hand object rotation)。长期以来的目标——也是我们在本工作中追求的——是开发一种通用 policy,能够在真实世界中跨多种手腕朝向和旋转轴旋转广泛分布的物体。

尽管取得了近期进展,学术界尚未达到这种通用性水平。现有方法 (Chen et al., 2022; Yang et al., 2024; Qi et al., 2023; Wang et al., 2024; Zhao et al., 2025; Yuan et al., 2023) 通常受限于特定场景:一些假设手持续朝上,另一些只处理有限的简单、正常尺寸物体,许多依赖昂贵的定制硬件和精密的触觉传感。虽然某些方法 (Yang et al., 2024) 在某一维度(如旋转轴)上展示了通用性,但在其他维度(如物体复杂度)上受到限制。据我们所知,此前没有工作展示过在多样手腕朝向和旋转轴下,对广泛物体范围——包括复杂形状、高长宽比和不同尺寸——的稳健空中旋转(in-the-air rotation)。

实现这一目标的主要障碍是巨大的"sim-to-real gap",这源于对复杂交互动力学建模的困难,该动力学以丰富的、快速变化的、载荷依赖的接触为特征。这同时削弱了基于模型(model-based) (Pang & Tedrake, 2021; Pang et al., 2023; Suh et al., 2025) 和无模型(model-free) (Qi et al., 2023; Chen et al., 2022; Yang et al., 2024) 的方法。一种有前景的 sim-to-real 迁移思路是从真实世界数据中学习 neural dynamics model(神经动力学模型) (He et al., 2025; bin Shi et al., 2024)。这种方法已在运动控制(locomotion)中被证明有效,其中相对容易的故障恢复和可直接观测的状态允许高效收集与任务分布相关的数据。

然而,这种成功不容易迁移到通用操作任务中,因为数据量和分布相关性(distributional relevance)的要求造成了不可避免的冲突。通用性的需求要求大量数据来覆盖多样的物体。然而,确保这些数据在分布上与任务相关有时是不可能的,且在操作上要复杂得多:次优的可部署 policy 无法操纵困难物体(例如长物体);灾难性失败(即物体掉落)需要频繁的人工干预来重置;严重的手部遮挡使得准确跟踪多样物体的状态变得复杂。这种冲突为该领域造成了关键瓶颈。

为克服这些挑战,我们提出了一个框架,通过从根本上重新思考模型和数据来打破这一不可避免的冲突。我们的核心见解是通过一个更具泛化能力的动力学模型来分解学习问题,这反过来又能实现更具可扩展性的数据收集策略。首先,我们不是将高维手-物系统作为整体来建模 (bin Shi et al., 2024),而是学习一个 joint-wise neural dynamics model。该模型对系统进行分解,仅使用每个关节自身的本体感觉历史(proprioceptive history)来预测其演化,这推广了 RMA (Kumar et al., 2021) 的思想。这种设计直接应对了上述挑战:它天然免疫于物体状态估计的困难;并且通过将系统级影响——自身驱动、关节间耦合和物体载荷——蒸馏为低维的、任务充分的(task-sufficient)净效应,同时减少了无关变异(nuisance variability),使模型在不牺牲表达能力(expressivity)的情况下具有高度的样本效率(sample-efficient)和泛化能力,实验结果也验证了这一点。这种增强的泛化能力是解锁我们第二项创新的关键:一种完全自主的数据收集策略。通过以与任务无关的方式向手施加随机载荷,我们在消除灾难性失败和人工重置需求的同时收集数据。这使我们能够从廉价且可扩展的数据中学习到在目标任务上泛化良好的动力学模型,然后用它来训练一个 residual policy(残差策略),将仿真训练的 base policy 适配到真实世界,实现广泛的通用性。

我们通过一个 specialist-to-generalist(专家到通才)流水线获得 base policy:先在涵盖长宽比和几何复杂度的数据上训练类别特定的专家 policy,然后将它们蒸馏为统一的 policy。

我们在仿真和真实世界中都验证了我们的方法。在仿真中,我们的 base policy 泛化到新颖的复杂形状,以 37%-81% 的优势超越强基线。

在真实世界中,我们的 sim-to-real 方法显著且持续地提升了旋转性能,在广泛的物体分布上实现了跨多样手腕朝向和旋转轴的灵活旋转——包括复杂几何形状(例如动物模型)、最高5.33的长宽比、以及0.31-1.68的物体与手尺寸比(图 1;视频请见我们的[项目网站](https://meowuu7.github.io/DexNDM/))。

值得注意的是,在具有挑战性的手掌朝下配置中,据我们所知,我们是第一个在空中将长物体(10-16cm)绕其长轴旋转约一整圈的工作。

与在大型定制 D'Claw 手上的 Visual Dexterity (Chen et al., 2022) 相比,我们更小的 LEAP 手达到了相当或更好的性能,并在其难以处理的形状上取得成功(例如大象、兔子、茶壶)。

我们还泛化到比先前多手腕 SOTA (Yang et al., 2024) 更广泛、更具挑战性的物体分布。

此外,我们展示了由通用旋转 policy 实现的应用:构建一个远程操控系统来执行复杂的灵巧任务,如工具使用(例如螺丝刀、刀)和组装 (Heo et al., 2023)。

系统性的消融研究(ablation study)验证了动力学模型和数据收集策略中关键设计选择的重要作用。

我们的主要贡献有四个方面:

- 一个用于灵巧手内旋转的新颖 sim-to-real 框架,建立在 joint-wise neural dynamics model 和自主数据收集之上,以应对学习复杂交互动力学和获取真实世界交互数据的核心挑战。

- 一个手内物体旋转 policy,在旋转挑战性物体(高长宽比、复杂形状、小尺寸)和困难的手腕朝向下实现了前所未有的通用性。

- 从理论和实证两个角度对 joint-wise neural dynamics model 的原理、优势和有效范围进行了深入分析。

- 在远程操控复杂灵巧任务方面的实际应用展示。

## 2 相关工作

*图 2: 从真实世界数据学习用于控制。(A) 从真实世界数据学习全身动力学模型(whole-body dynamics model),用于 policy 调优或基于模型的控制。(B) 学习 residual action model(残差动作模型)来微调 base policy。(C) 学习 joint-wise dynamics 和 residual policy 来适配 base policy。*

我们的工作与两个研究方向广泛相关:手内物体旋转和 sim-to-real 策略。

手内旋转是一项重要但具有挑战性的机器人任务。尽管取得了进展,先前的方法仍然 (i) 假设手朝上 (Qi et al., 2022; Wang et al., 2024; Yuan et al., 2023; Zhao et al., 2025),(ii) 只处理正常尺寸、几何多样性有限的物体 (Qi et al., 2023; Rostel et al., 2025; Pitz et al., 2024a; 2024b; Yang et al., 2024),或 (iii) 依赖昂贵的硬件和精密的触觉传感 (Yang et al., 2024; Wang et al., 2024; Qi et al., 2023)。AnyRotate (Yang et al., 2024) 实现了旋转轴和手腕的通用性,但在真实世界中仅限于正常尺寸的规则物体。Visual Dexterity (Chen et al., 2022) 在空中旋转复杂形状,但在小物体或高长宽比物体上的性能未经验证。

我们的目标是在多样手腕朝向和旋转轴下,实现旋转挑战性(例如长、小)和复杂物体的通用性。

实现这一目标的核心障碍是 sim-to-real gap:不匹配的参数、模型差异和未建模的效应使仿真训练 policy 的迁移偏离轨道。现有方法包括:(1) Domain Randomization (DR,域随机化),扩大训练分布 (Loquercio et al., 2019; Peng et al., 2017; Tan et al., 2018; Yu et al., 2019; Mozifian et al., 2019; Siekmann et al., 2020);(2) System Identification (SysID,系统辨识),从真实数据拟合仿真器参数 (An et al., 1985; Mayeda et al., 1988; Lee et al., 2023; Sobanbabu et al., 2025);(3) 在线自适应策略(online adaptive policies) (Kumar et al., 2021; Qi et al., 2022);以及 (4) 真实动力学的神经建模(neural modeling of real dynamics) 来引导迁移 (He et al., 2025; Fey et al., 2025; Hwangbo et al., 2019)。DR 依赖于启发式范围;SysID 受限于其参数化;在线适应通常依赖于训练中的动力学覆盖。

学习真实动力学提供了最高上限:神经控制中的经典路线是为整个系统学习残差或完整模型用于基于模型的控制 (图 2 (A),例如 Neural Lander (Shi et al., 2018)、MB-Max (bin Shi et al., 2024))。随着任务复杂度增加,学习全局精确、物理合理且足够鲁棒的动力学以支持 policy 调优或控制器开发变得困难 (Shi, 2025)。

因此,在 sim-to-real RL 中提出的另一趋势方法(例如 UAN (Fey et al., 2025) 和 ASAP (He et al., 2025)) 学习仿真-真实之间的 delta actions(动作差异),并基于此微调 policy 以弥合动力学差距(图 2 (B))。

成功取决于收集足够的与任务分布相关或能提供全面覆盖的真实世界数据——这在运动控制和静态接触任务中是次要问题,但在灵巧操作中是主要瓶颈。我们通过一个可泛化的 joint-wise neural dynamics model 来解决这一问题,该模型放宽了对训练数据分布的要求,随后通过 residual policy 弥合 reality gap(图 2 (C))。

## 3 方法

*图 3: 方法概览。(A) 通过 RL 训练物体类别特定的旋转专家 policy。(B) 通过 BC(行为克隆)将它们蒸馏为单一通才 policy。(C-E) 神经 sim-to-real:自主收集带随机载荷的真实世界 transition (C),学习 joint-wise neural dynamics model (D),训练 residual 以弥合 reality gap (E)。部署由 residual (E) 增强的 base generalist (B)。*

我们的目标是一个通才 policy,能够在真实世界中在各种条件下旋转广泛的物体。我们采用 model-free RL(无模型强化学习)方法。关键挑战是接触密集灵巧操作中显著的 sim-to-real 动力学差距以及对广泛物体泛化的需求。

我们通过两个设计来解决这些问题:(1) 一种 specialist-to-generalist 方法,首先在精心策划的物体类别上训练类别特定的 oracle policy(第 3.1 节),然后将它们蒸馏为通才 policy(第 3.2 节);以及 (2) 一种以表达力强、数据高效、可泛化的 joint-wise dynamics model 为核心的神经 sim-to-real 策略,配合自主数据收集和 residual policy,将 base policy 适配以弥合 sim-to-real gap(第 3.3 节)。工作流程如图 3 所示。

### 3.1 多手腕朝向下的多轴手内物体旋转

我们将手内旋转建模为有限时间步的部分可观测马尔可夫决策过程(Partially Observable Markov Decision Process, POMDP),$M=(\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{P},\mathcal{R})$,包含状态、动作和观测空间 $(\mathcal{S},\mathcal{A},\mathcal{O})$,转移动力学 $\mathcal{P}$ 和 reward $\mathcal{R}$。我们用 RL 训练神经 policy $\pi:\mathcal{O}\to\mathcal{A}$ 来最大化时间步 $N$ 上的期望累积回报:
$\pi^{*}=\arg\max_{\pi}\mathbb{E}_{\tau\sim p_{\pi}(\tau)}[\sum_{t=1}^{N}r(\mathbf{s}_{t},\mathbf{a}_{t})].$

观测与动作。
在时间步 $t$,policy 接收 $\mathbf{o}_{t}$:包含本体感觉(proprioception)的短历史、指尖和物体状态、每关节/每手指的力测量、二值接触信号、手腕朝向以及目标旋转轴(第 A.1 节)。Policy 输出相对目标位置的分布。我们从中采样 $\Delta\mathbf{a}_{t}\sim\pi(\mathbf{o}_{t})$ 并以 $\alpha=1/24$ 更新关节目标 $\mathbf{a}_{t}=\mathbf{a}_{t-1}+\alpha\,\Delta\mathbf{a}_{t}$。$\mathbf{a}_{t}$ 通过 PD 控制器转换为力矩并在机器人上执行。

Reward 函数。
Reward 由三个加权分量组成 $r=\alpha_{\text{rot}}r_{\text{rot}}+\alpha_{\text{goal}}r_{\text{goal}}+\alpha_{\text{penalty}}r_{\text{penalty}}$,其中 $r_{\text{rot}}$ 和 $r_{\text{penalty}}$ 沿用 RotateIt (Qi et al., 2023)。
旋转项 $r_{\text{rot}}$ 鼓励绕目标轴旋转。惩罚项 $r_{\text{penalty}}$ 抑制偏轴角速度、偏离标准手部姿态、物体线速度以及关节功/力矩。
由于这些 reward 单独无法解决困难情况(例如旋转长物体),我们添加了一个中间目标姿态 reward $r_{\text{goal}}$,引导物体到达目标旋转轴上的路标点(waypoint)。
详见第 A.1 节。

### 3.2 通过行为克隆训练通才 Policy

在获得了每个物体类别的 oracle policy(具有丰富的特权观测)之后,我们使用 Behavior Cloning (BC,行为克隆) 来训练统一的、可真实世界部署的多几何体通才 policy。
虽然 DAgger 式蒸馏在先前工作中已被证明有效,但在我们的设置中,即使单个 policy 的蒸馏也会在仿真中优化失败或在真实世界中崩溃,这与 PenSpin (Wang et al., 2024) 的发现一致。我们将此归因于任务难度高。
因此我们使用 BC:展开所有 oracle policy,只聚合成功轨迹(successful trajectories),然后通过监督学习训练通才 policy。这种方法在硬件上效果良好。
我们假设其成功源于仅模仿高质量的 oracle 行为。
通才 policy 的观测 $\mathbf{o}_{t}^{\text{gene}}$ 包含本体感觉历史 $\{(\mathbf{q}_{k},\mathbf{a}_{k-1})\}_{k=t-T+1}^{t}$、手腕朝向和旋转轴。
我们使用 $T=10$,并将 policy 实现为残差 MLP (He et al., 2015)。

### 3.3 通过 Joint-Wise Neural Dynamics 弥合 Reality Gap

虽然通才 policy 已经可以在真实世界部署,但持续存在的 sim-to-real gap——由不匹配的物理动力学和未建模效应造成——阻碍了其掌握挑战性的物体交互。我们通过一种新颖的神经 sim-to-real 策略弥合这一差距,该策略有效地学习复杂的真实世界动力学模型。

核心挑战是获取有用且足量的真实数据,使学到的动力学模型能够帮助 sim-to-real 迁移。对于灵巧操作,先前的数据获取方法 (Hwangbo et al., 2019; He et al., 2025; Fey et al., 2025; bin Shi et al., 2024) 通常不切实际。展开 base policy (He et al., 2025; bin Shi et al., 2024) 或执行波形动作 (Fey et al., 2025) 在多样和复杂物体上频繁失败,需要持续的人工干预,同时不完美的状态估计器引入大量噪声。这导致真实数据集小、有偏且在覆盖范围和质量上不足。我们通过重新思考模型和数据来解决这些挑战。

我们提出了一种 joint-wise neural dynamics model,通过从低维、信息压缩的(information-contractive)、任务充分的系统动力学表示中学习,极大地提高了样本效率和泛化能力,同时保持了表达能力。这使得一种自主数据收集策略成为可能,该策略通过施加随机载荷收集多样的大规模真实世界数据,消除了对特定任务展开和人工重置的需求。

Joint-Wise Neural Dynamics。
在不依赖噪声大且有限的物体状态估计的情况下建模系统动力学,一种方式是学习"全手"(whole-hand)神经模型。该模型从长度为 $W$ 的状态-动作历史预测手的下一状态,$\mathbf{q}^{t+1}=f_{\theta}({H}_{t})$,其中 ${H}_{t}=\{\mathbf{q}_{j},\mathbf{a}_{j}\}_{j=t-W+1}^{t}$,从而隐式捕获整个系统动力学,包括来自物体的外力 (Qi et al., 2022)。然而,这种方法仍然需要大量数据,继承了上述其他数据获取挑战。

我们的解决方案是分解问题。我们引入 joint-wise neural dynamics,将每个关节 $i$ 的动力学建模为 $\mathbf{H}_{t}^{\text{eff}}\ddot{\mathbf{q}}_{t}^{i}+\mathbf{G}_{t}^{\text{eff}}=\tau_{t}^{i}$,其中 $\mathbf{H}_{t}^{\text{eff}},\mathbf{G}_{t}^{\text{eff}}\in\mathbb{R}$ 是低维的有效项(effective terms),将高维的系统级影响(如关节间耦合、驱动和物体引起的效应)蒸馏其中。然后神经模型从每个关节 $i$ 自身的 $W$ 步状态-动作历史预测其下一状态:$\mathbf{q}_{t+1}^{i}=f_{\psi_{i}}(h_{t}^{i})$,其中 $h_{t}^{i}=\{\mathbf{q}_{j}^{i},\mathbf{a}_{j}^{i}\}_{j=t-W+1}^{t}$。

这种分解之所以有效,是因为它充当了信息瓶颈(information bottleneck),迫使模型丢弃虚假相关性(spurious correlations),只学习每个关节的本质动力学。这个投影后的历史具有充分信息性(sufficiently informative),包含足够的信息来准确预测关节的下一状态(第 4.2、A.3 节)。同时,它也足够鲁棒简洁(robustly simple),因为其维度太低,不可能重构原始的高维系统级影响,从而避免了对无关复杂性的建模(第 A.4 节)。

直接结果是一个高度样本高效、广泛泛化且保持表达能力的模型(第 4.2 节)。我们现在提供理论分析来形式化这种简化为何带来更好的泛化。

理论依据:通过信息压缩实现泛化。我们将全手模型写为 $f_{\theta}=\{f_{\theta}^{i}\}$,其中 $\mathbf{q}_{t+1}^{i}=f_{\theta}^{i}({H}_{t})$,
将 joint-wise 模型写为 $\mathbf{q}_{t+1}^{i}=f_{\psi_{i}}^{i}({h}_{t}^{i})$。
令 $\mathcal{P}$ 为 $({H}_{t},\mathbf{q}_{t+1}^{i})$ 的目标分布(例如由我们感兴趣的任务形成);考虑一个不同的分布 $\mathcal{Q}$
和投影 $g:({H}_{t},\mathbf{q}_{t+1}^{i})\mapsto({h}_{t}^{i},\mathbf{q}_{t+1}^{i})$,即 $g:\mathbb{R}^{2Wd}\times\mathbb{R}\to\mathbb{R}^{2W}\times\mathbb{R}$。我们比较这两种模型,即 $f_{\theta}^{i}$ 和 $f_{\psi_{i}}^{i}$,在目标分布 $\mathcal{P}$ 上对关节 $i$ 的预测误差,
以支持泛化优势:

在我们设置中典型的假设下,
$\forall 1\leq i\leq d$,在 $g(\mathcal{Q})$ 上训练的 joint-wise 模型 $f_{\psi_{i}}^{i}$ 泛化到 $g(\mathcal{P})$ 的能力优于在 $\mathcal{Q}$ 上训练的全手模型 $f_{\theta}^{i}$ 泛化到 $\mathcal{P}$ 的能力。

我们首先证明,在我们设置中通常满足的温和假设下,投影 $g$ 收缩分布偏移(contracts distribution shift): $\mathrm{KL}(g(\mathcal{P})\|g(\mathcal{Q})) < \mathrm{KL}(\mathcal{P}\|\mathcal{Q})$ (定理 3.1,证明推导至 A.2):

**定理 3.1 (KL 散度的数据处理不等式(严格形式))** 令 $\mathcal{P}$ 和 $\mathcal{Q}$ 为 $\mathbb{R}^{n}\times\mathbb{R}$ 上的概率分布,关于公共基础测度具有密度 $P$ 和 $Q$。令 $g:X\in\mathbb{R}^{n}\times\mathbb{R}\to Y\in\mathbb{R}^{m}\times\mathbb{R}$ 可测,$m\leq n$,用 $g(\mathcal{P})$ 和 $g(\mathcal{Q})$ 表示前推分布(pushforward distributions)。则
$\mathrm{KL}(\mathcal{P}\,\|\,\mathcal{Q})\;\geq\;\mathrm{KL}\bigl(g(\mathcal{P})\,\|\,g(\mathcal{Q})\bigr).$
此外,如果 $g$ 以合并 $\mathcal{P}$ 和 $\mathcal{Q}$ 具有不同相对结构的点的方式非单射(non-injective),则不等式严格成立。更具体地说,这意味着如果存在 $y_{0}\in\mathbb{R}^{m}$,$P(Y=y_{0})>0$,$P(X|Y=y_{0})\neq Q(X|Y=y_{0})$,则
$\mathrm{KL}(\mathcal{P}\,\|\,\mathcal{Q})\;>\;\mathrm{KL}\bigl(g(\mathcal{P})\,\|\,g(\mathcal{Q})\bigr).$

散度的收缩意味着更紧的泛化保证(定理 3.2,证明见 A.2):

**定理 3.2 (泛化差距收缩)** 令 $(X,Y)\in\mathbb{R}^{n}\times\mathbb{R}$ 且 $g(X,Y)=(g_{X}(X),Y)$,其中 $g_{X}:\mathbb{R}^{n}\to\mathbb{R}^{m}$,$m < n$。令 $\mathcal{P}$, $\mathcal{Q}$ 为 $(X,Y)$ 上的分布,满足协变量偏移条件,即 $\mathcal{P}(Y|X) = \mathcal{Q}(Y|X)$。令 $L$ 为有界损失函数。若 $\mathrm{KL}(g(\mathcal{P})\|g(\mathcal{Q})) < \mathrm{KL}(\mathcal{P}\|\mathcal{Q})$,则对于函数 $f_1: X \in \mathbb{R}^n \to Y \in \mathbb{R}$ 和 $f_2: g_X(X) \in \mathbb{R}^m \to Y \in \mathbb{R}$,有泛化差距的上界更紧。

假设 $f_{2}\circ g_{X}$ 具有足够的表达能力且从 $\mathcal{Q}$ 到 $\mathcal{P}$ 存在较大的域偏移(domain shift,这在我们的设置中很典型),$f_{2}\circ g_{X}$ 在目标域 $\mathcal{P}$ 上的预测误差低于 $f_{1}$,从而建立了声明 3.1。详见第 A.2 节。

在实践中,我们在仿真数据上预训练模型作为初始化。

*图 4: 状态-动作历史分布。*

自主数据收集。
我们模型从分布不同的数据中泛化的能力激发了我们的第二项创新:一种低成本的自主数据收集策略。这种方法我们称之为"Chaos Box"(混沌盒子)(图 3(C)),体现了四项原则:(i) policy 感知(policy-awareness,粗略对齐分布),(ii) 带物体载荷的交互,(iii) 广泛覆盖,(iv) 可扩展性。实现很简单:将机械手放入装满软球的容器中。然后我们开环回放(open-loop replay)仿真 base policy 的动作,这提供了粗略的分布先验 (i)。手与球的交互施加丰富的随机载荷 (ii-iii)。以0.5的概率,我们向每个动作添加高斯噪声($\sigma{=}0.01$)以扩大覆盖范围 (iii)。整个过程完全自主、硬件安全,无需人工重置 (iv)。图 4 支持了我们的模型和数据设计:单个关节的输入/输出历史覆盖了与任务相关的分布,而全手的历史则不能。

通过 Residual Policy 弥合动力学差距。
利用学到的动力学 $f_{\psi}$,我们训练一个 residual policy $\pi^{\mathrm{res}}$,补偿 base policy 的动作以弥合动力学差距(图 3(E))。
具体来说,给定 base policy 的观测 $\mathbf{o}_{t}^{\mathrm{gene}}$ 和 base action $\mathbf{a}_{t}$,$\pi^{\mathrm{res}}$ 输出修正 $\mathbf{a}_{t}^{\mathrm{res}}$,为了匹配仿真器的下一状态 $\mathbf{q}_{t+1}$,我们求解 ${\pi^{\mathrm{res}}}^{*}=\arg\min_{\pi^{\mathrm{res}}}\mathbb{E}_{\tau\sim p_{\pi^{*}}(\tau)}\sum_{t=1}^{N-1}\left\|\mathbf{q}_{t+1}-f_{\psi}\!\left(\{\mathbf{q}_{j},\ \mathbf{a}_{j}+\pi^{\mathrm{res}}(\mathbf{o}_{j}^{\mathrm{gene}},\mathbf{a}_{j})\}_{j=t-W+1}^{t}\right)\right\|.$
我们通过在训练 base policy 所用的轨迹数据集上以监督方式训练 $\pi^{\mathrm{res}}$ 来求解。
部署时,我们执行 $\mathbf{a}_{t}+\mathbf{a}_{t}^{\mathrm{res}}$。
关于 residual policy 与直接微调的讨论见第 B.4 节。

## 4 实验

我们在仿真和真实世界中对我们的方法进行了大量评估,并与强基线进行了比较(第 4.1 节)。
在仿真中,我们的通才 policy 对未见过的几何形状在多手腕姿态、多轴旋转方面实现了泛化。
在硬件上,它使用 LEAP 手 (Shaw et al., 2023) 在困难的手腕姿态下对困难物体(包括长物体 13.5-20cm、小物体 2-3cm 和复杂的动物形状)实现了前所未有的空中旋转(第 4.2 节)。
我们还展示了一个远程操控设置,将 policy 与 VR 配合执行复杂的灵巧任务(第 4.2 节),如工具使用和组装。

### 4.1 实验设置

训练和评估协议。
我们创建了一个跨长宽比、尺寸和复杂度的物体数据集,并随机化物理属性用于训练。
我们将物体分为五类,并在 Isaac Gym (Makoviychuk et al., 2021) 中使用 PPO (Schulman et al., 2017) 为每类训练一个 oracle policy。
我们使用 ContactDB (Brahmbhatt et al., 2019) 的物体作为仿真中的测试集来评估对形状变化的泛化能力。我们在随机化的手腕朝向和四组旋转轴上评估旋转:$\pm x$、$\pm y$、$\pm z$ 和包含26个轴的通用轴集。

我们在真实世界中评估三组物体(图 5):(1) 规则物体(包括高长宽比长方体);(2) 小物体;(3) 正常尺寸的不规则物体。紫色显示的物体和所有小物体是未见过的。

我们在三组主轴集和一个立方体对角线集上评估:(1,1,1)、(1,0,1)、(1,1,0)、(0,1,1)。结果在物体间取平均,以三次独立评估的均值 $\pm$ 标准差报告。
详见第 C 节。

*图 5: 真实实验物体。*

基线方法。
我们与手内旋转/重定向基线——AnyRotate (Yang et al., 2024) 和 Visual Dexterity (VD) (Chen et al., 2022)——以及 sim-to-real 方法 UAN (Fey et al., 2025) 和 ASAP (He et al., 2025) 进行比较。AnyRotate 的代码不可用且依赖专用触觉传感,因此我们使用自己在仿真中的重新实现;在硬件上,我们在其可复制的物体上评估并与其报告的性能比较。与 VD 的直接比较不切实际:将其 D'Claw 代码适配到 LEAP 在仿真中无法正常工作,因此我们与其定性结果比较([链接](https://taochenshh.github.io/projects/visual-dexterity))。UAN 和 ASAP 为手臂/腿式机器人设计且不建模物体,通过在无物体 transition 上训练补偿器来适配;使它们感知物体是非平凡的(见第 D 节)。

评估指标。
我们使用 RotateIt 指标 (Qi et al., 2023),加上面向目标的成功率(goal-oriented success):Time-to-Fall (TTF,坠落时间)——终止前的持续时间;在仿真中,回合上限为400步(20s),TTF 归一化到20s,而在真实世界中报告原始时间;Rotation Reward (RotR,旋转奖励)——$\bm{\omega}\cdot\mathbf{k}$ 的回合总和(仅仿真);Rotation Penalty (RotP,旋转惩罚)——$\bm{\omega}\times\mathbf{k}$ 的每步平均值(仅仿真);Radians Rotated (Rot,旋转弧度)——真实世界中旋转的总弧度;
Goal-Oriented Success (GO Succ.,面向目标成功率)沿用 Visual Dexterity:采样一个目标姿态;将目标轴设为相对旋转轴;如果朝向在目标的 $0.1\pi$ 以内则计为成功(仅仿真)。

### 4.2 手内旋转结果与分析

仿真结果。
我们的 policy 泛化到未见过的物体,并优于我们重新实现的基线(表 1)。
在所有设置中,沿重力方向($\pm z$ 轴)旋转是最简单的任务,这与先前工作的观察一致 (Qi et al., 2023; Yang et al., 2024)。

*表 1: 仿真中的泛化测试。在未见测试物体集上,沿每个轴在手腕朝向随机化条件下的旋转性能比较。*

*表 2: 与 AnyRotate 的比较。在 AnyRotate 提出的两种测试设置(表 12、13)下,在可复制物体上的旋转角度(Rot (radian))和坠落时间(TTF (s))的比较。*

*表 3: 与 Visual Dexterity 的生存角度($\lfloor\text{radian}/0.5\pi\rfloor$)比较,(从视频中)粗略衡量物体在掉落前能旋转多少个90度。下标 * 表示在有支撑桌辅助下旋转物体取得的性能。*

*图 6: 与 Whole-Hand Neural Dynamics 在模型表达能力、样本效率和可迁移性方面的比较。(A, A-0) 在高数据(3.1M)和低数据(7.5k)条件下的域内(in-domain)和分布外(out-of-distribution)性能。(B) 样本效率。(C) 从不同训练分布的可迁移性。*

*表 4: 真实世界多轴旋转。在手掌朝下手腕朝向下,沿每个轴的旋转角度(Rot (radian))和坠落时间(TTF (s))比较。指标首先在每次试验中对所有物体取平均。然后报告三次独立试验的均值 $\pm$ 标准差。*

*表 5: 真实世界多手腕朝向旋转。在六种代表性手部朝向下沿 $z$ 方向的旋转角度(Rot (radian))和坠落时间(TTF (s))比较。*

真实世界结果。
我们的 sim-to-real 方法持续改善真实世界性能,policy 展示了前所未有的灵巧性,在具有挑战性的手腕朝向下在空中旋转高长宽比几何体、小物体和复杂形状(表 4(多轴,手掌朝下)、表 5(多手腕姿态,$z$ 轴旋转);图 1;图 20、物体展示(图 19)(在附录中);[视频](https://meowuu7.github.io/DexNDM/))。

与 AnyRotate 发现"Thumb Up/Down"(拇指朝上/朝下)最困难不同,我们观察到"Base Up/Down"(基座朝上/朝下)更难,这可能是由于 Allegro 和 LEAP 之间不同的驱动器性能。

与 AnyRotate 的比较。
我们在 AnyRotate 套件中四个可复制物体上进行评估——"Tin Cylinder"、Cube、"Gum Box"和"Container"(第 C 节)——这些是他们最困难的情况(根据表 12-13),并与他们报告的真实世界结果进行比较。
表 2 显示我们的方法大幅优于 AnyRotate 且更具通用性:AnyRotate 针对中等大小、简单形状(最小5cm,最大长宽比1.67)采用保守动作,而我们的 policy 处理更小的物体(3cm)和高长宽比(最高5.3),使用精巧的手指步态(finger gaiting)。

与 Visual Dexterity 的比较。
与 Visual Dexterity (VD) 的直接比较不可行,因为任务定义不同(面向轴的连续旋转 vs. 面向目标的重定向)。为便于比较,我们引入生存旋转角度指标(survival rotation angle):物体在掉落前旋转的角度。
我们通过分析 VD 的[视频](https://taochenshh.github.io/projects/visual-dexterity)来估计其最佳性能。尽管该指标对 VD 更有利(其设置有时包含支撑桌),我们在其展示的可复制物体上达到了相当或更优的结果(表 3)。
此外,我们独特地能够操作小物体和高长宽比物体,并处理多样的手腕朝向(图 20)。

与 Whole-Hand Neural Dynamics 的比较。
我们与全手动力学模型进行比较以回答:
(Q1) 从每个关节自身历史预测其 transition(不包含全局信息)是否降低了表达能力?
(Q2) 我们的模型是否更具样本效率?
(Q3) 它是否泛化得更好?
(A1) 在310万条仿真轨迹上训练并在域内评估时,我们的模型几乎与全手模型一样具有表达能力(图 6(A, 第1列)(A-0))。
(A2) 在有限数据条件下——使用7.5k条自主收集的真实世界轨迹(图 6(A, 第3列))以及不同的真实世界数据集大小(图 6(B))——我们的模型实现了更好的域内性能,表明更高的样本效率。在数据不足的设置下优势更加明显。
(A3) 在分布外(OOD)真实世界测试集上(在"Thumb Up"手腕朝向下与任务相关的 transition),我们的模型在高数据和低数据条件下都泛化得更好;见图 6(A, 第2、4列)和图 6(B)。图 6(C) 系统地研究了各种设置下的跨域可迁移性。

总结:对于数据驱动的 neural dynamics,joint-wise 模型在数据不足或训练-测试分布偏移的设置下显著优于全手模型;在数据充足且域内评估时,性能相似,joint-wise 模型仅有轻微的表达能力损失。

与 ASAP 和 UAN 的比较。
我们实现了 UAN 和 ASAP,但它们产生的 policy 在真实世界测试中完全失败——连简单的圆柱体都无法旋转(图 27;[视频](https://meowuu7.github.io/DexNDM/))。我们将此归因于 OOD 问题:仅在无物体数据上训练的补偿器无法泛化到被操作物体引入的交互动力学。
请注意,他们的方法只能使用无物体数据或带物体状态的任务相关数据——后者难以获取且噪声大,甚至无法用于补偿器训练——且无法利用我们自主收集的带随机物体载荷的数据;见第 D 节。我们的策略对真实数据的不完美性更具容忍度(图 8、图 9、图 27)。

*表 6: "Sim-to-Sim" 迁移。*

"Sim-to-Sim" 比较。
我们进行了跨仿真器迁移评估(Isaac Gym 到 Genesis 和 MuJoCo)。
我们在目标仿真器中收集带物体载荷的旋转数据用于训练。
表 6 显示我们的方法持续优于先前工作,这归功于动力学建模的设计、更高的数据效率以及实际选择(例如在源仿真器中预训练)。我们发现 UAN 优于 ASAP,可能是因为其基于历史的设计更好地捕获了物体效应。详见第 C 节。

*图 7: 应用。我们的旋转 policy 使远程操控系统能够执行复杂的长时域操作任务。视频和更多结果请见我们的[项目网站](https://meowuu7.github.io/DexNDM/)。*

应用。
我们展示了旋转 policy 的一个应用:用于灵巧任务的远程操控系统(使用 Meta Quest 3 构建,详见第 C 节)。我们展示了其在执行长时域和复杂灵巧操作任务方面的强大能力(图 7、[视频](https://meowuu7.github.io/DexNDM/))。

## 5 消融研究

*图 8: 动力学模型的消融研究。(A) 不同模型消融版本的泛化误差(越低越好)。(B) 对应的真实世界任务性能。*

*图 9: 数据收集策略分析。(A) 不同收集方法的时间效率。(B) 在等量数据集上的模型性能。(C) 性能随数据集大小和数据收集迭代次数的缩放,包括用于外推的幂律拟合。*

我们进行消融实验来验证方法的关键设计选择。
真实世界实验在手掌固定朝下的条件下进行,评估 z 轴旋转;
数据在相同手腕姿态下收集。动力学模型在 OOD 测试设置中评估。详见第 C 节。

Joint-Wise Neural Dynamics Model 的设计选择。
我们消融了五个设计选择:(i) joint-wise vs. finger-wise(每根手指从自身历史预测)和 whole-hand 建模;(ii) 仿真预训练;(iii) 在真实世界数据收集期间向回放动作注入噪声;(iv) 使用带物体载荷而非无载荷的无物体手收集;(v) 回放 policy 展开而非基础波形 (Fey et al., 2025)。如图 8 所示,这些选择持续改善了学到的动力学泛化能力和真实世界性能。

真实世界数据收集策略。
我们将自主数据收集与三种基线进行比较——带视觉物体状态的任务感知收集、不带物体状态的任务感知收集、以及无物体手运动——评估局限性、效率和模型性能(图 9)。任务感知流水线缓慢且需要大量干预:估计物体姿态极其缓慢(平均约200s),需要持续的人工监督,产生噪声姿态和复杂设置,且在小物体、遮挡物体或轴对称物体上失败;不使用视觉时仍需干预,速度仍慢(42.86s),且产生低多样性、低覆盖的数据(数据受限于 policy 的能力)。相比之下,我们的方法完全自动化,通过持续变化手部载荷,收集跨越广泛外部影响范围的多样数据。图 9(B) 展示了由此带来的性能提升:更广泛的覆盖改善了预测,且 joint-wise 模型对训练分布偏移最为鲁棒,而其他变体倾向于过拟合到源数据。

随真实世界数据量和收集迭代次数的缩放。
如图 9 所示,我们的性能随更多真实世界数据而提升。然而,迭代数据收集——旨在对齐真实世界和仿真 transition 分布以获得更好的 policy 更新——仅产生适度的收益。我们假设这是因为动力学模型已经泛化良好,且向回放动作添加噪声提供了广泛覆盖,降低了对这种分布偏移的敏感性。相比之下,全手模型从额外数据中获益甚少,尤其是在自主收集下,可能是由于其更高的维度和自主数据与旋转任务 transition 之间的分布不匹配。

一个简单的外推表明,要匹配我们的4,000轨迹结果,需要750万条任务感知轨迹(417k小时;52k个8小时工作日),这是不切实际的。虽然这是近似的,但它突出了我们方法的优越性。

## 6 结论与局限性

我们提出了一个以 joint-wise neural dynamics model 和自主数据收集为核心的神经 sim-to-real 框架。
这使得在旋转挑战性物体方面实现了前所未有的灵巧性。
主要局限性在于模型的上限受限于部分观测(partial observations);联合建模手-物 transition 以利用更丰富的信号,以及集成触觉,是有价值的未来方向。

## 致谢

作者感谢 Ziqing Chen、Chi Chu、Chao Chen 对手稿早期版本的宝贵反馈,以及 Qianwei Han、Bowen Liu 对演示视频初始版本的建设性建议。

## 参考文献

- An et al. (1985)

Chae H. An, Christopher G. Atkeson, and John M. Hollerbach.

Estimation of inertial parameters of rigid body links of manipulators.

1985 24th IEEE Conference on Decision and Control*, pp. 990-995, 1985.

- bin Shi et al. (2024)

Hao bin Shi, Tingguang Li, Qing Zhu, Jiapeng Sheng, Lei Han, and Max Q.-H. Meng.

An efficient model-based approach on learning agile motor skills without reinforcement.

*2024 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 5724-5730, 2024.

URL [https://api.semanticscholar.org/CorpusID:268248331](https://api.semanticscholar.org/CorpusID:268248331).

- Brahmbhatt et al. (2019)

Samarth Brahmbhatt, Cusuh Ham, Charles C. Kemp, and James Hays.

Contactdb: Analyzing and predicting grasp contact via thermal imaging.

*2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 8701-8711, 2019.

URL [https://api.semanticscholar.org/CorpusID:118643835](https://api.semanticscholar.org/CorpusID:118643835).

- Chen et al. (2022)

Tao Chen, Megha H. Tippur, Siyang Wu, Vikash Kumar, Edward H. Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand reorientation of novel and complex object shapes.

*Science Robotics*, 8, 2022.

URL [https://api.semanticscholar.org/CorpusID:253734517](https://api.semanticscholar.org/CorpusID:253734517).

- Cheng & Schwing (2022)

Ho Kei Cheng and Alexander G. Schwing.

Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model.

In *European Conference on Computer Vision*, 2022.

URL [https://api.semanticscholar.org/CorpusID:250526250](https://api.semanticscholar.org/CorpusID:250526250).

- Cheng et al. (2024)

Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, and Xiaolong Wang.

Open-television: Teleoperation with immersive active visual feedback.

In *Conference on Robot Learning*, 2024.

URL [https://api.semanticscholar.org/CorpusID:270869903](https://api.semanticscholar.org/CorpusID:270869903).

- Craig (2009)

John J Craig.

*Introduction to robotics: mechanics and control, 3/E*.

Pearson Education India, 2009.

- Deisenroth & Rasmussen (2011)

Marc Peter Deisenroth and Carl Edward Rasmussen.

Pilco: A model-based and data-efficient approach to policy search.

In *International Conference on Machine Learning*, 2011.

- Ding et al. (2024)

Runyu Ding, Yuzhe Qin, Jiyue Zhu, Chengzhe Jia, Shiqi Yang, Ruihan Yang, Xiaojuan Qi, and Xiaolong Wang.

Bunny-visionpro: Real-time bimanual dexterous teleoperation for imitation learning.

2024.

URL [https://arxiv.org/abs/2407.03162](https://arxiv.org/abs/2407.03162).

- Fey et al. (2025)

Nolan Fey, G. Margolis, Martin Peticco, and Pulkit Agrawal.

Bridging the sim-to-real gap for athletic loco-manipulation.

*ArXiv*, abs/2502.10894, 2025.

URL [https://api.semanticscholar.org/CorpusID:276408331](https://api.semanticscholar.org/CorpusID:276408331).

- Guillemin & Pollack (2010)

Victor Guillemin and Alan Pollack.

*Differential topology*, volume 370.

American Mathematical Soc., 2010.

- He et al. (2015)

Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun.

Deep residual learning for image recognition.

*2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2015.

URL [https://api.semanticscholar.org/CorpusID:206594692](https://api.semanticscholar.org/CorpusID:206594692).

- He et al. (2025)

Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, Zeji Yi, Guannan Qu, Kris Kitani, Jessica Hodgins, "Jim" Fan, Yuke Zhu, Changliu Liu, and Guanya Shi.

Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills.

*ArXiv*, abs/2502.01143, 2025.

URL [https://api.semanticscholar.org/CorpusID:276095101](https://api.semanticscholar.org/CorpusID:276095101).

- Heo et al. (2023)

Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J. Lim.

Furniturebench: Reproducible real-world benchmark for long-horizon complex manipulation.

In *Robotics: Science and Systems*, 2023.

- Hwangbo et al. (2019)

Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter.

Learning agile and dynamic motor skills for legged robots.

*Science Robotics*, 4, 2019.

URL [https://api.semanticscholar.org/CorpusID:58031572](https://api.semanticscholar.org/CorpusID:58031572).

- Kumar et al. (2021)

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik.

Rma: Rapid motor adaptation for legged robots.

*ArXiv*, abs/2107.04034, 2021.

URL [https://api.semanticscholar.org/CorpusID:235650916](https://api.semanticscholar.org/CorpusID:235650916).

- Lee et al. (2023)

Taeyoon Lee, Jaewoon Kwon, Patrick M. Wensing, and Frank C. Park.

Robot model identification and learning: A modern perspective.

*Annu. Rev. Control. Robotics Auton. Syst.*, 7, 2023.

- Loquercio et al. (2019)

Antonio Loquercio, Elia Kaufmann, Rene Ranftl, Alexey Dosovitskiy, Vladlen Koltun, and Davide Scaramuzza.

Deep drone racing: From simulation to reality with domain randomization.

*IEEE Transactions on Robotics*, 36:1-14, 2019.

URL [https://api.semanticscholar.org/CorpusID:162183971](https://api.semanticscholar.org/CorpusID:162183971).

- Makoviychuk et al. (2021)

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al.

Isaac gym: High performance gpu-based physics simulation for robot learning.

*arXiv preprint arXiv:2108.10470*, 2021.

- Mayeda et al. (1988)

Hirokazu Mayeda, Koji Yoshida, and Koichi Osuka.

Base parameters of manipulator dynamic models.

*Proceedings. 1988 IEEE International Conference on Robotics and Automation*, pp. 1367-1372 vol.3, 1988.

- Mozifian et al. (2019)

Melissa Mozifian, Juan Camilo Gamboa Higuera, David Meger, and Gregory Dudek.

Learning domain randomization distributions for training robust locomotion policies.

*2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 6112-6117, 2019.

URL [https://api.semanticscholar.org/CorpusID:204185733](https://api.semanticscholar.org/CorpusID:204185733).

- Munkres (2018)

James R Munkres.

*Analysis on manifolds*.

CRC Press, 2018.

- Murray et al. (2017)

Richard M Murray, Zexiang Li, and S Shankar Sastry.

*A mathematical introduction to robotic manipulation*.

CRC press, 2017.

- O'Connell et al. (2022)

Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung.

Neural-fly enables rapid learning for agile flight in strong winds.

*Science Robotics*, 7, 2022.

URL [https://api.semanticscholar.org/CorpusID:248527107](https://api.semanticscholar.org/CorpusID:248527107).

- Pang & Tedrake (2021)

Tao Pang and Russ Tedrake.

A convex quasistatic time-stepping scheme for rigid multibody systems with contact and friction.

In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 6614-6620. IEEE, 2021.

- Pang et al. (2023)

Tao Pang, HJ Terry Suh, Lujie Yang, and Russ Tedrake.

Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models.

*IEEE Transactions on Robotics*, 2023.

- Peng et al. (2017)

Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and P. Abbeel.

Sim-to-real transfer of robotic control with dynamics randomization.

*2018 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 1-8, 2017.

URL [https://api.semanticscholar.org/CorpusID:3707478](https://api.semanticscholar.org/CorpusID:3707478).

- Pitz et al. (2024a)

Johannes Pitz, Lennart Rostel, Leon Sievers, and Berthold Bauml.

Learning time-optimal and speed-adjustable tactile in-hand manipulation.

*2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids)*, pp. 973-979, 2024a.

URL [https://api.semanticscholar.org/CorpusID:274150211](https://api.semanticscholar.org/CorpusID:274150211).

- Pitz et al. (2024b)

Johannes Pitz, Lennart Rostel, Leon Sievers, Darius Burschka, and Berthold Bauml.

Learning a shape-conditioned agent for purely tactile in-hand manipulation of various objects.

*2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 13112-13119, 2024b.

URL [https://api.semanticscholar.org/CorpusID:271516159](https://api.semanticscholar.org/CorpusID:271516159).

- Qi et al. (2022)

Haozhi Qi, Ashish Kumar, Roberto Calandra, Yinsong Ma, and Jitendra Malik.

In-hand object rotation via rapid motor adaptation.

In *Conference on Robot Learning*, 2022.

URL [https://api.semanticscholar.org/CorpusID:252781034](https://api.semanticscholar.org/CorpusID:252781034).

- Qi et al. (2023)

Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Y. Ma, Roberto Calandra, and Jitendra Malik.

General in-hand object rotation with vision and touch.

*ArXiv*, abs/2309.09979, 2023.

URL [https://api.semanticscholar.org/CorpusID:262045795](https://api.semanticscholar.org/CorpusID:262045795).

- Rostel et al. (2025)

Lennart Rostel, Dominik Winkelbauer, Johannes Pitz, Leon Sievers, and Berthold Bauml.

Composing dextrous grasping and in-hand manipulation via scoring with a reinforcement learning critic.

*ArXiv*, abs/2505.13253, 2025.

URL [https://api.semanticscholar.org/CorpusID:278768673](https://api.semanticscholar.org/CorpusID:278768673).

- Sadeghi & Levine (2016)

Fereshteh Sadeghi and Sergey Levine.

Real single-image flight without a single real image.

*ArXiv*, abs/1611.04201, 2016.

- Schulman et al. (2017)

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

*ArXiv*, abs/1707.06347, 2017.

URL [https://api.semanticscholar.org/CorpusID:28695052](https://api.semanticscholar.org/CorpusID:28695052).

- Shaw et al. (2023)

Kenneth Shaw, Ananye Agarwal, and Deepak Pathak.

Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning.

*ArXiv*, abs/2309.06440, 2023.

URL [https://api.semanticscholar.org/CorpusID:259327055](https://api.semanticscholar.org/CorpusID:259327055).

- Shi (2025)

Guanya Shi.

From sim2real 1.0 to 4.0 for humanoid whole-body control and loco-manipulation, 2025.

URL [https://opendrivelab.github.io/CVPR2025/Guangya_Shi_From_Sim2Real_1.0_to_4.0_for_Humanoid_Whole-Body_Control.pdf](https://opendrivelab.github.io/CVPR2025/Guangya_Shi_From_Sim2Real_1.0_to_4.0_for_Humanoid_Whole-Body_Control.pdf).

- Shi et al. (2018)

Guanya Shi, Xichen Shi, Michael O'Connell, Rose Yu, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung.

Neural lander: Stable drone landing control using learned dynamics.

*2019 International Conference on Robotics and Automation (ICRA)*, pp. 9784-9790, 2018.

URL [https://api.semanticscholar.org/CorpusID:53725979](https://api.semanticscholar.org/CorpusID:53725979).

- Siekmann et al. (2020)

Jonah Siekmann, Yesh Godse, Alan Fern, and Jonathan W. Hurst.

Sim-to-real learning of all common bipedal gaits via periodic reward composition.

*2021 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 7309-7315, 2020.

URL [https://api.semanticscholar.org/CorpusID:226237257](https://api.semanticscholar.org/CorpusID:226237257).

- Sobanbabu et al. (2025)

Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, and Guanya Shi.

Sampling-based system identification with active exploration for legged robot sim2real learning.

*ArXiv*, abs/2505.14266, 2025.

URL [https://api.semanticscholar.org/CorpusID:278768643](https://api.semanticscholar.org/CorpusID:278768643).

- Spong et al. (2005)

Mark W. Spong, Seth A. Hutchinson, and Mathukumalli Vidyasagar.

Robot modeling and control.

2005.

URL [https://api.semanticscholar.org/CorpusID:106678735](https://api.semanticscholar.org/CorpusID:106678735).

- Spong et al. (2020)

Mark W Spong, Seth Hutchinson, and M Vidyasagar.

Robot modeling and control.

*John Wiley &amp*, 2020.

- Suh et al. (2025)

H. J. Terry Suh, Tao Pang, Tong Zhao, and Russ Tedrake.

Dexterous contact-rich manipulation via the contact trust region.

*ArXiv*, abs/2505.02291, 2025.

URL [https://api.semanticscholar.org/CorpusID:278327864](https://api.semanticscholar.org/CorpusID:278327864).

- Taheri et al. (2020)

Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas.

Grab: A dataset of whole-body human grasping of objects.

In *Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part IV 16*, pp. 581-600. Springer, 2020.

- Tan et al. (2018)

Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, and Vincent Vanhoucke.

Sim-to-real: Learning agile locomotion for quadruped robots.

*ArXiv*, abs/1804.10332, 2018.

URL [https://api.semanticscholar.org/CorpusID:13750177](https://api.semanticscholar.org/CorpusID:13750177).

- Tedrake & the Drake Development Team (2019)

Russ Tedrake and the Drake Development Team.

Drake: Model-based design and verification for robotics, 2019.

URL [https://drake.mit.edu](https://drake.mit.edu).

- Wang et al. (2024)

Jun Wang, Ying Yuan, Haichuan Che, Haozhi Qi, Yi Ma, Jitendra Malik, and Xiaolong Wang.

Lessons from learning to spin" pens".

*arXiv preprint arXiv:2407.18902*, 2024.

- Wen et al. (2023)

Bowen Wen, Wei Yang, Jan Kautz, and Stanley T. Birchfield.

Foundationpose: Unified 6d pose estimation and tracking of novel objects.

*2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 17868-17879, 2023.

URL [https://api.semanticscholar.org/CorpusID:266191252](https://api.semanticscholar.org/CorpusID:266191252).

- Yang et al. (2024)

Max Yang, Chenghua Lu, Alex Church, Yijiong Lin, Christopher J. Ford, Haoran Li, Efi Psomopoulou, David A.W. Barton, and Nathan F. Lepora.

Anyrotate: Gravity-invariant in-hand object rotation with sim-to-real touch.

In *Conference on Robot Learning*, 2024.

URL [https://api.semanticscholar.org/CorpusID:269757396](https://api.semanticscholar.org/CorpusID:269757396).

- Yu et al. (2019)

Wenhao Yu, Visak C. V. Kumar, Greg Turk, and C. Karen Liu.

Sim-to-real transfer for biped locomotion.

*2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3503-3510, 2019.

URL [https://api.semanticscholar.org/CorpusID:67856268](https://api.semanticscholar.org/CorpusID:67856268).

- Yuan et al. (2023)

Ying Yuan, Haichuan Che, Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Kang-Won Lee, Yi Wu, Soo-Chul Lim, and Xiaolong Wang.

Robot synesthesia: In-hand manipulation with visuotactile sensing.

*2024 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 6558-6565, 2023.

URL [https://api.semanticscholar.org/CorpusID:265609488](https://api.semanticscholar.org/CorpusID:265609488).

- Zhao et al. (2025)

Shuqi Zhao, Ke Yang, Yuxin Chen, Chenran Li, Yichen Xie, Xiang Zhang, Changhao Wang, and Masayoshi Tomizuka.

Dexctrl: Towards sim-to-real dexterity with adaptive controller learning.

*ArXiv*, abs/2505.00991, 2025.

URL [https://api.semanticscholar.org/CorpusID:278310700](https://api.semanticscholar.org/CorpusID:278310700).

## 附录

我们提供了一个[视频](https://meowuu7.github.io/DexNDM/static/videos_lowres/demo_video_8(1).mp4)和一个[项目网站](https://meowuu7.github.io/DexNDM)来介绍我们的工作。网站和视频包含机器人视频。
我们强烈建议探索这些资源,以直观地理解挑战、我们方法的有效性及其相对于先前方法的优越性。

## 附录 A 方法的补充说明

### A.1 Policy 设计

观测。
Oracle policy 的观测包含:3步关节位置历史(48维)、3步关节位置目标历史(48维)、关节速度(16维)、指尖状态和速度(52维)、物体状态和速度(13维)、物体引导目标姿态(4维)、关节和刚体力(40维)、接触力和二值接触(92维)、手腕朝向(四元数,4维)和旋转轴(3维)。

Reward。
Reward 函数由三部分组成 $r=\alpha_{\text{rot}}r_{\text{rot}}+\alpha_{\text{goal}}r_{\text{goal}}+\alpha_{\text{penalty}}r_{\text{penalty}}$,其中 $r_{\text{rot}}$ 和 $r_{\text{penalty}}$ 沿用 RotateIt (Qi et al., 2023)。旋转项 $r_{\text{rot}}=\operatorname{clip}(\omega_{t}\cdot\mathbf{k},-c,c)$ 鼓励绕单位目标轴 $\mathbf{k}\in\mathbb{R}^{3}$($\lVert\mathbf{k}\rVert_{2}=1$)旋转,其中 $\omega_{t}$ 是物体角速度,$c=0.5$ 限制过大的速度。
惩罚项 $r_{\text{penalty}}$ 抑制偏轴角速度、偏离标准手部姿态、物体线速度以及关节功/力矩:$r_{\text{penalty}}=-\alpha_{\text{rotp}}\|\omega_{t}\times\mathbf{k}\|_{1}-\alpha_{\text{lin}}\|\mathbf{v}_{t}\|_{2}^{2}-\alpha_{\text{pose}}\|\mathbf{q}_{t}-\mathbf{q}_{\text{init}}\|_{2}^{2}-\alpha_{\text{work}}\tau^{T}\dot{\mathbf{q}}-\alpha_{\text{torque}}\|\tau\|_{2}^{2}$,其中 $\mathbf{v}_{t}$、$\mathbf{q}_{\text{init}}$ 和 $\mathbf{\tau}$ 分别表示物体姿态、初始手部关节位置和当前时间步 $t$ 的关节指令力矩,$\alpha_{\text{lin}}=0.3,\alpha_{\text{pose}}=0.3,\alpha_{\text{torque}}=0.1,\alpha_{\text{work}}=2.0$。我们对系数 $\alpha_{\text{rotp}}$ 进行线性调度:训练开始时设为零;使用重置次数来计数训练进程;在第10次重置时,保持 $\alpha_{\text{rotp}}$ 为零;从10到100次,线性增加到0.1;100次之后,保持在0.1。$\alpha_{\text{penalty}}=1.0$

我们发现仅依赖这些 reward 无法解决挑战性问题,如旋转长物体。因此,我们添加了一个中间目标:在回合开始时设置 $\mathbf{p}^{\text{goal}}$ 为沿目标旋转方向前方 $90^{\circ}$ 的位置,并在 $\text{ang\_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}}) < 15^{\circ}$ 时更新它;引导项为 $r_{\text{goal}} = \operatorname{clip}\left(\frac{g_{\text{goal}}}{\text{ang\_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}})+\epsilon}, 0, c_{\text{goal}}\right) + g_{\text{bonus}} \mathbf{1}_{\text{ang\_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}}) < c_{\text{threshold}}}$,其中 $\text{ang\_diff}(\cdot,\cdot)$ 是四元数角距离,$\epsilon > 0$ 确保数值稳定性,$c_{\text{threshold}}$ 是接近阈值。我们设置 $r_{\text{goal}}=1.0$。

控制策略。我们使用20Hz的力矩控制,每个控制步通过运行力矩控制6次来实现。每次关节力矩计算为 $\mathbf{\tau}_{t}=\mathbf{K}_{p}(\mathbf{q}_{t}^{\text{tar}}-\mathbf{q})-\mathbf{K}_{d}\dot{\mathbf{q}}_{t}$,其中 $\mathbf{q}$ 和 $\dot{\mathbf{q}}$ 表示当前关节位置和关节速度,$\mathbf{K}_{p}$ 和 $\mathbf{K}_{d}$ 是预设的常数位置增益和阻尼参数。

通才 Policy 架构。
我们使用具有五个残差块的残差 MLP。输入层是一个隐藏维度为1024的单线性网络。之后堆叠五个残差块,每个隐藏维度为1024。每个残差块处理输入 $\mathbf{x}$:$\mathbf{y}=\text{ReLU}(\text{NN}_{1}(\mathbf{x})+\text{NN}_{3}(\text{ReLU}(\text{NN}_{2}(\mathbf{x}))))$。输出层是一个单线性网络,将隐变量映射到输出维度。

设计选择的进一步讨论。
BC 式训练使我们能够以简单的方式,通过组合来自不同 oracle policy(每个针对特定物体类别训练)的数据集来训练多几何体 policy,从而实现真实世界可部署的通才 policy。
我们使用 BC 来同时实现真实世界部署能力和跨多样物体的通用性。一种替代方案是在教师(teacher)层面实现通用性,例如在所有物体类别上训练任意手腕朝向、任意轴的 RL。然而,这很难奏效。这可能需要我们添加自动或多阶段课程(curriculum)以确保最终 policy 至少能达到每个单独 policy 的性能。这是一个有价值的研究方向。在本工作中,我们选择保持 oracle policy 训练的简洁流水线,采用训练一组教师 policy 的方式,并在学生 policy 训练阶段一次性实现统一的真实世界可部署 policy。

### A.2 主要定理的证明

令 $\mathcal{P}$ 和 $\mathcal{Q}$ 为 $\mathbb{R}^{n}\times\mathbb{R}$ 上的两个概率分布,分别具有概率密度函数(PDF)$P(x)$ 和 $Q(x)$。令 $g:\mathbb{R}^{n}\times\mathbb{R}\to\mathbb{R}^{m}\times\mathbb{R}$ 为可测函数,其中 $m\leq n$。该函数将随机变量 $X\sim\mathcal{P}$(或 $X\sim\mathcal{Q}$)变换为新的随机变量 $Y=g(X)$。令 $g(\mathcal{P})$ 和 $g(\mathcal{Q})$ 表示 $\mathbb{R}^{m}\times\mathbb{R}$ 上的前推分布(pushforward distributions)。

变换后分布之间的 Kullback-Leibler (KL) 散度减小或保持不变,这一性质称为数据处理不等式(Data Processing Inequality):

$$\mathrm{KL}({\mathcal{P}}\|{\mathcal{Q}})\geq\mathrm{KL}({g(\mathcal{P})}\|{g(\mathcal{Q})}).$$ \tag{1}

不等式严格成立($\mathrm{KL}({\mathcal{P}}\|{\mathcal{Q}})>\mathrm{KL}({g(\mathcal{P})}\|{g(\mathcal{Q})})$)的条件是 $g$ 以合并 $\mathcal{P}$ 和 $\mathcal{Q}$ 具有不同相对结构的点的方式非单射。更具体地说,这意味着存在 $y_{0}\in\mathbb{R}^{m}\times\mathbb{R}$,$P(Y=y_{0})>0$,$P(X|Y=y_{0})\neq Q(X|Y=y_{0})$。

我们首先证明 $\mathrm{KL}(\mathcal{P}\|\mathcal{Q})\geq\mathrm{KL}(g(\mathcal{P})\|g(\mathcal{Q}))$ 对任意函数 $g$ 成立。令 $X$ 为从两个分布 $\mathcal{P}$ 或 $\mathcal{Q}$ 之一抽取的随机变量。将其 PDF 记为 $P_{X}(x)$ 和 $Q_{X}(x)$。

令 $Y$ 为通过对 $X$ 施加函数得到的新随机变量:$Y=g(X)$。$Y$ 的分布是前推分布 $f(\mathcal{P})$ 和 $f(\mathcal{Q})$,PDF 为 $P_{Y}(y)$ 和 $Q_{Y}(y)$。考虑 $(X,Y)$ 的联合分布,由于 $Y$ 是 $X$ 的确定性函数,联合概率简单表示为:

$$
\begin{aligned}
P_{X,Y}(x,y) &=P_{X}(x),\text{if}~y=g(x) \tag{2}
\end{aligned}
$$
$$
\begin{aligned}
P_{X,Y}(x,y) &=0,\text{if}~y\neq g(x) \tag{3}
\end{aligned}
$$

使用 KL 散度的"链式法则"(chain rule),我们可以用两种方式展开联合分布:

$$
\begin{aligned}
(A)~\mathrm{KL}(P_{X,Y}\|Q_{X,Y}) &=\mathrm{KL}(P_{X}\|Q_{X})+\mathrm{KL}(P_{Y|X}\|Q_{Y|X}) \tag{4}
\end{aligned}
$$
$$
\begin{aligned}
(B)~\mathrm{KL}(P_{X,Y}\|Q_{X,Y}) &=\mathrm{KL}(P_{Y}\|Q_{Y})+\mathrm{KL}(P_{X|Y}\|Q_{X|Y}) \tag{5}
\end{aligned}
$$

由于 $Y$ 完全由 $X$ 决定($Y=f(X)$),我们有

$$
\begin{aligned}
P(y|x) &=1,~\text{if}~y=f(x), \tag{6}
\end{aligned}
$$
$$
\begin{aligned}
P(y|x) &=0,~\text{if}~y\neq f(x) \tag{7}
\end{aligned}
$$

对 $Q(y|x)$ 同样成立:

$$
\begin{aligned}
Q(y|x) &=1,~\text{if}~y=f(x), \tag{8}
\end{aligned}
$$
$$
\begin{aligned}
Q(y|x) &=0,~\text{if}~y\neq f(x) \tag{9}
\end{aligned}
$$

因此 $P_{Y|X}=Q_{Y|X}$,它们之间的 KL 散度为零:

$$\mathrm{KL}(P_{Y|X}\|Q_{Y|X})=\mathbb{E}_{x~\sim P_{X}}\int_{y}P(y|x)\log\left(\frac{P(y|x)}{Q(y|x)}\right)dy=\mathbb{E}_{x~\sim P_{X}}[0]=0.$$ \tag{10}

因此,展开式 4 简化为

$$\mathrm{KL}(P_{X,Y}\|Q_{X,Y})=\mathrm{KL}(P_{X}\|Q_{X}).$$ \tag{11}

我们有:

$$\mathrm{KL}(P_{X}\|Q_{X})=\mathrm{KL}(P_{Y}\|Q_{Y})+\mathrm{KL}(P_{X|Y}\|Q_{X|Y}).$$ \tag{12}

由于 KL 散度始终非负,这意味着 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})\geq 0$,我们有

$$\mathrm{KL}(P_{X}\|Q_{X})\geq\mathrm{KL}(P_{Y}\|Q_{Y}).$$ \tag{13}

当且仅当等式 12 右侧第二项严格为正时,不等式严格成立,即 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})>0$。
该项是条件分布 $P(x|y)$ 和 $Q(x|y)$ 之间 KL 散度的期望,在分布 $P_{Y}(y)$ 上取平均。当且仅当 $\exists y_{0}\in\mathbb{R}^{m}\times\mathbb{R}$,$P_{Y}(y_{0})>0$,$P(X|Y=y_{0})\neq Q(X|Y=y_{0})$ 时,它严格为正。

这是直接的。我们在下面给出证明。

*充分性。* 由于 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})=\mathbb{E}_{y\sim P_{Y}}\left[\mathrm{KL}(P_{X|Y=y}\|Q_{X|Y=y})\right]$,如果条件满足,则 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})\geq P_{Y}(y_{0})\mathrm{KL}(P_{X|Y=y_{0}}\|Q_{X|Y=y_{0}})>0$。因此这是充分条件。

*必要性。* 可通过反证法证明。假设存在 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})>0$ 但对每个具有非零 $P_{Y}(y_{0})$ 的 $y_{0}$,都有 $\mathrm{KL}(P_{X|Y=y_{0}}\|Q_{X|Y=y_{0}})=0$,则 $\mathrm{KL}(P_{X|Y}\|Q_{X|Y})=\mathbb{E}_{y\sim P_{Y}}\left[\mathrm{KL}(P_{X|Y=y}\|Q_{X|Y=y})\right]=0$,与假设矛盾。因此这是必要条件。

在我们的设置中,由于 $g$ 严格降低了维度且是连续函数(因为它从全手历史中提取单个关节的历史),$g$ 是非单射函数,这将在定理 A.3 中证明。由于 $\mathcal{P}$ 和 $\mathcal{Q}$ 位于不同的数据域(可视化如图 16、图 17 所示),且如我们所展示的 $g(\mathcal{P})$ 和 $g(\mathcal{Q})$ 具有相似性(可视化如图 15 所示),条件 $\exists y_{0}\in\mathbb{R}^{m}\times\mathbb{R}$,$P(Y=y_{0})>0$,$P(X|Y=y_{0})\neq Q(X|Y=y_{0})$ 通常是满足的。

给定数据点 $(X,Y)\in\mathbb{R}^{n}\times\mathbb{R}$,可测函数 $g:(X,Y)\in\mathbb{R}^{n}\rightarrow(g_{X}(X),Y)\in\mathbb{R}^{m},m

其中 $R_{\mathcal{P}}(h)=\mathbb{E}_{(X,Y)\sim\mathcal{P}}[L(h(X),Y)]$ 是预测器 $h$ 的风险,$L$ 衡量预测误差且有上界 B。

利用全期望公式(law of total expectation)和协变量偏移(covariate shift)假设:

定义固定 $x$ 下的"内部风险"(inner risk)函数:

风险差异可以转化为对边际分布 $P_{X}$ 和 $Q_{X}$ 的期望:

两个分布 $P_{X}$ 和 $Q_{X}$ 在函数类 $\mathcal{F}$ 上的积分概率度量(IPM)定义为:

| | $$d_{\mathcal{F}}(P_{X},Q_{X})=\sup_{\phi\in\mathcal{F}}\left|\mathbb{E}_{X\sim P_{X}}[\phi(X)]-\mathbb{E}_{X\sim Q_{X}}[\phi(X)]\right|$$ | |
|---|---|---|---|---|

定义两类"内部风险"函数:

| | $\displaystyle\mathcal{F}_{1}$ | $\displaystyle=\{r_{f_{1}}\mid f_{1}:\mathbb{R}^{n}\to\mathbb{R}\text{ is in the function space for }f_{1}\}$ | |
|---|---|---|---|
| | $\displaystyle\mathcal{F}_{2}$ | $\displaystyle=\{r_{f_{2}\circ g_{X}}\mid f_{2}:\mathbb{R}^{m}\to\mathbb{R}\text{ is in the function space for }f_{2}\}$ | |

我们要证明的不等式变为:

| | $$d_{\mathcal{F}_{2}}(P_{X},Q_{X})考虑任意函数 $\phi\in\mathcal{F}_{2}$。根据定义,$\phi=r_{f_{2}\circ g_{X}}$ 对某个函数 $f_{2}$ 成立。
定义新函数 $f_{1}(x)=(f_{2}\circ g_{X})(x)$。假设 $\mathcal{F}_{1}$ 足够丰富以包含这个组合,则 $r_{f_{1}}=r_{f_{2}\circ g_{X}}=\phi$。这意味着 $\phi\in\mathcal{F}_{1}$。因此,$\mathcal{F}_{2}\subseteq\mathcal{F}_{1}$。

我们直接得到非严格不等式,因为我们在更小的集合上取上确界:

| | $$\sup_{\phi\in\mathcal{F}_{2}}\left|\mathbb{E}_{P_{X}}[\phi]-\mathbb{E}_{Q_{X}}[\phi]\right|\leq\sup_{\phi\in\mathcal{F}_{1}}\left|\mathbb{E}_{P_{X}}[\phi]-\mathbb{E}_{Q_{X}}[\phi]\right|$$ | |
|---|---|---|---|---|---|---|

考虑给定的 KL 条件 $\mathrm{KL}(g(P_{X})\|g(Q_{X}))\leq\mathrm{KL}(P_{X}\|Q_{X})$ 和协变量偏移条件,我们有:$\mathrm{KL}(g_{X}(P_{X})\|g_{X}(Q_{X}))不是区分 $P_{X}$ 和 $Q_{X}$ 的充分统计量(sufficient statistic)。这意味着似然比(likelihood ratio)$w(x)=p_{X}(x)/q_{X}(x)$ 不能写成 $g_{X}(x)$ 的函数。
这进一步意味着存在 $x_{a},x_{b}$ 使得 $g_{X}(x_{a})=g_{X}(x_{b})$ 但 $w(x_{a})\neq w(x_{b})$。

现在考虑函数类:

- $\mathcal{F}_{2}$ 中的任意函数 $\phi$ 在 $g_{X}$ 的水平集(level sets)上必须为常数。如果 $g_{X}(x_{a})=g_{X}(x_{b})$,则 $\phi(x_{a})=\phi(x_{b})$。这些函数对 $g_{X}$ 丢弃的信息是"盲"的。

- $\mathcal{F}_{1}$ 中最大化 IPM 差异的函数 $\phi^{*}$,即 $d_{\mathcal{F}_{1}}(P_{X},Q_{X})$,必须对 $P_{X}$ 和 $Q_{X}$ 之间的差异最大程度敏感。由于这种差异(由似然比 $w(x)$ 捕获)依赖于 $g_{X}$ 丢弃的信息,最优判别函数 $\phi^{*}$ 不能仅是 $g_{X}(x)$ 的函数。

这意味着对较大集合 $\mathcal{F}_{1}$ 取得上确界的函数 $\phi^{*}$ 不包含在较小集合 $\mathcal{F}_{2}$ 中(即 $\phi^{*}\notin\mathcal{F}_{2}$)。

因为 $\mathcal{F}_{1}$ 的上确界由一个在严格更小的集合 $\mathcal{F}_{2}$ 中不可用的函数取得,所以不等式严格成立。

| | $$\sup_{\phi\in\mathcal{F}_{2}}\left|\mathbb{E}_{P_{X}}[\phi]-\mathbb{E}_{Q_{X}}[\phi]\right|证明完毕。

定义在源分布 $\mathcal{Q}$ 上训练的最优预测器为:

$$
\begin{aligned}
f_{1}^{\mathcal{Q}} &=\arg\min_{f_{1}}R_{\mathcal{Q}}(f_{1}) \tag{15}
\end{aligned}
$$
$$
\begin{aligned}
f_{2}^{\mathcal{Q}} &=\arg\min_{f_{2}}R_{\mathcal{Q}}(f_{2}\circ g_{X}) \tag{16}
\end{aligned}
$$

我们接下来证明在特定条件下,在更简单表示上训练的预测器能更好地泛化到目标分布 $\mathcal{P}$。

令 $f_{1}^{\mathcal{Q}}$ 和 $f_{2}^{\mathcal{Q}}$ 分别为在源分布 $\mathcal{Q}$ 上全维和降维空间中的最优预测器。令以下假设成立:

函数类 $\{f_{2}\circ g_{X}\mid f_{2}:\mathbb{R}^{m}\to\mathbb{R}\}$ 具有足够的表达能力来建模源分布 $\mathcal{Q}$ 上的关系。
由降维表示导致的源域风险增加被一个小常数 $\epsilon_{A}$ 所界定:

在定理 A.2 的基础上,我们进一步假设从 $\mathcal{P}$ 到 $\mathcal{Q}$ 存在较大的分布偏移,使得 $f_{2}^{\mathcal{Q}}$ 表现出强的泛化优势,且 $f_{1}^{\mathcal{Q}}$ 和 $f_{2}^{\mathcal{Q}}$ 在泛化差距上的差异满足:

其中 $\epsilon_{B}$ 是正常数。

如果 $\epsilon_{B}>\epsilon_{A}$,则在降维空间中训练的预测器在目标分布上的风险严格更低:

| | $$R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})

分解目标风险:

进一步有:

| | $\displaystyle R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})$ | $\displaystyle=\left[R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})+\left(R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})\right)\right]$ | |
|---|---|---|---|
| | | $\displaystyle\quad-\left[R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})+\left(R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})-R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})\right)\right].$ | | (21) |

重新排列各项:

| | $\displaystyle R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})$ | $\displaystyle=\underbrace{\left[R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})\right]}_{\text{Term A: Approximation Error}}$ | |
|---|---|---|---|
| | | $\displaystyle\quad+\underbrace{\left[\left(R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})\right)-\left(R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})-R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})\right)\right]}_{\text{Term B: Difference in Generalization Gaps}}.$ | | (22) |

由假设 Assumption,Term A 等于 $\epsilon_{A}$:

$$R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})=\epsilon_{A}.$$ \tag{23}

由假设 Assumption,Term B 等于 $-\epsilon_{B}$:

$$\left(R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{Q}}(f_{2}^{\mathcal{Q}}\circ g_{X})\right)-\left(R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})-R_{\mathcal{Q}}(f_{1}^{\mathcal{Q}})\right)=-\epsilon_{B}.$$ \tag{24}

我们有:

$$R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})-R_{\mathcal{P}}(f_{1}^{\mathcal{Q}})=\epsilon_{A}-\epsilon_{B}.$$ \tag{25}

给定条件 $\epsilon_{B}>\epsilon_{A}$,我们有:

| | $$R_{\mathcal{P}}(f_{2}^{\mathcal{Q}}\circ g_{X})证明完毕。

这些假设何时成立?
假设 Assumption 刻画了 joint-wise neural dynamics model 和全手模型之间的域内性能差距。如第 4.2 节和图 6 所示,即使数据充足时也成立。在低数据条件下,joint-wise 模型不仅没有增加源域风险,反而降低了源域风险,这得益于更好的样本效率。

假设 Assumption 刻画了这两种模型的泛化行为。在训练-测试分布偏移下,在我们所有实验中均满足(第 4.2 节;图 6);joint-wise 模型表现出比全手动力学模型好得多的可迁移性。

在我们的灵巧操作设置中,数据稀缺和训练-测试偏移是普遍存在的,因为获取完美分布对齐的数据通常不可行或难以扩展(第 3.3 节),实证证据见第 5 和 B.4 节。即使有自主数据收集,真实世界数据的量远小于仿真,使我们处于低数据条件。因此,joint-wise 建模是我们任务的更优选择,也是成功的关键。相比之下,使用全手动力学模型会降低 sim-to-real 迁移效果(表 4 和表 5)。我们将 bin Shi et al. (2024) 所采用的全身动力学模型(whole-body dynamics model)的成功归因于其域内(in-distribution)设置以及比我们场景更简单的动力学。

$\forall$ $C^{1}$ 函数 $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m},m

对于任意点 $\mathbf{x}\in\mathbb{R}^{n}$,其导数是 Jacobian 矩阵 $Df_{\mathbf{x}}$,它代表从 $\mathbf{x}$ 处切空间(即 $\mathbb{R}^{n}$)到 $f(\mathbf{x})$ 处切空间(即 $\mathbb{R}^{m}$)的线性映射。$Df_{\mathbf{x}}$ 是 $m\times n$ 矩阵。该矩阵的秩最多为 $\min(m,n)=m$。对这个线性映射 $Df_{\mathbf{x}}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ 应用秩-零度定理(Rank-Nullity Theorem),我们发现其零空间的维度 $\geq n-m>0$。根据逆函数定理(Inverse Function Theorem) (Munkres, 2018; Guillemin & Pollack, 2010),函数在点 $\mathbf{x}$ 附近局部单射当且仅当其导数 $Df_{\mathbf{x}}$ 是单射的。如我们所示,当 $n>m$ 时 $Df_{\mathbf{x}}$ 从不单射。由于 $f$ 在任何点都不局部单射,它不可能全局单射。

### A.3 Joint-Wise 动力学建模的合理性(第一部分)

我们用标准操纵器方程 (Murray et al., 2017; Spong et al., 2020) 建模手部,将物体效应视为外力:

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}+\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}+\mathbf{G}(\mathbf{q})=\mathbf{\tau}+\mathbf{\tau}_{\text{ext}},$$ \tag{27}

其中 $\mathbf{M}(\mathbf{q})$、$\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$ 和 $\mathbf{G}(\mathbf{q})$ 分别是惯量矩阵、科里奥利矩阵和重力矩阵。$\mathbf{\tau}$ 是施加的关节力矩,$\mathbf{\tau}_{\text{ext}}$ 表示来自物体的外力。考虑到低速运动,我们忽略科里奥利项 (Craig, 2009; Spong et al., 2005),$\mathbf{C}(\mathbf{q}_{t},\dot{\mathbf{q}}_{t})\dot{\mathbf{q}}_{t}\approx 0$。

假设我们建模第 $i$ 个关节,使用 $(\mathbf{q}^{m},\dot{\mathbf{q}}^{m})$ 表示"被建模关节"(modeled joints)的状态,例如 $\mathbf{q}^{m}=[\mathbf{q}^{i}]^{T}\in\mathbb{R}^{1}$,同时将其余关节视为"从属关节"(slave joints),状态记为 $(\mathbf{q}^{s},\dot{\mathbf{q}}^{s})$,即 $\mathbf{q}^{s}=[\mathbf{q}^{j},\forall 1\leq j\leq 16,j\neq i]^{T}\in\mathbb{R}^{15}$。重排完整动力学方程(等式 27),写为

$$\begin{bmatrix}\mathbf{M}^{mm}_{t}&\mathbf{M}^{ms}_{t}\\ \mathbf{M}^{sm}_{t}&\mathbf{M}^{ss}_{t}\end{bmatrix}\begin{bmatrix}\ddot{\mathbf{q}}^{m}_{t}\\ \ddot{\mathbf{q}}^{s}_{t}\end{bmatrix}+\begin{bmatrix}\mathbf{G}^{m}_{t}\\ \mathbf{G}^{s}_{t}\end{bmatrix}=\begin{bmatrix}\tau^{m,\text{total}}_{t}\\ \tau^{s,\text{total}}_{t}\end{bmatrix}.$$ \tag{28}

推导被建模关节的方程:

$$(\mathbf{M}^{mm}-\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}\mathbf{M}^{sm})\ddot{\mathbf{q}}^{m}+\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}(\tau^{s,\text{total}}-\mathbf{G}^{s})+\mathbf{G}^{m}=\tau^{m}=[\tau^{i}+\tau^{i,\text{ext}}]^{T}.$$ \tag{29}

引入"有效力矩"(effective torque)为 $\tau^{\text{eff}}=[\tau^{i,\text{ext}}]^{T}\in\mathbb{R}^{1}$,方程写为:

$$(\mathbf{M}^{mm}-\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}\mathbf{M}^{sm})\ddot{\mathbf{q}}^{m}+\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}(\tau^{s,\text{total}}-\mathbf{G}^{s})+\mathbf{G}^{m}-\tau^{\text{eff}}=[\tau_{i}]^{T}.$$ \tag{30}

令 $\mathbf{H}_{t}^{\mathrm{eff}}$ 表示有效惯量矩阵,$\mathbf{H}_{t}^{\mathrm{eff}}\triangleq\mathbf{M}^{mm}-\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}\mathbf{M}^{sm}$,令 $\mathbf{G}_{t}^{\mathrm{eff}}$ 表示有效外部项,$\mathbf{G}_{t}^{\mathrm{eff}}\triangleq\mathbf{M}^{ms}(\mathbf{M}^{ss})^{-1}\bigl(\bm{\tau}^{s,\mathrm{total}}-\mathbf{G}^{s}\bigr)+\mathbf{G}^{m}-\bm{\tau}^{\mathrm{eff}}$。给定 $\mathbf{H}_{t}^{\mathrm{eff}}$、$\mathbf{G}_{t}^{\mathrm{eff}}$ 和被建模关节力矩 $\tau_{t}^{i}$,加速度 $\ddot{\mathbf{q}}_{t}^{i}$ 被唯一确定。
$\mathbf{H}_{t}^{\mathrm{eff}}$ 和 $\mathbf{G}_{t}^{\mathrm{eff}}$ 与其他关节的状态和力矩相关。

这表明在高度耦合的交互系统中,每个单独关节的动力学与其他关节的状态、力矩以及物体的外部影响相关。采用神经网络方法来求解动力学演化,以期考虑所有这些高自由度影响,将不可避免地需要大量具有正确分布的数据,无法解决数据方面的挑战。

聚焦于每个单关节动力学系统,joint-wise neural dynamics 从每个单关节自身的状态-动作历史预测其 transition。从历史中预测推广了 RMA 方法在旋转中的思想 (Qi et al., 2022),以隐式地在高层次上考虑时变影响。
我们将证明,在短时间窗口(例如10帧,对应0.5s)和某些假设下,这种方法是合理的。

具体来说,我们假设在动作轨迹执行期间的任何短时间窗口内,每个从属关节的状态轨迹(即 $\mathbf{q}^{s}$)、施加到每个从属关节的主动力矩(即 $\mathbb{\tau}^{s}$)以及施加到每个关节的有效外力矩($\mathbf{\tau}^{\text{ext}}$)可以用无限可微的连续函数在可接受的误差范围内近似。直觉上,在动作为 policy 网络输出的连续演化动力学系统中,关节状态和主动力矩(与输入位置目标相关)的这一假设成立。如果我们进一步假设软接触模型(soft contact model) (Tedrake & the Drake Development Team, 2019; Pang & Tedrake, 2021),由与物体接触力引起的有效外力矩的假设也是合理的。

我们提供这两个假设的统计证据。具体来说,我们证明它们可以用多项式函数(一类特殊的无限可微连续函数)拟合到可接受的误差。

*图 10: 每关节状态-动作序列(Free Hand, 无载荷)。*

*图 11: 每关节状态-动作序列(自主数据收集,有载荷)。*

*图 12: 每关节状态-动作序列(任务感知数据)。*

每关节状态轨迹的模式。
图 10、图 11 和图 12 展示了在三种条件下收集的真实世界状态-动作轨迹:无物体载荷的自由机械手、通过我们的自主数据收集系统带载荷收集、以及需要人工干预的任务感知数据收集。在这三种外部影响下,手部的动作和状态轨迹在视觉上都是平滑的。

我们进一步分析了它们的多项式拟合结果。图 32 展示了在10长度时间窗口上每关节状态序列的3阶多项式拟合结果。图 34 展示了在所有测试的10长度序列上取平均的每关节拟合误差。
我们可以观察到良好的拟合结果,原始曲线可以被拟合曲线大致近似。
如果将多项式阶数增加到5,可以观察到优秀的拟合结果(图 33、图 35)。这些统计结果展示了关节状态序列连续函数假设的合理性。

每关节主动力矩轨迹的模式。
由于我们无法直接感知力矩,对于每个关节 $i$,我们分析每个时间步 $t$ 的位置目标与关节状态之间的差异,即 $\mathbf{q}_{t}^{i,\text{tar}}-\mathbf{q}_{t}^{i}$,以反映驱动力矩的相应统计特性。
图 36 和图 37 分别展示了使用3阶和5阶多项式函数的拟合结果。图 38 和图 39 进一步展示了每关节的平均拟合误差。动作力的演化比关节状态更复杂。但我们仍然可以看到令人满意的拟合结果。随着多项式阶数增加,拟合结果变得更好。

每关节外力矩轨迹的模式。
由于我们无法直接从真实世界测量每关节有效外力矩(与物体和手之间的接触力相关),我们引入"虚拟物体力"(virtual object force,也称"虚拟力"或"虚拟力矩")作为实际外力矩的代理。
具体来说,我们首先训练每关节逆动力学模型,从状态-动作历史和下一实际状态预测施加的动作,即 $f^{\text{invdyn},i}:\{(\mathbf{s}^{i}_{k+1},\mathbf{a}_{k}^{i})\}_{k=t-W+1}^{t}\in\mathbb{R}^{2W}\rightarrow\hat{\mathbf{a}}_{t+1}\in\mathbf{R}^{2W}$,使用无物体手回放轨迹训练。因此,它预测应施加什么动作才能使下一关节状态达到期望值,不考虑物体影响(无外力矩)。然后,对于收集的任务感知轨迹,我们首先使用逆动力学模型预测期望动作 $\hat{\mathbf{a}}_{t+1}$。我们然后使用其与实际动作的差异计算"虚拟力",即 $\mathbf{a}_{t+1}-\hat{\mathbf{a}}_{t+1}$。由于这种差异反映了需要多少额外动作来抵抗物体以使关节达到期望状态。
我们然后分析这个量的统计特性。

如图 40、图 41、图 42、图 43 所示,我们仍然可以得到令人满意的拟合结果,尽管这个量的演化比主动力矩和关节状态都更复杂。

基于此,我们可以假设 $\mathbf{H}^{\text{eff}}$ 和 $\mathbf{G}^{\text{eff}}$ 的演化在所考虑的时间窗口上是良好的连续函数。然后我们可以用低阶函数(例如使用其 Taylor 展开)在可接受的误差范围内近似它们的演化。假设 $\mathbf{H}^{\text{eff}}$ 使用 $k_{1}$ 阶而 $\mathbf{G}^{\text{eff}}$ 使用 $k_{2}$ 阶,底层未知变量的数量变为 $k_{1}+k_{2}$。求解所有未知变量足以求解下一步 transition。每个关节的状态-动作历史可以被视为具有 $k_{1}+k_{2}$ 个未知参数的函数 30 的输入和输出,如果历史足够长,则包含足够的信息来求解它们。这进一步表明了使用神经网络从状态-动作历史预测下一 transition 的合理性,考虑到输入中包含的充分信息和神经网络的通用逼近能力。

### A.4 Joint-Wise 动力学建模的合理性(第二部分)

*图 13: 通过单关节状态-动作历史预测(泛化误差)。*

*图 14: 通过单关节状态-动作历史预测(域内验证误差)。*

在上一节中,我们证明了单个关节的状态-动作历史足以预测其自身的下一 transition。这表明单关节状态-动作历史中包含的信息至少足以考虑短时间窗口内低维有效变量的演化,即 $\mathbf{H}_{t}^{\text{eff}}$ 和 $\mathbf{G}_{t}^{\text{eff}}$。然而,这不足以证明从历史中学习预测的模型不会隐式地学习预测原始高维复杂力(如关节间耦合)来预测 transition。

证明这一点很重要,因为如果单关节状态-动作历史包含足够的信息来预测更高阶系统的状态,那么从单关节历史学习就不是有效的降维,会因模型仍然过拟合到系统的高方差影响而损害泛化能力。

我们通过实验来证明单个关节的状态-动作历史不包含足够的信息来预测其他关节的信息。

我们训练 joint-wise dynamics model 来预测以下信息:1) 下一个关节的当前状态,2) 前一个关节的当前状态,3) 下一个关节的动作(位置目标),4) 前一个关节的动作(位置目标)。
然后我们将它们的预测误差和泛化误差与 joint-wise dynamics model(预测自身下一状态)取得的结果进行比较分析。

我们使用真实世界 transition 数据从头训练所有模型,不使用仿真数据预训练。真实世界 transition 数据与我们在消融研究中使用的相同。
如图 14 和图 13 所示,利用单关节状态-动作历史预测其他关节的统计量甚至无法在原始分布上达到合理的性能。泛化误差比使用单关节状态-动作历史预测自身下一 transition 大三个数量级。至于域内验证误差(在域内验证集上取得,接近训练误差),预测相邻关节状态的性能略好于预测其动作。然而,这仍然远非合理的预测,误差比预测关节自身 transition 大两个数量级。

这些实验证明,即使预测导致复杂耦合的最简单信息(即相邻关节的状态和动作),通过单关节状态-动作历史也是不可行的。这进一步表明单关节状态-动作历史不包含足够的信息来考虑原始高维空间中的复杂影响因素。由于这些信息足以预测关节自身的 transition,一个合理的假设是网络倾向于从历史中隐式利用这些净效应(net effects)来预测动力学演化。

Joint-wise neural dynamics model 隐式捕获了什么?
第 A.3 和 A.4 节的分析和实验阐明了什么可以从单关节状态-动作历史预测,什么不可以。我们的综合实验(第 4.2 节)表明 joint-wise neural dynamics 具有表达能力、样本效率且泛化良好。第 A.3 节的分析表明单关节历史包含足够的信息来近似其下一 transition,而第 A.4 节表明它无法恢复每个底层耦合效应。因此,每关节历史捕获了低维净效应,同时避免过拟合到系统级变化。这种分解的、每关节的建模在全手交互变化中具有可迁移性,因为净效应的分布相比全系统交互的分布更加稳定。

Joint-wise neural dynamics 模型的局限性。
如图 6 所示,在多任务高数据条件下的域内测试设置中,joint-wise dynamics model 的性能略差于全手动力学模型。
优化速度也是一个局限,因为遍历所有关节需要时间,导致更长的训练时间。

### A.5 收集轨迹与旋转轨迹之间的数据分布比较

*图 15: 每关节分布*

*图 16: 每手指分布*

*图 17: 全手分布*

图 15、图 16 和图 17 总结了每关节、每手指和全手的数据分布。它比较了我们自主数据收集策略收集的轨迹和与任务相关的旋转轨迹。
任务相关轨迹是在"Thumb Up"手腕朝向下收集的20条立方体旋转轨迹(共约8,000个数据点)。
每关节状态-动作轨迹可以很好地覆盖任务感知旋转轨迹的分布。然而,每手指和全手分布表现出巨大的差异。

## 附录 B 补充实验与分析

### B.1 训练性能

*图 18: 训练性能。我们的方法与重新实现的 AnyRotate 在不同训练集上取得的最终训练性能(总 reward)比较。"DexEnv Objects" 表示不规则训练物体类别。*

AnyRotate (Yang et al., 2024) 在多样手腕朝向和各种旋转轴的通用性方面改进了先前工作。然而,他们只考虑了规则物体。为复杂物体实现这种通用旋转能力带来了额外挑战,即使在 policy 训练方面也是如此。
在我们的实验中,我们发现先前的旋转 policy RL 设计 (Qi et al., 2022; 2023; Yang et al., 2024),其中观测中只考虑本体感觉和物体及系统参数相关的特权信息(如质量),可能导致训练陷入局部最优。
因此,我们在观测中包含了更多特权信息,随后进行观测空间蒸馏(observation space distillation)用于 sim-to-real(第 3.1 节)。
我们与重新实现的 AnyRotate 比较以证明这种设计的优越性。
我们的方法显示出比 AnyRotate 明显更好的训练性能(图 18),特别是在挑战性物体集上,即具有不规则和复杂几何形状的"DexEnv Objects"和以小尺寸为特征的"Small Cylinders",其中 AnyRotate 中无法出现稳定的手指步态(finger gaiting)。
我们还在 Hora (Qi et al., 2022) 代码库中重新实现了 RotateIt (Qi et al., 2023),但发现它在最基本的圆柱体物体集上也很难取得令人满意的结果。
我们还将 Hora 适配到手掌朝下的场景,但发现它无法工作。

*图 19: 真实世界中评估的物体。*

### B.2 补充真实世界结果

*图 20: 真实世界结果。在空中旋转挑战性物体。更多内容和视频请见我们的[项目网站](https://meowuu7.github.io/DexNDM/)。*

*图 21: 多样的手腕朝向。*

图 20 和图 21 提供了更多真实世界的定性结果。更多结果和视频请见我们的[项目网站](https://meowuu7.github.io/DexNDM/)。

### B.3 Sim-to-Real 方法有效性的案例研究

*表 7: Sim-to-Real 方法在挑战性形状上的有效性。Base policy 有/无 DexNDM 在挑战性形状(即高长宽比、小尺寸和复杂几何形状)上 Rot (弧度)的比较。在手掌朝下条件下测试。括号中的符号表示旋转轴。数值为三次独立试验的平均值。*

如表 4 和表 5 所示,我们在学习 neural dynamics 和 residual policy 用于 sim-to-real 方面的设计可以取得明显优于无 sim-to-real 设计的 policy 的结果。下面我们介绍关于 sim-to-real 方法的几个经验观察和案例研究。值得注意的是,residual policy 可以有效地改善在挑战性形状上的性能,帮助我们解决之前无法解决的旋转任务,同时也增强了旋转的稳定性(表 7)。

旋转挑战性物体。
Residual policy 的一个重要特点是使我们能够旋转具有高长宽比或困难物体-手比例的挑战性物体。例如,在没有 sim-to-real 策略的情况下,policy 最多只能将长"Lego"腿(宽=3cm,长=13.5cm)旋转180度。然而,引入 residual policy 可以帮助我们将其旋转(几乎)一整圈(如图 20 和我们[项目网站](https://meowuu7.github.io/DexNDM/)上的视频所示)。
对"book"物体(16cm长)也有相同的观察。

改善稳定性。
除了赋予我们旋转挑战性物体的能力外,residual policy 还可以有效地使旋转更加稳定,从而帮助我们实现长时间旋转。一个代表性的例子是旋转3cm$\times$3cm$\times$10cm的长方体在竖直姿态下。在处理这种细物体时,policy 会使用三根手指——拇指、中指和小指——来旋转物体。
与使用四根手指相比,这种旋转步态是不稳定的。
如果不包含 residual policy,我们最多能旋转物体5圈。然而,包含 residual policy 可以让我们连续旋转物体超过5分钟,对应约30圈。
旋转"cube"物体沿 y 轴也有类似观察。

### B.4 进一步讨论、分析和消融研究

Residual Policy vs. 直接微调。
适配 base policy 的一个自然替代方案是直接微调(direct fine-tuning)。我们通过在学到的动力学模型上微调 base policy 来评估这一点。在实践中,该方法不稳定且对超参数高度敏感:使用与 residual policy 训练相同的训练策略且不进行额外稳定化,微调后的 policy 表现出不规则行为,甚至无法执行基本旋转。

我们没有进一步调查这个问题;相反,我们采用了 residual policy 补偿方法,它实现简单、训练稳定且需要最少的专门训练技巧。

真实世界中评估的物体。
我们的 policy 展示了在真实世界中旋转各种物体的有效性。真实世界物体展示照片:图 19。

*表 8: 每关节 Delta Action 幅度。在真实世界中沿 z 轴旋转圆柱体(半径=5.5cm,长度=5.5cm)时每关节 delta action 幅度的运行平均值。关节按 Isaac Gym 中的关节顺序排列。*

每关节 Delta Action 值。
表 8 总结了在真实世界实验中绕 z 轴旋转圆柱体(半径5.5cm,长度5.5cm)时观察到的每关节 delta action 幅度。这些值量化了施加到每个关节的补偿量。

任务相关数据收集的固有局限性。
收集带物体姿态估计的任务相关 transition 存在以下固有局限性:
1) 由于严重遮挡,无法应用于小物体;2) 无法为轴对称物体(如圆柱体)估计准确的完整姿态。
3) 由快速运动、跟踪不准确和严重遮挡导致的噪声姿态;
4) 首次设置的巨大时间成本(即数天),以及每次数据收集前启动流水线的较大时间成本(即约一分钟)。

此外,只能保留成功的轨迹,因为手此时不会经历载荷,物体掉落会导致快速运动和估计失败。我们只能展开 policy 并使用干净的动作,没有添加噪声的灵活性,这可能导致任务失败。因此,数据的多样性将受限于可以估计的物体,并偏向于简单几何形状。此外,使用的物体形状和尺寸应与训练中使用的匹配。即使我们能收集大量数据,如果仅从物体状态(不含形状信息)学习,动力学模型学习也是相对病态的(ill-posed),因为对于不同物体,相同的状态和动作可能导致不同的 transition。在动力学建模中包含物体形状将不可避免地进一步增加建模维度,需要更大量的数据来学习。

即使不估计物体姿态,收集任务相关数据也固有地受限于低效率、有限覆盖和受限多样性,因为 1) 数据会偏向于能被良好旋转的简单物体,2) 无法添加噪声(会导致旋转失败),3) 需要人工干预来重置物体到手中。根据我们的实验,平均时间成本为42.86s。

*图 22: 小物体操作过程中的姿态跟踪。*

*图 23: 轴对称物体的姿态跟踪。*

通过 Foundation Pose 估计物体姿态的案例研究。
通过利用视觉估计器跟踪物体姿态来收集真实世界 transition 是困难的,需要频繁且繁琐的人工干预,且容易产生噪声结果。
对于每个物体,我们需要其具有完全相同尺寸的 CAD 模型。初始化步骤包括通过相机拍摄图像并利用 XMem (Cheng & Schwing, 2022) 获取物体掩码(mask)。在每次试验开始时,需要将物体放在我们获取掩码时的姿态附近。之后,需要将物体从桌子上移到机械手中并启动 policy。

数据收集的难度因物体几何形状而异。
对于正常尺寸的物体,局限性主要在于噪声估计、耗时和人力密集。平均来说,我们需要200s来收集一条可用的 transition 轨迹。

然而,对于小物体,它很难产生成功甚至可用的数据。
如果我们最初将物体放在桌子上,然后将物体移到机械手附近,姿态跟踪会失败,即使我们移动得非常缓慢。
为了解决这个问题,我们用手将物体握在靠近机械手的姿态进行初始化。之后,需要将其插入机械手进行旋转。当人手从物体上缩回时,估计的姿态会偏离物体(图 22)。

此外,对于轴对称物体,Foundation Pose 无法给出稳定的估计,姿态在物体静止时持续"旋转"(图 23)。
这阻止了我们获得高质量和干净的姿态估计。

我们自主数据收集的优越性。
与任务相关数据相比,我们的自主数据收集是与物体无关的(object-agnostic)。在任务执行期间,手会持续受到时变物体影响。所有载荷对每个关节的联合效应模拟了来自耦合效应和物体的各种外部影响。也可以在数据收集中使用任何其他物体来扩展多样性。此外,我们可以向回放动作添加噪声以扩展多样性和覆盖范围。
而且,它是高效的且不需要人工干预。

回放基础波形收集数据的固有局限性。
获取真实世界 transition 的另一种方法(不同于开环回放 policy 动作展开和展开 policy)是播放参数化波形,如正弦波、方波和高斯噪声 (Fey et al., 2025)。
与使用 policy 数据相比,这种策略存在以下缺点:1) 对于灵巧手,向单个关节发送信号而保持其他关节不动会导致自碰撞,可能损坏硬件。2) 基于通过播放此类信号获得的 transition 数据学到的模型(无论是我们工作中的动力学模型还是 UAN 和 ASAP 中的补偿器),在应用于后续的 policy 微调或补偿器训练场景时可能会受到分布偏移的影响,特别是当模型输入包含历史时。
3) 设计此类波形的频率和幅度是劳动密集且耗时的。因此,我们采用使用 policy 展开来获取真实世界 transition。

*图 24: 性能随数据集大小的缩放。我们通过幂律拟合"Task-Aware w/ Obj. Pose"的曲线并外推,以估计达到期望结果所需的数据量。*

任务相关数据(带物体姿态)。
我们使用一个 5cm $\times$ 5cm $\times$ 5cm 的立方体收集带物体状态标注的真实世界 transition 轨迹。在数据收集过程中,我们在绕 z 轴旋转物体的同时展开 policy,并用 FoundationPose 估计其姿态。由于立方体是对称的,我们在跟踪开始时通过翻转模型来对齐我们的坐标系约定以解决姿态帧歧义。每个数据收集回合平均持续约200s。我们评估了包含17条和54条轨迹的数据集。在与消融实验相同的真实世界评估协议下,平均旋转分别为0.55和0.70。拟合这些点的学习曲线,我们估计需要多少轨迹才能匹配我们使用4,000条自主轨迹的方法的性能。如图 24 所示,估计为52,483,440条轨迹——显然不切实际。虽然这种外推基于少量数据点,但它突出了我们方法的数据效率和泛化能力。

我们尝试使用这些任务相关的、带物体状态标注的数据训练 sim-to-real 基线(ASAP 和 UAN),但即使是第一阶段——补偿器训练——也未能收敛,reward 没有显示有意义的改善,可能是由于数据质量差。

## 附录 C 补充实验细节

*图 25: 通用旋转轴。*

*表 9: 训练物体集和测试物体集的信息及物理参数随机化范围。*

*图 26: 真实世界实验中使用的小物体尺寸。*

数据集。
我们的训练物体包括以下子集:1) Hora (Qi et al., 2022) 的正常尺寸圆柱体;2) Hora (Qi et al., 2022) 的正常尺寸长方体;3) 长长方体;4) 小尺寸圆柱体;5) Visual Dexterity (Chen et al., 2022) 的正常尺寸复杂形状(记为"[DexEnv](https://github.com/Improbable-AI/dexenv) Objects")。带尺寸随机化范围的详细信息总结在表 9 中。

为了测试在未见形状上的泛化性能,我们从 ContactDB 数据集 (Taheri et al., 2020)(来自 [GRAB dataset](https://grab.is.tue.mpg.de/))中筛选长宽比不大于 2:1 的物体作为测试集,共得到26个物体。筛选规则遵循 RotateIt (Qi et al., 2023)。由于我们在此评估中旨在测试形状变化的泛化性能,我们不考虑高长宽比物体或将其缩放到小尺寸。

在真实世界中,我们在三个子集上测试性能(图 5,紫色物体和小物体是未见过的):

- 规则物体:立方体(5cm x 5cm x 5cm)、圆柱体(半径5.5cm,长度5.5cm)、苹果(GRAB/ContactDB 苹果,缩放到0.5x)、长方体(3cm x 10cm x 3cm)和灯泡(FurnitureBench 的"lamp_bulb")。

- 小物体:网上购买;供应商链接在审稿期间隐去以保护匿名性,将在接受后提供。图 26 展示了真实世界实验中使用的这些物体的尺寸。

- 正常尺寸不规则物体:Visual Dexterity 的熊、卡车和牛(各缩放到0.7x);以及 GRAB/ContactDB 的兔子、大象、鸭子、杯子、茶壶和老鼠(各缩放到0.5x)。

Policy 优化。我们使用 PPO 进行 policy 优化。圆柱体和长方体的训练环境为30,000个,长长方体、小圆柱体和"DexEnv Objects"为50,000个。我们在每次环境重置时随机采样手腕姿态和目标旋转轴。

通用旋转轴。为构建通用旋转轴集,我们在 SO(3) 中均匀生成32个轴。去除六个主轴 $\pm x$、$\pm y$ 和 $\pm z$ 后,得到通用旋转轴集。图 25 提供了所有32个均匀分布旋转轴的可视化。

通过行为克隆训练通才 Policy。为获得训练通才 policy 的数据集,我们在仿真中展开每个 oracle policy 来构建数据集。只有在完整400步内不会终止的 transition 轨迹才会被保存在数据集中。我们将测试环境的最大数量设为1,500,000。在每一步中,手部关节状态、位置目标、物体状态、旋转轴和手腕朝向都会被保存。
每个物体类别收集的轨迹数量总结在表 10 中。
成功展开的数量可以反映不同训练物体集的难度。在所有五个物体集中,规则圆柱体和长方体构成最简单的旋转任务。小圆柱体由于其小尺寸引入了额外挑战。几何复杂度进一步增加了难度。旋转具有大长宽比的长物体是最困难的任务,产生了最小的 transition 数据集。

*表 10: 仿真中收集的 Transition 轨迹数量。*

评估指标(详细版)。
我们在仿真和真实世界中使用 RotateIt 指标 (Qi et al., 2023),加上面向目标的成功率指标:Time-to-Fall (TTF)——物体掉落前的持续时间;在仿真中回合上限为400步(20s),TTF 归一化到20s,而在真实世界中报告原始时间;Rotation Reward (RotR)——$\bm{\omega}\cdot\mathbf{k}$ 的回合总和(仅仿真);Rotation Penalty (RotP)——$\bm{\omega}\times\mathbf{k}$ 的每步平均值(仅仿真);Radians Rotated (Rot)——真实世界中旋转的总弧度,从视频中测量。我们还报告 Goal-Oriented Success (GO Succ.,面向目标成功率)沿用 Visual Dexterity(仅仿真):采样一个随机目标姿态,将目标轴设为相对旋转轴,如果最终朝向在目标的 $0.1\pi$ 以内则计为成功。

自动系统辨识。
除了训练 neural dynamics model 和 delta action model 来弥合 sim-to-real gap 外,我们还会在开始时通过执行自动系统辨识(system identification)过程来对齐仿真器和真实世界之间的动力学。
该过程包括以下步骤:1) 在仿真器中使用默认 PD 增益和 URDF 中的连杆配置训练探测旋转技能(probing rotation skills)。2) 在仿真器中展开探测技能获取多条状态-动作轨迹(记为"探测轨迹")。在真实机器人上回放探测轨迹。
3) 收集结果状态和动作轨迹。4) 在仿真器中启动多个并行环境,每个具有不同的系统参数;5) 回放探测动作轨迹以获取结果状态轨迹。6) 选择结果状态轨迹与真实世界最相似的环境参数作为辨识出的系统参数。
我们辨识 PD 增益和每个连杆的质量。辨识值总结在表 11 和表 12 中。

*表 11: 辨识的 PD 增益。通过自动系统辨识过程辨识的每关节 PD 增益。关节按 Isaac Gym 中的关节顺序排列。*

*表 12: 辨识的连杆质量。通过自动系统辨识过程辨识的每连杆质量。连杆按 Isaac Gym 的连杆顺序排列。*

域随机化(Domain Randomization)。
我们在训练期间应用域随机化。我们还在仿真器中的测试期间随机化物理参数。每个物体集的随机化范围总结在表 9 中。沿用先前工作 (Qi et al., 2022; 2023),我们对物体施加随机扰动力。力的大小为 2m,其中 $m$ 是物体质量。我们还以概率0.25在每个时间步重新采样力。我们向关节位置添加从分布 $\mathcal{U}(0,0.005)$ 采样的噪声以增强鲁棒性。

基线方法(详细版)。
我们将方法与先前的手内旋转/重定向工作和先前的神经 sim-to-real 工作进行比较。
我们与两个强手内旋转/重定向工作进行比较,Visual Dexterity (Chen et al., 2022) 和 AnyRotate (Yang et al., 2024)。
AnyRotate 的实验设置与我们最相似。它展示了在各种手腕朝向下的多轴物体旋转。然而,其代码未公开,且该方法需要触觉信息。我们根据论文描述在 IsaacGym 中重新实现了其环境设置和训练流水线。
我们尽力在真实世界中建立公平比较。
不幸的是,仅从论文忠实地复制其触觉传感器模型和 sim-to-real 方法是困难的。
我们发现在其第二阶段训练中丢弃触觉信息很难产生在真实世界中具有基本旋转能力的 policy。因此,直接的真实世界比较不可行。相反,我们通过在其实验中使用的相同挑战性物体形状上评估来证明我们方法的优越性能。

对于 Visual Dexterity,开源代码是为 D'Claw 手设计的,该手比拟人灵巧手(如 Allegro 或 LEAP)大得多且形态学差异很大。
尽管我们大量努力将其代码适配到 LEAP 手,policy 在基本圆柱体形状上的仿真中也无法取得合理性能,即使训练了1.5天。因此,直接比较不可行。因此我们将方法的性能与其论文中报告的定量结果和其[项目网站](https://taochenshh.github.io/projects/visual-dexterity)上展示的定性结果进行比较。

我们还与先前为机械臂和腿式机器人设计的 sim-to-real 方法进行比较,即 UAN (Unsupervised Actuator Net) 和 ASAP。
UAN 和 ASAP 的核心相似,都在于收集真实世界的执行器 transition 数据,训练神经补偿器(neural compensators)来弥合仿真器和真实世界之间的动力学差距,随后基于学到的神经补偿器调优/训练任务 policy。主要差异在于两个方面,包括数据收集和模型设计。ASAP 在真实世界中展开跟踪 policy 和运动 policy 来收集真实世界 transition,而 UAN 通过播放正弦波、方波和高斯噪声来避免使用 policy 数据以防止过拟合。UAN 为每个执行器使用共享网络,而 ASAP 训练全身补偿器(四个踝关节用于 sim-to-real)。

如前所述(第 3.3 节),将物体纳入系统建模或在仿真器中复制物体影响都是不可能的。因此,我们收集了24,000条真实世界无物体手回放轨迹来训练其对应的补偿器。为比较 UAN,我们采用其真实世界收集策略,为手中每个关节训练共享补偿器。为比较 ASAP,我们回放 policy 展开,在 sim-to-real 比较中为每根手指训练补偿器,反映其四踝关节 sim-to-real 设置。在 sim-to-sim 中,我们为全手和物体训练补偿器。

与 AnyRotate 的比较(详细版)。
我们将真实世界性能与 AnyRotate 报告的值进行比较。由于他们没有提供获取真实世界测试物体的链接,我们在其四个易于复制的测试物体上测试模型,包括"Tin Cylinder"、Cube、"Gum Box"和"Container"(详见下文)。而其余的塑料蔬菜模型和"Rubber Toy"根据其表10提供的物体尺寸信息无法复制。根据其实验,具有锐利边缘的物体比塑料蔬菜模型更难旋转(其在"Tin Cylinder"、"Gum Box"和"Container"上的性能在旋转次数和存活时间方面是所有测试物体中最差的,如其表12和13所示)。我们在旋转轴测试设置中测试来自 AnyRotate 的三个测试旋转轴的性能。我们还在手部朝向测试设置中采用与 AnyRotate 相同的旋转轴设置和手部朝向设置。
我们进行三次独立实验,并在表 2 中呈现三次试验的平均值和偏差。
如所示,我们可以大幅优于 AnyRotate。

此外,如所展示的,我们的 policy 可以旋转具有多样长宽比和各种物体-手比例的广泛物体。旋转其中一些物体(如长 Lego 腿和动物形状)需要相当精巧的手指步态(finger gaiting)。然而,AnyRotate 仅展示了使用保守行为旋转正常尺寸、表面相对平坦的物体的能力。如其论文所述,他们在旋转具有锐利边缘的物体时会遇到困难。此外,他们展示有效性的最小物体是"Rubber Toy"(8cm $\times$ 5.3cm $\times$ 4.8cm)、"Tin Cylinder"(4.5 $\times$ 4.5cm $\times$ 6.3cm)和"Cube"(5.1cm $\times$ 5.1cm $\times$ 5.1cm)。然而我们可以处理更小的物体,如尺寸为 3cm $\times$ 3cm $\times$2.5cm、3cm $\times$ 2.75cm $\times$ 2.75cm 和 3cm $\times$ 2cm $\times$ 2.1cm 的蔬菜模型。
此外,其物体最具挑战性的长宽比为1.67(Rubber Toy),而我们可以处理具有挑战性长宽比的物体,如 Lego 腿(4.5)、Book(5.3)和长长方体(3.33)。
这些比较进一步证明了我们方法在解决困难手内旋转问题方面的优越性。

关于我们复制 AnyRotate 物体的细节。
我们复制了其四个测试物体如下:

- Cube:我们3D打印了一个符合指定尺寸 5.1cm $\times$ 5.1cm $\times$ 5.1cm 的立方体。

- Container:我们购买了一个与其实验中使用的容器精确匹配的商业产品。我们移除了容器上的标签以维护地区匿名性。

- Tin Cylinder:我们3D打印了一个指定半径4.5cm、长度6.3cm的圆柱体。

- Gum Box:我们发现记录的尺寸(9cm $\times$ 8cm $\times$ 7.6cm)存在差异,这与"Container"的尺寸相同。然而,原论文中的图片表明"Gum Box"明显更小。因此,我们从图片中估计其尺寸约为 5cm $\times$ 4cm $\times$ 8cm,并3D打印了该尺寸的物体作为代理。

与 Visual Dexterity 的比较(详细版)。
与先前工作相比,Visual Dexterity 在旋转具有不平坦表面的更复杂物体和对未见几何形状的更好泛化能力方面展示了改进的结果。然而,由于不同的任务设置(即我们的面向轴的连续旋转 vs. Visual Dexterity 的面向目标姿态的重定向),进行直接且完全公平的比较是不可行的。因此,我们引入了生存旋转角度(survival rotation angles)这一新指标,可以从两种设置的定性结果中计算,以便于比较。具体来说,它评估物体在从手中掉落前能被旋转的角度。这个指标对 Visual Dexterity 更友好,因为在某些设置中它有一个支撑桌。物体在旋转过程中可以接触桌子。我们通过仔细检查其[项目网站](https://taochenshh.github.io/projects/visual-dexterity)所有视频中呈现的所有演示来获取 Visual Dexterity 的结果。其最佳性能和与我们结果的比较总结在表 3 中。尽管该指标对 Visual Dexterity 更友好,我们仍然可以在其演示中包含的所有不规则物体上达到相当的性能或超越其结果(视频请见我们的[项目网站](https://meowuu7.github.io/DexNDM/))。具体来说,我们做出以下观察:
1) 对于 Visual Dexterity 展示了强结果的物体,包括牛、熊和卡车(它们展示了在不掉落的情况下连续旋转物体达到多个目标的能力),我们至少可以达到与其相当的性能。
2) 对于它所困难的物体,包括大象、兔子、鸭子、茶壶和龙,我们可以超越它,在生存角度方面取得更好的性能。
3) 我们在旋转具有挑战性长宽比(最高5.33)和困难物体-手比例(即长物体如 Lego 腿和小塑料蔬菜模型,图 1)方面展示了优越性。然而,Visual Dexterity 没有展示这种能力。

*图 27: 基线(UAN 和 ASAP)和消融版本(Joint-Wise(无载荷)和 Whole Hand(有载荷))失败案例的案例研究。*

*图 28: 定性 "Sim-to-Sim" 评估。左:Genesis 中的结果。右:MuJoCo 中的结果。*

与 ASAP 和 UAN 的比较(详细版)。
我们在 sim-to-sim 和 sim-to-real 设置中对两种著名的 sim-to-real 迁移方法进行了评估。考虑到收集带物体状态的真实世界数据的困难以及其原始数据收集策略不考虑物体影响的事实,我们通过使用与带载荷数据收集相同的手腕配置回放 policy 动作展开,在真实世界中收集了24,000条无物体手轨迹。
之后,我们在相应的无物体手仿真设置中训练动力学补偿器。然后使用此补偿器微调原始 policy。
我们使用仅手部训练惩罚对补偿器训练进行 reward:$r^{\text{compensator}}=-\|\mathbf{q}^{\text{ref}}_{t}-\mathbf{q}\|_{2}$,其中 $\mathbf{q}_{h}^{\text{ref}}$ 和 $\mathbf{q}_{t}$ 分别是参考关节状态和当前关节状态。

虽然我们最初打算在表 4 和表 5 涵盖的所有设置中进行全面比较,但我们发现这些基线方法产生的 policy 在真实世界中无法工作。它们无法旋转最简单的圆柱体物体。典型的失败模式是机器人要么紧紧抓住物体不动,要么在一次奇怪的扰动后失败(图 27 (A))。(展示这些失败的视频可在我们的[项目网站](https://meowuu7.github.io/DexNDM/)上获取。)

值得注意的是,policy 微调过程确实取得了令人满意的结果。因此我们假设是 OOD 问题导致了这一点:仅在无物体手的动力学上训练的补偿器,在 policy 必须处理物体在旋转过程中引入的新动力学时会失败。
这一发现突出了在操作的 sim-to-real 策略设计中建模物体动力学的关键重要性,这也与消融研究中的发现一致(第 5 节)。

我们还尝试使用我们收集的任务相关的、带物体状态标注的数据集(54条轨迹)训练基线 sim-to-real 方法(ASAP 和 UAN)。然而,第一阶段——补偿器训练——未能收敛;reward 几乎没有改善。我们将此归因于数据集的有限大小和物体状态噪声。

Sim-to-sim 比较总结在表 6 中。

我们的补偿策略对真实世界 transition 的质量也显示出更好的抗性。如图 27 所示,我们的消融版本"Joint-Wise (w/o Load)"通过无物体手回放数据训练动力学模型,其数据量甚至小于训练 UAN 和 ASAP 所用的量,可以将基本圆柱体物体旋转至少一圈,尽管其最终性能甚至无法超越 base policy。然而,上述两种策略在此任务中完全失败。由于它们会使用补偿器来微调 base policy,其最终 policy 的性能对学到的补偿器的质量非常敏感。因此,只有当学到的补偿器质量非常高且泛化非常好时,其微调才能取得令人满意的结果。否则,最终 policy 可能完全失败,因为它们是在"错误"的动力学下学习的。然而,我们通过将 base policy 与学到的 residual policy 一起使用来补偿。有了好的 base,最终性能至少不会完全失败。

"Sim-to-Sim"。
我们在 Genesis 中通过使用30,000个环境运行统一 policy 的评估来收集数据。我们使用圆柱体收集数据。我们在每个圆柱体实例上运行评估,最大评估试验数设为1,500,000。我们使用所有展开数据训练 joint-wise neural dynamics model(使用 Isaac Gym 中的 transition 预训练)。训练在八个 A10 GPU 上进行2个 epoch,batch size 为64,大约需要两天。

我们在 MuJoCo 中使用一个环境收集数据。对于每个训练圆柱体实例,我们收集4000条轨迹,总共产生36,000条轨迹。我们使用所有数据训练 joint-wise dynamics model(使用 Isaac Gym 中的 transition 预训练)。

之后,我们训练 residual policy 2个 epoch,大约需要13小时。然后将 residual policy 与原始 base policy 一起部署到目标仿真器。Policy 在 ContactDB 测试物体集上测试。我们使用10个不同的初始抓取展开 policy。报告的值是10次试验中每物体平均结果的均值和标准差。

图 28 展示了有/无我们方法弥合动力学差距时 policy 性能的定性比较。

"Sim-to-Sim" 比较设置。
我们使用相同的数据收集策略在每个仿真器中收集 transition。区别在于只保留成功的展开,在 Genesis 中产生3,280,673条轨迹,在 MuJoCo 中产生23,650条轨迹。这些轨迹用于训练 ASAP 和 UAN 对应的动作补偿器。对于 ASAP,我们使用全手公式,不同于我们在 ASAP 的 sim-to-real 设置中使用的每手指补偿器。我们用跟踪物体状态和手部状态的 reward 训练 policy:$r^{\text{compensator}}=-k_{h}\|\mathbf{q}^{\text{ref}}_{t}-\mathbf{q}\|_{2}-k_{o}\text{ang\_diff}(\mathbf{o}^{\text{ref}}_{t},\mathbf{o}_{t})$,其中 $\mathbf{q}_{t}^{\text{ref}}$、$\mathbf{o}_{t}^{\text{ref}}$ 和 $\mathbf{o}_{t}$ 分别是手部参考关节状态、物体参考朝向和物体当前朝向。$k_{h}$ 和 $k_{o}$ 是平衡手部和物体跟踪的系数。$k_{h}$ 设为1.0。而 $k_{o}$ 使用课程调度。它首先设为小值,即0.001。我们使用第一个环境的重置次数来计数重置步骤。在前10个重置步骤中,$k_{o}$ 保持初始值。从此开始直到第200个重置步骤,$k_{o}$ 线性增加到2.0。
补偿器训练完成后,我们基于它调优 policy。然后将调优后的 policy 部署到目标仿真器。
我们采用与我们方法相同的评估策略。

*图 29: 带载荷的自主真实数据收集设置。(A) 装满软球的大箱子。(B) 将物体绑到三个指尖以避免物体掉落并向手添加外部物体影响。(C) 将物体绑到两个指尖,通过这些物体之间的碰撞向手添加外部影响。(D) 添加支撑桌以避免物体掉落。*

*图 30: 真实世界实验硬件设置。*

抓取姿态生成。
我们以"Palm Down"朝向生成抓取姿态,用于全手腕朝向旋转训练。详情请参阅补充材料中的代码('DexNDM-Code/RL/README.md')。LEAP 手的标准关节位置(qpos),我们从中采样随机噪声来生成抓取姿态,设为 [1.244, 0.082, 0.265, 0.298, 1.163, 1.104, 0.953, -0.138, 1.096, 0.005, 0.080, 0.150, 1.337, 0.029, 0.285, 0.317]。

真实世界硬件设置。
我们使用 LEAP 手 (Shaw et al., 2023) 和 Franka 机械臂进行真实世界实验(图 30)。
我们使用20Hz的位置控制。
位置增益和阻尼系数分别设为800和200。

真实世界数据收集设置。
为在最小化人工干预的同时收集具有不同载荷的真实世界 transition 数据,我们开发了几种策略,如图 29 所示。

其中,"Chaos Box"(装满球的箱子)被证明最为有效。其设置很简单:将箱子放在桌子上,打开它,将机器人的手以期望的朝向放入其中。关键的是,这种方法在数据收集期间完全自主运行,不需要人工干预。这种设置确保了与载荷的持续交互,因为机器人的手始终与球接触。轻质球不断变化的位置提供了多样且持续的载荷范围。此外,球的可变形表面确保这些交互不会损坏机器人硬件。系统的自主性使我们能够在晚间启动数据收集并让它在无人看管的情况下通宵运行。

Chaos Box 的一个关键局限是由于机械臂的运动学约束,无法在手掌朝上的朝向下收集数据。为解决这个问题,我们开发了第二种设置,用绷带将球固定在机器人三根手指上(图 29 (B))。与 Chaos Box 类似,这种方法一旦启动就自主运行。然而,绑球需要时间。
一个缺点是球的固定位置导致扰动模式的多样性较低。

另外两种方法被探索但最终未被采用(图 29 (C,D))。一种是将物体附着在手指上 (C),但这不可靠,因为物体可能掉落,需要手动重新附着。另一种使用支撑桌 (D),但物体经常移出机器人手的工作空间,需要人工干预来重新定位。

机械手尺寸。
我们将手的尺寸定义为指尖跨度:对于 D'Claw 手,是对角指尖之间的距离(19.10cm);对于 Allegro 和 LEAP 手,是食指和小指指尖之间的距离(分别为10.05cm和9.50cm)。

真实世界 Transition 数据收集。我们通过回放在仿真中展开的动作轨迹来收集真实世界 transition 数据。每个回合包含400步。动作以20Hz在硬件上执行。收集一条完整回合的轨迹大约需要20s。
我们在所有六种测试手腕朝向下收集 transition,即 palm up、palm down、thumb up、thumb down、base up 和 base down。在每种朝向下,我们收集4,000条 transition 轨迹。
更详细地说,我们从所有 oracle policy 在对应手腕朝向下的展开中均匀随机选择4,000条轨迹。
我们使用"Chaos Box"系统收集 transition。

消融研究的实验设置。
在消融研究中比较不同模型的真实世界性能时,我们将手保持在手掌朝下的朝向,并在三个代表性物体上测试 z 轴旋转性能,包括一个规则圆柱体、一个具有更高长宽比的圆柱体和一个不规则物体。
我们在这个特定手腕朝向和旋转方向下展开旋转规则圆柱体的 policy 来构建仿真数据集,该数据集由937,275条轨迹组成,每条有400个 transition 步。

真实世界数据收集。我们通过 Chaos Box 设置(图 29 (A))收集 transition 数据。我们在真实世界中回放在仿真中展开的动作轨迹来收集数据。
我们收集了4,000条轨迹,共产生1,600,000个 transition。此外,我们通过在真实环境中部署 policy,在 thumb up 朝向下使用5cm尺寸立方体收集了20条成功旋转轨迹(即物体在整个回合中不掉落)作为分布外测试数据。

任务相关数据收集。
我们使用三个物体每个物体收集了1小时的数据:5cm x 5cm x 5cm 立方体、Stanford Bunny 和圆柱体(半径5.5cm,长度5.5cm)。总共分别获得了111、87和54条立方体、圆柱体和 Stanford Bunny 的轨迹。

通过基础波形收集。
我们使用正弦波收集了2,000条轨迹,使用方波收集了1,000条轨迹,使用高斯噪声收集了1,000条轨迹。使用正弦波收集轨迹时,我们随机选择一个关节发送信号,同时保持其他关节固定。具体来说,我们将其他关节固定在其角度范围的中点。对于 LEAP 手,在固定其他关节时驱动 mcp link 到 pip link 之间的关节会导致自碰撞。因此在回放轨迹时我们不会选择这些关节。我们使用形式为 $f(t)=\sigma\sin(2\omega t)$ 的正弦波。在每次数据收集开始时,我们从均匀分布中采样 $\sigma$ 和 $\omega$,即 $\sigma\sim\mathcal{U}(0.5,1.0)$,$\omega\sim\mathcal{U}(0.2,0.5)$。
使用方波时,我们使用 $g(t)=A*\text{sign}(\text{sin}(2*\omega*t))$,其中 $A\sim\mathcal{U}(0.5,1.0)$,$\omega\sim\mathcal{U}(0.2,0.5)$。
我们向方波添加高斯噪声来收集剩余1,000条轨迹,即 $\hat{g}(t)=A*\text{sign}(\text{sin}(2*\omega*t))+\epsilon$,其中 $\epsilon\sim\mathcal{N}(0,0.01)$。

动力学模型训练。预训练动力学模型通过利用相同的模型架构拟合仿真中展开的轨迹获得。
然后我们在真实世界数据上直接调整模型权重进行微调。
从4000条训练轨迹中按 train:eval = 9:1 的比例分出评估数据集。具有最佳评估损失的模型然后用于训练 residual policy 模型。
我们在 OOD 测试数据集上报告最终结果作为泛化性能。
我们在仿真数据上训练 residual policy 1个 epoch,通常使用八个 A10 GPU 大约需要10小时。

*图 31: Quest 3。我们使用右控制器的姿态远程操控机械臂,左控制器的姿态指定期望的旋转轴。我们还提供了按钮控制模式,通过左控制器上的 X、Y 和 LG 按钮将旋转限制为三个固定轴。*

用于复杂灵巧操作数据收集的远程操控系统。
我们展示了旋转 policy 的一个重要应用:具有手内旋转功能的用于复杂灵巧操作任务的远程操控系统。我们通过将 policy 与 Quest 3 头盔配对来实现(图 31)。利用手内旋转,该系统完成了需要精细手指协调的复杂任务——传统远程操控系统 (Ding et al., 2024; Cheng et al., 2024) 在这些场景中通常会遇到困难。

我们适配了 BunnyVisionPro (Ding et al., 2024) 用于 Franka 机械臂远程操控。
机械臂由 Quest 3 右手控制器控制,我们通过 [oculus_reader](https://github.com/rail-berkeley/oculus_reader) 获取控制器状态。
我们使用左控制器的朝向来定义旋转轴,并降低其短轴方向分量的权重以减少从姿态推断轴时的误差。在实践中,这种基于朝向的指定不太直观,因此我们引入了按钮控制模式,通过按下左控制器上的 X、Y 或 LG 按钮来选择旋转轴。虽然这将可用轴限制为三个,但我们发现它对单个任务来说是足够的;例如,灯泡的组装和拆卸可以使用 z、-z 和 -y 旋转模式完成。

所有手部动作(包括抓取)均由 policy 控制。我们将机械手初始化为默认姿态。要抓取物体,我们接近它并激活旋转 policy。在给定初始张开手的观测条件下,policy 输出一个动作序列,使手指合拢包围物体以实现稳固抓取。

## 附录 D 相关 Sim-to-Real 工作的讨论

不匹配的物理参数、物理模型中的差异以及执行器和接触动力学中的大量未建模效应阻碍了将仿真中训练的 policy 成功迁移到真实世界。

缩小这一差距的努力主要分为四类方法:1) Domain Randomization (DR) 扩展训练环境的分布以训练鲁棒的 policy,期望其在不同环境中良好运行 (Loquercio et al., 2019; Peng et al., 2017; Tan et al., 2018; Yu et al., 2019; Mozifian et al., 2019; Siekmann et al., 2020; Sadeghi & Levine, 2016)。2) System Identification (SysID) 通过从真实数据估计关键物理参数,以有原则且可解释的方式对齐仿真器动力学与真实世界 (An et al., 1985; Mayeda et al., 1988; Lee et al., 2023; Sobanbabu et al., 2025)。
3) Adaptive Policy(自适应策略)根据从真实世界反馈中隐式辨识的真实世界动力学在线调整 policy。
4) 基于神经网络的真实世界建模(Neural-based Real World Modeling)学习真实动力学以帮助 policy 迁移 (He et al., 2025; Fey et al., 2025; Deisenroth & Rasmussen, 2011; Shi et al., 2018; Hwangbo et al., 2019)。

作为流行的标准策略,DR 需要启发式设计 (Sobanbabu et al., 2025) 来找到合适的随机化范围。
虽然具有泛化性和可解释性,但 SysID 的上限受限于待辨识参数的覆盖范围。
为了成功适应,训练环境应覆盖广泛的分布,这通常通过 DR 实现。当真实世界动力学无法通过随机化仿真环境来覆盖时,这限制了其有效性。

凭借对齐所有类型差异的潜力,通过建模真实世界动力学来引导 policy 迁移具有最高的能力上限,使其成为我们工作的重点。
一种方法是利用神经网络执行系统辨识,学习残差动力学(residual dynamics)或表示 (Shi et al., 2018; O'Connell et al., 2022),随后开发基于模型的控制器(图 2 (A))。
对于涉及更高自由度(DoFs)和更复杂动力学的系统,学习支持控制器优化的全面动力学模型是困难的。
另一种策略是通过学习 delta function(差值函数) (He et al., 2025; Fey et al., 2025) 来弥合现有仿真器与真实世界之间的差距,随后进行 policy 微调(图 2 (B))。

然而,直接将这些方法扩展到灵巧操作(具有丰富的、快速变化的、作用于运动物体上的接触)是不可行的。
主要挑战在于收集能够覆盖广泛任务分布的高质量真实世界 transition 数据,从而反映任务执行期间的动力学。这通过回放波形(例如正弦波)或展开 policy 来实现——在我们的设置中两者都不能工作。

基于波形的收集不可行:被操纵物体扩大了 transition 空间并施加时变载荷,产生与无物体状态不同的动力学(见附录 A.3)。由于参数化波形无法可靠地在空中操纵物体,它们必须在没有物体的情况下运行,对手内动力学的覆盖不佳。跨多样物体的 on-policy 展开成本高昂且不可扩展——需要频繁的人工重置(将物体放回手中),数据偏向简单物体,覆盖范围限制在 policy 展开分布中,且质量低(不完美的 policy)。

将其方法扩展到操作还需要建模交互动力学,这不可避免地涉及建模物体。有两种方法来建模物体:1) 将物体显式包含在动力学系统中。实现这一点需要收集带物体状态标注的真实世界 transition 轨迹。然而,获取物体状态(例如使用基于视觉的姿态跟踪器如 FoundationPose (Wen et al., 2023)) 是困难的,且在某些情况下不可能。例如,FoundationPose (Wen et al., 2023) 对轴对称、微小和被遮挡的物体不可靠(见第 B.4 节)。此外,物体姿态跟踪结果有噪声。它也非常耗时,需要额外时间启动和频繁的人工干预。使用小的、有噪声的数据集甚至无法使第一阶段(补偿器训练)成功。

另一种策略是将物体建模为时变扰动(time-varying disturbance)。这要求我们 a) 收集带物体载荷的 transition 数据;b) 设法在仿真器中模拟物体对手的影响;c) 训练补偿器仅跟踪手部状态。然而,这几乎不可能,因为复现其影响需要几何形状、初始化和接触演化的近乎完美对齐——在不匹配的动力学下不现实。

我们可以使用什么数据来训练 ASAP 和 UAN 用于灵巧操作?
我们讨论三种选择:(1) 带物体状态标注的 transition——原则上可行但不实际,因为物体状态难以获取且噪声大,在我们的测试中这种小的和有噪声的数据无法训练其补偿器;(2) 我们自主收集的带随机物体载荷的轨迹——不适用,因为在仿真器中复制这种物体载荷对手的影响不可行;(3) 无物体手数据——唯一实际的选择,我们在其上训练补偿器以缩小无物体手场景中的动力学差距。因此,在与其方法比较时我们使用无物体手 transition。

*图 32: 每关节状态序列的多项式拟合(阶数=3)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用3阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 33: 每关节状态序列的多项式拟合(阶数=5)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用5阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 34: 每关节平均多项式拟合(阶数=3)误差。*

*图 35: 每关节平均多项式拟合(阶数=5)误差。*

*图 36: 每关节主动力序列的多项式拟合(阶数=3)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用3阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 37: 每关节主动力序列的多项式拟合(阶数=5)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用5阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 38: 每关节平均多项式拟合(阶数=3)误差。*

*图 39: 每关节平均多项式拟合(阶数=5)误差。*

*图 40: 每关节虚拟力序列的多项式拟合(阶数=3)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用3阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 41: 每关节虚拟力序列的多项式拟合(阶数=5)和误差分布(窗口长度=10)。每组两个子图中,左图绘制原始数据序列和使用5阶多项式函数的拟合序列,右图显示拟合误差分布。*

*图 42: 每关节平均多项式拟合(阶数=3)误差。*

*图 43: 每关节平均多项式拟合(阶数=5)误差。*

Generated on Thu Oct 9 12:19:21 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
