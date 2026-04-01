# DexMachina: 面向双手灵巧操作的功能性重定向

Zhao Mandi${}^{1\ *}$, Yifan Hou1, Dieter Fox2, Yashraj Narang2, Ajay Mandlekar2,†, Shuran Song1,†

1斯坦福大学 2英伟达 †共同指导 *部分工作在实习期间完成

## 摘要

我们研究功能性重定向（functional retargeting）问题：从人手-物体演示中学习灵巧操作策略，以跟踪物体状态。我们聚焦于长时间跨度（long-horizon）的双手操作任务，涉及铰接物体，这类任务因动作空间大、时空不连续性、以及人手与机器人手之间的形态差距（embodiment gap）而极具挑战。我们提出 DexMachina，一种新颖的基于课程学习的算法：核心思想是使用强度逐渐衰减的虚拟物体控制器——物体最初被自动驱动至目标状态，策略在运动和接触引导下逐步学会接管操作。我们发布了一个包含多种任务和灵巧手的仿真基准测试平台，并表明 DexMachina 显著优于基线方法。我们的算法和基准测试平台支持对不同硬件设计进行功能性比较，我们基于定量和定性结果给出了关键发现。随着灵巧手开发的蓬勃发展，我们希望本工作能为识别理想的硬件能力提供有用的平台，并降低未来研究的门槛。更多内容和视频请见 project-dexmachina.github.io

关键词：灵巧操作、强化学习、基于仿真的学习

*图1：功能性重定向。我们研究功能性重定向问题，目标是将人手演示重定向为功能性的灵巧机器人策略，使其操作物体沿演示轨迹运动。我们提出的算法 DexMachina 能够从单个人手演示出发，在多种现有灵巧手形态上实现功能性重定向，涵盖各类铰接物体。*

## 1 引言

灵巧机器人手因与人手相似而让人期待实现人类水平的灵巧性。然而现实中存在诸多硬件和算法挑战，制约着灵巧操作的进展。先前基于学习的方法在相对简单和短时间跨度的任务上取得了成功，但往往受限于人工奖励工程 [1, 2] 或因人手与灵巧手之间的形态差距而导致昂贵的数据收集 [3, 1]。

人手因此是学习引导的天然来源。在本工作中，我们将从人类学习的问题形式化为以任务能力为重点。我们将该问题定义为功能性重定向：给定一个人手演示，目标是学习灵巧手策略，使其能够操作物体沿演示轨迹运动（见图1）。这与运动学重定向（kinematic retargeting）[3] 不同，后者生成类人运动但不保证可行性。对于长时间跨度的双手操作演示和铰接物体，该问题更具价值——这些操作涵盖了人类日常活动的重要部分，但带来若干关键挑战：高维动作空间下的探索困难；复杂的接触序列要求稳定精确的手部运动；形态差距使得人手运动无法直接映射为可行的机器人动作，限制了模仿数据收集的可扩展性。

为应对这些挑战，我们提出 DexMachina（注：Deus ex machina，"机械降神"，指一个看似无解的问题被外力巧妙解决——正如我们的算法中物体最初自行运动，策略逐渐学会接管，故名 DexMachina），一种新颖的基于课程学习的强化学习算法，用于功能性重定向。成功操作物体往往需要精确的双手协调（例如在空中打开华夫饼机，见图1），但朴素方法常常陷入早期失败或次优动作。这促使我们设计课程学习机制，让策略在更稳健的环境中进行探索。核心思想是使用虚拟物体控制器——它们施加控制力将物体驱向演示轨迹——以及辅助运动和接触奖励，引导策略在虚拟控制器强度衰减的过程中逐步学会任务策略。策略首先学习模仿人类运动（无需担心任务失败），然后随着虚拟控制器的消退逐步接管操作。

尽管在开发新型手部和传感能力方面持续投入 [4, 5, 6, 7, 8, 9]，目前仍缺乏标准化且易用的评估基准。为此，我们构建了一个包含6种灵巧手和5种铰接物体 [10] 的仿真基准测试平台，提供了一个统一的测试环境，新的手部和任务可以方便地添加并快速评估。在此基准上，我们通过实验表明 DexMachina 显著优于基线方法，且成功适用于多种灵巧手、铰接物体和长时间跨度演示。

借助有效的功能性重定向算法和评估基准，现在可以跨不同硬件进行功能性比较：基于策略学习性能，我们获得了对灵巧手功能性和从人类引导中学习的就绪程度的有意义度量。这种比较具有通用性和易用性：我们的算法不需要针对特定手部的适配，任务环境运行快速且易于定制。随着机器人手硬件开发的蓬勃发展，我们希望这种功能性比较有助于在采购和设计新型手部时做出明智决策。

我们的贡献总结如下：

$\bullet$ 我们研究了功能性重定向问题，从人手-物体演示中学习可行的灵巧操作策略。我们提出 DexMachina，一种基于虚拟物体控制器课程学习和运动、接触引导的新算法。

$\bullet$ 我们推出了 DexMachina 基准测试平台，包含6种精选灵巧手资产和5种铰接物体，用于评估不同的功能性重定向算法和机器人手设计。

$\bullet$ 我们证明 DexMachina 在多种机器人手和任务上实现了最先进的学习性能。我们的仿真环境和学习算法将开源以促进未来研究。

## 2 相关工作

**用于灵巧操作的强化学习。** 强化学习（RL）已被用于灵巧操作任务，如手内物体旋转 [1, 11, 12, 13, 14] 和单手抓取 [2, 15, 16, 17, 18, 19]，但实现更复杂、更长时间跨度的操作仍然充满挑战，因为为此类任务设计引导探索的奖励函数负担沉重。基于模型的方法已被应用于球运球 [8] 和魔方翻转 [9] 等任务，但它们对每个物体和任务都需要仔细的工程设计。在我们的工作中，我们旨在研究双手长时间跨度任务，这些任务中难以指定具体目标或设计引导探索的RL奖励。这促使我们使用人手演示，既作为目标规范，也提供如何完成任务的引导。仿真是训练灵巧手策略的常用工具 [20]，因为在真实硬件上运行RL的探索成本很高 [21]。我们的仿真基准支持在多种灵巧手和由人手演示数据定义的多样化任务上进行评估，这与现有的灵巧操作RL基准 [22, 23] 不同。

**用于灵巧操作的模仿学习。** 模仿学习（IL）是RL的一种有吸引力的替代方案，因为使用演示可以减轻或消除探索负担，但它可能需要精确的机器人动作数据，这对灵巧手来说获取困难。大多数现有方法 [3, 24, 25, 26, 27, 28] 需要为特定机器人手的形态设置定制的遥操作系统。人手数据（如视频）是另一种数据来源。先前工作已使用人手数据学习粗略的抓取可供性 [29]、改进重定向 [30]，或将人手数据与遥操作数据联合训练 [31, 32]，但这些方法局限于短时间跨度操作（主要是抓取）。相比之下，我们的工作假设每个任务可获得一个已跟踪的手-物体演示，并使用该演示引导RL训练。类似方法已被用于仿人机器人运动 [33]、简单手部操作 [34] 和短时间跨度灵巧操作任务 [35, 36]。

**课程学习。** 在基于优化的运动规划中，从松弛的物理约束开始热启动优化以获得更好的收敛解是常见做法 [37, 38, 39]。这种从简单到困难的课程学习思想已被RL方法采用 [40, 41]。一些先前工作使用此方法松弛物理约束，如在接触前允许施力 [42] 或松弛重力、摩擦和约束求解器参数 [36]。我们的方法使用物体动力学上的课程学习，让智能体随时间逐步学会如何操作物体（图2）。

## 3 功能性重定向问题形式化

我们将功能性重定向问题定义如下：给定一个物体 $\eta$、一个人手-物体演示序列 $\mathcal{D}^{\eta}$ 和一对灵巧机器人手 $\zeta$，目标是学习一个能操作物体以跟踪演示物体状态的机器人策略。更形式化地，一个人手演示 $\mathcal{D}^{\eta}=\{G,H\}$ 包含 $T$ 个时间步的密集跟踪物体状态 $G$ 和手部姿态 $H$。我们聚焦铰接物体，因此物体状态包括部件位姿和旋转关节角度值。在任意时间步 $t$，给定实际物体状态 $\hat{g_{t}}$（位置、旋转和铰接角）和来自演示的目标物体状态 $g_{t}=\{g_{t}^{P},g_{t}^{R},g_{t}^{J}\}$，我们将距离函数记为 $F$（计算旋转、位置和铰接关节误差）。对于 $(\eta,\zeta)$ 的学习策略应最小化所有时间步的累积跟踪误差：$\pi_{\theta}^{\eta,\zeta}=\text{argmin}_{\theta}\sum_{t=1}^{T}(F(\hat{g_{t}},g_{t}))$。

## 4 方法

**概述。** 我们提出 DexMachina，一种用于功能性重定向的基于课程学习的RL算法。在§4.1中，我们首先介绍鼓励物体跟踪但不足以实现有效策略学习的任务奖励。在§4.2中，我们从演示中提取运动和接触信息，用于定义残差动作和辅助奖励。虽然这些组件改善了学习效果，但在复杂的长时间跨度任务中仍有不足。这促使我们在§4.3中提出课程学习策略，引入基于虚拟物体控制器的自动课程学习，以在不同灵巧手上实现高效的功能性重定向。

*图2：DexMachina 概览。DexMachina 是一种用于功能性重定向的基于课程学习的RL算法。我们处理密集跟踪的人手演示数据以提取参考机器人关节和关键点（粉色球体）以及物体网格顶点上的近似接触位置（绿色球体），用于定义任务奖励之外的辅助奖励。然后我们引入基于虚拟物体控制器的自动课程学习，控制器最初自行驱动物体沿演示运动，随后在RL训练过程中逐步衰减，让策略学会接管操作。*

### 4.1 RL环境与任务奖励

我们训练强化学习（RL）策略来实现功能性重定向任务。RL环境通过将一个演示 $\mathcal{D}^{\eta}$ 和一组双手灵巧机器人手 $\zeta$ 配对来构建。在每个时间步 $t$，将 $G_{t}=\{g_{t}^{P},g_{t}^{R},g_{t}^{J}\}$ 记为时间步 $t$ 时记录的物体位置、旋转和关节角度，$\hat{G_{t}}=\{\hat{g}_{t}^{P},\hat{g}_{t}^{R},\hat{g}_{t}^{J}\}$ 为各项对应的物体实际状态。任务奖励 $r_{\text{task}}$ 是衡量每个状态分量精度的三项之积，鼓励均衡学习 [35]。形式化表示：

$$d_{\text{pos}}=||\hat{g_{t}}^{T}-g_{t}^{T}||_{2};\ d_{\text{rot}}=2\cos^{-1}(|\langle\hat{g_{t}}^{R},g_{t}^{R}\rangle|);\ d_{\text{ang}}=||\hat{g_{t}}^{J}-g_{t}^{J}||_{2}$$

$$r_{\text{task}}=r_{\text{pos}}*r_{\text{rot}}*r_{\text{angle}}=\exp(-\beta_{\text{pos}}d_{\text{pos}})\exp(-\beta_{\text{rot}}d_{\text{rot}})\exp(-\beta_{\text{ang}}d_{\text{ang}})$$

其中 $\beta_{\text{pos}}$、$\beta_{\text{rot}}$ 和 $\beta_{\text{ang}}$ 为标量权重，控制每个分量的期望误差尺度。

### 4.2 动作形式化与辅助奖励

虽然任务奖励指定了期望的物体状态，但它没有提供如何实现这些状态的有用信息。为此，我们（1）提出混合动作形式化，约束腕部动作空间使其与人类演示者更加一致；（2）定义辅助奖励，引导策略遵循人类的手-物体交互策略。作为预备工作，我们首先对演示数据 $\mathcal{D}^{\eta}$ 进行预处理，提取相关的运动和接触信息。

**数据预处理。** 给定包含 $T$ 个时间步、$N$ 个物体部件的 $\mathcal{D}^{\eta}$，以及具有 $J$ 个驱动关节和 $K$ 个碰撞链接的灵巧手 $\zeta$，我们首先运行运动学重定向算法 [3]，将灵巧手姿态与人手运动匹配。然后我们获得：

1. **考虑碰撞的运动学重定向关节值** $\mathcal{Q}\in\mathbb{R}^{T\times J}$ 和**参考关键点** $\mathcal{X}\in\mathbb{R}^{T\times K\times 3}$。通过在仿真中重放重定向结果并记录 (1) 实际关节值和 (2) 灵巧手链接的3D关键点位置。为消除物体穿透，我们在仿真中将重定向的关节值作为软控制目标重放，同时保持物体固定——详见附录A.2。

2. **近似的手-物体接触。** 虽然运动学重定向产生类人的灵巧手姿态，但这些运动往往无法操作物体。因此我们提取接触信息作为物体交互的额外引导。我们使用基于距离的近似方法来获取特定灵巧手链接应在何时何处与特定物体部件接触（详见附录A.4）。结果为近似接触位置 $C\in\mathbb{R}^{(T\times N\times K\times 3)}$ 和指示物体部件与手部链接对是否存在有效接触的掩码 $M\in\mathbb{R}^{(T\times N\times K)}$。

**混合动作输出。** 给定重定向关节结果 $\mathcal{Q}$，我们使用6自由度腕部关节的关节值作为基础动作，在此基础上叠加策略输出的残差动作。其余手指关节使用按关节限位归一化的绝对动作（完整细节见附录A.3）。这种形式化有效约束了策略的动作空间，我们的实验表明它显著提升了学习效率。

**运动模仿奖励。** 为鼓励类人手部运动，我们取运动参考关键点 $\mathcal{K}$ 和重定向关节值 $\mathcal{Q}$，定义 (1) 基于关键点匹配的运动模仿奖励 $r_{\text{imi}}$，(2) 基于与参考关节角度距离的行为克隆奖励 $r_{\text{bc}}$。形式化表示：

$$r_{\text{imi}}=\frac{1}{K}\sum_{i=1}^{K}\exp({-\beta_{\text{imi}}||\hat{x_{i}}-x_{i}||_{2}});\ r_{\text{bc}}=\frac{1}{J}\sum_{i=1}^{J}\exp({-\beta_{\text{bc}}||\hat{q_{i}}-q_{i}||_{2}})$$

其中每个 $(\hat{x_{i}},x_{i})$ 表示第 $i$ 个关键点的实际位置和参考位置，$(\hat{q_{i}},q_{i})$ 表示第 $i$ 个关节的实际值和重定向值。

**接触奖励。** 我们读取每个手部链接与每个物体部件之间的接触位置，通过将策略接触与相应的演示接触匹配来计算接触奖励。对于手的每一侧，我们将策略和演示的接触位置及有效性掩码分别记为 $C,\hat{C}\in\mathbb{R}^{N\times K\times 3}$，$M,\hat{M}\in\mathbb{R}^{N\times K\times 1}$。我们计算由有效性掩码掩蔽的 $L_{2}$ 接触距离，并用它定义接触奖励 $r_{\text{con}}$：

$$D=\|C-\hat{C}\|_{2}\in\mathbb{R}^{N\times K};\ \text{set}\ D^{(i,j)}=\begin{cases}d_{\text{max}},&\text{if }M^{(i,j)}_{\text{demo}}\neq M^{(i,j)}_{\text{policy}}\\ 0,&\text{if }M^{(i,j)}_{\text{demo}}=M^{(i,j)}_{\text{policy}}=0\end{cases} \tag{1}$$

$$r_{\text{con}}=\frac{1}{2NK}(\sum_{i=1}^{N}\sum_{j=1}^{K}\exp(-\beta_{\text{con}}D^{(i,j)}_{\text{left}})+\sum_{i=1}^{N}\sum_{j=1}^{K}\exp(-\beta_{\text{con}}D^{(i,j)}_{\text{right}})) \tag{2}$$

最终RL奖励为上述各项的加权和：$r_{t}=\lambda_{\text{task}}r_{\text{task}}+\lambda_{\text{imi}}r_{\text{imi}}+\lambda_{\text{bc}}r_{\text{bc}}+\lambda_{\text{con}}r_{\text{con}}$。精确权重和额外奖励细节见附录A.4。

*算法1：DexMachina 课程学习*

### 4.3 基于虚拟物体控制器的自动课程学习

**动机。** 上述奖励项和动作约束有时足以应对短时间和简单的任务，但在具有复杂接触的长时间跨度片段上表现不佳。策略往往经历灾难性的早期失败：例如，双手抬起一个盒子后，策略可能无法预判一只手需要在空中重新定位以打开盖子，而另一只手需要调整以实现单手抓握。策略会尝试不同动作，其中大多数会导致盒子掉落并终止回合。

这促使我们提出课程学习方法，让策略在更稳健的环境中探索不同策略。核心思想是使用虚拟物体控制器：它们驱动物体自行跟随目标轨迹，使策略能够在整个序列中学习并避免短视策略。

**虚拟物体控制器。** 我们将演示状态 $G$ 作为控制目标，施加虚拟弹簧-阻尼约束使物体沿目标轨迹运动。最初，虚拟控制器承担大部分物体运动；随时间推移，控制器的影响逐渐减弱，要求策略承担更多控制以完成任务。控制器通过仿真中的特权信息（privileged information）实现。每个物体配备六个虚拟1自由度关节用于基础位姿，和一个1自由度关节用于铰接，所有关节由PD控制器 [43] 驱动。在每个时间步，这些控制器基于当前物体状态与来自演示的控制目标之间的误差施加虚拟力。控制强度由增益参数（$k_{p}$, $k_{v}$）参数化，随时间衰减以实现向学习策略的结构化交接。

**课程调度。** 算法1描述了我们提出的课程学习方案。在课程训练开始时，我们设置高虚拟控制器增益并采用临界阻尼；然后根据策略的学习进度指数衰减增益，学习进度通过过去奖励的历史记录来追踪。因此，策略在初期将始终获得高任务奖励；由于它接收任务奖励和辅助奖励的加权和，策略在改善运动和接触奖励的同时避免干扰物体轨迹。随后，随着物体控制器减弱，策略逐步学习调整运动以维持高任务奖励。由于辅助奖励使用远小的权重，策略可以偏离早期学到的参考手部运动，以优先获得高任务奖励。

## 5 实验

**实验设置。** 我们使用 ARCTIC [10] 的手-物体数据（见§A），包含5种铰接物体 [44] 和7个包含多样运动序列的演示（拾取和重定向物体、开关盖子等）。我们在短时间跨度（先前工作 [35] 使用的）和长时间跨度演示上评估算法。我们精选了6种开源灵巧机器人手模型的资产，具有不同尺寸和运动学设计。我们使用 Genesis [45] 进行物理仿真，PPO [46, 47] 作为基础RL算法。策略对所有手和任务使用相同结构的基于状态的输入观测空间，并同时控制双手。RL训练细节见附录§B.1，评估设置见§B.4。

**基线方法。** 由于物理仿真和训练配置的各种差异，我们在自己的训练框架中重新实现了基线方法，并进行了若干适配以确保公平比较——实现细节见§B.3。我们与以下方法进行比较：

1. **仅运动学（Kinematics Only）。** 直接使用运动学重定向 [3] 结果作为策略控制器目标。

2. **ObjDex [35]。** 学习一个高层腕部规划器用于腕部基础动作，和一个使用任务奖励和与我们相同混合动作的低层策略。我们通过在原始结果使用的相同演示上展示改善的性能来验证我们的重新实现。

3. **任务 + 辅助奖励（无课程学习）。** 为评估我们提出的课程学习的效果，我们仅使用我们提出的运动模仿和接触奖励运行RL训练。为公平比较，所有训练超参数与课程学习设置相同。

4. **ManipTrans [36]。** 一项同期工作，使用接触力奖励和基于误差阈值与物理参数的课程学习来微调运动模仿模型。由于原始方法在不同物理模拟器中对刚性物体进行评估，我们重新实现了其提出的课程学习，同时使用我们的混合动作和辅助奖励项。

**实验概览。** 我们展示的实证结果包括 (1) 评估 DexMachina 相对于基线和无课程学习设置的有效性（§5.1）；(2) 对方法关键组件的消融实验（§5.2）；(3) 展示 DexMachina 在各种灵巧手形态上的适用性及作为比较不同手部设计的评估框架的实用性（§5.3）。

*图3：DexMachina 核心结果。我们在四种代表性灵巧手上评估 DexMachina，配合七个包含不同物体和运动序列的演示。我们比较了直接重放运动学重定向结果（"仅运动学"）、仅用任务奖励训练（"任务奖励（ObjDex）"，即我们对 ObjDex [35] 的重新实现）、同时用任务和辅助奖励训练（"任务 + 辅助奖励"）、以及使用我们提出的辅助奖励和课程学习（"我们的方法"）。除少数例外，DexMachina 展示了相对于基线方法的明显改善，尤其在具有更复杂运动的长时间跨度任务上。*

### 5.1 DexMachina 主要结果

我们在四种代表性灵巧手（Inspire [48]、Allegro [49]、Xhand [50] 和 Schunk [51]）和七个演示片段上评估 DexMachina 及基线方法（可视化见附录C）。每个任务的平均成功率报告在图3中。关键要点如下：

**DexMachina 在所有手和任务上持续提升性能。** 我们重点关注图3中最右侧四列，对应具有复杂运动序列的长时间跨度演示（例如，'Waffleiron-300'要求策略拿起物体、打开和关闭盖子、来回翻转，然后再次打开和关闭盖子，全部在空中完成，见附录§C的可视化）。仅用任务奖励在这些片段上表现不佳；加入辅助奖励（"任务 + 辅助奖励"）在部分任务上有所改善，但收益不一致。相比之下，DexMachina 在使用相同奖励的情况下显著优于无课程学习设置。

仅用任务奖励和混合动作可以在短时间跨度任务上达到合理性能：我们对 ObjDex [35] 的重新实现（图3中的"任务奖励（ObjDex）"）在相同演示上优于其原始报告（图3左三列，详见§B.3）。仅运动学重定向结果无法完成任务（"仅运动学"）——我们的视频定性展示了它们在视觉上与人手对齐良好，但动作只能实现轻微抬起每个物体。

**DexMachina 让策略学习适应硬件约束的任务策略。** 辅助奖励并不总是与最佳任务策略一致，而是作为服务于课程学习的软引导，给予策略探索的灵活性。定性观察表明，策略可能偏离运动和接触引导，学习不同策略：如图4所示，在 Notebook-300 上，XHand 策略遵循人类演示者用左手托住物体、右手合上盖子；然而对于更小、驱动更少的 Inspire Hand，策略学会了用双手稳定物体并合上盖子。在 Mixer-300 上，Allegro Hand 的手指足够长可以轻松合上盖子，但 Schunk Hand 的策略展示了更多的腕部运动来实现相同效果。

### 5.2 DexMachina 消融实验

*图4：DexMachina 针对不同手部的策略。DexMachina 使策略能够学习适应硬件约束的任务策略。我们展示了不同手在相同任务上训练策略展开的快照：左侧展示 XHand 和 Inspire Hand 在 Notebook-300 任务上的表现；右侧展示 Schunk Hand 和 Allegro Hand 在 Mixer-300 任务上的表现。*

**动作消融。** 我们将混合动作形式化与以下方案比较：(1) 所有关节的绝对动作；(2) 约束较少的腕部关节残差动作，其中腕部关节限位设为覆盖整个演示片段的最大运动范围。我们在无课程学习设置下训练，使用任务和手的子集，每种方法平均三个随机种子。结果见图8。虽然所有方法都受益于辅助奖励，但使用更严格的腕部运动约束能获得最佳整体性能。

**课程学习消融。** 在图3中，我们将 DexMachina 与 ManipTrans [36] 进行比较，后者使用基于运动和物体姿态的误差阈值以及重力和摩擦参数的课程学习。我们观察到它相对于无课程学习设置没有明显改善，且训练稳定性较差：在相同的RL迭代预算下，ManipTrans 策略初期实现了高任务奖励，但随着课程推进性能下降且无法恢复。这表明仅衰减物理参数不足以应对铰接物体的长时间跨度任务，这些任务需要更强的引导来完全解决任务，直到策略逐步接管。

### 5.3 手部形态分析

在验证 DexMachina 在各种任务和手上实现功能性重定向之后，我们现在使用算法和基准对不同灵巧手进行功能性比较。我们聚焦§5.1中的四个长时间跨度任务，并在两种额外手——Ability [52] 和 DexRobot Hand 上评估 DexMachina（见图5）。我们讨论以下关键发现：

**更大的、全驱动手在最终性能和学习效率上都更优。** Allegro Hand 尽管外观不那么拟人，但由于其较长的手指提供了手内/空中操作的稳定性，能力出人意料地强。**尺寸相似性不如自由度重要。** 例如，Inspire、Ability 和 Schunk Hand 尺寸相似，但 Schunk 拥有驱动指尖和可折叠手掌，平均性能优于 Inspire 和 Ability。虽然欠驱动手在外观上更接近人手，但学习到的策略却不如更大但更强的手那样类人。因为所有手使用相同的人手运动参考（既作为腕部基础动作又作为运动奖励），策略偏离人类引导的程度取决于其尺寸和运动学约束。因此，Inspire 和 Ability 等手部通常需要不同的策略来完成任务。

*图5：在长时间跨度任务上使用 DexMachina 对所有六种手的完整评估。*

当然，我们的结论受限于所测试的物体和任务：例如，较大的手在操作较小物体（如镊子）时表现不佳。然而，我们的评估框架可以轻松扩展以添加新的灵巧手和测试任务或物体。

## 6 结论

我们提出了 DexMachina，一种用于功能性重定向的基于课程学习的RL算法，核心思想是使用虚拟物体控制器，让策略在运动和接触引导下轻松探索任务策略。在我们包含多样化任务和灵巧手的仿真基准上，DexMachina 显著优于基线方法，并支持不同灵巧手设计之间的功能性比较。我们希望算法和基准环境能为识别理想灵巧手能力提供有用的平台，并降低未来研究的贡献门槛。

## 7 局限性

DexMachina 存在若干关键局限性。

第一，我们的策略使用依赖物理模拟器特权信息的基于状态的输入；这些信息在真实世界中获取困难。该局限可通过基于视觉的RL策略训练 [2] 或更实际地通过蒸馏设置来解决——使用基于状态策略生成的演示数据训练视觉运动策略，如先前工作所展示 [35]。第二，我们的问题形式化假设可获得高质量人手-物体演示数据，这需要物体重建以及人手和铰接物体的精确姿态跟踪。此类数据收集昂贵且需要仔细筛选（例如 ARCTIC [10] 使用带有密集人工标注和后处理的动捕系统）。未来工作可探索扩展数据收集的替代方法：一个方向是利用3D生成模型和重建方法的最新进展。第三，由于我们使用灵巧手的开源资产并估计物理属性（如质量、惯性和碰撞形状），仿真手可能无法捕捉真实硬件的某些动力学和能力。为解决这一问题，需要参照真实硬件进行更仔细的调优；理想情况下，精确的仿真模型应由制造商直接提供。最后，由于缺乏硬件，我们学习的RL策略尚未在真实世界的灵巧手上进行评估。借助社区的投入，我们希望仿真基准能够支持无需物理硬件的可及研究和手部设计评估；此外，我们学习的策略可作为教师策略被蒸馏用于仿真到真实的迁移，先前类似工作已证明了这一点 [35, 36]。

## 致谢

本工作部分由 NVIDIA 和 NSF 资助（奖项号 #2143601、#2037101 和 #2132519）。本文所述观点和结论仅代表作者意见，不应被解读为必然代表资助方的官方政策。作者感谢 NVIDIA 的现任和前任同事：Kelly Guo、Milad Raksha、David Hoeller、Bingjie Tang 在物理仿真环境方面的帮助和算法开发中的深入讨论；以及斯坦福大学 REALab 全体成员对论文初稿提供的有用反馈。

## 参考文献

- [1] Andrychowicz et al. [2020] O. M. Andrychowicz 等. Learning dexterous in-hand manipulation. *The International Journal of Robotics Research*, 39(1):3–20, 2020.
- [2] Lum et al. [2024] T. G. W. Lum 等. Dextrah-g: Pixels-to-action dexterous arm-hand grasping with geometric fabrics, 2024.
- [3] Qin et al. [2023] Y. Qin 等. Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system. In *RSS*, 2023.
- [4] Rakić [1968] M. Rakić. Paper 11: The 'belgrade hand prosthesis'. 1968.
- [5] Loucks et al. [1987] C. Loucks 等. Modeling and control of the stanford/jpl hand. In *ICRA*, 1987.
- [6] Jacobsen et al. [1986] S. Jacobsen 等. Design of the utah/m.i.t. dextrous hand. In *ICRA*, 1986.
- [7] Butterfass et al. [1998] J. Butterfass 等. Dlr's multisensory articulated hand. In *ICRA*, 1998.
- [8] Shiokata et al. [2005] D. Shiokata 等. Robot dribbling using a high-speed multifingered hand. In *IROS*, 2005.
- [9] Higo et al. [2018] R. Higo 等. Rubik's cube handling using a high-speed multi-fingered hand. In *IROS*, 2018.
- [10] Fan et al. [2023] Z. Fan 等. ARCTIC: A dataset for dexterous bimanual hand-object manipulation. In *CVPR*, 2023.
- [11] Handa et al. [2023] A. Handa 等. Dextreme: Transfer of agile in-hand manipulation from simulation to reality. In *ICRA*, 2023.
- [12] Qi et al. [2023] H. Qi 等. In-hand object rotation via rapid motor adaptation. In *CoRL*, 2023.
- [13] Yin et al. [2023] Z.-H. Yin 等. Rotating without seeing: Towards in-hand dexterity through touch. *arXiv:2303.10880*, 2023.
- [14] Chen et al. [2023] T. Chen 等. Visual dexterity: In-hand reorientation of novel and complex object shapes. *Science Robotics*, 2023.
- [15] Caggiano et al. [2023] V. Caggiano 等. Myodex: A generalizable prior for dexterous manipulation. *ArXiv*, 2023.
- [16] Luo et al. [2024] Z. Luo 等. Grasping diverse objects with simulated humanoids. *ArXiv*, 2024.
- [17] Mandikal and Grauman [2021] P. Mandikal and K. Grauman. Dexterous robotic grasping with object-centric visual affordances. In *ICRA*, 2021.
- [18] Zhu et al. [2023] T. Zhu 等. Toward human-like grasp: Functional grasp by dexterous robotic hand. *IEEE TPAMI*, 2023.
- [19] Yuan et al. [2024] H. Yuan 等. Cross-embodiment dexterous grasping with reinforcement learning. *ArXiv*, 2024.
- [20] Rajeswaran et al. [2018] A. Rajeswaran 等. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations, 2018.
- [21] Xu et al. [2022] K. Xu 等. Dexterous manipulation from images: Autonomous real-world rl via substep guidance. *ICRA*, 2022.
- [22] Bao et al. [2023] C. Bao 等. Dexart: Benchmarking generalizable dexterous manipulation with articulated objects. In *CVPR*, 2023.
- [23] Company [2025] Shadow Robot Company. Shadow hand official website. 2025.
- [24] Wang et al. [2024] C. Wang 等. Dexcap: Scalable and portable mocap data collection system for dexterous manipulation. 2024.
- [25] Yang et al. [2024] S. Yang 等. Ace: A cross-platform visual-exoskeletons for low-cost dexterous teleoperation. 2024.
- [26] Cheng et al. [2024] X. Cheng 等. Open-television: Teleoperation with immersive active visual feedback. 2024.
- [27] Shaw et al. [2024] K. Shaw 等. Bimanual dexterity for complex tasks. In *CoRL*, 2024.
- [28] Zhang et al. [2025] H. Zhang 等. Doglove: Dexterous manipulation with a low-cost open-source haptic force feedback glove. 2025.
- [29] Mandikal and Grauman [2020] P. Mandikal and K. Grauman. Learning dexterous grasping with object-centric visual affordances. In *ICRA*, 2020.
- [30] Park et al. [2025] S. Park 等. Learning to transfer human hand skills for robot manipulations, 2025.
- [31] Shaw et al. [2022] K. Shaw 等. Videodex: Learning dexterity from internet videos, 2022.
- [32] Xu et al. [2023] M. Xu 等. Xskill: Cross embodiment skill discovery. In *CoRL*, 2023.
- [33] Peng et al. [2018] X. B. Peng 等. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. *ACM ToG*, 2018.
- [34] Wang et al. [2023] Y. Wang 等. Physhoi: Physics-based imitation of dynamic human-object interaction, 2023.
- [35] Chen et al. [2024] Y. Chen 等. Object-centric dexterous manipulation from human motion data. *ArXiv*, 2024.
- [36] Li et al. [2025] K. Li 等. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning. 2025.
- [37] Mordatch et al. [2012] I. Mordatch 等. Discovery of complex behaviors through contact-invariant optimization. *ACM ToG*, 2012.
- [38] Pang and Tedrake [2021] T. Pang and R. Tedrake. A convex quasistatic time-stepping scheme for rigid multibody systems with contact and friction. In *ICRA*, 2021.
- [39] Pang et al. [2023] T. Pang 等. Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models. *IEEE T-RO*, 2023.
- [40] Chiappa et al. [2024] A. S. Chiappa 等. Acquiring musculoskeletal skills with curriculum-based reinforcement learning. *bioRxiv*, 2024.
- [41] Zhang et al. [2024] H. Zhang 等. ArtiGrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation. In *3DV*, 2024.
- [42] Mao et al. [2025] X. Mao 等. Learning long-horizon robot manipulation skills via privileged action. 2025.
- [43] Franklin et al. [2002] G. F. Franklin 等. *Feedback control of dynamic systems*. Prentice hall, 2002.
- [44] Xu et al. [2025] X. Xu 等. Robopanoptes: The all-seeing robot with whole-body dexterity. 2025.
- [45] Authors [2024] Genesis Authors. Genesis: A universal and generative physics engine for robotics and beyond, 2024.
- [46] Schulman et al. [2017] J. Schulman 等. Proximal policy optimization algorithms, 2017.
- [47] Makoviichuk and Makoviychuk [2021] D. Makoviichuk and V. Makoviychuk. rl-games: A high-performance framework for reinforcement learning, 2021.
- [48] Beijing Inspire-Robots Technology Co. [2025] Inspire hand official website. 2025.
- [49] Robotics [2025] Wonik Robotics. Allegro hand official website. 2025.
- [50] ROBOTERA [2025] ROBOTERA. Xhand1 official website. 2025.
- [51] KG [2025] Schunk SE & Co. KG. Schunk 5-finger hand official website. 2025.
- [52] PSYONIC [2025] PSYONIC. Ability hand official website. 2025.
- [53] Romero et al. [2017] J. Romero 等. Embodied hands: Modeling and capturing hands and bodies together. *ACM ToG*, 2017.
- [54] Makoviychuk et al. [2021] V. Makoviychuk 等. Isaac gym: High performance gpu-based physics simulation for robot learning, 2021.
- [55] Wen et al. [2024] B. Wen 等. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In *CVPR*, 2024.
- [56] Xiang et al. [2018] Y. Xiang 等. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes, 2018.
- [57] Hinterstoisser et al. [2013] S. Hinterstoisser 等. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In *ACCV*, 2013.

## 附录 A 演示数据处理细节

### A.1 ARCTIC 演示选择与整理

我们使用 ARCTIC 数据集 [10] 中手-物体交互片段的子集，该数据集包含铰接物体扫描以及带有跟踪的 MANO [53] 手部姿态和物体状态的交互序列。每个选定片段由物体（如'box'）、标识人类演示者的主体标签（如's01-u01'）和用于裁剪序列到固定长度的（起始, 结束）元组定义，因此使用帧数 $T$ 定义为 $T=(\text{end}-\text{start})$。

**灵巧手资产处理。** 我们实验中所有灵巧手均整理自开源 URDF 模型，并手动编辑添加6自由度腕部关节以实现"浮动手"式腕部驱动。部分灵巧手模型需要额外处理以确保稳定仿真，如手动调整质量或惯性值、运行凸分解以改善碰撞网格质量、以及添加虚拟指尖链接以记录和跟踪关键点位置。对于每种灵巧手，我们手动指定哪些手指链接应与哪些 MANO [53] 手部关节匹配（如拇指对拇指），这是运动学重定向 [26] 算法所要求的。运动学重定向结果还用于控制器增益调优，确保灵巧手控制器足够稳定和快速，能在合理误差内匹配期望的人手运动和速度。

### A.2 考虑物体的重定向后处理

*图6：我们在纯运动学重定向基础上执行改进的重定向方案。*

由于我们使用密集跟踪的人手和物体交互作为演示，纯运动学重定向算法 [26] 对指尖位置的处理会导致与物体频繁穿透，这在策略学习中产生有害的基础动作，以及用于模仿奖励计算的不可行关键点位置。为解决此问题，我们对每对灵巧手运行仿真，在每个演示时间步，将物体固定到目标状态（包括根姿态和物体关节角度），并将重定向关节值设为控制目标。此过程让仿真解决碰撞问题。

然后我们记录实际关节值和关键点用于策略学习。在实现上，此过程可在仿真中方便地并行化，如图6所示。

### A.3 混合动作输出

形式化地，我们使用以下符号：

- $\text{clip}(x,a,b)$：输入值 $x$ 在 $a$ 和 $b$ 之间的逐元素裁剪
- $a_{t}\in\mathbb{R}^{J}$：策略在时间 $t$ 的关节动作输出，裁剪到 $[-1,1]$，即 $a_{t}=\text{clip}(\pi_{\theta}(o_{t}),-1,1)$
- $q_{t}^{(i)}$：第 $i$ 个关节在时间 $t$ 的目标位置
- $\mathcal{I}_{f}\subset\{1,\dots,J\}$：手指自由度对应的索引
- $\mathcal{I}_{w}^{\text{T}}\subset\{1,\dots,J\}$：三个腕部平移自由度的索引，$|\mathcal{I}_{w}^{\text{T}}|=3$
- $\mathcal{I}_{w}^{\text{R}}\subset\{1,\dots,J\}$：三个腕部旋转自由度的索引，$|\mathcal{I}_{w}^{\text{R}}|=3$
- $\mathbf{q}_{t}\in\mathbb{R}^{J}$：时间 $t$ 的重定向关节值
- $s_{\text{T}},s_{\text{R}}$：平移和旋转动作的缩放因子
- $\ell,u\in\mathbb{R}^{J}$：关节下限和上限向量
- $\hat{q_{t}}\in\mathbb{R}^{J}$：发送给策略控制器的关节目标值

则关节目标计算定义为：

$$a_{t}^{\text{wrist-T}}=a_{t}[\mathcal{I}_{w}^{\text{T}}]\in\mathbb{R}^{3},\quad q_{t}^{\text{wrist-T}}=\mathbf{q}_{t}[\mathcal{I}_{w}^{\text{T}}]+s_{\text{T}}\cdot a_{t}^{\text{wrist-T}}$$

$$a_{t}^{\text{wrist-R}}=a_{t}[\mathcal{I}_{w}^{\text{R}}]\in\mathbb{R}^{3},\quad q_{t}^{\text{wrist-R}}=\mathbf{q}_{t}[\mathcal{I}_{w}^{\text{R}}]+s_{\text{R}}\cdot a_{t}^{\text{wrist-R}}$$

$$a_{t}^{\text{fingers}}=a_{t}[\mathcal{I}_{f}],\quad q_{t}^{\text{fingers}}=\ell_{\mathcal{I}_{f}}+\frac{u[\mathcal{I}_{f}]-\ell[\mathcal{I}_{f}]}{2}\cdot(a_{t}^{\text{fingers}}+1)$$

$$\hat{q}_{t}=\text{concat}(q_{t}^{\text{wrist-T}},\ q_{t}^{\text{wrist-R}},\ q_{t}^{\text{fingers}})$$

### A.4 接触近似

设：$V_{o}=\{v_{i}^{o}\}_{i=1}^{N_{o}}$ 为一个物体部件网格的顶点，$V_{h}=\{v_{j}^{h}\}_{j=1}^{N_{h}}$ 为一个 MANO 手网格的顶点，$\gamma$ 为接触距离阈值，$N_{c}$ 为最大原始接触近似数（我们使用 $\gamma=0.01, N_{c}=50$），$K$ 为灵巧机器人手上碰撞链接的数量。

首先，我们通过查找物体网格顶点到 MANO 网格最近邻的 $L_{2}$ 距离在 $\gamma$ 以内的顶点进行接触近似：对每个 $v_{i}^{o}$，我们得到 $v_{j}^{*}=\arg\min_{j}\|v_{i}^{o}-v_{j}^{h}\|_{2}$，如果 $\|v_{i}^{o}-v_{j^{*}}^{h}\|_{2}<\gamma$ 则标记 $v_{i}^{o}$ 为近似接触点。结果为：

- 接触张量 $\mathcal{C}\in\mathbb{R}^{T\times N\times K\times 3}$
- 有效性掩码 $\mathcal{M}\in\{0,1\}^{T\times N\times K}$

其中 $T$ 为演示片段的时间步数，$N$ 为物体部件数（我们所有铰接物体资产中 $N=2$）。对每种灵巧手重复完全相同的过程，因此每个双手RL任务环境有两份相同形状的接触信息。

*图7：由我们训练的RL策略完成的长时间跨度任务的可视化。棕色盒子用作平台表面，沿袭了原始 ARCTIC 数据收集设置，物体放置在桌面上的方形纸板盒上 [10]。*

## 附录 B 实验细节

### B.1 RL训练与评估细节

我们使用 Genesis 进行物理仿真 [45]，PPO 作为基础RL算法，由 rl-games [47] 包实现。在报告的结果中（包括我们的方法和基线方法），所有灵巧手的RL训练使用 12,000 个并行环境，除了 Dex Hand 因内存限制使用 10,000 个环境。每次训练运行占用一块 NVIDIA L40s 或 H100 GPU，对每个演示和每对灵巧手的所有比较方法运行5个随机种子，§5.2的动作消融实验使用3个随机种子。

### B.2 RL策略观测和动作空间

我们使用基于状态的输入作为策略观测空间：包括物体状态、关节目标、手指到物体的距离和归一化的手-物体接触力。

### B.3 基线重新实现的细节

我们最相关的基线方法 [35, 36] 使用 Isaac-Gym [54] 构建，具有各种仿真特定的实现细节。为确保公平比较，我们在训练框架和RL环境中进行了忠实的重新实现。我们的某些修改可以为基线带来更好的性能：例如，Genesis [45] 使用更稳定的仿真接触建模且内存效率更高，支持高达 12,000 个并行环境的训练，学习效率远高于基线使用的 Isaac Gym 环境（即 ObjDex [35] 的 2048 个和 ManipTrans [36] 的 4096 个环境）。下面详述每个基线的重新实现细节：

**ObjDex [35] 重新实现细节。** 为确保公平比较，我们联系了原始作者以获取其未公开的设置细节，包括：1. 训练所用 ARCTIC 片段的精确帧起止参数的合理估计；2. 帧插值倍数，有效地将RL训练的回合长度延长至超过原始演示（例如一个 $T$ 时间步的 ARCTIC 片段需要在 $4T$ 或 $7T$ 的回合步数上训练RL）。我们复用了他们的片段范围，但在实证发现插值增加训练时间但不改善任务性能后选择不使用。

此外，原始 ObjDex [35] 方法使用两级框架，先在所有 ARCTIC 演示片段上学习高层腕部规划器，然后低层RL策略输出腕部残差动作。我们直接使用运动学重定向结果作为腕部基础动作。高层腕部规划器设计假设可获得更大数据集，使低层RL策略对学习的规划器输出敏感——我们推测这是我们重新实现能优于原始结果的主要原因（例如在 Ketchup-100 上，我们的重新实现对所有手达到 $>90\%$ 成功率，而原论文报告 $41.2\%$；在 Mixer-170 上，我们的实现在四种手中的三种达到 $>70\%$ 成功率，而原论文报告 $57.6\%$）。

**ManipTrans [36] 重新实现细节。** 由于原始方法未直接在 ARCTIC 演示上评估（尽管论文附录 A.1 节 [36] 报告了部分 ARCTIC 物体的定性结果），我们重新实现了其提出的课程学习，同时保持其他所有设置与我们最佳方案一致（包括混合动作形式化、同时使用任务和辅助奖励训练以及相同的RL超参数等）。我们遵循原始方法 [36] 在训练期间衰减四个参数，即物体姿态误差和手部关键点误差的阈值、z轴重力值和摩擦参数，分别记为 $\epsilon_{\text{object}}^{P},\epsilon_{\text{object}}^{R},\epsilon_{\text{finger}},g_{\text{gravity}},\mu$。我们修改了 Genesis [45] 的刚体求解器以支持在RL训练期间修改重力向量。ManipTrans [36] 未公开这些参数的精确衰减计划或重力和摩擦参数的范围，因此我们选择与虚拟物体控制器课程一致的指数调度器，范围为 $g_{\text{gravity}}\in[0,-9.81],\mu\in[4.0,1.0]$。更具体地，给定最大迭代次数 $I$、期望参数范围和衰减间隔 $v$，参数每 $v$ 次迭代衰减一次；参数达到最终值后，训练再继续固定次数的迭代（这也与我们的方法一致）。指数调度取决于给定的最大迭代次数 $\mathcal{I}$，对每个参数 $\omega\in\{\epsilon_{\text{object}}^{R},\epsilon_{\text{finger}},g_{\text{gravity}},\mu\}$：其在给定训练迭代时的值可写为 $\omega_{\text{current}}=\omega_{\text{init}}\cdot\left(\frac{\omega_{\text{final}}}{\omega_{\text{init}}}\right)^{t/I}$。注意我们使用伪值 $\bar{g}_{\text{gravity}}\in[9.81,0]$，因为衰减计算假设正值边界，实际施加的重力为 $g_{\text{gravity}}=9.81-\bar{g}_{\text{gravity}}$。

### B.4 策略评估设置

**跨随机种子评估。** 对每种方法和任务运行5个随机种子；每个种子运行基于累积任务奖励保存最佳策略检查点，每个检查点评估20个回合。对每个评估回合，记录实际物体状态（位姿和旋转关节角度）并与演示轨迹比较。

**性能指标。** 我们的功能性重定向任务要求操作策略实现铰接物体跟踪，需要在长时间序列上平衡位姿和关节角度误差。对于性能报告，先前工作探索了逐步成功率 [35] 或跟踪误差 [36]，但两者都有明显缺陷：成功率报告基于策略能在给定误差阈值内将物体移至跟踪目标的时间步比例，因此结果对阈值高度敏感（此情况需要位置、旋转和关节角度误差的三个不同阈值），且还取决于物体尺寸和几何形状。报告跟踪误差可以准确，但每个任务显示三个不同误差，难以从实验结果中得出高层比较和要点。为解决这些局限，我们提出参照物体姿态跟踪领域的先前工作 [55, 56, 57] 使用类似的 ADD-AUC（ADD 表示平均距离，AUC 表示曲线下面积，我们不使用 ADD-S 因为我们有来自演示的精确匹配目标）指标，关键区别在于我们对每个物体部件分别计算 ADD（以适应铰接物体），在计算 AUC 前平均 ADD 结果。我们发现这是一个不太敏感的指标，仍然为每种方法报告一个成功率值，同时反映了策略展开的定性结果。

## 附录 C 额外实验结果

我们在图中可视化了长时间跨度操作任务的关键帧。策略展开的额外定性结果请参见补充视频。下一页的图展示了§5.2中描述的动作消融实验结果。

*图8：手部动作消融。我们在灵巧手和物体的子集上对动作输出形式化进行消融，且在无课程学习下训练。具有更严格约束的混合动作（浅绿色和深绿色条）在任务奖励训练和任务+辅助奖励训练设置下均展示了优于绝对动作和腕部约束较少的完全残差动作的学习性能。*
