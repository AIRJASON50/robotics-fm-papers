# RL Token：用视觉-语言-动作模型引导在线强化学习

Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke

**Physical Intelligence**

https://pi.website/research/rlt

*图 1：我们的方法通过训练编码器和解码器，从 VLA 的内部特征中生成紧凑且有意义的表征，将一个"RL token"引入 VLA。提取的表征随后用于通过样本高效的在线强化学习训练轻量级的演员-评论家 (actor-critic) 网络，使得高精度任务能够在数小时甚至数分钟的机器人经验中完成微调。*

*摘要*——视觉-语言-动作 (Vision-Language-Action, VLA) 模型能够"开箱即用"地学习执行多样化的操控技能，但要达到现实世界任务所要求的精度和速度，则需要进一步微调——例如通过强化学习 (Reinforcement Learning, RL)。我们提出了一种轻量级方法，仅需数小时的真实世界练习即可实现对预训练 VLA 的样本高效在线 RL 微调。我们 (1) 对 VLA 进行适配以暴露一个"RL token"——一种紧凑的读出表征 (readout representation)，在保留任务相关预训练知识的同时作为在线 RL 的高效接口；(2) 在该 RL token 上训练一个小型演员-评论家头 (actor-critic head) 来优化动作，同时将学习到的策略锚定于 VLA。使用 RL token 的在线 RL（RLT）使得即使是大型 VLA 也能通过 RL 快速高效地进行微调。在四个真实机器人任务（螺丝安装、扎带紧固、充电器插入和以太网插入）中，RLT 在任务最困难部分的执行速度提升高达 $3\times$，并在数分钟到数小时的练习内显著提高了成功率。在某些任务上，它甚至可以超越人类遥操作的速度。

## I. 引言

通用视觉-语言-动作 (VLA) 模型能够从数据中学习各种各样的操控技能。然而，它们往往在执行的最后一毫米处遇到困难：动作可能很慢，成功完成可能需要暂停和重试，而在精密任务的关键阶段，微小的错误会不断累积导致失败。应对这一挑战的自然方式是用强化学习 (RL) 微调 VLA。通过在目标任务上练习，RL 可以精确改进任务中对成功最为关键的阶段——这些阶段往往对微小误差最为敏感，也是最难仅靠演示数据可靠覆盖的。但真实世界的机器人操作面临严格的预算约束：每个回合 (episode) 都耗费时间，每次失败都消耗精力和磨损，有意义的适应通常必须在数小时的练习内完成。

然而，对 VLA 进行样本高效的微调面临重大挑战。一方面，传统的基础模型 RL 训练方法 [1–3] 依赖大规模数据，对于快速在线适应而言效率不高。另一方面，数据高效的真实世界 RL 方法 [4, 5] 通常训练规模小得多的模型，这些模型可以在数小时内改进，但牺牲了 VLA 的泛化能力。因此，核心问题在于如何利用 VLA 的泛化能力，同时实现轻量级在线 RL 的速度和样本效率。

我们提出了一种实用方案，利用从预训练 VLA 策略中获取的表征来引导快速在线强化学习。我们的核心思想是对 VLA 进行适配，使其暴露一个可用于样本高效在线 RL 的紧凑接口。为此，我们训练 VLA 暴露一个 *RL token*——一种压缩表征，使任务相关的预训练知识可被轻量级在线 RL 策略访问。使用该 RL token 运行 RL（RLT）形成了一种简洁的分工：冻结的 VLA 提供广泛的感知理解和动作建议，而轻量级的演员 (actor) 和评论家 (critic) 则在线适配策略以应对任务中最困难的部分。为使该方法在样本高效的真实世界场景中切实可行，我们使用样本高效的在线 RL 算法来训练使用 RL token 表征的小型演员和评论家网络，并引入额外的正则化项将演员锚定于 VLA 的动作，使在线 RL 是在优化有前景的行为而非从零学习。

我们在四个具有挑战性的机器人操控任务上评估了 RLT，这些任务可能需要毫米级甚至亚毫米级精度：螺丝安装、扎带紧固、以太网插入和充电器插入。在这些任务中，RLT 在数小时的在线训练内同时提高了成功率和执行速度。最大的提升出现在任务的关键阶段——这些阶段需要高精度且决定任务成败——RLT 将执行速度提升高达 $3\times$，并大幅提高了成功率，例如在一个具有挑战性的螺丝插入任务中从 20% 提升到 65%。在我们任务中最需要灵巧操作的部分，使用我们方法训练的策略可以在保持可靠性的同时超越专家遥操作的速度。这些结果表明，将 VLA 模型与轻量级在线 RL 相结合，为实现高性能操控提供了一条实用路径，无需大量任务特定的工程设计。

## II. 相关工作

**视觉-语言-动作模型。** 从大规模演示数据集进行行为克隆 (Behavioral Cloning) 近年来已成为训练通用机器人操控策略的主导范式（参见例如 [6–11]）。促成这一成功的两个关键要素是动作分块 (action chunking) [12]——预测多个动作用于顺序开环执行，以及使用富有表达力的输出分布，如扩散模型 (diffusion) [13] 或自回归生成 (autoregressive generation) [6]，能够捕捉演示数据中固有的多模态性。进一步的进展来自于使用大型预训练视觉-语言模型作为语言条件通用策略的骨干网络，从而产生了视觉-语言-动作 (VLA) 模型 [6, 7]。这些模型将大规模网络先验知识引入闭环机器人策略。近期工作将 VLA 骨干与分块动作生成相结合，通过扩散 [8] 或自回归标记化 (autoregressive tokenization) [14, 15] 实现了最先进的通用操控性能。虽然这些策略展现出令人印象深刻的泛化能力 [9, 16]，但其在任何给定任务上的性能最终受限于训练所用遥操作数据的质量和覆盖范围——当演示本身存在噪声或不一致时，在精度关键任务上实现可靠的成功仍然困难重重。

**真实世界强化学习。** 强化学习提供了一种自然的方式来突破演示数据的性能上限：通过在任务上练习，智能体 (agent) 可以发现从未被演示过的更快、更精确或更鲁棒的策略。在实践中，面向机器人的真实世界 RL 在严格的样本预算下运行，因为每次机器人回合都耗费时间和磨损。离策略演员-评论家方法 (off-policy actor-critic methods)（例如 [17–20]）通过复用存储在经验回放缓冲区 (replay buffer) 中的转移来解决这一问题，并且可以通过增加更新-数据比 (update-to-data ratio) [21] 进一步提高样本效率，尽管可能需要正则化来避免不稳定性 [22]。至关重要的是，离策略方法还可以纳入人类演示数据来引导学习（例如 [23]），结合模仿学习和 RL 的优势。越来越多的工作开发了在物理机器人上部署 RL 的实用方案，包括自主数据收集流水线 [24]、高效学习框架如 SERL [4, 25] 和 RL$^{100}$ [5]，以及人机在环变体 (human-in-the-loop variants)，允许操作员在自主执行期间进行干预和提供纠正 [4]。这些系统已经证明，离策略演员-评论家方法结合演示和人类纠正，可以在数小时的机器人时间内解决接触丰富的操控任务。然而，它们通常在标准预训练视觉编码器（如 ResNet）之上从零训练小型策略，放弃了现代 VLA 模型中可用的丰富行为先验。RLT 通过将冻结的 VLA 同时用作感知骨干和行为先验来弥合这一差距，为轻量级在线 RL 策略服务。

**VLA 模型的 RL 微调。** 一个快速增长的研究方向探讨如何通过 RL 改进预训练的 VLA。这些方法主要在*更新什么*以及*如何引入 RL 信号*上有所不同。在一端，若干方法更新整个 VLA 模型。RECAP [3] 通过基于优势条件策略提取 (advantage-conditioned policy extraction) 的离线 RL 端到端训练整个 $\pi_{0.6}^+$ 模型：一个分布式价值函数估计每个时间步的优势值，VLA 在所有收集的数据——演示、自主回合和人类干预——上进行训练，并使用最优性指标来增加高优势动作的权重。通过在机器人数据收集和离线 RL 更新之间迭代，RECAP 在复杂长时域任务（如制作浓缩咖啡、折叠衣物和组装箱子）上将吞吐量提高了一倍以上。其他工作将近端策略优化 (Proximal Policy Optimization, PPO) 或其变体应用于 VLA 微调（例如 [1, 26, 27]），但在策略方法 (on-policy methods) 难以以样本高效且可扩展的方式推广到真实世界 RL。在另一端，轻量级方法避免更新整个 VLA，而是在冻结模型之上训练一个小型辅助模块。ConRFT [38] 冻结 VLA 编码器并使用基于一致性的训练目标 (consistency-based training objective) 和学习到的二元奖励分类器来微调动作头，但仅在短时域任务上操作单步动作而不使用分块。Policy Decorator [29] 学习一个残差策略 (residual policy)，其输出经手动调节的超参数缩放后与冻结 VLA 的预测相加，但仅在仿真中进行了验证，且样本需求较高（数百万步量级）。Probe-Learn-Distill (PLD) [30] 先用 Cal-QL [31] 在基础策略回合上预训练评论家，然后在冻结 VLA 之上学习单步残差策略，可选地通过监督微调将结果蒸馏回 VLA。GR-RL [2] 采用多阶段方法将通用 VLA 专门化于长时域鞋带系扎任务：首先进行离线过滤行为克隆，然后通过在潜在空间中学习噪声预测器来引导冻结 VLA 的扩散过程 [32]，从而进行在线 RL。DSRL [32] 类似地在扩散噪声空间中操作，学习一个潜在策略来调制去噪过程，将动作引导向高回报区域。

RLT 与这些方法共享在不承担全模型 RL 成本的情况下改进预训练 VLA 的目标，但在若干关键设计选择上有所不同。首先，RLT 引入了一个 *RL token*——一种经训练以压缩 VLA 内部嵌入的紧凑读出表征——作为轻量级演员-评论家的状态观测，在保留 VLA 预训练感知结构的同时实现高效在线学习。其次，RLT 在与 VLA 原生动作接口对齐的*分块动作*上操作，在高控制频率下稀疏奖励的时序差分学习 (temporal-difference learning) 中缩短了有效决策时域——与面临更长信用分配 (credit-assignment) 问题的单步方法 [28–30] 形成对比。第三，RLT 的演员不是预测残差或潜在噪声，而是直接*以 VLA 采样的参考动作块为条件并向其正则化*，将在线 RL 转变为对良好 VLA 先验行为策略的局部优化，而非无约束搜索或对扩散过程的隐式调制。这些设计选择共同实现了在真实机器人上的样本高效在线 RL——在数小时的练习内同时提高成功率和执行速度。

## III. 预备知识

**视觉-语言-动作模型。** 大规模 VLA 模型从涵盖数万小时的多样化人类演示数据集中学习操控行为，在某些情况下还使用非机器人视觉-语言数据进行增强 [7, 9, 16]。典型的 VLA 由两个组件构成：(i) *VLM 骨干网络*，即一个视觉-语言模型，将多模态输入（图像、语言指令和本体感受状态）编码为共享的 token 序列；(ii) *动作专家*，即一个基于扩散的模块，对骨干网络的 token 进行注意力计算，并通过迭代去噪生成连续动作。我们基于 $\pi_{0.6}$ 模型 [33] 构建。给定最多四张相机图像、一条语言指令 $\ell$ 和本体感受状态 $\mathbf{s}_t^p$，$\pi_{0.6}$ 生成一个动作序列（称为*动作块* (action chunk)）：$\tilde{\mathbf{a}}_{t:t+H-1} = (\tilde{\mathbf{a}}_t, \ldots, \tilde{\mathbf{a}}_{t+H-1}) \in \mathbb{R}^{H \times d}$，即一个包含 $H = 50$ 个动作的序列，对应 1 秒的控制。我们用 $\pi_{\text{vla}}$ 表示预训练 VLA 生成的分块策略。在实际操作中，机器人仅以开环方式执行该块的前缀部分（例如前 20 步），然后根据新的观测重新规划。由于某些任务的难度（例如高精度任务），大规模收集高质量的模仿学习数据可能十分困难，这限制了 VLA 在这些任务上的表现。这促使我们在下一节中开发在线 RL 精调方法。

**强化学习与演员-评论家方法。** 我们将机器人控制建模为马尔可夫决策过程 (Markov Decision Process, MDP) $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$，其中 $\mathcal{S}$ 为状态观测空间，$\mathcal{A}$ 为连续动作空间，$p(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$ 表示转移动态，$r(\mathbf{s}_t, \mathbf{a}_t)$ 为奖励函数，$\gamma \in [0, 1)$ 为折扣因子。RL 的目标是学习一个策略 $\pi(\mathbf{a}_t \mid \mathbf{s}_t)$，使期望折扣回报最大化：$J(\pi) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$，其中 $\rho_\pi(\tau)$ 表示策略 $\pi$ 诱导的轨迹分布。我们假设只能获得*稀疏二值奖励*：人类监督者在每个回合结束时标注成功或失败，成功时设 $r_T = 1$，否则设 $r_T = 0$。策略 $\pi$ 的动作-价值函数为 $Q^\pi(\mathbf{s}_t, \mathbf{a}_t) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \;\middle|\; \mathbf{s}_t, \mathbf{a}_t\right]$。

在我们的设定中，策略和评论家均在动作块 $\mathbf{a}_{t:t+C-1} = (\mathbf{a}_t, \ldots, \mathbf{a}_{t+C-1}) \in \mathbb{R}^{C \times d}$ 上操作，其中 $C$ 表示 RL 的块长度（$H$ 表示 VLA 预测的块时域）。我们选择 $C < H$ 以赋予策略更强的响应能力。我们将分块策略定义为 $\pi(\mathbf{a}_{t:t+C-1} \mid \mathbf{s}_t)$，以及对应的块级 C 步价值估计 $Q^\pi(\mathbf{s}_t, \mathbf{a}_{t:t+C-1}) = \sum_{t'=t}^{t+C-1} \gamma^{t'-t} r_{t'} + \gamma^C \mathbb{E}_{\mathbf{a}' \sim \pi | \mathbf{s}_{t+C}} \left[Q^\pi(\mathbf{s}_{t+C}, \mathbf{a}')\right]$。我们基于经典的离策略演员-评论家方法 [17, 19, 34] 构建，联合训练一个随机演员 $\pi_\theta$ 和一个评论家 $Q_\psi$。关键的是，学习是离策略的，使用存储在经验回放缓冲区 $\mathcal{B}$ 中的转移数据，无论这些数据由哪个策略生成。这一特性在我们的设定中至关重要，因为 $\mathcal{B}$ 汇聚了来自 VLA 策略、RL 学习器和人类遥操作干预的数据。

## IV. 从 RL Token 进行强化学习

图 1 总结了我们利用 RLT 从预训练 VLA 模型实现快速且稳定的在线 RL 的方案。核心思想是最大限度地利用预训练的 VLA 来提高 RL 训练过程的效率。使用在线 RL 训练整个 VLA 可能在计算和样本效率上代价过高，以至于无法在短短几小时内产生改进的策略。相反，我们使用冻结的 VLA 来提供 RL 状态表示、提供参考动作，并引导探索朝向接近其自身预测的动作方向，同时仍使用小型的演员和评论家网络。我们首先在少量任务特定的演示数据上对 VLA 进行适配，既为了改善其初始任务策略，也为了暴露一个用于下游 RL 的 RL token。然后我们冻结 VLA，在线训练轻量级的离策略演员和评论家网络，同时基于 RL token 表示和 VLA 的参考动作进行条件化，并正则化学习到的策略使其保持接近 VLA 模型。我们的方法将在线 RL 转化为对有前景行为的局部精调，而非无约束的搜索。这种设计使在线 RL 方法兼具小型演员-评论家算法的效率和预训练 VLA 模型的表示与行为。

### A. 适配 VLA 以暴露 RL 接口

样本高效的在线 RL 关键取决于状态表示的选择。将 RL 直接应用于完整的 VLA 模型与快速的真实世界适应极不匹配：表示是高维的，而对数十亿参数模型的在线更新在计算上昂贵且样本效率低下。与此同时，我们希望利用 VLA 预训练后内部已包含的表示，因为它在大规模网络数据和机器人数据上训练，已经包含了对许多任务生成动作有用的信息。然而，从基于 Transformer 的 VLA 中哪些特征构成良好的在线 RL 表示通常并不明显，且每个 Transformer 层中的嵌入都是高维的。因此，我们的目标是将 VLA 的表示压缩为一个紧凑的嵌入用于 RL，既保留与任务相关的信息，又足够小以支持轻量级的在线演员-评论家学习。

我们通过添加一个 *RL token*（图 2）来实现这一目标：一个可学习的读出嵌入 (readout embedding)，将 VLA 的知识总结为一个小向量，作为 RL 的状态。具体而言，我们通过向预训练 VLA 添加一个小型 Transformer 来获得 RL token。我们以编码器-解码器 [35] 的方式训练该 Transformer，编码器的最后一个输入为 RL token。由于 RL token 的表示必须保留足够的信息以使解码器能够重建输入，因此它起到了信息瓶颈的作用。令 $\mathbf{z} = f(s, \ell; \theta_{\text{vla}})$ 表示预训练 VLA 对状态 $s$ 和语言指令 $\ell$ 生成的最终层 token 嵌入。嵌入 $\mathbf{z}$ 分解为 $\mathbf{z}_{1:M} = \{\mathbf{z}_1, \ldots, \mathbf{z}_M\}$，其中每个 $\mathbf{z}_i$ 对应一个输入 token 的嵌入。我们在序列末尾附加一个可学习的嵌入 $\mathbf{e}_{\text{rl}} = \mathbf{e}_\phi(\texttt{<rl>})$，并用一个轻量级编码器 Transformer $g_\phi$ 处理增广后的序列。编码器在特殊 token 位置的输出，记为 $\mathbf{z}_{\text{rl}}$，即为我们的 RL token$^1$

$$\mathbf{z}_{\text{rl}} = g_\phi \big( [\mathbf{z}_{1:M}, \ \mathbf{e}_{\text{rl}}] \big)_{M+1}. \tag{1}$$

然后训练一个解码器 Transformer $d_\phi$ 和一个线性输出投影 $h_\phi$，从 $\mathbf{z}_{\text{rl}}$ 自回归地重建原始嵌入。令 $\bar{\mathbf{z}}_i = \text{sg}(\mathbf{z}_i)$ 表示对 VLA 嵌入施加的停止梯度 (stop-gradient) 操作，则在演示数据集 $\mathcal{D}$ 上的自回归重建目标为：

$$\mathcal{L}_{\text{ro}} = \mathbb{E}_{\mathcal{D}} \left[ \sum_{i=1}^{M} \left\| h_\phi \big( d_\phi ([\mathbf{z}_{\text{rl}}, \ \bar{\mathbf{z}}_{1:i-1}]) \big)_i - \bar{\mathbf{z}}_i \right\|^2 \right]. \tag{2}$$

我们在一个小型任务特定的演示数据集上训练参数 $\phi$，其中 VLA 相对于 $\mathcal{L}_{\text{ro}}$ 被视为冻结的，并且（可选地）将其与 VLA（$\theta_{\text{vla}}$）的监督微调相结合。之后，$\theta_{\text{vla}}$ 和 $\phi$ 均被冻结，在线 RL 在 RL token 表示 $\mathbf{z}_{\text{rl}}$ 上进行操作。

> $^1$ 在我们的实验中，每个任务有固定的语言指令，因此在此步骤中我们省略了语言嵌入；该构造一般适用于所有 VLA 嵌入。

*图 2:* **RL token 提取详情。** RLT 向预训练 VLA 添加了一个编码器-解码器 Transformer。它生成 VLA 表示的压缩嵌入（即 RL token）。该表示随后在在线 RL 微调过程中实现数据和参数的高效利用。

### B. 在线 RL 精调 VLA 动作块

在初始适配阶段之后，我们冻结 VLA 和 RL token 表示。然后在线训练轻量级的演员 ($\pi_\theta$) 和评论家 ($Q_\psi$) 网络。它们的输入 $x$ 将 RL token 与任何有助于实现闭环控制的附加信息（例如机器人的本体感受状态）相结合。评论家模型估计状态和动作的价值：$Q_\psi(\mathbf{x}, \mathbf{a}_{1:C}) \in \mathbb{R}$。值得注意的是，RL 演员 $\pi_\theta(\cdot | \mathbf{x}, \tilde{\mathbf{a}}_{1:C})$ 并非从头生成动作，而是被训练来精调由 VLA 提出的动作序列 $\tilde{\mathbf{a}}_{1:C}$（称为动作块）。

**训练评论家。** 我们的评论家 $Q_\psi(\mathbf{x}, \mathbf{a}_{1:C})$ 以状态和动作块 $\mathbf{a}_{1:C}$ 作为输入。我们使用标准的离策略时序差分学习，在从经验回放缓冲区 $\mathcal{B}$ 中采样的动作块转移上训练评论家：

$$\mathcal{L}_Q = \mathbb{E}_{(\mathbf{x}, \mathbf{a}_{1:C}, \mathbf{x}') \sim \mathcal{B}} \left[ \left( \hat{Q} - Q_\psi(\mathbf{x}, \mathbf{a}_{1:C}) \right)^2 \right],$$

$$\hat{Q} = \sum_{t'=1}^{C} \gamma^{t'-1} r_{t'} + \gamma^C \mathbb{E}_{\mathbf{a}' \sim \pi_\theta} \left[ Q_{\psi'}(\mathbf{x}', \mathbf{a}') \right]. \tag{3}$$

其中输入状态为 $\mathbf{x} = (\mathbf{z}_{\text{rl}}, \mathbf{s}^{\text{p}})$，$\mathbf{s}^{\text{p}}$ 表示本体感受状态信息，$\mathbf{z}_{\text{rl}}(\mathbf{s})$ 表示对状态 $\mathbf{s}$ 提取的 RL token；$\mathbf{x}'$ 表示下一个输入状态；$\mathbf{a}' \sim \pi_\theta$ 表示从 RL 策略中采样。在实践中，我们遵循 TD3 [19]，$\psi'$ 为目标网络的参数。

**训练 RL 策略。** 我们的演员网络 $\pi_\theta(\cdot | \mathbf{x}, \tilde{\mathbf{a}}_{1:C})$ 在动作块上生成高斯动作分布。它以输入状态*和*参考动作块 $\tilde{\mathbf{a}}_{1:C}$ 作为输入，并生成动作分布：

$$\pi_\theta \big( \mathbf{a}_{1:C} \mid \mathbf{x}, \tilde{\mathbf{a}}_{1:C} \big) = \mathcal{N} \Big( \mu_\theta \big( \mathbf{x}, \tilde{\mathbf{a}}_{1:C} \big), \sigma^2 \mathbf{I} \Big), \tag{4}$$

其中，如前所述，$\mathbf{x} = (\mathbf{z}_{\text{rl}}, \mathbf{s}^{\text{p}})$。对 $\tilde{\mathbf{a}}$ 的条件化使演员直接接触到 VLA 预测的动作，从而使在线 RL 精调一个强初始提议而非从头学习。第二个好处是，采样的参考块保留了 VLA 多模态动作分布中的模式信息，否则单模态高斯演员难以恢复这些信息 [36]。我们通过将演员的动作正则化到参考动作附近来进一步稳定学习。具体而言，我们优化演员以最大化评论家价值，同时保持接近 VLA 参考块 $\tilde{\mathbf{a}}$，这在精神上类似于 KL 正则化的 RL 方法（参见例如 [20, 37–40]）。这有效地将在线 RL 转化为围绕 VLA 生成的动作分布进行局部动作编辑，而非在高维动作块上进行无约束搜索。学习 RL 策略的目标函数为

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\substack{\mathbf{s} \sim \mathcal{B} \\ \tilde{\mathbf{a}}_{1:C} \sim \pi_\theta}} \left[ -Q_\psi(\mathbf{x}, \mathbf{a}_{1:C}) + \beta \left\| \mathbf{a}_{1:C} - \tilde{\mathbf{a}}_{1:C} \right\|_2^2 \right],$$

$$\tilde{\mathbf{a}}_{1:C} \sim \pi_{\text{vla}}(\cdot \mid \mathbf{s}, \ell), \tag{5}$$

其中系数 $\beta$ 控制演员向采样的 VLA 动作正则化的强度。

**参考动作丢弃。** 参考动作条件化的一个实际失败模式是演员可能简单地复制 $\tilde{\mathbf{a}}$ 而非学习改进它。这在评论家尚未提供有效信号之前尤其容易发生，因为对 $\tilde{\mathbf{a}}$ 的条件化和向其正则化都鼓励演员保持接近 VLA 的提议。为防止这种情况，我们应用*参考动作丢弃* (reference action dropout)：对于每个训练批次中的随机子集转移，我们在将参考块传递给演员之前将其替换为零。这迫使演员维持一条独立的动作生成路径，同时仍允许它在参考块存在时利用 VLA 的动作分布。在实践中，一旦评论家提供有用信号，演员会自然地学会在偏离参考能增加预测价值时进行偏离。

## V. 完整系统

算法 1 总结了我们完整的训练循环。在初始预热阶段使用基础 VLA 策略收集 episode 之后，训练在机器人上收集经验和对回放数据执行离策略 actor-critic 更新之间交替进行。回放缓冲区聚合了 VLA 预热数据、在线 RL 滚动采样以及可选的人类干预数据。此外，人类监督者提供稀疏的成功/失败标签。以下详细描述各步骤。

**预热。** 在训练 RL token 表征（第 IV-A 节）之后，我们通过滚动执行 VLA 参考策略 $N_{\text{warm}}$ 个环境步来预填充回放缓冲区 $B$。这为评论家提供了初始学习信号，并确保在线 RL 从有能力的 VLA 行为开始。

**滚动采样。** 在在线收集过程中的每个动作块边界处，冻结的 VLA 产生一个参考块 $\tilde{\mathbf{a}}_{1:H}$，RL token 模块提取 $\mathbf{z}_{\text{rl}}$。然后 actor 输出一个动作块 $\mathbf{a}_{1:C} \sim \pi_\theta(\cdot \mid \mathbf{x}, \tilde{\mathbf{a}}_{1:C})$。为了加速接触丰富或安全关键行为的学习，人类操作员可以选择性地进行干预，提供遥操作命令 $\mathbf{a}_{1:C}^{\text{h}}$ 来覆盖干预期间的 actor 输出。当发生这种情况时，干预在回放缓冲区中替代 VLA 参考。在所有情况下，存储在 $\mathcal{B}$ 中的每个转移都包含已执行的动作和对应的参考，使 actor 能够从自主滚动采样和人类修正中学习。

**动作块子采样。** 虽然 RL 策略使用长度为 $C$ 的动作块，但我们获得每个中间步骤的观测。因此，我们可以通过将中间步骤存储到回放缓冲区来增加数据量并提高学习效率。具体而言，我们选择步长为 2，将对应于 $\langle \mathbf{x}_0, \mathbf{a}_{0:C} \rangle, \langle \mathbf{x}_2, \mathbf{a}_{2:C+2} \rangle, \langle \mathbf{x}_4, \mathbf{a}_{4:C+4} \rangle, \ldots$ 的转移保存到回放缓冲区。注意，由于我们的 RL 算法的离策略特性，我们可以使用所有动作块（包括 VLA 生成的动作和人类干预）。

---

**算法 1** RLT

**输入:** *冻结的 VLA 骨干网络 $f_{\theta_v}$ 和 VLA 动作分布 $\pi_{\text{vla}}$；演示数据 $\mathcal{D}$，块长度 $C$，回放缓冲区 $B$，预热步数 $N_{\text{warm}}$，比率 $G$，VLA 微调权重 $\alpha$，策略约束 $\beta$。*

1: **训练 RL token 并（可选地）微调 VLA**

2: 使用 $\mathbf{z}_i = f_t(\mathbf{s}, \ell, \theta_{\text{vla}})$、$\mathbf{z}_{\text{rl}} = g_\phi([\mathbf{z}_{1:M}, \mathbf{e}_{\text{rl}}])_{M+1}$ 和 $\theta_{\text{vla}}$（仅当 $\alpha > 0$ 时）训练 $\phi$。

$$\mathcal{L}_{\text{ro}}(\phi) = \mathbb{E}_{\mathcal{D}} \left[ \sum_{i=1}^{M} \left\| h_\phi \big( d_\phi ([\mathbf{z}_{\text{rl}}, \ \bar{\mathbf{z}}_{1:i-1}]) \big)_i - \bar{\mathbf{z}}_i \right\|^2 \right].$$

3:

$$\phi, \theta_{\text{vla}} = \arg \min_{\phi, \theta_{\text{vla}}} \mathcal{L}_{\text{ro}}(\phi) + \alpha \mathcal{L}_{\text{vla}}(\theta_{\text{vla}})$$

4: **训练 RL actor 和 critic**

5: 初始化 critic $Q_\psi$ 和 RL 策略 $\pi_\theta$。

6: **for** 环境步 $t = 0, C, 2C \ldots$ **do**

7: $\quad$ 采样 VLA 参考块 $\tilde{\mathbf{a}}_{t:t+C-1} \sim \pi_{\text{vla}}(\mathbf{s}_t)$。

8: $\quad$ 构建 RL 状态 $\mathbf{x}_t = (\mathbf{z}_{\text{rl}}(\mathbf{s}_t), \mathbf{s}_t^{\text{p}})$。

9: $\quad \mathbf{a}_{t:t+C-1} \leftarrow \begin{cases} \mathbf{a}^{\text{human}} & \text{if intervention} \\ \tilde{\mathbf{a}}_{t:t+C-1} & \text{if } t < N_{\text{warm}} \\ \sim \pi_\theta(\cdot \mid \mathbf{x}_t, \tilde{\mathbf{a}}) & \text{otherwise} \end{cases}$

10: $\quad$ 执行 $\mathbf{a}_{t:t+C-1}$ 并观测 $r_t, \mathbf{s}_{t+1}, \mathbf{s}_{t+1}^{\text{p}}$

11: $\quad \tilde{\mathbf{a}}_{t:t+C-1} \leftarrow \mathbf{a}^{\text{human}}$ 如果发生干预

12: $\quad$ 将转移存入 $B$：$\langle \mathbf{x}_t, \mathbf{a}_{t:t+C-1}, \tilde{\mathbf{a}}, r_t, \mathbf{x}_{t+1} \rangle$

13: $\quad$ **for** $g = 1, \ldots, G$ **do**

14: $\quad\quad$ 从 $B$ 中采样数据批次 $\mathbf{b} \sim B$。

15: $\quad\quad$ 计算目标 Q 值

$$\hat{Q} = \sum_{t'=1}^{C} \gamma^{t'-1} r_{t'} + \gamma^C \mathbb{E}_{\mathbf{a}' \sim \pi_\theta} \left[ Q_{\psi'}(\mathbf{x}', \mathbf{a}') \right]$$

16: $\quad\quad$ 使用 TD 备份训练 Critic（公式 (3)）

$$\mathcal{L}_Q(\psi) = \mathbb{E}_{\mathbf{b}} \left[ \left( \hat{Q} - Q_\psi(\mathbf{x}, \mathbf{a}) \right)^2 \right]$$

17: $\quad\quad$ 训练策略 $\mathbf{a} \sim \pi_\theta(\cdot \mid \mathbf{s}, \tilde{\mathbf{a}})$（公式 (5)）

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\mathbf{b}} \left[ -Q_\psi(\mathbf{x}, \mathbf{a}) + \beta \| \mathbf{a} - \tilde{\mathbf{a}} \|_2^2 \right]$$

18: $\quad$ **end for**

19: **end for**

---

**更新。** 策略更新根据算法 1 从回放缓冲区进行离策略执行。为了在训练期间保持计算和时间效率，我们异步执行滚动采样和学习。在实践中，我们每进行一次 actor 更新就执行两次 critic 更新，并在预热阶段之后不久开始学习。我们使用较高的更新数据比为 5，这在低数据在线场景中至关重要。

**关键阶段的定向改进。** 为了学习的实用性和效率，我们将 RLT 应用于改进我们考虑的每个任务的关键阶段——对应于需要高精度的最困难部分——并让基础 VLA 执行任务中较简单的部分。具体而言，每个 episode 从执行基础模型开始。在数据收集期间，人类操作员可以选择在何时将控制权从基础 VLA 移交给 RL 策略。这类似于交互式模仿学习中的人类干预决策 [41]。然后我们的系统对选定的任务片段应用 RL，在此关键阶段存储和训练转移数据，直到收到人类操作员发出的指示 RL 任务成功或失败的终止信号。这将数据收集和信用分配集中在在线适应最重要的行为部分。为了在测试时实现自主执行，我们可以在训练结束时进行最终的短期 VLA 微调阶段，要求它额外预测何时将执行权移交给 RL 策略（使用人类干预作为标签）。然后我们可以在测试时自动触发策略切换。

*图 3:* **我们实验中的任务**：每个任务包含一个需要高精度的关键阶段：（上）使用螺丝刀安装螺丝，（中）扎紧扎带，（下）插入以太网电缆和插入充电器。

## VI. 真实世界实验

我们在四个需要灵巧控制和亚毫米精度的真实世界操作任务上评估 RLT。预训练的 VLA 为这些任务的大部分提供了良好的初始化，但成功率和速度最终取决于对需要最高精度的关键接触丰富阶段的精化。我们的实验测试了我们的方法能否在驱动该方法设计的实际约束条件下实现这种改进：有限的机器人交互时间、稀疏的人类监督以及轻量级的在线学习。

我们围绕以下问题组织评估：

**Q1.** RLT 能否在基础 VLA 模型的基础上提高操作性能？

**Q2.** 与其他 RL 方法相比，RLT 在这些任务上表现如何？

**Q3.** 方法的每个组件——RL token、分块动作预测、策略正则化和参考动作直通——对方法性能的贡献有多大？

**Q4.** RLT 是否能使策略发现更好的策略，其策略与原始演示数据相比如何？

### A. 任务与设置

我们在以下任务上评估我们的方法（图 3）：

- **螺丝安装。** 机器人必须使用电动螺丝刀将 M3 螺丝拧入螺纹孔。这要求螺丝头和螺丝刀尖端之间实现亚毫米级对准。该任务特别困难，因为 (1) 螺丝可能并不总是完全直立放置，(2) 握持螺丝刀时，末端执行器的任何旋转都会被螺丝刀尖端到握持点之间 10 cm 的距离放大，(3) 关键视觉线索主要从对侧手臂的广角腕部相机可见，这构成了一个具有挑战性的感知问题。

- **扎带紧固。** 机器人必须将扎带尾部穿过其狭窄的锁定槽。该任务涉及对可变形物体进行具有严格公差的协调双手控制。成功插入需要仅从腕部相机推断尖端和槽口的位置，并以毫米级精度执行。

- **以太网插入。** 机器人必须将以太网连接器插入凹陷的端口。这需要精确的位置和角度对准，然后执行坚定而果断的插入动作。微小的方向误差或犹豫的接触通常会导致连接器卡在外壳上而非插入端口，使得成功对精度和接触动力学都很敏感。

- **充电器插入。** 机器人必须将充电器对准并插入电源插排。该任务很困难，因为策略必须实现厘米级对准，同时并不总能清楚地观察到插脚和插座。微小的对准误差通常导致反复探测或插入尝试失败。

每个任务包括抓取、重新定位和对准，持续 30-120 秒（在 50 Hz 下大约 1500-6000 个控制步）。对于每个任务，我们识别出*关键阶段*——插入、紧固或旋转环节——其中精度要求最高，基础 VLA 最常减速或失败。这些阶段通常持续 5-20 秒（250-1000 个控制步）。

**关键阶段评估。** 由于我们的方法旨在改进这些关键阶段，我们首先将评估集中在仅对关键阶段比较方法和消融实验上。在此设置中，episode 在被重置到关键阶段之前的部分完成任务状态后开始，使用略微随机化的初始配置集。例如，在扎带紧固中，机器人在插入尝试开始之前已经握持着扎带的两端。此设置隔离了 RL 预期最重要的精度关键片段，并减少了来自任务早期阶段（如抓取和运输）的混淆方差，这些阶段已经被基础 VLA 处理得相当好。在此受控设置中，每个智能体对每个任务评估 50 个 episode。

**完整任务评估。** 受控的关键阶段评估对于隔离我们方法旨在改进的瓶颈很有用，但它无法捕捉长期执行的全部变异性。因此，我们还在更现实的设置中额外评估完整任务性能，其中机器人从其"初始位置"开始，使用基础策略执行任务的早期阶段，并在该执行引起的状态变化下进入关键阶段。此设置明显更困难，因为经 RL 改进的行为必须在前序策略产生的更广泛状态分布下保持有效。对于完整任务训练，我们首先让 RL 聚焦于具有小随机化的关键阶段，然后过渡到完整任务设置。

**实验细节。** RL 策略的输入由 RL token（从两个腕部相机图像和一个基座相机图像产生）和额外的本体感受状态组成。根据任务不同，此辅助状态可能包括关节位置（螺丝）、末端执行器位姿（扎带、以太网和充电器）。我们使用 $\pi_{0.6}$ [33] 作为基础 VLA 策略。机器人以 50 Hz 的控制频率运行。在 14 维的每时间步动作空间下，这对应于 RL actor 的 140 维分块动作。我们在附录 B 中提供更多实现细节。

### B. 基线方法与消融实验

我们从预训练的 VLA 模型 $\pi_{0.6}$ [33] 开始。对于每个任务，我们收集 1-10 小时的遥操作演示。然后我们在训练 RL token 表征的同时微调 VLA 模型。这产生了我们在所有实验中使用的基础 VLA 策略。根据任务难度，我们运行 400 到 1000 个 episode 的 RL 训练。排除重置和各种开销后，每个实验产生大约 15 分钟到 5 小时的实际机器人数据。我们以每个任务的成功率来衡量性能，由人类操作员的二元奖励信号判定。我们还报告吞吐量，即每 10 分钟间隔内成功完成任务的次数，以评估在鲁棒性和速度方面的改进。我们对所有任务评估其关键阶段，并对两个更困难的任务——螺丝和扎带任务——在完整任务设置中进行评估。

我们将 RLT 与四种从经验中改进策略的基线方法进行比较。为了公平比较，我们使用相同数量的数据训练每种 RL 方法（见附录 C）。

- **HIL-SERL** [4]：与我们的方法类似，HIL-SERL 使用经验和干预的组合训练小型 actor 和 critic，但与 RLT 不同的是，它不使用预训练 VLA 的表征，而是使用为标准计算机视觉任务预训练的简单 ResNet 编码器。

- **Probe-Learn-Distill** [30]：PLD 学习一个残差策略，为每个单步动作输出一个残差。它通过超参数缩放残差，并将其与冻结 VLA 动作预测的一步相加来执行。

- **DSRL** [32]：DSRL 在流式 VLA 模型的潜在噪声空间中学习在线 RL 策略。它通过选择输入到冻结 VLA 模型动作生成器的噪声来"引导"VLA 动作生成。此方法隐式地将探索约束在 VLA 可生成的动作范围内，并在其模式之间进行探索。

- **DAgger** [41, 42]：我们在训练期间收集的人类干预数据上微调基础 VLA 模型。

我们还通过逐个移除方法的每个组件来隔离其贡献：

- **不使用 RL token（w/o RL token）**：用来自 [25] 的冻结 ImageNet 预训练 ResNet-10 编码器替代 RL token。
- **不使用分块（w/o Chunk）**：RL 策略输出单步动作（$C$=1）而非动作块。因为该策略需要以 50 Hz 运行，而以 50 Hz 查询基础 VLA 模型是不可行的，我们必须用 ResNet-10 编码器替代 RL token。
- **不使用 BC 正则化器（w/o BC Regularizer）**：在公式 (5) 中设置 $\beta$=0；策略仅使用 $Q$ 函数进行训练。
- **不使用直通（w/o Pass-Through）**：从公式 (4) 的策略输入中移除 $\bar{\mathbf{a}}$；RL actor 仅从状态和 RL token 生成动作。

*图 4:* **RLT 显著提升了吞吐量**，相比基础 VLA 策略，改善了每个任务关键阶段的速度和一致性。对于 VLA 策略容易出错的较困难任务，改进尤为显著。

*图 5:* **RLT 能够在多个任务中提升成功率。** 在 VLA 已经胜任的任务（如以太网任务）上，它保持成功率并提升吞吐量。对于基础 VLA 策略具有挑战性的任务（螺丝刀和扎带），RLT 带来了成功率的显著提升。

*图 6:* **与其他 RL 算法的比较。** 我们将 RLT 与近期 RL 文献中的几个基线进行比较。仅考虑单个动作而非动作块的方法（HIL-SERL、PLD）表现不佳。DSRL 导致高成功率，但在吞吐量上明显落后。

### C. 实验结果

**Q1：在线 RL 能否在基础 VLA 策略基础上提升？** 我们在两种场景中评估我们的方法：隔离关键阶段的*受控*设置和要求 RL 策略更鲁棒的*完整任务*设置。*在线 RL 在两种设置中都提升了基础模型的成功率和执行速度*。在受控设置中，RLT 在所有四个任务的关键阶段都实现了持续改进。即使在相对较简单的充电器和以太网任务上，基础策略已经实现了良好的可靠性，RLT 学习到的策略在关键阶段也快了大约 $3\times$。在更困难的扎带和螺丝刀任务上，成功率的提升更为显著。在完整任务评估中，由于任务早期部分（抓取/提起物体等）的累积误差，总体成功率较低，但 RLT 仍然在螺丝刀任务上将成功率提升了 40%，在扎带任务上提升了 60%。

**Q2：RLT 与替代方法相比表现如何？** 如图 6 所示，RLT *相比基线方法实现了吞吐量的显著提升*。我们在以太网任务上与四个基线进行比较。HIL-SERL 和 PLD——两种单步在线 RL 方法——在此任务上未能有效学习，该任务跨越数百步且具有稀疏奖励。没有动作分块，任务的视界非常长，价值函数更新无法有效传播稀疏奖励信号。对于这个较简单的任务，DAgger 和 DSRL 达到了与 RLT 相当的成功率（图 6），但在速度方面的改进明显较少。DAgger 是一种模仿学习方法，受限于人类演示和干预的速度。DSRL 是一种 RL 方法，强约束策略保持接近基础 VLA，提供稳定训练但改进潜力相对较小。相比之下，RLT 在匹配基础策略高成功率的同时，将平均完成步数相比基础策略减少了 $2\times$。

**Q3：每个组件的贡献有多大？** *所有四个设计选择——RL token、动作块、BC 正则化器和参考动作直通——都有有意义的贡献*。我们验证了方法中每个组件都提供了正向贡献（图 7）：用 ResNet-10 编码器替代 RL token 使吞吐量降低了 50%，证实我们的 token 编码了标准计算机视觉任务训练的现成编码器无法提供的操作相关结构。将块（$C$=10）替换为单步动作大幅增加了任务的有效视界，因为价值函数需要在更长的视界上执行信用分配。这也使得使用 RL token 运行我们的方法变得不可行。在实践中，单步变体无法可靠地匹配基础策略的性能。移除 BC 正则化器（$\beta$=0）导致性能的最大单项下降，因为它迫使 actor 仅凭 $Q$ 函数的梯度在完整动作空间中探索。移除参考动作直通会减慢学习速度，导致早期探索漂移，偶尔出现退化行为。此消融最终在这个较简单的任务上达到了 RLT 的性能，但在训练过程中经历了更多失败，如图 7 中的学习曲线所示。

*图 7:* **以太网任务不同训练阶段的吞吐量。** 消融研究表明我们方法的每个部分对良好性能都很重要，完整系统学习最快且最终表现最佳。值得注意的是，RLT 仅消耗 5 分钟关键阶段数据就超越了替代策略（总实验时间约 40 分钟）。从 actor 输入中移除参考动作（"w/o Pass-Through"）仍可达到最佳最终性能，但代价是学习更慢且训练过程中失败显著增多。

*图 8:* **以太网任务训练过程中的成功率评估。** RLT 在以太网插入任务上迅速匹配 VLA 策略的成功率，同时提升吞吐量。不使用参考动作传递或不使用 RL token 会导致学习变慢。

*图 9:* **以太网任务的速度。** RLT 显著提高了以太网任务的速度。最终策略甚至比专家遥操作产生的演示更快，且显著快于基础 VLA 模型。RL 关键插入阶段的一半 episode（黄色）比所有遥操作演示（绿色）都快。

**Q4：RLT 是否能产生更有效的涌现策略？** 除了汇总指标之外，在线 RL 的效果是机器人*执行任务方式*的质性变化。在以太网任务的关键阶段，我们可视化了遥操作演示、基础策略和最终 RL 策略的速度分布（图 9）。基础 VLA 在接触附近频繁表现出"探测"行为：它接近目标，略微后退，重新调整，然后再次尝试——有时在成功之前循环经历多次这样的尝试。RLT 则接近端口，以流畅的动作插入连接器。即使在第一次尝试失败时，RLT 也会施加压力并略微摇晃连接器以利用柔顺性，从而实现更快的插入。这种行为在演示数据中未曾出现，纯粹从在线探索中涌现，说明该方法可以超越模仿人类策略。

## VII. 结论

我们提出了 RLT，一种在大型预训练 VLA 提取的表征之上进行快速在线 RL 的方法。通过训练 VLA 暴露一个紧凑的表征，我们的方法使得一个轻量级的演员和评论家网络能够仅通过几个小时的真实世界练习，就改进高精度和精细的任务。在四个需要精度和速度的困难任务中，RLT 持续提升了成功率和执行速度，在每个任务最困难的阶段实现了高达 $3\times$ 的加速，并且在某些情况下，通过在线 RL 中涌现的策略超越了人类专家遥操作的速度。

虽然 RLT 提供了快速且高效的学习，但它确实需要在训练过程中额外的人工介入，以提供奖励信号、干预修正，以及在 RL（用于关键阶段）和基础策略（用于其他阶段）之间进行切换。原则上，其中一些组件可以被自动化，例如通过使用奖励模型和进度预测。基于 RLT 开发一个完全自主的 RL 改进流水线是未来工作中一个有前景的方向。更广泛地说，我们相信我们的方法代表了朝向不仅能从演示数据中学习、还能在实际工作中直接改进的机器人系统迈出的重要一步。当改进快速且可靠时，VLA 的预训练阶段只需为下游探索提供一个良好的初始化就足够了，而最成功和最高效的策略可以通过强化学习来发现。我们希望 RLT 能够成为迈向这一未来的一步。

## 致谢

机器人学是一项团队工作。我们感谢 Physical Intelligence 所有为这项工作各个方面做出贡献的人员，包括数据收集、机器人操作和机器人基础设施。我们感谢 Liam Murphy 和 Cameron Myers 在夹爪设计上的帮助。我们感谢 PI 的机器人操作员以及操作和标注团队。我们感谢 Connor Jacobsen 在网站和博客文章方面的帮助，Brian Ichter 在图表方面的帮助，Kyle Vedder 的校对工作，Claudio Guglieri 在博客文章可视化方面的帮助，以及 Donald Jewkes 和 Thomas Burton 在视频拍摄和剪辑方面的帮助。

## 附录

### A. 贡献

CX、LK 启动了该项目。CX 构建了在线 RL 的基础设施。JTS 设计并训练了 RL token。ME 构建了干预接口。AA 和 AE 设计并构建了夹爪和机器人硬件。CX、LK 设计了系统实现、任务套件和实验。SL、LK 在整个项目中提供建议。LK、CX、JTS、SL、ME 参与了论文撰写、插图制作和视频制作。

### B. 额外实验细节

首先，我们在目标任务上收集演示数据集；然后我们在单任务数据上微调基础 VLA 模型并训练 RL token，进行 2000 到 10000 个梯度步。在在线 RL 训练期间，VLA 随后被冻结。

在在线 RL 期间，我们从头初始化 RL 演员和评论家网络，对于扎带固定、以太网和充电器插入任务使用两层 MLP（隐藏维度 256）。对于更具挑战性的螺丝安装任务，我们使用更大的网络，由三层 MLP 组成，隐藏维度为 512。两种网络都接收冻结的基础 VLA 模型产生的 RL token、本体感受位置和速度作为输入。评论家使用两个 Q 函数的集成进行训练，遵循 Fujimoto et al. [19]，并使用两个 Q 函数的最小值来计算目标值。演员额外接收 VLA 模型产生的参考动作块，该参考动作块在训练时以 50% 的概率被遮蔽，在推理时始终提供。演员被参数化为一个具有小的固定标准差的高斯策略，从当前观测输出一个动作块 $\mathbf{a}_{t:t+C-1} \in \mathbb{R}^{C \times d}$，其中 $C$=10。为了提高样本效率，我们在训练期间以间隔 2 个控制步对动作块进行子采样，因此每秒数据大约为 RL 网络产生 25 个样本。在训练期间，当 RL 任务完成时，操作员提供稀疏的 +1 奖励。

对于螺丝安装和扎带固定任务，我们首先仅在关键阶段设置下开始 RL 训练。然后我们推进到完整任务阶段，首先运行基础模型完成任务的非关键阶段，当到达关键阶段时切换到 RL 策略。这种两阶段训练策略提高了训练效率，同时确保 RL 策略对基础策略在任务早期部分引起的初始分布具有鲁棒性。我们报告了收集约 5 小时数据后的策略性能。

### C. 基线方法的额外实验细节

对于所有基线方法，我们使用与我们方法相同的环境和动作空间设置——策略在增量动作空间中以 50 Hz 执行。

**PLD**：遵循原始论文，我们首先在 50 个基础策略展开上使用 Cal-QL [31] 预训练评论家网络以获得更好的样本效率。然后我们进入在线 RL 阶段。

**DSRL**：遵循原始实现，我们的实现预测一个 $(1, 32)$ 维的潜在动作，在第一个维度上重复 50 次以匹配我们动作块 VLA 的噪声输入空间。

**HIL-SERL**：遵循原始实现，我们使用 20 个演示回合初始化 RLPD 训练，并在整个训练过程中提供干预。然而，由于与原始系统（10 Hz）相比更高的控制频率（50 Hz）以及缺少用于减少探索空间的动作空间边界框，该方法在我们的设置中无法成功。

**DAgger**：我们使用演示数据和在线 RL 训练期间收集的相同干预数据的混合来微调我们的 VLA。

## References

- [1] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning. *arXiv preprint*, arXiv:2509.09674, 2025.
- [2] Yunfei Li, Xiao Ma, Jiafeng Xu, Yu Cui, Zhongren Cui, Zhigang Han, Liqun Huang, Tao Kong, Yuxiao Liu, Hao Niu, Wanli Peng, Jingchao Qiao, Zeyu Ren, Haixin Shi, Zhi Su, Jiawen Tian, Yuyang Xiao, Shenyu Zhang, Liwei Zheng, Hang Li, and Yonghui Wu. Gr-rl: Going dexterous and precise for long-horizon robotic manipulation, 2025. URL https://arxiv.org/abs/2512.01801.
- [3] Physical Intelligence. $\pi_{0.6}^*$: a VLA That Learns From Experience, 2025. URL https://arxiv.org/abs/2511.14759.
- [4] Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. *arXiv preprint arXiv:2410.21845*, 2024.
- [5] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning, 2026. URL https://arxiv.org/abs/2510.14830.
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In *arXiv preprint arXiv:2307.15818*, 2023.
- [7] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. *arXiv preprint arXiv:2406.09246*, 2024.
- [8] Physical Intelligence. $\pi_0$: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024.
- [9] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D'Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jingwei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020.
- [10] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation, 2023.
- [11] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.
- [12] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/2304.13705.
- [13] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. *The International Journal of Robotics Research*, page 02783649241273668, 2023.
- [14] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. *arXiv preprint arXiv:2501.09747*, 2025.
- [15] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint, 2024. URL https://github.com/Stanford-ILIAD/openvla-mini.
- [16] Physical Intelligence. $\pi_{0.5}$: a vision-language-action model with open-world generalization. In *9th Annual Conference on Robot Learning*, 2025.
- [17] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In *International conference on machine learning*, pages 1861–1870. Pmlr, 2018.
- [18] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*, 2015.
- [19] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. *arXiv preprint arXiv:1802.09477*, 2018.
- [20] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a Posteriori Policy Optimisation. In *International Conference on Learning Representations (ICLR)*, 2018. URL https://openreview.net/forum?id=S1ANxQW0b.
- [21] Marcel Hussing, Claas Voelcker, Igor Gilitschenski, Amir massoud Farahmand, and Eric Eaton. Dissecting deep rl with high update ratios: Combatting value divergence, 2024. URL https://arxiv.org/abs/2403.05996.
- [22] Xinyue Chen, Che Wang, Zijian Zhou, and Keith Ross. Randomized ensembled double q-learning: Learning fast without a model. *arXiv preprint arXiv:2101.05982*, 2021.
- [23] Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In *International Conference on Machine Learning*, pages 1577–1594. PMLR, 2023.
- [24] Henry Zhu, Justin Yu, Abhishek Gupta, Dhruv Shah, Kristian Hartikainen, Avi Singh, Vikash Kumar, and Sergey Levine. The ingredients of real-world robotic reinforcement learning. *arXiv preprint arXiv:2004.12570*, 2020.
- [25] Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, and Sergey Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 16961–16969. IEEE, 2024.
- [26] Allen Z. Ren, Justin Lidard, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In *Proceedings of the 2025 International Conference on Learning Representations (ICLR)*, 2025.
- [27] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. $\pi_{\text{RL}}$: Online rl fine-tuning for flow-based vision-language-action models. *arXiv preprint*, arXiv:2510.25889, 2025.
- [28] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. *arXiv preprint arXiv:2502.05450*, 2025.
- [29] Xiu Yuan, Tongzhou Mu, Stone Tao, Yunhao Fang, Mengke Zhang, and Hao Su. Policy decorator: Model-agnostic online refinement for large policy model. In *The Thirteenth International Conference on Learning Representations*, 2025.
- [30] Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi "Jim" Fan, Guanya Shi, and Yuke Zhu. Self-improving vision-language-action models with data generation via residual rl, 2025.
- [31] Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. *Advances in Neural Information Processing Systems*, 36:62244–62269, 2023.
- [32] Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. In *Proceedings of the 9th Conference on Robot Learning (CoRL)*, 2025.
- [33] Physical Intelligence. $\pi_{0.6}$ model card, 2025. URL https:////website.pi-asset.com/pi06star/PI06_model_card.pdf.
- [34] Nicolas Heess, Gregory Wayne, David Silver, Timothy Lillicrap, Tom Erez, and Yuval Tassa. Learning continuous control policies by stochastic value gradients. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/file/148510031349642de5ca0c544f31b2ef-Paper.pdf.
- [35] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In *Advances in neural information processing systems*, pages 3104–3112, 2014.
- [36] Seohong Park, Qiyang Li, and Sergey Levine. Flow q-learning. In *International Conference on Machine Learning (ICML)*, 2025.
- [37] Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, and Sergey Levine. Learning agile robotic locomotion skills by imitating animals. *RSS*, 2020.
- [38] Jan Peters, Katharina Mülling, and Yasemin Altün. Relative entropy policy search. In *Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence*, AAAI'10, page 1607–1612. AAAI Press, 2010.
- [39] Peter Dayan and Geoffrey E. Hinton. Using expectation-maximization for reinforcement learning. *Neural Computation*, 9(2):271–278, 1997. doi: 10.1162/neco.1997.9.2.271.
- [40] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. URL https://arxiv.org/abs/1805.00909.
- [41] Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. Hg-dagger: Interactive imitation learning with human experts, 2019. URL https://arxiv.org/abs/1810.02890.
- [42] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Geoffrey Gordon, David Dunson, and Miroslav Dudík, editors, *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, volume 15 of *Proceedings of Machine Learning Research*, pages 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr.press/v15/ross11a.html.
