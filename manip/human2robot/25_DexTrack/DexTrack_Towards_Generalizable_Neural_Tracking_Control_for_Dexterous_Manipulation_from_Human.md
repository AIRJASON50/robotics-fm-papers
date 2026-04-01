# DexTrack: 面向从人类参考中实现灵巧操作的可泛化神经跟踪控制

Xueyi Liu1,2, Jianibieke Adalibieke2, Qianwei Han2, Yuzhe Qin4, Li Yi1,3,2

1Tsinghua University 2Shanghai Qi Zhi Institute 3Shanghai AI Laboratory 4UC San Diego

项目网站: [meowuu7.github.io/DexTrack](https://meowuu7.github.io/DexTrack/)

## 摘要

我们致力于解决从人类参考中开发可泛化的神经 tracking 控制器用于灵巧操作的挑战。该控制器旨在控制灵巧机器人手根据运动学人-物交互定义的多种目的来操作不同物体。开发这样的控制器面临诸多困难:灵巧操作中复杂的接触动力学,以及对 adaptivity(适应性)、generalizability(泛化性)和 robustness(鲁棒性)的需求。当前的强化学习(reinforcement learning, RL)和轨迹优化方法往往因依赖任务特定的 reward 或精确的系统模型而表现不佳。我们提出一种方法,通过整理大规模成功的机器人 tracking demonstration(跟踪演示,即人类参考与机器人动作的配对数据)来训练神经控制器。利用数据飞轮(data flywheel),我们迭代地提升控制器性能,同时增加成功 tracking demonstration 的数量和质量。我们充分利用已有的 tracking demonstration,并精心整合 reinforcement learning 和 imitation learning 以提升控制器在动态环境中的性能。同时,为获取高质量的 tracking demonstration,我们在 homotopy optimization(同伦优化)方法中利用学习到的 tracking 控制器对每条轨迹单独优化跟踪效果。homotopy optimization 类似于 chain-of-thought(思维链),有助于解决具有挑战性的轨迹跟踪问题,从而增加 demonstration 的多样性。我们通过在仿真和真实世界中训练可泛化的神经控制器并进行评估来展示方法的成功。与领先的 baseline 相比,我们的方法在成功率上提升了 10% 以上。项目网站及动画结果可在 [DexTrack](https://meowuu7.github.io/DexTrack/) 查看。

*图 1: [DexTrack](https://meowuu7.github.io/DexTrack/) 从人类参考中学习可泛化的神经 tracking 控制器用于灵巧操作。它根据运动学参考生成手部动作指令,确保对输入轨迹的精确跟踪(图 (a)),泛化到涉及薄物体、复杂运动和精细手内操作的新颖且具有挑战性的任务(图 (b)),并展示对大幅运动学噪声的鲁棒性以及在真实场景中的实用性(图 (c))。运动学参考以橙色矩形和背景显示。*

## 1 引言

机器人灵巧操作是指机器人手灵巧地处理和操作物体,以精确和适应性地达到各种目标状态的能力。这一能力受到广泛关注,因为对于工具使用等目标而言,熟练的物体操作对于机器人与世界交互至关重要。此前已有大量工作致力于推动灵巧手的能力向人类水平的灵巧性和多功能性发展 (Rajeswaran et al., 2017; Chen et al., 2023; 2021; Akkaya et al., 2019; Christen et al., 2022; Zhang et al., 2023; Qin et al., 2022; Liu et al., 2022; Wu et al., 2023; Gupta et al., 2016; Wang et al., 2023; Mordatch et al., 2012; Liu et al., 2024a; Li et al., 2024)。这也与我们的愿景一致。

实现人类水平的机器人灵巧操作面临两大主要困难:接触丰富的操作中复杂的动力学使得优化变得困难 (Pang & Tedrake, 2021; Pang et al., 2023; Liu et al., 2024a; Jin, 2024),以及机器人需要掌握超越特定任务的广泛多样技能。此前的方法主要依赖无模型的强化学习(RL) (Chen et al., 2023; 2021; Akkaya et al., 2019; Christen et al., 2022; Zhang et al., 2023; Qin et al., 2022; Liu et al., 2022; Wu et al., 2023; Gupta et al., 2016; Wang et al., 2023) 或基于模型的轨迹优化(trajectory optimization, TO) (Pang & Tedrake, 2021; Pang et al., 2023; Jin, 2024; Hwangbo et al., 2018)。RL 需要任务特定的 reward 设计,限制了其泛化能力;TO 依赖于具有已知接触状态的精确动力学模型,限制了对新物体和新技能的适应性。一种有前景的替代方案是利用人类手-物操作参考(这些参考可以通过视频或运动合成广泛获取),并专注于控制灵巧手来跟踪这些参考。这种方法将高层任务规划与低层控制分离,将多样化技能的获取构建为开发通用 tracking 控制器的问题。

然而,挑战依然存在:运动学参考含噪声、人手与机器人手的形态差异、接触丰富的复杂动力学,以及物体几何形状和技能的多样性。

现有方法在这些问题上表现不佳,往往局限于没有手内操作的简单任务 (Christen et al., 2022; Zhang et al., 2023; Wu et al., 2023; Xu et al., 2023; Luo et al., 2024; Singh et al., 2024; Chen et al., 2024) 或某些特定技能 (Qin et al., 2022; Liu et al., 2024a; Rajeswaran et al., 2017)。

在本工作中,我们旨在开发一个通用的 tracking 控制器,能够跟踪跨越多种技能和多样物体的手-物操作参考。具体而言,给定一组仅包含运动学信息的人类手-物操作轨迹,该控制器经过优化后能够驱动机器人灵巧手操作物体,使得生成的手和物体轨迹能够精确模仿其对应的运动学序列。

我们期望该 tracking 控制器展现出强大的多功能性,能够很好地泛化以精确跟踪新的操作,并对大幅运动学噪声和意外参考状态具有强鲁棒性。

为实现上述具有挑战性的目标,我们得出三个关键观察:1) 学习对于处理异构参考运动噪声和将数据先验迁移到新场景至关重要,从而支持鲁棒且可泛化的 tracking 控制;2) 利用大规模、高质量的机器人 tracking demonstration(将运动学参考与 tracking 动作序列配对)可以监督并显著增强神经控制器的能力,正如计算机视觉和自然语言处理中的数据缩放定律所证明的那样 (OpenAI, 2023; Brown et al., 2020);3) 获取大量高质量的 tracking demonstration 是具有挑战性的,但我们可以利用数据飞轮(data flywheel) (Chiang et al., 2024; Bai et al., 2023) 以自举(bootstrapping)的方式迭代改进 tracking 控制器并扩展 demonstration。

基于上述观察,我们提出 DexTrack,一种由人类参考引导的灵巧操作神经 tracking 控制器。具体来说,给定一组人类手-物操作轨迹,我们首先将其重定向到运动学机器人灵巧手序列,形成一组参考运动作为数据预处理。然后,我们的方法在挖掘成功的机器人 tracking demonstration 和使用挖掘到的 demonstration 训练控制器之间交替进行。为确保数据飞轮有效运作,我们引入两个关键设计。首先,我们精心整合 reinforcement learning 和 imitation learning 技术来训练神经控制器,确保其性能随着更多 demonstration 的加入而提升,同时保持对意外状态和噪声的鲁棒性。其次,我们开发了一种逐轨迹 tracking 方案,利用训练好的控制器通过 homotopy optimization 方法挖掘多样且高质量的 tracking demonstration。该方案将控制器中的 tracking prior(跟踪先验)迁移到单条轨迹,以简化逐轨迹 tracking 从而获得更高质量的 demonstration。此外,该方案会将一个 tracking 参考转换为一系列逐渐简化的参考运动,使得从简单到复杂地跟踪这些参考有助于更好地跟踪原始参考运动。这类似于 chain-of-thought,非常适合跟踪复杂的参考运动以增加 demonstration 的多样性。上述两个设计与迭代训练相结合,使 DexTrack 能够成功跟踪新颖且具有挑战性的人类参考。

我们在两个数据集上的具有挑战性的操作跟踪任务中证明了我们方法的优越性,并与先前方法进行了比较。这两个数据集描述了日常场景和功能性工具使用场景中富有表现力的手-物交互,涉及复杂的物体运动、困难且精细的手内重定向、薄物体交互以及频繁的手-物丰富接触变化。

我们在仿真环境(即 Isaac Gym (Makoviychuk et al., 2021))中进行了大量实验,并在真实世界中进行了评估,以证明我们的 tracker 在完成广泛操作跟踪任务方面的有效性、泛化能力和鲁棒性,甚至能出色地跟踪新的操作轨迹(图 1)。

我们的方法在定量和定性方面均成功超越了先前方法,比此前表现最佳的方法提高了 10% 以上的成功率。此外,我们进行了进一步分析,展示了控制器的各种恢复行为,证明了其对意外情况的鲁棒性。通过充分的消融实验验证了我们设计的有效性。

我们的贡献包括三方面:

- 我们提出了一个可泛化的神经 tracking 控制器,通过迭代挖掘和整合高质量的 tracking demonstration 来逐步提升性能。

- 我们引入了一种协同结合 reinforcement learning 和 imitation learning 的训练方法。该方法利用丰富的高质量机器人 tracking demonstration 产生一个可泛化、多功能且鲁棒的控制器。

- 我们开发了一种逐轨迹优化方案,在 homotopy optimization 框架中使用我们的 tracking 控制器。我们提出了一种数据驱动的方式来生成 homotopy path(同伦路径),从而能够解决具有挑战性的 tracking 问题。

## 2 相关工作

为机器人配备人类水平的灵巧操作技能对于未来发展至关重要。先前方法要么依赖基于模型的轨迹优化,要么依赖无模型的强化学习(RL)。基于模型的方法面临动力学复杂性的挑战,通常需要近似处理 (Pang et al., 2023; Jin, 2024; Pang & Tedrake, 2021)。无模型方法使用 RL (Rajeswaran et al., 2017; Chen et al., 2023; 2021; Christen et al., 2022; Zhang et al., 2023; Qin et al., 2022; Liu et al., 2022; Wu et al., 2023; Gupta et al., 2016; Wang et al., 2023; Mordatch et al., 2012),专注于使用任务特定 reward 的目标驱动任务,限制了其在多样化任务间的泛化。

我们的工作探索灵巧操作的通用控制器。此外,通过模仿运动学轨迹进行学习近年来已成为为智能体配备各种表现力丰富技能的流行方式 (Jenelten et al., 2023; Luo et al., 2023b; a)。DTC (Jenelten et al., 2023) 提出了一种策略,可以结合基于模型的运动规划和 RL 的优势来克服 RL 的样本效率低下问题。

在类人体运动跟踪领域,PHC (Luo et al., 2023b) 提出了一种有效的基于 RL 的训练策略来开发通用的类人体运动 tracker。最近,OmniGrasp (Luo et al., 2024) 提出训练一个通用的抓取和轨迹跟随 policy。该 policy 可以泛化到未见过的物体以及跟踪新的运动。然而,他们考虑的运动仍然局限于抓取和轨迹跟随,留下了跟踪更有趣且更困难的轨迹(例如精细手内操作)的问题在很大程度上未被探索。在本文中,我们专注于这些困难且具有挑战性的操作。此外,我们的工作也与近期结合 RL 和 imitation learning 的尝试相关。为了克服 RL 的样本效率低下问题并促进收敛,已开发了各种方法旨在通过 demonstration 增强 RL 训练 (Sun et al., 2018; Hester et al., 2017; Booher et al., 2024; Liu et al., 2023)。在我们的工作中,我们希望利用高质量的 demonstration 来引导智能体的探索。与先前 demonstration 已经现成可用的工作不同,在我们的任务中获取足够数量的高质量机器人 tracking demonstration 仍然是一个重大挑战。

## 3 方法

*图 2: [DexTrack](https://meowuu7.github.io/DexTrack/) 从人类参考中学习可泛化的神经 tracking 控制器用于灵巧操作。它在使用丰富且高质量的机器人 tracking demonstration 训练 tracking 控制器与通过 homotopy optimization 方案利用 tracking 控制器改进数据之间交替进行。*

术语和符号说明。
灵巧操作 "tracking" 涉及控制机器人手模仿一个运动学手-物状态序列(即目标轨迹,记为 $\{\hat{\mathbf{s}}_{n}\}_{n=0}^{N}$)。这些 "kinematic reference"(运动学参考)从人类操作轨迹重定向而来,其中 $\hat{\mathbf{s}}_{n}$ 表示时间步 $n$ 的机器人手状态和物体位姿。一个 "tracking demonstration" 将运动学参考 $\{\hat{\mathbf{s}}_{n}\}$ 与专家动作序列 $\{\mathbf{a}_{n}^{L}\}$ 配对,引导机器人从 $\mathbf{s}_{0}=\hat{\mathbf{s}}_{0}$ 出发,达到与 $\{\hat{\mathbf{s}}_{n}\}_{n=0}^{N}$ 对齐的状态序列 $\{\mathbf{s}_{n}\}_{n=0}^{N}$。一个 "robust"(鲁棒的)控制器能够容忍运动学噪声和不可达状态等干扰。如果控制器在未见场景(如新物体和新运动)上表现良好,则表现出较高的 "generalization ability"(泛化能力);如果尽管上下文发生变化(如接触和动力学改变)仍能保持有效性,则具有 "adaptivity"(适应性)。

给定一组运动学人-物操作轨迹,我们希望为灵巧机器人手学习一个可泛化的神经 tracking 控制器。

该问题具有挑战性,原因在于精确灵巧操作的困难(受底层复杂动力学制约),以及对控制器泛化能力和鲁棒性的高要求。

我们通过结合 reinforcement learning(RL)和 imitation learning(IL)来训练可泛化的 tracking 控制器以应对这些挑战:利用高质量且多样的机器人 tracking demonstration 的监督来共同缓解复杂问题的求解难度,并通过 RL 探索提升 policy 的鲁棒性。我们引入一种单轨迹 tracking 方案来挖掘由运动学参考和动作序列配对组成的 tracking demonstration。

对于每条运动学参考,我们使用 RL 训练一个轨迹特定的 policy 来生成跟踪该参考的动作。为克服 RL 的局限性,我们提出通过 homotopy optimization 方案利用 tracking 控制器来增强 demonstration 的质量和多样性。通过以自举方式迭代挖掘更好的 demonstration 和优化控制器,我们开发了一个有效的、可泛化的 tracking 控制器。

我们将在第 3.1 节解释如何从 demonstration 中学习神经 tracking 控制器,在第 3.2 节解释如何挖掘高质量 demonstration,在第 3.3 节解释如何在学习控制器和挖掘 demonstration 之间迭代。

### 3.1 从 Demonstration 中学习神经 Tracking 控制器

给定一组人-物操作轨迹和一组高质量的机器人 tracking demonstration,我们的目标是学习一个有效且可泛化的神经 tracking 控制器。

在开始时,我们将人类手-物操作重定向到机器人手作为数据预处理步骤。我们结合 RL 和 IL 来开发可泛化且鲁棒的神经 tracking 控制器。

通过模仿多样且高质量的机器人 tracking demonstration,我们可以有效地让 tracking 控制器掌握多种操作技能,并赋予其较高的泛化能力。

同时利用 RL 的能力,控制器避免过拟合到仅限于狭窄分布的成功 tracking 结果,从而在面对动态状态干扰时保持鲁棒的性能。

具体而言,我们为训练 tracking 控制器设计了基于 RL 的学习方案,包括精心设计的动作空间、观测量和为操作 tracking 任务量身定制的 reward。我们还引入了基于 IL 的策略,使 tracking 控制器受益于模仿高质量 demonstration 数据。

通过整合这两种方法,我们有效地解决了可泛化 tracking 控制的复杂问题。

神经 tracking 控制器。
在我们的形式化中,神经 tracking 控制器充当一个根据 tracking policy $\pi$ 与环境交互的智能体。在每个时间步 $n$,policy 观察 observation $\mathbf{o}_{n}$ 和下一个目标 $\hat{\mathbf{s}}_{n+1}$(指定为机器人手和物体的目标状态),然后计算动作的分布。智能体从 policy 中采样一个动作 $\mathbf{a}_{n}$,即 $\mathbf{a}_{n}\sim\pi(\cdot|\mathbf{o}_{n},\hat{\mathbf{s}}_{n+1})$。observation $\mathbf{o}_{n}$ 主要包含当前状态 $\mathbf{s}_{n}$ 和物体几何信息。

执行动作后,机器人灵巧手与物体发生物理交互,手和物体根据环境动力学转移到下一个状态,表示为 $\mathbf{s}_{n+1}\sim p(\cdot|\mathbf{s}_{n},\mathbf{a}_{n})$。

有效的 tracking 控制器应确保生成的手和物体状态与各自的下一个目标状态紧密对齐。

强化学习。
在基于 RL 的训练方案中,智能体在每次状态转移后获得 reward $r_{n}=r({\mathbf{s}}_{n},\mathbf{a}_{n},\hat{\mathbf{s}}_{n+1},\mathbf{s}_{n+1})$。训练目标是最大化折扣累积 reward:

$$J=\mathbb{E}_{p(\tau|\pi)}\left[\sum_{n=0}^{N-1}\gamma^{n}r_{n}\right],$$ \tag{1}

其中 $p(\tau|\pi)=p({\mathbf{s}}_{0})\prod_{n=0}^{N-1}p(\mathbf{s}_{n+1}|\mathbf{o}_{n},\mathbf{a}_{n})\pi(\mathbf{a}_{n}|\mathbf{s}_{n},\hat{\mathbf{s}}_{n+1})$ 是智能体转移轨迹 $\tau=(\mathbf{s}_{0},\mathbf{a}_{0},r_{0},...,\mathbf{s}_{N-1},\mathbf{a}_{N-1},r_{N-1},\mathbf{s}_{N})$ 的似然。折扣因子 $\gamma\in[0,1)$ 决定了 policy 的有效 horizon 长度。

在 tracking 控制问题中,下一个目标状态 $\hat{\mathbf{s}}_{n+1}$ 通常由运动学参考序列中的后续手部状态和物体状态组成。我们使用比例微分(PD)控制器来控制机器人手,遵循先前文献 (Luo et al., 2024; 2023b; Christen et al., 2022; Zhang et al., 2023)。动作 $\mathbf{a}_{n}$ 包含所有手指关节的目标位置指令。

为提高 RL 的样本效率,我们不让 tracking policy 学习绝对位置目标,而是引入 residual action space(残差动作空间)。具体来说,我们引入一条 baseline 手部轨迹,训练 policy 在每个时间步学习相对目标的残差 $\Delta\mathbf{a}_{n}$。baseline 轨迹在 tracking 问题中始终可用,可以简单地设置为运动学参考轨迹。

在每个时间步 $n$,我们通过 $\mathbf{a}_{n}=\mathbf{s}_{n}^{b}+\sum_{k=0}^{n}\Delta\mathbf{a}_{k}$ 计算位置目标,其中 $\mathbf{s}_{n}^{b}$ 是 baseline 轨迹中第 $n$ 步的手部状态。

每个时间步 $n$ 的 observation 编码了当前手部和物体状态、baseline 轨迹、动作、速度和物体几何信息:

$$\mathbf{o}_{n}=\{\mathbf{s}_{n},\dot{\mathbf{s}}_{n},\mathbf{s}^{b}_{n},\mathbf{a}_{n},\text{feat}_{\text{obj}},\text{aux}_{n}\},$$ \tag{2}

其中 $\text{feat}_{\text{obj}}$ 是由预训练物体点云编码器生成的物体特征。我们还引入了辅助特征 $\text{aux}_{n}$,基于可用状态计算得出,为智能体提供更多信息性上下文。更多细节将在附录 A 中说明。

我们用于操作 tracking 的 reward 鼓励转移后的手部状态和物体状态与各自的参考状态紧密匹配,同时促进手-物亲和性:

$$r=w_{o,p}r_{o,p}+w_{o,q}r_{o,q}+w_{\text{wrist}}r_{\text{wrist}}+w_{\text{finger}}r_{\text{finger}}+w_{\text{affinity}}r_{\text{affinity}},$$ \tag{3}

其中 $r_{o,p},r_{o,q},r_{\text{wrist}},r_{\text{finger}},r_{\text{affinity}}$ 分别代表物体位置、物体朝向、手腕、手指和手-物亲和性的 reward,而 $w_{o,p},w_{o,q},w_{\text{wrist}},w_{\text{finger}},w_{\text{affinity}}$ 是对应的权重。reward 计算的详细信息见附录 A。

模仿学习。
基于 RL 的学习方案受到样本效率低下及无法处理多个 tracking 问题的限制,难以解决可泛化的 tracking 控制问题,这在我们的早期实验中已得到验证。因此,我们提出基于 IL 的策略,将成功的、丰富的、多样的 "tracking knowledge"(跟踪知识)蒸馏到 tracking 控制器中。具体来说,我们训练 tracking 智能体来模仿大量高质量的机器人 tracking demonstration。

这种方法有效地引导智能体产生能够成功跟踪参考状态的 "expert action"(专家动作)。此外,通过模仿多样的 tracking demonstration,智能体可以避免在具有挑战性的 tracking 场景中反复遭遇低 reward,同时防止对较简单任务的过度利用。

形式上,长度为 $N$ 的机器人 tracking demonstration 由运动学参考序列 $(\hat{\mathbf{s}}_{0},...,\hat{\mathbf{s}}_{N})$ 和专家的状态-动作轨迹 $(\mathbf{s}_{0}^{L},\mathbf{a}_{0}^{L},...,\mathbf{s}_{N-1}^{L},\mathbf{a}_{N-1}^{L},\mathbf{s}_{N}^{L})$ 组成。

除了 actor loss 之外,我们加入一个动作监督损失,使 policy 在每个时间步的预测偏向于 demonstration 中对应的专家动作:

$$\mathcal{L}_{a}=\mathbb{E}_{\mathbf{a}_{n}\sim\pi(\cdot|\mathbf{o}_{n},\hat{\mathbf{s}}_{n+1})}\|\mathbf{a_{n}}-\mathbf{a}^{L}_{n}\|.$$ \tag{4}

这种引导使得 policy 的探索能够受到这些 demonstration 的启发,最终加速收敛并提升在复杂问题上的性能。从 IL 的角度来看,RL 探索向状态引入了噪声,使得模仿更加鲁棒,这与 DART (Laskey et al., 2017) 的思路类似。

由于在 tracking 控制任务中,智能体不应也不会探索距离参考状态太远的状态,因此同时使用模仿损失和 RL reward 来优化 policy 是可行的。

### 3.2 通过 Homotopy Optimization 方案使用神经控制器挖掘高质量机器人 Tracking Demonstration

为了从运动学参考轨迹 $(\hat{\mathbf{s}}_{0},...,\hat{\mathbf{s}}_{N})$ 准备用于训练 tracking 控制器的 demonstration,我们需要推断出能够成功跟踪参考序列的动作序列 $(\mathbf{a}^{L}_{0},...,\mathbf{a}^{L}_{N-1})$。

一种直接的方法是利用 RL 训练单轨迹 tracking policy $\pi$,并直接使用其生成的动作序列。

然而,仅依赖这种策略往往无法提供多样且高质量的 tracking demonstration 数据集,因为 RL 在灵巧操作固有挑战面前表现挣扎。

为了增强机器人 tracking demonstration 的多样性和质量,我们提出将 tracking 控制器与 homotopy optimization 方案结合使用,以改善逐轨迹 tracking 结果。

基于 RL 的单轨迹 tracking。
获取 demonstration 的基本方法是利用 RL 解决逐轨迹 tracking 问题,并将生成的动作序列作为 demonstration。给定一条运动学参考轨迹,我们的目标是优化一个 tracking policy $\pi$,使其能够精确跟踪该参考轨迹。在每个时间步,policy $\pi$ 观察当前状态 $\mathbf{s}_{n}$ 和下一个目标状态 $\hat{\mathbf{s}}_{n+1}$,并预测当前动作 $\mathbf{a}_{n}$ 的分布。

Policy $\pi$ 被优化以最小化每个时间步 $n$ 的转移状态 $\mathbf{s}_{n+1}\sim p(\cdot|\mathbf{s}_{n},\mathbf{a}_{n})$ 与目标状态 $\hat{\mathbf{s}}_{n+1}$ 之间的差异。与我们为 tracking 控制器的 RL 学习方案设计类似,我们采用 residual action space。对于每个 tracking 任务,我们引入一条 baseline 轨迹 $(\mathbf{s}^{b}_{0},...,\mathbf{s}^{b}_{N})$ 并仅使用 RL 学习 residual policy。observation 和 reward 遵循与 tracking 控制器相同的设计(见公式 8 和公式 10)。

一旦 $\pi$ 优化完成,我们可以在每个时间步从 $\pi$ 采样动作 $\mathbf{a}_{n}$。通过迭代地使用预测动作转移到下一个状态并查询 policy $\pi$ 生成新动作,我们可以获得输入运动学参考轨迹的专家动作序列 $(\mathbf{a}_{0}^{L},...,\mathbf{a}_{N-1}^{L})$。

迁移 "tracking prior"(跟踪先验)。
为改善 demonstration,我们设计了一种策略,利用已经编码了大量轨迹 tracking "知识"(即 "tracking prior")的 tracking 控制器来改进轨迹特定的 tracking policy。

具体来说,为跟踪参考轨迹 $(\hat{\mathbf{s}}_{0},...,\hat{\mathbf{s}}_{N})$,我们首先使用 tracking 控制器来跟踪它,将 baseline 轨迹设为参考序列。然后我们将 baseline 轨迹调整为生成的动作序列,并使用基于 RL 的单轨迹 tracking 方法重新优化 residual policy。这种方法可以帮助我们找到更好的 baseline 轨迹,促进单轨迹 tracking policy 学习并改善逐轨迹 tracking 结果。

Homotopy optimization 方案。
用从自身挖掘的数据训练控制器可能引入偏差并降低多样性,阻碍其泛化能力。为解决这个问题,我们提出 homotopy optimization 方案来改善逐轨迹 tracking 性能并解决此前无法解决的单轨迹 tracking 问题。对于 tracking 问题 $T_{0}$,homotopy optimization 不是直接求解它,而是迭代求解优化路径中的每个 tracking 任务,例如 $(T_{K},T_{K-1},...,T_{0})$,最终求解 $T_{0}$,这与 "chain-of-thought" (Wei et al., 2022) 的思路类似。首先,我们利用基于 RL 的 tracking 方法求解 $T_{K}$。之后,我们通过基于 RL 的 tracking 方法求解每个任务 $T_{m}$,将 baseline 轨迹设为 $T_{m+1}$ 的 tracking 结果。从其他任务迁移 tracking 结果有助于建立更好的 baseline 轨迹,最终产生更高质量的 tracking 结果。

寻找有效的 homotopy optimization path。
虽然利用 homotopy optimization 方案解决此前无法解决的问题已被证明对许多任务有效,但这依赖于识别有效的优化路径。一种直接的方法是暴力搜索。具体来说,给定一组运动学参考,我们首先优化它们的逐轨迹 tracking 结果。然后根据运动学参考轨迹对之间的相似性为每个任务识别邻居。接下来,我们迭代地从邻居任务迁移优化结果并重新优化每个 tracking 任务的 residual policy。对于特定任务,我们将提供比其运动学轨迹更好 baseline 轨迹的邻居视为有效的 "parent task"(父任务)。

在达到最大迭代次数 $K$ 后,我们可以通过从 $T_{0}$ 开始回溯有效的 "parent task" 来找到任务 $T_{0}$ 的有效 homotopy path。如果对于每个 $0\leq m\leq K$,$T_{m+1}$ 是 $T_{m}$ 的有效 parent task,则将 $(T_{K},...,T_{0})$ 定义为 $T_{0}$ 的有效 homotopy path。

学习 homotopy 生成器以高效规划 homotopy path。
为每条轨迹的 tracking 寻找有效的 homotopy optimization path 计算代价高且在推理时不实际。为解决这个问题,我们提出从小数据集学习 homotopy path 生成器 $\mathcal{M}$,使其能够高效地为其他 tracking 任务生成有效的 homotopy path。识别 homotopy path 的关键问题在于找到有效的 "parent task"。我们将此问题重新表述为 tracking 任务变换问题,目标是让生成器 $\mathcal{M}$ 为每个 tracking 任务 $T_{0}$ 提供有效 "parent task" 的分布:$\mathcal{M}(\cdot|T_{0})$,考虑到一个 tracking 任务可能有多个有效的 "parent task"。

一旦 $\mathcal{M}$ 训练完成,我们可以通过迭代寻找 parent task 来找到 homotopy path。

我们提出训练条件扩散模型(conditional diffusion model)作为 tracking 任务变换器,利用其强大的分布建模能力。给定一组 tracking 任务(以手和物体的运动学参考轨迹以及物体几何形状为特征),我们首先训练扩散模型来捕获 tracking 任务的分布。

为将该扩散模型微调为条件模型,我们首先在 tracking 任务数据集中搜索有效的 homotopy path。这产生了一组 tracking 任务 $T_{c}$ 及其对应有效 "parent task" $T_{p}$ 的配对数据。然后我们使用这些数据将扩散模型调整为条件扩散模型,使得 $T_{p}\sim\mathcal{M}(\cdot|T_{c})$。训练好的 $\mathcal{M}$ 可以通过从 $T_{0}$ 开始递归寻找 parent task 来高效地为 tracking 任务提出有效的 homotopy path,得到 $(T_{K},...,T_{0})$,其中对所有 $0\leq m\leq K-1$ 有 $T_{m+1}\sim\mathcal{M}(\cdot|T_{m})$。

### 3.3 通过迭代优化改进 Tracking 控制器

我们采用迭代方法,在使用丰富的机器人 tracking demonstration 训练 tracking 控制器与利用控制器整理更多样、更高质量的 demonstration 之间交替进行。我们的方法分为三个阶段。

在第一阶段,我们采样一小组 tracking 任务,通过对每个任务应用 RL 获取单轨迹 tracking 结果来生成初始 demonstration 集。使用这些 demonstration,我们用 RL 和 IL 训练 tracking 控制器。在此阶段,我们不训练 homotopy path 生成器,因为可用于训练的有效 homotopy path 数量有限,模型的泛化能力会受限。

在第二阶段,我们从剩余任务中根据控制器的 tracking 误差加权采样一批轨迹。然后使用带有 tracking prior 的 RL 优化逐轨迹 tracking,搜索 homotopy path,并根据得到的数据训练 homotopy path 生成器。所有成功 tracked 轨迹的最佳 tracking 结果被整理为新的 demonstration 集,用于重新训练 tracking 控制器。

在第三阶段,我们从剩余集合中重新采样 tracking 任务,并利用 RL、tracking 控制器和 homotopy 生成器整理另一组 tracking demonstration。这最终的 demonstration 集用于优化 tracking 控制器,得到我们的最终模型。

## 4 实验

我们进行了大量实验来评估 tracking 控制器的有效性、泛化能力和鲁棒性。在两个包含复杂日常操作任务的 HOI 数据集上测试,我们的方法通过仿真和真实世界评估进行了验证(见第 4.1 节)。我们与强 baseline 进行比较,展示了方法的优越性。我们的控制器成功处理了新的操作,包括精细动作、薄物体和动态接触(见第 4.2 节),而先前方法无法很好地泛化。平均而言,与此前最佳方法相比,我们的方法将 tracking 成功率提升了 10% 以上。此外,我们分析了其对大幅运动学噪声(如不现实状态和大量穿透)的鲁棒性(见第 4.3 节)。

### 4.1 实验设置

数据集。
我们在两个公开的人-物交互数据集上测试方法:
GRAB (Taheri et al., 2020)(包含日常交互)和
TACO (Liu et al., 2024b)(包含功能性工具使用交互)。
在仿真中,我们使用 Allegro 手(URDF 改编自 IsaacGymEnvs (Makoviychuk et al., 2021));在真实世界实验中,由于硬件限制,使用 LEAP 手 (Shaw et al., 2023)。人-物交互轨迹使用 PyTorch_Kinematics (Zhong et al., 2024) 进行重定向以创建机器人手-物序列。我们完整重定向了 GRAB 和 TACO 数据集,分别产生 1,269 和 2,316 条机器人手操作序列。GRAB 上的评估侧重于测试模型对未见交互序列的泛化能力。

具体来说,我们使用受试者 s1 的序列(197 条序列)作为测试数据,其余轨迹作为训练集。

对于 TACO 数据集,我们遵循作者建议的泛化评估设置 (Liu et al., 2024b) 将数据集分为包含 1,565 条轨迹的训练集和四个不同难度级别的测试集。主要定量结果在第一级测试集上报告。更多细节见附录 C。

评估指标。
我们引入五个指标来评估 tracking 准确性和任务成功率:1) 逐帧平均物体旋转误差:$R_{\text{err}}=\frac{1}{N+1}\sum_{n=0}^{N}\text{Diff\_Angle}(\mathbf{q}_{n},\hat{\mathbf{q}}_{n})$,其中 $\hat{\mathbf{q}}_{n}$ 和 $\mathbf{q}_{n}$ 分别为参考和 tracked 朝向。
2) 逐帧平均物体平移误差:$T_{\text{err}}=\frac{1}{N+1}\sum_{n=0}^{N}\|\mathbf{t}_{n}-\hat{\mathbf{t}}_{n}\|$,其中 $\mathbf{t}_{n}$ 和 $\hat{\mathbf{t}}_{n}$ 分别为 tracked 和参考平移。3) 逐帧平均手腕位置和旋转误差:$E_{\text{wrist}}=\frac{1}{N+1}\sum_{n=0}^{N}\left(0.5\,\text{Diff\_Angle}(\mathbf{q}_{n}^{\text{wrist}},\hat{\mathbf{q}}_{n}^{\text{wrist}})+0.5\|\mathbf{t}_{n}^{\text{wrist}}-\hat{\mathbf{t}}_{n}^{\text{wrist}}\|\right)$,其中 $\mathbf{q}_{n}^{\text{wrist}}$ 和 $\hat{\mathbf{q}}_{n}^{\text{wrist}}$ 为手腕朝向,$\mathbf{t}_{n}^{\text{wrist}}$ 和 $\hat{\mathbf{t}}_{n}^{\text{wrist}}$ 为平移。4) 逐帧逐关节平均位置误差:$E_{\text{finger}}=\frac{1}{N+1}\sum_{n=0}^{N}\left(\frac{1}{d}\|\mathbf{\theta}_{n}^{\text{finger}}-\hat{\mathbf{\theta}}_{n}^{\text{finger}}\|_{1}\right)$,其中 $\mathbf{\theta}$ 表示手指关节位置,$d$ 为自由度数。
5) 成功率:如果 $T_{\text{err}}$、$R_{\text{err}}$ 和 $0.5E_{\text{wrist}}+0.5E_{\text{finger}}$ 均低于阈值,则认为 tracking 尝试成功。成功率在两个阈值下计算:$10\text{cm}$-$20^{\circ}$-$0.8$ 和 $10\text{cm}$-$40^{\circ}$-$1.2$。

Baseline 方法。
据我们所知,此前没有基于模型的方法直接解决灵巧操作的 tracking 控制问题。大多数现有方法专注于使用简化动力学模型的单一目标驱动轨迹优化 (Jin, 2024; Pang et al., 2023; Pang & Tedrake, 2021),限制了它们在我们框架中可泛化 tracking 控制器的适用性。因此,我们主要与无模型方法进行比较:1) DGrasp (Christen et al., 2022):改编为通过将序列分成 10 帧的子序列来进行 tracking,每个子序列逐步求解。2) PPO (OmniGrasp rew.):我们重新实现了 OmniGrasp 的 reward (Luo et al., 2024) 来训练跟踪物体轨迹的 policy。3) PPO (w/o sup., tracking rew.):我们使用 PPO 配合我们提出的 tracking reward 和 observation 设计来训练 policy。

训练和评估设置。
我们使用 PPO (Schulman et al., 2017)(在 rl_games (Makoviichuk & Makoviychuk, 2021) 中实现),并使用 Isaac Gym (Makoviychuk et al., 2021) 进行仿真。逐轨迹 tracker 和 tracking 控制器均使用 8192 个并行环境进行训练。灵巧手的每个手指关节位置增益和阻尼系数分别设为 20 和 1。评估结果在 1000 个并行环境中取平均,真实世界评估使用 LEAP (Shaw et al., 2023) 配合 Franka 机械臂和 FoundationPose (Wen et al., 2023) 进行物体状态估计。更多细节见附录 C。

### 4.2 灵巧操作的可泛化 Tracking 控制

*图 3: 对不合理状态的鲁棒性。请查看[我们的网站](https://meowuu7.github.io/DexTrack/)和[视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

*图 4: 定性比较。请查看[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

*表 1: 定量评估。加粗红色和斜体蓝色分别表示最佳和次佳值。"Ours (w/o) data" 和 "Ours (w/o data, w/o homotopy)" 是关于 imitation learning 中使用的机器人 tracking demonstration 质量的两个消融版本(详见第 5 节)。*

我们展示了 tracking 控制器在涉及具有挑战性操作和新颖薄物体的未见轨迹上的泛化能力和鲁棒性。我们的控制器可以轻松处理精细动作、微妙的手内重定向和富有表现力的功能性操作,即使面对薄物体也不例外。

如表 1 所示,在两个数据集上的两个不同阈值下,与表现最佳的 baseline 相比,我们取得了显著更高的成功率。图 4 提供了定性示例和比较。我们展示了方法的真实世界有效性及相对于最佳 baseline 的优越性(图 4,表 2)。动画结果请访问我们的[项目网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)。

有趣的手内操作。
我们的方法有效地泛化到新颖、复杂且具有挑战性的功能性操作,以微妙的手内重定向为特征,这对于精确的工具使用任务至关重要。例如,在图 4a 中,铲子被提起、倾斜并通过精细的手指运动重新定向以完成搅拌动作。类似地,在图 4c 中,小铲子仅通过最小的手腕调整便被重新定向。这些结果展示了我们控制器的鲁棒性和泛化能力,优于在基本提起动作上都困难重重的 PPO baseline。

薄物体的精细操作。
我们的方法也很好地泛化到涉及薄物体的具有挑战性操作中。在图 4b 中,尽管薄铲子结构复杂且 CAD 模型部分缺失,我们的方法仍然成功地用第二和第三根手指紧紧抓握并控制了它。类似地,在图 4e 中,我们的控制器熟练地提起并操作了一支薄长笛,而表现最佳的 baseline 在初始抓握上就遇到了困难。这些结果突显了我们方法在处理复杂和精细操作方面的优势。

*表 2: 真实世界定量比较。加粗红色数字表示最佳值。*

真实世界评估和比较。
我们直接将 tracking 结果迁移到真实世界,以评估 tracking 质量并评估基于状态的控制器对状态估计器噪声的鲁棒性。成功率在三个阈值下测量并与最佳 baseline 进行比较。表 2 总结了在迁移控制器设置下的逐物体成功率(对其操作轨迹取平均)。如图 4f 和 4g 所示,我们使机器人能够在真实场景中跟踪复杂物体运动并成功提起难以抓握的圆苹果。而 baseline 则失败了。更多细节见附录 B.2。

### 4.3 进一步分析和讨论

对运动学参考运动中噪声的鲁棒性。
尽管图 4c(第 2、3 帧)和图 4a(第 2 帧)中存在严重的手-物穿透,手部仍然与物体有效交互,突显了我们的 tracking 控制器在具有挑战性场景中的韧性。

对不合理参考的鲁棒性。
如图 3 所示,我们的方法不受运动学参考中具有不合理状态的显著噪声影响。

我们有效地跟踪了整个运动轨迹,展示了控制器在处理意外噪声时的鲁棒性。

## 5 消融研究

机器人 tracking demonstration 的多样性和质量。
我们提出利用 tracking 控制器和 homotopy 生成器来增强 tracking demonstration 的多样性和质量。我们通过创建两个变体来消融这些策略:"Ours (w/o data, w/o homotopy)"(数据集通过无先验知识地优化每条轨迹构建)和 "Ours (w/o data)"(仅使用 homotopy optimization 方案改善 demonstration)。尽管使用相同数量的 demonstration,两个变体产生的数据质量较低。如表 1 所示,它们的性能不如我们的完整方法,强调了数据质量在训练控制器中的重要性。

*图 5: 扩展 demonstration 数量。*

扩展 demonstration 数量。
为研究 tracking 控制器性能与 demonstration 数量之间的关系,我们在训练过程中改变 demonstration 数据集的大小并在 TACO 数据集上测试性能。具体来说,在最终训练迭代中,我们将数据集下采样到原始大小的 0.1、0.3、0.5 和 0.9 并重新训练模型。如图 5 所示,demonstration 数量与模型性能之间存在明显的相关性。由于曲线尚未趋于平稳,我们推测增加高质量数据的数量可以进一步提升性能。

## 6 结论与局限性

我们提出 DexTrack 来开发灵巧操作的可泛化 tracking 控制器。利用高质量的 tracking demonstration 和逐轨迹 tracking 方案,我们通过自举优化控制器。大量实验验证了其有效性,为未来发展奠定了坚实基础。局限性:一个关键局限是获取高质量 demonstration 的过程非常耗时。未来工作可以探索更快速的、近似的 homotopy optimization 方法来加速训练。

## 参考文献

- Akkaya et al. (2019)

Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al.

Solving rubik's cube with a robot hand.

arXiv preprint arXiv:1910.07113*, 2019.

- Bai et al. (2023)

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenhang Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, K. Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Yu Bowen, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xing Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu.

Qwen technical report.

*ArXiv*, abs/2309.16609, 2023.

URL [https://api.semanticscholar.org/CorpusID:263134555](https://api.semanticscholar.org/CorpusID:263134555).

- Booher et al. (2024)

Jonathan Booher, Khashayar Rohanimanesh, Junhong Xu, Vladislav Isenbaev, Ashwin Balakrishna, Ishan Gupta, Wei Liu, and Aleksandr Petiushko.

Cimrl: Combining imitation and reinforcement learning for safe autonomous driving.

*ArXiv*, abs/2406.08878, 2024.

URL [https://api.semanticscholar.org/CorpusID:270440413](https://api.semanticscholar.org/CorpusID:270440413).

- Brown et al. (2020)

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Ma teusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

Language models are few-shot learners.

*ArXiv*, abs/2005.14165, 2020.

URL [https://api.semanticscholar.org/CorpusID:218971783](https://api.semanticscholar.org/CorpusID:218971783).

- Chen et al. (2021)

Tao Chen, Jie Xu, and Pulkit Agrawal.

A system for general in-hand object re-orientation.

*Conference on Robot Learning*, 2021.

- Chen et al. (2023)

Tao Chen, Megha Tippur, Siyang Wu, Vikash Kumar, Edward Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand reorientation of novel and complex object shapes.

*Science Robotics*, 8(84):eadc9244, 2023.

doi: 10.1126/scirobotics.adc9244.

URL [https://www.science.org/doi/abs/10.1126/scirobotics.adc9244](https://www.science.org/doi/abs/10.1126/scirobotics.adc9244).

- Chen et al. (2024)

Zerui Chen, Shizhe Chen, Cordelia Schmid, and Ivan Laptev.

Vividex: Learning vision-based dexterous manipulation from human videos.

*ArXiv*, abs/2404.15709, 2024.

URL [https://api.semanticscholar.org/CorpusID:269330215](https://api.semanticscholar.org/CorpusID:269330215).

- Chiang et al. (2024)

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica.

Chatbot arena: An open platform for evaluating llms by human preference.

*ArXiv*, abs/2403.04132, 2024.

URL [https://api.semanticscholar.org/CorpusID:268264163](https://api.semanticscholar.org/CorpusID:268264163).

- Christen et al. (2022)

Sammy Christen, Muhammed Kocabas, Emre Aksan, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 20577–20586, 2022.

- Gupta et al. (2016)

Abhishek Gupta, Clemens Eppner, Sergey Levine, and Pieter Abbeel.

Learning dexterous manipulation for a soft robotic hand from human demonstrations.

In *2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3786–3793. IEEE, 2016.

- Hester et al. (2017)

Todd Hester, Matej Vecerík, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John P. Agapiou, Joel Z. Leibo, and Audrunas Gruslys.

Deep q-learning from demonstrations.

In *AAAI Conference on Artificial Intelligence*, 2017.

URL [https://api.semanticscholar.org/CorpusID:10208474](https://api.semanticscholar.org/CorpusID:10208474).

- Hwangbo et al. (2018)

Jemin Hwangbo, Joonho Lee, and Marco Hutter.

Per-contact iteration method for solving contact dynamics.

*IEEE Robotics and Automation Letters*, 3(2):895–902, 2018.

- Jenelten et al. (2023)

Fabian Jenelten, Junzhe He, Farbod Farshidian, and Marco Hutter.

Dtc: Deep tracking control.

*Science Robotics*, 9, 2023.

URL [https://api.semanticscholar.org/CorpusID:263152143](https://api.semanticscholar.org/CorpusID:263152143).

- Jin (2024)

Wanxin Jin.

Complementarity-free multi-contact modeling and optimization for dexterous manipulation.

2024.

URL [https://api.semanticscholar.org/CorpusID:271874325](https://api.semanticscholar.org/CorpusID:271874325).

- Laskey et al. (2017)

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg.

Dart: Noise injection for robust imitation learning.

In *Conference on robot learning*, pp. 143–156. PMLR, 2017.

- Li et al. (2024)

Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, and Koushil Sreenath.

Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control.

*ArXiv*, abs/2401.16889, 2024.

URL [https://api.semanticscholar.org/CorpusID:267320454](https://api.semanticscholar.org/CorpusID:267320454).

- Liu et al. (2022)

Xingyu Liu, Deepak Pathak, and Kris M Kitani.

Herd: Continuous human-to-robot evolution for learning from human demonstration.

*arXiv preprint arXiv:2212.04359*, 2022.

- Liu et al. (2023)

Xuefeng Liu, Takuma Yoneda, Rick L. Stevens, Matthew R. Walter, and Yuxin Chen.

Blending imitation and reinforcement learning for robust policy improvement.

*ArXiv*, abs/2310.01737, 2023.

URL [https://api.semanticscholar.org/CorpusID:263609068](https://api.semanticscholar.org/CorpusID:263609068).

- Liu et al. (2024a)

Xueyi Liu, Kangbo Lyu, Jieqiong Zhang, Tao Du, and Li Yi.

Quasisim: Parameterized quasi-physical simulators for dexterous manipulations transfer.

*arXiv preprint arXiv:2404.07988*, 2024a.

- Liu et al. (2024b)

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi.

Taco: Benchmarking generalizable bimanual tool-action-object understanding.

*arXiv preprint arXiv:2401.08399*, 2024b.

- Luo et al. (2023a)

Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and Weipeng Xu.

Universal humanoid motion representations for physics-based control.

*ArXiv*, abs/2310.04582, 2023a.

URL [https://api.semanticscholar.org/CorpusID:263829555](https://api.semanticscholar.org/CorpusID:263829555).

- Luo et al. (2023b)

Zhengyi Luo, Jinkun Cao, Alexander W. Winkler, Kris Kitani, and Weipeng Xu.

Perpetual humanoid control for real-time simulated avatars.

In *International Conference on Computer Vision (ICCV)*, 2023b.

- Luo et al. (2024)

Zhengyi Luo, Jinkun Cao, Sammy Joe Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu.

Grasping diverse objects with simulated humanoids.

*ArXiv*, abs/2407.11385, 2024.

URL [https://api.semanticscholar.org/CorpusID:271217823](https://api.semanticscholar.org/CorpusID:271217823).

- Makoviichuk & Makoviychuk (2021)

Denys Makoviichuk and Viktor Makoviychuk.

rl-games: A high-performance framework for reinforcement learning.

[https://github.com/Denys88/rl_games](https://github.com/Denys88/rl_games), May 2021.

- Makoviychuk et al. (2021)

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al.

Isaac gym: High performance gpu-based physics simulation for robot learning.

*arXiv preprint arXiv:2108.10470*, 2021.

- Mordatch et al. (2012)

Igor Mordatch, Zoran Popovic, and Emanuel Todorov.

Contact-invariant optimization for hand manipulation.

In *Proceedings of the ACM SIGGRAPH/Eurographics symposium on computer animation*, pp. 137–144, 2012.

- OpenAI (2023)

OpenAI.

Gpt-4 technical report.

2023.

URL [https://api.semanticscholar.org/CorpusID:257532815](https://api.semanticscholar.org/CorpusID:257532815).

- Pang & Tedrake (2021)

Tao Pang and Russ Tedrake.

A convex quasistatic time-stepping scheme for rigid multibody systems with contact and friction.

In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 6614–6620. IEEE, 2021.

- Pang et al. (2023)

Tao Pang, HJ Terry Suh, Lujie Yang, and Russ Tedrake.

Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models.

*IEEE Transactions on Robotics*, 2023.

- Qin et al. (2022)

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang.

Dexmv: Imitation learning for dexterous manipulation from human videos.

In *European Conference on Computer Vision*, pp. 570–587. Springer, 2022.

- Rajeswaran et al. (2017)

Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.

Learning complex dexterous manipulation with deep reinforcement learning and demonstrations.

*arXiv preprint arXiv:1709.10087*, 2017.

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

- Singh et al. (2024)

Himanshu Gaurav Singh, Antonio Loquercio, Carmelo Sferrazza, Jane Wu, Haozhi Qi, Pieter Abbeel, and Jitendra Malik.

Hand-object interaction pretraining from videos.

2024.

URL [https://api.semanticscholar.org/CorpusID:272600324](https://api.semanticscholar.org/CorpusID:272600324).

- Sun et al. (2018)

Wen Sun, J. Andrew Bagnell, and Byron Boots.

Truncated horizon policy search: Combining reinforcement learning & imitation learning.

*ArXiv*, abs/1805.11240, 2018.

URL [https://api.semanticscholar.org/CorpusID:3533333](https://api.semanticscholar.org/CorpusID:3533333).

- Taheri et al. (2020)

Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas.

Grab: A dataset of whole-body human grasping of objects.

In *Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part IV 16*, pp. 581--600. Springer, 2020.

- Wang et al. (2023)

Yinhuai Wang, Jing Lin, Ailing Zeng, Zhengyi Luo, Jian Zhang, and Lei Zhang.

Physhoi: Physics-based imitation of dynamic human-object interaction.

*arXiv preprint arXiv:2312.04393*, 2023.

- Wei et al. (2022)

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou.

Chain of thought prompting elicits reasoning in large language models.

*ArXiv*, abs/2201.11903, 2022.

URL [https://api.semanticscholar.org/CorpusID:246411621](https://api.semanticscholar.org/CorpusID:246411621).

- Wen et al. (2023)

Bowen Wen, Wei Yang, Jan Kautz, and Stanley T. Birchfield.

Foundationpose: Unified 6d pose estimation and tracking of novel objects.

*2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 17868–17879, 2023.

URL [https://api.semanticscholar.org/CorpusID:266191252](https://api.semanticscholar.org/CorpusID:266191252).

- Wu et al. (2023)

Yueh-Hua Wu, Jiashun Wang, and Xiaolong Wang.

Learning generalizable dexterous manipulation from human grasp affordance.

In *Conference on Robot Learning*, pp. 618–629. PMLR, 2023.

- Xu et al. (2023)

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, et al.

Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

*arXiv preprint arXiv:2303.00938*, 2023.

- Zhang et al. (2023)

Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation.

*arXiv preprint arXiv:2309.03891*, 2023.

- Zhong et al. (2024)

Sheng Zhong, Thomas Power, Ashwin Gupta, and Peter Mitrano.

PyTorch Kinematics, February 2024.

概述。
附录提供了一系列材料以支撑主要论文。

- 补充技术说明(第 A 节)。我们给出补充说明以完善主要论文。

  - 详细方法概述图。我们提供了一幅方法概述图(图 6),比方法部分中的图更为详细地展示了方法。

  - 数据预处理(第 A.1 节)。我们介绍了用于从人类参考创建灵巧运动学机器人手操作数据集的运动学重定向策略的细节。

  - Tracking 控制器训练(第 A.2 节)。我们解释了基于 RL 的训练方案设计中的额外细节,包括 observation 空间和 reward。我们还解释了浮动基座灵巧手的控制策略。

  - Homotopy 生成器学习(第 A.3 节)。我们解释了 homotopy 生成器学习的细节。

  - 其他细节(第 A.4 节)。我们介绍了与技术相关的其他细节。

- 补充实验结果(第 B 节)。本节包含更多实验结果以支撑方法的有效性,包括:

  - 灵巧操作 Tracking 控制(第 B.1 节)。我们展示了方法的额外实验以及额外比较,包括使用不同训练设置获得的结果和额外的定性结果。我们还将讨论更多泛化能力评估实验。

  - 真实世界评估(第 B.2 节)。我们包含了更多真实世界评估结果。我们还将讨论真实世界评估中的失败案例。

  - Homotopy Optimization 方案分析(第 B.3 节)。我们展示了所提出的 homotopy optimization 方法获得的定性结果和比较,以证明 homotopy optimization 的能力以及 homotopy optimization path 生成器的有效性。此外,我们讨论了 homotopy path 生成器的泛化能力。

  - 失败案例(第 B.4 节)。我们讨论失败案例以全面评估和理解我们方法的能力。

- 实验细节(第 C 节)。我们详细说明了数据集、模型、训练和评估设置、仿真设置、真实世界评估设置,以及运行时间和复杂度分析。此外,我们尝试从计算角度衡量 tracking 控制器的一些关键特性,并呈现了关于这些特性的量化评估。

我们提供了[视频](https://youtu.be/zru1Z-DaiWE)和[网站](https://meowuu7.github.io/DexTrack/)来介绍我们的工作。网站和视频包含动画结果。我们强烈推荐探索这些资源,以直观理解挑战、我们模型的有效性以及相对于先前方法的优越性。

## 附录 A 补充技术说明

*图 6: [DexTrack](https://meowuu7.github.io/DexTrack/) 从人类参考中学习可泛化的神经 tracking 控制器用于灵巧操作。它在使用丰富且高质量的机器人 tracking demonstration 训练 tracking 控制器与通过 homotopy optimization 方案利用 tracking 控制器改进数据之间交替进行。*

方法详细概述图。在图 6 中,为了呈现方法的全面概述,我们绘制了一幅详细概述图,包含比第 3 节原始方法图更多的重要细节。

### A.1 数据预处理

运动学重定向。
我们通过从人类手部轨迹重定向机器人手操作序列来整理运动学机器人-物体交互数据。例如,给定一条人类手-物交互轨迹(描述以 MANO 表示的人手姿态序列和物体姿态序列 $(\mathbf{H}^{\text{human}}),\mathbf{O}$)以及关节机器人手的描述,我们将 $\mathbf{H}^{\text{human}}$ 重定向以获得机器人手轨迹 $\mathbf{H}$。我们手动定义机器人手网格与 MANO 手网格之间的对应关系。之后,优化机器人手自由度位置序列,使得生成的机器人手网格序列接近人手序列。形式上,设 $\mathbf{K}^{\text{human}}$ 和 $\mathbf{K}$ 分别表示人手关键点序列和机器人手关键点序列,优化目标为:

$$\text{minimize}\|\mathbf{K}-\mathbf{K}^{\text{human}}\|.$$ \tag{5}

我们使用 PyTorch_Kinematics (Zhong et al., 2024) 计算正向运动学。具体来说,给定时间步 $n$ 的机器人手逐关节自由度位置 $\mathbf{\theta}_{n}$,我们按如下方式计算 $\mathbf{h}_{n}$ 和 $\mathbf{k}_{n}$:

$$
\begin{aligned}
\mathbf{h}_{n} &=\text{Forward\_Kinematics}(\mathbf{\theta}_{n}), \tag{6}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{n} &=\text{KeyPoints}(\text{Forward\_Kinematics}(\mathbf{\theta}_{n})), \tag{7}
\end{aligned}
$$

其中 $\text{Forward\_Kinematics}(\cdot)$ 使用 PyTorch_Kinematics 提供的函数计算正向运动学,$\text{KeyPoints}(\cdot)$ 从转换后的关节网格中读取关键点。

我们使用二阶优化器(即 PyTorch 中实现的 L-BFGS)来求解优化问题 5。

### A.2 Tracking 控制器训练

浮动基座关节手的控制。关节手使用缩减坐标 $\mathbf{\theta}^{\text{finger}}$ 表示。我们额外添加三个平移关节和三个旋转关节来控制手的全局位置和朝向,得到 $\mathbf{\theta}=(\mathbf{\theta})^{\text{trans}},\mathbf{\theta}^{\text{rot}},\mathbf{\theta}^{\text{rot}})$。对于我们实验中使用的 Allegro 手和 LEAP 手,$\mathbf{\theta}^{\text{finger}}$ 是 16 维向量。因此,$\mathbf{\theta}$ 是 22 维向量。

Observation。
每个时间步 $n$ 的 observation 编码了当前手和物体状态、下一个目标状态、baseline 轨迹、动作和物体几何信息:

$$\mathbf{o}_{n}=\{\mathbf{s}_{n},\dot{\mathbf{s}}_{n},\hat{\mathbf{s}}_{n+1},\mathbf{s}^{b}_{n},\mathbf{a}_{n},\text{feat}_{\text{obj}},\text{aux}_{n}\}.$$ \tag{8}

其中 $\text{aux}_{n}$ 是辅助特征,计算如下:

$$\text{aux}_{n}=\{\hat{\mathbf{s}}_{n+1},\mathbf{f}_{n},\hat{\mathbf{s}}_{n+1}\ominus{\mathbf{s}}_{n},\},$$ \tag{9}

其中 $\hat{\mathbf{s}}_{n+1}\ominus{\mathbf{s}}_{n}$ 计算两个状态之间的差异(包括手部状态差异和物体状态差异),$\mathbf{f}_{n}$ 表示世界坐标系下的手指位置。

*表 3: 不同 reward 分量的权重。*

Reward。我们用于操作 tracking 的 reward 鼓励转移后的手部状态和物体状态接近其对应的参考状态以及手-物亲和性:

$$r=w_{o,p}r_{o,p}+w_{o,q}r_{o,q}+w_{\text{wrist}}r_{\text{wrist}}+w_{\text{finger}}r_{\text{finger}}+w_{\text{affinity}}r_{\text{affinity}},$$ \tag{10}

其中 $r_{o,p},r_{o,q},r_{\text{wrist}},r_{\text{finger}}$ 是跟踪物体位置、物体朝向、手腕、手指的 reward,$w_{o,p},w_{o,q},w_{\text{wrist}},w_{\text{finger}},w_{\text{affinity}}$ 是对应权重。
$r_{o,p},r_{o,q},r_{\text{wrist}},r_{\text{finger}}$ 计算如下:

$$
\begin{aligned}
r_{o,p} &=0.9-\|\mathbf{p}_{n}^{o}-\hat{\mathbf{p}}_{n}^{o}\|_{2}, \tag{11}
\end{aligned}
$$
$$
\begin{aligned}
r_{o,q} &=\text{np.pi}-\text{Diff\_Angle}(\mathbf{q}_{n}^{o}-\hat{\mathbf{q}}_{n}^{o})), \tag{12}
\end{aligned}
$$
$$
\begin{aligned}
r_{\text{wrist}} &=-(w_{\text{trans}}\|\mathbf{s}_{n}^{h}[:3]-\hat{\mathbf{s}}_{n}^{h}[:3]\|_{1}+w_{\text{ornt}}\|\mathbf{s}_{n}^{h}[3:6]-\hat{\mathbf{s}}_{n}^{h}[3:6]\|_{1} \tag{13}
\end{aligned}
$$
$$
$$
$$
\begin{aligned}
r_{\text{finger}} &=-w_{\text{finger}}\|\mathbf{s}_{n}^{h}[6:]-\hat{\mathbf{s}}_{n}^{h}[6:]\|_{1} \tag{14}
\end{aligned}
$$

其中 $\mathbf{p}^{o}_{n}$ 和 $\mathbf{q}_{n}^{o}$ 分别表示当前物体的位置和朝向(以四元数表示),$\mathbf{s}_{n}^{h}$ 表示当前手部状态。除这些 reward 外,如果物体被精确跟踪(即旋转误差保持在 $5$ 度以内且平移误差保持在 $5$ 厘米以内),我们会额外添加一个值为 $1$ 的 bonus reward。

表 3 总结了我们实验中使用的不同 reward 分量的权重。

物体特征预处理。我们在考虑的两个数据集(即 GRAB 和 TACO)中所有物体上训练基于 PointNet 的自编码器。之后,我们使用每个物体的潜在嵌入作为其潜在特征,输入到 tracking 控制器的 observation 中。在我们的实验中,物体特征维度为 256。

### A.3 Homotopy 生成器学习

挖掘有效的 homotopy optimization path。在我们的方法中,最大迭代次数 $K$ 设为 $3$,以在时间开销和有效性之间取得平衡。

我们需要为每个 tracking 任务识别邻居,以避免遍历所有任务并降低时间开销。我们使用跨运动学轨迹相似性来筛选邻近任务。我们为每个 tracking 任务预选 $K_{\text{nei}}=$10 个邻近任务。

### A.4 其他说明

在 reward 设计中,我们没有包含与速度相关的项,因为无法从运动学参考中获得精确的速度。可以想象通过计算相邻两帧之间的有限差分来获得速度,但这可能不够准确。因此,我们不使用它们以避免不必要的噪声。

## 附录 B 补充实验

### B.1 灵巧操作 Tracking 控制

*表 4: 定量评估与比较。加粗红色数字表示最佳值。模型在 GRAB 和 TACO 数据集的训练 tracking 任务上进行训练。*

在两个数据集上训练 tracking 控制器。在主要实验中,训练数据和测试数据来自同一数据集。考虑到跨数据集轨迹差异较大,我们采用了这种设置。具体来说,GRAB 主要包含日常物体的操作轨迹,而 TACO 主要涵盖功能性工具使用轨迹。然而,联合使用两个数据集的轨迹训练模型可能凭借增加的标注数据覆盖范围提供更强的控制器。因此,我们额外进行了这个实验:使用两个数据集提供的轨迹训练单一模型,并分别在各自的测试集上测试性能。结果总结在表 4 中,仍然证明了我们方法的有效性。

与标注整个数据集训练控制器的比较。在我们的方法中,我们仅以渐进方式尝试标注一小部分数据,利用 tracking 控制器提供的 tracking prior 和 homotopy 生成器提出的 homotopy path 的能力,从而在可承受的时间预算内获得高质量 demonstration。人们可能会好奇是否有可能标注所有训练数据集轨迹并使用这些 demonstration 训练 tracking 控制器。如果可能,通过这种方法训练的模型性能如何?因此我们设法通过在两台机器上的 16 块 GPU 上并行运行逐轨迹 tracking 实验来标注 GRAB 数据集训练集中的每条轨迹。

我们在一周内完成了优化。

之后,我们使用生成的 demonstration 来训练轨迹控制器。以这种方式训练的最终模型在两个阈值下分别达到 42.13% 和 60.41% 的成功率。性能仍然无法达到我们仅为部分数据优化高质量 demonstration 的原始方法(详见表 1)。我们认为是两个标注数据集之间的质量差异导致了这种差距。这进一步验证了我们的假设:标注数据集的数量和质量对训练良好的 tracking 控制器都很重要。

*表 5: 在 TACO 数据集上的泛化能力评估。*

*图 7: 对分布外物体和操作的鲁棒性。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

在 TACO 数据集上的进一步泛化能力评估。
我们进一步在 TACO 数据集的各种测试集上评估模型的泛化能力。如表 5 所示,控制器在类别级泛化设置(S1,物体类别已知但操作轨迹和物体几何形状新颖)中表现良好。S2(涉及新的交互三元组)上的表现令人满意,展示了控制器处理新操作序列的能力。然而,S3 的结果揭示了在处理新物体类别和未见交互三元组时面临的挑战。例如,从铲子和勺子的交互泛化到使用碗盛放物品尤其困难。如图 7 所示,尽管面对不熟悉的物体和交互,我们成功地提起了刀并模仿了运动,虽然执行并不完美,但突显了在具有挑战性场景中的改进空间和适应性。

额外结果。我们在图 8 中展示了额外的定性结果,以进一步证明我们方法的能力。

*图 8: 额外定性比较。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

*表 6: 使用不同数量 demonstration 数据训练的模型性能比较。*

扩展 demonstration 数据量。
在表 6 中,我们展示了关于 demonstration 数据量对模型性能影响的消融研究中,每个模型在所有五类指标上的完整评估结果(详见第 5 节)。

### B.2 真实世界评估

*表 7: 真实世界定量比较(GRAB 数据集)。加粗红色数字表示最佳值。*

*表 8: 真实世界定量比较(TACO 数据集)。加粗红色数字表示最佳值。*

*图 9: 额外真实世界定性结果。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

成功阈值。我们定义了三个级别的成功率。第一级成功定义为到达物体、找到良好的抓握姿态,并展示出将物体抬起的潜在运动(即物体的一侧成功从桌面抬起)。第二级成功定义为找到一种方法设法将整个物体从桌面抬起。第三级成功定义为抬起物体后,继续跟踪物体轨迹超过 100 个时间步。

更多结果。对于直接 tracking 结果迁移设置,我们在表 7(GRAB 数据集)和表 8(TACO 数据集)中展示了我们方法与表现最佳 baseline 的定量成功率评估。

如表中所示,我们方法获得的 tracking 结果可以很好地迁移到真实机器人,帮助我们取得了明显优于 baseline 方法的结果。这验证了我们 tracking 结果的真实世界适用性。

关于迁移控制器之间的定量比较,请参考正文(第 4 节)。

我们在图 9 中包含了更多定性结果,以展示我们方法的真实世界应用价值。

*图 10: 真实世界实验中的失败案例。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)以获取动画结果。*

失败案例。一种典型的失败模式是当尝试手内操作时,随着接触变化,物体倾向于从手中掉落,如图 10 所示。

### B.3 Homotopy Optimization 方案分析

*图 11: Homotopy optimization 方案的有效性。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

我们对所提出的 homotopy optimization 方案和 homotopy path 生成器进行了进一步分析以证明其有效性。如图 11 所示,通过沿 homotopy optimization path 优化,我们可以在逐轨迹 tracking 中获得更好的结果。

提起薄物体。
如图 11a 所示,对于原本无法解决的 tracking 问题(即需要设法将一支非常薄的长笛从桌面抬起),我们最终可以通过逐步求解生成器提出的 homotopy optimization path 中的每个 tracking 问题来降低 tracking 难度。

抓取小物体。
如图 11b 所示,原始的逐轨迹 tracker 无法找到合适的方式来抓取小球并将其从桌面抬起。然而,在 homotopy optimization 的支持下,我们最终找到了将其从桌面抬起的方法。

提起圆苹果。图 11c 展示了一条有效的 homotopy optimization path,使我们能够将苹果从桌面抬起,而此前由于圆滑表面,这对 policy 构成了挑战。

*表 9: Homotopy path 生成器的泛化实验。*

Homotopy path 生成器的泛化实验。
为进一步了解 homotopy 生成器的泛化能力,我们进行了以下测试:

- (a) 使用从 GRAB 训练集挖掘的 homotopy path 训练 path 生成器,并在第一个测试集上评估,该测试集包含从 GRAB 训练集中 homotopy 生成器未观察到的剩余 tracking 任务中均匀随机选取的 50 个 tracking 任务。

- (b) 在第二个测试集上评估(a)中训练的 path 生成器,该测试集包含从 GRAB 测试集的测试 tracking 任务中均匀随机选取的 50 个 tracking 任务。

- (c) 在从 TACO 第一级测试集的测试 tracking 任务中均匀随机选取的 50 个 tracking 任务上评估(a)中训练的 path 生成器。

- (d) 使用从 GRAB 和 TACO 训练集中挖掘的 homotopy path 训练 path 生成器,并在(c)中使用的测试集上评估。

对于每个 tracking 任务,如果通过优化 optimization path 获得的 tracking 结果优于基于 RL 逐轨迹 tracking 产生的原始 tracking 结果,我们认为生成的 homotopy optimization path 是有效的。否则,我们认为其无效。我们在表 9 中总结了有效 homotopy optimization path 的比例。

总结如下:

- (a) homotopy path 生成器在分布内测试设置中可以表现得相对良好;

- (b) 当操作模式略有偏移时,性能会轻微下降(请参考第 4.1 节了解 GRAB 训练集和测试集之间的差异);

- (c) path 生成器在泛化到相对分布外的 tracking 任务(涉及全新物体和相当新颖的操作模式)时会遇到困难;

- (d) 增加 homotopy path 生成器的训练数据覆盖范围会使其表现明显更好。

*图 12: 失败案例。请参考[我们的网站](https://meowuu7.github.io/DexTrack/)和[配套视频](https://youtu.be/zru1Z-DaiWE)以获取动画结果。*

### B.4 失败案例

在某些情况下,当物体来自全新类别且具有具有挑战性的薄几何形状时,我们的方法可能无法表现良好,如图 12 所示。

## 附录 C 补充实验细节

*图 13: 来自已见物体类别的新物体示例(TACO)。*

*图 14: 来自新物体类别的物体示例(TACO)。*

数据集。我们的灵巧机器人手-物操作数据集通过重定向两个公开的人-物数据集创建:GRAB (Taheri et al., 2020)(包含单手与日常物体的交互)和 TACO (Liu et al., 2024b)(以功能性工具使用交互为特色)。我们重定向了完整的 GRAB 数据集和完全发布的 TACO 数据集,分别获得 1269 和 2316 条机器人手操作序列。

对于 GRAB,我们使用受试者 s1 的序列(共 197 条)作为测试数据集。训练数据集由其他受试者的剩余序列构成。

对于 TACO 数据集,我们创建了一个训练集和四个不同泛化级别的测试集,以详细评估模型的泛化性能。

具体来说,整个数据集被分为:1) 训练数据集(包含 1565 条轨迹),2) 测试集 S0(工具物体几何形状和交互三元组在训练中均已见,共 207 条轨迹),3) 测试集 S1(工具几何形状新颖但交互三元组在训练中已见,共 139 条轨迹),4) 测试集 S2(交互三元组新颖但物体类别和几何形状已见,共 120 条轨迹),以及 5) 测试集 S3(物体类别和交互三元组对训练数据集均为新,共 285 条轨迹)。

图 13 和图 14 分别绘制了已见类别中未见物体和新类别物体的示例。

TACO 呈现的原始数据通常包含噪声的初始帧,其中手穿过桌面或物体。这种噪声虽然看似微妙,但会影响初始动力学。例如,如果手最初穿过桌面,在开始时会对手施加很大的力,这会严重影响后续步骤的仿真。此外,如果手最初穿过物体,物体在仿真开始时会被弹飞。为消除这些现象,我们对原始序列做了小修改。具体来说,我们将 GRAB 数据集中受试者 s2 的手机传递序列与这些 TACO 序列进行插值作为最终修改后的序列。具体来说,我们取 GRAB 序列前 60 帧的手部姿态。然后线性插值 GRAB 序列第 60 帧的手部姿态与 TACO 序列第 60 帧的手部姿态。详情请参考补充材料中的代码(参见 "README.md" 获取说明)。

*表 10: 总训练时间消耗(TACO 数据集)。*

训练和评估设置。
对于 GRAB 和 TACO,在第一阶段,我们首先从训练数据集中采样 100 条轨迹。我们训练其逐轨迹 tracker 以获取动作标注数据来构建第一版标注数据集。之后,训练第一个 tracking 控制器并在所有轨迹上进行评估。然后,我们使用与 tracking 物体位置误差成正比的权重从剩余轨迹中额外采样 100 条。这些采样的轨迹与第一阶段采样的轨迹共同构成第二版待标注数据集。我们同时利用逐轨迹 tracker 优化和来自第一版训练好的 tracking 控制器的 tracking prior 来标注数据,旨在获得高质量的标注轨迹。之后,我们从这 200 条轨迹中搜索 tracking curriculum(跟踪课程),并使用挖掘到的 curriculum 训练 tracking curriculum 调度器。然后使用最终最佳优化的轨迹构建第二版动作标注数据集。然后重新训练 tracking 控制器并在每条轨迹上评估其性能。在第三阶段,我们从剩余未选择的轨迹中额外采样 200 条。然后利用逐轨迹 tracking 优化、来自 tracking 控制器的 tracking prior 和 curriculum 调度器的 curriculum 共同标注它们。之后,使用最佳优化的标注轨迹构建第三版。然后使用这一版标注数据集重新训练 tracking 控制器。训练 tracking 控制器时,我们在 reward 中设置了一个阈值(即 50)。只有 reward 高于该阈值的轨迹才被用于提供监督。

仿真和 policy 均以 60Hz 运行。仿真中忽略手的重力。

详细设置请参考补充材料中的代码。

所有模型在 Ubuntu 20.04.6 LTS 上使用八块 NVIDIA A10 显卡和 CUDA 12.5 版本训练。所有模型在单卡上训练,不使用多 GPU 并行化。

*图 15: 真实世界实验设置。*

真实世界实验设置。
我们使用 Franka 机械臂和 LEAP 手进行真实世界评估(图 15)。迁移基于状态的 policy 时,使用 FoundationPose (Wen et al., 2023) 估计物体位姿。使用有限差分估计手关节速度以及物体线速度和角速度。考虑到仿真器中使用的控制策略与 Franka 机械臂和 LEAP 手控制之间的差距,我们设计了一种策略来缓解这两种控制方法之间的差异。具体来说,我们不直接将控制信号应用于 LEAP 手和 Franka 机械臂,而是建立一个具有与训练期间仿真设置相同的物理和控制相关参数的仿真器。然后,在每个时间步,我们首先将控制指令应用于仿真的 LEAP 手。然后进行仿真。之后,我们从仿真器中读出仿真状态。然后使用从仿真器获得的当前状态和真实 LEAP 手的当前状态计算应施加于真实 LEAP 手的位置目标信号。计算出的位置目标信号直接输入真实 LEAP 手控制器。根据我们的观察,一旦真实手接收到指令,它几乎可以精确到达指令中的相同位置。因此,在实践中,我们直接使用从仿真器获得的状态作为输入真实 LEAP 控制器的位置目标指令。实验证明了这种控制策略的有效性。

时间消耗和时间复杂度。表 10 总结了不同方法在 TACO 数据集上的总时间消耗。直接训练不带任何监督的 PPO 是最高效的方法,但由于缺乏适当引导,性能落后。解决逐轨迹 tracking 问题以为训练通用 tracking 控制器提供高质量数据会额外增加时间消耗,因为需要优化逐轨迹 tracker。由于我们仅从整个训练数据集中选择子集,时间消耗仍然可承受。通过挖掘 tracking curriculum 来改进逐轨迹 tracker 会引入额外时间开销。由于我们考虑用于学习 curriculum 调度器的轨迹数量仍控制在相对较小的范围内,最终时间开销仍然相对可承受。实验在配备八块 A10 GPU 的 Ubuntu 20.04 机器上进行。对于逐轨迹 tracker 优化,我们一次并行训练八个 tracker。

整体训练过程的时间复杂度为 $\mathcal{O}(|\mathcal{S}|+KK_{\text{nei}}|\mathcal{S}|)$。$\mathcal{S}$ 表示训练数据集。

*表 11: 泛化得分(GRAB 数据集)。加粗红色数字表示最佳值。*

*表 12: 鲁棒性得分(GRAB 数据集)。加粗红色数字表示最佳值。*

*表 13: 适应性得分(GRAB 数据集)。加粗红色数字表示最佳值。*

"Robustness"(鲁棒性)、"Generalization Ability"(泛化能力)和 "Adaptivity"(适应性)。
我们尝试对神经 tracking 控制器的一些关键特征给出计算定义。请注意,据我们所知,这些概念没有标准的正式计算定义。我们在此呈现的定义和量化仅来自我们的视角。

- 为量化 "generalization ability"(泛化能力),我们首先需要量化:1) 两个轨迹分布之间的分布差距,从而定义泛化的级别;2) 通过模型性能差距衡量从训练分布到测试分布的泛化能力:

记 $\mathcal{E}$ 为训练集的 tracking 任务分布,$\mathcal{D}$ 为测试分布,定义其差距如下:

$$d(\mathcal{D};\mathcal{E})=\mathbb{E}_{\mathbf{T}\sim\mathcal{D}}\left[\min_{% \mathbf{T_{train}\sim\mathcal{E}}}\left(\text{Tracking\_Task\_Diff}(\mathbf{T}% ,\mathbf{T_{train}})\right)\right],$$ \tag{15}

其中 $\text{Tracking\_Task\_Diff}(\cdot,\cdot)$ 衡量两个操作轨迹 tracking 问题之间的差异。

对于由运动学手状态序列 $\{\mathbf{s}_{n}^{h}\}_{n=0}^{N}$、运动学物体位姿序列 $\{\mathbf{p}^{o}_{n},\mathbf{q}^{o}_{n}\}_{n=0}^{N}$ 和物体几何形状(例如用点云表示)描述的轨迹 tracking 任务,我们计算两个 tracking 任务(即 $\mathbf{T}_{A}=\{\{\mathbf{s}^{h,A}_{n}\},\{\mathbf{p}^{o,A}_{n},\mathbf{q}^{o,A}_{n}\},\text{PC}^{A}\}$ 和 $\mathbf{T}_{B}=\{\{\mathbf{s}^{h,B}_{n}\},\{\mathbf{p}^{o,G}_{n},\mathbf{q}^{o,G}_{n}\},\text{PC}^{B}\}$)之间的轨迹 tracking 任务差异,作为手部轨迹差异、物体位姿序列差异和物体几何形状差异的加权和:

$$
\begin{aligned}
\text{Tracking\_Task\_Diff}(\mathbf{T}_{A},\mathbf{T}_{B})=\frac{% 1}{N+1}\sum_{n=0}^{N} &(w^{h}_{diff}\|\mathbf{s}_{n}^{h,A}-\mathbf{s}_{n}^{h,B}\|+w_{diff}^{o,p}\|\mathbf{p}^{o,A}_{n}-\mathbf{p}^{o,B}_{n}\| \tag{16} \\
&+w_{diff}^{o,q}\|\mathbf{q}^{o,A}_{n}-\mathbf{q}^{o,B}_{n}\|) \tag{17} \\
&+w_{diff}^{pc}\text{Chamfer-Distance}(\text{PC}^{A},\text{PC}^{B}), \tag{18}
\end{aligned}
$$

其中 $w_{diff}^{h}=0.1,w_{diff}^{o,p}=1,w_{diff}^{o,q}=0.3,w_{diff}^{pc}=0.5$。

对于 tracking policy $\pi$,通过 tracking 误差的期望定义其在分布 $\mathcal{E}$ 中 tracking 任务上的性能:

$$L_{\mathcal{E}}(\pi)=\mathbb{E}_{\mathbf{T}\sim\mathcal{E}}\left[\text{% Tracking\_Error}_{\pi}(\mathbf{T})\right],$$ \tag{19}

其中 $\text{Tracking\_Error}_{\pi}(\cdot)$ 评估 tracking policy $\pi$ 在轨迹 tracking 问题上的 tracking 误差。它是机器人手的 tracking 结果与参考轨迹之差以及物体之差的加权和:

$$\text{Tracking\_Error}_{\pi}(\mathbf{T})=w_{err}^{o,p}T_{err}+w_{err}^{o,q}R_{% err}+w_{err}^{h,wrist}E_{wrist}+w_{err}^{h,finger}E_{finger},$$ \tag{20}

其中 $w_{err}^{h,wrist}=0.1,w_{err}^{h,finger}=0.1,w_{err}^{o,p}=1.0,w_{err}^{o,q}=0.3$。

因此,在训练分布 $\mathcal{E}$ 上训练的 policy $\pi$ 到测试分布 $\mathcal{D}$ 的泛化能力衡量为:

$$s_{g}=\frac{d(\mathcal{D};\mathcal{E})}{\min(L_{\mathcal{D}}(\pi),\epsilon)},$$ \tag{21}

其中 $\epsilon$ 是避免数值问题的小值。得分 $s_{g}$ 随训练-测试分布差距增大或测试分布上 tracking 误差减小而增大。

使用上述关于 tracking 任务分布差距和泛化能力得分的量化方法,我们在表 11 中总结了不同模型在 GRAB 数据集上的泛化能力得分。

- 为量化 "robustness"(鲁棒性),由于使用神经控制器分析动态函数相当困难,我们通过 tracking policy 在具有相对高质量运动学轨迹的 tracking 任务和具有受干扰运动学参考的 tracking 任务之间的性能差异来衡量鲁棒性。

为衡量运动学操作轨迹的 "quality"(质量),我们引入三类量:

  - Smoothness(平滑度):计算状态有限差分之间的差异:

$$t_{s}(\mathbf{T}):=\frac{1}{N-1}\sum_{i=1}^{N-1}\frac{1}{\Delta t}\left(\frac{% \mathbf{w_{s}}\cdot(\mathbf{s}_{i+1}-\mathbf{s}_{i})}{\Delta t}-\frac{\mathbf{% w}_{s}\cdot(\mathbf{s}_{i}-\mathbf{s}_{i-1})}{\Delta t}\right),$$ \tag{22}

其中 $\mathbf{w}_{s}$ 是每个状态自由度的权重向量,$\Delta t$ 是相邻两帧之间的时间。

  - Consistency(一致性):计算手-物运动一致性:

$$t_{c}(\mathbf{T}):=\frac{1}{N-1}\sum_{i=1}^{N}\left\|\frac{\mathbf{p}_{i}^{o}-% \mathbf{p}_{i-1}^{o}}{\Delta t}-\frac{\mathbf{p}_{i}^{h}-\mathbf{p}_{i-1}^{h}}% {\Delta t}\right\|,$$ \tag{23}

其中 $\mathbf{p}_{i}^{o}$ 是第 $i$ 帧的物体位置,$\mathbf{p}_{i}^{h}$ 是第 $i$ 帧的手腕位置。

  - Penetrations(穿透):计算所有帧中手与物体之间的穿透:

$$t_{p}(\mathbf{T}):=\frac{1}{N+1}\sum_{i=0}^{N}\text{{Pene\_Depth}}(\mathbf{s}_% {i},\mathbf{P}^{o}),$$ \tag{24}

其中 $\mathbf{P}^{o}$ 表示物体点云,Pene_Depth 计算手与物体之间的最大穿透深度。

运动学操作参考的 "Quality"(质量)可以使用上述三种度量来衡量。结合上述量,定义测试分布 $\mathcal{L}$ 的整体 "quality" 为:

$$s_{quality}(\mathcal{L})=\mathbb{E}_{\mathbf{T}\sim\mathbf{L}}\left[\frac{t_{s% }(\mathbf{T})+t_{c}(\mathbf{T})+t_{p}(\mathbf{T})}{3}\right].$$ \tag{25}

"Robustness"(鲁棒性)可以通过模型在 "high-quality"(高质量)参考 tracking 任务和 "perturbed"(受干扰)因而 "low-quality"(低质量)tracking 任务上的性能差距来衡量。记 "高质量" tracking 任务分布为 $\mathcal{H}$,"低质量" 的为 $\mathcal{L}$。因此可以将 "robustness" 量化为:

$$s_{r}(\pi):=\frac{s_{quality}(\mathcal{L})}{\min(L_{\mathcal{L}}(\pi),\epsilon% )}.$$ \tag{26}

随着轨迹分布质量变差和 tracking 误差降低,"robustness score"(鲁棒性得分)会增大。

为评估这一点,我们通过向手部轨迹和物体位置轨迹添加随机噪声来构建 GRAB 数据集测试轨迹的受干扰测试集。之后,我们测试了我们方法和表现最佳 baseline(PPO (w/o sup., tracking rew.))的性能。

我们在表 12 中总结了结果。

- 我们主要关注真实世界适应性来评估 tracking 控制器的 "adaptivity"(适应性)。由于很难定量衡量真实世界动力学与仿真器中动力学之间的差异,我们使用模型的性能差距作为指标来直接评估适应性。我们使用真实世界逐轨迹平均成功率来衡量 "adaptivity"。对于 GRAB 数据集的真实世界测试轨迹,我们在表 14 中总结了结果。

*表 14: 泛化得分(GRAB 数据集)。加粗红色数字表示最佳值。*

*表 15: 轨迹难度统计。*

"hard-to-track"(难以跟踪)的计算解释。
最直接的量化方法是定义一个与 tracking 方法性能相关的分数。对于 tracking 方法 $\mathcal{M}$ 和轨迹 $\mathbf{T}$,定义 $\pi$ 为我们能为轨迹 $\mathbf{T}$ 优化的最佳 tracking policy(即 $\pi=\mathcal{M}(\mathbf{T})$)。给定 tracking 误差 $\text{Tracking\_Error}_{\pi}(\mathbf{T})$,"hard-to-track" 分数可定义为:$s_{ht}=\frac{1}{\min(\text{Tracking\_Error}_{\pi}(\mathbf{T}),\epsilon)}$,其中 $\epsilon$ 是避免数值问题的小值。

此外,我们可以使用手和物体运动学轨迹的一些统计量来量化 "hard-to-track" 特征。这里我们引入三类统计量:1) 物体运动平滑度 $s^{o}_{smooth}$:通过计算逐帧平均物体加速度来量化运动平滑度,即 $s_{smooth}^{o}=\frac{1}{N-1}\sum_{i=1}^{N-1}\|\frac{1}{\Delta t}(\frac{\mathbf{p}^{o}_{i+1}-\mathbf{p}^{o}_{i}}{\Delta t}-\frac{\mathbf{p}_{i}^{o}-\mathbf{p}_{i-1}^{o}}{\Delta t})\|$;2) 手-物接触变化速度 $v_{contact}$:量化接触图逐帧变化速度,即 $v_{contact}=\frac{1}{N}\sum_{i=1}^{N}\|\frac{\mathbf{c}_{i}-\mathbf{c}_{i-1}}{N_{p}\Delta t}\|$,其中 $\mathbf{c}_{i}\in\{0,1\}^{N_{p}}$ 是编码手表面采样点与物体之间接触标志的二值接触图,$N_{p}$ 是从手采样的点数;3) 物体形状分数 $s_{shape}$:为物体包围盒 z 轴范围的倒数以量化物体形状:$s_{shape}=\frac{1}{\min(\text{extent}_{z},\epsilon)}$,其中 $\text{extent}\in\mathbb{R}^{3}$ 是物体包围盒的范围。我们可以联合使用这三类分数来量化 "hard-to-track" 特征。随着 $s_{smooth}^{o}$ 增大、$v_{contact}$ 增大和 $s_{shape}$ 增大,轨迹会变得更 "difficult"(困难),因而更 "hard" 去训练 policy 来跟踪。

如表 15 所示,GRAB 中的测试轨迹在轨迹平滑度方面比 TACO 更 "difficult",而 TACO 中的测试轨迹在接触变化速度和形状难度方面比 GRAB 更 difficult。

补充细节。对于 tracking 误差指标,我们报告测试集中逐轨迹结果的中位值,考虑到平均值可能受到异常值的影响。

Generated on Thu Feb 13 12:52:34 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
