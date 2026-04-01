# DeepMimic -- 学习笔记
> 一句话: 用 motion capture clip 做 reward shaping + PPO，让物理仿真角色模仿参考动作，奠定了 motion tracking RL 的标准范式
> 论文: Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne, ACM SIGGRAPH 2018

## 这篇论文解决了什么问题
2018 年之前，deep RL 训练出的角色动作质量很差 -- 奇怪的步态、多余的上肢运动、不自然的姿态。手工设计 reward (如前进速度 + 力矩惩罚) 无法表达 "自然运动" 的概念。同时，传统 kinematic 方法依赖大量数据却无法应对物理扰动。DeepMimic 提出: 直接用 motion clip 定义 "什么是好动作"，让 RL 在物理仿真中学会复现这些动作，同时保留对扰动的鲁棒性和完成任务的灵活性。

## 核心想法 (用直觉解释)
把 "模仿参考动作" 当成 reward signal 喂给 RL。policy 的输出是 PD controller 的目标关节角度，reward 由两部分加权组成: imitation reward (和参考动作像不像) + task reward (完成目标好不好)。训练用 PPO。关键直觉: 参考动作不是硬约束，而是 soft guidance -- 角色可以偏离参考去完成任务或恢复平衡，但偏离越大 reward 越低。

## 关键设计决策
1. **Imitation reward 分解**: r_I = w_p * r_pose + w_v * r_vel + w_ee * r_endeffector + w_rp * r_rootpose。分别奖励关节朝向、关节角速度、末端执行器位置、重心位置的匹配度。每一项用 exp(-k * error) 映射到 [0,1]，然后加权乘积。这种 multiplicative 结构意味着任何一项差都会拉低总 reward。
2. **Reference State Initialization (RSI)**: 每个 episode 从参考动作的随机时间点初始化，而非总从开头开始。直觉: 学后空翻必须先学会落地，但如果总从起跳开始，policy 在学会落地前根本到不了落地阶段。RSI 让 agent 直接 "体验" 动作中后段的状态，大幅加速学习。这是论文最关键的 trick 之一。
3. **Early Termination (ET)**: 当角色的躯干/头部触地时立即终止 episode。效果类似于处理 class imbalance -- 防止 "在地上挣扎" 的无用样本淹没训练数据。
4. **Multi-clip 整合**: 三种方式 -- (a) max-over-clips reward: 自动选最匹配的 clip; (b) skill selector: 用 one-hot 输入让用户选择执行哪个技能; (c) value function 拼接: 独立训练多个 single-clip policy，用 value function 判断何时切换。
5. **Phase variable**: policy 接收一个 phase 变量 (当前时间在参考动作中的位置)，让 policy 知道 "该做到哪一步了"。

## 这篇论文之后发生了什么
- **AMP (2021)**: 用 adversarial discriminator 替代手工 imitation reward，不需要 phase alignment
- **PHC (2023)**: 加入 primitive-based hierarchical control，处理 motion capture 数据中的物理不可行片段
- **BeyondMimic (2025)**: 在 DeepMimic 的 motion tracking 基础上叠加 latent diffusion model，实现 test-time task adaptation
- **ASAP (2025)**: 在 DeepMimic-style tracking 之上加 delta action model 解决 sim2real gap
- RSI + ET 已成为几乎所有 motion tracking 工作的标配

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **数据作为 reward 比数据作为约束更灵活** -- motion clip 定义 reward 而非 trajectory to track，policy 可以自由偏离 | 灵巧手操作中，demo 也应作为 soft reward signal 而非 hard tracking target |
| 2 | **RSI 是 RL 探索的核心加速器** -- 从目标状态附近初始化等价于 curriculum + hindsight | 任何长 horizon 任务都应考虑 goal-state initialization |
| 3 | **Phase variable = 最简单的 temporal conditioning** -- 一个标量就能告诉 policy "你在动作序列的哪里" | 后续 diffusion policy / flow policy 的 timestep embedding 是同一思想的泛化 |
| 4 | **Multiplicative reward 比 additive 更严格** -- 一项为零则总 reward 为零，强制 policy 不能偏科 | 多目标 reward 设计时要考虑乘法 vs 加法的 trade-off |
| 5 | **这篇论文定义了 humanoid motion tracking 的 "标准协议"** -- RSI, ET, PD target, phase conditioning -- 所有后续工作都在此基础上修改 | 理解 DeepMimic 等于理解整个 motion tracking 流水线的 baseline |
