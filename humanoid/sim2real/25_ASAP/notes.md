# ASAP -- 学习笔记
> 一句话: 用 real-world rollout 数据训练 delta action model 补偿 sim2real dynamics gap，然后在 "修正后的 simulator" 中 fine-tune policy，实现 agile humanoid whole-body motion 的高保真迁移
> 论文: Tairan He et al. (CMU LeCAR Lab + NVIDIA), 2025

## 这篇论文解决了什么问题
Sim2real 的核心矛盾: simulation 和现实的动力学不一致。三种传统方案都有问题:
- **SysID (System Identification)**: 直接估物理参数 (电机响应、连杆质量等)。但参数空间不完整，无法覆盖所有 mismatch 来源; 而且很多平台没有 ground truth torque 测量。
- **Domain Randomization**: 随机化仿真参数让 policy 鲁棒。但 agile motion 对参数敏感 -- 太大的随机化会让 policy 变 conservative，牺牲 agility。
- **Learned dynamics**: 直接学一个 real-world dynamics model。但 humanoid 的高维状态空间使这非常困难。

ASAP 的方案: 不去学完整的 real dynamics，而是学一个 "delta"，即仿真和现实之间的差异补偿项。

## 核心想法 (用直觉解释)
**两阶段 pipeline**:
1. **Pre-training**: 在 IsaacGym 中用 DeepMimic-style motion tracking 训练 policy (PPO + phase variable + RSI + tracking reward)。部署到真实机器人上，虽然能动但质量不够好。
2. **Post-training**: 把 pre-trained policy 部署到真机上收集轨迹数据 (state + action)。然后在仿真中 "回放" 这些动作 -- 因为 dynamics mismatch，仿真中回放会偏离真实轨迹。训练一个 delta action model: delta_a = pi_delta(s, a)，使得 f_sim(s, a + delta_a) 的结果逼近真实的 s'。最后把这个 delta action model 冻结嵌入仿真器，形成一个 "更接近现实的仿真器"，在里面 fine-tune policy。

直觉: 就像给仿真器 "戴上一副眼镜"，让它看到的世界更接近现实。比如真实电机比仿真弱，delta model 就学会 "减小下肢动作幅度" 来模拟这种弱化。

## 关键设计决策
1. **Delta action (不是 delta dynamics)**: 之前的方法学 delta state: s' = f_sim(s, a) + delta_s。ASAP 学 delta action: s' = f_sim(s, a + delta_a)。区别很关键 -- delta action 是在仿真器的输入端修正，保持了仿真器内部的物理一致性 (contact、constraint 等依然由仿真器处理); delta dynamics 在输出端修正，可能产生物理上不合理的状态。
2. **Delta model 用 RL 训练 (不是 supervised learning)**: 每一步从真实状态 s_r 初始化，用 PPO 训练 delta action model，reward 是最小化 sim state 和 real state 的差距 + action 正则化。RL 比 supervised 更适合处理 sequential decision 和 multi-step rollout。
3. **Phase-based tracking + asymmetric actor-critic**: actor 只用 proprioception + phase variable (不需要外部定位)，critic 有 privileged info (reference motion 的全局位置)。这样部署时不需要 odometry。
4. **Termination curriculum**: 初始 tracking error tolerance 1.5m，逐步收紧到 0.3m。解决 "动作太难，policy 早期学不会就放弃" 的问题。
5. **多仿真器评估**: 不仅做 sim2real，还做 IsaacGym->IsaacSim 和 IsaacGym->Genesis 的 sim2sim，证明 delta action 方法的通用性。

## 这篇论文之后发生了什么
- 与 BeyondMimic 形成互补: BeyondMimic 解决 skill composition，ASAP 解决 sim2real gap
- Delta action learning 的思路可以扩展到任何 sim2real 场景 (不限于 humanoid)
- 开源了多仿真器训练和评估 codebase
- 局限: 需要 motion capture 设备收集真实数据; delta model 是 motion-specific 的，换新动作需要重新收集和训练

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **Delta action > delta dynamics** -- 在仿真输入端修正保持物理一致性，在输出端修正会破坏物理约束 | 灵巧手 sim2real 也应考虑 residual action 而非 residual state |
| 2 | **Real data 量很少就够用** -- ASAP 只需几分钟 real rollout 就能训出有效的 delta model | 不需要海量真实数据，几次部署就够 fine-tune |
| 3 | **"修仿真器" 比 "让 policy 鲁棒" 更直接** -- domain randomization 是间接方案 (让 policy 适应不确定性)，delta action 是直接方案 (让仿真更准) | 两条路可以组合: 先 ASAP 修仿真，再少量 DR 处理剩余不确定性 |
| 4 | **Phase-only goal (不需要 odometry) 是部署友好的设计** -- asymmetric actor-critic 让 training 用 privileged info 但 deployment 只需 onboard sensors | 灵巧手也可以用 asymmetric AC: training 用 object pose GT，deployment 用 tactile/vision |
| 5 | **Post-training 是 FM 时代的关键范式** -- pre-train in sim, post-train with real data，类似 LLM 的 pre-train + RLHF | 这就是 robotics 版的 alignment: 让仿真中学到的行为 align 到真实物理 |
