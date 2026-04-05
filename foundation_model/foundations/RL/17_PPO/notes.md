# Proximal Policy Optimization Algorithms -- 学习笔记

> 一句话: 用 clipped probability ratio 近似 TRPO 的信赖域约束, 实现简单且稳定的 on-policy policy gradient 算法, 成为 RL 领域的默认选择.
> 论文: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, OpenAI 2017
> 引用量级: ~18,000+

## 这篇论文解决了什么问题

2017 年时 deep RL 的主要方法各有严重缺陷: DQN 只能处理离散动作空间; vanilla policy gradient 的 sample efficiency 极低且更新步长难以控制; TRPO 虽然理论上保证单调改进, 但需要二阶优化 (conjugate gradient), 实现复杂且不支持 dropout / 参数共享. 需要一个既有 TRPO 的稳定性, 又像 vanilla PG 一样简单的算法.

## 核心想法 (用直觉解释)

Policy gradient 的核心问题是: 每次更新策略时, 步子迈多大? 太小则学得慢, 太大则策略可能崩溃 (因为旧数据在新策略下的分布已经改变). TRPO 的解法是在 KL divergence 约束下优化, 但需要昂贵的二阶计算.

PPO 的核心想法极其简洁: 计算新旧策略的 probability ratio r_t = pi_new(a|s) / pi_old(a|s), 然后直接 clip 它到 [1-epsilon, 1+epsilon] 范围内. 如果某个 action 的 advantage 是正的 (好动作), ratio 被 clip 在 1+epsilon, 阻止策略过度向这个动作偏移; 如果 advantage 是负的 (差动作), ratio 被 clip 在 1-epsilon. 最终 loss 取 clipped 和 unclipped 的 min, 形成一个悲观下界 -- 只允许保守的改进.

整个算法的流程是: N 个并行 actor 收集 T 步数据, 计算 GAE advantage, 然后在这批数据上跑 K 个 epoch 的 mini-batch SGD 优化 clipped loss. 相比 vanilla PG (收集一次用一次就丢), PPO 能对同一批数据多次更新, 大幅提高利用率.

## 关键设计决策

1. **Clip 而非 KL penalty**: 论文同时测试了 adaptive KL penalty 方案 (Section 4), 但 clipping (epsilon=0.2) 效果更好且实现更简单. 不需要调 KL 目标值, 一个超参数 epsilon 就够了.

2. **多 epoch mini-batch 更新**: 传统 policy gradient 收集一批数据只能用一次 (on-policy 约束). PPO 的 clipping 机制允许在同一批数据上跑多个 epoch (通常 3-10 个), 因为 clip 会自动阻止策略偏离太远. 这对大规模并行训练 (IsaacGym 数千个环境) 至关重要.

3. **联合优化 actor + critic + entropy**: 实际 loss 是三项之和 -- clipped policy loss + value function loss + entropy bonus. Entropy bonus 鼓励探索, value loss 训练 critic 提供更好的 advantage 估计.

## 这篇论文之后发生了什么

PPO 成为 RL 领域的事实标准: OpenAI 的 RLHF 用 PPO 训练 ChatGPT (language model 即 policy, token 即 action); IsaacGym/IsaacLab 的机器人 sim2real 几乎全部使用 PPO; DeepMind 的 locomotion/manipulation 研究也大量使用. PPO 的"简单 + 鲁棒 + 可并行"特性使得大规模 RL 成为可能.

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | PPO 的 clip 机制本质是"限制策略更新步长" -- 同一思想在 RLHF 中体现为 KL penalty (限制 LM 偏离 reference model) | 你的 PPO sim2real 经验可以直接迁移到理解 RLHF 的训练动态: 都是在约束下优化 policy |
| 2 | PPO 的 on-policy 特性要求大量并行数据收集, 这决定了它适合 sim (IsaacGym 4096 envs) 而非真机 | 真机 RL 场景考虑 SAC (off-policy, 可复用历史数据); 但当有高质量 sim 时 PPO 更稳定 |
| 3 | 关键超参: epsilon=0.2, GAE lambda=0.95, entropy coeff, learning rate -- 理解物理含义比调参重要 | epsilon 控制策略改变幅度, lambda 控制 bias-variance trade-off, entropy 控制探索程度 |
