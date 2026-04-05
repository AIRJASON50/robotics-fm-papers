# Soft Actor-Critic -- 学习笔记

> 一句话: 在 maximum entropy RL 框架下设计 off-policy actor-critic 算法, 同时最大化累积奖励和策略熵, 在连续控制任务上实现了 sample efficiency 和稳定性的兼顾.
> 论文: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine, UC Berkeley, ICML 2018
> 引用量级: ~10,000+

## 这篇论文解决了什么问题

2018 年时, deep RL 在连续控制上有两类方法都不令人满意: on-policy 方法 (TRPO, PPO, A3C) 训练稳定但 sample efficiency 极低, 需要海量新数据; off-policy 方法 (DDPG) 能复用历史数据但极度 brittle -- 对超参数敏感, 不同随机种子结果差异巨大, 在高维任务 (如 Humanoid 21 维动作) 上经常彻底失败. 根本原因是确定性策略 (deterministic policy) 容易收敛到局部最优且探索不足.

## 核心想法 (用直觉解释)

SAC 的核心洞察是: 在 RL 目标函数中加入策略的 entropy, 即 J(pi) = sum E[r(s,a) + alpha * H(pi(.|s))]. 这意味着 agent 不仅要最大化奖励, 还要"尽可能随机地行动" -- 在奖励相同的情况下优先选择更不确定的策略.

这看起来反直觉, 但有三个关键好处: (1) 鼓励探索 -- 策略不会过早坍缩到单一动作, 能发现更多高回报区域; (2) 多模态行为 -- 当多个动作同样好时, 策略会给它们分配近似相等的概率, 而非随意挑一个; (3) 训练稳定性 -- 论文实验显示, 确定性策略 (去掉 entropy 的 SAC) 在不同随机种子下方差极大, 而随机策略加 entropy 后方差显著缩小.

算法结构是标准的 actor-critic: actor 输出高斯分布 (均值+方差), critic 用两个独立 Q-network (取 min 防止过高估计, 借鉴 TD3). 训练时从 replay buffer 采样, 分别优化 critic (soft Bellman residual), actor (最大化 Q - alpha*log_pi), 和 target network (exponential moving average). Temperature alpha 控制 reward 和 entropy 的相对重要性, 是唯一需要仔细调的超参数.

## 关键设计决策

1. **Stochastic policy + entropy maximization 而非 deterministic policy**: 这是与 DDPG/TD3 的根本区别. DDPG 输出确定性动作, 探索靠加噪声; SAC 输出高斯分布, 探索是策略本身的属性. 论文 Figure 2 直接对比了两者: stochastic SAC 在 Humanoid 上 5 个 seed 表现一致, deterministic variant 则方差巨大.

2. **Dual Q-networks (clipped double Q)**: 借鉴 Fujimoto et al. (TD3) 的思路, 用两个独立 Q-network 取 min, 缓解 Q-value 过高估计. 论文指出单 Q-network 也能工作, 但双 Q 显著加速训练, 尤其在难任务上.

3. **Reward scale 作为隐式 temperature**: alpha 决定了 entropy 的权重. reward 乘以常数等价于调整 alpha. 论文发现 reward scale 是最敏感的超参 -- 太小则策略过于随机不学习, 太大则退化为确定性策略失去探索. 后续 SAC v2 引入了 automatic temperature tuning (constrained optimization 自动调 alpha).

## 这篇论文之后发生了什么

SAC 成为连续控制 off-policy RL 的标准算法. 后续 SAC v2 (2019) 加入自动 temperature 调节. SAC 的 max-entropy 思想影响了更广泛的领域: RLHF 中的 KL penalty (限制策略偏离参考模型) 与 entropy regularization 是同一类思想; Diffusion Policy 中的随机性生成也受到类似启发. 在机器人领域, SAC 广泛用于真机在线 RL (sample efficiency 关键) 和 sim2real 的 fine-tuning 阶段.

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | PPO vs SAC 的选择取决于场景: 大规模并行 sim (IsaacGym 4096 envs) 用 PPO; 真机/低并行度/需 sample efficiency 用 SAC | 你的灵巧手 sim2real 主要用 PPO (因为 IsaacGym 并行), 但真机 fine-tuning 阶段 SAC 可能更合适 |
| 2 | Entropy regularization 的思想是通用的 -- SAC 中是 max entropy, RLHF 中是 KL penalty, 本质都是"在优化目标上加正则项防止策略坍缩" | 理解 SAC 的 entropy bonus 直接帮助理解 RLHF 中 KL penalty 为什么重要以及如何调节 |
| 3 | Stochastic policy 比 deterministic policy 更稳定 -- 这个结论在机器人 FM 中体现为 Diffusion Policy 的随机采样 | 确定性 policy (如 BC) 容易 mode collapse; 随机生成 (diffusion, flow) 天然多模态, 这与 SAC 的 entropy 思想一脉相承 |
