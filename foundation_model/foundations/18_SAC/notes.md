# Soft Actor-Critic (Haarnoja et al., 2018) -- Takeaway Notes

> 一句话: 在 maximum entropy RL 框架下做 off-policy actor-critic, 兼顾 sample efficiency 和探索能力.

## 核心贡献
- 将 entropy bonus 加入 RL objective: maximize E[sum(r + alpha * H(pi))], 鼓励策略保持随机性
- Off-policy 训练: 用 replay buffer 复用历史数据, sample efficiency 远高于 PPO
- 自动调节 temperature alpha: 通过 constrained optimization 自适应平衡 reward 和 entropy

## 为什么重要
SAC 是连续控制领域最强的 off-policy 算法. 在 sample efficiency 受限的场景 (如真实机器人
训练, 不能跑几千个并行 sim) 中, SAC 比 PPO 更合适. 它的 max-entropy 思想也被广泛借鉴:
entropy regularization 已成为 RL 的标准技巧, 甚至影响了 RLHF 中 KL penalty 的设计.

## 对你 (RL->FM) 的 Takeaway
- PPO vs SAC 的选择: 大规模并行 sim (IsaacGym) 用 PPO; 真机或低并行度场景用 SAC.
  原因是 PPO 的 on-policy 特性需要大量新数据, 而 SAC 的 replay buffer 可以反复利用.
- SAC 的 entropy regularization 思想与 RLHF 中的 KL penalty (限制 policy 偏离
  reference model) 是同一类想法 -- 都是在 reward maximization 上加正则化防止 collapse.

## 与知识库其他内容的关联
- 17_PPO: 主要竞品 -- on-policy, 大规模并行友好, 但 sample efficiency 低
- 15_DQN: SAC 继承了 replay buffer + target network, 扩展到连续动作 + actor-critic
- 15_Adam: SAC 的三个网络 (actor, 2x critic) 都用 Adam 训练
- 14_GAN: SAC 的 entropy maximization 与 GAN 的 mode diversity 目标有精神联系
