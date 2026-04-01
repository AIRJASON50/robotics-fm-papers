# Proximal Policy Optimization (Schulman et al., 2017) -- Takeaway Notes

> 一句话: 用 clipped surrogate objective 近似 TRPO 的信赖域约束, 实现简单高效的 on-policy policy gradient.

## 核心贡献
- 提出 clipped surrogate loss: clip(r_t, 1-eps, 1+eps) * A_t, 限制策略更新幅度
- 相比 TRPO 不需要二阶优化 (conjugate gradient), 实现只需几行代码
- 支持多 epoch mini-batch 更新同一批数据, 大幅提高 sample 利用率 (相比 vanilla PG)

## 为什么重要
PPO 是目前最广泛使用的 RL 算法: OpenAI 的 RLHF, IsaacGym/IsaacLab 的 sim2real,
几乎所有 on-policy 机器人学习都用 PPO. 它的成功在于: 实现简单, 超参数不敏感,
并行化友好 (可以用数千个并行环境收集数据). 这让大规模 RL 训练成为可能.

## 对你 (RL->FM) 的 Takeaway
- 你日常用的 PPO 核心就是 clip ratio + GAE advantage estimation. 关键超参数:
  clip_range (0.2), GAE lambda (0.95), entropy bonus, value loss coefficient.
  理解每个超参数的物理含义比调参更重要.
- PPO 在 RLHF 中训练 LLM policy (language model 即 policy, token 即 action),
  这意味着你的 sim2real PPO 经验可以直接迁移到理解 RLHF 的训练过程.

## 与知识库其他内容的关联
- 18_SAC: PPO 的主要替代方案 -- off-policy, 适合 sample efficiency 要求高的场景
- 15_DQN: PPO 走了 policy gradient 路线, DQN 走了 value-based 路线
- 15_Adam: PPO 的 actor 和 critic 网络都用 Adam 优化
- 14_GAN: GAIL 将 GAN + PPO 结合做 imitation learning
