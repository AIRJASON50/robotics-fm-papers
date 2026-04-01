# Human-level Control through Deep RL (Mnih et al., 2015) -- Takeaway Notes

> 一句话: 用 CNN 直接从像素输入学玩 Atari, 首次证明 deep learning + RL 可以端到端工作.

## 核心贡献
- 提出 Deep Q-Network: CNN 直接从原始像素提取特征, 输出 Q-value
- 引入 experience replay buffer 打破数据时序相关性, 稳定训练
- 引入 target network (定期同步) 解决 Q-learning 的 moving target 问题

## 为什么重要
DQN 是 deep RL 的开山之作. 在此之前, RL 和 deep learning 是两个独立社区.
DQN 证明了: (1) 神经网络可以做 function approximator 来逼近 Q 函数;
(2) 用正确的工程技巧 (replay buffer, target net) 可以让训练稳定收敛.
所有后续 deep RL 工作 (A3C, PPO, SAC) 都建立在这个基础上.

## 对你 (RL->FM) 的 Takeaway
- experience replay 的思想在 off-policy 算法 (SAC, TD3) 中是核心; PPO 是 on-policy
  不用 replay buffer -- 理解这个区别是选择算法的关键.
- DQN 只处理离散动作空间; 机器人连续控制需要 PPO/SAC. 但 DQN 引入的
  "neural net + replay + target net" 三件套仍然是你理解 SAC 的前置知识.

## 与知识库其他内容的关联
- 17_PPO: 走了另一条路 -- policy gradient + clipping, 不需要 replay buffer
- 18_SAC: 继承了 DQN 的 replay buffer + target net, 但扩展到连续动作空间
- 15_Adam: DQN 用 RMSProp, 后续算法基本都切到了 Adam
