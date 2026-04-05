# Playing Atari with Deep Reinforcement Learning -- 学习笔记

> 一句话: 首次成功将 CNN 与 Q-learning 结合, 直接从原始像素端到端学习控制策略, 开创了 deep RL 领域.
> 论文: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves 等, DeepMind 2013 (Nature 2015 版)
> 引用量级: ~30,000+ (含 Nature 版)

## 这篇论文解决了什么问题

2013 年之前, RL 和 deep learning 是两个基本不交叉的社区. RL 依赖手工特征 + 线性函数逼近; deep learning 在监督学习上成功 (ImageNet) 但不知道能否用于 RL. 将两者结合面临三个技术障碍: (1) RL 的训练数据高度相关 (连续帧), 违反 i.i.d. 假设; (2) 数据分布随策略改变而变化 (non-stationary); (3) 奖励信号稀疏且延迟, 不像分类任务有明确标签.

## 核心想法 (用直觉解释)

DQN 的核心是: 用一个 CNN 直接将游戏画面 (84x84 灰度图, 堆叠 4 帧) 映射为每个动作的 Q-value. Q(s,a) 估计在状态 s 下执行动作 a 后能获得的未来累积回报. Agent 只需选择 Q 值最大的动作. 训练时最小化 Bellman error: 当前 Q 值与 (reward + 下一状态最大 Q 值) 之间的差距.

关键创新不在网络架构 (只是标准 CNN), 而在两个稳定训练的工程技巧: (1) **Experience Replay** -- 将所有经历 (s, a, r, s') 存入固定大小的 buffer (100 万条), 训练时随机采样 mini-batch. 这打破了数据的时序相关性, 同时让一条经验可以被多次学习. (2) **Target Network** -- 用一个参数定期同步 (而非实时更新) 的"目标网络"计算 TD target, 避免"自己追自己"的发散问题.

论文在 7 个 Atari 游戏上测试, 使用完全相同的网络结构和超参数, 在 6 个游戏上超过此前所有方法, 在 3 个游戏上超过人类专家. 网络只接收原始像素和分数, 不使用任何游戏特定的先验知识.

## 关键设计决策

1. **输出所有动作的 Q-value 而非单个**: 网络输入只有 state, 输出是每个离散动作对应的 Q-value. 这样一次 forward pass 就能得到所有动作的估值, 比逐个 (s,a) pair 输入高效得多. 但这也限制了 DQN 只能处理离散动作空间.

2. **Reward clipping 到 {-1, 0, +1}**: 不同游戏的奖励尺度差异巨大, 统一 clip 到 [-1, 1] 使得同一组超参数可以跨游戏工作. 代价是丧失了奖励幅度信息.

3. **Epsilon-greedy 探索**: 以 epsilon 概率随机选动作, 其余贪心. epsilon 从 1.0 线性退火到 0.1. 这是最简单的探索策略, 后续工作 (SAC 的 entropy, curiosity-driven 等) 提出了更好的方案.

## 这篇论文之后发生了什么

DQN 直接催生了整个 deep RL 领域: Double DQN (解决 Q 值过高估计), Dueling DQN, Prioritized Replay, Rainbow (集大成), A3C/A2C (并行 actor), 然后分化为 policy gradient 路线 (TRPO -> PPO) 和 off-policy value-based 路线 (DDPG -> TD3 -> SAC). DQN 的 experience replay 和 target network 这两个技巧至今仍是 off-policy 算法的标配.

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | DQN 证明了"端到端学习 perception + control"是可行的 -- 不需要手工特征, 网络自己学 | 这个思想直接延伸为 VLA: 从像素到语言理解到动作生成, 全部端到端, 不再需要手工 perception pipeline |
| 2 | Experience replay + target network 是 off-policy 稳定训练的核心技巧, 至今被 SAC 继承 | 理解 DQN 的这两个技巧是理解 SAC 的前置知识; PPO 走了另一条路 (on-policy, 不用 replay) |
| 3 | DQN 只能处理离散动作 -- 机器人需要连续控制, 这催生了 DDPG -> SAC 的技术路线 | 你用的 PPO/SAC 都是为了解决 DQN 无法处理连续动作空间的问题 |
