# REINFORCE (1992, Ronald Williams)

**来源**: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992, Machine Learning Journal)

**Williams 要解决的问题**: Q-Learning 学的是 value function, 然后从中导出 policy。能不能跳过 value function, 直接优化 policy 本身?

**核心 insight**: Policy Gradient Theorem — 可以直接对策略参数求 "期望回报" 的梯度

**REINFORCE 算法**:

```
策略: π_θ(a|s) — 一个参数化的概率分布 (如 NN 输出的高斯分布)
目标: 最大化期望回报 J(θ) = E[Σ γ^t r_t]

梯度: ∇J(θ) = E[ Σ_t ∇log π_θ(a_t|s_t) * G_t ]

G_t: 从 t 时刻开始的累积回报 (return)
∇log π_θ: 策略的 log 概率的梯度
→ 回报高的动作 → 增大其概率; 回报低的 → 减小

更新: θ ← θ + α * ∇J(θ)
```

**为什么重要**:

```
Q-Learning 路线:
  学 Q(s,a) → 从 Q 导出 policy → 只能处理离散动作 (argmax)
  
Policy Gradient 路线 (REINFORCE):
  直接学 π(a|s) → 天然支持连续动作 (输出高斯分布的 mean 和 std)
  → 机器人控制是连续动作 → Policy Gradient 路线成为机器人 RL 的主流
```

**REINFORCE 的问题和后续改进**:

```
REINFORCE 的问题:
  G_t 的方差很大 (一条轨迹的回报波动大)
  → 梯度估计不稳定 → 学习慢且不稳定

解决方案 (逐步改进):
  REINFORCE (1992):     G_t 做 baseline 减方差 → 仍然不够稳定
  A3C (2016, DeepMind): Actor-Critic, 用 Critic 估计 advantage → 减方差
  TRPO (2015, Schulman): 限制每步更新幅度 (KL 约束) → 稳定但计算复杂
  PPO (2017, Schulman):  用 clip 替代 KL 约束 → 简单且稳定 → 成为默认

  这条线: REINFORCE → A3C → TRPO → PPO
  PPO 就是 REINFORCE 的曾孙
```

**REINFORCE vs Q-Learning — 两条路线在 Actor-Critic 中汇合**:

```
Value-based (Q-Learning 路线):
  只学 Q/V → 离散动作 → DQN, Rainbow
  
Policy-based (REINFORCE 路线):
  只学 π → 连续动作 → REINFORCE, 但方差大

Actor-Critic (汇合):
  Actor (π): 学策略 (REINFORCE 的后代)
  Critic (V/Q): 学价值函数 (Q-Learning 的后代)
  → Actor 用 Critic 的估计减方差 → 两全其美
  → PPO, SAC 都是 Actor-Critic 方法
  → 你的 PPO 中 actor_net 和 critic_net 就是这两个
```

**对你的意义**: 你用的 PPO 的 actor 网络直接继承自 REINFORCE 的 policy gradient 思想, critic 网络继承自 Q-Learning 的 value estimation 思想。理解 REINFORCE 就理解了 PPO 的 actor 在做什么: ∇log π * Advantage。
