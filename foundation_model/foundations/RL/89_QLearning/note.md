# Q-Learning (1989, Christopher Watkins)

**来源**: "Learning from Delayed Rewards" (1989, PhD Thesis, Cambridge University)

**Watkins 要解决的问题**: Bellman 方程需要知道环境模型 P(s'|s,a)。现实中大多数情况不知道环境模型, 怎么办?

**核心 insight**: 不需要知道 P(s'|s,a), 只要能和环境交互, 就可以从经验中学习 Q*(s,a)

**Q-Learning 更新规则**:

```
Q(s,a) ← Q(s,a) + α * [ r + γ * max_a' Q(s',a') - Q(s,a) ]

α: 学习率
r: 当前奖励
γ: 折扣因子
max_a' Q(s',a'): 下一个状态最优动作的 Q 值 (来自 Bellman 方程)
r + γ * max_a' Q(s',a') - Q(s,a): TD error (实际 vs 预估的差)

→ 每次交互更新一个 (s,a) 对的 Q 值
→ 收敛后 Q*(s,a) 满足 Bellman 最优方程
→ 最优策略: π*(s) = argmax_a Q*(s,a)
```

**为什么重要**:

```
之前 (动态规划): 需要完整的环境模型 → 只能在已知环境中求解
Q-Learning:     不需要模型, 和环境交互就能学 → Model-free RL 诞生

但局限:
  Q 值存在表格中: 每个 (s,a) 对一个格子
  → Atari: 状态 = 210×160×3 像素 → 表格装不下
  → 2013 DQN: 用 NN 替代表格 → Deep RL
```

**思想传承**:

```
Q-Learning (表格)
  → DQN (Q 表 → NN)
  → Double DQN, Dueling DQN, Rainbow (DQN 的改进)
  → SAC (Q function + policy + 最大熵)

另一条线 (不学 Q, 直接学 policy):
  REINFORCE (1992) → A3C → TRPO → PPO
  → 你用的 PPO 走的是这条线
```

**对你的意义**: Q-Learning 是 model-free RL 的起点。你虽然用 PPO (policy gradient 路线) 而不是 Q-Learning (value-based 路线), 但 PPO 中的 Critic 网络本质上就是在做 Q/V 的近似。两条路线在 Actor-Critic 方法中汇合了。
