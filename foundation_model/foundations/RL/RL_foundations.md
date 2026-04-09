# 强化学习基础: 思想源头 + 概念速查 + 发展脉络

---

## "强化"这个词从哪来

Reinforcement Learning 的名字不是计算机科学家发明的, 而是从心理学借来的。理解 RL 必须从这个词的原始含义出发。

### 心理学根源: 行为如何被"强化"

```
1898  Thorndike "Animal Intelligence" (动物智能)
  实验: 猫被关在笼子里, 笼外有鱼
        猫随机乱动 → 偶然碰到机关 → 门开了 → 吃到鱼
        下一次碰到机关的速度更快
  观察: 带来满意结果的行为, 在后续更容易出现

1911  Thorndike "效果律" (Law of Effect)
  正式表述: "伴随满意结果的行为倾向于重复, 伴随不适结果的倾向于消失"
  → 这是一个心理学观察, 不是算法, 不是数学

1938  Skinner "The Behavior of Organisms"
  首次正式使用 "reinforcement" (强化) 这个词
  定义: reinforcement = 增强某个行为再次发生概率的刺激
    positive reinforcement: 给奖励 → 行为被加强 (猫碰机关 → 给鱼)
    negative reinforcement: 移除不适 → 行为被加强 (按按钮 → 电击停止)
  → "强化"的原始含义: 好的行为被加强, 坏的行为被抑制
```

### 数学根源: 序贯决策怎么求最优

```
1950s  Bellman "Dynamic Programming" (动态规划)
  完全独立于心理学, 来自工程/军事需求:
    导弹轨迹规划: 怎么选择一系列控制量让导弹命中目标?
    库存管理: 怎么决定每天订多少货让总成本最低?
  
  Bellman 方程: V*(s) = max_a [ R(s,a) + γ * Σ P(s'|s,a) * V*(s') ]
  → 把"规划整个未来"变成"当前一步 + 递归"
  → 纯数学, 不涉及"学习", 假设环境完全已知
```

### 中间 30 年: 为什么心理学和数学没有立刻合流

```
1938-1988 这 50 年间两条线独立发展, 没有合流:

心理学这边:
  Skinner 的行为主义统治心理学 (1940s-1960s)
  但行为主义只描述"什么行为被强化", 不给出"怎么学最优行为"的算法
  → 直觉有了, 但不知道怎么用数学实现

数学这边:
  Bellman 的动态规划在工程领域广泛使用 (控制、运筹学)
  但需要完整的环境模型 P(s'|s,a) → 只能在已知系统中求解
  → 框架有了, 但不能处理"不知道环境模型, 只能试错"的情况

缺的那一块:
  心理学: 知道"从结果中学习"但没有数学
  Bellman: 有数学但假设"什么都知道"
  需要: 一种"不知道环境模型, 但能从经验中逐步逼近最优"的算法
  → 这就是 1988-1989 年 Sutton 和 Watkins 填补的空白
```

### 两条线的合流: RL 作为学科诞生

```
1988  TD Learning (Sutton)
  问题: Bellman 方程需要完整环境模型, 但现实中不知道
        能不能只从"经验" (实际交互的 state-reward 序列) 中学?
  
  关键 insight: 不必等到最终结果才更新估计
    之前: 走完整条路 → 看最终回报 → 更新所有步 (蒙特卡罗, 方差大)
    TD:   每走一步 → 用"下一步的估计"修正"当前的估计" → 即时更新
    
    V(s) ← V(s) + α * [ r + γ V(s') - V(s) ]
                         └─ TD target ─┘  └ 当前估计 ┘
                         (一步实际 + 下一步估计)
    
    → 把 Bellman 方程从"已知环境的精确求解"变成"未知环境的在线近似"
    → 这是心理学 (从经验学习) + 数学 (Bellman 递归) 的合流点

1989  Q-Learning (Watkins)
  在 TD 基础上更进一步:
    TD 学的是 V(s) (状态价值) → 还需要知道模型才能选动作
    Q-Learning 学 Q(s,a) (状态-动作价值) → 直接 argmax 选动作, 不需要模型
    
    → 第一个完整的 model-free 最优控制算法
    → 但 Q 值存在表格中 → 状态空间大了表格放不下

1992  REINFORCE (Williams)
  另一条路: 不学 value function, 直接优化 policy
    → 天然支持连续动作 (输出高斯分布)
    → 机器人控制需要连续动作 → 这条路成为机器人 RL 的主流
    → 但梯度方差大, 学习不稳定

  至此两条技术路线成型:
    Value-based: Bellman → TD → Q-Learning (离散动作, 表格)
    Policy-based: REINFORCE (连续动作, 但不稳定)
    → 两线在 Actor-Critic 中汇合: Actor 学 policy + Critic 学 value → 互相帮助

1997  TD-Gammon (Tesauro)
  用 NN + TD Learning 下西洋双陆棋 → 达到人类世界冠军水平
  → 比 DQN 早 16 年证明了 NN + RL 可行
  → 但当时 NN 规模太小, 没有 GPU, 无法推广到其他任务
  → 被遗忘了 16 年, 直到 DQN (2013) 用更大的 NN + 更多算力重新证明

Sutton 的自述 (RL 教科书 Chapter 1):
  "RL 有两个独立的思想根源:
   1. 动物学习心理学 — trial-and-error, reinforcement (Thorndike, Skinner)
   2. 最优控制理论 — Bellman equation, dynamic programming
   Neither thread alone is adequate."
```

### 所以 "Reinforcement Learning" 的准确含义

```
Reinforcement: 来自心理学 — 行为被结果强化 (好结果 → 行为加强)
Learning:      来自机器学习 — 从经验中自动改进

合在一起: 
  不是"在环境中交互并获得 reward" (这是后来的数学形式化)
  原始含义是: 像动物一样, 通过尝试和结果反馈, 自动学会有效的行为
  
  后来被形式化为:
    agent + environment + state + action + reward + policy
    但这是 1990s 的数学包装, 不是 RL 的本质

  本质就一句话: 好的行为被强化, 坏的被抑制, 从而学会决策
```

## "强化"和"学习"分别指什么

```
Reinforcement (强化) = R = 环境返回的标量评价信号

  Williams 1992 原文 (p.231):
    "The evaluation consists of the scalar REINFORCEMENT SIGNAL r,
     which we assume is broadcast to all units in the network.
     At this point each unit performs an appropriate modification
     of its weights."

  → reinforcement 在论文中就是那个数字 r, 不是抽象概念

R 和 V 的关系:
  R: 环境给的即时评分 (外部信号, 每步或终点才有)
  V: "从这个状态开始的预期总 R" (内部估计, agent 自己维护)
  V 是对未来所有 R 的压缩/总结

Learning (学习) = 用 R 更新内部表征使决策改进

  Bellman (1957): 有 R 的概念, 但 V 是精确计算的, 不是学的
    → 这是 optimization/planning, 不是 learning

  TD/Q-Learning (1988-1989): V 从实际拿到的 R 中估计
    → 必须试错 → 每次拿到 R → 更新 V/Q → 逐步逼近
    → 这才是 learning: 权重在变 = 在学

  "强化学习" = 用环境的评分信号 (R) 通过试错来学习 (更新 V/policy)
  → Bellman 有"强化"(R) 但没有"学习"(V 是算的)
  → RL 两者都有: 从实际的 R 中学习 V
```

---

## RL 的数学框架 (Sutton & Barto 形式化)

上面是思想来源, 下面是后来的数学包装。

```
智能体 (Agent) 与环境 (Environment) 的交互循环:
  Agent 观察状态 s_t → 执行动作 a_t → 环境返回奖励 r_t 和新状态 s_{t+1} → 循环

目标: 找到策略 π(a|s), 使得从任意起始状态出发, 累积折扣奖励最大化:
  maximize  E[ Σ_{t=0}^{∞} γ^t * r_t ]

γ (折扣因子, 0 < γ < 1): 未来奖励的衰减系数, 体现"近期奖励比远期更确定"
```

**RL 的核心难题**:

- **探索 vs 利用 (exploration vs exploitation)**: 是继续尝试未知动作 (可能发现更好的), 还是重复已知的好动作 (保证当前收益)?
- **延迟奖励 (delayed reward)**: 当前动作的好坏可能要很久之后才知道 (如围棋要下完才知道输赢), 怎么把最终结果归功于中间的每一步?
- **高维连续空间**: 机器人的状态和动作都是高维连续向量, 不像棋盘是有限离散状态

RL 的 70 年发展历史, 就是在不同约束下逐步解决这些难题的过程。

---

## 核心概念速查

### State vs Observation (状态 vs 观测)

详见: `state_vs_observation.md`

```
正式定义 (Sutton & Barto):
  State:       环境的完整描述, 满足 Markov 性质
  Observation: agent 实际感知到的, 可能是 state 的部分/含噪版本

机器人社区的非正式用法:
  "State-based": 低维数值向量 (本体感觉 + 物体 ground-truth + 仿真特权信息)
  "Vision-based": 高维感知输入 (图像/点云) + 本体感觉

注意: "State-based" 在 manipulation 中几乎总是包含物体 ground-truth 位姿, 不只是本体感觉
```

### MDP vs POMDP

```
MDP (Markov Decision Process):
  (S, A, P, R, γ)
  Agent 能看到完整 state → 策略 π(a|s)
  你的仿真 RL 训练就是 MDP (仿真器给 ground truth)

POMDP (Partially Observable MDP):
  (S, A, P, R, O, Z, γ)
  Agent 只能看到 observation → 策略 π(a|o) 或 π(a|history)
  真机部署 (只有相机) 就是 POMDP
```

### Policy (策略)

```
π(a|s): 给定状态, 输出动作的概率分布
  确定性策略: a = π(s)
  随机策略:   a ~ π(·|s)

你用的:
  PPO policy: MLP, 输出高斯分布的 mean 和 std
  VLA policy: Transformer + Flow Matching, 输出 action chunk
```

### Value Function (价值函数)

```
V(s):  从状态 s 出发, 遵循策略 π, 期望累积回报
Q(s,a): 从状态 s 执行动作 a, 然后遵循策略 π, 期望累积回报
Advantage: A(s,a) = Q(s,a) - V(s) (这个动作比平均好多少)

PPO 用 Advantage 做策略更新
pi*0.6 的 RECAP 也是基于 Advantage 的离线 RL
```

### GAE: 把 Bellman、TD、Monte Carlo 统一成一个公式

GAE (Generalized Advantage Estimation, Schulman 2016) 是 PPO 的核心组件, 也是理解整个 RL 脉络的最佳切入点。

```
公式: A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
其中: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)   ← TD error (单步预测误差)

δ_t: "这一步实际比预期好多少" (近处的 δ 更可信, 远处的不确定)
A_t: 多个 δ 的加权和 = "这个动作综合来看比平均好多少" (Actor 用它更新 policy)
```

```
γ 和 λ 的区别:
  γ (gamma, 默认 0.99): 在 δ 内部, 定义"这个问题关心多远的未来"
    → V(s) 本身怎么算 → 问题的性质, 几乎不调
  λ (lambda, 默认 0.95): 在 δ 外部, 定义"学习信号看多远"
    → 多少步的 δ 聚合成 A → 算法的选择, 可调
  分成两个参数的原因: V 的时间尺度和 A 的信号范围可以独立控制
    γ=0.99: Critic 看得远 (长期规划)
    λ=0.5:  但 Actor 只看近处信号 (Critic 不准时不传太远)
```

```
λ 统一了 RL 的三种方法:
  λ=0:    A=δ_t               → 纯 TD Learning (1988, Sutton)
  λ=0.95: A=δ+0.94δ'+0.88δ"... → PPO 默认 (折中)
  λ=1:    A=G_t-V(s_t)        → 纯 Monte Carlo (1960s)

  → GAE 不是新方法, 是用一个参数 λ 把所有旧方法统一成一个连续谱
  → "Generalized" = 广义的, 包含所有特殊情况
```

```
GAE 和 RL 脉络中每个 milestone 的关系:

  Bellman (1957):   V(s) = r + γ·V(s') 的递归结构
                    → GAE 的 (γλ)^l 幂次衰减就是 Bellman 递归展开的直接结果

  TD Learning (1988): δ = r + γV(s') - V(s) 的单步更新
                    → GAE 的 δ_t 就是 TD error, GAE 把多个 δ 加权聚合

  Monte Carlo:      用真实总回报 G_t 更新
                    → GAE 在 λ=1 时退化为 G_t - V(s_t)

  Q-Learning (1989): 学 Q(s,a) 而非 V(s)
                    → GAE 用 V(s) 估 Advantage, Q-Learning 直接估 Q
                    → 两条路线在 Actor-Critic 中汇合

  REINFORCE (1992):  ∇J = E[∇log π · G_t] 用完整回报
                    → PPO 把 G_t 替换为 GAE 的 A_t → 减方差

  DQN (2013):       用 NN 近似 Q
                    → GAE 中的 V(s) 也是 NN (Critic) 近似的

  PPO (2017):       Actor 用 clip(ratio, 1±ε) · A_t 更新
                    → A_t 就是 GAE 算出来的
                    → Critic 用 δ_t 更新 V(s)
                    → GAE 是 PPO 的 Actor 和 Critic 之间的桥梁

  SAC (2018):       off-policy, 不用 GAE, 用 TD error 直接更新 Q
                    → 但 TD error 的概念和 GAE 的 δ 完全一样
```

### On-Policy vs Off-Policy

```
On-Policy (如 PPO):
  用当前策略收集数据 → 更新策略 → 旧数据丢弃 → 重新收集
  数据利用率低, 但稳定

Off-Policy (如 SAC):
  用任何策略收集的数据都能用 → 存在 replay buffer → 反复利用
  数据利用率高, 但可能不稳定
```

### Reward Shaping vs Sparse Reward

```
Dense reward (reward shaping):
  每步都给 reward signal (距离、角度误差、接触力...)
  → 学得快, 但需要人工设计
  → 你的灵巧手 PPO 用的就是这个

Sparse reward:
  只在成功时给 reward, 其他时候 0
  → 不需要设计, 但学得慢 (探索难)
  → OmniReset 用 diverse resets 让 sparse reward 也能 work
```

### Sim-to-Real Transfer

```
在仿真中训练 → 部署到真机
核心挑战: sim-real gap (仿真和真实世界的差异)

三种缓解方法:
  Domain Randomization (DR): 随机化仿真参数 → 策略对参数不敏感
  System Identification (SysID): 精确测量真机参数 → 仿真更逼真
  Teacher-Student Distillation: teacher 用特权信息训 → student 用可部署传感器
```

### Behavior Cloning (BC) vs RL

```
BC (行为克隆):
  数据: 人类演示 (成功轨迹)
  loss: 预测动作 vs 真实动作的距离
  → 不需要 reward, 不需要仿真器
  → 只能模仿, 不能超越演示者, 有 compounding error
  → RT-1, pi_0 pre-training 用的就是 BC

RL (强化学习):
  数据: agent 自己和环境交互
  loss: 最大化累积 reward
  → 需要 reward function 和环境
  → 可以超越演示者, 可以从失败中学
  → pi*0.6 post-training 用的是 offline RL
```

---

## 发展脉络

### 贯穿 70 年的核心: state → 估 value → 用 value 决策

变化的只有两件事:

| 时代 | state 是什么 | value 怎么估 |
|------|-------------|-------------|
| Bellman (1957) | index (迷宫第几格) | 精确计算 (需完整环境模型) |
| Monte Carlo (1960s) | index | 跑完整条路, 用真实终点分数 |
| TD (1988) | index | 走一步, 用旧估计代替未来 |
| Q-Learning (1989) | index | 表格存每个 (s,a) 的 value |
| DQN (2013) | 高维感知 (像素) | NN 压缩 Q 表格 |
| PPO (2017) | 连续向量 (关节角+位姿) | Critic NN 估 V, Actor NN 用 V 信号优化策略 |

State 和 Value 从来就是两个东西: state = "我在哪" (索引/感知), value = "在这里值多少分" (评分)。表格时代看起来混在一起是因为 state 太简单 (只是编号), NN 时代区别变明显 (state 是高维输入, value 是标量输出)。NN 从高维 state 中压缩出 value, 和 ViT/CLIP 做的事本质相同。

(Watkins Q-Learning 原文 (1992, p.281): "the agent observes its current state x_n, selects and performs an action a_n, observes the subsequent state y_n, receives an immediate payoff r_n" — 从一开始就是 观测 state → 决策 → 执行 → 获得反馈)

### 阶段 1: 理论根基 (1957-1992)

详细描述见上方"两条线的合流"章节, 这里只做索引。

| 年份 | 里程碑 | 做了什么 | 局限 | 文件 |
|------|-------|---------|------|------|
| 1957 | Bellman 方程 | 序贯决策的递归求解 | 需要完整环境模型 | `RL/57_BellmanEquation/` |
| 1988 | TD Learning (Sutton) | 从经验中一步步近似 V, 不需要模型 | 只估 V(s), 不直接给动作 | |
| 1989 | Q-Learning (Watkins) | 直接估 Q(s,a), 不需要模型 | 表格, 高维放不下 | `RL/89_QLearning/` |
| 1992 | REINFORCE (Williams) | 跳过 value, 直接优化 policy | 梯度方差大 | `RL/92_REINFORCE/` |
| 1997 | TD-Gammon (Tesauro) | NN + TD 下棋达世界冠军 | NN 太小无法推广 | |

至此两条技术路线成型: Value-based (Bellman→TD→Q-Learning) 和 Policy-based (REINFORCE), 后在 Actor-Critic 中汇合。

### 阶段 2: Deep RL 革命 (2013-2018)

| 年份 | 里程碑 | 问题 | 贡献 | 文件 |
|------|-------|------|------|------|
| 2013 | DQN (DeepMind) | Q 表格放不下高维状态 | CNN 近似 Q + Experience Replay + Target Network | `RL/15_DQN/` |
| 2017 | PPO (OpenAI) | Policy gradient 步长太大会崩 | Clip ratio 限制更新幅度, 简单稳定 | `RL/17_PPO/` |
| 2018 | SAC (Berkeley) | On-policy 数据利用率低 | Off-policy + 最大熵, 探索利用自动平衡 | `RL/18_SAC/` |

核心贡献不是新理论, 而是让 1990 年代的理论在深度学习时代真正可用。

### 阶段 3: RL 进入机器人 (2017-2021)

| 年份 | 里程碑 | 问题 | 贡献 | 文件 |
|------|-------|------|------|------|
| 2017 | Domain Randomization (Tobin) | 仿真到真机就崩 | 随机化仿真参数, 策略对 gap 鲁棒 | `RL/17_DomainRandomization/` |
| 2019 | OpenAI Rubik's Cube | 能不能纯仿真训灵巧操控? | 大规模 DR + PPO → 真机单手解魔方 | `RL/19_OpenAIDactyl/` |
| 2021 | RMA (Berkeley) | DR 需要猜参数范围 | Teacher 用特权信息 → Student 用历史推断 | `RL/21_RMA/` |

RL 从游戏走向物理世界。

### 阶段 4: FM 时代的 RL (2022-)

| 年份 | 里程碑 | RL 的新角色 |
|------|-------|-----------|
| 2022 | RLHF / InstructGPT | RL 做 LLM 的 post-training 对齐 |
| 2025 | OmniReset | diverse resets + sparse reward + 大规模并行 → 涌现 |
| 2025 | SONIC | PPO 训 universal motion tracker (学追踪, 不学技能) |
| 2025 | pi*0.6 | offline RL 做 VLA post-training (从失败中学) |

RL 从"完整的学习系统"变为"更大 pipeline 中的一个阶段"。

### 隐藏的第二条线: 强化信号怎么进入学习系统

上面四个阶段讲的是"RL 解决什么问题", 但还有一条被忽视的线: **reinforcement signal 以什么方式驱动学习**。

详见独立文档: `RL/reinforcement_signal_evolution.md` (基于 14 篇原始论文考证)

```
经典 RL 中, reward 同时扮演两种角色:
  评价器: "这个动作好还是坏" (Bellman 只需要这个)
  梯度缩放器: "好多少 → 梯度多大" (policy gradient 绑定了这个)

FM 时代最有意思的变化: 这两种角色开始被拆开
  RL 被前移到 evaluator / relabeling 这边 (value function 做评价)
  actor 训练则后退成条件 SL (advantage 做输入条件, 不做梯度乘子)

两个深层脉络:
  1. Bitter Lesson: dense reward shaping = 人类先验 → 模型够大时成为冗余
  2. 分布雕刻: pre-training 给了粗坯, RL 不再是从零搜索, 而是在已有分布上雕刻
     → 搜索需要方向+步长 (精确 reward)
     → 雕刻只需要保留/去掉 (二值 +/-)
     → pre-training 质量决定了 RL 信号可以多粗糙
```

## 演化总结: RL 的 70 年主线

```
RL 的发展可以用一句话概括: 在越来越少的先验知识下, 解决越来越大的决策问题。

阶段 1 -- 数学根基 (1957-1992):
  Bellman 方程给出最优解的递归结构, 但需要完整环境模型。
  TD Learning 去掉了对模型的依赖, 从经验中学习。
  Q-Learning 进一步去掉了对策略评估的依赖, 直接学最优值函数。
  REINFORCE 开辟了另一条路: 跳过值函数, 直接优化策略本身。
  → 两条路线 (value-based vs policy-based) 为后续 30 年的发展奠基。

阶段 2 -- Deep RL 革命 (2013-2018):
  DQN 用神经网络替代 Q 表格, 让 RL 处理高维感知输入 (像素)。
  PPO 让 policy gradient 训练变得简单稳定, 成为默认算法。
  SAC 用最大熵框架统一了探索与利用, 提高了样本效率。
  → 核心贡献不是新理论, 而是让 1990 年代的理论在深度学习时代真正可用。

阶段 3 -- RL 进入机器人 (2017-2021):
  Domain Randomization + 大规模并行仿真让 sim-to-real 成为可能。
  Teacher-Student distillation 让特权信息可以迁移到可部署传感器。
  → RL 从游戏走向物理世界。

阶段 4 -- FM 时代的 RL (2022-):
  RL 不再是独立的学习算法, 而是更大 pipeline 中的一个阶段:
    RLHF 中 RL 做 LLM 的 post-training 对齐;
    pi*0.6 中 offline RL 做 VLA 的 post-training 提升;
    SONIC 中 PPO 训 universal motion tracker 作为下游任务的基座。
  → RL 的角色从"完整的学习系统"变为"将被动知识转化为主动改进行为的不可替代组件"。
```

## 对你的意义

```
你已经会的:          对应 RL 脉络中的:
  PPO + reward design   阶段 2 (Deep RL) + 阶段 3 (Robotics RL)
  sim-to-real DR        阶段 3 (Domain Randomization)
  motion tracking       阶段 4 (SONIC)

你接下来会用到的:
  offline RL (pi*0.6)   阶段 4 (FM 时代的 RL)
  RLHF 思想             阶段 4 (对齐, 用于未来的人机交互)

理解的关键连接:
  PPO 的 actor = REINFORCE 的 policy gradient 后代
  PPO 的 critic = Q-Learning 的 value estimation 后代
  SAC 的 entropy bonus 与 RLHF 的 KL penalty 是同一类思想: 正则化防策略坍缩
  DQN 的 experience replay 至今是 off-policy 算法的标配
```
