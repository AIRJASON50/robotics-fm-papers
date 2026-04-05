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

## 发展脉络 (Milestone Only)

```
=== 理论根基 (1957-1992) ===
(详细描述见上方"两条线的合流"章节, 这里只做索引)

1957  Bellman 方程                                   → RL/57_BellmanEquation/
      序贯决策的递归求解, RL 的数学基础。需要完整环境模型。

1988  TD Learning (Sutton)
      从"下一步的估计"修正"当前的估计", 不需要环境模型, 不必等最终结果。
      心理学 (试错学习) 和数学 (Bellman 递归) 的合流点。

1989  Q-Learning (Watkins)                           → RL/89_QLearning/
      在 TD 基础上学 Q(s,a) 而非 V(s), 直接选最优动作, 完全不需要模型。
      第一个实用的 model-free RL 算法。局限: 表格, 放不下高维状态。

1992  REINFORCE (Williams)                           → RL/92_REINFORCE/
      跳过 value function, 直接优化 policy。天然支持连续动作空间。
      开创 policy gradient 路线 (PPO 的直系祖先)。局限: 梯度方差大。

1997  TD-Gammon (Tesauro)
      NN + TD Learning 下西洋双陆棋达到世界冠军水平。
      比 DQN 早 16 年, 但当时无法推广 (NN 太小, 无 GPU)。

至此两条技术路线成型:
  Value-based: Bellman → TD → Q-Learning (离散动作, 表格)
  Policy-based: REINFORCE (连续动作, 但不稳定)
  → 1990s-2000s 两条路线各自发展, 在 Actor-Critic 中汇合

### 从 Bellman 到 PPO 始终在做同一件事

```
RL 的 70 年, 核心从来没变:

  state → 估 value → 用 value 决策

变化的只有两件事:
  1. state 从什么变成了什么:
     Bellman: state = index (迷宫第几格, 棋盘第几种局面)
     Q-Learning: state = 同样是 index (表格的行)
     DQN: state = 高维感知数据 (像素 210×160×3)
     PPO: state = 连续向量 (关节角 + 物体位姿)
     
  2. value 怎么估:
     Bellman: 精确计算 (需要完整环境模型)
     Monte Carlo: 跑完整条路, 用真实终点分数 (不需要模型, 但慢)
     TD: 走一步, 用旧估计代替未来 (快, 但一开始不准)
     Q-Learning: 表格存每个 (state, action) 的 value (实用, 但装不下)
     DQN: NN 压缩 Q 表格 (高维也能估)
     PPO: Critic NN 估 V(s), Actor NN 用 V 信号优化策略

Watkins Q-Learning 原文 (1992, p.281) 的描述:
  "the agent observes its current state x_n,
   selects and performs an action a_n,
   observes the subsequent state y_n,
   receives an immediate payoff r_n"
  → 从一开始就是: 观测 state → 决策 → 执行 → 获得反馈

Q-Learning 的 state 只是 index (表格的行索引), 没有"内容":
  state 本身没有结构 → 只是编号 → 所有信息在 value 表里
  到 DQN 时 state 变成像素 → state 本身有丰富 pattern
  → NN 的作用 = 从高维 state 中压缩出 value
  → 和 ViT/CLIP 做的事本质相同: 从数据中压缩出有用的映射

State 和 Value 从来就是两个东西:
  state = "我在哪" (索引/位置/感知)
  value = "在这里值多少分" (评分)
  表格时代看起来混在一起, 是因为 state 太简单 (只是个编号)
  NN 时代区别变明显: state 是高维输入, value 是标量输出
```

=== 从表格到神经网络, 从离散到连续 (2013-2018) ===

2013  DQN (Mnih et al., DeepMind)                    → foundations/15_DQN/
      来源: 2013 NIPS Workshop 论文; 2015 Nature 正式版 (引用 30,000+)
      问题: Q-Learning 表格放不下高维状态 (Atari: 210x160x3 像素)
      贡献: CNN 近似 Q 函数 + 两个关键工程技巧:
            (1) Experience Replay: 打破数据时序相关性, 一条经验可多次学习
            (2) Target Network: 定期同步参数, 避免"自己追自己"的训练发散
            用完全相同的网络和超参数在 7 个 Atari 游戏上, 6 个超越此前方法, 3 个超越人类
      意义: 第一次 NN + RL 大规模成功, 引爆 Deep RL 领域
      局限: 输出所有离散动作的 Q 值 → 无法处理连续动作空间 (机器人需要)

2017  PPO (Schulman et al., OpenAI)                   → foundations/17_PPO/
      问题: Policy gradient 更新步长太大策略会崩溃;
            TRPO (2015) 用 KL 约束保证稳定, 但需要二阶优化, 实现复杂
      贡献: 计算新旧策略概率比 r_t = pi_new/pi_old, 直接 clip 到 [1-eps, 1+eps]
            取 clipped 和 unclipped 的 min → 悲观下界, 只允许保守改进
            同一批数据可跑多个 epoch (通常 3-10), 大幅提高利用率
      意义: "简单 + 鲁棒 + 可并行" → 成为 RL 默认算法
            RLHF (ChatGPT) 用 PPO; IsaacGym 机器人训练几乎全部使用 PPO

2018  SAC (Haarnoja et al., Berkeley)                 → foundations/18_SAC/
      问题: On-policy (PPO) 数据利用率低, 需要大规模并行;
            Off-policy (DDPG) 用确定性策略, 探索不足, 训练不稳定
      贡献: Off-policy + 最大熵: J(pi) = sum E[r + alpha * H(pi)]
            不仅最大化奖励, 还要"尽可能随机地行动" → 探索与利用自动平衡
            双 Q 网络 (取 min 防过高估计), 随机策略 (输出高斯分布), 自动 temperature
      意义: 连续控制任务的样本效率最优选择之一; 真机在线 RL 首选

=== RL 进入机器人 (2017-2021) ===

2017  Domain Randomization (Tobin et al.)
      问题: 仿真训练的策略到真机就崩 (sim-real gap)
      贡献: 随机化仿真的视觉/物理参数 → 策略对 gap 鲁棒
      意义: sim-to-real 的标准方法, 至今仍是基线

2019  OpenAI Rubik's Cube (OpenAI)
      问题: 能不能纯仿真训练解决真实灵巧操控?
      贡献: 大规模 DR + PPO → 真机单手解魔方
      意义: 证明 sim-to-real 灵巧操控可行

2021  RMA (Kumar et al., Berkeley)
      问题: DR 需要猜参数范围, 不够精确
      贡献: Teacher 用特权信息 → Student 用历史观测推断环境参数
      意义: Teacher-Student distillation 成为 sim-to-real 标准范式

=== FM 时代的 RL (2022-) ===

2022  RLHF / InstructGPT (Ouyang et al., OpenAI)
      RL 的新角色: 不训策略, 而是做 post-training 对齐
      RL 从 "training" 退到 "fine-tuning"

2025  OmniReset (UW + NVIDIA)
      RL 的新用法: diverse resets + sparse reward + 大规模并行
      证明 RL + scale 在 manipulation 中也有涌现行为

2025  SONIC (NVIDIA)
      RL 的新定位: PPO 训 universal motion tracker
      不是学技能, 而是学"追踪任意参考动作"

2025  pi*0.6 (Physical Intelligence)
      RL 的最新角色: offline RL 做 VLA post-training
      BC 只用成功 demo → offline RL 用所有数据 (含失败)
```

## "强化"和"学习"分别指什么

```
Reinforcement (强化) = R = 环境返回的标量评价信号

  Williams 1992 原文 (p.231):
    "The evaluation consists of the scalar REINFORCEMENT SIGNAL r,
     which we assume is broadcast to all units in the network.
     At this point each unit performs an appropriate modification
     of its weights."

  → reinforcement 在论文中就是那个数字 r
  → 不是抽象概念, 是具体的评分信号

R 和 V 的关系:
  R: 环境给的即时评分 (外部信号, 每步或终点才有)
  V: "从这个状态开始的预期总 R" (内部估计, agent 自己维护)
  V 是对未来所有 R 的压缩/总结

Learning (学习) = 用 R 更新内部表征使决策改进

  Bellman (1957): 有 R 的概念, 但 V 是精确计算的, 不是学的
    → 这是 optimization/planning, 不是 learning
    → 给定完整环境模型 → 算出 V → 不需要试错

  TD/Q-Learning (1988-1989): V 从实际拿到的 R 中估计
    → 必须试错 → 每次拿到 R → 更新 V/Q → 逐步逼近
    → 这才是 learning: 权重在变 = 在学

  "强化学习" = 用环境的评分信号 (R) 通过试错来学习 (更新 V/policy)
  → Bellman 有"强化"(R) 但没有"学习"(V 是算的)
  → RL 两者都有: 从实际的 R 中学习 V
```

---

## 贯穿 70 年的核心思想

```
RL 从 Bellman 到 PPO, 核心思想始终是同一个:

  "给每个状态算一个 value, 用 value 指导当前决策, 而不是穷尽到最后一步"

  Bellman (1957):  环境已知 → 精确算 value → 查表决策
  TD (1988):       环境未知 → 从经验中近似 value → 边走边更新
  Q-Learning (1989): 估 Q(s,a) = 每个动作的 value → 不需要环境模型
  DQN (2013):      用 NN 估 Q → 高维状态也能估
  PPO (2017):      Critic (NN) 估 V(s) + Actor (NN) 用 V 信号优化策略
                   → Actor-Critic = Bellman 的 "先算 value 再决策" 的 NN 版本

  变化的是: "value 怎么算" (精确→近似→NN)
  不变的是: "用 value 把全局问题变成局部决策"
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
