# 强化信号的演化: 从梯度缩放到分布雕刻

基于 14 篇原始论文考证, 追踪 "reinforcement signal 怎么进入学习系统" 这条被忽视的演化线。

---

## 核心论点

RL 中 reinforcement signal 的角色经历了根本性转变:

```
1957-2017: signal 在 loss 中缩放梯度 → 策略空间中搜索最优点
2019-2025: signal 在 input 中标记数据 → 已有分布上雕刻形状

驱动这个转变的是两个力量:
  1. Bitter Lesson: 精心设计的 reward = 人类先验 → 模型够大时成为冗余
  2. Pre-training 改变了 RL 的任务本质: 不再是从零搜索, 而是分布雕刻
```

---

## 1. Bellman 的 value: 排序器、压缩器、梯度缩放器

Reward 在 RL 系统中同时承担三种语义, 理解这三层是理解整条演化线的前提:

```
控制语义 (决策):
  Bellman 的 max_a Q(s,a) → 选最好的动作
  → 只需要排序: a1 比 a2 好就够了, 不需要好多少

估计语义 (bootstrap):
  Critic 学 V(s) 或 Q(s,a) → 用 r + gamma*V(s') 做 TD target
  → 仍然需要数值: V 是长期后果的压缩, 必须是有意义的数字
  → RECAP 的 value function 也用数值做 bootstrap (预测剩余步数)

优化语义 (梯度):
  PPO 的 clip(ratio) * A_t → advantage 的大小直接缩放 policy 梯度
  → 这一步把"数值"绑定成了"梯度幅度"
  → 是 policy gradient 的实现选择, 不是 Bellman 的要求
```

关键判断: **RECAP 解绑了第三层, 但保留了前两层。** Value function 仍然用数值做 bootstrap (估计语义), 但 advantage 二值化后做输入条件 (优化语义被替换为条件建模)。不是"数值不重要", 是"数值不必绑定为梯度幅度"。

```
Reward 不是"真理", 而是 evaluator 与 optimizer 之间的一种接口
  经典 RL: 接口是连续标量 (reward → advantage → gradient multiplier)
  FM 时代: 接口变成条件信号 (preference / binary advantage / text label)
  Bellman 没过时 → 它的职责变成"给数据排优先级"
  怎么把排序注入 policy → 不必拘泥于 A_t * log pi 这一种形式
```

---

## 2. 演化的七个阶段

每个阶段: **碰到什么墙 → 做了什么 → 底层原因**

### 阶段 1-3: reward 大小 = 梯度大小 (1957-2017)

```
Bellman (1957) → TD (1988) → Q-Learning (1989) → REINFORCE (1992) → PPO (2017)

reward 同时承担两个职责:
  (a) 方向: 好还是坏 (sign)
  (b) 幅度: 好多少 / 坏多少 (magnitude)

实践者体感: reward = 0.1*dist + 0.3*contact + 0.5*success
  这些系数是在手动调梯度的相对尺度, 不是物理量

隐含假设: 模型从随机初始化开始, 策略空间中搜索, 需要精确信号引导每一步
```

### 转折: 2019 年, 三条独立路线

三组人同时质疑 "advantage 必须在 loss 里缩放梯度"。共同的工程痛苦: RL 太难用, 能不能用 supervised learning 代替?

### 阶段 4a: AWR (Peng, Kumar, Levine 2019) — advantage 做数据筛选权重

```
墙: PPO 需要 on-policy + log pi; SAC 的 bootstrap 不稳定
做法: advantage 从"梯度乘子"变成"样本权重"
  好数据 → 权重大 → 多学; 坏数据 → 权重≈0 → 忽略
  loss 本身是标准 SL
局限: 坏数据被忽略, 不是被利用 → 只做"选择性模仿"
```

*"AWR can be interpreted as an advantage-weighted form of behavioral cloning"*

### 阶段 4b: Upside-Down RL (Schmidhuber 2019) — reward 从输出变输入

```
墙: 传统 RL 先"预测 reward" (value function) 再导出行动 → 为什么要绕这个弯?
做法: reward 从"网络的预测目标"变成"网络的输入命令"
  训练: "上次 100 步内拿了 50 分" → SL 学复现
  推理: 输入 "100 步内拿 100 分" → 泛化到更好的行为
局限: 没有实验; 推理时需要外推到训练没见过的高值 → 不可靠
```

*"It does not predict rewards at all. Instead it takes rewards as inputs."* (草稿 2017.12)

### 阶段 4c: RCP (Kumar, Peng, Levine 2019) — 所有数据都有用, 标条件即可

```
墙: AWR 忽略坏数据; UDRL 需要外推
做法: 训练条件策略 pi(a|s, Z), 好坏数据等权, 条件 Z 区分好坏
  推理时设 Z=高 → 只输出好动作
关键发现: advantage conditioning >> return conditioning (相对值比绝对值更好泛化)
局限: 仍需外推到更高条件值 → 不可靠
```

*"actions that lead to mediocre returns represent 'optimal' supervision for a mediocre policy"*

作者: AWR 同一组人。RCP 的 pi(a|s,Z) 在 6 年后几乎原样出现在 RECAP 中。

### 阶段 5: Decision Transformer (Chen et al. 2021) — return-to-go 就是 RL 的 prompt

```
墙: RCP 只看当前时间步 (context=1); Transformer 在 NLP/CV 靠长上下文成功
做法: 轨迹 = (return-to-go, state, action) token 序列, GPT 架构生成 action
  RCP = DT 在 K=1 时的特例; DT 用 K=20~50 大幅提升
  Attention 直接做信用分配, 不需要 Bellman 逐步回传
局限: return-to-go 设多少? 没有原则性方法 → 外推问题仍在
```

### 阶段 6: RLHF → DPO (2022-2023) — 独立分支, 偏好直接训策略

```
墙: LLM 的"好/坏"无法 rule-based 定义 → 只能人类比较
RLHF: 偏好 → reward model → PPO (复杂, 4 个模型)
DPO: 数学证明 reward model 是冗余表征 → 偏好直接训策略
  "你的语言模型本身就是 reward model"
局限: DPO 只适用于单步生成, 不适用于多步机器人控制
```

### 阶段 7: CFGRL → RECAP (2025) — 二值标签 + CFG = 有保证的改进

```
墙: AWR 梯度不均匀; RCP/DT 需要外推; PPO 在 flow matching 上算不出 log pi
CFGRL (Frans, Abbeel, Levine 2025):
  图像 CFG 和 RL policy improvement 是同一个数学
  训练: 标 "+/-" 做条件, 30% dropout
  推理: 放大 "positive - 无条件" 差异 = policy improvement
  不需要外推 (训练时 +/- 都见过), 有理论保证 (guidance 越大改进越强)

RECAP / pi*0.6 (PI, Levine 2025):
  CFGRL 的 VLA 工程落地
  "Advantage: positive/negative" 做 text token
  loss = flow matching + cross-entropy (和 SFT 完全一样)
  实验: 同样数据, RECAP 远超 AWR 和 PPO
```

---

## 3. 两个深层脉络

演化表面上是"信号精度降低 + 位置从 loss 移到 input", 但底层有两个更深的力量在驱动。

### 脉络 1: Reward Shaping 是 Bitter Lesson

Rich Sutton (2019) "The Bitter Lesson": 利用人类知识的方法短期有效, 利用计算规模的方法长期碾压。

```
Dense reward shaping = 人类先验注入
  你花一周设计 reward = 0.1*dist + 0.3*contact + 0.5*success
  每个系数都是你的领域知识 → 给小模型的详细导航指令
  → 有效, 但不可扩展 (每个任务重新设计)

二值 reward = 最少人类先验
  success / fail → 零先验
  模型从数据中自己压缩出"什么是好动作"
  → bitter lesson: 你调一周的系数, 大模型用一个二值标签就学会了

RL 中的 Bitter Lesson:
  人类知识: dense reward, reward model, 手工 value function
  计算规模: 大模型 + 大数据 + 极简信号

这也解释了 DPO 的 insight:
  reward model 是人类先验的载体 (把模糊偏好量化为精确分数)
  DPO 证明这个量化是冗余的 → reward model 是 bitter lesson 的又一例
```

但这里有一个重要限定:

```
Bitter Lesson 成立的前提: 模型已经从 pre-training 中获得了足够的基础能力

  大模型 + 从零训 (随机初始化 PPO): 二值 reward 不够 → 仍需 dense reward
  大模型 + 有 pre-training: 二值 reward 足够 → bitter lesson 成立

  → 不是"模型大就不需要精确信号"
  → 是"模型大 + pre-training 好就不需要精确信号"
  → pre-training 的质量决定了 RL 信号可以多粗糙
```

### 脉络 2: RL Post-training 不是搜索, 是分布雕刻

这是理解 RECAP 范式转变的关键。

```
传统 RL (PPO) 的心理模型:
  策略从随机初始化开始
  reward 告诉我方向和步长
  我在策略空间中一步步搜索最优点
  → "优化"范式: 搜索 → 需要方向 + 步长 → 需要精确 reward

RECAP 的心理模型:
  VLA 已经从海量 demo 数据中 pre-train
  → 模型已经有一个巨大的行为分布, 包含了好动作和坏动作
  → RL post-training 不是"从零搜索好策略"
  → 而是"在已有分布上雕刻 — 保留好的, 去掉坏的"
  → "分布雕刻"范式: 筛选 → 只需要好/坏标签 → 二值足够

类比:
  PPO = 在空白画布上画画 → 需要精确的笔触指令 (reward magnitude)
  RECAP = 在大理石粗坯上雕刻 → 只需要标记"保留/去掉" (binary label)
  粗坯 = pre-training 分布 (已包含答案, 但不够精确)
```

**这解释了为什么 RECAP 不存在"步长"概念**:

```
PPO: "我在哪 + 往哪走 + 走多远" → 优化问题 → 需要方向 + 步长
RECAP: "好的长什么样 + 坏的长什么样" → 建模问题 → 不存在步长

RECAP 学的是两个条件分布:
  P(action | state, "+") → 好动作的统计 pattern
  P(action | state, "-") → 坏动作的统计 pattern
  推理时从 P(action | state, "+") 采样 → 直接生成好动作
  
  不是"把当前策略往好方向挪一点"
  而是"直接从好动作的分布中生成"
  "调多少"这个问题消失了 → 取而代之的是"两个条件分布的统计距离"
  这个距离是数据本身决定的, 不需要 reward magnitude 来指定
```

**Negative 数据的角色 — 定义"雕掉的部分"**:

```
只有 positive: 模型知道"好动作大概长这样", 但不知道好坏的边界
有 positive + negative: 模型学到好和坏的精确边界 → P(a|s, "+") 更尖锐

更重要的是 CFG 推理时:
  output = unconditional + w × (positive - unconditional)
  unconditional = positive 和 negative 的混合
  没有 negative → unconditional ≈ positive → guidance 差异 ≈ 0 → 没效果
  有 negative → unconditional 更宽 → guidance 有清晰方向

  → negative 数据定义了"要雕掉的部分", 没有它不知道往哪雕
  → pre-train = 粗坯, positive = 保留区域, negative = 去除区域, CFG = 雕刻刀
```

**分布雕刻也解释了 RECAP 的失效条件**:

```
如果 pre-training 分布中根本没有"正确叠衣服"这个模式:
  → 再怎么雕刻也雕不出来
  → 分布中没有的东西, 筛选不出来
  → 这就是 pi*0.6 论文中 pre-training 占大量篇幅的原因:
     pre-train 的质量 = 粗坯的质量 = 决定了最终能雕多好
```

### 两个脉络的交汇

```
脉络 1: 大模型 + pre-training → 精确 reward 变得冗余 (bitter lesson)
脉络 2: 有了 pre-training → RL 从搜索变成雕刻 → 只需要好/坏标签

交汇点: 为什么二值信号对大模型足够?
  不是因为"模型大就能从粗糙信号中学"(这只是表面)
  而是因为"模型的 pre-training 分布已经包含了答案"
  RL 不是在教模型新东西 → 是在帮模型筛选已经知道的东西
  筛选只需要: "留下 / 去掉" = 二值
  搜索才需要: "往哪走 / 走多远" = 连续梯度
```

---

## 4. 和 DAgger 的关系: RECAP 是泛化版实机 DAgger

```
DAgger (Ross 2011):
  循环: 用策略 rollout → 专家在每个 state 标注正确动作 → SL 重训 → 重复
  信号: action 级纠正 ("你应该这样做")
  局限: 需要专家逐帧在场; 只有正面数据 (专家给的都是对的)

RECAP:
  循环: 用策略 rollout → value function 标注 +/- → 条件 SL 重训 → 重复
  信号: trajectory 级评价 ("这段比平均好/差")
  优势: 不需要专家逐帧; 正面+负面数据都参与训练

pi*0.6 实际上两个都用了:
  数据来源 1: 自主 rollout + success/fail 标注 → RECAP
  数据来源 2: 人类在旁看, 出错时接管纠正 → human-gated DAgger
  → RECAP 包含 DAgger 作为数据来源之一

DAgger 是 RECAP 的特例:
  所有纠正数据标 positive + 没有 negative 数据 + 没有自主 rollout
  → 退化成 DAgger
```

---

## 5. 殊途同归: 灵巧手和 FM 撞到同一面墙

QiHaoZhi 的灵巧手工作和 PI 的 VLA 工作在不同规模上独立发现了同一个 pattern。

```
QiHaoZhi 路线 (单任务极致难度):
  HORA (2022): sim PPO + RMA → zero-shot sim2real → 粗圆柱体成功
  PenSpin (2024): 笔旋转 → RMA 失败, DAgger 失败, 纯 BC 失败
    → 解法: oracle rollout → 筛选成功轨迹 (45条) → BC fine-tune
    → 本质: oracle 的行为分布中包含答案, 筛选成功 = 雕刻分布

PI 路线 (多任务泛化):
  pi_0/0.5 (2024): 大量 demo pre-train → VLA 能做但不够好
  pi*0.6 RECAP (2025): 真机 rollout → +/- 标注 → 条件 SL fine-tune
    → 本质: pre-trained 分布中包含答案, +/- 标注 = 雕刻分布

结构同构:
  Step 1: 造不完美 oracle
    QiHaoZhi: sim PPO + 特权信息 → 仿真中 work 但不能直接部署
    PI: BC on 海量 demo → 真机能做但失败率高
  
  Step 2: 从 oracle 分布中雕刻可部署策略
    QiHaoZhi: 开环回放 → 筛选成功 (rejection sampling) → BC
    PI: 真机 rollout → value function → +/- (advantage conditioning) → conditional SL
  
  共同发现: RL 单独做不到 (探索效率太低), BC 单独做不到 (被 demo 锁死)
  解法: transfer + distribution shaping
    → 先造粗坯 (oracle/pre-trained distribution)
    → 再用少量目标域数据雕刻 (成功筛选 / +/- 条件化)
```

PenSpin 的 45 条成功轨迹筛选 = RECAP 的 "Advantage: positive" 子集。

区别: PenSpin 只用 positive (成功) 数据; RECAP 用 positive + negative。RECAP 更高效的原因见 Section 3 "Negative 数据的角色"。

详细对比见: `TransferLearning_Origins/note.md` "迁移的本质: 从 Mapping 到 Distribution Shaping"

---

## 6. 思想血脉图

```
Bellman (1957) ←── 精确计算, reward 是精确输入
    ↓
TD/Q-Learning (1988-89) ←── reward 进入 delta 更新 (on-policy value 路线)
    ↓                                   ↓
REINFORCE (1992)              SAC (Haarnoja, Abbeel, Levine 2018)
    ↓                           off-policy, Q 网络学 value
PPO+GAE (2017)                  不需要 pi_old 密度
    ↓                           但需要 log pi + 高维 action 难学
    │
    ├── AWR (Peng, Kumar, Levine 2019)
    │     advantage → 样本权重; SL 子程序
    │
    ├── Upside-Down RL (Schmidhuber 2019, 草稿 2017)
    │     reward → input 命令; 范式宣言
    │
    └── RCP (Kumar, Peng, Levine 2019) ← AWR 同组, 合流 UDRL 思想
          advantage → 输入条件; 所有数据等权
              ↓
        Decision Transformer (Chen, Abbeel 2021)
          RCP + Transformer 长上下文; return-to-go = prompt
              ↓
        CFGRL (Frans, Abbeel, Levine 2025) ← 同实验室
          advantage conditioning = CFG; 解决外推; 理论保证
              ↓
        RECAP / pi*0.6 (PI, Levine 2025) ← 同一人
          CFGRL → VLA text token; 二值 +/-; 叠衣/咖啡/组装箱子

独立分支: RLHF (2022) → DPO (2023)
  偏好信号; reward 被隐式化; 限于单步生成, 未进入 robotics
```

**关键人物**: Sergey Levine 贯穿 AWR → RCP → CFGRL → RECAP, 持续 6 年推进 "用 SL 做 RL"。Pieter Abbeel 贯穿 SAC → DT → CFGRL, 从 off-policy value 路线走向条件化路线。

---

## 7. 回到原点: Thorndike 的猫 (有修正)

```
1898 年, Thorndike 的猫:
  二值信号 (吃到鱼 / 没吃到) + 条件性行为强化
  → 和 RECAP 在抽象层面同构

但有一个重要区别:
  猫没有 pre-training → 猫是从零搜索 → 需要大量试错
  VLA 有 pre-training → VLA 是分布雕刻 → 只需要少量 +/- 标签

  猫更像 PPO (从随机开始搜索)
  VLA 更像 RECAP (在已有知识上筛选)

  共同点: 二值信号足够 (好/坏)
  不同点: 猫用二值信号做搜索 (慢), VLA 用二值信号做雕刻 (快)
  → 二值信号 + 大规模 pre-training 才是 RECAP 真正 work 的原因, 不只是二值信号本身
```

---

## 8. 对 RL 实践者的意义

```
你当前的工作 (PPO sim2real 灵巧手):
  模型从随机初始化开始 → 你在做搜索 → dense reward 是最高效的
  reward shaping = 用人类知识压缩学习信号 → 加速收敛
  → 没有 pre-training → 不适用 bitter lesson → dense reward 合理

转型方向:
  当你的灵巧手有了 pre-trained VLA (比如在大量 demo 上训):
  → 分布中已经包含了"大致正确的操作"
  → dense reward 变成冗余 → 二值信号 + 分布雕刻就够 (见 Section 3 限定条件)
  → 角色转变:
     从 "设计 reward function" → "构造好的 pre-training 分布 + 设计 data pipeline"
     从 "调 reward coefficients" → "调数据配比和 advantage threshold"
```

### 以后看 RL 方法的三个诊断问题

```
(a) 谁在做评价?
    手工 reward / learned value / preference model / VLM judge?

(b) 评价结果怎么进入 policy?
    梯度乘子 (PPO) / 样本权重 (AWR) / 偏好对 (DPO) / 条件 token (RECAP)?

(c) policy 改进需要最少多少信息?
    连续 advantage / 二值好坏 / pairwise better-worse / 自然语言 rubric?
```

---

## 9. 开放问题

```
1. 二值 advantage 是信息论下限吗?
   RECAP 用 "+/-" work 了, 但连续 advantage 是否带来额外增益?
   → 可能 "+/-" 丢失了有用信息, 被大模型容量补偿了

2. 什么时候二值不够?
   精细技能 (灵巧手上螺丝, <1mm 精度) 可能需要更细粒度信号
   RECAP 目前只在粗粒度长时间任务上验证 (叠衣, 咖啡)

3. FPO: 让 flow matching 兼容 policy gradient 的另一条路
   近似 flow matching 的 log pi → PPO 可直接用 → 不需绕道 conditioning
   2025 最新工作, 还没在 VLA 规模上验证

4. negative 数据的贡献有多大?
   AWR 忽略坏数据; RECAP 用 "-" 条件学"坏动作长什么样"
   → 这是 RECAP 性能优势的核心来源吗? 论文没单独消融

5. pre-training 分布中没有的能力, RECAP 能雕出来吗?
   理论上不能 (分布中没有的东西筛选不出来)
   但 CFG w>1 时会产生"超出训练分布"的行为
   → 这算外推还是分布内的极端采样? 边界不清晰
```

---

## References

| # | Work | Year | Signal | Location | arxiv |
|---|------|------|--------|----------|-------|
| 1 | Thorndike, Law of Effect | 1898 | Binary | Behavioral conditioning | -- |
| 2 | Bellman, Dynamic Programming | 1957 | Exact R | Exact computation | Princeton |
| 3 | Sutton, TD Learning | 1988 | Scalar r | Value delta | ML 3(1) |
| 4 | Watkins, Q-Learning | 1989 | Scalar r | Q-table | PhD thesis |
| 5 | Williams, REINFORCE | 1992 | Return G | Gradient multiplier | ML 8(3-4) |
| 6 | Schulman, PPO+GAE | 2017 | Continuous A | Loss multiplier | 1707.06347 |
| 7 | Haarnoja+Abbeel+Levine, SAC | 2018 | Reward→Q | Q gradient to Actor | 1801.01290 |
| 8 | Peng+Kumar+Levine, AWR | 2019 | Continuous A | Sample weight | 1910.00177 |
| 9 | Schmidhuber, UDRL | 2019 | Return R | Input command | 1912.02875 |
| 10 | Kumar+Peng+Levine, RCP | 2019 | Advantage | Input condition | 1912.13465 |
| 11 | Chen+Abbeel, Decision Transformer | 2021 | Return-to-go | Input token | 2106.01345 |
| 12 | Ouyang, InstructGPT/RLHF | 2022 | Preference | Learned reward→PPO | 2203.02155 |
| 13 | Rafailov, DPO | 2023 | Preference | Implicit in loss | 2305.18290 |
| 14 | Frans+Abbeel+Levine, CFGRL | 2025 | Binary A | Input condition+CFG | 2505.23458 |
| 15 | PI+Levine, RECAP/pi\*0.6 | 2025 | Binary +/- | Input text token | pi.website |

### 论文文件位置

| Paper | Path |
|-------|------|
| Bellman 1952-1957 | `RL/57_BellmanEquation/` |
| Q-Learning 1992 | `RL/89_QLearning/` |
| REINFORCE 1992 | `RL/92_REINFORCE/` |
| DQN 2013 | `RL/15_DQN/` |
| PPO 2017 | `RL/17_PPO/` |
| SAC 2018 | `RL/18_SAC/` |
| AWR 2019 | `RL/signal_evolution_refs/19_AWR/` |
| Upside-Down RL 2019 | `RL/signal_evolution_refs/19_UpsideDownRL/` |
| RCP 2019 | `RL/signal_evolution_refs/19_RewardConditionedPolicies/` |
| Decision Transformer 2021 | `RL/signal_evolution_refs/21_DecisionTransformer/` |
| DPO 2023 | `RL/signal_evolution_refs/23_DPO/` |
| CFGRL 2025 | `RL/signal_evolution_refs/25_CFGRL/` |
| RECAP / pi\*0.6 | `robotics/families/pi_Series/25_pi06/` |
