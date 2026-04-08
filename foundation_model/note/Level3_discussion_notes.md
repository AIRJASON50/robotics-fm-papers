# Level 3 讨论笔记：Robot FM 的完整系统

基于 CS2Robotics_Roadmap Level 3 学习过程中的讨论整理。

---

## 1. 从动作库到端到端: 为什么放弃 rule-based?

### SayCan (2022) 的实际困境

```
SayCan 的架构:
  LLM 做规划 → 从 551 个预写 skill 中选择 → 执行
  skill = 人工写好的控制器 (pick, place, open, close...)

三个根本瓶颈:
  1. 新任务 = 新 skill = 人工编程 (551 个已经极限, 真实世界需要无限多)
  2. skill 之间没有连续过渡 (硬编码切换, 不自然)
  3. skill 对物体形状敏感 (换个杯子就失败, 每种物体要调一遍)
```

### RT-1 的动机 (来自论文原文)

RT-1 论文 Introduction 明确说了从"一任务一模型"到"一个通用模型"的转变动机:

**论文原文 (Section 1)**:
> "End-to-end robotic learning typically involves collecting TASK-SPECIFIC data... 
> This workflow mirrors the classic approach to supervised learning in other domains... 
> where task-specific datasets would be collected to solve INDIVIDUAL tasks, 
> with LITTLE INTERPLAY between the tasks themselves."

> "Recent years have seen a transformation in vision, NLP, and other domains, 
> away from siloed, small-scale datasets towards LARGE, GENERAL models 
> pre-trained on BROAD datasets."

作者的逻辑链:
1. NLP/CV 已经从"一任务一模型"变成"一个通用模型" (GPT, ViT)
2. 机器人还在"每个任务独立" (SayCan 的 551 个 skill)
3. 如果有足够多的演示数据 → Transformer 能不能学会所有 skill?
4. Google 恰好有 130k episodes × 700+ tasks → 试试

**推动力不是理论, 而是工程痛苦**: 手写 551 个 skill 需要一年, 训一个 RT-1 需要几天。

### RT-1 的训练方式: 纯 BC (行为克隆)

**论文原文 (Section 3)**:
> "We learn π using behavioral cloning, which optimizes π by minimizing 
> the negative log-likelihood of actions a_t given the images and language instructions."

就是模仿学习: 网络预测的动作 vs 真实动作的距离 (交叉熵 loss, 因为动作被离散化为 token)。不是 RL。

### 组合泛化 vs 涌现

RT-1 论文对"泛化"的描述:

**论文原文 (Section 1)**:
> "the model can discover the patterns between structurally similar tasks 
> and perform NEW TASKS that COMBINE those patterns in NOVEL WAYS"

这更准确叫**组合泛化 (compositional generalization)**, 不完全是 LLM 意义上的涌现:

```
RT-1 的组合泛化:
  见过 "pick apple" + 见过 "place in bowl"
  → 能做 "pick apple and place in bowl" (没见过这个组合)
  → 是已有能力的重组, 不是全新能力

RT-2 更接近涌现:
  robot 数据没有 "恐龙属于哪个大陆"
  VLM 数据没有 "怎么拿起物体"
  → 两者结合后能做 "把恐龙放到正确大陆" (两个数据源都没有的能力)
```

### "涌现"的权威定义 (Wei et al. 2022)

> "An ability is EMERGENT if it is not present in smaller models 
> but is present in larger models."

涌现 = 小模型没有, 大模型才有, 在规模到某个阈值后突然出现。
GPT-3 的 in-context learning 是典型: GPT-2 (1.5B) 不会, GPT-3 (175B) 突然会了。

机器人领域目前观察到的更多是**组合泛化** (已有能力重组) 而非严格的涌现 (阈值后突然出现新能力)。但 RT-2 中 VLM 知识和 robot 能力的交叉可能是真正的涌现。

---

## 核心认知框架: NN、表征与学习的本质

```
NN 的作用: 表征和压缩

在有结构/规律/pattern 的数据中:
  好的表征和压缩 = 找到数据中的规律从而理解数据

人为定义的 loss 如果和 pattern 的方向一致:
  → loss 低的权重配置恰好编码了 pattern → 获得好的表征

因此一个好的 model 需要三个条件:
  1. 结构化的数据 (数据中有可被提取的共性/规律)
  2. 足够的参数量 (能区分精细的 pattern, 但不能多到逐条记忆)
  3. 符合规律和结构的 loss (引导梯度下降的方向和 pattern 一致)

注意: NN 不"追求"理解, 它只追求 loss 低
      但 loss 低的唯一途径恰好是理解 (压缩出 pattern)
      理解是结果, 不是目标
```

这个框架贯穿了所有学过的内容:
- AE: loss = 重建误差 → 迫使 NN 压缩出好表征
- GPT: loss = next-token prediction → 迫使 NN 压缩出语言 pattern
- CLIP: loss = 图文对比 → 迫使 NN 压缩出视觉语言对齐
- DDPM: loss = 预测噪声 MSE → 迫使 NN 压缩出数据流形的结构
- RT-1: loss = 预测动作 token → 迫使 NN 压缩出动作 pattern (组合泛化)

Scaling Law 的本质也是这个:
- 参数少: 只能压缩成粗 pattern ("所有抓取差不多")
- 参数多: 能压缩成精细 pattern ("硬物用力, 软物轻柔")
- 数据少: pattern 不够丰富, 组合空间小
- 数据多且有结构: pattern 丰富, 组合泛化能力强
- Chinchilla: 参数和数据要匹配 → 压缩效率最优

### 学习失败的两种情况

```
Spurious Correlation (虚假相关):
  数据中存在捷径, 模型学了不该学的 pattern
  例: 训练集中红色杯子总在左边 → 模型学了"红色=左边"
  → 是 pattern 层面的问题, 数据中的虚假相关导致的
  → 解法: 消除虚假相关 (DR, 数据平衡, 消融验证)

Distribution Shift (分布漂移):
  训练分布 vs 测试分布不一致
  例: 仿真训练, 真机测试 → 图像质感完全不同
  → 是分布层面的问题, 即使 pattern 正确也会失效
  → 解法: 扩大训练分布 (DR, 更多数据)

两者都是数据问题, 不是 NN 的问题。
```

Domain Randomization 同时解决两者:
- 随机化背景/光照 → 破坏虚假 pattern (不让模型学"特定背景=特定任务")
- 随机化物理参数 → 扩大训练分布 (覆盖真机的参数范围)

### 对机器人工程师的启示

```
架构: 几乎收敛 (Transformer), 不需要发明新架构
Loss: 大多是对齐真值 (BC / MSE / 交叉熵), 不需要精心设计
数据: 真正的 craft — 什么该包含, 什么该排除, 怎么保证 pattern 干净

工程师的角色从"训练 NN"变成:
  1. 构造好的数据 (多样性, 排除捷径, DR)
  2. 选对的 backbone (预训练任务和数据决定了表征质量)
  3. 定义对的 fine-tune 策略 (LoRA/全参数, 冻结/解冻哪些模块)

一句话: 学习的本质是压缩, 压缩的质量取决于数据的结构,
       工程师的工作是确保数据的 pattern 和你期望的方向一致。
```

---

## 2. 端到端训练的本质: 数据即先验

用户的关键理解 (已验证):

```
1. 大量真实机器人数据 (130k episodes) 本身就是"正确完成工作的先验"
2. 这些数据不需要预处理就蕴含了正确的动作能力
3. 定义 loss = 预测动作 vs 真实动作的距离 → 模仿学习
4. 只要 loss 低 → 在相同 obs 下做出一样的反应
5. 数据规模大 → 数据中的规律被模型理解 → 组合泛化

RT-1 就是这个思路的直接实现:
  数据: 130k 条遥操作轨迹, 700+ 种任务
  模型: Transformer (35M 参数)
  训练: BC (行为克隆, loss = 交叉熵)
  结果: 97% 训练任务成功率 + 组合泛化到新任务
```

这个思路和 LLM 完全对应:
- GPT: 互联网文本作为先验, next-token prediction 作为 loss → 学会语言
- RT-1: 遥操作轨迹作为先验, action token prediction 作为 loss → 学会操作

---

## 3. 从 VLM 输出头到 Action Head: 为什么需要换头

### RT-2 的离散化问题根源

RT-2 把动作离散化为 256 bins 不是设计选择, 是**架构限制**:

```
VLM 的输出头:
  hidden state → Linear(hidden_dim, vocab_size) → softmax → 离散 token
  → 这是分类问题: 从 ~50000 个词中选一个
  → 不能输出 0.3742 这种连续浮点数
  → 动作只能编码成词表中的某个 token (0-255 的 bin index)

精度:
  range [-1, 1] / 256 bins ≈ 0.0078 per bin
  对关节角 [-pi, pi]: 约 2.8° 误差
  → 桌面抓放够用, 灵巧操控不够
```

### 为什么不直接输出连续值

```
要连续输出 → 必须换输出头:
  VLM 原生: hidden → Linear → softmax → 离散 (分类)
  回归头:    hidden → Linear(hidden_dim, action_dim) → 连续浮点数 (回归)

  RT-2 不想改 VLM → 复用语言输出头 → 被迫离散化
  pi_0 加了 Action Expert → 可以输出连续值 → 架构变复杂但精度解决
```

### 连续输出不等于需要生成模型

```
简单回归头:   hidden → Linear → 一个动作向量 (连续, 但输出均值)
  问题: 同一观测下有多种合理动作时, 回归取平均 → 灾难

Flow Matching: hidden → 迭代去噪 → 一个动作向量 (连续, 能采样不同模态)
  解决: 多模态动作分布 (抓杯子可以从左/右/上)

→ 连续输出本身用回归头就行
→ Flow Matching 解决的是"多模态"问题, 不只是"连续"问题
```

### Action Head 的设计需求

```
输入: VLM 的压缩表征 (spatial tokens, 包含场景+指令理解)
输出: 连续动作向量 (关节角/末端位姿)
要求:
  1. 接收 VLM 的表征 → 需要和 VLM 有交互机制 (attention)
  2. 输出连续值 → 不能用 softmax 分类头
  3. 处理多模态动作 → 需要生成模型 (diffusion/flow matching) 而非纯回归
  4. 训练时不破坏 VLM → 参数独立 (MoE 式分离)
  5. 推理够快 → flow matching 10 步 > DDPM 1000 步
```

### pi_0 的完整输入输出

```
输入 (全部变成 token 丢进同一个 Transformer):
  图像: 2-3 个相机 RGB → SigLIP ViT 编码 → spatial tokens
  语言: 指令文字 → Gemma tokenizer → text tokens
  本体感觉: 关节角 q_t → 线性投影 → state token
  噪声动作: 随机噪声 A^τ → 线性投影 → action tokens (flow matching 的起点)

输出:
  action chunk: 50 步动作, 每步 7-20 维连续向量
  → 经过 10 次 forward pass (flow matching 10 步去噪) 生成

推理时 KV cache:
  image/text/state tokens 在 10 步中不变 → 缓存, VLM 只算 1 次
  action tokens 每步变化 → Action Expert 算 10 次
  → VLM 不是推理瓶颈
```

### pi_0 的 Expert 分离: 不只是 FFN, QKV 也分开

```
每一层 Transformer:
  Attention: 所有 token 一起算 score (跨模态交互)
    但 QKV 权重分两组:
      W_Q_vlm, W_K_vlm, W_V_vlm → VLM tokens 用
      W_Q_act, W_K_act, W_V_act → Action tokens 用
  
  FFN: 分两组 (并行, 不是串联)
    VLM FFN:    image/text tokens → VLM 的 FFN
    Action FFN: action tokens → Action Expert 的 FFN

  冻结 VLM 时:
    VLM 的 QKV + FFN → 全部不动
    Action Expert 的 QKV + FFN → 全部更新
    → 梯度只流过 Action Expert, 不动 VLM
```

### 用户 insight: 多模态 = 多组 Expert

"N 个模态就有 N 组 QKV + FFN" — 这是 MoE 思想的本质:
- 不同模态的数据有不同的 pattern → 用不同的权重压缩各自的 pattern
- 通过 Attention 共享信息 (跨组 QKV 做 attention score)
- VLM 的 image + text + proprioception 共用一组是因为经过 PaliGemma 预训练已在同一表征空间
- Action 是全新模态 (VLM 没见过) → 需要独立一组
- 如果加触觉 → 可能需要第三组

---

## 4. pi_0.5: 泛化不是靠一个技巧, 而是系统性数据工程

pi_0.5 之前被简化为"只有 Knowledge Insulation"——实际上是五个层面的系统级改进:

```
1. 异构数据共训练: 97.6% 数据不是目标机器人的
   (其他机器人 + web 数据 + 语义预测 + 口头指令)
2. 两阶段训练: pre-train 用离散 token, post-train 切 flow matching
3. 同一模型层级推理: 先预测子任务 (ℓ̂), 再根据子任务出动作 (chain-of-thought 式)
4. Knowledge Insulation: 底层冻结 + 数据混合双重保护
5. 人类口头监督: 比遥操作更容易的数据采集方式

核心教训: 泛化 = 数据工程, 不是算法创新
  → 97.6% 非目标数据能帮到目标任务, 因为视觉语言理解是共享的
  → 和 Open X 的逻辑一致: 动作不通用, 但视觉语言通用
```
