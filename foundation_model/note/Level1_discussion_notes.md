# Level 1 讨论笔记：LLM 原理 -- GPT 系列的两个阶段

基于 CS2Robotics_Roadmap Level 1 学习过程中的讨论整理。

---

## 1. GPT 系列的两阶段结构

GPT 系列的 10 篇工作不是平铺的，它们分为两个明确的阶段：

### 阶段一：Pre-training 定型 (2018-2020)

探索"用什么架构、什么训练目标、怎么 scale"。架构在 GPT-2 定型，之后不再改变。

| 工作 | 做了什么 | 定型了什么 |
|------|---------|----------|
| GPT-1 (2018) | 确立 pre-train + fine-tune | Transformer decoder + next-token prediction |
| GPT-2 (2019) | 去掉 fine-tune, zero-shot | 架构定型 (pre-norm, GELU, KV cache) |
| Scaling Laws (2020.01) | 量化 scaling 规律 | 理论定型: loss 与 compute 的 power-law |
| GPT-3 (2020.05) | 放大到 175B | Scaling 验证完毕, in-context learning 涌现 |

**关键理解**: GPT-2 之后, pre-training 的架构 (Transformer decoder) 和训练目标 (next-token prediction) 基本不再改变。GPT-3 只是把 GPT-2 放大了 117 倍, 核心设计完全一样。

### 阶段二：Post-training 探索 (2020-2023)

Pre-training 定型后, 问题变成: 同一个 base model, 用什么方式 fine-tune 最有效?

| 工作 | 探索了什么 | 核心发现 |
|------|---------|---------|
| RLHF-Summarize (2020.09) | RLHF 可行性 | 人类"判断好坏"比"写标准答案"便宜且效果更好 |
| Codex (2021.07) | 领域 SFT | 同一个 base model + 领域数据 = 专业能力 |
| WebGPT (2021.12) | BC + RLHF + 工具使用 | LLM 可以学会使用外部工具 |
| InstructGPT (2022.03) | SFT + RM + PPO 三步法 | 对齐后 1.3B > 未对齐 175B |
| ChatGPT (2022.11) | InstructGPT 产品化 | 三步法可以产品化 |
| GPT-4 (2023.03) | 多模态 + 更好的 post-training | 扩展到图像输入 |

**关键理解**: 阶段二的所有工作都是在 GPT-3 这个已训好的 base model 上做不同方式的 fine-tune。本质上都是利用 base model 已有的能力, 用不同的训练信号将能力导向特定方向。

---

## 2. 范式演进：从 fine-tune 到 in-context learning

这是 Q1 考试漏掉的核心脉络：

```
GPT-1: 需要标注数据 fine-tune 才能做任务
GPT-2: 不需要 fine-tune, zero-shot 直接做 (但效果一般)
GPT-3: 不需要 fine-tune, 在 prompt 里放几个 example 就行 (in-context learning)
```

### Few-shot 不是微调

Few-shot 完全不更新权重, 没有任何训练过程:

```
传统 fine-tune (GPT-1):
  训练数据: 1000 条标注 → 梯度更新 → 权重改变 → 再推理
  权重变了: 是

Few-shot (GPT-3):
  prompt = "英译中: dog → 狗, cat → 猫, bird → ?"
  模型直接输出: "鸟"
  权重变了: 没有, 纯推理
```

**机制**: attention 看到前面 example 的输入-输出模式, 推断出规则 ("把英文翻译成中文"), 然后对新输入应用这个规则。本质是 pattern matching via attention, 不是学习。

**涌现**: 这个能力只在模型足够大时出现 (175B), 小模型做不到。

---

## 3. Scaling Law 与 Chinchilla 修正

### Kaplan Scaling Laws (2020)

核心发现: LM 的 loss 与 compute/参数量/数据量之间存在 power-law 关系。

```
L(N) = (N_c / N)^alpha_N    ; 参数量
L(D) = (D_c / D)^alpha_D    ; 数据量
L(C) = (C_c / C)^alpha_C    ; 计算量
```

Kaplan 的结论: 固定 compute budget 下, 应该优先扩大模型 (即使不训练到收敛)。

### Chinchilla 修正 (2022)

Chinchilla 发现 Kaplan 低估了数据的重要性: 模型参数和训练 token 应 1:1 等比增长。

```
Kaplan:     优先放大模型, 数据次要
Chinchilla: 模型和数据等比缩放 (N ~ 20D)
```

验证: Chinchilla-70B (1.4T tokens) 优于 Gopher-280B (300B tokens) -- 小模型 + 多数据 > 大模型 + 少数据。

### 对 pi_0 的启示

pi_0 的 3.3B 参数和 10k 小时数据是配套的。不是数据越多越好, 而是模型大小和数据量要匹配。如果只有 1k 小时数据, 3.3B 可能过大 (过拟合); 如果有 100k 小时数据, 3.3B 可能不够。

**Q9 的纠正**: 给定固定计算预算, 正确答案是等比缩放, 不是"大模型小数据"。"大模型小数据"是 Kaplan 的旧结论, 被 Chinchilla 否定了。

---

## 4. 架构细节：Pre-norm vs Post-norm

```
Post-norm (原始 Transformer):  x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm
Pre-norm  (GPT-2+):           x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add
```

Pre-norm 让残差路径上没有 normalization, 梯度可以直接流过, 训深层网络 (48层、96层) 更稳定。GPT-2 引入, 现在所有 LLM/VLA 都用 pre-norm。

**对机器人的相关性**: 不需要背细节, 但需要知道这个设计存在 -- pi_0 和 GR00T N1 的 Transformer 也用 pre-norm。如果将来改架构或调试训练不稳定, 会遇到。

---

## 5. Autoregressive 的根本限制 (对机器人很重要)

自回归生成: 逐个生成 token, 每步的输出作为下一步的输入。

根本限制 (不是 compounding error, 那是 imitation learning 的问题):

**核心瓶颈: autoregressive 的每一步都依赖上一步的输出作为输入, 因此必须串行执行, 无法并行。** 这是架构层面的硬限制, 不是工程优化能解决的。

```
1. 串行瓶颈: 第 n 个 token 的生成必须等第 n-1 个完成 (因为需要它作为输入)
   → 7 DoF = 7 个 token, 必须一个关节一个关节地出
   → 50Hz × 20 DoF (灵巧手) = 每秒 1000 个 token 串行, 几乎不可能实时

2. 离散化精度: 连续关节角必须映射到有限 bins
   → 256 bins 覆盖 [-pi, pi]: 每 bin ≈ 0.025 rad ≈ 1.4°
   → 输出只能是 bin 中心值, 相邻帧可能"跳变" (不平滑)
   → bins 加多 (4096) 可以缓解精度, 但 token 更多, 串行更慢

两个代价的对比:
  RT-2 (autoregressive): 串行 + 离散 → 速度和精度都受限
  flow matching (pi_0):  一次输出整个连续动作向量 → 两个问题都绕开

注意: autoregressive 不是"不能做"机器人, RT-2/OpenVLA 就在用
      问题在于精度和速度的 tradeoff, 对灵巧操控不够
      pi_0-FAST 后来用更好的 tokenizer 回到了 autoregressive, 效果也不错
```

---

## 6. MoE 与 pi_0 的 Dual-Expert

### MoE 核心思想

模型有多组专家权重 (expert), 每个 token 只激活其中几组。好处: 总参数很大 (容量大) 但每个 token 计算量小。

```
标准 MoE (Kimi-K2):
  384 个相同的小 FFN, 每个 token 由 learned router 选 8 个
  总参数 1T, 激活参数 32B (只用 3%)

pi_0 的 "MoE" (最简形式):
  2 个不同的大模块:
    Expert 1 (VLM 3B):       处理 image/text tokens
    Expert 2 (Action 300M):  处理 action tokens
  Routing: 按 token 类型硬编码 (不是 learned router)
```

MoE 不是 Kimi 独有的, 是通用架构思想, DeepSeek-V3、Mixtral 等都用。

### 为什么 pi_0 要分成两组 expert

如果用一组权重同时处理 image/text 和 action:
- 动作训练的梯度会破坏 VLM 已学到的视觉-语言知识 (灾难性遗忘)
- 视觉-语言表征需要语义对齐, 动作表征需要平滑连续, 两种需求冲突

分开后: 动作梯度只修改 action expert 权重, VLM 权重不变。两者通过 attention 共享信息但参数互不干扰。

---

## 7. RLHF：从判断到对齐

### 为什么用 RL 而不是 supervised learning

GPT-3 的 pre-training 目标是 next-token prediction (模仿互联网文本分布)。但互联网文本 ≠ 好回答:

```
互联网上常见: 长篇废话、重复内容、有害内容、不回答问题
人类想要:    简洁、准确、直接回答、安全无害
```

两者之间的差距就是 "misalignment"。RLHF 的 insight: 人类更擅长"判断好坏"而非"写标准答案":

```
写一个好回答:        5-10 分钟, 需要写作能力
比较两个回答哪个好:  1-2 分钟, 只需要阅读理解
```

### SFT + RM + PPO 三步法

```
Step 1: SFT (Supervised Fine-Tuning, 监督微调)
  人写"标准答案" → 微调模型 → 给 RL 一个合格的起点
  类比: 新员工入职培训

Step 2: RM (Reward Model, 奖励模型)
  模型生成多个回答 → 人类排序 → 训练一个自动评委
  RM 输出标量分数, 但只有相对大小有意义, 绝对值无意义
  类比: 训练一个自动考核系统

Step 3: PPO (Proximal Policy Optimization, 近端策略优化)
  用 RM 分数作为 reward → RL 优化模型
  reward = RM_score - beta * KL(和 SFT 模型的偏离)
  KL penalty 防止模型跑到 RM 没见过的区域 (reward hacking)
  类比: 员工根据考核反馈自主改进
```

### 为什么需要 SFT 作为起点

RLHF-Summarize (单任务, 只做摘要) 可以不加 SFT 直接 RL。但 InstructGPT (通用指令) 必须先 SFT, 因为:
- 没有 SFT 的 base model 只会"续写网页", 不知道"回答问题"的格式
- PPO 的探索空间太大, 收敛极慢
- SFT 把起点从"续写文本"移到"回答问题", PPO 只需要在小范围内优化质量

### Alignment (对齐) 的含义

让模型的行为和人类的意图一致。未对齐的模型有能力但方向偏 (模仿互联网), 对齐后能力方向指向"帮助用户"。

InstructGPT 的核心发现: **Alignment > Scale** -- 对齐后 1.3B > 未对齐 175B。调方向比加力量更有效。

### 对齐也是一种 fine-tune

Pre-train → SFT → RLHF 三者都是在调整同一个模型的权重, 区别只在于训练信号:
- Pre-train: 自监督 (next-token prediction)
- SFT: 监督 (人写的回答)
- RLHF: 强化学习 (人类偏好比较 → RM → PPO)

RLHF 不注入新知识, 而是重新分配已有能力的方向。证据: RLHF 后某些 NLP benchmark 性能下降 (alignment tax), 说明总能力没变, 分配方式变了。

### RLHF 对机器人的意义

**当前**: 机器人的 reward 大多可以用公式定义 (抓到了吗? 方块转到目标了吗?), 不需要 RLHF。

**未来 (可能)**: 当任务涉及人类主观偏好时 ("把杯子放到方便我拿的位置"), reward 写不出来, 可能需要类似 RLHF 的方法。

### RLHF 的演进趋势

```
2020 RLHF-Summarize:  人类比较 → RM → PPO          (人工标注, 单任务)
2022 InstructGPT:      人类比较 → RM → PPO          (人工标注, 通用)
2025 Kimi-K2:          自动验证 + 模型自我评判 → RL  (大幅减少人工)
```

人类参与越来越少, 但 SFT 作为起点一直没被去掉。

---

## 8. Token Embedding vs 上下文化表征

```
Token embedding: 查表操作, 同一个词永远得到相同向量
  "bank" 在 "river bank" → [0.3, -0.1, ...]
  "bank" 在 "bank account" → [0.3, -0.1, ...]  (完全相同!)

经过 Transformer 后: 上下文化表征
  "river bank" 中的 "bank" → 融入了 "river" 信息 → 偏向"河岸"含义
  "bank account" 中的 "bank" → 融入了 "account" 信息 → 偏向"银行"含义
```

**对机器人的重要性**: pi_0 中同一个 action token, 在不同 image/text context 下, 经过 attention 后的表征完全不同。这就是 Transformer 能做条件动作生成的原因 -- action 的含义取决于当前看到的图像和听到的指令。

---

## 9. "一个 Base Model, 多种 Fine-tune" 的模式

GPT-3 之后的 Codex/WebGPT/InstructGPT/ChatGPT 全部基于同一个 base model:

| 工作 | Base Model | Fine-tune 数据 | 获得的能力 |
|------|-----------|---------------|----------|
| Codex | GPT-3 | 159GB GitHub 代码 | 代码生成 |
| WebGPT | GPT-3 | 浏览器交互轨迹 | 搜索引擎使用 |
| InstructGPT | GPT-3 | 13K 指令 + 33K 偏好 | 指令遵循 |
| ChatGPT | GPT-3.5 | 对话数据 | 多轮对话 |

**核心启示**: 领域 fine-tune 的效果远大于模型 scale -- GPT-3 175B 不会写代码, 但 Codex 12B (小 14x) 经过代码 fine-tune 就能写。

**对机器人的直接映射**: PaliGemma 3B (VLM) 不会控制机器人, 通过机器人操作数据 fine-tune → pi_0, 就能控制机器人。

---

## 10. 开源 LLM 的意义

GPT-4 不公开架构、数据、训练方法 → 不可复现、不可研究。

开源 LLM (Qwen, LLaMA, DeepSeek) 让研究者可以:
- 阅读代码理解架构
- 在自己数据上 fine-tune
- 修改架构做实验

**对机器人的直接影响**: pi_0 开源 (openpi) → 你可以在自己的灵巧手数据上 fine-tune VLA, 而不需要从零训练。没有开源模型, 整个 VLA 微调生态不存在。

---

## 讨论中澄清的概念

### "范式"的定义

范式 = 做事的标准流程。"范式变了"指整个行业解决问题的方式变了, 不是某个模型变好了。

不同粒度的范式:
- 大粒度: "深度学习范式" vs "特征工程范式"
- 中粒度: "pre-train + fine-tune" vs "in-context learning" (GPT 系列 Q1 问的)
- 小粒度: "DDPM" vs "flow matching"

当一种做法变成了大家默认的做法, 它就是范式。

### Q1 理解误区: 范式演进 ≠ scaling

模型变大是涌现的**原因**, 范式变化是**结果**。两个不同层面:
- "Why": 模型变大 → 能力涌现 (你的回答, 对的)
- "What": 使用方式从 fine-tune → zero-shot → in-context learning (题目问的)

### Q2 理解误区: few-shot 不是微调

LLM 里的 few-shot 完全不更新权重, 是 attention 在推理时做 pattern matching。

### LLM few-shot vs 机器人 few-shot

| | LLM few-shot | 机器人 few-shot |
|---|---|---|
| 更新权重 | 不更新 | 更新 (LoRA / full fine-tune) |
| 数据形式 | prompt 里的几个 example | 几条遥操作轨迹 |
| 实质 | 临时提醒 (模型想起已有能力) | 真正学会 (获得新能力) |
| 共同点 | 用极少数据适应新任务 | 用极少数据适应新任务 |

机器人里的 few-shot 更接近 GPT-1 的 fine-tune, 只是数据量少。prompt 式 few-shot 在机器人动作控制层面没有用 -- 你没法把轨迹塞进 prompt 里让模型"临时学会"。但在 VLA 的语言指令层面可能有用。

### Q3 理解误区: "更多就是更好" vs "要匹配"

Chinchilla 的启示不是数据越多越好, 而是模型大小和数据量要匹配:
- 100 小时数据 + 3B 模型 → 过拟合 (模型太大)
- 10000 小时数据 + 300M 模型 → 浪费 (模型太小)
- 数据量决定了该用多大的模型

实际决策: 如果你只有 100 条演示轨迹, 用 pi_0 3.3B 做 full fine-tune 大概率过拟合, 应该用 LoRA 或更小的模型。

---

## 讨论中补充的关键理解

### Transformer 内部结构 (Level 0 遗留)

```
Transformer 一层 = Attention + FFN

Attention: token 之间交互信息 ("开会")
FFN:       每个 token 独立做非线性变换 ("独立思考")
           y = W2 * GELU(W1 * x), 先升维再降维
           占 Transformer 总参数约 2/3

MoE 替换的就是 FFN: 一个大 FFN → 多个小 FFN (expert), 每个 token 选几个用
```

### Scaling Law 的完整结构

```
三组独立的 power-law (Kaplan 和 Chinchilla 都同意):
  L(N) — loss vs 参数量
  L(D) — loss vs 数据量
  L(C) — loss vs 计算量

Chinchilla 修正的不是 power-law 本身, 而是固定 compute 下的分配策略:
  Kaplan:     N 大 D 小 (优先放大模型)
  Chinchilla: N 和 D 等比 (模型和数据一样重要)
```

### 开源 LLM 对机器人的意义

不是"社区情怀", 而是直接提供了 VLA 的核心组件:
- pi_0 的 backbone = PaliGemma (Google 开源)
- OpenVLA 的 backbone = LLaMA 2 (Meta 开源)
- 没有开源 LLM, 就没有可 fine-tune 的 VLA backbone, 你的灵巧手研究就做不了

---

## 考试纠错总结

| 题号 | 原始回答的问题 | 修正后的理解 |
|------|-------------|------------|
| Q1 | 把范式演进等同于 scaling | scaling 是原因, 范式变化 (fine-tune→zero-shot→ICL) 是结果 |
| Q2 | 以为 few-shot 是微调 | LLM few-shot 不更新权重; 机器人 few-shot 才更新权重 (实为小数据 fine-tune) |
| Q3 | "更多就是更好" | 不是更多更好, 是模型和数据要匹配 |
| Q5 | 以为限制是 compounding error | 核心是串行依赖 (每个 token 必须等上一个) + 离散化精度丢失 |
| Q6 | 不了解 MoE | 核心: 参数分开各管各的, attention 共享保证交互。pi_0 用最简版 (2 expert 硬编码) |
| Q8 | 不知道 | embedding 静态查表相同; attention 后融入 context 变成不同表征。pi_0 的 action token 含义由看到的图像和指令决定 |
| Q9 | 和 Q3 矛盾 | Chinchilla: 等比缩放, 不是大模型小数据 |
| Q10 | "促进社区发展" | 开源 LLM 直接提供了 VLA 的 backbone 组件, 不是情怀而是技术必需 |
