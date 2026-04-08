# Representation Learning: A Review and New Perspectives -- 学习笔记

> 一句话: 系统性地回答了"什么是好的数据表示"以及"如何自动学习这种表示", 为整个 deep learning 和后续 foundation model 的核心逻辑奠基.
> 论文: Yoshua Bengio, Aaron Courville, Pascal Vincent, IEEE TPAMI 2013
> 引用量级: ~25,000+

## 这篇论文解决了什么问题

2012 年之前, 机器学习的性能高度依赖人工特征工程 (feature engineering): 针对每个任务手动设计特征 (SIFT, HOG, MFCC 等). 这不仅费时费力, 还限制了泛化能力. 核心问题是: 能否让算法自动从原始数据中学习好的表示, 使得下游任务 (分类、预测) 变得更简单?

## 核心想法 (用直觉解释)

论文的中心论点是: **好的表示 = 好的 AI**. 一个好的表示应该能解开数据背后的 explanatory factors (解释性因素). 想象一张照片, 其中纠缠着物体形状、光照方向、背景颜色等多个因素. 一个好的表示应该把这些因素解开 (disentangle), 使得每个因素可以独立变化.

论文提出了好表示的关键先验: (1) Distributed representation -- 用 N 个特征的组合可以表示 2^N 种模式, 比 one-hot 指数级高效, 这就是 word embedding 用 300 维就能覆盖整个词汇表的原因; (2) 深度 (depth) 带来抽象 -- 每一层组合底层特征形成更抽象的概念 (像素 -> 边缘 -> 物体 -> 场景), 且深层网络可以指数级更高效地复用特征; (3) 不同任务共享底层 factor -- 这直接解释了为什么 pre-train + fine-tune 能工作.

论文还覆盖了当时的主要方法: probabilistic models (RBM, DBN), autoencoders (denoising/contractive/sparse), 以及 manifold learning. 虽然这些具体方法已被 Transformer 取代, 但它们背后的设计原则 (层次化、分布式、解耦) 仍然是现代模型的指导思想.

## 关键设计决策

1. **将"好表示"形式化为一组先验**: smoothness, multiple factors, hierarchy, shared factors across tasks, manifold structure, temporal coherence, sparsity. 这不是技术方案而是设计原则 -- 现代 foundation model 的每一个设计都可以追溯到这些先验.

2. **强调 unsupervised pre-training 的价值**: 论文在 2012 年就主张用大量无标签数据学习通用表示, 然后在少量标签数据上微调. 这正是 GPT/BERT 路线的理论基础, 也是 VLA 在机器人数据稀缺时能工作的原因.

3. **Disentangling > Invariance**: 论文区分了两个目标 -- 学习不变特征 (对无关因素不敏感) vs 解耦所有因素 (每个因素独立可控). 后者更强, 因为事先无法知道哪些因素"无关".

## 这篇论文之后发生了什么

表示学习的思想直接驱动了后续十年的进展: Word2Vec (2013, 词的分布式表示) -> Transformer (2017, 通过 attention 学习上下文表示) -> BERT/GPT (大规模预训练表示) -> CLIP (2021, 视觉-语言联合表示) -> VLA (2024, 视觉-语言-动作联合表示). 这篇综述中的每一条先验都在后续工作中得到了验证.

## 贯穿全局的表征学习原理 (基于学习过程中的讨论提炼)

> 交叉引用: 表征学习与迁移学习的关系 → `../TransferLearning_Origins/note.md`
> Bengio 的"shared factors across tasks"直接预言了迁移学习的成功

### 核心认知: 学习的本质是压缩

```
NN 的作用: 表征和压缩
在有结构/规律/pattern 的数据中:
  好的表征和压缩 = 找到数据中的规律从而理解数据

一个好的 model 需要三个条件:
  1. 结构化的数据 (数据中有可提取的共性/规律)
  2. 足够的参数量 (能区分精细的 pattern, 但不能多到逐条记忆)
  3. 符合规律和结构的 loss (引导梯度下降方向和 pattern 一致)

NN 不"追求"理解, 它只追求 loss 低
但 loss 低的唯一途径恰好是理解 (压缩出 pattern)
理解是结果, 不是目标
```

### MLP: 表征能力的基本单元

**为什么 MLP 能做表征 (Bengio Section 3.3 原文)**:

```
论文: "multi-layer neural networks can all represent up to O(2^k) input regions 
       using only O(N) parameters"

一个神经元 = 一个超平面 (把空间切一刀, 检测一个方向)
N 个神经元 = N 个超平面 = 把空间切成 2^N 个区域
→ O(N) 参数, O(2^N) 种区分能力 (指数级表达效率)

对比: 决策树/最近邻/SVM 用 O(N) 参数只能区分 O(N) 种模式 (线性增长)
```

**非线性激活函数的角色**:

```
没有非线性 (纯线性 MLP):
  y = W2 @ (W1 @ x) = (W2@W1) @ x → 等价于一层, 只能画直线/超平面
  无论堆多少层都等价于一层 → 不能弯曲空间

有非线性 (SiLU/GELU/ReLU):
  y = W2 @ SiLU(W1 @ x)
  W1: 把输入投影到各"检测方向" (线性切空间)
  SiLU: 正值通过, 负值抑制 (折叠空间)
  W2: 在折叠后的空间做线性组合 (压缩结论)
  
  → 空间被弯曲了 → 切割面能贴合数据流形的弯曲形状

SiLU/GELU 本身是固定函数, 不学习
真正学习的是 W: 决定"喂给 SiLU 什么输入"
  → W 把有用的 pattern 投影成正值 → SiLU 通过 (保留)
  → W 把无用的 pattern 投影成负值 → SiLU 抑制 (丢弃)
  → 训练 = 调 W = 决定"哪些 pattern 该保留, 哪些该丢弃"
```

**MLP 的升维→折叠→降维 (Transformer 中的 FFN)**:

```
input (768维) → Linear(768→3072) → SiLU → Linear(3072→768)

升维 (768→3072): 3072 个检测器, 每个检测一种 pattern 方向
SiLU: 激活有用的检测器, 抑制无用的 (折叠空间)
降维 (3072→768): 压缩回原始维度 (保留被激活的 pattern)

Bengio 论文的数学: 3072 个检测器 → 2^3072 种可能的激活模式 → 天文数字的区分能力
```

**深度 = 特征复用 = 指数级更高效 (Bengio Section 3.4)**:

```
论文: "deep architectures promote the re-use of features"
      "the number of paths can grow exponentially with its depth"

第 1 层 MLP: 检测底层 pattern (边缘, 颜色)
第 2 层 MLP: 复用第 1 层的输出, 组合成中层 pattern (形状, 纹理)
第 6 层 MLP: 复用之前所有层, 组合成高层 pattern (物体, 场景)

→ 不是"更多参数", 而是"特征复用 = 指数级更高效的表达"
→ 12 层 Transformer 的表达能力远超 12 个独立 MLP 的简单叠加
```

### 两种生成范式的几何本质

```
范式 A: 跨空间映射 (AE / VAE)
  data space (高维) ←→ latent space (低维)
  本质: 找到数据流形的坐标系 (降维后每个维度对应一个因子)
  类比: 地球表面 → 用经纬度描述

范式 B: 同空间映射 (DDPM / Flow Matching)
  data space (高维) → data space (高维)
  本质: 不找坐标, 学一个"引力场"把任意点推到流形上
  类比: 引力场把太空中任意位置拉回地球表面

两者的联系: 都是在学"数据流形的结构"
  范式 A 显式参数化流形 (找坐标)
  范式 B 隐式描述流形 (学引力方向)
```

### 表征视角下的 VLA 架构设计

**为什么不同模态需要不同的 MLP (2022-2025 多篇论文验证)**:

```
Bengio 论文: "different explanatory factors of the data tend to change independently"

不同模态有不同的 explanatory factors:
  视觉: 颜色/形状/空间关系 → 需要检测纹理/边缘/位置的 pattern
  语言: 语法/语义/逻辑 → 需要检测词序/搭配/推理的 pattern
  动作: 力度/轨迹/时序协调 → 需要检测运动学/物理的 pattern

→ 不同模态的 pattern 本质不同 → 需要不同的"检测器" (MLP)
→ 共用 MLP = 强制不同模态用同一组检测器 → 互相挤占 → 效果差

验证:
  VLMo (2022): 分开 FFN → 优于共享
  CogVLM (2024): Visual Expert (独立 FFN) → 大幅优于 adapter
  EVEv2 (2025): 共享 FFN 导致 "representational interference"
  MoE-LLaVA: 即使不显式分配, experts 也自然按模态分化

pi_0 的设计符合这个原理:
  VLM MLP: 检测视觉语言 pattern (预训练已优化)
  Action Expert MLP: 检测动作 pattern (从头学)
  Attention 共享: 跨模态信息路由 (不需要分开)
```

**MoE 的表征解释**:

```
一个 MLP (3072 检测器): 所有 pattern 挤在一起
MoE (384 个 MLP, 每次选 8 个): 不同 pattern 用不同的检测器组

Bengio 论文: "the most robust approach is to disentangle as many factors as possible"
→ MoE 是自动解耦: 让不同 expert 自动专注于不同 factor
→ 384 个 expert 比 1 个 MLP 能解耦更多 factor
→ 但需要足够多数据才能让每个 expert 充分特化 (Chinchilla 的 insight)
```

### Scaling Law 的表征解释

```
Bengio: "N 个参数 → 2^N 种区分能力"

参数少 (小模型): 检测器少 → 只能区分粗 pattern ("所有抓取差不多")
参数多 (大模型): 检测器多 → 能区分细 pattern ("硬物用力, 软物轻柔")
数据少: pattern 样本不够 → 检测器训不满 → 过拟合
数据多: pattern 样本充分 → 检测器充分特化 → 泛化

Chinchilla: 参数和数据要匹配 → 检测器数量和 pattern 样本数要匹配
→ 太多检测器+太少数据 → 检测器记忆而非压缩 → 不泛化
→ 太少检测器+太多数据 → 压缩太粗 → 丢细节
```

---

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | **学习的本质是压缩**: 有限参数+结构化数据+loss → 被迫找到 pattern | 贯穿你学过的所有内容: AE/GPT/CLIP/DDPM/RT-1/pi_0 都是这个原理 |
| 2 | **MLP 是表征的基本单元**: N 个神经元 → 2^N 种区分能力, 非线性提供弯曲流形的能力 | 理解为什么 Transformer 里每层都有 MLP, 为什么不同模态要分开 MLP |
| 3 | **深度 = 特征复用**: 多层 MLP 指数级更高效, 不是简单叠加 | 理解为什么 12 层比 1 层好得多, 但不是线性好 12 倍 |
| 4 | "多任务共享表示"先验直接预言了 cross-embodiment training 的成功 | pi_0 在 7 种机器人上联合训练, 共享 VLM 表示层 |
| 5 | "流形假设"解释了 Diffusion/Flow Matching 的几何本质 | 同空间映射 = 学引力场把点推到流形上 |
| 6 | "层次化表示"是 VLA 架构设计的指导原则 | ViT (感知) → VLM (语义) → Action Expert (动作) = hierarchy |
| 7 | **不同模态不同 MLP** 是当前已验证的最佳实践 | pi_0 分 VLM/Action Expert, GR00T 进一步分 embodiment |
| 8 | **选 backbone 看预训练任务和数据, 不看架构** | SigLIP > ImageNet ViT, 因为预训练 pattern 不同 |
| 9 | **数据中的 pattern 决定学到什么, loss 方向决定怎么学** | 数据质量 > 模型架构, 这是工程师的核心工作 |
