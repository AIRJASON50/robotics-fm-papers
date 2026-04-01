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

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | "多任务共享表示"先验直接预言了 cross-embodiment training 的成功 | pi_0 在 7 种机器人上联合训练, 共享 VLM 表示层, 各 embodiment 只需轻量 adapter |
| 2 | "流形假设"解释了为什么 Diffusion Policy / Flow Matching 能在动作空间工作 | 机器人动作分布在高维空间的低维流形上, 生成模型在这个流形上采样 |
| 3 | "层次化表示"是 VLA 架构设计的指导原则 | 图像编码器 (低层感知) -> VLM (高层语义理解) -> action expert (动作生成) 正是 hierarchy |
