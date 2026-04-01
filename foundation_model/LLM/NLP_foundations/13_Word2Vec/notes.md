# Word2Vec: Efficient Estimation of Word Representations in Vector Space -- 学习笔记
> 一句话: 用极简 log-linear 模型 (CBOW / Skip-gram) 从海量文本学词向量, 首次发现 embedding 具有线性代数结构 (king - man + woman = queen).
> 论文: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google), 2013, arXiv 1301.3781

## 这篇论文解决了什么问题
当时 NLP 系统把词当 one-hot 符号 (词表 100 万维, 每个词一个位置为 1 其余为 0), 词之间没有相似性概念. 已有的神经语言模型 (NNLM, RNNLM) 虽然能学到词表示, 但计算量太大 -- 主要瓶颈在 hidden layer 的非线性计算 (O(N*D*H)). 本文目标: 设计计算量最小的架构, 在数十亿词上快速训练高质量词向量.

## 核心想法 (用直觉解释)
既然 hidden layer 是瓶颈, 就干脆去掉它. CBOW (Continuous Bag-of-Words): 把上下文窗口内的词向量求平均, 直接预测中心词 -- 没有非线性层, 复杂度 O(N*D + D*log(V)). Skip-gram: 反过来, 用中心词预测上下文中的每个词. 用 hierarchical softmax (Huffman tree) 避免在全词表上做 softmax. 在 6B 词的 Google News 上不到一天就训练完成. 惊人发现: 学到的向量空间具有线性结构, 语义/语法关系可以用向量加减表示.

## 关键设计决策
- **去掉 hidden layer**: CBOW 和 Skip-gram 都是 log-linear 模型, 输入层直接连 projection 再连 output. 这把单样本复杂度从 O(N*D*H + H*V) 降到 O(N*D + D*log(V))
- **两种互补架构**: CBOW 在语法任务上更好 (利用上下文的平均信息), Skip-gram 在语义任务上更好 (每个上下文词独立预测, 学到更细粒度的关系). Skip-gram 后来更流行
- **Hierarchical softmax + Huffman tree**: 高频词路径短, 低频词路径长, 比标准 softmax 快约 2 倍. 后续工作引入 negative sampling 进一步简化
- **维度与数据同步 scaling**: Table 2 表明, 增加维度或数据量单独有收益但会饱和, 必须同步增加才能持续提升. 这是 scaling law 的朴素前身

## 这篇论文之后发生了什么
GloVe (Pennington 2014) 用矩阵分解统一了 Word2Vec 和共现矩阵方法. FastText 扩展到 subword 级别. ELMo (2018) 从 static embedding 进化到 context-dependent embedding (同一个词不同上下文有不同向量). BERT 和 GPT 用 Transformer 做 contextual embedding, 但输入层仍保留 Word2Vec 式的 token embedding. Word2Vec 的 self-supervised 思想 (用上下文预测目标) 延伸到 CV (contrastive learning) 和 robotics (task embedding).

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Embedding 是 FM 的基石: 离散/高维输入映射到有结构的连续空间 | VLA 的 language embedding, goal-conditioned RL 的 goal embedding, 本质都是 Word2Vec 开创的事 |
| 2 | Self-supervised pretext task 决定 representation 质量 | Word2Vec 用 "预测上下文" 学表示, robotics FM 用 "预测下一帧/动作" (DreamerV3, UniSim) 学表示, 核心逻辑相同 |
| 3 | 简单模型 + 大数据 > 复杂模型 + 小数据 | 1B 词上的 Skip-gram 碾压 300M 词上的 RNNLM, 对 robotics 的启示: 数据规模比模型花哨程度更关键 |
| 4 | 向量算术 = 概念组合 | 如果 robot skill embedding 有线性结构, skill composition 就变成向量运算 -- SayCan/RT-2 做 language grounding 时隐含了这个假设 |
