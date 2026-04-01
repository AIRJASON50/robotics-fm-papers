# Word2Vec: Efficient Estimation of Word Representations (Mikolov et al., 2013) -- Takeaway Notes

> 一句话: 用极简的 log-linear 模型 (CBOW / Skip-gram) 从海量文本中学到词向量, 首次证明 learned embedding 能捕获语义代数结构 (king - man + woman = queen).

## 核心贡献
- 提出两个无 hidden layer 的架构: CBOW (上下文预测中心词) 和 Skip-gram (中心词预测上下文), 将训练复杂度从 O(H x V) 降到 O(D x log(V))
- 发现 embedding 空间具有线性代数结构: 语义关系可通过向量加减运算, 这远超 "相似词聚类" 的预期
- 证明 "简单模型 + 大数据" 可以打败 "复杂模型 + 小数据" -- 在 6B token Google News 上用 1 天完成训练, 超过之前所有 NNLM/RNNLM
- 设计了 Semantic-Syntactic Word Relationship 测试集, 为 embedding 质量提供定量评估标准

## 为什么重要
Word2Vec 是 "learned representation 取代 hand-crafted feature" 这一范式转变在 NLP 中的起点.
之前 NLP 系统把词当作 one-hot 符号 (无相似性概念), Word2Vec 之后, 连续向量表示成为一切
下游模型的输入基础. 这条线直通 BERT/GPT 的 token embedding 和 robotics 中的 task embedding.

核心洞察: **pretext task 的设计决定了 representation 的质量**. 预测上下文这个 self-supervised
目标, 逼迫模型把语义/语法信息压缩进固定维度的向量 -- 这和后来 contrastive learning (MoCo/CLIP)
以及 masked prediction (MAE/BERT) 的思路一脉相承.

## 对你 (RL->FM) 的 Takeaway
- **Embedding 是 FM 的基石**: 无论是 VLA 中的 language embedding, 还是 task/goal conditioned RL
  中的 goal embedding, 本质上都在做 Word2Vec 开创的事 -- 把离散/高维输入映射到有结构的连续空间
- **Self-supervised pretext task 设计**: Word2Vec 用 "预测上下文" 来学表示, robotics FM 用
  "预测下一帧/动作" 来学表示 (DreamerV3, UniSim), 核心逻辑相同
- **Scaling 的早期证据**: 维度和数据量需要同步增长才能提升质量 (Table 2), 这是 Chinchilla scaling law 的朴素前身
- **向量算术 = 概念组合**: 如果 robot skill embedding 也有这种结构, 那么 skill composition 就变成向量运算 -- 这正是 SayCan/RT-2 做 language grounding 时隐含的假设

## 与知识库其他内容的关联
- foundations/12_RepresentationLearning: Word2Vec 是 representation learning 在 NLP 的第一个杀手级应用
- foundations/17_Transformer: Transformer 的 input embedding layer 直接继承 Word2Vec 的思想, 但改为可训练
- 14_Seq2Seq: encoder 将整个句子压缩为向量, 可看作 Word2Vec "词向量" 到 "句向量" 的升级
- 18_BERT: 从 static embedding 进化到 contextual embedding, 同一个词在不同上下文有不同向量
- CV/4_self_supervised (MoCo, DINO): 同样用 pretext task 学 visual representation, 思路与 Word2Vec 高度对称
