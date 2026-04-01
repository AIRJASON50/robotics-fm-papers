# BERT: Pre-training of Deep Bidirectional Transformers -- 学习笔记
> 一句话: 用 Masked LM (随机遮盖 15% token 让模型双向预测) 打破单向语言模型的限制, 确立 pre-train + fine-tune 为 NLP 标准范式.
> 论文: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language), 2018, arXiv 1810.04805

## 这篇论文解决了什么问题
GPT 等模型只能单向 (left-to-right) 预训练, 这对需要理解完整上下文的任务 (QA, NLI) 是致命缺陷. ELMo 虽然拼接了双向 LSTM, 但只是浅层拼接而非深度双向. BERT 要解决的是: 如何用 Transformer 做真正的 deep bidirectional pre-training, 然后通过 fine-tune 适配所有下游任务.

## 核心想法 (用直觉解释)
标准语言模型从左到右逐词预测, 双向则会让每个词 "看到自己". BERT 的解法是完形填空 -- 随机 mask 掉一些词让模型根据两侧上下文猜回来. 这样每个 token 的表示天然融合了左右信息. 同时训练 Next Sentence Prediction 来学句间关系. 预训练完成后, 对任何任务只需加一个输出层 fine-tune 即可.

## 关键设计决策
- **MLM (Masked Language Model)**: 随机选 15% token, 其中 80% 替换为 [MASK], 10% 随机词, 10% 保持不变. 80/10/10 策略缓解了 pre-train (有 [MASK]) 和 fine-tune (无 [MASK]) 之间的分布不匹配
- **NSP (Next Sentence Prediction)**: 50% 正样本 (真实下一句) + 50% 负样本 (随机句子), 学习句对关系. 后来 RoBERTa 证明 NSP 作用有限, 但 BERT 首次提出了这种多任务 pre-training 的思路
- **Input = Token Emb + Segment Emb + Position Emb**: 一个统一输入格式处理单句和句对任务. [CLS] token 的最终隐状态做分类, [SEP] 分隔句对
- **两个模型规模**: BERT_BASE (L=12, H=768, 110M) 和 BERT_LARGE (L=24, H=1024, 340M), BASE 与 GPT 同规模以便公平对比

## 这篇论文之后发生了什么
RoBERTa 去掉 NSP 并加大训练量证明 MLM 本身就够强. ALBERT 用参数共享压缩模型. XLNet 用 permutation LM 统一双向与自回归. GPT-2/3 走向纯 decoder 路线并最终以 scaling 取胜. 在 CV 中, BEiT 和 MAE 直接将 MLM 迁移为 masked image modeling. 在 robotics 中, Octo 用 masked trajectory prediction 做 multi-task pre-training.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | MLM = "破坏-重建" 自监督范式 | MAE 预训练 visual encoder, Octo 用 masked trajectory 学 multi-task policy, 都是 MLM 的直系后裔 |
| 2 | 双向 (encoder) 适合理解, 单向 (decoder) 适合生成 | VLA 的 vision encoder 用 BERT 式双向编码 (ViT), action decoder 用 GPT 式自回归, 两者分工 |
| 3 | pre-train + 极简 fine-tune 消灭 task-specific 架构 | robotics FM 的目标: 一个预训练模型, 加最少的适配层就能部署到不同任务/机器人 |
| 4 | [CLS] token 作为全局汇聚点 | RT-1/RT-2 的 action token 附在 visual/language token 后, 起同样的全局信息汇聚作用 |
