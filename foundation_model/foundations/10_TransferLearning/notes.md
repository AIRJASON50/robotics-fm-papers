# A Survey on Transfer Learning -- 学习笔记
> 一句话: 系统定义了 transfer learning 的形式化框架和分类体系 (inductive / transductive / unsupervised), 为 pre-train + fine-tune 范式奠定理论基础.
> 论文: Sinno Jialin Pan, Qiang Yang (Hong Kong UST), 2010, IEEE TKDE

## 这篇论文解决了什么问题
传统机器学习假设训练和测试数据同分布, 但现实中这个假设经常不成立: 新网站的分类器没有标注数据, WiFi 定位模型换了设备就失效, 产品评论的情感分类器换品类就不准. 重新收集标注数据代价极高. Transfer learning 要解决: 当 source 和 target 的 domain 或 task 不同时, 如何把已有知识迁移到新任务上.

## 核心想法 (用直觉解释)
核心 formalization: domain D = {feature space X, marginal distribution P(X)}, task T = {label space Y, predictive function f}. Transfer learning = 利用 source domain D_S 和 task T_S 的知识来提升 target domain D_T 上 task T_T 的学习, 其中 D_S != D_T 或 T_S != T_T. 把零散的 domain adaptation, multi-task learning, self-taught learning 等概念统一到一个框架下, 并按 "What to transfer / How to transfer / When to transfer" 三个问题组织方法.

## 关键设计决策
- **三大分类**: Inductive (target 有标注, task 不同), Transductive (task 相同, domain 不同, target 无标注), Unsupervised (两侧都无标注). 覆盖了从 multi-task learning 到 domain adaptation 的全部场景
- **四种迁移方式**: Instance-transfer (重加权 source 样本), Feature-representation-transfer (学跨域共享特征), Parameter-transfer (共享模型参数/先验), Relational-knowledge-transfer (迁移关系结构)
- **Negative transfer 警告**: 当 source 和 target 不相关时强行迁移会损害性能. 论文明确指出 "when NOT to transfer" 与 "how to transfer" 同等重要

## 这篇论文之后发生了什么
Deep transfer learning 主导了后续发展: ImageNet pre-trained CNN fine-tune (2014) 证明深度特征通用性. BERT/GPT 把 NLP 带入 pre-train + fine-tune 时代. Domain adaptation 发展出 adversarial DA (DANN), optimal transport DA 等. Foundation model 概念 (Bommasani 2021) 本质上是 transfer learning 的极致 -- 一个模型迁移到所有下游任务. Negative transfer 仍是开放问题, 体现为 catastrophic forgetting 和 sim-to-real gap.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Sim2real 是 transductive transfer (同任务, 不同域分布) | 论文的 domain adaptation 分析直接适用: 你需要对齐 sim 和 real 的特征分布 (domain randomization = instance re-weighting 的变体) |
| 2 | Pre-trained encoder + RL fine-tune = inductive transfer | 冻结 R3M/CLIP visual encoder + PPO 训练 policy head, 就是 feature-representation-transfer 的实例 |
| 3 | Negative transfer 是真实风险 | 在差距太大的 sim 上预训练可能比从头训练更差, 选择合适的 source domain (sim fidelity) 和 transfer 方式 (冻结/解冻哪些层) 至关重要 |
| 4 | Foundation model = transfer learning 的极致形式 | 理解这个分类体系有助于判断: 你的具体 robotics 问题属于哪种 transfer 设定, 该用什么策略 |
