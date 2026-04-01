# A Survey on Transfer Learning (Pan & Yang, 2010) -- Takeaway Notes

> 一句话: 系统定义了 transfer learning 的分类体系, 奠定了 pre-train + fine-tune 范式的理论基础.

## 核心贡献
- 给出 transfer learning 的形式化定义: source domain/task -> target domain/task, 允许分布不同
- 将方法分为 inductive / transductive / unsupervised 三大类, 统一了之前零散的 domain adaptation 文献
- 明确了 "negative transfer" 问题: 盲目迁移可能损害目标任务性能

## 为什么重要
这篇综述是 foundation model 时代的理论起点. 今天所有 pre-train + fine-tune 的做法
(BERT, GPT, RT-2 等) 都属于 inductive transfer learning 的实例.
没有这套框架, "为什么大模型能迁移到下游任务" 就缺少理论语言来讨论.

## 对你 (RL->FM) 的 Takeaway
- sim2real 本质上是 transductive transfer (同任务, 不同域分布). 论文中 domain adaptation
  的分析直接适用于 sim-to-real gap 问题.
- 当你用 pre-trained vision encoder + PPO fine-tune 做机器人操作时, 你在做 inductive
  transfer -- 理解这个分类有助于选择正确的 fine-tune 策略 (冻结/解冻哪些层).

## 与知识库其他内容的关联
- 15_DQN: 证明 representation 可以从像素端到端学习, 但没有 transfer
- 17_Transformer / BERT / GPT 系列: 将 pre-train+fine-tune 推向极致
- Representation Learning (12): transfer 的核心机制就是学好的 representation
