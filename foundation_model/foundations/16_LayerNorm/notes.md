# Layer Normalization (Ba, Kiros & Hinton, 2016) -- Takeaway Notes

> 一句话: 对单个样本的所有神经元做归一化 (而非跨 batch), 解决了 BatchNorm 在序列模型和小 batch 下的缺陷.

## 核心贡献
- 归一化维度从 batch 维 (BatchNorm) 改为 feature 维: 对每个样本独立计算 mean/var
- 不依赖 batch 统计量, train 和 eval 行为完全一致, 无需维护 running stats
- 在 RNN/LSTM 上首次验证有效, 后来成为 Transformer 的标准组件

## 为什么重要
LayerNorm 是 Transformer 架构的必备组件 -- 没有它, self-attention 的训练极不稳定.
GPT, BERT, ViT 全部使用 LayerNorm. 相比 BatchNorm, 它对 batch size 无依赖,
这让单样本推理、变长序列、RL 中的小 batch 场景都能正常工作.

## 对你 (RL->FM) 的 Takeaway
- 如果你在 PPO 的 actor/critic MLP 中加 normalization, 应该用 LayerNorm 而非
  BatchNorm -- RL 的 batch 统计量不稳定 (分布一直在变), BatchNorm 会引入额外噪声.
- 理解 Pre-LN vs Post-LN Transformer 的区别: Pre-LN (GPT-2 style) 训练更稳定,
  这对你理解 robotics FM (如 RT-2, Octo) 的架构选择有帮助.

## 与知识库其他内容的关联
- 15_BatchNorm: LayerNorm 是直接针对 BatchNorm 缺陷的改进
- 17_Transformer: Transformer 的 "Add & Norm" 中 Norm 就是 LayerNorm
- 15_Adam: LayerNorm + Adam + residual connection 构成 Transformer 可训练的三板斧
