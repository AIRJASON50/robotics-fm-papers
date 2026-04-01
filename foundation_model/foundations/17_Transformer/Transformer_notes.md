# Attention Is All You Need -- 学习笔记

> 一句话: 完全抛弃 RNN/CNN, 仅用 attention 机制构建 Transformer, 实现完全并行化的序列建模, 开启了从 NLP 到机器人的 foundation model 时代.
> 论文: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, NeurIPS 2017
> 引用量级: ~140,000+

## 这篇论文解决了什么问题

2017 年之前, 序列建模 (机器翻译、语言模型) 的主力架构是 RNN/LSTM. RNN 有一个根本瓶颈: h_t 依赖 h_{t-1}, 计算必须按时间步串行, 无法并行化. 序列越长训练越慢, 而且长距离依赖难以捕捉 (梯度消失). CNN 方案 (ConvS2S, ByteNet) 试图解决并行化问题, 但关联远距离 token 需要堆叠多层, 路径长度为 O(N) 或 O(log N).

## 核心想法 (用直觉解释)

Transformer 的核心洞察是: 序列中任意两个位置之间的关系, 可以通过一次 attention 操作直接计算, 不需要经过中间步骤. 这就像一个会议室里所有人可以同时互相交谈, 而不是像 RNN 那样必须一个接一个传话.

具体来说, 每个 token 生成三个向量: Query (我在找什么), Key (我是什么), Value (我能提供什么). 通过 Q 和所有 K 的点积计算相关度, softmax 归一化后加权求和 V, 得到该 token 的新表示. 关键公式: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`. 除以 sqrt(d_k) 是为了防止点积过大导致 softmax 饱和 (梯度消失).

Multi-Head Attention 将 Q/K/V 投影到 h=8 个不同子空间, 让不同 head 学习不同类型的关系 (语法、语义、位置). 单头 attention 被迫把所有信息压缩到一组投影中, 论文消融实验显示单头比 8 头差 0.9 BLEU.

由于 attention 不含任何位置信息, 需要额外注入 positional encoding (正弦/余弦函数), 使模型能区分 token 顺序.

## 关键设计决策

1. **去除循环, 完全并行化**: RNN 需要 O(N) 顺序步, Transformer 只需 O(1). 代价是 attention matrix 的内存为 O(N^2), 但这个 trade-off 使得大规模训练 (GPT-3, 175B 参数) 成为可能 -- RNN 架构下根本不可能.

2. **Encoder-Decoder + 三种 attention 用法**: Encoder 用 self-attention (全局可见), Decoder 用 masked self-attention (只看已生成的 token, 保证 auto-regressive) + cross-attention (attend to encoder 输出). 这三种 attention 是后续所有变体的基础: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5).

3. **Scaled Dot-Product + Residual + LayerNorm**: 每个 sub-layer 都有残差连接和 LayerNorm, 确保深层网络 (6 层 encoder + 6 层 decoder) 的梯度稳定传播. FFN 层 (d_ff=2048) 提供非线性变换能力.

## 这篇论文之后发生了什么

Transformer 成为几乎所有 AI 领域的基础架构: BERT (encoder-only, 理解), GPT 系列 (decoder-only, 生成), ViT (图像), Decision Transformer (RL trajectory), Diffusion Transformer/DiT (扩散模型). 机器人领域的 VLA (Vision-Language-Action) 模型, 如 pi_0 和 GR00T N1, 本质上都是 Transformer 在不同模态上的应用.

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | Attention 是通用的序列建模工具, 不限于语言 -- token 可以是文字、image patch、action chunk | 你的 sim2real policy 正在从 MLP 向 Transformer-based policy 迁移, 理解 attention 是理解 VLA 架构的前提 |
| 2 | Decoder-only 架构 (GPT 路线) 统治了生成式模型, 机器人 action generation 也走这条路 | Decision Transformer, pi_0 的 action expert, GR00T 的 DiT head 都是 decoder-only 思路 |
| 3 | Positional encoding 的设计决定了模型能处理的序列长度和泛化能力 | 机器人 policy 中 action horizon / chunk size 的设计与此直接相关 |
