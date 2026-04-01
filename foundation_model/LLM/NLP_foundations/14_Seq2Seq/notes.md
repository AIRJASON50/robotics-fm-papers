# Seq2Seq: Sequence to Sequence Learning with Neural Networks -- 学习笔记
> 一句话: 用两个 deep LSTM 组成 encoder-decoder, 首次证明纯神经网络可以端到端完成变长序列翻译, 超越统计机器翻译系统.
> 论文: Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google), 2014, NeurIPS 2014

## 这篇论文解决了什么问题
DNN 只能处理固定维度输入输出, 但翻译、语音等核心问题是变长序列到变长序列的映射. 传统 SMT (Statistical Machine Translation) 由多个独立调优的子模块拼凑而成. 本文要证明: 一个端到端的神经网络可以直接学会 sequence-to-sequence 映射, 无需手工特征或流水线.

## 核心想法 (用直觉解释)
一个 LSTM (encoder) 从头到尾读完输入句子, 最后一步的 hidden state 就是整句话的 "压缩码"; 另一个 LSTM (decoder) 从这个压缩码出发, 逐词生成翻译直到输出 \<EOS\>. 关键 trick: 把输入句子反转 (abc -> cba), 这样源句子开头的词在时间上离 decoder 更近, SGD 更容易 "建立联系". 用 4 层 deep LSTM (384M 参数) + beam search 解码, 在 WMT'14 英法翻译上达到 34.81 BLEU, 首次超过 phrase-based SMT baseline.

## 关键设计决策
- **Encoder-Decoder 分离**: 用不同参数的 LSTM 分别做编码和解码, 增加参数量但几乎不增加计算量, 也让多语言扩展更自然
- **反转输入序列**: BLEU 从 25.9 升到 30.6, 原理是缩短源语言起始词与目标语言起始词之间的 "minimal time lag", 让梯度更容易流通
- **Deep LSTM (4 层)**: 每多一层 perplexity 降约 10%. 深度比宽度更有效, 这是 "scale depth" 的早期证据
- **Beam search 解码**: 维护 B 个候选序列, 每步扩展所有可能词再保留 top-B. B=2 就能捕获大部分收益, B=12 接近最优

## 这篇论文之后发生了什么
Bahdanau (2015) 指出固定向量是瓶颈, 引入 attention 让 decoder 回看 encoder 每一步. Vaswani (2017) 用 self-attention 取代 LSTM, 保留 encoder-decoder 框架, 诞生 Transformer. GPT 走 decoder-only, BERT 走 encoder-only, 但骨架都源于 Seq2Seq. LSTM 被 Transformer 全面取代. 在 robotics 中, "encoder 压缩观测 + decoder 生成动作" 仍是 RT-1/2, Octo, pi0 等 VLA 的标准架构.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Encoder-Decoder = 感知-决策的祖先架构 | RT-1/2, Octo, pi0 本质都是 Seq2Seq: vision encoder 压缩观测, action decoder 生成动作序列 |
| 2 | 固定向量瓶颈在长序列上必然失败 | 用单帧 latent 做长时域控制注定不行, 这正是 attention 和 history stacking 被引入的原因 |
| 3 | 反转输入 = 不改架构的数据增强可以带来巨大收益 | 类比 sim2real 中的 domain randomization -- 不改模型, 只改数据呈现方式 |
| 4 | Deep > Shallow, 且收益可叠加 | 4 层 LSTM 远优于 1 层, 后来 scaling law 证实这一点: 深度和参数量是性能的关键驱动力 |
