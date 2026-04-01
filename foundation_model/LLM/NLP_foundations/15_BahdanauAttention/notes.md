# Bahdanau Attention: Neural MT by Jointly Learning to Align and Translate -- 学习笔记
> 一句话: 在 encoder-decoder 中引入 soft attention, 让 decoder 每步动态查询输入序列的相关位置, 彻底解决了固定向量瓶颈.
> 论文: Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio, 2015, ICLR 2015

## 这篇论文解决了什么问题
Seq2Seq (Sutskever 2014) 把整个输入句子压缩成一个固定长度向量, decoder 从这个向量生成翻译. 但 Cho et al. 实验表明, 句子一长 (>30 词) BLEU 就急剧下降 -- 单个向量装不下一句长话的全部信息. 本文要解决的就是这个 information bottleneck: 让 decoder 不再依赖单一固定向量, 而是能回看输入的任意位置.

## 核心想法 (用直觉解释)
翻译时, 人不会先把整句话记成一个 "压缩码" 再翻, 而是边翻边回头看原文相关部分. Attention 就是模拟这个过程: decoder 每生成一个词时, 先用当前 decoder 状态去和 encoder 每个位置 "打分" (alignment score), softmax 归一化后加权求和得到一个 context vector, 这个 context 包含了 "当前最该关注的输入信息". 这是 soft alignment -- 可微分, 端到端训练.

## 关键设计决策
- **Additive attention**: e_ij = v^T tanh(W * s_{i-1} + U * h_j), 用一个前馈网络计算 decoder 隐状态 s_{i-1} 和 encoder annotation h_j 的匹配分数. 这个 alignment model 不是预先定义的, 而是和翻译模型联合训练
- **Bidirectional RNN encoder**: 用 BiGRU 编码输入, 每个位置的 annotation h_j = [forward_h_j; backward_h_j] 同时包含前后文信息, 因为 RNN 更擅长编码局部上下文
- **每步不同的 context vector**: c_i = sum(alpha_ij * h_j), 每个 target word 有自己的 attention 分布, 不再共享单一固定向量. 这意味着信息存储从 O(1) 变成了 O(T_source)

## 这篇论文之后发生了什么
Luong (2015) 简化为 dot-product attention (去掉前馈网络, 直接点积). Vaswani (2017) 推广为 self-attention: Q/K/V 框架 + multi-head + scaled dot-product, 诞生 Transformer. Self-attention 让序列内部也能互相 attend (不只是 decoder→encoder), 从根本上取代了 RNN. Attention 可视化催生了可解释性研究. Cross-attention 成为所有多模态模型 (CLIP, LLaVA, VLA) 的标准组件.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Attention = 按需检索, 解决固定容量瓶颈 | robot policy 用单帧 latent 在长时域任务失败, 解法就是 attention over observation history (ACT, pi0) |
| 2 | Soft alignment 保留梯度, 端到端可训练 | 同理 NeRF/3DGS 的 differentiable rendering 优于 discrete feature matching, 都是 "soft 操作保留梯度" |
| 3 | Cross-attention 连接不同模态 | RT-2 中 language 与 vision 交互, Octo 中 task token 与 observation token 交互, 都是 Bahdanau 的后裔 |
| 4 | Attention 权重提供免费的可解释性 | 可视化 VLA 的 cross-attention 可诊断 policy 是否关注正确物体区域, 对 debug sim2real 有直接帮助 |
