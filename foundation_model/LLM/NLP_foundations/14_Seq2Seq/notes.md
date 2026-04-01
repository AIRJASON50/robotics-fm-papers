# Sequence to Sequence Learning (Sutskever et al., 2014) -- Takeaway Notes

> 一句话: 用两个 deep LSTM 组成 encoder-decoder, 首次证明纯神经网络可以端到端完成变长序列翻译, 性能超过传统 phrase-based SMT (统计机器翻译) 系统.

## 核心贡献
- 提出 encoder-decoder 架构: encoder LSTM 将变长输入压缩为固定维度向量 (8000-dim), decoder LSTM 从该向量自回归生成输出序列
- 发现 **反转输入序列** 这一简单技巧大幅提升性能 (BLEU 从 25.9 升到 30.6), 原因是缩短了 source 起始词与 target 起始词之间的 "minimal time lag"
- 证明 depth 很重要: 4 层 LSTM 显著优于浅层, 每多一层 perplexity 降约 10%
- 384M 参数的 LSTM ensemble 在 WMT'14 En-Fr 上达到 34.81 BLEU, 首次让纯神经翻译系统超越 phrase-based SMT baseline (33.3)

## 为什么重要
Seq2Seq 确立了 **encoder-decoder** 作为序列转换的通用架构范式. 这个框架后来被 Transformer
继承 (Vaswani 2017), 并成为 GPT (decoder-only) 和 BERT (encoder-only) 等所有大模型的祖先架构.

更深远的意义: 它证明了 **固定维度的向量可以编码一个变长句子的完整语义**. 这个 "信息瓶颈"
假设虽然后来被 attention 机制部分推翻 (Bahdanau 2015), 但 "先压缩再生成" 的思路至今
仍是 VAE, Diffusion Policy, latent-conditioned 策略等方法的核心.

## 对你 (RL->FM) 的 Takeaway
- **Encoder-Decoder = 感知-决策**: robotics FM 的标准架构 (RT-1/2, Octo, pi0) 本质上就是
  Seq2Seq -- vision encoder 压缩观测, action decoder 生成动作序列. 理解 Seq2Seq 就理解了
  这些模型的骨架
- **信息瓶颈的利弊**: 固定维度向量在短序列上够用, 但长序列会丢失细节. 同理, robot policy
  如果只用单帧 latent 决策, 长时域任务必然失败 -- 这正是 attention 和 history stacking 被引入的原因
- **反转输入的启示 -- 数据增强的力量**: 不改架构, 仅调整输入顺序就能大幅提升性能. 在 sim2real
  中, domain randomization 和 observation augmentation 起的是同样的作用
- **Beam search 与 action sampling**: Seq2Seq 用 beam search 在解码时保留多个候选. 类比
  Diffusion Policy 的多步 denoising 或 MPC 的 sampling-based planning
- **Deep 优于 shallow**: 4 层 LSTM > 1 层 LSTM, 这是后来所有 "scale depth" 工作的早期信号

## 与知识库其他内容的关联
- 13_Word2Vec: Seq2Seq 的 encoder 输出可看作 "句子级 Word2Vec" -- 把整句压缩为向量
- 15_BahdanauAttention: 直接解决 Seq2Seq 固定向量瓶颈问题, 引入 attention 让 decoder 回看 encoder 每一步
- foundations/17_Transformer: 用 self-attention 取代 LSTM, 但保留了 encoder-decoder 框架
- robotics/policy_learning/DiffusionPolicy: action chunk 生成本质上是 Seq2Seq 的 decoder, 只是用 diffusion 替代了自回归
- robotics/families/pi_Series: pi0 的 flow matching action head 就是一个 continuous Seq2Seq decoder
