# Attention Is All You Need -- 分析笔记

Vaswani et al., Google Brain / Google Research / University of Toronto, 2017 (arXiv:1706.03762)

## 1. Core Problem

序列转换 (sequence transduction) 任务 (机器翻译、文本生成等) 此前依赖 RNN/LSTM。RNN 的根本限制: **顺序计算** -- h_t 依赖 h_{t-1}，无法并行化，长序列训练极慢。

Transformer 的目标: **完全去除循环结构**，仅用 attention 机制处理序列，实现完全并行化训练。

## 2. Method

### 2.1 Self-Attention (核心操作)

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

- Q (Query), K (Key), V (Value) 都是输入序列的线性投影
- `Q @ K^T`: 计算每对 token 之间的相关度 (N x N attention matrix)
- `/ sqrt(d_k)`: 缩放防止 softmax 饱和
- `softmax(...)`: 归一化为概率分布
- `@ V`: 加权求和得到输出

**直觉**: 每个 token "查询"所有其他 token 的相关性，然后聚合它们的信息。距离不影响计算 -- 这是对 RNN 的根本改进。

### 2.2 Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

不同的 head 学习关注不同类型的关系 (语法、语义、位置等)。论文用 h=8 heads, d_k=d_v=d_model/h=64。

### 2.3 Positional Encoding

Transformer 没有循环/卷积，无法感知 token 顺序。用 sinusoidal 函数编码位置:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每个位置得到一个唯一的 d_model 维向量，加到 token embedding 上。

### 2.4 Encoder-Decoder 结构

- **Encoder**: N=6 层, 每层 = Multi-Head Self-Attention + Feed-Forward + 残差连接 + LayerNorm
- **Decoder**: N=6 层, 每层 = Masked Self-Attention (causal) + Cross-Attention (attend to encoder) + FFN
- **FFN**: 两层全连接, `max(0, xW1+b1)W2+b2`, 隐藏维度 d_ff=2048

### 2.5 后续变体

| 变体 | 用途 | 代表 |
|------|------|------|
| Encoder-only | 理解/分类 | BERT |
| Decoder-only | 生成 | GPT 全系列 |
| Encoder-Decoder | 翻译/摘要 | T5, BART |

机器人领域几乎全部使用 **decoder-only** 变体 (GPT 路线)。

## 3. Key Designs

### 3.1 去除循环 -- 完全并行化

RNN: O(N) 顺序步 (每步依赖前一步)。Transformer: O(1) 并行步 (所有 token 同时计算)。代价: 内存 O(N^2) (attention matrix)。

这使得 GPT-3 (96 层, 175B 参数) 的训练成为可能 -- RNN 架构下不可能实现这个规模。

### 3.2 Scaled Dot-Product Attention

除以 sqrt(d_k) 看似简单，但至关重要。论文 Section 3.2.1 解释: 当 d_k 很大时，Q@K^T 的方差为 d_k，softmax 会进入饱和区 (梯度接近 0)。缩放后方差为 1，训练稳定。

### 3.3 Multi-Head > Single-Head

论文 Table 3 消融: 单头 attention (h=1) 比 8 头差 0.9 BLEU。原因: 单头被迫在一组 QKV 投影中编码所有关系，信息瓶颈。

## 4. Experiments

WMT 2014 翻译:

| 任务 | Transformer | 之前 SOTA | 训练成本 |
|------|------------|----------|---------|
| EN-DE | 28.4 BLEU | 26.4 | 3.5 days, 8 P100 |
| EN-FR | 41.8 BLEU | 41.0 | 比之前 SOTA 训练成本低 1/4 |

## 5. 对机器人领域的意义

Transformer 是所有后续工作的根基:

| 后续工作 | 继承了什么 |
|---------|---------|
| GPT 系列 | Decoder-only Transformer + autoregressive generation |
| Decision Transformer | 把 RL trajectory 当作 token 序列, 用 Transformer 建模 |
| Diffusion Policy | 用 Transformer 做 denoising network (替代 U-Net) |
| pi_0 | PaliGemma VLM (Transformer) + action expert (也是 Transformer) |
| GR00T N1 | VLM (Transformer) + DiT action head (Diffusion Transformer) |

**核心迁移逻辑**: Transformer 证明了 attention 是通用的序列建模工具，不限于语言 -- 可以处理图像 patch 序列 (ViT)、动作序列 (Decision Transformer)、噪声动作序列 (Diffusion Transformer)。

## 6. 代码参考

原论文的代码在 `tensor2tensor` 库中 (已归档)。最佳的 decoder-only 实现参考:
- `LLM/gpt-2/src/model.py` -- 175 行 TensorFlow, 完整的 decoder-only Transformer
- `methods/24_pi0/openpi/src/openpi/models/gemma.py` -- 现代 JAX 实现, 含 MQA + dual-expert
