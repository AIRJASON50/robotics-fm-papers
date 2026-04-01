# BLIP-2: Bootstrapping Language-Image Pre-training -- 学习笔记

> 一句话: 用轻量级 Q-Former 桥接冻结的 image encoder 和冻结的 LLM, 以极低训练成本实现 SOTA 视觉-语言能力。
> 论文: Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi (Salesforce Research), 2023, ICML 2023

## 这篇论文解决了什么问题

Vision-Language Pre-training (VLP) 成本越来越高: Flamingo 用 80B 参数端到端训练, 计算代价极大。同时, 预训练好的 image encoder (CLIP ViT) 和 LLM (OPT, FlanT5) 各自已经非常强, 但它们之间存在 **modality gap** -- LLM 从未见过图像。

之前的方法 (Frozen, Flamingo) 只用 language modeling loss 做对齐, 论文认为这不够。核心挑战: **如何在完全冻结两个大模型的前提下, 用一个小模块弥合视觉和语言的 modality gap?**

## 核心想法 (用直觉解释)

**设计一个 "信息瓶颈" -- Q-Former (Querying Transformer, 仅 188M 参数), 作为冻结视觉模型和冻结语言模型之间的翻译器。**

Q-Former 有 32 个 learnable query embeddings (32 x 768), 通过 cross-attention 从冻结 image encoder 的输出 (如 257 x 1024) 中"提问"并提取最有用的视觉信息。输出是压缩过的 32 个 visual token -- 这就是瓶颈, 强制 query 只保留与语言相关的视觉信息。

训练分两个阶段:
- **Stage 1 (representation learning)**: Q-Former + 冻结 image encoder, 用 ITC + ITM + ITG 三个目标学习"什么视觉信息和文本有关"
- **Stage 2 (generative learning)**: Q-Former 输出通过 FC 层投影后作为 soft visual prompt 输入冻结 LLM, 学习"怎么把视觉信息翻译成 LLM 能理解的 token"

## 关键设计决策

**1. Q-Former 的双 Transformer 架构**

两个子模块共享 self-attention 层: (a) image transformer -- query 通过 cross-attention 与冻结视觉特征交互; (b) text transformer -- 可做 encoder 也可做 decoder。初始化自 BERT_base, cross-attention 层随机初始化。通过不同 attention mask 控制 query-text 交互, 一个架构服务三个预训练目标。

**2. 三个 Stage 1 目标 + attention mask 策略**

- **ITC (Image-Text Contrastive)**: query 和 text 互不可见 (unimodal mask), 各自编码后对比相似度
- **ITG (Image-grounded Text Generation)**: query 可看其他 query 但不可看 text, text 用 causal mask -- 迫使所有视觉信息必须经过 query 传递
- **ITM (Image-Text Matching)**: query 和 text 互相可见 (bi-directional mask), 做细粒度二分类匹配

三个目标互补: contrastive 学粗粒度对齐, matching 学细粒度对齐, generation 学信息完整性。

**3. 信息瓶颈设计**

32 个 query (32 x 768 = 24K 维) 远小于原始视觉特征 (257 x 1024 = 263K 维), 压缩比约 10x。迫使 query 只保留对语言任务有用的信息, 同时大幅减少 LLM 需要处理的 visual token 数量。

**4. Stage 2: Soft visual prompt 注入 LLM**

Q-Former 输出经 FC 层线性投影到 LLM 的 embedding 维度, 然后 prepend 到 text embedding 前。对 decoder LLM (OPT) 用 language modeling loss; 对 encoder-decoder LLM (FlanT5) 用 prefix language modeling loss。

**5. 冻结策略的好处**

(a) 极大减少训练参数 (188M vs Flamingo 10B+); (b) 避免 catastrophic forgetting; (c) 可以随时换更强的 backbone 或 LLM -- "模块化升级"。结果: 在 VQAv2 zero-shot 上超 Flamingo80B 8.7%, 可训参数少 54x。

## 这篇论文之后发生了什么

- **定义了 VLM 架构模板**: LLaVA, InstructBLIP, Qwen-VL 都采用 "冻结视觉编码器 + bridge + LLM" 范式
- **LLaVA 简化 bridge**: 用 MLP 替代 Q-Former, 发现配合 LLM 微调也能工作, 引发 "bridge 需要多复杂" 的讨论
- **Q-Former 概念影响 robot VLA**: 用少量 learnable token 从视觉中提取动作相关信息的思想

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|----------|
| 1 | **信息瓶颈迫使模型学 task-relevant representation** -- 32 个 query 比 257 个 patch token 少, 但效果更好 | Robot policy 不需要全部图像信息, 类似 bottleneck 可减少 observation 冗余 |
| 2 | **分阶段训练连接异构模型** -- 先学 representation, 再接 generative model | Robotics FM 也面临视觉 encoder 和 action decoder 异构的问题, 可分阶段对齐 |
| 3 | **冻结大模型 + 训练小 bridge 是高效范式** -- 54x 少于 Flamingo 可训参数, 效果更好 | 对 robot lab 尤其重要: 计算有限时冻结 backbone + 训 adaptation layer 最现实 |
| 4 | **Soft visual prompt 让 LLM "看到" 图像** -- 视觉信息作为 prefix token 注入 | VLA (RT-2, OpenVLA) 的核心: 把视觉转化为 token 序列让 LLM 输出动作 |
| 5 | **模块化允许独立升级** -- 换更好的 ViT 或 LLM 不需重训全部 | Robot FM 可独立升级视觉 backbone (CLIP->DINOv2) 和 action head (MLP->Diffusion) |
