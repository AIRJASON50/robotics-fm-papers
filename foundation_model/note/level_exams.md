# Foundation Model 学习路线 -- 升级考试题库

你是一个机器人领域的研究者。你学习 CS (NLP/CV) foundation model 的目的是：
- 理解 VLA (Vision-Language-Action) 模型的设计原理，用于灵巧手操控
- 理解编码/表征方法为什么重要，以及如何从语言/视觉迁移到机器人动作
- 建立从 Transformer -> GPT -> CLIP -> Diffusion -> VLA 的完整技术脉络
- 不是成为 NLP/CV 专家，而是能读懂 VLA 论文、理解架构选择背后的 why

每个 Level 学习完毕后，完成对应考试。80% 以上进入下一级。答题凭理解，不翻资料。

---

## Level 0 -> Level 1 入门考试 (表征学习 + Transformer)

1. 什么是表征学习？它和特征工程的根本区别是什么？

2. 分布式表征的泛化能力从何而来？为什么见过"红球能滚"就能推断"蓝球也能滚"，背后的数学机制是什么？

3. Foundation model 预训练阶段的目标是"保留尽可能多的信息"还是"筛选出有用的特征"？fine-tuning 阶段呢？如果只做其中一个会怎样？

4. Transformer 相比 RNN 的核心突破是什么？不是"效果更好"，而是它使得什么成为可能？

5. Self-attention 中 Q、K、V 各自的角色是什么？如果只用一个向量同时承担三个功能会怎样？

6. Attention 有自己独立的 loss 吗？如果 attention 学到了错误的关注模式，是什么机制在纠正它？

7. 好的表征需要满足"满秩"吗？好表征的核心标准到底是什么？

8. 多模态数据（图像+文本+动作）需要用一个统一的编码器，还是可以用多个？pi_0 怎么做的？为什么？

9. 为什么说 Transformer 是一种"信息路由"创新而不是"loss"创新？Transformer 是否绑定了特定的训练目标？

10. 从表征学习的角度，pi_0 为什么要把 VLM backbone 和 action expert 分成两组权重？如果用一组会有什么问题？

---

## Level 1 -> Level 2 入门考试 (LLM 原理)

1. GPT-1 到 GPT-3 的核心范式演进是什么？"pre-train + fine-tune" -> "in-context learning" 意味着什么？

2. In-context learning 时模型权重更新了吗？Few-shot 的 examples 在推理中起什么作用？

3. Scaling law 的核心结论是什么？Chinchilla 修正了什么？"模型和数据等比缩放"对你理解 pi_0 的 10k 小时数据有什么启示？

4. GPT-3 之后的 Codex、WebGPT、InstructGPT、ChatGPT 都是基于同一个 base model 做 fine-tune。这个 "一个 base model, 多种 fine-tune" 的模式对机器人领域有什么直接启示？

5. 自回归生成 (autoregressive generation) 的基本流程是什么？它的根本限制是什么？这个限制如何导致了 RT-2 在灵巧操控上的困难？

6. MoE (Mixture of Experts) 的核心思想是什么？它和 pi_0 的 dual-expert 架构有什么联系？

7. GPT-4 相比 GPT-3 增加了什么能力？RLHF 做了什么？为什么 RLHF 对机器人 foundation model 也重要？

8. Token embedding 和上下文化表征有什么区别？同一个词 "bank" 在 "river bank" 和 "bank account" 中，embedding 相同吗？Transformer 处理后呢？

9. 给定固定的计算预算，应该训一个大模型少数据，还是小模型多数据？依据是什么？

10. GPT 系列从 GPT-1 到 GPT-4 逐步减少了公开信息（GPT-4 几乎什么都不公开）。开源 LLM (Qwen, LLaMA) 的出现为什么重要？对机器人领域有什么影响？

---

## Level 2 -> Level 3 入门考试 (视觉-语言 + 生成模型)

1. CLIP 是如何对齐视觉和语言的？contrastive loss 的直觉是什么？为什么它能实现 zero-shot 图像分类？

2. DDPM 的前向过程和反向过程分别做什么？简化 loss L_simple 为什么有效？

3. Flow matching 和 DDPM 的核心区别是什么？为什么 pi_0 选择 flow matching 而不是 DDPM？

4. CLIP 的视觉编码器学到的表征和 ImageNet 预训练的 ResNet 表征有什么本质区别？哪个对机器人更有用？为什么？

5. 从图像 diffusion 到 Diffusion Policy，核心概念如何对应？（噪声图像 -> ?，去噪网络 -> ?，生成的图片 -> ?）

6. 为什么 VLA 模型需要视觉编码器（如 SigLIP）而不是直接把原始像素喂进 Transformer？从计算量和表征质量两个角度回答。

7. ViT 是如何把 Transformer 从 NLP 迁移到 CV 的？"An image is worth 16x16 words" 这个比喻的含义是什么？

8. DiT (Diffusion Transformer) 用 Transformer 替换 U-Net 作为 diffusion 的 backbone，这和 GR00T N1 的 action head 有什么关系？

9. 对比学习 (contrastive learning) 和重建学习 (reconstruction, 如 MAE) 学到的表征有什么不同？各自保留了什么、丢弃了什么？

10. 如果你要为灵巧手操控设计一个视觉编码器，你会选 CLIP 预训练的 ViT 还是 MAE 预训练的 ViT？为什么？

---

## Level 3 -> Level 4 入门考试 (RL/Robotics meets Transformer)

1. Decision Transformer 把 RL 重新定义为什么问题？它不需要 value function 和 policy gradient，那靠什么做决策？

2. Diffusion Policy 相比传统 behavior cloning（直接回归动作均值）的核心优势是什么？什么是 multimodal action distribution？举一个机器人场景说明为什么它重要。

3. Action chunking 解决了什么问题？为什么一次预测 50 步动作比逐步预测更好？有什么代价？

4. RT-1 到 RT-2 的核心跃迁是什么？RT-2 的 action tokenization (256 bins 离散化) 有什么优缺点？

5. World model (DreamerV3) 和 imitation learning (Diffusion Policy / pi_0) 是两条不同的路线，各自的核心假设和优势场景是什么？

6. Diffusion Policy 用 DDPM 做动作生成，pi_0 用 flow matching 做动作生成，RT-2 用 autoregressive token prediction 做动作生成。三种方法各自的优劣是什么？

7. 为什么 Diffusion Policy 需要 receding horizon control？如果不用，直接执行完整个 action chunk 会怎样？

8. RT-1 用了 130k 真实机器人 episode，这个数据规模意味着什么？和 GPT-3 的 300B tokens 相比，机器人数据的稀缺性带来了什么挑战？

9. DreamerV3 在 150+ 任务上用单一超参数配置就能工作，这说明了什么？它的世界模型和物理引擎（如 MuJoCo）的区别是什么？

10. 如果你要为你的灵巧手项目选择一个 policy 架构，你会选 Diffusion Policy、ACT、还是直接用 VLA？各自的 tradeoff 是什么？

---

## Level 4 毕业考试 (VLA 统一模型)

1. pi_0 的架构中，VLM backbone 和 action expert 通过什么机制交互？两者的参数分别负责什么？

2. pi_0 的 flow matching loss 和 GPT 的 next-token prediction loss 各自适合什么类型的输出？为什么 pi_0 不用 autoregressive 方式生成动作？

3. GR00T N1 的 "双系统" 设计 (System 1 + System 2) 的直觉是什么？为什么不把 VLM 和 action generation 合成一个？

4. Cross-embodiment training 是怎么处理不同机器人的不同自由度和动作空间的？pi_0 的具体做法是什么？

5. Pre-training 的低质量大规模数据和 post-training 的高质量小规模数据各自提供了什么？为什么缺一不可？

6. pi_0 推理时用 KV cache 优化了什么？10 步 flow matching 积分中，哪部分被缓存、哪部分每步重新计算？

7. pi_0-FAST 用 autoregressive token prediction 替换了 flow matching，这是开倒车吗？它在什么场景下比 pi_0 更好？

8. Octo (diffusion action head) -> OpenVLA (discrete action tokens) -> pi_0 (flow matching action expert) 三个开源 robot policy 的架构选择各有什么 tradeoff？为什么 pi_0 的团队（部分是 Octo 原班人马）放弃了 Octo 的方案？

9. 从 Bengio 2012 的表征学习到 pi_0 2024 的 VLA，画出完整的技术迁移链路（包含 ViT, CLIP, Flow Matching, ACT, Open X-Embodiment 等中间节点），每一步说明继承了什么、创新了什么。

10. 如果你要把 pi_0 的架构应用到你的灵巧手操控任务，需要做哪些适配？需要多少数据？主要的技术挑战是什么？

11. Foundation model 在机器人领域的 scaling law 和在 NLP 中的有什么异同？机器人数据的特殊性会如何影响 scaling 行为？

10. Foundation model 在机器人领域的 scaling law 和在 NLP 中的有什么异同？机器人数据的特殊性会如何影响 scaling 行为？

---

## 考试规则

- 每题 10 分，每级满分 100 分
- 80 分以上: 进入下一 Level
- 60-80 分: 重读对应材料中标注的重点 section，然后重考
- 60 分以下: 从头阅读该 Level 的全部材料
- 答题时不翻资料，凭理解回答
- 允许用不精确但方向正确的直觉描述

---

# 参考答案

## Level 0 参考答案

**1.** 表征学习是让算法自动学习数据到有用空间的映射，替代人工设计特征。特征工程需要领域专家手动定义（如 SIFT、HOG），表征学习通过优化目标让网络自动发现有用的特征。

**2.** 分布式表征 + 参数共享。[红,球] 和 [蓝,球] 共享形状维度的值，作用在形状维度上的权重 w_形状 是跨所有样本共享的参数。w_形状 从 [红,球] 学到"球形->能滚"，由于 [蓝,球] 的形状维度值相同，w_形状 对它产生相同的输出——这是数学上的必然，不需要见过蓝球。这把维度诅咒（指数增长的样本需求）变成了组合祝福（线性增长）。

**3.** 预训练 = 保留尽可能多的信息（最大化互信息 I(x;z)）。Fine-tuning = 针对下游任务调整关注重点（放大需要的维度，压制不需要的）。只做预训练：什么都会一点但不精通，且缺乏任务特化的流畅性。只做 fine-tuning：数据量不足容易过拟合，且不会从错误中恢复（缺少预训练中大量低质量数据提供的纠错/恢复行为）。

**4.** 把串行计算变成并行计算。RNN 必须逐步计算（h_t 依赖 h_{t-1}），GPU 利用率极低。Transformer 的 attention 所有位置同时计算（一次矩阵乘法），GPU 利用率接近 100%。这使得训练 175B 参数的模型在合理时间内成为可能，而 RNN 架构下同样规模需要几百年。

**5.** Q = "我在找什么信息"，K = "我能提供什么信息"，V = "我的实际信息内容"。Q@K 算相关性分数，scores@V 按相关性聚合信息。如果只用一个向量，"被检索"和"提供信息"两个功能互相制约——K 需要编码抽象的类别/角色信息用于匹配，V 需要编码具体的内容用于下游计算，两者关心的东西不同。

**6.** 没有独立 loss。Attention 是网络内部的计算模块，它的权重通过最终 loss（如 next-token prediction）的反向传播间接调整。如果 attention 关注错了 token 导致预测错误 -> loss 增大 -> 梯度回传调整 W_Q, W_K, W_V -> attention 模式改变。和 RL 中 policy network 的隐藏层一样——没有独立 eval，靠 reward 信号反传。

**7.** 不需要满秩。实际表征空间维度（如 768）远大于语义因素数量（可能几十个）。核心标准是：用最少的维度编码最多的独立语义因素，且保持平滑（输入小变化 -> 表征小变化）和可区分（不同语义的样本在空间中分开）。

**8.** 实际是混合的：各模态有独立编码器（SigLIP 编码图像、Tokenizer 编码文本、Linear 编码状态、MLP 编码动作），再映射到共享 Transformer 中通过 attention 交互。不能用一个编码器，因为不同模态的原始数据结构差异太大（图像 2D 像素 vs 文本离散 token vs 关节角连续向量），需要各自的编码器先映射到统一维度。

**9.** Transformer 只提供了信息路由机制（self-attention），不绑定任何特定 loss。同一个 Transformer 架构，接 next-token prediction loss 就是 GPT，接 masked prediction loss 就是 BERT，接 MSE 去噪 loss 就是 Diffusion Policy，接 flow matching loss 就是 pi_0。它是通用计算模块，不是训练范式。

**10.** VLM backbone 的权重已编码了互联网规模的视觉-语言知识。如果让动作 token（VLM 从未见过的新模态）直接走这组权重，动作训练的梯度会破坏已学到的视觉-语言表征（灾难性遗忘）。分开后：动作梯度只修改 action expert 权重，VLM 权重不变。两者通过 attention 共享需要的信息，但参数互不干扰。此外，视觉-语言表征和动作表征需要的好性质不同（前者需要语义对齐，后者需要平滑连续），一组参数无法同时优化两种性质。

---

## Level 1 参考答案

**1.** GPT-1: pre-train + fine-tune (需要 task-specific 标注数据)。GPT-2: pre-train only, zero-shot (去掉 fine-tune，直接 prompt)。GPT-3: in-context learning (不更新权重，在 context 中放几个 examples 就能做新任务)。范式转变意味着：模型在预训练中已经隐式学会了"如何从 context 中提取任务定义"，这是一种涌现能力。

**2.** 权重完全不更新。Few-shot examples 作为 context 的一部分被 attention 处理——模型通过 attention 关注这些 examples 的模式（输入-输出对应关系），推断出当前任务的规则，然后应用到新输入上。本质是 pattern matching，不是学习。

**3.** Kaplan (2020) 发现 loss 与 compute 之间存在 power law 关系，但建议主要增大模型。Chinchilla 修正：给定固定 compute，模型和数据应等比缩放（每增大模型 2x，数据也要增大 2x）。对 pi_0 的启示：10k 小时数据不是"堆得越多越好"，而是和模型规模 (3.3B) 匹配的结果。

**4.** GPT-3 是通用 base model, 通过不同数据 fine-tune 获得不同能力: Codex (代码) / WebGPT (工具使用) / InstructGPT (指令遵循)。领域 fine-tune 效果远大于 scale: GPT-3 175B 不会写代码, 但 Codex 12B (小 14x) 经过代码 fine-tune 就能写。对机器人的直接映射: PaliGemma 3B 不会控制机器人, 通过机器人操作数据 fine-tune → pi_0, 就能控制机器人。核心启示: 不需要从零训练, 复用已有的 VLM base model + 你的领域数据 fine-tune 就够了。

**5.** 逐个生成 token：预测第 1 个 -> 作为输入预测第 2 个 -> ... 。根本限制：(1) 每步只出一个 token，速度慢；(2) 连续值必须离散化为 token，丢失精度；(3) 不支持 action chunking（一次只能出一个动作 token）。RT-2 用 256 bins 离散化动作，50Hz 控制需要每秒预测 50x7=350 个 token，速度和精度都不够。

**6.** MoE: 模型有多组专家权重，每个 token 根据 routing 选择走哪组专家。好处是模型总参数量很大（容量大），但每个 token 只激活一小部分参数（计算量小）。pi_0 的 dual-expert 是最简单的 MoE：2 个 expert，routing 规则固定（图像/文本走 expert 0，状态/动作走 expert 1），不是学出来的。

**7.** GPT-4 增加了多模态（接受图像输入）和 RLHF alignment（对齐人类偏好）。RLHF 让模型不只是"预测最可能的下一个 token"，而是"生成人类觉得好的回答"。对机器人：pi_0 的 post-training 用高质量演示数据微调，本质上和 RLHF 的目标一样——让模型不只是模仿数据分布，而是模仿"好的"行为。

**8.** Token embedding 是查表操作，同一个词的 embedding 永远相同（context-independent）。"bank" 在任何句子中查到的 embedding 都一样。经过 Transformer 处理后，"river bank" 中的 "bank" 表征会融入 "river" 的信息（偏向河岸含义），"bank account" 中会融入 "account" 的信息（偏向银行含义）——这就是上下文化表征。

**9.** Chinchilla 的结论：等比缩放最优。给定计算预算 C，最优模型大小 N 和数据量 D 满足 N 正比于 D。实际含义：与其训一个 175B 模型跑 300B tokens（GPT-3 做法，模型大数据少），不如训一个 70B 模型跑 1.4T tokens（Chinchilla 做法，模型小数据多），后者效果更好。

**10.** GPT-4 不公开技术细节意味着闭源 LLM 不可复现、不可研究。开源 LLM 让研究者可以：阅读代码理解架构、在自己数据上微调、修改架构做实验。对机器人领域：pi_0 的开源 (openpi) 让研究者可以在自己的机器人数据上 fine-tune 预训练好的 VLA，而不需要从零训练——这直接降低了 VLA 的使用门槛。

---

## Level 2 参考答案

**1.** CLIP 用两个编码器分别编码图像和文本，用 contrastive loss 让匹配的图文对在表征空间中靠近、不匹配的远离。直觉："一张狗的照片"和"a photo of a dog"应该在同一个位置。Zero-shot 分类：把所有类别名变成文本（"a photo of a cat", "a photo of a dog", ...），编码后和图像表征比较距离，最近的就是分类结果——不需要任何分类标签训练。

**2.** 前向过程：逐步给数据加高斯噪声，T 步后变成纯噪声。反向过程：学一个网络预测每步加的噪声，然后减去。L_simple = MSE(预测的噪声, 实际加的噪声)，有效是因为它是变分下界的简化形式，且实践中效果更好（去掉了难以优化的权重系数）。

**3.** DDPM: 离散化的扩散过程，需要 1000 步迭代去噪，每步预测噪声。Flow matching: 连续的 ODE 路径，学习从噪声到数据的速度场，用 Euler 积分 10 步即可。pi_0 选 flow matching 因为推理步数少（10 vs 1000），且数学更简洁（直接回归速度场 vs 预测噪声）。

**4.** ResNet 表征是在 ImageNet 分类任务上学的，只保留了分类需要的信息（物体类别），丢弃了空间位置、大小等信息。CLIP 表征是在图文对比上学的，保留了语义对齐的所有信息（包括属性、关系、空间描述等）。对机器人：CLIP 更有用，因为机器人需要理解"红色的杯子在桌子左边"这种语言指令，需要空间和属性信息，不只是"这是一个杯子"。

**5.** 噪声图像 -> 噪声动作序列。去噪网络 -> policy network（条件是当前观测）。生成的图片 -> 生成的 action chunk（机器人接下来要执行的动作序列）。核心思想一样：从随机噪声出发，通过条件去噪生成目标输出。

**6.** 计算量：224x224x3 = 150,528 维像素直接作为 token 序列太长，attention 的 O(N^2) 计算量不可承受。表征质量：原始像素包含大量低级冗余信息（相邻像素高度相关），SigLIP 等视觉编码器将其压缩为 256 个语义 token，每个 token 编码一个 16x16 patch 的高级特征，对下游任务更有用。

**7.** ViT 把图像切成 16x16 的 patch，每个 patch 展平后通过线性投影变成一个 token，然后和 NLP 一样送进 Transformer。"An image is worth 16x16 words"：一个 224x224 的图像 = 196 个 16x16 patch = 196 个 "visual words"，可以用和处理文本完全相同的架构处理。

**8.** DiT 证明了 Transformer 可以替代 U-Net 作为 diffusion 的去噪骨干网络，且在大规模时效果更好。GR00T N1 的 System 1 (action head) 就是一个 DiT——用 Transformer 结构做 action diffusion，以 120Hz 输出动作。adaLN-Zero 条件注入也被 pi_0.5 继承为 adaRMSNorm。

**9.** 对比学习（CLIP）：保留跨模态共享的语义信息，丢弃模态特有的信息（如图像的精确像素、文本的精确措辞）。重建学习（MAE）：保留所有可重建的信息（包括纹理、光照等低级细节），但不保证语义对齐。前者对语言指令理解更好，后者对需要精确视觉信息的任务（如纹理识别）更好。

**10.** 灵巧手操控需要理解语言指令（"抓住红色的杯子"）+ 精确的空间感知（物体位置、朝向）。CLIP 预训练的 ViT 在语言指令理解上更强（因为预训练时对齐了语言），但可能丢失了精确的空间细节。如果任务以语言指令驱动为主 -> 选 CLIP；如果任务需要精确位姿估计 -> 可能需要额外的空间信息。实际 VLA (pi_0) 的做法是用 SigLIP（CLIP 变体）编码视觉，因为语言理解能力更重要。

---

## Level 3 参考答案

**1.** Decision Transformer 把 RL 定义为条件序列生成问题：给定期望回报 R、历史状态 s 和动作 a，预测下一个动作。靠 return conditioning 做决策——inference 时设定高回报值，模型生成对应的高质量动作序列。不需要 value function（不估计状态价值）和 policy gradient（不做策略梯度优化），纯监督学习。

**2.** 传统 BC 回归动作均值，面对同一观测下的多种合理动作时会取平均（"桌子左右两侧都能绕过去" -> 回归结果是"直接撞上去"）。Diffusion Policy 学习完整的动作分布，能生成多种合理动作中的一种。Multimodal action distribution 的例子：灵巧手抓杯子，可以从上方抓也可以从侧面抓，两种都对——BC 取平均会导致手移到两种抓取位置的中间（空气中），Diffusion Policy 会清晰地选择其中一种。

**3.** Action chunking 一次预测多步动作，解决：(1) 时间一致性——逐步预测每步独立，可能来回抖动；一次出 50 步自然平滑。(2) 高频控制——50Hz 控制如果每步都跑一次推理太慢。代价：对环境变化的反应延迟（执行完一个 chunk 才能看新的观测），以及如果预测的远期动作不准确会累积误差。

**4.** RT-1: robot-specific Transformer，专门为机器人设计的架构（EfficientNet + TokenLearner），只能输出机器人动作。RT-2: 直接复用 VLM（PaLM-E / PaLI-X），把动作 token 化为文本 token，继承了 VLM 的互联网知识。跃迁 = 从专用架构到通用 VLM。Action tokenization 优点：复用 LLM 架构，继承语言理解能力。缺点：256 bins 离散化丢失精度，不支持 action chunking（每步一个 token），高频控制困难。

**5.** World model (DreamerV3): 假设"可以从数据中学习一个环境的动力学模型，然后在模型中做 planning"。优势：数据效率高（在想象中训练，不需要真实交互），能做长期规划。Imitation learning (pi_0): 假设"可以直接从演示数据中学习状态到动作的映射"。优势：不需要建模物理（避免 sim-to-real gap），可以处理非常复杂的操作行为。

**6.** Autoregressive (RT-2): 复用 LLM，继承语言能力，但离散化丢失精度、速度慢。DDPM (Diffusion Policy): 连续动作、多模态分布，但推理需要多步去噪，速度中等。Flow matching (pi_0): 连续动作、多模态分布，且推理步数少（10 步 vs DDPM 的 100+ 步），速度最快。

**7.** 因为环境会变化。如果预测了 50 步 (1 秒) 的动作然后全部执行完才看新观测，1 秒内如果物体移动了或受到扰动，后续动作就是错的。Receding horizon: 预测 50 步但只执行前 16-25 步，然后重新观测、重新预测——兼顾了 chunk 的平滑性和对环境变化的响应。

**8.** 130k episodes x ~数分钟 ≈ 几千小时机器人数据。GPT-3 用 300B tokens ≈ 几十万小时文本。机器人数据比文本稀缺 100x+，因为每个 episode 需要真实机器人物理执行。挑战：(1) 数据收集成本极高（需要遥操作/自主探索），(2) 数据多样性受限（同一个实验室、同一组物体），(3) 不像文本可以从互联网免费爬取。

**9.** 说明 DreamerV3 的世界模型有很强的泛化能力——它学到了通用的动力学建模方式，而不是对每个任务过拟合。和物理引擎的区别：物理引擎基于精确方程（需要精确参数），世界模型从数据中学习（不需要精确参数，但可能不精确）。物理引擎有 sim-to-real gap，世界模型如果在真实数据上训练则没有。

**10.** Tradeoff: Diffusion Policy——成熟、有大量验证、代码开源，但不支持语言指令、不继承互联网知识。ACT——更简单（CVAE + Transformer），训练快，但表达能力有限（高斯假设）。VLA (pi_0)——功能最全（语言+视觉+动作），但需要大量数据微调、推理成本高、部署复杂。对灵巧手：如果只做单任务（如方块重定向），Diffusion Policy 或 ACT 够用；如果要做多任务 + 语言指令（如"把红色杯子放到左边"），需要 VLA。

---

## Level 4 参考答案

**1.** 通过 Transformer 的 self-attention 交互。图像/文本 token 走 VLM backbone 的权重，状态/动作 token 走 action expert 的权重，但在 attention 计算时 Q/K/V 被 concatenate 在一起做联合 attention——动作 token 可以直接 attend to 图像 token，获取视觉信息。VLM backbone 负责视觉-语言理解（复用预训练知识），action expert 负责连续动作生成（从头学习）。

**2.** Next-token prediction 适合离散输出（文本 token，有限词表）。Flow matching 适合连续输出（动作向量，连续值）。pi_0 不用 autoregressive 因为：(1) 动作是连续值，离散化丢失精度；(2) 50Hz 控制需要一次出 50 步动作（action chunk），autoregressive 逐 token 太慢；(3) flow matching 天然支持多模态分布。

**3.** 直觉来自 Kahneman 的 "Thinking, Fast and Slow"。System 2 (VLM, 10Hz): 慢思考，理解语言指令、推理任务步骤。System 1 (DiT, 120Hz): 快执行，根据 System 2 的指导生成高频动作。不合成一个因为：语义理解不需要 120Hz（1 秒想一次就够），但动作必须 120Hz（否则不够平滑）。分开后各自按需要的频率运行。

**4.** 所有机器人的动作/状态向量统一到最大维度（pi_0 中是 32 维），低维机器人零填充 (zero-pad)。图像不足 3 个的 mask 掉缺失槽位。数据集按 n^0.43 权重平衡，防止大任务压制小任务。

**5.** 预训练的大规模低质量数据提供：广泛的能力覆盖、各种错误和恢复行为、对多种场景的鲁棒性。Post-training 的高质量数据提供：流畅的执行策略、任务特化的效率和精度。只有前者：什么都会一点但不精通。只有后者：对预期情况表现好，但遇到意外（物体滑落、碰撞）就崩溃。

**6.** KV cache 缓存了图像/文本/状态 token 的 Key 和 Value（这些在 10 步 flow matching 中不变）。每步只重新计算动作 token（50 个 token）的前向传播。省了约 90% 的计算——355 个 token 中只需重算 50 个。

**7.** 不是开倒车。pi_0-FAST 用 FAST tokenizer 把连续动作编码为离散 token 序列，然后用 autoregressive 生成。优势：(1) 推理只需一次前向传播（不需要 10 步积分），延迟更低；(2) 不需要 action expert 的额外参数（更轻量）；(3) 在数据量少的场景下可能更稳定。代价是离散化的精度损失。适合对延迟敏感或数据量有限的场景。

**8.** Octo: 27M/93M 小模型, diffusion action head (DDPM, 20步去噪), 从头训 Transformer, 用 readout tokens 解耦观测和动作。优点: 轻量、开源、flexible I/O。缺点: 没有预训练 VLM 的语言理解能力, 模型太小无法吸收大规模数据。OpenVLA: 7B VLM (Prismatic) + 256-bin discrete action tokens, 继承 RT-2 的离散化方案。优点: 语言理解强, LoRA 微调高效。缺点: 离散化精度丢失, 不支持 action chunking, 6Hz 太慢。pi_0: 3B VLM (PaliGemma) + 300M action expert + flow matching。Octo 团队放弃原方案的原因: (1) 从头训的小 Transformer 无法继承互联网知识, VLM backbone 是质的飞跃; (2) DDPM 20 步太慢, flow matching 10 步更快且路径更直; (3) readout token -> action expert, 概念类似但参数独立, 避免了动作训练破坏 VLM 表征。

**9.** 表征学习 (2012): "好表示=好AI" -> Transformer (2017): 通用序列建模 -> GPT (2018-2020): autoregressive pre-training + scaling -> ViT (2020): patch embedding 将 Transformer 迁移到视觉 -> CLIP (2021): contrastive learning 对齐视觉-语言 -> DDPM (2020) + Flow Matching (2022): 连续数据生成 -> ACT (2023): action chunking 减少 compounding error -> Open X-Embodiment (2023): 跨机器人数据标准化 -> PaliGemma (2024): SigLIP+Gemma = 小而强的 VLM backbone -> Octo/OpenVLA (2024): 开源 generalist robot policy 探索 -> pi_0 (2024): VLM + flow matching action expert + cross-embodiment pre-train, 融合以上所有思想。每一步继承了前一步的表征能力, 创新在如何把表征扩展到新的模态或任务。

**10.** 适配：(1) 定义动作空间（关节角 20 DoF -> action_dim）；(2) 配置相机输入（手腕/外部）；(3) 收集演示数据（遥操作）；(4) 从 pi_0 base model fine-tune（LoRA 或全参数）。数据量：简单任务 5-10 小时，复杂灵巧任务（如方块重定向）可能需要 50-100 小时。挑战：(1) 灵巧手的动作空间比双臂操作更高维（20 vs 14 DoF），需要更多数据；(2) 接触丰富的任务需要更高的动作精度；(3) 50Hz 高频控制对推理延迟要求严格。

**11.** 相同：都存在 power law 关系（更多数据/更大模型 -> 更好性能），且都存在涌现现象。不同：(1) 机器人的 scaling exponent 比语言更大（scaling 更高效，同等数据增量带来更大性能提升）；(2) 机器人数据收集成本比文本高 100x+，数据是核心瓶颈而非计算；(3) 机器人性能用 success rate 衡量（有上限=100%），不像语言的 loss 可以无限下降；(4) 机器人数据的多样性受限于硬件和场景，不像文本可以从全互联网爬取。
