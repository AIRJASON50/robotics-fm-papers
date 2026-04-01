# Visual Instruction Tuning (LLaVA) -- 学习笔记
> 一句话: 用 GPT-4 生成 158K vision-language instruction-following 数据, 通过线性投影层连接 CLIP ViT 和 Vicuna LLM, 两阶段训练出第一个通用 visual instruction-following 模型。
> 论文: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (UW-Madison, Microsoft Research), NeurIPS 2023
> 引用量级: ~8000+

## 这篇论文解决了什么问题
NLP 领域, instruction tuning (用 instruction-following 数据微调 LLM) 已经产生了 ChatGPT/Vicuna 等强大的 assistant。但多模态领域缺少这样的尝试 -- 原因是 (1) 没有大规模 vision-language instruction-following 数据; (2) 已有的多模态模型 (Flamingo, BLIP-2) 没有经过 instruction tuning, 在多模态交互任务上表现不佳。LLaVA 要回答: 能否用最简单的架构 + 高质量的 instruction 数据, 让 VLM 学会 follow instructions?

## 核心想法 (用直觉解释)
人类用眼睛看到图像, 用语言描述和推理。LLaVA 模拟这个过程: CLIP ViT 是 "眼睛", 把图像变成一组 visual tokens; 一个线性投影层是 "翻译器", 把 visual tokens 映射到 LLM 的词嵌入空间; Vicuna LLM 是 "大脑", 接收 visual tokens + text tokens 后生成回答。关键不在架构复杂度, 而在训练数据: 作者用 GPT-4 从 COCO 图像的 caption+bbox 出发, 自动生成三类数据 -- 对话 (58K)、详细描述 (23K)、复杂推理 (77K), 共 158K 条。训练分两步: 先对齐 vision-language (只训投影层), 再 instruction tuning (训投影层+LLM)。

## 关键设计决策
- **极简架构**: CLIP ViT-L/14 (frozen) -> 线性投影矩阵 W -> Vicuna LLM。不用 Flamingo 的 gated cross-attention, 不用 BLIP-2 的 Q-Former, 一个 linear layer 就够。这使得迭代速度极快, 可以专注于数据实验
- **两阶段训练**: Stage 1 (feature alignment) -- 冻结 ViT + LLM, 只训 W, 用 595K filtered CC3M image-text pairs, 让 visual tokens 对齐到 LLM word embedding 空间。Stage 2 (instruction tuning) -- 冻结 ViT, 训 W + LLM, 用 158K GPT-4 生成的 instruction data
- **GPT-4 作为数据工厂**: 无法直接给 GPT-4 送图像 (当时 GPT-4 只接受 text), 所以用 caption + bounding box 作为图像的 symbolic representation 输入 GPT-4, 让它生成多样化的 QA 对。手工设计少量 seed examples 作为 in-context learning 的示范
- **Conversation format**: 多轮对话格式, image token 随机放在第一个 question 的前面或后面。只对 assistant 回答部分计算 loss (和 Vicuna 训练一致)

## 这篇论文之后发生了什么
- **LLaVA-1.5/NeXT/OneVision**: 后续版本把 linear projection 升级为 MLP, 支持更高分辨率, 效果持续提升
- **OpenVLA**: 直接继承 LLaVA 架构 (DINOv2+SigLIP -> MLP projector -> Llama 2), 把 text output 换成 discretized action tokens, 实现 VLA
- **VLM 成为 VLA 的 backbone**: LLaVA 证明了 "vision encoder + projector + LLM" 这个最小化 VLM 架构的有效性, 成为 RT-2, OpenVLA, pi0 等 VLA 的起点

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 两阶段训练是 VLA 的模板: Stage 1 alignment + Stage 2 task tuning | OpenVLA/RT-2 都沿用此范式; 理解 fine-tuning 时哪些参数冻结、哪些可训 |
| 2 | 线性投影就能对齐 vision-language, 架构不是瓶颈, 数据质量才是 | Robot instruction data 的多样性决定 policy 泛化边界; GPT-4 生成 robot instruction 也是可行的 |
| 3 | Vision encoder 可以完全冻结, pre-trained features 已经足够好 | Robot visual backbone 不需要 end-to-end 训练; frozen DINOv2/SigLIP + trainable adapter 是高效选择 |
| 4 | Instruction tuning 是 "task-specific -> general-purpose" 的关键一步 | 从 per-task policy 到 general-purpose robot policy, 需要的不是更复杂的架构, 而是多样化的 instruction 数据 |
