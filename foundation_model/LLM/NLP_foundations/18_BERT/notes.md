# BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018) -- Takeaway Notes

> 一句话: 用 Masked Language Model (随机遮盖 15% token 让模型预测) 实现真正的双向 Transformer pre-training, 然后只加一层输出头就能 fine-tune 到 11 个 NLP 任务的 SOTA.

## 核心贡献
- 提出 MLM (Masked Language Model, 掩码语言模型): 随机 mask 15% input token (80% 替换为 [MASK], 10% 随机词, 10% 保持不变), 预测被 mask 的原始 token. 这绕过了 "双向模型会看到自己" 的问题, 实现了深度双向 pre-training
- 提出 NSP (Next Sentence Prediction, 下一句预测): 训练模型判断两个句子是否相邻, 学习句间关系
- 确立 "pre-train + fine-tune" 的标准范式: 同一个预训练模型, 仅换输出层即可适配 QA / NLI / 分类等不同任务. BERT_BASE (110M) 和 BERT_LARGE (340M) 两个尺度
- 在 GLUE 上达到 80.5% (比 GPT 高 7.7 点), SQuAD v1.1 F1 达到 93.2, 全面刷新 11 项 NLP benchmark

## 为什么重要
1. **Masked prediction 作为 self-supervised objective**: MLM 证明 "破坏-重建" 是训练双向表示的有效方式, 直接启发了 CV 中的 BEiT/MAE 以及 robotics 中的 masked trajectory prediction.
2. **Pre-train + fine-tune 成为工业标准**: 一个通用模型 + 轻量 fine-tune 取代了为每个任务定制架构, 这条路线通过 GPT-3 演化为 prompt/in-context learning, 通过 RT-2/Octo 延伸到 robotics.

BERT (encoder-only, 双向, 理解) vs GPT (decoder-only, 单向, 生成) 定义了大模型两大流派.
Robotics FM 主要在做"生成动作", 所以多数 VLA 选择 decoder 架构.

## 对你 (RL->FM) 的 Takeaway
- **MLM = MAE = 机器人的 masked prediction**: mask 掉部分输入让模型补全, 是最通用的
  self-supervised pretext task. 在 robotics 中, VideoMAE 预训练 visual encoder,
  Octo 用 masked trajectory 预测来学 multi-task policy -- 都是 BERT MLM 思想的直系后裔
- **双向 vs 自回归的选择**: BERT 式双向编码更擅长"理解" (场景理解, 物体检测, 状态估计);
  GPT 式自回归更擅长"生成" (action sequence, trajectory). 你的 VLA pipeline 中,
  vision encoder 适合 BERT 式 (如 ViT+MAE), action decoder 适合 GPT 式 (如 autoregressive chunk)
- **Fine-tune 的极致简洁**: BERT 只加一个线性层就能适配新任务. 这预示了 robotics FM
  的目标 -- 一个预训练模型, 通过极少量 task-specific adaptation 部署到不同机器人/任务
- **[CLS] token 的设计**: BERT 在序列开头加 [CLS] token, 其最终表示用于句级分类.
  RT-1/RT-2 中 "action token" 附在 visual/language token 后面, 起同样的"汇聚全局信息"作用

## 与知识库其他内容的关联
- 13_Word2Vec: static embedding -> BERT contextual embedding 的演进. 同一个词 "bank" 在 Word2Vec 中只有一个向量, 在 BERT 中随上下文变化
- 15_BahdanauAttention -> foundations/17_Transformer -> BERT: attention -> self-attention -> masked self-attention pre-training 的技术链
- LLM/families/GPT_Series: BERT (encoder, 双向, 理解) vs GPT (decoder, 单向, 生成) 是大模型两大路线的分野
- CV/4_self_supervised (MAE, BEiT): 将 BERT MLM 迁移到 vision -- mask image patches 并重建, BEiT 连名字都致敬了 BERT
- robotics/vla/Octo: 使用 masked trajectory prediction 做 multi-task pre-training, 是 BERT MLM 在 robotics 的直接应用
