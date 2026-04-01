# Visual Instruction Tuning (LLaVA) -- Takeaway Notes

> 一句话: 用 GPT-4 生成 158K 条 vision-language instruction-following 数据, 把 CLIP ViT + Vicuna 通过一个线性投影层端到端微调, 证明了 visual instruction tuning 是通往通用 VLM 的可行路径。

## 核心贡献

1. **Visual Instruction Tuning 范式**: 首次将 NLP 的 instruction tuning 思想迁移到多模态领域, 定义了 "用 instruction-following 数据训练 VLM" 的标准流程
2. **GPT-4 生成训练数据**: 用 caption + bounding box 作为 symbolic representation 送入 text-only GPT-4, 生成三类数据 (conversation / detailed description / complex reasoning), 共 158K 条
3. **极简架构 + 两阶段训练**:
   - Stage 1: 冻结 CLIP ViT + LLM, 只训练线性投影层 W (595K image-text pairs, feature alignment)
   - Stage 2: 冻结 ViT, 微调投影层 + LLM (158K instruction data, end-to-end)

## 为什么重要

- **定义了 VLM 的 "GPT-3.5 moment"**: LLaVA 证明 frozen vision encoder + lightweight connector + LLM + instruction data 就够了, 不需要 Flamingo 那样的 billions of pairs
- **直接启发 OpenVLA**: OpenVLA 的 Prismatic VLM backbone 就是 LLaVA 架构的延伸 (DINOv2+SigLIP → MLP projector → Llama 2), 把 text output 换成 action tokens
- **数据 > 架构的佐证**: 简单的线性投影层就能对齐 vision-language, 关键在于 instruction-following 数据的质量和多样性

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动项 |
|---|----------|--------|
| 1 | **两阶段训练是 VLA 的模板**: Stage 1 alignment + Stage 2 task tuning, OpenVLA/RT-2 都沿用此范式 | 理解 VLA fine-tuning 时哪些参数冻结、哪些可训 |
| 2 | **Instruction data 是 zero-shot 泛化的关键**: 158K 条 GPT-4 生成数据就让模型获得 multimodal chat 能力 | 在 robot 场景, language instruction 的多样性决定了 policy 的泛化边界 |
| 3 | **线性投影 vs Q-Former**: LLaVA 用最简单的 linear projection 就超过 BLIP-2 的复杂 Q-Former 在 instruction following 上的表现 | 简单 connector 足够, 关键是 end-to-end tuning |
| 4 | **Vision encoder 可以冻结**: CLIP ViT 的表征已经足够好, 不需要在 VLM 训练中更新 | Sim2real 的 visual backbone 也可以 frozen pre-trained, 只训 adapter |

## 与知识库其他内容的关联

- **CLIP** (`CV/2_vl_alignment/21_CLIP`): LLaVA 的 vision encoder, 提供 visual feature Z_v
- **BLIP-2** (`CV/2_vl_alignment/23_BLIP2`): 对比架构 -- Q-Former vs linear projection, 两种 bridge modality gap 的策略
- **PaliGemma** (`CV/2_vl_alignment/24_PaliGemma`): 后续工作, 更强的 VLM baseline
- **OpenVLA** (`robotics/vla/24_OpenVLA`): 直接继承 LLaVA 的两阶段范式, 把 output 从 text 换成 action tokens
- **RT-2** (`robotics/families/Google_RT_Series/23_RT2`): 同期工作, 同样证明 VLM + action tokenization 可以做 robot policy
