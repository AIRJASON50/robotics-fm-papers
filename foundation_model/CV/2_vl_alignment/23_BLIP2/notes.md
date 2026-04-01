# BLIP-2: Bootstrapping Language-Image Pre-training -- Takeaway Notes

> 一句话: 提出 Q-Former (Querying Transformer) 作为 frozen image encoder 和 frozen LLM 之间的轻量级桥梁, 用两阶段 bootstrapping 策略以 54x 更少可训参数超越 Flamingo80B。

## 核心贡献

1. **Q-Former 架构**: 一个 188M 参数的 lightweight transformer, 内含 32 个 learnable query embeddings (32x768), 通过 cross-attention 从 frozen ViT 提取最相关的 visual features, 再投射给 frozen LLM
   - Image transformer: queries 与 frozen image features 交互 (cross-attention)
   - Text transformer: 可作 encoder 或 decoder, 与 queries 共享 self-attention layers
   - 关键: **information bottleneck** -- 32x768 远小于原始 257x1024 的 ViT features
2. **两阶段 bootstrapping**:
   - Stage 1 (Representation Learning): 冻结 image encoder, 训练 Q-Former 对齐 vision-language (ITC + ITM + ITG 三个 loss)
   - Stage 2 (Generative Learning): 冻结 LLM, 将 Q-Former 输出投射为 LLM 的 soft visual prompts
3. **模块化 VLP**: 首次证明可以完全冻结 vision + language 两个大模型, 只训中间的 bridge module

## 为什么重要

- **compute efficiency 标杆**: Flamingo 需要训 10.2B 参数, BLIP-2 只训 188M (Q-Former), 结果更好 -- 这是 "参数效率" 思想的极致体现
- **定义了 "adapter" 范式**: Q-Former 是 vision-language adapter 的代表性设计, 后续 InstructBLIP / MiniGPT-4 / 许多 VLA 的 visual projector 都受其影响
- **模块可替换性**: 因为 vision encoder 和 LLM 都是 frozen plug-in, 可以随时升级 (ViT-G → ViT-22B, OPT → FlanT5), 这就是 "modular FM" 的思想

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动项 |
|---|----------|--------|
| 1 | **Information bottleneck 是跨模态对齐的核心设计**: 32 个 query 把 257 个 visual tokens 压缩为最与 text 相关的信息 | Robot policy 中 visual encoder 输出也需要 bottleneck -- 不是所有 patch 都与 action 相关 |
| 2 | **冻结大模型 + 训小 adapter 是资源友好的范式**: 188M 可训参数 vs 数十B 冻结参数 | VLA fine-tuning 时优先尝试 LoRA / adapter, 不要 full fine-tune |
| 3 | **三个对齐 loss 的分工**: ITC (全局对齐) + ITM (细粒度匹配) + ITG (生成能力) 三管齐下 | Robotics representation learning 也可以多目标联合训练 (contrastive + reconstruction + prediction) |
| 4 | **LLaVA 简单线性层 vs Q-Former 复杂 adapter**: LLaVA 证明如果允许 end-to-end tune LLM, 简单 projector 就够; BLIP-2 证明如果冻结 LLM, 需要更强的 adapter | 根据 compute budget 选择策略 |

## 与知识库其他内容的关联

- **CLIP** (`CV/2_vl_alignment/21_CLIP`): Q-Former Stage 1 的 ITC loss 直接继承 CLIP 的 contrastive learning 思想
- **LLaVA** (`CV/2_vl_alignment/23_LLaVA`): 对比路线 -- complex adapter + frozen LLM vs simple projection + tuned LLM
- **OpenVLA** (`robotics/vla/24_OpenVLA`): 选择了 LLaVA 路线 (MLP projector + tune LLM), 没选 Q-Former, 说明在 action generation 场景 end-to-end tuning 更重要
- **R3M / VIP** (`robotics/visual_repr/`): 同样关注 visual representation 质量, 但用 video contrastive learning 而非 Q-Former
- **Transformer** (`foundations/17_Transformer`): Q-Former 本质是 cross-attention 的巧妙应用 -- learnable queries 做 key/value 的信息提取
