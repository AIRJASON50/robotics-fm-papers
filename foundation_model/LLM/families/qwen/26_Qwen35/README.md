# Qwen3.5 (2026.02)

**状态**: 已发布模型和权重, 尚无 arxiv 技术报告

**发布信息**:
- GitHub: https://github.com/QwenLM/Qwen3.5
- HuggingFace Blog: https://huggingface.co/blog/mlabonne/qwen35
- 首发模型: Qwen3.5-397B-A17B (MoE)

**核心创新**: 混合注意力架构
- 75% 层使用 Gated DeltaNet (线性注意力)
- 25% 层使用标准 softmax attention (GQA + RoPE)
- 每 4 层一个 full attention 层
- 推理加速 3.5x ~ 7.2x

**其他特性**:
- 稀疏 MoE: 397B 总参 / 17B 激活
- 201 种语言支持
- 原生多模态 (早期融合)

**待补充**: arxiv 技术报告发布后拉取
