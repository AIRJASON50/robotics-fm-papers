# Qwen Technical Report -- 分析笔记

Qwen Team, Alibaba Group, 2023 (arXiv:2309.16609)

## 1. Core Problem

构建一个全面的开源 LLM 系列，覆盖 base model、chat model、代码专用、数学专用、多模态等变体，与 GPT-4、LLaMA 等竞争。

## 2. 模型家族

```
Qwen (base) ──> Qwen-Chat (SFT) ──> Qwen-Chat-RLHF
    ├──> Code-Qwen ──> Code-Qwen-Chat
    ├──> Math-Qwen-Chat
    └──> Qwen-VL ──> Qwen-VL-Chat
```

参数规模: 1.8B, 7B, 14B, 72B

## 3. 关键技术

| 组件 | 选择 | 说明 |
|------|------|------|
| Tokenizer | BPE (tiktoken), 151,643 tokens | 包含中英文、代码、多语言 |
| Architecture | Transformer decoder, pre-RMSNorm, SwiGLU, RoPE | 标准现代 LLM 架构 |
| Context | 2048 -> 8192 (NTK-aware RoPE 扩展) | 动态 NTK interpolation |
| 训练数据 | 3T tokens (多语言, 含代码) | -- |
| Alignment | SFT + RLHF (PPO) | 与 GPT 路线一致 |
| Tool Use | ReAct 格式, Code Interpreter | Agent 能力 |

## 4. 学习价值

对机器人研究者，Qwen 的价值在于:

1. **完整的 LLM 工程实践**: 代码包含预训练、SFT、推理、量化的完整流程
2. **Tokenizer 设计**: BPE 的实际实现，理解 token 化如何影响下游任务 (RT-2 把动作也 token 化)
3. **RLHF pipeline**: Reward Model + PPO 的完整实现，与机器人 RL 有直接对应
4. **Context 扩展**: NTK-aware RoPE 是处理长序列的关键技术，机器人 action chunk 也面临序列长度问题
5. **多模态扩展**: Qwen-VL 展示了如何从 LLM 扩展到 VLM，与 VLA 的 VLM backbone 直接相关

## 5. 代码仓库

`qwen_repo/` -- 官方 Qwen 代码 (已归档，推荐使用 QwenLM/Qwen2)

关键文件:
- `finetune.py` -- 微调入口
- `cli_demo.py` -- 交互式推理
- `eval/` -- benchmark 评估
- `recipes/` -- 推理/微调配方 (vLLM, TensorRT, DeepSpeed, Swift)

注: 该仓库已不再活跃维护，Qwen2 系列已迁移到 QwenLM/Qwen2。
