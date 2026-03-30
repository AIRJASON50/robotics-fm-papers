# RT-1: Robotics Transformer for Real-World Control at Scale

**Paper**: Brohan et al., Google / Everyday Robots, 2022 (RSS 2023), arXiv:2212.06817

---

## 1. Core Problem

将 robot learning 从小规模实验室 demo 推向真实世界的大规模部署, 面临两大挑战:

- **数据瓶颈**: 现有方法通常只用几百到几千条 demonstration, 难以覆盖真实场景的多样性
- **泛化瓶颈**: 在少量任务上训练的 policy 无法泛化到新物体、新指令、新环境

RT-1 的核心问题: **能否通过大规模真实机器人数据 (130k episodes, 700+ tasks) 训练一个
高容量的 Transformer 模型, 使其在保持高性能的同时实现广泛泛化?**

## 2. Method Overview

RT-1 是一个将自然语言指令和图像观测映射到离散化机器人动作的 end-to-end 模型:

```
Input: text instruction + image history (6 frames)
  -> FiLM-conditioned EfficientNet (image encoder)
  -> TokenLearner (token compression)
  -> Transformer (sequence modeling)
  -> Discrete action tokens (autoregressive decoding)
Output: 7-DoF arm action + base movement + gripper + termination
```

整个 pipeline 以 3Hz 频率运行, 满足实时控制需求.

## 3. Key Designs

### 3.1 TokenLearner -- 高效的 visual token 压缩

直接将 image patch tokens 送入 Transformer 计算量过大. RT-1 引入 TokenLearner:
- 输入: EfficientNet 输出的空间 feature map (例如 9x9x512)
- 通过学习的 spatial attention maps 将其压缩为 **8 个 tokens**
- 大幅减少 Transformer 的 sequence length, 使实时推理成为可能
- 对比实验显示: 没有 TokenLearner, 模型速度下降且性能也降低

### 3.2 Action Tokenization -- 连续动作的离散化

RT-1 将连续动作空间离散化为 token:
- 每个动作维度 (共 11 维: x/y/z/rx/ry/rz/gripper + base x/y/yaw + terminate)
  被均匀分为 **256 个 bin**
- 每个 bin 对应一个 discrete token
- 动作通过 autoregressive decoding 逐维度生成
- 这种设计使得可以用标准的 categorical cross-entropy loss 训练, 避免了回归 loss 的调参困难

### 3.3 大规模真实数据 + 多任务学习

数据规模:
- **130,000 episodes** from 13 robots over 17 months
- **700+ tasks** (pick, place, open drawer, move near, etc.)
- 自然语言指令描述每个 task

这是当时 robot learning 领域最大规模的真实机器人数据集之一.
多任务训练使模型能在共享表征中学到可迁移的 manipulation primitives.

## 4. Experiments

### 4.1 Seen Tasks Performance

在 200+ seen tasks 上测试:
- RT-1 成功率 **97%** (SayCan baseline: 53%)
- 显著优于 Gato (55%) 和 BC-Z (72%) 等 baseline

### 4.2 Generalization

RT-1 展示了三个层面的泛化:

| 泛化类型 | 测试条件 | 成功率 |
|---------|---------|--------|
| Unseen objects | 训练中未出现的物体 | **76%** |
| Unseen environments | 新的厨房/场景布局 | **significantly > baselines** |
| Long-horizon (SayCan) | 与 LLM planner 组合执行长序列任务 | **67%** (vs SayCan 原版 47%) |

### 4.3 Data Scaling

实验表明 performance 随数据量 log-linear 增长:
- 从 100 tasks 增加到 700+ tasks, 性能持续提升
- 关键发现: 即使新增的 task 与测试 task 不直接相关, 也能带来泛化性能提升
  (positive transfer across tasks)

### 4.4 Ablation Studies

- TokenLearner: 去除后推理速度和性能均下降
- Action space: 离散化 (256 bins) 优于连续回归
- History length: 6 frames 最优, 过多过少都不好
- Model size: 更大的 EfficientNet backbone 带来提升

## 5. Impact

### 5.1 开启 VLA (Vision-Language-Action) 范式

RT-1 证明了: 大规模数据 + 高容量模型 = 可泛化的 robot policy.
这一 insight 直接催生了后续工作:

- **RT-2 (2023)**: 将 vision-language model (PaLM-E / PaLI) 直接 fine-tune 为 robot policy,
  实现了 "VLA" -- 用 web-scale 预训练知识增强机器人推理能力
- **RT-X (2023)**: 跨机构、跨 embodiment 的大规模 robot learning
- **Octo, OpenVLA**: 开源 VLA 模型, 延续 RT-1/RT-2 的路线

### 5.2 Robot Foundation Model 的可行性验证

RT-1 是第一个在工业级规模上验证 "robot foundation model" 可行性的工作:
- 证明了 data scaling 在 robotics 中同样有效
- 证明了 Transformer 架构可以统一处理多任务 manipulation
- 为后续 robot learning 社区大规模数据收集 (Open X-Embodiment) 提供了动力

### 5.3 从 task-specific 到 general-purpose

RT-1 推动了机器人控制从 "per-task policy" 向 "one model, many tasks" 的范式转变,
类似于 NLP 从 task-specific models 向 GPT/BERT 等 foundation model 的演进.
