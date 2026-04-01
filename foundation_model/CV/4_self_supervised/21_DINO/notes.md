# Emerging Properties in Self-Supervised Vision Transformers (DINO) -- 学习笔记

> 一句话: 用 self-distillation (无标签 teacher-student 框架) 训练 ViT, 发现 self-attention 自动学会了语义分割, k-NN 性能接近 linear probe。
> 论文: Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin (Facebook AI / Inria), 2021, ICCV 2021

## 这篇论文解决了什么问题

ViT (2020) 问世后, 和 CNN 性能差不多, 还需要更多数据和计算, 没有展现独特优势。核心问题: **ViT 表现平庸是因为架构本身, 还是因为监督学习限制了它?**

出发点: NLP 中 Transformer 靠 self-supervised pre-training (BERT, GPT) 成功, 而非监督学习。CV 的 SSL 方法 (MoCo, SimCLR, BYOL) 主要在 CNN 上验证, 用在 ViT 上没展现特别性质。

核心发现: **Self-supervised ViT 具备监督 ViT 和所有 CNN 都没有的 emergent properties -- self-attention map 直接包含语义分割信息, k-NN 分类几乎不需要 linear classifier。**

## 核心想法 (用直觉解释)

**DINO = self-DIstillation with NO labels。本质是知识蒸馏, 但 teacher 不是固定模型, 而是 student 的 EMA (Exponential Moving Average)。**

流程:
1. 同一张图做两种 augmentation: global view (224x224, >50% 面积) 和 local view (96x96, <50%)
2. **Student** 看所有 view, **Teacher** 只看 global view
3. Student 输出 softmax 和 Teacher 输出 softmax 做 cross-entropy loss
4. Teacher = student 参数的 EMA (lambda 从 0.996 cosine schedule 到 1)

关键: Teacher 只看 global view 但 Student 也看 local view -- 迫使 Student 从局部 patch 预测全局语义 ("local-to-global correspondence")。

## 关键设计决策

**1. Centering + Sharpening -- 防 collapse 的极简方案**

SSL 最怕 mode collapse (所有输入产生相同 embedding)。之前的方法用负样本、predictor、BN 防止 collapse。DINO 只用两个操作:
- **Centering**: teacher 输出减去 batch mean 的 EMA -- 防一个维度 dominate (鼓励均匀分布)
- **Sharpening**: teacher 用低温度 tau_t softmax -- 防输出太平 (鼓励尖锐分布)

两者效果相反, 合在一起恰好平衡。不需要负样本、memory bank 或 BN。

**2. Momentum teacher (EMA) 是必需的**

没有 momentum, 框架直接 collapse (Table 7, row 2: k-NN 0.1%)。EMA teacher 因 ensemble 效应始终比 student 好, 提供高质量 target。这和 DQN 的 target network 直觉一致: 缓慢变化的 target 稳定训练。

**3. Multi-crop + cross-entropy loss**

2 global crop + 若干 local crop, 只把 global view 送 teacher。Multi-crop 和 CE loss 都是关键组件 (Table 7 ablation)。去掉 multi-crop 性能显著下降。

**4. 小 patch 尺寸对 ViT 特别重要**

ViT-B/8 (8x8 patch) 比 ViT-B/16 (16x16) 性能好得多: k-NN 从 76.1->77.4, 检索和分割大幅改善。小 patch = 更多 token = 更精细空间信息。但 throughput 从 312 降到 63 im/s。

**5. Self-attention 的 emergent segmentation**

DINO + ViT 的 [CLS] token self-attention 在最后一层自动呈现语义分割 pattern -- 不同 head 关注不同物体/部件 (Fig. 3)。Jaccard similarity: DINO 45.9 vs supervised 27.3 (Fig. 4)。这在监督 ViT 和所有 CNN 中都不存在。

**6. 主要实验结果**

ImageNet linear eval: DINO + ViT-B/8 达 80.1% top-1 (超过所有同期 SSL); k-NN eval: 77.4% (几乎等于 linear 的 80.1%), 这个 k-NN 接近 linear 的性质只在 DINO + ViT 组合中出现。

## 这篇论文之后发生了什么

- **DINOv2 (2023)**: 扩大规模 (ViT-g), 加 iBOT (masked image modeling) + curated data, 成为当前最强通用视觉 backbone
- **DINO 特征成为 robotics 标准视觉表征**: robot learning 方法直接用冻结 DINO/DINOv2 作 observation encoder
- **Self-distillation 成为 SSL 标配**: EMA teacher 在 BYOL, DINO, iBOT, DINOv2 中都是核心组件
- **启发 "emergent properties" 讨论**: 规模 + 自监督 -> 不可预测的涌现能力

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|----------|
| 1 | **Self-supervised ViT 特征包含空间语义, 正是 robot 需要的** -- 不需标注就能定位物体 | R3M, VIP 等在 DINO/DINOv2 基础上构建 robot visual encoder, 提供开箱即用的 spatial+semantic 特征 |
| 2 | **EMA teacher = target network** -- DQN target network, SAC target Q, DINO momentum teacher 本质一样 | 你在 RL 中用的 target network 和 DINO momentum teacher 是同一个 idea: 缓慢更新提供稳定学习信号 |
| 3 | **k-NN 是评估 representation 质量的最诚实指标** -- 不训 classifier, 直接反映特征空间结构 | 评估 robot visual encoder 质量时可用 k-NN 测试, 比 linear probe 更快更公平 |
| 4 | **Local-to-global 学习范式对应 robot 的 partial observation** -- 从局部视角推断全局 | 机器人只看 workspace 部分, 需理解全局 scene; DINO multi-crop 正好训练这种能力 |
| 5 | **"训练方式比架构更重要"** -- 同一个 ViT, supervised vs self-supervised 产生截然不同的特征 | Robot FM 性能瓶颈可能不在模型架构, 而在训练方式 (数据、目标函数、课程设计) |
