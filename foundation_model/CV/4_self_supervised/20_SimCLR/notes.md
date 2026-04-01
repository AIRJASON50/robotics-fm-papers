# A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) -- 学习笔记 (选读)
> 一句话: 用极简框架 (强数据增强 + 非线性投影头 + 大 batch + NT-Xent loss) 把对比学习推到接近监督学习的水平.
> 论文: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton (Google Brain), ICML 2020

## 核心想法
SimCLR 的核心贡献不是提出新模块, 而是系统性地拆解对比学习的关键要素. 对同一张图做两次随机增强得到正对, batch 内其他样本为负对, 通过 NT-Xent (normalized temperature-scaled cross entropy) loss 训练. 三个关键发现: (1) **数据增强的组合**比单一增强重要得多, random crop + color distortion 是最佳组合 (因为只 crop 时颜色直方图就能区分图片); (2) **非线性投影头** g(h) 在 loss 和 representation 之间引入信息瓶颈, 投影后的 z 丢掉了变换信息, 但投影前的 h 保留了更多有用信息; (3) **大 batch size** (4096-8192) 和 **长训练** 对对比学习的收益远大于监督学习.

## 与主线论文的关系
- **与 MoCo 互补**: MoCo 用 queue 解决大负样本问题, SimCLR 用大 batch 暴力解决, 两者共同确立了对比学习范式
- **DINO 的直接前身**: DINO 的 multi-crop augmentation 和 projection head 设计都受 SimCLR 启发
- **投影头思想影响深远**: CLIP 的 projection layer, VIP/R3M 的 embedding head 都继承了"投影前表征更好"的发现

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 数据增强定义了对比学习的 "pretext task" -- 增强越强, 模型学到的不变性越好 | 机器人视觉也需要对光照/视角/遮挡的不变性, SimCLR 说明增强策略比模型架构更重要 |
| 2 | 投影头是信息瓶颈: g(h) 丢弃了变换特异信息, h 保留了下游任务需要的全部信息 | 设计机器人 visual encoder 时, 不应该直接在最终 embedding 上加 loss, 中间层往往更有用 |
| 3 | 对比学习从大模型受益更多 -- 无监督与监督的差距随模型增大而缩小 | Scaling law 不只属于 LLM, 视觉 SSL 也有类似规律, 暗示机器人 FM 也应走大模型路线 |
| 4 | Temperature tau 很关键: 太小导致 loss 由最难负样本主导, 太大信息量不足 | 对比学习 loss 的温度超参需要仔细调, 这在 CLIP/VIP 中也反复出现 |
