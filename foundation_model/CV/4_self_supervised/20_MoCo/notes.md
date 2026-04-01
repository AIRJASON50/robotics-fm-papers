# Momentum Contrast for Unsupervised Visual Representation Learning -- 学习笔记 (选读)
> 一句话: 用动量更新的 key encoder + queue 构建大而一致的负样本字典, 使对比学习首次在下游任务上超越监督预训练.
> 论文: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick (FAIR), CVPR 2020

## 核心想法
MoCo 将对比学习 (contrastive learning) 视为"字典查找"问题: query 应与其正样本 key 相似, 与所有负样本 key 不同. 关键洞察是字典需要同时满足 **大** (覆盖足够负样本) 和 **一致** (key 由相似的 encoder 编码). 之前的 end-to-end 方法受限于 batch size (字典小), memory bank 方法 key 来自不同 epoch 的 encoder (不一致). MoCo 用一个 **queue** 解耦字典大小与 batch size, 用 **momentum update** (m=0.999) 让 key encoder 缓慢跟随 query encoder, 同时保证大字典和编码一致性.

## 与主线论文的关系
- **ViT/MAE 的前置**: MoCo 证明了无监督预训练可以替代 ImageNet 监督预训练, 为后来 DINO/MAE 等 SSL 方法铺路
- **DINO 的对比学习前身**: DINO 的 EMA teacher 直接继承了 MoCo 的 momentum encoder 思想
- **SimCLR 的竞争对手**: SimCLR 走 "大 batch" 路线, MoCo 走 "queue + momentum" 路线, 两者共同验证了对比学习的威力

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Momentum update 是维持训练稳定性的通用技巧: 缓慢更新 target network 防止崩溃 | RL 中 target network (DQN/SAC) 用的就是同一思想, MoCo 证明它在表征学习中同样关键 |
| 2 | 无监督预训练 >= 监督预训练 (在 7/7 检测/分割任务上), 标签不是必需品 | 机器人数据缺标签, 对比学习提供了不依赖标注的预训练路径 (R3M/VIP 直接用了这个思路) |
| 3 | Queue 机制将负样本数量从 batch size 中解耦, 让小 GPU 也能做大规模对比学习 | 工程启示: 资源受限时, 用异步/缓存机制扩大有效训练规模 |
| 4 | 在 IG-1B (10亿 uncurated Instagram 图片) 上预训练依然有效, 证明对比学习对数据质量鲁棒 | 机器人数据天然 uncurated, MoCo 的鲁棒性暗示对比学习适合机器人 visual encoder 预训练 |
