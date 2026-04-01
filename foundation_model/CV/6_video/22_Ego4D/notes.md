# Ego4D: Around the World in 3,000 Hours of Egocentric Video -- Takeaway Notes

> 一句话: 3,670 小时、931 人、74 个地点的第一人称日常生活视频数据集 + 5 个 benchmark, 为 egocentric perception 研究奠基, 直接催生了 R3M/VIP 等 robot visual representation 工作。

## 核心贡献

1. **前所未有的 egocentric video 数据集**:
   - 3,670h video, 931 camera wearers, 74 locations, 9 countries
   - 多模态: RGB + audio + 3D scans + gaze + stereo + IMU + narrations (3.85M sentences, 13.2 sentences/min)
   - 比此前最大 ego dataset (EPIC-Kitchens 100h) 大一个数量级
2. **5 个 Benchmark 覆盖 past/present/future**:
   - Episodic Memory: "我把钥匙放哪了?" (NLQ / VQ / MQ)
   - Hands & Objects: object state change detection (pre → PNR → post)
   - Audio-Visual Diarization + Social Interactions: 谁在说什么 / looking at me
   - Forecasting: locomotion / hand movement / short-term interaction / long-term action
3. **以 daily life scenarios 为驱动的数据收集**: 不是 scripted lab 场景, 而是参考美国劳工统计局数据, 覆盖家务/烹饪/运动/社交/通勤等真实日常活动

## 为什么重要

- **Egocentric = Embodied AI 的视角**: 第三人称 (Kinetics/ImageNet) 是旁观者视角; 第一人称是 agent 的真实 observation -- robot 看到的就是 egocentric video
- **R3M / VIP 的数据基础**: R3M 用 Ego4D 做 time-contrastive + language-conditioned representation learning; VIP 用 Ego4D 做 value function pre-training -- 没有 Ego4D 就没有这两个 robotics visual representation 工作
- **Hands & Objects benchmark 直接对应 manipulation**: object state change (pre/PNR/post) 就是 robot manipulation 的核心 -- 物体状态变化 = 操作成功

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 行动项 |
|---|----------|--------|
| 1 | **Egocentric video 是 robot pre-training 的最佳 proxy data**: 人的第一人称 hand-object interaction 是大规模 manipulation data 的唯一来源 | 了解 R3M/VIP 如何在 Ego4D 上 pre-train, 以及表征如何 transfer 到 robot policy |
| 2 | **Object state change > action recognition**: Ego4D 的 Hands & Objects benchmark 关注物体状态变化而非动作标签 -- 因为同一个 state change 可以有多种实现方式 | Reward 设计应关注 state change (结果) 而非 action similarity (过程) |
| 3 | **Narrations 是 weak language supervision**: 13.2 sentences/min 的 dense narration 是 vision-language grounding 的天然训练数据 | Robot teleoperation 数据也可以加 narration 做 language-conditioned policy |
| 4 | **Forecasting benchmark = world model**: 预测未来 hand movement / object interaction 就是 world model 的雏形 | 连接到 DreamerV3 / UniSim 等 world model 工作 |
| 5 | **Domain gap 仍然存在**: Ego4D 是人手 + 日常物体, robot 是 gripper + 工业/实验室物体 | Pre-train on Ego4D → fine-tune on robot data 是必要的两步 |

## 与知识库其他内容的关联

- **R3M** (`robotics/visual_repr/`): 在 Ego4D 上用 time-contrastive learning + language alignment 训练 visual encoder, 然后 frozen 迁移到 robot manipulation
- **VIP** (`robotics/visual_repr/`): 在 Ego4D 上训练 value function 作为 zero-shot reward, 利用 video 中的 temporal distance 做 goal-conditioned representation
- **VideoMAE** (`CV/6_video/22_VideoMAE`): 在 Kinetics 等 third-person video 上做 masked video pre-training; Ego4D + VideoMAE 的组合 = egocentric video 的 self-supervised backbone
- **DreamerV3** (`robotics/world_model/23_DreamerV3`): Ego4D Forecasting benchmark 和 world model 目标一致 -- 从 past observation 预测 future
- **CLIP** (`CV/2_vl_alignment/21_CLIP`): Ego4D narrations 可作为 egocentric image-text pairs, 训 ego-CLIP
