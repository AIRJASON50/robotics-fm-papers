# OmniRetarget: Interaction-Preserving Retargeting -- 学习笔记
> 一句话: 用 interaction mesh + 硬约束优化生成高质量 retargeting 数据, 使下游 RL 只需 5 个 reward term 就能 zero-shot 部署
> 论文: Lujie Yang*, Xiaoyu Huang* et al. Amazon FAR + MIT + UCB + Stanford + CMU, 2025

## 这篇论文解决了什么问题
现有 humanoid retargeting 管线有两个致命缺陷:
1. **物理不可行**: foot skating、穿透等 artifact -- 因为优化无硬约束
2. **交互缺失**: 只做关键点匹配, 不建模人-物/人-环境的接触关系

后果: 下游 RL 被迫用大量 ad-hoc reward engineering 来弥补低质量参考轨迹。
**核心论点**: 与其在 RL 端补偿数据质量问题, 不如在数据生成端从根源解决。

## 核心想法 (用直觉解释)
**Interaction Mesh**: 把 robot 关节 + 物体/环境采样点一起做 Delaunay 四面体化, 构成一个共享拓扑的体积网格。优化目标是最小化 Laplacian deformation energy -- 保持每个点相对其邻居的局部几何关系不变。

直觉: 如果人手距离桌面 5cm, retarget 后机器人手也应该距离桌面 ~5cm。共享拓扑自然保持空间关系。

再加硬约束: non-penetration (signed distance), joint limits, velocity limits, foot sticking -- 用 SQP 逐帧求解。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| 优化目标 | Laplacian deformation energy + temporal smoothness | 保持局部空间关系, 不仅仅是关键点位置匹配 |
| 硬约束 vs 软约束 | SQP 硬约束 (non-penetration, joint/vel limits, foot sticking) | VideoMimic 用软约束调参困难; 硬约束有物理保证 |
| 数据增广 | 从单个示范变 object pose/shape/terrain height/embodiment | Retargeting 天然支持: 改环境参数重新求解即可 |
| 下游 RL | 仅 5 reward terms + 4 DR 项, 无 curriculum | 数据好 -> RL 简单; 直接复用 BeyondMimic 超参, 不调参 |
| Multi-sim 支持 | IsaacGym + IsaacSim + MJWarp + MuJoCo | 代码比论文丰富; 还有 FastSAC (off-policy) 支持 |

## 这篇论文之后发生了什么
- 开源 HoloSoma 框架, 成为 humanoid whole-body control 的新基础设施
- 代表"数据质量 > reward engineering"范式的胜利: 和 HDMI (reward 补偿派) 形成鲜明对比
- Wall-flip (30s 连续 parkour) 和 zero-shot sim-to-real 在 G1 上的演示刷新了 loco-manipulation SOTA

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | "数据好 -> 算法简单" 是通用规律 | 和 LLM 一样: 高质量 pre-training data >> 复杂训练 trick |
| 2 | Interaction mesh = 保持关系的 representation | 类比 graph neural network: 编码实体间空间关系而非绝对位置 |
| 3 | 从单个 demo 增广出多变体, 比纯 DR 有效 | RL 无法探索超出参考轨迹太远的行为; kinematic augmentation 直接给正确答案 |
| 4 | 硬约束 > 软约束 (retargeting 层面) | 带约束的 optimization 产出的数据可以让后续学习"轻装上阵" |
