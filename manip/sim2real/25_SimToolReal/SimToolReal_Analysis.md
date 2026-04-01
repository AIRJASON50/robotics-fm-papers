# SimToolReal: Object-Centric Dexterous Tool Manipulation -- 学习笔记
> 一句话: 在程序化生成的 primitive 物体上训练单一 goal-conditioned policy, zero-shot 部署到 12 种真实工具 / 24 个任务
> 论文: Kushal Kedia*, Tyler Ga Wei Lum* et al. Cornell + Stanford, 2025

## 这篇论文解决了什么问题
灵巧手的工具使用需要组合多种技能 (抓取细长物体 + 手内重定向 + 力交互保持), 但:
- Per-task reward engineering 不可扩展
- 遥操采集数据质量差 (人-机对应鸿沟)
- 现有方法只能做子问题 (抓取 OR 重定向 OR 旋转), 不能端到端

## 核心想法 (用直觉解释)
**所有工具使用任务都可以统一为"将工具依次移动到一系列目标 6D 位姿"。**

训练: 在随机生成的 cuboid/capsule 组合 (handle + head) 上, 训练一个 goal-conditioned policy 把物体移到随机目标位姿。
部署: 人类视频 -> SAM 3D + FoundationPose 提取目标位姿序列 -> 逐个位姿喂给 policy。

直觉: 掌握"把任意物体移到任意位姿"的能力, 就自然获得了工具使用所需的所有子技能。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| Object representation | 4 个 keypoint relative to palm + bounding box scale | 不用绝对坐标, 跨物体/坐标系泛化; keypoint 天然处理对称性 |
| Action space | Arm delta + Hand absolute + 强 EMA (alpha=0.1) | Delta 控小位移, absolute 适合手指; 强平滑减少高频振荡 |
| Reward | 三阶段: r_approach -> r_lift -> r_goal (with tolerance curriculum 0.075->0.01) | 自然 curriculum: 先学接近, 再学操控, 精度逐步收紧 |
| 网络 | LSTM(1024) + MLP[1024,1024,512,512] + SAPG (6 blocks) | LSTM 捕捉时序; SAPG 维护多策略种群促进探索多样性 |
| 物体生成 | Handle + Head 各选 cuboid/capsule, 随机尺寸/密度 | 不需要真实 mesh, primitive 覆盖工具空间; 只需 bounding box scale 作 obs |

## 这篇论文之后发生了什么
- DexToolBench: 24 任务 / 12 物体 / 6 工具类别的真机 benchmark, 成为灵巧手工具操作的标准评测
- 43.7% 失败来自位姿追踪丢失 (遮挡/对称/低对比度) -- 指出 perception 而非 control 是瓶颈
- 证明 sim-trained generalist policy 可以 match task-specific specialist 在训练配置上的性能

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | "Goal-conditioned pose reaching" 是 manipulation 的通用抽象 | 类比 LLM 的 next-token prediction: 一个足够通用的目标能覆盖所有下游任务 |
| 2 | 程序化物体多样性 > 精确建模少量物体 | 和 LLM pre-training 用 web-scale 粗糙数据同理: 多样性 > 精度 |
| 3 | Perception (位姿追踪) 是 sim-to-real 的真正瓶颈, 不是 control | Robotics FM 的 vision backbone 质量决定了系统天花板 |
| 4 | Progress-based reward (d* tracking) 比 L2 距离更稳定 | 只奖励"距离减小"而非距离本身, 防止策略震荡 -- 一个简单但关键的工程选择 |
