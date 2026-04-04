# RT-1: Robotics Transformer for Real-World Control at Scale -- 学习笔记

> 一句话: 用 130K 真实 episodes + 700 tasks 训练单个 Transformer, 第一次在工业级规模上证明 "大数据 + 大模型 = 可泛化的 robot policy"。
> 论文: Brohan, Hausman et al. (Google / Everyday Robots), RSS 2023, arXiv:2212.06817

## 这篇论文解决了什么问题

2022 年之前, robot policy 都是在几百到几千条 demo 上训练的小模型, 每个任务一个 policy, 换任务换物体就不行。核心矛盾: 数据太少→模型太小→不能泛化→收更多数据也只是解决当前任务。

RT-1 的问题: 如果我们像 NLP/CV 那样堆数据堆模型, robot 能不能也出现泛化?

## 核心想法 (用直觉解释)

答案是 yes。Google 用 13 台真实机器人花 17 个月采集了 130K episodes (700+ 任务), 然后训练一个足够大的 Transformer 端到端吃下所有数据。

架构极其简单: 图像→EfficientNet 提特征→TokenLearner 压缩成 8 个 token→Transformer 序列建模→输出 11 维离散动作 token (每维 256 bin)。以 3Hz 运行。

关键不是架构创新, 而是**规模验证**: seen tasks 97% 成功率, unseen objects 76%, 和 SayCan 组合做长序列任务 67%。更重要的是 data scaling 曲线: 从 100 tasks 到 700+ tasks, 性能 log-linear 增长, 甚至不相关的任务数据也能带来正向迁移。

## 关键设计决策

| 决策 | 选择 | 为什么 |
|------|------|--------|
| 视觉编码器 | EfficientNet (CNN) | 2022 年 ViT 还不是 robot 标配, EfficientNet 够用且快 |
| Token 压缩 | TokenLearner (81→8 tokens) | 不压缩 Transformer 推理太慢, 无法实时 3Hz |
| 动作表示 | 11D × 256 bin 离散化 | categorical cross-entropy 比回归 loss 更稳定, 但精度有限 |
| 数据 | 130K real episodes, 不用 sim | 纯真实数据, 证明 real data scaling 可行 |
| 不用 VLM | 自训 EfficientNet, 不继承 web 知识 | RT-1 没想到 web 知识能迁移 → RT-2 的核心突破就是发现这一点 |

## 这篇论文之后发生了什么

RT-1 留下了两个明确的 "下一步":
1. **为什么不用 VLM?** → RT-2 (2023) 回答: 用 VLM 替代 EfficientNet, web 知识直接迁移到 robot, unseen 物体泛化大幅提升
2. **256-bin 精度够吗?** → pi_0 (2024) 回答: 不够, 用 Flow Matching 输出连续动作

RT-1 + RT-2 + Open X-Embodiment 定义了整个 VLA 范式。核心团队 (Hausman, Ichter 等) 后来离开 Google 创办了 PI。

## 对你 (RL→FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|---------|
| 1 | **Data scaling 在 robot 上有效, 且存在跨任务正迁移** | 你的灵巧手数据不必只包含目标任务, 其他操作任务的数据也能帮助泛化 |
| 2 | **离散 token (256-bin) 够用但不够好** | 你的 PPO 用 Gaussian 连续输出, 比 256-bin 更灵活; 但 Flow Matching (pi_0) 是更好的方案 |
| 3 | **关键不是架构而是数据规模** | RT-1 的架构很普通 (EfficientNet+Transformer), 是 130K episodes 让它 work |
| 4 | **RT-1 没用 VLM 是最大的遗憾, RT-2 修正了这一点** | 如果你做 VLA, 直接用 VLM backbone, 不要像 RT-1 一样从头训视觉编码器 |
