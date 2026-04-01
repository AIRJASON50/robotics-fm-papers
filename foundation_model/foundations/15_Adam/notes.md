# Adam: A Method for Stochastic Optimization -- 学习笔记
> 一句话: 结合 momentum (一阶矩估计) 和 RMSProp (二阶矩估计) 并加 bias correction, 实现自适应学习率, 成为深度学习默认优化器.
> 论文: Diederik P. Kingma (OpenAI), Jimmy Lei Ba (U of Toronto), 2015, ICLR 2015

## 这篇论文解决了什么问题
SGD 需要精心调 learning rate 和 schedule; AdaGrad 对稀疏梯度好但学习率单调递减导致训练过早停滞; RMSProp 修复了 AdaGrad 但缺乏 bias correction, 且没有理论收敛保证. 深度学习需要一个 "开箱即用" 的优化器: 对稀疏/稠密梯度都有效, 对超参数不敏感, 适合大规模高维非凸优化.

## 核心想法 (用直觉解释)
维护两个指数移动平均: m_t 追踪梯度方向 (一阶矩, 类似 momentum), v_t 追踪梯度大小 (二阶矩, 类似 RMSProp). 更新时用 m_t / sqrt(v_t) -- 方向由 momentum 决定, 步长自动适应: 历史梯度大的参数走小步, 历史梯度小的参数走大步. 关键创新是 bias correction: 初始时 m_0=v_0=0, 前几步的估计严重偏低, 除以 (1-beta^t) 来修正.

## 关键设计决策
- **双指数移动平均**: m_t = beta1 * m_{t-1} + (1-beta1) * g_t (一阶矩), v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2 (二阶矩). 默认 beta1=0.9, beta2=0.999
- **Bias correction**: m_hat = m_t / (1-beta1^t), v_hat = v_t / (1-beta2^t). 在 t 很小时影响巨大; 没有 bias correction 等价于 RMSProp with momentum, 会导致初始步过大
- **Trust region 性质**: 有效步长 |Delta_t| 约等于 alpha (learning rate), 不受梯度 scale 影响. 这意味着 alpha=0.001 时参数空间中每步移动约 0.001, 与梯度大小无关
- **AdaMax 变体**: 将 L2 范数推广为 L-infinity 范数, 用 max 操作替代指数移动平均, 更简单但实际用得少

## 这篇论文之后发生了什么
AdamW (Loshchilov 2019) 将 weight decay 与 Adam 解耦 (decoupled weight decay), 成为大模型训练标配. LAMB/LARS 针对大 batch 分布式训练做适配. Lion (2023) 用符号更新 (sign of momentum) 进一步简化. 但 Adam/AdamW 至今仍是 Transformer 训练的默认选择. 在 RL 中, PPO/SAC 的 actor-critic 网络几乎都用 Adam, 但 epsilon 通常调大 (1e-5 而非 1e-8).

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Adam 的 epsilon 对 RL 训练稳定性有显著影响 | IsaacLab/IsaacGym 的 PPO 实现通常将 epsilon 调大到 1e-5, 因为 RL 的 value function 梯度波动大 |
| 2 | AdamW (decoupled weight decay) 是大模型标配 | pre-training VLA 时用 AdamW, fine-tune 阶段有时换 SGD+momentum 追求更好泛化 -- 理解机制才能选对 |
| 3 | Bias correction 在训练初期至关重要 | 当 warm-up 和 Adam bias correction 同时存在时, 不要重复 "慢启动", 否则初期学习率过低导致浪费算力 |
| 4 | 自适应学习率解放了超参调优 | 但 RL 中 actor 和 critic 的最佳学习率往往不同, 需要分别设置而非共用一个 Adam 实例 |
