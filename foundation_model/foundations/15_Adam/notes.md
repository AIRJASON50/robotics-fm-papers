# Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015) -- Takeaway Notes

> 一句话: 结合 momentum (一阶矩) 与 RMSProp (二阶矩), 实现自适应学习率, 成为深度学习默认优化器.

## 核心贡献
- 维护梯度的指数移动平均 (m_t, 一阶矩) 和梯度平方的指数移动平均 (v_t, 二阶矩)
- 引入 bias correction 解决初始阶段估计偏小的问题
- 超参数 (beta1=0.9, beta2=0.999, lr=1e-3) 几乎不用调, 对大多数任务开箱即用

## 为什么重要
Adam 是目前几乎所有深度学习训练的默认选择. 从 Transformer 到 PPO 的 policy/value
network, 都用 Adam 或其变体 (AdamW). 它让实践者不再需要精心调 learning rate schedule,
极大降低了训练深度网络的门槛.

## 对你 (RL->FM) 的 Takeaway
- PPO 训练中 Adam 的 epsilon 参数 (默认 1e-8 vs RL 常用 1e-5) 对训练稳定性有显著影响;
  IsaacLab/IsaacGym 的 PPO 实现通常会调大 epsilon.
- 大模型 pre-training 用 AdamW (decoupled weight decay), fine-tune 阶段有时换用
  SGD + momentum 以获得更好泛化 -- 理解 Adam 的机制才能做出正确选择.

## 与知识库其他内容的关联
- 15_BatchNorm, 16_LayerNorm: 与 Adam 一起构成"让深度网络可训练"的三大基础设施
- 17_PPO, 18_SAC: 两者的 actor/critic 网络都默认使用 Adam
- 17_Transformer: Transformer 训练标配 Adam + warmup lr schedule
