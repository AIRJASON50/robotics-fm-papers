# Robotic World Model (RWM) -- 学习笔记
> 一句话: 用 dual-autoregressive GRU 学习 neural network simulator (无 domain-specific inductive bias)，结合 MBPO-PPO 在 "想象中" 训练 policy，zero-shot 部署到真实四足和人形机器人
> 论文: Chenhao Li, Andreas Krause, Marco Hutter (ETH Zurich), 2025

## 这篇论文解决了什么问题
Model-free RL (PPO/SAC) 效果好但 sample efficiency 极差，不适合 real-world learning。World model 是解决方案: 学一个环境动力学模型，在 "想象" 中训 policy。但现有 world model 有三个问题:
1. **Domain-specific bias**: Dreamer 系列、FLD 等需要针对特定任务设计 state representation 或网络结构
2. **Long-horizon error accumulation**: autoregressive prediction 误差指数级增长，长 rollout 不可靠
3. **Sim2real**: 很少有 world model work 真正在 real robot 上验证

RWM 的目标: 一个通用的 (no task-specific design)、long-horizon 可靠的、能直接 sim2real 的 world model framework。

## 核心想法 (用直觉解释)
用 **self-supervised autoregressive training** 训一个 GRU-based world model。关键区别于 teacher-forcing: 训练时模型用自己的预测结果 (而非 ground truth) 作为下一步输入，这样训练分布和推理分布一致，减少 error accumulation。然后在这个 learned world model 中用 PPO 训 policy (MBPO 风格: 混合 real data 和 imagined data)。

直觉: 普通 world model 训练像 "考试时每道题都给标准答案再做下一题"，实际部署时却要 "连续做题不给答案"。RWM 的训练方式是 "做完一题用自己的答案继续做下一题"，训练和部署保持一致。

## 关键设计决策
1. **Dual-autoregressive mechanism**: (i) Inner autoregression: GRU hidden state 在 history horizon M 内逐步更新，编码历史信息; (ii) Outer autoregression: forecast horizon N 步的预测被 feed back 作为输入。Inner 处理 partial observability，outer 处理 long-horizon prediction。
2. **Self-supervised training (非 teacher-forcing)**: 训练 loss = 1/N * sum(alpha^k * [L_o + L_c])，其中 alpha 是 decay factor 让模型更关注近期预测的准确性。L_c 是 privileged info (如 contact) 的预测 loss，implicitly 编码重要物理信息。
3. **无 domain-specific inductive bias**: 用通用 GRU (不需要知道是四足还是人形)，输入是 observation-action pairs。跨环境通用 -- 从 cartpole 到 ANYmal 到 G1 humanoid 都用同一架构。
4. **MBPO-PPO**: 不完全依赖 world model (可能不准)，也不完全依赖 real data (太少)。混合两者: 先在 real env 收集数据训 world model，再在 imagined env 中用 PPO 训 policy，周期性更新。
5. **Gaussian output**: 模型预测下一步 observation 的 mean 和 std (Gaussian distribution)，自然处理 stochasticity。Reparameterization trick 保证梯度可传播。

## 这篇论文之后发生了什么
- 首个在 quadruped + humanoid 上验证的 general-purpose world model (不需要 domain knowledge)
- 与 DreamerV3 的比较: RWM 在 long-horizon prediction 上更可靠 (因为 autoregressive 训练 vs teacher-forcing)
- 局限: GRU 容量可能不如 Transformer; 高维 visual input 尚未处理; world model 的 prediction accuracy 仍然是 bottleneck

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **训练时的分布必须匹配推理时的分布** -- autoregressive training 消除 train/test mismatch，这和 LLM 的 exposure bias 问题是同一件事 | 任何 sequential prediction 模型都要考虑: 训练时用 GT 还是自己的预测? |
| 2 | **World model 是 sim2real 的另一条路** -- 不修仿真器 (ASAP)，不 randomize (DR)，而是直接从 real data 学一个 "neural simulator" | 灵巧手如果能收集少量 real data，world model 可能比 DR 更高效 |
| 3 | **No domain-specific bias = scalability** -- 同一个架构从 cartpole 到 humanoid，不需要人工设计 state representation | FM 的核心理念: 通用架构 + 大数据，而非特化设计 |
| 4 | **Privileged info prediction 是隐式编码物理知识的好方法** -- 预测 contact 不是为了用它，而是为了让 hidden state 包含 contact 信息 | 灵巧手 world model 可以加入 tactile prediction 作为 auxiliary loss |
| 5 | **MBPO 混合策略 = 安全网** -- 不完全信任 world model，用 real data 做 ground truth check | 实际部署 world model 时，real data 应始终参与训练 loop |
