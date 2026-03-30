# DreamerV3 分析笔记

**论文**: Mastering Diverse Domains through World Models
**作者**: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap (Google DeepMind / University of Toronto)
**发表**: ICLR 2023, arXiv:2301.04104
**代码**: `dreamerv3/` (JAX/Ninjax 实现)

---

## 1. Core Problem

DreamerV3 要解决的核心问题是: **如何构建一个通用 RL 算法，使其在不调整超参数的情况下跨越多种差异巨大的领域都能有效工作？**

传统 RL 算法面临的困境:

| 挑战维度 | 具体表现 |
|----------|----------|
| 领域多样性 | 连续控制、离散动作、图像输入、向量输入、稠密/稀疏奖励等场景需要不同算法 |
| 超参数敏感 | PPO、SAC 等算法从一个领域迁移到另一个领域时往往需要大量调参 |
| 信号尺度差异 | 不同环境的 reward 和 observation 数值范围差异可达数个数量级 |
| 探索与利用权衡 | 稀疏奖励需要强探索，稠密奖励需要强利用，但 entropy 系数难以统一设定 |

DreamerV3 的核心思想是 **"在想象中学习"(Learning in Imagination)**: 先学习一个环境的 world model，然后在 world model 内部生成假想轨迹(imagined trajectories)来训练 actor-critic，而非直接在真实环境中做 policy optimization。这个思路源自 Dyna 架构(Sutton, 1991)和 World Models(Ha & Schmidhuber, 2018)的传统，但 DreamerV3 通过一系列 normalization/balancing/transformation 技巧使其真正在 150+ 任务上实现了固定超参数。

**对 robotics sim-to-real 的潜力**: World model 的核心价值在于 data efficiency -- 每次真实交互的数据可以被 world model 反复"回放"和"延伸"，从而在 latent space 中生成大量虚拟经验。这对于 real-world robotics 尤其重要，因为真实交互成本高、数据稀缺。DreamerV3 的 unsupervised reconstruction loss 作为主要学习信号(而非 reward-dependent gradient)，也暗示了 world model 可以从无标签视频数据中预训练，再迁移到下游任务。

---

## 2. Method Overview

DreamerV3 由三个核心组件构成:

```
                          Environment
                              |
                         observations, rewards
                              |
                    +------------------+
                    |   World Model    |  <-- RSSM (Recurrent State-Space Model)
                    |  (Encoder/RSSM/  |
                    |   Decoder/Reward |
                    |   /Continue)     |
                    +--------+---------+
                             |
                      latent states s_t = {h_t, z_t}
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |     Actor        |          |     Critic       |
     |  pi(a_t | s_t)   |          |  v(R_t | s_t)    |
     +-----------------+          +-----------------+
```

### 2.1 RSSM (Recurrent State-Space Model)

RSSM 是 DreamerV3 world model 的核心架构，结合了确定性和随机性两种状态:

- **Sequence model**: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` -- 确定性 recurrent state (GRU-like)
- **Encoder**: `z_t ~ q(z_t | h_t, x_t)` -- 从观测中提取的后验随机表征
- **Dynamics predictor**: `z_hat_t ~ p(z_t | h_t)` -- 仅基于历史的先验预测
- **Reward predictor**: `r_hat_t ~ p(r_t | h_t, z_t)`
- **Continue predictor**: `c_hat_t ~ p(c_t | h_t, z_t)` -- 预测 episode 是否继续
- **Decoder**: `x_hat_t ~ p(x_t | h_t, z_t)` -- 重建输入

**Model state**: `s_t = concat(h_t, z_t)`，其中 `h_t` 是 deterministic recurrent state，`z_t` 是 stochastic representation (categorical, 32 variables x 64 classes)。

代码中 RSSM 的 `_core` 方法 (`dreamerv3/rssm.py` L135-159) 实现了一个基于 Block-diagonal GRU 的 sequence model:
- 将 `deter`, `stoch`, `action` 分别通过独立的 Linear + Norm + Act
- 使用 BlockLinear (分组线性层) 进行 GRU 的 gate 计算，blocks=8
- GRU 的 update gate 初始化偏向保持旧状态 (`update = sigmoid(x - 1)`)

### 2.2 Imagination (在想象中训练)

训练流程:
1. 从 replay buffer 采样真实轨迹
2. 用 encoder + sequence model 编码为 latent states
3. 从编码后的 latent states 出发，用 actor 生成动作，用 dynamics predictor 前推 H=15 步 (imagination horizon)
4. 在想象轨迹上训练 actor 和 critic

关键: imagination 过程中**不需要与真实环境交互**，所有梯度通过 world model 的可微分结构反向传播。

### 2.3 Actor-Critic in Latent Space

**Critic**: 学习预测 bootstrapped lambda-return `R^lambda_t` 的分布。使用 symexp twohot 参数化 (categorical distribution over exponentially spaced bins)，而非简单的均值回归。

**Actor**: 使用 REINFORCE estimator (对离散和连续动作统一使用)，带 return normalization 和 entropy regularizer:

```
L(theta) = -sum_t sg((R^lambda_t - v(s_t)) / max(1, S)) * log pi(a_t | s_t) + eta * H[pi(a_t | s_t)]
```

其中 `S` 是 return 的 5th-95th percentile range 的 EMA，`eta = 3e-4` 是固定的 entropy scale。

---

## 3. Key Designs

### 3.1 Symlog Predictions

**问题**: 不同环境的 observation 和 reward 尺度差异巨大。平方损失在大目标时容易梯度爆炸，Huber loss 在大目标时学习缓慢。

**方案**: 引入 symlog/symexp 变换对:

```
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)
```

- 对称、保号、在原点附近近似恒等
- 压缩大数值的梯度，但不截断
- 用于: vector observation 的编码/解码，reward predictor (symexp twohot loss)，critic (symexp twohot loss)

代码中 Encoder (`dreamerv3/rssm.py` L218-219) 对向量输入应用 symlog:
```python
squish = nn.symlog if self.symlog else lambda x: x
x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
```

**Symexp twohot**: 将 scalar 预测转为 categorical 分类问题。Bins 按 `symexp(linspace(-20, 20, 255))` 指数间距排列。网络输出 softmax logits，读出预测为 bins 的加权平均。训练用 twohot encoded targets (两个最近 bin 的线性插值) 的 cross-entropy loss。**核心优势: 梯度大小与目标值大小解耦**。

TwoHot 实现见 `dreamerv3/embodied/jax/outs.py` L273-330，其中 `pred()` 方法做了对称求和以保证初始化时预测为零。

### 3.2 Free Bits (KL Balancing)

**问题**: World model 的 KL loss 在不同视觉复杂度的环境下需要不同的权重。复杂 3D 环境需要强正则化简化表征，简单游戏需要弱正则化保留细节。

**方案**: 将 KL loss 分为两项并使用 free bits:

```
L_dyn = max(1, KL[sg(posterior) || prior])   -- weight 1.0
L_rep = max(1, KL[posterior || sg(prior)])    -- weight 0.1
```

- `L_dyn`: 训练 dynamics predictor 更好地预测后验 (stop-gradient on posterior)
- `L_rep`: 训练 encoder 使表征更可预测 (stop-gradient on prior)
- `free_nats = 1.0`: 当 KL 已经低于 1 nat 时停止优化，让训练集中在 reconstruction loss

代码见 `dreamerv3/rssm.py` L120-132:
```python
dyn = self._dist(sg(post)).kl(self._dist(prior))
rep = self._dist(post).kl(self._dist(sg(prior)))
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

### 3.3 Fixed Hyperparameters 的实现

DreamerV3 通过以下机制实现跨领域固定超参数:

| 技术 | 解决的问题 | 代码位置 |
|------|-----------|----------|
| Symlog predictions | 不同 observation/reward 尺度 | `rssm.py` Encoder, `outs.py` TwoHot |
| Free bits + KL balancing | 不同视觉复杂度 | `rssm.py` L127-129 |
| Return normalization (percentile) | 不同 reward 分布 | `agent.py` imag_loss L407-408 |
| Unimix (1% uniform) | 避免 KL spike | `rssm.py` L26, `outs.py` Categorical L212-216 |
| Critic as distribution | 多模态 return | `outs.py` TwoHot |
| Zero init for reward/value heads | 初始化 reward 偏差 | `configs.yaml` rewhead/value outscale: 0.0 |
| Slow critic (EMA target) | 稳定 value learning | `agent.py` L66-68, rate=0.02 |

### 3.4 Return Normalization

区别于传统的 advantage normalization (PPO) 或 reward normalization:

```
S = EMA(Per(R^lambda, 95) - Per(R^lambda, 5), 0.99)
```

只对超过阈值 `L=1` 的 return 做 scale down，不 scale up 小 return。这保留了稀疏奖励下的探索信号:
- 稀疏奖励: return 小 -> `max(1, S) = 1` -> 不缩放 -> entropy term 主导 -> 强探索
- 稠密奖励: return 大 -> `S > 1` -> 缩放 -> policy gradient 主导 -> 强利用

### 3.5 其他关键设计

- **Categorical latent**: `z_t` 由 32 个 categorical variable 组成，每个有 64 个 class (默认 200M 配置)。Straight-through gradient 用于反向传播。
- **Block-diagonal GRU**: 将 deter state 分为 8 个 block，block 间不通信，减少计算量。
- **单一优化器**: 所有模块用同一个 Adam 优化器联合训练 (`configs.yaml` L87)，通过 `loss_scales` 加权。
- **Replay context**: 支持从 replay buffer 恢复 RNN carry state (`replay_context: 1`)，避免 sequence 边界的 state 不连续。

---

## 4. Experiments

### 4.1 主要基准结果

| 领域 | 任务数 | 数据预算 | 对比方法 | DreamerV3 表现 |
|------|--------|----------|----------|---------------|
| Atari (200M) | 57 | 200M frames | MuZero, Rainbow, IQN | 超越 MuZero，计算资源仅为其一小部分 |
| ProcGen | 16 | 50M frames | PPG, Rainbow | 匹配 tuned PPG |
| DMLab | 30 | 100M frames | IMPALA, R2D2+ (1B steps) | 在 1/10 数据量下超越 |
| Atari100k | 26 | 400K frames | IRIS, TWM, SPR, SimPLe | 超越所有非 EfficientZero 方法 |
| Proprio Control (DMC) | 18 | 500K steps | D4PG, DMPO, MPO | SOTA |
| Visual Control (DMC) | 20 | 1M steps | DrQ-v2, CURL | SOTA |
| BSuite | 23 (468 configs) | 各异 | Boot DQN | SOTA |
| Minecraft Diamond | 1 | 100M steps | VPT, DreamerV2, PPO | 首个从零获取钻石的算法 |

### 4.2 Minecraft 突破

DreamerV3 是首个**不使用人类数据、不使用课程学习**就在 Minecraft 中从零获取钻石的算法:
- VPT: 需要人类专家数据 + 720 GPU x 9 天
- DreamerV3: 1 GPU x 9 天，从 sparse reward 自主学习
- 所有训练的 agent 都发现了钻石 (100% success rate)

### 4.3 Ablation 分析

关键发现:
1. **所有 robustness 技术都有贡献**，但每个技术只在部分任务上关键
2. **World model 的无监督重建损失是主要学习信号**: 去掉 reconstruction gradient 比去掉 reward/value gradient 对性能影响更大
3. **Scaling**: 12M -> 400M 参数，性能单调提升，且大模型需要更少的环境交互
4. **Replay ratio**: 更高的 replay ratio 提升性能和数据效率

### 4.4 Scaling Properties

| 模型大小 | RSSM deter | 总参数 | 相对性能 |
|----------|-----------|--------|---------|
| 12M | 2048 | ~12M | 基线 |
| 25M | 3072 | ~25M | + |
| 50M | 4096 | ~50M | ++ |
| 100M | 6144 | ~100M | +++ |
| 200M (default) | 8192 | ~200M | ++++ |
| 400M | 12288 | ~400M | +++++ |

来自 `configs.yaml` L120-153 的模型尺寸定义。

---

## 5. Related Work Analysis

### 5.1 与 Model-Free RL 的对比

| 维度 | Model-Free (PPO/SAC) | DreamerV3 |
|------|----------------------|-----------|
| 数据效率 | 低 (PPO 需要大量交互) | 高 (world model 生成虚拟经验) |
| 通用性 | PPO 较通用但性能不及专用算法 | 固定超参数超越专用算法 |
| 学习信号 | 完全依赖 reward | 主要依赖 unsupervised reconstruction |
| 计算模式 | 每次交互后更新 | 训练 world model + imagination |
| 表征学习 | 隐式 (value function 驱动) | 显式 (autoencoding + dynamics) |

### 5.2 与其他 World Model 方法的对比

| 方法 | World Model 类型 | 规划方式 | 局限 |
|------|-----------------|----------|------|
| PlaNet / RSSM (2018) | RSSM | CEM planning | 需要在线规划，慢 |
| DreamerV1 (2019) | RSSM + continuous latent | Imagination + AC | KL 权重需要调 |
| DreamerV2 (2020) | RSSM + categorical latent | Imagination + AC | 仍需按领域调参 |
| MuZero (2019) | Value prediction model | MCTS | 计算昂贵、复杂、未开源 |
| SimPLe (2019) | Video prediction | Planning | 仅适用简单环境 |
| **DreamerV3** (2023) | **RSSM + robust techniques** | **Imagination + AC** | **固定超参数** |

DreamerV3 相对于 DreamerV2 的关键改进:
- Symlog predictions 替代手动调整的 loss scales
- Free bits 替代手动调整的 KL weight
- Return normalization (percentile-based) 替代 advantage normalization
- Critic 输出分布 (symexp twohot) 替代均值回归
- 统一使用 REINFORCE (离散+连续) 替代不同 estimator

---

## 6. Limitations & Future Directions

### 6.1 论文明确的局限

- **计算成本**: 单个 Minecraft 训练需 1 GPU x 9 天，对于更复杂的 real-world 场景可能不够
- **探索**: 依赖 entropy regularization，没有专门的探索机制 (如 intrinsic motivation)
- **RSSM 架构**: 仍然是基于 RNN 的，sequence length 受限于 batch_length=64

### 6.2 从代码推断的局限

- **Replay buffer 大小**: 默认 `size: 5e6` (`configs.yaml` L41)，对于长时间训练可能不够
- **Imagination horizon**: 固定 H=15 步 (`imag_length: 15`)，对于需要很长规划的任务可能不足
- **图像尺寸**: 大多数配置使用 64x64 或 96x96 的图像，对于精细视觉任务分辨率较低
- **单一优化器**: 所有模块共享一个 optimizer (lr=4e-5)，无法为 world model 和 policy 分别调整学习率

### 6.3 未来方向

1. **从无标签视频预训练 world model**: 论文发现 unsupervised reconstruction 是主要学习信号，这意味着可以从 internet-scale 视频数据预训练，类似 vision foundation model
2. **跨领域统一 world model**: 一个 world model 服务于多个环境/任务
3. **Transformer 替代 RSSM**: 用 Transformer 替代 RNN 以支持更长的 context (后续工作如 STORM, TransDreamer 已在探索)
4. **Real-world robotics**: World model 的 data efficiency 和 imagination-based training 天然适合 sim-to-real，但需要解决 visual domain gap

---

## 7. Paper vs Code Discrepancies

通过对比论文描述和代码实现，发现以下差异:

### 7.1 代码中存在但论文未提及的特性

| 特性 | 代码位置 | 说明 |
|------|----------|------|
| Block-diagonal Linear | `rssm.py` L150, `nn.BlockLinear` | RSSM core 使用分块线性层，blocks=8，论文未详细描述 |
| Action normalization | `rssm.py` L137: `action /= sg(max(1, abs(action)))` | 对 action 做归一化防止大动作影响 dynamics |
| Replay value loss (repval) | `agent.py` L218-235, `configs.yaml` L115: `repval_loss: True` | 在 replay 数据上额外训练 critic，weight=0.3 |
| Replay context | `agent.py` L312-340, `configs.yaml` L15: `replay_context: 1` | 从 replay buffer 恢复 RNN carry state |
| Advantage normalization | `agent.py` L72: `advnorm` | 代码中有 advnorm 模块 (默认 impl=none)，论文未提及 |
| Value normalization | `agent.py` L71: `valnorm` | 对 value target 做额外 normalization |
| Reward gradient control | `configs.yaml` L114: `reward_grad: True` | 控制 reward prediction 梯度是否流回 encoder |
| Adaptive gradient clipping (AGC) | `agent.py` L345-346, `opt.clip_by_agc(0.3)` | 使用 AGC 替代传统 gradient clipping |
| Contdisc (continuous discount) | `configs.yaml` L107: `contdisc: True` | 用 `1 - 1/horizon` 替代 `gamma=0.997`，论文中提到 gamma=0.997 但代码实际用 `horizon=333` 计算 |
| Custom optimizer | `agent.py` L357-379 | 使用自定义 RMS + Momentum (非标准 Adam) |

### 7.2 论文描述与代码实现的差异

| 论文描述 | 代码实现 |
|----------|----------|
| 使用 Adam optimizer | 实际使用 AGC + RMS scaling + Momentum (类 Adam 但不完全相同)，见 `agent.py` L357-379 |
| gamma = 0.997 | 代码使用 `horizon = 333` (`1/(1-gamma) = 333.3`)，通过 `contdisc` flag 在 continue predictor 中实现 |
| 论文公式中的 beta_pred=1, beta_dyn=1, beta_rep=0.1 | 代码中通过 `loss_scales` 统一管理: `dyn: 1.0, rep: 0.1`，但增加了 `repval: 0.3` |
| Slow target network 用于计算 return | 代码中 `slowtar: False`(默认)，即直接用 current value 而非 slow target 计算 return |
| 论文描述 entropy scale eta=3e-4 | 代码一致: `actent: 3e-4` |

### 7.3 Symmetric TwoHot pred() 实现

`outs.py` L286-309 中 `TwoHot.pred()` 做了特殊处理: 标准的 `(probs * bins).sum(-1)` 会因浮点累积误差导致初始化时预测不为零。代码采用对称求和 (从两端向中间) 确保数值对称性，这个细节论文未提及但对初始化阶段很重要。

---

## 8. Cross-Paper Comparison

### 8.1 DreamerV3 vs Diffusion Policy

| 维度 | DreamerV3 | Diffusion Policy |
|------|-----------|-----------------|
| **范式** | Model-based RL (World Model + Imagination) | Behavior Cloning with diffusion denoising |
| **数据需求** | 在线交互 (self-play) | 离线专家演示数据 |
| **动作生成** | 单步 sampling from policy network | 多步 denoising process (迭代去噪) |
| **世界理解** | 显式学习环境 dynamics | 不学习 dynamics，直接学 state->action mapping |
| **多模态动作** | Entropy regularizer 鼓励多样性 | 天然建模多模态分布 (diffusion 优势) |
| **实时性** | 单次前向传播出动作 | 需要多步去噪 (~100 步)，延迟较高 |
| **泛化性** | 固定超参数跨 150+ 任务 | 每个任务需要专门训练 |
| **对 robotics 价值** | Data efficiency 高，适合 exploration | 适合 demonstration-rich 场景 |
| **核心局限** | 需要在线交互 | 需要高质量演示数据 |

**互补关系**: DreamerV3 的 world model 可以作为 Diffusion Policy 的数据增强器 -- world model 生成虚拟轨迹扩充训练数据。反过来，Diffusion Policy 的多模态动作建模能力可以替代 DreamerV3 中的简单 Gaussian/Categorical actor。

### 8.2 DreamerV3 vs Decision Transformer

| 维度 | DreamerV3 | Decision Transformer |
|------|-----------|---------------------|
| **范式** | Model-based RL (world model) | Offline RL as sequence modeling |
| **核心架构** | RSSM (RNN) + MLP actor/critic | GPT-like Transformer (autoregressive) |
| **训练方式** | 在线 + replay buffer | 纯离线 (offline dataset) |
| **条件输入** | State -> action | (Return-to-go, state, action) sequence |
| **环境模型** | 显式学习 dynamics + reward + continue | 不学习 dynamics，隐式建模 |
| **推理时控制** | Sample from policy | 指定 desired return-to-go |
| **长程依赖** | 受限于 RSSM 的 memory (GRU bottleneck) | Transformer attention 可处理长 context |
| **数据效率** | 高 (imagination augmentation) | 依赖数据集质量和规模 |
| **探索能力** | 有 (entropy regularizer + online interaction) | 无 (受限于 offline dataset coverage) |

**关键区别**: Decision Transformer 将 RL 问题转化为 sequence prediction，不需要 value function 或 policy gradient。但它完全依赖离线数据且无法做在线探索。DreamerV3 则在在线交互中学习 world model 并在想象中优化 policy，具有更强的自主学习能力。

### 8.3 三者的设计哲学对比

| 哲学维度 | DreamerV3 | Diffusion Policy | Decision Transformer |
|----------|-----------|-----------------|---------------------|
| 对世界的理解 | 显式建模 (predict future) | 不需要 (direct mapping) | 隐式建模 (sequence pattern) |
| 学习信号来源 | Reconstruction + reward | Demonstration matching | Sequence prediction |
| 对人类知识的依赖 | 最低 (可从零探索) | 中等 (需要演示) | 高 (需要好的 offline data) |
| 适用场景 | 自主探索 + long-horizon | 精确操作 + 短 horizon | 有丰富离线数据的决策 |

### 8.4 对 Robotics Sim-to-Real 的影响

DreamerV3 对 sim-to-real 的潜在价值在于:
1. **高 data efficiency**: 在真实机器人上样本珍贵，world model 可从少量交互中学习并在 imagination 中大量扩展
2. **Domain adaptation 潜力**: World model 的 latent space 可能学到 domain-invariant 的 dynamics，有助于 sim-to-real transfer
3. **Unsupervised pre-training**: 可从无标签的机器人运行视频中预训练 world model，然后在下游任务中 finetune

但挑战包括:
- 真实世界的视觉复杂度远超 Atari/DMC
- 接触力学 (contact dynamics) 难以从视觉中准确推断
- World model 的误差会在 imagination rollout 中累积 (compounding error)
