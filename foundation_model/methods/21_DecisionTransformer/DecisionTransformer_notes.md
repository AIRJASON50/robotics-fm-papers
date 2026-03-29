# Decision Transformer: Reinforcement Learning via Sequence Modeling -- 分析笔记

> Lili Chen*, Kevin Lu*, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch (2021)
> arXiv: 2106.01345

---

## 1. Core Problem: RL 和序列建模的关系

### 1.1 传统 RL 的局限

传统 RL 方法（如 Q-learning、Policy Gradient）将决策建模为 Markov Decision Process (MDP)，核心依赖 Bellman equation 进行 value estimation 或 policy optimization。这些方法面临以下问题：

- **Credit assignment 困难**: 稀疏奖励环境下，reward signal 需要通过 temporal difference 逐步传播，容易导致训练不稳定
- **Bootstrapping 误差累积**: 基于 Bellman backup 的方法（如 DQN、TD3）在 offline 设置下因 distribution shift 而产生严重的 overestimation
- **探索效率低**: 需要专门设计 exploration strategy（如 epsilon-greedy、entropy bonus），增加工程复杂度

### 1.2 核心洞察

Decision Transformer (DT) 的核心思想是：**将 RL 重新定义为一个条件序列生成问题 (conditional sequence modeling)**。

具体而言：

- 传统 RL 问 "如何最大化累积奖励"
- DT 问 "给定期望的回报目标，应该采取什么行动"

这一转换使得 NLP 领域中成熟的 Transformer/GPT 架构可以直接迁移到 RL 任务。不需要 value function、不需要 policy gradient、不需要 temporal difference learning -- 只需要 supervised learning 就能完成决策任务。

### 1.3 为什么 Transformer 适合 RL

| 特性 | NLP 场景 | RL 场景 |
|------|----------|---------|
| 序列结构 | token 序列 (word/subword) | trajectory 序列 (return, state, action) |
| 长程依赖 | 文本中远距离的语义关联 | 延迟奖励 (delayed reward) 的信用分配 |
| 条件生成 | 给定 prompt 生成文本 | 给定 target return 生成 action |
| Self-attention | 计算 token 间的相关性 | 自动识别 trajectory 中的关键 timestep |

Self-attention 机制天然能处理长程依赖，不需要像 temporal difference 方法那样逐步传播信息。这意味着在稀疏奖励场景下，DT 可以直接 "看到" 远处的 reward 并据此做出当前决策。

---

## 2. Method Overview: 将 RL 重新定义为序列建模

### 2.1 Trajectory 表示

DT 将一条 trajectory 表示为一个有序 token 序列：

```
tau = (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_T, s_T, a_T)
```

其中：
- `R_t = sum_{t'=t}^{T} r_{t'}` 是从时间步 t 开始的 return-to-go（即未来累积回报）
- `s_t` 是状态
- `a_t` 是动作

**关键设计**: 使用 return-to-go (RTG) 而非即时奖励 `r_t`，因为 RTG 直接编码了 "期望获得的未来总回报"，使模型能以目标驱动的方式生成 action。

### 2.2 训练过程

训练目标极其简单：

1. 从 offline dataset 中采样 trajectory 片段
2. 构造输入序列 `(R_t, s_t, a_t, R_{t+1}, s_{t+1}, ...)`
3. 训练 GPT 模型预测下一个 action token
4. Loss 函数仅为 **action prediction 的 MSE**（连续动作）或 **cross-entropy**（离散动作）

```python
# gym/experiment.py, line 254
loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)
```

### 2.3 推理过程 (Return-Conditioned Generation)

推理时的流程：

1. 设定一个目标 return `R_target`（如专家级别的分数）
2. 获取当前状态 `s_t`
3. 将 `(R_target, s_t)` 作为 prompt 输入模型
4. 模型输出 action `a_t`
5. 执行 action，获得 reward `r_t`
6. 更新 `R_{target} = R_{target} - r_t`（减去已获得的 reward）
7. 重复步骤 2-6

```python
# gym/decision_transformer/evaluation/evaluate_episodes.py, line 124-127
if mode != 'delayed':
    pred_return = target_return[0,-1] - (reward/scale)
else:
    pred_return = target_return[0,-1]
```

这个过程的直觉是：模型被训练为 "给定我还需要获得 X 分的情况下，当前应该做什么"。通过调整 `R_target`，用户可以控制 agent 的行为激进程度。

---

## 3. Key Designs: GPT 架构在 RL 中的适配

### 3.1 GPT-2 骨干网络

DT 使用 GPT-2 架构，但做了一个关键修改：**移除了 positional embedding**，改为使用自定义的 timestep embedding。

```python
# gym/decision_transformer/models/trajectory_gpt2.py, line 520-521
self.wte = nn.Embedding(config.vocab_size, config.n_embd)
# self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # REMOVED

# line 680-681
# position_embeds = self.wpe(position_ids)             # REMOVED
hidden_states = inputs_embeds  # + position_embeds      # no position embedding added
```

原因在于：NLP 中 positional embedding 编码的是 token 在句子中的位置，而 RL 中需要区分的是 *timestep*（一个 timestep 包含 R, s, a 三个 token）和 *modality*（return / state / action）。

### 3.2 三模态 Embedding 与 Timestep Embedding

每种模态有独立的 embedding head，然后共享同一个 timestep embedding：

```python
# gym/decision_transformer/models/decision_transformer.py, line 40-43
self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
self.embed_return = torch.nn.Linear(1, hidden_size)
self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
```

Timestep embedding 被加到所有三种模态上：

```python
# line 68-71
state_embeddings = state_embeddings + time_embeddings
action_embeddings = action_embeddings + time_embeddings
returns_embeddings = returns_embeddings + time_embeddings
```

### 3.3 Token 交织排列

三种模态按 `(R_t, s_t, a_t)` 的顺序交织排列，形成长度为 `3K` 的序列（K 是 context length）：

```python
# line 75-77
stacked_inputs = torch.stack(
    (returns_embeddings, state_embeddings, action_embeddings), dim=1
).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
```

这种交织排列的好处是：在 causal attention mask 下，预测 `a_t` 时模型能看到 `(R_t, s_t)` 但看不到未来的 token。

### 3.4 Prediction Head 设计

Action prediction 使用 state token 的输出（位置索引 1），因为在自回归序列 `(R_t, s_t, a_t)` 中，state 是 action 之前的最后一个已知信息：

```python
# line 99
action_preds = self.predict_action(x[:,1])  # predict next action given state
```

代码中也定义了 state 和 return 的 prediction head，但论文中并未使用它们（注释也明确指出）：

```python
# line 47
# note: we don't predict states or returns for the paper
```

### 3.5 Atari vs Gym 的架构差异

| 设计要素 | Gym (连续控制) | Atari (离散控制) |
|----------|---------------|-----------------|
| GPT 实现 | HuggingFace GPT2（去掉 pos embed） | minGPT（Karpathy） |
| State encoder | Linear projection | CNN (3-layer Conv + Linear) |
| Action representation | 连续向量，Linear embed | 离散 token，nn.Embedding |
| Loss | MSE | Cross-entropy |
| Context length K | 20 | 30 |
| 网络规模 | 3 layers, 1 head, 128 dim | 6 layers, 8 heads, 128 dim |
| Positional encoding | Timestep embed (加法) | Global pos embed + Local pos embed |

Atari 版本使用了双层位置编码：

```python
# atari/mingpt/model_atari.py, line 133
self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))

# line 258
position_embeddings = torch.gather(all_global_pos_emb, 1, ...) + self.pos_emb[:, :token_embeddings.shape[1], :]
```

- `global_pos_emb`: 编码在 episode 中的绝对时间步
- `pos_emb`: 编码在 context window 内的相对位置

### 3.6 Return Conditioning 机制

Return conditioning 是 DT 最核心的设计。在推理时，通过设定不同的 target return 值，可以控制 agent 的行为：

```python
# gym/experiment.py, line 41-42
env_targets = [3600, 1800]  # evaluation conditioning targets (Hopper)
scale = 1000.               # normalization for rewards/returns
```

各环境的 target return 设定：

| 环境 | 高目标 | 低目标 | Scale |
|------|--------|--------|-------|
| Hopper | 3600 | 1800 | 1000 |
| HalfCheetah | 12000 | 6000 | 1000 |
| Walker2d | 5000 | 2500 | 1000 |
| Breakout (Atari) | 90 | - | - |
| Seaquest | 1150 | - | - |
| Qbert | 14000 | - | - |
| Pong | 20 | - | - |

---

## 4. Experiments: 主要实验结果

### 4.1 OpenAI Gym (D4RL Benchmark)

DT 在 D4RL 的三个连续控制环境（HalfCheetah, Hopper, Walker2d）上评估，使用三种数据集质量级别：

| 数据集类型 | 含义 |
|-----------|------|
| Medium | 使用训练到中途的 policy 收集 |
| Medium-Replay | Medium policy 训练过程中 replay buffer 的全部数据 |
| Medium-Expert | Medium + Expert 数据的混合 |

与 CQL（Conservative Q-Learning，当时 SOTA 的 offline RL 方法）对比：
- 在 **medium** 和 **medium-replay** 数据集上，DT 和 CQL 表现相当或更好
- 在 HalfCheetah-medium 上 DT 略优，在 Hopper-medium-replay 上 DT 显著优于 CQL
- 在 **medium-expert** 上 CQL 一般更优，因为 DT 缺乏 stitching 能力（见 Limitations）

### 4.2 Atari

在 Breakout、Qbert、Seaquest、Pong 四个游戏上评估：
- DT 使用 DQN replay buffer 中 top 1% 的数据训练
- 在 Breakout 和 Qbert 上，DT 超过 CQL
- 在 Pong 上 DT 达到满分（20）
- 特别是在只使用 1% 数据量的情况下，DT 的样本效率显著优于传统 offline RL

### 4.3 Delayed Reward 实验

代码中实现了 "delayed" 模式，将所有 reward 移到 trajectory 末尾：

```python
# gym/experiment.py, line 77-79
if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
    path['rewards'][-1] = path['rewards'].sum()
    path['rewards'][:-1] = 0.
```

这模拟了极端稀疏奖励的场景。由于 DT 使用 return-to-go 而非逐步的 temporal difference，它天然适合这种设置。实验表明 DT 在 delayed reward 设置下的性能降幅远小于 TD-based 方法。

### 4.4 关键训练超参数

```python
# gym/experiment.py, line 283-305
K = 20              # context length
batch_size = 64
embed_dim = 128
n_layer = 3
n_head = 1
dropout = 0.1
learning_rate = 1e-4
weight_decay = 1e-4
warmup_steps = 10000
max_iters = 10
num_steps_per_iter = 10000
```

总训练步数 = 10 * 10000 = 100K 步，规模非常小。

---

## 5. Related Work Analysis: 与传统 RL 方法的对比

### 5.1 方法论层面的对比

| 维度 | 传统 RL (TD-based) | Decision Transformer |
|------|-------------------|---------------------|
| 核心机制 | Bellman backup / Policy gradient | Autoregressive sequence prediction |
| 训练方式 | On-policy / Off-policy | 纯 supervised learning |
| Value function | 必需（Q/V function） | 不需要 |
| Exploration | 需要（epsilon-greedy, entropy bonus 等） | 不需要（offline 设定） |
| Credit assignment | 通过 TD 逐步传播 | Attention 直接建模长程依赖 |
| Stitching 能力 | 有（通过 Q-learning 组合不同 trajectory 的最优片段） | 弱（无显式 value-based stitching） |
| 实现复杂度 | 高（replay buffer, target network, double Q 等） | 低（标准 Transformer 训练） |

### 5.2 与 Behavior Cloning 的关系

DT 本质上是一种 **conditioned behavior cloning**: 给定 return-to-go 条件下的 imitation learning。代码中也实现了纯 BC baseline (MLPBCModel):

```python
# gym/decision_transformer/models/mlp_bc.py
class MLPBCModel(TrajectoryModel):
    # Simple MLP that predicts next action a from past states s.
```

关键区别：
- BC: 无条件克隆所有行为 --> 被平均行为拖累
- DT: 条件生成 --> 可以选择性地生成高回报行为
- DT 的 context window 能利用历史信息，而 MLP BC 只看当前 state

### 5.3 与 Trajectory Transformer 的关系

同期发表的 Trajectory Transformer (Janner et al., 2021) 也将 RL 建模为序列问题，但方法不同：
- Trajectory Transformer 将所有维度（state、action、reward 的每一维）都离散化为 token
- DT 保持连续表示，只预测 action
- DT 更轻量，Trajectory Transformer 更通用但更昂贵

### 5.4 与 Upside-Down RL 的关系

Schmidhuber (2019) 和 Srivastava et al. (2019) 提出的 Upside-Down RL 也使用 return-conditioned policy，但使用简单的 MLP/RNN 架构。DT 的贡献是将 Transformer 引入这一范式，利用 attention 机制处理长程依赖。

---

## 6. Limitations & Future Directions

### 6.1 论文明确提出的局限

1. **缺乏 Stitching 能力**: DT 无法将不同 suboptimal trajectory 的最优片段拼接在一起。如果 dataset 中没有包含完整的高回报 trajectory，DT 难以生成超越数据集最优水平的行为。CQL 等 value-based 方法通过 Q-learning 天然具备 stitching 能力。

2. **离线学习限制**: DT 目前只适用于 offline RL 设定，不具备 online fine-tuning 能力。

### 6.2 从代码中推断的局限

1. **Context length 有限**: Gym 环境使用 K=20，Atari 使用 K=30。对于需要更长历史的任务（如长 horizon 的机器人操作），可能不足。

2. **State normalization 依赖全局统计**: 代码中使用全 dataset 的 mean/std 做 state normalization，这假设了对数据分布的完整访问：
   ```python
   # gym/experiment.py, line 87
   state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
   ```

3. **Return scale 需要手动设定**: 每个环境的 return scale 和 target return 都需要手动指定，不具备自动适配能力。

4. **训练采样策略有偏**: 按 trajectory 长度加权采样，偏向长 trajectory：
   ```python
   # gym/experiment.py, line 116
   p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
   ```

5. **Atari 评估只跑 10 个 episode**: `get_returns` 方法中只评估 10 次取平均，统计显著性不足：
   ```python
   # atari/mingpt/trainer_atari.py, line 182
   for i in range(10):
   ```

### 6.3 后续发展方向

- **Online DT (Zheng et al., 2022)**: 加入 online fine-tuning 能力
- **Multi-Game DT (Lee et al., 2022)**: 跨游戏的通用 agent
- **Generalist Agent (Gato, Reed et al., 2022)**: 将 DT 思路推广到多模态、多任务

---

## 7. Paper vs Code Discrepancies

### 7.1 两套完全不同的代码实现

最显著的差异是 **Gym 和 Atari 使用了完全不同的 GPT 实现**：

| 实现细节 | Gym | Atari |
|----------|-----|-------|
| GPT 来源 | HuggingFace `transformers` 库修改版 | Karpathy 的 minGPT |
| 文件 | `trajectory_gpt2.py` | `mingpt/model_atari.py` |
| Attention 实现 | HuggingFace Conv1D-based | 标准 nn.Linear |
| 位置编码 | 完全移除 positional embedding | 保留 pos_emb + 新增 global_pos_emb |

论文中并未提及两个实验使用了不同的 backbone 实现。

### 7.2 未使用的 Prediction Head

代码中定义了 state 和 return prediction head，但 `SequenceTrainer` 完全忽略了它们：

```python
# gym/decision_transformer/training/seq_trainer.py, line 21-24
loss = self.loss_fn(
    None, action_preds, None,   # state_preds and reward_preds passed as None
    None, action_target, None,
)
```

论文可能尝试过多任务 loss（同时预测 state 和 return），但最终只保留了 action loss。

### 7.3 Gradient Clipping

Gym 的 `SequenceTrainer` 中使用了 gradient clipping（max_norm=0.25），但 `ActTrainer` (BC baseline) 中没有：

```python
# gym/decision_transformer/training/seq_trainer.py, line 28
torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
```

这个实现细节在论文中未提及。

### 7.4 LayerNorm 的使用

Gym 版本在 stacked input 后加了 LayerNorm：

```python
# gym/decision_transformer/models/decision_transformer.py, line 45, 78
self.embed_ln = nn.LayerNorm(hidden_size)
stacked_inputs = self.embed_ln(stacked_inputs)
```

Atari 版本使用的是 GPT 自带的 `self.ln_f`。这种架构差异在论文中未讨论。

### 7.5 Action Embedding 的 Tanh 激活

Gym 版本的 predict_action 使用了可选的 `Tanh` 激活（默认开启）来约束输出范围：

```python
# gym/decision_transformer/models/decision_transformer.py, line 49-51
self.predict_action = nn.Sequential(
    *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
)
```

Atari 版本的 action embedding 也使用了 Tanh：

```python
# atari/mingpt/model_atari.py, line 156
self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
```

### 7.6 数据加载方式差异

- Gym: 预先下载 D4RL 数据为 pickle，随机采样 trajectory 片段（online batch construction）
- Atari: 从 DQN replay buffer 加载，构建完整的 PyTorch Dataset，使用 DataLoader

### 7.7 Trainer 基类的冗余代码

`gym/decision_transformer/training/trainer.py` 中的 `train_step` 方法调用签名与 `SequenceTrainer` 完全不同，实际从未被调用：

```python
# trainer.py, line 62 -- this is dead code for DT
states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
```

---

## 8. Cross-Paper Comparison: 与 Diffusion Policy、DreamerV3 的对比

### 8.1 范式对比

| 维度 | Decision Transformer (2021) | Diffusion Policy (2023) | DreamerV3 (2023) |
|------|---------------------------|------------------------|-----------------|
| 核心思想 | RL as sequence modeling | Policy as denoising diffusion | RL via learned world model |
| 建模对象 | (R, s, a) 序列 | Action distribution | Latent dynamics model |
| 训练范式 | Supervised (offline) | Supervised (offline) | Online RL + world model |
| 动作生成 | Autoregressive GPT | Iterative denoising | Actor-critic in latent space |
| Value function | 不需要 | 不需要 | 需要 (learned V function) |
| World model | 不需要 | 不需要 | 核心组件 |
| 条件控制 | Return conditioning | Task embedding / image conditioning | Reward-driven |

### 8.2 对 Robotics 的适用性对比

| 能力 | Decision Transformer | Diffusion Policy | DreamerV3 |
|------|---------------------|-----------------|-----------|
| 多模态动作分布 | 弱（MSE loss 趋向 mode averaging） | 强（天然支持多模态） | 中（通过 latent space） |
| 高维连续控制 | 支持 | 强（擅长高维 action space） | 支持 |
| 视觉输入 | 有限（Atari 用 CNN） | 强（CNN/ViT encoder） | 强（image-based RL） |
| 长 horizon 规划 | 受 context length 限制 | 受 action horizon 限制 | 通过 imagination 支持长 horizon |
| Online 适应 | 不支持（纯 offline） | 不支持（纯 offline） | 天然支持（online learning） |
| 数据效率 | 中等 | 较高 | 高（world model 的优势） |
| 实时性 | 快（单次 forward pass） | 慢（多步 denoising） | 中等 |

### 8.3 DT 范式对 Robotics 的启示

**优势方面**：

1. **简单性**: DT 将复杂的 RL pipeline 简化为标准的 sequence-to-sequence 训练，降低了 robotics 研究者的使用门槛
2. **Scalability**: 基于 Transformer 的架构可以利用 NLP 社区的 scaling law 经验，支持更大的模型和数据集
3. **Multi-task potential**: 通过 return conditioning 可以在单一模型中编码多种行为策略

**局限方面**：

1. **缺乏多模态动作建模能力**: 机器人操作任务通常涉及多模态动作分布（如抓取物体可以从左边或右边），DT 的 MSE loss 会导致 mode averaging，这正是 Diffusion Policy 的优势所在
2. **无法在线学习**: 真实机器人部署需要 online adaptation，DT 的纯 offline 范式在这方面不如 DreamerV3
3. **Return conditioning 的局限**: 在真实机器人场景中，设定合适的 target return 需要对任务有先验知识，且 return 本身可能无法充分描述期望行为

**后续发展**:

DT 最大的贡献在于开创了 "序列模型做决策" 的范式。后续工作如 Gato (2022)、RT-1/RT-2 (2023)、Octo (2024) 都继承了这一思路，但用更大规模的数据和模型来弥补 DT 的局限。特别是 GR00T N1 (2025) 等 foundation model for robots 的工作，本质上是 DT 范式在大规模数据和多模态输入上的自然延伸。

### 8.4 技术路线演进

```
Decision Transformer (2021)
    |
    |-- 将 RL 建模为序列问题
    |-- 证明 supervised learning 可以做 RL
    |
    v
Diffusion Policy (2023)              DreamerV3 (2023)
    |                                     |
    |-- 用 diffusion 解决                  |-- 用 world model 解决
    |   多模态动作问题                       |   在线学习和规划问题
    |                                     |
    v                                     v
Robotics Foundation Models (2024-2025)
    |
    |-- RT-2, Octo, GR00T N1 等
    |-- 融合大规模预训练 + 序列决策 + 多模态输入
```

DT 在这条演进路线中的定位是 "proof of concept": 它证明了 Transformer 可以有效地用于决策任务，为后续更复杂的方法奠定了概念基础。真正进入 robotics 实用领域，还需要结合 Diffusion Policy 的多模态表达能力、DreamerV3 的 world model 与在线学习能力，以及大规模预训练数据。

---

## References (Key Code Paths)

| Component | File Path |
|-----------|-----------|
| DT model (Gym) | `decision-transformer/gym/decision_transformer/models/decision_transformer.py` |
| GPT2 backbone (modified) | `decision-transformer/gym/decision_transformer/models/trajectory_gpt2.py` |
| Training loop (Gym) | `decision-transformer/gym/experiment.py` |
| Sequence trainer | `decision-transformer/gym/decision_transformer/training/seq_trainer.py` |
| Evaluation | `decision-transformer/gym/decision_transformer/evaluation/evaluate_episodes.py` |
| BC baseline | `decision-transformer/gym/decision_transformer/models/mlp_bc.py` |
| DT model (Atari) | `decision-transformer/atari/mingpt/model_atari.py` |
| Atari training | `decision-transformer/atari/run_dt_atari.py` |
| Atari evaluation | `decision-transformer/atari/mingpt/trainer_atari.py` |
| Dataset creation | `decision-transformer/atari/create_dataset.py` |
| D4RL download | `decision-transformer/gym/data/download_d4rl_datasets.py` |
