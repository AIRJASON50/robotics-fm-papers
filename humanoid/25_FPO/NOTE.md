# Flow Policy Gradients for Robot Control -- 研究笔记

**论文**: Flow Policy Gradients for Robot Control (FPO++)
**作者**: Brent Yi, Hongsuk Choi, et al. (Amazon FAR, UC Berkeley, Stanford, HKU, CMU)
**发布**: 2026-02-03, arXiv: 2602.02481
**项目页**: https://hongsukchoi.github.io/fpo-control/

---

## 1. 核心问题与动机

### 1.1 传统方法的根本限制

传统 policy gradient 方法 (PPO/SAC) 依赖于 **可微的 action likelihood**，这迫使策略输出限制在简单分布 (如 diagonal Gaussian) 上。Diagonal Gaussian 的固有问题:

- **各维度独立采样**: 无法表达动作维度之间的相关性 (如四足运动中左右腿的反相协调)
- **单模态**: 无法表达多模态动作分布 (如操控中同一状态可能有多条等效路径)
- **探索能力弱**: 探索模式单一，依赖各维度独立的高斯噪声

### 1.2 Flow/Diffusion Policy 的困境

Flow matching / diffusion policy 在模仿学习中取得了巨大成功 (Diffusion Policy, pi_0)，能表达任意复杂的动作分布。但将其用于 RL 训练面临一个核心矛盾:

- Policy gradient 需要 $\pi_\theta(a_t \mid o_t)$ 的 likelihood
- Flow policy 计算精确 likelihood 需要 divergence integration，**计算代价极高**
- 现有替代方案 (DPPO, ReinFlow) 把 denoising 过程建模为 MDP，引入额外的 credit assignment 层级

### 1.3 FPO 的核心思路

FPO (Flow Policy Optimization) 提出了一个关键洞察: **完全绕过 likelihood 计算**。利用 conditional flow matching (CFM) loss 的差值作为 log-likelihood ratio 的代理量。但原始 FPO 仅在简单 benchmark (DeepMind Control Suite) 上验证过，在真实机器人控制任务中不稳定。

本文的核心贡献是 **FPO++**: 通过两个简单但关键的改进，使 flow policy gradient 在真实机器人任务中可靠工作。

---

## 2. 方法论

### 2.1 背景: PPO 的 Likelihood Ratio

标准 PPO 目标函数:

$$\max_{\theta} \; \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \psi_{\text{PPO}}(\rho_\theta, \hat{A}_t) \right], \quad \rho_\theta = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$$

其中 clipped surrogate:

$$\psi_{\text{PPO}}(\rho_\theta, \hat{A}_t) = \min\left(\rho_\theta \hat{A}_t, \; \text{clip}(\rho_\theta, 1 \pm \varepsilon^{\text{clip}}) \hat{A}_t \right)$$

核心依赖: $\rho_\theta$ 需要可微的 $\pi_\theta(a_t \mid o_t)$。

### 2.2 FPO: 用 CFM Loss 代替 Likelihood Ratio

**关键公式 -- FPO ratio surrogate**:

$$\hat{\rho}_{\text{FPO}}(\theta) = \exp\left(\hat{\mathcal{L}}_{\text{CFM},\theta_{\text{old}}}(a_t; o_t) - \hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t)\right)$$

直觉: CFM loss 的差值近似 action log-likelihood 的差值。CFM loss 低 = 模型更好地预测从噪声到该 action 的 flow = 该 action 的概率更高。

**CFM loss 的计算**:

1. 对每个 action $a_t$，采样 $N_{\text{mc}}$ 个 noise-timestep 对 $(\epsilon_i, \tau_i)$
2. 线性插值得到 noised action: $a_t^{\tau_i} = \tau_i a_t + (1-\tau_i) \epsilon_i$
3. 目标 velocity field: $(\partial / \partial \tau_i) a_t^{\tau_i} = a_t - \epsilon_i$
4. 网络预测 velocity: $\hat{v}_\theta(a_t^{\tau_i}, \tau_i; o_t)$
5. 单样本 CFM loss:

$$\ell_\theta^{(i,t)} = \left\| \hat{v}_\theta(a_t^{\tau_i}, \tau_i; o_t) - (a_t - \epsilon_i) \right\|_2^2$$

6. 平均 CFM loss:

$$\hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t) = \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} \ell_\theta^{(i,t)}$$

### 2.3 FPO++ 的两个关键改进

#### (1) Per-sample Ratio (逐样本比率)

原始 FPO 对所有 MC 样本取平均后计算一个 ratio:

$$\hat{\rho}_{\text{FPO}}(\theta) = \exp\left(\frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} \left(\ell_{\theta_{\text{old}}}^{(i,t)} - \ell_\theta^{(i,t)}\right)\right)$$

问题: clipping 是在平均后的 ratio 上进行，要么所有样本都被 clip，要么都不被。

FPO++ 改为对每个 MC 样本计算独立的 ratio:

$$\hat{\rho}_{\text{FPO++}}^{(i)}(\theta) = \exp\left(\ell_{\theta_{\text{old}}}^{(i,t)} - \ell_\theta^{(i,t)}\right)$$

效果:
- On-policy 数据 (所有 ratio = 1) 时，梯度完全等价
- Off-policy 数据时，提供 **更细粒度的 trust region** -- 每个 $(\tau_i, \epsilon_i)$ 对独立 clip
- 增大有效 batch size，降低梯度方差

#### (2) Asymmetric Trust Region (ASPO)

根据 advantage 的正负使用不同的 clipping 策略:

$$\psi_{\text{ASPO}}(\rho_\theta, \hat{A}_t) = \begin{cases} \psi_{\text{PPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t \geq 0 \\ \psi_{\text{SPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t < 0 \end{cases}$$

其中 SPO (Simple Policy Optimization) 目标:

$$\psi_{\text{SPO}}(\rho_\theta, \hat{A}_t) = \rho_\theta \hat{A}_t - \frac{|\hat{A}_t|}{2\varepsilon^{\text{clip}}} (\rho_\theta - 1)^2$$

设计思路:
- **正 advantage** (好动作 -> 增大概率 -> 降低 CFM loss): 用标准 PPO clipping，超出 trust region 后梯度归零
- **负 advantage** (坏动作 -> 减小概率 -> 增大 CFM loss): 用 SPO，超出 trust region 后 **梯度反向拉回**，而不是归零

为什么不对称:
- 增大 CFM loss 意味着 (i) 激进降低 action likelihood 和 (ii) 增大变分后验的 KL divergence
- SPO 的二次惩罚项防止 entropy collapse: 不让 flow policy 过度远离 "坏动作"
- 保留足够的探索能力，尤其在 locomotion 等需要 emergent behavior 的任务中

ASPO 适用场景注意: 论文发现 ASPO 对从头训练 (locomotion, motion tracking) 非常关键，但对 fine-tuning (manipulation) 反而有害。Fine-tuning 场景下 policy 已有良好初始化，过多保持 entropy 会引入不必要的行为。

### 2.4 与 PPO / DPPO / ReinFlow 的本质区别

| 方法 | 策略表示 | Likelihood | 训练方式 |
|------|----------|-----------|---------|
| **PPO** | Diagonal Gaussian | 解析计算 | 直接 policy gradient |
| **DPPO** | Diffusion policy | noise likelihood (把 denoising 当 MDP) | 扩展的 credit assignment |
| **ReinFlow** | Flow policy | predicted noise likelihood | 类似 DPPO |
| **GenPO** | Invertible flow | normalizing flow likelihood | 需 unroll denoising |
| **FPO/FPO++** | Flow policy | **完全不需要** | CFM loss 差值代理 ratio |

FPO++ 的核心优势:
- 不依赖特定网络架构 (无需 invertible)
- 不依赖 denoising 过程的 unrolling (无梯度消失/爆炸风险)
- 不将 denoising 步骤建模为 MDP (不膨胀 credit assignment horizon)

### 2.5 Zero-sampling 推理策略

训练时: 从 $\epsilon \sim \mathcal{N}(0, I)$ 采样，通过多步 Euler integration 生成 action (如 64 步)。

测试 / 部署时: 使用 $\epsilon = \vec{0}$ (zero-sampling)，即从噪声分布的均值出发。

- 显著提升评估性能 (stochastic vs zero-sampling 差距可达 10% -> 70% 成功率)
- 允许大幅减少 integration 步数 (50 步 -> 5 步) 而几乎不影响性能
- 降低推理延迟，使实时部署可行

这与 behavior cloning 中类似的发现一致: "Much Ado About Noising" 论文也指出 zero-init 采样优于 stochastic 采样。

---

## 3. 算法架构

### 3.1 Flow Matching Policy 的网络设计

- **Actor**: 3-layer MLP (locomotion: 256 hidden units; motion tracking: 1024-512-256)
- **Critic**: 标准 value network (locomotion: 768 hidden units; motion tracking: 1024-512-256)
- **Actor 输入**: $(a_t^{\tau_i}, \tau_i; o_t)$ -- noised action, flow timestep, observation
- **Actor 输出**: velocity prediction $\hat{v}_\theta$，维度等于 action space 维度
- 对于 manipulation fine-tuning，使用 ViT encoder 处理图像观测

### 3.2 训练流程

```
1. Rollout Phase:
   - 对每个环境 step:
     a. 观测 o_t
     b. 采样 epsilon ~ N(0, I)
     c. 多步 Euler integration: a_0 = epsilon, a_{k+1} = a_k + (1/K) * v_theta(a_k, k/K; o_t)
     d. 最终 action a_t = a_K
     e. 执行 a_t, 获得 reward r_t
   - 收集 (o_t, a_t, r_t) trajectories

2. Advantage Estimation:
   - 使用 GAE 计算 advantage A_hat_t (与 PPO 完全相同)

3. Policy Update (FPO++):
   - 对 batch 中每个 (o_t, a_t, A_hat_t):
     a. 采样 N_mc 个 (epsilon_i, tau_i) 对
     b. 计算 noised actions: a_t^{tau_i} = tau_i * a_t + (1 - tau_i) * epsilon_i
     c. 计算逐样本 CFM loss: l_theta^{(i,t)} 和 l_theta_old^{(i,t)}
     d. 计算逐样本 ratio: rho_i = exp(l_old - l_new)
     e. 应用 ASPO clipping (正 advantage 用 PPO clip, 负 advantage 用 SPO)
     f. 平均所有样本的 clipped objective
   - 梯度更新 theta

4. 数值稳定性:
   - Clamp CFM loss before taking difference
   - Clamp difference before exponentiation
   - 标准梯度裁剪
```

### 3.3 推理流程

```
1. 输入观测 o_t
2. 初始化: a_0 = 0 (zero-sampling) 或 a_0 ~ N(0, I) (stochastic)
3. Euler integration (K 步, K = 5 for deployment, K = 50~64 for training):
   for k in range(K):
     v = v_theta(a_k, k/K; o_t)
     a_{k+1} = a_k + (1/K) * v
4. 输出: a_K 作为最终 action
```

关键参数:
- 训练时 flow steps: 50~64
- 部署时 flow steps: 5 (zero-sampling 允许大幅减少)
- Monte Carlo samples ($N_{\text{mc}}$): 8~32

### 3.4 Manipulation Fine-tuning 的 Action Chunk

对于操控任务，policy 输出 action chunk (长度 16)。CFM loss 在 chunk 内所有 timestep 上求和，计算 chunk-level ratio。这与 Diffusion Policy / pi_0 的 chunk 设计一致。

---

## 4. 实验设计与结果

### 4.1 任务覆盖

| 类别 | 任务 | 机器人 | 训练方式 |
|------|------|--------|---------|
| Locomotion | 速度跟踪 | Go2, Spot, H1, G1 | 从头训练 |
| Motion Tracking | 全身运动跟踪 | G1 (29 DoF) | 从头训练 + sim2real |
| Locomotion (sim2real) | 速度控制 | Booster T1 | 从头训练 + sim2real |
| Manipulation | Can, Square, Box Cleanup, Tray Lift, Threading | 单臂 + 双臂 | Fine-tuning (BC -> RL) |

### 4.2 关键结果

#### Locomotion: FPO++ vs FPO

- FPO 在 IsaacLab locomotion 环境中极不稳定 -- 即使大范围调参 (learning rate x clip x MC samples = 27 组合) 也无法可靠收敛
- FPO++ 在所有 4 个机器人上稳定训练，5 seeds 结果一致
- 两个改进 (per-sample ratio + ASPO) 都是关键的，缺少任一个都会导致性能下降和方差增大

#### Sim-to-Real: 首次验证

- **Booster T1 locomotion**: 直接部署 flow policy，实现稳定步态和速度跟踪
- **Unitree G1 motion tracking**: 跟踪 LAFAN 数据集中的 6 个运动序列 (舞蹈、行走、奔跑、格斗、跳跃)，每段约 2.5 分钟
- 训练 50 步 flow integration，部署时 zero-sampling + 5 步，延迟可控
- 这是首次: (i) 不依赖 expert distillation 的 flow policy humanoid sim2real; (ii) 不使用 explicit likelihood 的 policy gradient sim2real

#### Manipulation Fine-tuning

- FPO++ 在 5/5 个任务上收敛最快、成功率最高
- FPO (vanilla) 在简单任务上也能工作 (BC 初始化提供正则化)
- DPPO 对 base policy 质量敏感: 如果 stochastic sampling 成功率太低 (~10%)，fine-tuning 失败
- FPO++ 对 base policy 质量更鲁棒

#### FPO++ vs Gaussian PPO

- 在 locomotion 任务中，FPO++ 在几乎所有并行环境数 (256~4096) 下收敛到更高 return
- 更重要的是学到了 **更好的步态**: 同样的 reward 下，FPO++ 学到 trot (对角步态)，Gaussian PPO 倾向 pronk (四腿同步)
- 原因: flow policy 能表达动作维度间的相关性 (cross-correlation heatmap 显示左右腿关节负相关)
- 但 wall-clock time 更长: G1 locomotion 相同 return 需多 ~20% 时间

### 4.3 关键消融结果

- **Per-sample ratio**: 在所有任务 (locomotion, motion tracking, manipulation) 上一致有益
- **ASPO**: 对从头训练关键，但对 fine-tuning 有害
- **Zero-sampling**: 对评估和 sim2real 至关重要，成功率提升 10% -> 70%
- **Flow steps 数量**: 训练时 64 步，过少 (8, 16) 降低稳定性；部署时 zero-sampling 下可降至 5 步
- **CFM loss clamping**: 对数值稳定性必要 (exp of squared difference 容易溢出)

---

## 5. 对灵巧手操控 (Dexterous Manipulation) 的启示

### 5.1 直接相关性

论文的 manipulation fine-tuning 实验包含了 DexMimicGen 的灵巧手任务 (Threading, Tray Lift)，与我们的 WujiHand 项目直接相关。关键观察:

1. **高维动作空间的探索优势**: 灵巧手通常有 20+ DoF，diagonal Gaussian 的独立采样在高维空间效率极低。Flow policy 能表达手指间的协调关系 (如多指协调抓取中的拇指-食指对应)，这可能对 in-hand manipulation 特别有价值。

2. **动作相关性的表达**: 论文中 H1 locomotion 的 cross-correlation 分析表明 flow policy 自动学到了维度间相关性。对灵巧手而言，手指间的协调 (如 power grasp vs precision grasp 的模式切换) 本质上就是多维度的强相关行为。

3. **多模态策略的潜力**: 方块重定向 (cube reorientation) 任务中，同一目标姿态可能有多条等效的操控路径 (不同手指序列)。Gaussian policy 被迫选择一种模式的平均，而 flow policy 能保留多模态分布。

### 5.2 与当前 WujiHand 框架的对比

| 方面 | 当前方案 (Gaussian PPO) | FPO++ 方案 |
|------|----------------------|-----------|
| 动作分布 | Diagonal Gaussian + tanh squashing | Flow matching (任意分布) |
| 推理成本 | 1 次前向传播 | K 次前向传播 (K=5~64) |
| 训练时间 | 基线 | ~1.2x~3x (论文数据) |
| 手指协调 | 各维度独立探索 | 可学习维度间相关性 |
| Fine-tuning | 不适用 (从头训练) | 天然支持 BC -> RL |

### 5.3 潜在应用场景

1. **Motion tracking (bh_motion_track)**: 当前任务已用 717D/544D 非对称 AC。Flow policy 的多模态表达可能有助于处理复杂的轨迹跟踪，但训练时间增加是需要权衡的因素。

2. **Sim2Real 微调**: 如果未来从人类示教数据训练 flow policy (类似 pi_0)，然后用 FPO++ 在仿真中 fine-tune，可能是一条有竞争力的路径。

3. **探索效率**: 对于需要发现复杂操控策略的任务 (如 rubik cube solving)，flow policy 的分布表达能力可能带来探索优势。

### 5.4 需要注意的问题

- **推理延迟**: 多步 Euler integration 增加推理时间。对于 20 DOF 手部控制频率 (通常 50Hz)，5 步 flow integration 是否可行需要测试。
- **训练效率**: 3x wall-clock time 对大规模超参搜索不友好。
- **实现复杂度**: 需要修改 Brax PPO pipeline，增加 flow policy 的 rollout (多步采样) 和 FPO++ 的 policy update (per-sample ratio + ASPO)。
- **与 tanh_normal 的对比**: 当前 WujiHand 框架使用 `tanh_normal` distribution，已经能表达有界动作空间。FPO++ 的优势主要在于维度间相关性和多模态性。

---

## 6. 代码库现状与可用性

### 6.1 代码发布状态 (截至 2026-03-01)

GitHub 仓库: https://github.com/amazon-far-lab/fpo-control (推测)

| 计划 | 状态 | 日期 |
|------|------|------|
| 论文发布 | 已完成 | 2026-02-03 |
| Manipulation finetuning 代码 | 未发布 | 计划 2026-02-16 |
| IsaacLab locomotion & motion tracking 代码 | 未发布 | 计划 2026-03-01 |

当前仓库 **仅包含 README、LICENSE (Apache-2.0) 和 NOTICE 文件**，尚无任何可执行代码。

### 6.2 集成难度评估

如果要将 FPO++ 集成到 WujiHand 的 Brax PPO pipeline 中，需要:

1. **Flow policy 网络**: 将 actor 从 Gaussian MLP 改为 flow matching MLP (输入增加 $\tau$，输出改为 velocity prediction)
2. **Rollout 修改**: action 采样从单次前向传播改为多步 Euler integration
3. **Policy update 修改**: ratio 计算从 Gaussian log-prob 改为 CFM loss difference，加入 per-sample ratio 和 ASPO
4. **推理修改**: 支持 zero-sampling 和可调 flow steps

核心修改集中在 Brax 的 PPO 训练循环和网络定义中，环境侧 (env.py, rewards.py) 不需要改动。

---

## 7. 局限性与未来方向

### 7.1 论文承认的局限性

1. **Wall-clock time**: FPO++ 训练比 Gaussian PPO 慢 1.2x~3x。多步 flow integration 是推理和训练的瓶颈。
2. **Motion tracking 性能差距**: 与精心调参的 Gaussian PPO + entropy regularization + KL-adaptive LR 比较，FPO++ 的 return 稍低 (虽然 episode length 稍长)。
3. **缺乏 entropy regularization**: FPO++ 目前没有直接的 entropy 正则化机制 (尝试了 KL-based 和 KNN-based entropy 估计，但改进有限)。
4. **ASPO 不通用**: 对 fine-tuning 场景有害，需要根据任务类型选择是否启用。

### 7.2 未来方向

1. **Few-step distillation**: 通过蒸馏减少 flow integration 步数 (progressive distillation, rectified flow)。
2. **Entropy regularization for flow policies**: 开发适用于 flow policy 的 entropy 正则化方法。
3. **KL-adaptive learning rate**: 为 flow policy 设计基于 KL 散度的自适应学习率。
4. **更复杂的任务**: 论文未测试真正的 in-hand manipulation from scratch (如 cube reorientation)。
5. **与 VLA 模型结合**: 大规模 vision-language-action 模型 (pi_0, pi_0.5) 使用 flow policy，FPO++ 可作为其 RL fine-tuning 方法。

### 7.3 个人观察

- FPO++ 最有说服力的场景是 **BC pre-trained flow policy 的 RL fine-tuning**，因为这是其他方法 (Gaussian PPO) 根本无法处理的场景。
- 对于从头训练的任务，FPO++ 的优势主要体现在 **步态质量** (更自然的 trot vs pronk) 而非性能数值，这与更具表达力的分布有关。
- 论文的 sim2real 验证 (T1 locomotion + G1 motion tracking) 是一个重要的 existence proof，但尚未在灵巧手操控上验证。
- per-sample ratio 是一个通用的、几乎无代价的改进；ASPO 则需要根据任务特性选择。

---

## 8. 关键公式汇总

### 标准 PPO objective

$$\psi_{\text{PPO}}(\rho_\theta, \hat{A}_t) = \min\left(\rho_\theta \hat{A}_t, \; \text{clip}(\rho_\theta, 1 \pm \varepsilon^{\text{clip}}) \hat{A}_t\right)$$

### FPO ratio surrogate (核心创新)

$$\hat{\rho}_{\text{FPO}}(\theta) = \exp\left(\hat{\mathcal{L}}_{\text{CFM},\theta_{\text{old}}}(a_t; o_t) - \hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t)\right)$$

### CFM loss (逐样本)

$$\ell_\theta^{(i,t)} = \left\| \hat{v}_\theta(a_t^{\tau_i}, \tau_i; o_t) - (a_t - \epsilon_i) \right\|_2^2$$

### Per-sample ratio (FPO++ 改进 1)

$$\hat{\rho}_{\text{FPO++}}^{(i)}(\theta) = \exp\left(\ell_{\theta_{\text{old}}}^{(i,t)} - \ell_\theta^{(i,t)}\right)$$

### SPO objective

$$\psi_{\text{SPO}}(\rho_\theta, \hat{A}_t) = \rho_\theta \hat{A}_t - \frac{|\hat{A}_t|}{2\varepsilon^{\text{clip}}} (\rho_\theta - 1)^2$$

### ASPO objective (FPO++ 改进 2)

$$\psi_{\text{ASPO}}(\rho_\theta, \hat{A}_t) = \begin{cases} \psi_{\text{PPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t \geq 0 \\ \psi_{\text{SPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t < 0 \end{cases}$$

### Flow matching 线性插值

$$a_t^{\tau_i} = \tau_i a_t + (1 - \tau_i) \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, I), \; \tau_i \in [0, 1]$$
