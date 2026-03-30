# Flow Matching for Generative Modeling -- 阅读笔记

Lipman, Chen, Ben-Hamu, Nickel, Le (Meta AI / Weizmann), arXiv 2022 (v2 Feb 2023)

---

## 1. Core Problem

Flow Matching 解决的核心问题：**如何高效、稳定地训练 Continuous Normalizing Flows (CNFs)**。

CNF 是一类强大的生成模型，通过学习一个 time-dependent vector field (向量场) 来定义从噪声到数据的连续映射。此前训练 CNF 有两个根本困难：

| 困难 | 具体表现 |
|------|---------|
| 计算代价高 | Maximum likelihood training 需要前向/反向 ODE simulation，时间复杂度极高 |
| Simulation-free 方法不可行 | 已有 simulation-free 方法要么涉及不可计算的积分 (Rozen et al., 2021)，要么存在有偏梯度 (Ben-Hamu et al., 2022) |
| 被锁定在 diffusion paths 上 | Score matching 只能用于特定 diffusion process 派生的概率路径，无法探索更优的路径 |

Flow Matching 的突破：提出一个 **simulation-free、无偏的回归目标**，将 CNF 的训练从"解 ODE"简化为"回归 vector field"。更关键的是，这个框架不依赖 diffusion process，可以使用任意概率路径 -- 包括比 diffusion paths 更简洁高效的 Optimal Transport (OT) 路径。

直觉：不再问"这个 SDE/ODE 该怎么模拟"，而是直接问"从 noise 到 data 最好走什么路？走多快？" -- 然后训一个网络来拟合这条路上的速度。

---

## 2. Method Overview

### 2.1 基础框架：CNF

给定数据空间 R^d，定义：
- **Probability density path** p_t(x)：从 t=0 的简单先验 p_0 = N(0, I) 到 t=1 的数据分布 p_1 ≈ q(x)
- **Vector field** v_t(x)：通过 ODE d/dt phi_t(x) = v_t(phi_t(x)) 定义 flow phi_t
- 网络 v_t(x; theta) 学习这个 vector field

### 2.2 Flow Matching 目标

最朴素的 FM loss：

```
L_FM(theta) = E_{t~U[0,1], x~p_t} || v_t(x; theta) - u_t(x) ||^2
```

其中 u_t(x) 是能生成 p_t 的目标 vector field。问题是我们既不知道 p_t 也不知道 u_t 的闭式形式。

### 2.3 Conditional Flow Matching (CFM) -- 核心技巧

将不可计算的 marginal 问题分解为可计算的 conditional 问题：

**关键定理 (Theorem 2)**：CFM loss 和 FM loss 的梯度相同：

```
L_CFM(theta) = E_{t, x_1~q, x~p_t(x|x_1)} || v_t(x; theta) - u_t(x|x_1) ||^2
```

其中 conditional probability path p_t(x|x_1) = N(x; mu_t(x_1), sigma_t(x_1)^2 I) 是一个简单的 Gaussian path，给定 x_1 后其 vector field u_t(x|x_1) 有闭式解。

### 2.4 Gaussian Conditional Probability Paths

通用形式：
```
p_t(x|x_1) = N(x; mu_t(x_1), sigma_t(x_1)^2 I)
```
边界条件：mu_0 = 0, sigma_0 = 1 (标准正态), mu_1 = x_1, sigma_1 = sigma_min (集中在 x_1)

对应的 flow：psi_t(x) = sigma_t(x_1) * x + mu_t(x_1)

**Conditional vector field (Theorem 3)**：
```
u_t(x|x_1) = sigma'_t / sigma_t * (x - mu_t(x_1)) + mu'_t(x_1)
```

### 2.5 两种实例化

**Example I: Diffusion Paths (VP/VE)**

复用已有 diffusion process 的 noise schedule：
- VP: mu_t(x_1) = alpha_{1-t} * x_1, sigma_t(x_1) = sqrt(1 - alpha_{1-t}^2)
- VE: sigma_t(x_1) = sigma_{1-t}

trajectory 弯曲，vector field 方向随时间剧烈变化。

**Example II: Optimal Transport Paths (本文核心贡献)**

最简单直接的线性插值：
```
mu_t(x_1) = t * x_1,  sigma_t(x_1) = 1 - (1 - sigma_min) * t
```

对应 flow：psi_t(x) = (1 - (1-sigma_min)t) * x + t * x_1

CFM loss 简化为：
```
L_CFM(theta) = E_{t, x_1~q, x_0~N(0,I)} || v_t(psi_t(x_0); theta) - (x_1 - (1-sigma_min) * x_0) ||^2
```

注意：当 sigma_min -> 0 时，目标就是 x_1 - x_0，即"从噪声直接指向数据的方向"。

### 2.6 Training Pipeline

```
训练循环:
  1. 采样 x_1 ~ q(data), x_0 ~ N(0, I), t ~ U[0, 1]
  2. 计算 x_t = (1-t) * x_0 + t * x_1  (OT path)
  3. 计算 target velocity = x_1 - x_0
  4. loss = || v_theta(x_t, t) - (x_1 - x_0) ||^2
  5. 反向传播更新 theta

推理:
  1. 采样 x_0 ~ N(0, I)
  2. ODE 积分: dx/dt = v_theta(x, t), t: 0 -> 1
  3. 输出 x_1 = phi_1(x_0)
```

---

## 3. Key Designs

### 3.1 从 Conditional 到 Marginal 的等价性 (Theorem 1 + 2)

这是整篇论文最核心的理论贡献。

**直觉**：虽然我们不知道 marginal vector field u_t(x) 的闭式，但它实际上就是 conditional vector field u_t(x|x_1) 按 q(x_1) 加权平均的结果 (Eq. 8)。Theorem 1 证明这个加权平均的 vector field 确实生成了正确的 marginal probability path。Theorem 2 进一步证明优化 CFM loss 等价于优化 FM loss（梯度完全一致）。

**实际意义**：训练时只需要 per-sample 的 conditional target，不需要知道任何 marginal 量。这使得训练变成纯回归问题 -- 给定 (x_t, t)，回归 u_t(x|x_1)。

### 3.2 Optimal Transport 路径的直线性

**直觉**：Diffusion path 让粒子走弯路 -- noise 先被逐渐加到数据上（按 noise schedule 的复杂曲线），再沿复杂路径脱噪。OT path 让粒子走直线 -- 从 x_0 到 x_1 的最短路径就是线性插值。

具体对比：

| 性质 | Diffusion Path | OT Path |
|------|---------------|---------|
| Trajectory 形状 | 弯曲，可能"overshoot" | 直线，匀速 |
| Conditional VF 方向 | 随 t 剧烈变化 | 恒定方向 (x_1 - x_0) |
| 回归难度 | 高（网络需学习时变方向） | 低（方向恒定，只需学幅度） |
| 采样效率 | 需要多步 (弯路需精细离散化) | 少步即可 (直线用 Euler 就很准) |
| NFE (function evaluations) | 高 (264 for DDPM@ImageNet64) | 低 (138 for FM-OT@ImageNet64) |

**论文中的可视化 (Figure 2, 3)**：Diffusion path 的 conditional score function 方向随 t 旋转；OT path 的 conditional vector field 始终指向同一方向，仅幅度变化。这是 OT path 更容易学习的根本原因。

### 3.3 与 Score Matching / DDPM 的统一视角

Flow Matching 框架统一了 diffusion model 和 CNF 的训练：

- 当选择 VP diffusion path + noise prediction parameterization 时，CFM loss 等价于 DDPM 的 L_simple
- 但即使在相同的 diffusion path 上，Flow Matching 的 velocity regression 也比 score matching 更稳定
- 更重要的是，FM 解锁了 diffusion 之外的路径选择 -- 特别是 OT path

这不仅是理论上的统一，还带来了实践指导：**不需要为 diffusion process 设计复杂的 noise schedule，直接用 OT 线性插值就是最优选择之一**。

---

## 4. Experiments

### 4.1 主要结果 (Table 1)

**CIFAR-10 和 ImageNet 对比 (同一 U-Net 架构)**

| Method | CIFAR-10 NLL | CIFAR-10 FID | ImageNet-64 NLL | ImageNet-64 FID | ImageNet-64 NFE |
|--------|-------------|-------------|----------------|----------------|----------------|
| DDPM | 3.12 | 7.48 | 3.32 | 17.36 | 264 |
| Score Matching | 3.16 | 19.94 | 3.40 | 19.74 | 441 |
| ScoreFlow | 3.09 | 20.78 | 3.36 | 24.95 | 601 |
| FM w/ Diffusion | 3.10 | 8.06 | 3.33 | 16.88 | 187 |
| **FM w/ OT** | **2.99** | **6.35** | **3.31** | **14.45** | **138** |

FM-OT 在所有指标上全面领先：最优 NLL、最优 FID、最少 NFE。

**ImageNet 128x128**

FM-OT 取得 FID=20.9、NLL=2.90，超越当时所有无条件生成方法（BigGAN 25.3, PGMGAN 21.7）。

### 4.2 训练效率 (Figure 5)

FM-OT 的 FID 收敛速度显著快于所有 baseline。在 ImageNet-64 上：
- 50 epoch 时 FM-OT 已达到其他方法 200+ epoch 的 FID
- FM-OT 总训练步数仅 500k (batch 1536)，而 Dhariwal & Nichol 2021 需要 4.36M 步 (batch 256)

### 4.3 低步数采样 (Figure 7)

FM-OT 在低 NFE 时优势更明显：
- NFE=10 时 FM-OT 的 FID 约 15，diffusion 方法约 30-45
- FM-OT 达到与 diffusion 相同精度只需约 60% 的 NFE
- OT path 用最简单的 Euler solver 效果就很好，不需要高阶 solver

### 4.4 条件生成 (Table 2)

图像超分辨率 (64->256)：FM-OT FID=3.4, IS=200.8，与 SR3 (FID=5.2) 相比显著提升。

### 4.5 Ablation 要点

- FM w/ Diffusion vs Score Matching w/ Diffusion：同一 diffusion path 下，FM 目标函数更稳定，FID 更低
- FM-OT vs FM-Diffusion：OT path 在所有指标上优于 diffusion path
- 训练成本恒定：FM-OT 的每步采样成本不随训练进展变化，而 score matching 的采样成本会剧烈波动

---

## 5. Related Work Analysis

### 发展脉络

```
Score Matching (Song & Ermon 2019)
  |
  v
DDPM (Ho et al. 2020) ← Denoising Score Matching + reverse chain
  |
  v
Score SDE (Song et al. 2020b) ← 统一 diffusion 和 score-based
  |                |
  v                v
ScoreFlow (2021)   Probability Flow ODE (from SDE)
                     |
                     v
               Flow Matching (2022) ← 脱离 SDE 框架, 直接构造概率路径
                     |
              +------+------+
              |             |
        FM w/ Diffusion  FM w/ OT ← 本文核心贡献
```

并行工作：
- **Rectified Flow (Liu et al., 2022)**: 独立提出了类似的直线插值思想 ("flow straight and fast")
- **Albergo & Vanden-Eijnden (2022)**: 独立提出了类似的 conditional objectives
- **Action Matching (Neklyudov et al., 2023)**: 类似 CFM 但假设 u_t 是梯度场

### Flow Matching 的独特贡献

| 维度 | Diffusion Model (DDPM/Score SDE) | Flow Matching |
|------|----------------------------------|---------------|
| 理论基础 | Markov chain / SDE | Continuous Normalizing Flow |
| Forward process | 固定 stochastic process | 任意概率路径 (可设计) |
| 训练目标 | Denoising score matching / ELBO | Vector field regression |
| 路径选择自由度 | 被锁定在 diffusion process | 完全自由 (含 OT path) |
| Noise schedule 设计 | 关键超参数, 需精心调整 | 简单线性即可 (OT path) |
| Simulation-free | 是 (通过 score matching trick) | 是 (通过 CFM 等价性) |
| 采样 | SDE/ODE solver | ODE solver |

---

## 6. Limitations & Future Directions

### 论文明确提到

- 当前 OT path 是 per-sample conditional OT (每个 x_1 独立的直线)，不是真正的 global OT coupling -- marginal vector field 不保证是 OT 解
- 目前限于 isotropic Gaussian 先验，未探索非各向同性先验或更一般的 kernel
- 实验限于图像生成，未验证其他模态 (文本、音频、3D、动作等)

### 从代码推断的方向

1. **Riemannian Flow Matching**：代码库已实现 `GeodesicProbPath`、`RiemannianODESolver`、以及 Sphere/Torus manifold -- 论文中完全未提及，说明团队已在探索流形上的 flow matching
2. **Discrete Flow Matching**：代码库包含完整的 `MixtureDiscreteProbPath`、`MixtureDiscreteEulerSolver`、`MixturePathGeneralizedKL` loss -- 将 flow matching 扩展到离散空间 (文本生成)
3. **Schedule Transform**：代码中 `ScheduleTransformedModel` 允许训练后切换 scheduler (scale-time transformation)，无需重新训练 -- 论文未提及此实用功能
4. **多种 Scheduler**：代码支持 `VPScheduler`, `LinearVPScheduler`, `CosineScheduler`, `PolynomialConvexScheduler` 等，比论文展示的更丰富
5. **Skewed Timestep Sampling**：代码 `train_loop.py` 中实现了 EDM 风格的 log-normal 时间采样（偏向中间时刻），论文未讨论此训练技巧

---

## 7. Paper vs Code Discrepancies

### 7.1 代码库远超论文范围

论文只描述了连续欧氏空间上的 flow matching，但 Meta 开源的 `flow_matching` 库是一个完整框架：

| 功能 | 论文 | 代码 |
|------|------|------|
| 连续 FM (affine paths) | 有 | `AffineProbPath`, `CondOTProbPath` |
| 离散 FM | 无 | `MixtureDiscreteProbPath`, `MixtureDiscreteEulerSolver` |
| 流形 FM (Riemannian) | 无 | `GeodesicProbPath`, `RiemannianODESolver`, `Sphere`, `FlatTorus` |
| Schedule transform | 无 | `ScheduleTransformedModel` -- 训练后切换 scheduler |
| Generalized KL loss | 无 | `MixturePathGeneralizedKL` -- 离散 FM 的专用 loss |
| Text generation example | 无 | `examples/text/` -- 完整的 discrete FM 文本生成 pipeline |
| CFG (Classifier-Free Guidance) | 无 | `CFGScaledModel` 在 eval 中实现 |
| Representation conversion | 无 | `target_to_velocity`, `epsilon_to_velocity` 等 6 种互转函数 |

### 7.2 训练实现差异

**Skewed timestep sampling** (`train_loop.py` L26-33):
```python
P_mean, P_std = -1.2, 1.2
sigma = (torch.randn(n) * P_std + P_mean).exp()
time = 1 / (1 + sigma)  # log-normal 分布映射到 [0, 1]
```
这种采样策略来自 EDM (Karras et al. 2022)，偏向中间时刻而非均匀采样。论文中只使用 U[0,1] 均匀采样。

**离散 FM 的 Polynomial scheduler (n=3)**：代码中离散 flow matching 使用三次多项式 scheduler kappa_t = t^3 (慢启动快结束)，这比论文中讨论的线性 scheduler 更实用。

**混合精度训练**：代码使用 `torch.cuda.amp.autocast()` 进行 FP16 混合精度，论文未提及。

**EDM time discretization**：eval 采样时支持 EDM 风格的非均匀时间离散化 (`edm_time_discretization.py`)，论文未提及。

### 7.3 x_t 参数化的符号差异

论文中 x_t 的表示：
```
x_t = sigma_t * x_0 + mu_t(x_1)  // x_0 是 noise, x_1 是 data
    = (1 - (1-sigma_min)t) * x_0 + t * x_1  (OT path)
```

代码中 `AffineProbPath` 的统一表示：
```python
x_t = sigma_t * x_0 + alpha_t * x_1  // 更清晰
```

其中 `CondOTScheduler` 设置 alpha_t = t, sigma_t = 1 - t。代码将 x_0 定义为 source (noise)，x_1 定义为 target (data)，与论文一致，但参数命名更规范。

---

## 8. Cross-Paper Comparison: Flow Matching 如何替代 Diffusion 用于动作生成

### 8.1 从 DDPM 到 Flow Matching：理论演进

| 维度 | DDPM (Ho 2020) | Flow Matching (Lipman 2022) |
|------|---------------|---------------------------|
| 核心对象 | Forward/reverse Markov chain | Probability path + vector field |
| Forward process | q(x_t\|x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I) | x_t = (1-t)*x_0 + t*x_1 (OT) |
| 网络预测 | Noise epsilon | Velocity v_t |
| Loss | \|\| epsilon - epsilon_theta(x_t, t) \|\|^2 | \|\| v_theta(x_t, t) - (x_1 - x_0) \|\|^2 |
| 采样步数 | 1000 (原始), 20-100 (DDIM) | 10-50 (ODE solver) |
| 理论保证 | ELBO + score matching | CFM 梯度等价性 |
| Noise schedule | 关键超参数 (beta_min, beta_max) | 不需要 (OT path 线性) |
| Trajectory 性质 | 弯曲路径 | 直线路径 |

### 8.2 从 Diffusion Policy 到 pi_0：动作生成的范式转变

| 维度 | Diffusion Policy (Chi 2023) | pi_0 (Physical Intelligence 2024) |
|------|---------------------------|----------------------------------|
| 生成框架 | DDPM (denoising diffusion) | Flow Matching (vector field regression) |
| 动作表示 | Action chunk (T_p=16 步) | Action chunk (H=50 步 @50Hz) |
| 推理步数 | 100 步 DDPM 或 10 步 DDIM | 10 步 Euler 积分 |
| 推理延迟 | ~100ms (DDIM) | 27ms (flow matching 部分) |
| 训练目标 | MSE on noise epsilon | MSE on velocity (x_1 - epsilon) |
| Conditioning | FiLM / cross-attention | VLM + Action Expert (MoE attention) |
| 多模态建模 | 通过 iterative denoising | 通过 flow matching (更直接) |
| 模型规模 | ~100M (单任务) | 3.3B (跨任务/跨形态) |
| 预训练 | 无 | 10,000 小时多机器人数据 |

### 8.3 pi_0 为什么选择 Flow Matching 而非 Diffusion

这是本笔记的核心问题。从代码和论文分析，pi_0 选择 flow matching 有以下具体原因：

**1. 推理速度 -- 对实时控制至关重要**

| 方法 | 推理步数 | 原因 |
|------|---------|------|
| DDPM | 1000 步 | 离散 Markov chain 需要逐步去噪 |
| DDIM | 10-100 步 | 通过 deterministic 跳步加速，但 FID 有损失 |
| Flow Matching (OT) | 10 步 | 直线路径用 Euler 积分天然只需少步 |

pi_0 实际使用 10 步 Euler 积分，耗时仅 27ms。这对于 50Hz 的灵巧操控至关重要 -- 每 0.5 秒推理一次，生成 50 步 action chunk。如果用 DDPM 原始的 1000 步，推理延迟将不可接受。

**2. 训练简洁性 -- 无需 noise schedule 工程**

DDPM 需要精心设计 beta schedule (linear, cosine, polynomial 等)，不同任务可能需要不同 schedule。Flow Matching 使用 OT 线性插值，没有任何超参数需要调整：x_t = (1-t)*noise + t*data，目标 = data - noise。

pi_0 的训练 loss 极其简单：
```
L = || v_theta(tau*A + (1-tau)*epsilon, o) - (A - epsilon) ||^2
```

**3. 直线轨迹 -- 更适合动作空间**

这是最关键但最少被讨论的原因。Diffusion 的弯曲路径意味着：
- 中间状态 x_t 可能偏离 noise-to-data 的自然插值
- 少步采样时弯路上的近似误差会被放大
- 网络需要学习时变的速度方向

Flow Matching OT path 的直线轨迹意味着：
- x_t 始终是 noise 和 data 的凸组合，物理上有意义
- Euler 积分的近似误差小（直线上 Euler = exact）
- 网络只需学习恒定方向的速度

对于动作空间，这尤其重要：一个"半噪声半动作"的中间状态在直线路径下就是动作的粗糙版本，网络可以逐步 refine；在弯曲路径下，中间状态可能完全没有物理意义。

**4. 与 VLM 集成更自然**

pi_0 的 action expert 需要与 VLM backbone 通过 attention 交互。Flow matching 的 velocity prediction 输出与 attention 的连续输出空间天然兼容。而 DDPM 的 noise prediction 虽然数学上等价，但 velocity = (data - noise) 的语义 ("动作应该往哪走") 比 epsilon = noise ("数据中混了什么噪声") 对 action expert 更直觉。

**5. 时间采样灵活性**

pi_0 使用 Beta(1.5, 1) 分布偏向低 tau (高噪声) 采样。Flow matching 的连续时间框架天然支持任意采样分布，而 DDPM 的离散步数框架中修改采样分布相对不自然。

### 8.4 总结对比表

| 维度 | DDPM | Diffusion Policy | Flow Matching | pi_0 |
|------|------|-----------------|---------------|------|
| 年份 | 2020 | 2023 | 2022 | 2024 |
| 生成对象 | 图像 | 动作序列 | 图像 | 动作序列 |
| Forward 过程 | 固定 noise schedule | 同 DDPM | 线性插值 | 同 FM |
| 网络预测 | Noise | Noise | Velocity | Velocity |
| 训练 loss | MSE(epsilon) | MSE(epsilon) | MSE(velocity) | MSE(velocity) |
| 采样步数 | 1000 | 100 (DDPM) / 10 (DDIM) | 10-50 | 10 |
| Noise schedule 设计 | 关键超参数 | 继承 DDPM | 不需要 | 不需要 |
| Trajectory 性质 | 弯曲 | 弯曲 | 直线 | 直线 |
| 模态 | 图像 | 图像+动作 | 图像 | 语言+图像+动作 |
| 适合实时控制 | 否 | 勉强 | 是 | 是 |

**核心结论**：Flow Matching 对 DDPM 不是"换了个 loss"那么简单。OT 路径带来的直线轨迹从根本上改变了采样效率，使得 10 步推理成为可能。这正是 pi_0 能在 50Hz 实时控制场景中使用生成模型的基础。从 DDPM 到 Flow Matching，本质上是从"学去噪"到"学走路"的范式转变 -- 后者对机器人控制更自然、更高效。
