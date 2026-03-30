# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion -- Analysis Notes

> Chi C, Xu Z, Feng S, et al. RSS 2023 (extended journal version 2024)
> Code: https://github.com/real-stanford/diffusion_policy

---

## 1. Core Problem

机器人 visuomotor policy learning 中存在三个核心难题:

1. **Multi-modal action distribution**: 人类演示数据天然包含多模态性 -- 同一观测下可能有多种合理动作。传统显式策略 (explicit policy) 使用 GMM 或离散化动作空间来建模，但 GMM 需要预设模态数，离散化在高维空间中 bin 数量指数爆炸。隐式策略 (implicit policy, 如 IBC) 理论上能建模任意分布，但训练不稳定。

2. **Temporal action consistency**: 单步策略在多模态场景下容易产生"抖动" -- 连续帧在不同模态间切换，导致执行失败。

3. **High-dimensional action spaces**: 序列化动作预测需要对高维输出空间建模，传统方法 (IBC 的负采样、GMM 的模态数指定) 在高维下效率急剧下降。

**核心洞察**: Diffusion model 在图像生成领域已经证明了在高维空间建模复杂分布的能力。本文将其迁移到机器人控制领域 -- 把 action sequence 视为"需要去噪的信号"，通过 DDPM 的去噪过程生成动作序列，同时以视觉观测作为条件输入。

---

## 2. Method Overview

### 2.1 Pipeline Architecture

```
Observation O_t (image + proprioception)
    |
    v
Visual Encoder (ResNet-18 w/ SpatialSoftmax + GroupNorm)
    |
    v
Observation Feature (global conditioning or inpainting)
    |
    v
Noise Prediction Network epsilon_theta (CNN-based 1D UNet or Transformer)
    |  <- Gaussian noise A^K_t (initial)
    |  <- Denoising iteration k
    v
Action Sequence A^0_t (T_p steps predicted, T_a steps executed)
```

### 2.2 Key Formulas

**Forward diffusion (training)**:

Training loss: `L = MSE(epsilon^k, epsilon_theta(O_t, A^0_t + epsilon^k, k))`

**Reverse diffusion (inference)**:

`A^{k-1}_t = alpha * (A^k_t - gamma * epsilon_theta(O_t, A^k_t, k) + N(0, sigma^2 * I))`

**Training stability insight** -- Diffusion Policy 学习的是 score function 的梯度:

`nabla_a log p(a|o) = -nabla_a E_theta(a,o) - nabla_a log Z(o,theta)`

其中 `nabla_a log Z(o,theta) = 0`，因此无需估计归一化常数 Z，避免了 IBC 中负采样导致的训练不稳定。

### 2.3 Horizon Design

- **Observation horizon** `T_o`: 输入的观测帧数 (default=2)
- **Prediction horizon** `T_p`: 模型预测的动作序列长度 (default=16)
- **Action horizon** `T_a`: 实际执行的动作步数 (default=8)，执行完后重新规划

`T_a < T_p` 是 receding horizon control 的核心设计，在 long-horizon planning 和短期 responsiveness 间取得平衡。

---

## 3. Key Designs

### 3.1 Conditional Denoising -- Observation as Global Conditioning

**技术选择**: 将 observation 作为 conditional input 而非 joint distribution 的一部分:

- 论文选择建模 `p(A_t | O_t)` 而非 `p(A_t, O_t)` (Diffuser/Janner et al. 2022 的做法)
- 视觉特征只需计算一次，不随 K 次去噪迭代重复计算
- 这使得 end-to-end 训练 vision encoder 成为可能

**代码实现**: 在 CNN backbone 中，observation feature 通过 FiLM (Feature-wise Linear Modulation) 注入每个 Conv1d residual block; 在 Transformer backbone 中，observation embedding 通过 cross-attention 传入 decoder。

### 3.2 Position Control + Action Sequence Prediction 的协同效应

这是 Diffusion Policy 最 counter-intuitive 的发现之一:

- **Position control vs velocity control**: 大多数 BC 方法使用 velocity control，因为 position control 的多模态性更严重。但 Diffusion Policy 恰好擅长建模多模态分布，反而能利用 position control 的优势 -- 更少的 compounding error。
- **Action sequence prediction**: DDPM 对高维输出空间的 scaling 能力使得预测 action sequence 成为可能，这天然保证了 temporal consistency -- 整个序列在同一模态内。

### 3.3 Dual Backbone Architecture

论文提出两种 noise prediction network:

| 特性 | CNN (1D Temporal UNet) | Transformer |
|------|----------------------|-------------|
| Conditioning | FiLM modulation (per-channel scale+bias) | Cross-attention |
| Inductive bias | Temporal smoothness (low-frequency bias) | No temporal smoothing |
| Hyperparameter sensitivity | Low | High |
| Best for | Most tasks (smooth trajectories) | High-frequency action changes, velocity control |
| Attention mask | N/A | Causal (each token attends to self + previous) |

**推荐策略**: CNN-based 作为 first try，如果任务需要 high-rate action changes 再切换 Transformer。

---

## 4. Experiments

### 4.1 Main Results

在 4 个 benchmark 的 15 个任务上系统评估，平均提升 46.9%:

| Benchmark | Tasks | Key Result |
|-----------|-------|------------|
| RoboMimic (5 tasks x 2 data types) | Lift, Can, Square, Transport, Tool Hang | State & image 全面领先; Square (PH) 最高 98% |
| Push-T (IBC) | Planar pushing | Coverage 0.75 vs IBC 0.42 (state), 0.72 vs 0.29 (image) |
| Block Push (BET) | Multi-modal sequential pushing | p2 metric: 0.63 vs BET 0.31 (+32%) |
| Franka Kitchen | Multi-stage manipulation | p4 metric: 0.47 vs LSTM-GMM 0.15 (+213%) |

### 4.2 Ablation Key Findings

- **Action horizon**: `T_a=8` 对大多数任务最优，太短则抖动，太长则反应迟钝
- **Latency robustness**: Position control 在 4 步延迟内维持峰值性能
- **Vision encoder**: Fine-tuning pretrained CLIP ViT-B/16 达到 98% (Square)，但 scratch ResNet-18 已经足够好
- **Noise schedule**: Squared cosine schedule (iDDPM) 效果最佳

### 4.3 Real-world Results

| Task | Success Rate | vs Baseline |
|------|-------------|-------------|
| Push-T (UR5) | 95% (IoU 0.80) | LSTM-GMM 20%, IBC 0% |
| Mug Flipping (6DoF) | 90% | LSTM-GMM 0% |
| Sauce Pouring | Coverage 0.74 (human 0.79) | LSTM-GMM fails |
| Sauce Spreading | Coverage 0.77 (human 0.79) | LSTM-GMM fails |
| Bimanual Egg Beater | 55% | N/A |
| Bimanual Mat Unrolling | 75% | N/A |
| Bimanual Shirt Folding | 75% | N/A |

特别值得注意: 所有 real-world 任务使用相同超参数，无需 per-task tuning。

---

## 5. Related Work Analysis

### 5.1 Policy Representation 发展脉络

```
Explicit Policy (direct regression)
    |-- 缺陷: 无法建模多模态
    v
GMM/MDN (LSTM-GMM / BC-RNN)
    |-- 缺陷: 需预设模态数，mode collapse
    v
Discretization (BET clustering + offset)
    |-- 缺陷: bin 数指数爆炸
    v
Implicit Policy (IBC / EBM)
    |-- 缺陷: 训练不稳定 (negative sampling)
    v
Diffusion Policy (score function, no normalization constant)
    -- 优势: 稳定训练 + 多模态 + 高维 scaling
```

### 5.2 Diffusion in Decision Making

| Work | Approach | Key Difference from DP |
|------|----------|----------------------|
| Diffuser (Janner 2022) | Joint p(A,O) + reward-guided planning | DP uses conditional p(A|O), no reward needed |
| Decision Diffuser (Ajay 2022) | Classifier-free guidance for goal conditioning | DP focuses on BC, no RL component |
| Wang et al. 2022 | Diffusion for policy regularization in RL | DP uses diffusion as the policy itself |
| Pearce/Reuss 2023 | Concurrent; focus on sampling strategies | DP focuses on real-world deployment |

**Diffusion Policy 的独特贡献**: 第一个系统性地将 diffusion model 作为 visuomotor BC policy 在真实机器人上验证，并提出了 receding horizon control、visual conditioning、dual backbone 等关键工程设计。

---

## 6. Limitations & Future Directions

### 6.1 Author-stated Limitations

1. **Behavior cloning 固有限制**: 依赖高质量演示数据，不能利用次优或负样本
2. **Inference latency**: 比 LSTM-GMM 等方法计算量大 (需 K 步去噪)。DDIM 缓解但未根除
3. **High-rate control**: 对需要极高频控制的任务 (如力控) 可能不够快

### 6.2 From Code Inferred

4. **Memory footprint**: EMA model 需要维护模型的完整副本，内存占用翻倍 (`diffusion_policy/model/diffusion/ema_model.py`)
5. **Normalizer 依赖**: 训练前需遍历整个数据集计算 min/max 或 mean/std 统计量 (`diffusion_policy/model/common/normalizer.py`)，对增量学习不友好
6. **robomimic 强耦合**: Vision encoder 通过 robomimic 的 algo_factory 构建，与特定库版本绑定 (见 policy 文件中的 `get_robomimic_config`)

### 6.3 Future Directions

- **Consistency models / Distillation**: 将 K 步推理压缩为 1-2 步
- **RL fine-tuning**: 在 BC 预训练基础上用 RL 继续优化 (DDPO-style)
- **Foundation model integration**: 与大型 vision-language model 结合做 task conditioning
- **Adaptive horizon**: 动态调整 `T_a` 而非固定值

---

## 7. Paper vs Code Discrepancies

这是最关键的部分 -- 代码中有多处与论文描述不同或论文未提及的实现细节。

### 7.1 已知 Bug 保留

**ConditionalUnet1D local_cond 注入位置错误**:

文件: `diffusion_policy/model/diffusion/conditional_unet1d.py`, line 229-234

```python
# The correct condition should be:
# if idx == (len(self.up_modules)-1) and len(h_local) > 0:
# However this change will break compatibility with published checkpoints.
# Therefore it is left as a comment.
if idx == len(self.up_modules) and len(h_local) > 0:
    x = x + h_local[1]
```

代码明确注释说 `local_cond` 在 up-path 的注入条件永远不会为 True (因为 `idx` 最大值是 `len(self.up_modules)-1`)。这意味着 `local_cond` 的 up-path 分支实际上是 dead code，但为了兼容已发布的 checkpoint 而保留。论文中完全没有提及此 bug。

### 7.2 Vision Encoder 实现方式

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Vision encoder | ResNet-18 + SpatialSoftmax + GroupNorm | **通过 robomimic 的 algo_factory 构建**，用 BC-RNN config 初始化后提取 encoder (见 `policy/diffusion_unet_hybrid_image_policy.py` line 68-102) |
| GroupNorm 替换 | 直接替换 BatchNorm | 通过 `replace_submodules` 遍历模块树动态替换 (line 104-112) |
| 多相机支持 | 未详细讨论 | `MultiImageObsEncoder` 支持多相机独立编码后 concat (`model/vision/multi_image_obs_encoder.py`) |
| CropRandomizer | 未提及 | 训练时 random crop，eval 时 fixed crop (eval_fixed_crop=True) |

这种通过 robomimic 间接构建 vision encoder 的做法论文完全没有提到，是一个显著的工程实现差异。

### 7.3 Noise Scheduler 实际使用

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Scheduler | 自行描述 DDPM denoising 公式 | 直接使用 HuggingFace `diffusers` 库的 `DDPMScheduler` / `DDIMScheduler` |
| Noise schedule | Squared cosine (iDDPM) | `beta_schedule: squaredcos_cap_v2` |
| Variance type | 未提及 | `variance_type: fixed_small` (DDPM config), Transformer config 注释: "Yilun's paper uses fixed_small_log instead, but easy to cause NaN" |
| DDIM inference steps | "100 training, 10 inference" | 默认 config 中 `num_inference_steps: 100`，可独立配置 |

### 7.4 Inpainting-based Conditioning 作为备选路径

论文主要描述 global conditioning 方式。代码中实现了两种 conditioning 路径:

1. **obs_as_global_cond=True** (default, recommended): Observation feature flatten 后作为 global condition 注入
2. **obs_as_global_cond=False**: 使用 inpainting 方式 -- observation 嵌入 action-obs trajectory 的前几步，通过 mask 保持不变

文件: `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`, line 237-255

```python
if self.obs_as_global_cond:
    # condition through global feature
    global_cond = nobs_features.reshape(B, -1)
    cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
else:
    # condition through impainting
    cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
    cond_data[:,:To,Da:] = nobs_features
    cond_mask[:,:To,Da:] = True
```

### 7.5 Transformer 的额外设计细节

论文只简要提到 causal attention，代码中有多个未在论文中讨论的设计:

- **Memory mask**: Transformer decoder 中对 cross-attention 也施加了 causal-style mask (`memory_mask`)，使得每个 action token 只能 attend 到时间上之前的 observation token (`transformer_for_diffusion.py` line 123-134)
- **Encoder-only mode**: 支持 BERT-style 架构 (time token prepended, encoder-only)，论文未提及
- **n_cond_layers**: 当 `n_cond_layers > 0` 时，condition encoder 使用 TransformerEncoder 而非 MLP; 当 `n_cond_layers = 0` 时使用简单的 2-layer MLP

### 7.6 pred_action_steps_only

Transformer policy 独有的 `pred_action_steps_only` 参数 (line 45, `diffusion_transformer_hybrid_image_policy.py`):

- 当为 True 时，只预测 `n_action_steps` 步的动作而非整个 horizon
- 这改变了 trajectory shape 和 loss mask 的行为
- 论文未讨论此选项

### 7.7 EMA Warmup 策略

论文只提到使用 EMA。代码中实现了 @crowsonkb 的 EMA warmup 策略 (`ema_model.py`):

- Decay 随训练步数渐进增长: `decay = 1 - (1 + step / inv_gamma)^(-power)`
- 默认参数 `power=0.75, inv_gamma=1.0`: 在 ~10K 步达到 0.999 decay
- 跳过 BatchNorm 层的 EMA 更新 (直接 copy running stats)

### 7.8 Data Normalization

论文未讨论数据预处理，代码中有完整的 `LinearNormalizer` 系统:

- 支持 `limits` mode: 缩放到 [-1, 1] 范围
- 支持 `gaussian` mode: 零均值单位方差标准化
- Action 和 observation 分别归一化，inference 时 unnormalize action output

文件: `diffusion_policy/model/common/normalizer.py`

### 7.9 Training 工程细节

代码中的训练循环 (`workspace/train_diffusion_unet_hybrid_workspace.py`) 包含论文未提及的:

- **Gradient accumulation**: 支持 `gradient_accumulate_every` 参数
- **Top-K checkpoint**: 按 `test_mean_score` 保留最佳 K 个 checkpoint
- **Cosine LR schedule + warmup**: 使用 diffusers 库的 scheduler，500 步 warmup

---

## 8. Cross-Paper Comparison

### 8.1 Diffusion Policy vs Decision Transformer

| 维度 | Diffusion Policy | Decision Transformer |
|------|-----------------|---------------------|
| **Paradigm** | Generative model (DDPM) | Autoregressive sequence model (GPT) |
| **Input** | Observation `O_t` | Return-to-go + state + action sequence |
| **Output** | Action sequence `A_t` (parallel) | Next action (autoregressive) |
| **Multi-modality** | Naturally handled by diffusion sampling | Controlled by return-to-go conditioning |
| **Temporal consistency** | Sequence-level prediction guarantees consistency | Autoregressive, no explicit consistency mechanism |
| **Reward/Return** | Not needed (pure BC) | Required (return-conditioned) |
| **Inference latency** | K denoising steps | Single forward pass per action |
| **Training stability** | Very stable (MSE on noise) | Stable (standard CE/MSE loss) |
| **Real-world deployment** | Extensive validation (15 tasks, real robots) | Primarily simulated benchmarks |

**核心区别**: Decision Transformer 将 RL 转化为 sequence modeling 问题，需要 return 信号; Diffusion Policy 将 BC 转化为 generative modeling 问题，不需要 reward 但依赖高质量演示。两者可以互补 -- DT 的 return conditioning 思想可以用 classifier-free guidance 引入 Diffusion Policy。

### 8.2 Diffusion Policy vs DreamerV3

| 维度 | Diffusion Policy | DreamerV3 |
|------|-----------------|-----------|
| **Core idea** | Diffusion as policy representation | World model for imagination-based planning |
| **Learning paradigm** | Imitation learning (BC) | Model-based RL |
| **World model** | None (implicit in action prediction) | Explicit RSSM (learned dynamics) |
| **Data requirement** | Expert demonstrations | Any interaction data |
| **Generalization** | Within demonstration distribution | Can extrapolate through world model |
| **Multi-modality** | Core strength | Not a primary focus |
| **Compute** | Moderate (K denoising steps) | Heavy (world model + actor-critic + imagination) |
| **Task scope** | Manipulation-focused | Diverse domains (Atari, DMC, Minecraft, etc.) |

**关键洞察**: DreamerV3 通过学习 world model 实现了跨 domain 泛化，但不擅长建模多模态行为; Diffusion Policy 专注于从演示中学习复杂的多模态 manipulation 策略。一个有前景的方向是将 diffusion policy 作为 world model 中的 actor 组件 (类似于后续工作 Diffusion World Model)。

### 8.3 Methodology Positioning

| 特性 | Explicit Policy | IBC | Decision Transformer | DreamerV3 | Diffusion Policy |
|------|---------------|-----|---------------------|-----------|-----------------|
| Multi-modal distribution | Limited (GMM) | Good (EBM) | Via return conditioning | Limited | Excellent |
| Training stability | Good | Poor | Good | Moderate | Excellent |
| Inference speed | Fast | Slow (optimization) | Fast | Moderate | Slow (K steps) |
| Demo data efficiency | Moderate | Moderate | N/A (needs returns) | Low (needs RL) | Moderate |
| High-dim action space | Poor scaling | Poor scaling | Good | Good | Excellent |
| Real-world deployment | Common | Rare | Rare | Rare | Validated |

### 8.4 对 Manipulation 领域的意义

Diffusion Policy 对 robot manipulation 的影响是范式级别的:

1. **统一了多模态建模和高维动作预测**: 之前这两个问题需要分别处理，DP 用 diffusion 一次性解决
2. **Position control 的回归**: DP 证明了 position control 在正确的策略表示下优于 velocity control，这改变了 manipulation 社区长期以来的默认选择
3. **降低了部署门槛**: 同一套超参数适用于 push、pour、fold 等差异巨大的任务
4. **开启了 generative control 的时代**: 后续 GR00T N1、$\pi_0$ 等 foundation model 都采用了 diffusion/flow matching 作为 action head 的核心组件
5. **从图像生成到机器人控制的迁移范式**: 证明了"将 action space 视为 pixel space 的低维类比"这一思路的可行性 -- 图像去噪变成了动作去噪，FiLM conditioning 变成了视觉 conditioning

---
