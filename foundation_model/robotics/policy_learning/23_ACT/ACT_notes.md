# ACT: Action Chunking with Transformers -- Analysis Notes

> Zhao T Z, Kumar V, Levine S, Finn C. RSS 2023
> "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
> Code: https://github.com/tonyzhaozh/act

---

## 1. Core Problem

ACT 要解决的核心问题: **如何用低成本硬件 (<$20K) 实现精细双臂操作的 imitation learning**。

精细操作 (fine manipulation) 对机器人提出三重挑战:

| 挑战 | 具体表现 |
|------|---------|
| 精度要求 | 穿线、插电池等任务容差仅 2-5mm，毫米级误差即导致任务失败 |
| 闭环反馈 | 需要实时视觉反馈来补偿硬件不精确性 (ViperX 精度仅 5-8mm) |
| 接触协调 | 双臂协调、力控交互，传统规划方法难以建模 |

现有方法的不足:

- **高端遥操作系统** (Shadow, DexPilot): 成本 $100K-$400K，不可复现
- **Behavioral Cloning (BC)**: 单步预测面临 compounding error，在精细操作中尤为严重
- **DAgger 及其变体**: 需要专家在线纠正，遥操作场景下不自然且可能引入失败
- **RT-1, BeT**: 离散化动作空间、预处理视觉特征 (frozen encoder)，精度不足

ACT 的核心洞察: 通过 action chunking 将有效决策步长缩短 k 倍，用 CVAE 建模人类示教中的多模态性，在低成本硬件上实现 80-90% 的精细操作成功率。

---

## 2. Method Overview

### 2.1 系统架构: ALOHA + ACT

系统由两部分组成:

1. **ALOHA 遥操作系统**: 2 个 WidowX leader + 2 个 ViperX follower，joint-space mapping，4 个 Logitech C922 webcam (480x640)，50Hz 控制频率
2. **ACT 算法**: CVAE 架构，Transformer encoder-decoder，action chunking + temporal ensemble

### 2.2 Pipeline

```
训练:
  Demo Dataset (50 episodes, ~10 min)
    -> 采样 (o_t, a_{t:t+k})
    -> CVAE Encoder: [CLS] + qpos + action_seq -> z (style variable)
    -> CVAE Decoder: images + qpos + z -> a_hat_{t:t+k}
    -> Loss = L1_reconstruction + beta * KL_divergence

推理:
  观测 o_t (4 cameras + 14D qpos)
    -> CVAE Decoder (z=0): images + qpos -> a_hat_{t:t+k}
    -> Temporal Ensemble: 加权平均重叠 chunks
    -> PID 追踪目标关节位置
```

### 2.3 关键公式

**CVAE 训练目标:**

```
L = L_reconst + beta * L_reg
L_reconst = MSE(a_hat_{t:t+k}, a_{t:t+k})   -- 实际代码使用 L1 loss
L_reg = D_KL(q_phi(z | a_{t:t+k}, o_t) || N(0, I))
```

**Temporal Ensemble 加权:**

```
a_t = sum_i(w_i * A_t[i]) / sum_i(w_i)
w_i = exp(-m * i)   -- i 为该预测的"年龄"，m 控制新旧偏好
```

**有效 horizon 缩减:**

```
原始: pi(a_t | s_t)     -> T 步决策
Chunking: pi(a_{t:t+k} | s_t) -> T/k 步决策
```

### 2.4 模型尺寸

约 80M 参数，单卡 11G RTX 2080 Ti 训练约 5 小时，推理约 0.01 秒。

---

## 3. Key Designs

### 3.1 Action Chunking -- 降低 Compounding Error 的核心机制

**问题本质**: Behavioral Cloning 中，策略在每一步引入的微小误差会在后续步骤中累积。对于 T=400 步的轨迹，即使每步误差极小，最终也会偏离训练分布 (covariate shift)。在精细操作中这一问题尤为致命 -- 差之毫厘，失之千里。

**Action Chunking 的原理**: 不预测单步动作 `a_t`，而是预测未来 k 步动作序列 `a_{t:t+k}`。这将有效决策步长从 T 缩短为 T/k:

- k=1: 标准 BC，400 次决策机会产生 compounding error
- k=100 (论文默认): 仅 4 次决策，误差累积显著减少
- k=episode_length: 完全 open-loop，单次决策但无法响应环境变化

**额外收益**: Action chunking 同时解决了 temporally correlated confounders 问题。人类示教中存在的停顿、犹豫等非 Markovian 行为，在 chunk 级别被平滑掉。单步策略需要额外信息 (如历史) 来区分"正在等待"和"应该移动"，chunk 策略天然回避了这个问题。

**Temporal Ensemble**: 朴素的 chunking 每 k 步切换一次观测，会导致突兀的运动。Temporal ensemble 在每一步都查询策略，对重叠 chunks 做指数加权平均，兼顾精确性和平滑性。关键特性: 这只增加推理开销 (额外 forward pass)，不增加训练成本。

### 3.2 CVAE -- 建模人类示教的多模态性

**问题**: 同一观测下，人类可能采用不同策略完成任务 (如从左边或右边拿杯子)。标准回归会对多种策略取平均，产生"中间"无意义的动作。

**CVAE 的作用**: 引入 latent variable z 来编码示教的"风格"或"意图":
- 训练时: CVAE encoder 从 action sequence + observation 中推断 z 的后验分布 q(z|a,o)
- 推理时: z=0 (prior mean)，确定性解码，消除多模态干扰

**设计细节**:
- Latent dim = 32，经 linear 投影到 hidden_dim=512
- beta = 10 (较高的 KL 权重)，限制 z 的信息量，防止 CVAE encoder 将所有信息编码到 z 中
- 论文中 CVAE encoder 仅使用 proprioception (qpos + action)，不使用图像 -- 加速训练
- Ablation 显示: 对 scripted data 去除 CVAE 几乎无影响 (确定性数据)，对 human data 去除 CVAE 使成功率从 35.3% 降至 2%

### 3.3 低成本硬件方案 (ALOHA)

**Joint-space mapping 遥操作**: 使用运动学相似的 leader-follower 机器人对，直接映射关节角度而非 task-space 位姿。相比 VR controller + IK 的方案:

| 方面 | Joint-space Mapping | VR Controller + IK |
|------|--------------------|--------------------|
| 奇异点 | 天然避免 | IK 经常失败 |
| 关节限位 | 物理保证 | 需要软限位检查 |
| 操作感 | Leader 重量提供阻尼 | 手持设备过轻 |
| 带宽 | 高 (50Hz) | 受限于 IK 计算 |
| 成本 | 需要额外 leader 机器人 (~$3300/个) | 仅需一个 VR 手柄 |

**关键工程决策**:
- 记录 leader (人操作的) 关节位置作为 action，而非 follower 的 -- 因为 PID 追踪有延迟，follower 的实际位置滞后于期望位置
- 3D 打印透明夹爪 + grip tape，保证视觉通透性和抓取力
- 橡皮筋重力补偿，延长遥操作时间 (>30 min)

---

## 4. Experiments

### 4.1 Main Results

8 个任务 (2 仿真 + 6 真实)，与 4 个 baseline (BC-ConvMLP, BeT, RT-1, VINN) 对比:

| 任务 | ACT 成功率 | 最佳 Baseline | 提升 |
|------|-----------|-------------|------|
| Cube Transfer (sim, human) | 90% | BeT 16% | +74% |
| Bimanual Insertion (sim, human) | 86% | BeT 21% | +65% |
| Slide Ziploc (real) | 88% | VINN 28% | +60% |
| Slot Battery (real) | 96% | VINN/RT-1 20% | +76% |
| Open Cup (real) | 84% | BeT 12% (tip over only) | +72% |
| Thread Velcro (real) | 20% | All 0% | -- |
| Prep Tape (real) | 64% | All 0% | -- |
| Put On Shoe (real) | 92% | All 0% | -- |

关键观察:
- 所有 baseline 在真实世界 Slot Battery 和 Slide Ziploc 之外的任务上成功率为 0%
- ACT 在困难任务 (Thread Velcro, Prep Tape) 上虽成功率有限，但是唯一能完成的方法
- 仿真任务中 human data 比 scripted data 更具挑战性 (多模态性)

### 4.2 Ablation Findings

**Action Chunking 效果 (Fig 8a)**:

| Chunk Size k | 平均成功率 (4 settings) |
|-------------|---------------------|
| k=1 (无 chunking) | ~1% |
| k=10 | ~25% |
| k=100 | ~44% (最佳) |
| k=200 | ~42% (略降) |
| k=400 (全 open-loop) | ~38% |

Action chunking 对所有方法均有效 -- 为 BC-ConvMLP 和 VINN 添加 chunking 后也有显著提升，说明这是一个通用技术。

**Temporal Ensemble 效果 (Fig 8b)**:
- ACT: +3.3% 成功率
- BC-ConvMLP: +4% 成功率
- VINN: -20% (非参数方法不适合 temporal ensemble)

**CVAE 效果 (Fig 8c)**:
- Scripted data: 几乎无差异 (确定性数据)
- Human data: 35.3% -> 2% (去除 CVAE 后几乎不可用)

**高频控制用户研究 (Fig 8d)**:
- 50Hz 遥操作: 33s 穿线
- 5Hz 遥操作: 52s (+62% 时间)
- p-value < 0.001 证明 50Hz 优势显著

### 4.3 Hyperparameters (Table III)

| 参数 | 值 |
|------|------|
| Learning rate | 1e-5 |
| Batch size | 8 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| FFN dim | 3200 |
| Hidden dim | 512 |
| Heads | 8 |
| Chunk size | 100 |
| Beta (KL weight) | 10 |
| Dropout | 0.1 |

---

## 5. Related Work Analysis

### 5.1 发展脉络

```
Behavioral Cloning (ALVINN, 1988)
  -> DAgger (2010) -- 交互式纠正，但遥操作场景不实用
    -> BC-ConvMLP -- CNN + MLP 直接回归单步动作
      -> BeT (2022) -- Transformer + 离散化动作，历史 context
        -> RT-1 (2022) -- 大规模 Transformer policy
          -> ACT (2023) -- Action Chunking + CVAE + 低成本硬件

并行路线:
  Action Chunking 概念 (Lai et al., 2022, 心理学灵感)
  CVAE for structured output (Sohn et al., 2015)
  DETR object detection (Carion et al., 2020) -> 提供了 Transformer 架构基础
```

### 5.2 ACT 的独特定位

| 维度 | BC-ConvMLP | BeT | RT-1 | ACT |
|------|-----------|-----|------|-----|
| 动作表示 | 单步连续 | 单步离散+offset | 单步离散 | 连续序列 (chunk) |
| 视觉编码 | CNN (联合训练) | Frozen ViT | Frozen EfficientNet | ResNet18 (联合训练) |
| 多模态建模 | 无 | 离散化 | 离散化 | CVAE |
| Compounding error | 严重 | 严重 | 严重 | k 倍降低 |
| 硬件要求 | 不限 | 不限 | 高端 | 低成本 ($20K) |
| 精度 | 低 | 中 | 低 | 高 (毫米级) |
| 端到端训练 | 是 | 否 (视觉 frozen) | 否 (视觉 frozen) | 是 |

**ACT 的核心贡献**: 首次证明了在 <$20K 的低成本平台上，仅需 10 分钟示教数据，就能学习穿线、插电池等精细操作。此前这些任务被认为需要 $100K+ 的高端硬件。

---

## 6. Limitations & Future Directions

### 6.1 作者明确指出的局限

**硬件限制**:
- Parallel-jaw gripper 无法执行需要多指的任务 (如开儿童安全瓶盖)
- 低成本 Dynamixel 电机扭矩不足 (拧瓶盖、提重物)
- 夹爪边缘太厚，无法撬起贴合表面的胶带

**算法限制**:
- 糖果拆包: 50 个 demo 训练后，撕开成功率 0/10 -- 感知困难 (识别接缝位置) + 数据不足
- 平铺 Ziploc bag: 中空操作阶段失败率高 -- 透明物体感知困难 + 小差异造成大偏差

### 6.2 从代码推断的局限

1. **State dim 硬编码 14**: `detr_vae.py` 中 `state_dim = 14` 写死，无法直接适配非双臂 ViperX 平台 (如灵巧手)
2. **单 backbone 共享**: 所有相机共享同一个 ResNet18 backbone (`backbones[0]`)，不区分 wrist/front/top camera 的视角差异
3. **无 delta action 支持**: 论文提到 delta action 效果不如 absolute position，代码中也无 delta action 选项
4. **无数据增强**: 代码中仅有 ImageNet normalize，无 random crop / color jitter / rotation
5. **Normalization 简陋**: 仅使用 mean/std 归一化 qpos 和 action，代码中 `action_std` 下限 clip 到 1e-2
6. **无 pre-training**: 每个任务从 scratch 训练 ResNet18 backbone，未利用预训练视觉特征

### 6.3 未来方向

- **预训练+微调**: ACT 的架构可以自然地迁移到 foundation model 范式 (pi_0 已实现)
- **更好的感知**: 深度信息、触觉反馈可以解决透明物体感知问题
- **更多数据**: 50 个 demo 对复杂任务仍不足，数据效率是关键瓶颈
- **灵巧手整合**: ALOHA 后续已有 ALOHA 2 (dexterous hand version)

---

## 7. Paper vs Code Discrepancies

### 7.1 架构差异

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Backbone 数量 | 论文 Fig 11 暗示每个相机独立 backbone | 代码共享一个 ResNet18: `self.backbones[0](image[:, cam_id])` (detr_vae.py line 121，注释 HARDCODED) |
| Transformer 层数 | 论文 Fig 11 标注 4 层 encoder + 7 层 decoder | 代码中 CVAE decoder (policy Transformer) 内部还有 `self.encoder` 4 层 self-attention + `self.decoder` 7 层 cross-attention，即 decoder 侧实际有 4+7=11 层 |
| Loss 函数 | 论文 Algorithm 1 写 MSE | 代码使用 L1 loss: `F.l1_loss(actions, a_hat)` (policy.py line 30)。论文正文也提到 "We use L1 loss...we note that L1 loss leads to more precise modeling" |
| Padding 预测头 | 论文未提及 | 代码中有 `is_pad_head`: 预测 action 是否为 padding (detr_vae.py line 53)，但 loss 未使用此预测 |

### 7.2 CVAE Encoder 输入

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| CVAE encoder 输入 | 论文 Fig 4 显示输入为 "[CLS] + joints + action_sequence + PosEmb" | 代码与论文一致，但论文正文说 "we leave out the image observations and only condition on the proprioceptive observation and the action sequence" -- 即 CVAE encoder 不使用图像，仅用 qpos + actions，这是为了加速训练 |
| `env_state` 参数 | 论文未提及 | 代码中 forward 接收 `env_state` 参数，当 backbone 为 None 时使用 `input_proj_env_state` 将 7D env_state 投影到 hidden_dim (detr_vae.py line 62)。这是一个 state-only 模式的后门 |

### 7.3 Temporal Ensemble 实现细节

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| 权重参数 m | 论文 Algorithm 2 使用 `w_i = exp(-m*i)` | 代码中 `k = 0.01` (imitate_episodes.py line 255)，比论文默认值小得多，意味着更慢地纳入新观测 |
| Query frequency | 论文说 "query the policy at every timestep" (temporal agg 时) | 代码确认: `query_frequency = 1` when `temporal_agg=True` (imitate_episodes.py line 193) |
| 填充检查 | 论文未提及 | 代码检查 `actions_populated = torch.all(actions_for_curr_step != 0, axis=1)` 跳过全零条目 (line 253)，这在前几步 buffer 未满时很重要 |

### 7.4 数据处理

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| Action 对齐 | 论文未提及 | 真实数据中: `action = root['/action'][max(0, start_ts - 1):]` (utils.py line 47)，注释 "hack, to make timesteps more aligned" -- 存在 1 步偏移补偿 |
| 数据分割 | 论文未提及 | Train/Val = 80/20 random split (utils.py line 114-117) |
| ImageNet Normalize | 论文未提及 | 代码在 ACTPolicy.__call__ 中对图像做 ImageNet normalize (policy.py line 20-21)，每次前向都重新创建 transforms.Normalize 对象 |

### 7.5 DETR 遗留代码

代码基于 DETR (object detection) 修改而来，保留了大量未使用的代码:

- `masks` 参数 (segmentation head 开关，从未设为 True)
- `NestedTensor` 类型 (backbone.py 的 Joiner 声明接收 NestedTensor，实际传入普通 Tensor)
- `return_intermediate_dec=True` 在 `build_transformer` 中硬编码 (transformer.py line 302)，但返回的中间层结果并未使用
- `FrozenBatchNorm2d` 来自 DETR，带有 TODO 注释质疑是否需要 frozen BN (backbone.py line 94)
- `dilation` 参数保留但从不使用

### 7.6 训练中的 TODO

代码中多处 TODO 暗示未完成的优化:

- `self.latent_dim = 32 # final size of latent z # TODO tune` (detr_vae.py line 67)
- `state_dim = 14 # TODO hardcode` (detr_vae.py line 230)
- `self.action_head = nn.Linear(1000, state_dim) # TODO add more` (detr_vae.py line 156, CNNMLP baseline)
- `env_state = None # TODO` (policy.py line 19)
- `# TODO shared with VAE decoder` (detr_vae.py line 217, encoder layers 与 decoder 共享层数配置)

---

## 8. Cross-Paper Comparison

### 8.1 ACT vs Diffusion Policy vs pi_0: 核心方法对比

| 维度 | ACT (2023) | Diffusion Policy (2023) | pi_0 (2024) |
|------|-----------|----------------------|-------------|
| **动作生成方式** | CVAE decoder (单次前传) | DDPM 迭代去噪 (K步) | Flow Matching (10步 Euler) |
| **多模态处理** | CVAE latent z | Diffusion 天然多模态 | Flow Matching 天然多模态 |
| **Action Chunking** | 首次提出，k=100 | 采纳: prediction horizon T_p=16 | 采纳: action horizon=50 |
| **推理延迟** | ~10ms (1次前传) | ~100ms (K步去噪) | ~73ms (10步积分) |
| **视觉编码器** | ResNet18 (从头训练) | ResNet-18 + SpatialSoftmax | SigLIP ViT (预训练) |
| **语言条件** | 无 | 无 | PaliGemma VLM |
| **模型规模** | ~80M | ~30-100M | 3.3B |
| **预训练** | 无 | 无 | 10,000h 跨机器人数据 |
| **控制频率** | 50Hz | 10-20Hz (受去噪步骤限制) | 50Hz (chunked) |
| **跨机器人** | 否 (仅 ViperX 双臂) | 否 (per-task) | 是 (7种机器人形态) |

### 8.2 Action Chunking -- 三篇论文共享的关键思想

ACT 首先提出 action chunking 并给出了理论动机，后续两篇工作均采纳:

| 方面 | ACT | Diffusion Policy | pi_0 |
|------|-----|-----------------|------|
| **Chunk 大小** | k=100 (2秒@50Hz) | T_p=16 (预测), T_a=8 (执行) | H=50 (1秒@50Hz) |
| **执行策略** | Temporal ensemble (指数加权) | Receding horizon (执行 T_a 步后重新规划) | Open-loop 执行整个 chunk |
| **理论动机** | 降低 effective horizon T/k | 提供 temporal consistency，避免模态切换 | 高频控制 + 降低推理频率 |
| **设计差异** | 每步查询策略 + ensemble | 仅在 T_a 步耗尽后查询 | 每 0.5-0.8s 查询一次 |

**ACT 的贡献**: 提供了 action chunking 降低 compounding error 的清晰机制 -- 决策点减少 k 倍。Diffusion Policy 和 pi_0 各自扩展了这一思想:
- Diffusion Policy 将 action chunking 与 diffusion denoising 结合，chunking 同时保证了 temporal consistency (同一去噪过程生成的序列内在一致)
- pi_0 将 action chunking 与 flow matching 结合，在保持 50Hz 控制频率的同时降低了推理频率 (每秒仅需 1-2 次推理)

### 8.3 多模态动作分布的处理策略

| 方法 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| ACT (CVAE) | Latent z 编码 "风格"，推理时 z=0 取 prior mean | 推理确定性高，训练快 | 推理时固定为一种模态 (z=0)，无法采样多样动作 |
| Diffusion Policy | 从噪声迭代去噪 | 天然采样多模态分布 | 推理慢 (K步)，训练需更多 epoch |
| pi_0 (Flow Matching) | ODE 积分从噪声到动作 | 10步即可，比 DDPM 快 | 仍需多步推理 |

**深层差异**: ACT 在推理时将 z 设为 0，本质上选择了 "最常见" 的行为模式。这对于确定性较高的精细操作是合理的 (穿线就只有一种方式)。但对于真正需要多样性的任务 (如家务中的路径选择)，Diffusion Policy / pi_0 的分布采样更自然。

### 8.4 ACT 对后续工作的影响

```
ACT (2023)
  |
  |-- [Action Chunking 概念] --> Diffusion Policy 采纳 (T_p/T_a 设计)
  |                          --> pi_0 采纳 (H=50 chunk)
  |                          --> GR00T N1 采纳
  |
  |-- [ALOHA 硬件平台] --> ALOHA 2 (Mobile ALOHA, 移动底盘)
  |                    --> pi_0 直接使用 ALOHA 收集数据并适配
  |                    --> 催生了低成本遥操作研究热潮
  |
  |-- [CVAE for policy] --> pi_0 转向 flow matching (更强的分布建模)
  |                      --> 证明了 "生成模型做策略" 这一范式的可行性
```

**特别值得注意**: pi_0 的代码中包含专门的 `aloha_policy.py` 适配器，包含关节翻转和 gripper 转换逻辑，说明 ALOHA 已经成为 VLA 研究的标准数据收集平台之一。ACT 在 pi_0 的表述中被定位为 "灵巧但只能从头训练，无法利用大规模预训练数据" 的方法 -- pi_0 的核心创新之一就是将 ACT 的 action chunking 思想与大规模 VLM 预训练结合起来。

### 8.5 对 Dexterous Manipulation 研究者的核心启示

1. **Action Chunking 是通用技术**: 不依赖特定架构，可以与任何 BC 方法结合。这是本文最有长期价值的贡献。
2. **CVAE vs Diffusion 的选择**: 如果任务确定性高 (精密装配)，CVAE 更高效; 如果任务多模态性强 (日常操作)，Diffusion/Flow Matching 更合适。
3. **低成本硬件的关键**: 不是降低要求，而是用 high-frequency control + learning 补偿硬件不精确性。5-8mm 精度的 ViperX 能完成 2-5mm 容差的任务，全靠 50Hz 闭环视觉反馈。
4. **数据效率**: 50 个 demo (10 分钟) 达到 80-90% 成功率，这对资源有限的实验室非常友好。但更复杂的任务 (Thread Velcro: 100 个 demo 仍只有 20%) 仍需要更多数据或更好的表征。

---
