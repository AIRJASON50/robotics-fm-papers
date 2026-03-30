# Octo: An Open-Source Generalist Robot Policy -- 综合分析

**Paper**: Ghosh, Walke, Pertsch, Black, Mees et al., UC Berkeley / Stanford / CMU / Google DeepMind, 2024
**Code**: `/home/l/ws/doc/paper/foundation_model/methods/24_Octo/octo/`

---

## 1. 核心问题 (Core Problem)

如何构建一个**开源的、通用的机器人操作策略 (Generalist Robot Policy, GRP)**，使其能够:

| 挑战 | 具体表现 |
|------|---------|
| 跨机器人形态 | 不同臂 (WidowX, Franka, UR5, Google Robot 等) 有不同的自由度和传感器配置 |
| 异构数据 | 不同数据集有/无 wrist camera、有/无语言标注、不同动作空间 |
| 高效微调 | 预训练模型需在少量目标域数据 (~100 条轨迹) 上快速适配新传感器、新动作空间 |
| 多模态任务规范 | 同时支持自然语言指令和 goal image 两种任务描述 |
| 可复现 | 此前的 GRP (RT-2-X, RoboCat) 均为闭源，社区无法使用和研究 |

核心定位: Octo 是 Berkeley 在 pi_0 之前的 generalist policy 方案，强调**灵活性和开源**，而非极致性能。它是第一个完全开源的 GRP (模型权重 + 训练代码 + 数据加载器)。

---

## 2. 方法概览 (Method Overview)

### 2.1 整体架构

```
输入:
  [Language Instruction]  --> T5-base (111M, frozen) --> language tokens
  [Image Primary (256x256)]  --> SmallStem16 (shallow CNN) --> image patch tokens
  [Image Wrist (128x128)]    --> SmallStem16 --> image patch tokens
  [Goal Image]  --> SmallStem16 --> goal tokens (stacked with obs tokens)

  ↓ 投影到统一 token_embedding_size (ViT-B: 768, ViT-S: 384)
  ↓ 加 learnable positional embeddings

Transformer Backbone (blockwise causal):
  [task tokens (prefix)] | [obs tokens t=0] [readout t=0] | [obs tokens t=1] [readout t=1] | ...
  ↓
  readout token embeddings

Action Head (Diffusion):
  readout embedding -> MLP ResNet (score network) -> DDPM denoising (K=20 steps) -> action chunk (H=4)
```

### 2.2 关键参数

| 配置项 | Octo-Small | Octo-Base |
|--------|-----------|-----------|
| 参数量 | 27M | 93M |
| Transformer layers | 12 | 12 |
| Token dim | 384 | 768 |
| Attention heads | 6 | 12 |
| MLP dim | 1536 | 3072 |
| 训练数据 | 800k trajectories (25 datasets from OXE) | 同左 |
| Batch size | 512 (pretrain config), 2048 (paper) | 同左 |
| 训练步数 | 300k | 300k |
| Window size (history) | 2 frames | 2 |
| Action horizon (chunk) | 4 steps | 4 |
| Action dim | 7 (delta EE + gripper) | 7 |
| Diffusion steps | 20 | 20 |
| Image encoder | SmallStem16 (4-layer conv + patch) | 同左 |
| Language encoder | T5-base (111M, frozen) | 同左 |
| Optimizer | AdamW, inverse sqrt decay, weight decay 0.1, grad clip 1.0 | 同左 |
| Hardware | TPU v4-128 pod, 14h | 同左 |
| Finetune | single A5000, ~5h, 50k steps | 同左 |

### 2.3 关键公式

**Diffusion Training (DDPM)**:

前向过程: `x^k = sqrt(alpha_hat_k) * x^0 + sqrt(1 - alpha_hat_k) * epsilon`, epsilon ~ N(0, I)

训练目标: 最小化噪声预测误差
```
L = E_{k, epsilon} || epsilon_theta(x^k, e, k) - epsilon ||^2
```
其中 `e` 是 transformer readout embedding, `k` 是 diffusion step, `x^k` 是加噪后的 action chunk。

反向采样:
```
x^{k-1} = (1/sqrt(alpha_k)) * (x^k - (1-alpha_k)/sqrt(1-alpha_hat_k) * epsilon_theta(x^k, e, k)) + sqrt(beta_k) * z
```

Noise schedule: cosine schedule (Nichol & Dhariwal 2021)。

**Action Normalization**: 每个数据集独立统计 mean/std, 训练时将 action 标准化到 ~N(0,1), 推理时反归一化。对 gripper 维度不做归一化 (通过 normalization mask)。

---

## 3. 关键设计 (Key Designs)

### 3.1 Blockwise Causal Attention 与 Readout Token 机制

Octo 的 transformer 不是标准的 causal attention, 而是一种**分组因果注意力 (blockwise causal)**:

- **Task tokens (prefix)**: 语言 token 作为前缀, 只相互 attend, 不看 observation
- **Observation tokens**: 每个时间步的图像 token 可以看到所有 task tokens + 当前/过去时间步的所有 observation tokens (causal over timesteps, full within timestep)
- **Readout tokens**: 只被动地"读取"task + observation tokens, 但不被任何 observation/task token attend; 不同 readout 之间也完全独立

代码实现: `block_transformer.py` 中 `AttentionRule` enum 控制注意力规则:
```python
class AttentionRule(Enum):
    NEVER = "never"
    CAUSAL = "other.timestep <= self.timestep"
    CURRENT = "other.timestep == self.timestep"
    STRICT_PAST = "other.timestep < self.timestep"
    ALL = "all"
```

这种设计的好处:
1. Readout 不影响 observation 的计算, 可以在 finetune 时换掉 action head 而不影响预训练的表征
2. 可以灵活添加/删除 observation 输入 (如增加 wrist camera) 而不需要重新初始化 transformer 权重
3. 不同的 readout (action, value 等) 完全解耦, 可以独立添加

### 3.2 Diffusion Action Head -- 连续多模态动作预测

Octo 比较了三种 action head:

| Action Head | 原理 | 优势 | 劣势 |
|-------------|------|------|------|
| MSE (regression) | 直接回归 action, L2 loss | 简单高效 | mode averaging: 多模态数据下预测均值 |
| Discrete (tokenization) | 离散化为 256 bins, cross-entropy | 可建模多模态 | 精度损失, 离散化误差 |
| **Diffusion (DDPM)** | 从 readout embedding 条件化, MLP 做去噪 | 连续 + 多模态 | 推理需多步去噪 |

Ablation 结果: diffusion head 在 zero-shot 和 finetune 评估中均显著优于 MSE 和 discrete head。

Score network 架构 (代码 `diffusion.py`):
```
[Fourier Features(time)] -> MLP -> condition encoding
concat(condition_encoding, obs_embedding, noisy_actions) -> MLPResNet (3 blocks, 256 hidden) -> predicted noise
```

关键实现细节:
- Transformer 只做一次前向传播获取 readout embedding, 之后 20 步去噪全在小 MLP 中完成
- 训练时对 `n_diffusion_samples=1` 个随机 timestep 采样 (config 中), 代码支持多采样
- Action clip 到 [-5, 5] 范围 (`max_action=5.0`)
- 对于 cross-embodiment, 使用 `embodiment_action_dim` 参数: 低维机器人的多余 action 维度在去噪时保持为 noise, 只对有效维度做 denoise

### 3.3 Cross-Embodiment 处理策略

统一动作空间: 所有数据集的 action 统一为 7 维 (6D delta end-effector + 1D gripper), 不足的维度 zero-pad, 通过 `action_pad_mask` 标记有效维度。

数据混合策略 (`oxe_dataset_mixes.py`):
- 从 Open X-Embodiment 的 60+ 数据集中筛选 25 个 (`OXE_MAGIC_SOUP`)
- 筛选标准: 有图像、delta EE 控制、行为多样、非太 repetitive
- 权重调整: "更多样" 的数据集权重 x2, 太 repetitive 的降权
- 缺失模态 (wrist camera, language) 用 zero-padding + mask 处理

Goal relabeling: 使用 hindsight goal relabeling, 从轨迹未来随机选取一帧作为 goal image。训练时随机 dropout language 或 goal image, 使模型能处理两种条件。

---

## 4. 实验结果

### 4.1 Zero-Shot 多机器人控制

在预训练数据分布内的 3 个 robot setup 上评估:

| 对比方法 | WidowX | Google Robot | UR5 |
|---------|--------|-------------|-----|
| RT-1-X (35M) | 较低 | 较低 | 较低 |
| **Octo (93M)** | 较高 | 较高 | 较高 |
| RT-2-X (55B) | ~持平 Octo | ~持平 Octo | -- |

Octo 比 RT-1-X 平均高 29% 成功率, 与 55B 的 RT-2-X 性能相当。
Goal image 条件比 language 条件成功率高 25% (goal image 提供更多信息)。

### 4.2 Finetune 到新域

在 6 个 finetune setup 上 (~100 条轨迹, 50k steps), Octo 平均比次优 baseline 高 52%:

| 方法 | 平均表现 | 备注 |
|------|---------|------|
| ResNet+Transformer (from scratch) | 基线 | 小数据上用 ResNet 比 ViT 更不容易过拟合 |
| VC-1 (pretrained visual repr) | 中等 | 预训练视觉表征 + MLP |
| **Octo Finetune** | 最优 | 全模型微调, 同一套超参数适用所有 setup |

关键: Octo 成功适配了新的传感器 (force-torque), 新的动作空间 (joint position control), 新的机器人形态 (bimanual)。

### 4.3 Ablation 要点

| 设计选择 | 发现 |
|---------|------|
| ViT vs ResNet backbone | ViT 在大规模数据上显著更好; ResNet 在小数据 from-scratch 更好 |
| Diffusion vs MSE vs Discrete | Diffusion 显著最优 |
| 25 datasets vs 11 datasets vs single | 更多数据集 → 更好; 25 datasets > 11 (RT-X mix) > single |
| Model scale (Tiny/Small/Base) | 越大越好, Base 更鲁棒、抓取更精准 |
| Window size | 2 帧历史足够, 更多帧收益递减 |

---

## 5. Related Work 分析

### 5.1 领域发展脉络

```
RT-1 (2022, Google)
  ├── 大规模单机器人 (130k episodes), 离散化 action, EfficientNet+TokenLearner
  ↓
Open X-Embodiment + RT-X (2023, Google et al.)
  ├── 跨机构数据集汇集 (1.5M episodes), RT-1-X/RT-2-X 模型
  ↓
Octo (2024, Berkeley)                          OpenVLA (2024, Stanford/Berkeley)
  ├── 开源 GRP, transformer-first              ├── VLM-based (7B), 离散化 action
  ├── diffusion head, flexible I/O              ├── 更好的 language grounding
  ├── 93M params                                ├── 7B params
  ↓                                             ↓
pi_0 (2024, Physical Intelligence)
  ├── 在 Octo 团队核心成员 (Kevin Black, Karl Pertsch) 基础上发展
  ├── PaliGemma VLM (3B) + Action Expert (300M)
  ├── Flow Matching 替代 DDPM
  ├── 50 步 action chunk @ 50Hz
```

### 5.2 Octo 的独特贡献

1. **第一个完全开源的 GRP**: 模型权重 + 训练代码 + 数据 pipeline 全部公开
2. **第一个支持 finetune 到新 observation/action space 的 GRP**: 之前的 RT-X 等锁定了输入输出
3. **Transformer-first 架构**: 抛弃 ResNet+小 Transformer 的传统设计, 用浅层 CNN 编码 + 大 Transformer backbone
4. **系统性 ablation**: 对架构、数据、目标函数、规模的全面消融实验为后续工作提供了指导

---

## 6. 局限性与未来方向

### 6.1 作者指出的局限

- **Wrist camera 处理不佳**: 只有 27% 数据有 wrist camera, 导致模型未能充分利用 wrist 信息; finetune 时有时去掉 wrist camera 反而更好
- **Language 条件弱于 goal image**: 只有 56% 数据有语言标注, 语言条件性能明显低于 goal image 条件
- **数据质量瓶颈**: 当前只用 optimal demonstration, 未利用次优数据或在线交互数据
- **形态限制**: 仅训练/评估在单臂和双臂操作器, 未扩展到移动操作或导航

### 6.2 从代码推断的限制

- **JAX-only**: 整个框架基于 JAX/Flax, 对 PyTorch 用户不友好 (虽然提供了 PyTorch dataloader)
- **Action space 限制**: 代码中 action 固定为 7 维 (`max_action_dim=7`), 对更高自由度机器人需要修改
- **Diffusion 推理速度**: 20 步 DDPM 去噪在实时控制场景中可能偏慢 (相比 flow matching 的 10 步)
- **T5 encoder 冻结**: 语言编码器完全冻结 (`frozen_keys=("*hf_model*",)`), 无法在 finetune 中改善语言理解
- **Attention mask O(n^2) 生成**: `block_transformer.py` 中 attention mask 通过双重 for 循环逐 token 对生成 (L273-335), 对长序列效率较差

---

## 7. Paper vs Code 差异

| 主题 | 论文描述 | 代码实现 |
|------|---------|---------|
| **Batch size** | 论文说 ViT-B 用 batch_size=2048 | Pretrain config 中 `batch_size=512`, 可能是 per-device batch size (TPU v4-128 = 4x GPU equivalent) |
| **UNet Diffusion Head** | 论文未提及 | 代码中实现了完整的 `UNetDDPMActionHead` (基于 Chi et al. 的 1D U-Net), 与 MLP-based `DiffusionActionHead` 并存 |
| **repeat_task_tokens** | 论文未详细解释 | Pretrain config 设 `repeat_task_tokens=True`: 将 task prefix tokens 复制到每个 observation 时间步, 作为额外的 obs-level token group 参与 blockwise attention; 这相当于一种简单的 cross-modal attention |
| **Task augmentation** | 论文简单提到"randomly zero out" | 代码实现两种策略: (1) `delete_task_conditioning` (随机删除 goal image 或 language); (2) `delete_and_rephrase` (pretrain 专用, 从 `OXE_paraphrases` 库中随机替换语言指令的 paraphrase, p=0.5) |
| **use_correct_attention** | 论文未提及 | 2023年12月发布的旧模型有 attention 计算 bug (`side="left"` in `np.searchsorted`); 新模型用 `use_correct_attention=True` 修复 |
| **Action mask for diffusion** | 论文简单提到 cross-embodiment | Diffusion head 推理时对低维 embodiment 多余的 action 维度使用特殊处理: 将这些维度保持为对应 timestep 的 noise 水平, 而非 denoise, 防止"幻觉"动作 |
| **Multiple action head types** | 论文只讨论 diffusion/MSE/discrete | 代码额外实现了 `L1ActionHead`, `TokenPerDimActionHead`, 以及 `UNetDDPMActionHead` |
| **Default config 不同** | -- | 默认 base `config.py` 用 MSE head + FiLM-conditioned SmallStem + MuseEmbedding (无 T5); pretrain config (`octo_pretrain_config.py`) 才用 Diffusion head + T5 + readout tokens; 说明开发过程中架构经历了多次迭代 |
| **Token language** | 论文说用 T5-base | 默认 base config 用 MuseEmbedding (Google 的文本嵌入), pretrain config 覆盖为 T5-base; 代码保留了对多种文本处理后端的支持 |
| **Weight standardization** | 论文未提及 | ViT encoder (`vit_encoders.py`) 中的 Conv 层使用 weight standardization (`StdConv`), 这是 BiT/ViT-hybrid 的标准做法 |

---

## 8. 跨论文比较 (Cross-Paper Comparison)

### 8.1 架构对比

| 维度 | RT-1 (2022) | Octo (2024) | OpenVLA (2024) | pi_0 (2024) |
|------|-------------|-------------|----------------|-------------|
| **参数量** | 35M | 27M / 93M | 7B | 3.3B |
| **Vision encoder** | EfficientNet + TokenLearner | SmallStem16 (shallow CNN) | DINOv2 + SigLIP (600M) | SigLIP ViT |
| **Language encoder** | FiLM conditioning (嵌入到视觉) | T5-base (111M, frozen) | Llama 2 backbone 内部 | Gemma 内部 |
| **Backbone** | 小 Transformer (8 layers) | 大 Transformer (12 layers, ViT-B) | Llama 2 (7B) | Gemma 2B + Action Expert (300M) |
| **Action representation** | 离散 256 bins, autoregressive | Continuous, diffusion (DDPM) | 离散 256 bins, autoregressive | Continuous, flow matching |
| **Action chunking** | 不支持 (1 step) | 4 steps | 不支持 (1 step) | 50 steps |
| **任务条件** | Language only | Language + Goal image | Language only | Language |
| **Framework** | TensorFlow | JAX/Flax | PyTorch | JAX |

### 8.2 数据与训练

| 维度 | RT-1 | Octo | OpenVLA | Open X-Embodiment | pi_0 |
|------|------|------|---------|-------------------|------|
| **训练数据量** | 130k episodes (单机器人) | 800k trajectories | 970k trajectories | 1.5M episodes (全集) | 10,000+ hours + OXE |
| **数据来源** | Google 内部 | OXE 25 datasets | OXE 全部 | 60+ 研究机构 | PI 自有 + OXE |
| **机器人类型** | Google Robot only | 25 种 (WidowX, Franka, UR5, etc.) | 多种 | 各类 | 7 种 |
| **动作空间** | Delta EE (离散) | Delta EE (连续) | Delta EE (离散) | 各类 (标准化) | 关节角度, Delta EE (连续) |
| **开源程度** | 模型部分开源 | 全部开源 | 全部开源 | 数据开源, 模型部分 | 模型开源 (openpi) |

### 8.3 性能与泛化

| 维度 | RT-1 | Octo | OpenVLA | pi_0 |
|------|------|------|---------|------|
| **Zero-shot 多机器人** | 单机器人 | 9 robots, 与 RT-2-X 持平 | 29 tasks, 超 RT-2-X 16.5% | 7 robots, SOTA |
| **Finetune 到新域** | 未验证 | 52% avg improvement over baselines | 超 Octo, 超 Diffusion Policy 20.4% | pre-train + post-training recipe |
| **语言泛化** | 强 (单机器人数据充足) | 中等 (语言标注比例低) | 强 (VLM backbone) | 强 (VLM backbone) |
| **灵巧操作** | 简单 pick & place | 简单操作 | 简单操作 | 叠衣服、装箱 (高频 50Hz) |
| **长时域任务** | SayCan 组合 | 未测试 | 未测试 | 5-20 分钟任务 |

### 8.4 Octo 到 pi_0 的进化路径

Octo 的几个核心成员 (Kevin Black, Karl Pertsch) 后来加入 Physical Intelligence 开发了 pi_0。关键改进:

| 改进点 | Octo 方案 | pi_0 方案 | 改进原因 |
|--------|----------|----------|---------|
| **VLM backbone** | T5-base (冻结) + 从头训练 Transformer | PaliGemma (3B, 预训练 VLM) | VLM 预训练提供了远超 T5 的视觉-语言联合理解能力 |
| **动作生成** | DDPM (20 steps) | Flow Matching (10 steps Euler) | Flow matching 更高效, 训练更稳定, 概率路径更直接 |
| **Action Expert** | Readout token + 小 MLP diffusion | 独立的 300M Action Expert (MoE) | 防止 VLM 权重被连续动作 token 破坏; Octo 的 readout 机制是其前身思路 |
| **Action chunk** | 4 steps | 50 steps @ 50Hz | 灵巧任务需要长时域、高频动作预测 |
| **模型容量** | 93M | 3.3B | 更大容量能容纳更丰富的行为 |
| **Cross-embodiment** | zero-pad action + mask | zero-pad + 统一到 18D | 更多机器人类型 (mobile base, bimanual) |
| **训练策略** | 单阶段预训练 | Pre-training + Post-training (类 LLM) | 分离泛化能力和执行精度 |

核心洞察: Octo 的 **readout token 机制** 是 pi_0 **action expert** 的前身 -- 两者都试图将"观测理解"和"动作生成"解耦。Octo 用 attention mask 实现只读分离, pi_0 更进一步用独立权重实现完全解耦。

### 8.5 Octo 与 Open X-Embodiment 的关系

Octo 是 OXE 数据集最重要的"消费者"之一:
- OXE 提供了 60+ 数据集、1.5M episodes 的标准化数据格式 (RLDS)
- Octo 从中筛选 25 个数据集 (800k trajectories), 命名为 `oxe_magic_soup`
- Octo 的数据加载代码 (`octo/data/oxe/`) 直接构建在 OXE 标准之上
- RT-X 模型 (OXE 论文的官方模型) 只用了 11 个数据集 (350K episodes), Octo 的数据量更大
- Octo 证明了更大的数据 mix 带来更好的性能, 验证了 OXE 的核心假设

---

## 核心要点总结

1. **Octo 的定位**: 不是最大或最强的模型, 而是**第一个完全开源、支持灵活 I/O 适配的 generalist robot policy**。它为社区提供了一个可复现的 baseline。

2. **技术核心**: Blockwise causal attention + readout tokens 实现了观测编码与动作预测的解耦, 使得 finetune 时可以灵活更换输入/输出而不破坏预训练表征。Diffusion action head 是连续多模态动作预测的关键。

3. **与后续工作的关系**: Octo 是 pi_0 的直接前身。pi_0 在三个方面超越了 Octo: (a) 用 VLM backbone 替代从头训练的 Transformer, (b) 用 flow matching 替代 DDPM, (c) 用 MoE action expert 替代 readout token + MLP。这些改进使 pi_0 能处理更复杂的灵巧操作任务。

4. **实践启示**: 对于资源有限的研究者, Octo 的 finetune pipeline (单 GPU, 5小时, ~100 条轨迹) 仍然是目前最易用的 generalist policy 适配方案之一。
