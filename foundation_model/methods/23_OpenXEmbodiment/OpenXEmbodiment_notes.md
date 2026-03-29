# Open X-Embodiment: Robotic Learning Datasets and RT-X Models

**Paper**: Open X-Embodiment Collaboration (21 institutions, 100+ authors), 2023, ICRA 2024
**Repo**: `open_x_embodiment/` (Google DeepMind)

---

## 1. Core Problem

机器人学习领域面临一个结构性困境: NLP 和 CV 已经通过大规模多样数据上的预训练实现了 "generalist model" 范式 (CLIP, GPT 等), 但 robotics 仍然停留在 **每个机器人、每个任务、每个环境单独训练** 的阶段.

根本原因:
- **数据孤岛**: 各实验室采集的数据格式不统一、规模有限 (几千到十几万 episodes), 无法像 web data 一样直接聚合
- **Embodiment gap**: 不同机器人的传感器配置、动作空间、坐标系、控制频率差异巨大, 传统做法认为需要专门的 domain adaptation 机制
- **规模不足**: 即使是最大的单机器人数据集 (RT-1 的 130k episodes), 相比 CV/NLP 的 billion 级数据仍然太小

Open X-Embodiment 要解决的核心问题: **能否通过跨 embodiment 的数据聚合, 不添加任何专门的 embodiment gap bridging 机制, 直接训练出一个受益于 positive transfer 的 "generalist" 机器人 policy?**

这是一个 **infrastructure + empirical validation** 的工作, 重点不在算法创新, 而在于:
1. 构建统一格式的跨 embodiment 数据集
2. 证明现有架构 (RT-1, RT-2) 可以直接利用这些数据实现 positive transfer

---

## 2. Method Overview

### 2.1 数据集构建 Pipeline

```
34 labs, 60 datasets, 22 robot embodiments
  -> RLDS format conversion (tfrecord serialization)
  -> Action space alignment: 7-DoF end-effector (x,y,z,roll,pitch,yaw,gripper)
  -> Image: single canonical camera view, resize to common resolution
  -> Language: natural language instruction annotation
  -> Per-dataset action normalization (保留各自的坐标系和控制模式)
Output: 1M+ trajectories, 527 skills, 160,266 tasks
```

关键设计: 数据对齐是 **coarse alignment** (粗对齐), 不对齐坐标系、不区分 absolute/relative/velocity 控制模式. 同一个 action vector 在不同机器人上代表完全不同的运动. 模型必须自己学会这种区分.

### 2.2 RT-1-X Architecture

RT-1-X 使用与 RT-1 相同的网络架构, 仅在训练数据上做了扩展:

```
Input: 15-frame image history (300x300) + language instruction
  -> USE embedding (language) + EfficientNet-B3 (vision)
  -> FiLM conditioning (fuse language into visual features)
  -> 1x1 Conv projection
  -> TokenLearner (compress to 8 tokens per frame)
  -> Decoder-only Transformer (8 layers, 128 dim, 8 heads)
  -> 256-bin discrete action tokens (11 dimensions)
Output: 7-DoF end-effector action + base movement + termination
```

参数量: 35M

### 2.3 RT-2-X Architecture

RT-2-X 基于 PaLI-X VLM (55B 参数):

```
Input: image + language instruction
  -> ViT-22B (vision) + UL2 32B (language encoder-decoder)
  -> Co-fine-tuning on VLM data + robotics mixture (~1:1)
  -> Actions as text tokens: "1 128 91 241 5 101 127"
Output: 7-DoF action tokens (decoded from language token space)
```

参数量: 55B (RT-2-PaLI-X variant)

### 2.4 Training Details

- **Loss**: categorical cross-entropy over discrete action bins (256 bins for RT-1-X, language token vocabulary for RT-2-X)
- **Robotics data mixture**: 9 embodiments (RT-1, QT-Opt, Bridge, TACO Play, Jaco Play, Cable Routing, RoboTurk, NYU VINN, Austin VIOLA, Autolab UR5, TOTO, Language Table)
- **RT-1-X**: 只用 robotics mixture
- **RT-2-X**: co-fine-tuning, robotics data 和 VLM data 约 1:1 混合
- **Inference**: RT-1 local 3-10 Hz; RT-2 cloud service 1-3 Hz

### 2.5 关键公式

Action tokenization (每个维度独立离散化):
```
token = floor((clip(action, min, max) - min) / (max - min) * (vocab_size - 1))
```

Detokenization:
```
action = token / (vocab_size - 1) * (max - min) + min
```

训练目标 (RT-1-X, per-dimension cross-entropy):
```
L = -sum_{d=1}^{11} log p(a_d = a_d^* | image, language)
```

---

## 3. Key Designs

### 3.1 Coarse Action Space Alignment (粗粒度动作空间对齐)

这是本文最重要的设计决策. 与直觉相反, 作者 **没有** 对齐不同机器人的坐标系, 也没有统一控制模式 (absolute vs delta vs velocity). 只做了最基本的维度对齐: 所有机器人的动作都映射到 7-DoF end-effector space + termination.

为什么这样做有效:
- 不同的 embodiment 天然会出现在不同的视觉场景中 (不同相机视角、不同机器人外观), 模型可以通过视觉 context 隐式区分 embodiment
- 强制对齐坐标系反而可能引入额外的误差和工程复杂度
- 这种做法使得新数据集的接入成本极低 (只需转换为 RLDS format + 7-DoF action)

对后续工作的影响: pi_0 和 Octo 都沿用了这一策略, 用 zero-padding 处理不同维度的动作空间.

### 3.2 RLDS 标准化数据格式

选择 RLDS (Reinforcement Learning Datasets) 作为统一存储格式:
- 基于 tfrecord 的 episode-level 序列化
- 支持任意 observation modalities (多相机 RGB, depth, point cloud, proprioception)
- 支持任意 action spaces
- 支持高效并行数据加载 (兼容 TensorFlow, JAX, PyTorch)

这一基础设施选择对整个领域产生了深远影响: Octo, OpenVLA, pi_0 等后续工作都直接使用 OXE 的 RLDS 数据.

### 3.3 Model Capacity 与 X-Embodiment Transfer 的关系

论文通过对比实验揭示了一个关键 insight:

| 数据规模 | RT-1-X (35M) | RT-2-X (55B) |
|---------|-------------|-------------|
| 小数据集 (Cable Routing, Jaco Play 等) | 显著提升 (positive transfer) | N/A |
| 大数据集 (Bridge, RT-1 data) | 下降 (underfitting) | 提升 |

结论: X-embodiment 数据的 positive transfer 存在一个 **capacity threshold** -- 小模型在吸收多 embodiment 数据时会 underfit, 只有足够大的模型才能同时 represent 多个 embodiment 的 policy. 这直接启发了后续工作 (Octo, pi_0, OpenVLA) 对模型规模的追求.

---

## 4. Experiments

### 4.1 Main Results

**小数据集 domain (RT-1-X, Figure 2)**:

| Evaluation Domain | Original Method | RT-1 (single dataset) | RT-1-X (X-embodiment) |
|------------------|----------------|----------------------|----------------------|
| Kitchen Manipulation (Jaco) | baseline | baseline | significantly higher |
| Cable Routing (Franka) | baseline | baseline | significantly higher |
| NYU Door Opening | baseline | baseline | significantly higher |
| AutoLab UR5 | baseline | baseline | significantly higher |
| Robot Play (Franka) | baseline | baseline | comparable |

RT-1-X 平均成功率比 Original Method 和 RT-1 高 **50%**.

**大数据集 domain (Table I)**:

| | Bridge (WidowX) | Google Robot |
|--|-----------------|-------------|
| Original Method | baseline | baseline |
| RT-1 | comparable | comparable |
| RT-1-X | lower (underfitting) | lower (underfitting) |
| RT-2-X | higher | higher |

### 4.2 Emergent Skills (Table II)

RT-2-X 在 Google Robot 上展现了 emergent skills -- 这些技能来自 Bridge 数据集 (WidowX 机器人), 从未在 Google Robot 上训练过:
- RT-2-X 在 emergent skills 上比 RT-2 提高 ~3x
- 移除 Bridge 数据后, emergent skills 性能显著下降, 证实 positive transfer 确实来自跨 embodiment 数据

### 4.3 Ablation Findings (Table II)

| Design Decision | Impact |
|----------------|--------|
| Image history (vs single frame) | 显著提升泛化 |
| Web pre-training (VLM backbone) | 大模型的关键, 没有则崩溃 |
| 55B vs 5B model | 55B 在 emergent skills 上远优于 5B |
| Co-fine-tuning vs fine-tuning | 类似性能 (因为 robotics data 已经足够多样) |

### 4.4 评估规模

总计 3600 evaluation trials, 6 种机器人, 跨 5 个实验室物理评估.

---

## 5. Related Work Analysis

### 5.1 领域发展脉络

```
单机器人大规模数据 (RT-1, 2022)
  -> VLM + robot action (RT-2, 2023)
  -> 跨 embodiment 数据集 + 模型 (Open X-Embodiment, 2023) <-- 本文
  -> 社区可用的 generalist policy (Octo, 2024)
  -> 更大规模 VLA + flow matching (pi_0, 2024)
  -> 最终 robot foundation model 生态
```

### 5.2 本文的独特定位

与之前的 cross-embodiment 工作 (RoboNet, RoboCat, Gato) 的关键区别:

| 维度 | RoboNet (2019) | RoboCat (2023) | Gato (2022) | Open X-Embodiment (2023) |
|------|---------------|---------------|-------------|--------------------------|
| 数据来源 | 自主采集 | DeepMind 内部 | 多模态混合 | 21 机构开源贡献 |
| Embodiment 数量 | 4 | ~5 | 多种 | 22 |
| 数据格式 | 自定义 | 内部格式 | 混合 | RLDS (标准化开源) |
| 模型开源 | - | 否 | 否 | 是 (RT-1-X checkpoint) |
| 数据开源 | 是 | 否 | 否 | 是 |
| Cross-embodiment mechanism | domain adaptation | self-improvement | 无特殊处理 | 无特殊处理 (粗对齐) |

本文的核心贡献不在于算法创新, 而在于 **证明不需要特殊机制就能实现 cross-embodiment transfer**, 并提供了可复用的 dataset + infrastructure.

---

## 6. Limitations & Future Directions

### 6.1 作者明确指出的局限

- **Embodiment diversity 有限**: 实验只涉及 manipulation 场景, 未覆盖 locomotion, 飞行等模态差异更大的机器人
- **不研究 generalization to new robots**: 只评估了训练过的 embodiment, 未测试 zero-shot transfer 到全新机器人
- **无 positive transfer 判据**: 没有提供什么时候 transfer 有效、什么时候会 hurt performance 的理论分析
- **数据实际使用量受限**: 实验时只用了 9 个 embodiment 的数据 (而非全部 22 个), 因为数据集在实验期间仍在扩展

### 6.2 从代码推断的局限

- **Action space 约束**: RT-1-X 代码硬编码了 11 维动作 (7 DoF arm + gripper + 3 base), 无法处理更高自由度的系统 (如灵巧手, 人形机器人)
- **单相机限制**: 只支持单个 workspace camera RGB 输入, 不支持 wrist camera, depth, proprioception
- **固定图像分辨率**: EfficientNet-B3 要求 300x300 输入, 对不同分辨率的相机需要 resize
- **离散 action token 精度限制**: 256 bins 的离散化对高精度操作 (如 sub-mm 装配) 可能不够
- **无 action chunking**: RT-1-X 每步预测单帧动作, 没有 Diffusion Policy / ACT 那样的 action chunk, 限制了连贯动作序列的生成

---

## 7. Paper vs Code Discrepancies

### 7.1 FFN Block: SwiGLU 而非 Linear

论文描述 RT-1 使用标准 Transformer FFN, 但代码中 RT-1-X 默认使用 **SwiGLU FFN** (`FFNOptions.SWIGLU`):

```python
# rt1.py L39
ffn_option: FFNOptions = FFNOptions.SWIGLU  # default for RT-1-X
```

SwiGLU 是 LLaMA 引入的 gating mechanism, 比标准 FFN 更强. 论文未提及这一改进.

### 7.2 World Vector Range 扩大

RT-1 原版使用 `world_vector_range=(-1.0, 1.0)`, 但 RT-1-X 推理代码使用 `(-2.0, 2.0)`:

```python
# rt1_inference_example.py L163
world_vector_range=(-2.0, 2.0),
```

这是因为 X-embodiment 数据中某些机器人的 end-effector 位移范围更大, 需要扩展 action range. 论文只简单提到 "normalize each dataset's actions prior to discretization", 未说明具体 range 变化.

### 7.3 模型超参数差异

代码中 RT-1-X 推理使用的超参数与 RT-1 论文中的默认值不同:

| Parameter | RT-1 (default in code) | RT-1-X (inference example) |
|-----------|----------------------|---------------------------|
| vocab_size | 256 | 512 |
| num_image_tokens | 8 | 81 |
| layer_size | 128 | 256 |
| num_action_tokens | 11 | 11 |
| sequence_length | 15 | 15 |

vocab_size 从 256 扩展到 512 提供了更细粒度的动作离散化; num_image_tokens 从 8 扩展到 81 意味着 TokenLearner 压缩后保留了更多视觉信息; layer_size 翻倍增加了 Transformer 的容量.

### 7.4 Attention Mask 中的 Action Token 隐藏

代码实现了复杂的 attention mask 逻辑 (`_construct_attn_mask`), 确保:
- 默认模式 (`include_prev_timesteps_actions=False`): 所有 action token 位置被 mask 掉, 模型只基于 image tokens 预测 action
- Image tokens 使用 causal mask
- 这避免了 action tokens 通过 multi-layer Transformer 泄露信息到后续帧的预测中

论文对此没有详细讨论, 但这是保证 training/inference 一致性的关键设计.

### 7.5 TokenLearnerV11 重复定义

`rt1.py` 中重复定义了 `TokenLearnerModuleV11` (line 192-254), 与 `token_learner.py` 中的定义完全相同. 这似乎是代码组织问题, 但不影响功能.

---

## 8. Cross-Paper Comparison

Open X-Embodiment 的核心价值不在于模型本身, 而在于它构建的 **数据基础设施**. 从 RT-1 到 pi_0, 可以清晰看到一条 cross-embodiment VLA 的演进路线:

### 8.1 方法对比

| 维度 | RT-1 (2022) | RT-2 (2023) | Open X-Embodiment (2023) | Octo (2024) | pi_0 (2024) |
|------|------------|------------|--------------------------|-------------|-------------|
| **核心贡献** | 大规模单机器人 policy | VLM -> VLA | 跨 embodiment 数据集 | 开源 generalist policy | VLA + flow matching |
| **参数量** | 35M | 55B | 35M / 55B | 27M / 93M | 3.3B |
| **训练数据** | 130k episodes, 1 robot | 130k episodes + web data | 9 embodiments mixture | 800k trajectories (OXE) | OXE + 自有数据 |
| **动作表示** | 256-bin discrete | text tokens | 同 RT-1/RT-2 | diffusion (continuous) | flow matching (continuous) |
| **Action chunking** | 无 | 无 | 无 | 有 | 有 (H=50) |
| **数据格式** | 内部格式 | 内部格式 | RLDS (标准化) | RLDS (OXE) | RLDS (OXE) |
| **开源** | 否 | 否 | 数据 + RT-1-X checkpoint | 完全开源 | 部分开源 |
| **VLM pre-training** | 无 | PaLI-X / PaLM-E | 同 RT-1/RT-2 | 无 (from scratch) | PaliGemma 3B |
| **Embodiment 数量** | 1 (Google Robot) | 1 (Google Robot) | 9 (实验) / 22 (数据集) | 9+ | 7+ |
| **控制频率** | 3 Hz | 1-3 Hz | 3-10 Hz | 3-10 Hz | 50 Hz |

### 8.2 数据依赖关系

Open X-Embodiment 是连接这些工作的关键纽带:

```
RT-1 data (130k)  ─┐
QT-Opt data        ├── Open X-Embodiment Dataset (1M+ trajectories, RLDS)
Bridge data        │        |
60 other datasets ─┘        |
                            ├── RT-1-X / RT-2-X (本文实验, 9 embodiments, ~350k)
                            ├── Octo (curated 25 datasets, 800k trajectories)
                            ├── OpenVLA (OXE subset)
                            └── pi_0 (OXE Magic Soup + 自有 pi dataset)
```

Octo 代码中的 `oxe_dataset_mixes.py` 直接定义了多个基于 OXE 的数据混合方案:
- `RT_X_MIX`: 原始 RT-X 实验的 9 embodiment 混合
- `OXE_MAGIC_SOUP`: Octo 和 pi_0 使用的扩展混合 (25 datasets)
- `OXE_FLEX_ACT_SOUP`: 包含 ALOHA 等更多数据集的混合

### 8.3 技术演进路线

| 维度 | RT-1 -> OXE | OXE -> Octo | OXE -> pi_0 |
|------|-------------|-------------|-------------|
| **数据** | 单机器人 -> 多机器人 | 精选 subset -> 更大 subset | OXE + 自有高质量数据 |
| **动作空间** | 离散化 | 离散化 -> diffusion | 离散化 -> flow matching |
| **模型容量** | 35M 够用 | 93M 勉强够用 | 3.3B 必要 |
| **核心改进** | N/A | 开源 + finetune 能力 | 高频控制 + action chunk |
| **关键 insight** | 数据规模决定泛化 | 灵活性 > 容量 | web pre-training + 连续动作 |

### 8.4 OXE 对后续工作的关键影响

1. **证明 positive transfer 可行**: RT-1-X 的 50% 提升和 RT-2-X 的 3x emergent skills 提升给了社区信心 -- cross-embodiment training 是值得投入的方向

2. **建立数据标准**: RLDS format 成为事实上的 robot manipulation 数据标准, 后续所有基于 OXE 的工作都直接继承这一格式

3. **揭示 capacity threshold**: 小模型 (35M RT-1-X) 在大数据集上 underfit 的发现, 直接推动了 Octo (93M) -> pi_0 (3.3B) -> GR00T N1 (更大) 的模型规模增长

4. **Coarse alignment 范式**: 证明不需要精确对齐坐标系和控制模式, 模型可以通过视觉 context 隐式区分 embodiment. 这一范式被 Octo 和 pi_0 完全继承, pi_0 用 zero-padding 扩展到 18 维统一动作空间

5. **社区协作模式**: 21 机构的合作模式启发了后续更大规模的数据共享努力 (DROID 等)
