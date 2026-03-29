# GR00T N1 -- An Open Foundation Model for Generalist Humanoid Robots

**Paper**: NVIDIA, arXiv 2503.14734, March 2025
**Code**: [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (Apache-2.0)
**Model**: GR00T-N1-2B (2.2B params), 后续更新至 N1.5/N1.6

---

## 1. Core Problem

通用人形机器人 foundation model 面临三个核心挑战:

1. **数据孤岛问题 (Data Island Problem)**: 不同于 NLP/CV 拥有 Internet 规模的数据, 人形机器人数据极度稀缺。各 embodiment 的传感器、自由度、控制模式差异巨大, 导致跨平台数据难以统一使用。任何单一人形硬件的数据都远不够训练一个真正的 generalist model。

2. **跨 embodiment 统一控制**: 从单臂桌面机器人到双臂灵巧手人形, 不同 embodiment 的 state/action 维度和语义差异极大。如何用一套权重同时支持 Franka Panda (7-DoF + gripper)、Google Robot、GR-1 Humanoid (全身关节) 等截然不同的平台。

3. **高效适应新任务/新场景**: Real-world 场景变化大, 需要模型具备 zero-shot 泛化能力, 同时在少量示范数据下快速 post-training 适应具体任务。

NVIDIA 的核心 insight: 构建 **data pyramid** -- 将异构数据源按规模分层 (web 视频 > 合成数据 > 真实机器人数据), 结合 **cross-embodiment** 训练和 **dual-system architecture**, 从而绕过单一数据源不足的困境。

---

## 2. Method Overview

### 2.1 整体架构: Dual-System VLA

GR00T N1 采用 Vision-Language-Action (VLA) 架构, 受 Kahneman "Thinking, Fast and Slow" 启发的双系统设计:

- **System 2 (慢速推理)**: Vision-Language Model -- Eagle-2 VLM (基于 SmolLM2 LLM + SigLIP-2 image encoder), 负责理解视觉场景和语言指令。VLM 在 10Hz 运行。
- **System 1 (快速执行)**: Diffusion Transformer (DiT) + flow matching, 负责生成连续动作。Action 在 120Hz 执行 (每次生成 H=16 的 action chunk)。

两个系统通过 cross-attention 紧密耦合, 端到端联合训练。

```
Image + Language --> [Eagle-2 VLM (System 2)] --> VL token embeddings
                                                        |
                                                  cross-attention
                                                        |
State --> [Embodiment-specific State Encoder] -+--> [DiT Blocks (System 1)] --> [Action Decoder] --> Actions
                                               |
Noised Actions --> [Action Encoder + timestep] -+
```

### 2.2 关键模块

**Vision-Language Module**:
- Eagle-2 VLM: 1.34B params, SigLIP-2 编码 224x224 图像, pixel shuffle 后得到 64 image tokens/帧
- 使用中间层 (第 12 层) 而非最终层的 LLM embeddings -- 实验证明这既提速又提升成功率
- N1.6 版本升级为 Cosmos-Reason-2B VLM, 支持 flexible resolution (native aspect ratio, 无需 padding)

**Diffusion Transformer (DiT)**:
- Flow matching loss: $L_{fm}(\theta) = E_{\tau}[\|V_\theta(\phi_t, A_t^\tau, q_t) - (\epsilon - A_t)\|^2]$
- Beta 分布采样 timestep: $p(\tau) = \text{Beta}(\frac{s-\tau}{s}; 1.5, 1)$, $s=0.999$
- 推理时使用 K=4 步 forward Euler 去噪
- N1: 16 layers; N1.6: **32 layers** (2x)

**Embodiment-specific Encoders/Decoders**:
- 每个 embodiment 使用 `CategorySpecificMLP` -- 维护独立的权重矩阵 `W[num_categories, input_dim, hidden_dim]`
- 支持最多 32 个 embodiment (预设, 可扩展)
- Action encoder 融合 noised action + sinusoidal timestep encoding

### 2.3 训练数据 (Data Pyramid)

| 层级 | 数据源 | 规模 | Action 来源 |
|------|--------|------|-------------|
| Base (底层) | Human egocentric videos (Ego4D, EPIC-KITCHENS 等 7 个数据集) + web data | 最大 | Latent actions (VQ-VAE) |
| Middle (中层) | Neural trajectories (WAN2.1 视频生成) + Simulation (DexMimicGen) | ~827h neural + 6500h sim | Latent actions / IDM pseudo-actions |
| Peak (顶层) | 真实遥操作数据 (GR-1, OXE, AgiBot-Alpha) | 88h in-house + OXE + 140k AgiBot | Ground-truth robot actions |

**Latent Actions**: 训练 VQ-VAE 从 consecutive frames $(x_t, x_{t+H})$ 提取 latent action embedding, 统一跨 embodiment 的 action space。不同 embodiment (含人类) 的相似动作会映射到相似的 latent embedding。

**Neural Trajectories**: 在 real teleoperation data 上 fine-tune image-to-video model (WAN2.1-I2V-14B + LoRA), 给定初始帧和新 language prompt 生成 counterfactual 轨迹。用 LLM 做质量筛选。~300k 轨迹 = 827h, 约 105k L40 GPU hours (3600 GPUs x 1.5 days)。

**Simulation Trajectories**: 基于 DexMimicGen + RoboCasa, 从少量人类演示自动扩增。54 种 source/target receptacle 组合, 每种 10k demos, 共 540k demos。11 小时生成 780k 轨迹 = 6500h 等效人类数据。

### 2.4 训练流程

| 阶段 | 冻结策略 | Batch Size | Steps | 数据 |
|------|----------|------------|-------|------|
| Pre-training | VLM language frozen, rest trainable | 大 | ~50k H100 GPU-hours | 全数据金字塔 |
| Post-training | VLM language frozen | 128-1024 | 60k steps | 单 embodiment 任务数据 |

- **辅助损失**: Object detection loss -- 用 OWL-v2 标注目标物体 bounding box center, 附加 $L_{det} = \|x_{pred} - x_{gt}\|^2$, 增强空间理解
- 总损失: $L = L_{fm} + L_{det}$

---

## 3. Key Designs

### 3.1 Cross-Embodiment via Embodiment-Specific Projectors

这是 GR00T N1 处理多 embodiment 的核心机制。每个 embodiment 有独立的 state encoder / action encoder / action decoder 权重, 而 DiT 和 VLM 权重共享。

代码实现中 (`gr00t/model/modules/embodiment_conditioned_mlp.py`):
```python
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]  # per-embodiment weight selection
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

这种设计相比 pi_0 的 MoE (Mixture-of-Experts) 方案更简洁: 不需要 gating network, 而是通过 embodiment ID 直接索引对应权重。支持新增 embodiment 只需添加一组 projector 权重。

N1.6 的 `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` (`gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`) 已预注册 9 个 embodiment: `oxe_google(0)`, `oxe_widowx(1)`, `libero_panda(2)`, `unitree_g1(8)`, `new_embodiment(10)`, `robocasa_panda_omron(13)`, `oxe_droid(16)`, `gr1(20)`, `behavior_r1_pro(24)` -- 稀疏索引说明 pre-training 涉及更多 embodiment (最多 32 个)。

### 3.2 Data Pyramid + Neural Trajectory Augmentation

Data pyramid 的分层策略 (底层提供视觉先验, 中层提供行为多样性, 顶层提供 embodiment-grounded 精确动作) 是 GR00T N1 的方法论贡献。

Neural trajectory 的关键 insight: 用视频生成模型 (WAN2.1) 做 data augmentation, 将 88h teleoperation 数据扩增到 827h, 覆盖 counterfactual scenarios (不同物体、不同放置位置)。质量控制通过 LLM 判断视频是否 follow instruction。

实验证明 neural trajectory co-training 在 RoboCasa 上带来 +4.2%~+8.8% 提升, 在真机 GR-1 上 +5.8%。

### 3.3 AlternateVLDiT -- Image/Text Token 交替 Attention

N1.6 引入的架构改进 (`gr00t/model/modules/dit.py`): `AlternateVLDiT` 在 cross-attention 时将 image tokens 和 non-image (text) tokens 分开处理:

```python
for idx, block in enumerate(self.transformer_blocks):
    if idx % 2 == 1:
        # Self-attention blocks
        ...
    else:
        # Cross-attention: alternate between text and image
        if idx % (2 * self.attend_text_every_n_blocks) == 0:
            curr_mask = non_image_attention_mask  # attend to text
        else:
            curr_mask = image_attention_mask       # attend to image
```

默认 `attend_text_every_n_blocks=2`, 即每 4 个 cross-attention block 中, 1 个 attend text, 3 个 attend image (2:1 image-to-text ratio)。这种设计让 DiT 对视觉信息 attend 更多, 同时不遗漏 language conditioning。

---

## 4. Experiments

### 4.1 Simulation Benchmarks

| Benchmark | Embodiment | 任务数 | Demo/task | GR00T N1 (100 demos) | DP Baseline | BC-Transformer |
|-----------|-----------|--------|-----------|----------------------|-------------|----------------|
| RoboCasa Kitchen | Franka Panda | 24 | 30/100/300 | 38.6% | 30.1% | 11.5% |
| DexMimicGen | 3 embodiments (Panda+gripper, Panda+hand, GR-1) | 9 | 30/100/300 | 52.4% | 43.7% | 36.2% |
| GR-1 Tabletop | GR-1 Humanoid | 24 | 30/100/300 | **43.3%** | **25.8%** | 12.1% |

GR-1 Tabletop 上 GR00T N1 比 Diffusion Policy 高出 **17.5%**, 体现了 pre-training 在 humanoid embodiment 上的显著优势。

### 4.2 Real-World (GR-1 Humanoid)

| 任务类别 | 任务数 | GR00T N1 (10% data) | GR00T N1 (full) | DP (10% data) | DP (full) |
|----------|--------|---------------------|-----------------|---------------|-----------|
| Pick-and-Place | 5 | -- | -- | -- | -- |
| Articulated | 3 | -- | -- | -- | -- |
| Industrial | 3 | -- | -- | -- | -- |
| Coordination | 2 | -- | -- | -- | -- |
| **Average** | 13 | **60.0%** | **65.6%** | 27.6% | 35.2% |

关键发现:
- GR00T N1 用 10% 数据 (60.0%) 几乎追平 DP 用 full data (35.2%), 甚至超出 -- 极强的 data efficiency
- Pre-trained model zero-shot: 手部交接任务 76.6%, 新物体放置 73.3%
- Post-training 后 smooth motion + 高 grasping accuracy, DP baseline 常出现 "immobility" 和 inaccurate grasping

### 4.3 N1.6 额外 Benchmark (开源代码中)

| Benchmark | 结果 |
|-----------|------|
| BEHAVIOR-1K (50 tasks, Galaxea R1 Pro, loco-manipulation) | N1.6 avg 26.3% vs Pi0.5 avg 11.3% (Task Progress) |
| PointNav (Unitree G1, 导航) | In-dist 86.3%, OOD 76.5% (vs COMPASS baseline 84.7%/45.6%) |
| G1 LocoManipulation (pick-and-place + 行走) | 58% success rate |

---

## 5. Related Work Analysis

### 5.1 与主流 VLA 方法对比

| 方法 | VLM Backbone | Action Head | Cross-Embodiment | Data Strategy | 参数量 |
|------|-------------|-------------|-----------------|---------------|--------|
| **GR00T N1** | Eagle-2 (SmolLM2 + SigLIP-2) | DiT + flow matching | Embodiment-specific projectors | Data pyramid (video+sim+real) | 2.2B |
| RT-2 (Google) | PaLI-X / PaLM-E | Token-based (离散化 action) | 单一 embodiment | Web data + robot data | 55B |
| pi_0 (Physical Intelligence) | VLM + MoE | Flow matching + MoE | MoE gating | 多平台 teleop | ~3B |
| Octo | 轻量 transformer | Diffusion | Embodiment-specific heads | OXE 数据集 | 93M |
| Helix (Figure) | 未公开 | Diffusion | 单一 humanoid | 大规模 teleop | 未公开 |

**GR00T N1 vs RT-2**: RT-2 将 action 离散化为 token, 丢失连续动作的精度; GR00T N1 用 flow matching 生成连续 action, 更适合精细操作。RT-2 参数量 55B 远大于 GR00T N1 的 2.2B, 推理成本高得多。

**GR00T N1 vs pi_0**: pi_0 用 MoE 做 VLM-to-action bridging, GR00T N1 用简单 cross-attention + embodiment-specific projectors。GR00T N1 的方案更灵活 (VLM 和 action model 可独立替换), 而 pi_0 的 MoE 可能更适合需要 task-level routing 的场景。

**GR00T N1 vs Octo**: Octo 也使用 embodiment-specific projectors, 但没有 fine-tune VLM, 且缺少 data pyramid 策略。GR00T N1 通过 VLM fine-tuning + 多层数据融合显著超越 Octo。

### 5.2 Data Augmentation 路线比较

| 方法 | 增强手段 | 规模 |
|------|---------|------|
| GR00T N1 | Video generation (WAN2.1 + LoRA) | 827h neural trajectories |
| GenAug / ROSIE | Text-to-image inpainting | 单帧级别 |
| MimicGen / DexMimicGen | 仿真轨迹变换 | 百万级轨迹 |
| GR00T N1 sim | DexMimicGen | 780k trajectories / 6500h |

GR00T N1 是首个在视频生成模型 (video generation model) 规模上做 neural trajectory augmentation 的工作, 且用 LLM 做质量控制, 比之前的 image-level augmentation 高一个量级。

---

## 6. Limitations & Future Directions

### 论文明确指出的局限

1. **短 horizon 限制**: 当前只聚焦 short-horizon tabletop manipulation, 尚未验证 long-horizon loco-manipulation
2. **合成数据质量**: 视频生成模型仍难以产生 physically plausible + counterfactual 的多样数据, 生成质量是瓶颈
3. **Vision-language backbone 的空间推理能力**: 需要更强的 VLM 来增强空间理解和语言适应性

### 从代码推断的局限

4. **Action space 统一粗糙**: `max_state_dim=29`, `max_action_dim=29` 用 padding+mask 处理不同维度, 但 29 维对 full humanoid 可能不够 (N1.6 的 BEHAVIOR R1 Pro state 达 82 维)
5. **单帧输入**: 代码默认 `delta_indices=[0]` -- 只取当前帧做观测, 缺少时序信息 (与 ACT 等用多帧 observation history 的方法不同)
6. **训练方差大**: README 提到 "performance differences as large as 5-6% between runs", 且存在非确定性 image augmentation
7. **Locomotion 仍依赖外部控制器**: G1 WholeBodyControl 示例中, GR00T 只输出关节目标, 底层平衡仍由单独的 WBC (whole-body controller) 完成 -- 并非端到端 locomotion

### Future Directions

- Long-horizon loco-manipulation: 需要 hierarchical planning + whole-body control 整合
- 更强 VLM backbone: 从 2B 升级到更大模型, 或用 Cosmos 系列 world model
- World model 融合: NVIDIA 的 Cosmos 平台可能与 GR00T 结合, 做 model-based planning
- 更高效的合成数据: physics-aware video generation

---

## 7. Paper vs Code Discrepancies

这是论文与开源代码 (Isaac-GR00T repo, N1.6 branch) 之间的差异分析。

### 7.1 架构差异

| 方面 | 论文 (N1) | 开源代码 (N1.6) |
|------|-----------|-----------------|
| VLM Backbone | Eagle-2 (SmolLM2 + SigLIP-2) | Cosmos-Reason-2B VLM variant (README), 但代码中实际加载 `nvidia/Eagle-Block2A-2B-v2` |
| DiT layers | 16 layers | **32 layers** (`gr00t/configs/model/gr00t_n1d6.py`: `"num_layers": 32`) |
| Post-VLM adapter | 论文未详述 | N1.5 有 4-layer transformer adapter, N1.6 移除, 改为 unfreeze VLM top 4 layers (`tune_top_llm_layers: int = 4`) |
| DiT variant | 标准 DiT | **AlternateVLDiT** -- image/text token 交替 attention (`gr00t/model/modules/dit.py`) |
| Action representation | Absolute joint angles / EEF positions | N1.6 默认 **state-relative action** (`use_relative_action: bool = True`), 配合 `ActionRepresentation.RELATIVE` |

### 7.2 论文未提及但代码实现的内容

1. **State dropout** (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`): `state_dropout_prob` 参数, 训练时随机将 state features 替换为可学习的 `mask_token`, 增强对 state 信息缺失的鲁棒性。论文未讨论。

2. **State additive noise** (同文件): `state_additive_noise_scale` 添加 Gaussian noise 到 state features, 作为正则化。论文未提及。

3. **Timestep discretization** (`num_timestep_buckets=1000`): Flow matching 的连续 timestep 被离散化为 1000 个 bucket, 用于 action encoder 的 sinusoidal encoding。论文只描述了连续 timestep。

4. **EMBODIMENT_TAG_TO_PROJECTOR_INDEX** (`gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`): 稀疏索引 (0,1,2,8,10,13,16,20,24) 说明 pre-training 使用了更多 embodiment, 开源的只是 post-training 常用的子集。

5. **Image augmentation pipeline** (`gr00t/model/gr00t_n1d6/image_augmentations.py`): 大量增强策略 (albumentations, color jitter, random rotation, mask-guided background suppression), 远比论文描述的丰富。

6. **VL LayerNorm** (`use_vlln: bool = True`): 在 VLM output 和 DiT 之间有一个 LayerNorm, 论文未提。

7. **Attention dropout** (`attn_dropout: float = 0.2`): DiT 使用 0.2 dropout, 且最后一层有 final dropout, 论文表格中未提及具体值。

8. **辅助 object detection loss**: 论文放在 Appendix F, 代码中不在主开源仓库 (可能仅用于内部 pre-training)。

### 7.3 开源代码缺失的内容

| 功能 | 状态 |
|------|------|
| Pre-training 代码 | 未开源, 只提供 post-training / fine-tuning |
| VQ-VAE latent action 训练 | 未开源 |
| IDM (Inverse Dynamics Model) | 未开源 |
| Neural trajectory generation pipeline | 未开源 |
| DexMimicGen 数据生成 | 部分开源 (外部依赖) |
| Human video 数据处理 | 未开源 |
| Object detection 辅助损失 | 未在开源代码中 |
| Multi-node 分布式训练 (OSMO + Ray) | 未开源 |

---

## 8. Cross-Paper Comparison

### 8.1 与 Diffusion Policy 的方法论对比

| 维度 | Diffusion Policy (Chi et al., 2024) | GR00T N1 |
|------|--------------------------------------|----------|
| **Action generation** | DDPM denoising (U-Net), 100 steps | Flow matching (DiT), **4 steps** |
| **Conditioning** | Image encoder (ResNet/ViT) + spatial softmax | VLM (Eagle-2) full language+vision reasoning |
| **Language** | 不支持 | 原生 language-conditioned |
| **Architecture** | U-Net | Transformer (DiT) -- 更好的 scaling |
| **Cross-embodiment** | 不支持 | Embodiment-specific projectors |
| **Pre-training** | 无 (from scratch) | Data pyramid 大规模 pre-training |
| **推理延迟** | ~100 steps x U-Net forward | 4 steps x DiT forward, 63.9ms/chunk on L40 |
| **Data efficiency** | 需要较多 demo | 10% data 即可超过 DP full data |

核心差异: Diffusion Policy 是 task-specific 的 imitation learning 方法, 不做 pre-training, 不支持 language; GR00T N1 是 foundation model, 通过大规模 pre-training 获得 prior, 然后 data-efficient post-training。GR00T N1 继承了 Diffusion Policy 的 action chunking 思想 (H=16), 但用 flow matching 替代 DDPM (4 步 vs 100 步), 大幅降低推理延迟。

### 8.2 与 DreamerV3 的方法论对比

| 维度 | DreamerV3 (Hafner et al., 2023) | GR00T N1 |
|------|----------------------------------|----------|
| **范式** | Model-based RL (world model) | Imitation Learning (behavior cloning) |
| **World model** | RSSM (学习 transition model) | 无显式 world model (但 VLM 提供隐式 scene understanding) |
| **Exploration** | 主动探索 (reward-driven) | 被动模仿 (teleoperation data) |
| **Reward** | 需要 reward function | 不需要 |
| **Real-world transfer** | Sim-to-real gap 挑战 | 直接 real data 训练, 但需要人工演示 |
| **Scalability** | 单任务训练, 难以跨 embodiment | 多任务多 embodiment 统一训练 |
| **应用场景** | Game/sim 为主 | 真实机器人部署 |

方法论启示: DreamerV3 学习 dynamics model 做 planning, GR00T N1 用 VLM 做 implicit reasoning + DiT 做 reactive control。两者可能互补 -- GR00T 的 VLM 提供 semantic understanding, 而 world model 可以提供 physics-aware planning。NVIDIA 的 Cosmos 平台正在探索这个方向。

### 8.3 对 Manipulation + Locomotion 统一控制的启示

从 GR00T N1/N1.6 的代码和实验可以看出:

1. **当前状态: Manipulation 为主, Locomotion 为辅**
   - 核心验证仍是 tabletop manipulation (RoboCasa, DexMimicGen, real GR-1)
   - Locomotion 通过 UNITREE_G1 embodiment 的 `navigate_command` 和 `base_height_command` 支持 (`gr00t/configs/data/embodiment_configs.py`)
   - PointNav 结果证明 GR00T 可以做导航 (86.3% in-dist), 但底层仍依赖 COMPASS 的 whole-body controller

2. **统一控制的架构支撑**
   - Embodiment-specific projectors 天然支持不同 action space (manipulation 的 EEF + locomotion 的 velocity cmd)
   - BEHAVIOR-1K benchmark (loco-manipulation) 中 Galaxea R1 Pro 的 action space 包含 `base`, `torso`, `left_arm`, `right_arm` -- 全身控制
   - N1.6 对 G1 的 loco-manipulation (navigate + pick-and-place) 达到 58% 成功率

3. **瓶颈与方向**
   - 论文/代码中 locomotion 部分的 action 仍是 high-level command (velocity, height), 非直接关节力矩
   - 真正的端到端 whole-body control (类似 RL policy 直接输出所有关节) 尚未在 GR00T 框架中实现
   - NVIDIA 的路线图可能是: GR00T 做 high-level skill policy + RL-based low-level controller (类似 hierarchical 方案)

4. **NVIDIA Foundation Model 路线总结**

| 层级 | 组件 | 作用 |
|------|------|------|
| Scene Understanding | Cosmos World Model | Physics-aware video prediction |
| Task Planning | VLM (Cosmos-Reason / Eagle-2) | Language grounding + spatial reasoning |
| Skill Execution | GR00T N1 DiT | Action generation (manipulation + loco commands) |
| Low-level Control | COMPASS / WBC | Balance + joint-level control |

NVIDIA 的策略是逐层构建, 当前 GR00T N1 覆盖 "Skill Execution" 层, 未来通过与 Cosmos 集成扩展到 planning 和 world model 层。
