# DexGraspVLA Notes

DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping (arXiv 2025)
PKU (Yaodong Yang, Yuanpei Chen 组)

---

## 1. Core Problem

灵巧手抓取的**泛化性**问题。现有方法在三个维度上泛化不足:

1. **物体维度**: 大多数方法假设单物体或简单场景，无法处理杂乱场景中数百种未见物体 (不同几何、纹理、材料、重量)
2. **环境维度**: 光照变化、背景变化导致策略失效 -- 因为 IL (Imitation Learning, 模仿学习) 直接从原始像素学习，domain shift 是致命的
3. **任务维度**: 缺乏语言引导的长序列任务执行能力 (如 "清空桌面")

**核心 insight**: 泛化的瓶颈不在于策略架构本身，而在于输入的 domain variance。如果能把多变的视觉/语言输入转化为 domain-invariant (域不变) 的表征，IL 就可以在小数据集上高效学习并泛化。

**为什么重要**: 这是第一个在真实灵巧手 (6-DoF 手 + 7-DoF 臂) 上，用 ~2000 条 demo 实现 1287 种未见条件组合下 90%+ 抓取成功率的工作。同时首次在灵巧手上同时展示了 long-horizon 推理、失败恢复、人类干扰鲁棒性和 nonprehensile grasping 扩展。

---

## 2. Method Overview

### 架构: 分层式 Planner-Controller

```
User Prompt: "Clear the table"
       |
       v
+-------------------------------------------+
|  HIGH-LEVEL PLANNER (frozen VLM)          |
|  Qwen2.5-VL-72B / Qwen-VL-Chat           |
|                                            |
|  4 sub-tasks:                              |
|  1. Instruction Proposal                   |
|     "grasp the cookie" <- prompt + scene   |
|  2. Bounding Box Prediction                |
|     [x1,y1,x2,y2] <- locate object        |
|  3. Grasp Outcome Verification             |
|     True/False <- check success            |
|  4. Prompt Completion Check                |
|     True/False <- all done?                |
+-------------------------------------------+
       |  bbox (x1,y1,x2,y2)
       v
+-------------------------------------------+
|  LOW-LEVEL CONTROLLER (learned)            |
|                                            |
|  bbox -> SAM -> initial mask m_0           |
|           -> Cutie -> tracked mask m_t     |
|                                            |
|  Observation Encoding:                     |
|  - Head cam I_h -> DINOv2-B/14 (frozen)   |
|    -> z_h (1369x768)                       |
|  - Mask m_t -> trainable ViT              |
|    -> z_m (1369x768)                       |
|  - [z_h; z_m] -> MLP -> z_h~ (1369x1024) |
|  - Wrist cam I_w -> DINOv2-L/14 (frozen)  |
|    -> z_w -> MLP -> z_w~ (1369x1024)      |
|  - State s_t (13D) -> MLP -> z_s~ (1x1024)|
|                                            |
|  z_obs = [z_h~; z_w~; z_s~] (2739x1024)  |
|                                            |
|  Action Head: DiT (Diffusion Transformer)  |
|  - 12 layers, 8 heads, dim=1024           |
|  - Cross-attn: action tokens -> z_obs      |
|  - Self-attn: action tokens bidir          |
|  - DDIM 16 steps inference                 |
|  - Action chunk H=64, execute first 6     |
+-------------------------------------------+
       |  action a_t = (arm_7D, hand_6D)
       v
     Robot Execution @ 20Hz
```

### 核心设计哲学

逐层 domain variance 消除:
- **语言层**: 多变的自由文本 prompt -> VLM -> 统一格式的 bbox (domain-invariant affordance)
- **视觉层**: 多变的原始图像 -> frozen DINOv2 -> 语义一致的 feature (domain-invariant representation)
- **目标层**: bbox -> SAM + Cutie -> 精确的 binary mask (domain-invariant target specification)

在这些 domain-invariant 表征之上做 IL，domain shift 被大幅缓解。

---

## 3. Key Designs

### 3.1 Domain-Invariant Representation Pipeline (核心贡献)

**是什么**: 不是端到端训练一个大 VLA 模型 (如 pi_0, OpenVLA)，而是用 frozen foundation models 作为 feature extractor，把原始输入映射到 domain-invariant 空间，然后只训练 action head。

**为什么有效**:
- DINOv2 在 internet-scale 数据上预训练，其 patch features 对光照、背景、视角具有内在鲁棒性 -- 论文 Figure 4 用 PCA (Principal Component Analysis, 主成分分析) 可视化证实了同一场景在白色背景/校准板/花布/迪斯科灯下的 DINOv2 features 高度一致
- Frozen DINOv2 + trainable MLP projection 的方式避免了 catastrophic forgetting，而端到端微调 VLM (如 pi_0 full FT) 反而可能破坏预训练表征
- 论文 ablation 证明: 解冻 DINOv2 (DINOv2-train) 性能显著下降，说明保持 frozen 是关键

**insight**: Foundation model 的价值不在于端到端微调，而在于提供 frozen, domain-invariant feature space 作为 IL 的 "common ground"。这与 R3M/VIP 的思路一脉相承，但 DexGraspVLA 更进一步，把这个原则贯穿到整个 planner-controller pipeline。

### 3.2 Mask-Guided Target Specification

**是什么**: 用 SAM (Segment Anything Model) + Cutie (video object segmentation tracker) 将 planner 输出的 bbox 转化为逐帧跟踪的 binary mask m_t，作为目标物体的指示信号。mask 通过一个 trainable ViT 编码后与 DINOv2 head image features patch-wise 拼接。

**为什么有效**:
- Bbox/mask 是比语言更 "干净" 的 affordance signal -- 它直接告诉 controller "操作什么"，无需 controller 理解语言
- Cutie 提供闭环的目标跟踪，即使物体在抓取过程中移动/旋转/被遮挡，mask 也能持续更新
- 这个设计使得 controller 完全不需要语言输入，解耦了语言理解 (planner) 和动作生成 (controller)

**insight**: 在 VLA 框架中，语言并非必须直接输入 action head。Planner 负责理解语言并生成 grounded affordance (bbox/mask)，Controller 只需要理解 "在哪里" 和 "怎么做"。这种模块化分工比端到端训练更高效，也更容易调试和升级。

### 3.3 DiT-based Diffusion Action Head with Immiscible Diffusion

**是什么**: 使用 DiT (Diffusion Transformer) 替代 UNet 作为 diffusion policy 的 backbone。Action chunk 长度 H=64 (13D action, 即 7 arm + 6 hand joints)。训练时使用 Immiscible Diffusion (不互溶扩散) 技术优化 data-noise mapping。

**为什么有效**:
- DiT 的 cross-attention 机制天然适合处理长序列条件 (2740 tokens = 2739 obs + 1 timestep)，比 UNet 更好地捕获多模态条件信息
- Immiscible Diffusion: 通过 Hungarian matching 将每个 data sample 分配给 "最近" 的 noise sample，而非随机配对。这减少了 data-noise 映射的歧义性，提高训练效率
- Cross-attention mask 训练策略: 以 70% 概率不 mask，10% 概率分别 mask head/wrist/both 部分的条件 token，增强鲁棒性

**insight**: Action chunk + receding horizon control (执行前 6 步就重新预测) 在响应速度和动作一致性之间取得平衡。163M 参数的 controller 比 pi_0 (3B+) 轻量得多，但通过专注于 domain-invariant 表征上的动作建模，取得更好的泛化。

---

## 4. Experiments

### 4.1 硬件平台

- 7-DoF RealMan RM75-6F 机械臂 + 6-DoF PsiBot G0-R 灵巧手
- 头部 RealSense D435 相机 (第三人称视角) + 腕部 RealSense D405C 相机 (第一人称视角)
- 控制频率 20Hz
- 训练数据: 2,094 条 demo，36 种日常物体，杂乱场景，每条约 3.5 秒

### 4.2 大规模泛化评测 (核心实验)

| 条件 | 测试数 | Ours@1 | Ours@2 | Ours@3 |
|------|--------|--------|--------|--------|
| 360 unseen objects | 360 | 91.1% | - | - |
| 6 unseen backgrounds | 618 | 90.5% | - | - |
| 3 unseen lightings | 309 | 90.9% | - | - |
| **Aggregated** | **1287** | **90.8%** | **94.9%** | **96.9%** |

关键: 测试在完全不同的房间进行 (zero-shot environment)。

### 4.3 Baseline 对比

| 方法 | Seen Objects | Unseen Objects | Unseen Bgs. | Unseen Lightings | Aggr. |
|------|-------------|---------------|-------------|-----------------|-------|
| DexGraspVLA (Ours) | 95.8% | 89.6% | 91.7% | 91.7% | 91.5% |
| pi_0 (Full FT) | 75.0% | 25.0% | 8.3% | 6.3% | 19.8% |
| pi_0 (LoRA) | 20.8% | 8.3% | 4.2% | 0% | 6.3% |
| RDT (Full FT) | 37.5% | 22.9% | 25.0% | 22.9% | 25.0% |
| OpenVLA (LoRA) | 29.2% | 4.2% | 0% | 0% | 4.2% |
| OpenVLA-OFT (LoRA) | 8.3% | 6.3% | 4.2% | 2.1% | 5.2% |

pi_0 full FT 在 seen objects 上 75% 但 unseen 下断崖式下降，说明端到端 VLA 微调严重过拟合。

### 4.4 Ablation

| 方法 | Seen (130 tests) | Unseen (80 tests) |
|------|----------|---------|
| DexGraspVLA (Ours) | 98.5% | 98.8% |
| DINOv2-train (解冻 DINOv2) | 50.0% | 45.0% |
| ViT-small (替换为小 ViT) | 43.1% | 47.5% |

Frozen DINOv2 vs trainable: 98.5% vs 50.0%，差距巨大。

### 4.5 Long-Horizon 任务

| Prompt | Task SR | Avg Attempts/Obj | Instr. Proposal | BBox Pred. | Grasp Exec. | Completion Check |
|--------|---------|------------------|----------------|------------|-------------|-----------------|
| Clear the table | 83.3% | 1.08 | 91.7% | 100% | 92.7% | 100% |
| Grasp all bottles | 95.8% | 1.04 | 100% | 100% | 97.8% | 91.7% |
| Grasp all green | 87.5% | 1.09 | 93.8% | 95.8% | 91.3% | 91.7% |
| Grasp all food | 91.7% | 1.06 | 91.7% | 97.9% | 84.1% | 93.8% |
| **Aggregated** | **89.6%** | **1.07** | **94.3%** | **98.4%** | **91.5%** | **94.3%** |

每个物体平均只需 1.07 次尝试。Planner 的 bbox prediction 准确率高达 98.4%。

### 4.6 Nonprehensile Grasping 扩展

| 条件 | Ours | DINOv2-train | ViT-small |
|------|------|--------------|-----------|
| Unseen Objects | 88.9% | 66.7% | 50.0% |
| Unseen Lighting | 83.3% | 61.1% | 44.4% |
| Unseen Backgrounds | 83.3% | 55.6% | 31.9% |
| **Aggregated** | **84.7%** | **59.7%** | **38.9%** |

用 1029 条 demo 训练的 nonprehensile 策略，无需改架构。

---

## 5. Related Work Analysis

### 定位

DexGraspVLA 处于以下研究线的交汇处:

1. **Dexterous Grasping**: 相比 two-stage (grasp pose + motion planning) 方法，DexGraspVLA 是端到端闭环控制；相比 RL-based (UniDexGrasp++, GraspXL)，它避免了 sim-to-real gap
2. **Foundation Model for Robotics**: 相比 OpenVLA/pi_0 的端到端 VLA 微调，DexGraspVLA 用 frozen FM 做 feature extraction + 轻量 action head，数据效率更高、泛化更强
3. **Modular VLA Frameworks**: 类似 VoxPoser/ReKep 的模块化思路 (FM for affordance + learned controller)，但 DexGraspVLA 的 controller 是闭环的且在 domain-invariant 表征上学习

### 填补的 gap

现有工作在 dexterous grasping 上要么依赖仿真 (RL)、要么依赖精确标定 (two-stage)、要么需要海量数据 (端到端 VLA)。DexGraspVLA 用模块化的 frozen FM + 轻量 IL 方案，首次在真机上以 ~2000 条 demo 实现了大规模泛化的灵巧手抓取。

---

## 6. Limitations & Future Directions

### 论文自述局限

1. **未涉及功能性抓取** (functional grasping): 当前只考虑 "抓起来"，不考虑抓取方式对后续操作的影响 (如: 抓杯子用手柄 vs 抓杯壁)
2. **未集成触觉感知**: 仅依赖视觉，缺少触觉反馈来判断抓取力度和稳定性

### 更深层的局限

3. **硬件特异性**: 6-DoF 手的灵活度有限 (相比 16-24 DoF 的 Allegro/Shadow)，论文的 "dexterous" 在自由度上并不极致。PsiBot G0-R 是一个相对简单的手
4. **Planner 延迟**: VLM (72B 参数) 推理延迟较大，long-horizon 场景中每步都要调用多次 VLM，实时性是问题
5. **Action space 简单**: 13D (7 arm + 6 hand)，action chunk = 64 步，每步只执行 6 步。这对于复杂的 in-hand manipulation 可能不够
6. **数据采集方式**: Kinesthetic teaching (人手直接带动机器人) 的采集效率较低，且受限于操作者的技能水平
7. **缺乏 3D 信息**: 完全依赖 2D 视觉特征 (DINOv2)，不使用 depth 或 point cloud，可能在需要精确深度估计的场景 (如薄平物体、透明物体) 有局限
8. **Planner-Controller 之间无梯度流**: 这意味着如果 planner 给出错误的 bbox，controller 无法自我纠正。系统的鲁棒性上限受限于 VLM 的定位精度

### Future Directions

- 集成触觉: 用力/触觉传感器做 closed-loop force control
- Fine-grained affordance: planner 生成更丰富的 affordance (如抓取方向、手型提示)
- 跨手迁移: 结合 DexLatent 的 cross-hand latent space，实现不同灵巧手的策略复用
- 规模化数据: 结合 UltraDexGrasp 的合成数据方案，减少对人类演示的依赖

---

## 7. Paper vs Code Discrepancies

### 总体一致性

代码与论文描述高度一致，是少见的论文-代码对齐良好的工作。主要发现:

### 具体对比

| 方面 | 论文描述 | 代码实现 | 差异程度 |
|------|---------|---------|---------|
| DiT 层数 | "based on DiT, Diffusion Policy, RDT" | `RDTBlock`: self-attn + cross-attn + FFN，用 RmsNorm，命名为 RDTBlock | 一致 -- 论文未详述 block 内部，代码显示直接参考 RDT 的 block 设计 |
| 配置 n_layer | 论文附录未明确 | YAML 默认 `n_layer: 12`，`n_head: 8` | 一致 |
| DINOv2 模型 | head: ViT-B/14, wrist: ViT-L/14 | `model_type: dinov2_vitb14` / `dinov2_vitl14` | 一致 |
| Immiscible Diffusion | "employ Immiscible Diffusion to improve data-noise mapping" | `noise_assignment()` 用 Hungarian matching (scipy.linear_sum_assignment) | 一致 |
| Mask encoder | "randomly initialized ViT" | `mask_process_net`: Conv2d patchify + 4-layer TransformerEncoder | 基本一致 -- 论文说 "ViT"，代码用 Conv2d patch embedding + TransformerEncoder，功能等价 |
| DDIM scheduler | DDIM sampling for inference | Config: `diffusers.DDIMScheduler`, `num_train_timesteps: 50`, `num_inference_steps: 16` | 一致 |
| Action chunk | H=64, execute first 6 | `n_action_steps: 64`，inference 中执行步数由配置控制 | 一致 |
| Normalizer | "compute MSE loss" | `LinearNormalizer` 对 action 做归一化后再加噪/预测 | 一致 -- 论文未提 normalizer 但这是标准做法 |
| Observation normalization | 论文未提 | 代码中 `obs` 不经过 normalizer (`nobs = obs_dict`)，只有 action 被 normalize | 值得注意 |
| Cross-attention mask | 论文未提 | `use_attn_mask: False` (默认关闭)；但代码中实现了 4 种 mask 策略 (mask head/wrist/both/none, prob [0.1,0.1,0.1,0.7]) | 有差异 -- 代码有完整实现但默认禁用，论文未讨论 |
| EMA | 论文未提 | 配置 `use_ema: False`，代码中有完整 EMAModel 实现 | 有差异 -- 论文未提，默认关闭 |
| BGR to RGB | 论文未提 | `obs_encoder.py` 中有 `rgb_data[:, :, [2, 1, 0]]` 的通道翻转 | 实现细节 -- 因为相机输出 BGR |
| Color jitter | "apply domain randomization via color jittering" | `brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1` | 一致 |
| Planner | Qwen2.5-VL-72B | 代码通过 OpenAI-compatible API 调用，model_name 可配置 | 一致 |
| Gradient clipping | 论文未提 | `clip_grad_norm_(max_norm=0.5)` | 实现细节 |
| Mixed precision | bfloat16 | Accelerate `mixed_precision='bf16'` | 一致 |

### 重要发现

1. **Cross-attention mask 策略**: 代码中实现了一个有趣的训练时 attention mask 随机策略 (类似 dropout for condition)，但在默认配置中被关闭。这可能是一个尝试过但最终未采用的设计，或用于特定场景
2. **EMA 未使用**: 代码完整实现了 EMA (Exponential Moving Average, 指数移动平均) 但默认关闭，可能是因为训练数据量较小，EMA 对性能没有帮助
3. **观测不归一化**: 只有 action 被 normalize，observation features 直接传入 DiT

---

## 8. Cross-Paper Comparison

### 与 fm_manip 领域其他工作的对比

| 维度 | DexGraspVLA (2025) | UltraDexGrasp (2026) | DexLatent/XL-VLA (2026) | UniDex (2026) |
|------|-------------------|---------------------|----------------------|--------------|
| **核心问题** | Real-world 灵巧手抓取泛化 | 大规模合成数据驱动的灵巧抓取 | 跨手形态统一 action space | 人类视频到机器人灵巧操作 |
| **数据来源** | 真机 kinesthetic teaching, 2094 demo | 纯合成 (BODex + cuRobo), 20M 帧 | 真机遥操作 (多种手) | 人类 egocentric 视频转化, 50K+ 轨迹 |
| **手的复杂度** | 6-DoF PsiBot G0-R | 6-DoF XHand | 多种手 (Allegro, LEAP, Ability 等) | 8 种手 (6-24 DoF) |
| **视觉表征** | Frozen DINOv2 (ViT-B/14 + ViT-L/14) | Point cloud (PointNet++) | SigLIP (via pi_0) | 3D point cloud |
| **Action 空间** | 13D (7 arm + 6 hand) | ~19D (6 arm + 12 hand + 1 gripper) | Latent space (32D VAE) | FAAS (统一跨手) |
| **策略类型** | Diffusion (DiT) | BC (truncated normal) | Flow matching (pi_0) | VLA (基于 DP3) |
| **Planner** | VLM (Qwen2.5-VL-72B) | 无 (single step) | pi_0 内置 | 语言条件 VLA |
| **Sim/Real** | Real only | Sim->Real (zero-shot) | Real only | Sim+Real (human video) |
| **抓取成功率** | 90.8% (1287 unseen 组合) | 87% (100 unseen objects) | ~70% (cross-hand) | 81% (tool-use tasks) |
| **关键创新** | Frozen FM -> domain-invariant repr | 纯合成数据 pipeline (无 RL) | Cross-hand latent space (VAE) | FAAS action space + human video |
| **训练规模** | 163M 参数, <1 day, 8xA800 | 大规模 BC | 基于 pi_0 (3B+) | 基于 DP3 |

### 与经典 Policy Learning 方法的对比

| 维度 | DexGraspVLA | Diffusion Policy (2023) | ACT (2023) | pi_0 (2024) |
|------|-------------|------------------------|------------|-------------|
| **架构** | Frozen DINOv2 + DiT | ResNet/ViT + UNet | CVAE + Transformer | PaliGemma + Flow Matching |
| **Action 建模** | DDIM diffusion | DDPM diffusion | CVAE | Flow matching |
| **视觉编码** | Frozen DINOv2 | Trainable ResNet/ViT | Trainable ResNet | Trainable SigLIP |
| **语言条件** | 通过 VLM planner 间接 | 无 | 无 | 直接端到端 |
| **目标指定** | Mask (via SAM+Cutie) | 无/goal image | 无/goal image | Language |
| **泛化策略** | Domain-invariant features | 数据多样性 | 数据多样性 | 大规模预训练 |
| **数据效率** | 高 (~2K demo) | 中 | 中 | 低 (需要大量数据) |
| **参数量** | 163M (controller) | ~100M | ~80M | 3B+ |
| **灵巧手支持** | 原生 | 需适配 | 需适配 | 需适配 |

### 核心 Takeaway

1. **Frozen FM > End-to-end FT**: DexGraspVLA 最重要的实验结论是 -- 在数据有限的真机灵巧手场景，frozen foundation model 提取 domain-invariant feature + 轻量 action head 的方案，远优于端到端微调大型 VLA (pi_0, OpenVLA)。这与 robotics 社区越来越大的模型、越来越多的数据的趋势形成有趣的对比

2. **模块化 vs 端到端的 trade-off**: DexGraspVLA 的模块化 (VLM planner + FM feature extractor + diffusion controller) 牺牲了端到端优化的可能性，但换来了更好的可解释性 (attention map 可视化)、更强的泛化性、和更低的数据需求

3. **与 UltraDexGrasp 互补**: UltraDexGrasp 用合成数据解决数据量问题，DexGraspVLA 用 domain-invariant 表征解决泛化问题。两者可以结合: 用合成数据生成大量 demo，再在 DINOv2 feature space 上训练 diffusion policy

4. **与 DexLatent 正交**: DexLatent 解决的是跨手形态迁移 (同一个 latent space 适配不同手)，DexGraspVLA 解决的是单一手形态下的跨场景泛化。两者的 insight 互不冲突，理论上可以组合

5. **对机器人 FM 路线的启示**: DexGraspVLA 表明，至少在灵巧手抓取这个任务上，"小模型 + 好表征" 优于 "大模型 + 端到端"。这可能是因为灵巧手的 action space 和 dynamics 与 VLM 预训练时见过的 "常识" 关联较弱，端到端微调很难有效迁移 VLM 的知识到 action 生成
