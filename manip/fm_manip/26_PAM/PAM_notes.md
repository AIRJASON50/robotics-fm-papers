# PAM Notes

> PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation
> CVPR 2026 | Tsinghua, PKU, BAAI, SJTU, Cambridge
> arXiv: 2603.22193v3

## 1. Core Problem

PAM 解决的核心问题是: **如何从仿真器的 pose 数据出发, 生成逼真的 HOI (Hand-Object Interaction, 手-物交互) 视频, 实现 sim-to-real 的视觉迁移**。

现有 HOI 生成研究被割裂为三条互不相通的路线:

| 路线 | 代表方法 | 根本局限 |
|------|----------|----------|
| Pose-Only Synthesis | GraspXL | 只生成 MANO (Mesh ANimated HandObject, 参数化手部模型) 轨迹, 无像素输出 |
| Appearance Generation | HOI-Swap | 从 mask/2D cue 幻觉外观, 没有时序动态 |
| Motion Generation | InterDyn, ManiVideo | 需要 ground-truth 首帧 + 完整 pose 序列, 无法从仿真器直接部署 |

**关键瓶颈**: 第三类方法 (motion generation) 依赖真实首帧作为输入, 但仿真器只能提供 mesh 渲染, 这构成了 sim-to-real 的"首帧瓶颈" (first-frame bottleneck)。PAM 的核心贡献就是打通这三条路线 -- 只需初始/目标 pose + 物体几何, 即可生成完整的逼真 HOI 视频。

**为什么重要**: 对于 robotics 领域, HOI 视频的高质量合成直接关系到:
- 训练数据的可扩展性 (替代昂贵的真实数据采集)
- Hand pose estimation 等下游任务的数据增强
- Sim-to-real 的闭环验证

## 2. Method Overview

PAM 采用三阶段解耦架构, 将 pose/appearance/motion 的联合建模分解为三个独立可控的子问题:

```
                          PAM Pipeline
                          
Input: h_0 (initial pose), h_T (target pose), m (object mesh), o_0 (object pose)

Stage I: Pose Generation (GraspXL, pretrained)
    (h_0, o_0, m, h_T) --> {h_t, o_t}_{t=0}^{T}   (MANO trajectory)
           |
           v
    Render conditions: depth / seg mask / hand keypoints per frame
           |
           +---> First-frame conditions (D_0, S_0, K_0)
           |               |
           |               v
           |     Stage II: Appearance Generation (Flux + ControlNet)
           |         (D_0, S_0, K_0, text) --> I_0   (realistic first frame)
           |               |
           v               v
    Stage III: Motion Generation (CogVideoX + ControlNet-style)
        ({D_t, S_t, K_t}, I_0, text) --> {I_t}_{t=0}^{T}   (video)
```

核心思路: Stage I 复用成熟的 RL-based pose 生成器 (GraspXL); Stage II 用可控图像扩散模型生成逼真首帧, 打破首帧瓶颈; Stage III 用可控视频扩散模型驱动运动生成。三个阶段共享相同的多模态条件 (depth + seg + keypoints), 确保一致性。

## 3. Key Designs

### 3.1 Multi-Modal Bridge Conditions (多模态桥接条件)

**是什么**: PAM 使用三种互补的条件信号来桥接仿真与真实:
- **Depth maps**: 提供几何信息 (由 DepthCrafter 估计)
- **Semantic segmentation masks**: 提供语义信息 (instance-level, 区分手/物体/背景)
- **Hand keypoint maps**: 提供精确手部结构信息 (21个关节点的2D投影)

**为什么有效**: 单一条件不足以约束手部生成。Seg mask 只知道"哪里是手", 不知道手指细节; depth 只知道空间深度, 不知道语义边界; keypoints 只知道关节位置, 不提供全局几何上下文。三者互补形成了从仿真到真实的完整桥梁。

**关键洞察**: Ablation (Table 3) 验证了条件组合的累积效应 -- keypoints 单独使用时 MPJPE 最低 (因为直接提供 pose 监督), 但 FVD/LPIPS 较差; 三条件组合在所有指标上达到最优平衡。

### 3.2 Decoupled Three-Stage Architecture (解耦三阶段架构)

**是什么**: 将 pose-appearance-motion 的联合问题分解为三个独立阶段, 每个阶段可以独立训练和优化。

**为什么有效**:
1. **Stage I** 复用 GraspXL (RL-based), 已有成熟的物理交互建模能力
2. **Stage II** 基于 Flux (SOTA 图像生成模型) + ControlNet, 利用其强大的外观生成先验
3. **Stage III** 基于 CogVideoX (SOTA 视频生成模型) + ControlNet-style 注入, 利用其时序建模能力

**关键洞察**: 解耦架构的最大好处不仅是降低建模难度, 更重要的是实现了 diversity -- appearance 和 motion 可以独立采样, 从同一 pose 序列生成不同外观/不同背景的视频, 这是 end-to-end 方法做不到的。

### 3.3 ControlNet-Style Condition Injection (ControlNet 风格的条件注入)

**是什么**:
- **Stage II (Appearance)**: 在 Flux DiT (Diffusion Transformer, 扩散变换器) 中使用 ControlNet 分支, 复制前 2 层 DiT blocks, 条件通过 VAE 编码为 latent 后 channel-wise 拼接, 经 zero-convolution 注入主干
- **Stage III (Motion)**: 在 CogVideoX 中复制前 12 层 DiT blocks 作为 condition 处理分支, 5 种条件 (tracking/normal/depth/seg/keypoints) 各自独立经过 patch embedding, 然后通过 `initial_combine_linear` 将 5x inner_dim 映射为 inner_dim, 再经过 transformer_blocks_copy 处理后通过 zero-initialized linear 加回主干

**为什么有效**: Zero-initialization 保证训练初期不破坏预训练模型的生成能力; 复制 transformer blocks 而非简单的 MLP 保证了条件信号与生成信号在相同表征空间中交互。

**关键洞察**: Random masking (训练时每种条件以 0.2 概率随机置零) 显著提升了鲁棒性, 使模型在推理时能容忍条件质量波动 (Table 6: 无 masking 时加噪后 FVD 从 29.13 恶化明显, 有 masking 时退化小很多)。

## 4. Experiments

### 主要定量结果 (DexYCB)

| Method | Resolution | FVD (down) | LPIPS (down) | MF (up) | MPJPE mm (down) |
|--------|-----------|------|-------|------|------------|
| InterDyn | 256x256 | 38.83 | -- | -- | -- |
| ManiVideo | 256x384 | 50.55 | 0.22 | 0.38 | -- |
| CosHand | 480x720 | 32.89 | 0.28 | 0.38 | 30.05 |
| **PAM (ours)** | **480x720** | **29.13** | **0.20** | **0.42** | **19.37** |

### OAKINK2 结果

| Method | FVD (down) | LPIPS (down) | MF (up) | MPJPE mm (down) |
|--------|------|-------|------|------------|
| CosHand | 69.89 | 0.28 | 0.33 | 42.94 |
| **PAM (full, multi-cond)** | **46.31** | **0.22** | **0.36** | **26.12** |

### Ablation: 条件组合 (DexYCB)

| Conditions | FVD | LPIPS | MF | MPJPE |
|------------|-----|-------|-----|-------|
| Seg only | 37.95 | 0.24 | 0.38 | 23.63 |
| Depth only | 44.87 | 0.28 | 0.35 | 27.40 |
| Keypoints only | 58.49 | 0.40 | 0.28 | 14.80 |
| Depth + Seg | 33.51 | 0.22 | 0.38 | 22.37 |
| Depth + Keypoints | 36.08 | 0.23 | 0.38 | 18.95 |
| Seg + Keypoints | 34.37 | 0.22 | 0.39 | 17.66 |
| **Depth + Seg + Keypoints** | **29.13** | **0.20** | **0.42** | **19.37** |

### 下游任务验证 (SimpleHand pose estimation)

| Real Data % | + Synthetic | PA-MPJPE (down) | F-Score (up) |
|-------------|-------------|-----------------|--------------|
| 100% | No | baseline | baseline |
| 50% | + PAM 3400 videos | **matches 100% baseline** | matches |
| 25% | + PAM | improved but < 100% | improved |

关键发现: 仅 50% 真实数据 + PAM 合成数据即可匹配 100% 真实数据基线, 证明了 PAM 作为数据增强工具的实用价值。

### 计算资源

| Stage | Time (per sample) | Peak GPU Memory |
|-------|-------------------|-----------------|
| I (Pose) | 0.1s | minimal |
| II (Appearance) | 55.3s | 41.4 GB |
| III (Motion) | 245.7s | 34.4 GB |
| **Total** | **~301s / 40 frames** | -- |

训练配置: 8x NVIDIA H800, batch size 4x8, lr 1e-4, AdamW, 8000 steps, DeepSpeed.

## 5. Related Work Analysis

PAM 填补的空白是三条 HOI 生成路线的**统一**:

| 维度 | GraspXL (Pose) | HOI-Swap (Appearance) | InterDyn/ManiVideo (Motion) | PAM |
|------|--------|-------------|-----------------|-----|
| 需要真实首帧 | N/A | N/A | Yes | **No** |
| 生成像素 | No | Yes (single image) | Yes (video) | **Yes (video)** |
| 时序一致性 | N/A | No | Yes | **Yes** |
| Sim-to-Real | Pose only | No | Limited | **Full pipeline** |
| 下游验证 | No | No | No | **Yes** |

**与 ControlNet 系列的关系**: PAM 的核心技术构件 -- ControlNet-style condition injection -- 并非新贡献, 来自 ControlNet (Zhang et al., 2023) 和 DiffusionAsShader。PAM 的贡献在于**将这些技术应用于 HOI 场景, 并设计了针对手部的多模态条件组合**。

**与 CosHand 的差异**: CosHand 仅使用 hand mask 作为单一条件, 且缺乏显式时序建模; PAM 使用三模态条件 + video diffusion backbone (CogVideoX 内建 temporal attention), 在 FVD 和 MPJPE 上全面超越。

**与 DiffusionAsShader 的关系**: Motion generation stage 的模型架构 (`CogVideoXTransformer3DModelCombination`) 直接复用了 DiffusionAsShader 的设计模式 (从代码中的变量命名和 load_state_dict fallback 逻辑可以看出), 但扩展到 5 种条件输入并增加了随机 masking 机制。

## 6. Limitations & Future Directions

### 局限性

1. **Error Propagation (误差传播)**: 三阶段解耦架构的代价 -- Stage I 的几何误差 (如手指穿透物体) 会传播到后续阶段, 导致即使视频"看起来逼真"也可能物理不合理 (Section 7.7)。Stage II 低质量首帧同样会退化 Stage III 的纹理和时序稳定性。

2. **Stage I 依赖**: 完全依赖 GraspXL 作为 pose 生成器。GraspXL 本身的泛化能力 (新物体、复杂操作) 直接限制了 PAM 的适用范围。Ablation (Table 7) 显示换用 D-Grasp 后性能明显下降。

3. **计算开销**: 单个视频 (40帧) 需要 ~5 分钟推理, 其中 Stage III 占 82% 时间。大规模数据合成的成本不可忽视。

4. **单手/简单操作**: 虽然在 OAKINK2 (bimanual) 上做了 zero-shot 实验, 但主要训练和评估集中在单手抓取。对于灵巧操作 (如工具使用、双手协作) 的覆盖有限。

5. **外观多样性的评估不足**: PAM 声称能生成多样外观, 但缺乏对外观多样性的定量评估 (如 FID diversity)。

### 未来方向

- **End-to-End 统一**: 论文自己提到可以将 motion 和 appearance 阶段统一为端到端模型, 减少误差传播
- **更复杂交互**: 扩展到双手、工具使用、精细操作 (如按钮、旋转)
- **与 RL 的闭环**: PAM 生成的视频可以作为 world model 的训练数据, 反哺 RL policy 学习
- **条件扩展**: 加入 tactile/force 条件, 使生成视频包含接触力学信息

## 7. Paper vs Code Discrepancies

### 7.1 条件数量: 3 vs 5

**论文**: 明确声称使用 3 种条件 -- depth, segmentation, hand keypoints (Section 3.3-3.4, "combining depth, segmentation, and keypoints")。训练脚本中 `--used_conditions hand_keypoints,depth,seg_mask` 也是 3 种。

**代码**: `CogVideoXTransformer3DModelCombination` 的 forward 方法接受 **5 种条件**: tracking_maps, normal_maps, depth_maps, seg_masks, hand_keypoints。`initial_combine_linear` 的输入维度是 `inner_dim*5`, 硬编码为 5 种条件的拼接。未使用的条件 (tracking, normal) 在数据加载时被置零 (`torch.zeros_like`), 但仍然参与计算。

**影响**: 模型架构比论文描述的更"重" -- 即使只用 3 种有效条件, 模型仍然为 5 种条件分配了参数和计算。这意味着模型架构是为未来扩展预留的, 但当前浪费了约 40% 的 condition 处理开销。

### 7.2 ControlNet 层数

**论文 (Stage II)**: "Injected into two layers of DiT blocks" (Section 3.3)。
**代码 (Stage II)**: `ControlNetFlux.__init__` 中 `controlnet_depth=2`, 一致。

**论文 (Stage III)**: "12 duplicate DiT blocks" (Section 3.4)。
**代码 (Stage III)**: `num_tracking_blocks` 默认值是 18, 但训练脚本 `--num_tracking_blocks 12`, 一致 (但需要注意默认值不一致)。

### 7.3 Random Masking 概率

**论文**: "each cue is randomly masked with a probability of 0.2" (Section 3.4)。

**代码**: 不同条件使用不同的 masking 概率:
- tracking: `random.random() < 0.8` (保留概率 0.8, 即 drop 0.2)
- normal: `random.random() < 0.7` (保留概率 0.7, 即 drop 0.3)
- depth: `random.random() < 0.8` (drop 0.2)
- seg_mask: `random.random() < 0.8` (drop 0.2)
- hand_keypoints: `random.random() < 0.8` (drop 0.2)

此外, 在 RGB 数据加载路径中 (`_preprocess_video`), seg_mask 和 hand_keypoints 还有额外的 `random.random() > 0.8` 随机置零 (即额外 20% 的 drop)。Normal maps 的处理甚至被 `and False` 硬编码禁用。

**影响**: 实际 masking 策略比论文描述更复杂, normal condition 在 RGB 路径中完全被禁用, 与论文声称的统一 0.2 概率不符。

### 7.4 Condition Encoder

**论文**: "All cues are VAE-encoded to H/8 x W/8 x 16 latents, concatenated channel-wise" (Section 3.3, 指 Stage II)。

**代码 (Stage II)**: 实际使用一个 8 层的 CNN (`input_hint_block`) 将条件从像素空间下采样到 latent 空间, 而非 VAE。这个 CNN 由 Conv2d(condition_in_channels, 16) 开始, 通过 3 次 stride=2 下采样到 1/8 分辨率, 最后一层是 zero-initialized conv。

**代码 (Stage III)**: 条件通过预训练的 video VAE 编码 (在 `prepare_dataset` 脚本中离线完成), 然后通过 `self.patch_embed` 注入。

**影响**: Stage II 的条件编码方式与论文描述不完全一致 -- 是 CNN 而非 VAE 编码。

### 7.5 DoubleControl 架构

**代码**: `appearance_gen/src/flux/model.py` 中定义了 `DoubleControl` 类, 包含两个独立的 `ControlNetFlux` 实例 (`controlnet1` 和 `controlnet2`)。推理脚本 `double_infer.py` 和 `double_infer_sim.py` 暗示 Stage II 实际上使用双 ControlNet 架构。

**论文**: 未明确提及双 ControlNet 设计, 只描述了单一 ControlNet 分支。可能一个处理 depth+seg, 另一个处理 keypoints, 但论文未做说明。

## 8. Cross-Paper Comparison

| 维度 | PAM | UltraDexGrasp | UniSim | DreamerV3 | DexGraspVLA | UniDex |
|------|-----|---------------|--------|-----------|-------------|--------|
| **核心目标** | Sim-to-real HOI 视频生成 | 大规模灵巧抓取 (RL policy) | 通用 world model | 通用 world model (RL) | 灵巧抓取 VLA | 通用灵巧操作 |
| **生成什么** | HOI 视频 (像素) | 抓取 policy (actions) | 视频/场景模拟 | 潜空间预测 | Actions | Actions |
| **Foundation Model** | Flux + CogVideoX | -- | Video diffusion | RSSM + CNN | VLM backbone | -- |
| **合成数据角色** | 核心输出 (生成合成训练数据) | 训练数据来源 (Isaac Gym) | 训练信号 | 世界模型本身 | 可作为数据消费者 | 可作为数据消费者 |
| **手部建模** | MANO + 2D keypoints | 关节角控制 | 不特化 | 不特化 | 视觉-语言表征 | 关节角控制 |
| **Sim-to-Real 策略** | 仿真 pose --> 真实视频 | Domain Randomization | 学习真实动态 | 学习潜空间动态 | 真实数据微调 | Sim2Real transfer |
| **可扩展性** | 受限于 pose 生成器 | 大规模 (10k+ 物体) | 受限于训练数据 | 高效 (latent space) | 受限于真实数据 | 中等 |
| **与 PAM 的互补** | -- | 可提供更多 pose 数据给 PAM | PAM 可提供训练视频 | PAM 生成视频可辅助 world model 学习 | PAM 生成数据可增强 VLA 训练 | PAM 可提供 sim2real 视觉增强 |

### 关键对比分析

**PAM vs UltraDexGrasp**: 两者都利用仿真数据, 但方向互补 -- UltraDexGrasp 关注如何在仿真中训练通用抓取 policy (RL 方向), PAM 关注如何将仿真交互转化为逼真视频 (generative 方向)。PAM 可以作为 UltraDexGrasp 的下游: 利用 UltraDexGrasp 的大规模 pose 数据生成逼真视频, 进一步用于视觉模型训练。但 PAM 目前只用了 GraspXL 作为 pose source, 未验证与 UltraDexGrasp 的适配。

**PAM vs UniSim/DreamerV3 (World Models)**: World model 的目标是学习环境动态用于 planning/RL, PAM 的目标是生成逼真训练数据。但两者有交叉: PAM 本质上是一个条件化的 world model (给定 pose 预测视频), 只是不做闭环 planning。PAM 生成的大量 HOI 视频可以作为 world model 的训练数据, 尤其是 UniSim 这类需要大量视频数据的方法。反过来, world model 的 action-conditioned prediction 能力可以替代 PAM 的 Stage I + Stage III。

**PAM vs DexGraspVLA/UniDex (VLA)**: VLA (Vision-Language-Action, 视觉-语言-动作) 方法是 PAM 最直接的数据消费者。PAM 生成的带 pose annotation 的合成视频可以直接用于 VLA 的预训练/数据增强。论文在 SimpleHand 上验证了这一点 (pose estimation), 但尚未在 action prediction 任务上验证, 这是一个重要的未来方向。
