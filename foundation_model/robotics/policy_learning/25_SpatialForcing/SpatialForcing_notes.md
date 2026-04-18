# Spatial Forcing Notes

## 1. Core Problem

VLA (Vision-Language-Action, 视觉-语言-动作) 模型是当前机器人操作的主流范式, 但其核心矛盾在于: VLA 的视觉骨干 VLM (Vision-Language Model, 视觉-语言模型) 仅在 2D 图像-文本数据上预训练, 缺乏对 3D 物理世界的空间感知能力. 这直接导致了动作生成的空间精度不足.

现有的 3D 增强方案面临三个根本性障碍:

1. **传感器数据质量问题**: 深度相机和激光雷达获取的深度图/点云质量低、噪声大
2. **硬件异构性**: 不同机器人平台的传感器类型、安装位置、标定状态各异, 数据格式难以统一
3. **数据可扩展性**: 大规模机器人数据集 (如 Open X-Embodiment) 中相当一部分不包含深度信息, 限制了 3D VLA 的数据规模
4. **深度估计器上限**: 用深度估计器从 2D 图像推测 3D 信息的方案, 其性能天花板受限于估计器本身的精度

本文的核心问题是: 能否在不依赖显式 3D 传感器输入或深度估计器的前提下, 隐式地赋予 VLA 模型 3D 空间理解能力?

这个问题重要的原因在于: 如果能解决, 就意味着可以用纯 2D RGB 数据训练出具备 3D 空间感知的 VLA, 既绕开了硬件限制, 又能充分利用现有的大规模 2D 机器人数据集.

## 2. Method Overview

Spatial Forcing (SF) 的核心思想简洁而有效: 在训练阶段, 将 VLA 中间层的 visual embeddings 与预训练 3D 基础模型 VGGT (Visual Geometry Grounded Transformer, 视觉几何基础 Transformer) 产生的空间表征进行对齐. 推理阶段不需要 VGGT, VLA 的推理流程完全不变, 零额外开销.

```
Training Pipeline:
                                                          
  Multi-view RGB ──┬──> VLA backbone ──> Layer K ──> visual tokens x_V
                   |                                    |
                   |                          [BatchNorm + 2-layer MLP]
                   |                                    |
                   |                              projected x_V
                   |                                    |
                   └──> VGGT (frozen) ──> Layer L ──> spatial repr f_3D + PE
                                                        |
                                                  Cosine Similarity
                                                        |
                                                   L_align loss
                                                        
  Total Loss = L_action + alpha * L_align

Inference Pipeline (identical to vanilla VLA):

  RGB ──> VLA backbone ──> action tokens ──> actions
  (No VGGT needed, no extra cost)
```

关键设计选择:
- 对齐目标: VGGT 中间层特征 (不是最终预测输出, 而是 transformer backbone 的 latent representation)
- 对齐位置: VLA 的较深但非最深层 (OpenVLA-OFT 的第 24/33 层, pi_0 的第 12/18 层)
- 对齐方式: cosine similarity loss, 加权系数 alpha = 0.5
- 投影器: BatchNorm (论文) / LayerNorm (代码) + 两层 MLP (GELU 激活), 将 VLA 维度映射到 2x VGGT 维度

## 3. Key Designs

### 3.1 Depth Probing 验证: 暴露 VLA 的空间盲区

**是什么**: 冻结已训练的 VLA 全部参数, 仅训练一个 DPT (Dense Prediction Transformer) head 来从 visual embeddings 预测深度图. 这是一种 linear probing 的变体, 用于量化 VLA 表征空间中隐含的 3D 信息.

**为什么有效**: 这个实验的精妙之处在于, 它提供了一个定量的诊断工具. 如果 VLA 的 visual embeddings 确实编码了丰富的空间信息, 那么一个轻量的 probe 应该能从中恢复出有意义的深度结构. 实验结果表明原始 VLA 的 embeddings 无法产生有意义的空间结构, 而 SF 对齐后的 embeddings 则能恢复出清晰的深度图.

**核心洞察**: VLA 在标准训练中学到的 visual embeddings 几乎不包含 3D 结构信息, 这不是因为信息"丢失了", 而是因为从未有过显式的监督信号引导模型去编码空间信息. SF 正是提供了这个缺失的监督信号.

### 3.2 中间层对齐策略 (Intermediate Layer Alignment)

**是什么**: 不在 VLA 的最浅层或最深层做对齐, 而是选择"较深但非最深"的中间层 (OpenVLA-7B 的第 24/32 层, pi_0 PaliGemma 的第 12/18 层).

**为什么有效**: 这里有一个精妙的权衡:
- **太浅**: 对齐浅层特征后, 后续层可能逐渐"忘记"空间信息, 导致传到 action tokens 时空间信号衰减
- **太深**: VLA 的最深层中, 视觉和语言模态趋向于收敛到模态无关 (modality-agnostic) 的共享空间, 视觉特异性特征已经大量丢失, 此时再施加视觉空间监督信号效果不佳
- **适中的深层**: 对齐较深层会隐式地迫使浅层也更好地编码空间信息 (因为梯度反传), 同时保留了足够的视觉特异性特征来接受空间监督

**核心洞察**: 这与 REPA (REPresentation Alignment) 在 diffusion model 中的发现一致 -- 中间层对齐是一种全局性优化策略, 通过"自上而下"的梯度传播, 让整个网络的表征都向空间感知的方向演化. 这也呼应了 Huang et al. (2024) 关于 VLM 中跨模态对齐率 (Modality Integration Rate) 的结论.

### 3.3 VGGT 作为 3D 监督信号源

**是什么**: 选用 VGGT 而非 DINOv2 或 SigLIP 作为目标表征. VGGT 是一个 feed-forward 3D 基础模型, 通过 Alternating-Attention 机制 (交替的帧内 self-attention 和全局 self-attention) 从多视角 2D 图像直接输出 3D 属性. SF 使用的是 VGGT backbone 的 latent representation, 而非其最终的 3D 预测头输出.

**为什么有效**: Ablation 实验证明了目标表征的选择至关重要:
- 所有类型的对齐目标 (SigLIP, DINOv2, VGGT) 都能提升性能, 说明"表征对齐"本身是一个通用有效的范式
- VGGT 效果最好, 因为它在 2D-3D 配对数据上训练, 其 latent representation 天然编码了丰富的几何先验
- 对 target representation 添加 PE (Positional Embedding, 位置编码) 在长视野 (long-horizon) 任务上效果显著, 因为 VLA 的 auto-regressive 机制对 token 的相对位置敏感

**核心洞察**: 监督信号不需要是"精确的 3D 几何", 而是需要编码了"空间结构意识"的表征. VGGT 的优势不在于它能精确预测深度, 而在于其 backbone 特征天然具备多视角一致的空间理解能力.

## 4. Experiments

### 4.1 主要结果

**LIBERO Benchmark (仿真)**

SF 基于 OpenVLA-OFT 和 pi_0 两个基模型进行了实验, 每个任务评估 500 次:

| Method | 3D Input | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------|----------------|---------------|-------------|-------------|---------|
| OpenVLA | No | 84.6 | 88.4 | 79.2 | 53.2 | 76.4 |
| OpenVLA-OFT | No | 97.2 | 93.0 | 91.6 | 67.0 | 87.2 |
| GeoVLA | Yes (depth) | -- | -- | -- | -- | ~90+ |
| 3D-CAVLA | Yes (depth) | -- | -- | -- | -- | ~90+ |
| **SF (OpenVLA-OFT)** | **No** | **98.0** | **97.0** | **95.6** | **77.6** | **92.1** |
| pi_0 | No | -- | -- | -- | -- | baseline |
| **SF (pi_0)** | **No** | -- | -- | -- | -- | **best** |

核心发现: SF 在不需要额外 3D 输入的情况下, 达到甚至超过了使用 3D 传感器输入的方法 (GeoVLA, 3D-CAVLA) 的性能.

**RoboTwin Benchmark (仿真)**

SF 以 pi_0 为基模型, easy 任务评估 100 次, hard 任务评估 300 次. SF 在所有任务上均取得了最高平均成功率, 在 hard 任务上的提升尤为明显, 说明 SF 帮助模型准确捕捉物体位置和空间关系, 而非依赖快捷特征.

### 4.2 Ablation 分析

**Target Representation 选择 (LIBERO)**

| Target Model | Spatial | Object | Goal | Long | Avg |
|-------------|---------|--------|------|------|-----|
| Base (no align) | 97.2 | 93.0 | 91.6 | 67.0 | 87.2 |
| SigLIP | 97.8 | 94.8 | 93.2 | 72.4 | 89.6 |
| DINOv2 | 97.6 | 95.4 | 93.6 | 73.6 | 90.1 |
| **VGGT** | **98.0** | **97.0** | **95.6** | **77.6** | **92.1** |

**对齐层选择 (OpenVLA-7B, 32 层)**

| Layer | Avg Success Rate |
|-------|-----------------|
| Layer 8 (shallow) | 88.5 |
| Layer 16 (mid) | 89.8 |
| **Layer 24 (deep)** | **92.1** |
| Layer 32 (last) | 89.2 |

### 4.3 效率提升

| Metric | Value |
|--------|-------|
| 训练加速 | **3.8x** (达到相同成功率所需的 iteration 减少 3.8 倍) |
| 数据效率 | **5% 数据达到 75.8% 成功率**; 同等数据量下高出 base 25.8% |
| 推理开销 | **零额外开销** (推理时不需要 VGGT) |

### 4.4 真实世界实验

在双臂 AgileX 平台上 (6-DoF Piper + 1-DoF gripper), 仅用 40 个 demo (单臂) 和 20 个 demo (双臂) 训练:

| Task | Base SR | SF SR | Improvement |
|------|---------|-------|-------------|
| Stack glass cups (light variation) | ~30% | ~77.5% | +47.5% |
| Grasp right-side vegetable | lower | higher | significant |
| Place green block (height variation) | lower | 85% | significant |
| Bimanual lift pot | lower | higher | significant |

## 5. Related Work Analysis

SF 处于以下几条研究线的交汇处:

**2D VLA 谱系**: RT-1 -> RT-2 -> OpenVLA -> OpenVLA-OFT -> pi_0. 这条线的问题是: 视觉骨干在纯 2D 数据上预训练, 缺乏 3D 空间理解. SF 在这条线的下游, 用对齐策略补偿 3D 缺失.

**3D VLA 谱系**: GeoVLA, 3D-CAVLA, PointVLA, BridgeVLA. 这些方法通过注入深度图或点云来增强空间感知, 但受制于 3D 数据的获取和质量. SF 的对比实验表明, 隐式对齐可以达到甚至超过显式 3D 输入的效果.

**表征监督 (Representation Supervision)**: 
- 重建式: ROSS, ReconVLA (用重建目标监督中间表征)
- 对齐式: REPA (在 diffusion model 中对齐中间层与 DINOv2), 3DRS (在 MLLM 中引入 3D 表征监督)
- SF 属于对齐式, 但首次将这一范式系统地应用到 VLA 的空间理解问题上

**SF 填补的空白**: 此前的 3D VLA 工作聚焦于"丰富 VLA 的视觉输入", SF 则提出了一个正交的方向 -- 通过"监督 VLA 的中间表征"来隐式注入 3D 理解. 这两个方向是可以叠加的.

## 6. Limitations & Future Directions

### 局限性

1. **VGGT 依赖**: 训练阶段仍需要 VGGT 作为教师模型, 增加了训练的 GPU 显存需求和计算开销. 虽然 VGGT 是冻结的且推理时不需要, 但训练时需要同时加载 VLA + VGGT 两个大模型.

2. **目标表征的天花板**: SF 的性能上限受 VGGT 表征质量的约束. 如果 VGGT 的空间表征在某些场景下不准确 (如高度反射/透明物体), SF 可能无法提供正确的监督信号.

3. **多视角一致性假设**: VGGT 需要多视角输入才能建立 3D 一致性. 对于单相机 setup, VGGT 的空间表征质量可能下降.

4. **评估范围**: 仿真实验主要在 LIBERO 和 RoboTwin 上, 真实世界实验的 demo 数量有限 (40/20 个), 尚未在大规模真实场景中充分验证.

5. **对齐层选择**: 最优对齐层需要根据不同的 VLA 架构手动调参 (OpenVLA-7B 用第 24 层, pi_0 PaliGemma 用第 12 层), 缺乏自动化选择机制.

### 未来方向

1. **多层对齐**: 当前仅对齐单层, 可以探索多层联合对齐或渐进式对齐策略
2. **更强的 3D 教师**: 随着 3D 基础模型的发展, 可以替换为更强的 3D backbone (如未来版本的 VGGT 或其他 3D foundation model)
3. **与显式 3D 方法结合**: SF 与深度/点云注入方法是正交的, 可以叠加使用
4. **跨 embodiment 泛化**: 验证 SF 在跨机器人平台迁移时的效果
5. **动态场景**: 当前实验主要是 tabletop manipulation, 对动态/非结构化环境的适用性有待验证

## 7. Paper vs Code Discrepancies

### 7.1 Normalization 方式

- **论文**: 方程 (3) 中描述使用 "batch normalization Gamma" 后接两层 MLP
- **代码 (openvla-SF)**: `AlignProjector` 中实际使用的是 `nn.LayerNorm(llm_dim)` (可选, 由 `use_vlm_norm` 控制, 默认 False)
- **代码 (openpi-SF)**: 同样使用 `nn.LayerNorm`, 且在 openpi 配置中 `use_vlm_norm=True`
- **差异**: 论文说 BatchNorm, 代码用 LayerNorm, 且 openvla-SF 默认关闭 normalization

### 7.2 MLP 输出维度

- **论文**: 描述 MLP 将 VLA visual token 投影到"与 VGGT 兼容的特征维度"
- **代码**: MLP 输出维度实际是 `2 * vggt_dim` (即 2 * 1024 = 2048), 而非 VGGT 的原始维度 1024
- **可能原因**: `aggregated_tokens_list` 中 VGGT 的特征经 aggregator 处理后维度可能是 `2 * embed_dim`, 但论文未明确说明这一点

### 7.3 Mask 处理

- **论文**: 未讨论非矩形图像的 padding mask 处理
- **代码 (openpi-SF)**: `pi0_align_pytorch.py` 中有详细的 `align_mask` 处理逻辑, 包括空图像 mask 和非矩形 padding mask, 用于排除无效区域的 alignment loss 计算
- **代码 (openvla-SF)**: `AlignProjector.forward()` 不接收 mask 参数, alignment loss 在所有 token 上计算, 未做 mask 过滤. 但 `finetune_align.py` 的 `run_forward_pass` 中也没有传递 mask
- **差异**: openpi-SF 实现了更完善的 mask 机制, openvla-SF 忽略了 mask

### 7.4 Feature Shift (gain_feat_1move)

- **论文**: 未提及
- **代码 (openvla-SF)**: 有一个 `gain_feat_1move` 选项 (默认 True), 将 VLA 的 vision hidden states 索引偏移 1 位 (`boi_ids = 2 if gain_feat_1move else 1`)
- **意义**: 这可能是因为 causal attention 中, 每个位置的 hidden state 实际上编码的是"之前所有 token 的信息", 所以需要偏移 1 位才能让 vision token 的 hidden state 与正确的空间位置对应. openpi-SF 中未发现类似机制

### 7.5 PE 添加时机

- **论文**: 方程 (3) 中, PE 加到 VGGT 的输出上: `f_3D(I) + E`
- **代码**: PE 是在 bilinear pooling 之前添加到 VGGT 特征的 2D spatial grid 上 (`_apply_pos_embed`), 使用正弦余弦编码, 且有一个衰减系数 `ratio=0.1`
- **差异**: 论文描述是直接相加, 代码实现更复杂 -- 使用 UV grid 正弦编码并有缩放因子

### 7.6 对齐层的默认配置差异

- **openvla-SF**: 默认 `vla_layers_align = -1` (最后一层), 但 readme 和 train.sh 中推荐用 24
- **openpi-SF**: 配置中 `vla_layers_align = 12`, `vggt_layers_align = -1` (最后一层)
- **论文**: 最优对齐层是 VLA 的第 24/32 层 (OpenVLA-7B)

### 7.7 VGGT 初始化

- **代码**: VGGT 使用 `feature_only=True` 模式, 禁用 camera/point/depth/track 所有 prediction heads, 仅提取 backbone features
- **论文**: 正确描述了使用 VGGT backbone 的 latent representation, 但未强调 heads 被完全禁用

## 8. Cross-Paper Comparison

### 与 Policy Learning 方法的对比

| 维度 | ACT | Diffusion Policy | SF (本文) |
|------|-----|-------------------|-----------|
| 动作表示 | VAE + CVAE chunking | DDPM 去噪过程 | 与基模型一致 (L1/Flow) |
| 空间理解 | 无特殊设计 | 无特殊设计 | VGGT 表征对齐 |
| 视觉骨干 | ResNet | ResNet/ViT | 继承 VLA 骨干 + SF |
| 模型规模 | 小 (~10M) | 中 (~100M) | 大 (7B VLA) |
| 核心创新 | action chunking + 多头预测 | 将 action 建模为去噪过程 | 隐式空间表征对齐 |
| 推理开销 | 低 | 高 (多步去噪) | 与基模型相同 |
| 与 SF 的关系 | 无直接关联 | SF 可应用于 Flow-based pi_0 | SF 是训练策略, 不改变推理 |

### 与 VLA 方法的对比

| 维度 | OpenVLA | pi_0 | SF (本文) |
|------|---------|------|-----------|
| VLM 骨干 | Prismatic (SigLIP + DINOv2) | PaliGemma | 继承基模型 |
| 动作生成 | Token 分类 / L1 回归 | Flow Matching | 继承基模型 |
| 3D 理解 | 无 | 无 | VGGT 对齐 |
| 训练数据 | OXE 大规模预训练 | OXE + DROID | 基模型 + SF 微调 |
| 微调方式 | LoRA (OFT) | 全参数 | 基模型微调 + align loss |
| 推理差异 | -- | -- | 与基模型完全一致 |
| SF 关系 | SF 的基模型之一 | SF 的基模型之一 | 适用于两者的通用训练策略 |

### 与灵巧手操作 VLA 的对比

| 维度 | DexGraspVLA | UniDex | SF (本文) |
|------|-------------|--------|-----------|
| 聚焦场景 | 灵巧抓取 | 灵巧操作泛化 | 通用操作 (含双臂) |
| 3D 处理 | 点云输入 | 触觉 + 视觉 | 隐式对齐, 无显式 3D |
| 数据效率 | 需要大量抓取数据 | 需要多模态数据 | 5% 数据达 75.8% SR |
| 核心贡献 | VLA 用于 dexterous grasping | 跨具身灵巧操作 | 训练范式: 表征对齐 |
| 适用范围 | 灵巧手专用 | 灵巧手专用 | 通用 VLA 增强策略 |
| 对 robotics FM 的启示 | VLA 可以做 dexterous | 触觉融合提升 dexterous | 3D 感知不必依赖 3D 传感器 |

### 核心 Takeaway

SF 不是一个新的模型架构, 而是一个**通用的训练策略** (training paradigm). 它的最大价值在于:
1. 证明了"隐式空间监督"可以替代"显式 3D 输入", 为 3D VLA 提供了一条轻量化路径
2. 方法与基模型架构正交, 可以即插即用地增强任何 VLA (论文验证了 OpenVLA-OFT 和 pi_0 两个基模型)
3. 推理零开销, 训练效率提升显著 (3.8x 训练加速 + 数据高效), 实用性强
4. 从表征学习的视角重新审视 VLA 的 3D 能力缺陷, 与 REPA/3DRS 共同构成了"representation alignment"这一新兴研究方向
