# GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities -- 论文笔记

> CVPR 2025 Highlight | Brown University + ETH Zurich
> arXiv: 2412.04244 | [Project Page](https://ivl.cs.brown.edu/research/gigahands.html)

---

## 1. Core Problem

双手活动 (bimanual hand activities) 的理解是 AI 和机器人领域的关键问题, 但现有数据集在三个维度上存在根本性不足:

1. **规模不够**: 现有最大的 3D 手部运动数据集不超过数百分钟, 远低于 LLM/CV 领域的数据规模
2. **多样性不足**: studio 采集设置限制了动作种类; in-the-wild 数据虽然多样但 3D 精度差
3. **标注缺失或稀疏**: 大多数数据集要么没有文本标注, 要么只有动作类别标签, 缺少原子级别 (atomic-level) 的细粒度描述

核心矛盾在于: studio 环境能获得精确 3D 数据但动作不自然, marker 干扰真实交互; in-the-wild 环境动作自然但 3D 重建精度低, 标注成本高. GigaHands 试图在精度、多样性、真实性三者之间找到平衡.

---

## 2. Method Overview

GigaHands 的核心方法是一套名为 **Instruct-to-Annotate** 的程序化数据采集流水线, 覆盖从指令生成到 3D 重建的全链路:

```
Instruction Elicitation (LLM辅助生成脚本)
    -> Filming (51相机 markerless 采集)
        -> Action Annotation & Text Augmentation (人工分段 + LLM 5x改写)
            -> Hand Motion Estimation (YOLOv8 + HaMeR + ViTPose + 三角化 + MANO拟合)
                -> Object Motion Estimation (DINOv2 + SAM2 + 可微渲染 pose tracking)
```

**最终产出**: 34 小时双手活动, 56 名被试, 417 个物体, 14K 运动片段, 84K 文本标注, 183M 帧 RGB 图像 (51 视角), 全自动 3D hand/object shape + pose 估计.

**关键设计哲学**: 不发明新的感知算法, 而是组合现有 SOTA 工具 (HaMeR, ViTPose, DINOv2, SAM2, Instant-NGP, PyTorch3D) 并通过 dense multi-view 约束弥补单个工具的精度不足.

---

## 3. Key Designs

### 3.1 Instruct-to-Annotate Pipeline (程序化指令引出与标注策略)

**问题**: 传统 studio 采集要么让被试自由发挥 (动作多样但标注困难), 要么给出严格脚本 (标注简单但动作单一).

**解法**: 用 LLM (GPT-4) 从多个数据集 (Ego4D, Ego-Exo4D, OakInk2, TACO) 中提取动词, 按场景组织为时间连贯的指令脚本.

- **层级结构**: 5 scenarios -> 25 scenes -> 191 activities -> 1370 instructions, 包含 533 个动词
- **拍摄时**: 指令转为音频播放, 被试按指令表演; 若结束状态与下条指令不匹配则补拍过渡动作
- **标注时**: 人工将 13K motion sequences 切分为 14K clips, 修正 LLM hallucination 和被试误操作
- **文本增强**: LLM 对每条描述改写 5 次, 14K clips -> 84K text-motion pairs, 1467 个唯一动词

**与 Ego4D 的关键差异**: Ego4D 在 unscripted 场景中需要大量后期人工标注; GigaHands 通过 scripted 方式将标注工作前置, 极大降低了标注成本.

### 3.2 Markerless Multi-View 3D Estimation Pipeline (全自动 3D 重建)

**硬件**: 51 台 RGB 相机, 30fps, 1280x720, 立方体布局 (每面 3x3 网格), 透明玻璃台面, LED 均匀照明. 相机时间对齐误差 < 3ms, 内外参用 COLMAP + fiducial markers 标定.

**Hand Motion Estimation (混合方法)**:
1. YOLOv8 检测手部 bounding box
2. HaMeR 估计 MANO mesh -> 提取 2D keypoints (HaMeR 直接输出的深度不准, 所以只用 2D)
3. ViTPose 判断左右手
4. 多视角 2D keypoints 三角化得到 3D keypoints
5. One-euro filter 时间平滑
6. EasyMoCap 拟合 MANO 参数

**Object Motion Estimation (自研方法)**:
1. DINOv2 (1fps subsampling) 检测显著物体
2. Grounding DINO + 物体文本/渲染模板 -> 选择 top-k bbox
3. OpenCLIP 负样本过滤误检
4. SAM2 分割物体 mask
5. Instant-NGP 建立 radiance field -> 初始化物体位置
6. FoundPose 方法 + DINOv2 features -> 初始化旋转 (处理对称性)
7. PyTorch3D 可微渲染 + 多视角 mask 监督 -> 优化 6DoF pose

**统计**: 追踪了 17,979 个物体, 其中 3,356 个序列被人工验证为成功 (成功率约 18.7%, 说明物体追踪仍是主要瓶颈).

### 3.3 Dataset-as-Benchmark: Text-Motion 双向应用验证

GigaHands 不仅是数据集, 还通过两个 benchmark 任务验证了规模带来的收益:

**Text-to-Motion Synthesis**: 基于 T2M-GPT (VQ-VAE + GPT) 架构, 用 42 个 3D keypoints (双手) 作为运动表示. 在 GigaHands 上训练的模型在 R-Precision, FID, Diversity, Multimodality 上均优于 OakInk2 和 TACO. Scaling curve 显示性能随数据量持续提升 (10%->100%), 未出现饱和.

**Motion Captioning**: 基于 TM2T 架构, GigaHands 训练的模型能为自身和外部数据集 (TACO, OakInk2) 的运动生成细粒度描述. Pairwise BLEU 和 distinct-n 指标显著优于其他数据集.

**Dynamic Radiance Field Reconstruction**: 51 视角使得逐帧 2DGS (2D Gaussian Splatting) 重建成为可能, 甚至可以处理非刚性物体 (如裤子拉链). 这是现有手部数据集无法实现的应用.

---

## 4. Experiments

### 4.1 Text-to-Motion Synthesis

| 训练数据 | R-Precision Top1 | R-Precision Top3 | FID | Diversity | MM Dist | Multimodality |
|----------|------------------|------------------|-----|-----------|---------|---------------|
| OakInk2 | - | - | 较差 | 较差 | - | 较差 |
| TACO | - | - | 较差 | 较差 | - | 较差 |
| GigaHands | **最优** | **最优** | **最优** | **最优** | 次优 | **最优** |

- 在 MM Dist 上 GigaHands 不是最优, 论文未详细解释原因
- Scaling experiment: FID, MM Dist, Top1, Top3 均随数据量单调改善 (10% -> 100%), 表明数据规模的 scaling law 在手部运动领域同样成立
- 也在 GRU-based (TM2T) 和 diffusion (MDM) 架构上验证, 结论一致 (见 supp.)

### 4.2 Motion Captioning

- GigaHands 在 Pairwise BLEU 和 distinct-n 上显著领先, 说明生成的 caption 多样性更高
- 在 BLEU@4 和 ROUGE 上与 TACO 的简单三元组分类接近
- 跨数据集 captioning: GigaHands 训练的模型能为 TACO 和 OakInk2 的运动生成合理的细粒度描述

### 4.3 Dynamic Radiance Field

- 51 视角中去除 12 个光照问题相机, 随机留 1 个测试, 38 个训练
- 逐帧 2DGS 重建, 前一帧初始化下一帧保证时间一致性
- 能处理非刚性物体, 但论文只展示了定性结果

---

## 5. Related Work Analysis

论文对手部数据集的分类体系非常清晰, 按 **数据来源** 和 **标注类型** 两个维度组织:

**数据来源** (4类):
1. **Static poses**: 运动规划/策略学习的单帧抓取, 缺乏语义
2. **Synthetic**: 游戏引擎/VR/仿真生成, 运动不自然
3. **In-the-wild**: 穿戴式/便携传感器, 真实但 3D 精度差
4. **Studio**: 受控环境, 精确但多样性低; 又分 marker-based (精确但不自然) 和 markerless (自然但精度权衡)

**标注类型** (4类):
- Hand motion: bbox -> mask -> 2D kp -> 3D kp -> MANO params (精度递增)
- Object: 手工标注 / MoCap / 多视角 RGB-D
- Text: 动作类型 / 原子描述 / 活动叙述 / 物体功能
- Contact: 手工 / 热传感器 / 几何分析

GigaHands 定位为 "studio + markerless" 路线, 通过程序化指令模拟 in-the-wild 多样性, 通过 dense multi-view 弥补 markerless 精度不足.

---

## 6. Limitations & Future Directions

### 论文明确提到的局限

1. **空间受限**: studio 环境限制了大范围运动 (如行走中的手部活动)
2. **物体追踪不完整**: 仅能追踪刚性物体和物体部件, 铰接/非刚性物体的全自动追踪仍是开放问题 (3,356/17,979 成功率说明了这一点)
3. **应用验证有限**: 仅展示了 motion synthesis 和 captioning, 未验证在机器人操作、HCI 等下游任务上的效果

### 未明确说明但值得注意的局限

4. **无力/触觉数据**: 与 DexCanvas 等数据集类似, 没有 contact force 标注, 限制了在需要力控的机器人操作中的应用
5. **被试群体偏差**: 56 名被试多为右利手, 右手接触频率高于左手, 可能导致模型偏差
6. **Text augmentation 的可靠性**: LLM 改写 5 次将 14K 扩展到 84K, 但改写质量和语义保真度未系统评估
7. **非连续物体 pose**: 物体追踪成功率仅约 18.7%, 意味着大部分序列缺少物体 6DoF pose
8. **MANO 精度上界**: HaMeR -> 三角化 -> MANO 拟合的 pipeline 依赖于 2D keypoint 质量, 严重遮挡时可能失败 (论文未报告逐步精度)
9. **30fps 帧率**: 对于快速手指运动 (如打字、弹奏乐器), 30fps 可能不足

---

## 7. Paper vs Code Discrepancies

| 方面 | 论文描述 | 代码实现 | 差异分析 |
|------|---------|---------|---------|
| **运动表示** | 42 个 3D keypoints (双手各 21) | `HandVQVAE` 中 `input_dim = 126` (42x3), MANO 模式 `input_dim = 198` | 代码支持两种表示, 论文只报告了 keypoints 的结果 |
| **数据划分比例** | "train, test, val = 16:3:1" (即 80:15:5) | `train_test_split(test_size=0.2)` + `train_test_split(remaining, test_size=0.25)` = 80:15:5 | 一致 |
| **VQ-VAE codebook** | 论文未详细描述 quantization 方法 | 实现了 4 种: `ema_reset`, `orig`, `ema`, `reset`; 默认用 `ema_reset` (EMA + 死码重置) | 代码细节多于论文 |
| **T2M-GPT 架构** | 论文只说基于 T2M-GPT | `Text2Motion_Transformer`: 9层, embed_dim=1024, 16 heads, block_size=51, dropout=0.1 | 代码提供了完整超参, 论文未报告 |
| **Text encoding** | 论文未说明文本编码器 | 代码使用 CLIP ViT-B/32 的 `encode_text()` | 代码比论文描述更完整 |
| **Text augmentation** | "rephrase 5 times" | 代码中 `all_scripts` 需要长度恰好为 6 (1 原始 + 5 改写), 否则跳过 | 一致, 但代码有严格过滤 |
| **Motion window** | 论文未提及 | `max_motion_length` 在 VQ-VAE 训练中通过 `window_size=128` 控制 | 代码有窗口截断, 论文未说明 |
| **Evaluation metrics** | 论文说 feature extractor 独立训练 | `EvaluatorModelWrapper` 加载预训练的 motion/text encoder | 一致 |
| **Object pose 可视化** | 论文提到 hand-object 联合可视化 | `render_mesh_video.py` 完整实现了 MANO + object mesh + camera 的渲染 | 代码质量较高, 包含平滑和插值 |
| **数据采集 pipeline** | Instruct-to-Annotate 全流程 | 代码仓库不包含采集 pipeline, 只有下游任务训练和可视化代码 | 采集侧代码未开源 |
| **Object tracking pipeline** | DINOv2 + Grounding DINO + SAM2 + Instant-NGP + PyTorch3D | 代码仓库不包含 object tracking 代码 | 核心数据处理 pipeline 未开源 |
| **Hand estimation pipeline** | YOLOv8 + HaMeR + ViTPose + 三角化 + EasyMoCap | 仅依赖 EasyMoCap (作为 third-party), 其余未提供 | 手部估计代码大部分未开源 |

**总结**: 代码仓库主要覆盖**下游应用** (text-to-motion, captioning, 可视化), 不包含**数据采集和处理 pipeline**. 这是 dataset paper 的常见模式, 但意味着其他研究者无法复现数据处理流程.

---

## 8. Cross-Paper Comparison

### 8.1 数据集规模与覆盖度对比

| 维度 | GigaHands | ARCTIC | DexCanvas | Ego4D / Ego-Exo4D | DexCap |
|------|-----------|--------|-----------|-------------------|--------|
| **定位** | 大规模双手活动数据集 | 铰接物体双手交互 | 单手抓取操作分类 | 大规模第一人称视频 | 人手遥操作采集系统 |
| **年份** | 2025 (CVPR) | 2022 (CVPR) | 2025 | 2022/2024 | 2024 |
| **规模 (时长)** | **34 hours** | ~0.5 hours | ~4 hours | **3670+ hours** (Ego4D) | ~1.5 hours |
| **被试数量** | **56** | 10 | 15 | **923** (Ego4D) | 少量 |
| **物体数量** | **417** | 10 (铰接) | 30 | 大量 (无统计) | 少量 |
| **手部表示** | MANO (双手) | MANO (双手) | MANO (单手为主) | 2D/3D kp + MANO | Joint angles |
| **物体 6DoF** | 有 (3,356 seq 成功) | 有 (铰接状态) | 有 | 无/稀疏 | 无 |
| **文本标注** | **84K** dense atomic | 无 | 无 | 动作类型/叙述 | 无 |
| **3D 精度** | 高 (markerless multi-view) | 高 (MoCap markers) | 高 (MoCap) | 低 (单/少视角) | 中 (exo cameras) |
| **相机数量** | **51** | 8-10 | MoCap + RGB | 1-4 (ego/exo) | ~4 exo |
| **Contact force** | 无 (几何推断) | 无 | 未发布 | 无 | 无 |
| **License** | CC-BY-NC 4.0 | 需注册 | 申请制 | 需注册 | 开放 |

### 8.2 方法论对比

| 方面 | GigaHands | ARCTIC | DexCanvas | Ego4D | DexCap |
|------|-----------|--------|-----------|-------|--------|
| **采集方式** | Markerless multi-view studio | Marker-based MoCap | Marker-based MoCap + RGB | In-the-wild 穿戴式 | Exocentric cameras + 手套 |
| **标注策略** | Instruct-to-Annotate (LLM 生成脚本) | 预设任务 | Cutkosky 分类法指导 | Post-hoc 人工标注 | 无文本标注 |
| **3D 重建** | HaMeR + 三角化 + EasyMoCap | MoCap 直接获取 | MoCap 直接获取 | Ego-pose estimation | Camera + retargeting |
| **物体追踪** | DINOv2 + SAM2 + diff rendering | 附带 marker | MoCap marker | 无 | 无 |
| **可扩展性** | 高 (全自动 pipeline) | 低 (需 marker) | 低 (需 marker) | 高 (but 3D 质量低) | 中 |

### 8.3 对机器人应用的价值分析

| 数据集 | Retargeting 可用性 | Policy Learning 价值 | 关键优势 | 关键短板 |
|--------|-------------------|---------------------|---------|---------|
| **GigaHands** | 高: MANO -> 任意机械手 | 中: 无力数据, 无 robot demo | 规模最大, 文本丰富, 多视角 | 无 contact force; 物体 pose 覆盖率低 (~19%) |
| **ARCTIC** | 高: MANO 双手 | 低: 仅 10 个铰接物体 | 铰接物体交互独特 | 规模太小, 无文本 |
| **DexCanvas** | 高: MANO + Cutkosky 分类 | 中: 抓取分类学指导 | 操作分类学完整 | 规模小, 物体少, force 未发布 |
| **Ego4D** | 低: 3D 精度差 | 低: 无精确 3D pose | 规模巨大, 真实场景 | 3D 重建质量不足以训练操作策略 |
| **DexCap** | 高: 直接 retarget 到 LEAP | **高**: 端到端采集->部署 pipeline | 唯一提供 robot deployment 闭环 | 规模极小, 仅 2 个任务 |

### 8.4 核心 Takeaway

1. **GigaHands 是目前最大的 3D 双手活动数据集**, 但 "大" 主要体现在帧数 (183M) 和文本标注 (84K); 在有效物体 pose 序列方面 (3,356 seq) 并不比 ARCTIC 大很多
2. **Instruct-to-Annotate 是最值得迁移的设计**: 用 LLM 生成指令脚本 + 被试按脚本表演 + 人工微调, 这种 "半自动标注" 思路可以迁移到任何 studio 数据采集场景
3. **Multi-view markerless 是未来方向**: 相比 marker-based (ARCTIC, DexCanvas), markerless 采集更自然且可扩展, 但精度仍需通过 dense views (51 cameras) 来弥补
4. **对 robotics 的直接价值有限**: GigaHands 验证的 text-to-motion 和 captioning 任务偏向 CV/NLP, 未验证 retargeting + sim2real 等 robotics pipeline. DexCap 虽然规模小 100 倍, 但提供了完整的 human demo -> robot policy 闭环
5. **Scaling law 初步验证**: 10% -> 100% 的 scaling curve 在手部运动领域也成立, 这支持了 "更多数据 -> 更好性能" 的假设, 但曲线是否已经饱和还不确定
