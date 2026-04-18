# PALM Notes

> PALM: A Dataset and Baseline for Learning Multi-Subject Hand Prior
> arXiv: 2511.05403v3 | Meta Reality Labs, ETH Zurich, MPI
> Accepted at CVPR (推断，基于 v3 Feb 2026 更新)

---

## 1. Core Problem

高质量个性化手部 avatar 重建面临两个核心瓶颈:

1. **缺乏大规模多被试手部数据集**: 现有数据集要么被试数量少 (InterHand2.6M: 50 人, MANO: 31 人)，要么缺少精确 3D 几何 (MANO 仅提供参数化模型而非真实扫描)，要么数据不公开 (Handy 有 3dMD 扫描但未释放)
2. **单图像手部 avatar 个性化是严重欠约束问题**: 深度模糊、自遮挡 (手背/手掌)、未知光照，使得从一张图像恢复几何 + 外观 + 材质 + 光照几乎不可能

PALM 通过提供 263 被试的亚毫米级 3D 扫描 + 高分辨率多视角 RGB，构建跨被试手部先验 (multi-subject hand prior)，将欠约束的单图像重建转化为先验约束下的优化问题。

---

## 2. Method Overview

### 数据集部分 (PALM Dataset)

| 属性 | 值 |
|------|-----|
| 被试数 | 263 (131 male, 132 female) |
| 年龄 | 21-70 |
| 肤色 | 15% light, 20% tan, 38% medium, 27% dark |
| 3D 扫描 | ~13K (3dMD scanner, 亚毫米精度) |
| RGB 图像 | ~90K (7 viewpoints, 2448x2048) |
| 相机 | 7 RGB + 14 monochrome = 21 machine vision cameras |
| 手势 | ~50 predefined right-hand gestures per subject |
| MANO registration | 有 (2-stage: 先优化 shape+pose, 再冻结 shape 只优化 pose) |
| 3D keypoint | Semi-automatic (U-Net pretrained on InterHand2.6M, fine-tuned on 10K annotated PALM images + RANSAC triangulation) |
| MANO fitting error | 5.3mm (vs keypoints), comparable to InterHand2.6M |
| 3D keypoint recall | 95% at 10mm threshold |

### 方法部分 (PALM-Net Baseline)

基于 PBR (Physically Based Rendering, 基于物理的渲染) 的隐式神经先验，将手部分解为:

| 网络 | 输入 | 输出 |
|------|------|------|
| f_g (geometry) | canonical point x_c + MANO params (theta, beta) + shape code phi | density sigma + geometry features z |
| f_rf (radiance field) | x_c + z + reflected view dir + normal + theta + appearance code psi | outgoing radiance L |
| f_m (material) | x_c + z + theta + psi | albedo alpha + roughness r + metallicity m |

**关键组件**:
- **SNARF**: 使用 SNARF (differentiable forward skinning) 实现 inverse LBS (Linear Blend Skinning, 线性混合蒙皮)，将 deformed space 点映射到 canonical space
- **SDF-based opacity**: 通过 scaled Laplace CDF 将 SDF (Signed Distance Function, 有符号距离函数) 转换为 density
- **Hash grid encoding**: 对 canonical 点做 hash grid 编码，高效捕捉高频细节
- **Spherical Gaussians**: 环境光照用球面高斯参数化，所有被试共享同一光照 (因为采集环境固定)

---

## 3. Key Designs

### 3.1 Multi-Subject PBR Prior -- 跨被试解耦表示

**核心洞察**: 尽管人手在细节上 (皱纹、纹理) 因人而异，但皮肤色调、材质反射率、形变行为等基本属性是跨个体共享的，可以被先验模型捕捉。

**实现方式**:
- 每个被试有独立的 shape code phi 和 appearance code psi
- 网络权重 Phi 跨被试共享
- 环境光照 {SG_i} 跨被试共享 (同一采集环境)

**解耦策略**: 分离 shape code 和 appearance code 而非使用统一 latent code。论文实验发现分离后重建质量更好。

**优化目标** (Eq. 15):
```
min_{Phi, {phi_i}, {psi_i}, {SG_i}} L
```
同时优化网络权重、被试编码、和环境光照。

### 3.2 Loss 设计 -- 多约束联合监督

总 loss 包含 8 项:

| Loss 项 | 公式 | 作用 |
|---------|------|------|
| L_rf | RGB 渲染 vs GT | radiance field 颜色监督 |
| L_pbr | PBR 渲染 vs GT | physically based 颜色监督 |
| L_segm | BCE(rendered mask, GT mask) | 前景-背景分离 |
| L_normal | rendered normal vs scan normal | **3dMD 扫描的核心价值** -- 提供高精度法线监督 |
| L_eikonal | ||grad(SDF)|| = 1 | SDF 正则化 |
| L_LPIPS | perceptual similarity | 高频细节 |
| L_LAP | Laplacian 平滑 | 表面平滑性 |
| L_latent | MSE(codes, 0) | latent code 正则化 |

**关键实验发现**: L_normal (3dMD scan normal supervision) 是消除 pepper-like artifacts 和 floaters 的关键。没有 3dMD 扫描的法线监督，仅靠 7 视角 RGB 无法获得高保真 3D 重建 (Table 4, Figure 7)。

### 3.3 Single-Image Personalization -- 冻结先验 + 优化编码

**个性化流程**:
1. 冻结预训练的网络权重 Phi
2. 优化: shape code phi + appearance code psi + environment map {SG_i}
3. Loss: L_rf + lambda_pbr * L_pbr + lambda_segm * L_segm + lambda_LPIPS * L_LPIPS

**核心假设**: 先验模型约束了 albedo 和 material properties 的合理范围，所有环境效应由独立的 environment map 解释。这使得即使输入灰度图像，也能恢复合理的 albedo (因为先验"知道"手应该是什么颜色)。

---

## 4. Experiments

### 4.1 评估数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| InterHand2.6M | 真实图像个性化 | 12 sequences, 每个 20 帧评估, 第一帧训练 |
| HARP relit (合成) | Novel environment + novel pose | Blender 渲染, DART 皮肤纹理, HARP pose 动画 |
| In-the-wild images | 定性展示泛化 | 网络采集的多样光照/姿态图像 |

### 4.2 主要结果

**InterHand2.6M (Table 2)**: PALM-Net 在 novel pose 设置下所有指标 (PSNR, SSIM, LPIPS) 优于 baseline

**Synthetic dataset (Table 3)**: 在 novel environment + novel pose 设置下优于 baseline，说明外观重建更准确

### 4.3 Ablation Studies

| 消融 | 结论 |
|------|------|
| w/ vs w/o 3dMD normals (Table 4) | 3dMD 法线监督显著减少 artifacts，定量指标提升 |
| w/ vs w/o environment map optimization (Table 5) | 建模环境光照使拟合更准确 (Figure 8) |

### 4.4 Baseline 方法

论文对比了 prior-based 和 non-prior-based 方法，但具体 baseline 名称在我阅读的部分中未完全列出 (details in SupMat)。从 related work 推断可能包括 HARP、URHand、LISA/OHTA 等。

---

## 5. Related Work Analysis

### 手部数据集演进

| 阶段 | 代表 | 缺陷 |
|------|------|------|
| 深度相机时代 | NYU, MSRA, BigHand | 无 RGB, 被试少 |
| RGB 时代 | STB, FreiHAND | 无 3D 扫描 |
| 交互手 | InterHand2.6M (50人) | 无 3D 扫描 (仅有 keypoint) |
| 参数模型 | MANO (31人 3D 扫描) | 扫描少、不含 RGB |
| 手-物体 | DexYCB, ARCTIC, ContactPose | 非 hand-only |
| 合成 | DART, Re:InterHand | 合成数据, 非真实 |
| **PALM** | **263人 3D 扫描 + 90K RGB** | 首次大规模真实扫描 + 多视角 RGB + 公开 |

### 手部表示方法对比

| 方法 | 几何 | 外观 | 重光照 | 数据需求 |
|------|------|------|--------|---------|
| MANO/HTML | PCA mesh | PCA texture | 否 | 3D 扫描 |
| Handy | 参数化 | 参数化 texture | 否 | 3dMD (未公开) |
| NIMBLE | bone+muscle+skin | PCA albedo+specular | 部分 | **Light stage** |
| URHand | Pose-dependent | Material properties | 是 | **大规模 light stage** |
| LISA/OHTA | Implicit field | Baked-in lighting | 否 | 多视角 RGB |
| HARP | MANO mesh | Normal + albedo map | 部分 | 单目视频 |
| **PALM-Net** | SDF implicit | PBR (albedo + roughness + metallicity) | **是** | **3dMD 扫描 + 7 视角 RGB** (无需 light stage) |

---

## 6. Limitations & Future Directions

### 明确局限

1. **仅右手**: 数据集只包含右手，左手需要镜像处理
2. **静态姿态**: ~50 个预定义姿态 per subject，非连续动态序列。无法用于手部运动建模
3. **固定光照**: 所有数据在同一 3dMD 采集环境下录制。虽然有利于训练 (共享 environment map)，但先验模型的光照泛化仅在 personalization 阶段验证
4. **无物体交互**: 手部空手做姿势，不包含任何手-物体交互数据
5. **无接触力/触觉**: 纯视觉+几何数据
6. **申请制访问**: 虽然承诺公开，但可能需要审批流程

### 未来方向 (可推断)

- 扩展到左手 + 双手交互
- 增加动态序列以支持手部运动先验
- 手-物体交互场景下的手部 avatar
- 与触觉传感器数据集结合
- 从 NeRF-based 迁移到 3DGS (3D Gaussian Splatting) 以提升渲染速度
- 与 diffusion model 结合做手部图像生成

---

## 7. Paper vs Code Discrepancies

| 维度 | 论文描述 | 代码/数据现状 | 差异分析 |
|------|---------|-------------|---------|
| **代码仓库** | 项目页面指向 https://github.com/facebookresearch/PALM | **当前目录下无代码** (仅论文 MD 文件) | 需从 GitHub 获取。Meta 开源项目通常有延迟 |
| **数据公开** | "will be made publicly available for research use upon publication" | **未在目录中看到数据** | 可能需要申请，CLAUDE.md 记录为 "~137GB (申请制)" |
| **quality_dict** | CLAUDE.md 提到 "quality_dict 筛选机制" | 论文中未提及该术语 | 可能是数据释放时的质量筛选机制，非论文贡献 |
| **PALM-Net 完整实现** | 涉及 SNARF + NeRF + PBR + hash grid + SDF 等大量组件 | 需检查 GitHub 仓库是否完整 | PBR inverse rendering pipeline 实现复杂度高 |
| **训练细节** | Loss 权重 lambda_* 详见 SupMat | SupMat 未包含在 MD 中 | 关键超参数不在主论文中 |
| **环境光照** | 所有被试共享一个 environment map | 数据采集确实在固定环境 | 这个约束在部署时不存在 -- personalization 必须重新优化光照 |

---

## 8. Cross-Paper Comparison

### vs DexCap

| 维度 | DexCap (2024) | PALM (2025) |
|------|--------------|-------------|
| 核心目标 | 灵巧手操作数据采集 + 部署 pipeline | 手部外观/几何先验学习 |
| 手部表示 | 人手关节 -> LEAP hand retarget | MANO registration |
| 数据类型 | 动态操作轨迹 (wiping, packaging) | 静态姿态 (50 预定义手势) |
| 物体信息 | 有 (但格式非标准) | 无 (hand-only) |
| 被试多样性 | 少 (采集者人数未公开) | **263 人**, 广泛人口统计学分布 |
| 3D 几何精度 | 中 (SLAM + retarget) | **亚毫米** (3dMD scanner) |
| 互补性 | DexCap 提供运动轨迹 | PALM 提供手部形态先验 |

### vs DexGraspNet

| 维度 | DexGraspNet (2022) | PALM (2025) |
|------|-------------------|-------------|
| 数据内容 | ShadowHand 抓取姿态 (1.32M) | 人手 3D 扫描 + 多视角 RGB |
| 手部表示 | ShadowHand 28D joints | MANO (45D pose + 10D shape) |
| 物体信息 | 5355 objects | 无 |
| 数据来源 | 合成 (differentiable force closure) | 真实 (3dMD scanner) |
| 应用方向 | 抓取姿态生成 | 手部 avatar 重建 + 重光照 |
| 交集 | 几乎无直接交集 | PALM 可为 DexGraspNet 的 retarget 提供 human hand shape prior |

### vs EgoDex

| 维度 | EgoDex (2025) | PALM (2025) |
|------|--------------|-------------|
| 规模 | **829h, 338K episodes** | 13K scans, 90K images |
| 视角 | 第一人称 (egocentric) | 第三人称 (7 固定相机) |
| 手部表示 | ARKit skeleton (21 joints) | **MANO** (参数化模型) |
| 3D 几何 | 无真实 3D 扫描 | **亚毫米 3dMD 扫描** |
| 任务 | 194 种操作任务 | 50 种空手姿势 (无任务) |
| 被试 | 未明确 | **263 人** (人口统计学多样) |
| 核心差异 | 大规模操作行为数据 | 高精度手部形态+外观数据 |
| 互补性 | EgoDex 需要手部先验做 pose estimation | PALM 提供该先验 |

### vs GigaHands

| 维度 | GigaHands (2025) | PALM (2025) |
|------|-----------------|-------------|
| 规模 | 34h, 14K clips, 417 objects, 56 subjects | 13K scans, 263 subjects |
| 手部表示 | MANO mesh + keypoints | **MANO + 3dMD 亚毫米扫描** |
| 物体信息 | 3D mesh + 6D pose (18.7% 追踪成功率) | 无 |
| 外观数据 | 51 视角 RGB | 7 视角 2448x2048 RGB |
| 3D 几何质量 | 中 (无标记, HaMeR + 三角化) | **极高** (3dMD scanner) |
| 被试多样性 | 56 人 | **263 人** (4.7x 更多) |
| 核心差异 | 大规模手-物体交互运动 | 手部形态先验 (shape + appearance + material) |
| 互补性 | GigaHands 可用 PALM 先验提升 hand mesh 重建质量 | PALM 无操作数据 |

### 关键对比总结

| 特性 | DexCap | DexGraspNet | EgoDex | GigaHands | **PALM** |
|------|--------|-------------|--------|-----------|---------|
| 手部 3D 扫描 | 否 | 否 | 否 | 否 | **是 (亚毫米)** |
| 被试多样性 | 少 | 无 | 未知 | 56 人 | **263 人** |
| MANO 参数 | 否 | 否 | 否 | 是 | **是 + 3D scan** |
| 外观建模 | 否 | 否 | 否 | 否 | **是 (PBR)** |
| 重光照能力 | 否 | 否 | 否 | 否 | **是** |
| 物体交互 | 是 | 是 | 是 | 是 | **否** |
| 动态序列 | 是 | 否 | 是 | 是 | **否** |
| 接触力 | 否 | 否 | 否 | 否 | 否 |

**PALM 的独特定位**: 唯一提供大规模多被试 3D 手部扫描 + 多视角 RGB + MANO registration 的公开数据集。其价值不在于操作任务，而在于**手部形态和外观的先验知识** -- 这是一个基础性资源，可以被其他所有手部数据集/方法所利用 (更好的 hand mesh reconstruction, relightable avatar, cross-subject generalization)。

**对 robotics 的启示**: PALM 提供的 hand shape prior 可用于:
1. 从 MANO 到 robot hand 的 retargeting 时提供更准确的 source hand geometry
2. Vision-based 手部追踪中提供更强的 shape constraint
3. Sim2real 中渲染更逼真的人手图像用于域随机化
