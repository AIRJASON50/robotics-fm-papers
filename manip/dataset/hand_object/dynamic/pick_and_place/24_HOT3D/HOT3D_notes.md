# HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos -- Notes

> Paper: Banerjee et al., Meta Reality Labs, 2024 (arXiv: 2411.19167)
> Code: https://github.com/facebookresearch/hot3d (Apache 2.0)
> Dataset: https://www.projectaria.com/datasets/hot3D/ (HOT3D License)

---

## 1. Core Problem

HOT3D 要解决的核心问题是: **缺乏一个大规模、多视角、来自真实头戴设备的 egocentric (第一人称视角) 手-物交互数据集, 用于训练和评估 3D hand-object tracking 方法**.

现有数据集的主要缺陷:

| 缺陷 | 典型代表 | 说明 |
|------|----------|------|
| 非 egocentric 视角 | DexYCB, HO-3D, ARCTIC | 使用 exocentric 相机阵列, 与 AR/VR 应用场景不匹配 |
| 单视角采集 | HOI4D, FHPA | 无法研究 multi-view 方法, 而头戴设备天然具备多相机 |
| 非真实头戴设备 | ARCTIC | 使用头盔安装的模拟相机, 与实际 AR/VR 设备有 domain gap |
| GT (Ground Truth, 真值) 精度不足 | HO-Cap, HOI4D | 基于 RGB-D 优化而非 MoCap (Motion Capture, 动作捕捉), 精度有限 |
| 交互场景单一 | ContactPose, DexYCB | 仅静态抓取或简单 pick-and-place, 缺少日常生活场景 |
| 缺少物体模型 | Ego4D | 无 3D mesh 和 PBR (Physically Based Rendering, 基于物理的渲染) 材质 |

HOT3D 的核心定位: **第一个**同时满足以下条件的数据集 -- (1) 多视角 egocentric 视频, (2) 来自真实消费级/研究级头戴设备 (Aria + Quest 3), (3) 高精度 MoCap GT, (4) 丰富日常交互场景.

---

## 2. Method Overview

HOT3D 本质上是一个 **dataset paper**, 方法部分聚焦在数据集构建和 baseline 实验两方面.

### 2.1 数据集构建

```
Data Pipeline:
  MoCap Lab (OptiTrack IR cameras + light diffusers)
       |
       v
  19 subjects x 33 objects x 4 scenarios (inspect/kitchen/office/living room)
       |
       v
  425 recordings (~2 min each), 833 min total
       |
       +---> Aria: 198 recordings (1 RGB 1408x1408 + 2 mono 640x480, 30fps)
       |          + SLAM point cloud + eye gaze
       |
       +---> Quest 3: 226 recordings (2 mono 1280x1024, 30fps)
       |
       v
  GT Annotation: optical markers -> MoCap -> rigid body poses
       |
       +---> Hand poses: UmeTrack format + MANO format
       +---> Object poses: 6DoF rigid transformations (T_world_object)
       +---> Headset poses: 6DoF trajectory
       |
       v
  QA: 1.16M / 1.5M frames pass visual inspection
       |
       v
  HOT3D-Clips: 3832 curated 5-sec clips (150 frames each)
```

### 2.2 Baseline 实验

论文设计了四个 benchmark task 来验证 multi-view 数据的价值:

| Task | Method | Single-view | Multi-view |
|------|--------|-------------|------------|
| 3D hand pose tracking | UmeTrack | 15.4 MKPE | 10.9 MKPE (41% improvement) |
| 6DoF object pose estimation | FoundPose (extended) | ~35% recall | ~47% recall (13-34% relative gain) |
| 2D in-hand object segmentation | MRCNN-DA (Mask R-CNN + Depth Anything) | -- | best mIoU |
| 3D lifting of in-hand objects | StereoMatch (DINOv2 feature matching) | MonoDepth baseline | clear advantage at <10cm threshold |

核心结论: **multi-view 方法在所有任务上显著优于 single-view 方法**, 这对 AR/VR 设备上的视觉算法设计有直接指导意义.

---

## 3. Key Designs

### 3.1 Multi-View Egocentric 数据采集: 双设备策略

HOT3D 使用两款 Meta 头戴设备采集数据, 这是一个关键设计决策:

| 特性 | Project Aria | Quest 3 |
|------|-------------|---------|
| 定位 | AI 研究原型 (轻量 AR 眼镜) | 消费级 VR 头显 (百万级出货量) |
| 相机配置 | 1 RGB (1408x1408) + 2 mono (640x480) | 2 mono (1280x1024) |
| 额外传感器 | SLAM point cloud, eye gaze | -- |
| 录制数量 | 198 recordings | 226 recordings |
| 同步方式 | 硬件触发 (hardware trigger) | 硬件触发 |

这一设计的意义:
- **研究覆盖面**: Aria 提供 RGB+mono+eye gaze 的丰富多模态数据, 适合探索 foveated sensing (注视引导感知); Quest 3 代表真实消费场景的约束条件
- **多视角的实际可行性**: 证明 multi-view 方法不需要额外硬件, 现有 AR/VR 设备的相机配置已经足够
- **Cross-sensor generalization**: FoundPose 实验表明 DINOv2 backbone 可以跨 RGB/mono 传感器泛化

### 3.2 高精度 GT 标注: MoCap + Dual Hand Representation

GT 采集使用被动光学标记 + OptiTrack 红外 MoCap 系统, 比基于 RGB-D 优化的方案精度更高.

手部标注同时提供两种格式:
- **UmeTrack format**: Meta 自研的手部追踪格式, 基于 skeleton 的 FK (Forward Kinematics, 正向运动学) + skinning. 精度更高, 因为 UmeTrack 模型直接从被扫描的手部形状构建
- **MANO (hand Model with Articulated and Non-rigid defOrmations, 关节化非刚性变形手部模型) format**: 社区标准的参数化手部模型. 通用性更好, 与其他数据集兼容

代码实现中的关键细节:
- `UmeTrackHandDataProvider`: 加载 per-subject 的 hand profile (skeleton + mesh), 通过 `skin_vertices()` / `skin_landmarks()` 做 LBS (Linear Blend Skinning, 线性混合蒙皮)
- `MANOHandDataProvider`: 使用 `MANOHandModel.forward_kinematics()`, 需要额外的 `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` 模型文件
- UmeTrack 模型定义为左手, 右手通过翻转 X 轴实现 (`hand_wrist_pose_tensor[:, 0] *= -1`)

### 3.3 Multi-View FoundPose: 简洁有效的多视角扩展

论文对 FoundPose (一种基于 DINOv2 (Self-Distillation with No Labels v2, 自蒸馏视觉特征模型) 的 training-free 6DoF pose estimation 方法) 做了多视角扩展, 设计简洁但效果显著:

**原始 FoundPose (单视角)**:
1. Onboarding: 渲染 RGB-D templates -> 提取 DINOv2 patch features -> 注册到 3D
2. Inference: crop query image -> BoW (Bag of Words, 词袋) 检索相似 templates -> PnP-RANSAC 求解 pose

**Multi-View Extension (多视角扩展)**:
1. Onboarding: 不变
2. Inference 变化:
   - 在**所有视角**中 crop 物体
   - 用 **per-view score 之和**检索 templates (而非单视角 score)
   - 建立 **multi-view 2D-3D correspondences**
   - 用 **Generalized PnP (gPnP, 广义 PnP)** 代替标准 PnP 求解 pose

这个扩展几乎是 "free lunch": 不需要重新训练, 仅修改推理流程, 就获得 8-12% 的 recall 提升. 对于 robotics 的启示: 多视角几何约束是一种低成本、高回报的策略.

---

## 4. Experiments

### 4.1 3D Hand Pose Tracking

| Training Data | Eval on UmeTrack (1-view) | Eval on UmeTrack (2-view) | Eval on HOT3D (1-view) | Eval on HOT3D (2-view) |
|--------------|--------------------------|--------------------------|----------------------|----------------------|
| UmeTrack only | 13.6 | 15.9 | 24.2 | 31.8 |
| HOT3D only | 23.7 | 27.0 | 18.0 | 21.2 |
| UmeTrack + HOT3D | 13.4 | **9.5** | 15.4 | **10.9** |

MKPE (Mean Keypoint Position Error, 平均关键点位置误差) in mm, 越低越好.

关键发现:
- 单数据集训练存在严重 domain gap (手-手交互 vs 手-物交互; Quest 2 vs Quest 3 相机配置)
- 联合训练有效消除 domain gap, 且 multi-view 带来 **41%** 的精度提升
- 训练时的 random view masking 策略是使 single-view 和 multi-view 模式公平比较的关键

### 4.2 6DoF Object Pose Estimation

| Method | Aria (5cm, 5deg) | Aria (10cm, 10deg) | Quest3 (5cm, 5deg) | Quest3 (10cm, 10deg) |
|--------|-----------------|-------------------|-------------------|---------------------|
| FoundPose (1-view) | ~35% | ~60% | ~24% | ~47% |
| FoundPose (multi-view) | ~47% | ~68% | ~36% | ~55% |

Recall rate, 越高越好. Multi-view 在 Aria 和 Quest 3 上均显著优于 single-view.

### 4.3 2D In-Hand Object Segmentation

| Method | Aria Train mIoU | Aria Test mIoU | Quest3 Train mIoU | Quest3 Test mIoU |
|--------|----------------|---------------|-------------------|-----------------|
| EgoHOS | 49.5 | 46.1 | 30.3 | 27.6 |
| MRCNN | 59.1 | 59.3 | 42.3 | 42.1 |
| MRCNN-DA | **64.5** | **63.2** | **50.1** | **48.7** |

mIoU (mean Intersection over Union, 平均交并比), 越高越好. Depth Anything V2 的深度预测可显著提升前景-背景分离.

### 4.4 3D Lifting of In-Hand Objects

| Method | 5cm recall | 10cm recall | 20cm recall | 30cm recall |
|--------|-----------|------------|------------|------------|
| HandProxy | 7.1 | 25.3 | 62.8 | 80.9 |
| MonoDepth (GT mask) | 30.1 | 55.1 | 74.1 | 81.0 |
| StereoMatch (GT mask) | **42.7** | **65.3** | **79.1** | **85.7** |

Recall rate on Aria, 越高越好. StereoMatch 在严格阈值 (<10cm) 下显著优于其他方法.

---

## 5. Related Work Analysis

论文对 hand-object interaction 数据集做了系统梳理, 按 GT 获取方式分为三类:

| GT 获取方式 | 代表数据集 | 优点 | 缺点 |
|------------|-----------|------|------|
| 手工标注 | Sridhar et al. | 灵活 | 不可扩展, 精度有限 |
| 传感器辅助 | FHPA (磁传感器), ContactPose (热成像) | 提供额外物理信号 | 影响手/物体外观 |
| RGB-D 优化 | HO-3D, DexYCB, HOI4D, H2O | 接近全自动 | 精度受限于 depth 传感器质量 |
| MoCap 系统 | ARCTIC, **HOT3D** | 最高精度 | 需要专用实验室, 光学 marker 在手上 |

HOT3D 与 ARCTIC 是 MoCap 精度阵营的两个代表, 但 HOT3D 独特之处在于使用真实头戴设备而非模拟设备.

论文对 BOP Benchmark (Benchmark for 6D Object Pose, 6D 物体姿态基准) 的关系也值得注意: HOT3D-Clips 已经转换为 BOP 格式, 直接嵌入 BOP Challenge 2024, 扩展了 BOP 在 egocentric hand-object 场景的覆盖.

---

## 6. Limitations & Future Directions

### 论文承认的局限

| 局限 | 具体说明 |
|------|---------|
| 仅刚性物体 | 33 个物体全部是 rigid body, 无 articulated (关节化) 或 deformable (可变形) 物体 |
| 实验室环境 | 虽然家具和光照做了随机化, 但仍在固定 MoCap lab 中拍摄, 不是 in-the-wild |
| MoCap marker 可见 | 手和物体上的光学标记在图像中可能可见, 引入 domain gap |
| 无 contact 信息 | 没有手-物接触力或接触区域标注 |
| 测试集 GT 不公开 | 需要通过评估服务器提交, 增加了使用门槛 |

### 未来方向 (笔记推断)

1. **Articulated object 扩展**: ARCTIC 已证明关节化物体 (水壶、笔记本电脑等) 的需求, HOT3D 可以沿此方向补充
2. **In-the-wild 采集**: 从 lab 走向真实环境, 可能需要 SLAM-based pseudo GT 或 multi-view optimization
3. **Contact + force annotation**: dataset 领域最大的空白, 对 sim2real transfer 至关重要
4. **Eye gaze 利用**: 论文提及但未深入实验 -- 用 gaze 预测 intention 或做 foveated compute allocation
5. **Temporal modeling**: 当前 baseline 主要是 per-frame 方法, 缺少对时序信息的利用 (tracking vs detection)

---

## 7. Paper vs Code Discrepancies

通过对比论文描述与代码仓库 (`hot3d/` toolkit), 发现以下差异:

| 项目 | 论文描述 | 代码实现 | 影响 |
|------|---------|---------|------|
| FoundPose multi-view extension | 详细描述了 gPnP + multi-view correspondence | **代码未开源**, 仓库中没有 FoundPose 相关代码 | 无法复现核心 object pose 实验 |
| StereoMatch / MonoDepth | 详细描述了 DINOv2 stereo matching 和 Depth Anything V2 lifting | **代码未开源**, 仓库中没有 3D lifting 相关代码 | 无法复现 3D lifting 实验 |
| MRCNN-DA (in-hand segmentation) | 训练在 400K proprietary Aria images 上 | **训练数据和模型均未公开** | 无法复现 segmentation baseline |
| UmeTrack 训练代码 | 论文描述了训练配置 (random view masking 等) | 仓库仅提供 data loading + visualization, **不含训练代码** | 无法复现 hand tracking 实验 |
| Mask files | 论文提及 QA filtering | 代码提供了完整的 mask loading API (`loader_masks.py`), 支持 AND/OR 组合, 7 种 mask 类型 | 代码比论文描述更详细 |
| HOT3D-Clips | 论文简要提及 | 代码提供了完整的 Webdataset + BOP 格式转换工具 (`clips/bop_format_converters/`), 以及与 `hand_tracking_toolkit` 的集成 | 代码比论文更完善 |
| 数据版本 | 论文未提及版本迭代 | `VERSIONS.md` 记录了 v1->v2->v3 的更新 (v2 修正 MANO 拇指, v3 更新 visibility scores) | 论文发表时可能在 v2, 当前已 v3 |

**总结**: 仓库定位是 **dataset toolkit** (数据加载 + 可视化), 不是方法复现库. 论文的所有 baseline 方法 (FoundPose extension, StereoMatch, MonoDepth, MRCNN-DA, UmeTrack training) 均未在仓库中提供, 且 segmentation 训练数据是 proprietary 的.

代码架构一览:

```
hot3d/hot3d/
├── dataset_api.py           # Hot3dDataProvider: top-level API, dispatches Aria/Quest3
├── viewer.py                # Rerun-based 3D visualization
├── Hot3DVisualizer.py       # Visualization utilities
├── render_3d.py             # 3D rendering helpers
├── data_loaders/
│   ├── AriaDataProvider.py  # Aria VRS + MPS (SLAM/eye gaze) data loading
│   ├── QuestDataProvider.py # Quest 3 VRS + pyvrs data loading
│   ├── ManoHandDataProvider.py      # MANO FK + skinning
│   ├── UmeTrackHandDataProvider.py  # UmeTrack FK + skinning
│   ├── ObjectPose3dProvider.py      # Object 6DoF pose loading from CSV
│   ├── loader_object_library.py     # Object mesh library (GLB format)
│   ├── PathProvider.py      # Headset-specific file path resolution
│   ├── frameset.py          # Multi-stream timestamp synchronization
│   ├── loader_masks.py      # QA mask loading + combination
│   └── ...
├── clips/                   # HOT3D-Clips (curated 5-sec clips)
│   ├── clip_util.py         # Clip loading (tar/webdataset format)
│   ├── vis_clips.py         # Clip visualization
│   └── bop_format_converters/  # HOT3D -> BOP format conversion
└── data_downloader/         # Dataset download utilities
```

---

## 8. Cross-Paper Comparison

### 8.1 与同类 Hand-Object 数据集的定量对比

| 维度 | HOT3D | ARCTIC | GigaHands | DexCap | Ego4D |
|------|-------|--------|-----------|--------|-------|
| **年份** | 2024 | 2023 (CVPR) | 2024 | 2024 | 2022 (CVPR) |
| **来源** | Meta Reality Labs | ETH Zurich + MPI | KAIST | Stanford | Meta AI |
| **总时长** | 833 min | ~56 min (339 seq) | 34 hours | ~90 min | 3670 hours |
| **总图像数** | 3.7M+ | 2.1M | ~3.6M frames | -- | 数百万 |
| **被试数** | 19 | 10 | 56 | 少量 | 931 |
| **物体数** | 33 (rigid) | 11 (articulated) | 417 | 少量 | N/A |
| **视角** | Egocentric multi-view | 8 exo + 1 ego (helmet) | Exocentric multi-view (50-60 cameras) | Egocentric (GoPro) | Egocentric (single) |
| **头戴设备** | 真实 (Aria + Quest 3) | 模拟 (helmet camera) | 无 (外部相机阵列) | GoPro | GoPro / diverse |
| **GT 来源** | MoCap (OptiTrack) | MoCap (Vicon) | MoCap (多系统) | 手套 (LEAP Motion) | 无 3D GT |
| **手部格式** | UmeTrack + MANO | MANO + SMPL-X | MANO | 手指关节 (retarget) | -- |
| **物体模型** | 3D mesh + PBR 材质 | 3D mesh | 3D scanned mesh | -- | -- |
| **交互类型** | pick-place + 日常场景 | 双手操作关节化物体 | 多样 (bimanual) | wiping, packaging | 日常活动 (无 3D) |
| **Eye gaze** | 有 (Aria) | 无 | 无 | 无 | 无 |
| **Contact data** | 无 | 无 | 无 | 无 | 无 |
| **许可** | HOT3D License | 注册制 | CC-BY-NC | HuggingFace | Ego4D License |

### 8.2 定性分析

**HOT3D vs ARCTIC**:
- ARCTIC 聚焦 **articulated object interaction** (双手开合笔记本、倒水壶等), 物体有内部自由度, 更接近灵巧操作
- HOT3D 聚焦 **rigid object + real headset**, 物体更多样 (33 vs 11) 但全部刚性, 核心卖点是 multi-view egocentric
- 两者 GT 精度相当 (都用 MoCap), 但 ARCTIC 的 egocentric view 来自头盔模拟, 有 domain gap
- 对 robotics 的启示: 如果目标是 dexterous manipulation policy (灵巧操作策略), ARCTIC 的 articulated object data 更有价值; 如果目标是 egocentric perception for AR/VR, HOT3D 更贴近真实部署场景

**HOT3D vs GigaHands**:
- GigaHands 是目前 **最大规模的 hand-object 数据集** (34 hours, 417 objects, 56 subjects), 数据多样性远超 HOT3D
- GigaHands 使用 exocentric multi-view (50-60 cameras) 重建, 不是 egocentric, 与 AR/VR 应用有 gap
- GigaHands 的物体包含 rigid + non-rigid, 覆盖面更广
- 对 robotics 的启示: GigaHands 更适合训练 general-purpose hand-object interaction model; HOT3D 更适合 egocentric deployment

**HOT3D vs DexCap**:
- DexCap 的核心价值在于 **end-to-end pipeline** (人手采集 -> retarget -> 机器人部署), 而非 dataset 本身
- DexCap 使用 LEAP Motion 手套, 不提供 MANO 标注, 难以与其他 hand-object dataset 互操作
- DexCap 的数据量小 (~90 min), 场景有限 (wiping, packaging)
- 对 robotics 的启示: DexCap 和 HOT3D 处于不同层面 -- DexCap 是采集-部署 pipeline, HOT3D 是 perception benchmark

**HOT3D vs Ego4D**:
- Ego4D 是 **最大规模的 egocentric 视频数据集** (3670 hours, 931 participants), 但不提供 3D hand/object pose GT
- Ego4D 的标注主要是 activity recognition、narration、state change 等高层语义, 不适合 3D tracking
- HOT3D 可以看作 Ego4D 在 3D hand-object tracking 维度的精细化补充
- 对 robotics 的启示: Ego4D 用于学习 high-level activity understanding; HOT3D 用于 low-level 3D pose tracking

### 8.3 Robotics 视角的 Takeaway

| # | Takeaway | 依据 | 行动项 |
|---|----------|------|--------|
| 1 | Multi-view 是低成本高回报的精度提升策略 | 所有 task 上 multi-view 显著优于 single-view (hand tracking 41%, object pose 13-34%) | 在灵巧手 teleoperation 系统中, 优先利用多相机而非单相机 |
| 2 | DINOv2 features 可跨传感器/跨模态泛化 | FoundPose 在 RGB + mono 混合输入上表现良好, StereoMatch 用 DINOv2 做跨视角匹配 | 在 robot perception 中, 考虑用 DINOv2 作为 sensor-agnostic feature backbone |
| 3 | Egocentric multi-view dataset 是 sim2real 的桥梁 | PBR 物体模型可用于 photo-realistic 训练数据渲染 (BlenderProc), 缩小 sim-real gap | 利用 HOT3D 的 PBR 模型生成 sim 训练数据 |
| 4 | Contact data 仍然是 dataset 领域最大的空白 | HOT3D/ARCTIC/GigaHands/DexCap 均无 contact force 标注 | 未来 dataset 工作需要集成 tactile/force sensing |
| 5 | UmeTrack vs MANO 的 dual representation 值得借鉴 | UmeTrack 更精确 (per-subject scanning), MANO 更通用 (可跨 dataset 迁移) | 在自己的 dataset 中同时提供 task-specific 和 standard 两种格式 |
