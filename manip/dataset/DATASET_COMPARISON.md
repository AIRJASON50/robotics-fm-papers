# 手-物体交互数据集对比 (MANO/MediaPipe 格式)

聚焦 `hand_object/` 分类下的 5 个数据集：标准手格式 (MANO) + 物体资产 (mesh + 6D pose)。
非 MANO 格式的数据集 (robot_hand/, hand_only/) 见附录。

---

## 1. Executive Summary

**现有什么**: 5 个 MANO + 物体资产的动态轨迹数据集，总计约 50h+ 有效操作、500+ 物体、20K+ 轨迹。无静态 MANO 抓取数据集（但可从动态轨迹的稳定阶段提取）。

**缺什么**: (1) 真实传感器接触力 -- 零；仿真重建力仅 DexCanvas s02 有 ~10K 条但未完全开源。(2) 不同数据集格式碎片化（Parquet/NPY/Pickle/VRS），需统一转换。(3) In-hand manipulation 占比有限，大量数据是 pick-use-place。

**该做什么**: 数据格式统一（见 DATA_FORMAT_SPEC.md v0.3），优先接入 GigaHands + OakInk2 + ARCTIC 作为 MANO 种子数据，自采集补充接触力和 in-hand manipulation。

### 决策表

| 数据集 | 可获取性 | 有效规模 | 运动学精度 | 物体追踪精度 | 差异化价值 |
|--------|---------|---------|-----------|-------------|-----------|
| **DexCanvas** | 差 (1% preview, gated) | ~12K seq, 30 obj | 高 (MoCap 14 marker) | 高 (MoCap marker) | Cutkosky 21 类, RL 力重建 pipeline |
| **GigaHands** | 好 (Globus 公开) | 14K clips, 417 obj | 中 (无标记, HaMeR+三角化) | **待评估** (diff. rendering) | 最大规模, 双手, 84K 文本 |
| **OakInk2** | 好 (HuggingFace CC-BY-SA) | 70+ 类别, 120Hz | 高 (MoCap marker) | 高 (MoCap marker) | 任务层次结构, affordance, ManipTrans 上游 |
| **ARCTIC** | 中 (需注册 3 账号) | 339 seq, 10 obj | 高 (54 Vicon cameras) | 高 (MoCap marker) | 唯一铰接物体, 双手 |
| **HOT3D** | 好 (训练集公开) | 833 min, 33 obj | 高 (OptiTrack MoCap) | 高 (OptiTrack marker) | Egocentric multi-view, PBR mesh, eye gaze |

> 精度等级: **高** = 光学 MoCap marker 追踪 (亚毫米)；**中** = 无标记视觉方案 (数毫米)

---

## 2. 下载可用性验证

经 API/HTTP HEAD 实际验证，非论文自述。

| 数据集 | 实际可下载? | 核心数据大小 | 与论文一致? |
|--------|------------|-------------|------------|
| **DexCanvas** | 部分 (HuggingFace gated, 1% preview) | ~8 GB 可见 / 139 GB 存储 | **不一致**: 声称 7000h (RL 扩增，未发布) + 力数据 (未开源) |
| **GigaHands** | **是** (Globus 公开直链, 无需注册) | **~16 GB** (hand+object+text) / video TB 级 | 一致 (CC BY-NC 4.0) |
| **OakInk2** | **是** (HuggingFace, CC-BY-SA-4.0) | 多个 tarball (具体大小待确认) | 一致 (11.8K 月下载量) |
| **ARCTIC** | 是 (需注册 ARCTIC+SMPL-X+MANO) | **~803 GB** (images 649G + data 154G) | 一致 |
| **HOT3D** | **是** (训练集 GT 公开，测试集需评估服务器) | 数百 GB (3.7M+ images + mesh) | 一致 |

### 关键警示

1. **DexCanvas 接触力未开源**: HuggingFace 标注 "v0.1, contact force not included"。本地 `part_xxx.parquet` contact 字段全为 null。s02 的 ~10K 条 RL 仿真力数据可用，但非真实传感器测量。
2. **GigaHands 物体 6D pose 精度风险**: 物体追踪使用 differentiable rendering（非 MoCap marker），论文自述 "no existing method for object pose estimation worked well"。追踪成功率仅 18.7%（人工验证 IoU > 60%）。可用的 hand-object 轨迹实际为 3,356 条（非 14K 全量）。
3. **DexCanvas "7000h" 是 RL 扩增声称值**: 实际采集的 mocap 数据约 70h / 12K seq。100x 扩增数据未发布。

---

## 3. 有效数据量

| 数据集 | 有效轨迹数 | 有效操作时长 | 物体数 | 双手 | 帧率 | 存储 |
|--------|-----------|-------------|--------|------|------|------|
| **DexCanvas** | ~12K seq (mocap) + 9.8K (RL) | ~70h mocap | 30 | 否 | 100 Hz | ~8 GB / 139 GB (gated) |
| **GigaHands** | **3,356** (物体追踪成功的) | 34h 总量 (含手势/自接触) | 417 | **是** | - | ~16 GB core |
| **OakInk2** | 待确认 | 待确认 | 70+ 类别 | **是** | 120 Hz | HuggingFace |
| **ARCTIC** | 339 | ~数小时 | 10 铰接 | **是** | ~30 Hz | ~803 GB |
| **HOT3D** | ~3,832 (训练集) | 833 min | 33 刚性 | 是 | 30 Hz | 数百 GB |

> GigaHands 14K clips 含手势和自接触，物体交互成功追踪的仅 3,356 条。
> OakInk2 具体数字需从下载数据确认，是 ManipTrans/DexManipNet 的上游。

---

## 4. 每帧数据完整度

| 模态 | DexCanvas | GigaHands | OakInk2 | ARCTIC | HOT3D |
|------|-----------|-----------|---------|--------|-------|
| **MANO pose** | 45D axis-angle | MANO mesh + keypoints | 45D axis-angle (LH+RH) | 45D axis-angle (LH+RH) | MANO + UmeTrack |
| **MANO shape** | [10] per person | [10] per person | [10] per person | [10] per person | per-subject 手扫描 |
| **物体 6D pose** | euler (需转) | diff. rendering (精度待评估) | SE(3) 4x4 | axis-angle (mm, 需转m) | SE(3) rigid |
| **物体资产** | mesh + URDF | 3D mesh (310 扫描 + 31 生成) | mesh + affordance 部件 | 铰接 mesh (top/bottom) | mesh + PBR 材质 |
| **铰接** | 否 | 否 | 否 | **是** (1 DoF per object) | 否 |
| **接触力** | **仅 RL 仿真** (s02, ~10K) | 否 | 否 | 否 | 否 |
| **视觉** | 2cam RGBD 30Hz | 51 views | 多视角 30Hz | 9 views 2K | multi-view ego |
| **语言** | 否 | **84K text** | **有** (任务描述) | 否 | 否 |
| **额外** | Cutkosky 21 类 | NeRF, 56 subjects | Affordance, PDG, LLM | SMPL-X 全身 | Eye gaze, IMU, SLAM |

---

## 5. 逐数据集评估

### 5.1 DexCanvas -- Cutkosky 分类最系统, RL 力重建可复用

**可用数据**: ~12K 条 MANO mocap + 物体 6D + mesh/URDF。Cutkosky 21 种操作覆盖 power grasp (01-02)、precision grasp (03-15)、in-hand manipulation (16-21)。唯一系统化的操作分类学。

**接触力**: s02 包含 ~10K 条 RL 仿真重建的逐帧力 (finger_id + force_vector + multi-frame contact)。这些力是 Isaac Gym 物理引擎的精确输出，非网络预测。虽然非真实传感器测量，但物理一致性有保证。更重要的是，**RL 力重建 pipeline 是可迁移的方法论** -- 可应用到任何 MANO mocap 数据集。

**问题**: 仅 1% 预览可下载。物体仅 30 个。单手。

**项目定位**: (1) Cutkosky 操作分类参考；(2) RL 力重建方法论参考（非数据本身）；(3) 如完整开放则为核心种子数据。

### 5.2 GigaHands -- 最大规模, 但物体追踪质量需审慎

**可用数据**: 34h 双手活动，417 物体，56 名受试者，84K 文本标注。Globus 公开下载，~16 GB core。

**关键风险**: 物体 6D pose 来自 differentiable rendering（非 MoCap marker），人工验证 IoU > 60% 的追踪成功率仅 **18.7%**。实际有可靠物体 pose 的轨迹仅 3,356 条（非 14K）。无标记手部追踪 (HaMeR + multi-view triangulation) 精度低于光学 MoCap（数毫米 vs 亚毫米）。

**独特价值**: (1) 物体类别最多 (417)；(2) 双手 + 多视角 (51 views) + 文本标注三合一；(3) 活动类型多样 (cooking/crafting/office/entertainment/housework)。

**项目定位**: 大规模手部运动 prior + 文本标注。物体 pose 需筛选后使用（仅取 IoU > 60% 的 3,356 条）。

### 5.3 OakInk2 -- 任务结构最丰富, ManipTrans 上游

**可用数据**: 双手 MANO + SMPL-X，70+ 物体类别，120Hz MoCap，逐帧物体 SE(3)。3 层任务抽象 (Affordance -> Primitive Task -> Complex Task) + 文本描述 + Primitive Dependency Graph。HuggingFace 完全开放 (CC-BY-SA-4.0)。

**经下游验证**: ManipTrans/DexManipNet (CVPR 2025) 直接基于 OakInk2 构建，5+ 种机器人手 retarget 成功，说明数据质量经多个下游项目检验。

**独特价值**: (1) 唯一有任务层次结构的数据集；(2) 物体 affordance + 部件分解标注；(3) 完整的 MANO->robot hand retargeting pipeline 验证。

**项目定位**: 核心种子数据。任务结构可直接用于 RL reward 设计。

### 5.4 ARCTIC -- 铰接物体交互唯一

**可用数据**: 339 条双手铰接物体交互轨迹，MANO+SMPL-X+物体位姿+铰接状态。注册后完整下载。

**独特价值**: 唯一的双手铰接物体数据。被 ArtiGrasp、DexMachina、ObjDexEnvs、ManipTrans、SPIDER 5+ 篇后续论文直接使用。

**局限**: 仅 10 物体，规模小。~803 GB 以图像为主，纯运动学数据很小。

**项目定位**: 铰接物体操作的种子数据。规模小但独特性高。

### 5.5 HOT3D -- Egocentric tracking benchmark

**可用数据**: 833 min egocentric 多视角，1.16M 有效标注帧，MANO + UmeTrack 双格式，33 刚性物体 + PBR mesh。训练集 GT 公开。

**独特价值**: 唯一使用真实头显 (Aria + Quest 3) 录制。eye gaze + SLAM 点云 + IMU。PBR 物体模型可做 photo-realistic 渲染。与 BOP Challenge 生态打通。

**局限**: 操作偏简单 (pick-use-place)。核心定位是 vision tracking benchmark，非 RL/BC 训练数据。VRS 格式需通过专用 toolkit 加载。

**项目定位**: egocentric perception 参考。如需第一人称操作数据可接入。

---

## 6. Gap Analysis

### 结构性缺失（不可通过简单转换解决）

1. **真实传感器接触力: 零。** 全部 5 个数据集均无来自触觉传感器 (GelSight/DIGIT) 或力/力矩传感器的真实测量数据。DexCanvas s02 有 ~10K 条物理仿真重建力（Isaac Gym 精确输出，非网络预测），是当前最接近的替代品，但非真实传感器测量。
2. **不同模态分散在不同数据集中。** MANO + 物体 6D 在 DexCanvas/OakInk2/ARCTIC 有，语言标注在 GigaHands/OakInk2 有，视觉在 HOT3D/GigaHands 有，但没有一个数据集同时覆盖。需要通过统一格式 (DATA_FORMAT_SPEC.md) 来联合使用。
3. **In-hand manipulation 数据有限。** 仅 DexCanvas (types 16-21) 系统覆盖手内操作（finger gaiting/rotation/rolling/sliding/transfer/reorientation），其他数据集以 pick-use-place 为主。

### 可解决的工程挑战

4. **格式碎片化**: 5 种文件格式 (Parquet/NPY/Pickle/VRS/NPY+video)，需统一转换到 HDF5 shard。见 DATA_FORMAT_SPEC.md 第 8 节转换要点。
5. **坐标系和单位不统一**: DexCanvas 用米/euler，ARCTIC 用毫米/axis-angle，OakInk2/HOT3D 用 SE(3) 4x4。一次性转换代码即可解决。
6. **hand_object/static/ 为空**: 可从现有动态轨迹的稳定抓取阶段提取（DexCanvas 的 `object_move_start/end_frame` 外的 idle frames 即静态抓取姿态）。

### 接触力的三层现状

| 层级 | 数据集 | 可用性 | 质量 |
|------|--------|--------|------|
| 真实传感器测量 | **无** | - | - |
| 物理仿真重建 | DexCanvas s02 (~10K) | 部分开源 | Isaac Gym 精确输出，物理一致 |
| 几何/距离推导 | ARCTIC (3mm contact map), GigaHands (IoU-based) | 可复现 | 二值判定，无力大小 |

> DexCanvas 的 RL 力重建 pipeline 是可迁移方法论，可应用到 OakInk2、ARCTIC 等 mocap 数据集，为其补充仿真力数据。

---

## 7. 行动建议

### 数据接入优先级

| 优先级 | 数据集 | 原因 | 预期转换工作量 |
|--------|--------|------|--------------|
| P0 | **OakInk2** | 经下游验证、任务结构丰富、HuggingFace 开放、CC-BY-SA | pickle -> HDF5，MANO key 去前缀 |
| P0 | **ARCTIC** | 铰接物体唯一、被广泛引用、数据质量高 | NPY -> HDF5，mm->m，shape 去重 |
| P1 | **GigaHands** (筛选后) | 规模大但需筛选 IoU > 60% 子集 (~3,356 条) | pose JSON -> HDF5，手部精度需评估 |
| P1 | **DexCanvas** (如完整开放) | Cutkosky 分类参考、RL 力重建参考 | Parquet -> HDF5，euler -> axis-angle |
| P2 | **HOT3D** | 定位偏 perception，操作偏简单 | VRS -> HDF5，需专用 toolkit |

### 格式统一

所有转换输出遵循 DATA_FORMAT_SPEC.md v0.3:
- HDF5 shard 结构，manifest.json 索引
- MANO pose [T,45] + global_rotation [T,3] + translation [T,3] + shape [10]
- 物体 rotation 统一为 axis-angle（存储）/ SE(3) 4x4（二选一，标注 layout attr）
- 单位统一: 米 + 弧度 + 秒

### 自采集补充方向

| 缺口 | 建议 | 量级 |
|------|------|------|
| 接触力 | 触觉传感器 (GelSight/DIGIT) + MoCap 同步采集 | 先 100 条 pilot |
| In-hand manipulation | 参考 DexCanvas Cutkosky 16-21，在自有灵巧手上采集 | 按物体类型 x 操作类型的矩阵 |
| 物体多样性 | 从 GigaHands 417 物体或 DexGraspNet 5355 物体中选取代表性子集 | 50-100 物体 |

---

## 附录: 非 MANO 数据集摘要

| 数据集 | 分类 | 手部格式 | 规模 | 为何不在核心范围 |
|--------|------|---------|------|-----------------|
| DexGraspNet | robot_hand/static | ShadowHand 28D | 1.32M grasps, 5355 obj | 非 MANO，仅静态 |
| DexManipNet | robot_hand/dynamic | 多手型 joints | 3.3K eps, 61 tasks | Retarget 后格式，非原始 MANO (上游 OakInk2 是) |
| RealDex | robot_hand/dynamic | ShadowHand 22D | 2.6K seq, 52 obj | 非 MANO，唯一真实机器人数据 |
| DexMimicGen | robot_hand/dynamic | 机器人 joints | 21K demos, 9 tasks | 非 MANO，仿真扩增方法论 |
| PALM | hand_only | MANO | 263 subjects, 13K scans | 有 MANO 但无物体 (手部形态 prior) |
| DexCap | hand_only | 人手 joints | ~90 min | 无 MANO params, 无物体 |
| EgoDex | hand_only | ARKit skeleton | 829h, 338K eps | 非 MANO，无物体 |
