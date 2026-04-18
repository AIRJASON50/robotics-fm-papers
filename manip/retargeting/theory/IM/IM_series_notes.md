# Interaction Mesh 系列笔记

> 专题: 基于 Interaction Mesh 的运动重定向 (2004-2024)
> 关注点: Laplacian 数学基础、拓扑构建方式、权重设计演变、跨体耦合处理
> 与手部 retargeting 项目的关联: 理解 IM 的设计选择和各代改进

---

## 演变链

```
Sorkine 2004 (EUROGRAPHICS)         ← 数学基础: Laplacian 坐标用于 mesh 形变
Sorkine 2006 (Computers & Graphics) ← 综述: 微分坐标族系统化
    │
    └── Ho 2010 (SIGGRAPH/TOG)       ← 引入: IM 构造 + Laplacian 迁移到 retargeting
            ├── Nakaoka 2012          ← 机器人首次应用, 原样继承
            ├── Kim 2016              ← 偏转: 体积空间映射, 余切权重
            ├── Zhang 2023            ← 关键演进: 边向量替代 Laplacian 坐标, 指数距离权重
            └── Jang 2024             ← 最新: 几何感知 Transformer, anchor loss

OmniRetarget 2025 (arXiv)           ← 工程化: 均匀权重(退化), 每帧重建, 加硬约束
本项目                               ← 当前: 逐步复现并改进上述各选择
```

**关键区分**: Sorkine 发明了 Laplacian 形变作为 CG 工具；Ho 2010 发明了 Interaction Mesh 结构并将 Laplacian 引入 retargeting。两者贡献不同层次。

---

## 各论文详解

### Sorkine 2004 — Laplacian 坐标用于 mesh 编辑 (数学基础)

**论文**: Laplacian Surface Editing
**会议**: EUROGRAPHICS Symposium on Geometry Processing 2004
**文件**: `04_Sorkine_LaplacianEditing/`

**核心操作**:

```
δ_i = p_i - Σ_j w_j · p_j        (Laplacian 坐标 = "细节向量")

优化: 给定用户移动的控制点, 最小化 ||L·p' - δ||²
     → 保持每个顶点相对于其邻域重心的"偏移方向和幅度"
```

**应用场景**: 静态 3D mesh 的交互编辑 (drag-and-deform)。均匀权重。

**关键性质**:
- δ_i 是 3D 向量, 不是标量
- 对全局旋转敏感 (Laplacian 在世界坐标系, 不随形变旋转)
- 邻域是 mesh 的 1-ring 邻居 (连接边定义的邻居), 不是 Delaunay

**与 IM 的关系**: Ho 2010 直接使用了这套 Laplacian 坐标公式, 将 "mesh 1-ring 邻居" 替换为 "Delaunay 四面体邻居", 将 "静态 mesh 编辑" 替换为 "跨体运动 retargeting"。

---

### Sorkine 2006 — 微分表示综述

**论文**: Differential Representations for Mesh Processing
**期刊**: Computer Graphics Forum (Computers & Graphics), 2006
**文件**: `06_Sorkine_DiffRepresentations/`

系统化了以 Laplacian 坐标为代表的一族"微分坐标"方法, 包括：
- 梯度域编辑 (gradient-domain editing)
- 旋转无关的 Laplacian (ARAP 前身)
- 各种权重方案 (均匀、余切、测地线)

**对 IM 系列的意义**: Kim 2016 使用的余切权重来自这套框架；Zhang 2023 的旋转不敏感性动机也在这里。

---

### Ho 2010 — Interaction Mesh 奠基

**论文**: Spatial Relationship Preserving Character Motion Adaptation
**会议**: SIGGRAPH 2010 / ACM TOG 29(4)
**作者**: Edmond Ho, Taku Komura, Chiew-Lan Tai (Edinburgh)
**文件**: `10_Ho/`

**核心贡献**: 不是 Laplacian 坐标本身, 而是 **Interaction Mesh 结构**：

> 将角色骨架关节点 **和** 交互对象（物体表面、环境接触点）合并进同一个 Delaunay 四面体网格

这个构造让人与物的空间关系被同一个拓扑结构编码, 然后用 Sorkine 的 Laplacian 能量来约束 retargeting 过程保持这个关系。

**Laplacian 权重**: 距离倒数 w_j = 1/dist(i,j), 近邻权重大 (不是均匀权重)

**拓扑**: 每帧 Delaunay, 但论文建议固定首帧 (逐帧重建导致 "gradual drifting")

**优化**: 全序列联合 spacetime 优化 (不是逐帧贪心)

**局限**: Laplacian 旋转敏感; 骨架点稀疏时邻域均值语义弱

---

### Nakaoka 2012 — 机器人首次应用

**论文**: Interaction Mesh Based Motion Adaptation for Biped Humanoid Robots
**会议**: IEEE Humanoids 2012
**文件**: `12_Nakaoka/`

**与 Ho 2010 的区别**:
- IM 公式完全继承, 权重和拓扑没有改动
- 新增: marker-edge 模型将 IM 顶点映射到机器人关节
- 新增: 阻尼最小二乘 IK 求解关节角
- 新增: ZMP 动平衡调整

**对本项目的启示**: 全身场景下 1/d 权重的作用被掩盖 (四肢间距大, 权重差异显著)。手部场景中关节间距集中在 10-80mm, 1/d 和均匀权重差异很小。

---

### Kim 2016 — 体积映射路线

**论文**: Retargeting Human-Object Interaction to Virtual Avatars
**期刊**: IEEE TVCG 22(11), 2016
**文件**: `16_Kim/`

**根本不同**: 不在身体关键点做 IM, 而是在**物体周围空间**建体积网格

```
离线预计算: 在 source/target 物体周围建四面体网格, 最小化 Dirichlet 能量 (余切权重) 求双射映射 f
在线运行: 查询关节点在 source 网格中的重心坐标, 通过 f 映射到 target
```

**权重**: 余切权重 (Dirichlet 能量), 几何意义更强
**局限**: 需要明确物体; 无法处理纯手部场景

---

### Zhang 2023 — 关键演进: 边向量替代 Laplacian

**论文**: Simulation and Retargeting of Complex Multi-Character Interactions
**会议**: SIGGRAPH 2023
**文件**: `23_Zhang/`

**与 Ho 的两个关键区别** (论文明确称为 "major difference"):

**1. 比较单元: Laplacian 坐标 → 直接边向量**

```
Ho 2010:    δ_i = p_i - Σ(w_j · p_j)        (聚合: 一点对多邻居的加权均值偏移)
Zhang 2023: e_ij = p_j - p_i                  (直接: 一对一, 每条边独立)

cost = Σ_{(i,j)} w_ij · ||(e_robot_ij - e_ref_ij) / ||e_ref_ij|| ||²
```

归一化 /||e_ref|| 消除跨具身骨骼比例差异 (无量纲比值)。

**核心优势**: 每条边只涉及两端点, 不会被第三方远邻污染。

**2. 权重: 距离倒数 → 指数衰减**

```
w_ij = exp(-k_w · dist(i,j))    (近距离对: 权重→1; 远距离对: 权重→0)
```

软性过滤, 无需手动选择 topology。

**用途**: RL reward, 配合 PD controller 的关节空间正则使用。不是单独的优化目标。

---

### Jang 2024 — 几何感知 Transformer

**论文**: Geometry-Aware Retargeting for Two-Skinned Characters Interaction
**期刊**: ACM TOG (SIGGRAPH Asia 2024)
**文件**: `24_Jang/`

- Spatio Cooperative Transformer (SCT): intra + cross attention 双角色联合建模
- Anchor loss: 自动提取接触 anchor 点, 维持两角色接触距离
- 需要 skinned mesh (SMPL), 不适用于本项目的 21 点稀疏骨架

---

## 关键设计选择对比

| 设计维度 | Sorkine 04 | Ho 2010 | Nakaoka 12 | Kim 2016 | Zhang 2023 | OmniRetarget |
|---------|-----------|---------|-----------|---------|-----------|-------------|
| 比较单元 | Laplacian δ | Laplacian δ | Laplacian δ | 体积重心坐标 | **边向量** | Laplacian δ |
| 权重 | 均匀 | **1/d** | 同 Ho | 余切 | **指数衰减** | **均匀(退化)** |
| 拓扑来源 | mesh 1-ring | Delaunay | Delaunay | 预计算四面体 | Delaunay | Delaunay |
| 拓扑更新 | 固定 | 建议固定 | 每帧重建 | 固定(离线) | 每帧重建 | 每帧重建 |
| 优化方式 | 线性系统 | 全序列联合 | 逐帧 IK | 在线查询 | RL reward | 逐帧 SQP |

**OmniRetarget 的退化路径**: Ho 的 1/d 权重 → 均匀权重(更差); 固定拓扑建议 → 每帧重建。

---

## 对手部 Retargeting 项目的启示

**核心问题**: 21 稀疏手部关键点上, Delaunay 产生跨手指长程边, Laplacian 均值被语义无关邻居污染。

**各代方案在此问题上的位置**:
1. **Ho 1/d 权重**: 缓解但不解决 (手部关节间距集中, 距离差异不足)
2. **距离衰减 Laplacian**: 缓解 + 仍是聚合表征, 不解决 pairwise 信息损失
3. **Zhang 边向量 + 距离过滤**: 直接 pairwise, 无聚合损失, 归一化处理比例差

**当前实验状态**:
- Edge-ratio (Zhang 风格) 实现: `experiments/exp_edge_ratio.py`
- 定性观察: 局部手型保真度好 (近邻绿色边跟随)
- 待解决: 缺少关节空间正则导致 hyperextension 增加
- 推荐后续: 加 `λ||q - q_mid||²` 关节正则后重测

---

## 理论空缺：非均匀骨骼比例下的根本失效

> 来源: 项目讨论推导, 2026-04

### 问题陈述

所有基于笛卡尔空间点位的 retargeting 方法（Laplacian、edge-ratio）都隐含一个前提：
**源和目标在笛卡尔空间中具有可对齐的局部几何结构。**

当源（人手）与目标（机器人手）骨骼比例**非均匀**时，这个前提被破坏。

### Laplacian 的失效

邻域均值 = Σ w_j · p_j，混合了不同缩放比例的关节：

```
A 的邻域包含同指关节(比例 r_A)、跨指关节(比例 r_C)、掌心(比例 r_palm)
r_A ≠ r_C ≠ r_palm → 无法找到单一约束代表邻近平均 → δ 失真
```

即使用 global_scale 做全局补偿，也只能处理所有骨段比例一致的情况（r_i = const）。
真实场景中各手指、各骨段比例各异，全局缩放不能消除失真。

### Edge-ratio 的失效（更根本）

非均匀缩放有两个层次的破坏：

**1. 拓扑层：Delaunay 边集合改变**

不同的骨段长度导致点云几何构型不同 → 同一套关节做 Delaunay 会产生不同的边集合。
强制用 source 拓扑套在 robot 上，是在一个不自然的图结构上做优化。

**2. 向量层：对应边天然不对齐**

即使拓扑相同，边向量方向也不一致。
每条边的方向由整条运动链上各骨段长度累积决定，非均匀缩放使方向发生漂移：

```
human:  wrist → index_tip = [3cm,  0, 10cm]
robot:  wrist → index_tip = [2.5cm, 0, 7cm]   方向不同，不只是 norm 不同
```

归一化（除以各自 norm）只能消除长度差，无法修正方向偏差。

### 结论

**笛卡尔空间表征在跨比例 retargeting 中存在不可绕过的结构性失效**，
不是表征选择（Laplacian vs edge-ratio）的问题，而是假设前提的问题。
均匀缩放是特例，非均匀缩放是常态。

### 能绕过这个问题的方向

| 方向 | 核心思路 | 代价 |
|------|---------|------|
| 预处理几何对齐 | 先将 source 按 per-segment 骨骼比例缩放到接近 robot 结构，再做笛卡尔操作 | 需要事先标定每段骨骼比例 |
| 关节角空间操作 | 在角度域做 retargeting，完全回避骨骼长度差异 | 需要可靠的角度估计 |
| 纯方向约束 | 只约束骨段方向（单位向量），不约束长度 | 丢失长度信息，可能欠约束 |

当前 bone_scaling 实验（per-segment 比例补偿）是第一条路的局部实现，
但只在 source 预处理阶段做了比例补偿，没有解决 Delaunay 拓扑本身的差异。
