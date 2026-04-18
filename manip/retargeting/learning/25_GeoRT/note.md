# GeoRT: Geometric Retargeting -- 综合分析

Paper: Zhao-Heng Yin et al., Meta FAIR + UC Berkeley BAIR, IROS 2025, 17 citations
PDF: `GeoRT_Geometric_Retargeting_2503.07541.pdf`
Code: `GeoRT/`

---

## 1. 核心问题

GeoRT 要解决的是 **dexterous hand retargeting 的 scalability** 问题: 如何用最少的先验知识 (只需 URDF + 5 分钟人手数据), 将人手指尖运动映射到任意机器人灵巧手的关节角, 且满足实时 (1KHz) 部署要求。

核心挑战:

| 挑战 | 具体表现 |
|------|---------|
| 骨段比例差异 | 人手和 robot hand 的指节长度/比例不同, 传统 scaling factor 需手动调 |
| 手动先验 | DexPilot 类方法需要手工设计向量对, 换手就要重新设计 |
| 碰撞约束不可微 | 仿真器碰撞检测输出 bool, 无法参与梯度优化 |
| 实时性 | 遥操作要求 < 1ms 延迟 |

现有方法不足:
- **DexPilot (ICRA 2020)**: 完全 hand-crafted, Allegro-specific, 延迟 ~1s, 换手要重设计全部向量对
- **AnyTeleop (RSS 2023)**: 一个全局 scaling factor alpha, 不处理各指比例差异, 每种手仍需手动配置
- **优化类方法 (SPIDER 等)**: 离线 2.5 FPS, 无法实时

---

## 2. 方法概览

### 2.1 整体 Pipeline

```
[预训练 (仿真, 一次性)]
  随机采样关节角 q -> 仿真器 FK -> 训练 neural FK_i (可微, per-finger)
  随机采样关节角 q -> 仿真器碰撞检测 -> 训练 collision classifier C(q)

[数据采集 (~5 min)]
  人手: 戴手套随机活动 -> 指尖点云 KC_H (motion capture)
  机器人: 仿真随机采样 -> FK -> 指尖点云 KC_R

[训练 retargeting MLP f_i (per-finger, 3-5 min, single GPU)]
  输入: 人手指尖位置 (手腕坐标系)
  输出: 机器人关节角
  冻结 FK_i 和 C, 梯度穿过它们反传到 f_i

[推理]
  只部署 f_i (丢弃 FK 和 C)
  < 1ms per frame, 1KHz live
```

### 2.2 5 个几何 Loss

| Loss | 功能 | 机制 | 权重 |
|------|------|------|------|
| L_dir | 运动方向保持 | cosine similarity (只管方向不管幅度) | 1.0 |
| L_cover | C-space 覆盖 | Chamfer distance (防止只用 robot ROM 的一小部分) | 80.0 |
| L_flat | 映射平坦度 | 二阶有限差分 ≈ 0 (操控增益处处一致) | 0.1 |
| L_pinch | 捏取对应 | 两指靠近时 robot 两指也必须靠近 | 1.0 |
| L_col | 自碰撞避免 | 冻结的 collision classifier 输出 (详见下文) | 0.0 (开源版关闭) |

**注意**: 这是**自监督**训练 -- 没有 ground truth label, 没有人工标注的 "这个人手姿态对应的 robot 关节角是 [...]"。所有 loss 都是数据自身应满足的几何约束, 网络在约束下自己找到合理的映射。

### 2.3 Loss 各自工作在什么空间

训练循环的数据流 (`trainer.py:272-274`):
```python
point = batch.cuda()              # 人手指尖 x_H        [B, N_fingers, 3]
joint = ik_model(point)           # f(x_H) -> q         [B, DOF]
embedded_point = fk_model(joint)  # FK(q) -> x_R         [B, N_fingers, 3]
```

| Loss | 输入 | 工作空间 | 为什么必须在这个空间 |
|------|------|---------|-------------------|
| L_dir | x_H 和 x_R 的微扰方向 | task space (3D) | "方向"是笛卡尔空间的几何概念, q 空间的方向没有直觉对应 |
| L_cover | x_R 分布 vs robot 可达空间分布 | task space (3D) | 对齐的是指尖可达空间的形状 |
| L_flat | x_R 的二阶差分 | task space (3D) | "平坦度"是映射在笛卡尔空间的性质 |
| L_pinch | x_H 和 x_R 的指间距离 | task space (3D) | "两指靠近"是笛卡尔距离, q 空间的距离没有物理含义 |
| L_col | q (关节角) | **joint space** | 自碰撞完全由关节构型决定, 和指尖位置无关 |

前 4 个 loss 需要 FK 把 q 映射到笛卡尔空间才能计算; L_col 直接吃 q, 不经过 FK。**FK 在 pipeline 中的角色是 "q 空间到笛卡尔空间的桥梁"**, 服务于那些语义定义在笛卡尔空间的 loss。

### 2.4 Per-Finger 独立建模

每根手指有独立的:
- Neural FK_i: 该手指关节角 -> 指尖 3D 位置
- Retargeting MLP f_i: 人手该指尖位置 -> robot 该指关节角

优点: 降低问题维度, 各指独立训练
缺点: 无法建模手指间的联动 (如拇指对掌与食指弯曲的耦合)

---

## 3. 关键设计

### 3.1 5 个 Loss 的详细设计与直觉

#### L_dir -- 运动方向保持 (可控性的基础)

```python
# 在人手输入上加随机小扰动
direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
scale = 0.001 + torch.rand(...) * 0.01
point_delta = point + direction * scale       # x_H + delta

# 扰动后也走一遍 f -> FK
embedded_point_delta = fk_model(ik_model(point_delta))  # x_R'

# 比较两边的运动方向 (cosine similarity)
d1 = point_delta - point                     # 人手侧位移 delta_H
d2 = embedded_point_delta - embedded_point   # robot 侧位移 delta_R
direction_loss = -(cosine_similarity(d1, d2)).mean()
```

直觉: 人手指尖往右移了一点, robot 指尖也应该往右移。只管方向, 不管幅度 -- 这是遥操作最基本的要求: 操作者的肌肉记忆能直接迁移。

#### L_cover -- Chamfer 覆盖 (充分利用 ROM)

```python
# robot 可达空间: 仿真中随机采样大量 q -> FK -> 指尖位置集合
# 代表了该手指指尖能到达的所有位置
target = robot_points[:, selected_idx, :]     # [2048, N_fingers, 3]

# per-finger Chamfer distance
for i in range(n_keypoints):
    chamfer_loss += chamfer_distance(
        embedded_point[:, i, :],   # 映射后的 robot 指尖位置分布
        target[:, i, :]            # robot 完整可达空间分布
    )
```

**Chamfer distance** 是双向最近邻距离之和, 在 3D 重建领域 (PointNet, AtlasNet 等) 是标准的点云配准 loss。GeoRT 把它挪到 retargeting 语境, 只是配准对象从"物体表面"变成了"指尖位置分布":

- 点云 A (`embedded_point`): 人手数据经 f->FK 映射后的 robot 指尖位置
- 点云 B (`robot_points`): robot 随机采样 FK 得到的完整可达空间

**两团点云都在 robot 指尖空间**, 不是人手 vs robot 的直接比较。人手的骨段长度已经被 retargeting MLP f 吸收 -- f 输出 robot 关节角, 再经过 robot 自己的 FK。

Chamfer 双向的意义:

| 方向 | 惩罚什么 | 是否天然满足 |
|------|---------|-------------|
| A->B: 每个 mapped 点找可达空间最近点 | mapped 点不在可达空间内 | **天然满足** -- MLP 最后是 Tanh, 输出在关节限位内, FK 结果一定在可达空间 |
| B->A: 每个可达空间点找 mapped 点最近点 | 可达空间中有大片区域没被映射覆盖 | **不天然满足** -- 这才是 L_cover 真正起作用的方向 |

没有 L_cover 时, MLP 完全可以把所有人手输入映射到 robot 关节角的一个小区间 (比如食指只用 ROM 的 1/9)。输出合法, FK 结果在可达空间内, 但 robot 手指实际只用了一小部分运动范围。

**这也是比例差异的隐式解法**: 假设人手食指活动范围 10cm, robot 食指活动范围 6cm。L_cover 的 B->A 方向强制映射覆盖整个 6cm 区域, MLP 被迫学会非线性压缩, 不需要手动设定任何 scaling factor。

#### L_flat -- 平坦度 / 映射增益一致

```python
# 在 x_H 的正反方向各加一个小扰动
delta1 = direction * scale
point_delta_1p = point + delta1               # x_H + delta
point_delta_1n = point - delta1               # x_H - delta

# 各自走 f -> FK
embedded_point_p = fk_model(ik_model(point_delta_1p))  # x_R+
embedded_point_n = fk_model(ik_model(point_delta_1n))  # x_R-

# 二阶有限差分 approx 0
curvature_loss = ((embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2).mean()
```

这就是离散 Laplacian: `f(x+d) + f(x-d) - 2f(x) approx 0`, 要求映射局部线性。保证操控增益处处一致 -- 人手移 1mm, robot 不管在什么姿态都移差不多的量, 不会某个区域灵敏某个区域迟钝。

#### L_pinch -- 捏取对应 (任务语义注入)

```python
for i in range(n_finger):
    for j in range(i + 1, n_finger):
        distance = point[:, i] - point[:, j]
        mask = (torch.norm(distance, dim=-1) < 0.015).float()  # 15mm threshold

        e_distance = ((embedded_point[:, i] - embedded_point[:, j]) ** 2).sum(-1)
        pinch_loss += (mask * e_distance).mean()
```

人手两指距离 < 15mm 时 (正在捏), robot 两指距离也要尽量小。不捏的时候不管 (mask = 0)。

#### L_col -- 自碰撞 (直接在 joint space)

```python
# 论文 Eq.8 (代码中是 placeholder, 未完整发布):
# L_col = -E[log(1 - C(joint))]
# C(q) 直接吃关节角 q, 不经过 FK
# 只能检测自碰撞 (输入只有 q, 没有外部物体信息)
collision_loss = torch.tensor([0.0]).cuda()  # 开源版未实现
```

#### 5 个 Loss 的分工: 为什么这些约束能 shape 出 retargeting

Retargeting 没有唯一正确解 (人手和 robot 手形态不同, 不存在客观正确的对应), 这些约束定义的是"可用映射"的充分条件:

| Loss | 约束层面 | 保证的操作体验 | 缺失时的退化 |
|------|---------|-------------|------------|
| L_dir | 微分结构 | 操作者肌肉记忆可迁移 (我往右推, robot 也往右) | 方向混乱, 无法操控 |
| L_cover | 全局分布 | 充分利用 robot 运动范围 | 映射缩到一小块, robot "不听话" |
| L_flat | 二阶光滑性 | 操控增益可预测 (任何姿态下反应一致) | 某些区域极灵敏, 某些极迟钝 |
| L_pinch | **任务语义** | 捏合动作能正确执行 | 人手捏合时 robot 可能张开 |
| L_col | 安全约束 | 不自碰撞 | 危险构型 |

**关键洞察**: 前三个 loss (dir + cover + flat) 约束的是映射的**数学性质** -- 方向保持、覆盖均匀、增益一致, 保证映射是一个"拓扑上合理的连续双射"。但它们**不保证任务语义**: 一个满足前三条的映射完全可以把人手捏合映射成 robot 手指张开 (只要这个映射局部方向一致、全局覆盖、增益平滑就行)。

L_pinch 是 5 个 loss 中**唯一注入任务级语义**的。它显式要求"人手两指靠近 -> robot 两指也靠近", 在前三个 loss 定义的合理映射空间中, 挑出"任务上有用"的那个解。

这也意味着: 如果还需要其他操作语义 (手指包裹、侧捏、三指抓), 理论上需要加更多类似 L_pinch 的 task-specific loss -- 这是 GeoRT 的设计边界, 它只显式建模了 pinch 这一种操作原语。

---

### 3.2 Neural FK: 工程便利, 非理论必要

**论文原话**: "One can also use an analytical forward kinematics function."

Dexterous hand 的运动学模型完全足够精确, neural FK 的目的**不是替代解析 FK 的精度**, 而是让整个训练管线在同一个 PyTorch 计算图中端到端可微:

```
human keypoints x_H -> f_i(x_H) -> joint angles q -> FK_i(q) -> robot keypoints x_R -> losses
                                                    ^
                                          梯度要从 loss 穿过 FK 传回 f_i
```

| 方案 | 可微性 | GPU batch | per-finger 拆分 | 额外依赖 |
|------|--------|-----------|----------------|----------|
| Neural FK (2-layer MLP) | 天然 autograd | 天然支持 | 天然支持 | 无 |
| 解析 FK + pytorch_kinematics | 需引入库 | 需适配 | 需手动拆 URDF 树 | pytorch_kinematics 等 |
| 解析 FK + 手写微分 | 需手动实现 | 需手动 batch | 需手动拆 | 无 |

Neural FK_i 架构 (from `geort/model.py`):
```
Linear(n_joint, 128) -> LeakyReLU -> BatchNorm1d(128)
Linear(128, 128)     -> LeakyReLU -> BatchNorm1d(128)
Linear(128, 3)       # wrist frame 下的 3D fingertip position
```

**关键**: 推理时 neural FK 被丢弃。Retargeting MLP f_i 直接从人手 keypoint 映射到关节角, 不经过 FK。因此 neural FK 的近似误差**不影响最终部署精度**, 它只是训练时的辅助工具。

### 3.3 Learned Collision Classifier: 将不可微查询蒸馏为可微代理

> **这是本文最值得迁移的技术, 通用性远超 retargeting 本身。**

#### 问题

传统碰撞检测 (MuJoCo `mj_collision`) 输出 bool 值 -- 不可微, 无法作为 loss 参与梯度优化。

对灵巧手而言, 自碰撞是核心约束: 16+ 个 link 的铰接体, 手指间碰撞模式极其复杂, 解析 SDF 难以处理。

#### 解法: Surrogate Collision Model

训练一个二分类器 C(q) -> [0, 1], 预测关节构型 q 的自碰撞概率:

```
数据生成:
  1. 仿真中随机采样大量关节构型 q
  2. 对每个 q 查询仿真器碰撞检测器 -> 二值标签 (碰撞 / 无碰撞)
  3. 用 (q, label) 对训练 MLP 分类器, 标准 BCE loss

在 retargeting 训练中使用:
  L_col = -E[log(1 - C(f(x_H)))]     # Eq. 8

  C 冻结参数 (no parameter updates)
  但 C 的计算图保持活跃 -- 梯度穿过 C 反传到 f
  C(q) -> 1 (高碰撞概率): loss -> infinity, 强惩罚
  C(q) -> 0 (无碰撞):    loss -> 0, 无惩罚
  权重: lambda_4 in [1e-4, 1e-2] (soft constraint)
```

反向传播路径:
```
dL_col/df = (dL_col/dC) * (dC/dq) * (dq/df)
```
C 的参数不更新, 但 C 的 Jacobian dC/dq 提供了关于"哪个方向会增加碰撞概率"的梯度信号, 引导 f 的输出远离碰撞构型。

#### 为什么这个模式有效

1. **精度要求低**: L_col 只是 5 个 loss 之一, 权重小 (1e-4 ~ 1e-2), 作为 soft penalty 不需要精确的碰撞边界, 只要在碰撞区域附近给出正确的梯度方向即可
2. **训练数据免费**: 仿真器随机采样 + 碰撞查询, 无需人工标注
3. **一次训练, 反复使用**: 对同一 robot hand, C(q) 训练一次就可以在任意下游优化任务中复用
4. **推理时丢弃**: 和 neural FK 一样, C 只在训练时使用, 不增加部署开销

#### 通用模式: "不可微仿真器查询 -> 可微神经代理"

这个思路的应用远不限于碰撞检测:

| 不可微查询 | 神经代理 | 论文 | 年份 |
|-----------|---------|------|------|
| 自碰撞检测 | Collision classifier C(q) | GeoRT (本文) | 2025 |
| 穿透深度 | Neural penetration estimator | DexGraspNet (Wang et al.) | 2023 |
| 接触模型 | Learned contact model | ContactOpt (Grady et al.) | 2021 |
| Signed clearance distance | Clearance MLP | NCC (Koptev et al.) | 2022 |
| 稳定性判定 | Stability predictor | 抓取质量评估 (various) | -- |

#### 与替代方案的对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Learned proxy** (GeoRT) | 快, 简单, 适合复杂几何 | 需训练数据, 有近似误差 | 铰接体自碰撞 (灵巧手) |
| **Analytical SDF** (CHOMP) | 梯度精确, 无需训练 | 多 link 铰接体难以处理 | 凸障碍物, 简单几何 |
| **可微物理仿真器** (Brax, DiffTaichi) | 完整物理, 梯度精确 | 计算昂贵, 接触模型受限 | 带动力学的轨迹优化 |
| **Mesh-level 可微接触** (PyTorch3D, Kaolin) | 直接操作 mesh | 高分辨率时计算量大 | rendering-style 的接触 |

**结论**: learned proxy 在"几何复杂 (多 link 铰接体) + 需要快速查询"这个交叉点上是目前最实用的选择。该模式在灵巧手领域已被 DexGraspNet 等采用, 但尚未成为主流做法 -- 更多工作仍在用解析 SDF 或直接规避碰撞约束。

### 3.4 骨段比例处理: Chamfer 自适应

不做任何显式 scaling。L_cover 用 Chamfer distance 对齐映射后的 robot 指尖分布和 robot 完整可达空间:
- Chamfer 比较的两团点云都在 robot 指尖空间 (不是人手 vs robot 的直接比较)
- 人手骨段长度已被 retargeting MLP f 吸收 -- f 输出 robot 关节角, 再经过 robot FK
- MLP 在 Chamfer loss 驱动下自动学会非线性压缩/扩展, 将人手范围映射到 robot 全部可达空间
- 不需要手动设定 scaling factor (对比 AnyTeleop 的全局 alpha)

这是目前 retargeting 方法中最优雅的比例处理方案。

---

## 4. 代码与论文差异

| 内容 | 论文描述 | 代码实际 |
|------|---------|---------|
| Collision loss | 完整描述 Eq.8, lambda_4 in [1e-4, 1e-2] | 开源代码中是 placeholder (`collision_loss = torch.tensor([0.0]).cuda()`), **未完整发布** |
| Neural FK 架构 | 未详述 | 2-layer MLP, 128 hidden units, LeakyReLU + BatchNorm |
| Collision classifier 架构 | 未详述 | 未在开源代码中找到 |

---

## 5. 跨论文对比

### 与其他 retargeting 方法的定位

| 维度 | GeoRT | CMU Kinematic | SPIDER | DexPilot | AnyTeleop |
|------|-------|---------------|--------|----------|-----------|
| 表示空间 | 指尖位置 (task space) | 表面接触 (intrinsic) | 关节角 + 物理 | 指尖/指间向量 | keypoint 向量 |
| 碰撞处理 | learned proxy | 无 | 物理仿真 | 无 | 无 |
| 比例处理 | Chamfer 自适应 | atlas lambda_S/lambda_A | N/A | beta=1.6 heuristic | 全局 alpha |
| 手动先验 | 指对应 (哪指对哪指) | axial curves (artist 标) | URDF + IK | 全部向量对手工设计 | keypoint 对应 + alpha |
| 速度 | 1KHz | 4-12h per sequence | 2.5 FPS offline | ~1s | real-time |
| 换手成本 | 重训 neural FK (~min) | 重标 axial curves (~hr) | 加 URDF + IK | 重设计全部 (~days) | 重配置 + 调参 |

### 综述分类 (Meattini TRO 2022) 中的位置

GeoRT 属于 **Direct Cartesian (指尖位置匹配 + learned mapping)** 范式, 但用 5 个 geometric loss 取代了传统的手工距离度量, 是该范式的 learning-based 升级版。

---

## 6. 局限

- **只用指尖**: 忽略手指姿态和中间关节, 无法区分"同一指尖位置但关节构型不同"的情况
- **Per-finger 独立**: 无法建模手指联动 (如 ring-pinky coupling)
- **Pinch loss 需人工数据**: 5 分钟手部运动采集, 虽然不多但不是零
- **碰撞模块未完整开源**: collision classifier 的训练代码和预训练权重未发布
- **纯运动学**: 不考虑力/接触物理, 适合遥操但不适合需要物理交互的场景

---

## 7. 对我的价值

1. **Collision proxy 模式可直接迁移**: 任何需要可微自碰撞约束的 pipeline 都可以用这个模式 -- 采样 + 碰撞查询 + 训分类器 + 冻结作 loss。成本极低, 效果作为 soft penalty 足够
2. **"预训练辅助模块, 冻结, 梯度穿过" 的通用 pattern**: 不限于碰撞, 任何不可微的仿真器查询都可以这样处理
3. **L_cover 的 Chamfer 自适应**: 处理比例差异的最优雅方案, 不需任何 scaling 先验
4. **Per-finger 分解的取舍**: 降低维度但丢失联动信息 -- 如果要做协调抓取, 可能需要在 per-finger 基础上加一个 shared latent 或 cross-finger attention
5. **Scalability 标杆**: 5 min data + URDF -> 部署, 换手只需重训 neural FK, 是目前 retargeting 方法中 setup cost 最低的
