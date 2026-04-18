# GeoRT: Geometric Retargeting -- 论文笔记

> **论文**: Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm
> **作者**: Zhao-Heng Yin, Changhao Wang, Luis Pineda, Krishna Bodduluri, Tingfan Wu, Pieter Abbeel, Mustafa Mukadam
> **机构**: Meta FAIR + UC Berkeley BAIR
> **发表**: IROS 2025 | arXiv 2503.07541 | 17 citations
> **代码**: github.com/facebookresearch/GeoRT (CC-BY-NC 4.0)

---

## 1. Core Problem

GeoRT 要解决的是 dexterous hand retargeting 的 scalability 问题: 如何用最少的先验知识 (仅需 URDF + 5 分钟人手数据), 将人手指尖运动映射到任意机器人灵巧手的关节角, 同时满足实时 (1KHz) 部署要求。

### 核心挑战

| 挑战 | 具体表现 |
|------|---------|
| 骨段比例差异 | 人手和 robot hand 的指节长度/比例不同, 传统 scaling factor 需手动调整 |
| C-space 形状非线性 | 人手和 robot 指尖可达空间的形状明显不同 (Figure 2), 线性映射无法捕获 |
| 手动先验过多 | DexPilot 类方法需手工设计向量对, AnyTeleop 需手动配置 keypoint + 全局 alpha |
| 碰撞约束不可微 | 仿真器碰撞检测输出 bool, 无法参与梯度优化 |
| 实时性 | 遥操作要求 < 1ms 延迟 |

### 现有方法不足

| 方法 | 核心问题 |
|------|---------|
| DexPilot (ICRA 2020) | 完全 hand-crafted, Allegro-specific, 延迟 ~1s, 换手需重新设计全部向量对 |
| AnyTeleop (RSS 2023) | 一个全局 scaling factor alpha 不处理各指比例差异, 每种手仍需手动配置 |
| Eq.1 线性匹配 (传统) | L = sum of alpha_i * v_H - v_R 的线性目标, C-space 形状非线性时失效 (Figure 2) |
| 优化类方法 (SPIDER 等) | 离线 2.5 FPS, 无法实时遥操作 |

论文 Figure 2 给出了关键实验证据: 比较人手 ring finger 和 Allegro ring finger 的指尖可达空间 (C-space), 两者形状明显不同 -- 人手更弯曲窄长, robot 更宽扁规则。这直接否定了线性缩放的合理性。

---

## 2. Method Overview

### 整体 Pipeline

```
[预训练 (仿真, 一次性)]
  随机采样关节角 q -> 仿真器 FK -> 训练 neural FK_i (可微, per-finger)
  随机采样关节角 q -> 仿真器碰撞检测 -> 训练 collision classifier C(q)

[数据采集 (~5 min)]
  人手: 戴手套随机活动 -> 指尖点云 KC_H (motion capture)
  机器人: 仿真随机采样 -> FK -> 指尖点云 KC_R

[训练 retargeting MLP f_i (per-finger, 1-2 min, single GPU)]
  输入: 人手指尖位置 (手腕坐标系) [B, 3]
  输出: 机器人关节角 (归一化到 [-1, 1]) [B, n_joint]
  冻结 FK_i 和 C, 梯度穿过它们反传到 f_i

[推理]
  只部署 f_i (丢弃 FK 和 C)
  < 1ms per frame, 1KHz live
```

### 数据流 (trainer.py:272-274)

```python
point = batch.cuda()              # 人手指尖 x_H        [B, N_fingers, 3]
joint = ik_model(point)           # f(x_H) -> q         [B, DOF]
embedded_point = fk_model(joint)  # FK(q) -> x_R         [B, N_fingers, 3]
```

### 5 个几何 Loss

| Loss | 功能 | 机制 | 权重 (默认) |
|------|------|------|------------|
| L_dir (Eq.2) | 运动方向保持 | cosine similarity, 只管方向不管幅度 | 1.0 |
| L_cover (Eq.4) | C-space 覆盖 | Chamfer distance, 防止只用 robot ROM 的一小部分 | 80.0 |
| L_flat (Eq.5) | 映射平坦度 | 二阶有限差分 ≈ 0, 操控增益处处一致 | 0.1 |
| L_pinch (Eq.7) | 捏取对应 | 两指靠近时 robot 两指也必须靠近 (15mm threshold) | 1.0 |
| L_col (Eq.8) | 自碰撞避免 | 冻结的 collision classifier 输出 | 0.0 (开源版关闭) |

这是**自监督**训练 -- 没有 ground truth label, 没有人工标注的"这个人手姿态对应的 robot 关节角是 X"。所有 loss 都是数据自身应满足的几何约束。

### Per-Finger 独立建模

每根手指有独立的:
- Neural FK_i: 该手指关节角 -> 指尖 3D 位置 (2-layer MLP, 128 hidden)
- Retargeting MLP f_i: 人手该指尖位置 -> robot 该指关节角 (2-layer MLP, 128 hidden, Tanh output)

config 中通过 `fingertip_link` 字段定义手指-关节映射关系, 通过 `human_hand_id` 字段定义人手-机器人手的指尖对应。

---

## 3. Key Designs

### 3.1 Chamfer-based C-space Coverage (L_cover) -- 隐式骨段比例处理

**核心洞察**: 不做任何显式 scaling, 用 Chamfer distance 让映射后的 robot 指尖分布自动覆盖 robot 完整可达空间。

Chamfer distance (CD, Chamfer距离) 是双向最近邻距离之和, 源于 3D 重建领域 (PointNet, AtlasNet), GeoRT 把它应用到 retargeting:

- 点云 A (`embedded_point`): 人手数据经 f->FK 映射后的 robot 指尖位置
- 点云 B (`robot_points`): robot 随机采样 FK 得到的完整可达空间

**两团点云都在 robot 指尖空间**, 人手骨段长度已被 retargeting MLP f 吸收。

Chamfer 双向的功能分工:

| 方向 | 惩罚什么 | 是否天然满足 |
|------|---------|-------------|
| A->B: mapped 点找可达空间最近点 | mapped 点不在可达空间内 | 天然满足 (Tanh 输出在关节限位内) |
| B->A: 可达空间点找 mapped 点最近点 | 可达空间有大片区域没被映射覆盖 | **不天然满足 -- 这才是 L_cover 真正起作用的方向** |

**隐式比例处理**: 假设人手食指活动范围 10cm, robot 食指活动范围 6cm。L_cover 的 B->A 方向强制映射覆盖整个 6cm 区域, MLP 被迫学会非线性压缩。不需要手动设定任何 scaling factor, 这是目前 retargeting 方法中最优雅的比例处理方案。

代码实现 (trainer.py:299-305):

```python
selected_idx = np.random.randint(0, robot_points.shape[1], 2048)
target = torch.from_numpy(robot_points[:, selected_idx, :]).permute(1, 0, 2).float().cuda()
chamfer_loss = 0
for i in range(n_keypoints):
    chamfer_loss += chamfer_distance(embedded_point[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0))
```

### 3.2 Learned Collision Classifier -- 不可微查询的可微代理

**问题**: 传统碰撞检测 (MuJoCo `mj_collision`) 输出 bool -- 不可微, 无法作为 loss。

**解法**: 训练二分类器 C(q) -> [0, 1], 预测关节构型 q 的自碰撞概率:

```
数据生成:
  仿真中随机采样大量 q -> 查询碰撞检测器 -> 二值标签 -> BCE loss 训分类器

在 retargeting 训练中:
  L_col = -E[log(1 - C(f(x_H)))]     # Eq. 8
  C 冻结参数 (不更新), 但计算图保持活跃 -- 梯度穿过 C 反传到 f
```

反向传播路径: `dL_col/df = (dL_col/dC) * (dC/dq) * (dq/df)`, 其中 C 的 Jacobian dC/dq 提供了"哪个方向会增加碰撞概率"的梯度信号。

**通用模式 -- "不可微仿真器查询 -> 可微神经代理"**:

| 不可微查询 | 神经代理 | 论文 |
|-----------|---------|------|
| 自碰撞检测 | Collision classifier C(q) | GeoRT (本文) |
| 穿透深度 | Neural penetration estimator | DexGraspNet |
| 接触模型 | Learned contact model | ContactOpt |
| Signed clearance distance | Clearance MLP | NCC (Koptev 2022) |

这是本文最值得迁移的技术, 通用性远超 retargeting 本身。但在开源版中, collision loss 被设为 placeholder (`collision_loss = torch.tensor([0.0]).cuda()`), 未完整发布。

### 3.3 5 个 Loss 的协同: 从数学约束到任务语义

| Loss | 约束层面 | 工作空间 | 保证的操作体验 |
|------|---------|---------|-------------|
| L_dir | 微分结构 (一阶) | task space (3D) | 操作者肌肉记忆可迁移 |
| L_cover | 全局分布 | task space (3D) | 充分利用 robot 运动范围 |
| L_flat | 二阶光滑性 | task space (3D) | 操控增益可预测 |
| L_pinch | **任务语义** | task space (3D) | 捏合动作能正确执行 |
| L_col | 安全约束 | **joint space** | 不自碰撞 |

前三个 loss (dir + cover + flat) 约束的是映射的数学性质 -- 方向保持、覆盖均匀、增益一致。但它们不保证任务语义: 一个满足前三条的映射完全可以把人手捏合映射成 robot 手指张开 (只要这个映射局部方向一致、全局覆盖、增益平滑)。

L_pinch 是 5 个 loss 中唯一注入任务级语义的。这也意味着: 如果需要其他操作语义 (手指包裹、侧捏、三指抓), 理论上需要加更多类似 L_pinch 的 task-specific loss -- 这是 GeoRT 的设计边界。

---

## 4. Experiments

### 仿真评估 (Allegro Hand)

论文定义了两个量化指标:

- **Motion Preservation (运动保持, 越高越好)**: 均匀采样人手锚点和方向, 计算 Dir(x_H, d) 的平均值, 即 robot 运动方向与人手运动方向的对齐程度。取值 [-1, 1]
- **C-space Coverage (C-space 覆盖, 越高越好)**: 采样大量人手 x_H, 映射后的 robot 指尖覆盖了 robot 完整 C-space 的百分比 (用球体覆盖率近似)。取值 [0%, 100%]

| 方法 | Motion Preservation | C-space Coverage |
|------|-------------------|-----------------|
| Eq.1 Offline (传统线性) | 0.73 | 38% |
| Eq.1 Online (DexPilot/AnyTeleop 类) | -- | -- |
| **GeoRT (Ours)** | **0.94** | **90%** |

GeoRT 在两个指标上均大幅领先。C-space coverage 从 38% 提升到 90%, 意味着 robot 手指的运动范围被充分利用。

### 真实世界遥操作 (Allegro + Franka Panda + Manus 手套)

| 方法 | Onetime-Success (单次成功率) | Completion Time (完成时间) |
|------|--------------------------|--------------------------|
| Eq.1 Offline | 55% | 9.0s |
| Eq.1 Online | 42.5% | 19.3s |
| **GeoRT (Ours)** | **87.5%** | **3.2s** |

抓取任务: 操作者用 Manus 手套遥操作 Allegro Hand 抓取桌面物体。GeoRT 的成功率从 55% 提升到 87.5%, 完成时间从 9s 缩短到 3.2s。Figure 8 展示了在约 100 秒内清理 12 个桌面物体的 demo。

### 定性结果

Figure 9 展示了 Allegro Hand 和 LEAP Hand 两种 robot 的 retargeting 效果。即使 GeoRT 不使用任何 task-vector matching heuristic, 映射也能自动发现合理的手指对应关系 -- 这说明 5 个几何 loss 的约束空间足够窄, 能 shape 出语义合理的映射。

### 训练效率

| 阶段 | 时间 | 备注 |
|------|------|------|
| Neural FK 训练 | ~5 min | 一次性, per robot hand |
| Robot kinematics dataset 生成 | -- | 100K 采样, 一次性 |
| Human data 采集 | ~5 min | 戴手套随机活动 |
| Retargeting MLP 训练 | 1-2 min | 30-50 epochs, single NVIDIA 3060 |
| 推理 | < 1ms | 1KHz |

---

## 5. Related Work Analysis

论文引用精简 (19 篇), 定位在 teleoperation retargeting 的实用工具而非理论贡献。

### 论文的定位与分类

按 Meattini TRO 2022 综述的 6 大类分类法, GeoRT 属于 **Direct Cartesian (指尖位置匹配)** 范式, 但用 5 个 geometric loss 取代了传统的手工距离度量, 是该范式的 learning-based 升级版。

### 与论文中引用的方法的关系

| 引用方法 | GeoRT 对其的立场 |
|---------|----------------|
| DexPilot [2] | 主要 baseline, 批评其 hand-crafted 设计和 Allegro-specific |
| AnyTeleop [3] | baseline, 批评全局 alpha 不处理各指比例差异 |
| RTelekinesis [8] | baseline (Eq.1 类), 线性匹配不捕获 C-space 非线性 |
| Meattini Review [14] | 引用综述定位问题空间 |
| DPFM [16], FSF [17] | shape correspondence 方向, 论文提到 retarget 与 shape matching 的联系但未深入 |
| Diffusionnet [18] | shape analysis 工具, 提到可能的 future direction |
| Harmonic [19] | 流形映射方向, 论文未与其直接对比 |

论文自称是 DexterityGen [1] 系统的一部分, 作为 foundation controller 的前端。

### 未引用但相关的重要工作

- SPIDER (Meta FAIR 自家, 2025): 物理采样优化, 2.4M 帧, 离线
- QuasiSim (THU, ECCV 2024): 参数化准物理仿真 curriculum 优化
- CMU Kinematic Retargeting (TOG 2025): atlas/logmap 接触转移

---

## 6. Limitations & Future Directions

### 论文承认的局限

- **只用指尖**: 忽略手指姿态 (orientation) 和中间关节, 无法区分"同一指尖位置但关节构型不同"的情况 (如食指伸直 vs 弯曲到同一高度)
- **纯运动学**: 不考虑力/接触物理, 适合遥操但不适合需要物理交互的场景 (如需要特定接触力分布的任务)

### 代码/结构暴露的局限

- **Per-finger 独立**: 每根手指的 retargeting MLP 是独立的, 无法建模手指联动 (如 ring-pinky coupling, 拇指对掌与食指弯曲的耦合)
- **碰撞模块未完整开源**: collision classifier 的训练代码和预训练权重均未发布
- **Pinch loss 需人工数据**: 5 分钟手部运动采集不多但不是零; pinch 是唯一的 task-specific loss, 如果需要其他操作语义需手动增加
- **人手 mocap 分布偏移**: README 明确警告 vision-based mocap (MediaPipe) 在部署时存在 "significant input distribution shift", 推荐用手套系统
- **手指数固定**: config 中 `fingertip_link` 硬编码了手指对应, 不同指数 (3指/5指) 的 robot 需要新的 config 和重新考虑 pinch loss 的遍历逻辑
- **Chamfer 计算效率**: loss.py 中的 Chamfer distance 实现是 naive O(NM) 暴力计算, 未用 KD-tree 加速

### 可能的改进方向

| 方向 | 具体做法 |
|------|---------|
| 加入手指姿态信息 | 除指尖位置外, 加入中间关节位置或指节方向作为输入 |
| 跨手指联动 | 在 per-finger 基础上加 shared latent 或 cross-finger attention |
| 自动化 pinch loss | 从人手数据中自动发现 task-relevant 的接触模式, 而非硬编码 15mm threshold |
| 接触感知 | 引入力/触觉信息, 从纯运动学扩展到力觉映射 |
| Functional maps 自动对应 | 用 DPFM/FSF 自动发现人手-robot 手的对应, 替代手动 config |

---

## 7. Paper vs Code Discrepancies

| 内容 | 论文描述 | 代码实际 |
|------|---------|---------|
| Collision loss | 完整描述 Eq.8, lambda_4 in [1e-4, 1e-2] | placeholder: `collision_loss = torch.tensor([0.0]).cuda()`, 权重默认 0.0, **未完整发布** |
| Collision classifier 架构 | 未详述 | 开源代码中完全没有 classifier 的定义和训练代码 |
| Neural FK 架构 | 未详述 | 2-layer MLP, 128 hidden units, LeakyReLU + BatchNorm1d |
| Retargeting MLP 架构 | "multi-layer perceptron" | 2-layer MLP, 128 hidden, LeakyReLU + BatchNorm1d + Tanh output |
| 训练 epochs | 30-50 epochs (Section III.G) | 默认 200 epochs (trainer.py), README 说 30-50 后可 Ctrl+C |
| L_flat 扰动尺度 | 未明确 | 代码中固定 scale=0.002 (trainer.py:291) |
| L_dir 扰动尺度 | 未明确 | 0.001 + rand * 0.01 (trainer.py:309) |
| Pinch loss 归一化 | Eq.7 无归一化 | 代码中用 `mask.sum() + 1e-7` 归一化并乘以 batch_size (trainer.py:286) |
| L_cover 权重 | lambda_1 in [10, 100] | 默认 80.0, 与论文范围一致 |
| 支持的 robot hand | Allegro + LEAP (论文) | 只提供 Allegro left/right 的 config, LEAP 未包含 |
| DexterityGen 集成 | "developed as part of DexGen" | 开源代码独立, 无 DexGen 相关代码 |
| 人手数据格式 | [N, 3] numpy array (21 keypoints, MediaPipe 格式) | 与论文一致, 通过 `human_hand_id` 字段选取对应指尖 |

### 代码中的工程细节 (论文未提及)

- **数据重采样**: `dataset.py` 中用 Open3D voxel_down_sample (0.001m 分辨率) 减少空间不均衡, 然后上采样到 50000 点
- **关节角归一化**: `formatter.py` 将关节角归一化到 [-1, 1], IKModel 的 Tanh 输出直接在这个空间
- **SAPIEN 仿真器**: 底层用 SAPIEN (Pinocchio 内核) 做 FK 和碰撞检测, 不是 MuJoCo
- **PD 控制可视化**: `hand.py` 中可视化时用 PD 控制 (kp=400, kd=10), 纯训练不需要
- **Checkpoint 管理**: 每个 epoch 保存一次 + 维护 last.pth, 同时保存到时间戳目录和 `_last` 目录

---

## 8. Cross-Paper Comparison

### 与其他 retargeting 方法对比

| 维度 | GeoRT | GMR | AnyTeleop | ContactRetarget | QuasiSim |
|------|-------|-----|-----------|----------------|----------|
| **目标** | 灵巧手指尖映射 | 人形全身运动 | 灵巧手 keypoint 匹配 | 2 指夹爪 extrinsic 操作 | 灵巧手全手 |
| **表示空间** | 指尖位置 (task space) | 关节旋转 + 末端位置 | keypoint 向量 | contact configuration | 质点集 (point set) |
| **学习/优化** | Learning (MLP) | Optimization (diff-IK) | Optimization (SciPy) | Optimization (IK) | Optimization (curriculum) |
| **比例处理** | Chamfer 自适应 (无 scaling factor) | 非均匀局部缩放 (per-body s_b) | 全局 alpha | 不涉及 (2 指夹爪) | point set 松弛吸收 |
| **碰撞处理** | Learned proxy C(q) | 关节极限 clamp | 无 | 无 | 准物理仿真 |
| **接触建模** | L_pinch (15mm 阈值) | 无 | 无 | contact config 序列 | 渐进式接触收紧 |
| **手动先验** | 指对应 (哪指对哪指) | per-body 缩放因子 + key body 映射 | keypoint 对应 + alpha | 4 个原语手动设计 | MANO params |
| **推理速度** | 1KHz (MLP forward) | 60-70 FPS (diff-IK) | real-time (优化) | 离线 | 极慢 (三阶段迭代) |
| **换手成本** | 重训 neural FK (~5 min) | 新 config JSON | 重配置 + 调参 | N/A | N/A |
| **适用场景** | 遥操作 (real-time) | 遥操作 + RL 数据生成 | 遥操作 | 离线数据生成 | 离线数据生成 |

### 比例处理策略对比 (retargeting 的核心瓶颈)

| 方法 | 策略 | 优势 | 劣势 |
|------|------|------|------|
| GeoRT | Chamfer 自适应 (无显式 scaling) | 无需任何先验参数, MLP 自动学非线性压缩 | 只在指尖空间, 不保证中间关节合理 |
| GMR | 非均匀局部缩放 (per-body s_b) | 细粒度控制, 各身体部分独立调节 | 需手动为每个 robot 设定 s_b |
| AnyTeleop | 全局标量 alpha | 最简单 | 不处理各指比例差异 |
| DexPilot | beta=1.6 heuristic + 分段函数 | 对 Allegro 有效 | 完全 hand-crafted, 不可迁移 |
| CMU Kinematic | atlas lambda_S (骨段) + lambda_A (周长) | 基于内蕴几何, 理论优雅 | 需 artist 标注 axial curves |
| QuasiSim | point set alpha 松弛 | 天然 morphology-agnostic | 计算极昂贵, 三阶段迭代 |

### 方法范式对比

| 范式 | 代表方法 | 核心思想 | 优势 | 劣势 |
|------|---------|---------|------|------|
| 几何 loss 自监督 | **GeoRT** | 5 个 loss 约束映射的数学性质 + 任务语义 | 最快, 最 scalable, 无需配对数据 | 只用指尖, 无物理约束 |
| 内蕴几何接触转移 | CMU Kinematic | atlas/logmap 在手表面迁移接触分布 | 天然 bone-length-invariant, 最深操作语义 | 需 axial curves 标注, 离线 |
| 流形映射 | HAE (Harmonic) | CAE 学两个 pose 流形间的最小畸变映射 | 理论最优雅, 只需 ~8 标注 | 浅层网络, 纯 joint space, 无 task 语义 |
| 物理采样优化 | SPIDER | MPPI 采样 + 虚拟接触引导 | 物理可行, 大规模数据生成 | 离线 2.5 FPS |
| 非均匀缩放 + diff-IK | GMR | per-body scaling + mink IK | 全身实时, 17+ robot | 不建模接触, 全身而非手部 |

### 核心 Takeaway

| # | Takeaway | 对比基准 |
|---|---------|---------|
| 1 | L_cover 的 Chamfer 自适应是目前处理骨段比例差异最优雅的方案 -- 不需要任何手动 scaling factor | vs AnyTeleop (全局 alpha), DexPilot (beta heuristic), GMR (手动 per-body s_b) |
| 2 | "冻结辅助模型, 梯度穿过" 是通用的不可微查询->可微代理模式 | vs CHOMP (解析 SDF), Brax (可微仿真器) |
| 3 | 5 个几何 loss 中, L_pinch 是唯一注入任务语义的 -- 纯数学约束不保证操作有用性 | vs CMU Kinematic (接触区域 = 任务本质) |
| 4 | Per-finger 独立分解降低了维度但丢失了手指联动信息 | vs QuasiSim (point set 全手优化) |
| 5 | Setup cost 最低: URDF + 5 min data + 2 min train = deploy, 是 scalability 的当前标杆 | vs CMU (~hours axial curves), DexPilot (~days hand-crafted) |
