# 手部 Retargeting 核心论文蒸馏

> 目标: scalable 手部 retarget, 在 RL 之前获得好的重定向结果
> 筛选标准: 专注 retarget 算法本身, 不依赖 RL 做后续修正

---

## GeoRT (Meta FAIR, IROS 2025, 17 citations) ⭐⭐⭐

**一句话**: 5 个几何 loss 训 MLP, 人手指尖→机器人关节角, 不需配对数据, 1KHz

**做法**:
```
预训练 (纯仿真, 一次性):
  随机采 robot 关节角 → FK → 训 neural FK (可微)
  随机采 → 训 collision classifier C(q)

数据 (~5 min):
  人: 戴手套随机活动 → 指尖点云 KC_H
  机器人: 仿真随机采 → FK → 指尖点云 KC_R

训 MLP (每根手指独立):
  输入: 人手指尖位置 (手腕系)
  输出: robot 关节角

推理: <1ms, 1KHz live
```

**5 个 loss**:
```
L_dir    运动方向保持 (cosine similarity, 只管方向不管幅度)
L_cover  C-space 覆盖 (Chamfer, 防止只用 robot 运动范围的一小部分)
L_flat   平坦度 (二阶有限差分 ≈ 0, 操控增益处处一致)
L_pinch  捏取对应 (两指靠近时 robot 两指也必须靠近)
L_col    无碰撞 (frozen collision classifier 的输出)
```

**骨段比例**: 不做任何缩放。L_cover 用 Chamfer 自动学非线性 C-space 映射 — 短骨段 = 更小 C-space, 网络自适应压缩。这是目前最优雅的比例处理方案。

**keypoint 对应**: 半手动 — 只需指定 "人手第几指→robot 第几指", 只用指尖位置

**对我的价值**:
- 5 个 loss 不依赖特定 robot, 换手只需重训 neural FK
- L_cover 解决比例差异, 不需任何缩放先验
- L_pinch 直接保证捏取功能
- 5 min 数据 + URDF 就跑, scalable

**局限**: 只用指尖, 忽略手指姿态/中间关节; 手指独立 MLP 无法建模联动; pinch loss 需 5 min 人工数据

---

## CMU Kinematic Retargeting (CMU + Meta FAIR, ACM TOG 2025, 17 citations) ⭐⭐⭐

**一句话**: 用非等距形状匹配 (atlas + logmap/expmap) 把人手表面的接触分布转移到不同形态的手上 — "接触区域分布才是操作的本质"

**做法**:
```
1. Dense Contact Pairing
   source hand mesh + object mesh + 每帧接触点 (重心坐标存储)

2. Bulk Contact Transfer (核心)
   source/target 手表面建 atlas (多张 chart)
   每张 chart: logmap 投影到切平面极坐标 (r, θ)
   通过 axial curves 建立 chart 对应
   λ_A (周长缩放) + λ_S (骨段长度缩放) 处理比例差异
   一次性批量转移整个序列所有接触

3. 帧独立 IK → 4. 加速度平滑 → 5. Spline Fitting (C² 连续)
```

**核心洞察 — 内蕴坐标**:
```
接触点在切平面的极坐标 (r, θ):
  r = 距 landmark 的测地距离 (不是欧式距离)
  θ = 切平面上的方向角

性质:
  天然与手的全局姿态解耦 (内蕴量, 不依赖外在坐标)
  不同手上的同一 landmark → 同一 (r, θ) → 同一接触语义
  λ_S 拉伸/压缩测地距离 → 适配骨段长度差异
  λ_A 拉伸/压缩角度分布 → 适配手指粗细差异
```

**5 种目标手**: Human, Witch (长指甲), Alien (3 指), Allegro (4 指), Prosthetic

**对我的价值**:
- "接触即操作本质" — 最深层的 retarget 认知
- Atlas/logmap 天然 bone-length-invariant
- 支持不同指数 (3 指/4 指/5 指) 通过 axial curve 重映射
- 帧独立求解 + 后处理平滑 = 避免误差累积

**局限**: 需要 artist 标注 axial curves (per hand pair); 纯运动学无物理; 需要稠密接触标注; 单序列 4-12h (CPU)

---

## SPIDER (Meta FAIR + CMU, arXiv 2025, 15 citations) ⭐⭐

**一句话**: 采样优化 (非 RL) 在物理仿真中做 retarget, 虚拟接触引导解决接触歧义, 9 具身 2.4M 帧

**做法**:
```
1. 输入: MANO params + object mesh/pose
2. Kinematic IK → 初始 reference
3. Sampling optimization in MuJoCo:
   Annealed MPPI: 先全局搜索接触模式, 再局部细化
   Virtual contact guidance: 虚拟弹簧先"粘"物体, 再逐步撤去 (curriculum)
   Robustification: minimax over 物理参数
4. 输出: 物理可行的控制序列
5. 数据增强: 换 mesh / 改物理 / 加外力
```

**对我的价值**:
- 采样 > RL for 离线数据生成 (不需训 policy)
- 虚拟接触引导 = homotopy continuation (通用 trick)
- Embodiment-agnostic: 加新手只需 URDF + IK
- 2.4M 帧 scale 证明方法可扩展

**局限**: 离线 2.5 FPS, 不能实时; 无 learned component; 成功率 ~45%

---

## ARAP (Sorkine & Alexa, EG SGP 2007, 经典) ⭐

**一句话**: mesh 变形时保持每个顶点邻域"尽可能刚性" — 交替优化 per-vertex 旋转和全局位置

**做法**:
```
能量: E = Σ_i Σ_j w_ij || (p'_i - p'_j) - R_i (p_i - p_j) ||²
  p_i - p_j = 原始边向量
  R_i = 每个顶点的最优旋转 (SVD 求解)
  要求: 变形后的边 ≈ 原始边旋转后

交替迭代:
  1. 固定位置 → SVD 求每个顶点的最优 R_i (local step)
  2. 固定旋转 → 解稀疏线性系统 Lp' = b (global step)
  系统矩阵 L 只分解一次, 2-4 次迭代收敛
```

**对手部 retarget 的分析**:
```
✗ 不适合跨比例 retarget

原因: ARAP 设计目标是"保持边长 + 刚性"
  但跨比例 retarget 的本质是改变边长 (non-isometric)
  用"抵抗变形"的工具做变形 = 概念矛盾

如果强制关节位置 (handle constraints):
  ARAP 在约束附近产生高度局部化的拉伸
  不会平滑分布比例变化
```

**唯一价值**: 理解 Laplacian 和 ARAP 的关系 — ARAP 第一次迭代的初始猜测就是 Laplacian editing 的结果, ARAP 是 Laplacian 的非线性改进版。证明了 Laplacian 的比例敏感问题在 ARAP 中也存在且更严重。

---

## DexFlow (WHU/MIT/Georgia Tech, arXiv 2025, 5 citations) ⭐

**一句话**: MANO→robot hand 层级优化, 逐指 contact 精修, 292K 帧数据集

**做法**:
```
1. 粗 retarget: 13 个手动关键点匹配 + 二阶加速度平滑
2. 细精修: thumb→pinky 逐指优化 (每次 4 DOF)
   能量: 接触距离 + 穿透 + 法线对齐 + 自穿透 + 关节正则
数据集: 292K 帧, 50 YCB 物体, ShadowHand + Allegro
```

**对我的价值**: 逐指优化分解子问题; 292K 帧数据集可做 benchmark
**局限**: retarget 阶段仍手动关键点 + scaling; 预印本低引用; SSR 40%

---

## 三种变形方法对比 (结论)

```
Laplacian (Ho 2010/OmniRetarget):
  保持绝对偏移 → 比例敏感 → ✗ 跨比例

ARAP (Sorkine 2007):
  保持边长+刚性 → 主动抵抗长度变化 → ✗ 跨比例 (更严重)

Atlas/LogMap (CMU Kinematic):
  内蕴测地坐标 → 天然不依赖骨段长度 → ✓ bone-length-invariant
  λ_S/λ_A 参数化比例差异 → 优雅且可自动化

结论: Atlas/LogMap 是跨比例 retarget 最有前景的方向
  改进方向: 用 functional maps 自动化 landmark 对应, 替代手动 axial curves
```

---

## DexPilot (NVIDIA + CMU, ICRA 2020, ~350 citations) — Baseline

**一句话**: 手工设计指尖/指间向量对 + 非线性优化，实时映射人手→Allegro Hand，被 6/8 篇引用的事实标准 baseline

**做法**:
```
定义两类向量:
  S1: finger→thumb 向量 (捕捉 pinch)
  S2: finger→finger 向量 (捕捉间距)

Cost: C(q_r) = Σ s(d_i) · ||f(d_i)·r̂_i(q_h) - r_i(q_r)||²
  s(d_i): switching weight — 接近时权重暴增 (200-400x)
  f(d_i): distancing function — 远时线性缩放 β=1.6, 近时强制闭合

求解: SciPy SLSQP, 实时
```

**骨段比例**: 用 β=1.6 的线性缩放 + 分段函数 heuristic 绕过, 不是真正解决
**对我的价值**: 理解问题空间的起点; scalability 的反面教材 — 换手要重新设计全部向量对和调参
**局限**: 完全 hand-crafted, Allegro-specific, 延迟 ~1s

---

## AnyTeleop (UCSD + NVIDIA, RSS 2023, ~120 citations) — Baseline

**一句话**: 模块化遥操 pipeline, retarget 本身是 keypoint-vector 匹配 + 全局 scaling factor α

**做法**:
```
min_{q_r} Σ ||α·v_i - f_i(q_r)||² + β||q_r - q_{r,t-1}||²
s.t.  q_l ≤ q_r ≤ q_u

α: 全局 scaling factor (手动设定)
v_i: 人手 keypoint vector
f_i: robot FK
```

**骨段比例**: 一个全局标量 α, 不处理各指比例差异
**对我的价值**: 暴露了 scalable retarget 的核心瓶颈 — 换手需要手动指定 keypoint 对应 + 调 α
**局限**: 不算真正 "any" — 每种手仍需手动配置

---

## Harmonic Mapping / HAE (UCLA, IJRR 2021, ~50 citations) ⭐⭐

**一句话**: 把 retarget 建模为两个 pose 流形之间的 harmonic mapping (最小化映射畸变), 用 Contractive Autoencoder 数据驱动学习, 只需 ~8 个手动标注的对应 pose pair

**做法**:
```
HAE loss = CAE(重建) + Pin(对应点约束) + BA(域覆盖)

CAE: ||q - g∘f(q)||² + λ₁||J_f||²_F
  J_f 的 Frobenius norm ∝ harmonic mapping distortion
  → CAE 正则项天然近似 harmonic mapping!

Pin: L 个手工标注的 (human_pose, robot_pose) 对, 双向约束
BA: Chamfer distance 确保映射覆盖目标域

Adaptive Reference Point Selection:
  每轮: 训 HAE → 找目标域最远点 → 人工标注对应 → 加入训练集
  → 渐进式地用最少标注覆盖目标域 (~8 对够)
```

**骨段比例**: 理论上 harmonic mapping 最小化映射畸变 — 当源/目标 metric 不同时找"最均匀的变形"。但实现中用 identity metric, 假设关节角欧式距离有意义 (粗糙)
**对我的价值**: 理论框架最有价值 — retarget = "两个流形间的最小畸变映射"。只需 ~8 个标注点。如果能自动发现 reference pairs 就是 scalable 的
**局限**: 浅层网络 (2-layer ReLU), 只在关节角空间, 无 task-space 语义, 无实时验证

---

## Contact Transfer (CMU, ICRA 2022) ⭐⭐

**一句话**: CMU Kinematic Retargeting 的前驱 — 用 logmap 把人手上的接触 patch 迁移到任意构型机械手, 再 IK 求解

**做法**:
```
1. 在人手/目标手 skin mesh 上构建 logmap 坐标 (r, φ)
   (Vector Heat Method, 几何无关)
2. 用户指定 4 个锚点参数 (root vertex + tangent direction, 每侧 2 个)
3. 人手上的接触 patch → logmap 极坐标 → 目标手上 expmap 重建
4. IK 优化: min (接触距离 + 法线对齐 + rest pose 正则)
```

**骨段比例**: logmap 的内蕴几何天然适应不同表面形状, 但不做显式缩放 — 目标手太小时直接失败
**对我的价值**: 证明 "先迁移接触意图, 再求解 IK" 的分解策略可行; logmap 是 morphology-agnostic 的接触表示
**局限**: 需用户交互选 4 个锚点; 无碰撞检测; 纯运动学

---

## QuasiSim (THU, ECCV 2024) ⭐

**一句话**: 参数化准物理模拟器, 从松约束到紧约束 curriculum 优化, 将人手操作轨迹迁移到 Shadow Hand

**做法**:
```
将铰接体松弛为质点集 (参数 α 控制松弛程度):
  α→1: 点可自由运动 (容易优化)
  α→0: 退化为刚体 (物理真实)

三阶段 curriculum:
  Stage 1: α=0.1→0, 最软接触, track 人手轨迹
  Stage 2: 冻结 α=0, 收紧接触模型 (8 个子阶段)
  Stage 3: 激活 residual physics network, 逼近真实仿真器
```

**骨段比例**: point set 表示天然适应不同 morphology — 两侧都是质点集, α 松弛吸收差异
**对我的价值**: point set 表示无需手动关节对应; curriculum 从易到难是 contact-rich 优化的通用策略
**局限**: 计算极昂贵 (三阶段迭代); 只验证了 human→Shadow Hand 单方向; 未做 sim-to-real

---

## Contact Retargeting (Stanford, 2024) ⭐

**一句话**: 将 extrinsic manipulation (利用环境接触的操作) 分解为 contact configuration 原语序列, 迁移到新物体/新环境

**做法**:
```
操作 = contact configuration 序列 σ₀ → σ₁ → ... → σ_N
每个 σ 定义: 环境-物体接触 + 机器人-物体接触

retarget_x: demo 物体状态 → test 环境 (保持相对 transform)
retarget_q: 在接触切换点求解 IK (指尖满足接触关系)

原语库: Push/Pull/Pivot/Grasp (手动设计)
```

**骨段比例**: 不涉及 — 用的是 Franka 2 指夹爪, 不是灵巧手
**对我的价值**: "将操作分解为 contact configuration 序列, 各自独立 retarget" 的思路可借鉴到灵巧手
**局限**: 仅 parallel jaw gripper; 原语手动设计

---

## Hand Motion Mapping Review (Meattini, IEEE TRO 2022) — 综述

**一句话**: human-to-robot hand motion mapping 全景分类, 6 大类映射范式

**六大分类**:
```
1. Direct Joint    — 关节角一对一映射, 最简单但比例敏感
2. Direct Cartesian — 指尖位置匹配 + IK, 不保持手形
3. Task-Oriented   — virtual object 中间表示, 最接近 morphology-agnostic ⭐
4. Dimensionality Reduction — PCA synergy 低维子空间, 天然跨手
5. Posture Recognition — 离散分类, 不连续
6. Hybrid          — 组合上述方法
```

**核心洞察**:
- **所有传统方法都需要某种 manual prior** (关节对应/workspace 标定/synergy 数据)
- **Task-oriented 的 virtual object 思想最接近 morphology-agnostic** — 用任务空间中间表示解耦手结构差异
- 处理 non-anthropomorphic hand 是最大的开放挑战

**对我的价值**: 全景地图 — 快速定位自己的方法在整个领域中的位置; 确认了 "scalable without manual prior" 是公认的开放问题

---

## 12 篇论文全景排序

```
方法论贡献 (如何做 retarget):
  ⭐⭐⭐ GeoRT         — 几何 loss + MLP, 1KHz live, 最 scalable
  ⭐⭐⭐ CMU Kinematic  — atlas/logmap 接触转移, bone-length-invariant
  ⭐⭐  Harmonic/HAE   — 流形映射理论, 最少标注 (~8 对)
  ⭐⭐  Contact Transfer — logmap 接触迁移, CMU 的前驱
  ⭐⭐  SPIDER          — 物理采样优化, 2.4M 帧 scale
  ⭐   QuasiSim        — point set curriculum, 思路新颖但昂贵
  ⭐   DexFlow          — 逐指优化 + 292K 数据集

理论/背景 (理解问题空间):
  ⭐⭐  Meattini Review  — 6 大类全景, 确认开放问题
  ⭐   ARAP             — 为什么刚性保持不适合跨比例
  ⭐   Wu Contact Retarget — contact config 分解思路

Baseline (需要被超越的):
  DexPilot  — 事实标准, 完全 hand-crafted
  AnyTeleop — 工程最完整, retarget 本身仍是 manual prior
```

---

## 分类索引 (按方法范式)

论文按 learning-based vs optimization-based 分类存放:

```
retargeting/
├── core/                    # 综述 + 蒸馏笔记
│   ├── 22_Meattini_Survey   # 综述, 不属于任何一类
│   └── distill.md           # 本文件
│
├── learning/                # Learning-based: 离线训练 NN, 推理时前馈
│   ├── 25_GeoRT             # MLP 前馈映射, 5 geometric loss 自监督
│   └── 21_HarmonicMapping   # Contractive Autoencoder, 流形映射
│
└── optimization/            # Optimization-based: 每帧/每序列在线求解
    ├── 07_ARAP              # 交替优化 (SVD + 线性系统)
    ├── 20_DexPilot          # 非线性优化 (SLSQP), hand-crafted 向量对
    ├── 22_ContactTransfer   # logmap 接触迁移 + IK 优化
    ├── 23_AnyTeleop         # keypoint-vector 匹配优化
    ├── 24_CMU_KinematicRetarget  # 帧独立 IK + atlas 接触转移
    ├── 24_ContactRetarget   # contact config 分解 + IK
    ├── 24_QuasiSim          # 参数化准物理仿真 curriculum 优化
    └── 25_SPIDER            # 采样优化 (Annealed MPPI)
```

分类标准: **推理时是否需要迭代求解**。learning 方法离线训练后推理时一次 forward pass; optimization 方法每帧/每序列都要跑优化循环。QuasiSim 虽然有 residual physics network, 但核心是三阶段 curriculum 迭代优化, 归入 optimization。
