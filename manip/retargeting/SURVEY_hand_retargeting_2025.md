# 手部运动重定向 -- 综合调研与研究方向

> 日期: 2026-04-14, 经 CG/生物力学/机器人三方专家审查修正
> 项目: MediaPipe 21 点 -> WujiHand 20 DOF, Interaction Mesh Laplacian + SQP/SOCP
> 目的: 确定优化 formulation 变革 (B) 和 learning-based (C) 两个方向的可行路径

---

## 1. 当前系统的核心问题 (经 7 组实验验证)

```
反弓 (hyperextension) 根因量化:
  ~70% 骨段比例不匹配      ← Wuji baseline (无 Laplacian) 也有 84.1% 反弓
  ~15% Delaunay 跨手指耦合  ← bone scaling 失败, ARAP edge+Delaunay 89.6% 更差
  ~15% URDF 关节下限过松    ← PIP 允许 -27 度

Laplacian 的真正问题 (非"丢方向"):
  - Laplacian 坐标 = mean curvature normal, 方向信息存在
  - 问题是: 在全局坐标系下定义, 不随局部变形旋转 (Sorkine 2006)
  - Interaction Mesh 中 PIP 的 |L| 近 0 是因为处于凸包内部, 约束力弱

Delaunay 跨手指耦合是多个问题的共同根因:
  EXP-6: bone scaling adaptive alpha 收敛到 1.0 (跨指冲突自行取消)
  EXP-7: ARAP edge + Delaunay 89.6% > Laplacian 74.9% (跨指 R_i 互相矛盾)
  EXP-7: ARAP edge + 骨架拓扑 63.6%, DIP 反弓归零 (per-finger 独立链消除耦合)
```

---

## 2. Direction B: 优化 formulation 变革

### 2.1 问题定义

当前 formulation: `min ||L @ robot_pts - L @ source_pts||²`

三个独立缺陷:
1. **旋转敏感**: L 在全局坐标系, 大变形时目标方向失真
2. **凸包内部弱约束**: PIP 等内部点 |L| 近 0, 优化器自由度大
3. **Delaunay 跨手指耦合**: 非语义邻接导致优化目标冲突

### 2.2 已有理论工具 (CG 领域, 按演进排列)

| 年份 | 方法 | 解决了什么 | 没解决什么 | 本地 |
|------|------|-----------|-----------|------|
| 2004 | **Laplacian Editing** (Sorkine) | 稀疏线性系统, detail preservation | 旋转敏感 | theory/ |
| 2004 | **RSI Laplacian** (同篇) | similarity transform: 旋转+均匀缩放 | 非均匀缩放, 接触 | theory/ |
| 2004 | **Deformation Transfer** (Sumner) | 传递变形梯度, 天然跨尺度 | 需 mesh 对应 | 未收集 |
| 2005 | **Rotation-Invariant Coords** (Lipman) | 完全旋转不变, 线性重建 | 需 mesh | 未收集 |
| 2006 | **Diff Repr Survey** (Sorkine) | 统一理论框架 (到 Lipman 2005) | 不含 ARAP | theory/ |
| 2007 | **ARAP** (Sorkine & Alexa) | per-cell 最优旋转 | 保边长, 非线性迭代 | optimization/ |
| 2010 | **Interaction Mesh** (Ho) | multi-body 空间关系, Delaunay | 跨体耦合, 旋转敏感 | OmniRetarget ref/ |
| 2011 | **BBW** (Jacobson) | shape-aware 蒙皮权重 | 需骨骼定义 | theory/ |
| 2014 | **Projective Dynamics** (Bouaziz) | ARAP 实时扩展, 混合约束 | 学术实现 | 未收集 |
| 2025 | **ReConForM** (Cheynel) | pairwise descriptor + 自适应时空权重 | 需 mesh key-vertices | learning/ |
| 2025 | **OmniRetarget** (Yang) | Laplacian + 硬约束 (碰撞/脚/关节) | 旋转敏感, Delaunay 耦合 | 移植来源 |

### 2.3 可行思路

**思路 B1: 选择性拓扑 + 混合能量 (最小改动, 最高信心)**

```
核心洞察 (EXP-7 的教训):
  Delaunay:  有跨指关系 → 但全连接耦合导致 bone scaling 失败, R_i 矛盾
  纯骨架:    无跨指关系 → DIP 修复, 但退化成逐段追点 (≈ wuji baseline ||vec - α*vec||²)
  → interaction mesh 的核心价值 (跨体空间关系如 pinch 距离) 被纯骨架完全丢掉
  → 需要: 骨架链保方向 + 少量语义跨指边保交互, 不是全连接也不是零连接

cost = w_edge * Σ_{skeleton} ||(r_i-r_j) - R_i(s_i-s_j)||²  # 骨架边: per-finger 方向
     + w_inter * Σ_{cross_finger} ||d_robot - d_source||²     # 跨指边: 交互保持
     + w_hyper * ||max(0, -q_pip_dip)||²                       # 反弓惩罚

骨架边 (20 条): Wrist→MCP→PIP→DIP→TIP per finger
跨指边 (选择性, ~6-10 条):
  必选: thumb_tip ↔ index_tip (pinch)
  推荐: thumb_tip ↔ {mid,ring,pinky}_tip (多指 pinch)
  可选: adjacent_MCP pairs (手掌展开/收拢)

依据: EXP-7 骨架 DIP 归零 + 当前 pinch 0.88mm 需保留 + Delaunay 耦合分析
改动量: ~40 行
风险: 低. 跨指边数量和权重可独立调控
```

**思路 B2: RSI Laplacian / ASAP 变体 (理论优雅, 中等改动)**

```
核心: 用 Sorkine 2004 的 similarity transform 替代当前 naive Laplacian

每个顶点 i 求解最优 similarity transform T_i = s_i * R_i + t_i:
  min Σ_i Σ_{j∈N(i)} ||(p'_j - p'_i) - T_i(p_j - p_i)||²

T_i 包含旋转 R_i + 均匀缩放 s_i, 同时解决:
  - 旋转不变性 (ARAP 的 R_i)
  - 均匀缩放 (ARAP 做不到的)

与 ARAP 的区别: ARAP 的 R_i ∈ SO(3), ASAP 的 T_i ∈ Sim(3)
对手部: source/robot 骨段长度不同 → s_i 自动吸收比例差异

依据: Sorkine 2004 Section 3 (RSI Laplacian coords), 已有理论和实现
改动量: ~50 行 (替换 solve_single_iteration 中的 cost 构造)
风险: 中. 非线性迭代 (交替求 T_i 和 q), 收敛性需验证
```

**思路 B3: ReConForM 风格自适应 pairwise (中期目标)**

```
核心: 完全替换 Laplacian, 用 pairwise descriptor + 自适应时空权重

cost = Σ_{(i,j)} w(i,j,t) * ||d_robot^{ij} - d_source^{ij}||²

w(i,j,t) = f(distance(i,j,t), contact_state(i,j,t), temporal_proximity_to_contact)

pair 选择:
  必选 (20 条): 骨架拓扑边
  功能 (10 条): 拇指-各指 TIP, 相邻指间
  动态: 距离 < 30mm 的所有对, 随帧变化

注意: 21 点稀疏场景, 活跃约束可能不足 → 需关节空间正则化 (smooth + Q_diag) 补偿全局耦合缺失

依据: ReConForM (EG 2025) 在 41 key-vertices 上验证, 67fps 实时
改动量: ~100 行 (重写 cost 构造, pair 管理, 权重计算)
风险: 中高. 全局耦合缺失可能导致欠约束, 需仔细调正则化
```

### 2.4 不推荐的思路 (经实验/审查否决)

| 思路 | 否决原因 |
|------|---------|
| ARAP rotation compensation (EXP-4) | 2.5x 速度代价, 间接路径不可靠, Q_diag 更优 |
| DMI field (MeshRet) | 需 skinned mesh + tangent space, 21 点云不适用 |
| Functional Maps 自动对应 | 需连续 2-流形, robot URDF 多刚体不满足 |
| 纯 ARAP edge + Delaunay (EXP-7) | 跨指耦合让 R_i 互相矛盾, 反弓比 Laplacian 更差 |

---

## 3. Direction C: Learning-Based Retargeting

### 3.1 问题定义

```
输入: MediaPipe 21 点手部关键点 (30Hz, 可能含噪声)
输出: WujiHand 20 DOF 关节角
要求: <5ms/frame (live), contact-aware, 跨任务泛化

当前最优化 pipeline: ~50fps, pinch 间距 0.88mm (优于 baseline), 但反弓严重
目标: 接近 GeoRT 的 1KHz 速度, 同时保持/超越当前 pinch 质量
```

### 3.2 已有方法分类

**纯 retargeting (不含下游 RL):**

| 方法 | 类型 | 速度 | 接触感知 | 本地 |
|------|------|------|---------|------|
| GeoRT (Meta 2025) | MLP, 几何 loss | 1KHz | 无 (仅 L_pinch 距离) | learning/ |
| ReConForM (EG 2025) | 优化, pairwise descriptor | 67fps | 自适应权重 | learning/ |
| Skeleton-Aware (SIGGRAPH 2020) | GCN, cycle consistency | 实时 | 无 | OmniRetarget ref/ |
| MeshRet (NeurIPS 2024) | Transformer, DMI field | 离线 | DMI 统一编码 | learning/ |

**retargeting + 下游 (重点在 RL, 非纯 retargeting):**

| 方法 | 范式 | 核心 insight | 本地 |
|------|------|-------------|------|
| SPIDER (Meta 2025) | physics sampling + curriculum | contact gradient 不可靠, 用 sampling 替代 | optimization/ |
| ManipTrans (CVPR 2025) | kinematic retarget + residual RL | 分离 kinematic 和 dynamic | 未收集 |
| DexMachina (2025) | functional retarget + curriculum RL | 保持功能而非姿态 | 未收集 |

### 3.3 可行思路

**思路 C1: GeoRT 变体 -- 加入指间 coupling (最直接)**

```
核心: 基于 GeoRT 框架, 修复其 per-finger 独立 MLP 的局限

GeoRT 的 5 个 loss: L_fk (距离), L_pinch (对指), L_dir (方向), L_col (碰撞), L_flat (C-space)
问题: 每指独立 MLP, 缺乏 ring-pinky 联动、包裹抓取协同

改进: 共享底层特征 + per-finger head, 或直接用一个 MLP 输出 20 DOF
加入: 骨架拓扑 pairwise loss (复用 B1 的 functional pairs 定义)

训练数据: MuJoCo 随机采样 robot 关节角 → FK → 指尖位置 (和 GeoRT 一样)
推理: <1ms, 天然 live retargeting

依据: GeoRT 代码已在本地 (learning/25_GeoRT/GeoRT/)
改动量: 新建训练脚本 (~300 行), 网络结构改动 (~50 行)
```

**思路 C2: Kinematic retarget + Residual RL (最成熟的 physics 引入)**

```
核心: 当前优化 pipeline 做 kinematic backbone, RL 学 contact-aware residual

Stage 1: 现有 pipeline (Laplacian/pairwise) 输出 q_kinematic
Stage 2: RL policy 输出 delta_q = f(q_kinematic, observation)
         → q_final = q_kinematic + delta_q
Stage 2 在 MuJoCo 中训练, 有物理接触反馈

优势: 不需要改 Stage 1, 现有 pipeline 直接作为 backbone
ManipTrans (CVPR 2025) 已验证: 73%+ 成功率

依据: ManipTrans, DexH2R, PKDA 多篇 2024-2025 顶会验证
改动量: 新建 RL 训练环境 (~500 行), policy 网络 (~200 行)
```

**思路 C3: 条件化 synergy autoencoder (长期探索)**

```
核心: 不用 PCA (Todorov 证明 task-dependent), 用条件化 VAE

encoder: 21 点 → latent z (task-conditioned)
decoder_human: z → 27 DOF human joint angles
decoder_robot: z → 20 DOF WujiHand joint angles

训练:
  - encoder + decoder_human: 自监督, 重建 MediaPipe 序列
  - decoder_robot: 用 retarget 数据 (当前 pipeline 输出) 做弱监督
  - latent alignment: 功能性对齐 (相同 grasp → 相近 z)

比 eigengrasp 的优势: 非线性, task-conditioned, 可增量学习新任务
比 GeoRT 的优势: latent space 可做插值、生成、异常检测

依据: Todorov 2004 (PCA 不够), SAME (SIGGRAPH Asia 2023), WalkTheDog (SIGGRAPH 2024)
改动量: 新建训练框架 (~1000 行)
风险: 高. latent alignment 是开放问题
```

### 3.4 不推荐的思路

| 思路 | 否决原因 |
|------|---------|
| DMI field + Transformer (MeshRet) | 需 skinned mesh, MediaPipe 21 点不适用 |
| CrossDex eigengrasp universal space | 实际是 learned latent + per-hand decoder, 非经典 PCA; 且 eigengrasp 是 task-dependent (Todorov) |
| 纯 PCA synergy retargeting | Santello 2 PC 仅限 imagined static grasping; 动态操作需 6+ 维且 task-dependent |

---

## 4. 推荐执行顺序

```
Phase 1 (立即, 1-2 天):
  B1 的最小版: Q_diag 反弓惩罚 + 收紧 PIP 下限 + 骨架拓扑+骨段缩放组合
  → 验证: 反弓率 < 50%, 指尖误差 < 13mm

Phase 2 (短期, 1 周):
  B1 完整版: 骨架 Laplacian + pairwise functional pairs + Q_diag
  → 验证: 反弓率 < 30%, pinch 间距 < 1mm, 速度 > 50fps

Phase 3 (中期, 2-4 周):
  B2: RSI Laplacian / ASAP (如果 B1 的方向误差仍高)
  或 C1: GeoRT 变体 (如果需要 live retargeting)

Phase 4 (中长期):
  C2: Residual RL (当需要物理接触保真时)
  B3: ReConForM 自适应权重 (如果 pairwise 在 HO-Cap 数据上需要)

Phase 5 (长期探索):
  C3: 条件化 synergy autoencoder
```

---

## 5. 论文集

### 已在本地 (24 篇)

**optimization/ (8)**
| 编号 | 论文 | 会议/年份 |
|------|------|----------|
| 1 | ARAP Surface Modeling (Sorkine & Alexa) | SGP 2007 |
| 2 | DexPilot (Handa et al.) | ICRA 2020 |
| 3 | Contact Transfer (CMU, Lakshmipathy) | ICRA 2022 |
| 4 | AnyTeleop (Qin et al.) | RSS 2023 |
| 5 | CMU Kinematic Retarget (Lakshmipathy et al.) | TOG 2025 |
| 6 | Contact Retarget (Wu et al.) | 2024 |
| 7 | QuasiSim (Liu et al.) | ECCV 2024 |
| 8 | SPIDER (Meta FAIR) | arXiv 2025 |

**learning/ (4)**
| 编号 | 论文 | 会议/年份 |
|------|------|----------|
| 9 | Harmonic Mapping (Chong et al.) | IJRR 2021 |
| 10 | MeshRet (Yang et al.) | NeurIPS 2024 Spotlight |
| 11 | GeoRT (Meta FAIR) | IROS 2025 |
| 12 | ReConForM (Cheynel et al.) | EG/CGF 2025 |

**theory/ (9)**
| 编号 | 论文 | 会议/年份 |
|------|------|----------|
| 13 | Hand Synergies (Santello et al.) | J Neurosci 1998 |
| 14 | Laplacian Editing + RSI (Sorkine et al.) | SGP 2004 |
| 15 | Hand Synergies Dynamic (Todorov & Ghahramani) | IEEE EMBC 2004 |
| 16 | Diff Representations Survey (Sorkine) | CGF 2006 |
| 17 | Eigengrasps (Ciocarlie & Allen) | IJRR 2009 |
| 18 | BBW (Jacobson et al.) | SIGGRAPH 2011 |
| 19 | Functional Maps (Ovsjanikov et al.) | SIGGRAPH 2012 |
| 20 | Arm Dimensionality Reduction (Gloumakov et al.) | 2020 |
| 21 | DPFM (Attaiki et al.) | 3DV 2021 |

**core/ (2)**
| 编号 | 论文 | 会议/年份 |
|------|------|----------|
| 22 | Meattini Survey | TRO 2022 |
| 23 | distill.md (12 篇蒸馏) | -- |

**humanoid/ (关联)**
| 编号 | 论文 | 位置 |
|------|------|------|
| 24 | OmniRetarget (含 Interaction Mesh 原文 + 6 篇 ref) | humanoid/teleoperation/ |
| 25 | GMR (全身 retarget) | humanoid/retargeting/ |

### 值得收集 (未在本地)

**Direction B 高优先级:**
| 论文 | 会议/年份 | 价值 |
|------|----------|------|
| Deformation Transfer (Sumner & Popovic) | SIGGRAPH 2004 | 传递变形梯度, 天然跨尺度 |
| Rotation-Invariant Coords (Lipman et al.) | SMI 2004/2005 | Laplacian→ARAP 关键中间形式, 线性重建 |
| Projective Dynamics (Bouaziz et al.) | SCA 2014 | ARAP 实时扩展, 混合多类约束 |
| Analyzing Key Objectives in H2R Retargeting (Xie) | arXiv 2025 | fingertip orientation 应在 objective 中的实验证据 |

**Direction C 高优先级:**
| 论文 | 会议/年份 | 价值 |
|------|----------|------|
| ManipTrans (Li et al.) | CVPR 2025 | kinematic retarget + residual RL, 最成熟范式 |
| DexMachina (Zhao et al.) | arXiv 2025 | functional retargeting 概念 |
| Skeleton-Aware Networks (Aberman et al.) | SIGGRAPH 2020 | learning retarget 奠基, 已在 OmniRetarget ref/ |

**理论补充:**
| 论文 | 会议/年份 | 价值 |
|------|----------|------|
| Todorov & Ghahramani 2009 扩展版 | IEEE TNSRE | 动态 synergy 完整分析 |
| As-Rigid-As-Possible Shape Interpolation (Alexa 2000) | SIGGRAPH | ARAP 概念起源 |

---

## 6. 修正记录

| 日期 | 修正内容 |
|------|---------|
| 2026-04-14 初版 | 44 篇调研, B/C 路径初步建议 |
| 2026-04-14 专家审查 | 7 条错误判断修正 (Laplacian/ARAP/eigengrasp/DMI/Functional Maps/Sorkine 时间线) |
| 2026-04-14 实验整合 | EXP-1~7 结论纳入, 反弓根因量化, 骨架拓扑验证结果 |
| 2026-04-14 路径重写 | B: 骨架拓扑优先 → 混合能量 → RSI/ASAP; C: 当前不急 → GeoRT 变体 → residual RL |
