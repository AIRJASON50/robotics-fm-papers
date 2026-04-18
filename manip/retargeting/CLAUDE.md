# Retargeting -- 灵巧手运动重定向

> 目标: scalable 手部 retarget, 在 RL 之前获得好的重定向结果
> 人形全身 retarget 见 `humanoid/retargeting/`
> 综合调研见 `SURVEY_hand_retargeting_2025.md`

## 目录

```
retargeting/
├── core/                              # 蒸馏 + 综述
│   ├── distill.md                     #   全部蒸馏 + 方法对比 + 全景排序
│   └── 22_Meattini_Survey             #   6 大类全景综述 (TRO 2022)
│
├── optimization/                      # 优化类方法 (11 篇)
│   ├── 07_ARAP                        #   刚性保持变形 (SGP 2007, Sorkine)
│   ├── 20_DexPilot                    #   指间距优化 baseline (ICRA 2020)
│   ├── 22_ContactTransfer             #   接触区域迁移 (ICRA 2022, CMU)
│   ├── 23_AnyTeleop                   #   position/vector/DexPilot 三模式 (RSS 2023)
│   ├── 24_CMU_KinematicRetarget       #   非等距 shape matching + contact (TOG 2025, CMU)
│   ├── 24_ContactRetarget             #   contact config 分解 (Wu 2024)
│   ├── 24_QuasiSim                    #   point set curriculum (ECCV 2024)
│   └── 25_SPIDER                      #   物理采样 + curriculum contact (arXiv 2025, Meta)
│
├── learning/                          # 学习类方法 (7 篇)
│   ├── 21_ContactAwareRetarget        #   contact 一致性优化, skinned mesh (ICCV 2021, Villegas)
│   ├── 21_HarmonicMapping             #   流形映射 (IJRR 2021)
│   ├── 23_ACE                         #   GAN 跨形态 retarget, Spot 机器人验证 (SIGGRAPH Asia 2023)
│   ├── 23_Semantics2Hands             #   ASM 语义矩阵, 解耦接触与姿态 (ACM MM 2023)
│   ├── 24_MeshRet                     #   Dense Mesh Interaction field (NeurIPS 2024 Spotlight)
│   ├── 25_GeoRT                       #   几何 loss + MLP, 1KHz (IROS 2025, Meta)
│   └── 25_ReConForM                   #   自适应 descriptor + 实时 contact-aware (EG 2025)
│
├── CG_transfer/                       # CG 骨架/mesh 迁移方法论 (8 篇)
│   ├── 04_DeformationTransfer         #   变形梯度跨 mesh 迁移 (SIGGRAPH 2004, Sumner)
│   ├── 09_SemanticDT                  #   语义变形迁移, 少量示例对 (SIGGRAPH 2009, Baran)
│   ├── 18_NKN                         #   可微 FK + cycle consistency 无监督 retarget (CVPR 2018, Villegas)
│   ├── 20_SkeletonAware               #   skeletal pooling 跨结构 retarget (SIGGRAPH 2020, Aberman)
│   ├── 23_HandAvatar                  #   手→任意 avatar 跨拓扑映射, 三目标优化 (CHI 2023, CMU)
│   ├── 23_R2ET                        #   残差网络 skeleton+shape-aware retarget, normalized DM (CVPR 2023, Zhang)
│   ├── 23_SAME                        #   skeleton-agnostic GCN, 任意拓扑 (SIGGRAPH Asia 2023, Lee)
│   └── 25_Motion2Motion               #   training-free 跨拓扑 patch matching, 稀疏对应 (SIGGRAPH Asia 2025, Chen)
│
├── theory/                            # 理论基础 (5 篇 + IM 专题)
│   ├── 98_HandSynergies               #   2 PC 解释 80%+ 手姿态方差 (J Neurosci 1998, Santello)
│   ├── 04_LaplacianEditing            #   Laplacian 坐标 mesh 编辑 (SGP 2004, Sorkine)
│   ├── 06_DiffRepresentations         #   微分表示综述: Laplacian->ARAP 全链条 (CGF 2006, Sorkine)
│   ├── 09_Eigengrasps                 #   eigengrasp 子空间抓取 (IJRR 2009, Ciocarlie & Allen)
│   ├── 20_ArmDimensionalityReduction  #   上肢 7/4/3 DOF 运动降维聚类 (Gloumakov 2020)
│   └── IM/                            #   Interaction Mesh 专题 (4 篇, 2010-2024)
│       ├── 10_Ho/                     #     奠基: 距离权重 + Laplacian (SIGGRAPH 2010)
│       ├── 12_Nakaoka/                #     机器人首次应用 (Humanoids 2012)
│       ├── 16_Kim/                    #     体积映射路线: 余切权重+固定拓扑 (TVCG 2016)
│       ├── 23_Zhang/                  #     边长比较+指数距离权重 (SIGGRAPH 2023)
│       ├── 24_Jang/                   #     几何感知 Transformer (TOG 2024, 待下载)
│       └── IM_series_notes.md         #     引用链 + 设计选择对比
│
└── SURVEY_hand_retargeting_2025.md    # 综合调研 (44 篇, 2026-04-14)
```

## 核心认知

```
retarget 方法评估 (经实验和专家审查修正, 2026-04-14):
  ✗ Laplacian       — 旋转敏感 (非"丢方向", 方向在但不随变形旋转), 凸包内部点约束力弱
  ✗ ARAP (标准)     — 保边长, 抵抗比例变化
  △ ASAP/RSI        — Sorkine 2004 的 similarity transform 变体, 允许均匀缩放, survey 此前遗漏
  ✓ Atlas/LogMap    — 内蕴测地坐标, 天然比例无关
  ✓ Geometric Loss  — C-space Chamfer 自动学非线性缩放 (GeoRT)
  ✓ Harmonic Map    — 最小畸变映射, 理论最优雅
  △ Pairwise Desc.  — per-edge 方向完整, 但 21 点稀疏场景丢失全局耦合, 需正则化补偿
  ✗ DMI Field       — 需 skinned mesh + tangent space, 对 MediaPipe 21 点云不适用

反弓的真正根因 (经实验验证):
  ~70% 来自骨段比例不匹配 (Wuji baseline 也有 84.1% 反弓)
  ~15% 来自 Laplacian 旋转敏感性 + Delaunay 跨手指耦合
  ~15% 来自 URDF 关节下限过松 (PIP 允许 -27 度)

Delaunay 跨手指耦合是多个问题的共同根因:
  bone scaling 失败 (alpha 收敛到 1.0)
  ARAP edge+Delaunay 比 Laplacian 更差 (89.6% vs 74.9%)
  → 骨架拓扑 (20 条边, per-finger 独立链) 是正确方向
  → EXP-7 验证: Edge+Skeleton DIP 反弓归零, 整体 63.6%

retarget 的本质:
  1. 接触区域分布 (CMU: "接触即操作本质")
  2. 运动方向 + C-space 覆盖 (GeoRT)
  3. 两个流形间的最小畸变映射 (HAE)
  4. pairwise 空间关系保持 (Interaction Mesh / ReConForM)
  5. 骨架拓扑对应 + 语义保持 (CG_transfer/ 系列新增认知):
     字面对应 (关节角复制) 在同构手间可用, 跨结构手必须用语义对应
     Aberman: static(骨架结构) / dynamic(关节旋转) 解耦是核心
     SAME: 单网络+attention 自动发现跨拓扑对应, 最 scalable

eigengrasp/synergy 限制 (专家审查纠正):
  Santello 1998: 2 PC = 80%+ 仅对 imagined static grasping, 不推广到动态操作
  Todorov 2004: 操作任务有效维度 ~6.5, synergy 结构 task-dependent > subject-dependent
  Ciocarlie 2009: eigengrasp 是 per-hand 独立定义, 从未做跨 embodiment 迁移
  → eigengrasp 适合单手降维/grasp planning, 不是"天然跨 embodiment 表示"
  → 跨 embodiment 需要额外 alignment mechanism, 这本身就是 retargeting 核心难题
```

## 理论链条

```
变形能演进 (修正: Sorkine 2006 不覆盖 ARAP, 时间线在其之前):
  Laplacian Editing (Sorkine 2004)         — 稀疏线性系统, 旋转敏感
    └→ RSI Laplacian (同篇)                — similarity transform (旋转+缩放), 已解决 scale
  Lipman rotation-invariant coords (2005)  — 切平面编码, 完全旋转不变, 线性重建
  Diff Representations Survey (Sorkine 2006) — 统一框架, 覆盖到 Lipman 2005 为止 (不含 ARAP)
  ARAP (Sorkine & Alexa 2007)              — per-cell 最优旋转, 非线性迭代
  Interaction Mesh (Ho 2010)               — Laplacian 用于 multi-body 空间关系保持
  ReConForM (2025)                         — per-pair descriptor, 自适应时空加权

CG 骨架/mesh 迁移演进 (CG_transfer/, 8 篇):
  Deformation Transfer (Sumner 2004)       — per-triangle 仿射变换跨 mesh 迁移, ~50 markers
  Semantic DT (Baran 2009)                 — 5-12 个示例 pose 对学语义对应 (非字面), patch-based LRI
  NKN (Villegas 2018)                      — 可微 FK + cycle consistency, 首个 learning-based 无监督 retarget
  Skeleton-Aware (Aberman 2020)            — skeletal pooling 归约到 primal skeleton, 跨结构但需同胚图
  R2ET (Zhang 2023, CVPR)                  — 残差结构: skeleton-aware (normalized DM) + shape-aware (RDF/ADF) + balancing gate
  HandAvatar (Jiang 2023, CHI)             — 手→任意 avatar 映射, 三目标优化 (precision/similarity/comfort)
  SAME (Lee 2023)                          — skeleton-agnostic GCN, 单网络处理任意拓扑, 打破同胚限制
  Motion2Motion (Chen 2025, SIG Asia)      — training-free 跨拓扑迁移, 稀疏对应 + patch matching, CPU-only
  → 演进: 字面迁移 → 语义迁移 → 无监督学习 → 跨结构 → 任意拓扑 → training-free few-shot

遗漏但可能重要的方法:
  Projective Dynamics (Bouaziz 2014)       — ARAP 的实时扩展, 混合约束
  Cage-based deformation (MVC, Green Coords) — 21 点天然就是 cage

Shape 对应 (已移出 theory/):
  Functional Maps (Ovsjanikov 2012)        — 需连续 2-流形, robot 多刚体 mesh 不满足 → 移出
  BBW (Jacobson 2011)                      — 依赖 Functional Maps 框架, 同样不适用 → 移出
  DPFM (Attaiki 2021)                      — 深度偏函数映射, 部分 shape 对应 → 移出
  → 更实用: skeleton mapping, 或 ReConForM 的 optimal transport on T-pose

手部低维表示 (修正: 不是"天然跨 embodiment"):
  Santello 1998                            — 2 PC, imagined static grasping only
  Todorov 2004                             — 动态操作维度 ~6.5, task-dependent structure
  Ciocarlie 2009                           — per-hand eigengrasp, 用于 grasp planning 降维
  CrossDex 2025                            — learned latent space + per-hand decoder, 非经典 PCA
```

## 两个研究方向 (经实验和专家审查修正)

### Direction B: 优化 formulation 变革

```
已验证的结论 (EXP-4 ~ EXP-7):
  1. ARAP rotation compensation: 间接路径, -25pp 反弓, -2.5x 速度 → 性价比低
  2. ARAP edge + Delaunay: 比 Laplacian 更差 (跨手指耦合)
  3. ARAP edge + 骨架拓扑: DIP 反弓归零, 整体最优 63.6%
  → 骨架拓扑是关键, 比换能量函数更重要

近期最佳路径 (按优先级):
  1. Q_diag 反弓惩罚 (~5 行) + 收紧 PIP 下限到 -5 度 (0 行)  ← 立即可做
  2. 骨架拓扑 + 骨段缩放组合测试  ← 验证跨指耦合消除后缩放是否重新有效
  3. Laplacian + pairwise functional pairs 混合 cost  ← 保留全局耦合, 叠加方向信息
     (不是"替代"Laplacian, 而是"互补")

中期路径:
  4. RSI Laplacian (Sorkine 2004 ASAP 变体) ← 解决旋转+缩放不变, survey 此前遗漏
  5. ReConForM 风格自适应时空权重
  6. 切换 QP solver (mink/DAQP) ← 迭代次数降到 3-5 次后再考虑
```

### Direction C: Learning-based retargeting

```
当前不急 — 优化空间未榨干 (骨架拓扑 + Q_diag 还没测试组合)

当优化 pipeline 稳定后:
  路径 1: GeoRT 几何 loss, 但加 per-finger coupling (GeoRT 每指独立 MLP 缺联动)
  路径 2: kinematic retarget (当前 pipeline) + residual RL (ManipTrans CVPR 2025)
  路径 3: 用 VAE/conditional autoencoder 替代 PCA 做 synergy (Todorov 证明 PCA 不够)
  路径 4: SAME 风格 skeleton-agnostic GCN (CG_transfer/ 新增)
    - 训练: 多种手骨架 (MANO, Allegro, Shadow, Wuji) + 抓取运动数据
    - 推理: 任意源手 -> 任意目标手, 单网络, 无需 per-pair 配置
    - 优势: scalable, 换手零配置; 可重建缺失关节运动
    - 挑战: 需要手部运动训练数据 (可用当前 pipeline 生成)
    - 依据: SAME 已验证 body 级别, 手部验证待做

DMI field (MeshRet): 不适用 — 需 skinned mesh, MediaPipe 21 点云无法支撑
```
