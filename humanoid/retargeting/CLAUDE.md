# Retargeting -- 人形全身运动重定向

> 目标: 高质量 human->humanoid 全身运动映射, 为 RL motion tracking 提供干净参考轨迹
> 手部 retarget 见 `manip/retargeting/`

## 目录

```
retargeting/
└── 25_GMR (arXiv 2510.02252)         # General Motion Retargeting, Stanford
    # 非均匀局部缩放 + 两阶段 diff-IK (mink), 支持 17+ 人形机器人
    # 输入: SMPL-X / BVH / FBX -> 输出: robot (base_trans, base_rot, joint_pos)
    # 60-70 FPS 实时, 用于遥操 (TWIST) 和 RL tracking policy 训练数据生成
    # 对比 PHC/ProtoMotions, 在 BeyondMimic 评测中接近 Unitree 闭源质量
    # 代码: github.com/YanjieZe/GMR
    ├── GMR_2510.02252.pdf             #   论文 PDF
    ├── GMR_2510.02252.md              #   论文 markdown (arxiv2md 转换)
    ├── general_motion_retargeting/    #   核心代码
    ├── assets/                        #   17+ 机器人 MuJoCo 模型
    └── scripts/                       #   retarget 脚本入口
```

## 本目录外的 retarget 相关内容

| 位置 | 角色 | 说明 |
|------|------|------|
| `teleoperation/25_OmniRetarget/` | **方法** | Interaction mesh retarget + 硬约束优化 (SQP), 建模人-物接触, 核心贡献就是 retarget |
| `motion_tracking/23_PHC/` | **baseline** | SMPL fit robot + gradient IK, GMR 论文中作为对比方法; `PHC/docs/retargeting.md` 有使用文档 |
| `motion_tracking/25_BeyondMimic/` | **评测框架** | 用于评估 retarget 质量对 RL policy 的影响, GMR 论文的主要评测工具 |
| `video_world_model/25_HDMI/` | **下游应用** | video -> retarget -> RL pipeline, 证明单目 RGB 也能作为 retarget 数据源 |
| `sim2real/25_ASAP/` | **下游应用** | `fit_smpl_motion.py` 中有 SMPL retarget 工具代码 |

## 三种方法对比

```
                       GMR              OmniRetarget           PHC
                       ─────────────    ─────────────────      ───────────────
缩放策略               per-body 非均匀   全局缩放              SMPL fit robot
IK 求解器              mink (diff-IK)   SQP 约束优化           gradient descent
接触建模               无               interaction mesh       无
物理约束(碰撞/关节极限) 关节极限          碰撞+关节极限+脚不滑    关节极限(clamp)
实时性                 60-70 FPS        离线                   离线
支持机器人数            17+              G1, H1, T1             G1 (主要)
核心优势               速度+通用性       数据质量(接触保持)      生态广泛(后续工作多)

共同结论: retarget 质量 >> reward engineering
  GMR: "retarget artifact 显著降低 policy 鲁棒性"
  OmniRetarget: "5 个 reward term + 高质量数据 > 10+ reward terms + 低质量数据"
```

## 核心认知

```
全身 retarget 的两个关键瓶颈:

1. 缩放策略 (scaling) -- GMR 主攻
  ✗ SMPL fit robot (PHC)           — 人体模型拟合机器人, 形态差异大时失真
  ✗ 全局轴对齐缩放 (ProtoMotions)   — 无法处理上下半身比例差异
  ✓ 非均匀局部缩放 (GMR)           — per-body 缩放因子, 灵活处理比例差异

2. 接触/交互保持 -- OmniRetarget 主攻
  ✗ 关键点匹配 (PHC, GMR)          — 只做位置/旋转匹配, 不建模人-物接触
  ✓ Interaction mesh (OmniRetarget) — Delaunay 四面体化 + Laplacian 变形能量
    + 硬约束: 碰撞、关节极限、脚不滑动
    代价: 离线优化, 不能实时

GMR Pipeline (5 步):
  1. Key body matching         — 人-机关节映射 + 权重
  2. Rest pose alignment       — 静态姿态旋转对齐
  3. Non-uniform local scaling — h/h_ref * s_b, per-body 缩放
  4. Two-stage diff-IK         — Stage 1: rotation + end-effector pos
                                 Stage 2: fine-tune all body positions
  5. Height correction         — 减去最低 body 高度, 消除浮空/穿地

必须避免的 retarget artifact (直接影响 RL policy 鲁棒性):
  1. 脚部穿透地面 (ground penetration) — PHC 常见
  2. 自交叉 (self-intersection) — ProtoMotions 常见
  3. 关节角突变 (sudden jumps in joint values) — GMR 偶发
  4. 脚滑动 (foot skating) — 无接触约束时常见, OmniRetarget 用硬约束解决
```

## 与手部 retarget 的思路关联

```
共通点:
  - 形态差异 (embodiment gap) 是两者的核心挑战
  - 缩放策略 (scaling) 都是关键: 手部比例差异 ~ 全身比例差异
  - IK 求解器选择: GMR 和手部方法都用 mink (diff-IK)
  - artifact 对下游 RL 的影响模式一致
  - "data quality > reward engineering" 在两边都成立

可借鉴:
  手部 -> 人形: 接触保持 (contact-aware) 目前只有手部做了细粒度接触迁移
  人形 -> 手部: GMR 非均匀局部缩放可迁移到手指级别 per-joint scaling
  OmniRetarget 的 interaction mesh 理论上可同时用于手部和全身
```
