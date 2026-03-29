# DexMachina 分析笔记

## 概述

DexMachina 解决的是双手灵巧操作铰接物体的 **functional retargeting** 问题。
给定人手-物体交互的跟踪演示（ARCTIC 数据集），训练 RL 策略，用各种机器人手复现物体轨迹（位置、旋转、铰接关节角度）。

- 论文: arXiv 2505.xxxxx
- 代码: 基于 Genesis 的 RL，支持 6 种手型
- 任务: 5 种铰接物体 (box, notebook, mixer, waffleiron, ketchup)，长时序（最长 300 帧）

---

## 仿真设置

- **无机械臂**，仅浮动手
- 每个 URDF 手动添加 6-DOF 腕部关节（3 平移 + 3 旋转）
- 支持 6 种手型: Inspire, Allegro, XHand, Schunk, Ability, Dex3
- 仿真器: Genesis（非 IsaacGym），最多 12,000 并行环境

---

## 控制方案: Hybrid Action（纯 RL，无 BC）

### 腕部 (6-DOF): 基于 retarget 参考的残差控制

```
wrist_trans_target = retarget_qpos[t, :3] + scale_trans * a_wrist[:3]   # scale_trans = 0.04 (4cm)
wrist_rot_target  = retarget_qpos[t, 3:6] + scale_rot  * a_wrist[3:6]  # scale_rot   = 0.5 (rad)
```

- Base = 运动学 retarget 结果（来自 AnyTeleop [3] + object-aware 后处理）
- 残差范围很小: +/-4cm 平移, +/-0.5rad 旋转
- 有效地将腕部搜索空间约束在 demo 轨迹附近的窄带内

### 手指 (remaining DOFs): 绝对控制

```
finger_target = lower_limit + (upper_limit - lower_limit) * (a_finger + 1) / 2
```

- 不涉及参考动作。策略对手指关节有完全自由度。
- 原因（推断，论文未明确说明）:
  - 手指 retarget 质量差（频繁穿透物体网格，形态差异大）
  - 手指关节范围有限，搜索空间即使不加参考约束也可控

### 为什么不用全残差或全绝对?

消融实验（Section 5.2, Figure 8）在无 curriculum 设定下对比了三种方案:

| Action mode | Description | Result |
|-------------|-------------|--------|
| Absolute | All joints absolute | Worst (wrist search space too large) |
| Residual (loose) | All joints residual, wrist limits = full demo range | Middle |
| **Hybrid** | Wrist residual (tight +/-4cm/+/-0.5rad), fingers absolute | **Best** |

> "using more restrictive bounds on wrist motion results in the best overall performance"

核心洞察: **紧约束的腕部残差**比手指是否用参考更重要。

### Action smoothing

所有目标经过 EMA 平滑: `new_target = alpha * target + (1-alpha) * prev_target`

---

## Object-aware Retarget 后处理 (Appendix A.2)

**DexMachina 自己的贡献**（无引用先前工作）。

### 问题

纯运动学 retarget（AnyTeleop）只匹配指尖位置，忽略物体碰撞。
结果: 机器人手指频繁穿透物体网格。这导致:
1. 腕部残差控制的 base-action 被损坏
2. imitation reward 计算时 keypoint 位置不可行

### 解决方案: 基于物理的碰撞解析

对每个 demo 时间步 t:
1. **固定物体** 在 demo 状态（pose + 铰接角度）-- 物体不可移动
2. **设置机器人手关节目标** 为运动学 retarget 值
3. **运行一步物理仿真** -- PD 控制器驱动手向目标运动，碰撞检测阻止手指穿透
4. **记录达到的关节值和 keypoint 位置**

用碰撞解析后的值替换原始 retarget 结果。
该过程可在仿真中对所有时间步并行处理。

### 与 ObjDex 方法的对比

ObjDex [35] 使用学习的 high-level wrist planner。DexMachina 发现直接用 retarget + 后处理（不用学习的 planner）效果更好:
- Ketchup-100: >90% (DexMachina reimpl) vs 41.2% (ObjDex original)
- Mixer-170: >70% vs 57.6%

---

## Virtual Object Controllers（核心创新）

### 机制

- 物体在所有 DOF 上获得 PD actuators（6-DOF root + 铰接关节）
- 控制器目标 = 下一时间步的 demo 轨迹
- 初始增益: kp=100~1000, kv=10, force_range=50
- 效果: 物体被虚拟力"引导"沿 demo 轨迹运动

### Curriculum 衰减

基于学习进度自动衰减 kp/kv/force_range:

衰减条件:
- 超过 `wait_epochs`（默认 500）
- 奖励均值高于阈值（task reward > 0.5）
- 奖励梯度稳定（abs < grad_threshold）
- episode 长度接近最大值
- 距上次衰减至少 40 epochs

支持三种调度: `fixed`, `exp`（自动）, `uniform`（per-env 采样）。

**Dialback**: 如果衰减后 episode 长度显著下降，回退到之前的增益。

最终状态: 所有增益 = 0，策略完全通过手-物体接触控制物体。

### 与 ManipTrans curriculum 的对比

ManipTrans 衰减物理参数（重力、摩擦力、误差阈值）。
DexMachina 发现这对长时序铰接任务不够用:
> "ManipTrans policy initially achieves high task reward, but performance drops as
> the curriculum progresses and cannot recover"

---

## 观测空间

每只手（左/右各自）:
- `dof_target_pos`: 目标 - 当前关节位置 [ndof]
- `dof_pos`: 归一化当前关节位置 [ndof]
- `dof_vel`: 关节速度 * 0.1 [ndof]
- `kpt_pos`: 碰撞体 3D 位置 [n_kpts * 3]
- `wrist_pose`: 腕部位置 + 四元数 [7]

物体:
- `parts_pos` + `parts_quat`: 所有 link 位姿 [n_links * 7]
- `dof_pos`: 铰接关节角度 [1]
- `state_diff`: 当前 vs 下一步 demo 目标 [8]
- `root_ang_vel` [3], `root_lin_vel` [3]

附加:
- 接触力 (object-hand link pairs, 默认开启)
- 指尖距离 (optional)
- `episode_length` [1]

全部为 state-based，无图像。非 dict obs，flat tensor。

---

## 奖励结构

Task reward（物体跟踪）:
```
r_task = w_pos * exp(-beta_pos * d_pos) + w_rot * exp(-beta_rot * d_rot) + w_ang * exp(-beta_ang * d_ang)
```
其中 d_pos = 物体位置误差, d_rot = 旋转误差, d_ang = 铰接角度误差。

辅助奖励:
- `r_imi`: keypoint 匹配（手 link 位置 vs 参考）[权重可配置]
- `r_bc`: 关节角度匹配（当前 vs retarget 参考）[默认权重=0，需 `-bc` flag]
- `r_con`: 接触位置匹配（从 MANO-物体 mesh 距离近似）

注意: `r_bc` 命名有误导性 -- 不是 behavior cloning，只是关节级 imitation reward。

---

## 关键设计决策汇总

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| BC pretraining | None | Pure RL from scratch |
| Teacher-student | None | Single policy |
| Wrist control | Residual (tight bounds) | 约束搜索空间，腕部 retarget 质量好 |
| Finger control | Absolute | 手指 retarget 质量差（穿透），关节限位已约束搜索空间 |
| Exploration aid | Virtual object PD controllers | 比 ManipTrans 的物理参数 curriculum 更强 |
| Retarget fix | Object-aware post-processing | 训练前修复穿透，无学习组件 |
| Multi-hand support | 6 embodiments, same code | Abstract DexHand class + per-hand joint mapping |

---

## 与相关工作的对比

| | DexMachina | ManipTrans | DexCanvas | DeXtreme |
|---|---|---|---|---|
| Task | 双手铰接物体跟踪 | 抓取/操作跟踪 | 力数据生成 | Cube rotation |
| Ref motion | Yes (ARCTIC) | Yes (GRAB/OakInk2) | Yes (mocap) | No |
| BC | No | No | No | No |
| Wrist | Residual (tight) | Base_model + residual | Residual (MANO) | N/A (fixed) |
| Fingers | Absolute | Base_model + residual | Residual (MANO) | Absolute |
| Exploration | Virtual object PD | Gravity/friction decay | Privileged future obs | ADR |
| Sim2real | No | No | No | Yes |
