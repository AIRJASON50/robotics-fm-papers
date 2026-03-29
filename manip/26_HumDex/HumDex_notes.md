# HumDex 研究笔记

HumDex: Humanoid Dexterous Manipulation Made Easy (arXiv:2603.12260, 2026)
USC PSI Lab + WorldEngine AI

---

## 1. 核心问题

人形机器人灵巧操作的数据收集瓶颈:
- 光学 mocap (OptiTrack): 高精度但需固定场地，不可移动
- VR 遥操 (AVP/PICO): 便携但手部遮挡严重，工具使用时手指跟踪丢失
- 优化式手部重定向 (dex-retargeting): 慢、需逐手标定、输出抖动

HumDex 提出一套完整的 IMU 遥操 + 学习式重定向 + 两阶段模仿学习的 pipeline，目标是让人形灵巧操作数据采集变得便携、高效、可迁移。

---

## 2. 方法概览

### 系统架构

```
Human Operator
  |
  +-- 全身: 15 个 IMU tracker (VDMocap or SlimeVR <$200)
  |     -> VMC/OSC 协议 -> GMR 全身重定向 (pelvis-centric IK)
  |     -> q_body (29 joint + 6 locomotion = 35D action)
  |
  +-- 双手: 惯性手套 (VDHand or Manus, 21 keypoints/手)
  |     -> MediaPipe 坐标变换 -> 优化式重定向 或 学习式 MLP
  |     -> q_hand (20D/手)
  |
  +-- q_ref = [q_body, q_hand]
        -> 底层控制器: TWIST2/SONIC (body RL tracking) + PD (hand)
        -> 关节力矩
```

### 三个关键模块

**A. IMU 遥操**: 15 个 <20g tracker, >10h 续航, >50m 范围。全身重定向用 pelvis-centric relative rotation/position, 避免 IMU 全局漂移。

**B. 自适应手部重定向**: 核心贡献。
- 优化式: NLopt SLSQP, 50 iterations, 自适应在 TipDirVec (捏合精度) 和 FullHandVec (全手形状) 之间混合
- 学习式: 5 个独立 per-finger MLP (3D fingertip -> 4 joint angles), 用优化结果做监督训练

**C. 两阶段模仿学习**:
- Stage 1: ACT policy 在人类数据上 pretrain (100-700 episodes, 多样化场景)
- Stage 2: 在机器人数据上 fine-tune (50 episodes, 目标场景)
- 解决 human-robot embodiment gap: 人类数据无本体感受状态, 用 state[i]=action[i-1] 近似

---

## 3. 关键设计

### 3.1 自适应重定向的 alpha 混合

```
alpha_i = clip( (d2 - distance_i) / (d2 - d1), 0, 0.7 )
```
- distance_i = 拇指指尖到第 i 指尖的距离 (cm)
- d1=2cm, d2=4cm
- 捏合时 (d<2cm): alpha=0.7, 主要追踪指尖位置+方向 (TipDirVec)
- 张开时 (d>4cm): alpha=0, 主要追踪全手形状 (FullHandVec)

直觉: 捏合时精确的指尖相对位置最重要 (拇食指间距决定抓握), 张开时整体手形更重要。alpha 在两种模式间平滑切换。

### 3.2 Huber Loss 而非 L2/Gaussian

重定向优化使用 Huber loss:
- 小误差 (<delta): 二次惩罚, 类似 MSE
- 大误差 (>delta): 线性惩罚, 防止 outlier 主导优化
- position delta=2.0cm, direction delta=0.5

对比我们的 Gaussian kernel `exp(-(e/sigma)^2)`:
- Gaussian 在大误差时梯度趋零 (饱和/死区问题)
- Huber 在大误差时保持恒定梯度 (鲁棒, 但没有自然的 [0,1] 范围)
- 两者适用场景不同: Huber 适合优化器, Gaussian 适合 RL reward

### 3.3 两阶段顺序训练 (非混合)

关键发现: naive 混合人类+机器人数据直接失败 (0% success), 因为 embodiment gap 让混合数据自相矛盾。

解决方案: 先用人类数据 pretrain (学习 visual affordance + task structure), 再用机器人数据 fine-tune (适应具体 embodiment)。human data 用 `state[i]=action[i-1]` 近似本体感受。

---

## 4. 实验结果

### 遥操性能 (60 attempts each)

| 任务 | Baseline 耗时 | HumDex 耗时 | Baseline 成功率 | HumDex 成功率 |
|------|-------------|------------|----------------|--------------|
| Scan&Pack | 不可行 | 68min | 0% | 90% |
| HangTowel | 68min | 52min | 70% | 88% |
| OpenDoor | 42min | 30min | 90% | 95% |
| PlaceBasket | 62min | 48min | 75% | 97% |
| PickBread | 67min | 47min | 80% | 87% |
| **平均** | **59.8min** | **44.3min** | **74.6%** | **91.7%** |

### 策略泛化 (PickBread, 30 trials)

| 设置 | RobotOnly | 两阶段 (Ours) |
|------|-----------|--------------|
| 见过的场景 | 100% | 93% |
| 未见位置 | 57% | 87% |
| 未见物体 | 50% | 87% |
| 未见背景 | 30% | 73% |

关键: 人类数据不直接帮 in-distribution, 但大幅提升 OOD 泛化 (+30-40%)。

### 学习式 vs 优化式重定向

学习式 MLP (Ours) 在精细操作子任务上优于优化式 retargeting:
- Scanner triggering: 7/10 vs 3/10
- Hanger stabilization: 9/10 vs 7/10
- Doll grasping: 9/10 vs 5/10

---

## 5. 相关工作分析

HumDex 在 humanoid manipulation 领域的位置:

| 维度 | TWIST2/SONIC | ManipTrans | DexCanvas | HumDex |
|------|-------------|-----------|-----------|--------|
| 全身控制 | RL motion tracking | 无 | 无 | 底层用 TWIST2/SONIC |
| 手部控制 | PD joint | 优化重定向 | RL tracking | 学习式 MLP + PD |
| 数据来源 | 人类穿戴 | 人手视频 | DexCanvas 手套 | IMU + 惯性手套 |
| 训练方式 | RL | RL + demo | RL | 模仿学习 (ACT) |
| 独特性 | 便携遥操 | 视频到策略 | 精细跟踪 | 全 pipeline + 泛化 |

与 bh_motion_track 的区别:
- bh_motion_track 用 RL + Gaussian kernel tracking, HumDex 用 imitation learning (ACT)
- bh_motion_track 在 MJX 仿真中训练, HumDex 在真机上 teleop + 学习
- HumDex 的手部重定向是独立模块, bh_motion_track 的 tracking 是 reward signal 的一部分

---

## 6. 局限性与未来方向

**作者提到**:
- 惯性手套仍有 drift, 长时间操作需要偶尔校准
- 两阶段训练需要分别收集人类和机器人数据, 增加工作量
- ACT 策略对 unseen 场景泛化仍有限 (73% 在 unseen background)

**从代码推断**:
- per-finger segment scaling 需要针对每种手套手动标定
- alpha 混合的 d1/d2 阈值是固定的, 不同任务可能需要调整
- 底层控制器 (TWIST2) 是训练好的 RL policy, 全身性能受限于这个 policy 的能力
- 没有力反馈, 操作者纯靠视觉判断接触

---

## 7. 论文 vs 代码差异

| 方面 | 论文描述 | 代码实现 |
|------|---------|---------|
| 手部重定向 | "optimization-based retargeting" 作为 baseline, 学习式 MLP 作为贡献 | 优化式是默认选项, 包含复杂的 adaptive analytical optimizer (50 iter SLSQP, Huber loss, per-finger alpha blend, per-segment scaling) |
| MLP 训练 | "lightweight MLP regressor" | 两步训练: 先训 FK model (random sample), 再训 IK model (supervised on teleop data); per-finger 独立子网络 |
| 损失函数 | 未提及 Chamfer loss | `loss.py` 包含 Chamfer distance 实现, 暗示探索过无监督/几何训练 |
| 归一化策略 | 未提及 | sequential_unified_stats: 跨 human+robot 数据集统一计算归一化统计量 |
| Human data 处理 | "previous action as state" | 具体实现: state_body 从 action 中提取 roll/pitch + 29 joints (去掉 locomotion 指令), 首尾帧裁剪 |
| 多控制器后端 | 仅提 TWIST2 | 同时支持 TWIST2 和 SONIC, 通过 ZMQ 切换 |
| SlimeVR 支持 | 简要提及 | 完整 VMC 协议解析, BVH FK 重建, local/global rotation mode |
| 数据鲁棒性 | 未提及 | HDF5 读取有 retry 逻辑 (20 次), 处理 corrupted chunk |

---

## 8. 跨论文比较

### 与 bh_motion_track 任务的比较

| 维度 | HumDex | bh_motion_track |
|------|--------|-----------------|
| **训练范式** | 模仿学习 (ACT, behavior cloning) | 强化学习 (PPO) |
| **手部跟踪** | 指尖位置 (15D per hand) | 多层: wrist pos/ori + anchor-relative tips + BC joints |
| **误差映射** | Huber loss (优化器内) | Gaussian kernel exp(-(e/sigma)^2) |
| **梯度特性** | Huber: 小误差二次, 大误差线性 | Gaussian: 小误差弱梯度 (饱和), 大误差零梯度 (死区) |
| **接触处理** | 无显式接触 reward, 完全靠 demo 学习 | 3-term contact reward (touch + match - FP), contact gate |
| **Object 跟踪** | 无显式 object tracking, 隐式通过 demo 学习 | Gaussian kernel 乘法组 (pos x ori x face) |
| **Bootstrap 问题** | 无 -- demo 直接提供正确行为 | 严重 -- contact reward 前 50M 步为零, 稀疏奖励 |
| **Embodiment gap** | 两阶段解决 (human pretrain + robot finetune) | 不适用 (仿真训练) |
| **部署** | 真机 (G1 humanoid) | MJX 仿真, 待 sim2real |

### 对 bh_motion_track 的启发

1. **Huber loss 替代 Gaussian kernel**: HumDex 的 Huber loss 在大误差时保持线性梯度, 正好解决我们观察到的 Gaussian 死区问题。但 Huber 输出不在 [0,1] 范围, 用于 RL reward 需要额外归一化 (类似我们试过的 L1 但效果不好, 可能因为缺少 Huber 的小误差二次特性)。

2. **Adaptive alpha 混合**: 根据接触状态动态调整跟踪目标的权重, 概念上类似我们的 contact gate, 但 HumDex 是连续的基于距离的 alpha, 而我们是二元的 contact sensor gate。可以考虑: 在 bh_motion_track 中根据手指到物体距离动态调整 tips tracking 的 sigma 或 weight。

3. **Per-finger segment scaling**: 人手-机器手的 embodiment gap 通过 per-finger per-segment 缩放因子解决。类似的思路可以用于 MotionGen trajectory retargeting 的质量提升。

4. **两阶段训练的核心洞察**: 人类数据不直接提升 in-distribution 性能, 但显著改善 OOD 泛化。这暗示: 对于 bh_motion_track, 多轨迹训练 (更多 MotionGen trajectories) 可能不会直接降低单轨迹 tracking error, 但会显著提升策略在新轨迹上的泛化能力。
