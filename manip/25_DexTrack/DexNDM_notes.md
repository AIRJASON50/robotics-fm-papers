# DexNDM 笔记

DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model
Xueyi Liu, He Wang, Li Yi (Tsinghua / Peking / Qi Zhi / Galbot), 2025.10
Project: meowuu7.github.io/DexNDM/

## 1. Core Problem

解决灵巧手 in-hand rotation 的 **sim-to-real gap**。现有方法受限于：
- 简单几何体、固定腕部朝向、有限旋转轴
- 学 whole-body dynamics model 需要大量 distributionally relevant 真实数据，但灵巧操作中采集这类数据极其困难（物体掉落需人工 reset、遮挡导致 tracking 不准、需要视觉系统等）

核心矛盾：模型泛化需要大量多样数据，但灵巧操作的数据采集本身就很难规模化。

## 2. Method Overview

### 2.1 整体 Pipeline (Fig. 3)

五阶段流水线：

1. **Oracle Policy Training** (Sec 3.1): 在 Isaac Gym 中按物体类别训 RL oracle policy（PPO），每类一个 specialist，使用 privileged observation
2. **Generalist Policy via BC** (Sec 3.2): 聚合所有 oracle 的成功轨迹，用 Behavior Cloning 蒸馏成一个 generalist policy
3. **Autonomous Data Collection** ("Chaos Box"): 真实世界自主采集 interaction data
4. **Joint-Wise Neural Dynamics Model** (Sec 3.3): 用采集的真实数据训练 joint-wise dynamics model
5. **Residual Policy Training** (Sec 3.3): 训练 residual policy 补偿 sim-real gap

### 2.2 Specialist-to-Generalist

**Oracle (per-category)**:
- 观测 (oracle): joint pos/vel history (3-step, 48+48D), joint target history (48D), fingertip state+vel (52D), object state+vel (13D), object goal pose (4D), contact force (40D), binary contact (92D), wrist orientation (4D), rotation axis (3D)
- 动作: delta joint position, alpha=1/24 exponential smoothing: `a_t = a_{t-1} + alpha * delta_a_t`
- 奖励: `r = alpha_rot * r_rot + alpha_goal * r_goal + alpha_penalty * r_penalty`
  - r_rot: `clip(omega . k, -c, c)`, 鼓励绕目标轴旋转
  - r_penalty: 抑制离轴角速度、object linear velocity、偏离标准手型、关节力矩
  - r_goal: 中间目标引导（每90度设一个 waypoint），解决长物体等困难情况
- alpha_rotp 线性 schedule: 0 -> 0 (前10 reset) -> 0.1 (10~100 reset) -> 0.1 (thereafter)

**Generalist (BC)**:
- 观测 (deployable): proprioception history (T=10), wrist orientation, rotation axis — 不用 object state
- 架构: Residual MLP, 5 residual blocks, hidden dim 1024
  - `y = ReLU(NN1(x) + NN3(ReLU(NN2(x))))`
- BC 只用 oracle 成功轨迹，DAGger 在此场景下不 work（task difficulty 太高）

### 2.3 Joint-Wise Neural Dynamics Model (核心创新)

**动机**：传统 whole-hand dynamics model `q^{t+1} = f_theta(H_t)` 从整手 state-action history 预测下一步。数据需求大、泛化差。

**Joint-wise 分解**：每个关节 i 只从自己的 W-step state-action history 预测自己的下一步状态：
```
q_i^{t+1} = f_{psi_i}(h_i^t),  h_i^t = {q_j^i, a_j^i}_{j=t-W+1}^{t}
```

**理论支撑** (Information Contraction):
- 投影 g: (H_t, q_i^{t+1}) -> (h_i^t, q_i^{t+1}) 是 non-injective 的降维
- Data Processing Inequality (Theorem 3.1): `KL(g(P)||g(Q)) < KL(P||Q)`，投影后分布偏移更小
- Generalization Gap Contraction (Theorem 3.2): 分布偏移越小 -> 泛化 gap 越小
- 直观理解：joint-wise model 天然过滤了高维 nuisance 信息（其他关节的干扰），用 history 隐式捕获 inter-joint coupling 和 object 影响

**物理直觉** (Appendix A.3)：
- 标准操作臂动力学: `M(q)q'' + C(q,q')q' + G(q) = tau + tau_ext`
- 单关节视角: `H_i^{eff} * q_i'' + G_i^{eff} = tau_i`，其中 H_i^{eff} 和 G_i^{eff} 压缩了所有 inter-joint 和 external 影响
- 关键假设: 短时间窗口(10 frames, 0.5s)内，per-joint state/action/external torque 可被低阶多项式逼近 -> neural net 从 history 学出这些 effective terms

**实现细节**:
- W = 10 (history window)
- 用 simulation data 预训练，再用 real-world data finetune
- 训练 loss: `L_{dyn} = ||q_{t+1} - q_hat_{t+1}||_2`

### 2.4 Autonomous Data Collection ("Chaos Box")

解决 "data 从哪来" 的问题：
- 把机械手放入一个**装满软球的容器**中
- 开环 replay sim policy 的 action（来自 base policy rollout）
- 50% 概率加 Gaussian noise (sigma=0.01) 到 action 上
- 手与球的持续交互产生丰富的 contact-loaded transition
- 完全自主、无需人工干预、无需 reset、无需视觉系统

**为什么 work**: joint-wise model 只需要 per-joint state-action transition，不需要 object state -> 训练数据不要求 task-relevant，只要有 diverse interaction dynamics 就够

**覆盖度局限**: Chaos Box 无法穷尽 manipulation 中的所有 dynamics 情况。软球交互的力 profile (持续法向力) 与实际操作 (指尖滑动、多指协调挤压、物体边角卡住) 差异显著。在 manipulation 中，外力可能是关节运动的主导项而非 perturbation，此时 history 不一定能隐式捕获这些 unseen 力。

**为什么 "够用"**: 不是因为 Chaos Box 覆盖了 manipulation dynamics，而是三者组合兜底:
1. Sim pretrain 已学了 dynamics 的结构性部分，real data finetune 只修正 residual gap (摩擦系数偏差、腱绳弹性等)，这部分不太依赖 task-specific 负载
2. In-hand rotation 任务的 contact force 不算极端 (无大力抓取或高速冲击)
3. Residual policy 可在线补偿 NDM 的预测误差

**对比其他采集方式** (Fig. 9):
- Task-aware (with vision): ~200s per trajectory, 需要人工干预, 对遮挡/对称物体 fail
- Task-aware (no vision): 仍需人工, 数据覆盖受限于 policy 能力
- Free-hand (no load): 数据多样性低, 无 object interaction
- Chaos Box: ~3min per trajectory, fully automated, broad coverage

### 2.5 Residual Policy

在 learned dynamics model 上训练 residual policy `pi^{res}`：
- 输入: generalist policy 的观测 o_t^{gene} + base action a_t
- 输出: residual correction a_t^{res}
- 执行时: a_t + a_t^{res}
- 训练目标: 让 dynamics model 预测的下一步尽量匹配 simulator 的下一步
  `min ||q_{t+1} - f_psi({q_j, a_j + pi^{res}(o_j^{gene}, a_j)})||`
- 用 supervised learning 在 oracle 生成的轨迹数据集上训练

**NDM 在训练中的角色**: NDM 参数 freeze，仅作为可微的梯度桥梁。梯度路径为:
```
Loss = ||q_{t+1}^{oracle} - f_NDM(s_t, a_t + da)||
grad: dL/d(da) = dL/dq_hat × df_NDM/da    <- NDM 的 Jacobian 提供梯度方向
                                               NDM 不更新，只传梯度给 residual policy
```
NDM 质量的关键要求: 对 action 的响应方向和趋势必须正确 (Jacobian 方向对)，绝对精度可以有误差。

**Label 来源**: q_{t+1}^{oracle} 来自 step 1 的 oracle 在 sim 中的 rollout 轨迹。这里 "oracle" 是 "sim 中的最优策略"，不是 "real 上的正确答案"。Residual 学的是: "在 real dynamics 下，action 需要怎么改才能到达 oracle 在 sim 中达到的 state"。

**部署形态**: 两个网络同时运行，叠加输出:
```
a_final = pi^{gene}(o_t) + pi^{res}(o_t, a_t)
           generalist        residual correction
           (task-level)      (dynamics compensation)
```

**性能上界**: 上界就是 sim oracle 的水平。Residual 的优化目标是复现 sim 轨迹，不可能超越 sim。论文证明的是 "DexNDM 比 direct transfer 恢复更多 sim 性能"，不是 "超越 sim"。要突破此上界，需在 NDM 上用 RL reward 而非 supervised label 训练，但论文未这样做 (可能因为 NDM 精度不支撑 RL 长 horizon rollout)。

## 3. Key Designs

### 3.1 Joint-Wise vs Whole-Hand Dynamics (核心区别)

| | Whole-Hand | Joint-Wise |
|---|---|---|
| 输入维度 | ~320D (16 joints * W * 2) | ~20D (1 joint * W * 2) |
| 输出 | 16D (all joints) | 1D (single joint) |
| 数据需求 | 大，需要 in-distribution | 小，可用 task-agnostic data |
| 泛化 | 受限于训练分布 | 跨分布泛化好 (理论保证) |
| 表达力 | 高（能建模 inter-joint coupling）| 略低（隐式通过 history 捕获）|

Ablation (Fig. 6): whole-hand model 在数据充足时(3.1M)和 joint-wise 性能相当，但数据不足时(7.5k) joint-wise 大幅领先。在 OOD 场景下 joint-wise 一致更好。

### 3.2 Specialist-to-Generalist 而非 End-to-End

为什么不直接训一个跨物体的 RL policy 然后 transfer?
- Task difficulty 太高: in-hand rotation 对复杂形状 (高 aspect ratio、animal shapes) 的 RL 训练本身就很难
- DAGger 在此场景 fail: distilled policy 在 sim 中 diverge 或在 real 中 collapse (echoing PenSpin 的发现)
- BC from successful rollouts only: 更稳定

### 3.3 Data Collection 的 5 个 Design Choices (Ablation Fig. 8)

1. **Joint-wise > Finger-wise > Whole-hand**: 最细粒度的分解最好
2. **Sim pretraining**: 先在 sim data 上预训练 dynamics model，再用 real data finetune
3. **加 noise 到 replay action**: 增加 data diversity
4. **有 load (Chaos Box) > 无 load (free-hand)**: object 交互数据至关重要
5. **Replay policy action > Base wave action** (Fey et al., 2025): policy-relevant trajectory 更有效

## 4. 完整 I/O 规格

### 4.1 Oracle Policy 观测 (~352D)

| Component | Dim | Note |
|-----------|-----|------|
| Joint pos history (3-step) | 48 | 16 joints * 3 |
| Joint positional target history (3-step) | 48 | |
| Joint velocity | 16 | |
| Fingertip state + velocity | 52 | 4 tips * 13D |
| Object state + velocity | 13 | pos(3)+quat(4)+linvel(3)+angvel(3) |
| Object guiding goal pose | 4 | quaternion |
| Joint + rigid body forces | 40 | privileged |
| Binary contact | 92 | privileged |
| Wrist orientation | 4 | quaternion |
| Rotation axis | 3 | |

### 4.2 Generalist Policy 观测

| Component | Dim | Note |
|-----------|-----|------|
| Proprioception history (T=10) | 10 * (q + a_prev) | joint pos + prev action |
| Wrist orientation | 4 | quaternion |
| Rotation axis | 3 | |

### 4.3 动作 (16D)

- Delta joint position, alpha=1/24 smoothing
- PD control at 20Hz, torque control substep 6x
- `tau = K_p * (q_tar - q) - K_d * q_dot`

### 4.4 Joint-Wise Dynamics Model

- 输入: per-joint (q_i, a_i) history, W=10 steps -> 20D per joint
- 输出: q_i^{t+1} (1D)
- 16 个独立的小网络 (或共享参数 + joint index)

## 5. Experiments

### 5.1 Simulation (Isaac Gym)

| Setting | DexNDM improvement over baseline |
|---------|------|
| Generalization (unseen objects, Table 1) | 37-81% over Direct Transfer |
| Multi-wrist orientation (Table 5) | Consistently better |
| Sim-to-Sim transfer (Isaac Gym -> Genesis/MuJoCo, Table 6) | Outperforms UAN, ASAP |

5 object categories, ContactDB test set for generalization evaluation.

### 5.2 Real World (LEAP Hand)

**Hardware**: LEAP hand (Shaw et al., 2023) — 16 DOF, PD control at 20Hz

**Real-world capabilities (Table 4)**:

| Object Set | DexNDM Rot (rad) | Best Baseline Rot (rad) |
|---|---|---|
| Regular, +x | 11.36 | 9.84 (Direct Transfer) |
| Small, +x | 5.24 | 4.71 (Direct Transfer) |
| Irregular, +x | 6.35 | 5.51 (Whole-Hand NDM) |

**vs AnyRotate** (Table 2): DexNDM 在所有 AnyRotate 的 test objects 上大幅领先，且能处理 AnyRotate 无法处理的 challenging objects (small, high-aspect-ratio, complex shapes)

**vs Visual Dexterity** (Table 3): 在 VD 展示的 replicable objects 上 comparable or better (survival angle metric)，且 DexNDM 能额外处理 small objects + diverse wrist orientations

**Multi-wrist orientation** (Table 5): 6 种腕部朝向 (palm down/up, base up/down, thumb up/down)，DexNDM 均可工作

**Applications** (Fig. 7): teleoperation system (Meta Quest 3) 执行 tool-using (hammer, brush, pen) 和 furniture assembly

### 5.3 Key Ablations

**Dynamics model design** (Fig. 8):
- Joint-wise > finger-wise > whole-hand
- Sim pretraining essential
- Action noise helps
- Object load (Chaos Box) essential
- Policy replay > wave action

**Data collection strategy** (Fig. 9):
- Chaos Box: 3min/trajectory, fully autonomous, best performance
- Task-aware: 200s/trajectory, needs human, fails on many objects
- Joint-wise model most robust to training distribution shift

**Scaling** (Fig. 9C): performance improves with more real data; power-law scaling observed

## 6. Limitations

- 模型上限受限于 partial observation: joint-wise model 无法显式建模 hand-object interaction
- 触觉信号未利用: 作者指出 integrating tactile 是重要 future direction
- Chaos Box 数据质量: 软球交互与实际 object 操作的 dynamics 仍有差异，可能限制上限
- Residual policy 依赖 base generalist 的质量: 如果 base policy 在某个 regime 完全不 work, residual 也难以救回
- 只支持 in-hand rotation: 不涉及 grasp, reorientation to specific pose, tool use planning 等更复杂任务
- 数据采集仍需物理机器人和一定时间 (4000 trajectories 的 Chaos Box)

## 7. DexNDM vs ASAP: Residual 机制对比

表面看都是 "residual 补偿"，但参数化方式不同:

| | DexNDM | ASAP |
|---|---|---|
| 补偿什么 | sim->real dynamics gap | sim->real action gap |
| 怎么学 | 在 learned NDM 上做 supervised learning | 在 sim 中用 RL 对抗 delta model |
| 是否有 dynamics model | 有 (joint-wise NDM，核心贡献) | 无 (model-free) |
| 真实数据用途 | 训练 NDM | 训练 delta action model |
| Residual 语义 | "real dynamics 下需要什么 action 修正" | "sim 和 real 的 action 映射差异" |

**功能等价视角**: 两者都在逼近真实的状态转移函数:
```
ASAP:    f_real(s,a) ~ f_sim(s, a + delta(s,a))    # sim dynamics + action correction
DexNDM:  f_real(s,a) ~ f_NDM(s,a)                   # learned dynamics 直接替代 sim
```

**核心 trade-off**: 你是否信任 sim 的结构？
- 信 sim 结构 -> ASAP: 只修 action 偏差，数据效率高，但 sim 结构本身错时 fail
- 不信 sim -> DexNDM: 从头学 dynamics，需要更多数据，但不受 sim 结构限制

对灵巧手场景 (腱绳传动、摩擦、间隙难以在 sim 中准确建模)，DexNDM 选择 "不信 sim" 是合理的。

## 8. Cross-Paper Comparison

| | DexNDM | DexTrack | ASAP | UAN |
|---|---|---|---|---|
| 任务 | In-hand rotation | Tracking (grasp+manip) | Humanoid whole-body | Humanoid locomotion |
| Sim-to-Real 策略 | Joint-wise dynamics + residual policy | 无 (pure sim) | Delta action model | Finetuning |
| 动力学建模 | Joint-wise neural dynamics | 无 | 无 (model-free) | 无 (model-free) |
| 数据采集 | Chaos Box (autonomous) | N/A | Policy rollout | Policy rollout + human |
| 物体泛化 | 多物体 (BC distillation) | 多物体 (PointNet 256D) | N/A | N/A |
| Action Space | Delta pos + smoothing (alpha=1/24) | Double-integration residual | Sim-real delta | Residual |
| 手 | LEAP 16 DOF | Allegro 22 DOF | Humanoid | Humanoid |
| Real 硬件 | LEAP hand | LEAP+Franka | Humanoid | Humanoid |

### 8.1 DexNDM vs DexTrack (同一作者 Xueyi Liu)

两篇论文解决不同层面的问题:
- **DexTrack**: sim-only, 解决 "如何训一个 generalist tracking controller"，核心是 data flywheel (homotopy bootstrapping + RL/IL)
- **DexNDM**: sim-to-real, 解决 "sim policy 如何 transfer 到 real"，核心是 joint-wise dynamics model + autonomous data collection

互补关系: DexTrack 的 generalist controller 可以作为 DexNDM 的 base policy (目前 DexNDM 用的是 rotation-specific oracle + BC distillation)

### 8.2 与 bh_motion_track 可借鉴点

1. **Joint-wise dynamics model**: 如果未来需要 sim-to-real, 可以考虑 per-joint dynamics 而非 whole-hand, 大幅降低真实数据需求
2. **Chaos Box 数据采集**: 完全自主、无需视觉、无需人工 reset — 如果有硬件条件，这是最 scalable 的数据采集方案
3. **Residual policy on top of base**: 与 DexTrack 的 double-integration residual 不同，DexNDM 的 residual 是在 dynamics model 上训练的，本质上学的是 "sim 和 real 的 dynamics 差异" 而非 "tracking error 的修正"
4. **Information bottleneck 思路**: per-joint decomposition 作为 information bottleneck, 丢弃 nuisance variance, 类似于 DexTrack 的 PointNet latent (256D) 丢弃物体几何细节。可以推广到其他 low-data regime
5. **BC > DAGger for hard tasks**: 在 task difficulty 很高时，DAGger 的 on-policy correction 可能 diverge, 纯 BC (from successful rollouts) 反而更稳定
