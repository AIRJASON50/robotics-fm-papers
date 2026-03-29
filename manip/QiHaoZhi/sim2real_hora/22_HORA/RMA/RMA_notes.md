# RMA - 论文笔记

**论文**: RMA: Rapid Motor Adaptation for Legged Robots
**作者**: Ashish Kumar (UC Berkeley), Zipeng Fu (CMU), Deepak Pathak (CMU), Jitendra Malik (UC Berkeley, Facebook)
**发表**: RSS 2021; arXiv:2107.04034
**项目**: https://ashish-kmr.github.io/rma-legged-robots/
**代码**: https://github.com/antonilo/rl_locomotion.git (CMS 扩展版，包含 RMA 训练流程)

> 此论文是 HORA 的直接方法论来源。Ashish Kumar 同时是 RMA 和 HORA 的共同一作。

---

## 一句话总结

提出两阶段框架（privileged teacher → proprioceptive adaptation module），使四足机器人能在部署时通过 0.5 秒的本体感受历史实时推断环境参数，实现对未见地形/载荷/摩擦的快速自适应，无需任何真实世界微调。

---

## 核心问题

四足机器人在真实世界需要适应不断变化的地形（岩石、泥地、草地）、载荷、磨损等。现有方法的困境：

| 方法 | 问题 |
|------|------|
| Domain Randomization | 以鲁棒性换最优性，学到过于保守的策略 |
| System Identification | 需要真实世界数据采集 (4-8 min rollouts [40])，且精确估计物理参数既困难又不必要 |
| Meta-learning | 仍需多次真实世界 rollout 来适应 |
| 直接真实世界 RL | 采样效率低、不安全、难以规模化 |

RMA 的核心问题：**如何在不收集任何真实世界数据的前提下，实现亚秒级的在线环境自适应？**

---

## 方法概述

### Phase 1: Base Policy + Environmental Factor Encoder (仿真 RL)

联合训练策略 $\pi$ 和环境编码器 $\mu$：

$$z_t = \mu(e_t), \quad a_t = \pi(x_t, a_{t-1}, z_t)$$

- **环境参数** $e_t \in \mathbb{R}^{17}$: 质量 + 质心位置(3D) + 摩擦系数 + 电机强度(12D) + 局部地形高度(1D)
- **Extrinsics** $z_t \in \mathbb{R}^{8}$: $\mu$ 将 17D 压缩为 8D 低维嵌入
- **状态** $x_t \in \mathbb{R}^{30}$: 关节位置(12) + 关节速度(12) + IMU roll/pitch(2) + 足部接触二值(4)
- **动作** $a_t \in \mathbb{R}^{12}$: 12 个关节的 PD 目标位置偏移
- **算法**: PPO, 联合优化 $\pi$ 和 $\mu$

### Phase 2: Adaptation Module (仿真监督学习)

训练 $\phi$ 从历史 proprioception 预测 extrinsics：

$$\hat{z}_t = \phi(x_{t-k:t-1}, a_{t-k:t-1}), \quad k = 50 \text{ (0.5s)}$$

- **冻结** Phase 1 的 $\pi$ 和 $\mu$
- **On-policy 数据收集**: 用随机初始化的 $\phi$ 预测 $\hat{z}_t$ 驱动 $\pi$ 做 rollout，同时记录 ground-truth $z_t$
- **损失**: $\text{MSE}(\hat{z}_t, z_t)$
- **关键**: 用 on-policy（非 expert）rollout 保证 $\phi$ 对 imperfect prediction 的鲁棒性（类似 DAgger）

### 部署: 异步双频率

- Base policy $\pi$: **100 Hz** (快速响应)
- Adaptation module $\phi$: **10 Hz** (计算量大，低频更新)
- 两者异步运行，$\pi$ 使用 $\phi$ 最近一次输出的 $\hat{z}_t$
- 无需中央时钟同步

---

## 关键设计

### 1. Extrinsics 而非精确物理参数

这是 RMA 最核心的设计决策。论文给出三个理由：

1. **可辨识性 (Identifiability)**: 某些物理参数可能协变（covariate）但对行为影响相同，extrinsics 自动处理这种冗余
2. **不需要"正确"**: extrinsics 不需要在物理意义上精确，只需要导致"正确的动作"。端到端训练自动优化这一点
3. **低维**: 8D 比 17D 环境参数更容易从短历史中估计

对比 SysID 基线：RMA 73.5% success vs SysID 56.5% success。显式预测精确物理参数既困难又不必要。

### 2. 仿生奖励 (Bioenergetics-Inspired Rewards)

论文明确列出 10 项奖励，权重为 [20, 21, 0.002, 0.02, 0.001, 0.07, 0.002, 1.5, 2.0, 0.8]:

| 编号 | 奖励项 | 公式 | 设计理由 |
|------|-------|------|---------|
| 1 | 前进速度 | $\min(v_x^t, 0.35)$ | 鼓励前进但不超速 |
| 2 | 侧向+旋转 | $-\|v_y\|^2 - \|\omega_{yaw}\|^2$ | 保持直线行走 |
| 3 | 机械功 | $-|\tau^T \cdot (\mathbf{q}^t - \mathbf{q}^{t-1})|$ | 仿生：最小化能量消耗 |
| 4 | 地面冲击 | $-\|\mathbf{f}^t - \mathbf{f}^{t-1}\|^2$ | 仿生：减少地面反作用力突变 |
| 5 | 平滑性 | $-\|\tau^t - \tau^{t-1}\|^2$ | 减少力矩跳变 |
| 6 | 动作幅度 | $-\|a^t\|^2$ | 限制关节偏移量 |
| 7 | 关节速度 | $-\|\dot{q}\|^2$ | 限制关节速度 |
| 8 | 朝向 | $-\|\theta_{roll,pitch}\|^2$ | 保持身体水平 |
| 9 | Z 加速度 | $-\|v_z\|^2$ | 减少上下弹跳 |
| 10 | 足部滑移 | $-\|\text{diag}(g^t) \cdot v_f^t\|^2$ | 着地脚不应滑动 |

**关键洞察**: 不使用人工参考轨迹或预定义步态生成器。自然步态完全从能量约束中涌现。这与 HORA 的手指步态涌现设计理念完全一致。

### 3. 训练课程 (Training Curriculum)

为避免策略因惩罚过大而学会"站着不动"：
- 初始训练时惩罚项系数很小
- 逐步增大惩罚系数（固定 curriculum）
- 同时线性增大域随机化的扰动范围（质量、摩擦、电机强度）
- 地形难度不做 curriculum，从训练开始就固定

### 4. On-policy 数据收集 for $\phi$

为什么不用 expert rollout 训练 $\phi$：
- Expert rollout 只有"好"轨迹，$\phi$ 在这些数据上训练后对部署时的偏差不鲁棒
- On-policy 用**随机初始化的 $\phi$** 做 rollout，轨迹天然包含偏差和失败
- 迭代训练直到收敛，类似 DAgger [Ross et al., 2011]

---

## 实验

### 平台

- **机器人**: Unitree A1 (18 DoF, ~12kg)
- **仿真**: RaiSim (CPU)
- **训练**: Phase 1 约 24h/1.2B 步; Phase 2 约 3h/80M 步 (单 GPU 桌面机)

### 仿真结果 (Table II)

| 方法 | Success (%) | TTF | Reward | Distance (m) | Torque | Smoothness |
|------|------------|-----|--------|-------------|--------|------------|
| Robust (DR) | 62.4 | 0.80 | 4.62 | 1.13 | 527.59 | 122.50 |
| SysID | 56.5 | 0.74 | 4.82 | 1.17 | 565.85 | 149.75 |
| AWR | 41.7 | 0.65 | 4.17 | 0.95 | 599.71 | 162.60 |
| RMA w/o Adapt | 52.1 | 0.75 | 4.72 | 1.15 | 524.18 | 106.25 |
| **RMA** | **73.5** | **0.85** | **5.22** | **1.34** | **500.00** | **92.85** |
| Expert (上界) | 76.2 | 0.86 | 5.23 | 1.35 | 485.07 | 85.56 |

RMA 接近 Expert 上界，远超所有基线。AWR（需要 40k 真实世界样本）反而最差，因为环境持续变化导致离线优化的 $\hat{z}$ 过时。

### 真实世界室内结果 (Figure 3)

| 场景 | RMA | RMA w/o Adapt | A1 原厂控制器 |
|------|-----|---------------|------------|
| 不平坦泡沫 | 80% | 0% | 20% |
| 上坡 | 100% | 20% | 100% |
| 记忆棉床垫 | 100% | 0% | 100% |
| 下台阶 15cm | 100% | 0% | 60% |
| 上台阶 6cm | 100% | 40% | 80% |
| 上台阶 8cm | 60% | 0% | 20% |
| 载荷 12kg (=体重) | 高成功 | 8kg 后失败 | 5kg 后开始退化 |

### 真实世界户外

沙地、泥地、长草、碎石、徒步径 -- 全部成功，无单次失败。下山台阶 70% 成功率（训练中从未见过楼梯）。

### 适应分析 (Figure 4)

当机器人走上涂油塑料片时：
- extrinsics $\hat{z}$ 的第 1、5 维在约 2 秒内响应变化
- 步态从常规 → 适应期（不稳定）→ 恢复（更高力矩、相似步态周期）
- $\hat{z}$ 不会恢复原值 -- 持续反映"地面仍然很滑"的事实

---

## RMA 的贡献与创新（对照原文）

### 贡献 1: 两阶段训练框架的提出

**原文 (Section III)**: "The truly novel contribution of this paper is the adaptation module, trained in simulation, which makes RMA possible."

RMA 的框架设计解决了一个根本矛盾：
- 训练时需要环境参数来学好策略
- 部署时环境参数不可获取

之前的方法要么完全忽略环境参数（DR，过于保守），要么需要真实世界数据来估计参数（SysID/AWR）。RMA 的创新在于**将 system identification 转化为监督学习问题**，在仿真中完成，部署时零样本。

### 贡献 2: Extrinsics 而非精确物理参数

**原文 (Section I)**: "Note that instead of predicting $e_t$, which is the case in typical system identification, we directly estimate the extrinsics $z_t$ that only encodes how the behavior should change."

这不仅是工程简化，而是一个**认识论转变**：
- 传统 SysID 追求"物理世界是什么样的" → 太难且不必要
- RMA 问的是"我的行为应该怎么变" → 更容易学且端到端优化

Table II 中 SysID 56.5% vs RMA 73.5% 量化地验证了这一点。

### 贡献 3: 仿生奖励替代参考轨迹

**原文 (Section III-A)**: "The reward function is motivated from bioenergetic constraints of minimizing work and ground impact. We found these reward functions to be critical for learning realistic gaits in simulation."

之前的 sim-to-real locomotion [22, 31] 依赖：
- 预定义的步态轨迹生成器 [24]
- 人工设计的足部运动模板 [22]

RMA 完全不用这些，步态从物理约束自然涌现。这个设计被 HORA 直接继承——HORA 同样不强制指定手指步态。

### 贡献 4: 异步双频率部署

**原文 (Section III-C)**: "This asynchronous design was critical for seamless deployment of RMA on low-cost robots like A1 with limited on-board compute."

$\pi$ 在 100Hz 运行保证快速响应，$\phi$ 在 10Hz 运行降低计算需求。两者异步无需同步时钟。这是一个实用但重要的工程创新，使 RMA 能在 A1 的有限算力上部署。

HORA 继承了这一设计但频率不同：策略 20Hz，adaptation 也是 20Hz（HORA 的 adaptation module 更小，算力压力较低）。

### 贡献 5: 大规模真实世界验证

RMA 在论文中展示了迄今（2021 年）最广泛的四足机器人真实世界测试：岩石、泥地、沙地、草地、碎石、台阶、床垫、泡沫、油面 -- 全部用同一个策略，零微调。这个验证力度建立了该框架的可信度。

---

## RMA → HORA 的继承关系

| 设计元素 | RMA (locomotion) | HORA (manipulation) | 变化 |
|---------|-----------------|--------------------|----|
| **环境参数 $e_t$** | 17D (质量/COM/摩擦/电机/地形) | 9D (位置/尺寸/质量/摩擦/COM) | 维度和内容不同 |
| **Extrinsics $z_t$** | 8D | 8D | 完全相同 |
| **状态 $x_t$** | 30D (关节+IMU+足部接触) | 96D (3帧 x [joint_pos+target]) | HORA 用多帧历史替代速度/IMU |
| **Adaptation input** | 50 帧 x (state+action) | 30 帧 x (joint_pos+target) | 不同窗口长度 |
| **Adaptation arch** | 2层 MLP + 3层 Conv1d → 8D | 2层 MLP + 3层 Conv1d → 8D | 几乎相同 |
| **Env encoder $\mu$** | 3层 MLP (17→256→128→8) | 3层 MLP (9→256→128→8) | 结构相同 |
| **奖励设计** | 仿生约束 (10 项) | 旋转+正则化 (5 项) | 理念相同: 自然约束 → 涌现行为 |
| **部署频率** | $\pi$=100Hz, $\phi$=10Hz (异步) | $\pi$=$\phi$=20Hz (同步) | HORA 简化 |
| **Curriculum** | 惩罚权重 + DR 范围渐进增大 | 无 curriculum (HORA 固定) | HORA 更简单 |
| **训练规模** | 24h + 3h (1 GPU, CPU sim) | ~数小时 (1 GPU, IsaacGym) | GPU sim 更快 |

### HORA 相对 RMA 的简化

1. 去掉了 curriculum (HORA 的任务相对简单)
2. 去掉了异步部署 (单手控制算力足够)
3. 去掉了仿生奖励的部分项 (无足部接触、无步态周期)
4. 用 IsaacGym GPU 并行替代 RaiSim CPU sim

### HORA 相对 RMA 的创新

1. **将 RMA 从 locomotion 迁移到 manipulation**: 验证了框架的通用性
2. **Grasp cache**: locomotion 不需要初始化机制，manipulation 需要预计算稳定抓取
3. **训练物体范围 → 真实世界形状泛化**: RMA 的地形泛化 vs HORA 的物体形状泛化，本质相同但领域不同
4. **手指步态涌现**: 类似 RMA 的步态涌现，但在 16 DoF 手指空间中更复杂

---

## 局限性（论文原文 Section VI）

1. **仅靠 proprioception 是盲的**: 下台阶或碰到大石头时可能突然失败。论文明确指出需要视觉（exteroception）补充
2. **无法处理突发大扰动**: 如被踢或突然碰撞
3. **训练环境有限**: 没见过楼梯但能 70% 成功，暗示泛化有上限

---

## 对 Qi 系列工作的意义

RMA 是 HORA → PenSpin → DexScrew → ... 整个技术树的**根节点**。它建立了三个被后续所有工作继承的核心原则：

1. **Privileged teacher → proprioceptive student**: 两阶段范式
2. **低维 extrinsics 优于精确 SysID**: 不追求物理真实
3. **自然约束驱动行为涌现**: 不手工设计步态/接触模式

Qi 的独特贡献在于**验证这些原则在 manipulation（而非 locomotion）中同样成立**，并在后续工作中系统性地探索了当这些原则不够时应该怎么办（PenSpin: 加真实数据；DexScrew: 降低 sim 要求；AINA: 完全绕过 sim）。
