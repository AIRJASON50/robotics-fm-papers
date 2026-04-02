# Humanoid Locomotion Survey Notes

Evolution of Humanoid Locomotion Control
Yan Gu, Guanya Shi, Fan Shi, et al. (Purdue / CMU / NUS / UC Berkeley / NYU / Caltech / Meta), 2025.12
GitHub: github.com/purdue-tracelab/Humanoid-Locomotion-Survey

## 1. Survey Scope

Humanoid locomotion control 的演化综述，从 classical control 到 RL 到 generative models。不是技术深度论文，而是 **全景式分类框架 + 范式间联系分析**。

核心论点: 三个范式（classical / learning / emerging）共享三个统一原则:
1. Physics-based modeling
2. Constrained decision making
3. Adaptation to uncertainty

## 2. 三大范式总结 (Fig. 1, Table 1)

### 2.1 Classical Control (1960s-)

| 方法 | 核心思路 | 频率 | 局限 |
|------|--------|------|------|
| Linear feedback (ZMP, capture point, PD) | Reduced-order model + linear control | ~1000 Hz | 仅限预规划行为，鲁棒性差 |
| Nonlinear feedback (HZD, CLF-QP, whole-body QP) | Full-order model + nonlinear control | ~1000 Hz | 能 tracking，但无预测 horizon |
| Trajectory optimization (DDP, iLQR) | Offline long-horizon planning | Offline | 慢，无法实时 |
| MPC (convex, contact-implicit, sampling-based) | Online short-horizon predictive | ~100 Hz | 依赖模型精度，计算受限 |

**Performance boundaries**: 依赖精确状态估计，对 sensor noise/contact uncertainty 脆弱；reduced-order model 快但不准，full-order model 准但慢；需要大量手工调参。

### 2.2 Learning-based Control (2015-)

**仿真器的核心贡献——隐式处理混合动力学**:

传统混合系统建模需要人手写: 每个 domain 的 ODE、切换面 S 的条件、碰撞映射 Δ。接触组合一多（双手双脚+物体），domain 数量组合爆炸，手写不现实。仿真器将这个问题转化为: 只需定义几何体形状和物理属性（摩擦、刚度），碰撞检测+约束求解器自动处理所有接触切换。物理上做的事情和手写混合模型一样，但对使用者透明。

这意味着: 仿真器的出现不仅是"算得快"（GPU 并行），更根本的贡献是**让研究者不再需要显式建模接触切换**，从而可以处理任意复杂的接触场景。没有仿真器，RL 训练中涉及的大量随机接触事件根本无法用手写混合系统描述。

仿真器在自由度上通常是 full-order 的（30+ 关节全保留），但在物理精度上有近似（刚体假设、简化接触模型、离散时间步）。这些近似是 sim-to-real gap 的主要来源。

**关键使能技术**:
- GPU-parallel simulators (Isaac Gym, MuJoCo MJX) — 千倍加速
- Policy gradient (PPO) — 稳定的 on-policy 算法
- Domain randomization — sim-to-real 鲁棒性
- Curriculum learning — 从简单到复杂渐进训练
- Teacher-student / privileged learning — privileged info 加速训练，student 用 deployable obs

**与 classical 的内在联系** (Table 2, Section 3.3):
- Domain randomization ↔ robust control
- Adaptation module (history encoder) ↔ adaptive control / system identification
- Reward shaping ↔ constrained optimization
- Latent space ↔ reduced-order modeling
- Hierarchical RL ↔ cascaded control
- RL policy = offline solve 大量 optimization problem，encode 到 NN 中

**Real-world data 的角色**:
- Human motion data: reference trajectory for tracking reward (DeepMimic, H2O, HumanPlus, ExBody)
- System identification: 采样法辨识 actuator dynamics (ANYmal 系列)
- Learned dynamics: residual learning (ASAP) 或完整 forward dynamics (DexNDM)

**Performance boundaries**: 无 motion reference 时产生不自然步态; 跨任务/embodiment 迁移困难; sim-to-real fidelity 有限; 多数只用 proprioception, 缺 perception; 安全性是黑盒。

### 2.3 Emerging Frontiers (2023-)

五大研究趋势 (Fig. 4):

**1. Discriminative → Generative**
- Discriminative: state → deterministic action (标准 RL policy)
- Generative: state → action distribution (diffusion policy, world model)
- 优势: 多模态行为、不确定性建模、explicit conditioning
- Generative policy 的统一公式 (Eq. 4): maximize data likelihood + KL(physics prior)
- Diffusion model ↔ stochastic optimal control (Hamilton-Jacobi-Bellman)

**2. Uni-modal → Multi-modal**
- 当前: 多数 policy 仅用 proprioception (blind)
- 趋势: 融合 vision, depth, tactile, language, audio
- 架构: LocoTransformer, VB-Com, cross-attention fusion
- VLA (Vision-Language-Action): RT-2, pi_0, GR00T N1, Helix

**3. Single-task → Multi-task**
- 当前: 每个任务独立训练 (walk, run, jump 分开)
- 趋势: shared representation, hierarchical composition, diffusion blending
- Cross-embodiment: BridgeData V2, RT-X, meta-learning
- Foundation models: 统一 task/morphology/sensing

**4. Locomotion → Loco-manipulation**
- 当前: locomotion 和 manipulation 独立研究
- 趋势: co-tracking (robot + object + contact), dual-agent RL, unified controllers
- 代表: FALCON, ULC, Helix, GR00T N1

**5. Offline training → Test-time adaptation**
- In-context learning: 从 input-output behavior 在线调整，无需显式参数更新
- Diffusion test-time adaptation
- 对应 classical 的 adaptive control

## 3. Unified View (Fig. 3)

### 3A: 按 optimization timing × model complexity 分类

```
                   Reduced-order    Full-order    Simulator    Sim+Real data    Internet-scale
Offline            Traj Opt                       RL           Motion Mimic     Emerging
Online             Convex MPC       Nonconv MPC   Hybrid
Minor              Linear FB        Nonlinear FB
```

### 3B: 按 deployment hierarchy 分类

```
Pre-deployment:   Traj Opt  |  RL training        |  Pre-train + Fine-tune
                            |                      |
Deployment:                 |                      |
  Task planning (~1 Hz)     |                      |  Generative / Foundation model  "System 2"
  Motion gen (~100 Hz)      |  RL inference        |  Diffusion inference
  Joint torque (~1000 Hz)   |  (RL inference)      |  (RL inference)               "System 1"
                  MPC       |                      |
                  Nonlinear FB                     |
                  Linear FB                        |
```

System 1 (fast, reactive) vs System 2 (slow, deliberative) 的分层，暗示未来需要两层整合。

## 4. Key Insights for Our Work

### 4.1 与 bh_motion_track / wuji-hand 的关联

| Survey 中的技术 | 我们项目中的对应 |
|---------------|------------|
| Teacher-student privileged learning | Asymmetric AC (privileged critic) |
| Domain randomization | DR in training config |
| Motion tracking reward (DeepMimic family) | bh_motion_track reference trajectory tracking |
| Residual learning (ASAP, DexNDM) | 未来 sim-to-real 方向 |
| Curriculum learning | Potential for staged training |
| PD target tracking (RL → PD) | 当前 action space (delta joint pos → PD controller) |

### 4.2 Classical-Learning 联系的实用价值

Survey 强调 RL 的很多设计本质上是 classical control 的 neural 版本:
- **Reward shaping = 隐式 constraint**: 设计 reward 等价于定义 optimization objective
- **History encoder = adaptive controller**: 从历史推断未知参数
- **Latent space = reduced-order model**: 用低维表示压缩高维 dynamics

这意味着在设计 reward 或 obs space 时，可以显式借鉴 classical control 的理论框架。

### 4.3 Emerging 方向的参考

- **Diffusion policy for manipulation**: 如果未来 bh_motion_track 需要处理多模态行为（不同抓取策略），generative policy 是自然选择
- **World model**: DexNDM 的 joint-wise NDM 本质上是一个简化的 world model，survey 将其归类为 "learned dynamics" (Fig. 2)
- **Test-time adaptation**: 对 sim-to-real 场景，比固定 domain randomization 更灵活

## 5. 推荐阅读 (Section 5)

Survey 推荐的核心教材:
- Dynamics & classical control: *Feedback Control of Dynamic Bipedal Robot Locomotion* [Grizzle et al.]
- RL: *Reinforcement Learning: An Introduction* [Sutton & Barto] + PPO paper
- World model: DreamerV3
- Generative: *Deep Learning* [Goodfellow], *VAE* [Kingma], *Diffusion Models* [Yang et al.]

## 6. Limitations of This Survey

- README 中多个 Section 标记 TBD (Learning from real-world data, Emerging frontiers 的细节)，repo 仍在建设中
- 偏重 locomotion，manipulation 和 loco-manipulation 的覆盖相对简略
- 引用偏向 Berkeley/CMU 生态圈的工作
- 对 generative 方向的讨论偏理论展望，缺少 benchmark 对比
