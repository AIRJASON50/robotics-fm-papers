# Humanoid Robot Research Landscape

基于 Humanoid Locomotion Survey (Gu et al., 2025) 的分类框架，结合 humanoid 目录下 11 篇论文的深度分析，梳理人形机器人研究的技术脉络、验证综述观点、并指出综述的盲区与未来方向。

---

## 1. 领域全景

### 1.1 三大范式演进

综述 (Section 1, Table 1) 将人形机器人运动控制划分为三大范式，各范式间并非替代关系，而是通过共享的三个统一原则 (physics-based modeling, constrained decision making, adaptation to uncertainty) 逐步融合。

| 范式 | 时期 | 核心方法 | 优势 | 局限 | 本库代表论文 |
|------|------|---------|------|------|------------|
| Classical Control | 1960s-- | ZMP, HZD, MPC, whole-body QP | 稳定性、可解释性、安全保证 | 依赖精确模型、手工调参多、适应性差 | -- |
| Learning-based Control | 2015-- | RL + GPU sim + domain randomization + teacher-student | 敏捷性、鲁棒性、可从数据学习 | 无 motion ref 步态不自然、跨任务迁移难、sim-to-real gap | `18_DeepMimic`, `23_PHC`, `24_H2O`, `25_ASAP`, `25_SONIC`, `25_HDMI`, `25_TWIST2` |
| Emerging Frontiers | 2023-- | Diffusion policy, world model, VLA, foundation model | 多模态行为、不确定性建模、零样本泛化 | 数据稀缺、推理延迟、安全性未验证 | `25_BeyondMimic`, `25_FPO`, `25_RWM` |

综述 (Section 3.3, Table 2) 特别强调 learning-based 与 classical 之间的内在联系:

| Learning-based 技术 | Classical 对应 | 本库验证 |
|---------------------|---------------|---------|
| Domain randomization | Robust control | `24_H2O` (10+ DR 类型), `25_ASAP` (对比 DR baseline) |
| Adaptation module / history encoder | Adaptive control / system ID | `24_H2O` (25-step history 替代 MoCap), `25_SONIC` (FSQ latent) |
| Reward shaping | Constrained optimization | `25_OmniRetarget` (极简 5 项 reward 证明数据质量可替代 reward engineering) |
| Latent space | Reduced-order modeling | `25_SONIC` (FSQ tokenized latent), `25_BeyondMimic` (diffusion latent space) |
| Hierarchical RL | Cascaded control | `25_TWIST2` (motion tracker + diffusion policy 两层), `25_BeyondMimic` (tracker + diffusion planner) |

### 1.2 关键时间线

```
2018  DeepMimic (humanoid/18_DeepMimic)
      |  Motion imitation + RL 的开创性框架
      |  Reference State Initialization + Early Termination
      |
2023  PHC (humanoid/23_PHC)
      |  PMCP 渐进式学习 → AMASS 11K clips 100% 成功率
      |  Fail-state recovery (永续控制)
      |  建立 SMPL → Isaac Gym pipeline
      |
2024  H2O / OmniH2O (humanoid/24_H2O)
      |  首个 learning-based whole-body teleoperation
      |  Teacher-student distillation + history 替代 MoCap
      |  零样本 sim-to-real (Unitree H1)
      |
2025  [多个方向同时突破]
      |
      +-- ASAP (humanoid/25_ASAP)
      |   Delta action model: 残差学习桥接 sim-to-real gap
      |
      +-- SONIC (humanoid/25_SONIC)
      |   Motion tracking 作为 foundation task
      |   规模化: 100M frames, 42M params, 9K GPU hours
      |   Universal token space 支持多模态输入
      |
      +-- BeyondMimic (humanoid/25_BeyondMimic)
      |   Compact RL formulation + Latent diffusion model
      |   Classifier guidance 实现零样本任务泛化
      |   人类级别的敏捷性 (aerial cartwheel)
      |
      +-- HDMI (humanoid/25_HDMI)
      |   Robot-object co-tracking: motion tracking → loco-manipulation
      |   Unified interaction reward + residual action
      |
      +-- OmniRetarget (humanoid/25_OmniRetarget)
      |   Interaction mesh retargeting: 数据质量 > reward engineering
      |   HoloSoma 开源框架, 跨 embodiment
      |
      +-- FPO (humanoid/25_FPO)
      |   Flow policy gradient: 替代 Gaussian PPO 的表达性策略
      |   首次 flow policy sim-to-real (Booster T1, Unitree G1)
      |
      +-- RWM (humanoid/25_RWM)
      |   Neural network world model 替代 physics simulator
      |   MBPO-PPO: imagination 中训练, 零样本部署
      |
      +-- TWIST2 (humanoid/25_TWIST2)
          便携式全身遥操作数据采集系统 (VR-based, mocap-free)
          Hierarchical visuomotor policy (motion tracker + diffusion)
```

---

## 2. 技术分类与论文映射

### 2.1 Motion Imitation 谱系

Motion imitation 是 learning-based humanoid control 的核心使能技术。综述 (Section 3.2) 将其分为 indirect (adversarial, 如 AMP) 和 direct tracking (DeepMimic 系) 两大流派。本库论文全部属于 direct tracking 的演化链。

```
DeepMimic (2018, humanoid/18_DeepMimic)
  |  Phase-aware policy + motion imitation reward
  |  RSI (Reference State Initialization) + Early Termination
  |
  +--→ PHC (2023, humanoid/23_PHC)
  |     Progressive Multiplicative Control Policy (PMCP)
  |     万级 motion clip 的可扩展训练
  |     Fail-state recovery (getup)
  |     ** + AMP discriminator (paper 未充分强调)
  |
  +--→ H2O/OmniH2O (2024, humanoid/24_H2O)
  |     从仿真 avatar → 真实机器人的跨越
  |     Teacher-student distillation 去除 MoCap 依赖
  |     Standing data augmentation 解决静止平衡
  |
  +--→ SONIC (2025, humanoid/25_SONIC)
        Motion tracking 作为 foundation task
        去掉 AMP disc, 去掉 progressive training
        "数据规模 + 网络容量" >> "渐进式训练"
        Universal token space 统一多模态输入
        100M frames, 42M params, 零样本 sim-to-real
```

**演化趋势**: DeepMimic 的核心框架 (imitation reward + RSI + early termination) 在所有后续工作中被保留，但 **规模化路径** 从 PHC 的"渐进式学习"转向 SONIC 的"暴力 scale up"。PHC 的 PMCP 是重要的历史性贡献，但后续工作证明数据规模 + 网络容量可能比训练技巧更重要。

**Motion Tracking Reward 的统一结构** (贯穿所有论文):

$$r = \sum_i w_i \cdot \exp\left(-k_i \cdot \|x_i^{ref} - x_i^{sim}\|^2\right)$$

| 论文 | 跟踪项 | 特殊设计 |
|------|--------|---------|
| DeepMimic | pos + rot + vel + ang_vel | Phase variable |
| PHC | pos + rot + vel + ang_vel | + AMP disc reward (50/50 split) |
| H2O/OmniH2O | pos + rot + vel | + max feet height reward, upper/lower body 分权 |
| SONIC | pos + rot + vel + ang_vel | FSQ token alignment loss |
| HDMI | pos + rot + vel | + interaction reward (contact position + force) |
| OmniRetarget | pos + rot + vel | 极简 5 项, 直接复用 BeyondMimic 超参 |
| BeyondMimic | pos + rot + vel | 仅 3 个 regularization term + 1 个 unified task reward |

#### BeyondMimic: Motion Imitation 的方法论转折

BeyondMimic (`humanoid/25_BeyondMimic`) 代表了 motion imitation 方法论的一次重要转折。它挑战了"鲁棒 sim-to-real 需要大量 domain randomization + reward regularization + 复杂 observation"的传统假设。其核心洞察:

1. **Principled reward formulation**: 仅用 3 个 regularization term + 1 个 unified task reward，无需逐动作调参
2. **Careful system implementation**: 基于经典力学原则建模执行器，最小化部署差异 (delay 等)
3. **Moderate domain randomization**: 仅对真正不确定的物理属性随机化

结果: 用同一组超参数学习数百种技能 (含 aerial cartwheel 等极端动作)，并零样本迁移到真实硬件。这与综述 (Section 3.4) 关于"RL 训练需要大量手动调参"的论断形成了有趣的对比。

### 2.2 Sim-to-Real 谱系

综述 (Section 3.1, Section 3.2) 详细讨论了 sim-to-real 的三大策略: domain randomization, system identification, learned dynamics。

| 方法 | 综述分类 | 本库代表 | 核心思路 |
|------|---------|---------|---------|
| Domain Randomization | Section 3.1 | `24_H2O` (10+ DR 项), `25_BeyondMimic` (moderate DR) | 在训练中注入噪声, 策略对不确定性鲁棒 |
| System Identification | Section 3.2 | `25_ASAP` (delta action model) | 用真实数据辨识/校正仿真器 |
| Learned Dynamics | Section 3.2 | `25_RWM` (neural world model) | 直接从数据学习 forward dynamics |
| Principled Formulation | -- | `25_BeyondMimic` (minimal DR) | 精确建模 + 最小 DR，而非大量 DR 补偿 |

**ASAP** (`humanoid/25_ASAP`) 的两阶段框架最直接地对应综述的 sim-to-real 讨论:
- Stage 1: 仿真中预训练 motion tracking policy (标准 RL pipeline)
- Stage 2: 部署采集真实数据 → 训练 delta action model (残差模型) → 用 delta model 修正仿真器 → 微调策略

Delta action model 的本质是在仿真器的 forward dynamics 上叠加一个残差修正项: $a_{corrected} = a_{policy} + \delta(s_t)$。这比 domain randomization 更精确 (针对性补偿)，比 full dynamics learning 更可行 (只学残差)。综述 (Section 3.2) 将其归类为 residual learning，与 ASAP 的前身 (ANYmal actuator model) 属同一技术路线。

**BeyondMimic vs ASAP 的对比** 代表了 sim-to-real 的两种哲学:

| 维度 | BeyondMimic | ASAP |
|------|------------|------|
| 哲学 | "把系统建模做好，sim-to-real gap 自然小" | "gap 不可避免，用数据补偿" |
| 真实数据需求 | 无 (零样本) | 需要 (采集真实 rollout) |
| DR 策略 | 最小化 (仅真正不确定的参数) | 标准 DR 作为预训练基础 |
| 适用场景 | 系统实现精良的新平台 | 仿真器精度受限的已有平台 |

### 2.3 数据采集与重定向

综述 (Section 3.2) 讨论了人类运动数据在 RL 训练中的角色，但对数据采集和重定向 (retargeting) 的讨论相对简略。本库的多篇论文在此方向有重要贡献。

#### 2.3.1 Retargeting 技术演进

```
关键点匹配 (PHC/GMR)
  |  无穿透保证, foot skating 严重
  |
  +--→ 梯度优化 (H2O, ASAP)
  |     gradient-based SMPL → robot joint space
  |     11 个关键点对应, 优化关节角度
  |     物理可行性依赖 "sim-to-data" 过滤
  |
  +--→ 软约束 (VideoMimic)
  |     碰撞约束与关键点目标竞争, 难调参
  |
  +--→ 硬约束 + 交互保持 (OmniRetarget, humanoid/25_OmniRetarget)
        Interaction Mesh: Laplacian deformation energy
        SQP 逐帧求解, 硬约束保证:
        - Non-penetration (signed distance)
        - Joint limits
        - Velocity limits
        - Foot sticking during stance
```

**OmniRetarget 的核心论点**: "数据质量 > reward engineering"。高质量参考轨迹 + 简单 RL (5 个 reward term) > 低质量参考轨迹 + 复杂 RL (大量 ad-hoc regularizer)。

这个论点与综述 (Section 3.4) 的 performance boundary 讨论形成互补: 综述指出"RL 策略对 motion reference 依赖大，无参考时步态不自然"，OmniRetarget 进一步指出"参考轨迹的质量直接决定了 RL 的上限"。

#### 2.3.2 数据采集系统

| 系统 | 设备 | 成本 | 全身控制 | 便携性 | 本库对应 |
|------|------|------|---------|--------|---------|
| MoCap-based | Vicon + MoCap suit | 高 ($50K+) | 完整 | 差 (实验室) | TWIST (SONIC 前身) |
| VR-based | PICO4U (头+手+脚) | 低 ($1K) | 完整 | 好 | `25_TWIST2` |
| RGB camera | 消费级相机 | 极低 | 有限 (估计误差大) | 极好 | `24_H2O` (HybrIK) |
| 视频重建 | 录制 → TRAM/GVHMR | 极低 | 有限 | 极好 | `25_ASAP`, `25_HDMI` |

**TWIST2** (`humanoid/25_TWIST2`) 代表了数据采集的实用化方向:
- VR-based, mocap-free: 用 PICO4U VR 头显获取全身运动
- 自研 2-DoF 机器人颈部 ($250): 支持第一人称视觉 (egocentric vision)
- 15 分钟采集 100 个成功 demo
- Hierarchical visuomotor policy: motion tracker (low-level) + diffusion policy (high-level)

TWIST2 与 OmniH2O 的定位差异: OmniH2O 聚焦于"高质量实时遥操作"，TWIST2 聚焦于"可扩展的数据采集"。TWIST2 明确面向 VLA/foundation model 的数据需求，是连接 teleoperation 和 foundation model 训练的桥梁。

### 2.4 World Models 与 Generative Control

综述 (Section 4.1) 将 "discriminative → generative" 作为第一大趋势。本库有两篇论文直接贡献于此方向。

#### 2.4.1 RWM: Neural Network World Model

**RWM** (`humanoid/25_RWM`) 提出用 neural network simulator 替代 physics simulator 进行策略训练:

- **Dual-autoregressive mechanism**: 结合历史观测-动作对和自身预测，在 POMDP 中实现长时域可靠预测
- **Self-supervised training**: 无需 domain-specific inductive bias
- **MBPO-PPO**: 在学习的 world model 中用 PPO 训练策略，零样本部署到 ANYmal D (四足) 和 Unitree G1 (人形)
- 据作者称，这是首个无 domain-specific knowledge、从 learned simulator 训练策略并部署到真实硬件的框架

RWM 对应综述 (Fig. 2) 中的 "world models" 类别 -- 预测高维机器人和环境行为的潜在动力学模型。但 RWM 的独特之处在于它不是用于 latent imagination (如 DreamerV3)，而是直接替代 physics simulator 作为 RL 的训练环境。

| 维度 | Physics Simulator (IsaacGym) | RWM (Neural Simulator) |
|------|-----|-----|
| 建模方式 | 显式物理方程 + 碰撞检测 | 数据驱动的神经网络 |
| 精度来源 | 物理引擎质量 | 训练数据质量 + 泛化能力 |
| Sim-to-real gap | 固有 (模型简化) | 可通过真实数据训练缩小 |
| 可扩展性 | GPU 并行 | GPU 并行 |
| 适应性 | 需要手动更新参数 | 可在线学习新数据 |

#### 2.4.2 BeyondMimic: Diffusion-based Versatile Control

**BeyondMimic** (`humanoid/25_BeyondMimic`) 的 diffusion 部分实现了综述 (Section 4.1) 描述的 "generative policy" 范式:

1. **State-action co-diffusion model**: 在 latent space 中对动作分布建模，以 predictive control 方式运作
2. **Classifier guidance**: 利用 diffusion model 的梯度场，在 test-time 对任意可微目标进行在线优化
3. **零样本任务泛化**: 无需重新训练即可解决 waypoint navigation, obstacle avoidance, motion inpainting 等新任务

这直接验证了综述 (Section 4.1, Eq. 4) 的核心公式:

$$\max_{\phi} J(\pi_\phi) = \mathbb{E}_{\tau \sim p_\tau(\cdot|\pi_\phi)} \left[ \log p_{\text{data}}(\tau) - \lambda D(\pi_\phi(\tau) \| p_{\text{phys}}(\tau)) \right]$$

BeyondMimic 将 motion tracking (RL) 产生的 atomic skills 编码为 diffusion model 的 data distribution $p_{data}$，physics consistency 则通过 RL 训练阶段已隐式保证。

#### 2.4.3 FPO: 生成式策略的训练方法

**FPO** (`humanoid/25_FPO`) 解决的是另一个维度的问题: 如何用 policy gradient 方法训练 flow/diffusion 策略。

核心矛盾: flow/diffusion policy 表达力强，但 likelihood 计算代价极高，传统 PPO 无法直接使用。

FPO++ 的解决方案: 用 conditional flow matching (CFM) loss 差值作为 log-likelihood ratio 的代理:

$$\hat{\rho}_{\text{FPO}}(\theta) = \exp\left(\hat{\mathcal{L}}_{\text{CFM},\theta_{\text{old}}} - \hat{\mathcal{L}}_{\text{CFM},\theta}\right)$$

两个关键改进: (1) per-sample ratio (细粒度 trust region), (2) ASPO (不对称 clipping 防止 entropy collapse)。

**FPO 对 humanoid control 的意义**: 打通了 "imitation learning pre-trained flow policy → RL fine-tuning" 的路径。这与综述 (Section 4.1) 所讨论的 "从判别式到生成式策略的路径" 高度吻合 -- diffusion/flow policy 不仅可以用于模仿学习 (如 pi_0)，也可以从 RL reward 中学习，使 RL 策略具有多模态行为的表达能力。

### 2.5 Loco-Manipulation

综述 (Section 4.4) 将 "locomotion → loco-manipulation" 列为第四大趋势。本库有两篇论文直接贡献于此。

#### HDMI: 从视频学习人-物交互

**HDMI** (`humanoid/25_HDMI`) 将 motion tracking 扩展到接触密集的 human-object interaction:

核心框架: **Robot-Object Co-Tracking**

```
RGB 视频 → 人体运动 + 物体轨迹 + 接触信号
            |
            v
        RL Co-Tracking: 同时跟踪 robot 和 object 的参考轨迹
            |
            v
        门打开/箱子搬运/攀爬等交互技能
```

三个关键设计:

1. **Unified Object Representation**: 物体状态统一为 root-relative pose + 接触点，不区分铰接/刚体/固定/浮动
2. **Residual Action Space**: $\theta_t^{target} = \theta_t^{ref} + a_t$ -- 探索以参考姿态为中心，对极端姿态学习至关重要
3. **Interaction Reward**: $R = \exp(-\|p_{eef} - p_{target}\|) \cdot \min(\exp(\|F\| - F_{thres}), 1) \cdot c_t$ -- position + force, 由 contact signal 门控

#### OmniRetarget: 数据端的 Loco-Manipulation

**OmniRetarget** (`humanoid/25_OmniRetarget`) 从数据生成端解决 loco-manipulation:

核心论点: "与其在 RL 端用 ad-hoc reward 补偿低质量参考轨迹，不如在数据端从根源保证质量"。

Data Augmentation 的核心价值:
- 单个示范 → 多变体 (不同物体位置/尺寸/平台高度)
- 与纯 DR 随机化的关键区别: 运动学增广给出了不同场景下的"正确答案"，而非靠 RL 自行探索

**HDMI vs OmniRetarget: 同一问题的两种解法**

| 维度 | HDMI | OmniRetarget |
|------|------|-------------|
| 核心策略 | 数据不完美 → reward 补偿 | 数据做好 → reward 简化 |
| 数据来源 | 单目 RGB 视频 (获取成本低) | MoCap + constrained optimization (质量高) |
| RL 复杂度 | 中等 (tracking + interaction reward) | 极简 (5 reward terms) |
| 可扩展性 | 视频数据易获取 | 需要 MoCap + 优化 pipeline |
| 理想方案 | 两者结合: OmniRetarget 级 retargeting + HDMI 风格 interaction reward 作 safety net |

---

## 3. 综述验证与补充

### 3.1 综述观点的验证

#### 趋势 1: Discriminative → Generative (Section 4.1)

**强验证**:
- `25_BeyondMimic`: Latent diffusion model + classifier guidance 实现零样本任务泛化，直接验证了 Eq. 4 的生成式控制公式
- `25_FPO`: FPO++ 打通了 flow policy 的 RL 训练路径，验证了生成式策略可以从 reward 中学习 (不仅限于模仿)
- `25_RWM`: Neural world model 实现了 "在想象中训练" 的 model-based 路线

综述的预测 "层次化设计 -- generative planner + discriminative controller" 被 BeyondMimic 和 TWIST2 精确验证:
- BeyondMimic: diffusion planner 提出运动分布 → RL tracker 执行
- TWIST2: diffusion policy 提出全身关节目标 → motion tracker 执行

#### 趋势 2: Uni-modal → Multi-modal (Section 4.2)

**部分验证**:
- `25_SONIC`: Universal token space 统一了 VR teleoperation, 人类视频, 文本/音乐, VLA model 四种模态输入
- `24_H2O`: 从 RGB camera → VR → language (GPT-4o) 的多输入源支持

综述提到的 VLA (Vision-Language-Action) 方向在本库中通过 SONIC 与 GR00T N1 (`foundation_model/25_GR00T_N1`) 的集成得到验证。

但**视觉感知在 humanoid locomotion 中的深度融合**在本库论文中仍然较少。大部分论文的 observation 仍以 proprioception 为主，视觉感知主要用于 high-level planning (如 TWIST2 的 egocentric vision) 而非 low-level control。

#### 趋势 3: Single-task → Multi-task (Section 4.3)

**强验证**:
- `25_SONIC`: 100M frames 训练的 universal tracker，一个策略处理所有运动
- `25_BeyondMimic`: 同一组超参数学习数百种技能，classifier guidance 实现零样本任务切换
- `23_PHC`: PMCP 渐进式学习是早期的多任务尝试，虽然后来被 SONIC 的暴力 scale up 超越

综述提到的 "catastrophic forgetting" 和 "reward conflict" 问题在 PHC → SONIC 的演进中被证实: PHC 需要 PMCP 处理难样本，而 SONIC 证明足够大的网络容量 (42M params) 可以直接解决。

#### 趋势 4: Locomotion → Loco-manipulation (Section 4.4)

**强验证**:
- `25_HDMI`: 从 motion tracking 扩展到 robot-object co-tracking
- `25_OmniRetarget`: 搬箱子、攀爬、翻墙等 loco-manipulation 任务

但综述提到的 "dual-agent RL" (如 FALCON) 在本库中未见。HDMI 和 OmniRetarget 都采用统一策略而非分离的上/下半身控制。

#### 趋势 5: Offline training → Test-time adaptation (Section 4.5)

**部分验证**:
- `25_BeyondMimic`: Classifier guidance 是一种 test-time optimization -- 在推理时对新目标进行梯度优化
- `25_ASAP`: Delta action model 本质上是 offline adaptation (真实数据 → 仿真器校正)，但非 online test-time

综述提到的 "in-context learning" 和 "diffusion test-time adaptation" 在本库中尚未见到完整实现。

### 3.2 综述的盲区

通过对 11 篇论文的分析，发现综述有以下几个未充分覆盖的方面:

#### (1) 数据质量 vs Reward Engineering 的范式之争

综述 (Section 3.1-3.2) 讨论了 reward shaping 和 motion imitation 的技术细节，但**未明确提出 "数据质量可以替代 reward engineering" 这一范式性观点**。OmniRetarget 和 BeyondMimic 共同验证了:

> 高质量参考轨迹 + 简单 reward (3-5 项) >> 低质量参考轨迹 + 复杂 reward (10+ 项)

这对 humanoid control 的实践有重大指导意义: 研究者应**优先投入数据质量**，而非 reward engineering。

#### (2) Retargeting 技术的深度讨论

综述仅在 Section 3.2 简要提到 retargeting，但本库中 retargeting 质量对下游 RL 的影响被多篇论文证实:

| 论文 | Retargeting 方法 | 发现 |
|------|-----------------|------|
| `24_H2O` | Gradient-based (SMPL → H1) | Sim-to-data 过滤不可行动作显著提升成功率 |
| `25_OmniRetarget` | Interaction mesh + SQP | Retargeting 质量与 RL 成功率正相关 |
| `25_SONIC` | GMR + PyRoki | 170 位受试者的 retargeting 支持跨 embodiment |

这表明 retargeting 不是简单的数据预处理步骤，而是一个独立的、对下游性能有决定性影响的研究方向。

#### (3) Code-Paper Discrepancy

综述作为元分析自然无法覆盖论文代码中的实现细节。但本库的深度代码分析揭示了大量"论文未提及但对复现至关重要"的实现:

- PHC: AMP discriminator 实际贡献 50% reward (论文未充分强调)
- H2O/OmniH2O: 10+ 种 observation version, 多维 curriculum, motion package loss 模拟通信丢包
- OmniRetarget: 代码支持 FastSAC (off-policy), 对称性利用, penalty curriculum -- 论文均未提及

#### (4) Flow Policy 的 RL 训练

综述 (Section 4.1) 讨论了 diffusion policy 在 humanoid control 中的应用前景，但主要聚焦于 behavior cloning 和 offline RL。FPO++ (`humanoid/25_FPO`) 展示了一种全新的路径: **直接用 policy gradient 训练 flow policy，无需 likelihood 计算**。这开辟了 "RL from scratch with expressive policy" 的可能性，是综述 "discriminative → generative" 趋势的补充方向。

#### (5) 便携式数据采集的系统设计

综述未讨论数据采集系统的设计。TWIST2 (`humanoid/25_TWIST2`) 展示了一个完整的、面向 foundation model 训练的数据采集 pipeline:

```
VR 设备 (PICO4U) → 全身 retargeting → Motion tracker → Robot 执行
                                                          |
                                                          v
                                                    Egocentric vision data
                                                          |
                                                          v
                                                    Diffusion Policy 训练
```

这个 pipeline 是连接 "teleoperation" 和 "foundation model" 的关键基础设施，综述虽然提到了两端，但未讨论中间的数据闭环。

### 3.3 综述预测 vs 实际发展

| 综述预测 (Section 4.6) | 2025 年实际发展 | 验证状态 |
|------------------------|---------------|---------|
| "Physics-guided generative intelligence" 将成为统一范式 | BeyondMimic: RL (physics) + diffusion (generative) 的结合 | 部分验证 |
| System 1 (fast reactive) + System 2 (slow deliberative) 分层 | TWIST2, BeyondMimic: motion tracker (System 1) + diffusion planner (System 2) | 验证 |
| Test-time adaptation 将替代 offline DR | BeyondMimic: classifier guidance 实现 test-time optimization | 部分验证 |
| Foundation-scale models 赋能 humanoid | SONIC + GR00T N1 集成; TWIST2 为 VLA 训练提供数据 | 进行中 |
| 安全性与可靠性是关键瓶颈 | 本库论文几乎未涉及安全性证明 | 未验证 |
| 平台成本下降使研究民主化 | Unitree G1 成为多篇论文的标准平台 | 验证 |

**关键未实现预测**:
- 综述预测的 "test-time adaptation" 主要指在线参数调整，但实际发展 (BeyondMimic) 是 test-time optimization for new objectives，这比综述预想的更 general
- 综述预测的 "formal safety guarantees" 在 2025 年的工作中仍然缺席 -- 所有 learning-based 系统仍然是黑箱

---

## 4. 发展趋势总结

### 4.1 当前主流路线

2025 年的 humanoid whole-body control 研究形成了三条清晰的主流路线:

**路线 A: Scale Up Motion Tracking (SONIC 路线)**
- 核心: 将 motion tracking 作为 foundation task，用规模化解决泛化
- 方法: 大数据 (100M frames) + 大模型 (42M params) + 大算力 (9K GPU hours)
- 优势: 一个策略处理所有运动，支持多模态输入
- 挑战: 算力需求高，不涉及物体交互

**路线 B: Clean Data + Simple RL (BeyondMimic/OmniRetarget 路线)**
- 核心: 数据质量决定性能上限，RL formulation 越简单越好
- 方法: Principled reward + careful system implementation + moderate DR
- 优势: 不需要大量 reward engineering，泛化性好，可用 diffusion 扩展
- 挑战: 对系统实现精度要求高

**路线 C: Teleoperation → Foundation Model (TWIST2 路线)**
- 核心: 建立便携式数据采集基础设施，为 VLA/foundation model 提供训练数据
- 方法: VR teleoperation + hierarchical policy (motion tracker + diffusion/VLA)
- 优势: 面向长期的 foundation model 路线
- 挑战: 数据采集效率和质量的 trade-off

### 4.2 未解决的问题

1. **Object Interaction at Scale**: SONIC 实现了运动的规模化，但不涉及物体。HDMI 处理物体交互，但每个技能需要单独训练。如何将两者统一 -- 在 universal motion tracker 中加入物体交互能力 -- 仍是开放问题。

2. **灵巧手控制的集成**: 本库所有论文中，灵巧手控制要么不涉及，要么通过独立 IK 处理 (如 OmniH2O)。将手指级别的灵巧操作纳入全身控制框架是重要的缺失。

3. **Online Adaptation**: 虽然 BeyondMimic 的 classifier guidance 实现了 test-time optimization，但这是对新目标的优化，而非对新动力学 (如地形变化、载荷变化) 的适应。真正的 online dynamics adaptation 仍然缺乏。

4. **Safety**: 无论是 RL-trained 策略还是 diffusion-based 控制器，都无法提供形式化的安全保证。综述 (Section 4.6) 将此列为首要挑战，2025 年的论文几乎未触及。

5. **Long-horizon Autonomy**: 当前的 autonomous execution (如 TWIST2 的 diffusion policy) 仍限于短时域任务。如何在 10 分钟以上的时间尺度上保持稳定和目标一致性 (即 HDMI 中 67 次连续开门的能力扩展到更复杂场景)。

6. **Evaluation Standardization**: 各论文使用不同的评估 metric (成功率定义、MPJPE 计算方式、termination 条件)，缺乏统一的 benchmark。SONIC 在这方面做了一些工作 (统一的大规模 AMASS 测试集)，但还不够。

### 4.3 未来方向预测

基于当前趋势，预测以下方向在 2025-2027 年将成为重点:

1. **Unified Loco-Manipulation Tracker**: 将 SONIC 的 universal token space 扩展到包含物体状态，实现 "track everything" (机器人 + 物体 + 接触)。这是 SONIC + HDMI 的自然融合。

2. **Dexterous Whole-Body Control**: 将灵巧手操作纳入全身控制框架。需要统一处理 30 DOF 身体 + 20 DOF 双手的高维控制问题。

3. **World Model-based Planning**: RWM 展示了 neural simulator 的可行性。下一步是将其与 diffusion-based planning (BeyondMimic) 结合 -- 在 learned world model 中用 diffusion planner 做长时域推理。

4. **VLA for Humanoid**: SONIC 已初步展示了与 VLA 模型 (GR00T N1) 的集成。随着 TWIST2 等系统提供更多训练数据，预计将出现专门面向人形机器人的 VLA 模型。

5. **Test-time Dynamics Adaptation**: BeyondMimic 的 classifier guidance + ASAP 的 delta action model 的结合 -- 在 test-time 同时优化目标和动力学补偿。

---

## 5. 与 Manipulation / Foundation Model 的交叉

### 5.1 Humanoid x Manipulation

humanoid 和 manip 两个目录的论文存在明确的交叉点:

| 交叉维度 | Humanoid 侧 | Manipulation 侧 | 融合方向 |
|---------|-------------|-----------------|---------|
| Contact modeling | HDMI: interaction reward (position + force) | Contact-rich manipulation reward | 统一的接触建模框架 |
| Residual action | HDMI: ref + delta | Wrist residual + finger absolute | 多层次 residual |
| Data augmentation | OmniRetarget: 运动学增广 | Object pose variation | 交互增广 |
| Teleoperation | TWIST2: whole-body VR | Dexterous hand teleoperation | 全身 + 灵巧手统一遥操 |

**关键洞察**: Humanoid loco-manipulation (HDMI, OmniRetarget) 与桌面 dexterous manipulation 共享许多技术要素 (contact reward, residual action, trajectory tracking)，但前者需要额外处理 balance, locomotion, floating base 的复杂性。

### 5.2 Humanoid x Foundation Models

综述 (Section 4.2) 讨论了 VLA 等 foundation model 对 humanoid control 的赋能。本库的 foundation_model 目录与 humanoid 目录有以下关键交叉:

| Foundation Model | Humanoid 应用 | 本库论文 |
|-----------------|---------------|---------|
| `24_DiffusionPolicy` | BeyondMimic 的 diffusion planner, TWIST2 的 high-level policy | Diffusion 作为 motion prior |
| `24_pi0` | TWIST2 类似的 hierarchical 框架; FPO++ 可作为 pi_0 的 RL fine-tuning 方法 | VLA → Humanoid |
| `25_GR00T_N1` | SONIC 直接与 GR00T N1 集成 | 专用 humanoid VLA |
| `23_DreamerV3` | RWM 继承了 latent world model 的思路 | World model → Humanoid |
| `25_AwesomeWorldModels` | RWM 所在的更广泛 world model 研究背景 | 综述交叉 |
| `25_RobotScalingLaws` | SONIC 展示了 humanoid control 的 scaling behavior | Scaling 验证 |

**SONIC + GR00T N1 的集成路径**:

```
GR00T N1 (Vision + Language → high-level action)
    |  输出: 抽象运动指令 / motion token
    v
SONIC Universal Token Space
    |  将 VLA 输出转换为 motion tracking 目标
    v
SONIC Motion Tracker (RL policy)
    |  输出: joint targets
    v
PD Controller → Unitree G1
```

这个架构对应综述 (Fig. 3B) 的 "System 2 (slow, deliberative) + System 1 (fast, reactive)" 分层:
- System 2: GR00T N1 / diffusion planner (~1-10 Hz)
- System 1: SONIC motion tracker (~50 Hz)

**FPO++ 对 Foundation Model 的意义**: pi_0 等 VLA 模型使用 flow matching 训练，FPO++ 提供了直接用 RL reward 微调这些模型的方法。这开辟了 "大规模预训练 (BC) → task-specific 微调 (RL)" 的路径，与 LLM 领域的 RLHF 范式类似。

---

## 附录: 论文分类速查表

| 论文 | 目录路径 | 年份 | 综述范式 | 综述趋势 | 核心技术 | Sim-to-Real |
|------|---------|------|---------|---------|---------|------------|
| DeepMimic | `humanoid/18_DeepMimic` | 2018 | Learning | -- | Motion imitation, RSI | 无 |
| PHC | `humanoid/23_PHC` | 2023 | Learning | Single→Multi task | PMCP, fail recovery, AMP | 无 |
| H2O/OmniH2O | `humanoid/24_H2O` | 2024 | Learning | Uni→Multi modal | Teacher-student, history obs, teleoperation | 有 (H1) |
| ASAP | `humanoid/25_ASAP` | 2025 | Learning | Offline→Adaptation | Delta action model, residual dynamics | 有 (G1) |
| BeyondMimic | `humanoid/25_BeyondMimic` | 2025 | Emerging | Disc→Generative, Single→Multi task | Compact RL + latent diffusion, classifier guidance | 有 (G1) |
| FPO | `humanoid/25_FPO` | 2025 | Emerging | Disc→Generative | Flow policy gradient, ASPO | 有 (T1, G1) |
| HDMI | `humanoid/25_HDMI` | 2025 | Learning | Loco→Loco-manipulation | Robot-object co-tracking, interaction reward | 有 (G1) |
| OmniRetarget | `humanoid/25_OmniRetarget` | 2025 | Learning | Loco→Loco-manipulation | Interaction mesh retargeting, data augmentation | 有 (G1, H1, T1) |
| RWM | `humanoid/25_RWM` | 2025 | Emerging | Disc→Generative | Neural world model, MBPO-PPO | 有 (ANYmal, G1) |
| SONIC | `humanoid/25_SONIC` | 2025 | Learning/Emerging | All five trends | Universal token space, motion tracking at scale | 有 (G1) |
| TWIST2 | `humanoid/25_TWIST2` | 2025 | Learning/Emerging | Uni→Multi modal, Loco→Loco-manip | VR teleoperation, hierarchical visuomotor policy | 有 (G1) |
