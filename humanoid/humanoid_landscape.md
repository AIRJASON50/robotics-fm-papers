# Humanoid Whole-Body Control -- Landscape Overview

> 读者背景: PPO sim2real 实践者 (灵巧手), 正在向 foundation model 方向转型
> 目的: 以 takeaway 驱动, 快速建立 12 篇 humanoid 论文的全局认知
> 参考: `humanoid_research_landscape.md` (详细技术分析), 本文为精简结构化版本

---

## 1. 主题分类

### Theme A: Motion Tracking / Imitation -- 从跟踪到基础能力

**定义**: 给定人类运动参考轨迹, 训练 RL 策略让仿真/真实 humanoid 精确复现该运动。

**核心问题**: 如何让单一策略学习大规模、多样化的运动技能, 并泛化到未见过的运动?

| 论文 | 年份 | 一句话 |
|------|------|--------|
| DeepMimic | 2018 | 开创 motion imitation + RL 框架: reference state init + early termination + per-joint tracking reward |
| PHC | 2023 | Progressive Multiplicative Control Policy (PMCP) 实现万级 motion clip 学习 + fail-state recovery |
| SONIC | 2025 | 去掉 progressive training, 用暴力 scale up (100M frames, 42M params) 直接训练 universal tracker |
| BeyondMimic | 2025 | 证明 compact RL (3 reg + 1 task reward) + 精确建模 > 复杂 reward engineering; diffusion planner 实现零样本任务泛化 |

**演化逻辑**: DeepMimic 定义了问题, PHC 解决了 scalability (渐进式), SONIC 用另一种方式解决了 scalability (暴力规模化), BeyondMimic 则挑战了 "需要复杂 reward" 的假设。

**与你的关联**: 你做 PPO sim2real, motion tracking 的 reward 结构 ($r = \sum w_i \exp(-k_i \|x^{ref} - x^{sim}\|^2)$) 与灵巧手轨迹跟踪本质相同。SONIC 证明这个范式可以 scale 到 foundation task 级别。

**Takeaway**:
- Motion tracking 是 humanoid WBC 的 "ImageNet moment" -- 统一的 foundation task, 数据天然丰富
- 规模化路径: PHC 的 "聪明训练" 被 SONIC 的 "暴力算力" 超越, 与 LLM scaling law 一致
- BeyondMimic 的教训: reward engineering 的投入回报递减, 不如投入数据质量和系统建模

---

### Theme B: Teleoperation -- 从遥操作到数据基础设施

**定义**: 人类操作员通过 VR/RGB/MoCap 等设备实时控制 humanoid 全身运动。

**核心问题**: 如何跨越 embodiment gap (人-机器人的运动学/动力学差异), 实现低延迟、高保真的全身控制?

| 论文 | 年份 | 一句话 |
|------|------|--------|
| H2O / OmniH2O | 2024 | 首个 learning-based 全身遥操作; teacher-student distillation 去除 MoCap 依赖 |
| FPO | 2025 | Flow policy gradient 替代 Gaussian PPO, 打通 flow/diffusion policy 的 RL 训练路径 |
| OmniRetarget | 2025 | Interaction mesh retargeting: 数据质量 > reward engineering; HoloSoma 开源框架 |
| TWIST2 | 2025 | VR-based 便携式全身数据采集系统, hierarchical policy (motion tracker + diffusion) |

**演化逻辑**: H2O 验证了 RL-based teleoperation 可行, OmniRetarget 从数据端提升质量, TWIST2 将 teleoperation 重新定位为 "foundation model 的数据采集基础设施", FPO 提供了更强表达力的策略训练方法。

**与你的关联**: TWIST2 的 hierarchical architecture (low-level tracker + high-level diffusion) 是你从 PPO 向 foundation model 转型的模板。low-level 仍然是 PPO tracker (你的强项), high-level 换成 diffusion/VLA。

**Takeaway**:
- Teleoperation 的价值正从 "控制手段" 转变为 "数据采集手段" -- 面向 VLA 训练
- FPO++ 的意义: 打通了 "BC 预训练 flow policy + RL 微调" 的路径, 类似 LLM 的 SFT + RLHF
- OmniRetarget 的核心论点: 高质量参考轨迹 + 简单 RL (5 reward terms) >> 低质量轨迹 + 复杂 RL (10+ terms)

---

### Theme C: Sim-to-Real Transfer -- 缩小仿真与现实的鸿沟

**定义**: 将仿真中训练的策略部署到真实机器人上, 解决物理参数、延迟、执行器模型等差异。

**核心问题**: Domain randomization 是万能药还是有更精准的方法?

| 论文 | 年份 | 一句话 |
|------|------|--------|
| ASAP | 2025 | Delta action model: 用真实数据训练残差模型修正仿真器, 比 DR 更精准 |
| BeyondMimic | 2025 | 精确建模 + 最小 DR, 零样本迁移; 证明系统实现质量可替代大量 DR |

**两种哲学的对比**:

| 维度 | ASAP | BeyondMimic |
|------|------|-------------|
| 哲学 | "Gap 不可避免, 用数据补偿" | "把系统建模做好, gap 自然小" |
| 真实数据需求 | 需要 (采集真实 rollout) | 无 (零样本) |
| 适用场景 | 仿真器精度受限的已有平台 | 系统实现精良的新平台 |

**与你的关联**: 你做灵巧手 sim2real, DR 是标配。ASAP 的 delta action model 提供了更精确的替代方案 -- 特别适合执行器建模困难的场景 (如腱驱动灵巧手)。

**Takeaway**:
- DR 不是唯一选择: residual learning (ASAP) 和 principled formulation (BeyondMimic) 都能实现 zero-shot 迁移
- Delta action model 的本质: $a_{corrected} = a_{policy} + \delta(s_t)$, 只学残差, 比全 dynamics learning 更可行
- 系统建模的投入回报 > 盲目加 DR; 先精确建模, 再最小化 DR

---

### Theme D: Video/World Model-based Control -- 从视觉到行动

**定义**: 从视频中提取运动和交互信息, 或用 learned world model 替代 physics simulator 做策略训练。

**核心问题**: 如何利用海量视频数据和 learned dynamics 扩展机器人能力?

| 论文 | 年份 | 一句话 |
|------|------|--------|
| HDMI | 2025 | Robot-object co-tracking: 从单目视频学习接触密集的人-物交互技能 |
| RWM | 2025 | Neural world model 替代 physics simulator, MBPO-PPO 在 imagination 中训练策略 |

**与你的关联**: RWM 是 DreamerV3 思路在 humanoid 上的落地, 验证了 "learned simulator + PPO" 的可行性。HDMI 将 motion tracking 从自由空间扩展到 loco-manipulation, 是灵巧操作的全身版本。

**Takeaway**:
- 视频是未被充分利用的数据源: HDMI 证明单目 RGB 视频 -> retargeting -> RL 可以学习复杂交互技能
- RWM 开辟了 "无 physics simulator" 的训练路线, 但当前精度仍低于 IsaacGym 等仿真器
- HDMI + OmniRetarget 的互补: 前者 "数据不完美, reward 补偿", 后者 "数据做好, reward 简化"

---

## 2. 研究者脉络

### CMU LeCAR Lab (核心人物: Tairan He)

```
H2O (2024) --> ASAP (2025) --> HDMI (2025)
遥操作验证     sim2real 精化    视频驱动交互
```

**轨迹解读**: 从验证 RL-based teleoperation 可行 (H2O), 到解决部署中的 sim2real gap (ASAP), 再到利用视频数据扩展交互能力 (HDMI)。逐步从 "需要专业设备" 走向 "只要一段视频"。

**方法论一致性**: 都基于 Unitree G1, 都使用 gradient-based retargeting (SMPL -> robot joint space), RL 训练用 IsaacGym。HDMI 的 co-tracking 框架是 H2O 的 motion tracking 的自然扩展。

---

### NVIDIA (核心人物: Zhengyi Luo)

```
PHC (2023) --> SONIC (2025) --> GR00T WBC (产品化)
渐进式学习     规模化 tracker    融入 GR00T 生态
```

**轨迹解读**: PHC 用 PMCP 解决了大规模 motion tracking 的技术难题, SONIC 发现暴力 scale up 更有效 (去掉 PMCP, 直接训练), 最终融入 GR00T 产品线成为 foundation model 的 low-level controller。

**关键转折**: PHC -> SONIC 体现了 "算法创新 vs 规模化" 的张力。PHC 的 PMCP 是精巧的工程, 但 SONIC 证明 42M params + 100M frames 可以 brute-force 解决同样的问题。这与 LLM 领域的 scaling law 启示一致。

**产品化路径**: SONIC Universal Token Space 成为 GR00T N1 (VLA) 的 low-level executor:
- GR00T N1: vision + language -> high-level motion command (~1-10 Hz)
- SONIC tracker: motion command -> joint targets (50 Hz)
- PD controller: joint targets -> torque (500 Hz)

---

### Amazon FAR Lab

```
FPO (2025) + OmniRetarget (2025) + TWIST2 (2025)
训练方法       数据质量            数据采集系统
```

**轨迹解读**: 三篇论文互补形成完整的 pipeline -- OmniRetarget 生成高质量训练数据, TWIST2 建立便携式数据采集系统, FPO 提供 flow/diffusion policy 的 RL 训练方法。

**独特贡献**: Amazon FAR 是唯一同时解决 "数据从哪来" (TWIST2), "数据质量怎么保证" (OmniRetarget), "策略怎么训练" (FPO) 三个问题的团队。HoloSoma 开源框架降低了 humanoid 研究的门槛。

---

## 3. 发展脉络 (2018-2025)

```
2018  DeepMimic
      | motion imitation + RL 的范式定义
      | 每个技能单独训练一个策略
      |
2023  PHC
      | 万级 motion clip, 一个策略
      | Progressive training 解决 scalability
      | Fail-state recovery 实现永续控制
      |
2024  H2O / OmniH2O
      | 从仿真跨越到真实机器人
      | Teacher-student 去除 MoCap 依赖
      | 多模态输入 (VR / RGB / language)
      |
2025  [四路并进]
      |
      +-- 规模化路线: SONIC
      |   "Motion tracking 是 foundation task"
      |   100M frames, 42M params, universal token space
      |   融入 GR00T 产品生态
      |
      +-- 精确建模路线: BeyondMimic + ASAP
      |   "数据质量和系统建模 > reward engineering + DR"
      |   零样本迁移 aerial cartwheel 等极端动作
      |
      +-- 数据基础设施路线: TWIST2 + OmniRetarget
      |   "Teleoperation 服务于 foundation model 数据采集"
      |   便携 VR + 高质量 retargeting + hierarchical policy
      |
      +-- 视觉/世界模型路线: HDMI + RWM
          "视频数据 + learned dynamics 扩展能力边界"
          Robot-object co-tracking, neural simulator
```

**宏观趋势**:
1. **从单技能到 universal**: DeepMimic (1 skill/policy) -> PHC (10K clips/policy) -> SONIC (all motions/policy)
2. **从 reward engineering 到 data engineering**: 复杂 reward (10+ terms) -> 简单 reward + 高质量数据 (BeyondMimic, OmniRetarget)
3. **从 motion tracking 到 interaction tracking**: 自由空间运动 (SONIC) -> 人-物交互 (HDMI)
4. **从独立系统到 foundation model 集成**: standalone RL policy -> GR00T N1 + SONIC (VLA + tracker)

---

## 4. 对你的行动建议

### 4.1 短期 (继续 PPO sim2real)

- **Motion tracking reward 结构** 可直接复用到灵巧手: $r = \sum w_i \exp(-k_i \|x^{ref} - x^{sim}\|^2)$
- **BeyondMimic 的 compact RL** 启示: 减少 reward terms, 增加系统建模精度
- **ASAP 的 delta action model**: 如果灵巧手执行器建模困难, 考虑 residual learning

### 4.2 中期 (向 foundation model 转型)

- **TWIST2 的 hierarchical architecture** 是转型模板: 保留 PPO tracker 作为 low-level, 上层换 diffusion/VLA
- **FPO++ 的 flow policy gradient**: 打通 "BC 预训练 + RL 微调" 的路径, 类似 RLHF
- **SONIC 的 universal token space**: 理解如何统一多模态输入到单一 RL 策略

### 4.3 长期 (foundation model 时代)

- **GR00T N1 + SONIC 集成**: 这是当前最清晰的 "VLA -> humanoid" 产品路径
- **World model (RWM)**: 如果 learned dynamics 精度提升, 可能替代 IsaacGym 做训练
- **Loco-manipulation at scale**: SONIC (universal tracking) + HDMI (interaction) 的融合是未解决的关键问题

---

## 5. 交叉引用

| 本库论文 | 交叉目录 | 关联 |
|---------|---------|------|
| SONIC | `foundation_model/robotics/families/GR00T_Series/` | SONIC 是 GR00T WBC 的 low-level controller |
| FPO | `foundation_model/robotics/policy_learning/` | Flow policy 与 Diffusion Policy 同源 |
| BeyondMimic | `foundation_model/robotics/policy_learning/` | Latent diffusion model 用于运动规划 |
| RWM | `foundation_model/robotics/world_model/` | 与 DreamerV3 同属 learned world model 路线 |
| TWIST2 | `foundation_model/robotics/families/pi_Series/` | Hierarchical architecture 类似 pi_0 |
| Humanoid-Locomotion-Survey | `foundation_model/surveys/robotics/` | 综述交叉参考 |

**GR00T 系列在本库的分布**:
- `foundation_model/robotics/families/GR00T_Series/`: GR00T N1, N1.5, N1.6, DreamGen, DreamZero
- `humanoid/25_SONIC/`: SONIC / GR00T WBC (motion tracking 核心)
- SONIC 同时属于 humanoid (技术) 和 GR00T (产品) 两个维度

---

## 附录: 12 项目速查

| # | 项目 | 路径 | 主题 | 核心贡献 | Sim2Real |
|---|------|------|------|---------|----------|
| 1 | DeepMimic | `motion_tracking/18_DeepMimic` | A | Motion imitation + RL 范式定义 | -- |
| 2 | PHC | `motion_tracking/23_PHC` | A | PMCP 万级 clip 训练 + fail recovery | -- |
| 3 | H2O | `teleoperation/24_H2O` | B | 首个 RL-based 全身遥操作 | H1 |
| 4 | ASAP | `sim2real/25_ASAP` | C | Delta action model 修正仿真器 | G1 |
| 5 | BeyondMimic | `motion_tracking/25_BeyondMimic` | A+C | Compact RL + diffusion planner | G1 |
| 6 | FPO | `teleoperation/25_FPO` | B | Flow policy gradient for RL | T1, G1 |
| 7 | HDMI | `video_world_model/25_HDMI` | D | Robot-object co-tracking from video | G1 |
| 8 | OmniRetarget | `teleoperation/25_OmniRetarget` | B | Interaction mesh retargeting | G1, H1, T1 |
| 9 | RWM | `video_world_model/25_RWM` | D | Neural world model for RL training | ANYmal, G1 |
| 10 | SONIC | `25_SONIC` | A | Universal motion tracking at scale | G1 |
| 11 | TWIST2 | `teleoperation/25_TWIST2` | B | VR-based data collection for VLA | G1 |
| 12 | Survey | `Humanoid-Locomotion-Survey/` | -- | 三大范式 + 五大趋势的元分析 | -- |
| 13 | GMR | `retargeting/25_GMR` | A+B | 非均匀局部缩放 + 两阶段 diff-IK, 17+ 机器人 | G1 |
