# 灵巧操作研究全景 (Dexterous Manipulation Landscape)

> **读者画像**: PPO sim2real 灵巧手实践者, 正在向 Foundation Model 方向转型
> **本文目的**: 梳理 manip/ 目录下 5 个主题的定义、核心问题和相互关系, 为 FM 转型提供导航

---

## 0. 从人形机器人到灵巧操作: 先验、桥梁与新挑战

灵巧操作不是从零开始的——它继承了 humanoid whole-body control 的大量先验, 同时面临全新的挑战。

### 从 humanoid 带来的先验

| 先验 | humanoid 中怎么用 | manip 中怎么用 |
|------|-----------------|---------------|
| **PPO + sim2real** | 全身步态控制 (PHC, SONIC) | 灵巧手关节控制 (ArtiGrasp, OmniReset) |
| **Motion tracking 作为统一目标** | SONIC: 100M 帧动捕→全身追踪 | DexTrack: 3585 条手部轨迹→灵巧追踪 |
| **Domain randomization** | ASAP: 摩擦/质量/延迟随机化 | SimToolReal/Dex4D: 物体几何/纹理随机化 |
| **Teacher-student 蒸馏** | PHC/SONIC: state teacher → deploy | BiDexHD: state teacher → vision student |
| **Universal Token Space** | SONIC: FSQ 对齐人和机器人 | DexLatent: VAE 对齐不同灵巧手 |

### humanoid→manip 的自然演进

```
humanoid 解决的问题:             manip 面临的新问题:
  "怎么走和平衡"                   "怎么抓和操作"
  全身 29 DOF                      手部 16-24 DOF (更密集的接触)
  动捕数据丰富 (AMASS 100M帧)       手部数据稀缺 (ARCTIC ~几千条)
  运动是周期性的 (步态)              操作是非周期的 (每个任务不同)
  reward 可以简化 (跟踪位置)         reward 必须包含接触 (力/滑动)
  sim2real gap 主要在地面            sim2real gap 在接触面 (更难)
```

### 三个核心新挑战

**挑战 1: 接触建模**
Humanoid 的主要接触是脚-地面 (大面积、可预测)。灵巧操作的接触是指尖-物体 (小面积、高度不确定、力敏感)。仿真中的接触模型 (MuJoCo soft contact) 和真实接触差距更大, 这使得 sim2real 在 manipulation 中比 locomotion 更难。

**挑战 2: 物体多样性**
Humanoid 的环境相对固定 (平地/楼梯/斜坡)。灵巧操作面对的物体是开放集合 (任意形状/材质/重量)。这要求策略对物体有 zero-shot 泛化能力——传统 per-object RL 做不到, 必须用 object-centric representation (SimToolReal) 或 foundation model (DexLatent)。

**挑战 3: 数据获取**
Humanoid 有 AMASS (100M 帧全身动捕)。灵巧手没有同等规模的数据集——ARCTIC 只有几千条, TACO 只有 141 个任务。这就是为什么 UltraDexGrasp 用合成数据 (20M 帧) 和 DexTrack 用 data flywheel 来绕过人类数据瓶颈。

### Takeaway

**humanoid 教会了我们 "motion tracking + PPO + sim2real" 这套方法论。manip 继承了这套方法论, 但发现它在接触密集、物体多样、数据稀缺的灵巧操作场景中不够用。五个主题 (traditional_rl→human2robot→scaling_rl→sim2real→fm_manip) 就是解决这三个新挑战的五次尝试。**

---

## 1. Traditional RL -- 单任务强化学习

**定义**: 用 PPO 等 on-policy RL 算法, 在仿真中为特定物体/任务训练灵巧操作策略。策略直接映射本体感知到关节动作, 训练依赖精心设计的 reward shaping 和 curriculum。这是灵巧操作领域最成熟的范式, 也是绝大多数后续工作的起点。

**核心问题**: 如何用 RL 让高自由度手(MANO 51-DOF / Shadow 24-DOF)完成接触密集的操作任务?

**代表论文**:

| 论文 | 一句话总结 |
|------|-----------|
| ArtiGrasp (2023) | PPO + curriculum, 双手 MANO 在 RaiSim 中抓取并操作铰接物体, 统一策略处理 grasp + articulation |
| PhysHOI (2023) | 全身(含手指)人-物交互的物理模仿, 提出 Contact Graph Reward 解决接触稀疏问题 |
| ObjDexEnvs (2024) | 两层架构: Transformer planner 生成腕部轨迹 + PPO controller 学指尖动作, 利用腕部运动跨越 embodiment gap |

**与你的关联**: 这是你当前的主力范式。三篇论文展示了 PPO 在灵巧操作中的三种典型架构:

- **单一策略直通** (ArtiGrasp): 观测 → 一个策略网络 → 动作, 中间不做模块分解。策略同时学习手指控制和腕部跟随, 靠 curriculum (先固定物体单手训→再自由物体双手协调) 降低学习难度。注意: "直通"指网络架构, 不是没有仿真器 -- 训练在 RaiSim 中用 PPO 完成。
- **模仿奖励** (PhysHOI): 用人类动作参考计算 reward (Contact Graph Reward), 替代手工 reward 设计。
- **层次化控制** (ObjDexEnvs): 上层 Transformer 规划腕部轨迹 → 下层 PPO 学指尖动作, 模块化分解。

ObjDexEnvs 的"腕部规划 + 指尖 RL"分层思路在后续工作中被反复验证。

**Takeaway**: Reward shaping 和 curriculum 是传统 RL 的生命线, 但也是其可扩展性的天花板 -- 每增加一个任务, 就要重新调一套 reward。

---

## 2. Human2Robot -- 从人类演示到机器人策略

**定义**: 利用人手交互数据(ARCTIC、GRAB、TACO 等数据集或实时遥操)作为先验, 通过 retargeting、tracking、蒸馏等手段将人类操作技能迁移到机器人手上。核心挑战在于人手与机器手之间的 embodiment gap -- 运动学结构、自由度、接触面积都不同。

**核心问题**: 如何跨越 human-robot embodiment gap, 将人类灵巧操作的"知识"迁移给机器人?

**代表论文**:

| 论文 | 一句话总结 |
|------|-----------|
| BiDexHD (2024) | TACO 数据集 -> 自动构建仿真任务 -> IPPO teacher -> DAgger 蒸馏 vision student, 141 个双手工具任务 |
| DexMachina (2024) | ARCTIC 人手数据 retarget 到 6 种机器手, hybrid residual action + PPO 学习铰接物体操作 |
| DexTrack (2025) | 通用 neural tracking controller, RL+IL 混合训练 + data flywheel 迭代扩充, 覆盖 GRAB+TACO 3585 条轨迹 |
| HumDex (2026) | IMU 遥操 + 学习式 retargeting + 两阶段模仿学习, 面向人形机器人全身灵巧操作的完整 pipeline |

**与你的关联**: human2robot 路线直接扩展了你的 PPO 经验 -- DexMachina 和 DexTrack 的底层 controller 就是 PPO, 只是把 reward 从手工设计换成了 tracking reference。HumDex 展示了从数据采集到部署的端到端流程, 是你做 sim2real 部署时的重要参考。关键趋势是从"离线数据集 retarget"(BiDexHD, DexMachina)演进到"在线 tracking"(DexTrack)再到"实时遥操采集"(HumDex)。

**Takeaway**: Embodiment gap 的最优解法不是精确复制人手动作, 而是提取 task-level intent (物体轨迹、接触模式) 然后让 RL 在机器人 embodiment 上重新发现执行策略。

> **深入阅读**: Retargeting 技术的独立调研 (13 篇, 2024-2025) 见 `retargeting/CLAUDE.md`.
> 从优化方法 (Interaction Mesh) 到学习方法 (GeoRT, Residual RL) 的完整技术脉络.
> 核心演变: "运动等价" → "功能等价", 1:1 keypoint 对应 → 几何/接触语义驱动.

---

## 3. Scaling RL -- 通过规模化突破 RL 瓶颈

**定义**: 探索如何通过增加计算规模、多样化初始状态分布等手段, 让简单的 RL 算法(PPO)在灵巧操作中产生涌现行为, 而非依赖人工 reward engineering 和 curriculum。这一主题受 LLM scaling 的启发, 试图回答: 机器人 RL 是否也存在 scaling law?

**核心问题**: 能否用"简单算法 + 大规模计算"替代"精巧 reward + 人工 curriculum", 实现 emergent dexterity?

**代表论文**:

| 论文 | 一句话总结 |
|------|-----------|
| OmniReset (2025) | 离线预计算 diverse reset states + 大规模 PPO (Isaac Lab), 无 per-task reward shaping 即可完成 pick-reorient-insert 等长 horizon 任务 |

**与你的关联**: 这是你最应该关注的方向之一。OmniReset 证明了 PPO 在灵巧操作中的 exploration saturation 问题可以通过 diverse resets 缓解 -- 不需要换算法, 只需要改 reset 分布。其核心洞察是: 增加 parallel environments 不等于增加 state coverage, 真正的 scale 需要 diversity。这与 LLM 中"数据多样性比数据量更重要"的结论一致。

**Takeaway**: Scale 不是简单堆算力, 而是扩展状态空间覆盖。Diverse resets 是 RL 版的"数据增强", 可以让你现有的 PPO pipeline 直接受益。

---

## 4. Sim2Real -- 从仿真到真实世界的迁移

**定义**: 研究如何让仿真中训练的灵巧操作策略在真实世界中 zero-shot 或 few-shot 工作。核心手段包括: domain randomization、object-centric 表示(解耦物体形状与任务逻辑)、task-agnostic 策略(一个策略适配多任务)。与 traditional RL 的区别在于, sim2real 更关注泛化和鲁棒性而非单任务性能。

**核心问题**: 如何设计策略表示和训练流程, 使得 sim-trained policy 能够 zero-shot 泛化到真实世界的未见物体和任务?

**代表论文**:

| 论文 | 一句话总结 |
|------|-----------|
| SimToolReal (2025) | "所有工具使用 = 依次到达 6D 目标位姿", 在 primitive 物体上训练 goal-conditioned policy, zero-shot 迁移 12 种真实工具 |
| Dex4D (2026) | 以 4D point tracks (物体表面关键点随时间的 3D 轨迹) 为统一任务表示, 仿真中训练 task-agnostic policy, 通过 video demo 指定新任务 |

**与你的关联**: 作为 PPO sim2real 实践者, 这两篇论文是你最直接的技术升级路径。关键 insight 是: 不要为每个任务设计 reward, 而是设计一个足够通用的 goal representation (6D pose sequence 或 4D point tracks), 然后训练 goal-conditioned policy。这样一个策略可以覆盖整个任务族, sim2real 的 domain gap 也被限缩到感知层面(pose estimation / point tracking)。

**Takeaway**: Object-centric representation 是 sim2real 泛化的关键 -- 把"任务"从 reward function 中解耦出来, 变成 goal conditioning 的输入, 可以极大提升策略复用性。

---

## 5. FM for Manipulation -- Foundation Model 驱动的灵巧操作

**定义**: 将大规模预训练模型(VLA, Vision-Language-Action)应用于灵巧操作, 或用大规模合成数据训练通用操作模型。这一主题的核心张力在于: VLA 提供了强大的语义理解和泛化能力, 但灵巧操作对精度和接触力控制的要求远超一般 pick-and-place 任务, 且不同手型的 action space 完全不同。

**核心问题**: 如何让 Foundation Model 的泛化能力与灵巧操作的精度要求兼容? 如何跨越不同手型的 action space 鸿沟?

**代表论文**:

| 论文 | 一句话总结 |
|------|-----------|
| RL Token (2025, PI) | 冻结 VLA (pi-0.6), 训练小型 encoder-decoder 提取 "RL token" 作为状态表示, 用 off-policy RL 在线精调动作头, 真机数小时内提升亚毫米精度 |
| DexGraspVLA (2025) | 层级式 VLA: VLM 做高层规划 + Diffusion 做低层控制, 90%+ unseen cluttered scene 灵巧抓取, 支持 failure recovery |
| DexLatent (2026) | Multi-headed VAE 将不同灵巧手映射到共享 32D latent space, VLA 在 latent space 预测动作, 新手型 zero-shot 接入 |
| UltraDexGrasp (2026) | 纯合成数据 pipeline (BODex + cuRobo + SAPIEN) 生成 20M 帧 demo, BC 训练 point cloud 策略, 1000+ 物体 zero-shot 部署 |
| UniDex (2026) | 从 egocentric 人类视频生成 50K 轨迹覆盖 8 种灵巧手 (6-24 DoF), FAAS 统一动作空间实现跨手迁移, 3D VLA 达 81% 任务完成率 |
| PAM (2026) | Sim-to-real HOI 视频生成引擎 (Pose-Appearance-Motion), 合成数据可替代 50% 真实数据, FVD 29.13 |

**与你的关联**: 这是你的目标方向。六篇论文展示了 FM + manipulation 的多种结合模式:
- **RL Token**: FM 提供表示, RL 提供精度 -- 最能利用你的 PPO 经验
- **DexGraspVLA**: VLM 规划 + Diffusion 执行的层级架构 -- 展示 VLA 如何处理 cluttered scene 灵巧抓取
- **DexLatent**: 解决跨 embodiment 问题, 如果你需要支持多种灵巧手这是必读
- **UltraDexGrasp**: 用合成数据替代人工采集, BC 替代 RL -- 代表了"数据工程 > 算法工程"的趋势
- **UniDex**: 从人类视频到跨手迁移的完整 pipeline -- FAAS 与 DexLatent 的 latent space 思路互补
- **PAM**: HOI 视频生成做数据增强 -- 与 UltraDexGrasp 的合成数据思路一脉相承, 但走视频生成路线

**Takeaway**: FM 时代的 RL 不再是端到端训练策略, 而是在 FM 提供的 representation/prior 之上做精度校准。你的 PPO 经验将演变为"如何高效 fine-tune FM 的 action head"。

**CS→Robotics 的核心范式**: 直接复用 CV 预训练的 VLM 作为视觉理解 backbone, 只需接 action head 教它"怎么动"。VLM 在互联网图文上学到的空间理解/动作语义/物体状态识别对机器人直接有用, 经 CLIP(2021)→RT-2(2023)→OpenVLA(2024)→pi_0(2024)→GR00T N1(2025) 五年逐步验证。不用 VLM backbone 的方案 (如 OmniReset 的 ResNet-18 distillation, 50% 成功率) 反而是需要论证的。单任务少数据可能不需要 VLM backbone, 但多任务/语言指令驱动场景下这是必需的。

---

## 发展脉络

```
传统 RL (2022-2024)
  |
  +---> human2robot (2024-2025): 引入人类演示作为先验, 减少 reward engineering
  |       |
  |       +---> scaling RL (2025): 不依赖人类数据, 用 diverse resets + scale 产生涌现行为
  |       |
  |       +---> sim2real (2025-2026): 从 per-task 转向 task-agnostic, 统一目标表示
  |       |
  |       +---> FM manipulation (2025-2026): VLA + RL fine-tune / cross-hand latent / synthetic data
  |
  (每条路线都保留了 PPO 作为底层训练算法)
```

**为什么领域这样演化**:

1. **传统 RL -> human2robot**: 单纯 RL 的 reward engineering 成本随任务数线性增长。人类演示提供了 task-level prior, 将 reward 从"手工设计目标函数"简化为"跟踪参考轨迹"。

2. **human2robot -> scaling RL**: 人类数据本身也有获取瓶颈(需要 mocap 设备、retargeting 流程)。OmniReset 试图绕过人类数据, 纯靠 RL + scale 达到类似效果。这条路线还很早期, 但方向明确。

3. **human2robot -> sim2real**: 积累了足够多的仿真策略后, 自然面临"如何部署"的问题。从 per-task sim2real 到 task-agnostic sim2real 的转变, 本质是用更通用的 goal representation (6D pose / point tracks) 替代 per-task reward。

4. **全部 -> FM manipulation**: 当操作技能需要覆盖开放词汇的物体和指令时, 传统方法(无论是 RL 还是 BC)的 scalability 都不够。FM 提供了语义理解和跨任务泛化, 但需要与灵巧操作的精度要求对接。RL Token 和 DexLatent 分别从"精度"和"跨 embodiment"两个角度解决这个对接问题。

**关键趋势**: 底层的 PPO 训练范式始终存在, 但其角色从"端到端训练策略"逐步演变为"在先验之上精调"(human2robot 中的 tracking controller, FM 中的 action head fine-tune)。

---

## 对 Foundation Model 的启示

从灵巧操作研究中可以提炼出以下对 robotics FM 设计的关键教训:

### 1. Action Space 是灵巧 FM 的核心难题

不同于 navigation 或 pick-and-place (6-7 DOF), 灵巧手的 action space 高达 16-24 DOF 且跨手型差异巨大。DexLatent 的 VAE latent space 方案表明: **FM 不应直接预测关节角, 而应在一个与硬件无关的 latent space 中预测**, 然后由 per-hand decoder 映射到具体关节。

### 2. FM 的泛化能力与 RL 的精度能力互补

RL Token 证明了一个实用的组合模式: 冻结 VLA 做 representation, 轻量 RL 做 fine-tune。这意味着未来的灵巧操作 FM 不需要在预训练阶段解决精度问题 -- **pre-train for generalization, fine-tune for precision**。

### 3. 合成数据 pipeline 可能比真实数据更重要

UltraDexGrasp 的 20M 帧合成数据 + 简单 BC 的效果表明: 对于灵巧抓取这类可以精确仿真的任务, **数据工程(grasp synthesis + motion planning + physics validation)比算法创新更能推动性能**。这与 LLM 领域"数据质量 > 模型架构"的结论高度一致。

### 4. Object-centric representation 是跨任务泛化的关键

SimToolReal 和 Dex4D 都将任务统一为物体的目标轨迹(6D pose 或 4D point tracks)。这提示 FM 的 visual encoder 应该能输出 **object-centric 的空间表示**(而非 image-level feature), 才能与灵巧操作的需求对接。

### 5. Diverse resets / curriculum 的思想可以迁移到 FM fine-tune

OmniReset 的 diverse resets 本质是扩大 RL 的 state coverage。在 FM fine-tune (如 RL Token 的 online RL 阶段) 中, 类似的思想同样适用: 在更多样的初始条件下 fine-tune, 可以避免 catastrophic forgetting 和 mode collapse。

---

## 交叉参考

| 相关目录 | 内容 | 与本文的关系 |
|----------|------|-------------|
| `manip/QiHaoZhi/` | 齐浩之组的 sim2real 系列 (HORA, PenSpin, DexScrew) 和 human demo 系列 (HOP, AINA, SPIDER) | sim2real 和 human2robot 主题的更多实例, 代码实现参考价值高 |
| `manip/dataset/` | 手部数据集库 (ARCTIC, GRAB, TACO, DexGraspNet, DexCanvas 等) | human2robot 和 FM 主题的数据基础设施 |
| `foundation_model/robotics/` | Google RT Series, PI Series, GR00T Series, 以及 policy learning 方法 | FM manipulation 主题的上游知识 (VLA 架构、训练范式) |
| `foundation_model/CV/` | 视觉基础 (ViT, CLIP, SAM, 3DGS, DINOv2 等) | sim2real 感知模块 (FoundationPose, point tracking) 的基础 |
| `humanoid/` | 全身人形控制 (DeepMimic, SONIC, ASAP 等) | HumDex 连接了灵巧操作与人形控制, 是两个领域的交叉点 |
