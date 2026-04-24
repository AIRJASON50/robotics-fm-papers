# GR00T Family -- NVIDIA 人形机器人基础模型总览

> **目的**: 理解 NVIDIA 的全栈人形机器人方案 -- 不是逐篇论文笔记, 而是看整个系统是怎么从零搭起来的。

---

## 1. NVIDIA 的人形机器人战略

NVIDIA **不造机器人**。它的战略是做 **"机器人的操作系统"** -- 提供从训练到部署的全栈软件, 让硬件厂商 (Unitree, Fourier, AGIBot, Galaxea) 都来用 NVIDIA 的方案。

技术栈: Isaac Sim (仿真) -> Isaac Lab (RL 训练) -> GR00T (基础模型: VLA + WBC) -> Jetson (边缘部署) -> Cosmos (世界模型)。GR00T 是这个栈里的 "模型层"。

---

## 2. 两条独立的技术线: Isaac-GR00T 和 SONIC

GR00T 由**两个独立开发的控制器**最终合流而成:

### Isaac-GR00T 线 (VLA: 看 -> 理解 -> 规划)

**团队**: NVIDIA GEAR lab (Jim Fan, Yuke Zhu)
**角色**: 高层决策 -- 理解语言和视觉, 输出目标动作轨迹

**N1 (2025.03) -- 第一个开源人形 VLA**

架构继承自 RT-2 (见 `robotics/policy_learning/`): 冻结的 VLM 编码 vision+language, 动作头 (这里是 Flow Matching DiT) 生成电机指令。双系统设计: VLM 跑慢拍做语义理解, DiT 跑快拍输出平滑动作。数据金字塔 (web video > sim > real) 和 cross-embodiment (一个模型多种机器人) 是核心想法。

*局限*: VLM 冻结导致 language-following 很弱, 零样本泛化差。

**N1 -> N1.5 (2025.12) -- VLM 适配**

- *问题*: N1 的 VLM 冻结后, 视觉特征无法针对 manipulation 任务表达。
- *洞察*: 适配预训练 VLM 有两条路 -- (a) 保持冻结加 adapter 层, (b) 解冻一些层。N1.5 选了 (a): 在 VLM 和 DiT 之间插 adapter, 并换更强的 VLM backbone。
- *新能力*: **FLARE** 训练目标 -- 用 human egocentric video 在没有 action label 的情况下学习, 方法是对齐未来视觉 latent。把无限的人类视频变成免费数据源。
- *结果*: Language-following 和真机成功率大约翻倍; FLARE 带来了 novel-object 泛化。

> **交叉引用**: PI 的 "Knowledge Insulation" 原则 (见 `robotics/vla/`) 处理的是同一个 VLM 冻结困境 -- 他们认为冻结能保护预训练知识但限制下游表达力。N1.5 的 adapter 是一种折中, N1.6 走了相反的路。

**Takeaway**: VLM 冻结成为瓶颈时, adapter 是最轻量的第一剂补救。但 adapter 引入了间接性 -- 动作 loss 的梯度只能间接影响 VLM。

**N1.5 -> N1.6 (2026.03) -- 直接解冻 VLM**

- *问题*: Adapter 层是间接的, 计算也浪费 -- 加了参数但没直接改善 VLM 的表征。
- *洞察*: 解冻 VLM 的顶层更有效。梯度直接从动作 loss 流进 VLM, 用更少的额外参数产生更适合 manipulation 的视觉特征。
- *其他改进*: Native aspect-ratio 图像 (不再 resize/pad 丢信息); state-relative actions (相对 delta 对标定偏差不敏感, sim2real 更好)。
- *新 embodiment*: Bimanual arms, 更多人形平台, locomotion 支持。

**Takeaway**: Adapter 的间接性变成瓶颈时, 解冻 VLM 顶层给出更直接的梯度路径。这和 LLM 里 LoRA vs 全参 fine-tune 的教训一致 -- 有时直接的方法赢。

**N1.6 -> N1.7 (2026.04) -- 人类视频 scaling + 商用开源**

- *问题*: N1.6 主要还是靠 "几千小时 robot teleoperation 数据"。Dexterous manipulation (接触丰富、22 自由度灵巧手) 依然脆弱。Teleop 数据不 scale。
- *洞察 (核心新主张)*: 人类 egocentric video 是 dexterity 的正确底座。如果把 robot action 表示为 **relative EEF delta** (而不是绝对关节角), 人和机器人就能用同一种表征 -- 可以直接 co-train。
- *架构变化*: VLM backbone 升级为 **Cosmos-Reason2-2B (Qwen3-VL 架构)**; 32 层 DiT 保持; 完整 ONNX / TensorRT pipeline。
- *数据变化*: **20,854 小时 EgoScale 人类 egocentric 视频**, 覆盖 20+ 任务类别 (制造/零售/医疗/家居), 加上原有 robot demonstration。大约是 N1.6 预训练数据的 5-10 倍。
- *Scaling law (标志性科学发现)*: 从 1k 小时扩到 20k 小时人类视频, **dexterous manipulation 的平均任务完成率超过 2 倍**。这是**人形机器人领域第一个明确的 dexterity scaling law** -- 地位类似 LLM 的 Chinchilla。
- *License*: Apache 2.0 代码 + NVIDIA Open Model License 权重; 明确标注 "factory-floor ready", 直接支持物料处理/包装/质检等工业部署。

**Takeaway**: N1.7 的赌注是 **"对 dexterity 来说, 数据 scaling 战胜算法 tinkering"** -- 而且跨 embodiment 扩展的正确 primitive 不是什么花哨架构, 而是一个能自然对齐人和机器人数据的简单动作表征 (relative EEF delta)。这和 PI 的 "prompt engineering 优先于 data scaling" (见 pi_0.7 family notes, 同月发布) 是**同目标、对立赌注**的两条路线。

**各版本的关键设计决策**:

| 问题 | N1 | N1.5 / N1.6 | N1.7 | 为什么改 |
|---------|------|--------------|-------|------------|
| VLM-to-DiT 接口 | 冻结 VLM + cross-attention | Adapter (N1.5) / 解冻顶层 (N1.6) | Cosmos-Reason2-2B (Qwen3-VL) | 更强的 reasoning VLM |
| 数据稀缺 | Data pyramid (sim + real) | FLARE (人类视频, 无动作标签) | **20K 小时 EgoScale 人类视频** | 首次显式的 dexterity scaling law |
| 动作表征 | 绝对关节角 | State-relative (N1.6) | **Relative EEF delta** (人机共享) | 统一表征使 co-training 成为可能 |
| 图像编码 | Resize 到固定分辨率 | Native aspect ratio (N1.6) | Native aspect ratio | Resize 破坏信息 |
| License | 仅研究 | 仅研究 | **Apache 2.0 + 商用权重** | 工厂部署就绪 |

### SONIC 线 (WBC: 运动跟踪 -> 关节控制)

**团队**: NVIDIA Research (Zhengyi Luo 等, 从 PHC 脉络来的)
**角色**: 低层执行 -- 跟踪目标运动轨迹, 输出关节角度

**核心思想**:

**(1) Motion Tracking 作为可 scale 的通用目标**

以前: 每个动作 (走/跑/跳/搬) 都要自定义 reward -- 这不 scale。SONIC 把所有动作统一为 "跟踪 mocap 数据", 一套 reward 公式搞定。行为的多样性来自 mocap 数据的多样性, 而不是 reward 工程的多样性。

**(2) Universal Token Space (跨 embodiment 迁移)**

人类骨骼和机器人骨骼不同。SONIC 训练 encoder 把人和机器人的运动都映射到共享的离散 token space (用 FSQ)。推理时: 人类 mocap -> 人 encoder -> 共享 token -> 机器人 decoder -> 关节角。这是一种隐式的 motion retargeting, 不需要手动骨骼映射。

**(3) 人形控制的 Scaling Laws**

数据 scaling (更多样的 mocap) 提升最大, 还没饱和。模型 scaling 有用但次之。计算 scaling 影响的是渐近性能而不只是训练速度。排名 -- data > model > compute -- 和 LLM 的 Chinchilla 发现一致。

---

## 3. 两条线如何合流

从 N1.5 开始, Isaac-GR00T 和 SONIC 以串联方式部署:

```
Language instruction
  |
  v
Isaac-GR00T VLM          (slow: semantic understanding)
  "Understand: move right hand to apple"
  |
  v
Isaac-GR00T DiT          (medium: generate motion trajectory chunks)
  "Plan: target motion trajectory"
  |  target trajectory (SMPL or latent tokens)
  v
SONIC Planner             (plan motion segments)
  |
  v
SONIC Tracker             (fast: track motion -> output joint angles)
  |
  v
PD Controller             (hardware-rate: torque control)
  |
  v
Physical Robot
```

**为什么不 end-to-end?** 每层训练用的数据从根上就不同: VLM 用 TB 级互联网文本+图像 (不需要机器人); DiT 用遥操作 demo (需要机器人); SONIC 用 human mocap (不需要机器人)。End-to-end = 一个模型吃三类数据, 优化目标互相冲突。分层 = 每层独立用自己最好的数据训。

**替代方案**: Decoupled WBC (也在 GR00T-WBC 代码库里) 分得不一样 -- 下半身 gait 用 RL, 上半身精度用 IK。末端精度更好 (例如端杯子不洒), 但全身协调性更差。

---

## 4. 世界模型路线 (DreamGen -> DreamZero)

和 VLA+WBC 并行, NVIDIA 还在探索世界模型路线:

### DreamGen (2025.05): 世界模型做数据增广

角色: 服务 VLA 训练的辅助工具。少量真实 demo 喂给 video world model, 合成各种变体 (背景/光照/物体), 把数据放大几个数量级。这是 data flywheel 的工程落地。

### DreamZero (2026.02): 世界模型 = Policy (GR00T N2 核心)

角色: 下一代架构, 替代 VLA。

> **交叉引用**: DreamZero 的 WAM (World-Action Model) 代表 VLA -> WAM 的范式转移 (见 `world_model/26_DreamZero/`)。

**核心 Takeaway: VLA = 背答案; WAM = 懂原理**

| | VLA (N1.x) | WAM (N2) |
|---|---|---|
| 流程 | 看当前帧 -> 输出记住的动作 | 看当前帧 -> 想象未来 N 帧 -> 从想象中抽动作 |
| 强项 | 训练分布内强 | 训练分布外强 |
| 类比 | 背答案的学生 | 理解因果的学生 |

WAM 的泛化优势和 LLM chain-of-thought 的洞察一致: **推理时多花点算力去 "想"**, OOD 性能更好。

---

## 5. 完整时间线

```
=== Infrastructure (2022-2024) ===
2022    Isaac Sim + Isaac Gym
2023    Isaac Lab (replaces Isaac Gym)
2024    Cosmos (world model infra), Jetson Thor (announced)

=== GR00T Gen 1 (2025) ===
2025.03  N1 -- first open-source humanoid VLA
2025.05  DreamGen -- world model for data augmentation
2025.11  SONIC -- large-scale whole-body motion control
2025.12  N1.5 -- VLM upgrade + FLARE; first N1.5+SONIC combined deployment

=== GR00T Gen 2 (2026) ===
2026.02  DreamZero -- WAM architecture (N2 core)
2026.03  N1.6 -- VLM unfreezing, native aspect ratio, state-relative actions
2026.04  N1.7 -- Cosmos-Reason2-2B, 20K h EgoScale video, relative EEF, Apache 2.0
2026 H2  N2 (announced) -- WAM replaces VLA

=== Hardware Partners ===
Unitree G1, Fourier GR-1, AGIBot Genie-1, Galaxea R1 Pro, Bimanual YAM
```

---

## 6. 核心 Takeaway

| # | Takeaway | 原理 | 对你的行动项 |
|---|----------|-----------|-------------|
| 1 | **分层解耦 > end-to-end** | 不同层用不同数据, 各自独立优化 | 自己实现 Layer 1 (tracking), Layer 3 接开源 VLM |
| 2 | **Motion tracking = 可 scale 的通用目标** | 一个 tracking reward 覆盖所有行为 | 不要给每个动作单独设计 reward |
| 3 | **Data > Model > Compute** | 瓶颈是 mocap 多样性, 不是模型大小 | 优先扩数据, 其次再想扩网络 |
| 4 | **Universal token space** | 共享离散 latent 对齐人和机器人运动 | 可以替代手动 motion retargeting |
| 5 | **Sim2Real 靠 domain randomization, 不靠仿真保真度** | DR 足够 = zero-shot 迁移 | 把精力花在 DRCfg, 而不是调仿真 |
| 6 | **FLARE: 从人类视频学** | 对齐未来 latent, 不需要动作标签 | 人类视频是免费无限的数据源 |
| 7 | **VLA = 背答案; WAM = 懂原理** | "想象未来" 比 "直接映射" 泛化更强 | 紧盯 DreamZero 的后续 |
| 8 | **VLM 冻结/解冻是个连续谱** | Adapter (间接) vs 解冻 (直接) -- 按数据预算和表达力需求选 | 先用冻结+adapter, 碰到瓶颈再解冻 |
| 9 | **Dexterity scaling law: 人类视频而不是 teleop (N1.7)** | 1k -> 20k 小时人类 egocentric 视频 = 任务完成率 2 倍 | 停止 scale teleop 数据, 改去 scale ego video |
| 10 | **Relative EEF delta 统一人和机器人 (N1.7)** | 跨 embodiment 用同一套动作表征就能直接 co-train | 跨 embodiment 迁移时, 选一个天然共享的表征, 而不是要做 retargeting 的表征 |

### SONIC 是人形控制的 SOTA 吗?

**对 "人形全身运动跟踪", 是。** 但有几点 caveat:
- 是 motion tracking 的 SOTA, 不是通用 robot 的 SOTA
- 全身协调性强, 但末端精度弱 (灵巧操作不是它的强项)
- 在有限硬件上验证, 跨 embodiment 泛化没完全测试
- MLP 架构相对于 Transformer 类方案, scaling 上限未知

---

## 7. 文件索引

```
GR00T_Series/
+-- GR00T_family_notes.md                  <-- this file
+-- vla_wbc/
|   +-- Isaac-GR00T/                       # Brain (VLA)
|   |   +-- code/                          #   NVIDIA/Isaac-GR00T repo
|   |   +-- 25_N1/                         #   N1 paper + notes
|   |   +-- 25_N15/                        #   N1.5 blog report
|   |   +-- 26_N16/                        #   N1.6 blog report
|   |   +-- 26_N17/                        #   N1.7 blog report (EgoScale scaling law + Apache 2.0)
|   +-- SONIC/                             # Cerebellum (WBC)
|       +-- code/                          #   NVlabs/GR00T-WholeBodyControl repo
|       +-- SONIC_...md                    #   Paper
|       +-- SONIC_notes.md                 #   Notes (with bh_motion_track comparison)
+-- world_model/
    +-- 25_DreamGen/                       # Data augmentation (Cosmos world model)
    |   +-- GR00T-Dreams/                  #   Code repo
    +-- 26_DreamZero/                      # WAM (N2 core, video diffusion)
        +-- dreamzero/                     #   Code repo
```
