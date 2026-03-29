# Twisting Lids Off with Two Hands - 论文笔记

**论文**: Twisting Lids Off with Two Hands
**作者**: Toru Lin, Zhao-Heng Yin, Haozhi Qi, Pieter Abbeel, Jitendra Malik
**机构**: UC Berkeley
**发表**: CoRL 2024 (arXiv:2403.02338v2)
**代码**: HORA PPO 的 fork，但架构已显著分叉

---

## 从 HORA 系列到 TwistingLids: 阅读引导

读完 HORA → PenSpin → DexScrew 之后，你已经知道：
1. 关节级 sim-to-real gap 其实不大 (PenSpin 开环 47%, DexScrew 直接迁移 41.6%)
2. 直接 sim-to-real 在 PenSpin/DexScrew 上**不行** → 需要三阶段 pipeline
3. 蒸馏方法由任务容错性决定

**TwistingLids 是反方向的对比**: 双手拧瓶盖的直接 sim-to-real **成功了** (零样本 946°)，不需要 RMA，不需要蒸馏，不需要真实数据微调。为什么？

核心答案：
- **物理可仿真**: Brake Link 提供了比关节摩擦参数更可靠的旋转阻力建模
- **双手持握降低不确定性**: 一只手稳定瓶身 = 大幅减少对物体属性估计的需求
- **极简观测**: 只用 2 个 3D 质心点 + 不看关节速度 + 不看物体旋转 → 观测空间极小 → sim-to-real gap 被压缩
- **但开环完全失败** (128°): 和 PenSpin 的"开环 47% 成功"形成直接对比

---

## 与 HORA 系列的架构对比

### 代码来源: 同源但已分叉

TwistingLids 的 PPO 和 HORA 是同一套代码 fork 而来，核心 GAE + clipped PPO 逻辑完全一致，但有两个关键分叉：

| 维度 | HORA 系列 | TwistingLids |
|------|-----------|-------------|
| Actor-Critic | **共享 trunk** (一个 MLP 同时输出 mu 和 value) | **完全分离** (独立 actor_mlp + value_mlp) |
| 特权信息处理 | env_mlp 压缩 → concat 到 obs → 共享 trunk | **非对称 AC**: actor 看 obs, critic 看完整特权状态 |
| 适应模块 | ProprioAdaptTConv (Stage 2 用) | **无** — 不需要在线适应 |
| 设计目的 | 两阶段: Stage 1 训练 + Stage 2 RMA 适配 | 单阶段: 直接 sim-to-real |

**为什么不需要 RMA?** HORA 系列需要 RMA 是因为单手操控中物体属性（摩擦、质量）对策略影响太大，需要在线估计。双手拧盖中，一只手做稳定持握本身就抵消了大量不确定性——不需要精确估计物体属性就能成功。

### PPO 实现差异

| 维度 | HORA | TwistingLids |
|------|------|-------------|
| 代码量 | 396 行 | **750 行** (增加了 multi-GPU, video, WandB) |
| Reward 全局缩放 | `0.01` | `0.001` |
| 多 GPU | 无 | `torch.distributed` + `all_reduce` |
| 帧堆叠 | 3 帧滑窗 | **2 帧堆叠** (`n_stack_frame=2`) |

---

## 方法

### 物理建模 — Brake Link (核心创新)

Isaac Gym 中直接调旋转关节的摩擦参数不够真实（静摩擦行为不稳定）。解决方案是**用物理接触摩擦间接产生旋转阻力**：

```
URDF 结构:
  link2 (瓶身) ─── revolute joint (b_joint, 零摩擦) ──→ link1 (瓶盖)
       │
       └─── prismatic joint (brake_joint) ──→ brake (小碰撞块 0.02x0.02x0.005)
```

- `b_joint`: 连接瓶身和瓶盖的旋转关节，**摩擦设为 0**
- `brake_joint`: prismatic 关节，运行时施加恒定向下力 (`torque = -0.3`)
- brake 体被压向瓶盖 → 二者之间的**物理接触摩擦**产生旋转阻力
- 真实世界用 3D 打印复制了相同的三体结构

**为什么比直接调关节摩擦好**: 关节摩擦参数在 PhysX 中是解析模型，对静摩擦的建模不够准确。Brake Link 利用的是碰撞引擎本身的接触摩擦模型（PhysX 在这方面更可靠），是一种"迂回"的仿真工程。

### 感知 — 极简到极致

```
Actor 能看到的:
  dof_pos_scaled (32D, 带噪声)       ← 关节角度
  prev_targets (32D)                   ← 上一步目标
  cube_base_pos (3D)                   ← 瓶身质心
  cube_cap_pos (3D)                    ← 瓶盖质心
  × 2 帧堆叠 = ~140D

Actor 看不到的 (代码中显式置零):
  dof_vel (no_dof_vel=True)            ← 关节速度
  cube_quat (no_obj_quat=True)         ← 物体旋转四元数
  object_id (no_obj_id=True)           ← 哪个瓶子

Critic 能看到的 (非对称, 训练时特权):
  dof_pos, dof_vel (无噪声)
  cube_pos, cube_quat
  left/right 指尖位置 (4×3 = 24D)
  base markers (16×3 = 48D) + cap markers (8×3 = 24D)
  physics params (scale, friction, mass)
```

**2 个点就够**: 直觉上拧盖应该需要精确的物体姿态。实验证明瓶身和瓶盖的 3D 质心坐标就足够——SAM 分割 + XMem 跟踪 → mask 中心 + 深度 → 3D 坐标。10 Hz 感知频率匹配控制频率。

### 奖励设计

| 奖励项 | 公式 | scale | 备注 |
|--------|------|-------|------|
| **Rotation** | `clamp(dof_pos[t] - dof_pos[t-1], -0.02, 0.02)` | 500.0 | 关节角位移差分，不是角速度 |
| **Finger Contact (base)** | `sum(0.1 / (dist*2 + 0.03))` per finger, clamp max=2.0 | 2.5 | 左手 4 指到瓶身 base markers 最近距离 |
| **Finger Contact (cap)** | 非拇指用 sum, 拇指用 mean | 2.5 | 右手到瓶盖 cap markers |
| **Pose (z-axis align)** | `-clamp(arccos(z_axis · [0,1,0]), 0, 1.0)` | 20.0 | 瓶子主轴对齐 y 方向 |
| **Action penalty** | `-sum(action²)` | 0.001 | **左手×3, 右手×1** (不对称!) |
| **Work penalty** | `-|torque · vel|` | 1.0 | **左手×3, 右手×1** |
| **Failure** | -50.0 when z < 0.4 | 1.0 | 掉落 |
| **角度早停** | reset when step>100 AND angle_diff>0.2 | — | |

**关键细节 — 左右手不对称惩罚**: 论文完全没提及。左手（持握手）的 action 和 work 惩罚是右手（旋转手）的 3 倍——引导左手保持稳定持握（少动），右手积极旋转。

**Finger Contact Reward 的实际形式**: 论文描述为"负指数"，代码实际用的是**倒数** `0.1/(d*2+0.03)`，在 dist<0.01m 时饱和。Marker 体是 URDF 中定义的微小碰撞体 (0.01³)，沿瓶身/瓶盖表面均匀分布 (cap 8 个, base 16 个)。

### 训练细节

- PPO + 非对称 Actor-Critic
- Actor: 论文说 [256, 256, 128]，**代码默认 [512, 512, 512]** — 可能 "Large Network" 消融就是代码默认配置
- Critic: [512, 512, 512] + 特权观测
- 动作 EMA 平滑 (系数 0.75): `actions = actions * 0.75 + last_actions * 0.25`
- 控制频率: sim 60Hz / controlFrequencyInv=6 = **10 Hz** (匹配真实 Allegro Hand 控制频率)
- **手部无重力** (`disable_gravity=True`): 论文未提及但代码中明确设置——消除了重力建模误差

---

## 关键结果

| 方法 | Blue Bottle (avg deg) |
|------|----------------------|
| Open-loop Replay | 128° (TTF=7.67s) |
| No-Vision | 1.3° |
| No-Asymmetric | 0.6° |
| Large Network | 2.0° (仿真同等但无法迁移) |
| **完整方法** | **946°** (30s 内约 4 圈) |

- 泛化到 10 种家用瓶子 (训练中未见)
- 仅 1 圈即可开盖: 60% 成功率; 需 5 圈: 10% 成功率
- 策略能抵抗外力扰动并恢复

---

## 为什么直接 sim-to-real 在这里可行

| 因素 | TwistingLids (直接迁移成功) | PenSpin (直接迁移失败) |
|------|---------------------------|----------------------|
| 接触模式 | 双手持握 = 被动稳定 | 单手 finger gaiting = 主动动态 |
| 物理建模 | Brake Link 准确建模旋转阻力 | 笔的接触几何在仿真中误差大 |
| 观测简化 | 2 个 3D 点 + 不看 dof_vel/obj_quat | 需要点云 + 触觉 + 指尖位置 |
| 隐含技巧 | 手部无重力 + 大量噪声注入 (dof: 0.4, action: 0.2) | 噪声相对小 (dof: 0.01, action: 0.005) |
| 适应需求 | 一只手持握抵消物体不确定性 → 不需要在线估计物理参数 | 需要 RMA 推断物体属性 |

**总结**: TwistingLids 通过 (1) 好的物理建模 (Brake Link), (2) 极简观测缩小 gap, (3) 大量噪声注入做隐式域随机化, (4) 双手结构降低对精确估计的需求, 实现了无需 RMA/蒸馏的直接迁移。

## 为什么开环失败 (128°) — 对比 PenSpin

PenSpin 开环 ~47% 成功 → TwistingLids 开环只有 128° (约 1/3 圈)。完全相反的表现。

| | PenSpin 开环 | TwistingLids 开环 |
|--|-------------|------------------|
| 成功率 | ~47% | 128° (几乎立刻失败) |
| 原因 | 笔在手指间靠重力部分维持 | 瓶子靠两手摩擦力悬空，无闭环补偿立刻滑落 |
| 启示 | 开环可以碰运气采数据 | 开环不可行，**闭环视觉反馈是必要条件** |

**深层规律**: 开环 vs 闭环的优劣取决于任务的**被动稳定性**。PenSpin 的笔有重力辅助（部分支撑），开环能维持。TwistingLids 的瓶子完全靠双手摩擦力悬空，任何偏差都会被重力放大——必须闭环修正。

这和 HandelBot 的"闭环比开环差"又不同——HandelBot 是精度需求超过了 dynamics gap 的闭环精度。三者构成完整的图景：

```
PenSpin:      被动稳定 + gap 小 → 开环可行
TwistingLids: 被动不稳定 → 闭环必须
HandelBot:    gap > 闭环精度 → 开环反而好
```

---

## 为什么大网络 sim-to-real 失败

论文的 Large Network 在仿真中性能等同但真实世界只有 2°（Blue Bottle）。

**代码揭示**: 默认 config (`DualURBottlePPO.yaml`) 中 actor/critic 都是 [512,512,512]——这可能就是 "Large" 配置。成功的配置是论文中说的 [256,256,128] actor。

**原因**: 接触密集任务中，大网络有足够容量来记住仿真器**特有的**接触动力学细节（接触点位置分布、法向量角度、PhysX 的碰撞检测精度等），而这些细节在真实世界不存在。小网络被迫忽略这些高频细节，只学到更鲁棒的低频控制策略。

**类比**: 大网络像一个把仿真考卷答案全背下来的学生——仿真考试满分但换了真题就不会。小网络只记住了解题方法（低频策略），对题目变化更鲁棒。

这和 approximation error 的关系：大网络 approximation error 更小（在仿真分布内），但**泛化误差更大**（到真实分布时）。不是越拟合越好——是 overfitting to sim dynamics。

---

## 非显而易见的洞察

1. **2 个 3D 关键点就够了**: 最小化感知输入反而有利于 sim-to-real——观测空间越小，sim-to-real gap 的维度越少
2. **多物体训练比单物体更好 (甚至在单物体评估上)**: ~60 种瓶子自然形成从易到难的课程
3. **大网络 = 过拟合仿真**: 仿真等效但 sim-to-real 完全失败——网络容量对迁移极敏感
4. **开环完全失败 (128°)**: 和 PenSpin 的 47% 形成对比——被动稳定性决定开环可行性
5. **左右手不对称惩罚**: 论文未提及但代码中明确——左手 3 倍惩罚引导稳定持握
6. **手部无重力**: 论文未提及但代码中 `disable_gravity=True`——消除了一个 sim-to-real gap 来源
7. **训练"无限旋转"零样本泛化到"有限旋转+开盖"**: 策略从未见过瓶盖脱离瓶身

---

## 作者展望

1. 扩展到更多旋转关节类任务 (魔方、灯泡、罐头)
2. 更鲁棒的感知系统替代 SAM+XMem
3. 引入手臂运动自由度
4. 提高多圈物体开盖成功率 (当前 10-50%)

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| Actor 网络 | [256, 256, 128] | **默认 [512, 512, 512]** — 可能 "Large" 就是默认配置 |
| Rotation reward | "角速度" | **dof_pos 差分** (关节角位移增量) |
| Finger contact | "距离的负指数" | **倒数** `0.1/(dist*2+0.03)`, clamp 2.0 |
| 左右手 | 未提及 | **不对称**: 左手 action/work 惩罚 ×3 |
| dof_vel | 未提及 | `no_dof_vel=True` → 策略看不到关节速度 |
| obj_quat | 未提及 | `no_obj_quat=True` → 策略看不到物体旋转 |
| 外力扰动 | "Random Force Scale: 2.0" | **`force_scale=0.0`** → 默认训练不用 |
| 关节噪声 | "N(0, 0.1)" | **`dofpos_noise_scale=0.4`** → 4 倍于论文 |
| 动作噪声 | "N(0, 0.1)" | **`action_noise_scale=0.2`** → 2 倍 |
| 手部重力 | 未提及 | **`disable_gravity=True`** |
| Reward 缩放 | 未提及 | `reward_scale_value: 0.001` |

**注意**: 论文和代码默认配置有多处不一致（网络大小、噪声量级、外力扰动）。实际训练的配置可能通过 CLI override 实现，但公开代码中未提供对应的训练脚本。

---

## 值得学习的代码设计

1. **Brake Link**: prismatic joint + 碰撞体靠物理接触摩擦传递力矩 — 比参数调节更稳定的仿真工程
2. **Marker-based 距离计算**: URDF 中放置微小碰撞体作为参考点 (cap 8 个, base 16 个) — 奖励设计的物理化
3. **模块化环境**: Initializer/Rewarder/Randomizer 通过 `importlib.import_module` 动态加载 — 比 HORA 的单文件环境更利于消融实验
4. **非对称 AC (独立网络)**: 相比 HORA 共享 trunk，更符合现代实践 — actor 和 critic 的优化不冲突
5. **AssetManager**: 自动扫描 URDF 目录 → 每个 env 随机选一个瓶子 → 多物体训练的基础设施
6. **大量隐含的 sim-to-real 设计**: 手部无重力、极大噪声注入、EMA 平滑、10Hz 低频控制 — 这些不是论文的 "方法"，但对成功至关重要

---

## 局限

- 仅限瓶状物体，方形瓶泛化有限 (43° vs 圆形 946°)
- 手臂固定 (UR5e DOF_MODE_POS 常量目标)
- 10 Hz 控制频率
- 感知系统对严重遮挡脆弱 (鲁棒性测试改用 marker-based 检测)
- 论文和代码默认配置不一致——可复现性存疑
