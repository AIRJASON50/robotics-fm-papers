# PI (Physical Intelligence) Series -- 从 VLA 到自主改进的机器人基础模型

> **目的**: 理解 PI 做通用机器人 VLA 的完整思路——每一步解决了什么问题，为什么要那样做，以及对你的启示。

---

## 1. PI 的战略定位

PI 是一家纯模型公司 (不造硬件不做仿真), 目标是做**机器人的 foundation model**, 让任何机器人硬件都能用 PI 的模型开箱即用。

```
PI 的产品逻辑:
  1. 训练一个通用 VLA (视觉+语言→动作)
  2. 开源推理代码 (openpi), 让社区在自己的机器人上 fine-tune
  3. 通过 API/服务收费 (商业模式)

对比 NVIDIA:
  NVIDIA: 造平台 (仿真+训练+部署+芯片), 模型只是平台的一部分
  PI: 只做模型, 极致纵深 — 一条 VLA 路线走到底
```

**创始人阵容** (理解论文思路的关键):
- **Sergey Levine** (Berkeley): RL for Robotics 开创者, 定义了 offline RL/BC 的理论框架 → 解释了为什么 PI 后期走 RL 路线 (pi\*0.6)
- **Chelsea Finn** (Stanford): MAML 元学习发明者 → 解释了为什么 PI 强调 few-shot 泛化
- **Karol Hausman** (前 DeepMind SayCan): LLM-for-robot planning 先驱 → 解释了为什么 PI 用 VLM 做高层理解

PI 的论文本质上是这三个人的研究方向的工程化整合。

---

## 2. 完整演进: 每一步解决了什么问题

### Phase 1: 建立基础 -- pi_0 (2024.10)

**要解决的问题**: 能不能做一个通用的机器人 policy, 一个模型控制多种机器人做多种任务?

**解法**: VLM (PaliGemma, 3B) + Flow Matching Action Expert

```
架构:
  图像 + 语言 → PaliGemma VLM → vision-language embeddings
                                         │
                                   cross-attention
                                         │
  本体感觉 + 噪声动作 → Flow Matching Transformer → 去噪 → 动作

关键设计:
  1. Action Expert 和 VLM 分离 (独立参数)
     → 动作生成不干扰语言理解
  2. Flow Matching 而非 Diffusion
     → ODE 比 SDE 更快, 推理步数更少
  3. Action Chunking (生成一段动作而非单步)
     → 减少 compounding error, 来自 ACT 的思想
```

**训练**: 7 个机器人平台, 68 个任务, 10K 小时数据
**结果**: 首次证明单一 VLA 可以跨平台工作 (从单臂到双臂到灵巧手)

**Takeaway 1**: **VLM + Flow Matching 是当前 VLA 的最佳架构组合。** GR00T N1.5 后来也选了同样的组合 (SigLip2+T5 + Flow Matching DiT), 说明这不是 PI 的偶然选择而是行业共识。

**关于 Flow Matching vs Diffusion**: Flow Matching 用直线 ODE 路径 (5-10 步), 比 Diffusion 的弯曲 SDE 路径 (50-100 步) 更快更稳定, 数学也更简单。PI 和 GR00T 独立选择了同一方案, 已成行业共识。

### Phase 2: 提速 -- FAST (2025.01)

**要解决的问题**: Flow Matching 需要多步迭代推理, 对高频灵巧任务太慢。能不能用 LLM 的 autoregressive 方式一次前向就出动作?

**难点**: 动作是连续值, LLM 只能生成离散 token。之前的方案 (RT-2 的 256-bin) 精度太低。

**解法**: DCT (Discrete Cosine Transform, 离散余弦变换) 做动作压缩

```
传统离散化 (RT-2):
  连续动作 → 256 bin 量化 → token
  问题: 每个自由度独立量化, 丢失了关节间的协调信息

FAST (DCT):
  连续动作序列 → DCT 变换 → 只保留低频分量 → 量化为 token

  为什么 DCT 有效:
    机器人动作是平滑的 (不会突变), 能量集中在低频
    DCT 把时间序列变换到频域, 低频 = 主要运动, 高频 = 噪声
    只编码低频分量 → 大幅压缩 (从几百维到几十个 token)
    且保留了关节间的时间协调性 (全局变换, 不是逐点量化)
```

**结果**: pi_0-FAST 推理速度提升数倍, 且在灵巧任务上精度不降

**Takeaway 2**: **动作压缩的关键不是量化精度, 而是选对变换域。** 时域量化 (RT-2) 丢信息, 频域量化 (FAST/DCT) 利用了动作的物理先验 (平滑性)。这个思路可以迁移到任何需要 tokenize 连续信号的场景。

### Phase 3: 层级化 -- Hi Robot (2025.02)

**要解决的问题**: pi_0 是扁平的 "观察→动作", 无法处理复杂多步指令 (如 "先清理桌子, 然后把碗放进洗碗机")。

**解法**: 两层 VLA

```
Layer 2 (高层): VLM 推理
  输入: 图像 + 复杂指令
  输出: 子目标序列 + 进度判断
  "清理桌子" → ["拿起碗", "放进洗碗机", "拿起杯子", ...]

Layer 1 (低层): pi_0 执行
  输入: 图像 + 当前子目标
  输出: 动作
  "拿起碗" → 具体关节运动
```

**和 GR00T 分层的区别**:
- GR00T: VLM (语义) + DiT (动作) + SONIC (关节) → 三层, 按**频率**分
- Hi Robot: VLM (规划) + VLA (执行) → 两层, 按**抽象层次**分
- GR00T 的分层是工程驱动 (频率匹配), Hi Robot 的分层是认知驱动 (任务分解)

**Takeaway 3**: **复杂任务需要层级分解, 但分层的维度有两种选择: 频率 (GR00T) vs 抽象层次 (Hi Robot)。** 两者不矛盾, 可以组合 (高层 Hi Robot 规划 + 中层 pi_0 动作 + 低层 SONIC 关节)。

### Phase 4: 泛化 -- pi_0.5 (2025.04)

**要解决的问题**: pi_0 只能在训练时见过的物体/场景上工作。换一个新杯子就可能失败。

**核心发现**: 灾难性遗忘 (Catastrophic Forgetting) 是泛化的最大敌人

```
问题:
  PaliGemma 预训练时: 见过数十亿张图, 认识几乎所有物体
  在 robot 数据上 fine-tune 后: 忘掉了大部分视觉知识
  结果: 只认得训练时见过的几种物体

Knowledge Insulation (知识隔离):
  不冻结全部 VLM (太保守, 不适应 robot), 也不解冻全部 (灾难性遗忘)
  选择性冻结: 保护存储通用视觉知识的层, 解冻需要适应 robot 的层

  具体做法:
    VLM 底层 (低级视觉特征): 冻结 → 保护 "什么是杯子" 的知识
    VLM 顶层 (任务相关推理): 解冻 → 适应 "看到杯子该怎么抓"
```

**另一个关键**: 异构数据共训练
```
不只用 robot demo, 还混入:
  - Web 数据 (保持 VLM 的通用知识)
  - 语义预测任务 (预测 "接下来会发生什么")
  - 多种机器人数据 (cross-embodiment 共享)

混合训练防止模型 "缩进" 到单一 embodiment
```

**结果**: 新物体泛化率从 ~0% 到有意义的成功率

**Takeaway 4**: **fine-tune robot policy 时, 保护预训练知识是第一要务。** 这和 LLM 的 alignment tax (InstructGPT 的 PPO-ptx) 是同一个问题——怎么让模型学新东西而不忘旧的。Knowledge Insulation 是 robot 领域的解法, 类似 LLM 的 LoRA (冻结大部分, 只调小部分)。

### Phase 5: 自我改进 -- pi\*0.6 (2025.11)

**要解决的问题**: BC (Behavior Cloning, 行为克隆) 只能模仿好的 demo。坏的 demo 不能用, 失败的尝试被浪费了。能不能从失败中学习?

**这是 PI 创始人的学术基因的集中体现**: Sergey Levine 的核心研究方向就是 offline RL。

```
BC (之前所有 VLA 的做法):
  训练数据: 只有成功的 demo
  学习信号: "模仿这个动作"
  问题: demo 之外的情况不知道该怎么办

RECAP (pi*0.6 的做法):
  训练数据: 成功的 + 失败的 + 人类纠正的
  学习信号: advantage-conditioned — "这个动作比平均好多少/差多少"

  核心思想:
    好 demo 学 "该怎么做", 坏 demo 学 "不该怎么做" = 数据利用率翻倍
    不需要 online RL (不需要在真机上试错)
    用离线数据 (之前收集的所有轨迹) 就能做 = offline RL 预训练
```

**Takeaway 5**: **离线 RL 是 VLA 的天然下一步。** BC 只用成功 demo, offline RL 用所有数据。对你做 RL 的背景来说, pi\*0.6 的 RECAP 是把 RL 思想嫁接到 VLA 的最直接方案。不需要 online 试错, 不需要 reward engineering, 只需要给已有轨迹标 advantage。

### Phase 6: 前沿 -- MEM + RLT (2026.03)

**MEM (Multi-scale Embodied Memory)**: 解决长时任务

```
问题: VLA 的 context window 只有几秒到几十秒
     "打扫整个房间" 需要 15+ 分钟, 中间有几十个子任务

解法: 显式记忆机制
  短期记忆: 最近几帧视频 (看到什么)
  长期记忆: 文本摘要 (之前做了什么, 还剩什么)

  每隔一段时间: VLM 把短期视频总结为文本 → 存入长期记忆
  决策时: 读取长期记忆 + 当前短期视频 → 输出动作
```

**RLT (RL Tokens)**: 解决精密操作

```
问题: BC/VLA 对精密操作不够精确 (插USB, 拧螺丝)
     精密操作需要 closed-loop 微调, 不是 open-loop 预测

解法: 在 VLA 的输出 token 中加入 RL token
  VLA 输出: [action tokens] + [RL tokens]
  RL tokens: 被一个轻量 actor-critic 读取, 做 15 分钟的在线 RL

  15 分钟在线 RL → 学会精密操作 (VLA 做粗定位, RL 做精调)
```

**Takeaway 6**: **PI 的发展路线回答了 "BC 之后怎么办"**:
1. BC 不够准 → 加 RL (pi\*0.6 离线, RLT 在线)
2. BC 不够泛化 → 加 Knowledge Insulation (pi_0.5)
3. BC 不够快 → 换 tokenizer (FAST)
4. BC 不够长 → 加记忆 (MEM)

这条路线的终点是: **VLA + offline RL + online RL + memory = 完整的 robot agent。**

---

## 3. PI vs GR00T: 两种完全不同的哲学

| | PI 的思路 | NVIDIA 的思路 |
|---|---|---|
| **核心信念** | 一个足够好的 VLA 能解决一切 | 不同问题需要不同模块 |
| **架构** | 单一 VLA (观察→动作, 端到端) | 分层系统 (VLM + DiT + SONIC + PD) |
| **对低层控制的态度** | 不需要 (VLA 直接出关节角) | **必须** (SONIC 做运动追踪) |
| **对世界模型的态度** | 不需要 (VLA 隐式学物理) | **下一代核心** (DreamZero/WAM) |
| **改进方向** | 纵深: tokenizer→泛化→RL→记忆 | 横向: VLA + WBC + WorldModel |
| **目标机器人** | 操作臂 (7-DOF, 桌面任务) | **人形全身** (29+ DOF) |
| **Sim2Real** | 几乎不用仿真 (直接真机 demo) | **重度依赖仿真** (Isaac Sim) |
| **数据策略** | DROID + 自采真实数据 | data pyramid (sim > synthetic > real) |
| **对 RL 的态度** | 后期加入 (pi\*0.6 离线 RL) | 从一开始就用 (SONIC 全程 PPO) |

**两种哲学的根本分歧: 需不需要低层控制器?**

```
PI 的观点:
  VLA 直接输出关节角 → PD 控制器执行
  "模型足够强就不需要中间层"
  适用于: 桌面操作臂 (动作简单, 不需要平衡)

NVIDIA 的观点:
  VLA 输出目标轨迹 → SONIC 追踪 → PD 执行
  "低层控制必须专门处理"
  适用于: 人形全身 (需要平衡, 动作复杂)

这不是对错, 是问题不同:
  PI 做的: 单臂抓放 → 7 DOF, 不需要平衡, VLA 直接出关节角足够
  NVIDIA 做的: 人形走+抓 → 29 DOF, 必须平衡, VLA 直接出关节角会摔
```

**对你的启示**: 你做人形机器人, NVIDIA 的分层方案更适合你。但 PI 的 RL 集成 (pi\*0.6) 和 Knowledge Insulation (pi_0.5) 是通用技巧, 可以用在任何架构上。

---

## 4. PI 的开源策略

PI 开源了推理栈 (openpi 代码 + HuggingFace 权重 + fine-tune 脚本 + 数据工具), 但没开源训练栈 (预训练代码、训练数据、RECAP RL 代码、FAST tokenizer 训练、Knowledge Insulation 实现)。对比 NVIDIA 开源了完整训练栈 (Isaac-GR00T + GR00T-WBC)。**结论: PI 论文学思想, 复现靠 NVIDIA。**

---

## 5. 核心 Takeaway (按可执行性排序)

| # | Takeaway | 原理 | 对你的行动项 |
|---|----------|------|------------|
| 1 | **Flow Matching 是 VLA 动作生成的行业共识** | PI 和 NVIDIA 独立选择了同一方案, 比 diffusion 更快更稳 | 你的 CV 知识库已有 `22_FlowMatching`, 优先学这个 |
| 2 | **VLM + Action Expert 分离是正确设计** | 动作生成不应干扰语言理解, 用 cross-attention 连接 | 如果你做 VLA, 不要把动作 head 和 VLM 共享参数 |
| 3 | **Knowledge Insulation 防止灾难性遗忘** | 底层冻结 (保护通用视觉) + 顶层解冻 (适应 robot) | fine-tune 任何预训练模型时都应考虑这个 |
| 4 | **离线 RL 是 BC 的天然下一步 (pi\*0.6)** | 好 demo 学 "该怎么做", 坏 demo 学 "不该怎么做" | 你的失败轨迹不要扔, 标 advantage 后能用于 RL |
| 5 | **DCT tokenizer 利用了动作的物理先验** | 机器人动作是平滑的 → 频域压缩比时域量化更高效 | 如果你要 tokenize 动作, 考虑频域方法 |
| 6 | **层级 VLA 处理复杂多步任务 (Hi Robot)** | 高层规划 + 低层执行, 和 GR00T 分层互补 | 长 horizon 任务需要分层, 不要试图端到端 |
| 7 | **异构数据共训练 + web 数据保泛化 (pi_0.5)** | 只用 robot 数据 → 过拟合; 混入 web 数据 → 保持通用性 | 训练时混入非 robot 数据 (视频, 图文) |
| 8 | **显式记忆解锁长时任务 (MEM)** | 短期视频 + 长期文本摘要 = 多尺度记忆 | 超过 1 分钟的任务需要记忆, context window 不够 |
| 9 | **PI 不做低层控制, 但你应该做 (如果做人形)** | 桌面臂可以 VLA→关节角; 人形必须有 WBC 层 | SONIC 的方案对你更实用 |
| 10 | **PI 走 VLA 纵深, 但 NVIDIA 正在用 WAM 替代 VLA** | WAM (World-Action Model): 想象→做, 替代 VLA 的看→做 | 关注 GR00T 的 DreamZero/WAM 范式, 这是下一代潜在方向 |

---

## 6. 文件索引

```
pi_Series/
├── pi_family_notes.md              <- 本文件
├── 24_DROID/ → symlink             # 数据集 (实体在 policy_learning/24_DROID/)
├── 24_pi0/                         # pi_0 论文 + notes
│   ├── openpi/                     #   推理代码 (github.com/Physical-Intelligence/openpi)
│   └── pi-data-sharing/            #   数据工具
├── 25_FAST/                        # FAST DCT tokenizer 论文
├── 25_HiRobot/                     # Hi Robot 层级 VLA 论文
├── 25_pi05/                        # pi_0.5 开放世界泛化论文
├── 25_pi06/                        # pi*0.6 离线 RL (RECAP) 论文
└── 26_MEM/                         # MEM 多尺度记忆论文
```
