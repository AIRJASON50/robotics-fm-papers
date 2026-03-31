# PI (Physical Intelligence) Series -- 通用机器人 VLA 的演进

> **公司**: Physical Intelligence (PI), 2024 年成立, 旧金山
> **创始人**: Karol Hausman (CEO, 前 DeepMind), Sergey Levine (Berkeley), Chelsea Finn (Stanford)
> **融资**: $1.1B+ (Bezos, Alphabet CapitalG), 估值 $5.6B (2025.11)
> **代码**: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (Apache 2.0)

---

## 1. 演进脉络

```
=== 数据基础 ===
DROID (2024.03): 76K 轨迹, 350 小时, 564 场景
  跨机构数据集 (Stanford/Berkeley/CMU/Google/Toyota)
  PI 创始团队主导, pi_0 的训练数据核心之一

=== 基础模型 ===
pi_0 (2024.10): 第一个通用 VLA
  PaliGemma VLM + Flow Matching action expert
  7 个机器人平台, 68 个任务
  核心创新: flow matching 做连续动作生成 (不是 diffusion)

=== 效率优化 ===
FAST (2025.01): 高效动作 tokenizer
  DCT (离散余弦变换) 压缩动作序列
  让自回归 VLA 也能做高频灵巧任务
  pi_0-FAST = pi_0 + FAST tokenizer

=== 层级扩展 ===
Hi Robot (2025.02): 层级 VLA
  先用 VLM 做高层推理 → 再用低层执行
  支持多阶段指令和实时纠正
  (和 GR00T 的 VLM + DiT 分层思路一致)

=== 开放世界 ===
pi_0.5 (2025.04): 开放世界泛化
  异构任务共训练 (多机器人 + 语义预测 + web 数据)
  Knowledge Insulation: 微调时保护预训练知识不被覆盖
  开放世界: 见过/没见过的物体和场景都能泛化

=== RL 自我改进 ===
pi*0.6 (2025.11): 离线 RL 预训练
  RECAP: RL with Experience and Corrections via Advantage-conditioned Policies
  不只模仿 demo, 还从失败经验中学习
  离线 RL 做 VLA 预训练 → 比纯 BC 更鲁棒

=== 前沿 ===
MEM (2026.03): 多尺度记忆
  短期: 视频帧 (看到什么)
  长期: 文本摘要 (之前做了什么)
  支持 15 分钟以上长时任务

RLT (2026.03): RL Tokens
  给 VLA 加 RL token 输出
  轻量 actor-critic, 15 分钟学会精密操作
```

---

## 2. 核心技术演进

### pi_0 → pi_0.5: 从单任务到通用

| 维度 | pi_0 (2024.10) | pi_0.5 (2025.04) |
|------|---------------|-----------------|
| VLM | PaliGemma (3B) | PaliGemma (3B, 改进微调) |
| 动作生成 | Flow Matching | Flow Matching (同) |
| 训练数据 | 单一数据源 | **异构多源共训练** |
| 泛化 | 见过的任务/物体 | **开放世界** (新物体/场景) |
| 关键技术 | -- | **Knowledge Insulation** (保护预训练知识) |

**Knowledge Insulation** 解决了一个关键问题:
```
问题: VLM 在 robot 数据上 fine-tune → 忘掉了 web 预训练的知识 (灾难性遗忘)
解法: 隔离 VLM 的知识层, 微调时冻结部分权重
效果: 泛化到从未见过的物体 (用预训练的视觉知识识别)
```

### pi_0 → FAST → pi*0.6: 动作生成的三种范式

```
pi_0:      Flow Matching (连续值, 迭代去噪)
  优点: 处理多模态动作分布, 精度高
  缺点: 多步推理慢

FAST:      DCT Tokenization (离散 token, 自回归)
  将连续动作 → DCT 变换 → 量化为 token → LLM 自回归生成
  优点: 快 (一次前向), 可用 LLM 的 next-token prediction
  缺点: 量化损失

pi*0.6:    Flow Matching + RL (离线 RL 微调)
  在 pi_0 基础上加入 advantage-conditioned RL
  不只模仿好的 demo, 还从差的 demo 学 "什么不该做"
  优点: 比纯 BC 更鲁棒, 能自我改进
```

---

## 3. PI vs GR00T: 两种做法

| 维度 | PI (pi_Series) | NVIDIA (GR00T_Series) |
|------|---------------|----------------------|
| **定位** | 做模型 (产品公司) | 做平台 (基础设施公司) |
| **目标机器人** | 桌面操作臂为主 | **人形全身**为主 |
| **VLM** | PaliGemma (Google 开源) | 自研 Eagle → Cosmos |
| **动作生成** | Flow Matching | Flow Matching (同) |
| **低层控制** | 无 (直接关节角) | **SONIC** (motion tracking WBC) |
| **世界模型** | 无 (纯 VLA) | DreamGen → DreamZero (WAM) |
| **RL 集成** | pi\*0.6 (离线 RL) | PPO in sim (SONIC) |
| **数据策略** | DROID + 自采 | data pyramid + DreamGen 合成 |
| **开源** | openpi (Apache 2.0) | Isaac-GR00T (Apache 2.0) |
| **发展方向** | 记忆 (MEM) + 精密 RL (RLT) | 世界模型 (DreamZero/N2) |

**核心区别**: PI 沿着 VLA 路线纵深 (更好的 tokenizer → 更好的泛化 → RL 自改进 → 记忆)。NVIDIA 在 VLA 之外开辟了 WBC (SONIC) 和 WAM (DreamZero) 两条新路线。

---

## 4. Takeaway

| # | Takeaway | 对你的启示 |
|---|----------|-----------|
| 1 | **Flow Matching 是 VLA 动作生成的主流** | pi_0 和 GR00T 都选了它, 不是 DDPM diffusion |
| 2 | **Knowledge Insulation 防止灾难性遗忘** | fine-tune 机器人 policy 时要保护 VLM 预训练知识 |
| 3 | **离线 RL 可以改进 VLA (pi\*0.6)** | 不只看好 demo, 也从坏 demo 学, 和你的 RL 背景直接相关 |
| 4 | **DCT tokenizer (FAST) 是动作压缩的新思路** | 比 256-bin 离散化更高效, 保留频域信息 |
| 5 | **异构数据共训练是泛化的关键 (pi_0.5)** | 多机器人 + web 数据 + 语义预测联合训练 |
| 6 | **记忆机制 (MEM) 解锁长时任务** | 15 分钟以上的任务需要显式记忆, 纯 context window 不够 |

---

## 5. 文件索引

```
pi_Series/
├── pi_family_notes.md              ← 本文件
├── 24_DROID/ → (symlink)           # 数据集, 实体在 policy_learning/24_DROID/
├── 24_pi0/                         # pi_0 论文 + notes + openpi 代码 + pi-data-sharing
├── 25_FAST/                        # FAST tokenizer 论文
├── 25_HiRobot/                     # Hi Robot 层级 VLA 论文
├── 25_pi05/                        # pi_0.5 开放世界泛化论文
├── 25_pi06/                        # pi*0.6 离线 RL 论文
└── 26_MEM/                         # MEM 多尺度记忆论文
```
