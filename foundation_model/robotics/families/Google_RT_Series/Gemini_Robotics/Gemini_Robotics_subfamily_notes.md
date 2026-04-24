# Gemini Robotics 子家族 -- Google DeepMind 在 RT 之后的 robotics 主线

> **位置**: Google_RT_Series 的延续, 接 RT-1/RT-2/PaLM-E/Open-X-Embodiment/AutoRT/RT-H 之后, 把 Google 的 robotics FM 路线整合进 Gemini 多模态主干。
> **核心区别 (vs RT 系列)**: 不再训练独立 VLM, 直接复用 Gemini frontier 模型的多模态能力 + tool use + reasoning, 加 robot action data 微调即成 VLA。

---

## 1. Gemini Robotics 路线一览 (按时间)

| 模型 | 发布 | 来源 | 角色 |
| --- | --- | --- | --- |
| **Gemini Robotics 1.0** | 2025.03 | arXiv:2503.20020 | 第一代 Gemini 基的 VLA + ER VLM (基于 Gemini 2.0 Flash) |
| **Gemini Robotics-ER 1.0** | 2025.03 | 同上 | 强化 embodied reasoning 的 Gemini 2.0 Flash 变体 |
| **Gemini Robotics 1.5** | 2025.10.03 (v3 2025.11.28) | arXiv:2510.03342 | **多 embodiment VLA + Motion Transfer + Embodied Thinking** |
| **Gemini Robotics-ER 1.5** | 2025.10 | 同上 | ER VLM 升级版, agentic orchestrator |
| **Gemini Robotics-ER 1.6** | 2026.04.15 | DeepMind blog | 加 instrument reading (93% vs ER 1.5 的 23%) |

---

## 2. Gemini Robotics 1.0 (2025.03) -- 第一代

**两个并列模型**:

### Gemini Robotics-ER (VLM)
- 基础: Gemini 2.0 Flash, 强化 embodied reasoning
- 能力 (新增 "embodied" 类):
  - **Object detection**: open-world 2D bounding box (explicit + implicit query)
  - **Pointing**: 自然语言描述 → 物体/物体部位/affordance/free space 的 2D 点
  - **Trajectory prediction**: 2D motion trajectory grounded in observation
  - **Grasp prediction**: 顶视抓取 (1.0 新引入)
  - **Multi-view correspondence**: 立体图之间预测 2D 对应点
  - **3D bounding box**: 单目预测 metric 3D box
- 部署模式: 不需要 fine-tune 直接做 zero-shot robot control 通过 code generation, 或 few-shot in-context learning

### Gemini Robotics (VLA)
- 把 ER 模型加 robot action 数据微调 → 输出连续动作
- 高频 dexterous control + 鲁棒泛化 + 快速适应

### ERQA 基准 (论文新建)
- **400 道多选 VQA**, 7 类: spatial reasoning, trajectory reasoning, action reasoning, state estimation, pointing, multi-view reasoning, task reasoning
- 28% 的题目含多张图 (跨图理解, 难度更大)
- Gemini 2.0 Flash + CoT > Gemini 2.0 Pro Experimental (无 CoT)
- 已开源: github.com/embodiedreasoning/ERQA

---

## 3. Gemini Robotics 1.5 (2025.10) -- 多 embodiment + Thinking VLA (核心升级)

### 三大创新

#### 创新 1: Motion Transfer (MT) — 跨 embodiment 学习的核心机制

**问题**: 不同机器人 (ALOHA / Bi-arm Franka / Apollo humanoid) 形态完全不同, 怎么用一个 checkpoint 控制三个?

**解法**: 新架构 + 新训练 recipe (论文称为 Motion Transfer)
- 让模型学到一个**统一的 motion 和 physical interaction 理解**, 而不是按 embodiment 分头训
- 训完一个 checkpoint **直接控制 ALOHA / Bi-arm Franka / Apollo humanoid 三种机器人**, 无需 per-embodiment post-training
- 还能实现**零样本 skill transfer** (一个机器人学到的技能直接迁移到另一个)

**Ablation 证据 (Fig. 4)**:
- 单 embodiment 训练 → 性能差
- 多 embodiment 训练但无 MT recipe → 比单 embodiment 好但不够
- 多 embodiment + MT → **三个机器人上全面最佳**

#### 创新 2: Embodied Thinking — Thinking VLA

**核心**: 在动作之前/之间穿插**多级自然语言推理** ("think before acting")

```
用户指令: "Pack the suitcase for a trip to London"
  ↓ (orchestrator GR-ER 1.5 上 thinking trace)
"访问行程表 + 天气预报 → 决定带什么衣服"
  ↓
"pack the rain jacket from the wardrobe" (高层指令给 action model)
  ↓ (action model GR 1.5 上 thinking trace)
"Next step: pick up rain jacket"
"Motion description: predicted low-level motion trajectory"
  ↓
具体关节运动
```

**两层 thinking**:
- **GR-ER 1.5 (orchestrator)**: 高层任务分解, 进度判断, 工具调用 (web search 等)
- **GR 1.5 (action model)**: 在每个 atomic action 前用自然语言写出"下一步该做什么"和运动描述

**好处**:
- 复杂 multi-step 任务分解能力大涨
- 失败检测 + 恢复行为
- 行为对人类**可解释** (能看到 thinking trace)
- 简化复杂指令 → primitive skill 序列

#### 创新 3: GR-ER 1.5 — 新一代 ER VLM (SOTA)

- 在多个 embodied intelligence benchmark 上 SOTA
- 保留 frontier 通用能力 (tool use, code, video, audio)
- **比 frontier 模型显著快**

### Agentic 系统架构

```
User input + environmental feedback
     |
     v
   Orchestrator (GR-ER 1.5)
   - 分解任务为子步骤
   - 成功检测, 决定何时切换
   - 调用外部工具 (search, code, function call)
   - 多模态交互 (video, audio)
     |
     | natural language instruction
     v
   Action Model (GR 1.5)
   - open-vocabulary 自然语言指令
   - 多 embodiment 控制 (ALOHA / Franka / Apollo)
   - 自带 thinking trace
     |
     v
   机器人动作
```

### 训练数据
- ALOHA + Bi-arm Franka + Apollo humanoid 上采的多 embodiment robot data (数千个任务, 多样场景)
- 公开互联网 text/image/video 数据
- 总规模未具体披露

### 评估
- **230 个任务 benchmark** (扩展自 GR 1.0 论文)
- 4 类泛化: visual / instruction / action / task
- A/B/n testing on real robots (在同一 work cell 上交错跑)
- **90%+ 评估在 MuJoCo 仿真里跑**, 与真机 rank consistency 经验证 (Appendix B.1)
- → 大幅减小真机评估成本, 加速架构迭代

### 关键结果 (Fig. 3, progress score)

| 任务类别 | ALOHA | Bi-arm Franka | Apollo |
| --- | --- | --- | --- |
| In-Distribution | 0.83 | 0.74 | 0.74 |
| Instruction Generalization | 0.76 | 0.73 | 0.62 |
| Action Generalization | 0.54 | 0.70 | 0.66 |
| Visual Generalization | 0.81 | 0.77 | 0.73 |
| Task Generalization | 0.70 | 0.50 | 0.63 |

GR 1.5 在三个机器人 4 类泛化上**全部超越** baseline (Gemini Robotics 1.0 + Gemini Robotics On-Device)。

---

## 4. Gemini Robotics-ER 1.6 (2026.04.15) -- 最新 ER 升级

**仅 ER 模型升级, VLA 部分仍是 GR 1.5**。

### 新能力 — Instrument Reading
- 与 Boston Dynamics 合作发现的 use case
- 读取 analog gauges, pressure meters, sight glasses
- **93% 准确率, vs ER 1.5 的 23%**, vs Gemini 3.0 Flash 也大幅提升
- 通过 agentic vision (而不是单 frame 直接读)

### 通用 ER 能力提升
- pointing, counting, success detection 等 spatial/physical reasoning 全面改进
- 在 Gemini API + Google AI Studio 已上线

### 迁移
- gemini-robotics-er-1.5-preview 将于 **2026.04.30 关闭**, 强制迁移到 1.6

---

## 5. 与 PI / GR00T 对比 (同期 robotics FM 三家)

| 维度 | PI (pi_0.7) | NVIDIA (GR00T N1.7) | **Google (GR 1.5 + ER 1.6)** |
| --- | --- | --- | --- |
| 总参数 | ~5B | 3B | 未披露 (依赖 Gemini frontier) |
| 架构 | 单一 VLA + flow matching | Action Cascade (VLM + DiT) | **Orchestrator (VLM) + Action Model (VLA), agentic** |
| Embodiment | 桌面/移动 manipulator | 人形全身 + 灵巧手 | ALOHA / Bi-arm Franka / Apollo humanoid (3 embodiment 同 ckpt) |
| 跨 embodiment 桥梁 | subgoal image (BAGEL 14B 生成) | Relative EEF delta + 人类 video co-train | **Motion Transfer training recipe (架构层面)** |
| 数据策略 | autonomous eval rollout + metadata 标签 | 20K h EgoScale 人类视频 (scaling law) | **多 embodiment robot data + Gemini 互联网知识 + 90% sim 评估** |
| Thinking / 推理 | metadata + CFG | 标准 VLA + LLM-style task decomposition | **Embodied Thinking (两层 thinking trace, 跨 orchestrator + VLA)** |
| 部署 | 云端 API | 完全开源 (Apache 2.0) | Gemini API (闭源, gauge ER 1.6 公开 API) |
| 闭源/开源 | 论文开源, 权重闭源 | 全开源 | **完全闭源** |

**三家定位**:
- PI: 押 prompt engineering + diverse data + compositional generalization
- NVIDIA: 押人类 video scaling + relative EEF + 全栈开源
- **Google**: 押 Gemini frontier 复用 + Embodied Thinking + agentic orchestration + sim 评估闭环

---

## 6. 对你的启示

1. **Embodied Thinking 是新范式**: GR 1.5 第一次把 LLM 的 chain-of-thought 完整搬到 robot action 层 (不只是在高层规划做 CoT, 连低层 action 之间也插 thinking)。pi_0.7 的 metadata + CFG 是另一种"间接 thinking"。两条路都值得关注。
2. **MuJoCo 仿真做 90%+ 评估**: 这个工程实践很值得借鉴 — 真机评估贵且慢, 仿真做 rank consistency 校准后, 大部分 ablation 在仿真里跑。你做 PPO sim2real 已经在用 MuJoCo, 这个评估闭环可以复用到 VLA 评估。
3. **多 embodiment 一个 checkpoint**: GR 1.5 在 ALOHA + Franka + Apollo 三个完全不同的形态上用同一个 checkpoint, MT recipe 是关键。这种思路对你将来跑多种灵巧手的统一控制有借鉴意义。
4. **Orchestrator + Action 解耦**: 类似 PI Hi Robot 的两层架构, 但 Google 把 orchestrator 做成 frontier VLM (ER 1.5/1.6) + tool use, 灵活性远高于 PI Hi Robot 的固定层级。

---

## 7. 文件索引

```
Gemini_Robotics/
├── Gemini_Robotics_subfamily_notes.md   <- 本文件
├── 25_GR1/                              # Gemini Robotics 1.0 (2025.03, arXiv:2503.20020)
│   └── Gemini_Robotics_Bringing_AI_into_the_Physical_World.md
└── 25_GR15/                             # Gemini Robotics 1.5 (2025.10, arXiv:2510.03342)
    └── Gemini_Robotics_15.pdf
```

ER 1.6 没有独立 paper, 仅 DeepMind blog: https://deepmind.google/blog/gemini-robotics-er-1-6/
