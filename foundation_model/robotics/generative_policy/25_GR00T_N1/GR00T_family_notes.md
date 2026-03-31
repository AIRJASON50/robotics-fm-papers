# GR00T Family -- NVIDIA 人形机器人基础模型生态

## 覆盖项目

| 版本/组件 | 论文 | arxiv | 时间 | 代码仓库 |
|----------|------|-------|------|---------|
| **GR00T N1** | An Open Foundation Model for Generalist Humanoid Robots | 2503.14734 | 2025.03 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (`n1-release`) |
| **GR00T N1.5** | (blog only, 无独立论文) | -- | 2025.12 | Isaac-GR00T (`n1.5-release`) |
| **GR00T N1.6** | (blog only, 无独立论文) | -- | 2026.03 GTC | Isaac-GR00T (`main`) |
| **GR00T N1.7** | (未公开) | -- | 2026.03 GTC | Early access + 商业授权 |
| **SONIC** | Supersizing Motion Tracking for Natural Humanoid WBC | 2511.07820 | 2025.11 | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) |
| **DreamGen** | Unlocking Generalization through Video World Models | 2505.12705 | 2025.05 | [NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) |
| **DreamZero (N2 核心)** | World Action Models are Zero-shot Policies | 2602.15922 | 2026.02 | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) |

---

## 1. 家族演进脉络

```
=== Phase 1: VLA 架构 (2025) ===

GR00T N1 (2025.03): 第一版, VLM + DiT
  |  VLM: SigLIP + Qwen-2.5-1.5B (2B total)
  |  DiT: 16 层, 动作去噪生成
  |  双系统: VLM 10Hz (理解) + DiT 120Hz (执行)
  |
DreamGen (2025.05): 数据增强
  |  用 Cosmos 世界模型生成合成轨迹
  |  解决 robot 数据不足问题
  |
SONIC (2025.11): 全身控制
  |  42M 参数, 100M 帧动捕数据, 128 GPU
  |  Universal Token Space: 人类动作 → 机器人动作
  |  零样本 sim2real, 真机 100% 成功率
  |
GR00T N1.5 (2025.12): VLM 升级
  |  VLM: Eagle VLM (更强)
  |  DiT: 16 层 + 4 层 adapter
  |  SONIC 集成: VLA + WBC 完整 pipeline
  |
GR00T N1.6 (2026.03 GTC): DiT 翻倍
  |  VLM: Cosmos-Reason-2B (原生宽高比)
  |  DiT: 32 层 (2x N1.5)
  |  去掉 adapter, 解冻 VLM 顶层 4 层
  |  新增: 双臂 YAM + AGIBot + Galaxea R1 + G1 locomotion-manipulation

=== Phase 2: WAM 架构转型 (2026) ===

DreamZero (2026.02): 范式转换
  |  从 VLA (VLM + Action Head) → WAM (World Action Model)
  |  14B 视频扩散模型, 同时预测未来世界状态 + 动作
  |  不再是 "看图→出动作", 而是 "想象未来→提取动作"
  |  泛化能力 >2x VLA
  v
GR00T N2 (2026 预告): 基于 DreamZero
  |  WAM 替代 VLA, 彻底改变架构
  |  预计 2026 年底
```

---

## 2. 架构对比

### N1 → N1.5 → N1.6 (VLA 架构内的演进)

| 维度 | N1 (2025.03) | N1.5 (2025.12) | N1.6 (2026.03) |
|------|-------------|----------------|----------------|
| VLM | SigLIP + Qwen-2.5-1.5B | Eagle VLM | **Cosmos-Reason-2B** |
| VLM 参数 | 2B | 3B | 3B |
| DiT 层数 | 16 | 16 + 4 adapter | **32** (2x) |
| VLM 微调 | 冻结 | 冻结 + adapter | **解冻顶部 4 层** |
| 图像处理 | resize | resize | **原生宽高比** |
| 动作表示 | 绝对位置 | 绝对位置 | **state-relative** |
| WBC 集成 | 无 | SONIC 集成 | SONIC + Decoupled WBC |

### VLA (N1 系列) vs WAM (N2/DreamZero)

```
VLA (N1 系列):
  观察图像 → VLM 理解 → DiT 生成动作
  "看到什么 → 做什么"
  限制: 只基于当前帧决策, 不预测后果

WAM (N2 / DreamZero):
  观察图像 → 视频扩散模型 → 同时预测未来画面 + 动作
  "想象未来会怎样 → 从想象中提取该做什么"
  优势: 隐式学会了物理预测, 泛化更好
```

这是从 **"反应式控制"** 到 **"想象式控制"** 的范式转换:
- VLA = System 1 (看到就反应, 快但不深)
- WAM = System 2 (先想象后行动, 慢但泛化)

---

## 3. SONIC 在家族中的角色

SONIC 不是独立产品, 而是 GR00T 的 **低层运动执行器**:

```
完整 GR00T pipeline (N1.5+):

  语言: "拿苹果放盘子"
    │
    ▼
  GR00T VLM (10Hz) ── "什么任务, 手该去哪"
    │
    ▼
  SONIC Planner (100ms) ── "生成运动片段 (0.8-2.4s)"
    │
    ▼
  SONIC Tracker (50Hz) ── "追踪运动, 输出关节角"
    │
    ▼
  PD Controller (500Hz) ── 硬件执行
    │
    ▼
  Unitree G1 机器人
```

SONIC 的 Universal Token Space 让 GR00T VLM 的高层输出 (latent plan) 可以直接映射为 SONIC 理解的运动命令, 无需中间转换。

---

## 4. DreamGen → DreamZero: 世界模型路线

**DreamGen (2025.05)**: 用世界模型做**数据增强**
```
少量真实 demo → Cosmos 世界模型 → 生成大量合成 demo
  改变背景、光照、物体位置 → 数据多样性提升
  本质: 世界模型服务于 VLA 的训练, 辅助角色
```

**DreamZero (2026.02)**: 世界模型**就是** policy
```
不再需要单独的 VLA
视频扩散模型同时输出:
  - 未来 N 帧画面 (世界预测)
  - 对应的动作序列 (策略)

14B 参数, 7Hz 实时闭环
训练: internet video + robot demo
推理: text prompt → 想象 video → 提取 action
```

**对 robotics 的范式意义**:
- VLA (N1): 感知 → 动作 (跳过了"理解物理")
- WAM (N2): 感知 → **想象未来** → 动作 (隐式学会了物理)
- 这就是 LeCun 一直在推的 JEPA/World Model 思想的工程实现

---

## 5. 与知识库其他内容的关联

| 组件 | 技术来源 (你库里的) |
|------|-------------------|
| VLM backbone | `CV/2_vl_alignment/24_PaliGemma` (N1) → Eagle → Cosmos-Reason (N1.6) |
| DiT action head | `CV/1_generation/23_DiT` |
| Flow Matching (pi_0) vs Diffusion (GR00T) | `CV/1_generation/22_FlowMatching` vs `CV/1_generation/20_DDPM` |
| SONIC 的 PPO 训练 | `foundations/17_PPO` |
| SONIC 的 Motion Tracking 思路 | `humanoid/23_PHC` (PHC 也做 motion tracking, 但规模小得多) |
| DreamGen 的世界模型 | `robotics/world_model/23_DreamerV3` (概念一致, 但 DreamGen 用视频扩散) |
| DreamZero 的 WAM | `robotics/world_model/23_UniSim` (UniSim 是 "video generation = simulation" 的先驱) |
| Universal Token Space (FSQ) | 概念类似 `CV/1_generation/14_VAE` 的 VQ-VAE 路线 |
| 数据飞轮 | DreamGen: 11h demo → 世界模型 → 大量合成数据 |

---

## 6. 文件索引

| 文件 | 内容 |
|------|------|
| `GR00T_N1_An_Open_Foundation_Model_...md` | N1 原论文 (2503.14734) |
| `GR00T_N1_notes.md` | N1 分析笔记 |
| `SONIC_Supersizing_Motion_Tracking_...md` | SONIC 原论文 (2511.07820) |
| `SONIC_GR00T_WBC_notes.md` | SONIC 分析笔记 (含与 bh_motion_track 对比) |
| `DreamGen_Unlocking_Generalization_...md` | DreamGen 原论文 (2505.12705) |
| `World_Action_Models_are_Zero-shot_Policies.md` | DreamZero/N2 核心论文 (2602.15922) |
| `Isaac-GR00T/` | 代码仓库 (N1/N1.5/N1.6, 含多分支) |
| `GR00T_family_notes.md` | 本文件, 家族总览 |

> **代码仓库说明**: GR00T-WholeBodyControl (SONIC) 代码仍在 `humanoid/25_SONIC/GR00T-WBC/`, 因为它直接服务于你的 humanoid 项目。GR00T-Dreams 仓库未 clone (按需拉取)。
