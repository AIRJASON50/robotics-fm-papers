# GR00T Family -- NVIDIA 人形机器人基础模型全景

> **目的**: 理解 NVIDIA 做人形机器人的完整思路——不是看单篇论文，而是看**一整套系统怎么从零搭起来的**。

---

## 1. NVIDIA 的人形机器人战略

NVIDIA 不造机器人。它的策略是做**机器人的操作系统**——提供从训练到部署的全栈软件，让硬件厂商 (Unitree G1, Fourier GR-1, AGIBot, Galaxea) 都用 NVIDIA 的方案。

```
NVIDIA 机器人全栈:
  Isaac Sim (仿真环境)
  → Isaac Lab (RL 训练框架)
  → GR00T (基础模型: VLA + WBC)
  → Jetson Orin (边缘部署硬件)
  → Cosmos (世界模型)

GR00T 是这个全栈中的 "模型层"。
```

---

## 2. 两条独立脉络：Isaac-GR00T 和 SONIC

GR00T 家族包含**两个独立发展的控制器**，最终被组合在一起：

### Isaac-GR00T 脉络 (VLA: 看→理解→规划)

**团队**: NVIDIA Research, GEAR lab (Jim Fan, Yuke Zhu 领导)
**定位**: 高层决策——理解语言指令和视觉场景，输出目标动作轨迹

```
2025.03  N1: 第一版 VLA
  架构: SigLIP + Qwen-2.5-1.5B (VLM, 2B) + Flow Matching DiT (16层)
  贡献: 双系统架构 (VLM 10Hz + DiT 50Hz)
        data pyramid (web video > sim > real)
        cross-embodiment (一套权重多种机器人)
  限制: 语言跟随率仅 46.6%, 零样本泛化差

2025.12  N1.5: VLM 大升级
  架构: SigLip2 + T5 (Eagle 系统, 2.1B) + Flow Matching DiT (16层 + 4层adapter)
  新增: FLARE 训练目标 (从人类 ego-video 学习, 不需 action label)
  效果: 语言跟随率 46.6% → 93.3% (2x)
        真机成功率 43.3% → 83.0% (2x)
        新物体泛化 0% → 55% (FLARE)
  训练: 250K steps, 1K H100, batch 16384

2026.03  N1.6: DiT 翻倍
  架构: Cosmos-Reason-2B (VLM, 解冻顶层4层) + Flow Matching DiT (32层)
  改进: 去掉 adapter → 解冻 VLM 层 (更直接的梯度流)
        原生宽高比图像 (不再 resize/pad)
        state-relative action (更好的 sim2real)
  新 embodiment: 双臂 YAM, AGIBot, Galaxea R1, G1 locomotion
```

**关键设计决策演进**:

| 问题 | N1 怎么做 | N1.5/1.6 怎么改 | 为什么改 |
|------|----------|----------------|---------|
| VLM 怎么接 DiT | 冻结 VLM + 直接 cross-attention | N1.5 加 adapter / N1.6 解冻顶层 | 冻结 VLM 表达力不够 |
| 数据不够怎么办 | data pyramid (sim + real) | N1.5 加 FLARE (可用人类视频) | 人类视频无限且免费 |
| 动作怎么表示 | 绝对关节角 | N1.6 用 state-relative | 绝对值对 sim2real gap 敏感 |
| 图像怎么编码 | resize 到 224x224 | N1.6 原生宽高比 | resize 丢信息 |

### SONIC 脉络 (WBC: 运动追踪→关节控制)

**团队**: NVIDIA Research (Zhengyi Luo 骆政一 等, PHC 作者)
**定位**: 低层执行——追踪目标运动轨迹，输出关节角度

```
背景线: PHC (2023) → SONIC (2025)
  PHC (ICCV 2023): 骆政一的前作, 单人小规模 motion tracking
  → BeyondMimic (2025): 扩展到全身, 但数据和模型仍小
  → SONIC (2025.11): 超大规模 motion tracking

SONIC 的核心数字:
  42M 参数 (MLP, 不是 Transformer — 因为需要 50Hz 实时)
  100M+ 帧动捕数据 (700 小时, 50Hz)
  128 GPU × 3-7 天 = 9000 GPU-hours
  真机: 50 条多样化轨迹, 100% 成功率, 零样本迁移
```

**SONIC 的三个核心创新**:

**(1) Motion Tracking 作为统一可扩展目标**
```
之前: 每个动作 (走/跑/跳/搬) 设计单独的奖励函数 → 不可扩展
SONIC: 所有动作统一为 "追踪动捕数据" → 一个奖励公式, 所有动作通用
  reward = ||当前关节位置 - 目标关节位置||^2
  不同动作的差异来自不同的动捕数据, 不来自不同的奖励设计
```

**(2) Universal Token Space (跨身体迁移)**
```
人类 SMPL 22 关节 ≠ 机器人 29 DOF
  Human motion → Human Encoder → FSQ → token z_h
  Robot motion → Robot Encoder → FSQ → token z_r
  训练: 强制 z_h ≈ z_r
  推理: 人类动捕 → z_h → Robot Decoder → 关节角
  = 隐式运动重定向, 无需手工骨骼映射
```

**(3) Scaling 三轴验证**
```
数据: 0.4M → 100M 帧 → 提升最大, 未饱和
模型: 1.2M → 42M 参数 → 提升显著
计算: 8 → 128 GPU → 影响渐近性能 (不只是速度)

对人形控制: 数据 > 模型 > 计算 (和 LLM 的 Chinchilla 结论一致)
```

---

## 3. 两者如何组合

N1.5 开始, Isaac-GR00T 和 SONIC 被串联部署:

```
语言: "拿苹果放盘子"                      频率     模型
    │
    ▼
Isaac-GR00T VLM (Eagle/SigLip2+T5)       10 Hz    2-3B
    "理解任务: 需要移动右手到苹果位置"
    │
    ▼
Isaac-GR00T DiT (Flow Matching)           50 Hz    16-32层 DiT
    "生成目标运动轨迹 (action chunks)"
    │
    │ 目标轨迹 (SMPL 格式 or latent token)
    ▼
SONIC Planner (Transformer/Conv1D)         100ms    轻量
    "规划 0.8-2.4s 运动片段"
    │
    ▼
SONIC Tracker (MLP + PPO)                  50 Hz    42M
    "追踪运动 → 输出 29 DOF 关节角"
    │
    ▼
PD Controller                              500 Hz   硬件层
    │
    ▼
Unitree G1 真实机器人
```

**为什么分开而不是端到端?**

每一层用完全不同的数据训练:
- VLM: TB 级互联网文本+图像 (不需要机器人)
- DiT: 千小时级遥操作 demo (需要机器人)
- SONIC: 百万帧级人类动捕 (不需要机器人)

端到端 = 一个模型同时吃三种数据, 优化冲突。分层 = 各自用最佳数据独立优化。

**但这不是唯一方案**: Decoupled WBC (同在 GR00T-WBC 代码库) 是另一个思路:
- 下半身: RL (50Hz) 负责步态/平衡
- 上半身: IK (100Hz) 负责末端精确控制
- 优势: 末端精度高 (搬水杯不洒), 劣势: 上下身协调不自然

---

## 4. World Model 路线 (DreamGen → DreamZero)

与 VLA+WBC 并行, NVIDIA 还在探索世界模型路线:

### DreamGen (2025.05): 世界模型做数据增强

```
角色: 辅助工具, 服务于 VLA 训练
方法: 少量真实 demo → Cosmos 视频世界模型 → 变换背景/光照/物体 → 合成大量 demo
效果: 11h 真实 demo → 6500h 等效合成数据 (590x 放大)
意义: 数据飞轮的工程实现
```

### DreamZero (2026.02): 世界模型 = 策略 (GR00T N2 核心)

```
角色: 替换 VLA, 下一代架构
核心思想: 不再是 "看到→做", 而是 "想象未来→从想象中提取动作"

VLA (N1.x):  观察当前帧 → 直接输出动作 (不预测后果)
WAM (N2):    观察当前帧 → 视频扩散模型想象未来 N 帧 → 同时提取动作

14B 参数, 7Hz 实时闭环
训练数据: internet video (无 action label) + robot demo (有 action label)
声称: 泛化能力 >2x VLA
```

**范式意义**:
```
VLA = 背答案 (见过这个场景→输出记住的动作)
WAM = 理解原理 (想象动作后果→选择好的动作)

VLA 在训练分布内强, WAM 在分布外泛化强
这和 LLM 的 o1 思路一致: 推理时花更多计算 "想一想" → 泛化更好
```

---

## 5. NVIDIA 人形机器人的完整时间线

```
=== 基础设施期 (2022-2024) ===
2022    Isaac Sim + Isaac Gym (仿真环境)
2023    Isaac Lab (RL 训练框架, 替代 Isaac Gym)
2024    Cosmos (世界模型基础设施)
        Jetson Thor (机器人专用边缘芯片, 预告)

=== GR00T 第一代 (2025) ===
2025.03  GR00T N1 -- 首个开源人形 VLA
2025.05  DreamGen -- 世界模型做数据增强
2025.11  SONIC -- 大规模全身运动控制
2025.12  GR00T N1.5 -- VLM 升级 (Eagle) + FLARE
         N1.5 + SONIC 首次组合部署, 苹果放盘子 95% 成功率

=== GR00T 第二代 (2026) ===
2026.02  DreamZero -- WAM 架构 (N2 核心技术)
2026.03  GR00T N1.6 -- DiT 翻倍 (32层), Cosmos VLM
         GR00T N1.7 -- Early access (加灵巧操作)
2026 H2  GR00T N2 (预告) -- WAM 替代 VLA

=== 合作硬件厂商 ===
Unitree G1       -- SONIC 主要测试平台
Fourier GR-1     -- N1/N1.5 主要测试平台
AGIBot Genie-1   -- N1.6 新增
Galaxea R1 Pro   -- N1.6 新增 (仿真)
Bimanual YAM     -- N1.6 新增
```

---

## 6. 核心 Takeaway

### 对你做人形机器人的直接启示

| # | Takeaway | 原理 | 对你的行动项 |
|---|----------|------|------------|
| 1 | **分层解耦 > 端到端** | 不同层用不同数据, 独立优化 | 你做 Layer 1 (追踪), 接开源 VLM 做 Layer 3 |
| 2 | **Motion Tracking = 可扩展的统一目标** | 一个追踪奖励搞定所有动作 | 不要为每个动作设计奖励 |
| 3 | **数据 > 模型 > 计算** | 42M MLP 够了, 瓶颈是动捕多样性 | 优先扩充数据, 不是调大网络 |
| 4 | **Universal Token Space** | FSQ 对齐人和机器人的 latent | 可替代手工运动重定向 |
| 5 | **Sim2Real 靠 DR 不靠 sim 精度** | 充分域随机化 = 零样本迁移 | 实现你的 DRCfg |
| 6 | **FLARE: 从人类视频学习** | 对齐未来 latent, 不需 action label | 人类视频是免费的无限数据源 |
| 7 | **WAM 可能是下一代架构** | 想象未来 > 直接映射动作 | 关注 DreamZero 的后续发展 |

### SONIC 是否是人形控制的 SOTA?

**在 "humanoid whole-body motion tracking" 这个子问题上, 是。** 但需要限定:
- 是 motion tracking SOTA, 不是通用机器人 SOTA
- 全身协调强, 但末端精度弱 (灵巧操作不如 pi_0/你的方向)
- 只在 Unitree G1 验证, 跨 embodiment 泛化未测试
- 42M MLP 的 scaling 上限未知 (LLM 到 1T, SONIC 才 42M)

---

## 7. 文件索引

```
GR00T_Series/
├── GR00T_family_notes.md                  ← 本文件
├── vla_wbc/
│   ├── Isaac-GR00T/                       # 大脑 (VLA)
│   │   ├── code/                          #   NVIDIA/Isaac-GR00T 仓库
│   │   ├── 25_N1/                         #   N1 论文 + notes
│   │   ├── 25_N15/                        #   N1.5 blog report
│   │   └── 26_N16/                        #   N1.6 blog report
│   └── SONIC/                             # 小脑 (WBC)
│       ├── code/                          #   NVlabs/GR00T-WholeBodyControl 仓库
│       ├── SONIC_...md                    #   论文
│       └── SONIC_notes.md                 #   笔记 (含 bh_motion_track 对比)
└── world_model/
    ├── 25_DreamGen/                       # 数据增强 (Cosmos 世界模型)
    │   └── GR00T-Dreams/                  #   代码仓库
    └── 26_DreamZero/                      # WAM (N2 核心, 14B 视频扩散)
        └── dreamzero/                     #   代码仓库
```
