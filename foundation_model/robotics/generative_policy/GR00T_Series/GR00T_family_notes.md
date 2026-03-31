# GR00T Family -- NVIDIA 人形机器人基础模型生态

> **阅读目的**: 不是复现, 而是学习其中的**设计思想和训练策略**。
> **核心问题**: 怎么设计一个能 scale 的人形机器人控制系统?

## 覆盖项目

| 版本/组件 | 论文 | arxiv | 时间 | 代码 |
|----------|------|-------|------|------|
| **GR00T N1** | An Open Foundation Model for Generalist Humanoid Robots | 2503.14734 | 2025.03 | Isaac-GR00T `n1-release` |
| **GR00T N1.5** | (blog + HuggingFace model card, 无 arxiv) | -- | 2025.12 | Isaac-GR00T `release` |
| **GR00T N1.6** | (blog, 无 arxiv) | -- | 2026.03 | Isaac-GR00T `main` |
| **SONIC** | Supersizing Motion Tracking for Natural Humanoid WBC | 2511.07820 | 2025.11 | GR00T-WholeBodyControl |
| **DreamGen** | Unlocking Generalization through Video World Models | 2505.12705 | 2025.05 | GR00T-Dreams |
| **DreamZero** | World Action Models are Zero-shot Policies (N2 核心) | 2602.15922 | 2026.02 | dreamzero |

---

## 1. 先回答核心问题: SONIC 是 SOTA 吗?

**在人形全身运动控制这个子问题上, SONIC 可以认为是当前 SOTA**, 但需要限定范围:

| 维度 | SONIC 的位置 | 限制 |
|------|-------------|------|
| **运动追踪** (motion tracking) | SOTA -- 100M 帧, 42M 参数, 真机 100% 成功率 | 追踪精度依赖动捕数据质量 |
| **全身协调** (walking + manipulation) | 接近 SOTA -- 自然人类动作先验 | 末端精度不如 Decoupled WBC (抖动) |
| **零样本泛化** (unseen motions) | SOTA -- 显著超越 AnyTrack/BeyondMimic/GMT | 仅在 Unitree G1 上验证 |
| **Scaling** | 首次证明人形控制的 scaling 有效 | 42M 参数远小于 LLM, scaling 上限未知 |
| **操作精度** (dexterous manipulation) | 不是 -- SONIC 是全身, 不做灵巧手 | 灵巧操作需要别的方案 |
| **通用机器人** (多 embodiment) | 不是 -- 这是 N1/N1.5/N1.6 的工作 | SONIC 只做全身控制层 |

**精确的说法**: SONIC 是 "humanoid whole-body motion tracking at scale" 的 SOTA。它不是通用机器人的 SOTA (那是 GR00T N1.x + SONIC 的组合), 也不是灵巧操作的 SOTA (那是 pi_0 或你的 bh_motion_track 方向)。

---

## 2. GR00T 的核心设计思想 (Takeaway)

### Takeaway 1: 分层解耦比端到端更实用

```
GR00T 不是一个模型, 而是一个分层系统:

Layer 3: VLM -- "理解任务" (10Hz)
  输入: 语言 + 图像
  输出: latent task representation
  训练: web-scale 视觉-语言预训练

Layer 2: Flow Matching DiT -- "规划动作" (50Hz)
  输入: task representation + 本体感觉
  输出: 目标运动轨迹 (action chunks)
  训练: robot demonstration data

Layer 1: SONIC Tracker -- "执行运动" (50Hz policy, 500Hz PD)
  输入: 目标运动轨迹
  输出: 关节角度
  训练: 100M 帧动捕数据 + RL (PPO)
```

**为什么不端到端?** 因为每一层有完全不同的数据源和训练方式:
- VLM 用互联网数据训练 (TB 级文本+图像)
- DiT 用 robot demo 训练 (千小时级遥操作)
- SONIC 用人类动捕训练 (百万帧级 MoCap)

端到端需要一个模型同时吃三种数据, 优化冲突严重。分层后各自独立优化, 再组合。

**对你的启示**: 你的 bh_motion_track 目前是 Layer 1 (运动追踪)。如果要做完整系统, 不需要从零训 VLM -- 直接接 GR00T N1.5 的 VLM 或其他开源 VLM 做 Layer 3。

### Takeaway 2: Motion Tracking 是人形机器人的 "next-token prediction"

SONIC 找到了人形控制的**统一可扩展目标**:

```
LLM 的统一目标:  next-token prediction
  → 一个目标, 所有能力 (对话/翻译/代码/推理) 自然涌现
  → 数据越多越好, 不需要为每个任务设计目标

SONIC 的统一目标: motion tracking
  → 一个目标, 所有运动 (走/跑/跳/舞/搬) 自然涌现
  → 动捕数据越多越好, 不需要为每个动作设计奖励
```

之前的方法 (PHC, H2O, BeyondMimic) 也做 motion tracking, 但没有 scale 到 SONIC 的量级。SONIC 证明了: **和 LLM 一样, scale 本身就能涌现能力** -- 100M 帧训练后, 模型可以零样本追踪从未见过的运动。

**对你的启示**: 你的 ~200 条轨迹在 SONIC 的 scaling 曲线最左端。按他们的数据, 增长到数千条就会有显著提升, 到百万帧级别可能涌现泛化。

### Takeaway 3: Universal Token Space 是跨身体迁移的关键

```
问题: 人类 SMPL 22 关节 ≠ 机器人 29 DOF
传统: 手工写运动重定向 (骨骼映射, IK 约束)

SONIC:
  Human motion → Human Encoder → FSQ quantize → token z_h
  Robot motion → Robot Encoder → FSQ quantize → token z_r

  训练时: L_token = ||z_h - z_r||^2  (强制对齐)
  推理时: 人类动捕 → z_h → Robot Decoder → 关节角

  结果: 隐式学会了运动重定向, 不需要骨骼映射
```

FSQ (Finite Scalar Quantization, 有限标量量化) 比 VQ-VAE 更稳定 -- 不需要 codebook, 直接在每个维度上做均匀量化。

**为什么这个设计重要?** 它意味着:
- 任何新动捕数据 (视频提取/VR 录制/motion generation) 都能直接用, 不需要重新写重定向
- 换一个机器人 (从 G1 到 H1), 只需要重新训练 Robot Encoder/Decoder, 其他不变
- GR00T VLA 的输出可以直接作为 SONIC 的输入 (通过 token space 对接)

### Takeaway 4: 数据 > 模型 > 计算 (人形控制的 Scaling 优先级)

SONIC 的 scaling 实验给出了明确排序:

```
数据放大 (0.4M → 100M 帧):  提升最大, 且未饱和
模型放大 (1.2M → 42M 参数):  提升显著, 但边际递减
计算放大 (8 → 128 GPU):      提升存在, 但主要影响训练动力学
```

**和 LLM 的 Scaling Laws 对比**:
- Kaplan (2020): 优先放大模型 → 被 Chinchilla 修正为数据=模型
- SONIC (2025): 对人形控制, **数据优先** → 因为 42M 参数的策略网络已经足够表达, 瓶颈在数据多样性

**对你的启示**: 在你的项目中, 扩充动捕数据的投入产出比远高于调大网络或增加 GPU。

### Takeaway 5: Sim2Real 的关键不是 sim 有多准, 而是 domain randomization 有多全

SONIC 真机 100% 成功率 (50 条多样化运动), 零样本迁移。靠的不是精确仿真, 而是**充分的域随机化** (Domain Randomization, DR):

```
关键 DR 参数:
  地面摩擦: 0.3 ~ 1.6 (4x 范围)
  关节位零点: 随机扰动 (模拟校准误差)
  基座重心: 随机偏移 (模拟负载变化)
  外力推: 随机速度/角速度扰动
  目标运动: 随机扰动 (增强鲁棒性)

+ 自适应采样: 失败率高的运动被更频繁采样
  p_i = 0.1 * cap(failure_rate_i) + 0.9/N
```

**对你的启示**: 你的 DRCfg 已定义但未实现。SONIC 证明 DR 是 sim2real 的 #1 因素 -- 没有 DR 的策略在真机上会直接失败。

---

## 3. N1 → N1.5 → N1.6 的架构修正

之前笔记有错误, 根据 HuggingFace model card 修正:

### N1.5 实际架构 (HF model card 确认)

```
视觉: SigLip2 (Vision Transformer)      ← 不是 "Eagle VLM" (Eagle 是系统名)
文本: T5 (Text Encoder)
本体: MLP (按 embodiment ID 索引, 处理不同 DOF)
动作: Flow Matching Transformer (DiT + AdaLN)  ← 不是 DDPM diffusion

引用的三篇论文:
  1. Eagle 2 (2501.14818) -- VLM 系统
  2. Rectified Flow (2209.03003) -- 动作生成范式
  3. pi_0 (2410.24164) -- 架构参考
```

**关键修正**: GR00T N1.5 的动作生成是 **Flow Matching** (和 pi_0 同源), 不是 DDPM 式 diffusion。之前笔记中 "Flow Matching (pi_0) vs Diffusion (GR00T)" 的对比是**错误的** -- 两者都用 flow matching。

### 版本对比 (修正版)

| 维度 | N1 (2025.03) | N1.5 (2025.12) | N1.6 (2026.03) |
|------|-------------|----------------|----------------|
| 视觉编码 | SigLIP | **SigLip2** | Cosmos-Reason-2B 内部变体 |
| 文本编码 | Qwen-2.5-1.5B | **T5** | Cosmos-Reason-2B 内部变体 |
| 系统名 | -- | Eagle (SigLip2+T5 的组合名) | Cosmos-Reason |
| VLM 参数 | 2B | 2.1B | ~3B |
| VLM 训练 | 冻结 | **冻结** + MLP adapter | **解冻顶部 4 层** |
| 动作生成 | Flow Matching DiT (16层) | Flow Matching DiT (16层) + 4层 adapter | Flow Matching DiT (**32层**) |
| 新增训练目标 | -- | **FLARE** (future latent alignment) | FLARE + 更多数据 |
| 动作表示 | 绝对位置 | 绝对位置 | **state-relative** |
| 训练 | -- | 250K steps, 1K H100, bs=16384 | 300K steps, bs=16384 |

### N1.5 的核心创新: FLARE

```
问题: robot demo 数据太少, 能不能从人类 ego-video 学?
难点: 人类视频没有 robot action labels

FLARE (Future LAtent Representation Alignment):
  不预测未来帧 (太难), 而是对齐 "未来帧的 latent embedding"

  训练时:
    当前帧 → VLM → embedding_now
    未来帧 → VLM → embedding_future (target, 不回传梯度)
    loss = align(model_prediction, embedding_future)

  效果: 模型学会了 "当前观察 → 未来会变成什么样"
        这个能力对 robot 有用: 预测动作后果, 即使没有 action label
```

FLARE 让 N1.5 的 language following 从 46.6% → **93.3%** (2x), 新物体泛化从 0% → **55%**。

---

## 4. SONIC 的训练设计精要

### 4.1 为什么 Motion Tracking 可以 Scale

```
对比:
  Locomotion RL: 奖励 = 前进速度 + 能量 + 姿态 (人工设计, 每个行为不同)
  Motion Tracking: 奖励 = ||当前关节 - 目标关节||^2 (一个公式, 所有行为通用)

Motion tracking 的奖励是 "数据驱动" 的:
  走路的 "奖励" 来自走路的动捕数据
  跳舞的 "奖励" 来自跳舞的动捕数据
  不需要人为定义 "什么是好的走路" -- 人类的动捕本身就是定义

这和 LLM 的 next-token prediction 完全一样:
  "什么是好的回答" 由训练数据定义, 不是由 reward function 定义
```

### 4.2 训练配方

```
环境: Isaac Lab (GPU 加速 MuJoCo)
算法: PPO (你的 foundations/17_PPO)
并行: 128 GPU, HuggingFace Accelerate + TRL
策略网络: MLP, 42M 参数 (不是 Transformer, 因为策略频率要 50Hz)
数据: 100M+ 帧, 700 小时人类动捕 (50Hz)

关键超参:
  策略频率: 50Hz (每 20ms 一次决策)
  PD 执行频率: 500Hz (每 2ms 一次力矩)
  训练时间: 128 GPU × 3-7 天 = 9000 GPU-hours
```

### 4.3 自适应采样 (最值得借鉴)

```
问题: 100M 帧中, 大部分是 "走路" (容易), 少部分是 "翻滚" (难)
      均匀采样 → 策略在难动作上性能差

解法:
  把动作按类型分成 bins
  统计每个 bin 的失败率 f_i
  采样概率: p_i = 0.1 * cap(f_i) + 0.9/N

  失败率高的 bin 被更频繁采样 (10% 权重)
  但保留 90% 均匀采样防止遗忘简单动作
```

### 4.4 Universal Token Space 的训练损失

```
四项损失联合优化:

L_total = L_PPO + L_recon + L_token + L_cycle

L_PPO:   标准 PPO RL 损失 (追踪奖励)
L_recon: Robot Decoder(z_r) ≈ 真实关节 AND Robot Decoder(z_h) ≈ 真实关节
         → z_h (人类 token) 通过 robot decoder 也能重建关节 = 隐式重定向
L_token: ||z_r - z_h||^2
         → 同一个动作的人类 token 和机器人 token 应该一样
L_cycle: Robot Encoder(Robot Decoder(z_h)) ≈ z_r
         → 人类→机器人→再编回来应该等于机器人原始 token
```

---

## 5. DreamGen → DreamZero: 范式转换

### DreamGen: 世界模型做数据增强

```
角色: VLA 训练的辅助工具, 不替换 VLA

少量真实 demo (11h)
  → Cosmos 世界模型 → 变换背景/光照/物体
  → 生成 6500h 等效合成数据
  → 用合成数据训练 GR00T N1.x

本质: 数据飞轮的工程实现
```

### DreamZero: 世界模型 = 策略

```
角色: 替换 VLA, 下一代架构 (GR00T N2 核心)

不再有单独的 VLM 和 Action Head
14B 视频扩散模型同时输出:
  - 未来 N 帧画面 (世界预测)
  - 对应的动作序列 (策略)

训练数据: internet video (无 action label) + robot demo (有 action label)
推理: text prompt → 想象 video → 提取 action, 7Hz 闭环

关键声称: 泛化能力 >2x VLA
```

**从 VLA 到 WAM 的思想转变**:
- VLA: 感知当前 → 输出动作 (类比: 背答案)
- WAM: 想象未来 → 从想象中推导动作 (类比: 理解原理后推导)

---

## 6. 五个核心 Takeaway

| # | Takeaway | 原理 | 对你的启示 |
|---|----------|------|-----------|
| 1 | **分层解耦 > 端到端** | VLM/DiT/SONIC 各自用最适合的数据独立训练 | 你做 Layer 1 (追踪), 不需要同时做 VLM |
| 2 | **Motion Tracking = 人形的 next-token prediction** | 统一目标 + scale 数据 = 涌现能力 | 不要为每个动作设计奖励, 统一用追踪 |
| 3 | **数据 > 模型 > 计算** | 42M 参数够了, 瓶颈是动捕数据多样性 | 优先扩充数据, 不是调大网络 |
| 4 | **Universal Token Space 解决跨身体迁移** | FSQ 对齐人和机器人的 latent space | 可替代你的手工运动重定向 |
| 5 | **Sim2Real 靠 DR 而不是 sim 精度** | 充分的域随机化 + 自适应采样 | 你的 DRCfg 需要实现并充分随机化 |

---

## 7. 文件索引

| 路径 | 内容 |
|------|------|
| `vla_wbc/25_N1/` | N1 论文 + 笔记 |
| `vla_wbc/25_N15/` | N1.5 blog report (含 HF model card 修正) |
| `vla_wbc/25_SONIC/` | SONIC 论文 + 笔记 (含 bh_motion_track 对比) |
| `vla_wbc/26_N16/` | N1.6 blog report |
| `vla_wbc/Isaac-GR00T/` | 代码仓库 (n1/release/main 多分支) |
| `world_model/25_DreamGen/` | DreamGen 论文 + GR00T-Dreams 仓库 |
| `world_model/26_DreamZero/` | DreamZero 论文 + dreamzero 仓库 |

> **SONIC 代码**: 完整版在 `humanoid/25_SONIC/GR00T-WBC/`, 含 GEAR-SONIC + Decoupled WBC。
