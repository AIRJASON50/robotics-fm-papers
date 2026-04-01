# PHC: Perpetual Humanoid Control for Real-time Simulated Avatars

> ICCV 2023 | arxiv 2305.06456
> Authors: Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, Weipeng Xu
> Code: https://github.com/ZhengyiLuo/PHC
> 日期: 2026-03-12

---

## 1. 核心问题

Physics-based humanoid control 面临两个长期未解问题:

1. **Scalability (可扩展性)**: 如何让一个 policy 学习上万条动捕序列而不发生 catastrophic forgetting -- 以往方法在序列数量增加时，成功率快速下降
2. **Perpetual Control (永续控制)**: 真实应用中（视频驱动、语言指令），输入信号带噪声且 humanoid 可能跌倒。以往方法依赖 external forces 或 reset 来处理失败，无法实现"永不停歇"的连续控制

**核心 Insight**: 将大规模运动模仿分解为多个渐进式学习阶段，每个阶段专注于更难的子集；同时学习"失败状态恢复"能力，使得 controller 可以自主起身并继续跟踪。

---

## 2. 方法概览

### 2.1 整体 Pipeline

```
AMASS (10K+ motion clips)
    |
    v
Progressive Multiplicative Control Policy (PMCP)
    |
    +---> Primitive 0: learn "easy" motions (all data)
    +---> Primitive 1: learn "hard" motions (failed subset of Prim 0)
    +---> Primitive 2: learn "harder" motions (failed subset of Prim 1)
    +---> Primitive N: fail-state recovery (getup from ground)
    |
    v
MCP Composer (Mixture of Control Primitives)
    |  Softmax weights over primitive outputs
    v
Joint PD targets --> Isaac Gym Physics --> Simulated Avatar
```

### 2.2 核心架构

PHC 由三大组件构成:

**A. PNN (Progressive Neural Network)**
- 每个 primitive 是独立的 MLP (`[2048, 1536, 1024, 1024, 512, 512]`, activation=SiLU)
- 通过 lateral connections 连接前序 primitive 的中间 activation (可选，代码中默认 `has_lateral=False`)
- 新增 primitive 时 freeze 之前所有 primitive 的参数
- 文件: `phc/learning/pnn.py`

**B. MCP Composer**
- 输入与 primitive 相同的 observation
- 输出 softmax 权重 `w` (维度 = num_prim)
- 最终 action = sum(w_i * action_i) -- 各 primitive 输出的加权和
- 文件: `phc/learning/amp_network_mcp_builder.py`

**C. Fail-State Recovery Module**
- 基于 `HumanoidImGetup` 类实现
- 在训练中随机注入 fall state (随机 root rotation + 自由下落 150 步)
- 使用 `recoveryEpisodeProb` 和 `fallInitProb` 控制 getup episode 比例
- 恢复期间 (90 步) 不计算 imitation termination
- 文件: `phc/env/tasks/humanoid_im_getup.py`

### 2.3 Reward Function

imitation reward 是经典的多项加权形式:

```
r = w_pos * exp(-k_pos * ||p_ref - p_sim||^2)        # 关节位置
  + w_rot * exp(-k_rot * ||angle(q_ref, q_sim)||^2)    # 关节旋转
  + w_vel * exp(-k_vel * ||v_ref - v_sim||^2)          # 线速度
  + w_ang_vel * exp(-k_ang_vel * ||w_ref - w_sim||^2)  # 角速度
```

默认权重: `w_pos=0.5, w_rot=0.3, w_vel=0.1, w_ang_vel=0.1`
默认系数: `k_pos=100, k_rot=10, k_vel=0.1, k_ang_vel=0.1`

文件: `phc/env/tasks/humanoid_im.py` L1524-1554

当 `zero_out_far=True` (fail recovery 模式) 时，额外计算 point_goal_reward:
- 远距离时只给导航奖励 (靠近参考 root 位置)
- 近距离时加入 imitation reward
- 实现了 getup → walk back → resume tracking 的完整流程

Power penalty: `-power_coefficient * |tau * dq|` (默认 `5e-5`)

### 2.4 Observation Space

支持多种 observation 版本 (obs_v):

| obs_v | 内容 | 用途 |
|-------|------|------|
| 1 | diff_pos + diff_rot + diff_vel + diff_ang_vel (全 local) | 默认 full body |
| 6 | v1 + 参考帧的 local body pos/rot (不是 diff) | rotation-based model |
| 7 | 仅 position + velocity (无 rotation) | keypoint model (PHC-KP) |

obs_v=7 是 keypoint-only 版本，不使用 rotation 信息，留给 RL 自己求解 IK。这是论文中推荐的 video/language control 模型。

### 2.5 训练超参数

| 参数 | 值 |
|------|-----|
| 算法 | PPO (rl_games) |
| Learning rate | 2e-5 (constant) |
| Gamma | 0.99 |
| GAE tau | 0.95 |
| Horizon length | 32 |
| Minibatch size | 16384 |
| Mini epochs | 6 |
| Clip | 0.2 |
| Grad norm | 50.0 |
| Task reward weight | 0.5 |
| Discriminator reward weight | 0.5 |
| AMP disc units | [1024, 512] |
| Control freq | 30 Hz (controlFrequencyInv=2) |

---

## 3. 关键设计

### 3.1 Progressive Multiplicative Control Policy (PMCP)

**问题**: 直接在万级别运动数据上训练单一 policy，performance 会在某个点 plateau，剩余"难"动作无法学会。

**方案**: 渐进式学习 + Progressive Neural Network:

1. 训练 Primitive 0 在全部数据上，等 performance 饱和
2. 运行 `forward_pmcp.py` 收集失败序列
3. 将 Primitive 0 的权重**复制**到 Primitive 1，Freeze Primitive 0
4. 在失败序列上训练 Primitive 1 (只更新 Primitive 1 的权重)
5. 重复直到无失败序列
6. 最后一个 Primitive 专门训练 fail-state recovery

**直觉**: 类似课程学习 (curriculum learning)，但不是把简单到难排序，而是让每个 primitive 专注于"前一个学不会的东西"。新 primitive 从旧的 checkpoint 初始化，加速收敛。

代码实现中的关键细节:
- `forward_pmcp.py` 从最近 5 轮 eval 的 failed_keys 取并集，dump 为下一阶段的训练集
- 新 primitive 的初始权重直接从 checkpoint 复制 (L60-61 in `forward_pmcp.py`)
- lateral connections 在代码中默认关闭 (`has_lateral=False`)，说明作者发现简单的 weight copy + independent training 就足够好

### 3.2 Mixture of Control Primitives (MCP) Composer

**问题**: 有了多个 primitive，如何在推理时选择正确的 primitive?

**方案**: 训练一个 Composer 网络，输出 softmax 权重，将所有 primitive 的 action 加权求和。

MCP step 逻辑 (`humanoid_im_mcp.py` L54-82):
```python
# 1. Normalize observation
curr_obs = (obs - running_mean) / sqrt(running_var + 1e-5)
curr_obs = clamp(curr_obs, -5, 5)

# 2. Forward all primitives through PNN
_, actions = self.pnn(curr_obs)    # list of [N, action_dim]
x_all = stack(actions, dim=1)     # [N, num_prim, action_dim]

# 3. Weighted sum with composer output
actions = sum(weights[:,:,None] * x_all, dim=1)
```

Composer 架构: 同样是深层 MLP `[2048, 1536, 1024, 1024, 512, 512]`，最后接 softmax (或不接，取决于 `has_softmax` flag)。

**关键发现**: 代码中 `im_mcp_big.yaml` 设置 `has_softmax: false`，而 Composer MLP 的最后一层输出维度就是 `num_prim`。这意味着 PHC+ 版本实际上用的不是 softmax weights，而是让 RL 自由学习 blending coefficients。

### 3.3 Fail-State Recovery (起身恢复)

**问题**: 在 wild 场景中 (视频/语言输入)，humanoid 不可避免会跌倒。如何在不 reset 的情况下恢复?

**方案**: 在训练最后一个 primitive 时，注入 fall state:

1. **Fall state 生成** (`_generate_fall_states`): 随机化 root rotation + 零 dof velocity → 模拟 150 步自由下落 → 保存 fall state
2. **训练策略**: 以概率 `fallInitProb=0.3` 从 fall state 开始; 以概率 `recoveryEpisodeProb=0.5` 将失败的 episode 转为 recovery episode
3. **Zero-out-far 机制**: 当 sim humanoid 距离 reference > close_distance 时，只保留 root position 的导航信号，其他 body 的参考设为当前状态。这引导 humanoid 先走回 reference 位置，再开始精确模仿
4. **Recovery counter**: 恢复期间 (90 步) 不做 termination check，不推进 progress buffer

**直觉**: Zero-out-far 巧妙地将"起身 → 走回 → 继续跟踪"分解为自然的行为序列，而无需显式的阶段划分。

---

## 4. 实验结果

### 4.1 AMASS 评估 (11313 sequences)

| 模型 | 成功率 | G-MPJPE (mm) | ACC |
|------|--------|-------------|-----|
| PHC (rotation) | 98.9% | 37.5 | 3.3 |
| PHC-KP (keypoint-only) | 98.7% | 40.7 | 3.5 |
| PHC+ (rotation+keypoint, in PULSE) | 100% | 26.6 | 2.7 |
| PHC-Prim (single primitive) | 99.9% | 25.9 | 2.3 |
| PHC-Fut (using future frames) | 100% | 25.3 | 2.5 |
| PHC-X-Prim (SMPLX, single) | 99.9% | 24.7 | 3.6 |

关键观察:
- PHC+ 达到 100% 成功率，这是在 AMASS 全集上的首次
- 单 primitive 模型 (PHC-Prim) 已达到 99.9%，说明 PMCP 的渐进式学习在 coverage 方面提升不大，但**提供了 getup 能力**
- README 明确说"MCP/MoE 模型对高成功率非绝对必要，但 getup 功能需要它"

### 4.2 支持的应用场景

| 场景 | 输入 | 关键配置 |
|------|------|---------|
| Motion imitation | AMASS pkl | `env=env_im` |
| Video-based control | Webcam → HyBriK | `task=HumanoidImMCPDemo`, obs_v=7 |
| Language-to-motion | MDM → SMPL | `task=HumanoidImMCPDemo` |
| VR controller tracking | Head + 2 hands | `env=env_vr`, trackBodies=["Head", "L_Hand", "R_Hand"] |
| H1/G1 robot | Retargeted data | `robot=unitree_h1/g1`, `sim=robot_sim` |

---

## 5. 领域背景

### 5.1 PHC 在 humanoid motion tracking 领域的地位

PHC 是当前 humanoid whole-body control 领域被引用最多的 baseline 之一:

| 后续工作 | 如何使用 PHC |
|----------|-------------|
| **PULSE** (ECCV 2024, 同一作者) | 在 PHC+ 基础上构建 language-conditioned controller |
| **SONIC** (NeurIPS 2025, NVIDIA) | 引用 PHC 作为 SMPL-based motion tracking 的 SOTA baseline |
| **HDMI** (2025, CMU) | 使用 PHC 的思路但扩展到人-物交互场景 |
| **OmniRetarget** (2025, Amazon) | 使用 BeyondMimic 风格训练 (与 PHC reward 类似)，引用 PHC retargeting |

PHC 的贡献不仅在算法层面，更在于建立了一套完整的 SMPL → Isaac Gym 的 pipeline:
- SMPL humanoid XML 自动生成 (SMPLSim)
- AMASS 数据预处理
- 多种 observation 方案
- 评估 protocol (sequential batch evaluation)

### 5.2 学术谱系

```
DeepMimic (Peng 2018)
    |
    v
UHC (Luo 2022, NeurIPS) -- Universal Humanoid Control
    |
    v
PHC (Luo 2023, ICCV) -- PMCP + Getup
    |
    +---> PULSE (Luo 2024) -- Language control
    +---> PHC+ (100% success)
    +---> H1/G1 support
```

---

## 6. 局限性与未来方向

### 6.1 作者承认的局限

1. **训练流程不自动化**: PMCP 训练需要手动多步操作 -- 监控 plateau、运行 forward_pmcp、修改 config、重启训练。README 原文: "Training PHC is not super automated yet, so it requires some (a lot of) manual steps"
2. **坐标系不一致**: 使用了非标准坐标系 (负 z 为重力方向, humanoid 面向正 x)，需要额外转换才能映射回 SMPL
3. **SMPL 依赖**: 高度依赖 SMPL 模型做 height fix 和 mesh 可视化

### 6.2 从代码推断的局限

1. **无 sim-to-real**: 整套系统仅在 Isaac Gym 仿真中运行，无 domain randomization (除 H1/G1 config 外)
2. **单环境交互**: 无物体操作、无地形变化 (HDMI 和 OmniRetarget 在此方向扩展)
3. **MCP 架构开销**: 推理时需要 forward 所有 primitive + composer，计算量随 primitive 数增长
4. **Auto-PMCP 简化版**: 后续 `auto_pmcp_soft` flag 表明作者发现可以用 soft sampling weight update 替代完整的 PMCP 流程，暗示原始 PMCP 过于复杂

### 6.3 未来方向

- 自动化 PMCP 流程 (已部分通过 auto_pmcp_soft 实现)
- 扩展到 SMPLX (手指控制, 已实现)
- 扩展到真实机器人 (H1/G1 retargeting, 已实现)
- IsaacLab 集成 (已在 2025 年 8 月实现)
- Offline RL dataset 生成 (PHC_Act, 2024 年 12 月)

---

## 7. 代码细节与 Paper 未提及的实现

### 7.1 Auto-PMCP Soft

代码中存在 `auto_pmcp_soft` flag (默认 True)，实现了一种自动化的 PMCP 替代方案:
- 不需要手动运行 `forward_pmcp.py`
- 通过 `update_soft_sampling_weight` 自动调整 motion 采样概率
- 失败次数越多的 motion clip 采样概率越高
- 这本质上是 hard negative mining 的 soft 版本

文件: `phc/learning/im_amp.py` L126-132

### 7.2 Observation Occlusion Training

代码支持 `_occl_training` 模式:
- 随机遮挡部分 tracked body 的参考信号 (用当前 sim 状态替代)
- 遮挡持续 30-60 帧
- 提升对 noisy/missing input 的鲁棒性

文件: `phc/env/tasks/humanoid_im.py` L1081-1092

### 7.3 AMP (Adversarial Motion Prior) Discriminator

代码框架完整保留了 AMP discriminator 的训练逻辑:
- Discriminator: `[1024, 512]` MLP
- disc_reward_w=0.5, task_reward_w=0.5
- 实际训练中 AMP 的 disc reward 与 imitation task reward 各占一半
- AMP replay buffer: 200K samples

这一点 paper 中可能未充分强调: PHC 不仅用 task reward，还同时用 AMP discriminator reward。

### 7.4 VR Tracking 模式

`env_vr.yaml` 揭示 VR 模式只追踪 3 个点:
```yaml
trackBodies: ["Head", "L_Hand", "R_Hand"]
reset_bodies: ["Head", "L_Hand", "R_Hand"]
```
这是经典的 3-point VR tracking 设定 (头 + 双手)。

### 7.5 MLP Bypass 模式

`humanoid_im_mcp.py` 支持 `mlp_bypass` flag:
- 跳过 PNN + Composer，直接用单个 MLP 从 observation 映射到 action
- 用于 behavior cloning / offline RL (PHC_Act)
- MLP 结构: `[2048, 1024, 512]`, activation=SiLU

### 7.6 Residual Action

代码支持 `_res_action` 模式:
- 输出是基于 reference dof_pos 的残差: `pd_target = ref_dof_pos + scale * action`
- 并 clamp 到当前 dof_pos +/- pi/2

### 7.7 Isaac Gym → IsaacLab 迁移

2025 年 8 月新增了 `scripts/eval_in_isaaclab.py`，表明 PHC policy 可直接在 IsaacLab 中推理，为后续工作 (如 SONIC) 的 IsaacLab 框架迁移铺平道路。

---

## 8. 跨论文对比

### 8.1 PHC vs SONIC vs HDMI vs OmniRetarget

| 维度 | PHC (2023) | SONIC (2025) | HDMI (2025) | OmniRetarget (2025) |
|------|-----------|-------------|-------------|---------------------|
| 核心任务 | 大规模 motion imitation + getup | 大规模 motion tracking as foundation | 人-物交互 co-tracking | 高质量 retargeting 数据生成 |
| 仿真器 | Isaac Gym | Isaac Lab | Isaac Lab | Isaac Lab (HoloSoma) |
| 机器人 | SMPL/SMPLX/H1/G1 | Unitree G1 (29 DOF) | Unitree G1 | G1/H1/T1 |
| Motion 数据 | AMASS 11K clips | AMASS 100K+ clips | RGB 视频 | AMASS + 交互场景 |
| 策略架构 | PNN + MCP Composer | Encoder-FSQ-Decoder (42M params) | PPO + residual action | PPO + BeyondMimic reward |
| Reward 设计 | Task (imitation) + AMP disc | Task only (no disc) | Imitation + contact | Imitation (5 terms) |
| 独特能力 | Getup recovery, perpetual control | Multi-modal input, sim-to-real | Object manipulation | Cross-embodiment, artifact-free data |
| Sim-to-Real | 无 | 有 (zero-shot G1) | 有 (zero-shot G1) | 有 (zero-shot G1/H1) |
| 成功率 | 100% (PHC+) | ~100% | N/A (task-specific) | N/A (task-specific) |
| Observation | local diff + ref pose | body-centric + command | proprio + object + phase | proprio + command |
| 关键创新 | PMCP 渐进式学习 | FSQ tokenized motion + scalability | Unified object repr + co-tracking | Interaction mesh retargeting |

### 8.2 PHC 对后续工作的影响

| 继承关系 | 具体内容 |
|----------|---------|
| PHC → SONIC | SONIC 使用 SMPL motion data pipeline (AMASS 处理); 类似的 imitation reward 结构; 但用 FSQ 替代 PNN/MCP |
| PHC → HDMI | HDMI 的 RL 训练框架继承 PHC 的 Isaac Gym 基础设施; retargeting 步骤参考 PHC 的 SMPL → robot 流程 |
| PHC → OmniRetarget | OmniRetarget 的 BeyondMimic reward 与 PHC imitation reward 结构高度相似 (位置+旋转+速度); 引用 PHC retargeting docs |
| PHC → PHC_Act | 将 PHC 的 online RL 数据转为 offline dataset，支持 behavior cloning 和 offline RL 研究 |

### 8.3 技术演进路线

```
PHC (2023): PMCP + AMP disc + Isaac Gym
  |
  +--> 简化: auto_pmcp_soft (自动 hard negative mining, 去掉手动 PMCP 流程)
  +--> 简化: single primitive achieves 99.9% (PMCP 对 coverage 不是必需的)
  |
  v
SONIC (2025): 去掉 AMP disc, 去掉 progressive training
             直接 scale up data + 更大网络 (42M)
             + FSQ latent space + sim-to-real
  |
  v
结论: PHC 的 PMCP 是重要的历史性贡献，但后续工作证明
      "数据规模 + 网络容量" 可能比 "渐进式训练" 更重要
```

---

## 附录: 代码目录结构

```
PHC/
├── phc/
│   ├── run_hydra.py                    # Main entry (Hydra config)
│   ├── env/tasks/
│   │   ├── humanoid.py                 # Base humanoid (dof, obs, PD control)
│   │   ├── humanoid_amp.py             # AMP + motion loading
│   │   ├── humanoid_amp_task.py        # Task obs interface
│   │   ├── humanoid_im.py              # Motion imitation (reward, obs, reset) [CORE]
│   │   ├── humanoid_im_getup.py        # Fail-state recovery
│   │   ├── humanoid_im_mcp.py          # MCP step logic (weighted primitive sum)
│   │   ├── humanoid_im_mcp_getup.py    # MCP + Getup (multiple inheritance)
│   │   ├── humanoid_im_mcp_demo.py     # Live demo (webcam/language)
│   │   └── humanoid_teleop.py          # Teleop with domain rand
│   ├── learning/
│   │   ├── pnn.py                      # Progressive Neural Network [CORE]
│   │   ├── amp_network_pnn_builder.py  # PNN network wrapper
│   │   ├── amp_network_mcp_builder.py  # MCP Composer network
│   │   ├── amp_network_builder.py      # Base AMP network (actor + disc)
│   │   ├── im_amp.py                   # Training agent (eval, auto_pmcp)
│   │   ├── network_loader.py           # Checkpoint loading utilities
│   │   └── mlp.py                      # MLP bypass model
│   ├── utils/
│   │   ├── motion_lib_smpl.py          # SMPL motion library [CORE]
│   │   ├── motion_lib_real.py          # Real robot motion library
│   │   └── motion_lib_base.py          # Base motion lib (sampling, height fix)
│   └── data/cfg/
│       ├── config.yaml                 # Hydra defaults
│       ├── env/                        # Environment configs
│       ├── learning/                   # RL training configs
│       └── robot/                      # Robot configs (SMPL, SMPLX, H1, G1)
├── scripts/
│   ├── pmcp/forward_pmcp.py            # PMCP progression script
│   ├── data_process/                   # AMASS data processing
│   ├── demo/                           # Live demo server
│   └── phc_act/                        # Offline dataset generation
└── docs/
    ├── retargeting.md                  # Cross-embodiment retargeting guide
    └── offline_dataset.md              # PHC_Act documentation
```
