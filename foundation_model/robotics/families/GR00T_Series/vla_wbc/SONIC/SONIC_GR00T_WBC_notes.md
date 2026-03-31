# SONIC & GR00T Whole-Body Control 分析笔记

> 论文: SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control (arXiv:2511.07820)
> 代码: https://github.com/NVlabs/GR00T-WholeBodyControl
> 日期: 2026-03-10

---

## 1. 核心思想

**一句话**: 用大规模运动追踪 (Motion Tracking) 替代逐任务手工奖励工程，作为人形机器人全身控制的统一基础任务。

传统方法的瓶颈在于每个新能力 (走路、舞蹈、起身、遥操) 都需要手工设计奖励函数。SONIC 的关键洞察是: **运动追踪天然具备可扩展性** ——人类动捕数据已有数百万帧，提供逐帧密集监督，无需奖励工程。

**与 bh_motion_track 的关系**: 我们的双手 MotionGen 轨迹跟踪任务与 SONIC 在方法论上高度一致 (BeyondMimic 风格的运动追踪 RL)，但 SONIC 将这个思路从灵巧手扩展到了全身 29 DOF 人形机器人。

---

## 2. 系统架构

```
                    ┌─────────────────────────────┐
                    │  Multi-Modal Input Layer     │
                    │  (VR / Video / Text / Music) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  GENMO (Generative Motion)   │
                    │  扩散模型: 任意模态 → SMPL    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │   Generative Kinematic Motion Planner   │
              │   潜空间自回归补间, 0.8-2.4s 片段       │
              │   推理: Jetson Orin 12ms                │
              └────────────────────┬────────────────────┘
                                   │ motion command
              ┌────────────────────▼────────────────────┐
              │   Universal Motion Tracking Policy      │
              │   PPO, 42M params, 100M+ frames         │
              │   3 编码器 → FSQ → 2 解码器             │
              │   策略频率 50Hz, PD 执行 500Hz          │
              └────────────────────┬────────────────────┘
                                   │ joint targets
                          Unitree G1 (29 DOF)
```

三个核心模块:
1. **Universal Motion Tracking Policy** — RL 训练的运动追踪策略 (核心)
2. **Generative Kinematic Motion Planner** — 从用户指令生成短视界运动片段
3. **Multi-Modal Motion Generation (GENMO)** — 视频/文本/音乐 → 人类运动

---

## 3. 编码器-解码器架构 (Universal Token Space)

这是 SONIC 最重要的架构创新。通过统一的 token 空间实现跨身体运动迁移:

```
Robot Motion ──→ Robot Encoder ──┐
                                 │
Human Motion ──→ Human Encoder ──┼──→ FSQ Quantizer ──→ Universal Token
                                 │           │
Sparse Points ─→ Hybrid Encoder ─┘           │
                                             ├──→ Control Decoder → joint targets (PD tracking)
                                             └──→ Motion Decoder  → robot motion (辅助监督)
```

**FSQ (Finite Scalar Quantization)**: 将连续向量量化为离散 token, 维度 D_z, 每维 L_z 级。相比 VQ-VAE 更稳定。

**三种编码器** (MLP, hidden dims [2048, 1024, 512, 512]):

| 编码器 | 输入 | 未来帧数 | 用途 |
|--------|------|---------|------|
| Robot Encoder | 关节位置+速度 | F_r 帧 | 机器人自身运动 |
| Human Encoder | SMPL 3D 关节 | F_h 帧 | 人类动捕数据 |
| Hybrid Encoder | 稀疏关键点 (头+双手) + 机器人下身 | F_m 帧 | VR 遥操 (3 点追踪) |

**训练损失** (4 项):

$$\mathcal{L}_{total} = \mathcal{L}_{PPO} + \mathcal{L}_{recon} + \mathcal{L}_{token} + \mathcal{L}_{cycle}$$

| 损失 | 公式 | 作用 |
|------|------|------|
| $\mathcal{L}_{recon}$ | $\|D_r(z_r) - g_r\|^2 + \|D_r(z_h) - g_r\|^2$ | 重建 + 隐式重定向 |
| $\mathcal{L}_{token}$ | $\|z_r - z_h\|^2$ | 跨身体 token 对齐 |
| $\mathcal{L}_{cycle}$ | $\|E_r(D_r(z_h)) - z_r\|^2$ | 循环一致性 |

**关键设计**: 人类 token $z_h$ 通过 Robot Control Decoder 直接生成机器人动作，$\mathcal{L}_{recon}$ 项自动学习了人→机器人的运动重定向，无需 MANO 或骨骼映射。

---

## 4. 运动重定向

**身体重定向**:
- 输入: SMPL 格式的全身 3D 关键点
- 方法: GMR (General Motion Retargeting) + PyRoki
- 170 位受试者, 身高 145-199cm (均值 174.3cm)

**手部重定向** (如果适用):
- 非等距形状匹配 + 接触约束 IK
- 20 DOF 灵巧手从人手运动映射

**与 bh_motion_track 对比**: 我们使用 tracker → wrist body 的两步旋转链 (QUAT_Z90 + R_OP2MANO^{-1})，SONIC 用端到端学习隐式替代了这类手工映射。

---

## 5. RL 训练

### 5.1 MDP 公式

$$s_t = (s_t^p, s_t^g) = (q_t, \dot{q}_t, \omega_t, \psi_t, a_{t-1}) + (g_r | g_h | g_m)$$

- $s_t^p$: 本体感知 (关节位置/速度/角速度/重力向量/上一动作)
- $s_t^g$: 运动命令 (三种编码器之一的输出)
- 所有旋转量在 heading frame 下表示 (旋转不变性)

**动作空间**: 29D 目标关节位置, PD 控制器跟踪

### 5.2 奖励设计

$$r_t = R(s_t^p, s_t^g) + P(s_t^p, a_t)$$

**追踪项 R**:

| 奖励项 | 内容 |
|--------|------|
| Root position | 根部位置匹配 |
| Root orientation | 根部朝向匹配 |
| Body link position | 所有身体链接位置 (相对根) |
| Body link orientation | 所有身体链接朝向 (相对根) |
| Linear velocity | 身体链接线速度 |
| Angular velocity | 身体链接角速度 |

**惩罚项 P**:

| 惩罚项 | 内容 |
|--------|------|
| Action smoothness | 动作剧烈变化惩罚 |
| Joint limit | 关节限位违反惩罚 |
| Undesired contact | 不期望接触惩罚 |

**与 bh_motion_track 的异同**:
- 相同: 都用 Gaussian kernel 形式的追踪奖励, 都有 action rate + energy 惩罚
- 不同: SONIC 无 contact matching reward (人形无需指尖接触匹配); 我们有 object tracking + contact guidance curriculum, SONIC 没有 (人形无需抓取物体)

### 5.3 域随机化

| 参数 | 范围 | 说明 |
|------|------|------|
| 静摩擦 $\mu_s$ | 0.3 - 1.6 | 地面接触 |
| 动摩擦 $\mu_d$ | 变化 | 地面接触 |
| 恢复系数 $e$ | 0 - 0.5 | 弹跳行为 |
| 默认关节位 $q_0$ | 扰动 | 校准误差 |
| 基座重心偏移 | 随机 | 重量分布 |
| 根部速度/角速度 | 随机扰动 | 模拟外部推力 |
| 目标运动命令 | 扰动 | 增强鲁棒性 |

**自适应采样** (非均匀):

$$p_i = \alpha \cdot \text{cap}(f_i) + (1-\alpha)/N$$

$f_i$ 是每个 bin 的失败率, $\alpha=0.1$ 平衡挑战性运动与均匀覆盖。

### 5.4 PPO 超参数

| 参数 | 值 |
|------|-----|
| 模型规模 | 1.2M → 42M 参数 |
| 数据量 | 100M+ 帧 (700 小时, 50Hz) |
| 计算量 | 128 GPU, 9000 GPU-hours, 3-7 天 |
| 策略频率 | 50 Hz |
| 硬件执行 | 500 Hz (Unitree 低级 API) |
| 分布式训练 | HuggingFace Accelerate + TRL |
| 仿真平台 | Isaac Lab (GPU 加速) |

---

## 6. 缩放实验 (核心发现)

这是论文最重要的实验结论——运动追踪具有良好的缩放特性:

### 6.1 数据缩放

| 数据集 | 帧数 | 效果 |
|--------|------|------|
| LaFAN | 0.4M | 基线 |
| 子集 | 7.4M | 显著提升 |
| 完整 | 100M+ | 持续提升, 未饱和 |

### 6.2 模型缩放

| 参数量 | 效果 |
|--------|------|
| 1.2M | 基线 |
| 10M | 提升 |
| 42M | 最佳, 性能单调递增 |

### 6.3 计算缩放

| GPU 数量 | 效果 |
|----------|------|
| 8 | 基线 |
| 32 | 提升 |
| 128 | 最佳渐近性能 |

**关键洞察**: GPU 数量影响训练动力学 (不仅是速度)，更多并行环境产生更好的梯度估计。

---

## 7. 评估指标与结果

### 7.1 运动追踪指标

| 指标 | 定义 |
|------|------|
| Success Rate | 成功追踪的运动百分比 |
| MPJPE | Mean Per-Joint Position Error (mm) |
| $E_{vel}$ | 速度误差 (mm/frame) |
| $E_{acc}$ | 加速度误差 (mm/frame^2) |

### 7.2 Baseline 对比

SONIC 在所有指标上显著超越 AnyTrack, BeyondMimic, GMT。

### 7.3 真实机器人

- **50 条多样化运动轨迹** (舞蹈/跳跃/操作): **100% 成功率**
- **零样本迁移**: 无需真实世界微调
- **VR 遥操延迟**: 右腕 121.9ms; 位置误差 6cm (均值), 13.3cm (P95); 方向误差 0.145 rad (8.3°)
- **VLA 集成 (GR00T N1.5)**: 苹果放盘子任务 20 次中 **95% 成功率**

---

## 8. Decoupled WBC (解耦全身控制, GR00T N1.5/N1.6)

SONIC 是统一策略方案; GR00T 还有一个解耦方案作为对比:

```
          ┌──────────────────┐
          │  Shared Obs      │
          │  (全身状态)       │
          └────┬────────┬────┘
               │        │
   ┌───────────▼──┐  ┌──▼───────────┐
   │  Lower Body  │  │  Upper Body  │
   │  RL (50 Hz)  │  │  IK (100 Hz) │
   │  步态/平衡   │  │  末端精确控制 │
   └──────────────┘  └──────────────┘
```

**解耦方案优势**:
- 末端执行器加速度降低 50-80%
- 稳定端着水杯行走
- 分别优化避免目标冲突

**统一方案 (SONIC) 优势**:
- 自然的全身协调
- 无需手工设计上下身协调奖励
- 利用人类运动先验自动学习协调

---

## 9. 运动规划器 (Generative Kinematic Motion Planner)

潜空间自回归补间模型, 从用户指令生成短片段运动:

**架构**: Transformer/Conv1D backbone, mask token prediction (迭代精化)

**输入**: 上下文关键帧 + 目标关键帧 (弹簧模型生成)
**输出**: 0.8-2.4 秒运动片段
**推理**: Jetson Orin 12ms, 笔记本 <5ms

**支持的交互模式**:

| 模式 | 输入 | 延迟 |
|------|------|------|
| Gamepad | 速度/方向/风格 | 实时 |
| VR 全身 | SMPL 全身姿态 | 121.9ms (腕部) |
| VR 3 点 | 头+双手+导航 | 低延迟 |
| 视频 | 单目视频 → SMPL | ≥60fps |
| 文本 | 自然语言描述 | 批处理 |
| 音乐 | 音频节拍 | 批处理 |

---

## 10. 与 bh_motion_track 项目的关联

### 10.1 方法论对应

| 维度 | bh_motion_track | SONIC |
|------|----------------|-------|
| 核心任务 | Motion Tracking (双手灵巧操作) | Motion Tracking (全身人形) |
| 追踪奖励 | Gaussian kernel (wrist/tips/object) | 链接位置/朝向/速度 |
| 控制模式 | absolute finger + residual wrist | PD 目标关节位置 |
| 动作分布 | tanh_normal (52D) | Gaussian (29D) |
| 参考格式 | .pt (MotionGen, 双手+物体) | SMPL + 机器人关节 |
| 数据规模 | ~200 轨迹 | 100M+ 帧 (700 小时) |
| 特有机制 | Contact Guidance (weld) + Contact Gate | Universal Token Space (FSQ) |
| DR | 未实现 | 全面 (摩擦/质量/重心/外力) |
| Sim2Real | 未验证 | 零样本迁移, 100% 成功率 |

### 10.2 可借鉴的设计

1. **自适应采样**: SONIC 的 bin-based failure rate 采样比我们的均匀随机采样更高效，失败率高的轨迹被更频繁采样
2. **域随机化**: SONIC 的 DR 范围全面 (我们的 DRCfg 已定义但未实现), 是 sim2real 的关键
3. **旋转不变性**: SONIC 所有观测在 heading frame 下表示, 我们的 V2 obs 也用了类似设计 (object-frame fingertip errors)
4. **缩放特性**: SONIC 证明了更多数据+更大模型+更多 GPU = 更好性能, 我们目前 ~200 轨迹远未饱和

### 10.3 关键差异

- **SONIC 无接触奖励**: 人形全身控制不需要精确的指尖接触匹配, 我们的 contact reward + contact gate 是灵巧操作特有的
- **SONIC 无 curriculum**: 人形不需要 contact guidance weld (无物体抓取), 我们的 GuidanceCurriculum 解决的是"物体在策略学会抓握前掉落"的问题
- **SONIC 无 object tracking**: 人形追踪自身运动, 我们还需追踪物体 (object reward w=8.0)
- **编码器多样性**: SONIC 3 种编码器支持多种输入模态, 我们目前固定使用 .pt 格式

---

## 11. 关键 Takeaway

1. **Motion Tracking 是可扩展的基础任务** — 不需要为每个新能力设计奖励, 数据驱动
2. **缩放三轴 (数据/模型/计算) 均未饱和** — 更多数据和更大模型持续带来收益
3. **Universal Token Space 是跨身体迁移的关键** — FSQ 量化器 + 对齐损失实现人→机器人
4. **零样本 Sim2Real 需要全面 DR** — SONIC 的 100% 真实机器人成功率依赖于充分的域随机化
5. **解耦 vs 统一全身控制各有优势** — 统一策略自然协调, 解耦方案更稳定
6. **部署效率关键** — 策略 1-2ms, 规划器 12ms, 全部本地 Jetson Orin 推理
