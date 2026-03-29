# HATO: Learning Visuotactile Skills with Two Multifingered Hands - 论文笔记

**论文**: Learning Visuotactile Skills with Two Multifingered Hands
**作者**: Toru Lin, Yu Zhang\*, Qiyang Li\*, Haozhi Qi\*, Brent Yi, Sergey Levine, Jitendra Malik
**机构**: UC Berkeley
**发表**: arXiv:2404.16823, 2024
**项目**: https://toruowo.github.io/hato/

---

## 一句话总结

首个双臂多指手视触觉 imitation learning 系统：VR 手柄遥操作 2x UR5e + 2x Psyonic Ability Hand (假肢产品)，3 个 RealSense + 60 通道触觉，Diffusion Policy 端到端学习 4 个双手任务。

---

## 核心问题

双臂多指手的 imitation learning 缺两个关键基础设施：
1. 低成本遥操作系统 (现有系统几乎都用平行夹爪)
2. 配备触觉传感的多指手硬件 (极度稀缺)

**解法**: 将假肢产品 (Psyonic Ability Hand，自带触觉) repurpose 为研究平台 + VR 手柄极简映射遥操作。

---

## 方法

### 硬件
- 2x UR5e + 2x Psyonic Ability Hand (6 DOF/手, 30 触觉通道/手)
- 3x RealSense (2 腕部 + 1 固定)
- 定制 PCB 简化电气接线

### 遥操作 (核心贡献)
- VR 手柄 6-DOF → UR5e IK
- **手部极简映射**: Grip button → 4 指同步屈伸 (power grasp); Thumbstick → 拇指 2 DOF
- Pause-and-adjust: Trigger 断开/重连手臂控制
- 牺牲 finger-gaiting 换取直觉操作体验

### 策略 (Diffusion Policy)
- **视觉**: 3 个独立 ResNet-18 (BN→GroupNorm, 输出 32D)
- **触觉**: 2 层 MLP (input→256→64, ELU)。支持二值化 (>1000→1.0)
- **Proprioception**: EEF pose (axis-angle 12D) 而非关节角 (避免奇异点问题)
- **扩散**: 1D ConditionalUnet [256,512,1024], FiLM 条件注入, 100 train / 15 inference steps
- **Observation horizon=1** (单帧就够)

### 部署 (异步推理)
- ZMQ 分布式: camera_node / robot_node / inference_node 异步通信
- Temporal Ensemble: 多次预测在相同 timestep 上的聚合 (avg/act/old/new)

---

## 关键结果

| 任务 | 演示数 | 成功率 | 关键发现 |
|------|--------|--------|----------|
| Slippery Handover | 100 | 10/10 | img+proprio 够用 |
| Block Stacking | 100 | 10/10 | 触觉在 rare init 下必要 (无触觉 4/10) |
| Wine Pouring | 300 | 9/10 | 深度信息有时有害 |
| **Steak Serving** | **300** | **5/10** | **无触觉 0/10 但 ActionMSE 几乎相同** |

- ActionMSE 不是好的策略质量度量 (0.07 vs 0.08 对应 5/10 vs 0/10)
- 腕部相机 > 第三视角 (3/4 任务)

---

## 作者展望

1. 遥操作加入触觉反馈 (将触觉信号反馈给操作员)
2. 更鲁棒、更通用的策略 (当前无预训练)

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| MLP 激活函数 | ReLU | **ELU** (代码 TODO: "is ELU the best?") |
| ResNet FC 输出 | 32D | 部署默认 **128D** |
| 手部关节范围 | 未提及 | 部署 hack: `grip_range=75` (论文范围 110) |
| 数据增强 | 未详述 | img_gaussian_noise, img_masking_prob, color_jitter, binarize_touch |
| LR 调度 | 未提及 | Cosine + 500 步 warmup |

### 值得学习的代码设计

1. **ZMQ 分布式节点**: camera/robot/inference 完全解耦，异步通信 → 高频控制的工程核心
2. **Agent Protocol 模式**: Quest 遥操/DP 推理/ZMQ 异步推理实现统一 `act(obs)->action` 接口
3. **SafetyWrapper**: delta clipping (arm 0.5, hand 0.1) 硬安全层
4. **Memmap 缓存**: 图像首次加载后 memmap 到磁盘，后续训练直接读取
5. **Ability Hand 串口协议**: fixed-point 位置编码 + 12 位 ADC 紧凑触觉编码，独立线程 ~100Hz

---

## 非显而易见的洞察

1. **假肢产品 > 研究平台**: Ability Hand 自带触觉+人形形态，改造仅需定制 PCB
2. **EEF pose > joint positions**: UR5e 奇异点附近关节角剧变，用 axis-angle EEF pose 更稳定
3. **Observation horizon=1 就够**: 原版 Diffusion Policy 用短历史，HATO 发现单帧足够
4. **平行夹爪的失败是系统性的**: 多指手的冗余接触面积从根本上降低遥操作难度
5. **手部归一化用固定范围**: `[5, [110,110,110,110,90,120]]` 硬编码，避免数据覆盖不全时归一化出问题
