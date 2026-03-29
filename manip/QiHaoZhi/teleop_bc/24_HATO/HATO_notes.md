# HATO - 论文笔记

**论文**: Learning Visuotactile Skills with Two Multifingered Hands
**作者**: Toru Lin, Yu Zhang, Qiyang Li, Haozhi Qi, Brent Yi, Sergey Levine, Jitendra Malik (UC Berkeley)
**发表**: arXiv:2404.16823, 2024
**项目**: https://toruowo.github.io/hato/
**代码**: https://github.com/ToruOwO/hato.git

---

## 一句话总结

构建了一套低成本双臂多指手遥操作系统 HATO, 利用视觉-触觉多模态数据通过 Diffusion Policy 从人类示教中学习双手灵巧操控技能, 首次在真实世界中实现了 visuotactile bimanual dexterous manipulation 的端到端 imitation learning。

---

## 核心问题

在双臂多指手操控领域存在两个关键瓶颈:

1. **缺乏可用的双臂多指手遥操作系统**: 现有遥操作系统大多只支持 parallel-jaw gripper 或单手场景, 缺少面向多指手双臂系统的低成本、直觉化操作方案。多指手的高自由度使得传统的 retargeting 方案延迟大、不直观。

2. **缺少带触觉传感的多指手硬件**: 市面上的研究用多指手 (如 Allegro Hand) 通常不配备触觉传感器, 而触觉信息对于精细操控至关重要。

这两个瓶颈导致无法高效采集包含视觉-触觉的双手灵巧操控示教数据, 进而无法训练端到端的多模态操控策略。

---

## 方法概述

### 硬件系统

| 组件 | 规格 | 说明 |
|------|------|------|
| 机械臂 | 2x UR5e | 6 DOF 工业臂, 关节范围 [-2pi, 2pi] |
| 灵巧手 | 2x Psyonic Ability Hand | 原为假肢手, 5 指 6 DOF (每指 1 DOF + 拇指 2 DOF), 每指 6 个触觉传感器 |
| 遥操作设备 | Meta Quest 2 | VR 头盔 + 2 个控制器, 包含 thumbstick、trigger、grip 按钮 |
| 相机 | 3x Intel RealSense | 2 个腕部 + 1 个固定第三视角, RGB-D 480x640 |
| 触觉传感器 | 60 个 (每手 30) | ADC 值, 无接触时 ~200-400, 接触时 >1000 |

### 遥操作映射 (Teleoperation Pipeline)

- **手臂控制**: Quest 控制器 pose --> 坐标变换到 UR5e 坐标系 --> 逆运动学 (IK) 求解关节角 --> 发送关节位置指令。三种 IK 实现: (1) 完整 IK 求解 (默认); (2) 一阶近似线性映射; (3) 机载 IK。
- **手指控制**: grip 按钮 --> 4 个非拇指手指的屈伸 (power grasp); thumbstick 2D 位置 --> 拇指的 2 DOF (屈伸 + 外展/内收)。这种简化映射牺牲了独立 finger gaiting 能力, 但提供了直觉化操作体验。
- **暂停调整**: trigger 按钮控制手臂控制序列的开始/中断, 允许操作者在操作中调整姿势。

### 数据采集与预处理

数据在 10 Hz 采集, 包含:
- **本体感知**: 关节位置 (24 维: 2 臂 x 6 + 2 手 x 6), EEF pose (translation + axis-angle)
- **视觉**: 3 个 RGB-D 相机, 原始 480x640, 下采样到 240x320
- **触觉**: 60 维触觉传感器读数 (h in R^60)
- **动作**: 24 维关节位置指令 (2 臂 x 6 + 2 手 x 6)

**数据归一化**: 所有值线性缩放到 [-1, 1]。手部关节位置使用固定范围 min=0, max=[110, 110, 110, 110, 90, 120]。RGB 使用原始值 [0, 255], depth 使用 [0, 65535]。

### 策略学习 (Policy Learning)

采用 Diffusion Policy (DDPM), 核心架构:

- **观测编码**:
  - 本体感知 (EEF pose, 12 维): 2 层 MLP + ReLU, hidden=256, output=64
  - 触觉 (60 维): 同上结构, output=64
  - 视觉 (3 相机): 各自独立的 ResNet-18 (BatchNorm 替换为 GroupNorm), output=32 per camera
- **Diffusion 骨架**: 1D Conditional U-Net, 输入观测编码的拼接作为 global condition, 预测 16 步 action sequence
- **关键超参数**: obs_horizon=1, pred_horizon=16, action_horizon=8, training diffusion steps=100, inference diffusion steps=15

### 异步部署 (Asynchronous Deployment)

部署时采用 client-server 架构:
- **Local process** (robot control): 每个 control step 发送最新观测到远程推理服务器
- **Remote inference server**: 持续在最新观测上运行 diffusion model, 产出 action sequence
- **Temporal ensemble**: 对多个时间步的预测进行加权平均, 提高动作平滑性 (论文中原始 Diffusion Policy 不使用 temporal ensemble)

---

## 关键设计

### 1. Ability Hand 的创新复用

将原本用于假肢的 Psyonic Ability Hand 改造为研究用途。关键设计:
- **自定义 PCB**: 简化电气接线, 集成通信接口与电源分配
- **触觉传感**: 每指 6 个触觉传感器 (共 60 个), 提供连续压力值, 这在研究用多指手中极为少见
- **四连杆机构**: 非拇指手指的 MCP 关节通过四连杆连接到 PIP 关节, 提供额外的欠驱动自由度

这使得 HATO 成为同期少有的配备丰富触觉传感的双臂多指手系统。

### 2. 直觉化手指映射方案

放弃传统的 retargeting (将人手姿态映射到机器人手关节), 转而使用极简映射:
- grip 按钮 (1 维力度) --> 4 指同步 power grasp
- thumbstick (2 维) --> 拇指 2 DOF

这种方案虽然牺牲了独立手指控制, 但极大降低了操作难度, 新手操作者可在 5-10 分钟练习后高效采集数据。实验证明这种 power grasp 足以完成多种复杂任务 (符合 grasp taxonomy 中人类日常操作的常见模式)。

### 3. 异步推理 + Temporal Ensemble

论文提出的异步推理是部署顺畅性的关键。与原始 Diffusion Policy 的同步推理不同, HATO 的 inference server 不断运行, local process 在每个 control step 只需取最新预测。结合 temporal ensemble (对重叠预测取平均), 显著提升了动作平滑度, 避免了 diffusion model 较慢推理速度带来的控制卡顿。

---

## 实验

### 任务与数据量

| 任务 | 示教数量 | 每条时长 | 总时长 | 特点 |
|------|----------|----------|--------|------|
| Slippery Handover | 100 | ~6s | ~10 min | 滑手物体双手传递 |
| Tower Block Stacking | 100 | ~20s | ~33 min | 大块积木堆叠 |
| Wine Pouring | 300 | ~25s | ~2.5 hr | 酒瓶倒酒, 质心变化 |
| Steak Serving | 300 | ~40s | ~3.3 hr | 铲子翻牛排, 长 horizon |

### 主要成功率

| 任务 | 抓取成功率 | 任务成功率 | 使用模态 |
|------|-----------|-----------|----------|
| Slippery Handover | 100% | 100% | img + proprioception |
| Tower Block Stacking | 100% | 100% | img + proprioception + touch |
| Wine Pouring | 100% | 90% | img + proprioception |
| Steak Serving | 100% | 50% | img + proprioception + touch |

### Visuotactile Ablation 核心发现

1. **视觉至关重要**: 移除视觉后所有任务的 prediction error 显著上升, Steak Serving 无法完成抓取 (0/10)。
2. **触觉不可或缺**:
   - Block Stacking rare initialization: 有触觉 10/10, 无触觉 4/10, 无视觉 0/10
   - Steak Serving: 有触觉 5/10, 无触觉 0/10, 尽管二者 ActionMSE 接近 (0.07 vs 0.08)
   - 这揭示了 ActionMSE 无法完全反映策略质量, 因为 diffusion model 的输出分布难以用均值误差衡量
3. **腕部相机优于第三视角**: 腕部相机提供更丰富的任务相关信息 (更少遮挡 + 运动中的透视线索)
4. **Depth 信息无明显帮助**: 在所有任务上, 加入 depth 几乎不改善甚至略微降低性能, 可能因为 RealSense 的 depth 噪声较大

### 数据量消融

Block Stacking 在 75 条示教时饱和, Steak Serving 在 100 条, Wine Pouring 在 200 条。这表明几百条示教数据足以学习有效的双手灵巧策略。

---

## 相关工作分析

HATO 处于多个研究方向的交叉点:

1. **双臂操控**: 此前工作几乎全部使用 parallel-jaw gripper (ALOHA、UMI 等)。DexCap (Wang et al., 2024) 是同期工作, 展示了双手灵巧操控但仅使用视觉+本体感知, 无触觉。HATO 是首个在双臂多指手系统上使用 visuotactile imitation learning 的工作。

2. **遥操作**: 现有系统要么限于 parallel-jaw gripper, 要么依赖 retargeting (延迟大)。HATO 通过 power grasp + thumbstick 映射提供了更直觉化的方案。

3. **Visuotactile learning**: 此前工作局限于单手或 gripper, 且未将视触觉与多指手灵巧操控结合。HATO 首次在双臂多指手场景下验证了 visuotactile sensing 对策略成功率和鲁棒性的关键作用。

---

## 局限性与未来方向

### 论文明确提到的

1. **缺少触觉反馈**: 遥操作时操作者无法感受手指接触力, 导致数据质量受限。加入 haptic feedback (如振动马达) 可显著提升遥操作体验和数据质量。
2. **无预训练**: 策略完全从头训练, 对场景外观变化敏感。训练更鲁棒、可泛化的策略是重要方向。

### 从代码推断的

3. **非拇指手指无独立控制**: power grasp 映射限制了精细 finger gaiting 能力, 这在需要 in-hand manipulation 的任务中会成为瓶颈。
4. **固定的数据归一化范围**: 手部关节的 min/max 硬编码 (`[110,110,110,110,90,120]`), 无法自适应不同场景或不同手部硬件。
5. **Temporal ensemble 模式有限**: 代码中实现了多种 ensemble 模式 (avg, act, new, old), 但论文只讨论了 avg, 其他模式的效果未被充分探索。
6. **IK failure fallback**: 当 IK 求解失败时直接使用上一帧关节位置, 可能在快速运动或奇异位形附近导致控制不连续。

---

## 论文与代码差异

### 1. 手部关节范围的 hack

代码中 `dp_agent.py` (第 171 行) 存在明显的工程 hack:

```python
# TODO: remove hack
self.hand_new_uppers = np.array([75] * 4 + [90.0, 120.0])
```

部署时, policy 输出的手指动作先用论文中的 `hand_uppers=[110,110,110,110,90,120]` 反归一化, 再用 `hand_new_uppers=[75,75,75,75,90,120]` 重新归一化。这意味着实际部署时非拇指手指的有效范围被缩小到 75 度而非 110 度。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/agents/dp_agent.py` 第 171, 260-276 行

### 2. 多种未在论文中讨论的 action 模式

代码支持三种 action 模式, 论文只讨论了绝对关节位置:

| 模式 | 标志 | 说明 |
|------|------|------|
| 绝对关节位置 | (default) | 论文描述的模式 |
| EEF delta | `predict_eef_delta` | 预测 EEF pose 的增量 |
| Joint position delta | `predict_pos_delta` | 预测关节位置的增量 |

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/pipeline.py` 第 74-75 行

### 3. Simple BC baseline

代码中实现了完整的非 diffusion BC baseline (`SimpleBCModel` + `without_sampling` flag), 论文未提及。这是一个 3 层 MLP (512 hidden) 直接回归动作序列的方案, 用于和 diffusion policy 对比。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/models.py` 第 415-433 行

### 4. DDIM scheduler 支持

代码支持 DDIM (Denoising Diffusion Implicit Models) 作为 noise scheduler 的替代, 论文只讨论了 DDPM。DDIM 可以用更少的推理步数达到类似效果。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/learner.py` 第 82-103 行

### 5. 丰富的数据增强选项

代码实现了多种论文未提及的数据增强:
- **Color jitter** (`color_jitter`): RGB 亮度随机扰动
- **Gaussian noise** (`img_gaussian_noise`): 图像高斯噪声
- **Patch masking** (`img_masking_prob`): 随机遮挡图像 patch (类似 MAE)
- **State noise** (`state_noise`): 对非图像状态添加高斯噪声
- **Touch binarization** (`binarize_touch`): 将触觉值二值化 (>1000 为接触)
- **Depth clipping** (`clip_far`): 基于深度值裁剪远处像素

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/pipeline.py` 第 86-89, 119-199 行

### 6. EEF 空间遥操作 agent

代码包含一个独立的 `quest_agent_eef.py`, 直接在 EEF task space 操作 (发送 pos + axis-angle 指令给 UR5e 的 servoL), 而非论文描述的关节空间 IK 方案。该 agent 使用不同的坐标变换矩阵 (quest2isaac, left2isaac, right2isaac), 暗示可能还有 Isaac Sim 集成。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/agents/quest_agent_eef.py`

### 7. Velocity IK 实现

代码中 `quest_agent.py` 包含一个完整的 velocity IK 实现 (基于 Jacobian 和 nullspace method), 作为 `use_vel_ik` 选项提供。这是论文提到的"第二种 IK 实现" (一阶近似), 但论文声称不使用此方案。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/agents/quest_agent.py` 第 48-84 行

### 8. 脚踏板控制与键盘触发

数据采集脚本使用 `pynput` 键盘监听 ('l' 和 'r' 键) 模拟脚踏板, 控制数据采集的开始和结束。这一操作流程在论文中未描述。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/run_env.py` 第 14-44 行

### 9. 部署任务特定的 grip range

部署脚本 (`gen_deploy_scripts.py`) 中对 pour 和 steak 任务自动设置 `ability_gripper_grip_range=75` (而非默认 110), 与 `dp_agent.py` 中的 `hand_new_uppers` hack 对应。这说明不同任务需要不同的手指活动范围。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/workflow/gen_deploy_scripts.py` 第 55-56 行

### 10. 快速相机 buffer

相机服务器有两个实现: `ZMQServerCamera` (同步) 和 `ZMQServerCameraFaster` (异步 buffer, 30Hz 刷新)。默认使用后者, 通过后台线程持续读取相机, 保证请求时立即返回最新帧, 避免相机读取延迟影响控制频率。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/camera_node.py` 第 73-117 行

### 11. 数据过滤逻辑

`data_processing.py` 中的 `iterate()` 函数会跳过 `activated` 状态为 `False` 的帧 (即操作者暂停时), 并过滤掉以 `failed`, `ood`, `ikbad`, `heated`, `stop`, `hard` 结尾的轨迹文件夹。这些数据质量控制逻辑在论文中未提及。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/data_processing.py` 第 37-38, 80-93 行

### 12. Cosine LR scheduler + warmup

训练使用 cosine learning rate schedule + 500 步线性 warmup, 论文只提到了 AdamW optimizer 的超参数, 未描述 LR schedule。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/learning/dp/learner.py` 第 116-123 行

### 13. RealSense depth 滤波链

代码对 depth 图像应用了 5 个滤波器: decimation -> disparity transform -> spatial filter -> temporal filter -> disparity transform (inverse) -> hole filling。这种重度预处理在论文中未讨论, 可能是 depth 效果不佳的原因之一 (过度平滑)。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/cameras/realsense_camera.py` 第 101-107 行

### 14. Robotiq gripper 和 Allegro 接口

UR robot driver 除了支持 Ability Hand, 还包含对 Robotiq gripper 和 Allegro Hand 的接口 (`ur.py` 第 126-128 行), 说明系统设计时考虑了更广泛的末端执行器兼容性。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/robots/ur.py` 第 26-36, 121-133 行

### 15. MuJoCo 用于 IK 而非仿真

代码中的 MuJoCo 模型 (`universal_robots_ur5e/`) 仅用于前向运动学和 IK 求解 (在 `quest_agent.py` 中), 不用于物理仿真或策略训练。

**文件路径**: `/home/l/ws/doc/paper/manip/QiHaoZhi/24_HATO/code/agents/quest_agent.py` 第 108-113, 262-270 行

---

## 跨论文比较

### HATO 在 QiHaoZhi 研究脉络中的定位

HATO 是 Haozhi Qi 系列工作中唯一一个完全基于真实世界 imitation learning 的双臂系统, 其遥操作系统和数据采集框架被后续工作直接复用。

### 与同系列工作的对比

| 维度 | HATO | TwistingLids | PenSpin | DexScrew | MinBC | AINA |
|------|------|-------------|---------|----------|-------|------|
| **年份** | 2024.4 | 2024.3 | 2024.7 | 2025.12 | 2025.12 | 2025.11 |
| **任务** | 4种双手日常 | 双手拧瓶盖 | 单手旋笔 | 螺母/螺丝刀 | 人形洗碗等 | 9种日常 |
| **手部** | 2x Ability (6 DOF) | 2x Allegro (32 DOF) | 1x Allegro (16 DOF) | 1x XHand (12 DOF) | 简化双通道 | 1x Psyonic (6 DOF) |
| **方法** | 真实数据 IL | Sim-to-Real RL | Oracle->BC->微调 | 简化仿真->技能遥操->触觉 BC | Choice Policy BC | 人类视频->等变策略 |
| **数据来源** | VR 遥操作 | 仿真 | 仿真+开环回放 | 技能辅助遥操作 | 遥操作 (复用 HATO) | 野外人类视频 |
| **触觉** | 关键 (验证) | 不使用 | Oracle 训练用 | 关键 (真实 BC) | 不使用 | 不使用 |
| **视觉** | RGB (-D), ResNet-18 | 2 点视觉 | RGB | Point cloud | RGB | Point cloud |
| **策略** | Diffusion Policy | PPO + 小 MLP | PPO->BC | PPO->DAgger->BC | Choice Policy | VN-GPT |

### 关键对比分析

**HATO vs TwistingLids**:
- 同为双臂操控, 但方法论完全不同: HATO 走 real-world imitation learning 路线, TwistingLids 走 sim-to-real RL 路线。
- TwistingLids 使用 Allegro Hand (每手 16 DOF, 独立手指控制), HATO 使用 Ability Hand (每手 6 DOF, power grasp 为主)。HATO 的手指自由度更低但带触觉。
- TwistingLids 不使用触觉 (仿真触觉 gap 太大), HATO 充分利用真实世界触觉。
- 这体现了 Qi 的系统性思考: 当任务需要触觉时, sim-to-real 路线受限 (触觉仿真不可靠), 不如直接在真实世界中采集含触觉的数据。

**HATO vs PenSpin**:
- PenSpin 是单手任务, HATO 是双手。PenSpin 使用 Allegro 的独立手指控制实现 finger gaiting, HATO 的 power grasp 无法做到。
- PenSpin 的策略仍需经过仿真 (Oracle->BC->开环回放微调), HATO 完全绕过仿真。
- PenSpin 发现初始状态设计比奖励更重要, HATO 则发现足够多的人类示教 (~200-300 条) 隐式覆盖了初始状态多样性。

**HATO vs DexScrew**:
- 最相似的工作: 都使用真实世界数据 + visuotactile sensing。
- DexScrew 发现触觉 + 时序历史强互补 (单独无用), HATO 发现触觉即使在 obs_horizon=1 时也有效, 但 DexScrew 的触觉角色更精细 (旋拧反馈 vs 抓取稳定性)。
- DexScrew 使用仿真 RL 策略作为遥操作原语, HATO 使用 VR 控制器直接遥操。DexScrew 的方案对操作者友好度更高 (不需要控制手指), 但需要预先训练好仿真策略。
- DexScrew 发现简化仿真足以提供有效运动原语, 这与 HATO 完全不用仿真形成互补: 任务越复杂, 仿真越难建模, 纯真实数据方案越有优势。

**HATO vs MinBC**:
- **MinBC 直接复用了 HATO 的遥操作系统**。MinBC 论文明确引用 HATO 的硬件和软件基础设施。
- MinBC 使用 Choice Policy 替代 Diffusion Policy, 解决了 diffusion 推理速度慢的问题, 同时避免了 mode averaging。
- MinBC 进一步扩展到人形全身操控, HATO 聚焦桌面双手操控。
- MinBC 不使用触觉, HATO 的消融实验为 MinBC 未来加入触觉提供了动机和基线。

**HATO vs AINA**:
- 都从人类演示中学习, 但数据来源不同: HATO 通过 VR 遥操作获取机器人端数据, AINA 从野外人类视频学习 (无需机器人端数据采集)。
- AINA 使用 3D 等变网络消除人/机器人 embodiment 差异, HATO 通过直接遥操作避免了这个问题。
- AINA 使用同型号手 (Psyonic Ability Hand), 但只使用单手。
- AINA 不使用触觉, 将力反馈列为未来方向, 而 HATO 已经验证了触觉的重要性。
- 从方法论演进角度看, HATO (机器人端遥操作数据) 和 AINA (人类端野外数据) 代表了两种互补的示教数据获取路径。

### HATO 的独特贡献

在整个 QiHaoZhi 研究脉络中, HATO 的独特价值在于:

1. **硬件基础设施**: 提供了一套完整的、可复用的双臂多指手遥操作系统, 被 MinBC 等后续工作直接采用。
2. **Visuotactile 的实证验证**: 通过系统的消融实验, 首次在双臂多指手场景下定量证明了触觉对成功率和鲁棒性的关键作用, 为后续工作 (如 DexScrew) 的触觉使用提供了经验基础。
3. **纯真实数据路线的可行性验证**: 证明了不需要仿真, 几百条遥操作示教数据就足以学习复杂的双手灵巧技能, 这为后续工作的数据采集策略提供了重要参考。
