# MinBC (Choice Policy) - 论文笔记

**论文**: Coordinated Humanoid Manipulation with Choice Policies
**作者**: Haozhi Qi\*, Yen-Jen Wang\* (equal contribution), Toru Lin, Brent Yi, Yi Ma, Koushil Sreenath, Jitendra Malik (UC Berkeley)
**发表**: arXiv:2512.25072, 2025
**项目**: 见论文中引用
**代码**: https://github.com/x-robotics-lab/minbc.git

---

## 一句话总结

提出 Choice Policy -- 一种单次前向传播即可生成多个候选动作并学习评分选择的模仿学习方法，结合模块化遥操作接口，实现人形机器人全身协调操控（头-手-腿），在洗碗机装载和全身擦白板任务上显著优于 Diffusion Policy 和标准 BC。

---

## 核心问题

人形机器人在非结构化环境中执行复杂操控任务面临两个核心挑战:

1. **遥操作数据采集困难**: 人形机器人 40+ DoF (GR-1: 44 DoF, Star-1: 55 DoF)，同时控制头、手臂、手指、腿部极其困难。现有系统要么只控制上半身，要么缺乏主动头部控制，无法采集高质量的全身协调演示。

2. **策略学习中的多模态问题**: 遥操作数据固有地存在多模态性 -- 同一观测对应多种合理动作。Diffusion Policy 可以建模多模态但推理速度太慢（需迭代采样），无法满足实时操控需求；标准 BC 推理快但会将多模态平均化，产生次优动作。

**本质矛盾**: 推理效率与多模态表达能力之间的 trade-off。

---

## 方法概述

### 系统架构

整体系统包含两个部分:

```
[模块化遥操作接口] → 高质量演示数据 → [Choice Policy 学习] → 自主策略
```

### 模块化遥操作接口

将全身控制分解为四个功能模块，使用 VR 控制器 (Quest) 操作:

| 模块 | 控制方式 | 关键设计 |
|------|---------|---------|
| 手臂控制 | Trigger 按下时追踪 VR 控制器位姿变化 → IK 求解关节位置 | On-demand activation: 释放 trigger 时手臂冻结不动，避免漂移 |
| 手部控制 | Grip 按钮控制四指 (连续值), 摇杆控制拇指 | 降维但保留 power/precision grasp 分类 |
| 头部控制 (hand-eye) | 按钮触发头部追踪左手或右手 | 基于 arctan2 计算 yaw/pitch 角度，保持操控区域在视野内 |
| 行走控制 | 摇杆指定速度命令 → RL 行走策略 (100Hz) | 与拇指控制共享摇杆，按压切换模式 |

Head-eye coordination 的计算:
$$\text{yaw} = \arctan2(r_y, r_x), \quad \text{pitch} = \arctan2(-r_z, \sqrt{r_x^2 + r_y^2})$$
其中 $r = p_{\text{hand}} - p_{\text{head}}$。

### Choice Policy

**核心思路**: 受多选择学习 (multi-choice learning) 和 SAM 的启发，生成 K 个候选动作序列 + 学习评分选择最优。

**架构**: Feature Encoder → 两个并行网络:
- Action Proposal Network: 2-layer MLP, 输出 $K \times T \times |A|$ 维度
- Score Prediction Network: 2-layer MLP, 输出 $K$ 个分数

**训练目标**:

1. 计算每个 proposal 与 ground truth 的 MSE:
$$\ell^{(k)} = \frac{1}{|A||T|} \sum_{i,j} (a_t^{(k)}[i,j] - a_t[i,j])^2$$

2. Score loss -- 让分数预测网络回归真实 MSE:
$$\mathcal{L}_{\text{score}} = \frac{1}{K} \sum_{k=1}^K (\sigma_t^{(k)} - \ell^{(k)})^2$$

3. Winner-takes-all -- 只更新最佳 proposal:
$$k^* = \arg\min_k \ell^{(k)}, \quad \mathcal{L}_{\text{action}} = \ell^{(k^*)}$$

4. 总损失: $\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{score}}$

**推理**: 单次前向传播 → 选 $\arg\min_k \sigma_t^{(k)}$。

---

## 关键设计

### 1. Winner-Takes-All + Score Regression 的解耦训练

这是方法的核心创新。两个网络虽然共享 encoder feature，但有不同的训练信号:

- **Action Proposal Network**: 只有"赢家"接收梯度。不同 head 会自然分化，每个 head 专注于不同的行为模式（论文 Figure 5 可视化验证了这一点 -- 不同 head 在不同子任务阶段被选中）。
- **Score Network**: 回归所有 head 的真实 MSE，学习评估动作质量。

直觉解释: 类比 Mixture of Experts -- 每个 proposal head 是一个"专家"，score network 是"门控网络"。但不同于 MoE 的端到端训练，winner-takes-all 强制竞争，避免所有 head 坍缩为相同行为。

### 2. Hand-Eye Coordination 的系统级影响

论文中最令人印象深刻的发现之一: **没有 hand-eye coordination，所有方法在 insertion 阶段的成功率接近 0%**。

这不仅是一个"有用的特征"，而是一个**系统级必要条件**:
- 在 dishwasher loading 中，洗碗机不在初始视野内
- 手腕相机被盘子遮挡
- 只有头部主动跟踪手部，才能看到 dishrack 完成 insertion

这一发现凸显了全身协调（特别是 gaze control）在长时序操控中的不可替代性。

### 3. 模块化遥操作的抽象层次选择

将高维连续手指控制简化为 {grip button (4指), joystick (拇指)} 的离散-连续混合控制:
- 降低操作者负担 (< 10 分钟学会)
- 数据质量更高 (避免手指追踪的抖动)
- 保留核心抓取分类 (power grasp, precision grasp, flattening)
- 框架可扩展 -- 可以添加更复杂的手指技能作为新模块

---

## 实验

### 任务 1: Dishwasher Loading (100 demos, GR-1, 固定底座)

| 方法 | Pickup | Handover | Insertion |
|------|--------|----------|-----------|
| BC (w/ hand-eye) | 10/10 | 7/10 | 5/10 |
| DP (w/ hand-eye) | 10/10 | 7/10 | 5/10 |
| **Choice Policy (w/ hand-eye)** | **10/10** | **9/10** | **7/10** |
| BC (w/o hand-eye) | 10/10 | 5/10 | 0/10 |
| DP (w/o hand-eye) | 10/10 | 5/10 | 1/10 |
| Choice Policy (w/o hand-eye) | 10/10 | 6/10 | 0/10 |

关键观察:
- Choice Policy 在所有阶段都优于或持平 BC 和 DP
- 没有 hand-eye coordination，insertion 阶段几乎全部失败
- 性能瓶颈在 handover 和 insertion，pickup 所有方法都完美

### OOD 评估

| 设置 | CP Pickup/Handover/Insert | BC Pickup/Handover/Insert | DP Pickup/Handover/Insert |
|------|--------------------------|--------------------------|--------------------------|
| Color OOD (绿盘) | 10/9/5 | 10/6/4 | 10/6/4 |
| Position OOD | 8/6/2 | 6/4/0 | 7/3/0 |

Choice Policy 在 OOD 设置下依然保持最高鲁棒性。

### Ablation -- 动作选择策略

| 选择方式 | Pickup | Handover | Insertion |
|---------|--------|----------|-----------|
| Random Choice | 10/10 | 6/10 | 3/10 |
| Mean Choice | 8/10 | 4/10 | 0/10 |
| Single Best | 10/10 | 6/10 | 3/10 |
| Single Worst | 5/10 | 2/10 | 0/10 |
| **Score-based (ours)** | **10/10** | **9/10** | **7/10** |

Mean Choice 表现最差 -- 验证了多模态平均化的灾难性后果。

### 任务 2: Whiteboard Wiping (50 demos, loco-manipulation)

| 方法 | Head Move | Pickup | Walk | Wipe |
|------|-----------|--------|------|------|
| DP | 无法部署 (推理慢/训练不稳定) | - | - | - |
| BC | 5/5 | 2/5 | 2/5 | 2/5 |
| Choice Policy | 5/5 | 3/5 | 3/5 | 1/5 |

这是一个极度困难的任务: 初始姿态随机化 + 行走不确定性 + 全身精细调整。虽然整体成功率低，但验证了系统的可行性。Diffusion Policy 完全无法部署。

---

## 相关工作分析

### 在 Humanoid Manipulation 领域的定位

| 特征 | ExBody系列 | HumanPlus | TWIST/Sonic | HOMIE | AMO | **本文** |
|------|-----------|-----------|-------------|-------|-----|--------|
| 上半身操控 | - | 上下分离 | 上半身 | 上半身 | 全身(小型) | 全身 |
| 主动头部控制 | - | - | - | - | - | hand-eye |
| 灵巧手 | - | - | - | 简单夹爪 | - | Ability Hand/XHand |
| 自主策略 | - | 分离 | - | 分离 | 统一 | 统一 |
| 全尺寸人形 | 部分 | 部分 | 部分 | 部分 | 否(G-1) | 是(GR-1, Star-1) |

本文的独特贡献: **首个在全尺寸人形机器人上实现头-手-腿统一策略的全身操控系统**。

### 在 Policy Representation 领域的定位

| 方法 | 推理方式 | 多模态能力 | 推理速度 |
|------|---------|----------|---------|
| BC | 单次前向传播 | 差 (模式平均) | 快 |
| Diffusion Policy | K 步迭代采样 | 好 | 慢 |
| BET (tokenized) | 单次前向传播 | 中等 | 快 |
| **Choice Policy** | **单次前向传播** | **好** | **快** |

Choice Policy 在保持快速推理的同时有效捕获多模态，填补了 BC 和 DP 之间的空白。

---

## 局限性与未来方向

### 作者明确指出的局限

1. **视觉泛化能力有限**: 对显著不同的场景和物体泛化不足。可能的解决方案: 更多样化的训练数据或大规模预训练。
2. **Hand-eye coordination 依赖启发式**: 当前基于 arctan2 的几何计算，尚未学习化。更自适应的学习机制可能提升性能。

### 从代码推断的局限

3. **Action chunking 的时间设计**: 代码中 `pre_horizon=16, obs_horizon=1, act_horizon=8`。observation 只用单帧历史，可能丧失时序信息。
4. **数据归一化依赖 percentile**: 使用 2nd-98th percentile 归一化，对异常值敏感。clip 到 [-1.5, 1.5] 的选择也比较 ad hoc。
5. **部署时 action smoothing 是事后处理**: `closeloop.py` 使用 `MovingAverageQueue(tau=0.7)` 对预测动作做指数移动平均，但手指动作 (index 7:13, 20:) 绕过平滑直接执行 -- 这种手工区分暗示策略本身的平滑性不够。
6. **只有 50-100 条演示**: 对复杂长时序任务来说数据量偏小。Whiteboard wiping 只有 50 条演示时成功率不高。

---

## 论文与代码差异

### 1. 多种 Action Decoder 架构 (论文未详述)

论文只提到"2-layer MLP"作为 action proposal 和 score network。但代码实现了三种 decoder:

- **MLPDecoder**: 5 层 MLP (1024→1024→1024→1024→256) + 输出头。最简单的实现。
  - 路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/exp/action_decoder.py` (L6-48)
- **HourglassDecoder**: U-Net 风格的 1D Conv 网络 (无条件)，使用 ResidualBlock1D，down_dims=[256,512,1024]。
  - 路径: 同上 (L51-152)
- **CondHourglassDecoder**: 带 FiLM 条件注入的 U-Net。使用可学习的 `cls_token` 作为初始输入（形状 $[1, K, A, T]$），observation feature 通过 FiLM modulation 注入各层。
  - 路径: 同上 (L155-253)

README 中推荐使用 `--dp.action_decoder cond_hourglass`，表明 CondHourglassDecoder 是实际部署的架构。这与论文中"2-layer MLP"的描述差距很大。

### 2. Score Prediction 在 CondHourglassDecoder 中是 per-proposal 独立的

论文描述 score network 是一个共享的 MLP 输出 K 个通道。但在 `CondHourglassDecoder` 中:
```python
self.score_pred = nn.Linear(256 * self.pre_horizon, 1)  # 输出 1 个值
```
每个 proposal 独立通过 U-Net 后各自预测一个 score，而不是共享一个 score head。这意味着 score prediction 实际上是每个 proposal 的自评估，而非全局比较。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/exp/action_decoder.py` (L208)

### 3. 可学习 cls_token 初始化

`CondHourglassDecoder` 使用可学习的 cls_token 作为 U-Net 的输入:
```python
self.cls_token = torch.nn.Parameter(
    torch.randn(1, self.num_proposals, act_dim, self.pre_horizon)
)
```
这类似 ViT 的 [CLS] token 设计，但论文没有提及。每个 proposal 有独立的初始化 token，通过 FiLM modulation 注入 observation 信息。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/exp/action_decoder.py` (L210-213)

### 4. EMA (Exponential Moving Average) 模型

代码使用 `EMAModel(power=0.75)` 对训练权重做指数滑动平均，推理时使用 EMA 权重。论文未提及这一标准但重要的训练技巧。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/vanilla_bc.py` (L61-62)

### 5. Score Loss Clipping

代码中有 `clip_score_loss` 和 `clip_score_loss_max` 选项:
```python
if self.clip_score_loss:
    score_loss = torch.clip(score_loss, max=self.clip_score_loss_max)
```
默认 `clip_score_loss_max=10.0`。论文未提及这种稳定训练的技巧。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/vanilla_bc.py` (L158-159)

### 6. Dropout 机制

代码在 decoder 中支持 `last_dropout` 和 `cond_dropout` -- 对最后一层特征和 condition 分别应用 dropout。论文未讨论。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/exp/action_decoder.py` (L107-114, L214-221)

### 7. DINOv3 而非 DINOv2

论文提到使用"frozen DINOv3 feature encoder"作为 RGB encoder。代码中确实实现了 DINOv3 (dinov3_vits16)，通过 `torch.hub.load` 从本地加载。这比论文引用的原始 DINOv2 更新。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/simple_act_pred.py` (L103-115)

### 8. 部署时的 Action Smoothing 逻辑

`closeloop.py` 中的动作执行逻辑:
```python
action = action_queue.add(action)       # EMA 平滑
action[7:13] = copied_action[7:13]      # 手指动作绕过平滑
action[20:] = copied_action[20:]        # 右手手指也绕过
```
这表明手臂动作需要平滑（策略输出抖动），但手指动作不能平滑（会丢失抓取精度）。论文未讨论这种工程细节。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/tools/deploy/closeloop.py` (L72-78)

### 9. 代码库是跨项目共享的

README 和代码中多处引用 "screw_driver" 任务 (DexScrew 论文)。`setup.py` 中的包名是 `minbc_screw_driver`，`data_processing.py` 处理 "control" + "xhand_act" 格式的原始遥操作数据。这说明 MinBC 不仅用于本文的人形操控，还是 DexScrew (灵巧手拧螺丝) 的训练框架。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/setup.py`, `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/data_processing.py`

### 10. Temporal Transformer 选项

代码中保留了 TemporalTransformer 作为 temporal aggregation 的备选方案 (cls_token + Transformer Encoder)，但默认使用 concat。

路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/simple_act_pred.py` (L74-79), `/home/l/ws/doc/paper/manip/QiHaoZhi/25_MinBC/code/dp/models/block.py` (L6-57)

---

## 跨论文比较

### 与同作者 (Qi Haozhi) 其他工作的比较

| 维度 | PenSpin (2024) | TwistingLids (2024) | DexScrew (2025) | **MinBC/Choice Policy (2025)** |
|------|---------------|--------------------|-----------------|-----------------------------|
| 任务类型 | 单手手内旋转 | 双手拧瓶盖 | 单手拧螺母/螺丝刀 | 双臂全身操控 |
| 机器人 | LEAP Hand | 2x Allegro Hand | XHand + 7DoF Arm | GR-1/Star-1 全尺寸人形 |
| 学习范式 | Sim RL → Oracle BC → 真实微调 | Sim RL (sim-to-real) | Sim RL 原语 → 遥操作 → BC | 遥操作 → BC/Choice Policy |
| 仿真依赖 | 高 (三阶段都需要) | 高 (主要训练在仿真) | 中 (只需简化仿真) | 低 (仅行走策略在仿真训练) |
| 视觉输入 | 无 (proprioception only) | 2 个 3D 关键点 | 无 (proprioception + tactile) | RGB (head + 2 wrist) |
| 触觉 | 无 | 无 | 关键 (1800 维) | 无 |
| 多模态处理 | 不涉及 | 不涉及 | 标准 BC | Choice Policy |
| 全身协调 | 无 | 无 | 无 | 头-手-腿统一 |
| 关键代码共享 | 独立 | 独立 | **共享 MinBC 框架** | MinBC 框架 |

**技术演进路线**: 从单手灵巧操控 (PenSpin/TwistingLids) → 手臂+手指协调 (DexScrew) → 全身协调 (MinBC)。MinBC 的代码库是 DexScrew 的直接扩展（共享训练框架），反映了作者组在 imitation learning infrastructure 上的积累。

### 与同批次其他论文的比较

| 维度 | **MinBC/Choice Policy** | HandelBot (2026) | AINA (2025) |
|------|------------------------|-----------------|-------------|
| 核心问题 | 人形全身操控 + 多模态BC | 双手钢琴演奏 (毫米精度) | 从人类视频学灵巧操控 |
| 数据来源 | VR 遥操作 | 仿真 RL + 真实微调 | 人类佩戴智能眼镜录制 |
| 策略学习 | Choice Policy (BC变体) | Residual RL | 3D point-based imitation |
| 学习方法创新 | Winner-takes-all + score regression | Sim RL → structured refinement → residual RL | 人类手部关键点 → 机器人关键点映射 |
| Sim-to-real | 仅行走策略 | 核心挑战 (毫米级 gap) | 完全不需要仿真 |
| 机器人数据需求 | 50-100 条遥操作 | 30 分钟真实交互 | 0 (仅人类视频) |
| 手部 DoF | 6 DoF (Ability Hand/XHand) | 高 DoF (多指) | 高 DoF (多指) |
| 推理速度 | 单次前向传播 (实时) | RL 策略 (实时) | 单次前向传播 |
| 全身协调 | 头-手-腿 | 双手手指 | 单臂+手 |
| 主要贡献类型 | 系统+算法 | 算法 (sim-to-real 适应) | 数据+系统 (无需机器人数据) |

**互补关系**:
- MinBC 提供了一个通用的 BC 训练框架，DexScrew 已经复用它，未来 HandelBot/AINA 类工作也可能在其上构建
- AINA 完全消除了对遥操作的依赖，但目前只验证在桌面操控上；MinBC 的遥操作系统在全身操控中仍然必要
- HandelBot 的 residual RL 微调策略可能与 Choice Policy 结合 -- 先用 Choice Policy 从演示中学粗略策略，再用 residual RL 精细化
