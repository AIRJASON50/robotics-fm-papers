# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

CMU LeCAR Lab, 2025 | arxiv 2509.16757

## 1. 核心问题

现有人形机器人 motion tracking 方法只能跟踪自由空间运动（行走、跑步、跳舞），无法处理**接触密集的人-物交互**（开门、搬箱子、爬楼梯）。原因：
- 3D 人-物交互数据稀缺（相比纯运动数据）
- 接触丰富的任务对 RL 训练提出额外挑战（不完美参考轨迹下如何引导接触行为）

## 2. 方法概览

三阶段 pipeline：

```
RGB video ──→ Stage 1: retargeting ──→ Stage 2: RL co-tracking ──→ Stage 3: real deployment
             (GVHMR + LocoMujoco)     (robot + object joint tracking)   (zero-shot)
```

**核心思想**：把交互技能学习转化为 **robot-object co-tracking** 问题——同时跟踪机器人和物体的参考轨迹。

### 2.1 数据准备 (Stage 1)

从单目 RGB 视频中提取：
- 人体运动：GVHMR (pose estimation) → GMR/LocoMujoco (retargeting to G1)
- 物体轨迹：位置、朝向（铰接物体还有关节状态）
- 接触信号：binary $c_t \in \{0, 1\}$（是否应有接触）
- 参考接触点：$p^{contact}_t$（物体局部坐标系下的目标接触位置）

参考状态格式：
$$s_t^{ref} = (s_t^{robot}, s_t^{obj}, c_t)$$

### 2.2 三个关键设计

#### (1) Unified Object Representation

物体观测统一为：
- 物体 pose 相对于 robot root frame（空间不变性）
- 参考接触点 $p^{contact}$（也在 root frame 下）
- 不区分物体类型（铰接/刚体/固定基座/浮动基座），统一格式

Policy observation = proprioception + phase variable $\phi \in [0,1]$ + object state (root-relative)

#### (2) Residual Action Space

不直接输出目标关节角，而是输出基于参考轨迹的残差：

$$\theta_t^{target} = \theta_t^{ref} + a_t$$

- $\theta_t^{ref}$: 当前帧参考关节角（来自 retargeting）
- $a_t$: policy 输出的修正量

**关键优势**：当 episode 从跪姿等极端姿态初始化时，标准 policy 会立刻"弹回"站立姿态（因为探索以默认站姿为中心），导致失衡和无效样本。Residual action 让探索以当前参考姿态为中心，大幅提高样本效率。

#### (3) Unified Interaction Reward

$$R_{contact,i} = \exp\left(-\frac{\|p_{eef,i} - p_{target,i}\|_2}{\sigma_{pos}}\right) \cdot \min\left(\exp\left(\frac{\|F_{contact,i}\|_2 - F_{thres}}{\sigma_{frc}}\right), 1\right)$$

$$R_{interaction} = \frac{1}{N_c} \sum_{i=1}^{N_c} R_{contact,i} \cdot c_{t,i}$$

- position term: 末端执行器接近目标接触点
- force term: 接触力达到阈值但不超过上限
- 由 contact signal $c_t$ gating

**核心价值**：处理不完美参考轨迹。retargeting 得到的运动往往接触不精确或有穿透，interaction reward 允许 policy 偏离参考轨迹以实现正确的抓取/接触。

### 2.3 训练框架

- DeepMimic-style：random state initialization (RSI) 从参考轨迹随机帧初始化
- Phase variable $\phi$: 告诉 policy 当前在动作序列中的进度
- Tracking-error-based termination: 偏离参考过大则终止
- PPO 优化
- IsaacSim 仿真
- 每个技能单独训练一个 policy（one policy per skill）

## 3. 实验结果

### Real-world (Unitree G1)
| Task | Result |
|------|--------|
| Door open + traversal | 67 consecutive trips |
| Suitcase manipulation | 7 consecutive successful runs |
| Bread box carrying | 2 full trials |
| Foam mats relocation | Successful |
| Truman's Bow (multi-stage) | 3 continuous executions |

### Ablation 发现
1. **Interaction reward**: 多数任务中移除不影响最终成功率，但在参考轨迹不完美或需要精确接触定位时是关键的
2. **Residual action space**: 一致性地降低跟踪误差，对极端姿态（如跪姿）的学习不可或缺

## 4. 与 SONIC 的对比

| 维度 | HDMI | SONIC |
|------|------|-------|
| **核心任务** | Human-Object Interaction (loco-manipulation) | Motion Tracking (locomotion-centric) |
| **物体交互** | 显式建模物体状态 + 接触奖励 | 无物体交互 |
| **Policy 范围** | 每个技能一个 policy | 一个 universal policy |
| **Action space** | Residual（基于参考轨迹的修正） | Absolute（直接输出关节角） |
| **数据来源** | 单目 RGB 视频 | 大规模动捕 (700h, 170 subjects) |
| **数据规模** | 小（单视频级别） | 大（100M+ frames） |
| **仿真平台** | IsaacSim | IsaacLab |
| **Observation** | proprioception + phase + object state | proprioception + universal token |
| **多模态输入** | 无（单一参考轨迹） | 3种 encoder 支持多模态 |
| **Retargeting** | GVHMR + LocoMujoco（离线） | GMR + PyRoki（离线） |
| **Contact reward** | position + force gated by contact signal | 无专门 contact reward |
| **关键创新** | robot-object co-tracking with unified interaction reward | universal token space + data scaling |

## 5. 与 bh_motion_track 的对比

| 维度 | HDMI | bh_motion_track |
|------|------|-----------------|
| **操作对象** | 门/箱子/椅子/楼梯（全身） | 方块（双手灵巧操作） |
| **Action space** | Residual (ref + delta) | Hybrid: wrist residual + finger absolute |
| **Contact reward** | eef position + force, binary gating | 5-finger sensor matching, temporal dilation, split aggregation |
| **物体建模** | object pose in root frame | object pose + face orientation + anchor-relative fingertip |
| **Phase variable** | $\phi \in [0,1]$, sufficient for single-motion | 无 phase variable（用 3-frame history 替代） |
| **Curriculum** | 无 | GuidanceCurriculum (weld constraint decay) |
| **Termination** | tracking error + contact loss | object deviation + NaN |
| **一个 Policy** | 一个技能 | 一个轨迹（计划支持多轨迹） |

## 6. 关键洞察

1. **Residual action 的思路有参考价值**：bh_motion_track 的 wrist 控制已经是 residual（ref + delta），但 finger 是 absolute。HDMI 的消融实验证明 residual 对极端姿态的学习至关重要。

2. **Interaction reward 的设计简洁有效**：position + force 双项，由 contact signal 门控。相比 bh_motion_track 的多层 contact reward（touch/match/FP/temporal dilation），HDMI 的设计更简单但同样能处理不完美参考轨迹。

3. **局限性明确**：一个技能一个 policy，部署依赖 mocap（物体 pose 来自外部），不支持在线感知。这些限制使得 HDMI 更像一个"技能获取 pipeline"而非"通用控制器"。

4. **与 SONIC 互补**：SONIC 解决"如何做一个 universal motion tracker"，HDMI 解决"如何让 motion tracker 支持物体交互"。理论上可以在 SONIC 的 universal token space 中加入物体信息来统一两者。
