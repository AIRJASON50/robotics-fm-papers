# HandelBot - 论文笔记

**论文**: HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies
**作者**: Amber Xie (Stanford University), Haozhi Qi (Amazon FAR), Dorsa Sadigh (Stanford University)
**发表**: arXiv:2603.12243, 2026
**项目**: https://amberxie88.github.io/handelbot
**代码**: https://github.com/amberxie88/handelbot.git

---

## 一句话总结

HandelBot 提出了一个三阶段框架 (sim RL + structured refinement + residual RL)，通过仅 30 分钟的真实交互数据将仿真训练的钢琴弹奏策略适配到真实世界，实现了首个基于学习的双手机器人钢琴演奏系统。

---

## 核心问题

钢琴弹奏要求毫米级精度的手指控制和严格的时序协调，是 dexterous manipulation 中最具挑战性的任务之一。现有方法面临三重困难：

1. **数据瓶颈**: 遥操作无法完成快速独立手指运动，人体数据存在 embodiment gap
2. **Sim-to-Real Gap**: 仿真中的策略无法直接迁移到真实钢琴，因为毫米级误差即导致按错键
3. **高维动作空间**: 双手 10 根手指 (实际使用 6 根) 需要协调控制，传统 RL 从头学习效率极低

---

## 方法概述

HandelBot 将问题分解为三个阶段的 pipeline：

### Stage 0: RL in Simulation

在 ManiSkill 仿真器中使用 PPO 训练策略 $\pi_{sim}$：

- **机器人**: 两个 Tesollo DG-5F 五指手 + Franka Panda 机械臂
- **观测空间**: 机器人本体感知 + 当前钢琴激活状态 + 目标音符 (lookahead 10 步) + active fingers
- **动作空间**: delta joint positions (每手指 3 个关节，第 4 关节固定为 1 radian)
- **末端执行器**: scripted trajectory，根据乐谱计算手腕轨迹 (X/Y 位置)
- **奖励函数**:
  - Key Press Reward: $0.7 \cdot \frac{1}{K}\sum_i g(\|k_s^i - 1\|_2) + 0.3 \cdot (1 - \mathbf{1}_{\text{false positive}})$
  - Fingering Reward: tolerance-based distance reward (Gaussian sigmoid)
  - Action L1 penalty

训练后选取 F1 score 最高的轨迹作为 open-loop trajectory $\tau_{sim}$。

### Stage 1: Policy Refinement

在真实环境中执行 $\tau_{sim}$，利用 MIDI output 比较期望按键与实际按键，迭代调整手指 lateral joint：

- **核心算法**: 对每个手指，计算 signed directional error $\Delta_t$，调整横向关节
- **Chunked Updates**: 将轨迹分为 sub-chunks (长度 K=2)，每 chunk 统一调整，带 lookahead L=6
- **Annealing**: 每次迭代递减 $\delta$ (衰减率 0.94)，避免振荡
- **邻居传播**: 对相邻手指施加 $0.3\Delta_t$ 的修正，减少自碰撞
- **Momentum Damping**: 方向翻转时衰减 0.7

### Stage 2: Residual Reinforcement Learning

在 refined trajectory $\tau^*$ 基础上学习残差策略 $\pi_{res}$:

$$\hat{s}_{t+1} = \pi_{res}(o_t) + s^*_{t+1}$$

- **算法**: TD3 (actor-learner 分离架构)
- **动作空间**: 限制为 3 个活跃手指 (食指、中指、无名指)，每手 9 维
- **左右手独立训练**: 减少维度，简化 credit assignment
- **Chunk size 2**: 每个 residual action 重复两次 (5Hz residual over 10Hz base)
- **Guided Noise**: 以 $P=0.5$ 的概率将噪声方向调整为正确横向移动方向
- **Correlated Noise**: $\hat{\epsilon} = \beta \cdot \epsilon_{prev} + \sqrt{1-\beta^2} \cdot \epsilon$, $\beta=0.2$

---

## 关键设计

### 1. Sim-to-Real 分解策略: Simulation as Structural Prior

核心洞察: 仿真擅长提供 finger coordination 的结构性先验 (哪根手指在什么时候移动)，但无法精确建模接触动力学。因此将仿真策略提取为 open-loop trajectory，而非直接做 closed-loop sim-to-real transfer。

论文验证了这一点: closed-loop $\pi_{sim}$(CL) 性能远差于 open-loop $\pi_{sim}$，因为动力学 gap 导致 compounding error。甚至 hybrid execution (并行仿真提供观测) 也无法弥补这一差距。

### 2. Structured Policy Refinement

这是一个利用任务领域知识的轻量级修正步骤。关键在于:

- **仅调整 lateral joint**: 钢琴弹奏中，横向偏移是 sim-to-real gap 的主要表现
- **基于 MIDI 反馈**: 使用钢琴的 MIDI output 作为 ground truth，无需视觉或复杂传感
- **可解释性强**: 通过比较 target key vs pressed key 直接推断修正方向

这步是 system identification 的结构化版本，不是学习过程，而是利用人类先验的几何修正。

### 3. Real-World Residual RL with Safety Layer

残差 RL 的设计有几个重要考量:

- **安全层 (PyRoki IK)**: 每步通过 constrained IK 求解可行关节配置，惩罚自碰撞和钢琴表面穿透
- **动作插值**: 10Hz 策略输出 -> 80Hz 线性插值发送给手部，减少抖动
- **Key-on curriculum**: reward 中 key_on 系数随训练进度变化 (代码中 `key_on_curriculum`)
- **Evaluation protocol**: 每 20 个 trajectory 评估 5 次，取最大 validation F1

---

## 实验

### 主要结果

在 5 首歌曲上评估 (F1 score x 100):

| 方法 | Twinkle | Ode to Joy | Hot Cross | Prelude C | Fur Elise |
|------|---------|------------|-----------|-----------|-----------|
| $\pi_{sim}$(CL) | 最低 | 最低 | 最低 | 最低 | 最低 |
| RL from Scratch | 中等 | 中等 | 中等 | 中等 | 中等 |
| $\pi_{sim}$ (open-loop) | 低 | 低 | 低 | 低 | 低 |
| $\pi_{sim}$ + ResRL | 中高 | 中高 | 中高 | 中高 | 中高 |
| HandelBot w/o ResRL | 高 | 高 | 高 | 高 | 高 |
| **HandelBot** | **最高** | **最高** | **最高** | **最高** | **最高** |

- HandelBot 在所有 5 首歌曲上均取得最佳 F1 score
- 相比直接仿真部署 ($\pi_{sim}$) 提升 1.8x
- 仅需 30 分钟真实交互数据 (短曲)，最多 1 小时 (长曲)
- 约 30k 环境交互步

### 消融实验发现

1. **初始化质量递减效果**: refined trajectory > $\pi_{sim}$ > random，更强的初始化 -> 更小的探索空间 -> 更高效的训练
2. **Discount factor**: $\gamma=0.8$ > $\gamma=0.99$，低折扣导致更抖的动作
3. **Guided noise**: $P=0.5$ 略等于无 guided noise，但 $P=1.0$ (always) 导致性能下降 (探索偏差)
4. **Closed-loop vs Open-loop**: open-loop 显著优于 closed-loop，closed-loop 因 dynamics gap 和 compounding error 失败
5. **Hybrid execution**: 比 closed-loop 好，但远不及使用真实数据的方法

---

## 相关工作分析

HandelBot 位于 dexterous manipulation、robotic piano playing 和 real-world RL 的交叉领域。

**与 RoboPianist (Zakka et al., CoRL 2023) 的关系**: RoboPianist 是仿真中的钢琴弹奏 RL benchmark，HandelBot 继承了其 reward design、F1 evaluation protocol 和 piano URDF。但 RoboPianist 仅停留在仿真层面，HandelBot 将其扩展到真实世界。

**与 Zeulner et al. (2025) 的关系**: 这是近期最接近的工作，实现了 hybrid transfer 的单手钢琴弹奏。HandelBot 的创新在于: (1) 双手弹奏 (2) 两阶段适配方法 (3) 完整的 residual RL 框架。

**独特性**: 据作者所知，HandelBot 是首个基于学习的真实世界双手钢琴弹奏系统。其方法论贡献是将 sim-to-real 问题分解为 "structured refinement + residual RL"，这个范式可泛化到其他高精度 dexterous 任务。

---

## 局限性与未来方向

### 作者明确提出的局限

1. **Scripted end-effector**: 末端执行器轨迹固定方向，需要手动调参，限制了拇指和小指的使用
2. **仅 3 指弹奏**: 由于物理约束只使用食指/中指/无名指，限制了可演奏曲目的复杂度
3. **Policy refinement 依赖领域知识**: lateral joint correction 是钢琴特有的启发式，不直接适用于其他任务
4. **Per-song training**: 每首歌需要独立训练，无泛化能力

### 从代码推断的局限

5. **硬编码常量**: 代码中大量硬编码的位置偏移 (`x_off_left=0.022`, `z_off_right=0.027` 等)，说明真实部署需要大量手动标定
6. **Song-specific 调整**: `"elise"` 歌曲需要特殊的臂部轨迹缩放 (`*= 1.5`)，表明方法对不同曲目的泛化需要手工干预
7. **安全约束**: collision checker 使用半平面近似钢琴表面 (`HalfSpace.from_point_and_normal`)，对复杂几何场景不适用

### 未来方向

- 学习 end-effector 运动 (而非 scripted)，允许旋转和更灵活的手指使用
- 使用 VLM 辅助 policy refinement，使其适用于其他任务
- 探索 multi-song 训练或 meta-learning 以提高泛化性

---

## 论文与代码差异

### 1. Reward 结构差异

论文描述的 reward 为 key press reward + fingering reward + energy penalty。代码中 (`/home/l/ws/doc/paper/manip/QiHaoZhi/26_HandelBot/code/envs/piano/rewards.py`)：

- **KeyPressReward**: 论文提到 key_on 系数为 0.7/0.3 (Appendix)，但代码默认 `key_on=0.5`，作为可调参数 (`args.key_on`)
- **Action L1 penalty**: 论文说替换了 RoboPianist 的 "power penalty"，代码中默认系数 `coef_action_l1=0.01`
- **Critic output scaling**: TD3 的 QNetwork 输出经过 `(tanh(x) + 1) * 6` 缩放，手动设定 reward 范围为 [0, 12]，论文未提及

### 2. 第 4 关节固定

论文 Appendix 提到固定每根手指的最后一个关节为 1 radian。代码中 (`setup_trajectories` in `handelbot_resrl_actor.py`) 验证了这一点：15 DoF -> 20 DoF 的转换中添加 `np.ones` 作为第 4 关节。但论文未详细说明这一设计的原因是手指末端较窄，固定弯曲可确保 fingertip 按键。

### 3. 真实世界 TD3 的额外细节

代码 (`/home/l/ws/doc/paper/manip/QiHaoZhi/26_HandelBot/code/real/src/dg5f_driver/script/handelbot_resrl_learner.py`) 实现了论文未详述的细节：

- **Multiple critics**: `n_critics` 个 Q-network (不止 TD3 标准的 2 个)，随机采样 2 个计算 target
- **Reward 分解**: rewards 维度为 2 (`rewards[:, 0]` 和 `rewards[:, 1]`)，按 `key_on` 加权组合，对应 correct press 和 no false positive 两部分
- **Key-on curriculum**: `key_on_curriculum` 允许训练过程中动态调整 key press 和 false positive 的权重比例
- **UTD (Update-to-Data) ratio**: `rb_left.n_inserts * args.utd * args.chunk_size < global_update` 控制更新频率

### 4. Policy Refinement 的额外机制

代码 (`/home/l/ws/doc/paper/manip/QiHaoZhi/26_HandelBot/code/real/src/dg5f_driver/script/handelbot_refinement.py`) 中有论文未详述的：

- **Chunk-level annealing**: `use_chunk_annealing` 选项允许每个 chunk 独立衰减 (成功 chunk 衰减快 *0.7，失败慢 *0.97)，论文仅描述全局 annealing
- **Momentum damping**: `use_momentum_damping` 选项，当修正方向翻转时衰减 0.7，防止振荡
- **Best-chunk 更新**: `update_best_residuals` 按 chunk 级别比较 reward 是否改善，只更新改善的 chunk

### 5. Actor-Learner 架构

代码使用 ZMQ socket (`tcp://127.0.0.1:5588`) 实现 actor-learner 分离架构 (inspired by SERL)，论文仅简要提及。actor 以 batch 形式通过 ZMQ 发送 transitions，learner 异步训练。支持 PAUSE/RESUME 信号控制训练节奏。

### 6. Guided Noise 实现

论文描述 guided noise 为调整噪声的符号方向。代码中实现更复杂：
- `get_guided_action`: 基于上一 episode 的按键结果计算修正 action (不仅是符号)
- `get_guidance_sign`: 仅返回符号方向 (+1/-1)，用于 `Actor.get_guided_action` 中调整 correlated noise 的绝对值方向
- 实际使用的是 `policy_stages.GUIDED_NOISE` 模式，交替于 SAMPLE 模式

### 7. 手动标定与 Song-specific 调整

代码中存在大量手工调整：
- 每首歌的 horizon 不同 (Twinkle: 160, Ode: 330, Fur Elise: 320)
- Hot Cross Buns 使用 `control_scale=2`，Prelude in C 使用 `factor=2` 重复音符
- Fur Elise 的臂部轨迹需要特殊缩放 (`*= 1.5`) 和 z 轴补偿
- 末端执行器位置偏移对左右手分别硬编码

### 8. Observation 空间

论文提到 observation 包含 robot proprioception + piano activation + goal + active fingers。代码中 (`_get_obs_extra`) 确认了这些，但 proprioception 的具体内容 (`get_proprioception`) 包括: ee_pose (position + euler) + 手部 qpos (joint 7+)，论文未明确说明。

---

## 跨论文比较

### 与 Haozhi Qi 同作者论文比较

| 维度 | PenSpin (CoRL 2024) | TwistingLids (2024) | DexScrew (2025) | HandelBot (2026) |
|------|---------------------|---------------------|-----------------|------------------|
| 任务 | 笔旋转 | 拧瓶盖 | 拧螺丝 | 双手钢琴弹奏 |
| 手数 | 单手 | 单手 | 单手 | 双手 |
| Sim-to-Real | rapid motor adaptation | sim-to-real + 人类演示 | sim-to-real | sim RL + refinement + residual RL |
| 真实 RL | 无 | 无 | 无 | 有 (TD3 residual) |
| 精度要求 | 中等 | 中等 | 高 | 极高 (毫米级) |
| 传感 | 触觉/本体感知 | 视觉/本体感知 | 触觉/视觉 | MIDI output + 本体感知 |
| 核心贡献 | 快速电机适配 | 灵巧操作 pipeline | 螺丝操作 | structured refinement + residual RL |

HandelBot 的独特之处在于: (1) 首次在真实世界做双手灵巧任务的 residual RL (2) 利用任务特定信号 (MIDI) 作为 reward (3) 三阶段适配范式。

### 与同批次论文比较

| 维度 | AINA (2025) | MinBC (2025) | HandelBot (2026) |
|------|-------------|--------------|------------------|
| 方法类型 | Imitation Learning | Behavior Cloning | RL (sim + real) |
| 是否使用仿真 | 否 | 是 (数据生成) | 是 (策略训练) |
| 真实世界学习 | 无 (纯模仿) | 无 (纯 BC) | 有 (residual RL) |
| 数据需求 | 大量人类演示 | 最小化 BC 数据 | 30 min 交互 |
| 任务泛化性 | 多任务 | 多任务 | 单曲/单任务 |
| 精度要求 | 中等 | 中等 | 极高 |
| 关键思想 | 模仿学习的效率 | 最少数据量的 BC | 仿真结构先验 + 真实微调 |

HandelBot 与 MinBC 代表了两种不同的数据效率范式: MinBC 通过最小化行为克隆数据量实现泛化；HandelBot 通过仿真预训练 + 少量真实交互实现单任务极致精度。两者互补: MinBC 适合多任务场景，HandelBot 适合需要极高精度的特定任务。
