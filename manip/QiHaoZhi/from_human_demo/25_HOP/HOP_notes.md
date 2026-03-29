# HOP - 论文笔记

**论文**: Hand-Object Interaction Pretraining from Videos
**作者**: Himanshu Gaurav Singh, Antonio Loquercio, Carmelo Sferrazza, Jane Wu, Haozhi Qi, Pieter Abbeel, Jitendra Malik (UC Berkeley)
**发表**: arXiv:2409.08273, ICRA 2025
**项目**: https://hgaurav2k.github.io/hop/
**代码**: https://github.com/hgaurav2k/hop.git

---

## 一句话总结

从 in-the-wild 人类操控视频中提取 3D hand-object 交互轨迹，重定向到机器人 embodiment，用 GPT-2 风格的 autoregressive transformer 进行 generative pretraining，得到 task-agnostic 的操控先验 (manipulation prior)，该先验可通过 RL 或 BC 微调高效适配下游任务。

---

## 核心问题

如何从大规模人类视频中学习可复用的 sensorimotor 操控表示 (manipulation prior)？

现有方法存在两个根本性困难：

1. **视频缺乏运动信号**: 视频只是观测序列，不包含动作、力、触觉等运动信息。之前的方法 (R3M, VIP, MVP) 只提取视觉表示 (visual representation)，忽略了运动组件 (motor component)
2. **Human-robot embodiment gap**: 人手与机器人手的运动学差异巨大，人类动作无法直接在机器人上执行

HOP 的核心贡献在于构建了一条完整的 pipeline：从 in-the-wild 视频出发，经过 3D lifting、retargeting、physics-based 优化，最终生成物理合理的机器人感觉运动轨迹 (sensorimotor trajectories)，并在此数据上训练一个既包含 perception 也包含 motor 组件的 base policy。

---

## 方法概述

### 整体 Pipeline

```
In-the-wild Videos → 3D Lifting (MCC-HO) → Retargeting (IK优化) → Robot Trajectories
                                                                      ↓
                                                       GPT-2 Pretraining → π_b (base policy)
                                                                      ↓
                                                           RL/BC Finetuning → π_task
```

### 3.1 从视频到 3D: Lifting Hand-Object Interactions

- 数据来源: 100 Days of Hands (100DOH) + Epic Kitchens + DexYCB
- 使用 HaMeR 检测手部 3D 几何，MCC-HO 联合推断 hand-object 点云
- 关键简化: 放弃 CAD model 的 pose refinement，接受较低的重建质量换取通用性
- 时序平滑: 将物体重建锚定到时间平滑的手部检测上
- 假设静态相机
- 总计约 450 个视频

### 3.2 重定向到机器人: Retargeting via IK Optimization

目标: 找到机器人关节轨迹 $q_t$，使末端执行器跟踪人手轨迹：

$$q^* = \arg\min_q \sum_t \|f(q_t) - x_t\|^2 + \lambda R(q_t, q_{t-1})$$

其中 $f$ 是正运动学，$x_t$ 是目标手指位姿，$R$ 是正则化项。

关键创新: **环境随机化 (scene randomization)**。在场景中随机放置桌子、墙壁等障碍物，增加 IK 优化问题的约束，从而增加生成轨迹的多样性。这对有运动学冗余的机器人 (7 DoF 手臂 + 16 DoF Allegro 手) 尤为重要。

多次运行优化 + 基于碰撞和跟踪误差的轨迹筛选 → 保留高质量数据。最终生成约 70,000 条轨迹。

### 3.3 Generative Pretraining

训练目标: 在轨迹数据集 $\mathcal{T}$ 上进行 next-token prediction：

$$\pi_b = \arg\min_\pi \sum_{(s,a) \in \mathcal{T}} \mathcal{L}(\pi(a_t | s_{t-k:t}), a_t)$$

模型架构: GPT-2 style causal transformer
- 输入: 过去 16 步的 proprioception + object point cloud
- 输出: 预测下一步的 action (joint positions)
- 损失: L1 Loss

### 3.4 下游微调

- **RL 微调**: 用 pretrained transformer 初始化 actor，PPO 训练
- **BC 微调**: 在有限真实世界演示 (15-50 条) 上微调

---

## 关键设计

### 1. Sensorimotor Pretraining vs Visual Pretraining

HOP 的核心论点: 之前的 visual pretraining (R3M, VIP, MVP) 只学习 "看" (perception)，不学习 "做" (motor)。HOP 通过在重定向后的机器人轨迹上做 autoregressive pretraining，同时学习了：
- **Object affordance**: 在哪里、如何抓取
- **基础物理直觉**: 先靠近再抬起
- **Wrist-hand coordination**: 手腕运动和手指塑形的协调

这些是 motor skill 的组成部分，纯视觉表示无法捕获。

### 2. Point Cloud + Proprioception 的多模态输入

模型输入包含两种模态：
- **Proprioception** (23 维): 7 DoF xArm 手臂 + 16 DoF Allegro 手的关节位置
- **Object point cloud** (100 x 3): 物体表面采样点

这种设计绕过了 RGB 图像在 sim-to-real 中的 domain gap 问题。Point cloud 提供了物体形状和位姿的几何表示，对外观变化更鲁棒。

PointNet 编码器 (3 层 MLP + MaxPool) 将 100 个 3D 点压缩为单个 embedding，与 proprioception embedding 交错 (interleave) 送入 causal transformer。

### 3. 两种微调范式的互补验证

HOP 同时验证了 RL 和 BC 两种微调方式：
- **RL 微调 (仿真)**: 用 pretrained weights 初始化 PPO actor → 更好的初始探索 → 更快收敛
- **BC 微调 (真实世界)**: 在 15-50 条演示上微调 → 比 visual pretraining baselines 泛化更好

这说明 learned prior 的价值不依赖于特定的微调算法。

---

## 实验

### 真实世界 BC 微调 (Section 5.1)

三个任务，每个 20 次评估：

| 方法 | Grasp & Drop | Grasp & Pour | Grasp & Lift (4 objects) |
|------|-------------|-------------|------------------------|
| R3M | ~高 | ~高 | ~低 |
| VIP | ~高 | ~高 | ~低 |
| MVP | ~高 | ~高 | ~低 |
| ImageNet ZS | ~高 | ~高 | ~低 |
| ImageNet F | ~高 | ~高 | ~中 |
| Diffusion Policy | ~高 | ~高 | ~中 |
| **HOP (ours)** | **~高** | **~高** | **最高 (+30%)** |

关键发现: 在简单单物体任务上所有方法表现相当；在多物体 (4 种不同形状) 的 Grasp & Lift 任务上，HOP 比最佳 baseline (ImageNet Finetuned) 高出约 30 个百分点。

### 仿真 RL 微调 (Section 5.2)

三个任务: Grasp & Lift, Grasp & Throw, Open Cabinet

**与 demo-guided RL baselines 的对比** (DAPG, DexVIP, DexMV, PPO from scratch):
- HOP 在所有任务上收敛最快
- 在 Grasp & Throw 上优势最大 (行为与预训练数据最不相关 → 说明 prior 的泛化能力)
- PPO from scratch 在 Open Cabinet 上完全不收敛

**鲁棒性和泛化性**:
- **OOD 物体**: HOP 微调的策略在未见过的物体上表现更优
- **扰动恢复**: HOP 策略面对外力扰动后能更好地恢复
- **Human-like affordance**: HOP 倾向于学习更像人类的抓取姿态

### Hand-only Prior vs Hand-Object Prior (Section 5.3)

与只学手部运动先验 (无物体信息) 的 baseline 对比:
- 加入物体点云信息后收敛显著更快
- 说明 object affordance 信息是 manipulation prior 的重要组成部分

---

## 相关工作分析

HOP 处于 "learning from videos" 和 "dexterous manipulation" 的交叉领域：

| 方向 | 代表工作 | 与 HOP 的区别 |
|------|---------|-------------|
| Visual pretraining | R3M, VIP, MVP | 只学视觉表示，无运动组件 |
| Demo-guided RL | DAPG, DexMV | 需要机器人遥操作数据 |
| Hand pose prior | DexVIP, ManipTrans | 只学手部姿态，忽略物体 |
| Video retargeting | DexMV, H2R | 针对特定任务，非 task-agnostic |
| Foundation policy | RT-1/2, Octo | 需要大量机器人数据 |

HOP 的独特性在于: (1) 从 in-the-wild 视频而非机器人数据出发; (2) 同时学习 perception 和 motor 组件; (3) 生成 task-agnostic 的 base policy 而非 task-specific 的策略。

---

## 局限性与未来方向

### 论文明确提出的局限

1. **3D 重建质量**: 放弃 CAD refinement 后点云质量降低，特别是遮挡严重的情况
2. **静态相机假设**: 限制了可用的视频数据范围
3. **物理不真实**: retargeting 时忽略物体动力学，生成的轨迹在物理上不完全合理
4. **规模有限**: 70,000 条轨迹相比 NLP/CV 的预训练数据集仍然很小

### 从代码推断的局限

5. **仅限 xArm + Allegro**: 代码中硬编码了 23 DoF (7 arm + 16 hand) 的关节限位，迁移到其他机器人需要重新 retargeting
6. **Point cloud 固定 100 点**: 代码中 `pc_num=100` 硬编码，对复杂物体的表示能力有限
7. **无 vision encoder**: 预训练阶段不使用 RGB/depth 图像，仅用点云。虽然论文说 BC 微调时加了 depth encoder，但预训练的 prior 本身不包含视觉信息
8. **IsaacGym 依赖**: 整个训练框架建立在 IsaacGym 上，目前 NVIDIA 已停止维护该项目

### 未来方向

- 扩大预训练数据规模 (更多视频来源)
- 使用更先进的 3D 重建方法提高轨迹质量
- 扩展到双手操控
- 与 language conditioning 结合实现 task specification

---

## 论文与代码差异

### 1. 多头预测: 不只预测 action

论文主要描述 action prediction 作为训练目标。代码中 `RobotTransformerAR.forward()` 实际输出三种预测：
- `action`: 预测下一步动作 (论文描述的)
- `next_proprio`: 预测下一步本体感知状态 (论文未详述)
- `pc`: 预测下一步物体点云 (论文未详述)

配置中 `use_pc_loss` 和 `use_proprio_loss` 控制是否启用辅助损失。默认配置中两者均为 `False`，但代码基础设施支持多任务预训练。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/pretrained/robot_transformer_ar.py` (L158-180)

### 2. Data-driven noise 的硬编码统计量

论文提到对 proprioception 和 action 添加噪声用于数据增强。代码中 `Trainer` 类包含一套硬编码的 per-joint 噪声统计量 (`data_driven_noise`)，包含每个关节的 mean、max、min、std。这些是从真实数据中预计算的，论文完全没有提及。

默认使用简化的 uniform noise (`noise_arm=0.1`, `noise_hand=0.1`)，但 `add_data_driven_noise` 选项允许使用更精细的 per-joint noise schedule。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/pretrained/trainer.py` (L60-68)

### 3. IG (IsaacGym) 关节顺序重映射

代码中 `RobotDataset.change_order()` 执行一个固定的关节索引重排序:

```python
IG_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 11, 12, 13, 14, 15, 16, 17, 18]
```

这将 retargeting 数据中的关节顺序映射到 IsaacGym 环境中的顺序。论文未提及这种 index mapping 的存在，但它是代码正确运行的必要条件。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/pretrained/robot_dataset.py` (L88-90)

### 4. L1 Loss 而非 MSE

论文描述的是一般性的 loss function。代码中 `pretrain.py` 显式使用 `torch.nn.L1Loss()` 而非 MSE:

```python
loss_fn = torch.nn.L1Loss()  # torch.nn.MSELoss()
```

注释中保留了 MSELoss 的痕迹，说明曾经尝试过 MSE 但最终选择了 L1。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/scripts/pretrain.py` (L138)

### 5. Critic 独立于 Actor 的 value network 设计

论文没有详述 RL 微调时的 value function 设计。代码中 `RTActorCritic` 使用独立的 4 层 MLP (512→256→128→1) 作为 value network，而 actor 使用 pretrained transformer。关键设计选择:

- `value_grads_to_pointnet=False`: 默认不将 value function 的梯度传播到 PointNet 编码器 (finetune 脚本中设置)
- `point_cloud_input_to_value=True`: value function 接收 PointNet 编码的点云特征 (替换原始点云维度)
- `critic_warmup_steps=200`: 先单独训练 critic 200 步，再联合训练

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/models/rt_actor_critic.py` (L44-58)

### 6. 分离的 arm/hand 探索噪声初始化

微调脚本中 `initEpsArm=0.1` 和 `initEpsHand=0.1`，即 PPO 的 logstd 对手臂和手指分别初始化。这允许对不同运动学链使用不同的探索程度。论文未提及此设计。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/scripts/finetune/finetune_grasp.sh` (L5-6)

### 7. Reward shaping: 0.1 scale factor

PPO 训练中对 reward 施加固定的 `0.1` 缩放:

```python
shaped_rewards = 0.1 * rewards.clone()
```

这是为了数值稳定性，但论文中的 reward 数值没有反映这一缩放。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/ppo_transformer/ppo_transformer.py` (L579)

### 8. GPT-2 的修改: 移除 positional embedding

代码中的 `GPT2Model` 是 HuggingFace GPT-2 的修改版本，关键区别是移除了标准的 positional embedding:

```python
self.wte = None  # 移除 word token embedding
# self.wpe = nn.Embedding(...)  # 注释掉 positional embedding
hidden_states = inputs_embeds  # + position_embeds  # 不加 position embedding
```

位置信息通过外部的 `embed_timestep` (learned timestep embedding) 直接加到各模态 embedding 上，而非通过 GPT-2 内部的 position embedding。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/pretrained/transformer.py` (L525-527, L688)

### 9. Gradient clipping = 0.25

预训练时使用 `clip_grad_norm_(model.parameters(), 0.25)` 进行梯度裁剪，论文未提及这一超参数。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/pretrained/trainer.py` (L545 in RobotTrainer.train_step)

### 10. AdaptiveScheduler 的 KL-based 学习率调整

微调使用自适应学习率调度: 当 KL divergence > 2x threshold 时降低 lr (除以 1.5)，当 KL < 0.5x threshold 时提高 lr (乘以 1.5)。这种 KL-based adaptive scheduling 源自 rsl_rl，论文未详述。

文件路径: `/home/l/ws/doc/paper/manip/QiHaoZhi/25_HOP/code/algo/ppo_transformer/ppo_transformer.py` (L787-800)

---

## 跨论文比较

### 与 Haozhi Qi 同作者论文的技术路线对比

| 维度 | PenSpin (2024) | TwistingLids (2024) | DexScrew (2025) | **HOP (2025)** |
|------|---------------|-------------------|-----------------|---------------|
| 核心问题 | 笔旋转 sim-to-real | 双手拧瓶盖 | 简化仿真 + 遥操作 | 从视频学操控先验 |
| 数据来源 | 仿真 oracle 回放 | 仿真 RL (零样本) | 仿真 RL + 遥操作 | **in-the-wild 人类视频** |
| 预训练范式 | Oracle BC 蒸馏 | 无预训练 | 无预训练 | **Generative trajectory modeling** |
| 机器人 | LEAP Hand (16 DoF) | 2x Allegro (32 DoF) | XHand + 7DoF Arm | **xArm + Allegro (23 DoF)** |
| 手部 | 固定 (无臂) | 固定 (无臂) | 有臂 | **有臂** |
| 感知 | Proprioception only | 2 个 3D 关键点 | Proprio + 触觉 | **Point cloud + Proprio** |
| 微调方式 | 45 条真实 BC | 零样本 | BC (50-72 条) | **RL 或 BC** |
| Task-agnostic | 否 (单任务) | 否 (单任务) | 否 (单任务) | **是 (多任务 base policy)** |
| 物理仿真依赖 | 高 (完整仿真) | 高 (精确仿真) | 低 (简化仿真) | **低 (仅 retargeting 用)** |

**技术演进分析**:

HOP 在 Qi 的工作序列中占据一个独特位置: 它是唯一一个以 **task-agnostic pretraining** 为目标的工作。PenSpin/TwistingLids/DexScrew 都是 task-specific 的（先为特定任务训练 RL，再迁移），而 HOP 试图学习一个通用的操控先验，可以适配到任意下游任务。

这反映了两种不同的 learning paradigm:
- **Task-specific pipeline** (PenSpin/TwistingLids/DexScrew): 仿真 RL → 蒸馏/微调 → 部署。优势是最终性能高，劣势是每个新任务需要重新设计
- **Pretraining-finetuning** (HOP): 大规模预训练 → 轻量微调。优势是 amortized cost 低，劣势是单任务性能可能不及专门训练的策略

### 与同批次论文 (HandelBot, AINA, MinBC) 的对比

| 维度 | **HOP (2025)** | AINA (2025) | HandelBot (2026) | MinBC (2025) |
|------|---------------|-------------|-----------------|-------------|
| 核心范式 | Video pretraining + RL/BC | In-the-wild video BC | Sim RL + Residual RL | 遥操作 BC |
| 数据来源 | Internet 人类视频 | Smart glass 人类视频 | 仿真 + 30min 真实 | VR 遥操作 |
| 从视频到动作 | 3D lifting + IK retargeting | 3D point cloud 直接映射 | 不使用视频 | 不使用视频 |
| 预训练规模 | 70K 轨迹 | ~50 条/任务 | 仿真 RL | 无预训练 |
| 策略架构 | GPT-2 causal transformer | VN-MLP + nanoGPT | PPO (sim) + TD3 (real) | Choice Policy (WTA + Score) |
| Task-agnostic | 是 | 否 (per-task 训练) | 否 (per-song) | 否 (per-task) |
| 物体表示 | Point cloud (100 点) | Point cloud (500 点) | 无 (本体感觉) | RGB 图像 |
| 机器人类型 | xArm + Allegro | Kinova + Ability Hand | Franka + Tesollo | GR-1/Star-1 人形 |
| 真实世界验证 | 3 个 BC 任务 | 9 个日常任务 | 5 首钢琴曲 | 洗碗/擦白板 |

**关键对比**: HOP 与 AINA 都试图从人类视频学习灵巧操控，但路径截然不同:

- **HOP**: 视频 → 3D 重建 → 机器人轨迹 (retargeting) → pretraining → 微调。数据量大 (70K 轨迹)，但中间环节多 (3D 重建质量损失、retargeting 近似)
- **AINA**: 视频 → 3D point cloud → 直接策略学习 (在 point cloud 空间消除 embodiment gap)。数据量小 (~50 条)，但端到端更直接

HOP 的 **retargeting 到机器人关节空间** 是一个显著的设计选择: 它允许在机器人的 motor space 中训练，学到物理合理的运动先验；但代价是 retargeting 引入的近似误差和数据多样性损失。AINA 通过在 task space (fingertip positions) 操作来绕过这个问题。

**与 HandelBot 的互补视角**: HandelBot 证明了对于需要极高精度的任务 (毫米级按键)，sim-to-real 需要结构化的适配过程 (refinement + residual RL)。HOP 的 pretraining 提供了更通用但精度较低的先验，更适合抓取/操控等不要求极致精度的任务类别。

**与 MinBC 的方法论对比**: MinBC 关注的是策略学习算法层面的创新 (如何用单次前向传播处理多模态行为)，而 HOP 关注的是数据层面的创新 (如何从视频生成有用的训练信号)。两者在不同维度上推进了灵巧操控的 frontier。

---

## 代码补充分析

### Transformer 精确规格

| 参数 | 值 |
|------|-----|
| hidden_dim | 192 |
| n_layer / n_head | 4 / 4 (48 dim/head) |
| context_length | 16 (0.8s at 20Hz) |
| PointNet | 3 层 MLP (3→192→192→192) + **ELU** + MaxPool2d, 无 BN/T-Net |
| 参数量 | ~750K (transformer blocks only) |
| Dropout | 全部 0.0 |

**Cross-modal prediction**: action 从 pc token stream (`x[:,1]`) 预测而非 proprio stream → 有意的跨模态设计

### 代码 vs 论文额外差异

| 项目 | 论文 | 代码 |
|------|------|------|
| 域随机化 (RL finetuning) | 未明确 | **全部禁用** (randomize_friction/mass/com/table 全 False) |
| PPO mini_epochs | 未提及 | **1** (极保守，保护预训练特征) |
| RL reward 缩放 | 未提及 | `shaped_rewards = 0.1 * rewards` |
| Cabinet 探索噪声 | 未提及 | **0.5** (grasp/throw 的 5 倍，因预训练先验与 cabinet 不匹配) |
| Critic warmup | 未提及 | 200 步 `0*a_loss` (保护预训练 actor 权重) |
| Action 输出 bound | 未提及 | 预训练无 tanh (`action_tanh: false`)，RL clamp [-1,1] |
| Moving average action | 提到 PPO 基线需要 | HOP 微调 `actionsMovingAverage: 1.0` (不用，预训练先验已足够平滑) |

### Phase-gated Reward (代码特有)

Grasp/Lift 用 "ratcheting" 机制 (closest-distance memory)：
```
delta = closest_dist - current_dist  (positive = improvement)
closest_dist = min(closest_dist, current_dist)
reward = clip(delta, 0, 10)
```
通过 `lifted_object` flag 在 approach/lift/transport 三阶段间切换奖励。论文未描述此细节。

### 代码设计模式

1. **PointNet 特征共享**: value function 可用 actor 的 PointNet 嵌入替代原始点云，通过 `value_grads_to_pointnet` 控制梯度流 (finetuning 时 detach)
2. **Attention mask 处理 episode 边界**: reset 时 mask 清零历史，防止 causal transformer attend 到跨 episode 的旧数据
3. **MemoryEfficientExperienceBuffer**: 只存最新 timestep 的 context window，训练时回溯重建，以 compute 换 memory
4. **17 种初始臂姿态版本** (`v1`-`v17`): 默认 `v16`，体现了大量的手动姿态工程

### 作者展望

1. 扩大预训练数据规模和视频来源
2. 更先进的 3D 重建提升轨迹质量
3. 扩展到双手操控
4. 与 language conditioning 结合
5. 使用更复杂的 scene reconstruction (当前只提取单手单物体交互，丢弃场景上下文)
