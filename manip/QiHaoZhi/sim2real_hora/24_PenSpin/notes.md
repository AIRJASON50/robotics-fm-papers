# Lessons from Learning to Spin "Pens" - 论文笔记

**论文**: Lessons from Learning to Spin "Pens"
**作者**: Haozhi Qi et al. (UC Berkeley, UCSD, CMU)
**发表**: arXiv:2407.18902v2, 2024
**项目**: https://penspin.github.io/
**代码**: HORA codebase 的直接扩展

---

## 从 HORA 到 PenSpin: 阅读引导

读完 HORA 后，PenSpin 的核心问题是：**HORA 的 RMA 方案在笔旋转上失败了，怎么办？**

HORA 在粗圆柱体上成功了，但笔状物体暴露了三个 HORA 框架无法处理的问题：
1. **接触模型不准确**: 笔是细长体，与指尖的接触近似线接触（非面接触），PhysX 对这种几何的误差大
2. **稳定域极小**: 笔在指尖间的稳定抓取域远小于方块/球体，微小力/位置误差就导致掉落
3. **Finger gaiting 更激进**: 旋转笔需要手指在某些时刻只有 1-2 个接触物体，这种临界状态下任何 sim-real 误差都被放大

PenSpin 的解法不是改进 sim-to-real 的精度，而是**承认直接迁移不可行**，改用三阶段 pipeline 绕过。

---

## HORA → PenSpin: 每个组件的变化

### 1. 训练物体

| | HORA | PenSpin |
|--|------|---------|
| 形状 | 粗圆柱体 | 细长笔状圆柱体 (`cylinder_pencil-5-7`) |
| Scale 列表 | `[0.78, 0.80, 0.82, 0.84, 0.86]` | `[0.28, 0.29]` (细很多) |
| 旋转轴 | `-z` (向下) | `+z` (向上) |

### 2. 初始状态设计 (最大变化)

**HORA**: 1 个 canonical grasp + 随机扰动。对粗圆柱够用，因为任何合理的抓取都能开始旋转。

**PenSpin**: **6 个人工设计的 canonical grasp**，对应旋转周期中的不同接触相位：

| 编号 | 接触模式 | 对应相位 |
|------|----------|----------|
| 0 | 拇指 vs 食指+中指+无名指 | 三指支撑，拇指拨动 |
| 1 | 拇指 vs 中指+无名指 | 两指支撑 |
| 2 | 拇指+食指 vs 中指+无名指 | 两组对握 |
| 3 | 食指 vs 中指 | 双指交替 |
| 4 | 食指 vs 拇指+中指 | 食指拨动 |
| 5 | 拇指 vs 食指 | 最小接触 |

**为什么必须改？** 单一初始姿态导致 RL 收敛到**永不断开接触**的局部最优——手指全程夹住笔，只是原地扭动不产生旋转。多个初始 pose 覆盖了旋转周期的不同相位，RL 从中发现"断开-重建接触"的 finger gaiting 模式。这是**探索瓶颈**，不是奖励不足。

**Grasp cache 生成流程** (`gen_grasp.py`):
1. 20000 并行环境，CPU pipeline，position control
2. 每个 env 随机选一个 canonical pose + 0.05 关节噪声
3. 模拟 40 步后筛选：≥2 指尖接触 + 未掉落 + 笔两端 z 值在 [0.60, 0.63]
4. 存活的 env 保存 `(hand_pos[16D], obj_pose[7D])` 到 `.npy`
5. 累积 50k 条后按 scale 分桶存储

### 3. Oracle 观测空间

| 输入 | HORA | PenSpin | 为什么加 |
|------|------|---------|----------|
| obs (关节角历史) | 96D (3帧x32D) | 96D (不变) | — |
| env_mlp 输入 | 9D (pos+scale+mass+friction+com) | ~61D (9D + 朝向4D + 角速度3D + 指尖12D + 触觉...) | 笔需要更多状态信息 |
| 点云 | 无 | PointNet(100点x3D) → 32D | 笔的几何变化大，需要形状信息 |
| 触觉 | 无 | 32D 二值接触 (带噪声和延迟) | 知道哪些手指在接触 |

Oracle 总输入: `obs(96D) + tanh(env_mlp(8D) + PointNet(32D))` = 136D → actor_mlp [512,256,128]

### 4. 奖励函数

| 奖励项 | HORA | PenSpin | 变化原因 |
|--------|------|---------|----------|
| $r_{rot}$ | clip(ω·k̂, -0.5, 0.5) | clip(ω·k̂, -0.5, 0.5) | 不变 |
| $r_{pose}$ | -\|q-q_init\|² | -\|q-q_init\|² | 不变 |
| $r_{linvel}$ | -\|v\|₁ | -\|v\|₁ | 不变 (论文写 L2 但 HORA 代码已是 L1) |
| $r_{torque}$ | -\|τ\|² | -\|τ\|² (最后一个 substep) | 不变 |
| $r_{work}$ | -τᵀq̇ | -(Σ\|τ·q̇\|)² | 取绝对值后平方 |
| **$r_z$** | **无** | **-1.0 × (z_max - z_min of point cloud)** | **新增: 保持笔水平** |
| **$r_{rotate\_penalty}$** | **无** | **-0.3 × max(0, \|ω\|-1.0)** | **新增: 限制旋转速度** |
| **$r_{position}$** | **无** | **-0.1 × \|pos-canonical\|²** | **新增: 防止物体漂移** |

**$r_z$ 是最关键的新奖励**: 它惩罚笔两端高度差。在仿真中对旋转指标几乎没影响，但它迫使 oracle 学到**笔始终保持水平**的轨迹——这种轨迹在开环回放时成功率远高于笔倾斜的轨迹。这是一个**面向下游部署**的奖励设计。

### 5. Student 训练 (最大方法论变化)

这是 HORA → PenSpin 最根本的路径改变：

```
HORA (RMA 原版):
  Stage 2: adaptation module φ 自己 rollout → 冻结的 π 执行 → 收集 (proprio_hist, z_gt) → MSE(ẑ, z)
  → Student on-policy rollout + latent 对齐

PenSpin (DemonTrain):
  Stage 2: teacher (oracle) rollout → 收集 (obs, action) → MSE(student_action, teacher_action)
  → Teacher rollout + 行为克隆
```

**为什么放弃 HORA 的 RMA？**

实验尝试了三条路，全部失败：

| 尝试 | 方法 | 结果 | 原因 |
|------|------|------|------|
| HORA 原版 RMA | student 自己 rollout + latent 对齐 | 不收敛 | student 太差，笔立刻掉，数据全是失败状态 |
| DAgger | student 自己 rollout + teacher action 标签 | 不收敛 | 同上——student 进入 teacher 从未见过的状态 |
| 视触觉蒸馏 | teacher rollout + 视觉/触觉 student | 仿真中可行，真实中 ~0% | 视觉/触觉的 sim-to-real gap 在动态任务中是灾难性的 |

#### BC vs DAgger vs Teacher-rollout BC 辨析

理解 PenSpin 的 Stage 2 选择，需要先搞清三种监督式策略学习的本质区别。三者的学习信号完全相同（MSE(student_action, teacher_action)，没有 reward/value），区别**只在于数据怎么来的**：

**纯 BC (Behavior Cloning)**:
- 预录一个固定数据集 {(obs_i, action_i)}，训练过程和图像分类一样
- Student 遍历 teacher 的 obs，给出自己的 action，和 teacher 标答算 loss
- **训练中不启动仿真器，不做 rollout，样本之间无因果关系**
- 问题：**Compounding error** — 训练假设样本 i.i.d. (Independent and Identically Distributed, 独立同分布: 样本之间互不影响且来自同一分布)，但部署时是序贯决策，**i.i.d. 的两个假设同时被打破**: 当前输出改变下一步输入（不独立），student 偏移导致后续状态偏离训练分布（不同分布）
- Compounding error 的根源: student 对 teacher 的 **approximation error (逼近误差)**——有限数据 + 有限模型容量下不可能零误差复制 teacher。这个误差在独立预测场景无所谓（图像分类错一张不影响下一张），但在序贯决策中通过状态传播链从 O(ε) 放大到 O(εT²)
- 最终结果: 部署时产生 **distribution shift (分布漂移)**——student 的偏移状态落在训练分布之外 (**OOD, Out-of-Distribution**)，输出变得不可预测

**DAgger (Dataset Aggregation)**:
- Student 自己在仿真器中 rollout 完整 episode → 到达自己的偏移状态 {s'} → **实时查询 teacher 网络** teacher(s') 得到标签 → (s', teacher(s')) 加入训练集 → 回到 Phase 2 在累积数据集上离线训练多轮 → 下一轮用更新后的 student 重新 rollout
- 注意: **rollout 时网络冻结** (只采集数据)，rollout 结束后才离线训练。不是边走边更新
- 每轮 student 见过的偏移状态越来越多 → 训练分布逐渐覆盖部署分布 → compounding error 从 O(εT²) 降到 O(εT)
- **DAgger 的适用条件**: (1) student 偏了之后还有得救——任务容错性要足够高，episode 要够长，才能收集到足够多的有效训练数据; (2) teacher 在偏移状态上能给出有效纠正方向——不是 teacher "处理不了"，而是偏移状态在物理上必须仍可恢复 (存在合理的纠正动作)
- PenSpin 上失败的原因：笔旋转**一步就到不可恢复状态** (笔掉落) → episode 只有 2-3 步 → 训练集全是失败状态的垃圾数据 → teacher 在 "笔已掉" 状态给的标签没有物理意义 (不存在 "把笔从地上捡回来" 的合理动作) → 迭代永远无法启动

**Teacher-rollout BC (PenSpin 的选择)**:
- Teacher 在仿真器中 rollout → 收集高质量数据 → student 在 teacher 的数据上做 BC
- 和纯 BC 的区别：数据是在线生成的（不是预录数据集），每次训练迭代重新 rollout
- 和 DAgger 的区别：rollout 执行者是 teacher 不是 student → 数据分布是 teacher 的不是 student 的 → 仍然有 distribution shift
- **为什么可接受**: PenSpin 不指望 Stage 2 解决一切。Student 学到的是 "运动先验"（旋转的动力学模式），distribution shift 的问题留给 Stage 3 的 45 条真实数据微调来修正

| | 数据采集 | 仿真器 | 数据分布 | Compounding error |
|--|---------|--------|----------|-------------------|
| 纯 BC | 固定数据集 | 不用 | teacher 的 | 部署时暴露 (O(εT²)) |
| DAgger | student 在线 rollout | 持续使用 | 迭代趋向 student 的 | 训练时消除 (O(εT))，但需要任务容错性 + 足够长的 episode |
| Teacher-rollout BC | teacher 在线 rollout | 持续使用 | teacher 的 | 存在但由 Stage 3 补偿 |

**最终选择**: Teacher-rollout BC (proprioceptive only)
- `is_demon=True`: 用 teacher 的确定性动作 step 环境
- `enable_latent_loss=False`: 不对齐 latent，只做动作模仿
- Student 输入: 关节角历史 (30步x32D) → TemporalTransformer → 40D proprio feat
- Loss: MSE(student_mu, teacher_mu)

**代码中的关键配置** (对比 HORA):

```bash
# HORA Stage 2 的配置 (假设)
proprio_adapt=True          # 启用 RMA adaptation module
priv_info_stage2=True       # 走 HORA 的 Stage 2 路径
is_demon=False              # student 自己 rollout

# PenSpin Stage 2 的实际配置 (train_student_sim.sh)
proprio_adapt=False         # 关掉 HORA 的 RMA 路径
distill=True                # 启用 DemonTrain
is_demon=True               # teacher rollout
enable_latent_loss=False    # 不对齐 latent
proprio_mode=True           # 用 TemporalTransformer 处理历史
```

**代码共存**: 因为 PenSpin 继承 HORA codebase，`models.py` 中 `_actor_critic()` 有两套路径共存。通过 `priv_info_stage2` 和 `student` 标志切换。PenSpin 走 `student=True` 路径，HORA 走 `priv_info_stage2=True` 路径。

#### PenSpin 与 HORA/RMA 的方法论辨析

##### Teacher 的 env_mlp: 为什么保留？

PenSpin 的 teacher 和 HORA 一样有 env_mlp (privileged_info → 8D latent)，但 **PenSpin 的蒸馏不做 latent 对齐**（`enable_latent_loss=False`），蒸馏信号纯粹是 action BC loss。

那为什么 teacher 还保留 env_mlp 而非直接把 raw privileged 拼接到 obs 上？**代码继承是主要原因**，但 env_mlp 的信息瓶颈结构客观上也有好处：8D latent 约束了 actor_mlp 的输入维度，使 student 替换 env_mlp 后可以复用 actor_mlp 权重。

理论上完全可以去掉 env_mlp，让 teacher 直接用 `obs + raw_privileged` → actor_mlp → action，然后 student 用 `obs` → actor_mlp → action 做纯 BC 蒸馏。PenSpin 没这么做是因为继承了 HORA codebase，不是方法论必要。

##### Student 的 env_mlp: 退化为指尖编码器

PenSpin student 中也有 env_mlp，但输入不再是完整特权信息 (61D)，而是**只有指尖位置 (12D，加噪声)**。指尖位置可以通过正运动学从关节角度近似计算，在真实世界近似可获得。所以 student 的 env_mlp 退化为一个普通的指尖位置编码器，不承担 RMA 式的 extrinsics 推断任务。

```
HORA student env_mlp:  关节历史 → adaptation module → 8D → 推断环境 extrinsics
PenSpin student env_mlp: 指尖位置 12D (近似可观测) → env_mlp → 8D → 普通特征编码
```

##### TemporalTransformer: 不是特权，是代偿

TemporalTransformer **只在 student 侧**，teacher 没有。

```
Teacher:  obs(单帧) + privileged(直接给) → 不需要历史
Student:  obs(单帧) + TemporalTransformer(30帧历史→40D) → 用历史代偿特权信息缺失
```

Teacher 不需要看历史，因为仿真器每帧直接给它物体位置/质量/摩擦力。Student 看不到这些，所以用 30 帧关节角历史间接推断——关节角的时序变化模式隐含了速度、加速度、接触状态变化、物体物理特性等信息。

**这和 RMA 的 adaptation module 做的事情本质相同**（从 proprioception 历史推断环境特性），但训练信号不同：

| | RMA / HORA | PenSpin |
|--|-----------|---------|
| 时序编码器 | TemporalConv (1D CNN) | TemporalTransformer (2层2头) |
| 训练信号 | MSE(encoder_output, env_mlp_output) — latent 对齐 | action BC loss 的梯度反传 — 隐式学习 |
| 输出要求 | 必须对齐 teacher 的 8D latent | 只需帮助产生正确动作，学到什么都行 |

PenSpin 的 TemporalTransformer 通过 action loss 的梯度**隐式地**学到有用的时序特征，不显式对齐任何 latent。最终学到的表示可能和 RMA 的不一样，但目的相同：从关节角历史中提取单帧看不到的信息。

**为什么不直接把 30 帧 flatten 成 960D 送 MLP？** 理论上可以，但 Transformer 的归纳偏置更匹配时序数据——attention 天然做"比较序列中不同位置"的操作（提取速度/加速度需要的就是帧间差异），而 MLP 需要自己发现这种结构，数据效率更低且更容易过拟合。

##### 架构演进对比

| 工作 | 时序编码器 | 训练信号 | 功能定位 |
|------|-----------|----------|----------|
| RMA (2021, 四足) | 1D CNN | latent 对齐 | 推断地形 extrinsics |
| HORA (2022) | TemporalConv | latent 对齐 | 推断物体 extrinsics |
| PenSpin (2024) | TemporalTransformer | **action BC** (隐式) | 代偿特权信息缺失 |

架构在演进 (CNN → Conv → Transformer)，但更根本的变化是**训练信号从显式 latent 对齐退化为隐式 action loss**——PenSpin 放弃了 RMA 的核心机制（latent 对齐），只保留了"从历史中提取信息"这个功能需求。

#### Stage 2 的整体思路：为什么这样分两步

BC 和 DAgger 都失败后，PenSpin 的解法是**承认单一阶段无法同时解决所有问题**，拆成两步各解决一个：

```
Stage 2 (仿真 teacher-rollout BC):
  目标: 学会 "手指怎么转" (运动模式)
  数据: 仿真中无限量 teacher rollout
  遗留: sim-to-real 的 distribution shift (仿真数据 ≠ 真实世界)

Stage 3 (真实数据微调):
  目标: 修正 "真实世界和仿真有什么不同" (物理残差)
  数据: 45 条真实世界成功轨迹
  前提: Stage 2 的运动先验 (否则 45 条从零学不会旋转笔)
```

为什么 45 条真实数据就够？因为 student 不是从零开始——仿真预训练已经学会了运动模式，微调只需修正物理残差（PD 跟踪误差、接触摩擦差异等）。修正残差比学完整技能需要的数据量少得多。论文实验验证：仿真预训练 + 45 条 > 无预训练 + 75 条。

真实数据从哪来？**开环回放 teacher 的仿真轨迹**（详见 Stage 3）。遥操作不可行（笔旋转太快太动态），所以真实数据的唯一来源是把 teacher 的动作序列在真实机器人上盲目执行，~40% 碰巧成功（靠能量惩罚的平滑性 + $r_z$ 的水平性）。

最终 student 部署时是**闭环策略**（实时读关节角 → 推断动作），超越了 teacher 的开环回放（~40%→~60.8%）。超越的原因不是"学得更好"而是"用法不同"——student 从 45 条轨迹中学到了成功模式的共性（泛化），而开环回放只是固定重复某一条轨迹（无适应能力）。

### 6. Stage 3: 开环回放 + 微调 (PenSpin 完全新增)

HORA 没有这一步——训练完直接部署。PenSpin 新增了两步：

**Step 3a: 开环回放** (`real/robot_controller/teacher_replay.py`)
1. 从仿真 oracle 选 15 条 >800 步的轨迹（不同初始 pose）
2. 将 oracle 的 target qpos 序列**原封不动**发送到真实 Allegro Hand（无闭环控制）
3. 关节顺序重映射: HORA 空间 → Allegro 物理排列 (`_action_hora2allegro()`)
4. 每个训练物体回放多次，筛选旋转 >180° 的成功轨迹
5. 每物体 15 条 → 共 45 条

**为什么开环能成功？** 这是最反直觉的发现：
- 能量惩罚 ($r_{work}$, $r_{torque}$) 使 oracle 轨迹**极其平滑**——target qpos 变化缓慢
- $r_z$ 保持笔水平 → 重力方向上更稳定
- sim-to-real gap 在**关节级别**较小 (位置控制目标 → PD 跟踪 → 实际关节角，这个映射在 sim 和 real 中高度一致)
- ~40% 的成功率足够用于数据收集

**Step 3b: 真实数据微调** (`real/finetune_ppo.py`)
1. 加载 Stage 2 的 student checkpoint
2. 从 45 条 h5 轨迹中用滑动窗口生成 (obs, proprio_hist, fingertip_pos, action) 样本
3. L1 loss 行为克隆: `loss = F.l1_loss(student_action, real_action)`
4. 3000 epochs, lr=1e-3
5. 关节数据需要 Allegro → HORA 重映射 (`_obs_allegro2hora()`)

#### 对 Stage 3 的重新审视：这不是克服 sim-to-real gap，而是提升成功率

开环回放的成功率揭示了一个被论文叙事掩盖的事实：**关节级别的 sim-to-real gap 其实没那么大。** 论文 Table 1 数据显示 Oracle Replay 平均成功率约 47%（部分物体如 D:78.2%, E:67.1% 相当高），如果 gap 是灾难性的，这个数字应该接近 0%。

微调后训练物体 ~60.8%——提升约 13 个百分点。这说明 Stage 3 做的修正量不大，大部分能力来自仿真。论文把 Stage 3 定位为"补偿 sim-to-real gap"，但更诚实的描述是：**补偿 BC 蒸馏的 distribution shift + 最后一点物理残差修正。**

真正困难的不是"仿真和真实世界差太多"，而是"没有好的蒸馏方案把 oracle 的能力完整转移出来"（DAgger 崩溃 + 视触觉 gap）。

**实用参考价值**: 如果你的策略可以部署但成功率不高，可以用成功的 rollout 数据做后训练（post-training）来提升性能——筛选成功轨迹 → 在成功数据上微调模型。这本质上是一种 **self-improvement / 成功经验过滤**，类似于 LLM 领域的 rejection sampling finetuning (RSF)：用模型自己生成的成功样本微调自己。

```
LLM 领域:    模型生成多个回答 → 筛选最好的 → 微调模型 (RSF / ReST)
PenSpin:     oracle 开环回放多次 → 筛选成功的 → 微调 student
共同逻辑:    生成 → 筛选 → 微调，用自己的成功经验改进自己
```

40% 的开环成功率也暗示了大量的 reward 调参工作——$r_z$（保持水平）、能量惩罚（保持平滑）、角速度限制（保持保守）等都是让开环轨迹在真实世界中物理可行的关键。这些 reward 项对仿真性能几乎没影响，纯粹是面向"生成可回放轨迹"的设计。

### 7. 网络架构变化

| 组件 | HORA | PenSpin | 为什么 |
|------|------|---------|--------|
| Actor MLP | [512, 256, 128] | [512, 256, 128] | 不变 |
| env_mlp | 9D → [256,128,8] | 61D → [256,128,8] (teacher); 12D → [256,128,8] (student) | Student 只看指尖位置+噪声 |
| Adaptation | TemporalConv (1D CNN) | **TemporalTransformer** (2层, 2头, CLS token) | Transformer 对时序建模更好 |
| 点云编码 | 无 | PointNet (3→64→256, GELU, max pool) → 32D | 笔形状多变需要几何信息 |
| 物体端点 | 无 | MLP [6,6,6] per endpoint, max pool → 18D | 笔姿态的低维表示 |

### 8. 域随机化变化

| 参数 | HORA | PenSpin | 意义 |
|------|------|---------|------|
| 随机外力 | `forceScale=2.0, prob=0.25` (config 默认 0 但训练脚本覆盖) | `forceScale=2.0, prob=0.25` | 不变 (两者都用) |
| 弹性系数 | 不随机化 | `restitution ∈ [0, 1]` 随机 | 新增 |
| Student 噪声 | 同 teacher | **加倍** (episodic 0.02, temporal 0.01) | 弥补真实世界噪声 |

---

## 关键结果

| 方法 | 训练物体成功率 | 未见物体成功率 |
|------|-------------|-------------|
| Oracle 开环回放 | ~40% (A:37.6%, B:54.3%, C:29.5%) | ~47% (D:78.2%, E:67.1%, ..., J:34.4%) |
| 视触觉蒸馏 | ~8.7% | ~0% |
| Proprioceptive 蒸馏 (DAgger) | 不收敛 | 不收敛 |
| **PenSpin (完整 pipeline)** | **~60.8%** (A:54.9%, B:70.0%, C:57.6%) | **~63.3%** |

注意：Oracle 开环回放在部分物体上成功率相当高 (D:78%, E:67%)，整体平均约 47%，说明关节级 sim-to-real gap 并不大。

- 仿真预训练 + 45 真实演示 > 无预训练 + 75 真实演示
- 物体范围: 10.8g~49.7g, 14.4cm~22.2cm

---

## 非显而易见的洞察

1. **初始状态设计比奖励工程更重要**: 这是本文最深刻的发现。不是奖励不够好，是探索被初始状态锁死了
2. **DAgger 失败有特定条件**: 不是 DAgger 本身有问题，而是当 student 初始能力太差 → on-policy rollout 立刻崩溃 → 数据全是失败状态。在更稳定的任务上 (DexScrew) DAgger 是可行的
3. **开环轨迹在真实中成功**: 能量惩罚 + $r_z$ 使轨迹平滑且水平 → 关节级 sim-to-real gap 足够小
4. **Sim-to-real gap 在模态间不对称**: 关节角迁移好，视觉/触觉在动态任务中灾难性失败
5. **$r_z$ 是面向部署的奖励设计**: 仿真中影响小，但对开环回放成功率至关重要

---

## 作者展望

1. 扩展到完整 SO(3) 旋转 (当前仅 z 轴)
2. 集成视觉/触觉感知以提升闭环能力
3. 将三阶段 pipeline 推广到更多手内操控任务 (后续 DexScrew 即为此方向)

---

## 代码导读: 关键文件

| 文件 | 功能 | 从 HORA 的变化 |
|------|------|---------------|
| `penspin/tasks/allegro_hand_hora.py` | 环境主体 | 新增触觉/点云/obj_ends/z_dist_penalty/pencil 终止条件 |
| `penspin/tasks/allegro_hand_grasp.py` | **新增** Grasp cache 生成 | 6 个 canonical poses + 稳定性筛选 |
| `penspin/algo/models/models.py` | Actor-Critic 网络 | 新增 PointNet/TemporalTransformer/student 路径 |
| `penspin/algo/models/block.py` | **新增** 时序模块 | TemporalConv + TemporalTransformer |
| `penspin/algo/ppo/demon.py` | **新增** DemonTrain | Teacher-rollout BC (替代 HORA 的 RMA Stage 2) |
| `real/robot_controller/teacher_replay.py` | **新增** 开环回放 | Stage 3a |
| `real/finetune_ppo.py` | **新增** 真实数据微调 | Stage 3b |
| `real/agent/ppo_agent.py` | **新增** 部署 agent | 含关节顺序重映射 |
| `scripts/train_teacher.sh` | Oracle 训练脚本 | CLI 参数覆盖 HORA config |
| `scripts/train_student_sim.sh` | Student 训练脚本 | is_demon=True, 关闭 RMA |
| `scripts/gen_grasp.sh` | Grasp cache 生成脚本 | — |
| `configs/task/AllegroHandHora.yaml` | 环境配置 | **与 HORA 共享同一个文件**，通过 CLI 覆盖 |

### 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| Reward 全局缩放 | 未提及 | `shaped_rewards = 0.01 * rewards` |
| 角速度计算 | 使用仿真器 angvel | **四元数差分再转轴角**手动计算 |
| Action scale | 未详述 | `action_scale = 0.04167` |
| Scale 随机化 | 连续 [0.95, 1.05] | 离散列表 `[0.28, 0.29]` |
| Student latent loss | 论文提到可用 | **默认关闭** (`enable_latent_loss=False`) |
| 微调参数范围 | 只训练 actor_mlp | `param_dicts` 设置了但 optimizer 用 `model.parameters()` (矛盾) |
| HORA RMA 路径 | — | **代码保留但 PenSpin 不走** (`proprio_adapt=False`) |

---

## 局限

- 仅限 z 轴旋转，非完整 SO(3)
- 需要手动放置物体到稳定抓取
- 不使用视觉/触觉部署
- 开环回放数据收集需要人工筛选
- 代码中 HORA/PenSpin 两套路径混杂，需要仔细对照 config 才能理解走哪条路
