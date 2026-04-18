# BiDexHD: Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations

Bohan Zhou, Haoqi Yuan, Yuhui Fu, Zongqing Lu (PKU, BAAI), ICLR 2025
arXiv: 2410.02477 | 代码: `BiDexHD/`

---

## 1. Core Problem

双手灵巧操作 (Bimanual Dexterous Manipulation) 面临三个核心难题:

1. **高维动作空间**: 双手各含 arm (6 DOF) + hand (16 DOF) = 22 DOF, 总计 44 DOF, 远超 gripper-based 双手操作
2. **任务多样性不足**: 现有方法 (ArtiGrasp, DynamicHandover, TwistingLids) 针对特定任务设计 per-task reward, 缺乏通用性和可扩展性
3. **数据获取瓶颈**: 遥操需要实时人工介入; RL 需要手动设计仿真任务和奖励函数

BiDexHD 的提问: "能否以统一、可扩展的方式, 从人类演示中学习多样化的双手灵巧操作技能?"

**核心思路**: 将人类双手操作数据集 (TACO) 自动转化为 Dec-POMDP (Decentralized Partially Observable Markov Decision Process, 去中心化部分可观测马尔可夫决策过程) 仿真任务, 用统一的两阶段奖励函数训练 state-based teacher, 再蒸馏为 vision-based student.

---

## 2. Method Overview

三阶段框架:

| 阶段 | 输入 | 输出 | 方法 |
|------|------|------|------|
| Phase 1: Task Construction | TACO 数据集 (MANO 参数 + 物体 6D 位姿) | IsaacGym 中的 Dec-POMDP 任务集 | MANO 解码 + retargeting + IK 回放验证 |
| Phase 2: Teacher Learning | State-based 观测 (关节角/速度, 腕部位姿, 指尖位置, 物体状态) | 多任务 state-based policy | IPPO (Independent PPO, 独立近端策略优化) + 两阶段奖励 |
| Phase 3: Student Learning | 点云 + 本体感受 + K-step 未来位置 | Vision-based policy | DAgger (Dataset Aggregation, 数据集聚合) 在线蒸馏 |

**硬件配置**: 双 RealMan RM65 机械臂 (6-DOF) + LEAP Hand (4 指, 16 DOF), 间距 0.68m, 桌面高度 0.7m. 每个 agent 输出 22 维关节角度, position control.

**数据流**: TACO 数据集 -> MANO 手部模型提取腕部/指尖位姿 -> dex-retargeting 映射到 LEAP Hand 关节角 -> IK 求解机械臂关节角 -> IsaacGym 回放验证 -> 构建 141 个任务 (6 类: Dust, Empty, Pour, Put out, Skim off, Smear).

---

## 3. Key Designs

### 3.1 统一两阶段奖励函数

这是本文最核心的技术贡献. 所有 141 个任务共享同一个奖励函数, 无需 per-task 设计.

**Stage 1: Simulation Alignment (仿真对齐)**

目标: 从固定初始零位姿, 将物体移至人类演示的参考位姿. 包含三个奖励项:

| 奖励项 | 公式形式 | 功能 |
|--------|----------|------|
| r_appro (接近奖励) | -\|\|wrist - x_gc\|\| - w_r * sum(\|\|fingertip - x_gc\|\|) | 引导双手接近功能性抓取中心 |
| r_lift (举升奖励) | max(1 - d_curr/d_init, 0) + w_q * (-D_quat) | 门控触发: 仅在手部足够接近时有效 |
| r_bonus (成功奖励) | 1/(1 + d) if d <= eps_succ else 0 | 信号阶段过渡, 阈值触发 |

**功能性抓取中心 (Functional Grasping Center)**: 不使用几何中心, 而是基于人类演示计算. 从物体 mesh 均匀采样 1024 点, 找到离锚点 (演示腕部/指尖位置的均值) 最近的 L=50 个点, 取平均值. 消融实验表明这对薄片状/有手柄的物体至关重要.

**Stage 2: Trajectory Tracking (轨迹追踪)**

Stage 1 持续 u 步成功后激活. 使用指数衰减奖励:

```
r_track = exp(-w_t * ||x_curr - x_ref||)    if stage 1 succeeds
          0                                   otherwise
```

频率常数 f: 每 f 个仿真步对应人类演示的 1 步 (f=3 in code).

**总奖励**: r_total = r_align + w4 * r_track. 两阶段通过条件门控自动切换, 无需手动 curriculum.

### 3.2 IPPO (Independent PPO) 双手独立策略

左右手各有独立的 actor-critic 网络, 不共享参数.

| 方面 | IPPO | 集中式 PPO |
|------|------|------------|
| 观测/动作空间 | 每侧约 100D obs + 22D act | 全局约 200D obs + 44D act |
| 扩展性 | 独立学习, 更小空间更易优化 | 联合优化, 维度灾难 |
| 泛化 | 可自由组合新任务 | 依赖见过的组合 |
| Train r2 | 74.59% | 显著低于 IPPO |

代码验证: `ippo.py` 中明确创建了 `self.left_agent` 和 `self.right_agent`, 各有独立的 optimizer 和 rollout storage.

### 3.3 Teacher-Student 蒸馏与 Vision-based 泛化

**Teacher 网络**: 5 层 MLP [1024, 1024, 512, 512], ELU 激活, 独立 actor-critic.

**Student 网络**: PointNet backbone (Conv1D -> max pooling -> MLP, 输出 128D 特征) + MLP head. Actor 和 critic 共享 backbone. 代码中使用 `PointNetBackbone` 类.

**DAgger 蒸馏策略**:
- 在线蒸馏, 5% 概率使用 teacher 动作 (混合采样加速早期训练)
- 点云替代精确物体位姿 (4096 点预采样, 运行时子采样 + 高斯噪声)
- 移除 one-hot 物体标识 -- 提升对新物体的泛化
- 可选 K-step 未来位置作为条件输入 (K=5 最佳, K=0 差距仅 ~3%)

**代码细节** (`m3dagger.py`):
- Student 的 robostate indices: 关节角度 + 指尖位置 + last action + 腕部位姿, 不含关节速度和物体精确状态
- Expert (teacher) 的 indices 包含完整的 dof velocity, 物体位姿/速度/角速度, one-hot label
- Future object positions 作为额外 input 附加在观测末尾

---

## 4. Experiments

### 4.1 数据集与评估

- **数据来源**: TACO 数据集, 6 个任务类别, 141 个任务, 16 个语义子任务组
- **划分**: 80% 训练, 20% 测试. 测试集分为 Test Comb (物体/工具都出现在训练集) 和 Test New (含新物体)
- **评估指标**: r1 = Stage 1 平均成功率, r2 = Stage 2 平均轨迹追踪率 (主指标)
- **阈值**: eps_succ = eps_track = 0.1 (约 10cm, 相当宽松)

### 4.2 主要结果

| 方法 | Train r2 | Test Comb r2 | Test New r2 |
|------|----------|--------------|-------------|
| BiDexHD-PPO | 低于 IPPO | 低于 IPPO | 低于 IPPO |
| BiDexHD-IPPO (state) | ~74.59% | 高 | 显著下降 (one-hot 干扰) |
| BiDexHD-IPPO + DAgger (vision) | 接近 teacher | -- | 平均 51.07% |
| BC (Behavior Cloning) | 差 | 差 | 差 |

### 4.3 消融实验

| 消融项 | 影响 |
|--------|------|
| w/o Stage 1 (仅保留 r_track) | 仅 30.5% 简单任务有正 r2, 其余完全失败 |
| w/o Functional Grasping Center (用几何中心) | r1/r2 下降, 尤其是刷子/平底锅等有手柄物体 |
| w/o r_bonus | r2 下降, 阶段过渡信号缺失 |
| K=0 vs K=5 (未来步数) | 差距仅 ~3%, 纯模仿即可达到可接受性能 |

### 4.4 计算资源

- 单 sub-task state-based 训练: ~2 天 (单 A100 40G)
- 每类 vision-based 蒸馏: ~1 天
- 并行环境数: 10000 (训练), 100 (评估)

---

## 5. Related Work Analysis

BiDexHD 定位在三个研究领域的交叉点:

| 领域 | 现有方法 | BiDexHD 的区别 |
|------|----------|----------------|
| Bimanual Dexterous Manipulation | ArtiGrasp, DynamicHandover, TwistingLids | 统一奖励 vs per-task 设计; 141 tasks vs 1-3 tasks |
| Learning from Human Demo | DexMV, VividDex, DexPilot | 双手 + 工具使用 vs 单手; 自动任务构建 vs 手动 |
| Teacher-Student for Dexterity | UniDexGrasp++, DexPoint | 双手 Dec-POMDP vs 单手; IPPO vs PPO |

**数据集选择**: 使用 TACO 而非 ARCTIC (仅 5 种铰接物体) 或 OakInk (单手), 因为 TACO 提供了最丰富的双手工具使用场景.

**RL 算法选择**: IPPO 来自 MARL (Multi-Agent RL, 多智能体强化学习) 文献. De Witt et al. (2020) 在 StarCraft 中已证明 independent learning 在大规模任务上的优势.

---

## 6. Limitations & Future Directions

### 论文自述的局限

1. **追踪精度不足**: eps = 0.1 (10cm) 的阈值过于宽松, 精确空间/时间追踪是未来方向
2. **任务类型有限**: 未涵盖柔性物体操作、双手交接等场景
3. **仅限仿真**: 无 sim2real 验证, 缺乏 domain randomization

### 进一步分析的局限

4. **Stage 2 仅追踪位置不追踪姿态**: 代码中 r_track 使用 `exp(-15 * pos_dist)`, 无姿态追踪. 对于需要精确旋转的任务 (如倒水、翻转) 不够
5. **无接触奖励**: 完全依赖距离引导, 没有显式的接触/碰触奖励. 可能导致"抓空"现象
6. **无 domain randomization**: 论文未提及任何 DR, 多样化物体本身提供了一定泛化, 但不足以支持 sim2real
7. **无 early termination 策略**: 物体掉落桌面/出界会 reset, 但无基于物理合理性的 early termination
8. **action space 设计**: 纯绝对位置控制, 无残差控制或 delta action, 这在长时序追踪中可能不够平滑
9. **one-hot ID 的 workaround**: 训练 teacher 时使用 one-hot 提升性能, 蒸馏时移除. 但这意味着 teacher 本身对新物体泛化能力有限, 蒸馏质量受限

---

## 7. Paper vs Code Discrepancies

通过代码审查 (`BiDexHD/rl_policy/`) 发现以下差异:

| # | 论文描述 | 代码实现 | 影响 |
|---|---------|---------|------|
| 1 | Stage 2 r_track = exp(-w_t * d) | `torch.exp(-15 * ref_object_pos_dist)`, w_t 硬编码为 15 | 论文未说明具体数值 |
| 2 | 论文 Fig.2 显示 r_lift 门控条件为 wrist/fingertip 到 grasping center 的距离 | 代码中 `is_grasp` 条件为 fingertips+palm 到**物体位置**的距离 (非 grasping center), 阈值 0.12 per finger | `compute_ab_stage1_rewards` 中 approach 用 `object_pose[:, :3]` 而非 `object_grasp_pos` |
| 3 | 论文描述 r_quat 使用四元数距离 D_quat | 代码使用 `quat_rew` = cos(2*arcsin(\|\|orient_error\|\|)), 范围 [-1,1], 更接近 cosine similarity | 数学形式不同但目标相似 |
| 4 | 论文描述 r_lift 的门控条件是到 grasping center 的距离 | 代码中存在**两版**奖励函数: `compute_bvdex_stage12_rewards` (使用 grasp center) 和 `compute_ab_stage1_rewards` (使用物体位置, 且 Stage 1 lift reward 被注释掉) | 实际使用版本不明确 |
| 5 | 论文称使用 5 层 MLP [1024, 1024, 512, 512] | 代码 `module.py` 默认 `actor_hidden_dim = [1024, 1024, 512, 512]` = 4 hidden + 1 output = 5 层 | 一致 |
| 6 | 论文称 DAgger 5% teacher action | 代码中使用 `M3DaggerValueIPPO` 类, 但 mixing ratio 需从 config 确认 | 未直接在代码中找到 0.05 |
| 7 | 论文未提及距离 clipping | 代码中 palm 距离 clip 到 0.5, fingertip 总距离 clip 到 3.0 | 工程细节, 防止初期大距离主导梯度 |
| 8 | 代码中 `compute_ab_stage1_rewards` **注释掉了** r_bonus 和 Stage 1 lift reward | 最终版本可能使用简化奖励 (仅 approach + Stage 2 tracking, 无 Stage 1 lift) | 与论文描述的两阶段奖励有显著差异 |
| 9 | 论文称 one-hot ID 在蒸馏时移除 | 代码中 `objlabel_indices = []` 在 student 侧, teacher 侧有 objlabel indices | 一致 |
| 10 | 代码中存在 `compute_ab_stage1_rewards` 的变体, Stage 1 有 timeout 机制: `progress_buf >= 0.2 * max_episode_length` 强制进入 Stage 2 | 论文未提及此 timeout | 工程细节, 防止 Stage 1 卡死 |

**关键发现**: 代码中存在两个奖励函数版本. `compute_ab_stage1_rewards` 更像是最终使用的版本, 其中 Stage 1 的 lift reward 和 bonus 都被注释掉, 且增加了强制 timeout 进入 Stage 2 的机制. 这与论文描述的精心设计的两阶段奖励有较大出入, 暗示实际训练可能更多依赖 Stage 2 的 exp 奖励而非 Stage 1 的多组分对齐奖励.

---

## 8. Cross-Paper Comparison

### 与同领域 human2robot 方法的对比

| 维度 | BiDexHD (2024) | DexMachina (2025) | DexTrack (2025) | HumDex (2026) |
|------|---------------|-------------------|-----------------|---------------|
| **任务类型** | 双手工具使用 (6 类, 141 tasks) | 双手铰接物体 (5 物体) | 单手日常物体 + 工具使用 (3585 轨迹) | 人形双手操作 (单物体) |
| **手型** | LEAP Hand (4 指 16 DOF) | 6 种手型 (Allegro, Inspire 等) | Shadow Hand / LEAP / Allegro | Inspire Hand (5 指 12 DOF) |
| **仿真器** | IsaacGym | Genesis | IsaacGym | IsaacGym |
| **有无臂** | 有 (RealMan RM65 6-DOF) | 无 (浮动手) | 无 (浮动手) | 有 (全人形) |
| **数据集** | TACO | ARCTIC | GRAB + TACO | 自采 (IMU 遥操) |
| **Action Space** | 绝对位置 22D/手 | Hybrid: 腕部残差 + 手指绝对 | Double integration residual | 绝对位置 (全身 + 手) |
| **RL 算法** | IPPO (独立 PPO) | PPO | PPO + 少量 IL loss | ACT (模仿学习, 非 RL) |
| **奖励设计** | 统一两阶段 (approach + lift + bonus + track) | 物体位姿追踪 + 关节 imitation | 物体位姿 + 手部运动学追踪 + 接触 | N/A (纯 IL) |
| **蒸馏/泛化** | DAgger 蒸馏到 vision-based | 无蒸馏 | RL+IL 混合, homotopy 数据增强 | 两阶段 IL (human pretrain + robot finetune) |
| **Sim2Real** | 无 | 无 | 无 | 有 (人形实机部署) |
| **关键创新** | 统一奖励函数 + 自动任务构建 | Object-aware retarget + hybrid action | Data flywheel + homotopy optimization | IMU 遥操 + 自适应手部重定向 |

### 关键技术差异分析

**1. Action Space 设计**

| 方法 | 形式 | 优劣 |
|------|------|------|
| BiDexHD: 绝对位置控制 | a_t in [-1, 1]^22 | 简单, 但长时序追踪不够平滑 |
| DexMachina: Hybrid 残差 | wrist = ref + scale * a, finger = absolute | 腕部紧约束 (+/-4cm), 减小搜索空间 |
| DexTrack: Double integration | delta_delta -> cur_delta -> target = ref + cur_delta | 最平滑, 自动获得连续性, 但需要调 speed scale |
| HumDex: 绝对 + PD | 关节角绝对值, 底层 PD 控制 | 简单可靠, 但不利用参考轨迹 |

**启示**: BiDexHD 的绝对位置控制在 Stage 1 (从零位到参考位姿的大幅运动) 效果可接受, 但在 Stage 2 精细追踪时, DexTrack 的 double integration 或 DexMachina 的紧约束残差设计更优. 这可能部分解释了 BiDexHD 在精确追踪 (eps < 0.075) 时性能下降显著的原因.

**2. 奖励函数设计哲学**

| 方法 | 核心奖励 | 关键信号 |
|------|----------|----------|
| BiDexHD | 距离 + 线性 + exp, 无接触奖励 | 物体位置追踪 (仅位置, 不追踪姿态 in Stage 2) |
| DexMachina | 物体位姿 + joint imitation | 物体轨迹 + 手部关节角对齐 |
| DexTrack | 物体位姿 + 手部运动学 + 接触 | 多层信号, 含接触匹配 |

BiDexHD 的奖励设计最"简洁"但也最"粗糙": Stage 2 仅追踪物体位置, 不追踪姿态, 不追踪手部状态, 无接触奖励. 这是其追踪精度不如 DexTrack 的重要原因.

**3. 泛化策略**

| 方法 | 泛化机制 | 效果 |
|------|----------|------|
| BiDexHD | 多任务训练 (141 tasks) + DAgger + 移除 one-hot | 51.07% unseen tasks |
| DexMachina | 多手型 retarget, 不追求 cross-object 泛化 | 5 种手型 cross-embodiment |
| DexTrack | Data flywheel + homotopy optimization + 巨型 MLP (8192 hidden) | 3585 轨迹, 高泛化 |
| HumDex | Human data pretrain + robot data finetune | 实际场景迁移 |

BiDexHD 的泛化策略最系统化 (大规模多任务 + 蒸馏 + 去标识), 但受限于 teacher 质量. DexTrack 通过 data flywheel 持续扩充数据, 实现了更强的泛化.

**4. 整体评价**

BiDexHD 的核心价值在于**证明了从人类演示自动构建大规模双手操作任务并用统一奖励训练的可行性**. 其系统设计 (三阶段框架) 清晰, Task Construction 的自动化程度高. 但在技术深度上:
- 奖励函数虽然"统一"但过于简化 (无接触/无姿态追踪)
- 绝对位置控制不利于精细操作
- 无 sim2real, 实际部署价值未验证
- 代码与论文存在较大差异, 实际效果可能依赖于未充分描述的工程细节

相比之下, DexTrack 在技术深度 (double integration, homotopy, data flywheel) 和规模 (3585 轨迹) 上都更强, 但仅限单手. DexMachina 在 cross-embodiment 和 action space 设计上有独到见解. HumDex 是唯一有 sim2real 部署的方法.
