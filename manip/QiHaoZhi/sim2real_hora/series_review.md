# HORA → PenSpin → DexScrew 系列评价

三个工作共享增量式 codebase (HORA fork → PenSpin 改 → DexScrew 改)，构成一条从"直接 sim-to-real"到"仿真只做最小先验"的演化线。

---

## 1. 系列概要

| 维度 | HORA (2022) | PenSpin (2024) | DexScrew (2025) |
|------|-------------|----------------|-----------------|
| 任务 | 粗圆柱旋转 | 细笔旋转 | 螺母紧固/螺丝刀 |
| 手部 | Allegro 16 DOF | Allegro 16 DOF | XHand 12 DOF |
| 仿真精度 | 精确 | 精确 | 故意简化 (无螺纹) |
| Sim-to-Real | 直接迁移 (RMA) | BC 蒸馏 + 真实微调 | DAgger 蒸馏 + 遥操作 + 触觉 BC |
| 蒸馏 loss | latent MSE | action BC (latent 关掉) | latent MSE + action BC (dual loss) |
| 核心代码量 | ~3500 行 | ~5500 行 (膨胀) | ~4500 行 (有意回缩) |
| PPO 实现 | 自写 396 行 | 继承 + 改 | 继承 |
| 触觉 | 无 | 仿真中用，部署不用 | 真实世界核心模态 |
| 真实数据 | 无 (零样本) | 45 条开环回放 | 50-72 条遥操作 |

---

## 2. 方法论演进

### 2.1 Sim-to-Real 策略

```
HORA:     仿真 RL → RMA 适配 → 直接部署 (零样本)
PenSpin:  仿真 RL → BC 蒸馏 → 开环回放采数据 → 真实数据微调
DexScrew: 简化仿真 RL → DAgger → student 作为遥操作原语 → 真实数据 BC
```

**关键发现**: 关节级 sim-to-real gap 其实不大。PenSpin 开环回放 ~47% 成功，DexScrew 直接 sim-to-real 41.6% 进度。真正的瓶颈不在仿真精度，而在：
1. 蒸馏方案 (如何把 oracle 知识转移到可部署策略)
2. 感知模态 (触觉在仿真中无法建模)
3. 任务覆盖 (仿真没建模的部分如手臂推进)

### 2.2 蒸馏方法

| | HORA | PenSpin | DexScrew |
|--|------|---------|---------|
| 方法 | RMA (student rollout + latent MSE) | Teacher-rollout BC (被迫) | DAgger + dual loss |
| 为什么 | 粗圆柱容错高，student rollout 可行 | 笔一掉就崩，DAgger 不可行 | 螺母在螺栓上不会掉，DAgger 又可行 |
| 训练参数 | 只训练 adapt_tconv | 全部 student 参数 | 只训练 adapt_tconv |
| Latent loss | 是 (唯一 loss) | 关掉 | 是 (+ action BC) |

**核心规律**: 蒸馏方法的选择完全由任务容错性决定，不是方法本身的优劣。DAgger 是首选 (因为轨迹数据非 i.i.d.)，BC 是 DAgger 崩溃时的 fallback。

### 2.3 奖励设计

三个项目共享的 pattern:
- `shaped_rewards = 0.01 * rewards` (PPO 中硬编码)
- rotation reward: `clip(angular_velocity · rotation_axis)`
- 能量惩罚: torque² + work²
- 有限差分计算角速度 (不信任仿真器 `dof_vel`)

演进:
- HORA: 5 项基础奖励
- PenSpin: +3 项 ($r_z$ 保持水平, rotate_penalty 限速, position_penalty 防漂移)。**$r_z$ 是面向部署的奖励设计**——对仿真指标无影响但让开环回放成功率从 ~0% 到 ~47%
- DexScrew: +proximity_reward (鼓励接触) + 角速度阈值课程。reverse_penalty 计算了但未加入 reward (死代码)

**两者都用随机外力** (`forceScale=2.0, prob=0.25`)。这不是 PenSpin 的新增——HORA 的训练脚本已经覆盖了 config 默认值 0.0。

---

## 3. 代码质量评价

### 3.1 架构: 极简自写 PPO

整个 RL pipeline 约 1400 行核心代码 (PPO 396 + models ~170 + stage2 ~250 + env ~660)。不依赖 rl_games/stable-baselines/brax 等外部 RL 库，从零手写。

**优势**:
- 完全可控，每一行都能理解和修改
- 调试直接 (没有框架抽象层)
- 适合研究探索阶段的快速迭代

**代价**:
- 没有抽象层 → 每个项目 copy-paste 修改
- 缺少标准化接口 → 不同项目间代码结构相似但细节不兼容

### 3.2 代码继承的优劣

```
代码量演变:
  HORA models.py:    153 行  (简洁)
  PenSpin models.py: 284 行  (条件分支膨胀: if student, if proprio_mode, if input_mode...)
  DexScrew models.py:177 行  (有意回缩，清理了多模态分支)

  HORA padapt.py:    184 行
  PenSpin demon.py:  549 行  (膨胀: 双模型, ExperienceBuffer, GIF 录制...)
  DexScrew padapt.py:253 行  (回缩: 在线训练, 去掉双模型, 保留 dual loss)
```

PenSpin 是典型的"研究代码快速迭代"产物——功能堆叠但结构恶化。DexScrew 做了**有意识的清理**，代表了从功能爆炸到聚焦的回归。

### 3.3 持续存在的技术债

从 HORA 继承到 DexScrew，**三代都没修**的问题:

| 技术债 | 描述 | 风险 |
|--------|------|------|
| `eval()` dispatch | `agent = eval(config.train.algo)(...)` 动态实例化类 | 安全隐患 + 不可 type-check |
| `eval()` 属性查找 | `eval(f'self.enable_priv_{name}')` 在 env 中查找配置 | 应该用字典 |
| `computer_return` | 拼写错误 (应为 compute_return) | 三代未修，说明无 code review |
| 手工四元数 | `quat_to_axis_angle`, `quat_mul` 在每个项目中重复实现 | 应提取为共享 utils |
| 零测试 | 三个项目都没有任何单元测试 | 重构时无安全网 |
| 零类型注解 | 几乎没有 type hints | IDE 支持差 |
| Magic numbers | `0.01` reward 缩放, `1/24` action scale 等硬编码 | 跨项目不一致时难以追踪 |

### 3.4 测试与文档

- **零单元测试** (所有三个项目)
- HORA 有 changelog (少见的好实践，记录了版本间的 bug fix)
- Config YAML 是主要文档，但实际训练参数被 shell 脚本覆盖 (config 默认值和实际训练值不一致——如 HORA 的 `forceScale` config 默认 0 但脚本覆盖为 2)
- 部署代码基本无注释

---

## 4. 方法论 Takeaway

### 可复用的洞察

1. **初始状态设计比奖励设计更重要** (PenSpin): RL 探索瓶颈往往在初始状态分布，不在奖励函数。多个 canonical grasp 覆盖旋转周期不同相位 → finger gaiting 涌现

2. **面向部署的奖励设计** (PenSpin $r_z$): 奖励不仅要让仿真中表现好，还要让生成的轨迹在下游 (开环回放/迁移) 中物理可行。$r_z$ 对仿真指标无影响但对部署至关重要

3. **训练几何体是步态设计旋钮** (DexScrew): 厚三角→高间隙步态，球形→保守步态。不需要匹配真实物体——形状决定的是行为特性而非外观相似度

4. **蒸馏方法选择由任务容错性决定**: DAgger 是首选 (尊重因果)，BC 是 fallback (当 DAgger 崩溃时)。不是方法优劣问题，是任务性质问题

5. **关节级 sim-to-real gap 没有想象中大**: 开环回放 ~47% 成功 (PenSpin)，直接迁移 41.6% 进度 (DexScrew)。真正困难的是蒸馏和感知，不是仿真精度

6. **触觉需要时序上下文才有用** (DexScrew): 单帧触觉 + 无历史改进有限，5 帧历史 + 触觉 → 69.2% 到 95.0%

7. **filtered BC 可以超越训练数据**: 只用成功轨迹训练的 BC 策略 > 开环回放原始数据 (PenSpin 60.8% > replay 47%; DexScrew BC 69.2% > replay 50.8%)

### 被高估/低估的贡献

**被高估**:
- DexScrew "不完美仿真" 的 framing: 手指旋转 sim-to-real 本身就能工作，主要提升来自补全手臂协同和触觉反馈

**被低估**:
- HORA 有限差分速度的洞察: v0.0.1→v0.0.2 的 bug fix 揭示了 IsaacGym `angvel` 读取的严重问题，改用四元数差分计算
- PenSpin canonical grasps 的探索瓶颈分析: 这是对 contact-rich RL 探索问题的通用洞察

---

## 5. 代码 Takeaway

### 值得学习

1. **极简自写 PPO**: ~400 行完整 PPO 实现，不依赖外部框架。完全可控、容易理解和调试。适合研究探索阶段
2. **Grasp cache 范式**: 预计算 50k 个稳定初始状态 → 按 scale 分桶 → 训练时随机采样。将初始状态工程和策略训练解耦
3. **有限差分关节速度**: 不信任仿真器的 `dof_vel`，用 `(pos_t - pos_{t-1}) / dt` 手工计算。多个项目验证了这比直接读仿真器更可靠
4. **JIT 导出管线** (DexScrew): PolicyWrapper 将归一化统计量作为 buffer 注册 → `torch.jit.trace` → 单文件部署。干净的 sim→real 桥接
5. **KL-adaptive LR**: 根据策略更新的 KL divergence 自动调节学习率。简单有效，三个项目一致使用
6. **Value bootstrap on timeout**: episode 超时时用 value 估计做 bootstrap，不把 timeout 当作真正的 terminal state
7. **关节顺序映射解耦**: policy 空间和硬件排列通过映射表解耦 (hora2allegro, xhand2policy)，允许两侧独立设计

### 应避免

1. **`eval()` 做动态 dispatch**: `eval(config.train.algo)` 和 `eval(f'self.enable_priv_{name}')` — 应该用字典查找或 registry 模式
2. **copy-paste 继承**: 没有基类/接口抽象，每个项目直接 fork 改文件 — PenSpin 的 14+ 个条件分支是后果
3. **拼写错误传播**: `computer_return` 从 HORA 传播到 DexScrew，三代未修 — 说明无代码审查
4. **config 默认值 ≠ 实际训练值**: HORA `forceScale` config 默认 0 但脚本覆盖为 2。只看 config 文件会误解实际训练设置
5. **注释掉的代码留在 production 中**: PenSpin 有 `# bad engineering because of urgent modification` 注释和 `exit(0)` 的 SparseCNN
6. **Magic numbers**: `0.01` reward 缩放、`1/24` action scale、`1.1` soft bound 等硬编码散落在不同文件中

### 与 MinBC 等人形机器人代码的对比

| 维度 | HORA 系列 | MinBC / 人形代码 |
|------|-----------|-----------------|
| RL 框架 | 自写 PPO (~400 行) | 依赖 rl_games / RSL_RL / Brax |
| 依赖哲学 | 零依赖，完全自包含 | 重基础设施，组件化 |
| 调试难度 | 低 (每行都能追踪) | 高 (框架抽象层多) |
| 团队协作 | 不适合 (无接口/测试/类型注解) | 更适合 (标准化接口) |
| 迭代速度 | 快 (直接改代码) | 慢但更安全 (框架约束) |
| 部署 | JIT trace → 单文件 | 更复杂的 infra |
| 动作解码 | MLP → 关节位置增量 | 可选 Diffusion/ACT/Choice |
| 视觉处理 | 无 (纯 proprio) | ResNet/DINO encoder |

**HORA 系列的极简风格在个人研究者快速迭代时有优势，但不适合团队开发或长期维护。** MinBC 类项目的重基础设施在初期投入高，但长期来看更可持续。

两者共同的问题: 关节映射手工维护、reward 调参 magic numbers 散落、缺乏系统性的 sim-to-real 评估 pipeline。

---

## 6. 总评

**功能上 solid，工程上不 solid。**

三个项目都能跑、能出结果、能在真实机器人上部署——从这个意义上说代码库是 functional 的。但从软件工程角度看，零测试、零类型注解、技术债三代未修、copy-paste 继承，说明这是典型的"一次性研究代码"而非可维护的系统。

**这个系列的最大价值不在代码本身，而在代码揭示的方法论洞察**: 初始状态设计的重要性、蒸馏方法与任务容错性的关系、面向部署的奖励设计、关节级 gap 其实不大的发现。这些洞察跨任务通用，比任何单个项目的代码都更持久。
