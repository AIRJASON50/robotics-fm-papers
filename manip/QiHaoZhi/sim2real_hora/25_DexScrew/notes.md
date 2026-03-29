# DexScrew: Learning Dexterous Manipulation Skills from Imperfect Simulations - 论文笔记

**论文**: Learning Dexterous Manipulation Skills from Imperfect Simulations
**作者**: Elvis Hsieh\*, Wen-Han Hsieh\*, Yen-Jen Wang\*, Toru Lin, Jitendra Malik, Koushil Sreenath, Haozhi Qi
**机构**: UC Berkeley
**框架名**: DexScrew
**年份**: 2025
**代码**: HORA → PenSpin → DexScrew 同一 codebase 继承链

---

## 从 PenSpin 到 DexScrew: 阅读引导

PenSpin 的关键发现：
1. 关节级 sim-to-real gap 没那么大（开环回放 ~47% 成功）
2. Oracle 的核心价值是运动先验，不是直接部署
3. BC 虽有 distribution shift 但能转移运动模式，真实数据微调可以补偿

DexScrew 延续这个方向，任务换成螺母紧固和螺丝刀旋拧。论文标题 "Imperfect Simulations" 暗示仿真精度是核心瓶颈，但仔细审查后发现：**手指旋转部分的 sim-to-real 其实已经能工作（螺丝刀 41.6% 进度），真正缺失的是手臂协同和触觉反馈。**

---

## 论文做了什么

### Stage 1: 简化仿真 RL

**仿真简化**: 螺纹用旋转关节 (revolute joint) 近似——螺母只能绕螺栓旋转，不能沿螺栓下降。没有螺旋力学。

**训练几何体是步态设计旋钮** (本文真正新的洞察):
- **螺母任务**: 厚三角形 → 防止策略从底部施力 → 产生高间隙步态 → 泛化到六角/方形/十字
- **螺丝刀任务**: 球形原语 → 保守步态 → 更鲁棒。训练混合八角+十二角手柄
- 训练几何体不需要匹配真实物体——形状决定学到的步态特性

**蒸馏**: Oracle (PPO + 特权信息, 15 亿步) → DAgger 蒸馏 student

**为什么 DAgger 又能用了？** PenSpin 中 DAgger 崩溃是因为笔一步就掉。螺母在螺栓上、螺丝刀在手中——物体有物理约束不会掉，student 做得不好也能跑完 episode → 有足够有效数据 → DAgger 迭代能启动。

**DexScrew 的蒸馏比 PenSpin 更接近原版 RMA**: dual loss (latent MSE + action MSE)，PenSpin 关掉了 latent loss 只用 action BC。

### Stage 2: 技能辅助遥操作

把 student 策略嵌入遥操作系统：
- 人用 VR 摇杆控制手腕位置
- 按按钮激活 student 的手指旋转
- 人不直接操作手指关节

**本质上就是 student 能力不完整（只会转不会推进），人工补上手臂的部分。** 论文包装为 "skill primitive"，实际就是一个不完整的 student + 人工补缺。

**同时采集触觉数据**: XHand 内置触觉传感器 (5 指 x 120 传感器 x 3 轴力)，遥操作过程中自动记录。PenSpin 放弃了触觉（sim-to-real gap 太大），DexScrew 通过在真实世界直接采集绕过了仿真触觉 gap。

数据量: 螺母各 50 条 (~80s/条)，螺丝刀 72 条 (120-180s/条)。

### Stage 3: 多传感器 BC

Hourglass 编码器, 5 步历史, 触觉 MLP 编码后融合, Action chunking H=16 步。不使用视觉。

---

## 重新审视：贡献到底有多大

### 论文自己的数据揭示了真实情况

```
螺母任务:
  直接 sim-to-real: "can rotate the nut" → 手指旋转本身成功
  缺失: "cannot drive the nut downward because the arm does not move" → 只缺手臂推进

螺丝刀任务:
  直接 sim-to-real: 41.6% 进度 → 论文原话 "meaningful behavior"
  专家回放: 50.8% 进度
  BC (无触觉无历史): 69.2%
  BC + 触觉 + 历史: 95.0%
```

**手指 RL 部分已经有不错的成功率。** 真正的提升来自：
1. 人工通过遥操作补上了手臂向下推进（仿真中没有螺纹所以没学到）
2. 触觉 + 时序历史让 BC 策略能做精细调整

### "不完美仿真" 这个 framing 的审视

论文标题暗示核心挑战是仿真不精确。但实际情况是：
- 手指旋转的 sim-to-real 本身就能工作（和 PenSpin 的发现一致——关节级 gap 不大）
- 真正缺失的不是 "仿真不够精确"，而是 "仿真没有建模手臂推进"（因为没有螺纹力学）和 "仿真没有触觉"
- Stage 2-3 做的是**补全仿真没覆盖的部分**（手臂+触觉），不是**修正仿真不精确的部分**

更诚实的 framing: **仿真只覆盖了任务的一部分（手指旋转），遥操作补全了另一部分（手臂协同），触觉 BC 提供了精细调整。**

### 真正的贡献 (从 PenSpin 的增量来看)

1. **训练几何体作为步态设计旋钮** — 厚三角→高间隙，球形→保守。PenSpin 没有这个维度的探索。这是有意义的新洞察
2. **触觉 + 时序历史的互补性** — 单独加任一改进有限，同时加效果巨大 (69.2% → 95.0%)。重要的实验发现
3. **技能辅助遥操作** — 把 student 作为遥操作中的手指控制器，人只控手臂。工程上比 PenSpin 的开环碰运气更系统化

增量没有论文标题暗示的 "不完美仿真哲学" 那么大。PenSpin 已经证明了关节级 gap 不大、oracle 价值在先验。DexScrew 进一步简化仿真（去掉螺纹），但代价只是缺了手臂推进——通过遥操作轻松补上。

---

## 关键结果

### 螺母紧固

| 方法 | 三角螺母 | 六角 (未见) | 十字 (未见) |
|------|---------|-----------|-----------|
| 直接 sim-to-real | 能转但不能下沉 | - | - |
| BC (无触觉无历史) | 基线 | ~30% | ~60% |
| **BC + 触觉 + 历史** | **最佳** | **~65%** | **~80%** |

### 螺丝刀

| 方法 | 进度比 |
|------|--------|
| 直接 sim-to-real | 41.6% (从不完成) |
| 专家回放 | 50.8% (从不完成) |
| BC 基线 | 69.2% |
| **BC + 触觉 + 历史** | **95.0%** |

BC 超越专家回放 (69.2% vs 50.8%): filtered BC 效应，与 PenSpin 的 "student 超越开环回放" 同一原理。

---

## 与 PenSpin 的关键对比

| 维度 | PenSpin | DexScrew | 为什么不同 |
|------|---------|---------|-----------|
| 仿真精度 | 基本准确 | 故意简化 (无螺纹) | 缺了手臂推进但手指旋转够用 |
| 蒸馏方法 | Teacher-rollout BC (被迫) | DAgger + latent 对齐 | 螺母不会掉 → 容错高 → DAgger 可行 |
| 真实数据获取 | 开环回放碰运气 | 人工引导遥操作 | 需要手臂协同，开环不够 |
| 触觉 | 不使用 | 核心模态 | 在真实世界直接采集绕过仿真 gap |
| Latent loss | 关掉 | 启用 (dual loss) | DAgger 可行 → 回到 RMA 路线 |

---

## 触觉 + 时序历史: 为什么互补

这是 DexScrew 最有实验价值的发现:

```
单帧触觉: "传感器 42 被压了 0.3N" → 不知道力在增大还是减小、是滑动还是稳定接触
5 帧触觉历史: "力从 0.1→0.2→0.3→0.4→0.5" → 接触力稳定增大，说明螺母在正确啮合

单独加触觉: 改进有限 (缺时序上下文)
单独加历史: 改进有限 (缺接触信息)
同时加: 69.2% → 95.0%
```

---

## 非显而易见的洞察

1. **训练几何体是设计旋钮**: 厚三角→高间隙步态; 球形→保守步态。训练物体不需要匹配真实物体
2. **DAgger 回归说明方法选择是任务驱动的**: PenSpin 被迫放弃，DexScrew 用回来——不是方法变好了，是任务容错性变了
3. **BC 可以超越训练数据**: filtered BC 效应，与 PenSpin 同一原理
4. **触觉只有配合时序上下文才有用**: 对所有触觉相关工作都有启示
5. **"不完美仿真" 的实际增量**: 手指旋转 sim-to-real 本身就能工作 (41.6%)，论文的主要提升来自补全手臂协同和引入触觉反馈，不是克服仿真不精确

---

## 作者展望

1. 全自主数据收集 (无需人类遥操作)
2. 扩展到更多接触密集型任务
3. 结合视觉实现长时域装配 (拾取→对准→穿入→紧固)
4. 利用高精度力/力矩传感进一步提升精度

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| 手指 mask | 未详述 | ScrewDriver: pinky+ring 置零; NutBolt: 4 DOF 置零 |
| 螺母关节 | 未提及 | `F.pad(actions, (0,1), value=0.0)` → 完全靠接触驱动 |
| reverse_penalty | 未提及 | 计算了但未加到 reward (死代码) |
| Reward 缩放 | 未提及 | `shaped_rewards = 0.01 * rewards` |
| 部署时点云 | 未提及 | 设为零向量 → 真机不使用 |

### 代码设计

1. **继承链**: 与 PenSpin 同 codebase，理解 PenSpin 代码就理解 DexScrew
2. **ProprioAdapt**: 冻结 teacher，只训练 adapt_tconv，student on-policy + dual loss。比 PenSpin 的 DemonTrain 更接近原版 RMA
3. **adapt_tconv 输入维度变化**: XHand 12 DOF → 每帧 24D (12 joint + 12 target)，不同于 HORA/PenSpin 的 32D (16+16 Allegro)
4. **JIT 导出**: `student_eval.py` → `torch.jit.trace` → PolicyWrapper (含归一化统计量)
5. **关节映射**: `XHAND_TO_POLICY_IDX = [3,4,5,6,7,10,11,8,9,0,1,2]` 处理策略↔硬件排列差异
6. **部署频率**: 底层 200Hz (5ms/step)，策略推理 20Hz (每 10 步查询一次)

---

## 局限

- 仍需人类遥操作 (Stage 2)
- 物体需预置 (螺母已在螺栓上，螺丝刀已插入)
- 不使用视觉
- 仅两个任务
- 硬件特定 (XHand)
