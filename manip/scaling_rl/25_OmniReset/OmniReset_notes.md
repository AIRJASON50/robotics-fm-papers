# OmniReset: Emergent Dexterity via Diverse Resets -- 学习笔记
> 一句话: 通过程序化生成 4 类多样 reset states, 让标准 PPO 无需 demo/curriculum/reward shaping 就能解决 long-horizon contact-rich 操作
> 论文: Patrick Yin*, Tyler Westenbroek* et al. UW + NVIDIA + Microsoft Research, 2026

## 这篇论文解决了什么问题
Massively parallel RL 在灵巧操作上遇到 **exploration saturation**:
- 增加并行 env 数量后性能快速饱和 -- 所有 env 都在探索同一片状态空间
- Long-horizon 任务 (pick -> reorient -> insert) 中 agent 发现不了完整路径
- 现有方法依赖 per-task reward shaping / curriculum / demo, 不可扩展

核心问题: **机器人操作能否也像 LLM 一样, 靠简单算法 + scale 产生涌现能力?**

## 核心想法 (用直觉解释)
与其让 agent 自己探索到关键中间状态, 不如通过 simulator reset 直接把 agent "放到"这些状态上。
具体做法: 离线预计算 4 类 reset 分布, 均匀采样:
- **Reaching**: 物体在桌面随机, gripper 在空间随机
- **Near-Object**: gripper 在物体附近的 grasp point (+offset)
- **Stable Grasp**: 物体在空中, gripper 已稳定抓取
- **Near-Goal**: 物体在目标附近 (partial assembly offsets)

直觉: 这些 reset 覆盖了"物体路径"和"交互模式"两个轴。4 类状态之间没有预定义连接, 路径完全由 RL 自己发现。Dense coverage 使 sparse reward 信号能顺畅传播。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| Reset 策略 | 4 类程序化生成, 无 demo 无 curriculum | 核心卖点: reset 覆盖度 > reward shaping |
| Reward | Task-agnostic: r_reach + r_dist + r_success + r_smooth | 所有任务共享同一 reward, 权重固定 |
| 探索 | gSDE (state-dependent exploration) | Reaching 阶段大探索, insertion 阶段精调 -- 自适应 |
| Scale | 4096 并行 env, 4x L40S GPU | Ablation: 512->4096 性能持续提升, 大 batch 防 catastrophic forgetting |
| Sim-to-real | Two-stage: Train (ideal dynamics) -> Finetune (sysid + ADR curriculum) | 论文说"no curriculum"仅指 Stage 1; sim2real 实际需要 finetune |
| Distillation | 80K expert rollouts -> ResNet-18 + MLP RGB policy | 瓶颈: RGB policy ~50% vs state expert ~100% |

## 这篇论文之后发生了什么
- 开源 UWLab 框架 (Isaac Lab 扩展), 代码含 DAgger / FastSAC 等论文未提的功能
- Emergent behavior: drawer insertion 中 policy 自学了"翻转推入"而非 pick-and-place
- Retrying behavior 在真机上被验证 -- RL 的天然 closed-loop 优势
- RGB distillation 仍是主要瓶颈, 50% 到 real-world 成功率说明 vision gap 待解

## 论文展望与局限 (Section 6 + Appendix)

| 局限 | 影响 |
|------|------|
| Grasp sampler 依赖 | 复杂非凸物体 grasp 采样会失败, reset 质量受限 |
| 双手/灵巧手未验证 | 为多指手预计算 stable grasp 更难, 能否 scale 到灵巧手是 open question |
| Domain randomization 幅度有限 | 用了"relatively modest"的 DR, 更大范围工况适应需要更多工作 |
| RGB distillation gap | 仿真内 RGB policy 只有 ~50% (vs state expert ~100%), 视觉是主要瓶颈 |
| Seed selection | 训多个 seed 选最好的, 基于 noise-robustness proxy metric -- 有手工调选的成分, 削弱了"minimal engineering"的 claim |
| 接触模型精度 | 接触密集行为的 sim-to-real 迁移仍不可靠 |
| Distillation scaling 未充分探索 | 80K 轨迹时性能仍在提升, 计算预算限制了进一步 scale |

---

## Paper vs Code 关键不一致

### 1. "No curriculum" -- 仅限 Stage 1, sim2real 的 Stage 2 有完整 curriculum

论文主文 claim "does not rely on curricula to stabilize and accelerate learning"。**但部署到真机的完整 pipeline 是两阶段:**

```
Stage 1 (论文 Section 4 的结果): 无 curriculum, OmniReset + PPO, 确实如 claim
Stage 2 (论文 Section 5 sim2real): FinetuneCurriculumsCfg 包含:
  - ADR curriculum: 根据成功率 (up=0.95, down=0.9) 逐步增大 DR 幅度 (摩擦/armature/延迟)
  - Action scale curriculum: 逐步缩小动作尺度 (从 0.02 缩到 0.002)
  - Obs noise curriculum: 逐步增大观测噪声
  - Actuator swap: 从理想驱动器切换到系统辨识后的真实驱动器模型 (含摩擦/延迟)
```

主文说"no curriculum"是对 Stage 1 而言, 技术上没错, 但**读主文的人会以为整个 pipeline 都不需要 curriculum, 这是误导性的**。Appendix A.3.9 有解释但很容易被忽略。

### 2. Reset 类型与论文描述不完全匹配

| 论文描述 | 代码实际 | 差异 |
|---------|---------|------|
| $S^R$ Reaching: 物体在桌面, gripper 随机 | `ObjectAnywhereEEAnywhere`: 物体不仅在桌面, 还包括空中 (高度达 0.3m), 全随机朝向 | 代码比论文更激进 |
| 均匀采样 | `probs: [0.25, 0.25, 0.25, 0.25]` | 确认一致 |

### 3. "Task-agnostic reward" -- 结构一致但 success 阈值是 per-object 的

Reward 权重和结构确实所有任务相同, 但 success 判定的阈值 (position_threshold, orientation_threshold) 从物体 metadata 文件加载, 每个物体不同。严格来说 reward 函数本身是 agnostic 的, 但"什么算成功"是 task-specific 的。

### 4. gSDE 实现与论文描述不一致

论文强调 gSDE 提供 "state-dependent exploration"。但代码中 `state_dependent_std=False`, 实际上 noise std 并不依赖 state features, 只是一个独立的可学习参数。这削弱了论文对 gSDE 优势的论述。

### 5. DAgger 已有完整实现, 论文称 "leave to future work"

代码中 `Base_DAggerRunnerCfg` + `RslRlFancyPpoAlgorithmCfg` 已经实现了 DAgger (BC loss + PPO), 所有 RGB 环境注册了 DAgger runner。论文 Appendix 说"leave this to future work", 但代码表明他们实际已经实验过, 只是选择不报告结果。

### 6. Reward 的 sparse vs dense 比例

```
实际 reward 量级:
  success_reward:        weight=1.0, value=0/1     → 最大贡献 1.0
  dense_success_reward:  weight=0.1, value∈[0,1]   → 最大贡献 0.1
  ee_asset_distance:     weight=0.1, value∈[0,1]   → 最大贡献 0.1
  action/vel penalties:  weight=1e-4~1e-2           → 接近可忽略
  abnormal_robot:        weight=-100                → 硬约束, 不是 shaping

→ sparse reward (success) 比 dense shaping 强 5-10x
→ 验证了论文核心 thesis: reset 覆盖度比 reward shaping 更重要
```

---

## Code 中的隐藏 Insight

### 1. `ProgressContext` reward 返回全零 -- 是监控模块伪装成 reward

`ProgressContext.__call__()` 始终返回 `torch.zeros()`, weight=0.1 但实际贡献 = 0。它的真实作用是在 reward manager 的生命周期中维护共享状态 (xyz_distance, success_counter 等), 供其他 reward 和 curriculum 消费。这是一种工程 trick -- 把状态管理嵌入 reward framework。

### 2. 训练和部署的 OSC 增益差 5 倍

```
训练: Kp = (200, 200, 200, 3, 3, 3), damping_ratio = (3, 3, 3, 1, 1, 1) -- 软/柔顺
部署: Kp = (1000, 1000, 1000, 50, 50, 50), damping_ratio = (1, 1, 1, 1, 1, 1) -- 硬/精确
```

训练时用低刚度鼓励探索, 部署时切到高刚度保证执行精度。这个 5x 增益变化论文未提及。

### 3. Reset 状态验证流程比论文描述复杂得多

代码中 reset state 生成后需要通过:
- 稳定性检查: 连续 5 步低速度
- SDF 碰撞检查: 1024 采样点/对
- Gripper 朝向: 必须偏离竖直方向 <60°
- 位置偏移限制
- Assembly 对齐随机过滤: `assembly_success_prob=0.5`, 控制 near-goal 的难易比例

### 4. 存在 failure-weighted reset sampling 的基础设施但未启用

`SuccessMonitor` 按 reset 类型维护 100 步滑窗成功率, 有 `failure_rate_sampling` 接口可实现"失败多的 reset 类型采样更多", 但当前 config 未启用, 仍用均匀采样。说明团队探索过自适应采样但最终选择了简单均匀方案。

### 5. PPO 网络比论文暗示的更大

论文未给出网络结构。代码: actor 和 critic 均为 `[512, 256, 128, 64]` + ELU, 比 ArtiGrasp 的 `[128, 128]` 大得多。这对 4096 并行 env 的大 batch 训练是合理的。

---

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Initial state distribution > reward engineering | 和 RLHF 类比: 好的 prompt 分布比复杂的 reward model 更能引导行为 |
| 2 | Scale 在 RL manipulation 中开始起作用了 (4096 env 持续提升) | 但和 LLM 不同: manipulation 的 scale 更依赖 state coverage 而非数据量 |
| 3 | Emergent behavior (翻转推入等) 真的出现了 | 类比 in-context learning: 没有显式教, 但足够多的"见过"就能涌现 |
| 4 | RGB distillation 是 sim-to-real 的主要瓶颈, 不是 dynamics gap | Robotics FM 的 vision encoder 质量直接决定 sim-to-real 成功率 |
| 5 | "No curriculum" 的 claim 需谨慎对待 | Stage 1 确实无 curriculum, 但 sim2real 部署需要完整的 ADR + actuator swap pipeline |
| 6 | 论文的 core thesis (reset > reward) 被代码的 reward 权重验证 | sparse success (1.0) 远大于 dense shaping (0.1), shaping 几乎可忽略 |
| 7 | 对灵巧手的适用性是 open question | 论文明确指出多指手的 grasp sampling 和 stable grasp 预计算是未解决的挑战 |
