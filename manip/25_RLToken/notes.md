# RL Token (RLT) - 论文笔记

**论文**: RL Token: Bootstrapping Online RL with Vision-Language-Action Models
**机构**: Physical Intelligence (PI)
**作者**: Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke
**基础模型**: $\pi_{0.6}$ (PI 自家 VLA)

---

## 一句话总结

冻结大型 VLA，训练一个小型 encoder-decoder 从 VLA 内部特征中提取紧凑的 "RL token" 作为状态表示，然后用轻量级 off-policy actor-critic 在线精调 VLA 的动作块，实现真实机器人上数小时内的精度提升。

---

## 核心问题

VLA 模型（如 $\pi_0$, $\pi_{0.6}$）在泛化性上很强，但在精度关键任务（亚毫米级插入等）上表现受限于演示数据的质量。直接用 RL 微调整个 VLA 代价太大（参数量大、样本效率低），而传统 real-world RL（如 SERL）从头训练小模型又无法利用 VLA 的先验。

**核心矛盾**: VLA 的泛化能力 vs. 轻量级在线 RL 的样本效率，如何兼得？

---

## 方法架构

```
                    ┌──────────────────────────┐
                    │     Frozen VLA ($\pi_{0.6}$)    │
                    │  (images + lang + prop)  │
                    └──────────┬───────────────┘
                               │ VLA embeddings z_{1:M}
                    ┌──────────▼───────────────┐
                    │   RL Token Encoder g_φ   │
                    │  (lightweight transformer)│
                    │   append <rl> token       │
                    └──────────┬───────────────┘
                               │ z_rl (compact vector)
                    ┌──────────▼───────────────┐
          ┌─────────│   RL Actor π_θ           │
          │         │   input: (z_rl, s^p, ã)  │──── a_{1:C} (action chunk)
          │         └──────────────────────────┘
          │
          │  regularize      ┌──────────────────┐
          └──────────────────│  RL Critic Q_ψ   │
                             │  input: (x, a)   │
                             └──────────────────┘
```

### 两阶段流程

**阶段 1: RL Token 训练 (离线, 少量任务数据)**
- 在 VLA 最后一层嵌入上添加 encoder-decoder transformer
- Encoder 末尾附加可学习 `<rl>` token，输出位置即为 RL token $\mathbf{z}_{\text{rl}}$
- Decoder 自回归重建原始 VLA 嵌入 → 信息瓶颈迫使 RL token 压缩关键信息
- 损失: $\mathcal{L}_{\text{ro}}$ (重建) + 可选 $\alpha \mathcal{L}_{\text{vla}}$ (VLA 微调)
- 训练 2000-10000 步后冻结

**阶段 2: 在线 RL (真实机器人)**
- 冻结 VLA + RL token encoder
- Actor: 输入 $(z_{\text{rl}}, s^p, \tilde{a}_{1:C})$，输出精调后的动作块
- Critic: 标准 TD3 双 Q 网络
- 关键设计:
  - **动作块 (C=10)**: 缩短有效决策时域，利于稀疏奖励传播
  - **参考动作条件化**: actor 以 VLA 采样动作为输入，局部编辑而非从零搜索
  - **BC 正则化**: $\beta \| a - \tilde{a} \|_2^2$ 锚定于 VLA 动作
  - **参考动作 dropout**: 训练时 50% 概率将参考动作置零，防止简单复制

---

## 关键设计决策及其原因

| 设计 | 为什么 | 消融影响 |
|------|--------|----------|
| RL token (而非 ResNet) | 保留 VLA 预训练的操控相关结构 | 移除后吞吐量降低 50% |
| Action chunk (C=10, 而非 C=1) | 稀疏奖励下缩短信用分配距离 | 单步动作完全无法学习 |
| BC 正则化 ($\beta > 0$) | 防止探索偏离太远导致不稳定 | 移除后性能下降最大 |
| 参考动作直通 | 利用 VLA 多模态分布信息加速学习 | 移除后学习变慢、早期失败增多 |
| 关键阶段聚焦 | 集中数据和信用分配于精度瓶颈 | 提高样本效率 |

---

## 实验结果摘要

**任务**: 螺丝安装、扎带紧固、以太网插入、充电器插入 (均需亚毫米精度)

| 指标 | 改进 |
|------|------|
| 关键阶段速度 | 提升高达 $3\times$ |
| 螺丝安装成功率 | 20% → 65% (关键阶段) |
| 扎带成功率 (full-task) | 提升 60% |
| 以太网完成步数 | 减少 $2\times$ |
| 训练数据量 | 15 分钟 ~ 5 小时真实机器人数据 |

**vs. 基线**:
- HIL-SERL / PLD: 单步 RL 方法在长时域稀疏奖励下完全失败
- DSRL: 成功率相当但速度提升有限（强约束于 VLA 动作空间）
- DAgger: 速度受限于人类演示
- RLT: 唯一同时提升成功率和速度的方法

**涌现行为**: RL 策略学到了演示数据中不存在的策略——如以太网插入时施加压力并摇晃利用柔顺性，而非 VLA 的反复"探测"行为。甚至超越了人类遥操作速度。

---

## 与其他 VLA RL 微调方法的对比

| 方法 | 更新对象 | 动作粒度 | 在线/离线 | 真实机器人 |
|------|----------|----------|-----------|------------|
| RECAP ($\pi_{0.6}^*$) | 全 VLA | chunk | 离线 | 是 |
| SimpleVLA-RL | 全 VLA | - | PPO (on-policy) | 否 |
| ConRFT | action head | 单步 | 在线 | 是 (短任务) |
| Policy Decorator | residual | 单步 | 在线 | 仿真 |
| PLD | residual | 单步 | 在线 | 是 |
| DSRL | latent noise | chunk (隐式) | 在线 | 是 |
| **RLT** | **小型 actor-critic** | **chunk (C=10)** | **在线** | **是** |

RLT 的独特之处: RL token 压缩 + chunk-level RL + 参考动作条件化 + BC 正则化的组合。

---

## 网络细节

- **RL token**: 轻量级 encoder-decoder transformer (具体参数未公开)
- **Actor**: 2 层 MLP (256 hidden) 或 3 层 MLP (512 hidden, 螺丝任务)
  - 高斯策略, 固定小标准差
  - 输入: RL token + 本体感受 + VLA 参考动作块
  - 输出: 动作块 $\mathbf{a} \in \mathbb{R}^{C \times d}$, C=10, d=14
- **Critic**: TD3 双 Q 网络, 同 actor 架构
- **控制频率**: 50 Hz
- **Update-to-data ratio**: 5
- **Critic/Actor 更新比**: 2:1
- **动作块子采样**: stride=2, 每秒约 25 个样本
- **参考动作 dropout**: 训练 50%, 推理 0%

---

## 局限性

1. 仍需人工介入: 奖励标注、干预修正、策略切换时机
2. 仅优化关键阶段, 全任务仍依赖 base VLA
3. 实验限于 PI 自家 $\pi_{0.6}$ 模型, 泛化到其他 VLA 未验证
4. 任务设置相对受控 (固定工位、固定物体)

---

## 对我们项目的启示

1. **表征压缩思路**: RL token 的信息瓶颈思想可借鉴——从大型预训练模型中提取紧凑表示供小型策略使用
2. **Action chunk RL**: 在高频控制 + 稀疏奖励场景下，chunk-level RL 比 step-level RL 更有效
3. **参考动作条件化 + BC 正则化**: 有预训练策略时，RL 做局部编辑比从零搜索高效得多
4. **关键阶段聚焦**: 不必对整个任务做 RL，识别瓶颈阶段集中优化
5. **参考动作 dropout**: 防止策略退化为简单复制的实用技巧
