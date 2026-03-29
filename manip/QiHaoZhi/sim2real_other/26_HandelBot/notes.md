# HandelBot: Real-World Piano Playing - 论文笔记

**论文**: HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies
**作者**: Amber Xie (Stanford), Haozhi Qi (Amazon FAR), Dorsa Sadigh (Stanford)
**发表**: arXiv:2603.12243v2, 2026
**项目**: https://amberxie88.github.io/handelbot

---

## 一句话总结

仿真 PPO 训练钢琴弹奏 → 提取最佳开环轨迹 → 真实世界启发式精修横向关节 → 残差 TD3 在线适应，30 分钟真实数据实现毫米级精度的双手钢琴弹奏，F1 比纯仿真策略提升 1.8x。

---

## 核心问题

钢琴弹奏需要毫米级精度 + 快速独立手指运动：
- 遥操作 20-DOF 灵巧手弹钢琴不可行
- 纯仿真策略精度不足 (sim-to-real gap 在毫米级被放大)
- **闭环仿真策略反而比开环更差** (dynamics gap + 误差累积)

---

## 三阶段方法

### Stage 0: 仿真 RL
- ManiSkill + PPO，512 并行环境，40M steps
- **末端执行器轨迹脚本化** (由乐谱自动计算)，只学手指控制
- 只用 3 个手指 (食指/中指/无名指)
- 奖励: key press (按对+不按错) + fingering (指尖到目标键高斯容差) + action L1
- 选 F1 最高的轨迹作为开环基础

### Stage 1: Policy Refinement (启发式精修)
- 核心发现：**横向关节 (0 号关节) 是 sim-to-real gap 主要来源**
- 在真实钢琴上开环执行 → 对比目标键 vs 实际按键 → 计算有符号方向误差 → 调整横向关节
- Chunked updates + 退火 + 动量阻尼 + 邻指关联修正 (0.3x)

### Stage 2: Residual RL (残差 TD3)
- 在精修轨迹上叠加残差策略 (additive correction)
- TD3, off-policy, 左右手独立训练
- MIDI 键盘输出作为奖励信号 (无需视觉)
- Guided noise: 50% 概率调整噪声符号朝向正确键
- **30 分钟真实数据**即可完成适应

---

## 关键结果

- 5 首歌曲上一致最高 F1，比纯仿真提升 **1.8x**
- 闭环 sim 策略 (pi_sim CL) **比开环更差** → 违反直觉
- RL from scratch (纯真实世界) 在某些歌曲上能匹敌 pi_sim → 末端脚本化是关键简化
- Stage 1 精修本身已显著提升，Stage 2 残差 RL 进一步改善

---

## 作者展望

1. 学习末端执行器运动 (含旋转)，利用拇指和小指弹更复杂曲目
2. 用 VLM 替代人工启发式做策略精修 → 推广到其他任务
3. 残差 RL 扩展到控制末端执行器
4. 处理更复杂音乐作品

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| Actor 激活函数 | 未指定 | **Tanh** (非 ReLU) |
| TD3 Q 网络输出 | 标准 | `(tanh(x)+1)*6` 限制到 [0,12] + LayerNorm + Dropout(0.5) |
| 探索噪声 | 高斯 | **Colored noise** (beta=0.2 时间相关) |
| Per-joint 噪声 | 未提及 | 横向关节噪声更大 (0.04 vs 0.015)，动作上界也更宽 (0.22 vs 0.15) |
| Key-on 课程 | 未提及 | `[[0.7, 10000], [0.5, -1]]` 权重课程 |
| Action chunking | 未提及 | `chunk_size=2` (每动作持续 2 步) |
| Actor-Learner | 未描述 | ZMQ 分离架构，UTD=8 |
| 精修细节 | 简略 | Per-chunk 退火 (改进 x0.7/未改进 x0.97) + 动量阻尼 (0.7) + 按对键邻键反修正 (0.15) |

### 值得学习的代码设计

1. **开环→闭环渐进适应**: 仿真闭环 → 开环提取 → 启发式优化开环 → 受约束闭环残差。约束探索空间加速真实世界学习
2. **PyRoki IK 安全层**: 半平面约束近似键盘表面，执行层面硬约束 (比奖励惩罚更安全)
3. **手指-键归属推断**: 基于规则的 chunk 分配系统 (按键索引分手 → 空间聚类 → 手指顺序分配)
4. **Tolerance 函数奖励**: 高斯 sigmoid 对二值奖励连续化，提供更好梯度信号
5. **CUDA Graph + torch.compile**: PPO 训练中用 CudaGraphModule 减少 kernel launch overhead

---

## 非显而易见的洞察

1. **闭环比开环差**: 高精度任务中观测 gap 的闭环误差累积比开环"盲执行"更致命
2. **RL from scratch 并不差**: 脚本化末端运动作为结构先验已足够强，手指控制的 RL 维度不高
3. **横向关节是 gap 主来源**: 主要误差是 Y 方向 (左右) 偏移，纵向 (按压) 误差次要
4. **退火 + 动量阻尼**: 策略精修是迭代优化过程，需要控制论思维 (退火避振荡，阻尼防过冲)
