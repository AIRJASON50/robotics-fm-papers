# State vs Observation: RL 理论定义与机器人社区的非正式用法

## 正式定义 (Sutton & Barto, POMDP)

| 概念 | 框架 | 定义 |
|------|------|------|
| **State** | MDP | 环境的完整描述, 满足 Markov 性质 (未来只取决于当前 state + action) |
| **Observation** | POMDP | agent 实际感知到的信号, 可能是 state 的部分/含噪映射 |

```
MDP:   agent 直接看到完整 state → 策略 pi(a|s)
POMDP: agent 只看到 observation → 策略 pi(a|o) 或 pi(a|history)
```

## 机器人社区的非正式用法 (非正式行话, 无正式定义)

| 社区用语 | 实际含义 | 输入维度 |
|----------|---------|---------|
| "State-based policy" | 低维数值向量: 本体感觉 + 物体精确位姿 + 仿真特权信息 | 几十到几百维 |
| "Vision-based policy" | 图像/点云 + 本体感觉 | 几万到几十万维 |
| "Privileged state" | 仿真中独有的完整信息 (最接近正式定义的 state) | 低维 |

**核心区分点不是"信息完整性", 而是"输入维度和传感器模态"。**

## 证据: 你的论文库中的 "state-based" 含义

| 论文 | "State-based" 包含什么 | 只有本体感觉? |
|------|----------------------|------------|
| OmniReset (2025) | proprioception + ground-truth 物体 pose + privileged params | 不是 |
| BiDexHD (2024) | 关节角/速度 + 物体位置/朝向/线速度/角速度 | 不是 |
| DexMachina (2024) | object states + joint targets + finger-object distances + 接触力 | 不是 |
| HORA (2022) | proprioception + 物体位置/质量/摩擦/尺寸 | 不是 |

→ Manipulation 的 "state-based" 几乎总是包含物体 ground-truth, 不仅是本体感觉
→ 这些信息在真机上需要视觉/mocap 才能获得, 但在仿真中直接读取

Locomotion 有时 "state-based" 确实仅指 proprioception + 地形信息 (如 HORA 的 student 只用本体感觉)

## 术语演变脉络

```
控制理论: state x (内部动力学) vs output y (可测量)      → 清晰区分
RL 理论:  state s (MDP 假设完全可观)                    → 不区分 state/observation
POMDP:    state s + observation o + observation function → 正式区分
OpenAI Gym API: 一切返回值都叫 "observation"             → 混淆开始
机器人社区: "state" = 低维数值, "vision" = 高维图像        → 按模态分, 非按完整性分
Teacher-Student: teacher 用 "privileged state", student 用 "observation" → 回归正式定义
```

## 对机器人工程师的实际含义

```
当你读到论文说 "state-based RL":
  → 不是"只用了关节角"
  → 而是"在仿真中用了所有可直接读取的低维数值信息 (含物体 ground-truth)"
  → 部署到真机时需要用视觉/触觉替代这些信息 → Teacher-Student distillation

当你设计自己的系统:
  仿真训练 (teacher): 尽可能用完整 state (物体位姿、接触力、物理参数)
  真机部署 (student): 只能用本体感觉 + 相机/触觉
  → "state-based" teacher 的信息优势就是 distillation 的学习信号来源
```

## 参考来源

- Sutton & Barto, Reinforcement Learning: An Introduction
- OpenAI Spinning Up: Key Concepts in RL
- POMDP (Wikipedia)
- OmniReset, BiDexHD, DexMachina, HORA 论文 (本库)
