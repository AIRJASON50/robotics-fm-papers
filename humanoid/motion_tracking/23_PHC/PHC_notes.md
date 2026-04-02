# PHC: Perpetual Humanoid Control -- 学习笔记
> 一句话: 通过渐进式多 primitive 学习 + 跌倒恢复, 首次在 AMASS 全集 (11K+ clips) 上达到 100% motion tracking 成功率
> 论文: Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, Weipeng Xu. ICCV 2023

## 这篇论文解决了什么问题
Physics-based humanoid 想模仿上万条 MoCap 动作, 面临两个死结:
1. **Scalability**: 单一 policy 学不完所有动作 -- 序列数上万后 catastrophic forgetting 严重, 成功率断崖
2. **Perpetual control**: 真实输入 (视频/语言) 带噪声, humanoid 不可避免会跌倒, 但之前的系统跌倒 = 结束

## 核心想法 (用直觉解释)
把"一个 policy 学所有动作"拆成多个 primitive 渐进式学习:
- Primitive 0 学全部数据, 等它学不动了 (plateau)
- Primitive 1 **只学 Prim 0 失败的子集**, 参数从 Prim 0 复制并 freeze Prim 0
- 重复直到没有失败序列; 最后一个 primitive 专门学"从地上爬起来"
- 推理时一个 Composer 网络输出 softmax 权重, 把所有 primitive 的 action 加权混合

直觉: 不是 curriculum "由易到难", 而是每个 primitive 专注"前一个学不会的东西"。加上 fail-state recovery, 系统可以"永不停歇"。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| 多 primitive 组合 | PNN + MCP Composer (softmax 加权) | 避免 catastrophic forgetting, 新 primitive 不干扰旧的 |
| Fail recovery | 注入 fall state (随机旋转+自由下落), zero-out-far 机制 | 远距离只给导航奖励, 近距离才给 imitation reward, 自然分解"起身->走回->继续" |
| Reward | Imitation (pos+rot+vel+angvel) + AMP discriminator, 各 50% | 双信号互补: task reward 追精度, AMP 保自然度 |
| Auto-PMCP soft | 自动调整 motion 采样概率 (hard negative mining) | 代码默认开启, 替代手动 PMCP 流程 |

## 这篇论文之后发生了什么
- **PHC+ / PULSE (2024, 同作者)**: 在 PHC+ 基础上加 language-conditioned control
- **SONIC (NeurIPS 2025, NVIDIA)**: 去掉 PMCP + AMP, 靠更大网络 (42M) + 更多数据 (100K clips) 直接 scale -- 证明"数据+容量" > "渐进训练"
- **HDMI (2025, CMU)**: PHC 思路扩展到人-物交互场景
- 后来 single primitive 就达到 99.9%, PMCP 的核心遗产其实是 **getup recovery 能力**

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Progressive learning 是 scaling 的过渡方案, 最终被"更大模型+更多数据"取代 | 和 LLM 的演进一模一样: 早期靠 curriculum/progressive, 后来 Chinchilla scaling 就够了 |
| 2 | Fail-state recovery 通过训练数据注入 (而非 rule-based) 实现 | Robotics FM 需要 recovery 能力, 应该在训练分布中包含 failure states |
| 3 | auto_pmcp_soft = hard negative mining 的 RL 版本 | 和 RLHF 中 rejection sampling 异曲同工: 让模型多练它不擅长的 |
| 4 | AMP disc 50% + task reward 50% 的双信号设计 | 类比 RLHF: reward model (naturalness) + task objective 共同引导 |
