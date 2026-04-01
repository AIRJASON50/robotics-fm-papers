# BeyondMimic -- 学习笔记
> 一句话: Compact motion tracking + latent diffusion model，实现 "训练时学技能，测试时组合技能解新任务"，首次在真实人形机器人上 zero-shot 完成 unseen downstream tasks
> 论文: Qiayuan Liao et al. (UC Berkeley + Stanford), arXiv 2508.08241, 2025

## 这篇论文解决了什么问题
DeepMimic-style motion tracking 能学到很自然的单个动作，但有两个根本缺陷: (1) motion-specific tuning -- 每个新动作都要调参; (2) 没有 versatility -- 学会了空翻和走路，不代表能"先走再翻再走"或者"边走边避障"。AMP 虽然学了 style 但每个 task 要重新训练。VAE-based 方法 (如 ASE) 在新任务上会 out-of-distribution 抖动。BeyondMimic 的目标: 一个 framework 同时搞定 agility (高动态动作)、naturalness (像人)、versatility (组合技能解新任务)。

## 核心想法 (用直觉解释)
**两阶段架构**:
- **Stage 1 - Scalable Motion Tracking**: 用一套固定的 reward + 超参训练所有动作 (每个动作一个 policy)。关键: 不靠大量 domain randomization 和 reward engineering，而是靠精确建模执行器 + 减少部署延迟，使得一个 compact formulation 就够用。
- **Stage 2 - Latent Diffusion for Composition**: 把所有 single-skill policy 的 rollout 数据训成一个 state-action co-diffusion model。这个 diffusion model 学到了"可行动作的分布"。部署时用 **classifier guidance** 做 test-time optimization -- 给一个新的 cost function (如速度跟踪、避障)，diffusion 的 denoising 过程会自动调整生成的 trajectory 去满足这个 cost。

直觉: 把 diffusion model 当作 "会做很多动作的大脑"，classifier guidance 当作 "当前任务需求"，两者结合就像人类 "我会跑会翻，现在需要绕过障碍物，那我就组合着来"。

## 关键设计决策
1. **Compact tracking formulation**: 只用 3 个 regularization term (joint limit, action rate, self-collision) + 一个 unified task reward (body tracking error 的 Gaussian kernel)。所有动作共享同一套权重和超参。关键 insight: sim2real gap 的主要来源不是物理参数不确定性，而是执行器模型不准和部署延迟。把这两个修好后，不需要 heavy domain randomization。
2. **Anchor-relative tracking**: 追踪目标用 anchor body (如 pelvis) 的局部坐标系表达，允许全局漂移但保持动作 style。这使得扰动下 policy 不会因为全局位置偏差拿不到 reward。
3. **State-action co-diffusion**: diffusion model 同时预测未来 state 和 action (不只是 action)。这使得 classifier guidance 可以对未来 state 施加 cost (如障碍物距离)，形成 predictive control。这是区别于纯 action diffusion policy 的关键。
4. **Classifier guidance = test-time task specification**: velocity tracking、waypoint navigation、obstacle avoidance 都只是不同的 cost function，在 denoising 时通过梯度注入实现。不需要重新训练。多个 cost 可以直接相加实现 task composition。
5. **C++ deployment framework**: 全部在 C++ 实现低延迟推理，即使 diffusion 的 iterative denoising 也能实时运行。

## 这篇论文之后发生了什么
- 已被 MJLab 和 Unitree RL Lab 采纳为默认 motion tracking 方法
- 开源后社区广泛使用，成为 humanoid RL 的新 baseline
- 展示了 diffusion model 在 robotics control 中的独特优势 (test-time guidance)，这条路线与 Diffusion Policy (manipulation) 形成呼应
- 局限: diffusion 的 prediction horizon 只有 0.64s，不够做 long-horizon planning; history conditioning 可能导致重复 pattern

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **Sim2real 的核心不是 domain randomization，而是执行器建模和延迟控制** -- BeyondMimic 证明 moderate DR + careful system implementation 优于 heavy DR | 灵巧手也应该优先把执行器模型和通信延迟搞对，而不是靠大范围随机化 |
| 2 | **Diffusion 的独特价值 = classifier guidance** -- VAE/AMP 做不到 test-time optimization，diffusion 可以 | 如果未来灵巧手也用 generative policy，diffusion 的 guidance 能力是选择它而非 VAE 的核心理由 |
| 3 | **State-action co-diffusion > pure action diffusion** -- 同时预测 state 才能对 future state 做 planning | World model 的思路: 知道 action 会导致什么 state，才能做 predictive control |
| 4 | **"一个 recipe 训所有动作" 是 scalability 的关键** -- 如果每个 skill 要调参，就不可能 scale 到数百个 | 灵巧手的 multi-task RL 也需要统一的 reward formulation |
| 5 | **Hierarchical 路线 (tracker + planner) 的替代方案: generative model + guidance** -- 避免了 planner-controller mismatch | 这是 FM 思维: 不拆成 hierarchy，而是用一个大模型 + conditioning 来统一 |
