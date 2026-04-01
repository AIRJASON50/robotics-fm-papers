# FPO++ (Flow Policy Optimization) -- 学习笔记
> 一句话: 用 conditional flow matching (CFM) loss 差值绕过 likelihood 计算，让 flow/diffusion policy 能直接用 PPO-style policy gradient 训练，首次实现 flow policy 的 humanoid sim2real
> 论文: Brent Yi, Hongsuk Choi et al. (Amazon FAR + UC Berkeley + Stanford + CMU), 2026

## 这篇论文解决了什么问题
PPO 是 robotics RL 的主力算法，但它依赖 action likelihood ratio (rho = pi_new(a|o) / pi_old(a|o))。问题: Gaussian policy 的 likelihood 容易算，但 **flow/diffusion policy 的 likelihood 算不了** (需要 divergence integration，计算量炸)。之前的 workaround:
- DPPO/ReinFlow: 把 denoising 过程当 MDP，用 noise 的 likelihood 代替 action likelihood -- 但这不等价于 marginalized action likelihood
- NCDPO/GenPO: backprop through unrolled denoising steps -- 计算贵且有 vanishing gradient 风险

FPO 的思路: **完全绕过 likelihood**，用 CFM loss 差值近似 log-likelihood ratio。FPO++ 解决了原始 FPO 在复杂 robotics 任务上不稳定的问题。

## 核心想法 (用直觉解释)
Flow policy 本质是一个 velocity field: 从噪声 epsilon 沿 learned flow 走到 action a。CFM loss 衡量 "这个 velocity field 在某个 (tau, epsilon) 采样点上预测得多好"。FPO 的核心 insight:

**CFM loss 差值 ≈ log-likelihood ratio**

rho_FPO = exp(L_CFM(theta_old) - L_CFM(theta_new))

直觉: 如果新 policy 在某个 action 上的 CFM loss 比旧 policy 小，说明新 policy "更容易生成这个 action"，即 likelihood 提高了。这个近似 ratio 可以直接塞进 PPO 的 clipped objective。

## 关键设计决策
1. **Per-sample ratio (FPO -> FPO++)**: 原始 FPO 对每个 action 算一个 averaged ratio，clipping 是 all-or-nothing 的。FPO++ 对每个 Monte Carlo sample (tau_i, epsilon_i) 单独算 ratio 并单独 clip。提供更细粒度的 trust region，防止单个极端 sample 拉飞整个更新。
2. **Asymmetric trust region (ASPO)**: 对 positive advantage (好动作，要加强) 用 PPO clipping; 对 negative advantage (坏动作，要抑制) 用 SPO (Simple Policy Optimization) -- SPO 提供 "拉回" 梯度而不是 "截断" 梯度。直觉: 增强好动作用 clip 就行; 抑制坏动作要更温柔 (SPO)，避免 CFM loss 被推得太大导致 denoising posterior 崩溃。
3. **Zero-step sampling at test time**: 推理时不需要完整的 iterative denoising，直接用 network 输出 (zero step)，类似于 flow matching 的 straight-path 性质。速度快很多，实测性能反而更好。
4. **适用于 from-scratch training 和 fine-tuning**: 从头训练 locomotion/motion tracking; fine-tune pretrained Diffusion Policy (manipulation)。两种场景都 work。
5. **Flow policy 的 exploration 优势**: Gaussian policy 只能在均值附近探索，flow policy 可以表达 multimodal 分布。实验发现 flow policy 训出的 quadruped locomotion gait 更自然 (更对称、更少不必要的动作)。

## 这篇论文之后发生了什么
- 首次证明 flow policy + RL 可以 sim2real (Booster T1 locomotion, Unitree G1 motion tracking)
- 为 "先用 Diffusion Policy 做 imitation learning，再用 RL fine-tune" 的路线提供了可行方案
- 与 BeyondMimic 的 diffusion 用法不同: BeyondMimic 用 diffusion 做 motion planner (supervised)，FPO++ 用 flow 做 RL policy (online training)
- 局限: CFM loss 差值只是 likelihood ratio 的近似，不是精确值; 仍需要调 hyperparameter (clip param, MC samples)

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **Flow matching 是统一 imitation learning 和 RL 的桥梁** -- 同一个 flow policy 可以先 supervised pre-train，再 RL fine-tune | 灵巧手: Diffusion Policy pre-train + RL fine-tune 是可行路线，FPO++ 提供了 fine-tuning 工具 |
| 2 | **绕过 likelihood 而非计算 likelihood 是更聪明的方案** -- FPO 用 loss 差值做 proxy，避免了 integration | 设计 RL for generative policy 时，不要执着于精确 likelihood，找好的 surrogate |
| 3 | **Multimodal action distribution 对 exploration 有实质帮助** -- 不只是理论上更 expressive，实际训出来动作更自然 | Gaussian policy 的 unimodal 限制可能是灵巧手 RL 中 "次优动作" 的原因之一 |
| 4 | **Asymmetric trust region 对应 RL 的 "小心抑制坏动作"** -- 鼓励好动作和抑制坏动作应该用不同强度 | PPO 改进方向: positive 和 negative advantage 的 clipping 策略应该不同 |
| 5 | **Pre-train + RL fine-tune 是 robotics FM 的核心范式** -- FPO++ 就是这个范式在 policy gradient 层面的实现 | 这条路线: Internet-scale data -> pre-train generative policy -> RL fine-tune for specific task/robot |
