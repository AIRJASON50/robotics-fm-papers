# Diffusion 家族发展脉络 -- 从热力学到机器人动作生成

## 演化路线

```
2015 Sohl-Dickstein et al.         从热力学借来"可逆破坏"的直觉
  |                                 证明了: 如果破坏过程已知, 重建过程可学
  |                                 但: 实现复杂, 效果一般, 没人用
  |
2019 Song & Ermon (NCSN)           从另一个角度到达同一个地方
  |                                 "score matching" = 学数据分布的梯度方向
  |                                 关键: 加多级噪声, 每级分别学 score
  |                                 效果第一次接近 GAN, 但推理慢
  |
2020 Ho et al. (DDPM)              统一两条线, 极致简化
  |                                 loss = ||predicted_noise - actual_noise||^2
  |                                 就是一个回归问题, 任何人都能训
  |                                 效果超越 GAN, 引爆 diffusion 时代
  |
  +→ 2020 Song et al. (DDIM)       加速推理: 1000 步 → 50 步
  |                                 不改训练, 只改采样公式 (确定性 ODE)
  |                                 Diffusion Policy 实际推理用的是 DDIM
  |
  +→ 2022 Ho & Salimans (CFG)      条件生成: 引导模型"生成特定内容"
  |                                 Classifier-Free Guidance
  |                                 Diffusion Policy 的"根据观测生成动作"用的就是这个
  |
  +→ 2022 Lipman et al. (Flow Matching)  进一步简化: ODE 直线路径
  |                                       训练: 学速度场, 不学噪声
  |                                       推理: 10 步 ODE 积分就够
  |                                       pi_0 的 action generation 核心
  |
  +→ 2023 Peebles & Xie (DiT)     替换骨干: U-Net → Transformer
                                    证明 Transformer 在 diffusion 中也 scale
                                    GR00T N1 的 action head 就是 DiT
```

## 每一步的核心思路迁移

### 1. 热力学扩散 (2015, Sohl-Dickstein)

**idea 来源**: 非平衡统计热力学

```
物理现象: 一滴墨水滴入水中
  t=0: 有结构 (一个点)
  t→∞: 完全扩散 (均匀分布)
  → 从有序到无序的过程是已知的 (高斯扩散)

逆过程: 如果能把均匀分布的墨水"收回来", 就能生成有结构的数据
  → 学习逆扩散过程 = 学习生成模型
```

**贡献**: 提出了框架, 但实现复杂 (需要算完整的后验分布), 生成质量差, 5 年没人跟进。

### 2. Score Matching (2019, Song & Ermon)

**idea 来源**: 概率论中的 score function

```
Score function: ∇_x log p(x) = 数据分布的梯度方向
  在数据密度高的地方, score 指向那里
  从噪声出发, 沿 score 方向走, 就能走到数据密度高的地方
  → Langevin dynamics: x_{t+1} = x_t + step * score(x_t) + noise

问题: 在低密度区域 score 估计不准 (没有数据, 怎么算梯度?)
解决: 给数据加不同级别的噪声 (σ1 > σ2 > ... > σN)
  → 大噪声: 填满整个空间, score 到处都有信号
  → 小噪声: 精确还原数据细节
  → 从大噪声到小噪声逐步去噪 = 粗到细的生成
```

**贡献**: 效果第一次接近 GAN, 建立了"多级噪声"的直觉。但推理需要数千步 Langevin dynamics, 很慢。

### 3. DDPM (2020, Ho et al.) -- 统一和简化

**核心突破**: 把 2015 和 2019 两条线用一个极简框架统一。

```
2015 的框架 (复杂) + 2019 的多级噪声 (有效) → DDPM (简单且有效)

关键简化:
  2015: 需要算完整后验 q(x_{t-1}|x_t, x_0) → 推导复杂
  DDPM: 直接预测噪声 ε, loss = ||ε_predicted - ε_actual||^2 → 一个 MSE

  为什么预测噪声就够了:
    x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε     (前向: 已知)
    知道了 ε → 可以算出 x_0 → 可以算出 x_{t-1}  (反向: 一步推导)
```

**为什么叫 L_simple**: 原始变分下界 (VLB) loss 包含复杂的权重系数, DDPM 发现去掉权重 (均匀加权) 反而效果更好。简化后就是纯 MSE, 极其简单。

### 4. DDIM (2020, Song et al.) -- 加速推理

```
DDPM 的问题: 推理需要 1000 步去噪, 太慢
  生成一张图: ~20 秒

DDIM 的 insight: 去噪过程可以是确定性的 (不加随机噪声)
  → 变成一个 ODE (常微分方程) 而不是 SDE (随机微分方程)
  → ODE 可以用大步长积分: 1000 步 → 50 步, 质量几乎不变

  不需要重新训练! 用同一个 DDPM 训好的模型, 只改采样公式
```

**对机器人的意义**: Diffusion Policy 实际推理时用 DDIM (不是 DDPM), 把去噪步数从 100+ 降到 ~20, 才能满足控制频率。

### 5. Classifier-Free Guidance (2022, Ho & Salimans) -- 条件生成

```
问题: 怎么让 diffusion 生成"特定内容"?
  无条件: 随机生成任意图片
  有条件: 生成"一只猫" / 根据当前观测生成动作

CFG 的做法:
  同时训两个模型 (实际上是同一个模型, 随机 dropout 条件):
    有条件的: ε_θ(x_t, t, c)     (c = 条件, 如"一只猫"或"当前图像观测")
    无条件的: ε_θ(x_t, t, ∅)     (∅ = 空条件)
  
  推理时: ε_guided = ε_无条件 + w * (ε_有条件 - ε_无条件)
  w > 1: 增强条件的影响 (更像"猫" / 更贴合当前观测)
```

**对机器人的意义**: Diffusion Policy 的"根据当前图像生成动作"就是条件生成, CFG 是实现这个的标准方法。

### 6. Flow Matching (2022, Lipman et al.) -- 进一步简化

```
DDPM: 噪声到数据的路径是弯曲的 (需要很多步)
Flow Matching: 路径是直线 (需要很少步)

  DDPM 路径:     噪声 ~~~曲线~~~ → 数据    (1000 步)
  Flow Matching: 噪声 ————直线——→ 数据    (10 步)

  训练: 学速度场 v(x_t, t), 不学噪声 ε
  loss = ||v_predicted - (x_1 - x_0)||^2   (目标速度 = 终点 - 起点, 直线)
  推理: 10 步 Euler 积分: x_{t+dt} = x_t + v(x_t, t) * dt
```

**对机器人的意义**: pi_0 选择 flow matching 而非 DDPM, 因为 10 步 vs 100 步的速度差距对 50Hz 控制很关键。

### 7. DiT (2023, Peebles & Xie) -- 替换骨干网络

```
DDPM/DDIM/Flow Matching 都用 U-Net 做去噪网络
DiT: 用 Transformer 替换 U-Net

  U-Net: CNN 结构, encoder-decoder + skip connections
  DiT:   ViT 结构, patch tokens + self-attention

  为什么换: Transformer 的 scaling 特性比 U-Net 好
  → 模型越大效果越好, 和 LLM 一样的 scaling law
```

**对机器人的意义**: GR00T N1 的 action head 就是 DiT — 用 Transformer 做 action diffusion, 120Hz 输出动作。

## 真实思路提炼

每一步的 idea 迁移都遵循同一个模式:

```
1. 从其他领域借直觉
   2015: 热力学 → "可逆破坏"
   2019: 概率论 → "沿梯度方向走"
   2020: 信号处理 → "预测噪声然后减去"

2. 极致简化, 让任何人都能用
   2015 → DDPM: 复杂变分推导 → MSE 回归
   DDPM → DDIM: 1000 步 → 50 步 (不改训练)
   DDPM → Flow Matching: 弯曲路径 → 直线路径

3. 替换组件, 利用更好的 scaling
   U-Net → DiT (Transformer): 更好的 scaling law
   ImageNet → CLIP/SigLIP: 更好的预训练数据

4. 迁移到新领域
   图像生成 → Diffusion Policy (动作生成)
   Diffusion Policy → pi_0 (VLM + flow matching)
   DiT → GR00T N1 action head (120Hz 动作)
```

核心教训: **真正有影响力的工作往往不是最先提出 idea 的 (2015), 而是把 idea 简化到人人能用的 (DDPM 2020)**。这和 GPT 系列一样 — Transformer 2017 就有了, 但 GPT-1 2018 把它简化成 "pre-train + fine-tune" 才引爆了 LLM。
