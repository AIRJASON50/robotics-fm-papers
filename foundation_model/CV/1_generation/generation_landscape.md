# 生成模型全景: VAE → GAN → Diffusion → Flow Matching

## 三大生成范式

```
2013 VAE          2014 GAN          2015→2020 Diffusion
 (变分自编码器)     (生成对抗网络)      (去噪扩散模型)
  |                 |                  |
  编码+解码         对抗训练            逐步去噪
  有理论但模糊      效果好但不稳定      理论+效果+稳定
```

### 训练目的与产物

| 模型 | 训练目的 | 训练产物 | 部署时保留什么 |
|------|---------|---------|-------------|
| AE | 压缩还原, 学好的表征 | encoder + decoder | encoder (做降维/特征提取) |
| VAE | 学概率分布 P(x), 生成新数据 | encoder + decoder | decoder (做生成, encoder 推理时不用) |
| GAN | 隐式学数据分布, 生成新数据 | generator + discriminator | generator (做生成) |
| DDPM | 学去噪, 从噪声生成数据 | 一个去噪网络 | 去噪网络 (迭代调用做生成) |
| Flow Matching | 学速度场, 从噪声生成数据 | 一个速度预测网络 | 速度网络 (ODE 积分做生成) |

训练目的的演进: AE 是"理解数据" → VAE/GAN/DDPM 转向"生成数据"。encoder 从目的本身 (AE) 变成了训练的辅助工具 (VAE), 再到完全不需要 (DDPM)。

在 robotics 中, 表征和生成被重新组合:
- 表征部分: CLIP/SigLIP (AE/对比学习的后代) 提供 vision encoder
- 生成部分: Diffusion/Flow Matching (DDPM 的后代) 生成动作
- pi_0 = SigLIP (表征) + Flow Matching (生成) 的组合

### 两种生成范式的几何本质

```
范式 A: 跨空间映射 (AE / VAE)
  data space (150528维) ←→ latent space (50维)
  encoder: 数据空间 → 流形的低维坐标 (降维)
  decoder: 低维坐标 → 数据空间 (升维)
  本质: 找到数据流形的坐标系, 用坐标描述数据
  类比: 地球表面 (3D 中的 2D 流形) → 用经纬度 (坐标系) 描述
  优势: 有 latent space → 可做插值/编辑/理解
  劣势: 跨维度映射跨度大 → 一步到位学不好 → 模糊

范式 B: 同空间映射 (DDPM / Flow Matching)
  data space (150528维) → data space (150528维)
  起点: 纯噪声 (空间中任意一点)
  终点: 数据 (流形上一点)
  本质: 不找坐标, 学一个"引力场"把任意点推到流形上
  类比: 不知道经纬度, 但引力场能把太空中任意位置拉回地球表面
  优势: 不降维, 无信息丢失; 每步只移一点, 简单稳定
  劣势: 没有 latent space, 不能直接做插值; 需要多步, 推理慢
```

Robotics 把两者组合: CLIP/ViT 做跨空间映射 (表征), Diffusion/Flow Matching 做同空间映射 (生成动作)。

### 核心思想对比

| 范式 | 怎么生成 | 训练信号 | 优点 | 缺点 |
|------|---------|---------|------|------|
| VAE (Variational AutoEncoder, 变分自编码器) | 数据→压缩到 latent→从 latent 重建 | 重建 loss + KL 散度 | 有理论框架, 训练稳定, 有 latent space | 生成模糊 (高斯假设太强) |
| GAN (Generative Adversarial Network, 生成对抗网络) | Generator 从噪声生成, Discriminator 判真假 | 对抗 loss (骗/识别) | 生成锐利清晰 | 训练不稳定, mode collapse, 无理论保证 |
| Diffusion (DDPM) | 加噪→学去噪→从纯噪声迭代去噪 | MSE (预测噪声) | 理论完备, 训练稳定, 效果最好 | 推理慢 (需要多步迭代) |
| Flow Matching | 学噪声→数据的直线速度场, ODE 积分 | MSE (预测速度) | 比 diffusion 更快 (直线 vs 弯曲) | 较新, 理论仍在发展 |

---

## 演化脉络

```
=== 重要: 降维/表征 和 生成 是两条独立发展的线 ===

线 A (降维/表征): AE (1986) → Hinton Science (2006) → Bengio (2012)
  目标始终是好的压缩/特征, 聚类是副产品, 没有人从中推出"生成"

线 B (概率生成):  玻尔兹曼机 (1985) → Hinton DBN (2006) → VAE (2013) → GAN (2014) → DDPM (2020)
  概率生成模型一直在尝试, 只是之前训不好

Hinton 2006 的两篇论文:
  DBN 论文 (Neural Computation): 目标是生成模型 (线 B)
    → RBM 天生是概率生成模型, 可以采样
    → 原话: "we simply generate an image from its high-level representations"
  Science 论文: 目标是降维 (线 A)
    → 用 RBM 权重初始化 AE, 做比 PCA 更好的降维
    → 聚类可视化是降维结果, 不是生成的前奏

=== 第一代生成模型: 概率框架 (2006-2014) ===

Hinton DBN (2006):
  RBM 逐层预训练 → 深层概率生成模型第一次能训
  能生成但质量差, 框架复杂 (吉布斯采样, 对比散度)

VAE (2013, Kingma & Welling):
  从概率推断出发 (不是从 AE 改进): "怎么用 NN 训概率生成模型?"
  结构长得像 AE (encoder-decoder), 但出发点是变分推断
  数据 → encoder → latent z ~ N(μ, σ) → decoder → 重建数据
  创新: 重参数化技巧让 latent space 可微 → 比 RBM 简洁得多
  问题: 假设 latent 是高斯分布 → 生成结果模糊
  遗产: ACT 用 CVAE 做 action chunking
        DreamerV3 用 VAE 做 world model 的 latent space
        DiT 用 Stable Diffusion 的 VAE 做 latent 压缩

GAN (2014, Goodfellow):
  完全放弃概率建模, 用对抗训练隐式学习数据分布
  Generator: 噪声 z → 假数据
  Discriminator: 判断输入是真数据还是假数据
  创新: 不需要显式建模数据分布
  问题: mode collapse, 训练不稳定, 无法评估生成质量

=== VAE 周边: 填满 latent space 的其他尝试 ===

β-VAE (2017): 加大 KL 权重 → 各维度更独立 → 解耦表征 (z[0]=旋转, z[1]=大小)
VQ-VAE (2017): 不用连续高斯, 用离散 codebook → K 个码本向量, 没有空洞

**VQ-VAE 与 robotics 的连接**:
  VQ-VAE 的离散化思想天然适合有界/有结构的数据:
  - 动作空间 (关节角) 天然有界 [-pi, pi], 天然有物理约束
  - 离散成 256 bins → 每个 bin 对应可行区域 → 没有无意义空洞
  - RT-2/OpenVLA 的动作 token 就是 VQ-VAE 的简化版 (均匀切分, 不学 codebook)

  两条路线的底层分歧:
  - 有界 + 低维 + 有物理约束 → 离散化自然 (RT-2, OpenVLA)
  - 无界 + 高维 + 分布复杂 → 连续生成更合适 (Diffusion Policy, pi_0)

=== 第二代: 迭代去噪 (2015-2020) ===

Diffusion Thermodynamics (2015, Sohl-Dickstein):
  从热力学借来"可逆扩散"的直觉
  理论正确但实现复杂, 效果不如 GAN → 5 年无人跟进

Score Matching (2019, Song & Ermon):
  学数据分布的梯度方向, 沿梯度走到高密度区域
  多级噪声: 大噪声给全局方向, 小噪声给细节
  效果追上 GAN, 但推理需要数千步

DDPM (2020, Ho et al.):
  统一 2015 和 2019 两条线
  loss = MSE(predicted_noise, actual_noise) → 极致简化
  效果超越 GAN, 训练稳定, 不会 mode collapse
  → 引爆 diffusion 时代, GAN 在图像生成领域被取代

=== 第三代: 加速+替换骨干 (2020-2023) ===

DDIM (2020, Song et al.): 1000 步 → 50 步 (不改训练, 只改采样)
Flow Matching (2022, Lipman): 弯曲路径 → 直线, 10 步就够
DiT (2023, Peebles): U-Net → Transformer, 更好的 scaling
```

---

## 在机器人领域的影响

### 直接用于动作生成

| 方法 | 机器人应用 | 怎么用的 |
|------|---------|---------|
| DDPM/DDIM | Diffusion Policy (2023) | 从噪声去噪生成 action chunk, 条件是当前观测 |
| DDPM | Octo (2024) | Diffusion action head, 20 步去噪 |
| Flow Matching | pi_0 (2024) | 10 步 ODE 积分生成动作, 比 DDPM 更快 |
| Flow Matching + DiT | GR00T N1 (2025) | DiT 架构做 flow matching, 120Hz action |

### 间接影响 (通过 GAN 的判别器思想)

| 方法 | 机器人应用 | 怎么用的 |
|------|---------|---------|
| GAN 判别器 → AMP (Adversarial Motion Priors) | PHC, SONIC, humanoid motion tracking | Discriminator 判断"动作像不像人类" → 作为 RL 的 reward |
| GAN 判别器 → GAIL (Generative Adversarial Imitation Learning) | 模仿学习 | Discriminator 区分 expert 和 policy 的轨迹 → reward |

**GAN 在机器人中的遗产**: 生成图片的能力被 diffusion 替代, 但**判别器作为 reward signal** 的思想通过 AMP/GAIL 在运动控制中仍然活跃。

### VAE 的持续使用

| 方法 | 机器人应用 | 怎么用的 |
|------|---------|---------|
| CVAE (Conditional VAE) | ACT (2023) | 编码动作多模态性, 一个观测对应多种合理动作 |
| VAE latent space | DreamerV3 (2023) | World model 在 latent space 做 imagination |
| VAE encoder | DiT (2023) | 用 Stable Diffusion 的 VAE 把图像压到 latent, DiT 在 latent 上做 diffusion |

---

## 对机器人工程师的 Takeaway

```
必须理解:
  Diffusion (DDPM → Flow Matching) — 当前 VLA 动作生成的主流方法
  VAE 的 latent space 概念 — ACT, DreamerV3, DiT 都用

了解即可:
  GAN — 已被 diffusion 替代, 但 AMP/GAIL 借鉴了判别器思想
  DDIM — Diffusion Policy 实际推理用的采样方法
  CFG — 条件生成 (根据观测生成动作) 的标准方法

不需要深入:
  GAN 的各种变体 (WGAN, StyleGAN, ProgressiveGAN...)
  VAE 的变体 (β-VAE, VQ-VAE...)
  → 除非你做图像生成, 否则不需要
```
