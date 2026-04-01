# Generative Adversarial Networks (Goodfellow et al., 2014) -- Takeaway Notes

> 一句话: 提出用 generator 与 discriminator 对抗训练来隐式学习数据分布, 开创了对抗生成范式.

## 核心贡献
- 提出 minimax 博弈框架: G 生成假样本骗 D, D 区分真假, 两者交替优化
- 证明在足够容量下, G 的最优解就是真实数据分布 (Nash equilibrium)
- 无需显式定义似然函数, 绕过了传统生成模型 (VAE, PixelRNN) 的 intractable inference 问题

## 为什么重要
GAN 是第一个成功的隐式生成模型, 证明"让两个网络互相博弈"就能产生高质量样本.
这个思路后来演化为 WGAN, StyleGAN 等, 也深刻影响了 RLHF 中 reward model 的
对抗性训练思想. 虽然 diffusion model 在图像生成上已取代 GAN, 但对抗训练的思想仍然活跃.

## 对你 (RL->FM) 的 Takeaway
- sim2real 中的 domain randomization 与 GAN-based domain adaptation (如 SimGAN) 直接相关:
  用 GAN 将 sim 图像转为 real 风格, 缩小 visual domain gap.
- GAIL (Generative Adversarial Imitation Learning) 将 GAN 框架搬到 RL: discriminator
  区分 expert/agent trajectory, 替代手工 reward -- 这是从 demonstration 学习的关键方法.

## 与知识库其他内容的关联
- 17_PPO: GAIL 通常用 PPO/TRPO 作为 policy optimizer
- 18_SAC: 对比 -- SAC 用 entropy bonus 鼓励探索, GAN 用对抗损失鼓励多样性
- Diffusion Models (如有): GAN 的后继/替代方案
