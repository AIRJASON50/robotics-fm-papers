# Generative Adversarial Networks -- 学习笔记
> 一句话: 用 Generator 和 Discriminator 的 minimax 博弈隐式学习数据分布, 无需显式似然函数, 开创对抗生成范式.
> 论文: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (UdeM), 2014, NeurIPS 2014

## 这篇论文解决了什么问题
传统深度生成模型 (RBM, DBN, VAE) 要么需要 intractable 的 partition function, 要么需要近似推断 (variational inference, MCMC), 训练困难且生成质量有限. 本文提出一个全新思路: 不显式建模 p(x), 而是训练一个 Generator 直接从噪声映射到数据空间, 用另一个 Discriminator 的对抗信号来引导生成.

## 核心想法 (用直觉解释)
Generator 是造假者, 从随机噪声 z 生成假样本 G(z); Discriminator 是鉴别者, 判断输入是真数据还是 G 的输出. 两者交替训练: D 学会区分真假, G 学会骗过 D. 达到均衡时 G 生成的分布等于真实数据分布, D 输出恒为 1/2 (无法区分). 数学上是 minimax: min_G max_D [E_x[log D(x)] + E_z[log(1-D(G(z)))]]. 论文证明在无限容量假设下全局最优解 p_g = p_data, 且 JSD (Jensen-Shannon Divergence) 为 0.

## 关键设计决策
- **Minimax 目标函数**: 将生成建模转化为博弈论问题. 实际训练中 G 不最小化 log(1-D(G(z))) 而是最大化 log(D(G(z))), 因为前者在训练初期梯度太小 (D 轻松拒绝低质量样本)
- **交替训练, k:1 比例**: 先训练 D k 步, 再训练 G 1 步. 保持 D 接近最优, 使 G 得到有效梯度. 论文用 k=1 (最便宜选项)
- **纯 backprop, 无 MCMC**: G 和 D 都是 MLP, 只需前向传播和反向传播, 无需 Markov chain sampling 或近似推断. 这是相比 RBM/DBN 的核心优势
- **理论保证**: Proposition 1 给出最优 D 的闭式解, Theorem 1 证明全局最优点, Proposition 2 在一定条件下证明收敛. 但实际中有限容量网络不保证收敛 (mode collapse)

## 这篇论文之后发生了什么
DCGAN (2015) 引入 CNN 架构提升图像质量. WGAN (2017) 用 Wasserstein 距离替代 JSD 解决训练不稳定. StyleGAN (2019) 实现高分辨率人脸生成. Conditional GAN, CycleGAN 扩展到有条件生成和无配对风格迁移. 但 2020 后 Diffusion Model 在图像生成上全面超越 GAN (更稳定, 更多样). GAN 的对抗思想存活于 GAIL (模仿学习), RLHF (reward model 对抗), adversarial domain adaptation.

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 对抗训练思想延续到 imitation learning | GAIL 用 discriminator 区分 expert/agent trajectory 替代手工 reward, 通常配合 PPO 训练 policy |
| 2 | GAN-based domain adaptation 缩小 sim2real gap | SimGAN 用 adversarial loss 将 sim 图像转为 real 风格, 是 domain randomization 的替代方案 |
| 3 | 隐式建模 vs 显式建模的 tradeoff | GAN 无需似然函数但训练不稳定; VAE/Diffusion 有显式目标但需要更多计算. Diffusion Policy 选择了后者, 换取训练稳定性 |
| 4 | Mode collapse 问题至今未彻底解决 | 在 robot skill learning 中, 多模态行为 (同一 state 多种有效 action) 需要注意生成多样性, Diffusion/Flow Matching 比 GAN 更擅长此事 |
