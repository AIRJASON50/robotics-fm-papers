# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis -- 学习笔记 (选读)
> 一句话: 用 MLP 隐式表示场景的 5D 辐射场 (3D 位置 + 2D 视角 -> 颜色 + 密度), 通过可微体渲染从稀疏图像优化, 实现照片级新视角合成.
> 论文: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, et al. (UC Berkeley + Google), ECCV 2020 (Oral)

## 核心想法
NeRF 将场景表示为一个连续的 5D 函数 F(x,y,z,theta,phi) -> (RGB, sigma), 用简单的 MLP (8 层, 256 channels) 参数化. 渲染过程: 沿每条相机射线采样 3D 点, 查询 MLP 得到颜色和密度, 通过经典体渲染 (volume rendering) 积分得到像素颜色. 因为体渲染是可微的, 可以直接用光度 loss 从一组已知视角的图像反向传播优化 MLP. 两个关键技巧: (1) **Positional encoding** -- 将低维坐标映射到高维 sin/cos 频率空间, 使 MLP 能表示高频几何和纹理细节; (2) **Hierarchical sampling** -- 先用 "coarse" 网络定位大致表面, 再用 "fine" 网络在重要区域密集采样, 提高渲染效率.

## 与主线论文的关系
- **3DGS 的前身**: NeRF 开创了神经场景表示范式, 3DGS 在同一目标上用显式高斯原语替代隐式 MLP, 实现实时渲染
- **Positional encoding 的视觉版**: Transformer 的 positional encoding 编码序列位置, NeRF 的编码连续空间坐标 -- 同一思想
- **可微渲染 + 梯度优化**: 与 diffusion model 类似, NeRF 证明了"可微 pipeline + 梯度下降"可以解决极复杂的逆问题

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 隐式神经表示: 用网络权重编码整个场景, 不需要显式 3D 数据结构 | 机器人可以用类似方法学习场景的隐式表示, 用于 6DOF 抓取规划和碰撞检测 |
| 2 | Positional encoding 是让网络学习高频函数的通用技巧 | 机器人中的连续控制信号 (关节角/末端位置) 也可用 positional encoding 提升 policy 网络的表达力 |
| 3 | Per-scene optimization (每个场景单独训练一个 MLP) 是最大局限 -- 无法泛化 | 这正是后续 generalizable NeRF 和机器人 world model (如 UniSim) 要解决的问题 |
| 4 | 可微渲染使"2D 图像监督 -> 3D 理解"成为可能 | 机器人 sim2real 的核心也是可微仿真: 用真实观测梯度更新仿真中的策略 |
