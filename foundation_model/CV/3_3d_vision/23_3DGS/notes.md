# 3D Gaussian Splatting for Real-Time Radiance Field Rendering -- 学习笔记 (选读)
> 一句话: 用数百万个可学习的 3D 高斯原语 (均值/协方差/颜色/不透明度) 显式表示场景, 通过可微 splatting 渲染实现 NeRF 级质量 + 实时 (>100 FPS) 速度.
> 论文: Bernhard Kerbl et al. (INRIA), SIGGRAPH 2023; 本文件实为 Keselman & Hebert (CMU, 2023) 的扩展工作
> 注: 目录中的 PDF 实际是 "Flexible Techniques for Differentiable Rendering with 3D Gaussians", 以下结合原始 3DGS 论文和该扩展工作总结.

## 核心想法
3DGS 用显式的 3D 高斯基元 (primitive) 替代 NeRF 的隐式 MLP: 每个高斯由均值 mu (位置), 协方差 Sigma (形状/方向), 球谐系数 (view-dependent 颜色), 不透明度 alpha 参数化. 从 SfM 点云初始化, 通过可微 splatting (将 3D 高斯投影到 2D 屏幕空间后 alpha compositing) 渲染图像, 用光度 loss 反向传播优化所有高斯参数. 训练中还包括自适应密度控制: 在梯度大的区域 split/clone 高斯, 在不透明度低的区域剪枝. 扩展工作 (Keselman) 进一步证明: 3D 高斯可以导出 mesh (Poisson reconstruction), 计算 optical flow, 渲染法线, 且不同的 3DGS 实现 (splatting vs raytracing) 是互操作的.

## 与主线论文的关系
- **NeRF 的实时化替代**: 解决了 NeRF 渲染慢 (per-pixel MLP query) 的核心瓶颈, 用显式原语 + rasterization 实现 100+ FPS
- **与 Depth Anything 互补**: 3DGS 从多视角重建 3D, Depth Anything 从单张图估计深度 -- 两者分别解决 multi-view 和 mono 的 3D 感知
- **可微渲染范式的延续**: 继承了 NeRF "可微渲染 + 梯度优化" 的核心思路, 但换了更高效的场景表示

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 显式表示 (高斯原语) vs 隐式表示 (MLP) 的 trade-off: 显式更快更可编辑, 隐式更紧凑 | 机器人 world model 面临相同选择: latent space (隐式) vs point cloud/mesh (显式) |
| 2 | 不需要预训练, 直接从 SfM 点云 + 梯度优化, 无 dataset bias | 机器人操作新物体时, 可以用 3DGS 在线重建 -- 不需要物体数据库 |
| 3 | 实时渲染 (>100 FPS) 使 3D 重建可以嵌入 real-time control loop | 机器人可以用 3DGS 做实时场景感知: 重建 -> 规划 -> 执行, 全在线完成 |
| 4 | Gaussian 表示天然支持组合/分割/追踪 (每个物体用一组高斯) | 机器人操作需要物体级分割, 3DGS 的显式原语比 NeRF 更容易实现物体级操作 |
