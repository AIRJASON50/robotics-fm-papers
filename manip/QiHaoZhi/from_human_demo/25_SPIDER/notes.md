# SPIDER: Scalable Physics-Informed Dexterous Retargeting - 论文笔记

**论文**: SPIDER: Scalable Physics-Informed Dexterous Retargeting
**作者**: Chaoyi Pan (CMU), Changhao Wang, Haozhi Qi, + Meta FAIR 团队
**机构**: Meta FAIR + CMU
**发表**: arXiv:2511.09484, 2025
**项目**: https://jc-bao.github.io/spider-project/

---

## 一句话总结

无需训练任何网络，用采样优化 (CEM 变体) 在物理仿真中将运动学参考轨迹转化为动力学可行的控制序列。核心创新是虚拟接触引导 (contact guidance) 通过 PD gain 退火扩大可行解 basin of attraction。规模化到 262 episodes / 2.4M frames / 5 种灵巧手 / 103 种物体。

---

## 核心问题

运动学 retargeting (IK) 不满足物理约束 (穿透、滑移)；RL 每条轨迹需独立训练太慢。需要一个**无训练、可扩展**的物理可行 retargeting 方案。

---

## 方法: 七步无训练管线

1. **数据统一**: 各数据集 → MANO 21 关键点 NPZ
2. **物体凸分解**: CoACD
3. **接触检测**: MuJoCo 回放人手 → 记录接触时刻/位置/持续时间
4. **场景 XML 生成**: 自动将 freejoint → 6 个独立关节 + position actuator
5. **IK 求解**: mocap body + equality constraint (让物理引擎做 IK)
6. **采样 MPC 优化** (核心)
7. **导出部署**

### 采样优化器 (核心算法)

- **不是神经网络**: Cross-Entropy Method 变体 + 退火采样
- 1024 并行 sample, MuJoCo Warp GPU 前滚
- **Top 10% elite selection**: 只对最好 ~102 个样本做 softmax 加权 (比标准 MPPI 更激进)
- **Knot-point 参数化**: 在稀疏时间点 (0.4s 间隔) 采样 → 线性插值 → 40x 降维
- **退火**: 噪声每轮缩小为 `beta_traj` 倍，最终为初始的 10%
- **Terminate-resample**: 偏差大的 sample 被好 sample 替换 (粒子重生)

### 虚拟接触引导 (核心创新)

**论文**: 虚拟约束力 $\eta_i \to 0$
**代码实际实现**: 物体 position actuator 的 PD gains 做 curriculum 退火
- 初始 Kp=2.0~10.0 → 每轮乘 0.5 → 最后一轮强制为 0
- 效果: 优化初期物体被"钉"在参考位置，后期完全靠真实接触力

### 鲁棒化

Min-max 域随机化: 对每个候选控制序列在多组物理参数上 rollout，取**最差情况 reward**。

---

## 关键结果

- 比标准采样成功率提升 **18%** (contact guidance)
- 比 RL 快 **10x**
- **规模**: 2.4M frames, 5 种灵巧手, 4 种人形, 103 种物体
- SPIDER 生成的参考轨迹做残差 RL → 训练更快且物体跟踪更好

---

## 作者展望

1. 将 SPIDER 轨迹用于 behavior cloning 训练泛化策略
2. 更好的 3D 重建提升质量

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| 接触引导 | "虚拟约束力" | **PD actuator gain 退火** (更准确是"虚拟 PD 控制器") |
| Sample 数 | 1024 | Config 默认 2048，yaml 默认 1024 |
| 迭代数 | 16 | yaml 默认 32 |
| 加权方式 | "全样本 softmax" | **仅 top 10%** softmax |
| Gibbs 采样 | 未提及 | 双手交替优化 (噪声 mask 清零对方手维度) |
| 仿真频率 | 100Hz/50Hz | guidance 模式 200Hz/5Hz |
| Horizon | 1.2s | 默认 1.6s |

### 值得学习的代码设计

1. **函数式 Simulator-Optimizer 解耦**: 优化器接受 `step_env/save_state/load_state/get_reward` 函数指针 → 不同仿真器 (MJWP/Isaac/DexMachina) 复用同一优化器
2. **Warp stride trick**: `strides = (0,) + strides` 实现零拷贝 broadcast → 零额外显存
3. **CUDA Graph 捕获单步仿真**: `wp.ScopedCapture` → `wp.capture_launch` 消除 kernel launch overhead
4. **IK 用 equality constraint**: "让物理引擎做 IK" 而非 Jacobian 迭代
5. **自动 XML 生成**: DOM 操作将 freejoint → 6 独立关节 + actuator，contact guidance 无需手动编辑

---

## 非显而易见的洞察

1. **不是 RL 但像 RL**: 采样优化 = 无梯度策略搜索，区别仅在于直接优化控制序列而非策略参数
2. **Contact guidance 是 basin of attraction 扩大器**: PD gain 退火渐进松弛 → 物体从"被控制"过渡到"被接触力维持"
3. **Knot-point 参数化是关键降维**: 0.4s 间隔插值 → 40x 降维，使高维连续优化变得可行
4. **Terminate-resample = 粒子滤波重采样**: 不好的 sample 被好 sample "替换"(含仿真状态拷贝)
5. **Min-max 而非 average DR**: 取最差情况 reward 而非平均 → 更保守但更鲁棒
