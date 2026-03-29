# AINA: Dexterity from Smart Lenses - 论文笔记

**论文**: Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations
**作者**: Irmak Guzey (NYU, 一作), Haozhi Qi, + Meta 团队
**发表**: arXiv:2511.16661, 2025
**项目**: https://aina-robot.github.io

---

## 一句话总结

用 Aria 智能眼镜在日常环境中采集人类操作视频 (~50 条/任务) + 机器人场景中 1 条演示做域对齐，训练 3D 等变 Transformer 策略预测指尖轨迹，在 Psyonic 灵巧手上实现 9 个日常任务的多指操控。

---

## 核心问题

如何从人类日常视频 (in-the-wild) 学习多指灵巧手策略？核心矛盾：
- In-domain 数据精确但不可规模化
- Web 视频可规模化但缺乏可靠 3D 信号
- **Aria 智能眼镜**：日常佩戴 + 高质量 3D 手部姿态估计 + 立体深度 → 平衡点

---

## 方法

### 数据采集
- **In-the-wild**: ~50 条/任务，Aria Gen2 眼镜，10Hz，自带 21 关键点手部估计
- **In-scene**: 仅 1 条，机器人环境 RGB-D，Hamer 估计手部姿态

### 数据处理
- **物体跟踪**: GroundedSAM (首帧分割) → CoTracker (2D 跟踪) → FoundationStereo/RealSense (深度反投影)
- **域对齐**: 首帧物体质心平移 + Kabsch 算法提取**仅 z 轴旋转** (Aria 的 z 轴已通过 IMU 与重力对齐)

### 策略网络: VN Encoder → GPT Transformer → Deterministic Head
- **VN Encoder (VNMLP)**: SO(3) 等变 Vector Neuron 层，保证 3D 旋转不变性
- **GPT Transformer**: 非因果 encoder-only，4 层 2 头，512 维。500 物体点 + 21 手部关键点共享 attention
- **输出**: 5 指尖 x 30 步未来轨迹 (90 维)
- **Position encoding 仅用于指尖** (区分手指身份)，物体点无位置编码 (permutation invariant)

### 部署
- Kinova Gen3 + Psyonic Ability Hand (6-DOF)
- 自定义全臂-手 IK；抓取启发式：拇指与任意手指 < 5cm 时靠拢

---

## 关键结果

| 设置 | 成功率 |
|------|--------|
| AINA (完整) | 多数任务 70-100% |
| In-Scene Only | 空间泛化差 |
| In-The-Wild Only | 动作严重偏移 |
| vs. Masked BAKU (图像) | 80% vs 47% (Toy Picking) |

- 高度泛化：3 种高度 (3.5cm 增量)，仅需重采 1 条 in-scene (~1 min)
- 物体泛化：相似形状 zero-shot 良好；形状差异大时困难

---

## 作者展望

1. 集成 EMG 传感器获取力反馈
2. 实现 Aria 实时深度，统一训练/部署观测源
3. 更鲁棒的 3D 物体跟踪 (Stereo4D/FoundationPose)

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| 训练 epoch | 2000 | `train_epochs: 1001` |
| 手部噪声 | [-2cm, 2cm] | `gaussian_limit=0.015` (1.5cm) |
| 缩放增强 | [0.8, 1.2] | `scale: False` (默认关闭) |
| Loss reduction | MSE (mean) | `reduction: "sum"` |
| Transformer 输入 | 5 指尖 | 21 手部关键点 (block_size=521) |
| z 轴平移 | [-30cm, 30cm] | z 轴 clamp 到 [-5cm, 5cm] |
| 物体预测 | 仅指尖 | 预测全部 521 点但只监督指尖 (mask-based selective supervision) |

### 值得学习的代码设计

1. **VNLeakyReLU 的 SO(3) 等变实现**: 通过学习方向向量 + 内积分解实现旋转等变激活函数
2. **Mask-based selective supervision**: 网络预测所有点但 loss 仅作用于指尖 → 表征层面学物体运动但不被惩罚
3. **Position encoding 只用于指尖**: 物体点 permutation invariant，指尖需要身份区分
4. **缓存式增量处理**: 每步检查 .npy 是否存在，避免重复计算

---

## 非显而易见的洞察

1. **图像策略的"历史诅咒"**: 带历史的图像策略 (7%) 比单帧 (47%) 差 → 头部运动导致视角历史在部署时 OOD
2. **重力对齐是隐含的域桥接**: Aria IMU 保证 z 轴重力对齐 → 所有 in-the-wild 演示 z 方向一致 → 只需解决水平旋转
3. **In-scene 演示的双重作用**: 既做域对齐又做训练数据，去掉任何一个功能都显著降低性能
