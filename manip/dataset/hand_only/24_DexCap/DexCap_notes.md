# DexCap - 论文笔记

**论文**: DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation
**作者**: Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, C. Karen Liu
**机构**: Stanford University (The Movement Lab + Stanford Vision and Learning Lab)
**发表**: RSS 2024 / arXiv:2403.07788
**项目**: https://dex-cap.github.io/
**代码**: https://github.com/j96w/DexCap (MIT License)

---

## 1. Core Problem

灵巧操作的 imitation learning 面临两个核心瓶颈:

1. **数据采集系统不可扩展**: 现有灵巧操作数据采集主要依赖遥操作 (teleoperation), 需要真实机器人硬件, 操作速度慢 (约为人手自然速度的 1/3), 成本高且难以规模化。视觉追踪方案 (如 HaMeR) 在手-物遮挡场景下精度严重下降, 且只能提供 2D 估计, 缺乏准确的 3D 信息。现有 EMF (Electromagnetic Field, 电磁场) 手套虽然抗遮挡, 但无法追踪 6-DoF (Degrees of Freedom, 自由度) 腕部位姿, 也无法获取环境 3D 观测。

2. **人手数据到机器人策略的转换缺失**: 即使能采集人手 mocap (Motion Capture, 动作捕捉) 数据, 如何处理人手与机器人手之间的 embodiment gap (具身差异) -- 包括尺寸差异 (LEAP Hand 比人手大约 50%)、运动学结构差异 (LEAP 4 指 vs 人 5 指)、力反馈缺失 -- 并将其转化为可执行的低层控制策略, 缺乏成熟的算法框架。

**核心洞察**: 如果采集系统能同时提供 (a) 抗遮挡的精细指尖 3D 位置, (b) SLAM 基的 6-DoF 腕部位姿, (c) 与手部运动对齐的 3D 环境观测, 那么就可以用 fingertip IK (Inverse Kinematics, 逆运动学) + point cloud-based diffusion policy 直接从人手数据学习机器人策略, 完全绕过遥操作。

---

## 2. Method Overview

DexCap 包含两个子系统: 硬件系统 DexCap 和学习算法 DexIL (Dexterous Imitation Learning)。

### 2.1 DexCap 硬件

```
                    DexCap Hardware Architecture
                    ============================

         ┌───────────────────────────────────────────┐
         │             Chest Camera Vest              │
         │  ┌─────────┐    ┌──────────────────────┐  │
         │  │ L515    │    │ T265 (red, main SLAM)│  │
         │  │ RGB-D   │    │ defines world frame  │  │
         │  │ LiDAR   │    └──────────────────────┘  │
         │  └─────────┘                               │
         └───────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────┴────┐           │           ┌────┴────┐
    │ Left    │           │           │ Right   │
    │ Glove   │     ┌─────┴─────┐    │ Glove   │
    │ (Rokoko │     │ Backpack  │    │ (Rokoko │
    │  EMF)   │     │ NUC 13 Pro│    │  EMF)   │
    │ +T265   │     │ +40Ah     │    │ +T265   │
    │ (SLAM)  │     │ power bank│    │ (SLAM)  │
    └─────────┘     └───────────┘    └─────────┘
```

| 组件 | 型号/规格 | 功能 |
|------|-----------|------|
| 手指追踪 | Rokoko EMF gloves | 每指尖一个磁传感器, 相对手背 hub 的 3D 位移, 抗遮挡 |
| 腕部追踪 | 2x Intel RealSense T265 | 双目鱼眼 + IMU 的 SLAM, 6-DoF 腕部位姿, 自动修正漂移 |
| 环境观测 | Intel RealSense L515 LiDAR | 胸前安装, RGB-D 1280x720, 重建 3D 点云 |
| 世界坐标系 | 1x T265 (红色, 胸前) | 固定在 LiDAR 下方, 宽视角鱼眼增强 SLAM 鲁棒性, 初始位姿定义世界坐标系 |
| 计算/电源 | Intel NUC 13 Pro + 40000mAh 电源 | 放入背包, 总重 3.96 磅, 续航约 40 分钟 |
| 标定 | 3D 打印相机支架 (click-in 设计) | 所有相机先插入支架获取固定变换, 再取出安装到手套, 标定 <10 秒 |
| 成本 | <$4000 USD | 模块化设计, 不限品牌 |

### 2.2 DexIL 学习算法

三步框架:

**Step 1: Data Retargeting**
- **指尖 IK**: 基于 "fingertip 是最频繁接触区域" 这一先验 (来自 HO-3D, GRAB, ARCTIC), 用 PyBullet IK 将 LEAP Hand 16 维关节角求解到与人手指尖 3D 位置对齐。忽略小指 (LEAP 只有 4 指)。
- **腕部位姿**: 直接使用 T265 SLAM 追踪的 6-DoF 位姿, 并做 45 度安装角修正。
- **点云后处理**: RGB-D 转点云 -> 变换到世界坐标系 (消除人体运动影响) -> 裁剪到机器人工作空间 -> 去除桌面等无关点。

**Step 2: Point Cloud-based Diffusion Policy**
- 点云下采样到 K=10000 点, 拼接 RGB 颜色 -> R^{K x 6}。
- 用 FK (Forward Kinematics, 正运动学) 将机器人手 mesh 采样为点云 (80000 点/手), 合并到观测中, 弥合人手/机器人手视觉差异。
- Diffusion Policy (DDIM, 100 train / 10 inference steps) 作为 action decoder, 输出 d=20 步的动作轨迹。
- 动作空间: 2x 7-DoF 手臂 + 2x 16-DoF 灵巧手 = 46 维。
- Point cloud encoder: Perceiver (Set Transformer 架构, 优于 PointNet)。
- 数据增强: 点云和轨迹的随机 2D 平移。

**Step 3: Human-in-the-Loop Correction (可选)**
- **Residual mode**: 人手 delta 运动作为残差动作叠加到策略输出, 缩放因子 alpha (腕部) 和 beta (手指, 小值避免过大干扰)。
- **Teleoperation mode**: 踩脚踏板切换, 机器人完全跟踪人手运动。
- 纠正数据与原始数据等概率采样, 用于 fine-tune 策略 (类似 IWR, Intervention Weighted Regression)。

---

## 3. Key Designs

### 3.1 SLAM-based 6-DoF Wrist Tracking -- 填补 EMF 手套的关键缺失

**问题**: EMF 手套只能追踪相对于手背 hub 的指尖位移, 无法获取腕部在世界坐标系中的 6-DoF 位姿。IMU 方案随时间漂移严重。相机追踪需要固定第三视角, 不便携。

**设计**: 在每只手套背面安装 T265 SLAM 相机, 利用双目鱼眼 + IMU 融合实时建图, 实现:
- 便携性: 不需要外部相机, 手不需在第三视角可见。
- 抗漂移: SLAM 自动用已建地图纠正位置漂移, 适合长时间采集。
- 提供 6-DoF: 同时获取位置和朝向, 满足末端执行器策略学习需求。

**标定工程**: 所有 T265 (2 只手 + 1 只胸前) 在采集前先插入 3D 打印支架固定槽, 获取相机间恒定变换, 然后取出安装。Click-in 设计保证每次安装的一致性。整个标定 <10 秒。

**局限**: 2024-08 更新版本用 HTC VIVE Tracker 替换了 T265 (Intel 已停产 T265), 说明原始方案有硬件可用性问题。

### 3.2 Point Cloud 表征 + Robot Mesh 注入 -- 跨越 embodiment visual gap

**问题**: 直接使用 RGB-D 图像训练策略面临两大障碍: (1) 人手/机器人手的视觉外观差异巨大, 即使做 hand masking 也会丢失 in-hand manipulation 细节; (2) in-the-wild 采集时人体运动导致相机视角不断变化, 图像策略无法泛化。

**设计**:
- 点云变换到固定世界坐标系, 消除人体运动影响, 获得稳定观测。
- 关键创新: 用 FK 将 LEAP Hand URDF mesh 变换到当前关节角, 采样成点云 (80000 点/手), 合并到环境点云中。这样在训练时观测包含机器人手的点云, 在推理时也包含机器人手的点云, 实现一致的视觉输入。
- 点云下采样本身也模糊了人手手套和机器人手之间的外观差异。

**实验验证** (Table I):

| 方法 | 观测类型 | 平均成功率 |
|------|----------|-----------|
| BC-RNN-img | 原始图像 | 0% |
| DP-img | 原始图像 | 0% |
| DP-img-mask | 手部遮挡图像 | >30% |
| DP-point-raw | 原始点云 (无 robot mesh) | ~60% |
| DP-point | 点云 + robot mesh | >60% |
| DP-perc (default) | 点云 + Perceiver encoder | 最高 |

### 3.3 Fingertip IK Retargeting -- 以接触点为核心的简洁 retargeting

**问题**: 人手与 LEAP Hand 之间存在约 50% 的尺寸差异, 比例不同, 运动学结构不同 (5 指 vs 4 指)。直接映射关节角不可行。

**设计**: 基于 "指尖是最频繁接触区域" 这一先验:
- 只匹配 4 个指尖的 3D 位置 (忽略小指), 用 PyBullet null-space IK 求解 16 维关节角。
- 定义 null-space 参考姿态 `HAND_Q` 避免自碰撞。
- 补偿指尖 mesh 原点偏移 (`fingertip_offset`, `thumb_offset`) 和 LEAP Hand 根节点偏移 (`leap_center_offset`)。
- 腕部 6-DoF 直接使用 SLAM 追踪结果, 作为 IK 初始参考。

**局限**: 这种方法无法处理需要力反馈的任务 (如稳定夹持剪刀), 也无法保证手指深插入物体孔洞 (如剪刀柄)。对这些场景需要 human-in-the-loop correction 补充。

---

## 4. Experiments

### 4.1 任务设计

| 任务 | 难度 | 手数 | Demo 数 | 采集时间 | 特点 |
|------|------|------|---------|---------|------|
| Sponge picking | 基础 | 单手 | 251 | 30 min | 抓取+抬起 |
| Ball collecting | 基础 | 单手 | 179 | 30 min | 抓球+放入篮子 |
| Plate wiping | 中等 | 双手 | 102 | 30 min | 双手协调: 一手持盘一手擦 |
| Packaging | 困难 | 双手 | 104 | 60 min (in-the-wild) | 双手装箱+关盖, 测试泛化 |
| Scissor cutting | 极难 | 双手 | 96 | 60 min | 工具使用, 需力控 |
| Tea preparing | 极难 | 双手 | 55 | 60 min | 长序列: 开瓶->取茶->倒入壶 |

### 4.2 核心实验结论

**Q1 (数据质量)**: DexCap 的 EMF 追踪在重度遮挡场景下远优于视觉方法 HaMeR。数据采集速度约为遥操作的 3 倍。

**Q2 (无需机器人数据)**: 仅用 30 分钟 DexCap 数据, DP-perc 在基础任务上达到 72% 平均成功率。这是 "no on-robot data" 的灵巧操作 imitation learning 的重要验证。

**Q3 (模型架构)**: Diffusion Policy 比 BC-RNN 高约 25%; 点云输入比图像输入大幅领先; Perceiver encoder 优于 PointNet (尤其在多物体双手任务中提升约 20%)。

**Q4 (in-the-wild 数据)**: 纯 in-the-wild DexCap 数据训练的策略在 Packaging 任务上达到 47% 全任务成功率, 图像策略接近 0%。点云变换到世界坐标系是关键。已能泛化到未见物体 (40% 成功率)。

**Q5 (Human-in-the-loop)**: 30 次人工纠正 + fine-tune 后, 图像策略提升 33%, 点云策略提升 10%。纠正数据在固定相机下采集, 对图像策略增益更大。

**Q6 (极难任务)**: Scissor cutting 45% (拿剪刀) / 20% (剪纸); Tea preparing 65% (开瓶) / 25% (全任务)。证明框架能处理极具挑战性的任务。

### 4.3 关键消融发现

- 点云下采样本身能降低 human/robot 外观差异, 使得 DP-point-raw (无 robot mesh) 性能接近 DP-point。
- In-the-wild 数据的主要问题: (1) 无力信息导致双手协调困难; (2) 人体运动导致箱盖偶尔移出视野。
- Human correction 对未见物体泛化帮助有限 (纠正数据量太少)。

---

## 5. Related Work Analysis

DexCap 处于 "人手 mocap 数据 -> 机器人灵巧操作" 这条技术路线的关键节点:

| 技术路线 | 代表工作 | DexCap 的定位 |
|----------|----------|---------------|
| 遥操作采集 | DIME, Holo-Dex, HATO | DexCap 不需要真实机器人, 采集速度 3x, 但牺牲了力信息 |
| 人类视频学习 | DexMV, VideoDex, DexVIP | 这些只学高层先验/reward, 仍需 sim 或 teleop 数据; DexCap 直接学低层控制 |
| 便携采集器 | UMI, GELLO, Bunny-VisionPro | 这些面向 parallel-gripper; DexCap 面向多指手 |
| Hand mocap 系统 | Rokoko gloves, OptiTrack | 只有手指/不便携; DexCap 融合 SLAM 实现 6-DoF + 便携 |
| 从人类演示学习 | MimicPlay, R3M, VIP | 2D 图像空间, 缺 3D 信息; DexCap 提供完整 3D |

**DexCap 的独特价值**: 第一个同时满足 (1) 便携, (2) 精细指尖追踪, (3) 6-DoF 腕部, (4) 3D 环境观测的灵巧手数据采集系统, 且配套端到端的 retarget + policy learning pipeline。

---

## 6. Limitations & Future Directions

### 论文明确提出的局限

| 局限 | 影响 | 论文建议方向 |
|------|------|-------------|
| 续航 40 分钟 | 限制单次采集规模 | 提高电源效率 |
| 人手/机器手尺寸差异 | 某些任务 (如弹钢琴) 因机器手指粗无法执行 | 改进机器人手设计缩小差异 |
| 无力传感 | 双手协调时无法感知接触力, 装箱任务一只手难以稳定箱体 | 使用 conformal tactile textiles 采集触觉 |

### 从代码和系统角度补充的局限

| 局限 | 细节 |
|------|------|
| T265 已停产 | 2024-08 更新改用 HTC VIVE Tracker, 但引入了外部基站依赖, 牺牲了原始设计的便携性 |
| 手动坐标变换 | 代码中大量硬编码的旋转修正 (45 度安装角、Rokoko 坐标系切换), 可维护性差 |
| 数据格式碎片化 | 每帧存为独立文件夹 (pose.txt, color_image.jpg, depth_image.png...), 无统一数据格式, I/O 开销大 |
| PyBullet IK 不够鲁棒 | null-space IK 依赖手工参考姿态 `HAND_Q` 和硬编码偏移量, 对不同手型泛化性差 |
| 数据规模有限 | 总计约 90 分钟有效数据, 787 个 demo, 远小于 EgoDex 的 829 小时 |
| 仅支持 LEAP Hand | retargeting 代码硬编码了 LEAP Hand URDF, 无法直接迁移到其他灵巧手 |

### 未来方向展望

1. **规模化**: 结合更轻量传感器 (如 Apple Vision Pro ARKit) 和更长续航, 扩展到千小时级别。
2. **跨手迁移**: 用通用 retargeting 框架 (如 dex-retargeting) 替换硬编码 IK, 支持多种灵巧手。
3. **力感知**: 集成触觉传感 (如 GelSight, conformal textiles) 到采集系统。
4. **Foundation model 接入**: DexCap 数据可作为 VLA (Vision-Language-Action) 预训练的一个 embodiment 源。

---

## 7. Paper vs Code Discrepancies

| 维度 | 论文描述 | 代码实现 |
|------|----------|----------|
| 腕部追踪 | T265 SLAM | 2024-08 更新改为 HTC VIVE Tracker (OpenXR API), 需外部基站, 便携性降低 |
| EMF 手套数据 | "3D location of each fingertip" | 代码中 Rokoko 数据需要坐标轴翻转 (`switch_vector_from_rokoko`: x, -z, y) + z 轴取反, 论文未提及 |
| Perceiver encoder | 论文称使用 Perceiver, 是默认最优 | 代码中 `perceiverio.py` 实际是 Set Transformer (ISAB + PMA) 实现, 不是 DeepMind 的 Perceiver IO |
| 数据存储 | 论文未详述 | 每帧独立文件夹, 包含 6 个 txt + 2 个图像文件, 后处理为 robomimic HDF5 格式 |
| Robot hand mesh 注入 | "merge the point clouds of the transformed links" | 代码中每手采样 80000 点 (硬编码), 左手通过 y 轴翻转 (`new_points_left[:, 1] = -1.0 * ...`) 从右手 mesh 得到 |
| 安装角修正 | 论文一句话 "camera mount on the gloves are 45 degree facing up" | 代码中有复杂的 45 度旋转修正 + 标定偏移加载 (`calib_offset.txt`, `calib_ori_offset.txt`) |
| 点云下采样数 | 论文称 K 点 | 代码中默认 `num_points_to_sample = 10000` |
| Action gap | 论文称 action 为 next state | 代码中 `action_gap = 5`, 即 action 是 5 帧后的 state, 非紧邻下一帧 |
| Diffusion scheduler | 论文未明确说明 | 代码使用 DDIM (10 inference steps), 非 DDPM |
| BatchNorm 替换 | 论文未提及 | 代码中有关键注释: "replace all BatchNorm with GroupNorm to work with EMA" |
| 控制频率 | 论文称 60Hz 采集 -> 20Hz 控制 | 代码中数据确实每帧存储, 下采样在 HDF5 构建时通过 action_gap 间接实现 |

---

## 8. Cross-Paper Comparison

### 8.1 数据采集系统对比

| 维度 | DexCap (2024) | EgoDex (2025) | HATO (2024) | UniDex (2026) | DexMimicGen (2024) |
|------|---------------|---------------|-------------|---------------|-------------------|
| **数据来源** | 人手 mocap + 环境点云 | 人手视频 (Apple Vision Pro) | 遥操 (VR -> 真实机器人) | 人手视频 -> sim retarget | sim 中自动生成 |
| **是否需要机器人** | 否 (采集时) | 否 | 是 (2x UR5e + 2x Psyonic) | 否 (采集时) | 是 (sim 中) |
| **手部追踪** | Rokoko EMF + T265 SLAM | ARKit skeleton (Apple) | VR 控制器 -> IK | 视频手部追踪 (HaMeR 等) | 遥操记录关节角 |
| **环境观测** | 胸前 L515 LiDAR (点云) | 头戴 AVP 前置相机 (RGB) | 3x RealSense (RGB-D) | N/A (用现有数据集) | sim 渲染 |
| **触觉信息** | 无 | 无 | 有 (每指 6 传感器, 共 60) | 无 | sim 中有接触信号 |
| **数据规模** | ~90 min, 787 demos | 829 hours, 338K episodes | ~4h, 数百 demos/task | 50K trajectories (合成) | 21K demos (自动生成) |
| **成本** | ~$4K | Apple Vision Pro (~$3.5K) | >$100K (机器人+VR) | 低 (利用已有数据集) | 低 (仅需 sim) |
| **便携性** | 高 (背包, 3.96 磅) | 高 (头戴) | 无 (固定机器人) | N/A | N/A |
| **目标机器人手** | LEAP Hand (4 指) | 无特定 (hand skeleton) | Psyonic Ability (5 指 6 DoF) | 8 种手 (通用) | 多种 (sim) |
| **策略学习** | Diffusion Policy + 点云 | 手部轨迹预测 (benchmark) | Diffusion Policy + 多模态 | 3D VLA + flow matching | BC / Diffusion Policy |

### 8.2 核心差异分析

**DexCap vs EgoDex**:
- DexCap 提供精确的 EMF-based 指尖 3D 位置 + SLAM 6-DoF 腕部, 精度远高于 EgoDex 的视觉 skeleton。但 EgoDex 的规模 (829h vs 90min) 大近 500 倍, 更适合 foundation model 预训练。
- DexCap 是端到端的 "采集->retarget->策略" pipeline, EgoDex 主要作为数据集和 benchmark, 策略学习停留在轨迹预测层面。
- EgoDex 使用 Apple Vision Pro 的 on-device SLAM 追踪, 比 DexCap 的 T265 方案更商业化和稳定。

**DexCap vs HATO**:
- HATO 是遥操系统, 数据直接是 robot joint-space, 无需 retargeting, 无 embodiment gap。DexCap 的人手数据需要 IK retargeting, 引入额外误差, 但不需要机器人硬件。
- HATO 有触觉传感 (60 维), DexCap 完全缺失。触觉对精细力控任务 (如稳定夹持) 至关重要。
- HATO 的 VR 控制器到 Ability Hand 的映射极度简化 (grip 按钮控制 4 指, thumbstick 控制拇指), 牺牲了独立指控。DexCap 保留了独立指尖追踪。
- 采集速度: DexCap 约 3x 遥操, HATO 的遥操受限于操作者学习曲线。

**DexCap vs UniDex**:
- UniDex 不自行采集数据, 而是将现有人手数据集 (H2O, HOI4D, HOT3D, TACO) 统一处理, 通过 kinematic retargeting 生成 8 种机器人手的 50K+ 轨迹。DexCap 只支持 LEAP Hand。
- UniDex 提出了 FAAS (82-dim unified action space) 跨手统一动作空间, DexCap 的动作空间是 LEAP-specific 的 46 维。
- UniDex 是 3D VLA 架构 (Uni3D + PaliGemma + Flow Matching), 支持 language conditioning, 比 DexCap 的纯 diffusion policy 更通用。
- 但 UniDex 的数据来源全部是视频追踪, 精度不如 DexCap 的 EMF mocap。

**DexCap vs DexMimicGen**:
- DexMimicGen 在 sim 中从少量人类示教 (60 个) 自动扩增到 21K, 核心思路是轨迹变换 + 物理仿真回放。DexCap 直接采集真实世界数据, 无 sim-to-real gap。
- DexMimicGen 提供了完整的 real-to-sim-to-real pipeline, DexCap 只在 real 中工作。
- DexMimicGen 的 per-arm subtask segmentation 处理双手协调更系统化, DexCap 依赖 human-in-the-loop correction 补救。
- DexMimicGen 需要 sim 环境 (物体 URDF/mesh), DexCap 对物体无要求, 直接在 wild 采集。

### 8.3 对灵巧操作数据采集的总体启示

| 启示 | 说明 |
|------|------|
| 便携 vs 精度的 trade-off | DexCap 和 EgoDex 走便携路线, HATO 走精度路线。UniDex 试图用计算弥补数据精度 |
| 力信息是关键缺失 | DexCap、EgoDex、UniDex 均无力传感。HATO 是唯一集成触觉的, 且实验证明触觉对接触丰富任务提升显著 |
| 点云 > 图像 (对跨 embodiment) | DexCap 的实验明确证明点云表征在跨 embodiment 场景下的优势, UniDex 也用 3D 点云。这对 robotics FM 的输入表征选择有重要参考 |
| 自动数据增强是趋势 | DexMimicGen 的自动扩增 + UniDex 的跨手 retargeting 表明, 少量真实数据 + 计算扩增是更 scalable 的路径 |
| retargeting 精度决定上限 | DexCap 的 fingertip IK 简洁但粗糙, UniDex 的全身 retargeting 更通用, 但所有方案都回避了力/接触层面的 retargeting |
