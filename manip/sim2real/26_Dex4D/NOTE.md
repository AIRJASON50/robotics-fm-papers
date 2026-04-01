# Dex4D: Task-Agnostic Point Track Policy -- 学习笔记
> 一句话: 用 3D point tracks 作为 task-agnostic 目标表示, 训练 AnyPose-to-AnyPose policy, 结合视频生成模型实现 zero-shot 灵巧操作
> 论文: Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki, Shubham Tulsiani. CMU, 2025

## 这篇论文解决了什么问题
灵巧手操作的三角矛盾:
- 真机遥操数据昂贵且难规模化 (高自由度手的遥操本身就很难)
- 仿真学习需要为每个任务设计环境和 reward, 工程成本高
- 多数方法只能处理单任务/单物体, 缺乏跨任务迁移

## 核心想法 (用直觉解释)
**与 SimToolReal 类似的 insight**: 不为每个任务设计 RL 环境, 而是学一个 task-agnostic 的 AnyPose-to-AnyPose (AP2AP) 策略。

**关键区别在目标表示**: 用 **3D point tracks** (物体表面点的 3D 时间轨迹) 而非 6D pose。
- 视角无关 (domain-agnostic), 大幅简化 sim-to-real
- 视频生成模型 -> CoTracker3 追踪 -> depth 提升到 3D -> 作为 policy 输入

**Paired Point Encoding**: 把当前点和目标点 concat 成 6D pairs, 保持对应关系。

直觉: 球旋转时形状不变, 只有 correspondence 能区分不同 pose。Paired encoding 让 policy 知道"这个点要去哪"。

## 关键设计决策
| 决策 | 选择 | 为什么 |
|------|------|--------|
| 目标表示 | 3D point tracks (64 keypoints) | 比 6D pose 更 robust (视角无关); 比 2D flow 多深度信息 |
| Paired Point Encoding | Concat (current_xyz, target_xyz) per point -> PointNet | 保持 correspondence, 解决对称物体歧义 |
| 训练分阶段 | Stage 1-2 单类别 -> Stage 3 全类别 -> DAgger student | 先学接触动力学, 再泛化; 低频 (5Hz) 放松控制要求 |
| Student 架构 | PointNet backbone + 4-layer Transformer + future state prediction | Transformer tokenize 不同 obs; auxiliary loss 学物理动力学 |
| Z-axis reposing | 归一化掉物体 z 轴旋转 | 策略不需要学 z 旋转对称性, 大幅提高 sample efficiency |
| Keypoint masking | Random height masking (模拟桌面遮挡) | 而非让 tracker 完美工作 -- 训练 robust to missing info |

## 这篇论文之后发生了什么
- 3D point tracks 作为 manipulation 的通用接口被验证: geometry-aware + domain-robust
- 和 SimToolReal 形成对照: 两者都是 task-agnostic, 但目标表示不同 (point tracks vs 6D pose)
- Hardware 部分尚未开源, sim-to-real 完整流程无法完全复现
- Video generation model 的质量直接决定系统上限 -- 不合理生成 = 不可能任务

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | 3D point tracks 是比 6D pose 更好的 manipulation 接口 | 视角无关 + 保持 correspondence, 天然适合 sim-to-real; foundation model 的输出可以直接是 point tracks |
| 2 | Paired encoding 保持 correspondence 是关键 | 类比 attention: Q-K 的配对关系比独立编码更有效 |
| 3 | Video generation model + point tracking = 高层规划器 | 这是 foundation model 介入 robotics 的自然接口: 生成视频 -> 提取 tracks -> 控制 |
| 4 | Auxiliary future state prediction 加速 policy learning | 让 student 同时学动力学, 和 world model 的思路一致: 预测未来帮助当下决策 |
