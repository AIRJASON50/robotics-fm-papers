# HDMI -- 学习笔记
> 一句话: 从单目 RGB 视频中提取人-物交互轨迹，用 robot-object co-tracking 的 RL policy 学习全身人形机器人交互技能 (开门、搬箱子等)，zero-shot 部署到 Unitree G1
> 论文: Haoyang Weng, Yitang Li et al. (CMU Robotics Institute), 2025

## 这篇论文解决了什么问题
Humanoid 领域的 locomotion 和 manipulation 分别做得不错了，但 **whole-body loco-manipulation** (边走边操作物体) 仍然是难题。两个核心障碍:
1. **数据稀缺**: 自由运动的 motion capture 数据很多，但 human-object interaction (HOI) 的 3D 数据极少
2. **Contact-rich 的 RL 训练困难**: 物体交互涉及复杂接触，仅靠 motion tracking reward 不够 (参考动作从视频来，接触信息不准确)

之前的方法要么靠 task-specific reward engineering (不通用)，要么用 VLM/planner 做高层规划 (复杂且不够 robust)。HDMI 的方案: 用 RGB 视频作为 scalable 的数据源，学一个 unified 的交互策略。

## 核心想法 (用直觉解释)
**三阶段 pipeline**:
1. **Video -> Structured Dataset**: 用 GVHMR 从单目 RGB 视频估计人体 3D pose，用 LocoMujoco retarget 到机器人。手动标注 object trajectory 和 contact signal，生成 (robot_state, object_state, contact_flag) 的参考数据。
2. **Robot-Object Co-Tracking**: 不只 track 机器人姿态 (像 DeepMimic)，而是同时 track 机器人和物体的状态。policy 的目标是让机器人和物体都跟上参考轨迹。用 DeepMimic-style 训练 (RSI + phase variable + tracking reward + PPO)。
3. **Zero-shot Deployment**: 直接部署到 G1，不需要 fine-tuning。

直觉: 把 "人和物体一起动" 的视频变成 "机器人和物体应该怎么一起动" 的参考轨迹，然后让 RL 去学。关键是 "co-tracking" -- 机器人不只管自己的姿态，还要管物体到没到位。

## 关键设计决策
1. **Unified object representation**: 物体 pose 用 robot root frame 下的相对坐标表达，加上 reference contact points (物体上应该被触碰的点)。这个表示对 rigid body、articulated object 都通用。policy 不需要知道物体具体是什么形状。
2. **Residual action space**: policy 输出 delta_a (相对于参考关节角度的偏移)，而非绝对关节角度。直觉: 如果参考动作是跪姿，标准 policy 的初始探索在 standing pose 附近，会立刻摔倒; residual action space 让探索从参考姿态开始，大幅提升 sample efficiency。
3. **Unified interaction reward**: 除了 motion tracking reward，额外加一个 contact-promoting reward -- 奖励末端执行器靠近目标接触点 (position term) 并施加适当接触力 (force term, capped 防止过大)。这解决了视频 retargeting 产生的不准确接触信息问题。
4. **Contact signal gating**: 用二值 contact flag c_t 控制 interaction reward 是否激活。只有在参考动作指示 "应该有接触" 时才计算 contact reward，其他时间只做 motion tracking。
5. **Phase variable 足够 single-motion tracking**: 和 DeepMimic/ASAP 一致，一个 scalar phase variable 就能编码 "当前在动作序列的哪个位置"。

## 这篇论文之后发生了什么
- 67 次连续开门穿过 (34 分钟) 验证了长期鲁棒性
- 展示了 6 种真机任务 + 14 种仿真任务的通用性
- 从 RGB video 学交互技能的路线开辟了 scalable 的 humanoid skill acquisition 方向
- 局限: 需要手工标注 object trajectory 和 contact; 目前不支持 prehensile manipulation (需要灵巧手)

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | **Co-tracking (robot + object) 是 loco-manipulation 的自然 formulation** -- 把物体也纳入 tracking target，比 task-specific reward 更通用 | 灵巧手操作也可以用 co-tracking: 同时 track 手和物体 |
| 2 | **Residual action space 是处理 "远离 default pose" 动作的标配** -- 跪姿、弯腰等动作用 absolute action 几乎学不会 | 灵巧手的 precision grasp 也远离 default pose，应该考虑 residual |
| 3 | **RGB video 是最 scalable 的 demo source** -- 比 motion capture 便宜几个数量级 | YouTube 上海量人类操作视频，如何利用是 FM for robotics 的核心问题 |
| 4 | **Interaction reward 补偿 kinematic reference 的不准确** -- 视频来的 reference 没有力信息，用 force reward 在 RL 中 "修补" | 任何来自 noisy source (视频、teleoperation) 的 reference 都需要 RL 层面的补偿机制 |
| 5 | **简单方法 + 好 engineering > 复杂架构** -- HDMI 没用 diffusion/VAE/hierarchy，就是 DeepMimic-style tracking + 三个 targeted improvements | 不要过度设计，先把 baseline 做到位 |
