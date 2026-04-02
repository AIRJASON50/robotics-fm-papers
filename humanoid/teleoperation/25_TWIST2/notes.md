# TWIST2 (Ze 2025) -- Takeaway Notes

> 一句话: 用 VR 设备 (PICO4U) 替代 MoCap 实现 portable 全身遥操作 + 数据收集，配合 hierarchical visuomotor policy 实现 egocentric vision 驱动的自主全身控制。

## 核心贡献

- **Portable full whole-body teleoperation**: PICO4U VR headset + 2 ankle trackers ($1000)，替代传统 MoCap ($10k+)。1 分钟 setup，单人操作
- **TWIST2 Neck**: $250 的 2-DoF add-on 颈部 (Dynamixel 电机 + 3D 打印件)，为 Unitree G1 添加 egocentric active vision
- **Hierarchical visuomotor policy**: low-level = PPO motion tracker (task-agnostic)，high-level = Diffusion Policy (egocentric vision -> whole-body joint positions)
- **数据收集效率**: 15-20 分钟收集 100 个 demo，接近 100% 成功率。全系统开源

## 为什么重要

- 解决了 humanoid data collection 的可扩展性问题: MoCap 设备贵、不便携、需要实验室环境
- 首次实现 vision-based autonomous full-body humanoid control (不只是 upper-body manipulation 或 lower-body locomotion)
- 开源完整 pipeline: hardware design + retargeting + low-level controller + data collection + policy learning
- 直接回应了 robotics FM 对 large-scale humanoid data 的需求 (类比 DROID 对 manipulation data 的贡献)

## 对你 (RL->FM) 的 Takeaway

1. **Data infrastructure 是 FM 的前提**: LLM 有 web text，manipulation 有 DROID/Open-X，humanoid 缺数据。TWIST2 提供了一条 scalable 的数据收集路线
2. **Hierarchical control 是当前最实用的架构**: low-level RL tracker (task-agnostic, 训练在 sim) + high-level policy (task-specific, 训练在 real demo)。这种分层在 BeyondMimic / ASAP 中也出现
3. **Egocentric vision 是关键**: 第三人称视角限制了 mobile manipulation 的范围。Active stereo vision (ZED Mini) + 颈部追踪 = 人类级别的视觉灵活性
4. **Motion retargeting 的工程细节**: 下半身用 position+rotation constraint (减少 foot sliding)，上半身只用 rotation constraint (避免 global pose 噪声传播)。这类工程 trick 对 teleoperation 质量至关重要
5. **Low-level controller 的 scaling**: 20k motion clips 训练一个 general tracker，混合 GMR retargeted data + AMASS + in-house MoCap + 73 PICO clips。小量 teleoperation-specific data 弥合 domain gap

## 与知识库其他内容的关联

- **TWIST (v1)** (`humanoid/teleoperation/`): TWIST2 是其 portable 升级版，从 MoCap 切换到 VR，保留 full whole-body control
- **BeyondMimic** (`25_BeyondMimic`): 同样使用 motion tracking RL 做 low-level controller，但 BeyondMimic 的 high-level 是 diffusion planning，TWIST2 的 high-level 是 visuomotor BC
- **Diffusion Policy** (`robotics/policy_learning/DiffusionPolicy`): TWIST2 的 high-level policy 使用 Diffusion Policy 架构
- **DROID** (`robotics/policy_learning/DROID`): 类比关系 -- DROID 是 manipulation 的 scalable data infra，TWIST2 是 humanoid 的 scalable data infra
- **pi_0 / pi_0.5** (`robotics/families/pi_Series`): TWIST2 收集的数据格式 (egocentric stereo + whole-body joints) 适合训练类 VLA 的 humanoid foundation model
