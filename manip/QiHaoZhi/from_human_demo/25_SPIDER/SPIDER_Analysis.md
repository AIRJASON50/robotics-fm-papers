# SPIDER: Scalable Physics-Informed Dexterous Retargeting

> arXiv: 2511.09484 | Code: github.com/facebookresearch/spider
> Authors: Chaoyi Pan (CMU), Changhao Wang, Haozhi Qi, et al. (Meta FAIR), Guanya Shi (CMU), Jitendra Malik (FAIR/UC Berkeley)

---

## 1. 核心问题

人类运动数据（MoCap、视频、VR）海量可得，但是**纯运动学的**——只有位姿，没有力。由于 embodiment gap，这些数据不能直接在机器人上执行。

现有方法的不足：

| 方法 | 问题 |
|------|------|
| IK / dex-retargeting | 不满足物理约束，接触丰富的任务不可行 |
| 学习型网络 | OOD 泛化差，需要大量预训练 |
| RL (ManipTrans/DexMachina) | 每条轨迹要独立训练策略，不可扩展 |
| 遥操作 | 劳动密集，特定于单一形态 |

## 2. SPIDER 的核心思路

**不训练网络，不用 RL，用采样优化 (Sampling-based MPC) 在物理仿真器中直接优化控制序列。**

```
人类运动 → IK 运动学映射 → 采样优化 (物理仿真中) → 动力学可行的控制序列
```

## 3. 完整管线 (7 步，无训练)

```
Step 1: 数据集 → 统一 MANO 关键点 NPZ (5 指尖 3D + 腕 6D)
Step 2: 物体网格凸分解 (用于物理接触)
Step 3: 接触检测 (回放人手轨迹，记录接触时刻/位置)
Step 4: 场景 XML 生成 (机器人 + 物体 + 接触对)
Step 5: IK 求解 → trajectory_kinematic.npz (运动学参考)
Step 6: Sampling MPC + MuJoCo Warp → trajectory_mjwp.npz (物理优化)
Step 7: 导出到真实机器人控制格式 (可选)
```

## 4. 采样优化器 (核心算法)

**不是神经网络，是 Cross-Entropy Method 变体：**

```
for iteration in range(16):
    # 1. 在当前最优控制周围采样 1024 个候选
    candidates = best_ctrl + noise * annealing_scale(iteration)

    # 2. GPU 并行前滚 1024 个候选 (MuJoCo Warp)
    rewards = simulate_all(candidates)    # 跟踪误差 + 物理约束

    # 3. Top 10% softmax 加权平均更新
    best_ctrl = weighted_average(candidates, softmax(top_10%(rewards)))

    # 4. 退火: 逐步缩小搜索范围
    noise *= decay_factor
```

**退火调度 (Annealing)**：
- 外层 (迭代方向)：beta_1=0.85，从全局搜索 → 局部细化
- 内层 (时间维度)：beta_2=0.9，近期精确 / 远期探索
- 效果：早期大噪声发现可行接触模式，后期小噪声稳定接触

## 5. 虚拟接触引导 (核心创新)

**问题**：同一个物体运动可能对应多种接触模式（拇指-食指 vs 食指-中指），采样可能收敛到错误模式。

**方案**：在优化初期施加虚拟约束力，把物体"粘"在目标接触点上，随迭代逐渐松弛：

```
eta_t = eta_0 * 1.1^t    (eta_0 = 0.01)

优化初期: 虚拟力强 → 物体被强制移到参考位置 → 采样集中在正确接触附近
优化后期: 虚拟力消失 → 只靠真实接触力维持 → 物理可行
```

成功率提升 **18%**。

## 6. 奖励/Cost 函数

```
cost = W_pos * ||base_pos_sim - base_pos_ref||²    (1.0)
     + W_rot * ||base_rot_sim - base_rot_ref||²    (0.3)
     + W_joint * ||joint_sim - joint_ref||²         (0.003)
     + W_obj_pos * ||obj_pos_sim - obj_pos_ref||²  (1.0)
     + W_obj_rot * ||obj_rot_sim - obj_rot_ref||²  (0.3)
     + W_vel * ||qvel||²                            (0.0001)
     + contact_guidance_cost                         (curriculum)
```

## 7. 支持的机器人 (9 种形态)

### 灵巧手 (5 种)
| 手 | DOF | 手指数 |
|----|-----|--------|
| XHand | 12 (6腕+6手指) | 5 |
| Allegro | 16 | 4 (无小指) |
| Inspire | 12 | 5 |
| Ability | - | 5 |
| Schunk SVH | - | 5 |

### 人形机器人 (4 种)
| 机器人 | DOF |
|--------|-----|
| Unitree G1 | 29 |
| Unitree H1-2 | - |
| Fourier N1 | - |
| Booster T1 | - |

## 8. 数据规模

- **6 个数据集**: GigaHands, OakInk, ARCTIC, LAFAN1, AMASS, OMOMO
- **240 万帧** 动态可行的机器人数据
- **103 种物体**
- 比 RL 方法快 **10 倍**

## 9. 与 dex-retargeting 的对比

| 维度 | dex-retargeting (IK) | SPIDER |
|------|---------------------|--------|
| 优化变量 | 关节角度 | 控制序列 (力矩/位置指令) |
| 物理仿真 | 无 | 有 (MuJoCo Warp GPU) |
| 接触处理 | 几何距离近似 | 真实接触力 + 虚拟引导 |
| 输出 | 关节角序列 (需控制器执行) | **可直接部署的控制指令** |
| 重力/惯性 | 不考虑 | 自然包含 |
| 速度 | 毫秒级 | 秒级 (但比 RL 快 10x) |

## 10. 代码结构

```
spider/
├── examples/
│   ├── run_mjwp.py              # 主入口 (Hydra 配置驱动)
│   └── config/
│       ├── default.yaml         # 默认参数 (1024 samples, 32 iters)
│       └── override/            # 数据集/机器人特定覆盖
├── spider/
│   ├── config.py                # Config 数据类 (~150 字段)
│   ├── optimizers/sampling.py   # 采样 MPC 优化器 (核心)
│   ├── simulators/mjwp.py       # MuJoCo Warp 仿真器 (~1400 行)
│   ├── preprocess/
│   │   ├── ik.py                # 逆运动学
│   │   ├── generate_xml.py      # 场景生成
│   │   ├── decompose_fast.py    # 凸分解
│   │   └── detect_contact.py    # 接触检测
│   ├── process_datasets/        # 数据集处理器
│   ├── assets/robots/           # 11 种机器人 URDF/XML
│   └── viewers/                 # 可视化 (Rerun/Viser/MuJoCo)
```

## 11. 输入输出格式

**输入** (统一 NPZ):
```
qpos_wrist_right: [T, 7]       # 腕部 pos + quat(wxyz)
qpos_finger_right: [T, 5, 7]   # 5 指尖位姿
qpos_obj_right: [T, 7]         # 物体位姿
contact_right: [T, 5]          # 接触标记
contact_pos_right: [5, 3]      # 接触点位置
```

**输出** (优化后 NPZ):
```
qpos: [N_steps, ctrl_steps, nq]  # 物理可行的关节位置
qvel: [N_steps, ctrl_steps, nv]  # 关节速度
ctrl: [N_steps, ctrl_steps, nu]  # 可直接部署的控制指令
```

## 12. 对我的工作的启发

### 12.1 背景对照

| 维度 | SPIDER | 我的工作 (WujiHand + DexCanvas) |
|------|--------|-------------------------------|
| 手 | 5 种灵巧手 (12-22 DOF) | WujiHand 20-DOF |
| 重定向方法 | IK + 采样物理优化 | dex-retargeting (纯 IK) |
| 数据源 | MANO → 5 指尖 keypoint | MANO → 21 joint MediaPipe |
| 物理可行性 | 采样 MPC 保证 | RL motion tracking 学习 |
| 仿真器 | MuJoCo Warp (GPU) | MuJoCo/MJX (GPU) |

### 12.2 可借鉴的技术

#### (1) 物理可行的 Retargeting 替代纯 IK

当前流程:
```
DexCanvas MANO → dex-retargeting (IK) → retarget_angles → RL 学习跟踪
```
SPIDER 的流程:
```
MANO → IK → 采样优化 (物理仿真) → 物理可行的控制序列
```

**价值**: 如果用 SPIDER 做 retargeting，输出的轨迹本身就是物理可行的，RL 只需学残差修正。目前你的 RL 要同时学"跟踪运动学轨迹"和"满足物理约束"两件事。

#### (2) 虚拟接触引导 (Contact Guidance)

**价值**: 你的 DexCanvas 轨迹中手-cube 接触是关键。SPIDER 的虚拟接触引导思路可以应用到你的 RL 训练中——在训练早期用虚拟力辅助建立接触，后期逐渐退火移除。类似于 curriculum 式的接触辅助。

#### (3) Min-Max 鲁棒化

```python
# SPIDER: 取最差情况的 cost
for env_param in domain_randomization_params:
    reward = rollout(ctrl, env_param)
    min_reward = min(min_reward, reward)
use min_reward for optimization
```

**价值**: 比标准 DR (随机采样然后平均) 更保守，保证在最差物理参数下也能工作。可以应用到你的 RL 训练中——critic 评估最差情况而非平均情况。

#### (4) MuJoCo Warp 作为仿真后端

SPIDER 使用 MuJoCo Warp 做 GPU 并行仿真 (1024 环境)。你目前用 MJX，两者类似但 MuJoCo Warp 是更新的 GPU 后端，可能值得关注其发展。

### 12.3 关键差异：SPIDER 是离线优化，不是在线策略

SPIDER 为每条轨迹做离线优化 (秒级)，输出固定的控制序列——**开环的**。

你的 RL motion tracking 是在线策略——**闭环的**，能对扰动做实时反应。

两者的关系不是替代，而是互补：
- **SPIDER 生成高质量参考轨迹** → 物理可行
- **RL 策略学习跟踪这些轨迹** → 闭环鲁棒
- 论文自己也用了这种组合: `u_t = u_t^{SPIDER} + pi_theta(o_t)` (前馈 + 残差反馈)

### 12.4 优先级建议

| 优先级 | 可借鉴项 | 预期收益 |
|--------|---------|---------|
| **高** | 接触引导 curriculum (虚拟力退火) | 改善 hand-cube 接触建立 |
| **中** | SPIDER 替代 IK retargeting 生成参考轨迹 | 减轻 RL 学习负担 |
| **中** | Min-Max DR 替代 Average DR | 提升 worst-case 鲁棒性 |
| **低** | 前馈+残差架构 (`u_spider + pi(o)`) | 需要先有 SPIDER 轨迹 |
| **低** | MuJoCo Warp 后端迁移 | 等生态成熟 |
