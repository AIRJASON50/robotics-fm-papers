# UniDex Notes

UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos
arXiv:2603.22264, CVPR 2026
Tsinghua University, Shanghai Qizhi Institute, UNC Chapel Hill

---

## 1. Core Problem

灵巧手操作面临三大瓶颈:

1. **数据瓶颈**: 灵巧手遥操数据采集成本极高, 远超 gripper。现有大规模数据集 (如 DROID, OXE) 几乎全部面向 gripper, 灵巧手缺乏可用的预训练数据集。
2. **Embodiment 异构性**: 灵巧手在 DoF (6-24)、运动学结构、外观上差异巨大。不同手之间的数据和策略难以互通, 每换一种手就需要重新采集/训练。
3. **高维控制**: 灵巧手的关节维度远高于 gripper (1-2D), 需要更具表达力的动作空间和更高效的学习算法。

**核心洞察**: 灵巧机器手天然模仿人手设计, 而人类日常生活中自然产生大量操作数据 (尤其是 egocentric 视频)。如果能把人类视频转化为机器人可执行的轨迹, 就能突破数据瓶颈; 如果能定义跨手的统一动作空间, 就能解决 embodiment 异构性。

**为什么重要**: 这是第一个同时解决灵巧手预训练数据规模化、跨手统一表示、3D VLA 策略三个问题的完整 foundation suite, 提供了从数据构建到模型训练到人类数据采集的全链路方案。

---

## 2. Method Overview

UniDex 由三个组件构成:

```
                    UniDex Foundation Suite
                    =======================

┌─────────────────────────────────────────────────────────────────┐
│  UniDex-Dataset: 50K+ trajectories, 8 hands, 9M frames         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐│
│  │ H2O, HOI4D,  │──>│ Human-Robot  │──>│ Robot-centric data   ││
│  │ HOT3D, TACO  │   │ Transform    │   │ (pcd + action + lang)││
│  │ (egocentric) │   │ Pipeline     │   │ x 8 hands            ││
│  └──────────────┘   └──────┬───────┘   └──────────────────────┘│
│                            │                                    │
│          ┌─────────────────┴──────────────────┐                │
│          │   Kinematic Retargeting (IK)        │                │
│          │   + Visual Alignment (mask + render)│                │
│          └────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                    FAAS (82-dim unified action space)
                    ┌─────────┴──────────┐
                    │ 18 dim wrist       │
                    │ 64 dim hand joints │
                    │ (32 per hand)      │
                    └─────────┬──────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│  UniDex-VLA: 3D Vision-Language-Action Policy                   │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Uni3D    │  │PaliGemma  │  │ Action    │  │Flow Matching │ │
│  │ (3D enc) │──│ (VLM      │──│ Expert    │──│ (denoising)  │ │
│  │ EVA-02-L │  │  backbone) │  │ (proprio/ │  │ 10 steps     │ │
│  │ 512 grps │  │  18 layers │  │  action)  │  │ Euler        │ │
│  └──────────┘  └───────────┘  └───────────┘  └──────────────┘ │
│  Pretrain: UniDex-Dataset | Finetune: 50 demos per task        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│  UniDex-Cap: Portable Human Data Capture Setup                  │
│  Apple Vision Pro (hand/head pose) + RealSense L515 (RGB-D)    │
│  -> Human-Robot Transform -> Co-training with robot demos       │
│  Human:Robot exchange rate ~= 2:1                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Designs

### 3.1 FAAS (Function-Actuator-Aligned Space, 功能-执行器对齐空间)

**是什么**: 82 维的统一动作向量。前 18 维编码腕部位姿 (每只手 9D: 6D 连续旋转表示 + 3D 平移), 后 64 维编码手指关节命令 (每只手 32 个 slot)。每 5 个 slot 对应一根手指 (thumb 0-4, index 5-9, middle 10-14, ring 15-19, little 20-24), 加上 2 个额外腕关节 slot (Shadow Hand) 和 5 个预留 slot。

**核心机制**: 按 functional role 而非 URDF joint index 映射。不同手的拇指外展关节 (thumb abduction) 无论在 URDF 中编号是什么, 都映射到 FAAS index 0; 食指弯曲 (index flexion) 都映射到同一位置。未使用的 slot 填 0。

**为什么有效**:
- 比 DexLatent 的 VAE latent space 更直接: 不需要训练 encoder/decoder, 不引入重建误差
- 比 pi_0 的 left-aligned action 更有语义: 功能相似的关节共享坐标, 而非按 DoF 数量简单对齐
- 32 slot 中预留了扩展空间, 新手型只需定义 joint-to-FAAS mapping 即可接入
- 缺点: 手动定义 mapping 需要领域知识, 且假设不同手的"功能等价"是良定义的

**Robotics 启示**: FAAS 与 humanoid 领域的 FSQ (SONIC) 思路类似 -- 找到跨 embodiment 的 functional 对齐轴。区别在于 FAAS 是手动设计的显式映射, 而 FSQ/DexLatent 是学习出的隐式对齐。手动设计的优势是 zero-shot (不需要训练数据就能接入新手), 劣势是可能丢失手型特有的精细控制能力。

### 3.2 Human-to-Robot Transformation Pipeline (人机转换流水线)

**是什么**: 将 egocentric 人类视频转化为 robot-centric 训练数据的两阶段流程:

**Stage 1 -- Kinematic Retargeting (运动学重定向)**:
- 从人手姿态提取 m 个指尖位置作为 IK target
- 引入 6-DoF dummy base offset T_offset, 通过 human-in-the-loop 交互调整
- 使用 PyBullet IK solver 求解机器人关节角 q, 最小化指尖误差
- 对 mimic joint (联动关节) 做迭代修正: q_js = k * q_jm + c

**Stage 2 -- Visual Alignment (视觉对齐)**:
- 用 WiLoR + SAM2 分割并 mask 掉人手
- 将 retarget 后的机器手 mesh 渲染到场景 pointcloud 中
- 通过 pinhole 相机模型重投影, 处理遮挡排序

**为什么有效**: 同时解决了 kinematic gap (通过 fingertip IK + dummy base 调整) 和 visual gap (通过 mask + render)。"Robot-centric" 的数据格式意味着预训练和微调之间没有额外的 domain gap, 不像 EgoVLA 那样在 post-training 时还需要 IK 转换。

**关键设计决策**: Human-in-the-loop 而非全自动。论文承认全自动 retargeting 在接触密集场景下不够可靠, 通过轻量级 GUI 让人类调整 dummy base offset, 在质量和效率之间取得平衡。实际操作中只需对每个 (数据集, 手型) 组合做一次基础校准, 然后对少量 contact-rich 片段做微调。

### 3.3 Mixture-of-Expert Joint Architecture (混合专家联合架构)

**是什么**: 模型架构直接继承 pi_0 (Vision-Language-Action Flow Model, 视觉-语言-动作流匹配模型), 但做了关键修改:

- 用 Uni3D (基于 EVA-02-Large ViT, 512 groups x 64 points) 替换 SigLIP 2D 视觉编码器
- 三个 mixture: VLM (hidden=2048, Gemma 权重), Proprio (hidden=1024), Action (hidden=1024)
- Block attention mask: pointcloud/text 自注意力, proprio/action 可以看到 pointcloud/text, action 还可以看到 proprio
- Proprio 和 Action 共享权重 (tie_action_proprio_weights)
- 18 层 Transformer, 8 attention heads, 1 KV head (GQA), head_dim=256
- Flow matching training: conditional vector field v_theta, 10 步 Euler integration

**为什么有效**: 3D pointcloud 输入在 egocentric 单视角场景下提供了关键的深度信息, 尤其对工具使用的接触推理至关重要。相比 2D 图像, pointcloud 还支持几何数据增强 (DemoGen: 分割+平移物体生成新场景)。

**Code insight**: 推理时 VLM 和 proprio 的 KV cache 在第一次前向传播后缓存, action denoising 的 10 步只重新计算 action expert 部分, 显著降低推理开销。代码还实现了 guided inference (基于 Real-Time Execution of Action Chunking Flow Policies, arXiv:2506.07339), 用前一个 action chunk 引导当前 chunk 的生成。

---

## 4. Experiments

### 4.1 主要结果

5 个真实世界工具使用任务, 每个任务仅 50 条 demo 用于 fine-tuning:

| Task | Hand | UniDex-VLA | UniDex-VLA (No PT) | pi_0 | DP3 | DP |
|------|------|------------|---------------------|------|-----|-----|
| Make Coffee | Inspire 6DoF | ~85% | ~65% | ~40% | ~30% | ~25% |
| Sweep Objects | Inspire 6DoF | ~80% | ~55% | ~35% | ~40% | ~30% |
| Water Flowers | Wuji 20DoF | ~85% | ~60% | ~40% | ~25% | ~20% |
| Cut Bags | Wuji 20DoF | ~75% | ~40% | ~25% | ~15% | ~10% |
| Use Mouse | Wuji 20DoF | ~80% | ~65% | ~50% | ~35% | ~30% |
| **Average** | | **~81%** | ~57% | ~38% | ~29% | ~23% |

(注: 数值从论文 Figure 10 读取, 为近似值。论文报告的是 average task progress, 非 binary success rate)

### 4.2 泛化实验

| 泛化类型 | 设置 | 结果 |
|----------|------|------|
| Spatial | Make Coffee 任务, 物体放置在 OOD 位置 | UniDex-VLA 保持高成功率, +DemoGen 增强后接近满分 |
| Object | 替换黑色水壶为不同颜色/大小/形状的紫色水壶 | 性能基本保持 |
| Cross-Hand (zero-shot) | Inspire -> Oymotion (6DoF): 60% success | 基线接近 0% |
| Cross-Hand (zero-shot) | Inspire -> Wuji (20DoF): 40% success | 基线接近 0% |

### 4.3 Human-Robot Co-Training (UniDex-Cap)

| 人类 demo 数 (h) | 机器人 demo 数 (r) | Average Task Progress |
|-------------------|---------------------|----------------------|
| 0 | 50 | ~80% (baseline) |
| 100 | 25 | comparable to r=50 |
| 200 | 0 | ~0% (robot data indispensable) |

关键发现:
- Human:Robot exchange rate 约 2:1 (两条人类 demo 约等于一条机器人 demo)
- 人类 demo 采集速度比机器人快 ~5.2x
- 纯人类数据 (r=0) 完全不能工作, 说明 domain gap 仍然需要少量机器人数据弥合

### 4.4 训练设置

| 阶段 | GPU | Batch Size | LR | Epochs | 耗时 |
|------|-----|------------|-----|--------|------|
| Pre-training | 8x H800 | 128 (4x4x8) | 1e-4, cosine decay | 3 (~30K steps) | ~24h |
| Post-training | 2x H800 | 8 | 2.5e-5, no scheduler | 50 (~3K steps) | ~4h |

---

## 5. Related Work Analysis

### 在灵巧手 FM 领域的定位

| 维度 | UniDex | DexGraspVLA | DexLatent | UltraDexGrasp |
|------|--------|-------------|-----------|---------------|
| 数据来源 | 人类视频 retarget | 仿真 + 真机遥操 | 真机遥操 | 纯仿真合成 |
| 跨手机制 | FAAS (手动映射) | 无 (单手) | VAE latent space | 无 (单手) |
| 感知模态 | 3D pointcloud | 2D image + language | 2D image | 3D pointcloud |
| 任务范围 | 工具使用 (5 tasks) | 抓取 (cluttered scene) | 抓取 + 简单操作 | 抓取 (1000+ objects) |
| 预训练规模 | 50K traj, 8 hands | -- | cross-hand VAE | 20M frames |

**填补的 gap**: 此前的灵巧手 FM 要么只关注单一手型 (DexGraspVLA, UltraDexGrasp), 要么跨手方案依赖学习式 latent space 需要训练数据 (DexLatent)。UniDex 首次实现了:
1. 从免费的人类视频构建大规模多手预训练数据
2. 用显式功能映射 (FAAS) 实现 zero-shot 跨手迁移
3. 3D VLA 在工具使用 (非仅抓取) 任务上的验证

### 与 learning from human videos 的区别

大多数从人类视频学习的方法 (R3M, VIP, LAPA) 只提取视觉表征而不产生 action supervision。EgoVLA 虽然预训练于人类视频, 但在 post-training 时需要额外的 IK alignment 步骤。UniDex 通过 retargeting pipeline 直接生成 robot-centric action label, 使得预训练和微调完全同构。

---

## 6. Limitations & Future Directions

### 论文自述局限

- 未利用 large-scale action-free egocentric datasets (如 Ego4D 的非操作片段)。只使用了有明确手-物交互的 RGB-D 视频。

### 补充分析

1. **Retargeting 质量上限**: Human-in-the-loop retargeting 虽然实用, 但质量取决于人工校准。尤其是 contact plausibility -- fingertip IK 只对齐位置, 不保证力和接触法线正确。对于需要精确力控的任务 (如精密装配), retarget 数据的质量可能不够。

2. **单视角 RGB-D 限制**: 所有实验只用一个 egocentric RealSense L515 相机。对于存在严重遮挡的任务 (如从容器内取物), 单视角 pointcloud 信息不足。

3. **FAAS 的 expressiveness 上限**: 32 slot per hand 是否足以覆盖未来更高 DoF 的手? Shadow Hand (24 DoF) 已经需要额外的 wrist slot。如果出现 30+ DoF 的手型, slot 可能不够用。

4. **Tool-use 偏向**: 5 个任务全部是工具使用 (water kettle, sweeper, spray bottle, scissors, mouse)。对于 in-hand reorientation 或精密装配等任务的表现未知。

5. **Human-Robot co-training 的 exchange rate**: 2:1 的比例意味着人类数据的价值密度只有机器人数据的一半。随着机器人遥操效率提升 (如更好的手套/VR 系统), 人类数据的相对优势可能缩小。

### 未来方向

- 扩展到 action-free 视频 (Ego4D 规模) 做视觉预训练
- 利用仿真验证 retarget 轨迹的物理可行性 (contact filtering)
- 结合 RL fine-tuning (如 RL Token 的思路) 提升接触精度
- 双手协调操作 (FAAS 已经预留了双手的 82 维空间)

---

## 7. Paper vs Code Discrepancies

### 7.1 架构命名

论文称架构 "largely follows pi_0", 代码文件名为 `unidex.py` 但注释写 "PiZero model migrated from open-pi-zero"。模块引用路径在 config 中写作 `src.openmodel.modules` 和 `src.openmodel.joint_model`, 但实际代码中是 `src.unidex.modules` 和 `src.unidex.joint_model` -- 说明代码可能经历了从 openmodel 到 unidex 的重命名, config 中有残留。

### 7.2 Action Dimension

Config 中 `action_dim: 82`, `proprio_dim: 82`, 与论文描述的 FAAS 82 维一致。`horizon_steps: 30` (30 步 action chunk), 这个细节论文中未明确提及。

### 7.3 Pointcloud Encoder 配置

Config 使用 `uni3d_l.yaml` 指定 `eva02_large_patch14_448`, pc_feat_dim=1024, 但 pretrained 路径写的是 `model-b.pt` -- 文件名暗示是 base 模型而非 large。这可能是命名不一致或者 Uni3D 的 ViT-L backbone 使用了 base 预训练权重。`trans2embed: False` 表示输出不是单 token 而是 512 个 group tokens, 这与论文描述的 pointcloud token 化一致。

### 7.4 Guided Inference (论文未提及)

代码实现了 `guided_inference` 方法, 引用 "Real-Time Execution of Action Chunking Flow Policies" (arXiv:2506.07339)。这个 guided inference 使用前一时刻的 action chunk 作为 target, 通过梯度引导当前 denoising 过程, 实现 action chunk 之间的平滑衔接。论文中未讨论此技术, 但 `PointCloudUniDexInference` 的 `forward` 方法默认使用 guided inference 而非朴素 denoising。

### 7.5 LoRA 支持 (论文未提及)

代码实现了完整的 LoRA (Low-Rank Adaptation, 低秩适应) 和 4-bit 量化支持, 但 config 默认 `lora: False`, `quantize: False`。论文未讨论 LoRA fine-tuning, 可能是为后续工作预留的接口。

### 7.6 FAAS Joint Mapping

论文说 FAAS mapping 在 Appendix Fig. 14 中展示, 代码中实际映射定义在 `src/assets/utils/hand_utils.json` (JSON 文件), 由 `hand_utils.py` 加载。这个 JSON 文件包含 `JOINT_MAP`, `RETARGET_JOINT_MAP_SCALE`, `RETARGET_JOINT_MAP_OFFSET` 等详细映射, 但代码仓库中该 JSON 文件路径指向的位置在仓库根目录的 `src/assets/utils/` 下 (未在公开仓库文件列表中直接看到), 可能需要从 Hugging Face 下载模型时获取。

### 7.7 训练框架

代码使用 PyTorch Lightning + Hydra 配置系统。预训练和微调共用同一个 `train.py`, 通过配置切换。`finetune.py` 文件存在但似乎是 `train.py` 的变体。

---

## 8. Cross-Paper Comparison

### 8.1 与 DexLatent 的跨手迁移方案对比

| 维度 | UniDex (FAAS) | DexLatent (VAE Latent) |
|------|--------------|----------------------|
| 表示方式 | 82D 显式映射 (手动定义) | 32D 学习式 latent (VAE) |
| 新手接入 | Zero-shot: 只需定义 joint-to-FAAS mapping | 需训练新 encoder/decoder head |
| 对齐基础 | 功能角色 (thumb flex = thumb flex) | Pinch geometry (fingertip 距离/方向) |
| 语义保持 | 高 -- 每个维度有明确的物理含义 | 低 -- latent 维度无显式语义 |
| Expressiveness | 受限于预定义 slot 数 (32 per hand) | 可通过增大 latent dim 扩展 |
| 训练数据需求 | 无 (手动映射) | 需要各手的 FK 模型做自监督训练 |
| Arm 处理 | 9D wrist pose (6D rot + 3D trans) | Pass-through (7D arm joints 不走 VAE) |
| 与 VLA 的集成 | 直接作为 action space | VLA 在 latent space 预测, decoder 解码 |

**总结**: FAAS 更 practical (无训练, 语义明确), DexLatent 更 principled (学习对齐, 可处理非对称结构)。FAAS 假设"相同功能 = 相同控制语义", 这在结构相似的拟人手之间成立, 但对于非拟人手 (如 parallel jaw + 2 fingers) 可能失效。

### 8.2 灵巧手 Foundation Model 全景对比

| 特性 | UniDex | DexLatent | DexGraspVLA | UltraDexGrasp | PAM |
|------|--------|-----------|-------------|---------------|-----|
| **Year** | 2026 | 2026 | 2025 | 2026 | 2026 |
| **数据来源** | 人类视频 retarget | 真机遥操 | 仿真+真机 | 纯仿真 (合成) | 仿真 HOI 视频生成 |
| **数据规模** | 50K traj, 9M frames | ~200 demos/hand | -- | 20M frames | 生成视频 |
| **手型数量** | 8 种 (6-24 DoF) | 3 种 | 1 种 | 1 种 | -- |
| **跨手机制** | FAAS (显式映射) | VAE latent | 无 | 无 | -- |
| **感知模态** | 3D pointcloud | 2D image | 2D image + lang | 3D pointcloud | 2D video |
| **Policy 架构** | 3D VLA (PaliGemma + Uni3D) | pi_0 + VAE | VLM + Diffusion | BC + PointNet | 视频扩散模型 |
| **任务范围** | 工具使用 (5 tasks) | 抓取+操作 | 抓取 (cluttered) | 抓取 (1000+ obj) | 数据增强 |
| **实机结果** | 81% avg task progress | 74% SR (cross-hand) | 90%+ grasp SR | 93% grasp SR | FVD 29.13 |
| **关键贡献** | 人类视频->多手数据 | 跨手 latent space | 层级 VLA 架构 | 合成数据 pipeline | 仿真视频生成 |

### 8.3 与 Policy Learning Baselines 对比

| 方法 | 表示 | 数据效率 | 3D 感知 | 预训练 | 跨手 |
|------|------|----------|---------|--------|------|
| Diffusion Policy (DP) | low-dim state | 中 | 否 | 否 | 否 |
| 3D Diffusion Policy (DP3) | pointcloud + low-dim | 中 | 是 | 否 | 否 |
| ACT | image + joint | 高 (少量 demo) | 否 | 否 | 否 |
| pi_0 | image + lang + FAAS | 高 | 否 | 是 (gripper) | 受限 |
| **UniDex-VLA** | **pcd + lang + FAAS** | **高** | **是** | **是 (8 手)** | **是** |

UniDex-VLA 相比 DP/DP3: 预训练带来的 prior 在 50 demo 的 low-data regime 下优势明显 (81% vs 23-29%)。

UniDex-VLA 相比 pi_0: 同样基于 flow matching, 但 (1) 用 3D pointcloud 替代 2D image 获得更好的空间理解, (2) 在灵巧手数据上预训练而非 gripper 数据, 领域匹配更好。

### 8.4 与 Human2Robot 方法对比

| 方法 | 输入 | Retarget | 策略 | 物理验证 |
|------|------|----------|------|----------|
| BiDexHD | TACO 数据集 | 仿真中自动构建 | IPPO teacher + DAgger | 仿真 |
| DexMachina | ARCTIC 人手数据 | FK-based + residual | PPO (hybrid action) | 仿真+真机 |
| DexTrack | GRAB+TACO | RL tracking controller | RL+IL hybrid | 仿真+真机 |
| HumDex | IMU 遥操实时 | 学习式 retargeting | 两阶段 IL | 真机 (人形) |
| **UniDex** | **egocentric video** | **IK + human-in-loop** | **3D VLA (flow matching)** | **真机** |

UniDex 与 human2robot 方法的核心区别: human2robot 方法将人类数据作为 RL/IL 的 reference motion, 仍然需要在仿真中训练 controller; UniDex 直接把 retarget 数据作为 VLA 的预训练 supervision, 跳过了仿真训练环节。这带来了更好的 scalability (不需要为每个任务建仿真环境), 但牺牲了物理 plausibility 的验证。
