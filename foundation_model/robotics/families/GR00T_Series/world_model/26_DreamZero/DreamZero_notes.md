# DreamZero 分析笔记

**论文**: World Action Models are Zero-shot Policies
**作者**: Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng 等 (NVIDIA)
**系列**: GR00T Series -- WAM (World Action Model, 世界动作模型) 方向
**代码**: `dreamzero/` (PyTorch, 基于 Wan2.1-I2V-14B 视频扩散骨干)

---

## 1. Core Problem

DreamZero 要解决的核心问题是: **VLA (Vision-Language-Action, 视觉-语言-动作) 模型虽然继承了 VLM 的语义泛化能力，但缺乏时空动力学先验，无法泛化到训练数据中未见过的物理运动和新环境。**

| 挑战 | 具体表现 |
|------|----------|
| 语义-物理断层 | VLA 能识别 "move coke to Taylor Swift" (语义推理)，但无法执行 "untie the shoelace" (未见过的物理技能) |
| 数据依赖范式 | 当前 VLA 依赖大规模、重复性的 task-specific 遥操作数据；换一个环境或任务就需要重新采集 |
| 先验来源局限 | VLM 预训练来自 static image-text 数据，缺少 spatiotemporal dynamics 信息 |
| 推理延迟 | 视频扩散模型需要迭代去噪，naive 实现 14B 参数模型单次推理需 5.7 秒，无法闭环控制 |

**核心洞察**: 视频扩散模型在 web-scale 视频数据上预训练后，已经学会了丰富的物理动力学先验。如果将 action prediction 从 "直接模仿 state->action 映射" 转变为 "先预测未来视觉状态、再从中提取动作" (即 inverse dynamics)，就能利用视频模型的泛化能力来生成未见过的技能。这本质上是把 policy learning 分解为: (1) 视频预测 (继承自预训练) + (2) 逆动力学 (从视频中学习)，两者端到端联合训练。

---

## 2. Method Overview

DreamZero 是一个 14B 参数的 WAM，基于 Wan2.1-I2V-14B-480P 预训练视频扩散模型构建。核心公式将联合预测分解为视频预测和逆动力学:

$$\pi(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l) = \underbrace{\pi(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{video prediction}} \cdot \underbrace{\pi(\mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l+H}, \mathbf{q}_l)}_{\text{IDM}}$$

但与 DreamGen 使用两个独立模型不同，DreamZero 用 **单一端到端模型** 联合去噪 video + action。

### 架构流水线

```
Input:
  - Visual context: o_{0:l} --> VAE encoder --> latent z
  - Language instruction: c --> T5 text encoder --> text embedding
  - Proprioceptive state: q_l --> State encoder --> state embedding

Processing:
  Autoregressive DiT backbone (14B, flow matching)
  - Chunk-wise denoising: each chunk = K=2 latent frames
  - Teacher forcing: denoise current noisy chunk conditioned on clean previous chunks
  - Shared denoising timestep between video and action (standard) or decoupled (Flash)

Output:
  - Video latents --> VAE decoder --> predicted frames
  - Action latents --> Action decoder --> predicted actions (H=48 steps @ 30Hz)
```

### 训练目标

采用 Flow Matching (流匹配) 作为训练目标。给定 chunk index k 和 denoising timestep t_k:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\frac{1}{K}\sum_{k=1}^{K} w(t_k) \|\mathbf{u}_\theta([\mathbf{z}_{t_k}^k, \mathbf{a}_{t_k}^k]; \mathcal{C}_k, \mathbf{c}, \mathbf{q}_k, t_k) - \mathbf{v}^k\|^2\right]$$

其中 velocity target $\mathbf{v}^k = [\mathbf{z}_1^k, \mathbf{a}_1^k] - [\mathbf{z}_0^k, \mathbf{a}_0^k]$，即 clean signal 与 noise 之间的差。

### 推理闭环

关键设计: 在 autoregressive inference 中，执行完一个 action chunk 后，用 **真实观测 (ground-truth frames) 替换 KV cache 中的预测帧**。这消除了 autoregressive 视频生成固有的误差累积问题，是 WAM 相对于纯视频生成模型的独特优势。

---

## 3. Key Designs

### 3.1 Autoregressive Chunk-wise Architecture (vs Bidirectional)

DreamZero 采用 autoregressive (AR) 而非 bidirectional (BD) 架构进行视频-动作联合建模，这是与同期 WAM 工作的关键差异。

**AR 的三个优势**:

| 维度 | AR | BD |
|------|-----|-----|
| 推理速度 | KV cache 复用，3-4x 更快 | 每次需完整序列前向传播 |
| FPS 保真 | 保持原生帧率 (5FPS/30Hz)，视频-动作精确对齐 | 需要 subsampling 来匹配 caption 覆盖范围，扭曲原生 FPS |
| 语言对齐 | 逐 chunk 条件化，language 始终与当前视频窗口对齐 | 固定长度序列中，language annotation 可能描述了还未生成的部分 |
| 动作平滑性 | 通过整条序列的梯度反传实现更好的时间连续性 | -- |

**attention mask 设计 (从代码 `wan_video_dit_action_casual_chunk.py` 中提取)**:

序列结构为 `[first_image | image_blocks | action_blocks | state_blocks]`:
- **Image blocks**: 可以 attend to first image + 之前所有 image blocks (causal) + 当前 action block + 当前 state block
- **Action blocks**: 可以 attend to first image + 对应及之前的 image blocks + 自身 action block + 当前 state block
- **State blocks**: 仅 self-attention (作为 conditioning)

这种 block-wise causal attention 确保了: (1) video tokens 可以看到同 chunk 的 action/state 信息 (双向交互)，(2) 不同 chunk 之间严格 causal (autoregressive)。

### 3.2 DreamZero-Flash: Decoupled Noise Schedules

标准 DreamZero 中 video 和 action 共享相同的 denoising timestep $t_k \sim \mathcal{U}(0,1)$。这造成了 train-test mismatch: 训练时模型学的是 "video 和 action 同等噪声水平"，但 few-step inference 时 action 需要从 noise 完全去噪到 clean，而 video 仍然是 noisy 的。

**DreamZero-Flash 的解法**:

$$t_k^{\text{video}} = 1 - \eta, \quad \eta \sim \text{Beta}(\alpha, \beta), \quad t_k^{\text{action}} \sim \mathcal{U}(0,1)$$

使用 $\text{Beta}(7,1)$ 时，$\mathbb{E}[t_k^{\text{video}}] = 0.125$ (video 大部分时间处于高噪声状态)，而 action timestep 保持均匀分布。

**效果**: 将 diffusion steps 从 4 步减少到 1 步，推理从 ~350ms 降至 ~150ms，在 table bussing 任务上 task progress 仅从 83% 降至 74% (4-step baseline)。

**代码实现** (从 `wan_flow_matching_action_tf.py` 中确认):
- `decouple_video_action_noise: bool` -- 控制是否启用解耦噪声
- `video_noise_beta_alpha / video_noise_beta_beta` -- 控制 video 噪声 Beta 分布参数
- `video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)` -- 实际实例化

### 3.3 38x Inference Speedup Stack

DreamZero 通过三层优化将推理从 5.7s 降至 150ms:

| 层级 | 优化技术 | 加速倍数 | 核心思想 |
|------|---------|---------|---------|
| System-level | CFG Parallelism (2 GPU) | ~2x | Classifier-Free Guidance 的 conditional/unconditional forward 并行到两张 GPU |
| System-level | DiT Caching | ~4x (16 step -> 4 effective) | 当连续 velocity 预测的 cosine similarity 超过阈值时复用缓存 |
| Implementation | torch.compile + CUDA Graphs | -- | 消除 CPU overhead，算子融合 |
| Implementation | NVFP4 Quantization (Blackwell) | -- | 权重/激活量化到 NVFP4，QKV/Softmax 保留 FP8 |
| Implementation | cuDNN attention backend | -- | 替换默认 attention kernel |
| Model-level | DreamZero-Flash | ~2-4x | Decoupled noise schedule，1-step inference |

**代码中的 DiT Caching 实现** (从 `wan_flow_matching_action_tf.py`):
```python
# dit_step_mask controls which of 16 denoising steps actually run full DiT forward
# e.g., 8 steps: [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
```

这说明 DiT Caching 不是 runtime 动态决定的 (基于 cosine similarity)，而是预定义的 static mask schedule。论文描述的 "cosine similarity threshold" 在开源代码中简化为固定 step mask。

---

## 4. Experiments

### 4.1 实验设置

| 维度 | 详情 |
|------|------|
| 主要机器人 | AgiBot G1 (移动双臂操作) + Franka (单臂) |
| 跨体训练用 | YAM 机器人 + 人类第一人称视频 |
| 预训练数据 | AgiBot: ~500 小时遥操作数据，22 个真实环境；Franka: DROID 数据集 |
| Backbone | Wan2.1-I2V-14B-480P (14B image-to-video diffusion model) |
| 训练步数 | 100K steps, global batch size 128 |
| 冻结组件 | Text encoder, Image encoder, VAE |
| 更新组件 | All DiT blocks, State encoder, Action encoder, Action decoder |
| Baselines | GR00T N1.6 (from-scratch + from-pretrained), pi_0.5 (from-scratch + from-pretrained) |

**数据收集哲学的差异**: DreamZero 数据集平均每 episode 4.4 分钟、包含 ~42 个 subtask，强调多样性而非重复性。每个 task 采集 50 episodes 后即 deprecate，强制操作员不断提出新任务。这与 VLA 范式中 "每 task 数百次重复演示" 形成鲜明对比。

### 4.2 主要结果

| 实验 (Q) | 核心发现 | 数据 |
|----------|---------|------|
| Q1: 多样数据学习 | DreamZero 62.2% avg task progress vs 最佳 pretrained VLA 27.4% (>2x) | Seen tasks, unseen environments |
| Q2: 未见任务泛化 | DreamZero 39.5% vs pretrained VLA 16.3% (ironing, painting, untying shoelaces 等) | 10 unseen tasks |
| Q3: Post-training | DreamZero 匹配或超越 VLA，环境泛化在 fine-tuning 后保留 | Shirt folding, fruit packing, table bussing |
| Q4: 跨体迁移 | Robot-to-robot: 38.3% -> 55.4%; Human-to-robot: 38.3% -> 54.3% (仅 10-20 分钟 video-only data) | 9 unseen tasks |
| Q5: Few-shot 新体适应 | 30 分钟 play data 迁移到全新机器人 (YAM)，保留 zero-shot 泛化 | 11 tasks, 55 trajectories |
| Q6: Flash 性能 | 1-step Flash: 74% vs 4-step baseline: 83%，速度 ~2x | Table bussing |

### 4.3 消融实验

| 消融维度 | 结果 | 启示 |
|----------|------|------|
| 数据多样性 vs 重复性 | 500hr 多样 50% vs 500hr 重复 33% (PnP Easy) | WAM 的 IDM 学习需要多样的 state-action 对应关系 |
| 模型规模 (5B vs 14B) | 14B: 50% vs 5B: 21% | 更大的视频模型 = 更好的视频预测 = 更好的 action，VLA scaling 更大模型仍 0% |
| AR vs BD | Task progress 相似，但 AR 动作更平滑 + 推理 3-4x 更快 | AR 的 KV cache + 原生 FPS 保真 是关键优势 |

### 4.4 失败模式分析

论文明确指出: **大多数 DreamZero 失败源于 video generation 错误，而非 action prediction 错误**。Action 忠实地执行了 video 预测的轨迹，即使该轨迹是错误的 (如画白板时机器人把笔传给了另一只手)。这意味着:
- 提升 video backbone 质量会直接提升 policy 性能
- Policy 性能的 upper bound 由 video generation 质量决定

---

## 5. Related Work Analysis

DreamZero 在 robotics FM 的技术谱系中占据的位置:

| 范式 | 代表工作 | 先验来源 | 泛化维度 | 局限 |
|------|---------|---------|---------|------|
| VLA | RT-2, pi_0, GR00T N1.6 | VLM (static image-text) | 语义/对象泛化强，物理运动泛化弱 | 需要大量重复 task-specific 数据 |
| Latent World Model | DreamerV3, V-JEPA 2 | From-scratch latent dynamics | Data efficient (RL), 需要 test-time planning | 无法直接输出 action，需要 MPC/search |
| Video-as-Planner | SuSIE, UniPi | Video diffusion | 生成视觉计划 | 需要独立 IDM 提取 action，two-stage pipeline |
| Video-as-Data-Generator | DreamGen | Video diffusion | 合成数据增强 | 需要独立 IDM/LAPA，neural trajectory 质量有 ceiling |
| **WAM (Joint Video+Action)** | **DreamZero**, DreamGen+IDM, GR-2, LAPA-H | **Video diffusion (joint training)** | **同时泛化 skills + environments** | **推理成本高，视频质量决定上限** |

**DreamZero vs DreamGen 的本质区别**: DreamGen 是 "video model as data generator" (生成合成数据再训练独立 policy)，DreamZero 是 "video model IS the policy" (联合模型直接生成 action)。DreamZero 省去了 neural trajectory 的中间步骤，实现了 end-to-end gradient flow。

---

## 6. Limitations & Future Directions

| 限制 | 详情 | 论文提出的方向 |
|------|------|--------------|
| 推理成本 | 即使 38x 加速后仍需 2x GB200，7Hz 控制；VLA 可达 20Hz+ on consumer GPU | 等待更小的强泛化 video backbone |
| 高精度任务 | BC (Behavior Cloning, 行为克隆) 的固有局限，亚厘米精度任务 (钥匙插入等) 表现不佳 | 需要 dense 重复演示来弥补 |
| 长时序推理 | 当前 visual context 仅 6.6 秒 (33 frames)，System 1 only | System 2 planner 或扩展 context window |
| 多体预训练 | 目前 AgiBot 和 Franka 分别预训练，未做 multi-embodiment joint training | 未来方向 |
| Scaling law | 缺少系统性的 model size / data size / compute 的 scaling law 研究 | 需要类似 Chinchilla 的 WAM scaling law |
| 人类数据利用 | 跨体迁移实验仅用 12 分钟 in-lab 人类数据 | 需要大规模 egocentric human video (Ego4D 等) |
| 体设计 | 高 DoF 机器人需要更多 play data 学习 implicit IDM | 人形机器人可能因 human video prior 而更高效 |

**对 robotics FM 的启示**: DreamZero 暗示了一条与 VLA 不同的 scaling 路径 -- 不是通过收集更多 task-specific robot data，而是通过提升 video generation model 的质量来间接提升 policy 性能。如果 video backbone 遵循类似 LLM 的 scaling law，那么 WAM 的 policy 性能理论上也会随 video model scale 同步提升。

---

## 7. Paper vs Code Discrepancies

通过对比论文描述和开源代码 (`dreamzero/`)，发现以下差异:

| 维度 | 论文描述 | 代码实现 | 影响 |
|------|---------|---------|------|
| DiT Caching | "cosine similarity between successive velocities exceeds a threshold" 动态决定是否 skip | 使用固定 `dit_step_mask` (预定义哪些 step 需要完整 DiT forward)，如 8-step: `[T,T,T,F,F,F,T,F,F,F,T,F,F,T,T,T]` | 实际实现更简单、确定性更高，但失去了自适应性 |
| LoRA vs Full Fine-tuning | 论文称 "LoRA led to suboptimal results"，暗示用 full fine-tuning | 代码中 `train_architecture` 默认 "lora"，有完整的 LoRA 注入逻辑 (`add_lora_to_model`)，也支持 full | 开源代码可能因推理/分发考虑默认用 LoRA，但最佳结果来自 full fine-tuning |
| Backbone 版本 | 论文主体使用 Wan2.1-I2V-14B-480P | 代码同时支持 Wan2.1 (14B, z_dim=16) 和 Wan2.2 (5B, z_dim=48)，inference server 针对 Wan2.2-5B | 开源版本可能侧重 5B (更易部署)，14B 版本用于论文结果 |
| Denoising steps | 论文: 16 steps 默认 | 代码: `self.num_inference_steps = 16`，但 `NUM_DIT_STEPS` env var 可设 5/6/7/8，与 `dit_step_mask` 配合实际只跑 5-8 步 | DiT caching 让 effective steps 远少于 16 |
| Action chunk smoothing | 论文: Savitzky-Golay filter (window 21, poly 3) | 代码中未在开源部分明确看到此 post-processing | 可能在 eval_utils 或 control 层实现，未完全开源 |
| CFG Parallelism | "2 GPUs for conditional/unconditional parallel" | 代码有 `ip_rank`/`ip_size`/`ip_group` 字段支持推理并行 | 开源代码支持但默认单 GPU (`ip_size=1`) |
| Multi-view 处理 | "concatenate all views into a single frame" | 代码中 `collate` 函数和 data transform 支持 `num_views=3` 的视图拼接 | 一致 |

**总体评价**: 开源代码的核心架构 (AR DiT + joint video-action denoising + KV cache 闭环) 与论文一致。差异主要集中在推理优化的实现细节 (DiT caching 策略的简化) 和 backbone 版本选择上。开源侧重 5B 模型的可复现性，14B 结果需要更大的计算资源。

---

## 8. Cross-Paper Comparison

### DreamZero vs DreamGen vs DreamerV3 vs UniSim vs SONIC

| 维度 | DreamZero | DreamGen | DreamerV3 | UniSim | SONIC |
|------|-----------|----------|-----------|--------|-------|
| **核心范式** | WAM: joint video+action end-to-end | Video-as-Data-Generator: 合成 neural trajectory 再训 policy | Latent World Model: RSSM + imagination RL | Video World Simulator: universal sim engine | WBC (Whole-Body Control): tracking-based |
| **先验来源** | Video diffusion (Wan2.1 web-scale) | Video diffusion (Wan2.1 web-scale) | From-scratch latent dynamics (online RL) | Video diffusion (web-scale) | Motion tracking prior (retargeting) |
| **模型规模** | 14B | 与 backbone 一致 (Wan2.1 14B 用于 fine-tuning) | ~200M (RSSM+Actor+Critic) | 未公开 (Google Research) | ~1B (GR00T N1 backbone) |
| **Action 生成方式** | 端到端联合去噪 | 独立 IDM / LAPA 从生成视频提取 pseudo-action | Actor network in imagined latent trajectories | 不直接生成 action (需外部 controller) | Diffusion policy on proprioceptive targets |
| **训练数据** | ~500hr 真实遥操作 (多样、非重复) | 少量真实 + 大量合成 neural trajectory | Online RL interaction (可从零开始) | Real + simulated videos | Motion capture + RL + sim-to-real |
| **泛化能力** | Unseen tasks + unseen environments + cross-embodiment | Unseen behaviors + environments (via synthetic data) | 跨 150+ 任务 (固定超参)，但 per-task 训练 | Scene/action conditioned generation | Whole-body locomotion + manipulation |
| **推理方式** | 闭环实时 7Hz (KV cache + ground truth obs 替换) | 离线生成数据 -> 训练独立 policy -> policy 推理 | 在 world model 内 imagination -> policy 推理 | 离线视频生成 (非实时控制) | 直接 diffusion policy 推理 |
| **关键优势** | 端到端 gradient flow；video 质量直接决定 action 质量 | 解耦 video model 和 policy，灵活性高 | 无需 demo 数据，Data efficiency 极高 | 通用场景模拟 | 全身运动的物理可行性 |
| **关键劣势** | 推理成本极高 (2x GB200)，视频质量为瓶颈 | Two-stage pipeline 引入额外误差 | 无法利用 web-scale video prior | 不直接输出 robot action | 不涉及视觉场景泛化 |

### 关键对比分析

**DreamZero vs DreamGen (同一团队的技术演进)**:

DreamGen 是 DreamZero 的前身，两者的核心区别在于 "video model 的角色":
- DreamGen: video model 是 **数据生成器** -- 生成 neural trajectory，交给独立 policy (如 GR00T N1) 学习
- DreamZero: video model **本身就是 policy** -- 端到端联合训练 video + action，推理时直接输出 action

这一演进消除了 DreamGen 的两个瓶颈: (1) IDM/LAPA pseudo-action 的质量上限，(2) two-stage pipeline 的信息损失。但代价是推理成本大幅上升 (14B model 需实时去噪)。

**DreamZero vs DreamerV3 (两种 world model 路线的对比)**:

| 对比点 | DreamZero (pixel-space WAM) | DreamerV3 (latent-space WM) |
|--------|---------------------------|---------------------------|
| 状态表征 | Pixel-space video frames (高维、可解释) | Compact latent state (低维、抽象) |
| 先验来源 | Web-scale video pre-training (巨量 prior) | From-scratch online learning (zero prior) |
| Action 生成 | 联合去噪直接输出 (无需 planning) | Imagination + Actor-Critic (需 planning) |
| Data requirement | 需要 demonstration data (BC 范式) | 需要 reward signal (RL 范式) |
| 泛化类型 | Cross-task, cross-env, cross-embodiment | Cross-domain (固定超参，但 per-task training) |

两者代表了 robotics world model 的两条路线: DreamZero 走 "大模型 + 大数据预训练" 路线 (类似 LLM scaling)，DreamerV3 走 "轻量模型 + online 交互" 路线 (类似传统 RL)。在数据丰富的操作任务中 DreamZero 占优，在需要 exploration 的任务中 DreamerV3 可能更合适。

**DreamZero vs UniSim (video world model 的不同应用)**:

UniSim 和 DreamZero 都基于 video diffusion model，但目标不同:
- UniSim: 构建通用 **视觉模拟器** (action-conditioned video generation)，不直接生成 robot action
- DreamZero: 构建 **robot policy** (jointly generate video + action for closed-loop control)

DreamZero 可以看作 UniSim 思路的 "robotics 实例化": 把 video generation 与 action generation 融合，从 "世界模拟" 推进到 "世界控制"。

**DreamZero vs SONIC (GR00T Series 内部的互补关系)**:

SONIC 和 DreamZero 分别解决 GR00T 系列中不同层面的问题:
- SONIC: 解决 **全身运动控制** (locomotion + manipulation 的 physics-feasible tracking)
- DreamZero: 解决 **高层 task policy** (what to do + how it looks like)

两者在 GR00T 系统中有互补潜力: DreamZero 作为 high-level planner 生成 visual plan + action chunks，SONIC 作为 low-level controller 确保运动的物理可行性。论文中 DreamZero 直接输出 joint position actions，省略了底层控制器；而 SONIC 的 tracking-based WBC 可以接收更抽象的运动目标。
