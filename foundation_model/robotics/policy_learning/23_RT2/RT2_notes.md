# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

Paper: Brohan et al., Google DeepMind, 2023
Website: robotics-transformer2.github.io

---

## 1. Core Problem

大规模 VLM (如 PaLI-X, PaLM-E) 在 web data 上学到了丰富的视觉语义知识 (物体识别、空间推理、常识理解)，但这些能力无法直接用于 robot control, 因为:
- VLM 输出自然语言，而 robot 需要连续的 low-level actions (如末端位移、旋转)
- Robot 数据量远小于 web data (几十万 vs 几十亿)
- 此前 VLM 在 robotics 中只用于 high-level planning (如 SayCan), 低层控制仍然依赖单独训练的 policy

RT-2 的核心问题: 能否直接将 VLM fine-tune 成一个 end-to-end 的 closed-loop robot controller, 同时保留 web data 中学到的语义知识?

---

## 2. Method Overview

### 2.1 Architecture: VLM -> VLA

RT-2 提出 Vision-Language-Action (VLA) model -- 概念上极其简单:
1. 取一个预训练好的 VLM
2. 将 robot actions tokenize 成文本 token
3. 在 robot data 上 co-fine-tune, 让 VLM 学会输出 action tokens

两个 VLM backbone:
- **RT-2-PaLI-X-55B**: ViT-22B (vision) + 32B encoder-decoder (language), 总计 55B params
- **RT-2-PaLM-E-12B**: ViT + PaLM 12B (decoder-only)

### 2.2 Action Tokenization

关键设计 -- 将连续 action 离散化为整数 token:
- Action space: 7-DoF (6-DoF end-effector pose: dx, dy, dz, drx, dry, drz + gripper extension) + termination flag
- 每个连续维度均匀离散化为 **256 bins**
- Action 表示为 8 个整数的字符串: `"terminate dpos_x dpos_y dpos_z drot_x drot_y drot_z gripper"`
- 示例: `"1 128 91 241 5 101 127"`

Token mapping 策略因 VLM 而异:
- **PaLI-X**: 0-999 的整数各有对应 token, 直接映射
- **PaLM-E**: 没有这样的便利, 用 256 个最低频的 token 覆盖 action bins (symbol tuning)

### 2.3 Training: Co-Fine-Tuning

训练数据混合:
- 原始 web-scale VLM 训练数据 (VQA, captioning 等)
- Robot demonstration 数据 (RT-1 数据集: 13 robots, 17 months, ~130k episodes)

**Co-fine-tuning** 而非 naive fine-tuning:
- Robot data 与 web data 混合训练
- RT-2-PaLI-X: robot data 约占 50% of training mixture
- RT-2-PaLM-E: robot data 约占 66%
- 输入格式统一为 VQA: `"Q: what action should the robot take to [task]? A:"`

### 2.4 Inference

- 输出约束: inference 时 constrain decoding 只 sample valid action tokens (当检测到 robot action task 时)
- 部署方式: multi-TPU cloud service, 通过网络查询
- 55B model: 1-3 Hz; 5B model: ~5 Hz

---

## 3. Key Designs

### 3.1 Actions as Language Tokens (统一输出空间)

这是 RT-2 最核心的 insight: 不添加任何新参数或 action-specific head, 直接复用 VLM 的 token vocabulary 来表示 robot actions。

为什么这样做:
- 所有 model weights 在 language 和 action 任务间完全共享
- Web knowledge 可以直接影响 action 生成 (因为是同一个 decoder)
- 无需设计新的 policy architecture

对比: 传统做法是在 VLM 后面接一个 action head (如 RT-1 的 TokenLearner + action decoder)。RT-2 证明这是不必要的 -- VLM 本身就可以输出 actions。

### 3.2 Co-Fine-Tuning (防止灾难性遗忘)

Ablation 实验 (Figure 6b) 明确证明: co-fine-tuning > fine-tuning > training from scratch

直觉: 如果只在 robot data 上 fine-tune, VLM 在 web data 上学到的语义知识会被遗忘 (catastrophic forgetting)。混合训练让模型持续暴露于 web scale 的视觉概念，从而在 unseen objects/environments 上泛化更好。

量化结果:
- Fine-tuned 5B: unseen average ~40%
- Co-fine-tuned 5B: unseen average ~62%
- Co-fine-tuned 55B: unseen average ~62%
- 甚至 5B co-fine-tuned 和 55B co-fine-tuned 在 generalization 上相近, 说明 co-fine-tuning strategy 比单纯增大模型更重要

### 3.3 Emergent Capabilities (涌现能力)

RT-2 展示了 VLA 模型独有的、robot data 中从未出现过的能力:

- **Symbol Understanding**: "move apple to 3", "push coke can on top of heart" -- robot data 中没有数字/符号概念
- **Reasoning**: "move the apple to cup with same color", 多语言理解 ("mueve la manzana al vaso verde")
- **Human Recognition**: "move the coke can to the person with glasses"

定量结果 (Figure 6a): RT-2-PaLI-X-55B 在 emergent skills 上的平均成功率是 RT-1 (next best baseline) 的 3x 以上。

---

## 4. Experiments

### 4.1 Setup
- Robot: 7-DoF mobile manipulator
- 评估: ~6000 real robot trials (不是 simulation)
- Baselines: RT-1 (35M), VC-1 (ViT-L), R3M (ResNet-50), MOO

### 4.2 Seen Tasks
- RT-2 与 RT-1 性能相近 (~90%+), 不损害 in-distribution 表现

### 4.3 Generalization (核心指标)
- **Unseen Objects**: RT-2 ~2x improvement over RT-1 and MOO
- **Unseen Backgrounds**: RT-2 ~2x improvement
- **Unseen Environments**: RT-2 ~2x improvement
- **Unseen Average**: RT-2 ~6x improvement over VC-1 and R3M

### 4.4 Language-Table Simulation
- RT-2-PaLI-3B: 90 +/- 10 vs RT-1: 74 +/- 13, LAVA: 77 +/- 4
- 说明 VLM pre-training 即使在不同 robot (simulation) 上也有效

### 4.5 Chain-of-Thought Reasoning
- 在 PaLM-E 上 fine-tune 几百步, 加入 "Plan" step: "Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255."
- 能回答更复杂的语义指令 (如 "what object to use as improvised hammer" -> picks rock)

### 4.6 Limitations
- 不能学到新的 physical motions (只能重新组合已有的 skills)
- Inference 速度慢 (55B model 只有 1-3 Hz, 对高频控制场景不够)
- 依赖少数可用的大型 VLM

---

## 5. Impact on Robotics (对 pi_0 和 GR00T N1 的影响)

RT-2 是 VLA 范式的奠基工作, 直接影响了后续所有 VLA 模型:

### 5.1 对 pi_0 的影响
- **Action tokenization 的演进**: RT-2 用离散 bins, pi_0 改用 flow matching 输出连续 actions。pi_0 的作者 (部分来自同一团队) 认为离散化 256 bins 是信息瓶颈, 尤其对 dexterous manipulation, 于是引入 continuous action space via diffusion/flow
- **Co-fine-tuning -> pre-train + fine-tune**: pi_0 继承了 "在 web VLM 上 fine-tune" 的思路, 但进一步区分了 pre-train stage (多 robot 数据) 和 fine-tune stage (target task)
- **Architecture**: pi_0 用 PaliGemma (3B) 而非 55B model, 加上 action expert (diffusion head), 比 RT-2 更轻量

### 5.2 对 GR00T N1 的影响
- **Dual-system architecture**: GR00T N1 将 VLM (slow, semantic) 和 diffusion action head (fast, motor) 分离, 部分是因为 RT-2 暴露了 "VLM 单独做控制太慢" 的问题
- **Action representation**: GR00T N1 也采用 continuous actions via diffusion, 而非 RT-2 的离散 tokens
- **Co-fine-tuning 的继承**: GR00T N1 的 VLM backbone 同样在 web data + robot data 上联合训练

### 5.3 RT-2 确立的范式
1. **VLM 作为 robot brain**: 用大规模预训练的视觉语言理解能力驱动 robot control
2. **Unified input/output**: image + language instruction -> action, 不需要单独的 perception/planning 模块
3. **Emergent transfer**: web knowledge 可以 transfer 到 robot -- 这不是 obvious 的, RT-2 首次在 real robot 上大规模验证
4. **Scale matters**: 更大的 VLM -> 更好的 generalization (5B vs 55B in emergent skills)

### 5.4 RT-2 的局限催生的改进
| RT-2 的局限 | 后续改进 | 代表工作 |
|-------------|----------|----------|
| 离散 action tokens (256 bins) | Continuous actions via diffusion/flow | pi_0, Diffusion Policy |
| 55B model, 1-3 Hz | 更小的 VLM + action expert 分离 | pi_0 (3B), GR00T N1 |
| 单一 robot morphology | 多 robot generalization | pi_0 (cross-embodiment) |
| 无 action chunking | Multi-step action prediction | pi_0 (action chunks) |
| 单帧输入 | 多帧/历史 context | GR00T N1 (temporal context) |

---

## Summary for Robotics Researchers

RT-2 最关键的贡献是验证了一个大胆的假设: 大规模 VLM 可以直接 fine-tune 为 end-to-end robot controller, 且 web knowledge 确实能 transfer 到 robot control (emergent capabilities)。它确立了 VLA 范式, 但也暴露了离散 action tokenization 和推理速度的瓶颈。后续的 pi_0 和 GR00T N1 正是在保留 "VLM backbone + web co-training" 核心思路的同时, 用 continuous diffusion/flow actions 和更高效的架构解决了 RT-2 的不足。
