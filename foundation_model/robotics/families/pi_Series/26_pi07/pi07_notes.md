# pi_0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

> arXiv:2604.15483 (2026-04-16), Physical Intelligence
> Blog: https://www.pi.website/blog/pi07
> 家族内定位: pi_Series 的第 7 代, 基础模型 = pi_0.6-MEM, 首次出现 compositional generalization

---

## 1. Core Problem

**动机**: LLM 的通用能力源自 compositional generalization — 能把训练中学到的 pattern 重新组合去解没见过的问题。但**之前的 VLA 全都做不到**:
- 不仅不会新任务, 连训练集里的所有任务都未必流畅 (需要 task-specific fine-tune)
- 增加数据规模不一定提升 (混入低质量/失败数据常常让模型变差, 因为 BC 会把不同 mode 平均掉)

**问题**: 能否只靠 prompt + 数据工程, 在**不 fine-tune** 的前提下, 让单一 VLA 模型:
1. 在训练任务上匹配/超过 RL 后训的 specialist (pi\*0.6)
2. 零样本跨 embodiment 迁移 (例: 在 UR5e 上折 T 恤, 尽管没在 UR5e 上采过 folding 数据)
3. 跟随 OOD 的语言指令 (违反数据集 bias 的指令, 如 "把垃圾放餐具桶")
4. 组合未见过的技能做新任务 (如用 air fryer 烤红薯)

---

## 2. Method Overview

核心方法: **Diverse Prompting** — 训练时给每条 episode 注入丰富的上下文, 允许模型在训练时吸收任意质量的数据, 在部署时被精确 steer。

### 2.1 Prompt 的五个模态 (context C_t)

1. **Task description ℓ_t** — 粗粒度任务描述 ("清理厨房")
2. **Subtask instruction ℓ̂_t** — 当前子任务的详细文字 ("开冰箱门"), 来自 high-level 策略 / 人类 coaching / 被 dropout
3. **Subgoal images g_t** — 未来 ~4 秒期望状态的多视角图像. 部署时由独立的 **14B BAGEL world model** 依据 ℓ̂_t 生成, 不是当前帧
4. **Episode metadata m** — 三种离散标签:
   - overall speed (episode 长度, 按 500 步 bin)
   - overall quality (1-5 分)
   - mistake flag (分段标注, 是否出错)
5. **Control mode c ∈ {joint, ee}** — joint-level 或 end-effector

训练时对前 4 个 modality 做独立 dropout:
- Subgoal images 只在 25% batch 中出现 (因为太强, 会让训练捷径化)
- Subgoal 存在时额外 30% dropout 掉 subtask
- Metadata 整体 15% 丢, 每个 component 再 5% 独立丢
- Control mode 不 dropout

### 2.2 部署时的 prompt

- Speed = 该任务下 15th percentile 的 episode 长度 (prompt for fast)
- Quality = 5, Mistake = false (prompt for good)
- Subtask 由学习到的 high-level 策略 / 人类 coaching 提供
- Subgoal 每 4 秒或子任务切换时刷新, 异步推理
- 对 metadata 施加 **CFG** (classifier-free guidance, β ∈ {1.3, 1.7, 2.2}) 放大 "高速高质量" 方向

---

## 3. Key Designs

### 3.1 用 metadata 解决 "混入失败/低质量数据反而变差" 的问题 (核心创新)

传统 VLA 训练的老难题: 数据 filtering 很费力, 而且丢弃了大量信息。但如果不 filter, 把 5 分 demo 和 2 分 demo 混训, 模型会 average 两种 mode → 输出既不好又不坏的垃圾。

**pi_0.7 的解**: 每条 episode 附带 `{speed, quality, mistake}` 标签作为 prompt 的一部分:
- 训练时: 5 分数据标 "quality=5", 2 分数据标 "quality=2", 模型学会了**条件分布** p(action | observation, quality)
- 部署时: 永远 prompt "quality=5" → 模型只输出 "5 分模式" 的动作
- 失败数据通过 `mistake=true` 标签被模型识别为负例, 但仍然参与训练, 提供状态分布

ablation (Fig 7/18): 去掉 metadata 后, 数据越大模型越差 (因为 avg 掉了); 有 metadata 后随数据量持续提升, 即使平均质量下降。

**这是"带条件的 BC"**, 不需要 advantage 函数, 但达到了 RECAP (pi\*0.6) 类似的效果 — 能吸收 RL 训练过程中 specialists 产生的全部 rollout 数据, 实质上是把 specialist **distill** 回 generalist。

### 3.2 Subgoal images 作为 visual prompt

语言指令有时说不清 "怎么做" (例: "开冰箱门" 没说抓握方式)。解法:
- 训练时: 25% 的样本注入未来 0-4 秒的真实图像作为 subgoal
- 世界模型 g_ψ: 初始化自 **BAGEL 14B** (mixture-of-transformers, 可做图像编辑/生成, web-scale 预训练), 被蒸馏为 CFM (flow matching) objective
- 部署时: 世界模型基于当前观测和 ℓ̂_t 生成 subgoal → 作为视觉 hint 喂给 pi_0.7
- 25/75 采样策略: 25% 用段末帧 (对齐世界模型训练目标), 75% 用 0-4 秒均匀采样
- 训练时额外加入世界模型生成的图像 (模拟 test-time 的图像质量), 减小 train-test mismatch

**意义**: 把 web-scale 图像生成能力通过 subgoal 这个 "接口" 注入到 VLA — 比直接 captioning 信息量更大。在 "Reverse Fridge to Microwave" 这种违反数据 bias 的任务中, 只有加上 subgoal 才能成功。

### 3.3 架构细节 (基于 pi_0.6-MEM)

- VLM backbone: **Gemma3 4B** + 400M vision encoder
- Action expert: **860M** flow matching, 50 个 action tokens, 5 个去噪步
- 总参数 ≈ 5B
- History encoder: 来自 MEM, 6 帧 (5 历史 + 当前), stride=1s, 时空压缩为单帧的 token 数
- 最多 4 个相机视角 (front + 2 wrist + optional rear) + 3 个 subgoal (无 rear)
- 所有图像 resize 448×448
- Block-causal mask: 观测 + subgoal token 双向注意 (subgoal 可看观测), 文字 token 因果注意
- Proprioception: linear projection 嵌入 (pi_0.6 用离散 token — 这里回到连续), 每个历史 state 一个 token
- History frame 30% 整体 dropout, rear view 30% dropout
- **RTC (real-time action chunking)**: 训练时模拟 0-12 步推理延迟 (50Hz 下最大 240ms), 让推理时的 chunk 边界过渡平滑

---

## 4. Experiments

### 4.1 Out-of-the-box dexterity (Fig 6-7)

在 pi\*0.6 报告的三个 RL specialist 任务上:

| 任务 | pi\*0.6 (RL specialist) | pi_0.7 (generalist) |
| --- | --- | --- |
| Espresso | 高 | 匹配 |
| Box building | 高 | **throughput 超过 specialist** |
| Laundry (mixed) | 高 | **throughput 超过 specialist** |

pi_0.7 (no metadata) 和 pi_0.7 (no eval data) 显著弱于 pi_0.7 — 证明 metadata + 吸收 autonomous 评估数据 (包括 pi\*0.6 RL 过程中的 rollout) 都是关键。

在 MEM 论文的记忆任务上, 没有 fine-tune 的 pi_0.7 匹配或超过 MEM 的 fine-tuned 特化版。

### 4.2 Instruction following (Fig 9-11)

- 14 个 instruction following 场景, 4 个未见厨房 + 2 个未见卧室, 每个任务 3-6 条 open-ended 指令序列
- pi_0.7 显著超过 pi_0.5/pi_0.6
- 复杂 referential instructions ("拿我喝汤用的物品") pi_0.7 优势明显, 加 subgoal (GC) 更强
- 数据 bias 挑战 ("把垃圾放餐具桶" 违反训练数据): 只有 pi_0.7 能做到, "Reverse Fridge to Microwave" 必须配 subgoal

### 4.3 Cross-embodiment transfer (Fig 12-13, 核心亮点)

从小的 static bimanual robot 训练 → 在 bimanual UR5e 上测试 (形态差异极大: 更大, 更重, 更难精细抓取)。

- **Shirt folding on UR5e** (无任何 UR5e folding 数据):
  - pi_0.7 task progress **85.6%**, 成功率 **80%**
  - 10 名 top 2% 人类遥操作者 (平均 375 小时经验) 首次尝试: 90.9% / 80.6%
  - **pi_0.7 匹配零样本人类专家水平**
- 模型会**自动改变策略**: 源机器人倾斜末端先压再抬 → UR5e 用竖直抓取 (更适合 UR5e 的运动学)
- 加 subgoal (GC) 进一步提升

**实际价值**: 用便宜好遥操作的轻量机器人采集数据 → 迁移到贵重工业臂, 绕开工业臂数据采集的高成本。

### 4.4 Compositional generalization (Fig 14-17)

- **短时任务零样本** (Fig 17): 擦耳机、转风扇/齿轮组、舀米进电饭煲、按 French press — 训练数据里都**没有**这些任务
- **长时任务靠 coaching** (Fig 15-16):
  - 未见任务: Loading/Unloading Air Fryer, Toasting a Bagel
  - 人口头一步步指导 → pi_0.7 能做, 其他 VLA 做不了 (缺 language following)
  - Coaching 产生的 (subtask) 序列可以反过来训一个 high-level 策略 → pi_0.7 (autonomous) 几乎匹配 live-coaching 版本
  - → 新范式: **用"说"来教, 不用 teleoperation 采 action 数据**

### 4.5 Scaling ablation (Fig 18)

- Laundry 数据按 quality 分 4 桶 (top 30/50/80/100%), 训 8 个模型
  - 无 metadata: 更大数据集反而更差 (mixing hurts)
  - 有 metadata: 越多越好, 即使平均质量下降
- 去掉最 diverse 的 20% 数据 vs 去掉随机 20%: 前者显著变差 → 多样性比数据量更重要

---

## 5. Related Work Analysis

pi_0.7 在家族里的位置:

| 模型 | 核心能力 | pi_0.7 继承了什么 |
| --- | --- | --- |
| pi_0 (24) | VLM + Flow Matching action expert | 架构范式 |
| FAST (25) | DCT action tokenizer | (未直接使用, pi_0.7 继续用 flow matching) |
| Hi Robot (25) | 层级 VLA | 高层策略做 coaching → 低层 pi_0.7 执行 |
| pi_0.5 (25) | 异构数据 + Knowledge Insulation + subtask prompt | subtask prompt, 多源数据共训 |
| pi\*0.6 (25) | RECAP 离线 RL | **蒸馏 pi\*0.6 rollout 作为训练数据** (代替显式 RL) |
| pi_0.6-MEM (26) | 多尺度记忆 | backbone 直接继承, history encoder 架构 |

**相对外部工作**:
- 与 LLM 的 "prompt expansion" / SuSIE 的 subgoal image 思路一致, 但 pi_0.7 的贡献是**证明这在 robotics 上能 work 出 compositional generalization**
- 不是新架构, 是**数据工程 + prompt 工程**
- 与 NVIDIA GR00T 对比: GR00T 走分层 + 仿真; pi 走 VLA 纵深 + 真机数据多样性

---

## 6. Limitations & Future Directions

论文显式写的:
1. **OOD 成功率仍明显低于 in-distribution**: in-distribution >90%, 未见任务或未见 task-robot 组合 60-80%
2. **"Unseen" 的定义含糊**: 数据集太大, 无法判断某个技能是否在别处以不同 label 出现过 → 这其实是 compositional generalization 的本质, 和 LLM 面临的问题同构
3. **未来方向**: 利用 pi_0.7 的高 steerability 做 test-time 自适应 — coaching 或在线 RL

潜在未讨论的问题:
- 依赖云端部署 (real-time action chunking 需要走 API), 对无网络场景不适用
- 世界模型 14B + VLA 5B = 19B 总参数, 部署成本高
- Subgoal 质量对世界模型的视觉先验高度依赖, OOD 物体可能生成错误 subgoal

---

## 7. Paper vs Code Discrepancies

openpi 仓库目前 (2026-04) 尚未发布 pi_0.7 权重或推理代码。可验证细节均来自论文本身和博客。Algorithm 1 (runtime prompting) 在论文给出伪代码, 未提供配套实现。

世界模型 g_ψ (基于 BAGEL 14B 的 SuSIE-style 微调) 也未开源。

---

## 8. Cross-Paper Comparison

### pi_0.7 vs pi\*0.6 (替代关系?)

| 维度 | pi\*0.6 (RECAP) | pi_0.7 (metadata) |
| --- | --- | --- |
| 利用失败数据的机制 | 训练 Value → 估 Advantage → 作为 ± token | 人工标注 quality/mistake → 作为离散 prompt |
| 算法难度 | 需要 offline RL 训 critic | 只需分桶标注 |
| 是否需要成功/失败标签 | 是 (reward) | 是 (quality/mistake) |
| 吸收的数据 | 演示 + 自主经验 + 人类纠正 | 同上 + pi\*0.6 RL rollout |
| 部署时 prompt | 固定 "+" token | 固定 "quality=5, mistake=false" + CFG |
| 泛化能力 | 任务特化 | 通用, 单模型多任务 |

**结论**: pi_0.7 用更简单的 metadata 机制取得了 RECAP 类似的数据效率, 而且保留了单模型多任务能力。pi\*0.6 可以看作 pi_0.7 训练数据的 teacher。

### pi_0.7 vs GR00T N1.6 (同代对比)

- GR00T: 分层 (VLM + DiT + SONIC), 仿真 + real, 针对人形全身
- pi_0.7: 单一 VLA, 主要真机 + 人类 video, 针对桌面/移动 manipulator
- GR00T 的 DreamZero / WAM 走 world-action model; pi_0.7 把 world model 用作 prompt 生成器而不是 policy 本身 — 哲学不同

### 对 RL 从业者的启示 (PPO sim2real 视角)

1. **metadata-conditioning 是 advantage-conditioning 的轻量版**: 如果你有离散的 quality/speed/mistake 标签, 可以在 BC 基础上直接加 prompt, 不需要训 critic
2. **CFG 可以放大高质量方向**: 类似 reward weighted regression 的推理时增强
3. **失败轨迹不扔**: 用 mistake=true 标签吸收进训练, 同时提供 OOD 状态覆盖

---

## 附: 核心数字速查

| 项 | 值 |
| --- | --- |
| 总参数 | ~5B (VLM 4B + action expert 860M) |
| VLM backbone | Gemma3 4B |
| Vision encoder | 400M (Gemma3) |
| 世界模型 | BAGEL 14B (独立) |
| 图像分辨率 | 448x448 |
| History 帧数 | 6, stride 1s |
| 相机数 | 最多 4 (front + 2 wrist + rear) |
| Subgoal 数 | 最多 3 (无 rear) |
| Action chunk | 50 步, 5 次去噪 |
| 执行 horizon | 15 或 25 步 (chunk 中的前几步) |
| RTC 训练延迟模拟 | 0-12 步 (50Hz 下 ≤240ms) |
| Subgoal 刷新 | 4 秒或子任务切换 |
| CFG 权重 β | 1.3 / 1.7 / 2.2 |
| 控制模式 | joint (主用) / ee |
| Speed prompt | 任务的 15th percentile episode 长度 |
| Quality prompt | 5 (最高) |
| Mistake prompt | false |
| 跨 embodiment zero-shot shirt fold | 85.6% task progress / 80% 成功 (匹配人类专家 90.9% / 80.6%) |
