# PI (Physical Intelligence) Series -- 从 VLA 到自主改进的机器人基础模型

> **目的**: 理解 PI 做通用机器人 VLA 的完整思路——每一步解决了什么问题，为什么要那样做，以及对你的启示。

---

## 1. PI 的战略定位

PI 是一家纯模型公司 (不造硬件不做仿真), 目标是做**机器人的 foundation model**, 让任何机器人硬件都能用 PI 的模型开箱即用。

```
PI 的产品逻辑:
  1. 训练一个通用 VLA (视觉+语言→动作)
  2. 开源推理代码 (openpi), 让社区在自己的机器人上 fine-tune
  3. 通过 API/服务收费 (商业模式)

对比 NVIDIA:
  NVIDIA: 造平台 (仿真+训练+部署+芯片), 模型只是平台的一部分
  PI: 只做模型, 极致纵深 — 一条 VLA 路线走到底
```

**创始人阵容** (理解论文思路的关键):
- **Sergey Levine** (Berkeley): RL for Robotics 开创者, 定义了 offline RL/BC 的理论框架 → 解释了为什么 PI 后期走 RL 路线 (pi\*0.6)
- **Chelsea Finn** (Stanford): MAML 元学习发明者 → 解释了为什么 PI 强调 few-shot 泛化
- **Karol Hausman** (前 DeepMind SayCan): LLM-for-robot planning 先驱 → 解释了为什么 PI 用 VLM 做高层理解

PI 的论文本质上是这三个人的研究方向的工程化整合。

---

## 2. 完整演进: 每一步解决了什么问题

### 从 RT-2 到 pi_0: 为什么需要重新设计架构

RT-2 验证了"VLM 的互联网 pattern 可以迁移到 robot"这个核心假设, 但架构有三个根本问题:

```
RT-2 的问题 (架构层面):

1. 动作 = 语言 token → 离散化精度差 (256 bins ≈ 2.8° 误差)
   原因: VLM 输出头是 softmax 分类器, 只能从词表中选 token
   → 不能输出连续浮点数, 被迫把动作编码为离散 bin index

2. 无法冻结 VLM → 灾难性遗忘风险
   原因: 动作和语言共用同一个输出头
   → 冻结 VLM = 冻结输出头 = 不能输出动作
   → 只能整体 fine-tune, 可能破坏 VLM 的视觉语言 pattern

3. 55B 太大 → 推理 1-3Hz → 不能实时控制
   原因: 整个 VLM 参与每一步推理

pi_0 的解法: 加一个独立的 Action Expert

  RT-2:  VLM (55B) → 语言输出头 → 离散动作 token
  pi_0:  VLM (3B, 可冻结) → 表征 → Action Expert (300M) → Flow Matching → 连续动作

  一个改动解决三个问题:
    独立输出头 → 可以输出连续值 (精度解决)
    VLM 和 Action Expert 参数分离 → 可以冻结 VLM (遗忘解决)
    VLM 缩小到 3B + Action Expert 只有 300M → 14Hz (速度解决)
```

### Phase 0: 数据基础 -- DROID (2024)

**DROID (Distributed Robot Interaction Dataset)**: PI 训练数据的核心来源之一

```
规模: 76k 轨迹, 350 小时, 18 机构 / 50 采集者 / 564 场景 / 52 栋建筑 / 3 大洲
硬件: Franka Panda 7DoF + Robotiq 2F-85 夹爪, Meta Quest 2 遥操作, 15Hz
相机: 3 路 Zed 立体相机 (2 外置可调 + 1 腕部), 均有 depth

采集方法论 (可复用的最佳实践):
  - 随机任务抽样: GUI 从预设列表随机选, 防止偏向简单任务
  - 场景增强提示: 定期挪相机/改灯光/增减物品, 保证视觉多样性
  - 每场景 ~100 轨迹 (~20 分钟) 后换场景
  - 失败轨迹也保留 (~16k 条, 不计入 76k)

关键结论: 场景多样性 > 数据量
  7k 轨迹 + 多样场景 > 7k 轨迹 + 仅 20 个场景 (同等数据量, OOD 性能更高)
  Co-training DROID 平均提升 20%+ 成功率 (OOD 场景提升尤为显著)
```

### Phase 1: 建立基础 -- pi_0 (2024.10)

**要解决的问题**: 在 RT-2 验证的"VLM→robot"假设基础上, 能不能解决架构问题, 做一个真正可部署的通用 VLA?

**解法**: VLM (PaliGemma, 3B) + Flow Matching Action Expert

```
架构:
  Transformer 每层有两组独立的 QKV + FFN:
    VLM Expert:    处理 image + language + proprioception tokens
    Action Expert: 处理 action tokens (噪声 → 去噪 → 连续动作)
  两组 Expert 通过 shared attention 交互 (QKV 各自独立, 但 attention score 跨组计算)

训练 (只有一个阶段, 从一开始就有 Action Expert):
  加载 PaliGemma 权重 (VLM) + 随机初始化 Action Expert
  → robot 数据 (Open X + 自采 10k 小时)
  → 只有 flow matching loss (在 action tokens 上)
  → VLM 和 Action Expert 一起训练
  → 不能吃 web 数据 (flow matching loss 处理不了没有 action 的数据)

  pre-training:  大量低质量 robot 数据 → 训 base model
  post-training: 少量高质量数据 → fine-tune 到具体任务 (折衣服等)
  → 两个阶段都是 flow matching loss, 网络结构不变, 只是数据质量不同

推理: 10 步 flow matching 去噪 → 50 步 action chunk, ~14Hz
```

**pi_0 的两个核心局限**:
1. **不能输出文字** → 不能规划长任务 ("清理厨房"不知道先做什么)
2. **不能吃 web 数据** → VLM 在 robot 训练过程中遗忘互联网知识 → 泛化差

**Takeaway 1**: **VLM + Flow Matching 是当前 VLA 的最佳架构组合, 但只有 flow matching loss 限制了数据多样性。**

### Phase 2: 提速 -- FAST (2025.01)

**要解决的问题**: Flow Matching 需要多步迭代推理, 对高频灵巧任务太慢。能不能用 LLM 的 autoregressive 方式一次前向就出动作?

**难点**: 动作是连续值, LLM 只能生成离散 token。之前的方案 (RT-2 的 256-bin) 精度太低。

**解法**: DCT (Discrete Cosine Transform, 离散余弦变换) 做动作压缩

```
传统离散化为什么失败 (论文 didactic experiment):
  autoregressive 预测的学习信号 ∝ 每个 token 的边际信息量 (marginal information)
  高频平滑信号 (如 15Hz 的关节角序列): 相邻 token 几乎相同 → 边际信息趋近于零
  → 模型退化为"复制上一个 token" → MSE 随采样频率上升而急剧恶化
  → 这就是为什么 OpenVLA 在 BridgeV2 (5Hz) 能用, 但在 DROID (15Hz) 上失败

FAST 完整 pipeline:
  1. Quantile normalization: 用 1st/99th percentile 映射到 [-1,1] (对 outlier robust)
  2. DCT 变换: 每个 action dimension 独立做, 能量集中到低频系数
  3. Scale-and-round: 缩放系数 gamma=10 (压缩/精度 tradeoff, 几乎不需调)
  4. Column-first flattening: 低频在前 → autoregressive 先预测整体形状再细节
  5. BPE 压缩: vocab=1024, 把稀疏系数矩阵压成紧凑 token

  结果: 不管原始频率是 15Hz 还是 50Hz, 统一压缩为 ~30 token/arm/chunk
  对比 naive tokenization: 50Hz bimanual 数据产生数百个 token

FAST+ (universal tokenizer):
  在 1M 条真机轨迹上训练, 覆盖 single-arm/bi-manual/mobile
  已发布 HuggingFace AutoProcessor (3 行代码即用)
  对未见过的机器人仍能 2x+ 压缩, 与 per-dataset tokenizer 性能持平
```

**结果与 tradeoff**:
- pi_0-FAST 匹配 diffusion pi_0 性能 (含最难的 laundry folding), 训练时间减少 **5x**
- 但推理时 FAST **比 diffusion 慢** (autoregressive 逐 token vs diffusion 并行去噪)
- FAST 在 language following 上更好 (autoregressive 天然有 language grounding)
- **首次实现 DROID 上 zero-shot 跨场景泛化** (此前所有方法都需要 fine-tune)

**Takeaway 2**: **动作压缩的关键不是量化精度, 而是选对变换域。** 时域量化 (RT-2) 丢信息, 频域量化 (FAST/DCT) 利用了动作的物理先验 (平滑性)。这个思路可以迁移到任何需要 tokenize 连续信号的场景。

### Phase 3: 层级化 -- Hi Robot (2025.02)

**要解决的问题**: pi_0 是扁平的 "观察→动作", 无法处理复杂多步指令 (如 "先清理桌子, 然后把碗放进洗碗机")。

**解法**: 两层 VLA

```
Layer 2 (高层): VLM 推理
  输入: 图像 + 复杂指令
  输出: 子目标序列 + 进度判断
  "清理桌子" → ["拿起碗", "放进洗碗机", "拿起杯子", ...]

Layer 1 (低层): pi_0 执行
  输入: 图像 + 当前子目标
  输出: 动作
  "拿起碗" → 具体关节运动
```

**关键设计: 反向合成训练数据**
```
问题: 复杂指令数据采集困难 (人类很少自然地对机器人说多步指令)
解法: 反向合成 — 给 VLM 看 (图像 + atomic skill label), 让它反向生成用户指令
  输入: 桌面图像 + "pick up lettuce"
  VLM 合成: "Can you make me a vegetarian sandwich?" / "帮我加点生菜"
  同时合成机器人的口头回复
  → 用现有 demo 数据自动扩展出复杂指令, 不需要额外人工标注

结果:
  没有合成数据 → 高层策略忽略用户约束 ("不要番茄" 却仍然拿番茄)
  有合成数据 → 学会 compositional language understanding
  Fine-tuned 小模型 (PaliGemma-3B) > GPT-4o 做高层, 因为 GPT-4o 缺乏物理 grounding
```

**和 GR00T 分层的区别**:
- GR00T: VLM (语义) + DiT (动作) + SONIC (关节) → 三层, 按**频率**分
- Hi Robot: VLM (规划) + VLA (执行) → 两层, 按**抽象层次**分
- GR00T 的分层是工程驱动 (频率匹配), Hi Robot 的分层是认知驱动 (任务分解)

**Takeaway 3**: **复杂任务需要层级分解, 但分层的维度有两种选择: 频率 (GR00T) vs 抽象层次 (Hi Robot)。** 两者不矛盾, 可以组合。反向合成数据的方法论可直接复用到任何需要复杂指令数据的场景。

### Phase 4: 开放世界泛化 -- pi_0.5 (2025.04)

**要解决的问题**: pi_0 只能在训练时见过的场景中工作。能否让机器人在从未见过的家庭中清理厨房?

**pi_0.5 不只是"Knowledge Insulation"——是一个系统级的泛化方案, 有五个层面的改进:**

**改进 1: 异构数据共训练 (核心创新)**

```
97.6% 的训练数据不是目标机器人 (移动操作臂) 的!

数据来源:
  其他机器人的操作数据 (非移动底盘的桌面臂)
  Web 数据 (图像描述/问答/物体定位 — VLM 任务)
  高层语义预测 ("下一步该做什么" — 从图像预测子任务)
  人类口头指令 (supervisor 走在机器人旁边口头说 "现在拿那个碗")
  目标机器人数据 (移动操作臂, 约 400 小时) — 只占 2.4%

→ 不是简单混入保护性数据
→ 是一套完整的多源知识迁移框架: 不同数据教不同能力
  其他机器人: 教抓取/放置的通用 pattern
  Web 数据: 保持 VLM 的物体识别/场景理解能力
  语义预测: 教任务规划能力
  口头指令: 教从人类语言中理解意图 (比遥操作更容易采集)
```

**改进 2: 两阶段训练 — 解决 pi_0 "flow matching 不能吃 web 数据" 的核心限制**

```
pi_0 的问题:
  只有 flow matching loss → web 数据没有 action → 喂不进去
  → VLM 在 robot 训练中遗忘互联网知识 → 泛化差

pi_0.5 的解法: 两阶段, 先离散后连续

Stage 1 Pre-train (280k steps, α=0):
  没有 Action Expert, 没有 flow matching
  所有数据 (含 action) 离散化为 token → 全部用 cross-entropy loss
  → action 被 FAST 编码成离散 token, 和文字一样用 Language head 输出
  → web 数据, 语义预测, robot 数据全部统一为 next-token prediction
  → 一种 loss, 一种流程, 和训 GPT 完全一样
  → 这个阶段训的是 "会说话+会规划+会预测离散动作的 VLM"

Stage 2 Post-train (80k steps, α=10):
  新建 Action Expert (随机初始化, 之前不存在)
  同时训两种 loss:
    cross-entropy: 保持文字输出 (子任务预测)
    flow matching: 训 Action Expert 输出连续动作
  → VLM 的知识通过 attention 传给新建的 Action Expert
  → pre-train 的离散 action 能力被 Action Expert 替代 (不再使用)
  → pre-train 的真正价值: VLM backbone 吃了 web 数据不会遗忘

pi_0:   从头就有 Action Expert + flow matching → 只能吃 robot 数据
pi_0.5: pre-train 没有 Action Expert → 能吃一切 → post-train 再加 Action Expert
```

**改进 3: 同一模型做层级推理 (高层规划 + 低层执行)**

```
每一步推理, 同一个模型跑两次:
  第 1 次: π(ℓ̂|o_t, ℓ) → 预测语义子任务 "拿起盘子"
  第 2 次: π(a|o_t, ℓ̂) → 根据子任务输出动作

和 Hi Robot 的区别:
  Hi Robot: 两个独立模型 (VLM 规划 + VLA 执行)
  pi_0.5:  同一个模型, 两次推理 → 类似 chain-of-thought (先想再做)
  → 更简洁, 共享参数, 高层和低层的知识互相增强
```

**改进 4: Knowledge Insulation (防灾难性遗忘)**

```
选择性冻结:
  VLM 底层 (低级视觉特征): 冻结 → 保护 "什么是杯子"
  VLM 顶层 (任务推理): 解冻 → 适应 "看到杯子该怎么抓"

+ 异构数据中的 web 数据持续保持 VLM 能力
→ 双重保护: 参数冻结 + 数据混合
```

**改进 5: 人类口头监督 (全新的数据采集方式)**

```
不是遥操作 (人操控机器人关节)
而是: 人走在机器人旁边, 口头说 "现在去拿那个碗"
→ 机器人自己执行 → 录下来
→ 比遥操作容易得多 (不需要操控设备, 任何人都能做)
→ 采集的是 (语言指令, 动作) 对, 训练语言→动作的映射
```

**结果**: 在从未见过的家庭中清理厨房/卧室, 10-15 分钟长任务

**Takeaway 4**: **pi_0.5 的核心教训是: 泛化不是靠一个技巧, 而是靠系统性的数据工程。** 97.6% 非目标数据 + 异构共训练 + 层级推理 + 知识隔离 + 口头监督, 五个层面一起才实现了开放世界泛化。单独任何一个都不够。

### Phase 5: 从模仿到自我改进 -- pi\*0.6 (2025.11)

**要解决的问题**: pi_0 和 pi_0.5 的训练全部是 BC (模仿学习) → 策略上限 = 演示者水平 → 失败轨迹被浪费。能不能从失败中学, 超越演示者?

**基础模型**: pi_0.6 (pi_0.5 的升级版, 更大 backbone + 更多条件输入, 未单独发论文)

**方法: RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)**

```
和 PPO 的关键区别: 不用 policy gradient, 用 advantage conditioning

Step 1: 收集三种数据
  演示数据:   人类遥操作的成功轨迹
  自主经验:   机器人自己跑, 成功或失败都录下来
  人类纠正:   机器人跑的时候人发现要犯错 → 介入纠正 (human-gated DAgger)

Step 2: 训 Value Function
  输入: 任意 (state, image)
  输出: "从这里开始, 最终成功概率多大" (标量)
  监督信号: sparse — 成功轨迹终点=1, 失败终点=0, 中间用 TD 回传
  → 和你 PPO 的 Critic 一样, 只是用离线数据训

Step 3: 估 Advantage
  A_t ≈ V(s_{t+1}) - V(s_t)
  → "这一步之后情况变好了 (A>0) 还是变差了 (A<0)"

Step 4: Advantage Conditioning (核心创新, 不是 policy gradient)
  把 advantage 二值化: A>0 → "+" token, A<0 → "-" token
  作为额外条件输入给 VLA:
    训练时: "+" 数据正常学, "-" 数据反向学 (classifier-free guidance)
    部署时: 永远输入 "+" → 模型只输出"好的"动作
  
  → loss 还是 flow matching MSE → 和 BC 一样稳定
  → 不需要算 log π(a|s) (flow matching 算不出来)
  → 不需要 on-policy 数据 (用已有离线数据)
  → 本质是"带权重的 BC": 好数据权重大, 坏数据权重小

为什么不直接用 PPO:
  论文原文: "approaches that are difficult to extend to real-world RL 
  in an efficient and scalable fashion"
  → PPO 需要 on-policy → 真机上太贵
  → PPO 需要 log π → flow matching 算不出来
  → PPO 在 3B+ 模型上梯度方差大 → 不稳定

可以迭代:
  部署 → 收集经验 → 训 value → 估 advantage → 更新 policy → 再部署
  → 每轮都变好
```

**结果**: 折衣服/组装箱子/做咖啡, 失败率降低 2x+, 吞吐量翻倍, 可连续运行 13 小时

**Takeaway 5**: **RECAP 是把 RL 思想嫁接到 VLA 的最实际方案:**
- 不需要 reward engineering (只需标注成功/失败)
- 不需要仿真器 (用真机数据)
- 不需要 on-policy (离线, 所有旧数据都能用)
- 你的 PPO 经验直接适用: GAE 算 advantage 的思想完全一致, 只是用法不同 (PPO 乘到梯度里, RECAP 作为条件输入)

### Phase 6: 前沿 -- MEM + RLT (2026.03)

**MEM (Multi-scale Embodied Memory)**: 解决长时任务

**基础模型升级**: pi_0.6, backbone 从 PaliGemma-3B → **Gemma3-4B**, Action Expert 从 300M → **860M**

```
问题: VLA 的 context window 只有几秒到几十秒
     "打扫整个房间" 需要 15+ 分钟, 中间有几十个子任务

短期记忆 (Video Encoder):
  核心: 不引入新参数 — 仅修改 ViT 的 attention pattern
  每 4 层做一次 causal temporal attention (同一 patch 跨时间步)
  其余层保持标准 spatial attention
  只传当前帧 token 给 VLA backbone (丢弃历史帧 token → 压缩)
  可从任何预训练 VLM 的 ViT 权重直接初始化
  Pre-train: 6 帧 (5 历史+当前), stride=1s; Post-train 可扩展到 18 帧/54s

长期记忆 (语言压缩):
  用 LLM 生成训练数据: 给定子任务序列+成功/失败标记 → 生成压缩摘要
  关键: LLM 被指示主动丢弃不再相关的信息 (如具体颜色→"三个碗")
  
  为什么 naive 拼接历史子任务效果差:
    训练数据中子任务只出现一次 (人类 demo 近似最优)
    推理时同一子任务可能反复失败重试
    → train-inference distribution shift → 压缩摘要自然处理了这个问题

附加能力 — In-context adaptation:
  短期视觉记忆使策略能看到自己的失败尝试并即时调整
  例: 抓筷子失败 → 调整高度; 拉门方向错 → 换方向
  训练: 收集人类 correction 数据 (失败后人介入示范), 失败放入短期记忆
  无记忆的 pi_0.6 只会重复同样的失败
```

**重要 ablation**: 必须在 pre-training 阶段就训练记忆 (含 robot + non-robot video 数据), 仅在 post-training 引入记忆效果显著更差。加记忆不降低非记忆任务性能。

**RLT (RL Tokens)**: 解决精密操作

```
问题: BC/VLA 对精密操作不够精确 (插USB, 拧螺丝)
     精密操作需要 closed-loop 微调, 不是 open-loop 预测

解法: 在 VLA 的输出 token 中加入 RL token
  VLA 输出: [action tokens] + [RL tokens]
  RL tokens: 被一个轻量 actor-critic 读取, 做 15 分钟的在线 RL

  15 分钟在线 RL → 学会精密操作 (VLA 做粗定位, RL 做精调)
```

**Takeaway 6**: **PI 的发展路线回答了 "BC 之后怎么办"**:
1. BC 不够准 → 加 RL (pi\*0.6 离线, RLT 在线)
2. BC 不够泛化 → 加 Knowledge Insulation (pi_0.5)
3. BC 不够快 → 换 tokenizer (FAST)
4. BC 不够长 → 加记忆 (MEM)

这条路线的终点是: **VLA + offline RL + online RL + memory = 完整的 robot agent。**

---

## 3. PI vs GR00T: 两种完全不同的哲学

| | PI 的思路 | NVIDIA 的思路 |
|---|---|---|
| **核心信念** | 一个足够好的 VLA 能解决一切 | 不同问题需要不同模块 |
| **架构** | 单一 VLA (观察→动作, 端到端) | 分层系统 (VLM + DiT + SONIC + PD) |
| **对低层控制的态度** | 不需要 (VLA 直接出关节角) | **必须** (SONIC 做运动追踪) |
| **对世界模型的态度** | 不需要 (VLA 隐式学物理) | **下一代核心** (DreamZero/WAM) |
| **改进方向** | 纵深: tokenizer→泛化→RL→记忆 | 横向: VLA + WBC + WorldModel |
| **目标机器人** | 操作臂 (7-DOF, 桌面任务) | **人形全身** (29+ DOF) |
| **Sim2Real** | 几乎不用仿真 (直接真机 demo) | **重度依赖仿真** (Isaac Sim) |
| **数据策略** | DROID + 自采真实数据 | data pyramid (sim > synthetic > real) |
| **对 RL 的态度** | 后期加入 (pi\*0.6 离线 RL) | 从一开始就用 (SONIC 全程 PPO) |

**两种哲学的根本分歧: 需不需要低层控制器?**

```
PI 的观点:
  VLA 直接输出关节角 → PD 控制器执行
  "模型足够强就不需要中间层"
  适用于: 桌面操作臂 (动作简单, 不需要平衡)

NVIDIA 的观点:
  VLA 输出目标轨迹 → SONIC 追踪 → PD 执行
  "低层控制必须专门处理"
  适用于: 人形全身 (需要平衡, 动作复杂)

这不是对错, 是问题不同:
  PI 做的: 单臂抓放 → 7 DOF, 不需要平衡, VLA 直接出关节角足够
  NVIDIA 做的: 人形走+抓 → 29 DOF, 必须平衡, VLA 直接出关节角会摔
```

**对你的启示**: 你做人形机器人, NVIDIA 的分层方案更适合你。但 PI 的 RL 集成 (pi\*0.6) 和 Knowledge Insulation (pi_0.5) 是通用技巧, 可以用在任何架构上。

---

## 4. PI 的开源策略

PI 开源了推理栈 (openpi 代码 + HuggingFace 权重 + fine-tune 脚本 + 数据工具), 但没开源训练栈 (预训练代码、训练数据、RECAP RL 代码、FAST tokenizer 训练、Knowledge Insulation 实现)。对比 NVIDIA 开源了完整训练栈 (Isaac-GR00T + GR00T-WBC)。**结论: PI 论文学思想, 复现靠 NVIDIA。**

---

## 5. 核心 Takeaway (按可执行性排序)

| # | Takeaway | 原理 | 对你的行动项 |
|---|----------|------|------------|
| 1 | **Flow Matching 是 VLA 动作生成的行业共识** | PI 和 NVIDIA 独立选择了同一方案, 比 diffusion 更快更稳 | 你的 CV 知识库已有 `22_FlowMatching`, 优先学这个 |
| 2 | **VLM + Action Expert 分离是正确设计** | 动作生成不应干扰语言理解, 用 cross-attention 连接 | 如果你做 VLA, 不要把动作 head 和 VLM 共享参数 |
| 3 | **Knowledge Insulation 防止灾难性遗忘** | 底层冻结 (保护通用视觉) + 顶层解冻 (适应 robot) | fine-tune 任何预训练模型时都应考虑这个 |
| 4 | **离线 RL 是 BC 的天然下一步 (pi\*0.6)** | 好 demo 学 "该怎么做", 坏 demo 学 "不该怎么做" | 你的失败轨迹不要扔, 标 advantage 后能用于 RL |
| 5 | **DCT tokenizer 利用了动作的物理先验** | 机器人动作是平滑的 → 频域压缩比时域量化更高效 | 如果你要 tokenize 动作, 考虑频域方法 |
| 6 | **层级 VLA 处理复杂多步任务 (Hi Robot)** | 高层规划 + 低层执行, 和 GR00T 分层互补 | 长 horizon 任务需要分层, 不要试图端到端 |
| 7 | **异构数据共训练 + web 数据保泛化 (pi_0.5)** | 只用 robot 数据 → 过拟合; 混入 web 数据 → 保持通用性 | 训练时混入非 robot 数据 (视频, 图文) |
| 8 | **显式记忆解锁长时任务 (MEM)** | 短期视频 + 长期文本摘要 = 多尺度记忆 | 超过 1 分钟的任务需要记忆, context window 不够 |
| 9 | **PI 不做低层控制, 但你应该做 (如果做人形)** | 桌面臂可以 VLA→关节角; 人形必须有 WBC 层 | SONIC 的方案对你更实用 |
| 10 | **PI 走 VLA 纵深, 但 NVIDIA 正在用 WAM 替代 VLA** | WAM (World-Action Model): 想象→做, 替代 VLA 的看→做 | 关注 GR00T 的 DreamZero/WAM 范式, 这是下一代潜在方向 |

---

## 6. 文件索引

```
pi_Series/
├── pi_family_notes.md              <- 本文件 (含 DROID/FAST/HiRobot/MEM 的蒸馏)
├── pi0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.pdf
├── 24_pi0/                         # pi_0 论文 + notes
│   ├── openpi/                     #   推理代码 (github.com/Physical-Intelligence/openpi)
│   └── pi-data-sharing/            #   数据工具
├── 25_pi05/                        # pi_0.5 开放世界泛化论文
├── 25_pi06/                        # pi*0.6 离线 RL (RECAP) 论文
└── supporting_papers/              # 支撑论文原文 (内容已蒸馏至本文件)
    ├── 24_DROID/                   #   数据集: 76k 轨迹, 564 场景
    ├── 25_FAST/                    #   DCT 动作 tokenizer
    ├── 25_HiRobot/                 #   层级 VLA (高层规划+低层执行)
    └── 26_MEM/                     #   多尺度记忆 (短期视频+长期语言)
```
