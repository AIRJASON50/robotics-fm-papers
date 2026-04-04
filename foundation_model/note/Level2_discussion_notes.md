# Level 2 讨论笔记：视觉-语言 + 生成模型

基于 CS2Robotics_Roadmap Level 2 学习过程中的讨论整理。

---

## 1. ViT: 图片变成 Token

### 核心流程 (4 步)

```
1. 切 patch:       224×224 图片 → 196 个 16×16 patch (14×14 网格, 不重叠)
2. 线性投影:       每个 patch (16×16×3=768 个像素值) → 矩阵乘法 W → D 维向量
                   实现: Conv2D(kernel=16, stride=16, out=D), 等价于 flatten + Dense
                   W 是可学习的, 训练中学会从像素提取特征
3. + position embedding: 每个 token += 对应位置的 D 维可学习向量
4. 进 Transformer:  196 个 D 维 token 做 attention + FFN
```

### 线性投影的本质

不是归一化或转置, 是一个可学习的矩阵乘法:
- 输入: 768 个原始像素值 (冗余, 相邻像素高度相似)
- 输出: D 维特征向量 (压缩 + 提取有用特征)
- W 在训练中通过反向传播学会特征提取 (类似 "边缘检测" "纹理识别")
- Conv2D 只是工程实现, 本质就是矩阵乘法, GPU 对卷积有硬件加速

### Position Embedding 的必要性

**Transformer 的 attention 是位置无关的 (permutation invariant)**: 如果把 196 个 token 随机打乱顺序, attention 计算结果完全一样。模型本身没有任何机制能区分"第一个 token"和"最后一个 token"。

因此 position embedding 不是多余的辅助, 而是**弥补 Transformer 架构本身缺失的位置感知能力**:
- 没有它: 模型不知道哪个 patch 在左上角, 哪个在右下角
- attention 学会 "这两个 patch 内容相关" (无序关系)
- position embedding 告诉模型 "这两个 patch 在空间上相邻" (有序关系)

**实现方式**: 不是在向量后面 append 一个位置数字, 而是加一个同维度的向量 (逐元素相加):

```
token    = [0.3, -0.1, 0.5, ...] 768 维 (patch 特征)
pos_emb  = [0.01, 0.03, -0.02, ...] 768 维 (第 n 个位置的可学习向量)
结果     = [0.31, -0.07, 0.48, ...] 768 维 (位置信息渗透到每一维)
```

为什么不 append 而是相加: append 一个数字占 1/769, 被淹没; 相加让位置信息渗透到向量的每一维, attention 能充分利用。

**来源**: 随机初始化, 和模型其他参数一起端到端训练。没有独立的 loss 或真值, 通过最终任务 loss 反传间接学习。ViT 发现 1D position embedding 就够 (不需要编码 2D 行列), 模型能自己从 1D 位置中学到 2D 空间关系。

### Attention 不改变维度

```
输入: 196 个 token, 每个 768 维
输出: 196 个 token, 每个 768 维

数量不变, 维度不变, 变的是每个向量的"内容"
每个 token 从"只包含自己 patch 的信息"变成"融合了全局 context 的信息"
```

### ViT 的意义 (对机器人)

ViT 之前 CV 用 CNN, 之后图像和文本可以用同一种 Transformer 架构处理。这使得:
- CLIP (图文对齐) 成为可能
- VLA (视觉语言动作) 可以把图像、文字、动作 token 放进同一个 Transformer
- pi_0 / GR00T N1 的视觉 backbone 都是 ViT 变体 (SigLIP 用的 ViT-So400m)

---

## 2. CLIP: 图文对齐

### 核心结构

两个完全独立的 Transformer (不共享参数, 无 cross-attention):

```
图片 → ViT (self-attention) → 768维 → 线性投影 → 512维 → L2 归一化
                                                                ↓
                                                          余弦相似度
                                                                ↑
文字 → Text Transformer (self-attention) → 512维 → 线性投影 → 512维 → L2 归一化
```

对比 loss 对齐的不是两个 encoder 内部的 attention 模式, 而是**最终输出向量在共享空间中的位置**。两个 encoder 内部的 attention 可以完全不同 — Image ViT 学"图片内部哪些区域相关", Text Transformer 学"语言内部哪些词相关"。线性投影的作用是"翻译", 让两种不同的内部表征可以在同一空间里比较距离。

### 对比 loss 不是 L1/L2

```
L1/L2 loss: 只拉近正样本, 不推远负样本 → 所有向量坍缩到一个点
对比 loss (InfoNCE): 正样本拉近 + batch 内 N-1 个负样本推远
  → 需要大 batch (CLIP 用 32768) — 负样本越多, 表征区分度越好
```

### CLIP vs ViT 的本质区别

```
ViT:   图片 → 固定 1000 类概率 (封闭词表, 只保留类别, 丢弃属性/空间)
CLIP:  图片 → 和任意文字描述的匹配概率 (开放词表, 保留丰富语义)
```

CLIP 的"库"是推理时你自己给的文字, 不是训练时固定的。同一个模型可以做分类/检索/zero-shot 识别, 只需换候选文字。

### CLIP 是 encoder-only, 不是 VLM

```
CLIP: 只能匹配 — "图和文有多像?" (不能生成文字或动作)
VLM:  能生成 — 图 → encoder → LLM decoder → 生成文字/动作

CLIP + LLM = VLM:
  SigLIP (CLIP 后代) + Gemma 2B = PaliGemma → pi_0 的 backbone
```

### CLIP 输出的全局向量丢失空间信息

```
CLIP: 196 个 patch tokens → 只取 [CLS] → 压成 1 个全局向量
  → 知道"图里有红色杯子", 不知道"杯子在左边还是右边"

后续 VLM (SigLIP/PaliGemma): 保留全部 196 个 spatial tokens
  → 每个 token 编码对应区域的语义, 通过 attention 实现空间定位
```

### CLIP 的遗产

| 层面 | 被继承 | 被替代 |
|------|-------|-------|
| 方法 | 对比学习对齐图文 → SigLIP 改进 loss 但思想一致 | softmax loss → SigLIP 改成 sigmoid |
| 架构 | ViT + Transformer 双 encoder → 所有 VLM 基本结构 | CLIP text encoder → 被 LLM 替代 |
| Backbone | CLIP ViT 被 Diffusion Policy 直接用 | 被 SigLIP ViT (pi_0) 和 DINOv2 (OpenVLA) 替代 |

### Vision backbone 选择 (考试 Q3 纠错)

| 选项 | 适合场景 | 不适合 |
|------|---------|--------|
| A: 从头训 ResNet-18 | 单任务 + 少数据 + 需要 100Hz + 不需要语言 | 多任务泛化 |
| B: ImageNet ViT 冻结 | **几乎不推荐** — 无语言对齐不如 C, 速度不如 A | 两头不靠 |
| C: SigLIP ViT + LoRA | 多任务 + 语言指令 + 中等数据量 | 极端高频 (>50Hz 无 chunk) |

### 关键认知: 架构不是瓶颈, 预训练任务才是

```
同一个 ViT 架构, 不同预训练:
  ImageNet 预训练: 学到"这是杯子" (1000 类分类)    → 对机器人帮助有限
  SigLIP 预训练:  学到"红色杯子在桌子左边" (图文对齐) → 直接可用

ResNet-18 在某些场景反而合适:
  不是因为架构好, 而是够小够快 (11M, 100Hz)
  单任务从头训就够了, 不需要预训练知识
```

B 不推荐的原因不是"ViT 比 ResNet 差", 而是 ImageNet 分类预训练给的先验对机器人帮助不大, 却白白增加了计算量 (86M vs 11M)。

**OmniReset 的验证**: 试了 DINO 和 pi_0.5 替代 ResNet-18, 在 80K 单任务数据下没有显著提升。说明瓶颈不在 backbone 架构/能力, 而在 BC distillation 方法本身 (student 只见过 teacher 的完美轨迹, 没见过错误恢复, compounding error)。

### 偶得: 生成模型的源头是压缩重建, 压缩重建就是表征学习

生成模型的思想脉络不是凭空出现的, 而是从"压缩→重建"自然演化出来的:

```
PCA (1901)           → 线性压缩, 保留最大方差
AutoEncoder (1986)   → 非线性压缩 (第一次用 NN 做表征)
VAE (2013)           → 压缩成概率分布 → 意外获得了生成能力
GAN (2014)           → 放弃压缩, 直接学生成
Diffusion (2020)     → 放弃压缩, 用迭代去噪生成
```

核心 insight: **好的压缩 = 好的表征 = 保留互信息**。这个思想贯穿了所有内容:
- AE/VAE: 压缩重建, 直接优化互信息
- CLIP: 把图文压缩到同一空间, 保留跨模态互信息
- ViT: patch → token, 保留空间语义互信息
- VLM backbone: 互联网知识压缩在权重里, 迁移到机器人时仍然有效

**从"压缩"到"生成"的关键跳跃**: 如果 latent space 被填满 (VAE 用 KL loss 做到), 任意采样都能生成合理数据。压缩做得足够好, 就顺便获得了生成能力。

---

## 3. 生成模型的演化: 从 AE 到 DDPM

### 从"latent 重建"到"从噪声生成"的四步演化

```
AE:     latent → 重建 (只能重建见过的, latent space 有空洞)
  ↓ VAE 用 KL loss 填满 latent space → 任意采样都有意义
VAE:    z ~ N(0,1) → decoder → 生成 (可生成但模糊)
  ↓ GAN 去掉 encoder, 直接从噪声到数据
GAN:    z ~ N(0,1) → generator → 生成 (清晰但训练不稳定)
  ↓ DDPM 把一步生成拆成 1000 步小步去噪
DDPM:   z ~ N(0,1) → 去噪 × 1000 → 生成 (清晰且稳定)
```

每一代解决上一代的痛点:
- AE: latent 有空洞 → VAE 用 KL 填满
- VAE: 高斯假设太强, 模糊 → GAN 隐式学习, 不做假设
- GAN: 对抗训练不稳定 → DDPM 拆成简单回归小步
- DDPM: 1000 步太慢 → Flow Matching 直线路径, 10 步

### VAE 和 CVAE 的训练目的

```
VAE:   学习数据的概率分布 P(x), 从而能生成新数据
CVAE:  学习条件概率分布 P(x|c), 从而能在给定条件下生成新数据
```

都是通用概率生成框架, 不绑定任何领域:
- VAE: x=图像 → 生成图像; x=分子 → 生成药物
- CVAE: x=图像, c=标签 → 条件生成图像; x=动作, c=观测 → ACT

### CVAE 的条件分离机制

CVAE 的 encoder 不是把条件编码进 z, 而是把条件"剥离"出去:

```
训练一张粗体斜写的 "3", 条件 c = "3":
  条件已经告诉了"是3" → z 不需要再编码"是几"
  z 只编码: 粗细, 倾斜 (条件解释不了的变化)

两个 loss 的博弈:
  重建 loss: z 多放信息 → 重建更准确
  KL loss:   z 少放信息 → 越接近 N(0,1) 越好
  平衡: z 只放条件解释不了的最少必要信息

结果: 条件控制"生成什么", z 控制"怎么生成"
```

这就是为什么 CVAE 能处理 multimodal action distribution:
- condition = 看到杯子 → 决定 "要抓杯子"
- z 采样不同值 → 从左抓 / 从右抓 / 从上抓 (都合理)

### 条件注入方式

| 方式 | 做法 | 适合场景 | 代表 |
|------|------|---------|------|
| 拼接 | 条件和数据 concat 成一个向量 | 简单任务, 小模型 | ACT |
| FiLM | 条件通过 scale+shift 调制每一层 | 中等复杂度 | Diffusion Policy |
| Cross-attention | 数据 attend to 条件 | 复杂多模态 | GR00T N1 |
| Shared self-attention | 数据和条件 token 拼在一起做 attention | 大模型 | pi_0 |

拼接最简单但效率最低 (NN 要自己学分辨哪部分是条件); cross-attention 显式分开, 学习效率更高。

### 两种生成范式的几何本质 (关键 insight)

```
范式 A: 跨空间映射 (AE / VAE)
  data space (高维) ←→ latent space (低维)
  本质: 找到数据流形的坐标系
  类比: 地球表面 → 用经纬度描述
  → 有 latent space, 能插值/编辑, 但跨维度一步到位学不好 → 模糊

范式 B: 同空间映射 (DDPM / Flow Matching)
  data space (高维) → data space (高维)
  本质: 不找坐标, 学一个"引力场"把任意点推到流形上
  类比: 不知道经纬度, 但引力场能把太空中任意位置拉回地球表面
  → 不降维无信息丢失, 每步简单稳定, 但没有 latent space, 推理慢
```

Robotics 把两者组合: **CLIP/ViT 做跨空间映射 (表征) + Diffusion/Flow Matching 做同空间映射 (生成动作)**。

DDPM 从物理学扩散过程借来的 insight: 高维空间中真实数据只占据低维流形, 扩散过程把数据从流形上推到全空间 (加噪), 逆过程把任意点拉回流形 (去噪)。每一小步都是高斯回归 (有数学保证), 1000 个简单问题组合解决一个复杂问题。

### DDPM 的训练和推理

```
训练 (随机采样, 独立, 快):
  1. 取干净图 x_0, 随机采 t ∈ {1,...,1000}, 随机采噪声 ε
  2. 一步算出: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
  3. 网络预测: ε_pred = UNet(x_t, t)   ← 输入加噪图+时间步, 输出预测噪声
  4. loss = ||ε_pred - ε||^2            ← MSE, 预测噪声 vs 实际噪声

  关键设计:
  - 只有一个网络, 靠 t 区分不同噪声程度 (不是 1000 个独立模型)
  - x_t 直接从 x_0 用公式算出, 不需要跑 1000 步
  - 每个样本独立, 不依赖上一步输出 → 训练时步骤间解耦
  - 预测噪声而非去噪后图片 (数学等价, 但实践更稳定)

推理 (顺序迭代, 慢):
  x_1000 (纯噪声) → UNet 预测噪声 → 减去 → x_999 → ... → x_0
  → 必须顺序执行, 每步依赖上一步输出 (1000 次 forward pass)
  → 这就是推理慢的原因

不同 t 的难度不均匀:
  t≈1000: 学大结构 (全局统计规律, 相对简单, 收敛快)
  t≈1:    学像素级细节 (精细分布, 收敛慢)
  → 训练都能学, 只是收敛速度不同
  → 后续改进 (DDIM 跳步, Flow Matching 均匀化) 都在优化这一点
```

### DDPM → Flow Matching: 弯路变直路

DDPM 和 Flow Matching 的核心区别只在**前向插值公式** (怎么混合数据和噪声):

```
DDPM:          x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε   (弯曲路径)
Flow Matching: x_t = (1-t) * x_0 + t * ε                  (直线路径)
```

两者都是人为设计的路径, 不是自然物理过程。但直线全面优于弯曲:
- 推理快: 10 步 vs 1000 步 (直线步长可以很大)
- 训练简单: 目标速度 = x_1 - x_0 (常数), 不需要设计 noise schedule
- 更均匀: 直线上每段难度差不多, 弯曲路径不同 t 难度差异大

**数学框架的区别**:
- DDPM: SDE (随机微分方程) → 每步加随机噪声 → 同一起点走出不同路径 → 网络学"平均方向"
- Flow Matching: ODE (常微分方程) → 确定性 → 同一起点只有一条路径 → 网络学唯一方向

**直线是唯一最短路径** — 弯曲路径有无穷多条, 网络需要学平均; 直线只有一条, 方向恒定, 学习最简单。

网络真正学的不是"某条路径", 而是"数据流形在哪"。路径只影响学多快和推理多快, 不影响最终能不能学会。所以 2024+ 新工作几乎都选 Flow Matching (pi_0, GR00T N1, Stable Diffusion 3)。

**名字的由来**: Flow = 从噪声流向数据的连续速度场; Matching = 让网络预测的速度匹配目标速度。DDPM 是随机轨迹不是确定的"流", 所以不叫 flow。

### DiT: U-Net → Transformer

ViT 在分类上证明 Transformer > CNN, DiT 在生成 (去噪) 上重复了同一结论:
- U-Net (CNN 架构) → DiT (Transformer 架构), 功能完全相同 (输入加噪图+t, 输出噪声预测)
- DiT 的 scaling 特性更好: 计算量和生成质量呈 log-log 线性 (和 GPT scaling law 一致)
- GR00T N1 的 action head 就是 DiT + Flow Matching

### Level 2 考试纠错

**Q1 纠正**: SigLIP 优于 ImageNet ViT 的原因不是 LoRA (LoRA 是微调方法, 不是预训练方法)
- 真正原因是**预训练任务和数据**不同:
  - ImageNet ViT: 1.3M 图片 × 1000 类标签 → 只学到类别
  - SigLIP ViT: 数十亿图文对 × 开放词汇 → 学到类别+空间+属性+语言对齐
- 选型: 单任务不需要语言 → ResNet-18; 多任务/语言指令 → SigLIP + LoRA

**Q2 纠正**: VAE 模糊 + 组合方式
- VAE 模糊: 从 50 维 latent 一步跳到 150528 维 → 跨度太大 decoder 学不好 + 高斯假设限制表达力
- DDPM 清晰: 在 150528 维原地操作, 不跨维度, 每步只改一点
- 组合方式: CLIP/SigLIP (跨空间映射做表征) + Flow Matching (同空间映射做动作生成) = pi_0

**Q3d**: 训练/推理不对称的设计
- 前向加噪有精确公式 → x_t 直接从 x_0 一步算出 → 训练时每个样本独立, 可随机采 t
- 反向去噪没有精确公式 → x_{t-1} 必须等网络输出 → 推理时必须串行 1000 步
- 这是 DDPM 设计的精巧之处: 训练时解耦了步骤依赖, 推理时才串联

**Q4 补充**: SDE vs ODE
- SDE (DDPM): 每步加随机噪声 → 同一起点走出多条路径 → 学平均方向 → 需要多步
- ODE (Flow Matching): 确定性 → 同一起点只有一条路径 → 学唯一方向 → 少步够用

**Q5 补充**: GAN 和训练产物
- GAN 解决了 VAE 的模糊问题: 不做高斯假设, 隐式学习 → 清晰, 但引入对抗训练不稳定
- 训练目的和产物:
  - AE: 压缩表征 → 保留 encoder
  - VAE: 概率生成 → 保留 decoder
  - GAN: 对抗生成 → 保留 generator; 判别器思想 → AMP/GAIL (humanoid motion reward)
  - DDPM: 去噪生成 → 保留去噪网络 (U-Net)
  - FM: 速度场生成 → 保留速度网络

---

## Level 2 完成后的全局回顾: CS → VLA 的完整链路

### VLA 的核心公式

```
图像 + 语言指令 → VLM backbone (理解) → Action Head (生成动作) → 执行器
```

你学过的每一块对应 VLA 的哪个组件:

```
Level 0 表征学习:     好的表征=好的AI → VLM backbone 的存在意义
Level 0 Transformer:  VLA 全部组件的底层架构
Level 1 GPT 范式:     pre-train+fine-tune → VLA 的训练范式
Level 1 Scaling Law:  模型和数据要匹配 → pi_0 的 3.3B 配 10k 小时
Level 1 RLHF:         post-training 思想 → pi*0.6 的 offline RL
Level 2 ViT:          图片切 patch 变 token → VLA 的图像输入格式
Level 2 CLIP/SigLIP:  图文对齐 → VLA 的 vision backbone 从这里来
Level 2 VAE/CVAE:     条件生成 → ACT 的动作生成
Level 2 DDPM:         同空间去噪生成 → Diffusion Policy 的动作生成
Level 2 Flow Matching: 直线路径, 10 步 → pi_0 的动作生成
Level 2 DiT:          Transformer 做去噪 → GR00T N1 的 action head
```

### 范式如何收敛到 VLA

```
2022 前: 各自为政, 没有统一范式
  RL (PPO): 每任务独立训, 不可扩展
  BC: 简单但不泛化
  LLM-as-planner (SayCan): LLM 规划, 低层手写

2022: RT-1 证明 Transformer 在机器人上 work
2023: RT-2 证明 VLM 知识可以迁移到机器人 ← 范式确立
2024: pi_0/OpenVLA/Octo 同时验证 ← 范式被广泛接受
2025: GR00T N1 扩展到人形机器人

收敛原因 (三个独立技术在 2023 年交汇):
  1. VLM 预训练的 backbone 远好于从头训 (你的关键 insight)
  2. 生成模型 (diffusion/flow) 解决了多模态动作分布
  3. Transformer 统一了所有模态 (image+text+action)
```

### 当前核心问题

```
1. 精度不够:  VLA 擅长粗操作 (抓杯子), 做不好亚毫米装配/灵巧操控
   → 你的灵巧手工作正好在这个前沿
2. 数据效率:  pi_0 用 10k 小时遥操作, 采集成本极高 (GPT 用免费互联网数据)
3. 部署速度:  VLM 推理 ~14Hz, 灵巧操控需要 50-120Hz
4. 泛化边界:  新物体 OK, 新任务/新机器人还是未知
5. 安全可控:  黑箱, 无法解释"为什么这样动"
```

### 正在探索的新方向

```
1. World Model (DreamZero):       执行前先想象结果 → 选最好的方案
2. Online RL Post-training (pi*0.6): VLA 部署后继续用 RL 学 → 你的 PPO 经验直接适用
3. 分层系统 (GR00T):              VLM 10Hz 大脑 + SONIC 120Hz 小脑
4. 数据飞轮 (AutoRT):             机器人自主探索 + VLM 评估, 不靠人类遥操作
5. 跨模态 (触觉/力反馈):          相机不够 → 加触觉, 对灵巧手尤其重要
```

### 你在 VLA 世界的位置

```
你有的:                    对应 VLA 的:
  PPO + reward design      pi*0.6 的 RL post-training
  sim2real (DR, sysid)     GR00T SONIC 的 sim2real pipeline
  motion tracking          SONIC 的 universal tracker
  灵巧手 20 DoF            VLA 精度前沿 (大多数 VLA 只做粗操作)

你缺的:
  VLM 使用经验             需要跑 openpi
  大规模数据工程            需要理解 Open X
  生成模型训练经验          需要看 flow matching 代码
```

---

## DDPM + Flow Matching 考试纠错 (续)

Q1 纠正: **网络每次只预测一个噪声向量 ε, 不是整个去噪序列**
- 输入: 加噪图片 x_t + 时间步 t
- 输出: 预测的噪声 ε_pred (一个向量, 和 x_t 同维度)
- loss: ||ε_pred - ε_actual||^2 (MSE)
- 推理时同一个网络调用 1000 次, 每次只去掉一步噪声: x_1000 → x_999 → ... → x_0
- 预测噪声而非去噪图片: 数学等价 (知道 ε 可算出 x_0), 但梯度更稳定

Q3 补充: pi_0 选 Flow Matching 不只是"快"
1. 快: 10 步 vs 1000 步, 50Hz 控制必须快
2. 简单: 不需要设计 noise schedule (没有 α_t 衰减曲线)
3. 语义直观: 在 action space 中, 速度场 = 动作变化方向, 物理含义清晰

### DDPM 的去噪网络: U-Net

U-Net (2015, 原为医学图像分割) 是一种像素到像素的 CNN 架构:

```
功能: 输入一张图, 输出同尺寸的图 (每个像素映射到一个值)
  医学分割: 每像素 → 分类标签 (细胞/背景)
  DDPM 去噪: 每像素 → 噪声预测值 (浮点数)

结构: U 形 encoder-decoder + skip connections
  encoder (左半边): 卷积下采样, 分辨率逐步缩小, 捕捉全局结构
  decoder (右半边): 上采样, 分辨率逐步恢复
  skip connections: 左边每层直接连到右边对应层, 保留空间细节
```

DDPM 用的不是原版 U-Net, 加了两个改动:
- Timestep embedding: 把时间步 t 编码成向量注入每一层 (告诉网络当前噪声程度)
- Self-attention 层: 让像素之间交互信息 (U-Net 原版只有局部卷积)
- 但整体 U 形结构不变

**U-Net 就是 DDPM 的全部** — 没有 encoder, 没有 discriminator, 没有 latent space。只有一个 U-Net 做像素级噪声预测。DiT 做的事就是把这个 U-Net 换成 Transformer。

### CS → Robotics 的迁移模式

**框架不变, 数据换了:**
- VAE (2013): x=图像 → 生成图像
- CVAE (2015): x=图像, c=标签 → 条件生成图像
- ACT (2023): x=动作, c=观测 → 条件生成动作 (把数据从图像换成动作)
- Diffusion Policy (2023): DDPM 的 x 从图像换成动作
- pi_0 (2024): Flow Matching 的 x 从图像换成动作

### AutoEncoder: 第一次用 NN 做表征学习

AE 之前的表征/压缩方法都是**基于数学公式的, 人工定义压缩规则**:
- PCA: 定义"投影到方差最大的方向" → 线性, 规则固定
- JPEG: 定义 DCT 变换 + 量化表 → 手工设计, 不自动适应数据
- 稀疏编码: 定义"用尽量少的基向量组合" → 优化问题, 但基向量要人选

AE (1986, Rumelhart) 的突破: **让 NN 自己学压缩和解压的规则**
- 不需要人定义"保留什么, 丢弃什么"
- 只给目标: 输出 = 输入 (重建 loss)
- NN 自己找到最优的压缩策略
- 这是 NN 用于表征学习的开创性工作, 后续所有 learned representation 都继承了这个思路

### 历史修正: "聚类 → 生成" 不是真实的思路链条

**降维/表征 和 生成模型 是两条独立发展的线, 不是从一条推出另一条:**

```
线 A: 降维/表征 (不涉及生成)
  AE (1986) → Hinton Science 2006 (AE 比 PCA 降维更好) → Bengio 2012 (表征学习综述)
  目标始终是: 好的压缩、好的特征
  聚类现象是副产品, 没有人从中"推导出"生成

线 B: 概率生成模型 (一直在尝试生成)
  玻尔兹曼机 (1985) → Hinton DBN 2006 (RBM 逐层预训练, 第一次训出深层生成模型)
  → VAE (2013) → GAN (2014) → DDPM (2020)
  RBM/DBN 天生就是概率生成模型, 不需要从聚类"反推"

交叉点:
  Hinton 2006 用线 B 的 RBM 权重来初始化线 A 的 AE (工程技巧, 不是思想传承)
```

**Hinton DBN 论文的原话** (Section 7): "The network has a full generative model, so it is easy to look into its mind — we simply generate an image from its high-level representations." → 他从一开始就是在做生成模型, 不是"从聚类观察反推可以生成"。

"聚类 → 生成"的叙事事后看合理 (好的 latent space 确实有利于生成), 但历史上做生成的人是从概率论出发的, 不是从降维的聚类观察出发的。

### 对机器人工程师的启示

**选 vision backbone 不是看架构 (ViT vs ResNet vs CNN), 而是看它的预训练方法和数据集决定了它学会了什么。** 具体需要判断:

1. **预训练任务**: 分类 (只知道类别) vs 图文对齐 (理解语言+空间) vs 自监督 (空间细节强)
2. **预训练数据**: ImageNet 1.3M 张 vs 互联网数十亿图文对 → 见过的视觉概念差几个量级
3. **输出格式**: 全局向量 (丢空间信息) vs spatial tokens (保留位置) → 机器人需要后者
4. **和你的场景匹配度**: 互联网图片 vs 机器人视角差异大 → 可能需要解冻微调

这些判断能力要求机器人工程师了解 CV 的关键工作 (ViT, CLIP, SigLIP, DINOv2) 的训练方式, 而不只是知道"用 ViT"。

---

## 关键理解: CS→Robotics 的核心范式 -- VLM backbone 直接复用

这是理解 LLM/CV 如何迁移到 Robotics 的关键转折点。

### 范式

**直接复用 CV 预训练的 VLM (Vision-Language Model) 作为机器人的视觉理解 backbone, 只需要接一个 action head 教它"怎么动"。**

VLM 在互联网图文数据上学到的不只是 "这是猫":
- "红色杯子在桌子左边" → 空间关系理解
- "手正在拿起杯子" → 动作语义理解
- "杯子是倒着的" → 物体状态理解

这些能力对机器人直接有用。action head 不需要理解语言和视觉, 只需要把 VLM 输出的表征 (已编码了场景理解) 映射成关节动作。这个映射远比"从像素学会一切"简单。

### 验证链条 (5 年逐步验证)

| 时间 | 工作 | 验证了什么 |
|------|------|---------|
| 2021 | CLIP | 视觉特征可以跨任务 zero-shot 迁移 |
| 2023 | RT-2 | VLM 的语言理解能力可以转移到机器人指令理解 |
| 2024 | Octo (反例) | 不用预训练 VLM, 从头训 93M 小模型 → 有效但上限低 |
| 2024 | OpenVLA | VLM backbone fine-tune 后, 7B 超过 55B RT-2; 冻结 vision 效果差, 解冻后好 |
| 2024 | pi_0 | 3B VLM backbone (PaliGemma) 直接支撑灵巧操作级别的 VLA |
| 2025 | GR00T N1 | VLM backbone 带来量级级别的 data efficiency (10% 数据 > Diffusion Policy 全量) |

### 反面对比: 不用 VLM backbone 的 OmniReset

OmniReset 用 ImageNet 预训练的 ResNet-18 (冻结) 做 RGB distillation:
- State expert: ~100% → RGB student: ~50%
- 试过 DINO 和 pi_0.5 → 80K 单任务数据下没有显著提升
- 原因: 80K 单任务数据太少, 撑不起大 backbone 的微调

这说明: backbone 质量很重要, 但**必须和数据量匹配** (Chinchilla 的 insight)。
单任务少数据: ResNet-18 够用, 大 backbone 浪费。
多任务大数据: 必须用 VLM backbone, 否则上限太低。

### 这个范式对灵巧手研究的实际意义

```
以前的思路 (OmniReset 式):
  仿真训 RL expert (state) → distill 到 vision policy (ResNet-18)
  → 每个任务独立, 不共享视觉理解

现在的思路 (pi_0 式):
  复用 VLM backbone (已理解语言+视觉) → 接 action head → fine-tune
  → 换任务只需换数据, backbone 的理解能力是共享的
  → 新任务不需要从头训视觉理解

对你:
  如果你做单任务灵巧手 (如方块重定向) → OmniReset 式 ResNet-18 可能够用
  如果你做多任务/语言指令驱动 → 必须走 VLM backbone 路线
```

---

## Vision Encoder 全景对比

### 速度 vs 能力的 tradeoff

```
快但弱                                              慢但强
  ←─────────────────────────────────────────────────→

  ResNet-18     SigLIP ViT    DINOv2+SigLIP    ViT-22B
  (11M, ~100Hz) (400M, ~14Hz)  (600M, ~6Hz)    (22B, ~1-3Hz)
  ACT/DP/OmniReset  pi_0/GR00T    OpenVLA         RT-2
```

### 完整对比表

| 模型 | Vision Encoder | Encoder 大小 | 推理频率 | 预训练数据 | 输出格式 |
|------|---------------|------------|---------|----------|---------|
| ACT | ResNet-18 | 11M | ~100Hz | 无 (从头训) | global feature vector |
| Diffusion Policy | ResNet-18 + SpatialSoftmax | 11M | ~10-20Hz | 无 (从头训) | global feature vector |
| OmniReset (distill) | ResNet-18 (frozen) | 11M | ~200Hz | ImageNet (冻结) | global feature vector |
| RT-1 | EfficientNet-B3 | 15M | 3Hz | ImageNet | 8 compressed tokens |
| Octo | SmallStem16 (浅层 CNN) | ~几M | ~10Hz | 无 (从头训) | patch tokens |
| pi_0 | SigLIP ViT (PaliGemma) | 400M | ~14Hz | 互联网图文对 (数十亿) | spatial tokens |
| GR00T N1 | SigLIP-2 (Eagle VLM) | 400M | 10Hz VLM + 120Hz DiT | 互联网图文对 (数十亿) | 64 spatial tokens |
| OpenVLA | DINOv2 + SigLIP (双 encoder) | 600M | ~6Hz | 互联网图文对 | concat patch tokens |
| RT-2 | ViT-22B | 22B | 1-3Hz | 互联网图文 | patch tokens |

### 灵巧手 (50Hz) 的适用性

```
ResNet-18 (100Hz): 速度远超 50Hz, 单任务够用, 但无语言/多任务泛化
SigLIP 400M (14Hz): 速度不够 50Hz, 但通过 action chunking 补偿
  pi_0: 14Hz 生成, 每次出 50 步 chunk → 等效 50Hz 控制
  GR00T N1: VLM 10Hz + DiT 120Hz 异步 → 动作频率远超视觉频率
ViT-22B (1-3Hz): 灵巧操作完全不可用
```

### Vision → Action 的传递方式

不同模型把视觉表征传给 action head 的方式不同, 但本质都是让 action head "看到"视觉信息:

| 传递方式 | 代表 | 做法 |
|---------|------|------|
| Shared self-attention | pi_0 | vision + action tokens 拼在一起做 attention, VLM 和 Action Expert 各自有独立权重 |
| Cross-attention | GR00T N1 | DiT 通过 cross-attention "查询" VLM 输出的 vision tokens |
| Autoregressive 续写 | RT-2, OpenVLA | vision tokens 当 prefix, action tokens 接在后面逐个生成 |
| FiLM conditioning | Diffusion Policy | 视觉特征通过 scale/shift 调制去噪网络每一层 |
| 直接拼接 | ACT | 视觉特征 flatten 后直接作为 Transformer decoder 输入 |

### 更深的理解: VLM 输出不只是"一个 latent vector"

```
浅层理解:
  VLM 输出一个 latent → action head 接入 → 训练

更完整的理解:
  1. VLM 输出的是一组 spatial tokens (不是一个向量)
     每个 token 编码图像中一个区域的语义信息 (如 pi_0: ~256 tokens)
  
  2. Action head 通过 attention 选择性关注需要的区域
     抓杯子时 attend to 杯子区域, 忽略背景
     不是"盲目接收全部信息"
  
  3. VLM 不同层的输出编码不同信息:
     浅层: 边缘/纹理 (低级视觉)
     深层: 语义/类别 (高级语义)
     GR00T N1 发现取第 12 层 (中间层) 比最终层效果更好
     → 机器人需要"中等抽象度"的特征, 不是最高级的语义
     → 太抽象会丢失空间精度, 太底层没有语义理解

  4. Vision encoder 只计算一次, action head 可以多次使用
     pi_0: vision token 的 KV cache 在 10 步 flow matching 中复用
     GR00T N1: VLM 10Hz 出 vision tokens, DiT 120Hz 反复用同一组 tokens
     → vision 不是推理瓶颈, action generation 的迭代步数才是
```
