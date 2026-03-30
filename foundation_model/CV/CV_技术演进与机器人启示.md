# CV 技术演进图谱 -- 从 CNN 到视觉基础模型的分岔与汇聚

**目的**: 从机器人基础模型的视角, 梳理 CV 领域的技术演进脉络, 识别哪些视觉技术已被迁移到 robotics, 哪些正在迁移, 哪些值得关注。

> **与 LLM 图谱的区别**: LLM 的演进是以"产品/公司"驱动的 (GPT→ChatGPT), CV 的演进是以"方法/概念"驱动的。
> 机器人 80% 的问题是视觉问题 -- 这张图谱比 LLM 图谱对你更直接相关。

---

## 1. 一张图看清全部技术交织

```
2012 ──── AlexNet ─────────────────────────────────────────────────────────
          |  CNN 时代开始, ImageNet 竞赛驱动
          |
2014 ──── VGG ── GoogLeNet ── GAN ── VAE ─────────────────────────────────
          |      更深       inception   生成对抗   变分自编码
          |                              |          |
2015 ──── ResNet ── BatchNorm ─────────────────────────────────────────────
          |  残差连接     训练稳定性
          |  ImageNet pre-train + fine-tune 范式确立
          |  *** 第一个 "foundation model" 思想: 一个 backbone 多个任务 ***
          |
2017 ──── Transformer (NLP) ───────────────────────────────────────────────
          |  attention 机制, 但此时 CV 还在用 CNN
          |
2019 ──── MoCo v1 ─────────────────────────────────────────────────────────
          |  对比学习的开端: 不需要标注, 从数据自身学表征
          |
2020 ──── ViT ───── MoCo v2 ── SimCLR ── DDPM ── DETR ── NeRF ───────────
          |  CNN→Transformer  对比学习双雄   扩散模型  Transformer检测  3D表示
          |  *** CV 全面 Transformer 化 ***
          |                   |              |
2021 ──── SwinTransformer ── MAE ── DINO ── BEiT ── CLIP ── TimeSformer/ViViT
          |  层级式ViT     掩码图像   自蒸馏  掩码token  图文对齐  视频Transformer
          |                建模(MIM)                      |
          |                 |                             |
          |     *** 自监督视觉表征的黄金年 ***              |
          |                                               |
2022 ──── VideoMAE ── Ego4D ── R3M ── VIP ── LDM/SD ── FlowMatching ──────
          |  MIM→视频   第一人称   人类视频   视频=reward  潜空间扩散  ODE生成
          |             视频数据集  →robot repr  自动生成    (Stable     (pi_0核心)
          |                         |             |        Diffusion)
          |                         |             |
          |  *** 视频表征开始服务机器人 ***         |
          |                                       |
2023 ──── DINOv2 ── SAM ── BLIP-2 ── LLaVA ── DiT ── 3DGS ───────────────
          |  自监督v2  通用分割  Q-Former   视觉指令  Diffusion+  高斯溅射
          |  (Meta)   (Meta)    VLM桥接    微调      Transformer  (实时3D)
          |                                          |
          |  *** 视觉基础模型全面成熟 ***              |
          |                                          |
2024 ──── PaliGemma ── DepthAnything ── pi_0 ── GR00T N1 ─────────────────
          |  VLM backbone   深度FM      VLM+flow  VLM+DiT
          |  (pi_0用它)     (单目深度)   *** VLA 时代: CV+LLM→Robot ***
          |
2025 ──── SAM 2 ── Depth Anything v2 ── 更多视觉基础模型 ─────────────────
```

---

## 2. 五条技术演进线与机器人对应

### 线 1: 视觉 Backbone 演进 -- 机器人看世界的"眼睛"

```
CNN 时代:
  AlexNet (2012) → VGG (2014) → ResNet (2015) → EfficientNet (2019)
  特点: 局部感受野, 平移不变性, ImageNet pre-train

Transformer 时代:
  ViT (2020) → SwinTransformer (2021) → 混合架构
  特点: 全局注意力, patch tokenization, 更强的 scaling

自监督时代:
  MoCo/SimCLR (2020) → DINO (2021) → MAE (2021) → DINOv2 (2023)
  特点: 不需要标注, 自动学通用特征
```

**对 robotics 的影响**:
- RT-1 用 EfficientNet (CNN 时代) → RT-2 用 PaLI (ViT 时代) → pi_0 用 SigLIP-ViT
- **趋势**: 机器人视觉 backbone 跟随 CV 主流, 延迟约 1-2 年
- **关键 insight**: DINOv2 的自监督特征在 robot manipulation 中已超越 CLIP, 因为 DINOv2 不需要文本配对, 纯从视觉结构学习 -- 这对没有语言标注的 robot 数据更友好

**本库对应**: `CV/0_backbone/` (ResNet, ViT, SwinTransformer) + `CV/4_self_supervised/` (MoCo→DINOv2)

### 线 2: 生成模型演进 -- 机器人动作生成的来源

```
GAN 时代 (2014-2019):
  GAN → DCGAN → StyleGAN
  问题: 训练不稳定, mode collapse

VAE 时代 (2014-2020):
  VAE → CVAE → VQ-VAE
  特点: 连续 latent space, 可控生成

Diffusion 时代 (2020-2023):
  DDPM (2020) → LDM/Stable Diffusion (2022) → DiT (2023)
  突破: 生成质量超越 GAN, 训练稳定

Flow Matching 时代 (2022+):
  Flow Matching (2022) → Rectified Flow
  突破: 比 diffusion 更快 (更少步数), ODE 而非 SDE
```

**对 robotics 的关键分岔**:

```
图像生成 (DDPM/LDM)
  |
  +--→ Diffusion Policy (2023): 把图像扩散用于动作扩散
  |    将 action sequence 当作"图像"去噪
  |    处理多模态动作分布 (同一场景多种有效动作)
  |
  +--→ DiT → GR00T N1 (2025): Transformer 做扩散 backbone
  |    DiT 替换 U-Net, 享受 Transformer scaling
  |    GR00T N1 的 System 1 (120Hz action head) 就是 DiT
  |
  +--→ Flow Matching → pi_0 (2024): ODE 做动作生成
       比 diffusion 更快 (fewer denoising steps)
       pi_0 选择 flow matching 而非 diffusion 的原因:
       机器人需要实时性, flow matching 的推理步数更少
```

**本库对应**: `CV/1_generation/` (VAE→DDPM→LDM→FlowMatching→DiT)

### 线 3: 视觉-语言对齐 -- 机器人理解指令的基础

```
独立编码时代:
  CNN 做视觉, LSTM 做语言, 后期融合

对比对齐时代 (2021):
  CLIP: 图像和文本编码到同一空间, 对比学习
  开启了 "零样本视觉理解" -- 不用微调就能识别新类别

VLM 时代 (2022-2024):
  BLIP-2 (Q-Former) → LLaVA (visual instruction tuning) → PaliGemma
  从"对齐"到"融合": 视觉 token 直接送入 LLM
```

**对 robotics 的影响**:

| CV 阶段 | Robot 应用 | 代表 |
|---------|-----------|------|
| CLIP (对齐) | 语言指令→视觉 affordance | SayCan (2022) |
| BLIP-2 (Q-Former) | 视频理解→任务描述 | Video-LLM for robot |
| LLaVA (instruction tuning) | 视觉指令→动作 | OpenVLA (2024) |
| PaliGemma (compact VLM) | VLA 的 backbone | pi_0 (2024) |

**关键 insight**: CLIP 是 "理解" (这是什么), PaliGemma 是 "行动" (看到这个该做什么)。机器人从 CLIP 到 PaliGemma 的演进, 本质是从 "视觉理解" 到 "视觉-语言-动作" 的统一。

**本库对应**: `CV/2_vl_alignment/` (CLIP→BLIP2→LLaVA→PaliGemma)

### 线 4: 3D 视觉 -- 机器人在三维世界中操作的基础

```
传统 3D:
  点云 (PointNet, 2017) → 体素 (VoxNet) → 多视图

隐式表示 (2020-2023):
  NeRF (2020): 神经辐射场, 从多视图学 3D 场景
  → 3DGS (2023): 3D 高斯溅射, 实时渲染, 比 NeRF 快 100x
  → Depth Anything (2024): 单目深度估计 FM, 从单张图推 3D

趋势: 从"需要多视图"到"单张图就够"
```

**对 robotics 的影响**:
- **NeRF/3DGS** → 机器人场景重建: 扫描一次环境, 构建 3D 地图用于 planning
- **Depth Anything** → 实时深度感知: 单个 RGB 摄像头 → 深度图 → 3D 场景理解
- **关键问题**: 当前 VLA (pi_0, GR00T N1) 几乎不用显式 3D 表示, 而是让 VLM 从 2D 图像隐式理解 3D。这是否足够? PerAct (2023) 证明显式 3D 表示在 6-DoF manipulation 上远优于 2D -- **3D 表示是 VLA 的未探索方向**

**本库对应**: `CV/3_3d_vision/` (NeRF, 3DGS, DepthAnything)

### 线 5: 视频理解 -- 机器人的时间感知

```
CNN 时代:
  I3D (2017): inflate 2D CNN to 3D
  SlowFast (2019): 双速率处理 (快通道 + 慢通道)

Transformer 时代:
  TimeSformer (2021): factorized space-time attention
  ViViT (2021): tubelet embedding (3D patch)

自监督时代:
  VideoMAE (2022): mask 90% 时空 token, 学时空预测

数据:
  Ego4D (2022): 3670h 第一人称视频, robot 视觉预训练的核心数据源
```

**对 robotics 的直接影响**:

```
视频表征 → 机器人视觉预训练:
  Ego4D (人类第一人称视频)
    → R3M (2022): time-contrastive learning → robot 视觉表征
    → VIP (2023): 视频 embedding 距离 = value function → 自动 reward

这条线对你最重要:
  不是教机器人"看视频", 而是"从人类视频中学会看世界"
  R3M 证明: 人手抓杯子的视频特征 ≈ 机器人抓杯子的视频特征
  VIP 进一步: 视频中"越来越接近目标"的 embedding 变化 = reward signal
```

**GR00T N1 的 SlowFast 类比**: GR00T N1 的 VLM (10Hz) + DiT (120Hz) 双系统, 概念上直接来自 SlowFast Networks -- 慢通道理解语义 (什么任务), 快通道处理细节 (怎么执行)。这是 CV 视频处理思想在机器人架构中的直接体现。

**本库对应**: `CV/6_video/` (TimeSformer, ViViT, VideoMAE, Ego4D) + `robotics/visual_repr/` (R3M, VIP)

---

## 3. CV 技术迁移到 Robotics 的完整图谱

### 3.1 已完成的迁移

| CV 技术 | 谁发明的 | Robotics 中的形态 | 代表论文 |
|---------|---------|-----------------|---------|
| CNN backbone (ResNet) | He, Microsoft (2015) | RT-1 的视觉编码器 | RT-1 |
| ViT backbone | Dosovitskiy, Google (2020) | VLA 的视觉编码器 | RT-2, pi_0, GR00T N1 |
| CLIP 视觉-语言对齐 | Radford, OpenAI (2021) | 语言指令理解 + affordance | SayCan, CLIPort |
| VLM (PaliGemma) | Google (2024) | VLA 的视觉-语言 backbone | pi_0, OpenVLA |
| Diffusion 生成 (DDPM) | Ho, Google (2020) | 连续动作生成 | Diffusion Policy |
| DiT (Diffusion+Transformer) | Peebles, Meta (2023) | 动作生成 backbone | GR00T N1 System 1 |
| Flow Matching | Lipman, Meta (2022) | 动作生成 (更快) | pi_0 |
| 对比学习 (MoCo/SimCLR) | He/Chen (2020) | 视觉预训练 | R3M 的 time-contrastive |
| 掩码图像建模 (MAE) | He, Meta (2021) | 视觉预训练 | VideoMAE → 视频表征 |
| 第一人称视频 (Ego4D) | Meta (2022) | robot 视觉预训练数据 | R3M, VIP 的数据源 |

### 3.2 正在迁移的技术

| CV 技术 | 来源 | Robotics 潜在应用 | 当前状态 |
|---------|------|-------------------|---------|
| DINOv2 自监督特征 | Meta (2023) | 无需标注的 robot 视觉 backbone | 初步实验, 部分论文用 DINOv2 替代 CLIP |
| SAM 通用分割 | Meta (2023) | 零样本物体分割→抓取区域检测 | 部分集成到 robot perception pipeline |
| 单目深度 (Depth Anything) | HKU (2024) | 单 RGB 摄像头获取 3D 信息 | 部分机器人开始用, 但精度待验证 |
| BLIP-2 Q-Former | Salesforce (2023) | 视频理解→任务语义提取 | Video LLM 用于 robot task description |
| 3D Gaussian Splatting | Inria (2023) | 实时场景重建→manipulation planning | 正在研究, 还未大规模部署 |
| LDM/Stable Diffusion | Rombach (2022) | 深度估计 (Marigold), 数据增强 | Marigold 用 SD 做深度, 效果好 |

### 3.3 值得关注但尚未迁移的技术

| CV 技术 | 为什么值得关注 |
|---------|-------------|
| **SwinTransformer 层级式 attention** | 当前 VLA 用的 ViT 是全局 attention, 对高分辨率图像计算量大。Swin 的窗口 attention 可能让 VLA 处理更高分辨率输入 |
| **VideoMAE 的极高 mask 比例 (90-95%)** | 如果 robot 视频也能 mask 90% 还原, 说明视频中大部分信息是冗余的 -- 可以大幅压缩 robot observation |
| **DETR 端到端检测** | 当前 robot 的物体检测还在用传统 pipeline (Faster R-CNN + NMS)。DETR 的 set prediction 可能更适合 robot 的多物体场景 |
| **Tubelet embedding (ViViT)** | 将视频体积直接切成 3D token, 保留时空局部性。对 robot 视频输入的 tokenization 可能比逐帧 ViT 更高效 |
| **VIP 的 implicit value from video** | 自动从视频生成 dense reward, 解决 robot RL 的 reward engineering 痛点。目前只在简单任务上验证, 复杂任务待探索 |

---

## 4. CV vs LLM: 对 Robotics 的不同贡献

| 维度 | LLM 教你的 | CV 教你的 |
|------|-----------|----------|
| **核心问题** | 怎么理解语言指令、怎么规划 | 怎么看世界、怎么理解空间 |
| **对 robot 的贡献** | 高层语义 (什么任务) | 低层感知 (物体在哪、深度多少) |
| **Scaling 经验** | 数据/模型 power-law | 自监督预训练 (不需标注) |
| **生成经验** | autoregressive token 生成 | diffusion/flow 连续值生成 |
| **时间建模** | 序列建模 (因果注意力) | 视频理解 (时空注意力) |
| **3D 理解** | 无 (LLM 不懂 3D) | NeRF/3DGS/深度估计 |
| **训练范式** | pre-train → RLHF → 产品 | pre-train → self-supervised → domain-specific |

**关键结论**: LLM 给了 robot "大脑" (理解指令、做规划), CV 给了 robot "眼睛" (看世界、理解空间)。当前 VLA 的架构正是这两者的融合:

```
pi_0 = PaliGemma (CV: VLM backbone) + Flow Matching (CV: 生成模型) + LLM reasoning
GR00T N1 = Eagle VLM (CV+LLM) + DiT (CV: 生成模型)
```

**但 CV 的贡献被低估了**: 当前的讨论过于集中在 "LLM 如何用于 robot", 忽视了 robot 的核心瓶颈往往是**视觉感知** -- 看不准物体位置比理解不了语言指令更致命。

---

## 5. 对你 (从 CV 学习做机器人基础模型) 的建议

### 5.1 优先学什么

```
=== 第一层: 理解视觉 backbone 演进 (必读) ===
1. ResNet → ViT: 从 CNN 到 Transformer 的视觉表征变革
2. MAE / DINOv2: 自监督视觉预训练 -- robot 数据没有标注, 必须自监督

=== 第二层: 理解生成模型 (必读, 直接关联 robot action generation) ===
3. DDPM → Flow Matching: 连续分布生成的两种范式
4. DiT: Transformer 做 diffusion backbone -- GR00T N1 直接用

=== 第三层: 理解视频时空表征 (核心, 你最关注的) ===
5. TimeSformer / ViViT: 怎么把 Transformer 从图像扩展到视频
6. VideoMAE: 怎么用自监督从视频学时空表征
7. Ego4D → R3M → VIP: 人类视频 → robot 视觉表征 → 自动 reward

=== 第四层: 理解 3D 视觉 (操作任务的基础) ===
8. NeRF → 3DGS → Depth Anything: 3D 表示的三个时代
```

### 5.2 一句话总结

**LLM 用 8 年走完了从 GPT-1 到 ChatGPT 的路。CV 也用了类似的时间从 AlexNet 走到 DINOv2/SAM, 但 CV 的路更"安静" -- 没有 ChatGPT 那样的产品爆发, 但技术积累同样深厚。对机器人而言, CV 的自监督视觉预训练 (MoCo→MAE→DINOv2) 和视频时空表征 (VideoMAE→R3M→VIP) 可能比 LLM 的 RLHF 更直接有用 -- 因为机器人的第一个问题永远是"我看到了什么", 而不是"我理解了什么指令"。**
