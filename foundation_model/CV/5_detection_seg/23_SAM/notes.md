# Segment Anything (SAM) -- 学习笔记
> 一句话: 定义 promptable segmentation task, 通过 model-in-the-loop data engine 自动标注 1.1B masks, 训练出分割领域的 foundation model -- 一个模型通过 prompt engineering 解决所有分割任务。
> 论文: Alexander Kirillov, Eric Mintun, Nikhila Ravi et al. (Meta AI, FAIR), ICCV 2023
> 引用量级: ~10000+

## 这篇论文解决了什么问题
NLP 有 GPT (用 next-token prediction 统一所有语言任务 + prompt engineering 做 zero-shot transfer), CV 的分割领域能否也有类似的 foundation model? 之前的分割模型都是 task-specific 的: semantic segmentation, instance segmentation, panoptic segmentation 各有各的模型和数据集, 没有统一框架。而且分割标注极其昂贵 (不像文本数据可以从 web 爬取), 最大的分割数据集也只有几百万 masks。

## 核心想法 (用直觉解释)
三个组件形成闭环: (1) Task -- 定义 "promptable segmentation": 给模型任意 prompt (点击、框、文字、粗略 mask), 模型输出 valid segmentation mask。这就像 GPT 接受任意 text prompt 后生成回答。(2) Model -- 三件套架构: 重量级 image encoder (MAE pre-trained ViT-H) 每张图只跑一次, 轻量级 prompt encoder + mask decoder (2 个 Transformer blocks) 实时响应不同 prompts, ~50ms on CPU。(3) Data -- 用模型自身当标注工具, 三阶段 data engine: 人工辅助标注 -> 半自动 -> 全自动, 从 120k 图像 4.3M masks 扩展到 11M 图像 1.1B masks (SA-1B), 比之前最大数据集大 400 倍。

## 关键设计决策
- **Promptable segmentation task**: 将所有分割任务统一为 "prompt -> mask"。关键是 ambiguity-aware: 一个 prompt (如点击衬衫) 可能对应多个合理分割 (衬衫 / 穿衬衫的人 / 上半身), 模型输出 3 个 masks + confidence scores, 取最高置信度的那个。这让同一个模型既能做 interactive annotation, 也能做 zero-shot transfer
- **Image encoder 和 prompt/mask decoder 解耦**: image embedding 可以预计算和缓存 (heavweight ViT-H, ~0.6s), prompt encoder + mask decoder 实时运行 (~50ms on CPU)。同一张图的 embedding 可被不同 prompt 反复查询 -- amortized real-time
- **Data engine 的三阶段进化**: (1) Assisted-manual: 人用 SAM 交互式标注, 34s/mask, 收集 4.3M masks, 训练第一版 SAM; (2) Semi-automatic: SAM 自动检测 confident objects, 人工标注剩余的 less prominent objects, mask 数量从 44 增到 72/image; (3) Fully automatic: 32x32 grid of points 作为 prompt, 自动生成 ~100 masks/image, NMS 去重, 生成 1.1B masks。模型训了 6 次, encoder 从 ViT-B 逐步升级到 ViT-H
- **Mask quality 验证**: 500 张图的 ~50k masks 让人工精修后对比, 94% masks 的 IoU > 90% -- 自动标注质量接近人工

## 这篇论文之后发生了什么
- **SAM 2 (2024)**: 扩展到视频分割, 支持 video object tracking with prompts
- **Grounded-SAM**: SAM + Grounding DINO, 用 text prompt 检测 + 分割, 实现 open-vocabulary segmentation
- **Robot manipulation 中的应用**: 用 SAM 做 object-centric segmentation, 为 policy 提供 clean object mask 作为 visual input
- **DINOv2 + SAM 工具链**: DINOv2 提供 semantic features, SAM 提供 fine-grained masks, 互补使用

## 对你 (RL->FM) 的 Takeaway
| # | Takeaway | 与你的关联 |
|---|----------|-----------|
| 1 | Promptable task = foundation model 的预训练目标: NLP 有 next-token, segmentation 有 promptable mask | Robotics 的 "promptable task" 是什么? 可能是 goal-conditioned policy (goal image/text 作为 prompt) |
| 2 | Data engine > 一次性标注: model-in-the-loop 的数据飞轮是 SAM 真正的护城河 | Robot data collection 也可以做 data engine: 用当前 policy 辅助 teleoperation -> 收集数据 -> retrain -> 循环 |
| 3 | Heavyweight encoder + lightweight decoder 的部署设计 | Robot 部署: visual backbone 异步/低频运行 (如 5Hz), action head 高频推理 (如 50Hz) |
| 4 | Ambiguity-aware output: 同一 prompt 输出多个 valid masks + confidence ranking | Robot policy 的 multi-modality 问题 (同一 observation 有多种 valid actions) 可输出多个候选 + 排序 |
