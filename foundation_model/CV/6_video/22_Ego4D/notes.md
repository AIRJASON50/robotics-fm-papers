# Ego4D: Around the World in 3,000 Hours of Egocentric Video -- 学习笔记

> 一句话: 3,670 小时大规模第一人称视频数据集 + 五项 benchmark, 定义了 egocentric perception 的研究范式。
> 论文: Kristen Grauman et al. (Facebook AI + 13 所大学, 88 位作者), 2022, CVPR 2022

## 这篇论文解决了什么问题

现有 CV 数据集 (ImageNet, COCO, Kinetics) 都是第三人称 "旁观者" 视角: 摄影师主动拍摄的短片段。但 **robotics 和 AR 的输入是第一人称视角** -- 长时间连续、未经筛选的视频流, 需要理解手物交互、3D 空间、长时记忆和社交互动。

之前的 egocentric 数据集 (EPIC-Kitchens <100h, 主要是厨房, 拍摄者多为研究生) 规模和多样性都不够。第一人称视频有三个根本不同: (a) 连续的流式视频而非精选片段; (b) 需要 3D 空间理解; (c) 需要解读人-物交互和社交语境。

## 核心想法 (用直觉解释)

**构建一个足够大、足够多样、足够真实的第一人称视频数据集, 并围绕 "第一人称视觉体验的 past/present/future" 定义五项 benchmark。**

数据集规模: 3,670 小时 | 931 佩戴者 | 74 个地点 | 9 个国家 | 非脚本日常活动

多模态: RGB (3,670h) + 文本旁白 (3.85M 句, 13.2 句/分钟) + 音频 (2,535h) + 3D mesh (491h) + 立体视觉 (80h) + 眼动 (45h) + IMU (836h) + 多机位 (224h)

五项 Benchmark:
1. **Episodic Memory** (过去): "我把 X 放哪了?" -- 在数小时视频中定位过去事件
2. **Hands & Objects** (现在): object state change detection (pre -> PNR -> post)
3. **Audio-Visual Diarization** (现在): 谁在说话? 在哪?
4. **Social Interaction** (现在): 谁在看我? 谁在跟我说话?
5. **Forecasting** (未来): 接下来会发生什么?

## 关键设计决策

**1. 分布式数据收集确保多样性**

14 个团队, 9 国, 5 大洲。参与者通过社区招募 (不是研究生): 面包师、木匠、园丁、修车工... 年龄 18-80+, 45% 女性。每人佩戴 1-10 小时, 非脚本。7 种不同头戴相机 (GoPro, Vuzix 等) 避免对单一设备 overfitting。

**2. Narration 作为基础标注层**

所有视频先做 pause-and-talk narration: 标注员看 5 分钟视频, 逐句描述佩戴者动作。平均 13.2 句/分钟, 总计 380 万句, 1,772 个动词 + 4,336 个名词。三重用途: (a) 构建动作/物体 taxonomy; (b) 按内容分配视频到 benchmark; (c) 作为弱对齐的 video-language 数据。

**3. Hands & Objects -- 对 robot 最直接相关**

定义 object state change: 同一结果 (木头切成两半) 可由多种方式达到 (不同工具/力/速度)。标注三关键帧: pre-condition, point-of-no-return (PNR), post-condition。抽象操作本质: 不是 "做了什么动作" 而是 "物体发生了什么变化"。

**4. Episodic Memory -- 长时视频理解新范式**

三种查询: (a) NLQ (自然语言): "我把剪刀放哪了?"; (b) VQ (视觉): 给物体照片, 找上次出现时间和位置; (c) MQ (moment): "我什么时候做饭?" 需要在数小时视频中做时空定位, 远超现有 short-clip QA。74K 总查询, 800 小时标注视频。

**5. 多模态 + 3D 的 robot 关联**

部分视频配有 Matterport3D 环境扫描, 可关联动态视频和静态 3D 场景 (Fig. 4)。IMU + 立体视觉 + 眼动追踪的模态组合对应 robot 传感器配置。

## 这篇论文之后发生了什么

- **Ego4D 挑战赛** (2022 起每年): 推动 episodic memory, hand-object, forecasting 方向快速进步
- **Ego-Exo4D (2024)**: 同一事件的第一/第三人称同步视频, 解决 ego-exo domain gap
- **R3M, VIP**: 在 Ego4D 上做 video contrastive / value function pre-training, 成为 robot visual representation 工作的数据基础
- **EgoVLP, LaViLa**: 在 Ego4D narration 上做 video-language pre-training
- **Robot learning from human video**: Ego4D 提供大规模 "人在真实场景操作物体" 视频, 是 imitation from human video 的数据来源

## 对你 (RL->FM) 的 Takeaway

| # | Takeaway | 与你的关联 |
|---|----------|----------|
| 1 | **Egocentric video 是 robotics 最接近的 "人类数据源"** -- 第一人称、手物交互、长时连续 | 你做灵巧手 sim2real, Ego4D hand-object 数据可预训练 visual encoder 或理解人类操作策略 |
| 2 | **Object state change 比 action label 更本质** -- 不关心怎么做, 关心物体变成了什么 | Manipulation 目标是改变物体状态; pre/PNR/post 框架可用于 reward 定义或 success detection |
| 3 | **长时 episodic memory 是 robot autonomy 的关键** -- "我上次在哪见过这个工具?" | 长时运行的 robot 需要记住之前做过什么、物体在哪, 避免重复搜索 |
| 4 | **数据多样性比数据量更重要** -- 931 人 x 74 地点 x 多种职业 > 一个实验室拍万小时 | Robot 数据的 diversity (不同物体/场景/任务) 比单纯增加 demo 数量更能提升泛化 |
| 5 | **Narration 是低成本弱监督信号** -- 自然语言描述 > 精确 bbox/action label | Robot demo 可加 narration: 让标注员描述"在做什么", 成本低且信息更丰富 |
