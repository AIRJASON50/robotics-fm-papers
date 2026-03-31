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
