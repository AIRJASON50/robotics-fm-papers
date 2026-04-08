# 反向传播与多层感知机 -- 神经网络的创世

> Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors", Nature 1986
> 这是确立"神经网络能自动学习"的论文。之后所有深度学习都是这个范式。

---

## 神经网络的灵感来源: 从生物神经元到数学模型

```
NN 的每一步发展都来自神经科学/心理学, 不是纯数学:

1943 McCulloch-Pitts 神经元:
  → 神经科学家 McCulloch + 数学家 Pitts
  → 直接模仿生物神经元:
    生物: 树突接收电信号 → 细胞体求和 → 超过阈值则沿轴突发放脉冲
    模型: 输入加权求和 → 超过阈值输出 1, 否则 0
  → y = 1 if (Σ w_i * x_i > threshold) else 0
  → 这个"加权求和 + 阈值"就是一个神经元, 至今未变
  → 2025 年 Transformer FFN 的每个神经元还是这个结构 (只是阈值换成了 SiLU)

1949 Hebb 学习规则:
  → 心理学家 Hebb: "neurons that fire together wire together"
  → 一起激活的神经元之间连接加强
  → 这是"学习"的生物学直觉: 经常一起出现的 pattern → 连接变强

1958 感知机 Perceptron:
  → 心理学家 Rosenblatt (Cornell)
  → 在 McCulloch-Pitts 神经元上加了赫布式学习: 权重根据数据自动调整
  → 第一次: 机器从数据中学习 (不需要人设权重)
  → 但: 单层 → 只能线性分割 → Minsky 1969 证明做不了 XOR → AI 寒冬

1986 反向传播 + MLP:
  → 认知心理学家 Rumelhart (UCSD) + 心理学家 Hinton (Edinburgh) + Williams
  → 三个人都不是纯 CS, 都有认知科学背景
  → 论文说 "networks of neurone-like units" → 仍用生物神经元类比
  → 多层网络 + 反向传播 → 能做 XOR → 能学任何 pattern
  → 但论文最后诚实承认: 反向传播不是大脑的真实学习机制

关键 insight:
  NN 的灵感是生物神经元 (加权求和 + 阈值/激活)
  学习的灵感是赫布规则 (一起激活 → 连接加强)
  但实现方式 (反向传播/梯度下降) 是数学的, 不是生物的
  → "灵感来自大脑, 实现不模仿大脑"
  → 这种"借生物直觉做数学优化"的模式贯穿了 Hinton 的所有工作
```

---

## 它解决了什么

1958 年 Rosenblatt 的感知机 (Perceptron) 证明了单层神经元可以学习,
但 1969 年 Minsky 证明了单层感知机连 XOR 都做不了 → AI 第一次寒冬。

问题不是"多层能不能解" (理论上能), 而是**多层网络的权重怎么调**:
- 输出层的权重好调: 直接看输出和真值的差
- 隐藏层的权重怎么调? 隐藏层没有"真值", 不知道该往哪调

## 这篇论文做了什么

**反向传播 (Backpropagation)**: 用链式法则从输出层的 loss 逐层反推每个权重该怎么调。

```
前向传播 (计算输出):
  input → W1 → 激活 → W2 → 激活 → ... → output
  
loss = ||output - 真值||^2

反向传播 (计算梯度):
  ∂loss/∂W_最后一层 → 直接算
  ∂loss/∂W_倒数第二层 → 链式法则: ∂loss/∂output × ∂output/∂hidden × ∂hidden/∂W
  → 逐层反推, 每个权重都有了"该往哪调"的方向
  
更新权重:
  W = W - 学习率 × ∂loss/∂W
```

## 为什么这是创世级别的

这篇论文同时确立了三件事, 至今未变:

```
1. 多层网络 (MLP):
   input → 线性变换(W@x+b) → 非线性激活 → 线性变换 → 非线性激活 → output
   → 2025 年 Transformer 的 FFN 还是这个结构

2. 反向传播:
   从 loss 逐层反推梯度 → 更新每个权重
   → 2025 年 PyTorch/JAX 的 autograd 还是这个算法

3. "学习表征" (论文标题就是 "Learning representations"):
   隐藏层自动学到了有用的中间表示
   → 不需要人设计特征
   → 2025 年 VLM 的 backbone 自动学到视觉语言表征还是这个原理
```

## MLP 的几何直觉

```
一个神经元: output = 激活(w · input + b)
  w 定义了一个方向 (超平面法线)
  b 定义了一个阈值 (超平面位置)
  激活函数 在超平面一侧通过, 另一侧抑制
  → 一个神经元 = 把空间切一刀

N 个神经元 = N 刀切空间 → 2^N 个区域
  每个区域对应一种"激活模式" (哪些神经元开, 哪些关)
  → O(N) 参数, O(2^N) 种区分能力 (Bengio 2012 的形式化)

MLP 的一层: Linear(d_in → d_hidden) → 激活 → Linear(d_hidden → d_out)
  升维: 更多检测器, 更多切割
  激活: 折叠空间 (把某些区域压扁)
  降维: 压缩结论

多层 MLP: 每层折叠一次 → 多层 = 反复折叠 → 能逼近任意弯曲的流形
  → Universal Approximation Theorem: 理论上一层无限宽的 MLP 就够
  → 实践上深层窄的 MLP 比浅层宽的更高效 (特征复用, Bengio 2012)
```

## 非线性激活函数: 从 sigmoid 到 SiLU

```
激活函数本身是固定的, 不学习。学习的是 W (决定"喂给激活函数什么输入")。

  sigmoid (1986): 1/(1+e^(-x))     → 平滑, 但深层梯度消失 (饱和区梯度≈0)
  ReLU (2010):    max(0, x)         → 简单, 正区间梯度=1, 解决了梯度消失
  GELU (2016):    x * Φ(x)          → 平滑版 ReLU, GPT/ViT 使用
  SiLU (2017):    x * sigmoid(x)    → 类似 GELU, Llama/Gemma 使用

  核心作用不变: 正值通过, 负值抑制 → 折叠空间 → 提供非线性表达能力
  → 选哪个差异很小 (1-2%), 不影响理解
```

## MLP 在现代架构中的位置

```
1986 MLP:        input → MLP → output (独立处理)
2012 CNN:        input → 卷积(局部信息收集) → MLP → output  
2017 Transformer: input → Attention(全局信息收集) → MLP(FFN) → output

MLP 从未被替代 — 只是"信息收集方式"在变:
  卷积: 看局部邻居
  RNN: 看历史序列  
  Attention: 看所有 token

收集完信息后, 都用 MLP 做非线性变换 (表征/压缩/检测 pattern)
→ MLP 是深度学习的"原子操作"
→ Attention 是"信息路由", MLP 是"信息处理"
→ 2025 年 pi_0 的 FFN 和 1986 年的 MLP 本质完全一样
```

## 论文中被低估的发现 (原文核实)

### 隐藏层自动学到了 disentangled representation (Fig. 1 & Fig. 4)

这是最早的"NN 自动学到有意义表征"的可视化证据。

**Fig. 1 对称检测任务**:
```
输入: 一维二值数组 (如 [1,0,0,1] 或 [0,1,1,0])
任务: 判断是否关于中心对称
网络: 只用 2 个隐藏神经元

训练后权重可视化:
  两个隐藏神经元的权重自动变成了左右对称结构
  → 权重大小相等, 符号相反
  → 没人告诉它"对称=左右相等", 纯粹从数据中被 loss 逼出来的
```

**Fig. 4 家族关系任务** (更惊人):
```
输入: 24 个人名 (one-hot), 来自两个同构的家族树 (英国+意大利)
隐藏层: 6 个神经元
任务: 预测家族关系 (father, mother, uncle, aunt...)
训练: 100 个三元组 (person1, relationship, person2)

可视化 6 个隐藏神经元的权重后发现:
  Unit 1: 主要编码"英国人 vs 意大利人" (国籍)
  Unit 2: 编码"第几代" (辈分)
  Unit 6: 编码"哪个分支" (家族分支)

  → 6 个神经元自动解耦了: 国籍 / 辈分 / 家族分支 等独立因子
  → 这正是 Bengio 2012 形式化的 "disentangled representation"
  → 1986 年就观察到了, 比 Bengio 的理论总结早了 26 年
```

### 自动发现跨域结构相似性 (cross-domain transfer 的最早证据)

```
论文原文:
  "the representation of an English person is very similar to 
   the representation of their Italian equivalent. The network 
   is making use of the isomorphism between the two family trees 
   to allow it to share structure"

两个家族结构完全相同 (同构), 只是名字不同:
  Colin (英国) 和 Alfonso (意大利) 在各自家族中的位置相同
  → 网络自动给他们相似的隐藏层表征
  → 没人告诉它这两个家族有对应关系

这预言了:
  2021 CLIP: 图像和文本在同一空间中对齐 (跨模态共享结构)
  2024 pi_0: 不同机器人的相似操作在 VLM 中得到相似表征 (跨 embodiment)
```

### 论文最后的谦虚与远见

```
论文最后一段原文:
  "The learning procedure, in its current form, is not a plausible 
   model of learning in brains. However, applying the procedure to 
   various tasks shows that interesting internal representations can 
   be constructed by gradient descent in weight-space, and this suggests 
   that it is worth looking for more biologically plausible ways of 
   doing gradient descent in neural networks."

翻译:
  "这个学习过程在当前形式下, 不是大脑学习的合理模型。
   然而, 将它应用于各种任务表明, 梯度下降可以构造出有趣的内部表征,
   这暗示着值得寻找更具生物学合理性的梯度下降方式。"

→ 1986 年就明确说: 反向传播不是大脑的真实机制, 但结果证明它能学到好表征
→ 39 年后, 我们仍然在用反向传播, 仍然没找到"更具生物学合理性"的替代
→ 反向传播可能不是唯一的解, 但至今是最好的解
```

### 泛化能力的早期观察

```
家族关系任务中:
  训练: 100 个三元组 (共 104 个可能的三元组)
  测试: 剩余 4 个没见过的三元组
  
  论文原文:
    "Because the hidden features capture the underlying structure 
     of the task domain, the network generalizes correctly to 
     the four triples on which it was not trained."
  
  → 4/4 泛化成功
  → 因为隐藏层学到了"结构" (因子), 不是死记硬背每个三元组
  → 这就是"压缩 = 泛化"的最早实验证据
```

---

## 1986 之后 MLP 的关键发展

这篇论文确立了 MLP + 反向传播的范式, 之后的发展不是改变 MLP, 而是**让它能训得更深更稳**:

```
万能近似定理 (Cybenko 1989, Hornik 1991):
  理论证明: 一层无限宽的 MLP + 任何非线性激活 → 能逼近任意连续函数
  → 给了 MLP 表达能力的数学保证
  → 但"无限宽"在实践中不可能 → 所以需要深度 (多层) 来代替宽度

梯度消失问题 (1991-2010):
  深层 MLP 的 sigmoid 激活在饱和区梯度≈0 → 深层权重收不到梯度 → 训不动
  → 这是 1990 年代 AI 寒冬的技术原因
  
ReLU (Nair & Hinton, 2010):
  max(0, x) → 正区间梯度恒=1, 不会消失
  → 简单到不可思议, 但直接解决了梯度消失
  → 深层网络突然能训了

BatchNorm (Ioffe & Szegedy, 2015):
  归一化每层的输入分布 → 训练更稳定, 收敛更快
  → 不改变 MLP 结构, 只是在层间加了归一化

ResNet 残差连接 (He et al., 2015):
  output = MLP(input) + input → 梯度可以跳过 MLP 直接流过
  → 训 152 层 → 越深越好, 不再退化
  → 2025 年 Transformer 的每一层都用残差连接

Adam 优化器 (Kingma & Ba, 2015):
  自适应学习率 → 不同参数用不同步长更新
  → 不改变 MLP, 改善了梯度下降的更新策略

Dropout (Srivastava et al., 2014):
  训练时随机关掉部分神经元 → 防止过拟合
  → 但 2025 年大模型几乎不用 (数据足够大, 不会过拟合)
```

**关键认知: MLP 的 "线性→非线性→线性" 结构从 1986 年到 2025 年没有本质变化。**
所有后续工作 (ReLU/BatchNorm/ResNet/Adam) 都是在**让这个基本结构能训得更深更稳**,
而不是发明新的表征方式。Transformer 的 FFN 和 1986 年的 MLP 在数学上完全一样。

---

## 万能近似定理 (Cybenko 1989)

> Cybenko, "Approximation by Superpositions of a Sigmoidal Function", 1989

```
论文原文 (Abstract):
  "finite linear combinations of compositions of a fixed, univariate 
   function and a set of affine functionals can uniformly approximate 
   any continuous function"

数学形式: f(x) ≈ Σ α_j × σ(y_j · x + θ_j)   → 就是一层 MLP

含义:
  一层 MLP + 任意连续 sigmoid 型非线性 + 足够多神经元
  → 能逼近任意连续函数, 精度任意高

几何直觉 (和 Bengio 2012 的联系):
  N 个神经元 = N 个超平面切空间 → 2^N 个区域
  每个区域内是线性的 (激活函数的分段特性)
  无穷多个线性片段拼起来 → 逼近任意曲线
  → 类似微积分: 无穷多小直线段逼近圆

定理的局限:
  只说"存在"这样的权重, 不说"怎么找到"
  不保证: 梯度下降能收敛到最优解
  不保证: 需要多少神经元
  不保证: 需要多少数据

这就是为什么实践中用深度代替宽度:
  一层无限宽 → 理论够但实践不可能
  多层窄的 → 特征复用 → 指数级更高效 (Bengio 2012 的 insight)
  1000 个参数:
    一层: 1000 种表达 (线性增长)
    十层: 每层复用上层 → 10^10 种表达 (指数增长)
  → Transformer 用 12 层而不是 1 层超宽
```

---

## Mixture of Experts 的起源 (Jacobs, Jordan, Nowlan, Hinton 1991)

> "Adaptive Mixtures of Local Experts", Neural Computation 1991
> **Hinton 是 MoE 的共同发明人**

```
原始动机: 不是"模态分离", 而是"任务分解"

  问题: 一个网络同时学所有东西 → 不同 pattern 互相干扰
  解法: 多个小"专家网络" + 门控网络 (gating network)
        门控网络决定: 对于这个输入, 该用哪个专家?
        每个专家只负责输入空间的一个子区域 → 自动分工

  实验: 元音识别任务
    系统自动将任务分解为若干子任务
    每个子任务由一个很简单的专家就能解决
    → "分而治之" 比 "一个网络学一切" 更高效

  核心 insight (论文原文):
    "competitive learning" + "supervised learning" 的结合
    专家之间竞争: 谁做得好谁就负责这类输入
    → 不需要人预定义"哪个专家负责什么"
    → 自动特化
```

```
从 1991 到 2025 的 MoE 演进:

  1991 Jacobs+Hinton:   几个专家, 所有专家都参与 (soft routing)
  2017 Shazeer+Hinton:  数千专家, 稀疏激活 top-k (sparsely-gated MoE, 137B)
  2024 DeepSeek-V3:     256 专家 + 1 共享专家, top-8 激活, 671B
  2025 Kimi-K2:         384 专家 + 1 共享专家, top-8 激活, 1040B

不变的核心:
  条件计算: 不是所有参数都参与每个输入的计算
  门控机制: 可学习的 router 决定激活哪些专家
  自动特化: 不同专家自动学习处理不同类型的 pattern

pi_0 的 2-expert 设计:
  是 MoE 思想的最简形式
  Expert 1 (VLM): 处理视觉语言 pattern
  Expert 2 (Action): 处理动作 pattern
  Router: 硬编码按 token 类型分配 (不需要学习)
  → 从"子任务分工"(1991) 到"模态分工"(2024), 思想一脉相承
```

### MoE 论文中被低估的发现 (原文核实)

**合作 vs 竞争: MoE 的核心设计决策**

```
论文原文 Section 1:
  如果所有专家线性混合 (合作):
    "This strong coupling between the experts causes them to cooperate 
     nicely, but tends to lead to solutions in which many experts are 
     used for each case." → 每个专家学一点, 没有特化

  改成只选一个专家 (竞争):
    "a simpler remedy is to redefine the error function so that the 
     local experts are encouraged to compete rather than cooperate."
    → 赢者通吃 → 强制特化

  → 2025 年延续: Kimi-K2 top-8 = 部分竞争; pi_0 硬编码 = 不竞争直接分配
```

**训练动态: 对称性打破 → 自动特化 (Fig. 3)**

```
论文原文:
  "The system begins in an unbiased state, with the gating network 
   assigning equal mixing proportions to all experts in all cases."

  训练开始: 所有专家均等, 都朝着平均解 (X 点) 移动
  对称性打破: 某个专家碰巧在某类数据上误差更小 → 被分配更多
  → 分化加速: Expert 5 专攻 [i] vs [I], Expert 4/6 专攻 [a] vs [A]
  → 和 RL 中 policy 从随机探索到策略收敛是同一种动态
```

**自动稀疏: 给 8 个专家, 系统只用 3 个**

```
论文原文:
  "in all simulations with mixtures of 4 or 8 experts all but 2 or 
   3 experts had mixing proportions that were effectively 0 for all cases."

  → 不需要人预设"该用几个" → 网络自动决定
  → 2025 Kimi-K2 的 top-8 from 384 是这个思想的工程化放大
```

**MoE 训练速度是 MLP 的两倍, 最终效果一样 (Table 1)**

```
  4 Experts: 1124 epochs → 90% | BP 6 Hidden: 2209 epochs → 90%
  → 准确率完全一样, 但 MoE 收敛快 ~2 倍
  → MoE 优势不是效果更好, 是效率更高 (同等效果更少计算)
```

**认知科学背景: MoE 源于"模块化大脑"假说**

```
作者单位:
  Jacobs, Jordan: MIT Department of Brain and Cognitive Sciences (认知科学)
  Nowlan, Hinton: University of Toronto Computer Science

  一半作者是认知科学家, 不是纯 CS 论文
  MoE 的灵感: 大脑不是统一网络 → 不同脑区各自特化 (视觉/语言/运动)
  → 各自专精, 通过交互协调 → MoE 是这个生物学直觉的计算实现

  Hinton 的背景: 博士是 Edinburgh 的实验心理学 (Artificial Intelligence)
  → 认知科学+计算的交叉背景贯穿了他所有工作
  → 反向传播 (1986): 受赫布学习 (Hebbian learning) 启发
  → MoE (1991): 受模块化大脑假说启发
  → Dropout (2014): 受有性生殖中基因组合的随机性启发
  → 每一个发明都有心理学/生物学的直觉根源
```

---

## Hinton 的贡献全景

```
Geoffrey Hinton (1947-)
  博士: Edinburgh, 实验心理学/AI (1978)
  核心信念: "大脑的学习原理可以被计算模拟"
  
  1986: 反向传播+MLP (和 Rumelhart, Williams) → 确立 NN 学习范式
  1991: MoE (和 Jacobs, Jordan, Nowlan) → 模块化/分工
  2006: 深度信念网络 (DBN) → 深度学习复兴
  2012: AlexNet (学生 Krizhevsky) → 深度学习引爆
  2014: Dropout (和 Srivastava) → 正则化
  2014: GAN 的 Goodfellow 是 Hinton 的学生
  2017: 大规模 MoE (和 Google) → MoE 扩展到 NLP
  2018: 图灵奖 (和 Bengio, LeCun)
  2024: 诺贝尔物理学奖 (和 Hopfield)

  → 深度学习的几乎每一个关键节点都有 Hinton 的直接参与
  → 他的心理学背景使他总是从"大脑怎么做"出发设计算法
  → 反向传播不是数学推导出来的, 是"大脑可能在做类似的事"的直觉
```

---

## 本目录文件索引

```
86_Backpropagation_MLP/
├── note.md                                    ← 本文件
├── Rumelhart_1986_...pdf                      ← 反向传播+MLP 原论文 (Nature, 4 页)
├── Cybenko_1989_...pdf                        ← 万能近似定理 (13 页)
└── Jacobs_1991_...pdf                         ← MoE 原论文 (9 页)
```

---

## 从这篇论文到 VLA 的完整链路

```
1986 反向传播+MLP: NN 能自动学表征
  ↓
2012 AlexNet: 深层 CNN + ReLU + GPU → 证明 scale + depth work
  ↓
2017 Transformer: Attention 替代 CNN 做信息收集, MLP 不变
  ↓
2018 GPT: pre-train + fine-tune 范式
  ↓
2020 ViT: 图像也用 Transformer → 统一了 NLP 和 CV
  ↓
2021 CLIP: 图文对齐 → VLM backbone
  ↓
2023 RT-2: VLM 直接输出动作 → 第一个 VLA
  ↓
2024 pi_0: VLM (共享 attention) + Action Expert (独立 MLP) + Flow Matching
  → 1986 年的 MLP 仍然是 pi_0 的 action head 的核心组件
```
