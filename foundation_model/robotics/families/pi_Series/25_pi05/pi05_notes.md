# pi_0.5: 开放世界泛化的 VLA -- 学习笔记

**论文**: pi_0.5: a Vision-Language-Action Model with Open-World Generalization
**作者**: Physical Intelligence (Kevin Black, Chelsea Finn, Sergey Levine, Karol Hausman et al.)
**发布**: 2025.04

---

## 1. pi_0.5 解决了什么问题

pi_0 只能在训练时见过的场景中工作。pi_0.5 的目标: **让机器人在从未见过的家庭中完成 10-15 分钟的清理任务**。

---

## 2. 核心改进: 一个模型同时输出文字和动作

### pi_0 vs pi_0.5 的根本区别

```
pi_0:
  输入: image + language + state + 噪声
  输出: 只有动作 (flow matching)
  loss: 只有 flow matching loss
  → 只能执行, 不能规划

pi_0.5:
  输入: image + language + state + (噪声 or 无)
  输出: 动作 + 子任务文字 ("拿起盘子")
  loss: flow matching loss (动作) + cross-entropy loss (文字)
  → 能规划 (输出子任务文字) 也能执行 (输出动作)
  → 同一个模型, 两种输出
```

### 层级推理: 同一个模型跑两次

```
论文公式:
  π(a, ℓ̂ | o, ℓ) = π(a | o, ℓ̂) · π(ℓ̂ | o, ℓ)

  第 1 次推理 (高层, 输出文字):
    输入: image + "清理厨房"
    输出: "拿起盘子" (文字 token, cross-entropy 生成)
    → 规划下一步该做什么

  第 2 次推理 (低层, 输出动作):
    输入: image + "拿起盘子" + state + 噪声
    输出: 50 步 action chunk (flow matching 生成)
    → 执行具体动作

  → 不是两个模型, 是同一个模型的两种模式
  → 类似 chain-of-thought: 先想 (预测子任务) 再做 (输出动作)
  → 高层和低层共享参数, 互相增强
```

---

## 3. 两阶段训练: 离散 → 连续

### Stage 1 Pre-training (280k steps): 全部离散, 统一 loss

```
所有数据 (包括 action) 都编码为离散 token:
  web 数据:   image → "a red cup on the table" → 离散文字 token
  语义预测:   image → "拿起盘子" → 离散文字 token
  robot 动作: action → FAST (DCT 压缩) → 离散 action token
  → 全部用 next-token prediction (cross-entropy loss)
  → 和训 GPT 完全一样, 一种 loss, 一种流程

为什么全部离散:
  → 不同数据类型用同一种 loss → 可以无缝混训
  → 直接复用 LLM 的训练基础设施
  → 此阶段 α=0 (flow matching loss 权重为零, 不用 Action Expert)

Pre-training 数据配方 (97.6% 不是目标机器人的):
  MM (移动操作臂, 400h):       目标任务, 只占 2.4%
  ME (非移动臂, 多环境):       不同 embodiment, 但视觉多样性高
  CE (实验室跨 embodiment):     含 Open X, 各种机器人
  HL (高层子任务标注):          教规划能力
  WD (web 图文数据):            保持 VLM 视觉语言理解
```

### Stage 2 Post-training (80k steps): 加入连续 flow matching

```
论文公式 (Eq. 1):
  L = H(text_tokens, predicted_logits)                    ← cross-entropy (文字)
    + α · ||ω - a - f_θ^a(a^τ, o, ℓ)||²                  ← flow matching (动作)

  α = 10.0 (动作 loss 权重是文字 loss 的 10 倍)

这个阶段做的事:
  1. 初始化 Action Expert (从零开始, 随机权重)
  2. 同时训两种 loss:
     cross-entropy: 保持文字预测能力 (子任务/问答)
     flow matching: 训 Action Expert 输出连续动作
  3. 数据: 只用 MM + ME (目标相关) + WD (保持 VLM) + HL + VI (口头指令)
     去掉了 CE (实验室数据, 和目标任务差距太大)

为什么不一开始就用 flow matching:
  → flow matching 只能处理 action, 不能处理 web/语义数据
  → 先用离散 token 把所有知识吃进去 (pre-train)
  → 再加 Action Expert 做精细连续控制 (post-train)
  → 先粗后细
```

---

## 4. 异构数据详解

### 数据类型和各自的作用

| 代号 | 数据类型 | 内容 | 教模型什么 | Pre-train | Post-train |
|------|---------|------|---------|-----------|------------|
| MM | 移动操作臂 | 400h 家庭遥操作 | 目标任务的动作 | ✓ | ✓ |
| ME | 非移动臂多环境 | 多种家庭中的桌面臂 | 视觉多样性 (不同厨房/卧室) | ✓ | ✓ |
| CE | 实验室跨 embodiment | Open X + 自采实验室数据 | 通用操作 pattern | ✓ | ✗ (去掉了) |
| HL | 高层子任务标注 | 人工标注的子任务分解 | 任务规划能力 | ✓ | ✓ |
| WD | Web 图文 | 图像描述/问答/物体定位 | 保持 VLM 视觉语言理解 | ✓ | ✓ |
| VI | 口头指令 | 人走在旁边口头指导 | 从语言指令理解意图 | ✗ | ✓ |

### 其他机器人数据 (CE/ME) 的动作有迁移价值吗?

```
论文做法: 动作直接 zero-pad 到统一维度, 不做 retarget
  所有 action 归一化到 [-1, 1] (用 1%/99% 分位数)
  不同机器人的 action 维度补零对齐

迁移价值:
  视觉语言层: 有 (不同机器人看到的"杯子"是一样的)
  动作层: 可能接近零 (7 DoF 桌面臂的关节角对 18 DoF 移动臂没用)
  → 论文消融显示去掉这些数据性能下降
  → 但下降可能主要来自视觉语言层的损失, 不是动作层
  → 这是 open question
```

### 人类口头监督 (VI) -- 全新的数据采集方式

```
不是遥操作 (人操控关节)
而是: 人走在机器人旁边, 口头说 "现在拿那个碗"
→ 机器人用已有的 low-level policy 自己执行
→ 录下来的是 (语言指令, 动作) 对
→ 训练的是 高层子任务预测 π(ℓ̂|o, ℓ)

优点:
  比遥操作容易得多 (不需要操控设备)
  任何人都能做 (不需要专业操作员)
  采集的是"规划能力"不是"动作能力"
```

---

## 5. Knowledge Insulation (知识隔离)

```
问题: VLM 预训练时认识所有物体, fine-tune 后忘了大部分

双重保护:
  1. 参数层面: VLM 底层冻结 (保护低级视觉特征), 顶层解冻 (适应 robot)
  2. 数据层面: 混入 web 数据持续训练 VLM (保持视觉语言能力)

→ 不只是 "冻结几层", 是参数冻结 + 数据混合的组合
```

---

## 6. 推理流程

```
每 0.5 秒执行一次推理 (50Hz 控制, 每次出 25 步 action chunk):

  Step 1 (高层): 用 4 个相机图像 + 总指令 "清理厨房"
    → 模型输出子任务文字: "拿起盘子" (autoregressive token 生成)
    
  Step 2 (低层): 用 3 个相机图像 + 子任务 "拿起盘子" + 本体感觉 + 噪声
    → 模型输出 50 步 action chunk (flow matching 10 步去噪)
    → 执行前 25 步, 然后重新推理

  机器人: 18-19 DoF (双臂 6 DoF×2 + 夹爪×2 + 移动底盘 3 DoF + 升降 1-2 DoF)
  控制: 直接命令关节目标位姿 + 夹爪 + 底盘速度, 用 PD 控制器跟踪
  → 完全端到端, 没有额外的轨迹规划或碰撞检测
```

---

## 7. 和 pi_0 的完整对比

| 维度 | pi_0 | pi_0.5 |
|------|------|--------|
| 输出 | 只有动作 | 动作 + 子任务文字 |
| Loss | 只有 flow matching | cross-entropy + flow matching |
| 训练阶段 | 一阶段 (全程 flow matching) | 两阶段 (先离散后连续) |
| 推理 | 一次 (直接出动作) | 两次 (先规划子任务, 再出动作) |
| 数据 | robot 数据 + Open X | robot + web + 语义预测 + 口头指令 (5 种) |
| 数据中目标任务占比 | 大部分 | 2.4% |
| 泛化 | 训练场景内 | 从未见过的家庭 |
| 任务长度 | 几十秒 | 10-15 分钟 |

---

## 8. 对你的意义

```
关键 takeaway:
  1. 泛化不靠一个技巧, 靠系统性数据工程 (5 种数据源)
  2. 同一模型能规划+执行 (chain-of-thought 式层级推理)
  3. 离散 pre-train → 连续 post-train 的两阶段策略让杂食混训成为可能
  4. 口头监督是比遥操作更 scalable 的数据采集方式

如果你做灵巧手 VLA:
  → 用 pi_0.5 的思路: 先用离散 token 混训所有数据 (含 web)
  → 再切到 flow matching 精训目标任务
  → 口头指令可能不适用 (灵巧手需要精细动作, 口头说不清)
  → 但异构数据混训的框架可以复用
```
