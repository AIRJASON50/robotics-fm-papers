# MinBC: Coordinated Humanoid Manipulation with Choice Policies - 论文笔记

**论文**: Coordinated Humanoid Manipulation with Choice Policies
**作者**: Haozhi Qi*, Yen-Jen Wang* (共同一作), Toru Lin, Brent Yi, Yi Ma, Koushil Sreenath, Jitendra Malik
**机构**: UC Berkeley
**发表**: arXiv:2512.25072, 2025
**代码**: https://github.com/x-robotics-lab/minbc

---

## 一句话总结

为人形机器人全身操作设计模块化遥操作接口 + Choice Policy (多候选动作 + 得分选择)，解决 BC 的 mode averaging 问题，同时保持推理速度，在洗碗机装载等任务上超越标准 BC 和 Diffusion Policy。

---

## 核心问题

1. **遥操作 44+ DOF 人形机器人太难**: 现有系统通常只控制上半身或缺少主动头部控制
2. **BC 的 mode averaging**: 多模态动作被平均化 → 不稳定行为
3. **Diffusion Policy 太慢**: 无法满足实时全身控制 (需要多步去噪)

---

## 方法

### 模块化遥操作
- **手臂**: VR 控制器相对位姿 → IK，按需激活 (扳机键)
- **手部**: 四指一组 (握把按钮) + 拇指独立 (摇杆)
- **手眼协调**: 按钮触发头部追踪左/右手 (arctan2 计算 yaw/pitch)
- **步行**: RL locomotion policy (100Hz) + 摇杆方向

### Choice Policy (核心贡献)
灵感来自 multi-choice learning (SAM 多掩码预测)：
1. **特征编码**: DINOv3 (frozen) + ResNet-18 (depth, GroupNorm) + MLP (proprioception) → 拼接
2. **K=5 个动作提案**: 每个 $a_k \in \mathbb{R}^{T \times |A|}$
3. **得分网络**: 预测每个提案的 MSE
4. **Winner-takes-all 训练**: 只更新 MSE 最小的提案；得分网络回归预测所有提案的 MSE
5. **推理**: 选得分最低的提案执行，单次前向传播

---

## 关键结果

**洗碗机装载 (100 demos, 10 trials)**:

| 方法 | Pickup | Handover | Insertion |
|------|--------|----------|-----------|
| BC + hand-eye | 10/10 | 7/10 | 5/10 |
| Diffusion + hand-eye | 10/10 | 7/10 | 5/10 |
| **Choice + hand-eye** | **10/10** | **9/10** | **7/10** |
| 无 hand-eye (所有方法) | - | - | 0-1/10 |

- **手眼协调是 insertion 成功的必要条件**
- Diffusion Policy 在 loco-manipulation 中因速度/稳定性问题无法部署

---

## 作者展望

1. 视觉感知泛化需更多样化数据或大规模预训练
2. 可学习的自适应注视机制替代启发式手眼协调
3. 长链全身操作端到端成功率仍需提高

---

## 代码 vs 论文差异

| 项目 | 论文 | 代码 |
|------|------|------|
| 动作解码器 | "two-layer MLP" | **CondHourglassDecoder** (1D UNet + FiLM 调制，默认架构) |
| 归一化 | 未提及 | 双重归一化：obs 用 P2/P98 百分位，action 用 minmax |
| 模型推理 | 未提及 | **EMA 模型** (power=0.75) 用于推理 |
| 动作平滑 | 未提及 | **MovingAverageQueue** (k=20, tau=0.7)，手部动作排除在外 |
| Score loss | 未提及 | 可选 clipping (max=10.0) |
| LR 策略 | 未提及 | Cosine + 500 步 warmup，per-batch step |
| 多 GPU | 未提及 | 手动 all_reduce 梯度 (非 DDP) |

### 值得学习的代码设计

1. **CondHourglassDecoder 的 cls_token**: 每个 proposal 从不同可学习初始化开始，通过 FiLM 调制产生差异化输出 → 优雅的多提案生成
2. **Policy-type 统一接口**: `num_proposal=1` 是 BC，`num_proposal=5` 是 Choice，`policy_type="dp"` 是 Diffusion → 共享编码器和训练循环
3. **SafetyWrapper**: delta clipping (臂 0.5, 手指 0.1) 作为硬安全层
4. **Per-timestep .pkl 数据格式**: 内存友好，支持不同长度 episode

---

## 非显而易见的洞察

1. **手眼协调是长链操作瓶颈**: 不是抓取失败而是看不到目标导致 insertion 失败
2. **Proposal heads 自动特化**: 不同 head 在不同任务阶段被选中 → mixture of experts 涌现
3. **Diffusion Policy 在 loco-manipulation 完全不可用**: 不仅速度问题，训练也不稳定 → 重要负面结果
