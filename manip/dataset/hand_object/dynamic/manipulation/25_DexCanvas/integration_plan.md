# DexCanvas 数据集研究与集成计划

## 任务概述

将 arXiv:2510.15786 (DexCanvas) 论文拉取并转换为 Markdown，克隆开源代码，研究数据集格式，并规划与 WujiMJX RL workspace 的集成方案。

---

## 第一阶段：论文与代码获取

### 1.1 拉取论文并转换为 Markdown

```bash
cd /home/l/ws/doc/paper/html2aitext_convert
./arxiv2md.sh 2510.15786
```

**预期输出**: `output/DexCanvas_*.md`

### 1.2 创建论文文件夹并整理

```bash
mkdir -p /home/l/ws/doc/paper/DexCanvas
cp output/DexCanvas*.md /home/l/ws/doc/paper/DexCanvas/paper.md
```

### 1.3 克隆开源代码

```bash
cd /home/l/ws/doc/paper/DexCanvas
git clone https://github.com/dexrobot/dexcanvas.git code
```

---

## 第二阶段：数据集格式与组成研究

### 2.1 代码库结构分析

探索克隆后的代码库，重点关注：
- 数据加载脚本 (dataloader, dataset classes)
- 数据格式定义 (schema, data structures)
- 示例/教程代码

### 2.2 数据集内容分析

**根据论文摘要，DexCanvas 包含**：

| 数据类型 | 描述 | 格式（待确认） |
|----------|------|----------------|
| RGB-D 视频 | 多视角同步录制 | 可能是 .mp4/.avi + 深度 |
| MANO 参数 | 人手运动捕捉 | 可能是 .pkl/.npy |
| 接触点 | 每帧接触位置 | 可能是 .json/.npy |
| 力分布 | 物理一致的接触力 | 可能是 .npy |
| 物体状态 | 物体位姿、速度 | 可能是 .npy |

**需要确认的关键信息**：
1. 数据下载方式和存储格式
2. 每个样本的数据结构
3. 21 种操作类型的标注方式
4. Cutkosky taxonomy 的具体分类

### 2.3 输出文档

创建 `/home/l/ws/doc/paper/DexCanvas/data_format.md`，包含：
- 完整的数据格式文档
- 字段定义和维度说明
- 加载示例代码

---

## 第三阶段：WujiMJX 集成方案设计

### 3.1 集成方案概述

**目标**：使用 DexCanvas 数据集与 WujiHand 进行训练/回放

**核心挑战**：
1. **手部形态差异**: DexCanvas 使用 MANO 人手模型，WujiHand 是灵巧机械手
2. **关节映射**: MANO (51 DoF) → WujiHand (具体 DoF 待查)
3. **接触点转换**: 人手接触点 → 机械手指尖/指腹
4. **坐标系对齐**: 可能需要统一参考坐标系

### 3.2 方案 A：轨迹回放 (Retargeting)

**适用场景**: 快速验证、可视化演示

**流程**:
```
DexCanvas MANO → 手部姿态提取 → WujiHand 关节映射 → MuJoCo 回放
```

**实现步骤**:
1. 创建 MANO → WujiHand 关节映射器
2. 加载 DexCanvas 轨迹
3. 在 MuJoCo 中回放重定向后的动作

**需要新建文件**:
- `source/wujihand_tasks/utils/dexcanvas_loader.py` - 数据加载
- `source/wujihand_tasks/utils/mano_retargeting.py` - 姿态重定向
- `scripts/tools/play_dexcanvas.py` - 回放脚本

### 3.3 方案 B：模仿学习 (Imitation Learning)

**适用场景**: 利用演示数据预训练策略

**方法选择**:

| 方法 | 优点 | 缺点 | 复杂度 |
|------|------|------|--------|
| 行为克隆 (BC) | 简单直接 | 分布漂移 | 低 |
| DAgger | 在线纠正 | 需要专家标注 | 中 |
| GAIL/BC+RL | 利用奖励 | 训练复杂 | 高 |

**推荐**: 先用 BC 预训练，再用 RL 微调

**实现步骤**:
1. 将 DexCanvas 轨迹转换为 (obs, action) 对
2. 训练 BC 策略初始化网络
3. 在现有 PPO 训练框架上继续 RL 训练

**需要新建文件**:
- `scripts/pretrain_bc.py` - BC 预训练脚本
- `source/wujihand_tasks/utils/demo_buffer.py` - 演示缓存

### 3.4 方案 C：参考轨迹奖励 (Reference-Motion Reward)

**适用场景**: 将演示融入 RL 奖励

**流程**:
```
DexCanvas 轨迹 → 参考姿态/接触 → 奖励函数增强 → PPO 训练
```

**奖励设计**:
```python
r_total = r_task + α * r_reference

r_reference = (
    w_pose * pose_tracking_reward(q_demo, q_agent) +
    w_contact * contact_similarity_reward(c_demo, c_agent)
)
```

**需要修改文件**:
- `source/wujihand_tasks/tasks/rh_cube_reorient/rewards.py` - 添加参考奖励
- `source/wujihand_tasks/tasks/rh_cube_reorient/config.py` - 添加参考轨迹配置

---

## 第四阶段：实现优先级

### 4.1 推荐实施顺序

1. **Phase 1: 数据探索** (本计划)
   - 克隆代码、下载示例数据
   - 理解数据格式
   - 编写基础加载器

2. **Phase 2: 轨迹回放**
   - 实现 MANO → WujiHand 重定向
   - 可视化验证

3. **Phase 3: 模仿学习/RL 集成**
   - 选择具体方法
   - 实现并训练

### 4.2 关键文件清单

| 阶段 | 文件 | 用途 |
|------|------|------|
| 数据探索 | `DexCanvas/data_format.md` | 数据格式文档 |
| 数据探索 | `utils/dexcanvas_loader.py` | 数据加载器 |
| 回放 | `utils/mano_retargeting.py` | 姿态重定向 |
| 回放 | `tools/play_dexcanvas.py` | 回放脚本 |
| 训练 | `scripts/pretrain_bc.py` | BC 预训练 |
| 训练 | `rewards.py` (修改) | 参考奖励 |

---

## 验证计划

### 数据加载验证
```python
from wujihand_tasks.utils.dexcanvas_loader import DexCanvasDataset
dataset = DexCanvasDataset(path="...")
sample = dataset[0]
print(sample.keys())  # 验证数据结构
```

### 重定向验证
```bash
python scripts/tools/play_dexcanvas.py --demo_id=0 --visualize
# 目视检查动作是否合理
```

### 训练验证
```bash
# BC 预训练后，验证 loss 收敛
# RL 微调后，验证 success_rate 提升
```

---

## 用户确认的方案

- **集成深度**: ✅ 全部方案（依次实现回放 → BC预训练 → 参考轨迹RL）
- **聚焦任务**: ✅ 精细操作（抓取、旋转类）

---

## 执行步骤总结

### Step 1: 论文与代码获取
```bash
# 1.1 转换论文为 Markdown
cd /home/l/ws/doc/paper/html2aitext_convert
./arxiv2md.sh 2510.15786

# 1.2 创建论文文件夹
mkdir -p /home/l/ws/doc/paper/DexCanvas
cp output/DexCanvas*.md /home/l/ws/doc/paper/DexCanvas/paper.md

# 1.3 克隆代码
cd /home/l/ws/doc/paper/DexCanvas
git clone https://github.com/dexrobot/dexcanvas.git code
```

### Step 2: 数据格式研究
- 分析 `code/` 目录结构
- 识别数据加载脚本和格式定义
- 筛选与精细操作（抓取、旋转）相关的数据子集
- 输出 `data_format.md` 文档

### Step 3: 轨迹回放实现
- 创建 `utils/dexcanvas_loader.py`
- 创建 `utils/mano_retargeting.py` (MANO → WujiHand 映射)
- 创建 `tools/play_dexcanvas.py` (MuJoCo 回放脚本)
- 验证：可视化检查重定向动作

### Step 4: BC 预训练实现
- 创建 `utils/demo_buffer.py` (演示数据缓存)
- 创建 `scripts/pretrain_bc.py` (行为克隆训练)
- 验证：loss 收敛、策略在开环上合理

### Step 5: 参考轨迹 RL 集成
- 修改 `rewards.py` 添加参考奖励项
- 修改 `config.py` 添加参考轨迹配置
- 验证：success_rate 相比纯 RL baseline 提升
