# GR00T N1.7 Technical Report (from HuggingFace model card + blog + GitHub, no dedicated arxiv paper)

**Sources** (按信息密度排序):
- HuggingFace 模型卡: https://huggingface.co/nvidia/GR00T-N1.7-3B — 最权威, 含完整架构/数据/算力细节
- HuggingFace blog: https://huggingface.co/blog/nvidia/gr00t-n1-7 — 面向开发者, 含 scaling law 图
- GitHub 代码 (N1.7 = main 分支): https://github.com/NVIDIA/Isaac-GR00T
- Forums 早期访问公告: https://forums.developer.nvidia.com/t/early-access-isaac-gr00t-n1-7-open-reasoning-vla-model-for-humanoid-robotics/366916
- **白皮书**: NVIDIA 没为 N1.7 单独发 whitepaper/arxiv, 官方引用的仍是 N1 原论文 **arXiv:2503.14734** (2025.03)

**Date**: 2026.04 早期访问
**Total params**: 3B
**License**: Apache 2.0 (代码) + NVIDIA Open Model License (权重), 商用/研究均可
**Tag**: factory-floor ready

---

## Architecture Changes from N1.6

### VLM (System 2)
- Backbone: **Cosmos-Reason2-2B** (Qwen3-VL architecture)
  - 升级自 N1.6 的 Cosmos-Reason-2B (内部变体)
- 保留 N1.6 的 flexible resolution + native aspect ratio
- 处理 image tokens + language instruction → high-level action tokens
- 任务分解和多步推理仍在 VLM 侧

### 各编码器 (来自模型卡)
- **Vision Encoder**: SigLip2 ViT, 处理 RGB 相机帧
- **Language Encoder**: 预训练 T5 transformer
- **Proprioception Encoder**: MLP, 按 embodiment ID 索引
- **Action Decoder**: **Diffusion Transformer (DiT) + flow matching + AdaLN** (adaptive layernorm 注入时间步)
- Model format: safetensors, BF16

### DiT (System 1)
- **32 层** (与 N1.6 一致)
- 输入: VLM 输出 + live robot proprio state (joint positions, velocities, EEF poses)
- 输出: continuous action vector → 机器人 DoF
- 推理: 4 个 denoising steps (default), single camera view

### Action Representation (核心改动)
- **Relative EEF Action Space** — 把动作表达为"相对当前 pose 的 delta"
- N1.6 已经引入"state-relative action chunks", N1.7 把它统一到 EEF 层面
- **跨 embodiment 的对齐桥梁**: 人类 video 的 hand pose delta 和 robot EEF delta 在同一坐标系 → 可以直接 co-train
- 论文/blog 明确表示这是"关键泛化要素"

### Pipeline 改动
- 新数据处理脚本 `processing_gr00t_n1d7.py`
- ONNX + TensorRT 完整 pipeline 导出 + 频率提升
- 部署支持: NVIDIA Ampere / Hopper / Lovelace / Blackwell + Jetson

---

## Training Data

### EgoScale 人类视频预训练 (核心新增)
- **20,854 小时**人类 egocentric 视频
- 覆盖 20+ 任务类别 (制造业 / 零售 / 医疗 / 家居)
- 包含: ego camera, wrist camera, hand tracking
- 对比 N1.6: 从"几千小时遥操作数据" → "20K 小时人类视频 + 多样 robot demo"

### 关键发现 — Dexterity Scaling Law (论文级新发现)
> "More human egocentric data produces predictable, consistent improvements in dexterous manipulation"

- **1k 小时 → 20k 小时, 平均任务完成率超过 2x**
- **机器人界第一个明确报告的 dexterity scaling law**
- 让 22 DoF 灵巧手能做 contact-rich 任务 (小零件装配)

### Robot 数据
- 继承 N1.6 的多样 robot demo (YAM bimanual, Unitree G1, AGIBot Genie-1, 仿真 Galaxea 等)

### 模型卡披露的总训练数据规模
- **21.6M 数据点, 13 个数据集**
- 采集方式混合: human + robot + simulated
- 标注混合: 人工 + 自动
- Post-training 公开 checkpoint 用的数据集:
  1. **SimplerEnv Bridge** — BridgeData V2, 60,096 轨迹 / 24 环境
  2. **SimplerEnv Fractal** — 同 Bridge
  3. **DROID** — ~76K 轨迹 / ~350 小时 / 564 场景 / 52 建筑 / 86 任务
  4. **LIBERO** — 130 个 language-conditioned manipulation 任务

### 训练算力 (模型卡披露)
- 总训练算力: 200 节点 × 4 GPU (== 800 GPU)
- 功率: 1200W × 120 小时
- 能耗: 41,288 kWh
- 碳排: 16.949 tCO2e
- 测试硬件: A6000

---

## 新能力

| 能力 | N1.6 | N1.7 |
| --- | --- | --- |
| VLM backbone | Cosmos-Reason-2B (内部) | Cosmos-Reason2-2B (Qwen3-VL) |
| 主预训练数据 | 几千小时 robot teleop | **20,854 小时人类 video** |
| 灵巧手能力 | 一般 | 22 DoF 接触丰富任务 |
| 多步推理 | 有 | 强化 (task + subtask 级) |
| Apache 2.0 | 受限 | **完全商用** |
| 工厂部署 | research | factory-floor ready |

---

## 评估

### 仿真基准
- **9 个 DexMG 任务**
- **24 个 RoboCasa mobile manipulator 任务**
- **24 个 Digital Cousin GR-1 humanoid manipulation 任务**
- 成功率自动测量

### 真机评估
- 超市打包任务
- 未见物体操作 (训练中没出现的)
- 多机器人工业协作 (带 handoff)
- 实验室人观察验证

### 目标机器人
- **Unitree G1**, **Bimanual YAM**, **AgiBot Genie-1**
- Post-training benchmark: LIBERO (Franka Panda), DROID, SimplerEnv Bridge (WidowX), SimplerEnv Fractal (Google Robot)
- 已发布 fine-tuned checkpoint 用于以上 benchmark
- Blog 和模型卡都**没给完整 success rate 表**, 主要量化结果是 scaling law (1k→20k = 2x)

---

## 部署 / 微调

### 部署
```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag GR1 \
    --model-path nvidia/GR00T-N1.7
```

### 客户端
```python
from gr00t.policy.server_client import PolicyClient
policy = PolicyClient(host="localhost", port=5555)
obs, info = env.reset()
action, info = policy.get_action(obs)
```

### 微调 (LeRobot dataset format)
```bash
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7 \
    --dataset-path <YOUR_DATASET_PATH> \
    --embodiment-tag <YOUR_EMBODIMENT> \
    --modality-config-path <YOUR_MODALITY_CONFIG> \
    --num-gpus 1 \
    --output-dir <OUTPUT_PATH> \
    --max-steps 2000 \
    --global-batch-size 32
```
- Pre-registered embodiments: `UNITREE_G1`, `LIBERO_PANDA`, `OXE_WIDOWX`
- 从 N1.6 升级: drop-in 替换 `--model-path` 为 `nvidia/GR00T-N1.7`, embodiment 配置不变

### 硬件需求
- Inference: 1 GPU, 16GB+ VRAM
- Fine-tuning: 1+ GPU, 40GB+ VRAM (H100 / L40 推荐)

---

## 与 pi 系列的对比 (同期: 2026.04)

两条不同的赌注:

| 维度 | pi_0.7 (PI) | GR00T N1.7 (NVIDIA) |
| --- | --- | --- |
| 总参数 | ~5B | 3B |
| 架构 | 单一 VLA + flow matching | Action Cascade (VLM + DiT) |
| 主要新数据源 | autonomous eval rollout + metadata 标签 | **20K 小时 EgoScale 人类视频** |
| 核心新机制 | Diverse Prompting (subgoal images + metadata) | **Relative EEF + 人类 video 预训练** |
| 跨 embodiment 桥梁 | subgoal image (BAGEL 14B 生成) | **Relative EEF action 直接对齐人 / 机器人** |
| Scaling 主张 | 数据多样性 + prompt 多样性 | **数据规模 (人类 video, dexterity scaling law)** |
| 部署 | 闭源 (云端 API) | **完全开源 + Apache 2.0** |
| 目标场景 | 桌面 / 移动 manipulator | 人形全身 + 灵巧手 |
| 关键论证 | compositional generalization 涌现 | dexterity scaling law 首次显式证明 |

**核心分歧**: 都想用"非机器人数据 (人类 video)"突破数据瓶颈, 但接口不同 — pi 用 subgoal image 当桥梁, GR00T 用 relative EEF delta 当桥梁。

---

## 代码获取

N1.7 代码就是 `NVIDIA/Isaac-GR00T` 的 **main 分支**, 没有独立 tag:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
```

已存在的仓库镜像: `GR00T_Series/vla_wbc/Isaac-GR00T/code/` (全家共用一份, 对应 N1 → N1.5 → N1.6 → N1.7 的历次 commit)

本地如需切到 N1.7 对应 commit, 看 HuggingFace 模型卡的 "Code Repository commit hash" 字段。

## 资料可得性小结

| 类型 | 是否存在 | 位置 |
| --- | --- | --- |
| 独立 arxiv 论文 | **否** | — |
| 独立 whitepaper / technical report PDF | **否** | NVIDIA 仅为 N1 发过 whitepaper |
| HuggingFace 模型卡 | 有 | https://huggingface.co/nvidia/GR00T-N1.7-3B |
| HuggingFace blog | 有 | https://huggingface.co/blog/nvidia/gr00t-n1-7 |
| 开源代码 | 有 | github.com/NVIDIA/Isaac-GR00T (main branch) |
| 权重 | 有 | HuggingFace collection: GR00T-N1.7 |
| 论文引用占位 | N1 原论文 | arXiv:2503.14734 |

## 文件索引

```
26_N17/
└── GR00T_N17_blog_report.md   <- 本文件 (无 arxiv/whitepaper, 整合自 HF model card + blog + GitHub)
```
