# 手部轨迹数据格式规范 v0.5

面向 MANO/MediaPipe 手部关键点 + 可选物体位姿的轨迹数据，支持百万级 episode 管理。
基于 DexCanvas、ARCTIC、OakInk2、HOT3D、GigaHands、PALM 的实际字段布局。

v0.5 变更: 修正 MANO 45D 布局、新增坐标系规范、新增 `flat_hand_mean` 标记、
manifest 升级为 Parquet 主格式、新增 HDF5 并发/压缩指导、定义 source-of-truth 策略。

---

## 1. 设计原则

1. **双模式手部表示**: MANO 参数 (full mode) 或 21 关键点 (joints-only mode)，二选一或兼有
2. **公制单位**: 米、弧度、秒
3. **旋转**: axis-angle 为主 (MANO 原生)；物体变换接受 SE(3) 4x4
4. **canonical 坐标系**: 所有数据统一到 Z-up 右手系，per-episode 存储原始→canonical 变换
5. **分片存储**: shard 级组织，单 shard 1-10 GB，支持并行读取和增量扩展
6. **资产解耦**: 物体 mesh/URDF 独立于轨迹，通过 ID 引用
7. **HDF5 attrs 为 source of truth**: manifest 是派生缓存，可重建

---

## 2. 目录结构

```
dataset_root/
├── manifest.parquet            # 全局索引 (episode 级，主格式)
├── manifest_summary.json       # 人类可读汇总 (sources + shards 统计)
├── shards/
│   ├── shard_0000.hdf5
│   └── ...
├── assets/                     # 物体资产 (可选)
│   ├── registry.json
│   └── {object_id}/
│       ├── visual.obj
│       ├── collision/          # 凸分解 (仿真用)
│       │   └── decomposed.obj
│       ├── object.urdf         # 可选
│       └── meta.json           # mass, friction, bbox, etc.
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Shard 内部结构

```
shard_xxxx.hdf5
└── {episode_id}/
    ├── [attrs]                 # 元数据 (Section 3) -- SOURCE OF TRUTH
    ├── hand/
    │   ├── right/
    │   └── left/               # 可选 (双手)
    ├── object/                 # 可选 (hand-object 数据集)
    │   └── {object_id}/
    ├── attention_mask  [T]     # f32, 1.0=有效帧
    └── timestamp       [T]     # f64, 秒
```

每个 shard 包含 100-1000 条 episode。episode 按 `{source}/{operator}/{episode_id}` 寻址。

### HDF5 存储指导

| 项目 | 推荐值 | 说明 |
|------|--------|------|
| 压缩 | `gzip` level 4 或 `zstd` (via hdf5plugin) | float32 轨迹数据压缩率 ~30-40% |
| chunk shape | `[min(T, 256), D]` | D=45 (mano_pose) 或 D=63 (joint_positions) |
| 并发读取 | 设 `HDF5_USE_FILE_LOCKING=FALSE` + `swmr=True` | 多 worker DataLoader 必需 |
| 驱动 | `h5py.File(..., driver='sec2', swmr=True)` | 默认 driver，SWMR 模式 |

---

## 3. 元数据 (episode attrs) -- SOURCE OF TRUTH

### 3.1 必填

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `format_version` | str | `"0.5"` | |
| `fps` | float | `100.0` | DexCanvas=100, ARCTIC~30, OakInk2=120, HOT3D=30 |
| `total_frames` | int | `1500` | |
| `hand_mode` | str | `"mano"` / `"joints_only"` / `"mano+joints"` | |
| `is_bimanual` | bool | `false` | |
| `data_source` | str | `"mocap"` | `mocap` / `teleoperation` / `simulation` / `egocentric` |
| `source_dataset` | str | `"ARCTIC"` | |
| `operator_id` | str | `"s01"` | |
| `task_name` | str | `"cylindrical_grasp"` | |
| `is_generated` | bool | `false` | |
| `coordinate_frame` | str | `"z_up_rh"` | 见 3.3 坐标系枚举 |
| `T_canonical_from_source` | float32[4,4] | 单位阵 | 原始坐标系→canonical 的 SE(3) 变换 |

### 3.2 可选

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_category` | str | `power_grasp` / `precision_grasp` / `in_hand_manipulation` / `pick_and_place` |
| `active_start_frame` | int | 有效操作起始帧 |
| `active_end_frame` | int | 有效操作结束帧 |
| `language_description` | str | 自然语言任务描述 |
| `quality_rating` | float | 0-1 |
| `fitting_error_mean` | float | MANO 拟合平均误差 (mm) |

### 3.3 坐标系枚举

| 枚举值 | 说明 | 对应数据集 |
|--------|------|-----------|
| `z_up_rh` | Z-up 右手系 (**canonical**，Isaac Gym/MuJoCo 标准) | 转换后所有数据 |
| `mocap_original` | MoCap 室原始坐标系 | DexCanvas, ARCTIC, OakInk2 原始数据 |
| `arkit_session` | ARKit 每 session 不同的坐标系 | EgoDex 原始数据 |
| `headset_local` | 头显局部坐标系 | HOT3D 原始数据 |

**规则**: 转换入库时，所有数据统一到 `z_up_rh`。`T_canonical_from_source` 记录变换矩阵供溯源。
如果原始数据已是 `z_up_rh`，则 `T_canonical_from_source` = 单位阵。

---

## 4. 手部数据 (`hand/{side}/`)

`{side}` = `right` 或 `left`。

### 4.1 模式 A: MANO 参数 (`hand_mode` = `"mano"` 或 `"mano+joints"`)

| 路径 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `mano_pose` | [T, 45] | f32 | 15 关节 x 3 axis-angle，父关节相对 |
| `mano_global_rotation` | [T, 3] | f32 | 手腕朝向，axis-angle，canonical 系 |
| `mano_translation` | [T, 3] | f32 | 手腕位置，米，canonical 系 |
| `mano_shape` | [10] | f32 | Beta，每人固定 |

**MANO 必填 attrs** (在 `hand/{side}/` group 上):

| 属性 | 类型 | 说明 |
|------|------|------|
| `mano_side` | str | `"right"` 或 `"left"` |
| `flat_hand_mean` | bool | `True`: zero pose = flat hand; `False`: zero pose = mean hand pose (ARCTIC 用 False) |
| `mano_model` | str | `"MANO_v1.2"` |

> **`flat_hand_mean` 至关重要**: 不同实现默认值不同 (manotorch default=True, ARCTIC body_models=False)。
> 相同的 45D pose 数值在不同 flag 下 FK 输出完全不同。跨数据集使用前必须统一。

### 4.2 模式 B: 仅关节坐标 (`hand_mode` = `"joints_only"` 或 `"mano+joints"`)

| 路径 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `joint_positions` | [T, 21, 3] | f32 | 21 关键点，canonical 系，米 |

### 4.3 可选辅助字段

| 路径 | Shape | 说明 |
|------|-------|------|
| `fitting_error` | [T] | 每帧 MANO 拟合残差 (mm) |
| `confidence` | [T, 21] | 每关节置信度 0-1 (EgoDex/MediaPipe) |
| `robot_joint_angles` | [T, N] | retarget 后机器人关节角 |

`robot_joint_angles` attrs: `robot_hand_type` (str), `robot_urdf_path` (str), `joint_names` (str list, 有序)

### 4.4 MANO Pose 布局 (45D) -- MANO 原生运动链顺序

```
[0:9]    食指 Index    MCP(3) + PIP(3) + DIP(3)
[9:18]   中指 Middle   MCP(3) + PIP(3) + DIP(3)
[18:27]  小指 Pinky    MCP(3) + PIP(3) + DIP(3)
[27:36]  无名指 Ring   MCP(3) + PIP(3) + DIP(3)
[36:45]  拇指 Thumb    CMC(3) + MCP(3) + IP(3)
```

> **注意**: 这是 MANO 模型内部运动链顺序 (Index→Middle→Pinky→Ring→Thumb)，
> 与 4.5 节的 21-joint 输出顺序 (Thumb-first, MediaPipe 兼容) **不同**。
> 45D pose 是输入参数空间，21-joint 是 FK 输出空间，二者通过 reorder 映射：
> `reorder = [0, 13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]`

### 4.5 关节拓扑 (21 关键点, FK 输出 / MediaPipe 兼容)

```
Wrist(0)
├── Thumb:  CMC(1) -> MCP(2) -> IP(3) -> TIP(4)
├── Index:  MCP(5) -> PIP(6) -> DIP(7) -> TIP(8)
├── Middle: MCP(9) -> PIP(10) -> DIP(11) -> TIP(12)
├── Ring:   MCP(13) -> PIP(14) -> DIP(15) -> TIP(16)
└── Pinky:  MCP(17) -> PIP(18) -> DIP(19) -> TIP(20)
```

> MANO 原生输出 16 joints (无 fingertip)，5 个 TIP 通过顶点采样获得。
> `joint_positions[T,21,3]` 统一使用此 reorder 后的 Thumb-first 拓扑。

---

## 5. 物体数据 (`object/{object_id}/`, 可选)

仅 hand-object 数据集需要。

### 5.1 变换 -- 二选一

**方案 A: 分离字段** (DexCanvas、ARCTIC)

| 路径 | Shape | 类型 |
|------|-------|------|
| `translation` | [T, 3] | f32，米，canonical 系 |
| `rotation` | [T, 3] | f32，axis-angle，canonical 系 |

**方案 B: SE(3) 矩阵** (OakInk2、HOT3D)

| 路径 | Shape | 类型 |
|------|-------|------|
| `transform` | [T, 4, 4] | f32，模型→canonical |

attrs: `transform_layout` (`"separate"` / `"se3_4x4"`), `mesh_id` (str), `scale` (float), `is_articulated` (bool)

### 5.2 铰接物体 (ARCTIC 等)

| 路径 | Shape | 说明 |
|------|-------|------|
| `articulation` | [T, K] | 关节角度，弧度。ARCTIC K=1 (如笔记本翻盖角度) |

attrs: `articulation_dof` (int), `joint_type` (str, `"revolute"` / `"prismatic"`)

> **不可省略**: ARCTIC 的核心差异化就是铰接物体状态。转换时必须保留。

---

## 6. 有效性标记与时间戳

| 路径 | Shape | 说明 |
|------|-------|------|
| `attention_mask` | [T] | f32, 1.0=有效帧, 0.0=填充 |
| `timestamp` | [T] | f64, 从 episode 开始的秒数 |

---

## 7. 全局索引

### 7.1 主索引: `manifest.parquet`

Parquet 是主格式，支持 100K-1M 级 episode 的毫秒级列筛选。

**Schema (每行一条 episode)**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | string | `{source}/{operator}/{episode_id}` |
| `shard` | string | shard 文件路径 |
| `source_dataset` | string | |
| `operator_id` | string | |
| `task_name` | string | |
| `task_category` | string | nullable |
| `object_ids` | list[string] | nullable |
| `total_frames` | int32 | |
| `fps` | float32 | |
| `hand_mode` | string | `mano` / `joints_only` / `mano+joints` |
| `is_bimanual` | bool | |
| `is_generated` | bool | |
| `quality_rating` | float32 | nullable |

```python
import pyarrow.parquet as pq

# millisecond-level filtering at 1M episodes
table = pq.read_table("manifest.parquet",
    filters=[("source_dataset", "=", "ARCTIC"), ("hand_mode", "=", "mano")])
```

### 7.2 人类可读汇总: `manifest_summary.json`

```json
{
  "format_version": "0.5",
  "dataset_name": "dex_unified_v1",
  "total_episodes": 350000,
  "total_frames": 42000000,
  "canonical_fps": 30,
  "sources": {
    "ARCTIC":    { "episodes": 339,    "hand": "mano",        "bimanual": true  },
    "GigaHands": { "episodes": 14000,  "hand": "mano+joints", "bimanual": true  },
    "EgoDex":    { "episodes": 338000, "hand": "joints_only", "bimanual": true  }
  },
  "shards": {
    "total": 700,
    "avg_episodes_per_shard": 500,
    "avg_size_gb": 2.5
  }
}
```

### 7.3 Source of truth 策略

- **HDF5 episode attrs 是 source of truth**
- `manifest.parquet` 是派生缓存，可通过 `rebuild_manifest.py` 从 shards 重建
- 字段不一致时以 HDF5 attrs 为准

### 7.4 版本兼容策略

- 读取器必须兼容 `format_version >= v0.N-1` (即 v0.5 读取器必须能读 v0.4 shards)
- 低于兼容范围的 shard 必须迁移后使用
- 每个 shard 内 episode 版本可不同，读取器按 episode 级 `format_version` 分发

---

## 8. 转换要点

### 8.1 推荐 canonical FPS

**30 Hz** (ARCTIC / HOT3D / GigaHands 的原生帧率)。
- 高帧率来源 (DexCanvas 100Hz, OakInk2 120Hz): 降采样，rotation 用 slerp，translation 用线性插值
- 低帧率来源: 保持原始，manifest 中记录实际 fps

### 8.2 各数据集转换

| 来源 | 格式 | → hand_mode | 关键转换 |
|------|------|------------|---------|
| **DexCanvas** | Parquet | mano | 物体旋转 Euler→axis-angle；shape 去重；100Hz→30Hz 降采样 |
| **ARCTIC** | NPY | mano | **仅 obj_trans 做 mm→m** (手部 trans 已是 m)；物体 7D 拆分为 `articulation[T,1]` + `rotation[T,3]` + `translation[T,3]`；shape 去重 (取首帧或验证全帧一致)；`flat_hand_mean=False` |
| **OakInk2** | Pickle | mano | MANO key 去 `lh__`/`rh__` 前缀；SE(3) 4x4 直接保留；120Hz→30Hz |
| **HOT3D** | custom | mano | 通过 toolkit 加载；选 MANO (非 UmeTrack)；SE(3) 直接保留 |
| **GigaHands** | NPY | mano+joints | 双手 `[T, 42*3]` flat → 拆分前 21 joints 为右手、后 21 为左手 (验证 GigaHands 的 L/R 顺序) |
| **EgoDex** | HDF5 | joints_only | ARKit 68 joints 映射: wrist=joint[0], thumb=[1-4], index=[5-8]... (需验证 ARKit→21 的具体索引表) |
| **PALM** | NPY | mano+joints | 标准 MANO；FK 出 joints；`flat_hand_mean` 查 PALM 代码确认；静态手势 T=1 |
| **MediaPipe** | realtime | joints_only | 21 点 3D 直出；normalized→world 需乘 image 尺寸和深度 |

---

## 附录 A. 接触力数据 (可选，前瞻性)

> 当前接触信息在仿真中由物理引擎生成，不从数据集读取。此节仅为未来真实触觉数据预留。

| 路径 | Shape | 说明 |
|------|-------|------|
| `contact/{id}/has_contact` | [T] | bool |
| `contact/{id}/finger_id` | [T, MAX_C] | 0-4=手指, -1=手掌 |
| `contact/{id}/force_vector` | [T, MAX_C, 3] | 牛顿 |

## 附录 B. 视觉数据 (可选，建议独立文件)

| 路径 | Shape | 说明 |
|------|-------|------|
| `visual/{cam}/rgb` | [T_vis, H, W, 3] | uint8 |
| `visual/{cam}/intrinsic` | [3, 3] | 静态内参 |
| `visual/{cam}/extrinsic` | [T_vis, 4, 4] | cam→canonical SE(3) |

## 附录 C. 质量筛选标准

### C.1 帧级

| 检查项 | 阈值 | 来源 |
|--------|------|------|
| 关节完整性 | 21 关节全部有效 | RealDex |
| 关节限位 | MANO 合理范围 | DexGraspNet |
| 加速度异常 | 二阶差分 < threshold | ARCTIC |
| 拟合残差 | `fitting_error` < P70 分位 | PALM (top 30%) |
| 置信度 | `confidence` 均值 > 0.5 | EgoDex |

### C.2 序列级

| 检查项 | 阈值 | 来源 |
|--------|------|------|
| 最短长度 | >= `min_length` 帧 | GigaHands |
| 标注有效性 | 非 `'None'`/`'Buggy'` | GigaHands |
| 有效帧占比 | > 80% | HOT3D multi-dim mask |

### C.3 pipeline

```
Raw → 帧完整性 → 运动学合理性 → 质量评分 → attention_mask + quality_rating
```

## 附录 D. 资产管理 (hand-object 数据集适用)

### D.1 资产注册表 (`assets/registry.json`)

```json
{
  "objects": {
    "cube1": {
      "category": "cube",
      "mesh_path": "cube1/visual.obj",
      "collision_path": "cube1/collision/decomposed.obj",
      "urdf_path": "cube1/object.urdf",
      "scale": 1.0,
      "is_articulated": false,
      "mass_kg": 0.15,
      "friction": 0.5
    },
    "laptop": {
      "is_articulated": true,
      "articulation_dof": 1,
      "parts": ["top", "bottom"]
    }
  }
}
```

### D.2 绑定与版本

- 轨迹 HDF5 中仅存 `mesh_id` 字符串，通过 `registry.json` 查找 (HOT3D 模式)
- `registry.json` 含 `"asset_version": "v1"`，变更时在 `CHANGELOG.md` 记录
