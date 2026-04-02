# dataset/ - Dexterous Manipulation Dataset Library

## Classification Logic

```
Entry requirement: standard hand format (MANO/MediaPipe) + object assets (mesh/URDF)?
├── Yes -> hand_object/
│   ├── Static (grasp poses only, no trajectory)
│   └── Dynamic (trajectories over time)
│       ├── manipulation (in-hand: rotation, sliding, finger gaiting, articulation...)
│       └── pick_and_place (approach, grasp, transport, place)
├── No, but has robot joints + object assets -> robot_hand/
└── No, hand data only (no object info) -> hand_only/
```

## Directory Structure

```
dataset/
├── hand_object/                        # MANO/MediaPipe + object assets [CORE]
│   ├── static/                         # (empty -- no MANO static grasp dataset exists)
│   ├── dynamic/
│   │   ├── manipulation/               # In-hand manipulation, articulated interaction
│   │   │   ├── 25_DexCanvas/          #   MANO + 30 objects, Cutkosky 21 types
│   │   │   ├── 24_OakInk2/           #   MANO+SMPL-X, 70+ categories, bimanual, 3-level task hierarchy
│   │   │   ├── 25_GigaHands/         #   MANO + 417 objects, 34h, 14K clips, bimanual
│   │   │   └── 22_ARCTIC/            #   MANO + 10 articulated objects, bimanual
│   │   └── pick_and_place/             # Approach, grasp, transport
│   │       ├── 24_HOT3D/              #   MANO + UmeTrack, 33 objects, egocentric multi-view, 833min
│   │       └── 25_DexCanvas (symlink) #   Cutkosky 01-15 (power + precision grasps)
├── robot_hand/                         # Robot-specific joints + object assets
│   ├── static/
│   │   └── 22_DexGraspNet/            #   ShadowHand 1.32M grasps, 5355 objects
│   └── dynamic/
│       ├── 24_ManipTrans/             #   DexManipNet: MANO->multi-hand retarget, 61 tasks, 3.3K eps
│       ├── 24_RealDex/                #   ShadowHand teleop, 52 objects, grasp sequences
│       └── 24_DexMimicGen/            #   Bimanual sim, 9 tasks, 21K demos
├── hand_only/                          # Hand motion without object assets/pose
│   ├── 25_PALM/                       #   MANO, 263 subjects, 13K scans, hand morphology prior
│   ├── 25_EgoDex/                     #   ARKit skeleton, 194 tasks, 829h, 1.73TB
│   └── 24_DexCap/                     #   Human joints -> LEAP retarget, ~100GB
├── CLAUDE.md
└── DATASET_COMPARISON.md
```

## Key Observations

### hand_object/ is the core category for our tracking project

**DexCanvas**, **GigaHands**, **OakInk2**, **ARCTIC**, and **HOT3D** provide standard human hand format (MANO) + object assets.
This is the minimal requirement for training Dex Sonic-style tracking policies:
- MANO enables retargeting to any robot hand
- Object assets + 6D pose enable object-reference tracking
- Dynamic trajectories enable RL/BC training

### hand_object/static/ is EMPTY

No dataset provides MANO-parameterized static grasp poses with object assets.
DexGraspNet has 1.32M grasps but in ShadowHand joint format, not MANO.
Gap: a large-scale MANO static grasp dataset would bridge robot_hand/static -> hand_object/static.

### hand_object/dynamic/pick_and_place/ has DexCanvas (symlink)

DexCanvas's Cutkosky types 01-15 (power + precision grasps) are dynamic grasp trajectories
(approach -> grasp -> lift -> place back), which belong here. Symlinked from manipulation/.
Still limited: only 30 objects, no contact force data.

### Contact force data: ZERO across all categories

No dataset in any category provides real contact force annotations.
DexCanvas claims to but hasn't released it. This is the biggest gap.

## Dataset Index

### hand_object/dynamic/manipulation/

| Folder | Dataset | arXiv | Scale | Verified Size | Key Value |
|--------|---------|-------|-------|---------------|-----------|
| 25_DexCanvas | DexCanvas | - | 12K seq, 30 objects, 21 Cutkosky types | ~8GB preview (gated 139GB) | MANO + object 6D + in-hand manipulation |
| 24_OakInk2 | OakInk2 | 2403.19417 | 70+ object categories, bimanual, 120Hz mocap | HuggingFace: kelvin34501/OakInk-v2 | MANO+SMPL-X + object mesh + 6D pose + 3-level task hierarchy + affordance |
| 25_GigaHands | GigaHands | 2412.04244 | 34h, 14K clips, 417 objects, 56 subjects, bimanual | ~16GB core (Globus, CC-BY-NC) | MANO + object 3D mesh + 6D pose, largest hand-object dataset |
| 22_ARCTIC | ARCTIC | 2204.13662 | 339 seq, 10 articulated objects, bimanual | ~803GB (registration) | MANO + articulated object interaction |

### hand_object/dynamic/pick_and_place/

| Folder | Dataset | arXiv | Scale | Verified Size | Key Value |
|--------|---------|-------|-------|---------------|-----------|
| 24_HOT3D | HOT3D | 2411.19167 | 833 min, 3.7M+ images, 33 objects, 19 subjects | Public (training GT open) | MANO + UmeTrack, egocentric multi-view (Aria + Quest 3), PBR object mesh, eye gaze |

### robot_hand/static/

| Folder | Dataset | arXiv | Scale | Key Value |
|--------|---------|-------|-------|-----------|
| 22_DexGraspNet | DexGraspNet | 2210.02697 | 1.32M grasps, 5355 objects | Largest grasp pose dataset, ShadowHand |

### robot_hand/dynamic/

| Folder | Dataset | arXiv | Scale | Verified Size | Key Value |
|--------|---------|-------|-------|---------------|-----------|
| 24_ManipTrans | DexManipNet (ManipTrans) | 2503.21860 | 3.3K eps, 1.34M frames, 61 tasks, bimanual | HuggingFace: LiKailin/DexManipNet | MANO->multi-hand retarget (Shadow/Allegro/Inspire/XHand), OakInk-V2 upstream |
| 24_RealDex | RealDex | 2402.13853 | 2.6K seq, 52 objects | ~tens GB (Google Drive) | Only real ShadowHand data |
| 24_DexMimicGen | DexMimicGen | 2410.24185 | 21K demos, 9 tasks | ~60GB (HuggingFace) | Automated bimanual data augmentation |

### hand_only/

| Folder | Dataset | arXiv | Scale | Verified Size | Key Value |
|--------|---------|-------|-------|---------------|-----------|
| 25_PALM | PALM | 2511.05403 | 263 subjects, 13K scans, 90K images | ~137GB (申请制) | MANO shape prior (263人), 亚毫米3D扫描, quality_dict筛选机制 |
| 25_EgoDex | EgoDex | 2505.11709 | 338K episodes, 194 tasks, 829h | ~1.73TB (Apple CDN) | Largest scale, language annotations |
| 24_DexCap | DexCap | 2403.07788 | ~90 min, wiping + packaging | ~100GB (HuggingFace) | End-to-end collection -> deployment pipeline |

