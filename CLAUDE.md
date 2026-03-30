# paper/ - Research Paper Library

## Directory Structure

```
paper/
├── humanoid/              # Humanoid whole-body control papers
├── manip/                 # Dexterous manipulation papers
├── foundation_model/      # Foundation models, generative control, world models
└── html2aitext_convert/   # arxiv2md tool (DO NOT modify)
```

## Paper Acquisition Workflow

### 1. arxiv Paper Fetching

Use the conversion tool to fetch arxiv papers as markdown:

```bash
bash /home/l/ws/doc/paper/html2aitext_convert/arxiv2md.sh <arxiv_id>
# Output: /home/l/ws/doc/paper/html2aitext_convert/output/<title>.md
```

### 2. Folder Naming Convention

Create a folder under the appropriate category (`humanoid/`, `manip/`, or `foundation_model/`) with:

- **Format**: `<YY>_<ShortName>`
- **YY**: 2-digit year of publication (e.g., `25` for 2025)
- **ShortName**: Short, memorable abbreviation of the paper/project name

Examples:
- `25_HDMI` (HDMI, 2025)
- `25_OmniRetarget` (OmniRetarget, 2025)
- `25_SONIC` (SONIC, 2025)
- `24_ManipTrans` (ManipTrans, 2024)

### 3. Folder Contents

Each paper folder should contain:

```
<YY>_<ShortName>/
├── <paper_title>.md        # arxiv2md output (copy from html2aitext_convert/output/)
├── <repo_name>/            # Git clone of the codebase (if available)
└── <ShortName>_notes.md    # Analysis notes (Chinese)
```

### 4. Notes Writing Standard (`<ShortName>_notes.md`)

Notes must include the following sections:

1. **Core Problem**: What problem does this paper solve?
2. **Method Overview**: Architecture, pipeline, key formulas
3. **Key Designs**: The 2-3 most important technical contributions, with intuitive explanations
4. **Experiments**: Main results, ablation findings
5. **Related Work Analysis**: Field development context, what makes this work unique
6. **Limitations & Future Directions**: Author-stated + inferred from code
7. **Paper vs Code Discrepancies**: Things the paper didn't mention but code implements (e.g., teacher-student distillation, extra reward terms, per-task tuning). This section is critical.
8. **Cross-Paper Comparison**: Compare with other papers in this library and with bh_motion_track where relevant

**Format rules**:
- Written in Chinese, technical terms in English
- No emoji
- Use markdown tables for structured comparisons
- Include file paths when referencing specific code locations

## Paper Index

### humanoid/

| Folder | Paper | Year | Notes |
|--------|-------|------|-------|
| 18_DeepMimic | DeepMimic: Example-Guided Deep RL | 2018 | - |
| 23_PHC | Perpetual Humanoid Control (ICCV) | 2023 | PHC_notes.md |
| 24_H2O | H2O + OmniH2O (Whole-Body Teleoperation) | 2024 | H2O_notes.md |
| 25_ASAP | ASAP: Aligning Sim and Real Physics | 2025 | - |
| 25_BeyondMimic | BeyondMimic: Versatile Humanoid Control | 2025 | - |
| 25_FPO | First-Person Operation Control | 2025 | - |
| 25_HDMI | HDMI: Interactive Humanoid Control from Video | 2025 | HDMI_notes.md |
| 25_OmniRetarget | OmniRetarget + HoloSoma Framework | 2025 | OmniRetarget_notes.md |
| 25_RWM | Robotic World Model | 2025 | - |
| 25_SONIC | SONIC: Supersizing Motion Tracking | 2025 | - |
| 25_TWIST2 | TWIST2: Teleoperated Whole-Body Imitation | 2025 | - |
| Humanoid-Locomotion-Survey | Evolution of Humanoid Locomotion Control (Survey) | 2025 | survey_notes.md |

### manip/

| Folder | Paper | Year | Notes |
|--------|-------|------|-------|
| 23_ArtiGrasp | ArtiGrasp: Articulated Object Grasping | 2023 | - |
| 23_PhysHOI | PhysHOI: Physics-based Human-Object Interaction | 2023 | - |
| 24_BiDexHD | BiDexHD: Bimanual Dexterous Hand | 2024 | - |
| 24_DexMachina | DexMachina: Dexterous Manipulation | 2024 | - |
| 24_ManipTrans | ManipTrans: Manipulation Transfer | 2024 | - |
| 24_ObjDexEnvs | ObjDexEnvs: Object Dexterous Environments | 2024 | - |
| 24_QiHaoZhi | QiHaoZhi Dexterous Manipulation | 2024 | - |
| 25_MinBC | MinBC: Choice Policy for Humanoid Manipulation | 2025 | MinBC_notes.md |
| 25_DexCanvas | DexCanvas: Dexterous Motion Tracking | 2025 | - |
| 25_SimToolReal | SimToolReal: Sim-to-Real Tool Use | 2025 | - |
| 25_SPIDER | SPIDER: Dexterous Manipulation | 2025 | - |
| 25_OmniReset | OmniReset: Emergent Dexterity via Diverse Resets and Large-Scale RL | 2025 | OmniReset_notes.md |
| 26_Dex4D | Dex4D: 4D Dexterous Manipulation | 2026 | - |
| 26_HandelBot | HandelBot: Real-World Piano Playing | 2026 | HandelBot_notes.md |

### foundation_model/

Roadmap: `CS2Robotics_Roadmap.md` -- CS→Robotics migration paths and reading order.

#### foundations/ (universal ML foundations)

| Folder | Paper | Year |
|--------|-------|------|
| 10_TransferLearning | A Survey on Transfer Learning (Pan & Yang, IEEE TKDE) | 2010 |
| 12_RepresentationLearning | Representation Learning: A Review (Bengio, IEEE TPAMI) | 2013 |
| 17_Transformer | Attention Is All You Need (Vaswani et al., NeurIPS) | 2017 |

#### LLM/ (LLM knowledge base)

| Folder | Content | Notes |
|--------|---------|-------|
| NLP_foundations/ | Word2Vec, Seq2Seq, BahdanauAttention, BERT, Chinchilla | 5 NLP-specific foundations |
| families/GPT_Series/ | GPT-1/2/3/4 + Scaling Laws + RLHF + Codex + WebGPT + InstructGPT | GPT_series_notes.md |
| families/kimi/ | k1.5, MoBA, Moonlight, Audio, K2, K2.5 (Moonshot AI) | kimi_series_notes.md |
| families/qwen/ | Qwen 1/2/2.5/3/3.5, VL, Audio, Omni (Alibaba) | qwen_series_notes.md |
| families/deepseek/ | DeepSeekMoE, V2 (MLA+MoE), V3, R1 | deepseek_series_notes.md |
| families/llama/ | LLaMA 1, Llama 2/3/4 (Meta) | llama_series_notes.md |

#### CV/ (CV knowledge base, by technique)

| Folder | Content |
|--------|---------|
| 0_backbone/ | ResNet, ViT, TransferFeatures |
| 1_generation/ | VAE, DDPM, FlowMatching, DiT |
| 2_vl_alignment/ | CLIP, LLaVA, PaliGemma |
| 3_3d_vision/ | (待填: NeRF, 3DGS, Depth Anything) |
| 4_self_supervised/ | (待填: MAE, DINO, DINOv2) |
| 5_detection_seg/ | SAM |
| 6_video/ | (待填) |

#### robotics/ (robotics applications, by approach)

| Folder | Papers |
|--------|--------|
| llm_planning/ | SayCan, CodeAsPolicies, InnerMonologue, Voyager |
| policy_learning/ | DecisionTransformer, RT-1, RT-2, ACT, OpenXEmbodiment |
| vla/ | PaLME, Octo, OpenVLA |
| generative_policy/ | DiffusionPolicy, pi_0, GR00T_N1 |
| world_model/ | DreamerV3 |

#### surveys/

| Folder | Content |
|--------|---------|
| CV/ | 7 surveys: ViT(TPAMI), NeRF+3DGS, VLM(TPAMI), SSL(TPAMI), Depth, Video(TCSVT), MIM(IJCV) |
| robotics/ | 7 surveys: FMRobotics(IJRR), GeneralPurposeRobots, LangCondManip, LanguageGrounding, WorldModels, DynamicsModels, RobotScalingLaws |
