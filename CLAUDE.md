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

Roadmap document: `CS2Robotics_Roadmap.md` -- CS (NLP/CV) to Robotics migration paths and reading order.

#### foundation_model/LLM/ (LLM foundations)

| Folder | Paper | Year | Notes |
|--------|-------|------|-------|
| LLM/ (root) | GPT-1/2/3/4 Series (OpenAI) | 2018-2023 | LLM_notes.md |
| LLM/12_RepresentationLearning | Representation Learning: A Review (Bengio) | 2012 | RepresentationLearning_notes.md |
| LLM/17_Transformer | Attention Is All You Need (Vaswani et al.) | 2017 | Transformer_notes.md |
| LLM/gpt-2 | GPT-2 official code repo (OpenAI, TF) | 2019 | covered by LLM_notes.md |
| LLM/25_KimiK2 | Kimi K2: Open Agentic Intelligence (Moonshot AI, 1T MoE) | 2025 | - |
| LLM/24_Qwen | Qwen Technical Report (Alibaba, 1.8B-72B) | 2023 | Qwen_notes.md |

#### foundation_model/methods/ (core method papers)

| Folder | Paper | Year | Notes |
|--------|-------|------|-------|
| methods/20_DDPM | Denoising Diffusion Probabilistic Models (Ho et al.) | 2020 | DDPM_notes.md |
| methods/20_ViT | ViT: An Image is Worth 16x16 Words (Dosovitskiy et al.) | 2020 | ViT_notes.md |
| methods/21_CLIP | CLIP: Learning Transferable Visual Models (Radford et al.) | 2021 | CLIP_notes.md |
| methods/21_DecisionTransformer | Decision Transformer: RL via Sequence Modeling | 2021 | DecisionTransformer_notes.md |
| methods/22_FlowMatching | Flow Matching for Generative Modeling (Lipman et al., Meta) | 2022 | FlowMatching_notes.md |
| methods/22_RT1 | RT-1: Robotics Transformer for Real-World Control (Brohan et al.) | 2022 | RT1_notes.md |
| methods/23_ACT | ACT: Learning Fine-Grained Bimanual Manipulation (Zhao et al.) | 2023 | ACT_notes.md |
| methods/23_DiT | DiT: Scalable Diffusion Models with Transformers (Peebles & Xie) | 2023 | DiT_notes.md |
| methods/23_DreamerV3 | Mastering Diverse Domains through World Models (Hafner et al.) | 2023 | DreamerV3_notes.md |
| methods/23_OpenXEmbodiment | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 2023 | OpenXEmbodiment_notes.md |
| methods/23_RT2 | RT-2: Vision-Language-Action Models (Brohan et al.) | 2023 | RT2_notes.md |
| methods/24_DiffusionPolicy | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | 2024 | DiffusionPolicy_notes.md |
| methods/24_Octo | Octo: An Open-Source Generalist Robot Policy (Ghosh et al.) | 2024 | Octo_notes.md |
| methods/24_OpenVLA | OpenVLA: Open-Source Vision-Language-Action Model (Kim et al.) | 2024 | OpenVLA_notes.md |
| methods/24_PaliGemma | PaliGemma: A versatile 3B VLM for transfer (Beyer et al., Google) | 2024 | PaliGemma_notes.md |
| methods/24_pi0 | pi_0: A VLA Flow Model for General Robot Control (Physical Intelligence) | 2024 | pi0_notes.md |
| methods/25_GR00T_N1 | GR00T N1: Open Foundation Model for Humanoid Robots (NVIDIA) | 2025 | GR00T_N1_notes.md |

#### foundation_model/surveys/ (survey papers and meta-analyses)

| Folder | Paper | Year | Notes |
|--------|-------|------|-------|
| surveys/22_Chinchilla | Training Compute-Optimal LLMs (Hoffmann et al., DeepMind) | 2022 | Chinchilla_notes.md |
| surveys/23_FMRobotics | Foundation Models in Robotics (Firoozi et al., IJRR) | 2023 | FMRobotics_notes.md |
| surveys/23_GeneralPurposeRobots | Toward General-Purpose Robots via FM: Survey and Meta-Analysis | 2023 | GeneralPurposeRobots_notes.md |
| surveys/23_LangCondManip | Bridging Language and Action: Language-Conditioned Robot Manipulation | 2024 | LangCondManip_notes.md |
| surveys/24_AwesomeSurvey | Awesome-Robotics-Foundation-Models (GitHub survey) | 2024 | AwesomeSurvey_notes.md |
| surveys/24_LanguageGrounding | A Survey of Robotic Language Grounding: Symbols vs Embeddings | 2024 | LanguageGrounding_notes.md |
| surveys/25_AwesomeWorldModels | Comprehensive Survey on World Models for Embodied AI (GitHub survey) | 2025 | AwesomeWorldModels_notes.md |
| surveys/25_DynamicsModels | Learned Dynamics Models (Science Robotics) | 2025 | - |
| surveys/25_ScalingLaws | Neural Scaling Laws in Robotics (meta-analysis, 327 papers) | 2025 | ScalingLaws_notes.md |
| surveys/25_TransferLearningAgriculture | Transfer Learning in Agriculture Review | 2025 | - |
