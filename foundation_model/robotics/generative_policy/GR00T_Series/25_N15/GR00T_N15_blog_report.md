# GR00T N1.5 Technical Report (from research blog, no arxiv paper)

**Source**: https://research.nvidia.com/labs/gear/gr00t-n1_5/
**Date**: 2025.12
**Code**: Isaac-GR00T `n1.5-release` branch

---

## Architecture Changes from N1

### VLM
- Based on **Eagle 2.5**, tuned for improved grounding and physical understanding
- Model size: **2.1B** parameters (vs N1 2B)
- VLM is **frozen** during both pretraining and finetuning
- Simplified adapter MLP connecting vision encoder to LLM
- Added layer normalization to visual and text token embeddings

### VLM Grounding Performance

| Model | Size | GR-1 Grounding IoU | RefCOCOg-val IoU |
|-------|------|-------------------|-----------------|
| Qwen2.5-VL | 3B | 35.5 | 85.2 |
| **GR00T N1.5 VLM** | **2.1B** | **40.4** | **89.6** |

### DiT
- Same 16-layer DiT as N1
- Added **4-layer post-VLM transformer adapter**
- DiT processes state + noised actions via cross-attention to VLM embeddings

### New: FLARE Loss
- **Future LAtent Representation Alignment** (FLARE)
- Aligns model with target future embeddings (not generative future prediction)
- FLARE loss coefficient: 0.2
- Enables learning from **human ego-video** demonstrations

---

## Training

- **Steps**: 250K
- **Hardware**: 1K H100 GPUs
- **Global batch size**: 16,384
- **Optimizer**: AdamW + cosine LR schedule
- **Warmup ratio**: 0.05

### Data Sources
- Internal GR-1 data
- OpenXE dataset
- Simulated GR-1 (DexMimicGen)
- Neural trajectories from DreamGen
- AgiBot-Beta

---

## Performance (N1 → N1.5)

### Language Following

| Benchmark | N1 | N1.5 |
|-----------|-----|-------|
| Language Table | 52.8% | **93.2%** |
| Sim GR-1 Language | 36.4% | **54.4%** |

### Data-Limited Post-training (Sim)

| Benchmark | N1 | N1.5 |
|-----------|-----|-------|
| RoboCasa (30 demos/task) | 17.4 | **47.5** |
| Sim GR-1 (0-shot) | 39.6 | **43.9** |
| Sim GR-1 (30 demos/task) | 43.2 | **47.4** |

### Real GR-1

| Metric | N1 | N1.5 |
|--------|-----|-------|
| Language following rate | 46.6% | **93.3%** |
| Overall success rate | 43.3% | **83.0%** |

### Novel Object Generalization

| Setting | N1 | N1.5 |
|---------|-----|-------|
| 0-shot | 0% | **15.0%** |
| FLARE post-trained | - | **55.0%** |

### Unitree G1 (1K teleoperation episodes)

| Task | N1 | N1.5 |
|------|-----|-------|
| Pick/place 1 of 2 (4 fruits) | 44.0% | **98.8%** |
| Pick/place 1 of 2 (5 novel) | - | **84.2%** |

---

## Key Improvements
1. Language following: 46.6% → **93.3%** (2x)
2. Overall success: 43.3% → **83.0%** (2x)
3. Novel object generalization via FLARE + DreamGen
4. Learning from human ego-video (no robot action labels needed)
