# GR00T N1.6 Technical Report (from research blog, no arxiv paper)

**Source**: https://research.nvidia.com/labs/gear/gr00t-n1_6/
**Date**: 2026.03 (GTC 2026)
**Code**: Isaac-GR00T `main` branch

---

## Architecture Changes from N1.5

### VLM
- **Backbone**: Internal NVIDIA **Cosmos-Reason-2B** VLM variant
- Supports **flexible resolution** and **native aspect ratio** encoding (no padding)
- Trained on general vision-language tasks + embodied reasoning (next action prediction)
- **Top 4 VLM layers unfrozen** during pretraining (vs N1.5 fully frozen)

### DiT
- **32 layers** (2x N1.5's 16 layers)
- **Removed** the 4-layer post-VLM transformer adapter
- VLM unfreezing replaces adapter -- more direct gradient flow

### Action Representation
- Shifted to **state-relative action chunks** for most embodiments
- (vs N1/N1.5's absolute joint angles / EEF positions)

---

## Training

- **Pretraining**: 300K steps, global batch size 16,384
- **Post-training**: 10K-30K steps, batch size 1K or less

### New Data Sources (beyond N1.5)
- **Bimanual YAM arms** teleoperation data
- **AGIBot Genie-1** data
- **Simulated Galaxea R1 Pro** (BEHAVIOR benchmark suite)
- **Unitree G1** whole-body locomotion-manipulation data
- Total: several thousand hours of new teleoperated data

---

## New Embodiments
1. Bimanual YAM robots
2. AgiBot humanoids
3. Unitree G1 legged manipulators

---

## Key Architecture Decisions

| Decision | N1.5 | N1.6 | Rationale |
|----------|------|------|-----------|
| VLM | Eagle 2.5 (frozen) | **Cosmos-Reason-2B (top 4 unfrozen)** | Direct gradient flow > adapter |
| Post-VLM | 4-layer adapter | **Removed** | Redundant with unfrozen VLM layers |
| DiT depth | 16 layers | **32 layers** | Double capacity for more embodiments |
| Image encoding | Resize to fixed | **Native aspect ratio** | No information loss from padding |
| Action space | Absolute positions | **State-relative** | Better sim2real transfer |

---

## Notes
- No quantitative performance comparison published (blog-only release)
- N1.7 announced same day (early access + commercial license, adds dexterous manipulation)
- N2 (DreamZero/WAM architecture) previewed at same GTC event
