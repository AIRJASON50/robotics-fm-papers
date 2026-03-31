# GR00T Family -- NVIDIA Humanoid Robot Foundation Model Overview

> **Purpose**: Understand NVIDIA's full-stack approach to humanoid robots -- not individual papers, but how an entire system is built from scratch.

---

## 1. NVIDIA's Humanoid Robot Strategy

NVIDIA does not build robots. Its strategy is to be the **operating system for robots** -- providing full-stack software from training to deployment, so hardware vendors (Unitree, Fourier, AGIBot, Galaxea) all adopt NVIDIA's solution.

The stack: Isaac Sim (simulation) -> Isaac Lab (RL training) -> GR00T (foundation model: VLA + WBC) -> Jetson (edge deployment) -> Cosmos (world model). GR00T is the "model layer" in this stack.

---

## 2. Two Independent Lines: Isaac-GR00T and SONIC

GR00T consists of **two independently developed controllers**, eventually combined:

### Isaac-GR00T Line (VLA: See -> Understand -> Plan)

**Team**: NVIDIA GEAR lab (Jim Fan, Yuke Zhu)
**Role**: High-level decision-making -- understand language + vision, output target action trajectories

**N1 (2025.03) -- First open-source humanoid VLA**

The architecture inherits from RT-2 (see `robotics/policy_learning/`): a frozen VLM encodes vision+language, then an action head (here Flow Matching DiT) generates motor commands. The dual-system design runs the VLM at a slow rate for semantic understanding, and the DiT at a faster rate for smooth motor output. Data pyramid (web video > sim > real) and cross-embodiment (one model, multiple robots) were the key ideas.

*Limitation*: Frozen VLM severely limited language-following ability and zero-shot generalization.

**N1 -> N1.5 (2025.12) -- VLM Adaptation**

- *Problem*: The frozen VLM from N1 could not express task-relevant visual features for manipulation.
- *Insight*: Two strategies exist for adapting a pretrained VLM -- (a) keep it frozen and add adapter layers, or (b) unfreeze some layers. N1.5 chose strategy (a): add adapter layers between the VLM and DiT, plus a stronger VLM backbone.
- *New capability*: FLARE training objective -- learn from human egocentric video without action labels, by aligning future visual latents. This turns unlimited human video into a free data source.
- *Result*: Language-following and real-robot success roughly doubled; novel-object generalization emerged via FLARE.

> **Cross-reference**: PI's "Knowledge Insulation" principle (see `robotics/vla/`) also addresses this VLM freezing dilemma -- they argue freezing preserves pretrained knowledge but limits downstream expressiveness. N1.5's adapter approach is one compromise; N1.6 takes the other path.

**Takeaway**: When VLM freezing limits performance, adapter layers are a lightweight first remedy. But adapters add indirection -- the gradient signal from the action loss reaches the VLM only indirectly.

**N1.5 -> N1.6 (2026.03) -- Direct VLM Unfreezing**

- *Problem*: Adapter layers are indirect and computationally wasteful -- they add parameters without directly improving the VLM's representations.
- *Insight*: Unfreezing the top layers of the VLM is more effective. Gradients flow directly from the action loss into the VLM, producing better visual features for manipulation with fewer extra parameters.
- *Other improvements*: Native aspect-ratio images (no more resize/pad information loss); state-relative actions (better sim2real transfer since relative deltas are less sensitive to calibration offsets).
- *New embodiments*: Bimanual arms, additional humanoid platforms, locomotion support.

**Takeaway**: When adapter indirection is a bottleneck, unfreezing the top VLM layers gives a more direct gradient path. This is the same lesson as LoRA vs full fine-tuning in LLMs -- sometimes the direct approach wins.

**Key Design Decisions Across Versions**:

| Problem | N1 | N1.5 / N1.6 | Why Change |
|---------|------|--------------|------------|
| VLM-to-DiT interface | Frozen VLM + cross-attention | Adapter (N1.5) / Unfreeze top layers (N1.6) | Frozen VLM lacks expressiveness |
| Data scarcity | Data pyramid (sim + real) | FLARE (human video, no action labels) | Human video is unlimited and free |
| Action representation | Absolute joint angles | State-relative (N1.6) | Absolute values amplify sim2real gap |
| Image encoding | Resize to fixed resolution | Native aspect ratio (N1.6) | Resizing destroys information |

### SONIC Line (WBC: Motion Tracking -> Joint Control)

**Team**: NVIDIA Research (Zhengyi Luo et al., from PHC lineage)
**Role**: Low-level execution -- track target motion trajectories, output joint angles

**Core ideas**:

**(1) Motion Tracking as the Universal Scalable Objective**

Previously: each action (walk/run/jump/carry) needs a custom reward function -- this does not scale. SONIC unifies all actions as "track mocap data" with one reward formula. The diversity of behaviors comes from diverse mocap data, not diverse reward engineering.

**(2) Universal Token Space (Cross-Embodiment Transfer)**

Human skeleton and robot skeleton differ. SONIC trains encoders that map both human and robot motions into a shared discrete token space (via FSQ). At inference: human mocap -> human encoder -> shared tokens -> robot decoder -> joint angles. This is implicit motion retargeting without manual skeleton mapping.

**(3) Scaling Laws for Humanoid Control**

Data scaling (more diverse mocap) gives the largest improvement and has not saturated. Model scaling helps but less. Compute scaling affects asymptotic performance, not just training speed. The ranking -- data > model > compute -- mirrors LLM scaling findings (Chinchilla).

---

## 3. How They Combine

From N1.5 onward, Isaac-GR00T and SONIC are deployed in series:

```
Language instruction
  |
  v
Isaac-GR00T VLM          (slow: semantic understanding)
  "Understand: move right hand to apple"
  |
  v
Isaac-GR00T DiT          (medium: generate motion trajectory chunks)
  "Plan: target motion trajectory"
  |  target trajectory (SMPL or latent tokens)
  v
SONIC Planner             (plan motion segments)
  |
  v
SONIC Tracker             (fast: track motion -> output joint angles)
  |
  v
PD Controller             (hardware-rate: torque control)
  |
  v
Physical Robot
```

**Why not end-to-end?** Each layer trains on fundamentally different data: VLM uses TB-scale internet text+images (no robot needed); DiT uses teleoperation demos (robot needed); SONIC uses human mocap (no robot needed). End-to-end = one model eating three data types with optimization conflicts. Layered = each layer independently optimized on its best data source.

**Alternative**: Decoupled WBC (also in GR00T-WBC codebase) splits differently -- RL for lower-body gait, IK for upper-body precision. Better end-effector accuracy (e.g., carrying a cup without spilling), worse whole-body coordination.

---

## 4. World Model Path (DreamGen -> DreamZero)

Parallel to VLA+WBC, NVIDIA explores the world model path:

### DreamGen (2025.05): World Model for Data Augmentation

Role: auxiliary tool serving VLA training. A small set of real demos is fed to a video world model which synthesizes variations (background, lighting, objects), amplifying data by orders of magnitude. This is an engineering implementation of the data flywheel.

### DreamZero (2026.02): World Model = Policy (GR00T N2 Core)

Role: next-generation architecture replacing VLA.

> **Cross-reference**: DreamZero's WAM (World-Action Model) represents the VLA -> WAM paradigm shift (see `world_model/26_DreamZero/`).

**Headline Takeaway: VLA = rote memorization; WAM = understanding principles**

| | VLA (N1.x) | WAM (N2) |
|---|---|---|
| Process | See current frame -> output memorized action | See current frame -> imagine future N frames -> extract action from imagination |
| Strength | Strong within training distribution | Strong outside training distribution |
| Analogy | A student who memorizes answers | A student who understands cause-and-effect |

WAM's generalization advantage comes from the same insight as LLM chain-of-thought: spending more compute at inference to "think" yields better out-of-distribution performance.

---

## 5. Complete Timeline

```
=== Infrastructure (2022-2024) ===
2022    Isaac Sim + Isaac Gym
2023    Isaac Lab (replaces Isaac Gym)
2024    Cosmos (world model infra), Jetson Thor (announced)

=== GR00T Gen 1 (2025) ===
2025.03  N1 -- first open-source humanoid VLA
2025.05  DreamGen -- world model for data augmentation
2025.11  SONIC -- large-scale whole-body motion control
2025.12  N1.5 -- VLM upgrade + FLARE; first N1.5+SONIC combined deployment

=== GR00T Gen 2 (2026) ===
2026.02  DreamZero -- WAM architecture (N2 core)
2026.03  N1.6 -- VLM unfreezing, native aspect ratio, state-relative actions
2026 H2  N2 (announced) -- WAM replaces VLA

=== Hardware Partners ===
Unitree G1, Fourier GR-1, AGIBot Genie-1, Galaxea R1 Pro, Bimanual YAM
```

---

## 6. Core Takeaways

| # | Takeaway | Principle | Action Item |
|---|----------|-----------|-------------|
| 1 | **Layered decoupling > end-to-end** | Different layers use different data; optimize independently | Build Layer 1 (tracking), plug in open-source VLM for Layer 3 |
| 2 | **Motion tracking = scalable universal objective** | One tracking reward handles all behaviors | Do not design per-action reward functions |
| 3 | **Data > Model > Compute** | Bottleneck is mocap diversity, not model size | Prioritize expanding data, not scaling the network |
| 4 | **Universal token space** | Shared discrete latent aligns human and robot motions | Can replace manual motion retargeting |
| 5 | **Sim2Real via domain randomization, not sim fidelity** | Sufficient DR = zero-shot transfer | Invest in DRCfg, not in sim tuning |
| 6 | **FLARE: learn from human video** | Align future latents; no action labels needed | Human video is a free, unlimited data source |
| 7 | **VLA = memorize answers; WAM = understand principles** | Imagining futures generalizes better than direct mapping | Watch DreamZero's follow-up closely |
| 8 | **VLM freezing/unfreezing is a spectrum** | Adapter (indirect) vs unfreeze (direct) -- choose based on data budget and expressiveness need | Start frozen + adapter; unfreeze if plateau |

### Is SONIC the SOTA for humanoid control?

**For "humanoid whole-body motion tracking", yes.** But with caveats:
- Motion tracking SOTA, not general robot SOTA
- Strong whole-body coordination, but weak end-effector precision (dexterous manipulation is not its strength)
- Validated on limited hardware; cross-embodiment generalization not fully tested
- MLP architecture has unknown scaling ceiling compared to Transformer-based approaches

---

## 7. File Index

```
GR00T_Series/
+-- GR00T_family_notes.md                  <-- this file
+-- vla_wbc/
|   +-- Isaac-GR00T/                       # Brain (VLA)
|   |   +-- code/                          #   NVIDIA/Isaac-GR00T repo
|   |   +-- 25_N1/                         #   N1 paper + notes
|   |   +-- 25_N15/                        #   N1.5 blog report
|   |   +-- 26_N16/                        #   N1.6 blog report
|   +-- SONIC/                             # Cerebellum (WBC)
|       +-- code/                          #   NVlabs/GR00T-WholeBodyControl repo
|       +-- SONIC_...md                    #   Paper
|       +-- SONIC_notes.md                 #   Notes (with bh_motion_track comparison)
+-- world_model/
    +-- 25_DreamGen/                       # Data augmentation (Cosmos world model)
    |   +-- GR00T-Dreams/                  #   Code repo
    +-- 26_DreamZero/                      # WAM (N2 core, video diffusion)
        +-- dreamzero/                     #   Code repo
```
