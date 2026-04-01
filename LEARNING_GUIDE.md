# Learning Guide: From Dexterous Hand RL to Robotics Foundation Model

> **Reader**: PPO sim2real dexterous manipulation practitioner
> **Goal**: Understand how CS foundation model techniques reshape robotics, and how your RL expertise fits in
> **This file**: The FIRST thing to read. Points you to everything else.

## Your Learning Path (overview)

```
 Phase 1                      Phase 2                         Phase 3                    Phase 4
 YOUR DOMAIN                  CS METHODOLOGY                  THREE FAMILIES             BACK TO WORK
 ~~~~~~~~~~~                  ~~~~~~~~~~~~~~                  ~~~~~~~~~~~~~~             ~~~~~~~~~~~~
 manip_landscape.md           CS2Robotics_Roadmap.md          RT_family_notes.md         Apply to your
   5 themes of dexterous        Level 0: Representation         "VLA origin story"       PPO sim2real
   manipulation                 Level 1: Pre-training+Scale   pi_family_notes.md         pipeline
 humanoid_landscape.md          Level 2: Generative Policy      "VLA + offline RL"
   4 themes of humanoid         Level 3: Full Robot FM        GR00T_family_notes.md
   whole-body control           Level 4: Frontier               "Hierarchy + WBC"
                                                                                         
 ~2h                          ~30h (self-paced)               ~6h                        ongoing
```

## Phase 1: Understand Your Domain (manip + humanoid)

**Why first**: You need to articulate *what problems* FM should solve before studying *how* FM works.

| File | Path | Key Takeaway | Time |
|------|------|-------------|------|
| manip_landscape.md | `manip/manip_landscape.md` | 5 themes (traditional_rl -> human2robot -> scaling_rl -> sim2real -> fm_manip) show that per-task PPO hits a wall at contact diversity and object generalization | 1h |
| humanoid_landscape.md | `humanoid/humanoid_landscape.md` | Motion tracking is humanoid's "ImageNet moment"; SONIC proves PPO + scale works at foundation level | 1h |

**Takeaway**: Your PPO sim2real skill is the starting point, not the destination. The field is moving from "one policy per task" to "one model, many tasks". Phase 2 explains how.

## Phase 2: Understand the Methodology (foundation_model)

**Main document**: `foundation_model/CS2Robotics_Roadmap.md` -- the Level 0-4 progressive curriculum.

| Level | Core Question | What You Read | Time |
|-------|--------------|---------------|------|
| 0 | What patterns transfer from CS to robotics? | Representation Learning (Bengio) + Transformer | 3h |
| 1 | How does large-scale pre-training work? | GPT series + Scaling Laws + MAE + DINOv2 + CV overview | 8h |
| 2 | Why move from RL to generative policy? | CLIP + DDPM + Flow Matching + Diffusion Policy + ACT + DT | 12h |
| 3 | What does a complete robot FM look like? | RT/PI/GR00T family notes + pi_0/SONIC/N1 papers | 12h |
| 4 | What comes next? | DreamZero + Robot Scaling Laws + surveys | 7h |

**Upgrade exams**: After each Level, take the exam in `foundation_model/note/level_exams.md`. 80% to advance.

## Phase 3: Deep Dive into the Three Families (RT / PI / GR00T)

Read the family notes in `foundation_model/robotics/families/`. Each teaches you something different:

| Family | Notes Path | What It Teaches You |
|--------|-----------|-------------------|
| Google RT Series | `Google_RT_Series/RT_family_notes.md` | How VLA was born: from LLM-as-planner (SayCan) to end-to-end (RT-2). Why the team left to build PI. The lesson: web knowledge transfers to robots, but autoregressive action tokens have limits. |
| PI Series | `pi_Series/pi_family_notes.md` | The full VLA pipeline: flow matching action expert + knowledge insulation + cross-embodiment pre-training. pi\*0.6 shows offline RL as post-training -- your PPO experience is directly relevant here. |
| GR00T Series | `GR00T_Series/GR00T_family_notes.md` | Hierarchical architecture: VLA brain (10Hz) + WBC cerebellum (120Hz) + world model imagination (DreamZero). Your sim2real and motion tracking skills map to the cerebellum layer. |

## Phase 4: Back to Your Work

How to apply what you learned to your PPO sim2real dexterous manipulation work:

1. **Short term**: Your PPO + sim2real pipeline remains valid for the WBC / low-level control layer (SONIC proves this). Refine it.
2. **Medium term**: Learn Diffusion Policy or flow matching as your high-level policy. Keep PPO for low-level tracking. This is the TWIST2 / GR00T N1 architecture pattern.
3. **Long term**: Fine-tune a pre-trained VLA (e.g., openpi) on your dexterous hand demos. Use your RL expertise for post-training (the pi\*0.6 route).
4. **Data strategy**: Build a teleoperation pipeline to collect demonstration data. This is the bottleneck -- not algorithms.
5. **Read pi\*0.6 carefully**: It shows how offline RL fine-tunes a VLA -- the exact intersection of "your RL skills" and "new FM paradigm".

## Quick Lookup Index

| "I want to understand..." | Go read |
|--------------------------|---------|
| Why per-task RL doesn't scale | `manip/manip_landscape.md` Section 0-1 |
| How motion tracking unifies humanoid control | `humanoid/humanoid_landscape.md` Theme A |
| The full CS-to-Robotics transfer roadmap | `foundation_model/CS2Robotics_Roadmap.md` |
| How GPT's pre-train+fine-tune maps to robots | `CS2Robotics_Roadmap.md` Level 1 |
| Why diffusion/flow replaces Gaussian policy | `CS2Robotics_Roadmap.md` Level 2 |
| pi_0's architecture and design choices | `robotics/families/pi_Series/pi_family_notes.md` |
| GR00T's hierarchical VLA+WBC design | `robotics/families/GR00T_Series/GR00T_family_notes.md` |
| How RL is used as post-training for VLA | PI family notes, search for "pi\*0.6" |
| What comes after VLA (world models) | `CS2Robotics_Roadmap.md` Level 4, DreamZero notes |
| Exam questions to test my understanding | `foundation_model/note/level_exams.md` |
| All papers in this repo | `paper/papers.yaml` or `paper/CLAUDE.md` |
