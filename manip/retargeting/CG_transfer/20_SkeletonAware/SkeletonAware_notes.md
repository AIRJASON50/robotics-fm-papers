# Skeleton-Aware Networks for Deep Motion Retargeting (Aberman et al., SIGGRAPH 2020)

> ~200+ citations. First automatic method for unpaired cross-structural skeleton retargeting.

## 1. Core Problem

Motion retargeting between characters with **different skeleton structures** (different number of joints) without paired training data. Previous methods (NKN/Villegas 2018) handle different bone lengths but require the SAME topology.

## 2. Method Overview

```
Key Insight: homeomorphic skeletons can be reduced to a common "primal skeleton"
by merging adjacent edges (removing degree-2 nodes).

Architecture:
  Per-domain encoder E_m = [E^Q_m, E^S_m]  (dynamic rotations + static offsets)
  Per-domain decoder D_m
  Per-domain discriminator C_m

Retargeting A -> B:
  1. Encode source motion:     z = E^Q_A(Q_A, S_A)    (dynamic features)
  2. Combine with target skeleton: S_B = E^S_B(S_B)    (static features)
  3. Decode to target:         Q_B = D_B(z, S_B)
  4. Apply FK:                 positions = FK(Q_B, S_B)

Training: unpaired, each domain has its own motion dataset
```

## 3. Key Designs

### 3.1 Skeletal Pooling (core contribution)
- Pooling on skeleton graph by merging edges of degree-2 nodes
- Reduces different skeletons to common primal skeleton
- **Topology-preserving**: pooled graph is homeomorphic to original
- Applied to both static and dynamic branches
- Formal: S_hat_i = pool{S_j | j in P_i}, Q_hat_i = pool{Q_j | j in P_i}
  where P_i is a set of edges merged into one (pool = max or average)

### 3.2 Skeleto-Temporal Convolution
- Two parallel branches: static (skeleton offsets) + dynamic (joint rotations)
- Convolution kernel has local support along both armature axis and temporal axis
- **Kernels NOT shared across armatures** -- different body parts learn different patterns
- Adjacency defined by kinematic chain distance d: N^d_i = {edges within distance d}

### 3.3 Static-Dynamic Disentanglement
- Static component S: skeleton bone offsets (3D vectors), time-independent
- Dynamic component Q: joint rotations (quaternions), time-dependent
- Root R: global translations + orientations, separate
- In raw motion, static and dynamic are coupled (same rotation on different skeletons -> different positions)
- Encoders learn to decouple: dynamic latent is skeleton-agnostic, static latent is skeleton-specific

### 3.4 Training Losses
- L_rec: reconstruction (joint rotation + FK position), prevents error accumulation
- L_ltc: latent consistency -- E^Q_B(retargeted) should match E^Q_A(original) in primal skeleton space
- L_adv: adversarial -- retargeted motion should look natural for target character
- L_ee: end-effector velocity matching (normalized by kinematic chain length)
- IK post-processing: automatic foot contact cleanup

## 4. Key Results

- Dataset: Mixamo, 15 characters with 3 skeleton structures (different joint counts)
- Cross-structural retargeting between structures with different number of joints
- Outperforms NKN (Villegas 2018) and Holden 2016 on both intra and cross-structural tasks

## 5. Limitations & Relevance to Hand Retargeting

**Critical limitation -- homeomorphic graphs only:**
- Requires source and target skeletons to be **topologically equivalent** (homeomorphic)
  - Same set of end-effectors
  - Same branching structure
  - May differ only in number of intermediate joints along branches
- Human hand (5 fingers, 4 joints each) -> Allegro (4 fingers, 4 joints each): **NOT homeomorphic** (different branching)
- Human hand -> Shadow Hand (5 fingers, different joint counts): homeomorphic IF same branching
- -> For different-finger-count hands, need SAME (2023) which drops this constraint

**What transfers to hand retargeting:**
- Skeletal pooling idea: merge intermediate phalanges to get "primal hand skeleton" (palm + fingertips)
- Static-dynamic disentanglement: bone lengths (static) vs joint rotations (dynamic) separation
- Latent consistency loss: ensure retargeted motion has same "meaning" regardless of skeleton
- End-effector velocity normalization by chain length: directly applicable to fingertip velocity matching

**What doesn't transfer:**
- Full-body focus: 22 joints, simple chains; hands have 20+ DOF with coupled joints
- No contact awareness: CG animation doesn't need contact-preserving retargeting
- Temporal focus (RNN-based in NKN, temporal convolutions here): hand retarget often frame-independent

## 6. Evolution Chain

```
Gleicher 1998 (optimization)
  -> NKN / Villegas 2018 (neural, cycle consistency, same topology only)
    -> Aberman 2020 (skeletal pooling, cross-structural, homeomorphic only)
      -> SAME 2023 (GCN, arbitrary topology, single network)
        -> AnyTop 2025 (diffusion, text-conditioned, truly arbitrary)
```
