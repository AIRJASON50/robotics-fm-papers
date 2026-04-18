# SAME: Skeleton-Agnostic Motion Embedding for Character Animation (Lee et al., SIGGRAPH Asia 2023)

> Breaks the homeomorphic constraint of Aberman 2020. Single network handles arbitrary skeleton topologies.

## 1. Core Problem

Aberman 2020 requires skeletons to be **homeomorphic** (same branching structure, different joint counts along branches). Real-world scenarios have truly different topologies: 5-finger vs 4-finger hands, biped vs quadruped, characters with tails/wings. SAME learns a skeleton-agnostic embedding space that works for ANY skeleton topology.

## 2. Method Overview

```
Architecture: GCN Autoencoder with attention-based message passing

Encoder: Enc(S_src, D^t_src) -> z^t   (fixed-size embedding per frame)
  - Input: skeleton data S + motion data D^t (joint rotations, positions, velocities, contacts)
  - Multi-head graph attention convolution layers
  - Graph max pooling -> fixed-size vector z^t regardless of joint count

Decoder: Dec(S_tgt, z^{1:T}) -> D^{1:T}_tgt  (motion for target skeleton)
  - z^t concatenated with each target joint's features
  - Multiple GCN layers (NO graph pooling -- preserve joint resolution)
  - Output: joint rotations, root movement, contact labels

Key difference from Aberman 2020:
  - Aberman: per-skeleton encoder/decoder, shared only at primal skeleton level
  - SAME: SINGLE encoder, SINGLE decoder, shared parameters across ALL skeletons
  -> Θ (linear transform) and a (attention vector) are shared across all joints/skeletons
```

## 3. Key Designs

### 3.1 Shared GCN Kernels (vs Aberman's per-joint kernels)
- Aberman 2020: each armature has its own convolution kernel -> must be homeomorphic for alignment
- SAME: all joints share the same kernel parameters Θ, differentiated only by learned attention weights α_{i,j}
- Attention mechanism: α_{i,j} = softmax(LeakyReLU(a^T [Θx_i || Θx_j]))
  -> Network LEARNS which joints are important neighbors, not hard-coded by topology
- This is why it handles arbitrary topologies: the graph structure is an input, not an architectural assumption

### 3.2 Skeleton Augmentation
- Key training trick: generate diverse skeletons by randomly:
  - Adding/removing spine, neck, shoulder joints
  - Scaling limb lengths randomly
  - Adding dummy joints near hips and end-effectors (as observed in real datasets)
- Augmented skeleton database S' has H >> K skeletons (160 types from 92 original)
- Motion augmentation: retarget each motion to random augmented skeletons using MotionBuilder
- Result: 780 min of motion across 160 skeleton types

### 3.3 Training Losses
- L_rec: reconstruction (joint rotation + FK position + root movement), multi-term weighted
- L_vel: velocity matching (joint linear velocity) + jerk penalty
- L_con: contact label prediction + foot sliding prevention + ground penetration avoidance
- L_z: embedding consistency -- same motion on different skeletons -> same z
  L_z = ||Enc(S_src, D^t_src) - Enc(S_tgt, D^{1:T}_tgt)||^2
  This is the critical loss that makes the embedding skeleton-agnostic

### 3.4 Missing Joint Reconstruction
- If target skeleton has EXTRA joints not in source, SAME can still generate motion for them
- Demonstrated: upper-body motion -> full-body reconstruction (missing legs)
- Demonstrated: body without fingers -> body with fingers (finger motion synthesized!)
- -> This is directly relevant to hand retargeting: even if source has fewer DOF, decoder can fill in

## 4. Key Results

- Training: 92 skeleton variations, 130 min motions, augmented to 160 skeletons / 780 min
- Retargeting error comparable to Aberman 2020 (2.91 intra / 2.47 cross) vs (2.76 / 2.25)
  but with a SINGLE network vs Aberman's per-domain networks
- Motion classification: 95% accuracy in SAME space (skeleton-agnostic features are meaningful)
- Motion similarity search across different skeleton databases
- Interactive character control via motion matching in SAME space
- Embedding arithmetic: z_walk_with_wave - z_walk + z_stand = z_stand_with_wave

## 5. Limitations & Relevance to Hand Retargeting

**Limitations:**
- Biped-focused: trained on humanoid characters, may not generalize to hands without retraining
- Requires motion pairs (from MotionBuilder) for training -- not truly unsupervised
- Ornamental joints (tails, decorations) with completely different T-poses degrade quality
- Per-frame encoding: no explicit temporal modeling (unlike NKN's RNN)

**HIGH relevance to hand retargeting:**
- **Arbitrary topology support**: 5-finger human hand -> 4-finger Allegro, 3-finger gripper, 2-finger pincer
  ALL handled by the same trained network -- just change the target skeleton graph input
- **Missing joint reconstruction**: if robot hand has fewer joints than human hand,
  SAME can still generate plausible motion by "hallucinating" the missing DOF
  (demonstrated: generating finger motion from fingerless body motion!)
- **Shared kernel GCN**: the attention mechanism automatically discovers which joints
  in the target skeleton correspond to which joints in the source -- no manual mapping needed
- **Embedding space**: enables grasp similarity search across different hand types
  (is this Allegro grasp similar to that Shadow Hand grasp? Just compare z vectors)

**What needs adaptation for hands:**
- Need hand-specific training data: diverse hand skeletons + motion pairs
  -> MANO variations + URDF hand models + retargeted grasping data
- Contact awareness: SAME's contact loss is foot-ground only; hand needs hand-object contact
- Finger coupling: hands have strong inter-finger coupling (synergies) that full-body doesn't have
- Real-time: SAME's per-frame GCN is already fast, but hand retarget needs < 5ms

## 6. Evolution Chain & Positioning

```
NKN (2018): same topology, different bone lengths, unsupervised
  Limitation: requires identical kinematic tree structure

Aberman (2020): homeomorphic topologies, different joint counts, unsupervised
  Limitation: same branching structure required; per-domain networks
  Key idea: skeletal pooling to common primal skeleton

SAME (2023): arbitrary topologies, single network, semi-supervised
  Breakthrough: shared GCN kernels + attention -> topology as input not architecture
  Key idea: skeleton-agnostic embedding space via consistency loss

For hand retargeting, SAME's approach is the most promising because:
  1. Different finger counts are NORMAL (human 5, Allegro 4, Barrett 3, pincer 2)
  2. Different joint counts per finger are NORMAL (human 4, some robots 3)
  3. Single network for all hands >> per-pair training (scalability)
```
