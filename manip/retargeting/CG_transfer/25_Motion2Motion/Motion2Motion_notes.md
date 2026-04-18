# Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence (Chen et al., SIGGRAPH Asia 2025)
> Recent. Training-free framework for cross-topology motion transfer using patch-based motion matching with sparse joint correspondences.

## 1. Core Problem

Transfer animations between characters with **substantially different skeletal topologies** (e.g., biped to quadruped, limbless to biped, 143-joint dragon to 48-joint bat). Two key challenges:
1. **No large-scale paired datasets** across diverse topologies -- data-driven methods cannot train
2. **Topological inconsistency** prevents one-to-one bone correspondence -- need to handle unbound joints

Previous methods (SAN, R2ET, WalkTheDog) either require same topology, large training data, or tailored per-pair training. Motion2Motion works with **one or a few example motions** on the target skeleton and **sparse** (not dense) joint correspondences.

## 2. Method Overview (with pseudocode)

```
# Motion2Motion: Training-free Cross-topology Transfer
Input:
  S in R^{F_s x D_s}           # source motion (F_s frames, D_s = 3 + 6*J_s channels)
  T_set = {T^(i)}              # few-shot target motion examples
  M = {(t,s)}                  # sparse bone correspondences (K pairs)

# Step 1: Build correspondence matrix and mask
C in R^{D_t x D_s}             # identity blocks at corresponding joints, zeros elsewhere
m[i] = sum_j C[i,j]            # mask: 1 for bound dimensions, 0 for unbound

# Step 2: Project source to target space + noise initialization
T_hat = S @ C^T + (1 - m) * N  # bound joints: copy source; unbound joints: random noise

# Step 3: Patchify source and target motions
P_s = sliding_window(S, patch_size=11, step_size=1)
P_t = sliding_window(T_set, patch_size=11, step_size=1)

# Step 4: Iterative matching-and-blending (L=3 iterations)
for l in range(L):
    P_t_hat = sliding_window(T_hat, patch_size=11, step_size=1)
    for each patch p in P_t_hat:
        # Masked motion matching: balance bound vs unbound dimensions
        p_match = argmin over P_t of:
            alpha * MSE(m * P, m * p) +           # bound joints: match source
            (1-alpha) * MSE((1-m) * P, (1-m) * p) # unbound: match current estimate
        # Replace with matched target patch
    # Blend overlapping patches by averaging
    T_hat = blend(matched_patches)

Output: T_hat in R^{F_s x D_t}  # retargeted motion on target skeleton
```

Key: runs on CPU only (MacBook M1), no GPU needed, no training.

## 3. Key Designs

### 3.1 Sparse Correspondence + Noise Initialization

The formulation elegantly splits target joints into two sets:
- **Bound joints**: have direct source correspondence, initialized from source motion via correspondence matrix C
- **Unbound joints**: no source mapping, initialized with Gaussian noise

The iterative matching progressively replaces noise with coherent motion by leveraging the coupling between bound and unbound joints observed in the few-shot target examples. This is motivated by the observation that, e.g., a dog's front legs can be inferred from its hind legs given a few walking examples.

**Relevance**: This bound/unbound split is directly relevant to hand retargeting where source (MediaPipe 21) and target (WujiHand 20 DOF) may have partial correspondence. Joints without direct mapping can be inferred from coupled motion patterns.

### 3.2 Masked Patch Matching with Alpha Balancing

The matching objective (Eq. 3) uses alpha to balance:
- alpha * MSE on bound dimensions (enforce source motion semantics)
- (1-alpha) * MSE on unbound dimensions (maintain temporal coherence of inferred joints)

Default alpha=0.85 (strong source adherence). Higher alpha = more faithful transfer but less diversity. Lower alpha = more diversity but potentially erratic.

The patch size P_S=11 frames captures meaningful temporal dynamics. Too small: loses semantic content. Too large: overly deterministic matching.

### 3.3 Training-free Patchwise Approach

Unlike neural methods that require large datasets and GPU training:
- Works with as few as 1-3 target motion sequences
- Runs on CPU in real time
- Naturally handles arbitrary topologies (no architecture changes needed)
- Supports diverse motion features: 6D rotation (default), 3D position, velocity

The "test-time scaling" property: more target examples = better quality, analogous to LLM test-time compute scaling.

## 4. Key Results

**Dataset**: Truebones-Zoo (animal animations, 9-143 joints) + LAFAN (human motion).

**Similar skeleton transfer** (e.g., bear -> dog):

| Metric | WalkTheDog | Pose2Motion | Motion2Motion |
|--------|-----------|-------------|---------------|
| FID (lower=better) | 1.447 | 2.040 | **0.780** |
| Freq Align (higher=better) | 0.877 | 0.721 | **0.971** |
| Contact Consistency | 0.781 | 0.685 | **0.861** |
| Diversity | 0.177 | 0.210 | **0.422** |

**Cross-species transfer** (e.g., biped -> quadruped):

| Metric | WalkTheDog | Pose2Motion | Motion2Motion |
|--------|-----------|-------------|---------------|
| FID | 2.224 | 2.850 | **0.925** |
| Freq Align | 0.795 | 0.645 | **0.948** |

Demonstrated transfers:
- Dragon (143 joints) -> Bat (48 joints), binding 2 pairs
- Bear (76 joints) -> Dog (55 joints), binding 6 pairs
- Flamingo (41 joints) -> Monkey (76 joints), binding 6 hind limb bones
- Anaconda (27 joints) -> Raptor (36 joints), binding 4 vertebral points
- SMPL (22 joints) -> Character with 331 joints (including skirt/hair), binding 21 joints

User study: significantly better quality and alignment scores than baselines.

Automatic bone binding via fuzzy subgraph matching works for similar skeletons; manual binding recommended for cross-species.

## 5. Limitations & Relevance to Hand Retargeting

**Limitations**:
- Requires at least 1 example motion on target skeleton (not zero-shot)
- Poor with semantically very different source/target motions (kungfu vs. dancing)
- No geometry/contact awareness (purely kinematic, no interpenetration handling)
- Diversity comes from noise -- not controllable in a task-specific way
- No object interaction consideration

**Relevance to our problem (MediaPipe 21 -> WujiHand 20 DOF)**:

| Aspect | Applicable? | Notes |
|--------|-------------|-------|
| Sparse correspondence | Yes | Our 21-to-20DOF mapping is inherently sparse |
| Few-shot target examples | Partially | We have continuous streaming, not discrete clips |
| Training-free | Yes (appealing) | No dataset collection needed |
| Patch-based temporal matching | No (real-time) | We need frame-by-frame online retargeting, not sequence-level |
| Cross-topology | Not needed | MediaPipe and WujiHand are roughly isomorphic (5 fingers each) |
| Contact/object interaction | No | Our primary concern, not addressed |

**Key takeaway**: Motion2Motion's main insight -- that unbound joint motion can be **inferred from bound joints via motion coupling patterns** observed in a few examples -- is powerful. For hand retargeting, this suggests: if we have a few example grasps performed on both human hand and robot hand, we can learn the coupling between directly-mapped joints and joints that need indirect inference. However, our problem is fundamentally online (real-time, frame-by-frame), not sequence-level batch processing, which limits direct applicability of the patch matching approach.

The **sparse correspondence matrix C** formulation is clean and directly implementable. For joints without 1:1 mapping, initializing from coupled motion patterns (rather than noise) could be adapted to our setting.

## 6. Cross-Paper Comparison

| Feature | Motion2Motion | SAME (2023) | SAN (2020) | R2ET (2023) | NKN (2018) |
|---------|--------------|-------------|------------|-------------|------------|
| Cross-topology | Yes (any) | Yes (any) | Yes (primal skeleton) | No | No |
| Training needed | No (zero) | Yes (large dataset) | Yes | Yes | Yes |
| Target examples needed | 1-few | 0 (generalized) | 0 | 0 | 0 |
| GPU required | No (CPU) | Yes | Yes | Yes | Yes |
| Geometry-aware | No | No | No | Yes (mesh) | No |
| Contact-aware | No | No | No | Yes (RDF/ADF) | No |
| Online (per-frame) | No (sequence) | Yes | Yes | Yes | Yes |
| Handles bone ratio | Implicitly (matching) | Attention-based | Pooling | DM normalization | FK+cycle |

Motion2Motion represents a new paradigm: **training-free, few-shot, cross-topology** transfer via motion matching. It's the most topology-flexible method but lacks geometry/contact awareness. For hand manipulation retargeting, a hybrid approach combining Motion2Motion's sparse correspondence idea with R2ET's normalized DM or ReConForM's contact-aware weighting would be ideal.

The evolution trajectory: NKN (same topology, learned) -> SAN (cross-topology, learned, primal skeleton) -> SAME (any topology, learned, single network) -> Motion2Motion (any topology, training-free, few-shot). The trend is toward greater generality with less data/compute requirements.
