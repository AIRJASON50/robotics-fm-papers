# R2ET: Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry (Zhang et al., CVPR 2023)
> ~150 citations. Primary learning-based baseline for skinned motion retargeting; cited by MeshRet and ReConForM.

## 1. Core Problem

Motion retargeting between characters with different **skeleton proportions** and **body shapes** (skinned mesh geometry). Previous learning-based methods (NKN, SAN, PMnet) either:
- Ignore shape geometry entirely (skeleton-only retargeting produces interpenetration/contact-missing on skinned characters)
- Use post-processing for geometry (Contact-Aware Model), which is slow and unstable

The two key differences to handle: (1) bone length ratio differences, (2) body shape geometry differences (skinny vs. bulky). These two objectives inherently conflict -- preserving motion semantics can cause interpenetration, while avoiding interpenetration can distort semantics.

## 2. Method Overview (with pseudocode)

R2ET uses a **residual** structure: starts from source motion copy, then applies learned modifications.

```
# R2ET Pipeline
Input: source_motion q_A, source_skeleton gamma_A, target_skeleton gamma_B, target_shape phi_B

# Step 1: Copy source motion as initialization
q_cp = q_A  # motion copy (preserves coherence, reduces search space)

# Step 2: Skeleton-aware module (semantics preservation)
Delta_q_s = F_s(gamma_A, gamma_B, q_cp)  # Transformer encoder, outputs quaternion residual
q_B_gamma = Delta_q_s (Hamilton product) q_cp  # apply residual rotation

# Step 3: Shape-aware module (geometry correction, per-limb MLP x4)
Delta_q_g = F_g(phi_B, q_B_gamma)  # perceives body part bounding boxes
q_B_gamma_phi = Delta_q_g (Hamilton product) q_B_gamma

# Step 4: Balancing gate (learned linear interpolation)
w = F_w(gamma_B, phi_B, q_B_gamma)  # per-joint weight in [0,1]
q_B = (1 - w) * q_B_gamma + w * q_B_gamma_phi

Output: q_B (target motion, single forward pass, no post-processing)
```

Training: self-reconstruction (no paired GT), adversarial loss, rotation constraint loss.

## 3. Key Designs

### 3.1 Normalized Distance Matrix (DM) for Semantics Preservation

The core innovation for handling skeleton mismatch. Motion semantics is modeled as a normalized pairwise joint Distance Matrix:

```
D in R^{N x N}: d_{i,j} = Euclidean distance between joint i and joint j

Semantics Similarity Loss:
L_sem = || eta(D_A / h_A) - eta(D_B / h_B) ||^2

where:
  h = skeleton height (normalizes for different scales)
  eta = L1 row-normalization (eliminates absolute bone length differences)
```

This is conceptually similar to Interaction Mesh (Ho 2010) but uses a **fully-connected** distance matrix rather than Delaunay triangulation. The L1 normalization is key -- it removes the effect of different bone length ratios, focusing on relative spatial relationships.

**Relevance to hand retargeting**: This normalized DM approach handles bone ratio mismatch (our 1.5x problem) more gracefully than raw Laplacian coordinates. The L1 normalization per row is essentially doing what we want -- making the representation scale-invariant per reference joint.

### 3.2 Distance Fields for Geometry (RDF + ADF)

For skinned characters, two voxelized distance fields on the target mesh:
- **RDF (Repulsive Distance Field)**: distance from body interior surface -- pushes penetrating vertices out
- **ADF (Attractive Distance Field)**: distance from body exterior surface -- pulls near-contact vertices closer

Four independent MLPs for four limbs. The shape information phi is the bounding box edge-lengths of each body part.

This is mesh-dependent and requires skinned characters -- **not applicable** to our MediaPipe 21-point cloud setup.

### 3.3 Balancing Gate

Per-joint learned weight w in [0,1] that interpolates between skeleton-aware output (semantics-preserving) and shape-aware output (geometry-corrected). Visualizations show w is high on joints prone to interpenetration (arms near torso) and low elsewhere. User-adjustable for interactive control.

## 4. Key Results

**Dataset**: Mixamo, 1952 training sequences (7 characters), 800 test sequences (11 characters), N=22 joints.

| Metric | NKN | PMnet | SAN | Copy | R2ET |
|--------|-----|-------|-----|------|------|
| MSE (lower=better) | 2.298 | 0.806 | 0.321 | 0.267 | **0.297** |
| Local MSE | 0.575 | 0.281 | 0.118 | 0.060 | **0.094** |
| Penetration % | 8.96 | 7.11 | 8.91 | 9.23 | **5.94** |
| Contact (cm) | 4.42 | 14.7 | 4.86 | 4.95 | **3.57** |

Key observations:
- MSE slightly higher than Copy (0.297 vs 0.267) but **48% less penetration** and **28% better contact**
- User study: 71.2% users prefer R2ET over all baselines
- Single inference, no post-processing
- Residual structure provides smooth, temporally coherent output

## 5. Limitations & Relevance to Hand Retargeting

**Limitations**:
- Requires skinned mesh (body shape phi) for geometry module -- not usable with point clouds
- Trained on Mixamo humanoid characters (22 joints) -- hand topology not demonstrated
- Fixed joint count N between source/target (same skeleton structure, different proportions)
- No cross-topology support (same-structure only, unlike SAME or Motion2Motion)

**Relevance to our problem (MediaPipe 21 -> WujiHand 20 DOF)**:

| Aspect | Applicable? | Notes |
|--------|-------------|-------|
| Normalized DM for semantics | Yes (strong) | L1-normalized pairwise distances handle 1.5x bone ratio well |
| Residual structure | Yes | Copy + residual is natural for same-topology retargeting |
| Hamilton product for rotation residual | Partially | We work in joint angle space, not quaternion |
| RDF/ADF geometry | No | Requires skinned mesh, we have 21 keypoints |
| Balancing gate | No | Only needed for semantics-geometry tradeoff |

**Key takeaway**: The **normalized Distance Matrix** idea is the most transferable concept. Our Laplacian formulation encodes similar spatial relationships but is sensitive to rotation and absolute scale. Switching to normalized pairwise distances (or adding it as a regularizer) could directly address our bone ratio mismatch problem.

## 6. Cross-Paper Comparison

| Feature | R2ET | NKN (2018) | SAN (2020) | SAME (2023) | MeshRet (2024) | ReConForM (2025) |
|---------|------|------------|------------|-------------|----------------|------------------|
| Input | Joint rotations | Joint positions | Joint rotations | Joint rotations | Mesh + skeleton | Mesh + skeleton |
| Cross-topology | No | No | Yes (primal skeleton) | Yes (any topology) | No | No |
| Geometry-aware | Yes (mesh) | No | No | No | Yes (dense mesh) | Yes (pairwise desc) |
| Training-free | No | No | No | No | No | No |
| Scale handling | DM L1 norm | FK+cycle | Skeletal pooling | Attention | DMI field | Adaptive desc |
| Contact | RDF/ADF | No | No | No | Interaction field | Contact-aware weight |
| Real-time | Yes (single pass) | Yes | Yes | Yes | No (optimization) | Yes |

R2ET occupies a unique position: it's the first to jointly handle skeleton proportion mismatch AND body shape geometry in a single forward pass. But it's limited to same-topology characters. For cross-topology hand retargeting, SAME or Motion2Motion approaches are more relevant architecturally, while R2ET's normalized DM loss is transferable as a training objective.
