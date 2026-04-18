# Semantic Deformation Transfer (Baran et al., SIGGRAPH 2009)

> Key extension of Sumner 2004: transfers semantic intent, not literal geometry.

## 1. Core Problem

Deformation Transfer (Sumner 2004) copies literal geometric deformations -- if Alex walks normally and Bob walks on his hands, literal correspondence maps Alex's legs to Bob's legs. But **semantic** correspondence should map Alex's legs to Bob's arms (the parts performing the walking function). This paper learns semantic correspondence from a few example pose pairs.

## 2. Method Overview

```
Input:
  - p example pose pairs: {(source_pose_i, target_pose_i)}  (5-12 pairs typically)
  - New source pose q (not in examples)

Pipeline:
  1. Encode all poses into shape space using patch-based LRI coordinates
     C_src: source mesh -> coordinate vector
     C_tgt: target mesh -> coordinate vector

  2. Project q into source shape space:
     Find weights w_i such that q ≈ Sigma w_i * x_i  (affine combination)
     Solution: pseudoinverse projection

  3. Interpolate in target shape space:
     target_pose = C_tgt^{-1}(Sigma w_i * C_tgt(target_example_i))

  4. Reconstruct target mesh from coordinates
```

## 3. Key Designs

### 3.1 Patch-Based LRI Coordinates (core representation)
- Problem with existing representations:
  - Vertex positions: interpolation causes shrinking (global rotation issue)
  - Deformation gradients: better, but interpolating > 180 degree rotations fails
  - Linear Rotation-Invariant (LRI) coords (Lipman 2005): noise-sensitive for projection
- Solution: **Patch-based LRI** = partition mesh into 5-15 patches, factor out per-patch rotation
  - Per-face: scale/shear S_f
  - Per-patch: average rotation G_k (stored as matrix logarithm)
  - Between patches: relative rotation log(G_i^{-1} G_j)
  - Within patches: relative face rotation log(G_{p(f)}^{-1} Q_f)
- Properties: rotation-invariant, good for both interpolation AND projection, handles large rotations

### 3.2 Linear Transfer Model
- The transfer is a linear map in shape space: source coordinates -> target coordinates
- Constructed from example pairs via pseudoinverse
- Extremely simple at runtime: one matrix multiply + reconstruction
- Assumption: deformation space is well-approximated by affine span of examples
- Limitation: can't represent nonlinear correspondences (e.g., "bend only if angle > 45 degrees")

### 3.3 Independent Parts Decomposition
- User can specify independent body parts (e.g., upper body / lower body)
- Each part gets its own transfer map with its own examples
- Reduces number of required example pairs (only half need full-body poses)

## 4. Results

- Crane walk -> Flamingo (7 example pairs)
- Alex march -> Handstand character (5 pairs)
- Gallop -> Alex and Bob (8+6 pairs for upper/lower body)
- Largest mesh: Flamingo with 52,895 triangles
- Runtime: 0.22s encode + 0.25s reconstruct per frame

## 5. Limitations & Relevance to Hand Retargeting

**Limitations:**
- Linear transfer model: can't represent nonlinear semantic relationships
- Requires continuous mesh -- not directly applicable to multi-rigid-body robot hands
- Example poses must span the relevant motion space (underdetermined regions will fail)
- No temporal coherence (frame-independent, like Sumner 2004)

**High relevance to hand retargeting:**
- Core insight: **semantic correspondence > literal correspondence**
  - Human thumb opposition -> robot thumb opposition (semantic)
  - vs. mapping joint angles directly (literal, fails for different kinematics)
- This is exactly what Harmonic Mapping (Chong 2021) does in joint space:
  few example pairs -> learn manifold mapping -> interpolate for new poses
- Patch-based LRI coordinates solve the rotation problem that plagues Laplacian-based retargeting
  -> Direct relevance to your Interaction Mesh rotation sensitivity issue
- The "5-12 example pairs" approach is highly practical for hand retargeting:
  demonstrate a few key grasps -> system learns the semantic mapping

## 6. Cross-Paper Comparison

```
Deformation Transfer (Sumner 2004):
  Literal geometric transfer. Needs ~50 markers. No semantic understanding.

Semantic DT (Baran 2009):
  Semantic transfer via example pairs. Needs ~5-12 pose pairs. Linear model.

Harmonic Mapping (Chong 2021):
  Same idea in joint angle space. Needs ~8 reference pairs. Nonlinear (CAE).
  -> HAE is essentially "Semantic DT in configuration space with a neural encoder"

GeoRT (Meta 2025):
  Bypasses correspondence entirely with geometric losses. No examples needed.
  But loses semantic control (can't specify "this grasp maps to that grasp").
```
