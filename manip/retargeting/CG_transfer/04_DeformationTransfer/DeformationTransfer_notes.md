# Deformation Transfer for Triangle Meshes (Sumner & Popovic, SIGGRAPH 2004)

> ~900+ citations. Foundational work for mesh-level deformation reuse across different characters.

## 1. Core Problem

Reusing deformations across different meshes is expensive: each new target character requires re-creating the animation from scratch. The goal is to automatically copy deformations from a source mesh onto a target mesh with different topology (vertex count, connectivity).

## 2. Method Overview

```
Input:
  - Source mesh: reference pose S + deformed poses S'_1, S'_2, ...
  - Target mesh: reference pose T (different topology from S)
  - Correspondence map M: {(s_i, t_i)} mapping source triangles -> target triangles

Pipeline:
  1. Compute per-triangle affine transformations from source deformation
     Q = V_deformed * V_original^{-1}   (3x3 matrix per triangle)
     (4th vertex added perpendicular to triangle for full 3x3)

  2. Map transformations through correspondence: source tri -> target tri

  3. Solve constrained optimization for target vertex positions:
     min Sigma_{j} ||S_{s_j} - T_{t_j}||^2_F
     s.t. shared vertices transform to same position (consistency)

  4. Reformulate as vertex positions -> sparse linear system A^T A x = A^T c
     Factor once (LU), then each new pose = backsubstitution only
```

## 3. Key Designs

### 3.1 Deformation Gradient Representation
- Each triangle's deformation encoded as 3x3 affine transformation Q
- Non-translational part captures orientation, scale, skew changes
- 4th virtual vertex (perpendicular to triangle plane) resolves the under-determined system
- This representation is **purely mesh-based** -- no skeleton or rig knowledge needed

### 3.2 Consistency Constraints
- Naive application of per-triangle transforms breaks mesh connectivity
- Key constraint: shared vertices between adjacent triangles must map to the same position
- Formulated as constrained least-squares -> eliminated by vertex formulation
- Result: single sparse linear system, factored once per source/target pair

### 3.3 Semi-Automatic Correspondence
- User specifies ~42-77 marker point pairs
- Algorithm deforms source mesh to match target using:
  - E_S: deformation smoothness (adjacent triangles have similar transforms)
  - E_I: deformation identity (changes should be minimal)
  - E_C: closest valid point (source vertices -> nearest target surface)
- Two-phase optimization with increasing E_C weight
- Then: triangle correspondence by centroid proximity

## 4. Results

- Horse -> Camel (65 markers, LU: 1.6s, backsubst: 0.29s per pose)
- Cat -> Lion, Face -> Head, Horse -> Flamingo
- Works for skeletal deformations AND non-rigid (rubber sheet collapsing)
- Source and target need NOT share vertex count, triangle count, or connectivity

## 5. Limitations & Relevance to Hand Retargeting

**Limitations:**
- Requires "gross similarity" between source and target (horse <-> camel OK, horse <-> fish problematic)
- Correspondence is many-to-many but still requires manual markers (~1 hour)
- Transfers literal deformation, not semantic intent (fixed by Baran 2009 Semantic DT)
- No temporal coherence handling (frame-independent)
- Global position must be manually constrained

**Relevance to hand retargeting:**
- Core idea directly applicable: human hand deformation -> robot hand deformation via triangle correspondence
- But: robot hands are multi-rigid-body, not continuous meshes
  -> Need adaptation: treat each rigid link as a separate mesh patch, or use skinned mesh
- The deformation gradient representation is the theoretical ancestor of:
  - CMU Kinematic Retargeting's atlas/logmap (intrinsic version of same idea)
  - MeshRet's DMI field (learned version encoding interaction deformations)
- Correspondence problem is the bottleneck -- exactly what SAME/Aberman try to automate

## 6. Cross-Paper Comparison

| Aspect | Deformation Transfer | Semantic DT (Baran 2009) | Aberman 2020 |
|--------|---------------------|--------------------------|--------------|
| Representation | Triangle affine transforms | Patch-based LRI coordinates | Joint rotations + skeleton offsets |
| Correspondence | Manual markers (~50) | Example pose pairs (~5-12) | Automatic via primal skeleton |
| Transfer type | Literal (geometric) | Semantic (functional) | Learned (data-driven) |
| Topology | Different mesh topology OK | Different mesh topology OK | Different skeleton topology OK |
| Domain | Mesh (continuous surface) | Mesh (continuous surface) | Skeleton (articulated graph) |
| For hands | Needs continuous mesh | Needs continuous mesh | Needs homeomorphic skeleton |
