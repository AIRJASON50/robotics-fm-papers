# HandAvatar: Embodying Non-Humanoid Virtual Avatars through Hands (Jiang et al., CHI 2023)
> ~30 citations. CMU Augmented Perception Lab. HCI perspective on hand-to-avatar cross-embodiment mapping.

**Note**: PDF behind ACM paywall, no arxiv preprint available. Notes based on available abstract, project page, and related publications.

## 1. Core Problem

How to enable a user to control **non-humanoid virtual avatars** (e.g., spider, octopus, bird) using their **bare hands**. The challenge is the fundamental topology mismatch: a human hand has 5 fingers with specific joint limits, while target avatars can have arbitrary limb counts, joint structures, and DOF.

This is essentially a **cross-embodiment retargeting** problem from the HCI perspective: mapping 21 hand keypoints to arbitrary avatar skeletons with different topology, optimizing for both controllability and user comfort.

## 2. Method Overview (with pseudocode)

```
# HandAvatar Pipeline (inferred from available descriptions)
Input: hand_skeleton (21 joints), avatar_skeleton (arbitrary joints)

# Step 1: Observation study
# Conducted user study on 8 avatars to understand mapping preferences
# Found: users prefer mappings that preserve structural similarity

# Step 2: Automated mapping generation
# Joint optimization of three objectives:
mapping = optimize(
    control_precision,       # accuracy of avatar pose control
    structural_similarity,   # topological correspondence quality
    comfort                  # ergonomic cost for the user
)

# Step 3: Joint-to-joint mapping
# Generates explicit joint correspondences between hand joints and avatar joints
# Handles many-to-one and one-to-many mappings for different topologies

Output: mapping(hand_pose) -> avatar_pose
```

## 3. Key Designs

### 3.1 Three-Objective Joint Optimization

The mapping is found by jointly optimizing:
1. **Control precision**: how accurately the avatar follows intended hand movements
2. **Structural similarity**: how well the mapping preserves topological relationships between hand and avatar
3. **Comfort**: how much physical effort the user needs (penalizes extreme joint angles, awkward postures)

This multi-objective formulation is relevant because it explicitly acknowledges that pure kinematic accuracy is insufficient -- ergonomic and structural constraints matter.

### 3.2 Observation Study on Mapping Preferences

Studied 8 non-humanoid avatars (likely including multi-limbed, winged, serpentine characters). Key finding: users prefer mappings where **structurally similar** sub-chains in the hand are mapped to **semantically corresponding** parts of the avatar. This is empirical evidence that structural/topological similarity matters for intuitive cross-embodiment control.

### 3.3 Evaluation Across Three Task Types

- **Static posing**: hold avatar in specific poses
- **Dynamic animation**: continuous motion control
- **Creative exploration**: free-form interaction

Results: more precise control, less physical effort, comparable embodiment vs. body-to-avatar baseline.

## 4. Key Results

- Tested on 8 non-humanoid avatars with varying topology
- User study: HandAvatar provides more precise control and requires less physical effort
- Comparable sense of embodiment to full-body control methods
- Applications demonstrated: VR social interaction, 3D animation composition, VR scene design

## 5. Limitations & Relevance to Hand Retargeting

**Limitations** (inferred):
- HCI-focused, not RL/robotics-focused -- optimization likely for user experience, not physical accuracy
- Likely uses position-based mapping rather than contact-aware transfer
- No mention of object interaction scenarios
- Evaluation is subjective (user study) rather than quantitative kinematic metrics

**Relevance to our problem (MediaPipe 21 -> WujiHand 20 DOF)**:

| Aspect | Applicable? | Notes |
|--------|-------------|-------|
| Hand as source modality | Yes (directly) | They also start from hand skeleton (likely MediaPipe-like) |
| Cross-topology mapping | Yes (concept) | Multi-objective optimization for joint correspondence |
| Structural similarity metric | Potentially | Their metric for "how similar are two skeleton sub-chains" could inform our mapping |
| Comfort optimization | No | Irrelevant for robot hand (no user fatigue) |
| Object interaction | No | VR avatar control, not manipulation |

**Key takeaway**: HandAvatar validates the approach of using hand dexterity as a universal controller for arbitrary embodiments. Their finding that users prefer structurally-similar mappings supports our approach of explicit joint correspondence (rather than purely learned latent spaces). However, the HCI framing means the method is optimized for user experience rather than physical accuracy or contact preservation, making it less directly applicable to our manipulation-focused retargeting problem.

The most transferable idea is the **multi-objective formulation** for finding joint correspondences -- we could add a "structural similarity" term to our optimization that prefers mappings where kinematic chain structure is preserved.

## 6. Cross-Paper Comparison

| Feature | HandAvatar | R2ET | SAME | Motion2Motion |
|---------|-----------|------|------|---------------|
| Source | Human hand (21 joints) | Full body skeleton | Any skeleton | Any skeleton |
| Target | Non-humanoid avatars | Same-topology body | Any skeleton | Any skeleton |
| Approach | Multi-objective optimization | Residual neural network | Skeleton-agnostic GCN | Training-free patch matching |
| Cross-topology | Yes (hand -> arbitrary) | No | Yes | Yes |
| Evaluation | User study (HCI) | MSE + penetration | Joint angle error | FID + frequency |
| Object interaction | No (VR control) | No | No | No |
| Real-time | Yes | Yes | Yes | Yes |

HandAvatar is unique in starting from hand specifically (our exact source modality) and mapping to arbitrary targets. But its HCI focus means it prioritizes user comfort over kinematic fidelity. For our robotics application, the mapping quality needs to be evaluated by contact preservation and task success rather than user preference.
