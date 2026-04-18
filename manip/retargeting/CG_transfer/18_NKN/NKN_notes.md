# Neural Kinematic Networks for Unsupervised Motion Retargeting (Villegas et al., CVPR 2018)

> First learning-based unsupervised motion retargeting. Foundational for Aberman 2020 and SAME 2023.

## 1. Core Problem

Transfer motion from one character to another with different bone lengths/proportions, WITHOUT paired training data. Previous methods (Gleicher 1998, Choi & Ko 2000) required manual kinematic constraints per motion type.

## 2. Method Overview

```
Architecture: Encoder RNN + Decoder RNN + differentiable FK layer

Input:  motion sequence x^A_{1:T} from skeleton A (joint positions + global motion)
Output: motion sequence x^B_{1:T} for skeleton B (different bone lengths)

Encoder: h^enc_t = RNN_enc(x_t, h^enc_{t-1})
Decoder: h^dec_t = RNN_dec(x_{t-1}, h^enc_t, s_bar, h^dec_{t-1})
         q_hat_t = W^p * h^dec_t / ||W^p * h^dec_t||   (unit quaternion per joint)
         p_hat_t = FK(q_hat_t, s_bar)                   (forward kinematics)

Key: FK layer is differentiable, embedded in the network
     -> Network outputs joint ROTATIONS, not positions
     -> FK enforces bone length constraints automatically
     -> No post-processing needed for bone length consistency
```

## 3. Key Designs

### 3.1 Differentiable Forward Kinematics Layer
- Takes quaternion rotations + target skeleton bone configuration
- Produces joint positions via recursive FK: p^n = p^{parent(n)} + R^n * s_bar^n
- Fully differentiable -> gradient flows through FK to rotation outputs
- Forces network to discover valid IK solutions implicitly
- This is the paper's most lasting contribution -- reused by Aberman 2020, SAME 2023, GeoRT

### 3.2 Adversarial Cycle Consistency Training
- Cycle consistency: A -> B -> A should recover original motion
  C(x^A_{1:T}, x_hat^A_{1:T}) = ||x^A - x_hat^A||^2
- Adversarial: discriminator assesses whether retargeted motion on B looks natural
- Joint twist loss: penalizes excessive bone twisting (euler angle > 100 degrees)
- Smoothing loss: global motion velocity consistency between frames

### 3.3 Online Inference
- RNN processes frames sequentially -> online retargeting as frames arrive
- No need for full sequence -> applicable to live streaming (e.g., from video)

## 4. Key Results

- Mixamo dataset: ~2400 motions, 71 characters, 7 training + 6 test characters
- 22 joints: full body without fingers
- MSE 7.10 (full model) vs 9.00 (copy quaternions baseline) vs 13.65+ (other baselines)
- Demonstrated video -> 3D character retargeting using off-the-shelf pose estimator

## 5. Limitations & Relevance to Hand Retargeting

**Critical limitations:**
- **Same topology required**: source and target must have identical kinematic structure (same joint names, same tree)
- Only handles different bone LENGTHS, not different number of joints
- Fingers not modeled: "For joints not modeled by our network (e.g., fingers), we directly copy joint rotations"
- RNN temporal model may be overkill for hand retargeting (often frame-independent)

**What transfers:**
- Differentiable FK layer: directly reusable for hand retargeting networks
- Cycle consistency for unsupervised training: if paired hand retargeting data is unavailable, this is the way
- Joint twist loss: relevant for hand joints to prevent unnatural twisting

**What Aberman 2020 fixed:**
- NKN: same topology only -> Aberman: cross-structural via skeletal pooling
- NKN: per-pair training -> Aberman: shared latent space for multiple skeletons

## 6. Historical Significance

This paper established three principles that every subsequent learning-based retargeting method builds on:
1. **Differentiable FK** as a structural prior in the network
2. **Cycle consistency** for unsupervised cross-domain motion translation
3. **Joint rotation output** (not positions) to naturally decouple motion from skeleton structure
