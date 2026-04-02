# Action Space & Controller Design Review

> 22 papers across humanoid / dexterous manipulation / robotics FM
> Question: policy outputs what? Is there an external controller between policy and robot?

---

## Comparison Table

| # | Paper | Robot Type | Policy Output | External Controller | Policy Hz | Ctrl Hz |
|---|-------|-----------|--------------|-------------------|-----------|---------|
| **Humanoid -- Motion Tracking** |
| 1 | DeepMimic (18) | Humanoid (sim) | Joint angle targets (PD target) | PD position ctrl | -- | -- |
| 2 | PHC (23) | Humanoid (sim) | Joint angle targets (multi-primitive weighted) | PD position ctrl | -- | -- |
| 3 | BeyondMimic (25) | Humanoid (real, Berkeley Blue) | Joint angle targets + latent diffusion planner | PD position ctrl | -- | -- |
| 4 | SONIC (25) | Humanoid 29-DOF (Unitree G1) | 29D joint position targets | PD position ctrl | 50 | 500 |
| **Humanoid -- Teleoperation** |
| 5 | H2O/OmniH2O (24) | Humanoid (Unitree H1) | Joint angle targets | PD position ctrl | -- | -- |
| 6 | FPO++ (26) | Humanoid (Booster T1 / G1) | Joint angle targets (via flow policy) | PD position ctrl | -- | -- |
| **Humanoid -- Sim2Real / World Model** |
| 7 | ASAP (25) | Humanoid (Unitree H1) | Joint angle targets + delta action model | PD position ctrl | -- | -- |
| 8 | RWM (25) | Quadruped + Humanoid (G1) | Joint angle targets (MBPO-PPO) | PD position ctrl | -- | -- |
| 9 | HDMI (25) | Humanoid (Unitree G1) | Residual joint angle (delta from ref) | PD position ctrl | -- | -- |
| **Manip -- Traditional RL** |
| 10 | ArtiGrasp (23) | MANO hand (51-DOF, sim) | Residual joint angle (PD target) | PD + feedforward torque | 100 | 400 |
| 11 | ObjDexEnvs (24) | Arm + DexHand (60D) | Arm: EE delta (IK->joint); Hand: absolute joint pos | IK (arm) + PD (hand) | -- | -- |
| **Manip -- Human2Robot** |
| 12 | DexMachina (24) | Floating DexHand (6 types) | Wrist: residual from retarget; Finger: absolute joint pos | PD position ctrl | -- | -- |
| 13 | DexTrack (25) | Allegro 22-DOF | Double-integration residual (acceleration-level) | PD (kp=20, kd=1) | -- | -- |
| **Manip -- Scaling RL** |
| 14 | OmniReset (26) | UR7e arm (no hand) | 6D EE pose delta | OSC (Jacobian->torque) | -- | 500 |
| **Manip -- Sim2Real** |
| 15 | SimToolReal (25) | Arm + DexHand (16-DOF) | Arm: EE delta; Hand: absolute joint pos | IK (arm) + PD (hand) | -- | -- |
| 16 | Dex4D (25) | Allegro + arm | Joint pos targets (via DAgger student) | PD position ctrl | 5 | -- |
| **Manip -- FM Integration** |
| 17 | RLToken (26) | Arm (pi 0.6 robot) | Action chunk (C=10, 14D joint+gripper) | Low-level PID | 50 | -- |
| 18 | UltraDexGrasp (26) | UR5e + XHand (12-DOF) | Joint position targets (BC, truncated normal) | PD position ctrl | 10 | -- |
| **Robotics Foundation Models** |
| 19 | PI Series (pi_0 -> pi*0.6) | Multi-arm (7-DOF each) | Joint angle chunk (flow matching / DCT tokens) | PD/PID position ctrl | 50 | -- |
| 20 | GR00T (N1->N1.6+SONIC) | Humanoid 29-DOF | VLA->motion trajectory->SONIC->joint pos targets | PD position ctrl | VLA ~5, SONIC 50 | 500 |
| 21 | Diffusion Policy (23) | Arm (Franka etc.) | EE position / joint position sequence | Position ctrl (robot SDK) | ~10 | -- |
| 22 | ACT (23) | Dual-arm ViperX (14D) | Joint position chunk (k=100) | PID position ctrl | 50 | -- |

---

## Analysis

### 1. Pattern: robot type determines action space (机器人类型决定动作空间)

**Humanoid (full-body)** -- **100% joint position targets + PD controller**

All 9 humanoid papers output joint angle targets, tracked by a PD controller at higher frequency (typically 500 Hz). No exceptions. The reason is clear: humanoids need whole-body coordination including balance (balancing, 平衡), and Cartesian/EE-space control cannot express full-body posture. Motion tracking reward naturally maps to joint-level control.

**Dexterous hand (fingers)** -- **Joint position (absolute or residual) + PD controller**

All hand-manipulation papers use joint-level position targets for fingers. The PD controller is always present. The choice between absolute vs residual depends on whether reference motion is available:
- With reference: residual (DexMachina wrist, HDMI, DexTrack, ArtiGrasp)
- Without reference: absolute (DexMachina fingers, SimToolReal hand, UltraDexGrasp)

**Arm (manipulation)** -- **Mixed: EE delta + IK/OSC for arm, joint pos for hand**

This is the key split. When an arm + hand system is used:
- Arm controlled via EE delta -> IK or OSC converts to joint torque/position
- Hand controlled via direct joint position
- Examples: ObjDexEnvs, SimToolReal, OmniReset (arm-only, OSC)

**Foundation models (VLA)** -- **Joint position chunks**

PI, ACT, GR00T all output joint-level position sequences (action chunks). The chunk abstracts away the frequency gap: policy runs at 5-50 Hz, chunks executed at 50 Hz by interpolation/PID.

### 2. Tradeoffs (权衡)

| Action Type | Pros | Cons | Best For |
|------------|------|------|----------|
| Joint pos targets + PD | Simple; full-body expressivity; stable with DR | Need accurate joint model; no force control | Humanoid, dex hand |
| EE delta + OSC/IK | Task-space intuition; low-dim search | Cannot express full posture; IK singularity | Arm reaching/placing |
| Residual from reference | Narrow search space; faster convergence | Need reference trajectory; less flexible | Motion tracking |
| Double-integration residual | Smooth output; auto-rate-limiting | Slower response; needs reference | Fine manipulation (DexTrack) |
| Joint torque (direct) | Full physical control | Extremely hard to learn; sim2real nightmare | Almost nobody uses this |
| Action chunk (VLA) | Temporal consistency; reduced compounding error | Latency; cannot react within chunk | Foundation model deployment |

**Key insight**: nobody outputs raw torques. The PD controller is the universal interface between policy and robot, providing:
- Compliance (柔顺性) against unexpected contacts
- Frequency decoupling: policy at 5-50 Hz, execution at 200-1000 Hz
- Sim2real robustness: PD gains are easier to transfer than raw torques

### 3. Trend (趋势)

**Past (2018-2023)**: Per-paper action space design. Each paper picks its own formulation.

**Present (2024-2026)**: Convergence on a two-layer pattern:

```
Foundation Model / RL Policy  (5-50 Hz)
    outputs: joint position targets (chunk or per-step)
        |
    PD Controller  (200-1000 Hz)
    outputs: joint torques
        |
    Robot Hardware
```

For humanoids specifically, a **three-layer** pattern emerges (GR00T):

```
VLA / High-level  (~5 Hz)   -> motion trajectory / latent tokens
WBC / Mid-level   (50 Hz)   -> joint position targets
PD  / Low-level   (500 Hz)  -> joint torques
```

**Trend direction**:
1. **Joint position is the winner** -- even FM papers (PI, ACT) output joints, not EE
2. **Residual is replacing absolute** for tasks with reference motions
3. **Action chunking** is becoming standard for FM-based policies
4. **OSC/IK are shrinking** in scope -- only used for simple arm reaching, not for dexterous tasks
5. **Flow matching / diffusion** as policy representation, but output is still joint positions

### Conclusion (结论)

The action space question has a clear answer in 2025-2026: **joint position targets + PD controller is the universal standard**. The real design choices are now at higher levels:
- Absolute vs residual (whether reference motion exists)
- Single-step vs chunk (whether temporal consistency matters)
- How many layers of hierarchy (1 for arm, 2 for dex hand, 3 for humanoid)

The PD controller is NOT going away -- it is the essential bridge that decouples policy learning frequency from hardware execution frequency, and provides the compliance needed for contact-rich tasks.
