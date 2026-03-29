# HOP - Supplementary Notes (Code Deep-Dive)

Supplement to `HOP_notes.md`. Focuses on implementation details, code-paper divergences, and architectural specifics not covered in the main notes.

---

## 1. Transformer Architecture: Exact Specifications

### 1.1 Model Dimensions (from `AllegroXarmNew.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_dim` | 192 | All embedding dimensions |
| `n_layer` | 4 | Transformer blocks |
| `n_head` | 4 | Attention heads (48 dim/head) |
| `context_length` | 16 | History window (0.8s at 20Hz) |
| `proprio_dim` | 23 | 7 arm + 16 hand |
| `action_dim` | 23 | Same as proprio |
| `pc_num` | 100 | Points per point cloud |
| Dropout (all) | 0.0 | attn, resid, embd all zero |

This is very small by NLP standards -- roughly 750K parameters in the transformer blocks alone. Choosing 192 hidden dim with 4 heads gives 48 dim/head, which is significantly smaller than standard GPT-2's 64 dim/head.

### 1.2 Token Sequence Construction

The model interleaves proprioception and point cloud tokens at each timestep. With `action_input=False` (default for pretraining):

```
Sequence: [proprio_0, pc_0, proprio_1, pc_1, ..., proprio_15, pc_15]
Total tokens: 2 * 16 = 32
```

When `action_input=True` (3-modality mode):
```
Sequence: [proprio_0, pc_0, action_0, proprio_1, pc_1, action_1, ...]
Total tokens: 3 * 16 = 48
```

Each token gets a **learned timestep embedding** added before entering the transformer. The timestep embedding table has `n_ctx + 1` entries (33 for default), with the extra entry reserved for padding.

### 1.3 PointNet Encoder

```python
embed_pc = nn.Sequential(
    nn.Linear(3, 192),
    nn.ELU(inplace=True),
    nn.Linear(192, 192),
    nn.ELU(inplace=True),
    nn.Linear(192, 192),
    nn.MaxPool2d((100, 1))  # pool over 100 points
)
```

Key observations:
- Uses **ELU** activation, not ReLU (unlike canonical PointNet which uses ReLU + batch norm)
- No batch normalization or T-Net (simplified PointNet)
- MaxPool2d is used to collapse the 100-point dimension into a single vector
- The pooling happens over the first spatial dimension (pc_num), preserving the feature dimension

### 1.4 Prediction Heads

| Head | Architecture | Purpose |
|------|-------------|---------|
| `predict_action` | Linear(192, 23) | Action prediction (main) |
| `predict_proprio` | Linear(192, 23) | Next proprioception (auxiliary) |
| `predict_pc` | Linear(192, 300) | Next point cloud flattened (auxiliary) |

No `action_tanh` in default config (`action_tanh: false`), so action outputs are unbounded during pretraining. During RL finetuning, actions are clamped to [-1, 1].

### 1.5 Which Hidden State Predicts What

With 2-modality mode (`action_input=False`):
- `x[:, 1]` (pc tokens) -> `predict_action`: action prediction reads from the point cloud stream
- `x[:, 0]` (proprio tokens) -> `predict_proprio`: next proprio reads from the proprio stream

This is a deliberate cross-modal design: the action prediction is conditioned on the point cloud representation (which already attended to the proprioception via causal attention), not on the proprioception stream directly.

---

## 2. Pretraining Data Pipeline (Detailed)

### 2.1 Data Format

Each trajectory is stored as a `.npy` file containing a dict:
```python
{
    'robot_qpos': List[np.array],    # (T, 23) actual joint positions
    'target_qpos': List[np.array],   # (T, 23) target joint positions (actions)
    'object_pc': List[np.array],     # (T, 100, 3) object point clouds
}
```

Directory structure: `retarget_data/{train,val}/subject_*/trajectory_*.npy`

### 2.2 Joint Order Remapping (Critical Implementation Detail)

The retargeting data uses a different joint ordering than IsaacGym. The mapping is:

```python
IG_mapping = [0,1,2,3,4,5,6,  # arm joints (unchanged)
              7,8,9,10,        # index finger (unchanged)
              19,20,21,22,     # thumb (was indices 19-22, moved to 11-14)
              11,12,13,14,     # middle finger (was 11-14, moved to 15-18)
              15,16,17,18]     # ring finger (was 15-18, moved to 19-22)
```

The key reordering: thumb joints (originally at indices 19-22 in the retargeting data) are moved to positions 11-14 in the IsaacGym order. This swap of thumb and middle/ring finger positions is crucial for correct behavior.

### 2.3 Joint Scaling

Joint positions are scaled to [-1, 1] using hardcoded joint limits from the Allegro hand + xArm 7:
- xArm 7: limits roughly [-6.28, 6.28] for revolute joints (some asymmetric)
- Allegro hand: typical range [-0.47, 1.71] for finger joints

Scaling formula: `q_scaled = 2 * (q - lower) / (upper - lower) - 1`

The same limits are hardcoded in both `RobotDataset` (pretraining) and `RTActorCritic` (finetuning), which is fragile but works because both are for the same robot.

### 2.4 Data Augmentation (Noise Injection)

During pretraining, noise is added to both proprio and action inputs (but NOT to the target action labels):

**Default mode** (`add_data_driven_noise=False`):
```python
noise_arm = noise_hand = 0.1  # in scaled [-1,1] space
# Arm joints: N(0, 0.1) noise
# Hand joints: N(0, 0.1) noise
```

**Data-driven mode** (`add_data_driven_noise=True`, disabled by default):
Per-joint Gaussian noise with statistics computed from the dataset:
- Arm joints (0-6): std ~ 0.01-0.04
- Hand joints (7-22): std ~ 0.01-0.09 (higher variance, especially for thumb/index)
- Thumb joints have highest noise tolerance (std up to 0.08-0.09)
- Some joints have near-zero noise (the "zero" entries in the data-driven stats)

---

## 3. GPT-2 Pretraining: How It Actually Works

### 3.1 Training Loop

```
For each epoch:
    For each batch of 256 windows:
        1. Sample (proprio, action, pc) windows of length 16
        2. Add noise to proprio and action inputs (NOT targets)
        3. Forward pass through transformer
        4. Compute L1 loss between predicted and target actions
        5. Gradient clip at 0.25
        6. AdamW step (lr=1e-4, weight_decay=0.01)
```

Training runs for 100 epochs. No learning rate scheduler (cosine annealing is defined but set to `None`).

### 3.2 Time Shift Mechanism

`time_shift=0` in default config, but the code supports a time shift parameter that shifts the prediction target forward in time:

```python
if self.action_shift > 0:
    action_target = torch.roll(action_target, shifts=-self.action_shift, dims=1)
    action_target = action_target[:, :-self.action_shift, :]
    action_preds = action_preds[:, :-self.action_shift, :]
```

When `time_shift=0` and `modality_aligned=False`:
- `proprio_shift = 1`: predict next-step proprio
- `action_shift = 0`: predict current-step action (i.e., the action given current observation)
- `pc_shift = 0`: predict current point cloud

This means the default pretraining objective is: given history of proprio and pc, predict the action at each timestep (not future action).

### 3.3 Modality Alignment Options

The code supports two token ordering conventions:
- `modality_aligned=False` (default): action is predicted from the pc token at the same timestep
- `modality_aligned=True`: action is predicted from the action token at the same timestep (autoregressive over actions too)

### 3.4 Multi-GPU Pretraining

The pretrain script launches with `num_gpus=4` and supports DDP via `MultiGPUTrainer`. The training command:
```bash
python scripts/pretrain.py num_gpus=4 headless=True \
    pretrain.training.root_dir=$DATADIR/train \
    pretrain.validation.root_dir=$DATADIR/val \
    pretrain.wandb_activate=True seed=-1
```

### 3.5 Pretraining with Video Capture

An interesting feature: the pretrain script can optionally instantiate an IsaacGym environment and periodically run the model in simulation to visualize pretraining progress. This is controlled by `task.env.enableVideoLog=True`.

---

## 4. RL Finetuning: Complete Architecture

### 4.1 Actor-Critic Split

```
Actor (pretrained transformer):
  Input: proprio_hist (16, 23) + pc_hist (16, 100, 3)
  Processing: Timestep embed + Interleave + GPT-2 + predict_action head
  Output: mu (23,)  -- deterministic mean action

Critic (randomly initialized MLP):
  Input: obs (full_state_size,) with pc replaced by PointNet embed
  Processing: Linear(input, 512) -> ELU -> Linear(512, 256) -> ELU
              -> Linear(256, 128) -> ELU -> Linear(128, 1)
  Output: value (1,)
```

### 4.2 Exploration Noise

The logstd is a **learned parameter** (state-independent) initialized per joint group:

| Config | Grasp | Throw | Cabinet |
|--------|-------|-------|---------|
| initEpsArm | 0.1 | 0.1 | 0.5 |
| initEpsHand | 0.1 | 0.1 | 0.5 |

Cabinet task uses 5x higher initial exploration -- makes sense because cabinet opening requires exploring a wider action space (the pretrained prior is primarily about grasping, not pulling).

### 4.3 Critic Warmup

```python
if self.gradient_steps < self.warmup_critic_gradsteps:  # 200 steps
    loss = 0*a_loss + c_loss * self.critic_coef - entropy * 0
else:
    loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef
```

During warmup, the actor loss is multiplied by 0 (but kept in the computation graph via `0*a_loss` to avoid unused parameter issues). This ensures the pretrained actor weights are preserved while the randomly initialized critic catches up.

### 4.4 PPO Hyperparameters (from config)

| Parameter | Default | Grasp/Throw | Cabinet |
|-----------|---------|-------------|---------|
| learning_rate | 1e-4 | **1e-5** (finetune) | **1e-5** |
| gamma | 0.99 | 0.99 | 0.99 |
| tau (GAE lambda) | 0.95 | 0.95 | 0.95 |
| e_clip | 0.2 | 0.2 | 0.2 |
| mini_epochs | 1 | 1 | 1 |
| horizon_length | 10 | 10 | 10 |
| minibatch_size | 4096 | 512 | 512 |
| num_envs | 4096 | 512 | 512 |
| critic_coef | 4 | 4 | 4 |
| entropy_coef | 0.0 | 0.0 | 0.0 |
| bounds_loss_coef | 0.0001 | 0.0001 | 0.0001 |
| grad_norm | 1.0 | 1.0 | 1.0 |
| kl_threshold | 0.02 | 0.02 | 0.02 |
| value_bootstrap | True | True | True |
| max_agent_steps | 5B | 5B | 5B |

Key differences from typical PPO:
- **mini_epochs=1**: only 1 pass through collected data (very conservative to preserve pretrained features)
- **entropy_coef=0.0**: no entropy bonus at all (relying on logstd for exploration)
- **Learning rate 10x lower** for finetuning (1e-5 vs 1e-4 for pretraining)
- **horizon_length=10**: very short rollouts (0.5s at 20Hz)

### 4.5 Reward Scaling Pipeline

1. Environment computes raw reward (sum of reward components)
2. PPO multiplies by 0.1: `shaped_rewards = 0.1 * rewards`
3. Value bootstrap for timeouts: `shaped_rewards += gamma * V * timeout_mask`
4. Value normalization: RunningMeanStd normalizes both values and returns

---

## 5. Reward Functions: Detailed Breakdown

### 5.1 Grasp & Lift Task (`AllegroXarmNew`)

Enabled reward components and their scales:

| Component | Scale | Formula | Phase |
|-----------|-------|---------|-------|
| `fingertip_delta_rew` | 50.0 | `clip(closest_dist - curr_dist, 0, 10) * finger_coeffs` | Pre-lift |
| `lifting_rew` | 20.0 | `clip(z_lift + 0.02, 0, 0.5)` | Pre-lift threshold |
| `lift_bonus_rew` | 300.0 | One-time bonus when object lifts > 10cm | Transition |
| `keypoint_rew` | 200.0 | `clip(closest_keypoint_dist - curr_dist, 0, 100)` | Post-lift |
| `bonus_rew` | 1000/50=20 per step | When keypoints within success_tolerance | Success |
| `kuka_actions_penalty` | -0.003 | `sum(abs(arm_dof_vel))` | Always |
| `allegro_actions_penalty` | -0.0003 | `sum(abs(hand_dof_vel))` | Always |

Disabled: `palm_delta_rew`, `hand_delta_penalty`, `fingertip_shape_delta_rew`, `hand_joint_pose_rew`

**Phase-gating pattern** (not in paper): Rewards are phase-gated using the `lifted_object` boolean flag:
1. **Pre-lift**: Fingertip approach rewards encourage approaching the object
2. **Lift transition**: 300-point bonus for lifting > 10cm, then lifting reward stops
3. **Post-lift**: Keypoint reward guides the object to the goal position
4. **Success**: Additional bonus spread over `success_steps=50` frames

### 5.2 Throwing Task (`AllegroXarmThrowing`)

Same reward structure as Grasp & Lift but with different goal positions (basket location instead of fixed height). The code is nearly identical (~2000 lines, mostly duplicated).

### 5.3 Cabinet Task (`AllegroXarmCabinet`)

Completely different reward structure:

| Component | Scale | Formula |
|-----------|-------|---------|
| `finger_dist_reward` | 0.04 | `1.0 / (0.04 + dist)` per finger |
| `thumb_dist_reward` | 0.08 | 2x finger scale (thumb more important) |
| `around_handle_reward` | 0.0 (disabled) | Index above handle, thumb below |
| `open_bonus_reward` | 2.0 | Binary: drawer opened > 1cm |
| `goal_dist_reward` | 6.0 | `0.4 - to_goal` (drawer opening distance) |
| `open_pose_reward` | 0.0 (disabled) | Maintain grip while opening |
| `goal_bonus_reward` | 2.0 | Binary: fully opened (to_goal < 0.1) |
| `action_penalty` | -0.01 | `sum(actions^2)` |

---

## 6. Environment Details (Not in Paper)

### 6.1 Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics dt | 1/120 s (0.00833s) |
| Control freq | 20 Hz (controlFrequencyInv=6) |
| Episode length | 600 steps (30s) |
| Substeps | 2 |
| Solver | TGS (solver_type=1) |
| Position iterations | 8 |

### 6.2 PD Controller Gains

| Joint Group | Stiffness (Kp) | Damping (Kd) | Effort Limit |
|-------------|---------------|--------------|--------------|
| xArm 7 | 1000 | 100 | 500 Nm per joint |
| Allegro hand | 40 | 5 | 0.35 Nm |

Very high arm stiffness (1000) vs hand stiffness (40) -- the arm tracks position commands very precisely while the hand has more compliance. This is a deliberate choice for sim-to-real (real Allegro hand has limited torque).

### 6.3 Object Representation

Point cloud is generated at runtime by rotating pre-computed object surface points:
```python
self.point_cloud_buf = quat_apply(
    self.object_rot[:, None].repeat(1, pc_num, 1),
    self.object_point_clouds
) + self.object_pos[:, None]
```

Point clouds are **scaled** by a factor of 0.01 (`pointCloudScale: 0.01`) -- converting from meters to centimeters in the observation space. This is not mentioned in the paper.

### 6.4 Objects Used

Training objects (from config `simpleObjects`):
- Grasp: YCB objects `['002', '036', '010', '025', '024', '005']` (6 objects)
- Throw: `['002', '011', '036', '010', '025', '024', '005', '007']` (8 objects)
- Cabinet: No grasping objects, but cabinet with drawer

OOD evaluation uses DexYCB set: 13-19 objects including power drill, cracker box, etc.

### 6.5 Domain Randomization

Default config has all randomization **disabled**:
- `randomize_friction: False`
- `randomize_object_mass: False`
- `randomize_object_com: False`
- `randomize_table_position: False`

This is surprising -- no domain randomization during RL finetuning. The diversity comes from the pretrained prior (which was trained on retargeted data with scene randomization during IK optimization).

### 6.6 Init Pose Versioning

The code contains 17 different initial arm poses (`v1` through `v17`). The default for all tasks is `v16`:
```python
"v16": [-0.79, -0.30, 0.16, 0.43, 2.20, 0.72, -1.67]
```

This represents a specific arm configuration where the hand is positioned above the table, palm facing down -- suitable for top-down grasping.

---

## 7. Notable Code Design Patterns

### 7.1 Delta Reward with Closest-Distance Memory

The reward functions use a "ratcheting" mechanism: they track the closest distance the fingertips have ever been to the object, and reward only improvements beyond that record:

```python
# Only reward when getting closer than ever before
delta = closest_dist - current_dist  # positive = improvement
closest_dist = min(closest_dist, current_dist)  # update record
reward = clip(delta, 0, 10)  # only positive improvements
```

This prevents the agent from oscillating back and forth for repeated reward. Once a distance milestone is passed, it becomes the new baseline.

### 7.2 PointNet Feature Sharing Between Actor and Critic

The value function can optionally use PointNet features from the actor:
```python
if self.pc_to_value:
    obs = torch.cat([obs[:,:pc_begin], pc_embed[:,-1], obs[:,pc_end:]], dim=1)
```

This replaces the raw 300-dim point cloud (100*3) with the 192-dim PointNet embedding in the value function input. The gradient flow to the PointNet is controlled by `value_grads_to_pointnet`:

- Pretraining default: `value_grads_to_pointnet: True` (share gradients)
- Finetuning override: `value_grads_to_pointnet: False` (detach -- protect pretrained PointNet)

### 7.3 Observation History Buffer Management

The environment maintains history buffers as part of the observation dict:
```python
obs_dict = {
    'obs': full_state_vector,        # (num_envs, full_state_size)
    'proprio_buf': proprio_history,   # (num_envs, ctx_len, 23)
    'pc_buf': pc_history,            # (num_envs, ctx_len, 100, 3)
    'action_buf': action_history,    # (num_envs, ctx_len, 23)
    'attn_mask': attention_mask,     # (num_envs, ctx_len)
    'timesteps': timestep_indices,   # (num_envs, ctx_len)
}
```

The attention mask handles episode boundaries -- at reset, the mask zeros out history from previous episodes. This is critical for the causal transformer to not attend to stale data across episode boundaries.

### 7.4 Memory-Efficient Experience Buffer

There's an alternative `MemoryEfficientExperienceBuffer` that stores only the latest timestep of proprio/pc/action history per rollout step (instead of the full context window). The full context window is reconstructed during training by looking back through the buffer. This trades compute for memory.

### 7.5 JIT Export Pipeline

The code supports exporting the finetuned policy as a TorchScript JIT model for real robot deployment:
```python
saving_model = SavingModel(actor_critic_model, running_mean_std)
model_trace = jit.trace(saving_model, obs)
jit.save(model_trace, 'policy.jit.pt')
```

The `SavingModel` wrapper chains normalization and inference in a single forward pass.

---

## 8. Paper Details Not in Existing Notes

### 8.1 Retargeting: Exact Procedure

From Section 4 (Retargeting):
- Optimizer: L-BFGS from NLOpt library (not gradient descent)
- Scene: Ground floor + static table at 65cm from robot
- Optimization runs: **700 per video** with randomized table location and initial joint state
- Filtering: Trajectory kept only if retargeting error < **3cm** at any time AND no arm-table/floor collisions
- Code base: Built upon Qin et al. (DexMV)

### 8.2 7-DoF vs 6-DoF Arm

Paper explicitly mentions: "we empirically found that since the robot base is fixed, a 7-DoF arm can track much better human motions than a 6-DoF arm, which often encounters singularities during such trajectories."

### 8.3 Two Separate Base Policies

Paper mentions training **two** base policies from the same trajectory data:
1. **Point cloud policy** for simulation experiments
2. **Depth image policy** for real-world BC experiments

The depth trainer infrastructure exists in code (`depth_trainer.py`, `depth_trainer_multigpu.py`) but the depth model is not in the released `RobotTransformerAR` class -- suggesting the depth variant may use a different architecture or the code is not fully released.

### 8.4 Inference Speed

The policy runs at 20Hz. The xArm low-level controller runs at 120Hz, Allegro at 300Hz. The policy outputs absolute joint position targets.

### 8.5 Moving Average Action Space

Paper notes: "training from scratch is unsuccessful using joint-position control... we use the moving-average action space proposed by Petrenko et al."

Config parameter `actionsMovingAverage: 1.0` -- when set to 1.0, this effectively disables the moving average (target = action directly). The baseline PPO presumably uses a different value. HOP doesn't need this because the pretrained prior already produces smooth trajectories.

### 8.6 Camera Setup

Single Zed-2 stereo camera mounted on the robot's right side. For the depth-based real-world experiments, only depth images are used (not RGB). Camera parameters:
- FOV: 60 degrees
- Resolution: 60x60 (reduced from full resolution)
- Single camera (not multi-view)

---

## 9. Code Quality and Debt

### 9.1 Significant Code Duplication

`xarm_grasping_new.py`, `xarm_throwing.py`, and `xarm_cabinet.py` are ~2000+ lines each with massive duplication. The grasping and throwing tasks share ~95% of the code (diff is mainly goal definition and throwing-specific reward shaping). A proper inheritance hierarchy would reduce this by ~4000 lines.

### 9.2 Commented-Out Code

Extensive commented-out code throughout, including:
- Entire rl_games configuration blocks in PPO yaml files
- Camera sensor code in task files
- Alternative reward formulations
- KV caching TODO in `get_action()`

### 9.3 Missing Error Handling

`restore_train` has a bare `except` clause:
```python
try:
    self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
except:
    print("Could not restore running mean std for value function")
```

### 9.4 Inconsistent Optimizer Choices

- Pretraining: AdamW (with weight_decay)
- Finetuning: Adam (with optional weight_decay, default 0.0)
- MultiGPUTrainer: Adam (not AdamW)

---

## 10. Additional Paper-Code Divergences (Beyond Existing Notes)

### 10.1 Default Config Has Exploration Noise = 1.0

The default PPO config (`AllegroXarmNewPPO.yaml`) sets `initEpsArm: 1.0` and `initEpsHand: 1.0`. The finetune scripts override this to 0.1. If someone runs the default config without the shell script, they'd get 10x the intended exploration noise.

### 10.2 The "Privileged Actions" Mechanism

The task code supports a `privilegedActions` mode that adds 3 extra action dimensions for directly commanding object position:
```python
if self.privileged_actions:
    self.num_allegro_kuka_actions += 3
```
This is disabled in all configs (`privilegedActions: False`) but represents infrastructure for a "cheating" mode used during development.

### 10.3 DAPG Support

The PPO config includes DAPG (Demo Augmented Policy Gradient) parameters that are referenced in the code but not in the paper's HOP training:
```yaml
dapg:
    l1: 0.1
    l2: 0.999
    dapg_threshold: 0.002
```

### 10.4 Point Cloud Normalization Options

The PPO config has `normalize_pc: False` and `normalize_proprio_hist: False`, but the infrastructure for running mean/std normalization of these inputs exists. These were likely tried and found unhelpful.

### 10.5 Moving Average Action Space Is NOT Used by HOP

Despite the paper mentioning "moving average action space" for the PPO baseline, the HOP finetuning config sets `actionsMovingAverage: 1.0` (which means no smoothing). The paper states this is needed "for PPO from scratch" but not for HOP since the pretrained prior already produces smooth actions.

---

## 11. Conclusion Supplement: The Paper's Key Limitation Admission

From Section 7, the paper's most candid admission:

> "While our approach demonstrates a way to pre-train on a single object interaction, this can, in practice, be limiting. Indeed, human behavior in a video can potentially be conditioned on information encompassing multiple objects in the current and previous scenes. This leads to a loss of signal that could be extracted from the raw video."

This is a fundamental limitation: the pipeline reduces rich video to single-hand-single-object 3D interactions, discarding scene context, multi-object relationships, and human intent. The paper predicts that "advances in 3-D reconstruction will enable us to use a more complex scene reconstruction and pretraining."
