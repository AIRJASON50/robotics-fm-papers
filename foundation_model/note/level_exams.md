# Foundation Model Learning Path -- Upgrade Exams

You are a robotics researcher with PPO sim2real dexterous manipulation experience.
You study CS (NLP/CV) foundation models to understand:
- How VLA (Vision-Language-Action) models work and why they are designed this way
- How to transfer encoding/representation ideas from language/vision to robot actions
- The complete technical lineage: Transformer -> GPT -> CLIP -> Diffusion -> VLA
- Not to become an NLP/CV expert, but to read VLA papers fluently and understand the "why" behind architecture choices

**Exam format**: Transfer-type questions. Not "explain X", but "how would X change your dexterous manipulation work?"
**Rules**: 5 questions per level. Answer from understanding, no reference materials. 80% to advance.

---

## Level 0 -> Level 1 Exam: What Patterns Transfer from CS to Robotics (passed, 90/100)

**Scope**: Representation learning + Transformer fundamentals

1. Your PPO policy for in-hand cube rotation uses a 3-layer MLP with joint angles + object pose as input. From the perspective of representation learning (Bengio 2013), what information is this representation missing, and how would adding a pre-trained visual encoder change the generalization behavior of your policy?

2. Transformer's self-attention replaced RNN's sequential computation. If you were to replace the MLP in your PPO policy with a Transformer that takes a sequence of (state, action, reward) tuples, what would change about how the policy uses history? How does this connect to Decision Transformer?

3. pi_0 uses separate weight groups for VLM backbone and action expert, connected through attention. Explain this design choice using the concept of "representation interference" -- what happens if you train vision-language understanding and continuous action generation with a single set of weights?

4. Your sim2real pipeline uses domain randomization to bridge the reality gap. From the representation learning perspective, what does domain randomization actually do to the learned representation? Is it creating invariance or equivariance, and why does this matter?

5. Transformer is called an "information routing" innovation, not a "loss" innovation. The same Transformer backbone powers GPT (next-token loss), BERT (masked prediction loss), and pi_0 (flow matching loss). What does this architectural universality mean for your future work -- should you invest more in designing better architectures or better training objectives for dexterous manipulation?

---

## Level 1 -> Level 2 Exam: Pre-training + Scaling Understanding

**Scope**: GPT paradigm evolution, scaling laws, self-supervised visual pre-training, and their robotics implications

1. GPT went from "pre-train + fine-tune" (GPT-1) to "in-context learning" (GPT-3). pi_0 currently uses "pre-train + fine-tune" for new robots. What would "in-context learning" look like for your dexterous hand -- and what data/scale barriers stand between current VLAs and that capability?

2. Chinchilla showed that given fixed compute, model size and data should scale proportionally. Your dexterous manipulation lab can collect ~100 hours of teleoperation demo per month. Using the Chinchilla principle, how would you decide model size for a dexterous hand VLA? What makes this calculation fundamentally different from LLM scaling?

3. MAE pre-trains by masking 75% of image patches and reconstructing them -- no labels needed. Your dexterous hand setup generates thousands of hours of unlabeled video. Design a concrete pre-training scheme for your hand's visual encoder using the MAE principle. What would you mask, what would you reconstruct, and what representation properties would you expect?

4. GPT-3's base model spawned Codex (code), InstructGPT (alignment), and WebGPT (tool use) through different fine-tuning datasets. Map this "one base, many fine-tunes" pattern to your dexterous hand work: what would be your base model, and what fine-tuned variants would you create for different manipulation tasks?

5. DINOv2 learns strong visual representations without any text supervision, while CLIP requires image-text pairs. For your dexterous hand cameras (wrist-mounted, no natural language annotations), which pre-training paradigm is more practical and why? Under what conditions would you switch to the other?

---

## Level 2 -> Level 3 Exam: Vision-Language + 生成模型原理

**Scope**: ViT, CLIP, DDPM, Flow Matching, DiT 的原理和对机器人的意义

1. ViT 和 CLIP 都用了 ViT 架构, 但训练目标不同。(a) 各自的训练目标是什么? (b) 同一个 ViT 架构, 为什么 SigLIP 预训练的比 ImageNet 预训练的对机器人更有用? (c) 你要为灵巧手选 vision backbone, 在什么场景下选从头训 ResNet-18, 什么场景下选 SigLIP ViT + LoRA?

2. 从流形的角度, AE/VAE 和 DDPM 分别在做什么? (a) 各自的几何本质是什么? (b) 为什么 VAE 生成模糊而 DDPM 生成清晰? (c) Robotics 中如何组合这两种范式?

3. DDPM 的去噪网络 (a) 输入什么、输出什么、loss 是什么? (b) 训练时为什么不需要跑 1000 步加噪? (c) 推理时为什么必须跑 1000 步? (d) 训练和推理的这种不对称性是怎么设计出来的?

4. Flow Matching 相比 DDPM 改了什么? (a) 为什么直线路径比弯曲路径需要的步数少? (b) DDPM 是 SDE 而 Flow Matching 是 ODE, 这个区别意味着什么? (c) 为什么 2024 年之后的 VLA 工作几乎都选 Flow Matching?

5. 生成模型的演化链 AE→VAE→GAN→DDPM→Flow Matching 中, 每一步解决了上一步的什么问题? 各自的训练目的和训练产物 (encoder/decoder/去噪网络) 分别是什么? 在 robotics 中哪些被继承了?

---

## Level 3 -> Level 4 Exam: Complete Robot FM Design Understanding

**Scope**: RT/PI/GR00T family architectures, cross-embodiment training, and system-level design decisions

1. You want to build a foundation model for your 20-DOF dexterous hand that also works on a 7-DOF arm. PI's approach: zero-pad all action spaces to max dimension + unified flow matching. GR00T's approach: separate VLA brain (10Hz) + WBC cerebellum (120Hz). Design your system -- which architecture do you choose for dexterous manipulation and why? What specific properties of contact-rich manipulation drive your choice?

2. pi\*0.6 uses offline RL (specifically, advantage-weighted regression) to fine-tune a pre-trained VLA, improving over pure imitation learning. You have years of PPO experience. Design a concrete pipeline: starting from a pre-trained pi_0 checkpoint, how would you use RL to improve its dexterous manipulation performance? What reward signal would you use, and what failure modes would you watch for?

3. GR00T's SONIC trains a universal humanoid tracker on 100M motion frames using PPO -- proving that RL + scale works at foundation level. Your dexterous hand has no equivalent of AMASS (large-scale motion capture). How would you create a "SONIC for dexterous hands" -- what data source would replace motion capture, what training objective would replace motion tracking, and what scale would you need?

4. RT-2 tokenizes actions into 256 discrete bins; pi_0 uses continuous flow matching; pi_0-FAST re-discretizes with a learned tokenizer. You need 50Hz control with sub-millimeter precision for your pen-spinning task. Analyze which action representation (discrete token / continuous flow / hybrid FAST) best fits your requirements, with quantitative reasoning about precision loss and latency.

5. DreamZero (GR00T N2) uses a world model to "imagine" outcomes before acting, achieving >2x generalization over VLA alone. Your sim2real pipeline already has a physics simulator (MuJoCo) that serves as a "world model." Compare MuJoCo-as-world-model vs DreamZero's learned world model for your dexterous tasks: what does each capture that the other cannot, and could they be complementary?

---

## Exam Rules

- 5 questions per level, each 20 points
- 80+ points: advance to next Level
- 60-80: re-read key sections marked in CS2Robotics_Roadmap.md, then retake
- Below 60: re-read all materials for that Level
- Answer from understanding, no reference lookup
- Imprecise but directionally correct intuitions are acceptable

---

# Reference Answers

## Level 0 Reference Answers

**1.** The MLP representation has joint angles + object pose (proprioception + a few geometric features), but is missing: (a) visual texture/shape details (distinguishing a smooth ball from a rough cube), (b) spatial context (table edge, obstacles), (c) contact-rich information (force distribution, slip). Adding a pre-trained visual encoder (e.g., SigLIP) would provide high-dimensional visual features that encode object affordances the MLP never sees. Generalization changes: the policy could potentially handle unseen objects with similar visual affordances (a new cup that looks like a trained cup) -- this is the distributional representation transfer that Bengio describes. The MLP policy can only generalize to interpolation within the trained state space; the visual encoder enables extrapolation along visual similarity dimensions.

**2.** The MLP treats each timestep independently (Markov assumption). A Transformer over (s,a,r) tuples would: (a) attend to any previous timestep (not just the last one), (b) learn which historical events are relevant (adaptive memory vs fixed window), (c) model the trajectory as a whole rather than step-by-step. This is exactly Decision Transformer's insight -- given a trajectory sequence and a desired return, generate actions by sequence prediction rather than Bellman backup. The key shift: from "estimate value of current state" (PPO) to "predict action given desired outcome" (DT).

**3.** Vision-language representations need semantic alignment (understanding "the red cup") while action representations need smooth, continuous, high-frequency output (precise joint trajectories). Training both with one set of weights causes "representation interference": action training gradients destroy the pre-trained vision-language knowledge (catastrophic forgetting), and VLM gradients push action weights toward semantic features rather than motor-control features. pi_0's solution (separate weight groups + attention cross-talk) lets each group optimize for its own objective while sharing information through attention.

**4.** Domain randomization creates invariant representations -- the policy learns to ignore randomized features (texture, lighting, friction coefficients) because they are uninformative for the task across randomized distributions. This is invariance, not equivariance (equivariance would mean the representation changes systematically with the domain parameter). This matters because: too much invariance can discard useful information (e.g., if friction actually matters for your task, randomizing it teaches the policy to ignore friction cues, which hurts real-world adaptation). The representation learning perspective suggests: randomize only what is truly irrelevant; for task-relevant physical parameters, use system identification or adaptive representations instead.

**5.** Invest more in training objectives (loss functions + data). Transformer is already a near-optimal information routing architecture -- the same backbone works for language (GPT), vision (ViT), and robot control (pi_0). The differentiator is not the architecture but: (a) what data you train on (Internet text vs robot demos), (b) what loss you optimize (next-token vs flow matching vs PPO reward), (c) how you combine modalities. For dexterous manipulation specifically: the open question is whether flow matching (pi_0), offline RL (pi\*0.6), or hybrid (RL as post-training) is the best training objective -- not whether to use Transformer or CNN.

## Level 1 Reference Answers

**1.** "In-context learning" for dexterous hand would mean: show the robot 2-3 video demos of a new task (e.g., opening a specific jar) in the prompt, and the model performs it without any weight updates. Barriers: (a) Data scale -- GPT-3's ICL emerged at 175B parameters trained on 300B+ tokens; current VLAs are 3-10B with ~10K hours of data, orders of magnitude less; (b) Action grounding -- language tokens have shared semantics across contexts, but robot actions are embodiment-specific, making cross-context transfer harder; (c) Evaluation -- we don't yet know what "emergent capability" looks like for manipulation (is it zero-shot object generalization? tool use? task composition?).

**2.** With 100h/month teleoperation data, accumulating ~1000h in a year. Chinchilla says model size should match data size. pi_0 used ~10K hours for a 3B model. By proportion, ~1000h might support ~300M parameters -- roughly the size of pi_0's action expert alone. Critical difference from LLM: (a) robot data has much higher bits-per-second than text (50Hz x 20-DOF vs ~5 tokens/sec), so "hours" is misleading -- 1000h of dexterous data may contain more training signal per hour than text; (b) but robot data has far less diversity (same lab, same objects, same camera), so the effective information is lower. Practical solution: use a pre-trained VLM backbone (freezing vision-language layers) and only train the action expert with your data -- this is exactly what openpi enables.

**3.** Concrete scheme: (a) Record continuous video from wrist camera during manipulation; (b) Sample frames, convert to ViT patch tokens; (c) Randomly mask 75% of patches; (d) Train encoder+decoder to reconstruct masked patches. For dexterous hand specifically, also mask temporal patches across consecutive frames (VideoMAE-style) to learn motion dynamics. Expected properties: the encoder learns spatial layout of objects and hand, texture features for grasp prediction, and temporal dynamics of contact events -- all without manual annotation. Limitation: MAE learns to reconstruct pixels, not semantics; for language-conditioned tasks, you'd need CLIP-style alignment afterwards.

**4.** Base model: VLA pre-trained on large-scale cross-embodiment manipulation data (like pi_0's pre-training on DROID + Open X-Embodiment). Fine-tuned variants: (a) "DexGrasp" -- fine-tuned on your lab's grasping demos for diverse objects; (b) "DexTool" -- fine-tuned on tool-use demonstrations (pen, screwdriver); (c) "DexAssembly" -- fine-tuned on assembly tasks with tight tolerances. Just as Codex (12B) outperformed GPT-3 (175B) on code through domain fine-tuning, your specialized variants would outperform the base model on specific dexterous tasks while being much smaller. The key insight: don't train from scratch; the base model provides world knowledge and motor primitives that transfer across tasks.

**5.** DINOv2 is more practical for your setup because: (a) your wrist camera footage has no natural language labels; (b) collecting image-text pairs for manipulation is expensive and unnatural; (c) DINOv2's self-supervised features (patch-level, position-aware) are well-suited for spatial reasoning needed in grasping. Switch to CLIP-style pre-training when: you add language-conditioned tasks ("pick up the red pen") and need the visual encoder to understand language-grounded concepts. In practice, VLAs like pi_0 use SigLIP (CLIP variant) because language conditioning is a key capability for general-purpose robots. If your work is single-task (no language), DINOv2; if multi-task with language, CLIP/SigLIP.

## Level 2 Reference Answers

**1.** Gaussian failure: pen handover -- the same observation (pen held vertically) has two equally valid grasps (grab from top, grab from bottom). Gaussian PPO averages these two modes, outputting a grasp at the middle of the pen with half-confidence -- the hand collides with the pen's center. Diffusion Policy samples from the full bimodal distribution and commits to one mode. PPO strictly better: high-frequency reactive control in contact -- e.g., maintaining a stable pinch grasp on a slippery object where you need 200Hz feedback and single-step corrections. Diffusion Policy's action chunking (predict 50 steps) cannot react to sub-chunk perturbations, while PPO's single-step Gaussian is cheap and fast enough to close the loop at every timestep.

**2.** At 50Hz, each action must be generated in 20ms. pi_0's flow matching: 10 integration steps, each a forward pass through the action expert (~300M params). On an RTX 4090, one forward pass ~2ms, so 10 steps = ~20ms -- barely within budget. DDPM with 100 steps = ~200ms, 10x too slow. Tradeoff: you could use DDPM with distillation (consistency model, ~1-4 steps) at some quality loss, or use flow matching at 10 steps. Engineering tradeoffs: (a) batch size 1 inference is GPU-inefficient; (b) action chunking helps -- predict 50 steps (1 second) at once, execute 16 steps while computing the next chunk (pipeline parallelism); (c) for real hardware, network latency (if inference is remote) may dominate compute latency.

**3.** CLIP-based VLM features are optimized for semantic alignment (matching "red pen" to the image of a red pen), which is excellent for task-level understanding. But this alignment discards fine-grained spatial information (exact position, orientation, distance) because CLIP's contrastive loss doesn't require spatial precision -- "a pen on a table" matches whether the pen is left or right. RT-2 solution: use VLM as-is and discretize actions (loses precision). pi_0 solution: add a separate action expert with flow matching that receives spatial information through attention to VLM features. GR00T N1 solution: split into System 2 (VLM, 10Hz, coarse understanding) and System 1 (DiT, 120Hz, precise motor control with proprioception). For dexterous manipulation, spatial precision is critical -- pure CLIP features are not enough.

**4.** Replacement pipeline: (a) Collect 50-100 human hand videos of the target task (e.g., pen spinning, filmed from similar angle as robot camera); (b) Train VIP on these videos to get a video-based value function V(current_frame, goal_frame); (c) Use V as reward: r_t = V(o_t, o_goal) - V(o_{t-1}, o_goal); (d) Train your PPO policy with this VIP reward instead of hand-crafted terms. What you lose: (a) explicit contact-phase shaping (your current reward might have bonuses for "contact established", "object lifted", etc. -- VIP only sees visual progress); (b) safety constraints (joint limits, force limits) that you currently encode in reward penalties; (c) precision in occluded scenarios (if the hand occludes the object, VIP's visual signal degrades). Practical compromise: use VIP for the main task reward + keep a few safety/constraint penalties from your current design.

**5.** Yes, you could. Feed your PPO rollout data as (return-to-go, state, action) sequences and train a Transformer to predict actions conditioned on high return. Gains: (a) no reward engineering at deployment time (just set high return); (b) can leverage suboptimal rollouts (by conditioning on their actual return); (c) simpler training loop (supervised, no value estimation). Losses: (a) no online exploration (DT is pure offline, cannot improve beyond the best behavior in your dataset); (b) return conditioning is coarse (doesn't handle multi-objective tradeoffs your shaped reward captures); (c) in contact-rich manipulation, small action errors compound rapidly -- DT's open-loop generation struggles compared to PPO's closed-loop correction. Makes sense when: you have a large dataset of diverse-quality rollouts and want a simple baseline, or when reward design is the bottleneck and you'd rather just label trajectories by quality.

## Level 3 Reference Answers

**1.** For dexterous manipulation, choose GR00T-style hierarchy (VLA brain + WBC cerebellum), because: (a) Contact-rich manipulation requires 50-120Hz control with sub-degree joint precision -- a single VLA running at 10Hz cannot achieve this; (b) The WBC layer (PPO-trained tracker, your expertise) handles reactive contact control, while VLA handles task semantics; (c) Your existing PPO policy can serve as the initial WBC layer -- you're not starting from scratch. PI's unified approach works better for simpler manipulation (pick-and-place, 5-10Hz sufficient) but struggles with the precision and speed requirements of dexterous tasks. Specific design: System 2 (VLA, 5-10Hz) outputs target fingertip poses or hand configuration references; System 1 (your PPO tracker, 50Hz) tracks these references with reactive contact control.

**2.** Pipeline: (a) Start with openpi pre-trained pi_0 checkpoint; (b) Collect 10-50 hours of teleoperation demos for your dexterous tasks; (c) Fine-tune with supervised flow matching loss (standard openpi fine-tuning); (d) Then apply advantage-weighted regression (AWR): roll out the fine-tuned policy in sim, score trajectories by task success + smoothness, weight the training loss by advantage -- upweight good trajectories, downweight bad ones. Reward signal: use sim task success (object reaches goal pose) + contact quality (stable grasp, no excessive force). Failure modes: (a) The RL fine-tuning might "forget" VLM capabilities (mitigate with KL penalty to pre-trained model); (b) Reward hacking in sim that doesn't transfer to real (mitigate with conservative reward design); (c) Distribution shift -- AWR assumes the rollout distribution covers good behaviors, but early rollouts of a VLA on a new task may all fail (mitigate with curriculum).

**3.** Data source: procedural generation of object manipulation trajectories in simulation -- analogous to what UltraDexGrasp does with BODex (20M grasps across 3000+ objects). Unlike AMASS for humanoid motion (human motion capture), dexterous manipulation has no natural "motion capture" equivalent because hand-object interaction is too complex to capture. Training objective: track procedurally generated reference trajectories (grasp sequences, in-hand rotations, tool uses) -- the dexterous equivalent of SONIC's motion tracking. Scale needed: SONIC used 100M frames with 42M parameters; for a 20-DOF hand with similar complexity, estimate 50-100M frames across 1000+ objects. Key difference from SONIC: hand manipulation requires modeling contact physics accurately, so the sim quality matters more than for humanoid locomotion.

**4.** Pen spinning at 50Hz with sub-millimeter precision. Precision analysis: 256-bin discrete tokenization over a +/-1 radian range = 7.8 mrad per bin = ~0.45 degrees. For a finger link of 40mm, this is ~0.3mm positional error at fingertip -- borderline acceptable. Flow matching (continuous): arbitrary precision, limited only by float32 (~1e-7 rad). FAST tokenizer: learned codebook, precision depends on codebook size and training, typically comparable to 1024 bins. Latency analysis: discrete autoregressive (RT-2 style) at 50Hz needs to generate 20-DOF action tokens sequentially -- 20 forward passes per step = ~40ms >> 20ms budget, too slow. Flow matching: 10 steps x 2ms = 20ms, tight but feasible. FAST: single forward pass ~5ms, fastest. Recommendation: flow matching for precision-critical tasks (sub-mm needed); FAST if latency is the hard constraint and you can accept slightly lower precision. The 256-bin discrete approach is unsuitable for high-DOF dexterous control.

**5.** MuJoCo captures: exact rigid-body dynamics, contact geometry, known physical parameters (if calibrated). Cannot capture: visual appearance changes, deformable objects, unmodeled real-world noise (sensor drift, cable tension). Learned world model (DreamZero) captures: visual dynamics (appearance, occlusion), soft/deformable contacts from data, and implicit modeling of hard-to-specify effects. Cannot capture: precise physical constraints (conservation laws, contact complementarity) as accurately as MuJoCo. Complementary use: MuJoCo for PPO pre-training of the low-level WBC (where precise physics matters); DreamZero for high-level planning (where visual prediction and novel scenario generalization matter). The sim2real gap you fight with domain randomization is MuJoCo's weakness and DreamZero's strength -- DreamZero trained on real video has no sim2real gap by construction. Conversely, DreamZero cannot replace MuJoCo for high-frequency contact simulation needed in PPO training.
