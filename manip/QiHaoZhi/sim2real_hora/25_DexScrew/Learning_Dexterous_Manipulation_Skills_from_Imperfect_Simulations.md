[# Learning Dexterous Manipulation Skills from Imperfect Simulations Elvis Hsieh∗, Wen-Han Hsieh∗, Yen-Jen Wang∗, Toru Lin, Jitendra Malik, Koushil Sreenath†, Haozhi Qi† UC Berkeley ∗ Equal contribution (listed in alphabetical order).† Equal advising. ###### Abstract Reinforcement learning and sim-to-real transfer have made significant progress in dexterous manipulation. However, progress remains limited by the difficulty of simulating complex contact dynamics and multisensory signals, especially tactile feedback. In this work, we propose DexScrew, a sim-to-real framework that addresses these limitations and demonstrates its effectiveness on nut-bolt fastening and screwdriving with multi-fingered hands. The framework has three stages. First, we train reinforcement learning policies in simulation using simplified object models that lead to the emergence of correct finger gaits. We then use the learned policy as a skill primitive within a teleoperation system to collect real-world demonstrations that contain tactile and proprioceptive information. Finally, we train a behavior cloning policy that incorporates tactile sensing and show that it generalizes to nuts and screwdrivers with diverse geometries. Experiments across both tasks show high task progress ratios compared to direct sim-to-real transfer and robust performance even on unseen object shapes and under external perturbations. ## I Introduction Reinforcement learning (RL) paired with sim-to-real transfer has recently delivered a number of promising results in dexterous manipulation [32](#bib.bib26), [31](#bib.bib27), [9](#bib.bib10), [6](#bib.bib11), [36](#bib.bib5)]. Policies trained in massively parallel simulation [[[24](#bib.bib64)]] with domain randomization [[[34](#bib.bib104)]] have demonstrated strong robustness and generalization capabilities in the real world.

However, in practice, sim-to-real transfer faces two major limitations. First, due to the complexity of physics simulation, only a limited range of tasks can be accurately modeled. Prior work either relies on specialized techniques for high-fidelity simulation [[[27](#bib.bib117), [10](#bib.bib118)]] or seeks generalization through domain randomization [[[18](#bib.bib119), [16](#bib.bib103), [32](#bib.bib26), [22](#bib.bib129)]]. However, as the task becomes more dynamic, the sim-to-real gap grows [[[45](#bib.bib77)]], and simulation alone becomes insufficient. Second, existing sensing modalities have an intrinsic sim-to-real gap. While vision can be partially mitigated through domain randomization [[[42](#bib.bib123)]], tactile sensing remains difficult to approximate reliably. Although some work aims to improve tactile simulation [[[46](#bib.bib120), [1](#bib.bib121)]] or trains policies using alternative proxy representations [[[37](#bib.bib4), [48](#bib.bib83), [49](#bib.bib122)]], these approaches cannot leverage the full power of tactile sensing. These limitations remain widely viewed as major constraints on the complexity of tasks that can be achieved.

On the other hand, teleoperation and imitation learning [[[7](#bib.bib108), [52](#bib.bib107)]] remove the need for simulation entirely. In this setting, policies can learn directly from real-world interactions and sensorimotor signals, which avoids the challenges introduced by sim-to-real transfer. However, teleoperating dexterous hands is challenging because of the intrinsic morphology differences between human and robot hands [[[3](#bib.bib124), [2](#bib.bib84), [38](#bib.bib109)]]. As a result, it is difficult to collect datasets that are large and diverse enough to achieve the desired behavior and generalization.

Motivated by these observations, we introduce DexScrew, a framework that combines the strengths of both approaches to expand the capability of sim-to-real reinforcement learning under imperfect simulation. The key idea is that the motion primitives underlying contact-rich dexterous manipulation do not need to be learned from a perfect physics model. A simplified simulator is sufficient to induce the core rotational behaviors required for these tasks. Once this motion is learned, the resulting policy can be used as a skill primitive to collect real-world demonstrations, from which a new policy can be learned. In this way, sensing modalities and physical interactions that are difficult to simulate can be obtained directly from real-world data, while the fine-grained motions that are hard to teleoperate are provided by the simulation-trained policy. We demonstrate this idea through two tasks, nut-bolt fastening and screwdriving with a multifingered hand. Both tasks are traditionally viewed as requiring complex contact dynamics understanding and tactile sensing. We instead show that effective policies can be learned without relying on high-fidelity simulation.

More specifically, our framework consists of three stages. First, we train reinforcement learning (RL) policies in simulation using a simplified physics model. Instead of modeling the thread structure of the nut and screw, we approximate their interaction with a revolute joint that connects two simple geometric shapes, which allows the policy to efficiently learn rotational behavior. Second, we use this learned skill as a primitive within a teleoperation system to collect real-world demonstrations. The operator controls the arm motion and triggers the finger rotation skill rather than issuing low-level joint commands, which enables efficient collection of tactile data during teleoperated execution. Finally, using the resulting multisensory dataset, we train a behavior cloning policy that coordinates arm and finger motions while leveraging tactile feedback.

We evaluate our framework on two tasks: nut-bolt fastening and screwdriving. Policies trained with simplified dynamics can generate reasonable rotational behavior but cannot complete the tasks. By learning from real-world multisensory demonstrations, our method overcomes these limitations and achieves stable and reliable performance under challenging contact conditions. These results show that complex contact-rich manipulation skills can be bootstrapped from simplified simulators and that real-world tactile feedback is essential. Our framework provides a scalable path toward dexterous manipulation and supports broader deployment of general-purpose robot hands in unstructured environments.

## II Related Work

Dexterous manipulation has been a long-standing challenge in robotics [[[4](#bib.bib110), [30](#bib.bib111)]]. Early work focused on classical model-based control and analytic grasp planning [[[40](#bib.bib14), [25](#bib.bib16), [8](#bib.bib17), [26](#bib.bib19), [33](#bib.bib30)]]. Recent years have seen rapid progress in learning-based approaches, which can be grouped into two primary directions: sim-to-real learning paired with reinforcement learning [[[32](#bib.bib26), [36](#bib.bib5), [6](#bib.bib11)]] and imitation learning from teleoperation [[[20](#bib.bib78), [2](#bib.bib84), [38](#bib.bib109)]] or human data [[[44](#bib.bib133), [47](#bib.bib134)]].

Both directions, however, face notable limitations. Sim-to-real methods benefit from large-scale simulated data and can generalize across diverse objects, yet they remain limited by inaccuracies in modeling complex contact dynamics and sensing, a challenge that becomes more significant as task complexity increases [[[45](#bib.bib77), [12](#bib.bib80)]]. Imitation learning benefits from multisensory real-world data, yet collecting high-quality dexterous demonstrations is considerably more difficult than collecting data for simpler end-effectors. Our work seeks to combine the strengths of both approaches. We use large-scale simulation to learn motion primitives while leveraging real-world data to close the dynamics and sensing gaps. Moreover, our skill-based framework enables efficient collection of dexterous real-world data by using the simulation-trained policy itself as a reusable skill primitive.

In the context of nut fastening and screwdriving, there has been recent work combining sim-to-real transfer with teleoperation. For example, Liu et al. [[[21](#bib.bib126)]] build a residual model from real-world interactions to compensate for the sim-to-real gap and achieve robust in-hand manipulation. Yin et al. [[[50](#bib.bib127)]] use simulation-trained policies as stability controllers to enable complex manipulation skills. Both approaches can be integrated with teleoperated arm control to complete these tasks. However, they do not produce autonomous policies that incorporate tactile sensing. Another line of work is Kumar et al. [[[15](#bib.bib112)]], who demonstrate screwdriver turning by combining learning with trajectory optimization. Noseworthy et al. [[[29](#bib.bib130)]] present an autonomous sim-to-real policy but only show results with a parallel-jaw gripper and do not demonstrate regrasping.

Another way to address the sim-to-real gap is to refine the policies in the real world. For example, Transic [[[13](#bib.bib128)]] shows that sim-to-real policies can adapt to complex real-world dynamics with only a few human interventions as demonstrations, although the demonstrations are primarily performed with simple end-effectors. In contrast, we apply this idea to dexterous hands. Maddukuri et al. [[[23](#bib.bib131)]] show that co-training with both simulation and real-world data can reduce the gap and improve manipulation performance. RLPD [[[51](#bib.bib136)]] and Sparsh-X [[[11](#bib.bib135)]] share a similar philosophy of augmenting modalities through human-guided data collection; however, they are not motivated by the possibility of learning skills from imperfect simulations, as their policies are designed to be directly transferable.

*Figure 1: An overview of our approach. We first train a reinforcement learning policy in simulation using a simplified object model, which serves as a motion prior for nut-bolt fastening and screwdriving. We then collect real-world trajectories by using the learned policy as a skill primitive during teleoperation. Finally, we train a behavior cloning policy on the collected data to obtain coordinated behavior between the arm and the fingers.*

## III Dexterity from Imperfect Simulation

An overview of our method is shown in Figure [1](#S2.F1). It consists of three stages. First, we train a reinforcement learning (RL) policy in simulation using a simplified object model (Section [III-A](#S3.SS1)). The resulting policy learns the desired finger motions but does not experience real-world dynamics and lacks tactile feedback. To address this, we collect real-world trajectories using the learned policy as a skill primitive for teleoperation (Section [III-B](#S3.SS2)). Finally, using this dataset, we train a new multisensory policy using behavior cloning (Section [III-C](#S3.SS3)).

### III-A Training a Reinforcement Learning Policy in Simulation

Simplified Object Modeling. Our goal is to design a simulation environment that enables fast training and encourages the emergence of desired finger gaits for rotation. To achieve this, we construct a simplified simulated object (Figure [2](#S3.F2)) that captures the essence of rotational motion. The object consists of a fixed cylindrical base with a nut or handle attached via a revolute joint. This setup allows the policy to learn rotational motion efficiently without relying on expensive contact-rich simulations. A similar idea was explored in [[[19](#bib.bib75)]] to model bottle caps using a heuristic friction design. In contrast, we further simplify the model, since we can leverage real-world data to compensate for the resulting dynamics mismatch.

Specifically, for the nut-bolt task, we use a thick triangular shape as the training object (Figure [2](#S3.F2)A). The extra thickness is used to prevent the policy from learning suboptimal strategies that apply a large force from the bottom. The learned policy also discovers a high-clearance gait that transfers well to diverse real-world nuts such as hexagonal and cube-shaped nuts. For the screwdriver task, where the primary difficulty arises from slippage around the handle, we use spherical primitives to keep the learned behavior conservative (Figure [2](#S3.F2)B). This observation, that different training shapes lead to different rotational gaits, is also consistent with the findings in [[[36](#bib.bib5)]]. Note that these objects do not need to be visually aligned with real world objects, as they are only used to learn the coarse motions used for real world data collection, as we discussed in Section [III-B](#S3.SS2).

![Figure](x2.png)

*Figure 2: Simplified Object Models. Each nut or handle is modeled as a rigid body attached to a fixed base through a revolute joint. This abstraction ignores thread-level mechanics while retaining the essential rotational dynamics needed for learning.*

Training Pipeline. Following [[[37](#bib.bib4), [45](#bib.bib77)]], we first train an oracle policy and then distill it into a sensorimotor policy. The oracle policy $\bm{f}$ is trained with access to an embedding of privileged information [[[5](#bib.bib105)]] $\bm{z}_{t}$. The sensorimotor policy operates without privileged sensing and instead conditions on a predicted embedding $\hat{\bm{z}}_{t}=\bm{\phi}(\bm{h}_{t})$ inferred from proprioceptive history $\bm{h}_{t}$ by a prediction module $\bm{\phi}$.

Privileged Information. The oracle policy has access to ground-truth environment and object properties, including object attributes (e.g., position, scale, mass, center of mass, friction coefficients), hand pose and finger configurations, and low-level controller parameters. The full set of privileged inputs is documented in the appendix.

Actions. At each step, the policy outputs a relative target position. The position command is computed as $\bm{a}_{t}^{\text{Hand}}=\eta\bm{f}(\bm{o}^{\text{RL}})+\bm{a}_{t-1}^{\text{Hand}}$, where $\eta$ is the action scale. This command is sent to the robot and converted into torque via a low-level PD controller. Here, $\bm{o}^{\text{RL}}$ contains the robot’s proprioceptive state, including joint positions and previous target positions from a sliding window of recent 3 timesteps.

Reward. The goal of the policy in simulation is to rotate the simplified object around the revolute joint. The reward consists of a task reward, energy penalties, and stability penalties (time index $t$ omitted for simplicity). Each component includes several terms defined in the appendix:

| | $$r_{t}=\lambda_{\text{task}}r^{\text{task}}+\lambda_{\text{energy}}r^{\text{energy}}_{t}+\lambda_{\text{stability}}r^{\text{stability}}_{t}.$$ | |
|---|---|---|

Oracle Policy Training. We train the oracle policy using proximal policy optimization (PPO) [[[41](#bib.bib39)]] with the reward described above. The robot state and privileged information are each encoded with separate MLPs. These embeddings are concatenated and passed through an MLP to produce the final action and value predictions. We train the policy for 1.5$\times$10${}^{\text{9}}$ environment steps.

Sensorimotor Policy Training. The sensorimotor policy receives proprioceptive states and a latent code $\hat{\bm{z}}_{t}=\bm{\phi}(\bm{h}_{t})$ inferred from a 30-timestep history. We train the policy using DAgger [[[39](#bib.bib113)]]: at each step, the sensorimotor policy acts in the environment, while the oracle policy provides target actions and ground-truth privileged embeddings. The training objective is

| | $$\mathcal{L}=\|a^{\text{Hand}}_{t}-\hat{a}^{\text{Hand}}_{t}\|_{2}^{2}+\|z_{t}-\hat{z}_{t}\|_{2}^{2},$$ | |
|---|---|---|---|---|---|---|

where $\hat{a}^{\text{Hand}}_{t}$ denotes the actions produced by the sensorimotor policy. The embedding predictor $\phi$ is optimized using Adam [[[14](#bib.bib42)]] until convergence.

Randomization. We apply domain randomization during training to improve the robustness of the RL policy [[[34](#bib.bib104)]]. Following [[[45](#bib.bib77)]], we randomize the nut/handle mass, center of mass, friction coefficient, size, and PD gains, and we also add observation noise. Detailed parameters are provided in the appendix.

Termination Conditions. To prevent the policy from getting stuck in unrecoverable states, we terminate an episode when any of the following conditions are met: (1) the distance between the thumb or index finger and the nut/handle exceeds a reset threshold ($7\text{\,}\mathrm{cm}$ for nuts, $10\text{\,}\mathrm{cm}$ for handles); (2) the nut or handle remains stagnant over a sliding time window ($3.5\text{\,}\mathrm{s}$ for nuts, $3\text{\,}\mathrm{s}$ for handles); or (3) near-zero contact forces are detected for the same duration. These conditions accelerate training by penalizing failure modes such as drifting away from the object or failing to maintain contact.

![Figure](x3.png)

*Figure 3: Teleoperation Interface. The human operator controls the wrist position using the VR controller buttons and adjusts yaw and pitch through the joystick. This setup allows the operator to guide the arm motion while relying on the learned finger-rotation skill during data collection.*

### III-B Real World Data Collection with Learned Policy

The policy trained in simulation with the simplified object model learns the desired rotational behavior; however, it inevitably misses key physical dynamics such as thread interactions. These effects are difficult to model but are crucial for reliable real-world fastening. It also lacks tactile information, which is hard to simulate but crucial for fine-grained wrist adjustments.

To bridge this gap, we introduce a skill-based assisted teleoperation* for real-world data collection. The core idea is to reuse the simulation-trained policy as a skill primitive for finger motion control. Instead of commanding every joint individually, the human operator controls only the wrist movement and decides when to activate the skill primitive (Figure [3](#S3.F3)). Wrist position is specified using the Quest VR111https://www.meta.com/quest/products/quest-2 controller’s joystick. This approach is inspired by [[[20](#bib.bib78)]], but we use a much finer-grained skill for data collection.

This framework offers several advantages. First, it delegates complex finger motions to a robust simulation-trained policy that generalizes across different objects, eliminating the need for humans to learn finger coordination under morphological differences. Second, using a joystick for arm control enables precise and intuitive wrist positioning. While arm motion is relatively straightforward to simulate, we intentionally exclude it from the initial simulation training stage, as learning the specific downward progression required for nut-bolt fastening would necessitate simulating thread-level mechanics. Finally, collecting data in the physical environment provides the multisensory observations, particularly tactile signals, that are essential for reliable task completion but remain difficult to approximate reliably in simulation.

Concretely, at each timestep we record two actions: (1) the action generated by the RL policy $\pi_{\rm RL}$, which defaults to the current hand joint positions when the policy is not activated, and (2) the arm action generated by human teleoperation. Formally, we define $\bm{a}_{t}=[\bm{a}_{t}^{\text{Hand}},\bm{a}_{t}^{\text{Arm}}]$, where $\bm{a}_{t}^{\text{Hand}}\in\mathbb{R}^{12}$ denotes the hand target joint positions, and $\bm{a}_{t}^{\text{Arm}}\in\mathbb{R}^{6}$ denotes the arm joint positions. We also record the multisensory observation $\big(\bm{q}_{t},\bm{c}_{t}\big)$, where $\bm{q}_{t}=[\bm{q}_{t}^{\text{Hand}},\bm{q}_{t}^{\text{Arm}}]$ contains all joint positions, and $\bm{c}_{t}$ represents the raw tactile signals from all five fingers.

Tactile Signal. In this work, we use the XHand’s built-in tactile sensors to capture contact information. Each fingertip is equipped with a pressure-based tactile array comprising 120 sensing elements, each measuring three-axis forces with a minimum detectable force of $0.05\text{\,}\mathrm{N}$. At each timestep, we record the tactile signal as $\mathbf{c}_{t}\in\mathbb{R}^{5\times 120\times 3}$, which includes the signals from all five fingers across three axes.

### III-C Behavior Cloning with Multisensory Data

With the dataset $\mathcal{D}_{\text{Real}}$ collected using the skill-based assisted teleoperation, we can train a behavior cloning (BC) policy $\pi_{\text{BC}}$ using the paired multisensory observations and expert actions. Vision is not used in our work.

Neural Network Architecture. We use a feedforward network as the policy. The past $K$ timesteps of observations $(\bm{q}_{t-K+1:t},\bm{c}_{t-K+1:t})$ are concatenated into a single feature vector. Tactile signals are first flattened and passed through an MLP. The fused feature vector is then processed by an hourglass encoder [[[28](#bib.bib106)]], which outputs the action predictions.
We also apply an action chunking strategy [[[52](#bib.bib107), [7](#bib.bib108)]], where the policy predicts a sequence of future actions $\hat{\bm{a}}_{t:t+H}$ rather than a single-step action. We use $K=5$ and $H=16$ unless otherwise noted.

*TABLE I: Real-world fastening performance on square, triangular, hexagonal, and cross-shaped nuts. We report progress ratio and rotation time (mean ± standard deviation over 10 trials) for different observation modalities. Tactile sensing and temporal history both improve performance, and their combination yields the highest accuracy and fastest execution. * indicates that only one completely successful trial was recorded, so no standard deviation is reported.*

Training. The policy is trained with supervised learning using the loss

| | $$\mathcal{L}_{\text{BC}}=\sum_{t=1}^{T}\sum_{h=0}^{H}\left\|\,\hat{\bm{a}}_{t+h}-\bm{a}_{t+h}\,\right\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $\hat{\bm{a}}_{t:t+H}$ is the action chunk predicted by $\pi_{\text{BC}}$, and $\bm{a}_{t:t+H}$ is the corresponding expert sequence. This objective encourages consistent predictions over the full horizon.
We train the policy using Adam [[[14](#bib.bib42)]] for 200 epochs and normalize observations following [[[43](#bib.bib115)]].

## IV Experiments

In this section, we first introduce the experiment setup (Section [IV-A](#S4.SS1)). We then evaluate the performance of our policies on two challenging tasks, nut-bolt fastening (Section [IV-B](#S4.SS2)) and screwdriving (Section [IV-C](#S4.SS3)). We conclude with qualitative experiments that provide additional analysis and design ablations in simulation.

### IV-A Experiment Setup

Hardware Setup. Our system consists of a UR5e robot arm (6 DoF) and a 12-DoF XHand. The XHand has five fingers: the thumb and index finger each have 3 DoF, and the remaining fingers have 2 DoF. Only the thumb and index provide abduction/adduction.

Simulation. We train our policies in IsaacGym [[[24](#bib.bib64)]] using 8,192 parallel environments. Each environment contains a simulated XHand and the simplified object model described in Section [III-A](#S3.SS1). The simulation runs at $200\text{\,}\mathrm{Hz}$, with control applied at $20\text{\,}\mathrm{Hz}$. Each episode lasts up to 800 simulation steps ($40\text{\,}\mathrm{s}$).

Object Set. For the nut-bolt task, we train on triangular nuts. For the screwdriver task, we approximate handles by octagon- and dodecagon-shaped nuts. This multi-geometry training in simulation helps the policy generalize to diverse real-world shapes.

Metrics. In simulation, we report the episode reward and episode length during training. In real-world evaluation, we measure the *progress ratio*, defined as the number of successful rotations divided by the total number of rotations required for full fastening. We also report the *time* for trials that fully complete the fastening process (progress ratio = 100), defined as the time needed to fully fasten the nut or fully tighten the screw. Note that some baseline methods do not achieve a single successful fastening or screwing attempt across ten trials. In such cases, we cannot report the standard deviation or the completion-time metric.

### IV-B Nut-Bolt Experiments

We first evaluate the system on the nut-bolt task. This task requires the fingers to establish correct contact patterns, sense progress through tactile feedback, and adjust the arm position accordingly. We choose this task because nut-bolt interactions are difficult to simulate efficiently, and completing the task relies heavily on tactile sensing, making it a strong testbed for our method.

We note that direct sim-to-real transfer can rotate the nut, but it cannot drive the nut downward because the arm does not move. Since thread interactions are not simulated, the nut remains at the same height even after completing full revolutions in simulation.

Setting. In simulation, we use the thick triangular nut shown in Figure [2](#S3.F2) (left). Training with this shape produces high-clearance gaits that transfer well and can rotate both square and triangular nuts in the real world.

During skill-based assisted teleoperation, we collect 50 trajectories each for the square and triangular nuts. Each trajectory lasts about 80 seconds. We then train a behavior cloning policy using the combined dataset. We evaluate performance on four types of nuts, which include square and triangular nuts as well as two unseen shapes, namely hexagonal nuts and cross-shaped nuts. Our main results are shown in Table [I](#S3.T1).

Observation History. We first study the effect of providing a short temporal history in the observation. Adding history significantly improves progress ratio and reduces execution time across all modalities and nut geometries. Temporal cues help the policy track fine-grained rotational progress and distinguish local geometric features. The benefit is especially clear for non-tactile policies, where history stabilizes performance and narrows much of the gap to tactile-based policies on easier geometries. When combined with tactile sensing, history yields the strongest overall performance, achieving the highest accuracy and fastest completion times across all nut types, including unseen shapes.

Tactile Information. Across all nut geometries, adding tactile input generally improves progress ratio. The effect is most pronounced on challenging shapes such as triangular and cross-shaped nuts, where progress ratios rise from roughly 30 to 65 percent for triangular nuts and from about 60 to 80 percent for cross-shaped nuts. This underscores the importance of tactile feedback for maintaining stable contact and detecting effective rotation. We also observe that certain non-tactile settings achieve high performance, even on unseen shapes, particularly on well-constrained geometries such as hexagonal and cross-shaped nuts. In these cases, the geometry provides strong passive guidance, and the finger gait learned in simulation is often sufficient to maintain contact without relying on tactile cues.

Failure Modes. We observe two main failure modes. First, the policy without observation history struggles to infer object shape from single-step proprioception. As a result, it fails to generalize across nut geometries, since different nuts require different rotational gaits. The policy cannot adjust its gait because the limited sensing information prevents it from identifying the correct motion pattern. Second, non-tactile policies frequently drift into unstable contact states and lose alignment with the bolt. Once misalignment occurs, the nut cannot sustain continuous rotation. In contrast, tactile policies can recover by adjusting wrist orientation or applying corrective downward force to re-establish contact.

### IV-C Screwdriving Experiments

*TABLE II: Real-world screwdriving performance. We report progress ratio and rotation time (mean ± standard deviation over 10 trials) for direct sim-to-real, expert replay, and our behavior cloning (BC) ablations. Tactile sensing and temporal history each improve performance, and their combination achieves the highest progress ratio and fastest execution. Here * indicates that the policy never fully completed the task, so no rotation time is reported.*

We also demonstrate that our method extends to the more challenging screwdriving task. Compared to nut-bolt fastening, screwdriving is inherently less stable: the shaft is not kinematically constrained along the screw axis, and even small tilts or misalignments can lead to slipping or loss of contact. As a result, the task requires more fine-grained control to maintain continuous rotation. The complex interaction between the screwdriver and the screw is also difficult to simulate accurately, which is why we include this task in our study.

Setting. In simulation, we use a mix of octagonal and dodecagonal handles as the training objects, as shown in Figure [2](#S3.F2) (right). This curated set encourages the learned rotational gaits to remain conservative in clearance and maintain stability. In the real world, we collect 72 trajectories, each lasting between 120 and 180 seconds.

Sim-to-Real Policy. Unlike the nut-bolt experiments, the screwdriving task can still make progress even when the wrist does not move, since downward motion is not required. We first evaluate a direct sim-to-real policy that uses only proprioception. The results are shown in Table [II](#S4.T2). The direct sim-to-real policy achieves a 41.60% progress ratio, indicating that it can produce meaningful behavior, which is a prerequisite for collecting expert trajectories. However, because it inevitably makes mistakes, it never completes the task even once, and therefore we cannot report statistics for completion time. In contrast, during data collection, the human operator can adjust the wrist position to recover from such mistakes.

As a result, when we replay the expert data from the data we collected, it can achieve higher success rate to be 50.80%. However, because it cannot adapt changes during deployment. It also fails at completely finishing the task. Next, we show the results of our policies.

![Figure](x4.png)

*Figure 4: Top: The policy with tactile information maintains a consistent alternating pattern of thumb and index finger contact, which supports stable engagement as the nut is rotated downward. Bottom: The policy without tactile information does not maintain a clear contact pattern. This leads to unsuccessful engagement and prevents proper downward wrist motion. The resulting pattern reflects the index finger pressing against the bolt after losing stable contact.*

Main Results. Our behavioral cloning policies show clear improvements over the direct sim-to-real and expert-replay baselines. Adding tactile sensing or temporal history individually improves progress ratio. Combining both modalities gives the strongest performance.

The baseline behavior cloning model already achieves a 69.20% progress ratio, which substantially outperforms expert replay. This phenomenon, where a behavior cloning policy surpasses the policy that generated the data, is consistent with filtered behavior cloning [[[17](#bib.bib116)]]. In this approach, only successful trials are used for training. A similar effect has also been reported in [[[45](#bib.bib77)]].

Using history alone produces a comparable improvement of 67.63%. This indicates that temporal information helps the policy track rotational progress and recover from partial failures. When history is not used, tactile information provides only limited benefit. However, once history is included, the two modalities become complementary. The progress ratio increases to 95.00%, and the average rotation time decreases substantially. These results show that tactile feedback and temporal history work together to produce stable, consistent, and efficient screwdriving behaviors.

Failure Modes. We observe that open-loop baselines frequently fail due to gradual handle slipping and accumulated orientation drift. Without feedback, small misalignments grow over time. Among behavioral cloning methods, the policy trained without observation history struggles to stabilize the screwdriver since it lacks the temporal information needed to infer handle pose and orientation. Non-tactile policies fail to detect subtle torque imbalances and often lose stable contact under slight perturbations. With both tactile feedback and temporal history, BC with tactile and history compensates for these effects by adjusting wrist orientation and applying appropriate forces.

### IV-D Qualitative Experiments

![Figure](x5.png)

*Figure 5: Top row: The policy recovers back to the nut-bolt fastening motion when the fingers are dragged by an external force. Bottom row: The policy recovers back to the screwdriving motion when the screwdriver is rotated counterclockwise during the clockwise rotation by the policy.*

Out-of-distribution Robustness. We study the robustness of the learned policy under external perturbations that are not encountered during training. These disturbances include (1) dragging the fingers away from the object or rotating the screwdriver in the opposite direction of the intended direction. In Figure [5](#S4.F5), we show that despite these perturbations, the policy consistently recovers to stable fastening behavior. Specifically, it recovers from reverse rotation of the fastening object by re-establishing contact and restoring the correct rotational direction. When finger contact is disrupted or temporarily blocked, the policy repositions the fingers and wrist to regain stable engagement.

Tactile Visualization. Tactile signals provide structured spatiotemporal patterns that the policy uses to infer contact phases. As shown in Fig. [4](#S4.F4), stable activation signatures emerge when the nut or screwdriver handle is correctly engaged. The policy learns to preserve these patterns by adjusting wrist orientation and contact force, effectively using tactile feedback as a local reference for alignment. When these patterns deviate, corrective actions such as re-engagement or downward pressing are triggered.

### IV-E Simulation Ablations

In simulation, we verify the design choices used to train the screwdriving policies. We evaluate our two-stage training procedure against two alternatives: training without any privileged information, and using an asymmetric actor-critic [[[35](#bib.bib132)]] architecture where the critic has access to full tactile information while the actor does not. The actor in this setting is directly deployable in the real world. The results are shown in Figure [6](#S4.F6).

We compare episode reward and episode length across these training strategies. When both the actor and critic have access to privileged information, our method (Ours, Oracle) achieves the highest performance. Removing privileged information from the actor, as in the asymmetric actor-critic variant, leads to a noticeable drop in performance. Removing privileged information entirely results in a further decline. These results show that privileged information plays an important role during policy learning. We also observe that a sensorimotor policy can approach similar performance when proprioceptive history is provided as input. We also experimented with training the asymmetric and non-privileged models for longer horizons but have not observed further improvement. This suggests that the performance gap is not due to insufficient training but instead reflects the importance of privileged information during learning.

![Figure](x6.png)

*Figure 6: Simulation ablations of screwdriving policy training. We compare our privileged-information oracle policy, its sensorimotor policy, an asymmetric actor–critic variant, and a policy trained without privileged information. Providing privileged information during training leads to significantly higher reward and more stable episode lengths. Each curve shows the mean and standard deviation over 5 seeds.*

## V Conclusion and Future Work

We present a framework for learning dexterous manipulation skills for contact-rich tasks using imperfect simulation. The approach first learns transferable rotational skills through reinforcement learning with simplified object modeling. It then uses these skills for skill-based teleoperation to collect real-world trajectories, and finally incorporates tactile feedback and learns a sensorimotor policy through behavior cloning. Experiments on nut-bolt fastening and screwdriver usage show that simulation alone cannot capture the complex dynamics required for reliable task execution. However, when behavior cloning is combined with tactile sensing and temporal history, the resulting policies become robust and reliable across diverse and unseen object geometries. This staged pipeline provides a practical and scalable solution for contact-rich manipulation and highlights the value of tactile sensing and skill-based teleoperation as effective bridges between simulation and real-world deployment.

Limitations. Although our skill-based teleoperation reduces the burden on the human operator, it remains a constraint for scalable data acquisition. Fully autonomous data collection or learning skill-level guidance from human videos would further improve efficiency. In our current tasks, the nut is already installed and the screwdriver is already inserted. Extending the approach to fully long-horizon assembly will require vision sensing and possibly high-accuracy force-torque sensing. A broader evaluation across more contact-rich manipulation tasks is also needed to assess generality.

## Acknowledgment

This work is supported in part by the program “Design of Robustly Implementable Autonomous and Intelligent Machines (TIAMAT)”, Defense Advanced Research Projects Agency award number HR00112490425. We thank Mengda Xu for his valuable feedback.

## References

-
[1]
I. Akinola, J. Xu, J. Carius, D. Fox, and Y. Narang (2025)

Tacsl: a library for visuotactile sensor simulation and learning.

T-RO.

Cited by: [§I](#S1.p2.1).

-
[2]
S. P. Arunachalam, I. Güzey, S. Chintala, and L. Pinto (2023)

Holo-dex: teaching dexterity with immersive mixed reality.

In ICRA,

Cited by: [§I](#S1.p3.1),
[§II](#S2.p1.1).

-
[3]
S. P. Arunachalam, S. Silwal, B. Evans, and L. Pinto (2023)

Dexterous imitation made easy: a learning-based framework for efficient dexterous manipulation.

In ICRA,

Cited by: [§I](#S1.p3.1).

-
[4]
A. Bicchi (2002)

Hands for dexterous manipulation and robust grasping: a difficult road toward simplicity.

IEEE Transactions on robotics and automation.

Cited by: [§II](#S2.p1.1).

-
[5]
D. Chen, B. Zhou, V. Koltun, and P. Krähenbühl (2020)

Learning by cheating.

In CoRL,

Cited by: [§III-A](#S3.SS1.p3.5).

-
[6]
T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal (2023)

Visual dexterity: in-hand reorientation of novel and complex object shapes.

Science Robotics.

Cited by: [§I](#S1.p1.1),
[§II](#S2.p1.1).

-
[7]
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2024)

Diffusion policy: visuomotor policy learning via action diffusion.

IJRR.

Cited by: [§I](#S1.p3.1),
[§III-C](#S3.SS3.p2.5).

-
[8]
R. Fearing (1986)

Implementing a force strategy for object re-orientation.

In ICRA,

Cited by: [§II](#S2.p1.1).

-
[9]
A. Handa, A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, Y. Narang, J. Lafleche, D. Fox, and G. State (2023)

Dextreme: transfer of agile in-hand manipulation from simulation to reality.

In ICRA,

Cited by: [§I](#S1.p1.1).

-
[10]
E. Heiden, M. Macklin, Y. Narang, D. Fox, A. Garg, and F. Ramos (2021)

Disect: a differentiable simulation engine for autonomous robotic cutting.

In RSS,

Cited by: [§I](#S1.p2.1).

-
[11]
C. Higuera, A. Sharma, T. Fan, C. K. Bodduluri, B. Boots, M. Kaess, M. Lambeta, T. Wu, Z. Liu, F. R. Hogan, and M. Mukadam (2025)

Tactile beyond pixels: multisensory touch representations for robot manipulation.

In CoRL,

Cited by: [§II](#S2.p4.1).

-
[12]
B. Huang, Y. Chen, T. Wang, Y. Qin, Y. Yang, N. Atanasov, and X. Wang (2023)

Dynamic handover: throw and catch with bimanual hands.

In CoRL,

Cited by: [§II](#S2.p2.1).

-
[13]
Y. Jiang, C. Wang, R. Zhang, J. Wu, and L. Fei-Fei (2024)

Transic: sim-to-real policy transfer by learning from online correction.

In CoRL,

Cited by: [§II](#S2.p4.1).

-
[14]
D. P. Kingma and J. Ba (2015)

Adam: a method for stochastic optimization.

In ICLR,

Cited by: [§III-A](#S3.SS1.p8.3),
[§III-C](#S3.SS3.p3.3).

-
[15]
A. Kumar, T. Power, F. Yang, S. A. Marinovic, S. Iba, R. S. Zarrin, and D. Berenson (2025)

Diffusion-informed probabilistic contact search for multi-finger manipulation.

In ICRA,

Cited by: [§II](#S2.p3.1).

-
[16]
A. Kumar, Z. Fu, D. Pathak, and J. Malik (2021)

RMA: rapid motor adaptation for legged robots.

In RSS,

Cited by: [§I](#S1.p2.1).

-
[17]
A. Kumar, J. Hong, A. Singh, and S. Levine (2022)

When should we prefer offline reinforcement learning over behavioral cloning?.

In ICLR,

Cited by: [§IV-C](#S4.SS3.p6.1).

-
[18]
J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter (2020)

Learning quadrupedal locomotion over challenging terrain.

Science robotics.

Cited by: [§I](#S1.p2.1).

-
[19]
T. Lin, Z. Yin, H. Qi, P. Abbeel, and J. Malik (2024)

Twisting lids off with two hands.

In CoRL,

Cited by: [§III-A](#S3.SS1.p1.1).

-
[20]
T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik (2025)

Learning visuotactile skills with two multifingered hands.

In ICRA,

Cited by: [§II](#S2.p1.1),
[§III-B](#S3.SS2.p2.1).

-
[21]
X. Liu, H. Wang, and L. Yi (2025)

DexNDM: closing the reality gap for dexterous in-hand rotation via joint-wise neural dynamics model.

arXiv:2510.08556.

Cited by: [§II](#S2.p3.1).

-
[22]
A. Loquercio, E. Kaufmann, R. Ranftl, M. Müller, V. Koltun, and D. Scaramuzza (2021)

Learning high-speed flight in the wild.

Science Robotics, pp. eabg5810.

Cited by: [§I](#S1.p2.1).

-
[23]
A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, S. Reed, K. Goldberg, A. Mandlekar, L. Fan, and Y. Zhu (2025)

Sim-and-real co-training: a simple recipe for vision-based robotic manipulation.

RSS.

Cited by: [§II](#S2.p4.1).

-
[24]
V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State (2021)

Isaac gym: high performance gpu-based physics simulation for robot learning.

In NeurIPS Datasets and Benchmarks,

Cited by: [§I](#S1.p1.1),
[§IV-A](#S4.SS1.p2.3).

-
[25]
I. Mordatch, Z. Popović, and E. Todorov (2012)

Contact-invariant optimization for hand manipulation.

In Eurographics,

Cited by: [§II](#S2.p1.1).

-
[26]
A. S. Morgan, K. Hang, B. Wen, K. Bekris, and A. M. Dollar (2022)

Complex in-hand manipulation via compliance-enabled finger gaiting and multi-modal planning.

RA-L.

Cited by: [§II](#S2.p1.1).

-
[27]
Y. Narang, K. Storey, I. Akinola, M. Macklin, P. Reist, L. Wawrzyniak, Y. Guo, A. Moravanszky, G. State, M. Lu, A. Handa, and D. Fox (2022)

Factory: fast contact for robotic assembly.

In RSS,

Cited by: [§I](#S1.p2.1).

-
[28]
A. Newell, K. Yang, and J. Deng (2016)

Stacked hourglass networks for human pose estimation.

In ECCV,

Cited by: [§III-C](#S3.SS3.p2.5).

-
[29]
M. Noseworthy, B. Tang, B. Wen, A. Handa, C. Kessens, N. Roy, D. Fox, F. Ramos, Y. Narang, and I. Akinola (2025)

Forge: force-guided exploration for robust contact-rich manipulation under uncertainty.

RA-L.

Cited by: [§II](#S2.p3.1).

-
[30]
A. M. Okamura, N. Smaby, and M. R. Cutkosky (2000)

An overview of dexterous manipulation.

In ICRA,

Cited by: [§II](#S2.p1.1).

-
[31]
OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang (2019)

Solving rubik’s cube with a robot hand.

arXiv:1910.07113.

Cited by: [§I](#S1.p1.1).

-
[32]
OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Józefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba (2019)

Learning dexterous in-hand manipulation.

IJRR.

Cited by: [§I](#S1.p1.1),
[§I](#S1.p2.1),
[§II](#S2.p1.1),
[TABLE V](#Sx2.T5).

-
[33]
S. Patidar, A. Sieler, and O. Brock (2023)

In-hand cube reconfiguration: simplified.

In IROS,

Cited by: [§II](#S2.p1.1).

-
[34]
X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel (2018)

Sim-to-real transfer of robotic control with dynamics randomization.

In ICRA,

Cited by: [§I](#S1.p1.1),
[§III-A](#S3.SS1.p9.1).

-
[35]
L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel (2017)

Asymmetric actor critic for image-based robot learning.

arXiv:1710.06542.

Cited by: [§IV-E](#S4.SS5.p1.1).

-
[36]
H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik (2022)

In-hand object rotation via rapid motor adaptation.

In CoRL,

Cited by: [§I](#S1.p1.1),
[§II](#S2.p1.1),
[§III-A](#S3.SS1.p2.1).

-
[37]
H. Qi, B. Yi, S. Suresh, M. Lambeta, Y. Ma, R. Calandra, and J. Malik (2023)

General in-hand object rotation with vision and touch.

In CoRL,

Cited by: [§I](#S1.p2.1),
[§III-A](#S3.SS1.p3.5).

-
[38]
Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y. Chao, and D. Fox (2023)

AnyTeleop: a general vision-based dexterous robot arm-hand teleoperation system.

In RSS,

Cited by: [§I](#S1.p3.1),
[§II](#S2.p1.1).

-
[39]
S. Ross, G. Gordon, and D. Bagnell (2011)

A reduction of imitation learning and structured prediction to no-regret online learning.

In AISTATS,

Cited by: [§III-A](#S3.SS1.p8.1).

-
[40]
D. Rus (1999)

In-hand dexterous manipulation of piecewise-smooth 3-d objects.

IJRR.

Cited by: [§II](#S2.p1.1).

-
[41]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)

Proximal policy optimization algorithms.

arXiv:1707.06347.

Cited by: [§III-A](#S3.SS1.p7.2).

-
[42]
R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. Van Wyk (2024)

Dextrah-rgb: visuomotor policies to grasp anything with dexterous hands.

arXiv:2412.01791.

Cited by: [§I](#S1.p2.1).

-
[43]
T. L. Team, J. Barreiros, A. Beaulieu, A. Bhat, R. Cory, E. Cousineau, H. Dai, C. Fang, K. Hashimoto, M. Z. Irshad, M. Itkina, N. Kuppuswamy, K. Lee, K. Liu, D. McConachie, I. McMahon, H. Nishimura, C. Phillips-Grafflin, C. Richter, P. Shah, K. Srinivasan, B. Wulfe, C. Xu, M. Zhang, A. Alspach, M. Angeles, K. Arora, V. C. Guizilini, A. Castro, D. Chen, T. Chu, S. Creasey, S. Curtis, R. Denitto, E. Dixon, E. Dusel, M. Ferreira, A. Goncalves, G. Gould, D. Guoy, S. Gupta, X. Han, K. Hatch, B. Hathaway, A. Henry, H. Hochsztein, P. Horgan, S. Iwase, D. Jackson, S. Karamcheti, S. Keh, J. Masterjohn, J. Mercat, P. Miller, P. Mitiguy, T. Nguyen, J. Nimmer, Y. Noguchi, R. Ong, A. Onol, O. Pfannenstiehl, R. Poyner, L. P. M. Rocha, G. Richardson, C. Rodriguez, D. Seale, M. Sherman, M. Smith-Jones, D. Tago, P. Tokmakov, M. Tran, B. V. Hoorick, I. Vasiljevic, S. Zakharov, M. Zolotas, R. Ambrus, K. Fetzer-Borelli, B. Burchfiel, H. Kress-Gazit, S. Feng, S. Ford, and R. Tedrake (2025)

A careful examination of large behavior models for multitask dexterous manipulation.

arXiv:2507.05331.

Cited by: [§III-C](#S3.SS3.p3.3).

-
[44]
C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu (2024)

Dexcap: scalable and portable mocap data collection system for dexterous manipulation.

In RSS,

Cited by: [§II](#S2.p1.1).

-
[45]
J. Wang, Y. Yuan, H. Che, H. Qi, Y. Ma, J. Malik, and X. Wang (2024)

Lessons from learning to spin ‘pens’.

In CoRL,

Cited by: [§I](#S1.p2.1),
[§II](#S2.p2.1),
[§III-A](#S3.SS1.p3.5),
[§III-A](#S3.SS1.p9.1),
[§IV-C](#S4.SS3.p6.1).

-
[46]
S. Wang, M. Lambeta, P. Chou, and R. Calandra (2022)

Tacto: a fast, flexible, and open-source simulator for high-resolution vision-based tactile sensors.

RA-L.

Cited by: [§I](#S1.p2.1).

-
[47]
M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song (2025)

DexUMI: using human hand as the universal manipulation interface for dexterous manipulation.

In CoRL,

Cited by: [§II](#S2.p1.1).

-
[48]
M. Yang, C. Lu, A. Church, Y. Lin, C. Ford, H. Li, E. Psomopoulou, D. A. Barton, and N. F. Lepora (2024)

AnyRotate: gravity-invariant in-hand object rotation with sim-to-real touch.

In CoRL,

Cited by: [§I](#S1.p2.1).

-
[49]
J. Yin, H. Qi, J. Malik, J. Pikul, M. Yim, and T. Hellebrekers (2025)

Learning in-hand translation using tactile skin with shear and normal force sensing.

In ICRA,

Cited by: [§I](#S1.p2.1).

-
[50]
Z. Yin, C. Wang, L. Pineda, F. Hogan, K. Bodduluri, A. Sharma, P. Lancaster, I. Prasad, M. Kalakrishnan, J. Malik, M. Lambeta, T. Wu, P. Abbeel, and M. Mukadam (2025)

Dexteritygen: foundation controller for unprecedented dexterity.

arXiv:2502.04307.

Cited by: [§II](#S2.p3.1).

-
[51]
H. Zhang, Z. Li, X. Zeng, L. Smith, K. Stachowicz, D. Shah, L. Yue, Z. Song, W. Xia, S. Levine, et al. (2025)

Traversability-aware legged navigation by learning from real-world visual data.

T-RO.

Cited by: [§II](#S2.p4.1).

-
[52]
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)

Learning fine-grained bimanual manipulation with low-cost hardware.

In RSS,

Cited by: [§I](#S1.p3.1),
[§III-C](#S3.SS3.p2.5).

## Appendix

### V-A Privileged Information for Oracle Policy

Nut-Bolt Task. The oracle policy receives privileged information about object properties, fingertip states, nut-specific dynamics, hand states, and PD controller gains. The full list of privileged inputs is provided in Table [III](#Sx2.T3).

Screwdriver Task.
The screwdriver task uses a subset of the privileged information from the nut-bolt task. Specifically, it excludes hand base position, hand orientation, hand joint positions, and PD controller gains. All other privileged inputs remain the same.

### V-B Reward

Our reward function is a weighted combination of task rewards, energy penalties, and stability penalties:

-
•

Task Rewards encourage successful rotation:

-
$\circ$

Rotation Rewards:
$r^{\text{rot}}_{t}=\text{clip}(\omega_{t},\omega_{\min},\omega_{\max})$. It encourages positive angular velocity $\omega$ along the fastening axis, clipped to $[-4.0,4.0]$ rad/s.

-
$\circ$

Proximity Rewards:
$r^{\text{prox}}_{t}=\max(0,\,1-d_{t}/d_{\text{thresh}})$. It encourages fingers to remain close to the object, where $d_{t}$ is the mean distance from thumb and index finger to the nut/handle.

-
•

Energy Penalties discourage inefficient motions:

-
$\circ$

Torque Penalty: $r^{\text{torque}}_{t}=-\|\tau_{t}\|^{2}$ penalizes large joint torques.

-
$\circ$

Work Penalty: $r^{\text{work}}_{t}=-(|\tau_{t}|^{\top}|\dot{q}_{t}|)^{2}$ penalizes excessive joint power.

-
•

Stability penalties maintain stable behavior:

-
$\circ$

Pose Difference Penalty: $r^{\text{pose}}_{t}=-\|q_{t}-q^{0}\|^{2}$ penalizes deviations from the initial finger configuration (thumb joints masked).

-
$\circ$

Large Rotation Penalty: $r^{\text{rp}}_{t}=-\max(0,\omega_{t}-\omega_{\text{thresh}})$ penalizes excessive angular velocity above threshold $\omega_{\text{thresh}}$ (10.0 rad/s for nut-bolt, curriculum from 7.5 to 15.0 rad/s for screwdriver).

We sum the above rewards with weights listed in Table [IV](#Sx2.T4).

*TABLE III: Privileged information for nut-bolt task. The screwdriver task uses a subset of these features (excludes hand pose and PD gains).*

*TABLE IV: Reward function hyper-parameters for nut-bolt and screwdriver tasks.*

*TABLE V: Domain Randomization Parameters. Object scale is discretely sampled from the specified set and multiplied by the base scale. Mass, center of mass, friction, restitution, and PD controller gains are uniformly sampled at environment initialization. Observation and action noise are sampled i.i.d. from Gaussian distributions at each timestep. Following [[[32](#bib.bib26)]], we apply a random disturbance force with magnitude $2.0m$ (where $m$ is object mass) with probability 0.25 at each timestep.*

### V-C Training Hyperparameters

The inputs to our oracle policy contains the proprioceptive observations consist of $q_{t}$ (joint positions over the last 3 timesteps) and $a_{t-1}$ (previous joint targets over the last 3 timesteps). The privileged information includes $p_{t}$ (5 fingertip positions), $w_{t}$ (object state with 3D position, quaternion orientation, and angular velocity), and additional features detailed in Table [III](#Sx2.T3). We follow the domain randomization parameters in Table [V](#Sx2.T5).

We train our oracle policy with PPO, and the training hyperparameters are shown in Table [VI](#Sx2.T6). Specifically, we train with 8,192 parallel environments. Each environment gathers 12 steps of data to train in each epoch of PPO. The data is split into minibatches of size 16,384 and optimized with PPO loss. $\gamma$ and $\lambda$ are used for computing generalized advantage estimate (GAE) returns. We use the Adam optimizer to train PPO and adopt gradient clipping to stabilize training. We train 1.5 billion environment steps in total, which takes less than one day on a single GPU. We train our sensorimotor policy with on-policy behavioral cloning, and the training hyperparameters are shown in Table [VII](#Sx2.T7).

*TABLE VI: Hyperparameters for training the oracle policy.*

*TABLE VII: Hyperparameters for training the sensorimotor policy in simulation.*

Generated on Wed Feb 25 00:03:21 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)