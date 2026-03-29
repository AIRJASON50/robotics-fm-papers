[1]FAIR at Meta 2]Carnegie Mellon University \contribution[*]Work done at Meta \contribution[†]Joint last author # SPIDER: Scalable Physics-Informed Dexterous Retargeting Chaoyi Pan Changhao Wang Haozhi Qi Zixi Liu Homanga Bharadhwaj Akash Sharma Tingfan Wu Guanya Shi Jitendra Malik Francois Hogan [ [ chaoyip@andrew.cmu.edu](mailto:chaoyip@andrew.cmu.edu)

(February 5, 2026)

###### Abstract

Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive.
In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem.
However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots.
To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale.
Our key insight is that human demonstrations should provide global task structure and objective, while *large-scale physics-based sampling* with *curriculum-style virtual contact guidance* should refine trajectories to ensure dynamical feasibility and correct contact sequences.
SPIDER scales across diverse $9$ humanoid/dexterous hand embodiments and $6$ datasets, improving success rates by $18\%$ compared to standard sampling, while being $10\times$ faster than reinforcement learning (RL) baselines, and enabling the generation of a $2.4$M frames dynamic-feasible robot dataset for policy learning.
As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.

\correspondence\metadata

[Website][https://jc-bao.github.io/spider-project](https://jc-bao.github.io/spider-project)

*Figure 1: Overview of SPIDER method. SPIDER converts human-object interaction trajectories to dynamically feasible robot-object interaction trajectories using sampling with parallel physics simulator. We introduce an additional virtual contact guidance method to minimize the solution ambiguity in contact-rich tasks. With the combination of the two, SPIDER converts human dataset to deployable robot data at scale and supports multiple distinct robot embodiments and task domains.*

## 1 Introduction

Table-top manipulation has rapidly advanced by learning from massive internet-scale datasets [kimOpenVLAOpenSourceVisionLanguageAction2024, khazatsky2024droid, brohan2023rt2visionlanguageactionmodelstransfer] and human demonstrations [rdt2, qu2025eo1interleavedvisiontextactionpretraining].
Yet, acquiring generalizable whole-body manipulation across embodiments, from dexterous hands [yinDexterityGenFoundationController2025, zhongDexGraspVLAVisionLanguageActionFramework2025, qi2023general] to humanoid whole-body control [liAMOAdaptiveMotion2025, zeTWISTTeleoperatedWholeBody2025], remains prohibitively expensive due to hardware constraints, task complexity, and lack of large-scale embodiment-specific data [zhangDOGloveDexterousManipulation2025, xuDexUMIUsingHuman2025].
On the other hand, abundant human motion data is readily available, including large-scale motion-capture datasets [zhanOAKINK2DatasetBimanual2024, mahmoodAMASSArchiveMotion2019], internet-scale video collections [renMotionTracksUnified2025], and VR-based interfaces [hoqueEgoDexLearningDexterous2025].
Recent advances in computer vision have made it possible to reconstruct 3D human body [goelHumans4DReconstructing2023] and hand motion [pavlakosReconstructingHands3D2023], as well as object meshes [xiangStructured3DLatents2025] and trajectories [wenFoundationPoseUnified6D2024], directly from videos.
These developments create a unique opportunity to leverage human motion as a scalable source of demonstrations for learning humanoid and dexterous robot control.
However, a fundamental challenge arises: the embodiment gap* - the mismatch in morphology, dynamics, and actuation between humans and robots - which leads to infeasible motion during transfer.
This motivates our central research question:

How can we efficiently and reliably transform human motion into feasible robot trajectories that respect dynamics and contact?

We formulate this as a physics-based retargeting problem [redaPhysicsbasedMotionRetargeting2023]: given kinematic human demonstrations, generate robot motions that (a) align the robot’s poses with human poses, (b) establish consistent contact with the environment, and (c) preserve the task objectives of the demonstrated behavior.
The problem presents three key challenges:
(a) *Dynamical feasibility:* There is a substantial embodiment gap between humans and robots, and reconstructed demonstrations from mesh estimation and state tracking are often noisy, making direct kinematic transfer infeasible.
(b) *Scalability and efficiency:* The abundance of internet-scale human data requires approaches that are both computationally efficient and scalable to large datasets.
(c) *Robustness and missing contact information:* Most human datasets lack the force and contact data required to ensure dynamical feasibility and preserve manipulation intent.
Existing methods struggle to bridge these gaps:
inverse-kinematics (IK) approaches [qinOneHandMultiple2023, yinGeometricRetargetingPrincipled2025] are efficient but dynamically infeasible;
reinforcement learning (RL) approaches [liManipTransEfficientDexterous2025, lumCrossingHumanRobotEmbodiment2025] are general but requires expensive trajectory-specific training and tedious reward engineering;
teleoperation [yinDexterityGenFoundationController2025] is dynamically feasible but often labor-intensive and embodiment-specific.

To achieve scalable, general, and flexible physics-based cross-embodiment retargeting, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a sampling-based approach with contact guidance.
Our key insight is that human demonstrations provide high-level guidance in terms of robot motion and task specification, while sampling in simulation grounds trajectories with physics to ensure dynamical feasibility and contact correctness.
To reduce contact ambiguity during sampling, we introduce a virtual contact guidance mechanism: a virtual force is added between the robot and the object to “stick” the object to the desired contact point in the initial stage, and is gradually relaxed as the optimization progresses.
Importantly, the framework is embodiment-agnostic and task-general:
it can be applied to any robot-environment interaction as long as the scene can be simulated.

Contributions. Our contributions are summarized as follows:

-
•

We introduce SPIDER, a flexible and general physics-informed retargeting framework that scales across six datasets, nine distinct robot morphologies, and two task domains (dexterous hand and humanoid).

-
•

We propose a contact-aware scheme that incorporates object- and environment-centric guidance to preserve manipulation and locomotion intent. It improves the success rate by $18\%$ compared to the baseline and increases retargeting speed by an order of magnitude compared to RL-based methods.

-
•

Our pipeline enables the generation of large-scale, robot-feasible datasets, with 262 episodes, 800 hours of data, and 2.4M frames across five distinct robotic hands and 103 different objects, derived from human data. We release the full data generation pipeline to assist future research.

-
•

SPIDER is further extended for downstream applications, such as robustifying trajectories for direct real-robot deployment, augmenting a single demonstration to diverse physical environments/objects, and boosting the learning process of RL policies.

*Figure 2: Overview of the SPIDER pipeline. The pipeline takes reconstructed object meshes, reference robot motion, and object motion and converts them into a dynamically feasible robot trajectory with corrected contacts. The generated trajectory is further robustified and augmented before deployment or policy learning.*

## 2 Physics-based Retargeting with Sampling

This section introduces our pipeline for retargeting human manipulation data to physical robot demonstrations, as illustrated in [figure˜2](#S1.F2).
We first formulate the physics-based retargeting problem in [section˜2.1](#S2.SS1), followed by its sampling-based solver design in [section˜2.2](#S2.SS2).
Then, we further improve sampling efficiency and quality with virtual contact guidance in [section˜2.3](#S2.SS3).
Finally, to handle model mismatch between simulation and reality, we introduce a robustification strategy in [section˜2.4](#S2.SS4).

### 2.1 Physics-based Retargeting Problem

We formulate physics-based retargeting as a constrained optimization problem where a robot control sequence $u_{0:T-1}$ is optimized to minimize the distance to the reference trajectory $x^{\text{ref}}_{0:T}$ and the control effort.
As input*, a kinematic reference state sequence $x^{\text{ref}}_{t},\forall t\in\{0,\ldots,T\}$ is first provided from human demonstrations.
The state $x^{\text{ref}}_{t}=\{q^{\text{ref}}_{t},\dot{q}^{\text{ref}}_{t}\}$ is composed of reference position $q^{\text{ref}}_{t}$ and velocity $\dot{q}^{\text{ref}}_{t}$,
where $q^{\text{ref}}_{t}=\{q^{\text{ref,robot}}_{t},q^{\text{ref,object}}_{t}\}$.
Specifically, for robots with $n_{u}$ joints, its reference position $q^{\text{ref,robot}}_{t}$ is composed of robot joint position $q^{\text{ref,joint}}_{t}\in\mathbb{R}^{n_{u}}$ and its base transformation $T^{\text{ref,base}}_{t}\in\mathrm{SE}(3)$.
The object reference motion is $q^{\text{ref,object}}_{t}\in\mathrm{SE}(3)$.
Physics-based retargeting will *output* a dynamically feasible control sequence of the robot $u_{0:T-1}$ to minimize the distance to the reference trajectory and the control effort:

| |
|---|
| | $\displaystyle\begin{split}\min_{u_{0:T-1}}J(u_{0:T-1})&=\min_{u_{0:T-1}}\|x_{T}-x_{T}^{\text{ref}}\|_{Q_{T}}^{2}+\sum_{t=0}^{T-1}\left(\left\|x_{t+1}-x_{t+1}^{\text{ref}}\right\|_{Q_{t}}^{2}+\|u_{t}\|_{R_{t}}^{2}\right)\end{split}$ | | (1a) |
| | $\displaystyle\text{s.t.}\quad x_{t+1}$ | $\displaystyle=f(x_{t},u_{t},t)\quad\forall t\in\{0,\ldots,T-1\}$ | | (1b) |
| | $\displaystyle x_{0:T}$ | $\displaystyle\in\mathcal{X},\quad u_{0:T-1}\in\mathcal{U}$ | | (1c) |

where $x_{0:T}$ is the optimized feasible state, $x_{t+1}=f(x_{t},u_{t},t)$ is the state transition function, $\mathcal{X}$ is the state space, $\mathcal{U}$ is the control input space, and $Q_{t}$ and $R_{t}$ are the state and control input weighting matrices.
In practice, we use diagonal weighting matrices for $Q_{t}$ and $R_{t}$, where $Q_{t}=\text{diag}(\{q_{\text{robot}},q_{\text{object}}\})$ and $R_{t}=\text{diag}(\{r_{\text{robot}},r_{\text{object}}\})$.

### 2.2 Sampling for Physics-based Retargeting

Due to the contact-rich nature of physics-based retargeting, the optimization problem in [equation˜1a](#S2.E1.1) is highly non-convex and often non-continuous.
Sampling-based optimization [mannorCrossEntropyMethod] provides a natural way to handle such landscapes, as it does not rely on smoothness or convexity assumptions.
Intuitively, this resembles RL in that both rely on sampled trajectories from parallelized simulation to guide decision-making, but instead of updating a policy network, we directly use the samples to optimize the control sequence $U=u_{0:T-1}$.
To this end, we adopt a sampling-based optimizer with an annealed sampling kernel [xueFullOrderSamplingBasedMPC2024, panModelBasedDiffusionTrajectory2024]:

$$
\begin{aligned}
U^{i+1} &=U^{i}+\frac{\sum_{j=1}^{N_{W}}\exp\left(-\frac{J(U^{i}+[W]_{j})}{\lambda}\right)[W]_{j}}{\sum_{j=1}^{N_{W}}\exp\left(-\frac{J(U^{i}+[W]_{j})}{\lambda}\right)}, \tag{2}
\end{aligned}
$$
$$
\begin{aligned}
\Sigma^{i}_{h} &=\exp\left(-\frac{N-i}{\beta_{1}N}-\frac{H-h}{\beta_{2}H}\right)I, \tag{3}
\end{aligned}
$$

where $U^{i}$ is the solution at iteration $i$, $i=1,\ldots,N$ is the iteration index with $N$ being the total number of iterations, $[W]_{j}\sim\mathcal{N}(0,\Sigma_{0:H-1}),\penalty 10000\ j=1,\ldots,N_{W}$ is sampled $N_{W}$ Gaussian noise with scheduled covariance $\Sigma_{h},h\in\{0,\ldots,H-1\}$, $\beta_{1},\beta_{2}\in(0,1)$ are annealing parameters controlling sampling covariance in line [6](#alg1.l6) of [algorithm˜1](#alg1).

*Algorithm 1 Sampling for Physics-based Retargeting*

In contact-rich dexterous manipulation, the *feasible solution set is typically narrow*, so effective optimization requires a combination of *coarse search* to discover feasible contact modes and *fine refinement* to achieve stable contact.
The annealed sampling covariance in [equation˜2](#S2.E2) implements this exploration-exploitation trade-off: $\beta_{1}$ controls the rate across outer iterations (global-to-local search over optimization updates), and $\beta_{2}$ controls the rate along the prediction horizon (allocating more or less perturbation across timesteps).
As the schedule progresses, the effective sampling radius transitions from broad exploration to targeted exploitation around promising trajectories.
[Figure˜3](#S2.F3) illustrates this effect: unlike standard sampling (e.g. MPPI) using a fixed search radius, annealed sampling shrinks the radius in later iterations, effectively reducing the variance of the sampled trajectories.
To further improve speed, we adopt an early stopping strategy [kamatBITKOMOCombiningSampling2022], where optimization halts when improvements from further sampling become small.
[Algorithm˜1](#alg1) presents a base sampling framework for physics-based retargeting.

### 2.3 Virtual Contact Guidance

Solutions Ambiguity.
For a retargeting method to be valid, it is also important to preserve the human-preferred contact in the demonstration.
Take the stir stick manipulation in [figure˜3](#S2.F3) as an example, the robot can hold the stick either between the thumb and index finger or between the index and middle finger; both achieve the task, but the former is more natural and aligns with the human demonstration. However, due to the non-convex cost landscape of [equation˜1a](#S2.E1.1) there may be multiple solutions that achieve similar object motion with different contact behaviors.
As a result, a sampling-based optimizer may converge to alternative contact modes, leading to implausible contact trajectories in a demonstration even when the object trajectory is reproduced.

Virtual Contact Guidance.

*Figure 3: Contact mode mismatch in sampling and virtual guidance method to correct it. (a) Given the same task, the robot can hold the object in different contact modes while still finishing the task. However, the contact mode from human is preferred. (b) Given different sampling methods when seeking a feasible motion with correct contact: Standard sampling (left): uses a fixed search radius, leading to high variance. The resulting solution fails to converge well. Annealed sampling (middle): gradually shrinks the search radius, starting coarse and narrowing down to a finer solution, but may drift toward a feasible solution with wrong contact. Annealed sampling with virtual contact guidance (right): expands the feasible region by adding virtual guidance near target contacts. This enlarges the feasible region to a relaxed feasible set, steering sampling away from undesired feasible solutions and towards the intended contact sequence.*

To guide sampling towards the desired contact mode, we introduce virtual contact guidance*. This guidance factor expands the feasible solution set and biases the optimization process towards the target configuration.
Unlike a soft contact reward or cost - which may still fail when the desired mode is harder to sample - virtual contact guidance explicitly *enlarges* the basin of attraction to make it easier to sample from, as illustrated in [figure˜3](#S2.F3).
Our method applies a virtual constraint between intended contact points on the object and robot fingers, “sticking” the object to the target configuration in early stages and gradually relaxing this constraint in a curriculum-like fashion.
Concretely, we impose a virtual constraint that maintains the relative position between contact pairs. For the $k$-th object-hand contact pair (e.g., handle-thumb), we define the current relative position as ${}^{\text{robot}}p_{k,t}^{\text{object}}=p^{\text{robot}}_{k,t}-p^{\text{object}}_{k,t}$ and the desired reference position as ${}^{\text{robot}}p_{k,t}^{\text{object,ref}}=p^{\text{robot,ref}}_{k,t}-p^{\text{object,ref}}_{k,t}$.
The constraint is activated when the contact indicator $c_{k,t}=1$, which occurs when the reference relative position is within the contact threshold: $c_{k,t}=\mathbf{1}(\|{}^{\text{robot}}p_{k,t}^{\text{object,ref}}\|_{2}\leq\epsilon_{\text{contact}})$.
The constraint strength is controlled by a penalty parameter $\eta_{i}\rightarrow\infty$ when $i\rightarrow N$:

$$c_{k,t}\|{}^{\text{robot}}p_{k,t}^{\text{object}}-{}^{\text{robot}}p_{k,t}^{\text{object,ref}}\|_{2}^{2}\leq\eta_{i}$$ \tag{4}

This strategy connects to prior ideas: virtual object constraint [mandiDexMachinaFunctionalRetargeting2025] $\|p_{k,t}^{\text{object}}-p_{k,t}^{\text{object,ref}}\|_{2}^{2}\leq\eta$, which expands feasibility around absolute object states in RL, and contact cost [lakshmipathyKinematicMotionRetargeting2024], biasing optimization towards desired hand–object *relative* states. Our formulation reduces sampling complexity while preserving the intended contact sequence by maintaining *relative* hand–object relationships.

Robustness against imperfect reference contact. It is important to note that, in order to maintain robustness against imperfect demonstrations with noisy or unstable contact, virtual constraints should be selectively relaxed rather than enforced indiscriminately.
To achieve this, we apply a contact filter that detects unstable interactions: if a contact duration is shorter than $t_{c,\text{min}}$ or if the contact point drifts more than $d_{c,\text{max}}$ during that period, the contact is classified as unstable and the corresponding virtual constraint is disabled. This implementation ensures that only reliable contacts contribute to guidance, thereby preventing noisy demonstrations from biasing the optimization process.

### 2.4 Trajectory Robustification

Handling imperfection in reference motion.
To bridge the gap from reconstructed demonstrations to real hardware, we robustify trajectories against unknown or misspecified dynamics (e.g., friction, contact compliance) and reconstruction noise (e.g., object mesh estimation error, pose estimation error) that could otherwise make a nominal plan infeasible.
Our approach optimizes a control sequence with a pessimistic (min–max) objective over a bounded parameter set $\mathcal{D}$, similar to domain randomization (DR) method [tobinDomainRandomizationTransferring2017a] but replacing expectation with worst-case cost to ensure universal feasibility:

$$J_{\text{rob}}(U)=\max_{d\in\mathcal{D}}J(U,d),$$ \tag{5}

where $J(U,d)$ is the cost in [equation˜1a](#S2.E1.1) under dynamics parameter $d$.
In practice, $\mathcal{D}$ spans variations such as contact margin size, friction coefficients, and object mass.
During optimization, each candidate sequence $U_{j}$ is rolled out over a mini-batch $d_{1:K}\sim\mathcal{D}$ and evaluated by $\max_{k\leq K}J(U_{j},d_{k})$, following robust sampling-based control [williamsRobustSamplingBased2018].
This formulation integrates seamlessly with the update rule in [equation˜2](#S2.E2), leverages GPU parallelization through batched rollouts, and has minimum computational overhead.

### 2.5 Physics-based Data Augmentation

Apart from generating feasible robot actions, another advantage of physics-based retargeting is the ability to systematically augment the retargeted data with diverse *physics-aware* behaviors.
Starting from a single human demonstration, SPIDER is capable of diversifying a single behavior into a diverse set of physically feasible actions which could be used for downstream training [mandlekar2023mimicgen, jiangDexMimicGenAutomatedData2025, yinDexterityGenFoundationController2025].

*Figure 4: Physics-based data augmentation. We augment the retargeted data from a single demonstration into a diverse set of physically feasible actions. Here we demonstrate (a) generating motion with new object mesh for dexterous manipulation, (b) moving a lighter and smaller object for humanoid robot tasks, (c) adding stairs to the scene for humanoid running motion. (d) applying external forces to the robot when it is pulling a heavy object.*

Geometric Variations.
SPIDER supports generating diverse geometric variations of the objects while still grounding it with physics.
For dexterous manipulation tasks, we perturb the initial object pose and scale to generate diverse interaction behaviors while the reference motion is kept the same.
We can even directly replace the object mesh with a different one to generate new interaction behaviors.
For instance, in [figure˜4](#S2.F4), we augment the grasping motion of a coke to a cat toy simply by replacing the object mesh.
Similarly, the same idea can be applied to human-object interaction for humanoid robot tasks.
By modifying the object geometry to make it smaller and lighter, the robot can automatically adapt to the new object by lowering the body down and switching to single-hand pushing motion with smaller force.
One can also changing the terrain to stairs to transform a running in flat ground motion into a running on stairs motion, which is challenging for kinematic-only retargeting due to climbing stairs requires new contact patterns.

Physics Variations.
The distinct advantage of SPIDER as a physics-based approach is to be force-aware.
We demonstrate the ability of SPIDER to generating new behaviors by applying external forces to the robot when it is pulling a heavy object.
A large force ($120$N and $240$N respectively, where the gravity is $320$N) is applied to pull the robot back.
With physics-based retargeting, the generated trajectory is able to resist the force by leaning forward and moving slower, as demonstrated in [figure˜4](#S2.F4).

## 3 Performance Evaluation

This section numerically evaluates the proposed method with settings detailed in [section˜3.1](#S3.SS1).
First, in [section˜3.2](#S3.SS2), we ablate the effect of the annealed kernel and virtual contact guidance in sampling and compare them with the kinematic-retargeting baseline.
Then, to provide a quantitative assessment of generated motion quality, we compare SPIDER with state-of-the-art retargeting baselines on dexterous manipulation [section˜3.3](#S3.SS3) and humanoid whole-body control [section˜3.4](#S3.SS4).

### 3.1 Experimental Setup

Dataset Selection.
For dexterous manipulation tasks, we evaluate on 3 bimanual manipulation datasets: GigaHands [fuGigaHandsMassiveAnnotated2024], Oakink [zhanOAKINK2DatasetBimanual2024] and ARCTIC [li2023object], comprising in total 1262 episodes and 2.4M frames across five distinct robotic hands and 103 different objects.
For humanoid tasks, we choose commonly used LAFAN1 [harvey2020robust], AMASS [mahmoodAMASSArchiveMotion2019] for locomotion and OMOMO [li2023object] dataset for human-robot interaction.
To the best of our knowledge, SPIDER is the first universal retargeting method that can handle both dexterous manipulation and humanoid whole-body control tasks at this scale.

Robot Embodiments.
For dexterous manipulation, we evaluate across 5 different robotic hands to demonstrate the generalizability of our method: Allegro, XHand, Inspire, Ability, and Schunk.
On humanoid whole-body control tasks, we use Unitree G1, Unitree H1-2, Fourier N1 and Booster T1 as the target embodiment.
These platforms exhibit significant variations in degrees of freedom, dimensions, and finger configurations (see [figure˜5](#S3.F5)), showcasing our method’s cross-embodiment capabilities.

![Figure](x6.png)

*Figure 5: Specifications of robots used in evaluation. SPIDER supports both dexterous hand and humanoid robot. The significant variations in DoF, dimensions, and finger count demonstrate the cross-embodiment generalizability of our approach. We employ a simulated 12-DoF configuration [liManipTransEfficientDexterous2025] of the Inspire and Ability hands, removing the joint constraints compared to their real-world versions.*

Evaluation Metrics.
For dexterous manipulation and humanoid-object interaction tasks, we evaluate tracking performance on the object motion, since the goal is to move the object rather than to precisely track the robot hand.
For object tracking error, we compute per-step averaged rotation error $E_{\text{rot}}=\frac{1}{T}\sum_{t=1}^{T}\arccos\left(2\langle q_{{\text{obj}},t},q_{{\text{obj}},t}^{\text{ref}}\rangle^{2}-1\right)$ and position error $E_{\text{pos}}=\frac{1}{T}\sum_{t=1}^{T}\|p_{{\text{obj}},t}-p_{{\text{obj}},t}^{\text{ref}}\|_{2}$, where $T$ is the number of timesteps, $q_{{\text{obj}},t}$ and $p_{{\text{obj}},t}$ are the object quaternion and position at timestep $t$, respectively.
We follow the same evaluation setting as ManipTrans [liManipTransEfficientDexterous2025] and DexMachina [mandiDexMachinaFunctionalRetargeting2025], where we report task success only concerning object motion. Task success is defined using: (1) the rotation error $E_{\text{rot}}Oakink [zhanOAKINK2DatasetBimanual2024], we evaluate on 7 distinct bimanual two-object manipulation tasks that were previously benchmarked in ManipTrans [liManipTransEfficientDexterous2025]; and (2) for GigaHands [fuGigaHandsMassiveAnnotated2024], we utilize their released example dataset comprising 3 manipulation trajectories.

Baseline Selection. We compare four approaches: (1) Kinematic retargeting: fingertip inverse kinematics as an initial-guess quality baseline; (2) vanilla Sampling (based on MPPI [howellPredictiveSamplingRealtime2022]): standard sampling-based control; (3) Sampling with an annealed kernel (similar to DIAL-MPC [xueFullOrderSamplingBasedMPC2024]): differs from our method only in the absence of virtual guidance; and (4) Sampling with an annealed kernel and virtual contact guidance (our full method).
We exclude RL-based comparisons here since they produce policies rather than direct control sequences, making direct comparison inappropriate. RL methods are evaluated separately in [section˜3.3](#S3.SS3) for motion quality assessment.

![Figure](x7.png)

*Table 1: Ablation study success rates across different datasets and robot hands. Results evaluated on eight example trajectories of Oakink from ManipTrans and five example trajectories of GigaHands over five seeds. Sampling with both annealing and contact guidance consistently achieves the highest success rates across all robot-dataset combinations.*

Retargeted Dataset Quality.
To validate the quality of the generated data on large scale dataset, we first report the success rates on full dataset used the same metrics as in [section˜3.2](#S3.SS2).
Full results are shown in [section˜3.3](#S3.SS3), where we demonstrate the first large-scale retargeting evaluation across such diverse embodiments and datasets.
Across datasets, Gigahand demonstrates higher success rates due to its focus on pick-and-place motions, which are inherently more amenable to retargeting. In contrast, Oakink presents greater challenges as objects are often pre-grasped, making success heavily dependent on achieving precise initial contact configurations.
Across robot embodiments, hands with higher degrees of freedom (e.g., Inspire, Allegro) are generally easier to retarget as the additional actuators provide greater flexibility.
[Figure˜6](#S3.F6) shows some example trajectories generated by SPIDER on different hands.

*Table 2: Success rates of SPIDER across different datasets and robot hands on the full* dataset. Different hands have various success rates due to their size and dof differences. Both Inspire and Ability hands use 12-DoF simulation model instead of the original one.*

### 3.4 Humanoid Whole-Body Control Retargeting Results

![Figure](x8.png)

*Figure 8: Open-loop rollout of the retargeted control on diverse humanoid robots with different datasets. We demonstrate SPIDER on the LAFAN1, AMASS and OMOMO datasets. SPIDER remove all artifacts from the kinematic trajectory by grounding them with physics and can be applied to various humanoid robots without additional adaptation.*

As a universal physics-based retargeting method, SPIDER is versatile and can be applied to various robot embodiments, including humanoid robots.
To adapt SPIDER for humanoid robots, virtual contact guidance is implemented between the robot’s feet and the floor.
In [figure˜8](#S3.F8), we showcase the application of SPIDER on the humanoid AMASS [mahmoodAMASSArchiveMotion2019] and LAFAN1 [harvey2020robust] datasets.
The retargeting process corrects artifacts such as foot penetration and slipping, enabling the robot to execute highly dynamic motions, as illustrated in [figure˜8](#S3.F8), which shows an open-loop rollout* of the retargeted control and highlights the dynamical feasibility of the retargeted motion.
We compare SPIDER with popular kinematics-based retargeting method, named general motion retargeting (GMR) [araujo2025retargetingmattersgeneralmotion].

![Figure](x9.png)

*Table 4: Tracking Error and FPS Comparison on Humanoid Datasets. Joint Err.: mean joint angle difference (degrees). Pos. Err.: mean end-effector position error (cm). Ori. Err.: mean end-effector orientation error (degrees). Obj. Err.: mean object pose error (cm or degrees). FPS: trajectory generation speed. Our method achieves the lowest tracking errors and highest FPS compared to GMR across all datasets.*

To demonstrate the flexibility of SPIDER for upstreaming data, we evaluate it on converting single RGB camera video into executable robot trajectories.
The pipeline consists of: (1) 3D mesh reconstruction with Trellis [xiangStructured3DLatents2025], (2) hand pose estimation with HAMER [pavlakosReconstructingHands3D2023] to obtain MANO parameters, and (3) object pose tracking with FoundationPose [wenFoundationPoseUnified6D2024] for 6D trajectories.
Due to the occlusion and limited resolution, the human and object motion are more noisy than the dataset, which requires more robust physics grounding.
As shown in [figure˜9](#S4.F9), we validate on single-hand manipulation tasks like pouring with a cup and installing a bolt.
Real-world noise, artifacts, and penetrations are corrected through physics grounding in SPIDER, and the retargeted trajectories are directly executable on physical robots.

### 4.2 Retargeting for RL Policy Training

RL Policy Training. For unstable systems like humanoid robots, the generated trajectory is not directly executable on the real robot due to the lack of feedback.
One common solution is to train a RL policy to track the generated trajectory [heOmniH2OUniversalDexterous2024, yang2025omniretarget, liao2025beyondmimic].
However, starting from infeasible human motion, general tracking policy training relies on heavy regularization [zeTWISTTeleoperatedWholeBody2025, heASAPAligningSimulation2025] and complex curriculum design [heOmniH2OUniversalDexterous2024, liManipTransEfficientDexterous2025] to handle the noisy and infeasible reference motion.
On the other hand, [liao2025beyondmimic, yang2025omniretarget, yangPhysicsDrivenDataGeneration2025] shows with careful retargeting, the policy learning can be easier with less artifical designs.
With trajectory retargeted by SPIDER, which already comes with a feedforward nominal control and a feasible trajectory, the RL policy only needs to learn a residual feedback term to correct the deviation from the nominal control:

$$u_{t}=u^{\textbf{{SPIDER}}}_{t}+\pi_{\theta}(o_{t})$$ \tag{6}

where $u^{\textbf{{SPIDER}}}_{t}$ is the nominal control from SPIDER and $\pi_{\theta}(o_{t})$ is the RL policy output at time $t$.
In practice, we found only joint position tracking reward with object/pelvis pose tracking is sufficient to get a stable motion.

Training Results on Humanoid Robots.
We evaluate the policy training with the adopted framework from [weng2025hdmilearninginteractivehumanoid].
We take standard practice of training PPO but removing the auxiliary contact reward for simplicity.
[Figure˜10](#S4.F10) shows training progress on the OMOMO dataset along side with generated motion.
The original human motion is offsetted from the object position, thus missing the contact.
When training with the original human motion, the robot fails to grasp the object and only achieves body tracking goal.
On the other hand, the policy trained with SPIDER is not only converged faster, but also achieves better object tracking performance.
This again highlights the importance of physics grounding for reference motion.

![Figure](x10.png)

*Figure 10: RL Policy Training on Humanoid Robots. In reference motion, the robot will first pick up the box and then place it on the floor. Due to the motion capture error, the original reference motion missed the contact with the object. After retargeting with SPIDER, the contact is corrected and a feasible feedforward control is provided to assist the policy learning. Left: task progress of the original human motion, demonstrating how many percentage of the target motion is achieved. Right: resulted policy after training.*

## 5 Related Work

### 5.1 Learning Manipulation from Human Data

Recent work explores learning robot skills from large-scale human videos by first detecting actions and transferring them to robots.
One common strategy is kinematic retargeting, where human poses or keypoints are mapped to robot motions, as in DemoDiffusion [parkDemoDiffusionOneShotHuman2025], OKAMI [liOKAMITeachingHumanoid2024], R+X [papagiannisR+XRetrievalExecution2025], and EgoZero [liuEgoZeroRobotLearning2025].
Another strategy is to train a human-centric policy and then adapt it to robots through fine-tuning with in-domain data, as in MimicPlay [wangMimicPlayLongHorizonImitation2023], Track2Act [bharadhwaj2024track2act], and VideoDex [shawVideoDexLearningDexterity2022].
Our approach is complementary to these pipelines. Specifically, SPIDER can serve as a drop-in replacement for the human-to-robot action transfer components, providing more accurate and contact-aware mappings for dexterous and whole-body control.

### 5.2 Retargeting from Human Data

Motion retargeting seeks to convert human data into robot trajectories that are physically consistent and executable.

Kinematic Retargeting.
These methods map human motion to robot configurations [qinOneHandMultiple2023]. While efficient and easy to compute, they often rely on specialized hardware [xuDexUMIUsingHuman2025], handcrafted motion primitives [wuOneShotTransferLongHorizon2024], and struggle with realism in contact-rich tasks [qinAnyTeleopGeneralVisionBased2024]. Being kinematic only, the generated motions are not compliant with physics constraints.

Learning-Based Retargeting Networks.
Neural mapping approaches train networks to convert human motions into robot motions [parkDemoDiffusionOneShotHuman2025, yinDexterityGenFoundationController2025]. Such models can outperform direct kinematic mappings and retain fast inference, but they require extensive pretraining and may fail when facing out-of-distribution motions or novel embodiments.

Optimization-Based Retargeting.
Optimization-based approaches explicitly incorporate physics and contact constraints to ensure dynamical feasibility [redaPhysicsbasedMotionRetargeting2023]. They can generate high-quality, physically plausible motions, but often depend on detailed contact data [lakshmipathyKinematicMotionRetargeting2024], specific data pipelines [yangPhysicsDrivenDataGeneration2025], and strong priors [nakaokaTaskModelLower2005].
Due to the non-convex natural of the problem, sampling-based approaches [yangPhysicsDrivenDataGeneration2025, si2025exostartefficientlearningdexterous] have emerged as a promising solution.

RL-Based Retargeting.
RL has been used to retarget human demonstrations across embodiments [lumCrossingHumanRobotEmbodiment2025, liManipTransEfficientDexterous2025]. When combined with curriculum learning [mandiDexMachinaFunctionalRetargeting2025, liuQuasiSimParameterizedQuasiPhysical2024], RL can produce dexterous, physically feasible robot motions. However, it typically requires training on each trajectory and significant computation, which limits scalability to internet-scale data and real-time deployment.

Existing methods trade off between efficiency (kinematics, neural networks) and physical fidelity (optimization, RL). SPIDER combines the generality of RL with the efficiency of optimization-based pipelines, making it a scalable and practical drop-in replacement for human-to-robot transfer.

### 5.3 Sampling-based Optimization for Robot Control

Sampling-based optimization methods such as the cross-entropy method [deboerTutorialCrossEntropyMethod2005], evolutionary algorithms [salimansEvolutionStrategiesScalable2017], and Bayesian optimization [frazierTutorialBayesianOptimization2018a] are powerful tools for solving non-convex and non-smooth problems.
Due to their parallelizability and flexibility, these methods have been applied to navigation [williamsAggressiveDrivingModel2016], legged locomotion [xueFullOrderSamplingBasedMPC2024], and dexterous manipulation [howellPredictiveSamplingRealtime2022, liDROPDexterousReorientation2024]. Despite their success in contact-rich control, they can suffer from instability and solution ambiguity in trajectory sampling [kimSmoothModelPredictive2022].
SPIDER addresses these challenges by guiding sampling with contact information, which helps preserve the intended contact sequence.

## 6 Conclusion and Future Work

This paper introduces SPIDER, a flexible and efficient physics-based retargeting framework that enables large-scale robot demonstration generation from human data.
SPIDER achieves competitive performance compared to state-of-the-art methods while being an order of magnitude faster.
Despite its generality, SPIDER’s performance depends on the quality of reconstructed 3D human-object interaction data; noisy meshes and motion can yield degraded trajectories.
As one promising application, SPIDER can be applied to behavior cloning pipeline to unlock generalizable dexterous manipulation policies.

## Acknowledgements

Chaoyi Pan thanks Mandi Zhao for her help in integrating DexMachina into the framework, and thanks Haoyang Weng for the support for the humanoid-object interaction integration from HDMI.
Guanya Shi holds concurrent appointments as an Assistant Professor at Carnegie Mellon University and as an Amazon Scholar. This paper describes work performed at Carnegie Mellon University and is not associated with Amazon.

## References

\beginappendix

## 7 Implementation Details

### 7.1 Preprocessing

We extract a 21D keypoint representation per hand using 3D fingertip positions and 6D wrist pose from MANO. An IK solver maps these keypoints to robot-specific joint positions by minimizing
$\mathcal{L}_{\text{IK}}=\sum_{i=1}^{n}\|\mathbf{p}_{i}^{\text{robot}}-\mathbf{p}_{i}^{\text{human}}\|^{2}+0.1\|\mathbf{R}_{\text{wrist}}^{\text{robot}}-\mathbf{R}_{\text{wrist}}^{\text{human}}\|_{F}^{2}$.
Trajectories are resampled at 50 Hz and low-pass filtered at 10 Hz.

### 7.2 Hyperparameters and Setup

All experiments use: horizon $H=1.2$ s, particles $N=1024$, temperatures $\beta_{1}=0.85,\beta_{2}=0.9$, iterations $M=16$, and annealing $\eta_{t}=\eta_{0}\cdot 1.1^{t}$ with $\eta_{0}=0.01$. Each experiment is repeated 5 times with different seeds.
The method supports multiple simulators: MuJoCo Warp, MJX, Genesis, and Isaac Gym.
All simulations run at 100 Hz physics and 50 Hz control in MuJoCo Warp (most datasets) and Genesis (ARCTIC dataset, as in DexMachina [mandiDexMachinaFunctionalRetargeting2025]). Ablations and speed tests use RTX 4090 GPUs; dataset generation uses H100 GPUs.

### 7.3 Retargeting for RL Policy Training

This section describes the training details for the RL policy in [section˜4.2](#S4.SS2).
We use the PPO algorithm [schulman2017proximalpolicyoptimizationalgorithms] and port implementation from [weng2025hdmilearninginteractivehumanoid].

Rewards.
We use the following rewards:

-
•

Body tracking reward: tracking humanoid pelvis, hand and legs body motion.

-
•

Object tracking reward: tracking the object motion.

-
•

Action rate penalty: penalize the action rate to avoid jittering.

We remove the contact reward, self-collision penalty as well as the auxiliary contact reward proposed in the paper to evaluate the reference motion retargeting quality.

Termination.
We terminate the training when the average body tracking error is greater than 10 cm and the average object tracking error is greater than 10 cm.

Generated on Thu Feb 5 22:15:57 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)