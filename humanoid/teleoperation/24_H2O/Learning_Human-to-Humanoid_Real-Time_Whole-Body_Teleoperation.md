HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: etoc

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: arXiv.org perpetual non-exclusive license

arXiv:2403.04436v1 [cs.RO] 07 Mar 2024

[# Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation Tairan He† Zhengyi Luo† Wenli Xiao Chong Zhang Kris Kitani Changliu Liu Guanya Shi Carnegie Mellon University †Equal Contributions https://human2humanoid.com](https://human2humanoid.com)

###### Abstract

We present Human to Humanoid (H2O), a reinforcement learning (RL) based framework that enables real-time whole-body teleoperation of a full-sized humanoid robot with only an RGB camera.
To create a large-scale retargeted motion dataset of human movements for humanoid robots, we propose a scalable “sim-to-data” process to filter and pick feasible motions using a privileged motion imitator.
Afterwards, we train a robust real-time humanoid motion imitator in simulation using these refined motions and transfer it to the real humanoid robot in a zero-shot manner.
We successfully achieve teleoperation of dynamic whole-body motions in real-world scenarios, including walking, back jumping, kicking, turning, waving, pushing, boxing, \etc.
To the best of our knowledge, this is the first demonstration to achieve learning-based real-time whole-body humanoid teleoperation.

## I Introduction

We aim to enable real-time teleoperation of a full-sized humanoid robot by a human teleoperator using an RGB camera. Humanoid robots, with their physical form closely mirroring that of humans, present an unparalleled opportunity for real-time teleoperation. This alignment of the embodiment allows for a seamless integration of human cognitive skills with versatile humanoid capabilities [[[1](#bib.bibx1)]].
Such synergy stimulated by human-to-humanoid teleoperation is crucial for complex tasks (\eg, household chores, medical assistance, high-risk rescue operations) that are yet too challenging for a fully autonomous robot, but possible for existing hardware teleoperated by humans [[[2](#bib.bibx2), [3](#bib.bibx3)]].
In this paper, we transfer human motions to humanoid behaviors in a real-time fashion using an RGB camera. This system also has the potential to enable large-scale and high-quality data collection of human operations for robotics [[[4](#bib.bibx4), [2](#bib.bibx2)]], where human-teleoperated actions can be used for imitation learning.

However, whole-body control of full-sized humanoids is a long-standing problem in robotics [[[5](#bib.bibx5)]], and complexity increases when controlling the humanoid to replicate free-form human movements in real-time [[[1](#bib.bibx1)]]. Existing work on whole-body humanoid teleoperation has achieved remarkable results via model-based controllers [[[6](#bib.bibx6), [7](#bib.bibx7), [8](#bib.bibx8), [9](#bib.bibx9)]], but they all use simplified models due to the high computational cost of modeling the full dynamics of the system [[[10](#bib.bibx10), [11](#bib.bibx11)]], which limits the scalability to dynamic motions. Furthermore, these works are highly dependent on contact measurement [[[12](#bib.bibx12), [13](#bib.bibx13)]], leading to reliance on external setups such as the exoskeleton [[[9](#bib.bibx9)]] and force sensors [[[8](#bib.bibx8), [14](#bib.bibx14)]] for teleoperation.

Recent advances in reinforcement learning (RL) for humanoid control provide a promising alternative. First, in the graphics community, RL has been used to generate complex human movements [[[15](#bib.bibx15), [16](#bib.bibx16)]], perform a variety of tasks [[[17](#bib.bibx17)]], and track real-time human motions captured by a webcam [[[18](#bib.bibx18)]] in simulation. However, due to unrealistic state-space design and partial disregard of the hardware limit (\egtorque / joint limit), it remains a question whether these methods can be applied to a full-sized humanoid.
On the other hand, RL has achieved robust and agile biped locomotion in the real world [[[19](#bib.bibx19), [20](#bib.bibx20), [21](#bib.bibx21)]].
To date, however, there has been no existing work on RL-based whole-body humanoid teleoperation.
The most closely related effort is a concurrent study [[[22](#bib.bibx22)]], which focuses on learning to replicate upper-body motions and uses root velocity tracking for the lower body, from offline human motions rather than real-time teleoperation.

In this paper, we design a complete system for humanoid teleportation in real time. First, we identify one of the primary challenges in whole-body humanoid teleoperation as the lack of a dataset with feasible motions tailored to the humanoid, which is essential for training a controller that can track diverse motions. Although direct human-to-humanoid retargeting has been explored in previous locomotion-focused efforts [[[23](#bib.bibx23), [24](#bib.bibx24), [25](#bib.bibx25)]], retargeting a large-scale human motion dataset to the humanoid presents new challenges. That is, the significant dynamics discrepancy between humans and humanoids means that some human motions could be infeasible for the humanoid (\egcartwheeling, steps wider than the leg lengths of the humanoid). In light of this, we introduce an automated “sim-to-data” process to retarget and refine a large-scale human motion dataset [[[26](#bib.bibx26)]] into motions that are feasible for real-world humanoid embodiment. Specifically, we first retarget the human motions to the humanoid via inverse kinematics, and train a humanoid controller with access to privileged state information [[[18](#bib.bibx18)]] to imitate the unfiltered motions in simulation. Afterwards, we remove the motion sequences that the privileged imitator fails to track. By doing so, we create a large-scale humanoid-compatible motion dataset.

After obtaining a dataset of feasible motions, we develop a scalable training process for the real-world motion imitator that incorporates extensive domain randomization to bridge the sim-to-real gap. To facilitate real-time teleoperation, we design a state space that prioritizes the inputs available in the real world using an RGB camera, such as the keypoint positions. During inference, we use an off-the-shelf human pose estimator [[[27](#bib.bibx27)]] to provide global human body positions for the humanoid to track.

In summary, we demonstrate the feasibility of an RL-based real-time Human-to-Humanoid (H2O) teleoperation system. Our contributions include:

-
1.

A scalable retargeting and “sim-to-data” process to obtain a large-scale motion dataset feasible for the real-world humanoid robot;

-
2.

Sim-to-real transfer of the RL-based whole-body tracking controller that scales to a large number of motions;

-
3.

A real-time teleoperation system with an RGB camera and 3D human pose estimation, demonstrating fulfillment of various whole-body motions including walking, pick-and-place, stroller pushing, boxing, hand-waving, ball kicking, etc.

## II Related Works

### II-A Physics-Based Animation of Human Motions

Physics-based simulation has been used to generate realistic and natural motions for avatars [[[28](#bib.bibx28), [15](#bib.bibx15), [29](#bib.bibx29), [16](#bib.bibx16), [30](#bib.bibx30), [31](#bib.bibx31), [17](#bib.bibx17), [32](#bib.bibx32), [18](#bib.bibx18), [33](#bib.bibx33)]]. With motion capture as the main source of human motion data [[[26](#bib.bibx26)]], RL is often used to learn avatar controllers that can mimic these motions, offering distinctive styles [[[15](#bib.bibx15), [31](#bib.bibx31)]], scalability [[[16](#bib.bibx16), [34](#bib.bibx34), [18](#bib.bibx18)]], and reusability [[[17](#bib.bibx17), [32](#bib.bibx32)]].

However, realistic animation in physics-based simulators does not guarantee real-world applicability, especially for humanoids. Simulated humanoid avatars typically have high degrees of freedom and large joint torques [[[35](#bib.bibx35)]], and sometimes need non-physical assistive external forces [[[36](#bib.bibx36)]]. In this work, we demonstrate that, with carefully designed sim-to-real training, approaches in the humanoid animation community can be applied to a real-world humanoid robot.

### II-B Transferring Human Motions to Real-World Humanoids

Before the emergence of RL-based humanoid controllers, traditional methods typically employ model-based optimization to track retargeted motions while maintaining stability [[[1](#bib.bibx1)]]. To this end, these methods minimize tracking errors under the constraints of stability and contacts, requiring predefined contact states [[[6](#bib.bibx6), [12](#bib.bibx12), [10](#bib.bibx10), [13](#bib.bibx13), [37](#bib.bibx37), [38](#bib.bibx38), [39](#bib.bibx39), [40](#bib.bibx40)]] or estimated contacts from sensors [[[41](#bib.bibx41), [14](#bib.bibx14), [42](#bib.bibx42), [7](#bib.bibx7), [9](#bib.bibx9)]], hindering large-scale deployment outside the laboratory. [[[11](#bib.bibx11), ]] use contact-implicit model predictive control (MPC) to track motions extracted from videos, but trajectories must first be optimized offline to ensure dynamic feasibility. Furthermore, the model used in MPC needs to be simplified due to computational burden [[[6](#bib.bibx6), [14](#bib.bibx14), [11](#bib.bibx11)]], which limits the capability of trackable motions.

RL-based controllers may provide an alternative that does not require explicit contact information. Some works [[[43](#bib.bibx43), [44](#bib.bibx44)]] use imitation learning to transfer human-style motions to the controller, but do not accurately track human motions. [[[22](#bib.bibx22), ]] train whole-body humanoid controllers that can replicate upper body movements from offline human motions, but the lower body relies on root velocity tracking and does not track precise lower body movements. In comparison, our work achieves real-time whole-body tracking of human motions.

### II-C Teleoperation of Humanoids

Teleoperation of humanoids can be categorized into three types: 1) task-space teleoperation [[[45](#bib.bibx45), [46](#bib.bibx46)]], 2) upper-body-retargeted teleoperation [[[47](#bib.bibx47), [48](#bib.bibx48)]], and 3) whole-body teleoperation [[[6](#bib.bibx6), [13](#bib.bibx13), [42](#bib.bibx42), [49](#bib.bibx49), [7](#bib.bibx7)]].
For the first and second types, the shared morphology between humans and humanoids is not fully utilized, and whole-body control must be solved in a task-specified way. This also raises the concern that, if tracking lower body movement is not necessary, the robot could opt for designs with better stability, such as a quadruped [[[50](#bib.bibx50)]] or wheeled configuration [[[51](#bib.bibx51)]].

Our work belongs to the third type and is the first to achieve learning-based whole-body teleoperation.
Moreover, our approach does not require capture markers or force sensors on the human teleoperator, as we directly employ an RGB camera to capture human motions for tracking, potentially paving the way for collecting large-scale humanoid data for training autonomous agents.

## III Preliminaries

The whole-body real-time humanoid teleoperation we are tackling is formulated as a goal-conditioned RL problem, which tracks versatile human motion with a single RL control policy. In [Section III-A](#S3.SS1), we set up the preliminary for our control framework. [Section III-B](#S3.SS2) describes the human model and dataset we use in the RL policy training. As a notation convention, we use $\widetilde{\cdot}$ to represent kinematic quantities (without physics simulation) from pose estimator/keypoint detectors, $\widehat{\cdot}$ to denote ground truth quantities from Motion Capture (MoCap), and normal symbols without accents for values from the physics simulation.

### III-A Goal-conditioned RL for Humanoid Control

We formulate our problem as goal-conditioned RL where $\pi$ is trained to track real-time human motions. We formulate the learning task as a Markov Decision Process (MDP) defined by the tuple $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\rangle$ of state ${\bm{s}_{t}}\in\mathcal{S}$, action ${\bm{a}_{t}}\in\mathcal{A}$, transition dynamics $\mathcal{T}$, reward function $\mathcal{R}$, and discount factor $\gamma$. The state ${\bm{s}_{t}}$ contains the proprioception ${\bm{s}^{\text{p}}_{t}}$ and the goal state ${\bm{s}^{\text{g}}_{t}}$. The goal state ${\bm{s}^{\text{g}}_{t}}$ is a unified representation of the whole-body motion of the human teleoperator, which we will discuss in detail in [Section V-A](#S5.SS1). Based on proprioception ${\bm{s}^{\text{p}}_{t}}$ and goal state ${\bm{s}^{\text{g}}_{t}}$, we define the reward $r_{t}=\mathcal{R}\left({\bm{s}^{\text{p}}_{t}},{\bm{s}^{\text{g}}_{t}}\right)$ for the policy training. The action ${\bm{a}_{t}}\in\mathbb{R}^{19}$ specifies the joint target positions that the PD controller will use to actuate the degrees of freedom. We apply the proximal policy gradient (PPO) [[[52](#bib.bibx52)]] to maximize the cumulative discounted reward $\mathbb{E}\left[\sum_{t=1}^{T}\gamma^{t-1}r_{t}\right]$. We formulate the teleoperation task as the motion imitation/tracking/mimicking task, where we train the humanoid to track the reference motion at every frame.

### III-B Parametric Human Model and Human Motion Dataset

Popular in the vision and graphics community, parametric human models such as SMPL [[[53](#bib.bibx53)]] are easy to work with representations of human shapes and motions. SMPL represents the human body as body shape parameters ${\bm{{\beta}}}\in\mathcal{R}^{10}$, pose parameters ${\bm{{\theta}}}\in\mathcal{R}^{24\times 3}$, and root translation ${\bm{p}}\in\mathcal{R}^{24\times 3}$. Given ${\bm{{\beta}}}$, ${\bm{{\theta}}}$ and ${\bm{p}}$, $\mathcal{S}$ denotes the SMPL function, where $\mathcal{S}({\bm{{\beta}}},{\bm{{\theta}}},{\bm{p}}):{\bm{{\beta}}},{\bm{{%
\theta}}},{\bm{p}}\rightarrow\mathcal{R}^{6890\times 3}$ maps the parameters to the position of the vertices of a triangular human mesh of 6890 vertices. The AMASS [[[26](#bib.bibx26)]] dataset contains 40 hours of motion capture expressed in the SMPL parameters.

*Figure 1: Fitting the SMPL body to the H1 humanoid. (a) Visualization of the humanoid keypoints (red dots) (b) Humanoid keypoints vs SMPL keypoints (green dots and mesh) before and after fitted SMPL shape ${\bm{{\beta}}}^{\prime}$. (c) Corresponding 12 joint position before and after fitting.*

![Figure](extracted/5455183/fig/H2O-shape.jpeg)

*Figure 2: Effect of using a fitted SMPL shape ${\bm{{\beta}}}^{\prime}$ instead of mean body shape on position-based retargeting. (a) Retargting without using ${\bm{{\beta}}}^{\prime}$, which results in unstable “in-toed” humanoid motion. (b) Retargeting using ${\bm{{\beta}}}^{\prime}$, which result in balanced humanoid motion.*

## IV Retargeting Human Motions for Humanoid

To enable humanoid motion imitation for unscripted human motion, we require a large amount of whole-body motion to train a robust motion imitation policy. Since humans and humanoids also have a nontrivial difference in body structure, shape, and dynamics, naively retargeted motion from a human motion dataset can result in a large number of motions impossible for our humanoid to perform. These imfeasible motion sequences can hinder imitation training as observed in prior work [[[32](#bib.bibx32)]]. To resolve these issues, we design a “sim-to-data” approach to complement traditional retargeting to convert a large-scale human motion dataset to feasible motions for humanoids.

![Figure](x1.png)

*Figure 3: Overview of H2O: (a) Retargeting ([Section IV](#S4)): H2O first aligns the SMPL body model to a humanoid’s structure by optimizing shape parameters. Then H2O retargets and removes the infeasible motions using a trained privileged imitation policy, producing a clean motion dataset. (b) Sim-to-Real Training: ([Section V](#S5)) An imitation policy is trained to track motion goals sampled from a cleaned dataset. (c) Real-time Teleoperation Deployment ([Section VI-B](#S6.SS2)): The real-time teleoperation deployment captures human motion through an RGB camera and a pose estimator, which is then mimicked by a humanoid robot using the trained sim-to-real imitation policy.*

### IV-A Motion Retargeting

As there is a non-trivial difference between the SMPL kinematic structure and the humanoid kinematic tree, we perform a two-step process for the initial retargeting. First, since the SMPL body model can represent different body proportions, we first find a body shape ${\bm{{\beta}}}^{\prime}$ closest to the humanoid structure. We choose 12 joints that have a correspondence between humans and humanoids, as shown in Fig.[1](#S3.F1) and perform gradient descents on the shape parameter $\bm{s}$ to minimize the joint distances using a common rest pose. After finding the optimal ${\bm{{\beta}}}^{\prime}$, given a sequence of motions expressed in SMPL parameters, we use the original translation ${\bm{p}}$ and pose ${\bm{{\theta}}}$, but the fitted shape ${\bm{{\beta}}}^{\prime}$ to obtain the set of body keypoint positions. Then we retarget motion from human to humanoid by minimizing the 12 joint position differences using Adam optimizer [[[54](#bib.bibx54)]]. Notice that our retargeting process try to match the end effectors of the human to the humanoid (\egankles, elbows, wrists) to preserve the overall motion pattern. Another approach is direct copying the local joint angles from human to humanoid, but that approach can lead to large differences in end-effector positions due to the large difference in kinematic trees. During this process, we also add some heuristic-based filtering to remove unsafe sequences, such as sitting on the ground. The motivation to find ${\bm{{\beta}}}^{\prime}$ before retargeting is that in the rest pose, our humanoid has a large gap between its feet. If naively trying to match the foot movement between the human and the humanoid, the humanoid motion can have an in-toed artifact. Using ${\bm{{\beta}}}^{\prime}$, we can find a human body structure has a large gap between its rest pose (as shown in Fig.[1](#S3.F1)). Using ${\bm{{\beta}}}^{\prime}$ during fitting can effectively create motion that is more feasible for the humanoid, as shown in Fig.[2](#S3.F2). From the AMASS dataset $\bm{\hat{Q}}$ that contains 13k motion sequences, this process computes 10k retargted motion sequences $\bm{\hat{Q}}^{\text{retarget}}$.

### IV-B Simulation-based Data Cleaning

As shown in [Figure 3](#S4.F3)$,\bm{\hat{Q}}^{\text{retarget}}$ contain a large number of implausible motions for the humanoid due to the significant gap between the capabilities of a human and a motor-actuated humanoid. Manually finding these data sequences from a large-scale dataset can be a rather cumbersome process. Thus, we propose a “sim-to-data” procedure, where we train a motion imitator ${\pi_{\text{privileged}}}$ (similar to PHC [[[18](#bib.bibx18)]]) with access to privileged information and no domain randomization to imitate all uncleaned data $\bm{\hat{Q}}^{\text{retarget}}$. Without domain randomization, ${\pi_{\text{privileged}}}$ can perform well in motion imitation, but is not suitable for transfer to the real humanoid. However, ${\pi_{\text{privileged}}}$ represents the upper bound of motion imitation performance, and sequences which ${\pi_{\text{privileged}}}$ fails to imitate represent implausible ones. Specifically, we train ${\pi_{\text{privileged}}}$ following the same state space, control parameters, and hard-negative mining procedure proposed in PULSE [[[32](#bib.bibx32)]], and train a single imitation policy to imitate the entire retargeted dataset. After training, $\sim$8.5k out of 10k motion sequences from AMASS turn out to be plausible for the H1 humanoid, and we denote the obtained clean dataset as $\bm{\hat{Q}}^{\text{clean}}$.

#### Privileged Motion Imitation Policy

To train ${\pi_{\text{privileged}}}$, we follow PULSE [[[32](#bib.bibx32)]] and train a motion imitator with access to the full rigid body state of the humanoid. Specifically, for the privileged policy ${\pi_{\text{privileged}}}$, its proprioception is defined as ${\bm{s}^{\text{p-privileged}}_{t}}\triangleq[{\bm{{p}}_{t}},{\bm{{\theta}}_{t}%
},{\bm{v}_{t}},{\bm{{\omega}}_{t}}]$, which contains the global 3D rigid body position ${\bm{{p}}_{t}}$, orientation ${\bm{{\theta}}_{t}}$, linear velocity ${\bm{v}_{t}}$, and angular velocity ${\bm{{\omega}}_{t}}$ of all rigid bodies in the humanoid. The goal state is defined as ${\bm{s}^{\text{g-privileged}}_{t}}\triangleq[{\bm{\hat{\theta}}_{t+1}}\ominus{%
\bm{{\theta}}_{t}},{\bm{\hat{p}}_{t+1}}-{\bm{{p}}_{t}},\bm{\hat{v}}_{t+1}-\bm{%
v}_{t},\bm{\hat{\omega}}_{t}-\bm{\omega}_{t},{\bm{\hat{\theta}}_{t+1}},{\bm{%
\hat{p}}_{t+1}}]$, which contains the one-frame difference between the reference and current simulation result for all rigid bodies on the humanoid. It also contains the next frame’s reference rigid body orientation and position. All values are normalized to the humanoid’s coordinate system. Notice that all values are global, and values such as global rigidbody linear velocity ${\bm{v}_{t}}$ and angular velocity ${\bm{{\omega}}_{t}}$ are hard to obtain accurately in the real world.

## V Whole-Body Teleoperation Policy Training

### V-A State Space

To achieve real-time teleoperation of humanoid robots, the state space of RL policy must contain only quantities available in the real world. This differs from the simulation-only approaches, where all the physics information (\eg, foot contact force) is available. For example, in the real world, we have no access to each joint’s precise global angular velocity due to the lack of IMUs, but the privileged policy ${\pi_{\text{privileged}}}$ requires them.

In our state space design, proprioception is defined by ${\bm{s}^{\text{p}}_{t}}\triangleq\left[{\bm{{q}}_{t}},{\bm{\dot{q}}_{t}},{\bm{%
{v}}_{t}},{\bm{\omega}_{t}},{\bm{g}_{t}},{\bm{a}_{t-1}}\right]$, with joint position ${\bm{{q}}_{t}}\in\mathbb{R}^{19}$ (DoF position), joint velocity ${\bm{\dot{q}}_{t}}\in\mathbb{R}^{19}$ (DoF velocity), root linear velocity ${\bm{{v}}_{t}}\in\mathbb{R}^{3}$, root angular velocity ${\bm{\omega}_{t}}\in\mathbb{R}^{3}$, root projected gravity ${\bm{g}_{t}}\in\mathbb{R}^{3}$, and last action ${\bm{a}_{t-1}}\in\mathbb{R}^{19}$. The goal state is ${\bm{s}^{\text{g}}_{t}}\triangleq[{\bm{\hat{p}}_{t}}^{\text{kp}},{\bm{\hat{p}}%
_{t}}^{\text{kp}}-{\bm{{p}}_{t}}^{\text{kp}},{\bm{\hat{\dot{p}}}_{t}}^{\text{%
kp}}]$. ${\bm{\hat{p}}_{t}}^{\text{kp}}\in\mathbb{R}^{8\times 3}$ is the position of eight selected reference body positions (shoulders, elbows, hands, ankles); ${\bm{\hat{p}}_{t}}^{\text{kp}}-{\bm{{p}}_{t}}^{\text{kp}}$ is the position difference between the reference joints and humanoid’s own joints; ${\bm{\hat{\dot{p}}}_{t}}^{\text{kp}}$ is the linear velocity of the reference joints. All values are normalized to the humanoid’s own coordinate system.
As a comparison, we also consider a reduced goal state ${\bm{s}^{\text{g-reduced}}_{t}}\triangleq({\bm{\hat{p}}_{t}}^{\text{kp}})$, where only the reference position ${\bm{\hat{p}}_{t}}^{\text{kp}}$ but not the position difference. The action space of the agile policy consists of 19-dim joint targets. A PD controller tracks these joint targets by converting them to joint torques: $\tau=K_{p}({\bm{a}_{t}}-{\bm{{q}}_{t}})-K_{d}{\bm{\dot{q}}_{t}}$.

### V-B Reward Design

We formulate the reward function $r_{t}$ with the summation of three terms: 1) penalty; 2) regularization; and 3) task rewards, which are summarized in detail in [Table I](#S5.T1).
Note that while we only have eight selected body positions ${\bm{\hat{p}}_{t}}^{\text{kp}}$ in our state space, we provide six full-body reward terms for all joints (DoF position, DoF velocity, body position, body rotation, body velocity, body angular velocity) for the imitation task. These expressive rewards give more dense reward signals for efficient RL training.

*TABLE I: Reward components and weights: penalty rewards for preventing undesired behaviors for sim-to-real transfer, regularization to refine motion, and task reward to achieve successful whole-body tracking in real-time.*

### V-C Domain Randomization

*TABLE II: The range of dynamics randomization. Describing simulated dynamics randomization, external perturbation, and randomized terrain, which are important for sim-to-real transfer and boost robustness and generalizability.*

Domain randomization has been shown to be the key source of robustness and generalization to achieve successful sim-to-real transfers [[[58](#bib.bibx58), [19](#bib.bibx19)]]. All the domain randomization we use in H2O are listed in [Table II](#S5.T2), including ground friction coefficient, link mass, Centor-of-Mass (CoM) position of the torso link, PD gains of the PD controller, torque noise on the actually applied torques on each joint, control delay, terrain types.
The link mass and PD gains are independently randomized for each link and joint, and the rest are episodic randomized.
These domain randomization together can effectively facilitate the sim-to-real transfer for the real-world dynamics and hardware gaps.

### V-D Early Termination Conditions

We introduce three early termination conditions to make the RL training process more sample-efficient: 1) low height: the base height is lower than 0.3m; 2) orientation: the projected gravity on x or y axis exceeds 0.7; 3) teleoperation tolerance: the average link distance between the robot and reference motions is further than 0.5m.

## VI Experimental Results

*TABLE III: Quantitative motion imtiation results the uncleaned retargeted AMASS dataset $\bm{\hat{Q}}^{\text{retarget}}$.*

### VI-A Simulation Experiments

#### Baselines

To reveal the effect of different retargeting, state space designs, and sim-to-real training techniques on whole-body teleoperation performance, we consider four baselines:

-
1.

Privileged policy ${\pi_{\text{privileged}}}$: The privileged policy (trained without any sim-to-real regularizations or domain randomizations) is used to filter the dataset to find infeasible motion. It has no sim-to-real capability and has a much higher input dimension.

-
2.

H2O-w/o-sim2data: H2O without the “sim-to-data” retargeting, trained on the $\bm{\hat{Q}}^{\text{retarget}}$;

-
3.

H2O-reduced: H2O with a state space of goal state consisting only of selected body positions ${\bm{s}^{\text{g-reduced}}_{t}}$.

-
4.

H2O: Our full H2O system, with all the retargeting process introduced in [Section IV](#S4) and the state space design introduced in [Section V-A](#S5.SS1), trained on $\bm{\hat{Q}}^{\text{clean}}$;

#### Metrics

We evaluate these baselines in simulation on the uncleaned retargeted AMASS dataset (10k sequences $\bm{\hat{Q}}^{\text{retarget}}$). The metrics are as follows:

-
1.

Success rate: the success rate (Succ) as in PHC [[[18](#bib.bibx18)]], deeming imitation unsuccessful when, at any point during imitation, the average difference in body distance is on average further than 0.5m. Succ measures whether the humanoid can track the reference motion without losing balance or significantly lag behind.

-
2.

$E_{\text{mpjpe}}$ and $E_{g-\text{mpjpe}}$: the global MPJPE $E_{g-\text{mpjpe}}$ and the root-relative mean per-joint position error (MPJPE) $E_{\text{mpjpe}}$ (in mm), measuring our imitator’s ability to imitate the reference motion both globally and locally (root-relative).

-
3.

$E_{\text{acc}}$ and $E_{\text{vel}}$: To show physical realism, we also compare acceleration $E_{\text{acc}}$ $\text{(mm/frame}^{2}$ and velocity $E_{\text{vel}}$ (mm/frame) difference.

#### Results

The experimental results are summarized in [Table III](#S6.T3), where H2O significantly outperforms H2O-w/o-sim2data and H2O-reduced by a large margin, demonstrating the importance of the “sim-to-data” process and the state-space design of motion goals for RL. Note that the privileged policy and H2O-w/o-sim2data are trained on the entire retargeted AMASS dataset $\bm{\hat{Q}}^{\text{retarget}}$ while H2O and H2O-reduced are trained on the filtered dataset $\bm{\hat{Q}}^{\text{clean}}$.
The success rate gap between H2O and the privileged policy comes from two factors: 1) H2O uses a much more practical and less informative observation space compared to the privileged policy; 2) H2O is trained with all sim-to-real regularizations and domain randomization. These two factors will both lead to degradation in simulation performance. This shows that while the RL-based avatar control frameworks have achieved impressive results in simulation, transferring them to the real world requires more robustness and stability. With the carefully chosen dataset and the state space, we could make H2O achieve a higher success rate compared to H2O-w/o-sim2data and H2O-reduced.
By comparing H2O with H2O-w/o-sim2data, we can see that our “sim-to-data” process is effective in obtaining higher success rate, even when the RL policy is trained on less data. Intuitively, an implausible motion may cause the policy to waste resources trying to achieve them, and filtering them out can lead to better overall performance, as also observed in PULSE [[[32](#bib.bibx32)]]. Comparing H2O with H2O-reduced, the only difference is the design of the state space of the goal, which indicates that including more informative physical information about motions helps RL to generalize to large-scale motion imitation.

#### Ablation on Motion Dataset Size

To show how motion tracking performance scales with the size of the motion dataset, we test H2O with different size of $\bm{\hat{Q}}^{\text{clean}}$ by randomly selecting $1\%$, $10\%$ of $\bm{\hat{Q}}^{\text{clean}}$. The results are summarized in [Table IV](#S6.T4), where policies trained larger motion datasets continue to improve the tracking performance. Notice that a policy trained on only 0.1% of the data can achieve a surprisingly high success rate, most likely due to the ample domain randomization applied to the humanoid, such as push robot significantly widens the state the humanoid has encountered, improving its generalization capability.

*TABLE IV: Quantitative results of H2O on different sizes of motion dataset for training, evaluated on the uncleaned retargeted AMASS dataset $\bm{\hat{Q}}^{\text{retarget}}$.*

### VI-B Real-world Demonstrations

#### Deployment Details

For real-world deployment tests, we use a standard 1080P webcam as the RGB camera, and use HybrIK [[[27](#bib.bibx27)]] as the 3D human pose estimator running at 30Hz. For the linear velocity estimation of the robot, we leverage the motion capture system (50Hz), and all the other proprioception is obtained from built-in sensors (200Hz) of Unitree H1 humanoid. Linear velocity state estimation could be replaced by onboard visual/LiDAR odometry methods, though we opt in to MoCap for this work due to its simplicity.

#### Real-world Teleoperation Results

For real-time teleoperation, the 3D pose estimation from the RGB camera is noisy and can suffer from perspective bias, but our H2O policy shows a strong generalization ability to real-world estimated motion goals in real-time. The real-world teleoperation is shown in LABEL:fig:firstpage, [Figure 4](#S6.F4) and [Figure 5](#S6.F5), where H2O enables precise real-time teleoperation of humanoids to do whole-body dynamic motions like ball kicking, walking, and back jumping. More demonstrations can be found on our [website](https://human2humanoid.com).

![Figure](x2.png)

*Figure 4: The humanoid robot is able to track the precise lower-body movements of the human teleoperator.*

![Figure](x3.png)

*Figure 5: The humanoid robot is able to track walking motions of human-style pace and imitate continuous back jumping.*

#### Robustness

Our H2O system can keep balance under external force disturbances, as shown in [Figure 6](#S6.F6). These tests demonstrate the robustness of our system.

![Figure](x4.png)

*Figure 6: Robustness Tests of our H2O system under powerful kicking. The policy is able to maintain balance for both stable and dynamic teleoperated motions.*

## VII Discussions, Limitations, and Future Work

#### Towards Universal Humanoid Teleoperation

Our ultimate goal is to enable the humanoid to follow as many human-demonstrated motions as possible. We emphasize three key factors that can be improved in the future.
1) Closing the representation gap: as shown in [Section VI-A](#S6.SS1), the state representation of the motion goals critically affects the scalability of RL training with more diverse motions, leading to a trade-off. While incorporating more expressive motion representations into the state space can accommodate finer-grained and more diverse motions, the expanded dimensionality will lead to a curse of sample efficiency in scalable RL.
2) Closing the embodiment gap: as evident in [Section VI-A](#S6.SS1) and prior work [[[32](#bib.bibx32)]], training on infeasible or damaged motions might largely harm performance. The feasibility of motions varies from robot to robot due to hardware constraints, and we lack systematic algorithms to identify feasible motions. We need more efforts to close this embodiment gap: on one end, more human-like humanoids would help; on the other, more teleoperation research is expected to improve the learnability of human motions.
3) Closing the sim-to-real gap: to achieve a successful sim-to-real transfer, regularization (e.g., reward regularization) and domain randomization are needed. However, over-regularization and over-randomization will also hinder the policy from learning the motions. It remains unknown how to strike the best trade-off between motion imitation leaning and sim-to-real transfer into a universal humanoid control policy.

#### Towards Real-time Humanoid Teleoperation

In this work, we leverage RGB and 3D pose estimator to transform the motions of human teleoperators into humanoid robots. The latency and error from RGB cameras and pose estimation also lead to an inevitable trade-off between efficiency and precision in teleoperation. Also, in this work, the human teleoperator receives feedback from the humanoid only in the form of visual perception.
More research is needed on human-robot interaction to study this emerging multimodal interaction (e.g., force feedback [[[59](#bib.bibx59)]], verbal and conversational feedback [[[60](#bib.bibx60)]]), which could further enhance the capability of humanoid teleoperation.

#### Towards Whole-body Humanoid Teleoperation

One may wonder if lower-body tracking is necessary, as the major embodiment gap between humans and humanoids is the lower-body capability.
A large proportion of skillful motions of humans (e.g., sports, dancing) need diverse agile lower-body movements.
We emphasize the scenarios where legged robots hold an advantage over wheeled robots, in which lower-body tracking is necessary to follow human lower-body movements, including stepping stones, kicking, spread legs, etc.
In the future, a teleoperated humanoid system that learns to switch between robust locomotion and skillful lower-body tracking would be a promising research direction.

## VIII Conclusions

In this study, we introduced Human to Humanoid (H2O), a scalable learning-based framework that enables real-time whole-body humanoid robot teleoperation using just an RGB camera. Our approach, leveraging reinforcement learning and a novel “sim-to-data” process, addresses the complex challenge of translating human motion into actions a humanoid robot can perform. Through comprehensive simulation and real-world tests, H2O demonstrated its capability to perform a wide range of dynamic tasks with high fidelity and minimal hardware requirements.

## ACKNOWLEDGMENT

The authors express their gratitude to Jessica Hodgins for providing assistance in conducting hardware experiments. Special thanks are extended to Ziqiao Ma, Zhongyu Li, Yiyu Chen, Xuxin Cheng, and Unitree for their valuable help on graphics design and hardware debugging. Furthermore, we acknowledge the significance of CMU Wean Hall room 1334, formerly utilized as the recording location for the CMU MoCap dataset. In the present study, this dataset is used for real-world humanoid teleoperation within the same room.

## References

-
[1]
Kourosh Darvish et al.

“Teleoperation of humanoid robots: A survey”

In IEEE Transactions on Robotics*

IEEE, 2023

-
[2]
Zipeng Fu, Tony Z Zhao and Chelsea Finn

“Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation”

In *arXiv preprint arXiv:2401.02117*, 2024

-
[3]
Cheng Chi et al.

“Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots”

In *arXiv preprint arXiv:2402.10329*, 2024

-
[4]
Tony Z Zhao, Vikash Kumar, Sergey Levine and Chelsea Finn

“Learning fine-grained bimanual manipulation with low-cost hardware”

In *arXiv preprint arXiv:2304.13705*, 2023

-
[5]
Dana Kulić et al.

“Anthropomorphic movement analysis and synthesis: A survey of methods and applications”

In *IEEE Transactions on Robotics* 32.4

IEEE, 2016, pp. 776–795

-
[6]
Francisco-Javier Montecillo-Puente, Manish Sreenivasa and Jean-Paul Laumond

“On real-time whole-body human to humanoid motion transfer”, 2010

-
[7]
Yasuhiro Ishiguro et al.

“High speed whole body dynamic motion experiment with real time master-slave humanoid robot system”

In *2018 IEEE International Conference on Robotics and Automation (ICRA)*, 2018, pp. 5835–5841

IEEE

-
[8]
Joao Ramos and Sangbae Kim

“Humanoid dynamic synchronization through whole-body bilateral feedback teleoperation”

In *IEEE Transactions on Robotics* 34.4

IEEE, 2018, pp. 953–965

-
[9]
Yasuhiro Ishiguro et al.

“Bilateral humanoid teleoperation system using whole-body exoskeleton cockpit TABLIS”

In *IEEE Robotics and Automation Letters* 5.4

IEEE, 2020, pp. 6419–6426

-
[10]
Katsu Yamane, Stuart O Anderson and Jessica K Hodgins

“Controlling humanoid robots with human motion data: Experimental validation”

In *2010 10th IEEE-RAS International Conference on Humanoid Robots*, 2010, pp. 504–510

IEEE

-
[11]
John Z Zhang et al.

“Slomo: A general system for legged robot motion imitation from casual videos”

In *IEEE Robotics and Automation Letters*

IEEE, 2023

-
[12]
Alessandro Di Fava et al.

“Multi-contact motion retargeting from human to humanoid robot”

In *2016 IEEE-RAS 16th international conference on humanoid robots (humanoids)*, 2016, pp. 1081–1086

IEEE

-
[13]
Kazuya Otani and Karim Bouyarmane

“Adaptive whole-body manipulation in human-to-humanoid multi-contact motion retargeting”

In *2017 IEEE-RAS 17th International Conference on Humanoid Robotics (Humanoids)*, 2017, pp. 446–453

IEEE

-
[14]
Joao Ramos and Sangbae Kim

“Dynamic locomotion synchronization of bipedal robot and human operator via bilateral feedback teleoperation”

In *Science Robotics* 4.35

American Association for the Advancement of Science, 2019, pp. eaav4282

-
[15]
Xue Bin Peng, Pieter Abbeel, Sergey Levine and Michiel Van de Panne

“Deepmimic: Example-guided deep reinforcement learning of physics-based character skills”

In *ACM Transactions On Graphics (TOG)* 37.4

ACM New York, NY, USA, 2018, pp. 1–14

-
[16]
Jungdam Won, Deepak Gopinath and Jessica Hodgins

“A scalable approach to control diverse behaviors for physically simulated characters”

In *ACM Transactions on Graphics (TOG)* 39.4

ACM New York, NY, USA, 2020, pp. 33–1

-
[17]
Xue Bin Peng et al.

“Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters”

In *ACM Transactions On Graphics (TOG)* 41.4

ACM New York, NY, USA, 2022, pp. 1–17

-
[18]
Zhengyi Luo et al.

“Perpetual Humanoid Control for Real-time Simulated Avatars”

In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 10895–10904

-
[19]
Zhongyu Li et al.

“Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control”

In *arXiv preprint arXiv:2401.16889*, 2024

-
[20]
Ilija Radosavovic et al.

“Learning Humanoid Locomotion with Transformers”

In *arXiv preprint arXiv:2303.03381*, 2023

-
[21]
Jonah Siekmann et al.

“Blind bipedal stair traversal via sim-to-real reinforcement learning”

In *arXiv preprint arXiv:2105.08328*, 2021

-
[22]
Xuxin Cheng et al.

“Expressive Whole-Body Control for Humanoid Robots”

In *arXiv preprint arXiv:2402.16796*, 2024

-
[23]
Kourosh Darvish et al.

“Whole-body geometric retargeting for humanoid robots”

In *2019 IEEE-RAS 19th International Conference on Humanoid Robots (Humanoids)*, 2019, pp. 679–686

IEEE

-
[24]
Rafael Cisneros-Limón et al.

“A cybernetic avatar system to embody human telepresence for connectivity, exploration, and skill transfer”

In *International Journal of Social Robotics*

Springer, 2024, pp. 1–28

-
[25]
Ilija Radosavovic et al.

“Humanoid Locomotion as Next Token Prediction”

In *arXiv preprint arXiv:2402.19469*, 2024

-
[26]
Naureen Mahmood et al.

“AMASS: Archive of motion capture as surface shapes”

In *Proceedings of the IEEE/CVF international conference on computer vision*, 2019, pp. 5442–5451

-
[27]
Jiefeng Li et al.

“Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation”

In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2021, pp. 3383–3393

-
[28]
Xue Bin Peng, Glen Berseth, KangKang Yin and Michiel Van De Panne

“Deeploco: Dynamic locomotion skills using hierarchical deep reinforcement learning”

In *ACM Transactions on Graphics (TOG)* 36.4

ACM New York, NY, USA, 2017, pp. 1–13

-
[29]
Tingwu Wang, Yunrong Guo, Maria Shugrina and Sanja Fidler

“Unicon: Universal neural controller for physics-based character motion”

In *arXiv preprint arXiv:2011.15119*, 2020

-
[30]
Levi Fussell, Kevin Bergamin and Daniel Holden

“Supertrack: Motion tracking for physically simulated characters using supervised learning”

In *ACM Transactions on Graphics (TOG)* 40.6

ACM New York, NY, USA, 2021, pp. 1–13

-
[31]
Xue Bin Peng et al.

“Amp: Adversarial motion priors for stylized physics-based character control”

In *ACM Transactions on Graphics (ToG)* 40.4

ACM New York, NY, USA, 2021, pp. 1–20

-
[32]
Zhengyi Luo et al.

“Universal Humanoid Motion Representations for Physics-Based Control”

In *The Twelfth International Conference on Learning Representations*, 2024

-
[33]
Alexander Winkler, Jungdam Won and Yuting Ye

“QuestSim: Human motion tracking from sparse sensors with simulated avatars”

In *SIGGRAPH Asia 2022 Conference Papers*, 2022, pp. 1–8

-
[34]
Zhengyi Luo, Ryo Hachiuma, Ye Yuan and Kris Kitani

“Dynamics-regulated kinematic policy for egocentric pose estimation”

In *Advances in Neural Information Processing Systems* 34, 2021, pp. 25019–25032

-
[35]
Bullet Physics

“humanoid urdf in bullet3” Accessed: 2024-03-01, 2023

-
[36]
Ye Yuan and Kris Kitani

“Residual force control for agile human behavior imitation and extended motion synthesis”

In *Advances in Neural Information Processing Systems* 33, 2020, pp. 21763–21774

-
[37]
Luigi Penco et al.

“Robust real-time whole-body motion retargeting from human to humanoid”

In *2018 IEEE-RAS 18th International Conference on Humanoid Robots (Humanoids)*, 2018, pp. 425–432

IEEE

-
[38]
Jonas Koenemann, Felix Burget and Maren Bennewitz

“Real-time imitation of human whole-body motions by humanoids”

In *2014 IEEE International Conference on Robotics and Automation (ICRA)*, 2014, pp. 2806–2812

IEEE

-
[39]
Oscar E Ramos et al.

“Dancing humanoid robots: Systematic use of osid to compute dynamically consistent movements following a motion capture pattern”

In *IEEE Robotics & Automation Magazine* 22.4

IEEE, 2015, pp. 16–26

-
[40]
Luigi Penco et al.

“Mixed Reality Teleoperation Assistance for Direct Control of Humanoids”

In *IEEE Robotics and Automation Letters*

IEEE, 2024

-
[41]
Ko Ayusawa and Eiichi Yoshida

“Motion retargeting for humanoid robots based on simultaneous morphing parameter identification and motion optimization”

In *IEEE Transactions on Robotics* 33.6

IEEE, 2017, pp. 1343–1357

-
[42]
Kai Hu, Christian Ott and Dongheui Lee

“Online human walking imitation in task and joint space based on quadratic programming”

In *2014 IEEE International Conference on Robotics and Automation (ICRA)*, 2014, pp. 3458–3464

IEEE

-
[43]
Steven Bohez et al.

“Imitate and repurpose: Learning reusable robot movement skills from human and animal behaviors”

In *arXiv preprint arXiv:2203.17138*, 2022

-
[44]
Annan Tang et al.

“Humanmimic: Learning natural locomotion and transitions for humanoid robot via wasserstein adversarial imitation”

In *arXiv preprint arXiv:2309.14225*, 2023

-
[45]
Mingyo Seo et al.

“Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation”

In *2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids)*, 2023, pp. 1–8

IEEE

-
[46]
Stefano Dafarra et al.

“iCub3 avatar system: Enabling remote fully immersive embodiment of humanoid robots”

In *Science Robotics* 9.86

American Association for the Advancement of Science, 2024, pp. eadh3834

-
[47]
Jean Chagas Vaz, Dylan Wallace and Paul Y Oh

“Humanoid loco-manipulation of pushed carts utilizing virtual reality teleoperation”

In *ASME International Mechanical Engineering Congress and Exposition* 85628, 2021, pp. V07BT07A027

American Society of Mechanical Engineers

-
[48]
Mohamed Elobaid et al.

“Telexistence and teleoperation for walking humanoid robots”

In *Intelligent Systems and Applications: Proceedings of the 2019 Intelligent Systems Conference (IntelliSys) Volume 2*, 2020, pp. 1106–1121

Springer

-
[49]
Susumu Tachi, Yasuyuki Inoue and Fumihiro Kato

“Telesar vi: Telexistence surrogate anthropomorphic robot vi”

In *International Journal of Humanoid Robotics* 17.05

World Scientific, 2020, pp. 2050019

-
[50]
C Dario Bellicoso et al.

“Alma-articulated locomotion and manipulation for a torque-controllable robot”

In *2019 International conference on robotics and automation (ICRA)*, 2019, pp. 8477–8483

IEEE

-
[51]
Christian Lenz et al.

“Nimbro wins ana avatar xprize immersive telepresence competition: Human-centric evaluation and lessons learned”

In *International Journal of Social Robotics*

Springer, 2023, pp. 1–25

-
[52]
John Schulman et al.

“Proximal Policy Optimization Algorithms”

In *CoRR* abs/1707.06347, 2017

arXiv: [http://arxiv.org/abs/1707.06347](http://arxiv.org/abs/1707.06347)

-
[53]
Matthew Loper et al.

“SMPL: A Skinned Multi-Person Linear Model”

In *ACM Trans. Graphics (Proc. SIGGRAPH Asia)* 34.6

ACM, 2015, pp. 248:1–248:16

-
[54]
Diederik P Kingma and Jimmy Ba

“Adam: A method for stochastic optimization”

In *arXiv preprint arXiv:1412.6980*, 2014

-
[55]
Nikita Rudin, David Hoeller, Philipp Reist and Marco Hutter

“Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning”, 2022

arXiv:[2109.11978 [cs.RO]](https://arxiv.org/abs/2109.11978)

-
[56]
Luigi Campanaro, Siddhant Gangapurwala, Wolfgang Merkt and Ioannis Havoutis

“Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization”, 2023

arXiv:[2209.12878 [cs.RO]](https://arxiv.org/abs/2209.12878)

-
[57]
Tairan He et al.

“Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion”, 2024

arXiv:[2401.17583 [cs.RO]](https://arxiv.org/abs/2401.17583)

-
[58]
Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba and Pieter Abbeel

“Sim-to-real transfer of robotic control with dynamics randomization”

In *2018 IEEE international conference on robotics and automation (ICRA)*, 2018, pp. 3803–3810

IEEE

-
[59]
Fengyu Yang et al.

“Touch and go: Learning from human-collected vision and touch”

In *arXiv preprint arXiv:2211.12498*, 2022

-
[60]
Joyce Y Chai et al.

“Language to Action: Towards Interactive Task Learning with Physical Agents.”

In *IJCAI*, 2018, pp. 2–9

Generated on Thu Mar 7 12:07:24 2024 by [LATExml](http://dlmf.nist.gov/LaTeXML/)