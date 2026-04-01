[# Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki†, Shubham Tulsiani† Carnegie Mellon University † Equal Advising https://dex4d.github.io](https://dex4d.github.io)

###### Abstract

Learning generalist policies capable of accomplishing a plethora of everyday tasks remains an open challenge in dexterous manipulation. In particular, collecting large-scale manipulation data via real-world teleoperation is expensive and difficult to scale. While learning in simulation provides a feasible alternative, designing multiple task-specific environments and rewards for training is similarly challenging. We propose Dex4D, a framework that instead leverages simulation for learning *task-agnostic* dexterous skills that can be flexibly recomposed to perform diverse real-world manipulation tasks. Specifically, Dex4D learns a domain-agnostic 3D point track conditioned policy capable of manipulating *any object to any desired pose*. We train this ‘Anypose-to-Anypose’ policy in simulation across thousands of objects with diverse pose configurations, covering a broad space of robot-object interactions that can be composed at test time.
At deployment, this policy can be zero-shot transferred to real-world tasks without finetuning, simply by prompting it with desired object-centric point tracks extracted from generated videos. During execution, Dex4D uses online point tracking for closed-loop perception and control.
Extensive experiments in simulation and on real robots show that our method enables zero-shot deployment for diverse dexterous manipulation tasks and yields consistent improvements over prior baselines. Furthermore, we demonstrate strong generalization to novel objects, scene layouts, backgrounds, and trajectories, highlighting the robustness and scalability of the proposed framework.
Project page: [https://dex4d.github.io](https://dex4d.github.io).

![[Uncaptioned image]](x1.png)

*Figure 1: Overview of Dex4D. We leverage video generation and 4D reconstruction to generate object-centric point tracks. Conditioned on the point tracks, we use a task-agnostic sim-to-real policy trained via Anypose-to-Anypose for task execution. Our policy trained entirely in simulation can be seamlessly deployed in the real world and generalizes to diverse configurations.*

## I Introduction

The lack of high-quality, diverse, and scalable data remains a fundamental bottleneck in learning dexterous robot manipulation. Collecting real-world manipulation trajectories is expensive, difficult to instrument, and limited in coverage and diversity. Furthermore, learning dexterous manipulation via teleoperation poses unique challenges due to the difficulty of precisely controlling high-dimensional robotic hands and fingers, which makes large-scale data collection slow and error-prone [[[65](#bib.bib37)]].

Learning dexterous manipulation behaviors via sim-to-real reinforcement learning (RL) provides a promising alternative [[[62](#bib.bib43), [10](#bib.bib61)]]. Benefiting from highly parallel GPU-based simulation [[[33](#bib.bib59), [35](#bib.bib60)]] that largely improves interaction data bandwidth, RL-based policiess can be learnt in a few hours in simulation that are equivalent to years in the real world. However, training language-instructable ‘generalist’ robot policies in simulation requires substantial engineering effort, including designing complex simulation environments, specifying task descriptions and instructions, performing tedious reward shaping, and tuning RL pipelines across an ever-growing set of tasks [[[14](#bib.bib3), [57](#bib.bib2)]].

We argue that instead of learning language-conditioned and task-specific policies, we can use highly parallel simulation to learn fundamental task-agnostic manipulation skills that can be flexibly composed using a high-level planner, such as video generation models [[[59](#bib.bib41), [52](#bib.bib42)]] that have shown remarkable open-world generalization, to perform general downstream tasks. We operationalize this insight in our framework Dex4D, which learns a point track conditioned policy for Anypose-to-Anypose – manipulating any object from any current pose to any target pose. A key technical contribution lies in our goal representation – instead of separately encoding current and target object points, we propose Paired Point Encoding that leverages the correspondences across them.

We train our Anypose-to-Anypose policy across thousands of objects in simulation. The training process spans a broad space of object poses, trajectories, and hand-object interactions, enabling compositional generalization at test time. We show that our Anypose-to-Anypose policy, with geometry-aware and domain-robust point track representation as conditions, can be combined with video generation models to allow sim-to-real dexterous manipulation for generic tasks. Specifically, given a task description, Dex4D queries a foundational video model to generate a successful video task plan. We then leverage 4D reconstruction to extract object-centric point tracks from the generated video as an interface for goal specifications and policy conditions for our task-agnostic Anypose-to-Anypose policy, while using efficient online point tracking for closed-loop perception and control. As a result, Dex4D enables zero-shot transfer to real-world tasks without any real robot finetuning.

We evaluate Dex4D extensively in simulation and on real robotic platforms, comparing against state-of-the-art baselines. Our results show substantial improvements in success rate, task progress, and robustness. Furthermore, we demonstrate strong generalization to novel objects and poses, scene layouts, backgrounds, and task trajectories, highlighting the scalability and robustness of the proposed representation and learning framework. In summary, our contributions are as follows:

-
•

We propose Anypose-to-Anypose, a task-agnostic sim-to-real learning formulation without tedious simulation tuning and task-specific reward shaping.

-
•

We propose to leverage point tracks from generated videos and 4D reconstruction as an interface for goal specifications and policy conditions.

-
•

We propose Paired Point Encoding, an effective goal representation, along with a point track conditioned transformer-based action world model architecture to improve policy learning.

-
•

Extensive experiments demonstrate superior performance and strong generalization to unseen objects and poses, scene layouts, backgrounds, and task trajectories.

![Figure](x2.png)

*Figure 2: Overview of our Dex4D teacher and student network architectures. (a) We first learn a teacher policy via RL with privileged states and full points sampled on the whole object, leveraging our proposed Paired Point Encoding representation. (b) Given partial observation, i.e., robot proprioception, last action, and masked paired points, we distill from the teacher and learn a transformer-based student action world model that jointly predicts actions and future robot states.*

## II Related Works

### II-A Generalizable Dexterous Manipulation

Endowing robots with human-level and generalizable dexterity is a long-standing goal for generalist robots that work under diverse real-world scenarios. It’s also very challenging due to its high-DoF and high-dynamics nature. Prior optimization-based works often rely on contact-based optimization [[[24](#bib.bib23), [25](#bib.bib24), [56](#bib.bib25), [68](#bib.bib26), [64](#bib.bib27)]] to synthesize dexterous grasping poses, which are executed by motion planning.
However, these works are mainly limited to grasping and prone to disturbances without closed-loop feedback.
Another line of works uses mocap devices or teleoperation to collect dexterous manipulation data and train policies on them via imitation learning [[[54](#bib.bib28), [23](#bib.bib32), [58](#bib.bib29)]].
However, these works suffer from in-domain data collection and fail to generalize to unseen tasks, objects, and scenes.
Recently, reinforcement learning (RL) has shown promise on generalizable dexterous manipulation, including dexterous grasping [[[62](#bib.bib43), [53](#bib.bib44), [30](#bib.bib12), [49](#bib.bib13), [50](#bib.bib45)]], in-hand reorientation [[[43](#bib.bib34), [44](#bib.bib35), [65](#bib.bib37), [55](#bib.bib33), [28](#bib.bib36)]], and motion tracking [[[31](#bib.bib31), [27](#bib.bib38), [61](#bib.bib39), [39](#bib.bib72)]].
Nonetheless, they often lack autonomy for high-level tasks that require task-specific planning.
In contrast to all these works, our method leverages video generation and 4D reconstruction for high-level planning and trains an object-centric task-agnostic policy that works across tasks using sim-to-real RL with point tracks as an interface, achieving generalizable and autonomous dexterous manipulation.

### II-B Video-Based Robot Learning

Recent years witnessed huge progress in video generation [[[5](#bib.bib40), [59](#bib.bib41), [52](#bib.bib42)]] and learning from human videos [[[3](#bib.bib46), [4](#bib.bib47), [20](#bib.bib48), [60](#bib.bib4), [2](#bib.bib49), [38](#bib.bib50)]]. Video generation models not only can be used for entertainment or simulation, but also serve as world models or powerful high-level planners for robotics tasks since they are trained on enormous amounts of Internet videos and contain rich human priors [[[34](#bib.bib21)]]. Recent works [[[40](#bib.bib7), [70](#bib.bib73), [21](#bib.bib1), [8](#bib.bib22), [12](#bib.bib58)]] leverage video generation models or flow models as planners and use either pose estimation with motion planning or heuristic retargeting to map generated pixels to actions. However, these works suffer from large embodiment gaps and a lack of closed-loop feedback, which are crucial for highly dynamic tasks such as dexterous manipulation. They also require either object mesh [[[40](#bib.bib7)]] or clean point tracks [[[21](#bib.bib1)]] for pose estimation, which is hard to satisfy in the real world, especially with finger occlusions. In contrast, we train a closed-loop policy via sim-to-real, leveraging our proposed Paired Point Encoding representation along with extensive point masking and domain randomization. Therefore, our method is robust to real-world noisy sensor input and can generalize to diverse unseen configurations.

### II-C 3D Policy Learning

Spatial understanding is crucial for robot agents to reason about the 3D scene around us. Therefore, it’s important to find a good 3D representation for policy learning. [[[72](#bib.bib52), [67](#bib.bib51), [17](#bib.bib56), [13](#bib.bib57)]] leverage point cloud as input for imitation learning, and [[[67](#bib.bib51)]] proves the sufficiency of minimal PointNet [[[41](#bib.bib5)]] to encode the point cloud. Others use scene representations (voxelized neural fields [[[66](#bib.bib53)]], occupancy [[[26](#bib.bib55)]], and Gaussian Splatting [[[29](#bib.bib54)]]) for policy learning. Compared to these works, our work extends goal-conditioned policy learning by using 3D representations as goal conditions. We propose Paired Point Encoding as policy conditions that combine current object points with target object points, supporting task-agnostic learning without specific language instructions as conditions. We also leverage world modeling as auxiliary supervision signals to jointly learn action prediction and robot dynamics from proprioception and 3D perception.

## III Learning Point Track Policy via Task-Agnostic Sim-to-Real

In this section, we introduce our point track policy trained via Anypose-to-Anypose (AP2AP), a task-agnostic sim-to-real learning formulation for dexterous manipulation. We detail our AP2AP setup (§ [III-A](#S3.SS1)), our proposed Paired Point Encoding (§ [III-B](#S3.SS2) and Fig. [3](#S3.F3)), and teacher-student policy learning (§ [III-C](#S3.SS3) and Fig. [2](#S1.F2)). We then outline how to deploy the sim-to-real AP2AP policy using point tracks from generated videos in § [IV](#S4).

### III-A Anypose-to-Anypose

Anypose-to-Anypose (AP2AP) is a task-agnostic sim-to-real learning formulation for dexterous manipulation. AP2AP abstracts manipulation as directly transforming an object from an arbitrary initial pose to an arbitrary target pose in 3D space, without assuming task-specific structure, predefined grasps, or motion primitives. Unlike prior approaches [[[40](#bib.bib7), [21](#bib.bib1)]] that decompose manipulation into grasp generation, pose estimation, and planning, AP2AP treats object pose transformation itself as the fundamental learning objective, enabling a unified and reactive control policy for high-DoF dexterous hands.

We formulate AP2AP as a goal-conditioned Markov Decision Process (MDP) $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma,\mathcal{G}\rangle$ of state $s\in\mathcal{S}$, action $a\in\mathcal{A}$, transition function $\mathcal{T}$, reward $r\in\mathcal{R}$, discount factor $\gamma$, and goal $g\in\mathcal{G}$. The objective is to maximize the expected return $\mathbb{E}\left[\sum_{t}\gamma^{t}r_{t}\right]$ by finding an optimal policy $\pi^{*}(a_{t}|s_{t},g_{t})$, where the subscript $t$ indexes the timestep.

At the beginning of each episode, an object is placed on the table with a random initial position and orientation. The first goal requires the robot to grasp and lift the object to a specified pose. Once a goal is stably achieved, the next goal is randomly set as a nearby target pose, encouraging continuous pose-to-pose transitions and effective local exploration.

Our AP2AP policy is trained entirely in simulation using 3,200 objects from UniDexGrasp [[[62](#bib.bib43)]] under diverse pose configurations and extensive domain randomization.
By learning to perform arbitrary pose-to-pose transformations on a wide range of objects, the policy acquires embodiment grounding and contact-rich manipulation skills in a task-agnostic manner.
As a result, our method doesn’t require task-specific tuning and generalizes zero-shot to unseen objects and downstream manipulation tasks in the real world.

### III-B Goal Representation via Paired Point Encoding

![Figure](x3.png)

*Figure 3: Comparison between our Paired Point Encoding with other representations. Point features encoded from our Paired Point Encoding keep correspondence and permutation-invariance of the current and target object points, which shows better performance for policy learning.*

A key design choice for training the AP2AP policy is selecting a goal representation that can be robustly extracted at deployment time while remaining informative for pose-conditioned control. In this work, we represent objects using sparse object points, which are widely used, geometry-aware, and can be reliably obtained in the real world using point trackers. We can also easily obtain target object points given the desired transformation. Then a key design question is how to encode current and target object points as an effective goal representation so that they are maximally useful for policy learning.

A common approach is to encode current and target object points into two latent features and condition the policy on these features [[[32](#bib.bib8)]].
However, such encodings discard correspondence between current and target object points, which is critical for differentiating object poses.
For example, when a ball undergoes pure rotation without translation, the shape of the points remains unchanged, even though the object pose is different. In this case, correspondence is the only information that distinguishes between identical shapes under different poses. To address this limitation, we propose Paired Point Encoding, a representation that explicitly preserves correspondence between the current and target object points. As illustrated in Fig. [3](#S3.F3), given the current object points $\{\bm{p}^{i}_{t}\}_{i=1}^{N}$ and the target object points $\{\bm{\bar{p}}^{i}_{t}\}_{i=1}^{N}$ at timestep $t$, we construct paired points $\{\bm{q}^{i}_{t}\}_{i=1}^{N}$ by concatenating each pair of corresponding points.
Each pair point is therefore 6-dimensional, consisting of a 3D current object point and its 3D target counterpart. For point index $i$ at timestep $t$, the paired point is defined as below:

$$\bm{q}_{t}^{i}=\begin{bmatrix}\bm{p}^{i}_{t}\\ \bm{\bar{p}}^{i}_{t}\end{bmatrix}\in\mathbb{R}^{6},$$ \tag{1}

These paired points $\{\bm{q}^{i}_{t}\}_{i=1}^{N}\in\mathbb{R}^{N\times 6}$ are then fed into a PointNet-style encoder [[[41](#bib.bib5), [42](#bib.bib6)]], which consists of shared MLP layers and mean-max mixed pooling to encode them into paired point features. In this way, we keep both correspondence and permutation-invariance of these points. Building on the Paired Point Encoding as goal representation, we now describe how it is used to train the AP2AP policy in simulation via a teacher–student learning framework.

### III-C Teacher-Student Policy Learning

To train the AP2AP policy in simulation, we follow a standard teacher-student distillation framework [[[30](#bib.bib12), [49](#bib.bib13)]]. As shown in Fig. [2](#S1.F2), we first learn a teacher policy via visual RL [[[47](#bib.bib9)]] with proprioception, last action, privileged states and points uniformly sampled on the whole object, leveraging the Paired Point Encoding. Then, given only proprioception, last action, and partial points by masking, we leverage DAgger [[[46](#bib.bib10)]] to distill teacher to the student policy.

#### III-C1 RL Teacher Policy Learning

In the first phase, we train a teacher policy using PPO [[[47](#bib.bib9)]] with privileged states and fully observed object geometry in simulation. As shown in Fig. [2](#S1.F2), the state $s_{t}$ consists of robot proprioception (joint angles and velocities), the last action, and privileged information (e.g., joint torques, fingertip-to-object distances, etc.). Following Sec. [III-B](#S3.SS2), we compose current and target object points into paired points as the goal representation $g_{t}$ and encode them using a lightweight PointNet [[[41](#bib.bib5)]] to preserve both correspondence and permutation invariance. The resulting feature is concatenated with the state components and provided as input to the PPO actor and critic networks.

To facilitate effective exploration and stable RL training, we adopt a three-stage curriculum. In the first stage, training is restricted to a single object category with a low environment reset threshold and a high robot arm speed limit to encourage early reward acquisition. In the second stage, the arm speed limit is reduced for real-world safety, and the reset threshold is increased. In the third stage, we train on all 3,200 objects with more challenging initializations and resets, lower control frequency, and more conservative learning updates. Throughout training, we apply extensive domain randomization, including observation and action noise, PD gains, hand–object friction, and external force disturbances, to enable smooth and robust sim-to-real transfer.

For reward shaping, instead of directly using the 6D pose (object position + rotation), our reward function leverages object points for a smoother reward landscape [[[69](#bib.bib71)]]. These rewards encourage the current object points to closely match the target object points, while promoting hand-object affinity and discouraging exaggerated motions:

$$r=r_{\mathrm{goal}}+r_{\mathrm{f,o}}+r_{\mathrm{h,o}}+r_{\mathrm{bonus}}+r_{\mathrm{curl}}+r_{\mathrm{table}}+r_{\mathrm{action}}$$ \tag{2}

where $r_{\mathrm{goal}}$, $r_{\mathrm{f,o}}$, $r_{\mathrm{h,o}}$, $r_{\mathrm{bonus}}$, $r_{\mathrm{curl}}$, $r_{\mathrm{table}}$, and $r_{\mathrm{action}}$ represent rewards for current-target point distances, finger-object distances, hand-object distance, success bonus, finger curl, table collision penalty, and action penalty, respectively. More details on curriculum design and reward shaping are in Sec. [-E](#A0.SS5) and Sec. [-F](#A0.SS6).

#### III-C2 Student Action World Model Learning

After training the teacher policy, we distill it into a student policy under partial observability using DAgger [[[46](#bib.bib10)]]. We introduce a transformer-based action world model that jointly learns action prediction and robot joint dynamics. This joint formulation improves action learning [[[19](#bib.bib14), [32](#bib.bib8), [71](#bib.bib15), [7](#bib.bib20)]] and supports safer and more controllable deployment, particularly for high-DoF and highly dynamic hand–arm systems.

As illustrated in Fig. [2](#S1.F2), the student policy takes as input robot proprioception (joint angles and velocities), the last action, and masked paired points. These inputs are first tokenized using MLPs and a PointNet-style encoder [[[41](#bib.bib5), [42](#bib.bib6)]] with mean–max mixed pooling, and then processed by self-attention layers [[[51](#bib.bib16)]]. The output token corresponding to the last action $a_{t-1}$ is used to decode the robot action $a_{t}$, while the tokens corresponding to the current joint angle $\bm{\theta}_{t}$ and velocity $\dot{\bm{\theta}}_{t}$ are used to predict the next-state joint angle $\bm{\theta}_{t+1}$ and velocity $\dot{\bm{\theta}}_{t+1}$.

The action world model is trained with a combination of a DAgger behavior cloning loss and a world modeling loss:

$$
\begin{aligned}
\mathcal{L} &=\mathcal{L}_{\mathrm{bc}}+\mathcal{L}_{\mathrm{wm}} \tag{3}
\end{aligned}
$$
| | | $\displaystyle=\lVert a_{t}^{\mathrm{stu}}-a_{t}^{\mathrm{tea}}\rVert_{1}+\lVert\begin{bmatrix}\hat{\bm{\theta}}_{t+1}-\bm{\theta}_{t+1}\\ \hat{\dot{\bm{\theta}}}_{t+1}-\dot{\bm{\theta}}_{t+1}\end{bmatrix}\rVert_{1}$ | |

where $a_{t}^{\mathrm{stu}}$ and $a_{t}^{\mathrm{tea}}$ denote the student and teacher actions, respectively, and $\hat{\bm{\theta}}_{t+1}$, $\hat{\dot{\bm{\theta}}}_{t+1}$ and $\bm{\theta}_{t+1}$, $\dot{\bm{\theta}}_{t+1}$ denote the predicted and ground-truth next-step joint angles and velocities.

To emulate object point occlusions caused by the hand in real-world settings and improve robustness to monocular viewpoints, we introduce a random plane-height masking strategy. Specifically, we sample a random plane through the object center and mask out points on one side of the plane, followed by sampling a random height that masks the majority of points above it and a minority below it. Target points are masked accordingly based on correspondence. This strategy enables the student policy to generalize across diverse camera viewpoints under partial observation. Further implementation details are provided in Sec. [-H](#A0.SS8).

## IV Generalizable Dexterous Manipulation from Point Track Policy

To leverage the AP2AP policy in the real world, we condition it on desired 3D point tracks – a sequence of target object points that specifies the desired object configuration over time. Such point tracks can be obtained from diverse sources, including video generation models or one-shot human demonstrations. In this section, we describe how we extract point tracks from generated videos and how they are used to drive closed-loop policy execution.

### IV-A From Generated Video to Object-Centric Point Tracks

Large-scale video generation models provide a rich source of object motion and manipulation information, as they are trained on vast collections of Internet videos [[[34](#bib.bib21), [40](#bib.bib7), [21](#bib.bib1), [8](#bib.bib22)]]. To use these videos for real-world robot deployment, we lift the videos into object-centric point tracks, which define a target trajectory that can directly condition the AP2AP policy.

Formally, given a language instruction $l$ and an initial RGBD observation $\{I_{0},D_{0}\}$, we first generate a sequence of future RGB frames $\{I_{t}\}_{t=1}^{T}$ using an off-the-shelf video generation model. Using the initial object segmentation mask and the generated frames, we first perform 2D point tracking to obtain object 2D point tracks $\{\bm{\bar{u}}_{t}^{i}\}_{t=1,i=1}^{T,N}\in\mathbb{R}^{T\times N\times 2}$.

Next, we perform relative depth estimation for each frame and calibrate it using the initial depth observation $D_{0}$. Specifically, each estimated depth map is scaled based on the ratio between the median depth of the frame and the median depth of the initial observation. This allows us to lift the 2D point tracks into metric 3D point tracks $\{\bm{\bar{p}}_{t}^{i}\}_{t=1,i=1}^{T,N}\in\mathbb{R}^{T\times N\times 3}$. The object-centric point tracks serve as goal specifications of the task and target object points to condition the policy.

The resulting point tracks provide a structured, object-centric representation of the desired pose over time. This representation can be directly used as a plan for the AP2AP point track policy, guiding the robot to perform pose-to-pose manipulation in the real world. Compared to prior approaches that rely on fully metric depth estimation and explicit calibration [[[40](#bib.bib7), [21](#bib.bib1)]], calibrating relative depth produces smoother and more stable metric depths, resulting in cleaner point tracks. Additional implementation details and visualizations are provided in Sec. [-C](#A0.SS3).

### IV-B Closed-Loop Perception and Control

With the target point tracks defined, we now describe how the AP2AP policy executes it in a closed loop in the real world.

At the start of execution, the first set of target points is assigned to the policy. The initial tracked 2D points from the generated video are also provided for an online point tracker [[[16](#bib.bib11)]], which is used to track the object 2D points from the image observation in real time, which are then back-projected to 3D using the RGBD camera.

At each timestep, the current object 3D points and target object 3D points are composed into paired points and provided to the student policy along with robot proprioception and the last action. The policy then outputs actions for robot control. To determine when to advance to the next set of target points, we compute the average distance between corresponding visible points at each timestep:

$$d_{t}=\frac{1}{N^{\prime}}\sum_{i=1}^{N^{\prime}}\lVert\bm{p}_{t}^{i}-\bm{\bar{p}}_{t}^{i}\rVert_{2}$$ \tag{4}

where $N^{\prime}$ is the number of visible points. When $d_{t}$ falls below a threshold, the target points are updated to the next ones in the goal 3D point tracks. This process repeats in a closed loop until the final target points $\{\bm{\bar{p}}_{T}^{i}\}_{i=1}^{N}$ are reached.

*TABLE I: Quantitative results measured by Success Rate (SR) and Task Progress (TP) in simulation. CL stands for closed-loop. We adapt NovaFlow and NovaFlow-CL to dexterous hands by using our method for first-stage dexterous grasping.*

## V Experiments

### V-A Implementation Details

In our work, we use a 22-DoF dexterous hand-arm system that comprises a 6-DoF xArm6 robot arm and a 16-DoF LEAP hand [[[48](#bib.bib63)]]. We use a single-view RealSense D435 camera for RGBD sensing and Apriltags [[[37](#bib.bib64)]] for calibration.

For the action space, the policy outputs a 22-dimensional action, which contains 6-DoF arm delta joint angles and 16-DoF LEAP hand absolute joint angles. We found that this configuration is more robust in RL training without compromising on full hand dexterity [[[11](#bib.bib69), [1](#bib.bib68), [30](#bib.bib12)]]. The action is then converted to the 22-DoF joint target used for motor control.

For desired point tracks acquisition, we use Wan2.6 [[[52](#bib.bib42)]] for video generation, Video Depth Anything [[[9](#bib.bib66)]] for relative video depth estimation, SAM2 [[[45](#bib.bib67)]] for initial frame segmentation, and CoTracker3 [[[16](#bib.bib11)]] for 2D point tracking. All these components are modular, reusable, and replaceable, which can be updated and swapped to new state-of-the-art models.

We train teacher and student policies using the Isaac Gym simulator [[[33](#bib.bib59)]]. For point numbers, we sample 128 points for teacher training, and 64 points for student training and inference. For RL teacher training, we use symmetric PPO [[[47](#bib.bib9)]] to learn 15k steps for the first stage, 10k steps for the second stage, and another 25k steps for the third stage to ensure convergence. For student policy learning, we select the best teacher checkpoint and learn 25k steps using DAgger [[[46](#bib.bib10)]]. It takes 2-3 days for teacher training, and around 20 hours for student training, using a single NVIDIA RTX A6000 GPU.

### V-B Simulation Experiments

We evaluate our method with several baselines in a simulation suite to provide a fair and reproducible comparison.

#### V-B1 Tasks

We evaluate our method and baselines on six dexterous manipulation tasks, namely Apple2Plate, Pour, Hammer, StackCup, RotateBox, and Sponge2Bowl. These tasks involve dexterous grasping, arm movement, object reorientation, and spatial reasoning. We select objects from our object set from UniDexGrasp [[[62](#bib.bib43)]] and the unseen YCB dataset [[[6](#bib.bib19)]].
For more details on task specifications, please refer to Sec. [-B](#A0.SS2) in the appendix.

#### V-B2 Baselines

We compare against NovaFlow [[[21](#bib.bib1)]], a method that leverages 3D actionable point tracks for robot manipulation. NovaFlow first extracts 3D action point tracks from generated videos, and then applies a grasp generator [[[36](#bib.bib17)]] and performs open-loop motion planning based on transformation estimation between current and target object points using the Kabsch algorithm [[[15](#bib.bib18)]]. Since NovaFlow is parallel gripper-only, for fair comparison, we adapt it to dexterous hands by applying our method for dexterous grasping and locking the fingers after lifting. Then we conduct pose estimation based on the current state and perform motion planning in an open-loop manner.

Moreover, since NovaFlow [[[21](#bib.bib1)]] is open-loop without feedback, which is crucial for high-dynamics tasks like dexterous manipulation, we also implemented a closed-loop version of NovaFlow (NovaFlow-CL) as a baseline. Specifically, we first apply our method for dexterous grasping, and then at each timestep, we estimate the transformation between current and target object points, and do real-time IK to move the robot arm accordingly.

#### V-B3 Evaluation Protocols and Metrics

In the simulation, we manually pre-define several waypoints as goal poses and compute the target points for policy condition. We also apply random Gaussian noise to both current and target object points independently for realism. The goal pose will update once the current goal pose is achieved.

For observation, instead of using masked points as input, we put an RGBD camera in the scene and compute real-time object points based on visibility.

To measure performance, we use Success Rate (SR) and Task Progress (TP) as quantitative metrics. Success is determined by whether the last waypoint is achieved or not, and Task Progress is defined as the average number of waypoints achieved. For each trial, we initialize the object with a random pose, and we conduct 100 trials for each task.

#### V-B4 Results and Analysis

We report our simulation results in Table. [I](#S4.T1). As shown in the results, our method outperforms all baselines by a large margin, demonstrating the effectiveness and superiority of our method.

First by comparing open-loop NovaFlow [[[21](#bib.bib1)]] and closed-loop NovaFlow-CL, we can see that closed-loop feedback greatly improves performance ($+9.2\%$ in SR and $16\%$ in TP) since the object may move around in the hand. Closed-loop pose estimation could perform replanning so that the object could remain on the correct track instead of accumulating pose errors, achieving a huge bonus on task progress.

Further comparing our method with NovaFlow-CL, we also surpass it, achieving a significant performance gain ($+16.3\%$ in SR and $+10.4\%$ in TP). This demonstrates that our policy, trained on AP2AP, exhibits strong generalizability to unseen tasks, objects, and trajectories, despite both methods utilizing closed-loop feedback. We attribute this performance gain to the fact that during the RL and DAgger process, the policy is able to traverse sufficient states so that it can generalize to unseen scenarios. NovaFlow-CL fails mainly because of insufficient hand reactivity to high dynamics that causes object falling, limited motion planning solutions that severely occlude object points, and pose estimation errors due to point noises, especially when the number of visible points is low. This further highlights our method’s robustness and adaptivity to out-of-distribution scenarios and effectiveness in generalizing across objects and scenes.

### V-C Ablation Studies

To further analyze our framework design, we conduct ablation studies on various components to highlight the importance of each element in our method. Ablation results are shown in Table [II](#S5.T2) and Fig. [4](#S5.F4), showing that our method surpasses all ablations, proving the effectiveness of all the modules.

#### V-C1 Importance of Paired Point Encoding

First, we discuss whether using our proposed Paired Point Encoding can improve the performance for both student and teacher policies. We compare against two variants as shown in Fig. [3](#S3.F3) (a)(b): (a) MLP Point Encoding that directly tokenizes current and target object points using MLPs; (b) Decoupled Point Encoding that use two PointNet tokenizers to encode current and target object points separately, without using our proposed paired points. Results of student policy distillation and RL teacher training (for stage 1 & 2) are shown in Table [II](#S5.T2) and Fig. [4](#S5.F4), respectively. Results show that using our Paired Point Encoding significantly improve performance for both policies.

For the student policy, our finding suggests that using MLP to encode the points leads to severe performance degradation, making SR reduce to $5.7\%$. We also find out that using separate PointNet encoders to encode current and target object points individually will also severely lower the performance since the policy loses the correspondence between the two sets of points. This highlights our representation’s advantage to effectively encode the object geometry information and point correspondence.

We also ablate with MLP Point Encoding and Decoupled Point Encoding in our RL teacher training to verify the effectiveness of our Paired Point Encoding representation in RL training. As shown in Fig. [4](#S5.F4), our method outperforms both variants, demonstrating that our representation and framework can even boost the performance for visual RL, which is considered harder in prior works [[[18](#bib.bib62), [22](#bib.bib70), [50](#bib.bib45)]].

*TABLE II: Ablation studies on the student policy. SR and TP are averaged across tasks.*

#### V-C2 Ablation on Policy Architecture

Furthermore, we also ablate different neural network architecture design choices for our student policy. We compare against two other variants: (c) w/o Self-Attention that concatenates tokens and uses MLP to decode actions; and (d) w/o World Modeling that discards next state prediction. The results are shown in Table [II](#S5.T2), showing that our method outperforms both ablations, proving the effectiveness of our student policy design.

Compared with w/o Self-Attention that naïvely concatenates proprioception and paired point features and uses MLP to decode actions, self-attention layers (i.e., transformer encoder) can attend to different tokens from proprioception and paired points, which better captures relations from different input components and achieves non-trivial performance gains. We also find that integrating world modeling can improve performance, which correlates well with the synergistic effects of policy learning and world modeling.

![Figure](x4.png)

*Figure 4: Mean reward curve of the first two stages of teacher training. Step 15k is the curriculum boundary. Our method outperforms both ablation variants.*

### V-D Real-World Experiments

![Figure](x5.png)

*Figure 5: Overview of real-world dexterous manipulation tasks. Two frames are shown in each column for each task.*

![Figure](x6.png)

*Figure 6: Qualitative comparison between our method and the baseline. The baseline method suffers from object dropping and inaccurate post-grasping movement due to the lack of hand feedback and vulnerability to few and noisy visible points, while our method performs robustly.*

*TABLE III: Quantitative results measured by Success Rate (SR) in the real world. All the objects are unseen and there are no real robot demonstrations.*

We deploy our simulation-trained policy to the real world and conduct extensive experiment suites on four tasks, namely LiftToy, Broccoli2Plate, Meat2Bowl, and Pour, as shown in Fig. [5](#S5.F5). Note that all the objects are unseen and there are no real robot demonstrations for any task. We deploy our policy as the procedure detailed in Sec. [IV](#S4). We compare our method against NovaFlow-CL [[[21](#bib.bib1)]], which uses the real-time current and target object points to estimate 6D transformation at each planning step and do motion planning to reach the pose. Note that for NovaFlow-CL, we first leverage our method to grasp the object and then perform closed-loop pose estimation, which is the same as in simulation experiments.

As shown in Table [III](#S5.T3), our method achieves a 22.5% performance gain in SR compared to the baseline, demonstrating our method’s superiority against the motion planning-based method. The superiority of our method mainly comes from the closed-loop reactivity for both arm and hand, and robust action prediction under noisy point input. Video results and comparisons can be found in the supplementary video.

We also show the qualitative comparison in Fig. [6](#S5.F6) and the supplementary video. As we can see, the baseline fails due to a couple of reasons. First, since the baseline is unaware of the hand and object grasping, the object would gradually fall off the hand during arm moving due to the lack of feedback. In contrast, our method learns to adjust or regrasp the object and then proceeds with the task. Moreover, in the real world the 3D point tracking poses large amounts of noise, including ones coming from inaccurate 2D tracking, noisy depth sensing, and latency, especially when the LEAP hand fingers severely occlude the object. Since the baseline method leverages the Kabsch algorithm [[[15](#bib.bib18)]] to solve the 6D pose, it’s prone to noisy observation especially when visible points are few. The Kabsch algorithm can hardly solve the correct rotation between noisy current and target object points under real-world scenarios when the object is occluded by the hand, as in the Pour task, the baseline has a 0 success rate. In contrast, our method remains robust even if there are less than 10 visible points left. Finally, some failures of the baseline come from limited solutions of motion planning since we use a 6-DoF xArm6, which completely occludes the object from the camera.

Moreover, our method is robust to various generalization tests in the real world. As shown in Fig. [1](#S0.F1), although our policy is only trained on single-object scenarios purely in simulation, it generalizes well to unseen object types and poses, backgrounds, camera views, task trajectories, and external disturbances.

However, we also noticed some failure modes of our policy, which can be further improved in future works. First, in the real world, the real-time CoTracker3 [[[16](#bib.bib11)]] will lose track of the object when there are significant object movements, similar nearby textures, or unintended object rotation that blocks initially tracked points. This is the major cause of failures. Sometimes the policy also tends to push the object to form a firm grasp, but this might pose extra forces that in turn knocks over the object.

## VI Conclusions, Limitations, and Future Works

Conclusions. In this work, we propose Dex4D, a framework for generalizable dexterous manipulation via object-centric point tracks and task-agnostic sim-to-real learning. At the core of Dex4D is to decouple recognition and control by leveraging video generation and 4D reconstruction to generate object-centric point tracks as high-level planning, and training a task-agnostic sim-to-real policy for low-level control. We further propose a novel Paired Point Encoding representation and a transformer-based action world model to enhance 3D goal-conditioned policy learning. Extensive experiments in simulation and the real world verify the effectiveness of our framework, and show our remarkable generalization to unseen tasks, objects, and scenes. We hope our work can benefit future research on generalizable dexterous manipulation.

Limitations and Future Works. Despite compelling results, our work has certain limitations that can be further improved in future works. First, in our work we didn’t incorporate human grasp priors from HOI datasets and Internet videos due to the lack of amount and diversity of clean mocap sequences and the large embodiment gap between human hands and the LEAP hand, which is large in size and only has four thick fingers. It’s promising to leverage these abundant hand-related data sources along with thinner and more human-like dexterous hands to unlock more functional behaviors. Second, our AP2AP formulation is currently limited to single-object manipulation. Extending it to objects with more complicated geometries, such as articulated objects, would be a promising direction. Additionally, how to incorporate other modalities, such as tactile sensing, is also an interesting question. Finally, in the future we could develop more accurate, robust, and faster online tracking models for better point tracking that enables lower latency and better tracking performance on the deployment side.

## Acknowledgments

We express our sincere gratitude to Jiashun Wang, Jialiang Zhang, and Andrew Wang for fruitful discussions, to Kenny Shaw for hardware support, and to Himangi Mittal, Minsik Jeon, Yehonathan Litman, Qitao Zhao, Zihan Wang, Hanzhe Hu, and Lucas Wu for presentation feedback.
This work was supported in part by gifts from Google and CISCO, NSF Award IIS-2442282, DARPA SAFROn award HR0011-25-3-0203, NSF Career award, an Amazon robotics award, and AFOSR Grant FA9550-23-1-0257.

## References

-
[1]
A. Agarwal, S. Uppal, K. Shaw, and D. Pathak (2023)

Dexterous functional grasping.

arXiv preprint arXiv:2312.02975.

Cited by: [§V-A](#S5.SS1.p2.1).

-
[2]
H. Bharadhwaj, D. Dwibedi, A. Gupta, S. Tulsiani, C. Doersch, T. Xiao, D. Shah, F. Xia, D. Sadigh, and S. Kirmani (2024)

Gen2act: human video generation in novel scenarios enables generalizable robot manipulation.

arXiv preprint arXiv:2409.16283.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[3]
H. Bharadhwaj, A. Gupta, V. Kumar, and S. Tulsiani (2024)

Towards generalizable zero-shot manipulation via translating human interaction plans.

In 2024 IEEE International Conference on Robotics and Automation (ICRA),

pp. 6904–6911.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[4]
H. Bharadhwaj, R. Mottaghi, A. Gupta, and S. Tulsiani (2024)

Track2act: predicting point tracks from internet videos enables generalizable robot manipulation.

In European Conference on Computer Vision,

pp. 306–324.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[5]
T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, et al. (2024)

Video generation models as world simulators.

OpenAI Blog 1 (8), pp. 1.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[6]
B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel, and A. M. Dollar (2015)

Benchmarking in manipulation research: the ycb object and model set and benchmarking protocols.

arXiv preprint arXiv:1502.03143.

Cited by: [§V-B1](#S5.SS2.SSS1.p1.1).

-
[7]
J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al. (2025)

WorldVLA: towards autoregressive action world model.

arXiv preprint arXiv:2506.21539.

Cited by: [§III-C2](#S3.SS3.SSS2.p1.1).

-
[8]
B. Chen, T. Zhang, H. Geng, K. Song, C. Zhang, P. Li, W. T. Freeman, J. Malik, P. Abbeel, R. Tedrake, et al. (2025)

Large video planner enables generalizable robot control.

arXiv preprint arXiv:2512.15840.

Cited by: [§II-B](#S2.SS2.p1.1),
[§IV-A](#S4.SS1.p1.1).

-
[9]
S. Chen, H. Guo, S. Zhu, F. Zhang, Z. Huang, J. Feng, and B. Kang (2025)

Video depth anything: consistent depth estimation for super-long videos.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 22831–22840.

Cited by: [§V-A](#S5.SS1.p3.1).

-
[10]
Y. Chen, Y. Geng, F. Zhong, J. Ji, J. Jiang, Z. Lu, H. Dong, and Y. Yang (2023)

Bi-dexhands: towards human-level bimanual dexterous manipulation.

IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (5), pp. 2804–2818.

Cited by: [§I](#S1.p2.1).

-
[11]
M. Ciocarlie, C. Goldfeder, and P. Allen (2007)

Dexterous grasping via eigengrasps: a low-dimensional approach to a high-complexity problem.

In Robotics: Science and systems manipulation workshop-sensing and adapting to the real world,

Cited by: [§V-A](#S5.SS1.p2.1).

-
[12]
K. Dharmarajan, W. Huang, J. Wu, L. Fei-Fei, and R. Zhang (2025)

Dream2Flow: bridging video generation and open-world manipulation with 3d object flow.

arXiv preprint arXiv:2512.24766.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[13]
N. Gkanatsios, J. Xu, M. Bronars, A. Mousavian, T. Ke, and K. Fragkiadaki (2025)

3D flowmatch actor: unified 3d policy for single- and dual-arm manipulation.

Arxiv.

Cited by: [§II-C](#S2.SS3.p1.1).

-
[14]
T. He, Z. Wang, H. Xue, Q. Ben, Z. Luo, W. Xiao, Y. Yuan, X. Da, F. Castañeda, S. Sastry, et al. (2025)

VIRAL: visual sim-to-real at scale for humanoid loco-manipulation.

arXiv preprint arXiv:2511.15200.

Cited by: [§I](#S1.p2.1).

-
[15]
W. Kabsch (1976)

A solution for the best rotation to relate two sets of vectors.

Foundations of Crystallography 32 (5), pp. 922–923.

Cited by: [§V-B2](#S5.SS2.SSS2.p1.1),
[§V-D](#S5.SS4.p3.1).

-
[16]
N. Karaev, Y. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht (2025)

Cotracker3: simpler and better point tracking by pseudo-labelling real videos.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 6013–6022.

Cited by: [§IV-B](#S4.SS2.p2.1),
[§V-A](#S5.SS1.p3.1),
[§V-D](#S5.SS4.p5.1).

-
[17]
T. Ke, N. Gkanatsios, and K. Fragkiadaki (2024)

3D diffuser actor: policy diffusion with 3d scene representations.

Arxiv.

Cited by: [§II-C](#S2.SS3.p1.1).

-
[18]
Y. Kuang, H. Geng, A. Elhafsi, T. Do, P. Abbeel, J. Malik, M. Pavone, and Y. Wang (2025)

SkillBlender: towards versatile humanoid whole-body loco-manipulation via skill blending.

arXiv preprint arXiv:2506.09366.

Cited by: [§V-C1](#S5.SS3.SSS1.p3.1).

-
[19]
Y. Kuang, Q. Han, D. Li, Q. Dai, L. Ding, D. Sun, H. Zhao, and H. Wang (2024)

Stopnet: multiview-based 6-dof suction detection for transparent objects on production lines.

In 2024 IEEE International Conference on Robotics and Automation (ICRA),

pp. 5389–5396.

Cited by: [§III-C2](#S3.SS3.SSS2.p1.1).

-
[20]
Y. Kuang, J. Ye, H. Geng, J. Mao, C. Deng, L. Guibas, H. Wang, and Y. Wang (2024)

Ram: retrieval-based affordance transfer for generalizable zero-shot robotic manipulation.

arXiv preprint arXiv:2407.04689.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[21]
H. Li, L. Sun, Y. Hu, D. Ta, J. Barry, G. Konidaris, and J. Fu (2025)

Novaflow: zero-shot manipulation via actionable flow from generated videos.

arXiv preprint arXiv:2510.08568.

Cited by: [§-C](#A0.SS3.p1.1),
[§-C](#A0.SS3.p2.1),
[§II-B](#S2.SS2.p1.1),
[§III-A](#S3.SS1.p1.1),
[§IV-A](#S4.SS1.p1.1),
[§IV-A](#S4.SS1.p4.1),
[TABLE I](#S4.T1.2.3.1),
[TABLE I](#S4.T1.2.4.1),
[§V-B2](#S5.SS2.SSS2.p1.1),
[§V-B2](#S5.SS2.SSS2.p2.1),
[§V-B4](#S5.SS2.SSS4.p2.2),
[§V-D](#S5.SS4.p1.1),
[TABLE III](#S5.T3.2.1.2.1).

-
[22]
T. Lin, K. Sachdev, L. Fan, J. Malik, and Y. Zhu (2025)

Sim-to-real reinforcement learning for vision-based dexterous manipulation on humanoids.

arXiv preprint arXiv:2502.20396.

Cited by: [§V-C1](#S5.SS3.SSS1.p3.1).

-
[23]
T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik (2025)

Learning visuotactile skills with two multifingered hands.

In 2025 IEEE International Conference on Robotics and Automation (ICRA),

pp. 5637–5643.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[24]
M. Liu, Z. Pan, K. Xu, K. Ganguly, and D. Manocha (2020)

Deep differentiable grasp planner for high-dof grippers.

arXiv preprint arXiv:2002.01530.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[25]
T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S. Zhu (2021)

Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator.

IEEE Robotics and Automation Letters 7 (1), pp. 470–477.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[26]
W. Liu, Y. Wan, J. Wang, Y. Kuang, X. Shi, H. Li, D. Zhao, Z. Zhang, and H. Wang (2025)

Fetchbot: learning generalizable object fetching in cluttered scenes via zero-shot sim2real.

In 9th Annual Conference on Robot Learning,

Cited by: [§II-C](#S2.SS3.p1.1).

-
[27]
X. Liu, J. Adalibieke, Q. Han, Y. Qin, and L. Yi (2025)

Dextrack: towards generalizable neural tracking control for dexterous manipulation from human references.

arXiv preprint arXiv:2502.09614.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[28]
X. Liu, H. Wang, and L. Yi (2025)

DexNDM: closing the reality gap for dexterous in-hand rotation via joint-wise neural dynamics model.

arXiv preprint arXiv:2510.08556.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[29]
G. Lu, S. Zhang, Z. Wang, C. Liu, J. Lu, and Y. Tang (2024)

Manigaussian: dynamic gaussian splatting for multi-task robotic manipulation.

In European Conference on Computer Vision,

pp. 349–366.

Cited by: [§II-C](#S2.SS3.p1.1).

-
[30]
T. G. W. Lum, M. Matak, V. Makoviychuk, A. Handa, A. Allshire, T. Hermans, N. D. Ratliff, and K. Van Wyk (2024)

Dextrah-g: pixels-to-action dexterous arm-hand grasping with geometric fabrics.

arXiv preprint arXiv:2407.02274.

Cited by: [§II-A](#S2.SS1.p1.1),
[§III-C](#S3.SS3.p1.1),
[§V-A](#S5.SS1.p2.1).

-
[31]
Z. Luo, J. Cao, S. Christen, A. Winkler, K. Kitani, and W. Xu (2024)

Omnigrasp: grasping diverse objects with simulated humanoids.

Advances in Neural Information Processing Systems 37, pp. 2161–2184.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[32]
J. Lyu, Z. Li, X. Shi, C. Xu, Y. Wang, and H. Wang (2025)

Dywa: dynamics-adaptive world action model for generalizable non-prehensile manipulation.

arXiv preprint arXiv:2503.16806.

Cited by: [§III-B](#S3.SS2.p2.6),
[§III-C2](#S3.SS3.SSS2.p1.1).

-
[33]
V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, et al. (2021)

Isaac gym: high performance gpu-based physics simulation for robot learning.

arXiv preprint arXiv:2108.10470.

Cited by: [§-G](#A0.SS7.p1.1),
[§I](#S1.p2.1),
[§V-A](#S5.SS1.p4.1).

-
[34]
Z. Mei, T. Yin, O. Shorinwa, A. Badithela, Z. Zheng, J. Bruno, M. Bland, L. Zha, A. Hancock, J. F. Fisac, et al. (2026)

Video generation models in robotics-applications, research challenges, future directions.

arXiv preprint arXiv:2601.07823.

Cited by: [§II-B](#S2.SS2.p1.1),
[§IV-A](#S4.SS1.p1.1).

-
[35]
M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Muñoz, X. Yao, R. Zurbrügg, N. Rudin, et al. (2025)

Isaac lab: a gpu-accelerated simulation framework for multi-modal robot learning.

arXiv preprint arXiv:2511.04831.

Cited by: [§I](#S1.p2.1).

-
[36]
A. Murali, B. Sundaralingam, Y. Chao, W. Yuan, J. Yamada, M. Carlson, F. Ramos, S. Birchfield, D. Fox, and C. Eppner (2025)

Graspgen: a diffusion-based framework for 6-dof grasping with on-generator training.

arXiv preprint arXiv:2507.13097.

Cited by: [§V-B2](#S5.SS2.SSS2.p1.1).

-
[37]
E. Olson (2011)

AprilTag: a robust and flexible visual fiducial system.

In 2011 IEEE international conference on robotics and automation,

pp. 3400–3407.

Cited by: [§V-A](#S5.SS1.p1.1).

-
[38]
S. Park, H. Bharadhwaj, and S. Tulsiani (2025)

DemoDiffusion: one-shot human imitation using pre-trained diffusion policy.

arXiv preprint arXiv:2506.20668.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[39]
A. Patel, A. Wang, I. Radosavovic, and J. Malik (2022)

Learning to imitate object interactions from internet videos.

arXiv preprint arXiv:2211.13225.

Cited by: [§-C](#A0.SS3.p2.1),
[§II-A](#S2.SS1.p1.1).

-
[40]
S. Patel, S. Mohan, H. Mai, U. Jain, S. Lazebnik, and Y. Li (2025)

Robotic manipulation by imitating generated videos without physical demonstrations.

arXiv preprint arXiv:2507.00990.

Cited by: [§II-B](#S2.SS2.p1.1),
[§III-A](#S3.SS1.p1.1),
[§IV-A](#S4.SS1.p1.1),
[§IV-A](#S4.SS1.p4.1).

-
[41]
C. R. Qi, H. Su, K. Mo, and L. J. Guibas (2017)

Pointnet: deep learning on point sets for 3d classification and segmentation.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 652–660.

Cited by: [§-D](#A0.SS4.p1.1),
[§II-C](#S2.SS3.p1.1),
[§III-B](#S3.SS2.p4.1),
[§III-C1](#S3.SS3.SSS1.p1.2),
[§III-C2](#S3.SS3.SSS2.p2.6).

-
[42]
C. R. Qi, L. Yi, H. Su, and L. J. Guibas (2017)

Pointnet++: deep hierarchical feature learning on point sets in a metric space.

Advances in neural information processing systems 30.

Cited by: [§III-B](#S3.SS2.p4.1),
[§III-C2](#S3.SS3.SSS2.p2.6).

-
[43]
H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik (2023)

In-hand object rotation via rapid motor adaptation.

In Conference on Robot Learning,

pp. 1722–1732.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[44]
H. Qi, B. Yi, S. Suresh, M. Lambeta, Y. Ma, R. Calandra, and J. Malik (2023)

General in-hand object rotation with vision and touch.

In Conference on Robot Learning,

pp. 2549–2564.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[45]
N. Ravi, V. Gabeur, Y. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al. (2024)

Sam 2: segment anything in images and videos.

arXiv preprint arXiv:2408.00714.

Cited by: [§V-A](#S5.SS1.p3.1).

-
[46]
S. Ross, G. Gordon, and D. Bagnell (2011)

A reduction of imitation learning and structured prediction to no-regret online learning.

In Proceedings of the fourteenth international conference on artificial intelligence and statistics,

pp. 627–635.

Cited by: [§III-C2](#S3.SS3.SSS2.p1.1),
[§III-C](#S3.SS3.p1.1),
[§V-A](#S5.SS1.p4.1).

-
[47]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347.

Cited by: [§III-C1](#S3.SS3.SSS1.p1.2),
[§III-C](#S3.SS3.p1.1),
[§V-A](#S5.SS1.p4.1).

-
[48]
K. Shaw, A. Agarwal, and D. Pathak (2023)

Leap hand: low-cost, efficient, and anthropomorphic hand for robot learning.

arXiv preprint arXiv:2309.06440.

Cited by: [§V-A](#S5.SS1.p1.1).

-
[49]
R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. Van Wyk (2024)

Dextrah-rgb: visuomotor policies to grasp anything with dexterous hands.

arXiv preprint arXiv:2412.01791.

Cited by: [§II-A](#S2.SS1.p1.1),
[§III-C](#S3.SS3.p1.1).

-
[50]
R. Singh, K. Van Wyk, P. Abbeel, J. Malik, N. Ratliff, and A. Handa (2025)

End-to-end rl improves dexterous grasping policies.

arXiv preprint arXiv:2509.16434.

Cited by: [§II-A](#S2.SS1.p1.1),
[§V-C1](#S5.SS3.SSS1.p3.1).

-
[51]
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)

Attention is all you need.

Advances in neural information processing systems 30.

Cited by: [§III-C2](#S3.SS3.SSS2.p2.6).

-
[52]
T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025)

Wan: open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314.

Cited by: [§-C](#A0.SS3.p1.1),
[§I](#S1.p3.1),
[§II-B](#S2.SS2.p1.1),
[§V-A](#S5.SS1.p3.1).

-
[53]
W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang (2023)

Unidexgrasp++: improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 3891–3902.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[54]
C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu (2024)

Dexcap: scalable and portable mocap data collection system for dexterous manipulation.

arXiv preprint arXiv:2403.07788.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[55]
J. Wang, Y. Yuan, H. Che, H. Qi, Y. Ma, J. Malik, and X. Wang (2024)

Lessons from learning to spin” pens”.

arXiv preprint arXiv:2407.18902.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[56]
R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang (2022)

Dexgraspnet: a large-scale robotic dexterous grasp dataset for general objects based on simulation.

arXiv preprint arXiv:2210.02697.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[57]
Y. Wang, Z. Xian, F. Chen, T. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan (2023)

Robogen: towards unleashing infinite data for automated robot learning via generative simulation.

arXiv preprint arXiv:2311.01455.

Cited by: [§I](#S1.p2.1).

-
[58]
R. Wen, G. Chen, Z. Cui, M. Du, Y. Gou, Z. Han, L. Huang, M. Lei, Y. Li, Z. Li, et al. (2025)

GR-dexter technical report.

arXiv preprint arXiv:2512.24210.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[59]
T. Wiedemer, Y. Li, P. Vicol, S. S. Gu, N. Matarese, K. Swersky, B. Kim, P. Jaini, and R. Geirhos (2025)

Video models are zero-shot learners and reasoners.

arXiv preprint arXiv:2509.20328.

Cited by: [§I](#S1.p3.1),
[§II-B](#S2.SS2.p1.1).

-
[60]
M. Xu, Z. Xu, Y. Xu, C. Chi, G. Wetzstein, M. Veloso, and S. Song (2024)

Flow as the cross-domain manipulation interface.

arXiv preprint arXiv:2407.15208.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[61]
S. Xu, Y. Chao, L. Bian, A. Mousavian, Y. Wang, L. Gui, and W. Yang (2025)

Dexplore: scalable neural control for dexterous manipulation from reference scoped exploration.

In Conference on Robot Learning,

pp. 2184–2199.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[62]
Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. (2023)

Unidexgrasp: universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 4737–4746.

Cited by: [§-D](#A0.SS4.p1.1),
[§I](#S1.p2.1),
[§II-A](#S2.SS1.p1.1),
[§III-A](#S3.SS1.p4.1),
[§V-B1](#S5.SS2.SSS1.p1.1).

-
[63]
B. Yi, C. M. Kim, J. Kerr, G. Wu, R. Feng, A. Zhang, J. Kulhanek, H. Choi, Y. Ma, M. Tancik, et al. (2025)

Viser: imperative, web-based 3d visualization in python.

arXiv preprint arXiv:2507.22885.

Cited by: [§-A](#A0.SS1.p1.1).

-
[64]
Z. Yin and P. Abbeel (2025)

Lightning grasp: high performance procedural grasp synthesis with contact fields.

arXiv preprint arXiv:2511.07418.

Cited by: [§II-A](#S2.SS1.p1.1).

-
[65]
Z. Yin, C. Wang, L. Pineda, F. Hogan, K. Bodduluri, A. Sharma, P. Lancaster, I. Prasad, M. Kalakrishnan, J. Malik, et al. (2025)

Dexteritygen: foundation controller for unprecedented dexterity.

arXiv preprint arXiv:2502.04307.

Cited by: [§I](#S1.p1.1),
[§II-A](#S2.SS1.p1.1).

-
[66]
Y. Ze, G. Yan, Y. Wu, A. Macaluso, Y. Ge, J. Ye, N. Hansen, L. E. Li, and X. Wang (2023)

Gnfactor: multi-task real robot learning with generalizable neural feature fields.

In Conference on robot learning,

pp. 284–301.

Cited by: [§II-C](#S2.SS3.p1.1).

-
[67]
Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu (2024)

3d diffusion policy: generalizable visuomotor policy learning via simple 3d representations.

arXiv preprint arXiv:2403.03954.

Cited by: [§II-C](#S2.SS3.p1.1).

-
[68]
J. Zhang, H. Liu, D. Li, X. Yu, H. Geng, Y. Ding, J. Chen, and H. Wang (2024)

Dexgraspnet 2.0: learning generative dexterous grasping in large-scale synthetic cluttered scenes.

In 8th Annual Conference on Robot Learning,

Cited by: [§II-A](#S2.SS1.p1.1).

-
[69]
S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan (2025)

Resmimic: from general motion tracking to humanoid whole-body loco-manipulation via residual learning.

arXiv preprint arXiv:2510.05070.

Cited by: [§III-C1](#S3.SS3.SSS1.p3.1).

-
[70]
H. Zhi, P. Chen, S. Zhou, Y. Dong, Q. Wu, L. Han, and M. Tan (2025)

3DFlowAction: learning cross-embodiment manipulation from 3d flow world model.

arXiv preprint arXiv:2506.06199.

Cited by: [§II-B](#S2.SS2.p1.1).

-
[71]
C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta (2025)

Unified world models: coupling video and action diffusion for pretraining on large robotic datasets.

arXiv preprint arXiv:2504.02792.

Cited by: [§III-C2](#S3.SS3.SSS2.p1.1).

-
[72]
H. Zhu, Y. Wang, D. Huang, W. Ye, W. Ouyang, and T. He (2024)

Point cloud matters: rethinking the impact of different observation spaces on robot learning.

Advances in Neural Information Processing Systems 37, pp. 77799–77830.

Cited by: [§II-C](#S2.SS3.p1.1).

### -A Videos and Visualizations

Videos and visualization results can be found in our supplementary video. We use Viser [[[63](#bib.bib74)]] for all the visualizations. We thank the authors for their great work.

### -B Task Specifications

![Figure](x7.png)

*Figure 7: Overview of simulated tasks. Two frames are shown in each column for each task.*

In simulation, we evaluate our method and baselines on six dexterous manipulation tasks, namely Apple2Plate, Pour, Hammer, StackCup, RotateBox, and Sponge2Bowl. We illustrate these tasks in Fig. [7](#A0.F7). Their task objectives are:

-
•

Apple2Plate: Grasp an apple on the table and put it on the plate.

-
•

Pour: Grasp the mug on the table and tilt to pour.

-
•

Hammer: Grasp the hammer on the table and strike forward.

-
•

StackCup: Pick up the cup and stack it onto another cup on the table.

-
•

RotateBox: Pick up the Foam Box on the table and rotate it horizontally 90 degrees in the air.

-
•

Sponge2Bowl: Grasp a thin piece of sponge on the table and put it into the bowl.

In the real world, we evaluate on four tasks, namely LiftToy, Broccoli2Plate, Meat2Bowl, and Pour, as shown in Fig. [5](#S5.F5). Their task objectives are:

-
•

LiftToy: Grasp a toy on the table and lift it to a certain pose in the air.

-
•

Broccoli2Plate: Pick up the broccoli on the table and put it on the plate.

-
•

Meat2Bowl: Pick up the meat on the table and put it into the bowl.

-
•

Pour: Grasp the coffee cup on the table and tilt to pour.

### -C Video Generation and Point Track Extraction

For video generation, we use Wan2.6 [[[52](#bib.bib42)]] with its online platform, and use its native prompt enhancement with Chinese prompts, which show better performance than English prompts [[[21](#bib.bib1)]]. We use Wan’s first frame + language prompt conditioned generation mode and generate 5-second 30-FPS 720P videos.

For 3D point track extraction, we find that using relative depth estimation, rather than metric video depth estimation as in prior work [[[39](#bib.bib72), [21](#bib.bib1)]], yields better results, with greater spatio-temporal consistency and fewer floaters, as shown in Fig. [8](#A0.F8).

![Figure](x8.png)

*Figure 8: Comparison between metric depth estimation and relative depth estimation. Relative depth estimation yields smoother, more spatio-temporally consistent results and fewer floater points.*

### -D RL State Space

The state space $\mathcal{S}$ for the RL teacher policy training includes: joint angles, joint velocities, the last action, joint torques, fingertip states (state denotes 6D pose, linear and angular velocities, hereinafter the same), fingertip forces, hand state, object state, goal pose, 64-dimensional object point cloud feature encoded by a pretrained PointNet [[[41](#bib.bib5), [62](#bib.bib43)]], and fingertip-to-object distance vectors.

### -E RL Curriculum Learning

We detail our three-stage curriculum in Table [IV](#A0.T4).

*TABLE IV: Curriculum settings of our three-stage RL teacher training.*

### -F Reward Function

We detail our reward shaping in Table [V](#A0.T5). $p_{j}^{\mathrm{finger}}$, $p^{\mathrm{obj}}$, $p^{\mathrm{hand}}$, $h_{j}^{\mathrm{finger}}$ represent the 3D position of the finger $j$, the 3D position of the object, the 3D position of the hand palm and the height of the finger $j$, respectively.
And conditions $\mathrm{contact}$, $\mathrm{success}$, and $\mathrm{stay\_success}$ are defined as follows:

| | $$\mathrm{contact}=\begin{cases}1,&\begin{aligned} \sum_{j=1}^{4}d(p_{j}^{\mathrm{finger}},p^{\mathrm{obj}})random plane-height masking strategy for the student policy learning to improve our policy’s robustness to real-world point input.

In student learning, for each environment, we first perform plane masking. Specifically, we randomly sample one plane that crosses the object’s centroid, select one side of it, and then mask out all the points on that side. In this way, we obtain approximately half of the original object points. These point indices are kept the same throughout the whole environment. This process is to simulate the single-view observation in our real-world deployment so that our policy can generalize to varying camera views.

For the remaining points, at each timestep, we then apply height masking. Specifically, we first randomly sample a height ratio in $[0.2,1.0]$, and based on the height, we mask out 90% of the points above this height and 5% of the points below. Finally, we apply a Gaussian noise $\sim\mathcal{N}(0,0.005)$ to the remaining points. This process is to simulate the occlusion between fingers and object points so that our policy can generalize to fewer object points and point noises in the real world.

*TABLE VII: Hyperparameters of the Teacher Policy.*

*TABLE VIII: Hyperparameters of the Student Policy.*

### -I Additional Hyperparameters

Additional hyperparameters of the teacher and student policies are detailed in Table [VII](#A0.T7) and Table [VIII](#A0.T8), respectively.

Generated on Mon Feb 16 19:00:40 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)