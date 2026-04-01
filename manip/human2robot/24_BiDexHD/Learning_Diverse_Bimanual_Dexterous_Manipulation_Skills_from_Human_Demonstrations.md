[# Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations Bohan Zhou School of Computer Science Peking University zhoubh@stu.pku.edu.cn &Haoqi Yuan School of Computer Science Peking University yhq@pku.edu.cn &Yuhui Fu School of Computer Science Peking University fooyuhuii@gmail.com &Zongqing Lu School of Computer Science Peking University BAAI zongqing.lu@pku.edu.cn Corresponding author. ###### Abstract Bimanual dexterous manipulation is a critical yet underexplored area in robotics. Its high-dimensional action space and inherent task complexity present significant challenges for policy learning, and the limited task diversity in existing benchmarks hinders general-purpose skill development. Existing approaches largely depend on reinforcement learning, often constrained by intricately designed reward functions tailored to a narrow set of tasks. In this work, we present a novel approach for efficiently learning diverse bimanual dexterous skills from abundant human demonstrations. Specifically, we introduce BiDexHD, a framework that unifies task construction from existing bimanual datasets and employs teacher-student policy learning to address all tasks. The teacher learns state-based policies using a general two-stage reward function across tasks with shared behaviors, while the student distills the learned multi-task policies into a vision-based policy. With BiDexHD, scalable learning of numerous bimanual dexterous skills from auto-constructed tasks becomes feasible, offering promising advances toward universal bimanual dexterous manipulation. Our empirical evaluation on the TACO dataset, spanning 141 tasks across six categories, demonstrates a task fulfillment rate of 74.59% on trained tasks and 51.07% on unseen tasks, showcasing the effectiveness and competitive zero-shot generalization capabilities of BiDexHD. For videos and more information, visit our project page](https://sites.google.com/view/bidexhd).

## 1 Introduction

Bimanual manipulation is crucial and beneficial.
Humans use both hands to do manipulations like using scissors, tying shoelaces, or operating kitchen utensils.
The ability to manipulate objects with two hands is fundamental for everyday tasks, because with both hands, we can not only do some “symmetry" collaborative tasks like carrying a heavy box with two hands, but also do “asymmetry” tasks [(Liu et al., [2024a](#bib.bib22))] like twisting a bottle cap, which means one hand acts as an auxiliary hand for stabilizing objects and the other acts as an operator.

With the rapid development of embodied artificial intelligence, robotic bimanual dexterous manipulation is getting more and more important in manufacturing, healthcare, agriculture, construction, and tertiary industry [(Zhang et al., [2024b](#bib.bib55))].
This emphasizes the effective use of tools or manipulations over objects that are deformable or of irregular shapes, overcoming the limitations of low-DOF end-effectors like grippers. Moreover, it addresses complicated human-like hand-object interaction and collaboration.
Despite its significance, achieving proficient bimanual manipulation remains a substantial challenge because it severely struggles with the high-dimensional action space.
While a line of previous work [(Grannen et al., [2023](#bib.bib12); Yu et al., [2024](#bib.bib52); Kataoka et al., [2022](#bib.bib19); Liu et al., [2024a](#bib.bib22))] primarily focuses on bimanual manipulation with grippers, there is still much left to explore for bimanual manipulation with dexterous hands.
Previous attempts to solve bimanual dexterous manipulation tasks are mainly based on reinforcement learning (RL) [(Lin et al., [2024](#bib.bib21); Huang et al., [2023](#bib.bib17); Zhang et al., [2024a](#bib.bib54))]. However, they require intricate reward designs tailored to specific manually-designed tasks. Therefore, these approaches lack scalability and generalizability to a broader range of tasks. Recent research [(Sindhupathiraja et al., [2024](#bib.bib44); Fu et al., [2024](#bib.bib9); He et al., [2024](#bib.bib15))] has advanced robotic bimanual dexterous manipulation through teleoperation. Nevertheless, human intervention is inevitable.
We would ask a question:

“Can we learn diverse bimanual dexterous manipulation skills in a unified and scalable way?”

Our solution is to use human demonstrations. Compared to robotic demonstrations, human demonstrations are relatively easier to obtain with haptic gloves or MoCap devices rather than deploying a trained policy, and contain much more physically compliant and human-aligned behavior.
In this paper, we propose a novel approach to learn diverse bimanual dexterous manipulation skills from human demonstrations. Upon this setting, we propose BiDexHD, a unified and scalable framework to automatically turn a human bimanual manipulation dataset into a series of tasks in the simulation and conduct effective policy learning.

BiDexHD does not depend on manually-designed tasks or pre-defined tasks in existing benchmarks. Instead, it consistently constructs feasible tasks from any bimanual manipulation trajectory. Furthermore, we are not required to design specific rewards for each task but instead utilize a unified reward function for reinforcement learning followed by policy distillation.
In a word, BiDexHD breaks the bottleneck of limited tasks and manual designs, which is significant to the further development of general-purpose bimanual dexterous manipulation.
Though promising, several challenges must be addressed to fully realize this. It is essential to figure out how to accurately mimic fine-grained bimanual behaviors from human demonstrations and avoid collisions and disturbances while encouraging smooth trajectories and synchronous collaboration between both hands.
To address this, we carefully design a general two-stage reward function to assign curricula for RL training.

To sum up, our key contributions can be summarized as follows:

-
•

We formalize the problem of learning bimanual dexterous skills from human demonstrations as a preliminary attempt towards universal bimanual skills.

-
•

We propose BiDexHD, a unified and scalable reinforcement learning framework for learning diverse bimanual dexterous manipulation from human demonstrations, advancing the capabilities of robots in performing bimanual cooperative tasks.

-
•

We evaluate BiDexHD across 141 auto-constructed tasks over 6 categories from the TACO [(Liu et al., [2024b](#bib.bib24))] dataset and demonstrate the superior training performance and competitive generalization capabilities of BiDexHD.

## 2 Related Work

### 2.1 Bimanual Dexterous Manipulation

In recent years, the robotics community has increasingly focused on dexterous manipulation due to its remarkable flexibility and human-like dexterity. Researchers have developed methods using dexterous hands for tasks such as in-hand manipulation [(Arunachalam et al., [2023](#bib.bib1); Yin et al., [2023](#bib.bib51); Handa et al., [2023](#bib.bib14); Qi et al., [2023](#bib.bib32); Chen et al., [2023](#bib.bib4); [2022](#bib.bib3))], grasping [(Xu et al., [2023](#bib.bib49); Wan et al., [2023](#bib.bib47); Qin et al., [2023a](#bib.bib35); Ye et al., [2023](#bib.bib50); Qin et al., [2022a](#bib.bib33))], and manipulating deformable objects [(Bai et al., [2016](#bib.bib2); Ficuciello et al., [2018](#bib.bib8); Li et al., [2023](#bib.bib20); Hou et al., [2019](#bib.bib16))]. However, most existing work focuses on a single dexterous hand, revealing the potential of bimanual dexterity. In fact, for humans, bimanual collaboration takes place frequently in daily life such as riding, carrying heavy objects, and using tools. There are heterogeneous research directions towards bimanual dexterous manipulation. Some researchers attempt to settle down to specific tasks via reinforcement learning. For example, recent work [(Lin et al., [2024](#bib.bib21))] investigates twisting lids with two multi-fingered hands, DynamicHandover [(Huang et al., [2023](#bib.bib17))] explores throwing and catching, and ArtiGrasp [(Zhang et al., [2024a](#bib.bib54))] focuses on a few grasping and articulation tasks. [Gbagbe et al. ([2024](#bib.bib11))] leveraged large language models to design a system for bimanual robotic dexterous manipulation, while [Wang et al. ([2024](#bib.bib48))] proposed to solve bimanual grasping via physics-aware iterative learning and prediction of saliency maps. A recent work [(Gao et al., [2024](#bib.bib10))] adopts keypoints-based visual imitation learning to learn bimanual coordination strategies. Unlike existing work, in this paper, we offer a general solution to learn from bimanual demonstrations by designing a unified reward function to learn a state-based policy via reinforcement learning and distilling it into a vision-based policy.

### 2.2 Learning Dexterity From Human Demonstrations

As a sample-efficient data-driven way, learning from human demonstrations has been proven successful in robot learning [(Jia et al., [2024](#bib.bib18); Mandlekar et al., [2023](#bib.bib29); Odesanmi et al., [2023](#bib.bib30))].
Compared with learning dexterity via reinforcement learning which is notoriously challenging for policy learning due to the high degrees of freedom and the necessity of manually designing task-specific reward functions, learning complex dexterous behaviors from diverse accessible human demonstrations [(Smith et al., [2019](#bib.bib46); Schmeckpeper et al., [2020](#bib.bib40); Shao et al., [2021](#bib.bib41))] is a more stable and scalable approach.
A line of previous studies [(Arunachalam et al., [2023](#bib.bib1); Mandikal & Grauman, [2021](#bib.bib27); Sivakumar et al., [2022](#bib.bib45); Qin et al., [2022b](#bib.bib34); Mandikal & Grauman, [2022](#bib.bib28); Liu et al., [2023](#bib.bib23); Shaw et al., [2023b](#bib.bib43); Chen et al., [2024](#bib.bib5))] explicitly leverages human demonstrations to facilitate the acquisition of dexterous manipulation skills mainly by human-robot-arm-hand retargeting and imitation learning. However, these studies predominantly focus on single-hand manipulation and are often limited to tasks such as in-hand manipulation [(Arunachalam et al., [2023](#bib.bib1))] or video-conditioned teleoperation [(Sivakumar et al., [2022](#bib.bib45))].
With the recent advent of diverse and comprehensive human bimanual manipulation datasets [(Zhan et al., [2024](#bib.bib53); Liu et al., [2024b](#bib.bib24); Fan et al., [2023](#bib.bib7); Razali & Demiris, [2023](#bib.bib37))] which naturally provide a rich resource for high-quality posture sequences of dual hands and bimanual interaction with diverse real objects, a lot of bimanual manipulation tasks can be automatically defined. Thus, in this work, we aim to address more challenging and general bimanual dexterous skill learning purely based on automatically constructed tasks from human demonstrations.

## 3 Preliminaries

### 3.1 Task Formulation

Dec-POMDP. We formulate each bimanual manipulation task as a decentralized partially observable Markov decision process (Dec-POMDP). The Dec-POMDP can be represented
by the tuple $Z=\left(\mathcal{N},\mathcal{M},S,\bm{O},\bm{A},P,R,\rho,\gamma\right)$. Dual hands with arms are separated as $\mathcal{N}$ agents, which is represented by set $\mathcal{M}$. The proprioception of robots and the information about objects are initialized at $s_{0}\in S$ according to the initial state distribution $\rho(s_{0})$. At each time step $t$, the state is represented by $s_{t}$, and the $i$-th agent receives an observation $o_{t}^{i}\in\bm{O}$ based on $s_{t}$. Subsequently, the policy of the $i$-th agent, $\pi_{i}\in\bm{\Pi}$, takes $o_{t}^{i}$ as input and outputs an action $a_{t}^{i}\in A^{i}$. The joint action of all agents is denoted by $\bm{a}_{t}\in\bm{A}$, where $\bm{A}=A^{1}\times A^{2}\times\cdots A^{\mathcal{N}}$.
The state transits to the next state according to the transition function $s_{t+1}\sim P(s_{t+1}|s_{t},\bm{a}_{t})$. After this, the $i$-th agent receives a reward $r_{t}^{i}$ based on the reward function $R(s_{t},\bm{a}_{t})$.
The objective is to find the optimal policy $\bm{\pi}$ that maximizes the expected sum of rewards $\mathbb{E}_{\bm{\pi}}[\sum_{t=0}^{T-1}\gamma^{t}\sum_{i=1}^{\mathcal{N}}r_{t}^{i}]$ over an episode with $T$ time steps, where $\gamma$ is the discount factor.

Environment Setups. The leftmost subgraph in Figure [2](#S4.F2) illustrates the setups for each bimanual manipulation task in IsaacGym [(Makoviychuk et al., [2021](#bib.bib26))]. In general, there are a tool and a target object initialized on a table.
$\mathcal{N}=2$ robotic arms are installed in front of the table, with left LEAP Hand [(Shaw et al., [2023a](#bib.bib42))] mounted on the left arm and right hand on the right arm. The right hand reaches for the tool, and the left hand targets the object. Both hands coordinate to simultaneously move, pick up, and manipulate the objects above the table. Note that our method applies to all dexterous hand embodiments.
The observation space $\bm{O}$ contains robot proprioception and object information.
The left and right policies both output 22 joint angles normalized to $[-1,1]$, and the robots are controlled via position control. See more details in Appendix [B.4](#A2.SS4).

Dataset Preparation. A human bimanual manipulation dataset consists of $M$ trajectories $\mathcal{D}=\{\tau^{1},\tau^{2},\dots,\tau^{M}\}$, each of which describes a human using a tool with his right hand to manipulate a target object with his left hand. The behavior of each trajectory can be recapped with a triplet (action, tool, object). Any triplet belongs to a union $\mathcal{U}=\mathcal{V}\times\Omega\times\Omega$, where $\Omega$ denotes the set of all objects and tools, and $\mathcal{V}$ denotes the set of all human actions. According to different behaviors depicted in $\mathcal{V}$, we can split all the tasks into $|\mathcal{V}|$ categories.
Each trajectory $\tau^{i}=\{\mathbf{h}^{\text{tool}},\mathbf{h}^{\text{object}},\hat{\mathbf{x}}_{t}^{\text{tool}},\hat{\mathbf{q}}_{t}^{\text{tool}},\hat{\mathbf{x}}_{t}^{\text{object}},\hat{\mathbf{q}}_{t}^{\text{object}},\Theta^{\text{left}}_{t},\Theta^{\text{right}}_{t}\}^{i}_{t:1..N}$ involves a pair of meshes of the tool and object from a object mesh set $\mathbf{h}^{\text{tool}},\mathbf{h}^{\text{object}}\in\mathcal{H}$, $N$-step position $\mathbf{x}\in\mathbb{R}^{3}$ and orientation $\mathbf{q}\in\mathbb{R}^{4}$ sequence of the tool and the object, and the pose sequence of hands described in MANO [(Romero et al., [2017](#bib.bib38))] parameters $\Theta$.

### 3.2 Teacher-Student Learning

It is well known [(Chen et al., [2022](#bib.bib3); [2023](#bib.bib4))] that directly learning a multi-task vision-based policy for dexterous hands is extremely challenging. A more popular and scalable approach is teacher-student learning [(Wan et al., [2023](#bib.bib47))], which not only simplifies the complexity of multi-DoF robot multi-task learning but also enhances the efficiency of point cloud-based policy learning. In the teacher learning phase, a single state-based policy is first trained via reinforcement learning, leveraging privileged information to solve multiple similar tasks. Once trained, multiple teacher policies can effectively tackle all tasks.
In the student learning phase, a vision-based student policy is distilled from the teacher policies. A key distinction between teacher and student observations is how object information is represented. While the teacher’s observation space includes precise details about an object’s position, orientation, and linear and angular velocities, the student’s observation relies on point clouds consisting of $P$ sampled points from the object’s surface mesh. In this way, the learned student policy is promising to be deployed in real world to deal with multiple tasks provided that the real point clouds can be constructed from real-time multi-view RGB-D camera system.

## 4 Learning Bimanual Dexterity From Human Demonstrations

### 4.1 Overview

As illustrated in Figure [1](#S4.F1), we propose a scalable three-phase framework. In the first phase, we parallelize the construction of Dec-POMDP bimanual tasks from a human bimanual manipulation dataset within IsaacGym [(Makoviychuk et al., [2021](#bib.bib26))]. After task initialization, the subsequent two phases adopt a teacher-student policy learning framework. Following the approach of [Chen et al. ([2022](#bib.bib3); [2023](#bib.bib4)); Wan et al. ([2023](#bib.bib47))], we utilize Independent Proximal Policy Optimization (IPPO) [(De Witt et al., [2020](#bib.bib6))] during the second phase to independently train state-based teacher policies for constructed bimanual dexterous tasks in parallel. Each expert focuses on a subset of tasks that require similar behaviors. In the final phase, the teacher policies are distilled into a vision-based student policy, integrating skills across related tasks.

*Figure 1: The three-phase framework, BiDexHD, unifies constructing and solving tasks from human bimanual datasets instead of existing benchmarks. In phase one, BiDexHD constructs each bimanual task from a human demonstration. In phase two, BiDexHD learns diverse state-based policies from a generally designed two-stage reward function via multi-task reinforcement learning. A group of learned policies are then distilled into a vision-based policy for inference in phase three.*

### 4.2 Task Construction From Bimanual Dataset

In this work, we primarily focus on bimanual tool using tasks.
Recent datasets [(Liu et al., [2024b](#bib.bib24); Zhan et al., [2024](#bib.bib53); Fan et al., [2023](#bib.bib7); Razali & Demiris, [2023](#bib.bib37))]
capture a wide range of bimanual cooperative behaviors, involving the use of tools to manipulate objects via motion capture and 3D scanning. The rich data, including object pose trajectories and hand-object interaction postures, provides sufficient information to construct feasible bimanual tasks. The task construction process from the bimanual dataset involves data preprocessing and simulation initialization.

Data Preprocessing. We extract the wrist and fingertip pose of dual hands at each timestep $\{V^{\text{side}}_{t},J^{\text{side}}_{t}\}=\text{MANO}(\Theta^{\text{side}}_{t}),\text{ side}\in\{\text{left, right\}}$ with MANO [(Romero et al., [2017](#bib.bib38))] parameters $\Theta=\{\alpha,\beta,\hat{\mathbf{x}}^{\text{w}}\}$, where $\alpha\in\mathbb{R}^{48},\beta\in\mathbb{R}^{10},\text{ and }\hat{\mathbf{x}}^{\text{w}}\in\mathbb{R}^{3}$ represent hand pose, hand shape parameters, and wrist position respectively. $V\in\mathbb{R}^{778\times 3}$ and $J\in\mathbb{R}^{21\times 3}$ represent vertices and joints on a hand respectively. The quaternion of the wrist $\hat{\mathbf{q}}^{\text{w}}\in\mathbb{R}^{4}$ is translated from axis-angle $\beta_{0:3}$. Given that single LEAP Hand [(Shaw et al., [2023a](#bib.bib42))] has only four fingers, we can easily filter the corresponding positions of these $m=4$ fingers in $J$, denoting them as $\mathbf{x}^{\text{ft}}\in\mathbb{R}^{m\times 3}$. In the following sections, $\tau^{i}$ is denoted as:

$$\tau^{i}=\{\hat{\mathbf{x}}_{t}^{\text{tool}},\hat{\mathbf{q}}_{t}^{\text{tool}},\hat{\mathbf{x}}_{t}^{\text{object}},\hat{\mathbf{q}}_{t}^{\text{object}},\hat{\mathbf{x}}^{\text{lw}}_{t},\hat{\mathbf{q}}^{\text{lw}}_{t},\hat{\mathbf{x}}^{\text{rw}}_{t},\hat{\mathbf{q}}^{\text{rw}}_{t},\hat{\mathbf{x}}^{\text{lft}}_{t},\hat{\mathbf{x}}^{\text{rft}}_{t}\}^{i}_{t:1..N}.$$ \tag{1}

Simulation Initialization. After data preprocessing, we can construct bimanual manipulation tasks $\varGamma=\{\mathcal{T}^{1},...,\mathcal{T}^{M}\}$ in Issac Gym in parallel. For each task $\mathcal{T}^{i}$, the mesh of a tool $\mathbf{h}^{\text{tool}}$ and a target object $\mathbf{h}^{\text{object}}$, along with two arms with hands are initialized with a fixed initial observation vector:

| | | $\displaystyle o_{0}^{\text{side}}=\{\left(\mathbf{j,v}\right)^{\text{side}},\left(\mathbf{x,q}\right)^{\text{side,w}},\mathbf{x}^{\text{side,ft}},\left(\mathbf{x,q,v,w,}\text{id}\right)^{\text{obj}}\}_{0}^{\text{side}}$ | | (2) |
|---|---|---|---|---|
| | | $\displaystyle\text{where \ }\text{side, obj}\in\{\left(\text{left, object}\right),\left(\text{right, tool}\right)\}.$ | |

The robot proprioception includes arm-hand joint angles and velocities, wrist poses, and fingertip positions, and object information includes object positions, orientations, linear and angular velocities, and a unique object identifier for multi-task learning. For all tasks, $(\mathbf{j},\mathbf{v})_{0}$ are all reset to zero. The initial states of wrist and fingertips are calculated with forward kinematics accordingly. Except identifiers, the initial observations for all tools and target objects keep unchanged.
It is worth noting that we assume the robot to be right-handed by default, i.e., the left hand handles the target object and the right hand handles the tool. For brevity, the repeated notation $\text{side, obj}\in\{\text{(left, object),(right, tool)\}}$ is omitted in subsequent sections.

To ensure the feasibility of each task, after initialization, we use retargeting optimizers [(Qin et al., [2023b](#bib.bib36))] to map human hand motions to robot hand joint angles and solve inverse kinematics (IK) to determine the robot arm joint angles based on the robot’s palm base pose. By replaying all object-hand trajectories in the simulator, we can easily identify and remove invalid tasks to build up a complete task set $\varGamma$.

### 4.3 Multi-Task State-Based Policy Learning

In the second phase, we focus on learning a multi-task state-based policy for tasks that require similar behaviors. Broadly, these tasks can generally be divided into two stages: first, aligning the simulation state with initial $\tau^{i}_{0}$ of a trajectory, and second, following each step of the trajectory. During the alignment stage, both hands should prioritize approaching their objects as quickly as possible. The left hand learns to grasp or stabilize the target object, while the right hand learns to grasp the tool. Once simulation alignment is achieved, both hands are expected to maintain their hold and follow the pre-defined trajectory derived from the human demonstration dataset to perform the manipulations in sync. The pipeline is illustrated in Figure [2](#S4.F2). We initialize objects and robots at stage zero, finish simulation alignment at stage one, and conduct trajectory tracking at stage two via IPPO to learn state-based policies $\pi_{\theta}^{\text{side}}(\mathbf{a}_{t}^{\text{side}}|o_{t}^{\text{side}},\mathbf{a}_{t-1}^{\text{side}})$ conditioning on the current observation $o_{t}^{\text{side}}=\{\left(\mathbf{j,v}\right)^{\text{side}},\left(\mathbf{x,q}\right)^{\text{side,w}},\mathbf{x}^{\text{side,ft}},\left(\mathbf{x,q,v,w,}\text{id}\right)^{\text{obj}}\}_{t}^{\text{side}}$ and previously executed action $\mathbf{a}_{t-1}^{\text{side}}$ for dual hands.

![Figure](/html/2410.02477/assets/x2.png)

*Figure 2: General two-stage teacher learning. For each task $\mathcal{T}^{i}$, all joint poses are initialized at zero pose and a pair of tool-object are initialized at a fixed pose at stage zero. At stage one, approaching reward $r_{\text{appro}}$ encourages both hands to get close to their grasping centers $\hat{\mathbf{x}}_{\text{gc}}$, and lifting reward $r_{\text{lift}}$ along with extra bonus $r_{\text{bonus}}$ incentivizes moving both objects to thier reference poses respectively. After simulation alignment, dual hands will manipulate objects under the guidance of tracking reward $r_{\text{track}}$.*

Stage 1: Simulation Alignment. The central goal of stage one is to align the state of simulation to the first step in a trajectory by moving the tool and target object from the fixed initial pose to $\tau_{0}$, which serves as an essential yet challenging prerequisite for subsequent trajectory tracking in stage two. Through experiments in Section [5.4](#S5.SS4), we find that it is not feasible to directly acquire dynamic skills from static poses through imitation. Instead, we adopt reinforcement learning to develop skills like grasping, twisting and pushing. Some previous work [(Luo et al., [2024](#bib.bib25); Xu et al., [2023](#bib.bib49))] on grasping prefers introducing additional pre-grasp poses by estimating grasping pose upon manipulated objects. We adopt a simpler but more generalizable approach by learning skills directly from the object poses provided in the dataset.
Specifically, we anchor the first timestep in the dataset as the reference timestep to establish a tool-object reference pose pair for each manipulation task. Stage one is considered complete once both the tool and the object reach the specified pose for a sustained $u$-step duration.
Rewards are carefully designed to encourage the object to be lifted above the table in reference to the filtered reference poses. The total reward consists of an approaching reward, a lifting reward, and a bonus reward.

The approaching reward, $r_{\text{appro}}$, encourages both dexterous hands
to approach and remain close to the object. In other words, the goal is to minimize the distance between the robot’s palm, fingertips, and the grasp center.
Since functional grasping is critical for tool using, we do not simply select the geometric center of the object. Instead, we pre-compute the grasping center $\hat{\mathbf{x}}_{\text{gc}}$ for each tool and object based on the dataset. Specifically, for each task, we use the human-demonstrated wrist and fingertip positions at the reference timestep–$\hat{\mathbf{x}}^{\text{lw}}_{0},\hat{\mathbf{x}}^{\text{rw}}_{0},\hat{\mathbf{x}}^{\text{lft}}_{0},\hat{\mathbf{x}}^{\text{rft}}_{0}$–as anchor points. We then uniformly sample 1024 points from the surface of the object mesh $\mathbf{h}^{\text{tool}},\mathbf{h}^{\text{object}}$ to form a representative point set $\mathcal{P}$ and compute the average grasp center based on the top $L=50$ nearest points.
$r_{\text{appro}}$ penalizes the distance between the wrist, fingertips, and the grasp center, and is defined as

$$\begin{array}{l}r_{\text{appro}}^{\text{side}}=-\lVert\mathbf{x}_{t}^{\text{side,w}}-\hat{\mathbf{x}}_{\text{gc}}^{\text{obj}}\rVert_{2}-w_{r}\sum^{m}{\lVert\mathbf{x}_{t}^{\text{side,ft}}-\hat{\mathbf{x}}_{\text{gc}}^{\text{obj}}\rVert_{2}}\\ \text{where \ }\hat{\mathbf{x}}_{\text{gc}}^{\text{obj}}=\frac{1}{L}\sum{\text{NN}}\left(\mathcal{P},L,\frac{\hat{\mathbf{x}}_{0}^{\text{side,w}}+\sum^{m}\hat{\mathbf{x}}_{0}^{\text{side,ft}}}{m+1}\right).\end{array}$$ \tag{3}

The lifting reward $r_{\text{lift}}$ encourages holding objects tightly in hands and lifting to desired reference poses. As long as the lifting conditions are satisfied, the robots receive a lifting reward $r_{\text{lift}}$ composed of a non-negative linear position reward and a negative quaternion distance reward,

$$\begin{array}{l}r_{\text{lift}}^{\text{side}}=\left\{\begin{array}{l}r_{\text{pos}}^{\text{side}}+w_{q}r_{\text{quat}}^{\text{side}}\quad\ \text{if\,\,}\mathbb{I}\left(\lVert\mathbf{x}_{t}^{\text{side,w}}-\hat{\mathbf{x}}_{\text{gc}}^{\text{obj}}\rVert_{2}\leq\lambda_{\text{w}}\cap\sum^{m}{\lVert\mathbf{x}_{t}^{\text{side,ft}}-\hat{\mathbf{x}}_{\text{gc}}^{\text{obj}}\rVert_{2}\leq\lambda_{\text{ft}}}\right)\\ 0\qquad\qquad\qquad\ \ \text{otherwise}\\ \end{array}\right.\\ \text{where \ }r_{\text{pos}}^{\text{side}}=\max\left(1-\frac{\lVert\mathbf{x}_{t}^{\text{obj}}-\hat{\mathbf{x}}_{0}^{\text{obj}}\rVert_{2}}{\lVert\mathbf{x}_{0}^{\text{obj}}-\hat{\mathbf{x}}_{0}^{\text{obj}}\rVert_{2}},0\right),\ \ r_{\text{quat}}^{\text{side}}=-\mathbb{D}_{\text{quat}}\left(\mathbf{q}_{t}^{\text{obj}},\hat{\mathbf{q}}_{0}^{\text{obj}}\right).\end{array}$$ \tag{4}

Here, $\mathbf{x}_{0}^{\text{object}}$ and $\mathbf{x}_{0}^{\text{tool}}$ respectively represent the initial positions of the target object and tool in the simulator, while $\hat{\mathbf{x}}_{0}$ denotes the first reference position in a human demonstration.

The bonus reward $r_{\text{bonus}}$ incentivizes the target object or the tool to reach and finally stay at their reference poses, which lays a foundation for the second manipulation stage. $r_{\text{bonus}}$ becomes positive only when the distance between an object’s current position and its reference position becomes lower than $\varepsilon_{\text{succ}}$. Stage one is considered successful only if both $r_{\text{bonus}}^{\text{left}}$ and $r_{\text{bonus}}^{\text{right}}$ are positive for at least $u$ consecutive steps. Thus, the bonus reward $r_{\text{bonus}}$ is defined as

$$\begin{array}{c}r_{\text{bonus}}^{\text{side}}=\left\{\begin{array}{l}\frac{1}{1+\lVert\mathbf{x}_{t}^{\text{obj}}-\hat{\mathbf{x}}_{0}^{\text{obj}}\rVert_{2}}\qquad\text{if\,\,}\mathbb{I}\left(\lVert\mathbf{x}_{t}^{\text{obj}}-\hat{\mathbf{x}}_{0}^{\text{obj}}\rVert_{2}\leq\varepsilon_{\text{succ}}\right)\\ 0\qquad\qquad\qquad\quad\text{otherwise}.\\ \end{array}\right.\end{array}$$ \tag{5}

The total alignment reward is the linear weighted sum of the three components.

$$r_{\text{align}}^{\text{side}}=w_{1}r_{\text{appro}}^{\text{side}}+w_{2}r_{\text{lift}}^{\text{side}}+w_{3}r_{\text{bonus}}^{\text{side}}$$ \tag{6}

Stage 2: Trajectory Tracking.
Once stage one is completed, the left hand is securely holding the target object, and the right hand keeps grasping the tool at its desired reference pose. The next step is to maintain the grasp and follow a trajectory to perform the manipulation. To achieve this, we design a more fine-grained exponential reward, $r_{\text{track}}$, which encourages the dexterous hands to precisely track the desired positions at each timestep in a trajectory starting from the reference timestep.
Assuming that human hands are more flexible than robotic hands, we introduce a constant tracking frequency $f$, where $f$ simulation steps correspond to one step in the dataset. Let $\mathbf{\hat{x}}_{i}^{\text{obj}}$ represent the position of a object at $i$-th step in a $l$-step
human-demonstrated trajectory and $\mathbf{x}_{t_{i}}^{\text{obj}}$ represent the object’s position at the corresponding simulation step in IsaacGym. We have $i=\lceil{t_{i}}/{f}\rceil\in[0,l)$, and the tracking reward is defined as

$$r_{\text{track}}^{\text{side}}=\left\{\begin{array}{l}\exp\left(-w_{t}\lVert\mathbf{x}_{t_{i}}^{\text{obj}}-\hat{\mathbf{x}}_{i}^{\text{obj}}\rVert_{2}\right)\quad\,\,\,\text{ if stage 1 succeeds}\\ 0\qquad\qquad\qquad\qquad\qquad\qquad\text{otherwise}.\\ \end{array}\right.$$ \tag{7}

We adopt IPPO to learn a unified policy from the combination of all rewards for the two stages,

$$r_{\text{total}}^{\text{side}}=r_{\text{align}}^{\text{side}}+w_{4}r_{\text{track}}^{\text{side}}.$$ \tag{8}

$r_{\text{total}}$ unifies two stages of bimanual dexterous manipulation, enabling scaling up to multi-task policy learning for a wide range of constructed bimanual tasks.

### 4.4 Vision-Based Policy distillation

We employ DAgger [(Ross et al., [2011](#bib.bib39))], an on-policy imitation learning algorithm, to develop a vision-based policy for each task category $\nu\in\mathcal{V}$, under the supervision of a group of state-based teacher policies. To enhance generalization capabilities for new objects or unseen tasks, we propose transforming the student policy into a trajectory-conditioned in-context policy, denoted as $\pi_{\phi}^{\text{side}}(\mathbf{a}_{t}^{\text{side}}|\mathbf{o}_{t}^{\text{side}},\mathbf{p}^{\text{side}}_{t},\mathbf{a}_{t-1}^{\text{side}})$, where $\mathbf{o}_{t}=\{(\mathbf{j},\mathbf{v})^{\text{side}},(\mathbf{x},\mathbf{q})^{\text{side,w}},\mathbf{x}^{\text{side,ft}},\text{pc}^{\text{obj}}\}_{t}$, $K$-step future pose $\mathbf{p}^{\text{side}}_{t}\in\mathbb{R}^{K\times 3}$, and $\text{pc}_{t}^{\text{obj}}\in\mathbb{R}^{P\times 3}$. Specifically, to get point clouds $\text{pc}_{t}^{\text{tool}}$ and $\text{pc}_{t}^{\text{object}}$, we pre-sample 4096 points from the surface of $\mathbf{h}^{\text{tool}}$ and $\mathbf{h}^{\text{object}}$ for each task during initialization. At each timestep $t$, a subset of points are sampled from the pre-sampled point clouds, transformed according to current object pose and added with Gaussian noise for robustness.
Besides, it is important to note that during DAgger distillation, we augment traditional vision-based policy $\pi_{\phi}^{\text{side}}(\mathbf{a}_{t}^{\text{side}}|\mathbf{o}_{t}^{\text{side}},\mathbf{a}_{t-1}^{\text{side}})$ with next $K$ positions along the object’s trajectory as additional inputs.
This design allows the learned policy to utilize more information about the motion of objects, such as movement direction and speed in the near future, facilitating zero-shot transfer to unfamiliar tasks or objects. Notably, we can easily mask this additional input by setting $K=0$. We further investigate the influence of $K$ future steps in Section [5.4](#S5.SS4).
The whole teacher-student training process is summarized in Appendix [A](#A1). More implementation details can be found in Appendix [B](#A2).

## 5 experiments

### 5.1 Setups

Dataset. We evaluate the effectiveness of BiDexHD on the TACO [(Liu et al., [2024b](#bib.bib24))] dataset, a large-scale bimanual manipulation dataset that encompasses diverse human demonstrations using tools to manipulate target objects in real-world scenarios.
BiDextHD converts 6 categories $\mathcal{V}=\{\text{Dust, Empty, Pour in some, Put out, Skim off, Smear}\}$ of total 141 human demonstrations in the TACO dataset to Dec-POMDP tasks (See Appendix [D](#A4) for task examples). Task diversity and abundance make BiDexHD easy to scale up. All tasks can be separated into 16 semantic groups, each of which gathers a number of similar demonstrations with the same action, the same tool-object category but different tool and object instances.
BiDextHD constructs a task from single demonstration, and thus each semantic group correspond to a semantic subtask. We adopt teacher-student learning to train 16 semantic sub-tasks and distill teacher policies with similar skills into 6 vision-based policies for each category eventually.

To evaluate the effectiveness of the framework as well as the generalizability of the learned policies, we split 80% tasks for training (Train) and the rest 20% unseen tasks for testing. Detailed descriptions of dataset split are provided in Appendix [B.2](#A2.SS2). For each task in the testing set, if the object and tool both occur in the training set it is labeled as a kind of combinational task (Test Comb), and otherwise it is labeled as a new task (Test New).

Metrics. To measure the quality of our constructed tasks, we introduce two metrics $r_{1}$ and $r_{2}$.

-
•

The first is the average success rate $r_{1}$ of stage one. For a number of $n$ episodes, $r_{1}=\frac{1}{n}\sum^{n}_{e=1}\mathbb{I}_{1}^{e}$ averages over the number of episodes that satisfys conditions $\mathbb{I}_{1}$ at stage one.

| | $$\mathbb{I}_{1}:\quad\exists 0•

The second is the average tracking rate $r_{2}$ of stage two. Each task corresponds to $l$-step human-demonstrated trajectory. For each episode, calculate the proportion of steps where two objects both effectively follows their desired poses. $r_{2}$ is the average tracking rate over $n$ episodes.

| | $$r_{2}=\frac{1}{nl}\sum^{n}\sum^{l-1}_{i=0}\mathbb{I}\left(\lVert\mathbf{x}_{t_{i}}^{\text{object}}-\mathbf{x}_{i}^{\text{object}}\rVert_{2}\leq\varepsilon_{\text{track}}\cap\lVert\mathbf{x}_{t_{i}}^{\text{tool}}-\mathbf{x}_{i}^{\text{tool}}\rVert_{2}\leq\varepsilon_{\text{track}}\right)$$ | |
|---|---|---|

It is important to note that $r_{2}$ serves as the primary metric for indicating task completion while $r_{1}$ is an intermediate metric for assessing task progression. Considering the choice of $\varepsilon_{\text{succ}}$ and $\varepsilon_{\text{track}}$ has a non-legligible impact for the reported results, we will discuss the sensitivity of these thresholds in Section [5.4](#S5.SS4). By default, we choose $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$ for evaluation.

### 5.2 Teacher Learning

Upon the framework of BiDexHD, different base RL algorithms can be incorporated. We mainly compare the performance of independent PPO (BiDexHD-IPPO) and centralized PPO (BiDexHD-PPO). For BiDexHD-IPPO, two agents possess their own observations and execute their own actions. For BiDexHD-PPO, a single policy takes as input both observations and is trained to output all actions that maximize the sum of all total rewards in an episode, which essentially transforms a Dec-POMDP task into a POMDP task.

*Table 1: The average success rate of stage 1 and tracking rate of stage 2 during training and evaluation across all tasks constructed from the TACO dataset under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

RL Results.
The first and last rows in the green section of Table [1](#S5.T1) present the average performance across all auto-constructed bimanual tasks. For tasks with seen objects (Train and Test Comb), BiDexHD-IPPO nearly completes stage 1 by successfully reaching the reference poses and maintaining high-quality tracking during stage 2, which demonstrates its impressive scalability across diverse tasks in the TACO dataset. In contrast, BiDexHD-PPO underperforms compared to BiDexHD-IPPO, particularly on tasks with seen objects. This discrepancy arises because BiDexHD-IPPO is more efficient at acquiring robust skills within limited updates by independently learning left and right policies across a wide range of tasks with smaller observation and action spaces. Furthermore, two independent expert policies focusing solely on specific groups of target objects or tools adapt more easily to similar combinational tasks than a single policy that must attend to both. Consequently, we select IPPO as our base RL algorithm. See Appendix [C.1](#A3.SS1) for detailed evaluation results.

When applied to tasks with new objects (Test New), both BiDexHD-IPPO and BiDexHD-PPO experience a noticeable performance decline. The primary reason for this drop is that these approaches incorporate one-hot object labels in observations during state-based training, leading the policy to heavily rely on this information. As a result, during evaluation, the introduction of new labels disrupts decision-making. Therefore, we remove one-hot object labels during policy distillation to enhance generalization.

### 5.3 Ablations on Teacher Learning

We conduct ablation studies focusing on the key designs at stage one during teacher learning.

Alignment Stage.
To demonstrate the necessity of the design of dataset-simulation alignment stage, we compare BiDexHD-IPPO with a more naive version, denoted as (w/o stage-1), which retains only $r_{\text{track}}$ in RL training at stage 2 and maintains a fixed number of free exploration steps at stage 1. The second line in the green section of Table [1](#S5.T1) reveals a significant performance decline. We observe that only 30.5% of relatively easy tasks (See Appendix [C.1](#A3.SS1) for details) achieve positive $r_{1}$ and $r_{2}$, while for the remaining tasks, the success rate of stage 1 and the tracking rate of stage 2 remain at zero. This emphasizes the importance of $r_{\text{align}}$ during stage 1.

Functional Grasping Center.
In BiDexHD, we pre-compute the grasping center $\hat{\mathbf{x}}_{\text{gc}}$ to calculate $r_{\text{appro}}$ in Equation [3](#S4.E3). In this section, we explore replacing the grasping center with the geometric center of an object, denoted as (w/o gc). The results presented in the third line of Table [1](#S5.T1) show a decrease in $r_{1}$ and $r_{2}$, particularly on tasks involving seen objects compared to BiDexHD-IPPO. To further figure out their discrepancy in behavior, we deploy both policies for inference and visualize their grasping poses for a typical task (dust, brush, pan) in Figure [3](#S5.F3). BiDexHD-IPPO tends to align more closely with the calculated grasping centers (red points), exhibiting human-like grasping behavior. In contrast, BiDexHD-IPPO (w/o gc) with geometric centers (green points) struggles to find proper poses for using the brush or holding the pan. In fact, the geometric center of an object does not often fall within areas suitable for manipulation. These findings highlight the significance of incorporating a functional grasping center, particularly for objects that are thin, flat, or equipped with handles.



![[Uncaptioned image]](/html/2410.02477/assets/x3.png)

Figure 3: A comparison of grasping pose during policy deployment between BiDexHD-IPPO (w/o gc) and BiDexHD-IPPO.

Success Bonus.
The fourth line in the green section of Table [1](#S5.T1) investigates whether removing reward $r_{\text{bonus}}$ defined in Equation [5](#S4.E5) will affect performance. We observe a decline in $r_{2}$ on both the training set and unseen tasks involving new objects. We analyze the additional bonus in Equation [5](#S4.E5) effectively signals the transition between the two stages, enhancing the policy’s awareness of task progression.

### 5.4 Student Learning

For the BiDexHD variants, several trained multi-task state-based teacher policies from one task category are distilled into a single vision-based policy, which is then tested on all tasks. We also introduce behavior cloning (BC) as our baseline. To directly learn bimanual skills from a dataset, we employ Dexpilot [(Handa et al., [2020](#bib.bib13))] to retarget human hand motions in the TACO dataset to joint angles for dexterous hands, solving inverse kinematics (IK) for arm joint angles. All joint angles are collected and replayed in IsaacGym [(Makoviychuk et al., [2021](#bib.bib26))] to gather observations. BC learns purely from this static observation-action dataset and is ultimately tested under the same configuration as BiDexHD.

DAgger Results.
The blue section of Table [1](#S5.T1) displays the performance of the vision-based policies. Our BiDexHD-IPPO+DAgger significantly outperforms both PPO variant and BC, achieving a high task completion rate on the training set and an average $r_{2}=51.07\%$ across all unseen tasks (Test Comb and Test New). This evidence indicates the scalability and competitive generalization ability of BiDexHD framework. Among unseen tasks, we observe a slight decline in $r_{2}$ for combinational tasks, while tasks involving new objects show a sharp increase in $r_{2}$. This suggests that the vision-based policy relies more on information from the point clouds, such as shape and local features, rather than specific one-hot identifiers, enabling effective zero-shot generalization. Conversely, BC performs poorly due to the loss of true dynamics in the simulation, often getting confused by unfamiliar observations and stuck in stationary states. This also reflects the challenges associated with our constructed bimanual tasks. Our framework unifies bimanual skill learning through a combination of trial-and-error and distillation, providing a robust and scalable solution to diverse challenging bimanual manipulation tasks. See Appendix [C.2](#A3.SS2) for detailed evaluation results.

*Table 2: The metrics of different $K$ future steps under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 3: The sensitivity analysis of metrics of BiDexHD-IPPO+DAgger to different $\varepsilon$.*

Future Conditioned Steps.
We further examine the selection of $K\in\{0,1,2,5\}$ for future object positions. Specifically, when $K=0$, the vision-based policy relies exclusively on 3D information from object point clouds and the robot’s proprioception. As shown in Table [3](#S5.T3), the performance across different values of $K$ does not vary significantly.
Even when future conditioned steps are masked $(K=0)$, $r_{2}$ only exhibits slight declines of 2.5% on trained tasks and an average of 3.1% on all unseen tasks compared to $K=5$.
This evidence suggests that after the multi-task RL training phase, the teachers have acquired diverse and robust skills, making pure imitation sufficient for a student to achieve acceptable performance. Nonetheless, $K$ future steps provide additional informative and fine-grained, albeit implicit, clues such as motion and intention for more precise tracking.

Discussion.
To investigate the impact of different thresholds on the metrics, we re-evaluate all tasks and report the performance of our BiDexHD-IPPO+DAgger under varying thresholds, $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=\varepsilon\in\{0.05,0.075,0.1\}$ in Table [3](#S5.T3). Notably, stricter metrics have a more pronounced impact on the performance of unseen tasks compared to trained ones, underscoring the challenges of continuous spatial-temporal trajectory tracking in bimanual manipulation tasks. We will focus on addressing more precise behavior tracking in future work.

## 6 Conclusion & Limitations

In this paper, we introduce a novel approach to learning diverse bimanual dexterous manipulation skills that utilizes human demonstrations. Our framework, BiDexHD, automatically constructs bimanual manipulation tasks from existing datasets and employs a teacher-student learning approach for a vision-based policy that can tackle similar tasks. Our main technical contributions include designing a unified two-stage reward function for multi-task RL training and an in-context vision-based policy that enhances generalization capabilities. Experimental results demonstrate that BiDexHD facilitates robust RL training and policy distillation, successfully solves six categories of bimanual dexterous manipulation tasks, and effectively transfers to unseen tasks through zero-shot generalization.

Our work forwards a step toward universal bimanual manipulation skills, and some limitations need to be addressed in future research. Exploring strategies for achieving more precise spatial and temporal tracking is a valuable direction for future work. Additionally, incorporating a wider variety of real-world tasks–such as deformable object manipulation and bimanual handover–could reveal further potential in dynamic collaborative manipulation scenarios with bimanual dexterous hands.

## References

-
Arunachalam et al. (2023)

Sridhar Pandian Arunachalam, Sneha Silwal, Ben Evans, and Lerrel Pinto.

Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation.

In 2023 ieee international conference on robotics and automation (icra)*, pp. 5954–5961. IEEE, 2023.

-
Bai et al. (2016)

Yunfei Bai, Wenhao Yu, and C Karen Liu.

Dexterous manipulation of cloth.

In *Computer Graphics Forum*, 2016.

-
Chen et al. (2022)

Tao Chen, Jie Xu, and Pulkit Agrawal.

A system for general in-hand object re-orientation.

In *Conference on Robot Learning*, pp. 297–307. PMLR, 2022.

-
Chen et al. (2023)

Tao Chen, Megha Tippur, Siyang Wu, Vikash Kumar, Edward Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand dexterous manipulation from depth.

In *Icml workshop on new frontiers in learning, control, and dynamical systems*, 2023.

-
Chen et al. (2024)

Zerui Chen, Shizhe Chen, Cordelia Schmid, and Ivan Laptev.

Vividex: Learning vision-based dexterous manipulation from human videos.

*arXiv preprint arXiv:2404.15709*, 2024.

-
De Witt et al. (2020)

Christian Schroeder De Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip HS Torr, Mingfei Sun, and Shimon Whiteson.

Is independent learning all you need in the starcraft multi-agent challenge?

*arXiv preprint arXiv:2011.09533*, 2020.

-
Fan et al. (2023)

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges.

Arctic: A dataset for dexterous bimanual hand-object manipulation.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 12943–12954, 2023.

-
Ficuciello et al. (2018)

Fanny Ficuciello, Alessandro Migliozzi, Eulalie Coevoet, Antoine Petit, and Christian Duriez.

Fem-based deformation control for dexterous manipulation of 3d soft objects.

In *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 4007–4013. IEEE, 2018.

-
Fu et al. (2024)

Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn.

Humanplus: Humanoid shadowing and imitation from humans.

*arXiv preprint arXiv:2406.10454*, 2024.

-
Gao et al. (2024)

Jianfeng Gao, Zhi Tao, Noémie Jaquier, and Tamim Asfour.

Bi-kvil: Keypoints-based visual imitation learning of bimanual manipulation tasks.

*arXiv preprint arXiv:2403.03270*, 2024.

-
Gbagbe et al. (2024)

Koffivi Fidèle Gbagbe, Miguel Altamirano Cabrera, Ali Alabbas, Oussama Alyunes, Artem Lykov, and Dzmitry Tsetserukou.

Bi-vla: Vision-language-action model-based system for bimanual robotic dexterous manipulations.

*arXiv preprint arXiv:2405.06039*, 2024.

-
Grannen et al. (2023)

Jennifer Grannen, Yilin Wu, Brandon Vu, and Dorsa Sadigh.

Stabilize to act: Learning to coordinate for bimanual manipulation.

In *Conference on Robot Learning*, pp. 563–576. PMLR, 2023.

-
Handa et al. (2020)

Ankur Handa, Karl Van Wyk, Wei Yang, Jacky Liang, Yu-Wei Chao, Qian Wan, Stan Birchfield, Nathan Ratliff, and Dieter Fox.

Dexpilot: Vision-based teleoperation of dexterous robotic hand-arm system.

In *2020 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 9164–9170. IEEE, 2020.

-
Handa et al. (2023)

Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, et al.

Dextreme: Transfer of agile in-hand manipulation from simulation to reality.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 5977–5984. IEEE, 2023.

-
He et al. (2024)

Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi.

Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning.

*arXiv preprint arXiv:2406.08858*, 2024.

-
Hou et al. (2019)

Yew Cheong Hou, Khairul Salleh Mohamed Sahari, and Dickson Neoh Tze How.

A review on modeling of flexible deformable object for dexterous robotic manipulation.

*International Journal of Advanced Robotic Systems*, 16(3):1729881419848894, 2019.

-
Huang et al. (2023)

Binghao Huang, Yuanpei Chen, Tianyu Wang, Yuzhe Qin, Yaodong Yang, Nikolay Atanasov, and Xiaolong Wang.

Dynamic handover: Throw and catch with bimanual hands.

*arXiv preprint arXiv:2309.05655*, 2023.

-
Jia et al. (2024)

Xiaogang Jia, Denis Blessing, Xinkai Jiang, Moritz Reuss, Atalay Donat, Rudolf Lioutikov, and Gerhard Neumann.

Towards diverse behaviors: A benchmark for imitation learning with human demonstrations.

*arXiv preprint arXiv:2402.14606*, 2024.

-
Kataoka et al. (2022)

Satoshi Kataoka, Seyed Kamyar Seyed Ghasemipour, Daniel Freeman, and Igor Mordatch.

Bi-manual manipulation and attachment via sim-to-real reinforcement learning.

*arXiv preprint arXiv:2203.08277*, 2022.

-
Li et al. (2023)

Sizhe Li, Zhiao Huang, Tao Chen, Tao Du, Hao Su, Joshua B Tenenbaum, and Chuang Gan.

Dexdeform: Dexterous deformable object manipulation with human demonstrations and differentiable physics.

*arXiv preprint arXiv:2304.03223*, 2023.

-
Lin et al. (2024)

Toru Lin, Zhao-Heng Yin, Haozhi Qi, Pieter Abbeel, and Jitendra Malik.

Twisting lids off with two hands.

*arXiv preprint arXiv:2403.02338*, 2024.

-
Liu et al. (2024a)

I Liu, Chun Arthur, Sicheng He, Daniel Seita, and Gaurav Sukhatme.

Voxact-b: Voxel-based acting and stabilizing policy for bimanual manipulation.

*arXiv preprint arXiv:2407.04152*, 2024a.

-
Liu et al. (2023)

Qingtao Liu, Yu Cui, Qi Ye, Zhengnan Sun, Haoming Li, Gaofeng Li, Lin Shao, and Jiming Chen.

Dexrepnet: Learning dexterous robotic grasping network with geometric and spatial hand-object representations.

In *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3153–3160. IEEE, 2023.

-
Liu et al. (2024b)

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi.

Taco: Benchmarking generalizable bimanual tool-action-object understanding.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 21740–21751, 2024b.

-
Luo et al. (2024)

Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu.

Grasping diverse objects with simulated humanoids.

*arXiv preprint arXiv:2407.11385*, 2024.

-
Makoviychuk et al. (2021)

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al.

Isaac gym: High performance gpu-based physics simulation for robot learning.

*arXiv preprint arXiv:2108.10470*, 2021.

-
Mandikal & Grauman (2021)

Priyanka Mandikal and Kristen Grauman.

Learning dexterous grasping with object-centric visual affordances.

In *2021 IEEE international conference on robotics and automation (ICRA)*, pp. 6169–6176. IEEE, 2021.

-
Mandikal & Grauman (2022)

Priyanka Mandikal and Kristen Grauman.

Dexvip: Learning dexterous grasping with human hand pose priors from video.

In *Conference on Robot Learning*, pp. 651–661. PMLR, 2022.

-
Mandlekar et al. (2023)

Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox.

Mimicgen: A data generation system for scalable robot learning using human demonstrations.

*arXiv preprint arXiv:2310.17596*, 2023.

-
Odesanmi et al. (2023)

Gbenga Abiodun Odesanmi, Qining Wang, and Jingeng Mai.

Skill learning framework for human–robot interaction and manipulation tasks.

*Robotics and Computer-Integrated Manufacturing*, 79:102444, 2023.

-
Qi et al. (2017)

Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.

Pointnet: Deep learning on point sets for 3d classification and segmentation.

In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 652–660, 2017.

-
Qi et al. (2023)

Haozhi Qi, Ashish Kumar, Roberto Calandra, Yi Ma, and Jitendra Malik.

In-hand object rotation via rapid motor adaptation.

In *Conference on Robot Learning*, pp. 1722–1732. PMLR, 2023.

-
Qin et al. (2022a)

Yuzhe Qin, Hao Su, and Xiaolong Wang.

From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation.

*IEEE Robotics and Automation Letters*, 7(4):10873–10881, 2022a.

-
Qin et al. (2022b)

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang.

Dexmv: Imitation learning for dexterous manipulation from human videos.

In *European Conference on Computer Vision*, pp. 570–587. Springer, 2022b.

-
Qin et al. (2023a)

Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Hao Su, and Xiaolong Wang.

Dexpoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation.

In *Conference on Robot Learning*, pp. 594–605. PMLR, 2023a.

-
Qin et al. (2023b)

Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox.

Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system.

In *Robotics: Science and Systems*, 2023b.

-
Razali & Demiris (2023)

Haziq Razali and Yiannis Demiris.

Action-conditioned generation of bimanual object manipulation sequences.

In *Proceedings of the AAAI conference on artificial intelligence*, 2023.

-
Romero et al. (2017)

Javier Romero, Dimitrios Tzionas, and Michael J. Black.

Embodied hands: Modeling and capturing hands and bodies together.

*ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)*, 36(6), November 2017.

-
Ross et al. (2011)

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell.

A reduction of imitation learning and structured prediction to no-regret online learning.

In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

-
Schmeckpeper et al. (2020)

Karl Schmeckpeper, Oleh Rybkin, Kostas Daniilidis, Sergey Levine, and Chelsea Finn.

Reinforcement learning with videos: Combining offline observations with interaction.

*arXiv preprint arXiv:2011.06507*, 2020.

-
Shao et al. (2021)

Lin Shao, Toki Migimatsu, Qiang Zhang, Karen Yang, and Jeannette Bohg.

Concept2robot: Learning manipulation concepts from instructions and human demonstrations.

*The International Journal of Robotics Research*, 40(12-14):1419–1434, 2021.

-
Shaw et al. (2023a)

Kenneth Shaw, Ananye Agarwal, and Deepak Pathak.

Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning.

*arXiv preprint arXiv:2309.06440*, 2023a.

-
Shaw et al. (2023b)

Kenneth Shaw, Shikhar Bahl, and Deepak Pathak.

Videodex: Learning dexterity from internet videos.

In *Conference on Robot Learning*, pp. 654–665. PMLR, 2023b.

-
Sindhupathiraja et al. (2024)

Siddhanth Raja Sindhupathiraja, AKM Amanat Ullah, William Delamare, and Khalad Hasan.

Exploring bi-manual teleportation in virtual reality.

In *2024 IEEE Conference Virtual Reality and 3D User Interfaces (VR)*, pp. 754–764. IEEE, 2024.

-
Sivakumar et al. (2022)

Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak.

Robotic telekinesis: Learning a robotic hand imitator by watching humans on youtube.

*arXiv preprint arXiv:2202.10448*, 2022.

-
Smith et al. (2019)

Laura Smith, Nikita Dhawan, Marvin Zhang, Pieter Abbeel, and Sergey Levine.

Avid: Learning multi-stage tasks via pixel-level translation of human videos.

*arXiv preprint arXiv:1912.04443*, 2019.

-
Wan et al. (2023)

Weikang Wan, Haoran Geng, Yun Liu, Zikang Shan, Yaodong Yang, Li Yi, and He Wang.

Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 3891–3902, 2023.

-
Wang et al. (2024)

Shiyao Wang, Xiuping Liu, Charlie CL Wang, and Jian Liu.

Physics-aware iterative learning and prediction of saliency map for bimanual grasp planning.

*Computer Aided Geometric Design*, 111:102298, 2024.

-
Xu et al. (2023)

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, et al.

Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 4737–4746, 2023.

-
Ye et al. (2023)

Jianglong Ye, Jiashun Wang, Binghao Huang, Yuzhe Qin, and Xiaolong Wang.

Learning continuous grasping function with a dexterous hand from human demonstrations.

*IEEE Robotics and Automation Letters*, 8(5):2882–2889, 2023.

-
Yin et al. (2023)

Zhao-Heng Yin, Binghao Huang, Yuzhe Qin, Qifeng Chen, and Xiaolong Wang.

Rotating without seeing: Towards in-hand dexterity through touch.

*arXiv preprint arXiv:2303.10880*, 2023.

-
Yu et al. (2024)

Dongjie Yu, Hang Xu, Yizhou Chen, Yi Ren, and Jia Pan.

Bikc: Keypose-conditioned consistency policy for bimanual robotic manipulation.

*arXiv preprint arXiv:2406.10093*, 2024.

-
Zhan et al. (2024)

Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu.

Oakink2: A dataset of bimanual hands-object manipulation in complex task completion.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 445–456, 2024.

-
Zhang et al. (2024a)

Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation.

In *2024 International Conference on 3D Vision (3DV)*, pp. 235–246. IEEE, 2024a.

-
Zhang et al. (2024b)

Tianle Zhang, Dongjiang Li, Yihang Li, Zecui Zeng, Lin Zhao, Lei Sun, Yue Chen, Xuelong Wei, Yibing Zhan, Lusong Li, et al.

Empowering embodied manipulation: A bimanual-mobile robot manipulation dataset for household tasks.

*arXiv preprint arXiv:2405.18860*, 2024b.

## Appendix A Algorithm

*Algorithm 1 BiDexHD framework.*

## Appendix B Implementation Details

### B.1 DataSet Preprocessing

Reference Timestep.
Considering there are a number of useless preparation timesteps before grasping, the reference timestep in Section [4.2](#S4.SS2) is actually chosen based on the first sudden change of the distance between an object and a tool, because the distance between the tool and object almost stays unchanged before grasping.

More Details.
We further align the coordinates of human wrist to the coordinates of robot palm base to ensure the same dual-hand manipulation behavior.
Besides, due to the geometric discrepancy of objects, we found that the initial height of objects differ a lot in different tasks. Therefore, a translation offset in z-axis is added to all poses in the dataset to keep all the object at the same initial height on the same table.

### B.2 Constructed Tasks

Task Composition. Table [4](#A2.T4) describes the detailed task categories, sub-task names, the split of training and testing set and the diversity of tools and target objects.

*Table 4: 141 constructed tasks across 6 categories for BiDexHD. “All” refers to the total number of a kind of sub-task. “Train” refers to the number of tasks in the training set. “Test Comb” and “Test New” refer to the number of tasks in two types of testing sets. “Tool” and “Object” refer to number of objects in the corresponding sub-tasks.*

### B.3 Dexterous Hands

Currently we use LEAP Hands [(Shaw et al., [2023a](#bib.bib42))]. In future work we will introduce more kinds of dexterous hands.

### B.4 Simulation Setup

Two 6-DOF RealMan arms, spaced 0.68 meters apart, are placed in front of a 0.7m table. 16-DOF LEAP Hands are [Shaw et al. ([2023a](#bib.bib42))] mounted on both the left and right arms, with an initial stretching pose. The tool and target object are spaced 0.4m apart horizontally, 0.5m distant from the robotic arm base.

### B.5 Training Details

DAgger Details. To make the student policy learn more efficiently especially at the early training stage, we mix a few imitation samples to DAgger buffer. Specifically, we choose to use the actions labelled by the experts with probability $p=0.05$ and otherwise actions output by the policy itself. This is often proven desirable in practice, as the naive policy may make more mistakes and visit states that are irrelevant at the early training stage with relatively few datapoints [(Ross et al., [2011](#bib.bib39))].

Hyperparameters.
Table [5](#A2.T5) and [6](#A2.T6) outlines the hyperparameters for IPPO and DAgger in BiDexHD respectively.

*Table 5: Hyperparameters of IPPO*

*Table 6: Hyperparameters of DAgger*

### B.6 Model Architecture

Our codebase for RL and DAgger is built upon UniDexGrasp++ [(Wan et al., [2023](#bib.bib47))]. For each state-based policy, we employ five-layer multi-layer perceptrons (MLPs) for both the actor and the critic, featuring hidden layers with dimensions [1024, 1024, 512, 512] and using ELU activation functions. For the vision-based policy, we utilize a simplified PointNet [(Qi et al., [2017](#bib.bib31))] backbone that incorporates two 1D convolutional layers, a mixture of maximum and average pooling operations, and two MLP layers to process the object point cloud, resulting in an output dimension of 128. Both the actor and the critic share the output of this backbone.

### B.7 Computation Resources

we train a state-based IPPO policy for single sub-tasks for around two days, and distill teacher policies into a vision-based policy for each action category for around one day on single 40G A100 GPUs. All the evaluations are done on a 24G Nvidia RTX 4090 Ti GPU for about half an hour.

## Appendix C Evaluation Results

### C.1 Results of Teacher Learning

Tables below record the detailed evaluation results for each sub-task. ‘–’ represents the absence of testing tasks. The last row of each table shows the average results over all sub-tasks.

*Table 7: Detailed Metrics of BiDexHD-PPO for each sub-task under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 8: Detailed Metrics of BiDexHD-IPPO for each sub-task under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 9: Detailed Metrics of BiDexHD-IPPO(w.o. stage-1) for each sub-task under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 10: Detailed Metrics of BiDexHD-IPPO(w.o. gc) for each sub-task under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 11: Detailed Metrics of BiDexHD-IPPO(w.o. bonus) for each sub-task under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

### C.2 Results of Student Learning

Tables below record the detailed evaluation results for each task category. The last row of each table shows the average results over all sub-tasks in all task categories. ‘–’ in the table represents the absence of testing tasks.

*Table 12: Detailed Metrics of BiDexHD-PPO+DAgger for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 13: Detailed Metrics of BiDexHD-IPPO+DAgger(K=5) for task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 14: Detailed Metrics of BiDexHD-IPPO+DAgger(K=5) for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.075$.*

*Table 15: Detailed Metrics of BiDexHD-IPPO+DAgger(K=5) for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.05$.*

*Table 16: Detailed Metrics of BiDexHD-IPPO+DAgger(K=2) for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 17: Detailed Metrics of BiDexHD-IPPO+DAgger(K=1) for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

*Table 18: Detailed Metrics of BiDexHD-IPPO+DAgger(K=0) for each task category under $\varepsilon_{\text{succ}}=\varepsilon_{\text{track}}=0.1$.*

## Appendix D Additional Visualizations

Figures below visualize samples of bimanual human demonstrations and policy deployment of constructed bimanual dexterous manipulation tasks.

![Figure](/html/2410.02477/assets/figs/vis1.png)

*Figure 4: Task visualization of (pour in some, cup, teapot).*

![Figure](/html/2410.02477/assets/figs/vis2.png)

*Figure 5: Task visualization of (empty, bowl, bowl).*

[◄](/html/2410.02476)

[Feeling lucky?](/feeling_lucky)

[Conversion report](/log/2410.02477)
[Report an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2410.02477)
[View original on arXiv](https://arxiv.org/abs/2410.02477)[►](/html/2410.02478)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Nov 5 21:32:25 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)