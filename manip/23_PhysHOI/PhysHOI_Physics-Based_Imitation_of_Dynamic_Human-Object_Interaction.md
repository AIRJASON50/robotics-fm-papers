[# PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction Yinhuai Wang1,2§ Jing Lin3 Ailing Zeng2† Zhengyi Luo4 Jian Zhang1† Lei Zhang2 1Peking University 2International Digital Economy Academy 3Tsinghua University 4Carnegie Mellon University ###### Abstract Humans interact with objects all the time. Enabling a humanoid to learn human-object interaction (HOI) is a key step for future smart animation and intelligent robotics systems. However, recent progress in physics-based HOI requires carefully designed task-specific rewards, making the system unscalable and labor-intensive. This work focuses on dynamic HOI imitation: teaching humanoid dynamic interaction skills through imitating kinematic HOI demonstrations. It is quite challenging because of the complexity of the interaction between body parts and objects and the lack of dynamic HOI data. To handle the above issues, we present PhysHOI, the first physics-based whole-body HOI imitation approach without task-specific reward designs. Except for the kinematic HOI representations of humans and objects, we introduce the contact graph to model the contact relations between body parts and objects explicitly. A contact graph reward is also designed, which proved to be critical for precise HOI imitation. Based on the key designs, PhysHOI can imitate diverse HOI tasks simply yet effectively without prior knowledge. To make up for the lack of dynamic HOI scenarios in this area, we introduce the *BallPlay* dataset that contains eight whole-body basketball skills. We validate PhysHOI on diverse HOI tasks, including whole-body grasping and basketball skills. ††$\S$Work done during an internship at IDEA; † Corresponding authors. * Figure 1: Our method controls physically simulated humanoids to perform various dynamic interaction skills. Our method can imitate human-object interaction across dynamic scenarios without task-specific rewards. Top-to-bottom: Fingertip spin basketball (left), Grasp (right); Pick up and dribble; Walk while dribbling. Project page and video demonstrations: https://wyhuai.github.io/physhoi-page/ \etocdepthtag .tocmtchapter ## 1 Introduction Creating realistic and agile interactions between objects and humanoids is a long-standing and challenging problem in computer animation, robotics, and human-computer interaction (HCI) [57](#bib.bib57), [86](#bib.bib86), [22](#bib.bib22), [64](#bib.bib64)].
Recently, due to advances in deep reinforcement learning and physics simulation, methods that learn humanoid control from demonstrations (e.g., kinematic motions from motion capture data) have been widely used in animation [[[70](#bib.bib70), [69](#bib.bib69), [53](#bib.bib53), [71](#bib.bib71), [72](#bib.bib72), [102](#bib.bib102), [116](#bib.bib116)]] and robotics [[[17](#bib.bib17), [5](#bib.bib5), [81](#bib.bib81), [2](#bib.bib2)]]. However, most existing physics-based imitation learning methods focus on mimicking isolated human motion without considering human-object interactions, especially for whole-body (i.e., full-body with hands) motions, as involving articulated fingers and complex object dynamics can be exceedingly challenging.

There exist HOI tasks conditioned on goals, in which the humanoid is tasked with manipulating objects to reach certain kinematic states [[[1](#bib.bib1), [31](#bib.bib31), [116](#bib.bib116), [6](#bib.bib6)]]. Although they perform well on specific tasks, e.g., playing tennis [[[116](#bib.bib116)]], these methods may find the learning paradigm difficult to generalize to new types of interaction and skills, considering each needs specially designed task rewards. For example, basketball games consist of multiple basic skills: dribbling, shooting, passing, fingertip spinning, etc. It is extremely laborious to design task-specific rewards for all skills.

To solve diverse HOI tasks within a unified solution, we resort to the HOI imitation, i.e., recreating diverse HOI skills in simulation from kinematic demonstrations. It should be noted that the difference between HOI imitation and previous imitation methods [[[69](#bib.bib69), [71](#bib.bib71)]] is that they imitate isolated human motion, while our aim is to recreate motion for both the human and the object. During simulation, the object is passive and can only be controlled indirectly by the humanoid, making this task extremely challenging.

Correspondingly, we introduce the first physics-based whole-body HOI imitation framework, called PhysHOI*, which makes HOI imitation feasible in diverse scenarios. Given a kinematic HOI demonstration sequence, *PhysHOI* can imitate the HOI skill without task-specific knowledge and reward designs. Considering the importance of contact in describing HOI semantics, we introduce a novel general-purpose Contact Graph (CG), where nodes are whole-body humanoid body segments and objects, and each edge denotes the binary contact information between two nodes, to enhance the commonly-used kinematic HOI representation (e.g., the human motion, object motion, and interaction graph [[[122](#bib.bib122)]]). Based on the contact-aware HOI representation, we present a task-agnostic HOI imitation reward that multiplies the kinematic rewards with the proposed contact graph reward (CGR), which effectively eliminates local optima in kinematic-only rewards. To make up for the lack of dynamic HOI data, we introduce the *BallPlay* dataset, which contains eight diverse human-basketball interaction demonstrations with high-quality SMPL-X [[[68](#bib.bib68)]] and object motions, and contact labels.

Our core contributions can be summarized below:

-
•

Aiming at learning dynamic HOI skills with the whole-body humanoids in simulation, we present *PhysHOI*. This first whole-body HOI imitation approach learns interaction skills directly from kinematic HOI demonstrations.

-
•

To make the method task-agnostic towards general HOI learning, we introduce a general-purpose contact graph (CG) as a critical complement. Correspondingly, we design a novel contact graph reward (CGR), which effectively guides whole-body humanoids to manipulate objects or interact with high-dynamic objects precisely without task-specific reward designs.

-
•

We introduce the *BallPlay* dataset to fill the gap in missing dynamic HOI datasets.

We validate the effectiveness of our proposed method comprehensively on five whole-body grasping cases and eight basketball skills, as shown in Fig. [1](#S0.F1) and Fig. [4](#S3.F4). Compared with previous state-of-the-art methods, *PhysHOI* significantly improves the success rate and object trajectory errors. *PhysHOI* is simple yet effective and easy to generalize across diverse types of HOI tasks. We hope this work could pave the path for humanoids to learn general HOI skills.

*Table 1: Comparisons of the most related physics-based methods.*

## 2 Related Work

Related works on physically simulated humanoids can be divided into isolated motion imitation and human-object interaction learning. We compare the most related physics-based work in Tab. [1](#S1.T1) and present the related kinematics-based methods in the Appendix.

Physics-Based Isolated Motion Imitation.
Physics-based methods generate motions via motor control in physics simulation [[[62](#bib.bib62), [97](#bib.bib97), [13](#bib.bib13)]], which addresses the physically implausible artifacts of kinematic methods [[[28](#bib.bib28), [96](#bib.bib96), [120](#bib.bib120), [118](#bib.bib118), [9](#bib.bib9), [56](#bib.bib56)]], *e.g*., penetration, sliding, and floating. Early works rely on hand-crafted [[[35](#bib.bib35)]], model-based [[[12](#bib.bib12)]], and optimization-based [[[94](#bib.bib94)]] controllers, which suffer from motion artifacts and complex parameter tuning. Recently, DeepMimic [[[69](#bib.bib69)]] proposes tracking motion capture sequences via imitation learning and deep reinforcement learning (RL). To make the generation diverse, AMP [[[71](#bib.bib71)]] uses generative adversarial imitation learning (GAIL) [[[33](#bib.bib33)]], which learns the state transition distribution of unstructured motion dataset. Except for RL-based methods, some methods use physics simulation with fully differentiable pipelines [[[78](#bib.bib78), [20](#bib.bib20), [112](#bib.bib112)]]. Some works combine kinematic methods with physics-based imitation policy [[[114](#bib.bib114), [80](#bib.bib80), [60](#bib.bib60), [116](#bib.bib116)]] to obtain physically plausible motions. However, these methods only imitate isolated human motion.

*Figure 2: Framework overview: The proposed pipeline of learning HOI skills from HOI demonstrations. We can obtain kinematic HOI data using mocap devices or estimated from monocular videos. Then, we transfer the HOI data into the reference HOI states, which is a contact-aware HOI representation {human motion, object motion, interaction graph (IG), contact graph (CG)*} for *PhysHOI* to learn. *PhysHOI*: The training process of *PhysHOI* consists of loops of simulation and optimization. Given the simulated HOI state $\boldsymbol{g}_{t}$ and reference HOI state $\boldsymbol{h}_{t+1}$, the policy outputs the action $\boldsymbol{a}_{t}$, then the simulated HOI state will be updated by the physics simulator. For each time step, we calculate the proposed task-agnostic HOI imitation reward, including kinematic rewards and the key CG reward. We train the policy until converges, where it can control simulated humanoids to reproduce the reference HOI skills.*

Physics-based Human-Object Interaction.
Human-object interaction (HOI) faces more challenges and is less explored than isolated motion generation due to the complexity of human body parts, various objects, and diverse interactions.
Compared to kinematics-based methods [[[89](#bib.bib89), [105](#bib.bib105), [75](#bib.bib75)]], physics-based methods are especially advantageous for generating human-object interaction since the physics simulator can explicitly constrain the dynamics of humanoids and objects. Using simulation, one can create realistic human-object interactions such as carrying boxes [[[31](#bib.bib31), [109](#bib.bib109)]], striking [[[72](#bib.bib72)]], climbing ropes [[[1](#bib.bib1)]], grasping [[[6](#bib.bib6)]], and even challenging sports such as skating [[[54](#bib.bib54)]], playing basketball [[[51](#bib.bib51)]], soccer [[[108](#bib.bib108), [38](#bib.bib38)]], football [[[55](#bib.bib55)]] and tennis [[[116](#bib.bib116)]]. However, these methods rely on manual task-specific reward designs, making them difficult to generalize.
Besides, there are active topics on human-scene interaction [[[105](#bib.bib105), [65](#bib.bib65)]] and hand-object interaction [[[21](#bib.bib21), [11](#bib.bib11), [66](#bib.bib66), [111](#bib.bib111), [15](#bib.bib15), [115](#bib.bib115)]].

A highly related work is [[[122](#bib.bib122)]], where the proposed interaction graph [[[32](#bib.bib32)]] is integrated into kinematic rewards to learn the simulated multi-character interaction. Though effective in simple human-object interactions (*e.g*. picking up boxes), the kinematic-only rewards are prone to fall into local optimal when dealing with high-dynamic scenarios.
Another closely related work is [[[6](#bib.bib6)]], a framework for physically plausible whole-body grasping. Similar to previous kinematics-based grasping synthesis methods [[[104](#bib.bib104), [24](#bib.bib24)]], they design task-specific contact rewards as guidance for learning grasp tasks. However, their method contains multiple networks and training rounds with prior learning dedicated to the grasp tasks. Our work explores whole-body object interaction with static or high-dynamic objects (e.g., grasping, playing with balls) without designing task-specific rewards.

## 3 Method

### 3.1 Preliminaries on Reinforcement Learning

Our method is based on reinforcement learning, where the agent interacts with the environment according to a policy to maximize reward. At each time step $t$, the agent takes the system states $\mathbf{s}_{t}$ as input and outputs an action $\mathbf{a}_{t}$ by sampling the policy distribution $\boldsymbol{\pi}(\mathbf{a}_{t}|\mathbf{s}_{t})$. According to the physics simulator $\boldsymbol{f}(\mathbf{s}_{t+1}|\mathbf{a}_{t},\mathbf{s}_{t})$, the new action $\mathbf{a}_{t}$ will result in a new state $\mathbf{s}_{t+1}$. Then a reward $r_{t}=r(\mathbf{s}_{t},\mathbf{a}_{t},\mathbf{s}_{t+1})$ can be calculated, and the goal is to learn a policy that maximizes the expected return $\mathcal{R}(\boldsymbol{\pi})=\mathbb{E}_{p_{\boldsymbol{\pi}}(\boldsymbol{\tau})}\left[\sum_{t=0}^{T-1}\gamma^{t}r_{t}\right]$,
where $\boldsymbol{\tau}=\{\mathbf{s}_{0},\mathbf{a}_{0},r_{0},...,\mathbf{s}_{T-1},\mathbf{a}_{T-1},r_{T-1},\mathbf{s}_{T}\}$ represents the trajectory, $p_{\boldsymbol{\pi}}(\boldsymbol{\tau})$ is the probability density function (PDF) of the trajectory. $T$ denotes the time horizon of a trajectory, and $\gamma\in[0,1)$ is a discount factor. Following prior arts [[[71](#bib.bib71)]], We use PPO [[[79](#bib.bib79)]] to optimize the policy.

### 3.2 Task Definition

Given a reference demonstration of Human-Object Interaction (HOI), represented by a kinematic sequence of human and object states, HOI imitation aims to train a policy to control the simulated humanoid to reproduce the given HOI. Unlike kinematics-based methods [[[89](#bib.bib89), [24](#bib.bib24), [44](#bib.bib44)]], the core challenge here is that the object is not directly controllable. Instead, we can only indirectly manipulate objects by controlling the humanoid. The whole-body humanoid follows the SMPL-X [[[68](#bib.bib68)]] kinematic tree and has a total of 52 body parts and 51$\times$3 DoF actuators where 30$\times$3 DoF is for the hands and 21$\times$3 DoF for the rest of the body. The object is represented as a rigid body with a simplified mesh for collision detection. Details on the HOI data preprocessing can be found in the Appendix.

### 3.3 Overview of PhysHOI

In Fig. [2](#S2.F2), we present the overview of the proposed pipeline for HOI imitation.
The core of *PhysHOI* is the general-purpose contact graph (Sec. [3.4](#S3.SS4)), introducing it into the kinematics-based HOI representation as the contact-aware HOI representation (Sec. [3.5](#S3.SS5)), and the task-agnostic HOI imitation reward (Sec. [3.6](#S3.SS6)).
Similar to prior arts [[[69](#bib.bib69)]], policy training consists of a loop of simulation and optimization: the policy is first used for simulation and then optimized through RL (Sec. [3.1](#S3.SS1)), then the updated policy continues simulation to collect more experiences and repeat this simulation-optimization loop until converges.
Specifically, the state $\boldsymbol{s}_{t}$ (Sec. [3.7](#S3.SS7)) consists of the simulated HOI state $\boldsymbol{g}_{t}$ and the reference HOI state $\hat{\boldsymbol{h}}_{t+1}$ (Sec. [3.5](#S3.SS5)). The policy network (Sec. [3.8](#S3.SS8)) takes the state $\boldsymbol{s}_{t}$ as input and outputs an action $\boldsymbol{a}_{t}\in\mathbb{R}^{51\times 3}$. The PD controller outputs joint torques (omitted in Fig. [2](#S2.F2) for simplicity). Then the physics simulator calculates the updated simulated HOI state $\boldsymbol{g}_{t+1}$.

### 3.4 General-purpose Contact Graph

Contact information is essential for HOI. Though contact forces are hard to acquire, binary contact labels are easy to extract from kinematic HOI data by calculating the mesh collision or distances. In addition, estimating the contact region from images is also a feasible direction [[[98](#bib.bib98)]].
We propose a general-purpose Contact Graph (CG) to enhance the HOI representation across diverse interaction imitations.

Complete CG. As illustrated in Fig. [3](#S3.F3) (a), the CG is a complete graph in which every pair of distinct nodes connected by a unique edge, defined as $\mathcal{G}=\{\mathcal{V},\mathcal{E}\}$, where $\mathcal{V}$ is the set of $k$ nodes and $\mathcal{E}\in\{0,1\}^{k(k-1)/2}$ is the set of edges. The nodes consist of all the objects and humanoid body parts. Each edge stores a binary label that denotes the contact between two nodes, where 1 represents contact, and 0 means no contact. Here, we only use the edge set of CG. The edge value is calculated frame by frame. For an HOI sequence, we can calculate a corresponding CG sequence $\{\mathcal{E}_{t}\}$, which explicitly describes the mutual contact relationship between objects and bodies at different moments. Considering the trend of hinged objects, objects here can also be broken up into finer parts [[[18](#bib.bib18), [23](#bib.bib23), [115](#bib.bib115)]].

Aggregated CG.
There are three limitations in the implementation of the complete CG defined above. First, in the whole-body grasp scenario, the complete CG has 154 nodes (152 body parts, 1 table, and 1 object) and 11781 edges, which is costly in memory and computation. Second, many contact labels may be noisy due to inevitable errors from kinematic HOI estimation and annotation. Third, the contact APIs in existing physics simulation environments [[[62](#bib.bib62)]] are not yet complete. To address these challenges, we propose the aggregated CG, where the graph node can be composed of multiple aggregated parts, as illustrated in Fig. [3](#S3.F3) (b). For example, in basketball scenarios, we can aggregate two hands as one node, the rest of the bodies as a node, and the ball as a node. As long as there is any collision between two aggregated nodes, *e.g*., the fingertip touches the ball; the corresponding edge value becomes one. Contact between parts belonging to the same node is not considered. The aggregated CG significantly simplifies the use of CG and proves to be critical for simulated humanoid learning accurate HOI imitation.

*Figure 3: Contact Graph. (a) The nodes of the complete contact graph consist of all the objects and humanoid body parts. Each edge stores a binary contact label. (b) A node in the aggregated contact graph can contain multiple body parts.*

### 3.5 Contact-Aware HOI Representation

Zhang et al*. [[[122](#bib.bib122)]] introduced kinematics-based representation, including human (subject) motion $\hat{\boldsymbol{s}}^{sbj}_{t}$, object motion $\hat{\boldsymbol{s}}^{obj}_{t}$, interaction graph (IG) $\hat{\boldsymbol{s}}^{ig}_{t}$. We further introduce the proposed CG $\hat{\boldsymbol{s}}^{cg}_{t}$ to represent the reference HOI state $\hat{\boldsymbol{h}}_{t}$:

$$\hat{\boldsymbol{h}}_{t}=\{\hat{\boldsymbol{s}}^{sbj}_{t},\hat{\boldsymbol{s}}^{obj}_{t},\hat{\boldsymbol{s}}^{ig}_{t},\hat{\boldsymbol{s}}^{cg}_{t}\}.$$ \tag{1}

The human motion $\hat{\boldsymbol{s}}^{sbj}_{t}$ consists of the position $\hat{\boldsymbol{s}}_{t}^{p}\in\mathbb{R}^{52\times 3}$, 6D rotation $\hat{\boldsymbol{s}}_{t}^{r}\in\mathbb{R}^{52\times 6}$, position velocity $\hat{\boldsymbol{s}}_{t}^{pv}\in\mathbb{R}^{52\times 3}$, and rotation velocity $\hat{\boldsymbol{s}}_{t}^{rv}\in\mathbb{R}^{52\times 6}$. The object motion consists of the object position $\hat{\boldsymbol{s}}_{t}^{op}\in\mathbb{R}^{1\times 3}$, object 6D rotation $\hat{\boldsymbol{s}}_{t}^{or}\in\mathbb{R}^{1\times 6}$, object position velocity $\hat{\boldsymbol{s}}_{t}^{opv}\in\mathbb{R}^{1\times 3}$, and object rotation velocity $\hat{\boldsymbol{s}}_{t}^{orv}\in\mathbb{R}^{1\times 6}$. All velocities here are calculated via discrete differences, *e.g*. $\hat{\boldsymbol{s}}_{t}^{opv}=\hat{\boldsymbol{s}}_{t}^{op}-\hat{\boldsymbol{s}}_{t-1}^{op}$. The IG used in this paper is inspired by previous works [[[32](#bib.bib32), [122](#bib.bib122)]], but with a simpler definition: A set of positional vectors pointed from the objects to the contact bodies (the body parts that have been in contact with the object throughout the sequence). The IG $\hat{\boldsymbol{s}}^{ig}_{t}\in\mathbb{R}^{n\times m\times 3}$ can be seen as a relative position representation of the object, where $n$ denotes the contact body numbers and $m$ denotes the object numbers. The CG $\hat{\boldsymbol{s}}^{cg}_{t}\in\{0,1\}^{k(k-1)/2}$, where $k$ denotes the number of CG nodes.

*Figure 4: Our method controls simulated humanoids to perform various basketball skills. Top-to-bottom: 1) Rebound; 2) Single-hand toss and catch; 3) Back dribbling; 4) Cross-leg dribble. We mark red when the object has contact with the humanoid.*

### 3.6 Task-agnostic HOI Imitation Reward

Corresponding to the reference HOI state (Sec. [3.5](#S3.SS5)), the proposed task-agnostic HOI imitation reward consists of four parts: the body motion reward $r_{t}^{b}$, the object motion reward $r_{t}^{o}$, the IG reward $r_{t}^{ig}$, and the CG reward $r_{t}^{cg}$. To obtain balanced reward values, we multiply* these rewards to request that none of them be small:

$$r_{t}=r_{t}^{b}*r_{t}^{o}*r_{t}^{ig}*r_{t}^{cg},$$ \tag{2}

Body Motion Reward
can be formulated as:

$$r_{t}^{b}=r_{t}^{p}*r_{t}^{r}*r_{t}^{pv}*r_{t}^{rv},$$ \tag{3}

where $r_{t}^{p}$, $r_{t}^{r}$, $r_{t}^{pv}$, $r_{t}^{rv}$ are the humanoid position reward, rotation reward, position velocity reward, and rotation velocity reward, formulated below:

$$r_{t}^{p}=\text{exp}(-\lambda^{p}*e_{t}^{p}),\quad e_{t}^{p}=\text{MSE}(\boldsymbol{s}_{t}^{p},\hat{\boldsymbol{s}}_{t}^{p}),$$ \tag{4}

$$r_{t}^{r}=\text{exp}(-\lambda^{r}*e_{t}^{r}),\quad e_{t}^{r}=\text{MSE}(\boldsymbol{s}_{t}^{r},\hat{\boldsymbol{s}}_{t}^{r}),$$ \tag{5}

$$r_{t}^{pv}=\text{exp}(-\lambda^{pv}*e_{t}^{pv}),\quad e_{t}^{pv}=\text{MSE}(\boldsymbol{s}_{t}^{pv},\hat{\boldsymbol{s}}_{t}^{pv}),$$ \tag{6}

$$r_{t}^{rv}=\text{exp}(-\lambda^{rv}*e_{t}^{rv}),\quad e_{t}^{rv}=\text{MSE}(\boldsymbol{s}_{t}^{rv},\hat{\boldsymbol{s}}_{t}^{rv}),$$ \tag{7}

$\hat{\boldsymbol{s}}_{t}^{p}$, $\hat{\boldsymbol{s}}_{t}^{r}$, $\hat{\boldsymbol{s}}_{t}^{pv}$, $\hat{\boldsymbol{s}}_{t}^{rv}$ are the components of the reference HOI state: the humanoid position, rotation, velocity, and rotation velocity. $\boldsymbol{s}_{t}^{p}$, $\boldsymbol{s}_{t}^{r}$, $\boldsymbol{s}_{t}^{pv}$, $\boldsymbol{s}_{t}^{rv}$ are calculated from simulation and is aligned with the reference HOI state in format. $\lambda^{p}$, $\lambda^{r}$, $\lambda^{pv}$, $\lambda^{rv}$ are hyperparameters that conditions the sensitivity.

Object Motion Reward. Similar to the human motion reward, the object motion reward is:

$$r_{t}^{o}=r_{t}^{op}*r_{t}^{or}*r_{t}^{opv}*r_{t}^{orv},$$ \tag{8}

where $r_{t}^{op}$, $r_{t}^{or}$, $r_{t}^{opv}$, $r_{t}^{orv}$ are the object position reward, rotation reward, position velocity reward, and rotation velocity reward, respectively. The calculation of these rewards is similar to the Eq. [4](#S3.E4), with $\lambda^{op}$, $\lambda^{or}$, $\lambda^{opv}$, $\lambda^{orv}$ denotes the hyperparameters that condition the reward sensitivity.

Interaction Graph Reward. The reward for the IG is:

$$r_{t}^{ig}=\text{exp}(-\lambda^{ig}*e_{t}^{ig}),\quad e_{t}^{ig}=\text{MSE}(\boldsymbol{s}_{t}^{ig},\hat{\boldsymbol{s}}_{t}^{ig}),$$ \tag{9}

where $\hat{\boldsymbol{s}}_{t}^{ig}$ is the reference and $\boldsymbol{s}_{t}^{ig}$ is calculated from current simulation. Note that the body motion reward $r_{t}^{b}$, object motion reward $r_{t}^{o}$, and IG reward $r_{t}^{ig}$ are all measurements of kinematic properties. We call them kinematic rewards.

Contact Graph Reward.
Though kinematic rewards can handle simple interactions, they usually do not handle dynamic scenarios and are not robust to noisy data, *e.g*., during the training of grasp tasks, the contact between the humanoid and the object will, in most cases, cause the object to move away from the desired trajectory and the expected return becomes smaller. In this case, the policy may learn not to touch the object and falls into a local optimal, as demonstrated in Fig. [7](#S5.F7) (8). We observe that these issues are highly related to unwanted contacts. To encourage accurate contact with the object, we design the contact graph (Sec. [3.4](#S3.SS4)) and a corresponding contact graph reward (CGR) to guide the humanoid to learn correct contact. The CG error is defined as

$$\boldsymbol{e}_{t}^{cg}=|\boldsymbol{s}_{t}^{cg}-\hat{\boldsymbol{s}}_{t}^{cg}|,$$ \tag{10}

where $|\cdot|$ represent element wise absolute value. The CGR is measured by CG error, with independent weights on different edges:

$$r_{t}^{cg}=\text{exp}(-\sum_{j=1}^{J}\boldsymbol{\lambda}^{cg}[j]*\boldsymbol{e}_{t}^{cg}[j]),$$ \tag{11}

where $J=k(k-1)/2$ is the CG edge numbers; $\boldsymbol{e}_{t}^{cg}[j]$ is the $j$th element of $\boldsymbol{e}_{t}^{cg}\in\{0,1\}^{J}$, a binary label representing a CG edge error; $\boldsymbol{\lambda}^{cg}[j]$ is the $j$th element of $\boldsymbol{\lambda}^{cg}\in\mathbb{R}^{J}$, a hyperparameter controls the sensitivity of a CG edge.
We multiply the CGR with the previous kinematic rewards. With the guidance of CGR, the humanoid learns the correct contact and avoids the local optimal of kinematic rewards.
The ablation study in Fig. [7](#S5.F7) and Tab. [2](#S5.T2) justifies our design.

### 3.7 State

The state $\boldsymbol{s}_{t}$ is the input of the policy, consisting of two parts: the simulated HOI state $\boldsymbol{g}_{t}$ and the reference HOI state $\hat{\boldsymbol{h}}_{t+1}$ (Sec. [3.5](#S3.SS5)):

$$\boldsymbol{s}_{t}=\{\boldsymbol{g}_{t},\hat{\boldsymbol{h}}_{t+1}\},\vspace{-0.4cm}$$ \tag{12}

where the reference HOI state $\hat{\boldsymbol{h}}_{t+1}$ is the target that we expect the humanoid and objects to reach in the next time step.
The simulated HOI state represents the current humanoid and object state under simulation and is necessary for the policy to sense the current environment. Similar to previous works [[[69](#bib.bib69)]], we transform all coordinates into the root local coordinate of the humanoid, simplifying data distribution and making it more conducive to learning. For the humanoid, we observe its global root height, local body position, rotation, position velocity, and rotation velocity. These representations form the humanoid (subject) observation $\boldsymbol{o}^{sbj}_{t}$. In addition, we detect net contact forces $\boldsymbol{o}^{f}_{t}$ for the contact bodies (e.g., fingers), which helps to identify contact and accelerate training. For objects, we observe their local position, rotation, velocity, and rotation velocity, which form the object observation $\boldsymbol{o}^{obj}_{t}$. The simulated HOI state can be formulated as:

$$\boldsymbol{g}_{t}=\{\boldsymbol{o}^{sbj}_{t},\boldsymbol{o}^{f}_{t},\boldsymbol{o}^{obj}_{t}\}.$$ \tag{13}

### 3.8 Policy and Action

We follow the actor-critic framework widely used by prior arts [[[71](#bib.bib71), [72](#bib.bib72)]]. The policy output is modeled as a Gaussian distribution of dimensions $51\times 3$ with constant variance, and the mean is modeled by a two-layer MLP of [1024, 512] units and ReLU activations. The action $\boldsymbol{a}_{t}\in\mathbb{R}^{51\times 3}$ sampled from the policy is the target joint rotations for the PD controller. The PD controller adjusts and outputs the joint torques to reach the target rotations.

### 3.9 Simulation Setting

The simulation and the PD controller run at 60 Hz. The policy is sampled at 30Hz. The humanoid and objects need initialization when the simulation starts. We use fixed initialization by default; that is, we extract the rotations and root positions from the first reference frame to initialize the humanoids and objects. We do not use random initialization since the HOI data may have severe collisions that eject the object. We use early termination depending on the time and kinematic state errors. See the Appendix for details.

## 4 The BallPlay Dataset

To compensate for the lack of dynamic HOI scenarios, we introduce the *BallPlay* dataset containing eight whole-body basketball skills, including *back dribble, cross leg, hold, fingertip spin, pass, backspin, cross, and rebound*, as shown in Fig. [5](#S4.F5). Instead of using MoCap devices that are costly and hard to scale up, we apply a monocular annotation solution to estimate the high-quality human SMPL-X parameters and object translations from RGB videos. However, annotating these videos with high-speed and dynamic movements and complex interactions in the 3D camera coordinate is quite challenging.
Inspired by the whole-body annotation pipeline of Motion-X [[[52](#bib.bib52)]], our automatic annotation additionally introduces depth estimation [[[3](#bib.bib3)]], semantic segmentation [[[26](#bib.bib26)]], and CAD model selection to obtain high-quality whole-body human motions and object motions.
Besides, we manually annotate three key contact frames for each video to help the object depth estimation.
The dataset contains videos, estimated HOI sequences, contact labels, and physically rectified HOI sequences using *PhysHOI*. Annotation details can be found in the Appendix.

*Figure 5: The BallPlay* dataset. We show the eight HOI demonstrations of high-dynamic basketball skills. For each skill, the upper rows show the real-life videos, and the lower rows give the estimated whole-body SMPL-X human model and object mesh.*

*Figure 6: Qualitative results on HOI imitation. *means re-implemented methods. Previous methods that use kinematic-only rewards fail to reproduce the interaction accurately, e.g., the ball falls or the grasp fails. Guided by the contact graph, our method yields successful HOI imitation. We mark the object red when it has contact with the humanoid. We outline in red the frame where the failure begins.*

## 5 Experiment

### 5.1 Evaluation on HOI Imitation

Datasets.
We evaluate PhysHOI* on two types of HOI datasets: GRAB [[[92](#bib.bib92)]] and our proposed BallPlay (Sec. [4](#S4)).
We chose 5 cases from the GRAB S8 subset, including *grasping cube, cylinder, flashlight, flute, and bottle*.

Metrics.
We report three types of metrics: 1) the success rate (Succ) of HOI imitation, where the success is defined per frame, deeming imitation successful when the object position and body position errors are both under the thresholds and the contact graph edge value is correct. The Succ is calculated by averaging the success values of all frames. The object threshold is defined as 0.2$m$. The body threshold is defined as 0.1$m$.
The Succ reflects whether the humanoid can accurately imitate the reference body motion to interact with the object correctly. 2) the mean per-joint position error (MPJPE) of the humanoid ($E_{\text{b-mpjpe}}$) and object ($E_{\text{o-mpjpe}}$) to evaluate the positional tracking performance (in $mm$) following [[[60](#bib.bib60)]].
3) the contact accuracy $E_{\text{cg}}$, ranging from 0 to 1, defined as $\frac{1}{N}\sum_{t=1}^{N}\text{MSE}(\boldsymbol{s}_{t}^{cg},\hat{\boldsymbol{s}}_{t}^{cg})$ where N is the total frames of the reference HOI data.

*Figure 7: Ablation on Contact Graph Reward (CGR). Without CGR, the humanoid is easy to fall into local optimal, e.g., (2) using the head to help control the ball, (5) using the wrist to touch the ball, (8) being afraid of catching objects, (11) supporting the table to keep balance. In comparison, the use of CGR effectively guides the humanoid toward correct interaction, as shown in (3), (6), (9), and (12).*

*Table 2: Quantitative results on HOI imitation. Our method yields superior object control and success rate on HOI imitation. *means re-implemented methods. We repeat all sequences 10 times, report the average values, and bold the best score.*

*Table 3: Ablation on rewards. The interaction graph reward $r_{t}^{ig}$ can refine the interaction and motion imitation quality. The contact graph reward $r_{t}^{cg}$ significantly improves the overall performance, especially the success rate, on HOI imitation.*

Baseline Methods.
To the best of our knowledge, our proposed *PhysHOI* is the first whole-body HOI imitation method. For fair comparisons, we make proper adaptations for previous human motion imitation methods to experiment on HOI imitation tasks: (1) DeepMimic [[[69](#bib.bib69)]] is a motion imitation method using additive motion rewards; we add the object motion reward to it; (2) AMP [[[71](#bib.bib71)]] is an unaligned motion imitation method, we modify its reference state as our proposed contact-aware HOI representation from Sec.[3.5](#S3.SS5); (3) We implement a simplified version of Zhang’s method [[[122](#bib.bib122)]] via our interaction graph (Sec. [3.5](#S3.SS5)) since the code is not available yet.

Implementation Details.
We use Isaac Gym [[[62](#bib.bib62)]] as the physics simulation platform. All experiments are trained on a single Nvidia A100 GPU, with 2048 parallel environments and fixed simulation initialization. We train 5000 epochs for all experiments on the GRAB dataset, and 15000 epochs for all experiments on the BallPlay dataset. Each epoch contains 10 frames of sequential simulation. We use the aggregated contact graph (CG) for experiments. For GRAB, the CG contains 3 nodes: the table, the object, and the aggregated whole body. For BallPlay, the CG contains 3 nodes: the object, aggregated hands, and the aggregated rest body parts.
We do not consider the basketball rotation since it is not provided, *i.e*., we set $\lambda^{or}$ and $\lambda^{orv}$ as zero for experiments on BallPlay.
The setting of hyperparameters can be found in the Appendix.

Qualitative Results. From Fig. [6](#S4.F6), DeepMimic [[[69](#bib.bib69)]] and AMP [[[71](#bib.bib71)]] achieve reasonable humanoid motion but fail to control the object.
Zhang *et al*. [[[122](#bib.bib122)]] is effective on some interactions but is also prone to local optima due to the lack of contact guidance. We highlight the failure beginning frames in red outlines.
In contrast, the proposed *PhysHOI* introduces the contact graph that makes it able to handle various complex interactions.

Quantitative Results.
Tab. [2](#S5.T2) shows the average metrics on the GRAB subset and the BallPlay dataset. DeepMimic [[[69](#bib.bib69)]] achieves the best score in body motion metrics, *i.e*., $E_{\text{b-mpjpe}}$, since it only learns human motions. It fails to control the object and results in low interaction success rates. Zhang *et al*. [[[122](#bib.bib122)]] achieves better interaction than DeepMimic, but still yields low success rates because only learning kinematic information. Interestingly, their body imitation errors increase severely, and the contact accuracy worsens for dynamic HOI.
In contrast, *PhysHOI* with CG provides effective contact guidance on HOI learning and yields superior success rates.

### 5.2 Ablation on Contact Graph Reward

Tab. [3](#S5.T3) and Fig. [7](#S5.F7) study the effectiveness of CGR. Overall, the use of CGR significantly improves success rates. If it only applies kinematic reward, the humanoid is prone to fall into local optimal. For instance, the kinematic rewards result in decent kinematic metrics on the *Cross leg* data, but it uses false body parts to control the ball, resulting in inferior HOI imitation in Fig. [7](#S5.F7) (5). Similarly, in Fig. [7](#S5.F7) (2), on the *Toss* data, the humanoid wrongly learns to use its head to control the ball. On the contrary, by applying the contact graph reward in Fig. [7](#S5.F7) (3) and (6), the humanoid learns correct contact and avoids incorrect collisions. Another interesting phenomenon is that the HOI generated by our methods is more accurate than the reference. From Fig. [7](#S5.F7) (1), there is a slight floating between the hand and the ball in the reference, while our result fits perfectly.

Considering contact with multiple objects is also necessary. In the grasp tasks, the humanoid sometimes learns to use its hands to support the table to maintain balance if the table is not considered in the contact graph, as shown in Fig. [7](#S5.F7) (11). By involving the table in the contact graph, this local optimal can be well addressed (Fig. [7](#S5.F7) (12)). In summary, comprehensive studies have validated the essence and effectiveness of the proposed CGR.

## 6 Conclusion

We present a framework for learning Human-Object Interaction (HOI) skills from HOI demonstrations. Our method is able to imitate a diverse set of highly dynamic basketball skills. We involve the contact graph (CG) to guide the humanoid toward precise object control, which is proven to be critical for HOI imitation and is effective even when the reference HOI data is highly biased. The proposed task-agnostic HOI imitation reward is effective on diverse HOI types without the need to design task-specific rewards.
We also introduce an HOI dataset called BallPlay is provided to support research on dynamic HOI. We believe this work opens up many exciting directions for future exploration toward general HOI learning. See the Appendix for more discussions about limitations and future work.

## References

-
Bae et al. [2023]

Jinseok Bae, Jungdam Won, Donggeun Lim, Cheol-Hui Min, and Young Min Kim.

Pmp: Learning to physically interact with environments using part-wise motion priors.

*arXiv preprint arXiv:2305.03249*, 2023.

-
Bahl et al. [2022]

Shikhar Bahl, Abhinav Gupta, and Deepak Pathak.

Human-to-robot imitation in the wild.

In *RSS*, 2022.

-
Bhat et al. [2023]

Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller.

Zoedepth: Zero-shot transfer by combining relative and metric depth.

*arXiv preprint arXiv:2302.12288*, 2023.

-
Bhatnagar et al. [2022]

Bharat Lal Bhatnagar, Xianghui Xie, Ilya Petrov, Cristian Sminchisescu, Christian Theobalt, and Gerard Pons-Moll.

BEHAVE: Dataset and method for tracking human object interactions.

In *CVPR*, 2022.

-
Bohez et al. [2022]

Steven Bohez, Saran Tunyasuvunakool, Philemon Brakel, Fereshteh Sadeghi, Leonard Hasenclever, Yuval Tassa, Emilio Parisotto, Jan Humplik, Tuomas Haarnoja, Roland Hafner, et al.

Imitate and repurpose: Learning reusable robot movement skills from human and animal behaviors.

*arXiv preprint arXiv:2203.17138*, 2022.

-
Braun et al. [2024]

Jona Braun, Sammy Christen, Muhammed Kocabas, Emre Aksan, and Otmar Hilliges.

Physically plausible full-body hand-object interaction synthesis.

In *International Conference on 3D Vision (3DV)*, 2024.

-
Brown et al. [2020]

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

Language models are few-shot learners.

2020.

-
Chang et al. [2015]

Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al.

Shapenet: An information-rich 3d model repository.

*arXiv preprint arXiv:1512.03012*, 2015.

-
Chen et al. [2023a]

Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu.

Executing your commands via motion diffusion in latent space.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18000–18010, 2023a.

-
Chen et al. [2023b]

Yixin Chen, Sai Kumar Dwivedi, Michael J Black, and Dimitrios Tzionas.

Detecting human-object contact in images.

In *CVPR*, 2023b.

-
Christen et al. [2022]

Sammy Christen, Muhammed Kocabas, Emre Aksan, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20577–20586, 2022.

-
Coros et al. [2010]

Stelian Coros, Philippe Beaudoin, and Michiel van de Panne.

Generalized biped walking control.

*ACM Transactions on Graphics*, 29(4):1–9, 2010.

-
Coumans and Bai [2016]

Erwin Coumans and Yunfei Bai.

Pybullet, a python module for physics simulation for games, robotics and machine learning.

2016.

-
Dabral et al. [2023]

Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt.

Mofusion: A framework for denoising-diffusion-based motion synthesis.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9760–9770, 2023.

-
Dasari et al. [2023]

Sudeep Dasari, Abhinav Gupta, and Vikash Kumar.

Learning dexterous manipulation from exemplar object trajectories and pre-grasps.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 3889–3896. IEEE, 2023.

-
[16]

Dawson-Haggerty et al.

trimesh.

-
Escontrela et al. [2022]

Alejandro Escontrela, Xue Bin Peng, Wenhao Yu, Tingnan Zhang, Atil Iscen, Ken Goldberg, and Pieter Abbeel.

Adversarial motion priors make good substitutes for complex reward functions.

In *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 25–32. IEEE, 2022.

-
Fan et al. [2023]

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges.

Arctic: A dataset for dexterous bimanual hand-object manipulation.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12943–12954, 2023.

-
Fragkiadaki et al. [2015]

Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik.

Recurrent network models for human dynamics.

In *2015 IEEE International Conference on Computer Vision (ICCV)*, 2015.

-
Fussell et al. [2021]

Levi Fussell, Kevin Bergamin, and Daniel Holden.

Supertrack: Motion tracking for physically simulated characters using supervised learning.

*ACM Transactions on Graphics (TOG)*, 40(6):1–13, 2021.

-
Garcia-Hernando et al. [2020]

Guillermo Garcia-Hernando, Edward Johns, and Tae-Kyun Kim.

Physics-based dexterous manipulations with estimated hand poses and residual reinforcement learning.

In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 9561–9568. IEEE, 2020.

-
Geijtenbeek and Pronost [2012]

Thomas Geijtenbeek and Nicolas Pronost.

Interactive character animation using simulated physics: A state-of-the-art review.

In *Computer graphics forum*, pages 2492–2515. Wiley Online Library, 2012.

-
Geng et al. [2023]

Haoran Geng, Helin Xu, Chengyang Zhao, Chao Xu, Li Yi, Siyuan Huang, and He Wang.

Gapartnet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7081–7091, 2023.

-
Ghosh et al. [2023]

Anindita Ghosh, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, and Philipp Slusallek.

Imos: Intent-driven full-body motion synthesis for human-object interactions.

In *Eurographics*, 2023.

-
Gkioxari et al. [2018]

Georgia Gkioxari, Ross Girshick, Piotr Dollár, and Kaiming He.

Detecting and recognizing human-object interactions.

In *CVPR*, 2018.

-
Grounded-SAM Contributors [2023]

Grounded-SAM Contributors.

Grounded-Segment-Anything: https://github.com/IDEA-Research/Grounded-Segment-Anything, 2023.

-
Guo et al. [2020]

Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng.

Action2motion: Conditioned generation of 3d human motions.

In *Proceedings of the 28th ACM International Conference on Multimedia*, 2020.

-
Guo et al. [2022]

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng.

Generating diverse and natural 3d human motions from text.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5152–5161, 2022.

-
Harvey et al. [2020]

Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal.

Robust motion in-betweening.

*ACM Transactions on Graphics*, 2020.

-
Hassan et al. [2021]

Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and Michael Black.

Stochastic scene-aware motion prediction.

In *ICCV*, 2021.

-
Hassan et al. [2023]

Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, and Xue Bin Peng.

Synthesizing physical character-scene interactions.

In *ACM SIGGRAPH 2022 Conference Proceedings*, 2023.

-
Ho et al. [2010]

Edmond SL Ho, Taku Komura, and Chiew-Lan Tai.

Spatial relationship preserving character motion adaptation.

In *ACM SIGGRAPH 2010 papers*, pages 1–8, 2010.

-
Ho and Ermon [2016]

Jonathan Ho and Stefano Ermon.

Generative adversarial imitation learning.

*Advances in neural information processing systems*, 29, 2016.

-
Ho et al. [2020]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

*Advances in Neural Information Processing Systems (NeurIPS)*, 33, 2020.

-
Hodgins et al. [1995]

Jessica K. Hodgins, Wayne L. Wooten, David C. Brogan, and James F. O’Brien.

Animating human athletics.

In *Proceedings of the 22nd annual conference on Computer graphics and interactive techniques - SIGGRAPH ’95*, 1995.

-
Holden et al. [2016]

Daniel Holden, Jun Saito, and Taku Komura.

A deep learning framework for character motion synthesis and editing.

*ACM Transactions on Graphics*, page 1–11, 2016.

-
Holden et al. [2017]

Daniel Holden, Taku Komura, and Jun Saito.

Phase-functioned neural networks for character control.

*ACM Transactions on Graphics*, page 1–13, 2017.

-
Hong et al. [2019]

Seokpyo Hong, Daseong Han, Kyungmin Cho, Joseph S Shin, and Junyong Noh.

Physics-based full-body soccer motion control for dribbling and shooting.

*ACM Transactions on Graphics (TOG)*, 38(4):1–12, 2019.

-
Hou et al. [2023]

Zhi Hou, Baosheng Yu, and Dacheng Tao.

Compositional 3d human-object neural animation.

*arXiv preprint arXiv:2304.14070*, 2023.

-
Huang et al. [2022]

Yinghao Huang, Omid Taheri, Michael J. Black, and Dimitrios Tzionas.

InterCap: Joint markerless 3D tracking of humans and objects in interaction.

In *GCPR*, 2022.

-
Ji et al. [2021]

Jingwei Ji, Rishi Desai, and Juan Carlos Niebles.

Detecting human-object relationships in videos.

In *ICCV*, 2021.

-
Jiang et al. [2022]

Nan Jiang, Tengyu Liu, Zhexuan Cao, Jieming Cui, Yixin Chen, He Wang, Yixin Zhu, and Siyuan Huang.

CHAIRS: Towards full-body articulated human-object interaction.

*arXiv preprint arXiv:2212.10621*, 2022.

-
Jin et al. [2023]

Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Yang Wei, and Li Yuan.

Act as you wish: Fine-grained control of motion diffusion model with hierarchical semantic graphs.

In *NeurIPS*, 2023.

-
Kapon et al. [2023]

Roy Kapon, Guy Tevet, Daniel Cohen-Or, and Amit H Bermano.

Mas: Multi-view ancestral sampling for 3d motion generation using 2d diffusion.

*arXiv preprint arXiv:2310.14729*, 2023.

-
Kawar et al. [2022]

Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song.

Denoising diffusion restoration models.

In *ICLR Workshop on Deep Generative Models for Highly Structured Data (ICLRW)*, 2022.

-
Kim et al. [2023]

Taeksoo Kim, Shunsuke Saito, and Hanbyul Joo.

NCHO: Unsupervised learning for neural 3d composition of humans and objects.

In *ICCV*, 2023.

-
Lee and Joo [2023]

Jiye Lee and Hanbyul Joo.

Locomotion-action-manipulation: Synthesizing human-scene interactions in complex 3d environments.

*arXiv preprint arXiv:2301.02667*, 2023.

-
Li et al. [2023]

Jiaman Li, Jiajun Wu, and C Karen Liu.

Object motion guided human motion synthesis.

*arXiv preprint arXiv:2309.16237*, 2023.

-
Li et al. [2021]

Ruilong Li, Shan Yang, David A. Ross, and Angjoo Kanazawa.

Learn to dance with aist++: Music conditioned 3d dance generation, 2021.

-
Liang et al. [2022]

Yuanzhi Liang, Qianyu Feng, Linchao Zhu, Li Hu, Pan Pan, and Yi Yang.

Seeg: Semantic energized co-speech gesture generation.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10473–10482, 2022.

-
Libin Liu [August 2018]

Jessica Hodgins Libin Liu.

Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning.

*ACM Transactions on Graphics*, 37(4), August 2018.

-
Lin et al. [2023]

Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang.

Motion-x: A large-scale 3d expressive whole-body human motion dataset.

*Advances in Neural Information Processing Systems*, 2023.

-
Ling et al. [2020]

Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne.

Character controllers using motion vaes.

*ACM Transactions on Graphics*, 2020.

-
Liu and Hodgins [2017]

Libin Liu and Jessica Hodgins.

Learning to schedule control fragments for physics-based characters using deep q-learning.

*ACM Transactions on Graphics*, 36(3), 2017.

-
Liu et al. [2022]

Siqi Liu, Guy Lever, Zhe Wang, Josh Merel, SM Ali Eslami, Daniel Hennes, Wojciech M Czarnecki, Yuval Tassa, Shayegan Omidshafiei, Abbas Abdolmaleki, et al.

From motor control to team play in simulated humanoid football.

*Science Robotics*, 7(69):eabo0235, 2022.

-
Lu et al. [2023]

Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum.

Humantomato: Text-aligned whole-body motion generation.

*arxiv:2310.12978*, 2023.

-
Luck and Aylett [2000]

Michael Luck and Ruth Aylett.

Applying artificial intelligence to virtual reality: Intelligent virtual environments.

*Applied artificial intelligence*, 14(1):3–32, 2000.

-
Luo et al. [2021]

Zhengyi Luo, Ryo Hachiuma, Ye Yuan, and Kris Kitani.

Dynamics-regulated kinematic policy for egocentric pose estimation.

In *Advances in Neural Information Processing Systems*, 2021.

-
Luo et al. [2022]

Zhengyi Luo, Shun Iwase, Ye Yuan, and Kris Kitani.

Embodied scene-aware human pose estimation.

In *Advances in Neural Information Processing Systems*, 2022.

-
Luo et al. [2023]

Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, and Weipeng Xu.

Perpetual humanoid control for real-time simulated avatars, 2023.

-
Mahmood et al. [2019]

Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black.

AMASS: Archive of motion capture as surface shapes.

In *International Conference on Computer Vision*, pages 5442–5451, 2019.

-
Makoviychuk et al. [2021]

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State.

Isaac gym: High performance GPU based physics simulation for robot learning.

In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.

-
Martinez et al. [2017]

Julieta Martinez, Michael J. Black, and Javier Romero.

On human motion prediction using recurrent neural networks.

In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

-
Mourot et al. [2022]

Lucas Mourot, Ludovic Hoyet, François Le Clerc, François Schnitzler, and Pierre Hellier.

A survey on deep learning for skeleton-based human animation.

In *Computer Graphics Forum*, pages 122–157. Wiley Online Library, 2022.

-
Pan et al. [2023]

Liang Pan, Jingbo Wang, Buzhen Huang, Junyu Zhang, Haofan Wang, Xu Tang, and Yangang Wang.

Synthesizing physically plausible human motions in 3d scenes.

*arXiv preprint arXiv:2308.09036*, 2023.

-
Patel et al. [2022]

Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik.

Learning to imitate object interactions from internet videos.

*arXiv preprint arXiv:2211.13225*, 2022.

-
Pavlakos et al. [2019a]

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black.

Expressive body capture: 3D hands, face, and body from a single image.

In *Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*, pages 10975–10985, 2019a.

-
Pavlakos et al. [2019b]

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black.

Expressive body capture: 3d hands, face, and body from a single image.

In *Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2019b.

-
Peng et al. [2018]

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne.

Deepmimic.

*ACM Transactions on Graphics*, page 1–14, 2018.

-
Peng et al. [2020]

Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Edward Lee, Jie Tan, and Sergey Levine.

Learning agile robotic locomotion skills by imitating animals.

In *Robotics: Science and Systems*, 2020.

-
Peng et al. [2021]

Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa.

Amp: Adversarial motion priors for stylized physics-based character control.

*ACM Transactions on Graphics*, page 1–20, 2021.

-
Peng et al. [2022]

Xue Bin Peng, Yunrong Guo, Lina Halper, Sergexuey Levine, and Sanja Fidler.

Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters.

*ACM Trans. Graph.*, 41(4), 2022.

-
Petrov et al. [2023]

Ilya A Petrov, Riccardo Marin, Julian Chibane, and Gerard Pons-Moll.

Object pop-up: Can we infer 3d objects and their poses from human interactions alone?

In *CVPR*, 2023.

-
Petrovich et al. [2021]

Mathis Petrovich, Michael J. Black, and Gul Varol.

Action-conditioned 3d human motion synthesis with transformer vae.

In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.

-
Pi et al. [2023]

Huaijin Pi, Sida Peng, Minghui Yang, Xiaowei Zhou, and Hujun Bao.

Hierarchical generation of human-object interactions with diffusion probabilistic models.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 15061–15073, 2023.

-
Plappert et al. [2016]

Matthias Plappert, Christian Mandery, and Tamim Asfour.

The KIT motion-language dataset.

*Big Data*, 4(4):236–252, 2016.

-
Radford et al. [2019]

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

Language models are unsupervised multitask learners.

2019.

-
Ren et al. [2023]

Jiawei Ren, Cunjun Yu, Siwei Chen, Xiao Ma, Liang Pan, and Ziwei Liu.

Diffmimic: Efficient motion mimicking with differentiable physics.

In *The Eleventh International Conference on Learning Representations*, 2023.

-
Schulman et al. [2017]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

*arXiv preprint arXiv:1707.06347*, 2017.

-
Shi et al. [2023]

Yi Shi, Jingbo Wang, Xuekun Jiang, and Bo Dai.

Controllable motion diffusion model.

*arXiv preprint arXiv:2306.00416*, 2023.

-
Smith et al. [2023]

Laura M. Smith, J. Chase Kew, Tianyu Li, Linda Luu, Xue Bin Peng, Sehoon Ha, Jie Tan, and Sergey Levine.

Learning and adapting agile locomotion skills by transferring experience.

In *Robotics: Science and Systems XIX, Daegu, Republic of Korea, July 10-14, 2023*, 2023.

-
Song et al. [2020a]

Jiaming Song, Chenlin Meng, and Stefano Ermon.

Denoising diffusion implicit models.

*arXiv preprint arXiv:2010.02502*, 2020a.

-
Song et al. [2021]

Jiaming Song, Chenlin Meng, and Stefano Ermon.

Denoising diffusion implicit models.

In *International Conference on Learning Representations (ICLR)*, 2021.

-
Song and Ermon [2019]

Yang Song and Stefano Ermon.

Generative modeling by estimating gradients of the data distribution.

*Advances in Neural Information Processing Systems (NeurIPS)*, 32, 2019.

-
Song et al. [2020b]

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.

Score-based generative modeling through stochastic differential equations.

In *International Conference on Learning Representations (ICLR)*, 2020b.

-
Stacey and Suchman [2012]

Jackie Stacey and Lucy Suchman.

Animation and automation–the liveliness and labours of bodies and machines.

*Body & Society*, 18(1):1–46, 2012.

-
Starke et al. [2019]

Sebastian Starke, He Zhang, Taku Komura, and Jun Saito.

Neural state machine for character-scene interactions.

*ACM Trans. Graph.*, 38(6):209–1, 2019.

-
Starke et al. [2020a]

Sebastian Starke, Yiwei Zhao, Taku Komura, and Kazi Zaman.

Local motion phases for learning multi-contact character movements.

*ACM Transactions on Graphics*, 2020a.

-
Starke et al. [2020b]

Sebastian Starke, Yiwei Zhao, Taku Komura, and Kazi Zaman.

Local motion phases for learning multi-contact character movements.

*ACM Transactions on Graphics (TOG)*, 39(4):54–1, 2020b.

-
Starke et al. [2021]

Sebastian Starke, Yiwei Zhao, Fabio Zinno, and Taku Komura.

Neural animation layering for synthesizing martial arts movements.

*ACM Transactions on Graphics*, page 1–16, 2021.

-
Starke et al. [2022]

Sebastian Starke, Ian Mason, and Taku Komura.

Deepphase.

*ACM Transactions on Graphics*, 41(4):1–13, 2022.

-
Taheri et al. [2020]

Omid Taheri, Nima Ghorbani, Michael J. Black, and Dimitrios Tzionas.

GRAB: A dataset of whole-body human grasping of objects.

In *European Conference on Computer Vision (ECCV)*, 2020.

-
Taheri et al. [2022]

Omid Taheri, Vasileios Choutas, Michael J. Black, and Dimitrios Tzionas.

Goal: Generating 4d whole-body motion for hand-object grasping.

In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

-
Tan et al. [2014]

Jie Tan, Yuting Gu, C. Karen Liu, and Greg Turk.

Learning bicycle stunts.

*ACM Transactions on Graphics*, 33(4):1–12, 2014.

-
Tendulkar et al. [2023]

Purva Tendulkar, Dídac Surís, and Carl Vondrick.

Flex: Full-body grasping without full-body grasps.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 21179–21189, 2023.

-
Tevet et al. [2023]

Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano.

Human motion diffusion model.

In *The Eleventh International Conference on Learning Representations*, 2023.

-
Todorov et al. [2012]

Emanuel Todorov, Tom Erez, and Yuval Tassa.

Mujoco: A physics engine for model-based control.

In *2012 IEEE/RSJ international conference on intelligent robots and systems*, pages 5026–5033. IEEE, 2012.

-
Tripathi et al. [2023]

Shashank Tripathi, Agniv Chatterjee, Jean-Claude Passy, Hongwei Yi, Dimitrios Tzionas, and Michael J. Black.

DECO: Dense estimation of 3D human-scene contact in the wild.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 8001–8013, 2023.

-
Tseng et al. [2023]

Jonathan Tseng, Rodrigo Castellon, and Karen Liu.

Edge: Editable dance generation from music.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 448–458, 2023.

-
Wang et al. [2022]

Xi Wang, Gen Li, Yen-Ling Kuo, Muhammed Kocabas, Emre Aksan, and Otmar Hilliges.

Reconstructing action-conditioned human-object interactions using commonsense knowledge priors.

In *3DV*, 2022.

-
Wang et al. [2023]

Yinhuai Wang, Jiwen Yu, and Jian Zhang.

Zero-shot image restoration using denoising diffusion null-space model.

In *The Eleventh International Conference on Learning Representations*, 2023.

-
Won et al. [2022]

Jungdam Won, Deepak Gopinath, and Jessica Hodgins.

Physics-based character controllers using conditional vaes.

*ACM Transactions on Graphics (TOG)*, 41(4):1–12, 2022.

-
Wu et al. [2022a]

Xiaoqian Wu, Yong-Lu Li, Xinpeng Liu, Junyi Zhang, Yuzhe Wu, and Cewu Lu.

Mining cross-person cues for body-part interactiveness learning in hoi detection.

In *ECCV*, 2022a.

-
Wu et al. [2022b]

Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang.

Saga: Stochastic whole-body grasping with contact.

In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2022b.

-
Xiao et al. [2023]

Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang.

Unified human-scene interaction via prompted chain-of-contacts.

*arXiv preprint arXiv:2309.07918*, 2023.

-
Xie et al. [2022a]

Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll.

Chore: Contact, human and object reconstruction from a single rgb image.

In *ECCV*, 2022a.

-
Xie et al. [2023a]

Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll.

Visibility aware human-object interaction tracking from single rgb camera.

In *CVPR*, 2023a.

-
Xie et al. [2022b]

Zhaoming Xie, Sebastian Starke, Hung Yu Ling, and Michiel van de Panne.

Learning soccer juggling skills with layer-wise mixture-of-experts.

In *ACM SIGGRAPH 2022 Conference Proceedings*, pages 1–9, 2022b.

-
Xie et al. [2023b]

Zhaoming Xie, Jonathan Tseng, Sebastian Starke, Michiel van de Panne, and C. Karen Liu.

Hierarchical planning and control for box loco-manipulation, 2023b.

-
Xu et al. [2023a]

Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui.

Interdiff: Generating 3d human-object interactions with physics-informed diffusion.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 14928–14940, 2023a.

-
Xu et al. [2023b]

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, et al.

Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4737–4746, 2023b.

-
Yao et al. [2022]

Heyuan Yao, Zhenhua Song, Baoquan Chen, and Libin Liu.

Controlvae: Model-based learning of generative controllers for physics-based characters.

*ACM Transactions on Graphics (TOG)*, 41(6):1–16, 2022.

-
Yoon et al. [2019]

Youngwoo Yoon, Woo-Ri Ko, Minsu Jang, Jaeyeon Lee, Jaehong Kim, and Geehyuk Lee.

Robots learn social skills: End-to-end learning of co-speech gesture generation for humanoid robots.

In *2019 International Conference on Robotics and Automation (ICRA)*, pages 4303–4309. IEEE, 2019.

-
Yuan et al. [2022]

Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz.

Physdiff: Physics-guided human motion diffusion model, 2022.

-
Zhang et al. [2023a]

Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation.

*arXiv preprint arXiv:2309.03891*, 2023a.

-
Zhang et al. [2023b]

Haotian Zhang, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, and Kayvon Fatahalian.

Learning physically simulated tennis skills from broadcast videos.

*ACM Trans. Graph.*, 2023b.

-
Zhang et al. [2023c]

Juze Zhang, Haimin Luo, Hongdi Yang, Xinru Xu, Qianyang Wu, Ye Shi, Jingyi Yu, Lan Xu, and Jingya Wang.

NeuralDome: A neural modeling pipeline on multi-view human-object interactions.

In *CVPR*, 2023c.

-
Zhang et al. [2023d]

Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen.

T2m-gpt: Generating human motion from textual descriptions with discrete representations.

*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023d.

-
Zhang et al. [2020]

Jason Y Zhang, Sam Pepose, Hanbyul Joo, Deva Ramanan, Jitendra Malik, and Angjoo Kanazawa.

Perceiving 3d human-object spatial arrangements from a single image in the wild.

In *ECCV*, 2020.

-
Zhang et al. [2022a]

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu.

Motiondiffuse: Text-driven human motion generation with diffusion model.

*arXiv preprint arXiv:2208.15001*, 2022a.

-
Zhang et al. [2022b]

Xiaohan Zhang, Bharat Lal Bhatnagar, Sebastian Starke, Vladimir Guzov, and Gerard Pons-Moll.

COUCH: Towards controllable human-chair interactions.

In *ECCV*, 2022b.

-
Zhang et al. [2023e]

Yunbo Zhang, Deepak Gopinath, Yuting Ye, Jessica Hodgins, Greg Turk, and Jungdam Won.

Simulation and retargeting of complex multi-character interactions.

In *ACM SIGGRAPH 2023 Conference Proceedings*, 2023e.

-
Zhou et al. [2022]

Desen Zhou, Zhichao Liu, Jian Wang, Leshan Wang, Tao Hu, Errui Ding, and Jingdong Wang.

Human-object interaction detection via disentangled transformer.

In *CVPR*, 2022.

-
Zhu et al. [2023a]

Fangrui Zhu, Yiming Xie, Weidi Xie, and Huaizu Jiang.

Diagnosing human-object interaction detectors.

*arXiv preprint arXiv:2308.08529*, 2023a.

-
Zhu et al. [2023b]

Lingting Zhu, Xian Liu, Xuanyu Liu, Rui Qian, Ziwei Liu, and Lequan Yu.

Taming diffusion models for audio-driven co-speech gesture generation.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10544–10553, 2023b.

\thetitle

Supplementary Material

In the following document, we provide additional information to supplement the main paper. Video demonstrations can be found on our project page. We will release the BallPlay dataset and open source the code.

\etocdepthtag

.tocmtappendix
\etocsettagdepthmtchapternone
\etocsettagdepthmtappendixsubsection

## Appendix A Technical Details

### A.1 HOI Data Preprocessing

Since the shapes of humans and objects in HOI data are represented by mesh and cannot be directly used for the simulation environment, we build the simulation models of robots and objects to match their meshes and make necessary calibrations for the HOI data.

Raw HOI Data Format. The raw HOI data consists of frames at 30 fps. Each frame contains the human joint rotation, human root rotation, human root position, object position, and object rotation. We use the SMPL-X model [[[67](#bib.bib67)]] to parameterize the whole-body shape as $\mathcal{\beta}\in\mathbb{R}^{10}$ and whole-body pose as $\mathcal{\theta}\in\mathbb{R}^{51\times 3}$. The object is represented as mesh.

*Figure 8: The original human and object shape (bottom) and their approximated simulation model (up).*

Simulation Models for Human and Object.
Given an SMPL-X shape parameter $\mathcal{\beta}$, we generate a corresponding whole-body humanoid robot following UHC [[[58](#bib.bib58), [59](#bib.bib59)]]. To drive the poses, we use SMPL-X pose parameter $\mathcal{\theta}$. Accordingly, the robot has a total of 51$\times$3 DoF actuators where 21$\times$3 DoF for the body and 30$\times$3 DoF for the hands. To simplify collision calculation, bodies are simplified as simple geometries like capsules or boxes. For general objects, we use convex decomposition to approximate the object mesh as convex hulls, which is necessary for collision detection. For the basketball, we simply use a sphere as an approximation. Fig. [8](#A1.F8) shows the original human and object shape and the approximated simulation model.

HOI Data Calibration.
We transfer the original HOI data to the simulation environment and perform coordinate calibration to obtain preliminary data samples. To accurately model interaction information, we extract and save the body positions and the binary contact labels per frame, which can be easily acquired by reading the simulator’s API after loading the calibrated HOI data into the simulation. This data calibration process is effective in resolving hand-object penetration in the data, because the simulator will automatically adjust the body position to avoid collision.

### A.2 Details on Simulation Setting

Simulation Initialization.
We use first-frame initialization by default; that is, we extract the rotations and root positions from the first reference state to initialize the robot and objects.

Early Termination.
Since our imitation learning is strictly aligned with the reference frame, the maximum duration of the simulation is the length of the reference HOI sequence. In general, during training, the simulation is reset only when the maximum time is reached. However, during the training process, the robot often fails before reaching the maximum time, such as falling down or the object significantly deviating from the reference trajectory. In these cases, we reset the simulation to improve the simulation sample efficiency. Specifically, in three cases we reset the simulation: (1) the maximum time is reached; (2) the object deviates far from the reference trajectory; (3) the robot positions deviate from the reference. The threshold of position error is set to 0.5m.

### A.3 Experiment Details

Hyperparameters.
We use the aggregated contact graph (CG) for experiments. For GRAB, the CG contains 3 nodes: the table, the object, and the aggregated whole body. For BallPlay, the CG contains 3 nodes: the object, the aggregated hands, and the aggregated rest body parts. We provide the setting of reward weights in Tab. [4](#A1.T4). For BallPlay*, $\boldsymbol{\lambda}^{cg}[0]$ corresponding to the CG edge connecting the hands and the ball, $\boldsymbol{\lambda}^{cg}[1]$ corresponding to the CG edge connecting the hands and the rest body, $\boldsymbol{\lambda}^{cg}[2]$ corresponding to the CG edge connecting the rest body and the ball. For *GRAB*, $\boldsymbol{\lambda}^{cg}[0]$ corresponding to the CG edge connecting the body and the object, $\boldsymbol{\lambda}^{cg}[1]$ corresponding to the CG edge connecting the body and the table, $\boldsymbol{\lambda}^{cg}[2]$ corresponding to the CG edge connecting the table and the object.

*Table 4: Reward weights for PhysHOI. Note that for the *fingertip spin* case, we set $\boldsymbol{\lambda}^{cg}[0]$=0.01 to weak restrictions on contact between hands and the ball, as explained in Fig. [14](#A5.F14).*

*Table 5: Reward design for re-implemented methods.*

Details on Re-Implemented Methods.
For fair comparisons, we re-implement related state-of-the-art methods: DeepMimic [[[69](#bib.bib69)]], AMP [[[71](#bib.bib71)]], and Zhang’s method [[[122](#bib.bib122)]]. The main difference is the reward design, which is presented in Tab. [5](#A1.T5). Note that our code is based on the ASE project [[[72](#bib.bib72)]], which contains the implementation of AMP. We change the original AMP observation as our HOI representation for HOI imitation.

Contact Detection.
Since Isaac Gym does not yet provide contact detection APIs for the GPU pipeline, we use force detections as approximations for contact detections. For example, if the net contact force of the ball on either of the x and y axes is above a threshold and the net contact force of the hands-excluded body parts is under a threshold, we deem that the ball has contact with the hands.

## Appendix B The BallPlay Dataset

### B.1 HOI Data Annotation

To construct the BallPlay dataset, we propose a semi-automatic annotation pipeline to estimate 3D human-object interaction motion from monocular RGB videos. As depicted in Fig. [9](#A2.F9), our annotation pipeline mainly involved three stages: whole-body human motion annotation, object annotation, and contact-based manual correction.

Whole-Body Human Motion Annotation. Initially, we adopt an optimization-based approach, as introduced in Motion-X [[[52](#bib.bib52)]], to annotate high-quality 3D whole-body human motion from RGB videos. This optimization process also includes the estimation of camera parameters.

Object Annotation. Subsequently, we employ Grounded-SAM [[[26](#bib.bib26)]] to annotate the object category and the segmentation mask, along with utilizing ZoeDepth [[[3](#bib.bib3)]] for depth estimation. We then project the object mask into a point cloud in the world coordinate system based on the depth map and camera parameters. The central point of the resulting point cloud serves as the initialized object center. Meanwhile, based on the object category, we retrieve the object mesh from a CAD model library. The meshes of the CAD model library are collected from ShapeNet [[[8](#bib.bib8)]] and some geometries built by ourselves with Trimesh [[[16](#bib.bib16)]] tool.

Contact-based Manual Correction. Due to the inherent depth ambiguity of monocular video, the object center obtained from the previous steps can be noisy. To address this issue, we propose a contact-based manual correction procedure. Specifically, we manually annotate the human-object contact labels of three keyframes within each video, and then compute the average depth of the contact body parts to update the object center position. This manual correction can greatly mitigate depth ambiguity and enhance overall data quality. Ultimately, our refined object mesh and human mesh constitute the final human-object interaction dataset.

*Figure 9: Annotation pipeline of BallPlay*.*

*Figure 10: PhysHOI can rectify the error of reference HOI data. We can see that the reference HOI data suffers inaccuracies more or less, while the result generated by PhysHOI can be much better than the reference.*

Contact Graph Annotation.
We focus on basketball skills that involve only hands to interact with the ball. In this scenario, we aggregate all elements as 3 Contact Graph (CG) nodes: the ball, the hands, and the rest of body parts. For all sequences, the rest body parts do not come into contact with the ball and the hands, so we only need to manually label the CG edge between the hands and the ball, which is easy to obtain through observation.

### B.2 Refine The HOI Data Using PhysHOI

Though some HOI data in BallPlay may be biased, we may resort to PhysHOI to eliminate errors and yield physically plausible HOI data. As shown in Fig. [10](#A2.F10), we can see that the reference HOI data suffer inaccuracies, while the results generated by PhysHOI are much better than the reference, in terms of physical plausibility and smoothness. Specifically, we first train PhysHOI to imitate an HOI sequence. After convergence, we do the inference under physics simulation and record the HOI data as well as the CG through the API of the environment. Physically rectified HOI data is also provided in the BallPlay dataset.

## Appendix C Additional Experiments

### C.1 Varying Ball Sizes

During the inference time, we change the radius of the ball to different values and surprisingly find that the humanoids can still perform plausible interaction, even if they are not trained on these ball sizes. Fig. [13](#A5.F13) shows the result in three cases. Interestingly, we still get reasonable results without additional training for different ball sizes, which shows the robustness of PhysHOI. Note that the humanoid may fail if the ball size changes too drastically.

### C.2 Ablation on Contact Graph Reward

In addition to the ablation in the main paper, we provide more visual comparisons in Fig. [12](#A5.F12) for an intuitive understanding. In the cross leg* case, the humanoid trained w/o CGR tends to hold the ball between its left hand and leg (we provide another view in red boxes for better observation), which causes unnatural interaction. In the *pass* case, the ball drops without the CGR. In the *fingertip spin* case, the humanoid trained w/o CGR tends to hold the ball with its hands and head. In comparison, applying CGR addresses these problems well. In Fig. [11](#A4.F11) we visualize multiple environments to intuitively demonstrate the success rate w/ or w/o CGR. Each environment applies different random seeds. We can see that the ball falling w/o CGR is not accidental, but common. Instead, our method applies CGR and achieves steady success in ball holding.

## Appendix D Related Work on Kinematic-Based Methods

### D.1 Human Motion Generation

Kinematic motion generation methods are usually learned from motion capture datasets [[[76](#bib.bib76), [61](#bib.bib61), [28](#bib.bib28), [52](#bib.bib52)]]. Early attempts generate motions from the prefix [[[19](#bib.bib19), [63](#bib.bib63)]] and keyframes [[[29](#bib.bib29)]]. [[[36](#bib.bib36)]] generate motions using an autoencoder, and further apply motion phases to generate consistent motions [[[37](#bib.bib37), [89](#bib.bib89), [90](#bib.bib90), [91](#bib.bib91)]]. Beyond action-conditioned motion generation [[[27](#bib.bib27), [53](#bib.bib53), [74](#bib.bib74)]], recent advances make it possible to control motions with text description [[[28](#bib.bib28), [96](#bib.bib96), [120](#bib.bib120), [118](#bib.bib118), [9](#bib.bib9), [43](#bib.bib43), [56](#bib.bib56)]], music [[[49](#bib.bib49), [99](#bib.bib99), [14](#bib.bib14)]], and speech [[[113](#bib.bib113), [50](#bib.bib50), [125](#bib.bib125)]], based on recent progress in autoregressive models [[[77](#bib.bib77), [7](#bib.bib7)]] and diffusion models [[[34](#bib.bib34), [83](#bib.bib83), [84](#bib.bib84), [82](#bib.bib82), [85](#bib.bib85), [45](#bib.bib45), [101](#bib.bib101)]].

### D.2 Human-Object Interaction Generation

There are many research topics surrounding HOI, especially detection [[[25](#bib.bib25), [41](#bib.bib41), [103](#bib.bib103), [123](#bib.bib123), [107](#bib.bib107), [10](#bib.bib10), [124](#bib.bib124)]] and reconstruction [[[106](#bib.bib106), [119](#bib.bib119), [100](#bib.bib100), [73](#bib.bib73), [39](#bib.bib39), [46](#bib.bib46)]]. Based on recent development of 3D HOI datasets [[[92](#bib.bib92), [4](#bib.bib4), [42](#bib.bib42), [40](#bib.bib40), [117](#bib.bib117), [18](#bib.bib18), [87](#bib.bib87), [121](#bib.bib121), [30](#bib.bib30)]], there emerges a branch of research focuses on generating human motion that interacts with objects [[[88](#bib.bib88), [93](#bib.bib93), [104](#bib.bib104), [24](#bib.bib24), [47](#bib.bib47), [110](#bib.bib110), [48](#bib.bib48), [95](#bib.bib95), [75](#bib.bib75), [44](#bib.bib44)]]. However, all the methods mentioned learn directly from kinematic datasets without strict modeling of dynamics and contacts, causing artifacts that harm the motion quality, *e.g*., penetration, shaking, sliding, and floating. Besides, kinematic HOI generation can not be used for robot control yet.

*Figure 11: Ablation on Contact Graph Reward (CGR). We visualize multiple environments to intuitively demonstrate the success rate w/ or w/o CGR. We can see that if w/o CGR, the ball falling is not accidental, but common. Instead, our method applies CGR and achieves steady success in ball holding.*

## Appendix E Discussions

### E.1 Failure Cases

The failure cases of our method are mainly due to inaccurate data or incomplete contact graphs. For example, in the rebound* case in Fig. [14](#A5.F14), the ball in the reference data is always under the hand, which, combined with the fact that the ball bounces back to the hand very quickly, results in a local optimum: the ball is not firmly grasped during the learning process. This problem can be solved with precise HOI data.

High CGR weight may lead to unpleasant interactions if the CG node is not detailed enough. As shown in the *fingertip spin* case in Fig. [14](#A5.F14), when we set a high CGR weight on the CG edge between hands and the ball, the trained humanoid tends to use multiple fingers to maintain steady contact. The reasons can be twofold: (1) The CG is not detailed enough, e.g., we simply take two hands as one CG node, but it is necessary to take fingers as separate CG nodes in this case. (2) The frame rate of simulation and reference data is low, which yields frequent bouncing, i.e., less contact. While reducing the corresponding CG edge weights can improve this problem, as shown in the *fingertip spin* case in Fig. [13](#A5.F13) and Fig. [12](#A5.F12), a more general solution requires richer CGs and higher frame rates, which is also the solution we expect.

### E.2 Limitations

Though our method is able to mimic diverse dynamic HOI skills, it still faces several limitations, which we list here:

-
•

Our method may fail when the reference HOI data suffers severe biases, *e.g*., the false reference data can result in a ball drop, as shown in the *rebound* case in Fig. [14](#A5.F14).

-
•

When CG nodes are not detailed enough, some subtle operations can still easily fall into local optimality. For example, fingers should be independent CG nodes when learning complex in-hand manipulations.

-
•

Due to the low frame rate of HOI data and simulation frequency, some minor penetrations may appear.

-
•

Informing the policy with a single reference object state may not be sufficient for long-term hands-off object control. For example, when learning diverse jump shot sequences, it is difficult for the policy to determine how to control the ball through a single frame of HOI reference state because the ball is out of control after being shot out. One potential solution is to provide the policy with multi-frame reference states of the ball.

-
•

Our method can not generalize to HOIs that are not trained on, *e.g*., the policy trained on *back dribble* can not handle *fingertip spin*.

*Figure 12: Ablation on Contact Graph Reward (CGR). The humanoid trained w/o CGR tends to fall into the local optimal of kinematic rewards, e.g., using its left leg and left hand to hold the ball. We provide the cross leg* case another view in boxes for better observation.*

### E.3 Future Work

We believe this work opens up many exciting directions for future exploration, which we list here:

-
•

Our work demonstrates the potential to learn generic human skills from diverse HOI data. The way of acquiring HOI data accurately and conveniently is worth studying.

-
•

When large HOI data is available, studying generalization and generation based on HOI imitation will be an attractive direction toward future humanoid autonomy, considering no need for designing task-specific rewards. For example, train an MVAE [[[53](#bib.bib53)]] using HOI imitation based on a large HOI dataset.

-
•

Although building a real humanoid with 153 DOF seems far away, it is promising to explore retargeting-based HOI Imitation, *e.g*., teaching a robot arm with a dextrous hand to play basketball via HOI Imitation.

-
•

Multi-human dynamic interactions are also worth studying, such as multi-player basketball games.

*Figure 13: Varying Ball Sizes. During inference, we change the ball radius to different values. Interestingly, we still get reasonable results, which show the robustness of PhysHOI. The default ball radius is 12cm. R denotes the changed ball radius.*

![Figure](/html/2312.04393/assets/x11.png)

*Figure 14: Failure cases. The failure cases of our method are mainly due to inaccurate data or incomplete contact graphs. In the rebound* case, the biased ball position results in a local optimum: the ball not being firmly grasped during the learning process. High CGR weight may lead to unpleasant interactions if the CG node is not detailed enough. In the *fingertip spin* case, when we set a high CGR weight on the CG edge between hands and the ball, the trained humanoid tends to use multiple fingers to maintain steady contact. The reasons can be twofold: (1) The CG is not detailed enough, e.g., we simply take two hands as one CG node, but it is necessary to take fingers as separate CG nodes in this case. (2) The frame rate of simulation and reference data is low, which yields frequent bouncing, i.e., less contact. While reducing the corresponding CG edge weights can improve this problem, as shown in the *fingertip spin* case in Fig. [13](#A5.F13) and Fig. [12](#A5.F12), a more general solution requires richer CGs and higher frame rates, which is also the solution we expect.*

[◄](/html/2312.04392)

[Feeling lucky?](/feeling_lucky)

[Conversion report](/log/2312.04393)
[Report an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2312.04393)
[View original on arXiv](https://arxiv.org/abs/2312.04393)[►](/html/2312.04394)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Feb 27 15:21:09 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)