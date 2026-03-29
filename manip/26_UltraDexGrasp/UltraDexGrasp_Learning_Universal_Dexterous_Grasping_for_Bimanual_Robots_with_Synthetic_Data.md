##### Report GitHub Issue

**×




Title:



Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.05312v1 [cs.RO] 05 Mar 2026

[# UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data Sizhe Yang1,2 Yiman Xie1,3 Zhixuan Liang1,4 Yang Tian1,5 Jia Zeng1 Dahua Lin1,2 Jiangmiao Pang1 1Shanghai AI Laboratory 2The Chinese University of Hong Kong 3Zhejiang University 4The University of Hong Kong 5Peking University Project page: https://yangsizhe.github.io/ultradexgrasp/](https://yangsizhe.github.io/ultradexgrasp/)

###### Abstract

Grasping is a fundamental capability for robots to interact with the physical world.
Humans, equipped with two hands, autonomously select appropriate grasp strategies based on the shape, size, and weight of objects, enabling robust grasping and subsequent manipulation. In contrast, current robotic grasping remains limited, particularly in multi-strategy settings.
Although substantial efforts have targeted parallel-gripper and single-hand grasping, dexterous grasping for bimanual robots remains underexplored, with data being a primary bottleneck.
Achieving physically plausible and geometrically conforming grasps that can withstand external wrenches poses significant challenges.
To address these issues, we introduce UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots.
The proposed data-generation pipeline integrates optimization-based grasp synthesis with planning-based demonstration generation, yielding high-quality and diverse trajectories across multiple grasp strategies. With this framework, we curate UltraDexGrasp-20M, a large-scale, multi-strategy grasp dataset comprising 20 million frames across 1,000 objects.
Based on UltraDexGrasp-20M, we further develop a simple yet effective grasp policy that takes point clouds as input, aggregates scene features via unidirectional attention, and predicts control commands.
Trained exclusively on synthetic data, the policy achieves robust zero-shot sim-to-real transfer and consistently succeeds on novel objects with varied shapes, sizes, and weights, attaining an average success rate of 81.2% in real-world universal dexterous grasping. To facilitate future research on grasping with bimanual robots, we open-source the data generation pipeline at [https://github.com/InternRobotics/UltraDexGrasp](https://github.com/InternRobotics/UltraDexGrasp).

## I Introduction

Dexterous grasping is a critical step toward enabling robots to manipulate objects with human-level proficiency. As the first stage of many manipulation skills, it bridges the transition from non-contact to contact interaction.
In daily life, humans autonomously select grasping strategies that accommodate variations in the shape, size, and weight of objects: large, heavy objects generally require a coordinated bimanual grasp to maintain balance;
medium-sized objects can be stably grasped by a single hand using all five fingers; and small objects, which provide insufficient surface area, are best handled with two-finger pinch or three-finger tripod. Humans can also adapt hand postures to object geometry, maximizing surface conformity to ensure robust grasping and subsequent manipulation.
In contrast, current robotic grasping capabilities fall far short of human performance. Although the burgeoning humanoid-robot market has accelerated the development of dual-arm platforms and dexterous hands, data and algorithms remain the principal bottleneck.
Considerable efforts [[[6](#bib.bib46), [8](#bib.bib63), [32](#bib.bib2), [41](#bib.bib49), [5](#bib.bib41), [7](#bib.bib48), [34](#bib.bib69)]]
have been devoted to parallel-gripper and single-hand grasping, yet universal dexterous grasping for bimanual robots remains underexplored.

A central difficulty of universal dexterous grasping is producing high-quality data.
Existing data generation paradigms for dexterous grasping fall into three strands:
1) reinforcement learning (RL) with privileged information that trains experts for grasping [[[31](#bib.bib10), [20](#bib.bib11), [27](#bib.bib12), [15](#bib.bib70)]];
2) optimization-based synthesis that minimizes grasp-related costs [[[32](#bib.bib2), [17](#bib.bib5), [13](#bib.bib18), [29](#bib.bib19)]];
and 3) learning-based synthesis that trains generative models to predict diverse grasps [[[36](#bib.bib9), [37](#bib.bib7), [12](#bib.bib64)]].
However, RL-trained experts are typically deterministic after training, thus mapping a given observation to a single action, which leads to repetitive postures and a lack of diversity.
Optimization- and learning-based synthesis are largely open-loop, struggle in dynamic real-world scenarios and typically ignore arm kinematics. Moreover, previous approaches are predominantly limited to single-hand settings.
Universal dexterous grasping for dual-arm robots poses exceptional challenges for data generation, due to the extensive degrees of freedom, requirements for bimanual coordination, and the multitude of possible grasping strategies.
Whether data are collected via real-world teleoperation or generated in simulation, producing physically plausible and geometrically conforming grasps that resist external wrenches, is essential though difficult.

To address these challenges, we propose UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots. The proposed data-generation pipeline integrates an optimization-based grasp synthesizer with a planning-based demonstration generation module for coordinated dual-arm manipulation.
This integration yields kinematically feasible and natural closed-loop motions while preserving data diversity.
Notably, the data generation supports multiple grasp strategies, including holding large objects with both hands, whole-hand grasp for medium-sized objects, and two-finger pinch or three-finger tripod for small objects.
With this pipeline, we curate UltraDexGrasp-20M, the first large-scale multi-strategy dexterous grasp dataset for bimanual robots, comprising 20 million frames over 1,000 objects.

Building upon this, we develop a simple yet effective grasp policy that takes point clouds as input, aggregates scene features via unidirectional attention, and predicts control commands, enabling multiple grasp strategies and improving generalization across diverse objects. We conduct comprehensive experiments in both simulation and the real world to evaluate the robustness of the proposed policy trained on UltraDexGrasp-20M. Testing in simulation on 600 objects—including both seen and unseen objects during training—that vary widely in shape, weight (ranging from 5 g to 1,000 g), and size (from objects with a longest bounding box edge less than 0.03 m to those with a shortest edge exceeding 0.5 m), our policy attains an 84.0% average success rate, exceeding the next best baseline by 25.2 percentage points (approximately 43% relative improvement).
The policy, trained exclusively on synthetic data, is further deployed in real-world scenarios. It adapts grasp strategies to different objects and successfully handles diverse items, achieving an 81.2% success rate across all objects.

In summary, our paper makes three contributions. First, we present UltraDexGrasp-20M and its data-generation pipeline that integrates optimization-based grasp synthesis with planning-based demonstration generation, producing high-quality and diverse trajectories across multiple grasp strategies. Second, we introduce a novel grasp policy for universal dexterous grasping with bimanual robots that enhances generalization to diverse objects. Third, we demonstrate that our policy, trained solely on synthetic data, enables robust zero-shot sim-to-real transfer and strong generalization to novel objects with varied shapes, sizes, and weights.

## II Related Work

Dexterous grasp synthesis and dataset.
Grasp synthesis is crucial for the advancement of dexterous grasping, and existing approaches can be broadly categorized into three types: sampling-based, optimization-based, and learning-based methods. Sampling-based methods [[[21](#bib.bib1)]] typically require simplification of the search space, which leads to limited grasp diversity. Many studies focus on optimization-based methods [[[32](#bib.bib2), [17](#bib.bib5), [13](#bib.bib18), [29](#bib.bib19), [30](#bib.bib20), [43](#bib.bib3)]], with some utilizing differentiable force closure to optimize grasp poses [[[32](#bib.bib2), [17](#bib.bib5), [14](#bib.bib6)]]. [[[3](#bib.bib4)]] formulates grasp synthesis as a bilevel optimization problem to generate high-quality grasps. Supervised learning-based methods [[[36](#bib.bib9), [37](#bib.bib7), [33](#bib.bib13), [12](#bib.bib64)]] require a small amount of grasp data to train generative models that can produce a large number of grasps for novel objects. Additionally, some works employ reinforcement learning to train experts for grasp generation [[[31](#bib.bib10), [20](#bib.bib11), [27](#bib.bib12)]]. Effective grasp synthesis and teleoperation have enabled the creation of several dexterous grasp datasets [[[9](#bib.bib8), [37](#bib.bib7), [1](#bib.bib15), [2](#bib.bib16), [18](#bib.bib14)]].
Recently, [[[26](#bib.bib21)]] proposed a bimanual grasp generation algorithm, but it imposes a simplified
contact model, does not consider dual-arm coordination, and has not explored closed-loop control policies for universal grasping. Although [[[16](#bib.bib37)]] explores bimanual dexterous grasping, the RL-trained expert can only grasp limited objects, such as boxes. Scaling to a wider variety of objects is costly and generalization remains challenging.
In contrast, our work proposes a framework for generating grasps for arbitrary objects and trains policies capable of handling novel and diverse objects.

Generalizable policy for robotic grasping.
Grasping is a crucial skill for robotic manipulation. Several studies focus on data-efficient learning for grasping using real-world datasets [[[6](#bib.bib46), [7](#bib.bib48), [22](#bib.bib47), [41](#bib.bib49)]]. Some other works train and evaluate grasping policies in simulated environments, demonstrating high grasp success rates across large numbers of objects [[[36](#bib.bib9), [31](#bib.bib10), [42](#bib.bib51)]]. Some research targets functional grasping, emphasizing object part functionality and subsequent tasks [[[11](#bib.bib52), [10](#bib.bib42)]]. Recently, sim-to-real transfer of grasping policies has shown promising results. Methods such as [[[5](#bib.bib41), [40](#bib.bib44), [19](#bib.bib54), [37](#bib.bib7)]] generate large-scale grasping datasets using optimization-based or rule-based pipelines. Other approaches, including [[[20](#bib.bib11), [27](#bib.bib12), [39](#bib.bib55), [16](#bib.bib37), [4](#bib.bib56)]], employ reinforcement learning to train expert agents for grasp data generation, achieving strong real-world performance. Our proposed bimanual grasp policy is trained on data produced through the complementary integration of optimization-based and rule-based approaches, achieving robust real-world bimanual dexterous grasping.

## III Preliminaries

### III-A Definition of Grasp Pose for Bimanual Robots

A grasp is defined as the act of restraining the motion of an object by applying forces and torques at a set of contact points. The grasp pose represents the critical configuration of the hands required to achieve a stable grasp. Specifically, we parameterize the grasp pose for bimanual robots as a tuple:

$$g=\left\{\left(\boldsymbol{t}_{h},\boldsymbol{R}_{h},\boldsymbol{q}_{h}\right)\mid h=0,1\right\},$$ \tag{1}

where $h\in\{0,1\}$ denotes the index of the hand, $\boldsymbol{t}_{h}\in\mathbb{R}^{3}$ is the translation of the $h$-th hand, $\boldsymbol{R}_{h}\in SO(3)$ represents the rotation of the hand, and $\boldsymbol{q}_{h}\in\mathbb{R}^{n}$ denotes joint positions for the hand, with $n$ being the number of joints of the hand.

### III-B Basics of Grasp Modeling

Grasping focuses on restraining the motion of an object through application of forces and torques at a set of contact points. We use point-on-plane contacts as the contact type, and hard finger model as the contact model, which is commonly used in force closure grasps. In the hard finger model, forces of arbitrary magnitude and direction can be applied within the limits defined by the friction cone:

$$\mathcal{F}=\left\{\boldsymbol{f}\;\middle|\;\|\boldsymbol{f}_{\mathrm{tan}}\|\leq\mu\|\boldsymbol{f}_{\mathrm{n}}\|,\;\;f_{z}\geq 0\right\},$$ \tag{2}

where $\mu$ is the static friction coefficient of the object,
$\boldsymbol{f}_{\mathrm{n}}=[0,0,f_{\mathrm{z}}]$ is the vector component along the normal direction, and
$\boldsymbol{f}_{\mathrm{tan}}=[f_{\mathrm{x}},f_{\mathrm{y}},0]$ is the vector component tangent to the surface.
Each contact point applies a wrench to the manipulated object:

$$\boldsymbol{w}_{i}=\begin{bmatrix}\boldsymbol{f}_{i}\\ \alpha(\boldsymbol{d}_{i}\times\boldsymbol{f}_{i})\end{bmatrix},$$ \tag{3}

where $\boldsymbol{f}_{i}\in\mathbb{R}^{3}$ is the force, $\alpha(\boldsymbol{d}_{i}\times\boldsymbol{f}_{i})$ is the torque, $\alpha\in\mathbb{R}$ is an arbitrary constant, and $\boldsymbol{d}_{i}$ is the relative position of the $i$-th contact point with respect to the center of mass of the object. $\boldsymbol{w}_{i}$ can be represented as $\boldsymbol{G}_{i}\boldsymbol{f}_{i}$, where $\boldsymbol{G}_{i}$ is a wrench basis matrix for the $i$-th contact. For hard finger model, the grasp map $\boldsymbol{G}_{i}$ is defined as:

$$\boldsymbol{G}_{i}=\begin{bmatrix}\mathbf{I}_{3\times 3}\\ (\boldsymbol{p}_{i}-\boldsymbol{m})_{\times}\end{bmatrix}\boldsymbol{O}_{i},$$ \tag{4}

where $\mathbf{I}_{3\times 3}$ is the $3\times 3$ identity matrix, $(\boldsymbol{p}_{i}-\boldsymbol{m})_{\times}$ represents the cross product with $(\boldsymbol{p}_{i}-\boldsymbol{m})$, where $\boldsymbol{p}_{i}$ denotes the contact position, $\boldsymbol{m}$ represents the center of mass of the object, and $\boldsymbol{O}_{i}$ is the rotation matrix from the local contact frame to the frame of the object.

Thus, the grasp wrench space $\mathcal{W}$ for a grasp with $k$ contact points is defined as the set of possible wrenches $w$ that can be applied to the object:

$$\mathcal{W}:=\left\{\boldsymbol{w}\;\middle|\;\boldsymbol{w}=\sum_{i=1}^{k}\boldsymbol{G}_{i}\boldsymbol{f}_{i},~\boldsymbol{f}_{i}\in\mathcal{F}_{i},\;\;i=1,\ldots,k\right\}.$$ \tag{5}

The grasp wrench space $\mathcal{W}$ is expected to be sufficiently large to resist external wrenches.

## IV Universal Dexterous Grasp Dataset

![Figure](2603.05312v1/x1.png)

*Figure 2: Overview of data generation pipeline. We first collect diverse object assets and import the objects and the robot URDF files into the simulator. An optimization-based grasp synthesizer is then used to generate feasible grasps, from which the preferred grasp is selected. Finally, motion planning is employed to generate demonstration trajectories.*

As illustrated in Fig. [2](#S4.F2), the pipeline begins with scene initialization, where object assets and the robot URDF are imported into the simulation environment.
We select 1,000 distinct objects from DexGraspNet [[[32](#bib.bib2)]] to construct our object assets.
Camera pose and joint impedance are randomized to reduce the sim-to-real gap. Given the meshes and poses of both the table and object, the grasp synthesizer generates a batch of feasible grasps, which are subsequently filtered and ranked to select the best candidate. We employ bimanual motion planning to compute collision-free and coordinated trajectories, which are then executed in simulation. Utilizing this scalable data generation pipeline, we curate the UltraDexGrasp-20M dataset.

### IV-A Grasp Synthesis

Grasp synthesis involves two main stages. First, the hands are initialized near the object, and then, through optimization, physically plausible and geometrically conforming grasps are generated. In the initialization stage, we first obtain the convex hull of the object’s mesh.
For unimanual grasps, such as whole-hand, two-finger pinch, and three-finger tripod, we sample a point on the surface of the object’s convex hull. For bimanual grasps, two points are sampled on the convex hull surface, positioned on opposite sides of the object’s center. The hand is positioned along the normal vector at the sampled point, with the palm facing the target object.

Given the object mesh $\boldsymbol{M}$ with center of mass $\boldsymbol{m}$, two hands $h\in\{0,1\}$ with known kinematics, candidate contact sets $\bar{\mathcal{C}}_{h}$, a strategy-selected active contact set $\mathcal{C}\subset\bar{\mathcal{C}}_{0}\cup\bar{\mathcal{C}}_{1}$, and the contact point $\boldsymbol{c}\in\mathcal{C}$, the decision variables are the bimanual grasp pose $\boldsymbol{g}=\{(\boldsymbol{t}_{h},\boldsymbol{R}_{h},\boldsymbol{q}_{h})\mid h=0,1\}$ and the contact forces $\{\boldsymbol{f}_{\boldsymbol{c}}\in\mathbb{R}^{3}\}_{\boldsymbol{c}\in\mathcal{C}}$. Let $\boldsymbol{p}_{\boldsymbol{c}}=\boldsymbol{p}_{\boldsymbol{c}}(\boldsymbol{g},~\boldsymbol{c})$ and $\boldsymbol{O}_{\boldsymbol{c}}=\boldsymbol{O}_{\boldsymbol{c}}(\boldsymbol{g})$ denote the contact position and orientation obtained via forward kinematics, and define the grasp map
$\boldsymbol{G}_{\boldsymbol{c}}=\begin{bmatrix}\mathbf{I}\\
(\boldsymbol{p}_{\boldsymbol{c}}-\boldsymbol{m})_{\times}\end{bmatrix}\boldsymbol{O}_{\boldsymbol{c}}$.
Let $d_{M}(\cdot)$ be the signed distance to the object surface.
Given target wrenches $\{\boldsymbol{w}_{j}\}_{j=1}^{J}$, a scaling factor $\lambda>0$, and weights $\kappa_{\bullet}>0$, grasp synthesis for bimanual robots is formulated as:

$$
\begin{aligned}
\min_{\boldsymbol{g},\,\{\boldsymbol{f}_{\boldsymbol{c}}\}} &\kappa_{w}\sum_{j=1}^{J}\Big\|\lambda\boldsymbol{w}_{j}-\sum_{{\boldsymbol{c}}\in\mathcal{C}}\boldsymbol{G}_{\boldsymbol{c}}(\boldsymbol{g})\boldsymbol{f}_{\boldsymbol{c}}\Big\|_{2}^{2} \tag{6}
\end{aligned}
$$
| | | $\displaystyle\quad+\kappa_{\text{con}}\sum_{{\boldsymbol{c}}\in\mathcal{C}}\psi\!\big(d_{M}(\boldsymbol{p}_{\boldsymbol{c}})\big)$ | |
| | | $\displaystyle\quad+\kappa_{\text{coll}}\,\Phi_{M}(\boldsymbol{g})+\kappa_{\text{hh}}\,\Phi_{\text{hh}}(\boldsymbol{g})$ | |

| | s.t. | $\displaystyle\boldsymbol{q}_{h,\min}\leq\boldsymbol{q}_{h}\leq\boldsymbol{q}_{h,\max},\quad h\in\{0,1\},$ | | (7) |
|---|---|---|---|---|
| | | $\displaystyle\boldsymbol{f}_{\boldsymbol{c}}\in\mathcal{F},\quad{\boldsymbol{c}}\in\mathcal{C},$ | | (8) |
| | | $\displaystyle\boldsymbol{R}_{h}\in SO(3),\quad h\in\{0,1\},$ | | (9) |

where $\psi(d)$ is a distance energy that measures distance between contact points on the hands and object surface, $\Phi_{M}(\boldsymbol{g})$ is a hand–object collision energy based on signed-distance penalties, and $\Phi_{\text{hh}}(\boldsymbol{g})$ is an inter-hand penetration energy derived from hand–hand signed distances.

Following BODex [[[3](#bib.bib4)]], we consider the above as a nonlinear bilevel program, where the lower level quadratic programming optimizes contact forces for each contact point to realize the target wrench and the upper level updates the hand pose to reduce the error between target and achievable wrenches via gradient descent. cuRobo [[[28](#bib.bib58)]] and a GPU-based QP solver are utilized to solve the problem efficiently.

The optimization program is unified for various grasp strategies. To adapt to a specific strategy, we select different contact points on the dexterous hands, as illustrated in Fig. [3](#S4.F3).
A gallery of synthesized grasps is presented in Fig. LABEL:fig:teaser.

For each object placed in the scene, we generate 500 candidate grasps. Through physical validation, we filter out physically implausible grasps. Next, we use cuRobo for inverse kinematics analysis to determine whether these grasps are reachable by bimanual robots, and perform collision checking to exclude grasps that result in collisions with objects other than the target. Finally, we calculate the $SE(3)$ distance between the poses of the remaining grasps and the current end-effector pose of the bimanual robots, ranking them accordingly and selecting the grasp with the shortest distance as the preferred grasp.
Specifically, the $SE(3)$ distance between two poses $\boldsymbol{T}_{1}$ and $\boldsymbol{T}_{2}$ is defined as:

$$d(\boldsymbol{T}_{1},\boldsymbol{T}_{2})=\|\boldsymbol{t}_{1}-\boldsymbol{t}_{2}\|_{2}+\lambda\cdot d_{\mathrm{rot}}(\boldsymbol{R}_{1},\boldsymbol{R}_{2}),$$ \tag{10}

where $\boldsymbol{t}_{1}$, $\boldsymbol{t}_{2}$ are the translation vectors, $\boldsymbol{R}_{1}$, $\boldsymbol{R}_{2}$ are the rotation matrices, and $\lambda$ is a weighting factor. The rotation distance $d_{\mathrm{rot}}$ is given by:

$$d_{\mathrm{rot}}(\boldsymbol{R}_{1},\boldsymbol{R}_{2})=\arccos\left(\frac{\operatorname{trace}(\boldsymbol{R}_{1}^{-1}\boldsymbol{R}_{2})-1}{2}\right).$$ \tag{11}

This selection strategy favors grasps that require minimal motion from the current configuration, which not only improves the efficiency of the grasping process but also leads to more natural and smooth robot movements.

![Figure](2603.05312v1/x2.png)

*Figure 3: Hand contact points for various grasp strategies. Different grasp strategies select distinct fingertip contact points, which are used to compute energy terms in the optimization process of grasp synthesis.*

### IV-B Demonstration Generation

After obtaining the preferred grasp, we divide the entire grasping process into four stages: pregrasp—the end-effector moves to a position 0.1 m away from the generated grasp pose in the direction opposite to the palm to avoid unintended collisions during approach; grasp—the hand reaches the preferred grasp pose; squeeze—the fingers apply pressure to the object to achieve a stable grasp; and lift—the object is lifted by 0.2 m. We employ bimanual motion planning to generate collision-free coordinated trajectories. To prevent policies trained on such data from exhibiting hesitation, we merge adjacent steps with negligible movements. In simulation, the robot executes the planned trajectory and grasp using PD control, and we verify whether the object is stably lifted. Specifically, the object must be raised at least 0.17 m above its initial pose and remain elevated for at least one second without being dropped.
If the object is successfully lifted and no unexpected contacts or movements are detected, the trajectory is recorded and rendered. By generating demonstrations across a diverse set of objects and various grasp strategies, we construct UltraDexGrasp-20M, a large-scale, multi-strategy grasp dataset for bimanual robots, comprising 20 million frames over 1,000 objects.

It is noteworthy that, similar to [[[25](#bib.bib32)]], we supplement the robot’s point cloud during the rendering process with an additional imaged point cloud.
During real-world deployment, the robot’s joint positions are known, enabling us to also use simulation to generate an imaged point cloud.
This approach significantly reduces the sim-to-real gap.

## V Universal Dexterous Grasp Policy

![Figure](2603.05312v1/x3.png)

*Figure 4: Overview of the policy architecture. The proposed grasp policy takes point clouds as input, encodes them using a point encoder, aggregates scene features via unidirectional attention, and predicts control commands. The policy supports multiple grasp strategies and improves generalization across diverse objects.*

### V-A Overall Architecture

To accommodate multiple grasp strategies and improve generalization across a wide range of objects, we propose a universal dexterous grasp policy, as illustrated in Fig. [4](#S5.F4).
With simplicity and effectiveness in mind, we design the policy without redundant structures or auxiliary tasks.
The policy receives scene point clouds as input, which are encoded into point features using a point encoder. The point features are processed by a decoder-only transformer architecture. Finally, action decoders predict control commands.

### V-B Point Cloud Encoding

We first downsample the point cloud using farthest point sampling (FPS) [[[23](#bib.bib66)]] to adjust the point cloud density. In practice, we use 2,048 points to balance computational cost and the granularity of scene representation. We then encode the resulting point cloud using a point encoder based on the PointNet++ [[[24](#bib.bib57)]] architecture. Specifically, we employ two set abstraction layers. The first layer does not perform downsampling and thus maintains 2,048 points. For each point, a group is formed by identifying its 32 nearest neighbors (including the point itself) using k-nearest neighbors algorithm (k-NN). Local features for each group are extracted via a series of $1\times 1$ convolution, batch normalization, ReLU activation function, and max pooling. The second set abstraction layer has a similar structure but downsamples the 2,048 points to 256, thereby capturing higher-level features for action prediction.

### V-C Action Prediction

To read out action latents, a chunk of learnable action query tokens is fed into the transformer backbone, where they integrate scene information through unidirectional attention to the point features. Finally, a multi-layer perceptron (MLP) transforms the resulting action latents into action vectors to be executed by the robot.
Rather than directly regressing the action vectors, our decoder predicts a bounded Gaussian distribution over actions via a truncated normal parameterization and optimizes the negative log-likelihood of the ground truth actions. Modeling actions probabilistically yields more stable training and enhances overall performance.

## VI Experiments

We conduct comprehensive experiments in both simulation and real-world environments to evaluate the effectiveness of our data-generation pipeline for universal dexterous grasping, as well as the proposed universal dexterous grasp policy. Specifically, we aim to answer the following questions:

1) Does training on UltraDexGrasp-20M yield a policy with universal grasping capabilities and strong generalization across diverse objects?

2) How does the universal dexterous grasp policy perform compared to previous grasping methods and closed-loop control policies?

3) How does the policy perform as the amount of training data increases?

4) Do the key design components of the proposed policy effectively improve grasp success rates?

5) How does the policy trained on UltraDexGrasp-20M perform in real-world scenarios?

### VI-A Simulation Experiments

#### VI-A1 Experimental Setup

We construct a dual-arm, dual-hand system composed of two 6-DoF UR5e robots and two 12-DoF XHand in simulation. The test set consists of 600 objects, including both seen and unseen categories during training. These objects exhibit significant variation in shape, weight, and size. The weight of the objects ranges from 5 g to 1,000 g. The smallest object has a bounding box whose longest edge is less than 0.03 m, while the largest object has a bounding box whose shortest edge is greater than 0.5 m.

#### VI-A2 Baselines

To demonstrate the effectiveness of the proposed universal dexterous grasp policy, we select two strong baselines, DP3 [[[38](#bib.bib60)]] and DexGraspNet [[[32](#bib.bib2)]], for comparison. DP3 is a diffusion policy that takes point clouds and robot state as input and has shown strong performance in dexterous manipulation tasks. DexGraspNet takes the complete object mesh as input to generate grasp poses, and motion planning is utilized to obtain the full execution trajectory.

#### VI-A3 Evaluation Metrics

Each policy is evaluated on 600 objects, with 10 trials conducted for each object. The objects are categorized into three groups based on size: small, medium, and large.
For small and medium items, placements are randomized within a 0.8 m $\times$ 0.2 m area, while large items are randomly placed in a 0.15 m $\times$ 0.16 m region to ensure reachability for the robot’s end-effectors.
The success rate of each policy is reported.

#### VI-A4 Main Results

*TABLE I: Results on simulation benchmarks. The success rates (%) of each policy are reported. The proposed grasp policy trained on UltraDexGrasp-20M demonstrates strong generalization to diverse objects and consistently outperforms the baselines. The best results are highlighted in bold.*

To answer Questions 1 and 2, we conduct experiments on the simulation benchmark. Both DP3 and our policy are trained on the UltraDexGrasp-20M dataset, while DexGraspNet, an optimization-based method, does not require training. As shown in Table [I](#S6.T1), our policy achieves an average success rate of 84.0%. It effectively handles small, medium, and large objects.
Notably, it demonstrates strong generalization with an 83.4% success rate on unseen objects. These results indicate that training on UltraDexGrasp-20M yields a policy with universal grasping capabilities and robust generalization across diverse objects. When trained on the same dataset, our policy outperforms DP3 by 37.3 percentage points, highlighting the superiority of the proposed grasp policy.
We attribute this improvement to the carefully designed point encoder, which enables the policy to capture fine-grained object geometry, as well as the effective use of a unidirectional attention mechanism to aggregate scene features.
DexGraspNet achieves an average success rate of 58.8% on small and medium objects, with significantly lower performance on small objects compared to medium ones. Additionally, DexGraspNet, which can only synthesize unimanual grasps, cannot handle large objects. These findings underscore the importance of multi-strategy dexterous grasping for bimanual robots.

#### VI-A5 Scaling with Training Data

![Figure](2603.05312v1/x4.png)

*Figure 5: The variation in performance with growing amounts of training data. The performance of the policy consistently improves as the volume of data increases.*

To answer Question 3, we compare the average success rates of our policy across both seen and unseen objects, trained with different amounts of data.
As shown in Fig. [5](#S6.F5), the overall performance of the policy consistently improves as the amount of training data increases.
For comparison, the average success rate of grasping data generation is 68.5%. When training frames exceed 1M, the performance of the learned policy significantly surpasses that of data generation.

#### VI-A6 Effectiveness of Design Choices

*TABLE II: Ablation study on policy design choices. The success rates (%) are reported. Both the bounded Gaussian distribution prediction and the unidirectional attention mechanism significantly improve performance.*

To answer Question 4, we conduct ablation studies on key policy design choices. Specifically, we compare our policy with two ablated versions: one without bounded Gaussian distribution prediction (w/o Dist. Pred.) and another without the unidirectional attention mechanism (w/o Uni. Attn.), where all tokens input to the transformer decoder can freely attend to each other. As reported in Tab. [II](#S6.T2), our policy achieves more than a 10% improvement in success rate over both ablation baselines, demonstrating the effectiveness of our design choices.

### VI-B Real-World Experiments

![Figure](2603.05312v1/x5.png)

*Figure 6: Real-world experiment setup. We employ two UR5e robots, two XHand, and two eye-on-base Azure Kinect DK cameras. A variety of objects are collected for testing.*

#### VI-B1 Experimental Setup

The real-world experimental setup is illustrated in Fig. [6](#S6.F6). Two UR5e robots are placed on a table at a distance of 0.9 m apart. Each robot’s end-effector is equipped with a 12-DoF XHand, attached via a flange. Two Azure Kinect DK cameras are positioned on the tabletop to capture scene point clouds as input for the policy.
The control frequency of the robotic system is set to 10 Hz.

#### VI-B2 Sim-to-Real Implementation Details

To facilitate sim-to-real transfer, we implement several procedures. We establish a consistent coordinate system for both domains and calibrate the intrinsic and extrinsic parameters of the real-world cameras to obtain point clouds aligned with those in the simulation. Since depth cameras are subject to noise, Statistical Outlier Removal (SOR) is applied to filter out outlier points.
Following [[[25](#bib.bib32)]], imaged point clouds for robots are used to further mitigate the sim-to-real gap caused by low-quality, noisy, and incomplete point clouds obtained in the real world.
Additionally, during demonstration generation, joint impedance randomization is employed to reduce the dynamics gap between the real world and simulation.

#### VI-B3 Benchmarks

We evaluate the grasping performance of the policies using 25 objects, which are categorized into small, medium, and large based on their sizes. Each policy is tested 15 times on each object, with a different object pose for each trial. Consistent with the simulation benchmark, objects in the small and medium categories are randomly placed within a 0.8 m $\times$ 0.2 m area, while large objects are randomly placed within a 0.15 m $\times$ 0.16 m area.
DP3 and DexGraspNet are chosen as baselines. For DexGraspNet, which requires object poses and complete object meshes, object poses are estimated using FoundationPose [[[35](#bib.bib62)]], and meshes are acquired through AR code scanning.

#### VI-B4 Main Results

*TABLE III: Results on real-world benchmarks. The success rates (%) are reported. The proposed grasp policy trained on UltraDexGrasp-20M demonstrates robust sim-to-real transfer and consistently outperforms the baselines.*

To answer Question 5, we directly deploy the proposed universal dexterous grasp policy, trained on UltraDexGrasp-20M, in real-world scenarios. The policy demonstrates robust zero-shot sim-to-real transfer and successfully handles novel objects of various shapes, sizes, and weights. Among the tested objects, the smallest has a volume of only 18 $\mathrm{cm}^{3}$, while the largest is 26,400 $\mathrm{cm}^{3}$; the lightest weighs only 3.6 g, and the heaviest is 1,095 g. Our policy effectively adapts grasping strategies to these diverse objects, exhibiting strategies such as three-finger tripod, whole-hand grasp, and bimanual grasp. As presented in Tab. [III](#S6.T3), our policy achieves robust grasps with an average success rate of 81.2%, significantly outperforming the baselines.

## VII Conclusion

In this paper, we introduce UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots. We propose a data generation pipeline that features a complementary integration of optimization-based grasp synthesis and planning-based demonstration generation to produce universal dexterous grasp data for bimanual robots. With this pipeline, we curate UltraDexGrasp-20M, a large-scale multi-strategy grasp dataset, which contains 20 million frames across 1,000 objects.
Tested on 600 diverse objects in simulation, our proposed policy achieves an average success rate of 84.0%.
Deployed directly in the real world, the policy demonstrates robust zero-shot sim-to-real transfer.

## References

-
[1]
S. Brahmbhatt, C. Tang, C. D. Twigg, C. C. Kemp, and J. Hays (2020)

ContactPose: a dataset of grasps with object contact and hand pose.

In European Conference on Computer Vision,

pp. 361–378.

Cited by: [§II](#S2.p1.1).

-
[2]
Y. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, et al. (2021)

DexYCB: a benchmark for capturing hand grasping of objects.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 9044–9053.

Cited by: [§II](#S2.p1.1).

-
[3]
J. Chen, Y. Ke, and H. Wang (2024)

Bodex: scalable and efficient robotic dexterous grasp synthesis using bilevel optimization.

arXiv preprint arXiv:2412.16490.

Cited by: [§II](#S2.p1.1),
[§IV-A](#S4.SS1.p5.1).

-
[4]
Z. Chen, Q. Yan, Y. Chen, T. Wu, J. Zhang, Z. Ding, J. Li, Y. Yang, and H. Dong (2025)

ClutterDexGrasp: a sim-to-real system for general dexterous grasping in cluttered scenes.

arXiv preprint arXiv:2506.14317.

Cited by: [§II](#S2.p2.1).

-
[5]
S. Deng, M. Yan, S. Wei, H. Ma, Y. Yang, J. Chen, Z. Zhang, T. Yang, X. Zhang, H. Cui, et al. (2025)

Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data.

arXiv preprint arXiv:2505.03233.

Cited by: [§I](#S1.p1.1),
[§II](#S2.p2.1).

-
[6]
H. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu (2023)

Anygrasp: robust and efficient grasp perception in spatial and temporal domains.

IEEE Transactions on Robotics 39 (5), pp. 3929–3945.

Cited by: [§I](#S1.p1.1),
[§II](#S2.p2.1).

-
[7]
H. Fang, C. Wang, M. Gou, and C. Lu (2020)

Graspnet-1billion: a large-scale benchmark for general object grasping.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 11444–11453.

Cited by: [§I](#S1.p1.1),
[§II](#S2.p2.1).

-
[8]
H. Fang, H. Yan, Z. Tang, H. Fang, C. Wang, and C. Lu (2025)

AnyDexGrasp: general dexterous grasping for different hands with human-level learning efficiency.

arXiv preprint arXiv:2502.16420.

Cited by: [§I](#S1.p1.1).

-
[9]
C. Goldfeder, M. Ciocarlie, H. Dang, and P. K. Allen (2009)

The columbia grasp database.

In 2009 IEEE international conference on robotics and automation,

pp. 1710–1716.

Cited by: [§II](#S2.p1.1).

-
[10]
J. He, D. Li, X. Yu, Z. Qi, W. Zhang, J. Chen, Z. Zhang, Z. Zhang, L. Yi, and H. Wang (2025)

DexVLG: dexterous vision-language-grasp model at scale.

arXiv preprint arXiv:2507.02747.

Cited by: [§II](#S2.p2.1).

-
[11]
L. Huang, H. Zhang, Z. Wu, S. Christen, and J. Song (2025)

FunGrasp: functional grasping for diverse dexterous hands.

IEEE Robotics and Automation Letters.

Cited by: [§II](#S2.p2.1).

-
[12]
H. Jiang, S. Liu, J. Wang, and X. Wang (2021)

Hand-object contact consistency reasoning for human grasps generation.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 11107–11116.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[13]
A. H. Li, P. Culbertson, J. W. Burdick, and A. D. Ames (2023)

Frogger: fast robust grasp generation via the min-weight metric.

In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

pp. 6809–6816.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[14]
P. Li, T. Liu, Y. Li, Y. Geng, Y. Zhu, Y. Yang, and S. Huang (2022)

Gendexgrasp: generalizable dexterous grasping.

arXiv preprint arXiv:2210.00722.

Cited by: [§II](#S2.p1.1).

-
[15]
Z. Liang, Y. Mu, Y. Wang, T. Chen, W. Shao, W. Zhan, M. Tomizuka, P. Luo, and M. Ding (2025)

Interaction-aware diffusion planning for adaptive dexterous manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 1745–1755.

Cited by: [§I](#S1.p2.1).

-
[16]
T. Lin, K. Sachdev, L. Fan, J. Malik, and Y. Zhu (2025)

Sim-to-real reinforcement learning for vision-based dexterous manipulation on humanoids.

arXiv preprint arXiv:2502.20396.

Cited by: [§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[17]
T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S. Zhu (2021)

Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator.

IEEE Robotics and Automation Letters 7 (1), pp. 470–477.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[18]
Y. Liu, Y. Yang, Y. Wang, X. Wu, J. Wang, Y. Yao, S. Schwertfeger, S. Yang, et al. (2024)

Realdex: towards human-like grasping for robotic dexterous hand.

arXiv preprint arXiv:2402.13853.

Cited by: [§II](#S2.p1.1).

-
[19]
T. G. W. Lum, A. H. Li, P. Culbertson, K. Srinivasan, A. D. Ames, M. Schwager, and J. Bohg (2024)

Get a grip: multi-finger grasp evaluation at scale enables robust sim-to-real transfer.

arXiv preprint arXiv:2410.23701.

Cited by: [§II](#S2.p2.1).

-
[20]
T. G. W. Lum, M. Matak, V. Makoviychuk, A. Handa, A. Allshire, T. Hermans, N. D. Ratliff, and K. Van Wyk (2024)

Dextrah-g: pixels-to-action dexterous arm-hand grasping with geometric fabrics.

arXiv preprint arXiv:2407.02274.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[21]
A. T. Miller and P. K. Allen (2004)

Graspit! a versatile simulator for robotic grasping.

IEEE Robotics & Automation Magazine 11 (4), pp. 110–122.

Cited by: [§II](#S2.p1.1).

-
[22]
A. Mousavian, C. Eppner, and D. Fox (2019)

6-dof graspnet: variational grasp generation for object manipulation.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 2901–2910.

Cited by: [§II](#S2.p2.1).

-
[23]
C. R. Qi, H. Su, K. Mo, and L. J. Guibas (2017)

Pointnet: deep learning on point sets for 3d classification and segmentation.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 652–660.

Cited by: [§V-B](#S5.SS2.p1.1).

-
[24]
C. R. Qi, L. Yi, H. Su, and L. J. Guibas (2017)

Pointnet++: deep hierarchical feature learning on point sets in a metric space.

Advances in neural information processing systems 30.

Cited by: [§V-B](#S5.SS2.p1.1).

-
[25]
Y. Qin, B. Huang, Z. Yin, H. Su, and X. Wang (2023)

Dexpoint: generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation.

In Conference on Robot Learning,

pp. 594–605.

Cited by: [§IV-B](#S4.SS2.p2.1),
[§VI-B2](#S6.SS2.SSS2.p1.1).

-
[26]
Y. Shao and C. Xiao (2024)

Bimanual grasp synthesis for dexterous robot hands.

IEEE Robotics and Automation Letters.

Cited by: [§II](#S2.p1.1).

-
[27]
R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. Van Wyk (2024)

Dextrah-rgb: visuomotor policies to grasp anything with dexterous hands.

arXiv preprint arXiv:2412.01791.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[28]
B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. V. Wyk, V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, N. Ratliff, and D. Fox (2023)

CuRobo: parallelized collision-free minimum-jerk robot motion generation.

External Links: 2310.17274

Cited by: [§IV-A](#S4.SS1.p5.1).

-
[29]
D. Turpin, L. Wang, E. Heiden, Y. Chen, M. Macklin, S. Tsogkas, S. Dickinson, and A. Garg (2022)

Grasp’d: differentiable contact-rich grasp synthesis for multi-fingered hands.

In European Conference on Computer Vision,

pp. 201–221.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[30]
D. Turpin, T. Zhong, S. Zhang, G. Zhu, J. Liu, R. Singh, E. Heiden, M. Macklin, S. Tsogkas, S. Dickinson, et al. (2023)

Fast-grasp’d: dexterous multi-finger grasp generation through differentiable simulation.

arXiv preprint arXiv:2306.08132.

Cited by: [§II](#S2.p1.1).

-
[31]
W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang (2023)

Unidexgrasp++: improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 3891–3902.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[32]
R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang (2022)

Dexgraspnet: a large-scale robotic dexterous grasp dataset for general objects based on simulation.

arXiv preprint arXiv:2210.02697.

Cited by: [§I](#S1.p1.1),
[§I](#S1.p2.1),
[§II](#S2.p1.1),
[§IV](#S4.p1.1),
[§VI-A2](#S6.SS1.SSS2.p1.1).

-
[33]
W. Wei, D. Li, P. Wang, Y. Li, W. Li, Y. Luo, and J. Zhong (2022)

DVGG: deep variational grasp generation for dextrous manipulation.

IEEE Robotics and Automation Letters 7 (2), pp. 1659–1666.

Cited by: [§II](#S2.p1.1).

-
[34]
Z. Wei, Z. Xu, J. Guo, Y. Hou, C. Gao, Z. Cai, J. Luo, and L. Shao (2024)

D (r, o) grasp: a unified representation of robot and object interaction for cross-embodiment dexterous grasping.

arXiv preprint arXiv:2410.01702.

Cited by: [§I](#S1.p1.1).

-
[35]
B. Wen, W. Yang, J. Kautz, and S. Birchfield (2024)

Foundationpose: unified 6d pose estimation and tracking of novel objects.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 17868–17879.

Cited by: [§VI-B3](#S6.SS2.SSS3.p1.2).

-
[36]
Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. (2023)

Unidexgrasp: universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 4737–4746.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[37]
J. Ye, K. Wang, C. Yuan, R. Yang, Y. Li, J. Zhu, Y. Qin, X. Zou, and X. Wang (2025)

Dex1B: learning with 1b demonstrations for dexterous manipulation.

arXiv preprint arXiv:2506.17198.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1),
[§II](#S2.p2.1).

-
[38]
Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu (2024)

3D diffusion policy: generalizable visuomotor policy learning via simple 3d representations.

In Proceedings of Robotics: Science and Systems (RSS),

Cited by: [§VI-A2](#S6.SS1.SSS2.p1.1).

-
[39]
H. Zhang, Z. Wu, L. Huang, S. Christen, and J. Song (2025)

RobustDexGrasp: robust dexterous grasping of general objects.

arXiv preprint arXiv:2504.05287.

Cited by: [§II](#S2.p2.1).

-
[40]
J. Zhang, H. Liu, D. Li, X. Yu, H. Geng, Y. Ding, J. Chen, and H. Wang (2024)

Dexgraspnet 2.0: learning generative dexterous grasping in large-scale synthetic cluttered scenes.

In 8th Annual Conference on Robot Learning,

Cited by: [§II](#S2.p2.1).

-
[41]
Y. Zhong, X. Huang, R. Li, C. Zhang, Y. Liang, Y. Yang, and Y. Chen (2025)

Dexgraspvla: a vision-language-action framework towards general dexterous grasping.

arXiv preprint arXiv:2502.20900.

Cited by: [§I](#S1.p1.1),
[§II](#S2.p2.1).

-
[42]
Y. Zhong, Q. Jiang, J. Yu, and Y. Ma (2025)

Dexgrasp anything: towards universal robotic dexterous grasping with physics awareness.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 22584–22594.

Cited by: [§II](#S2.p2.1).

-
[43]
R. Zurbrügg, A. Cramariuc, and M. Hutter (2025)

GraspQP: differentiable optimization of force closure for diverse and robust dexterous grasping.

arXiv preprint arXiv:2508.15002.

Cited by: [§II](#S2.p1.1).



Experimental support, please
[view the build logs](./2603.05312v1/__stdout.txt)
for errors. Generated by
[L A T E xml](https://math.nist.gov/~BMiller/LaTeXML/).






## Instructions for reporting errors


We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
methods listed below:




- Click the "Report Issue"  button, located in the page header.



Tip:** You can select the relevant text first, to include it in your report.


Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we
may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
should not be a barrier to accessing research. Thank you for your continued support in championing open access for
all.


Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).




BETA