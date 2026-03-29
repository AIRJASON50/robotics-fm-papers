[# Object-Centric Dexterous Manipulation from Human Motion Data Yuanpei Chen1,2, Chen Wang1, Yaodong Yang2, C. Karen Liu1 1Stanford University, 2Peking University ###### Abstract Manipulating objects to achieve desired goal states is a basic but important skill for dexterous manipulation. Human hand motions demonstrate proficient manipulation capability, providing valuable data for training robots with multi-finger hands. Despite this potential, substantial challenges arise due to the embodiment gap between human and robot hands. In this work, we introduce a hierarchical policy learning framework that uses human hand motion data for training object-centric dexterous robot manipulation. At the core of our method is a high-level trajectory generative model, learned with a large-scale human hand motion capture dataset, to synthesize human-like wrist motions conditioned on the desired object goal states. Guided by the generated wrist motions, deep reinforcement learning is further used to train a low-level finger controller that is grounded in the robot’s embodiment to physically interact with the object to achieve the goal. Through extensive evaluation across 10 household objects, our approach not only demonstrates superior performance but also showcases generalization capability to novel object geometries and goal states. Furthermore, we transfer the learned policies from simulation to a real-world bimanual dexterous robot system, further demonstrating its applicability in real-world scenarios. Project website: https://cypypccpy.github.io/obj-dex.github.io/](https://cypypccpy.github.io/obj-dex.github.io/).

*Figure 1: Our system uses human hand motion capture data and deep reinforcement learning to train dexterous robot hands for effective object-centric manipulation (i.e., learning to manipulate an object to follow a goal trajectory) in both simulation and real world.*

Keywords: Dexterous Manipulation, RL, Learning from Human

00footnotetext: Correspondence to Chen Wang <chenwj@stanford.edu>

## 1 Introduction

Developing bimanual multi-fingered robotic systems capable of handling complex manipulation tasks with human-level dexterity has been a longstanding goal in robotics research. When a robot engages general manipulation tasks beyond a simple pick-and-place, the definition of a task* can be difficult to establish. Previous works defined manipulation tasks in a variety of ways, including the state of the environment [[[1](#bib.bib1), [2](#bib.bib2)]], symbolic representations [[[3](#bib.bib3), [4](#bib.bib4)]], or language descriptions [[[5](#bib.bib5), [6](#bib.bib6)]]. Regardless of how the goals are specified, a common element across these definitions is an object-centric perspective focusing on the state of the objects being manipulated. As such, the goal of our work is to train a policy for a bimanual dexterous robot to manipulate the objects according to the task goal defined as a sequence of object pose trajectories.

Prior works primarily utilize deep reinforcement learning (RL) to learn object-centric dexterous manipulation skills [[[7](#bib.bib7), [8](#bib.bib8), [9](#bib.bib9)]]. Despite the success of these methods in tasks such as in-hand object reorientation [[[10](#bib.bib10), [11](#bib.bib11), [12](#bib.bib12)]], they typically focus solely on learning the movements of the fingers, neglecting the integrated coordination of both arms and hands. Training RL policy that controls both robot arms and two multi-finger hands is possible in theory, but presents substantial challenges in practice due to the high degree of freedom of the robot action space. Imitation learning (IL) can potentially tackle this challenge by leveraging the guidance from human motion data to assist policy learning. However, another challenge arises due to the morphological differences between human and robotic hands, often referred to as the “embodiment gap”. For instance, some robotic hands are designed with only four fingers, each significantly larger than a human’s, making it difficult to retarget the human hand trajectories to the robot hand while achieving the intended manipulation tasks.

One critical observation is that human finger motions are not consistently useful across various manipulation tasks due to the embodiment gap. In contrast, human wrist motions offer valuable information less sensitive to the embodiment gap, such as where to place the palm and how to interact with the objects in 3D space. Such motion cues significantly reduce the complexity of the high-dimensional action space in RL training, allowing it to focus on exploring finger motions to achieve the object-centric task goal. Based on this observation, we propose a hierarchical policy learning framework consisting of a high-level planner for the wrist and a low-level controller for the hand. The high-level planner is a generative-based policy, trained by imitation learning with human wrist movements, to generate robot arm actions conditioned on a desired trajectory of the object’s movements. Based on the generated arm motions, the low-level controller outputs fine-grained finger actions learned through RL exploration rather than imitation of human data. In addition, the RL policy also learns the residual wrist movements to integrate with arm action into the final robot action. The reward function for the RL training is the likelihood between the object’s movements during interaction and the reference trajectory. By harnessing the strengths of both RL and IL, we enable the robot to adapt to its own hand embodiment while keeping the training tractable by refining the action space using human data.

To ensure that the learned policy can adapt to a variety of reference object trajectories, not just a single scenario, we utilize ARCTIC [[[13](#bib.bib13)]], a comprehensive dataset with 51 hours of diverse hand-object manipulation motion capture sequences. Our experiments demonstrate that the learned policy exhibits generalization to novel object geometries and unseen motion trajectories. In addition, we successfully transfer our policy from simulation environments to a real-world bimanual dexterous robot, further validating its practical applicability in real-world manipulation tasks.

## 2 Related Works

### 2.1 Dexterous Manipulation

Dexterous manipulation is a long-standing research topic in robotics [[[14](#bib.bib14), [15](#bib.bib15), [16](#bib.bib16), [17](#bib.bib17)]]. Traditional methods rely on analytical dynamic models for trajectory optimization [[[18](#bib.bib18), [17](#bib.bib17), [14](#bib.bib14), [15](#bib.bib15)]], which fall short in complex tasks due to the simplification of contact dynamics. [[[19](#bib.bib19)]] and [[[20](#bib.bib20)]] propose model-based optimization methods to solve the contact-rich dexterous manipulation task, but a correct dynamics model of the environment is required. Recently, deep reinforcement learning (RL) has showcased promising results in training dexterous manipulation skills such as in-hand object reorientation [[[21](#bib.bib21), [11](#bib.bib11), [10](#bib.bib10), [22](#bib.bib22), [23](#bib.bib23), [24](#bib.bib24), [12](#bib.bib12), [25](#bib.bib25), [26](#bib.bib26), [27](#bib.bib27), [28](#bib.bib28)]], bimanual manipulation [[[7](#bib.bib7), [29](#bib.bib29), [8](#bib.bib8)]], sequential manipulation [[[30](#bib.bib30), [31](#bib.bib31), [32](#bib.bib32)]], and human-like activities [[[33](#bib.bib33)]]. [[[24](#bib.bib24)]] used pre-grasp techniques to learn a grasping policy to make the object follows a manual designed trajectory, but we focus on more challenging tasks, such as articulation manipulation, bimanual manipulation. Despite the progress, successfully training a dexterous RL policy often requires extensive reward engineering and system design, which limits its practicality in some scenarios. Besides RL, imitation learning (IL) is also widely used for training dexterous policies [[[34](#bib.bib34), [35](#bib.bib35)]]. By performing supervised-learning with human teleoperation data [[[36](#bib.bib36), [37](#bib.bib37), [38](#bib.bib38), [39](#bib.bib39), [40](#bib.bib40)]], prior works show impressive results in dexterous grasping [[[41](#bib.bib41), [42](#bib.bib42)]] and general manipulation tasks [[[43](#bib.bib43), [44](#bib.bib44), [45](#bib.bib45), [46](#bib.bib46), [47](#bib.bib47), [48](#bib.bib48), [49](#bib.bib49), [50](#bib.bib50)]]. However, teleoperation data are often expensive to collect due to the requirement of a real-world robot system. In this work, we focus on learning directly from human hand motion capture data without the need for teleoperation systems.

### 2.2 Learning from Human Motion

Recently, learning from human motion data has started to receive more attention because it allows scaling up data collection without robot hardware. Prior works leverage human videos [[[51](#bib.bib51), [52](#bib.bib52), [53](#bib.bib53), [54](#bib.bib54), [55](#bib.bib55), [56](#bib.bib56)]], motion capture data [[[57](#bib.bib57), [58](#bib.bib58), [59](#bib.bib59), [60](#bib.bib60), [61](#bib.bib61), [62](#bib.bib62)]] to extract valuable motion hints for manipulation such as trajectory-level plans [[[51](#bib.bib51), [59](#bib.bib59)]], object affordance [[[53](#bib.bib53)]] and motion priors [[[52](#bib.bib52), [55](#bib.bib55)]]. For dexterous manipulation, VideoDex [[[55](#bib.bib55)]], DexMV [[[43](#bib.bib43)]], DexTransfer [[[58](#bib.bib58)]] and DexCap [[[63](#bib.bib63)]] showcase the potential of using analytical methods (e.g., inverse kinematics) to retarget human hand motion to robot hardware, such as matching joint angles [[[55](#bib.bib55), [43](#bib.bib43)]], fingertip positions [[[63](#bib.bib63)]]. [[[64](#bib.bib64)]] showing independent control of the fingers and arm for capturing detailed hand manipulation of objects. DexPilot [[[38](#bib.bib38)]] formulate the retargeting objective as a non-linear optimization problem and retarget human motion to robot by minimizing a cost function. However, due to the embodiment gap between human and robot hands, position-based retargeting methods do not guarantee the replication of task success. In contrast, our approach uses human data as guidance for RL training, which learns the motion retargeting conditioned on the robot’s embodiment.
Notably, [[[65](#bib.bib65), [34](#bib.bib34), [66](#bib.bib66), [67](#bib.bib67), [68](#bib.bib68), [69](#bib.bib69)]] share the same idea of utilizing human data as guidance or reward for reinforcement learning. While these works focus on learning human whole-body control in simulation, we study dexterous manipulation with multi-fingered robotic hands and transfer the learned policy from simulation to the real world.

## 3 Task Formulation

*Figure 2: Overview of our framework. (A) Training: Firstly, we use human motion capture data to train a generation model to synthesize dual hand trajectory conditions on object trajectory. Then we use the RL to train a low-level robot controller conditioned on the dual hand trajectory generated by the trained high-level planner. During this process we augment the data in simulation to improve the high-level planner and low-level controller simultaneously (B) Inference: Given a single object goal trajectory, our framework generates dual hand reference trajectory and guides the low-level controller to accomplish the task.*

The goal of an object-centric manipulation task is to let the robot physically interact with the object to achieve the desired motion trajectory. We define the motion trajectory as the sequence of the object’s $SE(3)$ transformation $G=(\bm{g}_{1},\bm{g}_{2},\dots,\bm{g}_{T})$,

where each time step $\bm{g}_{i}=(\bm{g}_{i}^{R},\bm{g}_{i}^{T},g^{J}_{i})$ consists a 3D rotation $\bm{g}_{i}^{R}$, a 3D translation $\bm{g}_{i}^{T}$, and the joint angle $g_{i}^{J}$. $g^{J}_{i}$ can be omitted if the object is a single rigid body.

We then formulate an object-centric manipulation task as a Markov Decision Process (MDP) $\mathcal{M}=(\boldsymbol{S},\boldsymbol{A},\pi,\mathcal{T},R,\gamma,\rho,G)$, where $\boldsymbol{S}$ is the state space, $\boldsymbol{A}$ is the action space, $\pi$ is the agent’s policy, $\mathcal{T}(\bm{s}_{t+1}|\bm{s}_{t},\bm{a}_{t})$ is the transition distribution, $R$ is the reward function, $\gamma$ is the discount factor, and $\rho$ is the initial state distribution. The policy $\pi$ conditions on the reference object state trajectory $G$ and the current state $\bm{s}_{t}$, and generates robot action distributions $\bm{a}_{t}$ to maximize the likelihood between the future object states $(\bm{s}_{t+1},\bm{s}_{t+2},\dots,\bm{s}_{t+T})$ and the reference trajectory $G$. This formulation is versatile to adapt to downstream tasks, such as moving objects to desired locations and in-hand object re-orientation with bi-manual hands.

## 4 Method

In this section, we introduce our framework for object-centric manipulation. The overview of the framework is shown in Figure [2](#S3.F2). Our framework consists of three parts: high-level planner (Section [4.1](#S4.SS1)), low-level controller (Section [4.2](#S4.SS2)) and the data augmentation loop (Section [4.3](#S4.SS3)). The details of our sim-to-real policy transfer are introduced in Section [4.4](#S4.SS4).

### 4.1 High-Level Planner

One of the key challenges in training a dexterous robot policy, especially with bi-manual dexterous hands, is managing the high-dimensional action space. Despite the embodiment gap between human and robot hands, human hand motion, particularly wrist movements, provides highly informative hints for how to interact with objects and environments. Such motion hints not only narrow the action space for the robot but also provide guidance toward the task goal. Based on this observation, we first train a generative model to synthesize wrist motion by imitating human wrist movements in the motion capture data.

We train a Transformer-based generative model $\pi^{H}$ that takes object category ID $c$, and the desired object motion trajectory $G=(\bm{g}_{t},\bm{g}_{t+1},...,\bm{g}_{t+T})$ as inputs and outputs a sequence of 6-DoF wrist actions $(\bm{a}^{W}_{t},\bm{a}^{W}_{t+1},...,\bm{a}^{W}_{t+T})$, where each action $\bm{a}^{W}_{i}=(\bm{p}_{i}^{l},\bm{p}_{i}^{r})$ consists the 6-DoF pose of the left hand $\bm{p}_{i}^{l}$ and right hand $\bm{p}_{i}^{r}$ in SE(3). In our experiments, we use $T=10$. We leverage the entire ARCTIC dataset [[[13](#bib.bib13)]] to train $\pi^{H}$ with a behavior cloning algorithm. The total training samples consist of 339 sequences of human hand mocap trajectories, totaling 2.1 million steps.

### 4.2 Low-Level Controller

Based on the generated wrist trajectory from the high-level planner $\pi^{H}$, the low-level controller $\pi^{L}$ can start from a reasonable wrist pose and focus on learning the fine-grained finger motions to physically interact with the object to achieve the task goal $G$. We use Proximal Policy Optimization (PPO) [[[70](#bib.bib70)]] to train $\pi^{L}$. The policy $\pi^{L}$ takes the current observation $\bm{s}_{i}$
, the desired object motion trajectory $G=(\bm{g}_{t},\bm{g}_{t+1},...,\bm{g}_{t+T})$, and a sequence of 6-DoF wrist actions $(\bm{a}^{W}_{t},\bm{a}^{W}_{t+1},...,\bm{a}^{W}_{t+T})$ generated by high-level planner as inputs, and outputs the finger joint action $\bm{a}^{F}_{t}$. Here the observation $\bm{s}_{t}$ contains the object pose and robot proprioception. The reward function is defined as $r_{t}=\exp^{-(\lambda_{1}*\|\bm{g}^{R}_{t}-\bm{\hat{g}}_{t}^{R}\|_{2}+\lambda_{2}*\|\bm{g}^{T}_{t}-\bm{\hat{g}}_{t}^{T}\|_{2}+\lambda_{3}*\|g^{J}_{t}-\hat{g}_{t}^{J}\|_{2})}$, aiming to minimize the distance between object’s movements and the desired goal trajectory. $\bm{\hat{g}}_{t}^{R}$, $\bm{\hat{g}}_{t}^{T}$ and $\hat{g}_{t}^{J}$ is the current 3D translation, 3D rotation and joint angle of the object respectively. $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ are the hyperparameters to balance the weight of each component of the reward. In some scenarios, when the robot’s hand is larger than the human’s, the generated wrist actions from the high-level planner $\pi^{H}$ need to be adjusted accordingly. To achieve this, $\pi^{L}$ learns to output a residual wrist action $\Delta\bm{a}^{W}_{t}$ within a fixed range ($\pm 4$ centimeters for transition and $\pm 0.5$ radian for rotation). With the residual wrist actions, the policy can now adjust the wrist position to better synthesize the motion of the robot hand. The final robot action is a combination of $(\bm{a}^{W}_{t}+\Delta\bm{a}^{W}_{t},\bm{a}^{F}_{t})$. Please refer to Appendix [B](#A2) for more detail about the observation space and the reward function.

### 4.3 Data Augmentation for Generalization

Training policy only with the ARCTIC dataset [[[13](#bib.bib13)]] limits the diversity of task objects and generated motions. For instance, given a box twice the size of the one used in the dataset, it is challenging for the learned policies to manipulate the box effectively because of the unseen object geometry. To improve the generalization capability, it is critical to augment the data and train the policies to generalize in these scenarios. Thereby, we propose Data Augmentation Loop (DAL) to handle different object geometries and goal trajectories during training $\pi^{L}$.

Specifically, we introduce three types of augmentation during the RL training of the $\pi^{L}$: randomizing the object’s size in all three dimensions (width, length, height), randomizing the object’s initial pose, and modifying the goal trajectories of the object with waypoint interpolation. The low-level controller then learns to adapt to the augmented scenarios during the RL training. Finally, we collect the successful motion trajectories executed by the low-level controller and add them to the training dataset to fine-tune the high-level planner. By leveraging the adaptability of the RL training process, we can synthesize novel motion trajectories to improve the policy’s generalization beyond the scope of the original training data.
Detailed implementations of the Data Augmentation Loop can be found in Appendix [A](#A1).

### 4.4 Sim-to-Real Transfer

When deploying the policy to the real world, some observation cannot be accurately estimated, such as joint velocity and object velocity. We use the teacher-student policy distillation framework [[[11](#bib.bib11), [21](#bib.bib21), [71](#bib.bib71)]] to remove the dependence on these observation inputs from the policy. In real-world deployments, our system uses four top-down cameras to perform articulated object pose estimation. We use FoundationPose [[[72](#bib.bib72)]] to estimate the 6-DoF pose of different parts of the articulated object and calculate the joint angle between them. The entire object pose tracking system runs at the speed of 15Hz. The learned policies generate actions to control the robot, and the low-level controller is processed by a low-pass filter with an exponential moving average (EMA) smoothing factor [[[11](#bib.bib11)]] to reduce the jittering motions of the robot fingers.

For more details on the sim-to-real setups, please refer to Appendix [C](#A3).

## 5 Experiments

![Figure](/html/2411.04005/assets/x3.png)

*Figure 3: Overview of the environment setups. (a) Workspace of the simulation. We employ two Shadow Hands, each individually mounted on separate UR10e robots, arranged in an abreast configuration. (b.1) Object sets in the simulation. (b.2) Object sets in the real-world. (c) Workspace of the real-world, mirroring the simulation, the robot system uses the same Shadow Hands and UR10e robots as the simulation.*

The experiments are designed to answer the following research questions: (1) Can the high-level planner generalize to unseen trajectories and unseen objects?
(Sec. [5.1](#S5.SS1)) (2) Does our hierarchical approach help bridge the embodiment gap between human and robot hands? (Sec. [5.2](#S5.SS2)) (3) Can our trained policy generalize to unseen object geometries and goal trajectories? (Sec. [5.3](#S5.SS3)) (4) Can we transfer the policy from simulation to a real-world bimanual dexterous robot system? (Sec. [5.4](#S5.SS4)). In this section, we first introduce our training data and baseline setups, followed by the results for each research question. The experimental setups are shown in Figure.[3](#S5.F3). Our simulation experiments are all evaluated in 10 different seeds, and the real-world experiments are all evaluated in 20 different trials. The small numbers in each table represent the standard deviations across different seeds and trials.

*Table 1: Results for the high-level planner*

*Table 2: Results for the real-world experiments*

*Table 3: Results for the experiments of using one policy per object.*

Data. The ARCTIC dataset consists of 10 objects, each with 20 sequences of motion capture data. We choose 9 objects, each with 16 sequences of data, as our training set. We left the remaining objects and motion sequences as the unseen testing set.

Baselines. We compare our approach with the following methods and ablations: (1) Finger Joint Mapping: Match the finger joint angles between human and robot hands as demonstration and perform end-to-end imitation learning [[[43](#bib.bib43)]].
(2) Fingertip Mapping: Match the fingertip positions between human and robot hands using IK as demonstration and perform end-to-end imitation learning [[[63](#bib.bib63)]]. (3) Vanilla RL: Train a PPO policy [[[70](#bib.bib70)]] to learn the motions of the arm and hand together. (4) Ours: Full implementation of our hierarchical policy learning approach. (5) Ours w. FR: Our method with an additional fingertip-matching reward during training.
(6). Ours w/o. DAL: Our method without data augmentation loop.

### 5.1 Performance of the high-level planner

Tasks. To validate the accuracy of the learned high-level planner, we introduce three experimental tasks: (1) Trained: Testing the policy with trained objects and goal trajectories. (2) Unseen Traj: Testing the policy conditioned on unseen goal trajectories with trained objects. (3) Unseen Obj: Testing the policy with unseen objects and trained goal trajectories.

Metric.
Evaluation metric is defined as the distance between the ground truth wrist pose trajectory of the hand and the wrist pose trajectory of the left and right hand output by our high-level planner.
For each step, we compute translation error (TE) using
Euclidean distance in centimeters and orientation error (OE)
using angle difference.
We report the cumulative error of the entire motion sequence in Table [1](#S5.T1)

Results. Table [1](#S5.T1) shows that Ours performs the best in generating wrist motions, with the lowest cumulative translation and orientation error. More importantly, the performance does not drop when testing with unseen goal trajectories (Unseen Traj), which demonstrates the generalization capability of our high-level planner to novel object-centric task goals.

### 5.2 Effectiveness of learning from human with hierarchical pipeline

Task. To validate the effectiveness of our framework, we first train one policy conditioned on a single trajectory in the training set for each object and test its rollout performance. Each policy is trained on a single object.

Metric. We use Completion Rate as the evaluation metric, which indicates the percentage of the goal object trajectory completed by the policy. Completion is achieved if and only if the object’s pose error is smaller than a threshold ($5$ centimeters in translation, $2.5$ centimeters in object’s longest dimension multiplied by rotation angle, and $0.5$ radians in joint angle).
Each goal trajectory has a total of 500 action steps.

![Figure](/html/2411.04005/assets/x4.png)

*Figure 4: Experiments on the different embodiment. We study four types of the dexterous hand in three tasks from the Section [5.2](#S5.SS2).*

Results. Table [3](#S5.T3) demonstrates that our hierarchical learning framework outperforms traditional hand pose matching methods (Finger Joint Mapping, Fingertip Mapping) by 50% in completion rate, indicating that our low-level RL significantly helps in bridging the embodiment gap when learning from human data. Moreover, Ours surpasses Vanilla RL by 47.3% on average, underscoring the challenge of training arm and hand actions together with RL, and emphasizing the advantage of our high-level planner for guiding the RL in high-dimensional action space. Additionally, the inclusion of human finger motions in the RL reward (Ours (w. FR)) does not yield benefits and even leads to lower performance, validating our hypothesis that the embodiment gap makes human finger motion unsuitable for training robot actions. Lastly, our data augmentation loop further brings an additional 7% improvement (Ours vs. Ours (w.o. DAL)). We further apply our method to four different types of multi-fingered dexterous hands, varying in size and degree of freedom. The results are shown in Figure [4](#S5.F4). Our method achieved more than 50% completion rate for all hands, demonstrating that our framework can effectively transfer human data to different robot hand embodiments.

### 5.3 Generalization to unseen scenarios

Tasks. We design four types of tasks to test the policy’s generalization capability: (1). Single Obj - Trained Traj: One policy for each object, and testing with trained goal trajectories. (2). Single Obj - Unseen Traj: Same as prior but testing the policy conditioned on unseen goal trajectories. (3). Multi Obj - Trained Obj: One policy trained with all objects, and testing with trained objects. (4). Multi Obj - Unseen Obj: Same as prior but testing the policy with unseen objects. We use the completion rate as the evaluation metric same as in Section [5.2](#S5.SS2).

Results. In Table [4](#S5.T4), our algorithm surpassing the results of Vanilla RL on Single Obj - Unseen Traj and Multi Obj - Unseen Obj by more than 28%. This indicates that our hierarchical structure substantially improves generalization capabilities across unseen trajectories and unseen object geometries. Notably, unlike the Section [5.2](#S5.SS2), we observe an average 18% improvement in Table [3](#S5.T3) with our DAL (Ours vs. Ours (w.o. DAL)), showcasing that the DAL greatly helps generalization. Traditional mapping methods (Finger Joint Mapping, Fingertip Mapping) and Ours (w. FR) cannot generalize to unseen trajectories and unseen object geometries due to their dependency on the finger information.

![Figure](/html/2411.04005/assets/x5.png)

*Figure 5: Real-world experiment tasks. All clips include snapshot of the simulation (top row) and the real-world (bottom row). (a) Coffee Maker : Pick and lift the coffee machine. (b) Laptop: Lift up a laptop and place it back to the table. (c) Mixer: Rotate the Mixer and then open it. (d) Notebook: Open the notebook on the table. Please refer to our website for more visualization results.*

*Table 4: Results for the generalization experiments.*

### 5.4 Transfer from simulation to real-world

Tasks. We train one policy per object conditioned on a single goal trajectory in simulation and test its rollout performance on a real-world robot system. We use the completion rate as the evaluation metric.

Results. In Table [2](#S5.T2) real-world experiments, our approach has more than a 50% completion rate improvements compared to prior methods, which barely achieve any success ($Tassa et al. [2018]

Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. d. L. Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq, et al.

Deepmind control suite.

arXiv preprint arXiv:1801.00690*, 2018.

-
James et al. [2020]

S. James, Z. Ma, D. R. Arrojo, and A. J. Davison.

Rlbench: The robot learning benchmark & learning environment.

*IEEE Robotics and Automation Letters*, 5(2):3019–3026, 2020.

-
Garrett et al. [2018]

C. R. Garrett, T. Lozano-Perez, and L. P. Kaelbling.

Ffrob: Leveraging symbolic planning for efficient task and motion planning.

*The International Journal of Robotics Research*, 37(1):104–136, 2018.

-
Garrett et al. [2021]

C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez.

Integrated task and motion planning.

*Annual review of control, robotics, and autonomous systems*, 4:265–293, 2021.

-
Huang et al. [2022]

W. Huang, P. Abbeel, D. Pathak, and I. Mordatch.

Language models as zero-shot planners: Extracting actionable knowledge for embodied agents.

In *International Conference on Machine Learning*, pages 9118–9147. PMLR, 2022.

-
Brohan et al. [2023]

A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, R. Julian, et al.

Do as i can, not as i say: Grounding language in robotic affordances.

In *Conference on robot learning*, pages 287–318. PMLR, 2023.

-
Huang et al. [2023]

B. Huang, Y. Chen, T. Wang, Y. Qin, Y. Yang, N. Atanasov, and X. Wang.

Dynamic handover: Throw and catch with bimanual hands.

*arXiv preprint arXiv:2309.05655*, 2023.

-
Chen et al. [2022]

Y. Chen, T. Wu, S. Wang, X. Feng, J. Jiang, Z. Lu, S. McAleer, H. Dong, S.-C. Zhu, and Y. Yang.

Towards human-level bimanual dexterous manipulation with reinforcement learning.

*Advances in Neural Information Processing Systems*, 35:5150–5163, 2022.

-
Zhang et al. [2023]

Y. Zhang, A. Clegg, S. Ha, G. Turk, and Y. Ye.

Learning to transfer in-hand manipulations using a greedy shape curriculum.

In *Computer Graphics Forum*, volume 42, pages 25–36. Wiley Online Library, 2023.

-
Yin et al. [2023]

Z.-H. Yin, B. Huang, Y. Qin, Q. Chen, and X. Wang.

Rotating without seeing: Towards in-hand dexterity through touch.

*arXiv preprint arXiv:2303.10880*, 2023.

-
Chen et al. [2022]

T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal.

Visual dexterity: In-hand dexterous manipulation from depth.

*arXiv preprint arXiv:2211.11744*, 2022.

-
OpenAI et al. [2019]

OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang.

Solving rubik’s cube with a robot hand.

*CoRR*, abs/1910.07113, 2019.

-
Fan et al. [2023]

Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges.

Arctic: A dataset for dexterous bimanual hand-object manipulation.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12943–12954, 2023.

-
Bai and Liu [2014]

Y. Bai and C. K. Liu.

Dexterous manipulation using both palm and fingers.

In *2014 IEEE International Conference on Robotics and Automation (ICRA)*, pages 1560–1565. IEEE, 2014.

-
Kumar et al. [2014]

V. Kumar, Y. Tassa, T. Erez, and E. Todorov.

Real-time behaviour synthesis for dynamic hand-manipulation.

In *2014 IEEE International Conference on Robotics and Automation (ICRA)*, pages 6808–6815. IEEE, 2014.

-
Mason and Salisbury Jr [1985]

M. T. Mason and J. K. Salisbury Jr.

Robot hands and the mechanics of manipulation.

1985.

-
Mordatch et al. [2012]

I. Mordatch, Z. Popović, and E. Todorov.

Contact-invariant optimization for hand manipulation.

In *Proceedings of the ACM SIGGRAPH/Eurographics symposium on computer animation*, pages 137–144, 2012.

-
Chen et al. [2024]

S. Chen, J. Bohg, and C. K. Liu.

Springgrasp: An optimization pipeline for robust and compliant dexterous pre-grasp synthesis.

*arXiv preprint arXiv:2404.13532*, 2024.

-
Pang et al. [2023]

T. Pang, H. T. Suh, L. Yang, and R. Tedrake.

Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models.

*IEEE Transactions on robotics*, 2023.

-
Cheng et al. [2023]

X. Cheng, S. Patil, Z. Temel, O. Kroemer, and M. T. Mason.

Enhancing dexterity in robotic manipulation via hierarchical contact exploration.

*IEEE Robotics and Automation Letters*, 9(1):390–397, 2023.

-
Chen et al. [2021]

T. Chen, J. Xu, and P. Agrawal.

A system for general in-hand object re-orientation.

*Conference on Robot Learning*, 2021.

-
Qi et al. [2023a]

H. Qi, B. Yi, S. Suresh, M. Lambeta, Y. Ma, R. Calandra, and J. Malik.

General in-hand object rotation with vision and touch.

In *Conference on Robot Learning*, pages 2549–2564. PMLR, 2023a.

-
Qi et al. [2023b]

H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik.

In-hand object rotation via rapid motor adaptation.

In *Conference on Robot Learning*, pages 1722–1732. PMLR, 2023b.

-
Dasari et al. [2023]

S. Dasari, A. Gupta, and V. Kumar.

Learning dexterous manipulation from exemplar object trajectories and pre-grasps.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 3889–3896. IEEE, 2023.

-
Yang et al. [2024]

M. Yang, C. Lu, A. Church, Y. Lin, C. Ford, H. Li, E. Psomopoulou, D. A. Barton, and N. F. Lepora.

Anyrotate: Gravity-invariant in-hand object rotation with sim-to-real touch.

*arXiv preprint arXiv:2405.07391*, 2024.

-
Pitz et al. [2023]

J. Pitz, L. Röstel, L. Sievers, and B. Bäuml.

Dextrous tactile in-hand manipulation using a modular reinforcement learning architecture.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 1852–1858. IEEE, 2023.

-
Handa et al. [2022]

A. Handa, A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, et al.

Dextreme: Transfer of agile in-hand manipulation from simulation to reality.

*arXiv preprint arXiv:2210.13702*, 2022.

-
Khandate et al. [2023]

G. Khandate, S. Shang, E. T. Chang, T. L. Saidi, Y. Liu, S. M. Dennis, J. Adams, and M. Ciocarlie.

Sampling-based exploration for reinforcement learning of dexterous manipulation.

*arXiv preprint arXiv:2303.03486*, 2023.

-
Lin et al. [2024]

T. Lin, Z.-H. Yin, H. Qi, P. Abbeel, and J. Malik.

Twisting lids off with two hands.

*arXiv preprint arXiv:2403.02338*, 2024.

-
Xu et al. [2023]

K. Xu, Z. Hu, R. Doshi, A. Rovinsky, V. Kumar, A. Gupta, and S. Levine.

Dexterous manipulation from images: Autonomous real-world rl via substep guidance.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 5938–5945. IEEE, 2023.

-
Chen et al. [2023]

Y. Chen, C. Wang, L. Fei-Fei, and C. K. Liu.

Sequential dexterity: Chaining dexterous policies for long-horizon manipulation.

*arXiv preprint arXiv:2309.00987*, 2023.

-
Gupta et al. [2021]

A. Gupta, J. Yu, T. Z. Zhao, V. Kumar, A. Rovinsky, K. Xu, T. Devlin, and S. Levine.

Reset-free reinforcement learning via multi-task learning: Learning dexterous manipulation behaviors without human intervention.

In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pages 6664–6671. IEEE, 2021.

-
Zakka et al. [2023]

K. Zakka, L. Smith, N. Gileadi, T. Howell, X. B. Peng, S. Singh, Y. Tassa, P. Florence, A. Zeng, and P. Abbeel.

Robopianist: A benchmark for high-dimensional robot control.

*arXiv preprint arXiv:2304.04150*, 2023.

-
Rajeswaran et al. [2017]

A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine.

Learning complex dexterous manipulation with deep reinforcement learning and demonstrations.

*arXiv preprint arXiv:1709.10087*, 2017.

-
Radosavovic et al. [2021]

I. Radosavovic, X. Wang, L. Pinto, and J. Malik.

State-only imitation learning for dexterous manipulation.

In *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 7865–7871. IEEE, 2021.

-
Arunachalam et al. [2023]

S. P. Arunachalam, I. Güzey, S. Chintala, and L. Pinto.

Holo-dex: Teaching dexterity with immersive mixed reality.

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 5962–5969. IEEE, 2023.

-
Guzey et al. [2023]

I. Guzey, B. Evans, S. Chintala, and L. Pinto.

Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play.

*arXiv preprint arXiv:2303.12076*, 2023.

-
Handa et al. [2020]

A. Handa, K. Van Wyk, W. Yang, J. Liang, Y.-W. Chao, Q. Wan, S. Birchfield, N. Ratliff, and D. Fox.

Dexpilot: Vision-based teleoperation of dexterous robotic hand-arm system.

In *2020 IEEE International Conference on Robotics and Automation (ICRA)*, pages 9164–9170. IEEE, 2020.

-
Sivakumar et al. [2022]

A. Sivakumar, K. Shaw, and D. Pathak.

Robotic telekinesis: Learning a robotic hand imitator by watching humans on youtube.

*arXiv preprint arXiv:2202.10448*, 2022.

-
Qin et al. [2023]

Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox.

Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system.

*arXiv preprint arXiv:2307.04577*, 2023.

-
Mandikal and Grauman [2021]

P. Mandikal and K. Grauman.

Learning dexterous grasping with object-centric visual affordances.

In *2021 IEEE international conference on robotics and automation (ICRA)*, pages 6169–6176. IEEE, 2021.

-
Chen et al. [2022]

Z. Q. Chen, K. Van Wyk, Y.-W. Chao, W. Yang, A. Mousavian, A. Gupta, and D. Fox.

Learning robust real-world dexterous grasping policies via implicit shape augmentation.

*arXiv preprint arXiv:2210.13638*, 2022.

-
Qin et al. [2022]

Y. Qin, Y.-H. Wu, S. Liu, H. Jiang, R. Yang, Y. Fu, and X. Wang.

Dexmv: Imitation learning for dexterous manipulation from human videos.

In *European Conference on Computer Vision*, pages 570–587. Springer, 2022.

-
Cui et al. [2022]

Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto.

From play to policy: Conditional behavior generation from uncurated robot data.

*arXiv e-prints*, pages arXiv–2210, 2022.

-
Haldar et al. [2023]

S. Haldar, J. Pari, A. Rai, and L. Pinto.

Teach a robot to fish: Versatile imitation from one minute of demonstrations.

*arXiv preprint arXiv:2303.01497*, 2023.

-
Qin et al. [2022]

Y. Qin, H. Su, and X. Wang.

From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation.

*IEEE Robotics and Automation Letters*, 7(4):10873–10881, 2022.

-
Arunachalam et al. [2022]

S. P. Arunachalam, S. Silwal, B. Evans, and L. Pinto.

Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation.

*arXiv preprint arXiv:2203.13251*, 2022.

-
Guzey et al. [2023]

I. Guzey, Y. Dai, B. Evans, S. Chintala, and L. Pinto.

See to touch: Learning tactile dexterity through visual incentives.

*arXiv preprint arXiv:2309.12300*, 2023.

-
Lin et al. [2024]

T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik.

Learning visuotactile skills with two multifingered hands.

*arXiv preprint arXiv:2404.16823*, 2024.

-
Ze et al. [2024]

Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu.

3d diffusion policy.

*arXiv preprint arXiv:2403.03954*, 2024.

-
Bahl et al. [2022]

S. Bahl, A. Gupta, and D. Pathak.

Human-to-robot imitation in the wild.

*arXiv preprint arXiv:2207.09450*, 2022.

-
Mandikal and Grauman [2022]

P. Mandikal and K. Grauman.

Dexvip: Learning dexterous grasping with human hand pose priors from video.

In *Conference on Robot Learning*, pages 651–661. PMLR, 2022.

-
Bahl et al. [2023]

S. Bahl, R. Mendonca, L. Chen, U. Jain, and D. Pathak.

Affordances from human videos as a versatile representation for robotics.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13778–13790, 2023.

-
Bharadhwaj et al. [2023]

H. Bharadhwaj, A. Gupta, V. Kumar, and S. Tulsiani.

Towards generalizable zero-shot manipulation via translating human interaction plans.

*arXiv preprint arXiv:2312.00775*, 2023.

-
Shaw et al. [2023]

K. Shaw, S. Bahl, and D. Pathak.

Videodex: Learning dexterity from internet videos.

In *Conference on Robot Learning*, pages 654–665. PMLR, 2023.

-
Bahety et al. [2024]

A. Bahety, P. Mandikal, B. Abbatematteo, and R. Martín-Martín.

Screwmimic: Bimanual imitation from human videos with screw space projection.

*arXiv preprint arXiv:2405.03666*, 2024.

-
Zhang et al. [2023]

H. Zhang, S. Christen, Z. Fan, L. Zheng, J. Hwangbo, J. Song, and O. Hilliges.

Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation.

*arXiv preprint arXiv:2309.03891*, 2023.

-
Chen et al. [2022]

Z. Q. Chen, K. Van Wyk, Y.-W. Chao, W. Yang, A. Mousavian, A. Gupta, and D. Fox.

Dextransfer: Real world multi-fingered dexterous grasping with minimal human demonstrations.

*arXiv preprint arXiv:2209.14284*, 2022.

-
Wang et al. [2023]

C. Wang, L. Fan, J. Sun, R. Zhang, L. Fei-Fei, D. Xu, Y. Zhu, and A. Anandkumar.

Mimicplay: Long-horizon imitation learning by watching human play.

*arXiv preprint arXiv:2302.12422*, 2023.

-
Garcia-Hernando et al. [2018]

G. Garcia-Hernando, S. Yuan, S. Baek, and T.-K. Kim.

First-person hand action benchmark with rgb-d videos and 3d hand pose annotations.

In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 409–419, 2018.

-
Hasson et al. [2019]

Y. Hasson, G. Varol, D. Tzionas, I. Kalevatykh, M. J. Black, I. Laptev, and C. Schmid.

Learning joint reconstruction of hands and manipulated objects.

In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11807–11816, 2019.

-
Taheri et al. [2020]

O. Taheri, N. Ghorbani, M. J. Black, and D. Tzionas.

Grab: A dataset of whole-body human grasping of objects.

In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16*, pages 581–600. Springer, 2020.

-
Wang et al. [2024]

C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu.

Dexcap: Scalable and portable mocap data collection system for dexterous manipulation.

*arXiv preprint arXiv:2403.07788*, 2024.

-
Ye and Liu [2012]

Y. Ye and C. K. Liu.

Synthesis of detailed hand manipulations using contact sampling.

*ACM Transactions on Graphics (ToG)*, 31(4):1–10, 2012.

-
Ho and Ermon [2016]

J. Ho and S. Ermon.

Generative adversarial imitation learning.

*Advances in neural information processing systems*, 29, 2016.

-
Peng et al. [2018]

X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne.

Deepmimic: Example-guided deep reinforcement learning of physics-based character skills.

*ACM Transactions On Graphics (TOG)*, 37(4):1–14, 2018.

-
Peng et al. [2021]

X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa.

Amp: Adversarial motion priors for stylized physics-based character control.

*ACM Transactions on Graphics (ToG)*, 40(4):1–20, 2021.

-
Peng et al. [2022]

X. B. Peng, Y. Guo, L. Halper, S. Levine, and S. Fidler.

Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters.

*ACM Transactions On Graphics (TOG)*, 41(4):1–17, 2022.

-
Wang et al. [2023]

Y. Wang, J. Lin, A. Zeng, Z. Luo, J. Zhang, and L. Zhang.

Physhoi: Physics-based imitation of dynamic human-object interaction.

*arXiv preprint arXiv:2312.04393*, 2023.

-
Schulman et al. [2017]

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.

Proximal policy optimization algorithms.

*arXiv preprint arXiv:1707.06347*, 2017.

-
Rusu et al. [2016]

A. A. Rusu, S. G. Colmenarejo, Ç. Gülçehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell.

Policy distillation.

In *ICLR (Poster)*, 2016.

-
Wen et al. [2023]

B. Wen, W. Yang, J. Kautz, and S. Birchfield.

Foundationpose: Unified 6d pose estimation and tracking of novel objects.

*arXiv preprint arXiv:2312.08344*, 2023.

-
Yu et al. [2020]

T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine.

Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning.

In *Conference on robot learning*, pages 1094–1100. PMLR, 2020.

-
Charlesworth and Montana [2021]

H. J. Charlesworth and G. Montana.

Solving challenging dexterous manipulation tasks with trajectory optimisation and reinforcement learning.

In *International Conference on Machine Learning*, pages 1496–1506. PMLR, 2021.

-
Ross et al. [2011]

S. Ross, G. Gordon, and D. Bagnell.

A reduction of imitation learning and structured prediction to no-regret online learning.

In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

-
Li et al. [2023]

J. Li, A. Clegg, R. Mottaghi, J. Wu, X. Puig, and C. K. Liu.

Controllable human-object interaction synthesis.

*arXiv preprint arXiv:2312.03913*, 2023.

-
Pezzato et al. [2023]

C. Pezzato, C. Salmi, M. Spahn, E. Trevisan, J. Alonso-Mora, and C. H. Corbato.

Sampling-based model predictive control leveraging parallelizable physics simulations, 2023.

## Appendix A Data Augmentation Loop

### A.1 Pseudo Code

*Algorithm 1 Data Augmentation Loop*

### A.2 Detail of the Data Augmentation Loop

Below are the details for each augmentation. The unit of length is centimeters and the unit of angle is degrees.

-
•

Random the object’s mesh scales with a small scale:

-
–

The scale of the width of the manipulated object ranges from 0.9 to 1.1.

-
–

The scale of the length of the manipulated object ranges from 0.9 to 1.1.

-
–

The scale of the height of the manipulated object ranges from 0.9 to 1.1.

-
•

Random the object’s initial pose with a small scale:

-
–

The x-coordinate of the manipulated object ranges from -0.02 to 0.02.

-
–

The y-coordinate of the manipulated object ranges from -0.02 to 0.02.

-
–

The manipulated object’s z-axis Euler degree ranges from 0 to 30.

-
•

Modify the goal trajectories of the object with waypoint interpolation:

-
–

The x-coordinate of the goal trajectories position is added by ranges from -0.02 to 0.02.

-
–

The y-coordinate of the goal trajectories position is added by ranges from -0.02 to 0.02.

-
–

The z-coordinate of the goal trajectories position is added by ranges from -0.02 to 0.02.

## Appendix B Detail Implementation of RL in Simulation

### B.1 Observation Space

Table.[5](#A3.T5) gives the specific information of the observation space.

### B.2 Reward Design

Denote the $\bm{\hat{g}}_{i}^{R}$, $\bm{\hat{g}}_{i}^{T}$ and $\hat{g}_{i}^{J}$ is the current 3D translation, 3D rotation and joint angle of the object respectively, the desired object 3D rotation $\bm{g}_{i}^{R}$, the desired object 3D translation $\bm{g}_{i}^{T}$, and the desired object joint angle $g_{i}^{J}$. $\lambda_{1}$, $\lambda_{2}$ and $\lambda_{3}$ is the hyperparameters to balance the weight of each component of the reward.

The reward function is defined as:

$$r_{t}=\exp^{-(\lambda_{1}*\|\bm{g}^{R}_{t}-\bm{\hat{g}}_{t}^{R}\|_{2}+\lambda_{2}*\|\bm{g}^{T}_{t}-\bm{\hat{g}}_{t}^{T}\|_{2}+\lambda_{3}*\|g^{J}_{t}-\hat{g}_{t}^{J}\|_{2})}$$ \tag{1}

where $\lambda_{1}=20$, $\lambda_{2}=1$, and $\lambda_{3}=5$.

We use an exponential map in the reward function, which is an effective reward shaping technique used in the case to minimize the distance, introduced by [[[73](#bib.bib73), [74](#bib.bib74)]]. To improve the calculation efficiency, we use quaternion to represent the object orientation. The angular position difference is then computed through the dot product between the normalized goal quaternion and the current object’s quaternion.

## Appendix C Detail Implementation in Real-World

![Figure](/html/2411.04005/assets/x6.png)

*Figure 6: Setup of the cameras.*

*Table 5: Observation space of our framework in simulation.*

### C.1 Perception

Our perception setup is shown in Figure [6](#A3.F6). We arranged 4 identical Femto Bolt cameras around the table and face towards the object. We use FoundationPose [[[72](#bib.bib72)]] to estimate the articulated object pose. To remove the abnormal results, we compare each pose to the desired pose and remove the pose if the error is smaller than a threshold ($5$ centimeters in translation and $0.5$ radians in orientation). Finally, we average the rest of the poses as our observation for the policy. If none of the poses is smaller than the threshold, we continue to use the pose from the previous frame.

### C.2 Policy Distillation

We use the DAgger [[[75](#bib.bib75)]] algorithm for policy distillation. Table.[6](#A3.T6) gives the specific information of the observation space of the distilled policy.

*Table 6: Observation space of our framework in the real-world.*

### C.3 Resets

In the real-world evaluation, we used a trajectory from the ARCTIC dataset as our goal trajectory. In terms of resets in our real-world experiments, we pose the object within 3 centimeters and 0.5 radians of the initial pose of the goal trajectory during resets. We evaluated 20 times and reported the completion rate.

### C.4 Evaluation

When evaluating real-world experiments, we take the same metric (completion rate) as in simulation. It will fail if the tolerance is exceeded($5$ centimeters in translation, $2.5$ centimeters in object’s longest dimension multiplied by rotation angle, and $0.5$ radians in joint angle), and record the completion rate.

## Appendix D Hyperparameters of the PPO

Table.[7](#A4.T7) gives the hyperparameters of the PPO.

*Table 7: Hyperparameters of PPO.*

## Appendix E Domain Randomization

Isaac Gym provides lots of domain randomization functions for RL training. We add the randomization for all the tasks as shown in Table. [8](#A5.T8) for each environment. we generate new randomization every 1000 simulation steps.

*Table 8: Domain randomization of all the tasks.*

## Appendix F Goal Representation

Our framework requires the full 6D pose of the object trajectory at each time-step. For downstream tasks, we are able to get the trajectories in many ways. For example, we can specify a few key poses of the object based on the task, and use linear interpolation to generate the pose trajectories between the key poses. Another solution is to use some object motion synthesis methods in the field of graphics to generate the 6D pose of the object trajectory, such as [[[76](#bib.bib76)]].
We design an additional experiment to prove that our method can work with this setup. Taking a task “lifting up the box in the air and dropping it down” as an example, we can just simply manually define the pose of the box in the air and the landing point after picking it up as the key poses, then interpolate it as our goal trajectory, and train the robot to manipulate the object to follow the goal trajectory. The results are shown in Table [9](#A6.T9) and the snapshot are shown in Figure [7](#A6.F7). The results show that we can complete the task under this setting, and shows that our framework is extensible.

*Table 9: Results for the goal representation experiments.*

![Figure](/html/2411.04005/assets/x7.png)

*Figure 7: Snapshots of the goal representation experiment.*

## Appendix G Time-indexed of the Goal Trajectory

For the time-indexed of the goal trajectory, we randomize the sampling of input trajectories with various time gaps during training. We added an additional experiment to show that varying the time gap does not significantly affect our performance, indicating that our approach maintains generalization despite these temporal constraints. Using Coffee Maker as an example, we interpolated the goal trajectories provided by ARCTIC datasets to varying levels, and tested the completion rate of our policy on it. Origin represents the original goal trajectory from ARCTIC, and Skip 2 and Skip 1 represent skipping 2/1 poses between every two poses on the basis of origin goal trajectory. Inter 2 and Inter 1 represent insertion of 2/1 pose between every two poses according to the linear interpolation method on the basis of origin goal trajectory. Our policy generates robot actions conditioned on a sequence of object goal poses, with the time gaps defined by the distance between each consecutive goal pose. In this experiment, we randomly use time gaps of 0 to 3 units when sampling the object trajectory to test whether the policy is sensitive to these time gaps (Random Sample). The results is shown in Table [10](#A7.T10), the performance of the trained policy will not vary greatly.

*Table 10: Results for the time-indexed experiments.*

## Appendix H Model-base Baselines

We add a baseline of the model-based method in the simulation, the result is shown in Table [11](#A8.T11). We use the sample-based model predictive control method that is modified from [[[77](#bib.bib77)]] as our low-level controller in our tasks. Our method outperforms the sample-based model predictive control by 46.3% on average.

*Table 11: Results for the experiments of using one policy per object.*

[◄](/html/2411.04004)

[Feeling lucky?](/feeling_lucky)

[Conversion report](/log/2411.04005)
[Report an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2411.04005)
[View original on arXiv](https://arxiv.org/abs/2411.04005)[►](/html/2411.04006)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Dec 5 14:37:16 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)