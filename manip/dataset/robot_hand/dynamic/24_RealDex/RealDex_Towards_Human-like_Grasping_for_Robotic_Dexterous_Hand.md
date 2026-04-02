[# RealDex: Towards Human-like Grasping for Robotic Dexterous Hand Yumeng Liu1,2∗ Yaxun Yang1∗ Youzhuo Wang1∗ Xiaofei Wu1 Jiamin Wang1 Yichen Yao1 Sören Schwertfeger1 Sibei Yang1 Wenping Wang3 Jingyi Yu1 Xuming He1 Yuexin Ma1† 1ShanghaiTech University, 2The University of Hong Kong, 3Texas A&M University lym29@connect.hku.hk, {yangyx12022,wangyzh2023,mayuexin}@shanghaitech.edu.cn ###### Abstract In this paper, we introduce RealDex, a pioneering dataset capturing authentic dexterous hand grasping motions infused with human behavioral patterns, enriched by multi-view and multimodal visual data. Utilizing a teleoperation system, we seamlessly synchronize human-robot hand poses in real time. This collection of human-like motions is crucial for training dexterous hands to mimic human movements more naturally and precisely. RealDex holds immense promise in advancing humanoid robot for automated perception, cognition, and manipulation in real-world scenarios. Moreover, we introduce a cutting-edge dexterous grasping motion generation framework, which aligns with human experience and enhances real-world applicability through effectively utilizing Multimodal Large Language Models. Extensive experiments have demonstrated the superior performance of our method on RealDex and other open datasets. The dataset and associated code are available at https://4dvlab.github.io/RealDex_page/](https://4dvlab.github.io/RealDex_page/).

![[Uncaptioned image]](extracted/6052044/img/teaser_test.png)

*Figure 1: RealDex provides extensive realistic grasping motions of dexterous hand, synchronized with human poses and reflective of typical human motion behaviors. Utilizing the visual data of target object as input, models trained on RealDex are capable of generating human-like grasping motions for robotic hand manipulation, making them highly applicable in real-world scenarios. The right figure provides a visual gallery showcasing various objects alongside grasping poses in RealDex.*

## 1 Introduction

Compared to standard parallel grippers with 7 degrees of freedom (DoF) [Fang et al. ([2022](#bib.bib12), [2020](#bib.bib11)); Dai et al. ([2023](#bib.bib9))], five-fingered dexterous hands [Sha ([2005](#bib.bib31))] boasting over 24 DoF closely mimic human hand structure and movement. This sophistication enhances their ability to adapt to human-centered environments and execute more intricate and nuanced operations, such as using complex tools, with greater ease. Furthermore, robots that mirror human appearance and behavior tend to be more quickly embraced and comprehended by people, making them particularly advantageous in service and healthcare sectors.

Dexterous grasping represents an initial step in the interaction between the dexterous hand and the real world, forming the foundation for research and application of human-like service robots. Due to the high DoF of dexterous hand, traditional motion planning-based methods [Andrews and Kry ([2013](#bib.bib2)); Bai and Liu ([2014](#bib.bib4)); Dogar and Srinivasa ([2010](#bib.bib10))] struggle to handle such complex hand joint movements. In the absence of real-world data, prevalent methodologies pivot towards reinforcement learning (RL) for training dexterous hand grasping behaviors. Considering real-world applications, some of them [Chen et al. ([2022](#bib.bib7)); Qin et al. ([2022](#bib.bib29)); Wan et al. ([2023](#bib.bib41))] have started using raw visual data, such as image or point cloud, as inputs to directly train grasping strategies for dexterous hand. However, reward functions in RL are artificially devised and cannot encompass all desired outcomes. In particular, it is difficult to model human behavioral habits in the reward mechanism. As a result, RL-trained dexterous hand grasping behaviors are only partially physically plausible and fail to truly mimic human-object interaction habits, not fully harnessing the inherent potential of humanoid robots. For example, people usually pick up a cup by its handle instead of sticking their thumb inside it, even though both ways can lift the cup physiccally. Without modeling human motion priors, current methods face notable challenges in replicating human-like grasping motions for robotic dexterous hands.

To overcome the challenges posed by data scarcity, some works [Wei et al. ([2022](#bib.bib44)); Li et al. ([2023](#bib.bib22)); Wang et al. ([2023](#bib.bib43)); Liu et al. ([2020](#bib.bib23))] have proposed synthesized dexterous grasping pose datasets. Models trained on these datasets with the supervision of ground truth can generate improved goal poses for dexterous grasping. However, these datasets, derived from simulation environments via optimization methods, rely on artificially defined energy functions and necessitate detailed physical and geometric information about objects, which are impractical for real-world applications. Furthermore, the substantial disparity between synthesized and real data makes the trained algorithms exhibit poor real-world applicability and limited generalization abilities. Additionally, the inability of simulated data to effectively model human-like behavior results in grasping poses devoid of human behavioral insights.

The teleoperation system of ShadowHand [Tel ([2022](#bib.bib37))] with integrated signal synchronization and control mechanisms enables the alignment of poses between the dexterous hand and human hand in real time. Utilizing this system, we have created a comprehensive real-world dataset, RealDex, for dexterous hand grasping, ensuring the actions closely mirror human grasping patterns. Moreover, we have established a multi-view vision system with multiple RGB-D cameras to support research in vision-based grasping algorithms. RealDex contains 52 objects with varied scales, shapes, and materials, 2.6k sequences of grasping motions with diverse initial positions and directions, and around 955k frames of visual data, including images and point clouds. This groundbreaking dataset, driven by human hands and gathered in real-world settings, is significant for robotic dexterous hand-object grasping. It stands apart from previous datasets in three critical ways: it replicates human hand-object interactions, enhancing the training of dexterous hands to more accurately emulate human movements, thereby advancing humanoid robotics; its precise ground truth data for real dexterous hand motions bridges the gap between model training and real application; and its multi-view, multimodal visual data pave the way for the development of diverse vision-based manipulation algorithms, fostering automated perception, cognition, and action of dexterous hands in real worlds.

Leveraging the RealDex dataset, we have developed a novel and effective method for dexterous grasping, which involves two key stages: grasp pose generation and motion synthesis, using only the point cloud data of objects captured by real vision sensor as input. Thanks to the dataset’s rich representation of human-like dexterous hand motions, our model can innately learn human behavior patterns, producing practical grasping motions. Recognizing the robust generalization capabilities of Multimodal Large Language Models (MLLMs) [Google ([2023](#bib.bib15)); OpenAI ([2023](#bib.bib26))], we have integrated an MLLM Selection Module. This module is adept at choosing the most natural, physically plausible, and human-like grasp pose from multiple generated options, enhancing the final motion synthesis. The inherent general knowledge within large models bolsters our method’s generalization capacity, ensuring effective performance even with unseen objects and thus significantly benefiting practical applications. We have conducted extensive experiments on RealDex and other open datasets, including dexterous grasping dataset and human grasping dataset, and also have tested our method on real robot hand. Quantitative and qualitative results demonstrate that our method outperforms others obviously for generation human-like practicable dexterous grasping motions.

## 2 Related Work

### 2.1 Dataset for Dexterous Hand Grasping

The complexity of dexterous hands, while allowing for precise operations, poses challenges in annotating grasping motions. Many datasets [Liu et al. ([2020](#bib.bib23)); Hasson et al. ([2019a](#bib.bib16)); Lundell et al. ([2021](#bib.bib24)); Goldfeder et al. ([2009](#bib.bib14))] use the planner [Miller and Allen ([2004](#bib.bib25))] to generate grasping poses. However, this method’s reliance on a restricted eigengrasp space limits the diversity of the data, failing to capture the full range of a multi-fingered hand’s dexterity. Some datasets try to improve the quality and diversity of the grasp poses. DVGG [Wei et al. ([2022](#bib.bib44))] utilizes MuJoCo physics simulator [Todorov et al. ([2012](#bib.bib38))] to synthesize data and filtering out unstable grasp poses through a hand-shaking test. DexGraspNet [Wang et al. ([2023](#bib.bib43))] leverages a deeply accelerated differentiable force closure estimator and synthesize stable and diverse grasp poses on a large scale. MultiDex [Li et al. ([2023](#bib.bib22))] synthesizes versatile dexterous grasp poses across five different robot hands. Nevertheless, existing datasets for dexterous grasping consist solely of synthesized data created through optimization, resulting in a significant disconnect with real robotic hand data and the absence of human-like behavior modeling. It is noted that these datasets only contain the static grasp poses and without any dynamic grasping motion sequences. Although many datasets [Yang et al. ([2022](#bib.bib50)); Chao ([2021](#bib.bib6)); Taheri et al. ([2020](#bib.bib35))] focusing on human-object interaction have emerged recently, the discrepancy between the representations of robot hand and human hand make these datasets difficult to be directly used for dexterous hand manipulation.

### 2.2 Dexterous Grasping Generation

Traditional dexterous grasping [Andrews and Kry ([2013](#bib.bib2)); Bai and Liu ([2014](#bib.bib4)); Dogar and Srinivasa ([2010](#bib.bib10))] usually leverage analytical methods to model the kinematics and dynamics of both hands and objects, performing limited for high-DoF robot hands and complex objects.
Current mainstream method for dexterous grasping is utilizing reinforcement learning (RL) for training dexterous hand grasping behaviors. Many algorithms [Andrychowicz et al. ([2020](#bib.bib3)); Christen et al. ([2022](#bib.bib8)); Huang et al. ([2021](#bib.bib18)); She et al. ([2022](#bib.bib32)); Wu et al. ([2023](#bib.bib47))] in this sphere demand exact object geometric and pose data, markedly curtailing their practical applicability. To better apply for real-world scenarios, some methods [Chen et al. ([2022](#bib.bib7)); Qin et al. ([2022](#bib.bib29)); Wan et al. ([2023](#bib.bib41))] have begun leveraging raw visual data, like images or point clouds, to train dexterous hand grasping motions. Yet, the reward functions in RL are artificially designed and fail to model human motion patterns, making RL-based method challenging to achieve human-like grasping motions. Because our dataset provides accurate ground truth of human-like dexterous grasping motions, inspired by human motion generation methods [Taheri et al. ([2022](#bib.bib36)); Wu et al. ([2022](#bib.bib46))], it is more appropriate to utilize supervised methods to guide the learning process.

## 3 RealDex Dataset

To advance research and practical applications of human-like robotic dexterous grasping in real-world environments, we introduce RealDex, the first extensive dataset that captures real dexterous hand grasping motions incorporating human behavioral patterns. RealDex features a diverse collection of 52 objects varying in scale, shape, and material, along with 2.6k sequences of grasping motions involving different initial object positions and grasping directions, and approximately 955k frames of raw visual data, including images and point clouds. We use the commercial 3D scanner, the EinScan-Pro+ [3D ([2018](#bib.bib1))], to reconstruct detailed 3D mesh models for collected objects. We have divided our dataset into training, validation, and test sets, ensuring that each object appears exclusively in one of these subsets. The three sets contain 2114 grasping motion sequences for 40 objects, 245 grasping motions for 6 objects, and 271 grasping motions for 6 objects, respectively. Detailed information about the division will be provided in the supplementary materials.

We will delve into the specifics of RealDex, covering aspects such as the hardware setup of the capture system in Section [3.1](#S3.SS1), the calibration and synchronization of all sensors in Section [3.2](#S3.SS2), and the characteristics of RealDex in Section [3.3](#S3.SS3).

### 3.1 Hardware System Setup

To fulfill the requirements of high-quality data collection and effective real-world application deployment, we have established a sophisticated dexterous system comprising three key components: a vision capture system designed to acquire multi-view and multimodal visual data, a manipulation system equipped with a robotic arm and a five-fingered robtic hand, and a teleoperation system that facilitates the control of the dexterous hand to emulate human-like movements. Them communicate via Robot Operation System (ROS) network.

#### Vision Capture System.

Much like human eyes, vision sensors are crucial for robots to perceive and understand their surroundings, thereby enabling appropriate action. Image-based and point cloud-based deep learning methods [Qi et al. ([2017a](#bib.bib27), [b](#bib.bib28)); Xu et al. ([2023b](#bib.bib49)); Zhu et al. ([2021](#bib.bib53)); Wang et al. ([2021](#bib.bib42))] have achieved significant breakthroughs in scene understanding, propelling the advancement of robotic perception systems. To support the practical deployment of dexterous hands, we have established a vision system equipped with four Azure Kinect RGB-D cameras with 15Hz capture frequency. These cameras are strategically positioned at four different locations, offering a quartet of perspectives focused on the operating desk, as illustrated in Figure [2](#S3.F2). This setup provides RGB images and point clouds from multiple angles, not only aiding in delivering precise object pose annotations for our datasets, but also enabling research and practical applications of vision-based dexterous grasping under various visual configurations, including monocular-based, multi-view-based, point-based, multimodal-based approaches, etc.

#### Dexterous Manipulation System.

The dexterous manipulation system is designed to execute grasping operations akin to human actions. It features a UR10e robotic arm with 6 DoF and a tendon-driven Shadow Right Hand robot [Sha ([2005](#bib.bib31))]. This robot hand is equipped with 20 motors, enabling adduction and abduction movements across 24 degrees of freedom. Additionally, the dexterous hand is outfitted with over 100 sensors operating at up to 1 kHz.

#### Teleoperation System.

For the teleoperation system [Tel ([2022](#bib.bib37))], users wear a lightweight glove to intuitively control the Shadow Hand and robotic arm, allowing them to mimic natural human movements in real time. Human hand poses are seamlessly transmitted to the robot hand through the glove, vision tracking devices, and WiFi services. This system is performed by well-trained individuals to collect data on human-like dexterous grasping motions.

![Figure](extracted/6052044/img/vision_system.png)

*Figure 2: Overview of our comprehensive dexterous system. The bottom row shows samples of synchronized hand and robot poses.*

*Table 1: Comparison with existing robotic dexterous grasping datasets. Sim represents simulated data of grasp pose. # denotes the number of the corresponding attribute. - means no related data in the dataset. Seq. denotes sequences. Multiple Types* includes EZGripper, Barrett, Robotiq-3F, Allegro, and ShadowHand.*

### 3.2 Calibration and Synchronization

The integral dexterous system, with diverse sensors and communication networks, is inherently complex. Precise calibration and synchronization are vital to ensure seamless coordination among all subsystems for executing fine-grained operations and capturing accurate data. We introduce the most important camera-camera and robot-camera calibration and synchronization in this section.

#### Camera-camera Calibration.

Camera intrinsics comes from manufacturer, while camera extrinsics are carefully measured through a two-stage process. Firstly, we use AprilTag [Krogius et al. ([2019](#bib.bib21))] to roughly calculate extrinsics for cameras. Then, we apply the Iterative Closet Point method (ICP) [Besl and McKay ([1992](#bib.bib5)); stytim ([2023](#bib.bib34))] between two neighbouring cameras to get more accurate alignment among point clouds from different views.

#### Camera-camera Synchronization.

The Kinect depth camera measures depth by timing the travel of emitted infrared light to objects and back. Interference from simultaneous infrared emissions by multiple cameras leads to depth inaccuracies, which we address by staggering camera sampling times and synchronizing auxiliary camera data streams with the main camera using aligned timestamps.

#### Robot-camera Calibration.

To synchronize the robotic hand’s motion with the vision system, we use hand-eye calibration [IFL-CAMP ([2021](#bib.bib19))]. This process starts with the Shadow Hand holding a calibration board, followed by moving it to over 30 locations for initial static transformation. We then refine this alignment by applying ICP method [Besl and McKay ([1992](#bib.bib5))] between the robot’s mesh vertices and the main camera’s point cloud. This thorough calibration, as shown in Figure [3](#S3.F3), ensures accurate alignment between the robotic and vision systems, enhancing data precision.

![Figure](extracted/6052044/img/hand_mesh_point_3.jpg)

*Figure 3: Visualization for well-aligned dexterous hand’s mesh (white), object’s mesh (blue), and the colored point cloud of the scene, demonstrating that the calibration and synchronization between robot and camera is accurate and the 6D pose estimation for object is also precise.*

#### Robot-camera Synchronization.

The robot’s timestamp, derived from the ROS TF tree, continuously updates the pose of each dexterous hand joint. To ensure precise time alignment, we match the initial camera frame’s point cloud with the robot’s mesh, reconstructed for each timestamp within a proximate time window. We then identify the robot frame most similar to the point cloud for synchronization. With this initial aligned timestamp established, we can extrapolate and align the remaining timestamps, ensuring a consistent and accurate correlation across the entire data set.

### 3.3 Characteristics

Robotic grasping has been extensively studied for decades, serving as an essential skill for agents to engage with their surroundings and conduct manipulation tasks. Some datasets focusing on dexterous grasping, including DVGG [Wei et al. ([2022](#bib.bib44))], MultiDex [Li et al. ([2023](#bib.bib22))], DexGraspNet [Wang et al. ([2023](#bib.bib43))], and DDGdata [Liu et al. ([2020](#bib.bib23))], have already been proposed. However, our dataset, RealDex, stands apart from these existing datasets in terms of its capture process, data types, annotations, and task orientation. We draw comparisons in Table [1](#S3.T1) and highlight three key novel features of RealDex in the following discussion.

#### Human-like Grasping Motion.

In the data capture process of our dataset, the dexterous hand is operated via a teleoperation system directly controlled by a human hand. This unique approach ensures that all grasping motions in RealDex authentically mirror human behavioral patterns. In contrast, previous datasets only have static grasp poses derived from physical optimization, lacking dynamic motion sequences and any real human behavioral insights. The inclusion of realistic, human-like motion data in RealDex is profoundly significant for advancing intelligent dexterous hand development, imbued with human habits. This dataset holds the potential to propel the evolution of humanoid robotics.

#### Real Robotic Dexterous Poses.

Using a real dexterous robotic hand for data collection, our RealDex dataset accurately captures the ground truth of dexterous poses in every frame. This allows for the direct and effective application of models trained on RealDex in real-world scenarios. In contrast, previous datasets with synthesized grasp poses exhibit a substantial disparity from real-world data, leading to performance gap in translating models trained on such data into practical applications.

#### Rich Real Visual Data.

We have set up a vision capture system to obtain multi-view and multimodal visual data. This wealth of information is dexterous grasping for advancing research in vision-based dexterous grasping methods, which are typically more applicable and effective in real-world scenarios. The lack of genuine visual data in previous datasets significantly limits their practical utility.

## 4 Methodology

![Figure](extracted/6052044/img/network.png)

*Figure 4: The architecture of our grasping motion generation framework. When observing the point cloud of the object, a cVAE-based generation module is used to generate multiple grasping pose candidates. Then, a MLLM selection module is utilized to select the most reasonable and human-like pose. Finally, based on the goal grasp pose, the MotionNet synthesizes the motion sequence for robot execution.*

A dexterous hand is designed to execute motions that closely resemble those performed by human hands and interactions with objects, necessitating the use of flexible strategies in different external conditions, similar to how a human would do.
For example, the way in which humans hold a cup can vary depending on several factors, including whether there is a handle, the contents of the cup, and the temperature of the liquid inside.
In this section, we propose a unified framework for generating complete dexterous hand motions to grasp objects, closely aligning with the human experience in various environmental settings, as depicted in Figure [4](#S4.F4).
Our framework primarily consists of two parts: grasping pose generation in Section [4.2](#S4.SS2) and pose-guided motion synthesis in Section [4.3](#S4.SS3). The former involves generating grasping poses for the input objects by using a conditional Variational Autoencoder (cVAE) to generate candidate poses and aligning them with human preferences through Multi-modal Large Language Models (MLLMs). The latter is responsible for predicting complete hand motion sequences for each pose through auto-regressive motion trajectory prediction.

### 4.1 Problem Formulation

#### Dexterous Hand Representation.

A dexterous hand is driven by revolute joints that only allow rotation around a fixed single axis. Hence, the pose of a dexterous hand is represented by the joint angle configuration, denoted as $\theta\in\mathbb{R}^{22}$. The global translation and orientation of the hand are represented as $\eta\in\mathbb{R}^{6}$, where the orientation is represented in angle-axis form. The pose of a robotic hand $\phi=(\theta,\eta)\in\mathbb{R}^{28}$ is the combination of joint angle configuration and global 6D pose of hand. The mesh representation of a dexterous hand can be generated by employing forward kinematics to calculate the global transformation for each segment. To represent the shape of hand mesh, a set of points is sampled, which we denote as $\mathbf{P}^{h}$.

#### Problem Formulation.

Given an observed object point cloud $\mathbf{P}^{o}$, we aim to generate the preferred grasping poses $\{\phi^{k}\}_{k=1}^{K}$ and synthesize the corresponding motion sequence $\Phi=\{\phi_{t}^{k}\}_{t=1}^{T}$ of dexterous hand to approach and grasp object for each grasping pose $\phi^{k}$.

### 4.2 Grasping Pose Generation

#### Pose Candidate Generation.

Inspired by the work in human grasping generation [Hasson et al. ([2019b](#bib.bib17)); Jiang et al. ([2021](#bib.bib20))], we use a contact-aware generative model for dexterous grasp generation. This model uses a module to predict the contact map and a cVAE to learn a latent embedding grasping space. By sampling from the Gaussian variational distribution, the decoder of the cVAE is capable of generating a diverse array of candidate grasping poses $\{\phi^{k}\}_{k=1}^{N}$ for any input object point cloud. Subsequently, these poses undergo optimization to align with the predicted contact map, ensuring they are feasible. We then filter out unstable grasps that have insufficient contact with the object, and a human-like selection module is utilized to determine the optimal target poses $\{\phi^{k}\}_{k=1}^{K}$ for execution, where $K

#### Human-like Pose Selection.

Despite the acquisition of a diverse range of candidate grasping poses through consideration of the interaction between the object and the hand, this set of candidate poses fails to account for human grasping preferences and priors. Since MLLMs [Google ([2023](#bib.bib15)); OpenAI ([2023](#bib.bib26))] have already demonstrated their capability to encode rich world knowledge, we propose using MLLMs to discern the most natural, physically plausible, and human-like grasp poses from the candidate poses, thereby aligning it with human experience.
Specifically, we begin by acquiring mesh representations of both the robotic hand and the object for each candidate grasping pose, followed by rendering the image depicting their interaction. Next, we prompt Gemini [Google ([2023](#bib.bib15))] using the specific prompt shown in Figure [4](#S4.F4). The prompt emphasizes a comprehensive understanding of the grasping posture, highlighting the nuanced interactions between the fingers and the object. Gemini scores the rendered image, which contains intricate hand-object interaction details, in terms of naturalness, physical plausibility, human-likeness, and preference in order to select the target poses $\{\phi^{k}\}_{k=1}^{K}$.

### 4.3 Pose-guided Hand Motion Synthesis

The synthesis of intermediate motion for the human hand in an auto-regressive manner, given a starting pose $\phi_{0}$ and a target pose $\phi_{\text{tgt}}\in\{\phi^{k}\}_{k=1}^{K}$, has been studied in previous works [Zhang et al. ([2021](#bib.bib51)); Taheri et al. ([2022](#bib.bib36))].
Like them, we also employ an autoregressive approach, namely MotionNet, to learn how to predict future movements based on past trajectories and the current state to reach the target object. However, unlike parameterized human hand models such as MANO [Romero et al. ([2022](#bib.bib30))], which estimate joint positions via skinning, a robotic hand’s joints link its components with fixed articulatory relationships.
By strengthening the modeling of the interdependence among these joints, we can improve dexterous grasping. Therefore, we encode the joints’ coordinates and utilize self-attention to model their spatial relationships.

Specifically, for each time frame $t$, let $J_{t}^{\text{PE}}$ denote the sinusoidal positional encoding to the joints’ coordinates. The encoded joint positions are given by

$$\mathcal{F}^{J}_{t}=[\text{Attn}(Q,K,V)]_{t},$$ \tag{1}

where the query $Q$ and the key-value pair $(K,V)$ are all positional code $J_{t}^{\text{PE}}$, and Attn means the self-attention mechanism [Vaswani et al. ([2017](#bib.bib40))].

Next, we utilize the hand pose information from the previous five timesteps in conjunction with the current timestep’s hand and object point cloud features to forecast the hand poses for the subsequent ten timesteps following [Taheri et al. ([2022](#bib.bib36))].
At timestep $t$, the static state of a dexterous hand is represented by hand poses $\phi_{t}$, sampled points on the hand’s mesh $\mathbf{P}_{t}^{h}$, and the joint feature $\mathcal{F}^{J}_{t}$. Joint features and hand poses from the preceding five timesteps, denoted as $\mathcal{F}^{J}_{t-5:t}$ and $\phi_{t-5:t}$ respectively, collectively characterize the motion trajectory of the hand.
The velocity of hand points at time $t$, represented by $\dot{\mathbf{P}}_{t}^{h}$ is instrumental since the velocity is directly correlated with the motion in subsequent frames. Let $\mathcal{F}^{h}_{\text{tgt}}$ represent the global feature of the target hand points. It is a guidance signal to encourage motion progress towards the predefined target. However, this guidance, when provided in isolation, is incomplete as it does not account for the current state of motion. To address this, we compute the displacement of the hand points from the current frame to the target frame, denoted by $\mathbf{d}^{h}_{t}$, which serves to quantify the spatial difference and provide more contextually relevant guidance toward the target. In the end, the input of MotionNet is given by:

$$\mathcal{M}_{\text{in}}=(\mathcal{F}^{J}_{t-5:t},\phi_{t-5:t},\mathbf{P}^{h}_{% t},\dot{\mathbf{P}}^{h}_{t},\mathcal{F}^{h}_{\text{tgt}},\mathbf{d}^{h}_{t}).$$ \tag{2}

The MotionNet integrates a gating mechanism to encode the motion phase and utilizes MLP layers to predict the change of poses [Taheri et al. ([2022](#bib.bib36)); Starke et al. ([2019](#bib.bib33))]. The change can capture the temporal dynamic of movement. Representing the changes of parameters relative to current frame $t$ as $\Delta$, the output MotionNet produce is:

$$\mathcal{M}_{\text{out}}=\Delta\phi_{t:t+10},$$ \tag{3}

where $\mathcal{M}_{\text{out}}$ is used to reconstruct future motion and compute the input in the next timestep.

## 5 Experiments

We developed and executed our algorithm using Python with PyTorch framework. Models were both trained and tested on an Ubuntu server, equipped with eight NVIDIA GeForce RTX 3090 GPU cards.
We leveraged Gemini [Google ([2023](#bib.bib15))] for our MLLM selection module. More details for the training process, inference process, loss functions are introduced in the supplementary material. In the following, we first show the comparision results for vision-based grasping methods, and then evaluate each stage of our method in detail. Especially, test result on real robot is also demonstrated.

*Table 2: Comparison for grasping motion generation on human grasping dataset GRAB and robotic grasping dataset RealDex.*

### 5.1 Comparison for Dexterous Grasping

With the same point cloud data as input, we generate grasping motion sequences with different methods for the object in the test set and visualize these motions in videos. We carry out a user study to evaluate these motions based on their humanoid characteristics, grasping stability, and hand-object interaction quality, etc. The user score for each method is derived by averaging the ratings across all samples, where rank 1 gets 3 points, rank 2 gets 2 points, and rank 3 gets 1 point. A higher score indicates better performance. We have 40 users involved.
Due to a prior shortage of real-world robotic hand grasping data, no supervised models exist for fair comparison. Therefore, we adapt methods, SAGA [Wu et al. ([2021](#bib.bib45))] and GOAL [Taheri et al. ([2022](#bib.bib36))], designed for human grasp generation to produce motions for robotic dexterous hands. In addition, to verify the generalization capability of our method, we also conduct evaluation on human hand grasping dataset GRAB [Taheri et al. ([2020](#bib.bib35))]. Results in Table [2](#S5.T2) shows that our method outperforms others obviously.

### 5.2 Evaluation for Grasping Pose Generation

*Table 3: Comparison for grasp pose generation on three datasets. D.G.N is the abbreviation of DexGraspNet.*

The self-intersection of robot hand and the intersection between hand and object are physically implausible, which are measured by the intersection volume (s.i.vol.) and the penetration distance (p.dist.). To measure the stability of grasping, we compute the simulation displacement (sim.disp.) following [Tzionas et al. ([2015](#bib.bib39)); Hasson et al. ([2019b](#bib.bib17))], which is defined as the average displacement of the object when the hand is stationary and the object is subjected to gravity. These metrics provide a quantitative basis for determining the realism and viability of the grasp. We generate 10 grasp poses for each object in the test set for the evaluation. In addition, we also use the user score by rendering images for randomly selected 3 grasp poses of each object for more comprehensive evaluation.

We compare our pose generation module with UniDexGrasp [Xu et al. ([2023a](#bib.bib48))], the latest work for dexterous grasping. UniDexGrasp uses conditional normalizing flows to learn the distribution of plausible grasp poses. We use its released code for experiments. We evaluate on GRAB [Taheri et al. ([2020](#bib.bib35))], DexGraspNet [Wang et al. ([2023](#bib.bib43))] and our dataset RealDex.
DexGraspNet is a synthetic dataset including various objects and grasp poses for ShadowHand. As shown in Table [3](#S5.T3), grasp poses produced by our method demonstrate the highest stability (sim.disp.) and achieves the highest user acceptance across all datasets. The ablation result (ours w/o MLLM) demonstrates the effectiveness of our MLLM selection module in taking advantage of more general knowledge in MLLM. Qualitative result is shown in Figure [5](#S5.F5), where our result is more approaching human-like grasp poses, more natural, and more stable.

### 5.3 Evaluation for Motion Synthesis

To purely evaluate the motion sequence quality, we input the GT goal pose and sample the same number of steps with the GT motion for evaluation. The results are shown in Table [4](#S5.T4). Mean Per-Joint Positional Error (MPJPE) and Average Variance Error (AVE) are calculated, quantifying the mean and variance errors in predicted joint positions compared to ground truth across all frames [Ghosh et al. ([2021](#bib.bib13))]. We also use hand mesh vertex offset and minimum distance to the object to measure the difference between final pose and the ground truth. Even with the same goal pose as the input, the motion synthesis stage in our method is still superior due to the consideration of the spatial relationship of joints.

*Table 4: Comparison for grasping motion generation on human grasping dataset GRAB and robotic grasping dataset RealDex.*

![Figure](x1.png)

*Figure 5: Visualization for grasp pose results on RealDex.*

### 5.4 Test on Real Robot

Because RealDex is collected by real robotic dexterous hand with accurate ground truth, our model trained on it can be directly tested on real robot, enabling fast application deployment. For motion sequences generated by our model, we send them to the simulator for safety testing before the robot execution.
The qualified motion sequence is then encoded into a stamped trajectory and seamlessly integrated into the real Shadow Hand, as shown in Figure [6](#S5.F6). The execution process from the initial hand pose to holding the object lasts about 20s. The generated human-like grasp and accompanying approaching motion exhibit a natural appearance. Throughout the entire process, the hand and the object maintain a relatively static and stable relationship, aligning with the essential requirements for successful grasping. More videos depicting entire robot grasping process are in the appendix.

![Figure](extracted/6052044/img/application.png)

*Figure 6: An example sequence of real-world grasping trajectory.*

## 6 Conclusion

In this work, we present RealDex, a groundbreaking dataset featuring genuine dexterous hand grasping motions, embedded with human behaviors and rich multi-view, multimodal visual data. Our dataset is instrumental for training dexterous hands in human-like movements, significantly enhancing humanoid robotics in real-world perception, cognition, and manipulation. We also introduce a novel dexterous grasping motion generation method using MLLM, enhancing model adaptability and real-world applicability. Extensive experiments and real robot testing show remarkable performance and practical value of our method and our dataset.

## Acknowledgements

This work was supported by NSFC (No.62206173), MoE Key Laboratory of Intelligent Perception and Human-Machine Collaboration (ShanghaiTech University), Shanghai Frontiers Science Center of Human-centered Artificial Intelligence (ShangHAI), Shanghai Engineering Research Center of Intelligent Vision and Imaging.

## Contribution Statement

* Yumeng Liu, Yaxun Yang and Youzhuo Wang contributed equally to this work.
$\dagger$ Yuexin Ma supervised the project.

## References

-
3D [2018]

Shining 3D.

Einscan-pro+.

[https://www.einscan.com/handheld-3d-scanner/einscan-pro-plus/](https://www.einscan.com/handheld-3d-scanner/einscan-pro-plus/), 2018.

-
Andrews and Kry [2013]

Sheldon Andrews and Paul G Kry.

Goal directed multi-finger manipulation: Control policies and analysis.

Computers & Graphics, 37(7):830–839, 2013.

-
Andrychowicz et al. [2020]

OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al.

Learning dexterous in-hand manipulation.

The International Journal of Robotics Research, 39(1):3–20, 2020.

-
Bai and Liu [2014]

Yunfei Bai and C Karen Liu.

Dexterous manipulation using both palm and fingers.

In ICRA, pages 1560–1565. IEEE, 2014.

-
Besl and McKay [1992]

Paul J. Besl and Neil D. McKay.

A method for registration of 3-d shapes.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2):239–256, 1992.

-
Chao [2021]

etc. Chao, Yu-Wei.

Dexycb: A benchmark for capturing hand grasping of objects.

In CVPR, pages 9044–9053, 2021.

-
Chen et al. [2022]

Zoey Qiuyu Chen, Karl Van Wyk, Yu-Wei Chao, Wei Yang, Arsalan Mousavian, Abhishek Gupta, and Dieter Fox.

Learning robust real-world dexterous grasping policies via implicit shape augmentation.

arXiv preprint arXiv:2210.13638, 2022.

-
Christen et al. [2022]

Sammy Christen, Muhammed Kocabas, Emre Aksan, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions.

In CVPR, pages 20577–20586, 2022.

-
Dai et al. [2023]

Qiyu Dai, Yan Zhu, Yiran Geng, Ciyu Ruan, Jiazhao Zhang, and He Wang.

Graspnerf: Multiview-based 6-dof grasp detection for transparent and specular objects using generalizable nerf.

In ICRA, pages 1757–1763. IEEE, 2023.

-
Dogar and Srinivasa [2010]

Mehmet R Dogar and Siddhartha S Srinivasa.

Push-grasping with dexterous hands: Mechanics and a method.

In IROS, pages 2123–2130. IEEE, 2010.

-
Fang et al. [2020]

Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu.

Graspnet-1billion: A large-scale benchmark for general object grasping.

In CVPR, pages 11444–11453, 2020.

-
Fang et al. [2022]

Hongjie Fang, Hao-Shu Fang, Sheng Xu, and Cewu Lu.

Transcg: A large-scale real-world dataset for transparent object depth completion and a grasping baseline.

IEEE Robotics and Automation Letters, 7(3):7383–7390, 2022.

-
Ghosh et al. [2021]

Anindita Ghosh, Noshaba Cheema, Cennet Oguz, Christian Theobalt, and Philipp Slusallek.

Synthesis of compositional animations from textual descriptions.

In ICCV, pages 1396–1406, 2021.

-
Goldfeder et al. [2009]

Corey Goldfeder, Matei Ciocarlie, Hao Dang, and Peter K Allen.

The columbia grasp database.

In 2009 IEEE international conference on robotics and automation, pages 1710–1716. IEEE, 2009.

-
Google [2023]

Google.

Gemini.

[https://deepmind.google/technologies/gemini/#introduction](https://deepmind.google/technologies/gemini/#introduction), 2023.

-
Hasson et al. [2019a]

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid.

Learning joint reconstruction of hands and manipulated objects.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11807–11816, 2019.

-
Hasson et al. [2019b]

Yana Hasson, Gül Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid.

Learning joint reconstruction of hands and manipulated objects, April 2019.

arXiv:1904.05767 [cs].

-
Huang et al. [2021]

Wenlong Huang, Igor Mordatch, Pieter Abbeel, and Deepak Pathak.

Generalization in dexterous manipulation via geometry-aware multi-task learning.

arXiv preprint arXiv:2111.03062, 2021.

-
IFL-CAMP [2021]

IFL-CAMP.

easy_handeye: Automated, hardware-independent hand-eye calibration.

[https://github.com/IFL-CAMP/easy_handeye](https://github.com/IFL-CAMP/easy_handeye), 2021.

Accessed: 2024-01-09.

-
Jiang et al. [2021]

Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang.

Hand-Object Contact Consistency Reasoning for Human Grasps Generation, April 2021.

arXiv:2104.03304 [cs].

-
Krogius et al. [2019]

Maximilian Krogius, Acshi Haggenmiller, and Edwin Olson.

Flexible layouts for fiducial tags.

In IROS, October 2019.

-
Li et al. [2023]

Puhao Li, Tengyu Liu, Yuyang Li, Yiran Geng, Yixin Zhu, Yaodong Yang, and Siyuan Huang.

GenDexGrasp: Generalizable Dexterous Grasping, March 2023.

arXiv:2210.00722 [cs].

-
Liu et al. [2020]

Min Liu, Zherong Pan, Kai Xu, Kanishka Ganguly, and Dinesh Manocha.

Deep differentiable grasp planner for high-dof grippers.

arXiv preprint arXiv:2002.01530, 2020.

-
Lundell et al. [2021]

Jens Lundell, Francesco Verdoja, and Ville Kyrki.

Ddgc: Generative deep dexterous grasping in clutter.

IEEE Robotics and Automation Letters, 6(4):6899–6906, 2021.

-
Miller and Allen [2004]

Andrew T Miller and Peter K Allen.

Graspit! a versatile simulator for robotic grasping.

IEEE Robotics & Automation Magazine, 11(4):110–122, 2004.

-
OpenAI [2023]

OpenAI.

Gpt-4.

[https://openai.com/research/gpt-4](https://openai.com/research/gpt-4), 2023.

-
Qi et al. [2017a]

Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.

Pointnet: Deep learning on point sets for 3d classification and segmentation.

In CVPR, pages 652–660, 2017.

-
Qi et al. [2017b]

Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas.

Pointnet++: Deep hierarchical feature learning on point sets in a metric space.

arXiv preprint arXiv:1706.02413, 2017.

-
Qin et al. [2022]

Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Hao Su, and Xiaolong Wang.

Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation.

In Deep Reinforcement Learning Workshop NeurIPS 2022, 2022.

-
Romero et al. [2022]

Javier Romero, Dimitrios Tzionas, and Michael J Black.

Embodied hands: Modeling and capturing hands and bodies together.

arXiv preprint arXiv:2201.02610, 2022.

-
Sha [2005]

Shadowrobot.

[https://www.shadowrobot.com/dexterous-hand-series/](https://www.shadowrobot.com/dexterous-hand-series/), 2005.

-
She et al. [2022]

Qijin She, Ruizhen Hu, Juzhan Xu, Min Liu, Kai Xu, and Hui Huang.

Learning high-dof reaching-and-grasping via dynamic representation of gripper-object interaction.

arXiv preprint arXiv:2204.13998, 2022.

-
Starke et al. [2019]

Sebastian Starke, He Zhang, Taku Komura, and Jun Saito.

Neural state machine for character-scene interactions.

ACM Trans. Graph., 38(6):209–1, 2019.

-
stytim [2023]

stytim.

k4a-calibration: Azure Kinect multi-camera extrinsic calibration.

[https://github.com/stytim/k4a-calibration](https://github.com/stytim/k4a-calibration), 2023.

Accessed: 2024-01-06.

-
Taheri et al. [2020]

Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas.

Grab: A dataset of whole-body human grasping of objects.

In European Conference on Computer Vision, pages 581–600. Springer, 2020.

-
Taheri et al. [2022]

Omid Taheri, Vasileios Choutas, Michael J Black, and Dimitrios Tzionas.

Goal: Generating 4d whole-body motion for hand-object grasping.

In CVPR, pages 13263–13273, 2022.

-
Tel [2022]

Teleoperation system of shadowrobot.

[https://www.shadowrobot.com/teleoperation/](https://www.shadowrobot.com/teleoperation/), 2022.

-
Todorov et al. [2012]

Emanuel Todorov, Tom Erez, and Yuval Tassa.

Mujoco: A physics engine for model-based control.

In IROS, pages 5026–5033. IEEE, 2012.

-
Tzionas et al. [2015]

Dimitrios Tzionas, Luca Ballan, Abhilash Srikantha, Pablo Aponte, Marc Pollefeys, and Juergen Gall.

Capturing hands in action using discriminative salient points and physics simulation.

CoRR, abs/1506.02178, 2015.

-
Vaswani et al. [2017]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

NeurIPS, 30, 2017.

-
Wan et al. [2023]

Weikang Wan, Haoran Geng, Yun Liu, Zikang Shan, Yaodong Yang, Li Yi, and He Wang.

Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

arXiv preprint arXiv:2304.00464, 2023.

-
Wang et al. [2021]

Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin.

Fcos3d: Fully convolutional one-stage monocular 3d object detection.

In ICCV, pages 913–922, 2021.

-
Wang et al. [2023]

Ruicheng Wang, Jialiang Zhang, Jiayi Chen, Yinzhen Xu, Puhao Li, Tengyu Liu, and He Wang.

DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation, March 2023.

arXiv:2210.02697 [cs].

-
Wei et al. [2022]

Wei Wei, Daheng Li, Peng Wang, Yiming Li, Wanyi Li, Yongkang Luo, and Jun Zhong.

Dvgg: Deep variational grasp generation for dextrous manipulation.

IEEE Robotics and Automation Letters, 7(2):1659–1666, 2022.

-
Wu et al. [2021]

Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang.

Saga: Stochastic whole-body grasping with contact.

arXiv preprint arXiv:2112.10103, 2021.

-
Wu et al. [2022]

Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang.

Saga: Stochastic whole-body grasping with contact.

In ECCV, pages 257–274. Springer, 2022.

-
Wu et al. [2023]

Yueh-Hua Wu, Jiashun Wang, and Xiaolong Wang.

Learning generalizable dexterous manipulation from human grasp affordance.

In Conference on Robot Learning, pages 618–629. PMLR, 2023.

-
Xu et al. [2023a]

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, Tengyu Liu, Li Yi, and He Wang.

UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy, March 2023.

arXiv:2303.00938 [cs].

-
Xu et al. [2023b]

Yiteng Xu, Peishan Cong, and etc Yao.

Human-centric scene understanding for 3d large-scale scenarios.

In ICCV, pages 20349–20359, 2023.

-
Yang et al. [2022]

Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu.

Oakink: A large-scale knowledge repository for understanding hand-object interaction.

In CVPR, pages 20953–20962, 2022.

-
Zhang et al. [2021]

He Zhang, Yuting Ye, Takaaki Shiratori, and Taku Komura.

Manipnet: Neural manipulation synthesis with a hand-object spatial representation.

ACM Transactions on Graphics (ToG), 40(4):1–14, 2021.

-
Zhou et al. [2018]

Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun.

Open3D: A modern library for 3D data processing.

arXiv:1801.09847, 2018.

-
Zhu et al. [2021]

Xinge Zhu, Hui Zhou, and etc Wang.

Cylindrical and asymmetrical 3d convolution networks for lidar-based perception.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6807–6822, 2021.

## Appendix A Data Process

### A.1 Point Cloud Denoising

To enhance the utility of our RealDex dataset, we have included precise mesh models and 6D poses for all featured objects. We automatically annotate the 6D poses of objects for all grasping motion sequences using the ICP method [Besl and McKay [[1992](#bib.bib5)]] and manually inspect and adjust them against the point cloud. Hardware limitations of RGBD cameras can introduce noise in the point clouds they generate, which may affect the accuracy of pose annotation. We apply a statistical outlier removal filter [Zhou et al. [[2018](#bib.bib52)]] to post-process raw point cloud data and merge point clouds from various viewpoints into a unified one. The denoising process involves analyzing each point’s average distance to its 20 nearest neighbors and excluding those points whose distance deviates by more than two standard deviations from the mean, effectively reducing noise.

### A.2 Object Pose Labeling

The 6D poses of objects are annotated mainly using the Iterative Closet Point method (ICP) [Besl and McKay [[1992](#bib.bib5)]] with human adjustment. Initially We manually determine the object pose in the first frame using the refined point cloud, setting the foundation for subsequent automated ICP adjustments. The pose for each subsequent frame is inferred from the preceding one. Finally, the resulting sequence is inspected and, if necessary, fine-tuned by a human annotator. In practice, most sequences require only a single annotation pass.

### A.3 Visualization of Dataset

Annotation Here we present a sample of the annotated results depicting the object motion and dexterous hand motion, as shown
in [Figure 7](#A1.F7).

![Figure](extracted/6052044/img/aligned_mesh_pc.png)

*Figure 7: The visualization for aligned point cloud and hand’s mesh, object’s mesh.*

Motion Sequence We present the the motion sequence of dexterous hand mesh in our RealDex dataset. We sampled 8 frames from a grasping motion and display the mesh of robotic hand with arm, as shown in [Figure 8](#A1.F8).

![Figure](extracted/6052044/img/dataset_motion.png)

*Figure 8: The visualization for grasping motion sequence in RealDex.*

## Appendix B Method

### B.1 Training

We train our framework in two stages, the training for grasp pose generation and the training for motion synthesis. Since our dataset includes precise annotations for object and hand poses along with complete dexterous hand motion, enables both stages of our training to benefit from ground truth data supervision.

Pose Generation During pose generation training, we first create the robotic hand’s mesh from the hand pose $\phi$ using forward kinematics and then generate the hand’s point cloud $\mathbf{P}^{h}$. The hand feature $\mathcal{F}^{h}$ and condition feature $\mathcal{F}^{o}$ is compressed into the latent space by cVAE encoder. Hand poses are reconstructed by the decoder using the concatenation of conditional feature and the latent code, sampled from the learned distribution. From the decoder’s output, we can then compute a binary contact map, $\mathcal{C}$ on object points that indicates whether the points are within the hand’s contact region. The loss to supervise the generated poses is the weighted sum of four losses:

$$\begin{split}&\mathcal{L}_{\text{KL}}=\dfrac{1}{2}(-\log{\sigma^{2}}-1+\sigma^% {2}+\mu^{2})\\ &\mathcal{L}_{\text{recon}}=||\phi-\phi^{\text{gt}}||_{2},\\ &\mathcal{L}_{\text{cmap}}=\text{BCE}(\mathcal{C}-\mathcal{C}^{\text{gt}}),\\ &\mathcal{L}_{\text{CD}}=\sum_{\mathbf{a}\in\mathbf{P}^{h}}\min_{\mathbf{b}\in% \mathbf{P}^{h,\text{gt}}}||\mathbf{a}-\mathbf{b}||^{2}+\sum_{\mathbf{b}\in% \mathbf{P}^{h,\text{gt}}}\min_{\mathbf{a}\in\mathbf{P}^{h}}||\mathbf{b}-% \mathbf{a}||^{2}.\end{split}$$ \tag{4}

In [Equation 4](#A2.E4), $\mathcal{L}_{\text{KL}}$ denotes the Kullback-Leibler divergence to measure the similarity between prior $\mathcal{N}(\mu,\sigma^{2})$ and standard Gaussian distribution $\mathcal{N}(0,1)$; $\mathcal{L}_{\text{recon}}$ is the MSE loss of reconstructed hand pose and ground truth hand pose; $\mathcal{L}_{\text{cmap}}$ is a binary cross entropy (BCE) to measure the difference between the contact map from reconstructed hand pose and the ground truth; and $\mathcal{L}_{\text{CD}}$ is the Chamfer distance between points sampled from reconstruction hand mesh and the points on GT hand mesh.

Motion Synthesis In the training of MotionNet, we first generate the hand points $\mathbf{P}^{h}$. Then we add noise to $\phi$ and $\mathbf{P}^{h}$ in the network input to enhance the generalization ability of network. The loss for MotionNet is the difference from predicted parameters to its GT value.

$$\begin{split}\mathcal{L}_{\text{M}}&=\omega_{\phi}||\phi-\phi^{\text{gt}}||_{1% }+\omega_{h}||\mathbf{P}^{h}-\mathbf{P}^{h,\text{gt}}||_{2}+\omega_{d}||% \mathbf{d}^{h}-\mathbf{d}^{h,\text{gt}}||_{2}\end{split}$$ \tag{5}

![Figure](extracted/6052044/img/MLLM_selection.png)

*Figure 9: The text in the first column provides the complete prompt input to Gemini. Adjacent to this, in the subsequent four columns on the right, we present the input images alongside the corresponding scores and explanations as given by the MLLM selection module, offering a transparent view of the decision-making process.*

### B.2 MLLM Selection

For each object, we sample 100 poses and generate 100 images through rendering. These images are collectively processed by Gemini, yielding a set of scores along with detailed explanations for each pose. Subsequently, we extract the top ten poses from the dataset, which are determined by the scores they received. These selected poses serve as the primary targets for our subsequent motion synthesis phase.

### B.3 Inference

At the inference stage, our pose generation module receives unseen object point clouds, which serve as the input conditions. Utilizing these conditions, cVAE decoder generates candidate grasping by randomly sampling the latent code from standard Gaussian distribution. Candidate poses are refined by test-time optimization and then get scores from LLM selection module, special requirements or conditions can be added to let the LLM select the most suitable pose as goal. Finally the MotionNet utilizes the selected poses as targets and initiates the motion synthesis process from the mean pose, indicating that all joint angles of the dexterous hand are set to zero. The output for the current time frame is then employed to determine the input data for the subsequent time frame. The termination of this process is defined by either fixed time steps or a threshold based on the distance between the current grasp and the target grasp.

![Figure](extracted/6052044/img/motion_synth_result.png)

*Figure 10: Motion synthesis result from our framework. The first row illustrates the initial and target hand poses, serving as inputs for the motion synthesis module. Subsequently, a sequence of hand motions is generated, using the target pose as a reference to guide the synthesis process.*

## Appendix C More Results and Discussions

### C.1 Pose generation

[Figure 11](#A3.F11) displays selected results from our grasping pose generation module, showcasing various automatically computed hand configurations for different object shapes.

### C.2 Motion synthesis

Given a initial pose and a target pose, our pose-guided hand motion synthesis module is capable of generating a sequence of hand motion, as shown in [Figure 10](#A2.F10), the initial pose we give is the mean pose of dexterous hand, which means that all the joint angles equal 0 in this pose. The translation of the hand is calculated from the average location across our dataset. Each one in the generated sequence represents a progressive step towards achieving the final target configuration.

### C.3 MLLM selection

In [Figure 9](#A2.F9), we show the output from our MLLM selection module, each grasp is represented by a rendered image of the hand and object mesh. These images are input into the MLLM selection module, which assigns a score to each grasp and give detailed explanation.

![Figure](extracted/6052044/img/supp_gallery.png)

*Figure 11: Visualization of the generated grasps from our grasp pose generation module. Given an object point cloud derived from RGB-D data, this module samples potential hand poses and employs MLLM to select the most plausible ones.*

## Appendix D Limitation

Our algorithm still has much room for improvement. For instance, in the result of pose generation, there is intersection between object and hand that need to be removed by optimization in test time. It could be improved by utilizing penalty loss for collision when training. In addition, when generating motion, it is guided solely by the target pose, without taking into account the actual conditions of the objects and the environment.

Generated on Sat Dec 7 04:18:46 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)