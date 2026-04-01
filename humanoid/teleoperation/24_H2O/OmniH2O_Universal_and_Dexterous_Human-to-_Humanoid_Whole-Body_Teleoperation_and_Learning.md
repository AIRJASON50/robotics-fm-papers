[# OmniH2O: Universal and Dexterous Human-to- Humanoid Whole-Body Teleoperation and Learning Tairan He${}^{{\dagger}\textnormal{1}}$, Zhengyi Luo${}^{{\dagger}\textnormal{1}}$, Xialin He${}^{{\dagger}\textnormal{2}}$, Wenli Xiao${}^{\textnormal{1}}$, Chong Zhang${}^{\textnormal{1}}$, Weinan Zhang${}^{\textnormal{2}}$, Kris Kitani${}^{\textnormal{1}}$, Changliu Liu${}^{\textnormal{1}}$, Guanya Shi${}^{\textnormal{1}}$ 1Carnegie Mellon University 2Shanghai Jiao Tong University †Equal Contributions Page: https://omni.human2humanoid.com](https://omni.human2humanoid.com)

###### Abstract

We present OmniH2O (Omni Human-to-Humanoid), a learning-based system for whole-body humanoid teleoperation and autonomy.
Using kinematic pose as a universal control interface, OmniH2O enables various ways for a human to control a full-sized humanoid with dexterous hands, including using real-time teleoperation through VR headset, verbal instruction, and RGB camera. OmniH2O also enables full autonomy by learning from teleoperated demonstrations or integrating with frontier models such as GPT-4o.
OmniH2O demonstrates versatility and dexterity in various real-world whole-body tasks through teleoperation or autonomy, such as playing multiple sports, moving and manipulating objects, and interacting with humans, as shown in LABEL:fig:firstpage.
We develop an RL-based sim-to-real pipeline, which involves large-scale retargeting and augmentation of human motion datasets, learning a real-world deployable policy with sparse sensor input by imitating a privileged teacher policy, and reward designs to enhance robustness and stability.
We release the first humanoid whole-body control dataset, *OmniH2O-6*, containing six everyday tasks, and demonstrate humanoid whole-body skill learning from teleoperated datasets.

Keywords: Humanoid Teleoperation, Humanoid Loco-Manipulation, RL

## 1 Introduction

How can we best unlock humanoid’s potential as one of the most promising physical embodiments of general intelligence?
Inspired by the recent success of pretrained vision and language models [achiam2023gpt], one potential answer is to collect large-scale human demonstration data in the real world and learn humanoid skills from it. The embodiment alignment between humanoids and humans not only makes the humanoid a potential generalist platform but also enables the seamless integration of human cognitive skills for scalable data collections [darvish2023teleoperation, he2024learning, seo2023deep, fu2024mobile].

However, *whole-body control* of a full-sized humanoid robot is challenging [moro2019whole], with many existing works focusing only on the lower body [siekmann2021sim, li2021reinforcement, duan2023learning, dao2022sim, radosavovic2024humanoid, radosavovic2024real, li2024reinforcement] or decoupled lower and upper body control [harada2005humanoid, seo2023deep, murooka2021humanoid].
To simultaneously support stable dexterous manipulation and robust locomotion, the controller must coordinate the lower and upper bodies in unison. For the humanoid *teleoperation interface* [darvish2023teleoperation], the need for expensive setups such as motion captures and exoskeletons also hinders large-scale humanoid data collection. In short, we need a robust control policy that *supports whole-body dexterous loco-manipulation*, while seamlessly integrating with *easy-to-use and accessible teleoperation interfaces* (e.g., VR) to enable scalable demonstration data collection.

In this work, we propose OmniH2O, a learning-based system for whole-body humanoid teleoperation and autonomy.
We propose a pipeline to train a robust whole-body motion imitation policy via teach-student distillation and identify key factors in obtaining a stable control policy that supports dexterous manipulation. For instance, we find these elements to be essential: *motion data distribution*, *reward designs*, and *state space design* and *history utilization*. The distribution of the motion imitation dataset needs to be biased toward standing and squatting to help the policy learn to stabilize the lower body during manipulation. Regularization rewards are used to shape the desired motion but need to be applied with a curriculum. The input history could replace the global linear velocity, an essential input in previous work [he2024learning] that requires Motion Capture (MoCap) to obtain. We also carefully design our control interface and choose the kinematic pose as an intermediate representation to bridge between human instructions and humanoid actuation. This interface makes our control framework compatible with many real-world input sources, such as VR, RGB cameras, and autonomous agents (GPT-4o). Powered by our robust control policy, we demonstrate teleoperating humanoids to perform various daily tasks (racket swinging, flower watering, brush writing, squatting and picking, boxing, basket delivery, *etc*.), as shown in LABEL:fig:firstpage. Through teleoperation, we collect a dataset of our humanoid completing six tasks such as hammer catching, basket picking, *etc*., annotated with paired first-person RGBD camera views, control input, and whole-body motor actions. Based on the dataset, we showcase training autonomous policies via imitation learning.

In conclusion, our contributions are as follows: (1) We propose a pipeline to train a robust humanoid control policy that supports whole-body dexterous loco-manipulation with a universal interface that enables versatile human control and autonomy. (2) Experiments of large-scale motion tracking in simulation and the real world validate the superior motion imitation capability of OmniH2O. (3) We contribute the first humanoid loco-manipulation dataset and evaluate imitation learning methods on it to demonstrate humanoid whole-body skill learning from teleoperated datasets.

## 2 Related Works

Learning-based Humanoid Control.
Controlling humanoid robots is a long-standing robotic problem due to their high degree-of-freedom (DoF) and lack of self-stabilization [grizzle2009mabel, hirai1998development].
Recently, learning-based methods have shown promising results [siekmann2021sim, li2021reinforcement, duan2023learning, dao2022sim, radosavovic2024humanoid, radosavovic2024real, li2024reinforcement, he2024learning, cheng2024expressive].
However, most studies [siekmann2021sim, li2021reinforcement, duan2023learning, dao2022sim, radosavovic2024humanoid, radosavovic2024real, li2024reinforcement] focus mainly on learning robust locomotion policies and do not fully unlock all the abilities of humanoids.
For tasks that require whole-body loco-manipulation, the lower body must serve as the support for versatile and precise upper body movement [fu2023deep]. Traditional goal-reaching [he2024agile, yang2023cajun] or velocity-tracking objectives [cheng2024expressive] used in legged locomotion are incompatible with such requirements because these objectives require additional task-specific lower-body goals (from other policies) to indirectly account for upper-lower-body coordination. OmniH2O instead learns an end-to-end whole-body policy to coordinate upper and lower bodies.

Humanoid Teleoperation. Teleoperating humanoids holds great potential in unlocking the full capabilities of the humanoid system. Prior efforts in humanoid teleoperation have used task-space control [seo2023deep, dafarra2024icub3], upper-body-retargeted teleoperation [chagas2021humanoid, elobaid2020telexistence] and whole-body teleoperation [montecillo2010real, otani2017adaptive, hu2014online, tachi2020telesar, ishiguro2018high, porges2019wearable, he2024learning, seo2023deep].
Recently, H2O [he2024learning] presents an RL-based whole-body teleoperation framework that uses a third-person RGB camera to obtain full-body keypoints of the human teleoperator.
However, due to the delay and inaccuracy of RGB-based pose estimation and the requirement for global linear velocity estimation, H2O [he2024learning] requires MoCap during test time, only supports simple mobility tasks, and lacks the precision for dexterous manipulation tasks.
By contrast, OmniH2O enables high-precision dexterous loco-manipulation indoors and in the wild.

Whole-body Humanoid Control Interfaces. To control a full-sized humanoid, many interfaces such as exoskeleton [ishiguro2020bilateral], MoCap [stanton2012teleoperation, dajles2018teleoperation], and VR [fritsche2015first, hirschmanner2019virtual] are proposed. Recently, VR-based humanoid control [winkler2022questsim, lee2023questenvsim, ye2022neural3points, Luo2023-er] has been drawing attention in the graphics community due to its ability to create whole-body motion using sparse input.
However, these VR-based works only focus on humanoid control for animation and do not support mobile manipulation. OmniH2O, on the other hand, can control a real humanoid robot to complete real-world manipulation tasks.

Open-sourced Robotic Dataset and Imitation Learning.
One major challenge within the robotics community is the limited number of publicly available datasets compared to those for language and vision tasks [padalkar2023open]. Recent efforts [padalkar2023open, khazatsky2024droid, walke2023bridgedata, brohan2022rt, brohan2023rt, team2024octo, driess2023palm, lin2024hato] have focused on collecting robotic data using various embodiments for different tasks. However, most of these datasets are collected with fixed-base robotic arm platforms. Even one of the most comprehensive datasets to date, Open X-Embodiment [padalkar2023open], does not include data for humanoids. To the best of our knowledge, we are the first to release a dataset for full-sized humanoid whole-body loco-manipulation.

## 3 Universal and Dexterous Human-to-Humanoid Whole-Body Control

In this section, we describe our whole-body control system to support teleoperation, dexterous manipulation, and data collection. As simulation has access to inputs that are hard to obtain from real-world devices, we opt to use a teacher-student framework. We also provide details about key elements to obtain a stable and robust control policy: dataset balance, reward designs, *etc*.

##### Problem Formulation

We formulate the learning problem as goal-conditioned RL for a Markov Decision Process (MDP) defined by the tuple $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\rangle$ of state $\mathcal{S}$, action ${\bm{a}_{t}}\in\mathcal{A}$, transition $\mathcal{T}$, reward function $\mathcal{R}$, and discount factor $\gamma$. The state ${\bm{s}_{t}}$ contains the proprioception ${\bm{s}^{\text{p}}_{t}}$ and the goal state ${\bm{s}^{\text{g}}_{t}}$. The goal state ${\bm{s}^{\text{g}}_{t}}$ includes the motion goals from the human teleoperator or autonomous agents. Based on proprioception ${\bm{s}^{\text{p}}_{t}}$, goal state ${\bm{s}^{\text{g}}_{t}}$, and action ${\bm{a}_{t}}$, we define the reward $r_{t}=\mathcal{R}\left({\bm{s}^{\text{p}}_{t}},{\bm{s}^{\text{g}}_{t}},{\bm{a}_{t}}\right)$. The action ${\bm{a}_{t}}$ specifies the target joint angles and a PD controller actuates the motors. We apply the Proximal Policy Optimization algorithm (PPO) [SchulmanWDRK17] to maximize the cumulative discounted reward $\mathbb{E}\left[\sum_{t=1}^{T}\gamma^{t-1}r_{t}\right]$. In this work, we study the motion imitation task where our policy ${\pi_{\text{OmniH2O}}}$ is trained to track real-time motion input as shown in [Figure 2](#S3.F2). This task provides a universal interface for humanoid control as the kinematic pose can be provided by many different sources. We define kinematic pose as ${\bm{{q}}_{t}}\triangleq({\bm{{\theta}}_{t}},{\bm{{p}}_{t}})$, consisting of 3D joint rotations ${\bm{{\theta}}_{t}}$ and positions ${\bm{{p}}_{t}}$ of all joints on the humanoid. To define velocities ${\bm{\dot{q}}_{1:T}}$, we have ${\bm{\dot{q}}_{t}}\triangleq({\bm{{\omega}}_{t}},{\bm{v}_{t}})$ as angular ${\bm{{\omega}}_{t}}$ and linear velocities ${\bm{v}_{t}}$. As a notation convention, we use $\widetilde{\cdot}$ to represent kinematic quantities from VR headset or pose generators, $\widehat{\cdot}$ to denote ground truth quantities from MoCap datasets, and normal symbols without accents for values from the physics simulation or real robot.

*Figure 1: (a) Source motion; (b) Retargeted motion; (c) Standing variant; (d) Squatting variant.*

Human Motion Retargeting. We train our motion imitation policy using retargeted motions from the AMASS [Mahmood2019-ki] dataset, using a similar retargeting process as H2O [he2024learning].
One major drawback of H2O is that the humanoid tends to take small adjustment steps instead of standing still.
In order to enhance the ability of stable standing and squatting, we bias our training data by adding sequences that contain fixed lower body motion. Specifically, for each motion sequence ${\bm{\hat{q}}_{1:T}}$ from our dataset, we create a “stable” version ${\bm{\hat{q}}^{\text{stable}}_{1:T}}$ by fixing the root position and the lower body to a standing or squatting position as shown in Fig. [1](#S3.F1). We provide ablation of this strategy in LABEL:appendix:ablation_motion_data.

![Figure](x2.png)

*Figure 2: (a) OmniH2O retargets large-scale human motions and filters out infeasible motions for humanoids. (b) Our sim-to-real policy is distilled through supervised learning from an RL-trained teacher policy using privileged information. (c) The universal design of OmniH2O supports versatile human control interfaces including VR headset, RGB camera, language, etc. Our system also supports to be controlled by autonomous agents like GPT-4o or imitation learning policy trained using our dataset collected via teleoperation.*

Reward and Domain Randomization. To train ${\pi_{\text{privileged}}}$ that is suitable as a teacher for a real-world deployable student policy, we employ both imitation rewards and regularization rewards.
Previous work [cheng2024expressive, he2024learning] often uses regularization rewards like feet air time* or *feet height* to shape the lower-body motions. However, these rewards result in the humanoid stomping to keep balanced instead of standing still.
To encourage standing still and taking large steps during locomotion, we propose a key reward function *max feet height for each step*. We find that this reward, when applied with a carefully designed curriculum, effectively helps RL decide when to stand or walk. We provide a detailed overview of rewards, curriculum design, and domain randomization in LABEL:sec:reward_appendix and LABEL:sec:dr_appendix.

Teacher: Privileged Imitation Policy. During real-world teleoperation of a humanoid robot, much information that is accessible in simulation (*e.g*., the global linear/angular velocity of every body link) is not available. Moreover, the input to a teleoperation system could be sparse (*e.g*., for VR-based teleoperation, only the hands and head’s poses are known), which makes the RL optimization challenging. To tackle this issue, We first train a teacher policy that uses privileged state information and then distill it to a student policy with limited state space. Having access to the privileged state can help RL find more optimal solutions, as shown in prior works [lee2020learning] and our experiments ( [Section 4](#S4)). Formally, we train a privileged motion imitator ${\pi_{\text{privileged}}}({\bm{a}_{t}}|{\bm{s}^{\text{p-privileged}}_{t}},{\bm{s}^{\text{g-privileged}}_{t}})$, as described in [Figure 2](#S3.F2). The proprioception is defined as ${\bm{s}^{\text{p-privileged}}_{t}}\triangleq[{\bm{{p}}_{t}},{\bm{{\theta}}_{t}},{\bm{\dot{q}}_{t}},{\bm{{\omega}}_{t}},{\bm{a}_{t-1}}]$, which contains the humanoid rigidbody position ${\bm{{p}}_{t}}$, orientation ${\bm{{\theta}}_{t}}$, linear velocity ${\bm{\dot{q}}_{t}}$, angular velocity ${\bm{{\omega}}_{t}}$, and the previous action ${\bm{a}_{t-1}}$. The goal state is defined as ${\bm{s}^{\text{g-privileged}}_{t}}\triangleq[{\bm{\hat{\theta}}_{t+1}}\ominus{\bm{{\theta}}_{t}},{\bm{\hat{p}}_{t+1}}-{\bm{{p}}_{t}},\bm{\hat{v}}_{t+1}-\bm{v}_{t},\bm{\hat{\omega}}_{t}-\bm{\omega}_{t},{\bm{\hat{\theta}}_{t+1}},{\bm{\hat{p}}_{t+1}}]$, which contains the reference pose (${\bm{\hat{\theta}}_{t}},{\bm{\hat{p}}_{t}}$) and one-frame difference between the reference and current state for all rigid bodies of the humanoid.

Student: Sim-to-Real Imitation Policy with History. We design our control policy to be compatible with many input sources by using the kinematic reference motion as the intermediate representation. As estimating full-body motion ${\bm{\tilde{q}}_{t}}$ (both rotation and translation) is difficult (especially from VR headsets), we opt to control our humanoid with position ${\bm{\tilde{p}}_{t}}$ only for teleoperation. Specifically, for real-world teleoperation, the goal state is ${\bm{s}^{\text{g-real}}_{t}}\triangleq({\bm{\tilde{p}}^{\text{real}}_{t}}-{\bm{{p}}_{t}^{\text{real}}},{\bm{\tilde{v}}^{\text{real}}_{t}}-{\bm{v}^{\text{real}}_{t}},{\bm{\tilde{p}}^{\text{real}}_{t}})$. The superscript ${}^{\text{real}}$ indicates using the 3-points available (head and hands) from the VR headset.
For other control interfaces (e.g., RGB, language), we use the same input 3-point input to maintain consistency, though can be easily extended to more keypoints to alleviate ambiguity.
For proprioception, the student policy ${\bm{s}^{\text{p-real}}_{t}}\triangleq({\bm{{d}}_{t-25:t}},{\bm{\dot{d}}_{t-25:t}},{\bm{\omega}^{\text{root}}_{t-25:t}},{\bm{g}_{t-25:t}},{\bm{a}_{t-25-1:t-1}})$ uses values easily accessible in the real-world, which includes 25-step history of joint (DoF) position ${\bm{{d}}_{t-25:t}}$, joint velocity ${\bm{\dot{d}}_{t-25:t}}$, root angular velocity ${\bm{\omega}^{\text{root}}_{t-25:t}}$, root gravity ${\bm{g}_{t-25:t}}$, and previous actions ${\bm{a}_{t-25-1:t-1}}$. The inclusion of history data helps improve the robustness of the policy with our teacher-student supervised learning. Note that no global linear velocity ${\bm{v}_{t}}$ information is included in our observations and the policy implicitly learns velocity using history information. This removes the need for MoCap as in H2O [he2024learning] and further enhances the feasibility of in-the-wild deployment.

Policy Distillation. We train our deployable teleoperation policy ${\pi_{\text{OmniH2O}}}$ following the DAgger [Ross2010-cc] framework: for each episode, we roll out the student policy ${\pi_{\text{OmniH2O}}}({\bm{a}_{t}}|{\bm{s}^{\text{p-real}}_{t}},{\bm{s}^{\text{g-real}}_{t}})$ in simulation to obtain trajectories of $({\bm{s}^{\text{p-real}}_{1:T}},{\bm{s}^{\text{g-real}}_{1:T}})$.
Using the reference pose ${\bm{\hat{q}}_{1:T}}$ and simulated humanoid states ${\bm{s}^{\text{p}}_{1:T}}$, we can compute the privileged states ${\bm{s}^{\text{g-privileged}}_{t}},{\bm{s}^{\text{p-privileged}}_{t}}\leftarrow({\bm{s}^{\text{p}}_{t}},{\bm{\hat{q}}_{t+1}})$. Then, using the pair $({\bm{s}^{\text{p-privileged}}_{t}},{\bm{s}^{\text{g-privileged}}_{t}})$, we query the teacher ${\pi_{\text{privileged}}}({\bm{a}_{t}}^{\text{privileged}}|{\bm{s}^{\text{p-privileged}}_{t}},{\bm{s}^{\text{g-privileged}}_{t}})$ to calculate the reference action ${\bm{a}_{t}}^{\text{privileged}}$. To update ${\pi_{\text{OmniH2O}}}$, the loss is: $\mathcal{L}=\|{\bm{a}_{t}}^{\text{privileged}}-{\bm{a}_{t}}\|^{2}_{2}$.

Dexterous Hands Control. As shown in [Figure 2](#S3.F2)(c), we use the hand poses estimated by VR [cheng2022open, park2024avp], and directly compute joint targets based on inverse kinematics for an off-the-shelf low-level hand controller. We use VR for the dexterous hand control in this work, but the hand pose estimation could be replaced by other interfaces (e.g., MoCap gloves [shaw2023leap] or RGB cameras [handa2020dexpilot]) as well.

## 4 Experimental Results

In our experiments, we aim to answer the following questions. Q1. ([Section 4.1](#S4.SS1)) Can OmniH2O accurately track motion in simulation and real world? Q2. ([Section 4.2](#S4.SS2)) Does OmniH2O support versatile control interfaces in the real world and unlock new capabilities of loco-manipulation? Q3. ([Section 4.3](#S4.SS3)) Can we use OmniH2O to collect data and learn autonomous agents from teleoperated demonstrations? As motion is best seen in videos, we provide visual evaluations in our [website](https://anonymous-omni-h2o.github.io/).

### 4.1 Whole-body Motion Tracking

Experiment Setup. To answer Q1, we evaluate OmniH2O on motion tracking in simulation ([Section 4.1.1](#S4.SS1.SSS1)) and the real world ([Section 4.1.1](#S4.SS1.SSS1)). In simulation, we evaluate on the retargeted AMASS dataset with augmented motions $\bm{\hat{Q}}$ (14k sequences); in real-world, we test on 20 standing sequences due to the limited physical lab space and the difficulty of evaluating on large-scale datasets in the real world. Detailed state-space composition (LABEL:sec:state_appendix), ablation setup (LABEL:sec:sim_ablations), hyperparameters ([Appendix K](#A11)), and hardware configuration (LABEL:sec:setup_appendix) are summarized in the Appendix.

Metrics. We evaluate the motion tracking performance using both pose and physics-based metrics. We report Success rate (Succ) as in PHC [Luo_2023_ICCV], where imitation is unsuccessful if the average deviation from reference is farther than 0.5m at any point in time. Succ measures whether the humanoid can track the reference motion without losing balance or lagging behind. The global MPJPE $E_{g-\text{mpjpe}}$ and the root-relative mean per-joint position error (MPJPE) $E_{\text{mpjpe}}$ (in mm) measures our policy’s ability to imitate the reference motion globally and locally (root-relative). To show physical realism, we report average joint acceleration $E_{\text{acc}}$ $\text{(mm/frame}^{2})$ and velocity $E_{\text{vel}}$ (mm/frame) error.

#### 4.1.1 Simulation Motion-Tracking Results

*Table 1: Simulation motion imitation evaluation of OmniH2O and baselines on dataset $\bm{\hat{Q}}$.*

Robustness Test.
As shown in [Figure 4](#S4.F4), we test the robustness of our control policy.
We use the same policy ${\pi_{\text{OmniH2O}}}$ across all tests, whether with fixed standing motion goals or motion goals controlled by joysticks, either moving forward or backward. With human punching and kicking from various angles, the robot, without external assistance, is able to maintain stability on its own. We also test OmniH2O on various outdoor terrains, including grass, slopes, gravel, etc*. OmniH2O demonstrates great robustness under disturbances and unstructured terrains.

*Figure 4: OmniH2O shows superior robustness against human strikes and different outdoor terrains.*

### 4.3 Autonomy via Frontier Models or Imitation Learning

To answer Q3, we need to bridge the whole-body tracking policy (physical intelligence*), with automated generation of kinematic motion goals through visual input (*semantic intelligence*). We explore two ways of automating humanoid control with OmniH2O: (1) using multi-modal frontier models to generate motion goals and (2) learning autonomous policies from the teleoperated dataset.

GPT-4o Autonomous Control.
We integrate our system, OmniH2O, with GPT-4o, utilizing a head-mounted camera on the humanoid to capture images for GPT-4o (Figure [5](#S4.F5)). The prompt (details in [Appendix M](#A13)) provided to GPT-4o offers several motion primitives for it to choose from, based on the current visual context.
We opt for motion primitives rather than directly generating motion goals because of GPT-4o’s relatively long response time.
As shown in [Figure 5](#S4.F5), the robot manages to give the correct punch based on the color of the target and successfully greets a human based on the intention indicated by human poses.

*Figure 5: OmniH2O sends egocentric RGB views to GPT-4o and executes the selected motion primitives.*

OmniH2O-6* Dataset.
We collect demonstration data via VR-based teleoperation. We consider six tasks: Catch-Release, Squat, Rope-Paper-Scissors, Hammer-Catch, Boxing, and Pasket-Pick-Place. Our dataset includes paired RGBD images from the head-mounted camera, the motion goals of H1’s head and hands with respect to the root, and joint targets for motor actuation, recorded at 30Hz.
For simple tasks such as Catch-Release, Squat, and Rope-Paper-Scissors, approximately 5 minutes of data are recorded, and for tasks like Hammer-Catch and Basket-Pick-Place, we collect approximately 10 minutes, leading to 40-min real-world humanoid teleoperated demonstrations in total.
Detailed task descriptions of the six open-sourced datasets are in Appendix LABEL:sec:lfddataset_appendix.

![Figure](x6.png)

*Figure 6: OmniH2O autonomously conducts four tasks using LfD models trained with our collected data.*

Catch-Release: Catch a red box and release it into a trash bin. This task has 13234 frames in total.
Squat: Squat when the robot sees a horizontal bar approaching that is lower than its head height. This task has 8535 frames in total.
Hammer-Catch: Use right hand to catch a hammer in a box. This task has 12759 frames in total.
Rock-Paper-Scissors: When the robot sees the person opposite it makes one of the rock-paper-scissors gestures, it should respond with the corresponding gesture that wins. This task has 9380 frames in total.
Boxing: When you see a blue boxing target, throw a left punch; when you see a red one, throw a right punch. This task has 11118 frames in total.
Basket-Pick-Place: Use your right hand to pick up the box and place it in the middle when the box is on the right side, and use your left hand if the box is on the left side. If you pick up the box with your right hand, place it on the left side using your left hand; if picked up with your left hand, place it on the right side using your right hand. This task has 18436 frames in total.

The detailed performance of 4 tasks is documented in Table [4.1.1](#S4.SS1.SSS1)

*Table 17: Quantitative LfD autonomous agents performance for 4 tasks.*

## Appendix L LfD Hyperparameters

In order to make the robot autonomous, we have developed a Learning from Demonstration (LfD) approach utilizing a diffusion policy that learns from a dataset we collected. The default training hyperparameters are shown below in [Table 19](#A12.T19).

*Table 19: Training Hyperparameters for the Lfd Training*

## Appendix M GPT-4o Prompt Example

Here is the example prompt we use for Autonomous Boxing task:

You’re a humanoid robot equipped with a camera slightly tilted downward on your head, providing a first-person perspective. I am assigning you a task: when a blue target appears in front of you, extend and then retract your left fist. When a red target appears, do the same with your right fist. If there is no target in front, remain stationary. I will provide you with three options each time: move your left hand forward, move your right hand forward, or stay motionless. You should directly respond with the corresponding options A, B, or C based on the current image. Note that, yourself is also wearing blue left boxing glove and right red boxing glove, please do not recognize them as the boxing target. Now, based on the current image, please provide me with the A, B, C answers.

For Autonomous Greetings with Human Task, our prompt is:

You are a humanoid robot equipped with a camera slightly tilted downward on your head, providing a first-person perspective. I am assigning you a new task to respond to human gestures in front of you. Remember, the person is standing facing you, so be mindful of their gestures. If the person extends their right hand to shake hands with you, use your right hand to shake their right hand (Option A). If the person opens both arms wide for a hug, open your arms wide to reciprocate the hug (Option B). If you see the person waving his hand as a gesture to say goodbye, respond by waving back (Option C). If no significant gestures are made, remain stationary (Option D). Respond directly with the corresponding options A, B, C, or D based on the current image and observed gestures. Directly reply with A, B, C, or D only, without any additional characters.

It is worth mentioning that we can use GPT-4 not only to choose motion primitive but also to directly generate the motion goal. The following prompt exemplifies this process:

You are a humanoid robot equipped with a camera slightly tilted downward on your head, providing a first-person perspective. I am assigning you a new task to respond to human gestures in front of you. If the person extends his left hand for a handshake, extend your left hand to reciprocate. If they extend their right hand, respond by extending your right hand. If the person opens both arms wide for a hug, open your arms wide to reciprocate the hug. If no significant gestures are made, remain stationary. Respond 6 numbers to represent the desired left and right hand 3D position with respect to your root position. For example: [0.25, 0.2, 0.3, 0.15, -0.19, 0.27] means the desired position of the left hand is 0.25m forward, 0.2m left, and 0.3m high compared to pelvis position, and the desired position of the right hand is 0.15m forward, 0.19m right and 0.27m high compared to pelvis position. The default stationary position should be (0.2, 0.2, 0.2, 0.2, -0.2, 0.2). Now please respond the 6d array based on the image to respond to the right hand shaking, left hand shaking, and hugging.

Generated on Mon Sep 29 17:15:56 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)