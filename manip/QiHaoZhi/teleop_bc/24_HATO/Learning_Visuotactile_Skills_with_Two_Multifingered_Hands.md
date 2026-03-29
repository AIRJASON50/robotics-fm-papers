[# Learning Visuotactile Skills with Two Multifingered Hands Toru Lin, Yu Zhang∗, Qiyang Li∗, Haozhi Qi∗, Brent Yi, Sergey Levine, and Jitendra Malik * Equal contribution.All authors are with University of California, Berkeley. Correspondence to toru@berkeley.edu ###### Abstract Aiming to replicate human-like dexterity, perceptual experiences, and motion patterns, we explore learning from human demonstrations using a bimanual system with multifingered hands and visuotactile data. Two significant challenges exist: the lack of an affordable and accessible teleoperation system suitable for a dual-arm setup with multifingered hands, and the scarcity of multifingered hand hardware equipped with touch sensing. To tackle the first challenge, we develop HATO, a low-cost hands-arms teleoperation system that leverages off-the-shelf electronics, complemented with a software suite that enables efficient data collection; the comprehensive software suite also supports multimodal data processing, scalable policy learning, and smooth policy deployment. To tackle the latter challenge, we introduce a novel hardware adaptation by repurposing two prosthetic hands equipped with touch sensors for research. Using visuotactile data collected from our system, we learn skills to complete long-horizon, high-precision tasks which are difficult to achieve without multifingered dexterity and touch feedback. Furthermore, we empirically investigate the effects of dataset size, sensing modality, and visual input preprocessing on policy learning. Our results mark a promising step forward in bimanual multifingered manipulation from visuotactile data. Videos, code, and datasets can be found here](https://toruowo.github.io/hato/).

## I Introduction

*Figure 1: Illustration of learned skills on four different tasks. Our learned policies complete long-horizon and high-precision tasks that require bimanual dexterity. (a) Slippery Handover tests the basic coordination skill between two hands. (b) Tower Block Stacking demonstrates the advantage of having a large contact area because of the flat palm, as well as the ability to maintain a stable pose during movements. (c) Wine Pouring requires the ability to grasp larger objects with the center of mass changing during manipulation. (d) Steak Serving requires the ability to use tools and maintain grasp stability when moving the objects on the tool.*

Achieving human-level dexterity is a long-term goal in the field of robotic manipulation. To this end, we explore the novel integration of a bimanual system with multifingered hands, visuotactile data modalities, and learning from human demonstrations, in hopes of mimicking the complexities of human anatomy, sensory experiences, and behavior patterns.

Most existing bimanual systems opt for parallel-jaw grippers [[[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3)]] as the end effectors, due to the high maintenance costs and limited availability of more advanced alternatives [[[4](#bib.bib4), [5](#bib.bib5), [6](#bib.bib6)]]. However, this choice greatly constrains the range of motions that can be achieved compared to multifingered hands [[[7](#bib.bib7)]], limiting abilities in adaptive grasping, in-hand manipulation, dexterous handovers, tool use, and bilateral coordination for complex tasks.

We assemble a bimanual system with multifingered hands to learn dexterous skills through visuotactile demonstrations. We confront two major challenges: the absence of an affordable and accessible teleoperation system suitable for a dual-arm setup with multifingered hands, and the scarcity of multifingered hand hardware equipped with touch sensing. Our main contributions include: (1) a novel hardware adaptation that repurposes prosthetic hands equipped with rich tactile sensing for research use; (2) HATO, a low-cost hands-arms teleoperation system built with commercially available virtual reality (VR) hardware, featuring novel mappings from teleoperator motion to robot control; (3) a comprehensive and versatile software suite for reliable and efficient data collection, multimodal data processing, scalable policy learning, and smooth policy deployment.

We design four complex tasks that examine our system’s capability in achieving intricate skills like two-hand coordination, manipulation of bulky objects, and sophisticated tool use. From only 30 minutes to 2 hours of teleoperation data collected using HATO (including around 5 to 10 minutes of practice time), we are able to obtain dexterous bimanual manipulation policies with visuotactile observations that can adeptly complete all tasks through pure end-to-end learning. Our system demonstrates natural and human-like skills and showcases unprecedented dexterity.

We also perform a thorough ablation study on the effects of dataset size, sensing modality, and visual input preprocessing on policy learning. Most notably, we find that vision and touch significantly enhance learning efficiency, policy success rate, and policy robustness. Without touch or vision, the policies are not able to consistently succeed or sometimes completely fail, highlighting the importance of high-quality touch sensing for enabling human-level dexterity. Our experiments also reveal that a dataset comprising a few hundred demonstrations is sufficient for learning effective bimanual dexterous policies. Additionally, we confirm the vital role of wrist-mounted cameras in boosting policy performance, while noting that depth information does not markedly benefit the learning process.

In the spirit of fostering further research and collaboration, we will open-source all our hardware and software systems and make the teleoperation dataset we have collected publicly available to facilitate evaluation and replication of our work.

## II Related Work

Bimanual hands-arms manipulation.
Prior bimanual manipulation systems usually use parallel-jaw grippers [[[1](#bib.bib1), [8](#bib.bib8), [9](#bib.bib9), [10](#bib.bib10)]] for their simplicity and durability. Early work uses classical control for dexterous manipulation [[[11](#bib.bib11), [12](#bib.bib12), [13](#bib.bib13)]], but these approaches are largely task-specific and require expert knowledge of system dynamics and task structures. Reinforcement learning can alleviate these issues, but they tend to have high sample complexity and are only tractable in simulation [[[14](#bib.bib14), [15](#bib.bib15)]]. Although there has been progress in transferring the policies from simulation to the real-world for a single hand [[[16](#bib.bib16), [17](#bib.bib17), [18](#bib.bib18), [19](#bib.bib19)]] or two hands [[[20](#bib.bib20), [21](#bib.bib21)]], the policies usually suffer from the sim-to-real gap.

Imitation learning. Another promising approach to achieve human-level dexterity is learning from demonstrations [[[22](#bib.bib22), [23](#bib.bib23)]]. More recently, researchers have utilized deep neural networks to obtain better representations and policies [[[24](#bib.bib24)]]. While bimanual manipulation has also been studied in this context [[[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3), [25](#bib.bib25)]], these studies only demonstrate systems with parallel-jaw grippers. Wang et al. [[[26](#bib.bib26)]] is concurrent work that showcases bimanual dexterous manipulation skills. Unlike [[[26](#bib.bib26)]], which utilizes only vision and proprioception as inputs, our work additionally utilizes tactile input and shows that touch sensing is critical for the reliable completion of many challenging tasks considered.

Bimanual teleoperation. Having access to a diverse set of high-quality demonstrations is crucial for learning a high-quality policy. Existing teleoperation systems are largely restricted to the use of parallel-jaw grippers [[[1](#bib.bib1), [2](#bib.bib2), [27](#bib.bib27), [28](#bib.bib28), [29](#bib.bib29), [30](#bib.bib30)]] or single-handed scenarios [[[31](#bib.bib31), [32](#bib.bib32), [33](#bib.bib33), [34](#bib.bib34)]]. In addition, the hand teleoperation systems heavily rely on retargeting to solve morphology differences, thus introducing unavoidable latency and are not intuitive to use. In contrast, our system teleoperates a robot hand by separating finger control into thumb control and power grasp control, providing a smoother user experience.

Learning visuotactile skills. Our system also features rich visuotactile sensory data. Integrated visuotactile sensing has been used for numerous applications such as grasping [[[35](#bib.bib35)]], in-hand manipulation [[[36](#bib.bib36)]], object shape reconstruction [[[37](#bib.bib37), [38](#bib.bib38), [39](#bib.bib39)]] and object recognition [[[40](#bib.bib40)]], cloth manipulation [[[41](#bib.bib41)]], and learning representation for object interactions [[[42](#bib.bib42), [43](#bib.bib43)]]. However, none of them use two multifingered hands, and previous bimanual manipulation systems are either not equipped with dexterous hands or lack rich sensory signals. To the best of our knowledge, our work is the first at the intersection of bimanual dexterous manipulation and imitation learning from visuotactile inputs.

*Figure 2: Fingertip Tactile Sensor Layout. There are six tactile sensors on each of the fingertips. Each tactile sensor provides a continuous value proportional to the sensed pressure.*

## III HATO: Hands-Arms Tele-Operation

*Figure 3: Common failures from parallel-jaw gripper teleoperation. When the object has a slippery and rigid surface or is larger than the gripper, grasping with a parallel-jaw gripper requires very accurate planning, which is difficult to achieve during teleoperation. This difficulty leads to various failure modes, including slipping objects, unstable holds, and unstable grasps. In contrast, multifingered hands provide additional redundancy and contact areas to maintain a grasp.*

We develop HATO, a novel teleoperation system for bimanual multifingered hands. Our system is easy to set up and intuitive to use, enabling efficient collection of bimanual dexterous manipulation data. An overview of our system is shown in Figure LABEL:fig:teaser.
For teleoperation of each hand-arm pair, HATO maps a Meta Quest 2 virtual reality (VR) controller’s pose to the end-effector pose of the robot arm, and the controller’s grip button and thumbstick to the hand’s joint positions. The HATO software suite includes a data collection pipeline that records and processes data from all available sensing modalities (vision, touch, and proprioception).
We provide details below on the hardware setup, teleoperation pipeline, and data infrastructure.

### III-A Robot Setup

Robot arms. We use two UR5e robot arms for our manipulation system. The UR5e is an industrial arm with six degrees of freedom (DoF). Each DoF is a revolute joint that has a working range of $[-2\pi,2\pi]$.

Robot hands. We attach two Psyonic Ability Hands to the UR5e arms as end effectors. These hands were originally designed for prosthetic use [[[44](#bib.bib44)]]; we repurpose them for research by designing custom printed circuit boards (PCBs) that simplify electrical wiring by integrating communication interfaces with power distribution. Each hand has five fingers, and each finger has six actuated DoFs (one DoF per finger, two for the thumb). Each actuated DoF of the hand is a revolute joint that has upper and lower limits similar to human finger limits (see the leftmost and rightmost columns of Figure LABEL:fig:teaser (b)). For the four non-thumb fingers, the metacarpophalangeal (MCP) joints serve as the actuating DoF; the MCP joints connect to the proximal interphalangeal (PIP) joints via a four-bar linkage mechanism, contributing to an additional underactuated DoF on each finger. Each fingertip also comes with six touch sensors (see Figure [2](#S2.F2)).

### III-B Teleoperation Setup

Our teleoperation system leverages the Meta Quest 2 platform. It comes with a VR headset and a pair of controllers, each designated for one hand. The Quest combines visual tracking with Inertial Measurement Unit (IMU) sensors to capture the spatial orientation and movements of the controllers. Each controller is also outfitted with a range of interactive buttons, including thumbsticks, trigger buttons, and grip buttons. Using a VR application like oculus_reader [[[45](#bib.bib45)]], one can stream data related to the controllers’ poses and button states in real-time. Our main contribution is the development of a software suite that provides flexible options for translating movements detected by the Quest controllers to precise control commands for a bimanual multifingered robotic system. We highlight a simple yet effective mapping from controller buttons to multifingered hand movements, which enables a smooth and intuitive user experience.

Arm control. We first read the pose measurements from the Quest controller, then transform the pose to a desired end-effector (EEF) pose of the robot’s coordinate system. We use inverse kinematics (IK) to solve for the joint positions given the desired EEF pose, and send the joint position command to the UR5e arm. In the case of IK solving failure, we use the last commanded joint positions. Aside from this control implementation, our software suite also contains two other implementations. The second implementation uses the first-order approximation of the IK solution which linearly maps the delta pose between the desired EEF pose and the current EEF pose of the arm to the delta positions of the joints. The third implementation directly sends the end-effector position to the arm and the IK is done onboard. For the rest of the paper, we use the first implementation for both teleoperation and policy execution.

Hand control. We map the controller’s grip button to the joint positions of the four non-thumb fingers (4 DoF), and map the thumbstick readings to joint positions of the thumb (2 DoF). Readings from the grip button and thumbstick of each controller are re-normalized based on the joint ranges of their corresponding hand DoF, and sent to the Ability Hand as position control targets. Specifically, pressing/releasing the grip button controls flexion/extension of the non-thumb fingers (Figure LABEL:fig:teaser (b)), and the 2-D positions of the thumbstick control the thumb joints’ flexion/extension and abduction/adduction (Figure LABEL:fig:teaser (c)). The hand’s power grasp is therefore a continuous movement proportional to the operator’s pressing force on the grip button, allowing fine-grained control when grasping soft or deformable objects. While such mapping sacrifices the ability to perform sophisticated finger-gaiting, it provides an intuitive user interface and is still able to complete various grasps for complex tasks [[[46](#bib.bib46)]].

Pause-and-adjust. We follow a conventional design where we use the controller’s trigger button to start and break a continuous arm control sequence, allowing the teleoperator to pause during a teleoperation trial and adjust their posture. This design greatly helps with tasks that are close to the teleoperator’s physical limit, e.g., those that require a large reach of the teleoperator’s arm.

Such mapping also makes HATO flexible enough to control potentially any robot arms with EEF control and hands with an independent anthropomorphic thumb and the ability to perform power grasps.

### III-C Data Collection and Preprocessing for Policy Learning

We collect multimodal data from both hands and arms by running HATO data collection pipeline at 10Hz. The data include the proprioceptive states of both the UR5e arms and the Ability Hands, the RGB-D images from three RealSense depth cameras (two mounted on the wrist of each hand, one mounted at a stationary “head-view” position), the touch sensor readings from the Ability Hands, and the control commands given to the UR5e arms and the Ability Hands.

Proprioception. Our proprioceptive state data at each time step includes the current joint positions of both arms, the current finger positions of the hands, and the current end-effector pose (represented as a concatenation of translation and axis-angle).

Vision. We obtain RGB-D image data from three cameras at each time step; both the RGB and depth images are streamed at a resolution of $480\times 640$. We resize all images to $240\times 320$ before feeding them into the network.

*Figure 4: How vision and touch affect the policy performance across four challenging tasks. Across all tasks, vision is crucial for the policy to achieve low prediction error. For Block Stacking and Steak Serving, removing the touch input does not significantly influence the prediction error, but as we show in Table [IV](#S5.T4) and [IV](#S5.T4), touch is also crucial for successfully completing these two tasks.*

Touch. Each finger of the Ability Hand has six touch sensors attached (Figure [2](#S2.F2)), contributing to a total of sixty touch sensor readings from both hands $h\in\mathbb{R}^{60}$. Each touch sensor reading is a continuous number whose range changes depending on manufacturing tolerances. The readings typically lie in the range of $[200,400]$ (ADC values) when there is no contact event, and above $1000$ on contact.

Control actions. For each arm, the control action is the desired joint position for each of the six revolute joints of the arm, where each value in the vector represents the angle of the revolute joint (in $[-2\pi,2\pi]$). For each hand, the control action is the desired joint position for each of the six finger joints (independent of the arm control implementation).

Data normalization. All values are scaled linearly per dimension to be between $-1$ and $1$, with the minimum value mapped to $-1$ and the maximum value mapped to $1$. The minimum and maximum values are obtained from the training data, except for the joint position reading and the control action of the hand. We use a minimum value of 0 and maximum values of $[110,110,110,110,90,120]$ for each of the six finger joint positions.
For depth and RGB images, we use their raw values: $[0,255]$ for the RGB images and $[0,65535]$ for the depth images.

## IV Learning Visuotactile Skills with HATO

With visuotactile demonstration data collected from HATO, we can learn a variety of bimanual dexterous skills for complex tasks. In this section, we describe how we train diffusion policies [[[47](#bib.bib47)]] using the collected data. Importantly, we also propose a novel asynchronous inference algorithm, which is the key to our fast and smooth policy deployment.

### IV-A Learning

Our system learns bimanual dexterous skills from demonstration data by treating action prediction as a conditional generation problem. Following [[[47](#bib.bib47)]], with an observation horizon of $1$, we predict the action sequence of length $16$ from the current input observations using a denoising diffusion probabilistic model (DDPM) [[[48](#bib.bib48)]]. Each input observation is a collection of observation data from multiple modalities: proprioception, vision, and touch. Each action is a $24$-dimensional vector that specifies the desired joint positions for the two arms and the two hands. We choose to use only a single observation input as opposed to a short horizon of observations (as done in [[[47](#bib.bib47)]]) as we have found that using one single observation is sufficient for the policy to perform well and is much faster to train.

Proprioception.
We use end-effector poses as the proprioception observations and do not include the arm joint positions. This is because the UR5 arms do not have redundant joints, and the joint positions can vary very unpredictably near singularity during teleoperation, posing bigger learning challenges. The proprioception is passed through a two-layer network with ReLU activation, a hidden size of $256$, and an output feature size of $64$.

Touch. The touch signal is also passed through a two-layer network in the same way as proprioception.

Vision. For image and depth observations from the three cameras, we follow the prior work on diffusion policy [[[47](#bib.bib47)]] to use the ResNet-18 architecture [[[49](#bib.bib49)]] and replace all the BatchNorm [[[50](#bib.bib50)]] in the network with GroupNorm [[[51](#bib.bib51)]]. The fully connected layer’s output size is adjusted to be $32$. We do not share network weights across camera inputs.

Diffusion architecture. All the encoded image, depth, tactile, and proprioceptive observations are then concatenated as the input to a diffusion model (CNN-based in [[[47](#bib.bib47)]]). We also use the same noise schedule (square cosine schedule) and the same number of diffusion steps (100) for training.

Output action. The diffusion output from the model is the normalized 6 DoF absolute desired joint positions of each UR5e arm, and the 6 DoF normalized (00 to $1$) desired joint positions of each Ability hand.

Optimization details. We use the AdamW optimizer [[[52](#bib.bib52), [53](#bib.bib53)]] with a learning rate of $0.0001$, weight decay of $0.00001$, and a batch size of 128. Following [[[47](#bib.bib47)]], we maintain an exponential weighted average of the model weights and use it during evaluation/deployment.

### IV-B Deployment

At deployment time, we use an asynchronous setup where the diffusion model prediction and the robot execution run in parallel. In particular, we use a remote inference server that keeps track of the most recent observation and the timestep to which the observation corresponds. The local process sends the new observation to the remote inference server at every control step. The inference server continuously runs the diffusion model on the latest observation and produces the action sequence prediction. Then, it sends the action sequence prediction (with the corresponding timesteps) to the local process, where it computes the average of the predictions over multiple timesteps (similar to the temporal ensemble in [[[1](#bib.bib1)]]). Note that this is different from how deployment is done in [[[47](#bib.bib47)]], where they do not use a temporal ensemble. We have found that the inclusion of action aggregation greatly improves motion smoothness. For inference, we use 15 diffusion steps.

## V Experiments

We consider four challenging real-world tasks (Figure [1](#S1.F1)) to study the bimanual dexterity enabled by our system. We validate the effectiveness of our system setup, data collection pipeline, policy learning, and deployment pipelines by demonstrating teleoperation capabilities and showcasing learned skills that successfully complete these tasks. We also conduct an empirical investigation on how policy performance is influenced by data size and sensing modalities.

Slippery Handover.* Handing over objects is a motor skill commonly required for a wide range of daily activities; it is also commonly used as a bimanual manipulation task [[[54](#bib.bib54)]].
Each task episode is initialized with the slippery object resting on a box. One hand needs to pick up the object and hand it over to the other hand. The episode succeeds when the other hand holds the object stably and moves away from the first hand by a distance of more than $10$ centimeters.
As we will demonstrate, parallel grippers can face more challenges when handling objects with slippery surfaces compared to multifingered hands.
The anthropomorphic hand morphology greatly mitigates the slippery challenge due to the larger contact area and additional support that it provides to objects.

*TABLE I: Success rate on each of the four challenging bimanual manipulation tasks. For Slippery Handover and Wine Pouring, we use only image observation and proprioceptive state as we find these two inputs are sufficient to achieve an almost 100% success rate. For Block Stacking and Steak Serving, we use image, proprioception, and touch as inputs. The pickup success is an intermediate metric that measures how often the hands successfully pick up both objects.*

*Tower Block Stacking*. Manipulation of bulky objects is another common skill required for many everyday tasks. Motivated by manual labor jobs in construction sites where workers need to use two hands to move bricks, we design a tower block stacking task. Each task episode begins with two piles of large blocks on the table, one consisting of two blocks (red and blue) and the other a single yellow block. The robot needs to move up the pile of two blocks and stack it on top of the yellow block. A successful episode is marked when the two moved blocks stay stably on the yellow block after being released from the robot hands.

*Wine Pouring*. At task initialization, a large wine bottle and a small cup are placed on a stool on top of a table. The robot needs to use one hand to grasp the wine bottle, use the other hand to grasp the cup, perform a pouring motion from the bottle to the cup, and put both the bottle and the cup back. The wine bottle is filled with transparent beads to simulate liquid. Since the center of mass quickly changes during the pouring process, we hypothesize that a power grasp with a multifingered hand can greatly reduce the difficulty.

*Steak Serving*. Inspired by cooking activities, we design a long-horizon task that requires prehensile grasps and intricate force feedback control loops for humans. The goal of this task is to serve a piece of cooked steak onto a plate. At task initialization, three objects are placed on the table: a cooking pan with a piece of steak inside, a spatula for serving, and a ceramic plate. The robot is expected to use one hand to hold the pan and the other hand to grasp the spatula. It then needs to insert the spatula under the bottom of the steak and lift it up. The task is completed when the robot successfully holds the spatula with the steak and serves it onto the plate.

Teleoperation data collection details. For *Slippery Handover* and *Tower Block Stacking*, we collect 100 demonstrations, each demonstration lasting around 6 seconds and 20 seconds respectively. For *Wine Pouring* and *Steak Serving*, we collect 300 demonstrations, each demonstration lasting around 25 seconds and 40 seconds respectively. Before the data collection for each task, we ask the human teleoperator to practice for 5 to 10 minutes.

*Figure 5: How does the demonstration dataset size affect policy prediction error? Across all tasks, having more demonstration trajectories consistently leads to lower prediction loss. In particular, the policy performance saturates for Block Stacking* at 75 demonstrations, *Wine Pouring* at 200 demonstrations, and *Steak Serving* at 100 demonstrations.*

### V-A Capabilities from Teleoperation

We qualitatively investigate whether having multifingered hands as end-effectors allows for better manipulation capabilities than parallel-jaw grippers by comparing their performances on the four manipulation tasks above. In particular, to test the manipulation capability of the parallel-jaw gripper, we keep the rest of the system the same while replacing the Ability hand with the Robotiq gripper and mapping the same grip button on the Quest controller to the gripper’s open/close control. With multifingered hand end effectors, previously inexperienced teleoperators are able to collect hundreds of high-quality demonstrations within a few hours. On the other hand, with parallel-jaw grippers, the teleoperation suffers from a variety of failure modes, including slipping objects, shaky holds, unstable grasping, and unstable balancing. Some common task failures are shown in Figure [3](#S3.F3).

### V-B Capabilities from Learning

We validate the effectiveness of HATO as a data collection pipeline by demonstrating successful policies trained from HATO-collected datasets. In particular, we record the task success rate of learned policies using 10 deployment trials. In addition to the success rate for the full task, we also record how many times each policy successfully picks up the object(s) (e.g., bottle and cup for pouring, pan and spatula for steak serving, two blocks for stacking, and banana for handover) as the partial task completion rate. As shown in Table [I](#S5.T1), our policy is able to pick up the object(s) with a 100% success rate across all tasks. Our policy is also able to complete three of the four tasks (handover, stacking, and pouring) consistently with near 100% success rate. The last task, steak serving, is much more difficult to learn due to its long task horizon and the demand for high-precision control (e.g., balancing the steak on a spatula held by a hand). Despite such difficulty, our policy is still able to achieve around a 50% success rate.

### V-C Learning Efficiency

We study the efficiency of our learning method by empirically evaluating the correlation between the number of demonstrations and policy performance. We use the mean squared error between predicted actions and ground truth actions (ActionMSE) on a held-out test set as the evaluation metric. The evaluation dataset consists of $10$ demonstration trajectories for Slippery Handover and Block Stacking, and $20$ demonstration trajectories for Wine Pouring and Steak Serving. The results are shown in Figure [5](#S5.F5). As we expect, the prediction error decreases as the training dataset size increases. It is worth highlighting that the ActionMSE metric roughly saturates for Block Stacking, Steak Serving, and Wine Pouring at around 75, 100, and 200 demonstrations each, respectively. For the other tasks, Slippery Handover, it is possible that more demonstrations may further improve policy robustness.

### V-D Importance of Vision and Touch

Our policy takes three types of sensing modalities as inputs: proprioception, vision, and touch. In this section, we quantitatively investigate how the visuotactile sensing modalities affect policy learning and performance.

Vision and touch are crucial for learning.
Figure [4](#S3.F4) shows the ActionMSE for policies trained with different sensing modalities. For all four tasks, the policy trained with no vision has a much higher prediction error than the policy trained with all three sensing modalities. For all tasks except for Steak Serving, the policy trained with no touch has a higher prediction error than the policy trained with all three sensing modalities. Policies trained with neither vision nor touch consistently have the highest ActionMSE. As we will show in the following section, such a correlation also translates to the policy success rate.

Vision and touch improve policy success rate.
On the Steak Serving task, we evaluate the success rate of the policies trained with different sensing modalities and report the results in Table [IV](#S5.T4). Without vision, the policy fails at the first task stage, i.e., properly picking up the objects (0/10 success rate). Without touch, the policy is able to accomplish the first task stage (i.e., pick up the pan and the spatula) but fails to transfer the steak over to the plate. It is worth highlighting that even though the ActionMSE metric is similar for the policies trained with or without touch ($0.07$ vs $0.08$), these policies have vastly different success rates: 0/10 (without touch) vs. 5/10 (with touch). We believe that this is because ActionMSE cannot fully capture how well the diffusion policy fits the dataset distribution. A potentially better metric would be to estimate the log-likelihood of the data under the diffusion policy output distribution, but that has been notoriously difficult to estimate [[[55](#bib.bib55)]]. In our experience, we found that the ActionMSE was able to mostly inform us which policy is expected to perform well.

Vision and touch improve policy robustness.
To further understand how the sensing modality affects the robustness of our policy, we experiment with a less common scene initialization for the Block Stacking. Specifically, we rotate the blocks in random directions. Figure [6](#S6.F6) shows a comparison between the default initialization and the rare initialization. This scene initialization is less encountered in the demonstration dataset, and the perturbed block configuration makes the two-block pile harder to be picked up by two robot hands. In Table [IV](#S5.T4), we show a comparison of the success rate for this rare initialization and the default initialization across three different sensing configurations. While touch and vision are not needed for the default scene initialization (they all achieve 100% success rate), for the rare scene initialization, the policy trained without touch can only succeed 4/10 times, and the policy trained without vision cannot succeed at all. This suggests that vision and touch sensing modalities allow the policy to be more robust to rare scenarios.

Wrist camera vs third-view camera. We study the effect of different camera positions. The results are shown in Figure [7](#S6.F7). Prediction error with only the wrist-view camera is consistently lower than that with only the third-view camera across the Slippery Handover, Block Stacking, and Steak Serving tasks; the two errors are comparable on the Wine Pouring task. We hypothesize that this is because wrist-view cameras contain richer information on task-relevant object states, due to the less occluded object view and more spatial hints via induced perspectives during arm movement.

Use of depth. We examine the prediction errors of the policies trained with and without depth information. The results are shown in Figure [7](#S6.F7). Across all four tasks, adding depth does not provide a marked benefit in terms of the ActionMSE metric, sometimes even hurting performance (e.g., Wine Pouring). We hypothesize that the noisy depth readings cause more harm than good for learning.

## VI Discussions

In this work, we share novel engineering insights that enable high-performance policy learning from human demonstrations using a bimanual system with multifingered hands and visuotactile sensing, and showcase the dexterous manipulation capabilities achieved by our system. In particular, we show that visuotactile sensing is the key for our policies to complete complex and long-horizon tasks consistently and robustly. Our low-cost teleoperation system opens up a number of avenues for future research. For example, equipping our teleoperation system with haptic feedback (e.g., attached to tactile sensing) could greatly improve the user experience and consequently the quality of teleoperation data. Furthermore, our policy is learned entirely from scratch with no pre-training, making it susceptible to appearance changes in the scene. Training more robust and generalizable policies is also an interesting future research direction.

*Figure 6: Two scene initializations for the Block Stacking task. The default initialization* is shown on the left and the *rare initialization* is shown on the right. In the *rare initialization*, the perturbed block configuration makes the two-block pile harder to pick up by two robot hands and more difficult to stack stably on another block.*

*Figure 7: How does each type of image observation affect policy prediction error? Aside from the vision modality, all policies are trained with proprioception and touch. The policies trained with only the third-view camera (3${}^{\text{rd}}$ View Only*) have higher prediction errors, except for the *Wine Pouring* task. Policies with all three cameras that additionally include depth images (*Ours with Depth*) have a similar prediction error to the policies trained without depth (*Ours*), except on *Wine Pouring*.*

## Acknowledgment

We thank Jesse Cornman from PSYONIC for help with setting up the Ability Hands, and Philipp Wu for help with setting up the UR5e robot arms and GELLO.
TL is supported by fellowships from the National Science Foundation and UC Berkeley.
QL is supported by ONR under N00014-20-1-2383, and NSF IIS-2150826.
HQ is supported by the DARPA Machine Common Sense and ONR MURI N00014-21-1-2801. This research was also partly supported by Savio computational cluster provided by the Berkeley Research Compute program.

## References

-
[1]

T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in *RSS*, 2023.

-
[2]

Z. Fu, T. Z. Zhao, and C. Finn, “Mobile ALOHA: Learning bimanual mobile manipulation with low-cost whole-body teleoperation,” *arXiv:2401.02117*, 2024.

-
[3]

C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots,” *arXiv:2402.10329*, 2024.

-
[4]

[https://www.allegrohand.com/](https://www.allegrohand.com/).

-
[5]

[https://schunk.com/us/en/gripping-systems/special-gripper/svh/c/PGR˙3161](https://schunk.com/us/en/gripping-systems/special-gripper/svh/c/PGR_3161).

-
[6]

[https://www.shadowrobot.com/dexterous-hand-series/](https://www.shadowrobot.com/dexterous-hand-series/).

-
[7]

A. Bicchi and V. Kumar, “Robotic grasping and contact: A review,” in *ICRA*, 2000.

-
[8]

F. Caccavale, P. Chiacchio, A. Marino, and L. Villani, “Six-DOF impedance control of dual-arm cooperative manipulators,” *Transactions on Mechatronics*, 2008.

-
[9]

N. Sarkar, X. Yun, and V. R. Kumar, “Dynamic control of 3-D rolling contacts in two-arm manipulation,” in *ICRA*, 1993.

-
[10]

R. Platt, A. H. Fagg, and R. A. Grupen, “Manipulation Gaits: Sequences of Grasp Control Tasks,” in *ICRA*, 2004.

-
[11]

C. Ott, O. Eiberger, W. Friedl, B. Bauml, U. Hillenbrand, C. Borst, A. Albu-Schaffer, B. Brunner, H. Hirschmuller, S. Kielhofer, R. Konietschke, M. Suppa, T. Wimbock, F. Zacharias, and G. Hirzinger, “A humanoid two-arm system for dexterous manipulation,” in *Humanoids*, 2006.

-
[12]

J. Steffen, C. Elbrechter, R. Haschke, and H. Ritter, “Bio-inspired motion strategies for a bimanual manipulation task,” in *Humanoids*, 2010.

-
[13]

N. Vahrenkamp, M. Przybylski, T. Asfour, and R. Dillmann, “Bimanual grasp planning,” in *Humanoids*, 2011.

-
[14]

Y. Chen, T. Wu, S. Wang, X. Feng, J. Jiang, Z. Lu, S. McAleer, H. Dong, S.-C. Zhu, and Y. Yang, “Towards human-level bimanual dexterous manipulation with reinforcement learning,” in *NeurIPS*, 2022.

-
[15]

K. Zakka, P. Wu, L. Smith, N. Gileadi, T. Howell, X. B. Peng, S. Singh, Y. Tassa, P. Florence, A. Zeng, and P. Abbeel, “RoboPianist: Dexterous piano playing with deep reinforcement learning,” in *CoRL*, 2023.

-
[16]

H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik, “In-hand object rotation via rapid motor adaptation,” in *CoRL*, 2022.

-
[17]

T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal, “Visual dexterity: In-hand dexterous manipulation from depth,” *Science Robotics*, 2023.

-
[18]

OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Józefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba, “Learning dexterous in-hand manipulation,” *IJRR*, 2019.

-
[19]

A. Handa, A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, Y. Narang, J.-F. Lafleche, D. Fox, and G. State, “Dextreme: Transfer of agile in-hand manipulation from simulation to reality,” in *ICRA*, 2023.

-
[20]

B. Huang, Y. Chen, T. Wang, Y. Qin, Y. Yang, N. Atanasov, and X. Wang, “Dynamic handover: Throw and catch with bimanual hands,” in *CoRL*, 2023.

-
[21]

T. Lin, Z.-H. Yin, H. Qi, P. Abbeel, and J. Malik, “Twisting lids off with two hands,” *arXiv:2403.02338*, 2024.

-
[22]

A. Billard, S. Calinon, R. Dillmann, and S. Schaal, “Survey: Robot programming by demonstration,” *Springer handbook of robotics*, 2008.

-
[23]

A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne, “Imitation learning: A survey of learning methods,” *ACM Computing Surveys*, 2017.

-
[24]

H. Ravichandar, A. S. Polydoros, S. Chernova, and A. Billard, “Recent advances in robot learning from demonstration,” *Annual review of control, robotics, and autonomous systems*, 2020.

-
[25]

J. Grannen, Y. Wu, B. Vu, and D. Sadigh, “Stabilize to Act: Learning to coordinate for bimanual manipulation,” in *CoRL*, 2023.

-
[26]

C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu, “DexCap: Scalable and portable mocap data collection system for dexterous manipulation,” *arXiv:2403.07788*, 2024.

-
[27]

H. Fang, H.-S. Fang, Y. Wang, J. Ren, J. Chen, R. Zhang, W. Wang, and C. Lu, “Low-cost exoskeletons for learning whole-arm manipulation in the wild,” in *ICRA*, 2023.

-
[28]

M. Seo, S. Han, K. Sim, S. H. Bang, C. Gonzalez, L. Sentis, and Y. Zhu, “Deep imitation learning for humanoid loco-manipulation through human teleoperation,” in *Humanoids*, 2023.

-
[29]

P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel, “Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators,” *arXiv:2309.13037*, 2023.

-
[30]

S. P. Arunachalam, S. Silwal, B. Evans, and L. Pinto, “Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation,” in *ICRA*, 2023.

-
[31]

S. P. Arunachalam, I. Güzey, S. Chintala, and L. Pinto, “Holo-Dex: Teaching dexterity with immersive mixed reality,” in *ICRA*, 2023.

-
[32]

Y. Qin, H. Su, and X. Wang, “From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation,” *RA-L*, 2022.

-
[33]

A. Sivakumar, K. Shaw, and D. Pathak, “Robotic telekinesis: Learning a robotic hand imitator by watching humans on youtube,” in *RSS*, 2022.

-
[34]

A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto, “Open teach: A versatile teleoperation system for robotic manipulation,” *arXiv:2403.07870*, 2024.

-
[35]

R. Calandra, A. Owens, D. Jayaraman, W. Yuan, J. Lin, J. Malik, E. H. Adelson, and S. Levine, “More than a feeling: Learning to grasp and regrasp using vision and touch,” *RA-L*, 2018.

-
[36]

H. Qi, B. Yi, S. Suresh, M. Lambeta, Y. Ma, R. Calandra, and J. Malik, “General in-hand object rotation with vision and touch,” in *CoRL*, 2023.

-
[37]

E. Smith, D. Meger, L. Pineda, R. Calandra, J. Malik, A. Romero Soriano, and M. Drozdzal, “Active 3d shape reconstruction from vision and touch,” in *NeurIPS*, 2021.

-
[38]

S. Suresh, H. Qi, T. Wu, T. Fan, L. Pineda, M. Lambeta, J. Malik, M. Kalakrishnan, R. Calandra, M. Kaess, J. Ortiz, and M. Mukadam, “Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation,” *arXiv:2312.13469*, 2023.

-
[39]

S. Suresh, Z. Si, J. G. Mangelson, W. Yuan, and M. Kaess, “Shapemap 3-d: Efficient shape mapping through dense touch and vision,” in *ICRA*, 2022.

-
[40]

J. Xu, H. Lin, S. Song, and M. Ciocarlie, “Tandem3d: Active tactile exploration for 3d object recognition,” in *ICRA*, 2023.

-
[41]

N. Sunil, S. Wang, Y. She, E. Adelson, and A. R. Garcia, “Visuotactile affordances for cloth manipulation with local control,” in *CoRL*, 2022.

-
[42]

I. Guzey, B. Evans, S. Chintala, and L. Pinto, “Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play,” in *CoRL*, 2023.

-
[43]

I. Guzey, Y. Dai, B. Evans, S. Chintala, and L. Pinto, “See to touch: Learning tactile dexterity through visual incentives,” *arXiv:2309.12300*, 2023.

-
[44]

A. Akhtar, J. A. Austin, J. M. Cornman, D. M. Bala, and Z. Wang, “System and method for an advanced prosthetic hand,” Mar 2021.

-
[45]

[https://github.com/rail-berkeley/oculus˙reader](https://github.com/rail-berkeley/oculus_reader).

-
[46]

T. Feix, J. Romero, H.-B. Schmiedmayer, A. M. Dollar, and D. Kragic, “The grasp taxonomy of human grasp types,” *Transactions on human-machine systems*, 2015.

-
[47]

C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in *RSS*, 2023.

-
[48]

J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in *NeurIPS*, 2020.

-
[49]

K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *CVPR*, 2016.

-
[50]

S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *ICML*, 2015.

-
[51]

Y. Wu and K. He, “Group normalization,” in *ECCV*, 2018.

-
[52]

D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *ICLR*, 2015.

-
[53]

I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” *arXiv preprint arXiv:1711.05101*, 2017.

-
[54]

Y. Li, C. Pan, H. Xu, X. Wang, and Y. Wu, “Efficient bimanual handover and rearrangement via symmetry-aware actor-critic learning,” in *ICRA*, 2023.

-
[55]

Y. Song, C. Durkan, I. Murray, and S. Ermon, “Maximum likelihood training of score-based diffusion models,” in *NeurIPS*, 2021.

Generated on Fri May 24 14:57:34 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)