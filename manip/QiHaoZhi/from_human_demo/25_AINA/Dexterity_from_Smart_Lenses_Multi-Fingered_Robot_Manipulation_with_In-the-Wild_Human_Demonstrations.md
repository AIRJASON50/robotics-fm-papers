[# Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations Irmak Guzey1,2 Haozhi Qi2 Julen Urain2 Changhao Wang2 Jessica Yin2 Krishna Bodduluri2 Mike Lambeta2 Lerrel Pinto1 Akshara Rai2 Jitendra Malik2 Tingfan Wu2 Akash Sharma2 Homanga Bharadhwaj2 1 New York University, 2 Meta aina-robot.github.io](https://aina-robot.github.io)

Correspondence to irmakguzey@nyu.edu.

###### Abstract

Learning multi-fingered robot policies from humans performing daily tasks in natural environments has long been a grand goal in the robotics community. Achieving this would mark significant progress toward generalizable robot manipulation in human environments, as it would reduce the reliance on labor-intensive robot data collection. Despite substantial efforts, progress toward this goal has been bottle-necked by the embodiment gap between humans and robots, as well as by difficulties in extracting relevant contextual and motion cues that enable learning of autonomous policies from in-the-wild human videos. We claim that with simple yet sufficiently powerful hardware for obtaining human data and our proposed framework Aina, we are now one significant step closer to achieving this dream. Aina enables learning multi-fingered policies from data collected by anyone, anywhere, and in any environment using Aria Gen 2 glasses. These glasses are lightweight and portable, feature a high-resolution RGB camera, provide accurate on-board 3D head and hand poses, and offer a wide stereo view that can be leveraged for depth estimation of the scene. This setup enables the learning of 3D point-based policies for multi-fingered hands that are robust to background changes and can be deployed directly without requiring any robot data (including online corrections, reinforcement learning, or simulation). We compare our framework against prior human-to-robot policy learning approaches, ablate our design choices, and demonstrate results across nine everyday manipulation tasks. Robot rollouts are best viewed on our website: [https://aina-robot.github.io](https://aina-robot.github.io).

## I Introduction

![Figure](x1.png)

*Figure 1: Comparison of Aina’s capabilities with some prior human-to-robot learning frameworks. In-The-Wild indicates whether data can be easily collected in natural settings outside the lab. Sensors describes the sensory outputs available from the data collection devices. Learning Extractions specifies which extractions can be utilized with the provided sensors to improve learning. Data Embodiment refers to the embodiment of the collected data (robot vs. human). Here, we also count online corrections [wang2024dexcap] and reinforcement learning [hudor] performed on the robot as part of the robot data. Robot Embodiment indicates which type of robot embodiment the framework targets (two-fingered gripper vs. multi-fingered hand). In Aina, we choose point-based approaches for their robustness to background variations, enabling robot learning from in-the-wild data for dexterous hands. This is made possible by the advanced sensing capabilities of the Aria Gen 2 glasses, which provide all the necessary 3D extractions.*

“The most profound technologies are those that disappear. They weave themselves into the fabric of everyday life until they are indistinguishable from it.”

— Mark Weiser, 1991

Robots autonomously performing diverse manipulation tasks by watching humans go about their daily lives has been a dream in Artificial Intelligence (AI) for decades. However, this remains challenging due to the embodiment gap between humans and robots, as well as the disparity between human video views and the sensor perspectives of a robot. To truly realize this dream for generalizable dexterous manipulation, we must overcome these challenges with general approaches that can leverage large-scale human video data. Encouragingly, this vision is now closer to reality with the development of increasingly more human-like robot embodiments [ability-hand, zorin2025ruka] and the potential widespread adoption of wearable devices such as smart glasses, which are lightweight, easy to wear in daily life and equipped with complex sensing capabilities that provide both an egocentric perspective and rich annotations. Building on this promise, we develop an approach to learn dexterous manipulation directly from smart-glass human data, without requiring any additional robot interaction data.

We are, of course, not the first to consider this setting of learning manipulation from human videos. Prior work has attempted to address these challenges by collecting human videos in structured settings, often within the exact scenarios of robot deployment [wang2023mimicplay, demo-diffusion, point-policy]. However, such approaches are difficult to scale to diverse environments, as they require data collection for each deployment scenario. Other efforts leverage large-scale, in-the-wild web videos [bharadhwaj2023generalizable, vrb, track2act, zeromimic], but they have not been successfully deployed on multi-fingered hands, since extracting the necessary annotations—such as reliable 3D hand poses—for learning dexterous policies is far more challenging in these settings. Smart-glasses data, offers the best of both worlds: it preserves scalability by being naturally collected as people go about daily life, while providing high-resolution egocentric imagery, stereo vision for 3D perception, and reliable hand-pose annotations via in-built software [nimble]. These characteristics make smart-glass data far richer and more robot manipulation-relevant than web video, while avoiding the scalability bottlenecks of lab-constrained data collection.

Thus, leveraging the complex sensing capabilities of smart glasses, in particular of Aria Gen 2, we develop Aina: a simple approach for learning a closed-loop dexterous manipulation policy from just human videos. Aina (english. Mirror) refers to mirroring human videos in a robot’s context and is based on a simple intuition: By lifting human videos to approximate 4D via hand-keypoint reconstruction, stereo depth estimation, and 3D object pointcloud extraction, we can re-purpose 3D policy learning approaches for learning to predict future hand keypoints, and use the same policy for robot manipulation. By operating in the space of 3D keypoints for the hand, and 3D pointclouds for objects, we minimize the human-robot domain gap when deploying the Aina policy on a dexterous robot hand, while being trained with only human demonstrations.

Concretely, Aina operates as follows: (a) humans wearing smart glasses collect data in arbitrary environments with any background or viewpoint, (b) then they collect a single video demonstration in the robot’s environment, and (c) the multi-fingered robot learns policies that generalize across both spatial configurations and object variations. We evaluate Aina on nine tasks and summarize our contributions as follows:

-
1.

Aina is the first framework that learns policies for multi-fingered hands without using any robot data, including no use of simulation (Section [III](#S3)).

-
2.

Aina leverages recent advances in computer vision techniques and smart glasses to accurately track hand and objects in 3D and learn closed-loop policies to transfer them to the robot environment.

In Section [IV](#S4), we show that Aina outperforms existing human-to-robot learning approaches demonstrating the effectiveness of learning manipulation from human videos alone through a simple framework operating on rich sensing from smart glasses. Robot videos are available on our website: [https://aina-robot.github.io](https://aina-robot.github.io).

![Figure](x2.png)

*Figure 2: Illustration of our overall Aina framework. On the left, we show how the data is processed: the human hand pose is extracted directly by the Aria Gen 2 glasses, and stereo depth is estimated from the surrounding SLAM camera frames. This enables the 3D policy learning methods on the right to succeed while remaining robust to background clutter.*

## II Related Works

Aina draws inspiration from extensive research in dexterous manipulation, learning from human videos, and imitation learning. Our aim is to develop a simple framework for closed-loop policy learning capable of performing diverse everyday manipulation tasks with dexterous multi-fingered hands. We highlight some of the comparisons with prior works in Fig. [1](#S1.F1) and describe them below.

##### Robot Learning with Non-Robot Datasets

Since robot interaction data collection is challenging due to operational constraints [arunachalam2023holo, iyer2024open], thanks to advances in representation learning [he2022masked, wu2023unleashing], motion prediction [cotracker, tapir], and hand–object reconstruction [ye2022s, pavlakos2024reconstructing], many approaches now leverage non-robot datasets such as human videos and images. These approaches differ both in the type of human data used—in-domain vs. in-the-wild—and in what is extracted or learned from such data.

In-domain demonstrations, collected in the same environment as deployment, allow rich extractions like 3D hand poses and object points [demo-diffusion, point-policy, hudor, wang2023mimicplay], but require new data per deployment and are thus hard to scale. In contrast, in-the-wild human datasets [ego4d, banerjee2024hot3d, grauman2024ego] support broader generalization, with works focusing on visual backbones [r3m, pvr2, voltron] or high-level cues such as hand-object trajectories [track2act, shaw2023videodex, zeromimic] and affordances [srirama2024hrp, vrb, chen2025web2grasp]. Yet, without reliable low-level signals like 3D hand pose, these methods often sacrifice accuracy or need additional demonstrations during deployment [zeromimic].

More recently, smart glasses [aria-gen-1] have simplified data collection [egomimic, egozero], enabling richer extractions and better generalization which Aina builds upon. However, most of these works focus on two-finger grippers, where manipulation can be modeled simpler. In Aina, we use Aria Gen 2 glasses [aria-gen-2] for scalable human data collection, but uniquely demonstrate policy learning from purely human demonstrations for dexterous multi-fingered robot hands.

##### Dexterous Manipulation from Human Data

Early research on dexterous manipulation relied on sim-to-real transfer [shaw2023leap, akkaya2019solving] or teleoperation for data collection [arunachalam2023holo, yang2024ace, iyer2024open, tavi], but these approaches are limited either by sim-to-real gaps or by the extensive human effort required for teleoperation. To address these challenges and reduce dependence on large-scale robot demonstrations, recent work has shifted toward learning from human data. However, leveraging large-scale datasets is harder for multi-fingered hands due to the lack of annotations needed for extracting reliable signals. As a result, most prior works have collected their own human data to train policies. Some collected in-domain human videos and extracted 3D hand poses [hudor, chen2024arcap], while others gathered in-the-wild demonstrations using portable custom hardware with multiple cameras and hand-pose estimators [wang2024dexcap, dexwild]. While promising, all of these approaches incorporated some robot data, obtained either through teleoperation [dexwild] or online corrections [hudor, wang2024dexcap]. Although such robot data can help in complex dexterous tasks—particularly given the absence of force feedback in human demonstrations—in Aina we demonstrate how we can learn to perform everyday manipulation activities with dexterous multi-fingered hands with just offline human videos captured through Aria glasses, without using any external sensors, mocap markers, or exo-skeletons.

##### Policy Architectures for Imitation Learning

Going beyond the standard of 2D image-based policies [rt1, r3m, cacti] for imitation, recent works have proposed 3D policy architectures that exploit geometric structure for manipulation [shridhar2023perceiver, ze2023gnfactor, gervet2023act3d], yielding improved generalization to cluttered scenes and complex object interactions. Beyond raw pixels and scene point clouds, some approaches incorporate intermediate object-centric representations such as keypoints or tracks. PointPolicy [point-policy] learns manipulation policies from 3D hand and object keypoints, while Track2Act [track2act] predicts future dense object tracks from video datasets and trains track-conditioned policies. These object-centric methods highlight the benefits of embodiment-agnostic cues for bridging human and robot domains. Building on this insight, our proposed approach, Aina extends 3D imitation learning frameworks by extracting hand keypoints and 3D object flow from human videos, enabling policies that generalize across embodiments and leverages (non-robot) human data for dexterous manipulation.

## III Method

### III-A Overview

Aina is a framework for learning closed-loop policies from in-the-wild human demonstrations collected with Aria Gen 2 glasses, without requiring any robot data. Our framework consists of three high-level steps:
(a) a human collects in-the-wild video demonstrations on arbitrary surfaces using the Aria Gen 2 glasses, along with a single in-scene video in the robot’s environment;
(b) the dataset is processed to extract 3D object tracks and hand fingertip points, which are then aligned to establish a uniform reference frame with the robot; and
(c) point-based policies are trained and deployed on a single-arm hand robot system.
We describe the overall structure of our framework in Fig. [2](#S1.F2) and describe the assumptions, challenges, and details in this section.

##### Assumptions

In Aina, our methodology is guided by two key assumptions:
(a) access to a calibrated scene to ensure a uniform operational space. For this, we perform hand–eye calibration to compute the extrinsic matrix of cameras on the robot setup with respect to the robot base. This process is straightforward, performed only once, and takes approximately 5–10 minutes during the initial setup of the robot.
(b) access to a single in-scene demonstration along with multiple in-the-wild demonstrations. Both types of demonstration are collected by humans (without using the robot). The in-scene demonstration takes less than a minute to collect, while the in-the-wild demonstrations take about 10 minutes in total for 50 demos per task.

##### Challenges

Unlike prior works that artificially constrain hand motions to be robot-like [lepert2024shadow] or require additional alignment hardware such as ArUco markers [egozero], our approach considers in-the-wild human videos as natural interactions. Our method does not require prior knowledge of the distance to manipulated objects, and it places no restrictions on the hand motions of the data collectors [point-policy]. These relaxations introduce challenges, as the hand motions are more varied and less structured.

### III-B Collecting and Processing Smart Glass Data

#### III-B1 Data Collection

Aina uses Project Aria Gen 2 [aria-gen-2] glasses to collect in-the-wild human demonstrations. The glasses are equipped with a front-facing RGB camera, four SLAM cameras positioned around the frame, and multiple IMUs. These sensors enable real-time estimation of the user’s head pose as well as left and right hand poses [nimble]. The head pose is defined with respect to a world frame arbitrarily assigned at initialization. This world frame is initialized using the gravity vector measured by the IMUs [aria-gen-1, aria-gen-2], ensuring that its $z$-axis is aligned with gravity. For each task, we collect 50 in-the-wild demonstrations using these glasses and record the camera streams along with head and hand pose estimates at 10 Hz.

During data collection, we do not assume any specific height for the surfaces where humans manipulate the objects. As a result, we need to ground in-the-wild demonstrations within the robot’s scene. To address this, we collect a single in-scene demonstration using the RGB-D cameras in the robot environment. We estimate the hand pose in 2D from both camera views using Hamer [hamer, mano], and then triangulate these estimates to obtain the 3D pose [point-policy, demo-diffusion]. We collect this in-scene demonstration at 10 Hz.

#### III-B2 Processing and Object Tracking

Aina uses object point clouds as observations during policy learning. This representation makes the observations invariant to background changes and visual differences between humans and robots. To obtain these object point clouds, we leverage off-the-shelf computer vision models. For each demonstration, we first segment the objects of interactions in the initial frame using a language prompt with Grounded-SAM [grounded-sam]. The language prompts used for each task are described in the Appendix [-A](#A0.SS1). Next, we track the segmented objects across frames using CoTracker [co-tracker], which produces 2D object points for each demonstration. Finally, given per-frame depth, we unproject these 2D points into 3D, effectively obtaining 3D point object point clouds across time.
While this process is straightforward for in-scene demonstrations, the Aria glasses do not provide depth. Therefore, for in-the-wild demonstrations, we use FoundationStereo [foundation-stereo], a framework that estimates a disparity map from rectified stereo images and the baseline between the cameras. For this, we use streams from the two front-facing SLAM cameras, rectify them, and use the translation norm provided by the Aria glasses as the baseline $B$. These inputs are passed to FoundationStereo to obtain a disparity map $d$ with respect to the left SLAM camera. Using classical stereo geometry, the depth relative to the left frame, $Z$, is then recovered as:

| | $$Z=\frac{f\cdot B}{d},$$ | |
|---|---|---|

where $f$ is the focal length of the left camera. 2D object tracks can then be unprojected to this estimated depth for in-the-wild demonstrations. For consistency, we transform all in-scene points into the robot base frame and all in-the-wild demonstrations into the world frame assigned during data collection. This ensures that all points lie on a similar horizontal plane since Aria glasses use the gravity vector to assign the world frame (explained in Section [III-B1](#S3.SS2.SSS1)).

#### III-B3 Domain Alignment

3D object tracks calculated with respect to the Aria glasses vary across demonstrations in the in-the-wild dataset, particularly due to differences in the height of the manipulated objects or the user collecting the data. To address this issue, we transform all 3D points into the robot base frame before training policies, using the in-scene demonstrations as an anchor.

Each demonstration consists of a trajectory of object points $\mathcal{O}^{t}\in\mathbb{R}^{N\times 3}$ and fingertip points $\mathcal{F}^{t}\in\mathbb{R}^{5\times 3}$, where $N$ is the number of objects, fixed at 500 across all tasks. We refer to in-the-wild trajectories as $\mathcal{T}_{w}=\{\mathcal{O}^{t}_{w},\mathcal{F}^{t}_{w}\}$ and in-scene trajectories as $\mathcal{T}_{s}=\{\mathcal{O}^{t}_{s},\mathcal{F}^{t}_{s}\}$.
To transform these trajectories into a uniform space, given a single in-scene trajectory $\mathcal{T}_{s}$ and an in-the-wild trajectory $\mathcal{T}_{w}$, we compute the translation between the centroids of the object points in their first frames,
$\Delta\mathcal{O}=\mathcal{O}^{0}_{s}-\mathcal{O}^{0}_{w}$.
We then translate the in-the-wild trajectory by this offset, yielding
$\hat{\mathcal{T}_{w}}=\{\mathcal{O}^{t}+\Delta\mathcal{O},\mathcal{F}^{t}+\Delta\mathcal{O}\}$.
This aligns the centroids of the object point clouds. However, since the world frame’s rotation around gravity is assigned arbitrarily, relying solely on this translation can lead to large variations in $z$-axis orientation. This may cause demonstrations where the initial hand pose is fully rotated and object positions appear swapped. Figure illustrating this issue can be found on [https://aina-robot.github.io](https://aina-robot.github.io).

To estimate a reliable rotation around the $z$-axis, we use the initial hand poses of both trajectories, $\mathcal{F}^{0}_{s}$ and $\mathcal{F}^{0}_{w}$, and apply the Kabsch algorithm [kabsch-algorithm] to compute the rigid transform between them. From this transform, we extract the rotation around the $z$-axis $R_{z}$, and apply it to the in-the-wild demonstrations, yielding the final transformed trajectories:

$$
\begin{aligned}
\hat{\mathcal{O}}^{t}_{w} &=R_{z}\cdot\mathcal{O}^{t}_{w}+\Delta\mathcal{O} \tag{1}
\end{aligned}
$$
$$
\begin{aligned}
\hat{\mathcal{F}}^{t}_{w} &=R_{z}\cdot\mathcal{F}^{t}_{w}+\Delta\mathcal{O} \tag{2}
\end{aligned}
$$
$$
\begin{aligned}
\hat{T}_{w} &=\{\hat{\mathcal{O}}^{t}_{w},\hat{\mathcal{F}}^{t}_{w}\} \tag{3}
\end{aligned}
$$

We apply this transformation to every in-the-wild demonstration, and both in-the-wild and in-scene demonstrations are then used for policy learning, as described in the next section.

### III-C Learning and Deploying Smart Glass Policies on Robots

#### III-C1 Policy Learning

To handle visual differences between the robot environment and in the wild human demonstrations, Aina utilizes transformer-based point-cloud policies and builds on top of the state-of-the-art imitation learning algorithm Point-Policy [point-policy]. We provide the policy with a trajectory of fingertips $\mathcal{F}^{t-T_{o}:t}$ and object points $\mathcal{O}^{t-T_{o}:t}$ as input, and train the model to predict the subsequent fingertip trajectory $\mathcal{F}^{t:t+T_{p}}$, where $T_{o}=10$ and $T_{p}=30$ denote the observation history and prediction horizon, respectively.
In our architecture, the observation history for each point is encoded into a single vector using Vector Neuron Multilayer Perceptrons (MLPs)[deng2021vector]. These differ from regular MLPs in two key ways: (1) points are represented with 3D perceptrons rather than 1D, and (2) they employ SO(3)-equivariant activation layers. We choose vector neuron MLPs due to their demonstrated ability to better capture 3D geometric information[deng2021vector].
The flattened vectors are then passed into a transformer encoder as tokens. Positional encoding is learned for only fingertip tokens and not keypoint tokens. The representations output by this encoder are subsequently fed into an MLP to predict the future fingertip trajectory. Mathematically, this can be expressed as follows:

$$\hat{\mathcal{F}}^{t:t+T_{p}}=\pi\bigl(\mathcal{F}^{t-T_{o}:t},\mathcal{O}^{t-T_{o}:t}\bigr).$$ \tag{4}

The entire system is trained end-to-end in a supervised manner using the mean squared error between the predicted and the ground-truth fingertips:

$$\mathcal{L}_{\text{MSE}}=\mathbb{E}\left[\bigl(\mathcal{F}^{t:t+T_{p}}-\hat{\mathcal{F}}^{t:t+T_{p}}\bigr)^{2}\right].$$ \tag{5}

In order to improve generalization, we apply augmentations during training. For each datapoint, we uniformly sample a 3D translation in the range $[-30\text{cm},30\text{cm}]$, a scaling factor in the range $[0.8,1.2]$, and a rotation between $[-60^{\circ},60^{\circ}]$ around the gravity axis. These augmentations are combined into a single transformation, which is then applied consistently to both the input to the model and the ground truth output used to calculate $L_{\text{MSE}}$. Finally, to prevent the model from overfitting to the fingertips, we add Gaussian noise in the range $[-2\text{cm},2\text{cm}]$ to the input fingertips, but not to the predicted actions. We train the model for 2000 epochs, which typically takes about 2 hours per task. Our architecture is visualized in Fig. [2](#S1.F2).

#### III-C2 Human Policy $\rightarrow$ Robot Deployment

![Figure](x3.png)

*Figure 3: Illustration of our robot setup.*

![Figure](x4.png)

*Figure 4: Robot rollouts of Aina across nine tasks. Spatial generalization is shown in the leftmost column for each task. The meaning of each symbol is explained below the figure. Dotted lines indicate the object’s orientation; when not shown, the orientation remains the same as in the showcased rollout. For the Oven Opening task, we showcase Aina’s performance when there is background disturbance.*

##### Robot Setup

Our robot setup consists of a single Kinova Gen3 robot arm [kinova-gen3] with 7 degrees of freedom (DOF) and a Psyonic Ability Hand with five fingers [ability-hand]. The Ability Hand has six DOFs: one in each finger and two in the thumb. It is designed as a prosthetic hand, making it compact and similar in size to a human hand. To observe the robot’s environment, we use two RealSense RGB-D cameras placed around the operation space. Our robot configuration is illustrated in Fig. [3](#S3.F3).

##### Inverse Kinematics

The kinematics of human arms and hands differ from those of tabletop manipulators and robot hands, making it non-trivial to replay human trajectories on a robot. Although the Ability Hand’s small size reduces this embodiment gap, the lack of wrist joints in tabletop manipulators means that naively moving the arm and hand separately often leads to infeasible configurations. To address this, we implemented a custom full arm–hand inverse kinematics (IK) module $\mathcal{I}$, similar to [hudor]. Given desired fingertips $\mathcal{F}^{t+1}\in\mathbb{R}^{5\times 3}$ and current Kinova and Ability joints $\mathcal{J}^{t}\in\mathbb{R}^{13}$, the module outputs next joint angles $\mathcal{J}^{t+1}=\mathcal{I}(\mathcal{F}^{t+1},\mathcal{J}^{t})$. The policy predicts fingertips as actions, and the resulting joint angles are applied to the robot during deployment. As in training, we segment and track objects in 3D to obtain object points and use forward kinematics to compute the fingertips.

##### Practical Implementation Details

Since the human demonstrations do not include force information, for tasks involving grasping, we set a grasping threshold: if the distance between the predicted thumb and any other finger position is less than 5 cm, the fingers are moved closer together. This helps mimic the force that humans apply during grasps.

## IV Experimental Evaluation

![Figure](x5.png)

*Figure 5: Visualization of in-the-wild human demonstrations collected for different tasks. These are collected with natural human motions and with the right hand performing the respective tasks (no additional sensors on the humans or the environments, except Aria glasses).*

We perform various real robot experiments and compare Aina against multiple baselines to answer the following questions:

-
1.

How important are the different types of data used in Aina?

-
2.

How does Aina compare to image-based approaches for learning from human data?

-
3.

How well does Aina perform when the height of the operation space changes?

-
4.

How well does Aina generalize spatially and across different objects?

### IV-A Task Descriptions

We evaluate Aina on nine tasks, each chosen to represent a distinct skill or motion modality (wiping, pick-place, reorientation) and to reflect common daily manipulation activities. Robot rollouts, success rates, and spatial generalization results for each task are shown in Fig. [4](#S3.F4). Human demonstrations used to train different tasks are shown in Fig. [5](#S4.F5). We describe each task in detail in the Appendix [-A](#A0.SS1).

### IV-B How important are the different types of data used in Aina?

Aina is a new framework for learning robot policies by co-training on in-the-wild and in-scene human video demonstrations. In-scene demonstrations are used both to standardize the input observations and to improve the policy. In this section, we evaluate the importance of this recipe.

We compare Aina against the following baselines and present the results in Table [I](#S4.T1):

-
1.

In-Scene Only [hudor]: A policy trained using only a single in-scene demonstration. Unlike HuDOR [hudor], we do not apply any reinforcement learning for this baseline.

-
2.

In-The-Wild Only [egozero]: A policy trained solely on in-the-wild demonstrations. These demonstrations are recorded with respect to the initial frame of the RGB camera and then transformed into the robot space by measuring the distance from the left camera to the center of the operation space and shifting the points accordingly. The closest approach to this is EgoZero [egozero], but our baseline differs in two key ways: (a) we do not use ArUco markers for data transfer, and (b) we perform closed-loop tracking of all the object points.

-
3.

In-Scene Transform and In-The-Wild [zeromimic]: A policy that does not use in-scene data during training, but uses the in-scene demonstration for transforming the in-the-wild demonstrations. This baseline is inspired by ZeroMimic [zeromimic] that trains policies with in-the-wild human videos and uses a single in-scene goal image to condition the framework.

-
4.

In-Scene Training and In-The-Wild: A policy that does not use in-scene data for transformation but includes it during training. The transformation is done as described for the In-The-Wild Only baseline.

*Table I: Comparison of success of Aina to policies trained with different datasets. All methods are evaluated in similar deployment scenarios, with minimum of 10 trials each.*

From these results, we make the following observations:

In-the-wild demonstrations improve spatial generalization. The In-Scene Only baseline succeeds when objects are placed close to the demonstrated position, but it fails to generalize beyond that location.

In-scene demonstrations improve training. Since deployment is performed using RGB-D cameras rather than Aria glasses, the actions predicted by the In-The-Wild Only and In-Scene Transform baselines appear highly misaligned, leading to behaviors that look out of distribution.

In-scene demonstrations help transform in-the-wild demonstrations. The in-the-wild data used here is collected on different surfaces, with varying heights and different initial head frames. This makes the transformation in the In-The-Wild Only baseline prone to unstable rotations of the object points, resulting in less accurate policies.

### IV-C How does Aina compare to image-based approaches for learning from human data?

![Figure](x6.png)

*Figure 6: Illustration of BAKU [haldar2024baku] used in the RGB-based baselines. The fingertip encoders are multilayer perceptrons, and the image encoders are ResNet-18 [resnet] models pretrained on the ImageNet classification task. The action token is set to zeros.*

Aina uses object-centric point clouds as input to reduce the visual disparity between human and robot observations. Using point clouds and the alignment module in Aina also improves robustness to viewpoint differences between in-the-wild demonstrations and robot deployment scenarios. To evaluate the impact of using point clouds, we compare Aina to two image-based architectures on two of our tasks, with the results shown in Table [II](#S4.T2). We implement the following baselines:

-
1.

Masked BAKU: We segment objects using the same approach as in Aina and track masks across trajectories using Cutie [cutie]. We then apply BAKU [haldar2024baku], a transformer-based imitation learning architecture, using the masked RGB image of the objects along with the history of fingertip positions. A visualization of this architecture is shown in Fig. [6](#S4.F6). In this baseline, we provide fingertip history as input, but only a single RGB frame.

-
2.

Masked BAKU with History: This version uses the same architecture as Masked BAKU but includes a history of RGB images instead of a single frame.

*Table II: Comparison of success of Aina to policies trained with RGB images as input. All methods are evaluated in similar deployment scenarios, with 15 trials.*

Both baselines are trained on the same dataset as Aina. We observe that Aina outperforms these image-based baselines on both tasks. Within the in-the-wild demonstrations, the human head naturally moves, whereas the robot’s camera remains fixed during deployment. This discrepancy causes the Masked BAKU with History inputs to fall out of distribution relative to the training data, causing the policies to perform extremely poorly. Masked BAKU performs better, succeeding in nearly half of the trials; however, we still observe that viewpoint disparity between human demonstrations and robot deployment negatively affects performance. These demonstrate the importance of ingesting 3D inputs, and point tracks instead of images, for effective human-to-robot transfer.

### IV-D How does Aina perform when the height of the operation space changes?

![Figure](x7.png)

*Figure 7: Illustration of the height experiments. Each yellow plate is 3.5 cm tall. Height 1 consists of 2 plates, Height 2 of 4 plates, and Height 3 of 5 plates. Thus, Height 1 is closest to the original deployment scenario, while Height 3 is the furthest.*

Aina does not assume prior knowledge about the height of the manipulated object, the data collector, or the robot’s operation space. To demonstrate its use in operation spaces with different heights, we placed 3.5 cm tall plates on top of the robot’s desk to create three height levels, as illustrated in Fig. [7](#S4.F7). For each height level, we collect an additional in-scene human demonstration for alignment (requiring less than a minute to collect), as described in Section [III-B1](#S3.SS2.SSS1) and use the same human data originally collected in-the-wild. We show the results in Table [III](#S4.T3).

*Table III: Success rate of Aina deployed on plates with different height levels.*

We find that the resulting policies perform robustly across tasks, reliably generalizing across heights. This demonstrates the flexibility of Aina in transferring manipulation skills from in-the-wild data to new scenarios with minimal human effort. Occasional failures arise when an in-scene human demonstration trajectory diverges significantly from the distribution of in-the-wild data. For example, in the Toy Picking task at Height 3, the in-scene demonstration brought the toy unusually close to the bowl. This atypical trajectory led the policy to reproduce the behavior during deployment, causing the toy to push the bowl.

### IV-E How does Aina generalize to different objects?

![Figure](x8.png)

*Figure 8: Generalization experiments on Toy Picking, Toaster Press and Wiping tasks. Language prompts used to track the objects are showcased next to each object.*

We evaluate the generalization of Aina by testing policies on novel objects across three tasks. Here, we do not train any new policies; instead, we deploy existing ones zero-shot in environments with new objects while prompting GroundedSAM with new task keywords. The success rates and corresponding text prompts are shown in Fig. [8](#S4.F8). We observe that for objects with similar shapes, such as the new toaster or the white eraser, Aina generalizes well. However, when the shape and weight of the objects differ significantly—such as a popcorn package compared to the toy or a board eraser compared to the sponge—Aina struggles to generalize.

## V Discussion, Limitations, Conclusion

In this work, we presented Aina, a new framework that leverages capabilities of Aria Gen 2 glasses to learn point-based multi-fingered policies from explicitly in-the-wild human demonstrations.

While promising, we observe three limitations. First, our framework cannot easily integrate force feedback, since hand pose estimation alone cannot capture this information, which is often crucial for accurate dexterous manipulation [sparsh-x, sharma2025sparshskin, tdex]. This could be addressed by integrating other wearables, such as EMG sensors or force-estimating gloves. Second, the Aria Gen 2 glasses exhibit a slight difference in shutter timing between the RGB and SLAM cameras. Rapid head movements during data collection can therefore cause misalignment between the object’s pixels in the RGB image and the corresponding depth in SLAM. To mitigate this, we currently instruct data collectors to avoid rapid head movements, though alternative solutions include using more robust 3D object tracking algorithms [stereo4d] or fitting and tracking a mesh representation of the object [foundationpose]. Finally, during deployment we currently use Realsense cameras, which causes the keypoints collected with Aria glasses to differ slightly from those observed at deployment. The reason we are not yet streaming Aria input is the difficulty of obtaining real-time depth estimates with FoundationStereo. However, this is an ongoing effort and we believe that with sufficient optimizations, we can receive near real-time depth.

## Acknowledgements

We thank our amazing colleagues at Meta FAIR and Reality Labs for helpful discussions.

### -A Task Descriptions

In this section, we describe each task in detail.

##### Toaster Press

The robot must locate and push down the lever of a bread toaster. The toaster is positioned within a $30,\text{cm}\times 50,\text{cm}$ area. The text prompt used is toaster.

##### Toy Picking

The robot must locate and pick up a soft pink toy, then drop it into a bowl. The toy is positioned within a $30,\text{cm}\times 30,\text{cm}$ area, while the bowl remains fixed. The text prompts used are bowl and pink toy.

##### Oven Opening

The robot must locate a toaster oven and open its door by pulling its lever. The oven is positioned within a $50,\text{cm}\times 30,\text{cm}$ area. The text prompt used is toaster oven.

##### Drawer Opening

The robot must locate a white storage drawer and slide it open. The drawer is positioned within a $50,\text{cm}\times 30,\text{cm}$ area. The text prompt used is white box.

##### Wiping

The robot must locate a sponge and wipe the board. The sponge is positioned within a $30,\text{cm}\times 30,\text{cm}$ area. The demonstrations do not specify where to wipe; wiping motions are collected arbitrarily. Success is therefore defined by whether the robot achieves a stable grasp of the sponge and wipes some portion of the board. The text prompt used is sponge.

##### Planar Reorientation

The robot must locate a banana, reorient it in place, and pick it up. The banana is positioned within a $30,\text{cm}\times 30,\text{cm}$ area. The text prompt used is banana.

##### Cup Pouring

The robot must locate a red cup, pick it up, and pour its contents into a bowl. The cup is positioned within a $30,\text{cm}\times 30,\text{cm}$ area, while the bowl remains fixed. The text prompts used are red cup and bowl.

##### Stowing

The robot must locate a bowl, pick it up, place it inside a toaster oven, and close the oven door. This is a long-horizon task involving multiple skills: picking up a rigid bowl, placing it in a spatially constrained location, and closing the oven door. The bowl is positioned within a $20,\text{cm}\times 20,\text{cm}$ area, while the oven remains fixed. The text prompts used are toaster oven and bowl.

##### Knob Rotating

The robot must locate the temperature knob of a toaster oven and rotate it 90 degrees. The toaster oven is positioned within a $20,\text{cm}\times 20,\text{cm}$ area. The text prompt used is toaster oven.

Generated on Thu Nov 20 18:57:30 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)