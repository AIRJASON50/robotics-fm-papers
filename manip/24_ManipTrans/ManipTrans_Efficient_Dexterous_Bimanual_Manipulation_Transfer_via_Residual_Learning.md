[# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning Kailin Li1 Puhao Li1,2 Tengyu Liu1 Yuyang Li1,3 Siyuan Huang1 1State Key Laboratory of General Artificial Intelligence, BIGAI 2Department of Automation, Tsinghua University 3Institute for Artificial Intelligence, Peking University https://maniptrans.github.io](https://maniptrans.github.io)

###### Abstract

Human hands play a central role in interacting, motivating increasing research in dexterous robotic manipulation. Data-driven embodied AI algorithms demand precise, large-scale, human-like manipulation sequences, which are challenging to obtain with conventional reinforcement learning or real-world teleoperation. To address this, we introduce ManipTrans, a novel two-stage method for efficiently transferring human bimanual skills to dexterous robotic hands in simulation. ManipTrans first pre-trains a generalist trajectory imitator to mimic hand motion, then fine-tunes a specific residual module under interaction constraints, enabling efficient learning and accurate execution of complex bimanual tasks.
Experiments show that ManipTrans surpasses state-of-the-art methods in success rate, fidelity, and efficiency. Leveraging ManipTrans, we transfer multiple hand-object datasets to robotic hands, creating DexManipNet, a large-scale dataset featuring previously unexplored tasks like pen capping and bottle unscrewing. DexManipNet comprises 3.3K episodes of robotic manipulation and is easily extensible, facilitating further policy training for dexterous hands and enabling real-world deployments.

*

*Figure 1: ManipTrans for Bimanual Dexterous Manipulations. Retargeting methods often struggle with transferring MoCap data to physically plausible motions, while our ManipTrans efficiently produces task-compliant, physically accurate motions. It also generalizes across embodiments like Inspire hands [[[3](#bib.bib3)]], Shadow hands [[[1](#bib.bib1)]], and articulated MANO hands [[[96](#bib.bib96), [27](#bib.bib27)]].*

## 1 Introduction

Embodied AI (EAI) has advanced rapidly in recent years, with increasing efforts to enable AI-driven embodiments to interact with physical or virtual environments. Just as human hands are pivotal for interaction, much research in EAI focuses on dexterous robotic hand manipulation [[[4](#bib.bib4), [20](#bib.bib20), [113](#bib.bib113), [75](#bib.bib75), [81](#bib.bib81), [131](#bib.bib131), [16](#bib.bib16), [18](#bib.bib18), [17](#bib.bib17), [66](#bib.bib66), [59](#bib.bib59), [70](#bib.bib70), [104](#bib.bib104), [21](#bib.bib21), [46](#bib.bib46), [19](#bib.bib19), [77](#bib.bib77), [118](#bib.bib118), [65](#bib.bib65), [63](#bib.bib63), [115](#bib.bib115), [72](#bib.bib72), [22](#bib.bib22), [130](#bib.bib130), [41](#bib.bib41), [68](#bib.bib68), [58](#bib.bib58), [82](#bib.bib82), [52](#bib.bib52)]]. Achieving human-like proficiency in complex bimanual tasks holds significant research value and is crucial for progress toward general AI.

Thus, the rapid acquisition of precise, large-scale, and human-like dexterous manipulation sequences for data-driven embodied agents training [[[11](#bib.bib11), [12](#bib.bib12), [25](#bib.bib25), [83](#bib.bib83), [133](#bib.bib133)]] becomes increasingly urgent. Some studies use reinforcement learning (RL) [[[54](#bib.bib54), [99](#bib.bib99)]] to explore and generate dexterous hand actions [[[121](#bib.bib121), [111](#bib.bib111), [135](#bib.bib135), [77](#bib.bib77), [27](#bib.bib27), [136](#bib.bib136), [69](#bib.bib69)]], while others collect human-robot paired data through teleoperation [[[45](#bib.bib45), [26](#bib.bib26), [113](#bib.bib113), [128](#bib.bib128), [44](#bib.bib44), [82](#bib.bib82), [103](#bib.bib103)]]. Both methods are limited: traditional RL requires carefully designed, task-specific reward functions [[[135](#bib.bib135), [78](#bib.bib78)]], restricting scalability and task complexity, while teleoperation is labor-intensive and costly, yielding only embodiment-specific datasets.

A promising solution is to transfer human manipulation actions to dexterous robotic hands in simulated environments via imitation learning [[[80](#bib.bib80), [112](#bib.bib112), [139](#bib.bib139), [71](#bib.bib71), [93](#bib.bib93)]]. This approach offers several advantages. First, imitating human manipulation trajectories creates naturalistic hand-object interactions, enabling more fluid and human-like motions. Second, abundant motion-capture (MoCap) datasets [[[37](#bib.bib37), [10](#bib.bib10), [107](#bib.bib107), [39](#bib.bib39), [14](#bib.bib14), [125](#bib.bib125), [73](#bib.bib73), [57](#bib.bib57), [119](#bib.bib119), [32](#bib.bib32), [74](#bib.bib74), [62](#bib.bib62), [134](#bib.bib134)]] and hand pose estimation techniques [[[13](#bib.bib13), [108](#bib.bib108), [43](#bib.bib43), [67](#bib.bib67), [123](#bib.bib123), [124](#bib.bib124), [120](#bib.bib120), [122](#bib.bib122), [126](#bib.bib126), [87](#bib.bib87)]] makes extracting operational knowledge from human demonstrations easily accessible [[[93](#bib.bib93), [102](#bib.bib102)]]. Third, simulations provide a cost-effective validation, offering a shortcut to real-world robot deployment [[[51](#bib.bib51), [44](#bib.bib44), [41](#bib.bib41)]].

Yet, achieving precise and efficient transfer is non-trivial. As shown in [Fig. 1](#S0.F1), morphological differences between human and robotic hands lead to direct pose retargeting suboptimal. Additionally, although MoCap data is relatively accurate, error accumulation can still lead to critical failures during high-precision tasks. Moreover, bimanual manipulation introduces a high-dimensional action space, significantly increasing the difficulty of efficient policy learning. Consequently, most pioneering work generally stops at single-hand grasping and lifting tasks [[[121](#bib.bib121), [111](#bib.bib111), [135](#bib.bib135), [27](#bib.bib27)]], leaving complex bimanual activities—such as unscrewing a bottle or capping a pen—largely unexplored.

In this paper, we propose a simple but efficient method, ManipTrans, which facilitates the transfer of hand manipulation skills—especially bimanual actions—to dexterous robotic hands in simulation, enabling accurate tracking of reference motions. Our key insight is to treat the transfer as a two-stage process: a pre-training trajectory imitation stage focusing on hand motion alone, followed by a specific action fine-tuning stage that meets interaction constraints. Specifically, we design a robust generalist model that learns to accurately mimic human finger motions with resilience to noise. Based on this initial imitation, we then introduce a residual learning module [[[53](#bib.bib53), [106](#bib.bib106), [48](#bib.bib48), [51](#bib.bib51)]] that incrementally refines the robot’s actions, focusing on two key aspects: 1) ensuring stable contact with object surfaces under physical constraints, enabling effective object manipulation, and 2) coordinating both hands to ensure precise, high-fidelity execution of complex bimanual operations.

The advantages of this design are threefold: 1) In the first stage, focusing on dynamic hand mimicry with large-scale pretraining effectively mitigates morphological differences. 2) Building on this advantage, the second stage concentrates on tracking bimanual object interactions, enabling precise capture of subtle movements and facilitating natural, high-fidelity manipulation. 3) It significantly reduces action space complexity by decoupling human hand motion imitation from physics-based object interaction constraints, thus improving training efficiency.

Building on this framework, ManipTrans corrects arbitrary, noisy hand MoCap data into physically plausible motion without predefined stages (e.g*., “approaching-grasping-manipulation”) or task-specific reward engineering. We, therefore, validate its effectiveness and efficiency across a range of complex single- and bimanual manipulations, including articulated object handling [[[62](#bib.bib62), [134](#bib.bib134), [107](#bib.bib107), [34](#bib.bib34), [32](#bib.bib32)]]. Using ManipTrans, we transfer several representative hand-object manipulation datasets [[[62](#bib.bib62), [134](#bib.bib134)]] to dexterous robotic hands in the Isaac Gym simulation [[[79](#bib.bib79)]], constructing the DexManipNet dataset, which achieves marked improvements in motion fidelity and compliance. Currently, DexManipNet comprises 3.3K episodes and 1.34 million frames of robotic hand manipulation, covering previously unexplored tasks such as pen capping, bottle cap unscrewing, and chemical experimentation.

We experimentally demonstrate that ManipTrans outperforms baseline methods in both motion precision and transfer success rate. Notably, it surpasses prior state-of-the-art (SOTA) approaches in transfer efficiency, even on a personal computer. To evaluate its extensibility, we conducted cross-embodiment experiments applying ManipTrans to dexterous hands with varying degrees of freedom (DoFs) and morphologies, achieving consistent performance with minimal additional effort.
Furthermore, we replay DexManipNet’s bimanual trajectories on real-world devices, demonstrating agile and natural dexterous manipulation that, to the best of our knowledge, has not been achieved by previous RL- or teleoperation-based methods. Finally, we benchmark DexManipNet using several imitation learning frameworks, underscoring its value to the research community.

In summary, our contributions are as follows:

-
•

We introduce ManipTrans, a simple yet effective two-stage transfer framework that enables precise transfer of human bimanual manipulation to dexterous robotic hands in simulation, ensuring accurate tracking of both hand and object reference motions.

-
•

Using this framework, we construct DexManipNet, a large-scale, high-quality dataset featuring a wide array of novel bimanual manipulation tasks with high precision and compliance. DexManipNet is extensible and serves as a valuable resource for future policy training.

-
•

Our experiments show that ManipTrans outperforms previous SOTA methods. We further demonstrate its generalizability across various dexterous hand configurations and its feasibility for real-world deployment.

## 2 Related Works

Dexterous Manipulation via Human Demonstration
Learning manipulation skills from human demonstrations offers an intuitive and effective approach to transferring human abilities to robots [[[6](#bib.bib6), [31](#bib.bib31), [132](#bib.bib132), [129](#bib.bib129)]]. Imitation learning has shown considerable promise in achieving this transfer [[[80](#bib.bib80), [112](#bib.bib112), [139](#bib.bib139), [71](#bib.bib71), [109](#bib.bib109), [89](#bib.bib89), [90](#bib.bib90), [142](#bib.bib142), [64](#bib.bib64), [7](#bib.bib7), [23](#bib.bib23)]].
Recent studies focus on learning RL policies guided by object trajectories [[[22](#bib.bib22), [77](#bib.bib77), [142](#bib.bib142), [21](#bib.bib21), [72](#bib.bib72)]]. QuasiSim [[[72](#bib.bib72)]] advances this approach by directly transferring reference hand motions to robotic hands via parameterized quasi-physical simulators. However, these methods are limited to simpler tasks and are computationally intensive. More recently, tailored solutions using task-specific reward functions have been developed for challenging tasks like bimanual lip-twisting [[[68](#bib.bib68), [70](#bib.bib70)]]. In contrast, our method enables efficient learning of complex manipulation tasks without task-specific reward engineering.

Dexterous Hand Datasets
Object manipulation is fundamental for embodied agents. Numerous MANO-based [[[96](#bib.bib96)]] hand-object interaction datasets exist [[[37](#bib.bib37), [9](#bib.bib9), [10](#bib.bib10), [107](#bib.bib107), [39](#bib.bib39), [40](#bib.bib40), [100](#bib.bib100), [14](#bib.bib14), [93](#bib.bib93), [125](#bib.bib125), [73](#bib.bib73), [57](#bib.bib57), [119](#bib.bib119), [32](#bib.bib32), [143](#bib.bib143), [74](#bib.bib74), [62](#bib.bib62), [55](#bib.bib55), [134](#bib.bib134), [141](#bib.bib141), [42](#bib.bib42), [36](#bib.bib36), [60](#bib.bib60), [28](#bib.bib28), [61](#bib.bib61)]]. However, these datasets often prioritize pose alignment with 2D images while neglecting physical constraints, limiting their applicability for robotic training.
Teleoperation methods [[[45](#bib.bib45), [26](#bib.bib26), [113](#bib.bib113), [128](#bib.bib128), [44](#bib.bib44), [140](#bib.bib140), [117](#bib.bib117), [92](#bib.bib92)]] collect human-to-robot hand matching data online using AR/VR systems [[[24](#bib.bib24), [30](#bib.bib30), [15](#bib.bib15), [52](#bib.bib52), [86](#bib.bib86)]] or vision-based MoCap [[[113](#bib.bib113), [94](#bib.bib94), [114](#bib.bib114)]] for real-time data acquisition and correction with humans in the loop. However, teleoperation is labor-intensive and time-consuming, and the absence of tactile feedback often yields stiff, unnatural actions, hindering fine-grained manipulation.
In contrast, our method enables offline transfer of human demonstrations to robots. Our DexManipNet offers a large, easily expandable collection of human demonstration episodes.

Residual Learning
Due to the sample inefficiency and time-consuming nature of RL training, residual policy learning [[[53](#bib.bib53), [106](#bib.bib106), [98](#bib.bib98)]], which incrementally refines action control, is widely adopted to enhance efficiency and stability. In dexterous hand manipulation, various studies explore residual strategies tailored to specific tasks [[[5](#bib.bib5), [29](#bib.bib29), [98](#bib.bib98), [38](#bib.bib38), [138](#bib.bib138), [118](#bib.bib118), [21](#bib.bib21), [139](#bib.bib139)]]. For instance, [[[38](#bib.bib38)]] integrates user input during residual policy training, while [[[51](#bib.bib51)]] learns corrective actions from human demonstrations. GraspGF [[[118](#bib.bib118)]] employs a pre-trained score-based generative model as a base, and [[[21](#bib.bib21)]] decomposes the imitation task into wrist following and finger motion control, integrating a residual wrist control policy. Additionally, [[[48](#bib.bib48)]] constructs a mixture-of-experts system [[[49](#bib.bib49)]] using residual learning, and DexH2R [[[139](#bib.bib139)]] applies residual learning directly to retargeted robotic hand actions.
Our method differs from these approaches by pre-training a finger motion imitation model that incorporates additional dynamic information, followed by fine-tuning a residual policy to adapt to task-specific physical constraints. This approach is more efficient and generalizable across various manipulation tasks.

## 3 Method

*Figure 2: Our ManipTrans Pipeline. We first pre-train a hand motion imitation model with large-scale human demonstrations, then fine-tune a residual policy to adapt to task-specific physical constraints.*

We provide an overview of our method in [Fig. 2](#S3.F2). Given reference human hand–object interaction trajectories, our goal is to learn a policy that enables dexterous robotic hands to accurately replicate these trajectories in simulation while satisfying the task’s semantic manipulation constraints. To this end, we propose a two-stage framework: the first stage trains a general hand trajectory imitation model, and the second stage employs a residual model to refine the initial coarse motion into task-compliant actions.

### 3.1 Preliminaries

Without loss of generality, we formulate the manipulation transfer problem in a complex bimanual setting, where the left and right dexterous hands, $\boldsymbol{d}=\{d_{l},d_{r}\}$, aim to replicate the behavior of human hands, $\boldsymbol{h}=\{h_{l},h_{r}\}$, which interact with two objects, $\boldsymbol{o}=\{o_{l},o_{r}\}$, in a cooperative manner (e.g., in a pen-capping task where one hand holds the cap while the other grips the pen body). The reference trajectories from human demonstrations are defined as $\boldsymbol{\mathcal{T}}_{\boldsymbol{h}}=\{\boldsymbol{\tau}_{\boldsymbol{h}}%
^{t}\}_{t=1}^{T}$ and $\boldsymbol{\mathcal{T}}_{\boldsymbol{o}}=\{\boldsymbol{\tau}_{\boldsymbol{o}}%
^{t}\}_{t=1}^{T}$, where $T$ represents the total number of frames.
The trajectory $\boldsymbol{\tau}_{\boldsymbol{h}}$ for each hand includes the wrist’s 6-DoF pose $\boldsymbol{w}_{\boldsymbol{h}}\in\mathbb{SE}(3)$, the linear and angular velocities $\dot{\boldsymbol{w}}_{\boldsymbol{h}}=\{\boldsymbol{v}_{\boldsymbol{h}},%
\boldsymbol{u}_{\boldsymbol{h}}\}$, and the finger joint positions $\boldsymbol{j}_{\boldsymbol{h}}\in\mathbb{R}^{F\times 3}$ defined by MANO [[[96](#bib.bib96)]], along with their respective velocities $\dot{\boldsymbol{j}}_{\boldsymbol{h}}=\{\boldsymbol{v}_{\boldsymbol{j}},%
\boldsymbol{u}_{\boldsymbol{j}}\}$; here, $F$ denotes the number of hand keypoints, including the fingertips. Similarly, the object trajectory $\boldsymbol{\tau}_{\boldsymbol{o}}$ for each object includes its 6-DoF pose $\boldsymbol{p}_{\boldsymbol{o}}\in\mathbb{SE}(3)$ and the corresponding linear and angular velocities $\dot{\boldsymbol{p}}_{\boldsymbol{o}}=\{\boldsymbol{v}_{\boldsymbol{o}},%
\boldsymbol{u}_{\boldsymbol{o}}\}$. To reduce spatial complexity, we normalize all translations relative to the dexterous hand’s wrist position while preserving the original rotations to maintain the correct gravity direction.

We model this problem as an implicit Markov Decision Process (MDP) $\mathcal{M}=\langle\boldsymbol{\mathcal{S}},\boldsymbol{\mathcal{A}},%
\boldsymbol{\mathsf{T}},\boldsymbol{\mathsf{R}},\gamma\rangle$, where $\boldsymbol{\mathcal{S}}$ represents the state space, $\boldsymbol{\mathcal{A}}$ the action space, $\boldsymbol{\mathsf{T}}$ the transition dynamics, $\boldsymbol{\mathsf{R}}$ the reward function, and $\gamma$ the discount factor. The action for each dexterous hand at time $t$, denoted as $\boldsymbol{a}^{t}\in\boldsymbol{\mathcal{A}}$, comprises the target positions of each dexterous hand’s joint $\boldsymbol{a}_{\boldsymbol{q}}^{t}\in\mathbb{R}^{K}$ for proportional-derivative (PD) control, and the 6-DoF force $\boldsymbol{a}_{\boldsymbol{w}}^{t}\in\mathbb{R}^{6}$ applied to the robotic wrist, similar to prior work [[[121](#bib.bib121), [111](#bib.bib111), [48](#bib.bib48)]], where $K$ denotes the total number of robotic hand revolute joints (i.e*. the DoF).

Our approach divides the transfer process into two stages: 1) a pre-trained hand-only trajectory imitation model $\boldsymbol{\mathcal{I}}$, and 2) a residual module $\boldsymbol{\mathcal{R}}$ that fine-tunes the coarse actions to ensure task compliance. The state at time $t$ is defined separately for each stage as $\boldsymbol{s}^{t}_{\boldsymbol{\mathcal{I}}}\in\boldsymbol{\mathcal{S}}_{%
\boldsymbol{\mathcal{I}}}$ and $\boldsymbol{s}^{t}_{\boldsymbol{\mathcal{R}}}\in\boldsymbol{\mathcal{S}}_{%
\boldsymbol{\mathcal{R}}}$, with corresponding reward functions $r^{t}_{\boldsymbol{\mathcal{I}}}=\boldsymbol{\mathsf{R}}(\boldsymbol{s}^{t}_{%
\boldsymbol{\mathcal{I}}},\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{I}}})$ and $r^{t}_{\boldsymbol{\mathcal{R}}}=\boldsymbol{\mathsf{R}}(\boldsymbol{s}^{t}_{%
\boldsymbol{\mathcal{R}}},\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{R}}})$ as described in [Sec. 3.2](#S3.SS2) and [Sec. 3.3](#S3.SS3). For both stages, we employ proximal policy optimization (PPO) [[[99](#bib.bib99)]] to maximize the discounted reward $\mathbb{E}\left[\smash{\textstyle\sum_{t=1}^{T}}\gamma^{t-1}r^{t}_{\text{stage%
}}\right]$, following previous methods [[[19](#bib.bib19), [89](#bib.bib89)]].

### 3.2 Hand Trajectory Imitating

In this stage, our objective is to learn a general hand trajectory imitation model, $\boldsymbol{\mathcal{I}}$, capable of accurately replicating detailed human finger motions. The state for each dexterous hand at time $t$ is defined as $\boldsymbol{s}^{t}_{\boldsymbol{\mathcal{I}}}=\{\boldsymbol{\tau}_{\boldsymbol%
{h}}^{t},\boldsymbol{s}^{t}_{\text{prop}}\}$, which includes the target hand trajectory $\boldsymbol{\tau}_{\boldsymbol{h}}^{t}$ and the current proprioception $\boldsymbol{s}^{t}_{\text{prop}}=\{\boldsymbol{q}_{\boldsymbol{d}}^{t},\dot{%
\boldsymbol{q}}_{\boldsymbol{d}}^{t},\boldsymbol{w}_{\boldsymbol{d}}^{t},\dot{%
\boldsymbol{w}}_{\boldsymbol{d}}^{t}\}$. Here, $\boldsymbol{q}_{\boldsymbol{d}}^{t}$ and $\boldsymbol{w}_{\boldsymbol{d}}^{t}$ denote the joint angles and wrist poses, respectively, along with their corresponding velocities. We aim to train the policy $\pi_{\boldsymbol{\mathcal{I}}}(\boldsymbol{a}^{t}|\boldsymbol{s}_{\boldsymbol{%
\mathcal{I}}}^{t},\boldsymbol{a}^{t-1})$ using RL to determine the actions $\boldsymbol{a}_{\boldsymbol{\mathcal{I}}}^{t}$.

Reward Functions.
The reward function $r^{t}_{\boldsymbol{\mathcal{I}}}$ is designed to encourage the dexterous hands to track the reference hand trajectory $\boldsymbol{\tau}_{\boldsymbol{h}}^{t}$ while ensuring stability and smoothness. It comprises three components: 1) Wrist tracking reward $r^{t}_{\text{wrist}}$: This reward minimizes the difference: $\boldsymbol{w}_{\boldsymbol{d}}^{t}\ominus\boldsymbol{w}_{\boldsymbol{h}}^{t}$ and $\dot{\boldsymbol{w}}_{\boldsymbol{d}}^{t}-\dot{\boldsymbol{w}}_{\boldsymbol{h}%
}^{t}$, $\ominus$ denotes the difference in $\mathbb{SE}(3)$ space.
2) Finger imitation reward $r^{t}_{\text{finger}}$: This component encourages the dexterous hand to closely follow the reference finger joint positions. We manually select $F$ finger keypoints on the dexterous hand corresponding to the MANO model, denoted as $\boldsymbol{j}_{\boldsymbol{d}}$. The weights $w_{f}$ and decay rates $\lambda_{f}$ are empirically set to emphasize the fingertips, particularly those of the thumb, index, and middle fingers. The parameters are in the Appx. This design helps mitigate the impact of morphological differences between human and robotic hands:

$$r^{t}_{\text{finger}}=\textstyle\sum_{f=1}^{F}{w_{f}\cdot\exp{(-\lambda_{f}\|% \boldsymbol{j}_{{\boldsymbol{d}_{f}}}^{t}-\boldsymbol{j}_{{\boldsymbol{h}_{f}}% }^{t}\|\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}})}}$$ \tag{1}

3) Smoothness Reward $r^{t}_{\text{smooth}}$: To alleviate jerky motions, we introduce a smoothness reward that penalizes the power exerted on each joint, defined as the element-wise product of joint velocities and torques, similar to the approach in [[[76](#bib.bib76)]].
The total reward is defined as: $r^{t}_{\boldsymbol{\mathcal{I}}}=w_{\text{wrist}}\cdot r^{t}_{\text{wrist}}+w_%
{\text{finger}}\cdot r^{t}_{\text{finger}}+w_{\text{smooth}}\cdot r^{t}_{\text%
{smooth}}$.

Training Strategy.
Decoupling hand imitation from object interaction offers additional benefits; specifically, $\pi_{\boldsymbol{\mathcal{I}}}$ does not require challenging-to-acquire manipulation data. We train the policy using hand-only datasets, including existing hand motion collections [[[134](#bib.bib134), [62](#bib.bib62), [107](#bib.bib107), [36](#bib.bib36), [137](#bib.bib137), [144](#bib.bib144), [14](#bib.bib14)]] and synthetic data generated via interpolation [[[105](#bib.bib105)]]. To balance training data between the left and right hands, we mirror these datasets; training time and additional details are provided in the Appx.
For efficiency, we employ reference state initialization (RSI) and early termination [[[89](#bib.bib89), [88](#bib.bib88)]]. If the dexterous hand keypoints $\boldsymbol{j}_{\boldsymbol{d}}$ deviate beyond a threshold $\epsilon_{\text{finger}}$, the episode terminates early and resets to a randomly sampled MoCap state. We also utilize curriculum learning [[[8](#bib.bib8)]], gradually reducing $\epsilon_{\text{finger}}$ to encourage broad exploration initially, then focusing on fine-grained finger control.

### 3.3 Residual Learning for Interaction

Building on the pre-trained $\pi_{\boldsymbol{\mathcal{I}}}$, we use a residual module $\boldsymbol{\mathcal{R}}$ to refine coarse actions and satisfy task-specific constraints.

State Space Expansion for Interaction.
To account for interactions between the dexterous hands and objects, we expand the state space beyond the hand-related states $\boldsymbol{s}^{t}_{\boldsymbol{\mathcal{I}}}$ by incorporating additional interaction-related information.
First, we compute the convex hull [[[116](#bib.bib116)]] of the object meshes $\boldsymbol{o}$ from MoCap data to generate the collidable object $\boldsymbol{\hat{o}}$ in the simulation environment. To manipulate the object along the reference $\boldsymbol{\mathcal{T}}_{\boldsymbol{o}}$, we include the object’s position $\boldsymbol{p}_{\boldsymbol{\hat{o}}}$ (relative to the wrist position $\boldsymbol{w}_{\boldsymbol{d}}$) and velocities $\dot{\boldsymbol{p}}_{\boldsymbol{\hat{o}}}$, center of mass $\boldsymbol{m}_{\boldsymbol{\hat{o}}}$, and gravitational force vector $\boldsymbol{G}_{\boldsymbol{\hat{o}}}$.
To better encode the object’s shape, we utilize the BPS representation [[[91](#bib.bib91)]]. Additionally, for enhancing perception, we encode the spatial relationship between the hands and the object using the distance metric: $\boldsymbol{D}(\boldsymbol{j}_{\boldsymbol{d}}^{t},\boldsymbol{p}_{\boldsymbol%
{\hat{o}}}^{t})=\textstyle{\|\boldsymbol{j}_{{\boldsymbol{d}}}^{t}-\boldsymbol%
{p}_{{\boldsymbol{\hat{o}}}}^{t}\|\smash{{}_{\scriptscriptstyle 2}^{%
\scriptscriptstyle 2}}}$, measuring the squared Euclidean distance between the dexterous hand keypoints and the object’s position.
Furthermore, we explicitly include the contact force $\boldsymbol{C}$ obtained from the simulation, capturing the interaction between the fingertips and the object’s surface. This tactile feedback is critical for stable grasping and manipulation, ensuring precise control during complex tasks.
In summary, the expanded interaction state for the residual module is defined as: $\boldsymbol{s}^{t}_{\text{interact}}=\{\boldsymbol{\tau}_{\boldsymbol{o}}^{t},%
\boldsymbol{p}_{\boldsymbol{\hat{o}}}^{t},\dot{\boldsymbol{p}}_{\boldsymbol{%
\hat{o}}}^{t},\boldsymbol{m}_{\boldsymbol{\hat{o}}}^{t},\boldsymbol{G}_{%
\boldsymbol{\hat{o}}}^{t},\text{BPS}(\boldsymbol{\hat{o}}),\boldsymbol{D}(%
\boldsymbol{j}_{\boldsymbol{d}}^{t},\boldsymbol{p}_{\boldsymbol{\hat{o}}}^{t})%
,\boldsymbol{C}^{t}\}$.

Residual Actions Combining Strategy.
Given the combined state $\boldsymbol{s}_{\boldsymbol{\mathcal{R}}}^{t}=\boldsymbol{s}^{t}_{\boldsymbol{%
\mathcal{I}}}\cup\boldsymbol{s}^{t}_{\text{interact}}$, our goal is to learn residual actions $\Delta\boldsymbol{a}_{\boldsymbol{\mathcal{R}}}^{t}$ that refine the initial imitation actions $\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{I}}}$ to ensure task compliance. During each step of the manipulation episode, we first sample the imitation action $\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{I}}}\sim\pi_{\boldsymbol{\mathcal{I}}%
}(\boldsymbol{a}^{t}|\boldsymbol{s}_{\boldsymbol{\mathcal{I}}}^{t},\boldsymbol%
{a}^{t-1})$. Conditioned on this action, we then sample the residual correction $\Delta\boldsymbol{a}_{\boldsymbol{\mathcal{R}}}^{t}\sim\pi_{\boldsymbol{%
\mathcal{R}}}(\Delta\boldsymbol{a}^{t}|\boldsymbol{s}_{\boldsymbol{\mathcal{R}%
}}^{t},\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{I}}},\boldsymbol{a}^{t-1})$. The final action is computed as: $\boldsymbol{a}^{t}=\boldsymbol{a}^{t}_{\boldsymbol{\mathcal{I}}}+\Delta%
\boldsymbol{a}_{\boldsymbol{\mathcal{R}}}^{t}$, where the residual action is added element-wise. The resulting action $\boldsymbol{a}^{t}$ is then clipped to adhere to the dexterous hand’s joint limits.
At the start of training, since the dexterous hand movements already approximate the reference hand trajectory, the residual actions are expected to be close to zero. This initialization helps prevent model collapse and accelerates convergence. We achieve this by initializing the residual module with a zero-mean Gaussian distribution and employing a warm-up strategy to gradually activate its training.

Reward Functions.
Our objective is to efficiently transfer human bimanual manipulation skills to dexterous robotic hands in a task-agnostic manner. To this end, we avoid task-specific reward engineering, which, although beneficial for individual tasks, can limit generalization. Therefore, our reward design remains simple and general. In addition to the hand imitation reward $r^{t}_{\boldsymbol{\mathcal{I}}}$ discussed in [Sec. 3.2](#S3.SS2), we introduce two additional components: 1) Object following reward $r^{t}_{\text{object}}$: Minimizes positional and velocity differences between the simulated object and its reference trajectory, specifically $\boldsymbol{p}_{\boldsymbol{\hat{o}}}^{t}\ominus\boldsymbol{p}_{\boldsymbol{o}%
}^{t}$ and $\dot{\boldsymbol{p}}_{\boldsymbol{\hat{o}}}^{t}-\dot{\boldsymbol{p}}_{%
\boldsymbol{o}}^{t}$. 2) Contact force reward $r^{t}_{\text{contact}}$: Encourages appropriate contact force when the hand-object distance in the MoCap dataset is below a specified threshold $\xi_{\text{c}}$. The reward is defined as:

| | $$r^{t}_{\text{contact}}=w_{\text{c}}\cdot\exp{(\frac{-\lambda_{\text{c}}}{% \textstyle\sum_{f=1}^{F}\boldsymbol{C}_{\boldsymbol{d}_{f}}^{t}\cdot\mathds{1}% \left(\boldsymbol{D}(\boldsymbol{j}_{\boldsymbol{h}_{f}}^{t},\boldsymbol{p}_{% \boldsymbol{o}}^{t}\cdot\boldsymbol{o})Training Strategy.
Inspired by prior work [[[85](#bib.bib85), [84](#bib.bib84), [72](#bib.bib72)]] that utilizes quasi-physical simulators to relax constraints during training and avoid local minima, we introduce a relaxation mechanism in the residual learning stage. Unlike [[[72](#bib.bib72)]], which employs custom simulations, we adjust the physical constraints directly within the Isaac Gym environment [[[79](#bib.bib79)]] to enhance training efficiency.
Specifically, we initially set the gravitational constant $\mathcal{G}$ to zero and the friction coefficient $\mathcal{F}$ to a high value. This setup allows the robotic hands to, early in training, grip objects firmly and efficiently align with reference trajectories.
As training progresses, we gradually restore $\mathcal{G}$ to its true value and reduce $\mathcal{F}$ to a suitable value to approximate real interactions.
Similar to the imitation stage, we adopt RSI, early termination, and curriculum learning strategies. Each episode initializes the robotic hands by randomly selecting a non-colliding near-object state from the preprocessed trajectory. During training, if the object’s pose $\boldsymbol{p}_{\boldsymbol{\hat{o}}}^{t}$ deviates beyond a predefined threshold $\epsilon_{\text{object}}$, the episode is terminated early. We progressively reduce $\epsilon_{\text{object}}$ to encourage more precise object manipulation.
Additionally, we introduce a contact termination condition: if MoCap data indicates a firm grasp by the human hands (i.e., $\boldsymbol{D}(\boldsymbol{j}_{\boldsymbol{h}_{f}}^{t},\boldsymbol{p}_{%
\boldsymbol{o}}^{t}\cdot\boldsymbol{o})ManipTrans, we generate DexManipNet, derived from two representative large-scale hand-object interaction datasets: FAVOR [[[62](#bib.bib62)]] and OakInk-V2 [[[134](#bib.bib134)]]. FAVOR employs VR-based teleoperation with human-in-the-loop corrections, focusing on foundational tasks like object rearrangement. In contrast, OakInk-V2 utilizes optical tracking-based motion capture, targeting more complex interactions such as pen capping and bottle unscrewing.

Due to the lack of standardization in dexterous robotic hands, we adopt the Inspire Hand [[[3](#bib.bib3)]] as our primary platform for its high dexterity, stability, cost-effectiveness, and extensive prior use [[[52](#bib.bib52), [35](#bib.bib35), [24](#bib.bib24)]].
To address the complexity of bimanual tasks, we employ a simulated 12-DoF configuration of the Inspire Hand, enhancing flexibility compared to its real-world 6-DoF mechanism. We demonstrate ManipTrans’s adaptability to other robotic hands and real-world deployment in [Sec. 4.4](#S4.SS4) and [Sec. 4.5](#S4.SS5).

Our DexManipNet encompasses 61 diverse and challenging tasks as defined in [[[134](#bib.bib134)]], comprising 3.3K episodes of robotic hand manipulation over 1.2K objects, totaling 1.34 million frames, including $\sim 600$ sequences involving complex bimanual tasks. Each episode executes precisely in the Isaac Gym simulation [[[79](#bib.bib79)]]. In comparison, a recent dataset generated via automated augmentation [[[52](#bib.bib52)]] includes only 60 source human demonstrations across 9 tasks.

## 4 Experiments

In experiments, we describe the dataset setup and metrics ([Sec. 4.1](#S4.SS1)), followed by implementation details ([Sec. 4.2](#S4.SS2)). We then compare ManipTrans with SOTA methods ([Sec. 4.3](#S4.SS3)), demonstrate cross-embodiment generalization ([Sec. 4.4](#S4.SS4)), validate real-world deployment ([Sec. 4.5](#S4.SS5)), conduct ablation studies ([Sec. 4.6](#S4.SS6)), and benchmark DexManipNet for learning manipulation policies ([Sec. 4.7](#S4.SS7)).

### 4.1 Datasets and Metrics

Datasets
For quantitative evaluation, we use the official validation dataset of OakInk-V2 [[[134](#bib.bib134)]], approximately half of which consists of bimanual tasks. To assess transfer capabilities, we manually select MoCap sequences that meet task completeness and semantic relevance, filtering them to durations of 4–20 seconds and downsampling to 60 fps. We exclude sequences involving deformable or oversized objects, resulting in $\sim 80$ episodes. For qualitative evaluation, we also incorporate the GRAB [[[107](#bib.bib107)]], FAOVR [[[62](#bib.bib62)]], and ARCTIC [[[32](#bib.bib32)]] datasets to demonstrate our advantages.

Metrics To evaluate ManipTrans in terms of manipulation precision, task compliance, and transfer efficiency, we introduce the following metrics. These are adapted from [[[72](#bib.bib72)]] but are more stringent due to the complexity of our bimanual tasks: 1) Per-frame Average Object Rotation and Translation Error:$E_{r}=\frac{1}{T}\textstyle\sum_{t=1}^{T}({\boldsymbol{p}_{\text{rot}}}_{%
\boldsymbol{\hat{o}}}^{t}\cdot{({\boldsymbol{p}_{\text{rot}}}_{\boldsymbol{o}}%
^{t})}^{-1})$ and $E_{t}=\frac{1}{T}\textstyle\sum_{t=1}^{T}\|{\boldsymbol{p}_{\text{tsl}}}_{%
\boldsymbol{\hat{o}}}^{t}-{{\boldsymbol{p}_{\text{tsl}}}_{\boldsymbol{o}}^{t}}%
\|\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}}$. Here, $\boldsymbol{p}_{\text{rot}}$ and $\boldsymbol{p}_{\text{tsl}}$ are the rotation and translation components of the 6-DoF pose $\boldsymbol{p}$, respectively. Errors $E_{r}$ and $E_{t}$ are reported in degrees and centimeters.
2) Mean Per-Joint Position Error (in $cm$): $E_{j}=\frac{1}{T\cdot F}\textstyle\sum_{t=1}^{T}\textstyle\sum_{f=1}^{F}\|%
\boldsymbol{j}_{{\boldsymbol{d}_{f}}}^{t}-\boldsymbol{j}_{{\boldsymbol{h}_{f}}%
}^{t}\|\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}}$. This metric measures the average error in the positions of the hand joints.
3) Mean Per-Fingertip Position Error (in $cm$): $E_{ft}=\frac{1}{T\cdot M}\textstyle\sum_{t=1}^{T}\textstyle\sum_{ft=1}^{M}\|%
\boldsymbol{t}_{{\boldsymbol{d}_{ft}}}^{t}-\boldsymbol{t}_{{\boldsymbol{h}_{ft%
}}}^{t}\|\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}}$.
This metric evaluates the mimicry quality of fingertip $\boldsymbol{t}$ motions, accounting for morphological differences between human and robotic hands. Here, $M$ equals 5 for single-hand tasks and 10 for bimanual tasks.
4) Success Rate ($SR$): A tracking attempt is deemed successful if $E_{r}$, $E_{t}$, $E_{j}$, and $E_{ft}$ are all below the specified thresholds: $30^{\circ}$, $3\ cm$, $8\ cm$, and $6\ cm$, respectively. For bimanual tasks, the trajectory is considered failed if either hand fails to meet these conditions, making the success criterion stricter compared to single-hand tasks.

### 4.2 Implementation Details

In ManipTrans, we manually selected $F=21$ keypoints on each dexterous robotic hand, corresponding to the fingertips, palm, and phalangeal positions on the human hand, to mitigate the morphological differences. Details on keypoint selection and weight coefficients $w$ for reward terms are provided in Appx.
For training, we use a curriculum learning strategy. The initial threshold $\epsilon_{\text{finger}}$ is set to $6\ cm$ and decays to $4\ cm$. Object alignment thresholds $\epsilon_{\text{object}}$ start at $90^{\circ}$ and $6\ cm$ for rotation and translation, gradually decreasing to $30^{\circ}$ and $2\ cm$.
We train both the imitation module $\boldsymbol{\mathcal{I}}$ and residual module $\boldsymbol{\mathcal{R}}$ using the Actor-Critic PPO algorithm [[[99](#bib.bib99)]], with a training horizon of 32 frames, a mini-batch size of 1024, and a discount factor $\gamma=0.99$. Optimization employs Adam [[[56](#bib.bib56)]] with an initial learning rate of $5\times 10^{-4}$ and a decay scheduler.
All experiments are run in Isaac Gym [[[79](#bib.bib79)]], simulating 4096 environments at a time step of $1/60$ s on a personal computer equipped with an NVIDIA RTX 4090 GPU and an Intel i9-13900KF CPU.

### 4.3 Evaluations

As discussed in [Sec. 2](#S2), dexterous hand manipulation advances rapidly, with previous approaches differing in problem formulations and task definitions. To offer a comprehensive and fair comparison, we evaluate two categories of methods—RL-combined and optimization-based—to demonstrate ManipTrans’s accuracy and efficiency.

*Figure 3: Qualitative Results of ManipTrans. We showcase the transfer results using the Inspire left and right hands on both single-hand tasks (top two rows) and bimanual tasks (bottom row) from the OakInk-V2 [[[134](#bib.bib134)]] dataset. Notably, the dexterous hands successfully manipulate delicate and slim objects, such as a pen and a flower stem.*

*Table 1: Quantitative Comparisons with RL-Combined Baselines. The first four metrics are computed only on successfully rolled-out sequences. The $SR$ includes the separated transfer success rates for single/bimanual tasks. The error scores on Retarget-Only are not available since it hardly works.*

Comparison with RL-Combined Methods
Due to the lack of publicly available code for prior RL-combined methods, we reimplement representative approaches: 1) RL-Only exploration using only trajectory-following rewards, employing the PPO algorithm to train the robotic hand from scratch based on [[[27](#bib.bib27)]]; 2) Retarget + Residual learning, applying residual action to retargeted robotic hand poses obtained via alignment between human and robot keypoints [[[94](#bib.bib94)]]. As a naive baseline, we also include the Retarget-Only method—retargeting without any learning.

As shown in [Tab. 1](#S4.T1), our method outperforms all baselines across multiple metrics, demonstrating superior precision in both single- and bimanual tasks. These results confirm that our two-stage transfer framework effectively captures subtle finger motions and object interactions, leading to high task success rates and motion fidelity.

We find that the Retarget-Only baseline is nearly infeasible due to the complexity of the dexterous hand action space and error accumulation. The RL-Only baseline performs suboptimally since exploration from scratch is time-consuming and reduces motion precision. Compared to the Retarget + Residual baseline, our method—leveraging a pre-trained hand imitation model—demonstrates improved control capabilities, enabling more accurate manipulation aligned with the reference trajectory.
Notably, the Retargeting method often causes collisions in contact-rich scenarios, resulting in instability during residual policy training.
We further study ManipTrans’s robustness and time cost in Appx. [Fig. 3](#S4.F3) shows the qualitative results on seldom-explored tasks, highlighting the natural and precision of ManipTrans transferring human manipulation skills. Additional details and more qualitative results applying our method to articulated objects are provided in Appx.

Comparison with Optimization-Based Method
QuasiSim [[[72](#bib.bib72)]] optimizes over customized simulations to track human motions. Currently, their full pipeline has not yet been released, and their “randomly" selected validation set is not available. Thus, a direct quantitative comparison is not feasible. Therefore, we provide a qualitative comparison in [Fig. 4](#S4.F4), demonstrating ManipTrans’s ability to transfer human motions to the Shadow Hand in a setting similar to QuasiSim’s, but with more stable contacts and smoother motions.
Notably, due to our two-stage design, for an unseen single-hand manipulation trajectory of 60 frames (“rotating a mouse"), our method requires $\sim 15$ minutes of training to achieve robust results, compared to QuasiSim’s $\sim 40$ hours of optimization111Results shown in QuasiSim’s Appx and its official repository: [https://github.com/Meowuu7/QuasiSim](https://github.com/Meowuu7/QuasiSim), highlighting ManipTrans’s significant efficiency.

### 4.4 Cross-Embodiments Validation

![Figure](x4.png)

*Figure 4: Qualitative Comparison with QuasiSim [[[72](#bib.bib72)]]. ManipTrans produces more natural motion of the Shadow hand (purple region) and is applicable to other dexterous hands.*

We demonstrate ManipTrans’s extensibility across various dexterous hand embodiments. As described in [Sec. 3](#S3), the imitation module $\boldsymbol{\mathcal{I}}$ addresses hand keypoint tracking, while the residual module $\boldsymbol{\mathcal{R}}$ captures physical interactions between fingertips and objects. Our framework is embodiment-agnostic since it relies solely on the correspondence between human fingers and robotic joints, allowing adaptation to different dexterous hands with minimal effort.
We evaluate ManipTrans on the Shadow Hand [[[1](#bib.bib1)]], articulated MANO hand [[[96](#bib.bib96), [27](#bib.bib27)]], Inspire Hand [[[3](#bib.bib3)]], and Allegro Hand [[[2](#bib.bib2)]], which have varying DoFs: $K=22$, 22, 12, and 16, respectively. Without altering network hyperparameters or reward weights, ManipTrans achieves consistent, fluid, and precise performance across all embodiments in both single-hand tasks ([Fig. 4](#S4.F4)) and bimanual tasks ([Fig. 5](#S4.F5)). Additional details on the Allegro Hand—a robotic hand with only four fingers—are provided in Appx.

![Figure](x5.png)

*Figure 5: Cross Embodiments Results: Putting off Alcohol lamp.*

### 4.5 Real-World Deployment

As illustrated in [Fig. 6](#S4.F6), we conduct experiments using two 7-DoF Realman arms [[[95](#bib.bib95)]] and a pair of upgraded Inspire Hands (same configuration yet adding tactile sensors). To bridge the gap between the simulated 12-DoF robotic hands and the 6-DoF real hardware, we employ a fitting-based method that optimizes the joint angles $\boldsymbol{q}_{\tilde{\boldsymbol{d}}}\in\mathbb{R}^{6}$ of the real robots (denoted as $\tilde{\cdot}$) for fingertip alignment, formulated as: $\operatorname*{argmin}_{\boldsymbol{q}_{\tilde{\boldsymbol{d}}}}\frac{1}{T%
\cdot M}\textstyle\sum_{t=1}^{T}\textstyle\sum_{ft=1}^{M}\|\boldsymbol{t}_{{%
\boldsymbol{d}_{ft}}}^{t}-\boldsymbol{t}_{{\tilde{\boldsymbol{d}}_{ft}}}^{t}\|%
\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}}$ with an additional temporal smoothness loss: $L_{\text{smooth}}=\frac{1}{T-1}\textstyle\sum_{t=1}^{T-1}\|\boldsymbol{q}_{%
\tilde{\boldsymbol{d}}}^{t+1}-\boldsymbol{q}_{\tilde{\boldsymbol{d}}}^{t}\|%
\smash{{}_{\scriptscriptstyle 2}^{\scriptscriptstyle 2}}$.
We control the arms by solving inverse kinematics to align the arms’ flanges with the dexterous hands’ wrists $\boldsymbol{w}_{\boldsymbol{d}}$. During replay, we do not enforce strict temporal alignment, as the real robots cannot always operate as quickly as human hands.

[Fig. 6](#S4.F6) showcases dexterous manipulation that, to the best of our knowledge, has not previously been achieved. For example, in “opening the toothpaste", the left hand stably holds the tube while the right hand’s thumb and index finger flexibly pop open the tiny cap—motions challenging to capture via teleoperation. This underscores the potential of our method for future real-world policy learning.

![Figure](x6.png)

*Figure 6: Real-world bimanual manipulation deployment. Purple box: human hand motion; orange box: close-up of dexterous hands. More results are on the website. (Zoom in for details. \faSearchPlus)*

### 4.6 Abalation Studies

![Figure](x7.png)

*(a) Tactile ablations training curve.*

![Figure](x8.png)

*(b) Curve on training strategies.*

Figure 7: Training Curve of Ablation Studies. We assess tactile feedback in contact-rich tasks (e.g*., turning off a lamp) and curriculum learning in complex ones (*e.g*., capping a pen).

*Table 2: Imitating Learning on Bottle Rearrangement Task.*

Tactile Information as Auxiliary Input
In [Sec. 3.3](#S3.SS3), we integrate tactile information, specifically the contact force $\boldsymbol{C}$, into the pipeline in three ways: (1) as an observation input, (2) as a reward component to encourage contact, and (3) as a condition for early termination. Ablation studies ([Fig. 7(a)](#S4.F7.sf1)) labeled w/o $\boldsymbol{C}$ obs, w/o $\boldsymbol{C}$ reward, and w/o $\boldsymbol{C}$ term demonstrate that including $\boldsymbol{C}$ in the reward function improves task success rates, and treating $\boldsymbol{C}$ as an observation accelerates convergence. We also find that omitting $\boldsymbol{C}$ as a termination condition seems to enhance initial training performance but lowers overall convergence speed, highlighting the importance of stable contact in task completion.

Training Strategy
We begin training with a curriculum learning strategy that includes (1) relaxing gravity effects, (2) increasing friction influence, and (3) relaxing thresholds $\epsilon_{\text{finger}}$ and $\epsilon_{\text{object}}$. Ablation studies ([Fig. 7(b)](#S4.F7.sf2)), labeled w/o relax-gravity, w/o increased friction, and w/o relax-thresholds, show that for precise, complex bimanual motions, ignoring gravity and using high friction coefficients in the early stages accelerate convergence and achieve higher overall $SR$. Without initial relaxation of the threshold constraints, the network may fail to converge entirely.

### 4.7 DexManipNet for Policy Learning

To benchmark DexManipNet’s potential, we evaluate representative imitation learning methods on a fundamental policy learning task: rearrangement. Specifically, we focus on moving a bottle to a goal position. Given the bottle’s current and goal 6D poses, the environment state (including obstacles on the table), and the dexterous hand’s proprioception, the policy generates a sequence of robotic hand actions to pick up the bottle and place it at the target.

We evaluate four representative imitation learning methods: two regression-based behavior cloning approaches—IBC [[[33](#bib.bib33)]] and BET [[[101](#bib.bib101)]]—and two diffusion policy methods [[[25](#bib.bib25)]] with UNet [[[97](#bib.bib97)]] and Transformer [[[110](#bib.bib110)]] backbones. Each policy is trained on 85% of the 140 sequences involving the bottle rearrangement task in DexManipNet and evaluated on the remaining 15%. We perform 20 rollouts per sequence. A rollout is considered successful if the object’s final position is within $10~{}cm$ of the goal. Further details are provided in Appx.

As shown in [Tab. 2](#S4.T2), all methods perform suboptimally due to the task’s difficulty and the complexity of the dexterous hand action space. Regression-based behavior cloning approaches, in particular, suffer from error accumulation. These results highlight the inherent challenges of dexterous manipulation tasks, which require precise finger control and effective object manipulation. We hope that DexManipNet will facilitate advancements in this domain.

## 5 Conclusion and Discussion

ManipTrans is a two-stage framework that efficiently transfers human manipulation skills to dexterous robotic hands. By decoupling hand motion imitation from object interaction via residual learning, ManipTrans overcomes morphological differences and complex task challenges, ensuring high-fidelity motions and efficient training. Experiments demonstrate that ManipTrans surpasses SOTA methods in motion precision and computational efficiency, while also exhibiting cross-embodiment adaptability and feasibility for real-world deployment. Furthermore, the extensible DexManipNet establishes a new benchmark to advance progress in embodied AI.

Discussion and Limitations
Although ManipTrans successfully handles most MoCap data, some sequences cannot be transferred effectively. We attribute this to two main reasons: 1) excessive noise in interaction poses and 2) insufficiently accurate object models for simulation, particularly for articulated objects. Enhancing ManipTrans’s robustness and generating physically plausible object models are valuable directions for future research.

## References

-
sha [2005]

ShadowRobot.

[https://www.shadowrobot.com/dexterous-hand-series](https://www.shadowrobot.com/dexterous-hand-series), 2005.

-
all [2013]

Allegro Hands.

[https://www.allegrohand.com](https://www.allegrohand.com), 2013.

-
ins [2019]

Inspire Hands.

[https://en.inspire-robots.com/product-category/the-dexterous-hands](https://en.inspire-robots.com/product-category/the-dexterous-hands), 2019.

-
Agarwal et al. [2023]

Ananye Agarwal, Shagun Uppal, Kenneth Shaw, and Deepak Pathak.

Dexterous functional grasping.

In *CoRL*, 2023.

-
Alakuijala et al. [2021]

Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, and Cordelia Schmid.

Residual reinforcement learning from demonstrations.

*arXiv preprint arXiv:2106.08050*, 2021.

-
Argall et al. [2009]

Brenna D Argall, Sonia Chernova, Manuela Veloso, and Brett Browning.

A survey of robot learning from demonstration.

*Robotics and autonomous systems*, 2009.

-
Arunachalam et al. [2023]

Sridhar Pandian Arunachalam, Sneha Silwal, Ben Evans, and Lerrel Pinto.

Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation.

In *ICRA*, 2023.

-
Bengio et al. [2009]

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston.

Curriculum learning.

In *Proceedings of the 26th annual international conference on machine learning*, 2009.

-
Brahmbhatt et al. [2019]

Samarth Brahmbhatt, Cusuh Ham, Charles C. Kemp, and James Hays.

ContactDB: Analyzing and predicting grasp contact via thermal imaging.

In *CVPR*, 2019.

-
Brahmbhatt et al. [2020]

Samarth Brahmbhatt, Chengcheng Tang, Christopher D. Twigg, Charles C. Kemp, and James Hays.

ContactPose: A dataset of grasps with object contact and hand pose.

In *ECCV*, 2020.

-
Brohan et al. [2022]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.

Rt-1: Robotics transformer for real-world control at scale.

*arXiv preprint arXiv:2212.06817*, 2022.

-
Brohan et al. [2023]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al.

Rt-2: Vision-language-action models transfer web knowledge to robotic control.

*arXiv preprint arXiv:2307.15818*, 2023.

-
Cao et al. [2021]

Zhe Cao, Ilija Radosavovic, Angjoo Kanazawa, and Jitendra Malik.

Reconstructing hand-object interactions in the wild.

In *ICCV*, 2021.

-
Chao et al. [2021]

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al.

Dexycb: A benchmark for capturing hand grasping of objects.

In *CVPR*, 2021.

-
Chen et al. [2024a]

Sirui Chen, Chen Wang, Kaden Nguyen, Li Fei-Fei, and C Karen Liu.

Arcap: Collecting high-quality human demonstrations for robot learning with augmented reality feedback.

*arXiv preprint arXiv:2410.08464*, 2024a.

-
Chen et al. [2022a]

Tao Chen, Jie Xu, and Pulkit Agrawal.

A system for general in-hand object re-orientation.

In *CoRL*, 2022a.

-
Chen et al. [2023a]

Tao Chen, Megha Tippur, Siyang Wu, Vikash Kumar, Edward Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand reorientation of novel and complex object shapes.

*Science Robotics*, 2023a.

-
Chen et al. [2024b]

Tao Chen, Eric Cousineau, Naveen Kuppuswamy, and Pulkit Agrawal.

Vegetable peeling: A case study in constrained dexterous manipulation.

*arXiv preprint arXiv:2407.07884*, 2024b.

-
Chen et al. [2022b]

Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuan Jiang, Zongqing Lu, Stephen McAleer, Hao Dong, Song-Chun Zhu, and Yaodong Yang.

Towards human-level bimanual dexterous manipulation with reinforcement learning.

*NeurIPS*, 2022b.

-
Chen et al. [2023b]

Yuanpei Chen, Chen Wang, Li Fei-Fei, and C Karen Liu.

Sequential dexterity: Chaining dexterous policies for long-horizon manipulation.

*arXiv preprint arXiv:2309.00987*, 2023b.

-
Chen et al. [2024c]

Yuanpei Chen, Chen Wang, Yaodong Yang, and Karen Liu.

Object-centric dexterous manipulation from human motion data.

In *CoRL*, 2024c.

-
Chen et al. [2024d]

Zerui Chen, Shizhe Chen, Cordelia Schmid, and Ivan Laptev.

Vividex: Learning vision-based dexterous manipulation from human videos.

*arXiv preprint arXiv:2404.15709*, 2024d.

-
Chen et al. [2022c]

Zoey Qiuyu Chen, Karl Van Wyk, Yu-Wei Chao, Wei Yang, Arsalan Mousavian, Abhishek Gupta, and Dieter Fox.

Dextransfer: Real world multi-fingered dexterous grasping with minimal human demonstrations.

*arXiv preprint arXiv:2209.14284*, 2022c.

-
Cheng et al. [2024]

Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, and Xiaolong Wang.

Open-television: Teleoperation with immersive active visual feedback.

*arXiv preprint arXiv:2407.01512*, 2024.

-
Chi et al. [2023]

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song.

Diffusion policy: Visuomotor policy learning via action diffusion.

*IJRR*, 2023.

-
Chi et al. [2024]

Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song.

Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots.

*arXiv preprint arXiv:2402.10329*, 2024.

-
Christen et al. [2022]

Sammy Christen, Muhammed Kocabas, Emre Aksan, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions.

In *CVPR*, 2022.

-
Corona et al. [2020]

Enric Corona, Albert Pumarola, Guillem Alenya, Francesc Moreno-Noguer, and Grégory Rogez.

Ganhand: Predicting human grasp affordances in multi-object scenes.

In *CVPR*, 2020.

-
Davchev et al. [2022]

Todor Davchev, Kevin Sebastian Luck, Michael Burke, Franziska Meier, Stefan Schaal, and Subramanian Ramamoorthy.

Residual learning from demonstration: Adapting dmps for contact-rich manipulation.

*RA-L*, 2022.

-
Ding et al. [2024]

Runyu Ding, Yuzhe Qin, Jiyue Zhu, Chengzhe Jia, Shiqi Yang, Ruihan Yang, Xiaojuan Qi, and Xiaolong Wang.

Bunny-visionpro: Real-time bimanual dexterous teleoperation for imitation learning.

*arXiv preprint arXiv:2407.03162*, 2024.

-
Englert and Toussaint [2018]

Peter Englert and Marc Toussaint.

Learning manipulation skills from a single demonstration.

*IJRR*, 2018.

-
Fan et al. [2023]

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges.

Arctic: A dataset for dexterous bimanual hand-object manipulation.

In *CVPR*, 2023.

-
Florence et al. [2022]

Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson.

Implicit behavioral cloning.

In *CoRL*, 2022.

-
Fu et al. [2024a]

Rao Fu, Dingxi Zhang, Alex Jiang, Wanjia Fu, Austin Funk, Daniel Ritchie, and Srinath Sridhar.

Gigahands: A massive annotated dataset of bimanual hand activities.

*arXiv preprint arXiv:2412.04244*, 2024a.

-
Fu et al. [2024b]

Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn.

Humanplus: Humanoid shadowing and imitation from humans.

*arXiv preprint arXiv:2406.10454*, 2024b.

-
Gao et al. [2022]

Daiheng Gao, Yuliang Xiu, Kailin Li, Lixin Yang, Feng Wang, Peng Zhang, Bang Zhang, Cewu Lu, and Ping Tan.

Dart: Articulated hand model with diverse accessories and rich textures.

*NeurIPS*, 2022.

-
Garcia-Hernando et al. [2018]

Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim.

First-person hand action benchmark with rgb-d videos and 3d hand pose annotations.

In *CVPR*, 2018.

-
Garcia-Hernando et al. [2020]

Guillermo Garcia-Hernando, Edward Johns, and Tae-Kyun Kim.

Physics-based dexterous manipulations with estimated hand poses and residual reinforcement learning.

In *IROS*, 2020.

-
Hampali et al. [2020]

Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit.

Honnotate: A method for 3d annotation of hand and object poses.

In *CVPR*, 2020.

-
Hampali et al. [2022]

Shreyas Hampali, Sayan Deb Sarkar, Mahdi Rad, and Vincent Lepetit.

Keypoint transformer: Solving joint identification in challenging hands and object interactions for accurate 3d pose estimation.

In *CVPR*, 2022.

-
Handa et al. [2023]

Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, et al.

Dextreme: Transfer of agile in-hand manipulation from simulation to reality.

In *ICRA*. IEEE, 2023.

-
Hasson et al. [2019]

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid.

Learning joint reconstruction of hands and manipulated objects.

In *CVPR*, 2019.

-
Hasson et al. [2020]

Yana Hasson, Bugra Tekin, Federica Bogo, Ivan Laptev, Marc Pollefeys, and Cordelia Schmid.

Leveraging photometric consistency over time for sparsely supervised hand-object reconstruction.

In *CVPR*, 2020.

-
He et al. [2024a]

Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi.

Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning.

*arXiv preprint arXiv:2406.08858*, 2024a.

-
He et al. [2024b]

Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, and Guanya Shi.

Learning human-to-humanoid real-time whole-body teleoperation.

*arXiv preprint arXiv:2403.04436*, 2024b.

-
Huang et al. [2023]

Binghao Huang, Yuanpei Chen, Tianyu Wang, Yuzhe Qin, Yaodong Yang, Nikolay Atanasov, and Xiaolong Wang.

Dynamic handover: Throw and catch with bimanual hands.

*CoRL*, 2023.

-
Huang et al. [2020]

Jingwei Huang, Yichao Zhou, and Leonidas Guibas.

Manifoldplus: A robust and scalable watertight manifold surface generation method for triangle soups.

*arXiv preprint arXiv:2005.11621*, 2020.

-
Huang et al. [2024]

Ziye Huang, Haoqi Yuan, Yuhui Fu, and Zongqing Lu.

Efficient residual learning with mixture-of-experts for universal dexterous grasping.

*arXiv preprint arXiv:2410.02475*, 2024.

-
Jacobs et al. [1991]

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton.

Adaptive mixtures of local experts.

*Neural computation*, 1991.

-
Jiang et al. [2021]

Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang.

Hand-object contact consistency reasoning for human grasps generation.

In *ICCV*, 2021.

-
Jiang et al. [2024a]

Yunfan Jiang, Chen Wang, Ruohan Zhang, Jiajun Wu, and Li Fei-Fei.

Transic: Sim-to-real policy transfer by learning from online correction.

In *CoRL*, 2024a.

-
Jiang et al. [2024b]

Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu.

Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning.

*arXiv preprint arXiv:2410.24185*, 2024b.

-
Johannink et al. [2019]

Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine.

Residual reinforcement learning for robot control.

In *ICRA*, 2019.

-
Kaelbling et al. [1996]

Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore.

Reinforcement learning: A survey.

*Journal of artificial intelligence research*, 1996.

-
Kim et al. [2024]

Jeonghwan Kim, Jisoo Kim, Jeonghyeon Na, and Hanbyul Joo.

Parahome: Parameterizing everyday home activities towards 3d generative modeling of human-object interactions.

*arXiv preprint arXiv:2401.10232*, 2024.

-
Kingma and Ba [2015]

Diederik Kingma and Jimmy Ba.

Adam: A method for stochastic optimization.

In *ICLR*, 2015.

-
Kwon et al. [2021]

Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys.

H2o: Two hands manipulating objects for first person interaction recognition.

In *ICCV*, 2021.

-
Li et al. [2024a]

Haoming Li, Qi Ye, Yuchi Huo, Qingtao Liu, Shijian Jiang, Tao Zhou, Xiang Li, Yang Zhou, and Jiming Chen.

Tpgp: Temporal-parametric optimization with deep grasp prior for dexterous motion planning.

In *ICRA*, 2024a.

-
Li et al. [2024b]

Jinhan Li, Yifeng Zhu, Yuqi Xie, Zhenyu Jiang, Mingyo Seo, Georgios Pavlakos, and Yuke Zhu.

Okami: Teaching humanoid robots manipulation skills through single video imitation.

*arXiv preprint arXiv:2410.11792*, 2024b.

-
Li et al. [2023a]

Kailin Li, Lixin Yang, Haoyu Zhen, Zenan Lin, Xinyu Zhan, Licheng Zhong, Jian Xu, Kejian Wu, and Cewu Lu.

Chord: Category-level hand-held object reconstruction via shape deformation.

In *ICCV*, 2023a.

-
Li et al. [2024c]

Kailin Li, Jingbo Wang, Lixin Yang, Cewu Lu, and Bo Dai.

Semgrasp: Semantic grasp generation via language aligned discretization.

In *ECCV*, 2024c.

-
Li et al. [2024d]

Kailin Li, Lixin Yang, Zenan Lin, Jian Xu, Xinyu Zhan, Yifei Zhao, Pengxiang Zhu, Wenxiong Kang, Kejian Wu, and Cewu Lu.

Favor: Full-body ar-driven virtual object rearrangement guided by instruction text.

*AAAI*, 2024d.

-
Li et al. [2023b]

Puhao Li, Tengyu Liu, Yuyang Li, Yiran Geng, Yixin Zhu, Yaodong Yang, and Siyuan Huang.

Gendexgrasp: Generalizable dexterous grasping.

In *ICRA*, 2023b.

-
Li et al. [2023c]

Sizhe Li, Zhiao Huang, Tao Chen, Tao Du, Hao Su, Joshua B Tenenbaum, and Chuang Gan.

Dexdeform: Dexterous deformable object manipulation with human demonstrations and differentiable physics.

*ICLR*, 2023c.

-
Li et al. [2024e]

Yuyang Li, Bo Liu, Yiran Geng, Puhao Li, Yaodong Yang, Yixin Zhu, Tengyu Liu, and Siyuan Huang.

Grasp multiple objects with one hand.

*RA-L*, 2024e.

-
Liconti et al. [2024]

Davide Liconti, Yasunori Toshimitsu, and Robert Katzschmann.

Leveraging pretrained latent representations for few-shot imitation learning on a dexterous robotic hand.

*arXiv preprint arXiv:2404.16483*, 2024.

-
Lin et al. [2021]

Kevin Lin, Lijuan Wang, and Zicheng Liu.

End-to-end human pose and mesh reconstruction with transformers.

In *CVPR*, 2021.

-
Lin et al. [2024]

Toru Lin, Zhao-Heng Yin, Haozhi Qi, Pieter Abbeel, and Jitendra Malik.

Twisting lids off with two hands.

*arXiv preprint arXiv:2403.02338*, 2024.

-
Liu et al. [2023]

Qingtao Liu, Yu Cui, Qi Ye, Zhengnan Sun, Haoming Li, Gaofeng Li, Lin Shao, and Jiming Chen.

Dexrepnet: Learning dexterous robotic grasping network with geometric and spatial hand-object representations.

In *IROS*, 2023.

-
Liu et al. [2024a]

Qingtao Liu, Qi Ye, Zhengnan Sun, Yu Cui, Gaofeng Li, and Jiming Chen.

Masked visual-tactile pre-training for robot manipulation.

In *ICRA*, 2024a.

-
Liu et al. [2024b]

Wenhai Liu, Junbo Wang, Yiming Wang, Weiming Wang, and Cewu Lu.

Force-centric imitation learning with force-motion capture system for contact-rich manipulation.

*arXiv preprint arXiv:2410.07554*, 2024b.

-
Liu et al. [2024c]

Xueyi Liu, Kangbo Lyu, Jieqiong Zhang, Tao Du, and Li Yi.

Parameterized quasi-physical simulators for dexterous manipulations transfer.

In *ECCV*, 2024c.

-
Liu et al. [2022]

Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi.

Hoi4d: A 4d egocentric dataset for category-level human-object interaction.

In *CVPR*, 2022.

-
Liu et al. [2024d]

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi.

Taco: Benchmarking generalizable bimanual tool-action-object understanding.

*arXiv preprint arXiv:2401.08399*, 2024d.

-
Lu et al. [2024]

Haoran Lu, Ruihai Wu, Yitong Li, Sijie Li, Ziyu Zhu, Chuanruo Ning, Yan Shen, Longzan Luo, Yuanpei Chen, and Hao Dong.

Garmentlab: A unified simulation and benchmark for garment manipulation.

In *NeurIPS*, 2024.

-
Luo et al. [2023]

Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al.

Perpetual humanoid control for real-time simulated avatars.

In *ICCV*, 2023.

-
Luo et al. [2024a]

Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu.

Grasping diverse objects with simulated humanoids.

*arXiv preprint arXiv:2407.11385*, 2024a.

-
Luo et al. [2024b]

Zhengyi Luo, Jiashun Wang, Kangni Liu, Haotian Zhang, Chen Tessler, Jingbo Wang, Ye Yuan, Jinkun Cao, Zihui Lin, Fengyi Wang, et al.

Smplolympics: Sports environments for physically simulated humanoids.

*arXiv preprint arXiv:2407.00187*, 2024b.

-
Makoviychuk et al. [2021]

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al.

Isaac gym: High performance gpu-based physics simulation for robot learning.

*arXiv preprint arXiv:2108.10470*, 2021.

-
Mandlekar et al. [2018]

Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al.

Roboturk: A crowdsourcing platform for robotic skill learning through imitation.

In *Conference on Robot Learning*, 2018.

-
Mao et al. [2024]

Xiaofeng Mao, Gabriele Giudici, Claudio Coppola, Kaspar Althoefer, Ildar Farkhatdinov, Zhibin Li, and Lorenzo Jamone.

Dexskills: Skill segmentation using haptic data for learning autonomous long-horizon robotic manipulation tasks.

*arXiv preprint arXiv:2405.03476*, 2024.

-
Oh et al. [2024]

Ji-Heon Oh, Ismael Espinoza, Danbi Jung, and Tae-Seong Kim.

Bimanual long-horizon manipulation via temporal-context transformer rl.

*RA-L*, 2024.

-
O’Neill et al. [2023]

Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al.

Open x-embodiment: Robotic learning datasets and rt-x models.

*arXiv preprint arXiv:2310.08864*, 2023.

-
Pang and Tedrake [2021]

Tao Pang and Russ Tedrake.

A convex quasistatic time-stepping scheme for rigid multibody systems with contact and friction.

In *ICRA*, 2021.

-
Pang et al. [2023]

Tao Pang, HJ Terry Suh, Lujie Yang, and Russ Tedrake.

Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models.

*IEEE Transactions on Robotics*, 2023.

-
Park et al. [2024]

Younghyo Park, Jagdeep Singh Bhatia, Lars Ankile, and Pulkit Agrawal.

Dexhub and dart: Towards internet scale robot data collection.

*arXiv preprint arXiv:2411.02214*, 2024.

-
Pavlakos et al. [2024]

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik.

Reconstructing hands in 3d with transformers.

In *CVPR*, 2024.

-
Peng et al. [2018]

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne.

Deepmimic: Example-guided deep reinforcement learning of physics-based character skills.

*ACM TOG*, 2018.

-
Peng et al. [2021]

Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa.

Amp: Adversarial motion priors for stylized physics-based character control.

*ACM TOG*, 2021.

-
Peng et al. [2022]

Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler.

Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters.

*ACM TOG*, 2022.

-
Prokudin et al. [2019]

Sergey Prokudin, Christoph Lassner, and Javier Romero.

Efficient learning on point clouds with basis point sets.

In *ICCV*, 2019.

-
Qin et al. [2022a]

Yuzhe Qin, Hao Su, and Xiaolong Wang.

From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation.

*RA-L*, 2022a.

-
Qin et al. [2022b]

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang.

Dexmv: Imitation learning for dexterous manipulation from human videos.

In *ECCV*, 2022b.

-
Qin et al. [2023]

Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox.

Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system.

In *RSS*, 2023.

-
Realman Robotics [2010]

Realman Robotics.

RM Series.

[https://www.realman-robotics.com/rm-series123](https://www.realman-robotics.com/rm-series123), 2010.

-
Romero et al. [2017]

Javier Romero, Dimitrios Tzionas, and Michael J. Black.

Embodied hands: Modeling and capturing hands and bodies together.

*ACM TOG*, 2017.

-
Ronneberger et al. [2015]

Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

U-net: Convolutional networks for biomedical image segmentation.

In *MICCAI*, 2015.

-
Schoettler et al. [2020]

Gerrit Schoettler, Ashvin Nair, Jianlan Luo, Shikhar Bahl, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine.

Deep reinforcement learning for industrial insertion tasks with visual inputs and natural rewards.

In *IROS*, 2020.

-
Schulman et al. [2017]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

*arXiv preprint arXiv:1707.06347*, 2017.

-
Sener et al. [2022]

Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao.

Assembly101: A large-scale multi-view video dataset for understanding procedural activities.

In *CVPR*, 2022.

-
Shafiullah et al. [2022]

Nur Muhammad Shafiullah, Zichen Cui, Ariuntuya Arty Altanzaya, and Lerrel Pinto.

Behavior transformers: Cloning $k$ modes with one stone.

*NeurIPS*, 2022.

-
Shaw et al. [2023]

Kenneth Shaw, Shikhar Bahl, and Deepak Pathak.

Videodex: Learning dexterity from internet videos.

In *CoRL*, 2023.

-
Shaw et al. [2024]

Kenneth Shaw, Yulong Li, Jiahui Yang, Mohan Kumar Srirama, Ray Liu, Haoyu Xiong, Russell Mendonca, and Deepak Pathak.

Bimanual dexterity for complex tasks.

*arXiv preprint arXiv:2411.13677*, 2024.

-
She et al. [2024]

Qijin She, Shishun Zhang, Yunfan Ye, Min Liu, Ruizhen Hu, and Kai Xu.

Learning cross-hand policies for high-dof reaching and grasping.

*ECCV*, 2024.

-
Shoemake [1985]

Ken Shoemake.

Animating rotation with quaternion curves.

In *Proceedings of the 12th annual conference on Computer graphics and interactive techniques*, 1985.

-
Silver et al. [2018]

Tom Silver, Kelsey Allen, Josh Tenenbaum, and Leslie Kaelbling.

Residual policy learning.

*arXiv preprint arXiv:1812.06298*, 2018.

-
Taheri et al. [2020]

Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas.

Grab: A dataset of whole-body human grasping of objects.

In *ECCV*, 2020.

-
Tekin et al. [2019]

Bugra Tekin, Federica Bogo, and Marc Pollefeys.

H+ o: Unified egocentric recognition of 3d hand-object poses and interactions.

In *CVPR*, 2019.

-
Tessler et al. [2024]

Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng.

Maskedmimic: Unified physics-based character control through masked motion inpainting.

*SIGGRAPH ASIA*, 2024.

-
Vaswani [2017]

A Vaswani.

Attention is all you need.

*NeurIPS*, 2017.

-
Wan et al. [2023]

Weikang Wan, Haoran Geng, Yun Liu, Zikang Shan, Yaodong Yang, Li Yi, and He Wang.

Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

In *ICCV*, 2023.

-
Wang et al. [2023a]

Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar.

Mimicplay: Long-horizon imitation learning by watching human play.

*arXiv preprint arXiv:2302.12422*, 2023a.

-
Wang et al. [2024a]

Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, and C Karen Liu.

Dexcap: Scalable and portable mocap data collection system for dexterous manipulation.

*arXiv preprint arXiv:2403.07788*, 2024a.

-
Wang et al. [2024b]

Jun Wang, Yuzhe Qin, Kaiming Kuang, Yigit Korkmaz, Akhilan Gurumoorthy, Hao Su, and Xiaolong Wang.

Cyberdemo: Augmenting simulated human demonstration for real-world dexterous manipulation.

In *CVPR*, 2024b.

-
Wang et al. [2023b]

Ruicheng Wang, Jialiang Zhang, Jiayi Chen, Yinzhen Xu, Puhao Li, Tengyu Liu, and He Wang.

Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation.

In *ICRA*, 2023b.

-
Wei et al. [2022]

Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su.

Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search.

*ACM TOG*, 2022.

-
Wu et al. [2023]

Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel.

Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators.

*arXiv preprint arXiv:2309.13037*, 2023.

-
Wu et al. [2024]

Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, and Hao Dong.

Learning score-based grasping primitive for human-assisting dexterous grasping.

*NeurIPS*, 2024.

-
Xie et al. [2023]

Wei Xie, Zhipeng Yu, Zimeng Zhao, Binghui Zuo, and Yangang Wang.

Hmdo: Markerless multi-view hand manipulation capture with deformable objects.

*Graphical Models*, 2023.

-
Xu et al. [2022]

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao.

Vitpose: Simple vision transformer baselines for human pose estimation.

*NeurIPS*, 2022.

-
Xu et al. [2023a]

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, et al.

Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In *CVPR*, 2023a.

-
Xu et al. [2023b]

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao.

Vitpose++: Vision transformer for generic body pose estimation.

*IEEE TPAMI*, 2023b.

-
Yang et al. [2021]

Lixin Yang, Xinyu Zhan, Kailin Li, Wenqiang Xu, Jiefeng Li, and Cewu Lu.

Cpf: Learning a contact potential field to model the hand-object interaction.

In *ICCV*, 2021.

-
Yang et al. [2022a]

Lixin Yang, Kailin Li, Xinyu Zhan, Jun Lv, Wenqiang Xu, Jiefeng Li, and Cewu Lu.

Artiboost: Boosting articulated 3d hand-object pose estimation via online exploration and synthesis.

In *CVPR*, 2022a.

-
Yang et al. [2022b]

Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu.

Oakink: A large-scale knowledge repository for understanding hand-object interaction.

In *CVPR*, 2022b.

-
Yang et al. [2023]

Lixin Yang, Jian Xu, Licheng Zhong, Xinyu Zhan, Zhicheng Wang, Kejian Wu, and Cewu Lu.

Poem: reconstructing hand in a point embedded multi-view stereo.

In *CVPR*, 2023.

-
Yang et al. [2024a]

Lixin Yang, Xinyu Zhan, Kailin Li, Wenqiang Xu, Junming Zhang, Jiefeng Li, and Cewu Lu.

Learning a contact potential field for modeling the hand-object interaction.

*IEEE TPAMI*, 2024a.

-
Yang et al. [2024b]

Shiqi Yang, Minghuan Liu, Yuzhe Qin, Runyu Ding, Jialong Li, Xuxin Cheng, Ruihan Yang, Sha Yi, and Xiaolong Wang.

Ace: A cross-platform visual-exoskeletons system for low-cost dexterous teleoperation.

*arXiv preprint arXiv:2408.11805*, 2024b.

-
Ye et al. [2023]

Jianglong Ye, Jiashun Wang, Binghao Huang, Yuzhe Qin, and Xiaolong Wang.

Learning continuous grasping function with a dexterous hand from human demonstrations.

*RA-L*, 2023.

-
Yin et al. [2023]

Zhao-Heng Yin, Binghao Huang, Yuzhe Qin, Qifeng Chen, and Xiaolong Wang.

Rotating without seeing: Towards in-hand dexterity through touch.

*RSS*, 2023.

-
Yuan et al. [2024]

Haoqi Yuan, Bohan Zhou, Yuhui Fu, and Zongqing Lu.

Cross-embodiment dexterous grasping with reinforcement learning.

*arXiv preprint arXiv:2410.02479*, 2024.

-
Zakka et al. [2023]

Kevin Zakka, Philipp Wu, Laura Smith, Nimrod Gileadi, Taylor Howell, Xue Bin Peng, Sumeet Singh, Yuval Tassa, Pete Florence, Andy Zeng, et al.

Robopianist: Dexterous piano playing with deep reinforcement learning.

*CoRL*, 2023.

-
Ze et al. [2024]

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu.

3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations.

In *RSS*, 2024.

-
Zhan et al. [2024]

Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu.

Oakink2: A dataset of bimanual hands-object manipulation in complex task completion.

In *CVPR*, 2024.

-
Zhang et al. [2024a]

Hui Zhang, Sammy Christen, Zicong Fan, Otmar Hilliges, and Jie Song.

Graspxl: Generating grasping motions for diverse objects at scale.

*ECCV*, 2024a.

-
Zhang et al. [2024b]

Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo, Jie Song, and Otmar Hilliges.

Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation.

In *3DV*, 2024b.

-
Zhang et al. [2017]

Jiawei Zhang, Jianbo Jiao, Mingliang Chen, Liangqiong Qu, Xiaobin Xu, and Qingxiong Yang.

A hand pose tracking benchmark from stereo matching.

In *ICIP*, 2017.

-
Zhang et al. [2023]

Xiang Zhang, Changhao Wang, Lingfeng Sun, Zheng Wu, Xinghao Zhu, and Masayoshi Tomizuka.

Efficient sim-to-real transfer of contact-rich manipulation skills with online admittance residual learning.

In *CoRL*, 2023.

-
Zhao et al. [2024]

Shuqi Zhao, Xinghao Zhu, Yuxin Chen, Chenran Li, Xiang Zhang, Mingyu Ding, and Masayoshi Tomizuka.

Dexh2r: Task-oriented dexterous manipulation from human to robots.

*arXiv preprint arXiv:2411.04428*, 2024.

-
Zhao et al. [2023]

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn.

Learning fine-grained bimanual manipulation with low-cost hardware.

*arXiv preprint arXiv:2304.13705*, 2023.

-
Zhong et al. [2024]

Licheng Zhong, Lixin Yang, Kailin Li, Haoyu Zhen, Mei Han, and Cewu Lu.

Color-neus: Reconstructing neural implicit surfaces with color.

In *3DV*, 2024.

-
Zhou et al. [2024]

Bohan Zhou, Haoqi Yuan, Yuhui Fu, and Zongqing Lu.

Learning diverse bimanual dexterous manipulation skills from human demonstrations.

*arXiv preprint arXiv:2410.02477*, 2024.

-
Zhu et al. [2023]

Zehao Zhu, Jiashun Wang, Yuzhe Qin, Deqing Sun, Varun Jampani, and Xiaolong Wang.

Contactart: Learning 3d interaction priors for category-level articulated object and hand poses estimation.

*arXiv preprint arXiv:2305.01618*, 2023.

-
Zimmermann et al. [2019]

Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox.

Freihand: A dataset for markerless capture of hand pose and shape from single rgb images.

In *ICCV*, 2019.

\thetitle

Supplementary Material

This appendix provides additional details and results that complement the main paper. We first validate the extensibility of ManipTrans in [Appendix A](#A1). We then evaluate the robustness of ManipTrans under noisy conditions in [Appendix B](#A2) and analyze its time cost in [Appendix C](#A3). Detailed information on the settings of ManipTrans is provided in [Appendix D](#A4), along with statistics for the DexManipNet dataset in [Appendix E](#A5). Finally, we present the training details for the rearrangement policies in [Appendix F](#A6).

## Appendix A Further Extension of ManipTrans

### A.1 Articulated Object Manipulation

We demonstrate the extensibility of ManipTrans by applying it to the ARCTIC dataset [[[32](#bib.bib32)]], which includes approximately 10 articulated objects, each with precise hand manipulation trajectories for bimanual single-object manipulation tasks.

To accommodate the articulated object manipulation task, we extend our method pipeline. For a single articulated object $o^{\text{A}}$, we define its trajectory as $\boldsymbol{\mathcal{T}}_{o^{\text{A}}}=\{\boldsymbol{\tau}^{t}_{o^{\text{A}}}%
\}_{t=1}^{T}$, where $\boldsymbol{\tau}_{o^{\text{A}}}=\{\boldsymbol{p}_{o^{\text{A}}},\dot{%
\boldsymbol{p}}_{o^{\text{A}}},\theta_{o^{\text{A}}},\dot{\theta}_{o^{\text{A}%
}}\}$ represents the object’s transformation, velocity, and the angle and angular velocity of its articulated part. The reward function for articulated objects, $r^{t}_{\text{object}^{\text{A}}}$, includes two additional terms compared to the reward for rigid objects: the angle difference $|\theta_{o^{\text{A}}}-\theta_{{\hat{o^{\text{A}}}}}|$ and the angular velocity difference $|\dot{\theta}_{o^{\text{A}}}-\dot{\theta}_{{\hat{o^{\text{A}}}}}|$, where ${\hat{o^{\text{A}}}}$ represents the collidable articulated object in the simulation environment [[[79](#bib.bib79)]]. Apart from this modification, the rest of the pipeline remains unchanged.

Qualitative results of ManipTrans applied to the ARCTIC dataset are presented in [Fig. 8](#A1.F8), demonstrating that our method successfully imitates human demonstrations and rotates the articulated object to the desired target angle. This highlights the extensibility of our pipeline when the physical properties of the articulated object can be accurately modeled in simulation.

### A.2 Challenging Hand Embodiments

We investigate the generalization capabilities of ManipTrans across different hand embodiments in the main paper. Here, we provide further details on adapting ManipTrans to a challenging hand model: the Allegro Hand [[[2](#bib.bib2)]], which possesses $K=16$ degrees of freedom. The challenges encountered stem from two primary factors: 1) the Allegro Hand has only four fingers, a significant deviation from the structure of the human hand, and 2) the Allegro Hand is approximately twice the size of a human hand. These morphological discrepancies present substantial challenges in transferring human demonstrations to the Allegro Hand.

To address these challenges, we adaptively modify the fingertip mapping relationships, mapping both the pinky and ring fingers to the same fingertip on the Allegro Hand. Additionally, we relax the fingertip keypoint threshold $\epsilon_{\text{finger}}$ to $8\ cm$ to accommodate the larger dimensions of the Allegro Hand. Successful application of ManipTrans to the Allegro Hand is demonstrated in [Fig. 9](#A1.F9).

*Figure 8: Applying ManipTrans to Articulated Object Manipulation. In the first row, the two hands collaborate to not only close the book but also place it stably on the table.*

![Figure](x10.png)

*Figure 9: Extending ManipTrans to the Allegro Hand. Despite the Allegro Hand having only four fingers and a significantly larger size, the transferred motion remains stable and natural.*

### A.3 Discussion on the Extension

To summarize, we present all settings for the extension experiments in [Tab. 3](#A1.T3). The green checkmark ( ✓) indicates the successful transfer of the dataset to the specified hand embodiment, with results included in DexManipNet. The blue checkmark ( ✓) denotes dataset verification, where ManipTrans is tested on only a subset of the dataset to assess generalizability. The results demonstrate that our pipeline effectively accommodates various morphological differences across hand embodiments and supports a wide range of tasks, including single-hand manipulation, bimanual articulated object manipulation, and bimanual two-object manipulation.

As discussed in Sec. 3.4 of the main paper, FAVOR [[[62](#bib.bib62)]] and OakInk-V2 [[[134](#bib.bib134)]] represent the largest datasets with the most diverse task types, while the Inspire Hand is distinguished by its high dexterity, stability, cost-effectiveness, and extensive prior use [[[52](#bib.bib52), [35](#bib.bib35), [24](#bib.bib24)]]. Consequently, this setup was chosen for collecting DexManipNet. However, ManipTrans is fully adaptable, and we demonstrate that all of the aforementioned MoCap datasets can be transferred to other robotic hands. We welcome further collaboration from the research community.

*Table 3: Extensibility of ManipTrans. Arti-MANO refers to the articulated MANO hand used in [[[27](#bib.bib27)]].*

## Appendix B Robustness Evaluation

*Table 4: Quantitative Results Under Different Noise Levels. We add the Gaussian noise $\mathcal{N}(0,\sigma^{2})$ to the target hand joints poses.*

MoCap data and model-based pose estimation results often contain noise. To assess whether ManipTrans can reliably transfer noisy real-world data into stable robotic motions within a simulation environment, we conduct robustness tests. Since ManipTrans is designed for general-purpose transfer and does not depend on task-specific reward functions (e.g., the twisting reward proposed in [[[68](#bib.bib68)]] for the lip-twisting task), noisy object trajectories may introduce instability during the rollout process.
Thus, to evaluate ManipTrans’s performance under such conditions, we introduce random Gaussian noise into the hand trajectory input and focus on single-hand manipulation tasks. This choice is motivated by the fact that most hand pose estimation methods [[[67](#bib.bib67), [127](#bib.bib127), [124](#bib.bib124)]] are optimized for single-hand scenarios.

The results, presented in [Tab. 4](#A2.T4), demonstrate that ManipTrans maintains acceptable performance even when the noise level reaches up to $1.5\ cm$. These findings highlight the potential of ManipTrans for real-world scaling, particularly in applications involving hand pose estimation from web video data, which may implicitly contain a vast array of dexterous manipulation skills.

## Appendix C Time Cost Analysis

![Figure](x11.png)

*Figure 10: Detailed Efficiency Comparison. The success rate curves for the “rotating a mouse" task.*

In Sec. 4.3 of the main paper, we compare the efficiency of our method with the previous SOTA method, QuasiSim. QuasiSim employs a set of quasi-physical simulations, dividing the transfer process into three primary stages, with each stage requiring approximately 10-20 hours for a 60-frame trajectory 111As reported in the official repository: [https://github.com/Meowuu7/QuasiSim](https://github.com/Meowuu7/QuasiSim). Since ManipTrans also follows a multi-stage framework, incorporating both a pre-trained hand imitation module and a residual refinement module tailored to physical dynamics, we provide a more comprehensive comparison of efficiency.

For a fair evaluation, we use the official QuasiSim demo data for the “rotating a mouse" task as a representative example. The success rate curves for three different settings, as discussed in Sec. 4.3 of the main paper, are shown in [Fig. 10](#A3.F10): 1) RL-Only: This approach trains the policy network from scratch using RL with our reward design. The curve illustrates the entire training process. 2) Retarget + Residual Learning: Inspired by [[[139](#bib.bib139)]], this method retargets human hand poses to initial dexterous hand poses via keypoint alignment [[[94](#bib.bib94)]], followed by residual learning for refinement. The retargeting process is performed via parallel optimization and only requires approximately several minutes on a single GPU to optimize full sequence. The training curve for the residual learning stage is represented by the orange line. 3) ManipTrans: We pre-train the hand imitation model on a large-scale training dataset, as described in Sec. 3.2, which takes approximately 1.5 days on a single GPU to obtain the reusable imitator. The residual learning stage training curve is shown by the blue line.

From the results in [Fig. 10](#A3.F10), we observe that for the relatively simple task of “rotating a mouse", the Retarget + Residual method achieves performance comparable to ManipTrans but requires slightly more time to converge. The RL-Only approach, while yielding suboptimal performance compared to the other methods, still produces acceptable motions within 20 minutes. This indicates that our reward design effectively accelerates the training process, facilitating faster convergence.

## Appendix D Details of ManipTrans Settings

### D.1 Correspondence Between Human Hand and Dexterous Hand

Due to the significant morphological differences between human hands and dexterous robotic hands, we manually establish correspondences between them.
For the human hand’s fingertip keypoints, we select the midpoint of the three tip anchors as defined in [[[127](#bib.bib127)]]. For the dexterous hands, given their varying shapes, we define the fingertip keypoints as the points of maximum curvature along the central axis of the finger pads, as these points are most likely to contact objects. For other keypoints, such as the wrist and phalanges, we intuitively align the rotation axes of the human joints with those of the robotic joints. For further details, please refer to our code implementation.

In addition, regarding the articulated MANO model, the original human hand model MANO [[[96](#bib.bib96)]] has 45-DoF, which presents extreme challenges for RL-based policies due to the vast exploration space. To mitigate this, we follow the approach in [[[127](#bib.bib127)]] by constraining certain DoFs and fixing the hand collision meshes, thereby reducing the original MANO model to a 22-DoF articulated MANO.

### D.2 Details of Training Parameters

*Table 5: Hyperparameters for the Finger Reward. The weight $w_{f}$ and decay rate $\lambda_{f}$ are used to balance the importance of each finger. Each cell in the table contains four values, representing the parameters for the proximal, intermediate, distal, and tip joints, respectively. For anatomical definitions, please refer to [[[127](#bib.bib127)]].*

In this section, we present the core parameters of our reward functions in ManipTrans. The reward parameters for $r^{t}_{\text{finger}}$ in Eq. (1) of the main paper are summarized in [Tab. 5](#A4.T5). These parameters are determined based on the observation that the thumb, index, and middle fingers play a pivotal role in grasping and manipulation tasks, as they statistically interact with objects more frequently than other fingers [[[134](#bib.bib134), [10](#bib.bib10), [9](#bib.bib9)]]. Consequently, the weights are assigned according to the contact frequency.
In our implementation, if a dexterous hand lacks a specific finger or joint (e.g*., the Inspire Hand does not have distal joints), the corresponding parameters are set to zero.
For the contact reward $r^{t}_{\text{contact}}$ in Eq. (2) of the main paper, we set both parameters, $w_{c}$ and $\lambda_{c}$, to 1.

### D.3 Details of Simulation Parameters

In the Isaac Gym environment, configuring physical properties significantly influences the success rate of transfer. Alongside domain randomization (DR) during training, we set physical constants as follows. For certain objects in OakInk-V2 [[[134](#bib.bib134)]], we obtained actual masses by directly measuring them in collaboration with the dataset authors. For the remaining objects, we assigned a constant density of $200\ kg/m^{3}$, approximating the average density of low-fill-rate 3D-printed models. Using this density, we recalculated the objects’ masses and moments of inertia.

It is worth noting that human skin is elastic. When grasping objects, fingertip skin undergoes slight deformations, enhancing contact with object surfaces and generating suitable friction, whereas dexterous robotic hands lack this behavior. Previous kinematics-based grasp generation methods [[[50](#bib.bib50), [61](#bib.bib61)]] often permit slight penetration between fingertips and object surfaces to improve interaction stability (for detailed discussion, please refer to [[[50](#bib.bib50)]]). Therefore, to compensate for the absence of skin deformation in simulation, we set the friction coefficient $\mathcal{F}$ slightly higher than the real-world value. Accurately simulating contact-rich scenarios remains an area for future exploration.

*Table 6: List of tasks in the DexManipNet dataset. Tasks with underlined names usually require bimanual manipulation.*

## Appendix E DexManipNet Statistics

To the best of our knowledge, no prior work has collected a large-scale bimanual manipulation dataset in which all trajectories are directly transferred from real human demonstrations without the use of teleportation. Leveraging the efficiency and precision of ManipTrans, our dataset, DexManipNet, comprises 3.3K diverse manipulation trajectories across 61 distinct tasks, as detailed in [Tab. 6](#A4.T6). To ensure stability during simulation, we fix the object meshes to a watertight state using ManifoldPlus [[[47](#bib.bib47)]] and may slightly adjust the object size to enhance object-object interactions (e.g., the cap and body of the bottle).

Additionally, we provide sample data on our website, showcasing trajectories generated from our policy in simulation. A simple first-order low-pass filter ($\alpha=0.4$) is applied to the rollouts, effectively reducing jitter with minimal impact on tracking accuracy.

## Appendix F Details of Rearrangement Policy Learning

![Figure](extracted/6313951/figure/supp_imitation.png)

*Figure 11: Qualitative Results of Rearrangement Policy Learning. The policy successfully moves the bottle to the goal position. Results are directly visualized in the IsaacGym environment, highlighting distinctions between these policies and ManipTrans’s rollouts.*

As discussed in Sec. 4.7 of the main paper, we benchmark DexManipNet using four data-driven imitation learning methods on the moving a bottle to a goal position task.

The primary challenge in this task is to enable the dexterous hand to maintain a stable grasp on the object while smoothly placing it at the specified goal position. We evaluate the dataset using four methods: IBC [[[33](#bib.bib33)]], BET [[[101](#bib.bib101)]], and Diffusion Policy [[[25](#bib.bib25)]], which include both UNet- and Transformer-based architectures. These policies are trained for 500 epochs using the Adam optimizer with a learning rate of $1\times 10^{-4}$, while all other hyperparameters remain at their default settings.

The dimensions of the observation and action spaces for these policies are provided in [Tab. 7](#A6.T7). The observation space includes the current object state $\{\boldsymbol{p}_{\boldsymbol{\hat{o}}},\dot{\boldsymbol{p}}_{\boldsymbol{\hat%
{o}}}\}$, the hand wrist state $\{\boldsymbol{w}_{\boldsymbol{d}},\dot{\boldsymbol{w}}_{\boldsymbol{d}}\}$, hand joint angles $\boldsymbol{q}_{\boldsymbol{d}}$, and the goal poses for both the object $\boldsymbol{g}_{\boldsymbol{\hat{o}}}$ and the hand wrist $\boldsymbol{g}_{\boldsymbol{w}}$. The action $\boldsymbol{a}=\{\boldsymbol{a}_{\boldsymbol{q}},\boldsymbol{a}_{\boldsymbol{w%
}}\}\in\boldsymbol{\mathcal{A}}$ specifies the target hand joint angles and wrist poses using a PD controller. Note that PD control is used for wrist poses rather than a 6-DoF force, as is done in ManipTrans.

We evaluate the policies’ performance on previously unseen goal positions within the IsaacGym environment [[[79](#bib.bib79)]]. A rollout is considered successful if the object’s distance from the goal position is within $10\ cm$; otherwise, it is classified as a failure. Qualitative results are presented in [Fig. 11](#A6.F11), while quantitative results are summarized in Tab. 2 of the main paper.

*(a) Observation space.*

*(b) Action space.*

Table 7: Observation and Action Definitions for the Imitation Policy. The policy’s 7-dimensional pose includes both position and orientation, represented as XYZW quaternions. The policy’s 13-dimensional state extends this pose by incorporating both linear and angular velocities.

Generated on Thu Mar 27 06:10:33 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)