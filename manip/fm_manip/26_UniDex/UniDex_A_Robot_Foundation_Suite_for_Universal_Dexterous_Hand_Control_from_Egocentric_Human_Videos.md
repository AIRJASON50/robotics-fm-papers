##### Report GitHub Issue

**×




Title:
*


Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.22264v1 [cs.RO] 23 Mar 2026

[# UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos Gu Zhang1,2,∗†, Qicheng Xu1,∗, Haozhe Zhang1,∗, Jianhan Ma2,∗, Long He1,∗, Yiming Bao1,∗, Zeyu Ping3, Zhecheng Yuan1,2, Chenhao Lu1, Chengbo Yuan1, Tianhai Liang1, Xiaoyu Tian1, Maanping Shao1, Feihong Zhang1, Mingyu Ding4, Yang Gao1,2, Hao Zhao1, Hang Zhao1,2, Huazhe Xu1,2 1 Tsinghua University 2 Shanghai Qizhi Institute 3 Sun Yat-sen University 4 The University of North Carolina at Chapel Hill ∗Core Contributors †Project Lead https://unidex-ai.github.io/](https://unidex-ai.github.io/)

###### Abstract

Dexterous manipulation remains challenging due to the cost of collecting real-robot teleoperation data, the heterogeneity of hand embodiments, and the high dimensionality of control. We present UniDex, a robot foundation suite that couples a large-scale robot-centric dataset with a unified vision–language–action (VLA) policy and a practical human-data capture setup for universal dexterous hand control. First, we construct UniDex-Dataset, a robot-centric dataset over 50K trajectories across eight dexterous hands (6–24 DoFs), derived from egocentric human video datasets. To transform human data into robot-executable trajectories, we employ a human-in-the-loop retargeting procedure to align fingertip trajectories while preserving plausible hand–object contacts, and we operate on explicit 3D pointclouds with human hands masked to narrow kinematic and visual gaps. Second, we introduce the Function–Actuator–Aligned Space (FAAS), a unified action space that maps functionally similar actuators to shared coordinates, enabling cross-hand transfer. Leveraging FAAS as the action parameterization, we train UniDex-VLA, a 3D VLA policy pretrained on UniDex-Dataset and finetuned with task demonstrations. In addition, we build UniDex-Cap, a simple portable capture setup that records synchronized RGB-D streams and human hand poses and converts them into robot-executable trajectories to enable human–robot data co-training that reduces reliance on costly robot demonstrations. On challenging tool-use tasks across two different hands, UniDex-VLA achieves 81% average task progress and outperforms prior VLA baselines by a large margin, while exhibiting strong spatial, object, and zero-shot cross-hand generalization. Together, UniDex-Dataset, UniDex-VLA, and UniDex-Cap provide a scalable foundation suite for universal dexterous manipulation.

††* Core Contributors: see contribution list [here](#A5).

## 1 Introduction

In recent years, learning from demonstrations [[[13](#bib.bib2), [67](#bib.bib1), [74](#bib.bib13), [58](#bib.bib7), [27](#bib.bib5), [7](#bib.bib3), [6](#bib.bib4)]] has become the de facto paradigm for visuomotor control, enabling robots to acquire complex skills and motion patterns. However, achieving general, human-level manipulation under supervised learning remains challenging. Collecting real-robot demonstrations is labor-intensive and scales poorly, creating a persistent data bottleneck. Moreover, most robot foundation policies focus on parallel-jaw grippers, while foundation models for dexterous hands remain scarce—even though everyday tool-use often requires dexterous hands and many tasks (e.g., using scissors or spray bottles) are infeasible with grippers.

Simply porting gripper-based VLA designs to dexterous hands is insufficient. Building foundation models for dexterous hands is substantially more challenging than for grippers. The key difficulties are: (i) dexterous hand data are harder to collect than gripper data, and large, broadly usable pretraining datasets remain limited; (ii) dexterous hands vary widely in DoFs, morphology, kinematics, and appearance, leading to poor transfer of data and policies across hands; and (iii) dexterous hand control is inherently high-dimensional, demanding expressive action spaces and effective learning algorithms.

To address pretraining data scarcity, we leverage the fact that dexterous robot hands are designed to mimic human hands and often share similar action patterns, while humans naturally generate abundant manipulation data in daily life. Egocentric human videos are cheaper, more diverse than robot teleoperation data and easier to scale. We therefore transform human videos into robot-executable trajectories to build a robot-centric dataset from human activity. However, there are substantial kinematic* and *visual* gaps between human and robot hands. To close these gaps, we (i) introduce a human-in-the-loop retargeting procedure that combines fingertip-based inverse kinematics with interactive adjustment to align robot fingertip trajectories with human trajectories, ensuring physically plausible hand–object contacts; and (ii) mask the human hand in the visual stream and attach the retargeted robot hand into scene pointclouds to reduce visual mismatch.

Following this human-to-robot transformation pipeline, we construct UniDex-Dataset by building on open-source egocentric RGB-D manipulation videos [[[28](#bib.bib18), [4](#bib.bib20), [36](#bib.bib21), [37](#bib.bib19)]]. UniDex-Dataset is a unified foundation dataset comprising 9M paired image–pointcloud–action frames and over 50K trajectories across eight dexterous hand platforms, covering active DoFs from 6 to 24. To our knowledge, UniDex-Dataset is the first dataset to span such a broad spectrum of dexterous hand morphologies at this scale. We also provide protocols that allow researchers to contribute new hands or human datasets with minimal effort, continually scaling UniDex-Dataset and accelerating progress on dexterous manipulation.

To tackle heterogeneous embodiments and high-dimensional control, we further define a unified action space, the Function–Actuator–Aligned Space (FAAS), which maps functionally similar actuators to shared coordinates. FAAS provides a function-centric control interface and enables skill transfer across different hands. Building on FAAS, we train UniDex-VLA, a 3D vision–language–action policy pretrained on UniDex-Dataset and finetuned with task demonstrations, serving as a foundation model that supports diverse dexterous hands.

In addition, we design a portable human-data capture setup, UniDex-Cap, which records synchronized RGB-D streams and human hand poses and converts them into robot-centric trajectories via the same transformation pipeline. UniDex-Cap enables efficient co-training on transformed human data together with smaller amounts of robot data, reducing teleoperation cost while preserving performance.

We evaluate UniDex-VLA on five challenging real-world tool-use tasks across two different hands. Across these tasks, UniDex-VLA achieves strong performance, outperforming other VLA baselines by a large margin (e.g., 81% average task progress vs. $\pi_{0}$ [[[7](#bib.bib3)]] at 38%), and demonstrates strong spatial, object, and cross-hand generalization; with FAAS and pretraining, it transfers skills to unseen hands in a zero-shot manner. Leveraging UniDex-Cap, we also provide a quantitative study showing how transformed human data can reduce post-training costs via human–robot co-training.

Our contributions are summarized as follows:

-
•

UniDex-Dataset: a unified, diverse dexterous hand dataset (9M paired frames, over 50K trajectories, 8 hands, 6–24 DoFs) that supports large-scale pretraining toward universal dexterous hand foundation models.

-
•

FAAS & UniDex-VLA: a function–actuator–aligned unified action space and a pretrained 3D vision–language–action model that achieves state-of-the-art performance on real-robot benchmarks, with strong spatial, object, and cross-hand generalization.

-
•

Human–Robot Data Co-training with UniDex-Cap: a simple portable capture setup and pipeline that support human–robot data co-training; we quantitatively study how transformed human data can partially substitute real-robot demonstrations during post-training, showing that egocentric human videos both scale pretraining and reduce real-robot data needs.

## 2 Related Work

### 2.1 Dexterous Manipulation

Early research on dexterous manipulation was grounded in analytic and classical control formulations [[[43](#bib.bib27), [40](#bib.bib29), [3](#bib.bib30), [26](#bib.bib28), [2](#bib.bib32)]], and has since progressed toward learning-based methods that enable in-hand reorientation, rotation, and grasping [[[45](#bib.bib34), [62](#bib.bib38), [31](#bib.bib36), [10](#bib.bib33), [1](#bib.bib37), [19](#bib.bib35), [75](#bib.bib24), [22](#bib.bib40), [65](#bib.bib93), [76](#bib.bib76), [50](#bib.bib92), [24](#bib.bib81), [54](#bib.bib82), [17](#bib.bib90), [70](#bib.bib91)]]. Despite these advances, most approaches are tailored to specific tasks (grasping) or hardware and struggle to generalize to everyday tool-use. In contrast, we present UniDex-VLA, a foundation model aimed at general-purpose dexterous hand control.

### 2.2 Robot Foundation Policies and Unified Action Space

Diffusion-based policies and their variants constitute strong imitation-learning baselines [[[13](#bib.bib2), [67](#bib.bib1), [58](#bib.bib7), [51](#bib.bib79), [57](#bib.bib87)]]. With the rise of LLMs and VLMs, vision–language–action (VLA) models [[[72](#bib.bib88), [27](#bib.bib5), [7](#bib.bib3), [6](#bib.bib4), [39](#bib.bib83), [23](#bib.bib85), [33](#bib.bib70), [8](#bib.bib72), [71](#bib.bib71), [73](#bib.bib89)]] further scale imitation learning, but most existing approaches are pretrained on large-scale gripper-centric datasets. Recent efforts toward dexterous VLAs [[[75](#bib.bib24), [22](#bib.bib40)]] leverage simulation or limited real-world data, typically focusing on grasping and relying on hand-specific representations. In contrast, UniDex-VLA is pretrained on UniDex-Dataset to serve as a unified foundation policy for more general dexterous manipulation.

Designing a unified action space for robot foundation policies to handle embodiment heterogeneity is crucial for cross-embodiment generalization. RDT-1B [[[33](#bib.bib70)]] preserves the semantic structure of control signals, while $\pi_{0}$ [[[7](#bib.bib3)]] adopts a left-aligned action representation, and other methods introduce latent action spaces [[[71](#bib.bib71), [8](#bib.bib72)]]. However, these approaches primarily target gripper-centric actions. EgoVLA [[[60](#bib.bib8)]] attempts to leverage human parameters as a dexterous representation, but requires inverse kinematics in the post-training stage, which introduces additional errors, particularly for high-DoF dexterous hands. In contrast, FAAS provides a function-centric unified action representation that is post-processing-free, enabling more reliable cross-hand skill transfer.

### 2.3 Learning from Human Videos

Learning from human videos mitigates the data cost bottleneck but introduces visual and kinematic domain gaps. Prior work uses human hand trajectories for planning or control [[[46](#bib.bib26), [52](#bib.bib25), [63](#bib.bib51), [55](#bib.bib52), [34](#bib.bib59), [29](#bib.bib84), [9](#bib.bib86)]]; others apply retargeting with sim-to-real pipelines [[[66](#bib.bib11), [30](#bib.bib31), [11](#bib.bib54)]] or human-in-the-loop corrections [[[53](#bib.bib39)]], and some co-train with robot data [[[25](#bib.bib12), [64](#bib.bib50), [48](#bib.bib10), [77](#bib.bib80)]] to bridge the gap. However, many such pipelines primarily target grippers or do not scale robustly. There are also approaches that pretrained on egocentric human videos without explicit supervision of hand motion [[[41](#bib.bib57), [61](#bib.bib55), [68](#bib.bib58), [42](#bib.bib56)]]. More recent methods pretrain foundation models on egocentric videos to predict human hand motion, followed by specialized post-training to align with robot actions [[[60](#bib.bib8), [38](#bib.bib9)]], however these additional alignment stages can be complex and brittle. Our approach instead generates robot-centric dexterous hand supervision for pretraining, removing the need for specialized alignment tricks during fine-tuning while maintaining cross-hand control.

## 3 UniDex-Dataset

### 3.1 Overview

UniDex-Dataset is derived from four RGB-D egocentric human-manipulation datasets—H2O [[[28](#bib.bib18)]], HOI4D [[[37](#bib.bib19)]], HOT3D [[[4](#bib.bib20)]], and TACO [[[36](#bib.bib21)]]. We annotate language instructions if needed, segment videos into trajectory clips aligned with those instructions, and filter out invalid segments.

*Figure 1: The figure illustrates the complete human–robot transformation pipeline. Starting from the raw scene pointcloud, we first mask out the human hands. We then perform human-in-the-loop retargeting through a user-friendly GUI in which the user only needs to adjust slider bars to modify the dummy base offset. ① shows the retargeted result without adjustment, whereas ③ shows the final configuration with improved, more plausible hand–object contact. Finally, after kinematic retargeting, we attach the retargeted robot dexterous hands to the scene.*

![Figure](2603.22264v1/x2.png)

*Figure 2: UniDex-Dataset visualization. We show a verb–object word cloud and a subset of UniDex-Dataset. Colors denote different hands (arbitrarily assigned; black corresponds to pretraining data). UniDex-Dataset spans diverse everyday tasks across a wide range of dexterous hand embodiments, including using a mobile phone, opening a milk carton, stir-frying with a spatula, lifting a chair, solving a Rubik’s cube, and more.*

The transformation from human data to robot-executable trajectories is illustrated in Fig. [1](#S3.F1) and detailed in the next subsection. Applying this pipeline, we construct UniDex-Dataset comprising 9M paired image–pointcloud–action frames (recorded at 30 fps) and over $50$k trajectories across eight dexterous hand platforms (Inspire, Leap, Shadow, Allegro, Ability, Oymotion, Xhand, and Wuji), covering active DoF from 6 to 24. Figure [2](#S3.F2) visualizes the verb–object word cloud for the dataset and a subset of the data, spanning diverse daily manipulation tasks such as using a mobile phone, opening a milk carton, and stir-frying with a spatula. Table [1](#S3.T1) compares UniDex-Dataset with released collected dexterous manipulation datasets [[[20](#bib.bib63), [35](#bib.bib65), [56](#bib.bib64)]] along the axes of trajectory count, hand variety and scene diversity, and supported perception modalities, highlighting the advantages of UniDex-Dataset. Owing to its diversity and robot-centric formulation—i.e., with minimal embodiment gap to the post-training stage—UniDex-Dataset serves as a strong foundation for pretraining dexterous manipulation models.

### 3.2 Human-Robot Transformation

Transforming human data into robot trajectories requires overcoming two core gaps: kinematic* and *visual*. We outline our methods below.

#### 3.2.1 Kinematic Retargeting

Fingertips are the primary contact points in human–object interaction. Our goal is to align human fingertip trajectories with those of the robot hand in 3D, while allowing a global hand-base adjustment to better ensure physically plausible contact.

Given a human hand pose, we extract $m$ fingertip targets

$$X^{\star}=\big[x_{1}^{\star},\ldots,x_{m}^{\star}\big]\in\mathbb{R}^{3\times m},$$ \tag{1}

where $m$ equals the number of robot fingers. The global human hand transform in the world frame is $T_{\text{hand}}$.

To precisely apply fingertip-based IK while permitting a base adjustment, we introduce a *6-DoF alignment offset*, implemented as a dummy base inserted before the real robot base. Let $T_{\text{offset}}$ be the rigid transform from the dummy base to the real base, and let $T_{\text{world}}^{\text{dummy}}$ be the dummy-base pose in the world frame. The forward kinematics of fingertip $i$ is

$$x_{i}(q;T_{\text{offset}})\;=\;\operatorname{Trans}\!\left(T_{\text{world}}^{\text{dummy}}\,T_{\text{offset}}\,T_{i}(q)\right)\in\mathbb{R}^{3},$$ \tag{2}

where $T_{i}(q)$ is the homogeneous transform from the robot base to fingertip $i$, and $\operatorname{Trans}(\cdot)$ extracts the translation. We set $T_{\text{world}}^{\text{dummy}}=T_{\text{hand}}$ and keep it fixed during optimization. Stacking fingertip residuals yields the IK error:

$$e(q,T_{\text{offset}})=\begin{bmatrix}x_{1}(q;T_{\text{offset}})-x_{1}^{\star}\\ \vdots\\ x_{m}(q;T_{\text{offset}})-x_{m}^{\star}\end{bmatrix}\in\mathbb{R}^{3m}.$$ \tag{3}

*Table 1: Comparison between UniDex-Dataset and other dexterous manipulation datasets. UniDex-Dataset advances in total trajectories, variety across hands/actions/scenes, and supports for all perception modalities. ✗ ✓ denotes the pointcloud in ActionNet [[[20](#bib.bib63)]] is very low-quality.*

For robot hands containing *mimic joint structures* (e.g., Inspire, Oymotion, Agility),
we handle dependent joints through an iterative correction process.
After solving the primary IK problem, each mimic joint $j_{s}$ is updated from its master joint $j_{m}$ as

$$q_{j_{s}}=k\,q_{j_{m}}+c$$ \tag{4}

consistent with the kinematic model specification, where $k$ and $c$ denote the mimic constraints. This correction is repeated for $N$ iterations, re-evaluating fingertip error each time until convergence.

For implementation, we provide a user-friendly and rapid process. The whole pipeline is a two-stage, human-in-the-loop retargeting procedure.

-
1.

Automatic stage.
Given an initial $T_{\text{offset}}$, we solve Eq. [3](#S3.E3) via PyBullet [[[15](#bib.bib67)]]’s
multi-end-effector IK solver to obtain a joint configuration $q$
that minimizes fingertip error while satisfying joint limits and damping.

-
2.

Interactive stage. A lightweight GUI exposes the six degrees of freedom of $T_{\text{offset}}$
(three translations and three rotations, as shown in Fig. [1](#S3.F1)) and other configuration for IK solver.
The user visually inspects alignment and manually adjusts $T_{\text{offset}}$;
after each adjustment, we re-solve the IK problem.
This process typically converges within a few manual tweaks, producing robust
fingertip alignment across diverse poses. ① and ③ in Fig. [1](#S3.F1) shows the comparison between and after the interactive stage.

For each human dataset and each dexterous hand, we perform a basic interactive calibration to select dummy base offsets to handle systematic differences across datasets (e.g., coordinate frames/ hand-pose estimation bias) and hand morphology differences. We then adjust a small subset of frames, focusing on contact-rich segments to improve contact plausibility. In practice, we find the basic calibration suffices to cover the vast majority of trajectories, enabling our transformation pipeline to scale to large egocentric datasets with modest human effort.

#### 3.2.2 Visual Alignment

We compute pointclouds from RGB-D frames. Then to reduce the visual gap, we mask human hands (using WiLoR [[[44](#bib.bib74)]] together with SAM2 [[[49](#bib.bib17)]]) and remove the corresponding points. We then place the retargeted robot-hand mesh into the scene and render its geometry into the pointcloud. Finally, we reproject the fused pointcloud back to the RGB-D frame via a pinhole camera model [[[21](#bib.bib66)]] to avoid occlusions caused by incorrect depth ordering, matching the single-view setting used during real-world fine-tuning.

## 4 UniDex-VLA

### 4.1 Unified Action Space: FAAS

We pretrain our robot foundation model on UniDex-Dataset, which spans diverse dexterous-hand embodiments. A unified action space that enables transfer across hands is therefore critical. To this end, we introduce a simple yet effective action representation, the Function–Actuator–Aligned Space (FAAS). For any dexterous hand with $n$ actuated DoFs in its kinematic model, each *actuator* is mapped to the FAAS *index* corresponding to its functional role. Here we use ”actuator” broadly to denote any controllable DoF/channel derived from the robot URDF, including mimic joints when present.

Conceptually, FAAS exposes a function-centric control interface shared across embodiments rather than a URDF-specific joint space. Although dexterous hands differ in link lengths, couplings, and layouts, they all implement a small set of functional primitives—such as thumb–index pinch, finger curling around handles, or lateral ab-/adduction for stabilization. FAAS groups actuators by these functional roles and maps them into a common coordinate system, discarding embodiment-specific nuisance factors while preserving task-relevant control semantics. Fig. [3](#S4.F3) illustrates, for the thumb and ring fingers of different hands, how individual joints are mapped to FAAS indices.

FAAS is an 82-dimensional action vector. The first 18 dimensions encode wrist poses (9 per hand), where each 9d pose consists of a 6d continuous rotation representation (two 3d vectors for the local $x$- and $y$-axes) followed by a 3d translation. and the remaining 64 dimensions encode joint commands, with 32 slots for each hand. Among these slots, we reserve 21 *base* actuator slots that are shared across all hands, and use the remaining slots for hand-specific DoFs (e.g., additional wrist joints on the Shadow Hand) and for future hands. The details of joint mapping for different hands are shown in Sec. [C](#A3) and Fig. [14](#A3.F14) in Appendix.

*Figure 3: Function–Actuator–Aligned Space (FAAS). We show the thumb and ring fingers of Oymotion (11 actuators), Allegro (16), Inspire (12), and Wuji (20), with colors denoting individual joints, curves indicating rotation directions, and dotted lines indicating rotation axes. Indices {0,1,3,5,6} are aligned across all four hands because the corresponding joints share similar functional roles.*

![Figure](2603.22264v1/x4.png)

*Figure 4: Overview of UniDex-VLA. At time $t$, the model consumes a single-view colored pointcloud $P_{t}$, a language instruction $\ell_{t}$, and proprioception $q_{t}$, and predicts an $H$-step action chunk $A_{t}=[a_{t},\ldots,a_{t+H-1}]$ expressed in the unified action space FAAS. Uni3D [[[78](#bib.bib14)]] encodes the colored pointcloud; features are fused with text and proprioception in the backbone and decoded into FAAS actions. The policy is pretrained on UniDex-Dataset and optimized with a conditional flow-matching objective.*

### 4.2 VLA Policy

UniDex-VLA aims to be a 3D, language-conditioned foundation model for dexterous control. Unlike prior VLAs that pair 2D encoders with low-dimensional gripper actions, our setting is inherently volumetric and high-DoF: tool-use requires reasoning about fine 3D geometry and contact affordances, especially in the egocentric single-view observation. By coupling 3D visual inputs with the unified FAAS action space, UniDex-VLA aligns geometric perception and control in a shared representation, supporting spatial, object, and cross-hand generalization.

#### 4.2.1 Observations and Action Outputs

As shown in Fig. [4](#S4.F4), the observation at time $t$ is
$o_{t}=[P_{t},\,\ell_{t},\,q_{t}]$, where $P_{t}$ is a single-view colored pointcloud derived from an RGB-D image and then cropped and downsampled, $\ell_{t}$ is a natural-language instruction, and $q_{t}$ is a vector of robot proprioceptive states. We model $p(A_{t}\mid o_{t})$, where $A_{t}=[a_{t},\ldots,a_{t+H-1}]$ denotes an $H$-step action chunk [[[74](#bib.bib13)]]. Both $q_{t}$ and each $a_{t}$ are represented in FAAS. For the wrist in $q_{t}$, we use an absolute* pose; for action outputs, we adopt a *relative* wrist pose with respect to the first frame of the action chunk, following UMI [[[14](#bib.bib6)]]. For dexterous-hand joints, we likewise use abstracted representations in both $q_{t}$ and $a_{t}$.

#### 4.2.2 Model Architecture

The UniDex-VLA architecture largely follows $\pi_{0}$ [[[7](#bib.bib3)]], with modifications for pointcloud inputs. Specifically, we replace the SigLIP [[[69](#bib.bib16)]] 2D vision encoder in PaliGemma [[[5](#bib.bib15)]] with Uni3D [[[78](#bib.bib14)]], a strong 3D pointcloud encoder. Uni3D adopts a vanilla ViT [[[18](#bib.bib68)]] design and is initialized from a 2D pretrained ViT, aligning pointcloud features with image–text–aligned features. We train the policy with a conditional flow-matching objective and generate denoised action chunks at inference time via forward–Euler integration [[[32](#bib.bib69)]].

More details of UniDex-VLA training are shown in Sec. [A](#A1) in Appendix.

## 5 Experiments

### 5.1 Experimental Setup

Hardware Platform. Our real-world experiments use a 7-DoF Franka robotic arm equipped with three dexterous end-effectors: an Inspire Hand (6 active, 12 full DoFs), a Wuji Hand (20 active DoFs), and an Oymotion Hand (6 active, 11 full DoFs), all mounted at the end-effector. An Intel RealSense L515 provides egocentric RGB-D observations for all experiments. The complete workstation is shown in Fig. [5](#S5.F5).

*Figure 5: Real-world experiments setup overview*

Task Description. Everyday manipulation commonly involves many tools designed for human hands—e.g., scissors, spray bottles, and sweepers—which impose stringent requirements on finger coordination and in-hand reconfiguration. To better assess the dexterity and generality of our approach, we evaluate five challenging tool-use tasks, with visualization of different stages in Fig. [6](#S5.F6): (i) Make Coffee (Inspire Hand): Grasp the kettle and lift it to the dripper to pour water to make pour-over coffee. Task decomposed into kettle grasping (Grasp) and water pouring (Pour).
(ii) Sweep Objects (Inspire Hand): Grasp a sweeper and sweep tabletop objects into a dustpan. Task decomposed into sweeper grasping (Grasp) and sweeping (Sweep).
(iii) Water Flowers (Wuji Hand): Grasp a spray bottle, lift it, and press the trigger with the thumb to water flowers. Task decomposed into bottle grasping (Grasp) and pressing trigger to water (Press).
(iv) Cut Bags (Wuji Hand): Insert thumb, middle and ring fingers into scissors and grasp them in a human-like manner to cut bags. Task decomposed into scissors grasping (Grasp) and cutting (Cut).
(v) Use Mouse (Wuji Hand): Place fingers on a computer mouse and use it to drag a file into a USB folder in the desktop interface and click the mouse to finish. We report the mean success rate across all task stages as the average task progress, which serves as our primary metric for comparing methods.

![Figure](2603.22264v1/x6.png)

*Figure 6: Our real-robot benchmark comprises 5 challenging tool-use tasks. We visualize the key stages of each task, illustrating the precise dexterous control required to successfully complete them.*

Demonstration Collection. We build our teleoperation system on OpenTeleVision [[[12](#bib.bib47)]] and dex-retargeting [[[47](#bib.bib48)]] with Apple Vision Pro. We only collect 50* demonstrations per task for fine-tuning.

Baselines. We compare UniDex-VLA with representative imitation learning and VLA methods: Diffusion Policy (DP) [[[13](#bib.bib2)]], 3D Diffusion Policy (DP3) [[[67](#bib.bib1)]], and the strong VLA baseline $\pi_{0}$ [[[7](#bib.bib3)]] pretrained on gripper action datasets. To directly assess the effect of pretraining, we include UniDex-VLA (No Pretrain). We adopt FAAS for UniDex-VLA (No Pretrain) and $\pi_{0}$, and retain low-dimensional outputs for DP and DP3.

*Figure 7: Spatial generalization. Left: the kettle and dripper are placed at out-of-distribution (OOD)* positions relative to training demonstrations. Red and green lines circling regions denote the training placement ranges for the kettle and dripper, respectively. Right: average task progress for different methods (10 trials each).*

*Figure 8: Object generalization. Left: we replace the original black kettle with a smaller purple kettle that differs in color, size, and functional parts (handle & spout). Right: average task progress for different methods (10 trials each).*

![Figure](2603.22264v1/x9.png)

*Figure 9: Hand generalization (zero-shot skill transfer). We transfer a policy trained on the Inspire Hand to Wuji and Oymotion. Table reports average task progress (%) under zero-shot deployment (10 trials each).*

### 5.2 Performance

We report results on five real-world manipulation tasks across two dexterous hands at Fig. [10](#S5.F10). The results show that, with only 50 demonstrations per task, UniDex-VLA attains high success rates on these challenging, long-horizon tool-use tasks and surpasses all baselines by a large margin, including on the especially difficult Use Scissors to Cut Bags* task. The performance gap between UniDex-VLA (No-Pretrain) and UniDex-VLA further provides a clear ablation of the benefit of pretraining on UniDex-Dataset. Computing relative improvement over the best competing method (Fig. [10](#S5.F10)), UniDex-VLA achieves the largest gain on the hardest setting, *Use Scissors to Cut Bags*, with an 84.6% increase in average task progress. Overall, these results indicate that pretraining endows UniDex-VLA with strong motion priors for dexterous hand control, particularly on highly dexterous tool-use tasks, enabling more efficient adaptation to new and challenging behaviors.

*Figure 10: Average task progress across five real-world tasks (top), with aggregate averages of average task progress and final success rate (bottom) over 5 tasks. Each task/algorithm uses 20 trials.*

### 5.3 Generalization

Beyond outperforming performance, UniDex-VLA demonstrates strong spatial, object, and hand generalization.

Spatial Generalization.
UniDex-VLA benefits from 3D perception, and pointclouds further enable simple, automatic data augmentation via geometric editing. In the Make Coffee experiment, we segment the pointclouds of the kettle and the dripper, and translate them along the table’s $x$/$y$ axes to sweep across the workspace and generate out-of-distribution (o.o.d.) placements. After editing the pointclouds, the corresponding robot states are aligned to the new scenes using Task and Motion Planning (TAMP) [[[16](#bib.bib49)]]. DemoGen [[[59](#bib.bib23)]] provides an automated pipeline for this procedure. As shown in Fig. [7](#S5.F7), UniDex-VLA generalizes well across spatial configurations; with DemoGen [[[59](#bib.bib23)]] augmentation, it approaches very high success rate over full workspace.

Object Generalization.
As in Fig. [8](#S5.F8), we replace the black kettle with a smaller purple kettle that differs in color, size, and functional parts (handle & spout). UniDex-VLA maintains strong performance on this unseen object, indicating generalizable tool understanding capacity crucial for robust and general tool-use.

Hand Generalization (Skill Transfer).
We evaluate cross-hand transfer by taking a policy trained to Make Coffee* on the Inspire Hand (6 active DoF) and deploying it *zero-shot* on Wuji (20 active DoFs) and Oymotion (6 active DoFs with different kinematics). As shown in Fig. [9](#S5.F9), UniDex-VLA achieves 60% success on Oymotion and 40% on Wuji without any fine-tuning, whereas baselines are near zero. These results highlight that pretraining across diverse dexterous hands—together with FAAS—indeed enables zero-shot cross-hand skill transfer.

### 5.4 UniDex-Cap for Human-Robot Data Co-train

We introduce UniDex-Cap, a practical data-capture setup that records synchronized RGB-D streams and hand/head poses. The system combines an Apple Vision Pro for hand and head pose estimation, an Intel RealSense L515 for high-quality RGB-D, and a custom 3D-printed mount to physically couples the two sensors with a fixed rigid transform. This transform is calibrated to ensure the RGB-D stream and the hand/head poses are time-synchronized and expressed in the shared coordinate frame. As illustrated in Fig. [11](#S5.F11), we then apply the human-to-robot transformation pipeline (Sec. [3.2](#S3.SS2)) to convert captured human data into robot-executable trajectories. In addition, we perform a viewpoint transformation to align human and robot perspectives and downsample the human motion to match typical teleoperation speeds.

*Figure 11: (a,b) show the components of UniDex-Cap. (c,d) shows the example captured data and converted robot-executable trajectories.*

Leveraging UniDex-Cap, we collect human demonstrations, transform them, and co-train* with real-robot data on *Make Coffee* task to quantitatively explore the effect of human demos during the finetuning stage. Figure [12](#S5.F12) reports average task progress versus the numbers of co-trained transformed human demos ($h$) and robot demos. We observe: (i) Retargeted human data helps, but robot data is indispensable. Although for a fixed $r$, increasing $h$ consistently improves average task progress within our evaluated range but success always remains near zero without any robot data.
(ii) Human–robot exchange rate $\approx$ 2:1. From Fig. [12](#S5.F12), the boundary separating the ”high-performance” region (comparable to the $r{=}50$ robot-only result green area) has slope $\approx 2$, suggesting roughly *two human demos can substitute for one robot demo*.
(iii) Cost efficiency. On *Make Coffee* task, human demos are $\sim$5.2$\times$ faster to collect than real robot demos; considering the $\approx$2:1 exchange rate, co-training with human demos can substantially reduce data collection cost.

*Figure 12: Human-Robot co-training. Average task progress versus the numbers of transformed human demos ($h$) and robot demos ($r$). Colors indicate different performance bands (green: comparable to the $r{=}50$ robot-only result). Each point averages over 20 trials.*

## 6 Conclusion and Limitation

We presented UniDex, a robot foundation suite built from egocentric human videos, comprising UniDex-Dataset, UniDex-VLA, and UniDex-Cap. We believe UniDex can serve as a practical foundation platform for the community, accelerating progress toward general, scalable, and transferable dexterous manipulation. A limitation of our current work is that we do not yet leverage large action-free* (or weakly labeled) egocentric activity datasets; extending UniDex to incorporate such data is a promising direction for further scaling dexterous pretraining.

## 7 Acknowledgment

We would like to give special thanks to Wuji Technology Inc. for
providing the hardware support, and Hojin Bae, Haoxu Huang, Shaoting Zhu for their technical support. Tsinghua University Dushi Program supports this project.

## References

-
[1]
I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. (2019)

Solving rubik’s cube with a robot hand.

arXiv preprint arXiv:1910.07113.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[2]
S. Arimoto (2004)

Intelligent control of multi-fingered hands.

Annual Reviews in Control 28 (1), pp. 75–85.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[3]
Y. Bai and C. K. Liu (2014)

Dexterous manipulation using both palm and fingers.

In 2014 IEEE International Conference on Robotics and Automation (ICRA),

pp. 1560–1565.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[4]
P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol, R. Newcombe, R. Wang, J. J. Engel, and T. Hodan (2025)

HOT3D: hand and object tracking in 3D from egocentric multi-view videos.

CVPR.

Cited by: [§1](#S1.p4.1),
[§3.1](#S3.SS1.p1.1).

-
[5]
L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. (2024)

Paligemma: a versatile 3b vlm for transfer.

arXiv preprint arXiv:2407.07726.

Cited by: [§4.2.2](#S4.SS2.SSS2.p1.1).

-
[6]
J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)

Gr00t n1: an open foundation model for generalist humanoid robots.

arXiv preprint arXiv:2503.14734.

Cited by: [§1](#S1.p1.1),
[§2.2](#S2.SS2.p1.1).

-
[7]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)

$\pi_{0}$: A vision-language-action flow model for general robot control.

arXiv preprint arXiv:2410.24164.

Cited by: [§A.4](#A1.SS4.p1.1),
[§A.4](#A1.SS4.p3.1),
[§1](#S1.p1.1),
[§1](#S1.p7.1),
[§2.2](#S2.SS2.p1.1),
[§2.2](#S2.SS2.p2.1),
[§4.2.2](#S4.SS2.SSS2.p1.1),
[§5.1](#S5.SS1.p4.2).

-
[8]
Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li (2025)

Univla: learning to act anywhere with task-centric latent actions.

arXiv preprint arXiv:2505.06111.

Cited by: [§2.2](#S2.SS2.p1.1),
[§2.2](#S2.SS2.p2.1).

-
[9]
H. Chen, B. Sun, A. Zhang, M. Pollefeys, and S. Leutenegger (2025)

VidBot: learning generalizable 3d actions from in-the-wild 2d human videos for zero-shot robotic manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 27661–27672.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[10]
T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal (2023)

Visual dexterity: in-hand reorientation of novel and complex object shapes.

Science Robotics 8 (84), pp. eadc9244.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[11]
Y. Chen, C. Wang, Y. Yang, and C. K. Liu (2024)

Object-centric dexterous manipulation from human motion data.

arXiv preprint arXiv:2411.04005.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[12]
X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang (2024)

Open-television: teleoperation with immersive active visual feedback.

arXiv preprint arXiv:2407.01512.

Cited by: [§5.1](#S5.SS1.p3.1).

-
[13]
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2023)

Diffusion policy: visuomotor policy learning via action diffusion.

The International Journal of Robotics Research, pp. 02783649241273668.

Cited by: [§A.4](#A1.SS4.p1.1),
[§A.4](#A1.SS4.p2.1),
[§1](#S1.p1.1),
[§2.2](#S2.SS2.p1.1),
[§5.1](#S5.SS1.p4.2).

-
[14]
C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song (2024)

Universal manipulation interface: in-the-wild robot teaching without in-the-wild robots.

arXiv preprint arXiv:2402.10329.

Cited by: [§4.2.1](#S4.SS2.SSS1.p1.13).

-
[15]
E. Coumans and Y. Bai (2016)

Pybullet, a python module for physics simulation for games, robotics and machine learning.

Cited by: [item 1](#S3.I1.i1.p1.2).

-
[16]
M. Dalal, A. Mandlekar, C. Garrett, A. Handa, R. Salakhutdinov, and D. Fox (2023)

Imitating task and motion planning with visuomotor transformers.

arXiv preprint arXiv:2305.16309.

Cited by: [§5.3](#S5.SS3.p2.2).

-
[17]
K. Ding, B. Chen, R. Wu, Y. Li, Z. Zhang, H. Gao, S. Li, G. Zhou, Y. Zhu, H. Dong, et al. (2024)

Preafford: universal affordance-based pre-grasping for diverse objects and environments.

In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

pp. 7278–7285.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[18]
A. Dosovitskiy (2020)

An image is worth 16x16 words: transformers for image recognition at scale.

arXiv preprint arXiv:2010.11929.

Cited by: [§4.2.2](#S4.SS2.SSS2.p1.1).

-
[19]
H. Fang, H. Yan, Z. Tang, H. Fang, C. Wang, and C. Lu (2025)

AnyDexGrasp: general dexterous grasping for different hands with human-level learning efficiency.

arXiv preprint arXiv:2502.16420.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[20]
Y. M. Fourier ActionNet Team (2025)

ActionNet: a dataset for dexterous bimanual manipulation.

Cited by: [§3.1](#S3.SS1.p2.1),
[Table 1](#S3.T1),
[Table 1](#S3.T1.2.3.1).

-
[21]
R. Hartley (2003)

Multiple view geometry in computer vision.

Vol. 665, Cambridge university press.

Cited by: [§3.2.2](#S3.SS2.SSS2.p1.1).

-
[22]
J. He, D. Li, X. Yu, Z. Qi, W. Zhang, J. Chen, Z. Zhang, Z. Zhang, L. Yi, and H. Wang (2025)

DexVLG: dexterous vision-language-grasp model at scale.

arXiv preprint arXiv:2507.02747.

Cited by: [§2.1](#S2.SS1.p1.1),
[§2.2](#S2.SS2.p1.1).

-
[23]
Y. Ji, H. Tan, J. Shi, X. Hao, Y. Zhang, H. Zhang, P. Wang, M. Zhao, Y. Mu, P. An, et al. (2025)

Robobrain: a unified brain model for robotic manipulation from abstract to concrete.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 1724–1734.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[24]
J. Jian, X. Liu, Z. Chen, M. Li, J. Liu, and R. Hu (2025)

G-dexgrasp: generalizable dexterous grasping synthesis via part-aware prior retrieval and prior-assisted generation.

arXiv preprint arXiv:2503.19457.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[25]
S. Kareer, D. Patel, R. Punamiya, P. Mathur, S. Cheng, C. Wang, J. Hoffman, and D. Xu (2025)

Egomimic: scaling imitation learning via egocentric video.

In 2025 IEEE International Conference on Robotics and Automation (ICRA),

pp. 13226–13233.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[26]
J. Kerr and B. Roth (1986)

Analysis of multifingered hands.

The International Journal of Robotics Research 4 (4), pp. 3–17.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[27]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)

Openvla: an open-source vision-language-action model.

arXiv preprint arXiv:2406.09246.

Cited by: [§1](#S1.p1.1),
[§2.2](#S2.SS2.p1.1).

-
[28]
T. Kwon, B. Tekin, J. Stühmer, F. Bogo, and M. Pollefeys (2021)

H2o: two hands manipulating objects for first person interaction recognition.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 10138–10148.

Cited by: [§1](#S1.p4.1),
[§3.1](#S3.SS1.p1.1).

-
[29]
G. Li, N. Tsagkas, J. Song, R. Mon-Williams, S. Vijayakumar, K. Shao, and L. Sevilla-Lara (2025)

Learning precise affordances from egocentric videos for robotic manipulation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 10581–10591.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[30]
K. Li, P. Li, T. Liu, Y. Li, and S. Huang (2025)

Maniptrans: efficient dexterous bimanual manipulation transfer via residual learning.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 6991–7003.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[31]
T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik (2025)

Learning visuotactile skills with two multifingered hands.

In 2025 IEEE International Conference on Robotics and Automation (ICRA),

pp. 5637–5643.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[32]
Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2022)

Flow matching for generative modeling.

arXiv preprint arXiv:2210.02747.

Cited by: [§4.2.2](#S4.SS2.SSS2.p1.1).

-
[33]
S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu (2024)

Rdt-1b: a diffusion foundation model for bimanual manipulation.

arXiv preprint arXiv:2410.07864.

Cited by: [§2.2](#S2.SS2.p1.1),
[§2.2](#S2.SS2.p2.1).

-
[34]
V. Liu, A. Adeniji, H. Zhan, S. Haldar, R. Bhirangi, P. Abbeel, and L. Pinto (2025)

Egozero: robot learning from smart glasses.

arXiv preprint arXiv:2505.20290.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[35]
Y. Liu, Y. Yang, Y. Wang, X. Wu, J. Wang, Y. Yao, S. Schwertfeger, S. Yang, W. Wang, J. Yu, et al. (2024)

Realdex: towards human-like grasping for robotic dexterous hand.

arXiv preprint arXiv:2402.13853.

Cited by: [§3.1](#S3.SS1.p2.1),
[Table 1](#S3.T1.2.5.1).

-
[36]
Y. Liu, H. Yang, X. Si, L. Liu, Z. Li, Y. Zhang, Y. Liu, and L. Yi (2024)

Taco: benchmarking generalizable bimanual tool-action-object understanding.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 21740–21751.

Cited by: [§1](#S1.p4.1),
[§3.1](#S3.SS1.p1.1).

-
[37]
Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi (2022)

Hoi4d: a 4d egocentric dataset for category-level human-object interaction.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 21013–21022.

Cited by: [§1](#S1.p4.1),
[§3.1](#S3.SS1.p1.1).

-
[38]
H. Luo, Y. Feng, W. Zhang, S. Zheng, Y. Wang, H. Yuan, J. Liu, C. Xu, Q. Jin, and Z. Lu (2025)

Being-h0: vision-language-action pretraining from large-scale human videos.

arXiv preprint arXiv:2507.15597.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[39]
C. Miao, T. Chang, M. Wu, H. Xu, C. Li, M. Li, and X. Wang (2025)

Fedvla: federated vision-language-action learning with dual gating mixture-of-experts for robotic manipulation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 6904–6913.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[40]
I. Mordatch, Z. Popović, and E. Todorov (2012)

Contact-invariant optimization for hand manipulation.

In Proceedings of the ACM SIGGRAPH/Eurographics symposium on computer animation,

pp. 137–144.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[41]
S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta (2022)

R3m: a universal visual representation for robot manipulation.

arXiv preprint arXiv:2203.12601.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[42]
D. Niu, Y. Sharma, H. Xue, G. Biamby, J. Zhang, Z. Ji, T. Darrell, and R. Herzig (2025)

Pre-training auto-regressive robotic models with 4d representations.

arXiv preprint arXiv:2502.13142.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[43]
J. Ponce, S. Sullivan, A. Sudsang, J. Boissonnat, and J. Merlet (1997)

On computing four-finger equilibrium and force-closure grasps of polyhedral objects.

The International Journal of Robotics Research 16 (1), pp. 11–35.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[44]
R. A. Potamias, J. Zhang, J. Deng, and S. Zafeiriou (2025)

Wilor: end-to-end 3d hand localization and reconstruction in-the-wild.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12242–12254.

Cited by: [§3.2.2](#S3.SS2.SSS2.p1.1).

-
[45]
H. Qi, B. Yi, S. Suresh, M. Lambeta, Y. Ma, R. Calandra, and J. Malik (2023)

General in-hand object rotation with vision and touch.

In Conference on Robot Learning,

pp. 2549–2564.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[46]
Y. Qin, Y. Wu, S. Liu, H. Jiang, R. Yang, Y. Fu, and X. Wang (2022)

Dexmv: imitation learning for dexterous manipulation from human videos.

In European Conference on Computer Vision,

pp. 570–587.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[47]
Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y. Chao, and D. Fox (2023)

AnyTeleop: a general vision-based dexterous robot arm-hand teleoperation system.

In Robotics: Science and Systems,

Cited by: [§5.1](#S5.SS1.p3.1).

-
[48]
R. Qiu, S. Yang, X. Cheng, C. Chawla, J. Li, T. He, G. Yan, D. J. Yoon, R. Hoque, L. Paulsen, et al. (2025)

Humanoid policy˜ human policy.

arXiv preprint arXiv:2503.13441.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[49]
N. Ravi, V. Gabeur, Y. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al. (2024)

Sam 2: segment anything in images and videos.

arXiv preprint arXiv:2408.00714.

Cited by: [§3.2.2](#S3.SS2.SSS2.p1.1).

-
[50]
Z. Si, G. Zhang, Q. Ben, B. Romero, Z. Xian, C. Liu, and C. Gan (2024)

Difftactile: a physics-based differentiable tactile simulator for contact-rich robotic manipulation.

arXiv preprint arXiv:2403.08716.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[51]
J. Tian, L. Wang, S. Zhou, S. Wang, J. Li, H. Sun, and W. Tang (2025)

PDFactor: learning tri-perspective view policy diffusion field for multi-task robotic manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 15757–15767.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[52]
C. Wang, L. Fan, J. Sun, R. Zhang, L. Fei-Fei, D. Xu, Y. Zhu, and A. Anandkumar (2023)

Mimicplay: long-horizon imitation learning by watching human play.

arXiv preprint arXiv:2302.12422.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[53]
C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu (2024)

Dexcap: scalable and portable mocap data collection system for dexterous manipulation.

arXiv preprint arXiv:2403.07788.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[54]
Y. Wang, J. Ye, C. Xiao, Y. Zhong, H. Tao, H. Yu, Y. Liu, J. Yu, and Y. Ma (2025)

DexH2R: a benchmark for dynamic dexterous grasping in human-to-robot handover.

arXiv preprint arXiv:2506.23152.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[55]
C. Wen, X. Lin, J. So, K. Chen, Q. Dou, Y. Gao, and P. Abbeel (2023)

Any-point trajectory modeling for policy learning.

arXiv preprint arXiv:2401.00025.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[56]
K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y. Zhao, Z. Xu, G. Yang, et al. (2024)

Robomind: benchmark on multi-embodiment intelligence normative data for robot manipulation.

arXiv preprint arXiv:2412.13877.

Cited by: [§3.1](#S3.SS1.p2.1),
[Table 1](#S3.T1.2.4.1).

-
[57]
H. Xu, J. Ding, J. Xu, R. Wang, J. Chen, J. Mai, Y. Fu, B. Ghanem, F. Xu, and M. Elhoseiny (2025)

Diffusion-based imaginative coordination for bimanual manipulation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 11469–11479.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[58]
H. Xue, J. Ren, W. Chen, G. Zhang, Y. Fang, G. Gu, H. Xu, and C. Lu (2025)

Reactive diffusion policy: slow-fast visual-tactile policy learning for contact-rich manipulation.

arXiv preprint arXiv:2503.02881.

Cited by: [§1](#S1.p1.1),
[§2.2](#S2.SS2.p1.1).

-
[59]
Z. Xue, S. Deng, Z. Chen, Y. Wang, Z. Yuan, and H. Xu (2025)

Demogen: synthetic demonstration generation for data-efficient visuomotor policy learning.

arXiv preprint arXiv:2502.16932.

Cited by: [§A.3](#A1.SS3.p1.2),
[§5.3](#S5.SS3.p2.2).

-
[60]
R. Yang, Q. Yu, Y. Wu, R. Yan, B. Li, A. Cheng, X. Zou, Y. Fang, X. Cheng, R. Qiu, et al. (2025)

Egovla: learning vision-language-action models from egocentric human videos.

arXiv preprint arXiv:2507.12440.

Cited by: [§2.2](#S2.SS2.p2.1),
[§2.3](#S2.SS3.p1.1).

-
[61]
S. Ye, J. Jang, B. Jeon, S. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y. Chao, B. Y. Lin, et al. (2024)

Latent action pretraining from videos.

arXiv preprint arXiv:2410.11758.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[62]
Z. Yin, B. Huang, Y. Qin, Q. Chen, and X. Wang (2023)

Rotating without seeing: towards in-hand dexterity through touch.

arXiv preprint arXiv:2303.10880.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[63]
C. Yuan, C. Wen, T. Zhang, and Y. Gao (2024)

General flow as foundation affordance for scalable robot learning.

arXiv preprint arXiv:2401.11439.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[64]
C. Yuan, R. Zhou, M. Liu, Y. Hu, S. Wang, L. Yi, C. Wen, S. Zhang, and Y. Gao (2025)

Motiontrans: human vr data enable motion-level learning for robotic manipulation policies.

arXiv preprint arXiv:2509.17759.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[65]
Z. Yuan, T. Wei, S. Cheng, G. Zhang, Y. Chen, and H. Xu (2024)

Learning to manipulate anywhere: a visual generalizable framework for reinforcement learning.

arXiv preprint arXiv:2407.15815.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[66]
Z. Yuan, T. Wei, L. Gu, P. Hua, T. Liang, Y. Chen, and H. Xu (2025)

Hermes: human-to-robot embodied learning from multi-source motion data for mobile dexterous manipulation.

arXiv preprint arXiv:2508.20085.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[67]
Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu (2024)

3d diffusion policy: generalizable visuomotor policy learning via simple 3d representations.

arXiv preprint arXiv:2403.03954.

Cited by: [§A.4](#A1.SS4.p1.1),
[§A.4](#A1.SS4.p2.1),
[§1](#S1.p1.1),
[§2.2](#S2.SS2.p1.1),
[§5.1](#S5.SS1.p4.2).

-
[68]
J. Zeng, Q. Bu, B. Wang, W. Xia, L. Chen, H. Dong, H. Song, D. Wang, D. Hu, P. Luo, et al. (2024)

Learning manipulation by predicting interaction.

arXiv preprint arXiv:2406.00439.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[69]
X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer (2023)

Sigmoid loss for language image pre-training.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 11975–11986.

Cited by: [§4.2.2](#S4.SS2.SSS2.p1.1).

-
[70]
G. Zhang, H. Fang, H. Fang, and C. Lu (2023)

Flexible handover with real-time robust dynamic grasp trajectory generation.

In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

pp. 3192–3199.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[71]
Y. Zhang, C. Wang, O. Lu, Y. Zhao, Y. Ge, Z. Sun, X. Li, C. Zhang, C. Bai, and X. Li (2025)

Align-then-steer: adapting the vision-language action models through unified latent guidance.

arXiv preprint arXiv:2509.02055.

Cited by: [§2.2](#S2.SS2.p1.1),
[§2.2](#S2.SS2.p2.1).

-
[72]
Z. Zhang, H. Xu, Z. Yang, C. Yue, Z. Lin, H. Gao, Z. Wang, and H. Zhao (2025)

Elucidating the design space of torque-aware vision-language-action models.

In 9th Annual Conference on Robot Learning,

Cited by: [§2.2](#S2.SS2.p1.1).

-
[73]
Z. Zhang, C. Yue, H. Xu, M. Liao, X. Qi, H. Gao, Z. Wang, and H. Zhao (2025)

RoboChemist: long-horizon and safety-compliant robotic chemical experimentation.

arXiv preprint arXiv:2509.08820.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[74]
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)

Learning fine-grained bimanual manipulation with low-cost hardware.

arXiv preprint arXiv:2304.13705.

Cited by: [§1](#S1.p1.1),
[§4.2.1](#S4.SS2.SSS1.p1.13).

-
[75]
Y. Zhong, X. Huang, R. Li, C. Zhang, Z. Chen, T. Guan, F. Zeng, K. N. Lui, Y. Ye, Y. Liang, et al. (2025)

Dexgraspvla: a vision-language-action framework towards general dexterous grasping.

arXiv preprint arXiv:2502.20900.

Cited by: [§2.1](#S2.SS1.p1.1),
[§2.2](#S2.SS2.p1.1).

-
[76]
Y. Zhong, Q. Jiang, J. Yu, and Y. Ma (2025)

Dexgrasp anything: towards universal robotic dexterous grasping with physics awareness.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 22584–22594.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[77]
J. Zhou, T. Ma, K. Lin, Z. Wang, R. Qiu, and J. Liang (2025)

Mitigating the human-robot domain discrepancy in visual pre-training for robotic manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 22551–22561.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[78]
J. Zhou, J. Wang, B. Ma, Y. Liu, T. Huang, and X. Wang (2023)

Uni3d: exploring unified 3d representation at scale.

arXiv preprint arXiv:2310.06773.

Cited by: [Figure 4](#S4.F4),
[Figure 4](#S4.F4.12.6.6),
[§4.2.2](#S4.SS2.SSS2.p1.1).

Appendix

## Appendix A Training Details

### A.1 UniDex-VLA Flow-Matching Loss

To train UniDex-VLA, we minimize a conditional flow-matching loss:

$$L^{\tau}(\theta)=\mathbb{E}_{p(A_{t}\mid o_{t}),\,q(A_{t}^{\tau}\mid A_{t})}\left[\bigl\|v_{\theta}(A_{t}^{\tau},o_{t})-u(A_{t}^{\tau}\mid A_{t})\bigr\|\right],$$ \tag{5}

where $\tau\in[0,1]$ and $q(A_{t}^{\tau}\mid A_{t})=\mathcal{N}\!\bigl(\tau A_{t},\;(1-\tau)I\bigr)$ is a linear-Gaussian probability path. We sample
$A_{t}^{\tau}=\tau A_{t}+(1-\tau)\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$ and compute the target conditional vector field
$u(A_{t}^{\tau}\mid A_{t})=A_{t}-\epsilon$. The network is trained such that the predicted vector field $v_{\theta}(A_{t}^{\tau},o_{t})$ approximates $u(A_{t}^{\tau}\mid A_{t})$.

At inference time, we integrate the learned vector field using a forward Euler scheme to generate a denoised action chunk:

| | $$A_{t}^{\tau+\delta}\;=\;A_{t}^{\tau}+\delta\,v_{\theta}\!\left(A_{t}^{\tau},\,o_{t}\right),$$ | |
|---|---|---|

with step size $\delta=0.1$ and initial condition $A_{t}^{0}\sim\mathcal{N}(0,I)$.

### A.2 UniDex-VLA Pretraining

During pre-training, we use 8 NVIDIA H800 GPUs with a total batch size of 128. The model used for subsequent post-training is trained for 3 epochs ($\sim$ 30k steps), which takes around 24 hours. We adopt the AdamW optimizer and a cosine learning-rate scheduler with an initial learning rate of 1e-4. The learning rate is decayed by a factor of 0.95 at the 2nd epoch. The weight decay is set to 1e-10, and we apply gradient clipping with a maximum norm of 1.0.

### A.3 UniDex-VLA Post-training

During post-training, we use 2 NVIDIA H800 GPUs for each task, with a total batch size of 8. We use the AdamW optimizer without a learning rate scheduler and set the initial learning rate to 2.5e-5. For common data, we train the model for 50 epochs ($\sim$ 3k steps), which takes around 4 hours. For DemoGen [[[59](#bib.bib23)]] augmented data, we train the model for 2 epochs ($\sim$ 1.8k steps), which takes around 2.5 hours. The weight decay is set to 1e-10, and we again use gradient clipping with a maximum norm of 1.0.

### A.4 Baselines

For all baselines (DP [[[13](#bib.bib2)]], DP3 [[[67](#bib.bib1)]], and $\pi_{0}$ [[[7](#bib.bib3)]]), we post-train the models until convergence on the validation set.

For DP [[[13](#bib.bib2)]] and DP3 [[[67](#bib.bib1)]], we use the AdamW optimizer with an initial learning rate of 1e-4. The state horizon is set to 4 and the action horizon to 32. We use a batch size of 32 and train for 400 epochs.

For $\pi_{0}$ [[[7](#bib.bib3)]], we use the AdamW optimizer with an initial learning rate of 2.5e-5. The batch size is set to 8 and the model is trained for 50 epochs. The number of diffusion steps is set to 10.

For our UniDex-VLA baseline without pretraining, we use the same training hyperparameters as UniDex-VLA with pretraining.

## Appendix B Human-in-the-loop Retargeting GUI

To minimum human efforts in our human-in-the-loop retargeting process, we develop a human-friendly web-based GUI, as shown in [Fig. 13](#A2.F13). Through this interface, users can adjust dummy base links, IK parameters, and other retargeting settings to obtain satisfactory robot trajectories.

![Figure](2603.22264v1/Pictures/web_gui.png)

*Figure 13: Human-friendly web-basedGUI for retargeting human demonstrations to robot executions. Users can adjust the IK parameters, dummy links, and other settings through the GUI to obtain satisfactory retargeted robot trajectories.*

## Appendix C FAAS Details

Here we show the details for the 32 dimensions encoding dexterous hand joints. Dimensions 0–4, 5–9, 10–14, 15–19, and 20–24 correspond to the thumb, index, middle, ring, and little fingers, respectively. Dimensions 25–26 are reserved for extra wrist joints of Shadow hands. Dimensions 27–31 are left unused for new hands. The detailed joint mappings of the robotic hands used in FAAS are shown in [Fig. 14](#A3.F14).

![Figure](2603.22264v1/Pictures/detailed_faas.png)

*Figure 14: Joint mappings of different robotic hands used in FAAS. From left to right are Ability, Allegro, Inspire, Leap, Oymotion, Shadow, Wuji, and Xhand. The two rows show different views of the joint mappings on the right hand.*

## Appendix D UniDex-Cap Setup Calibration

UniDex-Cap combines an Apple Vision Pro (for hand and head poses, denoted $\{P_{\mathrm{VP}}\}$) and an Intel RealSense L515 (for RGB-D). Because Vision Pro does not expose third-party RGB-D video recording, we physically couple the two sensors with a custom 3D-printed mount that rigidly fixes their relative pose. This mechanical constraint ensures that the extrinsic transform between the Vision Pro and the RealSense remains stable for a given user.

As shown in Fig. [15](#A4.F15), we provide a lightweight GUI to estimate the remaining constant extrinsics with minimal manual effort. The user records a short calibration clip and then uses a slider-based interface to adjust the hand and wrist poses in the Vision Pro coordinate frame—visualized as a skeleton—until they align with the 3D hand point cloud captured by the RealSense camera. The slider values directly correspond to the transform $T^{\mathrm{VP}}_{\mathrm{RS}}$. Once this transform is determined, all Vision Pro poses are converted into the RealSense camera frame, yielding temporally aligned hand and head trajectories:

$$P_{\mathrm{RS}}\;=\;T^{\mathrm{VP}}_{\mathrm{RS}}\,P_{\mathrm{VP}},$$ \tag{6}

where $P_{\mathrm{VP}}$ and $P_{\mathrm{RS}}$ are represented in homogeneous coordinates. This pipeline produces temporally synchronized, geometrically consistent annotations suitable for downstream retargeting and post-training.

![Figure](2603.22264v1/x13.png)

*Figure 15: GUI for UniDex-Cap calibration. (a) shows the initial state before calibration; (b) shows the calibrated result where the hand poses captured by Vision Pro align with the 3D point cloud captured by the RealSense L515 camera.*

## Appendix E Core Contribution List

The main contributions of the core contributors are as follows:

Gu Zhang: Project lead. Developed the overall dataset construction pipeline, model architecture, and unified action space; built the robot system infrastructure; and wrote the paper.

Qicheng Xu: Led VLA model training; optimized the dataset construction and policy inference pipelines; and contributed to paper writing.

Haozhe Zhang: Led dataset processing; improved the robot system and the human–robot data capture pipeline; and contributed to paper writing.

Jianhan Ma: Implemented retargeting algorithms; and developed dataset visualizations and contributed to early-stage exploration.

Long He: Implemented DemoGen algorithm; and contributed to paper writing.

Yiming Bao: Collected robot data and human data; and contributed to DemoGen algorithm implementation.



Experimental support, please
[view the build logs](./2603.22264v1/__stdout.txt)
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