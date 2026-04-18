##### Report GitHub Issue

**×




Title:



Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.22193v3 [cs.CV] 02 Apr 2026

[# PAM: A Pose–Appearance–Motion Engine for Sim-to-Real HOI Video Generation Mingju Gao*1,2, Kaisen Yang*2, Huan-ang Gao2, Bohan Li4,5, Ao Ding2, Wenyi Li2, Yangcheng Yu2, Jinkun Liu2, Shaocong Xu3, Yike Niu2, Haohan Chi2, Hao Chen6, Hao Tang1, Yu Zhang†4, Li Yi2, Hao Zhao†2,3 1Peking University 2Tsinghua University 3BAAI 4SJTU 5Eastern Institute of Technology 6University of Cambridge Project Page: https://gasaiyu.github.io/PAM.github.io/](https://gasaiyu.github.io/PAM.github.io/)

###### Abstract

Hand–object interaction (HOI) reconstruction and synthesis are becoming central to embodied AI and AR/VR. Yet, despite rapid progress, existing HOI generation research remains fragmented across three disjoint tracks: (1) pose-only synthesis that predicts MANO trajectories without producing pixels; (2) single-image HOI generation that hallucinates appearance from masks or 2D cues but lacks dynamics; and (3) video generation methods that require both the entire pose sequence and the ground-truth first frame as inputs, preventing true sim-to-real deployment. Inspired by the philosophy of [joo2018total], we think that HOI generation requires a unified engine that brings together pose, appearance, and motion within one coherent framework. Thus we introduce PAM: a Pose–Appearance–Motion Engine for controllable HOI video generation. The performance of our engine is validated by: (1) On DexYCB, we obtain an FVD of 29.13 (vs. 38.83 for InterDyn), and MPJPE of 19.37 mm (vs. 30.05 mm for CosHand), while generating higher-resolution 480×720 videos compared to 256×256/256×384 baselines. (2) On OAKINK2, our full multi-condition model improves FVD from 68.76 → 46.31. (3) An ablation over input conditions on DexYCB shows that combining depth, segmentation, and keypoints consistently yields the best results. (4) For a downstream hand pose estimation task using SimpleHand, augmenting training with 3,400 synthetic videos (207k frames) allows a model trained on only 50% of the real data plus our synthetic data to match the 100% real baseline.

00footnotetext: *Equal Contribution. †Corresponding Author.

## 1 Introduction

![Figure](2603.22193v3/x1.png)

*Figure 1: Overview of Four Approaches to HOI Synthesis. (a) Pose-Only Synthesis [[[78](#bib.bib30)]]: This method predicts the MANO trajectories without generating pixel data; (b) Apperance Generation [[[82](#bib.bib54)]]: This approach generates appearance based on masks or 2D cues but lacks dynamic motion; (c) Motion Generation [[[47](#bib.bib57), [3](#bib.bib58)]]: These methods require both the full pose sequence and the ground-truth first frame as inputs, limiting their application for true sim-to-real transfer. (d) Our Pipeline PAM: In this approach, video generation does not rely on the first frame or the whole HOI pose sequence, allowing for the transfer of HOI pose sequences from the simulator to real-world videos.*

The ability to manipulate objects with our hands represents a fundamental human skill, and the computational understanding of this capability—referred to as Hand-Object Interaction (HOI) understanding—has become increasingly significant in the fields of computer vision and embodied AI. The field has seen a shift towards data-driven paradigms, where large-scale HOI datasets are instrumental for accurate hand pose estimation [[[92](#bib.bib55), [64](#bib.bib80), [55](#bib.bib83), [74](#bib.bib88), [75](#bib.bib84), [32](#bib.bib91), [79](#bib.bib94)]], enabling realistic human-to-robot motion transfer [[[43](#bib.bib75), [37](#bib.bib76), [39](#bib.bib18)]] and 3D modeling [[[42](#bib.bib79), [75](#bib.bib84), [65](#bib.bib85), [57](#bib.bib86), [84](#bib.bib87), [10](#bib.bib95), [11](#bib.bib100), [51](#bib.bib101), [9](#bib.bib105), [87](#bib.bib102), [19](#bib.bib109), [18](#bib.bib108)]]. The critical challenge, however, lies in the data itself. Despite considerable investments in collecting real-world HOI sequences with detailed annotations [[[46](#bib.bib23), [20](#bib.bib28), [69](#bib.bib26)]], the reliance on costly and labor-intensive manual labeling poses a fundamental limitation to scalability.

The advent of deep learning and diffusion models has opened up promising avenues for scalable generation of HOI videos. However, as illustrated in Figure [1](#S1.F1), state-of-the-art methods [[[78](#bib.bib30), [82](#bib.bib54), [47](#bib.bib57), [3](#bib.bib58), [60](#bib.bib56), [89](#bib.bib97)]] still face significant challenges: (1) Pose-only synthesis approaches are limited in generating realistic appearances, as they predict MANO trajectories without producing pixel-level details, which diminishes their practical utility. (2) Single-image HOI generation (appearance generation) methods, which rely on hallucinating appearance from masks or 2D cues, fail to capture temporal dynamics, resulting in incoherent motion when applied to real-world scenarios. (3) Video generation (motion generation) methods that require both the complete pose sequence and the ground-truth first frame are unsuitable for sim-to-real deployment, as ground-truth first frame is unavailable from the simulator. These limitations emphasize the need for innovative approaches that can generate high-quality, temporally consistent HOI videos without depending on fixed pose sequences or ground-truth first frame inputs, thus providing a more flexible and robust solution for sim-to-real transfer.

In this paper, we introduce PAM: a pioneering Pose–Appearance–Motion engine for sim-to-real HOI video generation that requires only the initial and target poses, along with object geometry, as input. By integrating a motion-appearance diffusion process, our method bypasses the need for a conditioned first frame, thereby maximizing both motion and appearance diversity—a capability unattainable by prior work. To ensure high realism, we incorporate multiple conditions that effectively preserve fine hand details.

Our framework consists of three core stages. The pose sequence generation stage produces a plausible hand motion trajectory using a pre-trained model. Next, the appearance generation stage synthesizes a realistic initial frame using the controllable image diffusion model Flux [[[5](#bib.bib62)]]. This model is conditioned on a fusion of depth maps, semantic masks, and hand keypoint maps, ensuring geometric accuracy, semantic coherence, and the preservation of fine-grained hand details. The motion generation stage first renders the hand motion trajectory into depth, semantic, and hand keypoint sequences, and then generates videos using a controllable video diffusion model based on CogVideoX [[[70](#bib.bib31)]]. Importantly, the video model is conditioned on the same multi-modal inputs as the appearance generation stage to ensure consistency with the generated HOI pose sequence.

We evaluate our method on the DexYCB [[[7](#bib.bib36)]] and OAKINK2 [[[76](#bib.bib27)]] benchmarks, where it comprehensively surpasses existing approaches in video generation quality, motion plausibility, and hand pose fidelity. More importantly, as evidenced in Figure LABEL:fig:teaser, the synthesized videos from our method provide substantial value as synthetic data. When used for training, they lead to meaningful gains in the performance of a downstream hand pose estimation model, demonstrating their effectiveness as a data augmentation tool.

We summarize our contributions as follows:

-
•

Minimal-Conditioning Generation: We pioneer a pose–appearance–motion engine for sim-to-real HOI video generation that requires only sparse pose keyframes and object geometry as input, overcoming the first-frame bottleneck of prior methods.

-
•

Decoupled Generation Architecture: We design a novel pipeline that decouples pose, appearance and motion synthesis, leveraging multi-modal conditions to achieve superior realism, controllability and diversity.

-
•

State-of-the-Art Performance and Utility: Our method achieves superior results on established benchmarks and proves its practical value by enabling significant gains in downstream task performance through effective data augmentation.

## 2 Related Works

### 2.1 Hand-Object Motion Synthesis

Synthesizing high-fidelity hand-object motion is a fundamental challenge in computer animation and robotic grasping [[[1](#bib.bib10), [21](#bib.bib12), [12](#bib.bib11), [40](#bib.bib90), [86](#bib.bib104)]]. Prevailing data-driven approaches rely on supervised learning from large-scale, well-annotated datasets [[[31](#bib.bib14), [34](#bib.bib15), [15](#bib.bib74), [48](#bib.bib73), [13](#bib.bib16), [44](#bib.bib17), [39](#bib.bib18), [90](#bib.bib19), [92](#bib.bib55), [91](#bib.bib81), [85](#bib.bib82), [30](#bib.bib92), [52](#bib.bib98)]]. However, the scalability of these methods is constrained by their dependence on costly and difficult-to-acquire data [[[17](#bib.bib20), [26](#bib.bib21), [46](#bib.bib23), [45](#bib.bib24), [20](#bib.bib28), [69](#bib.bib26), [76](#bib.bib27), [7](#bib.bib36)]]. To circumvent this limitation, reinforcement learning (RL) has emerged as a promising alternative. Methods like [[[12](#bib.bib11), [68](#bib.bib29)]] generate reference grasps before synthesizing motions, while GraspXL [[[78](#bib.bib30)]] learns a generalizable grasping policy directly in simulation, eliminating the need for predefined references. These RL-based techniques produce high-quality interaction data, forming a robust foundation for sim-to-real transfer. However, these methods lack appearance modeling.

### 2.2 Controllable Video Generation

Recent breakthroughs in video generation foundation models [[[70](#bib.bib31), [6](#bib.bib32), [62](#bib.bib33), [35](#bib.bib34), [2](#bib.bib35)]] have intensified interest in controllable generation that precisely aligns with user intent. While text-to-video and image-to-video models [[[2](#bib.bib35), [62](#bib.bib33), [70](#bib.bib31), [58](#bib.bib37), [53](#bib.bib38), [25](#bib.bib39), [67](#bib.bib40), [80](#bib.bib41), [38](#bib.bib103)]] have demonstrated impressive capabilities, they often lack the granularity for specialized tasks. This has spurred research into integrating more precise control signals, such as semantic maps, depth, and camera motion. ControlNet [[[81](#bib.bib42)]] and its variants [[[22](#bib.bib43), [24](#bib.bib44), [41](#bib.bib106), [77](#bib.bib107)]] enable conditioning on dense inputs, while works like VideoComposer [[[66](#bib.bib45)]] fuse multiple conditions for enhanced control. Camera motion has been explicitly modeled by embedding parameters into diffusion models [[[27](#bib.bib46), [4](#bib.bib47)]]. However, generating videos of hand-object interactions (HOI) presents a unique challenge due to the high degrees of freedom in hand motion. This demands even more enriched and specialized control mechanisms—combining semantic, geometric, and precise pose cues—to achieve the necessary fidelity and accuracy.

### 2.3 Hand-Object Interaction Image & Video Generation

Generating Hand-Object Interaction (HOI) content is vital for understanding human activities. Prior work on HOI image generation [[[28](#bib.bib49), [36](#bib.bib50), [50](#bib.bib51), [63](#bib.bib52), [73](#bib.bib53), [82](#bib.bib54), [8](#bib.bib78)]] typically conditions on 2D signals like segmentation masks and keypoints. However, these static methods cannot capture the dynamic nature of interactions. Recently, several studies [[[60](#bib.bib56), [47](#bib.bib57), [3](#bib.bib58), [14](#bib.bib59), [72](#bib.bib77), [16](#bib.bib93), [88](#bib.bib96)]] have explored HOI video generation. InterDyn [[[3](#bib.bib58)]] conditions on hand mask sequences via ControlNet [[[81](#bib.bib42)]], but under-utilizes the rich conditions available from simulators. ManiVideo [[[47](#bib.bib57)]] introduces an occlusion-aware representation but requires human appearance data, which is not available from simulators like GraspXL [[[78](#bib.bib30)]]. More critically, these methods primarily focus on generation quality and have not thoroughly investigated the downstream utility of their synthesized data, which is essential for validating practical impact beyond perceptual metrics.

## 3 The Proposed Method

![Figure](2603.22193v3/x2.png)

*Figure 2: Overview of our three-stage generation pipeline. (1) Pose Generation: A pretrained pose generation model generates the intermediate hand-object interaction (HOI) poses based on the initial and target poses, along with the object mesh. (2) Appearance Generation: A controllable image diffusion model synthesizes the first frame of the video, conditioned on multi-modal inputs (depth maps, semantic masks, and keypoint annotations). (3) Motion Generation: The generated HOI sequence and the first frame are rendered into a full video sequence by a video diffusion model, conditioned on the same multi-modal inputs used in the appearance generation stage.*

### 3.1 Overview

Figure [2](#S3.F2) illustrates our method.
Conditioned on an initial MANO [[[56](#bib.bib60)]] hand pose $\mathbf{h}_{0}\!\in\!\mathbb{R}^{51\times 3}$, an object mesh $\mathbf{m}$ without appearance, an initial 6-DoF object pose $\mathbf{o}_{0}\!\in\!\mathbb{R}^{6}$, and a target hand pose $\mathbf{h}_{T}\!\in\!\mathbb{R}^{51\times 3}$, our generative model

$$f_{\bm{\theta}}:(\mathbf{h}_{0},\mathbf{m},\mathbf{o}_{0},\mathbf{h}_{T})\;\rightarrow\;\{I_{t}\}_{t=0}^{T}$$ \tag{1}

produces a photo-realistic video that (i) begins with $\mathbf{h}_{0}$, (ii) ends with $\mathbf{h}_{T}$, and (iii) depicts a temporally-coherent grasp-to-place motion. All hand poses are parameterised by global translation + rotation plus joint angles; frames $I_{t}$ are RGB images.

Jointly modeling pose, appearance and motion is inherently challenging due to the high-dimensional spatio-temporal manifold [[[23](#bib.bib61)]]. To address this, we decompose the generation process into three distinct stages:

-
1.

Pose Generation — A pretrained pose generation model synthesizes aligned hand-object pose sequences, $\{\mathbf{h}_{t},\mathbf{o}_{t}\}_{t=0}^{T}$, from the initial and target hand-object poses and object mesh, $(\mathbf{h}_{0},\mathbf{o}_{0},\mathbf{m})$ (Sec. [3.2](#S3.SS2)).

-
2.

Appearance Generation — A controllable image diffusion model generates the first frame, $I_{0}$, conditioned on the initial HOI pose and object mesh (Sec. [3.3](#S3.SS3)).

-
3.

Motion Generation — The synthetic HOI pose sequences and the first frame are injected into a video diffusion model, which animates $I_{0}$ into a photo-realistic video clip $\{I_{t}\}_{t=0}^{T}$ (Sec. [3.4](#S3.SS4)).

### 3.2 Stage I: Pose Generation Stage

In the first stage, we aim to generate an interpolated hand-object interaction (HOI) pose sequence, transitioning from the initial to the target pose while incorporating the object mesh. As shown in Figure [2](#S3.F2), we employ GraspXL [[[78](#bib.bib30)]], a pretrained model for hand-object interaction tasks. GraspXL takes as input the initial and target MANO hand pose $\mathbf{h}_{0}$, the 6-DoF object pose $\mathbf{o}_{0}$, and the object mesh $\mathbf{m}$, and produces temporally coherent trajectories $\{\mathbf{h}_{t},\mathbf{o}t\}_{t=0}^{T}$ for both hand and object poses. This ensures smooth interpolation while maintaining the physical consistency of the hand-object interaction, providing the foundation for subsequent video generation stages.

### 3.3 Stage II: Appearance Generation Stage

Bridge Conditions for Sim-to-Real HOI Video Synthesis: The primary objective of this work is to enhance the visual quality of simulated videos while preserving other conditions, thereby bridging the gap between simulation and real-world scenarios. By incorporating both geometric information (e.g., depth maps) and semantic data (e.g., segmentation masks) from the simulator, we seek to accurately reconstruct the visual representation of scenes and objects, while ensuring consistency across all other conditions. However, relying solely on these two data types proves insufficient for accurately generating Hand-Object Interactions (HOI) images or videos. This limitation stems from the complexity and high degree of freedom inherent in hand movements, which cannot be fully captured by geometric and semantic data alone. Specifically, these conditions fail to account for critical details, such as the number of fingers and their individual poses. To address this challenge, we introduce an additional condition—hand keypoint sequences, as proposed by [[[82](#bib.bib54)]]—to enable more precise and accurate hand pose generation. This approach facilitates the generation of realistic hand poses, thereby enhancing the overall realism of the interaction. In section [1](#S4.T1) and [4.2](#S4.SS2), we explore the influence of every condition.

We fine-tune Flux [[[5](#bib.bib62)]] with a ControlNet [[[81](#bib.bib42)]] fork that accepts depth $D_{0}$, segmentation $S_{0}$ and hand-keypoint image $K_{0}$ ($H\!\times\!W\!\times\!3$ each).
All cues are VAE-encoded to $\frac{H}{8}\!\times\!\frac{W}{8}\!\times\!16$ latents, concatenated channel-wise and Injected into two layers of DiT [[[49](#bib.bib64)]] blocks, with weights initialized from the first two layers of original Flux.:

$$f_{l}=f_{l}+\mathcal{Z}(f^{\prime}_{l}),$$ \tag{2}

where $f_{l}$ is the output of the $l$-th layer of the original DiT [[[49](#bib.bib64)]] blocks, and $f^{\prime}_{l}$ is the output of the $l$-th layer of the duplicated DiT blocks whose input is the concatenated conditions. Here, $l\in\{0,1\}$, and $\mathcal{Z}$ represent the zero-convolution layer, which is a $1\times 1$ convolution with all parameters initialized to zero. During training, only the parameters of ControlNet are updated.

### 3.4 Stage III: Motion Generation Stage

To generate the target video sequence, we combine the generated HOI pose sequence with a controllable video diffusion model. As shown in Figure [2](#S3.F2), depth maps, instance-level segmentation masks, and 2-D hand keypoint images are rasterized at each frame from Stage I. These conditions are then encoded into a latent tensor of shape $\mathbb{R}^{\frac{T+1}{4}\times\frac{H}{8}\times\frac{W}{8}\times 16}$ using a pretrained video VAE. Next, the latent representations are concentrated along channel dimensions and injected into CogVideo-X through 12 duplicate DiT blocks, as described in Eq. [2](#S3.E2). To prevent over-reliance on any single modality during training, each cue is randomly masked with a probability of 0.2.

## 4 Experiment

### 4.1 Experiment Settings

*Table 1: Quantitative comparison on DexYCB dataset. Our method is evaluated against CosHand, InterDyn, and ManiVideo. Results for InterDyn and ManiVideo are taken from their original papers. For fair comparison, CosHand was fine-tuned on the s0-split training set identical to ours. Our approach achieves state-of-the-art performance across all metrics (FVD, LPIPS, MF, MPJPE) while generating high-resolution 480x720 videos.*

*Table 2: Quantitative results on the OAKINK2 dataset. Comparison of our method with CosHand. For a fair evaluation, both models are trained on the same dataset. Our approach achieves state-of-the-art performance, outperforming CosHand across all evaluated metrics.*

Datasets and Data Processing. We evaluate our method on two standard benchmarks for HOI video generation: DexYCB [[[7](#bib.bib36)]] and OAKINK2 [[[76](#bib.bib27)]]. For DexYCB, we adopt the s0-split, comprising 6,400 training and 1,600 validation videos. Due to the scale of OAKINK2, we use a curated subset of 8,000 video clips (each 49 frames long), split into 6,400 for training and 1,600 for validation. The conditions for our model—depth maps, semantic masks, and hand keypoints—are derived as follows: depth maps are estimated using DepthCrafter [[[29](#bib.bib65)]], while semantic and keypoint information are obtained directly from the dataset annotations.

![Figure](2603.22193v3/x3.png)

*Figure 3: Qualitative comparison against CosHand. Example results on DexYCB and OAKINK2 highlight the strengths of our method in two key areas: (1) higher visual fidelity in both foreground and background generation, and (2) improved geometric accuracy of the synthesized hand poses.*

Evaluation Metrics. We employ a comprehensive set of metrics to evaluate our method from four perspectives:

-
•

Image Quality: We assess perceptual quality using Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) [[[83](#bib.bib67)]] and Peak Signal-to-Noise Ratio (PSNR).

-
•

Spatio-temporal Coherence: We adopt Fréchet Video Distance (FVD) [[[61](#bib.bib68)]] to evaluate overall video realism, using the implementation from [[[59](#bib.bib69)]].

-
•

Motion Fidelity: We use the Motion Fidelity (MF) metric [[[71](#bib.bib70)]] to quantify dynamic accuracy. For each video, we sample 100 foreground points (on hands/objects), track them using CoTracker3 [[[33](#bib.bib71)]], and compare the trajectories between generated and ground-truth videos. For a ground-truth tracklet $\mathcal{T}=\{\mathbf{\tau}_{1},\dots,\mathbf{\tau}_{T}\}$ and a generated tracklet $\mathcal{\tilde{T}}=\{\mathbf{\tilde{\tau}}_{1},\dots,\mathbf{\tilde{\tau}}_{T}\}$ where $\mathbf{\tau}_{t}\in\mathbb{R}^{2}$, MF is defined as:

$$\text{MF}=\frac{1}{|\mathcal{\tilde{T}}|}\sum_{\tilde{\tau}\in\mathcal{\tilde{T}}}\max_{\tau\in\mathcal{T}}\textbf{corr}(\tau,\tilde{\tau})+\frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\max_{\tilde{\tau}\in\mathcal{\tilde{T}}}\textbf{corr}(\tau,\tilde{\tau}).$$ \tag{3}

The correlation between two tracks is computed as:

$$\textbf{corr}(\tau,\tilde{\tau})=\frac{1}{F}\sum_{k=1}^{F}\frac{\mathbf{v}_{k}\cdot\mathbf{\tilde{v}}_{k}}{\|\mathbf{v}_{k}\|\|\mathbf{\tilde{v}}_{k}\|},$$ \tag{4}

where $\mathbf{v}_{k}=(v_{k}^{x},v_{k}^{y})$ and $\mathbf{\tilde{v}}_{k}=(\tilde{v}_{k}^{x},\tilde{v}_{k}^{y})$ are the displacement vectors at the $k$-th frame for tracks $\tau$ and $\tilde{\tau}$, respectively.

-
•

Hand Pose Accuracy: We report Mean Per-Joint Position Error (MPJPE) in millimeters [[[17](#bib.bib20)]], measuring the average Euclidean distance between the 21 predicted and ground-truth hand joints after root alignment. Lower MPJPE indicates better pose estimation accuracy. We utilize Hamer [[[48](#bib.bib73)]] to estimate the hand joints in the generated videos

### 4.2 Main Results

Baselines. We compare our method against state-of-the-art HOI video generation approaches: ManiVideo [[[47](#bib.bib57)]], InterDyn [[[3](#bib.bib58)]], and CosHand [[[60](#bib.bib56)]] on the DexYCB dataset [[[7](#bib.bib36)]]. For ManiVideo and InterDyn, we report results directly from their original publications (omitting metrics for which results were unavailable due to these methods not being open-source). For CosHand, we use the official implementation and fine-tune it on the DexYCB s0-split training set for a fair comparison. We also evaluate on OAKINK2, comparing against a similarly fine-tuned CosHand model. All baselines are evaluated in an image-to-video setting where the ground-truth first frame is provided, as this is required by these methods.

Quantitative Comparisons.
Our quantitative evaluation (Tables [1](#S4.T1), [2](#S4.T2)) demonstrates that our method achieves state-of-the-art performance across most metrics. We attribute this superiority to our multi-conditioning strategy, which provides the diffusion model with rich geometric and semantic cues (depth, masks, keypoints) to jointly optimize for visual realism and pose accuracy. In contrast, baseline methods exhibit limitations: InterDyn, ManiVideo, and CosHand rely on more limited conditioning signals or are built upon foundation models that struggle to capture the intricacies of hand-object interactions, leading to suboptimal performance.

*Table 3: Ablation study on input conditions on DexYCB dataset.*

Qualitative Comparisons.
As shown in Figure [3](#S4.F3), our method generates visually superior results compared to CosHand, even when CosHand is fine-tuned on the same training data. We identify two primary limitations in CosHand: (1) its reliance on hand masks as the sole conditioning signal provides insufficient geometric guidance for reconstructing precise hand poses, and (2) its lack of explicit temporal modeling mechanisms leads to inconsistent frame-to-frame outputs. In contrast, our approach addresses these issues by leveraging a video diffusion foundation model equipped with temporal attention to enforce coherence across frames. Furthermore, the use of hand keypoint maps as a conditioning signal explicitly preserves the structural details of hand configurations, resulting in more accurate and smooth video sequences.

![Figure](2603.22193v3/x4.png)

*Figure 4: Ablation study on input conditions on DexYCB dataset.*

### 4.3 Ablation Studies on Input Conditions

We conduct an ablation study on the DexYCB dataset to evaluate the contribution of different input conditions. The results (Tables [1](#S4.T1), [2](#S4.T2), and [3](#S4.T3)) yield three key observations:

-
•

The performance improves with an increasing number of conditions, validating the effectiveness of our multi-condition design.

-
•

Even when using the same segmentation mask condition as CosHand and InterDyn, our method achieves superior results, demonstrating the advantage of our pipeline.

-
•

While using only hand keypoints yields low MPJPE (due to explicit pose supervision), it underperforms on other metrics due to the lack of broader geometric and semantic context. This highlights the necessity of combining detailed local cues (keypoints) with global scene understanding (depth, semantics) for optimal performance.

Visual results in Figure [4](#S4.F4) further support these findings: using all conditions produces accurate poses; semantic masks or depth maps alone lead to pose inaccuracies; and keypoints alone degrade appearance quality.

### 4.4 Diverse Sim-to-Real Transfer

![Figure](2603.22193v3/x5.png)

*Figure 5: Sim-to-real transfer results. Our pipeline can generate realistic videos given initial and target states with diversity.*

*Table 4: Downstream task evaluation on SimpleHand [[[92](#bib.bib55)]].*

![Figure](2603.22193v3/x6.png)

*Figure 6: Data augmentation analysis with varying ratios of real data. We augment different portions of the DexYCB training set (25%, 50%, 75%, 100%) with our generated synthetic data. The baseline (dashed line) indicates performance when training solely on 100% of the real DexYCB data without synthetic augmentation.*

We conduct a sim-to-real transfer experiment to validate the effectiveness of our full pipeline. For this, we use GraspXL [[[78](#bib.bib30)]] as the hand motion generator, owing to its superior performance and strong generalization capabilities. Using objects from the DexYCB dataset, we randomly initialize both the hand and object poses, along with a target hand pose. GraspXL then generates the intermediate motion sequence, which is used to render the necessary conditions—depth, semantic masks, and keypoints—for our video generation model. As demonstrated in Figures LABEL:fig:teaser and [5](#S4.F5), our method successfully synthesizes diverse, realistic videos with varying subjects and backgrounds, using minimal input that includes only the initial and target poses, along with the object geometry. This capability arises from our decoupled generation architecture, which effectively combines the motion prior from GraspXL with the appearance modeling of our diffusion model. The utility of these synthesized videos for downstream tasks is explored in Section [4.5](#S4.SS5).

### 4.5 Downstream Task Validation

To evaluate the utility of our generated videos, we employ them for data augmentation in a hand pose estimation task. We use SimpleHand [[[92](#bib.bib55)]] as the pose estimation model, which regresses MANO parameters [[[56](#bib.bib60)]] from a single image. Our pipeline, trained on DexYCB, generates 3,400 video sequences (207,400 frames) for augmentation. We combine this synthetic data with varying subsets (25%, 50%, 75%, 100%) of the original DexYCB s0-split training set (406,888 frames). All models are evaluated on the DexYCB validation set using four metrics: Procrustes-Aligned Mean Per-Joint Position Error (PA-MPJPE), Procrustes-Aligned Mean Per-Vertex Position Error (PA-MPVPE), and F-Score. PA-MPJPE/PA-MPVPE measure the average Euclidean distance (in mm) after Procrustes alignment between the predicted and ground-truth joints/vertices, respectively.
The PA-MPJPE metric used here differs from the one in Table [1](#S4.T1). The MPJPE in this context measures data efficiency for downstream tasks, whereas the MPJPE in Table [1](#S4.T1) refers to the hand accuracy of the generated videos.

The quantitative results (Table [4](#S4.T4)) demonstrate that incorporating our generated data consistently improves hand pose estimation accuracy across all metrics. Figure [6](#S4.F6) reveals two key trends: (1) model performance improves monotonically with the amount of real data, and (2) most notably, using only 50% of the real data augmented with our synthetic samples achieves competitive performance with the 100% real data baseline. This indicates that our synthetic data can effectively compensate for reduced real data volume. Furthermore, the superior performance achieved using videos generated with multiple conditions validates the importance of our multi-conditioning approach for producing diverse and useful training data.

![Figure](2603.22193v3/x7.png)

*Figure 7: Zero-shot result on OAKINK2 dataset for i2v task. We use the weight trained on DexYCB dataset.*

### 4.6 Zero-Shot Results

To evaluate the generalizability of our approach, we test our video diffusion model trained on the DexYCB dataset (single-hand interactions) directly on the OAKINK2 dataset (bimanual interactions) in a zero-shot image-to-video setting. As shown in Figure [7](#S4.F7), our method generates plausible videos that maintain reasonable alignment with ground-truth hand poses and visual details, despite the significant domain shift. This cross-dataset generalization capability can be attributed to our use of pretrained video diffusion model weights as a strong foundation, combined with the ControlNet mechanism [[[81](#bib.bib42)]], which helps preserve the model’s original generation quality while adapting to new conditioning signals.

## 5 Conclusion

This paper proposed a framework that addresses the challenge of generating realistic HOI videos from minimal pose inputs. Our decoupled, multi-condition architecture produces superior results in both perceptual quality and geometric accuracy, and demonstrates practical utility through enhanced downstream task performance. While our method shows strong generalization, future work could explore extending it to more complex object interactions or unifying the motion and appearance stages into an end-to-end model. We believe our contributions provide a solid foundation for future research in generative models for embodied AI.

## Acknowledgments

This study is partially supported by Beijing Natural Science Foundation, QY25047.

## References

-
[1]
A. Agarwal, S. Uppal, K. Shaw, and D. Pathak (2023)

Dexterous functional grasping.

arXiv preprint arXiv:2312.02975.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[2]
N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025)

Cosmos world foundation model platform for physical ai.

arXiv preprint arXiv:2501.03575.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[3]
R. Akkerman, H. Feng, M. J. Black, D. Tzionas, and V. F. Abrevaya (2025)

InterDyn: controllable interactive dynamics with video diffusion models.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12467–12479.

Cited by: [Figure 1](#S1.F1.5.4),
[Figure 1](#S1.F1.8.2),
[§1](#S1.p2.1),
[§2.3](#S2.SS3.p1.1),
[§4.2](#S4.SS2.p1.1),
[Table 1](#S4.T1.2.1.3.1).

-
[4]
J. Bai, M. Xia, X. Fu, X. Wang, L. Mu, J. Cao, Z. Liu, H. Hu, X. Bai, P. Wan, et al. (2025)

Recammaster: camera-controlled generative rendering from a single video.

arXiv preprint arXiv:2503.11647.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[5]
black-forest-labs (2024)

Https://github.com/black-forest-labs/flux.

Cited by: [§1](#S1.p4.1),
[§3.3](#S3.SS3.p2.5).

-
[6]
A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. (2023)

Stable video diffusion: scaling latent video diffusion models to large datasets.

arXiv preprint arXiv:2311.15127.

Cited by: [§2.2](#S2.SS2.p1.1),
[§7.6](#S7.SS6.p1.1).

-
[7]
Y. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield, et al. (2021)

DexYCB: a benchmark for capturing hand grasping of objects.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 9044–9053.

Cited by: [§1](#S1.p5.1),
[§2.1](#S2.SS1.p1.1),
[§4.1](#S4.SS1.p1.1),
[§4.2](#S4.SS2.p1.1).

-
[8]
K. Chen, C. Min, L. Zhang, S. Hampali, C. Keskin, and S. Sridhar (2025)

FoundHand: large-scale domain-specific learning for controllable hand image generation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 17448–17460.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[9]
X. Chen, T. Liu, H. Zhao, G. Zhou, and Y. Zhang (2022)

Cerberus transformer: joint semantic, affordance and attribute parsing.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 19649–19658.

Cited by: [§1](#S1.p1.1).

-
[10]
X. Chen, Z. Song, X. Jiang, Y. Hu, J. Yu, and L. Zhang (2025)

HandOS: 3d hand reconstruction in one stage.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 17304–17314.

Cited by: [§1](#S1.p1.1).

-
[11]
Z. Chen, R. A. Potamias, S. Chen, and C. Schmid (2025)

HORT: monocular hand-held objects reconstruction with transformers.

arXiv preprint arXiv:2503.21313.

Cited by: [§1](#S1.p1.1).

-
[12]
S. Christen, L. Feng, W. Yang, Y. Chao, O. Hilliges, and J. Song (2024)

Synh2r: synthesizing hand-object motions for learning human-to-robot handovers.

In 2024 IEEE International Conference on Robotics and Automation (ICRA),

pp. 3168–3175.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[13]
S. Christen, M. Kocabas, E. Aksan, J. Hwangbo, J. Song, and O. Hilliges (2022)

D-grasp: physically plausible dynamic grasp synthesis for hand-object interactions.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 20577–20586.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[14]
L. Dang, R. Shao, H. Zhang, W. Min, Y. Liu, and Q. Wu (2025)

SViMo: synchronized diffusion for video and motion generation in hand-object interaction scenarios.

arXiv preprint arXiv:2506.02444.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[15]
H. Dong, A. Chharia, W. Gou, F. Vicente Carrasco, and F. D. De la Torre (2024)

Hamba: single-view 3d hand reconstruction with graph-guided bi-scanning mamba.

Advances in Neural Information Processing Systems 37, pp. 2127–2160.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[16]
Y. Fan, Q. Yang, K. Wang, H. Zhou, Y. Li, H. Feng, E. Ding, Y. Wu, and J. Wang (2025)

Re-hold: video hand object interaction reenactment via adaptive layout-instructed diffusion model.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 17550–17560.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[17]
Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges (2023)

ARCTIC: a dataset for dexterous bimanual hand-object manipulation.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 12943–12954.

Cited by: [§2.1](#S2.SS1.p1.1),
[4th item](#S4.I1.i4.p1.1).

-
[18]
Y. Feng, J. Lin, S. K. Dwivedi, Y. Sun, P. Patel, and M. J. Black (2024)

Chatpose: chatting about 3d human pose.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 2093–2103.

Cited by: [§1](#S1.p1.1).

-
[19]
Y. Feng, W. Liu, T. Bolkart, J. Yang, M. Pollefeys, and M. J. Black (2023)

Learning disentangled avatars with hybrid 3d representations.

arXiv preprint arXiv:2309.06441.

Cited by: [§1](#S1.p1.1).

-
[20]
R. Fu, D. Zhang, A. Jiang, W. Fu, A. Funk, D. Ritchie, and S. Sridhar (2025)

Gigahands: a massive annotated dataset of bimanual hand activities.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 17461–17474.

Cited by: [§1](#S1.p1.1),
[§2.1](#S2.SS1.p1.1).

-
[21]
A. Ghosh, R. Dabral, V. Golyanik, C. Theobalt, and P. Slusallek (2023)

IMoS: intent-driven full-body motion synthesis for human-object interactions.

In Computer Graphics Forum,

Vol. 42, pp. 1–12.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[22]
Z. Gu, R. Yan, J. Lu, P. Li, Z. Dou, C. Si, Z. Dong, Q. Liu, C. Lin, Z. Liu, et al. (2025)

Diffusion as shader: 3d-aware video diffusion for versatile video generation control.

In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers,

pp. 1–12.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[23]
X. Guo, J. Liu, M. Cui, L. Bo, and D. Huang (2024)

I4vgen: image as free stepping stone for text-to-video generation.

arXiv preprint arXiv:2406.02230.

Cited by: [§3.1](#S3.SS1.p2.1).

-
[24]
Y. Guo, C. Yang, A. Rao, M. Agrawala, D. Lin, and B. Dai (2024)

Sparsectrl: adding sparse controls to text-to-video diffusion models.

In European Conference on Computer Vision,

pp. 330–348.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[25]
Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai (2023)

Animatediff: animate your personalized text-to-image diffusion models without specific tuning.

arXiv preprint arXiv:2307.04725.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[26]
S. Hampali, M. Rad, M. Oberweger, and V. Lepetit (2020)

Honnotate: a method for 3d annotation of hand and object poses.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 3196–3206.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[27]
H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang (2024)

Cameractrl: enabling camera control for text-to-video generation.

arXiv preprint arXiv:2404.02101.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[28]
H. Hu, W. Wang, W. Zhou, and H. Li (2022)

Hand-object interaction image generation.

Advances in Neural Information Processing Systems 35, pp. 23805–23817.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[29]
W. Hu, X. Gao, X. Li, S. Zhao, X. Cun, Y. Zhang, L. Quan, and Y. Shan (2025)

Depthcrafter: generating consistent long depth sequences for open-world videos.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 2005–2015.

Cited by: [§4.1](#S4.SS1.p1.1).

-
[30]
M. Huang, F. Chu, B. Tekin, K. J. Liang, H. Ma, W. Wang, X. Chen, P. Gleize, H. Xue, S. Lyu, et al. (2025)

HOIGPT: learning long-sequence hand-object interaction with language models.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 7136–7146.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[31]
H. Jiang, S. Liu, J. Wang, and X. Wang (2021)

Hand-object contact consistency reasoning for human grasps generation.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 11107–11116.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[32]
S. Jiang, Q. Ye, R. Xie, Y. Huo, and J. Chen (2025)

Hand-held object reconstruction from rgb video with dynamic interaction.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12220–12230.

Cited by: [§1](#S1.p1.1).

-
[33]
N. Karaev, I. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht (2024)

Cotracker3: simpler and better point tracking by pseudo-labelling real videos.

arXiv preprint arXiv:2410.11831.

Cited by: [3rd item](#S4.I1.i3.p1.3).

-
[34]
K. Karunratanakul, J. Yang, Y. Zhang, M. J. Black, K. Muandet, and S. Tang (2020)

Grasping field: learning implicit representations for human grasps.

In 2020 International Conference on 3D Vision (3DV),

pp. 333–344.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[35]
W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. (2024)

Hunyuanvideo: a systematic framework for large video generative models.

arXiv preprint arXiv:2412.03603.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[36]
P. Kwon, C. Chen, and H. Joo (2024)

Graspdiffusion: synthesizing realistic whole-body hand-object interaction.

arXiv preprint arXiv:2410.13911.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[37]
M. Lepert, J. Fang, and J. Bohg (2025)

Phantom: training robots without robots using only human videos.

arXiv preprint arXiv:2503.00779.

Cited by: [§1](#S1.p1.1).

-
[38]
B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang, et al. (2025)

Uniscene: unified occupancy-centric driving scene generation.

In Proceedings of the computer vision and pattern recognition conference,

pp. 11971–11981.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[39]
K. Li, P. Li, T. Liu, Y. Li, and S. Huang (2025)

Maniptrans: efficient dexterous bimanual manipulation transfer via residual learning.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 6991–7003.

Cited by: [§1](#S1.p1.1),
[§2.1](#S2.SS1.p1.1).

-
[40]
M. Li, S. Christen, C. Wan, Y. Cai, R. Liao, L. Sigal, and S. Ma (2025)

LatentHOI: on the generalizable hand object motion generation with latent hand diffusion..

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 17416–17425.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[41]
W. Li, H. Xu, G. Zhang, H. Gao, M. Gao, M. Wang, and H. Zhao (2024)

Fairdiff: fair segmentation with point-image diffusion.

In International Conference on Medical Image Computing and Computer-Assisted Intervention,

pp. 617–628.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[42]
X. Liu, P. Ren, Q. Qi, H. Sun, Z. Zhuang, J. Wang, J. Liao, and J. Wang

Generalizable hand-object modeling from monocular rgb images via 3d gaussians.

In The Thirty-ninth Annual Conference on Neural Information Processing Systems,

Cited by: [§1](#S1.p1.1).

-
[43]
X. Liu, J. Adalibieke, Q. Han, Y. Qin, and L. Yi (2025)

Dextrack: towards generalizable neural tracking control for dexterous manipulation from human references.

arXiv preprint arXiv:2502.09614.

Cited by: [§1](#S1.p1.1).

-
[44]
X. Liu and L. Yi (2024)

Geneoh diffusion: towards generalizable hand-object interaction denoising via denoising diffusion.

arXiv preprint arXiv:2402.14810.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[45]
Y. Liu, H. Yang, X. Si, L. Liu, Z. Li, Y. Zhang, Y. Liu, and L. Yi (2024)

Taco: benchmarking generalizable bimanual tool-action-object understanding.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 21740–21751.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[46]
Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi (2022)

Hoi4d: a 4d egocentric dataset for category-level human-object interaction.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 21013–21022.

Cited by: [§1](#S1.p1.1),
[§2.1](#S2.SS1.p1.1).

-
[47]
Y. Pang, R. Shao, J. Zhang, H. Tu, Y. Liu, B. Zhou, H. Zhang, and Y. Liu (2025)

Manivideo: generating hand-object manipulation video with dexterous and generalizable grasping.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12209–12219.

Cited by: [Figure 1](#S1.F1.5.4),
[Figure 1](#S1.F1.8.2),
[§1](#S1.p2.1),
[§2.3](#S2.SS3.p1.1),
[§4.2](#S4.SS2.p1.1),
[Table 1](#S4.T1.2.1.4.1).

-
[48]
G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik (2024)

Reconstructing hands in 3d with transformers.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 9826–9836.

Cited by: [§2.1](#S2.SS1.p1.1),
[4th item](#S4.I1.i4.p1.1),
[§6](#S6.p2.1).

-
[49]
W. Peebles and S. Xie (2023)

Scalable diffusion models with transformers.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 4195–4205.

Cited by: [§3.3](#S3.SS3.p2.12),
[§3.3](#S3.SS3.p2.5).

-
[50]
A. Pelykh, O. M. Sincan, and R. Bowden (2024)

Giving a hand to diffusion models: a two-stage approach to improving conditional human image generation.

In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG),

pp. 1–10.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[51]
R. A. Potamias, J. Zhang, J. Deng, and S. Zafeiriou (2025)

Wilor: end-to-end 3d hand localization and reconstruction in-the-wild.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12242–12254.

Cited by: [§1](#S1.p1.1).

-
[52]
A. Prakash, B. Lundell, D. Andreychuk, D. Forsyth, S. Gupta, and H. Sawhney (2025)

How do i do that? synthesizing 3d hand motion and contacts for everyday interactions.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 7026–7036.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[53]
Z. Qing, S. Zhang, J. Wang, X. Wang, Y. Wei, Y. Zhang, C. Gao, and N. Sang (2024)

Hierarchical spatio-temporal decoupling for text-to-video generation.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 6635–6645.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[54]
S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He (2020)

Zero: memory optimizations toward training trillion parameter models.

In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis,

pp. 1–16.

Cited by: [§6](#S6.p1.1).

-
[55]
P. Ren, J. Wang, H. Sun, Q. Qi, X. Liu, M. Zhang, L. Zhang, J. Wang, and J. Liao (2025)

Prior-aware dynamic temporal modeling framework for sequential 3d hand pose estimation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 6476–6487.

Cited by: [§1](#S1.p1.1).

-
[56]
J. Romero, D. Tzionas, and M. J. Black (2022)

Embodied hands: modeling and capturing hands and bodies together.

arXiv preprint arXiv:2201.02610.

Cited by: [§3.1](#S3.SS1.p1.4),
[§4.5](#S4.SS5.p1.1).

-
[57]
M. U. Saleem, E. Pinyoanuntapong, M. J. Patel, H. Xue, A. Helmy, S. Das, and P. Wang (2025)

MaskHand: generative masked modeling for robust hand mesh reconstruction in the wild.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 8372–8383.

Cited by: [§1](#S1.p1.1).

-
[58]
U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni, et al. (2022)

Make-a-video: text-to-video generation without text-video data.

arXiv preprint arXiv:2209.14792.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[59]
I. Skorokhodov, S. Tulyakov, and M. Elhoseiny (2022)

Stylegan-v: a continuous video generator with the price, image quality and perks of stylegan2.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 3626–3636.

Cited by: [2nd item](#S4.I1.i2.p1.1).

-
[60]
S. Sudhakar, R. Liu, B. V. Hoorick, C. Vondrick, and R. Zemel (2024)

Controlling the world by sleight of hand.

In European Conference on Computer Vision,

pp. 414–430.

Cited by: [§1](#S1.p2.1),
[§2.3](#S2.SS3.p1.1),
[§4.2](#S4.SS2.p1.1),
[Table 1](#S4.T1.2.1.2.1),
[Table 2](#S4.T2.2.1.2.1).

-
[61]
T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly (2018)

Towards accurate generative models of video: a new metric & challenges.

ArXiv abs/1812.01717.

External Links: [Link](https://api.semanticscholar.org/CorpusID:54458806)

Cited by: [2nd item](#S4.I1.i2.p1.1).

-
[62]
T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025)

Wan: open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[63]
B. Wang, J. Zhou, J. Bai, Y. Yang, W. Chen, F. Wang, and Z. Lei (2025)

Realishuman: a two-stage approach for refining malformed human parts in generated images.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 39, pp. 7509–7517.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[64]
J. Wang, Q. Zhang, Y. Chao, B. Wen, X. Guo, and Y. Xiang (2024)

HO-cap: a capture system and dataset for 3d reconstruction and pose tracking of hand-object interaction.

arXiv preprint arXiv:2406.06843.

Cited by: [§1](#S1.p1.1).

-
[65]
S. Wang, H. He, M. Parelli, C. Gebhardt, Z. Fan, and J. Song (2025)

MagicHOI: leveraging 3d priors for accurate hand-object reconstruction from short monocular video clips.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 5957–5968.

Cited by: [§1](#S1.p1.1).

-
[66]
X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou (2023)

Videocomposer: compositional video synthesis with motion controllability.

Advances in Neural Information Processing Systems 36, pp. 7594–7611.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[67]
R. Wiersma, J. Philip, M. Hašan, K. Mullia, F. Luan, E. Eisemann, and V. Deschaintre (2025)

Uncertainty for svbrdf acquisition using frequency analysis.

In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers,

pp. 1–12.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[68]
Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. (2023)

Unidexgrasp: universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 4737–4746.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[69]
L. Yang, K. Li, X. Zhan, F. Wu, A. Xu, L. Liu, and C. Lu (2022)

Oakink: a large-scale knowledge repository for understanding hand-object interaction.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 20953–20962.

Cited by: [§1](#S1.p1.1),
[§2.1](#S2.SS1.p1.1).

-
[70]
Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. (2024)

Cogvideox: text-to-video diffusion models with an expert transformer.

arXiv preprint arXiv:2408.06072.

Cited by: [§1](#S1.p4.1),
[§2.2](#S2.SS2.p1.1).

-
[71]
D. Yatim, R. Fridman, O. Bar-Tal, Y. Kasten, and T. Dekel (2024)

Space-time diffusion features for zero-shot text-driven motion transfer.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 8466–8476.

Cited by: [3rd item](#S4.I1.i3.p1.3).

-
[72]
K. Ye, Y. Wu, S. Hu, J. Li, M. Liu, Y. Chen, and R. Huang (2025)

$\backslash$textsc $\{$gen2real$\}$: Towards demo-free dexterous manipulation by harnessing generated video.

arXiv preprint arXiv:2509.14178.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[73]
Y. Ye, X. Li, A. Gupta, S. De Mello, S. Birchfield, J. Song, S. Tulsiani, and S. Liu (2023)

Affordance diffusion: synthesizing hand-object interactions.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 22479–22489.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[74]
B. Yi, V. Ye, M. Zheng, Y. Li, L. Müller, G. Pavlakos, Y. Ma, J. Malik, and A. Kanazawa (2025)

Estimating body and hand motion in an ego-sensed world.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 7072–7084.

Cited by: [§1](#S1.p1.1).

-
[75]
Z. Yu, W. Xu, P. Xie, Y. Li, B. W. Anthony, Z. Zhang, and C. Lu (2025)

Dynamic reconstruction of hand-object interaction with distributed force-aware contact representation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 8590–8599.

Cited by: [§1](#S1.p1.1).

-
[76]
X. Zhan, L. Yang, Y. Zhao, K. Mao, H. Xu, Z. Lin, K. Li, and C. Lu (2024)

Oakink2: a dataset of bimanual hands-object manipulation in complex task completion.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 445–456.

Cited by: [§1](#S1.p5.1),
[§2.1](#S2.SS1.p1.1),
[§4.1](#S4.SS1.p1.1).

-
[77]
G. Zhang, H. Gao, Z. Jiang, H. Zhao, and Z. Zheng (2024)

Ctrl-u: robust conditional image generation via uncertainty-aware reward modeling.

arXiv preprint arXiv:2410.11236.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[78]
H. Zhang, S. Christen, Z. Fan, O. Hilliges, and J. Song (2024)

Graspxl: generating grasping motions for diverse objects at scale.

In European Conference on Computer Vision,

pp. 386–403.

Cited by: [Figure 1](#S1.F1.3.2),
[Figure 1](#S1.F1.8.2),
[§1](#S1.p2.1),
[§2.1](#S2.SS1.p1.1),
[§2.3](#S2.SS3.p1.1),
[§3.2](#S3.SS2.p1.4),
[§4.4](#S4.SS4.p1.1).

-
[79]
J. Zhang, J. Deng, C. Ma, and R. A. Potamias (2025)

HaWoR: world-space hand motion reconstruction from egocentric videos.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 1805–1815.

Cited by: [§1](#S1.p1.1).

-
[80]
L. Zhang and M. Agrawala (2025)

Packing input frame context in next-frame prediction models for video generation.

arXiv preprint arXiv:2504.12626.

Cited by: [§2.2](#S2.SS2.p1.1).

-
[81]
L. Zhang, A. Rao, and M. Agrawala (2023)

Adding conditional control to text-to-image diffusion models.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 3836–3847.

Cited by: [§2.2](#S2.SS2.p1.1),
[§2.3](#S2.SS3.p1.1),
[§3.3](#S3.SS3.p2.5),
[§4.6](#S4.SS6.p1.1).

-
[82]
M. Zhang, Y. Fu, Z. Ding, S. Liu, Z. Tu, and X. Wang (2024)

Hoidiffusion: generating realistic 3d hand-object interaction data.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 8521–8531.

Cited by: [Figure 1](#S1.F1.4.3),
[Figure 1](#S1.F1.8.2),
[§1](#S1.p2.1),
[§2.3](#S2.SS3.p1.1),
[§3.3](#S3.SS3.p1.1).

-
[83]
R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang (2018)

The unreasonable effectiveness of deep features as a perceptual metric.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 586–595.

Cited by: [1st item](#S4.I1.i1.p1.1).

-
[84]
Y. Zhang, Z. Cui, J. O. Kephart, and Q. Ji (2025)

Diffusion-based 3d hand motion recovery with intuitive physics.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 7306–7317.

Cited by: [§1](#S1.p1.1).

-
[85]
Z. Zhang, Y. Shi, L. Yang, S. Ni, Q. Ye, and J. Wang (2025)

OpenHOI: open-world hand-object interaction synthesis with multimodal large language model.

arXiv preprint arXiv:2505.18947.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[86]
H. Zhao, M. Lu, A. Yao, Y. Chen, and L. Zhang (2020)

Learning to draw sight lines.

International Journal of Computer Vision 128 (5), pp. 1076–1100.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[87]
H. Zhao, M. Lu, A. Yao, Y. Guo, Y. Chen, and L. Zhang (2017)

Physics inspired optimization on semantic transfer features: an alternative method for room layout estimation.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 10–18.

Cited by: [§1](#S1.p1.1).

-
[88]
H. Zhao, X. Liu, M. Xu, Y. Hao, W. Chen, and X. Han (2025)

TASTE-rob: advancing video generation of task-oriented hand-object interaction for generalizable robotic manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 27683–27693.

Cited by: [§2.3](#S2.SS3.p1.1).

-
[89]
Z. Zhao, L. Yang, P. Sun, P. Hui, and A. Yao (2025)

Analyzing the synthetic-to-real domain gap in 3d hand pose estimation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 12255–12265.

Cited by: [§1](#S1.p2.1).

-
[90]
Y. Zhong, Q. Jiang, J. Yu, and Y. Ma (2025)

Dexgrasp anything: towards universal robotic dexterous grasping with physics awareness.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 22584–22594.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[91]
B. Zhou, Y. Zhan, Z. Zhang, and Z. Lu (2025)

MEgoHand: multimodal egocentric hand-object interaction motion generation.

arXiv preprint arXiv:2505.16602.

Cited by: [§2.1](#S2.SS1.p1.1).

-
[92]
Z. Zhou, S. Zhou, Z. Lv, M. Zou, Y. Tang, and J. Liang (2024)

A simple baseline for efficient hand mesh reconstruction.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 1367–1376.

Cited by: [§1](#S1.p1.1),
[§2.1](#S2.SS1.p1.1),
[§4.5](#S4.SS5.p1.1),
[Table 4](#S4.T4),
[Table 4](#S4.T4.4.2).

## 6 Implementation Details

*Table 5: Resource Usage across Different Stages*

*Table 6: Ablation Study on Masking Probability and Input Condition Quality*

*Table 7: Ablation on Stage-I Method*

*Table 8: Ablation on Hand Representation*

Training Details: Our model was trained on a setup consisting of 8 x NVIDIA 800 GPUs, with a batch size of 4 x 8 and a learning rate of $1\times 10^{-4}$. The training process involved 8,000 training steps, using the AdamW optimizer and the DeepSpeed training architecture [[[54](#bib.bib72)]].

Evaluation Details: For the evaluation of video generation, we sample 1,600 videos, each consisting of 49 frames, from the test set. For the evaluation of Mean Per Joint Position Error (MPJPE), we utilize Hamer [[[48](#bib.bib73)]] to estimate the hand joints in the generated videos, and compute the loss by comparing the estimated joint positions with the ground truth hand joints. To assess the performance on downstream tasks, we train the SimpleHand model for 200 epochs using its official implementation.

Downstream Validation Details: For the downstream task, given the input, we first leverage the appearance generator (the controllable image diffusion model) to randomly sample 30 candidates. Subsequently, we filter out samples with low-accuracy hand poses using Hamer. Specifically, we predict the hand keypoints using Hamer for every frame, compare them with the ground truth, and discard the bottom 25% of the generated videos based on pose accuracy.

## 7 Additional Results

### 7.1 Compute and Memory Benchmarking

As shown in Table [5](#S6.T5), we report per-stage resource usage on an NVIDIA H20 GPU. Stage I is lightweight, Stage II has the highest peak memory (41.4 GB), and Stage III is the slowest (245.7 s) due to diffusion computation. Overall, the full pipeline runs in 301.1 s for 40 frames.

### 7.2 Ablation on Masking Probability and Input Condition Quality

We evaluate the robustness of our model by introducing random Gaussian noise into the input conditions. This noise is added to simulate real-world perturbations and assess how well the model can maintain performance despite such distortions. The results, as shown in Table [6](#S6.T6), highlight a noticeable performance degradation when no masking is applied — the model’s performance metrics decrease significantly when exposed to noise.

However, when masking is applied to the input conditions, the performance drops are considerably reduced. This observation strongly suggests that the random masking technique enhances the model’s robustness. By masking portions of the input, the model appears to be less sensitive to noise, likely because it focuses on more stable, less noisy features of the data. Therefore, the use of random masking improves the model’s generalization ability and resilience to external noise, thus supporting the hypothesis that masking is an effective strategy for improving robustness in challenging conditions.

### 7.3 Ablation on Appearance Generation Method

To evaluate the sensitivity of our method to Stage-I trajectories, we substituted GraspXL with D-Grasp. As shown in Table [7](#S6.T7), GraspXL achieves superior performance, demonstrating that the final generation quality relies on the quality of Stage-I pose sequence.

### 7.4 Ablation on Hand Representations

We ablate hand keypoints versus 2D hand mesh projections as conditioning. Table [8](#S6.T8) shows that keypoints perform better on most metrics, especially on MPJPE. This is possibly because mesh projections are more prone to self-occlusion.

### 7.5 Ablation on Hand Encoder

*Table 9: Ablation on Hand Encoder*

To validate the VAE-based condition encoder, we evaluate it on 1,000 sampled keypoint images and obtain a PSNR of 40.58, indicating a low reconstruction error for the conditioning images. As shown in Table [9](#S7.T9), the VAE outperforms an MLP-based 2D coordinate encoder across all metrics. This superior performance is primarily attributed to the VAE’s enhanced ability to preserve local spatial information.

### 7.6 Ablation on I2V Backbone

*Table 10: Ablation on Backbone*

![Figure](2603.22193v3/x8.png)

*Figure 8: More qualitative results on DexYCB dataset (a).*

To isolate the impact of our trimodal conditioning, we present a controlled ablation in Table [10](#S7.T10). Notably, even when employing a comparatively weaker backbone, the multi-conditioned SVD [[[6](#bib.bib32)]] baseline consistently outperforms the single-condition InterDyn across all metrics. This comparison demonstrates that integrating multiple conditions significantly enhances the overall generation performance.

### 7.7 Investigation in Error Propagation

Experiments in Figure [8](#S7.F8) show that Stage-I geometric errors (e.g., interpenetration or missing contact) can propagate, leading to physically implausible interactions even if the generated video appears photorealistic. Furthermore, we observe that Stage-III quality heavily relies on the appearance guidance from Stage II: low-quality initial reference frames degrade the final textures and exacerbate temporal flickering.

### 7.8 Additional Qualitative Results

We provide more qualitative results in Figure [9](#S7.F9) and Figure [10](#S7.F10) for DexYCB dataset, Figure [11](#S7.F11) and Figure [12](#S7.F12) for OANINK2 dataset.

![Figure](2603.22193v3/x9.png)

*Figure 9: More qualitative results on DexYCB dataset (a).*

![Figure](2603.22193v3/x10.png)

*Figure 10: More qualitative results on DexYCB dataset (b).*

![Figure](2603.22193v3/x11.png)

*Figure 11: More qualitative results on OAKINK2 dataset (a).*

![Figure](2603.22193v3/x12.png)

*Figure 12: More qualitative results on OAKINK2 dataset (b).*



Experimental support, please
[view the build logs](./2603.22193v3/__stdout.txt)
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