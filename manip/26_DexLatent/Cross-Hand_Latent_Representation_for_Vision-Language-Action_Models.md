##### Report GitHub Issue

**×




Title:
*


Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.10158v1 [cs.RO] 10 Mar 2026

[\useunder \ul # Cross-Hand Latent Representation for Vision-Language-Action Models Guangqi Jiang1∗ Yutong Liang1∗ Jianglong Ye1 Jia-Yang Huang1 Changwei Jing1 Rocky Duan2 Pieter Abbeel2,3 Xiaolong Wang1† Xueyan Zou1† 1UC San Diego 2Amazon FAR 3UC Berkeley ∗ Equal Contribution †Equal Advising https://xl-vla.github.io](https://xl-vla.github.io)

###### Abstract

Dexterous manipulation is essential for real-world robot autonomy, mirroring the central role of human hand coordination in daily activity. Humans rely on rich multimodal perception—vision, sound, and language-guided intent—to perform dexterous actions, motivating vision-based, language-conditioned manipulation systems for robots. However, training reliable vision-language-action (VLA) models for dexterous manipulation requires large-scale demonstrations across many robotic hands. In addition, as new dexterous embodiments appear rapidly, collecting data for each becomes costly and impractical, creating a need for scalable cross-embodiment learning. We introduce XL-VLA, a vision-language-action framework integrated with a unified latent action space shared across diverse dexterous hands. This embodiment-invariant latent space is directly pluggable into standard VLA architectures, enabling seamless cross-embodiment training and efficient reuse of both existing and newly collected data. Experimental results demonstrate that XL-VLA consistently outperforms baseline VLA models operating in raw joint spaces, establishing it as an effective solution for scalable cross-embodiment dexterous manipulation.

![[Uncaptioned image]](2603.10158v1/x1.png)

*Figure 1: Overview. XL-VLA enables direct decoding of a single latent action into multiple dexterous hand embodiments. Shown above, an action prediction can be instantiated on the Ability hand, Paxini DexH13 hand, X-Hand1, and Inspire hand for language-guided manipulation. We show our experiment settings on the right figure with collected objects and DexHands.*

## 1 Introduction

*Table 1: Related Work Summary. Summary of related work comparing data sources, deployment settings, and input/output capabilities for latent-based cross-embodiment methods. Data indicates the training modalities used in each work. Deployment specifies the robot embodiments evaluated and whether cross–end-effector transfer is supported. Input denotes which modalities (vision, language, proprioception) are used for training. Output reports whether a method includes a cross-embodiment decoder and whether it enables zero-shot transfer to unseen embodiments.*

Recent progress in vision-language-action (VLA) modeling has begun to extend the successes of large-scale vision and language models into robotics, enabling robots to interpret visual scenes, follow natural language instructions, and execute complex behaviors in the physical world. A key insight behind these advances is that unifying vision and language can be naturally expressed through sequence-to-sequence modeling, and VLA systems can adopt the same abstraction by treating actions as an additional output modality.

However, a fundamental obstacle emerges when moving from vision and language to action: while language possesses a relatively stable and universal vocabulary, robotic action spaces are inherently tied to the morphology of the robot. For dexterous hands in particular, action parameterizations—joint positions—vary significantly across embodiments and continue to evolve rapidly with new hardware designs. This raises two key questions for scalable robot learning: (1) How can we define a unified action representation within a family of robots? (2) How can we seamlessly integrate a new robot whose action space differs from existing ones?

In this work, we address these challenges by introducing a shared latent action space* tailored for dexterous hands. This latent space serves as an embodiment-invariant representation that enables joint training across heterogeneous hands. While prior VLA and cross-embodiment efforts have primarily focused on robotic arms equipped with parallel grippers, we focus on the substantially more complex, and more capable domain of dexterous manipulation. Moreover, we emphasize *real-world* datasets and physical robot evaluation, demonstrating that our method remains robust under significant cross-embodiment variation.

We summarize our contributions as follows:

-
•

We collect a large-scale teleoperation dataset covering 10 manipulation tasks across four newly introduced dexterous hands—Ability, Paxini DexH13, X-Hand1, and Inspire—containing 2M state-action pairs.

-
•

We propose an unsupervised latent autoencoder framework that learns a unified action space applicable to a wide range of hands.

-
•

We introduce XL-VLA, a full VLA pipeline built upon the cross-embodiment latent action space. XL-VLA achieves significantly stronger cross-embodiment performance than standard VLA baselines and exhibits zero-shot generalization to untrained cross-embodiment task configurations.

## 2 Related Work

Dexterous Manipulation.
The direction of dexterous manipulation focuses on utilizing DexHand for standard manipulation tasks, aiming to enable more complex operations. This field encompasses various areas of focus, including manipulator hardware [[[48](#bib.bib7), [34](#bib.bib8), [67](#bib.bib81)]], sensors [[[51](#bib.bib9), [63](#bib.bib10)]], learning and control algorithms [[[40](#bib.bib11), [28](#bib.bib12), [12](#bib.bib13), [31](#bib.bib84)]], and human-robot interaction [[[13](#bib.bib14), [23](#bib.bib15), [61](#bib.bib16)]]. In this work, we specifically concentrate on learning and control algorithms, leveraging vision-language-action (VLA) models [[[36](#bib.bib17), [46](#bib.bib18), [69](#bib.bib19), [50](#bib.bib20), [29](#bib.bib79)]]. Furthermore, we define a unified action space to support cross-embodiment dexterous manipulation [[[55](#bib.bib23), [24](#bib.bib21), [44](#bib.bib22)]].

*Figure 2: Model Pipeline. XL-VLA builds on $\pi_{0}$ [[[6](#bib.bib75)]] with vision and language encoders paired with an action expert that operates in a shared latent action space for cross-embodiment control. During VLA training, the action expert is finetuned while the pretrained latent encoders and decoders remain frozen.*

Cross Embodiment.
Cross embodiment typically refers to learning a single* policy that can flexibly adapt across diverse embodiments—e.g., different humanoids or dexterous hands—without per-robot retraining [[[52](#bib.bib24), [74](#bib.bib25), [26](#bib.bib26), [21](#bib.bib27), [22](#bib.bib28), [66](#bib.bib29), [25](#bib.bib30), [43](#bib.bib83)]]. Within this area, approaches leverage human video for supervision [[[70](#bib.bib31), [47](#bib.bib32), [15](#bib.bib33), [32](#bib.bib34), [45](#bib.bib35), [42](#bib.bib36)]], apply imitation learning with motion retargeting to bridge morphology gaps [[[49](#bib.bib37), [11](#bib.bib38), [39](#bib.bib39), [59](#bib.bib40)]], and employ generative models to synthesize action-consistent trajectories across bodies [[[3](#bib.bib41), [56](#bib.bib42), [1](#bib.bib43), [75](#bib.bib44), [27](#bib.bib82)]]. A complementary line of work constructs unified latent action spaces that factor out embodiment-specific details, enabling transfer and zero-shot reuse across platforms [[[73](#bib.bib45), [71](#bib.bib46), [9](#bib.bib47), [3](#bib.bib41), [53](#bib.bib48), [17](#bib.bib49)]]. This paper follows the latter paradigm, aligning actions in a shared representation for robust cross-embodiment control.

Hand/Dex Retargeting. Hand retargeting for teleoperation and imitation learning has progressed from kinematic pipelines to fast, principled learning: GeoRT delivers 1 kHz unsupervised mapping [[[68](#bib.bib55)]]; contact-aware and unified formulations improve human–object fidelity [[[35](#bib.bib56), [37](#bib.bib61)]], with objective ablations [[[60](#bib.bib57)]] and practical systems spanning real-time teleop and hardware-agnostic platforms [[[58](#bib.bib58), [14](#bib.bib59), [19](#bib.bib60)]]. Beyond copying humans, functional and policy-centric retargeting improves task outcomes [[[41](#bib.bib62), [62](#bib.bib63)]], while type-guided teleop exploits robot-specific dexterity [[[38](#bib.bib64)]].

Latent Action Space.
As shown in Table. [1](#S1.T1), latent action spaces provide embodiment agnostic control codes that align input modalities (vision, language, proprioception) and decode to diverse robots. Examples span discrete VQ tokens with per-robot decoders [[[8](#bib.bib50)]], continuous end-effector latents trained on retargeted pairs and generated by diffusion [[[3](#bib.bib41)]], and unified action VAEs that steer existing VLA policies [[[71](#bib.bib46)]]. Other directions align policy features or motion rather than a single EEF space optimal-transport co-training [[[45](#bib.bib35)]], Internet-video motion embeddings [[[65](#bib.bib52)]], diffusion transformers with standardized tokens [[[17](#bib.bib49)]] and cycle/adversarial mappings enabling cross-embodiment decoding and sim→real transfer [[[16](#bib.bib53), [54](#bib.bib54)]].

Vision-Language-Action Model. VLA models adapt large vision–language models (VLMs) to robot control by discretizing actions and predicting them autoregressively, enabling transfer of web-scale priors to manipulation [[[7](#bib.bib65), [33](#bib.bib66), [57](#bib.bib67), [30](#bib.bib80)]]. While these systems demonstrate broad generalization, their tokenized action decoding can hinder high-rate, dexterous control [[[72](#bib.bib68)]]. In contrast, we fine-tune a pre-trained VLM backbone initialized from PaliGemma [[[5](#bib.bib69)]] on teleoperated trajectories. Our action expert regresses continuous *latent action chunks*: each target is represented by a single latent vector produced by our hand-specific encoder. During training, we replace $\pi_{0}$’s original state tokens with these latent tokens and finetune on the next latent chunk, allowing a single hand-agnostic VLA policy to operate across multiple dexterous hands while preserving the benefits of VLM pretraining.

## 3 Method

### 3.1 Preliminary

Problem Formulation.
In this work, we address the problem of language-guided cross-embodiment dexterous manipulation based on visual perception. For a dexterous hand $h\in\mathcal{H}$ with $d_{h}$ actuated joints we control absolute joint rotations $\mathbf{q}^{(h)}\in\mathbb{R}^{d_{h}}$. At the policy level we operate on *action chunks*: each action $\mathbf{q}_{t}^{(h)}\in\mathbb{R}^{64\times d_{h}}$ is a sequence of $64$ joint-position commands sampled at $20$ Hz (3.2 s of motion). At control step $t$, the policy receives a short history of joint states, the previously executed action chunk $\mathbf{q}_{t}^{(h)}$, the current image observations $\mathbf{V}$, and a language instruction $\mathbf{T}$, and predicts the next chunk $\mathbf{q}_{t+1}^{(h)}$ via
$\mathbf{q}_{t+1}^{(h)}=F(\mathbf{q}_{t}^{(h)},\mathbf{V},\mathbf{T})$.

The objective is to predict embodiment-consistent joint-rotation trajectories conditioned on these multimodal inputs with a unified multi-task VLA model. Although the continuous joint spaces $\mathbf{q}^{(h)}$ are hand-specific, the sequence model $F$ itself is hand-agnostic; the hand identity $h$ is used only to choose the appropriate encoder/decoder that maps between joint space and the shared latent action space described below. To evaluate this setting, we consider a diverse set of dexterous robotic hands, including the Ability Hand, Inspire Hand, X-Hand1, and Paxini DexH13, which vary in structure, actuation, and kinematics.

To tackle this problem, we introduce an embodiment-invariant latent action space that integrates seamlessly into a vision–language–action (VLA) framework. This latent space provides a unified representation across diverse dexterous hands, enabling the model to train effectively on cross-embodiment data and generalize manipulation skills beyond a single hand morphology. Furthermore, the proposed latent space supports transferring control policies across different embodiments without requiring hand-specific retraining.

Pipeline.
As illustrated in Fig. [2](#S2.F2), our proposed framework consists of two main components:
(1) a VLA backbone that encodes multimodal inputs $(\mathbf{V},\mathbf{T})$, and (2) a pretrained set of latent encoders and decoders designed for cross-embodiment transfer. Our VLA design follows $\pi_{0}$ [[[6](#bib.bib75)]], which employs vision and language encoders together with an action expert. In the original $\pi_{0}$, proprioceptive history is provided through a stack of state tokens. In XL-VLA we instead feed *latent action tokens*: for each hand $h$, a hand-specific encoder $E_{h}$ maps the previous absolute joint-position action chunk $\mathbf{q}_{t}^{(h)}$ (64 frames at $20$ Hz) into a compact latent vector $\mathbf{z}_{t}=E_{h}(\mathbf{q}_{t}^{(h)})$. The VLA model conditions on a short history of such latent tokens, together with vision and language tokens, and predicts the next latent chunk $\widehat{\mathbf{z}}_{t+1}$. This latent is decoded by the embodiment-specific decoder $D_{h}$ to obtain the next joint command chunk $\widehat{\mathbf{q}}_{t+1}^{(h)}=D_{h}(\widehat{\mathbf{z}}_{t+1})$. During VLA finetuning we keep all latent encoders and decoders frozen.

This embodiment-invariant latent representation $\mathbf{z}$ acts as a unified action space shared across heterogeneous dexterous hands. By learning and decoding actions within this latent space, the model effectively bridges differences in morphology and actuation across embodiments, enabling a single hand-agnostic VLA policy to operate on diverse robotic hands. The hand identity $h$ is used only to select the appropriate encoder $E_{h}$ and decoder $D_{h}$ and is never provided as an explicit input token to the VLA backbone. We describe the detailed design and training of this latent action space in the following sections.

### 3.2 Latent Space

Definition.
Rather than defining a separate action space for each dexterous hand, we introduce a *shared latent action space* that provides a unified representation for all dexhand embodiments. This latent space is pretrained independently of the VLA model through a set of hand-specific encoders and decoders that all map to the same latent distribution. As a result, the latent embedding acts as an implicit, embodiment-agnostic action space that can be used by downstream policies to seamlessly control different dexterous hands.

#### 3.2.1 Modeling

To construct the latent representation, we employ a multi-headed VAE-style autoencoder.
For each hand type $h\in\mathcal{H}$ (e.g., X-Hand, Ability, Inspire, Paxini),
we define a hand-specific encoder $E_{h}$ and decoder $D_{h}$.
Each hand provides a joint configuration $\mathbf{q}^{(h)}\in\mathbb{R}^{d_{h}}$,
where $\mathbf{q}^{(h)}$ denotes the joint position values (q-pos) and $d_{h}$ is the dimensionality of that embodiment. The encoder outputs the parameters of a Gaussian posterior $(\boldsymbol{\mu}^{(h)},\boldsymbol{\sigma}^{(h)})=E_{h}(\mathbf{q}^{(h)})$, from which we sample a latent code $\mathbf{z}$ using the reparameterization trick, $q(\mathbf{z}\mid\mathbf{q}^{(h)})=\mathcal{N}(\boldsymbol{\mu}^{(h)},\mathrm{diag}((\boldsymbol{\sigma}^{(h)})^{2}))$. The decoder reconstructs back into the corresponding joint space $\hat{\mathbf{q}}^{(h)}=D_{h}(\mathbf{z})$.

In practice, each encoder and decoder is implemented as a lightweight MLP:
the input q-pos vector $\mathbf{q}^{(h)}$ is projected into a common latent space through the encoder MLP, and the decoder MLP reprojects the latent embedding back into the hand’s original joint configuration. This architecture provides a unified latent manifold while preserving the structure of each embodiment. To shape a meaningful cross-embodiment latent space, we impose three training constraints: (1) a *reconstruction constraint* $L_{1}$ ensuring $\hat{\mathbf{q}}^{(h)}$ matches $\mathbf{q}^{(h)}$, (2) a *retargeting constraint* $L_{2}$ aligning fingertip geometry across hands using differentiable forward kinematics,
and (3) a *latent constraint* $L_{3}$ regularizing the latent embedding to follow a smooth prior distribution. Together, these constraints encourage the latent space to capture embodiment-invariant structure, enabling consistent decoding into any dexterous hand.

#### 3.2.2 Objective

Reconstruction Loss ($L_{1}$).
Since each DexHand embodiment has its own joint space, we first require that the
hand-specific encoder-decoder pair behaves as an autoencoder on that hand.
Given a joint configuration $\mathbf{q}^{(h)}$ and its reconstruction
$\hat{\mathbf{q}}^{(h)}$ for hand $h\in\mathcal{H}$, the reconstruction loss
averaged over all hands is

$$L_{1}=\mathcal{L}_{\mathrm{rec}}=\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}\mathrm{MSE}\!\left(\hat{\mathbf{q}}^{(h)},\mathbf{q}^{(h)}\right),$$ \tag{1}

which ensures that the latent space preserves hand-specific kinematics and that
no embodiment is degraded by sharing the latent representation.

*Table 2: Vision-Language-Action Modeling. We compare XL-VLA with $\pi_{0}$ under cross-embodiment training. Although $\pi_{0}$ can handle different embodiments by adjusting sequence length, our method achieves consistently higher success rates across all hands and tasks. The first row PF, SC, SoC, etc. denote each task introduced in Task & Dataset, and each task is executed for 10 times, and we compute the success rate for each task cross hand.*

*Figure 3: Latent space pretraining pipeline. For each hand type, joint positions $\mathbf{q}_{h}$ are mapped through an encoder MLP into a shared latent space and reconstructed by a decoder MLP. The diagram also indicates the placement of the reconstruction loss $L_{1}$, retargeting loss $L_{2}$ via differentiable forward kinematics, and latent regularization loss $L_{3}$.*

Retargeting Loss ($L_{2}$).
To make the latent space truly cross-embodiment, we align fingertip geometry
between different DexHand robots.
For each hand $h$, we use differentiable forward kinematics (FK) to map joints
to fingertip positions $\mathbf{p}^{(h)}_{i}$, and define fingertip
displacements $\boldsymbol{\delta}^{(h)}_{ij}=\mathbf{p}^{(h)}_{i}-\mathbf{p}^{(h)}_{j}$ for fingertip pairs $(i,j)\in\mathcal{P}$. The pair set $\mathcal{P}$ contains thumb–finger pairs for the four aligned digits (thumb–index, thumb–middle, thumb–ring, thumb–little). We manually align finger indices across hands so that these digits correspond semantically; for Paxini DexH13, which lacks a little finger, we drop any pairs involving that digit when evaluating $L_{2}$. The retargeting loss penalizes discrepancies in pinch distances and directions between source hands $s$ and target hands $t$:

$$
\begin{aligned}
L_{2} &=\frac{1}{|\mathcal{H}|(|\mathcal{H}|-1)|\mathcal{P}|}\sum_{s\neq t}\sum_{(i,j)\in\mathcal{P}}w_{ij}^{(s)}\bigg[\lambda_{\mathrm{dis}}\big(\|\boldsymbol{\delta}_{ij}^{(s)}\|_{2}-\|\hat{\boldsymbol{\delta}}_{ij}^{(t)}\|_{2}\big)^{2} \tag{2}
\end{aligned}
$$
| | | $\displaystyle\hskip 16.38895pt\hskip 16.38895pt+\lambda_{\mathrm{dir}}\big(1-c_{ij}^{(s,t)}\big)\bigg].$ | |

where $\hat{\boldsymbol{\delta}}^{(t)}_{ij}$ is computed from the decoded
configuration of hand $t$, $c_{ij}^{(s,t)}$ denotes the cosine of the angle between the pinch directions $\boldsymbol{\delta}_{ij}^{(s)}$ and $\hat{\boldsymbol{\delta}}_{ij}^{(t)}$, and
$w_{ij}^{(s)}=\exp(-\lambda_{\mathrm{dis}}^{\mathrm{exp}}\|\boldsymbol{\delta}_{ij}^{(s)}\|_{2})$ emphasizes tighter pinches.
This loss encourages the same latent code to produce geometrically consistent
pinch behaviors across different hands.

Latent Loss ($L_{3}$).
Finally, we regularize the DexHand latent space to be smooth and well-behaved
by imposing a standard Gaussian prior on the latent variables.
For the approximate posterior $q(\mathbf{z}\mid\mathbf{q})$ produced by the
hand-specific encoders, the latent loss is

$$L_{3}=\mathcal{L}_{\mathrm{KL}}=\mathbb{E}_{\mathbf{q}}\big[\mathrm{KL}\big(q(\mathbf{z}\mid\mathbf{q})\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)\big],$$ \tag{3}

which encourages the shared DexHand latent space to follow a
$\mathcal{N}(0,I)$ distribution and facilitates sampling and interpolation
across embodiments.

Training Data and Protocol.
The latent autoencoder is trained without* any demonstration or IK-generated trajectories. Instead, for each hand $s\in\mathcal{H}$ we randomly sample joint configurations within the hardware joint limits to form synthetic joint-position vectors $\mathbf{q}^{(s)}$. For every such sample we encode $\mathbf{q}^{(s)}$ to a latent $\mathbf{z}$, decode it through all decoders $\{D_{t}\}_{t\in\mathcal{H}}$, and accumulate reconstruction and retargeting losses: the self-decoding $D_{s}(\mathbf{z})$ contributes to $L_{1}$, while cross-hand decodings $D_{t}(\mathbf{z})$ for $t\neq s$ contribute to $L_{2}$. Losses from all hands are aggregated and a single backward pass is applied, so all encoders and decoders are optimized jointly. Because $L_{2}$ uses only forward kinematics of each hand and decoded poses, the alignment of the latent space across embodiments is completely self-supervised and does not require any paired cross-hand trajectories.

Total Latent Objective.
The full latent training loss combines reconstruction, retargeting, and KL regularization:

$$L_{\mathrm{latent}}=L_{1}+L_{2}+\beta L_{3}.$$ \tag{4}

In all experiments we fix the weights to
$\beta=10^{-5}$,
$\lambda_{\mathrm{dis}}=2000.0$,
$\lambda_{\mathrm{dir}}=5.0$, and
$\lambda_{\mathrm{dis}}^{\mathrm{exp}}=12.0$.
These values yield a latent space that is both geometrically well-aligned across hands and smooth enough to support sampling and interpolation.

## 4 Experiments

*Figure 4: Zero-shot Unseen Tasks Generalization. For each hand, we randomly select some tasks as unseen tasks, whose data are held out from the training dataset. Then we test the unseen tasks with model trained on other data. Results show that by training with an aligned latent action space, XL-VLA gets the ability to generalize to novel hand-task combination in a zero-shot manner. PSR stands for “Partial Success Rate”, where policy is rewarded with half success if only one arm finishes its task.*

Tasks & Dataset.
We design 10 diverse tasks with different skills and objects to evaluate our VLA models. For each task, we collect 50 demonstrations each task per hand via [[[18](#bib.bib74)]], with 2000 demonstrations collected in total. Task descriptions are listed below:

(a) Prepare Fruits (PF). Put the banana and orange on the green board for cutting.
(b) Stack Cans (SC). Stack the cheese can on top of the salt.
(c) Sort Cans (SoC). Put the tomato can and the cheese can into the container.
(d) Hand over Bottle (HB). Hand over the white bottle from right hand to left hand.
(e) Re-organize Lemons (RL). Put the yellow lemon and the green lime into the bowl.
(f) Pour Sauce (PS). Pour mustard sauce into the meat can.
(g) Re-arrange Boxes (RB). Keep the table organized by re-arranging the two boxes.
(h) Push Sugar (PuS). Push the sugar boxes together.
(i) Pour Sugar (PoS). Add sugar to the starfruit.
(j) Push Cans (PC). Push the two tomato cans together.

Hardware.
To evaluate our method, we conduct comprehensive experiments on our real-world robot platform. We use a bimanual 7-DoF xArm and a Unitree G1 humanoid with various robot hands, shown in table [3](#S4.T3).

*Table 3: Dexterous Hand Comparison.*

Experiment Settings. We initialize the XL-VLA with weights from [[[6](#bib.bib75)]], then train the model on our collected multi-embodiment dataset. We use 8 NVIDIA H100 GPUs to train XL-VLA, each having 80GB memory. The model is trained 60K steps with a batch size of 128. Note that XL-VLA is a unified cross-embodiment multi-task policy. We use language to condition the policy on multiple tasks.

### 4.1 Main Results

In this section, we are trying to answer the following questions that mainly focus on Effectiveness of VLA + Latent Integration:
(1) Does XL-VLA outperform standard VLA models in cross-embodiment training? (2) Does XL-VLA enable zero-shot cross-embodiment skill transfer?

Cross-Hand Data Scaling.
Tab.[2](#S3.T2) presents the cross-embodiment manipulation results for XL-VLA compared with a strong $\pi_{0}$ baseline trained on the same multi-hand, multi-task dataset as a single shared policy across all four hands—Ability, Inspire, Paxini, and X-Hand—and ten manipulation tasks. Although $\pi_{0}$ can nominally accommodate different embodiments by varying sequence lengths, its performance remains inconsistent and generally low due to substantial kinematic and actuation differences across hands. In contrast, XL-VLA achieves strong and consistent improvements for every hand and task, demonstrating the benefit of learning a shared latent action representation.

Across hands, XL-VLA yields notable per-embodiment gains. The Ability Hand, which features relatively simple actuation, benefits from a large boost in reliability, improving from 0.37 to 0.73 overall. The Paxini Hand achieves highest performance among all embodiments (0.78 overall) , indicating strong compatibility between its actuation structure and the learned latent mapping. XHand , which is the most mechanically distinct from the rest, also improves significantly from 0.29 to 0.70, showing that XL-VLA can bridge large embodiment gaps.

Averaged over all tasks and hands, XL-VLA increases the mean success rate from 0.55 to 0.90 (+0.35). Particularly large improvements are observed for dexterity-heavy tasks such as Sort Cans, Hand over Bottle, and Re-arrange Boxes, underscoring the effectiveness of our embodiment-invariant latent space in capturing fine-grained manipulation behavior. Overall, these results demonstrate that XL-VLA enables robust cross-embodiment action prediction and consistently surpasses VLA models that lack a unified action representation.

Cross-Robot Data Scaling. To show the unified latent space benefit even for different robot systems, we test four manipulation tasks with data from the tabletop xArm and humanoid G1. We co-train the data from all embodiments on the four tasks with the same training parameters and show the G1 success rates in figure [5](#S4.F5). We can see that simply using aligned latent action space boost the performance of training on the raw action space, which has varied state/action lengths.

![Figure](2603.10158v1/x4.png)

*Figure 5: G1 Cross-Robot Performance. Co-training with latent xArm and humanoid data outperforms using raw actions.*

Zero-Shot Task Generalization.
A key advantage of using an embodiment-invariant latent action space is its ability to support seamless zero-shot generalization to unseen tasks. Because all dexterous hands share the same latent representation, a policy trained on a subset of tasks with one embodiment can transfer to a different task–hand combination without requiring additional training or retargeting.

*Table 4: Latent replay comparison. We compare our latent space with Latent Action Diffusion (LAD) [[[3](#bib.bib41)]]. For each hand combination, teleoperated trajectories collected on one source hand are encoded into the latent space, decoded onto the target hand, and replayed on real hardware. A replay is counted as successful if the encoded–decoded sequence can be executed without breaking contact or causing self-collisions. Higher replay success indicates better cross-embodiment consistency of the latent representation.*

To evaluate this capability, we hold out several manipulation tasks as unseen tasks for each hand and train XL-VLA on the remaining tasks. At test time, the trained policy is applied directly to the unseen task through the corresponding embodiment-specific decoder. As shown in Fig. [4](#S4.F4), we report both absolute success rate (SR) and partial success rate (PSR), where PSR accounts for intermediate progress (0.25, 0.5, 0.75, 1.0) to provide a more fine-grained measure of policy performance.

For comparison, we construct a $\pi_{0}$+RT baseline in which a policy is trained on all tasks using only the XHand embodiment. During evaluation, we apply a standard kinematic retargeting algorithm to map the predicted XHand joint trajectories to the other embodiments (Inspire, Ability, Paxini) by aligning fingertip positions. This baseline reflects common practice in cross-embodiment manipulation and allows us to assess whether our latent action representation provides genuine zero-shot benefits over retargeting-based transfer.

Across all embodiments and tasks, XL-VLA consistently outperforms the retargeting VLA baseline, often by a substantial margin. Notably, XL-VLA never underperforms the baseline on any hand or task, highlighting the robustness of the latent action representation. The gains are especially pronounced on fine-grained dexterous tasks (e.g., HB, RB), where geometric retargeting struggles to maintain coordinated finger motion.

*Table 5: Ablations. Ablation results comparing reconstruction accuracy, cross-embodiment retargeting, latent-space continuity, and interpolation smoothness. Exp denotes model variants: removing losses ($-L_{2}$, $-L_{2}^{dist}$, $-L_{2}^{dir}$, or both), changing hidden sizes ($H^{b}_{a}$), or changing latent dimension ($L_{d}$). Metrics include joint and tip RMSE for reconstruction; pinch- and random-motion direction/distance errors for retargeting; joint/tip latent continuity; and mean acceleration/jerk for interpolation.*

### 4.2 Ablation Results

In this section, we are trying to answer the question about Effectiveness of the Latent Action Space: (1) How well does the learned latent space function as a retargeting mechanism on its own? (2) What is the impact of different design choices within the latent space, as shown through ablation studies?

Latent Replay Comparison.
We further compare our latent action space against LAD [[[2](#bib.bib51)]], a supervised latent-space retargeting method.
To ensure a fair and challenging evaluation, we perform latent replay* by taking demonstrations from two embodiments and replaying them on the other two embodiments using each method’s latent mapping.
As shown in Table [4](#S4.T4), our approach achieves a mean success rate of 0.82 and 0.81 on the two hand pairs (Ability+Inspire and Paxini+XHand), substantially outperforming LAD, which attains only 0.60 and 0.61.
This improvement is consistent across all tasks, with gains particularly pronounced on fine-grained manipulation tasks such as SC, SoC, and HB, where LAD exhibits noticeable degradation.
Notably, our method achieves these results without any supervision data or paired labels, relying solely on unsupervised latent alignment.
These findings highlight that our latent space captures embodiment-invariant structure more effectively than supervised alternatives, enabling significantly more reliable cross-hand trajectory replay.

Visual Result. Figure [6](#S4.F6) shows latent decoding across different dexterous hands. We visualize one hand at full opacity and others with partial transparency, with the target grasp point marked in blue. Despite differing kinematics, all hands produce consistent poses from the same latent code, indicating that the learned latent space captures embodiment-invariant control.

![Figure](2603.10158v1/imgs/demo_transparent_1.png)

*(a) X-Hand*

![Figure](2603.10158v1/imgs/demo_transparent_2.png)

*(b) Inspire Hand*

Figure 6: Latent Visualizations. Latent decoding results cross embodiment.

Design Choice Comparison.
We conduct a comprehensive ablation study to evaluate architectural and loss-design choices for the latent action space, summarized in Tab. [5](#S4.T5).
Our final configuration uses the $H_{64}^{128\rightarrow 64}$ architecture with a latent dimension of 32. All metrics follow a “lower is better” convention, and the worst result compared with our method within each row are highlighted in green. Across reconstruction, cross-embodiment retargeting, latent continuity, and interpolation smoothness, our design choice achieves relatively stronger performance. Notably, performance remains stable across a wide range of architectures and latent dimensions, with degradation only occurring when the latent size is significantly increased (e.g., $L_{128}$), suggesting that excessively large latent spaces hinder embodiment-invariant structure. These results indicate that our chosen configuration offers an effective balance between model capacity and latent compactness.

And we explicitly write the evaluation metrics we design for this ablation study below:

Recon Joint/Tip RMSE.
Evaluates reconstruction fidelity of one hand. Synthetic random joint configurations are encoded and decoded, and we report the root-mean-square error (RMSE) between input and reconstructed joint angles (radians). In parallel, the original and reconstructed configurations are passed through each hand’s forward-kinematics model to obtain fingertip positions, and we compute the RMSE of fingertip displacement (meters). Lower values indicate that the latent representation preserves hand configurations.

Pinch and Random Tip Dir./Dist. Error
Assesses cross-hand transfer for pinch and random grasps. Pinch poses and random poses are encoded on a source hand and decoded on each target hand; for each pose, we form a line from the thumb tip to the opposing fingertip. Directional error is measured as the angle between predicted and reference lines (degrees), and distance error is measured as the absolute difference in thumb–finger distance (meters). Smaller values for both components indicate less directional drift and more faithful preservation of pinch aperture across hands.

Latent Continuity (Joint/Tip)
Test the local smoothness of the latent manifold. Encoded hand latents are perturbed with isotropic Gaussian noise of standard deviation $\epsilon=0.05$ and decoded back to joint angles, from which we compute the norm of the resulting joint-space deviation (radians). The corresponding finger tip effect is obtained by forwarding both perturbed and unperturbed reconstructions through forward kinematics and measuring the norm of fingertip displacement (meters). Small deviations indicate that the latent representation varies smoothly.

Interp. Accel./Jerk Mean
Characterizes the smoothness of latent-space interpolations. Two poses are encoded, and their latent codes are linearly interpolated. Decoding these intermediate codes yields fingertip trajectories, from which finite differences provide velocities and accelerations. Interp. Accel. Mean is the mean acceleration norm (meters per normalized interpolation step squared), while Interp. Jerk Mean is the mean norm of the jerk (the finite-difference derivative of acceleration, meters per step cubed). Lower values for both indicate smoother interpolation paths in the latent space.

## 5 Conclusion.

In this work, we introduced XL-VLA, a vision–language–action framework equipped with a unified latent action space for scalable cross-embodiment dexterous manipulation. By learning an embodiment-invariant latent representation, our approach enables seamless training across diverse robotic hands and supports zero-shot generalization to new hand–task combinations. Extensive real-world experiments demonstrate that XL-VLA consistently outperforms standard VLA models and retargeting-based baselines, while offering a flexible and plug-and-play interface for newly introduced hands. Overall, our results highlight latent action spaces as a powerful foundation for building generalizable, data-efficient dexterous manipulation systems. We believe this work takes a step toward more unified and adaptable robotic manipulation frameworks capable of keeping pace with rapid hardware innovation.

## References

-
[1]
Y. Bai, L. Yang, G. Eskandar, F. Shen, D. Chen, M. Altillawi, Z. Liu, and G. Kutyniok (2025)

RoboSwap: a gan-driven video diffusion framework for unsupervised robot arm swapping.

arXiv preprint arXiv:2506.08632.

External Links: [Link](https://arxiv.org/abs/2506.08632)

Cited by: [§2](#S2.p2.1).

-
[2]
E. Bauer, E. Nava, and R. K. Katzschmann (2025)

Latent action diffusion for cross-embodiment manipulation.

arXiv preprint arXiv:2506.14608.

External Links: 2506.14608,
[Document](https://dx.doi.org/10.48550/arXiv.2506.14608),
[Link](https://arxiv.org/abs/2506.14608)

Cited by: [§4.2](#S4.SS2.p2.1).

-
[3]
E. Bauer, E. Nava, and R. K. Katzschmann (2025)

Latent action diffusion for cross-embodiment manipulation.

arXiv preprint arXiv:2506.14608.

External Links: [Link](https://arxiv.org/abs/2506.14608)

Cited by: [Table 1](#S1.T1.3.3.2.1.1),
[§2](#S2.p2.1),
[§2](#S2.p4.1),
[Table 4](#S4.T4),
[Table 4](#S4.T4.2.2.1.1.1.1).

-
[4]
Q. Ben, F. Jia, J. Zeng, J. Dong, D. Lin, and J. Pang (2025)

Homie: humanoid loco-manipulation with isomorphic exoskeleton cockpit.

arXiv preprint arXiv:2502.13013.

Cited by: [Figure 15](#S6.F15),
[Figure 15](#S6.F15.4.2.1),
[§6.3](#S6.SS3.p2.1).

-
[5]
L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. (2024)

PaliGemma: a versatile 3b vlm for transfer.

arXiv preprint arXiv:2407.07726.

Cited by: [§2](#S2.p5.1).

-
[6]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky (2024)

$\pi_{0}$: A vision-language-action flow model for general robot control.

External Links: 2410.24164,
[Link](https://arxiv.org/abs/2410.24164)

Cited by: [Figure 2](#S2.F2),
[Figure 2](#S2.F2.2.1.1),
[§3.1](#S3.SS1.p4.11),
[Table 2](#S3.T2.1.1.1.1.1.1),
[§4](#S4.p4.1),
[Table 6](#S6.T6.1.1.1.1).

-
[7]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. (2023)

RT-2: vision-language-action models transfer web knowledge to robotic control.

arXiv preprint arXiv:2307.15818.

Cited by: [§2](#S2.p5.1).

-
[8]
Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li (2025-06)

Learning to Act Anywhere with Task-centric Latent Actions.

In Proceedings of Robotics: Science and Systems,

LosAngeles, CA, USA.

External Links: [Document](https://dx.doi.org/10.15607/RSS.2025.XXI.014),
[Link](https://www.roboticsproceedings.org/rss21/p014.pdf)

Cited by: [Table 1](#S1.T1.2.2.2.1.1),
[§2](#S2.p4.1).

-
[9]
Q. Bu, Y. Yang, J. Yang, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li (2025)

UniVLA: learning to act anywhere with task-centric latent actions.

arXiv preprint arXiv:2505.06111.

External Links: [Link](https://arxiv.org/abs/2505.06111)

Cited by: [§2](#S2.p2.1).

-
[10]
B. Calli, A. Singh, J. Bruce, A. Walsman, K. Konolige, S. Srinivasa, P. Abbeel, and A. M. Dollar (2017)

Yale-cmu-berkeley dataset for robotic manipulation research.

The International Journal of Robotics Research 36 (3), pp. 261–268.

Cited by: [§6.2](#S6.SS2.p4.1).

-
[11]
Z. Cao, B. Liu, S. Li, W. Zhang, and H. Chen (2025)

G-dream: graph-conditioned diffusion retargeting across multiple embodiments.

arXiv preprint arXiv:2505.20857.

External Links: [Link](https://arxiv.org/abs/2505.20857)

Cited by: [§2](#S2.p2.1).

-
[12]
A. S. Chen, P. Brakel, A. Bronars, A. Xie, S. Huang, O. Groth, M. Bauza, M. Wulfmeier, N. Heess, and D. Rao (2025)

Exploiting policy idling for dexterous manipulation.

In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

Cited by: [§2](#S2.p1.1).

-
[13]
X. Chi, C. Zhang, Y. Su, L. Dou, F. Yang, J. Zhao, H. Zhou, X. Jia, Y. Zhou, and S. An (2025)

Open teledex: a hardware-agnostic teleoperation system for imitation learning based dexterous manipulation.

arXiv preprint arXiv:2510.14771.

Cited by: [§2](#S2.p1.1).

-
[14]
X. Chi, C. Zhang, Y. Su, L. Dou, F. Yang, J. Zhao, H. Zhou, X. Jia, Y. Zhou, and S. An (2025)

Open teledex: a hardware-agnostic teleoperation system for imitation learning based dexterous manipulation.

arXiv preprint arXiv:2510.14771.

External Links: [Link](http://arxiv.org/abs/2510.14771)

Cited by: [§2](#S2.p3.1).

-
[15]
P. Dan, K. Kedia, A. Chao, E. W. Duan, M. A. Pace, W. Ma, and S. Choudhury (2025)

X-sim: cross-embodiment learning via real-to-sim-to-real.

In Proceedings of the 9th Conference on Robot Learning (CoRL 2025),

Proceedings of Machine Learning Research, Vol. 305, pp. 816–833.

External Links: [Link](https://proceedings.mlr.press/v305/)

Cited by: [§2](#S2.p2.1).

-
[16]
A. Dastider, H. Fang, and M. Lin (2025)

Cross-embodiment robotic manipulation synthesis via guided demonstrations through cyclevae and human behavior transformer.

arXiv preprint arXiv:2503.08622.

External Links: 2503.08622,
[Document](https://dx.doi.org/10.48550/arXiv.2503.08622),
[Link](https://arxiv.org/abs/2503.08622)

Cited by: [Table 1](#S1.T1.4.4.2.1.1),
[§2](#S2.p4.1).

-
[17]
T. Davies, Y. Huang, Y. Liu, X. Chen, H. Liu, and L. Hu (2025)

Tenma: robust cross-embodiment robot manipulation with diffusion transformer.

arXiv preprint arXiv:2509.11865.

External Links: [Link](https://arxiv.org/abs/2509.11865)

Cited by: [Table 1](#S1.T1.7.12.1.1.1),
[§2](#S2.p2.1),
[§2](#S2.p4.1).

-
[18]
R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang (2024)

Bunny-visionpro: real-time bimanual dexterous teleoperation for imitation learning.

External Links: [Link](https://arxiv.org/abs/2407.03162)

Cited by: [§4](#S4.p1.1),
[§6.3](#S6.SS3.p1.1).

-
[19]
Y. Dong, X. Liu, J. Wan, and Z. Deng (2025)

GEX: democratizing dexterity with fully-actuated dexterous hand and exoskeleton glove.

arXiv preprint arXiv:2506.04982.

External Links: [Link](http://arxiv.org/abs/2506.04982)

Cited by: [§2](#S2.p3.1).

-
[20]
Z. Dong, K. Chen, Z. Lv, H. Yu, Y. Zhang, C. Zhang, Y. Zhu, S. Tian, Z. Li, G. Moffatt, et al. (2025)

Digital twin catalog: a large-scale photorealistic 3d object digital twin dataset.

arXiv preprint arXiv:2504.08541.

Cited by: [§6.2](#S6.SS2.p4.1).

-
[21]
R. Doshi, H. Walke, O. Mees, S. Dasari, and S. Levine (2025)

Scaling cross-embodied learning: one policy for manipulation, navigation, locomotion and aviation.

In Proceedings of the 8th Conference on Robot Learning (CoRL 2024),

Proceedings of Machine Learning Research, Vol. 270, pp. 496–512.

External Links: [Link](https://proceedings.mlr.press/v270/doshi25a.html)

Cited by: [§2](#S2.p2.1).

-
[22]
A. Eftekhar, R. Hendrix, L. Weihs, J. Duan, E. Caglar, J. Salvador, A. Herrasti, W. Han, E. VanderBil, A. Kembhavi, K. Ehsani, K. Zeng, and R. Krishna (2024)

The one ring: a robotic indoor navigation generalist.

arXiv preprint arXiv:2412.14401.

External Links: [Link](https://arxiv.org/abs/2412.14401)

Cited by: [§2](#S2.p2.1).

-
[23]
H. Fang, B. Romero, Y. Xie, A. Hu, B. Huang, J. Alvarez, M. Kim, G. Margolis, K. Anbarasu, M. Tomizuka, et al. (2025)

DEXOP: a device for robotic transfer of dexterous human manipulation.

In Workshop on Dexterous Manipulation at Robotics: Science and Systems (RSS),

Note: Workshop paper

Cited by: [§2](#S2.p1.1).

-
[24]
X. Fei, Z. Xu, H. Fang, T. Zhang, and L. Shao (2025)

T(r,o) grasp: efficient graph diffusion of robot-object spatial transformation for cross-embodiment dexterous grasping.

arXiv preprint arXiv:2510.12724.

Cited by: [§2](#S2.p1.1).

-
[25]
M. Guaman Castro, S. Rajagopal, D. Gorbatov, M. Schmittle, R. Baijal, O. Zhang, R. Scalise, S. Talia, E. Romig, C. de Melo, B. Boots, and A. Gupta (2025)

VAMOS: a hierarchical vision-language-action model for capability-modulated and steerable navigation.

arXiv preprint arXiv:2510.20818.

External Links: [Link](https://arxiv.org/abs/2510.20818)

Cited by: [§2](#S2.p2.1).

-
[26]
Z. Hou, T. Zhang, Y. Xiong, H. Duan, H. Pu, R. Tong, C. Zhao, X. Zhu, Y. Qiao, J. Dai, and Y. Chen (2025)

Dita: scaling diffusion transformer for generalist vision-language-action policy.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

Cited by: [§2](#S2.p2.1).

-
[27]
T. Huang, G. Jiang, Y. Ze, and H. Xu (2024)

Diffusion reward: learning rewards via conditional video diffusion.

In European Conference on Computer Vision,

pp. 478–495.

Cited by: [§2](#S2.p2.1).

-
[28]
A. Hung, F. Yang, A. Kumar, S. A. Marinovic, S. Iba, R. S. Zarrin, and D. Berenson (2025)

AVO: amortized value optimization for contact mode switching in multi-finger manipulation.

arXiv preprint arXiv:2510.07548.

Cited by: [§2](#S2.p1.1).

-
[29]
G. Jiang, H. Chang, R. Qiu, Y. Liang, M. Ji, J. Zhu, Z. Dong, X. Zou, and X. Wang (2025)

Gsworld: closed-loop photo-realistic simulation suite for robotic manipulation.

arXiv preprint arXiv:2510.20813.

Cited by: [§2](#S2.p1.1).

-
[30]
G. Jiang, Y. Sun, T. Huang, H. Li, Y. Liang, and H. Xu (2024)

Robots pre-train robots: manipulation-centric robotic representation from large-scale robot datasets.

arXiv preprint arXiv:2410.22325.

Cited by: [§2](#S2.p5.1).

-
[31]
C. Jing, J. K. Bandi, J. Ye, Y. Duan, P. Abbeel, X. Wang, and S. Yi (2026)

Contact-aware neural dynamics.

arXiv preprint arXiv:2601.12796.

Cited by: [§2](#S2.p1.1).

-
[32]
H. Kim, J. Kang, H. Kang, M. Cho, S. J. Kim, and Y. Lee (2025)

UniSkill: imitating human videos via cross-embodiment skill representations.

In Proceedings of the 9th Conference on Robot Learning (CoRL 2025),

Proceedings of Machine Learning Research, Vol. 305.

External Links: [Link](https://proceedings.mlr.press/v305/)

Cited by: [§2](#S2.p2.1).

-
[33]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)

OpenVLA: an open-source vision-language-action model.

arXiv preprint arXiv:2406.09246.

Cited by: [§2](#S2.p5.1).

-
[34]
Y. Kuroda, T. Takahashi, C. C. Beltran-Hernandez, M. Hamaya, and K. Tanaka (2025)

PLEXUS hand: lightweight four-motor prosthetic hand enabling precision-lateral dexterous manipulation.

In Proceedings of the IEEE International Conference on Rehabilitation Robotics (ICORR),

Cited by: [§2](#S2.p1.1).

-
[35]
A. S. Lakshmipathy, J. K. Hodgins, and N. S. Pollard (2024)

Kinematic motion retargeting for contact-rich anthropomorphic manipulations.

arXiv preprint arXiv:2402.04820.

External Links: [Link](http://arxiv.org/abs/2402.04820)

Cited by: [§2](#S2.p3.1).

-
[36]
Q. Li, Y. Deng, Y. Liang, L. Luo, L. Zhou, C. Yao, L. Zeng, Z. Feng, H. Liang, S. Xu, et al. (2025)

Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos.

arXiv preprint arXiv:2510.21571.

Cited by: [§2](#S2.p1.1).

-
[37]
X. Lin, K. Yao, L. Xu, X. Wang, X. Li, Y. Wang, and M. Li (2025)

DexFlow: a unified approach for dexterous hand pose retargeting and interaction.

arXiv preprint arXiv:2505.01083.

External Links: [Link](http://arxiv.org/abs/2505.01083)

Cited by: [§2](#S2.p3.1).

-
[38]
Y. Lin, Y. Wei, H. Liao, M. Lin, C. Xing, H. Li, D. Zhang, M. Cutkosky, and W. Zheng (2025)

TypeTele: releasing dexterity in teleoperation by dexterous manipulation types.

arXiv preprint arXiv:2507.01857.

External Links: [Link](http://arxiv.org/abs/2507.01857)

Cited by: [§2](#S2.p3.1).

-
[39]
J. Liu, P. Ding, Q. Zhou, Y. Wu, D. Huang, Z. Peng, W. Xiao, W. Zhang, L. Yang, C. Lu, and D. Wang (2025)

TrajBooster: boosting humanoid whole-body manipulation via trajectory-centric learning.

arXiv preprint arXiv:2509.11839.

External Links: [Link](https://arxiv.org/abs/2509.11839)

Cited by: [§2](#S2.p2.1).

-
[40]
X. Liu, H. Wang, and L. Yi (2025)

DexNDM: closing the reality gap for dexterous in-hand rotation via joint-wise neural dynamics model.

arXiv preprint arXiv:2510.08556.

Cited by: [§2](#S2.p1.1).

-
[41]
Z. Mandi, Y. Hou, D. Fox, Y. Narang, A. Mandlekar, and S. Song (2025)

DexMachina: functional retargeting for bimanual dexterous manipulation.

arXiv preprint arXiv:2505.24853.

External Links: [Link](http://arxiv.org/abs/2505.24853)

Cited by: [§2](#S2.p3.1).

-
[42]
Y. Niu, Y. Zhang, M. Yu, C. Lin, C. Li, Y. Wang, Y. Yang, W. Yu, T. Zhang, Z. Li, J. Francis, B. Chen, J. Tan, and D. Zhao (2025)

Human2LocoMan: learning versatile quadrupedal manipulation with human pretraining.

In Proceedings of Robotics: Science and Systems (RSS),

Cited by: [§2](#S2.p2.1).

-
[43]
G. Pan, Q. Ben, Z. Yuan, G. Jiang, Y. Ji, J. Pang, H. Liu, and H. Xu (2024)

Roboduet: a framework affording mobile-manipulation and cross-embodiment.

arXiv preprint arXiv:2403.17367 6.

Cited by: [§2](#S2.p2.1).

-
[44]
E. Y. Puang, F. Ceola, G. Pasquale, and L. Natale (2025)

PCHands: pca-based hand pose synergy representation on manipulators with n-dof.

In Proceedings of the IEEE-RAS International Conference on Humanoid Robots (Humanoids),

Cited by: [§2](#S2.p1.1).

-
[45]
R. Punamiya, D. Patel, P. Aphiwetsa, P. Kuppili, L. Y. Zhu, S. Kareer, J. Hoffman, and D. Xu (2025)

EgoBridge: domain adaptation for generalizable imitation from egocentric human data.

In Advances in Neural Information Processing Systems (NeurIPS),

Note: Poster

Cited by: [Table 1](#S1.T1.7.10.1.1.1),
[§2](#S2.p2.1),
[§2](#S2.p4.1).

-
[46]
D. Qu, H. Song, Q. Chen, Z. Chen, X. Gao, X. Ye, Q. Lv, M. Shi, G. Ren, C. Ruan, et al. (2025)

EO-1: interleaved vision-text-action pretraining for general robot control.

arXiv preprint arXiv:2508.21112.

Cited by: [§2](#S2.p1.1).

-
[47]
J. Ren, P. Sundaresan, D. Sadigh, S. Choudhury, and J. Bohg (2025)

Motion tracks: a unified representation for human-robot transfer in few-shot imitation learning.

In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA),

Cited by: [§2](#S2.p2.1).

-
[48]
B. A. Richardson, F. Grüninger, L. Mack, J. Stueckler, and K. J. Kuchenbecker (2025)

ISyHand: a dexterous multi-finger robot hand with an articulated palm.

In Proceedings of the IEEE-RAS International Conference on Humanoid Robots (Humanoids),

Seoul, Korea.

Cited by: [§2](#S2.p1.1).

-
[49]
M. Seo, H. A. Park, S. Yuan, Y. Zhu, and L. Sentis (2024)

LEGATO: cross-embodiment imitation using a grasping tool.

arXiv preprint arXiv:2411.03682.

External Links: [Link](https://arxiv.org/abs/2411.03682)

Cited by: [§2](#S2.p2.1).

-
[50]
W. Song, J. Chen, P. Ding, Y. Huang, H. Zhao, D. Wang, and H. Li (2025)

CEED-vla: consistency vision-language-action model with early-exit decoding.

arXiv preprint arXiv:2506.13725.

Cited by: [§2](#S2.p1.1).

-
[51]
K. Sou, J. Gong, S. Li, C. Lyu, Z. Song, S. Mu, and W. Ding (2025)

Moirétac: a dual-mode visuotactile sensor for multidimensional perception using moiré pattern amplification.

arXiv preprint arXiv:2509.12714.

Cited by: [§2](#S2.p1.1).

-
[52]
W. Tan, B. Wang, H. Zhi, C. Liu, Z. Li, J. Liu, Z. Lin, Y. Dai, Y. Chen, W. Yang, E. Xie, H. Xue, B. Ji, C. Xu, Z. Wang, T. Wang, L. Zhu, and H. T. Shen (2025)

BLM1: a boundless large model for cross-space, cross-task, and cross-embodiment learning.

arXiv preprint arXiv:2510.24161.

External Links: [Link](https://arxiv.org/abs/2510.24161)

Cited by: [§2](#S2.p2.1).

-
[53]
T. Wang, D. Bhatt, X. Wang, and N. Atanasov (2024)

Cross-embodiment robot manipulation skill transfer using latent space alignment.

arXiv preprint arXiv:2406.01968.

External Links: [Link](https://arxiv.org/abs/2406.01968)

Cited by: [§2](#S2.p2.1).

-
[54]
T. Wang, D. Bhatt, X. Wang, and N. Atanasov (2024)

Cross-embodiment robot manipulation skill transfer using latent space alignment.

arXiv preprint arXiv:2406.01968.

External Links: 2406.01968,
[Document](https://dx.doi.org/10.48550/arXiv.2406.01968),
[Link](https://arxiv.org/abs/2406.01968)

Cited by: [Table 1](#S1.T1.6.6.3.1.1),
[§2](#S2.p4.1).

-
[55]
Y. Wei, Z. Luo, Y. Lin, M. Lin, Z. Liang, S. Chen, and W. Zheng (2025)

OmniDexGrasp: generalizable dexterous grasping via foundation model and force feedback.

arXiv preprint arXiv:2510.23119.

Cited by: [§2](#S2.p1.1).

-
[56]
J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng (2025)

DexVLA: vision-language model with plug-in diffusion expert for general robot control.

In Proceedings of the 9th Conference on Robot Learning (CoRL 2025),

Proceedings of Machine Learning Research, Vol. 305.

External Links: [Link](https://proceedings.mlr.press/v305/)

Cited by: [§2](#S2.p2.1).

-
[57]
J. Wen, Y. Zhu, J. Li, M. Zhu, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, et al. (2024)

TinyVLA: towards fast, data-efficient vision-language-action models for robotic manipulation.

arXiv preprint arXiv:2409.12514.

Cited by: [§2](#S2.p5.1).

-
[58]
R. Wen, J. Zhang, G. Chen, Z. Cui, M. Du, Y. Gou, Z. Han, J. Hu, L. Huang, H. Niu, W. Xu, H. Zhang, Z. Zhu, H. Li, and Z. Ren (2025)

Dexterous teleoperation of 20-dof bytedexter hand via human motion retargeting.

arXiv preprint arXiv:2507.03227.

External Links: [Link](http://arxiv.org/abs/2507.03227)

Cited by: [§2](#S2.p3.1).

-
[59]
Z. Wu, R. A. Potamias, X. Zhang, Z. Zhang, J. Deng, and S. Luo (2025)

CEDex: cross-embodiment dexterous grasp generation at scale from human-like contact representations.

arXiv preprint arXiv:2509.24661.

External Links: [Link](https://arxiv.org/abs/2509.24661)

Cited by: [§2](#S2.p2.1).

-
[60]
C. Xin, M. Yu, Y. Jiang, Z. Zhang, and X. Li (2025)

Analyzing key objectives in human-to-robot retargeting for dexterous manipulation.

arXiv preprint arXiv:2506.09384.

External Links: [Link](http://arxiv.org/abs/2506.09384)

Cited by: [§2](#S2.p3.1).

-
[61]
M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song (2025)

DexUMI: using human hand as the universal manipulation interface for dexterous manipulation.

In Proceedings of the 9th Conference on Robot Learning (CoRL),

Proceedings of Machine Learning Research, Vol. 305, pp. 437–459.

Cited by: [§2](#S2.p1.1).

-
[62]
S. Xu, Y. Chao, L. Bian, A. Mousavian, Y. Wang, L. Gui, and W. Yang (2025)

Dexplore: scalable neural control for dexterous manipulation from reference-scoped exploration.

arXiv preprint arXiv:2509.09671.

External Links: [Link](http://arxiv.org/abs/2509.09671)

Cited by: [§2](#S2.p3.1).

-
[63]
Z. Xu, Z. Si, K. Zhang, O. Kroemer, and Z. Temel (2025)

A multi-modal tactile fingertip design for robotic hands to enhance dexterous manipulation.

arXiv preprint arXiv:2510.05382.

Cited by: [§2](#S2.p1.1).

-
[64]
Cited by: [§6.3](#S6.SS3.p2.1).

-
[65]
J. Yang, Y. Shi, H. Zhu, M. Liu, K. Ma, Y. Wang, G. Wu, T. He, and L. Wang (2025)

CoMo: learning continuous latent motion from internet videos for scalable robot learning.

arXiv preprint arXiv:2505.17006.

External Links: 2505.17006,
[Document](https://dx.doi.org/10.48550/arXiv.2505.17006),
[Link](https://arxiv.org/abs/2505.17006)

Cited by: [Table 1](#S1.T1.7.11.1.1.1),
[§2](#S2.p4.1).

-
[66]
S. Yang, Z. Fu, Z. Cao, G. Junde, P. Wensing, W. Zhang, and H. Chen (2025)

Multi-loco: unifying multi-embodiment legged locomotion via reinforcement learning augmented diffusion.

In Proceedings of the 9th Conference on Robot Learning (CoRL 2025),

Proceedings of Machine Learning Research, Vol. 305, pp. 1030–1048.

External Links: [Link](https://proceedings.mlr.press/v305/)

Cited by: [§2](#S2.p2.1).

-
[67]
J. Ye, L. Wei, G. Jiang, C. Jing, X. Zou, and X. Wang (2025)

From power to precision: learning fine-grained dexterity for multi-fingered robotic hands.

arXiv preprint arXiv:2511.13710.

Cited by: [§2](#S2.p1.1).

-
[68]
Z. Yin, C. Wang, L. Pineda, K. Bodduluri, T. Wu, P. Abbeel, and M. Mukadam (2025)

Geometric retargeting: a principled, ultrafast neural hand retargeting algorithm.

arXiv preprint arXiv:2503.07541.

External Links: [Link](http://arxiv.org/abs/2503.07541)

Cited by: [§2](#S2.p3.1).

-
[69]
J. Yu, H. Liu, Q. Yu, J. Ren, C. Hao, H. Ding, G. Huang, G. Huang, Y. Song, P. Cai, et al. (2025)

ForceVLA: enhancing vla models with a force-aware moe for contact-rich manipulation.

In Advances in Neural Information Processing Systems (NeurIPS),

Note: Poster

Cited by: [§2](#S2.p1.1).

-
[70]
K. Zakka, A. Zeng, P. Florence, J. Tompson, J. Bohg, and D. Dwibedi (2021)

XIRL: cross-embodiment inverse reinforcement learning.

In Proceedings of the 5th Conference on Robot Learning (CoRL 2021),

Proceedings of Machine Learning Research, Vol. 164, pp. 537–546.

External Links: [Link](https://proceedings.mlr.press/v164/)

Cited by: [§2](#S2.p2.1).

-
[71]
Y. Zhang, C. Wang, O. Lu, Y. Zhao, Y. Ge, Z. Sun, X. Li, C. Zhang, C. Bai, and X. Li (2025)

Align-then-steer: adapting the vision-language action models through unified latent guidance.

arXiv preprint arXiv:2509.02055.

External Links: [Link](https://arxiv.org/abs/2509.02055)

Cited by: [Table 1](#S1.T1.7.9.1.1.1),
[§2](#S2.p2.1),
[§2](#S2.p4.1).

-
[72]
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)

Learning fine-grained bimanual manipulation with low-cost hardware.

arXiv preprint arXiv:2304.13705.

Cited by: [§2](#S2.p5.1).

-
[73]
J. Zheng, J. Li, D. Liu, Y. Zheng, Z. Wang, Z. Ou, Y. Liu, J. Liu, Y. Zhang, and X. Zhan (2025)

Universal actions for enhanced embodied foundation models.

arXiv preprint arXiv:2501.10105.

External Links: [Link](https://arxiv.org/abs/2501.10105)

Cited by: [§2](#S2.p2.1).

-
[74]
J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, Y. Zhang, J. Pang, J. Liu, T. Wang, and X. Zhan (2025)

X-vla: soft-prompted transformer as scalable cross-embodiment vision-language-action model.

arXiv preprint arXiv:2510.10274.

External Links: [Link](https://arxiv.org/abs/2510.10274)

Cited by: [§2](#S2.p2.1).

-
[75]
H. Zhi, P. Chen, S. Zhou, Y. Dong, Q. Wu, L. Han, and M. Tan (2025)

3DFlowAction: learning cross-embodiment manipulation from 3d flow world model.

arXiv preprint arXiv:2506.06199.

External Links: [Link](https://arxiv.org/abs/2506.06199)

Cited by: [§2](#S2.p2.1).

Cross-Hand Latent Representation for Vision-Language-Action Models

## 6 Appendix

### 6.1 Latent Visualizations

In addition to Figure 5, we include visualizations of additional hands in Figure [7](#S6.F7). This figure illustrates how the same latent representation is decoded across all four hands featured in our main paper. Furthermore, Figure [17](#S6.F17) presents a continuous trajectory rendered for all hands, with the X-Hand highlighted for clarity.

![Figure](2603.10158v1/imgs/demo_transparent_1.png)

*(a) X-Hand*

![Figure](2603.10158v1/imgs/demo_transparent_2.png)

*(b) Inspire Hand*

![Figure](2603.10158v1/imgs/demo_transparent_3.png)

*(c) Paxini Hand*

![Figure](2603.10158v1/imgs/demo_transparent_4.png)

*(d) Ability Hand*

Figure 7: More Latent Visualizations. Latent decoding results cross embodiment.

### 6.2 Hardware Setup

Tabletop Scene Description. For the real-world experiments, we use a bimanual arm with tabletop settings. The arms are mounted on the edge of the table. The distance between the arms are 80cm. Each hand is connected to the end effector of the arm with 3D-printed mounts. Figure [8](#S6.F8) shows the real-world tabletop scene with a pair of XHand, and figure [9](#S6.F9) includes all the dexterous hands we use in our experiments.

Camera Description.
We use a single RealSense L515 camera (round-shaped) mounted in front of the bimanual arms as the input view of the policy training. The camera pose is shown in figure [8](#S6.F8) and the camera view is in figure [10](#S6.F10). The raw resolution of RGB recordings from the camera is 960 $\times$ 540.

![Figure](2603.10158v1/imgs/camera_setup.jpeg)

*Figure 8: xArm Camera Setup. We use a single RealSense L515 camera with the front view. Note that the D435 camera here is not used for XL-VLA.*

![Figure](2603.10158v1/imgs/DexHands.jpg)

*Figure 9: Dexterous Hands. We use 4 kinds of hands, with various shapes, scales, degrees of freedom, and actuated joints.*

![Figure](2603.10158v1/imgs/rs_view.png)

*Figure 10: xArm Camera View. This is what our camera sees and also the input for XL-VLA and all the baseline methods.*

Humanoid Scene Description. Similar to xArm, we let G1 stand in front of a table. We use the same camera setting and mount L515 on the chest of G1 to have an egocentric view. Consider the mechanic design of G1, we only use the Inspire hand since it is light. Figure [11](#S6.F11) and [12](#S6.F12) show the real-world G1 setting.

![Figure](2603.10158v1/imgs/g1_scene.jpeg)

*Figure 11: G1 Scene. We mount an L515 camera near the neck of G1 to have an egocentric view.*

![Figure](2603.10158v1/imgs/g1_camera_view.png)

*Figure 12: G1 Egocentric Camera View.*

Object Description.
To demonstrate that XL-VLA is capable of doing various manipulation tasks, we use diverse objects in our experiments, most of which are common everyday objects from existing datasets [[[20](#bib.bib77), [10](#bib.bib78)]]. The objects vary in scale, shape, texture, weight, etc. All the objects used in listed in figure [13](#S6.F13).

![Figure](2603.10158v1/imgs/objects.jpg)

*Figure 13: Objects. We use various everyday objects from existing datasets. They vary in scale, shape, texture, weight, etc., thus requiring the manipulation policy to be robust.*

### 6.3 Policy Learning Details

xArm Data Collection.
We use Apple Vision Pro as the data collection tool [[[18](#bib.bib74)]], shown in figure [14](#S6.F14). During data collection, our teleoperator wears the VR headset and get the tracked hand poses and wrist poses from the headset. We use these data to do robot hand retargeting and Inverse Kinematics.

Unitree G1 Data Collection.
G1 is standing in front of a table, and we adapt HOMIE [[[4](#bib.bib86)]] and ACE-F [[[64](#bib.bib85)]] to do the upper-body teleoperation. We only use the upper-body system from HOMIE and replace their glove with the MANUS Mocap glove.

![Figure](2603.10158v1/imgs/avp.jpeg)

*Figure 14: Apple Vision Por for Data Collection. Apple Vision Pro is used to track the human teleoperator’s hands and wrists. Then the tracked data is processed with retargeting and Inverse Kinematics to control the real-world robots.*

![Figure](2603.10158v1/x5.jpeg)

*Figure 15: G1 Teleoperation System. We build the G1 upper-body teleoperation system from HOMIE [[[4](#bib.bib86)]]. We use a pair of MANUS Mocap glove to track the human hand pose.*

Task Visualizations.
Using XHand as an example, we show the real-world task visualizations in figure [16](#S6.F16). Four of the tasks are also tested on G1.

![Figure](2603.10158v1/imgs/task_vis/001.png)

![Figure](2603.10158v1/imgs/task_vis/002.png)

![Figure](2603.10158v1/imgs/task_vis/003.png)

![Figure](2603.10158v1/imgs/task_vis/004.png)

![Figure](2603.10158v1/imgs/task_vis/005.png)

![Figure](2603.10158v1/imgs/task_vis/006.png)

![Figure](2603.10158v1/imgs/task_vis/007.png)

![Figure](2603.10158v1/imgs/task_vis/008.png)

![Figure](2603.10158v1/imgs/task_vis/009.png)

![Figure](2603.10158v1/imgs/task_vis/010.png)

Figure 16: Task Visualizations. We design 10 various tasks to test all the models. The tasks require varied manipulation skills and have different difficulties.

Model Training.
Besides the model training details provided in the main paper, here are some more training details. RGB images is cropped and then resized from 960 $\times$ 540 to 320 $\times$ 240 during data post-processing. When loaded to train XL-VLA, they are resized to 224 $\times$ 224. We use natural language description as the task specification, which is part of the policy condition. The training usually takes around 10 hours for one multi-task policy.

Policy Evaluation.
For the real-world evaluation, we do 10 trials for each experiment setting. Among these trials, object positions are randomly initialized while the initial joint position for the robot arm and hand remains the same for the same hand.

For unseen tasks only, we record the partial success rate (PSR). If any of the bimanual robot arm finishes its task and the whole task is failed, the overall success rate is 0.5. For other experiments, we do not use PSR. Only rollout that completes a specified task is count as a success.

![Figure](2603.10158v1/imgs/latent_traj.png)

*Figure 17: Latent Visualization of a Grasping Trajectory. A trajectory is shown here with all the robot hands.*

### 6.4 G1 Experiment Results

Table [6](#S6.T6) is the numeric results for figure [5](#S4.F5).

*Table 6: G1 Policy Performances.*



Experimental support, please
[view the build logs](./2603.10158v1/__stdout.txt)
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