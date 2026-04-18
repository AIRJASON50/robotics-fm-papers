[# DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping Yifan Zhong1,2\equalcontrib, Xuchuan Huang1,2\equalcontrib, Ruochong Li2,3, Ceyao Zhang1,2, Zhang Chen1,2, Tianrui Guan1,2 Fanlian Zeng2,4, Ka Nam Lui1,2, Yuyao Ye1,2, Yitao Liang1,2, Yaodong Yang1,2${\dagger}$, Yuanpei Chen1,2 Corresponding author emails: yuanpei.chen312@gmail.com, yaodong.yang@pku.edu.cn. ###### Abstract Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on restrictive assumptions, such as single-object settings or limited environments, showing constrained *generalization*. We present DexGraspVLA, a hierarchical framework for robust generalization in language-guided general dexterous grasping and beyond. It utilizes a pre-trained Vision-Language model as the high-level planner and learns a diffusion-based low-level Action controller. The key insight to achieve *generalization* lies in iteratively transforming diverse language and visual inputs into domain-invariant representations via foundation models, where imitation learning can be effectively applied due to the alleviation of domain shift. Notably, our method achieves a *90+%* dexterous grasping success rate under *thousands of* challenging unseen cluttered scenes. Empirical analysis confirms the *consistency* of internal model behavior across environmental *variations*, validating our design. DexGraspVLA also, for the first time, simultaneously demonstrates free-form long-horizon prompt execution, robustness to adversarial objects and human disturbance, and failure recovery. Extended application to nonprehensile grasping further proves its generality. Project website: https://dexgraspvla.github.io](https://dexgraspvla.github.io).

## 1 Introduction

*Figure 1: We propose DexGraspVLA, a hierarchical VLA framework that reaches a 90+% dexterous grasping success rate under thousands of unseen cluttered scenes in a “zero-shot” environment. It robustly handles adversarial objects, human disturbance, failure recovery, and free-form long-horizon prompts. Extended application to nonprehensile grasping further proves its generality.*

Dexterous multi-fingered hands, as versatile robotic end-effectors, have demonstrated remarkable capabilities across various manipulation tasks [qi2023general, huang2023dynamic, lin2024twisting, chen2022towards, zakka2023robopianist, chen2023sequential]. Among these, grasping serves as the most fundamental prerequisite, yet remains one of the most challenging problems. Existing dexterous grasping approaches primarily consider isolated objects or simplified settings. Nevertheless, real-world applications demand more general grasping capabilities that can function reliably in diverse unseen scenarios, which presents multifaceted challenges. At the object level, the policy must generalize across diverse physical properties including geometries, masses, textures, and orientations. Beyond object characteristics, the system must also demonstrate robustness to various environmental factors, such as lighting conditions, background complexities, and potential disturbances. Compounding these challenges, cluttered scenarios further demand sophisticated reasoning capabilities, as planning the optimal sequence to grasp all objects becomes a crucial cognitive task that extends beyond simple grasp execution.

One line of research adopts a two-stage pipeline: first predicting a grasp pose from single-frame perception, then executing open-loop motion planning to reach the pose [chen2024springgrasp, turpin2023fast, turpin2022grasp]. However, these methods rely heavily on precise calibration and mechanical accuracy. By contrast, end-to-end paradigms, such as imitation and reinforcement learning, enable closed-loop control by continuously adjusting actions based on real-time feedback, offering more robust and adaptive solutions. Reinforcement learning has achieved notable successes in simulation [akkaya2019solving, yang2024anyrotate, pitz2023dextrous, handa2022dextreme], but simulating real-world physical complexity remains challenging, resulting in an inevitable sim-to-real gap. Imitation learning learns directly from human demonstrations and avoids this gap, but often struggles to generalize beyond the training data. This issue is further compounded by the impracticality of collecting expert trajectories across the full spectrum of objects and environmental variations required for general grasping. As a result, a key challenge is how to effectively leverage limited expert data to achieve broad generalization*.

The rapid emergence of vision and language foundation models [oquab2023dinov2, radford2021learning, hurst2024gpt, kirillov2023segment] presents promising opportunities for robotic manipulation. Pretrained on internet-scale data, these models exhibit remarkable world knowledge and generalization over visual and linguistic inputs. To harness these capabilities for decision making, researchers have integrated them into action generation, leading to the development of vision-language-action (VLA) models [zhong2025survey]. One straightforward approach directly trains vision-language models (VLMs) end-to-end on robot data [kim2024openvla, black2024pi0]. However, this paradigm demands massive manually collected demonstrations [o2023open] in an attempt to encompass real-world diversity and complexity. Even so, these models exhibit markedly reduced performance on unseen scenarios and still require fine-tuning to handle new conditions.
Alternatively, modular frameworks use frozen foundation models to infer task affordances more robustly across environments [huang2024rekep, huang2023voxposer, stoneopen], but their low-level policies are typically open-loop or lack generalization. Achieving generalizable closed-loop policies with foundation models remains an open challenge.

In this paper, we present DexGraspVLA, a hierarchical VLA framework for robust generalization in language-guided dexterous grasping and beyond, by integrating the complementary strengths of foundation models and imitation learning. The key idea is to leverage foundation models to iteratively transform *diverse* visual and linguistic inputs into *domain-invariant* representations, upon which imitation learning can be efficiently and effectively applied thanks to the alleviation of *domain shift*. As a result, novel scenarios no longer induce failure, as foundation models translate them into representations resembling those encountered during training—thus remaining within the learned policy’s domain. Following this principle, DexGraspVLA employs a pre-trained VLM as a high-level planner to plan the overall task and generate *domain-invariant* affordance signals. Guided by these signals, a low-level controller further refines multimodal inputs into *domain-invariant* representations using vision foundation models, and generates closed-loop action through a diffusion-based action head learned via imitation. This design combines the extensive world knowledge and generalization ability of foundation models with action modeling capacity of imitation learning, enabling strong performance in real-world scenarios.

Notably, DexGraspVLA achieves an unprecedented 90.8% success rate for grasping in cluttered scenes spanning 1,287 unseen object, lighting, and background combinations, all tested in a “zero-shot” environment. Its generalization performance significantly surpasses that of existing baselines. Moreover, DexGraspVLA robustly handles adversarial objects, human disturbances, and failure recovery. On a single-object benchmark, it achieves 98.6% success, outperforming ablated variants whose controller learns directly from raw visual inputs by at least 48%. Further analysis reveals *consistent* internal model behaviours across *varying* environments, validating our design and explaining its robustness. Beyond single-step tasks, DexGraspVLA executes free-form, long-horizon instructions with embodied reasoning, reaching 89.6% success rate. We further extend DexGraspVLA to nonprehensile object grasping [zhou2023learning], a challenging task that often requires dexterous pre-grasp maneuvers difficult for parallel grippers. DexGraspVLA achieves strong performance using only a modest number of demonstrations, further highlighting its generality across diverse manipulation scenarios. These results establish DexGraspVLA as a general, instruction-driven framework that learns from limited demonstrations and generalizes reliably to real-world settings, marking a promising step toward general dexterous grasping and beyond.

## 2 Related Work

*Figure 2: Overview of DexGraspVLA. A pre-trained VLM-based high-level planner (purple) decomposes prompts into object-level grasping instructions with bounding boxes. The diffusion-based low-level controller (pink) tracks the target mask, encodes multimodal observations (RGB images, mask, proprioception), and predicts an action chunk via a DiT model. The planner monitors execution and continually proposes new instructions based on the updated scene until the task is fully completed.*

#### Dexterous Grasping.

Dexterous grasping methods are typically divided into two-stage and end-to-end approaches. Two-stage methods first generate a grasp pose—via sampling [zhang2024dexgraspnet, fang2025anydexgraspgeneraldexterousgrasping], optimization [wang2023dexgraspnet, chen2024springgrasp], or regression [li2022hgc, liu2020deep]—and reach it with motion planning. Though they benefit from modularity and synthetic data, their open-loop nature makes them vulnerable to disturbances and calibration errors. End-to-end methods learn grasping policies via reinforcement learning in massively parallel simulation [wan2023unidexgrasp++, zhang2024graspxl, singh2024dextrah], which efficiently acquire emergent dexterity but suffer from sim-to-real gaps. In this work, we explore imitation learning from human demonstrations, which has shown promise on complex tasks [qin2022one, guzey2023see, lin2024learning]. Our core contribution is to address the central challenge of generalization in imitation learning [black2025pi]. We show that performing imitation learning on domain-invariant representations derived from foundation models enables strong generalization to unseen scenarios.

#### Foundation Models for Generalizable Robotic Policies.

Vision and language foundation models pre-trained on web-scale data have shown impressive world knowledge and generalization [kirillov2023segment, oquab2023dinov2, qwen2.5-VL], making them promising for robotics. A common approach, as seen in OpenVLA [kim2024openvla] and $\pi_{0}$ [black2024pi0], directly fine-tunes VLMs on robot data in the hope of transferring vision-language knowledge to the policy for broad generalization. However, this typically requires a massive amount of diverse demonstrations [o2023open], yet still struggles with unseen scenarios and catastrophic forgetting.
A more related line of work to ours instead leverages frozen foundation models to robustly infer task affordances—i.e., where and how to manipulate—in novel environments, guiding either motion planning [huang2024rekep, huang2023voxposer, pan2025omnimanip] or a learned action head [stoneopen]. However, the former often depends heavily on accurate calibration, involves considerable human design, or lacks robustness due to open-loop control. The latter still maps raw* visual inputs directly to actions, making it vulnerable to domain shift. In contrast, to achieve generalization across diverse real-world domains, our framework employs foundation models to iteratively transform free-form language prompts and diverse visual perceptions into domain-invariant representations. These representations enable imitation learning to be applied efficiently and effectively, collectively leading to robust generalization.

## 3 Problem Formulation

Our goal is to develop a vision-based control policy for language-guided dexterous grasping, formulated as a sequential decision-making problem.
Initially, a language instruction $l$ is given, *e.g.* “grasp the toy”, to directly specify the target object.
At each timestep $t$, the policy $\pi$ receives a first-view image $\mathbf{I}_{t}^{\mathrm{w}}\in\mathbb{R}^{H\times W\times 3}$ from the wrist camera ($H$ and $W$ denote the height and width of the image), a third-view image $\mathbf{I}_{t}^{\mathrm{h}}\in\mathbb{R}^{H\times W\times 3}$ from the head camera, and the robot proprioception $\mathbf{s}_{t}\in\mathbb{R}^{13}$ consisting of arm and hand joint angles $\mathbf{s}^{\mathrm{arm}}_{t}\in\mathbb{R}^{7},\mathbf{s}^{\mathrm{hand}}_{t}\in\mathbb{R}^{6}$.
Conditioned on these observations, the robot produces an action $\mathbf{a}_{t}=(\mathbf{a}^{\mathrm{arm}}_{t},\mathbf{a}^{\mathrm{hand}}_{t})\in\mathbb{R}^{13}$, where $\mathbf{a}^{\mathrm{arm}}_{t}\in\mathbb{R}^{7}$ and $\mathbf{a}^{\mathrm{hand}}_{t}\in\mathbb{R}^{6}$ denote the target joint angles for arm and hand respectively, by sampling from the action distribution $\pi(\cdot|\{\mathbf{I}_{j}^{\mathrm{w}}\}_{j=0}^{t},\{\mathbf{I}_{j}^{\mathrm{h}}\}_{j=0}^{t},\{\mathbf{s}_{j}\}_{j=0}^{t},l)$.
This process continues until a termination condition is reached. The robot receives a binary reward $r\in\{0,1\}$ indicating whether it has completed the instruction $l$ successfully.
The goal of the policy $\pi$ is to maximize the expected reward $\mathbb{E}_{l,\{(\mathbf{I}^{\mathrm{w}}_{j},\mathbf{I}^{\mathrm{h}}_{j},\mathbf{s}_{j},\mathbf{a}_{j})\}_{j=0}^{T}}[r].$

More generally, we consider cases where the user prompt $p$ may be a long-horizon task involving multiple grasping steps, such as “clear the table”. This requires the policy $\pi$ to reason about the prompt, decompose it into individual grasping instructions $\{l_{i}\}$, and complete them sequentially.

## 4 Methods

This section introduces DexGraspVLA, the first hierarchical VLA framework for dexterous grasping. We will first elaborate DexGraspVLA framework ([Section˜4.1](#S4.SS1)) and then detail our data collection procedure ([Section˜4.2](#S4.SS2)), which together enable the training of a dexterous grasping policy.

### 4.1 DexGraspVLA Framework

As illustrated in [Figure˜2](#S2.F2), DexGraspVLA adopts a hierarchical and modularized architecture composed of a planner and a controller.
Below we explain how each part is designed.

#### Planner.

We recognize that to achieve general dexterous grasping, the model must handle multimodal inputs, perform visual grounding, and conduct reasoning about user prompts. Building upon recent advances, we adopt an off-the-shelf pre-trained Qwen VLM [Qwen-VL, qwen2.5-VL] as a high-level planner to dynamically plan and monitor the dexterous grasping workflow. Given a user prompt $p$ (e.g., “clear the table”), the planner proposes a grasping instruction $l$ (e.g., “grasp the cookie”) as the first step.

For each $l$, the planner guides the low-level controller by marking the target object bounding box $(x_{1},y_{1},x_{2},y_{2})$ as task affordance in the head camera image $\mathbf{I}_{t_{0}}^{\mathrm{h}}$ at the initial timestep $t_{0}$. While the phrasing and content of language instruction can be diverse and flexible for different users and cases, *i.e.*, showing domain-variance, the bounding box is a consistent format for object localization regardless of the changes in language and visual inputs, *i.e.*, achieving domain-invariance. Thus, this transformation alleviates the learning challenge for the controller.

On issuing the bounding box, the planner monitors controller execution, resets robot after each grasp attempt, and proposes updated instruction $l$ until prompt $p$ is completed.

#### Controller.

Based on the bounding box $(x_{1},y_{1},x_{2},y_{2})$, the controller aims to grasp the intended object in cluttered environments. We feed this bounding box as input to SAM [kirillov2023segment] to obtain an initial binary mask $\mathbf{m}_{0}\in\{0,1\}^{H\times W\times 1}$ of the target object and then use Cutie [cheng2024putting] to continuously track the mask over time, producing $\mathbf{m}_{t}$ at each timestep $t$. This ensures accurate identification in cluttered scenes throughout the process. The problem is to learn the policy $\pi$ that effectively models the action distribution $\pi(\cdot|\mathbf{I}_{t}^{\mathrm{w}},\mathbf{I}_{t}^{\mathrm{h}},\mathbf{s}_{t},\mathbf{m}_{t})$.

To achieve general-purpose dexterous grasping, the system must generalize effectively across diverse real-world scenarios. However, the high variability in raw visual inputs $\mathbf{I}_{t}^{\mathrm{w}},\mathbf{I}_{t}^{\mathrm{h}}$ poses a fundamental challenge to learning task-critical representations. Traditional imitation learning approaches often fail catastrophically even under minor variations in objects or environmental conditions.
To address this issue, our solution is again to convert potentially *domain-varying* inputs into *domain-invariant* representations suitable for imitation learning.
We recognize that *while pixel-level perception vary widely, the fine-grained semantic features extracted by foundation models tend to be more robust and consistent* [tang2023emergentcorrespondenceimagediffusion, wang2023sparsedff].
Thus, we utilize a feature extractor $\phi$, DINOv2 [oquab2023dinov2], to obtain features from raw images. At timestep $t$, we obtain head camera image features
$\mathbf{z}^{\mathrm{h}}_{t}=\phi^{\mathrm{h}}(\mathbf{I}^{\mathrm{h}}_{t})\in\mathbb{R}^{L^{\mathrm{h}}\times D^{\mathrm{h}}},$
and wrist camera image features
$\mathbf{z}^{\mathrm{w}}_{t}=\phi^{\mathrm{w}}(\mathbf{I}^{\mathrm{w}}_{t})\in\mathbb{R}^{L^{\mathrm{w}}\times D^{\mathrm{w}}},$
where $L^{\mathrm{h}},D^{\mathrm{h}},L^{\mathrm{w}},D^{\mathrm{w}}$ denote length and hidden dimension of the feature sequences for head and wrist respectively. As we show in [Section˜5.5](#S5.SS5), these extracted features remain comparatively invariant to distracting visual factors.

Up to now, raw language and vision inputs, including instruction $l$ and images $\mathbf{I}_{t}^{\mathrm{w}},\mathbf{I}_{t}^{\mathrm{h}}$, have been iteratively transformed into domain-invariant representations, including mask $\mathbf{m}_{t}$ and features $\mathbf{z}^{\mathrm{h}}_{t},\mathbf{z}^{\mathrm{w}}_{t}$, by leveraging foundation models. This lays the stage for imitation learning. We now learn the policy $\pi$ that predicts an action chunk of horizon $H$ conditioning on these representations.

To fuse the object mask with head camera features, we project $\mathbf{m}_{t}$ into the head image feature space using a randomly initialized ViT, producing $\mathbf{z}^{\mathrm{m}}_{t}\in\mathbb{R}^{L^{\mathrm{h}}\times D^{\mathrm{h}}}$, and concatenate it with $\mathbf{z}^{\mathrm{h}}_{t}$ patch-wise to obtain
$\bar{\mathbf{z}}^{\mathrm{h}}_{t}\in\mathbb{R}^{L^{\mathrm{h}}\times 2D^{\mathrm{h}}}.$
Subsequently, we map $\bar{\mathbf{z}}^{\mathrm{h}}_{t}$, wrist-camera features $\mathbf{z}^{\mathrm{w}}_{t}$, and robot state $\mathbf{s}_{t}$ into a common embedding space with separate MLPs, yielding $\tilde{\mathbf{z}}^{\mathrm{h}}_{t}$, $\tilde{\mathbf{z}}^{\mathrm{w}}_{t}$, and $\tilde{\mathbf{z}}^{\mathrm{s}}_{t}$. These embeddings are then concatenated to form the full observation feature sequence
$\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}\in\mathbb{R}^{(1+L^{\mathrm{h}}+L^{\mathrm{w}})\times D}.$

For action prediction, we employ a DiT [peebles2023scalable] to generate multi-step actions, following the diffusion policy paradigm [chi2023diffusion, liu2024rdt]. At each timestep $t$, we bundle the next $H$ actions into a chunk $\mathbf{A}_{t}=\mathbf{a}_{t:t+H}=[\mathbf{a}_{t},\mathbf{a}_{t+1},\dots,\mathbf{a}_{t+H-1}]$. During training, a random diffusion step $t^{d}=k$ is sampled,
and Gaussian noise $\boldsymbol{\epsilon}$ is added to $\mathbf{A}_{t}$, yielding the noised action tokens $\mathbf{x}_{k}=\alpha_{k}\mathbf{A}_{t}+\sigma_{k}\boldsymbol{\epsilon},$
where $\alpha_{k}$ and $\sigma_{k}$ are DDPM coefficients. We then feed $\mathbf{x}_{k}$ into the DiT alongside the observation feature sequence $\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}$. Each DiT layer performs bidirectional self-attention over the action tokens, cross-attention to $\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}$, and MLP transformations, ultimately predicting the original noise $\boldsymbol{\epsilon}$. By minimizing the noise prediction error, the model learns to reconstruct the ground-truth action chunk $\mathbf{A}_{t}$. At inference time, iterative denoising steps recover the intended multi-step action sequence from the learned distribution, enabling imitation of multimodal behaviors. We also employ the receding horizon control strategy that only executes the first $H_{a}$ actions before generating a new action chunk prediction, enhancing responsiveness.

Overall, DexGraspVLA performs imitation learning on *domain-invariant* representations derived from *domain-varying* inputs via foundation models. This approach leverages the world knowledge and generalization capabilities of foundation models while effectively capturing the mapping from these abstracted representations to action output.

### 4.2 Data Collection

To train our dexterous grasping policy, we manually collect a dataset consisting of 2,094 successful demonstrations in cluttered scenes using 36 household objects varying in size, weight, geometry, texture, material, and category. Each episode
$\tau=\{(\mathbf{I}^{\mathrm{h}}_{t},\mathbf{I}^{\mathrm{w}}_{t},\mathbf{s}_{t},\mathbf{m}_{t},\mathbf{a}_{t})\}_{t=0}^{T}$
records raw camera images $\mathbf{I}^{\mathrm{h}}_{t},\mathbf{I}^{\mathrm{w}}_{t}$, robot proprioception $\mathbf{s}_{t}$, object mask $\mathbf{m}_{t}$, and action $\mathbf{a}_{t}$ at each timestep $t$. The mask $\mathbf{m}_{t}$ is labeled in the same way as in the controller.
For each object, we place it randomly and collect multiple grasping demonstrations, with the surrounding objects randomized between episodes.
These demonstrations are performed at typical human motion speeds, taking about $3.5\text{\,}\mathrm{s}$ each. They undergo rigorous inspection to ensure quality. The DexGraspVLA controller is trained on this dataset with imitation learning.

## 5 Experiments

In this section, we extensively evaluate DexGraspVLA. All experiments are conducted in a different environment from the demonstration setup, ensuring a "*zero-shot*" setting to rigorously assess generalization to novel real-world scenarios. Our experiments seek to address the following questions: (1) Large-scale Generalization ([Section˜5.2](#S5.SS2)): Can DexGraspVLA generalize to thousands of unseen object, lighting, and background combinations? (2) Baseline Comparison ([Section˜5.3](#S5.SS3)): How does its performance compare to baselines? (3) Ablation Study ([Section˜5.4](#S5.SS4)): How much does imitation learning on domain-invariant representations improve generalization? (4) Mechanism Analysis ([Section˜5.5](#S5.SS5)): Are its internal model behaviors consistent under varying environments? (5) Long-horizon Task ([Section˜5.6](#S5.SS6)): How effectively does DexGraspVLA handle free-form, long-horizon instructions? (6) Extension to Nonprehensile Grasping ([Section˜5.7](#S5.SS7)): Can it be extended to other dexterous manipulation skills beyond grasping?

### 5.1 Experiment Setups

#### Hardware Platform.

As shown in [Figure˜3](#S5.F3), our setup includes a 7-DoF RealMan RM75-6F arm and a 6-DoF PsiBot G0-R hand. A wrist-mounted RealSense D405C camera provides a first-person view, and a head-mounted D435 camera captures a third-person view. Objects are placed on a table in front, and the control frequency is 20 Hz.

#### Baselines.

We compare DexGraspVLA (Ours) with several state-of-the-art (SOTA) VLA baselines fine-tuned on our dataset, including full-parameter (Full FT) and LoRA fine-tuned variants of $\pi_{0}$ [black2024pi0], RDT [liu2024rdt], OpenVLA [kim2024openvla], and OpenVLA-OFT [kim2025fine]. We also evaluate two ablated versions of our method: 1) DINOv2-train: Identical to DexGraspVLA but with trainable DINOv2 encoders. 2) ViT-small: Identical to DexGraspVLA but replaces DINOv2 with smaller, trainable ViTs. Empirically, the ViT-small variant represents an enhanced version of Diffusion Policy [chi2023diffusion], a SOTA imitation learning baseline. For all experiments, the high-level planner is based on Qwen-VL-Chat [Qwen-VL], except in the long-horizon task ([Section˜5.6](#S5.SS6)), where we use Qwen2.5-VL-72B-Instruct [qwen2.5-VL]. Implementation details are in [Appendix˜A](#A1).
To account for inference randomness, we report Ours@$k$ ($k=1,2,3$) in [Section˜5.2](#S5.SS2), where up to $k$ attempts are allowed per test. Ours@1 is equivalent to Ours. Re-grasps performed by the policy after an initial failure within a single attempt are allowed and not counted separately.

*(a) Dexterous grasping in cluttered scenes.*

*(b) Long-horizon task performance of DexGraspVLA.*

*(c) Large-scale generalization evaluation of DexGraspVLA on dexterous grasping.*

*(d) Ablation results on single-object grasping.*

*(e) Nonprehensile grasping performance.*

Table 1: Comprehensive evaluation of DexGraspVLA and baselines across tasks. Bgs.: Backgrounds; Aggr.: Aggregated.

### 5.2 Large-Scale Generalization Evaluation

*Figure 3: Our hardware platform.*

#### Tasks.

We curate 360 unseen objects, 6 unseen backgrounds, and 3 unseen lighting conditions.
The objects span diverse sizes, weights, geometries, textures, materials, and categories, while remaining graspable by our dexterous hand. Backgrounds and lighting conditions are selected to be visually distinct.
We evaluate generalization through three grasping tasks in cluttered scenes (around 6 objects per scene): (1) Unseen Objects*: Each of the 360 objects is grasped once in a random scene on a white table under white light (360 tests). (2) *Unseen Backgrounds*: A subset of 103 objects $\mathcal{S}$ forms 103 scenes per background under white light, totaling 618 tests. (3) *Unseen Lightings*: The same $\mathcal{S}$ forms 103 scenes per lighting condition on a white table (309 tests). Details can be found in [Appendix˜B](#A2).

#### Metric.

A grasp is successful if the object is held $10\text{\,}\mathrm{cm}$ above the table for $20\text{\,}\mathrm{s}$. Success rate is the ratio of successes to total tests; aggregated performance is a weighted average by task proportion.

#### Results.

We present the quantitative results in [Table˜1(c)](#S5.T1.st3). From the 1st row (“Ours@1”), DexGraspVLA achieves a 91.1% single-attempt success rate on 360 unseen objects, 90.5% on 6 unseen backgrounds, and 90.9% under 3 unseen lighting conditions, yielding a 90.8% aggregated success rate.
These results demonstrate robust and accurate control of the dexterous hand to grasp specified objects from clutter in diverse unseen conditions, without domain-specific fine-tuning. This highlights strong generalization and suggests that our framework substantially alleviates the overfitting challenge in imitation learning. We further analyze the source of this generalization in [Section˜5.5](#S5.SS5) and extend its application in [Section˜5.7](#S5.SS7).

Qualitatively, DexGraspVLA robustly handles challenging cases involving transparent, deformable, reflective, or background-camouflaged objects. It also dexterously adapts to diverse geometries and poses — e.g., grasping a bottle from the side, picking up a small earbud case from the top, or retrieving an awkwardly placed box. The closed-loop policy enables re-grasping after failed attempts and tolerates human-induced perturbations by tracking object motion. Such robustness stems from three factors: first, foundation-model-based perception ensures semantic consistency under appearance variation; second, imitation learning avoids the need for explicit object modeling; and third, diffusion-based action head captures multi-modal action distributions.

From the 2nd and 3rd rows (“Ours@2” and “Ours@3”), we observe that allowing up to three attempts further boosts performance to 96.9%, indicating the capacity to reach even higher success rates.
Finally, our model takes around $6\text{\,}\mathrm{s}$ to grasp an object, which is close to that of humans and ensures practical usability in real-world scenarios.

### 5.3 Baseline Comparison

#### Tasks & Metrics.

We adopt the same setup as [Section˜5.2](#S5.SS2) but on a smaller scale for baseline comparison. The tasks involve 24 unseen objects, 2 unseen backgrounds, and 2 unseen lighting conditions. We also include 12 seen objects under white background and lighting (*Seen Objects*). Metrics remain unchanged; details are in [Appendix˜B](#A2).

#### Results.

As shown in [Table˜1(a)](#S5.T1.st1), DexGraspVLA consistently achieves 90+% success across all settings, significantly outperforming fine-tuned VLA models. While $\pi_{0}$ (Full FT) reaches 75% on seen objects, its performance drops sharply under visual variations. Similar declines are observed for $\pi_{0}$ (LoRA) and OpenVLA (LoRA), suggesting overfitting to training language and visual domains. Notably, RDT also uses frozen vision and language foundation models like ours and shows more consistent performance, but still falls short. This suggests that bounding boxes offer stronger grounding than language encoding, and that DINOv2 better preserves visual details than SigLIP [zhai2023sigmoid]. Overall, these results validate the design of DexGraspVLA and its superior generalization performance.

### 5.4 Ablation Study

#### Tasks & Metrics.

To compare DexGraspVLA with ablated variants that learn directly from raw visual inputs without frozen vision encoders, we conduct single-object grasping experiments using 13 seen and 8 unseen objects. Each object is tested at five table locations with two trials per location, yielding 210 tests under white tabletop and lighting. Success rates are computed as in [Section˜5.2](#S5.SS2).

#### Results.

[Table˜1(d)](#S5.T1.st4) shows that DexGraspVLA (Ours) consistently achieves over 98% success on both seen and unseen objects, significantly outperforming DINOv2-train and ViT-small variants. Its near-perfect performance in a zero-shot setting indicates strong robustness to domain shift. Interestingly, performance on unseen objects slightly exceeds that on seen ones, suggesting that the model learns the grasping task itself rather than overfitting to training data. In contrast, baselines that map raw inputs to actions fail to generalize, as perceptual changes easily push them out of distribution.

### 5.5 Internal Model Behavior Analysis

To further validate our design, we examine whether internal model behavior remains consistent under varying visual conditions, as shown in [Figure˜4](#S5.F4). We test DexGraspVLA on the same cluttered scene (9 objects, target: “grasp the blue yogurt in the middle”) across four environments: a white table, a calibration board, a colorful tablecloth, and the same tablecloth under disco lighting. For clarity, we display only the tabletop region; full images are in [Appendix˜B](#A2). While the head images (1st row) appear to be markedly diverse, the DINOv2 features (2nd row) look rather consistent. These features are visualized by mapping principal components to RGB channels as done in [oquab2023dinov2]. Across environments, the object properties are robustly maintained and matched, which fundamentally allows DexGraspVLA trained on a single data domain to generalize. The third row shows that Cutie accurately tracks the object, providing the correct guidance to the controller. Based on the domain-invariant mask and the DINOv2 features, the DiT action head now predicts the subsequent actions. In the fourth row, we average and normalize all cross-attentions to the head image from DiT. We find that all attention maps exhibit the same behavior of focusing on the target object instead of being distracted by environments. The fifth row overlays the attention map on the raw image to confirm the reasonable attention pattern. All visualization details are provided in [Appendix˜B](#A2). Therefore, we substantiate that DexGraspVLA indeed transforms perceptually diverse raw inputs into invariant representations, on which it effectively applies imitation learning to model the data distribution, explaining its superior generalization performance.

### 5.6 Long-Horizon Task Evaluation

#### Tasks.

We evaluate DexGraspVLA on long-horizon grasping tasks. We use four types of prompts—“Clear the table”, “Grasp all bottles”, “Grasp all green objects”, and “Grasp all food”—which require commonsense and physical reasoning to identify targets sequentially. Each prompt is evaluated in 24 randomly configured cluttered scenes. “Clear the table” scenes include three unseen objects; others involve 3–4 unseen objects, with two being relevant.

#### Metric.

For each task, we report the task success rate as the proportion of tests that fully complete all required stages. We further report the average grasping attempts per object in the successful tests, along with success rates for instruction proposal, bounding box prediction, completion check of the planner, and grasp execution of the controller.

#### Results.

[Table˜1(b)](#S5.T1.st2) shows that DexGraspVLA achieves an 89.6% aggregated task success rate across four long-horizon prompts, with each target object attempted slightly more than once. The high-level planner grounds prompt semantics on the observation and proposes correct instructions with a 94.3% average success rate. Its bounding box prediction accuracy is consistently above 98%, which we further substantiate with evaluations in distraction conditions in [Appendix˜D](#A4). The low-level controller, leveraging its robust and generalizable grasping policy, executes individual grasps with over 91% success, enabling reliable multi-step completion. Additionally, the planner detects task completion with over 94% accuracy, preventing redundant actions. These results highlight the synergy between the high-level and low-level modules in DexGraspVLA, showcasing the effectiveness of its hierarchical framework for long-horizon tasks. An example can be found in [Appendix˜C](#A3).

### 5.7 Extension to Nonprehensile Grasping

*Figure 4: DexGraspVLA is robust to environmental variations. The same cluttered scene (1st row) is arranged in four visually different environments (four columns). DINOv2 features (2nd row), masks (3rd row), and attention maps (4th row) are consistent across variations. The 5th row confirms DexGraspVLA is attending to the correct object.*

#### Tasks & Metric.

To show applicability beyond dexterous grasping, we apply DexGraspVLA to a nonprehensile grasping task ([Figure˜1](#S1.F1) last row). We curate 32 flat, wide-surface objects (e.g., plates, boxes, and books) that are difficult to grasp directly and collect 1,029 human demonstrations in cluttered scenes. In these demos, the robot first performs a pre-grasp manipulation by pushing the object toward the table edge, creating an accessible pose, and then executes a final grasp.
We keep the DexGraspVLA planner unchanged and train the controller on this dataset; details are provided in Appendix [A](#A1). To evaluate generalization, we curate 18 unseen nonprehensile objects and design three types of tasks: (1) Unseen Objects* (36 tests): Each object is placed in two cluttered scenes with varying poses on a white table under white light. (2) *Unseen Lighting* (36 tests): Same protocol under disco light. (3) *Unseen Backgrounds* (72 tests): Same protocol on a wooden tabletop or a yellow tablecloth.
Success rates are reported as in [Section˜5.2](#S5.SS2).

#### Results.

As shown in [Table˜1(e)](#S5.T1.st5), DexGraspVLA achieves an aggregated generalization performance of 84.7% in the nonprehensile grasping task, showing strong robustness to unseen object appearances, shapes, physical properties, as well as novel backgrounds and lightings—significantly outperforming ablated variants. We observe that DexGraspVLA reliably adapts to object poses, pushing until it extends sufficiently over the edge, followed by a stable grasp. This task is particularly challenging for parallel-jaw grippers, highlighting the dexterity we exhibit. Moreover, DexGraspVLA seamlessly extends to this new task without architectural changes, reflecting three key aspects of generality: (1) the high-level planner’s grounding and reasoning ability; (2) the use of bounding boxes as affordance guidance; and (3) applying imitation learning on domain-invariant representations iteratively obtained from foundation models.

## 6 Limitation and Conclusion

This paper presents DexGraspVLA, a hierarchical VLA framework aiming for robust generalization in language-guided dexterous grasping and beyond. By leveraging a pre-trained VLM as the high-level planner and vision foundation models in the low-level controller, the system transforms multimodal inputs into domain-invariant representations and learns robust closed-loop policies via imitation learning. Our large-scale evaluations show over 90% grasping success across thousands of unseen cluttered scenes in a zero-shot setting, with empirical evidence of consistent internal behavior. DexGraspVLA also handles free-form long-horizon prompts, recovers from failures, and extends to nonprehensile grasping, demonstrating broad applicability. While effective, it does not yet address functional grasping and subsequent manipulation, nor does it incorporate tactile sensing. In future work, we aim to extend the high-level planner to generate more fine-grained affordance and learn a task-oriented manipulation controller that also integrates tactile feedback, further broadening the scope of DexGraspVLA.

## Appendix A Implementation Details

In this section, we present the details of DexGraspVLA implementation ([Section˜A.1](#A1.SS1)), baseline implementation ([Section˜A.2](#A1.SS2)), and dataset collection ([Section˜A.3](#A1.SS3)).

### A.1 Details of DexGraspVLA Implementation

#### Planner.

The high-level planner operates as described in [Section˜4.1](#S4.SS1). By leveraging an off-the-shelf VLM as the planner, our framework gains remarkable flexibility, enabling easy utilization of more advanced models for enhanced performance. Our observations indicate that Qwen2.5-VL-72B-Instruct [qwen2.5-VL] outperforms Qwen-VL-Chat [Qwen-VL] in reasoning and instruction following, leading to improved long-horizon task completion. Therefore, we base the DexGraspVLA planner on Qwen2.5-VL-72B-Instruct in the long-horizon tasks and provide our prompts below.

These prompts mainly instruct the VLM to function as DexGraspVLA planner via four sub-tasks, including (1) Instruction Proposal: proposing the current grasping instruction $l$ based on the user prompt $p$, (2) Bounding Box Prediction: marking the target object bounding box, (3) Grasp Outcome Verification: checking if the grasp has succeeded, and (4) Prompt Completion Check: evaluating whether the entire user prompt is fully fulfilled.
Since instruction proposal, bounding box prediction, and prompt completion check only require information within the operational workspace on the table, we crop the relevant region from the head camera image and fill the remaining area with white pixels. The resulting cropped image is used as the planner’s visual input for these sub-tasks.

To start with, when a user prompt $p$ is provided, the planner first determines which object in the scene should be grasped next. This step involves interpreting the prompt in context and selecting the best matching object from the current visual input.

You are controlling a robotic arm that needs to complete the following user prompt: <user_prompt>.
I will show you two images.
The initial image (before any actions) is: <initial_head_image>.
The current image (after the latest action) is: <current_head_image>.
Your task is to select the best object to grasp next from the current image.
To identify objects, use common sense and everyday knowledge to infer what each item is.
For example, recognize cups, bottles, fruits, snacks, boxes, tools, etc.

When choosing the best object to grasp, follow these principles:
1. Prefer objects on the right, then center, then left.
2. Avoid objects that are blocked or surrounded.
3. Avoid grasping objects that would cause other items to topple.
4. Select objects that best match the user prompt.

Please output ONLY ONE object that the robot should grasp next.

Return format (in English, natural language):
A short sentence precisely describing the target object, including:
- color.
- shape.
- relative position (e.g., "on the right", "in front", "next to the red box").

Example:
Grasp the blue cube on the right side of the table.

After deciding on the next object to grasp, the planner proceeds to locate this object in the image by predicting its bounding box using the following prompts. The generated grasping instruction is used as input to this localization module.

You are a robotic vision assistant. Your task is to locate the object described below in the given image: <current_head_image> and return its bounding box.

Grasping instruction: <grasping_instruction>.

Instructions:
1. Carefully read the grasping instruction and match the target object to the best-fitting visible object in the image.
2. Select EXACTLY ONE object that best matches the description.
3. For the selected object, return the following in strict JSON format:
- "bbox_2d": [x1, y1, x2, y2] (integer pixel coordinates, top-left to bottom-right)
- "label": a short 2-4 word name, (e.g. "blue cup")
- "description": a complete, natural-language description of the object’s appearance and position

Requirements:
- Only return one object.
- Coordinates must be valid and within image boundaries.
- Do not guess if the object is not visible.

During the controller’s execution, the planner verifies whether the object has been successfully grasped, using the following prompt.

I will show you two images.
The top-down view from the head camera is: <current_head_image>.
The close-up view from the wrist camera is: <current_wrist_image>.

Grasping instruction: <grasping_instruction>.

Task:
Determine whether the robotic arm has successfully grasped the target object.

You should consider:
- Whether the target object is still visible on the table.
- Whether the object is securely held in the robotic hand.

Output format:
A reasoning and a boolean value (True=successfully grasped, False=not grasped).

Keep it short and simple.

Upon a successful grasp, it triggers a scripted placing motion. After each grasp attempt, the planner resets the robot to the initial state and checks whether the user prompt has been fulfilled with the following prompt.

The robot is trying to complete the following user prompt: <user_prompt>.
I will show you two images.
The initial image (before any actions) is: <initial_head_image>.
The current image (after the latest action) is: <current_head_image>.
Please compare the two images and determine whether the user prompt has been fully completed.

Instructions:
- Only consider visible 3D objects.
- If all target objects have been removed or grasped, return True.
- If some relevant objects remain, return False.

Output format:
A reasoning and a boolean value (True=completed, False=not completed).

Example:
All blue objects have been removed from the table: True.

In our experiments, we either query the online APIs of these models or host them on an 8-A800 GPU server by ourselves with vLLM [kwon2023efficient]. When hosting Qwen2.5-VL-72B-Instruct, we employ Qwen2.5-VL-7B-Instruct for speculative decoding to accelerate inference.

#### Controller.

We first elaborate on the implementation details for the controller in the general dexterous grasping experiments. All raw images are produced by head and wrist cameras at a resolution of $640\times 480\times 3$. Correspondingly, the resolution of mask is $640\times 480\times 1$.
Through preliminary model selection, we decide to use DINOv2 ViT-B/14 as the feature extractor $\phi^{\mathrm{h}}$ for head camera images and DINOv2 ViT-L/14 as the feature extractor $\phi^{\mathrm{w}}$ for wrist camera images. Before feeding images into DINOv2, we resize them to $518\times 518\times 3$. During training, we apply domain randomization via color jittering. Finally, the images are normalized and fed into DINOv2 models. This leads to features $\mathbf{z}^{\mathrm{h}}_{t}\in\mathbb{R}^{1369\times 768}$ and $\mathbf{z}^{\mathrm{w}}_{t}\in\mathbb{R}^{1369\times 1024}$. By processing the mask $\mathbf{m}_{t}$ with a randomly initialized ViT, we extract its features $\mathbf{z}^{\mathrm{m}}_{t}\in\mathbb{R}^{1369\times 768}$. Patch-wise concatenation of $\mathbf{z}^{\mathrm{h}}_{t}$ and $\mathbf{z}^{\mathrm{m}}_{t}$ leads to $\bar{\mathbf{z}}^{\mathrm{h}}_{t}\in\mathbb{R}^{1369\times 1536}$. We then project $\bar{\mathbf{z}}^{\mathrm{h}}_{t},\mathbf{z}^{\mathrm{w}}_{t},\mathbf{s}_{t}$ to the same feature space of dimension $1024$ with separate MLPs, yielding $\tilde{\mathbf{z}}^{\mathrm{h}}_{t}\in\mathbb{R}^{1369\times 1024},\tilde{\mathbf{z}}^{\mathrm{w}}_{t}\in\mathbb{R}^{1369\times 1024},\tilde{\mathbf{z}}^{\mathrm{s}}_{t}\in\mathbb{R}^{1\times 1024}$, and concatenate them to form the full observation feature sequence $\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}=(\tilde{\mathbf{z}}^{\mathrm{h}}_{t},\tilde{\mathbf{z}}^{\mathrm{w}}_{t},\tilde{\mathbf{z}}^{\mathrm{s}}_{t})\in\mathbb{R}^{2739\times 1024}$.

For action modeling, we define an action chunk horizon of $H=64$. When we add noise to the action during training, we employ Immiscible Diffusion [li2024immiscible] to improve data-noise mapping. The noised action chunk $\hat{\mathbf{A}}_{t}$ belongs to $\mathbb{R}^{64\times 13}$.

The DiT implementation is based on the original DiT paper [peebles2023scalable], diffusion policy [chi2023diffusion], and RDT [liu2024rdt]. It first embeds the diffusion timestep to the same hidden space as $\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}$, yielding $\tilde{\mathbf{z}}^{\mathrm{d}}_{t}\in\mathbb{R}^{1\times 1024}$, and concatenates it with $\tilde{\mathbf{z}}^{\mathrm{obs}}_{t}$ to form the condition sequence $\tilde{\mathbf{z}}_{t}=(\tilde{\mathbf{z}}^{\mathrm{obs}}_{t},\tilde{\mathbf{z}}^{\mathrm{d}}_{t})\in\mathbb{R}^{2740\times 1024}$. We project the noised action chunk to the same hidden space, deriving $\tilde{\mathbf{z}}^{\mathrm{A}}_{t}\in\mathbb{R}^{64\times 1024}$, and feed it into DiT. Each DiT layer performs bi-directional attention within action tokens, cross-attention to the condition sequence, and MLP projections. Finally, the output is projected back to the action space to be the model’s prediction of noise. During training, we compute MSE loss between the noise prediction and ground truth, and back-propagate the gradient to update all trainable parameters. During inference, we start from Gaussian noise and iteratively denoise it using DDIM sampling [songdenoising]. At each step, the DiT predicts the noise given the condition sequence, and we update the action chunk using the DDIM scheduler until we obtain the final action. The controller executes the first six actions in the predicted action chunk before making a new prediction.

In total, the controller possesses 163M trainable parameters. To accelerate training, we utilize bfloat16 mixed-precision training, reducing memory usage and improving computational efficiency. Additionally, we employ FusedAdamW as the optimizer to further speed up training through optimized memory access and fused kernel execution. With these techniques, we train the controller for 84 epochs over our dataset on an 8-A800 GPU server, which takes less than one day to complete.
All hyper-parameters in our implementation are presented in [Table˜2](#A1.T2).

In the nonprehensile grasping experiments, we keep most of the hyper-parameters the same but make the following changes: we use DINOv2 ViT-B/14 as the feature extractors $\phi^{\mathrm{h}},\phi^{\mathrm{w}}$ for both head and wrist camera images, and the action horizon is set to $100$. This controller has 106M trainable parameters and is trained for 200 epochs on an 8-A800 GPU server, which takes approximately two days to finish.

*(a) Our data collection site.*

![Figure](figures/test-env.jpg)

*(b) The test environment for experiments.*

Figure 5: A comparison of the data collection and test environments, located in different rooms. The visual scenes captured by the robot’s cameras differ significantly, especially for the wrist camera.

*Table 2: Hyper-parameters of DexGraspVLA.*

### A.2 Details of Baseline Implementation

#### Baselines.

In the general dexterous grasping experiments, we fine-tune several state-of-the-art VLA models on our datasets following their official instructions.

Since our datasets do not contain language annotations, we first construct language instructions for each episode by manually annotating the target object. We then use LLMs, including GPT-4o [openai2024gpt4ocard] and Gemini 2.5 Pro [gemini25], to expand and diversify the instructions. All generated instructions are manually verified.

For $\pi_{0}$, we perform both full-parameter and LoRA fine-tuning on 8 A800 GPUs. In the LoRA setup, we use a LoRA rank of 16 for the Gemma [team2024gemma] backbone and 32 for the action expert. The action horizon is 50, with a total batch size of 256, and the model is fine-tuned for 30K steps.

For RDT, we perform full-parameter fine-tuning on 8 A800 GPUs. The action horizon is 64, with a total batch size of 256, and training runs for 200K steps.

For OpenVLA, we perform LoRA fine-tuning on 4 A800 GPUs, using a total batch size of 16, LoRA rank of 32, and 60K fine-tuning steps. Note that OpenVLA does not support action chunking.

For OpenVLA-OFT, we apply LoRA fine-tuning on 8 A800 GPUs, with a total batch size of 32, LoRA rank of 8, action horizon of 25, and a total of 30K fine-tuning steps.

#### Ablation.

In both general dexterous grasping and nonprehensile grasping experiments, DexGraspVLA (DINOv2-train) is the same as DexGraspVLA (Ours) described in [Section˜A.1](#A1.SS1) except that the two DINOv2 models are trainable instead of frozen. DexGraspVLA (ViT-small) is the same as DexGraspVLA (Ours) except that the two DINOv2 models are replaced with two small trainable pre-trained ViTs (the R26-S-32 ResNet-ViT hybrid from [steiner2022train]). Correspondingly, we resize the images to $224\times 224\times 3$ to feed them into ViT-small. Each image is split into 49 patches, and the feature dimension is 384.

### A.3 Details of Data Collection

![Figure](x4.png)

*Figure 6: Environment conditions used in our generalization evaluations of dexterous grasping ([Section˜5.2](#S5.SS2)) and nonprehensile grasping ([Section˜5.7](#S5.SS7)).*

![Figure](x5.png)

*Figure 7: (Left) A representative part of all 360 unseen objects used to evaluate DexGraspVLA. (Right) A t-SNE projection illustrating the diversity and broad coverage of these objects in length, width, height, mass (denoted by marker size), roughness (marker type), and shape (marker color).*

![Figure](figures/nonprehensile-train-objects.png)

*(a) The 32 objects used to collect nonprehensile grasping demonstrations.*

![Figure](figures/nonprehensile-test-objects.png)

*(b) The 18 objects used to test methods’ performance in nonprehensile grasping.*

Figure 8: Objects used to train and test methods in nonprehensile grasping. DexGraspVLA achieves robust generalization performance on diverse unseen objects.

We collect demonstrations through kinesthetic teaching. At the beginning, the robot is set to teaching mode, allowing manual guidance to grasp target objects. The operator then physically guides the robot to the target position and performs the grasping motion.
Subsequently, we reset the environment and execute PD control using the recorded joint angles as target. At the same frequency, these target joint angles serve as actions, while images and current joint angles are collected as states. Following the same approach as the low-level controller, we post-process the collected data to generate masks, completing one demonstration sequence. In the general dexterous grasping experiments, each episode has a fixed duration of 75 timesteps, while in nonprehensile grasping, demonstrations have variable lengths, depending on the amount of manipulation required to push the object toward the table edge and complete the grasp. The control frequency is 20Hz.

We hire external contractors, provide them with training, and engage them to assist with data collection. All contractors were compensated with fair wages.

## Appendix B Experiment Details

### B.1 The “Zero-Shot” Evaluation Environment

[Figure˜5](#A1.F5) contrasts our data collection site and the test site, which are located in separate rooms. We gather all human demonstrations at the data collection site ([Figure˜5(a)](#A1.F5.sf1)), whereas the experiments in [Section˜5](#S5) are conducted at the test site ([Figure˜5(b)](#A1.F5.sf2)). Because these sites differ in layout and background, both the head camera and the wrist camera encounter scenes not present in the training data during evaluation — particularly the wrist camera, which observes a notably altered environment, capturing a variety of front and peripheral views during operation. Despite these environmental discrepancies, we do not collect any data from the test site to fine-tune the models. Instead, the models are deployed and evaluated directly, resulting in a genuinely “zero-shot” testing environment.
Even under these conditions, DexGraspVLA achieves an over 90% success rate in grasping tasks in cluttered scenes across thousands of unseen object, lighting, and background combinations, clearly demonstrating its strong generalization capability.

### B.2 Additional Details of Objects, Lightings, and Backgrounds in General Dexterous Grasping

We collect 360 unseen objects with diverse sizes, weights, geometries, textures, materials, and categories. [Figure˜7](#A1.F7) presents the collected objects along with a t-SNE visualization of their measured properties, clearly demonstrating the high diversity of the object set. From these, 103 items are randomly selected as the object subset* $\mathcal{S}$. In the large-scale generalization evaluation in the main paper ([Section˜5.2](#S5.SS2)), the *Unseen Objects* experiment is conducted on all 360 objects, while the *Unseen Lightings* and *Unseen Backgrounds* experiments use only the objects in $\mathcal{S}$. The three unseen lighting conditions comprise disco light, lamp light, and dark light. Meanwhile, the six unseen backgrounds include a black mouse pad, a pink towel, a colorful tablecloth, a black-and-white mouse pad, a wooden board, and a calibration board. These conditions are illustrated in [Figure˜6](#A1.F6). In the baseline comparison experiments ([Section˜5.3](#S5.SS3)), the two unseen lighting conditions are disco light and lamp light, while the unseen backgrounds are a colorful tablecloth and a black-white mouse pad.

### B.3 Additional Details of Objects, Lightings, and Backgrounds in Nonprehensile Grasping

In [Figure˜8](#A1.F8), we present the 32 objects curated for collecting nonprehensile grasping demonstrations and 18 unseen objects used for evaluation, covering a wide range of appearances, geometries, sizes, and categories. In [Figure˜6](#A1.F6), we show the unseen background and lighting conditions used in the generalization evaluation. DexGraspVLA demonstrates robust performance on challenging cases, including fully white or irregularly shaped objects. In these scenarios, it successfully pushes the objects toward the table edge to enable stable grasping, even under complex and unseen lighting and background conditions.

### B.4 Details of Visualization

In this part, we explain how we visualize the internal model behavior shown in [Figure˜4](#S5.F4). Due to space constraints, [Figure˜4](#S5.F4) only presents the relevant portion of images containing the tabletop workspace. The full version is shown in [Figure˜9](#A2.F9). The first row is raw images from the head camera resized to $518\times 518\times 3$. The second row illustrates the DINOv2 ViT-B/14 features following the practice introduced in DINOv2 paper [oquab2023dinov2]. To make the resulting feature map recognizable for visualization purpose, we enlarge both the height and weight of images by a factor of six before feeding them into DINOv2. After obtaining the feature sequences for all four images, we combine these features, perform a PCA between all patches, and set a threshold to remove background regions. We then apply PCA again, this time to the remaining foreground features, map the top three principal components to the RGB channels, and normalize the result. This yields the visualization shown in the second row. The third row showcases the binary masks $\mathbf{m}_{t}\in\mathbb{R}^{518\times 518\times 1}$ tracked by Cutie. The fourth row displays the averaged DiT attention maps over the head image features. This is computed by summing attention weights to each head image patch across all diffusion steps, DiT layers, DiT heads, and action tokens, and normalize the sum to one. The shape of the averaged attention map is $37\times 37\times 1$. Finally, we upsample the attention map to $518\times 518\times 1$, multiply it by 2 to increase brightness, and use it to scale the value channel of head images in HSV space, resulting in the visualization shown in the fifth row.

*Figure 9: The complete, uncropped version of [Figure˜4](#S5.F4).*

## Appendix C Additional Results

This section provides additional results for the experiments in the main paper. In [Table˜3](#A3.T3), we report the detailed success rates for our large-scale generalization evaluation under each environment condition, corresponding to [Table˜1(c)](#S5.T1.st3) in [Section˜5.2](#S5.SS2). From the first row (“Ours@1”), it is evident that DexGraspVLA maintains consistently high success rates across various unseen object, lighting, and background combinations. Many observed failures stem from randomness in policy inference; allowing additional attempts often recovers these failed cases. Accordingly, the second and third rows (“Ours@2” and “Ours@3”) show further improvements in performance, highlighting the potential for DexGraspVLA to reach even higher success rates.

In [Table˜4](#A3.T4), we present the detailed success rates of DexGraspVLA and baseline methods on the smaller-scale dexterous grasping experiments described in [Section˜5.3](#S5.SS3). Despite being evaluated in a zero-shot setting, DexGraspVLA consistently achieves success rates around or above 90%. In contrast, baseline methods often fail catastrophically even on seen objects and perform even worse in unseen cluttered scenes. These results underscore the superior generalization capabilities of DexGraspVLA.

*Table 3: The detailed performance of DexGraspVLA under different unseen conditions, which indicates that our approach consistently achieves high success rates across various objects, lightings, and backgrounds. The second and third rows highlight its potential to reach even higher success rates given more chances. Numbers in parentheses indicate the number of tests.*

*Table 4: Performance of various methods on dexterous grasping in cluttered scenes under different conditions. DexGraspVLA consistently outperforms baselines, demonstrating superior generalization. Numbers in parentheses indicate the number of tests.*

![Figure](x7.png)

*Figure 10: The head camera images observed by DexGraspVLA planner and generated masks for a long-horizon task: "Grasp all edible objects, including food and drinks."*

We next present the generated responses of DexGraspVLA planner in an exemplar long-horizon task, which demonstrates visual understanding and embodied reasoning. The cluttered scene is arranged as shown in [Figure˜10](#A3.F10), with the user prompt being "Grasp all edible objects, including food and drinks". Below is the generated responses of DexGraspVLA planner. We omit our system prompts as they have been elaborated in [Section˜A.1](#A1.SS1).

<System Prompt of Prompt Completion Check>
……

<Planner Response>
To determine if the user prompt has been fully completed, let’s identify all the edible objects in the initial image and check if they have been removed or grasped in the current image.

### Edible Objects in the Initial Image:
1. A bottle (drink).
2. A green packet with an orange label (food).
3. A small white object with a heart symbol (possibly food).
4. A packaged item with a brown appearance (food).

### Analysis of Current Image:
- The bottle is still present on the table.
- The green packet with an orange label is still present on the table.
- The small white object with a heart symbol is still present on the table.
- The packaged item with a brown appearance is still present on the table.

Since all the edible objects are still visible on the table in the current image, none of them have been removed or grasped.

**Reasoning:** All target objects (edible items) remain on the table, indicating that the user prompt to grasp all edible objects has not been completed.

**Output:** False

<System Prompt of Instruction Proposal>
……

<Planner Response>
The green and orange snack bag on the right side of the table.

<System Prompt of Bounding Box Prediction>
……

<Planner Response>
[⬇](data:text/plain;base64,ewogICAgImJib3hfMmQiOiBbMzU3LCAxMDQsIDQ4NiwgMjQ5XSwKICAgICJsYWJlbCI6ICJncmVlbiBvcmFuZ2Ugc25hY2siLAogICAgImRlc2NyaXB0aW9uIjogIlRoZSBncmVlbiBhbmQgb3JhbmdlIHNuYWNrIGJhZyBvbiB0aGUgcmlnaHQgc2lkZSBvZiB0aGUgdGFibGUuIgp9)
{

"bbox_2d": [357, 104, 486, 249],

"label": "green orange snack",

"description": "The green and orange snack bag on the right side of the table."

}

<System Prompt of Grasp Outcome Verification>
……

<Planner Response>
The target object is no longer visible on the table and appears to be securely held in the robotic hand.

True

<System Prompt of Prompt Completion Check>
……

<Planner Response>
To determine if the user prompt has been fully completed, let’s identify all the edible objects in the initial image and check if they have been removed or grasped in the current image.

### Initial Image Edible Objects:
1. A bottle (drink).
2. A green packet with a brown label (food).
3. A small packaged food item with a white label (food).

### Current Image Analysis:
- The bottle is still present on the table.
- The green packet with a brown label is no longer visible.
- The small packaged food item with a white label is still present on the table.

### Reasoning:
The user prompt was to grasp all edible objects. However, the bottle and the small packaged food item are still on the table, indicating that not all edible objects have been removed or grasped.

### Conclusion:
Since some relevant objects remain on the table, the user prompt has not been fully completed.

**Output:**
All edible objects have been removed or grasped: False.

<System Prompt of Instruction Proposal>
……

<Planner Response>
The brown snack packet on the right side of the table.

<System Prompt of Bounding Box Prediction>
……

<Planner Response>
[⬇](data:text/plain;base64,ewogICAgImJib3hfMmQiOiBbNDUwLCAxNjgsIDU3OSwgMjYzXSwKICAgICJsYWJlbCI6ICJicm93biBzbmFjayBwYWNrZXQiLAogICAgImRlc2NyaXB0aW9uIjogIlRoZSBicm93biBzbmFjayBwYWNrZXQgb24gdGhlIHJpZ2h0IHNpZGUgb2YgdGhlIHRhYmxlIgp9)
{

"bbox_2d": [450, 168, 579, 263],

"label": "brown snack packet",

"description": "The brown snack packet on the right side of the table"

}

<System Prompt of Grasp Outcome Verification>
……

<Planner Response>
The target object is not visible on the table, and it appears to be securely held in the robotic hand based on the close-up view.

Reasoning: The object is no longer on the table and is being held by the robotic arm.
True

<System Prompt of Prompt Completion Check>
……

<Planner Response>
The user prompt is to grasp all edible objects, including food and drinks. In the initial image, there are several items that could be considered edible:

1. A bottle of liquid (drink).
2. A bag of snacks.
3. A packaged food item.

In the current image, the bottle of liquid remains on the table, while the other two items (the bag of snacks and the packaged food) are no longer visible. This suggests that only some of the edible objects have been grasped or removed.

Reasoning: Since the bottle of liquid is still present on the table, not all edible objects have been grasped.

Output: False

<System Prompt of Instruction Proposal>
……

<Planner Response>
The white bottle with a brown label next to the yellow toy on the right side of the table.

<System Prompt of Bounding Box Prediction>
……

<Planner Response>
[⬇](data:text/plain;base64,ewogICAgImJib3hfMmQiOiBbMzA0LCAxNTAsIDM2OCwgMjI3XSwKICAgICJsYWJlbCI6ICJ3aGl0ZSBib3R0bGUiLAogICAgImRlc2NyaXB0aW9uIjogIlRoZSB3aGl0ZSBib3R0bGUgd2l0aCBhIGJyb3duIGxhYmVsIGlzIGxvY2F0ZWQgbmV4dCB0byB0aGUgeWVsbG93IHRveSBvbiB0aGUgcmlnaHQgc2lkZSBvZiB0aGUgdGFibGUuIgp9)
{

"bbox_2d": [304, 150, 368, 227],

"label": "white bottle",

"description": "The white bottle with a brown label is located next to the yellow toy on the right side of the table."

}

<System Prompt of Grasp Outcome Verification>
……

<Planner Response>
The target object is no longer visible on the table and appears to be securely held in the robotic hand.

True

<System Prompt of Prompt Completion Check>
……

<Planner Response>
The user prompt is to grasp all edible objects, including food and drinks. In the initial image, there are several items that could be considered edible:

1. A bottle of liquid (possibly a drink).
2. A packaged snack with visible text.
3. Another packaged item that appears to be food.

In the current image, these items are no longer visible on the table. This suggests they have been removed or grasped by the robot.

Reasoning: Since all the potential edible objects from the initial image are no longer present in the current image, it can be concluded that the user prompt has been fully completed.

Output: True

## Appendix D Additional Experiments

This section presents an additional experiment that aims to separately evaluate the bounding-box prediction accuracy of the DexGraspVLA planner.

#### Tasks.

The bounding-box prediction accuracy of the planner is crucial to the success of grasping, as it determines the target for the controller. To evaluate this accuracy, we design three types of tasks featuring different environmental distractions: (1) No Distraction* (1 scenario): The cluttered scene is arranged on a white table under white light; (2) *Background Distraction* (2 scenarios): The cluttered scene is placed on either a calibration board or a brightly colored tablecloth, both under white light; (3) *Lighting Distraction* (2 scenarios): The scene is set up in a dark room illuminated by either a desk lamp or a disco light. Scenarios with distractions are shown in [Figure˜11](#A4.F11). For each scenario, we randomly arrange five cluttered scenes, each containing six randomly selected objects, and then record head-camera images. For each object, we provide a textual prompt describing its appearance and location, and check whether the planner’s bounding-box prediction accurately marks the target. In total, *No Distraction* accounts for 30 tests, while *Background Distraction* and *Lighting Distraction* both have 60 tests, amounting to 150 tests overall.

#### Metric.

We define a bounding box as accurate if it tightly encloses the target object. Accuracy is then measured as the proportion of accurate bounding boxes over all tested objects.

![Figure](figures/bbox-1.png)

![Figure](figures/bbox-2.png)

![Figure](figures/bbox-3.png)

![Figure](figures/bbox-4.png)

Figure 11: Bounding-box predictions made by DexGraspVLA planner. Across diverse lighting and background conditions, it accurately grounds the language instruction to the target object in cluttered scenes and marks the correct bounding box.

#### Results.

The accuracy is reported in [Table˜5](#A4.T5). For 150 prompts, the planner only mislabels one bounding box while succeeding in the other 149 tests, resulting in an aggregated accuracy exceeding 99%. In [Figure˜11](#A4.F11), we present examples of bounding-box predictions produced by the DexGraspVLA planner. Despite substantial variation in environmental conditions, the planner consistently grounds grasping instructions in cluttered scenes and provides the correct bounding boxes. Notably, we can identify objects by names such as “Coca Cola” or “milk,” reflecting the system’s extensive common sense and world knowledge. By drawing on the broad knowledge embedded in each of its foundation models, DexGraspVLA achieves robust generalization across diverse scenarios.

*Table 5: Planner accuracy in bounding-box prediction under different environment conditions.*

Generated on Sat Nov 15 15:38:42 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)