[# Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence Ling-Hao Chen 0000-0002-2528-6178](https://orcid.org/0000-0002-2528-6178)
[evan@lhchen.top](mailto:evan@lhchen.top)

Tsinghua University, International Digital Economy AcademyChina

,
Yuhong Zhang

[0000-0001-6180-4457](https://orcid.org/0000-0001-6180-4457)
Tsinghua UniversityChina

,
Zixin Yin

[0000-0003-0443-7915](https://orcid.org/0000-0003-0443-7915)
The Hong Kong University of Science and TechnologyHong KongChina

,
Zhiyang Dou

[0000-0003-0186-8269](https://orcid.org/0000-0003-0186-8269)
The University of Hong KongHong KongChina

,
Xin Chen

[0000-0002-9347-1367](https://orcid.org/0000-0002-9347-1367)
ByteDanceSan JoseUnited States of America

,
Jingbo Wang

[0009-0005-0740-8548](https://orcid.org/0009-0005-0740-8548)
Shanghai Artificial Intelligence LaboratoryShanghaiChina

,
Taku Komura

[0000-0002-2729-5860](https://orcid.org/0000-0002-2729-5860)
The University of Hong KongHong KongChina

and
Lei Zhang

[0000-0001-6926-0538](https://orcid.org/0000-0001-6926-0538)
International Digital Economy AcademyChina

###### Abstract.

This work studies the challenge of transfer animations between characters whose skeletal topologies differ substantially. While many techniques have advanced retargeting techniques in decades, transfer motions across diverse topologies remains less-explored. The primary obstacle lies in the inherent topological inconsistency between source and target skeletons, which restricts the establishment of straightforward one-to-one bone correspondences. Besides, the current lack of large-scale paired motion datasets spanning different topological structures severely constrains the development of data-driven approaches. To address these limitations, we introduce Motion2Motion, a novel, training-free framework. Simply yet effectively, Motion2Motion works with only one or a few example motions on the target skeleton, by accessing a sparse set of bone correspondences between the source and target skeletons. Through comprehensive qualitative and quantitative evaluations, we demonstrate that Motion2Motion achieves efficient and reliable performance in both similar-skeleton and cross-species skeleton transfer scenarios. The practical utility of our approach is further evidenced by its successful integration in downstream applications and user interfaces, highlighting its potential for industrial applications. Code and data are available at [https://lhchen.top/Motion2Motion](https://lhchen.top/Motion2Motion).

Motion transfer, Motion synthesis

††copyright: none††journal: TOG††journalyear: 2025††ccs: Computing methodologies Animation††ccs: Computing methodologies Computer vision

*Figure 1. We propose Motion2Motion, enabling motion transfer across characters with vastly different topologies. From left to right, we show motion transfer results across increasingly different target characters, anaconda $\to$ king-kobra (in-species transfer) $\to$ T-rex (cross-species transfer).*

## 1. Introduction

Transferring a motion from one character to another with different topology is a long-standing research problem in computer animation. Motion transfer occupies a pivotal position in character animations, especially when mapping motion to unseen characters. To tackle this fundamental problem, a series of methods have been proposed [(Aberman et al., [2020](#bib.bib2); Zhang et al., [2023b](#bib.bib51))] in recent years. One category is skeleton-based motion transfer for various shapes of skeletons, the other takes the geometry of two characters into consideration for more fine-grained motion transfer.

Despite these progresses, previous methods face significant challenges in real applications, especially transfer motions between two characters with different topologies. Regardless of whether skeleton-based or geometric-based methods, most of these methods [(Ye et al., [2024](#bib.bib48); Zhang et al., [2023b](#bib.bib51))] rely on a quite amount of data. However, due to the limited accessibility of motion datasets with different characters, these methods often fail at test time. Another trickier issue is that the skeleton topologies and geometries of the target characters are often more complex than the training data. For instance, methods trained on humanoid-like data, such as Mixamo [(Adobe, [2024](#bib.bib3))], struggle to retarget motions to characters with complex dynamics like skirts or hair, let alone transfer from bipedal to quadrupeds. Although recent attempts [(Li et al., [2024](#bib.bib28))] tried to retarget animation between humans and quadrupeds, they still need tailored training to accommodate diverse morphologies, making generalization to more complex characters challenging.

There are two key challenges to transfer animations from one topology to another. The first challenge is the limited availability of animation data across diverse topologies. In practical applications, only a few motion examples exist for the target skeleton, forcing data-driven methods to rely on large-scale datasets. Unfortunately, even the most readily accessible human motion data, such as SMPL-based motion [(Loper et al., [2023](#bib.bib33))], suffers from an insufficient volume of high-quality examples. The second challenge involves binding correspondences for all bones between source and target skeletons, especially when the target one has a significantly different topology or number of joints. This makes it difficult to define consistent joint-level mappings, which are critical for motion transfer.

To make this problem tractable, we introduce two mild assumptions that align with real-world constraints. First, we assume access to a small number of motion sequences on the target skeleton, a few-shot setting that reflects the scarcity of data in most practical scenarios. If there is no motion example at all, transfer becomes ill-posed, as there is no signal to anchor temporal or spatial motion patterns on target skeletons. This minimal data availability provides satisfying contexts to guide meaningful transfer while avoiding the need for large-scale annotation. Second, we assume that a sparse joint correspondence between source and target skeletons is available, which can be specified by users or automatic matching ([Appendix A](#A1)). This minimal bone mapping offers rough semantic alignment and serves as a useful prior for motion-level matching, a step that is commonly required in both academic and commercial systems (e.g*. , AutoRig [(Artell, [2025](#bib.bib5))], and Rokoko [(Rokoko, [2021](#bib.bib38))]).

Considering these issues, we introduce Motion2Motion, a novel approach for animation transfer under sparse joint correspondences (sparse bone binding). Specifically, Motion2Motion models the cross-topology motion transfer problem as a conditional patch-based motion matching problem. The bone correspondences specified by users provide coarse motion semantics, serving as the input spatial conditions for transfer.
Departing from traditional neural transfer pipelines, Motion2Motion synthesizes motions for the target skeleton by matching motion patches of the bound joints from a few example animations. This design is motivated by a straightforward observation: motion coordination between bound and unbound joints can often be inferred by observing only a few examples. For instance, in quadruped locomotion, the movement of the hind legs can often be extrapolated from the motion patterns of the front legs alone by some examples, even in one/few-shot settings. This simple yet effective matching-based transfer allows Motion2Motion to outperform current state-of-the-art methods.

A key application of Motion2Motion is its capacity to retarget a SMPL-based motion [(Loper et al., [2023](#bib.bib33))] to those more complex topologies, such as characters featuring dynamic elements like skirts or hair. In addition to this capability, Motion2Motion further enables seamless motion transfer across structurally diverse skeletons, even between different species, e.g. anaconda v.s. raptor (in [Fig. 8](#S4.F8)). Our framework is also flexible in its choice of motion representation. Specifically, it supports a variety of features for transfer, including velocity fields, and the local pose positions, demonstrating its broad generalizability and robustness.

This work paves the way for topology-flexible motion transfer, enabling motion adaptation across different character structures in real time. To support practical use, we also develop a Blender add-on ([Section 6](#S6)), demonstrating the applicability in real-world animation workflows with good interpretability.

## 2. Related Work

Motion Retargeting.
Motion transfer was first introduced by [Gleicher ([1998](#bib.bib19))], who approached the problem as a space-time optimization challenge subject to kinematic constraints. This seminal work paved the way for various optimization-based methods that incorporated additional constraints [(Bernardin et al., [2017](#bib.bib9); Feng et al., [2012](#bib.bib17); Lee and Shin, [1999](#bib.bib26))]. More recently, advances in deep learning have enabled more flexible, data-driven solutions for motion retargeting. For instance, [Jang et al. ([2018](#bib.bib23))] proposed an autoencoder-based method for motion generation, [Aberman et al. ([2020](#bib.bib2))] introduced a Skeleton-Aware Network to address differences in skeletal topology, and [Lim et al. ([2019](#bib.bib32))] developed a framework that decouples the learning of local poses from global motion. [Li et al. ([2022b](#bib.bib30))] presented an iterative retargeting approach that uses motion autoencoders to refine results over multiple steps. Some methods have incorporated character geometry into motion retargeting by preserving contacts or avoiding interpenetration [(Lyard and Magnenat-Thalmann, [2008](#bib.bib35); Ho et al., [2010](#bib.bib21); Jin et al., [2018](#bib.bib25); Zhang et al., [2023b](#bib.bib51), [2024b](#bib.bib50))].
Unlike previous methods that rely on dense correspondences or large-scale data, our approach works under sparse joint mappings and limited target motions. We formulate motion transfer as a matching-and-blending process over motion patches, enabling flexible transfer across structurally different skeletons. This design allows generalization to complex topologies with minimal samples.

Generative motion models and motion matching.
Human motion generation [(Bhattacharya et al., [2021](#bib.bib10); Petrovich et al., [2022](#bib.bib37); Zhang et al., [2024a](#bib.bib52); Athanasiou et al., [2022](#bib.bib6); Tevet et al., [2022](#bib.bib43); Wang et al., [2022](#bib.bib46); Dabral et al., [2023](#bib.bib15); Yuan et al., [2023](#bib.bib49); Shafir et al., [2024](#bib.bib39); Zhang et al., [2023a](#bib.bib53); Xie et al., [2024](#bib.bib47); Lu et al., [2024](#bib.bib34); Barquero et al., [2024](#bib.bib7))] reaches significant progress in past years, benefiting from the foundational human-centric vision technologies [(Loper et al., [2023](#bib.bib33); Mahmood et al., [2019](#bib.bib36))]. Researchers have extensively explored the generation of human motion using various modalities—including audio [(Bhattacharya et al., [2021](#bib.bib10); Chen et al., [2024](#bib.bib12))], music [(Tseng et al., [2023](#bib.bib45); Li et al., [2023b](#bib.bib29))], and text [(Dai et al., [2024](#bib.bib16); Chen et al., [2023a](#bib.bib14); Tevet et al., [2022](#bib.bib43); Zhou et al., [2024](#bib.bib56))]—as well as through unconditional approaches [(Li et al., [2022a](#bib.bib27); Chen et al., [2023b](#bib.bib13))]. In addition to neural synthesis methods, techniques based on generative motion matching have also shown strong performance on synthesizing high-quality motions [(Li et al., [2023a](#bib.bib31); Büttner and Clavet, [2015](#bib.bib11); Holden et al., [2020](#bib.bib22); Bergamin et al., [2019](#bib.bib8))]. Motion matching [(Agata and Igarashi, [2025](#bib.bib4))] itself is a high-level concept in animation creation, primarily employed for character control. However, the application of motion matching and blending to motion transfer, especially across different skeletal topologies, remains underexplored from both research and engineering perspectives. The latest progress in generating motions across diverse topologies is represented by AnyTop [(Gat et al., [2025](#bib.bib18))]. However, AnyTop’s architecture does not include a motion transfer module, and its generalization capability is restricted by the relatively limited scale of its training data. In contrast, Motion2Motion takes an early step by transfer a source motion to a target skeleton using one or a few sample target motions, marking the first attempt to harness motion matching and blending in cross-topology transfer.

## 3. Method

### 3.1. Preliminaries

Motion representation. Let $\mathbf{M}\in\mathbb{R}^{F\times D}$ denote the overall motion representation, where $F$ is the number of frames and $D$ is the dimensionality of the representation. The source motion is represented by $\mathbf{S}\in\mathbb{R}^{F_{s}\times D_{s}}$. The target motion set is defined as, $\mathcal{T}=\{\mathbf{T}^{(i)}\in\mathbb{R}^{F_{i}\times D_{t}}\mid i=1,2,\dots,N\}$,
where $N$ is the number of target motion sequences and $D_{t}$ is the dimensionality of each target motion.
Generally, for the frame $f$ of motion $\mathbf{M}\in\mathbb{R}^{F\times D}$, the pose $\mathbf{M}_{f}\in\mathbb{R}^{D}$ are represented as the 6D joint rotations and the root velocity. Specifically, for a character with $J$ joints, $D\leftarrow 3+6J$.

Rest pose pre-alignment. As the definitions of rest poses in different files are not always the same, we align the protocol of rest poses of source and target motions before transfer. The process can be directly borrowed from existing animation tools [(Artell, [2025](#bib.bib5))].

### 3.2. Motion Patching

As illustrated in [Fig. 2](#S3.F2)-(A) and (B), prior to transfer, we perform motion patching on the source motion along the temporal axis without padding. In contrast to the patching operation in the image domain, we apply a sliding window to patchify the motion, ensuring the preservation of meaningful motion features. This process is grounded in the observation that motion patches capture significant temporal dynamics, and the patch size ($P_{S}$) is relatively atomic. After the motion patching operation, we obtain $P$ patches of the motion, where $P=\left\lfloor\frac{F_{s}-\text{patch size}+1}{\text{step size}}\right\rfloor$. Similarly, we apply the same operation to the target motion set using the same patch and step size, resulting in target motion patches in [Fig. 2](#S3.F2)-(C). The source and target motion patches are denoted as set $\mathcal{P}^{(s)}$ and set $\mathcal{P}^{(t)}$, respectively.

### 3.3. Spatial Correspondence between Skeletons

*Figure 2. System overview of Motion2Motion. (A) The source motion sequence $\mathbf{S}\in\mathbb{R}^{F_{s}\times D_{s}}$. (B) The source sequence is divided into overlapping motion patches $\mathcal{P}^{(s)}$. (C) Each source patch is projected to the target skeleton space via sparse mapping and noise initialization, serving as the query for retrieval. For each source patch, we retrieve target patches (D) from a pre-built motion patch database $\mathcal{P}^{(t)}$, based on sparse correspondences. (E) The matched target patches are averaged for blending. (F) The retargeted motion $\widehat{\mathbf{T}}\in\mathbb{R}^{F_{s}\times D_{t}}$ is reconstructed from the blended target patches. (C)-(F) are executed in $L$ times.*

Let $\mathcal{M}=\{(t,s)\mid t\in\mathcal{J}_{t},\;s\in\mathcal{J}_{s}\}$ denote the set of sparse keypoint correspondences between the target and source skeletons, where $\mathcal{J}_{t}$ and $\mathcal{J}_{s}$ represent the index sets of keypoints in the target and source skeletons, respectively. We denote the number of correspondences by $K=|\mathcal{M}|$.
A correspondence $(t,s)\in\mathcal{M}$ indicates that the keypoint $t$ in the target skeleton is semantically aligned with the keypoint $s$ in the source skeleton. The correspondence of bone binding can be specified by the user or be bound in an automatic way via fuzzy subgraph matching (algorithm detail in [Appendix A](#A1)).

To specify the indexes of each keypoint within the motion representation vectors, we introduce two index functions $\mathcal{I}_{1}(\cdot),\ \mathcal{I}_{2}(\cdot):\mathbb{N}\to\mathbb{N}$,
such that for any keypoint $t$ (or $s$), $\mathcal{I}_{1}(t)$ and $\mathcal{I}_{2}(s)$ denote the start and end indices of the channels in the corresponding bones (e.g., $\mathbf{T}\in\mathbb{R}^{F\times D_{t}}$ for the target or $\mathbf{S}\in\mathbb{R}^{F\times D_{s}}$ for the source).

Based on these definitions, we construct a correspondence matrix $\mathbf{C}\in\mathbb{R}^{D_{t}\times D_{s}}$ whose block components are defined as follows:

| (1) | | $$\mathbf{C}\Big{[}\mathcal{I}_{1}(t):\mathcal{I}_{2}(t),\;\mathcal{I}_{1}(s):\mathcal{I}_{2}(s)\Big{]}=\begin{cases}\ \mathbf{I}\ ,&\text{if }(t,s)\in\mathcal{M},\\ \mathbf{O}\ ,&\text{otherwise},\end{cases}$$ | |
|---|---|---|---|

where $\mathbf{I}$ denotes an identity matrix of appropriate dimensions and $\mathbf{O}$ denotes a zero matrix of corresponding size.
This formulation ensures that, in motion matching, only the segments of the motion representations corresponding to the predefined correspondences are aligned, thereby effectively capturing the core kinematic characteristics of the motion while discarding redundant information.

We further introduce a mask vector to identify target skeletal components that lack any correspondence in $\mathcal{M}$. Specifically, the mask is computed as $\mathbf{m}[i]=\sum_{j=1}^{D_{s}}\mathbf{C}[i,j]$, which effectively indicates, for each target dimension $i$, whether a valid mapping from the source exists. Note that the vector $\mathbf{m}$ can be sparse, that is a small number of joint correspondences.

### 3.4. Iterative Matching-based Transfer

With the sparse correspondence established above, we aim to transfer motion information between structurally different skeletons. Motivated by motion matching, we formulate the transfer process as a retrieval-and-blending procedure over motion patches. This allows us to explore the application of motion matching under cross-topology settings with sparse features, where the source and target skeletons differ significantly in structure and representation.

Motion projection and initialization. Using this mask, the mapping from the source motion to the target skeleton is defined as

| (2) | | $$\mathbf{P}^{s\to t}=\mathbf{S}\mathbf{C}^{\top}+\bigl{(}\mathbf{1}-\mathbf{m}\bigr{)}\odot\mathbf{N},$$ | |
|---|---|---|---|

where $\mathbf{N}\sim\mathcal{N}(\mathbf{O},\mathbf{I})\in\mathbb{R}^{F\times D_{t}}$ denotes a noise term drawn from a standard normal distribution. The first term, $\mathbf{S}\mathbf{C}^{\top}$, represents the motion mapping from the source skeleton to the target skeleton based on predefined keypoint correspondences. The second term, $\bigl{(}\mathbf{1}-\mathbf{m}\bigr{)}\odot\mathbf{N}$, is initialized by a noise for target dimensions without a corresponding source mapping. The transfer process can be viewed as the transformation of these noisy components into the desired locomotion, conditioned on the first term. After the procedure, the noisy parts will be replaced by the retargeted locomotion. This formulation ensures that for target dimensions without a corresponding source mapping, the motion is augmented by controlled stochastic variation. For convenience, we rename $\mathbf{P}^{s\to t}$ as $\widehat{\mathbf{T}}$ in following sections.

Masked motion matching. After obtaining the projected motion $\widehat{\mathbf{T}}$ in [Eq. 2](#S3.E2), we formulate the synthesis process of the noisy part as a generative process conditioning on $\mathbf{S}\mathbf{C}^{\top}$. We also patchify the the projected motion $\widehat{\mathbf{T}}$ as $\mathcal{P}^{(\widehat{t})}$. Motivated by generative motion matching [(Li et al., [2023a](#bib.bib31); Büttner and Clavet, [2015](#bib.bib11))], the generation process of target motion patches in Motion2Motion is masked motion patch retrieval. Specifically, for each $\mathbf{P}^{(\widehat{t})}\in\mathcal{P}^{(\widehat{t})}$,

| (3) | | $\displaystyle\mathbf{P}^{match}\leftarrow\mathop{\arg\min}\limits_{\mathbf{P}\in\mathcal{P}^{(t)}}$ | $\displaystyle\;\;\alpha\mathcal{L}\left(\mathbf{m}\odot\mathbf{P},\mathbf{m}\odot\mathbf{P}^{(\widehat{t})}\right)$ | |
|---|---|---|---|---|
| | | $\displaystyle+(1-\alpha)\mathcal{L}\left((\mathbf{1}-\mathbf{m})\odot\mathbf{P},(\mathbf{1}-\mathbf{m})\odot\mathbf{P}^{(\widehat{t})}\right),$ | |

where $\mathcal{L}(\cdot,\cdot)$ denotes the mean squared error (MSE) loss, and $\alpha\in[0,1]$ balances the contributions from mapped and unmapped parts in $\widehat{\mathbf{T}}$. A larger $\alpha$ in [Eq. 3](#S3.E3) places more emphasis on the sparse correspondences, enforcing stronger semantic alignment between source and target motions. In contrast, the second term leverages noise in unmapped dimensions, introducing diversity and flexibility into the generated motion.

*Algorithm 1 Iterative Matching-based Motion Retargeting*

Motion blending. For each source motion patch $\mathbf{P}^{(s)}\in\mathcal{P}^{(s)}$, we match a similar target patch $\mathbf{P}^{{}^{match}}\in\mathcal{P}^{(t)}$ according to [Eq. 3](#S3.E3), resulting a matching set $\mathcal{M}$. We blend all retrieved target motion patches of $\mathcal{M}$ in average, as illustrated in [Fig. 2](#S3.F2)-(E).

However, directly blending the retrieved patches may compromise motion naturalness or distort key actions, especially in the regions corresponding to the sparse correspondences. To address this, we repeat the matching-and-blending process $L$ times, progressively refining the result to ensure temporal coherence across the entire sequence. The whole algorithm is in [Algorithm 1](#algorithm1).

*Table 1. Main evaluation results for motion transfer. Our method (Motion2Motion), in both similar skeleton and cross-species settings, achieves the best performance on synthesis quality, temporal coherence, and diversity. Different from baselines, our method runs without GPUs and deep model training.*

## 4. Experiments

### 4.1. Implementation Details

Data. In this section, we collect a set of animal motions and human motions to present motion transfer across skeletons. The animal animation dataset is the Truebones-Zoo dataset [(Truebones, [2022](#bib.bib44))]. The number of joints ranges from 9 to 143. The human animation data is borrowed from LAFAN [(Harvey et al., [2020](#bib.bib20))]. For qualitative experiments, we choose a subset of Truebones-Zoo as the benchmark for comparison.

Implementation. The patch size of the motion is set as 11 frames, $\alpha$ is set as 0.85 by default. We follow [Li et al. ([2022a](#bib.bib27))], generating motion in an inverted pyramidal way. All of the baselines mentioned in the paper is run on $1\times$ NVIDIA RTX-3090-24GB GPU, while our method is run on $1\times$ MacBook-M1 chip without a GPU.

*Figure 3. Quantitative comparison with baselines. Each animation sequence is listed as frames from left to right. (Highlighted gray frames as comparison for baselines.) (A) Input source motion of a dragon character. (B–D) Retargeted sequences on the bat skeleton produced by (B) our method, (C) [Li et al. ([2024](#bib.bib28))], and (D) [Zhao et al. ([2024](#bib.bib55))]. Compared to baselines, our approach more faithfully preserves the original motion style, coherence, and frequency.*

### 4.2. Evaluation Protocol

Metrics. We evaluate the transfer result from the following aspects. (1) Motion quality is evaluated via the Fréchet Inception Distance (FID) between retargeted and real target skeleton motions. The FID metric measures the motion quality and the style consistency with the target motion distribution. We use the kinematic features [(Siyao et al., [2022](#bib.bib41))] as extracted features when calculating FID.
(2) The temporal alignment between the source and retargeted motion is evaluated by the cosine similarity of the Power Spectral Density (PSD) across all joints. We also follow [Zhao et al. ([2024](#bib.bib55))] using contact consistency as the metric to evaluate the temporal coherence, where the contact bones are labeled manually by researchers.
(3) Transfer diversity is the average joint distance between each two samples among the 5 retargeted results.
(4) We use inference FPS to evaluate efficiency. In remaining text, we calculate the binding rate as $\frac{2|\mathcal{M}|}{J_{S}+J_{T}}\times 100\%$, where $J_{S}$ and $J_{T}$ are bone numbers of source and target skeletons.

Baselines. We introduce two latest baselines for comparison. Pose-to-Motion [(Zhao et al., [2024](#bib.bib55))] transfers motion from a motion-rich source to a target with only pose data, using a learned pose prior to generate temporally coherent motions. It performs well with sparse or noisy poses and in cross-skeleton settings. In our comparison, we split the motion data into a poses set in ahead. WalkTheDog [(Li et al., [2024](#bib.bib28))] aligns motions across different morphologies by a shared phase manifold. It enables semantic and temporal alignment between structurally different characters (e.g., human to quadruped) and supports tasks like motion transfer and stylization.

Benchmark. Inspired by previous in-topological motion transfer evaluation, we collect 14 character animations to evaluate the cross-topology motion transfer results. The evaluation benchmark comes up with 1,167 frames in total, covering running, walking, jumping, and attacking actions from Truebones-Zoo [(Truebones, [2022](#bib.bib44))] and LAFAN [(Harvey et al., [2020](#bib.bib20))]. To thoroughly evaluate the algorithm bounds, we categorize the skeleton gap between the source and target characters as similar and cross-species transfer. In detail, we categorize the source-target characters into similar skeletons (e.g., biped-to-biped or crawling-to-crawling) and cross-species skeletons (e.g., biped-to-quadruped or crawling-to-biped).

### 4.3. Comparison with Baselines

Qualitative results.
As shown in [Fig. 3](#S4.F3), we present a visual comparison of motion transfer results across different methods. The top row (A) illustrates the source motion performed by a dragon (143 joints), showcasing large-amplitude wing flaps and dynamic movement patterns. The following rows show the motion retargeted onto a bat (48 joints) using (B) our method (binding 2 pairs, left and right UpperArm), (C) Walk-The-Dog [(Li et al., [2024](#bib.bib28))], and (D) Pose-to-Motion [(Zhao et al., [2024](#bib.bib55))].
Our method (B) clearly follows the full range of the source motion, preserving both temporal structure and stylistic characteristics such as up and down wing spread events. In contrast, motions produced by baselines suffer from artifacts and temporal inconsistency. Furthermore, our method shows consistent spatial-temporal transitions across frames, demonstrating superior coherence and fidelity in cross-topology transfer.

Quantitative results. [Table 1](#S3.T1) reports the quantitative comparison between our method and two recent baselines, Walk-The-Dog and Pose-to-Motion. Across both similar skeleton and cross-species settings, our method consistently achieves the best performance in all quality and diversity metrics. Specifically, we obtain the lowest FID, indicating high motion quality and style consistency. Our method also achieves the highest frequency alignment and contact consistency, demonstrating strong temporal coherence with the source motion. In terms of diversity, we outperform baselines by a large margin, suggesting the ability to generate varied retargeted motions. Despite being training-free, our method maintains a high inference FPS, comparable to Walk-The-Dog and significantly faster than Pose-to-Motion. Notably, two baselines are tested on one GPU, while ours is without a GPU. Moreover, our method is a model-free algorithm, without any model training.

Analysis. The reason why two baselines work worse than the proposed method is that the model-based methods are highly data-driven, requiring a large amount of data. This not only results in some artifacts on some OOD scenarios, but also easily leads to over-fitting issues and makes the result less diverse (see [Table 1](#S3.T1)). Moreover, baselines are highly reliant on the trained data distribution. If the desired motion frequency is unobserved in training, it is hard to retarget motion to a novel frequency provided by the source motion (see freq. align & contact con. in [Table 1](#S3.T1)). However, our matching-based method can flexibly compose the motion patches by motion patch blending, simulating as a frequency interpolator.

### 4.4. Temporal Matching Visualization

*Figure 4. Motion matching visualization. A) Source motion on a bear skeleton. (B) Target dog skeleton used for matching. (C) The retargeted motion result on the dog skeleton. The 2nd and 4th frames in (C) correspond to the 4th and 6th frames in (B), respectively, illustrating temporally non-linear yet semantically aligned motion matching.*

Matching patches visualization.
[Fig. 4](#S4.F4) provides a qualitative visualization of our motion transfer process. The source and target skeletons correspond to a bear (76 joints) and a dog (55 joints), respectively. As shown on the left side of panels (A) and (B), the correspondences between the two skeletons are limited to 6 hind leg joints. Notably, the second and fourth frames in panel (C) correspond to the fourth and sixth frames in panel (B), demonstrating effective temporal alignment. Moreover, the motion of the dog’s front legs is conditionally inferred from its hind leg motion, and the dynamic motion of the dog’s long tail is not directly provided by the source. This flexible matching is achieved by leveraging sparse keypoint bindings and masked patch comparisons, enabling robust motion transfer across diverse topology skeletons.

![Figure](x3.png)

*Figure 5. Motion temporal coherence visualization. (A) Source running motion from a Raptor character (36 joints). (B) Retargeted motion to a human character (22 joints). (C) Retargeted motion to a fox character (40 joints). The retargeted motions share the same temporal coherence and movement periodicity. Binding bones are in purple (right side).*

![Figure](x4.png)

*Figure 6. Phase visualization of the motion. (A) and (B) present the phases of RightToe and LeftToe. The bar figure is the phase variation curve, and the clock figure is the phase visualization at the 1-st and 10-th frames. The blue and orange colors denoted retargeted and source motion, respectively. Note that there is a consistent phase bias between the source and target.*

Phase manifold coherence.
In addition to maintaining temporal coherence, our method ensures that the retargeted motions lie on a smooth and consistent phase manifold. To present the robustness of the algorithm to human characters (from LAFAN dataset [(Harvey et al., [2020](#bib.bib20))]) outside the animal motion dataset. We randomly clip 5 motion sequences from the LAFAN dataset, each of which is no more than 80 frames (data with 60 FPS).
Previous research shows the periodicity of the motion, especially in the phase manifold. To verify the motion coherence of the source motion and the retargeted one, we visualize the periodic phase of the joint positions. In [Fig. 6](#S4.F6), we visualize the phase of the dominant frequency component (a.k.a. the frequency component with the largest amplitude in the FFT). The constant phase bias shown in [Fig. 6](#S4.F6) means the coherent motion frequency.
As visualized in [Fig. 5](#S4.F5), the phase progression in the retargeted motions (B and C) follows the same periodicity as the source Raptor motion (A), despite substantial differences in skeleton topology and motion dynamics. This coherence across species and motion styles highlights the robustness of our framework in preserving underlying phase structures during motion transfer.

### 4.5. Cross- Skeleton and Species Transfer

![Figure](x5.png)

*Figure 7. Cross skeletal organisms transfer (biped $\to$ quadruped). A walking motion is transferred from a flamingo (41 joints) to a monkey (76 joints) by binding only 6 corresponding hind limb bones. The hind limb motions are kept synchronous, while the monkey’s forelimb and tail movements are inferred based on the retargeted hind limb dynamics.*

Cross-skeletal motion transfer between species with significantly different anatomical structures is a challenging problem. Our approach resolve this issue with very sparse correspondences.

Biped $\to$ quadruped.
As illustrated in [Fig. 7](#S4.F7), we transfer a walking motion from a flamingo, a bipedal species with 41 joints, to a quadrupedal monkey skeleton comprising 76 joints. In this case, we establish sparse keypoint correspondences by binding only 6 hind limb bones between the source and target skeletons. These sparse bindings serve as anchors for transfer motion cues. In our framework, the synchronous motion of the hind limbs is maintained between the two skeletons; that is, the source motion directly drives the retargeted hind limb dynamics in the monkey. Meanwhile, the movements of the monkey’s forelimbs, head, fingers, and tail are inferred from these sparse hind limb dynamics, ensuring a natural and coherent overall gait despite the structural disparity.

![Figure](x6.png)

*Figure 8. Cross skeletal organisms transfer (limbless $\to$ biped). The diverse retargeted results of the same skeleton are shown in different colors. (A) Source motion of an anaconda. (B) Retargeted motion on the limbless skeleton (king-cobra). (C) Retargeted motion on the biped skeleton (raptor). The frames in (B) and (C) correspond to specific moments from (A), demonstrating the transformation from a limbless to a bipedal structure with semantically and temporally aligned motion.*

Limbless $\to$ biped.
We first retarget an attack motion from anaconda (27 joints) to the kingcobra (19 joints) in [Fig. 8](#S4.F8)-(A)/(B), where the retargeted kingcobra motion keeps its original huddling style. As shown in [Fig. 8](#S4.F8)-(A)/(C), we retarget a motion from a limbless species, an anaconda, to a bipedal raptor skeleton with 36 joints. In this case, we establish sparse keypoint correspondences by binding only 4 key vertebral points between source and target skeletons. These sparse bindings are crucial in guiding the transfer process, ensuring motion alignment across skeletons. Moreover, as discussed in [Section 3.4](#S3.SS4), the transfer diversity is controlled by the noise term. As shown in [Fig. 8](#S4.F8)-(B)/(C), different colors for retargeted skeletons represent diverse results driven by the same source motion.

Diversity of retargeted motions. In [Eq. 3](#S3.E3), $\alpha\in[0,1]$ controls the weights of joint correspondence matching and random noise matching. The larger $\alpha$ means more accurate local motion matching on specified joints, that is, less diversity. [Fig. 8](#S4.F8) shows diverse results of the cross-species motion transfer. As can be seen in [Table 1](#S3.T1), our method enjoys the best diversity over baselines, qualitatively verifying the superiority of our algorithm.

*Table 2. “Test time scaling” property. The comparison with different number of target samples.*

### 4.6. “Test-time Scaling” Property

[Table 2](#S4.T2) reports the performance of our method when varying the number of target samples in the dataset during inference.
We observe that increasing the number of samples improves the overall performance across all metrics.
Specifically, generating 3 samples significantly reduces the FID score and enhances both frequency alignment and contact consistency, indicating better quality and realism.
Furthermore, the diversity score increases notably, showing that more samples lead to more diverse motion generations.
Although the larger number of target examples benefits the transfer quality, the more #frames of the target skeletons help the overall quality.
This demonstrates that our method benefits from test-time (inference) sampling without training, showcasing strong scalability.

### 4.7. User Study

We evaluate the motion transfer result across the following aspects through a user study. Users are required to evaluate the transfer result through (1) the quality of retargeted motion. (2) action alignment with the source motion. In this study, 50 users rate 10 source retargeted motion pairs with score 1-5. Baselines are introduced in [Section 4.2](#S4.SS2). As shown in [Table 3](#S4.T3), our method leads baseline with a significant margin on both quality and action alignment.

![Figure](x7.png)

*Figure 9. Key-frame Cross-topology Retargeting. The frame number index is denoted as “¡X¿”. (A) The source motion of a bat. The purple frames are given key frames. The transparent blue frames are ground-truth frames of the source motion. (B) Retargeted motion of a dragon character.*

*Table 3. User study results. Our method leads baselines over a significant margin on both motion quality and alignment.*

*Table 4. Flexible transfer features. Our method support diverse motion matching features.*

### 4.8. Flexible Transfer Features

One of the key component of our method is the motion feature matching, whose matching feature is quite flexible. As shown in Alg. [1](#algorithm1), the defined motion feature of 6D rotation is the default setting. However, the 3D loco-position or joint velocity features can also be used as matching features. As shown in [Table 4](#S4.T4), our method enjoys good performance with diverse matching features. Especially, the velocity feature even works better than 6D rotations and with less diversity. This is mainly because velocity is a straightforward indicator in animation, capturing motion details sensitively, which also makes the matching process more deterministic.

### 4.9. Key-frame Cross-topology Transfer

We found that our method can also be used in key-frame interpolation when given only some animation frames from the source skeletons. In this case, the invisible frames are initialized with the noise. Although these frames are matched blindly initially, the completion of the whole sequence can be reached in $L$ turns, according to [Algorithm 1](#algorithm1). As shown in [Fig. 9](#S4.F9), our method robustly works even when the source animation is very sparse in the temporal axis, not only in the spatial axis. Specifically, the retargeted unseen frames are well aligned with the source motion. This capability demonstrates that our method remains effective even when the source motion is corrupted or poorly crafted.

### 4.10. Ablation Study

How can $\alpha$, patch size $Ps$, and $L$ affect the quality and diversity?
We conducted ablations over the patch‐matching hyperparameters to evaluate their impact on reconstruction quality and output diversity. Here, we do not distinguish two types of skeletal differences. The blending weight $\alpha$ controls the strength of adherence to the bound joints’ motion. When $\alpha$ is set too small (e.g. 0.3), retargeted motion becomes erratic and contact consistency drops. When set too large (e.g. 0.95), motion becomes overly “locked” to the source, reducing diversity significantly.
Varying the patch size $P_{S}$ shows that increasing $P_{S}$ results in comparable quality, and less diversity. This indicates that a too large horizon may include more complex motion patterns for matching. The decrease in $P_{S}$ compromises the consistency and quality, because the short temporal window size cannot capture the semantically fruitful motion feature.
Increasing $L$ (from 1 to 5 iterations) improves temporal consistency and motion and motion quality. However, excessively large $L$ offers minimal further improvement and incurs greater computational cost. We therefore adopt $L=3$ as a reliable default across both similar‐skeleton and cross‐species settings.

![Figure](x8.png)

*Figure 10. Key frames of the source captured SMPL-based motion (A) and the retargeted motion (B) on a character.*

*Table 5. Abalation study of different settings. The default setting of the Motion2Motion is $L=3$, $P_{S}=11$, and $\alpha=0.85$.*

*Table 6. Ablation study of binding mechanism and transfer strategies. We compare automatic and manual bone binding, as well as direct copying bound motion features from the source after executing [Algorithm 1](#algorithm1).*

Bone binding automatically v.s. manually?
[Table 6](#S4.T6) compares our default (manual) binding setting with the automatic mode using fuzzy graph matching (algorithm in [Appendix A](#A1)). For similar skeletons, automatic binding incurs a minor quality drop, and contact consistency lightly falls. This shows flexible binding choices. For cross‐species transfers, the gap widens, denoting the value of expert‐verified correspondences when topologies differ greatly. Nonetheless, the automatic mode still produces reasonable results without user effort, making it a practical solution.

Can binding bones be directly copied with source motion features?
We also evaluated a naïve strategy that directly copies source binding joint features to bound target joints (“Ours (directly copy)”) after patch matching in [Algorithm 1](#algorithm1). As shown in Table [6](#S4.T6), direct copying achieves slightly higher transfer quality on cross-similar-skeleton transfer. In cross‐species transfer, the drawback is amplified. These results confirm that we can directly copy the bound joint motion to the target skeleton when two skeletons share similar skeleton topologies (e.g. SMPL to other human characters).

What is the best binding rate for bone correspondence? Due to page limits, we answer this question in [Appendix B](#A2).

![Figure](x9.png)

*Figure 11. Application: Lifting SMPL-based motion to any characters. (A) The SMPL-based source motion captured from video. (B) The character with 331 joints was retargeted by the SMPL-based motion. The bound 21 joints are in purple. The retargeted frames are shown in [Fig. 10](#S4.F10).*

## 5. Application: Lifting A SMPL-based Motion to “Any” Character

A direct and realistic application of Motion2Motion is lifting the topological skeleton structure into more complex types. For instance, the generated SMPL-based motion with text or music [(Jiang et al., [2024](#bib.bib24); Li et al., [2023b](#bib.bib29))], even captured from videos [(Zhang et al., [2025](#bib.bib54); Shen et al., [2024](#bib.bib40))], cannot be directly retargeted to the characters used in industrial characters, whose skeletons have higher DoFs. As shown in [Fig. 11](#S4.F11), we retarget a SMPL-based motion captured by HumanMM [(Zhang et al., [2025](#bib.bib54))] to a new character with 331 joints, including skirt and hair points. This shows a strong application of retarget motion to private characters from generated/captured SMPL [(Loper et al., [2023](#bib.bib33))] motions, whose algorithm is the most widely studied by the research community.

## 6. User Interface

![Figure](x10.png)

*Figure 12. User interface: Blender add-on for Motion2Motion. (A) Source motion of the flamingo. (B) Users can select few-shot samples from three motions from the target monkey skeleton. (C) The retargeted motion of the monkey. (D) Global settings of the skeleton choices. (E) The bone binding module (options: automatic or manual binding modes).*

To demonstrate the practical viability of our Motion2Motion, we develop a Blender add-on that integrates seamlessly with the native animation workflow, shown in [Fig. 12](#S6.F12). The user first loads a source motion (A), for example a flamingo walking motion, then selects a few target-skeleton reference clips (B). The global panel (D) lets users specify the source armature, target motions, and blending weight $\alpha$, while the bone-binding module (E) supports both automatic binding and manual adjustment of bone pairs. After clicking the “transfer” button, the retargeted motion (C) will be synthesized in real time. This intuitive interface streamlines the cross-topology motion transfer application and indicates our method can be deployed directly within the animation creation workflow.

## 7. Conclusion

Conclusion. In this paper, we propose a novel framework, Motion2Motion, to retarget an animation from a source skeleton to a target one, requiring only very sparse bone correspondence. The proposed framework works in a training-free fashion with real-time efficiency on CPU-only devices. Our algorithm works in a patchwise motion matching mechanism, supporting flexible motion features for matching. Extensive experimental results indicate that our method is quite robust in in-species and cross-species motion transfer, demonstrating its applicability in diverse downstream tasks, especially transfer SMPL-based motion to “any” character. We also show the interaction interface of Motion2Motion in applications of animation creation pipelines.

Failure cases. Though our method works well in some scenarios, it may fail in some cases. If the target examples are quite semantically different from the source motion (kungfu v.s*. dancing), the retargeted result fails. Despite this, the community still lacks a reasonable solution to this. At present, requiring some key frames as target examples by humans is a relatively applicable method for this.

Limitations and future work. While this work represents a pioneering effort in cross-topology motion transfer, it still has limitations. Our algorithm relies on one- or few-shot target motions, which imposes laborious demands on the animation creation pipeline. Although this level of data requirement is minimal by current community standards, we will continue to explore more data-efficient and lightweight approaches in future work.

## Acknowledgement

The Motion2Motion author team would like to acknowledge all program committee members for their extensive efforts and constructive suggestions. In addition, Weiyu Li (HKUST), Shunlin Lu (CUHK-SZ), and Bohong Chen (ZJU) had discussed with the author team many times throughout the process. The author team would like to convey sincere appreciation to them as well.

## References

-
(1)

-
Aberman et al. (2020)

Kfir Aberman, Peizhuo Li, Dani Lischinski, Olga Sorkine-Hornung, Daniel Cohen-Or, and Baoquan Chen. 2020.

Skeleton-aware networks for deep motion retargeting.

*ACM TOG* 39, 4 (2020), 62–1.

-
Adobe (2024)

Adobe. 2024.

Mixamo.

[https://www.mixamo.com/](https://www.mixamo.com/).

Accessed: 2025-05-21.

-
Agata and Igarashi (2025)

Naoki Agata and Takeo Igarashi. 2025.

Motion Control via Metric-Aligning Motion Matching. In *Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers* *(SIGGRAPH Conference Papers ’25)*. Association for Computing Machinery, New York, NY, USA, Article 11, 12 pages.

[https://doi.org/10.1145/3721238.3730665](https://doi.org/10.1145/3721238.3730665)

-
Artell (2025)

Artell. 2025.

Auto-Rig Pro.

[https://superhivemarket.com/products/auto-rig-pro](https://superhivemarket.com/products/auto-rig-pro).

Accessed: 2025-05-21.

-
Athanasiou et al. (2022)

Nikos Athanasiou, Mathis Petrovich, Michael J Black, and Gül Varol. 2022.

Teach: Temporal action composition for 3d humans. In *3DV*. 414–423.

-
Barquero et al. (2024)

German Barquero, Sergio Escalera, and Cristina Palmero. 2024.

Seamless human motion composition with blended positional encodings. In *CVPR*. 457–469.

-
Bergamin et al. (2019)

Kevin Bergamin, Simon Clavet, Daniel Holden, and James Richard Forbes. 2019.

DReCon: data-driven responsive control of physics-based characters.

*ACM Transactions On Graphics (TOG)* 38, 6 (2019), 1–11.

-
Bernardin et al. (2017)

Antonin Bernardin, Ludovic Hoyet, Antonio Mucherino, Douglas Gonçalves, and Franck Multon. 2017.

Normalized Euclidean distance matrices for human motion retargeting. In *Proceedings of the 10th International Conference on Motion in Games*. 1–6.

-
Bhattacharya et al. (2021)

Uttaran Bhattacharya, Nicholas Rewkowski, Abhishek Banerjee, Pooja Guhan, Aniket Bera, and Dinesh Manocha. 2021.

Text2gestures: A transformer-based network for generating emotive body gestures for virtual agents. In *VR*. 1–10.

-
Büttner and Clavet (2015)

Michael Büttner and Simon Clavet. 2015.

Motion Matching - The Road to Next Gen Animation. In *Proceedings of Nucl.ai Conference 2015*.

[https://www.youtube.com/watch?v=z_wpgHFSWss&t=658s](https://www.youtube.com/watch?v=z_wpgHFSWss&t=658s)

Presentation Video.

-
Chen et al. (2024)

Bohong Chen, Yumeng Li, Yao-Xiang Ding, Tianjia Shao, and Kun Zhou. 2024.

Enabling synergistic full-body control in prompt-based co-speech motion generation. In *Proceedings of the 32nd ACM International Conference on Multimedia*. 6774–6783.

-
Chen et al. (2023b)

Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, and Tongliang Liu. 2023b.

Humanmac: Masked motion completion for human motion prediction. In *ICCV*. 9544–9555.

-
Chen et al. (2023a)

Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. 2023a.

Executing your commands via motion diffusion in latent space. In *CVPR*. 18000–18010.

-
Dabral et al. (2023)

Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. 2023.

Mofusion: A framework for denoising-diffusion-based motion synthesis. In *CVPR*. 9760–9770.

-
Dai et al. (2024)

Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. 2024.

MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model.

*ECCV* (2024).

-
Feng et al. (2012)

Andrew Feng, Yazhou Huang, Yuyu Xu, and Ari Shapiro. 2012.

Automating the transfer of a generic set of behaviors onto a virtual character. In *Motion in Games: 5th International Conference, MIG 2012, Rennes, France, November 15-17, 2012. Proceedings 5*. Springer, 134–145.

-
Gat et al. (2025)

Inbar Gat, Sigal Raab, Guy Tevet, Yuval Reshef, Amit H Bermano, and Daniel Cohen-Or. 2025.

AnyTop: Character Animation Diffusion with Any Topology.

*arXiv preprint arXiv:2502.17327* (2025).

-
Gleicher (1998)

Michael Gleicher. 1998.

Retargetting motion to new characters. In *Proceedings of the 25th annual conference on Computer graphics and interactive techniques*. 33–42.

-
Harvey et al. (2020)

Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. 2020.

Robust Motion In-Betweening.

39, 4 (2020).

-
Ho et al. (2010)

Edmond SL Ho, Taku Komura, and Chiew-Lan Tai. 2010.

Spatial relationship preserving character motion adaptation.

In *ACM SIGGRAPH 2010 papers*. 1–8.

-
Holden et al. (2020)

Daniel Holden, Oussama Kanoun, Maksym Perepichka, and Tiberiu Popa. 2020.

Learned motion matching.

*ACM Transactions on Graphics (ToG)* 39, 4 (2020), 53–1.

-
Jang et al. (2018)

Hanyoung Jang, Byungjun Kwon, Moonwon Yu, Seong Uk Kim, and Jongmin Kim. 2018.

A variational u-net for motion retargeting.

In *SIGGRAPH Asia 2018 Posters*. 1–2.

-
Jiang et al. (2024)

Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. 2024.

Motiongpt: Human motion as a foreign language.

*NeurIPS* (2024).

-
Jin et al. (2018)

Taeil Jin, Meekyoung Kim, and Sung-Hee Lee. 2018.

Aura mesh: Motion retargeting to preserve the spatial relationships between skinned characters. In *Computer Graphics Forum*, Vol. 37. Wiley Online Library, 311–320.

-
Lee and Shin (1999)

Jehee Lee and Sung Yong Shin. 1999.

A hierarchical approach to interactive motion editing for human-like figures. In *Proceedings of the 26th annual conference on Computer graphics and interactive techniques*. 39–48.

-
Li et al. (2022a)

Peizhuo Li, Kfir Aberman, Zihan Zhang, Rana Hanocka, and Olga Sorkine-Hornung. 2022a.

GANimator: Neural Motion Synthesis from a Single Sequence.

*ACM TOG* 41, 4 (2022), 138.

-
Li et al. (2024)

Peizhuo Li, Sebastian Starke, Yuting Ye, and Olga Sorkine-Hornung. 2024.

WalkTheDog: Cross-Morphology Motion Alignment via Phase Manifolds. In *SIGGRAPH, Technical Papers*.

[https://doi.org/10.1145/3641519.3657508](https://doi.org/10.1145/3641519.3657508)

-
Li et al. (2023b)

Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. 2023b.

Finedance: A fine-grained choreography dataset for 3d full body dance generation. In *ICCV*. 10234–10243.

-
Li et al. (2022b)

Shujie Li, Lei Wang, Wei Jia, Yang Zhao, and Liping Zheng. 2022b.

An iterative solution for improving the generalization ability of unsupervised skeleton motion retargeting.

*Computers & Graphics* 104 (2022), 129–139.

-
Li et al. (2023a)

Weiyu Li, Xuelin Chen, Peizhuo Li, Olga Sorkine-Hornung, and Baoquan Chen. 2023a.

Example-based Motion Synthesis via Generative Motion Matching.

*ACM TOG* 42, 4, Article 94 (2023).

[https://doi.org/10.1145/3592395](https://doi.org/10.1145/3592395)

-
Lim et al. (2019)

Jongin Lim, Hyung Jin Chang, and Jin Young Choi. 2019.

Pmnet: Learning of disentangled pose and movement for unsupervised motion retargeting. In *30th British Machine Vision Conference (BMVC 2019)*. British Machine Vision Association, BMVA.

-
Loper et al. (2023)

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. 2023.

SMPL: A skinned multi-person linear model.

In *Seminal Graphics Papers: Pushing the Boundaries, Volume 2*. 851–866.

-
Lu et al. (2024)

Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum. 2024.

Humantomato: Text-aligned whole-body motion generation.

*ICML* (2024).

-
Lyard and Magnenat-Thalmann (2008)

Etienne Lyard and Nadia Magnenat-Thalmann. 2008.

Motion adaptation based on character shape.

*Computer Animation and Virtual Worlds* 19, 3-4 (2008), 189–198.

-
Mahmood et al. (2019)

Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. 2019.

AMASS: Archive of motion capture as surface shapes. In *ICCV*. 5442–5451.

-
Petrovich et al. (2022)

Mathis Petrovich, Michael J Black, and Gül Varol. 2022.

TEMOS: Generating diverse human motions from textual descriptions. In *ECCV*. 480–497.

-
Rokoko (2021)

Rokoko. 2021.

Rokoko Motion Capture Solutions.

[https://www.rokoko.com/](https://www.rokoko.com/).

-
Shafir et al. (2024)

Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. 2024.

Human motion diffusion as a generative prior. In *ICLR*.

-
Shen et al. (2024)

Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. 2024.

World-Grounded Human Motion Recovery via Gravity-View Coordinates. In *SIGGRAPH Asia 2024 Conference Papers*. 1–11.

-
Siyao et al. (2022)

Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. 2022.

Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In *CVPR*. 11050–11059.

-
Sun et al. (2020)

Shixuan Sun, Xibo Sun, Yulin Che, Qiong Luo, and Bingsheng He. 2020.

Rapidmatch: A holistic approach to subgraph query processing.

*Proceedings of the VLDB Endowment* 14, 2 (2020), 176–188.

-
Tevet et al. (2022)

Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. 2022.

Human motion diffusion model. In *ICLR*.

-
Truebones (2022)

Truebones. 2022.

Truebones. Free FBX/BVH Zoo. Over 75 Animals with Animations and Textures.

[https://truebones.gumroad.com/l/skZMC/](https://truebones.gumroad.com/l/skZMC/).

Accessed: 2024-07-01.

-
Tseng et al. (2023)

Jonathan Tseng, Rodrigo Castellon, and Karen Liu. 2023.

Edge: Editable dance generation from music. In *CVPR*. 448–458.

-
Wang et al. (2022)

Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, Wei Liang, and Siyuan Huang. 2022.

Humanise: Language-conditioned human motion generation in 3d scenes.

*NeurIPS* (2022), 14959–14971.

-
Xie et al. (2024)

Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. 2024.

Omnicontrol: Control any joint at any time for human motion generation. In *ICLR*.

-
Ye et al. (2024)

Zijie Ye, Jia-Wei Liu, Jia Jia, Shikun Sun, and Mike Zheng Shou. 2024.

Skinned Motion Retargeting with Dense Geometric Interaction Perception.

*Advances in Neural Information Processing Systems* 37 (2024), 125907–125934.

-
Yuan et al. (2023)

Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. 2023.

Physdiff: Physics-guided human motion diffusion model. In *ICCV*. 16010–16021.

-
Zhang et al. (2024b)

Haodong Zhang, Zhike Chen, Haocheng Xu, Lei Hao, Xiaofei Wu, Songcen Xu, Zhensong Zhang, Yue Wang, and Rong Xiong. 2024b.

Semantics-aware motion retargeting with vision-language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2155–2164.

-
Zhang et al. (2023b)

Jiaxu Zhang, Junwu Weng, Di Kang, Fang Zhao, Shaoli Huang, Xuefei Zhe, Linchao Bao, Ying Shan, Jue Wang, and Zhigang Tu. 2023b.

Skinned motion retargeting with residual perception of motion semantics & geometry. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 13864–13872.

-
Zhang et al. (2024a)

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. 2024a.

Motiondiffuse: Text-driven human motion generation with diffusion model.

*IEEE TPAMI* (2024).

-
Zhang et al. (2023a)

Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. 2023a.

ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model. In *ICCV*.

-
Zhang et al. (2025)

Yuhong Zhang, Guanlin Wu, Ling-Hao Chen, Zhuokai Zhao, Jing Lin, Xiaoke Jiang, Jiamin Wu, Zhuoheng Li, Hao Frank Yang, Haoqian Wang, et al. 2025.

HumanMM: Global Human Motion Recovery from Multi-shot Videos.

*arXiv preprint arXiv:2503.07597* (2025).

-
Zhao et al. (2024)

Qingqing Zhao, Peizhuo Li, Wang Yifan, Olga Sorkine-Hornung, and Gordon Wetzstein. 2024.

Pose-to-Motion: Cross-Domain Motion Retargeting with Pose Prior.

*SCA*.

-
Zhou et al. (2024)

Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. 2024.

Emdm: Efficient motion diffusion model for fast, high-quality motion generation.

*ECCV* (2024).

## Appendix A Joint Binding Algorithm

Motivated by [Sun et al. ([2020](#bib.bib42))], we design an automatic bone binding method for users. Let $|\mathcal{N}|$ be the number of nodes in the tree and $L$ be the maximum path length. For each node, the algorithm traces upward for at most $L$ steps, resulting in a time complexity of $\mathcal{O}(|\mathcal{N}|L)$. Each valid path contains at most $L$ nodes, and up to $|\mathcal{N}|$ such paths can be collected, leading to a space complexity of $\mathcal{O}(|\mathcal{N}|L)$. The algorithm is detailed in [Algorithm 2](#algorithm2).

Before excusing the matching algorithm above, we calculate the unit direction vector of each bone for the source and target skeletons. For the source and target skeletons, given the same path length $L$, we construct two node sets for each ($\mathcal{C}_{s}$ and $\mathcal{C}_{t}$). We calculate the cosine similarity of two direction vectors from two sets pairwisely. Accordingly, we calculate all L-length subgraph similarities by the proposed similarity measurement. We return the pair with the highest similarity to the user as the binding bones, and feed them into the Motion2Motion.

*Algorithm 2 Trace $L$-length Paths of a skeleton tree*

## Appendix B Desired Binding Rate

Here, we raise a research question: What is the best binding rate for bone correspondence? To answer this question, we ablate a group of binding joint numbers. To simplify and align the setting, we set the binding mode to automatic. We calculate the bonding rate from 2% to 15%, and our default setting in the automatic binding mode is 6.1%.
As can be seen in [Fig. 13](#A2.F13), different binding rates have varying effects on both similar skeletons and cross-species motion transfer. Our default setting of 6.1% achieves a favorable balance between motion fidelity and anatomical plausibility. When the binding rate is too small, it is hard for the algorithm to find the coherence between two skeletons. If the rate is too large, it will introduce some mismatched correspondences, resulting in some inaccuracy. That is to say, sometimes “less is more”. Fortunately, our binding algorithm is interactive to choose the bindings, which is adjustable for users.

![Figure](x11.png)

*(a) (A) Similar skeleton motion transfer.*

![Figure](x12.png)

*(b) (B) Cross-species skeleton motion transfer.*

Figure 13. Comparison of different binding rates.

## Appendix C User Study Template

We provide the template of the user study cases in [Fig. 14](#A3.F14). The participants are asked to watch the source motion (top-left) and evaluate three retargeted results (A/B/C) based on motion quality and coherence. Each result is scored on a 5-point Likert scale, where 5 indicates the highest quality. This setup allows for a consistent and fair comparison of different transfer methods.

![Figure](x13.png)

*Figure 14. User study template.*

Generated on Mon Aug 18 16:04:41 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)