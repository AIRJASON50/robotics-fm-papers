[\ConferencePaper\CGFStandardLicense\biberVersion\BibtexOrBiblatex\addbibresource sample-bibliography.bib \electronicVersion\PrintedOrElectronic \teaser* Output of our retargeting method, showcasing several of our contributions. The source pose (left scene) shows complex contacts carrying semantic information : self-contacts, foot-ground contacts, and inter-character contacts). This pose is retargeted onto very different characters from the animated movie industry, evolving on a non-flat terrain. # ReConForM : Real-time Contact-aware Motion Retargeting for more Diverse Character Morphologies T. Cheynel1,2and T. Rossi1 and B. Bellot-Gurlet1and D. Rohmer2and M.P. Cani2 1Kinetix 2LIX, École Polytechnique, CNRS, IP Paris ###### Abstract Preserving semantics, in particular in terms of contacts, is a key challenge when retargeting motion between characters of different morphologies. Our solution relies on a low-dimensional embedding of the character’s mesh, based on rigged key vertices that are automatically transferred from the source to the target. Motion descriptors are extracted from the trajectories of these key vertices, providing an embedding that contains combined semantic information about both shape and pose. A novel, adaptive algorithm is then used to automatically select and weight the most relevant features over time, enabling us to efficiently optimize the target motion until it conforms to these constraints, so as to preserve the semantics of the source motion. Our solution allows extensions to several novel use-cases where morphology and mesh contacts were previously overlooked, such as multi-character retargeting and motion transfer on uneven terrains. As our results show, our method is able to achieve real-time retargeting onto a wide variety of characters. Extensive experiments and comparison with state-of-the-art methods using several relevant metrics demonstrate improved results, both in terms of motion smoothness and contact accuracy. {CCSXML} <ccs2012> <concept> <concept_id>10010147.10010371.10010352</concept_id> <concept_desc>Computing methodologies Animation</concept_desc> <concept_significance>500</concept_significance> </concept> <concept> <concept_id>10010147.10010371.10010352.10010380</concept_id> <concept_desc>Computing methodologies Motion processing</concept_desc> <concept_significance>300</concept_significance> </concept> </ccs2012> \ccsdesc [500]Computing methodologies Animation \ccsdesc[300]Computing methodologies Motion processing \printccsdesc ††volume: 44††issue: 2 ## 1 Introduction Motion transfer between animated characters, also called motion retargeting, is a crucial aspect of character animation with major applications in the fields of cinema, video games and virtual reality. The complexity of this task arises from the variety of 3D character models. While different skeletal topologies and skin meshes may make them difficult to compare, retargeting remains an ill-posed problem even when a correspondence exists between the source and target models. Differences in morphological proportions may cause self-penetrations, inaccurate contacts, and a loss of the overall sense of the pose. Finding the right compromises to mitigate these issues while maintaining the expected fidelity to the source motion is a complex task. In particular, well-thought-out metrics, capturing at least in part the semantics of motion, need to be designed. Alternatively, machine learning shows promise in enabling pose semantics to be discovered and retargeted without any explicit definition. However, the lack of paired retargeting data makes the use of supervised learning difficult. Our work was greatly inspired by the evidence that the preservation of contacts between different body parts is of utmost importance when humans assess the quality of a retargeted motion [basset22]. We argue that contacts with the ground, present in almost every motion, are equally important. It would, however, be very time-consuming to manually identify the most relevant contact interactions of a specific movement, which may change over time, and then find a compromise between their influences in order to preserve the relevant mesh contacts. Instead, we introduce specific motion features to describe the relative positioning of body parts and propose an automatic, adaptive solution that dynamically selects and weights these features over time, focusing on periods around key events such as collisions and contacts. The selected weighted features are then used to compute lightweight objective functions, optimized in real time to generate the target motion. In short, our main contributions are: • A new joint representation for the character’s morphology and motion, used to compute relevant motion features; • An adaptive proximity-based method to select and weight motion features over time; • Extensions to multi-character retargeting and non-flat grounds, enabled by the adaptability of our framework. The evaluation, both qualitative via a user study and quantitative thanks to various metrics, shows that our method achieves better results than the state of the art, both in terms of temporal fluidity of the target animation and semantic similarity with the source. Furthermore, adaptively selected constraints being sparse in space and time, our method (named ReConForM) achieves retargeting tasks several orders of magnitude faster than former optimization-based methods, while accommodating arbitrary morphologies and being easily extensible to new use cases. ## 2 Related Work (a) Penetration with the floor (legs, belly), hands floating above ground (b) Colliding limbs (arms and legs) (c) Self-collision on the thigh, feet floating above ground (d) Hand floating above ground Figure 1: Examples of issues found in NKN’s dataset [nkn]. Figures 1(a)](#S2.F1.sf1) and [1(b)](#S2.F1.sf2) are called “ground-truth” although they were retargeted using Mixamo, while figures [1(c)](#S2.F1.sf3) and [1(d)](#S2.F1.sf4) show issues without having undergone any retargeting (original source characters for those motions).

### 2.1 Optimization-based retargeting

A first category of methods formulate motion retargeting tasks as an optimization problem. Gleicher [gleicher] pioneered this approach
with a solution requiring the manual identification of a number of spacetime constraints. Still only considering skeletal motion, the method was improved thanks to inverse kinematics (IK) solvers [lee],
the addition of constraints to the IK framework [choi], and of dynamic constraints
to improve the respect of physical laws [tak].

Noting that character morphology should have a significant impact on motion, recent optimization-based methods
addressed motion retargeting for skinned characters.
Jin et al. [taeil17] introduced the concept of Aura Mesh to retarget two characters whose meshes are interacting, and Basset et al. [basset20]
optimized energy functions based on volume preservation and collision management. However, both methods are limited to parametric skin mesh models such as SMPL [smpl], used to directly pair the vertices of the source and target meshes.
Ho et al. [ho2010] performed multi-character motion retargeting using an optimization method based on interacting joints and hard constraints to prevent collisions of bone-attached primitives (serving as character shape proxies).
However, this method cannot retarget motion to characters of arbitrary morphologies, as the iterative motion morphing step requires the source and target characters to have identical skeleton topology.

### 2.2 Learning-based retargeting

Other methods leveraged machine learning techniques for motion retargeting. Shon et al. [shon] modeled common latent representation of motion using Gaussian process regression, but this method required datasets of paired motion data for both characters, which are usually unavailable.
To alleviate this issue, Villegas et al. [nkn] introduced a cycle consistency loss for an adversarial unsupervised training framework. Lim et al. [pmnet] learned to disentangle pose (joints’ local coordinates) from movement (the root bone’s global trajectory). By encoding motion in the shared latent space of a common primal skeleton [san, Aberman_2019, Hu_2023, yan2024imitationnet], retargeting was performed in deep feature space, thus increasing generalization to various skeletons.
Yet, as these methods only considered the skeleton and not the mesh,
they failed to capture self-contacts, despite them being a key component of motion.

In contrast, Villegas et al. [villegas] detected
self-contacts and used geometry-conditioned RNN to preserve them while optimizing motion in a latent space.
Despite yielding good results, this method may not be adapted to industrial use cases as the model cannot handle arbitrary skeleton topology and is not suitable for real-time applications. Besides, comparison with this method is difficult as no implementation is available online.
More recently, R2ET [r2et] introduced a shape-aware module that improves self-contacts as a post-processing step on the target mesh, and avoids unwanted penetrations, in real-time. Their method can handle very diverse characters, but collisions on the source mesh are not taken into account, meaning that semantic information can be lost during retargeting.

Recent works [zhang2024semanticsawaremotionretargetingvisionlanguage] focus on motion semantics by performing a differentiable rendering of the animated character’s mesh, and encoding the video through a vision-language model. However, this complex setup has only been shown to perform on simple motions that are easily transcribed into a textual description (e.g. “shrugging", “‘waving", “praying while standing up"), with little to no interaction between the limbs and minimal foot motion ; and requires the same skeleton between source and target characters.

Lastly, some recent works [reda23, yunbo23] used a combination of deep reinforcement learning and physically-based simulation to allow for a physically accurate retargeting, taking mesh contacts into account. The main issue is that of generalization: unlike previously mentioned methods, the model has to be retrained for each new target character, making it impossible to scale up to the diversity of virtual characters used in most 3D computer graphics applications.

### 2.3 Mesh correspondence

Some methods [zhou20unsupervised, wang2020neural, liao2022pose] aim to disentangle the pose and shape of a mesh to transfer poses to a target with a different shape. Because we have skeletal animations and a common standard pose amongst our characters, this disentanglement is trivial. However, we still need a way to compare shapes by computing a correspondence between the source and target meshes
(see [van2011survey] and [recent_advances] for surveys on this topic).
A recent family of solutions relied on optimal transport. Solomon et al. [solomon] used a Gromov-Wasserstein distance with an entropy term to find a dense mapping between 3D meshes. Mandad et al. [desbrun] increased efficiency by using a coarse-to-fine approach relying on diffusion geometry, while Schmidt et al. [schmidt2023surface]
used an automatic triangulation to approximate the source and target surfaces.
In this work, we take inspiration from these techniques to compute a correspondence between the source and target models.

### 2.4 Retargeting datasets

Recent works in the field of motion retargeting [pmnet, r2et, san] made use of the dataset introduced in NKN [nkn], with 2400 motion sequences sampled from Mixamo [mixamo].
We visually assessed its quality, and expose two major shortcomings.
First, the variety of characters was obtained using Mixamo’s in-house retargeting algorithm, despite many of these retargeted animations showing degraded motion semantics (where the intention behind a movement is completely lost), severe self-collisions, and ground collisions. Examples of such issues are shown in Figures [1(a)](#S2.F1.sf1) and [1(b)](#S2.F1.sf2). Second, some of the original motions from Mixamo show severe flaws on their original character (before any retargeting), as illustrated by Figures [1(c)](#S2.F1.sf3) and [1(d)](#S2.F1.sf4).
Therefore, we claim that using this dataset as ground truth, with metrics such as the MSE of joint positions, is not an accurate way to estimate the quality of motion retargeting methods. It may even have impacted the quality of previous work: while it was used for unsupervised training by Villegas et al. [nkn], some other methods used this dataset as evaluation data, training data [pmnet, r2et] and even as ground-truth for supervised learning [san].

In contrast, we use a few quality metrics such as self-penetration, foot sliding, and jerk, to assess the quality of the animations we generate, while referring to
human feedback, through a user-study, to assess the overall semantic-preserving quality of our method.

## 3 Shape and motion descriptors

Let us consider a source* and a *target* humanoid characters, both with a skin-mesh rigged to a skeleton. Neither the skeletons or the meshes need to have the same topologies.
The skeletons may have different numbers of joints so long as a *bone mapping*, i.e. a list of paired bones on both characters, is predefined; and the meshes can exhibit different numbers of vertices and large shape differences.
In addition, an input, kinematic animation to be transferred is provided for the source character.

To compute relevant, semantic-based motion transfer from source to target, some common representation for the characters’ shapes and a set of motion descriptors for the input animation are required. Tracking the distance between relevant pairs of points was already used as a simple and robust way to encode the semantic of motion in the fields of robotics [handa2019dexpilot] and motion retargeting [ho2010, yunbo23].
Using points located on the skin mesh instead of skeletal joints captures finer semantic information and helps preventing collisions [1013569].
Taking inspiration from these works, we propose a light morphological representation based on
key-vertices, described in section [3.1](#S3.SS1), and then use it to build a set of time-varying descriptors encoding the successive poses in the input animation.

### 3.1 Sparse shape encoding and correspondence

The ReConForM method uses a generic humanoid template mesh
(based on the SMPL model [smpl]), on which we pre-selected $N$ specific vertices, called
*key-vertices*
($N=41$ unless mentioned otherwise, see Figure [2](#S3.F2)).
The latter are chosen as to provide (i) a sparse yet comprehensive coverage of the character’s surface, and (ii) a good sampling of typical areas prone to contact, such as hands and feet.
While our method was developed for humanoid characters, its principle could easily be extended to different categories of characters, just by changing this template model.

To provide a sparse one-to-one correspondence between the source and target model, we automatically transfer the template’s keypoints to their two skin meshes, as follows.
Taking inspiration from optimal transport between arbitrary meshes [solomon], we view the task as an optimization problem.
We input both the template and destination (i.e., source or target) meshes in their T-pose. Using the skinning weights, we first split the mesh into its various limbs (arms, legs, torso, head, feet and hands). For each limb, we consider the position of all $N^{l}$ vertices, and normalize them to have zero mean and unit variance.
For each limb $l$, we convert the template and destination point-clouds into two distributions :

| | $$\mathcal{T}=\frac{1}{N^{l}_{t}}\sum\limits_{i=1}^{N^{l}_{t}}\delta_{v^{l}_{i,t% }}\ \ \text{ and }\ \ \mathcal{D}=\frac{1}{N^{l}_{d}}\sum\limits_{j=1}^{N^{l}_% {d}}\delta_{v^{l}_{j,d}}$$ | |
|---|---|---|

where $v^{l}_{i,t}$ and $v^{l}_{j,d}$ are, respectively, the normalized position of the $i$-th and $j-th$ vertices of limb $l$ for the template and destination meshes, and $\delta$ is the Dirac distribution.

The constraints of our setup (namely, that the meshes are in T-pose and normalized) allow us to use a simple euclidian distance as criterion, instead of relying on the Gromov-Wasserstein distance used in Solomon et al. [solomon]. Our goal is to find an optimal transport plan, which is a matrix $P^{l}\in\mathbb{R}_{+}^{N^{l}_{t}\times N^{l}_{d}}$ defining a coupling between the two distributions $\mathcal{T}$ and $\mathcal{D}$.

In order to account for an uneven distribution of vertices, we weight each vertex by the inverse of the local vertex density :

| | $$w^{l}_{i}\ =\ \sum\limits_{f\in\mathcal{F}_{i}}\frac{A(f)}{n(f)}$$ | |
|---|---|---|

where $\mathcal{F}_{i}$ is the set of faces containing vertex $i$, $A(f)$ the area of face $f$, and $n(f)$ its number of vertices.

We solve the following optimization problem:

| | $\displaystyle{P^{l}}^{*}\ \ =\ \$ | $\displaystyle\underset{P^{l}}{\operatorname{argmin}}\ \langle P^{l}\,,\,C^{l}% \rangle\ \ =\ \ \underset{P^{l}}{\operatorname{argmin}}\ \sum\limits_{i=1}^{N^% {l}_{t}}\sum\limits_{j=1}^{N^{l}_{d}}P^{l}_{i,j}C^{l}_{i,j}$ | |
|---|---|---|---|
| | | $\displaystyle\text{s.t. }\ \ \mathds{1}\cdot P^{l}\ =\ \left(w^{l}_{0,d}\ ;\ % \cdots\ ;\ w^{l}_{N^{l}_{d},d}\right)$ | |
| | | $\displaystyle\hskip 17.92537ptP^{l}\cdot\mathds{1}^{T}\ =\ \left(w^{l}_{0,t}\ % ;\ \cdots\ ;\ w^{l}_{N^{l}_{t},t}\right)^{T}$ | |

where $C^{l}_{i,j}=\left\lVert v^{l}_{i,t}-v^{l}_{j,d}\right\rVert^{2}$, and $\mathds{1}$ is a vector filled with ones.

Similarly to Feydy et al. [feydy2019interpolating], we approximate this optimal transport plan by solving the entropy regularized version of the problem, which is faster to compute using Sinkhorn’s algorithm [cuturi2013sinkhorndistanceslightspeedcomputation]. Thus, for each key-vertex on the template, we are thus able to get a corresponding vertex on the destination mesh, which effectively gives us the position of the key-vertices for both the source and target characters.

![Figure](extracted/6238052/contents/images/keypoint_transfer/YBot.jpg)

![Figure](extracted/6238052/contents/images/keypoint_transfer/Kaya.jpg)

![Figure](extracted/6238052/contents/images/keypoint_transfer/Michelle.jpg)

![Figure](extracted/6238052/contents/images/keypoint_transfer/Ortiz.jpg)

![Figure](extracted/6238052/contents/images/keypoint_transfer/BigVegas.jpg)

Figure 2: Chosen location of
key-vertices on the template mesh (top-left), and results of the key vertices transfer to several character from Mixamo [mixamo]. Key vertices are shown with corresponding colored
spheres to show the automatic transfer to the different meshes.

Examples of key-vertex transfer is shown in Figure [2](#S3.F2), and results on additional characters are shown in the Supp. Mat, along with results for our extended template with $N=96$ keypoints. Note that the output of this automatic transfer can be overridden, or further refined if needed via interactive authoring.
This enables the user to fine-tune key-vertex positions on the source, target, or even on the template shape, to account for very large morphological differences, or to highlight specific areas where careful sampling needs to be considered.

### 3.2 Motion encoding via time-varying pose descriptors

Using the key-vertices defined before, time-varying pose descriptors can be computed on the source animation to extract the semantic elements, in order to transfer them to the target character. Yet, not all descriptors convey the same amount of semantic information: it is necessary to adaptively focus on the pairs of key-vertices interacting with each other or with the ground.

Given an animated character, we call $p_{i}$ (or $p_{i,t}$ explicitly when necessary) the 3D position of the $i^{th}\in\llbracket 1,N\rrbracket$ key-vertex at time $t$, and $n_{i}$ its normal to the mesh surface. We compute three time-varying descriptors $\mathcal{M}_{\text{dist}}$, $\mathcal{M}_{\text{dir}}$, and $\mathcal{M}_{\text{pen}}$,
each stored in a matrix-like structure representing
the pairwise relationship between all possible pairs of key-vertices $(i,j)\in\llbracket 1,N\rrbracket^{2}$ of a character.

These descriptors respectively encode:

-
•

The distances between all pairs of key-vertices,

| | $$\mathcal{M}_{\text{dist}}\in\mathbb{R}^{N\times N}\text{, where }\mathcal{M}_{% \text{dist}}(i,j)=\left\lVert p_{j}-p_{i}\right\rVert;$$ | |
|---|---|---|

-
•

The vector,
and therefore the offset direction, between pairs of key-vertices,

| | $$\mathcal{M}_{\text{dir}}\in\mathbb{R}^{N\times N\times 3}\text{, where }% \mathcal{M}_{\text{dir}}(i,j)=p_{j}-p_{i};$$ | |
|---|---|---|

-
•

The signed distance between two key vertices $i$ and $j$, computed along the mesh normal at vertex $i$, which locally serves as a measure of penetration,

| | $$\mathcal{M}_{\text{pen}}\in\mathbb{R}^{N\times N}\text{, where }\mathcal{M}_{% \text{pen}}(i,j)=n_{i}\cdot(p_{j}-p_{i}).$$ | |
|---|---|---|

Two additional descriptors, $\mathcal{M}_{\text{height}}$ and $\mathcal{M}_{\text{sliding}}$, are used to represent the relationship between individual key-vertices and the environment, considered to be an horizontal ground floor at this stage.
Calling $\overrightarrow{up}$ the unit vertical direction, these two descriptors are stored in vector-like containers encoding:

-
•

The height of each key-vertex with respect to the ground,

| | $$\mathcal{M}_{\text{height}}\in\mathbb{R}^{N\times 1}\text{, where }\mathcal{M}% _{\text{height}}(i)=\overrightarrow{\text{up}}\cdot p_{i};$$ | |
|---|---|---|

-
•

The horizontal velocity of each key-vertex,

| | $$\displaystyle\mathcal{M}_{\text{sliding}}\in\mathbb{R}^{N\times 2}\text{, % where }\mathcal{M}_{\text{sliding}}(i)=\frac{H(p_{i,t+\Delta t})-H(p_{i,t})}{% \Delta t}$$ | |
|---|---|---|
| | $$\displaystyle\mbox{and }\;\text{H}(p_{i})=p_{i}-(\overrightarrow{\text{up}}% \cdot p_{i})\,\overrightarrow{\text{up.}}$$ | |

## 4 Adaptive weighting of constraints

![Figure](x5.jpg)

*Figure 3: Top: graphical representation of our method (left: key-vertices transfer presented in Section [3.1](#S3.SS1); middle: motion retargeting process presented in Sections [3.2](#S3.SS2) through [4.2](#S4.SS2)). Top-right shows two poses being retargeted onto different characters, with key-vertices shown in red. Bottom shows a complex pose showcasing collisions with the floor as well as self-collisions (the leftmost character is the source character, all the others are target characters from Mixamo).*

The motion descriptors defined above are computed at each time on the source animation and used as constraints to be matched on the target animation.
Trying to maintain all these constraints at once might however be unfeasible when the source and target morphologies are different, as they would conflict with one another.
Instead, our insight is that, depending on the input animation, only a few of these descriptors are significant at each time step to the perceived semantics of motion. We therefore introduce a new, dynamic weighting formulation to select only the relevant ones over time and weight them accordingly.

### 4.1 Weighting coefficients

Our descriptor weighting method exploits the notion of spatial proximity between pairwise key-vertices (indicating an interaction between limbs), and of height relative to the ground (indicating a likely occurrence of floor contact).
We therefore make use of two weighting characteristics, also stored in matrix/vector-like structures with values in $[0,1]$, obtained via a clamped normalized representation of the distances between, and heights of key-vertices:

| | $$\begin{array}{rl}\mathcal{W}_{\text{interaction}}(i,j)&\displaystyle=\text{% clamp}\left(1-\frac{\mathcal{M_{\text{dist}}}(i,j)-d_{\text{min}}}{d_{\text{% max}}-d_{\text{min}}}\right)\\ \mathcal{W}_{\text{floor}}(i)&\displaystyle=\text{clamp}\left(1-\frac{\mathcal% {M_{\text{height}}}(i)-h_{\text{min}}}{h_{\text{max}}-h_{\text{min}}}\right)% \end{array}$$ | |
|---|---|---|

where $\text{clamp}(x)=\min(1,\max(0,x))$ and $d_{\text{min}}$, $d_{\text{max}}$, and $h_{\text{min}}$, $h_{\text{max}}$ are per-character constant thresholds characterizing typical distances
for which nearby limb interaction or ground contact should be considered. For a character of height $h_{c}$, we typically set $d_{\min}=h_{\min}=5\%\,h_{c}$, and $d_{\max}=h_{\max}=15\%\,h_{c}$.

These weights provide a measure of the importance of each constraint, at each animation frame.
More precisely,
$\mathcal{W}_{\text{interaction}}(i,j)$ (resp. $\mathcal{W}_{\text{floor}}(i)$) will be close to 1 when a constraint between two interacting limbs (resp. between a limb and the floor) must be taken into account, and be 0 for unimportant relationships to be discarded. Furthermore, even though a pair of key-vertices may interact at a specific animation frame, the associated weight decreases back to 0 as soon as they spread apart. The resulting sparsity of constraints, in space and time, is a key-advantage for the efficiency of our solution.

### 4.2 Sparse objective to be optimized

We formulate the retargeting task as an optimization problem over the skeletal poses variables defined by the local joint rotation q and root joint position in world space $p$, which are related to the key-vertex positions $p_{i}$ and normal $n_{i}$ through linear-blend skinning.

The optimization starts with a rough initialization of q and $p$ through a naive retargeting, obtained by copying the rotation of the source bones onto their equivalent target bones using the bone mapping mentioned in Section [3](#S3).

We then optimize (q, $p$) over the entire animation at once to minimize the combination of the following three losses :

$$\begin{split}\min\limits_{\textbf{q},p}\quad&w_{\text{reg}}\mathcal{L}_{\text{% reg}}+w_{\text{smooth}}\mathcal{L}_{\text{smooth}}+w_{\text{sem}}\mathcal{L}_{% \text{sem}}\end{split}$$ \tag{1}

The regularization loss $\mathcal{L}_{\text{reg}}=\sum_{i\in\llbracket 1,N\rrbracket}\sum_{t}\|p_{i,t}-%
p_{i,t}^{\text{init}}\|^{2}$ is a simple Mean Squared Error (MSE) comparing the key-vertices’ positions to that of the naive retargeting $p_{i,t}^{\text{init}}$, ensuring that the solution stays in the neighborhood of the initialization.

The smoothness loss $\mathcal{L}_{\text{smooth}}=\sum_{i\in\llbracket 1,N\rrbracket}\sum_{t}\lVert%
\dddot{p}_{i,t}\rVert$ ensures that the change in acceleration, or jerk, is as small as possible.

The semantic loss $\mathcal{L}_{\text{sem}}$
is defined as the weighted sum of the following terms (where $\circ$ denotes the element-wise product):

-
•

A term ensuring that the distance between pairs of key-vertices is preserved:

| | $$\mathcal{L}_{\text{dist}}=\sum_{t}\left\lVert\mathcal{W}_{\text{interaction}}^% {\text{src}\to\text{targ}}\circ\left(\mathcal{M}^{\text{source}}_{\text{dist}}% -\mathcal{M}^{\text{target}}_{\text{dist}}\right)\right\rVert^{2}$$ | |
|---|---|---|

-
•

A term ensuring that the direction between pairs of key-vertices is preserved (with $S_{\text{cosine}}$ being the element-wise cosine similarity):

| | $$\mathcal{L}_{\text{dir}}=\sum_{t}\left\lVert\mathcal{W}_{\text{interaction}}^{% \text{src}\to\text{targ}}\circ S_{\text{cosine}}\left(\mathcal{M}^{\text{% source}}_{\text{dir}},\mathcal{M}^{\text{target}}_{\text{dir}}\right)\right% \rVert^{2}$$ | |
|---|---|---|

-
•

A term matching the amount of penetration of pairs of interacting key vertices, between source and target:

| | $$\mathcal{L}_{\text{pen}}=\sum_{t}\left\lVert\mathcal{W}_{\text{interaction}}^{% \text{src}\to\text{targ}}\circ\left(\mathcal{M}^{\text{source}}_{\text{pen}}-% \mathcal{M}^{\text{target}}_{\text{pen}}\right)\right\rVert^{2}$$ | |
|---|---|---|

-
•

A term penalizing key-vertices penetrating the floor, and encouraging similar heights on source and target for key-vertices close to the ground:

| | $$\begin{array}{ll}\mathcal{L}_{\text{height}}=&\displaystyle\sum_{t}\left% \lVert\text{min}\left(0,\mathcal{M}^{\text{target}}_{\text{height}}\right)% \right\rVert^{2}+\\ &\displaystyle\sum_{t}\left\lVert\mathcal{W}_{\text{floor}}^{\text{src}\to% \text{targ}}\circ\left(\mathcal{M}^{\text{source}}_{\text{height}}-\mathcal{M}% ^{\text{target}}_{\text{height}}\right)\right\rVert^{2}\end{array}$$ | |
|---|---|---|

-
•

A term preventing key-vertices that are in contact with the ground from sliding across consecutive frames:

| | $$\mathcal{L}_{\text{sliding}}=\sum_{t}\left\lVert\mathcal{W}_{\text{floor}}^{% \text{src}\to\text{targ}}\circ\left(\mathcal{M}^{\text{source}}_{\text{sliding% }}-\mathcal{M}^{\text{target}}_{\text{sliding}}\right)\right\rVert^{2}$$ | |
|---|---|---|

The matrices $\mathcal{W}_{\text{interaction}}^{\text{src}\to\text{targ}}$ and $\mathcal{W}_{\text{floor}}^{\text{src}\to\text{targ}}$ are a combination of the weighting matrices of the source and target characters:

$$\begin{array}{ll}\mathcal{W}_{\text{interaction}}^{\text{src}\to\text{targ}}% &\displaystyle=\mathcal{W}^{\text{source}}_{\text{interaction}}+\alpha\;\text{% Sg}\left(\mathcal{W}^{\text{target}}_{\text{interaction}}\right)\;,\\ \\ \mathcal{W}_{\text{floor}}^{\text{src}\to\text{targ}}&\displaystyle=\mathcal{W% }^{\text{source}}_{\text{floor}}+\alpha\,\text{Sg}\left(\mathcal{W}^{\text{% target}}_{\text{floor}}\right)\;,\end{array}$$ \tag{2}

where the mixing parameter $\alpha$ increases linearly from 0 to 1 during the optimization process. Both $\mathcal{W}^{\text{target}}_{\text{interaction}}$ and $\mathcal{W}^{\text{target}}_{\text{floor}}$ are weighting matrices computed from the target’s motion descriptors at the current optimization step.
This evolution from source to target weighting helps to guide the numerical optimization toward a correct solution during the first steps, and prevent false positive interactions (i.e., ending up with contacts that were not present in the source motion) for later steps. $\text{Sg}$ is the stop-gradient operation, through which gradient flow is prevented during the optimization to reduce computational time and selectively prevent the optimization process from attempting to minimize unrelated parameters.

The terms of $\mathcal{L}_{\text{sem}}$ were weighted manually to give each loss a roughly equal contribution at the start of the optimization. The values of the hyperparameters, which yield satisfying results regardless of the animation and characters, are provided in supplementary material.

## 5 Results

### 5.1 Implementation

We implemented our ReConForM method using the Pytorch framework to leverage the differentiability of the losses relative to the variables $(\textbf{q},p)$.
We optimized the target poses using Adam [adam].

### 5.2 Evaluation data

We tested our method using animations and characters from Mixamo [mixamo].
The evaluation sets provided by previous works [r2et, nkn, san] lack quality even on the source character.
Instead, we manually selected 45 animations from Mixamo [mixamo] with accurate contacts and asked a 3D animation expert to annotate their estimated difficulty to be retargeted based on the amount of floor contacts and self-collisions present in them: 24 of them were quantified as Hard*, 21 as *Easy*.

### 5.3 Quantitative results

#### 5.3.1 Inference time

Our optimization objectives being very sparse, our method is able to consistently run in real-time (for motions longer than 3 seconds) on a laptop with a Nvidia RTX 3060 GPU, with an asymptotic speed of 67 frames per second.
This makes it suitable for use in video games or motion capture pre-visualization. Since all frames are processed at once in a batched fashion, our method reaches higher framerates for longer animations. A three-second batch (around 75 frames) gives a good compromise between speed and delay for use cases where an online, on-the-fly retargeting is necessary, such as motion capture visualization.
Figure [4](#S5.F4) shows the inference speed for animations of various durations.

*Figure 4: Speed and framerate of our retargeting method*

*Table 1: Metrics on our evaluation set. Green : best, blue : second best. Bottom two lines show the results of the ablation study mentioned in section [5.3.3](#S5.SS3.SSS3).*

#### 5.3.2 Metrics

We evaluated, and report in Table [1](#S5.T1), the performance of ReConForM through several metrics, comparing it to the naive baseline (“Copy Rotations” – see Supp. Mat. for implementation details), to SAN [san], as well as R2ET [r2et] with and without the addition of a 1€ filter [1eurofilter] to
reduce its noise. We also provide comparison with industrial retargeting algorithms of Mixamo [mixamo], Unreal Engine 5 [ue5], and Autodesk Maya/MotionBuilder [humanIK]. Overall, our approach provides the best compromise to avoid self-penetration and foot skating, while preserving a smooth animation.

We evaluated the motion smoothness using the average and maximum magnitude of the jerk of joint positions throughout each animation sequence. Thanks to the integration of our smoothness loss within the optimization, our approach achieves remarkably low jerk (same order of magnitude as the source), leading to very smooth motion.

Quantifying the self-penetration by computing the collision volume between the limbs, normalized by the total volume of the character, we found that our method surpasses the current state-of-the-art while remaining temporally smooth. On the other hand, R2ET, being solely dedicated to reducing self-collisions, shows higher temporal noise. Using a post-process filter [1eurofilter] helps reduce the noise, but spoils, in return, the amount of self-penetration.

Similarly, we evaluated the penetration with the floor by computing the relative proportion of the volume of the character that is below the ground at each frame.
Our method shows a 70% reduction in floor penetration compared to the state-of-the-art.

We finally evaluated the quality of the feet-to-ground contacts along two quantitative measures.
First, we call a foot to be *grounded* when its
distance to the floor is between -1% and 1%
of the character’s height.
Second,we consider that the foot is *locked* on its position if its horizontal speed is lower than 0.1% of the character’s height per second. We computed the F1 and ROC AUC scores to quantify how much the semantic information of feet-to-ground contacts is kept by the retargeting method for these two metrics, with the source motion serving as ground truth. In both cases, our scores indicates that our approach
generates more accurate feet-to-ground contact than the academic state-of-the-art, much closer to the quality of IK-based methods such as Mixamo and MotionBuilder. These methods indeed allow for a very precise placement of the feet, however they tend to do so without consideration for other semantic aspects of motion, leading for example to higher self-collisions and more frequent loss of semantic contacts.

#### 5.3.3 Ablation study

We replaced the vertex transfer process described in section [3.1](#S3.SS1) by key-vertices placed manually by a human expert, to evaluate its impact on the overall retargeting method. We also evaluate with different amounts of key-vertices. These results (bottom lines of Table [1](#S5.T1)) confirm the robustness of our method with respect to the definition and placement of key-vertices. Additional experiments demonstrating the usefulness of each loss can be found in the supplementary material.

### 5.4 Qualitative results

#### 5.4.1 Visual output

Figure ReConForM : Real-time Contact-aware Motion Retargeting for more Diverse Character Morphologies and Figure [3](#S4.F3) show the results of our retargeting on several poses, and Figure [13](#S7.F13) provides a visual comparison to R2ET. Additional results are displayed in the Supp. Mat.

*Figure 5: User preference across all answers, split by difficulty*

None

Mod-
erate

Expert

None

Mod-
erate

Expert
00$0.2$$0.4$$0.6$$0.8$$1$

$13.7$%$8.3$%$4.1$%$13.5$%$6.3$%$6.5$%$28.5$%$23.3$%$9.5$%$35.8$%$31.7$%$9.1$%$57.7$%$68.3$%$86.5$%$50.7$%$61.9$%$84.4$%Proportion of answersFidelity w.r.t. sourcePleasantness

*Figure 6: User preference across all answers, split by expertise level*

#### 5.4.2 User study

We designed a preferential blind user study to evaluate the quality of ReConForM compared to the state-of-the-art R2ET.
We gathered a panel of 133 online respondents including both experts and non-experts.
Each participant evaluated twelve randomly selected animations displayed as videos containing the source animation, the result of our retargeting, and the one obtained using R2ET [r2et] (in random order) on one of eight different characters taken from Mixamo [mixamo]. Participants were asked to choose a preferred output along two questions: first with respect to *fidelity* with the source, and second, with respect to *pleasantness* of the motion.

Figure [5](#S5.F5) summarizes the reported preferences of the users over all the animations (Overall) covering a total of 1596 answers, as well as on both *Easy* and *Hard* subsets.
We observe a clear
preference for our results, with our outputs obtaining the majority of the votes on 41 out of 45 animations. Moreover, we can note that the average score on animations from the *Hard* subset exceeds that on the *Easy* subset.
This makes us hypothesize that the benefits of using our approach increase with the complexity of the retargeting.
A
$\chi^{2}$ test confirms
this claim with $p.

Figure [6](#S5.F6) shows another interesting result: the more respondents are familiar with the task of retargeting, the more they perceive our outputs to be best.
A $\chi^{2}$ test indicates a statistically significant distribution shift between the answers of experts and neophytes, with $p.
All study details are in the supplementary material.

### 5.5 Additional characters

To demonstrate the robustness of our method on various characters, we provide additional results on production-ready characters provided by Blender Studio (CC-BY license). Despite their extreme proportions, and their different skeleton structure (fewer spine bones, different hierarchy, presence of additional “roll bones”…), our key-vertex transfer method provides a remarkably robust output. Several retargeted poses are shown in Figure [11](#S6.F11). Whenever a good target pose is reachable, our optimization yields good results, and when the characters’ morphology prevents it from mimicking a complex pose, the outputs stay visually pleasing, without any catastrophic failure.

### 5.6 Failure cases

#### 5.6.1 Key-vertices transfer

We identified several failure cases which might cause suboptimal results:

-
•

Lack of geometry: low-poly character with few vertices can cause poor results because of a low number of vertices to choose from.

-
•

Asymmetry: while usually insignificant, some extreme asymmetries in the mesh can “pull” the key-vertices of some limbs toward one side, causing an asymmetry in the key-vertices that might not always be preferred.

-
•

Accessories / Props: character modeled with props (i.e. bags, hats, large haircuts…) can cause local errors on the placement of key vertices (directly on or close to the prop).

-
•

Overlapping meshes: when meshes overlap, the transfer might place key-vertices on a “hidden” part of the mesh, which could prevent the retargeting process from considering the outer part of the mesh.

A more complete description of the failure cases, along with examples can be found in the Supplementary Material. We also propose several straightforward solutions to alleviate these specific issues.

#### 5.6.2 Motion Retargeting

We also identified some failure cases of the retargeting process:

-
•

Complex poses: poses with several points of contact (such as yoga poses) are harder to retarget, and finding a compromise between all semantic aspects of the pose can prove challenging when limb proportions are very different.

-
•

Unrealistic motion trajectories: preventing floor collisions can cause a modification of the trajectory of the character, without enforcing the laws of dynamics, might create motions that are not physically accurate, although this effect is only visible on some cartoon characters with extreme proportions.

-
•

Fine-grained contacts on undersampled parts of the mesh: using few keypoints constrains us to a simplified representation of the mesh, which might fail to capture contacts on some parts of the mesh.

A detailed analysis of those failure cases can be found in the Supplementary Material, with examples of each type of failure case.

## 6 Extensions

Thanks to the modular nature of our method, the optimization terms can be easily extended to account for specific and challenging scenarios previously unaddressed in the literature, as well as providing some degree of authoring and interactive control.

### 6.1 Multi-character retargeting

Handling multi-character interaction can be achieved
by extending
the pose descriptors to account for all interactions between the characters.
These interactions may encompass not only physical contact
but also some long-distance interactions such as gaze direction. In particular,
handling eye contact
is key in modeling plausible human interaction.

To this end, we first expand $\mathcal{W}_{\text{interaction}}$ and $\mathcal{W}_{\text{height}}$ to account for all possible distances
between pairs of key-vertices of all the characters.
The respective dimensions of these matrices become, respectively, $N\,N_{\text{char}}\times N\,N_{\text{char}}$ and $N\,N_{\text{char}}\times 1$, where $N_{\text{char}}$ designates the number of characters.
Second, we propose a modified $\mathcal{W}_{\text{interaction}}$ descriptor formulation accounting for an additional gaze direction term. Let us call $\mathcal{I}^{\text{eye}}=(i_{c}^{\text{eye}})_{c\in\llbracket 1,N_{\text{char}%
}\rrbracket}$ the set of indices of the key-vertices associated with the eyes of a character $c$, and typically positioned at the center of the character’s face. We further call $\overrightarrow{\text{gaze}_{c}}$ the averaged gaze direction of character $c$ at a given time, and define the following sparse matrix:

| | $$\mathcal{M}_{\text{gaze}}\in\mathbb{R}^{N_{\text{char}}N\times 3}\text{, where% }\mathcal{M}_{\text{gaze}}(i)=\left\{\begin{array}{ll}\overrightarrow{\text% {gaze}_{c}}&\text{for }i\in\mathcal{I}^{\text{eye}}\\ 0&\text{otherwise.}\\ \end{array}\right.$$ | |
|---|---|---|

We finally
define the extended $\mathcal{W}_{\text{interaction}}$ as :

| | $$\mathcal{W}_{\text{interaction}}\mathrel{+}=\text{clamp}\left(\frac{S_{\text{% cosine}}(\mathcal{M}_{\text{dir}},\mathcal{M}_{\text{gaze}})-\cos(a_{\text{min% }})}{\cos(a_{\text{min}})-\cos(a_{\text{max}})}\right)$$ | |
|---|---|---|

The thresholds $a_{\min}$ and $a_{\max}$ are characteristic angles, set to 2° and 5°.
Figure [7](#S6.F7) illustrates our results on multi-character interactions and shows the separate impacts of adding the inter-character physical contact and the gaze-based extension. Note how the optimization automatically changes the
relative position of the characters, as well as their poses, to account for their height difference.

We also experiment with multi-character retargeting with strictly more than two characters. We design and retarget a five-character motion inspired from a traditional Breton dance, shown in Figure [8](#S6.F8). We found that the multi-character retargeting only runs 19% slower than five independent retargeting processes, due to the sparse nature of the interactions in the method. The speed and memory footprints could likely be optimized further using specific sparse computations libraries.

### 6.2 Non-flat grounds

A second extension of our approach is to generalize the retargeting to non-flat grounds. This can be straightforwardly implemented by adapting the height evaluation of the key-vertices to become coordinates-dependent. Considering a terrain defined as a differentiable height-field $y=f(x,z)$ and key-vertex position coordinates as $p_{i}=(x_{i},y_{i},z_{i})$, we can replace $p_{i}\leftarrow p_{i}-f(x_{i},z_{i})$ in the computation of $\mathcal{M}_{\text{height}}$ and $\mathcal{M}_{\text{sliding}}$ descriptors.
Illustrations of results obtained using this approach are provided in Figure [9](#S6.F9).

![Figure](extracted/6238052/contents/images/multichar/source_arrows.jpg)

*(a) Source characters*

![Figure](extracted/6238052/contents/images/multichar/target_separate_arrows.jpg)

*(b) With no extra pose descriptor.*

![Figure](extracted/6238052/contents/images/multichar/target_multi_arrows.jpg)

*(c) With inter-character contact features, but no gaze direction feature*

![Figure](extracted/6238052/contents/images/multichar/target_separate_with_direction_loss_arrows.jpg)

*(d) With both inter-character contact and gaze direction features.*

Figure 7: Results of the multi-character extension on a high-five pose. Arrows denote the gaze direction.

![Figure](x6.jpg)

*(a) Source (side view)*

![Figure](x7.jpg)

*(b) Source (top view)*

![Figure](x8.jpg)

*(c) Retargeted (side view)*

![Figure](x9.jpg)

*(d) Retargeted (top view)*

Figure 8: Results of the multi-character extension on a five-character motion.

![Figure](x10.jpg)

*(a) Source, flat ground*

![Figure](x11.jpg)

*(b) Target on slope*

![Figure](x12.jpg)

*(c) Target on step*

Figure 9: Results of the non-flat terrain extension.

### 6.3 Identifying and solving conflicting constraints

An advantage of our formulation is its ability to automatically detect and identify conflicting semantic objectives during the optimization process. This identification can be coupled with an interactive adjustment of the weights assigned to the associated losses, thus offering an efficient authoring tool to handle complex retargeting scenarios.
Let us consider the loss component as a scalar function $\mathcal{L}_{k}(\mathbf{q_{b}})$, where $k$ is the type of loss (among dist, dir, pen, height, and sliding), and $b$ is a body part. The cosine similarity

$$S_{\text{cosine}}\big{(}\nabla\mathcal{L}_{k_{1}}(\mathbf{q}_{b}),\nabla% \mathcal{L}_{k_{2}}(\mathbf{q}_{b})\big{)}$$ \tag{3}

is an indicator of the conflict between the loss $k_{1}$ and $k_{2}$ for the body part $b$. More precisely, a negative value close to -1 indicates strong conflicting goals between two losses.
Figure [10](#S6.F10) shows an example such use case where a conflict is identified between $\mathcal{L}_{\text{pen}}$ and $\mathcal{L}_{\text{dist}}$ on the torso, as the character is unable to reach its feet because of its large belly and short limbs.
The user can then use an interactive cursor to adapt a weight between the two losses, so as to find the most pertinent output for their use case. In one case (Figure [10](#S6.F10)), collision is neglected to account for more accurate contacts; in the other (Figure [10](#S6.F10)), the output is free of collisions, but contacts are lost.

![Figure](x13.jpg)

*(d) Source pose*

![Figure](x14.jpg)

*(e) Conflict b/w $\mathcal{L}_{\text{pen}}$ and $\mathcal{L}_{\text{dist}}$*

![Figure](x15.jpg)

*(f) User gives priority to $\mathcal{L}_{\text{dist}}$*

![Figure](x16.jpg)

*(g) User gives priority to $\mathcal{L}_{\text{pen}}$*

Figure 10: Example of conflicting losses and conflict resolution

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_reference.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_phil.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_sprite.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_rex.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_rain.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Backflip/Backflip_65_gabby.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_reference.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_phil.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_sprite.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_rex.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_rain.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/Kneeling/Kneeling_53_gabby.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_reference.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_phil.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_sprite.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_rex.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_rain.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/PikeWalk/PikeWalk_80_gabby.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_reference.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_phil.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_sprite.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_rex.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_rain.jpg)

![Figure](extracted/6238052/contents/images/blender_studio/rendus/ArmStretching/ArmStretching_51_gabby.jpg)

Figure 11: Results of retargeting on characters from Blender Studio. Source poses on the left, followed by the results on several characters (from left to right : Phil, Sprite, Rex, Rain, Gabby)

## 7 Conclusion and future work

We presented ReConForM, a novel motion retargeting method for characters of arbitrary shape and skeleton structure, focusing on self-collisions and ground contacts.
Our solution efficiently encodes shape and motion by creating time-varying pose descriptors based on the trajectory of selected key vertices of the character’s meshes. Thanks to a proximity-based criterion, we designed a way to select and weight the descriptors that carry the most importance in each frame. By optimizing the target poses to conform as well as possible to the source’s motion descriptions, we are able to retarget motion in real time.
Compared to state-of-the-art methods, our algorithm achieves more accurate contacts with the ground, smoother motion, and effectively avoids self-collisions.

Our method’s simplicity and flexibility allow easy adaptation to new use cases. We presented three extensions: accounting for inter-character contacts and gaze alignment to achieve multi-character retargeting; adapting the height of the ground for retargeting motion on non-flat terrains; and identifying conflicting constraints, allowing for interactive feedback and authoring, as prioritizing one of the constraints is often more desirable than getting high errors for them all.

### 7.1 Limitations and future work

Our method, while achieving real-time results,
has a few limitations, which we would like to address in future work.
Firstly, it does not account for how a character’s morphology (e.g., weight) affects its motion style (e.g., inertia).
Addressing this may request additional modifications beyond pure pose-based analysis, paving the way to include dynamic speed-based or
acceleration-based descriptors.
Secondly, while our method can handle different skeleton topologies
and large geometrical variations from source to target, it cannot transfer motions between drastically different articulated structures that lack a common template, e.g. a target character with additional limbs.
To handle this case, additional work
would be required to
extend the method to non-bijective mapping between key-vertices.
In addition, our method would highly benefit from an adaptive set of key-vertices, generated in collision areas of the source motion, and automatically transferred to the target character. This could enable the use of an even sparser set, but of more precise of motion descriptors over time.
Lastly, combining our semantic losses with deep learning methods could accelerate inference, while reinforcement learning methods could be used to ensure physically plausible motions.
The latter may be particularly useful when retargeting to non-flat grounds, where equilibrium might be lost.
Finally, since assessing the quality of retargeting strongly benefits from human
feedback, human-in-the-loop reinforcement learning might give new insights on how to move beyond manually-designed motion descriptors.

![Figure](extracted/6238052/contents/images/results/Phone_Call/source.jpg)

*(a) Source*

![Figure](extracted/6238052/contents/images/results/Phone_Call/copy_rotations.jpg)

*(b) Copy Rotations*

![Figure](extracted/6238052/contents/images/results/Phone_Call/r2et.jpg)

*(c) R2ET*

![Figure](extracted/6238052/contents/images/results/Phone_Call/ours.jpg)

*(d) Ours*

![Figure](extracted/6238052/contents/images/results/Shoulder_Rubbing/source.jpg)

*(e) Source*

![Figure](extracted/6238052/contents/images/results/Shoulder_Rubbing/copy_rotations.jpg)

*(f) Copy Rotations*

![Figure](extracted/6238052/contents/images/results/Shoulder_Rubbing/r2et.jpg)

*(g) R2ET*

![Figure](extracted/6238052/contents/images/results/Shoulder_Rubbing/ours.jpg)

*(h) Ours*

![Figure](extracted/6238052/contents/images/results/Pike_Walk/source.jpg)

*(i) Source*

![Figure](extracted/6238052/contents/images/results/Pike_Walk/copy_rotations.jpg)

*(j) Copy Rotations*

![Figure](extracted/6238052/contents/images/results/Pike_Walk/r2et.jpg)

*(k) R2ET*

![Figure](extracted/6238052/contents/images/results/Pike_Walk/ours.jpg)

*(l) Ours*

![Figure](extracted/6238052/contents/images/results/Angry/source.jpg)

*(m) Source*

![Figure](extracted/6238052/contents/images/results/Angry/copy_rotations.jpg)

*(n) Copy Rotations*

![Figure](extracted/6238052/contents/images/results/Angry/r2et.jpg)

*(o) R2ET*

![Figure](extracted/6238052/contents/images/results/Angry/ours.jpg)

*(p) Ours*

![Figure](extracted/6238052/contents/images/results/Gangnam_Style/source.jpg)

*(q) Source*

![Figure](extracted/6238052/contents/images/results/Gangnam_Style/copy_rotations.jpg)

*(r) Copy Rotations*

![Figure](extracted/6238052/contents/images/results/Gangnam_Style/r2et.jpg)

*(s) R2ET*

![Figure](extracted/6238052/contents/images/results/Gangnam_Style/ours.jpg)

*(t) Ours*

Figure 13: Results of our method on several animations taken from our validation dataset. Animation references from top to bottom : Phone Call, Shoulder Rubbing, Pike Walk, Angry, Gangnam Style. From left to right : Source, Copy Rotations, R2ET [r2et], Ours.

\printbibliography

Generated on Fri Feb 28 16:09:10 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)