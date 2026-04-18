[\addbibresource ref.bib # Skinned Motion Retargeting with Dense Geometric Interaction Perception Zijie Ye1,2, Jia-Wei Liu3, Jia Jia1,2, Shikun Sun1,2, Mike Zheng Shou3 1 Department of Computer Science and Technology, BNRist, Tsinghua University 2 Key Laboratory of Pervasive Computing, Ministry of Education 3 Show Lab, National University of Singapore Work is partially done during visiting NUS.Corresponding Author. ###### Abstract Capturing and maintaining geometric interactions among different body parts is crucial for successful motion retargeting in skinned characters. Existing approaches often overlook body geometries or add a geometry correction stage after skeletal motion retargeting. This results in conflicts between skeleton interaction and geometry correction, leading to issues such as jittery, interpenetration, and contact mismatches. To address these challenges, we introduce a new retargeting framework, MeshRet, which directly models the dense geometric interactions in motion retargeting. Initially, we establish dense mesh correspondences between characters using semantically consistent sensors (SCS), effective across diverse mesh topologies. Subsequently, we develop a novel spatio-temporal representation called the dense mesh interaction (DMI) field. This field, a collection of interacting SCS feature vectors, skillfully captures both contact and non-contact interactions between body geometries. By aligning the DMI field during retargeting, MeshRet not only preserves motion semantics but also prevents self-interpenetration and ensures contact preservation. Extensive experiments on the public Mixamo dataset and our newly-collected ScanRet dataset demonstrate that MeshRet achieves state-of-the-art performance. Code available at https://github.com/abcyzj/MeshRet](https://github.com/abcyzj/MeshRet).

## 1 Introduction

Skinned character animation is prevalent in virtual reality [lin2022digital], game development [mourot2022survey], and various other fields. However, animating these characters often presents significant challenges due to differences in body proportions between the motion source and the target character, leading to issues such as loss of motion semantics, mesh interpenetration, and contact mismatches. Consequently, motion retargeting is essential to adjust for these discrepancies in body proportions. This process is crucial for maintaining the integrity of the source motion’s characteristics in the animation of the target character.

Motion retargeting presents challenges due to the complex interactions among character limbs and the wide range of body geometries. Accurately preserving these interactions is crucial, as incorrect interactions can result in mesh interpenetration and contact mismatches.
Prior research has typically addressed these interactions from two perspectives: skeleton interactions and geometry corrections. Early methods [DBLP:journals/tog/AbermanLLSCC20, villegas2018neural, lim2019pmnet] employ cycle-consistency to implicitly align skeleton interaction semantics, yet they do not address the complexities of geometric interactions between different body parts. \textcitevillegas2021contact introduced mesh self-contact modeling; however, their approach does not extend to non-contact interactions. More recently, \textcitezhang2023skinned implemented a two-stage pipeline that first aligns skeleton interaction semantics and then corrects geometric artifacts. Nonetheless, the inherent conflict between preserving skeleton interaction semantics and correcting geometry leads to jittery movements, severe interpenetration and imprecise contacts. \textcitezhang2023semantics subsequently proposed adding a stage that aligns visual semantics with a visual language model, but this requires detailed pair-by-pair finetuning due to the loss of spatial information when projecting 3D motion into 2D images.

To resolve the conflict between skeleton interaction and geometry correction, we propose a new approach: focusing solely on dense geometric interaction for motion retargeting. Character animation videos, rendered from the skinned mesh, rely on geometric interactions to shape user perception. Skeleton interaction, in contrast, merely represents a simplified, sparse form of geometric interaction. Therefore, maintaining correct interactions between different body part geometries not only preserves motion semantics but also prevents mesh interpenetration and ensures contact preservation, as illustrated in Figure [1](#S2.F1).

Given the significance of geometric interactions, we propose a new framework, named MeshRet, for skinned motion retargeting. In contrast to earlier methods that adjust skeletal motion retargeting outcomes, our approach models the intricate interactions among character meshes without depending on predefined vertex correspondences.

The design of MeshRet necessitates several technical innovations. Initially, there is a requirement for dense mesh correspondence across different characters. Drawing inspiration from the medial axis inverse transform (MAIT) [nass2007medial], we have devised a technique, termed semantically consistent sensors (SCS), to automatically derive dense mesh correspondence from sparse skeleton correspondence. This technique enables us to sample a point cloud of sensors on the mesh to represent each character. Following this, to illustrate dense mesh interaction between body parts, we employ interacting mesh sensor pairs, maintaining generality. These pair-wise interactions are encoded within a novel spatial-temporal representation termed the Dense Mesh Interaction (DMI) field. The DMI field adeptly encapsulates both contact and non-contact interaction semantics. Finally, we proceed to learn a motion manifold that aligns with the target character geometry and the source motion DMI field.

To align our evaluation process more closely with real animation production, we gathered an in-the-wild motion dataset, termed ScanRet, characterized by abundant contact semantics and minimal mesh interpenetration. ScanRet consists of 100 human actors ranging from bulky to skinny, each performing 83 motion clips scrutinized by human animators. The MeshRet model is trained on both the ScanRet dataset and the widely used Mixamo [mixamo] dataset. We assessed our method across a large variety of motions and a diverse array of target characters. Both qualitative and quantitative analyses show that our MeshRet model significantly outperforms existing methods.

To summarize, we present the following contributions:

-
•

We introduce MeshRet, a pioneering solution that facilitates geometric interaction-aware motion retargeting across varied mesh topologies in a single pass.

-
•

We present the SCS and the novel DMI field to guide the training of MeshRet, effectively encapsulating both contact and non-contact interaction semantics.

-
•

We develop ScanRet, a novel dataset specifically tailored for assessing motion retargeting technologies, which includes detailed contact semantics and ensures smooth mesh interaction.

-
•

Our experiments demonstrate that MeshRet delivers exceptional performance, marked by accurate contact preservation and high-quality motion.

## 2 Related Work

#### Skeletal motion retargeting

Motion retargeting seeks to preserve the characteristics of source motions when transferring them to a different target character. Skeletal motion retargeting primarily addresses the challenge of differing bone ratios. \textcitegleicher1998retargetting initially formulated motion retargeting as a spatio-temporal optimization problem, using source motion features as kinematic constraints. Subsequent researches [bernardin2017normalized, feng2012automating, lee1999hierarchical] have focused on optimization-based approaches with various constraints. However, these methods, while requiring extensive optimization, often yield suboptimal results. Consequently, recent studies have explored learning-based motion retargeting algorithms. \textcitejang2018variational trained a motion retargeting network using a U-Net [ronneberger2015u] architecture on paired motion data. \textcitevillegas2018neural introduced a recurrent neural network combined with cycle-consistency [zhu2017unpaired] for unsupervised motion retargeting. \textcitelim2019pmnet propose to learn frame-by-frame poses and overall movements separately. \textciteDBLP:journals/tog/AbermanLLSCC20 develop differentiable operators for cross-structural motion retargeting among homeomorphic skeletons. However, these methods generally neglect the geometry of characters, leading to frequent contact mismatches and severe mesh interpenetrations.

*Figure 1: Comparison with the existing method. Contrary to the earlier retargeting-correction approach [zhang2023skinned], which suffer from internal contradictions leading to interpenetration, jitter, and contact mismatches, our pipeline leverages the DMI field to accurately model complex geometric interactions.*

#### Geometry-aware motion retargeting

Previous studies have generally processed character geometries through two approaches: contact preservation and interpenetration avoidance. \textcitelyard2008motion developed a heuristic optimization algorithm to maintain character self-contact, while \textciteho2010spatial proposed to maintain character interactions by minimizing the deformation of interaction meshes. \textciteho2013motion introduced a spatio-temporal optimization framework to prevent self-collisions in robot motion retargeting. \textcitejin2018aura employed a proxy volumetric mesh to preserve spatial relationships during retargeting. Subsequently, \textcitebasset2020contact combined both attraction and repulsion terms in an optimization-based method to avoid interpenetration and preserve contact. However, these methods necessitate per-vertex correspondence and involve costly optimization processes. More recently, \textcitevillegas2021contact attempted to retarget skinned motion through optimization in a latent space of a pretrained network, although their method does not accommodate non-contact interactions. \textcitezhang2023skinned implemented a two-stage pipeline that initially aligns skeleton interaction semantics and subsequently corrects geometric artifacts. Nevertheless, the inherent conflict between maintaining skeleton interaction semantics and correcting geometry often results in jittery movements and imprecise contacts. In a later study, \textcitezhang2023semantics added a stage that aligns visual semantics using a visual language model, but this approach requires extensive pair-by-pair fine-tuning due to the loss of spatial information when projecting 3D motion into 2D images.

Existing geometry-aware motion retargeting methods either require expensive optimization or employ multi-stage strategies for skeleton and geometry semantics, resulting in a contradiction between stages that often leads to unsatisfactory results. In contrast, our method processes both contact and non-contact semantics using a dense mesh interaction field in a single stage.

## 3 Method

![Figure](x2.png)

*Figure 2: Overview of the proposed MeshRet. The pipeline begins with the extraction of the DMI field using sensor forward kinematics, denoted as $\mathcal{F}_{k}$, and pairwise interaction feature selection, represented by $\mathcal{F}_{c}$. This DMI field, in conjunction with geometric features derived from $\mathcal{F}_{g}$, is fed into an encoder-decoder network. The network predicts the target motion sequence, which is aligned with the target character’s geometry and the original DMI field.*

### 3.1 Overview

We introduce a novel geometric interaction-aware motion retargeting framework MeshRet, as illustrated in Figure [2](#S3.F2). Unlike previous methods that either overlook character geometries [DBLP:journals/tog/AbermanLLSCC20, villegas2018neural, lim2019pmnet] or apply geometry correction after skeleton retargeting [zhang2023skinned, zhang2023semantics], our framework directly addresses dense geometric interactions with the Dense Mesh Interaction (DMI) field. This provides a detailed representation of the interactions within skinned character motions, preserving motion semantics by preventing mesh interpenetration and ensuring precise contact preservation.

#### Motion & geometry representations

Assume the motion sequence has $T$ frames and the character has $N$ skeletal joints. The motion sequence $\mathbf{m}$ is represented by the global root translation $\mathbf{X}\in\mathbb{R}^{T\times 3}$ and the local joint rotation $\mathbf{Q}\in\mathbb{R}^{T\times N\times 6}$, where we adopt the 6D representation [zhou2019continuity] for the joint rotations. The rest-pose geometry $\mathbf{G}$ of the character is represented by the rest-pose mesh $\mathbf{O}$ and the rest-pose joint locations $\mathbf{J}\in\mathbb{R}^{N\times 3}$.

#### Task definition

Given the source motion sequence $\mathbf{m}_{\text{A}}$, and the geometries $\mathbf{G}_{\text{A}}$ and $\mathbf{G}_{\text{B}}$ of the source and target characters in their T-poses, our objective is to generate the motion $\mathbf{m}_{\text{B}}$ for the target character. This process aims to retain essential aspects of the source motion, including its semantics, contact preservation, and the avoidance of interpenetration.

Following the definition of the task, our MeshRet model initially derives Semantically Consistent Sensors (SCS) $\mathbf{S}\in\mathbb{R}^{S\times 4\times 3}$, which provide dense geometric correspondences essential for the retargeting process, where $\mathbf{S}=\mathcal{F}_{\text{s}}(\mathbf{G})$. $\mathbf{S}$ captures the sensor location and the sensor tangent space matrix, facilitating an enhanced perception of the geometry surface. Subsequently, we conduct sensor forward kinematics (FK) and pairwise interaction extraction to generate the source DMI field $\mathbf{D}_{\text{A}}=\mathcal{F}_{\text{d}}(\mathbf{m}_{\text{A}},\mathbf{S}_%
{\text{A}})$, where $\mathbf{D}_{\text{A}}\in\mathbb{R}^{T\times K\times L\times P}$. Here, $K$ is the number of SCS in the DMI field, $L$ represents a hyper-parameter of feature selection, and $P$ indicates the feature dimension of the DMI. Lastly, a transformer-based network [vaswani2017attention] ingests $\mathbf{m}_{\text{A}}$, $\mathbf{D}_{\text{A}}$, $\mathbf{S}_{\text{A}}$, and $\mathbf{S}_{\text{B}}$, and predicts a target motion sequence $\mathbf{m}_{\text{B}}$ that aligns with the target character’s geometry and the source DMI field. The entire pipeline is denoted as follows:

$$\mathbf{m}_{\text{B}}=\mathcal{F}_{\text{r}}(\mathbf{m}_{\text{A}},\mathbf{D}_% {\text{A}},\mathbf{S}_{\text{A}},\mathbf{S}_{\text{B}})$$ \tag{1}

### 3.2 Semantically consistent sensors

To facilitate dense geometric interactions, our MeshRet framework necessitates establishing dense mesh correspondence between source and target characters. Previous studies have typically derived correspondence from vertex coordinates [zhou2022toch], virtual sensor [zhang2021manipnet] or through a bounding mesh [jin2018aura]; however, these methods are confined to template meshes sharing identical topology, such as MANO [remero2017embodied] or SMPL [loper2015smpl]. Villegas et al. [villegas2021contact] suggested determining vertex correspondence using nearest neighbor searches on predefined feature vectors. Nevertheless, this approach often lacks precision and brevity, resulting in inaccurate contact representations and substantial optimization burdens.

In this study, we introduce Semantically Consistent Sensors (SCS) that are effective across various mesh topologies while ensuring precise semantic correspondence. Our approach draws inspiration from the Medial Axis Inverse Transform (MAIT) [nass2007medial]. We conceptualize the skeleton bones of each character as approximate medial axes of their limbs and torso. For each bone, a MAIT-like transform is applied to generate the corresponding SCS. This involves casting rays from the bone axis across a plane perpendicular to it. The origin parameter $l$ and direction parameter $\phi$ of the rays, combined with the bone index $b$, establish the semantic coordinates of the SCS. The semantic coordinates describe connection between the sensor and the skeleton bones. A sensor is deemed valid if its ray intersects the mesh linked to the bone; otherwise, it is considered invalid. Through this method, we establish a dense geometric correspondence based on sparse skeletal correspondence. The procedure for deriving SCS is illustrated in Figure [3](#S3.F3). Given a unified set of SCS semantic coordinates $\{(b_{1},l_{1},\phi_{1}),(b_{2},l_{2},\phi_{2}),\cdots,(b_{S},l_{S},\phi_{S})\}$, we can derive SCS feature $\mathbf{S}=\{\mathbf{s}_{1},\mathbf{s}_{2},\cdots,\mathbf{s}_{S}\}$ for each character. Further details can be found in Algorithm [1](#algorithm1).

![Figure](x3.png)

*Figure 3: Left: Illustration of the method to derive a sensor feature $\mathbf{s}$ from the semantic coordinate $(b,l,\phi)$ across different characters. The red line represents the projected ray. The feature $\mathbf{s}$ encompasses the sensor’s location and its tangent space matrix. Right: The DMI field effectively captures both contact and non-contact interactions. Red lines represent $\mathbf{d}^{t,i,j}$ in the DMI field. In the second example, the body sensors (yellow points) are located in the tangent plane of the hand sensors (blue points), signifying a contact interaction.*

### 3.3 Dense mesh interaction field

To effectively represent the interactions between character limbs and the torso, we have developed the DMI field. Based on SCS detailed in Section [3.2](#S3.SS2), the DMI field comprehensively captures both contact and non-contact interactions across different body part geometries. Utilizing the DMI field allows for dense geometry interaction-aware motion retargeting, thereby eliminating the need for a geometry correction stage.

#### Sensor forward kinematics

For a given motion sequence, denoted as $\mathbf{m}$, we initially conduct forward kinematics (FK) on $\mathbf{S}$ to derive sensor features $\mathbf{S}^{1:T}\in\mathbb{R}^{T\times S\times 4\times 3}$. Each $\mathbf{S}^{t}$ encompasses the locations and tangent matrices for $S$ sensors at frame $t$. The FK transformation for an individual sensor is expressed as:

$$\mathbf{s}_{i}^{t}=\sum_{n=1}^{N}\omega(\mathbf{p}_{i})_{n}G_{n}(\mathbf{Q}^{t% })\cdot\mathbf{s}_{i},$$ \tag{2}

where $G_{n}(\mathbf{Q}^{t})\in SE(3)$ is the global transformation matrix for bone
$n$, derived from its local rotation matrix, and $\omega(\mathbf{p}_{i})_{n}$ represents the linear blend skinning (LBS) weight for sensor $\mathbf{s}_{i}$, determined through barycentric interpolation of its adjacent mesh vertices.

#### Pairwise interaction feature

Next, we model the geometric interactions as pairwise interaction features between sensors. Ideally, for each frame, we obtain a comprehensive DMI field, $\overline{\textbf{D}}^{t}$, representing pairwise vectors across $K^{2}$ sensor pairs:

$$\mathbf{d}^{t,i,j}=\mathbf{t}_{i}^{-1}(\textbf{p}_{j}^{t}-\textbf{p}_{i}^{t}),$$ \tag{3}

$$\overline{\textbf{D}}^{t}=\{(\textbf{d}^{t,i,j},b_{i},b_{j},l_{i},l_{j},\phi_{% i},\phi_{j})\}_{i=1:S}^{j=1:S},$$ \tag{4}

where $\mathbf{t}_{i}\in\mathbb{R}^{3\times 3}$ is the tangent matrix if sensor $i$, and $\mathbf{d}^{t,i,j}$ represents the relative position of target sensor $j$ in the tangent space of observation sensor $i$. $\overline{\textbf{D}}^{t}$ is composed of two components: the relative position of the sensor pair and the semantic coordinates of both the observation and target sensors. The use of semantic rather than spatial coordinates is essential, as it obviates the need for actual sensor positions, thereby making DMI suitable for motion retargeting applications.

However, $\overline{\textbf{D}}^{t}\in\mathbb{R}^{S\times S\times P}$ exhibits quadratic growth with respect to $S$ because it includes $S^{2}$ sensor pairs, rendering it impractical when managing thousands of sensors. To address this, we implement two sparsification strategies for $\overline{\textbf{D}}^{t}$. Initially, we restrict interactions to critical body parts only, such as arm-torso, arm-head, arm-arm, and leg-leg, rather than between all sensor pairs, thereby restricting our focus to $K$ observation sensors. Subsequently, for each observation sensor, we select $L$ target sensors from each relevant body part, where $L$ is a predetermined hyper-parameter. Specifically, we empirically choose $L/2$ nearest and $L/2$ furthest target sensors. We find that proximate sensor pairs are crucial for minimizing interpenetration and maintaining contact, while distant pairs delineate the overall spatial relationships between body parts, as shown in Figure [3](#S3.F3). These strategies lead to the formulation of the final DMI field $\mathbf{D}\in\mathbb{R}^{K\times L\times P}$, with selected sensor pairs indicated by the sparse DMI mask $\mathbf{M}_{\text{src}}\in\mathbb{R}^{S\times S}$ shown in Figure [2](#S3.F2).

### 3.4 Geometry interaction-aware motion retargeting

To avoid the conflict between skeleton interaction and geometric correction, the proposed MeshRet employs the DMI field to model geometric interactions directly. As shown in Figure [2](#S3.F2), MeshRet initially extracts the DMI field $\mathbf{D}_{\text{A}}$ from the source motion sequence $\mathbf{m}_{\text{A}}$, as described in Section [3.3](#S3.SS3). The field $\mathbf{D}_{\text{A}}$ encapsulates interactions among various body parts within the source motion, encompassing both contact and non-contact interactions, further depicted in Figure [3](#S3.F3). The DMI field, composed of sensor pair feature vectors, possesses the unordered characteristics of a point cloud. Consequently, we implement a PointNet-like architecture [qi2017pointnet] for our DMI encoder, which is divided into two components: the per-sensor encoder and the per-frame encoder. Given $\mathbf{D}_{\text{A}}\in\mathbb{R}^{T\times K\times L\times P}$, the per-sensor encoder initially processes it as $T*K$ separate point clouds, producing representations $\mathbf{H}_{\text{A}}^{\text{s}}\in\mathbb{R}^{T\times K\times D_{\text{model}}}$ for each observation sensor, where $D_{\text{model}}$ denotes the feature dimension. Subsequently, the per-frame encoder generates per-frame representations $\mathbf{H}_{\text{A}}^{\text{f}}\in\mathbb{R}^{T\times D_{\text{model}}}$ by encoding these $T$ point clouds.

Since DMI field $\mathbf{D}_{\text{A}}$ lacks geometric information about characters, we introduced a geometry encoder $\mathcal{F}_{g}$ to extract geometric features from their SCS. For each sensor, we form a feature vector by concatenating its rest-pose feature $\mathbf{s}_{i}$ with its semantic coordinates $(b_{i},l_{i},\phi_{i})$. The resultant geometric features are represented as $\mathbf{C}_{\text{A}}\in\mathbb{R}^{S_{\text{A}}\times C}$ for character A and $\mathbf{C}_{\text{B}}\in\mathbb{R}^{S_{\text{B}}\times C}$ for character B. The semantic coordinates of sensors act as intermediaries linking the DMI field to character geometry. The geometry encoder employs a PointNet-like architecture [qi2017pointnet] to transform the geometric features $\mathbf{C}$ into a geometric latent code $\mathbf{H}^{\text{g}}\in\mathbb{R}^{D_{\text{model}}}$.

The transformer-based retargeting network processes input features including the source DMI feature $\mathbf{H}_{\text{A}}^{\text{f}}$, source joint rotation $\mathbf{Q}_{\text{A}}$, source geometry latent $\mathbf{H}_{\text{A}}^{\text{g}}$, and target geometry latent $\mathbf{H}_{\text{B}}^{\text{g}}$. Specifically, the encoder processes $\mathbf{H}_{\text{A}}^{\text{f}}$ and $\mathbf{H}_{\text{B}}^{\text{g}}$, while the decoder processes $\mathbf{Q}_{\text{A}}$ and $\mathbf{H}_{\text{A}}^{\text{g}}$. The latents $\mathbf{H}_{\text{A}}^{\text{g}}$ and $\mathbf{H}_{\text{B}}^{\text{g}}$ serve as the initial tokens in the sequence, enabling both the encoder and decoder to operate over a sequence of length $T+1$. The output sequence’s final $T$ frames are represented as $\hat{\mathbf{Q}}_{\text{B}}$.

Due to the lack of paired ground-truth data, we employ the unsupervised method described by \textcitelim2019pmnet. Our network utilizes four loss functions for training: reconstruction loss, DMI consistency loss, adversarial loss, and end-effector loss. Supervision signals are derived from the source motion. We maintain geometric interactions by aligning the source DMI field $\mathbf{D}_{\text{A}}$ with the target DMI field $\hat{\mathbf{D}}_{\text{B}}$. The target DMI field $\hat{\mathbf{D}}_{\text{B}}$ is generated by first applying sensor forward kinematics to $\hat{\mathbf{Q}}_{\text{B}}$, followed by selecting sensor pairs using the target sparse DMI mask $\mathbf{M}_{\text{tgt}}\in\mathbb{R}^{S\times S}$. This mask, $\mathbf{M}_{\text{tgt}}$, is derived by excluding invalid sensors of the target character from $\mathbf{M}_{\text{src}}$. The DMI consistency loss is quantified as the cosine similarity loss between pair-wise relative positions in $\hat{\mathbf{D}}_{\text{B}}$ and $\mathbf{D}_{\text{A}}$:

$$\mathcal{L}_{\text{dmi}}=-\frac{1}{T}\sum_{t=1}^{T}\sum_{k=1}^{K}\sum_{l=1}^{L% }c(k,l)\frac{\mathbf{d}_{\text{A}}^{t,k,l}\cdot\hat{\mathbf{d}}_{\text{B}}^{t,% k,l}}{||\mathbf{d}_{\text{A}}^{t,k,l}||_{2}\cdot||\hat{\mathbf{d}}_{\text{B}}^% {t,k,l}||_{2}},$$ \tag{5}

where $c(k,l)$ takes the value $1$ if sensor pair $(k,l)$ is valid in both $\mathbf{M}_{\text{src}}$ and $\mathbf{M}_{\text{tgt}}$, and 00 otherwise.
The reconstruction loss serves as a regularization mechanism to minimize motion alterations during retargeting, defined as follows:

$$\mathcal{L}_{\text{rec}}=||\hat{\mathbf{Q}}_{\text{B}}-\mathbf{Q}_{\text{A}}||% _{2}^{2}.$$ \tag{6}

To facilitate realistic motion retargeting, a discriminator, denoted as $\delta(\cdot)$, is employed. The adversarial loss is subsequently defined as:

$$\mathcal{L}_{\text{adv}}=\mathbb{E}_{\mathbf{Q}\sim p_{\text{real}}}[\log% \delta(\mathbf{Q})]+\mathbb{E}_{\mathbf{Q}\sim p(\hat{\mathbf{Q}}_{\text{B}})}% [\log(1-\delta(\mathbf{Q}))].$$ \tag{7}

We observed that the global orientation of end-effectors significantly influences user experience. Consequently, we introduced an end-effector loss to promote consistent orientations of end-effectors in the retargeted motion.

$$\mathcal{L}_{\text{ef}}=\frac{1}{T|\mathcal{X}|}\sum_{t=1}^{T}\sum_{i\in% \mathcal{X}}||R(\mathbf{Q}_{\text{A}}^{t},i)-R(\hat{\mathbf{Q}}_{\text{B}}^{t}% ,i)||,$$ \tag{8}

where $R(\cdot)$ transforms local joint rotations into global rotations for joint $i$ along the kinematic chain and $\mathcal{X}$ represents the set of end-effectors. Our MeshRet is trained by:

$$\mathcal{L}_{\text{total}}=\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+% \lambda_{\text{dmi}}\mathcal{L}_{\text{dmi}}+\lambda_{\text{adv}}\mathcal{L}_{% \text{adv}}+\lambda_{\text{ef}}\mathcal{L}_{\text{ef}}.$$ \tag{9}

## 4 Experiments

### 4.1 Settings

#### Datasets

We trained and evaluated our method using the Mixamo dataset [mixamo] and the newly curated ScanRet dataset. We downloaded 3,675 motion clips performed by 13 cartoon characters from the Mixamo dataset contains, while the ScanRet dataset consists of 8,298 clips executed by 100 human actors. Notably, the Mixamo dataset frequently features corrupted data due to interpenetration and contact mismatches. To overcome these issues, we created the ScanRet dataset, which provides detailed contact semantics and improved mesh interactions, with each clip being scrutinized by human animators. The training set comprises 90% of the motion clips from both datasets, involving nine characters from Mixamo and 90 from ScanRet. Our experiments tested the motion retargeting capabilities between cartoon characters and real humans, aligning closely with typical retargeting workflows. During inference, we adopted four data splits based on character and motion visibility: unseen character with unseen motion (UC+UM), unseen character with seen motion (UC+SM), seen character with unseen motion (SC+UM), and seen character with seen motion (SC+SM), as delineated by \textcitezhang2023skinned. We present the average results across these splits. Additional details available in Appendix [A](#A1).

#### Implementation details

The hyper-parameters $\lambda_{\text{rec}}$, $\lambda_{\text{dmi}}$, $\lambda_{\text{adv}}$, $\lambda_{\text{ef}}$, and $L$ were empirically set to 1.0, 5.0, 1.0, 1.0, and 20, respectively. We use $\{0,1,\cdots,N_{\text{body}}-1\}\times\{0,0.25,0.5,0.75\}\times\{0,0.5\pi,\pi,%
1.5\pi\}$ as the SCS semantic coordinates set, where $N_{\text{body}}=18$ is the number of body bones and $\times$ represents the Cartesian product. We employed the Adam optimizer [kingma2015adam] with a learning rate of $10^{-4}$ to optimize our network. The training process required 36 epochs. For further details, please refer to Appendix [C](#A3).

#### Evaluation metrics

We assess the effectiveness of our method through three metrics: joint accuracy, contact preservation, and geometric interpenetration. Joint accuracy is quantified by calculating the Mean Squared Error (MSE) between the retargeted joint positions and the ground-truth data provided by animators in ScanRet. This analysis considers both global and local joint positions, normalized by the character heights. Contact preservation is evaluated by measuring the Contact Error, defined as the mean squared distance between sensors that were originally in contact in the source motion clip. Geometric interpenetration is determined by the ratio of penetrated limb vertices to the total limb vertices per frame. Further details are available in Appendix [B](#A2).

![Figure](x4.png)

*Figure 4: Qualitative comparison with baseline methods. Our method ensures precise contact preservation and minimal geometric interpenetration.*

### 4.2 Comparison with state-of-the-arts

#### Qualitative results

Figure [4](#S4.F4) demonstrates the performance of skinned motion retargeting across characters with diverse body shapes, where the motion sequences are novel to the target characters during training. Most baseline methods, except R2ET [zhang2023skinned], fail to consider the geometry of characters, leading to significant geometric interpenetration and contact mismatches. Unlike these methods, R2ET [zhang2023skinned] includes a geometry correction phase after skeleton-aware retargeting. However, this creates a conflict between the two stages, resulting in oscillations in R2ET’s outcomes, which manifest as alternating contact misses and severe interpenetrations, as shown in the first two rows. Additionally, these oscillations appear variably across different frames within the same motion clip, producing jittery motion, as illustrated in Figure [1](#S2.F1) and Figure [8](#A5.F8). A further limitation of R2ET is its neglect of hand contacts. In contrast, our method employs the innovative DMI field to preserve such detailed interactions, such as those observed in the “Praying” pose in the third row.

*Table 1: Quantitative comparison between our method and state-of-the-arts. Mixamo+ represents the mixed dataset of Mixamo and ScanRet. MSElc denotes the local MSE.*

#### Quantitative results

Table [1](#S4.T1) presents a comparison between our methods and state-of-the-arts. We initially measure the joint location error using MSE and MSElc on ScanRet. The ground truth in ScanNet is established by human animators. Our observations indicate that human animators typically retarget motions by initially replicating joint rotations and subsequently modifying frames that display incorrect interactions. Conversely, our method modifies the entire motion sequence, resulting in a higher MSE compared to the Copy strategy. Nevertheless, MSE remains a valuable auxiliary reference. In comparison to PMnet [lim2019pmnet], R2ET [zhang2023skinned], and SAN [DBLP:journals/tog/AbermanLLSCC20], our method achieves MSE reductions of 65%, 29%, and 8%, respectively. These results demonstrate that our approach more closely aligns with the outputs produced by human animators.

As shown in Table [1](#S4.T1), PMnet [lim2019pmnet] and SAN [DBLP:journals/tog/AbermanLLSCC20], exhibit high interpenetration ratios and contact errors due to their neglect of character geometries. R2ET [zhang2023skinned] effectively reduces interpenetration through a geometry correction stage; nonetheless, it still encounters high contact errors stemming from conflicts between the retargeting and correction stages. Our approach explicitly models geometry interactions and thereby achieves low contact error and penetration ratio, illustrating the effectiveness of our proposed MeshRet in generating high-quality retargeted motions with detailed contact semantics and smooth mesh interactions. Additionally, we observe that retargeting using the mixed Mixamo+ dataset is more challenging than with the ScanRet dataset, attributable to significant body shape variations between cartoon characters and real person characters.

![Figure](x5.png)

*Figure 5: Qualitative comparison of ablation studies. A red circle highlights areas of interpenetration, while a red rectangle identifies errors in non-contact semantics.*

### 4.3 Ablation Studies

We conducted ablation studies to demonstrate the significance of pairwise interaction feature selection and the implementation of DMI similarity loss. Initially, we evaluated the performance of a model trained exclusively with the nearest $L$ sensor pairs, denoted as Ourscls, and another model trained solely with the farthest $L$ sensor pairs, referred to as Oursfar. As indicated in Table [1](#S4.T1) and Figure [5](#S4.F5), Oursfar compromises contact semantics and leads to significant interpenetration, while Ourscls also exhibits inferior performance. This outcome suggests that proximal sensor pairs are essential for minimizing interpenetration and preserving contact, whereas distal pairs provide insights into the non-contact spatial relationships among body parts. Further, we investigated the effect of incorporating a distance matrix loss, as proposed by \textcitezhang2023skinned, on our sensor pairs, designated as Oursdm. The results imply that the distance matrix loss fails to yield meaningful supervisory signals, likely because distance is non-directional and insufficient to discern the relative spatial positions among numerous sensors.

*Table 2: Human preferences between our method and baselines.*

### 4.4 User study

We conducted a user study to assess the performance of our MeshRet model in comparison with the Copy strategy, PMnet [lim2019pmnet], SAN [DBLP:journals/tog/AbermanLLSCC20], and R2ET [zhang2023skinned]. Fifteen sets of motion videos were presented to participants, each consisting of one source skinned motion and five anonymized skinned results. Participants were requested to rate their preferences based on three criteria: semantic preservation, contact accuracy, and overall quality. Users were recruited from Amazon Mechanical Turk [amt], resulting in a total of 600 comparative evaluations. As indicated in Table [2](#S4.T2), approximately 81% of the comparisons favored our results. Details can be found in Appendix [D](#A4)

## 5 Conclusion

We introduce a novel framework for geometric interaction-aware motion retargeting, named MeshRet. This framework explicitly models the dense geometric interactions among various body parts by first establishing a dense mesh correspondence between characters using semantically consistent sensors. We then develop a unique spatio-temporal representation, termed the DMI field, which adeptly captures both contact and non-contact interactions between body geometries. By aligning this DMI field, MeshRet achieves detailed contact preservation and seamless geometric interaction. Performance evaluations using the Mixamo dataset and our newly compiled ScanRet dataset confirm that MeshRet offers state-of-the-art results.

#### Limitations

The primary limitation of MeshRet is its dependence on inputs with clean contact; motion clips exhibiting severe interpenetration yield poor outcomes. Consequently, it is unable to process noisy inputs effectively. Refer to Figure [12](#A5.F12) and Figure [13](#A5.F13) for failure cases under noisy inputs. Future efforts will focus on enhancing its robustness to noisy data. Additionally, SCS extraction can be compromised by noisy meshes, particularly those with complex clothing. A potential solution is to employ a Laplacian-smoothed proxy mesh for SCS extraction. Lastly, the method cannot handle characters with missing limbs.

## Acknowledgments and Disclosure of Funding

This work is supported by the National Key R&D Program of China under Grant No. 2024QY1400, the National Natural Science Foundation of China No. 62425604, and the Tsinghua University Initiative Scientific Research Program. Mike Shou does not receive any funding for this work.

\printbibliography

## Appendix A Dataset Details

#### ScanRet details

The primary motivation for collecting the ScanRet dataset stemmed from two main concerns. First, the data quality in the Mixamo [mixamo] dataset was relatively low, suffering from significant issues such as interpenetration and contact mismatch. Second, the Mixamo dataset exclusively contained cartoon characters, whose body type distributions differed markedly from those of real human motion capture actors. In response, we developed the ScanRet dataset. We recruited 100 participants, evenly split between males and females, representing common ranges of height and BMI. Each participant underwent a 3D scan to create a T-pose mesh. We intentionally did not collect texture information for the body or face to protect privacy. Subsequently, we used motion capture equipment to build a library of 83 actions characterized by extensive physical contact. We enlisted human animators to map each action onto the 100 T-pose meshes, ensuring both semantic integrity and correct physical contact were maintained. All participants and animators received fair compensation. After discarding some invalid data, we compiled a total of 8,298 motion data entries. The ScanRet dataset is designed to simulate data obtained from real human motion capture, such as the MoSh [mahmood2019amass, loper2014mosh] algorithm, thus enhancing the realism of our evaluation process in the context of actual animation production workflows.

![Figure](x6.png)

*Figure 6: Left: Characters of varying body types in the Mixamo dataset do not always maintain reasonable hand contact during clapping actions. Right: In our ScanRet dataset, characters of diverse body types consistently maintain appropriate hand contact while performing the same clapping actions.*

#### Data splits

We collected motion data for 13 characters from the Mixamo website, totaling 3,675 motion sequences, with each character having approximately the same number of sequences. The characters are: Aj, Amy, Kaya, Mousey, Ortiz, Remy, Sporty Granny, Swat, The Boss, Timmy, X Bot, and Y Bot. Among them, Ortiz, Kaya, X Bot, and Amy were not encountered by the network during training. Overall, our training set included motion data for 9 Mixamo characters and 90 randomly selected characters from the ScanNet dataset, where 90% of the motion sequences was randomly chosen from both datasets. Details regarding the train/test split for specific motion sequences and characters are provided in the code.

## Appendix B Evaluation metric details

We evaluate the performance of our method from three perspectives: joint accuracy, contact preservation, and geometric interpenetration. In terms of joint accuracy, we calculate the Mean Squared Error (MSE) between the ground-truth joint positions $X_{gt}$ and the retargeted joint positions $\hat{X}$, normalized by the character’s height $h$:

$$MSE=\frac{1}{h}||X_{gt}-\hat{X}||_{2}^{2}$$ \tag{10}

Previous work [zhang2023skinned] assessing the accuracy of self-contact measurements merely utilized the distance between hand vertices and the body surface to determine contact presence. Such experimental metrics fail to accurately reflect the precision of the contact location. Therefore, we adopted a metric similar to the vertex contact mean squared error (MSE) proposed by \textcitevillegas2021contact, termed “Contact Error”. Specifically, we first identified sensor pairs where the distance between hand and body sensors in the source action was less than the arm’s diameter $d_{src}$. We then located the same sensor pairs in the retargeted motion. If the distance between these sensor pairs in the retargeted motion exceeded that in the source action, we calculated the MSE of the distance differences; otherwise, the contact error was zero. The formula is as follows:

$$\text{Contact Error}=\begin{cases}(||\frac{\mathbf{d}_{\text{A}}^{t,k,l}}{R_{% \text{A}}}||_{2}-||\frac{\hat{\mathbf{d}}_{\text{B}}^{t,k,l}}{R_{\text{B}}}||_% {2})^{2},&\quad\text{if}||\frac{\mathbf{d}_{\text{A}}^{t,k,l}}{R_{\text{A}}}||% _{2}>||\frac{\hat{\mathbf{d}}_{\text{B}}^{t,k,l}}{R_{\text{B}}}||_{2}\\ 0,\quad\text{otherwise},\end{cases}$$ \tag{11}

where $\mathbf{d}_{\text{A}}^{t,k,l}$ indicates the contact sensor pairs with $||\mathbf{d}_{\text{A}}^{t,k,l}||_{2}\textcitefan2022faceformer, which ensures that each frame feature in the decoder attends only to the corresponding DMI frame and initial token, thereby aligning the network’s output motion sequence with the input features.

#### Training details

We implemented our network using PyTorch [paszke2019pytorch], running on a machine equipped with an NVIDIA RTX A6000 GPU and an AMD EPYC 9654 CPU. The dataset was uniformly processed at a frame rate of 30 fps. During training, we randomly clipped a sequence of 30 frames from the dataset. The target character was set to be the same as the source character with a 50% probability, and different with a 50% probability, selected randomly from the dataset. On our system, training for 36 epochs required approximately 40 hours. During inference, our MeshRet model can achieve performance exceeding 30 fps.

![Figure](extracted/6135609/img/user_study_screenshot.png)

*Figure 7: User interface presented to participants during the user study.*

## Appendix D User study details

We recruited participants via the Amazon Mechanical Turk [amt] platform to partake in a user study. As shown in Figure [7](#A3.F7), during each session, subjects were presented with one source video and two retargeted motion videos: Video A and Video B. Participants were asked to watch all three videos and then compare Video A and Video B. At the conclusion of the viewing, they were requested to answer the following three questions:

-
1.

Which video better matches the source motion in terms of the overall meaning and intent of the motion?

-
2.

Which video has more accurate and detailed motion? Look for less self-interpenetration and better self-contact precision.

-
3.

Considering all factors, which video do you think is better overall?

For each question answered, participants received a compensation of $0.04. We collected 600 comparison results in the end.

## Appendix E Additional results

#### Motion jitter comparison

![Figure](x7.png)

*Figure 8: Left: We visualized three consecutive frames within an motion sequence. It is evident that while there was no jitter in the motion source, significant jitter occurred in the $t$-th frame of the R2ET [zhang2023skinned] results, which was not the case with our method. Right: We visualize the corresponding right-hand height for this segment of the sequence. The results indicate that the jitter in the R2ET output was pronounced.*

To better illustrate the jitter issue present in the results from the R2ET [zhang2023skinned] method, we visualized consecutive frames generated by R2ET and our method in Figure [8](#A5.F8), and provided a line graph depicting the variations in height of the right-hand joint over time. These results demonstrate that R2ET is adversely affected by contradictions between skeletal retargeting and geometry correction phases, leading to significant motion jitter. In contrast, our method successfully avoids this problem.

#### Qualitative comparison with \textcitezhang2023semantics

![Figure](x8.png)

*Figure 9: Qualitative comparison with \textcitezhang2023semantics.*

Since \textcitezhang2023semantics did not open-source their code, we were unable to conduct a complete and fair comparison of their method with ours in our experiments. However, we endeavored to locate several examples presented in their paper and applied our MeshRet to the same motion sequences. The comparative results are displayed in Figure [9](#A5.F9). As observed in these examples, our method maintains the semantic integrity of the source motions, and it performs better in the Fireball case (the second motion sequence shown). This indicates that our method can achieve, and even surpass, the performance of their approach.

#### Metrics across different data splits

Tables [3](#A5.T3) and [4](#A5.T4) present the contact error and penetration ratio of our method compared to the baseline method across four different data splits. A consistent pattern observed is that performance improves for seen characters or motions. It is evident that our method outperforms the baseline across all data splits.

*Table 3: Contact errors of MeshRet and baselines across all data splits on Mixamo+.*

*Table 4: Penetration ratios of MeshRet and baselines across all data splits on Mixamo+.*

#### Ablation studies on ratios of proximal sensor pairs

The full approach can be considered a mixed version of Oursfar and Ourscls, utilizing an equal distribution of proximal and distal sensor pairs. To better illustrate this balance, we provide additional experimental results by testing different ratios of proximal to distal sensor pairs. Table [5](#A5.T5) compares our method’s performance with varying percentages of proximal sensor pairs under the Mixamo+ setting. As the percentage of proximal sensor pairs decreases, the interpenetration ratio fluctuates mildly, while the contact error initially decreases and then increases. Finally, with no proximal pairs (equivalent to the "far" version), the performance drops significantly. In Figure [10](#A5.F10), we present a qualitative comparison of our methods using different proximal sensor pair ratios. Except for the 100% Proximal version (equivalent to Ourscls) and the 0% Proximal version (equivalent to Oursfar), our method demonstrates fair robustness to the proximal sensor ratio in the 25%-75% interval. Based on these results, we conclude that choosing 50% proximal sensor pairs strikes a reasonable balance for achieving good performance.

*Table 5: Quantitative comparison between our methods with varing percentages of proximal sensor pairs under the Mixamo+ setting.*

![Figure](x9.png)

*Figure 10: Qualitative results with different proximal sensor pair ratios.*

#### Ablation studies on different sensor arragements

We conducted further ablation studies on different sensor arrangements. Specifically, we evaluated the performance of a model trained with half the sample points in the $\phi$ space in SCS, denoted as Oursϕ, and another model trained with half the sample points in the $l$ space in SCS, referred to as Oursl. As shown in Table [6](#A5.T6), Oursϕ compromises the interpenetration ratio, indicating that sufficient sample points in the space are crucial for avoiding interpenetration. We also found that both models introduce artifacts; please refer to Figure [11](#A5.F11).

*Table 6: Quantitative comparison between methods with different sensor arrangements.*

![Figure](x10.png)

*Figure 11: Qualitative comparison of additional ablation studies on sensor arrangements. The red rectangles identify artifacts introduced by different sensor arrangements.*

#### Failure cases with noisy inputs

We provide resutls with clean and noisy inputs in Figure[12](#A5.F12) and Figure[13](#A5.F13). The results of MeshRet exhibit interpenetration with noisy inputs.

![Figure](x11.png)

*Figure 12: Qualitative results on the Mixamo dataset with clean and noisy inputs. A red rectangle indicates interpenetration.*

![Figure](x12.png)

*Figure 13: Qualitative results on the Mixamo dataset with ScanRet characters as targets. A red rectangle indicates interpenetration.*

#### More cases

We present additional cases to validate the effectiveness of our MeshRet. Figures [14](#A5.F14), [15](#A5.F15), [16](#A5.F16), and [17](#A5.F17) depict four motion sequences retargeted from the source character to distinct target characters. These examples illustrate that our MeshRet is capable of generating high-quality motion sequences on target characters with diverse body shapes.

![Figure](x13.png)

*Figure 14: Snapshots of motion sequence 4 in ScanRet, retargeted from the source character to three distinct characters.*

![Figure](x14.png)

*Figure 15: Snapshots of motion sequence 43 in ScanRet, retargeted from the source character to three distinct characters.*

![Figure](x15.png)

*Figure 16: Snapshots of motion sequence 9 in ScanRet, retargeted from the source character to three distinct characters.*

![Figure](x16.png)

*Figure 17: Snapshots of motion sequence 45 in ScanRet, retargeted from the source character to three distinct characters.*

## Appendix F Broader impacts

Our work can provide animation professionals with enhanced results in motion retargeting, thereby alleviating their workload and increasing productivity in fields such as virtual reality, game development, and animation production. Regarding potential negative social impacts, we believe the likelihood of misuse of our work is minimal. This is because our work is situated in the midstream phase of the animation production pipeline, whereas privacy-invading forgeries, such as DeepFake, primarily occur during the downstream rendering phase.

Generated on Thu Jan 16 08:58:00 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)