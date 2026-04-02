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


arXiv:2511.05403v3 [cs.CV] 09 Feb 2026

[# PALM: A Dataset and Baseline for Learning Multi-Subject Hand Prior Zicong Fan1,2,3,† Edoardo Remelli1 David Dimond1 Fadime Sener1 Liuhao Ge1 Bugra Tekin1 Cem Keskin1 Shreyas Hampali1 1Meta Reality Labs 2ETH Zürich 3Max Planck Institute for Intelligent Systems, Tübingen ###### Abstract The ability to grasp objects, signal with gestures, and share emotion through touch all stem from the unique capabilities of human hands. Yet creating high-quality personalized hand avatars from images remains challenging due to complex geometry, appearance, and articulation, particularly under unconstrained lighting and limited views. Progress has also been limited by the lack of datasets that jointly provide accurate 3D geometry, high-resolution multi-view imagery, and a diverse population of subjects. To address this, we present PALM, a large-scale dataset comprising $13k$ high-quality hand scans from 263 subjects and $90k$ multi-view images, capturing rich variation in skin tone, age, and geometry. To show its utility, we present a baseline PALM-Net, a multi-subject prior over hand geometry and material properties learned via physically based inverse rendering, enabling realistic, relightable single-image hand avatar personalization. PALM’s scale and diversity make it a valuable real-world resource for hand modeling and related research. See the project page at this link](https://github.com/facebookresearch/PALM).

††$\dagger$ Work done during Meta internship

![[Uncaptioned image]](2511.05403v3/1_figures/dataset-teaser.jpg)

*Figure 1: Dataset overview: PALM is a large-scale dataset comprising calibrated multi-view high-resolution RGB images and 3dMD hand scans (a). It features $263$ subjects spanning a wide range of skin tones and hand sizes, $90k$ RGB images, and $13k$ high-quality hand scans with corresponding MANO registrations (b). This diversity and precision provide a foundation for learning a universal prior over human hand shape and appearance.*

## 1 Introduction

Human hands are central to how we interact with the physical and social world: we manipulate objects [[[19](#bib.bib62), [47](#bib.bib335), [52](#bib.bib336), [11](#bib.bib337), [12](#bib.bib315)]], express intent through gestures [[[49](#bib.bib340), [13](#bib.bib339), [44](#bib.bib269)]], and communicate affective cues via touch. Realistic and drivable hand avatars have the potential to transform virtual interaction, gaming, and telepresence. However, building such avatars from images remains a fundamentally challenging problem due to the complexity of hand geometry, appearance, and articulation, particularly under unconstrained lighting and from limited visual observations.

A critical missing component in the hand community is a large-scale, high-quality dataset that enables learning generalizable and physically grounded models of human hands. Existing datasets suffer from significant limitations: they often include only a small number of subjects [[[32](#bib.bib106), [27](#bib.bib321)]], lack accurate 3D hand geometry from real-world scans [[[32](#bib.bib106), [37](#bib.bib112)]], or are derived from hand-crafted synthetic data [[[15](#bib.bib318)]], limiting their utility for learning models that generalize across identity and illumination.

To fill this gap, we introduce PALM, a large-scale dataset of human hands containing publicly available-ready accurate hand scans, diverse in quantity and subject diversity. PALM includes $13k$ high-quality 3D hand scans and $90k$ high-resolution multi-view RGB images from $263$ subjects, each performing a diverse set of predefined hand poses designed to span a wide range of natural hand articulations. The subjects cover a broad range of skin tones and age groups. All data is captured using a commercial 3dMD scanner [[[43](#bib.bib39)]], providing precise, sub-millimeter geometry. Each scan is paired with synchronized multi-view images and a MANO registration (pose and shape), obtained via multi-view–consistent alignment to the 3D scans. Importantly, the capture environment, including lighting and scanner configuration, remained fixed throughout the entire collection process, enabling consistent illumination conditions across subjects.
While prior datasets have included either limited subjects or unreleased scan data, PALM will be made publicly available for research use upon publication, making it the most comprehensive and accessible dataset for studying generalizable hand models.

To highlight a practical use case for our data, we present a baseline model PALM-Net, an implicit neural prior over human hands that jointly models appearances, geometry and material properties using our dataset. PALM-Net is trained via physically based inverse rendering, decomposing each subject hand into geometry, albedo, specularity, roughness, and environment lights. The model conditions on pose and subject-specific latent codes, enabling it to capture pose-dependent effects. A key insight in PALM-Net is a shared environment lighting constraint across the subjects, which disentangles illumination from intrinsic hand appearance, allowing the model to generalize to novel lighting.

We apply our prior to the highly under-constrained task of single-image hand avatar personalization under unknown lighting. This monocular setting presents several challenges: depth ambiguity, occlusions (e.g*., self-occlusion of the palm or dorsum), and ambiguous lighting. Our central insight is that, despite the fine-scale complexity of hands (*e.g*., wrinkles, creases, texture), many fundamental properties – skin tone, material reflectance, and deformation behavior – are shared across individuals and can be captured by a learned prior. By optimizing the subject-specific latent code and scene illumination, our model reconstructs realistic, relightable, and articulated hand avatars from a single input image, even in uncontrolled conditions.

To perform extensive evaluation of the single-image personalization setting, we evaluate our method in both synthetic and real-world datasets.
Our results show that our method consistently outperforms other prior-based and non-prior-based methods for relightable hand personalization.

To summarize our contributions: 1) We introduce PALM, a large-scale dataset containing high-resolution RGB multi-view images of diverse subjects with detailed hand scans and accurate MANO registrations; 2)
We present a baseline, PALM-Net, a multi-subject implicit hand prior model that leverages PALM to learn physical hand properties such as geometry, albedo, specularity, and roughness;
3) We demonstrate the effectiveness of our prior model by using it to personalize and relight hand avatars from a single image under challenging and diverse environmental conditions.

## 2 Related Work

Hand datasets:
Recent years have seen an explosion of hand datasets, that can be categorized into hand-only [[[59](#bib.bib182), [30](#bib.bib105), [35](#bib.bib309), [15](#bib.bib318), [27](#bib.bib321)]], interacting hand [[[45](#bib.bib46), [32](#bib.bib106), [29](#bib.bib292)]], and hand-object [[[18](#bib.bib63), [14](#bib.bib272), [4](#bib.bib91), [26](#bib.bib266), [19](#bib.bib62), [16](#bib.bib56), [23](#bib.bib64), [2](#bib.bib89), [39](#bib.bib238), [51](#bib.bib330)]] datasets.
Early research on hand-only capture primarily focused on datasets collected using depth cameras [[[50](#bib.bib332)]]. Soon after, RGB-based datasets gained traction, with notable examples including STB [[[53](#bib.bib334)]] and FreiHAND [[[59](#bib.bib182)]].
Following this, interest expanded toward hand-object interaction datasets. Hampali *et al*. [[[18](#bib.bib63)]] released a dataset of single hands manipulating rigid YCB objects, while Fan *et al*. [[[14](#bib.bib272)]] used a marker-based MoCap setup to capture full-body interactions with articulated objects. More recently, Banerjee *et al*. [[[1](#bib.bib6)]] contributed a large-scale dataset featuring multi-object interactions.
Interacting-hand datasets have also grown in popularity. Tzionas *et al*. [[[45](#bib.bib46)]] pioneered capturing two-hand interactions using an RGB-D setup. Building on this, Moon *et al*. [[[32](#bib.bib106)]] leveraged a large-scale multi-view RGB system to recover 3D hand poses under strong self-contact, which significantly boosted interest in modeling two interacting hands from RGB data. Moon *et al*. [[[29](#bib.bib292)]] released a synthetic dataset using 3D annotations derived from [[[32](#bib.bib106)]].
Handy [[[35](#bib.bib309)]] provides 3dMD data but it is not publicly available.
Despite the rapid emergence of various hand datasets, most approaches for learning hand avatars still rely primarily on InterHand2.6M [[[34](#bib.bib312), [5](#bib.bib296)]] or on custom video recordings [[[21](#bib.bib304)]].
This is largely due to the absence of a high-quality dataset that simultaneously provides accurate 3D hand scans, high-resolution multi-view RGB imagery, and a diverse set of subjects.

Hand representations:
Learning articulated hand representations is a long-standing research problem.
Some methods solely focus on modelling hand joints [[[13](#bib.bib339), [41](#bib.bib114), [57](#bib.bib44)]] or geometries [[[20](#bib.bib324), [30](#bib.bib105), [37](#bib.bib112), [10](#bib.bib200), [56](#bib.bib119)]].
For example, MANO [[[37](#bib.bib112)]] pioneered parametric mesh-based hand geometry modeling, parameterizing shape with a PCA latent space. Moon *et al*. [[[31](#bib.bib311), [30](#bib.bib105)]] propose a non-linear approach for high-fidelity hand mesh modeling. HALO [[[22](#bib.bib34)]] is an implicit articulated hand geometry representation using occupancy network via a differentiable canonicalization layer.
There are also methods that model both geometry and appearances of hands [[[35](#bib.bib309), [36](#bib.bib313), [8](#bib.bib310), [24](#bib.bib308), [55](#bib.bib307), [9](#bib.bib314), [34](#bib.bib312), [5](#bib.bib296), [12](#bib.bib315)]].
HTML [[[36](#bib.bib313)]] extends MANO with a PCA-based texture model.
Handy [[[35](#bib.bib309)]] is a parametric hand model of shape and texture learned from proprietary hand scans.
Both Handy and HTML preprocess texture maps to minimize baked-in shadows and specularities.
LISA and OHTA [[[9](#bib.bib314)]] model shape and appearance fields, with lighting effects baked into the network and controlled by latent codes.
NIMBLE [[[24](#bib.bib308)]] models hands with bone, muscle, and skin deformation, using a light stage to capture pose-independent albedo and specular maps modeled with PCA bases.
HARP [[[21](#bib.bib304)]] optimizes the normal and albedo maps for the MANO hand mesh with a point light source to model shadow effects, demonstrating slight generalizability to novel illuminations.
URHand[[[8](#bib.bib310)]] models pose-dependent hand material properties and is trained on large-scale light stage data.
Capturing such pose-dependent material characteristics and lighting effects requires accurate environment map information, typically enabled by a sophisticated light stage setup with hundreds of synchronized cameras, as used in Nimble and URHand.
In contrast to these methods [[[9](#bib.bib314), [55](#bib.bib307), [8](#bib.bib310)]], our baseline method can jointly learn appearances, geometries and relight the hand avatar and does not rely on expensive light stage setup.

## 3 PALMDataset

Overview:
To study hand priors, we introduce PALM (see Figure [1](#S0.F1)), a high-quality dataset with accurate 3D hand annotations, high-resolution multi-view RGB images, and 3dMD hand scans. It contains $90k$ RGB images and $13k$ hand scans from 263 subjects (131 male, 132 female) with diverse skin tones and hand shapes. Data was captured using a 3dMD hand scanner [[[43](#bib.bib39)]] with 7 RGB and 14 monochrome machine vision cameras calibrated; the RGB images have a resolution of $2448\times 2048$; and the 3D hand scans were reconstructed using 3dMD’s software-driven triangulation technique based on Active Stereo Photogrammetry. Participants performed approximately 50 predefined right-hand gestures. See SupMat for more examples in PALM.

### 3.1 Data Characteristics

*Figure 2: Capture setup. Our 3dMD setup with $7$ RGB cameras.*

*Table 1: Publicly available datasets. Existing public datasets lack subject diversity, accurate hand scans, and high-quality multi-view RGB images that are important for training a strong hand appearance and geometry prior model. Our dataset contains a large number of subjects with high-resolution images and scans suitable for learning a universal hand prior.*

Dataset comparison:
Table [1](#S3.T1) compares publicly available hand-only datasets. Early datasets [[[50](#bib.bib332), [45](#bib.bib46), [33](#bib.bib333)]] focus on depth cameras and often have limited subjects and scale [[[45](#bib.bib46), [33](#bib.bib333)]]. Recent larger datasets focus on RGB images: InterHand2.6M [[[32](#bib.bib106)]] has 50 subjects with multi-view calibrated RGB, Re:InterHand [[[29](#bib.bib292)]] offers synthetically relit interacting-hand images of 10 subjects, and MANO [[[37](#bib.bib112)]] provides hand scans of 31 subjects. Handy [[[35](#bib.bib309)]] includes 3dMD scans, but these are unavailable. While MANO, InterHand2.6M, and our dataset are all suitable for learning priors for hands, most other datasets are suboptimal due to missing modalities, limited subject diversity, synthetic data, or low image quality, leading most methods [[[31](#bib.bib311), [34](#bib.bib312), [5](#bib.bib296), [21](#bib.bib304)]] to rely on InterHand2.6M despite lacking scans. Our dataset is substantially larger in scale, with $263$ subjects, real high-resolution calibrated RGB images, and $13k$ high-quality scans, making it ideally suited for learning robust hand priors; it will be released publicly to advance future research.

![Figure](2511.05403v3/1_figures/demographics.jpg)

*Figure 3: PALM demographics. (a) Age; (b) Height; (c) Skin tone distributions. Our dataset provides a wide distribution of skin tones and age groups representing a large variety of hand textures.*

Demographics:
Figure [3](#S3.F3) provides a detailed breakdown of the demographic distribution in PALM. The dataset includes participants aged 21–70 years, with the majority in the 31–40 (33%), 41–50 (26%), and 21–30 (24%) age brackets.
In terms of height, subjects range from 145 to 200 cm, with 48% in the 145–170 cm range and 35% in the 171–180 cm range. Only a small portion (4%) are taller than 190 cm.
Skin tone distribution is also diverse, comprising 38% medium, 27% dark, 20% tan, and 15% light tones. This diversity supports robust analysis across demographic variations.

### 3.2 Data Acquisition

Capture setup:
We use a 3dMD [[[43](#bib.bib39)]] hand scanner to capture high-resolution multi-view RGB images and 3D scans of the subjects. In particular, the 7-viewpoint setup consists of an array of 21 synchronized and calibrated machine vision cameras, including both RGB and monochrome sensors, arranged to provide full 360-degree coverage of the hand. The system captures images at a high resolution of $2448\times 2048$. Random light projectors are integrated to enhance surface detail and geometry accuracy. The 3D hand scans were reconstructed using 3dMD’s software-driven triangulation technique based on Active Stereo Photogrammetry. All cameras are calibrated, ensuring consistent alignment across views. This setup enables the acquisition of dense, accurate 3D hand meshes in various poses, suitable for studying both static poses and dynamic hand articulations.

Capture protocol:
Each participant is asked to stand still with their right hand placed inside the 3dMD capture volume.
Participants are instructed to perform approximately 50 predefined right-hand gestures, covering a wide range of articulations including open-hand poses, pinches, fist closures, and fine-grained finger movements.
To ensure consistency across subjects, a standardized gesture list is followed, and participants are guided through the sequence during the capture.
We capture each gesture as an independent hand scan.
This protocol ensures diverse and repeatable motion data across the entire subject pool.
All subjects are captured in the same lighting and camera setup.

3D keypoint annotation:
A semi-automatic pipeline is used to generate accurate 3D hand pose labels. The 2D keypoints are first manually annotated on a subset of images to train a 2D keypoint detector tailored to our capture setup. The detector, implemented as a U-Net [[[38](#bib.bib163)]] pre-trained on InterHand2.6M and fine-tuned on 10K manually annotated PALM images, is then used to estimate 2D keypoints for all camera views, which are subsequently triangulated to obtain 3D poses using multi-view geometry.
Specifically, we follow the approach used in InterHand2.6M and apply RANSAC-based triangulation to robustly solve for the 3D keypoint locations. This semi-automatic approach significantly reduces the manual labeling burden while ensuring reliable 3D pose annotations.

MANO registration:
The 3D MANO poses for each pose and subject are obtained by registering the MANO hand model to the hand scans. To register MANO to each hand gesture, we optimize the MANO hand model using a combination of 2D/3D keypoints, segmentation mask, and 3D hand scan supervision. Specifically, we minimize the closest-point distance from each MANO vertex to the corresponding hand scan surface.
Ground-truth masks are derived from the hand scans, and we employ Soft Rasterizer [[[25](#bib.bib84)]] for differentiable rendering to align the MANO silhouette with these masks.

The registration is independently performed for each subject in two stages. The first stage optimizes the per-subject hand shape parameters and per-frame pose parameters on a set of simple hand poses (e.g*., flat hand), and the second stage only optimizes the per-frame pose parameters while freezing the shape parameters. Our final 3D keypoints have a recall rate of 95% at 10mm threshold. Our MANO registrations show a mean fitting error of 5.3 mm with respect to the 3D keypoints (similar to that of InterHand2.6M [[[32](#bib.bib106)]]).

## 4 Method

PALM-Net, illustrated in Figure [4](#S4.F4), utilizes PALM, our large-scale collection of human hand data containing detailed hand scans and high-resolution RGB images, to train a personalizable and relightable hand prior.
In this section, we first introduce preliminary concepts on NeRF [[[28](#bib.bib283)]], the Neural Radiance Field technique at the core of our approach, and MANO [[[37](#bib.bib112)]], the parametric hand model that we leverage within our pipeline to map 3D points across different subjects and hand poses into a shared canonical representation. Next, we present PALM-Net, our novel framework for learning a multi-subject hand prior through physically based inverse rendering (PBR), and describe how to train our representation over multiple subjects. Finally, we detail how PALM-Net can be used to recover a personalized and relightable hand avatar from a single image. This is a highly under-constrained problem, and we show that a prior model such as PALM-Net helps in recovering realistic, personalized hand avatars even in extreme illumination settings.

### 4.1 Preliminaries

NeRF: Given a ray $\mathbf{r}=(\mathbf{o},\mathbf{d})$ defined by its camera
center $\mathbf{o}$ and viewing direction $\mathbf{d}$, NeRF [[[28](#bib.bib283)]] computes the output
radiance (*i.e*., pixel color) of the ray via:

$$
\begin{aligned}
C_{rf}(\mathbf{r})= &\int_{t_{n}}^{t_{f}}T(t_{n},t)\sigma_{t}(\mathbf{r}(t))L(\mathbf{r}(t),\mathbf{d})dt \tag{1}
\end{aligned}
$$
| | s.t | $\displaystyle\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ | |
| | | $\displaystyle T(t_{n},t)=\exp\left(-\int_{t_{n}}^{t}\sigma_{t}(\mathbf{r}(s))ds\right)$ | |

where $t_{n},t_{f}$ denote the near/far point for the ray integral; $\sigma_{t}(\mathbf{x}):\mathbb{R}^{3}\rightarrow\mathbb{R}$ is a neural network that models surface density at 3D point $\mathbf{x}$ ; $L(\mathbf{x},\mathbf{d}):\mathbb{R}^{3}\times\mathbb{R}^{3}\rightarrow\mathbb{R}$ is a neural network that parametrizes radiance color at 3D point $\mathbf{x}$ when observed from direction $\mathbf{d}$. In practice, the integrals above are approximated via quadrature, yielding:

$$
\begin{aligned}
C_{rf}(\mathbf{r})\approx &\sum_{i=1}^{N}w^{(i)}L(\mathbf{r}(t^{(i)}),\mathbf{d}) \tag{2}
\end{aligned}
$$
| | s.t | $\displaystyle\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ | |
| | | $\displaystyle w^{(i)}=T^{(i)}\left(1-\exp(-\sigma_{t}(\mathbf{r}(t^{(i)}))\delta^{(i)}\right)$ | |
| | | $\displaystyle T^{(i)}=\exp\left(-\sum_{jMANO [[[37](#bib.bib112)]] hand model is parametrized by $\mathbf{\Theta}=\{\mathbf{\theta},\mathbf{\beta},\mathbf{p}\}$, where $\mathbf{\theta}\in\rm I\!R^{45}$ denotes hand skeletal pose (joint angles), $\mathbf{\beta}\in\rm I\!R^{10}$ hand shape (parameterized by PCA coefficients) and $\mathbf{p}\in\rm I\!R^{6}$ global transformation.
The MANO model then maps $\mathbf{\Theta}$ to a posed 3D mesh $\mathcal{M}(\mathbf{\Theta})\in\rm I\!R^{778\times 3}$.
PALM-Net leverages MANO to perform inverse LBS using SNARF [[[7](#bib.bib219)]] that map points in 3D to a common canonical representation, i.e., given a 3D point in deformed coordinates $\mathbf{x}_{d}$, hand parameters $\mathbf{\Theta}$, we find the corresponding 3D point $\mathbf{x}_{c}$ in canonical space as,

$$\mathbf{x}_{d}=\operatorname*{arg\,min}_{\mathbf{x}}\Big\lvert\Big\lvert\sum_{i=1}^{n_{b}}w_{i}(\mathbf{x})\cdot B_{i}\cdot\mathbf{x}-\mathbf{x}_{d}\Big\lvert\Big\lvert_{2}^{2},$$ \tag{3}

where, $w_{i}(\mathbf{x})$ is the skinning weight associated with point $\mathbf{x}$ for bone $i$, and $B_{i}$ represents the $i^{\text{th}}$ bone transformation.

![Figure](2511.05403v3/1_figures/method.png)

*Figure 4: PALM-Net overview. Given (a) PALM, our multi-subject RGB dataset with $263$ subjects, PALM-Net explains each subject by optimizing subject-specific shape and appearance codes (b). (c) PALM-Net is an implicit physically-based network that is conditioned on the subject codes and renders to radiance, normal, and physically-based RGB images.*

### 4.2 PALM-Net

Physically based representations:
Inspired by [[[46](#bib.bib305), [6](#bib.bib329)]], PALM-Net (Figure [4](#S4.F4)c) decomposes the hand representation into a shape network $f_{g}(\cdot)$, a radiance field network $f_{rf}(\cdot)$, and a material network $f_{m}(\cdot)$. We model the canonical hand geometry with an implicit function:

$$
\begin{aligned}
f_{g}:\mathbf{x}_{c},\theta,\beta,\phi &\mapsto\sigma_{t}(\mathbf{x}_{c}),z(\mathbf{x}_{c}), \tag{4}
\end{aligned}
$$

where $\mathbf{x}_{c}\in\rm I\!R^{3}$ denotes a 3D point in the canonical space, $\theta,\beta$ are MANO parameter, and $\phi\in\rm I\!R^{d_{s}}$ captures subject-specific geometry latent code of dimension $d_{s}$.
$f_{g}(\cdot)$ outputs the opacity value, $\sigma_{t}(\mathbf{x}_{c})\in\rm I\!R$, as well as geometry features $z(\mathbf{x}_{c})\in\rm I\!R^{d_{s}}$ for the point $\mathbf{x}_{c}$.
Following [[[48](#bib.bib279)]], the opacity is obtained by converting the Signed Distance Function (SDF) values via the cumulative distribution function of the scaled Laplace distribution, $\Gamma_{\alpha_{1},\alpha_{2}}(s)$, where $\alpha_{1},\alpha_{2}>0$ are optimizable parameters. For details, we refer the reader to [[[48](#bib.bib279)]].

The outgoing radiance, $L(\mathbf{x}_{c},\mathbf{d})$ at canonical point $x_{c}$ viewed by the direction $\mathbf{d}\in\rm I\!R^{3}$ is obtained as,

$$
\begin{aligned}
f_{rf}:\mathbf{x}_{c},z,\text{ref}(\mathbf{d},\mathbf{n}),\mathbf{n},\theta,\psi &\mapsto L(\mathbf{x}_{c},\mathbf{d}), \tag{5}
\end{aligned}
$$

where, $\mathbf{n}$ is the surface normal obtained analytically from the SDF field, $\text{ref}(\mathbf{d},\mathbf{n})$ reflects the view direction $\mathbf{d}$ around the normal $\mathbf{n}$, and $\psi\in\rm I\!R^{d_{s}}$ is the appearance latent code.

Lastly, the spatially varying material field, $f_{m}$ is used to model the physically based rendering parameters, the albedo $\alpha\in\rm I\!R^{3}$, roughness $r\in\rm I\!R$, and metallicity $m\in\rm I\!R$ as,

$$
\begin{aligned}
f_{m}:\mathbf{x}_{c},z,\theta,\psi &\mapsto\alpha(\mathbf{x}_{c}),r(\mathbf{x}_{c}),m(\mathbf{x}_{c}) \tag{6}
\end{aligned}
$$

The canonical point $\mathbf{x}_{c}$ is encoded with hash grid encoding for all three networks, $f_{\{g,~rf,~m\}}$ to model high frequency details efficiently.

Physically based rendering:
For physically based rendering, we follow closely [[[46](#bib.bib305)]] and compute the radiance scattered by the volume along a certain camera ray ($\mathbf{o,d}$) using the quadrature approximation as,

| | $\displaystyle C_{pbr}(\mathbf{r})$ | $\displaystyle\approx\sum_{i=1}^{M}w^{(i)}BRDF\Big(\mathbf{d},\bar{\mathbf{d}}^{(i)},\alpha\big(\mathbf{r}(\bar{t}^{(i)})\big),r\big(\mathbf{r}(\bar{t}^{(i)})\big),$ | |
|---|---|---|---|
| | | $\displaystyle~~~~~~~~~~~~~~m\big(\mathbf{r}(\bar{t}^{(i)})\big),\mathbf{n}\Big)\cdot L_{i}(\mathbf{r}(\bar{t}^{(i)}),\bar{\mathbf{d}}^{(i)})\cdot\frac{1}{pdf(\bar{\mathbf{d}^{(i}})}$ | | (7) |
| | s.t | $\displaystyle\quad\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ | |
| | | $\displaystyle w^{(i)}=T^{(i)}\left(1-\exp(-\sigma_{t}(\mathbf{r}(\bar{t}^{(i)})\delta^{(i)})\right)$ | |
| | | $\displaystyle T^{(i)}=\exp\left(-\sum_{jEquation [2](#S4.E2); $M$ denotes the number of samples used to approximate the integrals along the ray; $\bar{\mathbf{d}}^{(i)}$ is the incoming light direction at sampling offset $\bar{t}^{(i)}$ sampled from the distribution, $pdf(\cdot)$ (uniform distribution over the unit sphere); $BRDF(\cdot)$ denotes the simplified version of Disney BRDF [[[3](#bib.bib26)]].
The term $L_{i}(\mathbf{x},\bar{\mathbf{d}})$ is the incoming radiance towards point $\mathbf{x}$ along direction $\bar{\mathbf{d}}$ and can be computed as the weighted sum of output radiance $C_{rf}(\mathbf{x},\bar{\mathbf{d}})$ (Equation [2](#S4.E2)) and radiance emitted from an environment map $\text{Env}(\bar{\mathbf{d}})$:

$$
\begin{aligned}
L_{i}(\mathbf{x},\bar{\mathbf{d}})= &C_{rf}(\mathbf{x},\bar{\mathbf{d}}) \tag{8}
\end{aligned}
$$
$$
\begin{aligned}
+ &\exp\left(-\int_{t_{n}^{\prime}}^{t_{f}^{\prime}}\sigma_{t}(\mathbf{x}+s\bar{\mathbf{d}})ds\right)\text{Env}(\bar{\mathbf{d}}), \tag{9}
\end{aligned}
$$

where ${t_{n}^{\prime}},{t_{f}^{\prime}}$ are near and far points of integration for secondary rays. The environment map is approximated by a set of Spherical Gaussians denoted by $\mathcal{SG}_{1},\mathcal{SG}_{2},...,\mathcal{SG}_{G}$. We refer the reader to [[[46](#bib.bib305)]] for derivation of the above equations.

Training losses:
Since reconstructing geometries and material properties from RGB images is a highly under-constrained problem, we devise a loss $\mathcal{L}$ that consists of several terms.
In particular, we first encourage RGB values to be consistent with an input image via

$$\mathcal{L}_{\text{rf}}=\sum_{\textbf{r}}\left\lVert C_{rf}(\mathbf{r})-\hat{C}(\mathbf{r})\right\rVert,$$ \tag{10}

where $\mathbf{r}$ is a ray casted from a sampled pixel on an image, and $C_{rf}(\mathbf{r})$ and $\hat{C}(\mathbf{r})$ are the rendered radiance value and ground-truth color.
The PBR rendered pixels $C_{pbr}$ are directly supervised with RGB values in a loss $\mathcal{L}_{\text{pbr}}$ similar to $\mathcal{L}_{\text{rf}}$.
Since our scans provide detailed geometries, we supervise the model with the rendered scan normals:

$$\mathcal{L}_{\text{normal}}=\sum_{\textbf{r}}\left\lVert\mathcal{N}(\mathbf{r})-\hat{\mathcal{N}}(\mathbf{r})\right\rVert.$$ \tag{11}

Note that the geometry information is shared by PBR and radiance field.
We encourage valid SDFs with the eikonal loss $\mathcal{L}_{\text{eikonal}}$ [[[17](#bib.bib1)]], which enforces the gradient at each point to have a unit norm.
To encourage smooth hand surfaces, we apply a Laplacian loss $\mathcal{L}_{\text{LAP}}$ on sampled points around the hand (see SupMat).
To encourage latent codes to be close to zeros, we penalize a MSE loss $\mathcal{L}_{\text{latent}}$ on both appearance and shape code.
To avoid foreground model explaining background pixels, we also supervise the networks with a segmentation loss

$$\mathcal{L}_{\text{segm}}=\sum_{\textbf{r}}\text{BCE}(\mathcal{S}(\mathbf{r}),\hat{\mathcal{S}}(\mathbf{r}))$$ \tag{12}

where $\mathcal{S}(\mathbf{r})\in\rm I\!R$ represents the probability of a pixel being the foreground and $\text{BCE}(\cdot,\cdot)$ is the binary cross entropy loss to the ground-truth hand segmentation mask $\hat{\mathcal{S}}(\mathbf{r})$ rendered from the hand scans.
Finally, to capture high-frequency details, we render image patches and compare them with the ground-truth using the perceptual similarity loss $\mathcal{L}_{\text{LPIPS}}$ [[[54](#bib.bib322)]].
The total loss $\mathcal{L}$ is defined as

| | $\displaystyle\mathcal{L}=\mathcal{L}_{\text{rf}}$ | $\displaystyle+\lambda_{\text{pbr}}\mathcal{L}_{\text{pbr}}+\lambda_{\text{segm}}\mathcal{L}_{\text{segm}}+\lambda_{\text{normal}}\mathcal{L}_{\text{normal}}$ | |
|---|---|---|---|
| | | $\displaystyle+\lambda_{\text{eikonal}}\mathcal{L}_{\text{eikonal}}+\lambda_{\text{LPIPS}}\mathcal{L}_{\text{LPIPS}}$ | | (13) |
| | | $\displaystyle+\lambda_{\text{LAP}}\mathcal{L}_{\text{LAP}}+\lambda_{\text{latent}}\mathcal{L}_{\text{latent}}$ | | (14) |

where $\lambda_{*}$ are the weights for the loss terms (see SupMat).
We gradually decrease $\lambda_{\text{segm}}$ over time.

Multi-subject PBR prior:
When training PALM-Net across multiple subjects,
we model the detailed hand shape and appearance of each subject by conditioning PALM-Net on a shape code $\phi$ and an appearance code $\psi$.
That is, we model subject identities by disentangling shapes from appearances. Empirically, we found that when training on multiple subjects, having separate latent codes for geometries and appearances yields better reconstructions than having a shared latent code for both.

Given a set of images of hands $\{\mathcal{I}\}$ from $N$ subjects, and randomly initialized latent codes $\{\phi_{1..N},\psi_{1..N}\}$, PALM-Net explains the images of multiple subjects by optimizing on the network weights $\Phi$, the subject codebook $\{\phi_{1..N},\psi_{1..N}\}$, and Spherical Gaussian parameters for the environment lights.
Note that we optimize for a single environment across all subjects as the subjects are captured in the same setup.
In particular, our objective function is,

$$\min_{\Phi,\{\phi_{i}\}_{i=1}^{N},\{\psi_{i}\}_{i=1}^{N},\{\mathcal{SG}_{i}\}_{i=1}^{G}}\mathcal{L}$$ \tag{15}

Personalization:
A strong prior model on the hand appearance and geometry allows us to personalize our model to images of hands captured in extreme environment settings. This is mainly because the prior model constrains the albedo and material properties of the hand during personalization and all the environment effects could be explained separately. This is achieved by solving an optimization problem where the shape code $\phi$ and the appearance code $\psi$ for a given input image are optimized along with the environment map while keeping the network weights $\Phi$ of the PBR prior model frozen. In particular, at each iteration, we sample a random batch of rays from the input image and optimize for the following objective:

| | $\displaystyle\min_{\phi,\psi,\{\mathcal{SG}\}_{i=1}^{G}}\mathcal{L}_{\text{rf}}$ | $\displaystyle+\lambda_{\text{pbr}}\mathcal{L}_{\text{pbr}}+\lambda_{\text{segm}}\mathcal{L}_{\text{segm}}$ | |
|---|---|---|---|
| | | $\displaystyle+\lambda_{\text{LPIPS}}\mathcal{L}_{\text{LPIPS}}$ | | (16) |

## 5 Experiments

![Figure](2511.05403v3/1_figures/itw_qualitative.jpg)

*Figure 5: In-the-wild image personalization. (a) The first column shows the images used for personalization, followed by the renderings of the geometry and materials of the hand avatar obtained using our prior model. The PBR rendering refers to the physically-based rendering with estimated environment map. The last column shows the relighting results of personalized hand avatar in a novel pose. Our method retrieves realistic hand avatars even when the input personalization image has complex lighting effects. (b) Additional relighting results with in-the-wild images.*

We evaluate our baseline on the task of hand avatar personalization and relighting from a single RGB image using three different datasets.
Metric and baseline details are in SupMat.

### 5.1 Datasets

InterHand2.6M:
To evaluate the performance on real images, we use images from InterHand2.6M [[[32](#bib.bib106)]] for personalization.
The dataset consists of accurate 3D hand poses of subjects in predefined poses as well as high-resolution RGB images.
We evaluate on right-hand only sequences using the test split of the dataset.
To reliably measure performance, we randomly select two views for each sequence, resulting in 12 sequences.
We uniformly sample 20 images for each sequence for the evaluation and use the first frame of each sequence for training a personalized model.

HARP relit:
Since there is no real dataset to evaluate hand avatar under novel environment and poses, following [[[21](#bib.bib304)]], we render a synthetic dataset using Blender.
In particular, to create synthetic hand template, for each sequence, we sample new MANO shape parameters and apply a new skin tone using UV textures from DART [[[15](#bib.bib318)]].
To animate the hand, we use the hand pose parameters from HARP data release [[[21](#bib.bib304)]].
We use the first frame of each sequence for training a personalized model and use the remaining frames for evaluation. We render the training and evaluation images using different environment maps to evaluate relighting.

In-the-wild images:
To show the generalization of our method under real world scenarios, we select in-the-wild images from the internet with diverse lighting conditions and poses. After personalization on these images, we render them with novel environment maps using novel poses from [[[21](#bib.bib304)]] and show qualitative results.

*Table 2: InterHand2.6M dataset evaluation. Comparison of methods on single-image personalization task using PSNR, SSIM, and LPIPS metrics. Our method outperforms previous methods in the novel pose setting where the training and evaluation environment maps are the same.*

*Table 3: Synthetic dataset evaluation. Comparison of methods based on PSNR, SSIM, and LPIPS. Our method outperforms previous methods in the novel environment, novel pose setting showing that the appearance reconstructions of our model is more accurate.*

![Figure](2511.05403v3/x1.png)

*Figure 6: Personalization results on InterHand2.6M. The first column shows the image used for personalization. (a) Personalized hand avatars rendered in novel poses in the training environment. (b) Personalized hand avatars rendered in novel poses in the novel environment. The hand avatars from our method are more realistic than other baselines.*

### 5.2 Comparison and Analysis

Baseline comparison: Table [2](#S5.T2) compares the performance of our method with the baselines on InterHand2.6M. Our method outperforms baseline methods on all metrics showing high quality rendering in novel poses and viewpoints. Figure [6](#S5.F6) shows the qualitative images on InterHand2.6M dataset for both the training and novel environments. The images from PALM-Net are more realistic than that of the baselines.
Table [3](#S5.T3) compares ours with the baselines on the synthetic dataset, where the evaluations are performed in novel environment settings. Our PALM-Net prior model outperforms the baselines by showing more realistic relighting results in novel environments (more results in SupMat).
Figure [5](#S5.F5) shows qualitative results of personalization on in-the-wild images with diverse lighting condition and poses. Despite these challenges, our method produces realistic avatars and plausible relit images.
For example, even in extreme setting where the input image is grayscale, our method can still recover plausible albedo thanks to our prior on hand appearance and the optimization over the environment map.

*Table 4: Effects of 3dMD normals.*

![Figure](2511.05403v3/1_figures/normal.jpg)

*Figure 7: Effects of 3dMD scans for hand personalization.*

Supervision with 3dMD normals:
Traditional multi-view techniques require a massive amount of RGB views for high-fidelity 3D reconstruction [[[28](#bib.bib283)]].
In our capture setup, we use a hybrid approach, combining 3dMD scans and sparse multi-view RGB images for training our prior models.
Figure [7](#S5.F7) shows a qualitative comparison with and without 3dMD normals.
In particular, we train two multi-subject PBR prior models on PALM, one with 3dMD normal supervision and one without.
Then we take these two prior models and personalize on an InterHand2.6M image.
Figure [7](#S5.F7) shows that, 3dMD normals from hand scans are crucial in reducing pepper-like artifacts and it helps to reduce floaters.
Table [4](#S5.T4) quantitatively compares in this novel pose evaluation.

*Table 5: Effects of modelling environment lightings.*

![Figure](2511.05403v3/x2.png)

*Figure 8: The effect of modelling environment.*

Environment map optimization:
During personalization, PALM-Net explains the image with geometry, material properties and environment lightings.
Table [5](#S5.T5) shows that modelling environment lightings enables the model to be more expressive in fitting the input image. An example of physically based rendered RGB images are show in Figure [8](#S5.F8). When allowed optimizing the environment in PALM-Net, the fitting results are closer to that of the input image.

## 6 Conclusion

PALM is a large-scale dataset combining accurate 3D hand geometry, high-resolution multi-view imagery, and a diverse subject pool, addressing key limitations in existing datasets. Through PALM-Net, we demonstrate that physically based inverse rendering with a multi-subject prior enables realistic, relightable single-image personalization. The dataset’s scale, diversity, and accompanying baseline make it a solid resource for future work.

## References

-
[1]
P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol, et al. (2025)

HOT3D: hand and object tracking in 3d from egocentric multi-view videos.

In CVPR,

pp. 7061–7071.

Cited by: [§2](#S2.p1.1).

-
[2]
S. Brahmbhatt, C. Tang, C. D. Twigg, C. C. Kemp, and J. Hays (2020)

ContactPose: A dataset of grasps with object contact and hand pose.

In ECCV,

Vol. 12358, pp. 361–378.

Cited by: [§2](#S2.p1.1).

-
[3]
B. Burley (2012)

Physically-based shading at disney.

In Proc. of SIGGRAPH,

Cited by: [§4.2](#S4.SS2.p5.12).

-
[4]
Y. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield, J. Kautz, and D. Fox (2021)

DexYCB: A benchmark for capturing hand grasping of objects.

In CVPR,

pp. 9044–9053.

Cited by: [§2](#S2.p1.1).

-
[5]
X. Chen, B. Wang, and H. Shum (2023)

Hand avatar: free-pose hand animation and rendering from monocular video.

In CVPR,

Cited by: [§2](#S2.p1.1),
[§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2).

-
[6]
X. Chen, T. Jiang, J. Song, J. Yang, M. J. Black, A. Geiger, and O. Hilliges (2022)

gDNA: towards generative detailed neural avatars.

In CVPR,

pp. 20427–20437.

Cited by: [§4.2](#S4.SS2.p1.3).

-
[7]
X. Chen, Y. Zheng, M. J. Black, O. Hilliges, and A. Geiger (2021)

SNARF: differentiable forward skinning for animating non-rigid neural implicit shapes.

In ICCV,

Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.9).

-
[8]
Z. Chen, G. Moon, K. Guo, C. Cao, S. Pidhorskyi, T. Simon, R. Joshi, Y. Dong, Y. Xu, B. Pires, H. Wen, L. Evans, B. Peng, J. Buffalini, A. Trimble, K. McPhail, M. Schoeller, S. Yu, J. Romero, M. Zollhöfer, Y. Sheikh, Z. Liu, and S. Saito (2024)

URhand: universal relightable hands.

In CVPR,

Cited by: [§2](#S2.p2.1).

-
[9]
E. Corona, T. Hodan, M. Vo, F. Moreno-Noguer, C. Sweeney, R. Newcombe, and L. Ma (2022)

LISA: learning implicit shape and appearance of hands.

In CVPR,

pp. 20533–20543.

Cited by: [§2](#S2.p2.1).

-
[10]
E. Duran, M. Kocabas, V. Choutas, Z. Fan, and M. J. Black (2024)

HMP: hand motion priors for pose and shape estimation from video.

In wacv,

Cited by: [§2](#S2.p2.1).

-
[11]
Z. Fan, T. Ohkawa, L. Yang, N. Lin, Z. Zhou, S. Zhou, J. Liang, Z. Gao, X. Zhang, X. Zhang, et al. (2024)

Benchmarks and challenges in pose estimation for egocentric hand interactions with objects.

In ECCV,

pp. 428–448.

Cited by: [§1](#S1.p1.1).

-
[12]
Z. Fan, M. Parelli, M. E. Kadoglou, M. Kocabas, X. Chen, M. J. Black, and O. Hilliges (2024)

HOLD: category-agnostic 3d reconstruction of interacting hands and objects from video.

In CVPR,

pp. 494–504.

Cited by: [§1](#S1.p1.1),
[§2](#S2.p2.1).

-
[13]
Z. Fan, A. Spurr, M. Kocabas, S. Tang, M. J. Black, and O. Hilliges (2021)

Learning to disambiguate strongly interacting hands via probabilistic per-pixel part segmentation.

In i3dv,

pp. 1–10.

Cited by: [§1](#S1.p1.1),
[§2](#S2.p2.1).

-
[14]
Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges (2023)

ARCTIC: a dataset for dexterous bimanual hand-object manipulation.

In Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

Cited by: [§2](#S2.p1.1).

-
[15]
D. Gao, Y. Xiu, K. Li, L. Yang, F. Wang, P. Zhang, B. Zhang, C. Lu, and P. Tan (2022)

DART: Articulated Hand Model with Diverse Accessories and Rich Textures.

In NeurIPS,

Cited by: [§1](#S1.p2.1),
[§2](#S2.p1.1),
[Table 1](#S3.T1.28.28.28.4),
[§5.1](#S5.SS1.p2.1).

-
[16]
G. Garcia-Hernando, S. Yuan, S. Baek, and T. Kim (2018)

First-person hand action benchmark with RGB-D videos and 3D hand pose annotations.

In CVPR,

pp. 409–419.

Cited by: [§2](#S2.p1.1).

-
[17]
A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman (2020)

Implicit geometric regularization for learning shapes.

In icml,

Cited by: [§4.2](#S4.SS2.p6.10).

-
[18]
S. Hampali, M. Rad, M. Oberweger, and V. Lepetit (2020)

HOnnotate: A method for 3D annotation of hand and object poses.

In CVPR,

pp. 3193–3203.

Cited by: [§2](#S2.p1.1).

-
[19]
Y. Hasson, G. Varol, D. Tzionas, I. Kalevatykh, M. J. Black, I. Laptev, and C. Schmid (2019)

Learning joint reconstruction of hands and manipulated objects.

In CVPR,

pp. 11807–11816.

Cited by: [§1](#S1.p1.1),
[§2](#S2.p1.1).

-
[20]
Z. Huang, Y. Chen, D. Kang, J. Zhang, and Z. Tu (2023)

PHRIT: parametric hand representation with implicit template.

In 2023 IEEE/CVF International Conference on Computer Vision (ICCV),

Vol. , pp. 14928–14938.

Cited by: [§2](#S2.p2.1).

-
[21]
K. Karunratanakul, S. Prokudin, O. Hilliges, and S. Tang (2023)

HARP: personalized hand reconstruction from a monocular rgb video.

In CVPR,

Cited by: [§2](#S2.p1.1),
[§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2),
[§5.1](#S5.SS1.p2.1),
[§5.1](#S5.SS1.p3.1),
[Table 2](#S5.T2.3.3.5.2.1),
[Table 3](#S5.T3.3.3.5.2.1).

-
[22]
K. Karunratanakul, A. Spurr, Z. Fan, O. Hilliges, and S. Tang (2021)

A skeleton-driven neural occupancy representation for articulated hands.

In i3dv,

pp. 11–21.

Cited by: [§2](#S2.p2.1).

-
[23]
T. Kwon, B. Tekin, J. Stühmer, F. Bogo, and M. Pollefeys (2021)

H2O: Two hands manipulating objects for first person interaction recognition.

In ICCV,

pp. 10138–10148.

Cited by: [§2](#S2.p1.1).

-
[24]
Y. Li, L. Zhang, Z. Qiu, Y. Jiang, N. Li, Y. Ma, Y. Zhang, L. Xu, and J. Yu (2022-07)

NIMBLE: a non-rigid hand model with bones and muscles.

ACM Trans. Graph. 41 (4).

External Links: ISSN 0730-0301,
[Link](https://doi.org/10.1145/3528223.3530079),
[Document](https://dx.doi.org/10.1145/3528223.3530079)

Cited by: [§2](#S2.p2.1).

-
[25]
S. Liu, T. Li, W. Chen, and H. Li (2019)

Soft rasterizer: a differentiable renderer for image-based 3d reasoning.

In ICCV,

pp. 7708–7717.

Cited by: [§3.2](#S3.SS2.p4.1).

-
[26]
Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi (2022)

HOI4D: a 4D egocentric dataset for category-level human-object interaction.

In CVPR,

pp. 21013–21022.

Cited by: [§2](#S2.p1.1).

-
[27]
J. Martinez, E. Kim, J. Romero, T. Bagautdinov, S. Saito, S. Yu, S. Anderson, M. Zollhöfer, T. Wang, S. Bai, C. Li, S. Wei, R. Joshi, W. Borsos, T. Simon, J. Saragih, P. Theodosis, A. Greene, A. Josyula, S. M. Maeta, A. I. Jewett, S. Venshtain, C. Heilman, Y. Chen, S. Fu, M. E. A. Elshaer, T. Du, L. Wu, S. Chen, K. Kang, M. Wu, Y. Emad, S. Longay, A. Brewer, H. Shah, J. Booth, T. Koska, K. Haidle, M. Andromalos, J. Hsu, T. Dauer, P. Selednik, T. Godisart, S. Ardisson, M. Cipperly, B. Humberston, L. Farr, B. Hansen, P. Guo, D. Braun, S. Krenn, H. Wen, L. Evans, N. Fadeeva, M. Stewart, G. Schwartz, D. Gupta, G. Moon, K. Guo, Y. Dong, Y. Xu, T. Shiratori, F. Prada, B. R. Pires, B. Peng, J. Buffalini, A. Trimble, K. McPhail, M. Schoeller, and Y. Sheikh (2024)

Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars.

NeurIPS.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p1.1).

-
[28]
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng (2021)

Nerf: representing scenes as neural radiance fields for view synthesis.

Communications of the ACM 65 (1), pp. 99–106.

Cited by: [§4.1](#S4.SS1.p1.3),
[§4](#S4.p1.1),
[§5.2](#S5.SS2.p2.1).

-
[29]
G. Moon, S. Saito, W. Xu, R. Joshi, J. Buffalini, H. Bellan, N. Rosen, J. Richardson, M. Mize, P. De Bree, et al. (2023)

A dataset of relighted 3d interacting hands.

arXiv preprint arXiv:2310.17768.

Cited by: [§2](#S2.p1.1),
[§3.1](#S3.SS1.p1.2),
[Table 1](#S3.T1.25.25.25.4).

-
[30]
G. Moon, T. Shiratori, and K. M. Lee (2020)

DeepHandMesh: A weakly-supervised deep encoder-decoder framework for high-fidelity hand mesh modeling.

In ECCV,

Vol. 12347, pp. 440–455.

Cited by: [§2](#S2.p1.1),
[§2](#S2.p2.1).

-
[31]
G. Moon, W. Xu, R. Joshi, C. Wu, and T. Shiratori (2024)

Authentic hand avatar from a phone scan via universal hand model.

In CVPR,

pp. 2029–2038.

Cited by: [§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2),
[Table 2](#S5.T2.3.3.6.3.1),
[Table 3](#S5.T3.3.3.6.3.1).

-
[32]
G. Moon, S. Yu, H. Wen, T. Shiratori, and K. M. Lee (2020)

InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image.

In ECCV,

Vol. 12365, pp. 548–564.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p1.1),
[§3.1](#S3.SS1.p1.2),
[§3.2](#S3.SS2.p5.1),
[Table 1](#S3.T1.22.22.22.4),
[§5.1](#S5.SS1.p1.1).

-
[33]
F. Mueller, D. Mehta, O. Sotnychenko, S. Sridhar, D. Casas, and C. Theobalt (2017)

Real-time hand tracking under occlusion from an egocentric rgb-d sensor.

In ICCV,

pp. 1154–1163.

Cited by: [§3.1](#S3.SS1.p1.2),
[Table 1](#S3.T1.13.13.13.4).

-
[34]
A. Mundra, J. Wang, M. Habermann, C. Theobalt, M. Elgharib, et al. (2023)

Livehand: real-time and photorealistic neural hand rendering.

In ICCV,

pp. 18035–18045.

Cited by: [§2](#S2.p1.1),
[§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2).

-
[35]
R. A. Potamias, S. Ploumpis, S. Moschoglou, V. Triantafyllou, and S. Zafeiriou (2023)

Handy: towards a high fidelity 3d hand shape and appearance model.

In CVPR,

pp. 4670–4680.

Cited by: [§2](#S2.p1.1),
[§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2),
[Table 2](#S5.T2.3.3.4.1.1),
[Table 3](#S5.T3.3.3.4.1.1).

-
[36]
N. Qian, J. Wang, F. Mueller, F. Bernard, V. Golyanik, and C. Theobalt (2020)

HTML: a parametric hand texture model for 3d hand reconstruction and personalization.

In ECCV,

pp. 54–71.

Cited by: [§2](#S2.p2.1).

-
[37]
J. Romero, D. Tzionas, and M. J. Black (2017)

Embodied hands: Modeling and capturing hands and bodies together.

ACM TOG 36 (6), pp. 245:1–245:17.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p2.1),
[§3.1](#S3.SS1.p1.2),
[Table 1](#S3.T1.30.30.30.3),
[§4.1](#S4.SS1.SSS0.Px1.p1.9),
[§4](#S4.p1.1).

-
[38]
O. Ronneberger, P. Fischer, and T. Brox (2015)

U-net: convolutional networks for biomedical image segmentation.

In International Conference on Medical image computing and computer-assisted intervention,

pp. 234–241.

Cited by: [§3.2](#S3.SS2.p3.1).

-
[39]
F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, and A. Yao (2022)

Assembly101: a large-scale multi-view video dataset for understanding procedural activities.

In CVPR,

pp. 21064–21074.

Cited by: [§2](#S2.p1.1).

-
[40]
T. Simon, H. Joo, I. Matthews, and Y. Sheikh (2017)

Hand keypoint detection in single images using multiview bootstrapping.

In CVPR,

pp. 4645–4653.

Cited by: [Table 1](#S3.T1.10.10.10.4).

-
[41]
A. Spurr, U. Iqbal, P. Molchanov, O. Hilliges, and J. Kautz (2020)

Weakly supervised 3D hand pose estimation via biomechanical constraints.

In ECCV,

Vol. 12362, pp. 211–228.

Cited by: [§2](#S2.p2.1).

-
[42]
D. Tang, H. Jin Chang, A. Tejani, and T. Kim (2014)

Latent regression forest: structured estimation of 3d articulated hand posture.

In CVPR,

pp. 3786–3793.

Cited by: [Table 1](#S3.T1.2.2.2.3).

-
[43]
3dMDhand system series.

Note: [https://3dmd.com/products/](https://3dmd.com/products/)

Cited by: [§1](#S1.p3.3),
[§3.2](#S3.SS2.p1.1),
[§3](#S3.p1.3).

-
[44]
T. H. E. Tse, F. Mueller, Z. Shen, D. Tang, T. Beeler, M. Dou, Y. Zhang, S. Petrovic, H. J. Chang, J. Taylor, et al. (2023)

Spectral graphormer: spectral graph-based transformer for egocentric two-hand reconstruction using multi-view color images.

In ICCV,

pp. 14666–14677.

Cited by: [§1](#S1.p1.1).

-
[45]
D. Tzionas, L. Ballan, A. Srikantha, P. Aponte, M. Pollefeys, and J. Gall (2016)

Capturing hands in action using discriminative salient points and physics simulation.

IJCV 118 (2), pp. 172–193.

Cited by: [§2](#S2.p1.1),
[§3.1](#S3.SS1.p1.2),
[Table 1](#S3.T1.7.7.7.4).

-
[46]
S. Wang, B. Antic, A. Geiger, and S. Tang (2024)

IntrinsicAvatar: physically based inverse rendering of dynamic humans from monocular videos via explicit ray tracing.

In CVPR,

pp. 1877–1888.

Cited by: [§4.2](#S4.SS2.p1.3),
[§4.2](#S4.SS2.p5.1),
[§4.2](#S4.SS2.p5.14).

-
[47]
S. Wang, H. He, M. Parelli, C. Gebhardt, Z. Fan, and J. Song (2025)

MagicHOI: leveraging 3d priors for accurate hand-object reconstruction from short monocular video clips.

In ICCV,

pp. 5957–5968.

Cited by: [§1](#S1.p1.1).

-
[48]
L. Yariv, J. Gu, Y. Kasten, and Y. Lipman (2021)

Volume rendering of neural implicit surfaces.

In NeurIPS,

Cited by: [§4.2](#S4.SS2.p1.13).

-
[49]
Z. Yu, S. Zafeiriou, and T. Birdal (2025-06)

Dyn-HaMR: recovering 4d interacting hand motion from a dynamic camera.

In CVPR,

Cited by: [§1](#S1.p1.1).

-
[50]
S. Yuan, Q. Ye, B. Stenger, S. Jain, and T. Kim (2017)

Bighand2. 2m benchmark: hand pose dataset and state of the art analysis.

In CVPR,

pp. 4866–4874.

Cited by: [§2](#S2.p1.1),
[§3.1](#S3.SS1.p1.2),
[Table 1](#S3.T1.4.4.4.3).

-
[51]
H. Zhang, S. Christen, Z. Fan, O. Hilliges, and J. Song (2024)

GraspXL: generating grasping motions for diverse objects at scale.

In ECCV,

pp. 386–403.

Cited by: [§2](#S2.p1.1).

-
[52]
H. Zhang, S. Christen, Z. Fan, L. Zheng, J. Hwangbo, J. Song, and O. Hilliges (2024)

ArtiGrasp: physically plausible synthesis of bi-manual dexterous grasping and articulation.

In i3dv,

pp. 235–246.

Cited by: [§1](#S1.p1.1).

-
[53]
J. Zhang, J. Jiao, M. Chen, L. Qu, X. Xu, and Q. Yang (2016)

3d hand pose tracking and estimation using stereo matching.

arXiv preprint arXiv:1610.07214.

Cited by: [§2](#S2.p1.1),
[Table 1](#S3.T1.16.16.16.4).

-
[54]
R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang (2018)

The unreasonable effectiveness of deep features as a perceptual metric.

In CVPR,

Cited by: [§4.2](#S4.SS2.p6.15).

-
[55]
X. Zheng, C. Wen, Z. Su, Z. Xu, Z. Li, Y. Zhao, and Z. Xue (2024)

OHTA: one-shot hand avatar via data-driven implicit priors.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 799–810.

Cited by: [§2](#S2.p2.1).

-
[56]
A. Ziani, Z. Fan, M. Kocabas, S. Christen, and O. Hilliges (2022)

TempCLR: reconstructing hands via time-coherent contrastive learning.

In i3dv,

pp. 627–636.

Cited by: [§2](#S2.p2.1).

-
[57]
C. Zimmermann and T. Brox (2017)

Learning to estimate 3D hand pose from single RGB images.

In ICCV,

pp. 4913–4921.

Cited by: [§2](#S2.p2.1).

-
[58]
C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, and T. Brox (2019)

FreiHAND: A dataset for markerless capture of hand pose and shape from single RGB images.

In ICCV,

pp. 813–822.

Cited by: [Table 1](#S3.T1.19.19.19.4).

-
[59]
C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, and T. Brox (2019)

Freihand: a dataset for markerless capture of hand pose and shape from single rgb images.

In ICCV,

Cited by: [§2](#S2.p1.1).



Experimental support, please
[view the build logs](./2511.05403v3/__stdout.txt)
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