HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: axessibility

- failed: epic

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: arXiv.org perpetual non-exclusive license

arXiv:2401.10891v2 [cs.CV] 07 Apr 2024

[# Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data Lihe Yang${}^{1}$††${\dagger}$ corresponding authors Bingyi Kang${}^{2\hskip 0.85358pt{\dagger}}$ Zilong Huang${}^{2}$ Xiaogang Xu${}^{3,4}$ Jiashi Feng${}^{2}$ Hengshuang Zhao${}^{1\hskip 0.56905pt{\ddagger}}$ ${}^{1}$HKU ${}^{2}$TikTok ${}^{3}$CUHK ${}^{4}$ZJU ${\dagger}$ project lead ${\ddagger}$ corresponding author https://depth-anything.github.io](https://depth-anything.github.io)

###### Abstract

††Work was done during an internship at TikTok.

This work presents Depth Anything 111While the grammatical soundness of this name may be questionable, we treat it as a whole and pay homage to Segment Anything [[[27](#bib.bib27)]]., a highly practical solution for robust monocular depth estimation. Without pursuing novel technical modules, we aim to build a simple yet powerful foundation model dealing with any images under any circumstances.
To this end, we scale up the dataset by designing a data engine to collect and automatically annotate large-scale unlabeled data ($\sim$62M), which significantly enlarges the data coverage and thus is able to reduce the generalization error.
We investigate two simple yet effective strategies that make data scaling-up promising. First, a more challenging optimization target is created by leveraging data augmentation tools. It compels the model to actively seek extra visual knowledge and acquire robust representations. Second, an auxiliary supervision is developed to enforce the model to inherit rich semantic priors from pre-trained encoders. We evaluate its zero-shot capabilities extensively, including six public datasets and randomly captured photos.
It demonstrates impressive generalization ability (Figure [1](#S0.F1)).
Further, through fine-tuning it with metric depth information from NYUv2 and KITTI, new SOTAs are set. Our better depth model also results in a better depth-conditioned ControlNet.
Our models are released [here](https://github.com/LiheYoung/Depth-Anything).

*

*Figure 1: Our model exhibits impressive generalization ability across extensive unseen scenes. Left two columns: COCO [[[36](#bib.bib36)]]. Middle two: SA-1B [[[27](#bib.bib27)]] (a hold-out unseen set). Right two: photos captured by ourselves. Our model works robustly in low-light environments (1st and 3rd column), complex scenes (2nd and 5th column), foggy weather (5th column), and ultra-remote distance (5th and 6th column), etc*.*

## 1 Introduction

The field of computer vision and natural language processing is currently experiencing a revolution with the emergence of “foundation models" [[[6](#bib.bib6)]] that demonstrate strong zero-/few-shot performance in various downstream scenarios [[[45](#bib.bib45), [59](#bib.bib59)]].
These successes primarily rely on large-scale training data that can effectively cover the data distribution.
Monocular Depth Estimation (MDE), which is a fundamental problem with broad applications in robotics [[[66](#bib.bib66)]], autonomous driving [[[64](#bib.bib64), [80](#bib.bib80)]], virtual reality [[[48](#bib.bib48)]], *etc*., also requires a foundation model to estimate depth information from a single image.
However, this has been underexplored due to the difficulty of building datasets with tens of millions of depth labels. MiDaS [[[46](#bib.bib46)]] made a pioneering study along this direction by training an MDE model on a collection of mixed labeled datasets. Despite demonstrating a certain level of zero-shot ability, MiDaS is limited by its data coverage, thus suffering disastrous performance in some scenarios.

In this work, our goal is to *build a foundation model for MDE* capable of producing high-quality depth information for any images under any circumstances. We approach this target from the perspective of dataset scaling-up.
Traditionally, depth datasets are created mainly by acquiring depth data from sensors [[[18](#bib.bib18), [55](#bib.bib55)]], stereo matching [[[15](#bib.bib15)]], or SfM [[[33](#bib.bib33)]], which is costly, time-consuming, or even intractable in particular situations.
We instead, for the first time, pay attention to large-scale unlabeled data. Compared with stereo images or labeled images from depth sensors, our used monocular unlabeled images exhibit three advantages:

(i) (*simple and cheap to acquire*) Monocular images exist almost everywhere, thus they are easy to collect, without requiring specialized devices.

(ii) (*diverse*) Monocular images can cover a broader range of scenes, which are critical to the model generalization ability and scalability.

(iii) (*easy to annotate*) We can simply use a pre-trained MDE model to assign depth labels for unlabeled images, which only takes a feedforward step. More than efficient, this also produces denser depth maps than LiDAR [[[18](#bib.bib18)]] and omits the computationally intensive stereo matching process.

We design a data engine to automatically generate depth annotations for unlabeled images, enabling data scaling-up to arbitrary scale. It collects 62M diverse and informative images from eight public large-scale datasets, *e.g*., SA-1B [[[27](#bib.bib27)]], Open Images [[[30](#bib.bib30)]], and BDD100K [[[82](#bib.bib82)]]. We use their raw unlabeled images without any forms of labels. Then, in order to provide a reliable annotation tool for our unlabeled images, we collect 1.5M labeled images from six public datasets to train an initial MDE model. The unlabeled images are then automatically annotated and jointly learned with labeled images in a self-training manner [[[31](#bib.bib31)]].

Despite all the aforementioned advantages of monocular unlabeled images, it is indeed not trivial to make positive use of such large-scale unlabeled images [[[73](#bib.bib73), [90](#bib.bib90)]], especially in the case of sufficient labeled images and strong pre-training models. In our preliminary attempts, directly combining labeled and pseudo labeled images failed to improve the baseline of solely using labeled images. We conjecture that, the additional knowledge acquired in such a naive self-teaching manner is rather limited. To address the dilemma, we propose to challenge the student model with a more difficult optimization target when learning the pseudo labels.
The student model is enforced to seek extra visual knowledge and learn robust representations under various strong perturbations to better handle unseen images.

Furthermore, there have been some works [[[9](#bib.bib9), [21](#bib.bib21)]] demonstrating the benefit of an auxiliary semantic segmentation task for MDE. We also follow this research line, aiming to equip our model with better high-level scene understanding capability. However, we observed when an MDE model is already powerful enough, it is hard for such an auxiliary task to bring further gains. We speculate that it is due to severe loss in semantic information when decoding an image into a discrete class space. Therefore, considering the excellent performance of DINOv2 in semantic-related tasks, we propose to maintain the rich semantic priors from it with a simple feature alignment loss. This not only enhances the MDE performance, but also yields a multi-task encoder for both middle-level and high-level perception tasks.

Our contributions are summarized as follows:

-
•

We highlight the value of data scaling-up of massive, cheap, and diverse unlabeled images for MDE.

-
•

We point out a key practice in jointly training large-scale labeled and unlabeled images. Instead of learning raw unlabeled images directly, we challenge the model with a harder optimization target for extra knowledge.

-
•

We propose to inherit rich semantic priors from pre-trained encoders for better scene understanding, rather than using an auxiliary semantic segmentation task.

-
•

Our model exhibits stronger zero-shot capability than MiDaS-BEiT${}_{\textrm{L-512}}$ [[[5](#bib.bib5)]]. Further, fine-tuned with metric depth, it outperforms ZoeDepth [[[4](#bib.bib4)]] significantly.

## 2 Related Work

Monocular depth estimation (MDE).
Early works [[[23](#bib.bib23), [37](#bib.bib37), [51](#bib.bib51)]] primarily relied on handcrafted features and traditional computer vision techniques. They were limited by their reliance on explicit depth cues and struggled to handle complex scenes with occlusions and textureless regions.

Deep learning-based methods have revolutionized monocular depth estimation by effectively learning depth representations from delicately annotated datasets [[[55](#bib.bib55), [18](#bib.bib18)]]. Eigen *et al*. [[[17](#bib.bib17)]] first proposed a multi-scale fusion network to regress the depth.
Following this, many works consistently improve the depth estimation accuracy by carefully designing the regression task as a classification task [[[3](#bib.bib3), [34](#bib.bib34)]], introducing more priors [[[32](#bib.bib32), [83](#bib.bib83), [76](#bib.bib76), [54](#bib.bib54)]], and better objective functions [[[68](#bib.bib68), [78](#bib.bib78)]], *etc*. Despite the promising performance, they are hard to generalize to unseen domains.

Zero-shot depth estimation.
Our work belongs to this research line. We aim to train an MDE model with a diverse training set and thus can predict the depth for any given image. Some pioneering works [[[10](#bib.bib10), [67](#bib.bib67)]] explored this direction
by collecting more training images, but their supervision is very sparse and is only enforced on limited pairs of points.

To enable effective multi-dataset joint training, a milestone work MiDaS [[[46](#bib.bib46)]] utilizes an affine-invariant loss to ignore the potentially different depth scales and shifts across varying datasets. Thus, MiDaS provides relative depth information. Recently, some works [[[79](#bib.bib79), [4](#bib.bib4), [22](#bib.bib22)]] take a step further to estimate the metric depth. However, in our practice, we observe such methods exhibit poorer generalization ability than MiDaS, especially its latest version [[[5](#bib.bib5)]]. Besides, as demonstrated by ZoeDepth [[[4](#bib.bib4)]], a strong relative depth estimation model can also work well in generalizable metric depth estimation by fine-tuning with metric depth information. Therefore, we still follow MiDaS in relative depth estimation, but further strengthen it by highlighting the value of large-scale monocular unlabeled images.

Leveraging unlabeled data.
This belongs to the research area of semi-supervised learning [[[31](#bib.bib31), [90](#bib.bib90), [56](#bib.bib56)]], which is popular with various applications [[[71](#bib.bib71), [75](#bib.bib75)]]. However, existing works typically assume only limited images are available. They rarely consider the challenging but realistic scenario where there are already sufficient labeled images but also larger-scale unlabeled images. We take this challenging direction for zero-shot MDE. We demonstrate that unlabeled images can significantly enhance the data coverage and thus improve model generalization and robustness.

## 3 Depth Anything

Our work utilizes both labeled and unlabeled images to facilitate better monocular depth estimation (MDE). Formally, the labeled and unlabeled sets are denoted as $\mathcal{D}^{l}=\{(x_{i},d_{i})\}_{i=1}^{M}$ and $\mathcal{D}^{u}=\{u_{i}\}_{i=1}^{N}$ respectively. We aim to learn a teacher model $T$ from $\mathcal{D}^{l}$. Then, we utilize $T$ to assign pseudo depth labels for $\mathcal{D}^{u}$. Finally, we train a student model $S$ on the combination of labeled set and pseudo labeled set. A brief illustration is provided in Figure [2](#S3.F2).

### 3.1 Learning Labeled Images

This process is similar to the training of MiDaS [[[46](#bib.bib46), [5](#bib.bib5)]]. However, since MiDaS did not release its code, we first reproduced it. Concretely, the depth value is first transformed into the disparity space by $d=1/t$ and then normalized to 0$\sim$1 on each depth map. To enable multi-dataset joint training, we adopt the affine-invariant loss to ignore the unknown scale and shift of each sample:

$$\mathcal{L}_{l}=\frac{1}{HW}\sum_{i=1}^{HW}\rho(d_{i}^{*},d_{i}),$$ \tag{1}

where $d^{*}_{i}$ and $d_{i}$ are the prediction and ground truth, respectively. And $\rho$ is the affine-invariant mean absolute error loss: $\rho(d^{*}_{i},d_{i})=|\hat{d}^{*}_{i}-\hat{d}_{i}|$, where $\hat{d}^{*}_{i}$ and $\hat{d}_{i}$ are the scaled and shifted versions of the prediction $d^{*}_{i}$ and ground truth $d_{i}$:

$$\hat{d}_{i}=\frac{d_{i}-t(d)}{s(d)},$$ \tag{2}

where $t(d)$ and $s(d)$ are used to align the prediction and ground truth to have zero translation and unit scale:

$$t(d)=\textrm{median}(d),\hskip 5.69054pts(d)=\frac{1}{HW}\sum_{i=1}^{HW}|d_{i}% -t(d)|.$$ \tag{3}

*Table 1: In total, our Depth Anything is trained on 1.5M labeled images and 62M unlabeled images jointly.*

*Figure 2: Our pipeline. Solid line: flow of labeled images, dotted line: unlabeled images. We especially highlight the value of large-scale unlabeled images. The S denotes adding strong perturbations (Section [3.2](#S3.SS2)). To equip our depth estimation model with rich semantic priors, we enforce an auxiliary constraint between the online student model and a frozen encoder to preserve the semantic capability (Section [3.3](#S3.SS3)).*

To obtain a robust monocular depth estimation model, we collect 1.5M labeled images from 6 public datasets. Details of these datasets are listed in Table [1](#S3.T1). We use fewer labeled datasets than MiDaS v3.1 [[[5](#bib.bib5)]] (12 training datasets), because 1) we do not use NYUv2 [[[55](#bib.bib55)]] and KITTI [[[18](#bib.bib18)]] datasets to ensure zero-shot evaluation on them, 2) some datasets are not available (anymore), e.g*., Movies [[[46](#bib.bib46)]] and WSVD [[[61](#bib.bib61)]], and 3) some datasets exhibit poor quality, *e.g*., RedWeb (also low resolution) [[[67](#bib.bib67)]]. Despite using fewer labeled images, our easy-to-acquire and diverse unlabeled images will comprehend the data coverage and greatly enhance the model generalization ability and robustness.

Furthermore, to strengthen the teacher model $T$ learned from these labeled images, we adopt the DINOv2 [[[43](#bib.bib43)]] pre-trained weights to initialize our encoder. In practice, we apply a pre-trained semantic segmentation model [[[70](#bib.bib70)]] to detect the sky region, and set its disparity value as 0 (farthest).

### 3.2 Unleashing the Power of Unlabeled Images

This is the main point of our work. Distinguished from prior works that laboriously construct diverse labeled datasets, we highlight the value of unlabeled images in enhancing the data coverage. Nowadays, we can practically build a diverse and large-scale unlabeled set from the Internet or public datasets of various tasks. Also, we can effortlessly obtain the dense depth map of monocular unlabeled images simply by forwarding them to a pre-trained well-performed MDE model. This is much more convenient and efficient than performing stereo matching or SfM reconstruction for stereo images or videos. We select eight large-scale public datasets as our unlabeled sources for their diverse scenes. They contain more than 62M images in total. The details are provided in the bottom half of Table [1](#S3.T1).

Technically, given the previously obtained MDE teacher model $T$, we make predictions on the unlabeled set $\mathcal{D}^{u}$ to obtain a pseudo labeled set $\hat{\mathcal{D}}^{u}$:

$$\hat{\mathcal{D}}^{u}=\{(u_{i},T(u_{i}))|u_{i}\in\mathcal{D}^{u}\}_{i=1}^{N}.$$ \tag{4}

With the combination set $\mathcal{D}^{l}\cup\hat{\mathcal{D}^{u}}$ of labeled images and pseudo labeled images, we train a student model $S$ on it. Following prior works [[[74](#bib.bib74)]], instead of fine-tuning $S$ from $T$, we re-initialize $S$ for better performance.

Unfortunately, in our pilot studies, we failed to gain improvements with such a self-training pipeline, which indeed contradicts the observations when there are only a few labeled images [[[56](#bib.bib56)]]. We conjecture that, with already sufficient labeled images in our case, the extra knowledge acquired from additional unlabeled images is rather limited. Especially considering the teacher and student share the same pre-training and architecture, they tend to make similar correct or false predictions on the unlabeled set $\mathcal{D}^{u}$, even without the explicit self-training procedure.

To address the dilemma, we propose to challenge the student with a more difficult optimization target for additional visual knowledge on unlabeled images. We inject strong perturbations to unlabeled images during training. It compels our student model to actively seek extra visual knowledge and acquire invariant representations from these unlabeled images. These advantages help our model deal with the open world more robustly. We introduce two forms of perturbations: one is strong color distortions, including color jittering and Gaussian blurring, and the other is strong spatial distortion, which is CutMix [[[84](#bib.bib84)]]. Despite the simplicity, the two modifications make our large-scale unlabeled images significantly improve the baseline of labeled images.

We provide more details about CutMix. It was originally proposed for image classification, and is rarely explored in monocular depth estimation. We first interpolate a random pair of unlabeled images $u_{a}$ and $u_{b}$ spatially:

$$u_{ab}=u_{a}\odot M+u_{b}\odot(1-M),$$ \tag{5}

where $M$ is a binary mask with a rectangle region set as 1.

The unlabeled loss $\mathcal{L}_{u}$ is obtained by first computing affine-invariant losses in valid regions defined by $M$ and $1-M$, respectively:

$$\mathcal{L}^{M}_{u}=\rho\big{(}S(u_{ab})\odot M,\,T(u_{a})\odot M\big{)},$$ \tag{6}
$$\mathcal{L}^{1-M}_{u}=\rho\big{(}S(u_{ab})\odot(1-M),T(u_{b})\odot(1-M)\big{)},$$ \tag{7}

where we omit the $\sum$ and pixel subscript $i$ for simplicity.
Then we aggregate the two losses via weighted averaging:

$$\mathcal{L}_{u}=\frac{\sum M}{HW}\mathcal{L}^{M}_{u}+\frac{\sum(1-M)}{HW}% \mathcal{L}^{1-M}_{u}.$$ \tag{8}

We use CutMix with 50% probability. The unlabeled images for CutMix are already strongly distorted in color, but the unlabeled images fed into the teacher model $T$ for pseudo labeling are clean, without any distortions.

*Table 2: Zero-shot relative depth estimation. Better: AbsRel $\downarrow$ , $\delta_{1}$ $\uparrow$. We compare with the best model from MiDaS v3.1. Note that MiDaS *does not* strictly follow the zero-shot evaluation on KITTI and NYUv2, because it uses their training images. We provide three model scales for different purposes, based on ViT-S (24.8M), ViT-B (97.5M), and ViT-L (335.3M), respectively. Best, second best results.*

### 3.3 Semantic-Assisted Perception

There exist some works [[[9](#bib.bib9), [21](#bib.bib21), [28](#bib.bib28), [72](#bib.bib72)]] improving depth estimation with an auxiliary semantic segmentation task. We believe that arming our depth estimation model with such high-level semantic-related information is beneficial. Besides, in our specific context of leveraging unlabeled images, these auxiliary supervision signals from other tasks can also combat the potential noise in our pseudo depth label.

Therefore, we made an initial attempt by carefully assigning semantic segmentation labels to our unlabeled images with a combination of RAM [[[86](#bib.bib86)]] + GroundingDINO [[[38](#bib.bib38)]] + HQ-SAM [[[26](#bib.bib26)]] models. After post-processing, this yields a class space containing 4K classes. In the joint-training stage, the model is enforced to produce both depth and segmentation predictions with a shared encoder and two individual decoders. Unfortunately, after trial and error, we still could not boost the performance of the original MDE model. We speculated that, decoding an image into a discrete class space indeed loses too much semantic information. The limited information in these semantic masks is hard to further boost our depth model, especially when our depth model has established very competitive results.

Therefore, we aim to seek more informative semantic signals to serve as auxiliary supervision for our depth estimation task. We are greatly astonished by the strong performance of DINOv2 models [[[43](#bib.bib43)]] in semantic-related tasks, *e.g*., image retrieval and semantic segmentation, even with frozen weights without any fine-tuning. Motivated by these clues, we propose to transfer its strong semantic capability to our depth model with an auxiliary feature alignment loss. The feature space is high-dimensional and continuous, thus containing richer semantic information than discrete masks. The feature alignment loss is formulated as:

$$\mathcal{L}_{feat}=1-\frac{1}{HW}\sum_{i=1}^{HW}\cos(f_{i},f^{\prime}_{i}),$$ \tag{9}

where $\cos(\cdot,\cdot)$ measures the cosine similarity between two feature vectors. $f$ is the feature extracted by the depth model $S$, while $f^{\prime}$ is the feature from a frozen DINOv2 encoder. We do not follow some works [[[19](#bib.bib19)]] to project the online feature $f$ into a new space for alignment, because a randomly initialized projector makes the large alignment loss dominate the overall loss in the early stage.

Another key point in feature alignment is that, semantic encoders like DINOv2 tend to produce similar features for different parts of an object, *e.g*., car front and rear. In depth estimation, however, different parts or even pixels within the same part, can be of varying depth. Thus, it is not beneficial to *exhaustively* enforce our depth model to produce exactly the same features as the frozen encoder.

To solve this issue, we set a tolerance margin $\alpha$ for the feature alignment. If the cosine similarity of $f_{i}$ and $f^{\prime}_{i}$ has surpassed $\alpha$, this pixel will not be considered in our $\mathcal{L}_{feat}$. This allows our method to enjoy both the semantic-aware representation from DINOv2 and the part-level discriminative representation from depth supervision. As a side effect, our produced encoder not only performs well in downstream MDE datasets, but also achieves strong results in the semantic segmentation task. It also indicates the potential of our encoder to serve as a universal multi-task encoder for both middle-level and high-level perception tasks.

Finally, our overall loss is an average combination of the three losses $\mathcal{L}_{l}$, $\mathcal{L}_{u}$, and $\mathcal{L}_{feat}$.

## 4 Experiment

### 4.1 Implementation Details

We adopt the DINOv2 encoder [[[43](#bib.bib43)]] for feature extraction. Following MiDaS [[[46](#bib.bib46), [5](#bib.bib5)]], we use the DPT [[[47](#bib.bib47)]] decoder for depth regression. All labeled datasets are simply combined together without re-sampling. In the first stage, we train a teacher model on labeled images for 20 epochs. In the second stage of joint training, we train a student model to sweep across all unlabeled images for one time. The unlabeled images are annotated by a best-performed teacher model with a ViT-L encoder. The ratio of labeled and unlabeled images is set as 1:2 in each batch. In both stages, the base learning rate of the pre-trained encoder is set as 5e-6, while the randomly initialized decoder uses a 10$\times$ larger learning rate. We use the AdamW optimizer and decay the learning rate with a linear schedule. We only apply horizontal flipping as our data augmentation for labeled images. The tolerance margin $\alpha$ for feature alignment loss is set as 0.85. For more details, please refer to our appendix.

### 4.2 Zero-Shot Relative Depth Estimation

As aforementioned, this work aims to provide accurate depth estimation for any image. Therefore, we comprehensively validate the zero-shot depth estimation capability of our Depth Anything model on six representative unseen datasets: KITTI [[[18](#bib.bib18)]], NYUv2 [[[55](#bib.bib55)]], Sintel [[[7](#bib.bib7)]], DDAD [[[20](#bib.bib20)]], ETH3D [[[52](#bib.bib52)]], and DIODE [[[60](#bib.bib60)]]. We compare with the best DPT-BEiT${}_{\textrm{L-512}}$ model from the latest MiDaS v3.1 [[[5](#bib.bib5)]], which uses more labeled images than us. As shown in Table [2](#S3.T2), both with a ViT-L encoder, our Depth Anything surpasses the strongest MiDaS model tremendously across extensive scenes in terms of both the AbsRel (absolute relative error: $|d^{*}-d|/d$) and $\delta_{1}$ (percentage of $\max(d^{*}/d,d/d^{*}), 0.127 *vs.* 0.076 in AbsRel and 0.850 *vs.* 0.947 in $\delta_{1}$ on KITTI.

### 4.3 Fine-tuned to Metric Depth Estimation

Apart from the impressive performance in zero-shot relative depth estimation, we further examine our Depth Anything model as a promising weight initialization for downstream *metric* depth estimation. We initialize the encoder of downstream MDE models with our pre-trained encoder parameters and leave the decoder randomly initialized. The model is fine-tuned with correponding metric depth information. In this part, we use our ViT-L encoder for fine-tuning.

We examine two representative scenarios: 1) *in-domain* metric depth estimation, where the model is trained and evaluated on the same domain (Section [4.3.1](#S4.SS3.SSS1)), and 2) *zero-shot* metric depth estimation, where the model is trained on one domain, *e.g*., NYUv2 [[[55](#bib.bib55)]], but evaluated in different domains, *e.g*., SUN RGB-D [[[57](#bib.bib57)]] (Section [4.3.2](#S4.SS3.SSS2)).

#### 4.3.1 In-Domain Metric Depth Estimation

As shown in Table [3](#S4.T3) of NYUv2 [[[55](#bib.bib55)]], our model outperforms the previous best method VPD [[[87](#bib.bib87)]] remarkably, improving the $\delta_{1}$ ($\uparrow$) from 0.964 $\rightarrow$ 0.984 and AbsRel ($\downarrow$) from 0.069 to 0.056. Similar improvements can be observed in Table [4](#S4.T4) of the KITTI dataset [[[18](#bib.bib18)]]. We improve the $\delta_{1}$ ($\uparrow$) on KITTI from 0.978 $\rightarrow$ 0.982. It is worth noting that we adopt the ZoeDepth framework for this scenario with a relatively basic depth model, and we believe our results can be further enhanced if equipped with more advanced architectures.

*Table 3: Fine-tuning and evaluating on NYUv2 [[[55](#bib.bib55)]] with our pre-trained MDE encoder. We highlight best, second best results, as well as most discriminative metrics. $*$: Reproduced by us.*

*Table 4: Fine-tuning and evaluating on KITTI [[[18](#bib.bib18)]] with our pre-trained MDE encoder. $*$: Reproduced by us.*

*Table 5: Zero-shot metric depth estimation. The first three test sets in the header are indoor scenes, while the last two are outdoor scenes. Following ZoeDepth, we use the model trained on NYUv2 for indoor generalization, while use the model trained on KITTI for outdoor evaluation. For fair comparisons, we report the ZoeDepth results reproduced in our environment.*

#### 4.3.2 Zero-Shot Metric Depth Estimation

We follow ZoeDepth [[[4](#bib.bib4)]] to conduct zero-shot metric depth estimation. ZoeDepth fine-tunes the MiDaS pre-trained encoder with metric depth information from NYUv2 [[[55](#bib.bib55)]] (for indoor scenes) or KITTI [[[18](#bib.bib18)]] (for outdoor scenes). Therefore, we simply replace the MiDaS encoder with our better Depth Anything encoder, leaving other components unchanged. As shown in Table [5](#S4.T5),
across a wide range of unseen datasets of indoor and outdoor scenes, our Depth Anything results in a better metric depth estimation model than the original ZoeDepth based on MiDaS.

*Table 6: Examine the zero-shot transferring performance of *each labeled training set* (left) to six unseen datasets (top). Better performance: AbsRel $\downarrow$ , $\delta_{1}$ $\uparrow$. We highlight the best, second, and *third best* results for each test dataset in bold, underline, and *italic*, respectively.*

*Table 7: Transferring our MDE pre-trained encoder to Cityscapes for semantic segmentation. We *do not* use Mapillary [[[1](#bib.bib1)]] for pre-training. s.s./m.s.: single-/multi-scale evaluation.*

### 4.4 Fine-tuned to Semantic Segmentation

In our method, we design our MDE model to inherit the rich semantic priors from a pre-trained encoder via a simple feature alignment constraint. Here, we examine the semantic capability of our MDE encoder. Specifically, we fine-tune our MDE encoder to downstream semantic segmentation datasets. As exhibited in Table [7](#S4.T7) of the Cityscapes dataset [[[15](#bib.bib15)]], our encoder from large-scale MDE training (86.2 mIoU) is superior to existing encoders from large-scale ImageNet-21K pre-training, *e.g*., Swin-L [[[39](#bib.bib39)]] (84.3) and ConvNeXt-XL [[[41](#bib.bib41)]] (84.6). Similar observations hold on the ADE20K dataset [[[89](#bib.bib89)]] in Table [8](#S4.T8). We improve the previous best result from 58.3 $\rightarrow$ 59.4.

We hope to highlight that, witnessing the superiority of our pre-trained encoder on both monocular depth estimation and semantic segmentation tasks, we believe it has great potential to serve as a generic multi-task encoder for both middle-level and high-level visual perception systems.

### 4.5 Ablation Studies

Unless otherwise specified, we use the ViT-L encoder for our ablation studies here.

Zero-shot transferring of *each* training dataset.
In Table [6](#S4.T6), we provide the zero-shot transferring performance of *each* training dataset, which means that we train a relative MDE model on *one* training set and evaluate it on the six unseen datasets. With these results, we hope to offer more insights for future works that similarly aim to build a general monocular depth estimation system. Among the six training datasets, HRWSI [[[68](#bib.bib68)]] fuels our model with the strongest generalization ability, even though it only contains 20K images. This indicates the data diversity counts a lot, which is well aligned with our motivation to utilize unlabeled images. Some labeled datasets may not perform very well, *e.g*., MegaDepth [[[33](#bib.bib33)]], however, it has its own preferences that are not reflected in these six test datasets. For example, we find models trained with MegaDepth data are specialized at estimating the distance of ultra-remote buildings (Figure [1](#S0.F1)), which will be very beneficial for aerial vehicles.

Effectiveness of 1) challenging the student model when learning unlabeled images, and 2) semantic constraint. As shown in Table [9](#S4.T9), simply adding unlabeled images with pseudo labels does not necessarily bring gains to our model, since the labeled images are already sufficient. However, with strong perturbations ($\mathcal{S}$) applied to unlabeled images during re-training, the student model is challenged to seek additional visual knowledge and learn more robust representations. Consequently, the large-scale unlabeled images enhance the model generalization ability significantly.

Moreover, with our used semantic constraint $\mathcal{L}_{feat}$, the power of unlabeled images can be further amplified for the depth estimation task. More importantly, as emphasized in Section [4.4](#S4.SS4), this auxiliary constraint also enables our trained encoder to serve as a key component in a multi-task visual system for both middle-level and high-level perception.

*Table 8: Transferring our MDE encoder to ADE20K for semantic segmentation. We use Mask2Former as our segmentation model.*

Comparison with MiDaS trained encoder in downstream tasks.
Our Depth Anything model has exhibited stronger zero-shot capability than MiDaS [[[46](#bib.bib46), [5](#bib.bib5)]].
Here, we further compare our trained encoder with MiDaS v3.1 [[[5](#bib.bib5)]] trained encoder in terms of the downstream fine-tuning performance. As demonstrated in Table [10](#S4.T10), on both the downstream depth estimation task and semantic segmentation task, our produced encoder outperforms the MiDaS encoder remarkably, *e.g*., 0.951 *vs.* 0.984 in the $\delta_{1}$ metric on NYUv2, and 52.4 *vs.* 59.4 in the mIoU metric on ADE20K.

*Figure 3: Qualitative results on six unseen datasets.*

*Table 9: Ablation studies of: 1) challenging the student with strong perturbations ($\mathcal{S}$) when learning unlabeled images, and 2) semantic constraint ($\mathcal{L}_{feat}$). Limited by space, we only report the AbsRel ($\downarrow$) metric, and shorten the dataset name with its first two letters.*

*Table 10: Comparison between our trained encoder and MiDaS [[[5](#bib.bib5)]] trained encoder in terms of downstream fine-tuning performance. Better performance: AbsRel $\downarrow$ , $\delta_{1}\uparrow$ , mIoU $\uparrow$ .*

Comparison with DINOv2 in downstream tasks.
We have demonstrated the superiority of our trained encoder when fine-tuned to downstream tasks. Since our finally produced encoder (from large-scale MDE training) is fine-tuned from DINOv2 [[[43](#bib.bib43)]], we compare our encoder with the original DINOv2 encoder in Table [11](#S4.T11). It can be observed that our encoder performs better than the original DINOv2 encoder in both the downstream metric depth estimation task and semantic segmentation task.
Although the DINOv2 weight has provided a very strong initialization, our large-scale and high-quality MDE training can further enhance it impressively in downstream transferring performance.

![Figure](x3.png)

*Figure 4: We compare our depth prediction with MiDaS. Meantime, we use ControlNet to synthesize new images from the depth map.*

*Table 11: Comparison between the original DINOv2 and our produced encoder in terms of downstream fine-tuning performance.*

### 4.6 Qualitative Results

We visualize our model predictions on the six unseen datasets in Figure [3](#S4.F3). Our model is robust to test images from various domains. In addition, we compare our model with MiDaS in Figure [4](#S4.F4). We also attempt to synthesis new images conditioned on the predicted depth maps with ControlNet [[[85](#bib.bib85)]]. Our model produces more accurate depth estimation than MiDaS, as well as better synthesis results. For more accurate synthesis, we re-trained a better depth-conditioned ControlNet based on our Depth Anything, aiming to provide better control signals for image synthesis and video editing. Please refer to our project page for more qualitative results on video editing [[[35](#bib.bib35)]] with our Depth Anything.

## 5 Conclusion

In this work, we present Depth Anything, a highly practical solution to robust monocular depth estimation. Different from prior arts, we especially highlight the value of cheap and diverse unlabeled images. We design two simple yet highly effective strategies to fully exploit their value: 1) posing a more challenging optimization target when learning unlabeled images, and 2) preserving rich semantic priors from pre-trained models. As a result, our Depth Anything model exhibits excellent zero-shot depth estimation ability, and also serves as a promising initialization for downstream metric depth estimation and semantic segmentation tasks.

Acknowledgement. This work is supported by the National Natural Science Foundation of China (No. 62201484), HKU Startup Fund, and HKU Seed Fund for Basic Research.

\thetitle

Supplementary Material

## 6 More Implementation Details

We resize the shorter side of all images to 518 and keep the original aspect ratio. All images are cropped to 518$\times$518 during training. During inference, we do not crop images and only ensure both sides are multipliers of 14, since the pre-defined patch size of DINOv2 encoders [[[43](#bib.bib43)]] is 14. Evaluation is performed at the original resolution by interpolating the prediction. Following MiDaS [[[46](#bib.bib46), [5](#bib.bib5)]], in zero-shot evaluation, the scale and shift of our prediction are manually aligned with the ground truth.

When fine-tuning our pre-trained encoder to metric depth estimation, we adopt the ZoeDepth codebase [[[4](#bib.bib4)]]. We merely replace the original MiDaS-based encoder with our stronger Depth Anything encoder, with a few hyper-parameters modified. Concretely, the training resolution is 392$\times$518 on NYUv2 [[[55](#bib.bib55)]] and 384$\times$768 on KITTI [[[18](#bib.bib18)]] to match the patch size of our encoder. The encoder learning rate is set as 1/50 of the learning rate of the randomly initialized decoder, which is much smaller than the 1/10 adopted for MiDaS encoder, due to our strong initialization. The batch size is 16 and the model is trained for 5 epochs.

When fine-tuning our pre-trained encoder to semantic segmentation, we use the MMSegmentation codebase [[[14](#bib.bib14)]]. The training resolution is set as 896$\times$896 on both ADE20K [[[89](#bib.bib89)]] and Cityscapes [[[15](#bib.bib15)]]. The encoder learning rate is set as 3e-6 and the decoder learning rate is 10$\times$ larger. We use Mask2Former [[[12](#bib.bib12)]] as our semantic segmentation model. The model is trained for 160K iterations on ADE20K and 80K iterations on Cityscapes both with batch size 16, without any COCO [[[36](#bib.bib36)]] or Mapillary [[[1](#bib.bib1)]] pre-training. Other training configurations are the same as the original codebase.

## 7 More Ablation Studies

All ablation studies here are conducted on the ViT-S model.

The necessity of tolerance margin for feature alignment.
As shown in Table [12](#S7.T12), the gap between the tolerance margin of 1.00 and 0.85 or 0.70 clearly demonstrates the necessity of this design (mean AbsRel: 0.188 vs.* 0.175).

Applying feature alignment to labeled data. Previously, we enforce the feature alignment loss $\mathcal{L}_{feat}$ on unlabeled data. Indeed, it is technically feasible to also apply this constraint to labeled data. In Table [13](#S7.T13), apart from applying $\mathcal{L}_{feat}$ on unlabeled data, we explore to apply it to labeled data. We find that adding this auxiliary optimization target to labeled data is not beneficial to our baseline that does not involve any feature alignment (their mean AbsRel values are almost the same: 0.180 *vs.* 0.179). We conjecture that this is because the labeled data has relatively higher-quality depth annotations. The involvement of semantic loss may interfere with the learning of these informative manual labels. In comparison, our pseudo labels are noisier and less informative. Therefore, introducing the auxiliary constraint to unlabeled data can combat the noise in pseudo depth labels, as well as arm our model with semantic capability.

*Table 12: Ablation studies on different values of the tolerance margin $\alpha$ for the feature alignment loss $\mathcal{L}_{feat}$. Limited by space, we only report the AbsRel ($\downarrow$) metric here.*

*Table 13: Ablation studies of applying our feature alignment loss $\mathcal{L}_{feat}$ to unlabeled data (U) or labeled data (L).*

## 8 Limitations and Future Works

Currently, the largest model size is only constrained to ViT-Large [[[16](#bib.bib16)]]. Therefore, in the future, we plan to further scale up the model size from ViT-Large to ViT-Giant, which is also well pre-trained by DINOv2 [[[43](#bib.bib43)]]. We can train a more powerful teacher model with the larger model, producing more accurate pseudo labels for smaller models to learn, *e.g*., ViT-L and ViT-B. Furthermore, to facilitate real-world applications, we believe the widely adopted 512$\times$512 training resolution is not enough. We plan to re-train our model on a larger resolution of 700+ or even 1000+.

## 9 More Qualitative Results

Please refer to the following pages for comprehensive qualitative results on six unseen test sets (Figure [5](#S9.F5) for KITTI [[[18](#bib.bib18)]], Figure [6](#S9.F6) for NYUv2 [[[55](#bib.bib55)]], Figure [7](#S9.F7) for Sintel [[[7](#bib.bib7)]], Figure [8](#S9.F8) for DDAD [[[20](#bib.bib20)]], Figure [9](#S9.F9) for ETH3D [[[52](#bib.bib52)]], and Figure [10](#S9.F10) for DIODE [[[60](#bib.bib60)]]). We compare our model with the strongest MiDaS model [[[5](#bib.bib5)]], *i.e*., DPT-BEiT${}_{\textrm{L-512}}$. Our model exhibits higher depth estimation accuracy and stronger robustness. Please refer to our project page for more visualizations.

*Figure 5: Qualitative results on KITTI. Due to the extremely sparse ground truth which is hard to visualize, we here compare our prediction with the most advanced MiDaS v3.1 [[[5](#bib.bib5)]] prediction. The brighter color denotes the closer distance.*

![Figure](x5.png)

*Figure 6: Qualitative results on NYUv2. It is worth noting that MiDaS [[[5](#bib.bib5)]] uses NYUv2 training data (not zero-shot*), while we do not.*

*Figure 7: Qualitative results on Sintel.*

![Figure](x7.png)

*Figure 8: Qualitative results on DDAD.*

![Figure](x8.png)

*Figure 9: Qualitative results on ETH3D.*

![Figure](x9.png)

*Figure 10: Qualitative results on DIODE.*

## References

-
Antequera et al. [2020]

Manuel López Antequera, Pau Gargallo, Markus Hofinger, Samuel Rota Bulò, Yubin Kuang, and Peter Kontschieder.

Mapillary planet-scale depth dataset.

In ECCV*, 2020.

-
Bao et al. [2022]

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei.

Beit: Bert pre-training of image transformers.

In *ICLR*, 2022.

-
Bhat et al. [2021]

Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka.

Adabins: Depth estimation using adaptive bins.

In *CVPR*, 2021.

-
Bhat et al. [2023]

Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller.

Zoedepth: Zero-shot transfer by combining relative and metric depth.

*arXiv:2302.12288*, 2023.

-
Birkl et al. [2023]

Reiner Birkl, Diana Wofk, and Matthias Müller.

Midas v3. 1–a model zoo for robust monocular relative depth estimation.

*arXiv:2307.14460*, 2023.

-
Bommasani et al. [2021]

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.

On the opportunities and risks of foundation models.

*arXiv:2108.07258*, 2021.

-
Butler et al. [2012]

Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black.

A naturalistic open source movie for optical flow evaluation.

In *ECCV*, 2012.

-
Cabon et al. [2020]

Yohann Cabon, Naila Murray, and Martin Humenberger.

Virtual kitti 2.

*arXiv:2001.10773*, 2020.

-
Chen et al. [2019]

Po-Yi Chen, Alexander H Liu, Yen-Cheng Liu, and Yu-Chiang Frank Wang.

Towards scene understanding: Unsupervised monocular depth estimation with semantic-aware representation.

In *CVPR*, 2019.

-
Chen et al. [2016]

Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng.

Single-image depth perception in the wild.

In *NeurIPS*, 2016.

-
Chen et al. [2023]

Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao.

Vision transformer adapter for dense predictions.

In *ICLR*, 2023.

-
Cheng et al. [2022]

Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar.

Masked-attention mask transformer for universal image segmentation.

In *CVPR*, 2022.

-
Cho et al. [2021]

Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn.

Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes.

*arXiv:2110.11590*, 2021.

-
Contributors [2020]

MMSegmentation Contributors.

MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark.

[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020.

-
Cordts et al. [2016]

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele.

The cityscapes dataset for semantic urban scene understanding.

In *CVPR*, 2016.

-
Dosovitskiy et al. [2021]

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.

An image is worth 16x16 words: Transformers for image recognition at scale.

In *ICLR*, 2021.

-
Eigen et al. [2014]

David Eigen, Christian Puhrsch, and Rob Fergus.

Depth map prediction from a single image using a multi-scale deep network.

In *NeurIPS*, 2014.

-
Geiger et al. [2013]

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun.

Vision meets robotics: The kitti dataset.

*IJRR*, 2013.

-
Grill et al. [2020]

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al.

Bootstrap your own latent-a new approach to self-supervised learning.

In *NeurIPS*, 2020.

-
Guizilini et al. [2020a]

Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon.

3d packing for self-supervised monocular depth estimation.

In *CVPR*, 2020a.

-
Guizilini et al. [2020b]

Vitor Guizilini, Rui Hou, Jie Li, Rares Ambrus, and Adrien Gaidon.

Semantically-guided representation learning for self-supervised monocular depth.

In *ICLR*, 2020b.

-
Guizilini et al. [2023]

Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rare\textcommabelows Ambru\textcommabelows, and Adrien Gaidon.

Towards zero-shot scale-aware monocular depth estimation.

In *ICCV*, 2023.

-
Hoiem et al. [2007]

Derek Hoiem, Alexei A Efros, and Martial Hebert.

Recovering surface layout from an image.

*IJCV*, 2007.

-
Jain et al. [2023]

Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi.

Oneformer: One transformer to rule universal image segmentation.

In *CVPR*, 2023.

-
Ji et al. [2023]

Yuanfeng Ji, Zhe Chen, Enze Xie, Lanqing Hong, Xihui Liu, Zhaoqiang Liu, Tong Lu, Zhenguo Li, and Ping Luo.

Ddp: Diffusion model for dense visual prediction.

In *ICCV*, 2023.

-
Ke et al. [2023]

Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu.

Segment anything in high quality.

In *NeurIPS*, 2023.

-
Kirillov et al. [2023]

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.

Segment anything.

In *ICCV*, 2023.

-
Klingner et al. [2020]

Marvin Klingner, Jan-Aike Termöhlen, Jonas Mikolajczyk, and Tim Fingscheidt.

Self-supervised monocular depth estimation: Solving the dynamic object problem by semantic guidance.

In *ECCV*, 2020.

-
Koch et al. [2018]

Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner.

Evaluation of cnn-based single-image depth estimation methods.

In *ECCVW*, 2018.

-
Kuznetsova et al. [2020]

Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al.

The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.

*IJCV*, 2020.

-
Lee et al. [2013]

Dong-Hyun Lee et al.

Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.

In *ICMLW*, 2013.

-
Li et al. [2015]

Bo Li, Chunhua Shen, Yuchao Dai, Anton Van Den Hengel, and Mingyi He.

Depth and surface normal estimation from monocular images using regression on deep features and hierarchical crfs.

In *CVPR*, 2015.

-
Li and Snavely [2018]

Zhengqi Li and Noah Snavely.

Megadepth: Learning single-view depth prediction from internet photos.

In *CVPR*, 2018.

-
Li et al. [2022]

Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang.

Binsformer: Revisiting adaptive bins for monocular depth estimation.

*arXiv:2204.00987*, 2022.

-
Liew et al. [2023]

Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng.

Magicedit: High-fidelity and temporally coherent video editing.

*arXiv:2308.14749*, 2023.

-
Lin et al. [2014]

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.

Microsoft coco: Common objects in context.

In *ECCV*, 2014.

-
Liu et al. [2008]

Ce Liu, Jenny Yuen, Antonio Torralba, Josef Sivic, and William T Freeman.

Sift flow: Dense correspondence across different scenes.

In *ECCV*, 2008.

-
Liu et al. [2023]

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al.

Grounding dino: Marrying dino with grounded pre-training for open-set object detection.

*arXiv:2303.05499*, 2023.

-
Liu et al. [2021]

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.

Swin transformer: Hierarchical vision transformer using shifted windows.

In *ICCV*, 2021.

-
Liu et al. [2022a]

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al.

Swin transformer v2: Scaling up capacity and resolution.

In *CVPR*, 2022a.

-
Liu et al. [2022b]

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.

A convnet for the 2020s.

In *CVPR*, 2022b.

-
Ning et al. [2023]

Jia Ning, Chen Li, Zheng Zhang, Chunyu Wang, Zigang Geng, Qi Dai, Kun He, and Han Hu.

All in tokens: Unifying output space of visual tasks via soft token.

In *ICCV*, 2023.

-
Oquab et al. [2023]

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.

Dinov2: Learning robust visual features without supervision.

*TMLR*, 2023.

-
Patil et al. [2022]

Vaishakh Patil, Christos Sakaridis, Alexander Liniger, and Luc Van Gool.

P3depth: Monocular depth estimation with a piecewise planarity prior.

In *CVPR*, 2022.

-
Radford et al. [2021]

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.

Learning transferable visual models from natural language supervision.

In *ICML*, 2021.

-
Ranftl et al. [2020]

René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun.

Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.

*TPAMI*, 2020.

-
Ranftl et al. [2021]

René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun.

Vision transformers for dense prediction.

In *ICCV*, 2021.

-
Rasla and Beyeler [2022]

Alex Rasla and Michael Beyeler.

The relative importance of depth cues and semantic edges for indoor mobility using simulated prosthetic vision in immersive virtual reality.

In *VRST*, 2022.

-
Roberts et al. [2021]

Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind.

Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding.

In *ICCV*, 2021.

-
Russakovsky et al. [2015]

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al.

Imagenet large scale visual recognition challenge.

*IJCV*, 2015.

-
Saxena et al. [2008]

Ashutosh Saxena, Min Sun, and Andrew Y Ng.

Make3d: Learning 3d scene structure from a single still image.

*TPAMI*, 2008.

-
Schops et al. [2017]

Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger.

A multi-view stereo benchmark with high-resolution images and multi-camera videos.

In *CVPR*, 2017.

-
Shao et al. [2019]

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun.

Objects365: A large-scale, high-quality dataset for object detection.

In *ICCV*, 2019.

-
Shao et al. [2023]

Shuwei Shao, Zhongcai Pei, Weihai Chen, Xingming Wu, and Zhengguo Li.

Nddepth: Normal-distance assisted monocular depth estimation.

In *ICCV*, 2023.

-
Silberman et al. [2012]

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus.

Indoor segmentation and support inference from rgbd images.

In *ECCV*, 2012.

-
Sohn et al. [2020]

Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.

Fixmatch: Simplifying semi-supervised learning with consistency and confidence.

In *NeurIPS*, 2020.

-
Song et al. [2015]

Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao.

Sun rgb-d: A rgb-d scene understanding benchmark suite.

In *CVPR*, 2015.

-
Strudel et al. [2021]

Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid.

Segmenter: Transformer for semantic segmentation.

In *ICCV*, 2021.

-
Touvron et al. [2023]

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.

Llama: Open and efficient foundation language models.

*arXiv:2302.13971*, 2023.

-
Vasiljevic et al. [2019]

Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al.

Diode: A dense indoor and outdoor depth dataset.

*arXiv:1908.00463*, 2019.

-
Wang et al. [2019a]

Chaoyang Wang, Simon Lucey, Federico Perazzi, and Oliver Wang.

Web stereo video supervision for depth prediction from dynamic scenes.

In *3DV*, 2019a.

-
Wang et al. [2021]

Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu.

Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation.

In *ICME*, 2021.

-
Wang et al. [2020]

Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer.

Tartanair: A dataset to push the limits of visual slam.

In *IROS*, 2020.

-
Wang et al. [2019b]

Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, and Kilian Q Weinberger.

Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving.

In *CVPR*, 2019b.

-
Weyand et al. [2020]

Tobias Weyand, Andre Araujo, Bingyi Cao, and Jack Sim.

Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval.

In *CVPR*, 2020.

-
Wofk et al. [2019]

Diana Wofk, Fangchang Ma, Tien-Ju Yang, Sertac Karaman, and Vivienne Sze.

Fastdepth: Fast monocular depth estimation on embedded systems.

In *ICRA*, 2019.

-
Xian et al. [2018]

Ke Xian, Chunhua Shen, Zhiguo Cao, Hao Lu, Yang Xiao, Ruibo Li, and Zhenbo Luo.

Monocular relative depth perception with web stereo data supervision.

In *CVPR*, 2018.

-
Xian et al. [2020]

Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao.

Structure-guided ranking loss for single image depth prediction.

In *CVPR*, 2020.

-
Xiao et al. [2018]

Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun.

Unified perceptual parsing for scene understanding.

In *ECCV*, 2018.

-
Xie et al. [2021]

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo.

Segformer: Simple and efficient design for semantic segmentation with transformers.

In *NeurIPS*, 2021.

-
Xu et al. [2021]

Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan Wang, Fangyun Wei, Xiang Bai, and Zicheng Liu.

End-to-end semi-supervised object detection with soft teacher.

In *ICCV*, 2021.

-
Xu et al. [2022]

Xiaogang Xu, Hengshuang Zhao, Vibhav Vineet, Ser-Nam Lim, and Antonio Torralba.

Mtformer: Multi-task learning via transformer and cross-task reasoning.

In *ECCV*, 2022.

-
Yalniz et al. [2019]

I Zeki Yalniz, Hervé Jégou, Kan Chen, Manohar Paluri, and Dhruv Mahajan.

Billion-scale semi-supervised learning for image classification.

*arXiv:1905.00546*, 2019.

-
Yang et al. [2022]

Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, and Yang Gao.

St++: Make self-training work better for semi-supervised semantic segmentation.

In *CVPR*, 2022.

-
Yang et al. [2023a]

Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, and Yinghuan Shi.

Revisiting weak-to-strong consistency in semi-supervised semantic segmentation.

In *CVPR*, 2023a.

-
Yang et al. [2023b]

Xiaodong Yang, Zhuang Ma, Zhiyu Ji, and Zhe Ren.

Gedepth: Ground embedding for monocular depth estimation.

In *ICCV*, 2023b.

-
Yao et al. [2020]

Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan.

Blendedmvs: A large-scale dataset for generalized multi-view stereo networks.

In *CVPR*, 2020.

-
Yin et al. [2019]

Wei Yin, Yifan Liu, Chunhua Shen, and Youliang Yan.

Enforcing geometric constraints of virtual normal for depth prediction.

In *ICCV*, 2019.

-
Yin et al. [2023]

Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen.

Metric3d: Towards zero-shot metric 3d prediction from a single image.

In *ICCV*, 2023.

-
You et al. [2020]

Yurong You, Yan Wang, Wei-Lun Chao, Divyansh Garg, Geoff Pleiss, Bharath Hariharan, Mark Campbell, and Kilian Q Weinberger.

Pseudo-lidar++: Accurate depth for 3d object detection in autonomous driving.

In *ICLR*, 2020.

-
Yu et al. [2015]

Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao.

Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop.

*arXiv:1506.03365*, 2015.

-
Yu et al. [2020]

Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell.

Bdd100k: A diverse driving dataset for heterogeneous multitask learning.

In *CVPR*, 2020.

-
Yuan et al. [2022]

Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, and Ping Tan.

New crfs: Neural window fully-connected crfs for monocular depth estimation.

*arXiv:2203.01502*, 2022.

-
Yun et al. [2019]

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo.

Cutmix: Regularization strategy to train strong classifiers with localizable features.

In *ICCV*, 2019.

-
Zhang et al. [2023a]

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

Adding conditional control to text-to-image diffusion models.

In *ICCV*, 2023a.

-
Zhang et al. [2023b]

Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al.

Recognize anything: A strong image tagging model.

*arXiv:2306.03514*, 2023b.

-
Zhao et al. [2023]

Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu.

Unleashing text-to-image diffusion models for visual perception.

In *ICCV*, 2023.

-
Zhou et al. [2017a]

Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba.

Places: A 10 million image database for scene recognition.

*TPAMI*, 2017a.

-
Zhou et al. [2017b]

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba.

Scene parsing through ade20k dataset.

In *CVPR*, 2017b.

-
Zoph et al. [2020]

Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin Dogus Cubuk, and Quoc Le.

Rethinking pre-training and self-training.

In *NeurIPS*, 2020.

Generated on Sun Apr 7 06:49:28 2024 by [LATExml](http://dlmf.nist.gov/LaTeXML/)