[\CVMsetup type = Review Article, doi = CVM.XXXX, title = Towards Depth Foundation Model: Recent Trends in Vision-Based Depth Estimation, author = Zhen Xu∗,1, Hongyu Zhou∗,1, Sida Peng1, Haotong Lin1, Haoyu Guo2, Jiahao Shao1, Peishan Yang1, Qinglin Yang1, Sheng Miao1, Xingyi He1, Yifan Wang1, Yue Wang1, Ruizhen Hu3, Yiyi Liao1, Xiaowei Zhou1, Hujun Bao1\cor, runauthor = Z. Xu, H. Zhou et al., abstract = Depth estimation is a fundamental task in 3D computer vision, crucial for applications such as 3D reconstruction, free-viewpoint rendering, robotics, autonomous driving, and AR/VR technologies. Traditional methods relying on hardware sensors like LiDAR are often limited by high costs, low resolution, and environmental sensitivity, limiting their applicability in real-world scenarios. Recent advances in vision-based methods offer a promising alternative, yet they face challenges in generalization and stability due to either the low-capacity model architectures or the reliance on domain-specific and small-scale datasets. The emergence of scaling laws and foundation models in other domains has inspired the development of ”depth foundation models”—deep neural networks trained on large datasets with strong zero-shot generalization capabilities. This paper surveys the evolution of deep learning architectures and paradigms for depth estimation across the monocular, stereo, multi-view, and monocular video settings. We explore the potential of these models to address existing challenges and provide a comprehensive overview of large-scale datasets that can facilitate their development. By identifying key architectures and training strategies, we aim to highlight the path towards robust depth foundation models, offering insights into their future research and applications. , keywords = Depth Estimation, Foundation Models, 3D Computer Vision, copyright = Zhejiang University, ∗ The first two authors contributed equally to this work. $1\qquad$ Zhejiang University, China. $2\qquad$ Shanghai AI Lab, China. $3\qquad$ Shenzhen University, China. Authors’ emails: zhenx@zju.edu.cn, hongy_zhou@zju.edu.cn, pengsida@zju.edu.cn, haotongl@outlook.com, guohaoyu@pjlab.org.cn, jhshao@zju.edu.cn, yangps0306@gmail.com, yangqinglin@zju.edu.cn, 12231099@zju.edu.cn, hexingyi8@gmail.com, accwyf@gmail.com, ywang24@zju.edu.cn, ruizhen.hu@szu.edu.cn, yiyi.liao@zju.edu.cn, xwzhou@zju.edu.cn, bao@cad.zju.edu.cn Manuscript received: 2025-01-01; accepted: 2025-01-01 ## 1 Introduction * Figure 1: Scaling trends in model capacity and data volume for depth estimation. Each point represents a published method, positioned by its approximate model size (bottom axis, logarithmic scale of parameter count) and dataset size (left axis, logarithmic scale of training images), and colored by publication year. Early models (lightest shading, before 2020) relied on sub-million-parameter networks trained on only thousands of images, yielding limited generalization. Over 2020–2021 (light blue shading), methods grew to tens of millions of parameters and hundreds of thousands of image datasets. In 2022-2023 (gray shading), methods continued to increase in size and scalable training, with around one billion parameters and datasets ranging from several million to ten million images. The most recent models (darkest shading, 2024–2025) further scale up to the billions of parameters and utilize over ten million images, with strong generalization ability, and demonstrate the potential evolution toward depth foundation models. Domain Tasks Dataset Name Dynamic Metric Scenes# Frames# Resolution Anno License Mono Stereo MV Video Real-World ✔ A2D2 [1](#bib.bib1)]

Dynamic
Yes
3
394K
$1920\times 1208$
Sparse
CC BY-ND 4.0

✔
✔

Arogoverse2 [[[2](#bib.bib2), [3](#bib.bib3)]]

Dynamic
Yes
1000
2.14M
$2048\times 1550$
Sparse
CC BY-NC-SA 4.0

✔
✔

CityScapes [[[4](#bib.bib4)]]

Dynamic
Yes
2975
89K
$2048\times 1024$
Sparse
Non-Commercial

✔

DDAD [[[5](#bib.bib5)]]

Dynamic
Yes
200
98K
$1936\times 1216$
Sparse
CC BY-NC-SA 4.0

✔
✔

DIML [[[6](#bib.bib6)]]

Static
Yes
200+
2M
$1334\times 756$
Sparse
Non-Commercial

✔

DIODE [[[7](#bib.bib7)]]

Static
Yes
30
27K
$1024\times 768$
Sparse
MIT

✔
✔

DSEC [[[8](#bib.bib8)]]

Dynamic
Yes
53
26K
$1440\times 1080$
Sparse
CC BY-NC-SA 4.0

✔

HM3D [[[9](#bib.bib9)]]

Static
Yes
1000
†
†
Dense
MIT

✔

iBims-1 [[[10](#bib.bib10)]]

Static
Yes
20
54
$640\times 480$
Sparse
CC BY 4.0

✔

Lyft [[[11](#bib.bib11)]]

Dynamic
Yes
366
158K
$1224\times 1024$
Sparse
CC0 1.0

✔

Mapillary PSD [[[12](#bib.bib12)]]

Dynamic
Yes
50K
750K
$1920\times 1080$
Sparse
CC BY-NC-SA

✔

NuScenes [[[13](#bib.bib13)]]

Dynamic
Yes
1K
40K
$1600\times 900$
Sparse
Non-Commercial

✔

NYU [[[14](#bib.bib14)]]

Static
Yes
464
435K
$640\times 480$
Dense
GPL

✔

Pandaset [[[15](#bib.bib15)]]

Dynamic
Yes
103
8K
$1920\times 1080$
Sparse
CC0

✔

Taskonomy[[[16](#bib.bib16)]]

Static
Yes
600
4.5M
$512\times 512$
Dense
MIT

✔
✔

KITTI [[[17](#bib.bib17)]]

Dynamic
Yes
22
41K
$1242\times 375$
Sparse
CC BY-NC-SA 3.0

✔
✔

DrivingStereo [[[18](#bib.bib18)]]

Dynamic
Yes
184
180K
$1762\times 800$
Sparse
MIT

✔

InStereo2K [[[19](#bib.bib19)]]

Static
Yes
50
2K
$1080\times 860$
Dense
Non-Commercial

✔
✔

Argoverse [[[20](#bib.bib20)]]

Dynamic
Yes
113
6,624
$2056\times 2464$
Sparse
CC BY-NC-SA 4.0

✔
✔

ETH3D [[[21](#bib.bib21)]]

Static
Yes
25
1K
$6233\times 4146$
Dense
CC BY-NC-SA 4.0

✔
✔
Waymo [[[22](#bib.bib22)]]

Dynamic
Yes
1,150
160K
$1920\times 1280$
Sparse
Non-Commercial

✔
✔
UASOL [[[23](#bib.bib23)]]

Static
Yes
676
160.9K
$2280\times 1282$
Dense
CC BY-NC-SA 3.0

✔

DTU [[[24](#bib.bib24)]]

Static
Yes
124
42.5K
$1600\times 1200$
Dense
MIT

✔

BlendedMVS [[[25](#bib.bib25)]]

Static
No
113
17k
$2048\times 1536$
Dense
CC BY 4.0

✔

✔
WildRGBD [[[26](#bib.bib26)]]

Static
Yes
23K
6M
$480\times 640$
Dense
MIT

✔

✔
MVImgNet [[[27](#bib.bib27)]]

Static
No
220K
6.8M
$1080\times 1920$
Dense
CC BY-NC 4.0

✔

✔
ARKitScenes [[[28](#bib.bib28)]]

Static
Yes
5,047
450M
$256\times 192$
Dense
Non-Commercial

✔

✔
ARKitScenes-HighRes [[[28](#bib.bib28)]]

Static
Yes
5,047
450M
$1920\times 1440$
Dense
Non-Commercial

✔

✔
Matterport3D [[[29](#bib.bib29)]]

Static
Yes
90
194.4K
$1280\times 1024$
Dense
MIT

✔
✔

Replica[[[30](#bib.bib30)]]

Static
Yes
18
36K
$1200\times 680$
Dense
Non-Commercial

✔
✔
✔
✔
Dynamic Replica [[[31](#bib.bib31)]]

Dynamic
Yes
484
145K
$1280\times 720$
Dense
CC BY-NC 4.0

✔

✔
ScanNet [[[32](#bib.bib32)]]

Static
Yes
1,513
2.5M
$640\times 480$
Dense
Non-Commercial

✔

✔
ScanNet++ [[[33](#bib.bib33)]]

Static
Yes
1,858
3.7M+
$1920\times 1440$
Dense
Non-Commercial

Synthetic

✔

✔

Sintel [[[34](#bib.bib34)]]

Dynamic
No
10
1K
$1024\times 436$
Dense
CC BY 3.0

✔

SceneFlow [[[35](#bib.bib35)]]

Dynamic
No
9
40K
$960\times 540$
Dense
CC BY 4.0

✔

CREStereo [[[36](#bib.bib36)]]

Static
No
0
103K
$1920\times 1080$
Dense
Apache-2.0

✔

FallingThings [[[37](#bib.bib37)]]

Dynamic
No
3
62K
$960\times 540$
Dense
CC BY-NC-SA 4.0

✔

FSD [[[38](#bib.bib38)]]

Dynamic
No
12
1M
$1280\times 720$
Dense
Non-Commercial

✔

UnrealStereo4K [[[39](#bib.bib39)]]

Static
No
8
7,720
$3840\times 2160$
Dense
MIT

✔
✔
✔
Spring [[[40](#bib.bib40)]]

Dynamic
No
47
6K
$1920\times 1080$
Dense
CC BY 4.0

✔
✔
✔
TartanAir [[[41](#bib.bib41)]]

Static
Yes
163
1M
$640\times 480$
Dense
CC BY 4.0

✔
✔
✔
VirtualKITTI2 [[[42](#bib.bib42)]]

Dynamic
Yes
5
25K
$1242\times 375$
Dense
CC BY-NC-SA 4.0

✔
✔
MatrixCity [[[43](#bib.bib43)]]

Static
Yes
3K
519K
$1000\times 1000$
Dense
Apache-2.0

✔

✔
✔
MVS-Synth [[[44](#bib.bib44)]]

Dynamic
No
120
12K
$1920\times 1080$
Dense
Non-Commercial

✔

✔

3D Ken Burns [[[45](#bib.bib45)]]

Static
No
32
536K
$512\times 512$
Dense
CC BY-NC-SA 4.0

✔

✔
OmniObject3D [[[46](#bib.bib46)]]

Static
Yes
6K
600K
$800\times 800$
Dense
CC BY 4.0

✔
✔

✔
IRS [[[47](#bib.bib47)]]

Static
Yes
70
100K
$960\times 540$
Dense
Apache 2.0

✔
✔
PointOdyssey [[[48](#bib.bib48)]]

Dynamic
No
131
200K
$540\times 960$
Dense
MIT

✔
✔
BEDLAM [[[49](#bib.bib49)]]

Dynamic
Yes
10,450
380K
$1280\times 720$
Dense
Non-Commercial

Table 1: Datasets. The term indicates that the scenes in these datasets are object-centric. Similarly, and refer to indoor and outdoor scenes, respectively. The depth annotations can be classified as dense or sparse, depending on whether most pixels have corresponding depth values. †: HM3D is a digital twin dataset created from real-world data, with the number of frames and resolutions being user-defined.

Depth estimation stands as a cornerstone in the field of 3D computer vision.
This task has been a focal point for researchers due to its critical role in various applications such as 3D reconstruction, 3D generative models, robotics, autonomous driving, and AR/VR technologies.
However, algorithms often struggle to achieve high-quality and consistent depth recovery akin to human perception, which leverages prior knowledge of the scene and the world.
Traditional depth recovery methods typically rely on active sensing hardware, including commercially available LiDAR, time-of-flight (ToF) sensors, and ultrasonic probes.
These sensors estimate depth by measuring the time taken for photons or sound waves to travel back and forth.
Despite their accuracy, the high cost of these sensors limits their widespread application.
Additionally, active sensors often suffer from low resolution and significant noise interference.
For instance, the LiDAR sensor on an iPhone can only achieve a reconstruction resolution within a limited range and struggles with high precision for very close or distant objects.
Moreover, these sensors are sensitive to environmental lighting conditions, making them less effective in outdoor, high-light scenarios.

Recently, there has been a growing interest in vision-based depth estimation methods that avoid active depth-sensing hardware, instead relying on readily available cameras commonly found in everyday devices.
Compared to active sensor-based approaches, vision methods are cost-effective, offer an unrestricted depth range, are less affected by environmental conditions, and provide high resolution.
For example, a standard iPhone camera can easily capture 4K resolution RGB information.
However, existing vision-based depth estimation algorithms still face numerous challenges.
Monocular depth estimation, in particular, is highly ill-posed, and standard deep learning algorithms struggle to achieve high-precision results.
To introduce constraints and reduce ill-posedness, researchers often explore depth estimation algorithms with multiple camera inputs or scenarios with more extensive scene observations, such as stereo, multi-view, or video-based depth estimation.
Yet, these methods are often trained on small-scale synthetic data, leading to instability across spatial and temporal domains, poor generalization across different scenes and input types, and difficulties in overcoming the domain gap between synthetic and real-world data.

With the validation and rise of scaling laws in natural language processing, text-based image generation, and video generation models, the concept of foundation models has emerged.
Foundation models are deep neural networks trained on vast amounts of data, exhibiting emergent zero-shot generalization capabilities in other domains.
To achieve such capabilities, researchers focus on the scale and variation of input training data, leveraging large-scale models from other domains and ingeniously constructing self-supervision architectures.
We define scalable depth estimation models and architectures capable of absorbing massive data as ”depth foundation models”.
For depth estimation tasks, including monocular, stereo, multi-view, and monocular video depth estimation, corresponding foundation models have the potential to address the aforementioned generalization issues and provide key solutions to long-standing Computer Vision tasks.

This paper aims to survey the evolution towards depth foundation models and paradigms for depth estimation across the monocular, stereo, multi-view, and monocular video settings.

-
•

We explore the development of deep learning model architectures and learning paradigms for each task and identify key paradigms with foundational capability or potential.

-
•

To aid the development of such depth foundation models, we also provide comprehensive surveys on large-scale datasets in each respective subfield.

-
•

We also list the current key challenges faced by the foundational architectures in each task to provide insight into future works.

*Table 2: Key methods in depth estimation. Model size and data size are approximate. Benchmark results are collected from the corresponding papers. Sections marked with different background colors ( light gray or medium gray) indicate different evaluation protocols used by the papers.*

## 2 Survey Scope

This paper primarily concentrates on depth estimation methods that leverage deep learning, with a particular emphasis on foundation models that utilize large-scale architectures and extensive datasets. We begin by defining depth foundation models and then outline the depth estimation tasks that will be addressed in the following sections.

### 2.1 Definition of Depth Foundation Models

We provide a brief overview of the development of foundation models in language models and self-supervised vision-based tasks to facilitate the understanding of the depth of foundation models. The field of language models has experienced explosive growth with the establishment of foundation models in recent years. This progress stems from the ability of these models to learn universal language and patterns from massive datasets, enabling them to generalize powerfully across various downstream tasks.
Moreover, development on scalable architectures and self-supervision tasks has also enabled the emergence of vision foundation models, which are trained on a vast amount of image or video data to facilitate the 2D or 3D perception task.

Convolutional neural networks and long short-term memory networks [[[83](#bib.bib83)]] played a main role at the early stage of language models, with limited network and data scales.
The concept of Word Embeddings [[[84](#bib.bib84)]] and the introduction of the self-attention Mechanism [[[85](#bib.bib85)]] allowed the model to process all words in a sequence simultaneously, vastly improving parallel computation efficiency and the ability to capture long-range dependencies. The original Transformer model had a relatively small number of parameters, but its architecture laid the groundwork for subsequent large-scale models.
BERTs [[[86](#bib.bib86)]] and GPTs [[[87](#bib.bib87)]] can be considered as the beginning of foundation models in large language models (LLMs).
Proposed by Google, BERT is a bidirectional pre-trained model based on the Transformer architecture, enabling better understanding of the polysemy of words in a sentence. Bert is trained on Toronto BookCorpus (800 million words) and English Wikipedia (2.5 billion words). BERT-Base has 110 million parameters, and BERT-Large has 340 million parameters.
Proposed by OpenAI, GPT is a unidirectional generative pre-trained model based on the Transformer architecture. GPT models learn language patterns by predicting the next word, excelling in text generation tasks. The GPT-3 is trained on a dataset that is larger than 45 TB, along with 175 billion parameters.

Vision foundation models have recently emerged as powerful tools for a wide range of visual perception tasks, closely following the development of scalable model architectures in language models. These models, such as DINO [[[88](#bib.bib88), [89](#bib.bib89), [90](#bib.bib90)]], MAE [[[91](#bib.bib91)]], and SAM [[[92](#bib.bib92), [93](#bib.bib93)]], are typically trained on massive image or video datasets using self-supervised or weakly supervised objectives. By leveraging scalable architectures (e.g., Vision Transformers) and large-scale pretraining, vision foundation models learn rich and generalizable visual representations that can be transferred to downstream tasks, including classification, detection, segmentation, and even 3D perception. Notably, recent diffusion-based vision models [[[94](#bib.bib94), [95](#bib.bib95)]] have also demonstrated strong capabilities in generative and dense prediction tasks. These advances in vision foundation models provide the architectural and data-driven basis for the development of depth foundation models, enabling improved generalization and scalability for depth estimation across diverse domains.

The development of depth estimation models is illustrated in [fig. 1](#S1.F1). Considering the foundation models scale in the areas of language models, we define a depth foundation model as one that is trained on a large-scale dataset (over 10 million images) and employs models with a substantial number of parameters (over 1 billion). Additionally, depth foundation models should exhibit strong generalizability across multiple data domains.

### 2.2 Depth Estimation Tasks

This survey covers several tasks, including monocular depth estimation, stereo depth estimation, multi-view depth estimation, and monocular video depth estimation using foundation models.
Let $\textbf{I}=\{I_{k,t},k=1,...,\mathcal{K},t=1,...,\mathcal{T}\}$ represent a collection of RGB images, where $\mathcal{K}$ denotes the number of cameras and $\mathcal{T}$ is the number of timestamps for the frames.
In the case of monocular depth estimation, the input consists of a single image $I_{1,1}$.
For stereo depth estimation, the input comprises a pair of images $\{I_{1,1},I_{2,1}\}$. In multi-view depth estimation, the input is a set of images captured at the same timestamp but varying in spatial locations, represented as $\{I_{k,1},k=1,...,\mathcal{K}\}$.
For monocular video depth estimation, the input consists of a sequence of images captured by a monocular camera at different timestamps, represented as $\{I_{1,t},t=1,...,\mathcal{T}\}$.
The scope of our survey excludes the task of multi-view video depth estimation, which can be represented as the most general form of inputs: $\{I_{k,t},k=1,...,\mathcal{K},t=1,...,\mathcal{T}\}$. This is due to the fact that foundation models for this task have not yet been thoroughly explored.

Moreover, we additionally define the task of metric depth estimation as one that requires the model to not only output the relative depth map, but also predict its real-world scale and offset. For the tasks of stereo, multi-view, and monocular video depth estimation, depth estimation requires predicting depth maps that, when unprojected to the world coordinate system, align across different views. When the real-world metric-scale camera parameters are known, these three tasks automatically become metric depth estimation tasks. Thus, we only distinguish between relative and metric depth estimation in the monocular depth estimation task.

For each task, we begin by reviewing the background and evolution of deep learning models specific to the task. We then delve into the development of foundation models.
Prominent examples of foundation models include transformer-based models and diffusion models. Furthermore, we also discuss the large-scale datasets used for training these foundation models, encompassing both synthetic and real-world datasets, which enable the models to generalize effectively across diverse scenes. Finally, we address valuable problems faced by existing depth foundation models.

## 3 Overview of Depth Estimation

In this section, we provide an overlook of paradigms and datasets used in depth models.

Paradigms.
In monocular image depth estimation, models have progressed from direct depth regression to affine-invariant depth, depth classification, and canonical camera depth. This progression facilitates more accurate depth map predictions using just a single input image.
In stereo image depth estimation, by utilizing the principles of stereo geometry, models can concentrate on matching corresponding pixels in image pairs. This has driven the evolution of paradigms from cost-volume methods to the attention mechanism, iterative optimizers, and ultimately to scalable training approaches.
In multi-view image depth estimation, similar to stereo paradigms, the incorporation of multi-view information has facilitated an evolution from patch-match stereo to cost volume methods, followed by a coarse-to-fine strategy and the implementation of token attention mechanisms.
In monocular image depth estimation, by incorporating an additional dimension of timestamps, models leverage temporal correlation and test-time optimization paradigms to establish the relationship between temporal and spatial information. Additionally, scaling up training enhances depth estimation performance and guides models toward becoming depth foundation models.

Datasets. We summarize the datasets commonly used in depth estimation tasks in [table 1](#S1.T1). These datasets can be categorized into two classes: real-world captured and synthesized. Each dataset may be applicable to multiple tasks, and we specify the tasks associated with each dataset in the table. Furthermore, we present details about the scenes within the datasets, including whether they provide metric information, if they include dynamic scenes, the number of scenes and frames, the image resolution, and whether the annotations are dense, based on the proportion of pixels that have corresponding depth values.

## 4 Monocular Image Depth Estimation

![Figure](x1.png)

*Figure 2: Overview of the key-idea paradigm evolution of monocular image depth estimation. From early direct regression and classification methods, through affine-invariant and canonical camera depth estimation, monocular depth estimation models have shown increasingly stronger generalization capabilities, paving the way for the emergence of depth foundation models.*

![Figure](x2.png)

*Figure 3: Representative pipelines of monocular image depth estimation. The pink component denotes operations without learnable parameters or with fixed parameters, while the blue component indicates operations with optimizable parameters. Vision Transformer-based approaches, leveraging their lightweight architectures, enable real-time monocular depth estimation. However, due to the presence of convolutional operations in their architectures, they may lose detailed features. Diffusion Model-based methods treat RGB images as conditional inputs, effectively preserving fine-grained details. Nevertheless, their denoising processes impose computational costs, making it challenging to achieve real-time performance.*

Monocular depth estimation aims to predict per-pixel depth distances for a scene from a single RGB image, establishing a geometric mapping from 2D images to 3D scenes.
The core challenges lie in scale ambiguity and the lack of geometric information inherent in monocular vision.
Compared to active depth sensors, monocular depth estimation requires no additional hardware and relies solely on image content to infer scene structure, making it valuable for various applications such as 3D reconstruction, video editing, autonomous driving, and augmented reality.
In recent years, with the advancement of deep learning techniques, monocular depth estimation has gradually shifted from traditional geometric methods to data-driven end-to-end learning paradigms, evolving toward foundation models with strong generalization capabilities, high accuracy, and robustness.

Evolution of model architectures. The evolution of monocular depth estimation has been driven by progressive innovations across network architectures, depth representations, and learning paradigms. Early approaches predominantly employed handcrafted image processing pipelines [[[96](#bib.bib96), [97](#bib.bib97), [98](#bib.bib98), [99](#bib.bib99)]], with seminal works like CNN [[[51](#bib.bib51)]] establishing the deep learning foundation. Subsequent refinements incorporated U-Net [[[100](#bib.bib100)]] and ResNet [[[101](#bib.bib101)]] residual blocks to enhance spatial continuity. The advent of dense prediction transformers [[[57](#bib.bib57)]] marked a paradigm shift: DPT [[[57](#bib.bib57)]] introduced global attention mechanisms through patch-wise sequence modeling, addressing CNN-inherent locality constraints. The latest frontier involves diffusion models [[[102](#bib.bib102), [103](#bib.bib103)]], exemplified by Marigold [[[58](#bib.bib58)]], which leverage conditional denoising frameworks to transfer generative priors into depth estimation, significantly improving geometric consistency and cross-domain generalization.

Evolution of method paradigms. Parallel advancements emerged in depth representation and learning strategies. Early methods [[[51](#bib.bib51), [100](#bib.bib100), [101](#bib.bib101)]] focused on absolute depth prediction but faced scale ambiguity challenges, prompting innovations like scale-invariant loss [[[55](#bib.bib55), [104](#bib.bib104)]] for relative depth estimation.
Another advancement came through bin classification approaches [[[53](#bib.bib53), [105](#bib.bib105), [106](#bib.bib106), [107](#bib.bib107), [108](#bib.bib108), [109](#bib.bib109)]], which discretize the continuous depth space into multiple bins. This reformulates depth estimation as a per-pixel classification problem, effectively handling non-uniform depth distributions and capturing uncertainty in predictions, particularly at depth discontinuities.
Recent breakthroughs [[[56](#bib.bib56), [110](#bib.bib110), [111](#bib.bib111), [62](#bib.bib62), [112](#bib.bib112)]] integrate camera parameters to resolve metric ambiguity while maintaining scale awareness. The data paradigm evolved through large-scale multi-view stereo (MVS) datasets like MegaDepth [[[113](#bib.bib113)]], enabling unprecedented generalization. Supervision methods expanded beyond fully labeled training: self-supervised approaches exploited photometric consistency in stereo sequences [[[114](#bib.bib114)]], while Depth Anything [[[59](#bib.bib59)]] demonstrated pseudo-label distillation’s effectiveness for knowledge transfer. Multi-task frameworks [[[115](#bib.bib115), [54](#bib.bib54), [116](#bib.bib116), [117](#bib.bib117)]] further enhanced robustness through joint depth-flow-pose estimation with geometric constraints, complemented by geometry-aware losses like Virtual Normal that explicitly enforce surface regularity. This multifaceted progression underscores how architectural innovation, representation learning, and supervision paradigms collectively advance monocular depth estimation toward human-level scene understanding.

Evolution towards foundation models. Models for monocular depth estimation are becoming larger and more data-intensive, evolving towards foundation models.
In terms of Model Size, advanced foundation models primarily utilize two key methods: Vision Transformer (ViT) architectures [[[118](#bib.bib118)]] and diffusion-based generative models [[[95](#bib.bib95)]]. Dense Prediction Transformers (DPT) are the main architecture for modern depth estimation [[[57](#bib.bib57)]], featuring 343 million parameters. Unlike older fully convolutional networks, DPT’s ViT backbone keeps high-resolution features and a global view throughout, leading to more detailed and consistent depth estimates. Pre-Trained Diffusion models are used as strong depth estimators to improve how well monocular depth estimation works [[[119](#bib.bib119), [58](#bib.bib58), [120](#bib.bib120), [121](#bib.bib121)]], thanks to their existing visual knowledge, which ranges from 200 million to 1 billion parameters. For example, Marigold [[[58](#bib.bib58)]] modifies text-to-image latent diffusion models (like Stable Diffusion v2 [[[94](#bib.bib94)]]) to predict depth from images.
In terms of Data scale, powerful hardware and better cameras have led to many high-quality depth datasets, typically ranging from thousands to millions of images; further details can be found in [table 1](#S1.T1). The current trend is to train models on large datasets to make them more adaptable. When training for large-scale depth estimation, models typically use loss functions that ignore scale and shift differences in the data [[[55](#bib.bib55)]]. This means models learn to predict relative depth, making relative depth estimation a standard task in computer vision. However, metric depth estimation, which provides absolute distance measurements, is crucial for real-world applications. Currently, there are two main ways to get metric depth: 1) Methods like those in [[[110](#bib.bib110), [111](#bib.bib111), [112](#bib.bib112), [62](#bib.bib62), [61](#bib.bib61), [64](#bib.bib64), [122](#bib.bib122)]] combine camera intrinsic estimations with relative depth predictions to get metric-scaled outputs through geometric calculations. 2) Directly learning metric depth: Approaches such as those in [[[123](#bib.bib123), [124](#bib.bib124), [125](#bib.bib125)]] train models directly on data that includes scale information, allowing them to predict metric depth without extra steps.

Valuable problems. Recent methods have made notable progress, yet critical challenges persist across four primary dimensions.
Depth accuracy.
Depth accuracy enhancement remains the foremost pursuit, aiming to develop a ”visual LiDAR” system where RGB cameras rival dedicated depth sensors. Current methods exhibit 4-5% relative error on standard benchmarks [[[62](#bib.bib62)]], escalating to 20-50% in challenging scenarios [[[126](#bib.bib126)]], while hardware sensors consistently achieve sub-1% accuracy.
Absolute scale recovery.
Though improved through works like Metric3D [[[111](#bib.bib111)]], it demonstrates fragility under complex illumination and texture-deficient conditions, necessitating more robust geometric priors. The Data Efficiency Bottleneck manifests through compounded limitations: pseudo-label noise restricts supervision quality while synthetic-to-real domain gaps constrain model generalization, demanding innovative low-cost high-precision annotation paradigms.
Multi-task generalization.
This presents an open research frontier, as current approaches struggle to unify depth estimation with complementary tasks like semantic segmentation and surface normal prediction within a single foundational model architecture. These interconnected challenges collectively underscore the need for fundamental breakthroughs in geometric understanding, data utilization, and cross-task knowledge integration to bridge the performance gap between learning-based methods and physical sensing systems.

## 5 Stereo Image Depth Estimation

![Figure](x3.png)

*Figure 4: Overview of the key-idea paradigm evolution of stereo image depth estimation. The paradigms have transitioned from cost-volume methods to attention mechanisms and iterative optimization techniques to effectively match the features of stereo images. The incorporation of monocular and diffusion priors facilitates large-scale training, paving the way for foundation models.*

![Figure](x4.png)

*Figure 5: Representative pipelines of stereo image depth estimation. The first architecture is for scalable training, which leverages all available datasets along with pseudo stereo pairs synthesized from a monocular dataset, to train a foundation model. The second architecture is migrating knowledge from a monocular foundation model to the stereo model, making it possible to achieve a stereo foundation model from relatively small-scale training datasets.*

Stereo Depth Estimation aims to estimate the per-pixel depth for a scene given the relative pose of a pair of stereo cameras and a pair of RGB observations as input.
Compared to monocular depth estimation, stereo depth estimation can utilize disparity priors to estimate depth through epipolar feature matching and triangulation, which theoretically yields better results than monocular depth estimation.
Additionally, since stereo depth estimation incorporates the known relative pose of the cameras, the estimated depth inherently includes scale information, addressing the scale ambiguity present in monocular depth estimation.
The core challenge of stereo depth estimation lies in accurately establishing the correspondence between pixels in the left and right images. The traditional matching process can be easily affected by factors such as changes in lighting, occlusions, repetitive textures, and weak textures.

With the rapid development of deep learning technology, stereo depth estimation has gradually become popular for using neural networks to replace traditional epipolar feature matching methods. The neural network architecture paradigms mainly include the CNN-based cost-volume paradigm, the Transformer-based attention paradigm, and the RNN-based iterative optimization paradigm.

Evolution of model architectures.
The CNN-based models are one of the most classic and widely used technical approaches in stereo depth estimation [[[127](#bib.bib127), [65](#bib.bib65), [128](#bib.bib128), [129](#bib.bib129), [130](#bib.bib130), [39](#bib.bib39), [131](#bib.bib131), [67](#bib.bib67), [132](#bib.bib132), [133](#bib.bib133), [134](#bib.bib134)]]. Early methods, such as PSMNet [[[128](#bib.bib128)]], used simple convolution layers to extract features, while subsequent works adopted more complex architectures like ResNet [[[135](#bib.bib135)]], DenseNet [[[136](#bib.bib136)]], and NAS search techniques [[[67](#bib.bib67)]] to extract more robust features.
Since 2020, with the success of Transformers [[[85](#bib.bib85)]] in the field of language models and the advancements of Vision Transformers [[[118](#bib.bib118)]] in computer vision, the attention paradigm was proposed [[[66](#bib.bib66), [137](#bib.bib137), [138](#bib.bib138), [139](#bib.bib139), [140](#bib.bib140), [141](#bib.bib141), [142](#bib.bib142)]]. To reduce computational complexity, [[[137](#bib.bib137)]] introduces local attention or sparse attention mechanisms.
Inspired by RNNs [[[143](#bib.bib143)]] and the application of the RAFT network architecture in optical flow estimation, [[[68](#bib.bib68), [144](#bib.bib144), [145](#bib.bib145)]] construct a correlation pyramid and use an RNN model to iteratively optimize the disparity map. [[[145](#bib.bib145)]] introduces learnable update strategies that dynamically adjust the update direction through attention mechanisms.
[[[146](#bib.bib146), [147](#bib.bib147), [148](#bib.bib148), [38](#bib.bib38)]] and [[[149](#bib.bib149)]] utilize monocular foundation models as priors and diffusion models as inpainting tools for stereo generation, respectively, advancing stereo depth estimation through large-scale training.

Evolution of model paradigms.
The CNN-based Cost Volume paradigm [[[65](#bib.bib65), [128](#bib.bib128), [129](#bib.bib129), [130](#bib.bib130), [39](#bib.bib39), [131](#bib.bib131), [67](#bib.bib67), [132](#bib.bib132), [133](#bib.bib133), [134](#bib.bib134)]] constructs a cost volume using 2D or 3D CNNs to represent the matching cost between the left and right images, and then performs Cost Aggregation as a post-processing step to ultimately regress the disparity map.
The attention paradigm [[[66](#bib.bib66), [137](#bib.bib137), [138](#bib.bib138), [139](#bib.bib139), [140](#bib.bib140), [141](#bib.bib141), [142](#bib.bib142), [145](#bib.bib145)]] models the epipolar matching problem as a sequence-to-sequence task, utilizing Self-Attention and Cross-Attention mechanisms to capture long-range dependencies between pixels.
Within a single image, the self-attention mechanism models the relationships between pixels to capture contextual information, while the Cross-Attention mechanism matches corresponding pixels between the left and right images. Specifically, each pixel in the left image interacts with all pixels in the right image to compute attention weights.
The iterative optimization paradigm [[[68](#bib.bib68), [144](#bib.bib144), [145](#bib.bib145)]] dynamically updates the Cost Volume or feature maps based on the current disparity map to capture more accurate matching information. It can balance performance and speed through early stopping, making it suitable for time-sensitive applications. Additionally, this paradigm exhibits strong robustness to initial errors.
With significant advancements in simulation technology, generative techniques, and monocular depth estimation methods in recent years, foundation models [[[150](#bib.bib150), [149](#bib.bib149), [38](#bib.bib38), [41](#bib.bib41), [151](#bib.bib151), [147](#bib.bib147), [148](#bib.bib148)]] in the field of stereo depth estimation are beginning to emerge.

Evolution towards foundation models.
Foundation models are emerging as the new paradigm for stereo depth estimation, leading to an increase in data intensity.
However, in contrast to foundation models used for monocular depth estimation, the Model Size for stereo depth estimation does not see a significant increase, ranging from 3.5 million to 11 million parameters [[[65](#bib.bib65), [66](#bib.bib66), [67](#bib.bib67), [68](#bib.bib68)]]. Instead, advances in monocular depth estimation foundation models allow stereo tasks to leverage monocular priors to enhance depth estimation. There are currently two main approaches to utilizing these priors. One approach [[[146](#bib.bib146), [38](#bib.bib38)]] involves injecting features from monocular depth models into the cost volume. The other approach [[[147](#bib.bib147), [148](#bib.bib148)]] applies stereo metrics to scale monocular depth estimates and then uses refinement networks to achieve better depth estimation results.
There are two methods to enable Large Data Scale Training for stereo depth estimation. 1) creating high-fidelity virtual scenes using advanced simulation and rendering technologies, such as FSD [[[38](#bib.bib38)]] (1 million image pairs) and TartanAir [[[41](#bib.bib41)]] (1 million image pairs). 2) synthesizing stereo data from monocular data. Learning stereo from single images [[[151](#bib.bib151)]] (597 thousand image pairs) and Stereo anything[[[150](#bib.bib150)]] (30 million image pairs) generate virtual stereo pairs by using estimated scene depth from monocular images [[[55](#bib.bib55), [59](#bib.bib59)]], allowing pixels to be warped to pseudo-stereo viewpoints and inpainting to fill in gaps. Recent advancements in diffusion models have led to stereoGen [[[149](#bib.bib149)]] (35 thousand image pairs) using stable diffusion as an inpainting tool.

Valuable Problems.
Research on foundation models for stereo depth estimation is still in its early stages, and we believe there are several key issues worth exploring in the future:
Limited data.
Generated stereo data from monocular faces challenges such as insufficient accuracy in monocular depth estimation and difficulties in filling in warp holes. Additionally, there remains a domain gap between synthetic data and the real world, and the diversity of the synthetic datasets is still not rich enough.
Lack of end-to-end training paradigm.
Current methods that leverage monocular priors treat monocular foundation models as cues, lacking end-to-end training of large model parameters. The model parameter counts are relatively small, and there is a lack of foundation model designs tailored for stereo tasks.
Limit utilization of available datasets.
There is an insufficient application of cross-domain datasets. Aside from Stereo Anything [[[150](#bib.bib150)]], most approaches typically utilize only one or two datasets, failing to fully leverage the existing stereo datasets.
Under-utilization of the diffusion architecture and priors.
A diffusion foundation model for stereo depth estimation, similar to the Marigold model [[[58](#bib.bib58)]], has not been explored.

## 6 Multi-View Image Depth Estimation

![Figure](x5.png)

*Figure 6: Overview of the key-idea paradigm evolution of multi-view image depth estimation. From early heuristic matching, through 2D CNNs and 3D CNNs, to advanced frameworks utilizing transformers and diffusion models, multi-view depth estimation models have shown increasing robustness and accuracy.*

![Figure](x6.png)

*Figure 7: Representative pipelines of multi-view image depth estimation. As shown in the upper part of the figure, the Transformer-based architecture utilizes a Transformer model to extract global features and employs a 3D-geometry Transformer to facilitate cross-view information interaction (the Cost Volume method is depicted in the figure). In contrast, the Diffusion-based model in the lower part of the figure leverages the SfM (Structure from Motion) point clouds from multi-view images as prior information and generates depth maps through a diffusion model.*

Multi-view image depth estimation (more commonly known as multi-view stereo, MVS) aims to recover the 3D structure of a scene from multiple images captured from different viewpoints, leveraging known or estimated camera poses. This task is fundamental to 3D reconstruction and underpins applications such as robotics, AR/VR, and autonomous navigation. Compared to monocular and stereo settings, the multi-view scenario provides richer geometric cues and enables more accurate and complete scene understanding, but also introduces new challenges in terms of view selection, feature aggregation, and computational efficiency. It’s worth noting that the estimated depth maps are typically already aligned with the camera parameters from multiple viewpoints; thus, we do not distinguish between relative and metric depth estimation in this section. Over the years, the field has evolved from early heuristic and patch-based matching methods, through cost volume-based deep learning approaches with 2D and 3D CNNs, to recent advances utilizing transformers and diffusion models. These developments have not only improved the robustness and accuracy of depth estimation, for example, in sparse-view cases, but also paved the way for large-scale pretraining and the emergence of foundation models with strong generalization capabilities. In this section, we review the key technical paradigms, architectural innovations, and open challenges in multi-view image depth estimation, highlighting the trajectory towards foundation models and the remaining obstacles for real-world deployment.

Evolution of model architectures.
The model architectures for multi-view depth estimation have evolved from early heuristic matching to state-of-the-art Transformer and diffusion-based models.
Early approaches [[[152](#bib.bib152), [153](#bib.bib153)]] typically assume that camera parameters were already known, obtained via SfM [[[154](#bib.bib154), [155](#bib.bib155), [156](#bib.bib156)]], SLAM [[[157](#bib.bib157), [158](#bib.bib158)]], or robotic arm control [[[159](#bib.bib159), [160](#bib.bib160)]], and rely on traditional dense pixel matching techniques [[[161](#bib.bib161)]] for depth estimation. This heuristic matching lays the groundwork for future developments.
With the advent of deep learning, researchers have begun employing 2D CNNs [[[162](#bib.bib162), [163](#bib.bib163)]] to process the cost volume, which requires low computational cost and enables real-time scene reconstruction.
At the same time, alternative approaches utilize 3D CNNs [[[164](#bib.bib164), [165](#bib.bib165), [72](#bib.bib72), [44](#bib.bib44), [166](#bib.bib166), [167](#bib.bib167), [168](#bib.bib168), [169](#bib.bib169), [170](#bib.bib170), [145](#bib.bib145)]] to achieve more accurate depth estimation.
Recent trends have started to explore the use of Transformers [[[171](#bib.bib171), [172](#bib.bib172), [74](#bib.bib74), [173](#bib.bib173), [174](#bib.bib174), [76](#bib.bib76), [75](#bib.bib75), [175](#bib.bib175), [176](#bib.bib176), [177](#bib.bib177), [178](#bib.bib178), [179](#bib.bib179)]] or diffusion models [[[180](#bib.bib180)]], which leverage global self-attention mechanisms or diffusion-based strategies to capture richer feature representations and enhance inference quality.

Evolution of method paradigms.
The method paradigms for multi-view depth estimation have evolved from early PatchMatch-based matching strategies to modern token attention mechanisms.
Traditional methods [[[152](#bib.bib152), [153](#bib.bib153)]] rely on the PatchMatch algorithm [[[161](#bib.bib161)]] for pixel-level matching—a simple yet influential approach that sets the stage for subsequent innovations.
With the integration of deep learning, constructing a cost volume by back-projecting multi-view features using camera parameters becomes the mainstream [[[164](#bib.bib164), [165](#bib.bib165), [72](#bib.bib72), [162](#bib.bib162), [44](#bib.bib44)]], allowing for more accurate depth regression by computing feature correlations.
Given the high memory consumption of cost volumes, modern approaches [[[166](#bib.bib166), [167](#bib.bib167), [168](#bib.bib168), [169](#bib.bib169), [170](#bib.bib170), [145](#bib.bib145)]] adopt a coarse-to-fine strategy: starting with a low resolution and a large number of depth hypotheses, and progressively refining the cost volume with predictions from previous stages to achieve a balance between high resolution and high accuracy.
Looking ahead, emerging research is investigating the incorporation of token attention [[[73](#bib.bib73), [173](#bib.bib173), [76](#bib.bib76), [181](#bib.bib181), [182](#bib.bib182), [183](#bib.bib183), [184](#bib.bib184), [185](#bib.bib185), [186](#bib.bib186), [75](#bib.bib75), [175](#bib.bib175), [176](#bib.bib176), [177](#bib.bib177), [178](#bib.bib178)]] mechanisms in the depth estimation process, aiming to better capture both local and global contexts and offering promising new directions for multi-view depth estimation.

Evolution towards foundation models.
In terms of Model Size, recent advances have been driven by the emergence of Transformer-based foundation models. MVSTR [[[171](#bib.bib171)]] is the first to introduce the Transformer architecture to multi-view depth estimation, employing a global-context Transformer and a 3D-geometry Transformer for intra-view global feature extraction and inter-view information interaction. MVSFormer [[[172](#bib.bib172)]] proposes to use a pre-trained Vision Transformer (ViT) to enhance Multi-View Stereo (MVS) tasks, leveraging priors learned from large-scale datasets. MVSFormer++ [[[74](#bib.bib74)]] further employs a pre-trained DINOv2 model with 1.1 billion parameters and introduces distinct attention mechanisms tailored for the feature encoder and cost volume regularization. Both PF-LRM [[[187](#bib.bib187)]] and DUSt3R [[[73](#bib.bib73)]] utilize ViT to directly predict Point Maps, with DUSt3R employing the CroCo model featuring a complete encoder-decoder Transformer architecture.
In terms of Data Scale, the evolution shows a clear trend from small-scale datasets to large-scale training. Early methods like MVSNet [[[72](#bib.bib72)]] used only 27K images from the DTU dataset, while MVSFormer++ [[[74](#bib.bib74)]] expanded to 40K images from the DTU and BlendedMVS datasets. The breakthrough came with sparse-view methods: PF-LRM [[[187](#bib.bib187)]] utilizes 1M objects from Objaverse and MVImgNet datasets, DUSt3R [[[73](#bib.bib73)]] is trained on 17M image pairs from eight datasets, including Habitat, MegaDepth, and Waymo, GRM [[[173](#bib.bib173)]] uses 40M objects from Objaverse, and VGGT [[[76](#bib.bib76)]] employs 30M images from multiple datasets, including Co3Dv2, BlendMVS, and MegaDepth. These models establish a paradigm shift from traditional optimization-based methods to data-driven architectures capable of unified feature learning and cross-view reasoning.

Valuable Problems.
Sparse view reconstruction.
In real-world applications, capturing a scene with dense and complete views may be feasible due to constraints on reachability. Therefore, the ability to reconstruct complete scene geometry from partial observations by leveraging prior knowledge represents an important direction for future research.
Find-grained depth estimation.
Current feed-forward methods [[[171](#bib.bib171), [172](#bib.bib172), [74](#bib.bib74), [173](#bib.bib173)]] can efficiently predict multi-view image depth in a single forward pass. However, accurately capturing fine-grained geometry remains challenging, as it demands high-precision geometric prediction capabilities from neural networks.
Depth estimation of objects with complex materials.
Reflective or transparent scenes pose significant challenges for geometry estimation due to their complex optical properties. Incorporating learned priors into depth estimation presents a promising approach for accurately capturing the geometry of complex materials.

## 7 Monocular Video Depth Estimation

![Figure](x7.png)

*Figure 8: Overview of the key-idea paradigm evolution of monocular video depth estimation. From RNN-based temporal modeling (2019) to CNNs with test-time optimization (2020, CVD), transformer-based scaling (2023, NVDS), and video diffusion for enhanced stability and generalization.*

Video depth estimation aims to estimate per-frame depth from a given monocular video while ensuring temporal consistency across the sequence. Compared to monocular depth estimation, its primary challenge lies in maintaining consistency over time. In contrast to multi-view stereo (MVS) [[[74](#bib.bib74), [173](#bib.bib173)]] and other multi-view depth estimation methods [[[171](#bib.bib171), [172](#bib.bib172)]], it needs to handle dynamic scenes, which imposes further challenges. Since video depth estimation requires both temporally consistent and accurate depth predictions, and its technical paradigm integrates elements from both monocular and multi-view depth estimation, we consider it the ultimate problem that a depth foundation model should address.

Evolution of model architectures. With the advancement of deep learning technologies and network architectures, various network structures have been applied in recent years to improve the temporal consistency of depth estimation.
With the rise and widespread adoption of RNNs in language models, in 2019, [[[188](#bib.bib188)]] proposed using the LSTM mechanism to integrate temporal information, thereby enhancing the stability of video depth estimation.
Meanwhile, in image tasks, CNN architectures (including U-Net and ResNet) have shown excellent performance. In 2020, [[[77](#bib.bib77)]] introduced test-time optimization to single-frame depth estimation methods based on CNN architectures, amplifying the capabilities of traditional CNNs through optimization and alignment.
Subsequently, attention mechanisms gained significant attention. [[[78](#bib.bib78), [176](#bib.bib176), [175](#bib.bib175), [177](#bib.bib177)]] applied attention mechanisms to integrate temporal features, maintaining memory through attention mechanisms rather than LSTMs, achieving more accurate and consistent depth estimation results.
Later, with the remarkable success of diffusion models in the field of image generation and the impressive results of video diffusion models, in 2024, [[[80](#bib.bib80), [81](#bib.bib81)]] proposed using video diffusion models to enhance consistency, achieving unprecedented stability and predictive performance.

Evolution of method paradigms. In recent years, innovative technical paradigms have been proposed to enhance the stability of video depth estimation.
In 2019, [[[188](#bib.bib188)]] introduced a memory mechanism to enable networks to integrate multi-frame information, implicitly encoding content from other frames to assist in depth estimation for the current frame. Subsequently, several works explored similar memory paradigms, such as MAMo [[[78](#bib.bib78)]].
In 2020, [[[77](#bib.bib77)]] proposed using test-time optimization to post-process predicted depth results. Since this approach better leverages prior techniques like SfM and SLAM while also performing bundle adjustment (BA) on depth, it often yields superior results compared to purely generalized methods.
Later, with the rise of scaling laws, interest in generalized methods was reignited, and attention shifted to the generation and utilization of large-scale training data. In 2023, [[[79](#bib.bib79)]] introduced a network-based post-processing method for monocular depth estimation results, finding a middle ground between direct video depth prediction and test-time training (TTT). It also introduced the first representative large-scale Video Depth in the Wild (VDW) dataset.
Video depth estimation has since begun to evolve toward the development of foundation models.

![Figure](x8.png)

*Figure 9: Representative pipelines of monocular video depth estimation. The Vision Transformer (ViT) paradigm processes patchified inputs through self-attention and temporal attention mechanisms to integrate spatial and temporal depth cues. The Video Diffusion paradigm leverages denoising diffusion models, using concatenated depth and image features, optionally enriched with CLIP embeddings, to generate consistent video depth estimates. These scalable architectures enhance generalization and enable zero-shot depth estimation across diverse datasets.*

Evolution towards foundation models.
In terms of Model Size, recent work has focused on using scalable network architectures to model video depth. Buffer Anytime [[[82](#bib.bib82)]] proposed using a Temporal ViT architecture with 343 million parameters to integrate temporal information and generate a large amount of pseudo ground truth (GT) based on single-image priors. Specifically, the input images are first patchified and converted into tokens recognizable by ViT. Then, self-attention is applied to these tokens. To fuse temporal information, cross-attention is performed on the single-frame attention results, reprocessing the features to obtain tokens integrated with temporal information. Finally, the tokens are decoded and unpatched to produce the final depth estimation results.
Another technical paradigm [[[80](#bib.bib80), [81](#bib.bib81), [59](#bib.bib59)]] proposed using video diffusion models with 200 million to 1 billion parameters to aid in recovering video depth. Specifically, these methods use the input video as a condition in the video diffusion denoising process. For the noisy depth during denoising, it is concatenated with the corresponding frame’s RGB color and, optionally, with CLIP embeddings as input to the denoising UNet.
In terms of Data scale, the insufficient volume of video depth data has always been a critical issue [[[189](#bib.bib189)]]. In 2023, NVDS [[[79](#bib.bib79)]] constructed the first large-scale, diverse video stereo depth dataset (14.2K videos, 2.24M frames) by using a video stereo matching method to generate pseudo labels. In 2024, methods like [[[80](#bib.bib80), [81](#bib.bib81), [59](#bib.bib59)]] addressed the lack of video data by leveraging video priors from video generation models. Similar to NVDS [[[79](#bib.bib79)]], DepthCrafter [[[81](#bib.bib81)]] annotated 200K videos using a video stereo matching method. Depth Any Video [[[59](#bib.bib59)]] created 40K videos, totaling 6M frames, using a game engine. Video Depth Anything [[[190](#bib.bib190)]] proposed using an image teacher to provide pseudo labels to compensate for the shortage of video depth ground truth (GT) labels.

Valuable problems.
Geometric inconsistency.
When camera motion is present in a monocular video, the ability to estimate consistent depth for the same statistical scene observed across multiple frames is crucial. To achieve this, jointly modeling camera motion and depth estimation is a promising approach and future direction for enhancing geometric consistency.
Temporal inconsistency.
In the presence of dynamic objects, the lack of multi-view observation in monocular video makes it particularly challenging to estimate their geometry. The method needs to learn strong priors to predict temporally consistent depth for dynamic objects across multiple frames.
Monocular video depth training data.
Learning strong dynamic priors needs extensive and diverse training data. However, due to the presence of dynamic objects, collecting real-world ground-truth monocular video training data with accurate ground-truth depth often requires additional depth sensors, limiting its scale-up capabilities.
Scaling up monocular video training data, either using real-world Internet unlabeled video with a self-supervised training strategy or simulating and rendering realistic synthetic data, is a valuable direction to be explored.

## 8 Applications

Depth foundation models are expected to see a wide range of practical applications across computer vision, robotics, and graphics.
By providing accurate, generalizable, and scalable depth estimation from monocular images, stereo pairs, multi-view images, or video sequences, these models could serve as a critical component for downstream tasks that require 3D scene understanding.
In this section, we highlight several representative application domains where depth foundation models would make a significant impact, including 3D reconstruction, novel view synthesis, video world modeling, and robotics and autonomous driving.

### 8.1 3D Reconstruction

For multi-view 3D reconstruction tasks, once the depth for each view is estimated, they can be directly fused to obtain the reconstructed scene geometry, which is usually represented as a point cloud or triangle mesh.
MVSNet [[[72](#bib.bib72)]] and CasMVSNet [[[167](#bib.bib167)]] adopt fusibile [[[71](#bib.bib71)]], which first converts the depth map of each view into a point cloud, then filters out points with poor consistency by projecting them into other views and checking, and finally fuses the point clouds from all views to obtain the reconstruction result of the entire scene.
Simplerecon [[[191](#bib.bib191)]] and Murre [[[180](#bib.bib180)]] use TSDF fusion [[[192](#bib.bib192)]], which converts depth maps into sparse truncated SDF grids, averages them across multiple views, and finally extracts the surface using marching cubes to obtain a triangle mesh.
Direct fusion methods are relatively efficient but require high-quality depth maps. NeuRIS [[[193](#bib.bib193)]] and MonoSDF [[[194](#bib.bib194)]] use the relative depth predicted from a monocular depth estimation method as a supervision signal, constraining the SDF field through a specially designed depth loss, thereby enhancing the quality of the 3D reconstruction.

### 8.2 Novel View Synthesis

In recent years, NeRF-based [[[195](#bib.bib195)]] and 3D Gaussian Splatting-based [[[196](#bib.bib196)]] methods have made significant progress in the task of novel view synthesis. High-quality depth estimations can serve as strong priors for these models, enhancing their performance and accelerating convergence. Specifically, the depth map can be utilized as sampling guidance for NeRF-based methods [[[194](#bib.bib194), [197](#bib.bib197), [198](#bib.bib198)]] and as a coarse initialization for 3D Gaussian Splatting-based methods [[[199](#bib.bib199), [200](#bib.bib200), [201](#bib.bib201)]]. Additionally, the dense depth map can also be employed to learn a dense SDF field during the training process, allowing for the alignment of the geometry of NeRF or 3D Gaussians with the SDF field. This alignment can improve the geometry of the reconstructed results and facilitate the synthesis of high-quality novel views, particularly for perspectives that are distant from the training views.
Some existing works [[[202](#bib.bib202), [203](#bib.bib203), [204](#bib.bib204), [205](#bib.bib205), [206](#bib.bib206)]] depend on lidar point clouds or RGB-D images as geometric priors. However, both lidar point clouds and RGB-D images can be costly to acquire, requiring additional sensors. High-quality depth estimations can serve as a substitute for these methods, unleashing the potential of novel view synthesis techniques.

### 8.3 Video World Models

As the popularity of diffusion models grows, video diffusion models have been proposed to generalize the image synthesis pipeline to video generation.
With the success of SORA [[[207](#bib.bib207)]] and other video foundation models [[[208](#bib.bib208), [209](#bib.bib209)]], there exist several attempts [[[210](#bib.bib210), [211](#bib.bib211), [212](#bib.bib212)]] at exploring video models’ capabilities as world models. Having a foundational model for depth estimation, preferably on videos, would significantly bridge the gap between image-only generation models’ understanding of 3D space and motion. As the ability to predict depth would indicate, the model at least possesses the capability of distinguishing the size and placement pattern of everyday objects, scenes, and people. Having depth cues for video generation models could potentially serve as a breaking point for further boosting current world models’ ability to understand everyday scenes and might even stimulate the underlying generalization ability even further, leading to more spatially and temporally consistent generation and future prediction results.

### 8.4 Robotics and Autonomous Driving

In robotics and autonomous driving, accurate and reliable depth perception plays a pivotal role in tasks such as navigation, obstacle detection, and collision avoidance.
Traditional solutions often rely on LiDAR or stereo camera systems, both of which come with increased hardware costs and complexity. Depth foundation models learned from large-scale datasets have the potential to deliver high-quality depth estimates from a single camera, making them particularly attractive for cost-sensitive real-world applications. Recent methods [[[125](#bib.bib125)]] demonstrate that monocular depth estimation can be integrated into robotic perception pipelines, serving as a complement or even a substitute for more expensive sensors.
For instance, some works [[[110](#bib.bib110), [111](#bib.bib111)]] adopt monocular depth estimation to enhance SLAM or visual odometry frameworks, showing improvements in localization and mapping under challenging lighting or weather conditions.
Similarly, in autonomous driving, depth estimates can facilitate large-scale 3D reconstructions for building realistic simulation environments, supporting algorithm development and testing [[[202](#bib.bib202)]].
Furthermore, by leveraging depth priors learned from diverse scenes, these models exhibit promising generalization capabilities, potentially enabling robust domain adaptation across varied environments—from urban streets to off-road terrains—thus paving the way for more versatile and scalable robotic and self-driving solutions.

## 9 Future Work

As discussed in the previous sections, there exist several fundamental problems to be solved before we reach a general-purpose depth foundation model, namely, data and consistency.

Data.
For all of the discussed depth estimation tasks, including monocular image, stereo image, multi-view image, and monocular video depth estimation, the lack of accurate, large-scale, high-quality, and high-variability data is currently the main concern for constructing and training a depth foundation model.
Due to the unique nature of the depth estimation task, current approaches to acquiring data usually fall into two categories: depth sensor or synthetic rendering.
For depth sensors, the main approach is to utilize LiDARs or ultrasonic devices. However, the acquired ground truth depth maps are usually incomplete or noisy due to the sensitive nature of the depth sensing devices.
For synthetic data generation, there exist several attempts at curating high-quality, hand-crafted, large static or dynamic 3D scenes by artists. However, these data are naturally limited to a small scale due to the amount of work required.
Future works should focus on either utilizing self-supervision techniques to better transfer the knowledge of vast image and video data to the task of depth estimation, or developing a better approach for simulation and generation, providing artist-quality synthetic rendering and depth pairs to boost generalization ability.

Consistency.
This includes both spatial and temporal consistency.
For the task of monocular image depth estimation, current methods typically fall short when merging depth estimation results together from different timestamps and viewports of the same scene.
For video depth estimation, although a vast amount of work has investigated the issue of temporal consistency throughout the target video, they still fail to produce accurate and consistent results when given multiple viewports of the same 3D scene or trying to unproject and merge the prediction results [[[125](#bib.bib125)]].
Notably, multi-view video reconstruction methods [[[213](#bib.bib213), [214](#bib.bib214), [215](#bib.bib215)]] have proved the existence of the dynamic and multi-view inductive bias of the 4D world by providing accurate reconstruction from only image-based optimization objects.
Future work should focus on exploring the intrinsic 3D or dynamic inductive bias present in the dynamic 3D world, further mitigating the problem of spatial and temporal inconsistency.

## 10 Conclusion

Since 2022, the advancements in foundation models within the natural language processing domain, along with the emergence of scaling laws, have led to a significant increase in the development of foundation models in the field of computer vision. In recent years, numerous foundation models have been introduced for depth estimation tasks, and new models continue to emerge at a rapid pace, making it challenging for practitioners to stay updated with the latest developments.

In this timely paper, we present a comprehensive survey of foundation models for depth estimation tasks, covering their background, development, and the latest advancements. We also address the valuable problems faced by existing depth foundation models and their downstream applications. We aim for this paper to serve as a valuable guide for practitioners and researchers interested in depth estimation foundation models.

Finally, there remain numerous challenges and opportunities in the realm of depth foundation models. We believe that as foundation models continue to evolve and depth estimation tasks advance, we will witness an increasing number of sophisticated and practical applications in the future.

### Availability of data and materials

Not applicable.

### Author contributions

Zhen Xu and Hongyu Zhou were responsible for the overall writing of the manuscript.
Sida Peng, Yiyi Liao, Yue Wang, Ruizhen Hu, Xiaowei Zhou, and Hujun Bao provided critical supervision and guidance throughout the project, shaping its framework, refining technical discussions, and ensuring clarity and coherence, providing valuable oversight and feedback during drafting and revision.
The remaining co-authors supported the work by evaluating key publications and charting the evolution timeline of depth estimation architectures across the monocular, stereo, multi-view, and monocular video depth estimation tasks to deliver a thorough survey.

### Acknowledgements

This research was supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. LD25F030001, and Information Technology Center and State Key Lab of CAD&CG, Zhejiang University.

### Declaration of competing interest

This survey offers an analysis of recent vision-based depth estimation research and its trend towards depth foundation modeo, and does not introduce new datasets or materials, nor involve any competing interests.

## References

-
[1]

Geyer J, Kassahun Y, Mahmudi M, Ricou X, Durgesh R, Chung AS, Hauswald L, Pham VH, Mühlegg M, Dorn S, et al.. A2d2: Audi autonomous driving dataset. arXiv preprint arXiv:2004.06320*, 2020.

-
[2]

Wilson B, Qi W, Agarwal T, Lambert J, Singh J, Khandelwal S, Pan B, Kumar R, Hartnett A, Kaesemodel Pontes J, Ramanan D, Carr P, Hays J. Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

-
[3]

Lambert J, Hays J. Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

-
[4]

Cordts M, Omran M, Ramos S, Rehfeld T, Enzweiler M, Benenson R, Franke U, Roth S, Schiele B. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, 3213–3223.

-
[5]

Guizilini V, Ambrus R, Pillai S, Raventos A, Gaidon A. 3d packing for self-supervised monocular depth estimation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 2485–2494.

-
[6]

Cho J, Min D, Kim Y, Sohn K. DIML/CVL RGB-D dataset: 2M RGB-D images of natural indoor and outdoor scenes. *arXiv preprint arXiv:2110.11590*, 2021.

-
[7]

Vasiljevic I, Kolkin N, Zhang S, Luo R, Wang H, Dai FZ, Daniele AF, Mostajabi M, Basart S, Walter MR, et al.. Diode: A dense indoor and outdoor depth dataset. *arXiv preprint arXiv:1908.00463*, 2019.

-
[8]

Gehrig M, Aarents W, Gehrig D, Scaramuzza D. DSEC: A Stereo Event Camera Dataset for Driving Scenarios. *IEEE Robotics and Automation Letters*, 2021, [10.1109/LRA.2021.3068942](https:/doi.org/10.1109/LRA.2021.3068942).

-
[9]

Ramakrishnan SK, Gokaslan A, Wijmans E, Maksymets O, Clegg A, Turner J, Undersander E, Galuba W, Westbury A, Chang A, Savva M, Zhao Y, Batra D. Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

-
[10]

Koch T, Liebel L, Fraundorfer F, Korner M. Evaluation of cnn-based single-image depth estimation methods. In *Proceedings of the European Conference on Computer Vision (ECCV) Workshops*, 2018, 0–0.

-
[11]

Houston J, Zuidhof G, Bergamini L, Ye Y, Chen L, Jain A, Omari S, Iglovikov V, Ondruska P. One thousand and one hours: Self-driving motion prediction dataset. In *Conference on Robot Learning*, PMLR2021, 409–418.

-
[12]

Antequera ML, Gargallo P, Hofinger M, Bulo SR, Kuang Y, Kontschieder P. Mapillary planet-scale depth dataset. In *European Conference on Computer Vision*, Springer2020, 589–604.

-
[13]

Caesar H, Bankiti V, Lang AH, Vora S, Liong VE, Xu Q, Krishnan A, Pan Y, Baldan G, Beijbom O. nuscenes: A multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 11621–11631.

-
[14]

Silberman N, Hoiem D, Kohli P, Fergus R. Indoor segmentation and support inference from rgbd images. In *Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12*, Springer2012, 746–760.

-
[15]

Xiao P, Shao Z, Hao S, Zhang Z, Chai X, Jiao J, Li Z, Wu J, Sun K, Jiang K, et al.. Pandaset: Advanced sensor suite dataset for autonomous driving. In *2021 IEEE international intelligent transportation systems conference (ITSC)*, IEEE2021, 3095–3101.

-
[16]

Zamir AR, Sax A, Shen W, Guibas LJ, Malik J, Savarese S. Taskonomy: Disentangling task transfer learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 3712–3722.

-
[17]

Geiger A, Lenz P, Urtasun R. Are we ready for autonomous driving? the kitti vision benchmark suite. In *2012 IEEE conference on computer vision and pattern recognition*, IEEE2012, 3354–3361.

-
[18]

Yang G, Song X, Huang C, Deng Z, Shi J, Zhou B. Drivingstereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019, 899–908.

-
[19]

Bao W, Wang W, Xu Y, Guo Y, Hong S, Zhang X. Instereo2k: a large real dataset for stereo matching in indoor scenes. *Science China Information Sciences*, 2020, 63: 1–11.

-
[20]

Chang MF, Lambert J, Sangkloy P, Singh J, Bak S, Hartnett A, Wang D, Carr P, Lucey S, Ramanan D, et al.. Argoverse: 3d tracking and forecasting with rich maps. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, 8748–8757.

-
[21]

Schops T, Schonberger JL, Galliani S, Sattler T, Schindler K, Pollefeys M, Geiger A. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017, 3260–3269.

-
[22]

Sun P, Kretzschmar H, Dotiwalla X, Chouard A, Patnaik V, Tsui P, Guo J, Zhou Y, Chai Y, Caine B, et al.. Scalability in perception for autonomous driving: Waymo open dataset. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 2446–2454.

-
[23]

Bauer Z, Gomez-Donoso F, Cruz E, Orts-Escolano S, Cazorla M. UASOL, a large-scale high-resolution outdoor stereo dataset. *Scientific Data*, 2019, 6.

-
[24]

Aanæs H, Jensen RR, Vogiatzis G, Tola E, Dahl AB. Large-scale data for multiple-view stereopsis. *International Journal of Computer Vision*, 2016, 120: 153–168.

-
[25]

Yao Y, Luo Z, Li S, Zhang J, Ren Y, Zhou L, Fang T, Quan L. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 1790–1799.

-
[26]

Xia H, Fu Y, Liu S, Wang X. RGBD objects in the wild: scaling real-world 3D object learning from RGB-D videos. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 22378–22389.

-
[27]

Yu X, Xu M, Zhang Y, Liu H, Ye C, Wu Y, Yan Z, Zhu C, Xiong Z, Liang T, Chen G, Cui S, Han X. MVImgNet: A Large-scale Dataset of Multi-view Images. *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023: 9150–9161.

-
[28]

Baruch G, Chen Z, Dehghan A, Dimry T, Feigin Y, Fu P, Gebauer T, Joffe B, Kurz D, Schwartz A, et al.. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. *arXiv preprint arXiv:2111.08897*, 2021.

-
[29]

Chang A, Dai A, Funkhouser T, Halber M, Niessner M, Savva M, Song S, Zeng A, Zhang Y. Matterport3d: Learning from rgb-d data in indoor environments. *arXiv preprint arXiv:1709.06158*, 2017.

-
[30]

Straub J, Whelan T, Ma L, Chen Y, Wijmans E, Green S, Engel JJ, Mur-Artal R, Ren C, Verma S, Clarkson A, Yan M, Budge B, Yan Y, Pan X, Yon J, Zou Y, Leon K, Carter N, Briales J, Gillingham T, Mueggler E, Pesqueira L, Savva M, Batra D, Strasdat HM, Nardi RD, Goesele M, Lovegrove S, Newcombe R. The Replica Dataset: A Digital Replica of Indoor Spaces. *arXiv preprint arXiv:1906.05797*, 2019.

-
[31]

Karaev N, Rocco I, Graham B, Neverova N, Vedaldi A, Rupprecht C. Dynamicstereo: Consistent dynamic depth from stereo videos. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 13229–13239.

-
[32]

Dai A, Chang AX, Savva M, Halber M, Funkhouser T, Nießner M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, 5828–5839.

-
[33]

Yeshwanth C, Liu YC, Nießner M, Dai A. Scannet++: A high-fidelity dataset of 3d indoor scenes. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 12–22.

-
[34]

Butler DJ, Wulff J, Stanley GB, Black MJ. A naturalistic open source movie for optical flow evaluation. In *Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12*, Springer2012, 611–625.

-
[35]

Mayer N, Ilg E, Hausser P, Fischer P, Cremers D, Dosovitskiy A, Brox T. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, 4040–4048.

-
[36]

Li J, Wang P, Xiong P, Cai T, Yan Z, Yang L, Liu J, Fan H, Liu S. Practical stereo matching via cascaded recurrent network with adaptive correlation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, 16263–16272.

-
[37]

Tremblay J, To T, Birchfield S. Falling things: A synthetic dataset for 3d object detection and pose estimation. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops*, 2018, 2038–2041.

-
[38]

Wen B, Trepte M, Aribido J, Kautz J, Gallo O, Birchfield S. FoundationStereo: Zero-Shot Stereo Matching. *arXiv preprint arXiv:2501.09898*, 2025.

-
[39]

Tosi F, Liao Y, Schmitt C, Geiger A. Smd-nets: Stereo mixture density networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2021, 8942–8952.

-
[40]

Mehl L, Schmalfuss J, Jahedi A, Nalivayko Y, Bruhn A. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 4981–4991.

-
[41]

Wang W, Zhu D, Wang X, Hu Y, Qiu Y, Wang C, Hu Y, Kapoor A, Scherer S. Tartanair: A dataset to push the limits of visual slam. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, IEEE2020, 4909–4916.

-
[42]

Cabon Y, Murray N, Humenberger M. Virtual KITTI 2, 2020.

-
[43]

Li Y, Jiang L, Xu L, Xiangli Y, Wang Z, Lin D, Dai B. MatrixCity: A Large-scale City Dataset for City-scale Neural Rendering and Beyond. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023: 3182–3192.

-
[44]

Huang PH, Matzen K, Kopf J, Ahuja N, Huang JB. Deepmvs: Learning multi-view stereopsis. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 2821–2830.

-
[45]

Niklaus S, Mai L, Yang J, Liu F. 3d ken burns effect from a single image. *ACM Transactions on Graphics (ToG)*, 2019, 38(6): 1–15.

-
[46]

Wu T, Zhang J, Fu X, Wang Y, Ren J, Pan L, Wu W, Yang L, Wang J, Qian C, et al.. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 803–814.

-
[47]

Wang Q, Zheng S, Yan Q, Deng F, Zhao K, Chu X. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In *2021 IEEE International Conference on Multimedia and Expo (ICME)*, IEEE2021, 1–6.

-
[48]

Zheng Y, Harley AW, Shen B, Wetzstein G, Guibas LJ. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 19855–19865.

-
[49]

Black MJ, Patel P, Tesch J, Yang J. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 8726–8737.

-
[50]

Menze M, Geiger A. Object scene flow for autonomous vehicles. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2015, 3061–3070.

-
[51]

Eigen D, Puhrsch C, Fergus R. Depth map prediction from a single image using a multi-scale deep network. *Advances in neural information processing systems*, 2014, 27.

-
[52]

Godard C, Mac Aodha O, Brostow GJ. Unsupervised monocular depth estimation with left-right consistency. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, 270–279.

-
[53]

Fu H, Gong M, Wang C, Batmanghelich K, Tao D. Deep ordinal regression network for monocular depth estimation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 2002–2011.

-
[54]

Xu D, Ouyang W, Wang X, Sebe N. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 675–684.

-
[55]

Ranftl R, Lasinger K, Hafner D, Schindler K, Koltun V. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE transactions on pattern analysis and machine intelligence*, 2020, 44(3): 1623–1637.

-
[56]

Yin W, Zhang J, Wang O, Niklaus S, Mai L, Chen S, Shen C. Learning to recover 3d scene shape from a single image. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021, 204–213.

-
[57]

Ranftl R, Bochkovskiy A, Koltun V. Vision transformers for dense prediction. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2021, 12179–12188.

-
[58]

Ke B, Obukhov A, Huang S, Metzger N, Daudt RC, Schindler K. Repurposing diffusion-based image generators for monocular depth estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 9492–9502.

-
[59]

Yang L, Kang B, Huang Z, Xu X, Feng J, Zhao H. Depth anything: Unleashing the power of large-scale unlabeled data. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 10371–10381.

-
[60]

Yang L, Kang B, Huang Z, Zhao Z, Xu X, Feng J, Zhao H. Depth anything v2. *Advances in Neural Information Processing Systems*, 2024, 37: 21875–21911.

-
[61]

Wang R, Xu S, Dai C, Xiang J, Deng Y, Tong X, Yang J. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision, 2024.

-
[62]

Piccinelli L, Yang YH, Sakaridis C, Segu M, Li S, Van Gool L, Yu F. UniDepth: Universal monocular metric depth estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 10106–10116.

-
[63]

Piccinelli L, Sakaridis C, Yang YH, Segu M, Li S, Abbeloos W, Van Gool L. Unidepthv2: Universal monocular metric depth estimation made simpler. *arXiv preprint arXiv:2502.20110*, 2025.

-
[64]

Wang R, Xu S, Dong Y, Deng Y, Xiang J, Lv Z, Sun G, Tong X, Yang J. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details, 2025.

-
[65]

Kendall A, Martirosyan H, Dasgupta S, Henry P, Kennedy R, Bachrach A, Bry A. End-to-end learning of geometry and context for deep stereo regression. In *Proceedings of the IEEE international conference on computer vision*, 2017, 66–75.

-
[66]

Li Z, Liu X, Drenkow N, Ding A, Creighton FX, Taylor RH, Unberath M. Revisiting stereo depth estimation from a sequence-to-sequence perspective with transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2021, 6197–6206.

-
[67]

Cheng X, Zhong Y, Harandi M, Dai Y, Chang X, Li H, Drummond T, Ge Z. Hierarchical neural architecture search for deep stereo matching. *Advances in neural information processing systems*, 2020, 33: 22158–22169.

-
[68]

Lipson L, Teed Z, Deng J. Raft-stereo: Multilevel recurrent field transforms for stereo matching. In *2021 International Conference on 3D Vision (3DV)*, IEEE2021, 218–227.

-
[69]

Wang Y, Liang Y, Li H, Fu Y. Mono2stereo: Monocular knowledge transfer for enhanced stereo matching. *arXiv preprint arXiv:2411.09151*, 2024.

-
[70]

Jensen R, Dahl A, Vogiatzis G, Tola E, Aanæs H. Large scale multi-view stereopsis evaluation. In *2014 IEEE Conference on Computer Vision and Pattern Recognition*, IEEE2014, 406–413.

-
[71]

Galliani S, Lasinger K, Schindler K. Massively parallel multiview stereopsis by surface normal diffusion. In *Proceedings of the IEEE International Conference on Computer Vision*, 2015, 873–881.

-
[72]

Yao Y, Luo Z, Li S, Fang T, Quan L. Mvsnet: Depth inference for unstructured multi-view stereo. In *Proceedings of the European conference on computer vision (ECCV)*, 2018, 767–783.

-
[73]

Wang S, Leroy V, Cabon Y, Chidlovskii B, Revaud J. Dust3r: Geometric 3d vision made easy. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 20697–20709.

-
[74]

cao c, ren x, Fu Y. MVSFormer++: Revealing the Devil in Transformer's Details for Multi-View Stereo. In B Kim, Y Yue, S Chaudhuri, K Fragkiadaki, M Khan, Y Sun, editors, *International Conference on Representation Learning*, volume 2024, 2024, 39816–39837.

-
[75]

Leroy V, Cabon Y, Revaud J. Grounding Image Matching in 3D with MASt3R, 2024.

-
[76]

Wang J, Chen M, Karaev N, Vedaldi A, Rupprecht C, Novotny D. VGGT: Visual Geometry Grounded Transformer. *arXiv preprint arXiv:2503.11651*, 2025.

-
[77]

Luo X, Huang JB, Szeliski R, Matzen K, Kopf J. Consistent video depth estimation. *ACM Transactions on Graphics (ToG)*, 2020, 39(4): 71–1.

-
[78]

Yasarla R, Cai H, Jeong J, Shi Y, Garrepalli R, Porikli F. Mamo: Leveraging memory and attention for monocular video depth estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 8754–8764.

-
[79]

Wang Y, Shi M, Li J, Huang Z, Cao Z, Zhang J, Xian K, Lin G. Neural video depth stabilizer. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 9466–9476.

-
[80]

Shao J, Yang Y, Zhou H, Zhang Y, Shen Y, Poggi M, Liao Y. Learning Temporally Consistent Video Depth from Video Diffusion Priors. *arXiv preprint arXiv:2406.01493*, 2024.

-
[81]

Hu W, Gao X, Li X, Zhao S, Cun X, Zhang Y, Quan L, Shan Y. Depthcrafter: Generating consistent long depth sequences for open-world videos. *arXiv preprint arXiv:2409.02095*, 2024.

-
[82]

Kuang Z, Zhang T, Zhang K, Tan H, Bi S, Hu Y, Xu Z, Hasan M, Wetzstein G, Luan F. Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors. *arXiv preprint arXiv:2411.17249*, 2024.

-
[83]

Graves A, Graves A. Long short-term memory. *Supervised sequence labelling with recurrent neural networks*, 2012: 37–45.

-
[84]

Almeida F, Xexéo G. Word embeddings: A survey. *arXiv preprint arXiv:1901.09069*, 2019.

-
[85]

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Lu, Polosukhin I. Attention is All you Need. In I Guyon, UV Luxburg, S Bengio, H Wallach, R Fergus, S Vishwanathan, R Garnett, editors, *Advances in Neural Information Processing Systems*, volume 30, Curran Associates, Inc.2017, 6000––6010.

-
[86]

Devlin J, Chang MW, Lee K, Toutanova K. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)*, 2019, 4171–4186.

-
[87]

Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A, et al.. Language models are few-shot learners. *Advances in neural information processing systems*, 2020, 33: 1877–1901.

-
[88]

Caron M, Touvron H, Misra I, Jégou H, Mairal J, Bojanowski P, Joulin A. Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2021, 9650–9660.

-
[89]

Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, Fernandez P, Haziza D, Massa F, El-Nouby A, et al.. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.

-
[90]

Siméoni O, Vo HV, Seitzer M, Baldassarre F, Oquab M, Jose C, Khalidov V, Szafraniec M, Yi S, Ramamonjisoa M, et al.. Dinov3. *arXiv preprint arXiv:2508.10104*, 2025.

-
[91]

He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, 16000–16009.

-
[92]

Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, Xiao T, Whitehead S, Berg AC, Lo WY, et al.. Segment anything. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2023, 4015–4026.

-
[93]

Ravi N, Gabeur V, Hu YT, Hu R, Ryali C, Ma T, Khedr H, Rädle R, Rolland C, Gustafson L, et al.. Sam 2: Segment anything in images and videos. *arXiv preprint arXiv:2408.00714*, 2024.

-
[94]

Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, 10684–10695.

-
[95]

Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 2020, 33: 6840–6851.

-
[96]

Saxena A, Chung S, Ng A. Learning depth from single monocular images. *Advances in neural information processing systems*, 2005, 18.

-
[97]

Hoiem D, Efros AA, Hebert M. Recovering surface layout from an image. *International Journal of Computer Vision*, 2007, 75: 151–172.

-
[98]

Liu C, Yuen J, Torralba A, Sivic J, Freeman WT. Sift flow: Dense correspondence across different scenes. In *Computer Vision–ECCV 2008: 10th European Conference on Computer Vision, Marseille, France, October 12-18, 2008, Proceedings, Part III 10*, Springer2008, 28–42.

-
[99]

Saxena A, Sun M, Ng AY. Make3d: Learning 3d scene structure from a single still image. *IEEE transactions on pattern analysis and machine intelligence*, 2008, 31(5): 824–840.

-
[100]

Eigen D, Fergus R. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In *Proceedings of the IEEE international conference on computer vision*, 2015, 2650–2658.

-
[101]

Laina I, Rupprecht C, Belagiannis V, Tombari F, Navab N. Deeper depth prediction with fully convolutional residual networks. In *2016 Fourth international conference on 3D vision (3DV)*, IEEE2016, 239–248.

-
[102]

Ji Y, Chen Z, Xie E, Hong L, Liu X, Liu Z, Lu T, Li Z, Luo P. Ddp: Diffusion model for dense visual prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 21741–21752.

-
[103]

Saxena S, Herrmann C, Hur J, Kar A, Norouzi M, Sun D, Fleet DJ. The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. *Advances in Neural Information Processing Systems*, 2023, 36: 39443–39469.

-
[104]

Yin W, Wang X, Shen C, Liu Y, Tian Z, Xu S, Sun C, Renyin D. Diversedepth: Affine-invariant depth prediction using diverse data. *arXiv preprint arXiv:2002.00569*, 2020.

-
[105]

Lee JH, Kim CS. Monocular depth estimation using relative depth maps. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, 9729–9738.

-
[106]

Bhat SF, Alhashim I, Wonka P. Adabins: Depth estimation using adaptive bins. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2021, 4009–4018.

-
[107]

Bhat SF, Alhashim I, Wonka P. Localbins: Improving depth estimation by learning local distributions. In *European Conference on Computer Vision*, Springer2022, 480–496.

-
[108]

Zhang S, Yang L, Mi MB, Zheng X, Yao A. Improving deep regression with ordinal entropy. *arXiv preprint arXiv:2301.08915*, 2023.

-
[109]

Shao S, Pei Z, Wu X, Liu Z, Chen W, Li Z. Iebins: Iterative elastic bins for monocular depth estimation. *Advances in Neural Information Processing Systems*, 2023, 36: 53025–53037.

-
[110]

Yin W, Zhang C, Chen H, Cai Z, Yu G, Wang K, Chen X, Shen C. Metric3d: Towards zero-shot metric 3d prediction from a single image. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 9043–9053.

-
[111]

Hu M, Yin W, Zhang C, Cai Z, Long X, Chen H, Wang K, Yu G, Shen C, Shen S. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

-
[112]

Bochkovskii A, Delaunoy A, Germain H, Santos M, Zhou Y, Richter SR, Koltun V. Depth pro: Sharp monocular metric depth in less than a second. *arXiv preprint arXiv:2410.02073*, 2024.

-
[113]

Li Z, Snavely N. Megadepth: Learning single-view depth prediction from internet photos. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 2041–2050.

-
[114]

Godard C, Mac Aodha O, Firman M, Brostow GJ. Digging into self-supervised monocular depth estimation. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2019, 3828–3838.

-
[115]

Qi X, Liao R, Liu Z, Urtasun R, Jia J. Geonet: Geometric neural network for joint depth and surface normal estimation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2018, 283–291.

-
[116]

Fu X, Yin W, Hu M, Wang K, Ma Y, Tan P, Shen S, Lin D, Long X. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In *European Conference on Computer Vision*, Springer2024, 241–258.

-
[117]

Eftekhar A, Sax A, Malik J, Zamir A. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, 10786–10796.

-
[118]

Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold G, Gelly S, Uszkoreit J, Houlsby N. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations*, 2021.

-
[119]

Song Z, Wang Z, Li B, Zhang H, Zhu R, Liu L, Jiang PT, Zhang T. Depthmaster: Taming diffusion models for monocular depth estimation. *arXiv preprint arXiv:2501.02576*, 2025.

-
[120]

Xu G, Ge Y, Liu M, Fan C, Xie K, Zhao Z, Chen H, Shen C. What matters when repurposing diffusion models for general dense perception tasks? *arXiv preprint arXiv:2403.06090*, 2024.

-
[121]

Lavreniuk M, Bhat SF, Müller M, Wonka P. Evp: Enhanced visual perception using inverse multi-attentive feature refinement and regularized image-text alignment. In *European Conference on Computer Vision*, Springer2024, 206–225.

-
[122]

Carreira J, Gokay D, King M, Zhang C, Rocco I, Mahendran A, Keck TA, Heyward J, Koppula S, Pot E, et al.. Scaling 4d representations. *arXiv preprint arXiv:2412.15212*, 2024.

-
[123]

Bhat SF, Birkl R, Wofk D, Wonka P, Müller M. ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth, 2023.

-
[124]

Viola M, Qu K, Metzger N, Ke B, Becker A, Schindler K, Obukhov A. Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion, 2024.

-
[125]

Lin H, Peng S, Chen J, Peng S, Sun J, Liu M, Bao H, Feng J, Zhou X, Kang B. Prompting depth anything for 4k resolution accurate metric depth estimation. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, 17070–17080.

-
[126]

Yang H, Huang D, Yin W, Shen C, Liu H, He X, Lin B, Ouyang W, He T. Depth any video with scalable synthetic data. *arXiv preprint arXiv:2410.10815*, 2024.

-
[127]

Chen Z, Sun X, Wang L, Yu Y, Huang C. A deep visual correspondence embedding model for stereo matching costs. In *Proceedings of the IEEE International Conference on Computer Vision*, 2015, 972–980.

-
[128]

Chang JR, Chen YS. Pyramid stereo matching network. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, 5410–5418.

-
[129]

Guo X, Yang K, Yang W, Wang X, Li H. Group-wise correlation stereo network. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, 3273–3282.

-
[130]

Zhang F, Prisacariu V, Yang R, Torr PH. Ga-net: Guided aggregation net for end-to-end stereo matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, 185–194.

-
[131]

Shen Z, Dai Y, Song X, Rao Z, Zhou D, Zhang L. Pcw-net: Pyramid combination and warping cost volume for stereo matching. In *European conference on computer vision*, Springer2022, 280–297.

-
[132]

Badki A, Troccoli A, Kim K, Kautz J, Sen P, Gallo O. Bi3d: Stereo depth estimation via binary classifications. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020, 1600–1608.

-
[133]

Xu H, Zhang J. Aanet: Adaptive aggregation network for efficient stereo matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 1959–1968.

-
[134]

Yang M, Wu F, Li W. Waveletstereo: Learning wavelet coefficients of disparity map in stereo matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 12885–12894.

-
[135]

He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, 770–778.

-
[136]

Huang G, Liu Z, Van Der Maaten L, Weinberger KQ. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, 4700–4708.

-
[137]

Guo W, Li Z, Yang Y, Wang Z, Taylor RH, Unberath M, Yuille A, Li Y. Context-enhanced stereo transformer. In *European Conference on Computer Vision*, Springer2022, 263–279.

-
[138]

Liu Z, Li Y, Okutomi M. Global Occlusion-Aware Transformer for Robust Stereo Matching. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024, 3535–3544.

-
[139]

Xu H, Zhang J, Cai J, Rezatofighi H, Yu F, Tao D, Geiger A. Unifying flow, stereo and depth estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2023.

-
[140]

Su Q, Ji S. Chitransformer: Towards reliable stereo from cues. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, 1939–1949.

-
[141]

Weinzaepfel P, Leroy V, Lucas T, Brégier R, Cabon Y, Arora V, Antsfeld L, Chidlovskii B, Csurka G, Revaud J. CroCo: self-supervised pre-training for 3D vision tasks by cross-view completion. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, ¡em class=”Highlight hta8af2fc9-2088-421c-9531-5cd31d54a019” highlight=”true” htmatch=”nips” htloopnumber=”856596281” style=”font-style: inherit;”¿NIPS¡/em¿ ’22, Red Hook, NY, USA: Curran Associates Inc.2022, 3502––3516.

-
[142]

Lou J, Liu W, Chen Z, Liu F, Cheng J. Elfnet: Evidential local-global fusion for stereo matching. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, 17784–17793.

-
[143]

Medsker LR, Jain L, et al.. Recurrent neural networks. *Design and Applications*, 2001, 5(64-67): 2.

-
[144]

Xu G, Wang X, Ding X, Yang X. Iterative geometry encoding volume for stereo matching. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 21919–21928.

-
[145]

Xu G, Wang X, Zhang Z, Cheng J, Liao C, Yang X. IGEV++: iterative multi-range geometry encoding volumes for stereo matching. *arXiv preprint arXiv:2409.00638*, 2024.

-
[146]

Bartolomei L, Tosi F, Poggi M, Mattoccia S. Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail. *arXiv preprint arXiv:2412.04472*, 2024.

-
[147]

Cheng J, Liu L, Xu G, Wang X, Zhang Z, Deng Y, Zang J, Chen Y, Cai Z, Yang X. MonSter: Marry Monodepth to Stereo Unleashes Power. *arXiv preprint arXiv:2501.08643*, 2025.

-
[148]

Jiang H, Lou Z, Ding L, Xu R, Tan M, Jiang W, Huang R. DEFOM-Stereo: Depth Foundation Model Based Stereo Matching. *arXiv preprint arXiv:2501.09466*, 2025.

-
[149]

Wang X, Yang H, Xu G, Cheng J, Lin M, Deng Y, Zang J, Chen Y, Yang X. StereoGen: High-quality Stereo Image Generation from a Single Image. *arXiv preprint arXiv:2501.08654*, 2025.

-
[150]

Guo X, Zhang C, Zhang Y, Nie D, Wang R, Zheng W, Poggi M, Chen L. Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data. *arXiv preprint arXiv:2411.14053*, 2024.

-
[151]

Watson J, Aodha OM, Turmukhambetov D, Brostow GJ, Firman M. Learning stereo from single images. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16*, Springer2020, 722–740.

-
[152]

Bleyer M, Rhemann C, Rother C. Patchmatch stereo-stereo matching with slanted support windows. In *Bmvc*, volume 11, 2011, 1–11.

-
[153]

Schönberger JL, Zheng E, Frahm JM, Pollefeys M. Pixelwise view selection for unstructured multi-view stereo. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14*, Springer2016, 501–518.

-
[154]

Schonberger JL, Frahm JM. Structure-from-motion revisited. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, 4104–4113.

-
[155]

Triggs B, McLauchlan PF, Hartley RI, Fitzgibbon AW. Bundle adjustment—a modern synthesis. In *Vision Algorithms: Theory and Practice: International Workshop on Vision Algorithms Corfu, Greece, September 21–22, 1999 Proceedings*, Springer2000, 298–372.

-
[156]

Hartley R, Zisserman A. *Multiple view geometry in computer vision*. Cambridge university press2003.

-
[157]

Montemerlo M, Thrun S, Koller D, Wegbreit B, et al.. FastSLAM: A factored solution to the simultaneous localization and mapping problem. *Aaai/iaai*, 2002, 593598: 593–598.

-
[158]

Davison AJ, Reid ID, Molton ND, Stasse O. MonoSLAM: Real-time single camera SLAM. *IEEE transactions on pattern analysis and machine intelligence*, 2007, 29(6): 1052–1067.

-
[159]

Tsai RY, Lenz RK, et al.. A new technique for fully autonomous and efficient 3 d robotics hand/eye calibration. *IEEE Transactions on robotics and automation*, 1989, 5(3): 345–358.

-
[160]

Daniilidis K. Hand-eye calibration using dual quaternions. *The International Journal of Robotics Research*, 1999, 18(3): 286–298.

-
[161]

Barnes C, Shechtman E, Finkelstein A, Goldman DB. PatchMatch: A randomized correspondence algorithm for structural image editing. *ACM Trans. Graph.*, 2009, 28(3): 24.

-
[162]

Wang K, Shen S. Mvdepthnet: Real-time multiview depth estimation neural network. In *2018 International conference on 3d vision (3DV)*, IEEE2018, 248–257.

-
[163]

Yang Z, Ren Z, Shan Q, Huang Q. Mvs2d: Efficient multi-view stereo via attention-driven 2d convolutions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, 8574–8584.

-
[164]

Kar A, Häne C, Malik J. Learning a multi-view stereo machine. *Advances in neural information processing systems*, 2017, 30.

-
[165]

Ji M, Gall J, Zheng H, Liu Y, Fang L. Surfacenet: An end-to-end 3d neural network for multiview stereopsis. In *Proceedings of the IEEE international conference on computer vision*, 2017, 2307–2315.

-
[166]

Cheng S, Xu Z, Zhu S, Li Z, Li LE, Ramamoorthi R, Su H. Deep stereo using adaptive thin volume representation with uncertainty awareness. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 2524–2534.

-
[167]

Gu X, Fan Z, Zhu S, Dai Z, Tan F, Tan P. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 2495–2504.

-
[168]

Yang J, Mao W, Alvarez JM, Liu M. Cost volume pyramid based depth inference for multi-view stereo. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, 4877–4886.

-
[169]

Yi H, Wei Z, Ding M, Zhang R, Chen Y, Wang G, Tai YW. Pyramid multi-view stereo net with self-adaptive view aggregation. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16*, Springer2020, 766–782.

-
[170]

Wang F, Galliani S, Vogel C, Pollefeys M. Itermvs: Iterative probability estimation for efficient multi-view stereo. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, 8606–8615.

-
[171]

Wang X, Zhu Z, Qin F, Ye Y, Huang G, Chi X, He Y, Wang X. MVSTER: Epipolar Transformer for Efficient Multi-View Stereo, 2022.

-
[172]

Cao C, Ren X, Fu Y. MVSFormer: Multi-View Stereo by Learning Robust Image Features and Temperature-based Depth. *Transactions of Machine Learning Research*, 2023.

-
[173]

Xu Y, Shi Z, Yifan W, Peng S, Yang C, Shen Y, Gordon W. GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation. *arxiv: 2403.14621*, 2024.

-
[174]

Wang J, Karaev N, Rupprecht C, Novotny D. Vggsfm: Visual geometry grounded deep structure from motion. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2024, 21686–21697.

-
[175]

Lu J, Huang T, Li P, Dou Z, Lin C, Cui Z, Dong Z, Yeung SK, Wang W, Liu Y. Align3r: Aligned monocular depth estimation for dynamic videos. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, 22820–22830.

-
[176]

Wang Q, Zhang Y, Holynski A, Efros AA, Kanazawa A. Continuous 3d perception model with persistent state. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, 10510–10522.

-
[177]

Wang H, Agapito L. 3D Reconstruction with Spatial Memory. *arXiv preprint arXiv:2408.16061*, 2024.

-
[178]

Duisterhof B, Zust L, Weinzaepfel P, Leroy V, Cabon Y, Revaud J. Mast3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. *arXiv preprint arXiv:2409.19152*, 2024.

-
[179]

Yifan W, Doersch C, Arandjelović R, Carreira J, Zisserman A. Input-level inductive biases for 3d reconstruction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, 6176–6186.

-
[180]

Guo H, Zhu H, Peng S, Lin H, Yan Y, Xie T, Wang W, Zhou X, Bao H. Multi-view reconstruction via sfm-guided monocular depth estimation. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, 5272–5282.

-
[181]

Zhang S, Wang J, Xu Y, Xue N, Rupprecht C, Zhou X, Shen Y, Wetzstein G. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. *arXiv preprint arXiv:2502.12138*, 2025.

-
[182]

Yang J, Sax A, Liang KJ, Henaff M, Tang H, Cao A, Chai J, Meier F, Feiszli M. Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass. *arXiv preprint arXiv:2501.13928*, 2025.

-
[183]

Jang W, Weinzaepfel P, Leroy V, Agapito L, Revaud J. Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors. *arXiv preprint arXiv:2503.17316*, 2025.

-
[184]

Li Z, Tucker R, Cole F, Wang Q, Jin L, Ye V, Kanazawa A, Holynski A, Snavely N. MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, 10486–10496.

-
[185]

Shriram J, Trevithick A, Liu L, Ramamoorthi R. RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion. *International Conference on 3D Vision (3DV)*, 2025.

-
[186]

Zhang J, Herrmann C, Hur J, Jampani V, Darrell T, Cole F, Sun D, Yang MH. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion. *arXiv preprint arxiv:2410.03825*, 2024.

-
[187]

Wang P, Tan H, Bi S, Xu Y, Luan F, Sunkavalli K, Wang W, Xu Z, Zhang K. PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape Prediction. *arXiv preprint arXiv:2311.12024*, 2023.

-
[188]

Zhang H, Shen C, Li Y, Cao Y, Liu Y, Yan Y. Exploiting temporal consistency for real-time video depth estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019, 1725–1734.

-
[189]

Wang Y, Luo L, Shen X, Mei X. DynOcc: Learning Single-View Depth from Dynamic Occlusion Cues. In *2020 International Conference on 3D Vision (3DV)*, IEEE2020, 514–523.

-
[190]

Chen S, Guo H, Zhu S, Zhang F, Huang Z, Feng J, Kang B. Video Depth Anything: Consistent Depth Estimation for Super-Long Videos. *arXiv preprint arXiv:2501.12375*, 2025.

-
[191]

Sörmann D, Dünser A, Fraundorfer F, Schmalstieg D. SimpleRecon: 3D Reconstruction Without 3D Convolutions. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, 12153–12162.

-
[192]

Curless B, Levoy M. A volumetric method for building complex models from range images. *Proceedings of the 23rd annual conference on Computer graphics and interactive techniques*, 1996: 303–312.

-
[193]

Wang J, Kar A, Fidler S. NeuRIS: Neural Reconstruction of Indoor Scenes Using Normal Priors. In *European Conference on Computer Vision*, Springer2022, 647–664.

-
[194]

Yu Z, Peng S, Niemeyer M, Sattler T, Geiger A. MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction. In *Advances in Neural Information Processing Systems*, volume 35, 2022, 3646–3658.

-
[195]

Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R. Nerf: Representing scenes as neural radiance fields for view synthesis. *Communications of the ACM*, 2021, 65(1): 99–106.

-
[196]

Kerbl B, Kopanas G, Leimkühler T, Drettakis G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics*, 2023, 42(4).

-
[197]

Guo J, Deng N, Li X, Bai Y, Shi B, Wang C, Ding C, Wang D, Li Y. Streetsurf: Extending multi-view implicit surface reconstruction to street views. *arXiv preprint arXiv:2306.04988*, 2023.

-
[198]

Miao S, Huang J, Bai D, Qiu W, Liu B, Geiger A, Liao Y. Efficient Depth-Guided Urban View Synthesis. In *European Conference on Computer Vision*, Springer2024, 90–107.

-
[199]

Fan Z, Wen K, Cong W, Wang K, Zhang J, Ding X, Xu D, Ivanovic B, Pavone M, Pavlakos G, et al.. InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds. *arXiv preprint arXiv:2403.20309*, 2024.

-
[200]

Zhou H, Lin L, Wang J, Lu Y, Bai D, Liu B, Wang Y, Geiger A, Liao Y. HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving. *arXiv preprint arXiv:2412.01718*, 2024.

-
[201]

Miao S, Huang J, Bai D, Yan X, Zhou H, Wang Y, Liu B, Geiger A, Liao Y. EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis. *arXiv preprint arXiv:2503.20168*, 2025.

-
[202]

Yan Y, Lin H, Zhou C, Wang W, Sun H, Zhan K, Lang X, Zhou X, Peng S. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In *European Conference on Computer Vision*, Springer2024, 156–173.

-
[203]

Sandström E, Tateno K, Oechsle M, Niemeyer M, Van Gool L, Oswald MR, Tombari F. Splat-slam: Globally optimized rgb-only slam with 3d gaussians. *arXiv preprint arXiv:2405.16544*, 2024.

-
[204]

Pan Y, Zhong X, Jin L, Wiesmann L, Popović M, Behley J, Stachniss C. PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map. *arXiv preprint arXiv:2502.05752*, 2025.

-
[205]

Matsuki H, Murai R, Kelly PH, Davison AJ. Gaussian splatting slam. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, 18039–18048.

-
[206]

Mao Y, Yu X, Zhang Z, Wang K, Wang Y, Xiong R, Liao Y. Ngel-slam: Neural implicit representation-based global consistent low-latency slam system. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, IEEE2024, 6952–6958.

-
[207]

OpenAI. Video Generation Models as World Simulators. [https://openai.com/index/video-generation-models-as-world-simulators/](https://openai.com/index/video-generation-models-as-world-simulators/), 2024, [Accessed 15-03-2025].

-
[208]

Hong W, Ding M, Zheng W, Liu X, Tang J. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. *arXiv preprint arXiv:2205.15868*, 2022.

-
[209]

Yang Z, Teng J, Zheng W, Ding M, Huang S, Xu J, Yang Y, Hong W, Zhang X, Feng G, et al.. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024.

-
[210]

Rigter M, Gupta T, Hilmkil A, Ma C. Avid: Adapting video diffusion models to world models. *arXiv preprint arXiv:2410.12822*, 2024.

-
[211]

Zhang Q, Zhai S, Bautista MA, Miao K, Toshev A, Susskind J, Gu J. World-consistent Video Diffusion with Explicit 3D Modeling. *arXiv preprint arXiv:2412.01821*, 2024.

-
[212]

Kang B, Yue Y, Lu R, Lin Z, Zhao Y, Wang K, Huang G, Feng J. How far is video generation from world model: A physical law perspective. *arXiv preprint arXiv:2411.02385*, 2024.

-
[213]

Xu Z, Xu Y, Yu Z, Peng S, Sun J, Bao H, Zhou X. Representing Long Volumetric Video with Temporal Gaussian Hierarchy. *ACM Transactions on Graphics*, 2024, 43(6).

-
[214]

Xu Z, Peng S, Lin H, He G, Sun J, Shen Y, Bao H, Zhou X. 4k4d: Real-time 4d view synthesis at 4k resolution. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2024, 20029–20040.

-
[215]

Xu Z, Xie T, Peng S, Lin H, Shuai Q, Yu Z, He G, Sun J, Bao H, Zhou X. EasyVolcap: Accelerating Neural Volumetric Video Research. *SIGGRAPH Asia 2023 Technical Communications*, 2023.

### Author biography

{biography}

[figures/authors/zhenxu.jpg]Zhen Xu is a forth-year PhD student in Computer Science at Zhejiang University, advised by Prof. Xiaowei Zhou and Sida Peng. He obtained his bachelor’s degree in Computer Science from Zhejiang University in 2022. His current research focuses on 3D/4D neural reconstruction and rendering, and volumetric videos.

{biography}

[figures/authors/hongyuzhou.jpg]Hongyu Zhou is currently pursuing his PhD at Zhejiang University, working under the guidance of Prof. Yiyi Liao at the X-Dim Lab. Prior to this, he was a master student at Zhejiang University supervised by Prof. Deng Cai and Prof. Xiaofei at the State Key Lab of CAD&CG. He received his bachelor’s degree from Tongji University in 2020.

{biography}

[figures/authors/sidapeng.png]Sida Peng is an assistant professor (ZJU-100 Young Professor) at Zhejiang University. He received his Ph.D. degree from College of Computer Science and Technology at Zhejiang University in 2023, supervised by Prof. Xiaowei Zhou and Prof. Hujun Bao, and obtained his bachelor degree in Information Engineering from Zhejiang University in 2018. He received the 2024 CCF Outstanding Doctoral Dissertation Award and was selected as the 2022 Apple Scholar in AI/ML.

{biography}

[figures/authors/haotonglin.jpg]Haotong Lin is a fourth-year PhD student in Computer Science at Zhejiang University, advised by Prof. Xiaowei Zhou and Sida Peng. He obtained his bachelor degree in Computer Science from Zhejiang University in 2021. In the summer of 2022, he had the opportunity to engage in a highly enjoyable collaboration with Noah Snavely at Cornell University.

{biography}

[figures/authors/haoyuguo.jpg]Haoyu Guo is a research scientist at Shanghai Artificial Intelligence Laboratory. He obtained his Ph.D. degree in Computer Science at Zhejiang University in 2025. His research interests include 3D vision, spatial intelligence and world model.

{biography}

[figures/authors/jiahaoshao.jpg]Jiahao Shao is a student researcher at X-D Lab advised by Yiyi Liao. He is also lucky to have collaboration with Matteo Poggi. Previously he obtained his B.Eng. degree in Automation from Zhejiang University in 2024.

{biography}

[figures/authors/peishanyang.jpg]Peishan Yang is a Master’s student of Computer Science at Zhejiang University (ZJU). His research interests focus on 3D reconstruction and scene understanding.

{biography}

[figures/authors/qinglinyang.jpg]Qinglin Yang is currently a master’s student at the School of Software Engineering, Zhejiang University, advised by Dr. Sida Peng and Dr. Xiaowei Zhou. He received his B.E. degree in the School of Remote Sensing and Information Engineering, Wuhan University in 2024. His research interests include 3D foundation models for perception, understanding and generation.

{biography}

[figures/authors/shengmiao.jpg]Sheng Miao is a PhD student at Zhejiang University. His research interests include 3D Reconstruction, Generative Model in autonomous driving.

{biography}

[figures/authors/xingyihe.png]Xingyi He is a fourth-year Ph.D. student in Computer Science at Zhejiang University, advised by Prof. Xiaowei Zhou. He received his bachelor’s degree from Huazhong University of Science and Technology in 2021. His research interests include 3D computer vision, robotics, image matching, and 3D reconstruction.

{biography}

[figures/authors/yifanwang.jpg]Yifan Wang is a MS student at Zhejiang University. His research interests include 3D vision, Dynamic Reconstruction, Neural Rendering.

{biography}

[figures/authors/yuewang.png]Yue Wang is a Professor in the Department of Control Science and Engineering at Zhejiang University. He received his Ph.D. from Zhejiang University in 2016, advised by Prof. Rong Xiong. His research focuses on persistent robot autonomy using AI, machine learning, and computer vision. He has developed various robotic systems, some of which have been applied in industry.

{biography}

[figures/authors/ruizhenhu.jpg]Ruizhen Hu is a Distinguished Professor and Deputy Director of the Visual Computing Research Center at Shenzhen University. She received her Ph.D. in Applied Mathematics from Zhejiang University in 2015, and was previously an Assistant Researcher at SIAT. Her research focuses on computer graphics and embodied AI, especially 3D modeling and agent interaction.

{biography}

[figures/authors/yiyiliao.jpg]Yiyi Liao is an assistant professor at Zhejiang University, leading the X-D Lab. She was previously a postdoctoral researcher in the Autonomous Vision Group at the University of Tübingen and the Max Planck Institute for Intelligent Systems. She received her Ph.D. from Zhejiang University in 2018 and her B.S. from Xi’an Jiaotong University in 2013. Her research interests include 3D computer vision, scene understanding, and 3D generative models.

{biography}

[figures/authors/xiaoweizhou.jpg]Xiaowei Zhou is a Professor in the College of Computer Science and the State Key Laboratory of CAD&CG at Zhejiang University. Before joining ZJU in 2017, he was a postdoctoral researcher at the GRASP Lab, University of Pennsylvania. His research focuses on 3D computer vision, including 3D reconstruction, pose estimation, motion capture, scene understanding, and their applications in mixed reality and robotics.

{biography}

[figures/authors/hujunbao.png]Hujun Bao is a Professor at the State Key Laboratory of CAD&CG and the College of Computer Science and Technology, Zhejiang University. He received his B.Sc. (1987) and Ph.D. (1993) in mathematics and applied mathematics from Zhejiang University. He leads the 3D graphics computing group, focusing on geometry computing, 3D visual computing, and real-time rendering. His contributed to systems such as VisioniX, ACTS, and 2D-to-3D video conversion.

### Graphical abstract

![Figure](figures/depth_foundation_model_fig.jpg)

Generated on Wed Oct 22 06:32:34 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)