[# Video Understanding with Large Language Models: A Survey Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, , Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, , Chao Huang, Zeliang Zhang, Pinxin Liu, Mingqian Feng, Feng Zheng, , Jianguo Zhang, , Ping Luo, , Jiebo Luo, , and Chenliang Xu Y. Tang, J. Bi, L. Song, S. Liang, D. Zhang, J. An, J. Lin, R. Zhu, A. Vosoughi, C. Huang, Z. Zhang, P. Liu, M. Feng, J. Luo, and C. Xu are with University of RochesterT. Wang and P. Luo are with The University of Hong KongS. Xu, T. Wang, F. Zheng, and J. Zhang are with Southern University of Science and TechnologyCorresponding to Y. Tang, J. Luo, and C. Xu ({yunlong.tang@, jluo@cs., chenliang.xu@}rochester.edu) ###### Abstract With the rapid growth of online video platforms and the escalating volume of video content, the need for proficient video understanding tools has increased significantly. Given the remarkable capabilities of large language models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of recent advances in video understanding that harness the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended multi-granularity (abstract, temporal, and spatiotemporal) reasoning combined with common-sense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into three main types: Video Analyzer $\times$ LLM, Video Embedder $\times$ LLM, and (Analyzer + Embedder) $\times$ LLM. We identify five subtypes based on the functions of LLMs in Vid-LLMs: LLM as Summarizer, LLM as Manager, LLM as Text Decoder, LLM as Regressor, and LLM as Hidden Layer. This survey also presents a comprehensive study of the tasks, datasets, benchmarks, and evaluation methods for Vid-LLMs. Additionally, it explores the extensive applications of Vid-LLMs in various domains, highlighting their remarkable scalability and versatility in real-world video understanding challenges. Additionally, it summarizes the limitations of existing Vid-LLMs and outlines directions for future research. For more information, readers are encouraged to visit the repository at https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding).

## I Introduction

![Figure](x1.png)

*Figure 1: The development of video understanding methods can be summarized into four stages: (1) Conventional Methods, (2) Early Neural Video Models, (3) Self-supervised Video Pretraining, and (4) Large Language Models for Video Understanding, i.e., Vid-LLMs. Their task-solving capability is continuously improving, and they possess the potential for further enhancement.*

We live in a multimodal world where video has become the predominant form of media. With the rapid expansion of online video platforms and the growing prevalence of cameras in surveillance, entertainment, and autonomous driving, video content has risen to prominence as a highly engaging and rich medium, outshining traditional text and image-text combinations in both depth and appeal.
This advancement has fueled an exponential increase in video production, with millions of videos being created every day.
However, manually processing such a sheer volume of video content is labor-intensive and time-consuming.
As a result, there is a growing need for tools to effectively manage, analyze, and process this abundance of video content.
To meet this need, video understanding methods have emerged that use intelligent analysis techniques to automatically recognize and interpret video content, significantly reducing the workload on human operators.
In addition, the ongoing development of these methods is improving their task-solving capabilities, enabling them to handle a wide range of video understanding tasks with increasing proficiency.

### I-A Development of Video Understanding Methods

The evolution of video understanding methods can be divided into four stages, as shown in Figure [1](#S1.F1):

#### I-A1 Conventional Methods

In the early stages of video understanding, handcrafted feature extraction techniques such as Scale-Invariant Feature Transform (SIFT) [[lindeberg2012scale]], Speeded-Up Robust Features (SURF) [[bay2008speeded]], and Histogram of Oriented Gradients (HOG) [[dalal2005histograms]] were used to capture key information in videos.
Background Subtraction [[sobral2014comprehensive]], optical flow methods [[tu2022optical]], and Improved Dense Trajectories (IDT) [[wang2013action, shu2015action]] were used to model the motion information for tracking.
Since videos can be viewed as time series data, temporal analysis techniques such as Hidden Markov Models (HMM) [[liu2003video]] have also been used to understand video content.
Before the popularity of deep learning, basic machine learning algorithms such as Support Vector Machines (SVM) [[sidenbladh2004detecting]], Decision Trees [[yuan2002automatic]], and Random Forests were also used in video classification and recognition tasks.
Cluster analysis [[chan2008modeling]] for classifying video segments, or Principal Component Analysis (PCA) [[bouwmans2014robust, hazelhoff2008video]] for data dimensionality reduction have also been commonly used methods for video analysis.

#### I-A2 Early Neural Video Models

Compared with classical methods, deep learning methods for video understanding possess superior task-solving capabilities. DeepVideo [[KarpathyCVPR14]] and [[earliest]] were early methods introducing a deep neural network, specifically a Convolutional Neural Network (CNN), for video understanding. However, the performance was not superior to the best handcrafted feature method due to the inadequate use of motion information. Two-stream networks [[feichtenhofer2016convolutional]] combined both CNN and IDT to capture the motion information to improve the performance, which verified the capability of deep neural networks for video understanding. To handle long-form video understanding, Long Short-Term Memory (LSTM) was adopted [[yue2015beyond]]. Temporal Segment Network (TSN) [[wang2016temporal]] was also designed for long-form video understanding by analyzing and aggregating video segments. Besides TSN, Fisher Vectors (FV) encoding [[sekma2015human]], Bi-Linear encoding [[diba2017deep]], and Vector of Locally Aggregated Descriptors (VLAD) [[mironicua2016modified]] encoding were introduced [[li2022transvlad]]. These methods improved performance on the UCF-101 [[soomro2012ucf101]] and HMDB51 [[kuehne2011hmdb]] datasets. Unlike two-stream networks, 3D networks started another branch by introducing 3D CNN to video understanding (C3D) [[tran2015learning]]. Inflated 3D ConvNets (I3D) [[carreira2017quo]] utilizes the initialization and the architecture of 2D CNN, Inception [[szegedy2015going]], to gain a huge improvement on the UCF-101 and HMDB51 datasets. Subsequently, people began employing the Kinetics-400 (K-400) [[kay2017kinetics]] and Something-Something [[goyal2017something]] datasets to evaluate the model’s performance in more challenging scenarios. ResNet [[he2016deep]], ResNeXt [[Xie_2017_CVPR]], and SENet [[Hu_2018_CVPR]] were also adapted from 2D to 3D, resulting in the emergence of R3D [[Hara_2017_ICCV]], MFNet [[Chen_2018_ECCV]], and STC [[Diba_2018_ECCV]]. To improve the efficiency, the 3D convolution has been decomposed into cascade 2D and 1D convolution in various studies (e.g., S3D [[S3D]], ECO [[zolfaghari2018eco]], P3D [[qiu2017learning]]). LTC [[LTC]], T3D [[T3D]], Non-local [[non-local]], and V4D [[zhang2019v4d]] focus on long-form temporal modeling, while CSN [[tran2019video]], SlowFast [[feichtenhofer2019slowfast]], and X3D [[feichtenhofer2020x3d]] tend to attain high efficiency. The introduction of Vision Transformers (ViT) [[dosovitskiy2021an]] promotes a series of prominent models (e.g., TimeSformer [[bertasius2021space]], VidTr [[li2021vidtr]], ViViT [[arnab2021vivit]], MViT [[fan2021multiscale]]).

![Figure](figures/timeline_2412.png)

*Figure 2: A comprehensive timeline depicting the development of video understanding methods with large language models (Vid-LLMs). This survey is based on advancements up to the end of June 2024.*

#### I-A3 Self-supervised Video Pretraining

Transferability [[li2023cross, wang2023feature]] in self-supervised pretraining models [[zhang2023rethinking]] for video understanding allows them to generalize across diverse tasks with minimal additional labeling, overcoming the early deep learning models’ requirements for extensive task-specific data.
VideoBERT [[sun2019videobert]] is an early attempt to perform video pretraining. Based on the bidirectional language model BERT [[kenton2019bert]], pertaining tasks are designed for self-supervised learning from video-text data. It tokenizes video features with hierarchical k-means. The pretrained model can be fine-tuned to handle multiple downstream tasks, including action classification and video captioning. Following the “pretraining-finetuning” paradigm, many studies on pretrained models for video understanding, especially video-language models, have emerged. They either use different architectures (ActBERT [[zhu2020actbert]], SpatiotemporalMAE [[feichtenhofer2022masked]], OmniMAE [[girdhar2023omnimae]], VideoMAE [[tong2022videomae]], MotionMAE [[yang2022self]]) or training strategies (MaskFeat [[wei2022masked]], VLM [[xu2021vlm]], ALPRO [[li2022align]], All-in-One transformer [[moritz2020all]], MaskViT [[gupta2022maskvit]], CLIP-ViP [[xue2022clip]], Singularity [[lei2022revealing]], LF-VILA [[sun2022long]], EMCL [[jin2022expectation]], HiTeA [[ye2023hitea]], CHAMPAGNE [[han2023champagne]]).

#### I-A4 Large Language Models for Video Understanding

Recently, large language models (LLMs) have advanced rapidly [[lyu2023gpt]]. The emergence of large language models pretrained on extensive datasets has introduced a novel in-context learning capability [[zhang2023dnagpt]]. This allows them to handle various tasks using prompts without the need for fine-tuning. ChatGPT [[OpenAIChatGPT]] is the first groundbreaking application built on this foundation. This includes capabilities like generating code and invoking tools or APIs of other models for their use. Many studies are exploring using LLMs like ChatGPT to call vision models APIs to solve the problems in the computer vision field, including Visual-ChatGPT [[wu2023visual]]. The advent of instruct-tuning has further enhanced these models’ ability to respond effectively to user requests and perform specific tasks [[liu2023visual]]. LLMs integrated with video understanding capabilities offer the advantage of more sophisticated multimodal understanding, enabling them to process and interpret complex interactions between visual and textual data. Similar to their impact in Natural Language Processing (NLP) [[zhao2023survey]], these models act as more general-purpose task solvers, adept at handling a broader range of tasks by leveraging their extensive knowledge base and contextual understanding acquired from vast amounts of multimodal data. This allows them to not only understand visual content but also reason about it in a way that is more aligned with human-like understanding. Many works also explore using LLMs in video understanding tasks, namely, Vid-LLMs.

### I-B Related Surveys

Previous survey papers either study specific sub-tasks in the area of video understanding or focus on methodologies beyond video understanding. For example, [[li2023multimodal]] surveys multimodal foundation models for general vision-language tasks, which includes both image and video applications. [[survey_vid_cap]] and [[survey_vid_act_reg]] focus on surveying video captioning and action recognition tasks, respectively. Other video understanding tasks, such as the video question answering and grounding, are not considered. Moreover, [[survey_vdm]], [[annepaka2024large]], and [[zhao2023survey]] survey video-related methodologies, such as video diffusion models and LLMs, lacking the concentration on video understanding.
[[madan2024foundation]] centers primarily on video foundation models, with insufficient attention given to language model-based approaches.
Despite the significant value to the community, previous survey papers leave a gap in surveying the general video understanding task based on large language models.
This paper fills this gap by comprehensively surveying the video understanding task using large language models.

### I-C Survey Structure

This survey is structured as follows:
Section [II](#S2) offers preliminaries for video understanding with LLMs, including a summary of various video understanding tasks that require handling different levels of granularity, their associated datasets, and evaluation metrics. The background of LLMs is also introduced in this section.
In Section [III](#S3), we delve into details of recent research leveraging LLMs for video understanding, presenting their unique approaches and impact in the field, where we divide these Vid-LLMs into three main categories, Video Analyzer $\times$ LLM and Video Embedder $\times$ LLM, and (Analyzer + Embedder) $\times$ LLM; and five sub-categories, LLM as Summarizer/Manager/Text Decoder/Regressor/Hidden Layer, shown as [Figure 5](#S3.F5). This section also includes the training strategies of Vid-LLMs.
Section [IV](#S4) adds more information about popular ways to evaluate Vid-LLMs, together with some benchmarks and performances of some Vid-LLMs on the most commonly used benchmarks.
Section [V](#S5) explores the application of Vid-LLMs across multiple significant fields and identifies unresolved challenges and potential areas for future research.

In addition to this survey, we have established a GitHub repository that aggregates various supporting resources for video understanding with large language models (Vid-LLMs). This repository, dedicated to enhancing video understanding through Vid-LLMs, can be accessed at [Awesome-LLMs-for-Video-Understanding](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding).

## II Preliminaries

In this section, we introduce the background of video understanding and Large Language Models (LLMs).

### II-A Video Understanding Tasks

Video understanding is a fundamental yet challenging task that has inspired the emergence of numerous tasks in a similar discipline aiming at interpreting complicated video content.
The pioneering work for video understanding includes video classification and action recognition approaches, which classify videos into class labels and action categories, respectively.
With the development of visual foundation models and expanding public datasets, current video understanding approaches can capture, analyze, and reason for more complicated video content.
For instance, video captioning, as a specific task of video understanding, not only requires the model to generate detailed descriptions of the video content, but the generated video captions should be logical and follow commonsense about the scenes depicted.
Additionally, the Video Question-Answering (VQA) task requires that the model understand the content and refer to external information to provide an accurate answer.
The development path of video understanding from simple classification to natural language comprehension and reasoning highlights a clear trend of the video understanding model towards near-human levels of video interpretation capability. We summarize the main tasks in video understanding as follows.

![Figure](x2.png)

*Figure 3: This figure categorizes tasks in video understanding, delineating the granularity required and the language involvement necessary for models to perform these tasks effectively. This diagram excludes tasks involving special modalities or specific, such as audio-visual and egocentric video understanding. Notably, the tasks presented could be unified into a question-answering paradigm, and all solved by generative large models, akin to recent advances in NLP.*

#### II-A1 Abstract Understanding Tasks

Video Classification, Action Recognition, Text-Video Retrieval, Video-to-Text Summarization, and Video Captioning.

-
•

Video Classification & Action Recognition: Video classification and action recognition classify videos based on class labels or activities and events categories within a video sequence. Datasets specifically introduced for these tasks include UCF-101 [[soomro2012ucf101]], HMDB51 [[kuehne2011hmdb]], Hollywood [[hollywood2]], ActivityNet [[caba2015activitynet]], Charades [[sigurdsson2016hollywood]], Kinetics-400 [[kay2017kinetics]], Kinetics-600 [[carreira2018short]], Kinetics-700 [[carreira2019short]], SomethingSomethingV2 [[goyal2017something]], HACS [[zhao2019hacs]], YouTube8M [[abuelhaija2016youtube8m]], and PortraitMode-400 [[han2024video_portraitmode-400]]. Usually, Top-K accuracy is adopted as the main metric for these tasks.

-
•

Text-Video Retrieval: Text-video retrieval task matches and retrieves relevant video clips based on the similarity between video clips and the input textual descriptions. Datasets like Kinetic-GEB [[wang2022geb+]], MSRVTT [[xu2017video]], DiDeMo [[anne2017localizing]], YouCook2 [[zhou2018towards]], and Youku-mPLUG [[xu2023youku]] are relevant to this task. The standard evaluation metric for this task is Recall at K (R@K), which measures the accuracy of the first K retrieved results.

-
•

Video-to-Text Summarization: Video-to-text summarization is a task that generates concise textual summaries of videos. The video summarization approaches are trained to extract and interpret key visual and audio content to produce coherent and informative summaries.
ViTT [[huang2020multimodal]], VideoXum [[lin2023videoxum]], VideoInstruct-100K [[maaz2023video]], and Instrcut-V2Xum [[hua2024v2xum]] are datasets that related to the task. Metrics of BLEU, METEOR, CIDEr, and ROUGE-L often evaluate this task.

-
•

Video Captioning: Video captioning generates descriptive and coherent textual captions of given videos. The video caption models usually use visual and auditory information from video to produce accurate and contextually relevant descriptions.
Notable datasets for this tasks are MSVD [[chen2011collecting]], MSR-VTT [[xu2016msr]], TGIF [[li2016tgif]], Charades [[sigurdsson2016hollywood]], Charades-Ego [[sigurdsson2018charades]], YouCook2 [[zhou2018towards]], Youku-mPLUG [[xu2023youku]], VAST-27M [[chen2023vast]], and VideoInstruct-100K [[maaz2023video]]. This task is often evaluated by metrics of BLEU, METEOR, CIDEr, and ROUGE-L.

-
•

Video QA: Video Question-Answering (VQA) aims to answer textual questions based on a given video, where the model analyzes visual and auditory information, understands the context, and eventually generates accurate responses.
Datasets involved in the QA task are VCR [[zellers2019recognition]], MSVD-QA [[xu2017video]], MSRVTT-QA [[xu2017video]], TGIF-QA [[jang2017tgif]], Pororo-QA [[kim2017deepstory]], TVQA [[lei2018tvqa]], ActivityNet-QA [[yu2019activitynet]], and NExT-QA [[NExT-QA]]. This task is evaluated using Top-1, Top-K accuracy.

#### II-A2 Temporal Understanding Tasks

-
•

Video Summarization: Video Summarization aims at condensing a long video into a shorter version while preserving essential content. F1-score, Spearman, and Kendall usually evaluate this task as metrics. Commonly-used datasets include SumMe [[gygli2014creating]], TVSum [[song2015tvsum]], Ads-1k [[tang2022multi]], VideoXum [[lin2023videoxum]], Instrcut-V2Xum [[hua2024v2xum]].

-
•

Video Highlight Detection: Video highlight detection aims at identifying and extracting the most important and interesting segments from a video. Commonly used datasets on this task include the YouTube Highlights [[youtubeHighlights]], the TV-Sum [[song2015tvsum]], and the Videos Titles in the Wild (VTW) [[VTWdataset]].

-
•

Temporal Action/Event Localization: This task aims at identifying the precise temporal segments of actions or events within a video. By analyzing sequential frames, models trained for this task must indicate when specific activities start and end. Datasets for Temporal Action/Event Localization involve THUMOS’14 [[THUMOS14]], ActivityNet-1.3 [[caba2015activitynet]], and UnAV-100 [[geng2023dense_unav100]].

-
•

Temporal Action Proposal Generation: Temporal action proposal generation involves generating candidate segments within a video that are likely to contain actions or events. Relevant datasets like THUMOS’14 [[THUMOS14]], ActivityNet [[caba2015activitynet]], and Charades [[sigurdsson2016hollywood]] are used for the training and evaluation for this task.

-
•

Video Temporal Grounding: Video temporal grounding is the task of locating specific moments or intervals within a video that correspond to a given textual query. This process involves aligning linguistic descriptions with visual content, enabling precise identification of relevant segments for applications in video search and content analysis.
Common benchmarks are Charades-STA [[Gao_2017_ICCV]], ViTT [[huang2020multimodal]], DiDeMo [[anne2017localizing]], and PU-VALOR [[tang2024avicuna]]. The metrics of R1@0.5 and R1@0.7 often evaluate this task.

-
•

Moment Retrieval: Moment retrieval is the task of identifying and extracting precise video segments that correspond to a given textual or visual query, which aligns semantic content between queries and video frames. DiDeMo [[anne2017localizing]] is a dataset for this task.

-
•

Generic Event Boundary Detection: Generic event boundary detection involves identifying certain frames in a video where significant changes occur and splitting videos based on different events or activities. Kinetics-GEBD [[shou2021generic]] is a widely-used dataset for this task.

-
•

Generic Event Boundary Captioning & Grounding: Generic event boundary captioning and grounding involve identifying and describing the transition points between significant events in a video, where Kinetics-GEB+ [[wang2022geb+]] is the dataset for this task.

-
•

Dense Video Captioning: Dense video captioning [[shao2022region, Wang_2021_ICCV, shao2023textual, long2023capdet, shao2024dcmstrd]] aims at generating detailed and continuous textual descriptions for multiple events and actions occurring throughout a video. Evaluation metrics like BLEU, METEOR, CIDEr, and ROUGE-L are used to evaluate this task. Relevant datasets are ActivityNet Captions [[krishna2017dense]], VidChapters-7M [[yang2023vidchapters]], YouCook2 [[zhou2018towards]], and ViTT [[huang2020multimodal]].

#### II-A3 Spatiotemporal Understanding Tasks

-
•

Object Tracking: Object tracking aims at continuously identifying and following the position of specific objects within a video over time. A good tracking model should maintain accurate and consistent trajectories of objects, even for videos with occlusions, changes in appearance, and motions. Benchmarks like OTB [[OTBbenchmark]], UAV [[UAVbenchmark]], and VOT [[VOTbenchmark]] are commonly used for this task.

-
•

Re-Identification: Re-Identification (ReID) is the task of recognizing and matching individuals or objects across different video frames or camera views. Common datasets in ReID are Market-1501 [[Market1501]], CUHK03 [[cuhk03]], MSMT17 [[wei2018person]], and DukeMTMC-reID [[dukeMTMCreID]].

-
•

Video Saliency Detection: Video saliency detection aims at identifying the most visually important and attention-grabbing regions in a video [[moskalenko2024aim]]. This task highlights areas that stand out due to factors like motion, contrast, and unique features.
Relevant datasets to this task are DHF1K [[wang2018revisiting_dhf1k]], Hollywood-[[sigurdsson2016hollywood]], UCF-Sports [[ucfsport]], AVAD [[avad_sal]], Coutrot1 [[coutrot1_2014saliency]], Coutrot2 [[coutrot2_2016multimodal]], ETMD [[ETMD_koutras2015perceptually]], and SumMe [[gygli2014creating]].

-
•

Video Object Segmentation: Video object segmentation aims at partitioning a video into segments that correspond to individual objects, accurately delineating their boundaries over time. YouTube-VOS [[youtubevos]] and DAVIS [[perazzi2016benchmark]] are datasets related to this task.

-
•

Video Instance Segmentation: Video instance segmentation is the task of identifying, segmenting, and tracking each unique object instance within a video. YouTube-VIS [[youtubevis]] and Cityscapes-Seq [[cityscape]] are two common benchmarks for this task.

-
•

Video Object Referring Segmentation: Video object referring segmentation involves segmenting specific objects in a video based on language descriptions. It identifies and isolates the referred objects accurately across frames, where MeViS [[ding2023mevis]] is a common benchmark for this task.

-
•

Spatiotemporal Grounding: Spatiotemporal grounding aims to identify and localize specific objects or events within a video’s spatial and temporal dimensions based on a given query.
Datasets like Vid-STG [[zhang2020does]], HC-STVG [[hc-stvg-tang2021human]], Ego4D-MQ and Ego4D-NLQ [[grauman2022ego4d]] are proposed to aid the training and testing for this task.

### II-B Background for LLMs

Language models are trained to learn a joint probability distribution $p(x_{1:L})$ with a sequence of text tokens $x_{1:L}$. This joint distribution is usually equivalent to a product of the conditional probabilities conditioned on each token with the chain rule:

$$p(x_{1:L})=\prod_{i=1}^{L}p(x_{i}|x_{1:i-1}),$$ \tag{1}

where $L$ is the sequence length.

Large Language Models (LLMs) refer to language models with a large number of parameters, e.g., billions. The architecture of LLMs incorporates a text tokenizer and multiple self-attention layers. LLMs are trained in a teacher-forcing manner to predict the next token’s probability, where the generation process utilizes the autoregressive paradigm:

$$\mathcal{M}(x_{1:i-1})=p(x_{i}\mid x_{1:i-1}),$$ \tag{2}

where $\mathcal{M}$ represents an LLM.

Decoding strategies dictate how to harness the next token probability and select the next token $y_{t}$ from the set $S$ of all possible tokens in the vocabulary, which includes special tokens such as $$, $$, and $$. Greedy decoding, the simplest strategy, selects the token with the highest probability, formalized as:

$$x_{t}=\operatorname*{arg\,max}_{s\in S}\log p_{\mathcal{M}}(s\mid\bm{x}_{1:t-1}).$$ \tag{3}

Besides deterministic strategies, sampling strategies that randomly select the next tokens using model probability are also popular in real applications. These strategies provide diverse outputs and enable self-consistency methods.

Large language models typically exhibit the following characteristics:

-
•

Scaling Laws [[kaplan2020scaling]]: With a significant extension of the model size (number of parameters), the pertaining data size, and computational resources, the performance of the model exhibits a pattern of regular growth, which can help researchers and engineers predict performance improvements or make effective decisions on model design and training.

-
•

Emergent Abilities [[zhao2023survey]]: When the parameter size and volume of training data for a large language model exceed a certain magnitude, some novel capabilities emerge, such as in-context learning, instruction following, and step-by-step reasoning.
In-context learning allows the model to learn and make predictions based on the context provided within the input text without requiring explicit retraining. Instruction following enables the model to perform tasks based on natural language instructions. Step-by-step reasoning, e.g. chain-of-thought (CoT), allows the model to follow a logical sequence of steps to arrive at a conclusion, which is particularly useful for solving complex problems.

LLMs possess extensive generalization abilities that can be applied to various downstream tasks, including multi-modal tasks. Multimodal Large Language Models (MLLMs) [[zhu2023minigpt, liu2023visual, chen2023shikra, xuan2023pink, hua2024finematch]] typically incorporate multimodal encoders, cross-modality aligners, and an LLM core structure. By combining multimodal encoders with the LLM, MLLMs excel at integrating visual and linguistic contexts to produce detailed content.

## III Vid-LLMs

In this section, we introduce a novel taxonomy of Vid-LLMs, providing a comprehensive overview of their classification. Following this, we explore the diverse training strategies that empower Vid-LLMs to achieve their capabilities.

![Figure](x3.png)

*Figure 4: The figure illustrates three primary frameworks for Vid-LLMs: (1) Video Analyzer $\times$ LLM, where video analyzers convert video inputs to textual analysis for the LLM; (2) Video Embedder $\times$ LLM, where video embedders generate vector representations (embeddings) for the LLM to process; (3) (Analyzer + Embedder) $\times$ LLM, a hybrid approach combining both analyzers and embedders to provide the LLM with textual analysis and embeddings. The arrows indicate the direction of information flow, with dashed arrows representing optional paths. Blue arrows denote textual information flow, while red arrows denote embeddings.*

*Figure 5: Taxonomy of Video Understanding with Large Language Models (Vid-LLMs), consists of Video Analyzer $\times$ LLM, Video Embedder $\times$ LLM and (Analyzer + Embedder) $\times$ LLM, and the sub-categories are LLM as Summarizer/Manager/Text Decoder/Regressor/Hidden Layer. Font color indicates the granularity of video understanding supported by the Vid-LLMs: black for abstract understanding, red for temporal understanding, and blue for spatiotemporal understanding.*

### III-A Taxonomy

Based on the method of processing input video, we categorize Vid-LLMs into three primary types: Video Analyzer $\times$ LLM, Video Embedder $\times$ LLM, and (Analyzer + Embedder) $\times$ LLM. Each category represents a unique approach to integrating video processing with LLMs, as illustrated in [Figure 4](#S3.F4).

#### III-A1 Video Analyzer $\times$ LLM

Video Analyzer is defined as a module that takes video input and outputs an analysis of the video, typically in text form, which facilitates LLM processing. This text may include video captions, dense video captions (detailed descriptions of all events in the video with timestamps), object tracking results (labels, IDs, and bounding boxes of objects), as well as transcripts of other modalities present in the video, such as speech recognition results from ASR or subtitle recognition results from OCR. The text generated by the Video Analyzer can be directly fed into the subsequent LLM, inserted into pre-prepared templates before being fed into the LLM, or converted into a temporary database format for the LLM to retrieve later.

![Figure](x4.png)

*Figure 6: Four types of Vid-LLMs fine-tuning strategies, classified according to their specific training methods: LLM fully fine-tuning, fine-tuning with connective adapters, insertive adapters, and hybrid methods.*

For the Video Analyzer $\times$ LLM category, we further create two subcategories, LLM as Summarizer and LLM as Manager, based on the function of the LLM within the Vid-LLM system:

-
•

LLM as Summarizer: In this subcategory, the primary function of the LLM is to summarize the analysis obtained from the Video Analyzers. The summarization approach varies based on the prompts provided to the LLM, ranging from highly condensed summary texts and captions to comprehensive summaries for answering specific questions. Notably, in Video Analyzer $\times$ LLM as Summarizer systems, the information flow is usually unidirectional (see [Figure 4](#S3.F4)), with data flowing from the video to the Video Analyzer and then to the LLM, without any reverse process. Examples of Vid-LLMs in the Video Analyzer $\times$ LLM as Summarizer category include: LaViLa [[zhao2023lavila]], VLog [[VLogGithub]], VAST [[chen2023vast]], AntGPT [[zhao2023antgpt]], VIDOSC [[Xue2024VIDOSC]], Grounding-Prompter [[Chen2023Grounding-prompter]], LLoVi [[Zhang2024LLoVi]], Video ReCap [[Islam2024VideoReCap]], MVU [[Ranasinghe2024MVU]], LangRepo [[Kahatapitiya2024LangRepo]], IG-VLM [[Kim2024IG-VLM]], MoReVQA [[min2024morevqa]], MM-Screenplayer [[wu2024mmscreenplayer]], GIT-LLaVA [[kalarani2024gitllava]], etc.

-
•

LLM as Manager: In this subcategory, the LLM primarily coordinates the overall system’s operation. It may actively generate commands to invoke various Video Analyzers to produce the desired results for the user, call the Video Analyzer and further process the obtained analysis before returning it to the user, or engage in multiple rounds of interaction with the Video Analyzer. Compared to the LLM as Summarizer category, the LLM as Manager category is more flexible and can be distinguished by its information flow complexity. Examples of Vid-LLMs in the Video Analyzer $\times$ LLM as Manager category include: ViperGPT [[suris2023vipergpt]], Video ChatCaptioner [[chen2023videochatcap]], ChatVideo [[wang2023chatvideo]], AssistGPT [[gao2023assistgpt]], HuggingGPT [[shen2024hugginggpt]], Hawk [[tang2024hawk]], ProViQ [[Choudhury2023ProViQ]], LifelongMemory [[Wang2024LifelongMemory]], DoraemonGPT [[Yang2024DoraemonGPT]], KEPP [[Nagasinghe2024KEPP]], VURF [[Mahmood2024VURF]], VideoAgent (by Stanford) [[Wang2024VideoAgent]], VideoAgent (by PKU) [[Fan2024VideoAgent]], TV-TREES [[S2024TV-TREES]], SCHEMA [[Niu2024SCHEMA]], RAVA [[Cao2024RAVA]], GPTSee [[Sun2024GPTSee]], TraveLER [[shang2024traveler]], LAVAD [[zanella2024lavad]], VideoTree [[wang2024videotree]], LVNet [[park2024lvnet]], OmAgent [[zhang2024omagent]], DrVideo [[ma2024drvideo]], etc.

#### III-A2 Video Embedder $\times$ LLM

A Video Embedder typically refers to a visual backbone/video encoder, such as ViT or CLIP, used to convert input videos into vector representations, known as video embeddings or video tokens. Some Embedders encode other modalities within the video, such as audio (e.g., CLAP [[elizalde2023clap]]), which are also categorized under Video Embedder here (note that we do not consider the LLM’s tokenizer as an embedder). Unlike the text generated by the Video Analyzer, the vectors generated by the Video Embedder cannot be directly utilized by the LLM and usually require an adapter to map these embeddings from the vision’s (or other modalities’) semantic space to the text semantic space of the LLM’s input tokens.

For the Video Embedder $\times$ LLM category, we also classify them into subcategories based on the LLM’s function within the Vid-LLM system:

-
•

LLM as Text Decoder: In this subcategory, the LLM receives embeddings from the Video Embedder as input and decodes them into text outputs based on prompts or instructions. These tasks generally do not require fine-grained understanding or precise spatiotemporal localization, focusing mainly on general QA or captioning. Thus, the Vid-LLM behaves like a standard LLM during decoding. Examples of Vid-LLMs in the Video Embedder $\times$ LLM as Text Decoder category include: VideoLLM [[chen2023videollm]], Otter [[li2023otter]], Video-LLaMA [[zhang2023video]], Video-ChatGPT [[maaz2023video]], VALLEY [[luo2023valley]], Macaw-LLM [[lyu2023macaw]], MovieChat [[song2023moviechat]], Video-LLaVA [[lin2023video]], Chat-UniVi [[Jin2024Chat-UniVi]], Vista-LLaMA [[Ma2023Vista-LLaMA]], VILA [[Lin2024VILA]], GPT4Video [[wang2023gpt4video]], MovieLLM [[Song2024MovieLLM]], InternVideo2 [[Wang2024InternVideo2]], MiniGPT4-Video [[ataallah2024minigpt4video]], VideoChat2 [[li2024mvbench]], VideoLLaMA 2 [[cheng2024videollama2]], etc. See [Figure 5](#S3.F5) for the complete list.

-
•

LLM as Regressor: In this subcategory, the LLM receives embeddings from the Video Embedder as input and, like the Text Decoder, can output text. However, unlike the Text Decoder, the LLM as Regressor can also predict continuous values, such as timestamp localization in videos and bounding box coordinates for object trajectories, functioning similarly to a regressor performing regression tasks, even though it is fundamentally performing classification. Examples of Vid-LLMs in the Video Embedder $\times$ LLM as Regressor category include: VTimeLLM [[huang2023vtimellm]], SeViLA [[Yu2023SeViLA]], TimeChat [[Ren2024TimeChat]], GroundingGPT [[Li2024GroundingGPT]], OmniViD [[Wang2024OmniViD]], LITA [[Huang2024LITA]], HawkEye [[Wang2024HawkEye]], Elysium [[wang2024elysium]], AVicuna [[tang2024avicuna]], V2Xum-LLaMA [[hua2024v2xum]], VLM4HOI [[bansal2024hoiref]], VideoLLM-online [[chen2024videollmonline]], Holmes-VAD [[zhang2024holmesvad]], etc.

-
•

LLM as Hidden Layer: In this subcategory, the LLM also receives video embeddings as input but does not directly output text. Instead, it connects to a specially designed task-specific head to perform actual regression tasks, such as event time localization or object bounding box prediction in videos, while maintaining the LLM’s text output capability. Examples of Vid-LLMs in the Video Embedder $\times$ LLM as Hidden Layer category include: GPT4Video [[wang2023gpt4video]], OneLLM [[Han2023OneLLM]], VidDetours [[Ashutosh2024VidDetours]], Momentor [[Qian2024Momentor]], VTG-GPT [[Xu2024VTG-GPT]], VITRON [[fei2024vitron]], VTG-LLM [[guo2024vtgllm]], etc.

#### III-A3 (Analyzer + Embedder) $\times$ LLM

This category of Vid-LLMs is relatively rare. As the name suggests, it involves simultaneously using a Video Analyzer to obtain textual analysis of the video and a Video Embedder to encode the video into embeddings. The LLM receives both types of inputs along with other prompts/instructions and outputs responses to complete tasks. The subcategories here can flexibly be any of the Summarizer/Manager/Text Decoder/Regressor/Hidden Layer categories. Vid-LLMs in the (Analyzer + Embedder) $\times$ LLM category include: Vid2Seq [[yang2023vid2seq]], VideoChat [[li2023videochat]], MM-VID [[lin2023mmvid]], Auto-AD II [[han2023autoad]], Vamos [[Wang2024Vamos]], PG-Video-LLaVA [[munasinghe2023pg]], MM-Narrator [[zhang2023mm-narrator]], SUM-shot [[Han2023SUM-shot]], Merlin [[yu2023merlinempowering]], Uni-AD [[Wang2024Uni-AD]], Vriptor [[yang2024vript]], etc.

### III-B Training Strategies for Vid-LLMs

#### III-B1 Training-free Vid-LLMs

Many Vid-LLMs systems are built on powerful LLMs with strong zero-shot, in-context learning, and Chain-of-Thought capabilities. These systems do not require training in the parameters of the LLM or other modules. Most Vid-LLMs in the Video Analyzer $\times$ LLM category are training-free because the information from the video and other accompanying modalities has already been parsed into text. At this point, the video understanding task has been transformed into a text understanding task. Since LLMs can unify almost all NLP tasks into generation tasks, they can also handle many video understanding tasks. SlowFast-LLaVA
[[xu2024slowfastllava]] is a training-free Video LLM that uses a two-stream input design to capture both spatial semantics and temporal context without fine-tuning and demonstrates capabilities across various video understanding benchmarks.

*TABLE I: Comparison of video understanding models with large language models, sorted by their release dates. The table presents key details for each method, including the number of training frames, video embedders, utilization of audio information, model adaptation approaches, computational resources, the specific large language model employed, and the corresponding number of parameters. Entries marked with a hyphen (“-”) indicate undisclosed details in the respective papers.*

*TABLE II: Contiune of [Table I](#S3.T1).*

#### III-B2 Fine-tuning Vid-LLMs

In contrast to most Vid-LLMs in the Video Analyzer $\times$ LLM category being training-free, almost all Vid-LLMs in the Video Embedder $\times$ LLM category undergo fine-tuning. The common methods for fine-tuning Vid-LLMs are categorized based on the types of adapters used during fine-tuning into four main types: LLM Fully Fine-tuning, Connective Adapter Fine-tuning, Insertive Adapter Fine-tuning, and Fine-tuning with Hybrid Adapters. An adapter is a small, trainable module added to a large model for fine-tuning. By only updating the parameters of these modules, the model can adapt to specific tasks without changing the entire model’s parameters, achieving efficient parameter updates and task adaptation while conserving computational resources. Illustrations of each type are shown in [Figure 6](#S3.F6).

-
•

LLM Fully Fine-tuning: This fine-tuning method does not use any adapters but instead employs supervised training with a lower learning rate, updating all the parameters in the LLM. This method allows the Vid-LLM to fully adapt to the respective task and achieve good performance, especially when the target task is quite different from the pretraining tasks. For end-to-end Vid-LLMs, especially those in the Video Embedder $\times$ LLM category, the Video Embedder may also be fine-tuned for more comprehensive learning. However, this method consumes more computational resources than adapter-based fine-tuning methods and may potentially impair the inherent capabilities of the LLM, such as zero-shot and in-context learning. Vid-LLM adopted LLM Fully Fine-tuning include AV-LLM [[Shu2023AV-LLM]] and Vid2Seq [[yang2023vid2seq]]. In [[Shu2023AV-LLM]], there are both fully fine-tuning and adapter fine-tuning versions of Vid-LLMs, and the former’s performance is better than the latter.

-
•

Connective Adapter Fine-tuning: Here, the term “Connective” refers to adapters that bridge the Video Embedder and the LLM externally, enabling information from the video to flow into the LLM through the Connective Adapter. As illustrated in [Figure 6](#S3.F6), during training, the parameters of both the Video Embedder and the LLM are frozen, and only the parameters of the Connective Adapter are updated. Common Connective Adapters include MLP/Linear Layer and Q-former [[li2023blip]], their combinations, etc., whose primary function is to map video embeddings from the visual semantic space to the text semantic space of the LLM input tokens (i.e., modality alignment). Typically, fine-tuning only the Connective Adapter does not alter the LLM’s inherent behavior.

-
•

Insertive Adapter Fine-tuning: As the name suggests, Insertive Adapters are inserted into the LLM itself. Similar to using Connective Adapters, during training, the parameters of the Video Embedder and the LLM are frozen, and only the parameters of the Insertive Adapter are updated. Insertive Adapters, often based on LoRA, affect the LLM’s behavior because they are added to the existing LLM parameters. This type of adapter is almost always present in Vid-LLMs classified as Video Embedder $\times$ LLM as Regressor and Video Embedder $\times$ LLM as Hidden Layer, as these types of Vid-LLMs require changes in the LLM’s behavior, such as outputting continuous prediction values.

-
•

Fine-tuning with Hybrid Adapters: Many Vid-LLMs use a combination of Connective and Insertive Adapters to achieve both modality alignment and changes in the LLM’s inherent behavior. Vid-LLMs employing Hybrid Adapters typically use multi-stage fine-tuning. A common approach is to fine-tune only the Connective Adapter in the first stage for modality alignment. In the second stage, the already fine-tuned Connective Adapter is frozen, the training task (from alignment task to target task) and the training data (from data used for modality alignment to data required for the target task) are changed, and only the parameters of the Insertive Adapter are updated. There are also single-stage approaches where both Connective and Insertive Adapters are updated simultaneously.

## IV Benchmarks and Evaluation

This section provides an overview of the evaluation methods for video question-answering models and related tasks, categorized into three types: closed-ended evaluation, open-ended evaluation, and other evaluation methods, which are shown in [Table III](#S4.T3).
Closed-ended evaluations rely on questions with predefined answers or formats, including multiple-choice questions and structured formats that allow for straightforward scoring. Open-ended evaluations involve questions without predefined answer options, often requiring more sophisticated scoring methods, including LLM-based evaluations. Other evaluation methods address specialized video understanding capabilities such as temporal/spatiotemporal reasoning.

### IV-A Closed-ended Evaluation

Closed-ended evaluations use pre-defined answers or structured formats [[yin2023survey]]. These include multiple-choice questions (TVQA [[lei2018tvqa]], How2QA [[How2QA]], STAR [[wu2021star]]) and questions with structured formats for direct comparison with ground truth (MSRVTT-QA [[xu2017video]], MSVD-QA [[xu2017video]], MVBench [[li2024mvbench]]). Multiple-choice performance is evaluated via accuracy percentages, while structured formats use metrics like CIDEr [[vedantam2015cider]], METEOR [[banerjee2005meteor]], ROUGE [[lin2004rouge]], and SPICE [[anderson2016spice]] to compare predictions with ground truth.
Notable benchmarks include MSRVTT-QA [[xu2017video]], TVQA [[lei2018tvqa]], MVBench [[li2024mvbench]], EgoSchema [[mangalam2023egoschema]], and Video-MME [[fu2024video-mme]]. Each targets different video understanding aspects: TGIF-QA [[jang2017tgif]] focuses on action recognition, state transition, frame-level QA, and counting; ActivityNet-QA [[yu2019activitynet]] covers motion, spatial, temporal, and free-form dimensions; VidComposition [[tang2024vidcomposition]] emphasizes compositional reasoning; while NExT-QA [[NExT-QA]] and MLVU [[zhou2024mlvu]] include causal and temporal action reasoning.
These diverse question types test various reasoning abilities, though many benchmarks still exhibit domain biases toward common scenarios and lack diversity in rare events or unusual contexts.

*TABLE III: The comparison of various benchmarks includes several important aspects: the total number of videos, the number of clips, the average duration of the videos, the number of QA pairs, and video content.*

### IV-B Open-ended Evaluation

Open-ended evaluation involves questions without pre-defined options or structured formats. While ground-truth answers serve as references, scoring methods are more sophisticated than option selection or string matching. GPT-3.5/4 models often evaluate predictions by comparing them with reference answers.
Notable open-ended benchmarks include MovieChat-1K [[song2023moviechat]], MLVU [[zhou2024mlvu]], NExT-QA [[NExT-QA]], VELOCITI [[saravanan2024velociti]], and EAGLE [[bi2024eagle]]. These require more complex responses demonstrating deeper reasoning. CinePile [[rawal2024cinepile]] incorporates analytical tasks like character dynamics and narrative analysis.
The most popular GPT-based evaluation methods, proposed in [[maaz2023video]], are Open-end Zero-shot Video QA Evaluation and Video-based Generative Performance Benchmarking. Performance comparisons of Vid-LLMs on these metrics are shown in [Table V](#S4.T5).
Originally closed-ended benchmarks like MSRVTT-QA [[xu2017video]], MSVD-QA [[xu2017video]], TGIF-QA [[jang2017tgif]], and ActivityNet-QA [[yu2019activitynet]] can be repurposed as open-ended in GPT-based evaluations, as LLMs generate free-form responses that GPT models compare to references.
These methods have limitations: evaluation scores change with GPT version updates, making cross-study comparisons difficult; results depend heavily on prompt engineering; and LLM evaluators may favor responses similar to their generation patterns rather than objectively assessing quality.

### IV-C Other Evaluations

Other benchmarks evaluate fine-grained temporal and spatiotemporal understanding. Dense captioning generates [[shao2022region, Wang_2021_ICCV, shao2023textual, long2023capdet, shao2024dcmstrd]] detailed descriptions for multiple video events/objects, using BLEU, METEOR, and CIDEr metrics that assess both temporal localization and descriptive accuracy. Vid-LLMs’ performance of dense video captioning on ActivityNet Captions [[krishna2017dense]] is shown in [Table IV](#S4.T4).
Several Vid-LLMs have already achieved performance comparable to traditional task-specific models in dense video captioning.
Video temporal grounding localizes specific moments based on textual queries, evaluated using tIoU and Recall@K. Spatiotemporal grounding extends this to localize in both space and time, assessed via spatiotemporal IoU and mAP.
Object tracking [[wang2024elysium, Li2024GroundingGPT]] follows objects across frames, evaluated using precision, success rate, and tracking accuracy. Video saliency detection [[tang2024cardiff]] identifies visually salient regions, evaluated with AUC-J, NSS, etc.
These tasks rely on temporal or spatiotemporal annotations as ground-truth, with metrics like IoU, Recall@K, and mAP widely adopted. Human evaluation is also used for subjective aspects, though this is labor-intensive and time-consuming.

As for qualitative evaluation, several approaches can effectively assess Vid-LLMs’ performance in addition to numerical metrics.
Error analysis [[li2024mvbench, hua2024mmcomposition]] for open-ended QA and difference comparisons [[huang2023vtimellm, tang2024avicuna]] between model outputs and ground truth annotations (e.g., intervals) for temporal/spatiotemporal understanding provide insights into model limitations. Attention visualization [[bi2024unveiling]] reveals what visual elements the models prioritize when generating responses. Self-explanation [[hua2024mmcomposition, tang2024vidcomposition]], where models justify their answers for closed-ended benchmarks, offers valuable insights into reasoning processes and potential misconceptions. Human studies, though resource-intensive, remain helpful in finding models that reflect human preferences.

*TABLE IV: Comparison of Vid-LLMs and conventional models (non-LLM-based) on dense video captioning models on ActivityNet Captions dataset.*

*TABLE V: This table comprehensively compares various Vid-LLMs across multiple open-end zero-shot video question answering and video-based generative performance benchmarks. It includes GPT-based metrics for MSVD-QA, MSRVTT-QA, and ActivityNet-QA datasets, as well as scores for Correctness of Information, Detail Orientation, Contextual Understanding, Temporal Understanding, and Consistency aspects in video-based generative performance.*

### IV-D Analysis of Model Performance

Analyzing the correlation between model attributes and benchmark performance reveals several key factors driving recent improvements in Vid-LLMs. From Tables [IV](#S4.T4) and [V](#S4.T5), we observe that models built on larger and more recent foundation LLMs (e.g., IG-VLM with 34B parameters) consistently outperform their smaller counterparts, particularly in zero-shot VideoQA tasks. Models employing more powerful visual embedders such as EVA-CLIP or ViT-G architectures (notably in PLLaVA, IG-VLM, and Video LLaMA 2) demonstrate superior performance across both dense captioning and QA benchmarks. The frame sampling strategy also significantly impacts results, with high performers on temporal tasks (like VTimeLLM, AVicuna, and ST-LLM) typically processing more frames (100+) than general understanding models, while sophisticated adaptation mechanisms beyond simple projection layers (such as Q-formers or cross-attention) contribute to better contextual understanding. Performance gains stem from a combination of stronger foundation models, better visual encoders, appropriate temporal modeling, and more sophisticated bridging architectures rather than any single innovation.

## V Applications and Future Directions

### V-A Application Scenarios

Vid-LLMs have revolutionized various sectors by enabling advanced video and language processing capabilities. We outlines their diverse applications, demonstrating the extensive and transformative impact of Vid-LLMs across industries.

#### V-A1 Media and Entertainment

-
•

Online Video Platforms and Multimedia Information Retrieval: Vid-LLMs significantly enhance search algorithms [[mao2023large]], generate context-aware video recommendations [[ju2022prompting]], and aid in natural language tasks such as subtitle generation and translation [[yang2023vid2seq]], contributing to online video platforms and multimedia retrieval systems. Their capabilities in analyzing videos for specific keyword retrieval [[zhao2023lavila, jin2023text, jin2023diffusionret]] improve intelligent recommendation systems. In the multimedia fields, it combines videos in domains like music [[xu2023launchpadgpt]], avatar [[song2021tacr, song2023emotional, song2024tri, song2021fsft]], and scene [[Song_2023_CVPR]], to assist with content generation.

-
•

Video Summarization and Editing: Vid-LLMs are integral in generating concise summaries of video content [[Pramanick_2023_ICCV]], which analyzes visual and auditory elements to extract key features for context-aware summaries. They also contribute to the field of video editing, as covered in existing literature [[wu2023visual]] and advertisement editing [[tang2022multi]].

#### V-A2 Interactive and User-Centric Systems

-
•

Virtual Education, Accessibility, and Sign Language: Vid-LLMs serve as virtual tutors in education, analyzing instructional videos for interactive learning environments [[gan2023large]]. They also facilitate sign language translation into spoken language or text [[liu2023survey, de2023machine]], improving accessibility for the deaf and hard of hearing.

-
•

Interactive Gaming and Virtual Environments: In the gaming industry, Vid-LLMs play a crucial role in creating dynamic dialogues and storylines, as well as aiding in generating procedural content, such as quests and in-game texts [[mishra2023generating, koomen2023text]]. They also power customer service chatbots [[soni2023large, medeiros2023analysis]]. Additionally, in AR/VR/XR, Vid-LLMs contribute to the generation of dynamic narrative content, enhancing user immersion [[gokce2023role, jung2023xr, yu2024promptfix, huang2023egocentric]].

-
•

State-Aware Human-Computer Interaction and Robot Planning: In the field of human-computer interaction, Vid-LLMs analyze user videos to discern context and provide customized assistance, as highlighted in Bi et al. [[bi2023misar]]. Interaction forms also involve video content understanding like captioning videos [[wang2023caption, hu2022promptcap, hua2024finematch]]. Concurrently, in autonomous robot navigation, the SayPlan method [[rana2023sayplan]] integrates LLMs with 3D scene graphs to enable robots to interpret and navigate complex spaces in large buildings.

#### V-A3 Healthcare and Security Applications

-
•

Healthcare Innovations: In the healthcare sector, Vid-LLMs play a crucial role in processing and interpreting medical literature, assisting in diagnostic and educational processes [[eysenbach2023role, liu2022beat, liu2022disco, liu2024emage]], and providing decision support for healthcare professionals. They are used in patient interaction tools, such as chatbots for symptom assessment and addressing health-related queries, thus improving patient care and access to information [[li2023llava]].

-
•

Security, Surveillance, and Cybersecurity: Vid-LLMs are crucial in security and protection, analyzing communications for potential threats [[al2023chatgpt, mouratidis2023modelling]] and detecting anomalous patterns in data [[lee2023lanobert, almodovar2023logfit]]. In surveillance video analysis, they identify suspicious behaviors, helping law enforcement [[de2023socratic]]. Their role in cybersecurity includes identifying phishing attempts and contributing to forensic analysis by summarizing case-related texts [[tang2023graphgpt]]. They may also improve video crowd counting [[cao2025efficientmaskedautoencodervideo]] for security applications.

-
•

Autonomous Vehicles: In autonomous vehicles, Vid-LLMs can process language inputs for interaction [[cui2024drive]], assist in understanding road signs and instructions [[li2023otter, lai2023lisa]], and improve user interfaces for vehicle control systems [[cui2024drive]], enhancing safety and user experience.

#### V-A4 Other Applications

Vid-LLMs offer valuable applications beyond those previously discussed. In video generation research [[zhou2024survey, lin2023videodirectorgpt, kondratyuk2023videopoet]], Vid-LLMs can evaluate model performance, refine text prompts, and provide reasoning capabilities that better reflect human intentions. Additionally, Vid-LLMs show promise in resource-constrained environments through edge computing applications [[jin2024efficient, lu2024b, hu2023edge]] and can enhance privacy-preserving distributed systems through federated learning frameworks [[yao2024federated, bastola2024fedmil, wang2023fedvmr]].

### V-B Future Directions

Despite enhancing multiple downstream tasks, Vid-LLMs face several challenges in real-world video understanding:

#### V-B1 More Fine-grained Video Understanding

Fine-grained understanding remains challenging due to limited datasets, insufficient research, and high computational demands. The frame-by-frame analysis increases computational load while capturing spatiotemporal information. Understanding deeper semantics (emotions, scene dynamics) is harder, though text-video alignment through LLMs offers promise [[tang2024vidcomposition]].

#### V-B2 Long-form Video Understanding

Long videos’ extended duration complicates the analysis, especially in understanding events over time. Thus, identifying key events and maintaining attention in long videos is difficult [[zhang2024longva, weng2024longvlm, Wang2024LifelongMemory]]. Effective mechanisms are needed to detect and highlight important parts, particularly in content-rich or complex plot videos.

#### V-B3 Multimodal Video Understanding

Multimodal video understanding requires integrating different types of data, such as visual, audio, and text, for a better understanding of videos [[tang2024avicuna, mohammadkhani2025survey, zhangmultimedia]]. Aligning these data, especially in terms of spatial and temporal synchronization, is particularly critical. This area lacks relevant research and suffers from a scarcity of datasets. The field lacks research and datasets, with challenges in ensuring high-quality data annotation.

#### V-B4 Hallucination in Video LLMs

Hallucination occurs when models generate responses disconnected from source material [[zhang2024eventhallusion]], caused by insufficient feature extraction, influence of video context, domain gap between vision and language, and inherent LLM hallucinations. Solutions include post-training strategies [[zhang2024llavahounddpo]], enhanced spatiotemporal context understanding, and visual-linguistic latent collaboration.

#### V-B5 Industrial Deployment and Scalability

Effective deployment strategies [[weng2024longvlm, lee2024video, tang2024enhancing, tan2024koala, shang2024interpolating, lu2024b]] include model compression, token merging, domain-specific fine-tuning, modular architectures, efficient caching, and standardized integration frameworks, balancing efficiency with performance for industrial systems.

### V-C Ethical Implications

The ethical implications of Vid-LLMs center on privacy, data security, and potential misuse.
These models perform tasks like video engagement analysis, transcription, summarization, and captioning, requiring access to sensitive content.
This raises privacy risks, as video data may contain private or confidential information that could be exposed without proper consent.
Also, Vid-LLMs can be misused for surveillance or generating misleading content.
Bias is another concern, especially if training data lacks diversity.
Addressing these issues requires robust data governance, consent mechanisms, and ethical deployment to prioritize privacy and fairness.

## VI Conclusion

This survey has examined the integration of LLMs in video understanding, which has enabled more sophisticated and versatile processing capabilities beyond traditional methods.
We categorized current approaches into three main types: Video Analyzer $\times$ LLM, Video Embedder $\times$ LLM, and (Analyzer + Embedder) $\times$ LLM, with sub-classifications based on LLM functional roles: Summarizer, Manager, Text Decoder, Regressor, and Hidden Layer.
Vid-LLMs demonstrate capabilities in multi-granularity reasoning from abstract to spatiotemporal analysis, showing potential across video summarization, captioning, question answering, and other applications. Despite progress, limitations remain in evaluation metrics, long-form video handling, and visual-textual modality alignment.
Future research will address these challenges through more efficient training strategies, improved Vid-LLM scalability, innovative architectures for multimodal integration, enhanced long-form video understanding, and methods to mitigate hallucinations. Expanding datasets and benchmarks will be critical for advancing video understanding with LLMs.

Generated on Tue Nov 25 03:39:48 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)