[# Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis Yafei Hu1111Equal contribution. {yafeih, quantinx, vidhij}@andrew.cmu.edu Quanting Xie111footnotemark: 1 Vidhi Jain111footnotemark: 1 Jonathan Francis1,2 Jay Patrikar1 Nikhil Keetha1 Seungchan Kim1 Yaqi Xie1 Tianyi Zhang1 Hao-Shu Fang3 Shibo Zhao1 Shayegan Omidshafiei4 Dong-Ki Kim4 ; Ali-akbar Agha-mohammadi4 Katia Sycara1 Matthew Johnson-Roberson1 Dhruv Batra5,6 Xiaolong Wang7 Sebastian Scherer1 Chen Wang8 Zsolt Kira5 Fei Xia9222Equal advising. xiafei@google.com, ybisk@cs.cmu.edu Yonatan Bisk1,622footnotemark: 2 1CMU 2Bosch Center for AI 3MIT 4Field AI 5Georgia Tech 6FAIR at Meta 7UC San Diego 8SAIR Lab 9Google DeepMind ###### Abstract Building general-purpose robots that operate seamlessly in any environment, with any object, and utilizing various skills to complete diverse tasks has been a long-standing goal in Artificial Intelligence. However, as a community, we have been constraining most robotic systems by designing them for specific tasks, training them on specific datasets, and deploying them within specific environments. These systems require extensively-labeled data and task-specific models. When deployed in real-world scenarios, such systems face several generalization issues and struggle to remain robust to distribution shifts. Motivated by the impressive open-set performance and content generation capabilities of web-scale, large-capacity pre-trained models (i.e., foundation models) in research fields such as Natural Language Processing (NLP) and Computer Vision (CV), we devote this survey to exploring (i) how these existing foundation models from NLP and CV can be applied to the field of general-purpose robotics, and also exploring (ii) what a robotics-specific foundation model would look like. We begin by providing a generalized formulation of how foundation models are used in robotics, and the fundamental barriers to making generalist robots universally applicable. Next, we establish a taxonomy to discuss current work exploring ways to leverage existing foundation models for robotics and develop ones catered to robotics. Finally, we discuss key challenges and promising future directions in using foundation models for enabling general-purpose robotic systems. We encourage readers to view our living GitHub repository 555The current version of this paper is v2.1-2024.09 (In the format of ‘[major].[minor]-YYYY.MM’). of resources, including papers reviewed in this survey, as well as related projects and repositories for developing foundation models for robotics: https://robotics-fm-survey.github.io/](https://robotics-fm-survey.github.io/).

*Figure 1: In this paper, we present a survey toward building general-purpose robots via foundation models*. We mainly categorize the foundation models into vision and language models used in robotics, and robotic foundation models. We also introduce how these models could mitigate the challenges of classical robotic challenges, and projections of the potential future research directions. 444Some images in this paper are screenshots from the papers we surveyed, icon images from Microsoft PowerPoint and MacOS Keynote, Google Images results, or are images we generated with OpenAI GPT-4.*

## 1 Overview

### 1.1 Introduction

We face many challenges in developing general-purpose robotic systems that can operate in and adapt to different environments and perform multiple tasks. Previous robotic perception systems that leverage conventional deep learning methods usually require a large amount of labeled data to train the supervised learning models [[[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3)]]; meanwhile, the crowdsourced labeling processes for building these large datasets remain rather expensive. Moreover, due to the limited generalization ability of classical supervised learning approaches, the trained models usually require carefully designed domain adaptation techniques to deploy these models to specific scenes or tasks [[[4](#bib.bib4), [5](#bib.bib5)]], which often require further steps of data-collection and labeling
Similarly, classical robot planning and control methods often require carefully modeling the world, the ego-agent’s dynamics, and other agents’ behavior [[[6](#bib.bib6), [7](#bib.bib7), [8](#bib.bib8)]]. These models are built for each individual environment or task and often need to be rebuilt as changes occur, exposing their limited transferability [[[8](#bib.bib8)]];
in fact, building an effective model can be either too expensive or intractable.
Although deep (reinforcement) learning-based motion planning [[[9](#bib.bib9), [10](#bib.bib10)]] and control methods [[[11](#bib.bib11), [12](#bib.bib12), [13](#bib.bib13), [14](#bib.bib14)]]
could help mitigate these problems, they also still suffer from distribution shifts and reductions in generalizability [[[15](#bib.bib15), [16](#bib.bib16)]].

![Figure](x2.png)

*Figure 2: Overall structure of this survey paper. The left side shows the time dimension governing the emergence of foundation models, and the right side shows the logical induction, for applying and developing foundation models for robotics. Sections [2](#S2) and [3](#S3) answer the “what” questions: what are foundation models for robotics, what are the challenges in building general-purpose robots. Section [4](#S4) and [5](#S5) deal with the “why” and “how” questions: why do we need foundation models in general-purpose robotics, how can foundation models be applied in robotics, and how do foundation models specially designed for robotics work. Section [6](#S6) closes with existing work, and posits what future models might look like.*

Concurrent to the challenges faced in building generalizable robotic systems, we notice significant advances in the fields of Natural Language Processing (NLP) and Computer Vision (CV)—with the introduction of Large Language Models (LLMs) [[[17](#bib.bib17)]] for NLP, diffusion models to generate high-fidelity images and videos [[[18](#bib.bib18), [19](#bib.bib19)]], and zero-shot/few-shot generalization of CV tasks with large-capacity Vision Foundation Models (VFM) [[[20](#bib.bib20), [21](#bib.bib21), [22](#bib.bib22)]], Vision Language Models (VLMs) and Large Multi-Modal Models (LMM) [[[23](#bib.bib23), [24](#bib.bib24)]]
, coined foundation models [[[25](#bib.bib25)]].

These large-capacity vision and language models have also been applied in the field of robotics [[[26](#bib.bib26), [27](#bib.bib27), [28](#bib.bib28)]], with the potential for endowing robotic systems with open-world perception, task planning, and even motion control capabilities. Beyond just applying existing vision and/or language foundation models in robotics, we also see considerable potential for the development of more robotics-specific models, e.g., the action model for manipulation [[[29](#bib.bib29), [30](#bib.bib30)]], motion planning model for navigation [[[31](#bib.bib31)]], and more generalist models which can conduct action generation though vision and language inputs hence Vision Language Action models (VLA) [[[32](#bib.bib32), [33](#bib.bib33), [30](#bib.bib30), [34](#bib.bib34)]]. These robotics foundation models show potential generalization ability across different tasks and even embodiments.

The overall structure of this paper is formulated as in Figure [2](#S1.F2). In Section [2](#S2), we provide a brief introduction to robotics research before the foundation model era and discuss the basics of foundation models. In Section [3](#S3), we enumerate challenges in robotic research and discuss how foundation models might mitigate these challenges. In Section [4](#S4), we summarize the current research status quo of foundation models in robotics. Finally, in Section [6](#S6) we offer potential research directions which are likely to have a high impact on this research intersection.

Although we see promising applications of vision and language foundation models to robotic tasks, and the development of novel robotics foundation models, many challenges in robotics out of reach. From a practical deployment perspective, models are often not reproducible, lack multi-embodiment generalization, or fail to accurately capture what is feasible (or admissible) in the environment. Furthermore,
most publications leverage transformer-based architectures and focus on semantic perception of objects and scenes, task-level planning, or control [[[30](#bib.bib30)]]; other components of a robotic system, which could benefit from cross-domain generalization capabilities, are under-explored—e.g., foundation models for world dynamics or foundation models that can perform symbolic reasoning.
Finally, we would like to highlight the need for more large-scale real-world data as well as high-fidelity simulators that feature diverse robotics tasks.

In this paper, we investigate how foundation models serve as a potential solution for general-purpose robotics, aiming to understand how foundation models could help mitigate the core challenges general-purpose robotics face. We use the term “foundation models for robotics” to include two distinct aspects: (1) the application of existing (mainly) vision and language models to robotics, largely through zero-shot and in-context learning; and (2) developing and leveraging robotics foundation models especially for robotic tasks by using robot-generated data. We summarize the methodologies of foundation models for robotics papers and conduct a meta-analysis of the experimental results of the papers we surveyed. A summary of the major components of this paper in Figure [4](#footnote4).

### 1.2 Related Survey Papers

Recently, with the popularity of foundation models, there are various survey papers on vision and language foundation models that are worth mentioning [[[35](#bib.bib35), [36](#bib.bib36), [37](#bib.bib37), [38](#bib.bib38)]]. These survey papers cover foundation models, including Vision Foundation Models (VFMs) [[[39](#bib.bib39), [40](#bib.bib40)]], Large Language Models (LLMs) [[[37](#bib.bib37)]], Vision-Language Models (VLMs) [[[41](#bib.bib41), [42](#bib.bib42)]], and Visual Content Generation Models (VGMs) [[[35](#bib.bib35)]]. There are a few existing survey papers combining foundation models and robotics, perhaps the most relevant survey papers are [[[43](#bib.bib43), [38](#bib.bib38), [44](#bib.bib44), [45](#bib.bib45), [46](#bib.bib46), [47](#bib.bib47)]], however, there are still significant differences between those papers and ours, for instance: Yang et al. [[[38](#bib.bib38)]] and Wang et al. [[[44](#bib.bib44)]] focus on broadly-defined autonomous agents, instead of physical robots; Lin et al. [[[45](#bib.bib45)]] focus on LLMs for navigation; the connection between foundation models and robotics is limited in [[[46](#bib.bib46)]]. Compared with [[[47](#bib.bib47)]], we propose a breakdown of current research methodologies, providing a detailed analysis of the experiments, and also focus on how foundation models could resolve the typical challenges for general-purpose robotics. Concurrently, Firoozi et al. [[[43](#bib.bib43)]] conducted a survey regarding foundation models in robotics. Both their and our works shed lights on the opportunities and challenges of using foundation models in robotics, and identifies key pitfalls to scale them further. Their work focuses on how foundation models contribute to improving robot capabilities, and challenges going forward. Comparatively, our survey attempts to taxonomize robotic capabilities together with foundation models to advance those capabilities. We also propose a dichotomy between robotic(-first) foundation models and other foundation models used in robotics, and provide a thorough meta-analysis of the papers we survey.

In this paper, we provide a survey that includes existing unimodal and multimodal foundation models applied in robotics, as well as all forms of robotics foundation models in various robotics tasks as we know of. We also narrowed the scope of papers being surveyed to only those with experiments on real physical robotics, in high-fidelity simulation environments, or using real robotics datasets. We believe that, by doing so, it could help us understand the power of foundation models in the real world robotic applications.

## 2 Problem Formulation and Preliminaries

In this section, we summarize foundation models for robotics as a unified problem formulation. Let us define the foundation model relevant for robotics as a function $f$, which takes sensory inputs $\mathbf{x_{t}}$, context $\mathbf{c}$, and outputs $\mathbf{y_{t}}$ to fulfill the downstream robotic tasks.

$$f(\mathbf{x_{tk},c_{k}})\rightarrow\mathbf{y_{tk}}\quad\quad\quad\forall k\in N% \quad\forall t\in T$$ \tag{1}

Here $\mathbf{x}$ denotes sensory inputs like observations through visual cameras, textual scene descriptions, scene graphs, object poses and detections, audio signals through a microphone, haptic sensors, and so on. For multi-task policies, we need contextual information $\mathbf{c}$ that denotes task specification [[[30](#bib.bib30)]] or details on the embodiment [[[48](#bib.bib48)]].

The state $\mathbf{x}$ and context $\mathbf{c}$ are often defined in terms of visual observations and language instructions respectively. However, both $\mathbf{x}$ and $\mathbf{c}$ could be images, videos, language prompts, or a combination of multiple modalities. This also depends on how we define the scope of the environment.

The action $\mathbf{y}$ can be defined in terms of target objects’ pose in the map, task plans, next state, reward functions, and control outputs like target end-effector pose. If $\mathbf{y}$ is a task plan, we often assume that the low-level components of the plan can be executed perfectly. However, there are open questions on how to enable suitable and timely recovery in the robot policies when the low-level plan is not executed completely.

The data to train or prompt the model $f$ have several open questions. For training, how should expert demonstrations, offline data, or online experiences with apt reward functions be collected? For prompting, what instructions and examples enable desirable in-context learning? How should the models be trained and fine-tuned? In the next following sections, we aim to answer these questions by summarizing and analyzing existing research works.

## 3 Challenges on General-purpose Robots

In this section, we summarize 5 core challenges that general-purpose robotic system face, each detailed in the following subsections. Whereas similar challenges have been discussed in prior literature (Section [1.2](#S1.SS2)), this section mainly focuses on the challenges that may potentially be solved by appropriately leveraging foundation models, given the evidence from current research results. We also depict the taxonomy in the section for easier review in Figure [3](#S3.F3).

*Figure 3: Taxonomy of the challenges in robotics that could be resolved by foundation models. We list five major challenges in the second level and some, but not all, of the keywords for each of these challenges.*

### 3.1 Generalization

Robotics systems often struggle with accurate perception and understanding of their environment. Limitations in computer vision, object recognition, and semantic understanding made it difficult for robots to effectively interact with their surroundings. Traditional robotics systems often relied on analytic hand-crafted algorithms, making it challenging to adapt to new or unseen situations. They also lacked the ability to generalize their training from one task to another, further limiting their usefulness in real-world applications. This generalization ability is also reflected in terms of the generalization of planning and control in different tasks, environments, and robot morphologies. For example, specific hyperparameters for, e.g., classical motion planners and controllers need to be tuned for specific environments [[[49](#bib.bib49), [50](#bib.bib50), [51](#bib.bib51)]]; RL-based controllers are difficult to transfer across different tasks and environments [[[52](#bib.bib52), [53](#bib.bib53), [54](#bib.bib54)]]. In addition, due to differences in robotic hardware, it is also challenging to transfer models across different robot morphologies [[[55](#bib.bib55), [56](#bib.bib56)]]. By applying foundation models in robotics, the generalization problem is partially resolved, which will be discussed in the next Section [4](#S4). Further challenges, such as generalization across different robotic morphologies, remain demanding.

### 3.2 Data Scarcity

Data has always been the cornerstone of learning-based robotics methods. The need for large-scale, high-quality data is essential to develop reliable robotic models. Several endeavors have been attempted to collect large-scale datasets in the real world, including autonomous driving [[[1](#bib.bib1), [2](#bib.bib2), [57](#bib.bib57)]], robot manipulation trajectories [[[58](#bib.bib58), [59](#bib.bib59), [60](#bib.bib60)]], etc. Collecting robot data from human demonstration is expensive [[[29](#bib.bib29)]]. The diverse range of tasks and environments where robots are used even complicates the process of collecting adequate and extensive data in the real world. Moreover, gathering data in real-world settings can be problematic due to safety concerns [[[49](#bib.bib49)]]. To overcome these challenges, many works [[[61](#bib.bib61), [62](#bib.bib62), [63](#bib.bib63), [64](#bib.bib64), [65](#bib.bib65), [66](#bib.bib66)]] attempt generating synthetic data in simulated environments. These simulations offer realistic virtual worlds where robots can learn and apply their skills to real-life scenarios. Simulations also allow for domain randomization, as well as the potential to update the parameters of the simulator to better match the real world physics [[[49](#bib.bib49)]], helping robots to develop versatile policies. However, these simulated environments still have their limits, particularly in the diversity of objects, making it difficult to apply the learned skills directly to real-world situations. Collecting real-world robotic data with a scale comparable to the internet-scale image/text data used to train foundation models is especially challenging. One promising approach is collaborative data collection across different laboratories and robot types [[[67](#bib.bib67), [68](#bib.bib68), [69](#bib.bib69)]], as shown in Figure [4](#S3.F4).
However, an in-depth analysis of the Open-X Embodiment Dataset reveals certain limitations regarding data type availability.
First, in Figure 4a, the robot morphology utilized for data collection is restrictive; out of the 73 datasets, 55 are dedicated to single-arm manipulation tasks. Only one dataset pertains to quadruped locomotion, and a single dataset addresses bi-manual tasks. Second, in Figure 4b, the predominant scene type for these manipulation tasks is tabletop setups, often employing toy kitchen objects. Such objects come with inherent assumptions including rigidity and negligible weight, which may not accurately represent a wider range of real-world scenarios. Third, in Figure 4c, our examination of data collection methods indicates a predominance of human expert involvement, predominantly through virtual reality (VR) or haptic devices. This reliance on human expertise highlights the challenges in acquiring high-quality data and suggests that significant human supervision is required. For instance, the RT-1 Robot Action dataset necessitated a collection period of 17 months, underscoring the extensive effort and time commitment needed for data accumulation with human involvement.

![Figure](extracted/5892243/figs/dataset_figure.png)

*Figure 4: Comprehensive visualizations of the Open-X Embodiment and Droid Dataset encompassing robot morphologies, environment types, and data collection methods.*

### 3.3 Requirements of Models and Primitives

Classical planning and control methods usually require carefully engineered models of the environment and the robot. Optimal control methods require good dynamics models (i.e., world transition models) [[[8](#bib.bib8), [70](#bib.bib70)]]; motion planning requires a map of the environment [[[71](#bib.bib71)]], the states of the objects robots interact with [[[72](#bib.bib72)]], or a set of pre-defined motion primitives [[[73](#bib.bib73)]]; task planning requires pre-computed object classes and pre-defined rules [[[74](#bib.bib74), [75](#bib.bib75)]], etc. Previous learning-based methods (e.g., imitation and reinforcement learning) train policies in an end-to-end manner that directly gets control outputs from sensory inputs [[[58](#bib.bib58)]], avoiding building and using models. These methods partially solve this problem of relying on explicit models, but they often struggle to generalize across different environments and tasks. This raises two problems: (1) How can we learn model-agnostic policies that can generalize well? Or, alternatively, (2) How can we learn good world models so that we can apply classical model-based approaches? We see some recent works that aim to resolve these problems using foundation models (especially in a model-free manner), which will be systematically discussed in Section [4](#S4).
However, the call for world models for robotics remains an intriguing frontier, which will be discussed in Section [6](#S6).

### 3.4 Task Specifications

Understanding the task specification and grounding it in the robot’s current understanding of the world is a critical challenge to obtaining generalist agents. Often, these task specifications are provided by users with limited understanding of the limitations on the robot’s cognitive and physical capabilities. This not only raises questions about what the best practices are for providing these task specifications, but also about the naturalness and ease of crafting these specifications. Understanding and resolving ambiguity in task specifications, conditioned on the robot’s understanding of its own capabilities, is also challenging. Foundation models are a promising solution for this challenge: task specification can be formulated as language prompts [[[29](#bib.bib29), [30](#bib.bib30), [26](#bib.bib26), [76](#bib.bib76)]], goal images [[[77](#bib.bib77)]], rewards for policy learning [[[28](#bib.bib28), [78](#bib.bib78)]], etc.

### 3.5 Uncertainty and Safety

One of the critical challenges in deploying robots in the real world comes from dealing with the uncertainty inherent in the environments and task specifications. Uncertainty, based on its source, can be characterized either as epistemic (uncertainty caused by a lack of knowledge) or aleatoric (noise inherent in the environment). Epistemic uncertainty often manifests as out-of-distribution errors when the robot encounters unfamiliar situations in the test distribution. While the adoption of learning-based techniques for decision-making in high-risk safety-critical fields has prompted efforts in uncertainty quantification (UQ) and mitigation [[[79](#bib.bib79)]], out-of-distribution detection, explainability, interpretability, and vulnerability to adversarial attacks remain open challenges.
Uncertainty quantification can be prohibitively expensive and may lead to sub-optimal downstream task performance [[[80](#bib.bib80)]]. Given the large-scale over-parameterized nature of foundation models, providing UQ methods that preserve the training recipes with minimal changes to the underlying architecture are critical in achieving the scalability without sacrificing the generalizability of these models. Designing robots that can provide reliable confidence estimates on their actions and in turn intelligently ask for clarification feedback remains an unsolved challenge [[[81](#bib.bib81)]]. Conformal predictions [[[82](#bib.bib82)]] provide a distribution-free way of generating statistically rigorous uncertainty sets for any black-box model and have been demonstrated in VLN tasks for robotics [[[83](#bib.bib83)]].

In its traditional setting, provable safety in robotics [[[84](#bib.bib84), [85](#bib.bib85)]] refers to a set of control techniques that provide theoretical guarantees on safety bounds for robots. Control Barrier Functions [[[86](#bib.bib86)]], reachability analysis [[[87](#bib.bib87), [88](#bib.bib88)]] and runtime monitoring via logic specifications [[[89](#bib.bib89)]] are well-known techniques in ensuring robot safety with bounded disturbances. Recent works have explored the use of these techniques to ensure safety of the robot [[[90](#bib.bib90)]]. While these contributions have led to improved safety, solutions often result in sub-optimal behavior and impede robot learning in the wild [[[91](#bib.bib91)]]. Thus, despite recent advances, endowing robots with the ability to learn from experience to fine-tune their policies while remaining safe in novel environments still remains an open problem.

## 4 Review of Current Research Methodologies

In this section, we summarize the current research methodologies of foundation models for robotics.

*Figure 5: Conceptual Framework of Foundation Models in Robotics: The figure illustrates a structured taxonomy of foundational models, categorized into two primary segments: the application of existing foundation models (vision and language models) to robotics, and the development of robotic-specific foundation models. This includes distinctions between vision and language models used as perception tools, in planning, and in action, as well as the differentiation between single-purpose and general-purpose robot foundation models.*

In Section [4.1](#S4.SS1), we mainly discuss foundation models for robotics in two categories: Foundation Models used in Robotics and Robotics Foundation Models (RFMs). For Foundation Models used in Robotics, we mainly highlight applications of vision and language foundation models used in a zero-shot manner, meaning no additional fine-tuning or training is conducted. In Section [4.2](#S4.SS2), however, we mainly focus on Robotics Foundation Models, wherein these approaches may warm-start models with vision-language pre-trained initialization and/or directly train the models on robotics datasets. Figure [5](#S4.F5) shows the detailed taxonomy of this section.

We review the methods presented in these papers following the convention of a typical robotic system which consists of perception, planning, and control modules. Here, we combine motion planning and control into one piece—action generation and treat motion planning modules as higher-level and control as lower-level action generation.
It is important to notice that although most of the works use foundation models in different functionality modules of the robotic systems, we will classify these papers into categories based on the module to which the paper contributes the most. There are, however, certain applications of the vision and language foundation models that go across these robotics modules, e.g., grounding of these models in robotics, and generating data from LLMs and VLMs. Given the autoregressive nature of current LLMs, they often grapple with extended horizon tasks. Thus, we also delve into advanced prompting methods proposed in the literature to ameliorate this limitation and enhance planning power. We list these applications in sections [4.1.4](#S4.SS1.SSS4), [4.1.5](#S4.SS1.SSS5) and [4.1.6](#S4.SS1.SSS6), as a different perspective to analyze these works.

We find that works in Section [4.1](#S4.SS1) typically follow a modular strategy, in applying vision and language foundation models to serve a single robot functionality, e.g., applying VLMs as open-set robot perception modules which are then “plugged in” to work alongside motion planners and controllers [[[27](#bib.bib27)]], downstream. Since such foundation models are applied in a zero-shot manner, there are no gradients flowing between the module in which the foundation models are applied and the other modules in the robotic system. Conversely, works in Section [4.2](#S4.SS2) mostly follow an end-to-end differentiability paradigm, which blurs the boundary of the typical robotics modules in methods (described in Section [4.1](#S4.SS1); e.g., perception and control [[[29](#bib.bib29), [92](#bib.bib92)]]), with some robotics foundation models even providing a unified model to perform different robot functions [[[32](#bib.bib32), [33](#bib.bib33)]].

![Figure](x3.png)

*Figure 6: Here we plot some representative works of foundation models used in robotics, and robotic foundation models. The horizontal axis represents the complexity of input data; the vertical axis represents the complexity of output action space. The complexity of the input data is brought by the modality of the data, e.g. image data is more complex than the text data. The complexity of output action space is mainly determined by the output dimension of the foundation model in the robotic tasks.*

### 4.1 Foundation Models used in Robotics

In this section, we focus on zero-shot applications of vision and language foundation models in robotic tasks, which include mainly zero-shot deployment of VLMs used in robotic perception, in context learning of LLMs for task-level and motion-level planning, as well as action-generation.

#### 4.1.1 VFMs and VLMs in Robot Perception

Recently, the grounding of vision and language foundation models with geometric and object-centric representations of the world has enabled tremendous progress in context understanding, which is a vital requirement for robots to interact with the real world. We will thoroughly examine the application of VFMs and VLMs in robotic perception from various perspectives.

##### VFMs, VLMs for Object and Scene Representations

The most straightforward application of VLMs in robotics is to leverage their ability to perform open-set object recognition and scene understanding in robotics-oriented downstream tasks, including semantic mapping and navigation [[[27](#bib.bib27), [93](#bib.bib93), [94](#bib.bib94), [95](#bib.bib95), [96](#bib.bib96)]], manipulation [[[97](#bib.bib97), [98](#bib.bib98), [99](#bib.bib99), [100](#bib.bib100)]], etc. The methods proposed by these works share a common attribute: they attempt to extract semantic information (from the VLM) and spatial information (from other modules or sensor modalities) from objects and scenes that the robots interact with. This information is then used as representations in semantic maps of scenes or representations of objects.

For semantic mapping and/or navigation, NLMap [[[27](#bib.bib27)]] is an open-set and queryable scene representation to ground task plans from LLMs in surrounding scenes.
The robot first explores the environment using frontier-based exploration to simultaneously build a map and extract class-agnostic regions of interest, which are then encoded by VLMs and aggregated to the map.
Natural language instructions are then parsed by an LLM to search for the availability and locations of these objects in the scene representation map.
ConceptFusion [[[95](#bib.bib95)]] builds open-set multimodal 3D maps from RGB-D inputs and features from foundation models, allowing queries from different modalities such as image, audio, text, and clicking interactions.
It is shown that ConceptFusion can be applied for real-world robotics tasks, such as tabletop manipulation of novel objects and semantic navigation of autonomous vehicles.
Similarly, CLIP-Fields [[[94](#bib.bib94)]] encodes RGB-D images of the scene to language-queryable latent representations as elements in a memory structure, that the robot policy can flexibly retrieve.
VLMap [[[93](#bib.bib93)]] uses LSeg [[[101](#bib.bib101)]] to extract per-pixel representations to then fuse with depth information, in order to create a 3D map. This semantic 3D map is then down-projected to get a 2D map with the per-pixel embedding; these embeddings can then be matched with the language embedding from LSeg to obtain the per-pixel semantic mask for the 2D map. As for applying VLMs in topological graphs for visual navigation, LM-Nav [[[96](#bib.bib96)]] is a good example:
it uses an LLM to extract landmarks used in navigation from natural language instructions.
These landmark descriptions, as well as image observations, are then grounded in a pre-built graph via a VLM. Then, a planning module is used to navigate the robot to the specified landmarks.

Most of the previous works discussed above utilize only 2D representation of the objects and environment. To enrich the representation of foundation models in 3D space, F3RM [[[98](#bib.bib98)]], GNFactor [[[99](#bib.bib99)]] and GeFF [[[102](#bib.bib102)]] distill 2D foundation model features into 3D space, by combining with (generalizable) NeRF. In addition, GNFactor [[[99](#bib.bib99)]] also apply these distilled features in policy learning. Act3D [[[103](#bib.bib103)]] takes a similar approach but build 3D feature field via sensed depth.

##### VLMs for State Estimation and Localization

Beyond context understanding, a few approaches explore the use of open-vocabulary properties of VLMs for state estimation [[[104](#bib.bib104), [105](#bib.bib105), [105](#bib.bib105), [106](#bib.bib106), [107](#bib.bib107)]].
Two such approaches, LEXIS [[[104](#bib.bib104)]] and FM-Loc [[[105](#bib.bib105)]], explore the use of CLIP [[[23](#bib.bib23)]] features to perform indoor localization and mapping.
In particular, FM-Loc [[[105](#bib.bib105)]] leverages the vision-language grounding offered by CLIP and GPT-3 to detect objects and room labels of a query image, then uses that semantic information to match it with reference images.
Similarly, LEXIS [[[104](#bib.bib104)]] builds a real-time topological SLAM graph where CLIP features are associated with graph nodes, enabling room-level scene recognition.
Although these approaches display the potential of vision-language features for indoor place recognition, they do not explore the broad applicability of foundation model features.
In this context, AnyLoc [[[106](#bib.bib106)]] explored the properties of dense foundation model features and combined them with unsupervised feature-aggregation techniques to achieve state-of-the-art place recognition, by large margin—anywhere, anytime, and under any view—showcasing broad applicability of self-supervised foundation model features for SLAM.
Extending this further, FoundLoc [[[107](#bib.bib107)]] coupled AnyLoc with a VIO pipeline to perform visual GNSS-denied localization.
For the first time, this work showcased that VLMs can be deployed on resource-constrained unmanned aerial vehicles (UAVs) & embedded Jetson hardware for state estimation in the wild.

##### VLMs for Interactive Perception

Several works consider the notion of enabling robots to leverage the process of interactive perception [[[108](#bib.bib108)]], for extrapolating implicit knowledge about object properties in order to obtain performance improvements on downstream interactive robot learning tasks [[[109](#bib.bib109), [110](#bib.bib110), [111](#bib.bib111), [112](#bib.bib112), [113](#bib.bib113), [114](#bib.bib114), [56](#bib.bib56), [55](#bib.bib55), [115](#bib.bib115), [116](#bib.bib116)]]. This process of interactive perception is often modeled after the way in which human infants first learn about the physical world—i.e., through interaction, and by learning representations of object concepts, such as weight and hardness, from the sensory information (haptic, auditory, visual) that is generated from those physical exploratory actions (e.g., grasping, lifting, dropping, pushing) on objects with diverse properties. In particular, MOSAIC [[[115](#bib.bib115)]] leverages LMMs to expedite the acquisition of unified multi-sensory object property representations; the authors show competitive performances of their framework in category recognition and ambiguous target-object fetch tasks, despite the presence of distractor objects, under zero-shot transfer conditions. UncOS [[[116](#bib.bib116)]] queries LVMs to generate a distribution of object segmentation hypotheses. They convert the hypotheses into a distribution over world models and use that to select robot perturbation actions for embodied disambiguation of object segmentation.

#### 4.1.2 LLMs and VLMs in Task Planning

The planning community in Robotics has always harbored aspirations of a model capable of generalizing across diverse tasks and environments, with minimal demonstrations for robotic tasks. Given the demonstrated prowess of vision and language foundation models in intricate reasoning and contextual generalization, it is a natural progression for the robotics community to consider the application of foundation models to robotic planning problems. This section organizes works based on the granularity of planning, delineating between task-level and motion-level planning. We will mainly introduce task-level planning in this part and leave motion-planning to the next part, together with action generation (Section [4.1.3](#S4.SS1.SSS3)).

Task-level planning is to divide a complicated task into small actionable steps. In this case, we mainly talk about the agent planning on its own, in contrast to, e.g., using LLMs as a parser like Vision Language Navigation [[[117](#bib.bib117)]]. The agent needs to take intelligent sub-steps to reach the goal by interacting with the environment. SayCan [[[26](#bib.bib26)]] is a representative example of task-level planning: it uses LLMs to plan for a high-level task, e.g., “I spilled my drink, can you help?”. Then it gives concrete task plans like going to the counter, finding a sponge, and so on. Similarly, VLP [[[118](#bib.bib118)]] aims to improve the long-horizon planning approach with an additional text-to-video dynamics model. These task-level planning methods do not have to worry about the precise execution of those sub-tasks in the environment, since they have the luxury of utilizing a set of pre-defined / pre-trained skills, then using LLMs to simply find proper ways to compose skills to achieve desired goals. There are more papers in this category, for example: LM-ZSP [[[119](#bib.bib119)]] introduce this task-level granularity as actionable steps; Text2Motion [[[120](#bib.bib120)]] uses similar ideas and augments the success rate of language based manipulation task. Previous methods typically generate task plans in the form of text, Some works like ProgPrompt [[[121](#bib.bib121)]], Code as Policy [[[122](#bib.bib122)]], GenSim [[[123](#bib.bib123)]], etc. obtain task plans in the form of code generation using LLMs. [[[124](#bib.bib124)]] take a different approach by obtaining relational constraints from code writing LLMs. Using code as a high-level plan has the benefit of expressing functions or feedback loops that process perception outputs and parameterize control primitive APIs. In addition, it can describe the spatial position of an object accurately. This improved compositionality saves time in collecting more primitive skills. It also prescribes precise values (e.g., velocities) to ambiguous descriptions like ’faster’ and ’to the left’, depending on the context. Therefore, due to these benefits, code appears to be a more efficient and effective task-level planning language than natural language.
Other forms of planning techniques such as expressing the high-level plans in Planning Domain Definition Language (PDDL)[[[125](#bib.bib125)]] also showed significant improvement in LLMs planning power over long horizon tasks, more on this will be discussed in Section [4.1.6](#S4.SS1.SSS6).

In addition to using LLMs to directly generate plans, they are also used in searching and evaluating with external memory structures such as scene graph. SayPlan [[[126](#bib.bib126)]] employs 3D scene graph (3DSG) representations to manage the complexity of expansive environments. By exploiting hierarchical 3DSGs, LLMs can semantically search for relevant sub-graphs in multi-floor household environments, reducing the planning horizon and integrating classical path-planning to refine initial plans, iteratively. Reasoned Explorer [[[127](#bib.bib127)]] employs LLMs as evaluators to score each node in a 2D undirected graph. It uses this graph as a map to store both visited points and the frontiers’ LLM evaluations. These external memories and incremental map-building approaches break the context length limit of using LLMs to generate long plans, which scales LLM-based navigation agents to large environments. One thing to note is that, although task-level planning is agnostic to physical embodiment, it does require grounding to a specific robot’s physical form (or “morphology”) and environment when deployed; grounding techniques will be covered in Section [4.1.4](#S4.SS1.SSS4).

#### 4.1.3 LLMs and VLMs in Action Generation

Directly controlling a robot just by prompting off-the-shelf LLMs/VLMs can be challenging, perhaps even unachievable, without first fine-tuning these models with action data. Unlike high-level robot task planning, where LLMs are used for their ability to compose and combine different skills for task completion, individual actions, both high-level actions like waypoints, and low-level actions like joint angles are usually not semantically meaningful or compositional. The community is attempting to find a suitable interface to circumvent this problem. For motion planning in navigation tasks, ReasonedExplorer [[[127](#bib.bib127)]] and Not_Train_Dragon [[[128](#bib.bib128)]] propose such an interface: using LLMs as evaluators for the expanded frontiers, which are defined as potential waypoints for exploration (typically in a two-dimensional space); here, LLMs are tasked with scoring frontiers based on the similarity between the given observations and the intended goal. Similarly, VoxPoser [[[129](#bib.bib129)]] apply VLMs to obtain affordance function (called 3D value map in the original paper) used in motion planning.

Some papers investigate the use of LLMs to directly output lower-level actions. Prompt2Walk [[[130](#bib.bib130)]] uses LLMs to directly output joint angles through few-shot prompts, collected from the physical environment. It investigates whether LLMs can function as low-level controllers by learning in-context with environment feedback data (observation-action pairs). Saytap [[[28](#bib.bib28)]], introduces a novel concept of utilizing foot contact patterns as an action representation. In this model, the language model outputs a ‘0’ for no contact and a ‘1’ for contact with the floor, thereby enabling Large Language Models (LLMs) to generate zero-shot actionable commands for quadrupedal locomotion tasks such as jumping and jogging. However, the generalizability of these approaches to different robot morphologies remains in question, since they have only been tested on the quadruped platform. Instead, language to reward [[[131](#bib.bib131), [132](#bib.bib132), [133](#bib.bib133)]] in robotics [[[134](#bib.bib134), [78](#bib.bib78)]] is a more general approach than direct action generation through LLMs; these approaches involve using LLMs as generators to synthesize reward functions for reinforcement learning-based policies and thus are usually not confined by robotic embodiments [[[78](#bib.bib78)]]. The reward synthesizing approach with LLM can generate rewards which are hard for human to design, e.g., Eureka [[[78](#bib.bib78)]] demonstrates that it enables robots to learn dexterous pen-spinning task that were considered very hard using human reward design.

#### 4.1.4 Action Grounding

Traditionally, grounding refers to associating abstract and contextual meaning with sensory signals, such as identifying objects in images [[[135](#bib.bib135)]] or recognizing sounds in audio [[[136](#bib.bib136)]]. This form of ”direct grounding” in perceptual modalities has been extensively studied. However, in robotics, grounding extends to a crucial additional dimension: associating abstract concepts with executable actions. In other words, it involves translating the output of foundational models into meaningful changes in the environment. Thus, in this survey, we focus on action grounding.

Action grounding is just as vital as sensory grounding for robotics. For robots to interact with the physical world, foundational models require a bridge that maps their outputs to actionable behaviors, either explicitly or implicitly.

The initial approaches [[[26](#bib.bib26), [48](#bib.bib48), [130](#bib.bib130), [76](#bib.bib76)]] that tried to explicitly map foundation models’ outputs or representations to robot actions are fairly straightforward. The foundational models used are mainly large language models, and, like traditional robotics approaches, they adopt a modular concept that breaks down the robot’s preliminary skills. These skills are obtained either through pre-trained policies or pre-programmed scripts, and the entire skill library is described to the foundation models in natural language, allowing LLMs to output action plans using these skills. Recent progress in robotic grasping foundation models like AnyGrasp [[[137](#bib.bib137)]] and other open knowledge models like CLIP [[[23](#bib.bib23)]] and SAM [[[22](#bib.bib22)]] makes this approach more attractable. However, there are still two bottlenecks that it needs to solve: first, the limitations of natural language as a general interface, since the granularity that natural language can describe is limited. For example, if you were to use language to describe how to play Cat’s Cradle with a dexterous robot hand, language would likely struggle to accurately describe it. The second limitation is that, if we want the model to generalize to different tasks, more pre-trained or pre-programmed skills are needed, making it difficult to scale.

Therefore, researchers are exploring different ways to address these issues. Code as an interface alleviates some limitations of natural language granularity [[[122](#bib.bib122), [121](#bib.bib121)]], as it can directly describe an object’s x, y, z position for grasping. Other novel interfaces try to involve directly anchoring foundational models to output joint torques, circumventing intermediary interfaces such as text tokens. A notable example is Gato [[[32](#bib.bib32)]], which directly maps the model’s output to the Atari game action space and robotic arm joint space. Gato dynamically decides its output format—be it text, joint torques, button presses, or other tokens—based on the context of the task at hand. Another related development is RT-2 [[[30](#bib.bib30)]], which, despite specifying the end-effector space in textual form, is capable of directly generating executable commands for robotic manipulator operation. SayTap [[[28](#bib.bib28)]] has experimented with directly generating binary foot patterns to control quadruped locomotion tasks.

Building on this idea of grounding models to real-world action spaces, approaches such as CLIP-Fields [[[94](#bib.bib94)]], VLMap [[[93](#bib.bib93)]], and NLMaps [[[27](#bib.bib27)]] project CLIP visual and semantic label representations directly onto 3D point clouds. By aligning semantic understanding with spatial information, these methods create more interpretable 3D maps for robotic applications. Going beyond explicit 3D mapping, GLAM [[[138](#bib.bib138)]] uses reinforcement learning to ground language models through interactions with the environment, demonstrating that LLMs can function effectively as RL agents in textual environments. Moreover, AutoRT [[[139](#bib.bib139)]] acts as an orchestrator for task performance by a fleet of robots, leveraging vision-language models [[[140](#bib.bib140), [141](#bib.bib141)]] for scene description to prompt LLMs for task generation.

Additionally, researchers are trying different interfaces to map the output of the foundation model to actions. Voxposer [[[129](#bib.bib129)]] maps the output to a 3D value map and then uses classical optimization techniques to calculate robot actions. This approach adds more robustness to disturbances in the pipeline and allows the model to manipulate trajectories to add safety constraints and avoid obstacles in real time. In addition to value maps, using constraints is also a popular approach to achieve grounding in more diverse tasks. Geometric constraints are used to compute feasible trajectories that avoid obstacles and achieve goals, while contact constraints are used to plan forceful or contact-rich behaviors. ReKep uses keypoint relationship constraints to represent both geometric and contact constraints as Python functions that map a set of keypoints to a numerical cost. Each keypoint is task-specific and represents a semantically meaningful 3D point in the scene.

While several grounding-to-action techniques are discussed above, there is no definitive answer as to which approach offers the most optimal solution. Pretrained skill libraries provide high levels of dexterity and precision in task execution, but at the expense of task diversity. Conversely, map- or constraint-based grounding techniques offer greater task flexibility but have primarily been demonstrated on simpler 2D gripper pick-and-place tasks, with no tests on more dexterous, complex actions. If we view grounding to action as a spectrum, the ideal interface should balance both task diversity and task complexity, excelling not only in the breadth of tasks it can perform but also in the intricacy of those tasks.

#### 4.1.5 Data Generation with LLMs and VGMs

Recently, we have witnessed the power of content generation ability of LLMs and VGMs. Utilizing this ability, researchers have begun attempts to address the data scarcity problem by using foundation models to generate data. Ha et al. [[[142](#bib.bib142)]] propose a framework, ‘scaling up and distilling down’, which, given natural language instructions, can automatically generate diverse robot trajectories labeled with success conditions and language. RoboGen by Wang et al. [[[143](#bib.bib143)]] further enhances this approach by incorporating automatic action trajectory proposals within a physics-realistic simulation environment, Genesis, enabling the generation of potentially-infinite data. Nevertheless, these approaches still face limitations: the data generated suffers from low diversity in assets and robot morphologies, issues that could be ameliorated with advanced simulations or hardware. GenSim [[[123](#bib.bib123)]] by Wang et al. proposes to generate novel long-horizon tasks with LLMs given language instructions, leading to over 100 simulation tasks for training language-conditioned multitask robotic policy. This framework demonstrates task-level generalization in both simulation and the real world, and sheds some light on how to distill foundation models to robotic policies through simulation programs. ROSIE by Yu et al. [[[144](#bib.bib144)]] uses a text-guided image generator to modify the robot’s visual observation to perform data augmentation when training the control policy. The modification commands are from the user’s language instruction, then the augmentation regions are localized by the open vocabulary segmentation model. RT-Trajectory [[[145](#bib.bib145)]] generates trajectories for the policy network to condition on. The trajectory generation also helps the task specification in the robot learning tasks. Black et al. [[[146](#bib.bib146)]] use a diffusion-based model to generate subgoals for a goal-conditioned RL policy for manipulation [[[147](#bib.bib147)]].

#### 4.1.6 Enhancing Planning and Control Power through Prompting

The technique of Chain-of-Thought, as introduced by Wei et al. [[[148](#bib.bib148)]], compels the LLM to produce intermediate steps alongside the final output. This approach leverages a broader context window to list the planning steps explicitly, which enhances the LLM’s ability to plan. The underlying reason for its effectiveness is the GPT series’ nature as an autoregressive decoder. Semantic similarities are more pronounced between instructions to steps and steps to goal, than between instructions to the direct output. Nonetheless, the sequential nature of the Chain-of-Thought implies that a single incorrect step can lead to exponential divergence from the correct final answer [[[149](#bib.bib149)]].

Alternative methodologies attempt to remedy this by explicitly listing steps within graph [[[150](#bib.bib150)]] or tree structures [[[151](#bib.bib151)]], which have demonstrated improved performance. Additionally, search-based methods such as Monte Carlo Tree Search (MCTS) [[[152](#bib.bib152)]] and Rapidly-exploring Random Tree (RRT) [[[127](#bib.bib127)]] have been explored to augment planning capabilities.

Furthermore, translating goal specifications from natural language into external planning languages, such as the Planning Domain Definition Language (PDDL), has also been shown to increase planning accuracy [[[153](#bib.bib153)]]. Finally, as opposed to an open-loop prompting style, iterative prompting approaches that incorporate feedback from the environment provide a more grounded and precise enhancement for long-term planning capability [[[154](#bib.bib154), [33](#bib.bib33)]].

### 4.2 Robotics Foundation Models (RFMs)

With the increasing amount of robotics datasets, containing state-action pairs from real robots, the class of Robotics Foundation Models (RFMs) have likewise become increasingly viable [[[30](#bib.bib30), [67](#bib.bib67), [31](#bib.bib31), [69](#bib.bib69), [155](#bib.bib155), [156](#bib.bib156)]]. These models are characterized by the use of robotics data to train them, to solve robotics tasks. In this subsection, we summarize and discuss different types of RFMs. We will first introduce RFMs that can perform tasks according to a single robotic capability (e.g., perception, planning, and control), which is defined as single-purpose Robotic Foundation Models. For example, an RFM that can generate low-level actions to control the robot, or a model that can generate higher-level motion plans. We later introduce RFMs that can carry out tasks in multiple robotic modules, hence general-purpose models that can perform perception, control, and even non-robotic tasks [[[32](#bib.bib32), [33](#bib.bib33)]].

#### 4.2.1 Robotics Action Generation Foundation Models

Robotic action foundation models could take raw sensory observations, e.g., images or videos, and learn control output that is directly applied to robotic end-effectors. Models in this category include RT series [[[29](#bib.bib29), [30](#bib.bib30), [67](#bib.bib67)]], RoboCat [[[92](#bib.bib92)]], MOO [[[157](#bib.bib157)]], etc. According to the papers’ results, these models show generalization in robot control tasks such as manipulation.

##### Imitation Learning

Imitation learning has been applied to robot control for a considerable period. Initially, methods primarily aimed at imitating a single skill. The concept of utilizing imitation learning to master multiple tasks, which is the focus of this review, emerged with the works of [[[158](#bib.bib158), [159](#bib.bib159)]]. This approach is known as one-shot imitation learning. Various conditions are employed to define the task, including robot goal images [[[158](#bib.bib158), [159](#bib.bib159), [160](#bib.bib160), [161](#bib.bib161), [162](#bib.bib162), [163](#bib.bib163), [164](#bib.bib164), [165](#bib.bib165)]], human goal images [[[16](#bib.bib16), [166](#bib.bib166), [167](#bib.bib167)]], language prompts [[[168](#bib.bib168), [169](#bib.bib169), [16](#bib.bib16)]], and task vectors [[[170](#bib.bib170)]], among others. A recent study also investigates the use of multi-modal input as a task descriptor [[[171](#bib.bib171)]]. These work represent early exploration in this direction. Lately, efforts have been made to scale this approach, emulating the success of large language models. Lynch et al.[[[172](#bib.bib172)]] employed behavioral cloning on a dataset of hundreds of thousands of language-annotated trajectories to learn tabletop block rearrangement in the real world. Brohan et al.[[[29](#bib.bib29)]] trained a large transformer model on over 130K tabletop rearrangement episodes with a more realistic setup and diverse objects, demonstrating the robustness of imitation learning algorithms on some unseen tasks. RoboCat[[[92](#bib.bib92)]] and RT-X[[[67](#bib.bib67)]] trained a single model with data from multiple embodiments, showing some generalization capability to unseen embodiments during testing. Overall, these efforts have primarily shown success in pick-and-place tasks or variants thereof, indicating significant potential for further exploration.

##### Reinforcement Learning at Scale

With the availability of large-scale robotic datasets, offline RL starts to play an important role in developing effective RFMs. Early large-scale offline RL models such as QT-OPT [[[58](#bib.bib58)]] use a Q-learning-based approach in an offline manner to learn policy from robotics data which are collected by a robot farm. The successors of QT-OPT extend it to multitask learning by incorporating multi-task curriculum or predictive information [[[15](#bib.bib15), [173](#bib.bib173), [174](#bib.bib174)]]. Recently, with the success of Transformer models, Q-learning based on transformer (Q-Transformer) also shows its potential [[[175](#bib.bib175)]]. PTR [[[176](#bib.bib176)]] is another promising work that adopts Conservative Q-Learning (CQL) [[[177](#bib.bib177)]] in a multi-task learning setting. When it comes to online RL, due to the large exploration space, most methods trained a visual policy in the simulation [[[178](#bib.bib178), [179](#bib.bib179)]] and then transferred the model to real world [[[180](#bib.bib180)]]. Still, this field is challenging, and we look forward to seeing more RL-based robotic foundation models.

##### Vision and Language Pre-training

Another direction of action foundation models involves vision or language pre-training [[[181](#bib.bib181), [182](#bib.bib182), [183](#bib.bib183), [184](#bib.bib184), [185](#bib.bib185), [186](#bib.bib186), [187](#bib.bib187), [30](#bib.bib30)]]. For example, inspired by the great generalization ability of self-supervised vision-based pre-training, MVP by Radosavovic et al. [[[184](#bib.bib184)]] trains visual representations on real-world images and videos from the internet and egocentric video datasets, via a masked autoencoder,
and demonstrated the effectiveness of scaling visual pre-training for robot learning. Following this work, RPT [[[185](#bib.bib185)]] proposes mask-pretraining with real robot trajectory data. VC-1 [[[187](#bib.bib187)]] did a comprehensive study on the effectiveness of vision-based pre-training on policy learning.
Despite the effectiveness of these visual pretraining methods, [[[186](#bib.bib186)]] reexamined some of these methods and discovered significant domain gaps, thus proposed learning-from-scratch approach. This provides us new perspective to think about visual pretraining in robotics.

Beyond just using visual information, RT-2 [[[30](#bib.bib30)]] and Moo [[[157](#bib.bib157)]] use vision and language pre-trained model as a control policy backbone. PaLM-E [[[33](#bib.bib33)]] and PALI-X [[[140](#bib.bib140)]] were used to transfer knowledge from the web into robot actions.
Slightly different from previous methods, VRB [[[188](#bib.bib188)]] learns affordance functions (instead of the policy itself) from large-scale video pertaining, providing another thought process for us to study how RFMs may generalize in real-world tasks.

Using a similar approach as in pretraining with vision or language modality, we also see self-supervised pretraining with less-explored modalities such as audio [[[189](#bib.bib189)]] and tactile sensing [[[190](#bib.bib190)]].

##### Robotics Motion Planning Foundation Models

Recently we have seen the rise of RFMs especially used for motion-planning purposes in visual navigation tasks [[[191](#bib.bib191), [31](#bib.bib31), [192](#bib.bib192)]]. These foundation models take advantage of the large-scale heterogeneous data and show generalization capability in predicting high-level motion-planning actions. These methods rely on rough topological maps [[[191](#bib.bib191), [31](#bib.bib31)]] consisting of only image observations
instead of accurate metric maps and accurate localization as in conventional motion-planning methods (as described in Section [3.3](#S3.SS3)). Unlike vision and language foundation models applied to motion planning, the robotic motion planning foundation model is still quite in its early stages.

#### 4.2.2 General-purpose Robotics Foundation Models

Developing general-purpose robotic systems is always a holy grail in robotics and artificial intelligence. Some existing works [[[32](#bib.bib32), [33](#bib.bib33)]] take one step towards this goal. Gato [[[32](#bib.bib32)]] proposes a multimodal, multi-task, and multi-embodiment generalist foundation model that can play Atari games, caption images, chat, stack blocks with a real robot arm, and more—all with the same model weights. Similar to Gato, PaLM-E [[[33](#bib.bib33)]] is also a general-purpose multimodal foundation model for robotic reasoning and planning, vision-language tasks, and language-only tasks.
Although not proven to solve all the robotics tasks that we introduced in Section [2](#S2), Gato and PaLM-E show a possibility of merging perception and planning into one single model. Moreover, Gato and PaLM-E show promising results of using the same model to solve various seemingly-unrelated tasks, highlighting the viability of general-purpose AI systems. Designed especially for robotic tasks, PACT [[[193](#bib.bib193)]] proposes one transformer-based foundation model with common pre-trained representations that can be used in various downstream robotic tasks, such as localization, mapping, and navigation. Although we have not seen many unified foundation models for robotics, we would expect more endeavors in this particular problem.

*Table 1: A quick summary of foundation models solving robotic challenges. Here we only list part of the works due to space limit. We find that uncertainly and safety are still largely unexplored.*

### 4.3 How do Foundation Models Help Solve Robotics Challenges

In Section [3](#S3), we listed five major challenges in Robotics. In this section, we summarize how foundation models—either vision and language models or robotic foundation models—could help resolve these challenges, in a more organized manner.

All the foundation models related to visual information, such as VFMs, VLMs, and VGMs, are used in the perception modules in Robotics. LLMs, on the other hand, are more versatile and can be applied in both planning and control. We also list RFMs here, and these robotic foundation models are typically used in planning and action generation modules. We summarize how foundation models solve the aforementioned robotic challenges in Table [1](#S4.T1). We notice from this table that all foundation models are good at generalization in tasks of various robotic modules. Also, LLMs are especially good at task-specification. RFMs, on the other hand, are good at dealing with the challenge of dynamics model
since most RFMs are model-free approaches. For robot perception, the challenges in generalization ability and model are coupled, since, if the perception model already has very good generalization ability, there’s no need to get more data for domain adaptation or additional fine-tuning. In addition, the call for solving the safety challenge is largely missing, and we will discuss the particular problem in Section [6](#S6).

##### Foundation Models for Generalization

Zero-shot generalization is one of the most significant characteristics of current foundation models. Robotics benefits from the generalization ability of foundation models in nearly all aspects and modules. For the first one, generalization in perception, VLM and VFM are great choices as the default robotics perception models. The second aspect is the generalization ability in task-level planning, with details of task plans generated by LLMs [[[26](#bib.bib26)]]. The third one is in generalization in motion-planning and control, by utilizing the power of RFMs.

##### Foundation Models for Data Scarcity

Foundation models are crucial in tackling data scarcity in robotics. They offer a robust basis for learning and adapting to new tasks with minimal specific data. For example, recent methods utilize foundation models to generate data to help with training robots, such as robot trajectories [[[142](#bib.bib142)]] and simulation [[[143](#bib.bib143)]]. These models excel in learning from a small set of examples, allowing robots to quickly adapt to new tasks using limited data. From this perspective, solving data scarcity is equivalent to solving the generalization ability problem in robotics. Apart from this aspect, foundation models—especially LLMs and VGMs—could generate datasets for robotics used in training perception modules [[[144](#bib.bib144)]] (see Section [4.1.5](#S4.SS1.SSS5), above), and for task-specification [[[145](#bib.bib145)]].

##### Foundation Models to Relieve the Requirement of Models

As discussed in Section [3.3](#S3.SS3), building or learning a model—either a map of the environment, a world model, or an environmental dynamics model—is vital for solving robotic problems, especially in motion-planning and control. However, the powerful few/zero-shot generalization ability that foundation models present may break that requirement. This includes using LLMs to generate task plans [[[26](#bib.bib26)]], using RFMs to learn model-free end-to-end control policies [[[29](#bib.bib29), [175](#bib.bib175)]], etc.

##### Foundation Models for Task-Specification

Task-specifications as language prompts [[[29](#bib.bib29), [30](#bib.bib30), [26](#bib.bib26)]], goal images [[[77](#bib.bib77), [171](#bib.bib171)]], videos of humans demonstrating the task [[[199](#bib.bib199), [200](#bib.bib200)]], rewards [[[28](#bib.bib28), [78](#bib.bib78)]], rough scratch of trajectory [[[145](#bib.bib145)]], policy sketches [[[201](#bib.bib201)]], and hand-drawn images [[[202](#bib.bib202)]] have allowed goal specifications in a more natural, human-like format. Multimodal foundation models allow users to not only specify the goal but also help resolve ambiguities via dialogue. Recent work in understanding trust and intent recognition within the human-robot interaction domain has opened up newer paradigms in our understanding of how humans use explicit and implicit cues to convey task-specifications. While significant progress has been made, recent work in prompt engineering for LLMs implies that even with a single modality, it is challenging to generate relevant outputs.
Vision-Language Models are proven to be especially good at task-specification, showing potential for resolving this problem in robotics. Extending the idea of vision-language-based task-specifications, explore methods to achieve multi-modal task specification using more natural inputs like images obtained from the internet [[[29](#bib.bib29)]] explores this idea of zero-shot transfer from task-agnostic data further, by providing a novel model class that exhibits promising scalable model properties. The model encodes high-dimensional inputs and outputs, including camera images, instructions, and motor commands into compact token representations to enable real-time control of mobile manipulators.

##### Foundation Models for Uncertainty and Safety

Though being a critical problem in robotics, uncertainty and safety using foundation models for robotics is still underexplored. Existing works like KNOWNO [[[83](#bib.bib83)]] proposes a framework for measuring and aligning the uncertainty of LLM-based task planners. Recent advances in Chain-of-Thought prompting [[[203](#bib.bib203)]], open vocabulary learning [[[204](#bib.bib204)]], and hallucination recognition in LLMs [[[205](#bib.bib205)]] may open up newer avenues to address these challenges.

## 5 Review of Current Experiments and Evaluations

In this section, we summarize the datasets, benchmarks, and experiments of the current research works.

### 5.1 Datasets and Benchmarks

Relying solely on knowledge learned from language and vision datasets is limiting. Some concepts, like friction or weight, are not easily learned through these modalities alone, as suggested by [[[206](#bib.bib206)]] and [[[115](#bib.bib115)]] in their works on physically grounded VLMs. Therefore, in order to make robotic agents that can better understand the world, researchers are not just adapting foundational models from the language and vision domains; they are also advancing the development of large, diverse, and multimodal robotic datasets for training or fine-tuning these foundation models. This effort is now diverging into two directions: collecting data from the real world, versus collecting data from simulations and then transferring it to the real world. Each direction has its pros and cons. We will cover these datasets and simulations in the following paragraphs and discuss their respective advantages and disadvantages.

#### 5.1.1 Real World Robotics Datasets

Real-world robotics datasets are highly appealing due to their diverse object classes and multimodal inputs, offering a rich resource for training robotic systems without the need for complex and often inaccurate physical simulations. However, creating these large-scale datasets presents a significant challenge, primarily due to the absence of a substantial ‘data flywheel’ effect. This effect, which greatly benefited fields like CV and NLP through contributions from millions of internet users, is less evident in robotics. The limited incentive for individuals to upload extensive sensory inputs and corresponding action sequences poses a major hurdle in data acquisition. Despite these challenges, current efforts are focused on addressing these gaps. RoboNet [[[207](#bib.bib207)]] is a notable effort in this direction, offering a large-scale, diverse dataset across different robotic platforms for multi-robot learning. Bridge Dataset V1 [[[208](#bib.bib208)]] collects 7200 hours of demonstrations in real household kitchen manipulation tasks, and its following Bridge-V2 [[[209](#bib.bib209)]] contains 60,096 trajectories collected across 24 environments on common low-cost robots. Language-Table [[[172](#bib.bib172)]] collects 600,000 language-labeled trajectories—an order of magnitude larger than prior available datasets. RT-1 [[[29](#bib.bib29)]] contains 130k episodes that cover 700+ tasks, collected using a fleet of 13 Google mobile manipulation robots, over 17 months. While the aforementioned datasets represent significant advancement over prior lab-scale datasets, offering a relatively large volume of data, they are limited to single modalities or specific robot tasks.

To overcome these limitations, some recent initiatives have made notable progress. For example, GNM [[[191](#bib.bib191)]] successfully integrated six different large-scale navigation datasets, utilizing a unified navigation interface based on waypoints. A recent collaborative effort among various laboratories called RT-X [[[67](#bib.bib67)]] has aimed to standardize data across different datasets, by using a 7-degree-of-freedom end-effector’s pose as a universal reference across different embodiments. To offer a more comprehensive data source, RH20T [[[68](#bib.bib68)]] collected over 110,000 manipulation episodes across 7 embodiments, covering more than 140 diverse, contact-rich skills. The modalities include RGB, depth, end-link force-torque, tactile, proprioception, audio and language instruction. All sensors are well-calibrated and synchronized.

Building on these advancements, the scale of real-world robotics datasets is beginning to grow, albeit still lagging behind the immense volume of internet-scale language and vision corpora. The accessibility of advanced hardware such as the Hello Stretch Robot, Unitree Quadrupeds, and open-source dexterous manipulators [[[210](#bib.bib210)]] is expected to catalyze this growth. As these technologies become more widely available, they are likely to initiate the desired ‘data flywheel’ effect in Robotics.

#### 5.1.2 Robotics Simulators

While we await the widespread deployment of robotic hardware to gather massive amounts of robot data, another approach is to develop simulators that closely mimic real-world graphics and physics. The advantage of using simulation is the ability to deploy tens of thousands of robot instances in a simulated world, enabling simultaneous data collection.

Simulators focus on different aspects, such as photorealism, physical realism, and human-in-the-loop interactions. For navigation tasks, photorealistic simulators are crucial. AI Habitat addresses this by utilizing realistically-scanned 3D scenes from the Matterport3D [[[211](#bib.bib211)]] and Gibson [[[212](#bib.bib212)]] datasets. Furthermore, Habitat [[[213](#bib.bib213)]] is a simulator that allows AI agents to navigate through various realistic 3D spaces and perform tasks, including object manipulation. It features multiple sensors and handles generic 3D datasets. Habitat 2.0 [[[65](#bib.bib65)]] builds upon the original by introducing dynamic scene modeling, rigid-body physics, and increased speed. Habitat 3.0 [[[66](#bib.bib66)]] further integrates programmable humanoids to enhance the simulation experience. Additionally, the AI2THOR simulator [[[214](#bib.bib214)]] is another promising framework for photorealistic visual foundation model research, as evidenced in [[[93](#bib.bib93), [215](#bib.bib215)]]. Other simulators, like Mujoco [[[61](#bib.bib61)]], focus on creating physically realistic environments for advanced manipulation and locomotion tasks.

Moreover, simulators like AirSim [[[216](#bib.bib216)]] and the Arrival Autonomous Racing Simulator [[[50](#bib.bib50)]], both built on Unreal Engine, provide a balance of reasonable physics and photorealism. Ultimately, while the aforementioned simulators excel in various areas, they face common challenges such as parallelism. Simulators like Issac Gym [[[62](#bib.bib62)]] and Mujoco 3.0 [[[217](#bib.bib217)]] have attempted to overcome these challenges by using GPU acceleration to expedite the data-collection process.

Despite the abundance of data available in simulators, there are inherent challenges in their use. Firstly, the domain gap between simulations and the real world makes transferring from sim to real problematic—issues that early works are already seeking to resolve [[[49](#bib.bib49)]]. Secondly, the diversity of environments and base objects is still lacking. Therefore, to effectively utilize simulations in the future, continuous improvements in these two areas are essential.

### 5.2 Analysis of Current Method Evaluation

We conduct a meta-analysis of the experiments of papers listed in Tables [2](#Sx3.T2) to [7](#Sx3.T7) and Figure [7](#S5.F7), encouraging readers to consider the following questions

-
1.

What tasks are being solved?

-
2.

On what datasets or simulators have they been trained? What robot platforms are used for testing?

-
3.

What foundation models are being utilized? How effectively are the tasks solved?

-
4.

What base foundation models are more frequently used in these methods?

We summarize several key trends observed in the current literature concerning the experiments conducted:

##### Imbalanced Focus among Manipulation Tasks:

There is a significant emphasis on general pick-place tasks, particularly tabletop and mobile manipulation. This is likely due to the ease of training for tabletop gripper-based manipulation skills and their potential to form skill libraries that interact with foundation models. However, there is a lack of extensive exploration in low-level action outputs, such as dexterous manipulation and locomotion.

##### Need for Improved Generalization and Robustness

Generalization and robustness of end-to-end foundational robotics models have room for improvement. In tabletop manipulation, the use of foundation models leads to performance drops ranging from 21% [[[129](#bib.bib129), [29](#bib.bib29)]] to 31% [[[30](#bib.bib30)]] in unseen tasks. In addition, these models still need improved robust to disturbances, performance drops 14% [[[29](#bib.bib29)]] to 18% [[[129](#bib.bib129)]] for similar tasks.

##### Limited Exploration in Low-Level Actions

There remains a gap in the exploration of direct low-level action outputs. The majority of research focuses on task-level planning and utilizes foundation models with pre-trained or pre-programmed skill libraries. However, existing papers [[[30](#bib.bib30), [67](#bib.bib67), [32](#bib.bib32)]] that explore low-level action outputs mainly focus on table-top manipulation, where the action space is limited to the end effector’s 7 degrees of freedom (DoF). Models that directly output joint angles for tasks like dexterous manipulation and locomotion still require a more thorough research cycle.

##### Control Frequencies Too Slow to be Deployed on Real Robots

Most current approaches to robotic control are open-loop, and even those that are closed-loop face limitations in inference speed. These speeds typically range from 1 to 10 Hz, which is considered low for the majority of robotics tasks. Particularly for tasks like humanoid locomotion, a high-frequency control of around 500 Hz is required for the stabilization of the robot’s body [[[218](#bib.bib218)]].

##### Lack of Uniform Benchmarks for Testing

The diverse nature of simulations, embodiments, and tasks in robotics leads to varied benchmarks, complicating the comparison of results. Additionally, while success rate is often used as the primary metric, it may not sufficiently evaluate the performance of real-world tasks involving large foundation models, as latency is not captured by the success rate alone. More nuanced evaluation metrics that consider inference time, such as the Compute Aware Success Rate (CASR) [[[127](#bib.bib127)]].

![Figure](x4.png)

*Figure 7: The histogram showing the number of times different base foundation models are used in developing robotics systems, among the papers we included in this survey. In the plot we can see GPT-4, GPT-3 are among top choices due to their few-shot promptable nature, as well as accessibility through APIs. CLIP and ViLD are frequently used to bridge image and text representations. Apart from CLIP, T5 family models are frequently used to encode text to get text features. PaLM and PaLM-E are used for robot planning. RT-1, which is originally developed for manipulation, emerges as a new base model which other manipulation models are built upon.*

## 6 Discussions and Future Directions

### 6.1 Remaining Challenges and Open Discussions

##### Grounding for Robot Embodiment

Although numerous strategies have been explored to address the problem of grounding, as discussed in Section [4.1.4](#S4.SS1.SSS4), there are many open challenges in this area.
First, grounding needs an effective medium or interface that bridges concepts and robot actions. Existing interfaces, such as those employing natural language [[[26](#bib.bib26), [33](#bib.bib33)]] and code [[[122](#bib.bib122), [121](#bib.bib121)]], are limited. While concepts can be articulated through language and code, they are not universally applicable to nuances such as dexterous body movements. Furthermore, these interfaces often depend on predefined skill libraries that are not only time-intensive to develop but also lack generalization to new environments. Using reward as an interface [[[134](#bib.bib134), [133](#bib.bib133), [78](#bib.bib78)]] may alleviate some of the generalization issues in simulations by acquiring skills dynamically. However, the time-consuming and potentially unsafe nature of training RL algorithms in the real world raises questions about the feasibility of this method, with real-world validations of its effectiveness yet to be demonstrated.

Second, we need to move from an unimodal notion of grounding, like mapping the word to meaning to a more holistic grounding of multiple sensory modalities.
Approaches that rely solely on visual data [[[206](#bib.bib206)]] may capture certain physical properties such as material, transparency, and deformability. Yet, they fall short in grasping concepts like friction, which requires interactive data with proprioceptive feedback, or the scent of an object, which cannot be acquired without additional modalities such as olfaction.

Lastly, we should consider grounding from an embodiment perspective. The same task may necessitate distinct actions based on the robot’s embodiment; for example, opening a door would require drastically different maneuvers from a humanoid robot compared to a quadruped. Current research on grounding often emphasizes environmental adaptation while affording less consideration to how embodiment shapes interaction strategies.

##### Safety and Uncertainty

As we pursue deployments of real robots to work alongside humans in factories, to provide elderly care, or to assist in homes and offices, these autonomous systems (and the foundation models that power them) will require more effective measures of safety. While formal hardware and software safety checks still apply, the use of foundation models to support provable safety analysis will become an increasingly necessary direction. With the goal of deploying robots to safety-critical scenarios, prior works have considered leveraging Lyapunov-style safety index functions [[[219](#bib.bib219), [220](#bib.bib220), [88](#bib.bib88)]], in attempts to provide hard safety guarantees for nonlinear systems with complex dynamics and external disturbances (see also Section [3.5](#S3.SS5)). Traditionally, the systems under consideration by the provable safety literature are of low dimension, often require careful specification of a world/dynamics model, require specifying an initial safe set and/or set-boundary distance functions, require some heuristics and training “tricks” to obtain useful safety value functions that balance conservativeness versus performance, do not naturally support multi-agent settings, and present challenges in safely updating the safety value function and growing the safe set online. Herbert et al. [[[220](#bib.bib220)]] synthesized several techniques into a framework—thereby easing computation, streamlining updates to the safe sets by one or more orders of magnitude compared to the prior art, and extending Hamilton-Jacobi Reachability analysis to 10-dimensional systems that govern quadcopter control. Chen et al. [[[88](#bib.bib88)]] combine RL with HJ Reachability analysis to learn safety value functions from high-dimensional inputs (RGB images, plus vehicle state), to trade off a performance-oriented policy and a safety-oriented policy, within a jointly-optimized dual actor-critic framework, for simulated autonomous racing. Tian et al. [[[221](#bib.bib221)]] integrate HJ Reachability analysis in the context of multi-agent interactions in urban autonomous driving, by formulating the problem as a general-sum Stackelberg game.

However, in all of these works, open questions remain on integrating socially-acceptable safety constraints and formal guarantees for systems with robotic foundational models. One of the directions is to formulate safety as an affordance [[[222](#bib.bib222)]]. The definition of safety changes based on the capability of the robot and social context. Another focus for safety is to ensure robust alignment of the robot’s inferred task specification to a human user’s communicative intent. Foundation models offer a way to encode the enormous world knowledge, which can serve as commonsense priors to decode the underlying intent. Recent works improve the use of LLMs for robotics with conformal prediction [[[83](#bib.bib83)]] and explicit constraint checking [[[223](#bib.bib223)]]. Despite these advances, foundation models currently lack native capacity to reason about the uncertainty associated with their outputs. If properly calibrated, uncertainty quantification in foundation models can be used to trigger fall-back safety measures like early termination, pre-defined safe maneuvers, or human-in-the-loop interventions.

##### Is there a Dichotomy between End-to-End and Modular Approaches?

The human brain serves as an example of a functional approach to learning and generalization. While neuroscientists have identified specific regions of the brain, such as the visual cortex, somatosensory cortex, and motor cortex, the brain demonstrates remarkable plasticity and the ability to reorganize its functions to adapt to changes or brain lesions. This flexibility suggests that the brain may have evolved to be modular as a consequence of unified training, combining specific functionalities while maintaining the capacity for general learning [[[224](#bib.bib224), [225](#bib.bib225)]].
Similarly, in “Bertology”, NLP researchers show how local parts of trained networks can specialize in one area over others. This indicates that certain modules of large-scale models may become highly specialized for specific functions, which can be adapted for downstream tasks without re-training the entire network. This transfer learning approach can lead to more efficient use of computational resources and faster adaptation to new tasks.

In the context of robotics, taking a premature stand for either modular or end-to-end policy architectures may limit the potential of foundation models for robotics. Modular solutions can provide specific biases and enable effective task-specific performance, but they may not fully leverage the potential of general learning and transferability. On the other hand, end-to-end solutions have a history of working well on certain tasks on CV and NLP, but they might not offer the desired flexibility for adaptation to new situations. As [[[226](#bib.bib226)]] pointed out, there appears to be a misconception about the modular versus end-to-end dichotomy. This is because the former pertains to architecture while the latter relates to optimization – they are not mutually exclusive.

Regarding the architecture and optimization design for foundation models used in robotics, we can focus on a functional approach rather than categorizing it as either modular or end-to-end differentiable. One of the goals of a robotic foundational model is to allow flexible modular components, each responsible for specific functionalities, with unified learning that leverages shared representations and general learning capabilities.

##### Adaptability to Physical Changes in Embodiment

From employing a pen to flip a light switch to maneuvering down a staircase with a broken leg encased in a cast, the human brain demonstrates versatile and adaptable reasoning. It is a single unit that controls perceptual understanding, motion control, and dialogue capabilities. For motion control, it adapts to the changes in the embodiment, due to tool use or injury. This adaptability extends to more profound transformations, such as individuals learning to paint with their feet or mastering musical instruments with specialized prosthetics. We want to build such interactive and adaptable intelligence in Robotics.

In the previous discussions, we saw existing works successfully deploying navigation foundation models for various robot platforms [[[31](#bib.bib31)]], such as different wheeled robots and quadrupedal robots. We also witnessed the manipulation foundation model used in different manipulators [[[30](#bib.bib30), [48](#bib.bib48)]] which can be used across different robotic platforms, ranging from tabletop robot arms to mobile manipulators.

One of the key open research question is how robotics foundational models should enable motion control across different physical embodiments [[[198](#bib.bib198)]]. Initial results [[[198](#bib.bib198)]] show the possibility of one model for different policies across different embodiment.

Robot policies deployed in homes and offices must be robust to mechanical motion failures, such as sensor malfunctions or actuator breakdowns, ensuring continued functionality in challenging environments. Furthermore, robotic systems must be designed to adapt to a variety of tools and peripherals, mirroring the human capability to interact with different instruments for specific tasks and physical tool uses.

##### World Model, or Model-agnostic?

In classical robotics, especially in planning and control problems, it was common to attempt to model as much as possible about the world that would be needed for robotics tasks. This was often carried out by leveraging structural priors about the tasks, or by relying on heuristics or simplifying assumptions. Certainly, if it was possible to perfectly model the world, solving robotics problems would become a lot simpler. Unfortunately, due to the complexity of the real world, world modeling in Robotics remains extremely difficult and sometimes even intractable. As a consequence, obtaining policies that generalize across tasks and environments remains a core problem.

The foundation models surveyed in this paper mostly take a model-agnostic (model-free) approach, leveraging the strength of expansive datasets and large-scale deep learning architectures. Some exceptions have attempted to emulate model-based approaches by directly employing LLMs as dynamic models. However, these attempts are still constrained by the inherent limitations of text-only descriptions and are prone to encountering issues with hallucinations, as discussed in [[[195](#bib.bib195), [127](#bib.bib127)]]. Many researchers would argue [[[227](#bib.bib227)]] that the data-scaled learning paradigm of these foundation models is still quite different from how humanity and animals learn, which is in an extreme data- and energy-efficient manner. Achieving even close to the joint performance and efficiency of human learning ability remains intriguing. In [[[227](#bib.bib227)]], LeCun argues that one possible answer to resolving that puzzle may lie in the learning of world models, a model that predicts how the state of the world going to change as consequences of the actions taken.

If we were to develop world models that can emulate the precision of the world’s representation through rigorous mathematical and physical modeling, it would bring us significantly closer to addressing and generalizing complex issues in robotics. These sophisticated and reliable world models would enable the application of established model-based methodologies, including search-based and sample-based planning, as well as trajectory optimization techniques. This approach would not only facilitate the resolution of planning and control challenges in robotics but also augment the explainability of these processes. It is posited that the pursuit of a ’foundation world model’, characterized by remarkable generalization abilities and zero-shot learning capabilities, holds the potential to be a paradigm-shifting development in the field. And we have already seen the reality of ’foundation world model’ getting closer with action/language conditioned video generation [[[228](#bib.bib228), [229](#bib.bib229)]]

##### Novel Robotics Platforms and Multi-sensory Information

As demonstrated in Figure [4](#S3.F4) and the Meta-analysis in Tables [2](#Sx3.T2)-[7](#Sx3.T7), existing real robot platforms utilized for deploying foundation models are predominantly limited to gripper-based, single-arm robot manipulators. The range of concepts learnable from tasks executed by these hardware systems is restricted, primarily because the simple opening and closing actions of a gripper are easily describable by language. To enable robots to achieve a level of dexterity and motor skills comparable to those of animals and humans, or to perform complex domestic tasks, it is essential for foundation models to acquire a deeper understanding of physical and household concepts. This learning necessitates a broader spectrum of information sources, such as diverse sensors (including smell, tactile, and thermal sensors), and more intricate data such as proprioception data from robot platforms with high degrees of freedom.

Current dexterous manipulators, e.g., Shadow Hand [[[230](#bib.bib230)]], are prohibitively expensive and prone to frequent breakdowns, hence they are predominantly experimented with in simulation. Moreover, tactile sensors are still limited in their application, often confined to the fingertips, as in [[[231](#bib.bib231)]], or offer only low resolution, as observed in the robot-sweater [[[232](#bib.bib232)]]. Recent progress has started to explore full-hand, high-resolution tactile sensing for humanoid hands [[[233](#bib.bib233)]].

Furthermore, since the bulk of data-collection is still conducted through human demonstrations, platforms that facilitate more accurate and efficient data acquisition, such as ALOHA [[[234](#bib.bib234)]], AirExo [[[235](#bib.bib235)]] and Leap Hands [[[210](#bib.bib210)]], are gaining popularity. Therefore, we posit that significant contributions are yet to be made—not only in terms of software innovations, but also in hardware. These advancements are crucial for providing richer data-collection and, thus, expanding the conceptual space of robotics foundation models.

##### Continual Learning

Continual learning broadly refers to the ability to learn and adapt to dynamic and changing environments. Specifically, it refers to learning algorithms that can learn and adapt to the underlying training data distribution and changing learning objective, as they evolve through time.

Continual learning is challenging, as neural network models often suffer from catastrophic forgetting, leading to a significant decrease in overall model performance on prior tasks.
One naive solution to mitigate performance degradation due to catastrophic forgetting involves periodically re-training models with the entire dataset collected, which generally allows models to avoid forgetting issues, since the process encompasses both old and new data. However, this method demands significant computational and memory resources. In contrast, training or fine-tuning only on new tasks or current data, without revisiting previous data, is less resource-intensive but incurs catastrophic forgetting due to the model’s tendency to overwrite previously learned information. This forgetting can be attributed to task interference between old and new data, concept drifts as data distributions evolve over time, and limitations in model expressivity based on their size.

Additionally, with the increasing capacities of models, continuously re-training them on expanding data corpora becomes less feasible. Recent works in vision and language continual learning [[[236](#bib.bib236), [237](#bib.bib237), [238](#bib.bib238), [239](#bib.bib239)]] have proposed various solutions, yet achieving effective continual learning, that can be applied to robotics, still remains a challenging objective.
For continual learning, large pre-trained foundational models currently face the above challenges and more, primarily because their extensive size makes retraining more difficult. In Robotics applications, specifically, continual learning is imperative to the deployability of robot learning policies in diverse environments, yet it is still a largely-unexplored domain. Whereas some recent works have studied various sub-topics of continual learning [[[240](#bib.bib240)]]—e.g., incremental learning [[[241](#bib.bib241)]], rapid motor adaptation [[[242](#bib.bib242)]], human-in-the-loop learning [[[243](#bib.bib243), [244](#bib.bib244)]]—these solutions are often designed for a single task/platform and do not yet consider foundation models.

We need continual learning algorithms that are designed with machine learning fundamentals in mind and practical real-time systems considerations. Some open research problems and viable approaches are:
(1) mixing different proportions of the prior data distribution when fine-tuning on latest data to alleviate catastrophic forgetting [[[245](#bib.bib245)]], (2) developing efficient prototypes from prior distributions or curriculum to learn new tasks [[[246](#bib.bib246)]] for task inference, (3) improving training stability and sample-efficiency of online learning algorithms [[[247](#bib.bib247), [248](#bib.bib248)]], and (4) identifying principled ways to seamlessly incorporate large-capacity models into control frameworks (perhaps with hierarchical learning [[[249](#bib.bib249), [250](#bib.bib250), [251](#bib.bib251)]] / slow-fast control [[[252](#bib.bib252)]]) for real-time inference.

##### Standardization and Reproducibility

The robotics community needs to encourage standardized and reproducible research practices to ensure that published findings can be validated and compared by others. To enable reproducibility at scale, we need to bridge the gap between simulated environments and real-world hardware and improve the transferability of ML models. Homerobot [[[100](#bib.bib100)]] is a promising step towards enabling both simulation and hardware platforms for open vocabulary pick-and-place tasks. However, we need to establish standardized task definitions and affordances to handle different robot morphologies, enabling more efficient model development.

##### Simulation or Real-world Data Collection

The path to robotic foundation models could be diverse. One method to achieve this is through large-scale real-world collected data [[[67](#bib.bib67), [253](#bib.bib253)]]. Another approach, however, is though simulation data and reduce the sim-to-real gap [[[254](#bib.bib254)]]. Simulation has the advantage of providing theoretically unlimited amount of data, training robot learning policies without worrying about safety concerns, and potential efficient training via parallelism. The large-scale training on simulation has also shown good generalization abilities in robot policy learning [[[255](#bib.bib255)]] and perception models [[[256](#bib.bib256)]]. However, building the twin version of the real world also faces various of challenges. We look forward to seeing the potential of both these two approaches to robotic foundation models.

##### Deployability to Industrial Settings

Discussions in previous subsections have primarily focused on academic settings, but foundation models for robotics have significant potential to transform unstructured industrial environments such as construction, oil and gas, solar farms, mines, and agriculture (see [Fig. 8](#S6.F8)).
These environments are dynamic, large in scale, and can lack prior maps or external communication (e.g., can be GPS-denied). Additionally, hazardous materials, physical constraints like background noise, and limited visibility (e.g., masks and helmets) pose significant human-robot interaction challenges that bring to surface novel problems that must be addressed in such settings.

![Figure](extracted/5892243/figs/field/aliengo_bg_v2_no_gradient_brighter.jpg)

![Figure](extracted/5892243/figs/field/humanoid_construction_behind_no_overlay.png)

![Figure](extracted/5892243/figs/field/racer_photo_16_9.jpg)

![Figure](extracted/5892243/figs/field/unitree_construction_smaller.png)

Figure 8: Industrial environments present unique challenges like dynamic obstacles, unstructured terrains, and safety risks. Foundation models in robotics offer the potential to navigate, inspect, and optimize operations in such demanding settings, improving safety and efficiency while enabling real-time decision-making and interaction with existing machinery.

Foundation models could revolutionize industrial operations by enabling robots navigate and inspect demanding outdoor settings, while supporting applications including automated data analysis [[[257](#bib.bib257)]], natural language processing for actionable insights [[[258](#bib.bib258)]], automated documentation, job site activity recognition [[[259](#bib.bib259)]], field operations optimization [[[260](#bib.bib260)]], and safety violation detection [[[261](#bib.bib261)]].
For instance, ConceptFusion [[[262](#bib.bib262)]] and Open-Fusion [[[263](#bib.bib263)]] enable open-set multimodal 3D mapping capabilities, potentially useful for dynamically-changing industrial settings where predefined object categories are unavailable.
Such advancements can enable more flexible and adaptive robot performance in unstructured settings.

However, deploying these models in industrial contexts presents unique challenges.
Ensuring safety and risk awareness to meet stringent industrial safety standards is crucial.
Industrial robots often must adapt to dynamic obstacles and situations, varying surface conditions, uneven slopes, weather conditions, while maintaining awareness of potential hazards, including heavy machinery, all of which must be managed by the foundation model.
For example, in construction environments, robots must navigate ever-changing sites due to ongoing project development, all while making safety-critical decisions [[[264](#bib.bib264)]].
In mining, such foundation models must handle adverse conditions including low visibility, variable terrains, high dust levels, and unstable surfaces [[[265](#bib.bib265)]].
The variability of these real-world environments means that foundation models deployed to such settings must operate in out-of-distribution settings, increasing the risk of unsafe decisions and erroneous outputs.
ConceptGraphs [[[194](#bib.bib194)]], for example, can provide a framework to represent aspects of such environments using 3D scene graphs, while methods inspired by PlayFusion [[[266](#bib.bib266)]], which uses language annotation of unstructured / unguided robotics interaction data, bear potential to support robots in acquiring the necessary skills to interact with machinery.

Deployment also requires edge computing due to potential network limitations and latency while on-site.
These mobile robots, are also constrained by computational resources, operating and collecting data for real-time monitoring.
Techniques such as model quantization [[[267](#bib.bib267)]], distillation [[[268](#bib.bib268)]], and teacher-student based frameworks [[[269](#bib.bib269), [270](#bib.bib270)]] can help address these constraints, though balancing performance with model size remains an ongoing challenge.

### 6.2 Summary

In this survey paper, we analyzed the current research works on foundation models for robotics based on two major categories: (1) works which apply foundation models to robotic tasks, and (2) works attempting to develop robotics foundation models for robotics tasks using robotics data. We went through the methodologies and experiments of these papers, and provided analysis and insights based on these research works. Furthermore, we specially covered how these foundation models help resolve the common challenges in robotics. Finally, we discussed remaining challenges in robotics that have not been solved by foundation models, as well as other promising research directions.

## Meta Analysis Tables

Below is a detailed analysis answering the questions raised in section
[5.2](#S5.SS2). We list all the tables by classifying the topics as manipulation in Table [2](#Sx3.T2), dexterous manipulation [3](#Sx3.T3), mobile manipulation [4](#Sx3.T4), locomotion [5](#Sx3.T5), navigation [6](#Sx3.T6) and multi-task learning [7](#Sx3.T7).

## Disclaimer

Due to the rapidly changing nature of the field, we checkpointed this version of literature review on September 1st 2024, and might have missed some relevant work. In addition, due to the rich body of literature and the extensiveness of this survey, there may be inaccuracies or mistakes in the paper. We welcome readers to send pull requests to our GitHub repository (inside [https://robotics-fm-survey.github.io/](https://robotics-fm-survey.github.io/)) so we may continue to update our references, correct the mistakes and inaccuracies, as well as updating the entries of the meta studies in the paper. Please refer to the contribution guide in the GitHub repository.

## Acknowledgments

We would like to thank Vincent Vanhoucke for feedbacks on a draft of this survey paper.
In addition, we would like to thank Yu Quan Chong and Kedi Xu for insightful discussions about the papers list.

*Table 2: Tabletop Manipulation*

*Table 3: Dexterous Manipulation*

*Table 4: Mobile Manipulation*

*Table 5: Locomotion*

*Table 6: Navigation*

*Table 7: Multi-Tasks*

## References

-
[1]

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun.

Vision meets robotics: The kitti dataset.

International Journal of Robotics Research (IJRR), 2013.

-
[2]

Daniel Maturana, Po-Wei Chou, Masashi Uenoyama, and Sebastian Scherer.

Real-time semantic mapping for autonomous off-road navigation.

In Field and Service Robotics, pages 335–350. Springer, 2018.

-
[3]

Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M Dollar.

Yale-cmu-berkeley dataset for robotic manipulation research.

In International Journal of Robotics Research, page 261 – 268, 2017.

-
[4]

Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell.

Decaf: A deep convolutional activation feature for generic visual recognition.

In ICML, 2014.

-
[5]

Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell.

Adversarial discriminative domain adaptation.

In CVPR, 2017.

-
[6]

William Shen, Felipe Trevizan, and Sylvie Thiébaux.

Learning domain-independent planning heuristics with hypergraph networks.

In Proceedings of the International Conference on Automated Planning and Scheduling, volume 30, pages 574–584, 2020.

-
[7]

Beomjoon Kim and Luke Shimanuki.

Learning value functions with relational state representations for guiding task-and-motion planning.

In Conference on Robot Learning, pages 955–968. PMLR, 2020.

-
[8]

Grady Williams, Paul Drews, Brian Goldfain, James M. Rehg, and Evangelos A. Theodorou.

Aggressive driving with model predictive path integral control.

In ICRA, 2016.

-
[9]

Ahmed H Qureshi, Yinglong Miao, Anthony Simeonov, and Michael C Yip.

Motion planning networks: Bridging the gap between learning-based and classical motion planners.

IEEE Transactions on Robotics, pages 1–9, 2020.

-
[10]

Adam Fishman, Adithyavairavan Murali, Clemens Eppner, Bryan Peele, Byron Boots, and Dieter Fox.

Motion policy networks.

In Proceedings of the 6th Conference on Robot Learning (CoRL), 2022.

-
[11]

Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, and Sergey Levine.

Learning agile robotic locomotion skills by imitating animals.

In RSS, 2020.

-
[12]

Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel.

End-to-end training of deep visuomotor policies.

In Journal of Machine Learning Research, 2016.

-
[13]

Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter.

Learning agile and dynamic motor skills for legged robots.

In Science Robotics, 30 Jan 2019.

-
[14]

Anusha Nagabandi, Kurt Konoglie, Sergey Levine, and Vikash Kumar.

Deep dynamics models for learning dexterous manipulation.

In CoRL, 2019.

-
[15]

Dmitry Kalashnkov, Jake Varley, Yevgen Chebotar, Ben Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman.

Mt-opt: Continuous multi-task robotic reinforcement learning at scale.

arXiv:2104.08212, 2021.

-
[16]

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn.

BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning.

In 5th Annual Conference on Robot Learning, 2021.

-
[17]

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

Language models are few-shot learners, 2020.

-
[18]

Aditya Ramesh, Prafulla Dhariwal Alex Nichol, Casey Chu, and Mark Chen.

Hierarchical text-conditional image generation with clip latents.

arXiv preprint arXiv:2204.06125, 2022.

-
[19]

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar, Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi.

Photorealistic text-to-image diffusion models with deep language understanding.

arXiv preprint arXiv:2205.11487, 2022.

-
[20]

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin.

Emerging properties in self-supervised vision transformers.

In Proceedings of the International Conference on Computer Vision (ICCV), 2021.

-
[21]

Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski.

Dinov2: Learning robust visual features without supervision, 2023.

-
[22]

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick.

Segment anything.

arXiv:2304.02643, 2023.

-
[23]

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.

Learning transferable visual models from natural language supervision.

In ICML, 2021.

-
[24]

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan.

Flamingo: a visual language model for few-shot learning.

ArXiv, abs/2204.14198, 2022.

-
[25]

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.

On the opportunities and risks of foundation models.

arXiv preprint arXiv:2108.07258, 2021.

-
[26]

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al.

Do as i can, not as i say: Grounding language in robotic affordances.

arXiv preprint arXiv:2204.01691, 2022.

-
[27]

Boyuan Chen, Fei Xia, Brian Ichter, Kanishka Rao, Keerthana Gopalakrishnan, Michael S. Ryoo, Austin Stone, and Daniel Kappler.

Open-vocabulary queryable scene representations for real world planning.

In arXiv:2209.09874, 2022.

-
[28]

Yujin Tang, Wenhao Yu, Jie Tan, Heiga Zen, Aleksandra Faust, and Tatsuya Harada.

Saytap: Language to quadrupedal locomotion, 2023.

-
[29]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.

Rt-1: Robotics transformer for real-world control at scale.

arXiv preprint arXiv:2212.06817, 2022.

-
[30]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich.

Rt-2: Vision-language-action models transfer web knowledge to robotic control.

In arXiv preprint arXiv:2307.15818, 2023.

-
[31]

Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, and Sergey Levine.

Vint: A foundation model for visual navigation.

In arxiv preprint arXiv:2306.14846, 2023.

-
[32]

Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas.

A generalist agent.

In Transactions on Machine Learning Research (TMLR), 2022.

-
[33]

Danny Driess, F. Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Ho Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Peter R. Florence.

PaLM-E: An embodied multimodal language model.

ArXiv, abs/2303.03378, 2023.

-
[34]

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn.

Openvla: An open-source vision-language-action model, 2024.

-
[35]

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy.

Challenges and applications of large language models.

arXiv:2307.10169, 2023.

-
[36]

Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon.

Text-to-image diffusion models in generative ai: A survey.

arXiv:2303.07909, 2023.

-
[37]

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu.

Harnessing the power of llms in practice: A survey on chatgpt and beyond.

arXiv:2304.13712, 2023.

-
[38]

Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans.

Foundation models for decision making: Problems, methods, and opportunities.

arXiv:2303.04129, 2023.

-
[39]

Chaoning Zhang, Fachrina Dewi Puspitasari, Sheng Zheng, Chenghao Li, Yu Qiao, Taegoo Kang, Xinru Shan, Chenshuang Zhang, Caiyan Qin, Francois Rameau, Lik-Hang Lee, Sung-Ho Bae, and Choong Seon Hong.

A survey on segment anything model (sam): Vision foundation model meets prompt engineering, 2023.

-
[40]

Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan.

Foundational models defining a new era in vision: A survey and outlook, 2023.

-
[41]

Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao.

A survey of vision-language pre-trained models.

IJCAI-2022 survey track, 2022.

-
[42]

Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Ruotong Liao Gengyuan Zhang, Yao Qin, Volker Tresp, and Philip Torr.

A systematic survey of prompt engineering on vision-language foundation models.

arXiv:2307.12980, 2023.

-
[43]

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, Brian Ichter, Danny Driess, Jiajun Wu, Cewu Lu, and Mac Schwager.

Foundation models in robotics: Applications, challenges, and the future, 2023.

-
[44]

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen.

A survey on large language model based autonomous agents.

arXiv:2308.11432, 2023.

-
[45]

Jinzhou Lin, Han Gao, Rongtao Xu, Changwei Wang, Man Zhang, Li Guo, and Shibiao Xu.

The development of llms for embodied navigation.

In IEEE/ASME TRANSACTIONS ON MECHATRONICS, volume 1, Sept. 2023.

-
[46]

Anirudha Majumdar.

Robotics: An idiosyncratic snapshot in the age of llms, 8 2023.

-
[47]

Xuan Xiao, Jiahang Liu, Zhipeng Wang, Yanmin Zhou, Yong Qi, Qian Cheng, Bin He, and Shuo Jiang.

Robot learning in the era of foundation models: A survey, 2023.

-
[48]

Montserrat Gonzalez Arenas, Ted Xiao, Sumeet Singh, Vidhi Jain, Allen Z. Ren, Quan Vuong, Jake Varley, Alexander Herzog, Isabel Leal, Sean Kirmani, Dorsa Sadigh, Vikas Sindhwani, Kanishka Rao, Jacky Liang, and Andy Zeng.

How to prompt your robot: A promptbook for manipulation skills with code as policies.

In 2nd Workshop on Language and Robot Learning: Language as Grounding, 2023.

-
[49]

Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, and Ding Zhao.

What went wrong? closing the sim-to-real gap via differentiable causal discovery.

In 7th Annual Conference on Robot Learning, 2023.

-
[50]

James Herman, Jonathan Francis, Siddha Ganju, Bingqing Chen, Anirudh Koul, Abhinav Gupta, Alexey Skabelkin, Ivan Zhukov, Max Kumskoy, and Eric Nyberg.

Learn-to-race: A multimodal control environment for autonomous racing.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9793–9802, 2021.

-
[51]

Jonathan Francis, Bingqing Chen, Siddha Ganju, Sidharth Kathpal, Jyotish Poonganam, Ayush Shivani, Vrushank Vyas, Sahika Genc, Ivan Zhukov, Max Kumskoy, et al.

Learn-to-race challenge 2022: Benchmarking safe learning and cross-domain generalisation in autonomous racing.

ICML Workshop on Safe Learning for Autonomous Driving, 2022.

-
[52]

Jonathan Francis, Nariaki Kitamura, Felix Labelle, Xiaopeng Lu, Ingrid Navarro, and Jean Oh.

Core challenges in embodied vision-language planning.

Journal of Artificial Intelligence Research, 74:459–515, 2022.

-
[53]

Jonathan Francis.

Knowledge-enhanced Representation Learning for Multiview Context Understanding.

PhD thesis, Carnegie Mellon University, 2022.

-
[54]

Sriram Yenamandra, Arun Ramachandran, Mukul Khanna, Karmesh Yadav, Jay Vakil, Andrew Melnik, Michael Büttner, Leon Harz, Lyon Brown, Gora Chand Nandi, et al.

Towards open-world mobile manipulation in homes: Lessons from the neurips 2023 homerobot open vocabulary mobile manipulation challenge.

arXiv preprint arXiv:2407.06939, 2024.

-
[55]

Gyan Tatiya, Jonathan Francis, and Jivko Sinapov.

Transferring implicit knowledge of non-visual object properties across heterogeneous robot morphologies.

In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11315–11321. IEEE, 2023.

-
[56]

Gyan Tatiya, Jonathan Francis, and Jivko Sinapov.

Cross-tool and cross-behavior perceptual knowledge transfer for grounded object recognition.

arXiv preprint arXiv:2303.04023, 2023.

-
[57]

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov.

Scalability in perception for autonomous driving: Waymo open dataset.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

-
[58]

Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, and Sergey Levine.

Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation.

In CoRL, 2018.

-
[59]

Sergey Levine, Peter Pastor, Alex Krizhevsky, and Deirdre Quillen.

Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection.

In arXiv:1603.02199, 2016.

-
[60]

Alexander Herzog*, Kanishka Rao*, Karol Hausman*, Yao Lu*, Paul Wohlhart*, Mengyuan Yan, Jessica Lin, Montserrat Gonzalez Arenas, Ted Xiao, Daniel Kappler, Daniel Ho, Jarek Rettinghouse, Yevgen Chebotar, Kuang-Huei Lee, Keerthana Gopalakrishnan, Ryan Julian, Adrian Li, Chuyuan Kelly Fu, Bob Wei, Sangeetha Ramesh, Khem Holden, Kim Kleiven, David Rendleman, Sean Kirmani, Jeff Bingham, Jon Weisz, Ying Xu, Wenlong Lu, Matthew Bennice, Cody Fong, David Do, Jessica Lam, Yunfei Bai, Benjie Holson, Michael Quinlan, Noah Brown, Mrinal Kalakrishnan, Julian Ibarz, Peter Pastor, and Sergey Levine.

Deep rl at scale: Sorting waste in office buildings with a fleet of mobile manipulators.

In Robotics: Science and Systems (RSS), 2023.

-
[61]

Emanuel Todorov, Tom Erez, and Yuval Tassa.

Mujoco: A physics engine for model-based control.

In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

-
[62]

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State.

Isaac gym: High performance gpu-based physics simulation for robot learning, 2021.

-
[63]

Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin Yuan, Pooria Poorsarvi Tehrani, Ritvik Singh, Yunrong Guo, Hammad Mazhar, Ajay Mandlekar, Buck Babich, Gavriel State, Marco Hutter, and Animesh Garg.

Orbit: A unified simulation framework for interactive robot learning environments, 2023.

-
[64]

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al.

Habitat: A platform for embodied ai research.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339–9347, 2019.

-
[65]

Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra.

Habitat 2.0: Training home assistants to rearrange their habitat, 2022.

-
[66]

Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, Vladimír Vondruš, Theophile Gervet, Vincent-Pierre Berges, John M. Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi.

Habitat 3.0: A co-habitat for humans, avatars and robots, 2023.

-
[67]

Embodiment Collaboration.

Open x-embodiment: Robotic learning datasets and rt-x models, 2023.

-
[68]

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu.

RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot.

In IEEE international conference on robotics and automation (ICRA). IEEE, 2024.

-
[69]

Nur Muhammad Mahi Shafiullah, Anant Rai, Haritheja Etukuru, Yiqian Liu, Ishan Misra, Soumith Chintala, and Lerrel Pinto.

On bringing robots home.

arXiv preprint arXiv:2311.16098, 2023.

-
[70]

Elena Arcari, Maria Vittoria Minniti, Anna Scampicchio, Andrea Carron, Farbod Farshidian, Marco Hutter, and Melanie N. Zeilinger.

Bayesian multi-task learning mpc for robotic mobile manipulation, 2023.

-
[71]

Chao Cao, Hongbiao Zhu, Howie Choset, and Ji Zhang.

TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments.

In ICRA, 2023.

-
[72]

Fahad Islam, Oren Salzman, Aditya Agarwal, and Maxim Likhachev.

Provably constant-time planning and replanning for real-time grasping objects off a conveyor belt.

In RSS, 2020.

-
[73]

Dhruv Mauria Saxena, Muhammad Suhail Saleem, and Maxim Likhachev.

Manipulation planning among movable obstacles using physics-based adaptive motion primitives.

In 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2021.

-
[74]

Caelan Reed Garrett, Rohan Chitnis, Rachel Holladay, Beomjoon Kim, Tom Silver, Leslie Pack Kaelbling, and Tomas Lozano-P´erez.

Integrated Task and Motion Planning.

In arXiv:2010.01083, 2020.

-
[75]

Aidan Curtis, Xiaolin Fang, Leslie Pack Kaelbling, Tomas Lozano-Perez, and Caelan Reed Garrett.

Long-horizon manipulation of unknown objects via task and motion planning with estimated affordances.

In “IEEE International Conference on Robotics and Automation (ICRA)“, 2022.

-
[76]

Peiqi Liu, Yaswanth Orru, Chris Paxton, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto.

Ok-robot: What really matters in integrating open-knowledge models for robotics.

arXiv preprint arXiv:2401.12202, 2024.

-
[77]

Yuchen Cui, Scott Niekum, Abhinav Gupta, Vikash Kumar, and Aravind Rajeswaran.

Can foundation models perform zero-shot task specification for robot manipulation?

In Learning for Dynamics and Control Conference, pages 893–905. PMLR, 2022.

-
[78]

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar.

Eureka: Human-level reward design via coding large language models.

In 2nd Workshop on Language and Robot Learning: Language as Grounding, 2023.

-
[79]

Jakob Gawlikowski, Cedrique Rovile Njieutcheu Tassi, Mohsin Ali, Jongseok Lee, Matthias Humt, Jianxiang Feng, Anna Kruspe, Rudolph Triebel, Peter Jung, Ribana Roscher, et al.

A survey of uncertainty in deep neural networks.

Artificial Intelligence Review, 56(Suppl 1):1513–1589, 2023.

-
[80]

Xuhong Li, Haoyi Xiong, Xingjian Li, Xuanyu Wu, Xiao Zhang, Ji Liu, Jiang Bian, and Dejing Dou.

Interpretable deep learning: Interpretation, interpretability, trustworthiness, and beyond.

Knowledge and Information Systems, 64(12):3197–3234, 2022.

-
[81]

Ta-Chung Chi, Minmin Shen, Mihail Eric, Seokhwan Kim, and Dilek Hakkani-tur.

Just ask: An interactive learning framework for vision and language navigation.

In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 2459–2466, 2020.

-
[82]

Anastasios N Angelopoulos and Stephen Bates.

A gentle introduction to conformal prediction and distribution-free uncertainty quantification.

arXiv preprint arXiv:2107.07511, 2021.

-
[83]

Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, and Anirudha Majumdar.

Robots that ask for help: Uncertainty alignment for large language model planners.

In 7th Annual Conference on Robot Learning, 2023.

-
[84]

Kim P Wabersich, Andrew J Taylor, Jason J Choi, Koushil Sreenath, Claire J Tomlin, Aaron D Ames, and Melanie N Zeilinger.

Data-driven safety filters: Hamilton-jacobi reachability, control barrier functions, and predictive methods for uncertain systems.

IEEE Control Systems Magazine, 43(5):137–177, 2023.

-
[85]

Kai-Chieh Hsu, Haimin Hu, and Jaime Fernández Fisac.

The safety filter: A unified view of safety-critical control in autonomous systems.

arXiv preprint arXiv:2309.05837, 2023.

-
[86]

Aaron D Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada.

Control barrier functions: Theory and applications.

In 2019 18th European control conference (ECC), pages 3420–3431. IEEE, 2019.

-
[87]

Somil Bansal, Mo Chen, Sylvia Herbert, and Claire J Tomlin.

Hamilton-jacobi reachability: A brief overview and recent advances.

In 2017 IEEE 56th Annual Conference on Decision and Control (CDC), pages 2242–2253. IEEE, 2017.

-
[88]

Bingqing Chen, Jonathan Francis, Jean Oh, Eric Nyberg, and Sylvia L Herbert.

Safe autonomous racing via approximate reachability on ego-vision.

arXiv preprint arXiv:2110.07699, 2021.

-
[89]

Karen Leung, Nikos Aréchiga, and Marco Pavone.

Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods.

The International Journal of Robotics Research, 42(6):356–370, 2023.

-
[90]

Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Yaodong Yang, and Alois Knoll.

A review of safe reinforcement learning: Methods, theory and applications.

arXiv preprint arXiv:2205.10330, 2022.

-
[91]

Charles Dawson, Sicun Gao, and Chuchu Fan.

Safe control with learned certificates: A survey of neural lyapunov, barrier, and contraction methods for robotics and control.

IEEE Transactions on Robotics, 2023.

-
[92]

Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X. Lee, Maria Bauza, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, Antoine Laurens, Claudio Fantacci, Valentin Dalibard, Martina Zambelli, Murilo Martins, Rugile Pevceviciute, Michiel Blokzijl, Misha Denil, Nathan Batchelor, Thomas Lampe, Emilio Parisotto, Konrad Żołna, Scott Reed, Sergio Gómez Colmenarejo, Jon Scholz, Abbas Abdolmaleki, Oliver Groth, Jean-Baptiste Regli, Oleg Sushkov, Tom Rothörl, José Enrique Chen, Yusuf Aytar, Dave Barker, Joy Ortiz, Martin Riedmiller, Jost Tobias Springenberg, Raia Hadsell, Francesco Nori, and Nicolas Heess.

Robocat: A self-improving foundation agent for robotic manipulation, 2023.

-
[93]

Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard.

Visual language maps for robot navigation.

In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023.

-
[94]

Nur Muhammad (Mahi) Shafiullah, Chris Paxton, Lerrel Pinto1 Soumith Chintala, and Arthur Szlam.

Clip-fields: Weakly supervised semantic fields for robotic memory.

In RSS, 2023.

-
[95]

Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer, Soroush Saryazd, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba.

Conceptfusion: Open-set multimodal 3d mapping.

In arXiv:2302.07241, 2023.

-
[96]

Dhruv Shah, Blazej Osinski, Brian Ichter, and Sergey Levine.

Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action.

In CoRL, 2022.

-
[97]

Mohit Shridhar, Lucas Manuelli, and Dieter Fox.

Cliport: What and where pathways for robotic manipulation.

In CoRL, 2021.

-
[98]

William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, and Phillip Isola.

Distilled feature fields enable few-shot language-guided manipulation.

CoRL, 2023.

-
[99]

Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, and Xiaolong Wang.

Multi-task real robot learning with generalizable neural feature fields.

CoRL, 2023.

-
[100]

Sriram Yenamandra, Arun Ramachandran, Mukul Khanna, Karmesh Yadav, Devendra Singh Chaplot, Gunjan Chhablani, Alexander Clegg, Theophile Gervet, Vidhi Jain, Ruslan Partsey, Ram Ramrakhya, Andrew Szot, Tsung-Yen Yang, Aaron Edsinger, Charlie Kemp, Binit Shah, Zsolt Kira, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton.

The homerobot open vocab mobile manipulation challenge.

In Thirty-seventh Conference on Neural Information Processing Systems: Competition Track, 2023.

-
[101]

Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl.

Language-driven semantic segmentation.

In ICLR, 2022.

-
[102]

Ri-Zhao Qiu*, Yafei Hu*, Ge Yang, Yuchen Song, Yang Fu, Jianglong Ye, Jiteng Mu, Ruihan Yang, Nikolay Atanasov, Sebastian Scherer, and Xiaolong Wang.

Learning generalizable feature fields for mobile manipulation, 2024.

-
[103]

Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, and Katerina Fragkiadaki.

Act3d: 3d feature field transformers for multi-task robotic manipulation, 2023.

-
[104]

Christina Kassab, Matias Mattamala, Lintong Zhang, and Maurice Fallon.

Language-extended indoor slam (lexis): A versatile system for real-time visual scene understanding.

arXiv preprint arXiv:2309.15065, 2023.

-
[105]

Reihaneh Mirjalili, Michael Krawez, and Wolfram Burgard.

Fm-loc: Using foundation models for improved vision-based localization.

arXiv:2304.07058, 2023.

-
[106]

Nikhil Keetha, Avneesh Mishra, Jay Karhade, Krishna Murthy Jatavallabhula, Sebastian Scherer, Madhava Krishna, and Sourav Garg.

Anyloc: Towards universal visual place recognition.

RA-L, 2023.

-
[107]

Yao He, Ivan Cisneros, Nikhil Keetha, Jay Patrikar, Zelin Ye, Ian Higgins, Yaoyu Hu, Parv Kapoor, and Sebastian Scherer.

Foundloc: Vision-based onboard aerial localization in the wild, 2023.

-
[108]

Jeannette Bohg, Karol Hausman, Bharath Sankaran, Oliver Brock, Danica Kragic, Stefan Schaal, and Gaurav S. Sukhatme.

Interactive perception: Leveraging action in perception and perception in action.

IEEE Transactions on Robotics (T-RO), 2017.

-
[109]

Jivko Sinapov, Connor Schenck, Kerrick Staley, Vladimir Sukhoy, and Alexander Stoytchev.

Grounding semantic categories in behavioral interactions: Experiments with 100 objects.

Robotics and Autonomous Systems, 62(5):632–645, may 2014.

-
[110]

Jivko Sinapov, Connor Schenck, and Alexander Stoytchev.

Learning relational object categories using behavioral exploration and multimodal perception.

In International Conference on Robotics and Automation (ICRA), pages 5691–5698, Hong Kong, China, may 2014. IEEE.

-
[111]

Mevlana C. Gemici and Ashutosh Saxena.

Learning haptic representation for manipulating deformable food objects.

In Intelligent Robots and Systems (IROS), pages 638–645, Chicago, IL, USA, Sep 2014. IEEE.

-
[112]

Gyan Tatiya, Ramtin Hosseini, Michael C. Hughes, and Jivko Sinapov.

Sensorimotor cross-behavior knowledge transfer for grounded category recognition.

In International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). IEEE, 2019.

-
[113]

Gyan Tatiya, Ramtin Hosseini, Michael Hughes, and Jivko Sinapov.

A framework for sensorimotor cross-perception and cross-behavior knowledge transfer for object categorization.

Frontiers in Robotics and AI, 7:137, 2020.

-
[114]

Gyan Tatiya, Yash Shukla, Michael Edegware, and Jivko Sinapov.

Haptic knowledge transfer between heterogeneous robots using kernel manifold alignment.

In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.

-
[115]

Gyan Tatiya, Jonathan Francis, Ho-Hsiang Wu, Yonatan Bisk, and Jivko Sinapov.

Mosaic: Learning unified multi-sensory object property representations for robot learning via interactive perception.

In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 15381–15387. IEEE, 2024.

-
[116]

Xiaolin Fang, Leslie Pack Kaelbling, and Tomás Lozano-Pérez.

Embodied Uncertainty-Aware Object Segmentation.

In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

-
[117]

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel.

Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674–3683, 2018.

-
[118]

Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al.

Video language planning.

arXiv preprint arXiv:2310.10625, 2023.

-
[119]

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch.

Language models as zero-shot planners: Extracting actionable knowledge for embodied agents, 2022.

-
[120]

Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg.

Text2motion: From natural language instructions to feasible plans.

arXiv preprint arXiv:2303.12153, 2023.

-
[121]

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg.

ProgPrompt: Generating situated robot task plans using large language models, 2022.

-
[122]

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng.

Code as policies: Language model programs for embodied control.

ArXiv, abs/2209.07753, 2023.

-
[123]

Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, and Xiaolong Wang.

Gensim: Generating robotic simulation tasks via large language models.

In CoRL, 2023.

-
[124]

Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei.

Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation.

arXiv preprint arXiv:2409.01652, 2024.

-
[125]

J. Seipp, Á. Torralba, and J. Hoffmann.

Pddl generators, 2022.

-
[126]

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf.

Sayplan: Grounding large language models using 3d scene graphs for scalable task planning.

In 7th Annual Conference on Robot Learning, 2023.

-
[127]

Quanting Xie, Tianyi Zhang, Kedi Xu, Matthew Johnson-Roberson, and Yonatan Bisk.

Reasoning about the unseen for efficient outdoor object navigation, 2023.

-
[128]

Junting Chen, Guohao Li, Suryansh Kumar, Bernard Ghanem, and Fisher Yu.

How to not train your dragon: Training-free embodied object goal navigation with semantic frontiers.

arXiv preprint arXiv:2305.16925, 2023.

-
[129]

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei.

Voxposer: Composable 3d value maps for robotic manipulation with language models.

arXiv preprint arXiv:2307.05973, 2023.

-
[130]

Yen-Jen Wang, Bike Zhang, Jianyu Chen, and Koushil Sreenath.

Prompt a robot to walk with large language models, 2023.

-
[131]

Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, and Jacob Andreas.

Guiding pretraining in reinforcement learning with large language models, 2023.

-
[132]

Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh.

Reward design with language models, 2023.

-
[133]

Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, and Tao Yu.

Text2reward: Automated dense reward function generation for reinforcement learning, 2023.

-
[134]

Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montse Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, and Fei Xia.

Language to rewards for robotic skill synthesis.

Arxiv preprint arXiv:2306.08647, 2023.

-
[135]

Angelo Cangelosi, Giorgio Metta, Gerhard Sagerer, Stefano Nolfi, Chrystopher Nehaniv, Kerstin Fischer, Jun Tani, Tony Belpaeme, Giulio Sandini, Francesco Nori, and et al.

Integration of action and language knowledge: A roadmap for developmental robotics.

IEEE Transactions on Autonomous Mental Development, 2(3):167–195, 2010.

-
[136]

Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang.

Pengi: An audio language model for audio tasks, 2023.

-
[137]

Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, and Cewu Lu.

Anygrasp: Robust and efficient grasp perception in spatial and temporal domains.

IEEE Transactions on Robotics, 2023.

-
[138]

Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer.

Grounding large language models in interactive environments with online reinforcement learning, 2023.

-
[139]

Michael Ahn, Debidatta Dwibedi, Chelsea Finn, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Karol Hausman, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Sean Kirmani, Isabel Leal, Edward Lee, Sergey Levine, Yao Lu, Isabel Leal, Sharath Maddineni, Kanishka Rao, Dorsa Sadigh, Pannag Sanketi, Pierre Sermanet, Quan Vuong, Stefan Welker, Fei Xia, Ted Xiao, Peng Xu, Steve Xu, and Zhuo Xu.

Autort: Embodied foundation models for large scale orchestration of robotic agents, 2024.

-
[140]

Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, AJ Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Peter Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut.

Pali-x: On scaling up a multilingual vision and language model, 2023.

-
[141]

Debidatta Dwibedi, Vidhi Jain, Jonathan Tompson, Andrew Zisserman, and Yusuf Aytar.

Flexcap: Generating rich, localized, and flexible captions in images, 2024.

-
[142]

Huy Ha, Pete Florence, and Shuran Song.

Scaling up and distilling down: Language-guided robot skill acquisition.

In Proceedings of the 2023 Conference on Robot Learning, 2023.

-
[143]

Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan.

Robogen: Towards unleashing infinite data for automated robot learning via generative simulation, 2023.

-
[144]

Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Jodilyn Peralta Dee M, Brian Ichter, Karol Hausman, and Fei Xia.

Scaling robot learning with semantically imagined experience.

In arXiv:2302.11550, 2023.

-
[145]

Jiayuan Gu, Sean Kirmani, Paul Wohlhart, Yao Lu, Montserrat Gonzalez Arenas, Kanishka Rao, Wenhao Yu, Chuyuan Fu, Keerthana Gopalakrishnan, Zhuo Xu, Priya Sundaresan, Peng Xu, Hao Su, Karol Hausman, Chelsea Finn, Quan Vuong, and Ted Xiao.

Rt-trajectory: Robotic task generalization via hindsight trajectory sketches, 2023.

-
[146]

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine.

Zero-shot robotic manipulation with pretrained image-editing diffusion models, 2023.

-
[147]

Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel.

Learning universal policies via text-guided video generation.

In NeurIPS, 2023.

-
[148]

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou.

Chain-of-thought prompting elicits reasoning in large language models, 2023.

-
[149]

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, and Yejin Choi.

Faith and fate: Limits of transformers on compositionality, 2023.

-
[150]

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler.

Graph of thoughts: Solving elaborate problems with large language models, 2023.

-
[151]

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan.

Tree of thoughts: Deliberate problem solving with large language models, 2023.

-
[152]

Shun Zhang, Zhenfang Chen, Yikang Shen, Mingyu Ding, Joshua B. Tenenbaum, and Chuang Gan.

Planning with large language models for code generation.

In The Eleventh International Conference on Learning Representations, 2023.

-
[153]

Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone.

Llm+p: Empowering large language models with optimal planning proficiency, 2023.

-
[154]

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter.

Inner monologue: Embodied reasoning through planning with language models, 2022.

-
[155]

Nur Muhammad Mahi Shafiullah, Zichen Jeff Cui, Ariuntuya Altanzaya, and Lerrel Pinto.

Behavior transformers: Cloning $k$ modes with one stone, 2022.

-
[156]

Vidhi Jain, Maria Attarian, Nikhil J Joshi, Ayzaan Wahid, Danny Driess, Quan Vuong, Pannag R Sanketi, Pierre Sermanet, Stefan Welker, Christine Chan, Igor Gilitschenski, Yonatan Bisk, and Debidatta Dwibedi.

Vid2robot: End-to-end video-conditioned policy learning with cross-attention transformers, 2024.

-
[157]

Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Brianna Zitkovich, Fei Xia, Chelsea Finn, et al.

Open-world object manipulation using pre-trained vision-language models.

arXiv preprint arXiv:2303.00905, 2023.

-
[158]

Yan Duan, Marcin Andrychowicz, Bradly Stadie, OpenAI Jonathan Ho, Jonas Schneider, Ilya Sutskever, Pieter Abbeel, and Wojciech Zaremba.

One-shot imitation learning.

Advances in neural information processing systems, 30, 2017.

-
[159]

Chelsea Finn, Tianhe Yu, Tianhao Zhang, Pieter Abbeel, and Sergey Levine.

One-shot visual imitation learning via meta-learning.

In Conference on robot learning, pages 357–368. PMLR, 2017.

-
[160]

Sudeep Dasari and Abhinav Gupta.

Transformers for one-shot visual imitation.

In Conference on Robot Learning, pages 2071–2084. PMLR, 2021.

-
[161]

Zhao Mandi, Fangchen Liu, Kimin Lee, and Pieter Abbeel.

Towards more generalizable one-shot visual imitation learning.

In 2022 International Conference on Robotics and Automation (ICRA), pages 2434–2444. IEEE, 2022.

-
[162]

Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and Pierre Sermanet.

Learning latent plans from play.

In Conference on robot learning, pages 1113–1132. PMLR, 2020.

-
[163]

Stephen James, Michael Bloesch, and Andrew J Davison.

Task-embedded control networks for few-shot imitation learning.

In Conference on robot learning, pages 783–795. PMLR, 2018.

-
[164]

Allan Zhou, Eric Jang, Daniel Kappler, Alex Herzog, Mohi Khansari, Paul Wohlhart, Yunfei Bai, Mrinal Kalakrishnan, Sergey Levine, and Chelsea Finn.

Watch, try, learn: Meta-learning from demonstrations and reward.

arXiv preprint arXiv:1906.03352, 2019.

-
[165]

Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar.

Mimicplay: Long-horizon imitation learning by watching human play.

arXiv preprint arXiv:2302.12422, 2023.

-
[166]

Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, and Sergey Levine.

One-shot imitation from observing humans via domain-adaptive meta-learning.

arXiv preprint arXiv:1802.01557, 2018.

-
[167]

Alessandro Bonardi, Stephen James, and Andrew J Davison.

Learning one-shot imitation from humans without humans.

IEEE Robotics and Automation Letters, 5(2):3533–3539, 2020.

-
[168]

Corey Lynch and Pierre Sermanet.

Language conditioned imitation learning over unstructured data.

arXiv preprint arXiv:2005.07648, 2020.

-
[169]

Simon Stepputtis, Joseph Campbell, Mariano Phielipp, Stefan Lee, Chitta Baral, and Heni Ben Amor.

Language-conditioned imitation learning for robot manipulation tasks.

Advances in Neural Information Processing Systems, 33:13139–13150, 2020.

-
[170]

Rouhollah Rahmatizadeh, Pooya Abolghasemi, Ladislau Bölöni, and Sergey Levine.

Vision-based multi-task manipulation for inexpensive robots using end-to-end learning from demonstration.

In 2018 IEEE international conference on robotics and automation (ICRA), pages 3758–3765. IEEE, 2018.

-
[171]

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan.

VIMA: general robot manipulation with multimodal prompts.

ArXiv, abs/2210.03094, 2022.

-
[172]

Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong, and Pete Florence.

Interactive language: Talking to robots in real time.

arXiv preprint arXiv:2210.06407, 2022.

-
[173]

Kuang-Huei Lee, Ted Xiao, Adrian Li, Paul Wohlhart, Ian Fischer, and Yao Lu.

Pi-qt-opt: Predictive information improves multi-task robotic reinforcement learning at scale.

CoRL, 2022.

-
[174]

Alexander Herzog, Kanishka Rao, Karol Hausman, Yao Lu, Paul Wohlhart, Mengyuan Yan, Jessica Lin, Montserrat Gonzalez Arenas, Ted Xiao, Daniel Kappler, Daniel Ho, Jarek Rettinghouse, Yevgen Chebotar, Kuang-Huei Lee, Keerthana Gopalakrishnan, Ryan Julian, Adrian Li, Chuyuan Kelly Fu, Bob Wei, Sangeetha Ramesh, Khem Holden, Kim Kleiven, David Rendleman, Sean Kirmani, Jeff Bingham, Jon Weisz, Ying Xu, Wenlong Lu, Matthew Bennice, Cody Fong, David Do, Jessica Lam, Yunfei Bai, Benjie Holson, Michael Quinlan, Noah Brown, Mrinal Kalakrishnan, Julian Ibarz, Peter Pastor, and Sergey Levine.

Deep rl at scale: Sorting waste in office buildings with a fleet of mobile manipulators.

In RSS, 2023.

-
[175]

Yevgen Chebotar, Quan Vuong, Alex Irpan, Karol Hausman, Fei Xia, Yao Lu, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, Keerthana Gopalakrishnan, Julian Ibarz, Ofir Nachum, Sumedh Sontakke, Grecia Salazar, Huong T Tran, Jodilyn Peralta, Clayton Tan, Deeksha Manjunath, Jaspiar Singht, Brianna Zitkovich, Tomas Jackson, Kanishka Rao, Chelsea Finn, and Sergey Levine.

Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions.

In CoRL, 2023.

-
[176]

Aviral Kumar, Anikait Singh, Frederik Ebert, Mitsuhiko Nakamoto, Yanlai Yang, Chelsea Finn, and Sergey Levine.

Pre-training for robots: Offline rl enables learning new tasks from a handful of trials.

In RSS, 2023.

-
[177]

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine.

Conservative q-learning for offline reinforcement learning.

In NeurIPS, 2020.

-
[178]

Wenlong Huang, Igor Mordatch, Pieter Abbeel, and Deepak Pathak.

Generalization in dexterous manipulation via geometry-aware multi-task learning.

arXiv preprint arXiv:2111.03062, 2021.

-
[179]

Tao Chen, Jie Xu, and Pulkit Agrawal.

A system for general in-hand object re-orientation.

In Conference on Robot Learning, pages 297–307. PMLR, 2022.

-
[180]

Tao Chen, Megha Tippur, Siyang Wu, Vikash Kumar, Edward Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand reorientation of novel and complex object shapes.

Science Robotics, 8(84):eadc9244, 2023.

-
[181]

Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta.

The unsurprising effectiveness of pre-trained vision models for control, 2022.

-
[182]

Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, et al.

Pre-trained language models for interactive decision-making.

Advances in Neural Information Processing Systems, 35:31199–31212, 2022.

-
[183]

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta.

R3m: A universal visual representation for robot manipulation, 2022.

-
[184]

Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell.

Real-world robot learning with masked visual pre-training.

In Conference on Robot Learning, pages 416–426. PMLR, 2023.

-
[185]

Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, and Jitendra Malik.

Robot learning with sensorimotor pre-training, 2023.

-
[186]

Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang.

On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline.

In ICML, 2023.

-
[187]

Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier.

Where are we in the search for an artificial visual cortex for embodied intelligence?, 2023.

-
[188]

Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak.

Affordances from human videos as a versatile representation for robotics.

CVPR, 2023.

-
[189]

Abitha Thankaraj and Lerrel Pinto.

That sounds right: Auditory self-supervision for dynamic robot manipulation, 2022.

-
[190]

Irmak Guzey, Ben Evans, Soumith Chintala, and Lerrel Pinto.

Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play, 2023.

-
[191]

Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, and Sergey Levine.

Gnm: A general navigation model to drive any robot.

In ICRA, 2023.

-
[192]

Joanne Truong, April Zitkovich, Sonia Chernova, Dhruv Batra, Tingnan Zhang, Jie Tan, and Wenhao Yu.

Indoorsim-to-outdoorreal: Learning to navigate outdoors without any outdoor experience.

In arXiv:2305.01098, 2023.

-
[193]

Rogerio Bonatti, Sai Vemprala, Shuang Ma, Felipe Frujeri, Shuhang Chen, and Ashish Kapoor.

Pact: Perception-action causal transformer for autoregressive robotics pre-training.

In arXiv:2209.11133, 2022.

-
[194]

Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al.

Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning.

arXiv preprint arXiv:2309.16650, 2023.

-
[195]

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu.

Reasoning with language model is planning with world model.

arXiv preprint arXiv:2305.14992, 2023.

-
[196]

Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel.

Learning interactive real-world simulators.

arXiv preprint arXiv:2310.06114, 2023.

-
[197]

Siddharth Karamcheti, Megha Srivastava, Percy Liang, and Dorsa Sadigh.

Lila: Language-informed latent actions.

In Conference on Robot Learning, pages 1379–1390. PMLR, 2022.

-
[198]

Ria Doshi, Homer Walke, Oier Mees, Sudeep Dasari, and Sergey Levine.

Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation.

arXiv preprint arXiv:2408.11812, 2024.

-
[199]

Shikhar Bahl, Abhinav Gupta, and Deepak Pathak.

Human-to-robot imitation in the wild.

In RSS, 2022.

-
[200]

Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, and Akshara Rai.

Transformers are adaptable task planners.

In 6th Annual Conference on Robot Learning, 2022.

-
[201]

Jacob Andreas, Dan Klein, and Sergey Levine.

Modular multitask reinforcement learning with policy sketches.

In International conference on machine learning, pages 166–175. PMLR, 2017.

-
[202]

Marjorie Skubic, Derek Anderson, Samuel Blisard, Dennis Perzanowski, and Alan Schultz.

Using a hand-drawn sketch to control a team of robots.

Autonomous Robots, 22:399–410, 2007.

-
[203]

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.

Chain-of-thought prompting elicits reasoning in large language models.

Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

-
[204]

Jianzong Wu, Xiangtai Li, Shilin Xu Haobo Yuan, Henghui Ding, Yibo Yang, Xia Li, Jiangning Zhang, Yunhai Tong, Xudong Jiang, Bernard Ghanem, et al.

Towards open vocabulary learning: A survey.

arXiv preprint arXiv:2306.15880, 2023.

-
[205]

Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao.

Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges.

arXiv preprint arXiv:2311.03287, 2023.

-
[206]

Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, and Dorsa Sadigh.

Physically grounded vision-language models for robotic manipulation.

In arxiv, 2023.

-
[207]

Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn.

Robonet: Large-scale multi-robot learning, 2020.

-
[208]

Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine.

Bridge data: Boosting generalization of robotic skills with cross-domain datasets, 2021.

-
[209]

Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, et al.

Bridgedata v2: A dataset for robot learning at scale.

arXiv preprint arXiv:2308.12952, 2023.

-
[210]

Kenneth Shaw, Ananye Agarwal, and Deepak Pathak.

Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning.

arXiv preprint arXiv:2309.06440, 2023.

-
[211]

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang.

Matterport3d: Learning from rgb-d data in indoor environments.

arXiv preprint arXiv:1709.06158, 2017.

-
[212]

Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese.

Gibson env: Real-world perception for embodied agents.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9068–9079, 2018.

-
[213]

Manolis Savva*, Abhishek Kadian*, Oleksandr Maksymets*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra.

Habitat: A Platform for Embodied AI Research.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

-
[214]

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al.

Ai2-thor: An interactive 3d environment for visual ai.

arXiv preprint arXiv:1712.05474, 2017.

-
[215]

So Yeon Min, Devendra Singh Chaplot, Pradeep Ravikumar, Yonatan Bisk, and Ruslan Salakhutdinov.

Film: Following instructions in language with modular methods, 2022.

-
[216]

Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor.

Airsim: High-fidelity visual and physical simulation for autonomous vehicles, 2017.

-
[217]

Google DeepMind.

Mujoco 3.0.

[https://github.com/google-deepmind/mujoco/releases/tag/3.0.0](https://github.com/google-deepmind/mujoco/releases/tag/3.0.0), 2023.

Accessed: [Insert date of access].

-
[218]

Matthew Chignoli, Donghyun Kim, Elijah Stanger-Jones, and Sangbae Kim.

The mit humanoid robot: Design, motion planning, and control for acrobatic behaviors.

In 2020 IEEE-RAS 20th International Conference on Humanoid Robots (Humanoids), pages 1–8. IEEE, 2021.

-
[219]

Weiye Zhao, Tairan He, and Changliu Liu.

Model-free safe control for zero-violation reinforcement learning.

In 5th Annual Conference on Robot Learning, 2021.

-
[220]

Sylvia Herbert, Jason J. Choi, Suvansh Sanjeev, Marsalis Gibson, Koushil Sreenath, and Claire J. Tomlin.

Scalable learning of safety guarantees for autonomous systems using hamilton-jacobi reachability, 2021.

-
[221]

Ran Tian, Liting Sun, Andrea Bajcsy, Masayoshi Tomizuka, and Anca D Dragan.

Safety assurances for human-robot interaction via confidence-aware game-theoretic human models.

In 2022 International Conference on Robotics and Automation (ICRA), pages 11229–11235. IEEE, 2022.

-
[222]

S.H. Cheong, J.H. Lee, and C.H. Kim.

A new concept of safety affordance map for robots object manipulation.

In 2018 27th IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN), pages 565–570, 2018.

-
[223]

Ziyi Yang, Shreyas Sundara Raman, Ankit Shah, and Stefanie Tellex.

Plug in the safety chip: Enforcing constraints for LLM-driven robot agents.

In 2nd Workshop on Language and Robot Learning: Language as Grounding, 2023.

-
[224]

Olaf Sporns and Richard F Betzel.

Modular brain networks.

Annual review of psychology, 67:613–640, 2016.

-
[225]

David Meunier, Renaud Lambiotte, and Edward T Bullmore.

Modular and hierarchically modular organization of brain networks.

Frontiers in neuroscience, 4:200, 2010.

-
[226]

Vincent Vanhoucke.

The end-to-end false dichotomy: Roboticists arguing lego vs. playmo.

Medium, October 28 2018.

-
[227]

Yann LeCun.

A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27.

Open Review, 62, 2022.

-
[228]

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.

Video generation models as world simulators.

2024.

-
[229]

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel.

Genie: Generative interactive environments, 2024.

-
[230]

Shadow Robot Company.

Dexterous hand series.

[https://www.shadowrobot.com/dexterous-hand-series/](https://www.shadowrobot.com/dexterous-hand-series/), 2023.

Accessed: 2023-12-10.

-
[231]

Siyuan Dong, Wenzhen Yuan, and Edward H. Adelson.

Improved gelsight tactile sensor for measuring geometry and slip.

In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, September 2017.

-
[232]

Zilin Si, Tianhong Catherine Yu, Katrene Morozov, James McCann, and Wenzhen Yuan.

Robotsweater: Scalable, generalizable, and customizable machine-knitted tactile skins for robots.

arXiv preprint arXiv:2303.02858, 2023.

-
[233]

Branden Romero, Hao-Shu Fang, Pulkit Agrawal, and Edward Adelson.

Eyesight hand: Design of a fully-actuated dexterous robot hand with integrated vision-based tactile sensors and compliant actuation.

arXiv preprint arXiv:2408.06265, 2024.

-
[234]

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn.

Learning fine-grained bimanual manipulation with low-cost hardware, 2023.

-
[235]

Hongjie Fang, Hao-Shu Fang, Yiming Wang, Jieji Ren, Jingjing Chen, Ruo Zhang, Weiming Wang, and Cewu Lu.

Airexo: Low-cost exoskeletons for learning whole-arm manipulation in the wild.

In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 15031–15038. IEEE, 2024.

-
[236]

Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui.

Realtime qa: What’s the answer right now?, 2022.

-
[237]

Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler.

Dsi++: Updating transformer memory with new documents.

ArXiv, abs/2212.09744, 2022.

-
[238]

Sanket Vaibhav Mehta, Darshan Patil, Sarath Chandar, and Emma Strubell.

An empirical investigation of the role of pre-training in lifelong learning.

Journal of Machine Learning Research, 24(214):1–50, 2023.

-
[239]

James Seale Smith, Paola Cascante-Bonilla, Assaf Arbelle, Donghyun Kim, Rameswar Panda, David Cox, Diyi Yang, Zsolt Kira, Rogerio Feris, and Leonid Karlinsky.

Construct-vl: Data-free continual structured vl concepts learning.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14994–15004, 2023.

-
[240]

Timothée Lesort, Vincenzo Lomonaco, Andrei Stoian, Davide Maltoni, David Filliat, and Natalia Díaz-Rodríguez.

Continual learning for robotics: Definition, framework, learning strategies, opportunities and challenges.

Information Fusion, 58:52–68, June 2020.

-
[241]

Guilherme Maeda, Marco Ewerton, Takayuki Osa, Baptiste Busch, and Jan Peters.

Active incremental learning of robot movement primitives.

In Sergey Levine, Vincent Vanhoucke, and Ken Goldberg, editors, Proceedings of the 1st Annual Conference on Robot Learning, volume 78 of Proceedings of Machine Learning Research, pages 37–46. PMLR, 13–15 Nov 2017.

-
[242]

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik.

Rma: Rapid motor adaptation for legged robots.

In Robotics: Science and Systems, 2021.

-
[243]

Joey Hejna and Dorsa Sadigh.

Inverse preference learning: Preference-based rl without a reward function, 2023.

-
[244]

Zihao Li, Zhuoran Yang, and Mengdi Wang.

Reinforcement learning with human feedback: Learning dynamic choices via pessimism, 2023.

-
[245]

Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang.

Fine-tuning can distort pretrained features and underperform out-of-distribution.

In International Conference on Learning Representations, 2022.

-
[246]

OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, Jonas Schneider, Nikolas Tezak, Jerry Tworek, Peter Welinder, Lilian Weng, Qiming Yuan, Wojciech Zaremba, and Lei Zhang.

Solving rubik’s cube with a robot hand, 2019.

-
[247]

Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, and Willie Neiswanger.

Sample efficient reinforcement learning from human feedback via active exploration, 2023.

-
[248]

Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, and Hyun Oh Song.

Direct preference-based policy optimization without reward modeling, 2023.

-
[249]

Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal.

Compositional foundation models for hierarchical planning, 2023.

-
[250]

Kevin Frans, Jonathan Ho, Xi Chen, Pieter Abbeel, and John Schulman.

Meta learning shared hierarchies, 2017.

-
[251]

Suraj Nair and Chelsea Finn.

Hierarchical foresight: Self-supervised learning of long-horizon tasks via visual subgoal generation.

In International Conference on Learning Representations, 2020.

-
[252]

Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.

Slowfast networks for video recognition, 2019.

-
[253]

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen,
Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn.

Droid: A large-scale in-the-wild robot manipulation dataset.

2024.

-
[254]

Murtaza Dalal, Ajay Mandlekar, Caelan Garrett, Ankur Handa, Ruslan Salakhutdinov, and Dieter Fox.

Imitating task and motion planning with visuomotor transformers.

2023.

-
[255]

Xuxin Cheng, Kexin Shi, Ananye Agarwal, and Deepak Pathak.

Extreme parkour with legged robots.

In arXiv:2309.14341, 2023.

-
[256]

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield.

Foundationpose: Unified 6d pose estimation and tracking of novel objects, 2024.

-
[257]

Noah Hollmann, Samuel Müller, and Frank Hutter.

Large language models for automated data science: Introducing caafe for context-aware automated feature engineering.

Advances in Neural Information Processing Systems, 36, 2024.

-
[258]

Zeyi Liu, Arpit Bahety, and Shuran Song.

Reflect: Summarizing robot experiences for failure explanation and correction.

In Conference on Robot Learning, pages 3468–3484. PMLR, 2023.

-
[259]

L Minh Dang, Kyungbok Min, Hanxiang Wang, Md Jalil Piran, Cheol Hee Lee, and Hyeonjoon Moon.

Sensor-based and vision-based human activity recognition: A comprehensive survey.

Pattern Recognition, 108:107561, 2020.

-
[260]

Yara Rizk, Praveen Venkateswaran, Vatche Isahagian, Austin Narcomey, and Vinod Muthusamy.

A case for business process-specific foundation models.

In International Conference on Business Process Management, pages 44–56. Springer, 2023.

-
[261]

David M Goldberg.

Characterizing accident narratives with word embeddings: Improving accuracy, richness, and generalizability.

Journal of safety research, 80:441–455, 2022.

-
[262]

Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, et al.

Conceptfusion: Open-set multimodal 3d mapping.

RSS, 2023.

-
[263]

Kashu Yamazaki, Taisei Hanyu, Khoa Vo, Thang Pham, Minh Tran, Gianfranco Doretto, Anh Nguyen, and Ngan Le.

Open-fusion: Real-time open-vocabulary 3d mapping and queryable scene representation.

In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 9411–9417. IEEE, 2024.

-
[264]

Sandra Robla-Gómez, Victor M Becerra, José Ramón Llata, Esther Gonzalez-Sarabia, Carlos Torre-Ferrero, and Juan Perez-Oria.

Working together: A review on safe human-robot collaboration in industrial environments.

Ieee Access, 5:26754–26773, 2017.

-
[265]

Joshua A Marshall, Adrian Bonchis, Eduardo Nebot, and Steven Scheding.

Robotics in mining.

Springer handbook of robotics, pages 1549–1576, 2016.

-
[266]

Lili Chen, Shikhar Bahl, and Deepak Pathak.

Playfusion: Skill acquisition via diffusion from language-annotated play.

In Conference on Robot Learning, pages 2012–2029. PMLR, 2023.

-
[267]

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio.

Quantized neural networks: Training neural networks with low precision weights and activations.

Journal of Machine Learning Research, 18(187):1–30, 2018.

-
[268]

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.

Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531, 2015.

-
[269]

Shayegan Omidshafiei, Dong-Ki Kim, Miao Liu, Gerald Tesauro, Matthew Riemer, Christopher Amato, Murray Campbell, and Jonathan P. How.

Learning to teach in cooperative multiagent reinforcement learning.

Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):6128–6136, Jul. 2019.

-
[270]

Lin Wang and Kuk-Jin Yoon.

Knowledge distillation and student-teacher learning for visual intelligence: A review and new outlooks.

IEEE transactions on pattern analysis and machine intelligence, 44(6):3048–3068, 2021.

-
[271]

S. Dass, J. Yapeter, J. Zhang, J. Zhang, K. Pertsch, S. Nikolaidis, and J. J. Lim.

Clvr jaco play dataset.

[https://github.com/clvrai/clvr-jaco-play-dataset](https://github.com/clvrai/clvr-jaco-play-dataset), 2023.

-
[272]

Jianlan Luo, Charles Xu, Xinyang Geng, Gilbert Feng, Kuan Fang, Liam Tan, Stefan Schaal, and Sergey Levine.

Multi-stage cable routing through hierarchical imitation learning, 2023.

-
[273]

Jyothish Pari, Nur Muhammad Shafiullah, Sridhar Pandian Arunachalam, and Lerrel Pinto.

The surprising effectiveness of representation learning for visual imitation.

arXiv preprint arXiv:2112.01511, 2021.

-
[274]

L. Y. Chen, S. Adebola, and K. Goldberg.

Berkeley ur5 demonstration dataset.

[https://sites.google.com/view/berkeley-ur5/home](https://sites.google.com/view/berkeley-ur5/home).

Accessed: [Insert Date Here].

-
[275]

Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, and Wolfram Burgard.

Latent plans for task agnostic offline reinforcement learning.

In Proceedings of the 6th Conference on Robot Learning (CoRL), 2022.

-
[276]

Xufeng Zhao, Mengdi Li, Cornelius Weber, Muhammad Burhan Hafez, and Stefan Wermter.

Chat with the environment: Interactive multimodal perception using large language models, 2023.

-
[277]

Coppelia Robotics.

Coppeliasim.

[https://www.coppeliarobotics.com/](https://www.coppeliarobotics.com/).

Accessed: [Insert Date Here].

-
[278]

E. Coumans and Y. Bai.

Pybullet, a python module for physics simulation for games, robotics and machine learning.

-
[279]

Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li.

Instruct2act: Mapping multi-modality instructions to robotic actions with large language model, 2023.

-
[280]

Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, et al.

Sapien: A simulated part-based interactive environment.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11097–11107, 2020.

-
[281]

Kaichun Mo, Leonidas Guibas, Mustafa Mukadam, Abhinav Gupta, and Shubham Tulsiani.

Where2act: From pixels to actions for articulated 3d objects, 2021.

-
[282]

Alex X. Lee, Coline Devin, Yuxiang Zhou, Thomas Lampe, Konstantinos Bousmalis, Jost Tobias Springenberg, Arunkumar Byravan, Abbas Abdolmaleki, Nimrod Gileadi, David Khosid, Claudio Fantacci, Jose Enrique Chen, Akhil Raju, Rae Jeong, Michael Neunert, Antoine Laurens, Stefano Saliceti, Federico Casarini, Martin Riedmiller, Raia Hadsell, and Francesco Nori.

Beyond pick-and-place: Tackling robotic stacking of diverse shapes, 2021.

-
[283]

Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, and Alexander Toshev.

Large language models as generalizable policies for embodied tasks, 2023.

-
[284]

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.

Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374, 2021.

-
[285]

Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong, Ivan Krasin, Dan Duong, Vikas Sindhwani, and Johnny Lee.

Transporter networks: Rearranging the visual world for robotic manipulation.

Conference on Robot Learning (CoRL), 2020.

-
[286]

Yecheng Jason Ma, William Liang, Vaidehi Som, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman.

Liv: Language-image representations and rewards for robotic control, 2023.

-
[287]

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, and Sergey Levine.

Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning, 2021.

-
[288]

Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser.

Tidybot: Personalized robot assistance with large language models.

arXiv preprint arXiv:2305.05658, 2023.

-
[289]

Yan Ding, Xiaohan Zhang, Chris Paxton, and Shiqi Zhang.

Task and motion planning with large language models for object rearrangement.

arXiv preprint arXiv:2303.06247, 2023.

-
[290]

N. Koenig and A. Howard.

Design and use paradigms for gazebo, an open-source multi-robot simulator.

In 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (IEEE Cat. No.04CH37566), volume 3, pages 2149–2154 vol.3, 2004.

-
[291]

Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, Alexander William Clegg, and Devendra Singh Chaplot.

Habitat-matterport 3d semantics dataset, 2023.

-
[292]

Ilija Radosavovic, Bike Zhang, Baifeng Shi, Jathushan Rajasegaran, Sarthak Kamat, Trevor Darrell, Koushil Sreenath, and Jitendra Malik.

Humanoid locomotion as next token prediction.

arXiv:2402.19469, 2024.

-
[293]

Théophane Weber, Sébastien Racaniere, David P Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria Puigdomenech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, et al.

Imagination-augmented agents for deep reinforcement learning.

arXiv preprint arXiv:1707.06203, 2017.

-
[294]

Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio.

Babyai: A platform to study the sample efficiency of grounded language learning.

arXiv preprint arXiv:1810.08272, 2018.

-
[295]

Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman.

Leveraging procedural generation to benchmark reinforcement learning.

In International conference on machine learning, pages 2048–2056. PMLR, 2020.

-
[296]

Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.

The arcade learning environment: An evaluation platform for general agents.

Journal of Artificial Intelligence Research, 47:253–279, 2013.

-
[297]

Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al.

Deepmind control suite.

arXiv preprint arXiv:1801.00690, 2018.

-
[298]

Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter.

Grounded decoding: Guiding text generation with grounded models for robot control.

ArXiv, abs/2303.00855, 2023.

-
[299]

M. Chevalier-Boisvert, L. Willems, and S. Pal.

Minimalistic gridworld environment for gymnasium.

[https://github.com/pierg/environments-rl](https://github.com/pierg/environments-rl), 2018.

-
[300]

Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuan Jiang, Zongqing Lu, Stephen McAleer, Hao Dong, Song-Chun Zhu, and Yaodong Yang.

Towards human-level bimanual dexterous manipulation with reinforcement learning.

Advances in Neural Information Processing Systems, 35:5150–5163, 2022.

-
[301]

Taylor Howell, Nimrod Gileadi, Saran Tunyasuvunakool, Kevin Zakka, Tom Erez, and Yuval Tassa.

Predictive sampling: Real-time behaviour synthesis with mujoco, 2022.

Generated on Tue Oct 1 08:48:21 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)