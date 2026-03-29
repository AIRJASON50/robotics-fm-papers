[# Neural Scaling Laws in Robotics Sebastian Sartor TUM Munich, Germany Email: sebastian.sartor@tum.de Neil C. Thompson MIT Cambridge, MA, US Email: neil_t@mit.edu ###### Abstract Neural scaling laws have driven significant advancements in machine learning, particularly in domains like language modeling and computer vision. However, the exploration of neural scaling laws within robotics has remained relatively underexplored, despite the growing adoption of foundation models in this field. This paper represents the first comprehensive study to quantify neural scaling laws for Robot Foundation Models (RFMs) and Large Language Models (LLMs) in robotics tasks. Through a meta-analysis of 327 research papers, we investigate how data size, model size, and compute resources influence downstream performance across a diverse set of robotic tasks. Consistent with previous scaling law research, our results reveal that the performance of robotic models improves with increased resources, following a power-law relationship. Promisingly, the improvement in robotic task performance scales notably faster than language tasks. This suggests that, while performance on downstream robotic tasks today is often moderate-to-poor, increased data and compute are likely to signficantly improve performance in the future. Also consistent with previous scaling law research, we also observe the emergence of new robot capabilities as models scale. ## I Introduction Significant advancements in deep learning over recent years have been primarily driven by scaling—training larger neural networks on increasing amounts of data with ever-growing computational resources [3](#bib.bib3)]. The concept that scaling neural networks leads to improved performance dates back to research conducted as early as 2010 [[[14](#bib.bib14)]]. However, it was not until around 2017 that researchers like Jonathan Rosenfeld [[[60](#bib.bib60), [62](#bib.bib62), [61](#bib.bib61), [59](#bib.bib59)]], teams at Baidu [[[26](#bib.bib26)]], and later OpenAI [[[25](#bib.bib25), [34](#bib.bib34)]] formally articulated “neural scaling laws.” These laws describe how the performance (e.g. the loss of neural networks) systematically improves as a function of model size, data size, and compute resources. The concept of neural scaling laws aligns with the principles articulated in the Bitter Lesson of 2019, which highlights the critical role of scalable computation in achieving superior performance [[[71](#bib.bib71)]].

Neural scaling laws have been studied extensively in various domains, ranging from language modeling to vision and reinforcement learning, following a power-law function (e.g., [Kaplan et al. [[34](#bib.bib34)], Hoffmann et al. [[28](#bib.bib28)], Zhai et al. [[90](#bib.bib90)], Hilton et al. [[27](#bib.bib27)]]). They not only provide a framework for understanding how neural network architectures and data distributions impact performance, but have also proven to be beneficial for planning sample sizes, particularly in data-scarce domains, and have helped reduce the environmental footprint of experimentation [[[68](#bib.bib68), [48](#bib.bib48)]]. Overall, scaling laws facilitate the identification of optimal scaling coefficients, allow performance prediction based on given inputs, and enable estimating the required inputs for desired performance outcomes [[[3](#bib.bib3)]].

Inspired by the capabilities and high generalizability of foundation models in domains like language and vision, robotics researchers are exploring their application to the physical world, envisioning general-purpose robotics and a potential resolution to Moravec’s Paradox (e.g., [Brohan et al. [[9](#bib.bib9)], Reed et al. [[57](#bib.bib57)], Padalkar et al. [[51](#bib.bib51)]]).

Traditionally, roboticists developed specialized models tailored to specific applications, robots, and environments, often using multi-module system architectures. Recently, the field has shifted toward training large, generalist, pre-trained policies end-to-end. These Vision-Language-Action (VLA) models are designed for efficient adaptation across diverse robots, tasks, and environments [[[51](#bib.bib51), [32](#bib.bib32)]].

Another significant trend is integrating foundation models trained on internet-scale data—such as Vision-Language Models (VLMs) and Large Language Models (LLMs) into robotic tasks. This approach enhances robots’ ability to interpret natural language commands and visually understand tasks [[[51](#bib.bib51)]]. By bridging high-level reasoning with low-level control, these models enable robots to generalize across tasks and environments, demonstrating semantic reasoning capabilities such as understanding and executing commands like “stack the red block on top of the green cylinder.”

Despite these advances, neural scaling laws in robotics remain mainly unexplored. Previous research hints that scaling principles may hold, but no comprehensive quantification has been conducted (e.g., [Padalkar et al. [[51](#bib.bib51)], Brohan et al. [[9](#bib.bib9)]]). Another weakness of previous scaling law work is that it focuses on measures that do not directly translate into real-world task performance. This is particularly important because identifying and quantifying scaling laws in the context of robotics provides a crucial framework for developing general-purpose robotic systems. Scaling laws enable researchers to predict performance outcomes, allocate resources more efficiently, and ensure adaptability across tasks. By understanding these principles, we can streamline experimentation, reduce costs, and enhance the environmental sustainability of robotics research.

This study addresses this gap by identifying neural scaling laws for robotics. Specifically, we study 1) Do scaling laws observed in other domains like language and vision hold true for RFM and LLMs in robotics in terms of data size, model size and compute resources? 2) What are the characteristic power-law coefficients? and 3) Do RFM and LLMs in robotics exhibit emergent capabilities similar to those observed in other domains?

## II Background and related work

#### Robotics & Embodied AI

The field of robotics has struggled for some time with challenges related to generalizability and scalability, as developing machine intelligence for autonomous systems in the physical world is both costly and time-intensive. Existing solutions are typically designed for specific tasks, limiting their ability to generalize across a wide array of applications [[[80](#bib.bib80)]]. Building on the remarkable success of foundation models in domains like language and vision, researchers and entrepreneurs have begun exploring how to extend these advancements to the field of robotics. These models can be divided into two primary categories [[[30](#bib.bib30)]]. The first category includes Robot Foundation Models (RFMs), which are large, versatile models trained on diverse datasets and designed to generalize across various downstream robotic tasks. These robotics models often incorporate various learning strategies, such as imitation learning, behavior cloning, reinforcement learning, and diffusion policies. Many of them, especially those that handle large-scale data or multimodal inputs, are built upon architectures like the Transformer [[[79](#bib.bib79)]]. RFMs are being applied in robotics across various domains, including low- and high-level perception, planning, and data augmentation [[[35](#bib.bib35)]]. Drawing inspiration from the broader concept of foundation models in AI [[[6](#bib.bib6)]], RFMs can be further classified into three types according to [Kawaharazuka et al. [[35](#bib.bib35)]]: 1)
Pre-trained Visual Representations (PVRs): Models such as R3M [[[49](#bib.bib49)]] and RPT [[[53](#bib.bib53)]] focus on learning visual features that can be applied to multiple robotic tasks; 2) Vision-Language Models (VLMs): These models, like Palm-E [[[16](#bib.bib16)]], integrate visual and textual information to enhance robotic reasoning and understanding; and 3) Vision-Language-Action (VLA) Models: Also known as Robot Transformers or Large Behavior Models (LBMs), those models are designed for end-to-end learning of control policies and dynamics (e.g., RT-2, $\pi_{0}$, and Open-VLA) [[[5](#bib.bib5), [37](#bib.bib37), [9](#bib.bib9)]]. The second approach besides RFMs integrates LLMs into robotics, leveraging their semantic reasoning to bridge the gap between language and physical action [[[89](#bib.bib89), [41](#bib.bib41)]]. Overall, research on foundation models for robotics is a relatively recent research field and has demonstrated exponential growth in the past, as illustrated in Figure [1](#S2.F1)111Citation data is based on Google Scholar, using the keywords “robot learning” and “foundation models” (Dec. 2024). Beginning with just 4 papers in 2021, the number of publications surged to 2,172 by 2024, with 74% of these papers published in 2024 alone. This trend highlights the rapidly growing interest and progress in the domain of robotics research with foundation models.
While these strategies hold great promise, they are hindered by several significant challenges. The lack of internet-scale robotics datasets remains a major obstacle, as the high cost and complexity of data collection coupled with the vast diversity of robot types, environmental conditions, and task requirements make high-quality data generation an expensive and time-consuming process. While the volume of data is important, its diversity is equally crucial to ensure robust and adaptable model training [[[18](#bib.bib18)]]. Oftentimes data is collected through manual teleoperation in the real world which scales linearly with human time ($“sim-to-real gap”, where models trained in simulated environments struggle to perform effectively in the unpredictable and complex conditions of the real world, highlighting the intricate challenges posed by real-world robot-environment interactions. Besides data generation, evaluation and benchmarking are challenging given the cost, time and diversity of conditions to test the system’s capabilities [[[91](#bib.bib91), [19](#bib.bib19)]]. Lastly, important concepts from AI such as in-context learning and prompt engineering are still to be figured out for robotics.

#### Neural Scaling Laws

Neural scaling laws are empirical principles that describe the relationship between a model’s performance and factors such as its model size, the size of its training dataset, and its compute resources. These laws typically follow a power-law function, often with a cross-entropy objective, indicating that as model size, data size and compute increase, the model’s performance improves predictably. The general form of the power law described in [[[34](#bib.bib34)]] for data size is given by:

| | $$L(D)=\left(\frac{A}{D}\right)^{\alpha},$$ | |
|---|---|---|

for model size by:

| | $$L(N)=\left(\frac{B}{N}\right)^{\beta},\text{ and}$$ | |
|---|---|---|

for compute by:

| | $$L(C)=\left(\frac{F}{C}\right)^{\gamma},$$ | |
|---|---|---|

where $L(N)$, $L(D)$, and $L(C)$ represent the loss for model, data size, and compute, respectively, $N$ is the model size, $D$ is the data size, $C$ is the compute, and $A$, $B$, and $F$ are constants. The exponents $\alpha$, $\beta$, and $\gamma$ describe the scaling behavior with respect to model size, data size, and compute. The choice of performance metric depends on the task and modality, ranging from negative log-likelihood per token and perplexity to accuracy or mean squared error. Among the parameters in the scaling law, $\alpha$, $\beta$, and $\gamma$ play central roles.
It quantifies the efficiency of scaling, determining how rapidly performance improves as resources increase. Higher values of $\alpha$, $\beta$, and $\gamma$ indicate more efficient scaling, where larger improvements are achieved with less additional investment in size, data, or compute. Tables [I](#S2.T1) and [II](#S2.T2) provide an overview of typical $\alpha$, $\beta$, and $\gamma$ values observed across various modalities. Interestingly, the language domain stands out as significantly less efficient compared to other domains [[[34](#bib.bib34), [25](#bib.bib25)]].

*TABLE I: Power law exponent of neural scaling laws for language models, as detailed in [Kaplan et al. [[34](#bib.bib34)]]*

*TABLE II: Compute and Model Size scaling laws in other domains, as detailed in [Henighan et al. [[25](#bib.bib25)]]*

Notably, more scaling is not always better. For example, the Chinchilla language model demonstrated that optimal performance could be achieved with a more balanced allocation of data and compute resources, reducing inefficiencies in scaling [[[28](#bib.bib28)]]. They introduce a parametric fit that accounts for both model size and data size, given by the following expression:

| | $$L(D,N)=\left(\frac{A}{D}\right)^{\alpha_{D}}+\left(\frac{B}{N}\right)^{\beta_{% N}}+E,$$ | |
|---|---|---|

where $E$ is a constant.

Scaling laws have been extensively studied across various domains in machine learning, including LLMs, image and video generative modeling, and reinforcement learning [[[34](#bib.bib34), [25](#bib.bib25), [27](#bib.bib27)]]. Most research has focused on pretraining and fine-tuning phases, but recently, test-time compute (or inference scaling) has also emerged as an active area of investigation [[[67](#bib.bib67)]]. As models scale, both in size and in the volume of data they are trained on, they exhibit not only quantitative improvements but also the emergence of novel qualitative behaviors, often referred to as emergent capabilities [[[82](#bib.bib82), [63](#bib.bib63)]]. These emergent abilities are absent in smaller models or those trained on limited datasets, arising in a way that cannot be directly predicted from the performance of smaller configurations.

While scaling laws provide high predictability in upstream performance improvements, the downstream capabilities of scaled models often remain challenging to predict [[[22](#bib.bib22)]].

The field of scaling laws is growing rapidly, as illustrated in Figure [1](#S2.F1)222The citation count presented here is based on data from Google Scholar, which includes publications that cite [Kaplan et al. [[34](#bib.bib34)]] and contain the keyword “Scaling Law” (Dec. 2024). Since the release of the seminal paper on neural scaling laws by [Kaplan et al. [[34](#bib.bib34)]] in 2020, research in this area has gained significant momentum. Despite substantial progress, it remains an intensely researched area with numerous open questions, both in breadth and depth.

![Figure](x2.png)

Figure 1: Growth of Robotics (a) and Scaling Laws (b) research over time

#### Neural Scaling Laws in robotics

While scaling laws have been extensively studied in domains such as language models and vision models, their exploration in robotics remains largely uncharted territory. Although some recent work suggests that scaling phenomena hold for robotics regarding data size (e.g., [[[8](#bib.bib8), [51](#bib.bib51), [9](#bib.bib9)]]), model size (e.g., [[[7](#bib.bib7), [47](#bib.bib47)]]), and compute (e.g., [[[45](#bib.bib45)]]), where larger models tend to perform better, very few studies explicitly mention “scaling laws” or systematically quantify scaling behavior, as shown in Table [III](#S2.T3). We identified only four papers that mention scaling laws or examine them in the context of robotics and embodied AI. First, [Villasevil et al. [[81](#bib.bib81)]] discusses scaling laws in terms of increasing the number of environments in a dataset. However, they neither quantify these laws nor define a specific functional form. Their study focuses on two tasks (pick-and-place) using a dataset with a maximum of 56 scenes. Second, [Lin et al. [[44](#bib.bib44)]] investigates data scaling laws for single-task robot policies and their generalization. They emphasize the importance of data diversity over the sheer number of demonstrations and find that performance approximately follows a power law with diminishing returns. The power-law exponent varies between -0.446 and -0.844 for the number of objects and environments. Third, [Duan et al. [[17](#bib.bib17)]] examines the performance of a VLM with respect to dataset size. Their analysis reveals an average quadratic fit gradient of 0.0022 across four metrics, indicating a scaling effect. This suggests that further increasing the dataset size could enhance model performance. Finally, [Pearce et al. [[52](#bib.bib52)]] explores scaling laws related to agents and world models, focusing on upstream compute scaling for behavior cloning and world models. They observe power-law relationships with exponents ranging from -0.03 to -0.31. Their findings highlight how task, architecture, and tokenizer choices heavily influence these coefficients. Additionally, they propose optimal dataset and model sizes in relation to compute resources. Overall, while these studies provide early insights, further research is needed to address the significant gaps in understanding scaling laws within robotics and embodied AI.

*TABLE III: Summary of Scaling Studies in robotics & Embodied AI*

In contrast to the prior studies, our work expands the understanding of scaling laws in robotics and embodied AI by addressing several key gaps. We analyze a significantly larger dataset, enabling robust insights that cover the full spectrum of scaling behaviors. Our findings confirm that power-law approximations effectively model the observed trends across diverse scenarios. Unlike previous studies, we examine scaling laws comprehensively across data, model size, and compute. Additionally, we contextualize these laws by comparing them with those observed in other modalities. Our analysis differentiates between model types, such as LLMs and VLAs, deployment contexts (simulation versus real-world robotics), and even variations in compute scaling, including pretraining versus fine-tuning. Furthermore, we explore task- and method-specific effects, shedding light on nuanced scaling dynamics. Finally, we identify the emergence of novel capabilities that manifest as models scale, providing critical insights into the transformative potential of these systems.

## III Empirical approach

### III-A Research paper meta-analysis

This study presents a meta-analysis of 327 research papers, drawing from diverse sources such as survey articles [[[19](#bib.bib19), [87](#bib.bib87), [80](#bib.bib80)]], GitHub repositories [[[29](#bib.bib29), [92](#bib.bib92), [58](#bib.bib58), [20](#bib.bib20)]], citations of the seminal work by [[[51](#bib.bib51)]], and recent publications featured in newsletters and personalized news feeds (e.g., X and [[[15](#bib.bib15)]]). The analyzed papers encompass a wide range of RFMs and LLMs applied in robotics, addressing tasks such as manipulation, navigation, reasoning, planning, and instruction following. Studies focusing solely on navigation are excluded, unless combined with manipulation tasks due to the defined focus of this study. These models vary in scope, from task-specific systems to versatile general-purpose architectures, and are implemented across diverse robotic embodiments, including industrial arms and legged robots, operating in both simulated and real-world environments. A notable focus is on manipulation tasks, primarily involving single-arm operations such as grasping and pick-and-place actions, which are prevalent in the largest robotics datasets, such as RT-X, DROID or AgiBot World [[[51](#bib.bib51), [36](#bib.bib36), [1](#bib.bib1)]]. These tasks are typically set in everyday environments, like kitchens and tabletops, though some studies extend to specialized domains, such as laboratory automation.

76% of all 327 analyzed papers were published in 2023 and 2024, aligning with the exponental surge in publications in the field as illustrated in Figure [1](#S2.F1). Moreover, a significant portion of these studies originates from collaborative projects between industry and academia (Figure [2](#S3.F2)), highlighting the increasing role of industry in AI research [[[2](#bib.bib2)]]. Notably, Google DeepMind contributes to 19% of all publications, making it the most prolific institution in this field, ahead of other leading entities like Stanford and UC Berkeley (Figure [2](#S3.F2)).

![Figure](x3.png)

![Figure](x4.png)

Figure 2: Publication by sector (a) and top 10 research institutions by publications.

We chose meta-analysis as our approach due to the nascent and exploratory nature of the robot learning field, which is still evolving toward standardization. This field is characterized by a diverse array of tasks, model architectures, and embodiment forms, making it challenging for individual experimental studies to provide a comprehensive understanding. Single studies often offer a narrow perspective and can be influenced by specific biases or constraints. By aggregating data across a wide variety of research efforts, meta-analysis allows us to uncover broader trends and insights that individual studies cannot capture.

### III-B Data extraction from papers

Each of the 327 papers was screened for relevance and evaluated for scalability studies across several metrics: Success Rate, computational resources (FLOP, PF-days), data usage (tokens, demonstrations, trajectories, episodes, images, frames), and model size (parameters). We focused exclusively on works that provided numeric performance values for the success rate, either directly in the paper or shared by the researchers in response to our inquiries, while acknowledging the existence of additional scaling studies using other metrics. The choice of success rate as the focus of our analysis is due to its status as the most commonly used metric.
We found that 13% of the papers include some form of scalability study, as shown in table [IV](#S3.T4)333The following papers provided the scaling law studies for this paper:
Data: [Reed et al. [[57](#bib.bib57)], Rana et al. [[56](#bib.bib56)], Wen et al. [[83](#bib.bib83)], Shridhar et al. [[66](#bib.bib66)], Gao et al. [[23](#bib.bib23)], Tang et al. [[72](#bib.bib72)], Jiang et al. [[33](#bib.bib33)], Li et al. [[43](#bib.bib43)], Fu et al. [[21](#bib.bib21)], Stone et al. [[70](#bib.bib70)], Li et al. [[42](#bib.bib42)], Nair et al. [[49](#bib.bib49)], Kuang et al. [[38](#bib.bib38)], Brohan et al. [[8](#bib.bib8)], Radosavovic et al. [[54](#bib.bib54)], Bousmalis et al. [[7](#bib.bib7)], Radosavovic et al. [[53](#bib.bib53)], Etukuru et al. [[18](#bib.bib18)], Bu et al. [[10](#bib.bib10)], Zhou et al. [[91](#bib.bib91)]] Model size: [Staroverov et al. [[69](#bib.bib69)], Cheang et al. [[11](#bib.bib11)], Jiang et al. [[33](#bib.bib33)], Lynch and Sermanet [[46](#bib.bib46)], Huang et al. [[31](#bib.bib31)], Stone et al. [[70](#bib.bib70)], Chen et al. [[12](#bib.bib12)], Wu et al. [[85](#bib.bib85)], Radosavovic et al. [[54](#bib.bib54), [53](#bib.bib53)], Wen et al. [[84](#bib.bib84)]] Compute: [Liu et al. [[45](#bib.bib45)]]

*TABLE IV: Number of scaling studies found out of 327 papers in total*

These studies typically explore the effects of scaling data size, model size, and compute often through ablations comparing these factors to downstream performance metrics such as success rate. In many cases, a paper reports multiple scaling studies, such as those conducted for different tasks. Each scaling study is then fitted to a power law, meaning the number of power laws we analyze statistically corresponds to the number of scaling studies with at least three data points, the minimum required for modeling a power law with a constant. To clarify, we do not fit a single model to all scaling studies. Instead, we fit separate models for each scaling study presented in the papers. While conditions can vary between scaling studies, the conditions within each study are consistent. This approach allows us to derive multiple models, which we then compare statistically. It is also important to note that our study does not aim to estimate scaling laws for predicting model performance based on specific inputs. Instead, our focus is on understanding how models generally improve under the conditions presented in each study and evaluating the efficiency of these scaling studies. However, we found that many of these scaling studies are limited by their small number of data points, often consisting of only two or three data points.

The relatively small number of studies reporting scaling analyses underlines that this is still an early stage in the exploration of scalable solutions in robotics. This mirrors the progress seen in other machine learning domains, where scalable methods have played a significant role in advancing the field. Currently, most scaling studies focus on data (26 papers), followed by model size (22 papers) and compute (1 paper). Often, papers report multiple scalability analyses, assessing model performance across different tasks with varied inputs (see table [IV](#S3.T4)). The scarcity of scalability studies related to computational resources underscores an important gap in the field. Previous research on LLMs indicates that model performance is not solely dependent on data, model size, or compute resources in isolation, but rather on the interplay among these factors [[[34](#bib.bib34)]]. Hence, in this paper we advocate for reporting computational resource usage [[[65](#bib.bib65)]], as well as conducting comprehensive studies on computational scalability, potentially together with a compute-optimal scaling law study.

In robotics, there is no standard benchmark that ensures comparability of results across studies, and task success rate is the most commonly used performance metric. To analyze this metric effectively, we categorize the tasks as “seen” (familiar), “unseen” (unfamiliar), or as having unreported performance.
Most studies utilized demonstrations, trajectories, and episodes as their primary data sources. Given that these are the only metrics with substantial sample sizes, all data quantifications presented in this analysis are based exclusively on these samples. In contrast, there is a notable deficiency in computational studies; only one paper has explored how the success rate varies with changes in the number of epochs. Using the hardware specifications and training duration details provided in this paper, we estimated the training FLOP with a tool444[https://epochai.org/blog/estimating-training-compute](https://epochai.org/blog/estimating-training-compute) by Epoch.ai. FLOP, or PetaFLOP-days, serve as more precise metrics for comparing the computational demands of AI model training than, for example, GPU days [[[76](#bib.bib76), [4](#bib.bib4), [64](#bib.bib64), [24](#bib.bib24)]].

### III-C Scaling Laws analysis

Using the dataset we compiled, we modeled the scaling behavior observed in the studies using the following power law equations:

| | $$ErrorRate(D)=\left(\frac{A}{D}\right)^{\alpha}+E,$$ | |
|---|---|---|

for data size ($D$),

| | $$ErrorRate(N)=\left(\frac{B}{N}\right)^{\beta}+E,$$ | |
|---|---|---|

for model size ($N$), and

| | $$ErrorRate(C)=\left(\frac{F}{C}\right)^{\gamma}+E,$$ | |
|---|---|---|

for compute ($C$).

Here, $A$, $B$, and $F$ are scaling constants, while $\alpha$, $\beta$, and $\gamma$ are the scaling exponents corresponding to each resource. The error rate, expressed as a percentage ($100-\text{success rate}$), decreases as resources increase. The additive constant $E$ represents an offset value.

Among these coefficients, the scaling exponents ($\alpha$, $\beta$, and $\gamma$) are particularly significant as they govern the rate at which performance improves with additional resources. Our analysis focuses on the regime where each of the scaling exponents falls within the range $-1“1x” marker denotes the smallest number of demonstrations used in the authors’ scaling studies, while “10x”, “100x”, and beyond represent extrapolations based on the minimum delta between the maximum and minimum number of demonstrations across all studies. This visualization highlights the variation in power law exponent coefficients, reflecting differences in task difficulty and architecture differences. Notably, this pattern parallels findings in image generation, where image size influences distinct scaling laws [[[25](#bib.bib25)]].

The data scaling studies analyzed span datasets ranging from 1 to 1 million demonstrations, with a median ratio of 10x between the maximum and minimum number of demonstrations across all scaling law studies (see Table [VI](#S4.T6)).

![Figure](x5.png)

*Figure 3: Scaling laws in robotics: (a, c, e) show scaling across data, model size, and compute, with fitted power laws. (b, d, f) illustrates the relative scaling behaviors as $D$, $N$, and $C$ increase, modeled by $\text{Error Rate}(X)=X^{(\bar{\sigma_{X}}\pm K_{X})}$, where $X$ denotes the input variable ($D$, $N$, or $C$), $\bar{\sigma_{X}}$ is the mean scaling exponent ($\alpha$, $\beta$, or $\gamma$), and $K_{X}$ represents the Confidence Interval of $\sigma_{X}$.*

*TABLE V: Explanatory power of power laws compared to linear models*

*TABLE VI: Statistics on Scaling Studies*

While we confirm the power law behavior, this result is consistent with prior studies across various domains that have established and validated the ubiquity of power law scaling. However, it is important to emphasize that power laws inherently exhibit diminishing returns as resources increase. This implies that achieving high accuracy and reliability requires disproportionately large amounts of resources, posing challenges due to data limitations. Furthermore, real-world industrial and home applications often demand accuracy and reliability levels of 99.X% or higher. Given the long-tail nature of power law scaling, reaching such levels is particularly challenging [[[39](#bib.bib39)]]. Across our dataset of 424 studies (comprising 214 studies with at least three data points per scaling study, plus additional studies with two data points), the mean top performance achieved was 67% (median 70%; SD = 24%), highlighting the need for more extensive research to achieve the desired levels of accuracy and reliability.

Examining scaling efficiency (as shown in Figure [4](#S4.F4)), we find an average power law gradient (exponent) of -0.276 (95% CI = -0.317, -0.236) across 131 data-scaling studies, where CI refers to the confidence interval. Most models analyzed are VLA models (n=93). Interestingly, while the mean $\alpha$ value shows little variation between VLA models, VLMs, and PVRs, LLMs appear to scale significantly more efficiently. However, this observation should be treated with caution due to the small sample size.

![Figure](x6.png)

*Figure 4: Data size scaling laws with error bars (mean, standard error). Statistical significance is indicated as follows: n.s. (not significant) $p>0.05$; * $p\leq 0.05$; ** $p\leq 0.01$; *** $p\leq 0.001$.*

### IV-B Scaling Laws: Model Size

In comparison to data scaling, we observe fewer studies focused on model size (n=34). Nevertheless, similar to data, model performance improves with increasing model size and adheres to a power law relationship (see Figure [3](#S4.F3) and Table [V](#S4.T5)). The model sizes in these studies range from millions to trillions of parameters (refer to Table [VI](#S4.T6)).

The average power law gradient for the scaling exponent is -0.246 (95% CI = -0.337, -0.155) across all 34 model size scaling studies. Beyond this, the insights largely align with those observed in data scaling. While some deviations exist (e.g., for LLMs), these are likely attributable to the low statistical power of the findings, as only three scaling studies per model type were available (see Figure [5](#S4.F5)).

![Figure](x7.png)

*Figure 5: Model size scaling laws with error bars (mean, standard error). Statistical significance is indicated as follows: n.s. (not significant) $p>0.05$; * $p\leq 0.05$; ** $p\leq 0.01$; *** $p\leq 0.001$.*

### IV-C Scaling Laws: Compute

Compared to data and model size, compute scaling has received considerably less attention in robotics research within our dataset. Specifically, we identified only one study that investigated the impact of compute scaling on performance, spanning 6 tasks and 4 architectures, resulting in 24 scaling studies in total [[[45](#bib.bib45)]]. While the limited number of studies on compute scaling is unfortunate, it offers valuable insights into how task and architecture influence scaling efficiency.

Our analysis reveals that the mean power law fit gradient for the scaling exponent ranges from -0.050 (95% CI = -0.117, 0.017) to -0.304 (95% CI = -0.538, -0.070), with an average of -0.141 (95% CI = -0.189, -0.093) across all compute scaling studies. Interestingly, we observe differences in scaling efficiency between tasks and across different methods. This suggests that both architectural design and task characteristics significantly influence scaling efficiency, aligning with previous research by [Rosenfeld et al. [[61](#bib.bib61)]] and contrasting with findings from [Hestness et al. [[26](#bib.bib26)]]. For example, variations in power coefficients have been reported for image generation tasks with differing pixel sizes (see Table [II](#S2.T2)), underscoring the complexity and diversity of scaling behaviors. Additionally, our analysis finds that pretraining is generally more scaling-efficient than fine-tuning (Figure [6](#S4.F6)).

The origins of scaling laws research in Natural Language Processing and other domains have traditionally placed a strong emphasis on compute. In robotics, however, data scaling appears to take higher priority, likely due to the absence of internet-scale datasets for embodied AI. This contrast highlights a fundamental difference between the two fields.

![Figure](x8.png)

*Figure 6: Compute scaling laws with error bars (mean, standard error). Statistical significance is indicated as follows: n.s. (not significant) $p>0.05$; * $p\leq 0.05$; ** $p\leq 0.01$; *** $p\leq 0.001$.*

### IV-D Summary of Scaling Law Findings: Data, Model Size, and Compute

The power coefficients $\alpha$, $\beta$ and $\gamma$ are the most significant factors in understanding scaling behaviors, as outlined in Tables [VII](#S4.T7) and [VIII](#S4.T8). While Table [VII](#S4.T7) presents all collected data, Table [VIII](#S4.T8) filters out outliers with low $R^{2}$ values, focusing on data points where increased resources directly correlate with improved performance. This filtering approach reduces deviations between the median and mean values, refining our understanding of scaling efficiency.

For data size, the power coefficients range from -0.217 (median) to -0.276 (mean); for model size, they span from -0.172 to -0.246, and for compute, from -0.105 to -0.141. This means, that that doubling performance in robotics requires scaling data by $24.39\times$, parameters by $56.26\times$, and compute by $736.13\times$ (median power exponent used for calculation).

Interestingly, scaling coefficients for seen data (mean -0.389; 95% CI: -0.502, -0.276) substantially outperform those for unseen data (mean -0.155; 95% CI: -0.216, -0.094), reinforcing the importance of diverse and comprehensive datasets in improving general-purpose robotics performance.

Our analysis indicates that 87% of scaling studies show $\alpha$, $\beta$ and $\gamma$ values between 0 and -1, suggesting diminishing returns as resources increase. Considering the constraints of compute, data availability in robotics (with the largest dataset comprising 2.5 million episodes [[[51](#bib.bib51)]]), and the challenges of inference on edge devices, achieving efficient scaling in robotics remains a significant challenge.

These findings broadly align with prior studies on scaling laws in robotics and embodied AI, though some differences in the observed $\alpha$, $\beta$ and $\gamma$ values are notable. Specifically, we observe power exponent values for data size ranging from -0.01 to -1, with a mean of -0.276 (131 scaling studies). In comparison, [Lin et al. [[44](#bib.bib44)]] reports a more constrained range from -0.446 to -0.844 across tasks (6 scaling studies). Our compute scaling laws span from -0.01 to -0.52 (24 scaling studies), which is somewhat broader than the range of -0.03 to -0.31 (mean of -0.141) reported by [Pearce et al. [[52](#bib.bib52)]] (4 scaling studies). Despite these variations, the overall trend of diminishing returns as resources increase remains consistent with previous studies, reinforcing the challenges of scaling in robotics and embodied AI.

*TABLE VII: $\alpha$, $\beta$ and $\gamma$ of power law approximation of Robot Foundation Models (RFMs) and LLMs in robotics*

*TABLE VIII: $\alpha$, $\beta$ and $\gamma$ of power law approximation of Robot Foundation Models (RFMs) and LLMs in robotics for $R^{2}>0.7$*

### IV-E Comparison of Robotics Scaling Laws to Other Modalities

Robotics scaling laws reveal $\beta$ values similar to those observed in image generation and text-to-image models, across both model size, data, and compute metrics, as detailed in Tables [I](#S2.T1) and [II](#S2.T2). Intriguingly, language training (classical LLM) remains more resource-intensive, even though robotics has traditionally been viewed as one of the most demanding domains in AI. This discrepancy may be attributed to the extensive reliance on image and video data in robotics. Future research should explore how varying data types (e.g., language, image, video, and action) and their respective proportions impact scaling efficiencies.

### IV-F Emergent Capabilities of Robot Foundation Models and LLMs used in robotics

![Figure](x9.png)

![Figure](x10.png)

Figure 7: Emergent Capabilities in Robotics: We find a wide range of emergent capabilities in robotics for both data (a) and model size (b).

In this study, we observed cases of emergent behavior, visualized in Figure [7](#S4.F7). Previous work (e.g., [[[13](#bib.bib13), [88](#bib.bib88), [55](#bib.bib55), [70](#bib.bib70), [10](#bib.bib10), [33](#bib.bib33)]]) has documented instances where models exhibited a 0% success rate in their smallest-scale ablation studies but developed capabilities with increased scale.

For example, we noted emergent capabilities in task and motion planning for LLMs in robotics. However, these findings were inconsistently replicated across studies, and no significant patterns emerged due to the limited number of cases available for analysis. For model size, 10 cases of emergent behavior were observed across 125 scaling studies. Similarly, for data size, 18 out of 275 scaling studies exhibited emergent capabilities.

One reason for this inconsistency might be that researchers often prioritize reporting tasks where models succeed, leaving failures—such as cases with 0% success—underreported. Yet, tasks with zero initial success rate are essential for identifying emergent capabilities. Future research should explicitly report tasks where models fail, aiding the study of emergent behavior and advancing our understanding of generalization.

## V Discussion

#### Summary

We conduct a meta-analysis on neural scaling laws in robotics, focusing on RFMs and LLMs in robotics, analyzed 327 papers to examine performance scaling with data, model size, and compute. Results confirm that power laws best describe these relationships, indicating diminishing returns with increased scale. Observed scaling laws for RFMs align with those in vision domains, with variations by task complexity and architectural designs. Emergent capabilities in RFMs and LLMs suggest potential for task generalization, offering insights for optimizing efficient robotic systems.

#### Limitations

Most scaling studies in robotics present only a limited number of data points, which constrains the precise estimation of power-law coefficients and contributes to variability in reported results. Moreover, model performance is often evaluated on tasks of varying complexity, leading to significant deviations even for the same model across different tasks. This underscores the pressing need for widely accepted, general-purpose, and open-ended benchmarks in robotics, similar to ImageNet in computer vision.
In addition to benchmarks, the field should establish standardized success rate metrics, as current practices vary significantly, further complicating comparisons. Unlike other machine learning domains, scaling studies in robotics often use validation sets that overlap with the training set, increasing the risk of overfitting to specific conditions.
Another key distinction is that scaling laws for foundation models in robotics do not follow predictable patterns like Chinchilla scaling laws, which assume proportional scaling of data, compute, and model size. This divergence makes direct comparisons between studies more challenging. Finally, there is a lack of differentiation between various architectural designs, which could also impact scalability and performance.

#### Ethical Considerations

While meta-analysis methods are typically not directly associated with ethical considerations, we recognize their significance and highlight the implications of our work. As we scale up robot systems, safety and control are paramount concerns due to the potential for physical harm. Additionally, as these models become more powerful, the complexity and risks increase, making robust safety protocols essential. However, our study not only acknowledges these risks but also offers positive contributions by enabling predictions about future models and their capabilities, assisting in establishing effective safety standards, and helping society prepare for and adapt to these technological advancements [[[73](#bib.bib73)]]. Moreover, the environmental impact of scaling AI cannot be ignored due to the immense computational resources required, necessitating eco-friendly approaches [[[77](#bib.bib77)]]. More specifically, the increasing cost of model training necessitate a shift towards algorithmic efficiency over relying on exponential compute scaling [[[74](#bib.bib74), [40](#bib.bib40), [75](#bib.bib75), [78](#bib.bib78)]]. Lastly, the societal impact of widespread deployment of capable robotic agents must be carefully considered. While such systems hold the potential for substantial benefits, their disruptive effects on job markets, social structures, and existing power dynamics could exacerbate inequalities if not managed responsibly.

## Acknowledgments

This project has greatly benefited from discussions, ideas, inspiration and feedback provided by Jonathan Rosenfeld, Ana Trisovic, Emanuele Del Sozzo, Daniel Zhao, Gabriel Filipe, Alexander Fogelson, Hans Gundlach, Harry Lyu, Zachary Brown, Leonard Meinzinger and Simon Bohnen. We are also thankful to Prof. Joachim Henkel for co-supervising an earlier version of this work.

## References

-
AgiBot [2025]

AgiBot.

[AgiBot World Dataset](https://agibot-world.com/), 2025.

-
Ahmed et al. [2023]

Nur Ahmed, Muntasir Wahed, and Neil C Thompson.

[The growing influence of industry in AI research](https://www.science.org/doi/10.1126/science.ade2420).

Science*, 379(6635):884–886, 2023.

-
Alabdulmohsin et al. [2022]

Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai.

[Revisiting neural scaling laws in language and vision](https://arxiv.org/pdf/2209.06640).

*Advances in Neural Information Processing Systems*, 35:22300–22312, 2022.

-
Amodei and Hernandez [2018]

Dario Amodei and Danny Hernandez.

[AI and compute](https://openai.com/index/ai-and-compute/), 2018.

-
Black et al. [2024]

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al.

[$\pi\_0$: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/pdf/2410.24164).

*arXiv preprint arXiv:2410.24164*, 2024.

-
Bommasani et al. [2021]

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.

[On the opportunities and risks of foundation models](https://arxiv.org/pdf/2108.07258).

*arXiv preprint arXiv:2108.07258*, 2021.

-
Bousmalis et al. [2023]

Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Manon Devin, Alex X Lee, Maria Bauza Villalonga, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, et al.

[RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/pdf/2306.11706).

*Transactions on Machine Learning Research*, 2023.

-
Brohan et al. [2022]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.

[Rt-1: Robotics transformer for real-world control at scale](https://arxiv.org/pdf/2212.06817).

*arXiv preprint arXiv:2212.06817*, 2022.

-
Brohan et al. [2023]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al.

[Rt-2: Vision-language-action models transfer web knowledge to robotic control](https://arxiv.org/pdf/2307.15818).

*arXiv preprint arXiv:2307.15818*, 2023.

-
Bu et al. [2024]

Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, and Yu Qiao.

[Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation](https://arxiv.org/pdf/2410.08001), 2024.

-
Cheang et al. [2024]

Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu.

[GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation](https://arxiv.org/pdf/2410.06158), 2024.

-
Chen et al. [2023a]

Boyuan Chen, Fei Xia, Brian Ichter, Kanishka Rao, Keerthana Gopalakrishnan, Michael S Ryoo, Austin Stone, and Daniel Kappler.

[Open-vocabulary queryable scene representations for real world planning](https://arxiv.org/pdf/2209.09874).

In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 11509–11522. IEEE, 2023a.

-
Chen et al. [2023b]

Yongchao Chen, Jacob Arkin, Yang Zhang, Nicholas Roy, and Chuchu Fan.

[Autotamp: Autoregressive task and motion planning with llms as translators and checkers](https://arxiv.org/pdf/2306.06531).

*arXiv preprint arXiv:2306.06531*, 2023b.

-
Coates et al. [2011]

Adam Coates, Andrew Ng, and Honglak Lee.

[An analysis of single-layer networks in unsupervised feature learning](https://proceedings.mlr.press/v15/coates11a.html).

In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pages 215–223. JMLR Workshop and Conference Proceedings, 2011.

-
DAIR.AI [2024]

DAIR.AI.

[ML Papers of The Week](https://github.com/dair-ai/ML-Papers-of-the-Week), 2024.

-
Driess et al. [2023]

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al.

[Palm-e: An embodied multimodal language model](https://arxiv.org/pdf/2303.03378).

*arXiv preprint arXiv:2303.03378*, 2023.

-
Duan et al. [2024]

Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, and Yijie Guo.

[AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation](https://arxiv.org/pdf/2410.00371).

*arXiv preprint arXiv:2410.00371*, 2024.

-
Etukuru et al. [2024]

Haritheja Etukuru, Norihito Naka, Zijin Hu, Seungjae Lee, Julian Mehu, Aaron Edsinger, Chris Paxton, Soumith Chintala, Lerrel Pinto, and Nur Muhammad Mahi Shafiullah.

[Robot utility models: General policies for zero-shot deployment in new environments](https://arxiv.org/pdf/2409.05865).

*arXiv preprint arXiv:2409.05865*, 2024.

-
Firoozi et al. [2023]

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al.

[Foundation models in robotics: Applications, challenges, and the future](https://arxiv.org/pdf/2312.07843).

*arXiv preprint arXiv:2312.07843*, 2023.

-
Firoozi et al. [2024]

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al.

[Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models), 2024.

-
Fu et al. [2024]

Zipeng Fu, Tony Z Zhao, and Chelsea Finn.

[Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation](https://arxiv.org/pdf/2401.02117).

*arXiv preprint arXiv:2401.02117*, 2024.

-
Ganguli et al. [2022]

Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, et al.

[Predictability and surprise in large generative models](https://arxiv.org/pdf/2202.07785).

In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*, pages 1747–1764, 2022.

-
Gao et al. [2024]

Jensen Gao, Annie Xie, Ted Xiao, Chelsea Finn, and Dorsa Sadigh.

[Efficient Data Collection for Robotic Manipulation via Compositional Generalization](https://arxiv.org/pdf/2403.05110), 2024.

-
Heim [2023]

Lennart Heim.

[FLOP for Quantity, FLOP/s for Performance](https://blog.heim.xyz/flop-for-quantity-flop-s-for-performance/), 2023.

-
Henighan et al. [2020]

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al.

[Scaling laws for autoregressive generative modeling](https://arxiv.org/pdf/2010.14701).

*arXiv preprint arXiv:2010.14701*, 2020.

-
Hestness et al. [2017]

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou.

[Deep learning scaling is predictable, empirically](https://arxiv.org/pdf/1712.00409).

*arXiv preprint arXiv:1712.00409*, 2017.

-
Hilton et al. [2023]

Jacob Hilton, Jie Tang, and John Schulman.

[Scaling laws for single-agent reinforcement learning](https://arxiv.org/pdf/2301.13442).

*arXiv preprint arXiv:2301.13442*, 2023.

-
Hoffmann et al. [2022]

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.

[Training compute-optimal large language models](https://arxiv.org/abs/2203.15556).

*arXiv preprint arXiv:2203.15556*, 2022.

-
Hu [2024]

Jeffrey Hu.

[robotics-fm-survey](https://github.com/JeffreyYH/robotics-fm-survey/tree/master), 2024.

-
Hu et al. [2023]

Yafei Hu, Quanting Xie, Vidhi Jain, Jonathan Francis, Jay Patrikar, Nikhil Keetha, Seungchan Kim, Yaqi Xie, Tianyi Zhang, Zhibo Zhao, et al.

[Toward general-purpose robots via foundation models: A survey and meta-analysis](https://arxiv.org/pdf/2312.08782).

*arXiv preprint arXiv:2312.08782*, 2023.

-
Huang et al. [2023]

Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li.

[Instruct2act: Mapping multi-modality instructions to robotic actions with large language model](https://arxiv.org/pdf/2305.11176).

*arXiv preprint arXiv:2305.11176*, 2023.

-
Jang [2024]

Eric Jang.

[Data Engines for Humanoid Robots](https://www.youtube.com/watch?v=laeJn2-CBTk), 2024.

-
Jiang et al. [2023]

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan.

[Vima: Robot manipulation with multimodal prompts](https://arxiv.org/pdf/2210.03094), 2023.

-
Kaplan et al. [2020]

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.

[Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361).

*arXiv preprint arXiv:2001.08361*, 2020.

-
Kawaharazuka et al. [2024]

Kento Kawaharazuka, Tatsuya Matsushima, Andrew Gambardella, Jiaxian Guo, Chris Paxton, and Andy Zeng.

[r](http://dx.doi.org/10.1080/01691864.2024.2408593)eal-world robot applications of foundation models: a review.

*Advanced Robotics*, 38(18):1232–1254, September 2024.

ISSN 1568-5535.

doi: 10.1080/01691864.2024.2408593.

-
Khazatsky et al. [2024]

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen,
Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn.

[DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/pdf/2403.12945), 2024.

-
Kim et al. [2024]

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al.

[OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/pdf/2406.09246).

*arXiv preprint arXiv:2406.09246*, 2024.

-
Kuang et al. [2024]

Yuxuan Kuang, Junjie Ye, Haoran Geng, Jiageng Mao, Congyue Deng, Leonidas Guibas, He Wang, and Yue Wang.

[RAM: Retrieval-Based Affordance Transfer for Generalizable Zero-Shot Robotic Manipulation](https://arxiv.org/pdf/2407.04689), 2024.

-
Kumar [2024]

Nishanth J. Kumar.

[Will Scaling Solve Robotics?](https://spectrum.ieee.org/solve-robotics), 2024.

-
Leiserson et al. [2020]

Charles E Leiserson, Neil C Thompson, Joel S Emer, Bradley C Kuszmaul, Butler W Lampson, Daniel Sanchez, and Tao B Schardl.

[There’s plenty of room at the Top: What will drive computer performance after Moore’s law?](https://www.science.org/doi/10.1126/science.aam9744)

*Science*, 368(6495):eaam9744, 2020.

-
Li et al. [2024a]

Dingzhe Li, Yixiang Jin, Hongze Yu, Jun Shi, Xiaoshuai Hao, Peng Hao, Huaping Liu, Fuchun Sun, Jianwei Zhang, Bin Fang, et al.

[What foundation models can bring for robot learning in manipulation: A survey](https://arxiv.org/pdf/2404.18201).

*arXiv preprint arXiv:2404.18201*, 2024a.

-
Li et al. [2022]

Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, et al.

[Pre-trained language models for interactive decision-making](https://arxiv.org/pdf/2202.01771).

*Advances in Neural Information Processing Systems*, 35:31199–31212, 2022.

-
Li et al. [2024b]

Xiang Li, Cristina Mata, Jongwoo Park, Kumara Kahatapitiya, Yoo Sung Jang, Jinghuan Shang, Kanchana Ranasinghe, Ryan Burgert, Mu Cai, Yong Jae Lee, and Michael S. Ryoo.

[LLaRA: Supercharging Robot Learning Data for Vision-Language Policy](https://arxiv.org/pdf/2406.20095), 2024b.

-
Lin et al. [2024]

Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao.

[Data scaling laws in imitation learning for robotic manipulation](https://arxiv.org/pdf/2410.18647).

*arXiv preprint arXiv:2410.18647*, 2024.

-
Liu et al. [2023]

Zuxin Liu, Jesse Zhang, Kavosh Asadi, Yao Liu, Ding Zhao, Shoham Sabach, and Rasool Fakoor.

[TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models](https://arxiv.org/pdf/2310.05905).

*arXiv preprint arXiv:2310.05905*, 2023.

-
Lynch and Sermanet [2020]

Corey Lynch and Pierre Sermanet.

[Language conditioned imitation learning over unstructured data](https://arxiv.org/pdf/2005.07648).

*arXiv preprint arXiv:2005.07648*, 2020.

-
Majumdar et al. [2024]

Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, et al.

[Where are we in the search for an artificial visual cortex for embodied intelligence?](https://arxiv.org/pdf/2303.18240)

*Advances in Neural Information Processing Systems*, 36, 2024.

-
Muennighoff et al. [2024]

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel.

[Scaling data-constrained language models](https://arxiv.org/pdf/2305.16264).

*Advances in Neural Information Processing Systems*, 36, 2024.

-
Nair et al. [2022]

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta.

[R3m: A universal visual representation for robot manipulation](https://arxiv.org/pdf/2203.12601).

*arXiv preprint arXiv:2203.12601*, 2022.

-
Nasiriany et al. [2024]

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu.

[RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://arxiv.org/pdf/2406.02523).

In *Robotics: Science and Systems (RSS)*, 2024.

-
Padalkar et al. [2023]

Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al.

[Open x-embodiment: Robotic learning datasets and rt-x models](https://arxiv.org/pdf/2310.08864).

*arXiv preprint arXiv:2310.08864*, 2023.

-
Pearce et al. [2024]

Tim Pearce, Tabish Rashid, Dave Bignell, Raluca Georgescu, Sam Devlin, and Katja Hofmann.

[Scaling Laws for Pre-training Agents and World Models](https://arxiv.org/pdf/2411.04434).

*arXiv preprint arXiv:2411.04434*, 2024.

-
Radosavovic et al. [2023a]

Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, and Jitendra Malik.

[Robot learning with sensorimotor pre-training](https://arxiv.org/pdf/2306.10007).

In *Conference on Robot Learning*, pages 683–693. PMLR, 2023a.

-
Radosavovic et al. [2023b]

Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell.

[Real-world robot learning with masked visual pre-training](https://arxiv.org/pdf/2210.03109).

In *Conference on Robot Learning*, pages 416–426. PMLR, 2023b.

-
Rana et al. [2023]

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf.

[Sayplan: Grounding large language models using 3d scene graphs for scalable task planning](https://arxiv.org/pdf/2307.06135).

*arXiv preprint arXiv:2307.06135*, 2023.

-
Rana et al. [2024]

Krishan Rana, Jad Abou-Chakra, Sourav Garg, Robert Lee, Ian Reid, and Niko Suenderhauf.

[Affordance-Centric Policy Learning: Sample Efficient and Generalisable Robot Policy Learning using Affordance-Centric Task Frames](https://arxiv.org/pdf/2410.12124), 2024.

-
Reed et al. [2022]

Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al.

[A generalist agent](https://arxiv.org/pdf/2205.06175).

*arXiv preprint arXiv:2205.06175*, 2022.

-
Rintamaki [2023]

Jacob Rintamaki.

[Everything-LLMs-And-Robotics](https://github.com/jrin771/Everything-LLMs-And-Robotics), 2023.

-
Rosenfeld [2021]

Jonathan S Rosenfeld.

[Scaling laws for deep learning](https://arxiv.org/pdf/2108.07686).

*arXiv preprint arXiv:2108.07686*, 2021.

-
Rosenfeld et al. [2019]

Jonathan S Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit.

[A constructive prediction of the generalization error across scales](https://arxiv.org/pdf/1909.12673).

*arXiv preprint arXiv:1909.12673*, 2019.

-
Rosenfeld et al. [2021]

Jonathan S Rosenfeld, Jonathan Frankle, Michael Carbin, and Nir Shavit.

[On the predictability of pruning across scales](https://arxiv.org/pdf/2006.10621).

In *International Conference on Machine Learning*, pages 9075–9083. PMLR, 2021.

-
Rosenfeld [2019]

Jonathan Shmuel Rosenfeld.

*[On the relation between neural network size and performance](https://dspace.mit.edu/handle/1721.1/122703)*.

PhD thesis, Massachusetts Institute of Technology, 2019.

-
Schaeffer et al. [2024]

Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo.

[Are emergent abilities of large language models a mirage?](https://arxiv.org/pdf/2304.15004)

*Advances in Neural Information Processing Systems*, 36, 2024.

-
Sevilla et al. [2022]

Jaime Sevilla, Lennart Heim, Marius Hobbhan, Tamay Besiroglu, Anson Ho, and Pablo Villalobos.

[Estimating Training Compute of Deep Learning Models](https://epochai.org/blog/estimating-training-compute), 2022.

-
Sevilla et al. [2023]

Jaime Sevilla, Anson Ho, and Tamay Besiroglu.

[Please Report Your Compute](https://cacm.acm.org/opinion/please-report-your-compute/).

*Communications of the ACM*, 66(5):30–32, 2023.

-
Shridhar et al. [2022]

Mohit Shridhar, Lucas Manuelli, and Dieter Fox.

[Cliport: What and where pathways for robotic manipulation](https://arxiv.org/pdf/2109.12098).

In *Conference on robot learning*, pages 894–906. PMLR, 2022.

-
Snell et al. [2024]

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar.

[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314), 2024.

-
Sorscher et al. [2022]

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos.

[Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/pdf/2206.14486).

*Advances in Neural Information Processing Systems*, 35:19523–19536, 2022.

-
Staroverov et al. [2023]

Aleksei Staroverov, Andrey S Gorodetsky, Andrei S Krishtopik, Uliana A Izmesteva, Dmitry A Yudin, Alexey K Kovalev, and Aleksandr I Panov.

[Fine-Tuning Multimodal Transformer Models for Generating Actions in Virtual and Real Environments](https://ieeexplore.ieee.org/document/10323309).

*IEEE Access*, 11:130548–130559, 2023.

-
Stone et al. [2023]

Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, et al.

[Open-world object manipulation using pre-trained vision-language models](https://arxiv.org/pdf/2303.00905).

*arXiv preprint arXiv:2303.00905*, 2023.

-
Sutton [2019]

Richard Sutton.

[The bitter lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf).

*Incomplete Ideas (blog)*, 13(1):38, 2019.

-
Tang et al. [2024]

Weiliang Tang, Jia-Hui Pan, Wei Zhan, Jianshu Zhou, Huaxiu Yao, Yun-Hui Liu, Masayoshi Tomizuka, Mingyu Ding, and Chi-Wing Fu.

[Embodiment-Agnostic Action Planning via Object-Part Scene Flow](https://www.arxiv.org/pdf/2409.10032), 2024.

-
The White House [2023]

Biden The White House.

[Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/), 2023.

-
Thompson [2017]

Neil Thompson.

[The economic impact of moore’s law: Evidence from when it faltered](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2899115).

*Available at SSRN 2899115*, 2017.

-
Thompson and Spanuth [2021]

Neil C Thompson and Svenja Spanuth.

[The decline of computers as a general purpose technology](https://cacm.acm.org/research/the-decline-of-computers-as-a-general-purpose-technology/).

*Communications of the ACM*, 64(3):64–72, 2021.

-
Thompson et al. [2020]

Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso.

[The computational limits of deep learning](https://arxiv.org/pdf/2007.05558).

*arXiv preprint arXiv:2007.05558*, 2020.

-
Thompson et al. [2021]

Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso.

[Deep learning’s diminishing returns: The cost of improvement is becoming unsustainable](https://spectrum.ieee.org/deep-learning-computational-cost).

*Ieee Spectrum*, 58(10):50–55, 2021.

-
Thompson et al. [2022]

Neil C Thompson, Shuning Ge, and Gabriel F Manso.

[The importance of (exponentially more) computing power](https://arxiv.org/pdf/2206.14007).

*arXiv preprint arXiv:2206.14007*, 2022.

-
Vaswani [2017]

A Vaswani.

[Attention is all you need](https://arxiv.org/pdf/1706.03762).

*Advances in Neural Information Processing Systems*, 2017.

-
Vemprala et al. [2023]

Sai Vemprala, Shuhang Chen, Abhinav Shukla, Dinesh Narayanan, and Ashish Kapoor.

[Grid: A platform for general robot intelligence development](https://arxiv.org/pdf/2310.00887), 2023.

-
Villasevil et al. [2024]

Marcel Torne Villasevil, Arhan Jain, Vidyaaranya Macha, Jiayi Yuan, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, and Abhishek Gupta.

[Scaling Robot-Learning by Crowdsourcing Simulation Environments](https://openreview.net/pdf?id=UPe0Sjspzr).

In *RSS 2024 Workshop: Data Generation for Robotics*, 2024.

-
Wei et al. [2022]

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al.

[Emergent abilities of large language models](https://arxiv.org/pdf/2206.07682).

*arXiv preprint arXiv:2206.07682*, 2022.

-
Wen et al. [2023]

Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel.

[Any-point trajectory modeling for policy learning](https://arxiv.org/pdf/2401.00025).

*arXiv preprint arXiv:2401.00025*, 2023.

-
Wen et al. [2024]

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang.

[TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/pdf/2409.12514), 2024.

-
Wu et al. [2023]

Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser.

[Tidybot: Personalized robot assistance with large language models](https://arxiv.org/pdf/2305.05658).

*Autonomous Robots*, 47(8):1087–1102, 2023.

-
Xian et al. [2024]

Zhou Xian, Yiling Qiao, Zhenjia Xu, Tsun-Hsuan Wang, Zhehuan Chen, Juntian Zheng, Ziyan Xiong, Yian Wang, Mingrui Zhang, Pingchuan Ma, Yufei Wang, Zhiyang Dou, Byungchul Kim, Yunsheng Tian, Yipu Chen, Xiaowen Qiu, Chunru Lin, Tairan He, Zilin Si, Yunchu Zhang, Zhanlue Yang, Tiantian Liu, Tianyu Li, Kashu Yamazaki, Hongxin Zhang, Huy Ha, Yu Zhang, Michael Liu, Shaokun Zheng, Zipeng Fu, Qi Wu, Yiran Geng, Feng Chen, Milky, Yuanming Hu, Chelsea Finn, Guanya Shi, Lingjie Liu, Taku Komura, Zackory Erickson, David Held, Minchen Li, Linxi ”Jim” Fan, Yuke Zhu, Wojciech Matusik, Dan Gutfreund, Shuran Song, Daniela Rus, Ming Lin, Bo Zhu, Katerina Fragkiadaki, and Chuang Gan.

[Genesis: A Generative and Universal Physics Engine for Robotics and Beyond](https://genesis-embodied-ai.github.io/), 2024.

-
Xiao et al. [2023]

Xuan Xiao, Jiahang Liu, Zhipeng Wang, Yanmin Zhou, Yong Qi, Qian Cheng, Bin He, and Shuo Jiang.

[Robot learning in the era of foundation models: A survey](https://arxiv.org/pdf/2311.14379).

*arXiv preprint arXiv:2311.14379*, 2023.

-
Yang et al. [2023]

Jingkang Yang, Yuhao Dong, Shuai Liu, Bo Li, Ziyue Wang, Chencheng Jiang, Haoran Tan, Jiamu Kang, Yuanhan Zhang, Kaiyang Zhou, et al.

[Octopus: Embodied vision-language programmer from environmental feedback](https://arxiv.org/pdf/2310.08588).

*arXiv preprint arXiv:2310.08588*, 2023.

-
Zeng et al. [2023]

Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and Philip S Yu.

[Large language models for robotics: A survey](https://arxiv.org/pdf/2311.07226).

*arXiv preprint arXiv:2311.07226*, 2023.

-
Zhai et al. [2022]

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer.

[Scaling vision transformers](https://arxiv.org/pdf/2106.04560).

In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 12104–12113, 2022.

-
Zhou et al. [2023]

Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, et al.

[Train offline, test misc: A real robot learning benchmark](https://arxiv.org/pdf/2306.00942).

pages 9197–9203, 2023.

-
Zsolt [2024]

Kira Zsolt.

[Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics), 2024.

## VI Appendix / supplemental material

### VI-A Scaling Laws results

![Figure](extracted/6155135/Images/Histogram.png)

*Figure 8: Histograms of $\alpha$, $\beta$ and $\gamma$ values, showing the distributions for data size, model size and compute.*

*TABLE IX: Data Size Scaling Laws (sorted by $\alpha$)*

*TABLE X: Model Size Scaling Laws (sorted by $\beta$)*

*TABLE XI: Compute Scaling Laws (based on [Liu et al. [[45](#bib.bib45)]]), sorted by $\gamma$*

Generated on Sat Jan 25 00:37:47 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)