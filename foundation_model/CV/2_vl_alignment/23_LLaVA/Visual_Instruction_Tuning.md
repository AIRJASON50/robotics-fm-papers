HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: anyfontsize

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: CC BY 4.0

arXiv:2304.08485v2 [cs.CV] 11 Dec 2023

[\setitemize itemsep=10pt,topsep=0pt,parsep=0pt,partopsep=0pt # Visual Instruction Tuning Haotian Liu${}^{1*}$, Chunyuan Li${}^{2*}$, Qingyang Wu${}^{3}$, Yong Jae Lee${}^{1}$ ${}^{1}$University of Wisconsin–Madison ${}^{2}$Microsoft Research ${}^{3}$Columbia University https://llava-vl.github.io](https://llava-vl.github.io)

###### Abstract

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field.
We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data.
By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding.
To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks.
Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset.
When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.

## 1 Introduction

Humans interact with the world through many channels such as vision and language, as each individual channel has a unique advantage in representing and communicating certain concepts, and thus facilitates a better understanding of the world. One of the core aspirations in artificial intelligence is to develop a general-purpose assistant that can effectively follow multi-modal vision-and-language instructions, aligned with human intent to complete various real-world tasks in the wild [[askell2021general](#bib.bib4); [li2022elevater](#bib.bib27); [li2023multimodal](#bib.bib26)].

To this end, the community has witnessed an emergent interest in developing language-augmented foundation vision models [[li2022elevater](#bib.bib27); [gan2022vision](#bib.bib16)], with strong capabilities in open-world visual understanding such as classification [([radford2021learning,](#bib.bib40); [openclip,](#bib.bib21); [yuan2021florence,](#bib.bib57); [yang2022unicl,](#bib.bib54); [pham2021combined,](#bib.bib39))], detection [[li2022grounded](#bib.bib29); [zhong2022regionclip](#bib.bib62); [liu2023grounding](#bib.bib33)], segmentation [([li2022language,](#bib.bib25); [zou2022generalized,](#bib.bib63); [zhang2023simple,](#bib.bib58))] and captioning [([wang2022git,](#bib.bib50); [li2023blip,](#bib.bib28))], as well as visual generation and editing [([DALLE2,](#bib.bib42); [LDM,](#bib.bib43); [PARTI,](#bib.bib56); [MAKEASCENE,](#bib.bib15); [Imagen,](#bib.bib44); [li2023gligen,](#bib.bib30))]. We refer readers to the Computer Vision in the Wild reading list for a more up-to-date literature compilation [([cvinw,](#bib.bib12))]. In this line of work, each task is solved independently by one single large vision model, with the task instruction implicitly considered in the model design. Further, language is only utilized to describe the image content. While this allows language to play an important role in mapping visual signals to language semantics—a common channel for human communication, it leads to models that usually have a fixed interface with limited interactivity and adaptability to the user’s instructions.

Large language models (LLM), on the other hand, have shown that language can play a wider role: a universal interface for a general-purpose assistant, where various task instructions can be explicitly represented in language and guide the end-to-end trained neural assistant to switch to the task of interest to solve it. For example, the recent success of ChatGPT [([chatgpt,](#bib.bib35))] and GPT-4 [([gpt4,](#bib.bib36))] have demonstrated the power of aligned LLMs in following human instructions, and have stimulated tremendous interest in developing open-source LLMs. Among them, LLaMA [([touvron2023llama,](#bib.bib49))] is an open-source LLM that matches the performance of GPT-3. Alpaca [([alpaca,](#bib.bib48))], Vicuna [([vicuna,](#bib.bib9))], GPT-4-LLM [([peng2023instruction,](#bib.bib38))] utilize various machine-generated high-quality instruction-following samples to improve the LLM’s alignment ability, reporting impressive performance compared with proprietary LLMs. Importantly, this line of work is *text-only*.

In this paper, we present visual instruction-tuning, the first attempt to extend instruction-tuning to the language-image multimodal space, to pave the way towards building a general-purpose visual assistant. In particular, our paper makes the following contributions:

-
•

Multimodal instruction-following data. One key challenge is the lack of vision-language instruction-following data. We present a data reformation perspective and pipeline to convert image-text pairs into an appropriate instruction-following format, using ChatGPT/GPT-4.

-
•

Large multimodal models. We develop a large multimodal model (LMM), by connecting the open-set visual encoder of CLIP [([radford2021learning,](#bib.bib40))] with the language decoder Vicuna [[vicuna](#bib.bib9)], and fine-tuning end-to-end on our generated instructional vision-language data.
Our empirical study validates the effectiveness of using generated data for LMM instruction-tuning, and suggests practical tips for building a general-purpose instruction-following visual agent. When ensembled with GPT-4, our approach achieves SoTA on the Science QA [[lu2022learn](#bib.bib34)] multimodal reasoning dataset.

-
•

Multimodal instruction-following benchmark. We present LLaVA-Bench with two challenging benchmarks, with a diverse selection of paired images, instructions and detailed annotations.

-
•

Open-source. We release the following assets to the public: the generated multimodal instruction data, the codebase, the model checkpoints, and a visual chat demo.

## 2 Related Work

Multimodal Instruction-following Agents.
In computer vision, existing works that build instruction-following agents can be broadly categorized into two classes:
$(i)$ End-to-end trained models, which are separately explored for each specific research topic. For example, the vision-language navigation task [[anderson2018vision](#bib.bib3); [hao2020towards](#bib.bib19)] and Habitat [[szot2021habitat](#bib.bib47)] require the embodied AI agent to follow natural language instructions and take a sequence of actions to complete goals in visual environments. In the image editing domain, given an input image and a written instruction that tells the agent what to do, InstructPix2Pix [[brooks2022instructpix2pix](#bib.bib6)] edits images by following the human instructions.
$(ii)$ A system that coordinates various models via LangChain [[langchain](#bib.bib1)] / LLMs [[chatgpt](#bib.bib35)], such as
Visual ChatGPT [[wu2023visual](#bib.bib53)], X-GPT [[zou2022generalized](#bib.bib63)], MM-REACT [[yang2023mm](#bib.bib55)], VisProg [[gupta2022visual](#bib.bib18)], and ViperGPT [[suris2023vipergpt](#bib.bib46)]. While sharing the same goal in building instruction-following agents, we focus on developing an end-to-end trained language-vision multimodal model for *multiple* tasks.

Instruction Tuning. In the natural language processing (NLP) community, to enable LLMs such as GPT-3 [[brown2020language](#bib.bib7)], T5 [[raffel2020exploring](#bib.bib41)], PaLM [[chowdhery2022palm](#bib.bib10)], and OPT [[zhang2022opt](#bib.bib60)] to follow natural language instructions
and complete real-world tasks, researchers have explored methods for LLM instruction-tuning [([ouyang2022training,](#bib.bib37); [wang2022benchmarking,](#bib.bib52); [wang2022self,](#bib.bib51))], leading to instruction-tuned counterparts such as InstructGPT [[ouyang2022training](#bib.bib37)]/ChatGPT [[chatgpt](#bib.bib35)], FLAN-T5 [[chung2022scaling](#bib.bib11)], FLAN-PaLM [[chung2022scaling](#bib.bib11)], and OPT-IML [[iyer2022opt](#bib.bib22)], respectively.
It turns out that this simple approach can effectively improve the zero- and few-shot generalization abilities of LLMs. It is thus natural to borrow the idea from NLP to computer vision. More broadly, the teacher-student distillation ideas with foundation models have been studied in other topics such as image classification [[faghri2023reinforce](#bib.bib14)]. Flamingo [[alayrac2022flamingo](#bib.bib2)] can be viewed as the GPT-3 moment in the multimodal domain, due to its strong performance on zero-shot task transfer and in-context-learning. Other LMMs trained on image-text pairs include BLIP-2 [[li2023blip](#bib.bib28)], FROMAGe [[koh2023grounding](#bib.bib24)], and KOSMOS-1 [([huang2023language,](#bib.bib20))].
PaLM-E [[driess2023palm](#bib.bib13)] is an LMM for embodied AI.
Based on the recent “best” open-source LLM LLaMA, OpenFlamingo [([anas_awadalla_2023_7733589,](#bib.bib5))] and LLaMA-Adapter [([zhang2023llama,](#bib.bib59))] are open-source efforts that enable LLaMA to use image inputs, paving the way to build open-source multimodal LLMs.
While these models present promising task transfer generalization performance, they are not explicitly tuned with vision-language instruction data, and their performance in multimodal tasks usually falls short compared to language-only tasks.
In this paper, we aim to fill this gap and study its effectiveness. Finally, note that visual instruction tuning is different from visual prompt tuning [[jia2022visual](#bib.bib23)]: the former aims to improve the model’s instruction-following abilities, while the latter aims to improve the parameter-efficiency in model adaptation.

## 3 GPT-assisted Visual Instruction Data Generation

The community has witnessed a surge in the amount of public multimodal data such as image-text pairs, ranging from CC [[changpinyo2021conceptual](#bib.bib8)] to LAION [[schuhmann2022laion](#bib.bib45)]. However, when it comes to multimodal instruction-following data, the available amount is limited, partially because the process for creating such data is time-consuming and less well-defined when human crowd-scouring is considered. Inspired by the success of recent GPT models in text-annotation tasks [[gilardi2023chatgpt](#bib.bib17)], we propose to leverage ChatGPT/GPT-4 for multimodal instruction-following data collection, based on the widely existing image-pair data.

For an image ${{\bf X}}_{\texttt{v}}$ and its associated caption ${{\bf X}}_{\texttt{c}}$, it is natural to create a set of questions ${{\bf X}}_{\texttt{q}}$ with the intent to instruct the assistant to describe the image content. We prompt GPT-4 to curate such a list of questions (see details in Appendix).
Therefore, a simple way to expand an image-text pair to its instruction-following version is
$\texttt{Human}:{{\bf X}}_{\texttt{q}}~{}{{\bf X}}_{\texttt{v}}\texttt{}~%
{}\texttt{Assistant}:{{\bf X}}_{\texttt{c}}\texttt{}$.
Though cheap to construct, this simple expanded version lacks diversity and in-depth reasoning in both the instructions and responses.

*Table 1: One example to illustrate the instruction-following data. The top block shows the contexts such as captions and boxes used to prompt GPT, and the bottom block shows the three types of responses. Note that the visual image is not used to prompt GPT, we only show it here as a reference.*

To mitigate this issue, we leverage language-only GPT-4 or ChatGPT as the strong teacher (both accept only text as input), to create instruction-following data involving visual content.
Specifically, in order to encode an image into its visual features to prompt a text-only GPT, we use two types of symbolic representations:
$(i)$ Captions typically describe the visual scene from various perspectives;
$(ii)$ Bounding boxes usually localize the objects in the scene, and each box encodes the object concept and its spatial location. One example is shown in the top block of Table [14](#A6.T14).

This symbolic representation allows us to encode the image as an LLM-recognizable sequence. We use COCO images [[lin2014microsoft](#bib.bib31)] and generate three types of instruction-following data.
One example per type is shown in the bottom block of Table [14](#A6.T14).
For each type, we first manually design a few examples. They are the only human annotations we have during data collection, and are used as seed examples in in-context-learning to query GPT-4.

-
•

Conversation. We design a conversation between the assistant and a person asking questions about this photo. The answers are in a tone as if the assistant is seeing the image and answering the question. A diverse set of questions are asked about the visual content of the image, including the object types, counting
the objects, object actions, object locations, relative positions between objects. Only questions that have definite answers are considered. Please see Appendix for the detailed prompt.

-
•

Detailed description. To include a rich and comprehensive description for an image, we create a list of questions with such an intent. We prompt GPT-4 then curate the list (see detailed prompts and curation process in Appendix). For each image, we randomly sample one question from the list to ask GPT-4 to generate the detailed description.

-
•

Complex reasoning. The above two types focus on the visual content itself, based on which we further create in-depth reasoning questions. The answers typically require a step-by-step reasoning process by following rigorous logic.

We collect 158K unique language-image instruction-following samples in total, including 58K in conversations, 23K in detailed description, and 77k in complex reasoning, respectively. We ablated the use of ChatGPT and GPT-4 in our early experiments, and found that GPT-4 consistently provides higher quality instruction-following data, such as spatial reasoning.

## 4 Visual Instruction Tuning

### 4.1 Architecture

The primary goal is to effectively leverage the capabilities of both the pre-trained LLM and visual model. The network archtecture is illustrated in Figure [1](#S4.F1). We choose Vicuna [[vicuna](#bib.bib9)] as our LLM $f_{\boldsymbol{\phi}}(\cdot)$ parameterized by $\boldsymbol{\phi}$, as it has the best instruction following capabilities in language tasks among publicly available checkpoints [[alpaca](#bib.bib48); [vicuna](#bib.bib9); [peng2023instruction](#bib.bib38)].

![Figure](x1.png)

*Figure 1: LLaVA network architecture.*

For an input image ${{\bf X}}_{\texttt{v}}$, we consider the pre-trained CLIP visual encoder ViT-L/14 [[radford2021learning](#bib.bib40)], which provides the visual feature ${\bf Z}_{\texttt{v}}=g({{\bf X}}_{\texttt{v}})$. The grid features before and after the last Transformer layer are considered in our experiments.
We consider a simple linear layer to connect image features into the word embedding space. Specifically, we apply a trainable projection
matrix ${{\bf W}}$ to convert ${\bf Z}_{\texttt{v}}$
into language embedding tokens ${\bf H}_{\texttt{v}}$, which have the same dimensionality as the word embedding space in the language model:

$${\bf H}_{\texttt{v}}={{\bf W}}\cdot{\bf Z}_{\texttt{v}},~{}\text{with}~{}~{}{% \bf Z}_{\texttt{v}}=g({{\bf X}}_{\texttt{v}})$$ \tag{1}

Thus, we have a sequence of visual tokens ${\bf H}_{\texttt{v}}$.
Note that our simple projection scheme is lightweight, which allows us to iterate data centric experiments quickly. More sophisticated schemes to connect the image and language representations can also be considered, such as gated cross-attention in Flamingo [[alayrac2022flamingo](#bib.bib2)] and Q-former in BLIP-2 [[li2023blip](#bib.bib28)]. We leave exploring possibly more effective and sophisticated architecture designs for LLaVA as future work.

### 4.2 Training

For each image ${{\bf X}}_{\texttt{v}}$, we generate multi-turn conversation data
$({{\bf X}}_{\texttt{q}}^{1},{{\bf X}}_{\texttt{a}}^{1},\cdots,{{\bf X}}_{%
\texttt{q}}^{T},{{\bf X}}_{\texttt{a}}^{T})$, where $T$ is the total number of turns.
We organize them as a sequence, by treating all answers as the assistant’s response, and the instruction ${{\bf X}}_{\texttt{instruct}}^{t}$ at the $t$-th turn as:

$${{\bf X}}_{\texttt{instruct}}^{t}=\left\{\begin{matrix}&\text{Randomly choose}~{}~{}[{{\bf X}}_{\texttt{q}}^{1},{{\bf X}}_{\texttt{v}}]~{}~{}\text{or}~{}~{}[{{\bf X}}_{\texttt{v}},{{\bf X}}_{\texttt{q}}^{1}],~{}~{}~{}\text{the first turn}~{}t=1\\ &{{\bf X}}_{\texttt{q}}^{t},\hskip 128.0374pt\text{the remaining turns}~{}t>1\end{matrix}\right.$$ \tag{4}

This leads to the unified format for the multimodal instruction-following sequence illustrated in Table [2](#S4.T2). We perform instruction-tuning of the LLM on the prediction tokens, using its original auto-regressive training objective.

*Table 2: The input sequence used to train the model. Only two conversation turns are illustrated here; in practice, the number of turns varies based on the instruction-following data. In our current implementation, we follow Vicuna-v0 [[vicuna](#bib.bib9)] to set the system message ${{\bf X}}_{\texttt{system-message}}$ and we set <STOP> = ###. The model is trained to predict the assistant answers and where to stop, and thus only green sequence/tokens are used to compute the loss in the auto-regressive model.*

Specifically, for a sequence of length $L$, we compute the probability of the target answers ${{\bf X}}_{\texttt{a}}$ by:

| | $$p({{\bf X}}_{\texttt{a}}|{{\bf X}}_{\texttt{v}},{{\bf X}}_{\texttt{instruct}})% =\prod_{i=1}^{L}p_{\boldsymbol{\theta}}({\color[rgb]{% 0.234375,0.70703125,0.29296875}\definecolor[named]{pgfstrokecolor}{rgb}{% 0.234375,0.70703125,0.29296875}\pgfsys@color@rgb@stroke{0.234375}{0.70703125}{% 0.29296875}\pgfsys@color@rgb@fill{0.234375}{0.70703125}{0.29296875}\boldsymbol% {x}_{i}}|{{\bf X}}_{\texttt{v}},{{\bf X}}_{\texttt{instruct},<STOP> for better readability.
For LLaVA model training, we consider a two-stage instruction-tuning procedure.

#### Stage 1: Pre-training for Feature Alignment.

To strike a balance between concept coverage and training efficiency, we filter CC3M to 595K image-text pairs. Please see Appendix for details of the filtering process. These pairs are converted to the instruction-following data using the naive expansion method describe in Section [3](#S3).
Each sample can be treated as a single-turn conversation. To construct the input ${{\bf X}}_{\texttt{instruct}}$ in ([4](#S4.E4)), for an image ${{\bf X}}_{\texttt{v}}$, a question ${{\bf X}}_{\texttt{q}}$ is randomly sampled, which is a language instruction to request the assistant to describe the image briefly. The ground-truth prediction answer ${{\bf X}}_{\texttt{a}}$ is the original caption. In training, we keep both the visual encoder and LLM weights frozen, and maximize the likelihood of ([5](#S4.E5)) with trainable parameters $\boldsymbol{\theta}={{\bf W}}$ (the projection matrix) only. In this way, the image features ${\bf H}_{\texttt{v}}$ can be aligned with the pre-trained LLM word embedding. This stage can be understood as training a compatible visual tokenizer for the frozen LLM.

#### Stage 2: Fine-tuning End-to-End.

We always keep the visual encoder weights frozen, and continue to update both the pre-trained weights of the projection layer and LLM in LLaVA; i.e., the trainable parameters are $\boldsymbol{\theta}=\{{{\bf W}},\boldsymbol{\phi}\}$ in ([5](#S4.E5)). We consider two specific use case scenarios:

-
•

Multimodal Chatbot. We develop a Chatbot by fine-tuning on the 158K language-image instruction-following data in Section [3](#S3). Among the three types of responses, conversation is multi-turn while the other two are single-turn. They are uniformly sampled in training.

-
•

Science QA. We study our method on the ScienceQA benchmark [[lu2022learn](#bib.bib34)], the first large-scale multimodal science question dataset that annotates the answers with detailed lectures and explanations. Each question is provided a context in the form of natural language or an image. The assistant provides the reasoning process in natural language and selects the answer among multiple choices.
For training in ([4](#S4.E4)), we organize the data as a single turn conversation, the question & context as ${{\bf X}}_{\texttt{instruct}}$, and reasoning & answer as ${{\bf X}}_{\texttt{a}}$.

## 5 Experiments

![Figure](extracted/5288275/figures/img_extreme_ironing.png)

We assess the performance of LLaVA in instruction-following and visual reasoning capabilities with two primary experimental settings: multimodal chatbot and the ScienceQA dataset, respectively.
We train all models with 8$\times$ A100s, following Vicuna’s hyperparameters [[vicuna](#bib.bib9)]. We pre-train our model on the filtered CC-595K subset for 1 epoch with a learning rate of 2e-3 and a batch size of 128, and fine-tune on the proposed LLaVA-Instruct-158K dataset for 3 epochs, with a learning rate of 2e-5 and a batch size of 32. See Appendix for more training details.

### 5.1 Multimodal Chatbot

We developed a chatbot demo to show the image understanding and conversation abilities of LLaVA, and to study how well LLaVA is able to digest visual inputs and exhibit instruction-following capabilities.
We first use the examples in the original GPT-4 paper [[gpt4](#bib.bib36)], shown in Table [3](#S5.T3) (more examples in Appendix), that require in-depth image understanding. For comparisons, we quote the prompt and response of the multimodal GPT-4 from their paper, and query BLIP-2 and OpenFlamingo model checkpoints to get their response.

Surprisingly, although LLaVA is trained with a small multimodal instruction-following dataset ($\sim$80K unique images), it demonstrates quite similar reasoning results with multimodal GPT-4 on these examples. Note that while these images are out-of-domain for LLaVA, LLaVA is still able to understand the scenes and follow the question instruction to provide a reasonable response. In contrast, BLIP-2 and OpenFlamingo focus on describing the image, instead of following the user instruction to answer in an appropriate manner.

#### Quantitative Evaluation.

To gain a systematic understanding of the performance of LLaVA, we propose a quantitative metric to measure the model’s instruction-following capability on multimodal data. Inspired by [[vicuna](#bib.bib9)], we leverage GPT-4 to measure the quality of generated responses.
Specifically, we create triplets consisting of image, ground-truth textual descriptions, and question. The candidate models (e.g., *LLaVA) predict the answers based on the question and the image.
To provide an *approximate theoretical upper bound*, we create a reference prediction based on the question and the *ground-truth* textual descriptions, using the text-only GPT-4.
After obtaining the responses from both models, we feed the question, visual information (in the format of textual descriptions), and the generated responses from both assistants, to the judge (*i.e., *text-only GPT-4). It evaluates the helpfulness, relevance, accuracy, and level of detail of the responses from the assistants, and gives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. It is also asked to provide a comprehensive explanation for the evaluation, for us to better understand the models.
We report relative scores w.r.t. the text-only GPT-4 model that uses the textural ground truth description as visual input.
We create two benchmarks to evaluate the model’s performance.

*Table 4: Ablation on LLaVA-Bench (COCO) with different training data. We report relative scores w.r.t. a text-only GPT-4 model that uses ground truth image captions and bounding boxes as visual input. We prompt GPT-4 with the answers from our model outputs and the answers by GPT-4 (text-only), and let it compare between both responses and give a rating with an explanation.*

*Table 5: Instruction-following capability comparison using relative scores on LLaVA-Bench (In-the-Wild). The results are reported in the format of *mean* $\pm$ *std*. For the first three rows, we report three inference runs. LLaVA performs significantly better than others. ${}^{\dagger}$ For a given set of LLaVA decoding sequences, we evaluate by querying GPT-4 three times; GPT-4 gives a consistent evaluation.*

#### LLaVA-Bench (COCO).

We randomly select 30 images from COCO-Val-2014, and for each image, we generate three types of questions (conversation, detailed description, complex reasoning) using the proposed data generation pipeline in Sec. [3](#S3), totaling 90 questions. This benchmark studies the model’s alignment behavior and capabilities with consistent visual inputs.
We vary the training datasets to study the effectiveness of different types of instruction-following data, and show the results in Table [4](#S5.T4). First, with instruction tuning, the model’s ability of following user instructions improves significantly by over 50 points. Second, adding a small amount of detailed description and complex reasoning questions contributes to a considerable improvement of the model’s overall capability by 7 points. Furthermore, it also improves the model’s performance on conversational questions, suggesting that improvements in reasoning capabilities complement conversational abilities. Finally, we show that having all three types of data yields the best performance at 85.1%.

![Figure](extracted/5288275/figures/sample_bench_fridge.jpg)

#### LLaVA-Bench (In-the-Wild).

To evaluate the model’s capability in more challenging tasks and generalizability to novel domains, we collect a diverse set of 24 images with 60 questions in total, including indoor and outdoor scenes, memes, paintings, sketches, etc.*, and associate each image with a highly-detailed and manually-curated description and a proper selection of questions.
We compare LLaVA, BLIP, and OpenFlamingo in Table [5](#S5.T5). Thanks to visual instruction tuning, LLaVA achieves significantly better performance compared with BLIP-2 (+29%) and OpenFlamingo (+48%).
Compared to the text-only GPT-4 that has access to ground-truth labels, LLaVA achieves an impressive 81.7% performance on complex reasoning questions, with an overall score of 67.3%.

#### Limitations.

This LLaVA-Bench (In-the-Wild) is designed to be challenging and to reveal a model’s weaknesses. We provide two examples with associated captions and questions in Table [6](#S5.T6).
For the ramen example (left), to correctly answer the name of the restaurant, it requires the model to have a large knowledge coverage and multilingual understanding capability; to correctly describe the side dishes, the model may need to retrieve relevant multimodal information from Internet.
For the fridge example (right), perceiving the correct brand of the yogurt requires the model to process high resolution images and possess extensive knowledge coverage. We also observed an interesting failure of LLaVA, as it responds with *yes* when asked if strawberry-flavored yogurt is present, even though the fridge contains only yogurt *and* strawberries. This indicates that, at times, LLaVA perceives the image as a “bag of patches”, failing to grasp the complex semantics within the image.
We hope LLaVA serves as a solid baseline on the benchmarks, on which our findings can inspire future work in developing more capable LMMs.

### 5.2 ScienceQA

ScienceQA [[lu2022learn](#bib.bib34)] contains 21k multimodal
multiple choice questions with rich domain diversity across
3 subjects, 26 topics, 127 categories, and 379 skills. The
benchmark dataset is split into training, validation, and test splits with 12726, 4241, and 4241 examples, respectively. We consider two representative methods, including GPT-3.5 model (text-davinci-002) with and without chain-of-thought (CoT), LLaMA-Adapter [([zhang2023llama,](#bib.bib59))], as well as multimodal chain-of-thought (MM-CoT) [[zhang2023multimodal](#bib.bib61)], which is the current SoTA method on this dataset. For more baseline numbers, please see [[lu2022learn](#bib.bib34)].

The results are reported in Table [7](#S5.T7).
For LLaVA, we use the visual features before the last layer, ask the model to first predict reasons and then the answer, and train it for 12 epochs. It yields 90.92% accuracy, which is quite close to the SoTA 91.68%.
To explore the limit of LLMs, we also prompt GPT-4 using 2-shot in-context-learning and achieve 82.69% accuracy, which is a 7.52% absolute gain compared with 75.17% from GPT-3.5. For a substantial number of questions, we note that GPT-4 fails simply because it reports that there is insufficient context such as images or plots. We consider two schemes to combine the outcomes from our model and GPT-4.
$(i)$ A GPT-4 complement. Whenever GPT-4 fails to provide answers, we use the prediction from our method. This schemes yields 90.97% accuracy, which is almost the same as applying our method alone.
$(ii)$ GPT-4 as the judge. Whenever GPT-4 and LLaVA produce different answers, we prompt GPT-4 again, asking it to provide its own final answer based on the question and two outcomes. The spirit is similar with CoT, but with the external knowledge from the other model. Surprisingly, this scheme is able to provide consistent improvement over all question classes, and achieves a new SoTA accuracy of 92.53%. Interestingly, the text-only GPT-4, which cannot process images, improves the overall performance of the model on questions that have an image as context. This is because some of these questions do not actually require the image context for a correct answer. The GPT-4 judge can identify such cases and correct some of the errors that LLaVA makes. See the example in Appendix. To the best of our knowledge, this is the first time that GPT-4 is used for model ensembling. We hope this finding can encourage future research to explore more effective methods to leverage LLMs for model ensembling.

*Table 7: Accuracy (%) on Science QA dataset. Question categories: NAT = natural science, SOC = social science, LAN = language science, TXT = text context, IMG = image context, NO = no context, G1-6 = grades 1-6, G7-12 = grades 7-12. ${}^{\dagger}$Text-only GPT-4, our eval. Our novel model ensembling with the text-only GPT-4 consistently improves the model’s performance under all categories, setting the new SoTA performance.*

#### Ablations.

We ablate several design choices on ScienceQA in Table [8](#S5.T8).
$(i)$ Visual features. We tried using the last layer feature from CLIP vision encoder, which yields 89.96% and is 0.96% lower than the feature before the last layer. We hypothesize that this is because CLIP’s last layer features may focus more on global and abstract image properties compared to the layer before it, which can focus more on localized properties that are useful for understanding specific image details.
$(ii)$ Chain-of-thought. To decide the order between the answer and reasoning process in the model prediction, we run both variants and observe that answer-first reports the best number 89.77% accuracy in 12 epochs, while reasoning-first can quickly reach 89.77% accuracy in 6 epochs, but no further improvement with more training. Training the model for 24 epochs does not improve the performance. We conclude that CoT-like reasoning-first strategy can largely improve convergence, but contributes relatively little to the final performance.
$(iii)$ Pre-training. We skip pre-training and directly train on Science QA from scratch – performance drops to 85.81% accuracy. The 5.11% absolute degradation indicates the importance of our pre-training stage, in aligning multimodal features while preserving the vast pre-trained knowledge.
$(iv)$ Model size. We keep all configurations the same as our best 13B model, and train a 7B model. This yields 89.84% accuracy, which is 1.08% lower than 90.92%, demonstrating the importance of model scale.

## 6 Conclusion

This paper demonstrated the effectiveness of visual instruction tuning.
We presented an automatic pipeline to create language-image instruction-following data, based on which we train LLaVA, a multimodal model to follow human intent to complete visual tasks. It achieves the new SoTA accuracy when fine-tuned on ScienceQA, and excellent visual chat capabilities when fine-tuned on multimodal chat data.
Besides, we present the first benchmark to study multimodal instruction-following capability.
This paper is an initial step in visual instruction tuning, and mainly focuses on real-life tasks. For more quantitative results of LLaVA on academic benchmarks, please refer to the improved baselines with visual instruction tuning [[liu2023improvedllava](#bib.bib32)]. We hope our work can inspire future research on building more capable multimodal models.

Acknowledgements.
We thank Baolin Peng and Pan Lu for valuable discussions on instruction-tuning language models and Science QA, respectively.
We thank the LLaMA team for giving us access to their models, and open-source projects, including Alpaca and Vicuna.
This work was supported in part by NSF CAREER IIS2150012, and Institute of Information & communications Technology Planning & Evaluation(IITP) grants funded by the Korea government(MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration) and (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training).

## References

-
(1)

Langchain.

[https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain), 2022.

-
(2)

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr,
Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds,
et al.

Flamingo: a visual language model for few-shot learning.

arXiv preprint arXiv:2204.14198, 2022.

-
(3)

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko
Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel.

Vision-and-language navigation: Interpreting visually-grounded
navigation instructions in real environments.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, 2018.

-
(4)

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan,
Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al.

A general language assistant as a laboratory for alignment.

arXiv preprint arXiv:2112.00861, 2021.

-
(5)

Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong
Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon
Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig
Schmidt.

Openflamingo, March 2023.

-
(6)

Tim Brooks, Aleksander Holynski, and Alexei A Efros.

Instruct pix2pix: Learning to follow image editing instructions.

arXiv preprint arXiv:2211.09800, 2022.

-
(7)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.

Language models are few-shot learners.

Advances in neural information processing systems,
33:1877–1901, 2020.

-
(8)

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut.

Conceptual 12m: Pushing web-scale image-text pre-training to
recognize long-tail visual concepts.

In CVPR, 2021.

-
(9)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin
Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and
Eric P. Xing.

Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt
quality, March 2023.

-
(10)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra,
Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian
Gehrmann, et al.

Palm: Scaling language modeling with pathways.

arXiv preprint arXiv:2204.02311, 2022.

-
(11)

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus,
Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al.

Scaling instruction-finetuned language models.

arXiv preprint arXiv:2210.11416, 2022.

-
(12)

CVinW.

Computer vision in the wild.

[https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings](https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings),
2022.

-
(13)

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery,
Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al.

PaLM-E: An embodied multimodal language model.

arXiv preprint arXiv:2303.03378, 2023.

-
(14)

Fartash Faghri, Hadi Pouransari, Sachin Mehta, Mehrdad Farajtabar, Ali Farhadi,
Mohammad Rastegari, and Oncel Tuzel.

Reinforce data, multiply impact: Improved model accuracy and
robustness with dataset reinforcement.

arXiv preprint arXiv:2303.08983, 2023.

-
(15)

Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv
Taigman.

Make-a-scene: Scene-based text-to-image generation with human priors.

ArXiv, abs/2203.13131, 2022.

-
(16)

Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng Liu, Jianfeng Gao, et al.

Vision-language pre-training: Basics, recent advances, and future
trends.

Foundations and Trends® in Computer Graphics and
Vision, 2022.

-
(17)

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli.

Chatgpt outperforms crowd-workers for text-annotation tasks.

arXiv preprint arXiv:2303.15056, 2023.

-
(18)

Tanmay Gupta and Aniruddha Kembhavi.

Visual programming: Compositional visual reasoning without training.

arXiv preprint arXiv:2211.11559, 2022.

-
(19)

Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, and Jianfeng Gao.

Towards learning a generic agent for vision-and-language navigation
via pre-training.

In CVPR, 2020.

-
(20)

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma,
Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al.

Language is not all you need: Aligning perception with language
models.

arXiv preprint arXiv:2302.14045, 2023.

-
(21)

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas
Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John
Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt.

Openclip.

July 2021.

If you use this software, please cite it as below.

-
(22)

Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov,
Dániel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh
Koura, et al.

Opt-iml: Scaling language model instruction meta learning through the
lens of generalization.

arXiv preprint arXiv:2212.12017, 2022.

-
(23)

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath
Hariharan, and Ser-Nam Lim.

Visual prompt tuning.

In ECCV, 2022.

-
(24)

Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried.

Grounding language models to images for multimodal generation.

arXiv preprint arXiv:2301.13823, 2023.

-
(25)

Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René
Ranftl.

Language-driven semantic segmentation.

ICLR, 2022.

-
(26)

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and
Jianfeng Gao.

Multimodal foundation models: From specialists to general-purpose
assistants.

arXiv preprint arXiv:2309.10020, 2023.

-
(27)

Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja,
Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng
Gao.

ELEVATER: A benchmark and toolkit for evaluating language-augmented
visual models.

In NeurIPS Track on Datasets and Benchmarks, 2022.

-
(28)

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.

Blip-2: Bootstrapping language-image pre-training with frozen image
encoders and large language models.

arXiv preprint arXiv:2301.12597, 2023.

-
(29)

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li,
Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al.

Grounded language-image pre-training.

In CVPR, 2022.

-
(30)

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao,
Chunyuan Li, and Yong Jae Lee.

Gligen: Open-set grounded text-to-image generation.

arXiv preprint arXiv:2301.07093, 2023.

-
(31)

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva
Ramanan, Piotr Dollár, and C Lawrence Zitnick.

Microsoft COCO: Common objects in context.

In ECCV, 2014.

-
(32)

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.

Improved baselines with visual instruction tuning, 2023.

-
(33)

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan
Li, Jianwei Yang, Hang Su, Jun Zhu, et al.

Grounding dino: Marrying dino with grounded pre-training for open-set
object detection.

arXiv preprint arXiv:2303.05499, 2023.

-
(34)

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu,
Oyvind Tafjord, Peter Clark, and Ashwin Kalyan.

Learn to explain: Multimodal reasoning via thought chains for science
question answering.

Advances in Neural Information Processing Systems, 2022.

-
(35)

OpenAI.

ChatGPT.

[https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/), 2023.

-
(36)

OpenAI.

Gpt-4 technical report, 2023.

-
(37)

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

Training language models to follow instructions with human feedback.

Advances in Neural Information Processing Systems,
35:27730–27744, 2022.

-
(38)

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao.

Instruction tuning with GPT-4.

arXiv preprint arXiv:2304.03277, 2023.

-
(39)

Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei
Yu, Jiahui Yu, Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, et al.

Combined scaling for open-vocabulary image classification.

arXiv preprint arXiv: 2111.10050, 2021.

-
(40)

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.

Learning transferable visual models from natural language
supervision.

arXiv preprint arXiv:2103.00020, 2021.

-
(41)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J Liu.

Exploring the limits of transfer learning with a unified text-to-text
transformer.

The Journal of Machine Learning Research, 2020.

-
(42)

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

Hierarchical text-conditional image generation with clip latents.

ArXiv, abs/2204.06125, 2022.

-
(43)

Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn
Ommer.

High-resolution image synthesis with latent diffusion models.

CVPR, pages 10674–10685, 2022.

-
(44)

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L.
Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara
Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet,
and Mohammad Norouzi.

Photorealistic text-to-image diffusion models with deep language
understanding.

ArXiv, abs/2205.11487, 2022.

-
(45)

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross
Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell
Wortsman, et al.

Laion-5b: An open large-scale dataset for training next generation
image-text models.

arXiv preprint arXiv:2210.08402, 2022.

-
(46)

Dídac Surís, Sachit Menon, and Carl Vondrick.

Vipergpt: Visual inference via python execution for reasoning.

arXiv preprint arXiv:2303.08128, 2023.

-
(47)

Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John
Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets,
Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech
Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis
Savva, and Dhruv Batra.

Habitat 2.0: Training home assistants to rearrange their habitat.

In Advances in Neural Information Processing Systems (NeurIPS),
2021.

-
(48)

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
Guestrin, Percy Liang, and Tatsunori B. Hashimoto.

Stanford alpaca: An instruction-following llama model.

[https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.

-
(49)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric
Hambro, Faisal Azhar, et al.

Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971, 2023.

-
(50)

Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan,
Zicheng Liu, Ce Liu, and Lijuan Wang.

Git: A generative image-to-text transformer for vision and language.

arXiv preprint arXiv:2205.14100, 2022.

-
(51)

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel
Khashabi, and Hannaneh Hajishirzi.

Self-instruct: Aligning language model with self generated
instructions.

arXiv preprint arXiv:2212.10560, 2022.

-
(52)

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza
Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva
Naik, David Stap, et al.

Benchmarking generalization via in-context instructions on 1,600+
language tasks.

arXiv preprint arXiv:2204.07705, 2022.

-
(53)

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan
Duan.

Visual chatgpt: Talking, drawing and editing with visual foundation
models.

arXiv preprint arXiv:2303.04671, 2023.

-
(54)

Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Lu Yuan, Ce Liu, and
Jianfeng Gao.

Unified contrastive learning in image-text-label space.

CVPR, 2022.

-
(55)

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal
Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang.

Mm-react: Prompting chatgpt for multimodal reasoning and action.

arXiv preprint arXiv:2303.11381, 2023.

-
(56)

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang,
Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Benton C.
Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and
Yonghui Wu.

Scaling autoregressive models for content-rich text-to-image
generation.

ArXiv, abs/2206.10789, 2022.

-
(57)

Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao,
Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al.

Florence: A new foundation model for computer vision.

arXiv preprint arXiv:2111.11432, 2021.

-
(58)

Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao, Jianwei
Yang, and Lei Zhang.

A simple framework for open-vocabulary segmentation and detection.

arXiv preprint arXiv:2303.08131, 2023.

-
(59)

Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu,
Hongsheng Li, Peng Gao, and Yu Qiao.

Llama-adapter: Efficient fine-tuning of language models with
zero-init attention.

arXiv preprint arXiv:2303.16199, 2023.

-
(60)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al.

OPT: Open pre-trained transformer language models.

arXiv preprint arXiv:2205.01068, 2022.

-
(61)

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola.

Multimodal chain-of-thought reasoning in language models.

arXiv preprint arXiv:2302.00923, 2023.

-
(62)

Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella,
Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al.

Regionclip: Region-based language-image pretraining.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 16793–16803, 2022.

-
(63)

Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang
Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al.

Generalized decoding for pixel, image, and language.

arXiv preprint arXiv:2212.11270, 2022.

## Appendix A Broader Impact

The broader impact of LLaVA, a general-purpose visual assistant, has potential benefits and risks associated with its deployment and release. Some considerations are unique to LLaVA due to its visual nature, while others share similarities with existing instruction-following LLMs (*e.g., *Alpaca, Vicuna, *etc.*). As LLaVA is built upon LLaMA, Vicuna, and CLIP, it inherits some of the issues associated with LLMs and vision encoders. In the following, we outline both the risks and mitigation strategies in place for the release of this model.

#### Malicious input.

To minimize potential misuse and harmful consequences, we employ two precautionary measures for LLaVA: (1) *OpenAI Filter API* for user input text to prevent harmful or inappropriate text instructions from being processed by the model, and (2) *NSFW Filter* for uploaded user images to detect and block Not Safe For Work (NSFW) content or any other potentially harmful image inputs.

#### Hallucination.

Similar to LLMs, LLaVA might generate outputs that aren’t grounded in facts or input data. This raises concerns about inferences made, especially in critical applications (*e.g., *medical).

#### Biases.

Bias can be transferred from the base models to LLaVA, both from the vision encoder (CLIP) and the language decoder (LLaMA/Vicuna). This may lead to biased outcomes or unfair representations of diverse content.

#### Energy consumption.

Though energy consumption is not a primary concern for LLaVA due to a smaller pretraining dataset (see details in Sec. [C](#A3)), it may become a concern when scaling up the pretraining dataset or increasing the model size, e.g., to a larger LLaMA version like the 65B model.

#### Evaluation complexities.

Assessing the performance of LLaVA is challenging as it involves both language and visual tasks. Our evaluation benchmark covers several aspects, including accuracy, concept coverage, reasoning ability, and creativity. However, additional aspects need consideration, such as the degree of visual content hallucination and fine-grained understanding of visual content. While text-only GPT-4 based multimodal evaluation is consistent and accurate in our study, its robustness in different situations and capability to evaluate other unexplored aspects are subjects for future work.

Despite these risks, we believe that the benefits of releasing LLaVA to the research community outweigh the potential harm. It allows for ongoing investigation and improvement of the model and engages the community in developing better mitigation strategies to address these concerns. Moreover, the release of LLaVA can spur the development of new applications and research directions, ultimately contributing to the progress and responsible deployment of foundation models in vision-language tasks.

## Appendix B More Results

We present more qualitative results of LLaVA to analyze its emergent behaviors and observed weaknesses. For more quantitative results of LLaVA on academic benchmarks, please refer to the improved baselines with visual instruction tuning [[[32](#bib.bib32)]]. In Table [9](#A2.T9), LLaVA demonstrates a similar behavior as GPT-4 in another example from its paper. Similar to the GPT-4 live demo by OpenAI, LLaVA is capable of generating the HTML/JS/CSS code for an interactive joke website based on a simplified user input sketch in Fig. [2](#A2.F2), despite a minor error. As shown in Fig. [3](#A2.F3), LLaVA can follow user’s instructions in a conversational style and provide detailed responses or creative writings. Furthermore, LLaVA is able to relate the visual content to the textual knowledge from the pretrained LLM, as demonstrated in Fig. [4](#A2.F4) and Fig. [5](#A2.F5).

One interesting emergent behavior of LLaVA is that it is able to understand visual contents that are not covered in the training. For example, in Fig. [6](#A2.F6), it is able to recognize Elon Musk both in a headshot and in a humorous meme where he is dressed as a doge, even though Elon Musk *never* appears in the training data for either the visual feature alignment or visual instruction tuning stages of LLaVA. LLaVA also demonstrates impressive OCR (optical character recognition) ability in Table [9](#A2.T9) and Fig. [2](#A2.F2), which is rarely covered in our training data.

We hope these additional results and observations showcase the potential of LLaVA in various application areas. In future work, it is important to investigate these emergent behaviors more thoroughly and to understand the underlying mechanisms that enable LLaVA to demonstrate such generalization abilities. This will pave the way towards building better LMMs, including enhancing robustness, reducing biases, and improving the alignment and the scope of the learned vision-language representations.

![Figure](x2.png)

*Figure 2: LLaVA generates HTML/JS code for an interactive website based on user sketch inputs. The interactive interface works after fixing a minor error (in red*) in the generated output. There is room for improvement in LLaVA’s output, such as splitting the joke and punchline into two rows, and only revealing the punchline upon button click, to better reflect the user’s intent.*

![Figure](x4.png)

*Figure 3: LLaVA is capable of recognizing the visual content following the user’s intent, without directly prompting for visual recognition. It also provides a detailed response when prompted with a follow-up request, and the generated response is closely related to the provided visual content.*

![Figure](x5.png)

*Figure 4: LLaVA relates the movie scenes to the textual knowledge from the pretrained LLM.*

![Figure](x6.png)

*Figure 5: LLaVA recognizes the famous art work, Mona Lisa, by Leonardo da Vinci. When we start a new conversation, it also explains the humourous artwork created on the web, mimicking the Mona Lisa.*

![Figure](x7.png)

*Figure 6: An interesting emergent behavior of LLaVA is its ability to recognize Elon Musk both in a headshot and in a humorous meme where he is dressed as a doge. This implies that the pre-trained CLIP vision encoder may have seen images of Elon Musk. However, it is still surprising because Elon Musk never* appears in the training data for either the visual feature alignment or visual instruction tuning stages of LLaVA, which indicates that the base language model generalizes to unseen visual concepts.*

![Figure](extracted/5288275/figures/sqa_judge_chair.png)

*Table 10: One example on how the text-only GPT-4 acts as a judge to ensemble the predictions from LLaVA and a text-only GPT-4, and gives a correct final answer.*

## Appendix C Training Details

We pre-train our model on the filtered CC-595K subset for 1 epoch with a learning rate of 2e-3 and a batch size of 128, and fine-tune on the proposed LLaVA-Instruct-158K dataset for 3 epochs, with a learning rate of 2e-5 and a batch size of 32. Following Vicuna, we use the Adam optimizer with no weight decay and a cosine learning rate with a warmup ratio of 3%. During finetuning, FSDP (Full Shard Data Parallel) and gradient checkpointing is used to save GPU memory, and offloading is not used. BF16 and TF32 are enabled to achieve a balance between speed and precision.

We train all models with 8$\times$ A100s. Pretraining on CC-595K completes within 4 hours. Finetuning on Instruct-158K completes within 10 hours. Finetuning on ScienceQA completes within 4 hours.

## Appendix D Assets

Our source code, generated instruction-tuning data, proposed benchmark are uploaded to the anonymized GitHub repository: [LLaVA-Annonymous/LLaVA](https://github.com/LLaVA-Annonymous/LLaVA).

-
1.

Source Code: [link](https://github.com/LLaVA-Annonymous/LLaVA)

-
2.

README: [link](https://github.com/LLaVA-Annonymous/LLaVA)

-
3.

Instructions to launch the demo: [link](https://github.com/LLaVA-Annonymous/LLaVA#web-ui)

-
4.

All prompts and few shot examples for querying GPT-4: [link](https://github.com/LLaVA-Annonymous/LLaVA/tree/master/playground/data/prompts)

-
5.

LLaVA-Instruct-158K: [link](https://github.com/LLaVA-Annonymous/LLaVA/blob/master/playground/data/llava_instruct_150k.json)

-
6.

LLaVA-Bench: [COCO](https://github.com/LLaVA-Annonymous/LLaVA/blob/master/playground/data/coco2014_val_gpt4_qa_30x3.jsonl), [In-The-Wild](https://github.com/LLaVA-Annonymous/LLaVA/tree/master/playground/data/llava_bench_in_the_wild)

-
7.

Model checkpoints. The size of the model checkpoints after compression is 25GB, which exceeds the 5GB limit of GitHub LFS (Large File Storage). We’ll release the checkpoint to the public, or upon request with reviewers for this submission.

## Appendix E Data

#### Instructions for brief image description.

The list of instructions used to briefly describe the image content are shown in Table [11](#A5.T11). They present the same meaning with natural language variance.

*Table 11: The list of instructions for brief image description.*

#### Instructions for detailed image description.

The list of instructions used to describe the image content in detail are shown in Table [12](#A5.T12). They present the same meaning with natural language variance.

*Table 12: The list of instructions for detailed image description.*

#### CC3M.

We extract noun-phrases using Spacy for each caption over the whole CC3M dataset,
and count the frequency of each unique noun-phrase. We skip noun-phrases whose frequency is smaller than $3$, as they are usually rare combinations concept and attributes that has already been covered by other captions. We then start from the noun-phrases with lowest remaining frequency, add the captions that contain this noun-phrase to the candidate pool. If the frequency of the noun-phrase is larger than $100$, we randomly choose a subset of size $100$ out of all its captions. This results in around 595K image-text pairs.

The comparison of noun-phrase statistics before and after filtering CC3M is shown in Figure [7](#A5.F7). The filtered dataset shows a good coverage of concepts whose frequency is higher from 3, but with a smaller number of image-text pairs.

![Figure](x8.png)

*Figure 7: Comparison of noun-phrase statistics before and after filtering CC3M. The total number of unique noun-phrases are reported in the legend.*

## Appendix F Prompts

The prompt used to generate image-based conversation from ChatGPT/GPT-4 is shown in Table [13](#A6.T13).

*Table 13: For each query, we illustrate the prompt construction process for ChatGPT/GPT-4 to collect query[‘response’] from query[‘context’], using few-shot in-context-learning, where examples are from fewshot_samples, each example including input sample[‘context’] and output sample[‘response’]. Note that messages is the final prompt. In this example, we provide the prompt used to generate the conversation response, please see also see its in-context-learning examples in Table [15](#A6.T15) and Table [16](#A6.T16) for details. We recommend readers to check out the codebase for the prompts to generated two other types of responses, including detailed decription and complex reasoning.*

![Figure](extracted/5288275/figures/car_bbox.jpg)

*Table 14: One example to illustrate the instruction-following data. The top block shows the contexts such as captions and boxes used to prompt GPT, and the bottom block shows the three types of responses. Note that the visual image is not used to prompt GPT, we only show it here as a reference.*

*Table 15: One example used in in-context-learning to construct visual conversation data.*

*Table 16: One example used in in-context-learning to construct visual conversation data.*

Generated on Thu Feb 15 19:20:46 2024 by [LATExml](http://dlmf.nist.gov/LaTeXML/)