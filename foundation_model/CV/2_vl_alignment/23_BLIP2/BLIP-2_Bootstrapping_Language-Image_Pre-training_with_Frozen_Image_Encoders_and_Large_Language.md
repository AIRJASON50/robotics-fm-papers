##### Report GitHub Issue

**×




Title:
*


Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2301.12597v3 [cs.CV] 15 Jun 2023

[# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models Junnan Li Dongxu Li Silvio Savarese Steven Hoi Salesforce Research ###### Abstract The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions. vision-language https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## 1 Introduction

Vision-language pre-training (VLP) research has witnessed a rapid advancement in the past few years,
where pre-trained models with increasingly larger scale have been developed to continuously push the state-of-the-art on various downstream tasks [(Radford et al., [2021](#bib.bib30); Li et al., [2021](#bib.bib21), [2022](#bib.bib22); Wang et al., [2022a](#bib.bib36); Alayrac et al., [2022](#bib.bib2); Wang et al., [2022b](#bib.bib38))].
However, most state-of-the-art vision-language models incur a high computation cost during pre-training, due to
end-to-end training using large-scale models and datasets.

![Figure](2301.12597v3/x1.png)

*Figure 1: Overview of BLIP-2’s framework. We pre-train a lightweight Querying Transformer following a two-stage strategy to bridge the modality gap. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen LLM, which enables zero-shot instructed image-to-text generation (see Figure [4](#S3.F4) for more examples).*

Vision-language research sits at the intersection between vision and language,
therefore it is naturally expected that vision-language models can harvest from the readily-available unimodal models from the vision and natural language communities.
In this paper, we propose a generic and compute-efficient VLP method by
bootstrapping from off-the-shelf pre-trained vision models and language models.
Pre-trained vision models offer high-quality visual representation.
Pre-trained language models, in particular large language models (LLMs), offer strong language generation and zero-shot transfer abilities.
To reduce computation cost and counteract the issue of catastrophic forgetting, the unimodal pre-trained models remain frozen during the pre-training.

In order to leverage pre-trained unimodal models for VLP, it is key to facilitate cross-modal alignment.
However, since LLMs have not seen images during their unimodal pre-training,
freezing them makes vision-language alignment in particular challenging.
In this regard, existing methods (e.g. Frozen [(Tsimpoukelli et al., [2021](#bib.bib35))], Flamingo [(Alayrac et al., [2022](#bib.bib2))]) resort to an image-to-text generation loss,
which we show is insufficient to bridge the modality gap.

To achieve effective vision-language alignment with frozen unimodal models, we propose a Querying Transformer (Q-Former) pre-trained with a new two-stage pre-training strategy.
As shown in Figure [1](#S1.F1),
Q-Former is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder.
It acts as an information bottleneck between the frozen image encoder and the frozen LLM,
where it feeds the most useful visual feature for the LLM to output the desired text.
In the first pre-training stage,
we perform vision-language representation learning
which enforces the Q-Former to learn visual representation most relevant to the text.
In the second pre-training stage,
we perform vision-to-language generative learning by connecting the output of the Q-Former to a frozen LLM,
and trains the Q-Former such that its output visual representation can be interpreted by the LLM.

We name our VLP framework as BLIP-2: Bootstrapping Language-Image Pre-training with frozen unimodal models.
The key advantages of BLIP-2 include:

-
•

BLIP-2 effectively leverages both frozen pre-trained image models and language models.
We bridge the modality gap using a Q-Former pre-trained in two-stages: representation learning stage and generative learning stage.
BLIP-2 achieves state-of-the-art performance on various vision-language tasks including visual question answering, image captioning, and image-text retrieval.

-
•

Powered by LLMs (e.g. OPT [(Zhang et al., [2022](#bib.bib45))], FlanT5 [(Chung et al., [2022](#bib.bib9))]), BLIP-2 can be prompted to perform zero-shot image-to-text generation that follows natural language instructions, which enables emerging capabilities such as visual knowledge reasoning, visual conversation, etc. (see Figure [4](#S3.F4) for examples).

-
•

Due to the use of frozen unimodal models and a lightweight Q-Former,
BLIP-2 is more compute-efficient than exisiting state-of-the-arts.
For example, BLIP-2 outperforms Flamingo [(Alayrac et al., [2022](#bib.bib2))] by 8.7% on zero-shot VQAv2, while using 54$\times$ fewer trainable parameters.
Furthermore,
our results show that BLIP-2 is a generic method that can harvest more advanced unimodal models for better VLP performance.

## 2 Related Work

### 2.1 End-to-end Vision-Language Pre-training

Vision-language pre-training aims to learn multimodal foundation models with improved performance on various vision-and-language tasks.
Depending on the downstream task, different model architectures have been proposed, including the dual-encoder architecture [(Radford et al., [2021](#bib.bib30); Jia et al., [2021](#bib.bib18))], the fusion-encoder architecture [(Tan & Bansal, [2019](#bib.bib33); Li et al., [2021](#bib.bib21))], the encoder-decoder architecture [(Cho et al., [2021](#bib.bib8); Wang et al., [2021b](#bib.bib39); Chen et al., [2022b](#bib.bib6))],
and more recently, the unified transformer architecture [(Li et al., [2022](#bib.bib22); Wang et al., [2022b](#bib.bib38))].
Various pre-training objectives have also been proposed over the years,
and have progressively converged to a few time-tested ones: image-text contrastive learning [(Radford et al., [2021](#bib.bib30); Yao et al., [2022](#bib.bib40); Li et al., [2021](#bib.bib21), [2022](#bib.bib22))],
image-text matching [(Li et al., [2021](#bib.bib21), [2022](#bib.bib22); Wang et al., [2021a](#bib.bib37))],
and (masked) language modeling [(Li et al., [2021](#bib.bib21), [2022](#bib.bib22); Yu et al., [2022](#bib.bib41); Wang et al., [2022b](#bib.bib38))].

Most VLP methods perform end-to-end pre-training using large-scale image-text pair datasets.
As the model size keeps increasing, the pre-training can incur an extremely high computation cost.
Moreover, it is inflexible for end-to-end pre-trained models to leverage readily-available unimodal pre-trained models,
such as LLMs [(Brown et al., [2020](#bib.bib3); Zhang et al., [2022](#bib.bib45); Chung et al., [2022](#bib.bib9))].

### 2.2 Modular Vision-Language Pre-training

More similar to us are methods that leverage off-the-shelf pre-trained models and keep them frozen during VLP.
Some methods freeze the image encoder, including the early work which adopts a frozen object detector to extract visual features [(Chen et al., [2020](#bib.bib7); Li et al., [2020](#bib.bib23); Zhang et al., [2021](#bib.bib44))],
and the recent LiT [(Zhai et al., [2022](#bib.bib43))] which uses a frozen pre-trained image encoder for CLIP [(Radford et al., [2021](#bib.bib30))] pre-training.
Some methods freeze the language model to use the knowledge from LLMs for vision-to-language generation tasks [(Tsimpoukelli et al., [2021](#bib.bib35); Alayrac et al., [2022](#bib.bib2); Chen et al., [2022a](#bib.bib5); Mañas et al., [2023](#bib.bib26); Tiong et al., [2022](#bib.bib34); Guo et al., [2022](#bib.bib15))].
The key challenge in using a frozen LLM is to align visual features to the text space.
To achieve this, Frozen [(Tsimpoukelli et al., [2021](#bib.bib35))] finetunes an image encoder whose outputs are directly used as soft prompts for the LLM.
Flamingo [(Alayrac et al., [2022](#bib.bib2))] inserts new cross-attention layers into the LLM to inject visual features,
and pre-trains the new layers on billions of image-text pairs.
Both methods adopt the language modeling loss,
where the language model generates texts conditioned on the image.

Different from existing methods,
BLIP-2 can effectively and efficiently leverage both frozen image encoders and frozen LLMs for various vision-language tasks,
achieving stronger performance at a lower computation cost.

## 3 Method

![Figure](2301.12597v3/x2.png)

*Figure 2: (Left) Model architecture of Q-Former and BLIP-2’s first-stage vision-language representation learning objectives. We jointly optimize three objectives which enforce the queries (a set of learnable embeddings) to extract visual representation most relevant to the text. (Right) The self-attention masking strategy for each objective to control query-text interaction.*

We propose BLIP-2, a new vision-language pre-training method that bootstraps from frozen pre-trained unimodal models.
In order to bridge the modality gap,
we propose a Querying Transformer (Q-Former) pre-trained in two stages:
(1) vision-language representation learning stage with a frozen image encoder and (2) vision-to-language generative learning stage with a frozen LLM.
This section first introduces the model architecture of Q-Former,
and then delineates the two-stage pre-training procedures.

### 3.1 Model Architecture

We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM.
It extracts a fixed number of output features from the image encoder,
independent of input image resolution.
As shown in Figure [2](#S3.F2),
Q-Former consists of two transformer submodules that share the same self-attention layers: (1) an image transformer that interacts with the frozen image encoder for visual feature extraction, (2) a text transformer that can function as both a text encoder and a text decoder.
We create a set number of learnable query embeddings as input to the image transformer.
The queries interact with each other through self-attention layers, and interact with frozen image features through cross-attention layers (inserted every other transformer block).
The queries can additionally interact with the text through the same self-attention layers.
Depending on the pre-training task,
we apply different self-attention masks to control query-text interaction.
We initialize Q-Former with the pre-trained weights of BERT${}_{\text{base}}$ [(Devlin et al., [2019](#bib.bib11))], whereas the cross-attention layers are randomly initialized.
In total, Q-Former contains 188M parameters.
Note that the queries are considered as model parameters.

In our experiments,
we use 32 queries where each query has a dimension of 768 (same as the hidden dimension of the Q-Former).
We use $Z$ to denote the output query representation.
The size of $Z$ ($32\times 768$) is much smaller than the size of frozen image features (e.g. $257\times 1024$ for ViT-L/14).
This bottleneck architecture works together with our pre-training objectives into forcing the queries to extract visual information that is most relevant to the text.

### 3.2 Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

In the representation learning stage,
we connect Q-Former to a frozen image encoder and perform pre-training using image-text pairs.
We aim to train the Q-Former such that the queries can learn to extract visual representation that is most informative of the text.
Inspired by BLIP [(Li et al., [2022](#bib.bib22))], we jointly optimize three pre-training objectives that share the same input format and model parameters.
Each objective employs a different attention masking strategy between queries and text to control their interaction (see Figure [2](#S3.F2)).

Image-Text Contrastive Learning (ITC) learns to align image representation and text representation such that their mutual information is maximized.
It achieves so by contrasting the image-text similarity of a positive pair against those of negative pairs.
We align the output query representation $Z$ from the image transformer with the text representation $t$ from the text transformer,
where $t$ is the output embedding of the [CLS] token.
Since $Z$ contains multiple output embeddings (one from each query),
we first compute the pairwise similarity between each query output and $t$, and then select the highest one as the image-text similarity.
To avoid information leak, we employ a unimodal self-attention mask,
where the queries and text are not allowed to see each other.
Due to the use of a frozen image encoder, we can fit more samples per GPU compared to end-to-end methods.
Therefore, we use in-batch negatives instead of the momentum queue in BLIP.

Image-grounded Text Generation (ITG) loss trains the Q-Former to generate texts, given input images as the condition.
Since the architecture of Q-Former does not allow direct interactions between the frozen image encoder and the text tokens,
the information required for generating the text must be first extracted by the queries,
and then passed to the text tokens via self-attention layers.
Therefore, the queries are forced to extract visual features that capture all the information about the text.
We employ a multimodal causal self-attention mask to control query-text interaction, similar to the one used in UniLM [(Dong et al., [2019](#bib.bib12))].
The queries can attend to each other but not the text tokens.
Each text token can attend to all queries and its previous text tokens.
We also replace the [CLS] token with a new [DEC] token as the first text token to signal the decoding task.

Image-Text Matching (ITM)
aims to learn fine-grained alignment between image and text representation.
It is a binary classification task where the model is asked to predict whether an image-text pair is positive
(matched) or negative (unmatched).
We use a bi-directional self-attention mask where all queries and texts can attend to each other.
The output query embeddings $Z$ thus capture multimodal information.
We feed each output query embedding into a two-class linear classifier to obtain a logit,
and average the logits across all queries as the output matching score.
We adopt the hard negative mining strategy from [Li et al. ([2021](#bib.bib21), [2022](#bib.bib22))] to create informative negative pairs.

![Figure](2301.12597v3/x3.png)

*Figure 3: BLIP-2’s second-stage vision-to-language generative pre-training, which bootstraps from frozen large language models (LLMs). (Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5). The fully-connected layer adapts from the output dimension of the Q-Former to the input dimension of the chosen LLM.*

### 3.3 Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

In the generative pre-training stage,
we connect Q-Former (with the frozen image encoder attached) to a frozen LLM to harvest the LLM’s generative language capability.
As shown in Figure [3](#S3.F3),
we use a fully-connected (FC) layer to linearly project the output query embeddings $Z$ into the same dimension as the text embedding of the LLM.
The projected query embeddings are then prepended to the input text embeddings.
They function as soft visual prompts that condition the LLM on visual representation extracted by the Q-Former.
Since the Q-Former has been pre-trained to extract language-informative visual representation,
it effectively functions as an information bottleneck that feeds the most useful information to the LLM while removing irrelevant visual information.
This reduces the burden of the LLM to learn vision-language alignment,
thus mitigating the catastrophic forgetting problem.

We experiment with two types of LLMs: decoder-based LLMs and encoder-decoder-based LLMs.
For decoder-based LLMs,
we pre-train with the language modeling loss,
where the frozen LLM is tasked to generate the text conditioned on the visual representation from Q-Former.
For encoder-decoder-based LLMs,
we pre-train with the prefix language modeling loss,
where we split a text into two parts.
The prefix text is concatenated with the visual representation as input to the LLM’s encoder.
The suffix text is used as the generation target for the LLM’s decoder.

### 3.4 Model Pre-training

Pre-training data.
We use the same pre-training dataset as BLIP with 129M images in total,
including COCO [(Lin et al., [2014](#bib.bib24))], Visual Genome [(Krishna et al., [2017](#bib.bib20))], CC3M [(Sharma et al., [2018](#bib.bib32))], CC12M [(Changpinyo et al., [2021](#bib.bib4))], SBU [(Ordonez et al., [2011](#bib.bib28))], and 115M images from the LAION400M dataset [(Schuhmann et al., [2021](#bib.bib31))].
We adopt the CapFilt method [(Li et al., [2022](#bib.bib22))] to create synthetic captions for the web images.
Specifically, we generate 10 captions using the BLIPlarge captioning model, and rank the synthetic captions along with the original web caption based on the image-text similarity produced by a CLIP ViT-L/14 model.
We keep top-two captions per image as training data and randomly sample one at each pre-training step.

Pre-trained image encoder and LLM.
For the frozen image encoder, we explore two state-of-the-art pre-trained vision transformer models: (1) ViT-L/14 from CLIP [(Radford et al., [2021](#bib.bib30))] and (2) ViT-g/14 from EVA-CLIP [(Fang et al., [2022](#bib.bib13))].
We remove the last layer of the ViT and uses the second last layer’s output features,
which leads to slightly better performance.
For the frozen language model,
we explore the unsupervised-trained OPT model family [(Zhang et al., [2022](#bib.bib45))] for decoder-based LLMs,
and the instruction-trained FlanT5 model family [(Chung et al., [2022](#bib.bib9))] for encoder-decoder-based LLMs.

![Figure](2301.12597v3/x4.png)

*Figure 4: Selected examples of instructed zero-shot image-to-text generation using a BLIP-2 model w/ ViT-g and FlanT5${}_{\text{XXL}}$, where it shows a wide range of capabilities including visual conversation, visual knowledge reasoning, visual commensense reasoning, storytelling, personalized image-to-text generation, etc.*

Pre-training settings.
We pre-train for 250k steps in the first stage and 80k steps in the second stage.
We use a batch size of 2320/1680 for ViT-L/ViT-g in the first stage and a batch size of 1920/1520 for OPT/FlanT5 in the second stage.
During pre-training, we convert the frozen ViTs’ and LLMs’ parameters into FP16, except for FlanT5 where we use BFloat16.
We found no performance degradation compared to using 32-bit models.
Due to the use of frozen models,
our pre-training is more computational friendly than existing large-scale VLP methods.
For example,
using a single 16-A100(40G) machine,
our largest model with ViT-g and FlanT5-XXL requires less than 6 days for the first stage and less than 3 days for the second stage.

The same set of pre-training hyper-parameters are used for all models.
We use the AdamW [(Loshchilov & Hutter, [2017](#bib.bib25))] optimizer with $\beta_{1}=0.9$, $\beta_{1}=0.98$, and a weight decay of 0.05.
We use a cosine learning rate decay with a peak learning rate of 1e-4 and a linear warmup of 2k steps.
The minimum learning rate at the second stage is 5e-5.
We use images of size 224$\times$224, augmented with random resized cropping and horizontal flipping.

## 4 Experiment

*Table 1: Overview of BLIP-2 results on various zero-shot vision-language tasks. Compared with previous state-of-the-art models. BLIP-2 achieves the highest zero-shot performance while requiring the least number of trainable parameters during vision-language pre-training.*

*Table 2: Comparison with state-of-the-art methods on zero-shot visual question answering.*

Table 1 provides an overview of the performance of BLIP-2 on various zero-shot vision-language tasks.
Compared to previous state-of-the-art models,
BLIP-2 achieves improved performance while requiring substantially fewer number of trainable parameters during vision-language pre-training.

### 4.1 Instructed Zero-shot Image-to-Text Generation

BLIP-2 effectively enables a LLM to understand images while preserving its capability in following text prompts,
which allows us to control image-to-text generation with instructions.
We simply append the text prompt after the visual prompt as input to the LLM.
Figure [4](#S3.F4) shows examples to demonstrate a wide range of zero-shot image-to-text capabilities including visual knowledge reasoning, visual commensense reasoning,
visual conversation,
personalized image-to-text generation, etc.

Zero-shot VQA.
We perform quantitative evaluation on the zero-shot visual question answering task.
For OPT models, we use the prompt “Question: {} Answer:”.
For FlanT5 models, we use the prompt “Question: {} Short answer:”.
During generation,
we use beam search with a beam width of 5.
We also set the length-penalty to -1 which encourages shorter answers that align better with human annotation.

As shown in Table [2](#S4.T2).
BLIP-2 achieves state-of-the-art result on the VQAv2 [(Goyal et al., [2017](#bib.bib14))] and GQA [(Hudson & Manning, [2019](#bib.bib17))] datasets.
It outperforms Flamingo80B by 8.7% on VQAv2, despite having 54x fewer trainable parameters.
On the OK-VQA [(Marino et al., [2019](#bib.bib27))] dataset,
BLIP-2 comes secondary to Flamingo80B.
We hypothesis that this is because OK-VQA focuses more on open-world knowledge than visual understanding,
and the 70B Chinchilla [(Hoffmann et al., [2022](#bib.bib16))] language model from Flamingo80B possesses more knowledge than the 11B FlanT5${}_{\text{XXL}}$.

We make a promising observation from Table [2](#S4.T2):
a stronger image encoder or a stronger LLM both lead to better performance.
This observation is supported by several facts:
(1) ViT-g outperforms ViT-L for both OPT and FlanT5.
(2) Within the same LLM family, larger models outperform smaller ones.
(3) FlanT5, an instruction-tuned LLM, outperforms the unsupervised-trained OPT on VQA.
This observation validates BLIP-2 as a generic vision-language pre-training method that can efficiently harvest the rapid advances in vision and natural language communities.

Effect of Vision-Language Representation Learning.

![Figure](2301.12597v3/x5.png)
![Figure](2301.12597v3/x6.png)

*Figure 5: Effect of vision-language representation learning on vision-to-language generative learning. Without representation learning, the Q-Former fails the bridge the modality gap, leading to significantly lower performance on zero-shot VQA.*

The first-stage representation learning pre-trains the Q-Former to learn visual features relevant to the text,
which reduces the burden of the LLM to learn vision-language alignment.
Without the representation learning stage,
Q-Former relies solely on the vision-to-language generative learning to bridge the modality gap, which is similar to the Perceiver Resampler in Flamingo.
Figure [5](#S4.F5) shows the effect of representation learning on generative learning.
Without representation learning,
both types of LLMs give substantially lower performance on zero-shot VQA.
In particular, OPT suffers from catastrophic forgetting where performance drastically degrades as training proceeds.

*Table 3: Comparison with state-of-the-art image captioning methods on NoCaps and COCO Caption. All methods optimize the cross-entropy loss during finetuning. C: CIDEr, S: SPICE, B@4: BLEU@4.*

*Table 4: Comparison with state-of-the-art models fine-tuned for visual question answering.*

### 4.2 Image Captioning

We finetune BLIP-2 models for the image captioning task,
which asks the model to generate a text description for the image’s visual content.
We use the prompt “a photo of” as an initial input to the LLM and trains the model to generate the caption with the language modeling loss.
We keep the LLM frozen during finetuning,
and updates the parameters of the Q-Former together with the image encoder.
We experiment with ViT-g and various LLMs.
Detailed hyperparameters can be found in the appendix.
We perform finetuning on COCO,
and evaluate on both COCO test set and zero-shot transfer to NoCaps [(Agrawal et al., [2019](#bib.bib1))] validation set.

The results are shown in Table [3](#S4.T3).
BLIP-2 achieves state-of-the-art performance with significant improvement on NoCaps over existing methods,
demonstrating strong generalization ability to out-domain images.

### 4.3 Visual Question Answering

Given annotated VQA data,
we finetune the parameters of the Q-Former and the image encoder while keeping the LLM frozen.
We finetune with the open-ended answer generation loss,
where the LLM receives Q-Former’s output and the question as input,
and is asked to generate the answer.
In order to extract image features that are more relevant to the question,
we additionally condition Q-Former on the question.
Specifically,
the question tokens are given as input to the Q-Former and interact with the queries via the self-attention layers,
which can guide the Q-Former’s cross-attention layers to focus on more informative image regions.

Following BLIP,
our VQA data includes the training and validation splits from VQAv2, as well as training samples from Visual Genome.
Table [4](#S4.T4) demonstrates the state-of-the-art results of BLIP-2 among open-ended generation models.

### 4.4 Image-Text Retrieval

*Table 5: Comparison with state-of-the-art image-text retrieval methods, finetuned on COCO and zero-shot transferred to Flickr30K.*

Since image-text retrieval does not involve language generation,
we directly finetune the first-stage-pretrained model w/o LLM.
Specifically,
we finetune the image encoder together with Q-Former on COCO using the same objectives (i.e. ITC, ITM, and ITG) as pre-training.
We then evaluate the model for both image-to-text retrieval and text-to-image retrieval on COCO and Flickr30K [(Plummer et al., [2015](#bib.bib29))] datasets.
During inference,
we follow [Li et al. ([2021](#bib.bib21), [2022](#bib.bib22))] which first select $k=128$ candidates based on the image-text feature similarity, followed by a re-ranking based on pairwise ITM scores.
We experiment with both ViT-L and ViT-g as the image encoder.
Detailed hyperparameters can be found in the appendix.

The results are shown in Table [5](#S4.T5).
BLIP-2 achieves state-of-the-art performance with significant improvement over existing methods on zero-shot image-text retrieval.

*Table 6: The image-grounded text generation (ITG) loss improves image-text retrieval performance by enforcing the queries to extract language-relevant visual features.*

The ITC and ITM losses are essential for image-text retrieval as they directly learn image-text similarity.
In Table [6](#S4.T6),
we show that the ITG (image-grounded text generation) loss is also beneficial for image-text retrieval.
This result supports our intuition in designing the representation learning objectives: the ITG loss enforces the queries to extract visual features most relevant to the text,
thus improving vision-language alignment.

## 5 Limitation

Recent LLMs can perform in-context learning given few-shot examples.
However,
our experiments with BLIP-2 do not observe an improved VQA performance when providing the LLM with in-context VQA examples.
We attribute the lack of in-context learning capability to our pre-training dataset,
which only contains a single image-text pair per sample.
The LLMs cannot learn from it the correlation among multiple image-text pairs in a single sequence.
The same observation is also reported in the Flamingo paper,
which uses a close-sourced interleaved image and text dataset (M3W) with multiple image-text pairs per sequence.
We aim to create a similar dataset in future work.

BLIP-2’s image-to-text generation could have unsatisfactory results due to various reasons including inaccurate knowledge from the LLM,
activating the incorrect reasoning path,
or not having up-to-date information about new image content (see Figure [7](#A0.F7)).
Furthermore,
due to the use of frozen models,
BLIP-2 inherits the risks of LLMs,
such as outputting offensive language, propagating social bias, or leaking private information.
Remediation approaches include using instructions to guide model’s generation or training on a filtered dataset with harmful content removed.

## 6 Conclusion

We propose BLIP-2,
a generic and compute-efficient method for vision-language pre-training that leverages frozen pre-trained image encoders and LLMs.
BLIP-2 achieves state-of-the-art performance on various vision-language tasks while having a small amount of trainable parameters during pre-training.
BLIP-2 also demonstrates emerging capabilities in zero-shot instructed image-to-text generation.
We consider BLIP-2 as an important step towards building a multimodal conversational AI agent.

## References

-
Agrawal et al. (2019)

Agrawal, H., Anderson, P., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson,
M., Batra, D., Parikh, D., and Lee, S.

nocaps: novel object captioning at scale.

In ICCV*, pp. 8947–8956, 2019.

-
Alayrac et al. (2022)

Alayrac, J., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K.,
Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S.,
Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J., Borgeaud, S.,
Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R.,
Vinyals, O., Zisserman, A., and Simonyan, K.

Flamingo: a visual language model for few-shot learning.

*arXiv preprint arXiv:2204.14198*, 2022.

-
Brown et al. (2020)

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A.,
Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin,
M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A.,
Sutskever, I., and Amodei, D.

Language models are few-shot learners.

In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H.
(eds.), *NeurIPS*, 2020.

-
Changpinyo et al. (2021)

Changpinyo, S., Sharma, P., Ding, N., and Soricut, R.

Conceptual 12M: Pushing web-scale image-text pre-training to
recognize long-tail visual concepts.

In *CVPR*, 2021.

-
Chen et al. (2022a)

Chen, J., Guo, H., Yi, K., Li, B., and Elhoseiny, M.

Visualgpt: Data-efficient adaptation of pretrained language models
for image captioning.

In *CVPR*, pp. 18009–18019, 2022a.

-
Chen et al. (2022b)

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A. J., Padlewski, P., Salz,
D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., Kolesnikov, A.,
Puigcerver, J., Ding, N., Rong, K., Akbari, H., Mishra, G., Xue, L.,
Thapliyal, A., Bradbury, J., Kuo, W., Seyedhosseini, M., Jia, C., Ayan,
B. K., Riquelme, C., Steiner, A., Angelova, A., Zhai, X., Houlsby, N., and
Soricut, R.

Pali: A jointly-scaled multilingual language-image model.

*arXiv preprint arXiv:2209.06794*, 2022b.

-
Chen et al. (2020)

Chen, Y., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., and Liu,
J.

UNITER: universal image-text representation learning.

In *ECCV*, volume 12375, pp. 104–120, 2020.

-
Cho et al. (2021)

Cho, J., Lei, J., Tan, H., and Bansal, M.

Unifying vision-and-language tasks via text generation.

*arXiv preprint arXiv:2102.02779*, 2021.

-
Chung et al. (2022)

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang,
X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M.,
Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., Zhao, V. Y., Huang,
Y., Dai, A. M., Yu, H., Petrov, S., Chi, E. H., Dean, J., Devlin, J.,
Roberts, A., Zhou, D., Le, Q. V., and Wei, J.

Scaling instruction-finetuned language models.

*arXiv preprint arXiv:2210.11416*, 2022.

-
Dai et al. (2022)

Dai, W., Hou, L., Shang, L., Jiang, X., Liu, Q., and Fung, P.

Enabling multimodal generation on CLIP via vision-language
knowledge distillation.

In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), *ACL
Findings*, pp. 2383–2395, 2022.

-
Devlin et al. (2019)

Devlin, J., Chang, M., Lee, K., and Toutanova, K.

BERT: pre-training of deep bidirectional transformers for language
understanding.

In Burstein, J., Doran, C., and Solorio, T. (eds.), *NAACL*,
pp. 4171–4186, 2019.

-
Dong et al. (2019)

Dong, L., Yang, N., Wang, W., Wei, F., Liu, X., Wang, Y., Gao, J., Zhou, M.,
and Hon, H.

Unified language model pre-training for natural language
understanding and generation.

In Wallach, H. M., Larochelle, H., Beygelzimer, A.,
d’Alché-Buc, F., Fox, E. B., and Garnett, R. (eds.), *NeurIPS*,
pp. 13042–13054, 2019.

-
Fang et al. (2022)

Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X.,
and Cao, Y.

Eva: Exploring the limits of masked visual representation learning at
scale.

*arXiv preprint arXiv:2211.07636*, 2022.

-
Goyal et al. (2017)

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D.

Making the V in VQA matter: Elevating the role of image
understanding in visual question answering.

In *CVPR*, pp. 6325–6334, 2017.

-
Guo et al. (2022)

Guo, J., Li, J., Li, D., Tiong, A. M. H., Li, B., Tao, D., and Hoi, S. C. H.

From images to textual prompts: Zero-shot VQA with frozen large
language models.

In *CVPR*, 2022.

-
Hoffmann et al. (2022)

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford,
E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T.,
Noland, E., Millican, K., Driessche, G. v. d., Damoc, B., Guy, A., Osindero,
S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L.

Training compute-optimal large language models.

*arXiv preprint arXiv:2203.15556*, 2022.

-
Hudson & Manning (2019)

Hudson, D. A. and Manning, C. D.

GQA: A new dataset for real-world visual reasoning and
compositional question answering.

In *CVPR*, pp. 6700–6709, 2019.

-
Jia et al. (2021)

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q. V., Sung,
Y., Li, Z., and Duerig, T.

Scaling up visual and vision-language representation learning with
noisy text supervision.

*arXiv preprint arXiv:2102.05918*, 2021.

-
Jin et al. (2022)

Jin, W., Cheng, Y., Shen, Y., Chen, W., and Ren, X.

A good prompt is worth millions of parameters: Low-resource
prompt-based learning for vision-language models.

In Muresan, S., Nakov, P., and Villavicencio, A. (eds.),
*ACL*, pp. 2763–2775, 2022.

-
Krishna et al. (2017)

Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S.,
Kalantidis, Y., Li, L., Shamma, D. A., Bernstein, M. S., and Fei-Fei, L.

Visual genome: Connecting language and vision using crowdsourced
dense image annotations.

*IJCV*, 123(1):32–73, 2017.

-
Li et al. (2021)

Li, J., Selvaraju, R. R., Gotmare, A. D., Joty, S., Xiong, C., and Hoi, S.

Align before fuse: Vision and language representation learning with
momentum distillation.

In *NeurIPS*, 2021.

-
Li et al. (2022)

Li, J., Li, D., Xiong, C., and Hoi, S. C. H.

BLIP: bootstrapping language-image pre-training for unified
vision-language understanding and generation.

In *ICML*, pp. 12888–12900, 2022.

-
Li et al. (2020)

Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong,
L., Wei, F., Choi, Y., and Gao, J.

Oscar: Object-semantics aligned pre-training for vision-language
tasks.

In *ECCV*, pp. 121–137, 2020.

-
Lin et al. (2014)

Lin, T., Maire, M., Belongie, S. J., Hays, J., Perona, P., Ramanan, D.,
Dollár, P., and Zitnick, C. L.

Microsoft COCO: common objects in context.

In Fleet, D. J., Pajdla, T., Schiele, B., and Tuytelaars, T. (eds.),
*ECCV*, volume 8693, pp. 740–755, 2014.

-
Loshchilov & Hutter (2017)

Loshchilov, I. and Hutter, F.

Decoupled weight decay regularization.

*arXiv preprint arXiv:1711.05101*, 2017.

-
Mañas et al. (2023)

Mañas, O., Rodríguez, P., Ahmadi, S., Nematzadeh, A., Goyal, Y.,
and Agrawal, A.

MAPL: parameter-efficient adaptation of unimodal pre-trained models
for vision-language few-shot prompting.

In *EACL*, 2023.

-
Marino et al. (2019)

Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R.

Ok-vqa: A visual question answering benchmark requiring external
knowledge.

In *CVPR*, 2019.

-
Ordonez et al. (2011)

Ordonez, V., Kulkarni, G., and Berg, T. L.

Im2text: Describing images using 1 million captioned photographs.

In Shawe-Taylor, J., Zemel, R. S., Bartlett, P. L., Pereira, F.
C. N., and Weinberger, K. Q. (eds.), *NIPS*, pp. 1143–1151, 2011.

-
Plummer et al. (2015)

Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J.,
and Lazebnik, S.

Flickr30k entities: Collecting region-to-phrase correspondences for
richer image-to-sentence models.

In *ICCV*, pp. 2641–2649, 2015.

-
Radford et al. (2021)

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
G., Askell, A., Mishkin, P., Clark, J., et al.

Learning transferable visual models from natural language
supervision.

*arXiv preprint arXiv:2103.00020*, 2021.

-
Schuhmann et al. (2021)

Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A.,
Coombes, T., Jitsev, J., and Komatsuzaki, A.

Laion-400m: Open dataset of clip-filtered 400 million image-text
pairs.

*arXiv preprint arXiv:2111.02114*, 2021.

-
Sharma et al. (2018)

Sharma, P., Ding, N., Goodman, S., and Soricut, R.

Conceptual captions: A cleaned, hypernymed, image alt-text dataset
for automatic image captioning.

In Gurevych, I. and Miyao, Y. (eds.), *ACL*, pp. 2556–2565,
2018.

-
Tan & Bansal (2019)

Tan, H. and Bansal, M.

LXMERT: learning cross-modality encoder representations from
transformers.

In Inui, K., Jiang, J., Ng, V., and Wan, X. (eds.), *EMNLP*,
pp. 5099–5110, 2019.

-
Tiong et al. (2022)

Tiong, A. M. H., Li, J., Li, B., Savarese, S., and Hoi, S. C. H.

Plug-and-play VQA: zero-shot VQA by conjoining large pretrained
models with zero training.

In *EMNLP Findings*, 2022.

-
Tsimpoukelli et al. (2021)

Tsimpoukelli, M., Menick, J., Cabi, S., Eslami, S. M. A., Vinyals, O., and
Hill, F.

Multimodal few-shot learning with frozen language models.

In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and
Vaughan, J. W. (eds.), *NeurIPS*, pp. 200–212, 2021.

-
Wang et al. (2022a)

Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou,
J., and Yang, H.

OFA: unifying architectures, tasks, and modalities through a simple
sequence-to-sequence learning framework.

In Chaudhuri, K., Jegelka, S., Song, L., Szepesvári, C., Niu,
G., and Sabato, S. (eds.), *ICML*, pp. 23318–23340,
2022a.

-
Wang et al. (2021a)

Wang, W., Bao, H., Dong, L., and Wei, F.

Vlmo: Unified vision-language pre-training with
mixture-of-modality-experts.

*arXiv preprint arXiv:2111.02358*, 2021a.

-
Wang et al. (2022b)

Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K.,
Mohammed, O. K., Singhal, S., Som, S., and Wei, F.

Image as a foreign language: Beit pretraining for all vision and
vision-language tasks.

*arXiv preprint arXiv:2208.10442*, 2022b.

-
Wang et al. (2021b)

Wang, Z., Yu, J., Yu, A. W., Dai, Z., Tsvetkov, Y., and Cao, Y.

Simvlm: Simple visual language model pretraining with weak
supervision.

*arXiv preprint arXiv:2108.10904*, 2021b.

-
Yao et al. (2022)

Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang,
X., and Xu, C.

FILIP: fine-grained interactive language-image pre-training.

In *ICLR*, 2022.

-
Yu et al. (2022)

Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y.

Coca: Contrastive captioners are image-text foundation models.

*arXiv preprint arXiv:2205.01917*, 2022.

-
Yuan et al. (2021)

Yuan, L., Chen, D., Chen, Y., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X.,
Li, B., Li, C., Liu, C., Liu, M., Liu, Z., Lu, Y., Shi, Y., Wang, L., Wang,
J., Xiao, B., Xiao, Z., Yang, J., Zeng, M., Zhou, L., and Zhang, P.

Florence: A new foundation model for computer vision.

*arXiv preprint arXiv:2111.11432*, 2021.

-
Zhai et al. (2022)

Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., and
Beyer, L.

Lit: Zero-shot transfer with locked-image text tuning.

In *CVPR*, pp. 18102–18112, 2022.

-
Zhang et al. (2021)

Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., Choi, Y., and Gao, J.

Vinvl: Making visual representations matter in vision-language
models.

*arXiv preprint arXiv:2101.00529*, 2021.

-
Zhang et al. (2022)

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C.,
Diab, M. T., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S.,
Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer,
L.

OPT: open pre-trained transformer language models.

*arXiv preprint arXiv:2205.01068*, 2022.

*Table 7: Hyperparameters for fine-tuning BLIP-2 with ViT-g on COCO captioning.*

*Table 8: Hyperparameters for fine-tuning BLIP-2 with ViT-g on VQA.*

*Table 9: Hyperparameters for fine-tuning BLIP-2 on COCO image-text retrieval.*

![Figure](2301.12597v3/x7.png)

*Figure 6: Incorrect output examples for instructed zero-shot image-to-text generation using a BLIP-2 model w/ ViT-g and FlanT5${}_{\text{XXL}}$.*

![Figure](2301.12597v3/x8.png)

*Figure 7: Model architecture for VQA finetuning, where the LLM receives Q-Former’s output and the question as input, then predicts answers. We also provide the question as a condition to Q-Former, such that the extracted image features are more relevant to the question.*



Experimental support, please
[view the build logs](./2301.12597v3/__stdout.txt)
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