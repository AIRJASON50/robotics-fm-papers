##### Report GitHub Issue

**×




Title:



Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2505.09388v1 [cs.CL] 14 May 2025

[\useunder \ul # Qwen3 Technical Report Qwen Team ###### Abstract In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models—–such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)—–and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0. ## 1 Introduction The pursuit of artificial general intelligence (AGI) or artificial super intelligence (ASI) has long been a goal for humanity. Recent advancements in large foundation models, e.g., GPT-4o (gpt4o), Claude 3.7 (claude3.7), Gemini 2.5 (gemini2.5), DeepSeek-V3 (deepseekv3), Llama-4 (llama4), and Qwen2.5 (qwen2.5), have demonstrated significant progress toward this objective. These models are trained on vast datasets spanning trillions of tokens across diverse domains and tasks, effectively distilling human knowledge and capabilities into their parameters. Furthermore, recent developments in reasoning models, optimized through reinforcement learning, highlight the potential for foundation models to enhance inference-time scaling and achieve higher levels of intelligence, e.g., o3 (o3), DeepSeek-R1 (r1). While most state-of-the-art models remain proprietary, the rapid growth of open-source communities has substantially reduced the performance gap between open-weight and closed-source models. Notably, an increasing number of top-tier models (llama4; deepseekv3; r1; qwen2.5) are now being released as open-source, fostering broader research and innovation in artificial intelligence. In this work, we introduce Qwen3, the latest series in our foundation model family, Qwen. Qwen3 is a collection of open-weight large language models (LLMs) that achieve state-of-the-art performance across a wide variety of tasks and domains. We release both dense and Mixture-of-Experts (MoE) models, with the number of parameters ranging from 0.6 billion to 235 billion, to meet the needs of different downstream applications. Notably, the flagship model, Qwen3-235B-A22B, is an MoE model with a total of 235 billion parameters and 22 billion activated ones per token. This design ensures both high performance and efficient inference. Qwen3 introduces several key advancements to enhance its functionality and usability. First, it integrates two distinct operating modes, thinking mode and non-thinking mode, into a single model. This allows users to switch between these modes without alternating between different models, e.g., switching from Qwen2.5 to QwQ (qwq). This flexibility ensures that developers and users can adapt the model's behavior to suit specific tasks efficiently. Additionally, Qwen3 incorporates thinking budgets, providing users with fine-grained control over the level of reasoning effort applied by the model during task execution. This capability is crucial to the optimization of computational resources and performance, tailoring the model's thinking behavior to meet varying complexity in real-world applications. Furthermore, Qwen3 has been pre-trained on 36 trillion tokens covering up to 119 languages and dialects, effectively enhancing its multilingual capabilities. This broadened language support amplifies its potential for deployment in global use cases and international applications. These advancements together establish Qwen3 as a cutting-edge open-source large language model family, capable of effectively addressing complex tasks across various domains and languages. The pre-training process for Qwen3 utilizes a large-scale dataset consisting of approximately 36 trillion tokens, curated to ensure linguistic and domain diversity. To efficiently expand the training data, we employ a multi-modal approach: Qwen2.5-VL (qwen2.5vl) is finetuned to extract text from extensive PDF documents. We also generate synthetic data using domain-specific models: Qwen2.5-Math (qwen2.5math) for mathematical content and Qwen2.5-Coder (qwen2.5coder) for code-related data. The pre-training process follows a three-stage strategy. In the first stage, the model is trained on about 30 trillion tokens to build a strong foundation of general knowledge. In the second stage, it is further trained on knowledge-intensive data to enhance reasoning abilities in areas like science, technology, engineering, and mathematics (STEM) and coding. Finally, in the third stage, the model is trained on long-context data to increase its maximum context length from 4,096 to 32,768 tokens. To better align foundation models with human preferences and downstream applications, we employ a multi-stage post-training approach that empowers both thinking (reasoning) and non-thinking modes. In the first two stages, we focus on developing strong reasoning abilities through long chain-of-thought (CoT) cold-start finetuning and reinforcement learning focusing on mathematics and coding tasks. In the final two stages, we combine data with and without reasoning paths into a unified dataset for further fine-tuning, enabling the model to handle both types of input effectively, and we then apply general-domain reinforcement learning to improve performance across a wide range of downstream tasks. For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models to enhance their capabilities. Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency. We evaluate both pre-trained and post-trained versions of our models across a comprehensive set of benchmarks spanning multiple tasks and domains. Experimental results show that our base pre-trained models achieve state-of-the-art performance. The post-trained models, whether in thinking or non-thinking mode, perform competitively against leading proprietary models and large mixture-of-experts (MoE) models such as o1, o3-mini, and DeepSeek-V3. Notably, our models excel in coding, mathematics, and agent-related tasks. For example, the flagship model Qwen3-235B-A22B achieves 85.7 on AIME'24 and 81.5 on AIME'25 (aime), 70.7 on LiveCodeBench v5 (livecodebench), 2,056 on CodeForces, and 70.8 on BFCL v3 (bfcl). In addition, other models in the Qwen3 series also show strong performance relative to their size. Furthermore, we observe that increasing the thinking budget for thinking tokens leads to a consistent improvement in the model's performance across various tasks. In the following sections, we describe the design of the model architecture, provide details on its training procedures, present the experimental results of pre-trained and post-trained models, and finally, conclude this technical report by summarizing the key findings and outlining potential directions for future research. ## 2 Architecture The Qwen3 series includes 6 dense models, namely Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, and Qwen3-32B, and 2 MoE models, Qwen3-30B-A3B and Qwen3-235B-A22B. The flagship model, Qwen3-235B-A22B, has a total of 235B parameters with 22B activated ones. Below, we elaborate on the architecture of the Qwen3 models. The architecture of the Qwen3 dense models is similar to Qwen2.5 (qwen2.5), including using Grouped Query Attention (GQA, gqa), SwiGLU (glu), Rotary Positional Embeddings (RoPE, rope), and RMSNorm (rmsnorm) with pre-normalization. Besides, we remove QKV-bias used in Qwen2 (qwen2) and introduce QK-Norm (pmlr-v202-dehghani23a) to the attention mechanism to ensure stable training for Qwen3. Key information on model architecture is provided in Table 1](#S2.T1).

The Qwen3 MoE models share the same fundamental architecture as the Qwen3 dense models.
Key information on model architecture is provided in Table [2](#S2.T2).
We follow Qwen2.5-MoE [qwen2.5] and implement fine-grained expert segmentation [deepseekmoe].
The Qwen3 MoE models have 128 total experts with 8 activated experts per token.
Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts.
Furthermore, we adopt the global-batch load balancing loss [global_balance] to encourage expert specialization.
These architectural and training innovations have yielded substantial improvements in model performance across downstream tasks.

Qwen3 models utilize Qwen's tokenizer [qwen], which implements byte-level byte-pair encoding (BBPE, [gpt3, wang2020neural, sennirch2016neural]) with a vocabulary size of 151,669.

*Table 1: Model architecture of Qwen3 dense models.*

*Table 2: Model architecture of Qwen3 MoE models.*

## 3 Pre-training

In this section, we describe the construction of our pretraining data, the details of our pretraining approach, and present experimental results from evaluating the base models on standard benchmarks.

### 3.1 Pre-training Data

Compared with Qwen2.5 [qwen2.5], we have significantly expanded the scale and diversity of our training data. Specifically, we collected twice as many pre-training tokens—covering three times more languages. All Qwen3 models are trained on a large and diverse dataset consisting of 119 languages and dialects, with a total of 36 trillion tokens. This dataset includes high-quality content in various domains such as coding, STEM (Science, Technology, Engineering, and Mathematics), reasoning tasks, books, multilingual texts, and synthetic data.

To further expand the pre-training data corpus, we first employ the Qwen2.5-VL model [qwen2.5vl] to perform text recognition on a large volume of PDF-like documents. The recognized text is then refined using the Qwen2.5 model [qwen2.5], which helps improve its quality. Through this two-step process, we are able to obtain an additional set of high-quality text tokens, amounting to trillions in total.
Besides, we employ Qwen2.5 [qwen2.5], Qwen2.5-Math [qwen2.5math], and Qwen2.5-Coder [qwen2.5coder] models to synthesize trillions of text tokens in different formats, including textbooks, question-answering, instructions, and code snippets, covering dozens of domains.
Finally, we further expand the pre-training corpus by incorporating additional multilingual data and introducing more languages. Compared to the pre-training data used in Qwen2.5, the number of supported languages has been significantly increased from 29 to 119, enhancing the model's linguistic coverage and cross-lingual capabilities.

We have developed a multilingual data annotation system designed to enhance both the quality and diversity of training data.
This system has been applied to our large-scale pre-training datasets, annotating over 30 trillion tokens across multiple dimensions such as educational value, fields, domains, and safety. These detailed annotations support more effective data filtering and combination.
Unlike previous studies [doremi, doge, regmix] that optimize the data mixture at the data source or domain level, our method optimizes the data mixture at the instance-level through extensive ablation experiments on small proxy models with the fine-grained data labels.

### 3.2 Pre-training Stage

The Qwen3 models are pre-trained through a three-stage process:

-
(1)

General Stage (S1): At the first pre-training stage, all Qwen3 models are trained on over 30 trillion tokens using a sequence length of 4,096 tokens. At this stage, the models have been fully pre-trained on language proficiency and general world knowledge, with training data covering 119 languages and dialects.

-
(2)

Reasoning Stage (S2): To further improve the reasoning ability, we optimize the pre-training corpus of this stage by increasing the proportion of STEM, coding, reasoning, and synthetic data. The models are further pre-trained with about 5T higher-quality tokens at a sequence length of 4,096 tokens. We also accelerate the learning rate decay during this stage.

-
(3)

Long Context Stage: In the final pre-training stage, we collect high-quality long context corpora to extend the context length of Qwen3 models. All models are pre-trained on hundreds of billions of tokens with a sequence length of 32,768 tokens. The long context corpus includes 75% of text between 16,384 to 32,768 tokens in length, and 25% of text between 4,096 to 16,384 in length. Following Qwen2.5 [qwen2.5], we increase the base frequency of RoPE from 10,000 to 1,000,000 using the ABF technique [ropeabf]. Meanwhile, we introduce YARN [yarn] and Dual Chunk Attention (DCA, [chunkllama]) to achieve a four-fold increase in sequence length capacity during inference.

Similar to Qwen2.5 [qwen2.5], we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions based on three pre-training stages mentioned above. Through extensive experiments, we systematically study the relationship between model architecture, training data, training stage, and optimal training hyper-parameters. Finally, we set the predicted optimal learning rate and batch size strategy for each dense or MoE model.

### 3.3 Pre-training Evaluation

We conduct comprehensive evaluations of the base language models of the Qwen3 series.
The evaluation of base models mainly focuses on their performance in general knowledge, reasoning, mathematics, scientific knowledge, coding, and multilingual capabilities. The evaluation datasets for pre-trained base models include 15 benchmarks:

-
•

General Tasks: MMLU [mmlu] (5-shot), MMLU-Pro [mmlupro] (5-shot, CoT), MMLU-redux [mmluredux] (5-shot), BBH [bbh] (3-shot, CoT), SuperGPQA [supergpqa](5-shot, CoT).

-
•

Math & STEM Tasks: GPQA [gpqa] (5-shot, CoT), GSM8K [gsm8k] (4-shot, CoT), MATH [math] (4-shot, CoT).

-
•

Coding Tasks: EvalPlus [evalplus] (0-shot) (Average of HumanEval [humaneval], MBPP [mbpp], Humaneval+, MBPP+) [evalplus], MultiPL-E [multiple] (0-shot) (Python, C++, JAVA, PHP, TypeScript, C#, Bash, JavaScript), MBPP-3shot [mbpp], CRUX-O of CRUXEval (1-shot) [gu2024cruxeval].

-
•

Multilingual Tasks: MGSM [mgsm] (8-shot, CoT), MMMLU [mmmlu] (5-shot), INCLUDE [romanou2024includeevaluatingmultilinguallanguage] (5-shot).

For the base model baselines, we compare the Qwen3 series base models with the Qwen2.5 base models [qwen2.5] and other leading open-source base models, including DeepSeek-V3 Base [deepseekv3], Gemma-3 [gemma3], Llama-3 [llama3], and Llama-4 [llama4] series base models, in terms of scale of parameters. All models are evaluated using the same evaluation pipeline and the widely-used evaluation settings to ensure fair comparison.

#### Summary of Evaluation Results

Based on the overall evaluation results, we highlight some key conclusions of Qwen3 base models.

-
(1)

Compared with the previously open-source SOTA dense and MoE base models (such as DeepSeek-V3 Base, Llama-4-Maverick Base, and Qwen2.5-72B-Base), Qwen3-235B-A22B-Base outperforms these models in most tasks with significantly fewer total parameters or activated parameters.

-
(2)

For the Qwen3 MoE base models, our experimental results indicate that: (a) Using the same pre-training data, Qwen3 MoE base models can achieve similar performance to Qwen3 dense base models with only 1/5 activated parameters. (b) Due to the improvements of the Qwen3 MoE architecture, the scale-up of the training tokens, and more advanced training strategies, the Qwen3 MoE base models can outperform the Qwen2.5 MoE base models with less than 1/2 activated parameters and fewer total parameters. (c) Even with 1/10 of the activated parameters of the Qwen2.5 dense base model, the Qwen3 MoE base model can achieve comparable performance, which brings us significant advantages in inference and training costs.

-
(3)

The overall performance of the Qwen3 dense base models is comparable to the Qwen2.5 base models at higher parameter scales. For example, Qwen3-1.7B/4B/8B/14B/32B-Base achieve comparable performance to Qwen2.5-3B/7B/14B/32B/72B-Base, respectively. Especially in STEM, coding, and reasoning benchmarks, the performance of Qwen3 dense base models even surpasses Qwen2.5 base models at higher parameter scales.

The detailed results are as follows.

*Table 3: Comparison among Qwen3-235B-A22B-Base and other representative strong open-source baselines. The highest, the second-best scores are shown in bold and underlined, respectively.*

#### Qwen3-235B-A22B-Base

We compare Qwen3-235B-A22B-Base to our previous similar-sized MoE Qwen2.5-Plus-Base [qwen2.5] and other leading open-source base models: Llama-4-Maverick [llama4], Qwen2.5-72B-Base [qwen2.5], DeepSeek-V3 Base [deepseekv3]. From the results in Table [3](#S3.T3), the Qwen3-235B-A22B-Base model attains the highest performance scores across most of the evaluated benchmarks. We further compare Qwen3-235B-A22B-Base with other baselines separately for the detailed analysis.

-
(1)

Compared with the recently open-source model Llama-4-Maverick-Base, which has about twice the number of parameters, Qwen3-235B-A22B-Base still performs better on most benchmarks.

-
(2)

Compared with the previously state-of-the-art open-source model DeepSeek-V3-Base, Qwen3-235B-A22B-Base outperforms DeepSeek-V3-Base on 14 out of 15 evaluation benchmarks with only about 1/3 the total number of parameters and 2/3 activated parameters, demonstrating the powerful and cost-effectiveness of our models.

-
(3)

Compared with our previous MoE Qwen2.5-Plus of similar size, Qwen3-235B-A22B-Base significantly outperforms it with fewer parameters and activated parameters, which shows the remarkable advantages of Qwen3 in pre-training data, training strategy, and model architecture.

-
(4)

Compared with our previous flagship open-source dense model Qwen2.5-72B-Base, Qwen3-235B-A22B-Base surpasses the latter in all benchmarks and uses fewer than 1/3 of the activated parameters. Meanwhile, due to the advantage of the model architecture, the inference costs and training costs on each trillion tokens of Qwen3-235B-A22B-Base are much cheaper than those of Qwen2.5-72B-Base.

*Table 4: Comparison among Qwen3-32B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.*

*Table 5: Comparison among Qwen3-14B-Base, Qwen3-30B-A3B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.*

*Table 6: Comparison among Qwen8B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.*

*Table 7: Comparison among Qwen3-4B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.*

*Table 8: Comparison among Qwen3-1.7B-Base, Qwen3-0.6B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.*

#### Qwen3-32B-Base

Qwen3-32B-Base is our largest dense model among the Qwen3 series. We compare it to the baselines of similar sizes, including Gemma-3-27B [gemma3] and Qwen2.5-32B [qwen2.5]. In addition, we introduce two strong baselines: the recently open-source MoE model Llama-4-Scout, which has three times the parameters of Qwen3-32B-Base but half the activated parameters; and our previous flagship open-source dense model Qwen2.5-72B-Base, which has more than twice the number of parameters compared to Qwen3-32B-Base.
The results are shown in Table [4](#S3.T4), which support three key conclusions:

-
(1)

Compared with the similar-sized models, Qwen3-32B-Base outperforms Qwen2.5-32B-Base and Gemma-3-27B Base on most benchmarks. Notably, Qwen3-32B-Base achieves 65.54 on MMLU-Pro and 39.78 on SuperGPQA, significantly outperforming its predecessor Qwen2.5-32B-Base. In addition, Qwen3-32B-Base achieves significantly higher encoding benchmark scores than all baseline models.

-
(2)

Surprisingly, we find that Qwen3-32B-Base achieves competitive results compared to Qwen2.5-72B-Base. Although Qwen3-32B-Base has less than half the number of parameters of Qwen2.5-72B-Base, it outperforms Qwen2.5-72B-Base in 10 of the 15 evaluation benchmarks. On coding, mathematics, and reasoning benchmarks, Qwen3-32B-Base has remarkable advantages.

-
(3)

Compared to Llama-4-Scout-Base, Qwen3-32B-Base significantly outperforms it on all 15 benchmarks, with only one-third of the number of parameters of Llama-4-Scout-Base, but twice the number of activated parameters.

#### Qwen3-14B-Base & Qwen3-30B-A3B-Base

The evaluation of the Qwen3-14B-Base and Qwen3-30B-A3B-Base is compared against baselines of similar sizes, including Gemma-3-12B Base, Qwen2.5-14B Base. Similarly, we also introduce two strong baselines: (1) Qwen2.5-Turbo [qwen2.5], which has 42B parameters and 6B activated parameters. Note that its activated parameters are twice those of Qwen3-30B-A3B-Base. (2) Qwen2.5-32B-Base, which has 11 times the activated parameters of Qwen3-30B-A3B and more than twice that of Qwen3-14B.
The results are shown in Table [5](#S3.T5), where we can draw the following conclusions.

-
(1)

Compared with the similar-sized models, Qwen3-14B-Base significantly performs better than Qwen2.5-14B-Base and Gemma-3-12B-Base on all 15 benchmarks.

-
(2)

Similarly, Qwen3-14B-Base also achieves very competitive results compared to Qwen2.5-32B-Base with less than half of the parameters.

-
(3)

With only 1/5 activated non-embedding parameters, Qwen3-30B-A3B significantly outperforms Qwen2.5-14B-Base on all tasks, and achieves comparable performance to Qwen3-14B-Base and Qwen2.5-32B-Base, which brings us significant advantages in inference and training costs.

#### Qwen3-8B / 4B / 1.7B / 0.6B-Base

For edge-side models, we take similar-sized Qwen2.5, Llama-3, and Gemma-3 base models as the baselines. The results can be seen in Table [6](#S3.T6), Table [7](#S3.T7), and Table [8](#S3.T8). All Qwen3 8B / 4B / 1.7B / 0.6B-Base models continue to maintain strong performance across nearly all benchmarks. Notably, Qwen3-8B / 4B / 1.7B-Base models even outperform larger size Qwen2.5-14B / 7B / 3B Base models on over half of the benchmarks, especially on STEM-related and coding benchmarks, reflecting the significant improvement of the Qwen3 models.

## 4 Post-training

![Figure](2505.09388v1/x1.png)

*Figure 1: Post-training pipeline of the Qwen3 series models.*

The post-training pipeline of Qwen3 is strategically designed with two core objectives:

-
(1)

Thinking Control:
This involves the integration of two distinct modes, namely the ``non-thinking'' and ``thinking'' modes, providing users with the flexibility to choose whether the model should engage in reasoning or not, and to control the depth of thinking by specifying a token budget for the thinking process.

-
(2)

Strong-to-Weak Distillation:
This aims to streamline and optimize the post-training process for lightweight models.
By leveraging the knowledge from large-scale models, we substantially reduce both the computational costs and the development efforts required for building smaller-scale models.

As illustrated in Figure [1](#S4.F1), the flagship models in the Qwen3 series follow a sophisticated four-stage training process. The first two stages focus on developing the models' ``thinking'' abilities. The next two stages aim to integrate strong ``non-thinking'' functionalities into the models.

Preliminary experiments suggest that directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes.
This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model.
It leads to better immediate performance, as indicated by higher Pass@1 scores, and also improves the model's ability of exploration, as reflected in improved Pass@64 results.
In addition, it achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method.

In the following sections, we present the four-stage training process and provide a detailed explanation of the Strong-to-Weak Distillation approach.

### 4.1 Long-CoT Cold Start

We begin by curating a comprehensive dataset that spans a wide range of categories, including math, code, logical reasoning, and general STEM problems. Each problem in the dataset is paired with verified reference answers or code-based test cases. This dataset serves as the foundation for the ``cold start'' phase of long Chain-of-Thought (long-CoT) training.

The dataset construction involves a rigorous two-phase filtering process: query filtering and response filtering.
In the query filtering phase, we use Qwen2.5-72B-Instruct to identify and remove queries that are not easily verifiable. This includes queries containing multiple sub-questions or those asking for general text generation.
Furthermore, we exclude queries that Qwen2.5-72B-Instruct can answer correctly without using CoT reasoning. This helps prevent the model from relying on superficial guessing and ensures that only complex problems requiring deeper reasoning are included.
Additionally, we annotate each query's domain using Qwen2.5-72B-Instruct to maintain balanced domain representation across the dataset.

After reserving a validation query set, we generate $N$ candidate responses for each remaining query using QwQ-32B [qwq32b].
When QwQ-32B consistently fails to generate correct solutions, human annotators manually assess the accuracy of the responses.
For queries with positive Pass@$N$, further stringent filtering criteria are applied to remove responses that (1) yield incorrect final answers, (2) contain substantial repetition, (3) clearly indicate guesswork without adequate reasoning, (4) exhibit inconsistencies between the thinking and summary contents, (5) involve inappropriate language mixing or stylistic shifts, or (6) are suspected of being overly similar to potential validation set items.
Subsequently, a carefully selected subset of the refined dataset is used for the initial cold-start training of the reasoning patterns.
The objective at this stage is to instill foundational reasoning patterns in the model without overly emphasizing immediate reasoning performance.
This approach ensures that the model's potential is not limited, allowing for greater flexibility and improvement during the subsequent reinforcement learning (RL) phase.
To achieve this objective effectively, it is preferable to minimize both the number of training samples and the training steps during this preparatory phase.

### 4.2 Reasoning RL

The query-verifier pairs used in the Reasoning RL stage must satisfy the following four criteria:
(1) They were not used during the cold-start phase.
(2) They are learnable for the cold-start model.
(3) They are as challenging as possible.
(4) They cover a broad range of sub-domains.
We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO [deepseekmath] to update the model parameters.
We observe that using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial to the training process.
We have also addressed how to balance exploration and exploitation by controlling the model’s entropy to increase steadily or remain stable, which is crucial for maintaining stable training.
As a result, we achieve consistent improvements in both training reward and validation performance over the course of a single RL run, without any manual intervention on hyperparameters. For instance, the AIME'24 score of the Qwen3-235B-A22B model increases from 70.1 to 85.1 over a total of 170 RL training steps.

### 4.3 Thinking Mode Fusion

The goal of the Thinking Mode Fusion stage is to integrate the ``non-thinking'' capabilities into the previously developed ``thinking'' model.
This approach allows developers to manage and control reasoning behaviors, while also reducing the cost and complexity of deploying separate models for thinking and non-thinking tasks.
To achieve this, we conduct continual supervised fine-tuning (SFT) on the Reasoning RL model and design a chat template to fuse the two modes.
Moreover, we find that models capable of handling both modes proficiently perform consistently well under different thinking budgets.

#### Construction of SFT data.

The SFT dataset combines both the ``thinking'' and ``non-thinking'' data.
To ensure that the performance of the Stage 2 model is not compromised by the additional SFT, the ``thinking'' data is generated via rejection sampling on Stage 1 queries using the Stage 2 model itself.
The ``non-thinking'' data, on the other hand, is carefully curated to cover a diverse range of tasks, including coding, mathematics, instruction-following, multilingual tasks, creative writing, question answering, and role-playing.
Additionally, we employ automatically generated checklists for assessing the response quality of ``non-thinking'' data. To enhance the performance on tasks with low-resource languages, we particularly increase the proportion of translation tasks.

#### Chat Template Design.

To better integrate the two modes and enable users to dynamically switch the model's thinking process, we design chat templates for Qwen3, as shown in Table [4.3](#S4.SS3.SSS0.Px3). Specifically, for samples in thinking mode and non-thinking mode, we introduce /think and /no_think flags in the user query or system message, respectively. This allows the model to follow the user's input and select the appropriate thinking mode accordingly.
For non-thinking mode samples, we retain an empty thinking block in the assistant's response. This design ensures internal format consistency within the model and allows developers to prevent the model from engaging in thinking behavior by concatenating an empty think block in the chat template.
By default, the model operates in thinking mode; therefore, we add some thinking mode training samples where the user queries do not include /think flags. For more complex multi-turn dialogs, we randomly insert multiple /think and /no_think flags into users' queries, with the model response adhering to the last flag encountered.

#### Thinking Budget.

An additional advantage of Thinking Mode Fusion is that, once the model learns to respond in both non-thinking and thinking modes, it naturally develops the ability to handle intermediate cases—generating responses based on incomplete thinking.
This capability lays the foundation for implementing budget control over the model's thinking process. Specifically, when the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction: ``Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n''.
After this instruction is inserted, the model proceeds to generate a final response based on its accumulated reasoning up to that point. It is worth noting that this ability is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion.

*Table 9: Examples of SFT data for thinking and non-thinking modes during the thinking mode fusion stage. For the thinking mode, the /think flag can be omitted since it represents the default behavior. This feature has been implemented in the chat template222[https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json) supported by the Hugging Face's tokenizer, where the thinking mode can be disabled using an additional parameter enable_thinking=False.*



Experimental support, please
[view the build logs](./2505.09388v1/__stdout.txt)
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