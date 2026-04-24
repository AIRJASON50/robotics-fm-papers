##### Report GitHub Issue

**×




Title:



Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2604.15804v2 [cs.CL] 21 Apr 2026

[# Qwen3.5-Omni Technical Report Qwen Team ###### Abstract In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-Plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis—often caused by encoding efficiency discrepancies between text and speech tokenizers—we introduce ARIA (Adaptive Rate Interleave Alignment). ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Beyond preset voices, the model enables zero-shot voice customization via user-provided samples. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding. Qwen3.5-Omni is publicly accessible via API111https://www.alibabacloud.com/help/en/model-studio/qwen-omni](https://www.alibabacloud.com/help/en/model-studio/qwen-omni).

![Figure](2604.15804v2/figures/image.png)

*Figure 1: Qwen3.5-Omni is a unified end-to-end model capable of processing multiple modalities, such as text, audio, image and video, and generating real-time text or speech response. Based on these features, Qwen3.5-Omni supports a wide range of tasks, including but not limited to voice dialogue, video dialogue, and audio-visual tool use.*

## 1 Introduction

Human interaction with the world is inherently omnimodal and agentic, involving the integration of visual, auditory, and linguistic information, and the production of responses through text, speech, and goal-directed tool-mediated actions, facilitating information exchange with other organisms and demonstrating intelligence. Building on the rapid advances in the understanding and reasoning capabilities of large models across text [(Brown et al., [2020](#bib.bib313); OpenAI, [2023](#bib.bib54); Gemini Team, [2024](#bib.bib263); Anthropic, [2023b](#bib.bib41); [a](#bib.bib42); [2024](#bib.bib257); Bai et al., [2023a](#bib.bib211); Yang et al., [2024a](#bib.bib458); [2025a](#bib.bib754); Touvron et al., [2023](#bib.bib435); Dubey et al., [2024](#bib.bib260))], vision [(Li et al., [2023](#bib.bib296); Liu et al., [2023](#bib.bib314); Zhu et al., [2023](#bib.bib315); Bai et al., [2023b](#bib.bib133); [2025a](#bib.bib753))], and audio [(Chu et al., [2023](#bib.bib268); [2024](#bib.bib708))], natively omnimodal systems that jointly process and generate across all modalities have drawn substantial attention [(OpenAI, [2024](#bib.bib258); Comanici et al., [2025](#bib.bib756); Xu et al., [2025a](#bib.bib752); [b](#bib.bib815))].
However, existing models predominantly operate within passive perception-response paradigms and exhibit limited capacity for scalable agentic behavior, real-time interaction, autonomous tool utilization, and cross-modal reasoning, which are essential prerequisites for practical deployment.

In this report, we present Qwen3.5-Omni, Qwen’s latest generation of fully omnimodal LLM, supporting the understanding of text, images, audio, and audio-visual content. Natively pretrained in an omnimodal manner on massive amounts of text, visual data, and more than 100 million hours of audio-visual data, Qwen3.5-Omni is designed as a native omni agent model: it not only perceives and reasons across all modalities, but also acts, autonomously invoking WebSearch, executing complex FunctionCall, generating speech outputs, and engaging in real-time streaming interaction. The model series includes Plus and Flash variants, all of which are instruct models with 256k-token long-context input.

Qwen3.5-Omni builds on the Thinker–Talker architecture introduced in Qwen2.5-Omni [(Xu et al., [2025a](#bib.bib752))] and introduces five key technical upgrades over Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))]: (1) both the Thinker and Talker adopt Hybrid-Attention Mixture-of-Experts (MoE) designs, enabling highly efficient inference; (2) supporting long-context modeling up to 256k tokens, supporting more than 10 hours of audio and over 400 seconds of 720P audio-visual content at 1 FPS; (3) on the speech generation side, a multi-codebook codec representation enables single-frame, immediate synthesis; (4) the Talker introduces ARIA, a technique that dynamically aligns text and speech units during streaming decoding, significantly improving naturalness and robustness; and (5) multilingual training is substantially expanded, covering 113 languages and dialects for speech recognition and 36 for speech synthesis.

Enabled by these technical advances, Qwen3.5-Omni delivers three major new capabilities over Qwen3-Omni: (1) controllable audio-visual captioning, capable of generating controllable, detailed, and structured captions as well as screenplay-level fine-grained descriptions, including automatic segmentation, timestamp annotation, and detailed descriptions of characters and their relationship to audio; (2) comprehensive real-time interaction, encompassing semantic interruption through native turn-taking intent recognition, end-to-end voice control over volume, speed, and emotion, and voice cloning from user-provided samples; and (3) native omnimodal agentic behavior, including autonomous WebSearch, complex FunctionCall invocation, and Audio-Visual Vibe Coding, an emergent capability wherein the model directly generates executable code from audio-visual instructions, enabling the model to respond to real-time queries without external orchestration.

Critically, Qwen3.5-Omni maintains state-of-the-art performance on text and visual modalities without degradation relative to same-size single-model Qwen counterparts. Across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, covering audio-visual benchmarks, audio benchmarks, ASR benchmarks, language-specific speech-to-text translation tasks, and language-specific ASR tasks, Qwen3.5-Omni-Plus achieves SOTA results, surpassing Gemini-3.1 Pro across general audio understanding, reasoning, recognition, translation, and dialogue, while its overall audio-visual understanding reaches the level of Gemini-3.1 Pro.

## 2 Architecture

![Figure](2604.15804v2/figures/model.jpg)

*Figure 2: The overview of Qwen3.5-Omni. Qwen3.5-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra–low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.*

### 2.1 Overview

As shown in Figure [2](#S2.F2), Qwen3.5-Omni continues to adopt the Thinker-Talker architecture [(Xu et al., [2025a](#bib.bib752))]. Compared with Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], Qwen3.5-Omni introduces several key improvements in scalability, alignment, and real-time interaction:

-
•

The overall backbone adopts a Hybrid Mixture-of-Experts (MoE) design, improving scalability while better balancing capacity and efficiency across multimodal understanding and generation.

-
•

The Thinker receives visual and audio signals through the Vision Encoder and AuT, respectively. Audio and video inputs are interleaved for unified multimodal modeling, with explicit timestamps inserted to improve temporal perception, especially for long video or audio-video contexts. This design enables the Thinker to handle extended inputs, supporting up to 256k tokens, 10 hours of audio, or 400 seconds of 720P video at 1 FPS.

-
•

The Talker is responsible for contextual speech generation by conditioning on multimodal inputs together with the textual outputs from the Thinker. Qwen3.5-Omni adopts the RVQ-based speech representation introduced in Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], which substantially improves inference efficiency.

-
•

To support real-time interaction, Qwen3.5-Omni adopts both chunk-wise streaming input processing in the Thinker and a streaming Talker design, enabling low-latency end-to-end multimodal conversation.

-
•

Different from the dual-track Talker input design in Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], the Talker in Qwen3.5-Omni adopts ARIA to dynamically align text and speech units before interleaving them. This design mitigates the instability caused by mismatched tokenization rates between text and speech, thereby reducing issues such as skipped words, incorrect pronunciations, and ambiguous rendering of numbers.

In the following sections, we first introduce with the AuT encoder, including its training methodology. Then, describe how Thinker processes various inputs. We then detail Talker’s multi-codebook streaming speech generation. Finally, we highlight a series of improvements on both the understanding and generation modules aimed at achieving ultra–low-latency, end-to-end streaming audio inference.

### 2.2 Audio Transformer (AuT)

![Figure](2604.15804v2/figures/3.5aut.png)

*Figure 3: The overview of AuT. Consuming 40 million hours of supervised data especially more multilingual data, AuT encoder in Qwen3.5-Omni obtain stronger general purpose audio representation in 6.25Hz.*

We use transformer based audio encoder trained from scratch in attention-encoder-decoder model AuT, as is shown in Figure [3](#S2.F3). The training of Qwen3.5-Omni encoder consumed 40 million hours of audio-text pair data generated by Qwen3-ASR. The filter bank features of the audio are downsampled 16 times using 4 Conv2D blocks and then fed into self-attention layers to obtain audio tokens in 6.25Hz token rate. Comparing to the training process of Qwen3-Omni encoder, the encoder of Qwen3.5-Omni adapts more multilingual data of more than 20 languages, and the proportion of Chinese, English and multilingual data comes to 3.5 : 3.5 : 3. The dynamic attention window size training mechanism is adopted for guaranting balance performance of inference under real-time prefill caching and for the offline audio understanding tasks.

### 2.3 Perceivation

##### Text, Audio, Image and Video (w/o Audio).

The Thinker converts text, audio, image, and silent video inputs into a unified sequence of representations. For text, we use the Qwen3.5 tokenizer [(Team, [2026](#bib.bib816))], which adopts byte-level byte-pair encoding with a vocabulary size of 250k (up from 150k), improving encoding and decoding efficiency by 10–60% across most languages. For audio inputs, including audio extracted from video, we resample the waveform to 16 kHz and convert it into a 128-channel mel-spectrogram using a 25 ms window and a 10 ms hop size. We use AuT as the audio encoder, trained from scratch on 40 million hours of audio data, where each output frame corresponds to approximately 160 ms of the original signal. For visual inputs, we adopt the vision encoder from Qwen3.5 [(Team, [2026](#bib.bib816))] to process both images and videos. Trained on a mixture of image and video data, this encoder provides strong capabilities in both image understanding and video comprehension. To preserve video information as much as possible while maintaining alignment with the audio stream, we sample video frames at a dynamic frame rate.

##### Audio-visual Timestamp.

Following Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], we apply TM-RoPE to endow the model with temporal awareness for audio-video synchronization. However, we find that directly encoding absolute time through temporal position IDs can lead to excessively sparse indices for visual patches from long video with audio inputs, which weakens long-range temporal modeling. In addition, such a design often requires large-scale and uniformly distributed training samples across different frame rates, increasing data construction cost. To address these issues, we prepend each video or audio-video temporal patch with an explicit timestamp represented as a formatted text string in seconds, allowing the model to learn timecode representations more naturally. For audio sequences, we further insert timestamps at random intervals to improve temporal alignment across modalities. Although this strategy slightly increases the context length, it enables more precise and robust temporal perception, especially when extrapolating long-context multimodal inputs.

In the context of multimodal audio-visual streams, the audio component is encoded with a temporal ID for every 160 ms. The video is treated as a sequence of frames with monotonically increasing temporal IDs that are dynamically adjusted based on their actual timestamps to ensure a consistent temporal resolution of 160 ms per ID. The height and width IDs for video frames are assigned in the same manner as for still images. To prevent positional conflicts when processing multiple modalities, the position numbering is made contiguous, with each subsequent modality commencing from one plus the maximum position ID of the preceding modality. This refined approach to positional encoding enables the model to effectively integrate and jointly model information from diverse modalities. Qwen3.5-Omni aligns these representations using their temporal IDs, which are explicitly anchored to absolute time. This design choice affords the model the flexibility to support streaming inputs of arbitrary duration.

### 2.4 Speech Generation

Talker operates directly on the RVQ tokens produced by Qwen3.5-Omni-Audio-Tokenizer. To model the residual codebooks, it employs a multi-token prediction (MTP) module, which enables fine-grained modeling and control of acoustic details. Coupled with a causal ConvNet for waveform reconstruction, Talker delivers high-fidelity speech synthesis with low inference latency and modest computational overhead.

In multi-turn spoken dialogue, Talker is conditioned on the rich contextual information provided by the Thinker component, including historical text tokens, multimodal representations, and the streamed text of the current turn. Such conditioning allows Talker to dynamically modulate acoustic attributes—such as prosody, loudness, and emotion—in accordance with the evolving conversational context.

Architecturally, our approach differs from Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))] in two key respects. First, we introduce a dedicated system prompt for Talker that specifies target voice characteristics, thereby enabling both zero-shot voice cloning and controllable speech generation. Compared with conventional speaker embeddings, this prompt can encode richer multimodal cues, including textual descriptions and codec sequences, providing substantially finer-grained control over acoustic realization. Second, we propose ARIA (Adaptive Rate Interleave Alignment), which unifies the conventional dual-channel generation paradigm into a single-channel formulation. Rather than relying on MFA-derived alignments or fixed interleaving rates, ARIA enforces an adaptive rate constraint: for any prefix of the generated sequence, the cumulative speech-to-text token ratio must not exceed the corresponding item-level global ratio. Despite its simplicity, this design affords flexible text-speech alignment across languages, including those with relatively low encoding efficiency, and naturally supports arbitrary text-token prefixes followed by coherent speech-token continuation.

### 2.5 Designs for Streaming and Concurrency

In streaming audio-visual interaction scenarios, the first-packet latency is a critical factor affecting user experience, and the model’s concurrency capability is key to reducing service costs and improving response speed. This section discusses how Qwen3.5-Omni enhances concurrency and reduces first-packet latency through algorithmic and architectural optimizations. Table [1](#S2.T1) provides an overview of the relevant architecture of the Qwen3.5-Omni and its associated latency.

*Table 1: Architecture of Qwen3.5-Omni and end-to-end first-packet latency under audio/video settings (ms).*

##### Chunked Prefilling and Hybrid MoE Architecture.

In Qwen3.5-Omni, we retain the chunked-prefilling mechanism as implemented in Qwen3-Omni and Qwen2.5-Omni, whose audio and vision encoders are capable of outputting chunks along the temporal dimension. This approach significantly reduces the Time-To-First-Token (TTFT) for both the Thinker and the Talker. Architecturally, both the Thinker and the Talker in Qwen3.5-Omni are built upon the Hybrid MoE architecture introduced in Qwen3.5. Beyond the general efficiency advantage of Hybrid MoE, this architecture includes the Gated Delta Net (GDN) module, which is particularly effective for accelerating the modeling of long audio-video sequences. As a result, it significantly reduces KV-cache I/O overhead in long-context inference, improving generation throughput and enabling higher serving concurrency.

##### Streaming Generation with ARIA.

For streaming speech generation and high-concurrency serving, Qwen3.5-Omni largely inherits the efficient design of Qwen3-Omni: Talker predicts RVQ codec tokens with a lightweight MTP module, and the generated multi-codebook tokens are converted to waveform by a causal and streaming ConvNet codec decoder. These components remain computationally lightweight, batch-friendly, and well-suited for low-latency deployment. Built on this shared foundation, the previously introduced ARIA further reformulates the dual-channel generation pattern in Qwen3-Omni into a unified interleaved single-stream formulation over text and speech tokens. By organizing text and speech generation under a monotonic interleaving constraint, ARIA reduces the synchronization overhead between separate generation tracks, enables more efficient token scheduling during decoding, and better matches the naturally incremental regime of streaming interaction.

In Table [2](#S2.T2), we report the theoretical first-packet latency of Qwen3.5-Omni under different concurrency levels for audio and video input, evaluated on internal vLLM with torch.compile and CUDA Graph acceleration enabled for the MTP module and codec decoder. Here, Thinker TTFT (Time-To-First-Token) denotes the time from receiving the input stream to the first text token generated by Thinker, while Talker TTFC (Time-To-First-Chunk) measures the time until Talker produces the first audio chunk. TPOP (Time-Per-Output-Token) represents the per-output-token latency during steady-state decoding, where Talker TPOP includes the combined latency of the Talker backbone and the MTP module. TPS (Tokens Per Second) denotes generation throughput. Since ARIA organizes text and speech generation in a unified interleaved stream, Overall Latency cannot be obtained by simply summing several row values, but instead reflects the end-to-end critical path to the first playable audio packet. We also note that, due to the substantial scale difference between Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus, the two variants adopt different deployment-time resource allocation and parallelization strategies; therefore, their latency and throughput numbers are not intended for strict horizontal comparison. As shown in the table, Qwen3.5-Omni maintains stable latency and decoding efficiency as concurrency increases, while the low Generation RTF provides sufficient margin for smooth streaming audio generation.

*Table 2: Theoretical first-packet latency of Qwen3.5-Omni under different concurrency levels. A/V denotes audio/video input.*

## 3 Pretraining

*Table 3: Supported languages and dialects in Qwen3.5-Omni-Plus.*

Qwen3.5-Omni is pre-trained on a diverse dataset that encompasses multiple languages and dialects as shown in Table [3](#S3.T3) and modalities, including image-text, video-text, audio-text, video-audio, video-audio-text, and pure text corpora. Following Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], we employ a wider range of natural language prompts to enhance both the generalization ability and instruction-following capabilities. To achieve robust performance across all modalities, our training strategy incorporates both unimodal and cross-modal data from the early pretraining stage.

In Qwen3-Omni [(Xu et al., [2025b](#bib.bib815))], we employ TMRoPE to endow the model with temporal awareness. However, we identify two key limitations of this approach: (1) By directly tying temporal position IDs to absolute time, it produces excessively large and sparse temporal position IDs for long audio-video or video inputs, which undermines the model’s ability to capture long-range temporal contexts. (2) Effective learning under this scheme typically requires large-scale and uniformly distributed sampling across different frame rates (fps), significantly increasing the cost of training data construction. To address these issues, we prepend each video or audio-video temporal patch with a timestamp represented as a formatted text string in seconds, enabling the model to better learn and interpret timecode representations. In addition, for audio sequences, we insert timestamps at random intervals to better align training across different modalities. Although this approach introduces a modest increase in context length, it allows the model to perceive temporal information more effectively and precisely.

The pre-training of Qwen3.5-Omni is structured into three distinct stages. In the first stage, we lock the LLM parameters and focus on training the vision and audio encoders, utilizing a vast corpus of audio-text and image-text pairs to enhance semantic understanding within the LLM. In the second stage, we unfreeze all parameters and train with a wider range of multimodal data for more comprehensive learning with a sequence length of 32,768. In the final stage, we use data with a sequence length of 262,144 to enhance the model’s ability to understand complex long-sequence data:

-
(1)

Encoder Alignment Stage (S1): During the initial pretraining phase, the LLM component of Qwen3.5-Omni is initialized with parameters from Qwen3.5, while the vision encoder is adopted from Qwen3.5, and the audio encoder is initialized with AuT. The two encoders are trained separately on the fixed LLM, with both initially focusing on training their respective adapters before training the encoders.

-
(2)

General Stage (S2): The second phase of pretraining utilizes a large-scale dataset containing approximately 4 trillion tokens, with the following distribution across modalities: text (0.92 trillion), audio (1.99 trillion), image (0.95 trillion), video (0.14 trillion), and video-audio 0.29 trillion). During this stage, the introduction of more diverse multimodal data and tasks enhances the model’s understanding and interaction capabilities in auditory, visual, textual, and audio-visual information.

-
(3)

Long Context Stage (S3): In the final pre-training phase, we increased the maximum token length from 32,768 to 262,144 and also raised the proportion of long audio and long video in the training data. Experimental results indicate that these adjustments lead to significant improvements in the model’s ability to understand long sequence data.

## 4 Post-training

### 4.1 Thinker

The post-training phase employs a three-stage strategy for the Thinker, designed to preserve the model’s capabilities across all modalities without degradation, ensure high response quality under audio queries, and optimize the overall interaction experience. The training corpus, structured in the ChatML [(OpenAI, [2022](#bib.bib33))] format, encompasses pure text, visual, audio, and mixed-modality conversational data. Specifically, the process consists of the following stages:

-
•

Stage 1: Specialist Distillation
To establish a strong foundation for omnimodal capabilities, we first train a suite of domain-specialized teacher models via independent Supervised Fine-Tuning (SFT) and reinforcement learning (RL). All teacher models are fine-tuned from the pre-trained Qwen-3.5 base checkpoint. Beyond text-related tasks, including agentic, coding, and foundational reasoning tasks, we also train specialized teacher models for vision and audio. These teacher models are used to generate domain-specific data, enabling the specialized capabilities learned in each domain to be distilled into a single unified model.

-
•

Stage 2: On-Policy Distillation
Through the specialist distillation described above, the model already achieves strong performance in domains such as multimodal understanding and reasoning, as well as text-based dialogue, reasoning, coding, and agentic tasks. Nevertheless, a substantial gap remains between the quality of responses conditioned on audio queries and that of responses conditioned on text queries, particularly in speech dialogue. To reduce this gap, we introduce a second-stage training procedure based on on-policy distillation (OPD), with the goal of distilling the model’s stronger response capabilities under text inputs into the audio-input setting. Concretely, for each audio-text paired query, we first obtain a response generated under the text condition, which typically exhibits higher quality in terms of fluency, reasoning, and task completion. We then use this response as the distillation target for the corresponding audio-conditioned query. By training on such on-policy targets, the model gradually aligns its audio-conditioned outputs with its text-conditioned behavior, thereby improving response quality under audio inputs and promoting modality-consistent generation.

-
•

Stage 3: Interaction-Aligned Reinforcement Learning
Although the previous two stages substantially improve the model’s domain capabilities and cross-modal response quality, they are not sufficient to fully optimize the model for real-world interactive use. In multi-turn conversations, we observe several interaction-specific issues, including unintended language code-switching, persona inconsistency, and degraded instruction-following over extended contexts. To mitigate these issues, we introduce Interaction-Aligned RL, a third-stage reinforcement learning procedure aimed at optimizing the model for interaction quality. We construct multi-turn interaction trajectories and design reward signals around these user experience objectives, enabling the model to learn behaviors that are more stable, consistent, and aligned in prolonged interactions. By explicitly optimizing for interaction quality, this stage improves the model’s overall usability in practical conversational scenarios.

### 4.2 Talker

We employ a four-stage training pipeline for Talker, enabling Qwen3.5-Omni to generate natural and contextually appropriate spoken responses jointly with text. All training data is organized in the ChatML format to maintain consistency with Thinker and to facilitate voice cloning.

-
(1)

General Stage: In the initial pre-training stage, we train Qwen3.5-Omni on more than 20 million hours of multilingual speech data paired with multimodal context. In particular, the introduction of more diverse tasks, such as instruction-following speech generation, substantially enhance contextual reasoning and paralinguistic alignment, going beyond a simple monotonic mapping from multimodal representations to speech.

-
(2)

Long-Context Stage: We perform data quality stratification through a dedicated curation pipeline and conduct continual pre-training (CPT) on high-quality subsets. Augmented by Qwen3-Omni-Captioner, this stage mitigates hallucinations introduced by noisy data in the initial pre-training phase and substantially improves the naturalness and quality of generated speech. Meanwhile, we extend the maximum context length to 64k tokens, allowing the model to better handle long and complex user inputs and to produce more contextually grounded speech responses.

-
(3)

Reinforcement Learning Stage: We further align model behavior with human preferences through Direct Preference Optimization (DPO) [(Rafailov et al., [2023](#bib.bib18))]. Concretely, we construct multilingual preference pairs based on human annotations and optimize the model with DPO. In addition, we incorporate rule-based rewards and adopt GSPO [(Zheng et al., [2025](#bib.bib765))] to further improve overall capability and training stability across diverse tasks.

-
(4)

Speaker Fine-tuning Stage: Finally, we perform lightweight speaker fine-tuning on top of the base model, enabling Qwen3.5-Omni to faithfully capture target speaker characteristics while further improving the naturalness, expressiveness, and controllability of its speech outputs.

## 5 Evaluation

A comprehensive evaluation was performed on two variants of models, including Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus. The evaluation results are divided into two main categories: understanding (X$\to$Text) and speech generation (X$\to$Speech).

### 5.1 Evaluation of X$\to$Text

In this section, we evaluate Qwen3.5-Omni’s ability to comprehend various multimodal inputs (text, audio, vision, and audio-visual video) and generate textual responses.

##### Text$\to$Text

Our evaluation of Qwen3.5-Omni on text $\to$ text primarily focuses on general knowledge tasks, instruction following, long context tasks, STEM tasks, reasoning tasks and general agent ability.
Specifically, we utilize
MMLU-Pro [(Wang et al., [2024d](#bib.bib276))], MMLU-Redux [(Gema et al., [2024](#bib.bib73))], SuperGPQA [(Team et al., [2025](#bib.bib779))] and C-Eval [(Huang et al., [2023](#bib.bib89))] for general knowledge tasks,
IFEval [(Zhou et al., [2023](#bib.bib274))] and IFBench [(Pyatkin et al., [2025](#bib.bib801))] for instruction following,
AA-LCR [(Team, [2025a](#bib.bib802))] and LongBench v2 [(Bai et al., [2025b](#bib.bib803))] for long context tasks,
GPQA [(Rein et al., [2023](#bib.bib271))] for STEM tasks,
LiveCodeBench v6 [(Jain et al., [2024](#bib.bib148))], HMMT Nov 25 [(Balunović et al., [2025](#bib.bib804))] and IMOAnswerBench [(Luong et al., [2025](#bib.bib805))] for reasoning tasks,
BFCL-V4 [(Yan et al., [2024](#bib.bib761))] and TAU2Bench [(Barres et al., [2025](#bib.bib780))] for general agent ability.

##### Audio$\to$Text

To evaluate audio-to-text capabilities, we employ benchmarks across four domains: audio understanding, end-to-end speech dialogue, speech-to-text translation (S2TT), and automatic speech recognition (ASR). For audio understanding, we utilize MMAU [(Sakshi et al., [2024](#bib.bib742))], MMAR [(Ma et al., [2025a](#bib.bib777))], MMSU [(Wang et al., [2025a](#bib.bib778))], RUL-MuchoMusic [(Zang et al., [2025](#bib.bib771))], and SongFormBench [(Hao et al., [2025](#bib.bib806))] to assess comprehension of sound effects, speech, and music. Dialogue performance is evaluated via VoiceBench [(Chen et al., [2024b](#bib.bib706))], URO-Bench-pro [(Yan et al., [2025](#bib.bib809))], SpeechRole [(Jiang et al., [2025](#bib.bib808))], and WildSpeech-Bench [(Zhang et al., [2025b](#bib.bib807))]. For S2TT, we focus on the translation of the top 59 languages in Fleurs [(Conneau et al., [2022](#bib.bib694))] into English and Chinese. Finally, ASR performance is measured using Fleurs [(Conneau et al., [2022](#bib.bib694))], Common Voice [(Ardila et al., [2020](#bib.bib814))], LibriSpeech [(Panayotov et al., [2015](#bib.bib664))], WenetSpeech [(Zhang et al., [2022](#bib.bib810))], KeSpeech [(Tang et al., [2021](#bib.bib812))], Opencpop-test [(Wang et al., [2022](#bib.bib811))], and MIR-1K (vocal) [(Hsu and Jang, [2010](#bib.bib813))], covering multilingual speech, Chinese dialects, and singing voice transcription.

##### Vision$\to$Text

The evaluation of the model’s vision-to-text capabilities encompasses a suite of benchmarks targeting diverse and challenging tasks. To assess performance in specialized domain of mathematical and STEM reasoning, we utilize MMMU [(Yue et al., [2023](#bib.bib322))], MMMU-Pro [(Yue et al., [2024](#bib.bib517))], MathVista [(Lu et al., [2024](#bib.bib459))], MathVision [(Wang et al., [2024a](#bib.bib460))], DynaMath [(Zou et al., [2025](#bib.bib817))], ZEROBench [(Roberts et al., [2025](#bib.bib818))]. For the general visual question answering, the model is evaluated on RealWorldQA [(Zhang et al., [2024](#bib.bib515))], MMStar [(Chen et al., [2024a](#bib.bib307))], HallusionBench [(Guan et al., [2024](#bib.bib787))], and SimpleVQA [(Cheng et al., [2025](#bib.bib819))]. The model’s proficiency in document understanding is measured using the CharXiv [(Wang et al., [2024e](#bib.bib582))], CC-OCR [(Yang et al., [2024b](#bib.bib820))], AI2D [(Kembhavi et al., [2016](#bib.bib427))], MMLongBench-Doc [(Ma et al., [2024](#bib.bib821))], and OCRBench [(Liu et al., [2024](#bib.bib578))]. Furthermore, the model’s spatial intelligence is specifically tested on ERQA [(Team, [2025b](#bib.bib822))], CountBench [(Paiss et al., [2023](#bib.bib786))], RefCOCO [(Kazemzadeh et al., [2014](#bib.bib385))], ODInW13 [(Li et al., [2022](#bib.bib823))], and EmbSpatialBench [(Du et al., [2024a](#bib.bib824))]. To evaluate performance on dynamic visual data, we report results on six video understanding benchmarks: Video-MME [(Fu et al., [2024](#bib.bib298))], MLVU [(Zhou et al., [2025a](#bib.bib789))], MVBench [(Li et al., [2024](#bib.bib300))], LVBench [(Wang et al., [2024b](#bib.bib788))], MMVU [(Zhao et al., [2025](#bib.bib825))] and MME-VideoOCR [(Shi et al., [2025](#bib.bib826))]. Specifically, we evaluate the model’s performance on medical VQA across three established benchmarks: SLAKE [(Liu et al., [2021](#bib.bib827))], PMC-VQA [(Zhang et al., [2023](#bib.bib828))], and MedXpertQA-MM [(Zuo et al., [2025](#bib.bib829))]. This assessment is designed to demonstrate the model’s comprehensive clinical reasoning capabilities and its potential utility as a reliable healthcare AI assistant.

##### Audio-Visual Video$\to$Text

We evaluate our model’s audio-visual understanding capabilities from multiple perspectives. For text-query evaluation, we use DailyOmni [(Zhou et al., [2025b](#bib.bib790))], WorldSense [(Hong et al., [2025](#bib.bib791))], AVUT [(Yang et al., [2025b](#bib.bib797))], AV-SpeakerBench [(Nguyen et al., [2025](#bib.bib796))], and VideoMME [(Fu et al., [2025](#bib.bib799))]. To assess the model’s ability in real-world audio-visual interactive scenarios, we use Qualcomm IVD [(Pourreza et al., [2025](#bib.bib798))] as the benchmark for audio-query-based evaluation. Beyond understanding, we also evaluate the model’s captioning capability on OmniCloze [(Ma et al., [2025b](#bib.bib800))] and its tool-use ability on OmniGAIA [(Li et al., [2026](#bib.bib795))].

#### 5.1.1 Performance of Text$\to$Text

We compare Qwen3.5-Omni-Plus and Qwen3.5-Omni-Flash with Qwen3.5-Plus-Instruct. As shown in Table [4](#S5.T4), Qwen3.5-Omni-Plus demonstrates text capabilities that are on par with its text-only counterpart across multiple dimensions, including knowledge, instruction following, long-context understanding, STEM, reasoning, and general agent tasks, highlighting its strong language ability. In particular, Qwen3.5-Omni ’s instruction-following performance is slightly better than the baseline. We believe that OPD and interaction-aligned RL have a positive effect on improving the instruction-following capabilities of an omni-model LLM.

*Table 4: Text $\to$ Text performance of Qwen3.5-Omni and Qwen3.5-Plus-Instruct. The highest scores are shown in bold.*

#### 5.1.2 Performance of Audio$\to$Text

*Table 5: Audio benchmark comparison across Gemini-3.1 Pro, Qwen3.5-Omni-Flash, and Qwen3.5-Omni-Plus. For most benchmarks, higher is better. For ASR benchmarks, lower WER is better. Best results are shown in bold.*

In Table [5](#S5.T5), we compare Qwen3.5-Omni with Gemini-3.1 Pro in terms of audio-to-text performance. Compared to Gemini-3.1 Pro, Qwen3.5-Omni exhibits superior performance on MMAU, MMSU, RUL-MuchoMusic, and SongFormBench, while achieving comparable results on MMAR, demonstrating its strong comprehension capabilities across multiple audio domains. Regarding end-to-end speech dialogue, Qwen3.5-Omni significantly outperforms Gemini-3.1 Pro on VoiceBench and matches its performance on other benchmarks, further validating Qwen3.5-Omni ’s robust capabilities in end-to-end voice interaction. For S2TT and ASR, Qwen3.5-Omni consistently outperforms Gemini-3.1 Pro, underscoring its superior translation and speech recognition performance across diverse languages, dialects, and domains.

#### 5.1.3 Performance of Vision $\to$ Text

To comprehensively evaluate vision-to-text capabilities, we compare Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus with Qwen3.5-Plus-Instruct. As shown in Table [6](#S5.T6), Qwen3.5-Omni-Plus achieves performance comparable to that of Qwen3.5-Plus-Instruct, while demonstrating stronger results on video understanding tasks involving both short and long videos. These findings highlight the strong dynamic visual perception ability of our model in real-world scenarios and suggest the effectiveness of joint video-audio training paradigms. Furthermore, we posit that audio-visual streams constitute the most naturalistic representation of real-world phenomena, wherein visual and auditory modalities are intrinsically coupled rather than independently processed.

*Table 6: Vision $\to$ Text performance of Qwen3.5-Omni and Qwen3.5-Plus-Instruct. The highest scores are shown in bold.*

#### 5.1.4 Performance of Audio-Visual Video$\to$Text

We compare Qwen3.5-Omni and Gemini-3.1 Pro across a diverse range of audio-visual tasks, as shown in Table [7](#S5.T7). For general understanding, Qwen3.5-Omni achieves state-of-the-art performance on DailyOmni and obtains comparable results on AVUT. Our model also surpasses Gemini-3.1-Pro by a substantial margin on Qualcomm IVD, demonstrating its effectiveness in real-world audio-visual interactive scenarios.
Moreover, our model shows strong performance on captioning tasks. It can provide detailed audio, visual, and audio-visual captions. In this version, we also enhance the model’s tool-use capability, achieving 57.2% on OmniGAIA.

*Table 7: Audio-Visual $\to$ Text performance of Qwen3.5-Omni and Gemini-3.1-Pro. The highest scores are shown in bold.*

### 5.2 Evaluation of X$\to$Speech

In this section, we evaluate the speech generation capability of Qwen3.5-Omni. Our evaluation mainly focuses on speech generation conditioned on text and prompt speech, following a zero-shot text-to-speech (TTS) setting. We study the model from four perspectives:

-
•

Zero-Shot Speech Generation: We evaluate content consistency, measured by WER, on SEED [(Anastassiou et al., [2024](#bib.bib723))].

-
•

Multilingual Speech Generation: We evaluate both content consistency and speaker similarity in zero-shot multilingual speech generation on the TTS multilingual test set [(Zhang et al., [2025a](#bib.bib731))] and an internal multilingual test set built on FLEURS [(Conneau et al., [2022](#bib.bib694))].

-
•

Cross-Lingual Speech Generation: We evaluate content consistency in zero-shot cross-lingual speech generation on CV3-Eval [(Du et al., [2025](#bib.bib729))].

-
•

Custom-Voice Speech Generation: We evaluate the stability of our speaker fine-tuned model on the TTS multilingual test set [(Zhang et al., [2025a](#bib.bib731))] and our internal multilingual test set.

#### 5.2.1 Evaluation of Zero-Shot Speech Generation

We compare Qwen3.5-Omni with state-of-the-art zero-shot TTS systems. As shown in Table [8](#S5.T8), Qwen3.5-Omni achieves highly competitive performance on the SEED-TTS benchmark, demonstrating strong content fidelity in zero-shot speech generation. These results reflect the effectiveness of our pretraining and continual pretraining pipeline in building robust speech generation and context modeling capabilities. Moreover, after RLHF optimization, Qwen3.5-Omni further improves generation stability and naturalness, achieving the best performance on the test-en split with a WER of 1.26.

*Table 8: Zero-shot speech generation on the SEED-TTS test set. Performance is measured by Word Error Rate (WER, $\downarrow$), where lower is better. The best results are highlighted in bold.*

#### 5.2.2 Evaluation of Multilingual Speech Generation

Qwen3.5-Omni supports speech generation in 29 languages. We compare its multilingual speech generation performance with two strong commercial systems, MiniMax-Speech and ElevenLabs. For the internal multilingual test set, we use GPT-4o-transcribe-2025-03-20 for automatic speech recognition.

As shown in Table [9](#S5.T9) and Table [10](#S5.T10), Qwen3.5-Omni achieves the lowest WER in 22 out of 29 evaluated languages on the multilingual test sets, outperforming the comparison systems by a clear margin in most cases. On the remaining languages, Qwen3.5-Omni remains competitive with state-of-the-art systems. In addition to content consistency, Qwen3.5-Omni also shows strong voice cloning fidelity. It obtains the highest speaker similarity scores in the majority of evaluated languages and consistently outperforms both MiniMax-Speech and ElevenLabs overall. These results suggest that Qwen3.5-Omni effectively preserves speaker characteristics, such as timbre and prosodic style, while maintaining robust multilingual speech generation quality.

Furthermore, in Table [10](#S5.T10), we report results on our internal multilingual test set, covering an additional 9 languages. Qwen3.5-Omni continues to achieve strong performance across all evaluated languages, indicating that its multilingual speech generation ability generalizes well beyond the public benchmark languages.

*Table 9: Multilingual speech generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$) for content consistency and cosine similarity (SIM, $\uparrow$) for speaker similarity. The best results are highlighted in bold.*

*Table 10: Multilingual speech generation on the internal multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$) for content consistency and cosine similarity (SIM, $\uparrow$) for speaker similarity. The best results are highlighted in bold.*

#### 5.2.3 Evaluation of Cross-Lingual Speech Generation

Beyond multilingual voice cloning, Qwen3.5-Omni also supports cross-lingual voice cloning, where the model is required to preserve speaker identity while generating speech in a different target language. We evaluate this capability on the Cross-Lingual benchmark and compare against the CosyVoice series as well as Qwen3-Omni-30B-A3B.

In Table [11](#S5.T11), we report the mixed error rate (WER for English and CER for the other languages) across different source–target language pairs. Overall, Qwen3.5-Omni achieves the best performance in 10 out of 12 evaluated directions and sets a new state of the art on most English-, Japanese-, and Korean-targeted pairs. In particular, for zh-to-ko, Qwen3.5-Omni reduces the error rate from 14.4 to 4.03 compared with CosyVoice3, corresponding to an approximately 72% relative reduction. Qwen3.5-Omni also performs strongly on commonly used language pairs such as zh-to-en and en-to-zh, indicating better content consistency under cross-lingual generation. These results demonstrate that Qwen3.5-Omni generalizes effectively across language boundaries while preserving target linguistic accuracy.

*Table 11: Cross-lingual speech generation on the Cross-Lingual benchmark. Performance is measured by mixed error rate (WER for English and CER for the other languages, $\downarrow$). The best results are highlighted in bold.*

#### 5.2.4 Evaluation of Custom-Voice Speech Generation

We evaluate the custom-voice speech generation capability of Qwen3.5-Omni in multilingual settings. We compare Qwen3.5-Omni with several strong commercial systems accessed through their official APIs in March 2026, including ElevenLabs Multilingual v2 (9YHcvj6GT2YYXdXww), Gemini-2.5 Pro-Preview-TTS (Achernar), GPT-Audio-2025-08-28 (Alloy), and MiniMax-Speech-2.8-HD (English_expressive_narrator).

*Table 12: Custom-voice multilingual speech generation on the multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$). The best results are highlighted in bold.*

As shown in Table [12](#S5.T12), although Qwen3.5-Omni is fine-tuned only on monolingual data, it demonstrates strong cross-lingual generalization in custom-voice speech generation. The model is able to transfer the target speaker characteristics to all 29 evaluated languages while maintaining stable generation quality. Overall, Qwen3.5-Omni achieves the best WER in 10 languages and remains competitive in many others. In particular, it shows clear advantages in several challenging languages, including Japanese (3.306) and Korean (1.309), indicating strong intelligibility under cross-lingual voice transfer. These results suggest that Qwen3.5-Omni can generate custom-voice speech with robust linguistic fidelity across a wide range of languages.

## 6 Conclusion

In this work, we present Qwen3.5-Omni, a fully omnimodal large language model that unifies understanding, reasoning, generation, and action across text, images, audio, and audio-visual inputs. Built on the Thinker–Talker framework, Qwen3.5-Omni introduces efficient Hybrid-Attention MoE architectures, 256k long-context modeling, improved streaming speech generation with multi-codebook codec prediction and ARIA, and substantially expanded multilingual speech support. These advances enable three key capabilities: controllable audio-visual captioning, comprehensive real-time interaction, and native omnimodal agentic behavior through autonomous tool use and audio-visual code generation. Empirically, Qwen3.5-Omni achieves state-of-the-art or highly competitive performance across a broad range of audio and audio-visual benchmarks, while maintaining the strong text and vision capabilities of same-scale Qwen models. These results suggest that scaling native omnimodal training can produce unified systems that not only perceive and reason across modalities, but also interact and act in real time. We hope Qwen3.5-Omni provides a strong foundation for future research on general-purpose omnimodal agents.

## References

-
P. Anastassiou, J. Chen, J. Chen, Y. Chen, Z. Chen, Z. Chen, J. Cong, L. Deng, C. Ding, L. Gao, et al. (2024)
Seed-tts: a family of high-quality versatile speech generation models.

arXiv preprint arXiv:2406.02430.

Cited by: [1st item](#S5.I3.i1.p1.1),
[Table 8](#S5.T8.5.3.2),
[Table 8](#S5.T8.5.4.1).

-
Anthropic (2023a)
Claude 2.

Technical report

Anthropic.

External Links: [Link](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)

Cited by: [§1](#S1.p1.1).

-
Anthropic (2023b)
Introducing Claude.

Anthropic.

External Links: [Link](https://www.anthropic.com/index/introducing-claude)

Cited by: [§1](#S1.p1.1).

-
Anthropic (2024)
The Claude 3 model family: Opus, Sonnet, Haiku.

Technical report

Anthropic, AI.

External Links: [Link](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model%5C_Card%5C_Claude%5C_3.pdf)

Cited by: [§1](#S1.p1.1).

-
R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. M. Tyers, and G. Weber (2020)
Common voice: A massively-multilingual speech corpus.

In Proceedings of The 12th Language Resources and Evaluation Conference,
LREC 2020, Marseille, France, May 11-16, 2020, N. Calzolari, F. Béchet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis (Eds.),

pp. 4218–4222.

External Links: [Link](https://aclanthology.org/2020.lrec-1.520/)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu (2023a)
Qwen technical report.

CoRR abs/2309.16609.

Cited by: [§1](#S1.p1.1).

-
J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou (2023b)
Qwen-VL: a frontier large vision-language model with versatile abilities.

CoRR abs/2308.12966.

Cited by: [§1](#S1.p1.1).

-
S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025a)
Qwen2. 5-vl technical report.

arXiv preprint arXiv:2502.13923.

Cited by: [§1](#S1.p1.1).

-
Y. Bai, S. Tu, J. Zhang, H. Peng, X. Wang, X. Lv, S. Cao, J. Xu, L. Hou, Y. Dong, J. Tang, and J. Li (2025b)
LongBench v2: towards deeper understanding and reasoning on realistic long-context multitasks.

In Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria,
July 27 - August 1, 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.),

pp. 3639–3664.

External Links: [Link](https://aclanthology.org/2025.acl-long.183/)

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
M. Balunović, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev (2025)
MathArena: evaluating llms on uncontaminated math competitions.

SRI Lab, ETH Zurich.

External Links: [Link](https://matharena.ai/)

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
V. Barres, H. Dong, S. Ray, X. Si, and K. Narasimhan (2025)
$\tau^{2}$-Bench: evaluating conversational agents in a dual-control environment.

External Links: 2506.07982,
[Link](https://arxiv.org/abs/2506.07982)

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020)
Language models are few-shot learners.

In NeurIPS,

Cited by: [§1](#S1.p1.1).

-
L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, et al. (2024a)
Are we on the right way for evaluating large vision-language models?.

arXiv:2403.20330.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Y. Chen, X. Yue, C. Zhang, X. Gao, R. T. Tan, and H. Li (2024b)
Voicebench: benchmarking llm-based voice assistants.

arXiv preprint arXiv:2410.17196.

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen (2024c)
F5-tts: a fairytaler that fakes fluent and faithful speech with flow matching.

arXiv preprint arXiv:2410.06885.

Cited by: [Table 8](#S5.T8.5.7.1).

-
X. Cheng, W. Zhang, S. Zhang, J. Yang, X. Guan, X. Wu, X. Li, G. Zhang, J. Liu, Y. Mai, Y. Zeng, Z. Wen, K. Jin, B. Wang, W. Zhou, Y. Lu, T. Li, W. Huang, and Z. Li (2025)
SimpleVQA: multimodal factuality evaluation for multimodal large language models.

CoRR abs/2502.13059.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, et al. (2024)
Qwen2-audio technical report.

arXiv preprint arXiv:2407.10759.

Cited by: [§1](#S1.p1.1).

-
Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and J. Zhou (2023)
Qwen-Audio: advancing universal audio understanding via unified large-scale audio-language models.

CoRR abs/2311.07919.

Cited by: [§1](#S1.p1.1).

-
G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. (2025)
Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.

arXiv preprint arXiv:2507.06261.

Cited by: [§1](#S1.p1.1).

-
A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna (2022)
FLEURS: few-shot learning evaluation of universal representations of speech.

2022 IEEE Spoken Language Technology Workshop (SLT), pp. 798–805.

External Links: [Link](https://api.semanticscholar.org/CorpusID:249062909)

Cited by: [2nd item](#S5.I3.i2.p1.1),
[§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
M. Du, B. Wu, Z. Li, X. Huang, and Z. Wei (2024a)
EmbSpatial-bench: benchmarking spatial understanding for embodied tasks with large vision-language models.

In Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), ACL 2024, Bangkok, Thailand,
August 11-16, 2024, L. Ku, A. Martins, and V. Srikumar (Eds.),

pp. 346–355.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Z. Du, C. Gao, Y. Wang, F. Yu, T. Zhao, H. Wang, X. Lv, H. Wang, C. Ni, X. Shi, K. An, G. Yang, Y. Li, Y. Chen, Z. Gao, Q. Chen, Y. Gu, M. Chen, Y. Chen, S. Zhang, W. Wang, and J. Ye (2025)
CosyVoice 3: towards in-the-wild speech generation via scaling-up and post-training.

CoRR abs/2505.17589.

Cited by: [3rd item](#S5.I3.i3.p1.1),
[Table 8](#S5.T8.5.10.1).

-
Z. Du, Y. Wang, Q. Chen, X. Shi, X. Lv, T. Zhao, Z. Gao, Y. Yang, C. Gao, H. Wang, et al. (2024b)
CosyVoice 2: scalable streaming speech synthesis with large language models.

arXiv preprint arXiv:2412.10117.

Cited by: [Table 8](#S5.T8.5.9.1).

-
A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Rozière, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. M. Kloumann, I. Misra, I. Evtimov, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, and et al. (2024)
The Llama 3 herd of models.

CoRR abs/2407.21783.

Cited by: [§1](#S1.p1.1).

-
S. E. Eskimez, X. Wang, M. Thakker, C. Li, C. Tsai, Z. Xiao, H. Yang, Z. Zhu, M. Tang, X. Tan, et al. (2024)
E2 tts: embarrassingly easy fully non-autoregressive zero-shot tts.

In 2024 IEEE Spoken Language Technology Workshop (SLT),

pp. 682–689.

Cited by: [Table 8](#S5.T8.5.6.1).

-
C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. (2024)
Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis.

arXiv:2405.21075.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. (2025)
Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 24108–24118.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
A. P. Gema, J. O. J. Leang, G. Hong, A. Devoto, A. C. M. Mancino, R. Saxena, X. He, Y. Zhao, X. Du, M. R. G. Madani, et al. (2024)
Are we done with mmlu?.

CoRR abs/2406.04127.

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
Gemini Team (2024)
Gemini 1.5: unlocking multimodal understanding across millions of tokens of context.

Technical report

Google.

External Links: [Link](https://storage.googleapis.com/deepmind-media/gemini/gemini%5C_v1%5C_5%5C_report.pdf)

Cited by: [§1](#S1.p1.1).

-
T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob, D. Manocha, and T. Zhou (2024)
Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models.

In IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2024, Seattle, WA, USA, June 16-22, 2024,

pp. 14375–14385.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
C. Hao, R. Yuan, J. Yao, Q. Deng, X. Bai, W. Xue, and L. Xie (2025)
SongFormer: scaling music structure analysis with heterogeneous supervision.

External Links: 2510.02797,
[Link](https://arxiv.org/abs/2510.02797)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
J. Hong, S. Yan, J. Cai, X. Jiang, Y. Hu, and W. Xie (2025)
WorldSense: evaluating real-world omnimodal understanding for multimodal llms.

CoRR abs/2502.04326.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
C. Hsu and J. R. Jang (2010)
On the improvement of singing voice separation for monaural recordings using the MIR-1K dataset.

IEEE Trans. Speech Audio Process. 18 (2), pp. 310–319.

External Links: [Link](https://doi.org/10.1109/TASL.2009.2026503),
[Document](https://dx.doi.org/10.1109/TASL.2009.2026503)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
Y. Huang, Y. Bai, Z. Zhu, J. Zhang, J. Zhang, T. Su, J. Liu, C. Lv, Y. Zhang, J. Lei, Y. Fu, M. Sun, and J. He (2023)
C-Eval: a multi-level multi-discipline chinese evaluation suite for foundation models.

In NeurIPS,

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica (2024)
LiveCodeBench: holistic and contamination free evaluation of large language models for code.

CoRR abs/2403.07974.

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
C. Jiang, J. Sun, Y. Cao, J. Zhuang, H. Li, X. Fan, M. Zhang, J. Ye, S. Dou, Z. Xi, et al. (2025)
SpeechRole: a large-scale dataset and benchmark for evaluating speech role-playing agents.

arXiv preprint arXiv:2508.02013.

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg (2014)
Referitgame: referring to objects in photographs of natural scenes.

In EMNLP,

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi (2016)
A diagram is worth a dozen images.

In ECCV,

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
J. Li, D. Li, S. Savarese, and S. Hoi (2023)
Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models.

arXiv:2301.12597.

Cited by: [§1](#S1.p1.1).

-
K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo, et al. (2024)
Mvbench: a comprehensive multi-modal video understanding benchmark.

In CVPR,

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J. Hwang, K. Chang, and J. Gao (2022)
Grounded language-image pre-training.

In CVPR,

pp. 10955–10965.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
X. Li, W. Jiao, J. Jin, S. Wang, G. Dong, J. Jin, H. Wang, Y. Wang, J. Wen, Y. Lu, et al. (2026)
OmniGAIA: towards native omni-modal ai agents.

arXiv preprint arXiv:2602.22897.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
B. Liu, L. Zhan, L. Xu, L. Ma, Y. Yang, and X. Wu (2021)
Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering.

In 18th IEEE International Symposium on Biomedical Imaging, ISBI
2021, Nice, France, April 13-16, 2021,

pp. 1650–1654.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023)
Visual instruction tuning.

arXiv:2304.08485.

Cited by: [§1](#S1.p1.1).

-
Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X. Yin, C. Liu, L. Jin, and X. Bai (2024)
OCRBench: on the hidden mystery of ocr in large multimodal models.

Science China Information Sciences 67 (12).

External Links: ISSN 1869-1919,
[Link](http://dx.doi.org/10.1007/s11432-024-4235-6),
[Document](https://dx.doi.org/10.1007/s11432-024-4235-6)

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K. Chang, M. Galley, and J. Gao (2024)
MathVista: evaluating mathematical reasoning of foundation models in visual contexts.

In ICLR,

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
T. Luong, D. Hwang, H. H. Nguyen, G. Ghiasi, Y. Chervonyi, I. Seo, J. Kim, G. Bingham, J. Lee, S. Mishra, A. Zhai, C. H. Hu, H. Michalewski, J. Kim, J. Ahn, J. Bae, X. Song, T. H. Trinh, Q. V. Le, and J. Jung (2025)
Towards robust mathematical reasoning.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,

External Links: [Link](https://aclanthology.org/2025.emnlp-main.1794/)

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
Y. Ma, Y. Zang, L. Chen, M. Chen, Y. Jiao, X. Li, X. Lu, Z. Liu, Y. Ma, X. Dong, P. Zhang, L. Pan, Y. Jiang, J. Wang, Y. Cao, and A. Sun (2024)
MMLONGBENCH-DOC: benchmarking long-context document understanding with visualizations.

In Advances in Neural Information Processing Systems 38: Annual Conference
on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang (Eds.),

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Z. Ma, Y. Ma, Y. Zhu, C. Yang, Y. Chao, R. Xu, W. Chen, Y. Chen, Z. Chen, J. Cong, K. Li, K. Li, S. Li, X. Li, X. Li, Z. Lian, Y. Liang, M. Liu, Z. Niu, T. Wang, Y. Wang, Y. Wang, Y. Wu, G. Yang, J. Yu, R. Yuan, Z. Zheng, Z. Zhou, H. Zhu, W. Xue, E. Benetos, K. Yu, C. E. Siong, and X. Chen (2025a)
MMAR: A challenging benchmark for deep reasoning in speech, audio, music, and their mix.

CoRR abs/2505.13032.

External Links: [Link](https://doi.org/10.48550/arXiv.2505.13032),
[Document](https://dx.doi.org/10.48550/ARXIV.2505.13032),
2505.13032

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
Z. Ma, R. Xu, Z. Xing, Y. Chu, Y. Wang, J. He, J. Xu, P. Heng, K. Yu, J. Lin, et al. (2025b)
Omni-captioner: data pipeline, models, and benchmark for omni detailed perception.

arXiv preprint arXiv:2510.12720.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
L. T. P. Nguyen, Z. Yu, S. L. Y. Hang, S. An, J. Lee, Y. Ban, S. Chung, T. Nguyen, J. Maeng, S. Lee, et al. (2025)
See, hear, and understand: benchmarking audiovisual human speech understanding in multimodal large language models.

arXiv preprint arXiv:2512.02231.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
OpenAI (2022)
ChatML.

External Links: [Link](https://github.com/openai/openai-python/blob/e389823ba013a24b4c32ce38fa0bd87e6bccae94/chatml.md)

Cited by: [§4.1](#S4.SS1.p1.1).

-
OpenAI (2023)
GPT4 technical report.

CoRR abs/2303.08774.

Cited by: [§1](#S1.p1.1).

-
OpenAI (2024)
Hello GPT-4o.

External Links: [Link](https://openai.com/index/hello-gpt-4o/)

Cited by: [§1](#S1.p1.1).

-
R. Paiss, A. Ephrat, O. Tov, S. Zada, I. Mosseri, M. Irani, and T. Dekel (2023)
Teaching CLIP to count to ten.

In IEEE/CVF International Conference on Computer Vision, ICCV 2023,
Paris, France, October 1-6, 2023,

pp. 3147–3157.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
V. Panayotov, G. Chen, D. Povey, and S. Khudanpur (2015)
Librispeech: an ASR corpus based on public domain audio books.

In 2015 IEEE International Conference on Acoustics, Speech and Signal
Processing, ICASSP 2015, South Brisbane, Queensland, Australia,
April 19-24, 2015,

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
R. Pourreza, R. Dagli, A. Bhattacharyya, S. Panchal, G. Berger, and R. Memisevic (2025)
Can vision-language models answer face to face questions in the real-world?.

arXiv preprint arXiv:2503.19356.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
V. Pyatkin, S. Malik, V. Graf, H. Ivison, S. Huang, P. Dasigi, N. Lambert, and H. Hajishirzi (2025)
Generalizing verifiable instruction following.

CoRR abs/2507.02833.

External Links: [Link](https://doi.org/10.48550/arXiv.2507.02833),
[Document](https://dx.doi.org/10.48550/ARXIV.2507.02833),
2507.02833

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023)
Direct preference optimization: your language model is secretly a reward model.

In NeurIPS,

Cited by: [item (3)](#S4.I2.i3.p1.1).

-
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman (2023)
GPQA: a graduate-level Google-proof Q&A benchmark.

CoRR abs/2311.12022.

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
J. Roberts, M. R. Taesiri, A. Sharma, A. Gupta, S. Roberts, I. Croitoru, S. Bogolin, J. Tang, F. Langer, V. Raina, V. Raina, H. Xiong, V. Udandarao, J. Lu, S. Chen, S. Purkis, T. Yan, W. Lin, G. Shin, Q. Yang, A. T. Nguyen, K. Han, and S. Albanie (2025)
ZeroBench: an impossible visual benchmark for contemporary large multimodal models.

CoRR abs/2502.09696.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
S. Sakshi, U. Tyagi, S. Kumar, A. Seth, R. Selvakumar, O. Nieto, R. Duraiswami, S. Ghosh, and D. Manocha (2024)
MMAU: a massive multi-task audio understanding and reasoning benchmark.

External Links: 2410.19168,
[Link](https://arxiv.org/abs/2410.19168)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
Y. Shi, H. Wang, W. Xie, H. Zhang, L. Zhao, Y. Zhang, X. Li, C. Fu, Z. Wen, W. Liu, Z. Zhang, X. Chen, B. Zeng, S. Yang, Y. Zhang, P. Wan, H. Wang, and W. Yang (2025)
MME-videoocr: evaluating ocr-based capabilities of multimodal llms in video scenarios.

CoRR abs/2505.21333.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Z. Tang, D. Wang, Y. Xu, J. Sun, X. Lei, S. Zhao, C. Wen, X. Tan, C. Xie, S. Zhou, R. Yan, C. Lv, Y. Han, W. Zou, and X. Li (2021)
KeSpeech: an open source speech dataset of mandarin and its eight subdialects.

In Proceedings of the Neural Information Processing Systems Track on
Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December
2021, virtual, J. Vanschoren and S. Yeung (Eds.),

External Links: [Link](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/0336dcbab05b9d5ad24f4333c7658a0e-Abstract-round2.html)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
A. A. Team (2025a)
Artificial analysis long context reasoning benchmark (lcr).

Note: Artificial Analysis, Inc.Dataset

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
G. R. Team (2025b)
Gemini robotics: bringing AI into the physical world.

CoRR abs/2503.20020.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
M.-A-P. Team, X. Du, Y. Yao, K. Ma, B. Wang, T. Zheng, K. Zhu, M. Liu, Y. Liang, X. Jin, Z. Wei, C. Zheng, K. Deng, S. Jia, S. Jiang, Y. Liao, R. Li, Q. Li, S. Li, Y. Li, Y. Li, D. Ma, Y. Ni, H. Que, Q. Wang, Z. Wen, S. Wu, T. Xing, M. Xu, Z. Yang, Z. M. Wang, J. Zhou, Y. Bai, X. Bu, C. Cai, L. Chen, Y. Chen, C. Cheng, T. Cheng, K. Ding, S. Huang, Y. Huang, Y. Li, Y. Li, Z. Li, T. Liang, C. Lin, H. Lin, Y. Ma, T. Pang, Z. Peng, Z. Peng, Q. Qi, S. Qiu, X. Qu, S. Quan, Y. Tan, Z. Wang, C. Wang, H. Wang, Y. Wang, Y. Wang, J. Xu, K. Yang, R. Yuan, Y. Yue, T. Zhan, C. Zhang, J. Zhang, X. Zhang, X. Zhang, Y. Zhang, Y. Zhao, X. Zheng, C. Zhong, Y. Gao, Z. Li, D. Liu, Q. Liu, T. Liu, S. Ni, J. Peng, Y. Qin, W. Su, G. Wang, S. Wang, J. Yang, M. Yang, M. Cao, X. Yue, Z. Zhang, W. Zhou, J. Liu, Q. Lin, W. Huang, and G. Zhang (2025)
SuperGPQA: scaling LLM evaluation across 285 graduate disciplines.

CoRR abs/2502.14739.

External Links: [Link](https://doi.org/10.48550/arXiv.2502.14739),
[Document](https://dx.doi.org/10.48550/ARXIV.2502.14739),
2502.14739

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
Q. Team (2026)
Qwen3.5: accelerating productivity with native multimodal agents.

External Links: [Link](https://qwen.ai/blog?id=qwen3.5)

Cited by: [§2.3](#S2.SS3.SSS0.Px1.p1.1).

-
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. (2023)
Llama 2: open foundation and fine-tuned chat models.

arXiv:2307.09288.

Cited by: [§1](#S1.p1.1).

-
D. Wang, J. Wu, J. Li, D. Yang, X. Chen, T. Zhang, and H. Meng (2025a)
MMSU: A massive multi-task spoken language understanding and reasoning benchmark.

CoRR abs/2506.04779.

External Links: [Link](https://doi.org/10.48550/arXiv.2506.04779),
[Document](https://dx.doi.org/10.48550/ARXIV.2506.04779),
2506.04779

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
K. Wang, J. Pan, W. Shi, Z. Lu, M. Zhan, and H. Li (2024a)
Measuring multimodal mathematical reasoning with math-vision dataset.

arXiv:2402.14804.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
W. Wang, Z. He, W. Hong, Y. Cheng, X. Zhang, J. Qi, S. Huang, B. Xu, Y. Dong, M. Ding, and J. Tang (2024b)
LVBench: an extreme long video understanding benchmark.

CoRR abs/2406.08035.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
X. Wang, M. Jiang, Z. Ma, Z. Zhang, S. Liu, L. Li, Z. Liang, Q. Zheng, R. Wang, X. Feng, W. Bian, Z. Ye, S. Cheng, R. Yuan, Z. Zhao, X. Zhu, J. Pan, L. Xue, P. Zhu, Y. Chen, Z. Li, X. Chen, L. Xie, Y. Guo, and W. Xue (2025b)
Spark-tts: an efficient llm-based text-to-speech model with single-stream decoupled speech tokens.

CoRR abs/2503.01710.

Cited by: [Table 8](#S5.T8.5.8.1).

-
Y. Wang, X. Wang, P. Zhu, J. Wu, H. Li, H. Xue, Y. Zhang, L. Xie, and M. Bi (2022)
Opencpop: A high-quality open source chinese popular song corpus for singing voice synthesis.

In 23rd Annual Conference of the International Speech Communication Association,
Interspeech 2022, Incheon, Korea, September 18-22, 2022, H. Ko and J. H. L. Hansen (Eds.),

pp. 4242–4246.

External Links: [Link](https://doi.org/10.21437/Interspeech.2022-48),
[Document](https://dx.doi.org/10.21437/INTERSPEECH.2022-48)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
Y. Wang, H. Zhan, L. Liu, R. Zeng, H. Guo, J. Zheng, Q. Zhang, X. Zhang, S. Zhang, and Z. Wu (2024c)
Maskgct: zero-shot text-to-speech with masked generative codec transformer.

arXiv preprint arXiv:2409.00750.

Cited by: [Table 8](#S5.T8.5.5.1).

-
Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen (2024d)
MMLU-Pro: A more robust and challenging multi-task language understanding benchmark.

CoRR abs/2406.01574.

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
Z. Wang, M. Xia, L. He, H. Chen, Y. Liu, R. Zhu, K. Liang, X. Wu, H. Liu, S. Malladi, A. Chevalier, S. Arora, and D. Chen (2024e)
CharXiv: charting gaps in realistic chart understanding in multimodal llms.

arXiv preprint arXiv:2406.18521.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, et al. (2025a)
Qwen2. 5-omni technical report.

arXiv preprint arXiv:2503.20215.

Cited by: [§1](#S1.p1.1),
[§1](#S1.p3.1),
[§2.1](#S2.SS1.p1.1),
[Table 8](#S5.T8.5.13.1).

-
J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, Y. Lv, Y. Wang, D. Guo, H. Wang, L. Ma, P. Zhang, X. Zhang, H. Hao, Z. Guo, B. Yang, B. Zhang, Z. Ma, X. Wei, S. Bai, K. Chen, X. L. Liu, P. Wang, M. Yang, D. Liu, X. Ren, B. Zheng, R. Men, F. Zhou, B. Yu, J. Yang, L. Yu, J. Zhou, and J. Lin (2025b)
Qwen3-omni technical report.

ArXiv abs/2509.17765.

Cited by: [§1](#S1.p1.1),
[§1](#S1.p3.1),
[3rd item](#S2.I1.i3.p1.1),
[5th item](#S2.I1.i5.p1.1),
[§2.1](#S2.SS1.p1.1),
[§2.3](#S2.SS3.SSS0.Px2.p1.1),
[§2.4](#S2.SS4.p3.1),
[§3](#S3.p1.1),
[§3](#S3.p2.1),
[Table 8](#S5.T8.5.14.2).

-
F. Yan, H. Mao, C. C. Ji, T. Zhang, S. G. Patil, I. Stoica, and J. E. Gonzalez (2024)
Berkeley function calling leaderboard.

Note: [https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
R. Yan, X. Li, W. Chen, Z. Niu, C. Yang, Z. Ma, K. Yu, and X. Chen (2025)
URO-bench: a comprehensive benchmark for end-to-end spoken dialogue models.

arXiv preprint arXiv:2502.17810.

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025a)
Qwen3 technical report.

arXiv preprint arXiv:2505.09388.

Cited by: [§1](#S1.p1.1).

-
A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al. (2024a)
Qwen2 technical report.

arXiv:2407.10671.

Cited by: [§1](#S1.p1.1).

-
Y. Yang, J. Zhuang, G. Sun, C. Tang, Y. Li, P. Li, Y. Jiang, W. Li, Z. Ma, and C. Zhang (2025b)
Audio-centric video understanding benchmark without text shortcut.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,

pp. 6580–6598.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
Z. Yang, J. Tang, Z. Li, P. Wang, J. Wan, H. Zhong, X. Liu, M. Yang, P. Wang, S. Bai, L. Jin, and J. Lin (2024b)
CC-OCR: A comprehensive and challenging OCR benchmark for evaluating large multimodal models in literacy.

CoRR abs/2412.02210.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. (2023)
Mmmu: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.

arXiv:2311.16502.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, M. Yin, B. Yu, G. Zhang, et al. (2024)
MMMU-pro: a more robust multi-discipline multimodal understanding benchmark.

arXiv preprint arXiv:2409.02813.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Y. Zang, S. O’Brien, T. Berg-Kirkpatrick, J. McAuley, and Z. Novack (2025)
Are you really listening? boosting perceptual awareness in music-qa benchmarks.

arXiv preprint arXiv:2504.00369.

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng, D. Wu, and Z. Peng (2022)
WENETSPEECH: A 10000+ hours multi-domain mandarin corpus for speech recognition.

In IEEE International Conference on Acoustics, Speech and Signal Processing,
ICASSP 2022, Virtual and Singapore, 23-27 May 2022,

pp. 6182–6186.

External Links: [Link](https://doi.org/10.1109/ICASSP43922.2022.9746682),
[Document](https://dx.doi.org/10.1109/ICASSP43922.2022.9746682)

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
B. Zhang, C. Guo, G. Yang, H. Yu, H. Zhang, H. Lei, J. Mai, J. Yan, K. Yang, M. Yang, P. Huang, R. Jin, S. Jiang, W. Cheng, Y. Li, Y. Xiao, Y. Zhou, Y. Zhang, Y. Lu, and Y. He (2025a)
MiniMax-speech: intrinsic zero-shot text-to-speech with a learnable speaker encoder.

CoRR abs/2505.07916.

Cited by: [2nd item](#S5.I3.i2.p1.1),
[4th item](#S5.I3.i4.p1.1),
[Table 8](#S5.T8.5.11.1).

-
L. Zhang, J. Zhang, B. Lei, C. Wu, A. Liu, W. Jia, and X. Zhou (2025b)
WildSpeech-bench: benchmarking end-to-end speechllms in the wild.

External Links: 2506.21875

Cited by: [§5.1](#S5.SS1.SSS0.Px2.p1.1).

-
X. Zhang, C. Wu, Z. Zhao, W. Lin, Y. Zhang, Y. Wang, and W. Xie (2023)
PMC-VQA: visual instruction tuning for medical visual question answering.

CoRR abs/2305.10415.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
X. L. T. D. Zhang, G. Wang, J. Xue, K. Fang, L. Zhao, R. Ma, S. Ren, S. Liu, T. Guo, W. Zhuang, X. Zhang, X. Song, Y. Yan, Y. He, Cici, B. Shen, C. Zhu, C. Ma, C. Chen, H. Chen, J. Li, L. Li, M. Zhu, P. Li, Q. Wang, S. Deng, W. Xiong, W. Huang, W. Yang, Y. Jiang, Y. Yang, Y. Tian, Y. Ma, Y. Yu, Z. Zhang, Z. Yue, B. Xiao, B. Xia, B. Gao, B. Ye, C. Cai, C. Liu, C. He, C. Li, D. Zhu, D. Zhang, F. Shi, G. Wang, H. Zhang, H. Lv, H. Li, H. Tian, H. Qu, H. Xu, H. Zhang, H. Liu, J. Duo, J. Zuo, J. Wei, J. Xiao, J. Dong, J. Shi, J. Hu, K. Bao, K. Zhou, L. Zhang, M. Chen, N. Chen, P. Zhang, Q. Chen, Q. Wang, R. Li, S. Liu, S. Wang, S. Li, S. Yu, S. Cao, S. Chen, S. Gu, W. Wang, W. Ma, X. Deng, X. Yong, X. Zhang, X. Wang, Y. Song, Y. Zhao, Y. Zhao, Y. Gao, Y. Cheng, Y. Tu, Y. Wang, Z. Huang, Z. Tang, Z. Lin, Z. Song, Z. Xu, Z. Zheng, and Z. Jiang (2025c)
MiMo-audio: audio language models are few-shot learners.

ArXiv abs/2512.23808.

External Links: [Link](https://api.semanticscholar.org/CorpusID:284351195)

Cited by: [Table 8](#S5.T8.5.12.1).

-
Y. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang, Q. Wen, Z. Zhang, et al. (2024)
MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?.

arXiv preprint arXiv:2408.13257.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Y. Zhao, H. Zhang, L. Xie, T. Hu, G. Gan, Y. Long, Z. Hu, W. Chen, C. Li, Z. Xu, C. Wang, Z. Shangguan, Z. Liang, Y. Liu, C. Zhao, and A. Cohan (2025)
MMVU: measuring expert-level multi-discipline video understanding.

In IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2025, Nashville, TN, USA, June 11-15, 2025,

pp. 8475–8489.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. (2025)
Group sequence policy optimization.

arXiv preprint arXiv:2507.18071.

Cited by: [item (3)](#S4.I2.i3.p1.1).

-
J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou (2023)
Instruction-following evaluation for large language models.

CoRR abs/2311.07911.

Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1).

-
J. Zhou, Y. Shu, B. Zhao, B. Wu, Z. Liang, S. Xiao, M. Qin, X. Yang, Y. Xiong, B. Zhang, T. Huang, and Z. Liu (2025a)
MLVU: benchmarking multi-task long video understanding.

In IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2025, Nashville, TN, USA, June 11-15, 2025,

pp. 13691–13701.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Z. Zhou, R. Wang, and Z. Wu (2025b)
Daily-omni: towards audio-visual reasoning with temporal alignment across modalities.

CoRR abs/2505.17862.

Cited by: [§5.1](#S5.SS1.SSS0.Px4.p1.1).

-
D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny (2023)
Minigpt-4: enhancing vision-language understanding with advanced large language models.

arXiv:2304.10592.

Cited by: [§1](#S1.p1.1).

-
C. Zou, X. Guo, R. Yang, J. Zhang, B. Hu, and H. Zhang (2025)
DynaMath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models.

In ICLR,

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

-
Y. Zuo, S. Qu, Y. Li, Z. Chen, X. Zhu, E. Hua, K. Zhang, N. Ding, and B. Zhou (2025)
MedXpertQA: benchmarking expert-level medical reasoning and understanding.

In ICML, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu (Eds.),

Proceedings of Machine Learning Research.

Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1).

## 7 Authors

Core Contributors222Alphabetical order. * denotes the corresponding author.

Bing Han
Baosong Yang
Bin Zhang
Bo Zheng
Dayiheng Liu
Fan Zhou
Hongkun Hao
Hangrui Hu
Jin Xu∗
Jianxin Yang
Jingren Zhou
Keqin Chen
Le Yu
Mingkun Yang
Peng Wang
Pei Zhang
Qize Yang
Rui Men
Ruiyang Xu
Shuai Bai
Sibo Song
Ting He
Xize Cheng
Xuejing Liu
Xingzhang Ren
Xian Shi
Xiong Wang
Xinyu Zhang
Xinfa Zhu
Yunfei Chu
Yuanjun Lv
Yuchong Sun
Yongqi Wang
Yuxuan Wang
Yang Zhang
Zhifang Guo
Zishan Guo
Ziyang Ma

Contributors††footnotemark:

Andong Chen
Anfeng Li
An Yang
Bei Chen
Bin Lin
Bingshen Mu
Bohan Wang
Buxiao Wu
Bowen Xu
Beichen Zhang
Cheng Chen
Chang Gao
Chengen Huang
Chenyang Le
Chenhao Li
Chenglong Liu
Chenxu Lv
Chen Qiang
Chenfei Wu
Chenhan Yuan
Chengruidong Zhang
Chujie Zheng
Daren Chen
Dake Guo
Fei Huang
Gaoji Liu
Guangdong Zhou
Hao Ge
Huiqiang Jiang
Haoran Lian
Hongjian Tu
Hao Yu
Hang Zhang
Hao Zhou
Haiquan Zhao
Humen Zhong
Jiawei Chen
Jian Guan
Jiayi Leng
Jiahao Li
Junrong Lin
Jiawei Liu
Jialong Tang
Jun Tang
Jianhong Tu
Jianqiang Wan
Jinxi Wei
Jianwei Zhang
Jing Zhou
Kai Dang
Kangxiang Xia
Kun Yan
Kexin Yang
Lianghao Deng
Lulu Hu
Linhan Ma
Lingchen Meng
Lei Xie
Laiwen Zheng
Miao Hong
Mei Li
Mingcheng Li
Mingze Li
Minsheng Li
Minghao Wu
Mingfeng Xue
Na Ni
Peng Liu
Peng Wang
Pengfei Wang
Peiyang Zhang
Qidong Huang
Qingfeng Lan
Qintong Li
Que Shen
Qiuyue Wang
Qin Zhu
Ruisheng Cao
Rongyao Fang
Rui Hu
Ruibin Yuan
Song Chen
Su Hao
Shen Li
Shixuan Liu
Shurui Li
Siqi Zhang
Tianyi Tang
Tingyu Xia
Wei Ding
Wenbin Ge
Weizhou Shen
Wei Wang
Wentao Yao
Xi Chen
Xiaotong Chen
Xionghui Chen
Xiaodong Deng
Xudong Guo
Xin Le
Xiao Li
Xie Chen
Xinyao Niu
Xuancheng Ren
Xuechun Wang
Xuwu Wang
Xingzhe Wu
Xipin Wei
Xiao Xu
Xian Yang
Yuxuan Cai
Yizhong Cao
Yilei Chen
Yuxiang Chen
Yiming Dong
Yang Fan
Yanpeng Li
Yucheng Li
Yang Liu
Yantao Liu
Yuqiong Liu
Yuxuan Liu
Yuyan Luo
Yubo Ma
Yang Su
Yuezhang Wang
Yuhao Wang
Yi Wu
Yunbao Wu
Yu Xi
Yi Zhang
Yichang Zhang
Yinger Zhang
Yuxiang Zheng
Zeyu Cui
Ziwei Ji
Ziyue Jiang
Zhaohai Li
Zheng Li
Zhi Li
Zihan Qiu
Zekun Wang
Zhihai Wang
Zhenghao Xing
Zhibo Yang
Zhuorui Ye
Zhenru Zhang
Zhipeng Zhou
Zhengyang Zhuge

## 8 Appendix

### 8.1 Detailed Multilingual Evaluation Results

##### Multilingual ASR.

As presented in Table [13](#S8.T13), Qwen3.5-Omni demonstrates superior speech recognition capabilities compared to state-of-the-art competitors on the FLEURS test set. Qwen3.5-Omni-Plus achieves the lowest average WER of 6.6%, outperforming both Gemini-3.1-Pro (7.3%) and GPT-4o-Transcribe (10.4%). It secures the best performance in the majority of languages, with particularly significant margins in complex tonal and low-resource languages such as Cantonese (2.2% vs. 6.3% for Gemini-3.1-Pro), Thai, and Vietnamese. Meanwhile, Qwen3.5-Omni-Flash offers a highly efficient alternative, achieving an average WER of 10.8% that remains competitive against Gemini-3-Flash (10.5%). Notably, Qwen3.5-Omni-Flash exhibits exceptional robustness in challenging scenarios, drastically reducing errors in Cantonese (3.1% vs. 10.8% for Gemini-3-Flash) and maintaining strong performance in Japanese and Korean, thereby highlighting its advantage for high-value Asian language pairs.

##### Multilingual Translation.

As shown in Tables [14](#S8.T14) and [15](#S8.T15), the Qwen3.5-Omni series demonstrates distinct advantages over state-of-the-art competitors on the FLEURS test set, particularly in Asian languages and specific high-resource pairs. Qwen3.5-Omni-Plus exhibits comprehensive superiority over Gemini-3.1-Pro in the many-to-many directions (en2xx/zh2xx), achieving higher average BLEU scores in both English-to-XX (33.8 vs. 31.8) and Chinese-to-XX (21.4 vs. 19.6). It also leads in key xx2en pairs such as Portuguese (49.4 vs. 47.7) and Indonesian (45.7 vs. 45.1). Although Gemini-3.1-Pro holds a slight edge in overall xx2zh averages, Qwen3.5-Omni-Plus significantly outperforms it in critical Asian languages, including Cantonese (+15.6 BLEU), Korean, and Japanese. Similarly, Qwen3.5-Omni-Flash shows targeted strengths against Gemini-3-Flash. While maintaining competitive general performance, it vastly surpasses Gemini in Cantonese translation across all directions (e.g., 37.5 vs. 22.4 in xx2zh and 37.3 vs. 26.7 in en2xx) and delivers better results in Japanese and Korean xx2zh tasks. These results underscore Qwen3.5-Omni’s robust optimization for complex Asian linguistic structures and key regional languages.

*Table 13: Multilingual ASR performance on the FLEURS test set. Results are reported using Word Error Rate (WER, ↓), where lower values indicate better performance. For italicized languages, Character Error Rate (CER, ↓) is reported instead. Compared with competing models, Qwen3.5-Omni-Plus achieves the best results on the majority of languages. The best results are highlighted in bold.*

*Table 14: Multilingual translation performance on the FLEURS en2xx and zh2xx test sets. Results are reported using BLEU (↑). Compared with competing models, Qwen3.5-Omni-Plus outperforms them on the majority of language pairs. The best results are highlighted in bold.*

*Table 15: Multilingual translation performance on the FLEURS xx2en and xx2zh test sets. Results are reported using BLEU (↑). The best results are highlighted in bold.*



Experimental support, please
[view the build logs](./2604.15804v2/__stdout.txt)
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