[\reportnumber 001 # DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model DeepSeek-AI research@deepseek.com ###### Abstract We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2).

{CJK*}

UTF8gbsn

![Figure](x2.png)

Figure 1:
(a) MMLU accuracy vs. activated parameters, among different open-source models.
(b) Training costs and inference efficiency of DeepSeek 67B (Dense) and DeepSeek-V2.

## 1 Introduction

In the past few years, Large Language Models (LLMs) [(OpenAI, [2022](#bib.bib37), [2023](#bib.bib38); Anthropic, [2023](#bib.bib3); Google, [2023](#bib.bib18))] have undergone rapid development, offering a glimpse into the dawn of Artificial General Intelligence (AGI).
In general, the intelligence of an LLM tends to improve as the number of parameters increases, allowing it to exhibit emergent capabilities across various tasks [(Wei et al., [2022](#bib.bib52))].
However, the improvement comes at the cost of larger computing resources for training and a potential decrease in inference throughput.
These constraints present significant challenges that impede the widespread adoption and utilization of LLMs.
In order to tackle this problem, we introduce DeepSeek-V2, a strong open-source Mixture-of-Experts (MoE) language model, characterized by economical training and efficient inference through an innovative Transformer architecture.
It is equipped with a total of 236B parameters, of which 21B are activated for each token, and supports a context length of 128K tokens.

We optimize the attention modules and Feed-Forward Networks (FFNs) within the Transformer framework [(Vaswani et al., [2017](#bib.bib51))] with our proposed Multi-head Latent Attention (MLA) and DeepSeekMoE.
(1)
In the context of attention mechanisms, the Key-Value (KV) cache of the Multi-Head Attention (MHA) [(Vaswani et al., [2017](#bib.bib51))] poses a significant obstacle to the inference efficiency of LLMs.
Various approaches have been explored to address this issue, including Grouped-Query Attention (GQA) [(Ainslie et al., [2023](#bib.bib2))] and Multi-Query Attention (MQA) [(Shazeer, [2019](#bib.bib46))].
However, these methods often compromise performance in their attempt to reduce the KV cache.
In order to achieve the best of both worlds, we introduce MLA, an attention mechanism equipped with low-rank key-value joint compression.
Empirically, MLA achieves superior performance compared with MHA, and meanwhile significantly reduces the KV cache during inference, thus boosting the inference efficiency.
(2)
For Feed-Forward Networks (FFNs), we follow the DeepSeekMoE architecture [(Dai et al., [2024](#bib.bib11))], which adopts fine-grained expert segmentation and shared expert isolation for higher potential in expert specialization.
The DeepSeekMoE architecture demonstrates great advantages compared with conventional MoE architectures like GShard [(Lepikhin et al., [2021](#bib.bib31))], enabling us to train strong models at an economical cost.
As we employ expert parallelism during training, we also devise supplementary mechanisms to control communication overheads and ensure load balance.
By combining these two techniques, DeepSeek-V2 features strong performance (Figure [1](#S0.F1)), economical training costs, and efficient inference throughput (Figure [1](#S0.F1)), simultaneously.

![Figure](x3.png)

*Figure 2: Illustration of the architecture of DeepSeek-V2. MLA ensures efficient inference by significantly reducing the KV cache for generation, and DeepSeekMoE enables training strong models at an economical cost through the sparse architecture.*

We construct a high-quality and multi-source pre-training corpus consisting of 8.1T tokens.
Compared with the corpus used in DeepSeek 67B (our previous release) [(DeepSeek-AI, [2024](#bib.bib13))], this corpus features an extended amount of data, especially Chinese data, and higher data quality.
We first pretrain DeepSeek-V2 on the full pre-training corpus.
Then, we collect 1.5M conversational sessions, which encompass various domains such as math, code, writing, reasoning, safety, and more, to perform Supervised Fine-Tuning (SFT) for DeepSeek-V2 Chat (SFT).
Finally, we follow DeepSeekMath [(Shao et al., [2024](#bib.bib45))] to employ Group Relative Policy Optimization (GRPO) to further align the model with human preference and produce DeepSeek-V2 Chat (RL).

We evaluate DeepSeek-V2 on a wide range of benchmarks in English and Chinese, and compare it with representative open-source models.
Evaluation results show that even with only 21B activated parameters, DeepSeek-V2 still achieves top-tier performance among open-source models and becomes the strongest open-source MoE language model.
Figure [1](#S0.F1) highlights that, on MMLU, DeepSeek-V2 achieves top-ranking performance with only a small number of activated parameters.
In addition, as shown in Figure [1](#S0.F1), compared with DeepSeek 67B, DeepSeek-V2 saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times.
We also evaluate DeepSeek-V2 Chat (SFT) and DeepSeek-V2 Chat (RL) on open-ended benchmarks.
Notably, DeepSeek-V2 Chat (RL) achieves 38.9 length-controlled win rate on AlpacaEval 2.0 [(Dubois et al., [2024](#bib.bib15))], 8.97 overall score on MT-Bench [(Zheng et al., [2023](#bib.bib59))], and 7.91 overall score on AlignBench [(Liu et al., [2023](#bib.bib34))].
The English open-ended conversation evaluations demonstrate that DeepSeek-V2 Chat (RL) has top-tier performance among open-source chat models.
In addition, the evaluation on AlignBench indicates that in Chinese, DeepSeek-V2 Chat (RL) outperforms all of open-source models, and even beats most of closed-source models.

In order to facilitate further research and development on MLA and DeepSeekMoE, we also release DeepSeek-V2-Lite, a smaller model equipped with MLA and DeepSeekMoE, for the open-source community.
It has a total of 15.7B parameters, where 2.4B are activated for each token.
Detailed descriptions about DeepSeek-V2-Lite can be found in Appendix [B](#A2).

In the rest of this paper, we first provide a detailed description of the model architecture of DeepSeek-V2 (Section [2](#S2)).
Subsequently, we introduce our pre-training endeavors, including the training data construction, hyper-parameter settings, infrastructures, long context extension, and the evaluation of model performance and efficiency (Section [3](#S3)).
Following this, we demonstrate our efforts in alignment, encompassing Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), the evaluation results, and other discussion (Section [4](#S4)).
Finally, we summarize the conclusion, deliberate on the current limitations of DeepSeek-V2, and outline our future work (Section [5](#S5)).

## 2 Architecture

By and large, DeepSeek-V2 is still in the Transformer architecture [(Vaswani et al., [2017](#bib.bib51))], where each Transformer block consists of an attention module and a Feed-Forward Network (FFN).
However, for both the attention module and the FFN, we design and employ innovative architectures.
For attention, we design MLA, which utilizes low-rank key-value joint compression to eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference.
For FFNs, we adopt the DeepSeekMoE architecture [(Dai et al., [2024](#bib.bib11))], a high-performance MoE architecture that enables training strong models at an economical cost.
An illustration of the architecture of DeepSeek-V2 is presented in Figure [2](#S1.F2), and we will introduce the details of MLA and DeepSeekMoE in this section.
For other tiny details (e.g., layer normalization and the activation function in FFNs), unless specifically stated, DeepSeek-V2 follows the settings of DeepSeek 67B [(DeepSeek-AI, [2024](#bib.bib13))].

### 2.1 Multi-Head Latent Attention: Boosting Inference Efficiency

Conventional Transformer models usually adopts Multi-Head Attention (MHA) [(Vaswani et al., [2017](#bib.bib51))], but during generation, its heavy Key-Value (KV) cache will become the bottleneck that limit the inference efficiency.
In order to reduce the KV cache, Multi-Query Attention (MQA) [(Shazeer, [2019](#bib.bib46))] and Grouped-Query Attention (GQA) [(Ainslie et al., [2023](#bib.bib2))] are proposed.
They require a smaller magnitude of KV cache, but their performance does not match MHA (we provide the ablation of MHA, GQA and MQA in Appendix [D.1](#A4.SS1)).

For DeepSeek-V2, we design an innovative attention mechanism called Multi-head Latent Attention (MLA).
Equipped with low-rank key-value joint compression, MLA achieves better performance than MHA, but requires a significantly smaller amount of KV cache.
We introduce its architecture in the following, and also provide a comparison between MLA and MHA in Appendix [D.2](#A4.SS2).

#### 2.1.1 Preliminaries: Standard Multi-Head Attention

We first introduce the standard MHA mechanism as background.
Let $d$ be the embedding dimension, $n_{h}$ be the number of attention heads, $d_{h}$ be the dimension per head, and $\mathbf{h}_{t}\in\mathbb{R}^{d}$ be the attention input of the $t$-th token at an attention layer.
Standard MHA first produces $\mathbf{q}_{t},\mathbf{k}_{t},\mathbf{v}_{t}\in\mathbb{R}^{d_{h}n_{h}}$ through three matrices $W^{Q},W^{K},W^{V}\in\mathbb{R}^{d_{h}n_{h}\times d}$, respectively:

$$
\begin{aligned}
\mathbf{q}_{t} &=W^{Q}\mathbf{h}_{t}, \tag{1}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{t} &=W^{K}\mathbf{h}_{t}, \tag{2}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{v}_{t} &=W^{V}\mathbf{h}_{t}, \tag{3}
\end{aligned}
$$

Then, $\mathbf{q}_{t},\mathbf{k}_{t},\mathbf{v}_{t}$ will be sliced into $n_{h}$ heads for the multi-head attention computation:

$$
\begin{aligned}
[\mathbf{q}_{t,1}; &\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}, \tag{4}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{k}_{t,1}; &\mathbf{k}_{t,2};...;\mathbf{k}_{t,n_{h}}]=\mathbf{k}_{t}, \tag{5}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{v}_{t,1}; &\mathbf{v}_{t,2};...;\mathbf{v}_{t,n_{h}}]=\mathbf{v}_{t}, \tag{6}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{o}_{t,i} &=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}}})\mathbf{v}_{j,i}, \tag{7}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{u}_{t} &=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}], \tag{8}
\end{aligned}
$$

where $\mathbf{q}_{t,i},\mathbf{k}_{t,i},\mathbf{v}_{t,i}\in\mathbb{R}^{d_{h}}$ denote the query, key, and value of the $i$-th attention head, respectively;
$W^{O}\in\mathbb{R}^{d\times d_{h}n_{h}}$ denotes the output projection matrix.
During inference, all keys and values need to be cached to accelerate inference, so MHA needs to cache $2n_{h}d_{h}l$ elements for each token.
In model deployment, this heavy KV cache is a large bottleneck that limits the maximum batch size and sequence length.

![Figure](x4.png)

*Figure 3: Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference.*

#### 2.1.2 Low-Rank Key-Value Joint Compression

The core of MLA is the low-rank joint compression for keys and values to reduce KV cache:

$$
\begin{aligned}
\mathbf{c}_{t}^{KV} &=W^{DKV}\mathbf{h}_{t}, \tag{9}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{t}^{C} &=W^{UK}\mathbf{c}_{t}^{KV}, \tag{10}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{v}_{t}^{C} &=W^{UV}\mathbf{c}_{t}^{KV}, \tag{11}
\end{aligned}
$$

where $\mathbf{c}_{t}^{KV}\in\mathbb{R}^{d_{c}}$ is the compressed latent vector for keys and values;
$d_{c}(\ll d_{h}n_{h})$ denotes the KV compression dimension;
$W^{DKV}\in\mathbb{R}^{d_{c}\times d}$ is the down-projection matrix;
and $W^{UK},W^{UV}\in\mathbb{R}^{d_{h}n_{h}\times d_{c}}$ are the up-projection matrices for keys and values, respectively.
During inference, MLA only needs to cache $\mathbf{c}_{t}^{KV}$, so its KV cache has only $d_{c}l$ elements, where $l$ denotes the number of layers.
In addition, during inference, since $W^{UK}$ can be absorbed into $W^{Q}$, and $W^{UV}$ can be absorbed into $W^{O}$, we even do not need to compute keys and values out for attention.
Figure [3](#S2.F3) intuitively illustrates how the KV joint compression in MLA reduces the KV cache.

Moreover, in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache:

$$
\begin{aligned}
\mathbf{c}_{t}^{Q} &=W^{DQ}\mathbf{h}_{t}, \tag{12}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{q}_{t}^{C} &=W^{UQ}\mathbf{c}_{t}^{Q}, \tag{13}
\end{aligned}
$$

where $\mathbf{c}_{t}^{Q}\in\mathbb{R}^{d_{c}^{\prime}}$ is the compressed latent vector for queries;
$d_{c}^{\prime}(\ll d_{h}n_{h})$ denotes the query compression dimension;
and $W^{DQ}\in\mathbb{R}^{d_{c}^{\prime}\times d},W^{UQ}\in\mathbb{R}^{d_{h}n_{h}%
\times d_{c}^{\prime}}$ are the down-projection and up-projection matrices for queries, respectively.

#### 2.1.3 Decoupled Rotary Position Embedding

Following DeepSeek 67B [(DeepSeek-AI, [2024](#bib.bib13))], we intend to use the Rotary Position Embedding (RoPE) [(Su et al., [2024](#bib.bib48))] for DeepSeek-V2.
However, RoPE is incompatible with low-rank KV compression.
To be specific, RoPE is position-sensitive for both keys and queries.
If we apply RoPE for the keys $\mathbf{k}_{t}^{C}$, $W^{UK}$ in Equation [10](#S2.E10) will be coupled with a position-sensitive RoPE matrix.
In this way, $W^{UK}$ cannot be absorbed into $W^{Q}$ any more during inference, since a RoPE matrix related to the currently generating token will lie between $W^{Q}$ and $W^{UK}$ and matrix multiplication does not obey a commutative law.
As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency.

As a solution, we propose the decoupled RoPE strategy that uses additional multi-head queries $\mathbf{q}_{t,i}^{R}\in\mathbb{R}^{d_{h}^{R}}$ and a shared key $\mathbf{k}_{t}^{R}\in\mathbb{R}^{d_{h}^{R}}$ to carry RoPE, where $d_{h}^{R}$ denotes the per-head dimension of the decoupled queries and key.
Equipped with the decoupled RoPE strategy, MLA performs the following computation:

$$
\begin{aligned}
[\mathbf{q}_{t,1}^{R};\mathbf{q}_{t,2}^{R};...;\mathbf{q}_{t,n_{h% }}^{R}]=\mathbf{q}_{t}^{R} &=\operatorname{RoPE}({W^{QR}}\mathbf{c}_{t}^{Q}), \tag{14}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{t}^{R} &=\operatorname{RoPE}({W^{KR}}\mathbf{h}_{t}), \tag{15}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{q}_{t,i} &=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}], \tag{16}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{t,i} &=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}], \tag{17}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{o}_{t,i} &=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}})\mathbf{v}_{j,i}^{C}, \tag{18}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{u}_{t} &=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}], \tag{19}
\end{aligned}
$$

where $W^{QR}\in\mathbb{R}^{d_{h}^{R}n_{h}\times d_{c}^{\prime}}$ and $W^{KR}\in\mathbb{R}^{d_{h}^{R}\times d}$ are matrices to produce the decouples queries and key, respectively;
$\operatorname{RoPE}(\cdot)$ denotes the operation that applies RoPE matrices;
and $[\cdot;\cdot]$ denotes the concatenation operation.
During inference, the decoupled key should also be cached.
Therefore, DeepSeek-V2 requires a total KV cache containing $(d_{c}+d_{h}^{R})l$ elements.

In order to demonstrate the complete computation process of MLA, we also organize and provide its full formulas in Appendix [C](#A3).

*Table 1: Comparison of the KV cache per token among different attention mechanisms. $n_{h}$ denotes the number of attention heads, $d_{h}$ denotes the dimension per attention head, $l$ denotes the number of layers, $n_{g}$ denotes the number of groups in GQA, and $d_{c}$ and $d_{h}^{R}$ denote the KV compression dimension and the per-head dimension of the decoupled queries and key in MLA, respectively. The amount of KV cache is measured by the number of elements, regardless of the storage precision. For DeepSeek-V2, $d_{c}$ is set to $4d_{h}$ and $d_{h}^{R}$ is set to $\frac{d_{h}}{2}$. So, its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA.*

#### 2.1.4 Comparison of Key-Value Cache

We demonstrate a comparison of the KV cache per token among different attention mechanisms in Table [1](#S2.T1).
MLA requires only a small amount of KV cache, equal to GQA with only 2.25 groups, but can achieve stronger performance than MHA.

### 2.2 DeepSeekMoE: Training Strong Models at Economical Costs

#### 2.2.1 Basic Architecture

For FFNs, we employ the DeepSeekMoE architecture [(Dai et al., [2024](#bib.bib11))].
DeepSeekMoE has two key ideas: segmenting experts into finer granularity for higher expert specialization and more accurate knowledge acquisition, and isolating some shared experts for mitigating knowledge redundancy among routed experts.
With the same number of activated and total expert parameters, DeepSeekMoE can outperform conventional MoE architectures like GShard [(Lepikhin et al., [2021](#bib.bib31))] by a large margin.

Let $\mathbf{u}_{t}$ be the FFN input of the $t$-th token, we compute the FFN output $\mathbf{h}_{t}^{\prime}$ as follows:

$$
\begin{aligned}
\mathbf{h}_{t}^{\prime} &=\mathbf{u}_{t}+\sum_{i=1}^{N_{s}}{\operatorname{FFN}^{(s)}_{i}\left(\mathbf{u}_{t}\right)}+\sum_{i=1}^{N_{r}}{g_{i,t}\operatorname{FFN}^{(r)}_{i}\left(\mathbf{u}_{t}\right)}, \tag{20}
\end{aligned}
$$
$$
\begin{aligned}
g_{i,t} &=\begin{cases}s_{i,t},&s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_{r}\},K_{r}),\\ 0,&\text{otherwise},\end{cases} \tag{21}
\end{aligned}
$$
$$
\begin{aligned}
s_{i,t} &=\operatorname{Softmax}_{i}\left({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}\right), \tag{22}
\end{aligned}
$$

where $N_{s}$ and $N_{r}$ denote the numbers of shared experts and routed experts, respectively;
$\operatorname{FFN}^{(s)}_{i}(\cdot)$ and $\operatorname{FFN}^{(r)}_{i}(\cdot)$ denote the $i$-th shared expert and the $i$-th routed expert, respectively;
$K_{r}$ denotes the number of activated routed experts;
$g_{i,t}$ is the gate value for the $i$-th expert;
$s_{i,t}$ is the token-to-expert affinity;
$\mathbf{e}_{i}$ is the centroid of the $i$-th routed expert in this layer;
and $\operatorname{Topk}(\cdot,K)$ denotes the set comprising $K$ highest scores among the affinity scores calculated for the $t$-th token and all routed experts.

#### 2.2.2 Device-Limited Routing

We design a device-limited routing mechanism to bound MoE-related communication costs.
When expert parallelism is employed, the routed experts will be distributed across multiple devices.
For each token, its MoE-related communication frequency is proportional to the number of devices covered by its target experts.
Due to the fine-grained expert segmentation in DeepSeekMoE, the number of activated experts can be large, so the MoE-related communication will be more costly if we apply expert parallelism.

For DeepSeek-V2, beyond the naive top-K selection of routed experts, we additionally ensure that the target experts of each token will be distributed on at most $M$ devices.
To be specific, for each token, we first select $M$ devices that have experts with the highest affinity scores in them.
Then, we perform top-K selection among experts on these $M$ devices.
In practice, we find that when $M\geqslant 3$, the device-limited routing can achieve a good performance roughly aligned with the unrestricted top-K routing.

#### 2.2.3 Auxiliary Loss for Load Balance

We take the load balance into consideration for automatically learned routing strategies.
Firstly, unbalanced load will raise the risk of routing collapse [(Shazeer et al., [2017](#bib.bib47))], preventing some experts being fully trained and utilized.
Secondly, when expert parallelism is employed, unbalanced load will diminish computation efficiency.
During the training of DeepSeek-V2, we design three kinds of auxiliary losses, for controlling expert-level load balance ($\mathcal{L}_{\mathrm{ExpBal}}$), device-level load balance ($\mathcal{L}_{\mathrm{DevBal}}$), and communication balance ($\mathcal{L}_{\mathrm{CommBal}}$), respectively.

##### Expert-Level Balance Loss.

We use an expert-level balance loss [(Fedus et al., [2021](#bib.bib16); Lepikhin et al., [2021](#bib.bib31))] to mitigate the risk of routing collapse:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ExpBal}} &=\alpha_{1}\sum_{i=1}^{N_{r}}{f_{i}P_{i}}, \tag{23}
\end{aligned}
$$
| | $\displaystyle f_{i}$ | $\displaystyle=\frac{N_{r}}{K_{r}T}\sum_{t=1}^{T}{\mathds{1}(\text{Token $t$ % selects Expert $i$})},$ | | (24) |
$$
\begin{aligned}
P_{i} &=\frac{1}{T}\sum_{t=1}^{T}{s_{i,t}}, \tag{25}
\end{aligned}
$$

where $\alpha_{1}$ is a hyper-parameter called expert-level balance factor;
$\mathds{1}(\cdot)$ denotes the indicator function;
and $T$ denotes the number of tokens in a sequence.

##### Device-Level Balance Loss.

In addition to the expert-level balance loss, we additionally design a device-level balance loss to ensure balanced computation across different devices.
In the training process of DeepSeek-V2, we partition all routed experts into $D$ groups $\{\mathcal{E}_{1},\mathcal{E}_{2},...,\mathcal{E}_{D}\}$, and deploy each group on a single device.
The device-level balance loss is computed as follows:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DevBal}} &=\alpha_{2}\sum_{i=1}^{D}{f_{i}^{\prime}P_{i}^{\prime}}, \tag{26}
\end{aligned}
$$
$$
\begin{aligned}
f_{i}^{\prime} &=\frac{1}{|\mathcal{E}_{i}|}\sum_{j\in\mathcal{E}_{i}}{f_{j}}, \tag{27}
\end{aligned}
$$
$$
\begin{aligned}
P_{i}^{\prime} &=\sum_{j\in\mathcal{E}_{i}}{P_{j}}, \tag{28}
\end{aligned}
$$

where $\alpha_{2}$ is a hyper-parameter called device-level balance factor.

##### Communication Balance Loss.

Finally, we introduce a communication balance loss to ensure that the communication of each device is balanced.
Although the device-limited routing mechanism guarantees that the sending communication of each device is bounded, if a certain device receives more tokens than other devices, the practical communication efficiency will also be affected.
In order to mitigate this issue, we design a communication balance loss as follows:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CommBal}} &=\alpha_{3}\sum_{i=1}^{D}{f_{i}^{\prime\prime}P_{i}^{\prime\prime}}, \tag{29}
\end{aligned}
$$
| | $\displaystyle f_{i}^{\prime\prime}$ | $\displaystyle=\frac{D}{MT}\sum_{t=1}^{T}{\mathds{1}(\text{Token $t$ is sent to% Device $i$})},$ | | (30) |
$$
\begin{aligned}
P_{i}^{\prime\prime} &=\sum_{j\in\mathcal{E}_{i}}{P_{j}}, \tag{31}
\end{aligned}
$$

where $\alpha_{3}$ is a hyper-parameter called communication balance factor.
The device-limited routing mechanism operates on the principle of ensuring that each device transmits at most $MT$ hidden states to other devices.
Simultaneously, the communication balance loss is employed to encourage each device to receive around $MT$ hidden states from other devices.
The communication balance loss guarantees a balanced exchange of information among devices, promoting efficient communications.

#### 2.2.4 Token-Dropping Strategy

While balance losses aim to encourage a balanced load, it is important to acknowledge that they cannot guarantee a strict load balance.
In order to further mitigate the computation wastage caused by unbalanced load, we introduce a device-level token-dropping strategy during training.
This approach first computes the average computational budget for each device, which means that the capacity factor for each device is equivalent to 1.0.
Then, inspired by [Riquelme et al. ([2021](#bib.bib43))], we drop tokens with the lowest affinity scores on each device until reaching the computational budget.
In addition, we ensure that the tokens belonging to approximately 10% of the training sequences will never be dropped.
In this way, we can flexibly decide whether to drop tokens during inference according to the efficiency requirements, and always ensure consistency between training and inference.

## 3 Pre-Training

### 3.1 Experimental Setups

#### 3.1.1 Data Construction

While maintaining the same data processing stages as for DeepSeek 67B [(DeepSeek-AI, [2024](#bib.bib13))], we extend the amount of data and elevate the data quality.
In order to enlarge our pre-training corpus, we explore the potential of the internet data and optimize our cleaning processes, thus recovering a large amount of mistakenly deleted data.
Moreover, we incorporate more Chinese data, aiming to better leverage the corpus available on the Chinese internet.
In addition to the amount of data, we also focus on the data quality.
We enrich our pre-training corpus with high-quality data from various sources, and meanwhile improve the quality-based filtering algorithm.
The improved algorithm ensures that a large amount of non-beneficial data will be removed, while the valuable data will be mostly retained.
In addition, we filter out the contentious content from our pre-training corpus to mitigate the data bias introduced from specific regional cultures.
A detailed discussion about the influence of this filtering strategy is presented in Appendix [E](#A5).

We adopt the same tokenizer as used in DeepSeek 67B, which is built based on the Byte-level Byte-Pair Encoding (BBPE) algorithm and has a vocabulary size of 100K.
Our tokenized pre-training corpus contains 8.1T tokens, where Chinese tokens are approximately 12% more than English ones.

#### 3.1.2 Hyper-Parameters

##### Model Hyper-Parameters.

We set the number of Transformer layers to 60 and the hidden dimension to 5120.
All learnable parameters are randomly initialized with a standard deviation of 0.006.
In MLA, we set the number of attention heads $n_{h}$ to 128 and the per-head dimension $d_{h}$ to 128.
The KV compression dimension $d_{c}$ is set to 512, and the query compression dimension $d_{c}^{\prime}$ is set to 1536.
For the decoupled queries and key, we set the per-head dimension $d_{h}^{R}$ to 64.
Following [Dai et al. ([2024](#bib.bib11))], we substitute all FFNs except for the first layer with MoE layers.
Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536.
Among the routed experts, 6 experts will be activated for each token.
In addition, the low-rank compression and fine-grained expert segmentation will impact the output scale of a layer.
Therefore, in practice, we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training.
Under this configuration, DeepSeek-V2 comprises 236B total parameters, of which 21B are activated for each token.

##### Training Hyper-Parameters.

We employ the AdamW optimizer [(Loshchilov and Hutter, [2017](#bib.bib35))] with hyper-parameters set to $\beta_{1}=0.9$, $\beta_{2}=0.95$, and $\mathrm{weight\_decay}=0.1$.
The learning rate is scheduled using a warmup-and-step-decay strategy [(DeepSeek-AI, [2024](#bib.bib13))].
Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps.
Subsequently, the learning rate is multiplied by 0.316 after
training about 60% of tokens, and again by 0.316 after training about 90% of tokens.
The maximum learning rate is set to $2.4\times 10^{-4}$, and the gradient clipping norm is set to 1.0.
We also use a batch size scheduling strategy, where the batch size is gradually increased from 2304 to 9216 in the training of the first 225B tokens, and then keeps 9216 in the remaining training.
We set the maximum sequence length to 4K, and train DeepSeek-V2 on 8.1T tokens.
We leverage pipeline parallelism to deploy different layers of a model on different devices, and for each layer, the routed experts will be uniformly deployed on 8 devices ($D=8$).
As for the device-limited routing, each token will be sent to at most 3 devices ($M=3$).
As for balance losses, we set $\alpha_{1}$ to 0.003, $\alpha_{2}$ to 0.05, and $\alpha_{3}$ to 0.02.
We employ the token-dropping strategy during training for acceleration, but do not drop any tokens for evaluation.

#### 3.1.3 Infrastructures

DeepSeek-V2 is trained based on the HAI-LLM framework [(High-flyer, [2023](#bib.bib22))], an efficient and light-weight training framework developed internally by our engineers.
It employs a 16-way zero-bubble pipeline parallelism [(Qi et al., [2023](#bib.bib41))], an 8-way expert parallelism [(Lepikhin et al., [2021](#bib.bib31))], and ZeRO-1 data parallelism [(Rajbhandari et al., [2020](#bib.bib42))].
Given that DeepSeek-V2 has relatively few activated parameters, and a portion of the operators are recomputed to save activation memory, it can be trained without the necessity of tensor parallelism, thereby decreasing the communication overhead.
Moreover, in order to further improve the training efficiency, we overlap the computation of shared experts with the expert parallel all-to-all communication.
We also customize faster CUDA kernels for communications, routing algorithms, and fused linear computations across different experts.
In addition, MLA is also optimized based on an improved version of FlashAttention-2 [(Dao, [2023](#bib.bib12))].

We conduct all experiments on a cluster equipped with NVIDIA H800 GPUs.
Each node in the H800 cluster contains 8 GPUs connected using NVLink and NVSwitch within nodes.
Across nodes, InfiniBand interconnects are utilized to facilitate communications.

![Figure](x5.png)

*Figure 4: Evaluation results on the “Needle In A Haystack” (NIAH) tests. DeepSeek-V2 performs well across all context window lengths up to 128K.*

#### 3.1.4 Long Context Extension

After the initial pre-training of DeepSeek-V2, we employ YaRN [(Peng et al., [2023](#bib.bib40))] to extend the default context window length from 4K to 128K.
YaRN was specifically applied to the decoupled shared key $\mathbf{k}^{R}_{t}$ as it is responsible for carrying RoPE [(Su et al., [2024](#bib.bib48))].
For YaRN, we set the scale $s$ to 40, $\alpha$ to 1, $\beta$ to 32, and the target maximum context length to 160K.
Under these settings, we can expect the model to respond well for a context length of 128K.
Slightly diverging from original YaRN, due to our distinct attention mechanism, we adjust the length scaling factor to modulate the attention entropy.
The factor $\sqrt{t}$ is computed as $\sqrt{t}=0.0707\ln{s}+1$, aiming at minimizing the perplexity.

We additionally train the model for 1000 steps, with a sequence length of 32K and a batch size of 576 sequences.
Although the training is conducted solely at the sequence length of 32K, the model still demonstrates robust performance when being evaluated at a context length of 128K.
As shown in Figure [4](#S3.F4), the results on the “Needle In A Haystack” (NIAH) tests indicate that DeepSeek-V2 performs well across all context window lengths up to 128K.

### 3.2 Evaluations

#### 3.2.1 Evaluation Benchmarks

DeepSeek-V2 is pretrained on a bilingual corpus, so we evaluate it on a series of benchmarks in English and Chinese.
Our evaluation is based on our internal evaluation framework integrated in our HAI-LLM framework.
Included benchmarks are categorized and listed as follows, where underlined benchmarks are in Chinese:

Multi-subject multiple-choice datasets include MMLU [(Hendrycks et al., [2020](#bib.bib20))], C-Eval [(Huang et al., [2023](#bib.bib25))], and CMMLU [(Li et al., [2023](#bib.bib32))].

Language understanding and reasoning datasets include HellaSwag [(Zellers et al., [2019](#bib.bib56))], PIQA [(Bisk et al., [2020](#bib.bib6))], ARC [(Clark et al., [2018](#bib.bib8))], and BigBench Hard (BBH) [(Suzgun et al., [2022](#bib.bib50))].

Closed-book question answering datasets include TriviaQA [(Joshi et al., [2017](#bib.bib27))] and NaturalQuestions [(Kwiatkowski et al., [2019](#bib.bib28))].

Reading comprehension datasets include RACE [Lai et al. ([2017](#bib.bib30))], DROP [(Dua et al., [2019](#bib.bib14))], C3 [(Sun et al., [2019](#bib.bib49))], and CMRC [(Cui et al., [2019](#bib.bib10))].

Reference disambiguation datasets include WinoGrande [Sakaguchi et al. ([2019](#bib.bib44))] and CLUEWSC [(Xu et al., [2020](#bib.bib54))].

Language modeling datasets include Pile [(Gao et al., [2020](#bib.bib17))].

Chinese understanding and culture datasets include CHID [(Zheng et al., [2019](#bib.bib58))] and CCPM [(Li et al., [2021](#bib.bib33))].

Math datasets include GSM8K [(Cobbe et al., [2021](#bib.bib9))], MATH [(Hendrycks et al., [2021](#bib.bib21))], and CMath [(Wei et al., [2023](#bib.bib53))].

Code datasets include HumanEval [(Chen et al., [2021](#bib.bib7))], MBPP [(Austin et al., [2021](#bib.bib4))], and CRUXEval [(Gu et al., [2024](#bib.bib19))].

Standardized exams include AGIEval [(Zhong et al., [2023](#bib.bib60))].
Note that AGIEval includes both English and Chinese subsets.

Following our previous work [(DeepSeek-AI, [2024](#bib.bib13))], we adopt perplexity-based evaluation for datasets including HellaSwag, PIQA, WinoGrande, RACE-Middle, RACE-High, MMLU, ARC-Easy, ARC-Challenge, CHID, C-Eval, CMMLU, C3, and CCPM, and adopt generation-based evaluation for TriviaQA, NaturalQuestions, DROP, MATH, GSM8K, HumanEval, MBPP, CRUXEval, BBH, AGIEval, CLUEWSC, CMRC, and CMath.
In addition, we perform language-modeling-based evaluation for Pile-test and use Bits-Per-Byte (BPB) as the metric to guarantee fair comparison among models with different tokenizers.

For an intuitive overview of these benchmarks, we additionally provide our evaluation formats for each benchmark in Appendix [G](#A7).

#### 3.2.2 Evaluation Results

*Table 2: Comparison among DeepSeek-V2 and other representative open-source models. All models are evaluated in our internal framework and share the same evaluation setting. Bold denotes the best and underline denotes the second-best. Scores with a gap smaller than 0.3 are regarded as at the same level. With only 21B activated parameters, DeepSeek-V2 achieves top-tier performance among open-source models.*

In Table [2](#S3.T2), we compare DeepSeek-V2 with several representative open-source models, including DeepSeek 67B [(DeepSeek-AI, [2024](#bib.bib13))] (our previous release), Qwen1.5 72B [(Bai et al., [2023](#bib.bib5))], LLaMA3 70B [(AI@Meta, [2024](#bib.bib1))], and Mixtral 8x22B [(Mistral, [2024](#bib.bib36))].
We evaluate all these models with our internal evaluation framework, and ensure that they share the same evaluation setting.
Overall, with only 21B activated parameters, DeepSeek-V2 significantly outperforms DeepSeek 67B on almost all benchmarks, and achieves top-tier performance among open-source models.

Further, we elaborately compare DeepSeek-V2 with its open-source counterparts one by one.
(1)
Compared with Qwen1.5 72B, another model that supports both Chinese and English, DeepSeek-V2 demonstrates overwhelming advantages on the majority of English, code, and math benchmarks.
As for Chinese benchmarks, Qwen1.5 72B shows better performance on multi-subject multiple-choice tasks while DeepSeek-V2 is comparable or better on others.
Note that for the CHID benchmark, the tokenizer of Qwen1.5 72B will encounter errors in our evaluation framework, so we leave the CHID score blank for Qwen1.5 72B.
(2)
Compared with Mixtral 8x22B, DeepSeek-V2 achieves comparable or better English performance, except for TriviaQA, NaturalQuestions, and HellaSwag, which are closely related to English commonsense knowledge.
Notably, DeepSeek-V2 outperforms Mixtral 8x22B on MMLU.
On code and math benchmarks, DeepSeek-V2 demonstrates comparable performance with Mixtral 8x22B.
Since Mixtral 8x22B is not specifically trained on Chinese data, its Chinese capability lags far behind DeepSeek-V2.
(3)
Compared with LLaMA3 70B, DeepSeek-V2 is trained on fewer than a quarter of English tokens.
Therefore, we acknowledge that DeepSeek-V2 still has a slight gap in basic English capabilities with LLaMA3 70B.
However, even with much fewer training tokens and activated parameters, DeepSeek-V2 still demonstrates comparable code and math capability with LLaMA3 70B.
Also, as a bilingual language model, DeepSeek-V2 outperforms LLaMA3 70B overwhelmingly on Chinese benchmarks.

Finally, it is worth mentioning that certain prior studies [(Hu et al., [2024](#bib.bib24))] incorporate SFT data during the pre-training stage, whereas DeepSeek-V2 has never been exposed to SFT data during pre-training.

#### 3.2.3 Training and Inference Efficiency

##### Training Costs.

Since DeepSeek-V2 activates fewer parameters for each token and requires fewer FLOPs than DeepSeek 67B, training DeepSeek-V2 will be more economical than training DeepSeek 67B theoretically.
Although training an MoE model will introduce additional communication overheads, through our operator and communication optimizations, the training for DeepSeek-V2 can attain a relatively high Model FLOPs Utilization (MFU).
During our practical training on the H800 cluster, for training on each trillion tokens, DeepSeek 67B requires 300.6K GPU hours, while DeepSeek-V2 needs only 172.8K GPU hours, i.e., sparse DeepSeek-V2 can save 42.5% training costs compared with dense DeepSeek 67B.

##### Inference Efficiency.

In order to efficiently deploy DeepSeek-V2 for service, we first convert its parameters into the precision of FP8.
In addition, we also perform KV cache quantization [(Hooper et al., [2024](#bib.bib23); Zhao et al., [2023](#bib.bib57))] for DeepSeek-V2 to further compress each element in its KV cache into 6 bits on average.
Benefiting from MLA and these optimizations, actually deployed DeepSeek-V2 requires significantly less KV cache than DeepSeek 67B, and thus can serve a much larger batch size.
We evaluate the generation throughput of DeepSeek-V2 based on the prompt and generation length distribution from the actually deployed DeepSeek 67B service.
On a single node with 8 H800 GPUs, DeepSeek-V2 achieves a generation throughput exceeding 50K tokens per second, which is 5.76 times the maximum generation throughput of DeepSeek 67B.
In addition, the prompt input throughput of DeepSeek-V2 exceeds 100K tokens per second.

## 4 Alignment

### 4.1 Supervised Fine-Tuning

Building upon our prior research [(DeepSeek-AI, [2024](#bib.bib13))], we curate our instruction tuning datasets to include 1.5M instances, comprising 1.2M instances for helpfulness and 0.3M instances for safety.
In comparison to the initial version, we improve the data quality to mitigate hallucinatory responses and enhance writing proficiency.
We fine-tune DeepSeek-V2 with 2 epochs, and the learning rate is set to $5\times 10^{-6}$.
For the evaluation of DeepSeek-V2 Chat (SFT), we mainly include generation-based benchmarks, except for several representative multiple-choice tasks (MMLU and ARC).
We also conduct an instruction-following evaluation (IFEval) [(Zhou et al., [2023](#bib.bib62))] for DeepSeek-V2 Chat (SFT), using prompt-level loose accuracy as the metric.
Moreover, we employ LiveCodeBench [(Jain et al., [2024](#bib.bib26))] questions from September 1st, 2023 to April 1st, 2024 to evaluate chat models.
In addition to the standard benchmarks, we further evaluate our model on open-ended conversation benchmarks including MT-Bench [(Zheng et al., [2023](#bib.bib59))], AlpacaEval 2.0 [(Dubois et al., [2024](#bib.bib15))], and AlignBench [(Liu et al., [2023](#bib.bib34))].
For comparison, we also evaluate Qwen1.5 72B Chat, LLaMA-3-70B Instruct, and Mistral-8x22B Instruct in our evaluation framework and settings.
As for DeepSeek 67B Chat, we directly refer to the evaluation results reported in our previous release.

### 4.2 Reinforcement Learning

In order to further unlock the potential of DeepSeek-V2 and align it with human preference, we conduct Reinforcement Learning (RL) to adjust its preference.

##### Reinforcement Learning Algorithm.

In order to save the training costs of RL, we adopt Group Relative Policy Optimization (GRPO) [(Shao et al., [2024](#bib.bib45))], which foregoes the critic model that is typically with the same size as the policy model, and estimates the baseline from group scores instead.
Specifically, for each question $q$, GRPO samples a group of outputs $\{o_{1},o_{2},\cdots,o_{G}\}$ from the old policy $\pi_{\theta_{old}}$ and then optimizes the policy model $\pi_{\theta}$ by maximizing the following objective:

$$\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1% }^{G}\sim\pi_{\theta_{old}}(O|q)]}\\ &\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{% \theta_{old}}(o_{i}|q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi% _{\theta_{old}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta% \mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right),\end{split}$$ \tag{32}

$$\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)=\frac{\pi_{ref}(o_{i}|q)}{% \pi_{\theta}(o_{i}|q)}-\log\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-1,$$ \tag{33}

where $\varepsilon$ and $\beta$ are hyper-parameters;
and $A_{i}$ is the advantage, computed using a group of rewards $\{r_{1},r_{2},\ldots,r_{G}\}$ corresponding to the outputs within each group:

$$A_{i}=\frac{r_{i}-{\mathrm{m}ean(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{s}td% (\{r_{1},r_{2},\cdots,r_{G}\})}}.$$ \tag{34}

##### Training Strategy.

In our preliminary experiments, we find that the RL training on reasoning data, such as code and math prompts, exhibits unique characteristics that are distinct from the training on general data.
For example, the mathematical and coding abilities of our model can keep improving over a longer period of training steps.
Therefore, we employ a two-stage RL training strategy, which first performs reasoning alignment, and then performs human preference alignment.
In the first reasoning alignment stage, we train a reward model $RM_{reasoning}$ for code and math reasoning tasks, and optimize the policy model with the feedback of $RM_{reasoning}$:

$$r_{i}=RM_{reasoning}(o_{i}).$$ \tag{35}

In the second human preference alignment stage, we adopt a multi-reward framework, which acquires rewards from a helpful reward model $RM_{helpful}$, a safety reward model $RM_{safety}$, and a rule-based reward model $RM_{rule}$.
The final reward of a response $o_{i}$ is

$$r_{i}=c_{1}\cdot RM_{helpful}(o_{i})+c_{2}\cdot RM_{safety}(o_{i})+c_{3}\cdot RM% _{rule}(o_{i}),$$ \tag{36}

where $c_{1}$, $c_{2}$, and $c_{3}$ are corresponding coefficients.

In order to obtain reliable reward models that play crucial roles in the RL training, we carefully collect preference data, and meticulously conduct quality filtering and proportion adjustments.
We obtain code preference data based on compiler-feedback, and mathematical preference data based on the ground-truth labels.
For reward model training, we initialize the reward models with DeepSeek-V2 Chat (SFT) and train them with either a point-wise or a pair-wise loss.
In our experiments, we observe that the RL training can fully tap into and activate the potential of our model, enabling it to select the correct and satisfactory answer from possible responses.

##### Optimizations for Training Efficiency.

Conducting RL training on extremely large models places high demands on the training framework.
It requires careful engineering optimization to manage the GPU memory and RAM pressure, and meanwhile maintain a fast training speed.
For this goal, we implement the following engineering optimizations.
(1) Firstly, we propose a hybrid engine that adopts different parallel strategies for training and inference respectively to achieve higher GPU utilization.
(2) Secondly, we leverage vLLM [(Kwon et al., [2023](#bib.bib29))] with large batch sizes as our inference backend to accelerate the inference speed.
(3) Thirdly, we carefully design a scheduling strategy for offloading models to CPUs and loading models back to GPUs, which achieves a near-optimal balance between the training speed and memory consumption.

### 4.3 Evaluation Results

Evaluations on Standard Benchmarks.
Initially, we evaluate DeepSeek-V2 Chat (SFT) and DeepSeek-V2 Chat (RL) on standard benchmarks.
Notably, DeepSeek-V2 Chat (SFT) demonstrates substantial improvements in GSM8K, MATH, and HumanEval evaluations compared with its base version.
This progress can be attributed to the inclusion of our SFT data, which comprises a considerable volume of math and code related content.
In addition, DeepSeek-V2 Chat (RL) further boosts the performance on math and code benchmarks.
We show more code and math evaluations in Appendix [F](#A6).

As for the comparisons with other models, we first compare DeepSeek-V2 Chat (SFT) with Qwen1.5 72B Chat, and find that DeepSeek-V2 Chat (SFT) surpasses Qwen1.5 72B Chat on almost all of English, math, and code benchmarks.
On Chinese benchmarks, DeepSeek-V2 Chat (SFT) demonstrates slightly lower scores than Qwen1.5 72B Chat on multi-subject multiple-choice tasks, consistent with the performance observed from their base versions.
When compared with the state-of-the-art open-source MoE model, Mixtral 8x22B Instruct, DeepSeek-V2 Chat (SFT) exhibits better performance on most benchmarks, except for NaturalQuestions and IFEval.
Furthermore, in comparison to the state-of-the-art open-source model LLaMA3 70B Chat, DeepSeek-V2 Chat (SFT) shows similar performance in code and math related benchmarks.
LLaMA3 70B Chat exhibits better performance on MMLU and IFEval, while DeepSeek-V2 Chat (SFT) showcases stronger performance on Chinese tasks.
Ultimately, DeepSeek-V2 Chat (RL) demonstrates further enhanced performance in both mathematical and coding tasks compared with DeepSeek-V2 Chat (SFT).
These comparisons highlight the strengths of DeepSeek-V2 Chat in relation to other language models in various domains and languages.

*Table 3: Comparison among DeepSeek-V2 Chat (SFT), DeepSeek-V2 Chat (RL), and other representative open-source chat models. Regarding TriviaQA and NaturalQuestions, it is worth noting that chat models, such as LLaMA3 70B Instruct, might not strictly adhere to the format constraints typically specified in the few-shot setting. Consequently, this can lead to underestimation of certain models in our evaluation framework.*

Evaluations on Open-Ended Generation.
We proceed with additional evaluations of our models on open-ended conversation benchmarks.
For English open-ended conversation generation, we utilize MT-Bench and AlpacaEval 2.0 as the benchmarks.
Evaluation results presented in Table [4](#S4.T4) demonstrate a significant performance advantage of DeepSeek-V2 Chat (RL) over DeepSeek-V2 Chat (SFT).
This outcome showcases the effectiveness of our RL training in achieving improved alignment.
In comparison to other open-source models, DeepSeek-V2 Chat (RL) demonstrates superior performance over Mistral 8x22B Instruct and Qwen1.5 72B Chat on both benchmarks.
When compared with LLaMA3 70B Instruct, DeepSeek-V2 Chat (RL) showcases competitive performance on MT-Bench and notably outperforms it on AlpacaEval 2.0.
These results highlight the strong performance of DeepSeek-V2 Chat (RL) in generating high-quality and contextually relevant responses, particularly in instruction-based conversation tasks.

*Table 4: English open-ended conversation evaluations. For AlpacaEval 2.0, we use the length-controlled win rate as the metric.*

In addition, we evaluate the Chinese open-ended generation capability based on AlignBench.
As presented in Table [5](#S4.T5), DeepSeek-V2 Chat (RL) exhibits a slight advantage over DeepSeek-V2 Chat (SFT).
Notably, DeepSeek-V2 Chat (SFT) surpasses all open-source Chinese models by a significant margin.
It significantly outperforms the second-best open-source model, Qwen1.5 72B Chat on both Chinese reasoning and language.
Moreover, both DeepSeek-V2 Chat (SFT) and DeepSeek-V2 Chat (RL) outperform GPT-4-0613 and ERNIEBot 4.0, solidifying the position of our models in the top-tier LLMs that support Chinese.
Specifically, DeepSeek-V2 Chat (RL) shows remarkable performance in Chinese language understanding, which outperforms all models including GPT-4-Turbo-1106-Preview.
On the other hand, the reasoning capability of DeepSeek-V2 Chat (RL) still lags behind giant models, such as Erniebot-4.0 and GPT-4s.

*Table 5: AlignBench leaderboard rated by GPT-4-0613. Models are ranked in descending order based on the overall score. Models marked with * represent that we evaluate them through their API service or open-weighted model, instead of referring to the results reported in their original papers. Suffixes of Erniebot-4.0 and Moonshot denote the timestamps when we called their API.*

### 4.4 Discussion

##### Amount of SFT Data.

The discussion surrounding the necessity of a large SFT corpus has been a topic of intense debate.
Previous works [(Young et al., [2024](#bib.bib55); Zhou et al., [2024](#bib.bib61))] argue that fewer than 10K instances of SFT data are enough to produce satisfactory results.
However, in our experiments, we observe a significant performance decline on the IFEval benchmark if we use fewer than 10K instances.
A possible explanation is that, a language model necessitates a certain amount of data to develop specific skills.
Although the requisite data amount may diminish with the model size increasing, it cannot be entirely eliminated.
Our observation underscores the critical need for sufficient data to equip an LLM with desired capabilities.
Moreover, the quality of SFT data is also crucial, especially for tasks involving writing or open-ended questions.

##### Alignment Tax of Reinforcement Learning.

During human preference alignment, we observe a significant performance enhancement on the open-ended generation benchmarks, in terms of the scores rated by both AI and human evaluators.
However, we also notice a phenomenon of “alignment tax” [(Ouyang et al., [2022](#bib.bib39))], i.e., the alignment process can negatively impact the performance on some standard benchmarks such as BBH.
In order to alleviate the alignment tax, during the RL stage, we make significant efforts in data processing and improving training strategies, finally achieving a tolerable trade-off between the performance on standard and open-ended benchmarks.
Exploring how to align a model with human preferences without compromising its general performance presents a valuable direction for future research.

##### Online Reinforcement Learning.

In our preference alignment experiments, we find that the online approach significantly outperforms the offline approach.
Therefore, we invest tremendous efforts in implementing an online RL framework for aligning DeepSeek-V2.
The conclusion about online or offline preference alignment can vary in different contexts, and we reserve a more thorough comparison and analysis between them for future work.

## 5 Conclusion, Limitation, and Future Work

In this paper, we introduce DeepSeek-V2, a large MoE language model that supports 128K context length.
In addition to strong performance, it is also characterized by economical training and efficient inference, benefiting from its innovative architecture including MLA and DeepSeekMoE.
In practice, compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger
performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times.
Evaluation results further demonstrate that with only 21B activated parameters, DeepSeek-V2 achieves top-tier performance among open-source models and becomes the strongest open-source MoE model.

DeepSeek-V2 and its chat versions share the acknowledged limitations commonly found in other LLMs, including the lack of ongoing knowledge updates after pre-training, the possibility of generating non-factual information such as unverified advice, and a chance to produce hallucinations.
In addition, since our data primarily consist of Chinese and English content, our model may exhibit limited proficiency in other languages.
In scenarios beyond Chinese and English, it should be used with caution.

DeepSeek will continuously invest in open-source large models with longtermism, aiming to progressively approach the goal of artificial general intelligence.

-
•

In our ongoing exploration, we are dedicated to devising methods that enable further scaling up MoE models while maintaining economical training and inference costs.
The goal of our next step is to achieve performance on par with GPT-4 in our upcoming release.

-
•

Our alignment team continuously strives to enhance our models, aiming to develop a model that is not only helpful but also honest and safe for worldwide users.
Our ultimate objective is to align the values of our model with human values, while minimizing the need for human supervision.
By prioritizing ethical considerations and responsible development, we are dedicated to creating a positive and beneficial impact on society.

-
•

Currently, DeepSeek-V2 is designed to support the text modality exclusively.
In our forward-looking agenda, we intend to enable our model to support multiple modalities, enhancing its versatility and utility in a wider range of scenarios.

## References

-
AI@Meta (2024)

AI@Meta.

Llama 3 model card, 2024.

URL [https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).

-
Ainslie et al. (2023)

J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai.

Gqa: Training generalized multi-query transformer models from multi-head checkpoints.

arXiv preprint arXiv:2305.13245*, 2023.

-
Anthropic (2023)

Anthropic.

Introducing Claude, 2023.

URL [https://www.anthropic.com/index/introducing-claude](https://www.anthropic.com/index/introducing-claude).

-
Austin et al. (2021)

J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al.

Program synthesis with large language models.

*arXiv preprint arXiv:2108.07732*, 2021.

-
Bai et al. (2023)

J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu.

Qwen technical report.

*arXiv preprint arXiv:2309.16609*, 2023.

-
Bisk et al. (2020)

Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi.

PIQA: reasoning about physical commonsense in natural language.

In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pages 7432–7439. AAAI Press, 2020.

[10.1609/aaai.v34i05.6239](https:/doi.org/10.1609/aaai.v34i05.6239).

URL [https://doi.org/10.1609/aaai.v34i05.6239](https://doi.org/10.1609/aaai.v34i05.6239).

-
Chen et al. (2021)

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba.

Evaluating large language models trained on code.

*CoRR*, abs/2107.03374, 2021.

URL [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374).

-
Clark et al. (2018)

P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord.

Think you have solved question answering? try arc, the AI2 reasoning challenge.

*CoRR*, abs/1803.05457, 2018.

URL [http://arxiv.org/abs/1803.05457](http://arxiv.org/abs/1803.05457).

-
Cobbe et al. (2021)

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al.

Training verifiers to solve math word problems.

*arXiv preprint arXiv:2110.14168*, 2021.

-
Cui et al. (2019)

Y. Cui, T. Liu, W. Che, L. Xiao, Z. Chen, W. Ma, S. Wang, and G. Hu.

A span-extraction dataset for Chinese machine reading comprehension.

In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 5883–5889, Hong Kong, China, Nov. 2019. Association for Computational Linguistics.

[10.18653/v1/D19-1600](https:/doi.org/10.18653/v1/D19-1600).

URL [https://aclanthology.org/D19-1600](https://aclanthology.org/D19-1600).

-
Dai et al. (2024)

D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang.

Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.

*CoRR*, abs/2401.06066, 2024.

URL [https://doi.org/10.48550/arXiv.2401.06066](https://doi.org/10.48550/arXiv.2401.06066).

-
Dao (2023)

T. Dao.

FlashAttention-2: Faster attention with better parallelism and work partitioning, 2023.

-
DeepSeek-AI (2024)

DeepSeek-AI.

Deepseek LLM: scaling open-source language models with longtermism.

*CoRR*, abs/2401.02954, 2024.

URL [https://doi.org/10.48550/arXiv.2401.02954](https://doi.org/10.48550/arXiv.2401.02954).

-
Dua et al. (2019)

D. Dua, Y. Wang, P. Dasigi, G. Stanovsky, S. Singh, and M. Gardner.

DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.

In J. Burstein, C. Doran, and T. Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pages 2368–2378. Association for Computational Linguistics, 2019.

[10.18653/V1/N19-1246](https:/doi.org/10.18653/V1/N19-1246).

URL [https://doi.org/10.18653/v1/n19-1246](https://doi.org/10.18653/v1/n19-1246).

-
Dubois et al. (2024)

Y. Dubois, B. Galambosi, P. Liang, and T. B. Hashimoto.

Length-controlled alpacaeval: A simple way to debias automatic evaluators.

*arXiv preprint arXiv:2404.04475*, 2024.

-
Fedus et al. (2021)

W. Fedus, B. Zoph, and N. Shazeer.

Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.

*CoRR*, abs/2101.03961, 2021.

URL [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961).

-
Gao et al. (2020)

L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, et al.

The Pile: An 800GB dataset of diverse text for language modeling.

*arXiv preprint arXiv:2101.00027*, 2020.

-
Google (2023)

Google.

Introducing gemini: our largest and most capable ai model, 2023.

URL [https://blog.google/technology/ai/google-gemini-ai/](https://blog.google/technology/ai/google-gemini-ai/).

-
Gu et al. (2024)

A. Gu, B. Rozière, H. Leather, A. Solar-Lezama, G. Synnaeve, and S. I. Wang.

Cruxeval: A benchmark for code reasoning, understanding and execution, 2024.

-
Hendrycks et al. (2020)

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt.

Measuring massive multitask language understanding.

*arXiv preprint arXiv:2009.03300*, 2020.

-
Hendrycks et al. (2021)

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt.

Measuring mathematical problem solving with the math dataset.

*arXiv preprint arXiv:2103.03874*, 2021.

-
High-flyer (2023)

High-flyer.

Hai-llm: 高效且轻量的大模型训练工具, 2023.

URL [https://www.high-flyer.cn/en/blog/hai-llm](https://www.high-flyer.cn/en/blog/hai-llm).

-
Hooper et al. (2024)

C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami.

Kvquant: Towards 10 million context length LLM inference with KV cache quantization.

*CoRR*, abs/2401.18079, 2024.

URL [https://doi.org/10.48550/arXiv.2401.18079](https://doi.org/10.48550/arXiv.2401.18079).

-
Hu et al. (2024)

S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al.

Minicpm: Unveiling the potential of small language models with scalable training strategies.

*arXiv preprint arXiv:2404.06395*, 2024.

-
Huang et al. (2023)

Y. Huang, Y. Bai, Z. Zhu, J. Zhang, J. Zhang, T. Su, J. Liu, C. Lv, Y. Zhang, J. Lei, et al.

C-Eval: A multi-level multi-discipline chinese evaluation suite for foundation models.

*arXiv preprint arXiv:2305.08322*, 2023.

-
Jain et al. (2024)

N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica.

Livecodebench: Holistic and contamination free evaluation of large language models for code.

*arXiv preprint arXiv:2403.07974*, 2024.

-
Joshi et al. (2017)

M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer.

TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension.

In R. Barzilay and M.-Y. Kan, editors, *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics.

[10.18653/v1/P17-1147](https:/doi.org/10.18653/v1/P17-1147).

URL [https://aclanthology.org/P17-1147](https://aclanthology.org/P17-1147).

-
Kwiatkowski et al. (2019)

T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. P. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov.

Natural questions: a benchmark for question answering research.

*Trans. Assoc. Comput. Linguistics*, 7:452–466, 2019.

[10.1162/tacl_a_00276](https:/doi.org/10.1162/tacl_a_00276).

URL [https://doi.org/10.1162/tacl_a_00276](https://doi.org/10.1162/tacl_a_00276).

-
Kwon et al. (2023)

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica.

Efficient memory management for large language model serving with pagedattention.

In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.

-
Lai et al. (2017)

G. Lai, Q. Xie, H. Liu, Y. Yang, and E. H. Hovy.

RACE: large-scale reading comprehension dataset from examinations.

In M. Palmer, R. Hwa, and S. Riedel, editors, *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017*, pages 785–794. Association for Computational Linguistics, 2017.

[10.18653/V1/D17-1082](https:/doi.org/10.18653/V1/D17-1082).

URL [https://doi.org/10.18653/v1/d17-1082](https://doi.org/10.18653/v1/d17-1082).

-
Lepikhin et al. (2021)

D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen.

Gshard: Scaling giant models with conditional computation and automatic sharding.

In *9th International Conference on Learning Representations, ICLR 2021*. OpenReview.net, 2021.

URL [https://openreview.net/forum?id=qrwe7XHTmYb](https://openreview.net/forum?id=qrwe7XHTmYb).

-
Li et al. (2023)

H. Li, Y. Zhang, F. Koto, Y. Yang, H. Zhao, Y. Gong, N. Duan, and T. Baldwin.

CMMLU: Measuring massive multitask language understanding in Chinese.

*arXiv preprint arXiv:2306.09212*, 2023.

-
Li et al. (2021)

W. Li, F. Qi, M. Sun, X. Yi, and J. Zhang.

Ccpm: A chinese classical poetry matching dataset, 2021.

-
Liu et al. (2023)

X. Liu, X. Lei, S. Wang, Y. Huang, Z. Feng, B. Wen, J. Cheng, P. Ke, Y. Xu, W. L. Tam, X. Zhang, L. Sun, H. Wang, J. Zhang, M. Huang, Y. Dong, and J. Tang.

Alignbench: Benchmarking chinese alignment of large language models.

*CoRR*, abs/2311.18743, 2023.

[10.48550/ARXIV.2311.18743](https:/doi.org/10.48550/ARXIV.2311.18743).

URL [https://doi.org/10.48550/arXiv.2311.18743](https://doi.org/10.48550/arXiv.2311.18743).

-
Loshchilov and Hutter (2017)

I. Loshchilov and F. Hutter.

Decoupled weight decay regularization.

*arXiv preprint arXiv:1711.05101*, 2017.

-
Mistral (2024)

Mistral.

Cheaper, better, faster, stronger: Continuing to push the frontier of ai and making it accessible to all, 2024.

URL [https://mistral.ai/news/mixtral-8x22b](https://mistral.ai/news/mixtral-8x22b).

-
OpenAI (2022)

OpenAI.

Introducing ChatGPT, 2022.

URL [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt).

-
OpenAI (2023)

OpenAI.

GPT4 technical report.

*arXiv preprint arXiv:2303.08774*, 2023.

-
Ouyang et al. (2022)

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al.

Training language models to follow instructions with human feedback.

*Advances in neural information processing systems*, 35:27730–27744, 2022.

-
Peng et al. (2023)

B. Peng, J. Quesnelle, H. Fan, and E. Shippole.

Yarn: Efficient context window extension of large language models.

*arXiv preprint arXiv:2309.00071*, 2023.

-
Qi et al. (2023)

P. Qi, X. Wan, G. Huang, and M. Lin.

Zero bubble pipeline parallelism.

*arXiv preprint arXiv:2401.10241*, 2023.

-
Rajbhandari et al. (2020)

S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He.

Zero: Memory optimizations toward training trillion parameter models.

In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–16. IEEE, 2020.

-
Riquelme et al. (2021)

C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pinto, D. Keysers, and N. Houlsby.

Scaling vision with sparse mixture of experts.

In *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021*, pages 8583–8595, 2021.

URL [https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html).

-
Sakaguchi et al. (2019)

K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi.

Winogrande: An adversarial winograd schema challenge at scale, 2019.

-
Shao et al. (2024)

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo.

Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

*arXiv preprint arXiv:2402.03300*, 2024.

-
Shazeer (2019)

N. Shazeer.

Fast transformer decoding: One write-head is all you need.

*CoRR*, abs/1911.02150, 2019.

URL [http://arxiv.org/abs/1911.02150](http://arxiv.org/abs/1911.02150).

-
Shazeer et al. (2017)

N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean.

Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.

In *5th International Conference on Learning Representations, ICLR 2017*. OpenReview.net, 2017.

URL [https://openreview.net/forum?id=B1ckMDqlg](https://openreview.net/forum?id=B1ckMDqlg).

-
Su et al. (2024)

J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu.

Roformer: Enhanced transformer with rotary position embedding.

*Neurocomputing*, 568:127063, 2024.

-
Sun et al. (2019)

K. Sun, D. Yu, D. Yu, and C. Cardie.

Investigating prior knowledge for challenging chinese machine reading comprehension, 2019.

-
Suzgun et al. (2022)

M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al.

Challenging big-bench tasks and whether chain-of-thought can solve them.

*arXiv preprint arXiv:2210.09261*, 2022.

-
Vaswani et al. (2017)

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin.

Attention is all you need.

*Advances in neural information processing systems*, 30, 2017.

-
Wei et al. (2022)

J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al.

Emergent abilities of large language models.

*arXiv preprint arXiv:2206.07682*, 2022.

-
Wei et al. (2023)

T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang.

Cmath: Can your language model pass chinese elementary school math test?, 2023.

-
Xu et al. (2020)

L. Xu, H. Hu, X. Zhang, L. Li, C. Cao, Y. Li, Y. Xu, K. Sun, D. Yu, C. Yu, Y. Tian, Q. Dong, W. Liu, B. Shi, Y. Cui, J. Li, J. Zeng, R. Wang, W. Xie, Y. Li, Y. Patterson, Z. Tian, Y. Zhang, H. Zhou, S. Liu, Z. Zhao, Q. Zhao, C. Yue, X. Zhang, Z. Yang, K. Richardson, and Z. Lan.

CLUE: A chinese language understanding evaluation benchmark.

In D. Scott, N. Bel, and C. Zong, editors, *Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020*, pages 4762–4772. International Committee on Computational Linguistics, 2020.

[10.18653/V1/2020.COLING-MAIN.419](https:/doi.org/10.18653/V1/2020.COLING-MAIN.419).

URL [https://doi.org/10.18653/v1/2020.coling-main.419](https://doi.org/10.18653/v1/2020.coling-main.419).

-
Young et al. (2024)

A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, H. Li, J. Zhu, J. Chen, J. Chang, et al.

Yi: Open foundation models by 01. ai.

*arXiv preprint arXiv:2403.04652*, 2024.

-
Zellers et al. (2019)

R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi.

HellaSwag: Can a machine really finish your sentence?

In A. Korhonen, D. R. Traum, and L. Màrquez, editors, *Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers*, pages 4791–4800. Association for Computational Linguistics, 2019.

[10.18653/v1/p19-1472](https:/doi.org/10.18653/v1/p19-1472).

URL [https://doi.org/10.18653/v1/p19-1472](https://doi.org/10.18653/v1/p19-1472).

-
Zhao et al. (2023)

Y. Zhao, C. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci.

Atom: Low-bit quantization for efficient and accurate LLM serving.

*CoRR*, abs/2310.19102, 2023.

URL [https://doi.org/10.48550/arXiv.2310.19102](https://doi.org/10.48550/arXiv.2310.19102).

-
Zheng et al. (2019)

C. Zheng, M. Huang, and A. Sun.

Chid: A large-scale chinese idiom dataset for cloze test.

In A. Korhonen, D. R. Traum, and L. Màrquez, editors, *Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers*, pages 778–787. Association for Computational Linguistics, 2019.

[10.18653/V1/P19-1075](https:/doi.org/10.18653/V1/P19-1075).

URL [https://doi.org/10.18653/v1/p19-1075](https://doi.org/10.18653/v1/p19-1075).

-
Zheng et al. (2023)

L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica.

Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

-
Zhong et al. (2023)

W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan.

AGIEval: A human-centric benchmark for evaluating foundation models.

*CoRR*, abs/2304.06364, 2023.

[10.48550/arXiv.2304.06364](https:/doi.org/10.48550/arXiv.2304.06364).

URL [https://doi.org/10.48550/arXiv.2304.06364](https://doi.org/10.48550/arXiv.2304.06364).

-
Zhou et al. (2024)

C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al.

Lima: Less is more for alignment.

*Advances in Neural Information Processing Systems*, 36, 2024.

-
Zhou et al. (2023)

J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou.

Instruction-following evaluation for large language models.

*arXiv preprint arXiv:2311.07911*, 2023.

## Appendix

## Appendix A Contributions and Acknowledgments

Research & Engineering
Aixin Liu
Bingxuan Wang
Bo Liu
Chenggang Zhao
Chengqi Deng
Chong Ruan
Damai Dai
Daya Guo
Dejian Yang
Deli Chen
Erhang Li
Fangyun Lin
Fuli Luo
Guangbo Hao
Guanting Chen
Guowei Li
H. Zhang
Hanwei Xu
Hao Yang
Haowei Zhang
Honghui Ding
Huajian Xin
Huazuo Gao
Hui Qu
Jianzhong Guo
Jiashi Li
Jingyang Yuan
Junjie Qiu
Junxiao Song
Kai Dong
Kaige Gao
Kang Guan
Lean Wang
Lecong Zhang
Liang Zhao
Liyue Zhang
Mingchuan Zhang
Minghua Zhang
Minghui Tang
Panpan Huang
Peiyi Wang
Qihao Zhu
Qinyu Chen
Qiushi Du
Ruiqi Ge
Ruizhe Pan
Runxin Xu
Shanghao Lu
Shangyan Zhou
Shanhuang Chen
Shengfeng Ye
Shirong Ma
Shiyu Wang
Shuiping Yu
Shunfeng Zhou
Size Zheng
Tian Pei
Wangding Zeng
Wen Liu
Wenfeng Liang
Wenjun Gao
Wentao Zhang
Xiao Bi
Xiaohan Wang
Xiaodong Liu
Xiaokang Chen
Xiaotao Nie
Xin Liu
Xin Xie
Xingkai Yu
Xinyu Yang
Xuan Lu
Xuecheng Su
Y. Wu
Y.K. Li
Y.X. Wei
Yanhong Xu
Yao Li
Yao Zhao
Yaofeng Sun
Yaohui Wang
Yichao Zhang
Yiliang Xiong
Yilong Zhao
Ying He
Yishi Piao
Yixin Dong
Yixuan Tan
Yiyuan Liu
Yongji Wang
Yongqiang Guo
Yuduan Wang
Yuheng Zou
Yuxiang You
Yuxuan Liu
Z.Z. Ren
Zehui Ren
Zhangli Sha
Zhe Fu
Zhenda Xie
Zhewen Hao
Zhihong Shao
Zhuoshu Li
Zihan Wang
Zihui Gu
Zilin Li
Ziwei Xie

Data Annotation
Bei Feng
Hui Li
J.L. Cai
Jiaqi Ni
Lei Xu
Meng Li
Ning Tian
R.J. Chen
R.L. Jin
Ruyi Chen
S.S. Li
Shuang Zhou
Tian Yuan
Tianyu Sun
X.Q. Li
Xiangyue Jin
Xiaojin Shen
Xiaosha Chen
Xiaowen Sun
Xiaoxiang Wang
Xinnan Song
Xinyi Zhou
Y.X. Zhu
Yanhong Xu
Yanping Huang
Yaohui Li
Yi Zheng
Yuchen Zhu
Yunxian Ma
Zhen Huang
Zhipeng Xu
Zhongyu Zhang

Business & Compliance
Bin Wang
Dongjie Ji
Jian Liang
Jin Chen
Leyi Xia
Miaojun Wang
Mingming Li
Peng Zhang
Shaoqing Wu
Shengfeng Ye
T. Wang
W.L. Xiao
Wei An
Xianzu Wang
Ying Tang
Yukun Zha
Yuting Yan
Zhen Zhang
Zhiniu Wen

Within each role, authors are listed alphabetically by first name.
Especially, Huazuo Gao and Wangding Zeng have made key innovations in the research of the MLA architecture.
Furthermore, we’d like to thank Jianlin Su for his helpful discussion on position embedding.
We thank all those who have contributed to DeepSeek-V2 but are not mentioned in the paper.
DeepSeek believes that innovation, novelty, and curiosity are essential in the path to AGI.

## Appendix B DeepSeek-V2-Lite: A 16B Model Equipped with MLA and DeepSeekMoE

### B.1 Model Description

##### Architectures.

DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048.
It also employs MLA and has 16 attention heads, where each head has a dimension of 128.
Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries.
For the decoupled queries and key, it has a per-head dimension of 64.
DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers.
Each MoE layer consists of 2 shared experts and 64 routed experts, where the intermediate hidden dimension of each expert is 1408.
Among the routed experts, 6 experts will be activated for each token.
Under this configuration, DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token.

*Table 6: Performance of DeepSeek-V2-Lite, DeepSeekMoE 16B, and DeepSeek 7B.*

##### Training Details.

DeepSeek-V2-Lite is also trained from scratch on the same pre-training corpus of DeepSeek-V2, which is not polluted by any SFT data.
It uses the AdamW optimizer with hyper-parameters set to $\beta_{1}=0.9$, $\beta_{2}=0.95$, and $\mathrm{weight\_decay}=0.1$.
The learning rate is scheduled using a warmup-and-step-decay strategy.
Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps.
Subsequently, the learning rate is multiplied by 0.316 after
training about 80% of tokens, and again by 0.316 after training about 90% of tokens.
The maximum learning rate is set to $4.2\times 10^{-4}$, and the gradient clipping norm is set to 1.0.
We do not employ the batch size scheduling strategy for it, and it is trained with a constant batch size of 4608 sequences.
During pre-training, we set the maximum sequence length to 4K, and train DeepSeek-V2-Lite on 5.7T tokens.
We leverage pipeline parallelism to deploy different layers of it on different devices, but for each layer, all experts will be deployed on the same device.
Therefore, we only employ a small expert-level balance loss with $\alpha_{1}=0.001$, and do not employ device-level balance loss and communication balance loss for it.
After pre-training, we also perform long context extension and SFT for DeepSeek-V2-Lite and get a chat model called DeepSeek-V2-Lite Chat.

*Table 7: Performance of DeepSeek-V2-Lite Chat, DeepSeekMoE 16B Chat, and DeepSeek 7B Chat.*

### B.2 Performance Evaluation

##### Base Model.

We evaluate the performance of DeepSeek-V2-Lite and compare it with our previous small-size base models in Table [6](#A2.T6).
DeepSeek-V2-Lite exhibits overwhelming performance advantages, especially in reasoning, coding, and math.

##### Chat Model.

We evaluate the performance of DeepSeek-V2-Lite Chat and compare it with our previous small-size chat models in Table [7](#A2.T7).
DeepSeek-V2-Lite also outperforms our previous small-size chat models by a large margin.

## Appendix C Full Formulas of MLA

In order to demonstrate the complete computation process of MLA, we provide its full formulas in the following:

$$
\begin{aligned}
\mathbf{c}_{t}^{Q} &=W^{DQ}\mathbf{h}_{t}, \tag{37}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{q}_{t,1}^{C};\mathbf{q}_{t,2}^{C};...;\mathbf{q}_{t,n_{h% }}^{C}]=\mathbf{q}_{t}^{C} &=W^{UQ}\mathbf{c}_{t}^{Q}, \tag{38}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{q}_{t,1}^{R};\mathbf{q}_{t,2}^{R};...;\mathbf{q}_{t,n_{h% }}^{R}]=\mathbf{q}_{t}^{R} &=\operatorname{RoPE}({W^{QR}}\mathbf{c}_{t}^{Q}), \tag{39}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{q}_{t,i} &=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}], \tag{40}
\end{aligned}
$$
$$
\begin{aligned}
\boxed{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}% {0,0,1}\mathbf{c}_{t}^{KV}} &=W^{DKV}\mathbf{h}_{t}, \tag{41}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{k}_{t,1}^{C};\mathbf{k}_{t,2}^{C};...;\mathbf{k}_{t,n_{h% }}^{C}]=\mathbf{k}_{t}^{C} &=W^{UK}\mathbf{c}_{t}^{KV}, \tag{42}
\end{aligned}
$$
$$
\begin{aligned}
\boxed{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}% {0,0,1}\mathbf{k}_{t}^{R}} &=\operatorname{RoPE}({W^{KR}}\mathbf{h}_{t}), \tag{43}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{k}_{t,i} &=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}], \tag{44}
\end{aligned}
$$
$$
\begin{aligned}
[\mathbf{v}_{t,1}^{C};\mathbf{v}_{t,2}^{C};...;\mathbf{v}_{t,n_{h% }}^{C}]=\mathbf{v}_{t}^{C} &=W^{UV}\mathbf{c}_{t}^{KV}, \tag{45}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{o}_{t,i} &=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}})\mathbf{v}_{j,i}^{C}, \tag{46}
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{u}_{t} &=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}], \tag{47}
\end{aligned}
$$

where the boxed vectors in blue need to be cached for generation.
During inference, the naive formula needs to recover $\mathbf{k}_{t}^{C}$ and $\mathbf{v}_{t}^{C}$ from $\mathbf{c}_{t}^{KV}$ for attention.
Fortunately, due to the associative law of matrix multiplication, we can absorb $W^{UK}$ into $W^{UQ}$, and $W^{UV}$ into $W^{O}$.
Therefore, we do not need to compute keys and values out for each query.
Through this optimization, we avoid the computational overhead for recomputing $\mathbf{k}_{t}^{C}$ and $\mathbf{v}_{t}^{C}$ during inference.

## Appendix D Ablation of Attention Mechanisms

### D.1 Ablation of MHA, GQA, and MQA

We show the evaluation results for 7B dense models with MHA, GQA, and MQA on four hard benchmarks in Table [8](#A4.T8).
All of these three models are trained on 1.33T tokens, and share the same architecture except for the attention mechanisms.
In addition, for a fair comparison, we align the number of parameters of them to around 7B by adjusting the number of layers.
From the table, we can find that MHA demonstrates significant advantages over GQA and MQA on these benchmarks.

*Table 8: Comparison among 7B dense models with MHA, GQA, and MQA, respectively. MHA demonstrates significant advantages over GQA and MQA on hard benchmarks.*

### D.2 Comparison Between MLA and MHA

In Table [9](#A4.T9), we show the evaluation results for MoE models equipped with MLA and MHA, respectively, on four hard benchmarks.
For a solid conclusion, we train and evaluate models across two scales.
Two small MoE models comprise about 16B total parameters, and we train them on 1.33T tokens.
Two large MoE models comprise about 250B total parameters, and we train them on 420B tokens.
Also, two small MoE models and two large MoE models respectively share the same architecture except for the attention mechanisms.
From the table, we can observe that MLA shows better performance than MHA.
More importantly, MLA requires a significantly smaller amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA.

*Table 9: Comparison between MLA and MHA on hard benchmarks. DeepSeek-V2 shows better performance than MHA, but requires a significantly smaller amount of KV cache.*

## Appendix E Discussion About Pre-Training Data Debiasing

During pre-training data preparation, we identify and filter out contentious content, such as values influenced by regional cultures, to avoid our model exhibiting unnecessary subjective biases on these controversial topics.
Consequently, we observe that DeepSeek-V2 performs slightly worse on the test sets that are closely associated with specific regional cultures.
For example, when evaluated on MMLU, although DeepSeek-V2 achieves comparable or superior performance on the majority of testsets compared with its competitors like Mixtral 8x22B, it still lags behind on the Humanity-Moral subset, which is mainly associated with American values.

Further, we conduct a manual analysis on this subset.
Three well-educated human annotators conduct independent annotations on 420 moral scenarios from the MMLU Humanity-Moral subset.
Then, we compute the agreement among their annotations and the ground-truth label.
As shown in Table [10](#A5.T10), three human annotators and the ground-truth label exhibit a low agreement with each other.
Therefore, we attribute the abnormal performance of DeepSeek-V2 on these value-sensitive test sets to our efforts in debiasing the pre-training corpus.

*Table 10: Three well-educated human annotators conduct independent annotations on 420 moral scenarios from the MMLU Humanity-Moral subset, on which DeepSeek-V2 and its competitive models demonstrate performance inconsistency. Three annotators and the ground-truth label exhibit a low agreement with each other. This indicates that the answers to the Humanity-Moral subset can be contentious according to specific regional cultures.*

## Appendix F Additional Evaluations on Math and Code

The evaluation employs the SC-Math6 corpus, which consists of thousands of Chinese math problems.
DeepSeek-V2 Chat (RL) outperforms all Chinese LLMs, including both open-source and close-source models.

*Table 11: SC-Math6 Model Reasoning Level. “R Level” stands for Reasoning Level, “Comp. Score” stands for Comprehensive Score, “Reas. Steps Score” stands for Reasoning Steps Score, and “OvrAcc Score” stands for Overall Accuracy Score.*

We further share more results in Figure [5](#A6.F5) on HumanEval and LiveCodeBench, where the questions of LiveCodeBench are selected from the period between September 1st, 2023, and April 1st, 2024.
As shown in the figure, DeepSeek-V2 Chat (RL) demonstrates considerable proficiency in LiveCodeBench, achieving a Pass@1 score that even surpasses some giant models.
This performance highlights the strong capability of DeepSeek-V2 Chat (RL) in tackling live coding tasks.

![Figure](x6.png)

*Figure 5: Evaluation results on HumanEval and LiveCodeBench. The questions of LiveCodeBench are selected from the period between September 1st, 2023 and April 1st, 2024.*

## Appendix G Evaluation Formats

We present our evaluation formats for each benchmark in Table [12](#A7.T12)-[37](#A7.T37), respectively.

*Table 12: An example of AGIEval.*

*Table 13: An example of ARC.*

*Table 14: An example of BBH.*

*Table 15: An example of C-Eval.*

*Table 16: An example of C3.*

*Table 17: An example of CCPM.*

*Table 18: An example of CMATH.*

*Table 19: An example of CMMLU.*

*Table 20: An example of CMRC2018.*

*Table 21: An example of DROP.*

*Table 22: An example of CHID.*

*Table 23: An example of CLUEWSC.*

*Table 24: An example of GSM8K.*

*Table 25: An example of HellaSwag.*

*Table 26: An example of HumanEval.*

*Table 27: An example of MATH.*

*Table 28: An example of MBPP.*

*Table 29: An example of MMLU.*

*Table 30: An example of NaturalQuestions.*

*Table 31: An example of OpenBookQA.*

*Table 32: An example of PIQA.*

*Table 33: An example of RACE.*

*Table 34: An example of TriviaQA.*

*Table 35: An example of WinoGrande. Note that there are multiple prefixes and only one completion for WinoGrande, and we choose the predicted prefix with the lowest perplexity of the completion.*

*Table 36: An example of CRUXEval-I.*

*Table 37: An example of CRUXEval-O.*

Generated on Wed Jun 19 06:04:00 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)