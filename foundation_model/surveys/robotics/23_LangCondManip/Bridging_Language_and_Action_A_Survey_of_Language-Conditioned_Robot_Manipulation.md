# Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation

Xiangtong Yao\*, Hongkuan Zhou\*, Oier Mees\*, Yuan Meng\*, Ted Xiao, Yonatan Bisk, Jean Oh, Edward Johns, Mohit Shridhar, Dhruv Shah, Jesse Thomason, Kai Huang, Joyce Chai, Zhenshan Bing, Alois Knoll

arXiv:2312.10807v6

###### Abstract

Language-conditioned robot manipulation is an emerging field aimed at enabling seamless communication and cooperation between humans and robotic agents by teaching robots to comprehend and execute instructions conveyed in natural language. This interdisciplinary area integrates scene understanding, language processing, and policy learning to bridge the gap between human instructions and robot actions. In this comprehensive survey, we systematically explore recent advancements in language-conditioned robot manipulation. We categorize existing methods based on the primary ways language is integrated into the robot system, namely language for state evaluation, language as a policy condition, language for cognitive planning and reasoning, and language in unified vision-language-action models. Specifically, we further analyze state-of-the-art techniques from five axes of action granularity, data and supervision regimes, system cost and latency, environments and evaluations, and cross-modal task specification. Additionally, we highlight the key debates in the field. Finally, we discuss open challenges and future research directions, focusing on potentially enhancing generalization capabilities and addressing safety issues in language-conditioned robot manipulators.

**Keywords:** language-conditioned learning, robot manipulation, diffusion models, large language models, vision language models, vision-language-action models, neuro-symbolic models, and foundation models.

## 1 Introduction

Robot manipulation, the ability of robots to physically interact with and manipulate objects in their environment (Billard and Kragic 2019), is an important component of autonomous systems. It has been widely adopted in structured environments, such as factory floors (Sanchez et al. 2018), where robots perform repetitive tasks with high precision and efficiency. One vision of robotics is to integrate intelligent systems into the fabric of everyday life, moving them from the environment with controlled settings to unstructured scenarios in homes, hospitals, and warehouses, where high levels of interaction between humans and robots are required (Silvera-Tawil 2024; Zhang and Xue 2025). A fundamental barrier, however, has long hindered this vision: the complexity of instructing robots in unstructured environments, particularly for non-expert users. Traditional methods for instructing robots are non-generalizable, such as specialized programming of control rules (Takeshita et al. 2025), teleoperation (Wu et al. 2019), or reward functions engineering (Ibarz et al. 2021). These methods demand expert efforts, which are inaccessible to the general public without extensive training. This limitation has impeded the broader deployment of general-purpose robots.

Language-conditioned robotic manipulation is emerging as a transformative solution to this challenge. It leverages natural language as an intuitive interface for humans to specify tasks, enabling robots to translate these high-level commands into physical actions. Enabling control via simple text or voice lowers the barrier for non-expert users and unlocks the potential for robots to function as general-purpose assistants, making robotics more accessible to a broader audience and increasing its importance. This paradigm shift is built on several key advantages.

- **Accessibility and usability:** Language conditioning allows people who are not experts to use robots. As noted by Tellex et al. (2020): "*most future robot users will not be programmers*". Natural language provides a "zero-learning" interface that allows anyone to specify complex and high-level goals without needing to understand the underlying code or control logic. Instead of navigating complex graphical user interfaces (GUIs) or writing scripts, a user can simply state their intent, such as "*bring me the red cup from the kitchen counter*", lowering the barrier to entry for a wide range of applications.

- **Trust through bidirectional communication:** Trust is essential for human-robot collaboration, especially in sensitive environments, such as assistive care (Jenamani et al. 2025). Language-conditioned systems offer a natural pathway to building that trust. The same channel used to command a robot can be used for on-the-fly correction and explanation. A user asks, "*I have a trach and have to eat slowly*" (Jenamani et al. 2025), and the robot can provide a rationale for its actions. These transparent feedback loops enable more intuitive and safer interactions, a key theme in human-robot interaction research focused on closing the loop between robot learning and communication (Habibian et al. 2025).

- **Transferring textual knowledge to robotics:** Real-world environments like homes and factories are unpredictable, with many unique situations that pre-programmed robots cannot handle. Language provides a compact interface for importing the world's common sense knowledge, such as properties, procedures, and safety rules, into the control system. By grounding natural-language commands to perception and action primitives, a system can parse goals like "*stack the two lightest boxes*" into perceptual queries, relational reasoning, and sequences of controllable skills. Leveraging broad linguistic priors reduces per-task engineering and enables zero/few-shot generalization to unseen tasks (Zitkovich et al. 2023; Zhang et al. 2025c), improving robustness when operating in dynamic environments.

Driven by these advantages, recent research goals in the emerging field of language-conditioned robot manipulation aim to enable natural language commands, instructions, and queries to robot systems that are translated into motor actions and behaviors conditioned on visual observations of the physical environment. Remarkable success has been seen in robotic arm control (Knoll et al. 1997; Zhang et al. 1999; Lynch and Sermanet 2021; Silva et al. 2021; Mees et al. 2022a; O'Neill et al. 2024; Bing et al. 2023a; Octo Model Team et al. 2024) cross-embodiment learning (Yang et al. 2024b; Doshi et al. 2025), and robot navigation tasks (Hermann et al. 2017; Fu et al. 2019; Hirose et al. 2025) such as autonomous driving (Sriram et al. 2019; Roh et al. 2020). Figure 1 demonstrates fundamental disciplines and research in language-conditioned robot manipulation. Success in this field requires tackling challenges across three key axes: language understanding, visual perception, and action generation. From a system perspective, it is useful to factor a language-conditioned manipulation pipeline into three interacting modules: Language, Perception, and Control. This decomposition is architectural rather than conventional robotics taxonomy: here, the language module includes instruction understanding, task representation, and often high-level semantic planning/reasoning; the perception module grounds language in observations and estimates the environment state; and the control module converts the resulting task specification into executable robot actions through learned policies or classical planners/controllers, as shown in Figure 2.

To extract semantics from natural language, early research works (Kress-Gazit et al. 2009; Raman et al. 2012) leverage formal specification languages such as temporal logic, which support formal verification of provided commands. Nevertheless, specifying instructions in these languages can be challenging, complex, and time-consuming. Large-scale pretraining approaches in natural language processing have led to text embedding models, like GloVe (Pennington et al. 2014), RoBERTa (Liu et al. 2019), and BERT (Devlin et al. 2019). In particular, large language models (LLMs) have shown impressive abilities in extracting semantic information and performing high-level planning and reasoning tasks.

To ground language commands in the physical scene, robots must perceive their environments, a challenge addressed by advances in computer vision. Foundational technologies include convolutional neural networks (e.g., ResNet (He et al. 2016)) for feature extraction, object detectors (e.g., Faster-RCNN (Girshick 2015), YOLO (Redmon et al. 2016)), and more recent transformer-based architectures like Vision Transformers (ViT) (Dosovitskiy et al. 2021). To connect visual perception with linguistic instructions, vision-language models (VLMs) such as CLIP (Radford et al. 2021) and Flamingo (Alayrac et al. 2022) have been developed. These models learn joint embeddings that align visual and textual data, enabling robots to perform tasks like language-conditioned object detection (Zang et al. 2025) and segmentation (Li et al. 2022), which are critical for identifying and localizing objects mentioned in a command.

To translate high-level linguistic goals into low-level robot actions, researchers explore several paradigms. One major approach is to learn a **language-conditioned policy** using methods like reinforcement learning (RL) (Sutton et al. 1998) or imitation learning (IL) (Argall et al. 2009). In this paradigm, the language command is provided as an input to the policy, which then directly outputs low-level actions (e.g., joint torques or end-effector velocities). For instance, in language-conditioned IL, the model learns to map from images and text commands to expert actions from a dataset of language-annotated demonstrations. A second popular approach decouples high-level reasoning from low-level control. In these methods, a large language model or vision-language model is used to parse the instruction and predict an intermediate goal, such as a target end-effector pose or a grasp point. This high-level prediction is then passed to a classical motion planner that uses inverse kinematics to compute the required joint movements (Habekost et al. 2024; Yang et al. 2025h). This modular design leverages the strong open-vocabulary and instruction-following priors of foundation models (FMs) for goal specification while relying on robust non-learning-based controllers for precise manipulation (Yang et al. 2025h).

While this "Language-Perception-Control" framework provides a useful high-level overview, a deeper analysis reveals that the central research questions are not just about these components, but about the *specific functional role language plays in bridging them*. Different approaches leverage language in fundamentally distinct ways to solve the manipulation problem. This survey will focus on this perspective, categorizing recent methods based on how language enters the manipulation control loop.

### 1.1 Contributions

**A New Taxonomy and Comprehensive Review:** We introduce a new taxonomy that organizes the field by the functional role language plays in the manipulation control loop. This taxonomy categorizes methods into four primary themes: language for state evaluation (using language to define goals and assess task-relevant states for high-level planning, such as task decomposition and subgoal sequencing, or for learning, such as reward design, value estimation, and policy optimization), language as a direct policy condition (using language to specify desired behavior), language for cognitive planning and reasoning, and language-driven end-to-end policy (vision-language-action models). Based on this framework, we provide a comprehensive review of state-of-the-art methods, from traditional language-conditioned policies to the latest approaches driven by FMs, including LLMs, VLMs, VLAs, and neuro-symbolic integrations. This structured review highlights how different methods leverage language to bridge perception (grounding language in sensory observations and extracting task-relevant state information), planning/reasoning (inferring task structure, constraints, and subgoals), and control (generating executable actions or policy outputs). Moreover, to provide a multifaceted understanding, we conduct a systematic comparative analysis from an orthogonal perspective. We evaluate the surveyed methods across several key dimensions: action granularity, data and supervision requirements, system cost and latency, the environments and benchmarks used for validation, and cross-modal task specification. This analysis offers practical insights into the trade-offs and applicability of different approaches.

**Discussion and Future Directions:** Additionally, we delve into the key debates, such as whether scaling up VLA models is the most effective path forward compared to incorporating structured world models or other hybrid approaches. We also discuss open challenges currently shaping the field. Building on this discussion, we outline the primary limitations and future directions, focusing on two critical areas: enhancing generalization capability and ensuring real-world safety. To improve generalization, we propose focusing on the development of large-scale, diverse datasets, integrating lifelong learning frameworks to enable continuous adaptation, and establishing methods for cross-embodiment alignment. To address safety, we highlight the need for mechanisms to handle language ambiguity, improve failure recovery, and guarantee the real-time performance required for unstructured environments.

### 1.2 Relation to existing surveys

Before the emergence of LLMs, Tellex et al. (2020) provided a foundational review of language grounding in robotics, categorizing approaches by their technical underpinnings (e.g., "lexically grounded" vs. "learning methods"). Recent surveys, prompted by the rise of FMs, have offered broad perspectives. Surveys by Hu et al. (2023); Li et al. (2024a); Xiao et al. (2025) discuss the application of FMs in robotics, organizing their findings by model type and the specific robotic module they enhance, such as perception and planning. Similarly, Firoozi et al. (2025) structures its analysis around general robotics capabilities like "Perception", "Decision-making", and "Control".

While these surveys provide essential overviews, our work offers a distinct and orthogonal perspective. Rather than categorizing by model type or the robotic module it replaces, our survey organizes the field by the functional role language plays within the manipulation control loop. This taxonomy allows for a finer-grained analysis that spans multiple models and algorithms. We categorize approaches into four types: language for state evaluation, language as a policy condition, language for cognitive planning and reasoning, and language in unified vision-language-action models (VLAs). This taxonomy allows us to systematically cover a wide range of techniques, including language-conditioned policy learning/planning, neuro-symbolic methods, and emerging paradigms based on LLMs, VLMs, and VLAs. By focusing on the role of language, our survey provides a new perspective for understanding the diverse ways we can bridge language and action in robotic manipulation.

### 1.3 Organization

The rest of this paper is organized as follows. Section 2 presents foundational concepts relevant to language-conditioned robot manipulation. In Section 3, we elaborate on the taxonomy of the recent approaches that they can be categorized into *language for state evaluation* (Section 4), *language as policy condition* (Section 5), *language for cognitive planning and reasoning* (Section 6), and *Large-model driven end-to-end policy* (Section 7). Additionally, in Section 8, we conduct a comprehensive comparative analysis of various approaches from a different perspective, focusing on the dimensions of *action granularity*, *data and supervision regime*, *system cost and latency*, as well as *environment and evaluation*. Finally, we present the key debates in this field in Section 9, outline the challenges and future directions in Section 10, and provide conclusions in Section 11.

## 2 Background

We present fundamental terms and concepts in this section. An understanding of these principles is crucial, as they serve as the cornerstone for the more advanced methods discussed throughout this article. Table 1 provides an overview of important abbreviations used in this article.

### 2.1 Markov decision process

A Markov decision process (MDP) is a discrete-time stochastic control model for sequential decision making under uncertainty (Sutton and Barto 1998). Formally, an MDP is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, where $\mathcal{S}$ is the state space and $\mathcal{A}$ is the action space. $\mathcal{P} : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ is the transition probability, where $\mathcal{P}(s' \mid s, a) = \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}$. The reward function $\mathcal{R} : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ gives the expected immediate reward $\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$. The discount factor $\gamma \in [0, 1]$ encodes the trade-off between near-term and long-term rewards and, in continuing tasks, ensures that the (potentially infinite-horizon) return $\sum_{t=0}^{\infty} \gamma^t R_{t+1}$ is finite.

### 2.2 Reinforcement learning

RL (Sutton and Barto 1998) studies how an agent interacts with an environment modeled as an MDP to learn a policy that maximizes expected return *when the environment's dynamics are not available to the agent*. Concretely, the transition probability $\mathcal{P}$ and reward function $\mathcal{R}$ are typically unknown in RL and the agent must improve its behavior from trial-and-error experience. A (stochastic) policy is a mapping $\pi : \mathcal{S} \to \mathcal{A}$ with $\pi(a \mid s)$ denoting the probability of taking action $a$ in state $s$.

The goal of RL is to select a policy $\pi^*$ which maximizes expected return $J(\pi)$ when the agent acts according to it. The expected return can be written as

$$J(\pi) = \mathbb{E}_{a_t \sim \pi(\cdot|s_t), s_{t+1} \sim \mathcal{P}(\cdot|s_t, a_t)} \left[ \sum_t \gamma^t \mathcal{R}(s_t, a_t) \right]. \tag{1}$$

The central optimization problem in RL can be expressed by

$$\pi^* = \arg \max_{\pi} J(\pi), \tag{2}$$

with $\pi^*$ being the optimal policy. RL algorithms typically define value functions

$$V^{\pi}(s) = \mathbb{E}_{a_t \sim \pi(\cdot|s_t), s_{t+1} \sim \mathcal{P}(\cdot|s_t, a_t)} \left[ \sum_t \gamma^t \mathcal{R}(s_t, a_t) \mid s_0 = s \right], \tag{3}$$

and action-value functions

$$Q^{\pi}(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(\cdot|s,a)} \left[ \mathcal{R}(s, a) + \gamma V^{\pi}(s') \right] \tag{4}$$

to guide the search for an optimal policy.

### 2.3 Imitation learning

Imitation Learning (IL) aims to learn a policy $\pi$ by mimicking expert demonstrations, which are sequences of state-action pairs $\tau = \{s_0, a_0, s_1, a_1, \ldots\}$, without relying on explicit reward signals. The main IL methodologies include behavioral cloning (BC), goal-conditioned imitation learning (GCIL), and inverse reinforcement learning (IRL).

#### 2.3.1 Behavioral cloning

In BC (Bain and Sammut 1995), the trajectory executed by expert agents is treated as the reference or ground truth trajectory. Through supervised learning, an imitation policy is acquired by minimizing the disparity between the anticipated actions and the actual actions observed in the ground-truth trajectory. Considering a set of trajectories are collected from experts $\tau \in \mathcal{T}$, the optimization problem can be defined as:

$$\hat{\pi}^* = \arg \min_{\pi} \sum_{\tau \in \mathcal{T}} \sum_{s \in \tau} L(\pi(s), \pi^*(s)). \tag{5}$$

where $L$ is the cost function, $\pi^*(s)$ and $\pi(s)$ are the expert's and predicted actions at the state $s$, respectively.

#### 2.3.2 Goal-conditioned imitation learning

GCIL extends standard imitation learning by conditioning the policy not only on the state but also on a desired goal, enabling agents to generalize across multiple tasks and achieve diverse outcomes from demonstrations (Ding et al. 2019). This extension is essential in robotics, where demonstrations often target different configurations, and a goal-conditioned policy $\pi_\theta(a|s, g)$ can leverage these demonstrations to reach any goal $g \in \mathcal{S}$. In the goal-conditioned setting, the reward function is defined as an indicator of goal achievement, $r(s_t, a_t, s_{t+1}, g) = \mathbb{1}[s_{t+1} == g]$, meaning that the agent succeeds if its next state matches the goal. To learn from demonstrations without explicitly engineering rewards, the most direct approach is goal-conditioned behavioral cloning (Ding et al. 2019), which minimizes the error between the expert's action $a_t^j$ and the agent's predicted action $\pi_\theta(s_t^j, g^j)$ given state-goal pairs. Here, we assume access to $D$ expert demonstration trajectories $\{(s_0^j, a_0^j, s_1^j, \ldots)\}_{j=1}^D \sim \tau_{\text{expert}}$, each produced by an expert policy pursuing a goal $g^j$, with $\{g^j\}_{j=1}^D$ uniformly sampled from the feasible goal space (Ding et al. 2019). The supervised loss is:

$$\mathcal{L}_{\text{BC}}(\theta, D) = \mathbb{E}_{(s_t^j, a_t^j, g^j) \sim D} \left[ \|\pi_\theta(s_t^j, g^j) - a_t^j\|_2^2 \right], \tag{6}$$

This formulation directly adapts standard BC to the goal-conditioned case by embedding goals into the input space of the policy. Beyond simple cloning, GCIL leverages the insight that trajectories labeled with a particular goal $g^j$ can also serve as valid demonstrations for any intermediate state along the trajectory, effectively enabling goal relabeling. This relabeling principle, analogous to hindsight experience replay (HER) (Andrychowicz et al. 2017) in RL, substantially improves sample efficiency and generalization in sparse reward settings.

#### 2.3.3 Inverse reinforcement learning

An alternative to direct BC in IL is to reason about and recover the hidden reward function that drives expert behavior. This approach, known as IRL (Arora and Doshi 2021), seeks to infer a reward function $R(s, a)$ from demonstrations rather than directly copying actions, thereby capturing the intent behind behavior and enabling agents to generalize or even surpass the expert. Formally, given demonstrations represented as trajectories $\tau = \{(s_0, a_0), (s_1, a_1), \ldots\}$, the objective of IRL is to find a reward function under which the expert's policy is (approximately) optimal:

$$\pi_E = \arg \max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid \pi \right]. \tag{7}$$

To address the ill-posed nature of IRL, two representative techniques are widely used. Apprenticeship Learning (Abbeel and Ng 2004) avoids recovering a unique reward and instead matches the expert's and learner's feature expectations. Assuming a linear reward form $R(s, a) = w^\top \phi(s, a)$, where $w$ is a weight vector and $\phi(s, a)$ is a feature representation of state-action pairs, it seeks a policy $\pi$ such that

$$\mu_\phi(\pi) \approx \mu_\phi(\pi_E), \quad \mu_\phi(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \phi(s_t, a_t) \right]. \tag{8}$$

In contrast, Maximum Entropy IRL (Ziebart et al. 2008) resolves ambiguity by modeling expert demonstrations as samples from a maximum entropy trajectory distribution

$$P(\tau) = \frac{1}{Z} \exp\left(\sum_t R(s_t, a_t)\right), \tag{9}$$

where $Z$ is a normalization term. These two approaches capture the main routes in IRL: feature-matching to approximate expert performance and probabilistic modeling to handle reward uncertainty, both of which extend imitation learning beyond BC.

### 2.4 Diffusion model-based policy learning

Denoising Diffusion Probabilistic Models (DDPMs) (Ho et al. 2020) are a class of generative models that define a latent-variable Markov chain to progressively denoise a sample drawn from Gaussian noise into a data sample. The model learns to reverse a fixed forward process that gradually adds Gaussian noise to data according to a variance schedule. Formally, the reverse (generative) process is defined as

$$p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t), \tag{10}$$

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)). \tag{11}$$

Here, $\mu_\theta(x_t, t)$ and $\Sigma_\theta(x_t, t)$ are the mean and variance predicted by a neural network parameterized by $\theta$, which attempt to approximate the true posterior of the forward process. The forward diffusion process is defined as

$$q(x_{1:T} \mid x_0) := \prod_{t=1}^{T} q(x_t \mid x_{t-1}), \tag{12}$$

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I\right), \tag{13}$$

where $\beta_t \in (0, 1)$ is a variance schedule that determines the amount of Gaussian noise injected at step $t$. This process admits a closed-form expression for sampling at step $t$:

$$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I\right), \tag{14}$$

$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s). \tag{15}$$

Here, $\alpha_t = 1 - \beta_t$ represents the retained signal at each step, and the cumulative product $\bar{\alpha}_t$ measures how much of the original data signal $x_0$ remains after $t$ steps of noise addition.

Building on this foundation, recent works in robotics propose Diffusion Policy (DP) (Chi et al. 2023), which represents visuomotor control as a conditional denoising diffusion process in action space. Instead of directly regressing robot actions, the policy refines Gaussian noise into an action sequence by learning the score function of the conditional distribution $p(\mathbf{A}_t \mid \mathbf{O}_t)$, where $\mathbf{O}_t$ are observations and $\mathbf{A}_t$ are action trajectories. This formulation leverages the advantages of diffusion, such as naturally modeling multimodal distributions, scaling to high-dimensional action sequences, and exhibiting stable training. These properties make it highly effective for robot manipulation tasks. The action generation in diffusion policy follows an iterative denoising process akin to Langevin dynamics (Welling and Teh 2011). At inference, starting from a noisy action sequence $\mathbf{A}_t^K$, the denoising step is given by

$$\mathbf{A}_t^{k-1} = \alpha\left(\mathbf{A}_t^k - \gamma \varepsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^k, k)\right) + \mathcal{N}(0, \sigma^2 I), \tag{16}$$

where $\varepsilon_\theta$ is a neural network that predicts noise conditioned on $\mathbf{O}_t$, $\gamma$ is the learning rate, $k$ is the iteration step, and $\alpha$ and $\sigma$ follow a noise schedule. Training minimizes the mean-squared error between predicted and true noise:

$$L = \mathbb{E}_{k, \varepsilon} \left[ \left\| \varepsilon - \varepsilon_\theta\left(\mathbf{O}_t, \mathbf{A}_t^0 + \varepsilon, k\right) \right\|_2^2 \right]. \tag{17}$$

This process allows the model to iteratively refine noise into structured action sequences that are both temporally consistent and reactive. As a result, diffusion-based approaches have emerged as a powerful paradigm for trajectory generation and visuomotor policy learning in robotics, bridging generative modeling and control.

### 2.5 Knowledge base & Knowledge graph

A knowledge base (KB) serves as a foundational repository of structured knowledge, encapsulating facts, rules, and relationships pertinent to a given domain. Information is organized in a structured format in a knowledge base, facilitating efficient retrieval and inference. Formally, knowledge bases often employ languages such as Resource Description Framework (RDF) or Web Ontology Language (OWL) to encode knowledge in a machine-readable form.

A Knowledge Graph (KG) can be considered as a specialized form of KB with a graph structure. Formally, a Knowledge Graph (KG) is defined as $\mathcal{G} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$ over a set $\mathcal{E}$ of entities and a set $\mathcal{R}$ of predicates. In the field of robot manipulation, entities often contain real-world objects, actions, skills, and other abstract concepts; relations represent the relations between these entities, such as *isComponentOf*, *withForce*, and *hasPose*. Several well-known knowledge graphs (KGs) have been developed in this field, as highlighted in various works (Waibel et al. 2011; Diab et al. 2019; Beetz et al. 2018; Kwak et al. 2022).

Knowledge Graph Embedding (KGE) methods represent the information from a KG as dense embeddings. Entities and predicates in the KG are mapped into a $d$-dimensional vector $M_\theta : \mathcal{E} \cup \mathcal{R} \to \mathbb{R}^d$. These methods can be either score function based (Bordes et al. 2013; Wang et al. 2014; Lin et al. 2015) or graph neural network based (Schlichtkrull et al. 2018; Nathani et al. 2019).

### 2.6 Language and Multimodal Fusion

Language-conditioned robot manipulation inherently requires integrating linguistic instructions with perceptual observations and robot states (Belkhale et al. 2024). This process relies on multimodal fusion, aiming to learn joint representations that align and combine information from heterogeneous modalities such as vision and language. In general multimodal learning, fusion strategies are often categorized according to how interactions between modalities are modeled during representation learning (Baltrusaitis et al. 2018). Three representative paradigms have emerged in recent vision-language multimodal frameworks. The first paradigm adopts *dual-stream architectures* in which visual and textual features are encoded by separate networks and interact through cross-modal attention mechanisms, enabling fine-grained alignment between modalities, as demonstrated in models such as ViLBERT (Lu et al. 2019). The second paradigm employs *single-stream architectures*, where tokens from different modalities are projected into a shared embedding space and processed by a unified transformer encoder, allowing deeper cross-modal interactions and joint reasoning (Chen et al. 2020b). The third paradigm focuses on *contrastive alignment*, which learns a shared embedding space by maximizing the similarity between paired vision and language representations, a strategy popularized by models such as CLIP (Radford et al. 2021). These multimodal fusion mechanisms provide the foundation for modern vision-language models and vision-language-action models, which ground language instructions in visual observations and enable robots to translate high-level semantic commands into executable manipulation behaviors.

### 2.7 Language model

Language models are designed to estimate the likelihood of sequences of words in human language. They learn patterns, structures, and semantics from vast amounts of textual data, enabling them to understand and predict language usage. In recent years, neural language models have advanced significantly, allowing for generating coherent and contextually relevant text. We provide a brief history of neural language models.

**Neural Language/Lexical Models (NLMs):** NLMs (Bengio et al. 2003; Mikolov et al. 2010; Kombrink et al. 2011) leverage neural networks to estimate the probability of word sequences, e.g., recurrent neural networks (RNNs), offering a more powerful alternative to traditional statistical methods. Collobert et al. (2011) make a remarkable contribution by developing a unified neural network approach capable of handling various Natural Language Processing (NLP) tasks within a single framework, demonstrating the versatility of neural models in language understanding. Furthermore, word2vec (Mikolov et al. 2013b,a) revolutionizes word representations employing a simple yet efficient shallow neural network to learn distributed word embeddings. These embeddings have proven to be highly effective across a wide range of NLP tasks and have been instrumental in advancing applications like language-conditioned robot manipulation by enhancing semantic understanding.

**Pre-trained Language Models (PLMs):** PLMs are an early attempt to extract semantic meaning from natural language. ELMo (Peters et al. 2018) aims to capture context-aware word representations by first pre-training a bidirectional LSTM (biLSTM) network and then fine-tuning the biLSTM network according to specific downstream tasks. Moreover, drawing inspiration from the Transformer architecture (Vaswani et al. 2017) and incorporating self-attention mechanisms, BERT (Devlin et al. 2019) takes language model pre-training a step further. It accomplishes this by conducting bidirectional pre-training exercises on extensive unlabeled text corpora. These specially crafted pre-training tasks imbue BERT with contextual understanding.

**Large Language Models:** LLMs become popular as scaling of PLMs leads to improved performance on downstream tasks. Many researchers (Hoffmann et al. 2022) attempt to study the performance limit of PLMs by scaling the size of models and datasets, e.g., the comparatively small 1.5B parameter GPT-2 (Radford et al. 2019) versus larger 175B GPT-3 (Brown et al. 2020) and 540B PaLM (Chowdhery et al. 2023). Although these models share similar architectures, larger models exhibit enhanced capabilities such as few-shot learning, in-context learning, and improved performance on language understanding and generation benchmarks. Additionally, models like ChatGPT have adapted the GPT family for dialogue by incorporating techniques such as instruction tuning and reinforcement learning from human feedback (Ouyang et al. 2022). This results in more coherent and contextually appropriate conversational abilities, which are crucial for interactive robot manipulation tasks where ongoing dialogue between the human and robot can enhance task execution and adaptability. By integrating LLMs into robotic systems, researchers (Ding et al. 2022; Kant et al. 2022) aim to leverage their advanced language understanding to enable robots to handle challenging tasks by reasoning and inferring missing information. This contributes to developing more robust and flexible manipulators that can operate effectively in unstructured real-world environments (Lin et al. 2023; Ren et al. 2023a; Wu et al. 2023a).

**Vision-Language Models:** VLMs play a key role in extracting and combining visual and textual content from large-scale web data. This paradigm equips the agent with an expansive understanding of the world. Seminal models in this domain, such as CLIP (Radford et al. 2021), Flamingo (Alayrac et al. 2022), PaLM-E (Driess et al. 2023), and PaLI-X (Chen et al. 2023d) underscore the potential of VLMs. Integrating these pre-trained VLMs into robotic systems makes it feasible for robots to tackle diverse tasks in real-world scenarios.

**Vision-Language-Action Models:** VLAs unify vision, language, and action within a single policy but differ in how they represent and generate actions. Early systems cast actions as discrete tokens and learn via autoregressive next-token prediction, analogous to language modeling, enabling joint training over text and action sequences (e.g., Gato (Reed et al. 2022), RT-1 (Brohan et al. 2022), and RT-2 (Zitkovich et al. 2023)). Recent VLAs also adopt continuous regression and diffusion-based action generation over trajectories or low-level controls, directly predicting joint velocities or end-effector poses (Kim et al. 2025c; Black et al. 2025b; Wen et al. 2025c).

## 3 Taxonomy of language-conditioned methods for robotic manipulation

Over the past few years, there has been a growing interest in leveraging natural language to enhance robotic manipulation tasks, particularly in making these systems more intuitive and accessible during interactions. To facilitate a clear review of these approaches, this section outlines the taxonomy employed in the narration. While robotic learning in bridging language instructions and robot actions is often categorized by the algorithmic paradigm, such as RL, IL, or planning, this can obscure the specific and varied roles that language plays. For instance, an RL agent might use language to shape its reward function or, in a completely different manner, to directly condition its policy learning. Although both involve RL methods, the function of language is fundamentally different. To provide a clearer, more orthogonal taxonomy, we structure this survey around the primary ways language is integrated into robot systems.

As illustrated in the fine-grained taxonomy tree in Figure 3, our classification logic progresses from macro-level functional roles down to specific algorithmic implementations and evolutionary process. The four primary categories and their underlying classification logic are detailed below:

- **Language for state evaluation (Section 4):** Language is used to define goals and quantify task progress, by converting text into numerical feedback signals (reward or cost). The policy is then optimized against these signals. The core question is: How can language specify whether a state or outcome is desirable (feedback *how the progress*)? This category is subdivided based on the type of algorithm receiving the signal and the technological evolution of the signal generation:
  - **Reward Functions:** For learning-based agents (RL), language is translated into rewards. We trace the logical evolution of this process from manual Reward Designing (sparse vs. dense), to data-driven Reward Learning (like Inverse RL), and finally to modern Foundation Model (FM)-driven automated reward generation.
  - **Cost Functions:** For optimization-based motion planners, language is translated into cost maps. The narrative trajectory moves from specific linguistic-to-cost mappings to automated 3D cost map generation empowered by FMs.

- **Language as a policy condition (Section 5):** Language is used as an explicit conditioning signal for a policy that maps observations to actions. The primary learning target is a control policy, and language specifies the desired behavior for the current episode or timestep. Although such methods may use multimodal encoders, language remains an external task condition rather than the organizing principle of the whole architecture. The core question is: How can language condition a policy to produce the correct behavior? This category is subdivided by the underlying algorithmic paradigms used to learn this language-conditioned mapping:
  - **Reinforcement Learning:** Integrates language to solve trial-and-error exploration, evolving from simple goal-conditioning to lifelong multi-task learning.
  - **Behavioral Cloning:** Learns directly from expert demonstrations to bypass the sample inefficiency and complex reward engineering inherent in RL.
  - **Diffusion-based Policy:** Represents the latest generative evolution to address BC's limitation in handling multi-modal action distributions (traditional BC averages out expert behaviors).

- **Language for cognitive planning and reasoning (Section 6):** Language serves as an internal reasoning medium for decomposing complex tasks and forming strategies. Core question: How a robot "thinks" in the language space to structure its own behavior, i.e., utilizing language to reason about its goals and plan its actions. This branch is subdivided based on the type of cognitive system used to bridge abstract reasoning and perceptual reality:
  - **Classic Neuro-symbolic approaches:** The foundational methods that use language to bridge explicit symbolic logic (like Knowledge Graphs) with neural perception.
  - **Empowered by Large Language Models:** Approaches that replace rigid symbolic systems with LLMs for open-vocabulary text reasoning and code generation, often without task-specific retraining for the planning module.
  - **Empowered by Vision-Language Models:** Methods that resolve the "blindness" of pure LLMs by directly grounding textual reasoning in visual observations.

- **Language in unified vision-language-action models (Section 7):** Language is not used merely as an external condition; instead, it is jointly modeled with visual observations and robot actions within a single unified backbone - VLA systems. In these systems, perception, semantic grounding, and action generation are increasingly learned within one embodied foundation model, often by extending pretrained VLMs/LLMs with robot action representations. The core question is: How can vision (perception), language (semantic grounding), and action be unified into a single scalable model for embodied decision-making? This branch is subdivided based on different optimization directions for bridging language and action:
  - **Perception:** Methods that focus on optimizing how VLAs perceive and understand their environment.
  - **Reasoning:** Approaches that enhance the model's internal "thought process", that is, how it forms plans, leverages prior knowledge, and predicts outcomes to solve complex tasks.
  - **Action:** Methods that focus on the optimization regarding the "output" stage of the policy, concerning the form and mechanism of the robot's actions. This bridges the gap between the model's internal plan and its physical embodiment.
  - **Learning & Adaptation:** Techniques for efficiently training, fine-tuning VLA models to enhance their adaptability to new situations and downstream tasks.

Each subsequent section explores the advanced techniques developed within these sub-branches, highlighting their unique contributions, chronological evolution, and the specific challenges they aim to resolve.

## 4 Language for state evaluation

Traditionally, specifying a robot's objective has required significant expert engineering, such as hand-crafting dense reward functions (Eschmann 2021) or defining precise goal coordinates in the state space (La Valle 2011). This process is not only labor-intensive but also rigid, and the robot cannot easily generalize to new goals without being reprogrammed. Natural language provides a powerful solution to this limitation. It offers a flexible, intuitive, and generalizable interface, enabling non-experts to communicate a vast range of complex, abstract, or compositional goals in multi-task scenarios without programming efforts (Tellex et al. 2020), such as from "*pick up the red block*" to "*tidy up the table*". This shifts the paradigm from low-level programming to high-level human-centric goal specification. This brings us to the first key research question: How can we use language to quantify task progress? The core idea is to translate a language instruction into a quantitative scoring function that evaluates how well a robot's state or action aligns with the desired outcome, guiding the robot towards desired behaviors efficiently. This numerical signal, which can serve as a reward for RL agents or a cost for planners, provides the essential feedback for the robot to learn or plan effectively. Grounding language in state-space valuations is a central challenge and can be implemented in two primary ways.

- Language-conditioned reward functions: In RL, the language-derived score serves as a reward function. It guides the agent's trial-and-error learning, reinforcing behaviors that bring it closer to fulfilling the instruction. These methods can be further categorized by the numerical properties of the reward signal, such as dense rewards, sparse rewards, or fully learned reward functions, as illustrated in Figure 4.

- Language-conditioned cost functions: In task and motion planning, the score serves as a cost function. It guides a search algorithm to find an optimal sequence of actions that minimizes the cost, thereby achieving the goal specified in the language command. For example, for command "*Pick up the apple, but stay away from the vase*", the cost function would assign a high penalty to any trajectory that nears the vase. A motion planner would then search for a path that minimizes this total cost, resulting in a trajectory that safely navigates around the vase to reach the apple.

This section reviews the methods developed to address this problem, including Language-conditioned reward designing/learning in Sec. 4.1 and Language-conditioned cost functions in Sec. 4.2. The taxonomy is illustrated in Figure 5. In addition, to provide a clearer overview of how language is utilized to quantify task progress, Table 2 summarizes representative state-of-the-art methods discussed in this section.

### 4.1 Language-conditioned reward designing/learning

In many RL scenarios, particularly for complex manipulation tasks, agents often learn from sparse rewards, which provide a positive signal only upon task completion (Andrychowicz et al. 2017; Riedmiller et al. 2018; Bing et al. 2023b). This is because defining a continuous measure of progress for abstract or contact-rich tasks like "folding a shirt" (Jangir et al. 2020) or "inserting a key" (Ocana et al. 2023) is difficult, as their success is often binary and depends on a complex combination of factors that are hard to quantify. This approach is sample-inefficient, as the agent may struggle to discover the goal through random exploration, resulting in long learning time. On the contrary, dense rewards, such as the distance to a target, provide stronger learning signals but are difficult to specify, and often require intensive manual engineering and domain expertise for each new task (Sutton and Barto 1998). A common technique to accelerate learning is reward shaping (Ng et al. 1999), which provides the agent with additional, intermediate rewards to guide it toward the goal. However, designing these shaping functions can also be a challenging and time-consuming process in multitask scenarios (Yu et al. 2020). Using natural language instructions offers a more intuitive solution, instead of requiring an expert to engineer a complex function, anyone can provide simple instructions, like "*Jump over the skull while going to the left*", to specify desired behaviors (Goyal et al. 2019). These instructions can then be translated into intermediate language-based rewards by a pre-trained language-action matching model, guiding the agent's exploration and accelerate learning. Language-conditioned reward designing/learning approaches make it convenient for non-experts to teach new skills for RL agents.

#### 4.1.1 Language-based reward signal designing

Language provides a natural and expressive medium for designing reward signals that guide robot learning. Instead of manually engineering complex reward functions, language allows users to specify desired behaviors, goals, or preferences intuitively. These signals can be broadly categorized into two types: *sparse reward* and *dense reward*, as illustrated in Figure 4. Each type presents a distinct trade-off between design simplicity and learning efficiency. A *sparse reward* provides a meaningful signal only upon task completion (e.g., a single positive reward for success), which is simple to define but often leads to sample-inefficient learning. In contrast, a *dense reward* offers continuous feedback at each step, guiding the agent more effectively but typically requiring significant manual engineering.

For the former one, language can be used to generate a sparse reward signal to indicate whether the agent's current state successfully fulfills the language instruction, providing a flexible way to specify complex goals or human preferences. For example, ZSRM (Mahmoudieh et al. 2022) derives the entire reinforcement learning reward from a natural language goal description. At each step, it generates a reward by calculating the similarity between a camera image and the goal text using CLIP encoders. This text-vision matching method enables reward specification for some newly described goals without retraining the reward model, although transfer degrades on tasks that require stronger spatial reasoning from the CLIP model.

To address the sample inefficiency of sparse rewards, another line of work uses language to create dense, intermediate reward signals that provide more continuous guidance. These approaches typically learn a model that scores how well an ongoing trajectory aligns with a language command, using this score to shape the reward at every timestep. An example is PixL2R (Goyal et al. 2021), which maps natural language and pixels directly to a continuous reward signal. It first learns a relatedness model from paired trajectory and language data. During policy training, this model evaluates the agent's trajectory and turns the language command into a potential function (Ng et al. 1999). This potential generates a dense shaping reward that is added to the environment's original reward at every step, significantly improving policy learning efficiency.

#### 4.1.2 Language-based reward function learning

The above reward designing approaches highlight the importance of designing effective reward signals with language instructions, while requiring extensive manual effort and are often challenging in real-world scenarios, limiting real-world manipulation performance (Deshpande et al. 2025). Reward function learning aims to learn a reward function from data (e.g., demonstrations or human video (Das et al. 2021; Zakka et al. 2022a)), rather than manually designing it. This flexible and scalable manner enables the reward models can adapt to multi-task (Dimitrakakis and Rothkopf 2011) or cross-embodiments (Zakka et al. 2022a; Kumar et al. 2023) scenarios without requiring expert knowledge. In language-conditioned manipulation, reward learning methods typically learn a function that maps observations and a language instruction to a scalar reward value, which can then be used to guide a standard RL agent.

IRL is utilized to deduce the underlying reward structure, alleviating the challenge of manual reward design. It infers the rewards that could explain the observed actions of an expert. Rather than simply learning from experts, IRL has the potential to achieve better performance than the expert by optimizing the inferred reward function (Finn et al. 2016). Integrating language instructions into IRL learn reward functions that are not only consistent with the expert's actions but also grounded in the specific textual command, showcasing promising performance in various tasks, such as navigation (Zhou and Small 2021) and mobile manipulation tasks (Fu et al. 2019).

#### 4.1.3 Foundation model-driven reward designing and learning

The limitations of aforementioned traditional reward models, particularly their reliance on task-specific datasets and rigid training objectives, have motivated a shift towards more flexible and generalizable solutions. Foundation models (such as LLMs and VLMs), with their broad pre-trained knowledge and powerful open-vocabulary inference abilities (Hurst et al. 2024; Yang et al. 2025a), provide a powerful alternative. Instead of learning a reward function from scratch on a limited dataset, these models can leverage their inherent understanding of language and vision in foundation models to directly infer task progress, thereby automating the reward design process and overcoming the need for extensive manual engineering or data collection.

**LLM-driven reward code generation:** Early works leveraging FMs for reward generation explored several distinct strategies. One initial approach repurposed LLMs into reward functions. For example, Kwon et al. (2023) reframe GPT-3 itself as a proxy reward function, prompting it with a trajectory transcript and asking whether the behavior satisfied a textual objective. Language2Reward (Yu et al. 2023c) pushes further by asking an LLM to write executable reward code from natural-language instructions. Text2Reward (Xie et al. 2024) lets GPT-4 and Codex write dense reward functions directly from a natural-language goal with a Pythonic environment sketch. EUREKA (Ma et al. 2024) generates a high-performance reward function in an evolutionary manner using RL policy performance as a fitness score. Methods like Reward-Self-Align (Zeng et al. 2024) and R* (Li et al. 2025c) move beyond just code generation to parameter optimization.

**VLM-driven reward learning:** A drawback of LLM-driven code-centric methods is their dependence on privileged simulator state information. To operate in the real world using visual input, the focus shifted to VLMs. Video-Language Critic (Alakuijala et al. 2025) trains a temporal contrastive VLM with ranking loss on large "Open X-Embodiment" videos. RealBEF (Wang et al. 2025c) fine-tunes the smaller VLM ALBEF backbone with pair-wise image comparisons. ReWiND (Zhang et al. 2025c) learns a progress-predicting reward from a handful of language-annotated demonstrations combined with "video-rewind" generated failures.

In summary, the integration of foundation models into reward designing and learning represents an advancement in language-conditioned robot manipulation. Early FM-driven methods showed that LLMs can replace manual reward engineering, but exposed the weight-tuning fragility of the generated reward code. VLM-based methods further bridge the sim-to-real gap by learning dense rewards directly from pixels and language. Hybrid approaches decrease demonstration cost and mitigate the sim-to-real gap, thereby showcasing a fully autonomous reward design.

### 4.2 Language-conditioned cost functions

While reward functions guide learning-based agents, cost functions are essential for optimization-based motion planners. Translating language into a cost function enables a robot to understand not only a goal, but also the constraints and preferences that indicate how to achieve it, turning natural language into a formal optimization objective.

#### 4.2.1 Specific linguistic-text cost mapping

Early motion-planning pipelines couldn't directly interpret a human's verbal goals, requiring hand-coded cost terms or exact goal configurations (Yamashita et al. 2003; Cambon et al. 2009). Park et al. (2019) introduce dynamic constraint mapping, where a Conditional Random Field grounds each noun phrase or adverb ("upright" and "slowly") into the continuous parameters of a trajectory optimizer. Sharma et al. (2022) shift the supervision burden to the user by learning a residual cost function from spoken corrections.

#### 4.2.2 FM-driven cost mapping

To tackle 3D complexity, VoxPoser (Huang et al. 2023b) harnesses foundation models. It uses GPT-4 to write Python code that queries a VLM (OWL-ViT) to compose dense 3D value maps. ReKep (Huang et al. 2025b) alternatively uses a VLM to write Python code defining symbolic constraints instead of creating a value map. LACO (Xie et al. 2023a) learns a collision function directly. IMPACT (Ling et al. 2025) leverages the advanced semantic and spatial reasoning capabilities of VLMs (GPT-4o) to automate the inference of "acceptable contact".

### 4.3 Summary

In its role as a tool for state evaluation, language provides a flexible and intuitive interface for specifying a robot's objectives, translating high-level instructions into quantitative scoring functions that guide robot behavior. This paradigm addresses our first core question by grounding language into reward functions for reinforcement learning or cost functions for motion planning. The overarching insight is that the role of language in state evaluation has evolved from a tool for manual reward/cost specification to a medium for automated, knowledge-driven reward and cost generation, powered by FMs' reasoning capabilities. This progression has enhanced the scalability, flexibility, and data efficiency of teaching robots complex manipulation tasks.

## 5 Language as a policy condition

While the previous section focused on using language to quantify the task progress for implicitly directing the robot's behaviors. This section shifts to an alternative paradigm for clarifying the role of the language in robotic manipulation: using language as an explicit condition for policy learning to specify *how* a robot should act. Instead of translating language into a reward or cost function that indirectly guides a learning or planning algorithm, the methods discussed here integrate language into the policy itself. This approach addresses our second key research question: How can language condition a policy to produce the correct behavior? Here, the policy $\pi_\theta$, parameterized by the neural network $\theta$, learns a direct mapping from the current observation $s_t$ and the language instruction $l$ to an action $a_t$, i.e., $\pi_\theta(a_t|s_t, l)$. This changes the role of language from a goal specifier to a behavior specifier. Figure 6 presents the taxonomy of this section.

### 5.1 Language in reinforcement learning

RL algorithms can be applied to tasks based on language-conditioned rewards. Early attempts at language-conditioned RL concentrated on games (Fu et al. 2019; MacGlashan et al. 2015; Bahdanau et al. 2018; Kaplan et al. 2017; Goyal et al. 2019). An early solution was to decouple language from control. LCGG (Colas et al. 2020) uses language to condition a goal generator. LanCon-Learn (Silva et al. 2021) achieves tighter integration by directly feeding a language embedding into an attention router that gates multiple skill modules. MILLION (Bing et al. 2023a) introduces a memory-based meta-learning approach. TALAR (Pang et al. 2023) proposes translating free-form text into a compact and discrete set of Task-Language predicates. LEGION (Meng et al. 2025a) clusters language embeddings online and uses the resulting cluster ID to gate reusable skills. FLaRe (Hu et al. 2025a) mitigates catastrophic forgetting by fine-tuning a large pre-trained behavioral cloning policy with PPO under sparse linguistic rewards. V-GPS (Nakamoto et al. 2025) learns a language-conditioned value function via offline RL. LIMT (Aljalbout et al. 2025) combines language with world-model imagination.

In summary, using language as a direct policy condition in RL transforms it from a goal specifier to a behavior specifier. Early approaches that decoupled language from control were modular but could not capture fine behaviors. This leads to end-to-end methods that integrate language directly into the policy loop. A powerful paradigm has emerged: fine-tuning large pre-trained behavioral cloning policies with sparse linguistic rewards, and steering the agent's behavior through a learned language-conditioned value function.

### 5.2 Language in behavioral cloning

Given the challenges of sample inefficiency and complex reward engineering in RL, an alternative paradigm is to learn policies directly from expert demonstrations. A direct approach is to treat language instruction $l$ as a conditional input to the imitation learning policy, such that $\pi_\theta(a_t|s_t, l)$. A mainstream strategy for implementing this is to adapt goal-conditioned imitation learning frameworks, where language instructions serve as the goal.

**Language as goals in goal-conditioned IL:** GCIL provides a natural foundation for language-conditioned IL. Early approaches tackled this by using task-focused visual attention mechanisms. CLIPORT (Shridhar et al. 2022) combines the broad semantic understanding of CLIP with the spatial precision of Transporter. Multi-context imitation learning (MCIL) (Lynch and Sermanet 2021) demonstrates learning effectively even when less than 1% of demonstration data is language-annotated. Motivated by architectural bottlenecks, a second wave of work replaces task-specific backbones with fully-attentional transformers. HiveFormer (Guhur et al. 2023) concatenates language tokens with all past visual-proprioceptive tokens into a single sequence. PerAct (Shridhar et al. 2023) frames the problem as "detecting the next best voxel action" by voxelizing both the RGB-D observations and the 6-DoF action space. Act3D (Gervet et al. 2023) tackles the computational cost using a coarse-to-fine attention mechanism. GNFactor (Ze et al. 2023) distills pre-trained 2D VLM features into a generalizable neural feature field. RVT (Goyal et al. 2023) re-renders the point cloud from virtual cameras and applies multi-view ViT. HULC (Mees et al. 2022a) introduces a transformer-based architecture with contrastive learning for long-horizon manipulations. ACT (Zhao et al. 2023a) introduces action chunking. SRT-H (Kim et al. 2025a) introduces a hierarchical framework for autonomous surgery.

### 5.3 Language in diffusion-based policy learning

While traditional imitation learning methods like BC have been successful, they may struggle with tasks that exhibit multi-modal action distributions. To overcome this limitation, generative models have been introduced, and diffusion models have emerged as a powerful and stable alternative. By incorporating language instructions as a condition, these models become language-conditioned diffusion policies. StructDiffusion (Liu et al. 2023c) introduced language as a high-level constraint to guide an object-centric language-conditioned diffusion model. ChainedDiffuser (Xian et al. 2023) utilizes language to guide a global transformer to predict discrete keyposes. LCD (Zhang et al. 2024a) tackles long-horizon tasks by employing a diffusion model as a high-level planner. PlayFusion (Chen et al. 2023b) introduces a skill discovery scheme. PoCo (Wang et al. 2024c) formulates policy composition as a conditional diffusion problem. RoLD (Tan et al. 2024) pre-trains a task-agnostic autoencoder to create a unified latent action space. GR-MG (Li et al. 2025d) uses the language instruction to generate a corresponding goal image with a diffusion model.

### 5.4 Summary

When used as a policy condition, language shifts from quantifying "task progress" to dictating "how to do it". This approach has been explored across RL, BC, and DP, each offering distinct advantages and challenges. Across all three learning paradigms, there is a clear progression from simple feature conditioning toward more deeply integrated roles for language. A key remaining challenge, particularly for diffusion policies, is their inference latency, which can be a bottleneck for real-time control.

## 6 Language for cognitive planning and reasoning

The previous sections explore how language can quantify task progress (state evaluation in Sec. 4) or *how to do* it (policy conditioning in Sec. 5). In both paradigms, language serves as an external instruction that guides the robot's low-level behavior. This section explores a more cognitive role for language, utilizing it as an internal tool for reasoning and planning. We now turn to our third key research question: *How can a robot "think" in language to structure its own behavior?* This involves leveraging language not just to follow commands, but to reason about the world, decompose complex problems reasonably, and formulate strategies, enabling more autonomous and intelligent manipulation in real-world environments. Figure 9 displays the taxonomy of this section.

### 6.1 Classic neuro-symbolic approaches

Neuro-symbolic artificial intelligence (Besold et al. 2021), combining both neural and symbolic traditions to solve tasks, continually develops alongside popular data-driven machine-learning approaches. This section discusses key neuro-symbolic methods in language-conditioned manipulation, categorized according to the framework by Yu et al. (2023a), namely *Learning for reasoning*, *Reasoning for learning*, and *Learning-reasoning*.

#### 6.1.1 Learning for reasoning

In this category, neural networks serve as perception and feature extraction modules, converting unstructured data into symbolic representations. A separate symbolic system then uses these symbols to perform high-level reasoning and planning. An early system proposed by Tenorth et al. (2010) sourced task knowledge from websites like wikiHow. She et al. (2014) present a framework where a robotic arm learns new high-level actions through natural language. HiTUT (Zhang and Chai 2021) uses a unified transformer architecture to learn a hierarchical task structure. DANLI (Zhang et al. 2022) operates on a task-oriented dialogue history between a human commander and the robot.

#### 6.1.2 Reasoning for learning

*Reasoning for Learning* indicates that the symbolic system provides structure, rules, or knowledge that constrains and guides a neural network's learning process or decision-making. Misra et al. (2016) propose a model that grounds free-form natural language instructions in a robot's environment. Nguyen et al. (2019) integrate the prior knowledge in KG. Silver et al. (2023) learn neuro-symbolic skills from demonstrations without direct language commands during training.

#### 6.1.3 Learning-reasoning approaches

This category includes integrated systems where neural and symbolic components work in a tight loop, mutually informing and enhancing each other. Chai et al. (2018) introduce an Interactive Task Learning framework. K et al. (2023) propose a system that translates natural language instructions into executable programs. Miao et al. (2023) use a vision module to generate a scene graph utilized as input to a regression planning network.

While these classic neuro-symbolic frameworks provide interpretable and structured reasoning for robotic manipulation, they also face several limitations including: labor-intensive KG construction, manual symbol engineering and ontology drift, limited coverage of commonsense and real-world knowledge, and scaling issues in long-horizon tasks.

### 6.2 Empowered by large language models

The limitations of classic neuro-symbolic methods stem from the fact that classic pipelines rely on explicitly engineered symbolic knowledge. Recent foundation models, particularly LLMs, offer a complementary path. Trained on trillions of tokens of numerous unstructured data, LLMs encode broad textual priors and often exhibit useful instruction-following, commonsense understanding, and contextual reasoning without domain-specific retraining. This capability allows LLMs to serve as a powerful tool that combines planner and reasoner for language-conditioned robots.

While LLMs address many of the scalability issues of classic methods, they introduce their own potential and non-negligible issues: the grounding problem, ambiguity in language, and lack of feedback and reactivity.

#### 6.2.1 Planning

**Open-loop planning:** Many researchers have designed LLM-based planners that integrate LLMs with different external components. Saycan (Ahn et al. 2022) translates natural language instructions into intermediate action sequences. Ren et al. (2023a) combine LLMs with conformal prediction to measure and align uncertainty. Huang et al. (2022a) let two pre-trained LLMs play different roles in task planning.

**Closed-loop planning:** More recent studies integrate LLMs with external components in a closed loop. Sayplan (Rana et al. 2023) operates in a larger-scale environment leveraging a hierarchical 3D scene graph. Lin et al. (2023) propose Text2Motion to leverage Sequencing Task-Agnostic Policies as geometric feasibility approaches. Wu et al. (2025b) propose SELP to mitigate hallucination issues of LLMs for long-horizon tasks incorporating Linear Temporal Logic formulation.

#### 6.2.2 Reasoning

**Summarization:** Summarization of LLMs shows the potential of embodied agents in the household scenario. Housekeep utilizes a large-scale dataset of human preferences (Kant et al. 2022). Tidybot (Wu et al. 2023a) reasons individual preference by few-shot prompting and summarizes a general strategy.

**Eliciting reasoning via prompt engineering:** The chain-of-thought (CoT) (Wei et al. 2022) decomposes a problem into a set of subproblems and solves them sequentially. Socratic models employ multimodal-informed prompting (Zeng et al. 2023). ECoT incorporates CoT into VLAs (Zawalski et al. 2025; Chen et al. 2025e).

**Code-generation:** LLMs generate Pythonic code for agents to perform tasks. Liang et al. (2023) utilize the prompting hierarchical code-gen approach. VOYAGER (Wang et al. 2024a) can code a new skill using skill retrieval. DAHLIA (Meng et al. 2025c) introduces a dual-tunnel "planner + reporter" loop. LYRA (Meng et al. 2025b) integrates a human-in-the-loop for lifelong skill acquisition.

**Iterative reasoning:** Inner Monologue (Huang et al. 2022b) uses feedback from a dedicated success detector. REFLECT (Liu et al. 2023d) summarizes hierarchical sensor data for explanations. HiCRISP (Ming et al. 2024) introduces a hierarchical closed-loop system.

#### 6.2.3 LLMs-driven structured planning

**Combining LLMs with symbolic systems:** PLANner (Lu et al. 2022) leverages an external KG (ConceptNet) to construct commonsense-infused prompts. PDDL models define structured symbolic blueprints. IALP (Wang et al. 2025a) develops a closed-loop system augmenting user instructions with feasibility information. LEMMo-Plan (Chen et al. 2025c) incorporates multi-modal demonstrations including tactile and force-torque data.

**Combining LLMs with behavior trees:** LLM-bt (Zhou et al. 2024b) uses an LLM to generate high-level descriptive steps parsed into an initial BT. LLM-OBTEA (Chen et al. 2024b) introduces a two-stage framework. BETR-XP-LLM (Styrud et al. 2025) uses the LLM to propose minimal preconditions and matching subtrees for execution failures. Ao et al. (2025) effectively exploit LLMs to directly generate BTs.

### 6.3 Empowered by vision-language models

While LLMs excel at high-level reasoning and planning, they suffer from a fundamental limitation: they are disembodied. VLMs offer a more direct and powerful solution by being pre-trained on vast datasets of paired images and text.

#### 6.3.1 Contrastive learning approaches

Contrastive learning is widely used in VLMs to align the text and vision modalities. CLIPORT (Shridhar et al. 2022) combines the broad semantic understanding of CLIP with the spatial precision of Transporter. EmbCLIP (Khandelwal et al. 2022) investigates the effectiveness of CLIP visual backbones. Instruction2Act (Huang et al. 2023a) leverages CLIP for object localization. R+X (Papagiannis et al. 2024) demonstrates how Gemini can retrieve relevant videos.

#### 6.3.2 Generative approaches

Generative approaches model data distributions to synthesize textual and visual content, conditioned on text, images, or both simultaneously.

**Text generation:** PaLM-E (Driess et al. 2023), a 562B parameter VLM, demonstrates that a single massive model could perform embodied reasoning. Socratic Models (Zeng et al. 2023) propose a framework where multiple specialized models communicate through multimodal prompts. PIVOT (Nasiriany et al. 2024) reframes manipulation as an iterative visual dialogue. PR2L (Chen et al. 2025f) demonstrates that VLMs can generate superior visual embeddings when prompted with task-relevant context. RoboPoint (Yuan et al. 2025) uses a synthetic data pipeline to instruction-tune a VLM to predict keypoint affordances.

**Image generation:** Dall-E-Bot (Kapelyukh et al. 2023) utilizes a text-conditioned diffusion model to generate goal images. SuSIE (Black et al. 2024) leverages pre-trained text-conditional image-editing models to generate subgoals. Generative models can also function as *world models* by predicting future video frames (Gu et al. 2024b; Ko et al. 2024; Du et al. 2023b). GR-MG (Li et al. 2025d) first trains a policy that accepts both language and image goals.

## 7 Language in unified vision-language-action models

As the field has evolved, increasing attention has shifted from hierarchical language-conditioned systems to end-to-end vision-language-action models (VLAs). By contrast to the hierarchical approaches, VLAs aim to learn a single policy model that jointly represents visual observations, language instructions, and robot actions within one architecture. In this sense, the role of language is no longer limited to high-level reasoning or external task specification. Instead, it is embedded in the same modeling space as perception and action.

Gato (Reed et al. 2022) represents a pioneering effort as a general-purpose agent. RT-1 (Brohan et al. 2022), RT-2 (Zitkovich et al. 2023), Gemini Robotics (Team et al. 2025; Abdolmaleki et al. 2025), and PI VLAs (Black et al. 2025b,a) utilize pre-trained vision-language models. RT-X (O'Neill et al. 2024) is trained on datasets from 22 different robots. OpenVLA (Kim et al. 2025c) utilizes the pre-trained open-source Llama 2 (7B). 3D-VLA (Zhen et al. 2024) is built on a 3D-based LLM. Our taxonomy follows: Perception $\to$ Reasoning $\to$ Action $\to$ Learning & Adapting.

### 7.1 Optimization for perception

#### 7.1.1 Data sources and augmentation

EgoVLA (Yang et al. 2025d) proposed learning from egocentric human videos in a two-stage process. H-RDT (Bi et al. 2025) seeks to scale up this paradigm from a single embodiment to multiple robot embodiments. Xing et al. (2025b) provide a comprehensive analysis of "shortcut learning" in aggregated datasets, proposing robotic data augmentation as a practical solution.

#### 7.1.2 3D scene representation and grounding

SpatialVLA (Qu et al. 2025) introduces Ego3D Position Encoding. PointVLA (Li et al. 2025a) proposes a modular framework that injects 3D information with minimal disruption. BridgeVLA (Li et al. 2025b) projects a 3D point cloud into multiple 2D orthographic images. GeoVLA (Sun et al. 2025) processes 2D and 3D modalities in parallel.

#### 7.1.3 Multimodal sensing and fusion

VTLA (Zhang et al. 2025a) tokenizes vision, tactile, and language. Tactile-VLA (Huang et al. 2025a) augments the policy's output to predict both target force and target position. OmniVTLA (Cheng et al. 2025) creates a unified tactile representation using a dual-path encoder. ForceVLA (Yu et al. 2025) introduces a dynamic "late fusion" mechanism using a force-aware Mixture-of-Experts module.

### 7.2 Optimization for reasoning

#### 7.2.1 Long-horizon planning

LoHoVLA (Yang et al. 2025f) and DexVLA (Wen et al. 2025a) explore hierarchical decomposition by training a unified model to first generate a linguistic sub-task and then predict the action. Long-VLA (Fan et al. 2025) introduces a more fine-grained phase-aware control strategy. MemoryVLA (Shi et al. 2026) incorporates an explicit memory system (Perceptual-Cognitive Memory Bank).

#### 7.2.2 Internal world models and reasoning

Seer (Tian et al. 2025) first predicts a future visual state and then uses an inverse dynamics model. CoT-VLA (Zhao et al. 2025) introduces a Visual Chain-of-Thought. WorldVLA (Cen et al. 2025) unifies the action model and world model into a single autoregressive framework. DreamVLA (Zhang et al. 2025d) proposes forecasting a compact latent "world embedding" instead of raw pixels.

**Preserving foundational VLM capabilities:** DiffusionVLA (Wen et al. 2025c) proposes a unified framework integrating autoregressive reasoning with a diffusion model. ChatVLA (Zhou et al. 2025c) introduces a MoE architecture. Knowledge Insulation (Driess et al. 2025) stops the gradient flow from the action expert back into the VLM backbone. InstructVLA (Yang et al. 2025e) introduces Vision-Language-Action Instruction Tuning. GR-3 (Cheang et al. 2025) employs extensive co-training with web-scale vision-language data.

### 7.3 Optimization for action

$\pi_0$ (Black et al. 2025b) shifts away from discrete action tokens toward a continuous generative process using *flow matching*. Pertsch et al. (2025) observe that autoregressive VLA performance suffers from redundancy and propose FAST tokenization. Discrete Diffusion VLA (Liang et al. 2025) models discretized action chunks using a discrete diffusion process within a single unified transformer. $\pi_{0.5}$ (Black et al. 2025a) answers the question of generalization by *jointly* learning discrete and continuous action heads.

### 7.4 Optimization for learning & adapting

OpenVLA-OFT (Kim et al. 2025b) investigates the efficiency bottleneck of VLA fine-tuning. ControlVLA (Li et al. 2025e) injects an object-centric mask into a frozen VLA backbone. ConRFT (Chen et al. 2025h) integrates RL into the VLA fine-tuning process. RIPT-VLA (Tan et al. 2025) introduces a streamlined post-training paradigm using sparse success rewards.

## 8 Comparative analysis

The preceding sections systematically categorized language-conditioned methodologies based on the functional role of language within the control loop. While this taxonomy clarifies *how* language is integrated, it is equally important to analyze the structural design choices and practical trade-offs that determine a system's real-world feasibility. In this section, we conduct a detailed comparative analysis of representative approaches, evaluating them along five cross-cutting axes:

- **Action granularity:** We analyze whether methods output high-level skills or low-level torques/poses. This is a critical design choice that determines a system's precision, ability to handle contact-rich tasks, and real-time feasibility.

- **Data and supervision regimes:** We compare methods based on their data sources and supervision signals. This axis is crucial as it directly impacts annotation cost, scalability, and a method's ability to generalize to out-of-distribution scenarios.

- **System cost and latency:** We explicitly compare the computational and memory costs of different approaches, both at training and inference time. This axis highlights the practical gap between theoretically powerful models and deployable systems.

- **Environments and evaluations:** We compare the benchmarks, tasks, and hardware used for evaluation, highlighting the critical distinction between results achieved in controlled simulations versus the complexities of the real world.

- **Cross-modal task specification:** We compare methods based on whether they condition on language instructions or other modality specifications (e.g., goal images/videos). This axis reveals the trade-offs between the expressiveness and flexibility of language versus the directness and precision of visual goals.

### 8.1 Action granularity

#### 8.1.1 Skill-level actions

Skill-level actions represent the nature composition of the language instruction. Representative approaches issue high-level skills or API calls rather than time-parameterized waypoints/poses or torques of the robot's joints. The planner-level LLMs such as SayCan (Ahn et al. 2022), Code as Policies (Liang et al. 2023), Inner Monologue (Huang et al. 2022b), ReKep (Huang et al. 2025b) follow this design.

#### 8.1.2 Trajectory-level actions

Trajectory-level actions involve outputting end-effector waypoints or trajectories, rather than skills. Many language-conditioned IL approaches (Lynch and Sermanet 2021; Jang et al. 2022; Wang et al. 2023a; Mees et al. 2023; Zhou et al. 2024a) fall here. Diffusion models have also emerged as powerful trajectory generators (Chi et al. 2023).

#### 8.1.3 Low-level control

Some tasks, such as dexterous manipulations (Ma et al. 2024) or handling fragile/deformable objects (Kobayashi et al. 2025), are defined by physical contact and force dynamics. This motivates integrating language with low-level torque/force control. ManiFoundation Model (Xu et al. 2024) frames manipulation as "contact synthesis". TA-VLA (Zhang et al. 2025f) explores the design space for creating "torque-aware" VLAs.

### 8.2 Data and supervision regime

#### 8.2.1 Data sources

Language-conditioned manipulation relies on two broad categories of data: (i) Robotic interaction data (direct experience) and (ii) Web-scale data (human-curated content). Robotic data includes expert demonstrations, unstructured play data, and web-scale data.

#### 8.2.2 Supervision

We classify supervision into three types: target labels (actions, subgoals, success/value labels), outcome evaluations (rewards, success, preferences, penalties), and auxiliary supervision (visual attention, reconstruction, future prediction).

### 8.3 System cost and latency

#### 8.3.1 Training cost

Three training strategies are identified: (i) Training from scratch, (ii) Fine-tuning, and (iii) Prompt engineering.

#### 8.3.2 Inference cost

The inference time largely depends on the model scale and the computation resources. Systems that prompt a frozen LLM/VLM inherit the reasoning breadth of very large FMs but typically run open-loop. For end-to-end VLA policies, fine-tuning larger backbones improves generalization but increases per-step compute and memory.

### 8.4 Environments and evaluations

**CALVIN** (Mees et al. 2022b): Open-source simulated benchmark for long-horizon language-conditioned tasks. **Meta-World** (Yu et al. 2020): 50 diverse robot manipulation tasks on MuJoCo. **RLBench** (James et al. 2020): 100 unique hand-designed tasks. **VIMAbench** (Jiang et al. 2023): Supports multimodal prompts for 17 tabletop manipulation task templates. **LoHoRavens** (Zhang et al. 2023b): Ten long-horizon language-conditioned tasks. **ARNOLD** (Gong et al. 2023b): 40 distinctive objects and 20 scenes. **LIBERO** (Liu et al. 2023a): Procedural task generation pipeline for lifelong learning. **Open X-Embodiment** (O'Neill et al. 2024): Real-world large-scale dataset from 22 robots. **DROID** (Khazatsky et al. 2024): 76k teleoperated trajectories across 564 scenes. **Galaxea Open-World Dataset** (Jiang et al. 2025b): 100k teleoperated trajectories across 150 tasks.

### 8.5 Cross-modal task specification: Language-conditioned vs. Image/Video-conditioned

#### 8.5.1 State evaluation: Language-derived objectives vs. video-derived progress signals

Language-conditioned state evaluation specifies objectives through explicit compositional semantics. Image and video-based approaches excel at providing implicit and dense progress signals.

#### 8.5.2 Policy conditioning: Language instructions vs. Perceptual grounding

As a condition for policy execution, language instructions serve as high-level behavior specifiers, whereas goal images and videos offer direct perceptual grounding. Language-conditioned policies abstract away low-level physical details to offer three distinct advantages: low-bandwidth specification, compositional generalization, and interactive editability.

#### 8.5.3 Planning: Language-based decomposition vs. Visual imitation

Language serves as an "internal reasoning medium" for planning. Video-conditioned methods utilize visual demonstrations as dense kinematic and structural priors. Language-conditioned planning excels in scenarios that are branching, constraint-rich, or highly interactive.

#### 8.5.4 Summary

Overall, image and video-conditioned manipulation provide stronger perceptual grounding. Conversely, language-conditioned manipulation offers unmatched text semantic expressiveness for constraints, compositionality for open-vocabulary generalization, and interactive editability for non-expert human collaboration. The contemporary consensus points toward a unified multimodal task-specification framework.

## 9 Discussion

This section discusses three heat debates in the community: Are large-scale vision-language-action models the most direct path to generalization? Can learned world models provide the necessary foresight for robust planning? And what are the trade-offs between model scaling and the strict real-time constraints of physical control?

### 9.1 Are VLAs the right path forward?

The question of whether VLA models represent the right direction for robotics ultimately depends on whether the scaling principles that have succeeded in language and vision domains can also extend to embodied AI. Current VLA models are still smaller and less data-rich than modern foundation models in computer vision or NLP. Genuine scaling in robotics lies in expanding the task manifold, namely the complexity, diversity, and structure of interactions, rather than merely increasing model parameters.

### 9.2 Are world models the right path forward?

A world model is a learned predictor of environmental dynamics that enables a robot to "imagine" how states will evolve under candidate actions. The Dreamer family, e.g., DreamerV3 (Hafner et al. 2025), shows that such imagination-based learning can yield strong sample efficiency and long-horizon behavior across diverse control domains. World models contribute to improved safety and generalization. However, there are notable drawbacks including model bias and distributional brittleness, and computational overhead.

### 9.3 Does scaling help under real-time constraints?

While scaling up models clearly improves representation quality and broadens the range of instructions they can handle, controlling physical robots is constrained by strict real-time deadlines. If a larger model increases inference time beyond the budget, the control loop can miss its deadline, leading to jitter and degraded stability. Consequently, simply using a "bigger" model is only helpful if its benefits can be realized within the strict timing deadlines of the control loop.

## 10 Limitations and future directions

While vision and language models provide a strong foundation for language-conditioned robotic manipulation, several limitations remain for future research, such as *generalization capability* and *real-world safety*.

### 10.1 Generalization capability

The generalization capability of language-conditioned robot manipulation systems remains a central challenge for developing agents that can operate robustly across diverse, real-world scenarios.

#### 10.1.1 Language-conditioned datasets and evaluation

**Data availability:** Training large-scale language-conditioned manipulation models inherently demands extensive datasets. **Beyond web-scale data:** Neuro-symbolic approaches offer a promising alternative. **Benchmarking:** Quantifying generalization in real-world settings is challenging.

#### 10.1.2 Lifelong learning

Lifelong learning represents a transformative direction for adaptive, continuously improving robotic agents.

#### 10.1.3 Cross-embodiment alignment

Language is inherently embodiment-agnostic. However, robots differ significantly in perception systems, actuation mechanisms, and control dynamics. Cross-embodiment learning aims to align shared semantic representations across heterogeneous robot morphologies.

#### 10.1.4 Effectiveness of zero-shot capability

In language-conditioned robot manipulation, zero-shot capability is not a uniform property but depends on what is being generalized. Current methods show the strongest zero-shot behavior at the semantic and planning levels. As the problem moves closer to physical execution, zero-shot capability weakens further.

### 10.2 Real-world safety

Ensuring real-world safety in language-conditioned robot manipulation is critical. We summarize three major concerns, namely ambiguity in language, failure recovery, and real-time performance.

#### 10.2.1 Ambiguity in language

Natural language instructions alone can be ambiguous. They are often underspecified and depend on rich context for a correct interpretation. To mitigate this risk, robots need to understand the intent of human users and implement feedback loops to confirm interpretation before acting.

#### 10.2.2 Recovering from failures

Safety issues also emerge when robots fail during task execution due to software and hardware limitations. On the software side, LLM hallucinations can lead to serious task failures. Future research should focus on establishing closed-loop feedback mechanisms and advancing computational efficiency.

#### 10.2.3 Real-time performance

Large-scale models such as LLMs, VLMs, and VLAs often incur long inference times, which extend the control loop latency. Model compression methods can accelerate inference while preserving accuracy. Ensuring secure, encrypted communication protocols, robust authentication mechanisms, and integrity verification of transmitted data and model parameters is essential.

## 11 Conclusion

In summary, this survey presents an overview of the current language-conditioned robot manipulation approaches. Our analysis focuses on the primary ways language is integrated into the robotic systems, namely *language for state evaluation*, *language as a policy condition*, and *language for cognitive planning and reasoning*. Furthermore, our comparative analysis offers a multifaceted perspective along four axes: *action granularity*, *data and supervision regimes*, *system cost and latency*, and *environment and evaluation*. We also examine the central debates in language-conditioned robot manipulation concerning VLAs, world models, and scaling. Finally, we articulate the key challenges and delineate future directions, with particular emphasis on *generalization capability* and *real-world safety*.

## References

Abbeel P and Ng AY (2004) Apprenticeship learning via inverse reinforcement learning. In: *Proceedings of the twenty-first international conference on Machine learning*. p. 1.

Achiam J, Adler S, Agarwal S, Ahmad L, Akkaya I, Aleman FL, Almeida D, Altenschmidt J, Altman S, Anadkat S et al. (2023) Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*.

Adeniji A, Xie A, Sferrazza C, Seo Y, James S and Abbeel P (2023) Language reward modulation for pretraining reinforcement learning. *arXiv preprint arXiv:2308.12270*.

Ahn M, Brohan A, Brown N, Chebotar Y, Cortes O, David B, Finn C, Fu C, Gopalakrishnan K, Hausman K et al. (2022) Do as i can, not as i say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*.

Alayrac JB, Donahue J, Luc P, Miech A, Barr I, Hasson Y, Lenc K, Mensch A, Millican K, Reynolds M et al. (2022) Flamingo: a visual language model for few-shot learning. *Advances in Neural Information Processing Systems* 35: 23716-23736.

Andrychowicz M, Wolski F, Ray A, Schneider J, Fong R, Welinder P, McGrew B, Tobin J, Pieter Abbeel O and Zaremba W (2017) Hindsight experience replay. In: *Advances in Neural Information Processing Systems*, volume 30.

Argall BD, Chernova S, Veloso M and Browning B (2009) A survey of robot learning from demonstration. *Robotics and Autonomous Systems* 57(5): 469-483.

Arora S and Doshi P (2021) A survey of inverse reinforcement learning: Challenges, methods and progress. *Artificial Intelligence* 297: 103500.

Bain M and Sammut C (1995) A framework for behavioural cloning. In: *Machine Intelligence* 15. pp. 103-129.

Besold TR, d'Avila Garcez AS, Bader S, Bowman H, Domingos PM, Hitzler P, Kuhnberger KU, Lamb LC, Lima PMV, de Penning L, Pinkas G, Poon H and Zaverucha G (2021) Neural-symbolic learning and reasoning: A survey and interpretation. In: *Neuro-Symbolic Artificial Intelligence: The State of the Art*, volume 342. IOS Press, pp. 1-51.

Billard A and Kragic D (2019) Trends and challenges in robot manipulation. *Science* 364(6446): eaat8414.

Bing Z, Koch A, Yao X, Huang K and Knoll A (2023a) Meta-reinforcement learning via language instructions. In: *2023 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, pp. 5985-5991.

Black K, Brown N, Driess D, Esmail A, Equi MR, Finn C, Fusai N, Groom L, Hausman K, Ichter B et al. (2025b) $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control. In: *Proceedings of Robotics: Science and Systems*.

Black K, Nakamoto M, Atreya P, Walke HR, Finn C, Kumar A and Levine S (2024) Zero-shot robotic manipulation with pre-trained image-editing diffusion models. *The Twelfth International Conference on Learning Representations*.

Black K, Brown N, Darpinian J, Dhabalia K, Driess D, Esmail A et al. (2025a) $\pi_{0.5}$: a vision-language-action model with open-world generalization. In: *Proceedings of The 8th Conference on Robot Learning*, volume 305. PMLR, pp. 17-40.

Brohan A, Brown N, Carbajal J, Chebotar Y, Dabis J, Finn C, Gopalakrishnan K, Hausman K, Herzog A, Hsu J et al. (2022) Rt-1: Robotics transformer for real-world control at scale. *arXiv preprint arXiv:2212.06817*.

Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A et al. (2020) Language models are few-shot learners. *neural information processing systems* 33: 1877-1901.

Chi C, Feng S, Du Y, Xu Z, Cousineau E, Burchfiel B and Song S (2023) Diffusion policy: Visuomotor policy learning via action diffusion. *Robotics: Science and Systems*.

Devlin J, Chang MW, Lee K and Toutanova K (2019) Bert: Pre-training of deep bidirectional transformers for language understanding. In: *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics*. pp. 4171-4186.

Ding Y, Zhang X, Paxton C and Zhang S (2023) Task and motion planning with large language models for object rearrangement. In: *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, pp. 2086-2092.

Driess D, Xia F, Sajjadi MS, Lynch C, Chowdhery A, Ichter B, Wahid A, Tompson J, Vuong Q, Yu T et al. (2023) Palm-e: an embodied multimodal language model. In: *Proceedings of the 40th International Conference on Machine Learning*. pp. 8469-8488.

Fu J, Korattikara A, Levine S and Guadarrama S (2019) From language to goals: Inverse reinforcement learning for vision-based instruction following. *International Conference on Learning Representations*.

Ho J, Jain A and Abbeel P (2020) Denoising diffusion probabilistic models. *Advances in neural information processing systems* 33: 6840-6851.

Huang W, Abbeel P, Pathak D and Mordatch I (2022a) Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In: *International Conference on Machine Learning*. PMLR, pp. 9118-9147.

Huang W, Xia F, Xiao T, Chan H, Liang J, Florence P, Zeng A, Tompson J, Mordatch I, Chebotar Y et al. (2022b) Inner monologue: Embodied reasoning through planning with language models. *arXiv preprint arXiv:2207.05608*.

Huang W, Wang C, Zhang R, Li Y, Wu J and Fei-Fei L (2023b) Voxposer: Composable 3d value maps for robotic manipulation with language models. In: *Conference on Robot Learning*. PMLR, pp. 540-562.

Kim MJ, Pertsch K, Karamcheti S, Xiao T, Balakrishna A, Nair S, Rafailov E, Foster EP, Sanketi PR, Vuong Q et al. (2025c) Openvla: An open-source vision-language-action model. In: *Conference on Robot Learning*. PMLR, pp. 2679-2713.

Liang J, Huang W, Xia F, Xu P, Hausman K, Ichter B, Florence P and Zeng A (2023) Code as policies: Language model programs for embodied control. In: *2023 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, pp. 9493-9500.

Lynch C and Sermanet P (2021) Language conditioned imitation learning over unstructured data. In: *Robotics: Science and Systems XVII*.

Ma YJ, Liang W, Wang G, Huang DA, Bastani O, Jayaraman D, Zhu Y, Fan L and Anandkumar A (2024) Eureka: Human-level reward design via coding large language models. In: *The Twelfth International Conference on Learning Representations*.

Mees O, Hermann L and Burgard W (2022a) What matters in language conditioned robotic imitation learning over unstructured data. *IEEE Robotics and Automation Letters* 7(4): 11205-11212.

Mees O, Borja-Diaz J and Burgard W (2023) Grounding language with visual affordances over unstructured data. In: *ICRA*. IEEE, pp. 11576-11582.

O'Neill A et al. (2024) Open x-embodiment: Robotic learning datasets and RT-X models: Open x-embodiment collaboration. In: *ICRA*. IEEE, pp. 6892-6903.

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J et al. (2021) Learning transferable visual models from natural language supervision. In: *International conference on machine learning*. PMLR, pp. 8748-8763.

Rana K, Haviland J, Garg S, Abou-Chakra J, Reid I and Suenderhauf N (2023) Sayplan: Grounding large language models using 3d scene graphs for scalable task planning. *arXiv preprint arXiv:2307.06135*.

Reed S, Zolna K, Parisotto E, Colmenarejo SG, Novikov A et al. (2022) A generalist agent. *Transactions on Machine Learning Research*.

Shridhar M, Manuelli L and Fox D (2022) Cliport: What and where pathways for robotic manipulation. In: *Conference on Robot Learning*. PMLR, pp. 894-906.

Shridhar M, Manuelli L and Fox D (2023) Perceiver-actor: A multi-task transformer for robotic manipulation. In: *Conference on Robot Learning*. PMLR, pp. 785-799.

Sutton R and Barto A (1998) *Reinforcement learning: An introduction*. MIT press Cambridge.

Tellex S, Gopalan N, Kress-Gazit H and Matuszek C (2020) Robots that use language. *Annual Review of Control, Robotics, and Autonomous Systems* 3(1): 25-55.

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L and Polosukhin I (2017) Attention is all you need. *Advances in neural information processing systems* 30.

Wei J, Wang X, Schuurmans D, Bosma M, Xia F, Chi E, Le QV, Zhou D et al. (2022) Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems* 35: 24824-24837.

Wen J, Zhu Y, Li J, Tang Z, Shen C and Feng F (2025a) Dexvla: Vision-language model with plug-in diffusion expert for general robot control. *arXiv preprint arXiv:2502.05855*.

Wen J, Zhu Y, Zhu M, Tang Z, Li J, Zhou Z, Liu X, Shen C, Peng Y, Chen R et al. (2025c) DiffusionVLA: Scaling robot foundation models via unified diffusion and autoregression. In: *Proceedings of the 42nd International Conference on Machine Learning*, volume 267. PMLR, pp. 66558-66574.

Xie T, Zhao S, Wu CH, Liu Y, Luo Q, Zhong V, Yang Y and Yu T (2024) Text2reward: Reward shaping with language models for reinforcement learning. In: *The Twelfth International Conference on Learning Representations*.

Yu T, Quillen D, He Z, Julian R, Hausman K, Finn C and Levine S (2020) Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. *Conference on robot learning*: 1094-1100.

Yu W, Gileadi N, Fu C, Kirmani S, Lee KH, Arenas MG, Chiang HTL, Erez T, Hasenclever L, Humplik J et al. (2023c) Language to rewards for robotic skill synthesis. In: *Conference on Robot Learning*. PMLR, pp. 374-404.

Zeng A, Florence P, Tompson J, Welker S, Chien J, Attarian M, Armstrong T, Krasin I, Duong D, Sindhwani V et al. (2021) Transporter networks: Rearranging the visual world for robotic manipulation. In: *Conference on Robot Learning*. PMLR.

Zeng A, Attarian M, brian ichter, Choromanski KM, Wong A et al. (2023) Socratic models: Composing zero-shot multimodal reasoning with language. In: *The Eleventh International Conference on Learning Representations*.

Zhang Y, Yang J, Pan J, Storks S, Devraj N et al. (2022) DANLI: Deliberative agent for following natural language instructions. In: *Proceedings of the Conference on Empirical Methods in Natural Language Processing*.

Zhao TZ, Kumar V, Levine S and Finn C (2023a) Learning fine-grained bimanual manipulation with low-cost hardware. In: *Robotics: Science and Systems XIX*.

Ziebart BD, Maas AL, Bagnell JA, Dey AK et al. (2008) Maximum entropy inverse reinforcement learning. In: *Aaai*, volume 8. Chicago, IL, USA, pp. 1433-1438.

Zitkovich B, Yu T, Xu S, Xu P, Xiao T et al. (2023) Rt-2: Vision-language-action models transfer web knowledge to robotic control. In: *Conference on Robot Learning*. PMLR, pp. 2165-2183.
