HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: stackengine

- failed: xpatch

- failed: cuted

- failed: xstring

- failed: eqparbox

- failed: datetime2

- failed: environ

- failed: titletoc

- failed: trimclip

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: CC BY 4.0

arXiv:2301.04104v2 [cs.AI] 17 Apr 2024

[\xapptocmd\NewColumnType L[1]Q[l,#1] \NewColumnTypeC[1]Q[c,#1] \NewColumnTypeR[1]Q[r,#1] \NewEnvironmytabular[1] \BODY # Mastering Diverse Domains through World Models Danijar Hafner,12 Jurgis Pasukonis,1 Jimmy Ba,2 Timothy Lillicrap1 †† 1Google DeepMind. 2University of Toronto. Correspondence: mail@danijar.com Abstract Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence. Although current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, configuring them for new application domains requires significant human expertise and experimentation. We present DreamerV3, a general algorithm that outperforms specialized methods across over 150 diverse tasks, with a single configuration. Dreamer learns a model of the environment and improves its behavior by imagining future scenarios. Robustness techniques based on normalization, balancing, and transformations enable stable learning across domains. Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula. This achievement has been posed as a significant challenge in artificial intelligence that requires exploring farsighted strategies from pixels and sparse rewards in an open world. Our work allows solving challenging control problems without extensive experimentation, making reinforcement learning broadly applicable. * Figure 1: Benchmark summary. a, Using fixed hyperparameters across all domains, Dreamer outperforms tuned expert algorithms across a wide range of benchmarks and data budgets. Dreamer also substantially outperforms a high-quality implementation of the widely applicable PPO algorithm. b, Applied out of the box, Dreamer learns to obtain diamonds in the popular video game Minecraft from scratch given sparse rewards, a long-standing challenge in artificial intelligence for which previous approaches required human data or domain-specific heuristics. (a) Control Suite (b) Atari (c) ProcGen (d) DMLab (e) Minecraft Figure 2: Diverse visual domains used in the experiments. Dreamer succeeds across these domains, ranging from robot locomotion and manipulation tasks over Atari games, procedurally generated ProcGen levels, and DMLab tasks, that require spatial and temporal reasoning, to the complex and infinite world of Minecraft. We also evaluate Dreamer on non-visual domains. ## Introduction Reinforcement learning has enabled computers to solve tasks through interaction, such as surpassing humans in the games of Go and Dota 1](#bib.bib1), [2](#bib.bib2).
It is also a key component for improving large language models beyond what is demonstrated in their pretraining data[[3](#bib.bib3), [4](#bib.bib4)].
While PPO [[5](#bib.bib5)] has become a standard algorithm in the field of reinforcement learning, more specialized algorithms are often employed to achieve higher performance.
These specialized algorithms target the unique challenges posed by different application domains, such as continuous control [[6](#bib.bib6)], discrete actions [[7](#bib.bib7), [8](#bib.bib8)], sparse rewards [[9](#bib.bib9)], image inputs [[10](#bib.bib10)], spatial environments [[11](#bib.bib11)], and board games [[12](#bib.bib12)].
However, applying reinforcement learning algorithms to sufficiently new tasks—such as moving from video games to robotics tasks—requires substantial effort, expertise, and computational resources for tweaking the hyperparameters of the algorithm [[13](#bib.bib13)].
This brittleness poses a bottleneck in applying reinforcement learning to new problems and also limits the applicability of reinforcement learning to computationally expensive models or tasks where tuning is prohibitive.
Creating a general algorithm that learns to master new domains without having to be reconfigured has been a central challenge in artificial intelligence and would open up reinforcement learning to a wide range of practical applications.

We present Dreamer, a general algorithm that outperforms specialized expert algorithms across a wide range of domains while using fixed hyperparameters, making reinforcement learning readily applicable to new problems.
The algorithm is based on the idea of learning a world model that equips the agent with rich perception and the ability to imagine the future[[14](#bib.bib14), [15](#bib.bib15), [16](#bib.bib16)].
The world model predicts the outcomes of potential actions, a critic neural network judges the value of each outcome, and an actor neural network chooses actions to reach the best outcomes.
Although intuitively appealing, robustly learning and leveraging world models to achieve strong task performance has been an open problem[[17](#bib.bib17)].
Dreamer overcomes this challenge through a range of robustness techniques based on normalization, balancing, and transformations.
We observe robust learning not only across over 150 tasks from the domains summarized in [Figure 2](#S0.F2), but also across model sizes and training budgets, offering a predictable way to increase performance.
Notably, larger model sizes not only achieve higher scores but also require less interaction to solve a task.

To push the boundaries of reinforcement learning, we consider the popular video game Minecraft that has become a focal point of research in recent years [[18](#bib.bib18), [19](#bib.bib19), [20](#bib.bib20)], with international competitions held for developing algorithms that autonomously learn to collect diamonds in Minecraft111
The MineRL Diamond Competitions were held in 2019, 2020, and 2021 and provided a dataset of human expert trajectories: [https://minerl.io/diamond](https://minerl.io/diamond). Competitions in the following years focused on a wide range of tasks..
Solving this problem without human data has been widely recognized as a substantial challenge for artificial intelligence because of the sparse rewards, exploration difficulty, long time horizons, and the procedural diversity of this open world game [[18](#bib.bib18)].
Due to these obstacles, previous approaches resorted to using human expert data and domain-specific curricula [[19](#bib.bib19), [20](#bib.bib20)].
Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch.

![Figure](x2.png)

*(a) World Model Learning*

![Figure](x3.png)

*(b) Actor Critic Learning*

Figure 3: Training process of Dreamer. The world model encodes sensory inputs into discrete representations $z_{t}$ that are predicted by a sequence model with recurrent state $h_{t}$ given actions $a_{t}$. The inputs are reconstructed to shape the representations. The actor and critic predict actions $a_{t}$ and values $v_{t}$ and learn from trajectories of abstract representations predicted by the world model.

## Learning algorithm

We present the third generation of the Dreamer algorithm[[21](#bib.bib21), [22](#bib.bib22)].
The algorithm consists of three neural networks: the world model predicts the outcomes of potential actions, the critic judges the value of each outcome, and the actor chooses actions to reach the most valuable outcomes.
The components are trained concurrently from replayed experience while the agent interacts with the environment.
To succeed across domains, all three components need to accommodate different signal magnitudes and robustly balance terms in their objectives.
This is challenging as we are not only targeting similar tasks within the same domain but aim to learn across diverse domains with fixed hyperparameters.
This section introduces the world model, critic, and actor along with their robust loss functions, as well as tools for robustly predicting quantities of unknown orders of magnitude.

### World model learning

The world model learns compact representations of sensory inputs through autoencoding [[23](#bib.bib23)] and enables planning by predicting future representations and rewards for potential actions.
We implement the world model as a Recurrent State-Space Model (RSSM) [[24](#bib.bib24)], shown in [Figure 3](#Sx1.F3).
First, an encoder maps sensory inputs $x_{t}$ to stochastic representations $z_{t}$.
Then, a sequence model with recurrent state $h_{t}$ predicts the sequence of these representations given past actions $a_{t-1}$.
The concatenation of $h_{t}$ and $z_{t}$ forms the model state from which we predict rewards $r_{t}$ and episode continuation flags $c_{t}\in\{0,1\}$ and reconstruct the inputs to ensure informative representations:

| | $$\displaystyle\begin{aligned} \begin{aligned} \raisebox{8.39578pt}{\hbox to 0.0% pt{\hss\vbox to 0.0pt{\hbox{$\text{RSSM}\hskip 4.30554pt\begin{cases}\hphantom% {A}\\ \hphantom{A}\\ \hphantom{A}\end{cases}$}\vss}}}&\text{Sequence model:}\hskip 35.00005pt&&h_{t% }&\ =&\ f_{\phi}(h_{t-1},z_{t-1},a_{t-1})\\ &\text{Encoder:}\hskip 35.00005pt&&z_{t}&\ \sim&\ q_{\phi}(z_{t}\;|\;h_{t},x_{% t})\\ &\text{Dynamics predictor:}\hskip 35.00005pt&&\hat{z}_{t}&\ \sim&\ p_{\phi}(% \hat{z}_{t}\;|\;h_{t})\\ &\text{Reward predictor:}\hskip 35.00005pt&&\hat{r}_{t}&\ \sim&\ p_{\phi}(\hat% {r}_{t}\;|\;h_{t},z_{t})\\ &\text{Continue predictor:}\hskip 35.00005pt&&\hat{c}_{t}&\ \sim&\ p_{\phi}(% \hat{c}_{t}\;|\;h_{t},z_{t})\\ &\text{Decoder:}\hskip 35.00005pt&&\hat{x}_{t}&\ \sim&\ p_{\phi}(\hat{x}_{t}\;% |\;h_{t},z_{t})\end{aligned}\end{aligned}$$ | | (1) |
|---|---|---|---|---|---|---|---|---|

[Figure 4](#Sx2.F4) visualizes long-term video predictions of the world world.
The encoder and decoder use convolutional neural networks (CNN) for image inputs and multi-layer perceptrons (MLPs) for vector inputs.
The dynamics, reward, and continue predictors are also MLPs.
The representations are sampled from a vector of softmax distributions and we take straight-through gradients through the sampling step [[25](#bib.bib25), [22](#bib.bib22)].
Given a sequence batch of inputs $x_{1:T}$, actions $a_{1:T}$, rewards $r_{1:T}$, and continuation flags $c_{1:T}$, the world model parameters $\phi$ are optimized end-to-end to minimize the prediction loss $\mathcal{L}_{\mathrm{pred}}$, the dynamics loss $\mathcal{L}_{\mathrm{dyn}}$, and the representation loss $\mathcal{L}_{\mathrm{rep}}$ with corresponding loss weights $\beta_{\mathrm{pred}}=1$, $\beta_{\mathrm{dyn}}=1$, and $\beta_{\mathrm{rep}}=0.1$:

$$\displaystyle\begin{aligned} \mathcal{L}(\phi)\doteq\operatorname{E}_{q_{\phi}% }\Big{[}\textstyle\sum_{t=1}^{T}(\beta_{\mathrm{pred}}\mathcal{L}_{\mathrm{% pred}}(\phi)+\beta_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}(\phi)+\beta_{% \mathrm{rep}}\mathcal{L}_{\mathrm{rep}}(\phi))\Big{]}.\end{aligned}$$ \tag{2}

![Figure](x4.png)
![Figure](x5.png)

*Figure 4: Multi-step video predictions of a DMLab maze (top) and a quadrupedal robot (bottom). Given 5 context images and the full action sequence, the model predicts 45 frames into the future without access to intermediate images. The world model learns an understanding of the underlying structure of each environment.*

The prediction loss trains the decoder and reward predictor via the symlog squared loss described later, and the continue predictor via logistic regression.
The dynamics loss trains the sequence model to predict the next representation by minimizing the KL divergence between the predictor $p_{\phi}(z_{t}\;|\;h_{t})$ and the next stochastic representation $q_{\phi}(z_{t}\;|\;h_{t},x_{t})$.
The representation loss, in turn, trains the representations to become more predictable allowing us to use a factorized dynamics predictor for fast sampling during imagination training.
The two losses differ in the stop-gradient operator $\operatorname{sg}(\cdot)$ and their loss scale.
To avoid a degenerate solution where the dynamics are trivial to predict but fail to contain enough information about the inputs, we employ free bits[[26](#bib.bib26)] by clipping the dynamics and representation losses below the value of 1 nat $\approx$ 1.44 bits.
This disables them while they are already minimized well to focus learning on the prediction loss:

$$\displaystyle\begin{aligned} \mathcal{L_{\mathrm{pred}}}(\phi)&\doteq-\ln p_{% \phi}(x_{t}\;|\;z_{t},h_{t})-\ln p_{\phi}(r_{t}\;|\;z_{t},h_{t})-\ln p_{\phi}(% c_{t}\;|\;z_{t},h_{t})\\ \mathcal{L_{\mathrm{dyn}}}(\phi)&\doteq\max\bigl{(}1,\operatorname{KL}\!\big{[% }\operatorname{sg}(q_{\phi}(z_{t}\;|\;h_{t},x_{t}))\;\big{\|}\;\hskip 13.77771% ptp_{\phi}(z_{t}\;|\;h_{t})\hphantom{)}\big{]}\bigr{)}\\ \mathcal{L_{\mathrm{rep}}}(\phi)&\doteq\max\bigl{(}1,\operatorname{KL}\!\big{[% }\hskip 13.77771ptq_{\phi}(z_{t}\;|\;h_{t},x_{t})\hphantom{)}\;\big{\|}\;% \operatorname{sg}(p_{\phi}(z_{t}\;|\;h_{t}))\big{]}\bigr{)}\end{aligned}$$ \tag{3}

Previous world models require scaling the representation loss differently based on the visual complexity of the environment [[21](#bib.bib21)].
Complex 3D environments contain details unnecessary for control and thus prompt a stronger regularizer to simplify the representations and make them more predictable.
In games with static backgrounds and where individual pixels may matter for the task, a weak regularizer is required to extract fine details.
We find that combining free bits with a small representation loss resolves this dilemma, allowing for fixed hyperparameters across domains.
Moreover, we transform vector observations using the symlog function described later, to prevent large inputs and large reconstruction gradients, further stabilizing the trade-off with the representation loss.

We occasionally observed spikes the in KL losses in earlier experiments, consistent with reports for deep variational autoencoders [[27](#bib.bib27)].
To prevent this, we parameterize the categorical distributions of the encoder and dynamics predictor as mixtures of 1% uniform and 99% neural network output, making it impossible for them to become deterministic and thus ensuring well-behaved KL losses.
Further model details and hyperparameters are included in the supplementary material.

### Critic learning

The actor and critic neural networks learn behaviors purely from abstract trajectories of representations predicted by the world model [[14](#bib.bib14)].
For environment interaction, we select actions by sampling from the actor network without lookahead planning.
The actor and critic operate on model states $s_{t}\doteq\{h_{t},z_{t}\}$ and thus benefit from the Markovian representations learned by the recurrent world model.
The actor aims to maximize the return $R_{t}\doteq\textstyle\sum_{\tau=0}^{\infty}\gamma^{\tau}r_{t+\tau}$ with a discount factor $\gamma=0.997$ for each model state.
To consider rewards beyond the prediction horizon $T=16$, the critic learns to approximate the distribution of returns [[28](#bib.bib28)] for each state under the current actor behavior:

$$\displaystyle\begin{aligned} &\text{Actor:}\quad&&a_{t}\sim\pi_{\theta}(a_{t}% \;|\;s_{t})\qquad&\text{Critic:}\quad&&v_{\psi}(R_{t}\;|\;s_{t})\end{aligned}$$ \tag{4}

Starting from representations of replayed inputs, the world model and actor generate a trajectory of
imagined model states $s_{1:T}$, actions $a_{1:T}$, rewards $r_{1:T}$, and continuation flags $c_{1:T}$.
Because the critic predicts a distribution, we read out its predicted values $v_{t}\doteq\operatorname{E}[v_{\psi}(\,\cdot\;|\;s_{t})]$ as the expectation of the distribution.
To estimate returns that consider rewards beyond the prediction horizon, we compute bootstrapped $\lambda$-returns [[29](#bib.bib29)] that integrate the predicted rewards and the values.
The critic learns to predict the distribution of the return estimates $R^{\lambda}_{t}$ using the maximum likelihood loss:

$$\displaystyle\begin{aligned} \mathcal{L}(\psi)\doteq-\textstyle\sum_{t=1}^{T}% \ln p_{\psi}(R^{\lambda}_{t}\;|\;s_{t})\qquad R^{\lambda}_{t}\doteq r_{t}+% \gamma c_{t}\Big{(}(1-\lambda)v_{t}+\lambda R^{\lambda}_{t+1}\Big{)}\qquad R^{% \lambda}_{T}\doteq v_{T}\end{aligned}$$ \tag{5}

While a simple choice would be to parameterize the critic as a Normal distribution, the return distribution can have multiple modes and vary by orders of magnitude across environments. To stabilize and accelerate learning under these conditions, we parameterize the critic as categorical distribution with exponentially spaced bins, decoupling the scale of gradients from the prediction targets as described later.
To improve value prediction in environments where rewards are challenging to predict, we apply the critic loss both to imagined trajectories with loss scale $\beta_{\mathrm{val}}=1$ and to trajectories sampled from the replay buffer with loss scale $\beta_{\mathrm{repval}}=0.3$. The critic replay loss uses the imagination returns $R^{\lambda}_{t}$ at the start states of the imagination rollouts as on-policy value annotations for the replay trajectory to then compute $\lambda$-returns over the replay rewards.

Because the critic regresses targets that depend on its own predictions, we stabilize learning by regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters.
This is similar to target networks used previously in reinforcement learning [[7](#bib.bib7)] but allows us to compute returns using the current critic network.
We further noticed that the randomly initialized reward predictor and critic networks at the start of training can result in large predicted rewards that can delay the onset of learning.
We thus initialize the output weight matrix of the reward predictor and critic to zeros, which alleviates the problem and accelerates early learning.

### Actor learning

The actor learns to choose actions that maximize return while exploring through an entropy regularizer [[30](#bib.bib30)].
However, the correct scale for this regularizer depends both on the scale and frequency of rewards in the environment.
Ideally, we would like the agent to explore more if rewards are sparse and exploit more if rewards are dense or nearby. At the same time, the exploration amount should not be influenced by arbitrary scaling of rewards in the environment.
This requires normalizing the return scale while preserving information about reward frequency.

To use a fixed entropy scale of $\eta=3\times 10^{-4}$ across domains, we normalize returns to be approximately contained in the interval $[0,1]$. In practice, substracting an offset from the returns does not change the actor gradient and thus dividing by the range $S$ is sufficient. Moreover, to avoid amplifying noise from function approximation under sparse rewards, we only scale down large return magnitudes but leave small returns below the threshold of $L=1$ untouched. We use the Reinforce estimator [[31](#bib.bib31)] for both discrete and continuous actions, resulting in the surrogate loss function:

$$\displaystyle\begin{aligned} \mathcal{L}(\theta)\doteq-\textstyle\sum_{t=1}^{T% }\operatorname{sg}\!\Big{(}\big{(}R^{\lambda}_{t}-v_{\psi}(s_{t})\big{)}/\max(% 1,\,S)\Big{)}\log\pi_{\theta}(a_{t}\;|\;s_{t})+\,\eta\operatorname{H}\big{[}% \pi_{\theta}(a_{t}\;\big{|}\;s_{t})\big{]}\end{aligned}$$ \tag{6}

The return distribution can be multi-modal and include outliers, especially for randomized environments where some episodes have higher achievable returns than others. Normalizing by the smallest and largest observed returns would then scale returns down too much and may cause suboptimal convergence. To be robust to these outliers, we compute the range from the 5th to the 95th return percentile over the return batch and smooth out the estimate using an exponential moving average:

$$\displaystyle\begin{aligned} S\doteq\operatorname{EMA}\!\big{(}\operatorname{% Per}(R^{\lambda}_{t},95)-\operatorname{Per}(R^{\lambda}_{t},5),0.99\big{)}\end% {aligned}$$ \tag{7}

Previous work typically normalizes advantages[[5](#bib.bib5)] rather than returns, which puts a fixed amount of emphasis on maximizing returns over entropy regardless of whether rewards are within reach.
Scaling up advantages when rewards are sparse can amplify noise that outweighs the entropy regularizer and stagnates exploration.
Normalizing rewards or returns by standard deviation can fail under sparse rewards where their standard deviation is near zero, drastically amplifying rewards regardless of their size.
Constrained optimization targets a fixed entropy on average across states [[32](#bib.bib32), [33](#bib.bib33)] regardless of achievable returns, which is robust but explores slowly under sparse rewards and converges lower under dense rewards.
We did not find stable hyperparameters across domains for these approaches.
Return normalization with a denominator limit overcomes these challenges, exploring rapidly under sparse rewards and converging to high performance across diverse domains.

### Robust predictions

Reconstructing inputs and predicting rewards and returns can be challenging because the scale of these quantities can vary across domains.
Predicting large targets using a squared loss can lead to divergence whereas absolute and Huber losses [[7](#bib.bib7)] stagnate learning.
On the other hand, normalizing targets based on running statistics [[5](#bib.bib5)] introduces non-stationarity into the optimization.
We suggest the symlog squared error as a simple solution to this dilemma.
For this, a neural network $f(x,\theta)$ with inputs $x$ and parameters $\theta$ learns to predict a transformed version of its targets $y$.
To read out predictions $\hat{y}$ of the network, we apply the inverse transformation:

$$\displaystyle\begin{aligned} \mathcal{L(\theta)}\doteq\textstyle\frac{1}{2}% \big{(}f(x,\theta)-\operatorname{symlog}(y)\big{)}^{2}\qquad\hat{y}\doteq% \operatorname{symexp}\!\big{(}f(x,\theta)\big{)}\end{aligned}$$ \tag{8}

Using the logarithm as transformation would not allow us to predict targets that take on negative values.
Therefore, we choose a function from the bi-symmetric logarithmic family [[34](#bib.bib34)] that we name symlog as the transformation with the symexp function as its inverse:

$$\displaystyle\begin{aligned} \operatorname{symlog}(x)\doteq\operatorname{sign}% (x)\ln\!\big{(}|x|+1\big{)}\qquad\operatorname{symexp}(x)\doteq\operatorname{% sign}(x)\big{(}\!\exp(|x|)-1\big{)}\end{aligned}$$ \tag{9}

The symlog function compresses the magnitudes of both large positive and negative values.
Unlike the logarithm, it is symmetric around the origin while preserving the input sign.
This allows the optimization process to quickly move the network predictions to large values when needed.
The symlog function approximates the identity around the origin so that it does not affect learning of targets that are already small enough.

For potentially stochastic targets, such as rewards or returns, we introduce the symexp twohot loss. Here, the network outputs the logits for a softmax distribution over exponentially spaced bins $b_{i}\in B$.
Predictions are read out as the weighted average of the bin positions weighted by their predicted probabilities. Importantly, the network can output any continuous value in the interval because the weighted average can fall between the buckets:

$$\displaystyle\begin{aligned} \hat{y}\doteq\operatorname{softmax}(f(x))^{T}B% \qquad B\doteq\operatorname{symexp}(\begin{bmatrix}-20&...&+20\end{bmatrix})% \end{aligned}$$ \tag{10}

The network is trained on twohot encoded targets [[8](#bib.bib8), [28](#bib.bib28)], a generalization of onehot encoding to continuous values. The twohot encoding of a scalar is a vector with $|B|$ entries that are all 00 except at the indices $k$ and $k+1$ of the two bins closest to the encoded scalar. The two entries sum up to $1$, with linearly higher weight given to the bin that is closer to the encoded continuous number.
The network is then trained to minimize the categorical cross entropy loss for classification with soft targets. Note that the loss only depends on the probabilities assigned to the bins but not on the continuous values associated with the bin locations, decoupling the size of the gradients from the size of the targets:

$$\displaystyle\begin{aligned} \mathcal{L}(\theta)\doteq-\operatorname{twohot}(y% )^{T}\operatorname{log\,softmax}(f(x,\theta))\end{aligned}$$ \tag{11}

Applying these principles, Dreamer transforms vector observations using the symlog functions, both for the encoder inputs and the decoder targets and employs the synexp twohot loss for the reward predictor and critic.
We find that these techniques enable robust and fast learning across many diverse domains.
For critic learning, an alternative asymmetric transformation has previously been proposed [[35](#bib.bib35)], which we found less effective on average across domains.
Unlike alternatives, symlog transformations avoid truncating large targets [[7](#bib.bib7)], introducing non-stationary from normalization [[5](#bib.bib5)], or adjusting network weights when new extreme values are detected [[36](#bib.bib36)].

![Figure](x6.png)

*Figure 5: Fraction of trained agents that discover each of the three latest items in the Minecraft Diamond task. Although previous algorithms progress up to the iron pickaxe, Dreamer is the only compared algorithm that manages to discover a diamond, and does so reliably.*

![Figure](x7.png)

*Figure 6: Ablations and robust scaling of Dreamer. a, All individual robustness techniques contribute to the performance of Dreamer on average, although each individual technique may only affect some tasks. Training curves of individual tasks are included in the supplementary material. b, The performance of Dreamer predominantly rests on the unsupervised reconstruction loss of its world model, unlike most prior algorithms that rely predominantly on reward and value prediction gradients[[7](#bib.bib7), [5](#bib.bib5), [8](#bib.bib8)]. c, The performance of Dreamer increases monotonically with larger model sizes, ranging from 12M to 400M parameters. Notably, larger models not only increase task performance but also require less environment interaction. d, Higher replay ratios predictably increase the performance of Dreamer. Together with model size, this allows practitioners to improve task performance and data-efficiency by employing more computational resources.*

## Results

We evaluate the generality of Dreamer across 8 domains—with over 150 tasks—under fixed hyperparameters.
We designed the experiments to compare Dreamer to the best methods in the literature, which are often specifically designed and tuned for the benchmark at hand.
We further compare to a high-quality implementation of PPO[[5](#bib.bib5)], a standard reinforcement learning algorithm that is known for its robustness. We run PPO with fixed hyperparameters chosen to maximize performance across domains and that reproduce strong published results of PPO on ProcGen[[37](#bib.bib37)].
To push the boundaries of reinforcement learning, we apply Dreamer to the challenging video game Minecraft, comparing it to strong previous algorithms.
Finally, we analyze the importance of individual components of Dreamer and its robustness to different model sizes and computational budgets.
All Dreamer agents are trained on a single Nvidia A100 GPU each, making it reproducible for many research labs.
A public implementation of Dreamer that reproduces all results is available on the project website.

#### Benchmarks

We perform an extensive empirical study across 8 domains that include continuous and discrete actions, visual and low-dimensional inputs, dense and sparse rewards, different reward scales, 2D and 3D worlds, and procedural generation.
[Figure 1](#S0.F1) summarizes the benchmark results, showing that Dreamer outperforms a wide range of previous expert algorithms across diverse domains. Crucially, Dreamer substantially outperforms PPO across all domains.

-
•

Atari This established benchmark contains 57 Atari 2600 games with a budget of 200M frames, posing a diverse range of challenges[[38](#bib.bib38)].
We use the sticky action simulator setting [[39](#bib.bib39)].
Dreamer outperforms the powerful MuZero algorithm[[8](#bib.bib8)] while using only a fraction of the computational resources.
Dreamer also outperforms the widely-used expert algorithms Rainbow [[40](#bib.bib40)] and IQN [[41](#bib.bib41)].

-
•

ProcGen This benchmark of 16 games features randomized levels and visual distractions to test the robustness and generalization of agents[[42](#bib.bib42)]. Within the budget of 50M frames, Dreamer matches the tuned expert algorithm PPG [[37](#bib.bib37)] and outperforms Rainbow[[42](#bib.bib42), [40](#bib.bib40)]. Our PPO agent with fixed hyperparameters matches the published score of the highly tuned official PPO implementation[[37](#bib.bib37)].

-
•

DMLab This suite of 30 tasks features 3D environments that test spatial and temporal reasoning [[43](#bib.bib43)]. In 100M frames, Dreamer exceeds the performance of the scalable IMPALA and R2D2+ agents [[35](#bib.bib35)] at 1B environment steps, amounting to a data-efficiency gain of over 1000%.
We note that these baselines were not designed for data-efficiency but serve as a valuable comparison point for the performance previously achievable at scale.

-
•

Atari100k This data-efficiency benchmark comntains 26 Atari games and a budget of only 400K frames, amounting to 2 hours of game time [[17](#bib.bib17)]. EfficientZero [[44](#bib.bib44)] holds the state-of-the-art by combining online tree search, prioritized replay, and hyperparameter scheduling, but also resets levels early to increase data diversity, making a comparison difficult.
Without this complexity, Dreamer outperforms the best remaining methods, including the transformer-based IRIS and TWM agents, the model-free SPR, and SimPLe [[45](#bib.bib45)].

-
•

Proprio Control This benchmark contains 18 control tasks with continuous actions, proprioceptive vector inputs, and a budget of 500K environment steps [[46](#bib.bib46)]. The tasks range from classical control over locomotion to robot manipulation tasks, featuring dense and sparse rewards. Dreamer sets a new state-of-the-art on this benchmark,
outperforming D4PG, DMPO, and MPO [[33](#bib.bib33)].

-
•

Visual Control This benchmark consists of 20 continuous control tasks where the agent receives only high-dimensional images as input and has a budget of 1M environment steps [[46](#bib.bib46)].
Dreamer establishes a new state-of-the-art on this benchmark, outperforming DrQ-v2 and CURL [[47](#bib.bib47)], which are specialized to visual environments and leverage data augmentation.

-
•

BSuite This benchmark includes 23 environments with a total of 468 configurations that are specifically designed to test credit assignment, robustness to reward scale and stochasticity, memory, generalization, and exploration [[48](#bib.bib48)]. Dreamer establishes a new state-of-the-art on this benchmark, outperforming Boot DQN and other methods [[49](#bib.bib49)].
Dreamer improves over previous algorithms especially in the scale robustness category.

#### Minecraft

Collecting diamonds in the popular game Minecraft has been a long-standing challenge in artificial intelligence[[18](#bib.bib18), [19](#bib.bib19), [20](#bib.bib20)].
Every episode in this game is set in a unique randomly generated and infinite 3D world. Episodes last until the player dies or up to 36000 steps equaling 30 minutes, during which the player needs to discover a sequence of 12 items from sparse rewards by foraging for resources and crafting tools. It takes about 20 minutes for experienced human players to obtain diamonds[[20](#bib.bib20)].
We follow the block breaking setting of prior work [[19](#bib.bib19)] because the provided action space would make it challenging for stochastic policies to keep a key pressed for a prolonged time.

Because of the training time in this complex domain, extensive tuning would be difficult for Minecraft.
Instead, we apply Dreamer out of the box with its default hyperparameters.
As shown in [Figures 5](#Sx2.F5) and [1](#S0.F1), Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without using human data as was required by VPT [[20](#bib.bib20)] or adaptive curricula[[19](#bib.bib19)].
All the Dreamer agents we trained on Minecraft discover diamonds in 100M environment steps.
While several strong baselines progress to advanced items such as the iron pickaxe, none of them discovers a diamond.

#### Ablations

In [Figure 6](#Sx2.F6), we ablate the robustness techniques and learning signals on a diverse set of 14 tasks to understand their importance.
The training curves of individual tasks are included in the supplementary material.
We observe that all robustness techniques contribute to performance, most notably the KL objective of the world model, followed by return normalization and symexp twohot regression for reward and value prediction. In general, we find that each individual technique is critical on a subset of tasks but may not affect performance on other tasks.

To investigate the effect of the world model, we ablate the learning signals of Dreamer by stopping either the task-specific reward and value prediction gradients or the task-agnostic reconstruction gradients from shaping its representations. Unlike previous reinforcement learning algorithms that often rely only on task-specific learning signals[[7](#bib.bib7), [8](#bib.bib8)], Dreamer rests predominantly on the unsupervised objective of its world model. This finding could allow for future algorithm variants that leverage pretraining on unsupervised data.

#### Scaling properties

To investigate whether Dreamer can scale robustly, we train 6 model sizes ranging from 12M to 400M parameters, as well as different replay ratios on Crafter [[50](#bib.bib50)] and a DMLab task[[43](#bib.bib43)]. The replay ratio affects the number of gradient updates performed by the agent.
[Figure 6](#Sx2.F6) shows robust learning with fixed hyperparameters across the compared model sizes and replay ratios.
Moreover, increasing the model size directly translates to both higher task performance and a lower data requirement.
Increasing the number of gradient steps further reduces the interactions needed to learn successful behaviors.
The results show that Dreamer learns robustly across model sizes and replay ratios and that its performance and provides a predictable way for increasing performance given computational resources.

## Previous work

Developing general-purpose algorithms has long been a goal of reinforcement learning research.
PPO [[5](#bib.bib5)] is one of the most widely used algorithms and is relatively robust but requires large amounts of experience and often yields lower performance than specialized alternatives.
SAC [[32](#bib.bib32)] is a popular choice for continuous control and leverages experience replay for data-efficiency, but in practice requires tuning, especially for its entropy scale, and struggles under high-dimensional inputs [[51](#bib.bib51)].
MuZero [[8](#bib.bib8)] plans using a value prediction model and has been applied to board games and Atari, but the authors did not release an implementation and the algorithm contains several complex components, making it challenging to reproduce.
Gato [[52](#bib.bib52)] fits one large model to expert demonstrations of multiple tasks, but is only applicable when expert data is available.
In comparison, Dreamer masters a diverse range of environments with fixed hyperparameters, does not require expert data, and its implementation is open source.

Minecraft has been a focus of recent research. With MALMO [[53](#bib.bib53)], Microsoft released a free version of the successful game for research purposes.
MineRL [[18](#bib.bib18)] offers several competition environments, which we rely on as the basis for our experiments.
The MineRL competition supports agents in exploring and learning meaningful skills through a diverse human dataset [[18](#bib.bib18)].
Voyager obtains items at a similar depth in the technology tree as Dreamer using API calls to a language model but operates on top of the MineFlayer bot scripting layer that was specifically engineered to the game and exposes high-level actions[[54](#bib.bib54)].
VPT [[20](#bib.bib20)] trained an agent to play Minecraft through behavioral cloning based on expert data of keyboard and mouse actions collected by contractors and finetuning using reinforcement learning to obtain diamonds using 720 GPUs for 9 days.
In comparison, Dreamer uses the MineRL competition action space to autonomously learn to collect diamonds from sparse rewards using 1 GPU for 9 days, without human data.

## Conclusion

We present the third generation of the Dreamer algorithm, a general reinforcement learning algorithm that masters a wide range of domains with fixed hyperparameters.
Dreamer excels not only across over 150 tasks but also learns robustly across varying data and compute budgets, moving reinforcement learning toward a wide range of practical applications.
Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch, achieving a significant milestone in the field of artificial intelligence.
As a high-performing algorithm that is based on a learned world model, Dreamer paves the way for future research directions, including teaching agents world knowledge from internet videos and learning a single world model across domains to allow artificial agents to build up increasingly general knowledge and competency.

#### Acknowledgements

We thank Mohammad Norouzi, Jessy Lin, Abbas Abdolmaleki, John Schulman, Adam Kosiorek, and Oleh Rybkin for insightful discussions.
We thank Bobak Shahriari, Denis Yarats, Karl Cobbe,
and Hubert Soyer for sharing training curves of baseline algorithms. We thank Daniel Furrer, Andrew Chen, and Dakshesh Garambha for help with Google Cloud infrastructure.

## References

-
Silver et al. 2016

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George
Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda
Panneershelvam, Marc Lanctot, et al.

Mastering the game of go with deep neural networks and tree search.

nature*, 529(7587):484, 2016.

-
OpenAI 2018

OpenAI.

OpenAI Five.

[https://blog.openai.com/openai-five/](https://blog.openai.com/openai-five/), 2018.

-
Ouyang et al. 2022

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

Training language models to follow instructions with human feedback.

*Advances in neural information processing systems*,
35:27730–27744, 2022.

-
Le et al. 2022

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven
Chu Hong Hoi.

Coderl: Mastering code generation through pretrained models and deep
reinforcement learning.

*Advances in Neural Information Processing Systems*,
35:21314–21328, 2022.

-
Schulman et al. 2017

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

*arXiv preprint arXiv:1707.06347*, 2017.

-
Lillicrap et al. 2015

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom
Erez, Yuval Tassa, David Silver, and Daan Wierstra.

Continuous control with deep reinforcement learning.

*arXiv preprint arXiv:1509.02971*, 2015.

-
Mnih et al. 2015

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness,
Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg
Ostrovski, et al.

Human-level control through deep reinforcement learning.

*Nature*, 518(7540):529, 2015.

-
Schrittwieser et al. 2019

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan,
Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis,
Thore Graepel, et al.

Mastering atari, go, chess and shogi by planning with a learned
model.

*arXiv preprint arXiv:1911.08265*, 2019.

-
Jaderberg et al. 2016

Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z
Leibo, David Silver, and Koray Kavukcuoglu.

Reinforcement learning with unsupervised auxiliary tasks.

*arXiv preprint arXiv:1611.05397*, 2016.

-
Anand et al. 2019

Ankesh Anand, Evan Racah, Sherjil Ozair, Yoshua Bengio, Marc-Alexandre
Côté, and R Devon Hjelm.

Unsupervised state representation learning in atari.

*Advances in neural information processing systems*, 32, 2019.

-
Driess et al. 2022

Danny Driess, Ingmar Schubert, Pete Florence, Yunzhu Li, and Marc Toussaint.

Reinforcement learning with neural radiance fields.

*arXiv preprint arXiv:2206.01634*, 2022.

-
Silver et al. 2017

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja
Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton,
et al.

Mastering the game of go without human knowledge.

*Nature*, 550(7676):354, 2017.

-
Andrychowicz et al. 2020

Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan
Girgin, Raphael Marinier, Léonard Hussenot, Matthieu Geist, Olivier
Pietquin, Marcin Michalski, et al.

What matters in on-policy reinforcement learning? a large-scale
empirical study.

*arXiv preprint arXiv:2006.05990*, 2020.

-
Sutton 1991

Richard S Sutton.

Dyna, an integrated architecture for learning, planning, and
reacting.

*ACM SIGART Bulletin*, 2(4):160–163, 1991.

-
Finn and Levine 2017

Chelsea Finn and Sergey Levine.

Deep visual foresight for planning robot motion.

In *2017 IEEE International Conference on Robotics and
Automation (ICRA)*, pages 2786–2793. IEEE, 2017.

-
Ha and Schmidhuber 2018

David Ha and Jürgen Schmidhuber.

World models.

*arXiv preprint arXiv:1803.10122*, 2018.

-
Kaiser et al. 2019

Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H
Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski,
Sergey Levine, et al.

Model-based reinforcement learning for atari.

*arXiv preprint arXiv:1903.00374*, 2019.

-
Guss et al. 2019

William H Guss, Cayden Codel, Katja Hofmann, Brandon Houghton, Noboru Kuno,
Stephanie Milani, Sharada Mohanty, Diego Perez Liebana, Ruslan Salakhutdinov,
Nicholay Topin, et al.

The minerl competition on sample efficient reinforcement learning
using human priors.

*arXiv e-prints*, pages arXiv–1904, 2019.

-
Kanitscheider et al. 2021

Ingmar Kanitscheider, Joost Huizinga, David Farhi, William Hebgen Guss, Brandon
Houghton, Raul Sampedro, Peter Zhokhov, Bowen Baker, Adrien Ecoffet, Jie
Tang, et al.

Multi-task curriculum learning in a complex, visual, hard-exploration
domain: Minecraft.

*arXiv preprint arXiv:2106.14876*, 2021.

-
Baker et al. 2022

Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien
Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune.

Video pretraining (vpt): Learning to act by watching unlabeled online
videos.

*arXiv preprint arXiv:2206.11795*, 2022.

-
Hafner et al. 2019

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi.

Dream to control: Learning behaviors by latent imagination.

*arXiv preprint arXiv:1912.01603*, 2019.

-
Hafner et al. 2020

Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba.

Mastering atari with discrete world models.

*arXiv preprint arXiv:2010.02193*, 2020.

-
Kingma and Welling 2013

Diederik P Kingma and Max Welling.

Auto-encoding variational bayes.

*arXiv preprint arXiv:1312.6114*, 2013.

-
Hafner et al. 2018

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha,
Honglak Lee, and James Davidson.

Learning latent dynamics for planning from pixels.

*arXiv preprint arXiv:1811.04551*, 2018.

-
Bengio et al. 2013

Yoshua Bengio, Nicholas Léonard, and Aaron Courville.

Estimating or propagating gradients through stochastic neurons for
conditional computation.

*arXiv preprint arXiv:1308.3432*, 2013.

-
Kingma et al. 2016

Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max
Welling.

Improved variational inference with inverse autoregressive flow.

*Advances in neural information processing systems*, 29, 2016.

-
Child 2020

Rewon Child.

Very deep vaes generalize autoregressive models and can outperform
them on images.

*arXiv preprint arXiv:2011.10650*, 2020.

-
Bellemare et al. 2017

Marc G Bellemare, Will Dabney, and Rémi Munos.

A distributional perspective on reinforcement learning.

In *International Conference on Machine Learning*, pages
449–458. PMLR, 2017.

-
Sutton and Barto 2018

Richard S Sutton and Andrew G Barto.

*Reinforcement learning: An introduction*.

MIT press, 2018.

-
Williams and Peng 1991

Ronald J Williams and Jing Peng.

Function optimization using connectionist reinforcement learning
algorithms.

*Connection Science*, 3(3):241–268, 1991.

-
Williams 1992

Ronald J Williams.

Simple statistical gradient-following algorithms for connectionist
reinforcement learning.

*Machine learning*, 8(3-4):229–256, 1992.

-
Haarnoja et al. 2018

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

Soft actor-critic: Off-policy maximum entropy deep reinforcement
learning with a stochastic actor.

*arXiv preprint arXiv:1801.01290*, 2018.

-
Abdolmaleki et al. 2018

Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas
Heess, and Martin Riedmiller.

Maximum a posteriori policy optimisation.

*arXiv preprint arXiv:1806.06920*, 2018.

-
Webber 2012

J Beau W Webber.

A bi-symmetric log transformation for wide-range data.

*Measurement Science and Technology*, 24(2):027001, 2012.

-
Kapturowski et al. 2018

Steven Kapturowski, Georg Ostrovski, John Quan, Remi Munos, and Will Dabney.

Recurrent experience replay in distributed reinforcement learning.

In *International conference on learning representations*, 2018.

-
Hessel et al. 2019

Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt,
and Hado van Hasselt.

Multi-task deep reinforcement learning with popart.

In *Proceedings of the AAAI Conference on Artificial
Intelligence*, volume 33, pages 3796–3803, 2019.

-
Cobbe et al. 2021

Karl W Cobbe, Jacob Hilton, Oleg Klimov, and John Schulman.

Phasic policy gradient.

In *International Conference on Machine Learning*, pages
2020–2027. PMLR, 2021.

-
Bellemare et al. 2013

Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.

The arcade learning environment: An evaluation platform for general
agents.

*Journal of Artificial Intelligence Research*, 47:253–279, 2013.

-
Machado et al. 2018a

Marlos C Machado, Marc G Bellemare, Erik Talvitie, Joel Veness, Matthew
Hausknecht, and Michael Bowling.

Revisiting the arcade learning environment: Evaluation protocols and
open problems for general agents.

*Journal of Artificial Intelligence Research*, 61:523–562, 2018a.

-
Hessel et al. 2018

Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski,
Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver.

Rainbow: Combining improvements in deep reinforcement learning.

In *Thirty-Second AAAI Conference on Artificial Intelligence*,
2018.

-
Dabney et al. 2018

Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos.

Implicit quantile networks for distributional reinforcement learning.

In *International conference on machine learning*, pages
1096–1105. PMLR, 2018.

-
Cobbe et al. 2020

Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman.

Leveraging procedural generation to benchmark reinforcement learning.

In *International conference on machine learning*, pages
2048–2056. PMLR, 2020.

-
Beattie et al. 2016

Charles Beattie, Joel Z Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright,
Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés,
Amir Sadik, et al.

Deepmind lab.

*arXiv preprint arXiv:1612.03801*, 2016.

-
Ye et al. 2021

Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, and Yang Gao.

Mastering atari games with limited data.

*Advances in Neural Information Processing Systems*,
34:25476–25488, 2021.

-
Micheli et al. 2022

Vincent Micheli, Eloi Alonso, and François Fleuret.

Transformers are sample efficient world models.

*arXiv preprint arXiv:2209.00588*, 2022.

-
Tassa et al. 2018

Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las
Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al.

Deepmind control suite.

*arXiv preprint arXiv:1801.00690*, 2018.

-
Yarats et al. 2021

Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto.

Mastering visual continuous control: Improved data-augmented
reinforcement learning.

*arXiv preprint arXiv:2107.09645*, 2021.

-
Osband et al. 2019

Ian Osband, Yotam Doron, Matteo Hessel, John Aslanides, Eren Sezener, Andre
Saraiva, Katrina McKinney, Tor Lattimore, Csaba Szepesvari, Satinder Singh,
et al.

Behaviour suite for reinforcement learning.

*arXiv preprint arXiv:1908.03568*, 2019.

-
Dizon-Paradis et al. 2023

Olivia Dizon-Paradis, Stephen Wormald, Daniel Capecci, Avanti Bhandarkar, and
Damon Woodard.

Investigating the practicality of existing reinforcement learning
algorithms: A performance comparison.

*Authorea Preprints*, 2023.

-
Hafner 2021

Danijar Hafner.

Benchmarking the spectrum of agent capabilities.

*arXiv preprint arXiv:2109.06780*, 2021.

-
Yarats et al. 2019

Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob
Fergus.

Improving sample efficiency in model-free reinforcement learning from
images.

*arXiv preprint arXiv:1910.01741*, 2019.

-
Reed et al. 2022

Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander
Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay,
Jost Tobias Springenberg, et al.

A generalist agent.

*arXiv preprint arXiv:2205.06175*, 2022.

-
Johnson et al. 2016

Matthew Johnson, Katja Hofmann, Tim Hutton, and David Bignell.

The malmo platform for artificial intelligence experimentation.

In *IJCAI*, pages 4246–4247. Citeseer, 2016.

-
Wang et al. 2023

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu,
Linxi Fan, and Anima Anandkumar.

Voyager: An open-ended embodied agent with large language models.

*arXiv preprint arXiv:2305.16291*, 2023.

-
Huang et al. 2022

Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto,
and Weixun Wang.

The 37 implementation details of proximal policy optimization.

*The ICLR Blog Track 2023*, 2022.

-
Hoffman et al. 2020

Matt Hoffman, Bobak Shahriari, John Aslanides, Gabriel Barth-Maron, Feryal
Behbahani, Tamara Norman, Abbas Abdolmaleki, Albin Cassirer, Fan Yang, Kate
Baumli, et al.

Acme: A research framework for distributed reinforcement learning.

*arXiv preprint arXiv:2006.00979*, 2020.

-
Schmitt et al. 2020

Simon Schmitt, Matteo Hessel, and Karen Simonyan.

Off-policy actor-critic with shared experience replay.

In *International Conference on Machine Learning*, pages
8545–8554. PMLR, 2020.

-
Schaul et al. 2015

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver.

Prioritized experience replay.

*arXiv preprint arXiv:1511.05952*, 2015.

-
Brock et al. 2021

Andy Brock, Soham De, Samuel L Smith, and Karen Simonyan.

High-performance large-scale image recognition without normalization.

In *International Conference on Machine Learning*, pages
1059–1071. PMLR, 2021.

-
Ziyin et al. 2020

Liu Ziyin, Zhikang T Wang, and Masahito Ueda.

Laprop: Separating momentum and adaptivity in adam.

*arXiv preprint arXiv:2002.04839*, 2020.

-
Kingma and Ba 2014

Diederik P Kingma and Jimmy Ba.

Adam: A method for stochastic optimization.

*arXiv preprint arXiv:1412.6980*, 2014.

-
Gruslys et al. 2017

Audrunas Gruslys, Will Dabney, Mohammad Gheshlaghi Azar, Bilal Piot, Marc
Bellemare, and Remi Munos.

The reactor: A fast and sample-efficient actor-critic agent for
reinforcement learning.

*arXiv preprint arXiv:1704.04651*, 2017.

-
Cho et al. 2014

Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
Fethi Bougares, Holger Schwenk, and Yoshua Bengio.

Learning phrase representations using rnn encoder-decoder for
statistical machine translation.

*arXiv preprint arXiv:1406.1078*, 2014.

-
Van Keirsbilck et al. 2019

Matthijs Van Keirsbilck, Alexander Keller, and Xiaodong Yang.

Rethinking full connectivity in recurrent neural networks.

*arXiv preprint arXiv:1905.12340*, 2019.

-
Machado et al. 2018b

Marlos C Machado, Marc G Bellemare, Erik Talvitie, Joel Veness, Matthew
Hausknecht, and Michael Bowling.

Revisiting the arcade learning environment: Evaluation protocols and
open problems for general agents.

*Journal of Artificial Intelligence Research*, 61:523–562, 2018b.

-
Espeholt et al. 2018

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom
Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al.

Impala: Scalable distributed deep-rl with importance weighted
actor-learner architectures.

*arXiv preprint arXiv:1802.01561*, 2018.

## Methods

### Baselines

We employ the Proximal Policy Optimization (PPO) algorithm [[5](#bib.bib5)], which has become a standard choice in the field, to compare Dreamer under fixed hyperparameters across all benchmarks. There are a large number of PPO implementations available publicly and they are known to substantially vary in task performance [[55](#bib.bib55)]. To ensure a comparison that is representative of the highest performance PPO can achieve under fixed hyperparameters across domains, we choose the high-quality PPO implementation available in the Acme framework [[56](#bib.bib56)] and select its hyperparameters in [Baselines](#Ax1.SSx1) following recommendations [[55](#bib.bib55), [13](#bib.bib13)] and additionally tune its epoch batch size to be large enough for complex environments [[42](#bib.bib42)], its learning rate, and its entropy scale. We match the discount factor to Dreamer because it works well across domains and is a common choice in the literature [[35](#bib.bib35), [8](#bib.bib8)]. We choose the IMPALA network architecture that we have found performed better than alternatives [[42](#bib.bib42)] and set the minibatch size to the largest possible for one A100 GPU. We verify the performance of our PPO implementation and hyperparameters on the ProcGen benchmark, where a highly tuned PPO implementation has been reported by the PPO authors [[37](#bib.bib37)]. We find that our implementation matches or slightly outperforms this performance reference.

*Table 1: PPO hyperparameters used across all benchmarks.*

Generated on Wed Apr 17 17:44:18 2024 by [LATExml](http://dlmf.nist.gov/LaTeXML/)