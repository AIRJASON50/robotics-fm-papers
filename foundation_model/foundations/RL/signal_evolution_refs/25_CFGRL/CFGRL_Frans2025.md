[# Diffusion Guidance Is a Controllable Policy Improvement Operator Kevin Frans∗ UC Berkeley kvfrans@berkeley.edu &Seohong Park∗ UC Berkeley &Pieter Abbeel UC Berkeley &Sergey Levine UC Berkeley ###### Abstract At the core of reinforcement learning is the idea of learning beyond the performance in the data. However, scaling such systems has proven notoriously tricky. In contrast, techniques from generative modeling have proven remarkably scalable and are simple to train. In this work, we combine these strengths, by deriving a direct relation between policy improvement and guidance of diffusion models. The resulting framework, CFGRL, is trained with the simplicity of supervised learning, yet can further improve on the policies in the data. On offline RL tasks, we observe a reliable trend—increased guidance weighting leads to increased performance. Of particular importance, CFGRL can operate without explicitly learning a value function, allowing us to generalize simple supervised methods (e.g., goal-conditioned behavioral cloning) to further prioritize optimality, gaining performance for “free” across the board. ## 1 Introduction **footnotetext: Equal contribution. Reinforcement learning (RL) provides a powerful framework for autonomous agents to attain strong performance by directly optimizing for task rewards. However, scaling up RL algorithms has proven notoriously challenging, particularly when using off-policy datasets. In contrast, modern generative modeling techniques have proven remarkably scalable, and have been used for related problems such as behavioral cloning [36](#bib.bib36), [5](#bib.bib5), [13](#bib.bib13)]. Can we leverage expressive generative modeling tools to derive simple, scalable RL approaches?

At the heart of RL is the idea of optimizing beyond the performance shown in the data. This is especially important when training agents from offline data that may have been collected by an exploratory or otherwise suboptimal policy. On one end of the spectrum, behavioral cloning methods are simple and can leverage stable generative modeling tools like diffusion [[[35](#bib.bib35), [47](#bib.bib47)]] and flow-matching, but are only as optimal as the data. On the other end, iterative RL techniques are in principle more optimal, but in practice can suffer from hyperparameter sensitivity and instability [[[56](#bib.bib56), [42](#bib.bib42), [23](#bib.bib23)]] that have made it challenging to scale to larger tasks.

In this work, we combine the strengths of both settings by developing a framework which is trained with the simplicity of behavioral cloning, yet can further improve on the data behaviors. We first define policies as products of two factors – a prior reference policy, and an “optimality” distribution. When the optimality distribution is proportional to a monotonically increasing function of advantage, we prove that the resulting product will be an improvement over the prior.

The key insight is that we can sample from this product distribution via techniques from diffusion modeling, and we can do so in a straightforward and controllable way. Rather than optimizing an optimality predictor, we instead learn an equivalent optimality-conditioned policy, as done in classifier-free guidance [[[34](#bib.bib34)]]. The prior and conditional factors can then by dynamically combined during sampling, allowing for a degree of policy improvement that can be controlled during test time, without the need for retraining.

Our framework, which we refer to as CFGRL, provides a principled and powerful connection between generative modeling and RL. Diffusion and flow-matching models already represent some of the most powerful approaches for imitation learning, but typically do not make use of guidance [[[5](#bib.bib5), [13](#bib.bib13)]]. CFGRL bridges a connection between guidance and traditional RL objectives—in fact, under certain choices, guided sampling results in a distribution that is equivalent to the solution of a KL-constrained policy improvement objective.

Particularly useful in practice, the CFGRL framework does not necessarily rely on explicitly learning a value function. Because of this property, CFGRL can be used as a drop-in replacement in settings such as goal-conditioned behavioral cloning, unlocking further policy improvement without additional training requirements.

We experimentally show applications of CFGRL both as an offline policy extraction method (when a value function is learned), and as a generalization of goal-conditioned behavioral cloning (where we avoid learning any value networks entirely). In the former, CFGRL provides a consistent improvement over standard weighted regression methods. Scaling trends highlight how increasing the guidance term results in stronger policies up to a divergence point. In the latter, CFGRL reliably outperforms goal-conditioned behavioral cloning baselines across the board on state-based, visual, and hierarchical settings, at times increasing success rates by a factor of two.

Our contributions are twofold. First, we propose a principled connection between diffusion model guidance and policy improvement in reinforcement learning. Second, we develop a set of simple practical algorithms that utilize the above connection to reliably improve policies in both the offline RL and goal-conditioned behavioral cloning settings.

Code to replicate experiments is released at [https://github.com/kvfrans/cfgrl](https://github.com/kvfrans/cfgrl).

## 2 Related work

Offline RL. Unlike standard RL, which involves exploring an environment, offline RL aims to learn a reward-maximizing policy solely from a previously collected dataset. The key challenge is to improve performance while preventing erroneous extrapolation when deviating too far from the dataset. Previous works target this problem from the value learning [[[42](#bib.bib42), [39](#bib.bib39), [75](#bib.bib75), [25](#bib.bib25), [3](#bib.bib3), [55](#bib.bib55)]], and policy extraction [[[74](#bib.bib74), [22](#bib.bib22), [70](#bib.bib70), [58](#bib.bib58), [59](#bib.bib59), [54](#bib.bib54), [8](#bib.bib8), [30](#bib.bib30)]] directions. Our work most closely relates to weighted regression [[[59](#bib.bib59), [54](#bib.bib54)]] and return-conditioned behavioral cloning [[[41](#bib.bib41), [11](#bib.bib11), [76](#bib.bib76)]] methods (of which goal-conditioned hindsight relabeling [[[4](#bib.bib4), [26](#bib.bib26), [18](#bib.bib18), [19](#bib.bib19)]] is a special case), due to their emphasis on simple supervised objectives. However, we instead frame the tradeoff between regularization and policy improvement in terms of guiding a diffusion model, providing a way to control this tradeoff during test time.

Diffusion and flow policies for RL.
Previous works have proposed diverse ways
to leverage the expressivity of iterative generative models,
such as diffusion [[[66](#bib.bib66), [35](#bib.bib35)]]
and flow models [[[45](#bib.bib45), [48](#bib.bib48), [2](#bib.bib2)]],
to enhance the capabilities of RL policies.
The main challenge with diffusion policy learning lies in *how to extract* [[[58](#bib.bib58)]]:
a diffusion policy to maximize the learned Q-function.
Prior works propose strategies
based on weighted regression [[[49](#bib.bib49), [37](#bib.bib37), [16](#bib.bib16), [80](#bib.bib80)]],
reparameterized gradients [[[72](#bib.bib72), [31](#bib.bib31), [17](#bib.bib17), [1](#bib.bib1), [79](#bib.bib79), [58](#bib.bib58)]],
rejection sampling [[[8](#bib.bib8), [30](#bib.bib30), [32](#bib.bib32)]],
and more [[[77](#bib.bib77), [40](#bib.bib40), [51](#bib.bib51), [10](#bib.bib10), [9](#bib.bib9), [61](#bib.bib61), [12](#bib.bib12), [43](#bib.bib43), [52](#bib.bib52), [21](#bib.bib21), [62](#bib.bib62)]].
Our method introduces classifier-free guidance as a policy extraction mechanism.
This has multiple benefits over previous approaches:
unlike reparameterized gradient-based methods, it does not require (potentially unstable) backpropagation through time [[[58](#bib.bib58)]];
unlike rejection sampling, it does not involve a costly sampling-then-filtering procedure.
Algorithmically, the closest work to ours is [Kuba et al. [[40](#bib.bib40)]], which also employs guidance over advantage-conditioned diffusion policies. Unlike this work, our framework supports a range of optimality functions rather than only $A=0$, and does not rely on further rejection sampling. Additionally, we do *not* necessarily require an explicit value function to perform policy improvement—in [Section 6](#S6), we further improve a goal-conditioned BC policy without additionally training value functions,
whereas all aforementioned techniques would require doing so.

## 3 Preliminaries

In the standard reinforcement learning (RL) setting, we define an environment as a Markov decision process with state space ${\mathcal{S}}$, action space ${\mathcal{A}}$, a transition function $p({\color[rgb]{0.6015625,0.6015625,0.6015625}s^{\prime}}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}a}%
):{\mathcal{S}}\times{\mathcal{A}}\to\Delta({\mathcal{S}})$, reward function $r({\color[rgb]{0.6015625,0.6015625,0.6015625}s}):{\mathcal{S}}\to{\mathbb{R}}$, and initial state distribution $p({\color[rgb]{0.6015625,0.6015625,0.6015625}s_{0}})\in\Delta({\mathcal{S}})$. We assume the state is fully observable. An agent is defined by a policy $\pi({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s}):{\mathcal{S}}\to\Delta(A)$ that represents a probability distribution over actions, and together with the environment, produces a distribution of state-action trajectories $\tau=(s_{0},a_{0},s_{1},a_{1},\ldots)$. The standard RL objective is to learn a parameterized policy $\pi_{\theta}$ that maximizes the expected sum of future discounted rewards along such trajectories:

$$J(\pi_{\theta})=\mathbb{E}_{\tau\sim p({\color[rgb]{% 0.6015625,0.6015625,0.6015625}\tau}\mid\pi_{\theta})}\sum_{t}\gamma^{t}r(s_{t}% ,a_{t}),$$ \tag{1}

where $p(\tau\mid\pi_{\theta})$ is defined as $p(s_{0})\prod_{t=0}^{\infty}p(s_{t+1}\mid s_{t},a_{t})\pi_{\theta}(a_{t}\mid s%
_{t})$.

A policy improvement operator is an update from
a reference policy $\hat{\pi}$ to a new policy $\pi$
such that the RL objective above does not decrease:
$J(\hat{\pi})\leq J(\pi)$.
This concept is formalized via $V_{\hat{\pi}}(s)$ and $Q_{\hat{\pi}}(s,a)$, which denote the discounted expected future reward under the reference policy starting from a given state or state-action pair, respectively [[[68](#bib.bib68)]]. The difference between these terms is the advantage, $A_{\hat{\pi}}(s,a)=Q_{\hat{\pi}}(s,a)-V_{\hat{\pi}}(s)$. A classic result shows that any update with non-negative advantage under the resulting state distribution, such that

$$\mathbb{E}_{(s,a)\sim p_{\pi}({\color[rgb]{0.6015625,0.6015625,0.6015625}s},{% \color[rgb]{0.6015625,0.6015625,0.6015625}a})}[A_{\hat{\pi}}(s,a)]\geq 0,$$ \tag{2}

with $p_{\pi}({\color[rgb]{0.6015625,0.6015625,0.6015625}s},{\color[rgb]{%
0.6015625,0.6015625,0.6015625}a})$ denoting the discounted state-action occupancy distribution,
results in policy improvement [[[63](#bib.bib63)]]. In practice, we require algorithms that operate over samples from a previous reference policy. Therefore, practical algorithms will approximate the above condition under the previous policy’s state distribution $p_{\hat{\pi}}({\color[rgb]{0.6015625,0.6015625,0.6015625}s})$, and aim to maximize:

$$\tilde{J}(\pi)=\mathbb{E}_{s\sim p_{\hat{\pi}}({\color[rgb]{% 0.6015625,0.6015625,0.6015625}s})}[\mathbb{E}_{a\sim\pi({\color[rgb]{% 0.6015625,0.6015625,0.6015625}a}\mid s)}[A_{\hat{\pi}}(s,a)]].$$ \tag{3}

Prior work has shown that, as long as the divergence between the reference and resulting policy is bounded, the approximate objective in [Equation 3](#S3.E3) provides a bound on the true objective in [Equation 2](#S3.E2), enabling monotonic incremental improvement [[[63](#bib.bib63)]].

To account for this divergence, it is common to utilize trust-region methods during policy improvement [[[63](#bib.bib63), [64](#bib.bib64)]]. While various divergence measures have been considered [[[65](#bib.bib65)]], a standard choice is to penalize the KL divergence between the reference and resulting policy, resulting in the following KL-penalized RL objective as parameterized by a constant $\beta$:

$$J(\pi_{\theta})=\mathbb{E}_{\tau\sim p({\color[rgb]{% 0.6015625,0.6015625,0.6015625}\tau}\mid\pi_{\theta})}\left[\sum_{t}\gamma^{t}r% (s_{t},a_{t})\right]-\beta\mathbb{E}_{s\sim p_{\pi}({\color[rgb]{% 0.6015625,0.6015625,0.6015625}s})}\left[D_{\mathrm{KL}}(\pi_{\theta}({\color[% rgb]{0.6015625,0.6015625,0.6015625}a}\mid s)\;\|\;\hat{\pi}({\color[rgb]{% 0.6015625,0.6015625,0.6015625}a}\mid s))\right].$$ \tag{4}

One strategy to optimize the above objective is via iteratively applying a policy gradient [[[69](#bib.bib69)]]; however, such methods require on-policy samples and can have high variance. The community has instead often relied on methods that resemble supervised learning, such as weighted regression [[[60](#bib.bib60), [59](#bib.bib59)]].
In the following sections, we introduce a policy improvement strategy based on generative modeling that maintains the simplicity of supervised learning, yet allows for controllable policy improvement.

## 4 Diffusion guidance is a controllable policy improvement operator

*Figure 1: While conditioning on optimality can create a baseline level of improvement, policies can be further improved by attenuating this conditioning. When $p(o\mid s,a)$ is proportional to a monotonically increasing function of advantage, then attenuation provably increases expected return, and this can be accomplished naturally with diffusion guidnace.*

In this work, we establish a connection between classifier-free guidance in diffusion modeling and policy improvement in RL. We then use this relation to develop a simple framework, CFGRL, which enables us to leverage stable and scalable diffusion model training methods to enable policies that can improve over the performance in the dataset. Specifically, CFGRL allows the degree of policy improvement to be controlled at test time, rather than having to be decided during training.
Additionally, CFGRL does not necessarily require the use of an explicit Q-function, allowing it to be used as a drop-in replacement that further improves on methods such as goal-conditioned behavioral cloning.

Product policies. We begin by parameterizing policies as a product of two factors—a reference policy, and an optimality function $f:\mathbb{R}\rightarrow\mathbb{R}$, which is conditional on advantage:

$$\pi(a\mid s)\propto\hat{\pi}(a\mid s)\;f(A(s,a)).$$ \tag{5}

The motivation behind this factorization is to frame improved policies in terms of a probabilistic adjustment to the current reference policy. As we will show later, diffusion guidance is naturally suited to sampling from such distributions.

We now show how product policies can be used as policy improvement operators. When $f$ fulfills certain common criteria, the resulting policy will provably be an improvement over the reference:

If $f$ is a non-negative, monotonically increasing function of $A_{\hat{\pi}}(s,a)$, then the product $\pi(a\mid s)\propto\hat{\pi}(a\mid s)f(A(s,a))$ is an improvement over $\hat{\pi}(a\mid s)$.

We provide a formalization and proof of this claim in the Appendix ([Theorem 1](#Thmtheorem1)),
which generalizes previous results with bandits [[[15](#bib.bib15), [60](#bib.bib60)]] and exponentiated advantages [[[59](#bib.bib59)]].

The above finding reveals a simple path towards policy improvement. If we can sample from a properly weighted product policy, then the resulting policy will achieve a higher expected return then the reference.

Crucially, we can control the degree of improvement by sampling from an attenuated optimality function. To do so, we consider product policies where the optimality functions are exponentiated:

Let $0\leq w_{1}

The proof of the above theorem is again provided in the Appendix ([Theorem 2](#Thmtheorem2)). Of course, there is no free lunch. While a higher exponent leads to an improved policy in terms of $A_{\hat{\pi}}$, the resulting policy is also further deviated from the reference. Thus, the empirical performance of an over-adjusted policy may suffer due to a distribution shift.

This tradeoff between adhering to the reference policy and maximizing return can be understood via the KL-regularized RL objective in [Equation 4](#S3.E4). Notably, the solutions to the mentioned objective naturally form a set of product policies:

The policies that maximize [Equation 4](#S3.E4) under a given KL-penalty $\beta$ take the form of

Equivalently, the $\beta$ term can be folded into the exponent, e.g. $\pi(a\mid s)\propto\hat{\pi}(a\mid s)\exp(A(s,a)/\beta)$. This condensed objective has been used in prior works [[[60](#bib.bib60), [59](#bib.bib59)]] to directly learn $\pi$; however, in their methods the $\beta$ hyperparameter must be specified ahead of time. As shown in the next sections, we will instead develop a framework where the product factors are represented independently, allowing their composition to be freely controlled during evaluation time.

### 4.1 Composing factors via diffusion guidance

Having understood that product policies are a natural way to induce policy improvement, we can now instantiate such policies using machinery from diffusion modeling. We start by casting the optimality function as a binary random variable111Slightly abusing notation, we abbreviate $o=1$ as $o$ when it is clear from context. $o=\varnothing$ represents an unconditional case where $o$ is not specified, i.e., the union of $o=0$ and $o=1$. $o\in\{\varnothing,0,1\}$ whose likelihood is defined via $f$:

$$p(o\mid s,a)=f(A(s,a))/Z(s)$$ \tag{7}

where $Z(s)=\int f(A(s,a^{\prime}))da^{\prime}$ is a state-dependent normalization factor. We will not require estimating $Z(s)$. The product policy from [Equation 5](#S4.E5) can now be equivalently defined as:

$$\pi(a\mid s)\propto\hat{\pi}(a\mid s)\;p(o\mid s,a).$$ \tag{8}

Recall that diffusion models implicitly model a distribution by learning its normalized score function, i.e., the gradient of log-likelihood under that distribution [[[67](#bib.bib67)]].
Score functions have the useful property that for product distributions, they are additively composable. As such, the score of the product policy above can be represented as the sum of two factors:

$$\nabla_{a}\log\pi(a\mid s)=\nabla_{a}\log\hat{\pi}(a\mid s)+\nabla_{a}\;\log p% (o\mid s,a).$$ \tag{9}

Avoiding an explicit optimality predictor. In many cases, we would rather avoid explicitly learning the $p(o\mid s,a)$ distribution. For one thing, optimality must hold a valid probability distribution, and calculating the normalization term $Z(s)$ can be tricky. Secondly, explicitly backpropagating through a neural network predictor may result in adversarial gradient attacks [[[28](#bib.bib28)]] especially at out-of-distribution actions [[[39](#bib.bib39), [42](#bib.bib42)]]. We also note that if one wanted to learn an optimality predictor, it would need to remain accurate under the support of partially-noised actions.

Instead, we can utilize an insight from classifier-free guidance [[[34](#bib.bib34)]], and use Bayes’ rule to invert the optimality distribution into an optimality-conditioned policy score function:

$$\nabla_{a}\log\pi(a\mid s)=\nabla_{a}\log\hat{\pi}(a\mid s)+(\nabla_{a}\log% \hat{\pi}(a\mid s,o)-\nabla_{a}\log\hat{\pi}(a\mid s)).$$ \tag{10}

With the above form, both factors can be unified into a single conditional policy. We can represent both factors with the same neural network, and train via a straightforward diffusion modeling objective.

Guidance controls the attenuation of optimality. A key benefit of defining the product distribution in terms of composable factors is that the ratio between these factors can be dynamically controlled. Introducing a guidance weight $w$, the score function

$$\nabla_{a}\log\hat{\pi}(a\mid s)+w\;(\nabla_{a}\log\hat{\pi}(a\mid s,o)-\nabla% _{a}\log\hat{\pi}(a\mid s))$$ \tag{11}

corresponds to the attenuated distribution

$$\pi(a\mid s)\propto\hat{\pi}(a\mid s)\;p(o\mid s,a)^{w},\;\text{and % equivalently}\;\;\hat{\pi}(a\mid s)f(A(s,a))^{w}.$$ \tag{12}

Recall that in [Remark 2](#Thmremark2), we showed that increasing $w$ results in further policy improvement. This implies a simple yet crucial relationship—by controlling the guidance weight during sampling, we can sample from product policies that controllably improve on the reference policy (at the cost of adherence to the prior). The tradeoff is a key hyperparameter to tune in offline RL [[[56](#bib.bib56)]], and often requires many sweeps. In contrast, with CFGRL, this sweep can be performed at test-time over a single network, without the need for retraining.

### 4.2 Training and sampling with CFGRL

*Algorithm 1 CFGRL Training*

*Algorithm 2 CFGRL Sampling*

We instantiate a single diffusion network to serve as both the conditional and unconditional policy. For simplicity, we adopt the flow-matching framework for training. While flow networks predict velocity rather than score, previous works have shown they retain similar properties in practice [[[24](#bib.bib24), [46](#bib.bib46)]]. The policy is modeled by a velocity field $v_{\theta}$ conditioned on a partially-noised action $a_{t}$, along with a noise scale $t$, the current state $s$, as well as the previously defined optimality variable $o\in\{\varnothing,0,1\}$. This network is trained via the following loss function:

$$\mathcal{L}(\theta)=\mathbb{E}_{s,a\sim D}\left[\|v_{\theta}(a_{t},t,s,o)-(a-a% _{0})\|^{2}\right]\qquad\text{where}\qquad a_{t}=(1-t)a_{0}+ta,$$ \tag{13}

and where the noise scale $t$ is sampled uniformly between $[0,1]$, and $a_{0}\sim N(0,1)$ is Gaussian noise.

## 5 CFGRL improves over weighted policy extraction in offline RL

A common approach to offline RL is to learn a state-action value function $Q_{\theta}(s,a)$, and then extract a policy from this value function via a regularized policy extraction method that stays close to the behavior policy while maximize the value function. This regularization is critical to avoid out-of-distribution actions for which the value function is likely to overestimate the value [[[59](#bib.bib59), [39](#bib.bib39)]]. Weighted regression methods are a particularly simple class of methods for doing this, with advantage-weighted regression (AWR) or its variants being a common choice in recent work [[[59](#bib.bib59), [54](#bib.bib54), [73](#bib.bib73)]]. AWR is trained in a fully supervised manner, and does not require querying the values of non-dataset actions, with the training objective given by

$$J_{\mathrm{AWR}}(\theta)=E_{(s,a)\sim D}\left[\log\pi_{\theta}(a\mid s)\exp(A(% s,a)\times(1/\beta))\right],$$ \tag{14}

where $A(s,a)=Q(s,a)-V(s)$ is calculated as the difference of learned $Q_{\theta}(s,a)$ and $V_{\theta}(s)$ networks, and $\beta$ is a temperature hyperparameter.

However, the weights in AWR can become peaked, such that much of the data in training is not used effectively, resulting in a weak learning signal. This phenomenon is shown in [Figure 2](#S5.F2) (left), which plots the magnitudes of per-element gradients within an AWR batch. Notably, the magnitudes are dominated by a few outlier state-action pairs that hold particularly high weights. In this case, the rest of the batch is effectively ignored, and AWR derives the gradient only from a small subset of the available data.

![Figure](x2.png)

*Figure 2: Weighted regression methods result in uneven gradient magnitudes within a batch. This can limit the effective signal that each batch provides. In contrast, CFGRL uses a simple conditional diffusion modeling loss with even weighting.*

We now show how we can instead train with CFGRL to alleviate this issue. Specifically, we will instantiate CFGRL with a particularly simple optimality criteria:

| | $$o=\begin{cases}1&\text{if }A(s,a)\geq 0\\ 0&\text{if }A(s,a)

| Task | AWR | CFGRL |
|---|---|---|
| walker-stand | $603$ $\pm 8$ | $\mathbf{782}$ $\pm 8$ |
| walker-walk | $444$ $\pm 4$ | $\mathbf{608}$ $\pm 32$ |
| walker-run | $247$ $\pm 10$ | $\mathbf{282}$ $\pm 6$ |
| quadruped-walk | $\mathbf{776}$ $\pm 15$ | $\mathbf{762}$ $\pm 25$ |
| quadruped-run | $485$ $\pm 7$ | $\mathbf{571}$ $\pm 25$ |
| cheetah-run | $168$ $\pm 7$ | $\mathbf{216}$ $\pm 15$ |
| cheetah-run-backward | $146$ $\pm 8$ | $\mathbf{262}$ $\pm 26$ |
| jaco-reach-top-right | $33$ $\pm 2$ | $\mathbf{72}$ $\pm 6$ |
| jaco-reach-top-left | $30$ $\pm 8$ | $\mathbf{46}$ $\pm 6$ |

*Table 2: OGBench results.*

| Task | AWR | CFGRL |
|---|---|---|
| pointmaze-large-navigate | $70$ $\pm 25$ | $\mathbf{100}$ $\pm 0$ |
| pointmaze-teleport-navigate | $3$ $\pm 7$ | $\mathbf{57}$ $\pm 7$ |
| antmaze-large-navigate | $\mathbf{50}$ $\pm 9$ | $20$ $\pm 9$ |
| antmaze-teleport-navigate | $22$ $\pm 19$ | $\mathbf{30}$ $\pm 22$ |
| humanoidmaze-large-navigate | $\mathbf{3}$ $\pm 4$ | 00 $\pm 0$ |
| antsoccer-arena-navigate | $7$ $\pm 0$ | $\mathbf{20}$ $\pm 5$ |
| cube-single-play | $\mathbf{85}$ $\pm 8$ | $\mathbf{82}$ $\pm 3$ |
| scene-play | $\mathbf{18}$ $\pm 3$ | $17$ $\pm 9$ |
| puzzle-3x3-play | $\mathbf{3}$ $\pm 7$ | $\mathbf{3}$ $\pm 4$ |

The training procedure with CFGRL is simple. Given an state-action sample $(s,a)\sim D$, we label that pair with $o\in\{0,1\}$ according the above criteria. We then train with the standard conditional diffusion-modeling loss as done in [Section 4.2](#S4.SS2). Notably, there is no weighting term used in training. As such, gradients within each batch remain reasonably distributed, as shown in [Figure 2](#S5.F2) (right).

There is a suggestive similarity between the temperature hyperparameter in AWR and the guidance weight in CFGRL. In fact, both these parameters play the same role—they control the tradeoff between adherence to a reference policy and maximization of rewards. However, with AWR, the temperature must be chosen beforehand and is folded into the optimization objective. CFGRL keeps the prior policy and the optimality-conditioned policy separate, and only combines them during sampling. Thus, this tradeoff can be adjusted without retraining when using CFGRL, making it easy to find the best value.

Furthermore, we will see that the guidance term in CFGRL is empirically more effective than the temperature of AWR. Note that while the absolute scale of $w$ and $(1/\beta)$ can vary, they have a proportional relationship and share the same base case at $w=(1/\beta)=0$, in which case the resulting policy simply mimics the dataset policy. We examine the scaling of performance over different guidance and temperature values in [Figure 3](#S5.F3). For AWR, performance saturates around a temperature of $(1/\beta)=10$. In contrast, guidance continues to improve beyond performance beyond this point, displaying a longer-lasting trend.

Experimental comparison. We further establish the comparison between AWR and CFGRL on $9$ tasks from the ExORL benchmark [[[78](#bib.bib78)]], which contains data collected by an exploratory agent, along with $9$ single-task environments from the OGBench suite [[[57](#bib.bib57)]]. In all experiments, we use the same state-action value function trained via implicit Q-learning for both methods [[[39](#bib.bib39)]], which notably does not require a policy in the loop to learn and is therefore independent of the extraction method that is used downstream. For AWR, we sweep over $1/\beta$ in the set of $\{1,3,10,30\}$, and for CFGRL, we sweep over $w\in\{1,1.25,1.5,2.0,3.0\}$. Results are presented in [Tables 1](#S5.T1) and [2](#S5.T2), and are averaged over $4$ seeds. On a strong majority of tasks, CFGRL achieves a better final performance than AWR. This indicates that policy extraction with CFGRL, which also corresponds to a simple generative modeling objective, is more effective than the widely used AWR method.

## 6 CFGRL unlocks hidden gains in goal-conditioned behavioral cloning

While the overall CFCRL framework is general, it is particularly appealing in the special-case of goal-conditioned RL, where it is common to use a simple (though crude) approximation that bypasses the need for a learned value estimator [[[19](#bib.bib19), [26](#bib.bib26), [6](#bib.bib6)]]. In such settings, the objective to find a goal-conditioned policy $\pi({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}g}%
):{\mathcal{S}}\times{\mathcal{S}}\to\Delta({\mathcal{A}})$ that maximizes likelihood of reaching the goal:

$$J(\pi)=\mathbb{E}_{\tau\sim p({\color[rgb]{0.6015625,0.6015625,0.6015625}\tau}% \mid\pi),\ g\sim p({\color[rgb]{0.6015625,0.6015625,0.6015625}g})}\left[\sum_{% t}\gamma^{t}\delta_{g}(s_{t})\right],$$ \tag{16}

where $p({\color[rgb]{0.6015625,0.6015625,0.6015625}g})\in\Delta({\mathcal{S}})$ is a goal distribution
and reward is $\delta_{g}$, i.e. the Dirac delta “function” at $g$.222This “function” is well-defined in a discrete state space, but requires a measure-theoretic formulation [[[71](#bib.bib71)]] to be well-defined in a continuous state space, which we omit for simplicity.
While in principle, we can optimize the above objective with a full RL procedure, an often-used simplification is to perform goal-conditioned behavioral cloning (GCBC), which maximizes:

$$J_{\mathrm{GCBC}}(\theta)=\mathbb{E}_{\begin{subarray}{c}(s_{t},a_{t})\sim{% \mathcal{D}},\ \Delta\sim\mathrm{Geom}(1-\gamma)\end{subarray}}[\log\pi_{% \theta}(a_{t}\mid s_{t},s_{t+\Delta})],$$ \tag{17}

where $\mathrm{Geom}(1-\gamma)$ denotes the Geometric distribution with parameter $1-\gamma$,
and we often denote $s_{t+\Delta}$ as $g$.
GCBC can be seen as a special case of conditional behavioral cloning methods that filter for actions that empirically reach a future goal.
While the GCBC objective above is simple,
it does not converge to the optimal goal-reaching policy, especially when the dataset is suboptimal [[[19](#bib.bib19), [27](#bib.bib27)]]

Generalizing past naïve GCBC.
Based on our CFGRL framework in [Section 4](#S4),
we now introduce a method to further improve GCBC policies
without training value functions*.
The key insight is that, since CFGRL enables one step of policy improvement over the base policy,
applying CFGRL with guidance on the goal $g$ will produce a policy that is better than the standard GCBC policy.
While this policy is still not as optimal as the full GCRL solution (which in general would require many steps of policy improvement),
prior work has shown that even one step of policy improvement can lead to significant performance gains in a range of RL settings [[[7](#bib.bib7), [20](#bib.bib20)]].
Indeed, our experiments will confirm that CFGRL enables a significant increase in performance over standard GCBC.

We begin by noting that the GCBC policy that maximizes [Equation 17](#S6.E17) is given as follows [[[20](#bib.bib20)]]:

$$\pi(a\mid s,g)=\frac{\hat{\pi}(a\mid s)p^{\gamma}(g\mid s,a)}{p^{\gamma}(g\mid s% )}\propto\hat{\pi}(a\mid s)\;Q_{\hat{\pi}}(s,a,g),$$ \tag{18}

where $p^{\gamma}$ denotes the distribution induced by the Geometric goal sampling procedure.

Next, we note that the second factor in [Equation 18](#S6.E18) satisfies the conditions of [Remark 1](#Thmremark1) (i.e., a bounded, non-negative, non-decreasing function in $A_{\hat{\pi}}(s,a)$).333Specifically, we set $f$ as $Q_{\hat{\pi}}(s,a,g)$, which is a bounded, non-negative, non-decreasing function of $A_{\hat{\pi}}(s,a,g)$. This can also be understood as defining $p(o\mid g,s,a)\propto p^{\gamma}(g\mid s,a)$, i.e., an action’s optimality is proportional to the likelihood of reaching the goal in the discounted future.
Thus, we can invoke [Remark 2](#Thmremark2) to achieve a policy improvement.
Specifically,
the resulting policy $\pi(a\mid s)\propto\hat{\pi}(a\mid s)\;p(g\mid s,a)^{w}$ under any exponent $w\geq 1$ will result in a policy improvement, and we can sample from the attenuated second factor via guidance:

$$\nabla_{a}\log\hat{\pi}(a\mid s)+w\;(\nabla_{a}\log\pi(a\mid s,g)-\nabla_{a}% \log\hat{\pi}(a\mid s)).$$ \tag{19}

The above CFGRL interpretation reveals a simple recipe for improving beyond the GCBC policy. Specifically, naive GCBC results in a product policy that implicitly assumes a weighting of $w=1$. Guidance allows us to instead consider $w>1$, leading to improved performance.

We emphasize the practical benefits of this setup—improvement can be gained for “free” relative to a standard GCBC setup. The components of [Equation 19](#S6.E19) are simply the original goal-conditioned BC policy $\pi({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}g})$ along with an unconditional BC policy $\hat{\pi}({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s})$. We do *not* require any additional techniques such as training an explicit value function, or sampling further on-policy actions.

### 6.1 Experimental results

*Figure 4: CFGRL can extrapolate beyond the GCBC policy, unlocking further performance gains. In fact, GCBC is implicitly a special case of the CFGRL policy where $w=1$. By instead considering $w>1$, the resulting policy is an improvement over the original. We show that performance steadily increases with $w$ on a range of environments.*

Tasks.
We empirically verify effectiveness over $17$ state-based and $7$ pixel-based goal-conditioned RL tasks from the OGBench task suite [[[57](#bib.bib57)]] ([Figure 5](#A2.F5)).
These tasks span a variety of robotic navigation and manipulation domains,
including whole-body humanoid control, maze navigation, sequential object manipulation, and combinatorial puzzle solving.
Among them, the tasks prefixed with “visual-” require image-based control.

Methods.
In this experiment, we consider four imitation learning baselines that do not involve value learning
(recall that CFGRL also does not train a value function):
(1) BC, (2) flow BC [[[13](#bib.bib13)]], (3) goal-conditioned BC (GCBC) [[[26](#bib.bib26)]], and (4) flow GCBC.
BC trains an (unconditional) behavioral cloning policy,
and GCBC trains a goal-conditioned policy with [Equation 17](#S6.E17).
Flow BC and flow GCBC maximize the same objectives, but with flow policies [[[13](#bib.bib13), [5](#bib.bib5)]].
In the CFGRL framework, flow BC corresponds to CFGRL with $w=0$ and flow GCBC corresponds to $w=1$.

In addition to the five “flat” methods above, we also consider hierarchical* behavioral cloning [[[29](#bib.bib29), [50](#bib.bib50)]] on state-based tasks,
where we train both a high-level policy $\pi^{h}({\color[rgb]{0.6015625,0.6015625,0.6015625}\ell}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}g}%
):{\mathcal{S}}\times{\mathcal{S}}\to\Delta({\mathcal{S}})$ that outputs subgoals $\ell$
and a low-level policy $\pi^{\ell}({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}%
\ell}):{\mathcal{S}}\times{\mathcal{S}}\to\Delta({\mathcal{A}})$ that takes subgoals and outputs actions.
In this setting, we can apply CFGRL to each level’s GCBC objective to enhance the optimality of both policies.
We call this variant hierarchical CFGRL (HCFGRL).
As baselines, we consider hierarchical GCBC [[[29](#bib.bib29), [50](#bib.bib50)]] and flow hierarchical GCBC,
which trains Gaussian and flow policies, respectively.

*Table 3: Improving on GCBC. CFGRL consistently improves performance over GCBC across the board. Numbers at or above the $95\%$ of the best performance in each category are boldfaced, as in OGBench [[[57](#bib.bib57)]].*

Results. [Table 3](#S6.T3) presents evaluation results averaged over $8$ averaged over $8$ seeds for state-based tasks and $4$ seeds for pixel-based tasks,
denoting standard deviations after the $\pm$ symbols.
The results suggest that CFGRL consistently outperforms all four baselines on most of the tasks,
even with a single fixed value of the guidance strength ($w=3$).
Notably, on some tasks (e.g., pointmaze-giant and visual-cube-single),
CFGRL achieves more than $3\times$ the success of the strongest baseline.
We again emphasize that this improvement is achieved simply by contrasting the prior and GCBC policies, *without* training a value function.
As in [Section 5](#S5), we also measure how the performance of CFGRL varies with different values of CFG weights $w$.
We present the results on four tasks in [Figure 4](#S6.F4) (see [Figure 6](#A2.F6) for the full results),
which shows that the performance generally improves as $w$ increases, as predicted by [Remark 2](#Thmremark2).

## 7 Discussion and Conclusion

In this work, we introduced a principled connection between diffusion guidance and policy improvement in RL. Using this connection, we derive a simple framework that combines the simplicity of existing generative modeling objectives with the policy improvement capabilities of RL. We then instantiate this framework for policy extraction in offline RL, and for improving over goal-conditioned BC (GCBC) in the goal-conditioned setting. We show that CFGRL improves over the widely used AWR approach in the offline RL setting, and achieves a substantial improvement over GCBC in the goal-conditioned setting, while maintaining the simplicity of these prior methods.

Limitations. Our method does not claim to replace full RL procedures—we assume a given value function and do not make any prescriptions about how to train it. In our experiments, CFGRL takes the place of prior supervised learning methods for policy extraction, maintaining their simplicity and stability. However, more advanced policy extraction methods and online RL techniques, such as policy gradients [[[44](#bib.bib44), [64](#bib.bib64)]], could provide for stronger extrapolation. By itself, CFGRL does not represent a state-of-the-art RL algorithm, but rather an additional tool in the algorithm designer’s toolbox that can take the place of policy extraction methods such as AWR, as well as a theoretical connection that we hope will inspire future work.

How should I use CFGRL in practice? The simplicity of our method lends to a plug-and-play approach. If you are already training a goal-conditioned policy, or are conditioning on outcome in any way, then try using guidance as we do in the paper. If you can, do a sweep over $w$ weights, which conveniently can be done without retraining the model.

Code. The presented experiments can be run via [https://github.com/kvfrans/cfgrl](https://github.com/kvfrans/cfgrl).

## 8 Acknowledgements

This work was supported in part by an NSF Fellowship for KF, under grant No. DGE 2146752, and by the Korea Foundation for Advanced Studies (KFAS) for SP. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the NSF. PA holds concurrent appointments as a Professor at UC Berkeley and as an Amazon Scholar. This paper describes work performed at UC Berkeley and is not associated with Amazon. This research used the Savio computational cluster resource provided by the Berkeley Research Computing program at UC Berkeley.

## References

-
Ada et al. [2024]

Suzan Ece Ada, Erhan Oztop, and Emre Ugur.

Diffusion policies for out-of-distribution generalization in offline reinforcement learning.

*IEEE Robotics and Automation Letters (RA-L)*, 9:3116–3123, 2024.

-
Albergo and Vanden-Eijnden [2023]

Michael S Albergo and Eric Vanden-Eijnden.

Building normalizing flows with stochastic interpolants.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
An et al. [2021]

Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song.

Uncertainty-based offline reinforcement learning with diversified q-ensemble.

In *Neural Information Processing Systems (NeurIPS)*, 2021.

-
Andrychowicz et al. [2017]

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba.

Hindsight experience replay.

In *Neural Information Processing Systems (NeurIPS)*, 2017.

-
Black et al. [2024a]

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al.

$\pi_{0}$: A vision-language-action flow model for general robot control.

*ArXiv*, abs/2410.24164, 2024a.

-
Black et al. [2024b]

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine.

Zero-shot robotic manipulation with pretrained image-editing diffusion models.

In *International Conference on Learning Representations (ICLR)*, 2024b.

-
Brandfonbrener et al. [2021]

David Brandfonbrener, William F. Whitney, Rajesh Ranganath, and Joan Bruna.

Offline rl without off-policy evaluation.

In *Neural Information Processing Systems (NeurIPS)*, 2021.

-
Chen et al. [2023]

Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, and Jun Zhu.

Offline reinforcement learning via high-fidelity generative behavior modeling.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Chen et al. [2024a]

Huayu Chen, Cheng Lu, Zhengyi Wang, Hang Su, and Jun Zhu.

Score regularized policy optimization through diffusion behavior.

In *International Conference on Learning Representations (ICLR)*, 2024a.

-
Chen et al. [2024b]

Huayu Chen, Kaiwen Zheng, Hang Su, and Jun Zhu.

Aligning diffusion behaviors with q-functions for efficient continuous control.

In *Neural Information Processing Systems (NeurIPS)*, 2024b.

-
Chen et al. [2021]

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, P. Abbeel, A. Srinivas, and Igor Mordatch.

Decision transformer: Reinforcement learning via sequence modeling.

In *Neural Information Processing Systems (NeurIPS)*, 2021.

-
Chen et al. [2024c]

Tianyu Chen, Zhendong Wang, and Mingyuan Zhou.

Diffusion policies creating a trust region for offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2024c.

-
Chi et al. [2023]

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song.

Diffusion policy: Visuomotor policy learning via action diffusion.

In *Robotics: Science and Systems (RSS)*, 2023.

-
da Silva [2023]

Bruno C. da Silva.

Reinforcement learning lectures notes, 2023.

URL [https://people.cs.umass.edu/~bsilva/courses/CMPSCI_687/Fall2023/Lecture_Notes_v1.0_687_F23.pdf](https://people.cs.umass.edu/~bsilva/courses/CMPSCI_687/Fall2023/Lecture_Notes_v1.0_687_F23.pdf).

-
Dayan and Hinton [1997]

Peter Dayan and Geoffrey E Hinton.

Using expectation-maximization for reinforcement learning.

*Neural Computation*, 9:271–278, 1997.

-
Ding et al. [2024]

Shutong Ding, Ke Hu, Zhenhao Zhang, Kan Ren, Weinan Zhang, Jingyi Yu, Jingya Wang, and Ye Shi.

Diffusion-based reinforcement learning via q-weighted variational policy optimization.

In *Neural Information Processing Systems (NeurIPS)*, 2024.

-
Ding and Jin [2024]

Zihan Ding and Chi Jin.

Consistency models as a rich and efficient policy class for reinforcement learning.

In *International Conference on Learning Representations (ICLR)*, 2024.

-
Emmons et al. [2022]

Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, and Sergey Levine.

Rvs: What is essential for offline rl via supervised learning?

In *International Conference on Learning Representations (ICLR)*, 2022.

-
Eysenbach et al. [2022a]

Benjamin Eysenbach, Soumith Udatha, Russ R Salakhutdinov, and Sergey Levine.

Imitating past successes can be very suboptimal.

In *Neural Information Processing Systems (NeurIPS)*, 2022a.

-
Eysenbach et al. [2022b]

Benjamin Eysenbach, Tianjun Zhang, Ruslan Salakhutdinov, and Sergey Levine.

Contrastive learning as goal-conditioned reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2022b.

-
Fang et al. [2025]

Linjiajie Fang, Ruoxue Liu, Jing Zhang, Wenjia Wang, and Bingyi Jing.

Diffusion actor-critic: Formulating constrained policy iteration as diffusion noise regression for offline reinforcement learning.

In *International Conference on Learning Representations (ICLR)*, 2025.

-
Fujimoto and Gu [2021]

Scott Fujimoto and Shixiang Shane Gu.

A minimalist approach to offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2021.

-
Fujimoto et al. [2018]

Scott Fujimoto, Herke van Hoof, and David Meger.

Addressing function approximation error in actor-critic methods.

In *International Conference on Machine Learning (ICML)*, 2018.

-
Gao et al. [2024]

Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, and Tim Salimans.

Diffusion meets flow matching: Two sides of the same coin, 2024.

URL [https://diffusionflow.github.io/](https://diffusionflow.github.io/).

-
Garg et al. [2023]

Divyansh Garg, Joey Hejna, Matthieu Geist, and Stefano Ermon.

Extreme q-learning: Maxent rl without entropy.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Ghosh et al. [2021]

Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Devin, Benjamin Eysenbach, and Sergey Levine.

Learning to reach goals via iterated supervised learning.

In *International Conference on Learning Representations (ICLR)*, 2021.

-
Ghugare et al. [2024]

Raj Ghugare, Matthieu Geist, Glen Berseth, and Benjamin Eysenbach.

Closing the gap between td learning and supervised learning–a generalisation point of view.

In *International Conference on Learning Representations (ICLR)*, 2024.

-
Goodfellow et al. [2015]

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.

Explaining and harnessing adversarial examples.

In *International Conference on Learning Representations (ICLR)*, 2015.

-
Gupta et al. [2019]

Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman.

Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning.

In *Conference on Robot Learning (CoRL)*, 2019.

-
Hansen-Estruch et al. [2023]

Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine.

Idql: Implicit q-learning as an actor-critic method with diffusion policies.

*ArXiv*, abs/2304.10573, 2023.

-
He et al. [2023]

Longxiang He, Li Shen, Linrui Zhang, Junbo Tan, and Xueqian Wang.

Diffcps: Diffusion model based constrained policy search for offline reinforcement learning.

*ArXiv*, abs/2310.05333, 2023.

-
He et al. [2024]

Longxiang He, Li Shen, Junbo Tan, and Xueqian Wang.

Aligniql: Policy alignment in implicit q-learning through constrained optimization.

*ArXiv*, abs/2405.18187, 2024.

-
Hendrycks and Gimpel [2016]

Dan Hendrycks and Kevin Gimpel.

Gaussian error linear units (gelus).

*ArXiv*, abs/1606.08415, 2016.

-
Ho and Salimans [2022]

Jonathan Ho and Tim Salimans.

Classifier-free diffusion guidance.

*ArXiv*, abs/2207.12598, 2022.

-
Ho et al. [2020]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

In *Neural Information Processing Systems (NeurIPS)*, 2020.

-
Janner et al. [2022]

Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine.

Planning with diffusion for flexible behavior synthesis.

In *International Conference on Machine Learning (ICML)*, 2022.

-
Kang et al. [2023]

Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, and Shuicheng Yan.

Efficient diffusion policies for offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2023.

-
Kingma and Ba [2015]

Diederik P. Kingma and Jimmy Ba.

Adam: A method for stochastic optimization.

In *International Conference on Learning Representations (ICLR)*, 2015.

-
Kostrikov et al. [2022]

Ilya Kostrikov, Ashvin Nair, and Sergey Levine.

Offline reinforcement learning with implicit q-learning.

In *International Conference on Learning Representations (ICLR)*, 2022.

-
Kuba et al. [2023]

Jakub Grudzien Kuba, Pieter Abbeel, and Sergey Levine.

Advantage-conditioned diffusion: Offline rl via generalization.

*OpenReview*, 2023.

-
Kumar et al. [2019]

Aviral Kumar, Xue Bin Peng, and Sergey Levine.

Reward-conditioned policies.

*ArXiv*, abs/1912.13465, 2019.

-
Kumar et al. [2020]

Aviral Kumar, Aurick Zhou, G. Tucker, and Sergey Levine.

Conservative q-learning for offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2020.

-
Li et al. [2024]

Zechu Li, Rickmer Krohn, Tao Chen, Anurag Ajay, Pulkit Agrawal, and Georgia Chalvatzaki.

Learning multimodal behaviors from scratch with diffusion policy gradient.

In *Neural Information Processing Systems (NeurIPS)*, 2024.

-
Lillicrap et al. [2016]

Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.

Continuous control with deep reinforcement learning.

In *International Conference on Learning Representations (ICLR)*, 2016.

-
Lipman et al. [2023]

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

Flow matching for generative modeling.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Lipman et al. [2024a]

Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat.

Flow matching guide and code.

*ArXiv*, abs/2412.06264, 2024a.

-
Lipman et al. [2024b]

Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat.

Flow matching guide and code.

*arXiv preprint arXiv:2412.06264*, 2024b.

-
Liu et al. [2023]

Xingchao Liu, Chengyue Gong, and Qiang Liu.

Flow straight and fast: Learning to generate and transfer data with rectified flow.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Lu et al. [2023]

Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu.

Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning.

In *International Conference on Machine Learning (ICML)*, 2023.

-
Lynch et al. [2019]

Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and Pierre Sermanet.

Learning latent plans from play.

In *Conference on Robot Learning (CoRL)*, 2019.

-
Mao et al. [2024]

Liyuan Mao, Haoran Xu, Xianyuan Zhan, Weinan Zhang, and Amy Zhang.

Diffusion-dice: In-sample diffusion guidance for offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2024.

-
Mark et al. [2024]

Max Sobol Mark, Tian Gao, Georgia Gabriela Sampaio, Mohan Kumar Srirama, Archit Sharma, Chelsea Finn, and Aviral Kumar.

Policy agnostic rl: Offline rl and online rl fine-tuning of any class and backbone.

*ArXiv*, abs/2412.06685, 2024.

-
Misra [2020]

Diganta Misra.

Mish: A self regularized non-monotonic activation function.

In *British Machine Vision Association (BMVC)*, 2020.

-
Nair et al. [2020]

Ashvin Nair, Murtaza Dalal, Abhishek Gupta, and Sergey Levine.

Accelerating online reinforcement learning with offline datasets.

*ArXiv*, abs/2006.09359, 2020.

-
Nikulin et al. [2023]

Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, and Sergey Kolesnikov.

Anti-exploration by random network distillation.

In *International Conference on Machine Learning (ICML)*, 2023.

-
Park et al. [2024]

Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar.

Is value learning really the main bottleneck in offline rl?

In *Neural Information Processing Systems (NeurIPS)*, 2024.

-
Park et al. [2025a]

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine.

Ogbench: Benchmarking offline goal-conditioned rl.

In *International Conference on Learning Representations (ICLR)*, 2025a.

-
Park et al. [2025b]

Seohong Park, Qiyang Li, and Sergey Levine.

Flow q-learning.

In *International Conference on Machine Learning (ICML)*, 2025b.

-
Peng et al. [2019]

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine.

Advantage-weighted regression: Simple and scalable off-policy reinforcement learning.

*ArXiv*, abs/1910.00177, 2019.

-
Peters and Schaal [2007]

Jan Peters and Stefan Schaal.

Reinforcement learning by reward-weighted regression for operational space control.

In *International Conference on Machine Learning (ICML)*, 2007.

-
Psenka et al. [2024]

Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma.

Learning a diffusion model policy from rewards via q-score matching.

In *International Conference on Machine Learning (ICML)*, 2024.

-
Ren et al. [2025]

Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz.

Diffusion policy policy optimization.

In *International Conference on Learning Representations (ICLR)*, 2025.

-
Schulman et al. [2015]

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz.

Trust region policy optimization.

In *International Conference on Machine Learning (ICML)*, 2015.

-
Schulman et al. [2017]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

*ArXiv*, abs/1707.06347, 2017.

-
Sikchi et al. [2024]

Harshit S. Sikchi, Qinqing Zheng, Amy Zhang, and Scott Niekum.

Dual rl: Unification and new methods for reinforcement and imitation learning.

In *International Conference on Learning Representations (ICLR)*, 2024.

-
Sohl-Dickstein et al. [2015]

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.

Deep unsupervised learning using nonequilibrium thermodynamics.

In *International Conference on Machine Learning (ICML)*, 2015.

-
Song and Ermon [2019]

Yang Song and Stefano Ermon.

Generative modeling by estimating gradients of the data distribution.

In *Neural Information Processing Systems (NeurIPS)*, 2019.

-
Sutton and Barto [2005]

Richard S. Sutton and Andrew G. Barto.

Reinforcement learning: An introduction.

*IEEE Transactions on Neural Networks*, 16:285–286, 2005.

-
Sutton et al. [1999]

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour.

Policy gradient methods for reinforcement learning with function approximation.

In *Neural Information Processing Systems (NeurIPS)*, 1999.

-
Tarasov et al. [2023]

Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov.

Revisiting the minimalist approach to offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2023.

-
Touati and Ollivier [2021]

Ahmed Touati and Yann Ollivier.

Learning one representation to optimize all rewards.

In *Neural Information Processing Systems (NeurIPS)*, 2021.

-
Wang et al. [2023]

Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou.

Diffusion policies as an expressive policy class for offline reinforcement learning.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Wang et al. [2020]

Ziyun Wang, Alexander Novikov, Konrad Zolna, Jost Tobias Springenberg, Scott E. Reed, Bobak Shahriari, Noah Siegel, Josh Merel, Caglar Gulcehre, Nicolas Manfred Otto Heess, and Nando de Freitas.

Critic regularized regression.

In *Neural Information Processing Systems (NeurIPS)*, 2020.

-
Wu et al. [2019]

Yifan Wu, G. Tucker, and Ofir Nachum.

Behavior regularized offline reinforcement learning.

*ArXiv*, abs/1911.11361, 2019.

-
Xu et al. [2023]

Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Chan, and Xianyuan Zhan.

Offline rl with no ood actions: In-sample learning via implicit value regularization.

In *International Conference on Learning Representations (ICLR)*, 2023.

-
Yamagata et al. [2023]

Taku Yamagata, Ahmed Khalil, and Raul Santos-Rodriguez.

Q-learning decision transformer: Leveraging dynamic programming for conditional sequence modelling in offline rl.

In *International Conference on Machine Learning (ICML)*, 2023.

-
Yang et al. [2023]

Long Yang, Zhixiong Huang, Fenghao Lei, Yucun Zhong, Yiming Yang, Cong Fang, Shiting Wen, Binbin Zhou, and Zhouchen Lin.

Policy representation via diffusion probability model for reinforcement learning.

*ArXiv*, abs/2305.13122, 2023.

-
Yarats et al. [2022]

Denis Yarats, David Brandfonbrener, Hao Liu, Michael Laskin, P. Abbeel, Alessandro Lazaric, and Lerrel Pinto.

Don’t change the algorithm, change the data: Exploratory data for offline reinforcement learning.

*ArXiv*, abs/2201.13425, 2022.

-
Zhang et al. [2024]

Ruoqi Zhang, Ziwei Luo, Jens Sjölund, Thomas B Schön, and Per Mattsson.

Entropy-regularized diffusion policy with q-ensembles for offline reinforcement learning.

In *Neural Information Processing Systems (NeurIPS)*, 2024.

-
Zhang et al. [2025]

Shiyuan Zhang, Weitong Zhang, and Quanquan Gu.

Energy-weighted flow matching for offline reinforcement learning.

In *International Conference on Learning Representations (ICLR)*, 2025.

## Appendix A Theoretical results

For any probability measure $\mu$ on ${\mathbb{R}}$ and any bounded, measurable, non-decreasing functions $g,h:{\mathbb{R}}\to{\mathbb{R}}$,

###### Proof.

Since $g$ and $h$ are non-decreasing, the signs of $g(y)-g(z)$ and $h(y)-h(z)$ are the same for any $y,z\in{\mathbb{R}}$.
Hence, we have

$$
\begin{aligned}
0 &\leq\int_{{\mathbb{R}}\times{\mathbb{R}}}(g(y)-g(z))(h(y)-h(z))(\mu\otimes\mu)({\mathrm{d}}y,{\mathrm{d}}z) \tag{21} \\
&=\int_{\mathbb{R}}\left(\int_{\mathbb{R}}(g(y)h(y)+g(z)h(z)-g(y)h(z)-g(z)h(y))\mu({\mathrm{d}}y)\right)\mu({\mathrm{d}}z) \tag{22} \\
&=2\int_{\mathbb{R}}g(x)h(x)\mu({\mathrm{d}}x)-2\int_{\mathbb{R}}g(x)\mu({\mathrm{d}}x)\int_{\mathbb{R}}h(x)\mu({\mathrm{d}}x), \tag{23}
\end{aligned}
$$

from which the conclusion follows,
where $\mu\otimes\mu$ denotes the product measure of $\mu$ and itself,
and we use Fubini’s theorem in the second line.
∎

Let $s\in{\mathcal{S}}$ be a state, $\pi,\hat{\pi}:{\mathcal{S}}\to\Delta({\mathcal{A}})$ be policies, and $f:{\mathbb{R}}\to{\mathbb{R}}$ be a bounded, measurable, non-negative, non-decreasing function.
Suppose that $\pi(a\mid s)=f(A_{\hat{\pi}}(s,a))\hat{\pi}(a\mid s)$ and $\mathbb{E}_{a\sim\hat{\pi}(\cdot\mid s)}[f(A_{\hat{\pi}}(s,a))]=1$.
Then,

###### Proof.

To apply [Lemma 1](#Thmlemma1), we first rewrite the left-hand side of [Equation 24](#A1.E24) using probability measures as follows:

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi(\cdot\mid s)}[Q_{\hat{\pi}}(s,a)] &=\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)\pi_{s}({\mathrm{d}}a) \tag{25} \\
&=\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)f(A_{\hat{\pi}}(s,a))\hat{\pi}_{s}({\mathrm{d}}a) \tag{26} \\
&=\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)f(Q_{\hat{\pi}}(s,a)-V_{\hat{\pi}}(s))\hat{\pi}_{s}({\mathrm{d}}a), \tag{27}
\end{aligned}
$$

where $\pi_{s}$ and $\hat{\pi}_{s}$ denote the probability measures corresponding to
the distributions $\pi(\cdot\mid s)$ and $\hat{\pi}(\cdot\mid s)$, respectively.
Then,

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi(\cdot\mid s)}[Q_{\hat{\pi}}(s,a)] &=\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)f(Q_{\hat{\pi}}(s,a)-V_{\hat{\pi}}(s))\hat{\pi}_{s}({\mathrm{d}}a) \tag{28} \\
&=\int_{\mathbb{R}}qf(q-V_{\hat{\pi}}(s))\lambda({\mathrm{d}}q) \tag{29} \\
&\geq\left(\int_{\mathbb{R}}q\lambda({\mathrm{d}}q)\right)\left(\int_{\mathbb{R}}f(q-V_{\hat{\pi}}(s))\lambda({\mathrm{d}}q)\right) \tag{30} \\
&=\left(\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)\hat{\pi}_{s}({\mathrm{d}}a)\right)\left(\int_{\mathcal{A}}f(Q_{\hat{\pi}}(s,a)-V_{\hat{\pi}}(s))\hat{\pi}_{s}({\mathrm{d}}a)\right) \tag{31} \\
&=\left(\int_{\mathcal{A}}Q_{\hat{\pi}}(s,a)\hat{\pi}_{s}({\mathrm{d}}a)\right)\left(\int_{\mathcal{A}}f(A_{\hat{\pi}}(s,a))\hat{\pi}_{s}({\mathrm{d}}a)\right) \tag{32} \\
&=V_{\hat{\pi}}(s)\mathbb{E}_{a\sim\hat{\pi}(\cdot\mid s)}[f(A_{\hat{\pi}}(s,a))] \tag{33} \\
&=V_{\hat{\pi}}(s), \tag{34}
\end{aligned}
$$

where $\lambda$ denotes the pushforward measure of $\hat{\pi}_{s}$ by $Q_{\hat{\pi}}(s,\cdot)$,
and we use [Lemma 1](#Thmlemma1) in the third line
with $g(x)=1$ and $h(x)=f(x-V_{\hat{\pi}}(s))$,
both of which are non-decreasing.
∎

For any policies $\pi$ and $\hat{\pi}$ satisfying $\mathbb{E}_{a\sim\pi(\cdot\mid s)}[Q_{\hat{\pi}}(s,a)]\geq V_{\hat{\pi}}(s)$ for all $s\in{\mathcal{S}}$,

###### Proof.

This is a straightforward generalization of the policy improvement theorem to stochastic policies.
See Section 4.2 of [Sutton and Barto [[68](#bib.bib68)]] and Theorem 3 of [da Silva [[14](#bib.bib14)]].
∎

Let $\pi,\hat{\pi}:{\mathcal{S}}\to\Delta({\mathcal{A}})$ be policies and $f:{\mathbb{R}}\to{\mathbb{R}}$ be a bounded, measurable, non-negative, non-decreasing function.
Suppose that $\pi$ satisfies $\pi(a\mid s)\propto f(A_{\hat{\pi}}(s,a))\hat{\pi}(a\mid s)$.
Then,

###### Proof.

Fix $s\in{\mathcal{S}}$.
Let $\pi(a\mid s)=f(A_{\hat{\pi}}(s,a))\hat{\pi}(a\mid s)/Z(s)$,
where the normalization function $Z:{\mathcal{S}}\to{\mathbb{R}}$ is defined as

$$Z(s)=\int_{\mathcal{A}}f(A_{\hat{\pi}}(s,a))\hat{\pi}_{s}({\mathrm{d}}a).$$ \tag{37}

Then, we have

$$
\begin{aligned}
1 &=\int_{\mathcal{A}}f(A_{\hat{\pi}}(s,a))/Z(s)\hat{\pi}_{s}({\mathrm{d}}a) \tag{38} \\
&=\mathbb{E}_{a\sim\hat{\pi}(\cdot\mid s)}[f(A_{\hat{\pi}}(s,a))/Z(s)]. \tag{39}
\end{aligned}
$$

Defining $g=f/Z(s)$, we get $\mathbb{E}_{a\sim\hat{\pi}(\cdot\mid s)}[g(A_{\hat{\pi}}(s,a))]=1$.
Since $f$ is non-negative and non-decreasing, so is $g$,
and the conclusion directly follows from [Lemma 2](#Thmlemma2)
(with $\pi(a\mid s)=g(A_{\hat{\pi}}(s,a))\hat{\pi}(a\mid s)$)
and [Lemma 3](#Thmlemma3).
∎

Let $0\leq w_{1}\leq w_{2}$ be real numbers,
$\pi_{1},\pi_{2},\hat{\pi}:{\mathcal{S}}\to\Delta({\mathcal{A}})$ be policies,
and $f:{\mathbb{R}}\to{\mathbb{R}}$ be a bounded, measurable, non-negative, non-decreasing function.
Suppose that $\pi_{i}$ satisfies
$\pi_{i}(a\mid s)\propto f(A_{\hat{\pi}}(s,a))^{w_{i}}\hat{\pi}(a\mid s)$ for $i=1,2$.
Then,

###### Proof.

Fix $s\in{\mathcal{S}}$. As in the proof of [Theorem 1](#Thmtheorem1), write

$$
\begin{aligned}
\pi_{1}(a\mid s) &=\frac{f(A_{\hat{\pi}}(s,a))^{w_{1}}\hat{\pi}(a\mid s)}{Z_{1}(s)}, \tag{41}
\end{aligned}
$$
$$
\begin{aligned}
\pi_{2}(a\mid s) &=\frac{f(A_{\hat{\pi}}(s,a))^{w_{2}}\hat{\pi}(a\mid s)}{Z_{2}(s)}, \tag{42}
\end{aligned}
$$

where $Z_{1},Z_{2}:{\mathcal{S}}\to{\mathbb{R}}$ are the normalization functions.
Then, we have

$$\pi_{2}(a\mid s)=f(A_{\hat{\pi}}(s,a))^{w_{2}-w_{1}}\frac{Z_{1}(s)}{Z_{2}(s)}\pi_{1}(a\mid s).$$ \tag{43}

Since $Z_{1}$ and $Z_{2}$ are both bounded (which follows from the boundedness of $f$),
measurable, and non-negative,
we can apply [Lemma 2](#Thmlemma2) to the bounded, measurable, non-negative, non-decreasing function
$x\mapsto f(x)^{w_{2}-w_{1}}Z_{1}(s)/Z_{2}(s)$ with $(\pi,\hat{\pi})=(\pi_{2},\pi_{1})$ (in the notation of [Lemma 2](#Thmlemma2)).
The result then directly follows from [Lemma 3](#Thmlemma3) as before.
∎

## Appendix B Additional results

![Figure](x5.png)

*Figure 5: OGBench environments.*

![Figure](x6.png)

*Figure 6: Full ablation results on CFG weight $\bm{w}$. The performance of CFGRL generally improves as the CFG weight increases.*

![Figure](x7.png)

*Figure 7: Ablation study on optimality conditioning. Shared policies lead to better performance and extrapolation than separate policies, likely because the former shares representations.*

Enviromments. [Figure 5](#A2.F5) illustrates OGBench tasks.

Ablation study on the CFG weight $\bm{w}$.
We present the full ablation study on the CFG weight $w$ across all $17$ state-based OGBench tasks in [Figure 6](#A2.F6).
The results show that the performance improves as the CFG weight increases,
although it sometimes declines beyond a certain point,
likely because the policy deviates too far from the data distribution.

Ablation study on optimality conditioning.
When modeling an optimality-conditioned policy $\pi({\color[rgb]{0.6015625,0.6015625,0.6015625}a}\mid{\color[rgb]{%
0.6015625,0.6015625,0.6015625}s},{\color[rgb]{0.6015625,0.6015625,0.6015625}o})$ with $o\in\{\varnothing,0,1\}$,
we can either have separate networks for each $o$ value, or share the same network with a learnable optimality embedding.
We choose the latter in our experiments, and present an ablation study in [Figure 7](#A2.F7).
The results suggest that the shared architecture generally works and extrapolates better than the separate one.
We believe this is likely because extrapolation benefits from shared representations.

## Appendix C Implementation details

We implement CFGRL on top of the reference implementations provided by OGBench [[[57](#bib.bib57)]].
Each experiment in this work takes no more than $4$ hours on a single A5000 GPU.
We provide our implementations in the supplemental material.

Tasks.
In [Section 5](#S5), we employ $9$ tasks from the ExORL benchmark [[[78](#bib.bib78)]]
and $9$ single-task (singletask) variants from the OGBench suite [[[57](#bib.bib57)]].
We use the RND datasets for our ExORL experiments.
In [Section 6](#S6), we employ the oraclerep variant of OGBench tasks
to remove confounding factors related to goal representation learning,
where this variant provides ground-truth goal representations
(e.g., in antmaze, a goal is specified by only the $x$-$y$ position, as opposed to the full $29$-dimensional state including proprioceptive information).

Methods and hyperparameters.
For baselines, we follow the original implementations and hyperparameters whenever possible [[[39](#bib.bib39), [57](#bib.bib57), [58](#bib.bib58)]].
For GCBC methods in [Section 6](#S6), we sample goals uniformly from future states, as in the original implementation in OGBench [[[57](#bib.bib57)]].
This can be viewed as an approximation of geometric sampling with a high $\gamma$.
We present the full list of the hyperparameters in [Tables 4](#A3.T4), [5](#A3.T5), [6](#A3.T6), [7](#A3.T7) and [8](#A3.T8).

*Table 4: Hyperparameters for ExORL offline RL experiments ([Table 1](#S5.T1)).*

*Table 5: Per-task hyperparameters for ExoRL offline RL experiments ([Table 4](#A3.T4)).*

*Table 6: Hyperparameters for OGBench offline RL experiments ([Table 2](#S5.T2)).*

*Table 7: Per-task hyperparameters for OGBench offline RL experiments ([Table 2](#S5.T2)).*

*Table 8: Hyperparameters for GCBC experiments ([Table 3](#S6.T3)).*

Generated on Thu May 29 14:04:10 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)