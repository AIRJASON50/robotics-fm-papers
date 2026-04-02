[# Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics Chenhao Li ETH Zurich, Switzerland chenhli@ethz.ch &Andreas Krause ETH Zurich, Switzerland krausea@ethz.ch &Marco Hutter ETH Zurich, Switzerland mahutter@ethz.ch ###### Abstract Learning robust and generalizable world models is crucial for enabling efficient and scalable robotic control in real-world environments. In this work, we introduce a novel framework for learning world models that accurately capture complex, partially observable, and stochastic dynamics. The proposed method employs a dual-autoregressive mechanism and self-supervised training to achieve reliable long-horizon predictions without relying on domain-specific inductive biases, ensuring adaptability across diverse robotic tasks. We further propose a policy optimization framework that leverages world models for efficient training in imagined environments and seamless deployment in real-world systems. This work advances model-based reinforcement learning by addressing the challenges of long-horizon prediction, error accumulation, and sim-to-real transfer. By providing a scalable and robust framework, the introduced methods pave the way for adaptive and efficient robotic systems in real-world applications. https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel)

![Figure](x1.png)

*Figure 1: Autoregressive imagination, ground-truth simulation, and real-world deployment of RWM. For each environment, the top row showcases the RWM autoregressively predicting future trajectories in imagination. The second row visualizes the ground truth evolution in simulation. Specifically for the ANYmal D quadruped and Unitree G1 humanoid, the framework achieves robust policy optimization through MBPO-PPO, enabling zero-shot deployment on hardware.*

### 1 Introduction

Robotic systems have achieved remarkable advancements in recent years, driven by progress in reinforcement learning (RL) [haarnoja2018soft, schulman2017proximal] and control theory [nguyen2019review, todorov2012mujoco].
A prevalent limitation in many approaches is the lack of adaptation and learning once the policy is deployed on the real system [tan2018sim, peng2020learning, li2023learning, li2023versatile].
This results in underutilization of the valuable data generated during real-world interactions.
Robotic systems operating in dynamic and uncertain environments require the ability to continually adapt their behavior to new conditions [lee2020learning].
The inability to exploit real-world experience for further learning restricts the system’s robustness and limits its ability to handle evolving scenarios effectively.
Truly intelligent robotic systems should operate efficiently and reliably using limited data, adapting to real-world conditions in a scalable manner [nagabandi2018neural, hafner2020mastering].
While model-free RL algorithms such as Proximal Policy Optimization (PPO) [schulman2017proximal] and Soft Actor-Critic (SAC) [haarnoja2018soft] have demonstrated impressive results in simulation, their high interaction requirements make them impractical for real-world robotics.
Sample-efficient methods are therefore essential for leveraging the information in real-world data without extensive environment interactions [chua2018deep, janner2019trust].

A promising solution is the use of predictive models of the environment, commonly referred to as world models [ha2018recurrent, hafner2019learning].
World models simulate environment dynamics to enable planning and policy optimization, often referred to as learning in imagination [sutton1991dyna].
These models have shown success across diverse robotic domains, including manipulation [ebert2018visual, finn2016unsupervised], navigation [hafner2020mastering], and locomotion [nagabandi2018neural].
However, developing reliable and generalizable world models poses unique challenges due to the complexity of real-world dynamics, including nonlinearities, stochasticity, and partial observability [wu2023daydreamer, song2024learning].
Existing approaches often incorporate domain-specific inductive biases, such as structured state representations or hand-designed network architectures [yang2020data, sancaktar2022curious, li2024fld], to improve model fidelity.
While effective, these methods are limited in their scalability and adaptability to novel environments or tasks.
In contrast, a general framework for learning world models without domain-specific assumptions has the potential to enhance generalization and applicability across a wide range of robotic systems and scenarios.

In this work, we present a novel approach for learning world models that emphasizes robustness and accuracy over long-horizon predictions.
Our method is designed to operate without handcrafted representations or specialized architectural biases, enabling broad applicability to diverse robotic tasks.
To evaluate the utility of these learned models, we further propose a policy optimization method using PPO and demonstrate successful deployment in both simulated and real-world environments.
To the best of our knowledge, this is the first framework to reliably train policies on a learned neural network simulator without any domain-specific knowledge and deploy them on physical hardware with minimal performance loss.

Our contributions are summarized as follows:
(i) We introduce a novel network architecture and training framework that enables the learning of reliable world models capable of long autoregressive rollouts, a critical property for downstream planning and control.
(ii) We provide a comprehensive evaluation suite spanning diverse robotic tasks to benchmark our method. Comparative experiments with existing world model frameworks demonstrate the effectiveness of our approach.
(iii) We propose an efficient policy optimization framework that leverages the learned world models for continuous control and generalizes effectively to real-world scenarios with hardware experiments, including both quadruped and humanoid systems.

By addressing the challenges associated with learning world models, this work contributes toward bridging the gap between data-driven modeling and real-world deployment.
The proposed framework enhances the scalability, adaptability, and robustness of robotic systems, paving the way for broader adoption of model-based reinforcement learning in real-world applications.
Supplementary videos for this work are available on [https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel).

### 2 Related work

#### 2.1 World Models for Robotics

World models have emerged as a cornerstone in robotics for capturing system dynamics and enabling efficient planning and control through simulated trajectories.
A prominent application of world models is in robotic control, where dynamics models are used to describe real-world dynamics for policy optimization [levine2016end].
Extensions to vision-based tasks have been realized through visual foresight techniques [finn2016unsupervised, finn2017deep, ebert2018visual], which learn visual dynamics for planning in high-dimensional sensory spaces.
Similar ideas are applied to train RL agents in such world models aiming to fully replicate real environment interactions [ha2018recurrent, alonso2024diffusion].
These approaches underline the versatility of world models in tasks requiring rich perceptual inputs.

To improve the generalization of black-box neural network-based world models beyond the training distribution, many works incorporate known physics principles or state structures into model design, addressing potential limitations in control performance.
Examples include foot-placement dynamics [yang2020data], object invariance [sancaktar2022curious], granular media interactions [choi2023learning], frequency domain parameterization [li2024fld], rigid body dynamics [song2024learning], and semi-structured Lagrangian dynamics models [levy2024learning].
While these methods demonstrate impressive results, they often require strong domain knowledge and carefully crafted inductive biases, which can restrict their scalability and adaptability to diverse robotic applications.
Latent-space dynamics models offer an alternative by abstracting the state space into compact representations, enabling efficient long-horizon planning.
Deep Planning Network (PlaNet) [hafner2019learning] and its successor Dreamer [hafner2019dream, hafner2020mastering, hafner2023mastering] exemplify this trend, achieving state-of-the-art performance in continuous control and visual navigation tasks.
These frameworks have been extended to real-world robotics [wu2023daydreamer, bi2024sample], demonstrating their potential in both simulation and hardware deployment.

#### 2.2 Model-Based Reinforcement Learning

Model-Based Reinforcement Learning (MBRL) has emerged as a powerful approach to address the limitations of model-free reinforcement learning, particularly in scenarios where sample efficiency and safety are critical.
Unlike model-free methods, which learn policies directly from interactions with the environment, MBRL leverages a learned model of the environment to simulate interactions, enabling more efficient and safer policy learning.
One of the pioneering methods in MBRL is Probabilistic Ensembles with Trajectory Sampling (PETS), which uses an ensemble of probabilistic neural networks to model the environment dynamics [chua2018deep].
Building on the idea of latent-space modeling, PlaNet leverages a latent dynamics model to plan directly in a learned latent space [hafner2019learning].
Dreamer extends the concept by incorporating an actor-critic framework into the latent dynamics model, enabling the simultaneous learning of both the dynamics model and the policy [hafner2019dream, hafner2020mastering, hafner2023mastering].
Variations on the architectural design also see success in improving generation capabilities of such latent dynamics models with autoregressive transformer [micheli2022transformers] and the stochastic nature of variational autoencoders [zhang2024storm].
Recent advancements in this area include TD-MPC and TD-MPC2, which integrate model-based learning with MPC to achieve high-performance control in dynamic environments [hansen2022temporal, feng2023finetuning, hansen2023td].

Recognizing the strengths of both model-based and model-free methods, several hybrid approaches have been developed to combine the sample efficiency of MBRL with the robustness of model-free reinforcement learning.
One notable example is Model-Based Policy Optimization (MBPO), which uses a model-based approach for planning and policy optimization but refines the policy using model-free updates [janner2019trust].
It emphasizes selectively relying on the learned model when its predictions are accurate, thus mitigating the negative effects of model inaccuracies.
Building on similar principles, Model-based Offline Policy Optimization (MOPO) extends the framework to the offline setting, where learning is conducted entirely from previously collected data without further environment interaction [yu2020mopo].
In contrast to using zeroth-order model-free reinforcement learning for policy optimization, first-order gradient-based optimization is used to improve policy learning [xu2022accelerated, georgiev2024adaptive].
This allows for more efficient and precise policy updates, particularly in complex, high-dimensional environments, where accurate gradient information is crucial for performance.
Our framework extends MBPO by integrating it with PPO over extensive autoregressive rollouts, making it particularly effective for complex robotic control tasks.

### 3 Approach

#### 3.1 Reinforcement Learning and World Models

We formulate the problem by modeling the environment as a Partially Observable Markov Decision Process (POMDP) [sutton2018reinforcement], defined by the tuple $\left({\mathcal{S}},{\mathcal{A}},{\mathcal{O}},T,R,O,\gamma\right)$, where ${\mathcal{S}}$, ${\mathcal{A}}$, and ${\mathcal{O}}$ denote the state, action, and observation spaces, respectively.
The transition kernel $T:{\mathcal{S}}\times{\mathcal{A}}\to{\mathcal{S}}$ captures the environment dynamics $p\left(s_{t+1}\mid s_{t},a_{t}\right)$, while the reward function $R:{\mathcal{S}}\times{\mathcal{A}}\times{\mathcal{S}}\to{\mathbb{R}}$ maps transitions to scalar rewards.
Observations $o_{t}\in{\mathcal{O}}$ are emitted according to probabilities $p\left(o_{t}\mid s_{t}\right)$, governed by the observation kernel $O:\mathcal{S}\to\mathcal{O}$.
The agent seeks to learn a policy $\pi_{\theta}:{\mathcal{O}}\to{\mathcal{A}}$ that maximizes the expected discounted return $\mathbb{E}_{\pi_{\theta}}\left[\sum_{t\geq 0}\gamma^{t}r_{t}\right]$, where $r_{t}$ is the reward at time $t$ and $\gamma\in\left[0,1\right]$ is the discount factor.

World models [ha2018recurrent] approximate the environment dynamics and facilitate policy optimization by enabling simulated environment interactions in imagination [sutton1991dyna].
Training typically involves three iterative steps: (1) collect data from real environment interactions; (2) train the world model using the collected data; and (3) optimize the policy within the simulated environment produced by the world model.

Despite the success of existing frameworks in achieving tasks in simplified settings, their application to complex low-level robotic control remains a significant challenge.
To address this gap, we propose Robotic World Model (RWM), a novel framework for learning robust world models in partially observable and dynamically complex environments.
RWM builds on the core concept of world models but introduces architectural and training innovations that enable reliable long-horizon predictions, even in stochastic and partially observable settings.
By incorporating historical context and autoregressive training, RWM addresses challenges such as error accumulation and partially observable and discontinuous dynamics, which are critical in real-world robotics applications.

#### 3.2 Self-supervised Autoregressive Training

To address the inherent complexity of partially observable environments, we propose a self-supervised autoregressive training framework as the backbone of RWM.
This framework trains the world model $p_{\phi}$ to predict future observations by leveraging both historical observation-action sequences and its own predictions, ensuring robustness over extended rollouts.

The input to the world model consists of a sequence of observation-action pairs spanning $M$ historical steps.
At each time step $t$, the model predicts the distribution of the next observation $p\left(o_{t+1}\mid o_{t-M+1:t},a_{t-M+1:t}\right)$.
Predictions are generated autoregressively: at each step, the predicted observation $o^{\prime}_{t+1}$ is appended to the history and combined with the next action $a_{t+1}$ to serve as input for subsequent predictions.
This process is repeated over a prediction horizon of $N$ steps, producing a sequence of future predictions.
The predicted observation $k$ steps ahead can thus be written as

$$o^{\prime}_{t+k}\sim p_{\phi}\left(\cdot\mid o_{t-M+k:t},o^{\prime}_{t+1:t+k-1},a_{t-M+k:t+k-1}\right).$$ \tag{1}

A similar process is also applied to predict privileged information $c$, such as contacts, providing an additional learning objective that implicitly embeds critical information for accurate long-term predictions.
Such a training scheme introduces the model to the distribution it will encounter at test time, reducing the mismatch between training and inference distributions.
Overall, the model is optimized by minimizing the multi-step prediction error:

$${\mathcal{L}}=\frac{1}{N}\sum_{k=1}^{N}\alpha^{k}\left[L_{o}\left(o^{\prime}_{t+k},o_{t+k}\right)+L_{c}\left(c^{\prime}_{t+k},c_{t+k}\right)\right],$$ \tag{2}

where $L_{o}$ and $L_{c}$ quantify the discrepancy between predicted and true observations and privileged information, and $\alpha$ denotes a decay factor.
This autoregressive training objective encourages the hidden states to encode representations that support accurate and reliable long-horizon predictions.

Training data is constructed by sliding a window of size $M+N$ over collected trajectories, providing sufficient historical context for prediction targets.
To improve gradient propagation through autoregressive predictions, we apply reparameterization tricks to enable effective end-to-end optimization.
By incorporating historical observations, RWM captures unobservable dynamics, addressing the challenges of partially observable and potentially discontinuous environments.
The autoregressive training mitigates error accumulation, a common issue in long-horizon predictions, and eliminates the need for handcrafted representations or domain-specific inductive biases, enhancing generalization across diverse tasks.
This process is illustrated in Fig. [2(a)](#S3.F2.sf1), in contrast to the teacher-forcing pipeline in Fig. [2(b)](#S3.F2.sf2), which is commonly adopted to train many popular architectures [hafner2019dream, chen2021decision].
Specifically, teacher-forcing can be viewed as a special case of autoregressive training with forecast horizon $N=1$, which boosts training with higher parallelization.

![Figure](x2.png)

*(a) Autoregressive training.*

![Figure](x3.png)

*(b) Teacher-forcing training.*

Figure 2: Comparison of training paradigms for world models with an example of a history horizon $H=3$. (a) Autoregressive training operates with an example of a forecast horizon $N=2$, leveraging historical data and its own predictions for long-horizon robustness. The dashed arrows denote the sequential autoregressive prediction steps. (b) Teacher-forcing training can be viewed as a special case of autoregressive training with a forecast horizon $N=1$, using ground truth observations for next-step predictions to optimize parallelization but limiting robustness to error accumulation.

While the proposed autoregressive training framework can be applied to any network architecture, RWM utilizes a GRU-based architecture for its ability to maintain long-term historical context while operating on low-dimensional inputs.
The network predicts the mean and standard deviation of a Gaussian distribution describing the next observation.
Our framework introduces a dual-autoregressive mechanism: (i) Inner autoregression updates GRU hidden states autoregressively after each historical step within the context horizon $M$.
(ii) Outer autoregression feeds predicted observations from the forecast horizon $N$ back into the network.
This architecture, visualized in Fig. [S6](#A1.F6), ensures robustness to long-term dependencies and transitions, making RWM suitable for complex robotics applications.

#### 3.3 Policy Optimization on Learned World Models

Policy optimization in RWM is conducted using the learned world model, following a framework inspired by Model-Based Policy Optimization (MBPO) [janner2019trust] and the Dyna algorithm [sutton1990integrated].
During imagination, the actions are generated recursively by the policy $\pi_{\theta}$ conditioned on the observations predicted by the world model $p_{\phi}$, which is further conditioned on the previous predictions.
The actions at time $t+k$ can thus be written as

$$a^{\prime}_{t+k}\sim\pi_{\theta}\left(\cdot\mid o^{\prime}_{t+k}\right),$$ \tag{3}

where $o^{\prime}_{t+k}$ is drawn autoregressively according to Eq. [1](#S3.E1).
Rewards are computed from imagined observations and privileged information.
The approach combines model-based imagination with model-free RL to achieve efficient and robust policy optimization, as outlined in Algorithm [1](#alg1).

*Algorithm 1 Policy optimization with RWM*

The replay buffer ${\mathcal{D}}$ aggregates real environment interactions collected by a single agent.
The world model $p_{\phi}$ is trained on this data following the autoregressive scheme described in Sec. [3.2](#S3.SS2).
Imagination agents are initialized from samples in ${\mathcal{D}}$ and simulate trajectories using the world model for $T$ steps, enabling policy updates through a reinforcement learning algorithm.
The training diagram is visualized in Fig. [S7](#A1.F7).

While PPO is known for its strong performance in robotic tasks, training it on learned world models poses unique challenges.
Model inaccuracies can be exploited during policy learning, leading to discrepancies between the imagined and true dynamics.
This issue is exacerbated by the extended autoregressive rollouts required for PPO, which compound prediction errors.
We denote this policy optimization method by MBPO-PPO.
Despite these challenges, RWM demonstrates its robustness by successfully optimizing policies over a hundred autoregressive steps with MBPO-PPO, far exceeding the capabilities of existing frameworks such as MBPO [janner2019trust], Dreamer [hafner2019dream, hafner2020mastering, hafner2023mastering], or TD-MPC [hansen2022temporal, hansen2023td].
This result underscores the accuracy and stability of the proposed training method and its ability to synthesize policies deployable on hardware.

### 4 Experiments

We validate RWM through a comprehensive set of experiments across diverse robotic systems, environments, and network architectures.
The experiments are designed to assess the accuracy and robustness of RWM, evaluate its architectural and training design choices, and demonstrate its effectiveness across diverse robotic tasks in Isaac Lab [mittal2023orbit] and in real-world deployment combined with MBPO-PPO.
We start the analysis by looking into the autoregressive prediction accuracy and robustness of the world model learned with simulation data induced by a velocity tracking policy.
The observation and action spaces of the world model are detailed in Table [S2](#A1.T2) and Table [S4](#A1.T4).
We then compare various network architectures and the error induced across diverse robotic environments and tasks to demonstrate the generality of RWM.
And finally, we learn a policy in RWM with the proposed MBPO-PPO and demonstrate the applicability and robustness of the method on ANYmal D [hutter2016anymal] and Unitree G1 hardware.

#### 4.1 Autoregressive Trajectory Prediction

The capability of a world model to maintain high fidelity during autoregressive rollouts is critical for effective planning and policy optimization.
To evaluate this aspect, we analyze the autoregressive prediction performance of RWM using trajectories collected from ANYmal D hardware.
The control frequency of the robot is at $50\,Hz$.
The model is trained with history horizon $M=32$ and forecast horizon $N=8$.
Further details on the network architecture and training parameters are summarized in Sec. [A.2.1](#A1.SS2.SSS1) and Sec. [A.3.1](#A1.SS3.SSS1), respectively.
The autoregressive trajectory predictions by RWM are visualized in Fig. [3(a)](#S4.F3.sf1).

![Figure](x4.png)

*(a) Autoregressive trajectory prediction by RWM.*

![Figure](x5.png)

*(b) Prediction error under Gaussian noise.*

Figure 3: (Left) Solid lines represent ground truth trajectories, while dashed lines denote predicted state evolution. Predictions commence at $t=32$ using historical observations, with future observations predicted autoregressively by feeding prior predictions back into the model. (Right) Yellow curves denote RWM at varying noise levels, demonstrating consistent robustness and lower error accumulation across forecast steps. Grey curves represent the MLP baseline, which exhibits significantly higher error accumulation and reduced robustness to noise.

The results demonstrate that RWM exhibits a remarkable alignment between predicted and ground truth trajectories across all observed variables.
This consistency persists over extended rollouts, showcasing the model’s ability to mitigate compounding errors—a critical challenge in long-horizon predictions.
This performance is attributed to the dual-autoregressive mechanism introduced in Sec. [3.2](#S3.SS2), which stabilizes predictions despite the short forecast horizon employed during training.
A comparison of state evolution between the RWM prediction and the ground truth simulation is illustrated in Fig. [1](#S0.F1) (bottom).
The visualization highlights the ability of RWM to maintain consistency in trajectory predictions over long horizons, even beyond the training forecast horizon.
This robustness is pivotal for stable policy learning and deployment, as discussed further in Sec. [4.4](#S4.SS4).

It is notable that the choice of history horizon $M$ and forecast horizon $N$ plays a critical role in the training and performance of RWM.
Our ablation study in Sec. [A.4.1](#A1.SS4.SSS1) reveals that, while extending both $M$ and $N$ improves accuracy, practical considerations of computational cost necessitate careful tuning of these hyperparameters to achieve optimal performance.

#### 4.2 Robustness under Noise

A critical challenge in training world models is their ability to generalize under noisy conditions, particularly when predictions rely on autoregressive rollouts.
Even small deviations from the training distribution can cascade into untrained regions, causing the model to hallucinate future trajectories.
To assess the robustness of RWM, we analyze its performance under Gaussian noise perturbations applied to both observations and actions.
We compare the results with an MLP-based baseline also trained autoregressively with the same history and forecast horizon, as shown in Fig. [3(b)](#S4.F3.sf2), where yellow curves denote the relative prediction error $e$ for RWM, and grey curves represent the MLP baseline.

The results indicate a clear advantage of RWM over the MLP baseline across all noise levels.
As forecast steps increase, the relative prediction error of the MLP model grows significantly, diverging more rapidly than RWM.
In contrast, RWM demonstrates superior stability, maintaining lower prediction errors even under high noise levels.
This robustness can be attributed to the dual-autoregressive mechanism introduced in Sec. [3.2](#S3.SS2), which ensures stability in long-horizon predictions.
This design minimizes the accumulation of errors by continually refining the state representation toward long-term predictions, even in the presence of noisy inputs.

#### 4.3 Generality across Robotic Environments

To assess the generality and robustness of RWM across a diverse range of robotic environments, we compare its performance with several baseline methods, including MLP, recurrent state-space model (RSSM) [hafner2019learning, hafner2019dream, hafner2020mastering, hafner2023mastering], and transformer-based architectures [chen2021decision, reed2022generalist].
These baselines represent widely adopted approaches in dynamics modeling and policy optimization.
All models are given the same context during training and evaluation.
Their training parameters are detailed in Sec. [A.2.2](#A1.SS2.SSS2).
The relative autoregressive prediction errors $e$ for these models are shown in Fig. [4](#S4.F4).
The tasks span manipulation scenarios as well as quadruped and humanoid locomotion tasks, allowing for a comprehensive evaluation of the models.
In addition, we highlight the importance of the autoregressive training introduced in Sec. [3.2](#S3.SS2) by including both RWM trained with teacher-forcing (RWM-TF) and autoregressive training (RWM-AR), demonstrating the significant performance gains achieved by the latter.

![Figure](x6.png)

*Figure 4: Autoregressive trajectory prediction errors across diverse robotic environments and network architectures. RWM trained with autoregressive training (RWM-AR) consistently outperforms baseline methods, including MLP, recurrent state-space model (RSSM), and transformer-based architectures. RWM-AR demonstrates superior generalization and robustness across tasks, from manipulation to locomotion. Autoregressive training (RWM-AR) reduces compounding errors over long rollouts, significantly improving performance compared to teacher-forcing training (RWM-TF).*

The results highlight the superiority of RWM trained with autoregressive training (RWM-AR), which consistently achieves the lowest prediction errors across all environments.
The performance gap between RWM-AR and the baselines is especially pronounced in complex and dynamic tasks, such as velocity tracking for legged robots, where accurate long-horizon predictions are critical for effective control.
The comparison also reveals that RWM-AR significantly outperforms its teacher-forcing counterpart (RWM-TF), underscoring the importance of autoregressive training in mitigating compounding prediction errors over long rollouts.
We additionally visualize the imagination rolled out by RWM-AR compared with the ground truth simulation in Fig. [1](#S0.F1) and Fig. [S9](#A1.F9).

Note that the baselines are trained using teacher forcing as they are traditionally implemented.
However, the proposed autoregressive training framework is architecture-agnostic and can also be applied to baseline models.
When trained with autoregressive training, RSSM achieves a performance comparable to the proposed GRU-based architecture.
Nevertheless, we opt for the GRU-based model due to its simplicity and computational efficiency.
On the other hand, training transformer architectures with autoregressive training does not scale effectively, as the multi-step gradient propagation in autoregressive forecasting leads to GPU memory constraints, limiting their practicality for this approach.
These results demonstrate that RWM, when combined with autoregressive training, achieves robust and generalizable performance across diverse robotic tasks.

#### 4.4 Policy Learning and Hardware Transfer

Using MBPO-PPO, we train a goal-conditioned velocity tracking policy for ANYmal D and Unitree G1 leveraging RWM.
The policy’s observation and action spaces are detailed in Sec. [A.1.1](#A1.SS1.SSS1), and its architecture is described in Sec. [A.2.3](#A1.SS2.SSS3).
Reward formulations are provided in Sec. [A.1.2](#A1.SS1.SSS2), while training parameters are summarized in Sec. [A.3.2](#A1.SS3.SSS2).
We compare MBPO-PPO with two baselines: Short-Horizon Actor-Critic (SHAC) [xu2022accelerated] and DreamerV3 [hafner2023mastering].
SHAC employs a first-order gradient-based method that propagates gradients through the world model to optimize the policy.
Dreamer integrates a latent-space dynamics model with an actor-critic framework, emphasizing sample efficiency and robustness in continuous control tasks.

![Figure](x7.png)

*Figure 5: Model error and policy mean reward for the ANYmal D (left) and Unitree G1 (right) velocity tracking task with MBPO-PPO. The policy is trained using estimated rewards computed from predicted observations by RWM. Ground truth rewards, visualized with solid lines, are reported by the simulator for evaluation purposes only.*

Figure [5](#S4.F5) illustrates the model error $e$ during policy optimization.
While MBPO-PPO demonstrates a significant reduction in model error over training, SHAC struggles with high and fluctuating model error throughout the process.
Its reliance on first-order gradients for optimization is not well-suited for discontinuous dynamics, such as those encountered in legged locomotion, where system behavior changes drastically due to varying contact patterns.
The resulting inaccurate gradients lead to suboptimal policy updates, producing chaotic robot behaviors during training.
These chaotic behaviors, in turn, generate low-quality training data for updating RWM, exacerbating model inaccuracies.
Although Dreamer effectively leverages its latent-space dynamics model for policy optimization, its reliance on shorter planning horizons during training limits its ability to handle long-horizon dependencies, particularly in stochastic environments.
As a result, Dreamer encounters moderate compounding errors during policy learning, which hinder its convergence to optimal behaviors.

On the right plot of rewards $r$, predicted rewards (dashed) from MBPO-PPO initially overshoot the ground truth (solid) due to the policy exploiting small inaccuracies in the model’s optimistic estimates.
As training progresses, predictions align more closely with ground truth, remaining accurate enough to guide effective learning.
In contrast, SHAC fails to converge, producing unstable behaviors that degrade both policy and model quality.
Dreamer demonstrates partial convergence, achieving higher rewards compared to SHAC but significantly lagging behind MBPO-PPO.

To evaluate the robustness of the learned policies, we deploy them on ANYmal D and Unitree G1 hardware in a zero-shot transfer setup.
SHAC and Dreamer fail to produce a deployable policy due to its collapse during training.
However, as shown in Fig. [1](#S0.F1), the policy learned using MBPO-PPO demonstrates reliable and robust performance in tracking goal-conditioned velocity commands and maintaining stability under external disturbances, such as unexpected impacts and terrain conditions.
The success of MBPO-PPO in hardware deployment is a direct result of the high-quality trajectory predictions generated by RWM, which enable accurate and effective policy optimization.
Videos showcasing the robustness of the policies on hardware, including their responses to external disturbances, are available on our webpage.
These results underline the effectiveness of RWM and MBPO-PPO in enabling robust and scalable policy deployment for real-world robotic systems.

### 5 Limitations

The policy learned with RWM and MBPO-PPO surpasses existing MBRL methods in both robustness and generalization.
However, it still falls short of the performance achieved by well-tuned model-free RL methods trained on high-fidelity simulators.
Model-free RL, being a more mature and extensively optimized paradigm, excels in settings where unlimited interaction with near-perfect simulators is possible.
In contrast, the strengths of MBRL are more pronounced in scenarios where accurate or efficient simulation is infeasible, making it an indispensable tool for enabling intelligent agents to eventually learn and adapt in complex, real-world environments.
To clarify the computational and performance aspects, we provide a comparison against a PPO-based method with a high-fidelity simulator in Table [1](#S5.T1).

*Table 1: Comparison with model-free method*

In this work, the world model is pre-trained using simulation data prior to policy optimization, reducing instability during training (see Sec. [A.4.3](#A1.SS4.SSS3)).
However, training from scratch remains challenging as policies can exploit model inaccuracies during exploration, leading to inefficiency and instability.
In addition, the need for additional interaction with the environment to fine-tune the world model highlights areas for further refinement.
Nevertheless, enabling safe and effective online learning directly on hardware remains challenging (see Sec. [A.4.4](#A1.SS4.SSS4)).
Current training in simulation avoids potential hardware damage, but incorporating safety constraints and robust uncertainty estimates will be critical for deploying RWM and MBPO-PPO in real-world, lifelong learning scenarios.
These limitations underscore the trade-offs inherent in MBRL frameworks, balancing data efficiency, safety, and performance while addressing the complexities of real-world robotic systems.

### 6 Conclusion

In this work, we present RWM, a robust and scalable framework for learning world models tailored to complex robotic tasks.
Leveraging a dual-autoregressive mechanism, RWM effectively addresses key challenges such as compounding errors, partial observability, and stochastic dynamics.
By incorporating historical context and self-supervised training over long prediction horizons, RWM achieves superior accuracy and robustness without relying on domain-specific inductive biases, enabling generalization across diverse tasks.
Through extensive experiments, we demonstrate that RWM consistently outperforms state-of-the-art approaches like RSSM and transformer-based architectures in autoregressive prediction accuracy across diverse robotic environments.
Building on RWM, we propose MBPO-PPO, a policy optimization framework that leverages long world model rollout fidelity.
Policies trained using MBPO-PPO demonstrate superior performance in simulation and transfer seamlessly to hardware, as evidenced by zero-shot deployment on the ANYmal D and Unitree G1 robots.
This work advances the field of model-based reinforcement learning by providing a generalizable, efficient, and scalable framework for learning and deploying world models.
The results highlight RWM ’s potential to enable adaptive, robust, and high-performing robotic systems, setting a foundation for broader adoption of model-based approaches in real-world applications.

### Acknowledgments and Disclosure of Funding

This research was supported by the ETH AI Center.

## Appendix A Technical Appendices and Supplementary Material

#### A.1 Task Representation

##### A.1.1 Observation and action spaces

The observation space for the ANYmal D and Unitree G1 world model is composed of base linear and angular velocities $v$, $\omega$ in the robot frame, measurement of the gravity vector in the robot frame $g$, joint positions $q$, velocities $\dot{q}$ and torques $\tau$ as in Table [S2](#A1.T2).

*Table S2: World model observation space*

The privileged information is used to provide an additional learning objective that implicitly embeds critical information for accurate long-term
predictions.
The space is composed of knee and foot contacts as in Table [S3](#A1.T3).

*Table S3: World model privileged information space*

The action space is composed of joint position targets as in Table [S4](#A1.T4).

*Table S4: Action space*

The observation space for the ANYmal velocity tracking policy is composed of base linear and angular velocities $v$, $\omega$ in the robot frame, measurement of the gravity vector in the robot frame $g$, velocity command $c$, joint positions $q$ and velocities $\dot{q}$ as in Table [S5](#A1.T5).

*Table S5: Policy observation space*

##### A.1.2 Reward functions

The total reward is sum of the following terms with weights detailed in Table [S6](#A1.T6).

*Table S6: Reward weights*

### Linear velocity tracking $x,y$

| | $$r_{v_{xy}}=w_{v_{xy}}e^{-\|c_{xy}-v_{xy}\|_{2}^{2}/\sigma_{v_{xy}}^{2}},$$ | |
|---|---|---|---|---|

where $\sigma_{v_{xy}}=0.25$ denotes a temperature factor, $c_{xy}$ and $v_{xy}$ denote the commanded and current base linear velocity.

### Angular velocity tracking

| | $$r_{\omega_{z}}=w_{\omega_{z}}e^{-\|c_{z}-\omega_{z}\|_{2}^{2}/\sigma_{\omega_{z}}^{2}},$$ | |
|---|---|---|---|---|

where $\sigma_{\omega_{z}}=0.25$ denotes a temperature factor, $c_{z}$ and $\omega_{z}$ denote the commanded and current base angular velocity.

### Linear velocity $z$

| | $$r_{v_{z}}=w_{v_{z}}\left\|v_{z}\right\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $v_{z}$ denotes the base vertical velocity.

### Angular velocity $x,y$

| | $$r_{\omega_{xy}}=w_{\omega_{xy}}\left\|\omega_{xy}\right\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $\omega_{xy}$ denotes the current base roll and pitch velocity.

### Joint torque

| | $$r_{q_{\tau}}=w_{q_{\tau}}\left\|\tau\right\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $\tau$ denotes the joint torques.

### Joint acceleration

| | $$r_{\ddot{q}}=w_{\ddot{q}}\left\|\ddot{q}\right\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $\ddot{q}$ denotes the joint acceleration.

### Action rate

| | $$r_{\dot{a}}=w_{\dot{a}}\|a^{\prime}-a\|_{2}^{2},$$ | |
|---|---|---|---|---|

where $a^{\prime}$ and $a$ denote the previous and current actions.

### Feet air time

| | $$r_{f_{a}}=w_{f_{a}}t_{f_{a}},$$ | |
|---|---|---|

where $t_{f_{a}}$ denotes the sum of the time for which the feet are in the air.

### Undesired contacts

| | $$r_{c}=w_{c}c_{u},$$ | |
|---|---|---|

where $c_{u}$ denotes the counts of the undesired contacts.

### Flat orientation

| | $$r_{g}=w_{g}g_{xy}^{2},$$ | |
|---|---|---|

where $g_{xy}$ denotes the $xy$-components of the projected gravity.

### Foot clearance

| | $$r_{f_{c}}=w_{f_{c}}h_{f_{c}},$$ | |
|---|---|---|

where $h_{f_{c}}$ denotes the clearance height of the swing feet.

### Joint deviation

| | $$r_{q_{d}}=w_{q_{d}}\left\|q-q_{0}\right\|_{1},$$ | |
|---|---|---|---|---|

where $q_{0}$ denotes the default joint position.

#### A.2 Network Architecture

##### A.2.1 RWM

The robotic world model consists of a GRU base and MLP heads predicting the mean and standard deviation of the next observation and privileged information such as contacts, as detailed in Table [S7](#A1.T7).
The training scheme is visualized in Fig. [S6](#A1.F6).

![Figure](x8.png)

*Figure S6: Dual-autoregressive mechanism employed in RWM. Inner autoregression updates GRU hidden states after each historical step within the context horizon, while outer autoregression feeds predicted observations from the forecast horizon back into the network. The dashed arrows denote the sequential autoregressive prediction steps, highlighting robustness to long-term dependencies and transitions.*

*Table S7: RWM architecture*

##### A.2.2 Baselines

The network architectures of the baselines are detailed in Table [S8](#A1.T8).

*Table S8: Baseline architecture*

##### A.2.3 MBPO-PPO

The network architectures of the policy and the value function used in MBPO-PPO are detailed in Table [S9](#A1.T9).
The training scheme is visualized in Fig. [S7](#A1.F7).

![Figure](x9.png)

*Figure S7: Model-Based Policy Optimization with learned world models. The framework combines real environment interactions with simulated rollouts for efficient policy optimization. Observation and action pairs from the environment are stored in a replay buffer and used to train the autoregressive world model. Imagination rollouts using the learned model predict future states over a horizon of $T$, providing trajectories for policy updates through reinforcement learning algorithms.*

*Table S9: Policy and value function architecture*

#### A.3 Training Parameters

The learning networks and algorithm are implemented in PyTorch 2.4.0 with CUDA 12.6 and trained on an NVIDIA RTX 4090 GPU.

##### A.3.1 RWM

The training information of RWM is summarized in Table [S10](#A1.T10).

*Table S10: RWM training parameters*

##### A.3.2 MBPO-PPO

The training information of MBPO-PPO is summarized in Table [S11](#A1.T11).

*Table S11: MBPO-PPO training parameters*

#### A.4 Additional Experiments and Discussions

##### A.4.1 Dual-autoregressive Mechanism

The heatmap on the left in Fig. [S8](#A1.F8) shows the relative autoregressive prediction error $e$ under different combinations of $M$ and $N$.
Models trained with a longer history horizon $M$ consistently exhibit lower prediction errors, demonstrating the importance of providing sufficient historical context to capture the underlying dynamics.
However, the influence of $M$ plateaus beyond a certain point, indicating diminishing returns for very large history horizons.
Forecast horizon $N$, on the other hand, plays a decisive role in improving long-term prediction accuracy.
Increasing $N$ during training leads to better performance in autoregressive rollouts, as it encourages the model to learn representations robust to compounding errors over extended prediction horizons.
This improvement comes at the cost of increased training time, as shown in the heatmap on the right.
Larger $N$ values require sequential computation during training due to the autoregressive nature of the process, significantly lengthening the training duration.

![Figure](x10.png)

*Figure S8: Ablation study on the history horizon $M$ and forecast horizon $N$ in RWM. The heatmap on the left shows the relative autoregressive prediction error, with darker colors indicating higher errors. Models trained with larger history horizons $M$ exhibit lower errors, although the improvements plateau beyond a certain point. Forecast horizon $N$ has a significant impact, with longer horizons leading to better long-term prediction accuracy due to exposure to extended rollouts during training. The heatmap on the right illustrates training time, with darker colors representing longer durations. Increasing $N$ significantly raises training time due to sequential computation, while shorter horizons (e.g., $N=1$, teacher-forcing) enable faster training but result in poor prediction accuracy.*

Interestingly, when the forecast horizon $N=1$ (teacher-forcing), training can be highly parallelized, resulting in minimal training time.
However, this setting leads to poor autoregressive performance, as the model lacks exposure to long-horizon prediction during training and fails to effectively handle compounding errors.
From the results, an optimal trade-off emerges: moderate values of $M$ and $N$ balance prediction accuracy and training efficiency.
For instance, a history horizon of $M=32$ and forecast horizon of $N=8$ achieve strong autoregressive performance with manageable training time.
These settings ensure sufficient historical context while training the model for robust long-term predictions.
Overall, the results highlight the critical interplay between history and forecast horizons in autoregressive training.
While extending both $M$ and $N$ improves accuracy, practical considerations of computational cost necessitate careful tuning of these hyperparameters to achieve optimal performance.

##### A.4.2 Visualization of Imagination Rollouts

The imagination rollouts across various robotic environments compared with the ground-truth simulation is visualized in Fig. [S9](#A1.F9).

![Figure](x11.png)

*Figure S9: Autoregressive imagination of RWM and ground-truth simulation across diverse robotic systems. For each environment, the top row showcases the RWM autoregressively predicting future trajectories in imagination. The second row visualizes the ground truth evolution in simulation. The visualized coordinate and arrow markers denote the predicted and measured end-effector pose and base velocity, respectively.*

##### A.4.3 Collision Handling and Model Pretraining

In both phases of the pretraining and online fine-tuning of RWM, we terminate rollouts and reset the environment when ground contact by the base is detected, signaling a failure.
We explicitly train RWM to predict such terminations in its privileged information prediction head.
This enables the world model to learn transitions leading to unsafe situations.
During policy optimization, MBPO-PPO treats these termination predictions as episode-ending events in imagination rollouts, affecting PPO’s return computation and state values.

RWM is pretrained with simulation data induced by policies trained for similar tasks under varied dynamics.
The policy is learned from scratch purely in imagination, with RWM fine-tuned using a single-environment online dataset.
Pretraining is essential for two key reasons.
First, the online dataset is extremely limited, as it is generated by only a single environment, akin to real-world constraints.
Training the world model entirely from scratch on such data would lead to severe overfitting and long training times.
Second, an immature policy would frequently cause the robot to fall, generating transitions with limited value.
In cases of significant failure or domain shift, training the world model solely on these data would result in chaotic imagined rollouts, which in turn would produce poor policy updates.
Pretraining stabilizes training and serves as a robust initialization for online fine-tuning, particularly in environments with challenging dynamics.

Importantly, RWM pretraining does not require data from optimal policies.
Figure [3](#S4.F3) demonstrate that RWM remains robust to domain shifts and injected noise.
As an alternative, we warm up the model using data from a suboptimal policy, which significantly stabilizes training.
Notably, this pretraining is only necessary for locomotion tasks due to the discontinuous dynamics and environment terminations.
Our manipulation experiments do not require such pretraining.

##### A.4.4 Challenges in Real-World Online Learning

We acknowledge that the advantages of our approach would be further demonstrated by performing the policy training phase directly on real hardware.
While this is a key long-term objective, several challenges currently prevent real-world deployment.

During online learning, the policy often exploits minor world model errors, leading to overly optimistic behaviors that result in collisions.
In simulation, these failures serve as corrective signals, but in real hardware, they pose a risk to the robot.
Our experiments show that such failures occur more than 20 times on average during online learning, which would be detrimental to real-world systems.
Even if hardware collisions were acceptable, fully automating online learning would require a recovery policy capable of resetting the robot to an initial state—a particularly challenging requirement for large platforms like ANYmal D or Unitree G1.
Additionally, privileged information used to fine-tune RWM (e.g., contact forces) must be either measured or estimated using onboard sensors, which may not always be available.
To mitigate error exploitation, uncertainty-aware world models could be explored, but integrating such models into RWM would require additional architectural modifications.
Due to these challenges, we approximate real-world constraints by using only a single simulation environment with domain shifts from pretraining environments.
This setup reduces engineering effort while proving the feasibility of our approach. Our ongoing work specifically addresses these issues.

#### A.5 Ethics and Societal Impacts

This work does not involve human subjects or sensitive data.
All experiments are conducted in simulation or on dedicated robotic hardware operated by the authors, with no use of third-party datasets.
The research complies with the Code of Ethics of the venue.
The proposed framework provides a robust and scalable method for learning world models tailored to complex robotic tasks.
This can benefit domains such as healthcare, disaster response, and logistics, and reduce environmental and hardware costs associated with physical experimentation.
Potential risks include misuse of the method in surveillance or autonomous enforcement systems, and the acceleration of automation in labor-sensitive sectors.
While such uses are not intended or explored in this work, the authors acknowledge the dual-use potential of generalizable control methods.
To mitigate safety risks, policy training occurs entirely in simulation, and deployment is limited to policies validated under domain shifts.
Failure events are explicitly modeled and used to terminate unsafe rollouts.
Online learning on hardware is deferred due to safety concerns and the absence of reliable recovery strategies.
Future work will explore uncertainty-aware models and safer online adaptation.

Generated on Sun Dec 14 07:23:12 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)