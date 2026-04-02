[# A Comprehensive Survey on World Models for Embodied AI Xinqing Li, Xin He, Le Zhang, , Min Wu, , Xiaoli Li, , and Yun Liu X. Li and Y. Liu are with the College of Computer Science, Nankai University, Tianjin 300350, China (e-mail: lixinqing@mail.nankai.edu.cn; liuyun@nankai.edu.cn). X. He is with the School of Computer Science and Engineering, Tianjin University of Technology, Tianjin 300384, China (e-mail: hexin@email.tjut.edu.cn). L. Zhang is with the School of Information and Communication Engineering, University of Electronic Science and Technology of China, Chengdu 611731, Sichuan, China (e-mail: lezhang@uestc.edu.cn). M. Wu is with the Institute for Infocomm Research (I2R), Agency for Science, Technology and Research (A*STAR), 138632, Singapore (e-mail: wumin@i2r.a-star.edu.sg). X. Li is with the Information Systems Technology and Design (ISTD) Pillar, Singapore University of Technology and Design (SUTD), 487372, Singapore (e-mail: xiaoli_li@sutd.edu.sg). Corresponding author: Yun Liu (e-mail: liuyun@nankai.edu.cn) This work was supported in part by the National Natural Science Foundation of China (No. 62576176) and in part by the Fundamental Research Funds for the Central Universities (Nankai University, No. 070-63253235). ###### Abstract Embodied AI requires agents that perceive, act, and anticipate how actions reshape future world states. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts to support perception, prediction, and decision making. This survey presents a unified framework for world models in embodied AI. Specifically, we formalize the problem setting and learning objectives, and propose a three-axis taxonomy encompassing: (1) Functionality, *Decision-Coupled* *vs. **General-Purpose*; (2) Temporal Modeling, *Sequential Simulation and Inference* *vs. **Global Difference Prediction*; (3) Spatial Representation, *Global Latent Vector*, *Token Feature Sequence*, *Spatial Latent Grid*, and *Decomposed Rendering Representation*. We systematize data resources and metrics across robotics, autonomous driving, and general video settings, covering pixel prediction quality, state-level understanding, and task performance. Furthermore, we offer a quantitative comparison of state-of-the-art models and distill key open challenges, including the scarcity of unified datasets and the need for evaluation metrics that assess physical consistency over pixel fidelity, the trade-off between model performance and the computational efficiency required for real-time control, and the core modeling difficulty of achieving long-horizon temporal consistency while mitigating error accumulation. Finally, we maintain a curated bibliography at https://github.com/Li-Zn-H/AwesomeWorldModels](https://github.com/Li-Zn-H/AwesomeWorldModels).

###### Index Terms:

World Models, Embodied AI, Temporal Modeling, Spatial Representation.

## I Introduction

Embodied AI aims to equip agents to perceive complex, multimodal environments, act within them, and anticipate how their actions will alter future world states [batra2020rearrangement, gupta2024essential]. Central to this capability is the world model, an internal simulator that captures environment dynamics to support both forward and counterfactual rollouts for perception, prediction, and decision making [liu2025aligning, ding2024understanding]. This survey focuses on world models that yield actionable predictions for embodied agents, distinguishing them from static scene descriptors or purely generative visual models that do not capture controllable dynamics.

Cognitive science suggests humans construct internal models of the world by integrating sensory inputs. These models do not merely predict and simulate future events but also shape perception and guide action [clark1998being, barsalou1999perceptions, friston2010free]. Motivated by this view, early AI research on world models was rooted in model-based reinforcement learning (RL), where latent state-transition models were used to improve sample efficiency and planning performance [fung2025embodied]. The seminal work of Ha and Schmidhuber [ha2018recurrent] crystallized the term world model and inspired the Dreamer series [hafner2020dream, hafner2021mastering, hafner2025mastering], highlighting how learned dynamics can drive imagination-based policy optimization. More recently, advances in large-scale generative modeling and multimodal learning have expanded world models beyond their initial focus on policy learning into general-purpose environment simulators capable of high-fidelity future prediction, exemplified by models like Sora [brooks2024video] and V-JEPA 2 [assran2025v]. This expansion has diversified functional roles, temporal modeling strategies, and spatial representations, while introducing inconsistencies in terminology and taxonomy across sub-communities.

Faithfully capturing environment dynamics requires addressing both the temporal evolution of states and the spatial encoding of scenes [liu2025aligning]. Long-horizon rollouts are susceptible to error accumulation, which establishes coherence as a central challenge in video prediction and policy imagination [venkatraman2015improving, asadi2018lipschitz]. Similarly, coarse or 2D-centric layouts provide insufficient geometric detail for handling challenges such as occlusion, object permanence, and geometry-aware planning. In contrast, volumetric or 3D occupancy representations such as neural fields [agro2024uno] and structured voxel grids [wei2024occllama] provide geometric structure that better supports forecasting and control. Taken together, these points establish temporal modeling and spatial representation as core design dimensions that fundamentally influence the predictive horizon, physical fidelity, and downstream performance of embodied agents.

Several recent surveys have organized the rapidly growing literature on world models. Overall, these surveys follow two main approaches. The first is a function-oriented perspective. For example, Ding et al. [ding2024understanding] categorized relevant works based on the two core functions of understanding and prediction, while Zhu et al. [zhu2024sora] presented a framework based on the core capabilities of world models. The second approach is application-driven, focusing on specific domains such as autonomous driving. Notably, Guan et al. [guan2024world] and Feng et al. [feng2025survey] provided overviews of world model techniques for autonomous driving.

To address the lack of a unified taxonomy in the context of embodied AI, this work introduces a framework centered on the three core axes of functionality, temporal modeling, and spatial representation. At the functional level, this framework distinguishes between decision-coupled and general-purpose models. At the temporal level, it differentiates between Sequential Simulation and Inference versus Global Difference Prediction. Finally, at the spatial level, it encompasses a range of representations from latent features to explicit geometry and neural fields. This framework offers a unified structure for organizing existing approaches and integrates standardized datasets and evaluation metrics. This structure facilitates quantitative comparisons and provides a panoramic, actionable knowledge map for future research.

Fig. [1](#S1.F1) presents an overview of the structure and taxonomy of this paper. We begin in §[II](#S2) by outlining the core concepts and theoretical foundations of world models. §[III](#S3) introduces our three-axis taxonomy and maps representative methods onto this framework. §[IV](#S4) surveys the datasets and evaluation metrics used for training and assessment. §[V](#S5) offers a quantitative comparison of state-of-the-art models. §[VI](#S6) discusses open challenges and promising research directions, and §[VII](#S7) concludes the survey.

![Figure](x2.png)
![Figure](x3.png)
![Figure](x4.png)
![Figure](x5.png)
![Figure](x6.png)
![Figure](x7.png)
![Figure](x8.png)
![Figure](x9.png)
![Figure](x10.png)
![Figure](x11.png)

*Figure 1: Structure of this survey. The figure classifies world models along three axes and illustrates representative methods for each, providing a unified view of the field. Figure design inspired in part by [assran2025v, hafner2025mastering, lu2025infinicube, chen2024drivinggpt].*

## II Background

### II-A Core Concepts

As discussed in §[I](#S1), a world model functions as an internal simulator of environmental dynamics. Its functionality rests on three aspects:

-
•

Simulation & Planning, which uses learned dynamics to generate plausible future scenarios, allowing agents to assess potential actions through imagination without real-world interaction.

-
•

Temporal Evolution, which learns how the encoded state evolves, enabling temporally consistent rollouts.

-
•

Spatial Representation, which encodes scene geometry at an appropriate fidelity, using formats such as latent tokens or neural fields to provide context for control.

These three pillars provide the conceptual foundation for the taxonomy introduced in §[III](#S3) and are formalized in the mathematical framework that follows.

### II-B Mathematical Formulation of World Models

We formalize the environment interaction as a POMDP [smallwood1973optimal]. For notational consistency, we define a null initial action $a_{0}$ at $t=0$, which allows the dynamics to be written uniformly. At each step $t\geq 1$, the agent receives an observation $o_{t}$ and takes an action $a_{t}$, while the true state $s_{t}$ remains unobserved. To handle this partial observability, world models infer a learned latent state $z_{t}$ using a one-step filtering posterior, where the previous latent state $z_{t-1}$ is assumed to summarize the relevant history. Finally, $z_{t}$ is used to reconstruct $o_{t}$:

$$\begin{array}{ll}\text{Dynamics Prior:}&p_{\theta}(z_{t}\mid z_{t-1},a_{t-1})\\ \text{Filtered Posterior:}&q_{\phi}(z_{t}\mid z_{t-1},a_{t-1},o_{t})\\ \text{Reconstruction:}&p_{\theta}(o_{t}\mid z_{t})\end{array}.$$ \tag{1}

Consistent with the Markovian structure, the joint distribution over observations and latent states factorizes as:

$$p_{\theta}(o_{1:T},z_{0:T}\!\mid\!a_{0:T-1})\!=\!p_{\theta}(z_{0})\!\prod_{t=1}^{T}\!p_{\theta}(z_{t}\!\mid\!z_{t-1},a_{t-1})p_{\theta}(o_{t}\!\mid\!z_{t}).$$ \tag{2}

To infer the latent states, we must approximate the intractable true posterior $p_{\theta}(z_{0:T}\!\mid\!o_{1:T},a_{0:T-1})$ with a time-factorized variational distribution:

$$q_{\phi}(z_{0:T}\!\mid\!o_{1:T},a_{0:T-1})=q_{\phi}(z_{0}\!\mid\!o_{1})\!\prod_{t=1}^{T}\!q_{\phi}(z_{t}\!\mid\!z_{t-1},a_{t-1},o_{t}),$$ \tag{3}

which indeed reduces to the action-free case when ignoring $a$ inputs. Directly maximizing the log-likelihood $\log p_{\theta}(o_{1:T}\!\mid\!a_{0:T-1})$ is intractable. Instead, we optimize an ELBO using the approximate posterior $q_{\phi}$, which provides a tractable objective for learning the model parameters:

$$
\begin{aligned}
\log p_{\theta}( &o_{1:T}\!\mid\!a_{0:T-1})=\!\log\!\int p_{\theta}(o_{1:T},z_{0:T}\!\mid\!a_{0:T-1})\,dz_{0:T} \tag{4}
\end{aligned}
$$
| | | $\displaystyle\geq\mathbb{E}_{q_{\phi}}\Big[\log\frac{p_{\theta}(o_{1:T},z_{0:T}\mid a_{0:T-1})}{\,q_{\phi}(z_{0:T}\mid o_{1:T},a_{0:T-1})}\Big]=:\mathcal{L}(\theta,\phi).$ | |

Under the assumption of Markov factorization for both $p_{\theta}$ and $q_{\phi}$, this ELBO decomposes into a reconstruction objective and a KL regularization term:

$$
\begin{aligned}
\mathcal{L}(\theta, &\phi)=\sum_{t=1}^{T}\mathbb{E}_{q_{\phi}(z_{t})}\!\big[\log p_{\theta}(o_{t}\mid z_{t})\big] \tag{5}
\end{aligned}
$$
| | | $\displaystyle-D_{\mathrm{KL}}\!\big(q_{\phi}(z_{0:T}\mid o_{1:T},a_{0:T-1})\,\|\,p_{\theta}(z_{0:T}\mid a_{0:T-1})\big).$ | |

Modern world models thus adopt a reconstruction–regularization training paradigm: the likelihood term $\log p_{\theta}(o_{t}\!\mid\!z_{t})$ encourages faithful observation prediction, and KL regularization terms align the filtered posterior $q_{\phi}(z_{t}\!\mid\!z_{t-1},a_{t-1},o_{t})$ with the dynamics prior $p_{\theta}(z_{t}\!\mid\!z_{t-1},a_{t-1})$. Such world models can be instantiated with recurrent models [sun2024learning, zhai2025recurrent, aljalbout2025accelerating], Transformer-based architectures [chen2022transdreamer, robine2023transformerbased, zhang2023storm], or diffusion-based decoders [zhu2025unified, qi2025strengthening, guo2025flowdreamer, huang2025enerverse, alhaija2025cosmos]. In all cases, the learned latent trajectory $z_{1:T}$ serves as a compact, predictive memory to support downstream policy optimization, model-predictive control, and counterfactual reasoning in embodied AI.

## III Taxonomy

We categorize world models along three core dimensions, which provide the foundation for the subsequent analysis in this survey.

The first dimension, decision coupling, distinguishes between Decision-Coupled* and *General-Purpose* world models. Decision-Coupled models are task-specific, learning dynamics optimized for a particular decision-making task. In contrast, General-Purpose models are task-agnostic simulators that focus on broad prediction, enabling generalization across various downstream applications.

The second dimension, temporal reasoning, delineates two distinct paradigms of prediction. *Sequential Simulation and Inference* models dynamics in an autoregressive manner, unfolding future states one step at a time. In contrast, *Global Difference Prediction* directly estimates entire future states in parallel, offering greater efficiency at the potential cost of reduced temporal coherence.

The third dimension, spatial representation, comprises four primary strategies used in current research to model spatial states:

-
1.

Global Latent Vector representations encode complex world states into a compact vector, enabling efficient real-time computation on physical devices.

-
2.

Token Feature Sequence representations model world states as sequences of tokens, focusing on capturing complex spatial, temporal, and cross-modal dependencies among tokens.

-
3.

Spatial Latent Grid representations incorporate spatial inductive biases into world models by leveraging geometric priors such as Bird’s-Eye View (BEV) features or voxel grids.

-
4.

Decomposed Rendering Representation involves decomposing 3D scenes into a set of learnable primitives, such as 3D Gaussian Splatting (3DGS) [kerbl20233d] or Neural Radiance Fields (NeRF) [mildenhall2020nerf], and then using differentiable rendering to achieve high-fidelity novel view synthesis.

The following tables apply this taxonomy to classify representative works. Tab. [I](#S3.T1) reviews approaches in robotics, while Tab. [II](#S3.T2) focuses on autonomous driving. Together, they provide a roadmap for the detailed analyses in the subsequent sections.

*TABLE I: A summary of representative world models in robotics and general-purpose domains.*

*TABLE II: A summary of representative world models for the autonomous driving domain.*

### III-A Decision-Coupled World Models

#### III-A1 Sequential Simulation and Inference

Global Latent Vector. Early decision-coupled world models combined sequential inference with global latent states. These approaches primarily use Recurrent Neural Networks (RNNs) for efficient real-time and long-horizon prediction.

Ha and Schmidhuber [ha2018recurrent] introduced an early world model that encodes observations into a latent space and employs an RNN to model dynamics for policy optimization. Building on this, PlaNet [hafner2019learning] introduced the Recurrent State-Space Model (RSSM), which fuses deterministic memory with stochastic components to enable robust long-horizon imagination. The successor models Dreamer, DreamerV2, and DreamerV3 [hafner2020dream, hafner2021mastering, hafner2025mastering] further advanced this formulation, inspiring a broad line of subsequent research.

Building on RSSM, several variants modify or eliminate the decoder to better capture dynamics. For example, Dreaming [okada2021dreaming] uses contrastive learning and linear methods to mitigate state shifts, whereas DreamerPro [deng2022dreamerpro] replaces the decoder with prototypes to suppress visual distractions. To further enhance robustness, HRSSM [sun2024learning] was proposed, featuring a dual-branch architecture that aligns latent observations and shares information without reconstruction. Beyond architectural refinements, DisWM [wang2025disentangled] disentangles semantic knowledge from video content, distilling it into a world model that enables cross-domain generalization.

A unifying theme across recent RSSM extensions is transferability, which reflects generalization across modalities, tasks, and embodiments for robust real-world robotics. At the representation level, PreLAR [zhang2024prelar] learns implicit action abstractions to bridge video-pretrained representations and control fine-tuning. Similarly, Wang et al. [wang2025latent] used optical flow as an embodiment-agnostic action representation to refine behavior-cloned policies, facilitating transfer across diverse embodiments. SENSEI [sancaktar2025sensei] distilled a Vision-Language Model (VLM) to derive semantic rewards and employed an RSSM that learns to predict and propagate these rewards internally. Under limited supervision, SR-AIF [nguyen2025sr] exploits prior preference learning and self-revision to enable adaptive learning in sparse-reward, continuous-control settings. To mitigate the Sim-to-Real (S2R) gap, ReDRAW [lanier2025adapting] is pretrained in simulation and adapted to the real environment using a limited amount of reward-free data, applying residual corrections to the latent dynamics. To handle mismatches, AdaWM [wang2025adawm] identifies discrepancies between the learned dynamics and the planner and selectively fine-tunes critical components. Other methods like WMP [lai2025world] address S2R transfer for challenging tasks, and DayDreamer [wu2023daydreamer] demonstrated sample-efficient deployment on physical robots. To broaden transfer, FOUNDER [wang2025founder] grounds representations from foundation models in the world-model state space, using temporal-distance prediction to handle flexible goals, and LUMOS [nematollahi2025lumos] introduced a language-conditioned imitation framework that operates on-policy in latent space with intrinsic rewards, enabling zero-shot transfer to real-world robotics.

RSSM-based models have also been developed for autonomous driving. MILE [hu2022model] leverages offline expert data to enable imagined future states for planning. SEM2 [gao2024enhance] integrates semantic filtering with multi-source sampling to extract driving-relevant features and balance data distribution. Popov et al. [popov2025mitigating] addressed covariate shift through a latent generative world model that realigns policies with expert states. For safety, VL-SAFE [qu2025vl] supervises world models using safety scores derived from a VLM to generate safe trajectories. Finally, CALL [wang2025ego] extended the RSSM framework to Multi-Agent RL by introducing ego-centric information sharing to enhance planning capabilities.

In contrast to RSSM, TransDreamer [chen2022transdreamer] introduced a Transformer State-Space Model (TSSM) that replaces the recurrent core in Dreamer, thereby substantially enhancing its capacity to capture long-horizon dependencies. The complementary OSVI-WM [goswami2025osvi] employs a causal Transformer for one-shot Imitation Learning (IL), autoregressively predicting the future latent trajectory and decoding it into physical waypoints for robot control.

Some approaches continue to employ RNNs to capture temporal dependencies. On the modeling side, RWM [li2025robotic] introduced a dual-autoregressive, domain-agnostic neural simulator for long-horizon prediction. X-MOBILITY [liu2025x], in contrast, disentangles modeling from policy learning, using multi-head decoders for large-scale pretraining followed by supervised fine-tuning to derive strong policies. For humanoid locomotion, DWL [gu2024advancing] and WMR [sun2025learning] adopted end-to-end (E2E) frameworks. These frameworks reconstruct states from partial observations using either denoising or a gradient-blocked state estimator, which enables zero-shot transfer across complex real-world terrains.

Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as alternatives to RNNs and Transformers, combining linear-time complexity with long-horizon modeling capacity. Building on this, GLAM [he2025glam] improves both fidelity and efficiency via a Mamba-based parallel framework that integrates global and local modules to capture contextual and fine-grained dynamics.

Beyond forward temporal modeling, Inverse Dynamics Modeling (IDM) is a key paradigm in world model construction. An IDM infers the actions required to transition between an initial and a target state. Agrawal et al. [agrawal2016learning] integrated a forward model with an IDM for multi-step prediction, establishing the basis for subsequent research. More recent work includes GLAMOR [paster2021planning], which trains an object-conditioned IDM to predict the actions necessary to reach a specified target. In Dreamer-style agents, Iso-Dream[pan2022iso] leverages IDM to decompose the world model into controllable and uncontrollable components, using the rollout of uncontrollable states to guide policy learning.

Token Feature Sequence. The Token Feature Sequence paradigm centers on modeling dependencies among discrete tokens. This representation supports causal inference, multimodal integration, and the reuse of Large Language Model (LLM).

Recent RSSM-centric studies have begun to exploit token-level dependencies to strengthen representation learning and temporal reasoning. For example, MWM [seo2023masked] decouples visual tokens from RSSM-based dynamics via a masked autoencoder, improving both performance and data efficiency. NavMorph [yao2025navmorph] introduced a self-evolving RSSM with a contextual evolution memory for online adaptation. For temporal abstraction, WISTER [burchi2025learning] employed action-conditioned contrastive predictive coding to train a TSSM that captures high-level temporal features. Similarly, TWM [robine2023transformerbased] used a Transformer to align multimodal tokens with historical states during training, while relying on a lightweight policy at inference. To handle long-horizon tasks, some approaches integrate LLMs with RSSMs to decompose objectives into subtasks. EvoAgent [feng2025evoagent], for example, uses an LLM to guide low-level actions and regularizes RSSM updates. RoboHorizon [chen2025robohorizon], in contrast, enhances task recognition with dense rewards and leverages key task segments via a masked autoencoder.

In autonomous driving, token-based sequence representations are increasingly adopted to model cross-modal interactions and spatiotemporal structures. DrivingWorld [hu2024drivingworld] pairs next-state prediction for temporal dynamics with next-token prediction for spatial structure. For multimodal control, Doe-1 [zheng2024doe] formulates closed-loop driving as autoregressive prediction over perception-description-action tokens, which unifies perception, prediction, and planning, and DrivingGPT [chen2024drivinggpt] interleaves vision and action tokens and casts world modeling and trajectory planning as next-token prediction. To enhance diversity and safety, LatentDriver [xiao2025learning] models future actions as a mixture distribution and actuates the world model with planner-sampled intermediate actions. At the same time, Vasudevan et al. [vasudevan2025planning] proposed an adaptive model that predicts surrounding agents for safe navigation.

The token-based paradigm also extends to broader robotics. Within RL, IRIS [micheli2023transformers] and TWM [dedieu2025improving] leverage discrete tokens to enable data-efficient policy learning via imagined or hybrid rollouts. DyWA [lyu2025dywa] improves action learning by conditioning on trajectory dynamics and jointly predicting future states with single-view point-cloud and proprioceptive modalities. EgoAgent [chen2025egoagent] interleaves state-action sequence modeling within a Transformer, enabling unified perception, prediction, and action inference. Tokenized representations unify multimodal inputs, including vision, language, and action (VLA), enabling generalist agents with cross-domain adaptability, as shown in WorldVLA [cen2025worldvla]. Recent studies encode environmental states as discrete symbolic tokens and condition next-token prediction on action, as demonstrated by DCWM [scannell2025discrete] and TrajWorld [yin2025trajectory].

Recent studies have strengthened the link between tokenized representations and planning, particularly through object-centric approaches. These models, such as CarFormer [hamdan2024carformer], the work of Jeong et al. [jeong2025object], and Dyn-O [wang2025dyn], represent scenes as a collection of slots. CarFormer autoregressively models the relationships between these slots in BEV. Jeong et al. added language-guided manipulation, while Dyn-O used a Mamba with dropout scheduling for temporal modeling and to disentangle static from dynamic elements.
$\Delta$-IRIS [micheli2024efficient] introduced a hybrid Transformer that integrates tokens with stochastic $\Delta$-tokens to capture dynamics. $\text{D}^{2}$PO [wang2025world] employed preference learning to jointly optimize state prediction and action selection, enhancing the model’s understanding of underlying dynamics. For efficiency, MineWorld [guo2025mineworld] accelerated token generation by predicting sequences in parallel and introduced an IDM as a controllability metric. Meanwhile, PIVOT-R [zhang2024pivot] and ReOI [chen2025reimagination] incorporated VLMs into control. PIVOT-R parses instructions to produce waypoint-based plans that an action module decodes into low-level controls, whereas ReOI detects implausible prediction elements, reimagines the distractors, and reintegrates the corrected content.

Based on tokenization, some studies employ autoregressive diffusion to achieve stable generation and long-horizon planning. Epona [zhang2025epona] decouples spatiotemporal modeling, handled by a Transformer, from long-horizon multimodal generation, realized through trajectory and vision Diffusion Transformers (DiTs). Goff et al. [goff2025learning] used a DiT to instantiate the state transition, which enables on-policy training and multi-second closed-loop rollouts. SceneDiffuser++[tan2025scenediffuser++] pushes this further to city-scale traffic simulation, applying multi-tensor diffusion over agents and traffic lights to produce stable closed-loop rollouts. For navigation, NWM [bar2025navigation] introduced an efficient conditional DiT to simulate visual trajectories for zero-shot planning.

Another emerging direction is to inject explicit reasoning into the world model using LLMs and Chain-of-Thought (CoT). NavCoT [lin2025navcot] decomposes navigation into imagination, filtering, and prediction, enabling parameter-efficient in-domain training, and ECoT [zawalski2025robotic] leverages a pipeline of foundation models to generate reasoning labels for training a VLA policy. Variants like MineDreamer [zhou2025minedreamer] introduced Chain-of-Imagination (CoI), where a multimodal LLM imagines future observations to steer diffusion and guide actions, and FSDrive [zeng2025futuresightdrive] generates physics-constrained future scenes and treats them as CoT supervision, enabling VLMs to function as IDMs for planning.

Other approaches directly couple LLMs with world models to operationalize planning and data generation. Dyna-Think [yu2025dyna] fuses reasoning and acting via a distilled LLM, and RIG [zhao2025rig] unifies reasoning and imagination end-to-end generalist policy. In terms of explicit dynamics and long-horizon, Gkountouras et al. [gkountouras2025language] trained a causal world-model simulator that grounds an LLM nvironment causal reasoning and planning skills, Statler [yoneda2024statler] enables LLMs to keep a structured world-state, using a reader for planning and a writer for updating, and Inner Monologue [huang2023inner] incorporates a closed-loop feedback into LLMs, enabling agents to reason and deliberate more akin to human thinking. Finally, WoMAP [yin2025womap] synthesized 3DGS scenes and trained a world model that refines VLM instructions for precise execution.

Spatial Latent Grid. By encoding features on geometry-aligned grids or incorporating explicit spatial priors, this paradigm preserves locality, enables efficient convolutional or attention-based updates and streaming rollouts.

In autonomous driving, many studies couple RNN-based dynamics with spatial grids to guide planning. For instance, DriveDreamer[wang2024drivedreamer] and GenAD[zheng2024genad] adopt GRU-based dynamics over grid or instance-centric tokens to predict motion and decode trajectories. In contrast, DriveWorld [min2024driveworld] and Raw2Drive [yang2025raw2drive] instantiate RSSM dynamics on BEV tokens. DriveWorld conditions on tokens and actions for joint prediction, while Raw2Drive adopts a dual-stream design for spatiotemporal learning.

Numerous studies focus on autoregressively predicting future 3D occupancy representations to enable motion planning for autonomous driving. One strand discretizes scenes into occupancy tokens for sequential prediction, exemplified by OccWorld [zheng2024occworld] and RenderWorld [yan2025renderworld]. Another strand directly forecasts volumetric features or embeddings, as in Drive-OccWorld [yang2025driving] and PreWorld [li2025semi]. Self-supervised variants predict future representations from current cues. For example, LAW [li2025enhancing] conditions on current representations and trajectories, SSR [li2025navigation] compresses scenes into sparse BEV tokens for future BEV features, and NeMo [huang2024neural] voxelizes multi-frame images and predicts occupancy to support imitation-based planning. Building on these representations, FASTopoWM [yang2025fastopowm] employs a unified decoder to align fast and slow systems from vehicle poses, enabling lane-topology reasoning, and WoTE [li2025end] simulates candidate trajectories in BEV and selects among them with a reward model. Extending the paradigm, OccLLaMA [wei2024occllama] unifies occupancy, actions, and text within a single token vocabulary and employs a LLaMA for next token forecasting, planning, and question answering.

Beyond autonomous driving, similar formulations have been extended to broader domains of robotics. WMNav [nie2025wmnav] leverages a VLM to maintain a curiosity-driven value map and adopts phased decision making to enable zero-shot, object-driven navigation. RoboOccWorld [zhang2025occupancy] targets indoor robotics by predicting fine-grained 3D occupancy with a pose-conditioned autoregressive Transformer, thereby supporting exploration and decision making. To achieve high-fidelity dynamics, EnerVerse [huang2025enerverse] applies chunk-wise autoregressive video diffusion with a sparse memory mechanism to produce 4D latent dynamics and integrates 4DGS to mitigate the S2R gap in robotic execution. For manipulation, ParticleFormer [huang2025particleformer] forecasts future point clouds with a Transformer-based particletized dynamics model, enabling robust handling of multi-object and multi-material interactions. At the representation level, DINO-WM [zhou2025dino] learns dynamics in the DINOv2 feature space and predicts future states to support zero-shot planning.

Decomposed Rendering Representation. This paradigm represents scenes with explicit renderable primitives such as NeRFs and 3DGS, updating them to simulate dynamics and render future observations. It provides view-consistent forecasts, object-level compositionality, and seamless integration with physics priors and digital twins, thereby supporting long-horizon rollouts.

Building on 3DGS, GAF [chai2025gaf] augments splats with learnable motion attributes to forecast future states and refines initial actions with diffusion. ManiGaussian [lu2024manigaussian] predicts per-point variations to generate future Gaussian scenes for manipulation under current states and actions, and ManiGaussian++ [yu2025manigaussian++] adds a hierarchical leader-follower design with task-oriented splats to model primitive deformations for multi-body and bimanual skills. Within simulation and digital twin coupling, DreMa [barcellona2025dream] integrates GS with a physics simulator to build twins for data synthesis in imitation learning, Abou-Chakra et al. [abou2025physically] introduced a dual Gaussian–Particle representation where Gaussian points are attached to particles driven by visual loss forces, DexSim2Real2 [jiang2025dexsim2real] builds twins of articulated objects with generative models and uses sampling based planning for precise manipulation, PIN-WM [li2025pin] combines 3DGS with differentiable physics to estimate physical parameters from limited observations and generates digital cousins for zero-shot S2R policy learning, and PWTF [ning2025prompting] constructs an interactive twin that simulates candidate action outcomes and uses VLMs for evaluation and selection. At the representation level, DTT [xu2025delta] adopts a triplane representation with multiscale Transformers to autoregressively capture incremental changes, forming a 4D world model for prediction and planning.

#### III-A2 Global Difference Prediction

Token Feature Sequence. The compact Global Latent Vector representation discards fine-grained spatiotemporal detail and is therefore rarely used for global prediction. In contrast, Token Feature Sequences predict the future sequence in parallel, reducing error accumulation while enabling multimodal diversity.

On the representation side, TOKEN [tian2025tokenize] tokenizes scenes into object-level tokens, aligning world representations with reasoning and leveraging LLMs to predict full future trajectories for long-tail scenarios. GeoDrive [chen2025geodrive] extracts 3D representations, renders trajectory-conditioned views, and edits the position of vehicles to guide DiT in producing editable generations. For control, FLARE [zheng2025flare] aligns diffusion policies with latent future representations, avoiding pixel-space video generation and learning effectively from action-free videos. In a related vein, LaDi-WM [huang2025ladi] predicts future states through interactive diffusion in a latent space aligned with visual foundation models, integrating geometric and semantic features while iteratively refining the diffusion policy to improve performance and generalization. villa-X [chen2025villa] and VidMan [wen2024vidman] both couple diffusion-based models with IDM for control. villa-X infers latent actions, aligns them with ego-centric forward dynamics, and maps them via joint diffusion, while VidMan adapts a pretrained video-diffusion model into an IDM using a self-attention adapter for accurate action prediction.

Spatial Latent Grid. Spatial grid models parallel-forecast BEV or voxel maps from ego-stabilized views, preserving locality and uncertainty while producing planner-ready maps for fast control.

Diffusion-based world models are commonly used for parallel generation. EmbodiedDreamer [wang2025embodiedreamer] couples differentiable physics with video diffusion to render photorealistic and physically consistent futures. TesserAct [zhen2025tesseract] reconstructs 4D spatiotemporal consistent scenes by jointly generating RGB, depth, and normal videos for IDM-based action learning. DFIT-OccWorld [zhang2024efficient] reformulates prediction as decoupled voxel warping and adopts image-assisted single-stage training for reliable and efficient dynamic scene modeling. For instruction-conditioned control, RoboDreamer [zhou2024robodreamer] decomposes instructions into low-level primitives that steer video diffusion, synthesizing novel compositional scenes beyond the training distribution while grounding execution via an IDM, and ManipDreamer [li2025manipdreamer] extends this design with an action-tree prior together with depth and semantic guidance to improve instruction following and temporal consistency.

On the planning side, 3DFlowAction [zhi20253dflowaction] employs a pretrained 3D optical flow world model to treat future motion as a unified action cue, enabling label-free and cross-robot manipulation through closed-loop optimization. Imagine-2-Drive [garg2025imagine] integrates video diffusion with a multimodal diffusion policy to accelerate policy learning. Drive-WM [wang2024driving] uses multi-view diffusion with image-based rewards to select safer trajectories, while World4Drive [zheng2025world4drive] leverages vision-based priors to construct intent-aware world models that support self-supervised multi-intent imagination. COMBO [zhang2025combo] composes multi-agent actions with diffusion, leverages VLMs to infer purposes, and integrates tree search for online cooperative planning.

### III-B General-Purpose World Models

#### III-B1 Sequential Simulation and Inference

Token Feature Sequence. General-purpose models pretrain task-agnostic dynamics to capture environmental physics and generate future scenes, prioritizing transferability over specific tasks.

Some general world models increasingly pretrain on unlabeled video and use tokenized latent space for robust forecasting and generation. iVideoGPT [wu2024ivideogpt] was pretrained on large-scale interaction videos for action-free forecasting and later adapted to downstream control. Genie [bruce2024genie] learned discrete latent actions and spatiotemporal tokens, enabling user-controllable interactive environments via autoregressive dynamics. RoboScape [shang2025roboscape] jointly learned video generation with temporal depth and keypoint dynamics to improve physical realism. PACT [bonatti2023pact] tokenized multimodal perception and action and trained a causal Transformer to obtain a unified representation for diverse tasks, and DINO-world [baldassarre2025back] learns generalizable dynamics by predicting the temporal evolution of DINOv2 features from large-scale unlabeled video corpora. Building on language priors, EVA [chi2025empowering] introduced a Reflection-of-Generation (RoG) policy that used a VLM for iterative self-correction, strengthening long-horizon anticipation. In the same vein, Owl-1 [huang2024owl] employs a VLM to forecast world dynamics conditioned on current states and generated fragments, explicitly guiding the subsequent fragments and enabling coherent long-horizon video synthesis, while World4Omni [chen2025world4omni] employs a reflective world model, where a VLM refines subgoal images from an image generator, and integrates them with pretrained modules for zero-shot robotic manipulation.

Recent work adapts video diffusion models into controllable world models that autoregressively imagine future scenes. AdaWorld [gao2025adaworld] introduced an action-aware pretraining scheme that extracted self-supervised latent actions between adjacent frames to condition diffusion, enabling efficient transfer with minimal interaction. Vid2World [huang2025vid2world] adapts pretrained video diffusion models into autoregressive interactive world models via causalization and a causal action-guidance mechanism. GenAD [yang2024generalized] employs a two-stage strategy to adapt diffusion into a general video-prediction model conditioned on text and actions, enabling large-scale driving simulation and planning. Pandora [xiang2024pandora] uses an instruction-tuned LLM to autoregressively steer a separate video-diffusion generator for explicit, goal-directed control, while Yume [mao2025yume] quantized camera motions into text tokens to guide a masked Video DiT, enabling autoregressive synthesis of dynamic 3D exploratory worlds.

To maintain geometric fidelity and long-horizon stability, recent methods couple explicit 3D priors with temporal-consistency modules in diffusion-based world models. At the geometric level, Geometry Forcing [wu2025geometry] aligns latent features with a geometric foundation model to inject explicit 3D priors, improving geometric consistency, while DeepVerse [chen2025deepverse] integrates visual and geometric prediction targets and introduces a geometry-aware memory to sustain consistent long-horizon generation. For temporal stability, VRAG [chen2025learning] proposed a Video Retrieval-Augmented Generation (RAG) framework that retrieves historical frames conditioned on a global state to stabilize autoregressive rollouts, StateSpaceDiffuser [savov2025statespacediffuser] combines Mamba with diffusion to alleviate long-term memory loss and content drift under short context windows, and InfinityDrive [guo2024infinitydrive] injects memory and adopts an adaptive loss within DiT, producing minute-scale driving videos with high fidelity, temporal consistency, and diverse content. Complementing these designs, LongDWM [wang2025longdwm] mitigates error accumulation in long-horizon video generation through distillation—where a fine-grained DiT learns continuous motion to guide a coarse model, whereas MiLA [wang2025mila] adopts a coarse-to-fine strategy that predicts sparse anchor frames and refines them during interpolation to improve temporal consistency and long-term fidelity. Finally, for dynamics and conditioning, Orbis [mousakhan2025orbis] employs a continuous-space flow-matching formulation that demonstrates greater robustness than discrete-token schemes for long-horizon rollouts, and DriVerse [li2025driverse] leverages multimodal trajectory prompts with latent motion alignment to synthesize long-horizon driving videos from a single image and navigation trajectories.

Sequential world models increasingly act as learned simulators, providing action-conditioned rollouts for policy evaluation and training. WorldGym [quevedo2025worldgym] and WorldEval [li2025worldeval] generate action-conditioned rollouts and use VLM-based critics for evaluation, while WorldEval further leverages latent action representations to drive a DiT-based synthesizer. RLVR-World [wu2025rlvr] fine-tunes world models with RL with Verifiable Rewards (RLVR), using explicit metrics to close the pretraining–task objective gap. For safety risk prediction, Guan et al. [guan2025world] presented a framework for autonomous driving accident prediction that augments data with a domain-informed world model and enhances spatiotemporal reasoning using graph and temporal convolutions.

Beyond diffusion, sequence models broadened the capacity for long-range consistency. Po et al. [po2025long] integrated block-wise state space models for long-term memory with local attention for short-term coherence, enabling video generation with sustained memory and consistent dynamics. S2-SSM [petri2025learning] employs a Mamba layer to model the independent evolution of object slots and a sparsity-regularized cross-attention mechanism to capture their causal interactions, enabling causal reasoning over the environment.

Spatial Latent Grid. Pretraining geometry-aligned spatial maps with self-supervised spatiotemporal objectives, the spatial latent grid paradigm preserves locality and enables efficient rollouts, multimodal fusion, and transferable planner-ready maps.

Building on this paradigm, structured-grid and physics-informed methods encode geometry and dynamics for controllable rollouts. PhyDNet [guen2020disentangling] disentangles physical priors expressed as partial differential equations from visual factors, improving prediction. ViDAR [Yang2024visual] unifies semantics, geometry, and dynamics through a pre-training task of point cloud forecasting and a latent rendering operator, enabling a scalable foundation for downstream autonomous driving tasks. FOLIAGE [liu2025foliage] models dynamics with an Accretive Graph Network and a Transformer-based predictor, executing rollouts on simulated data. Complementing these grid and physics lines, MindJourney [yang2025mindjourney] couples VLMs with a controllable world model to render egocentric rollouts along planned camera trajectories, enabling multi-view reasoning.

Building on grid-based representations, diffusion-based forecasting has become dominant for stable long-horizon generation. Within grid-centric predictors, DOME [gu2024dome] encodes observations into a continuous latent space and applies a spatiotemporal DiT for scene forecasting, Copilot4D [zhang2024copilot4d] tokenizes point clouds and couples a spatiotemporal Transformer with discrete diffusion to improve fidelity and coherence, and LidarDM [zyrianov2025lidardm] generates layout-conditioned static scenes, composes them with dynamic objects, integrating LiDAR simulation to produce controllable videos. For long-video generation, Vista [gao2024vista] adopts a two-stage large-scale training regime to produce controllable, high-fidelity, driving videos, and Delphi [ma2024unleashing] enforces long-horizon multiview consistency through shared noise and feature alignment and a failure-driven framework to synthesize targeted data for improved planning. To strengthen long-horizon stability, GEM [hassan2025gem] achieves controllable ego-vision generation through large-scale training and fine-grained control over motion, dynamics, and posture, Zhou et al. [zhou2025learning] maintains a persistent RGB-D 3D memory map to guide subsequent frames, and STAGE [wang2025stage] introduced hierarchical temporal feature transfer with multi-stage training.

Decomposed Rendering Representation. Scenes are decomposed into explicit primitives to synthesize view-consistent, simulatable trajectories over long horizons. Within this paradigm, GaussianWorld [zuo2025gaussianworld] models scene evolution as ego-motion, object dynamics, and newly observed regions, iteratively updating 3D Gaussian primitives to enable accurate and efficient dynamic perception. InfiniCube [lu2025infinicube] introduced a hybrid pipeline that combines voxel-based generation, video synthesis, and dynamic Gaussian reconstruction, enabling large-scale dynamic 3D driving scenes conditioned on HDmaps, bounding boxes, and text. Complementarily, Wu et al. [wu2025video] augmented a video world model with long-term spatial memory grounded in reconstructed geometry and an episodic memory, which together condition sequential generation for long-range consistency.

#### III-B2 Global Difference Prediction

Token Feature Sequence. For general-purpose world models, tokenized feature sequences support global prediction via masked and generative modeling, enabling parallel long-horizon rollouts with global constraints and multimodal conditioning.

Within Joint-Embedding Predictive Architecture (JEPA) [LeCun2022Path], V-JEPA [bardes2024revisiting] extends this architecture to video by predicting latent features of occluded spatiotemporal regions, learning generalizable representations for both appearance and motion without pixel reconstruction or contrastive learning. Building on this, V-JEPA 2 [assran2025v] scales pretraining to large-scale Internet videos with larger models and incorporates limited robot interaction data for post-training, transferring to robotic planning. AD-L-JEPA [zhu2025ad] adapts JEPA to BEV LiDAR, predicting masked embeddings in a self-supervised manner. Beyond JEPA-style prediction, WorldDreamer [wang2024worlddreamer] frames world modeling as masked visual sequence prediction to learn physics and motion for diverse video generation and editing, while MaskGWM [ni2025maskgwm] combines diffusion with masked feature reconstruction and a dual-branch masking strategy to improve long-horizon consistency and generalization.

In parallel, diffusion-based methods have become central to global-difference modeling. Sora [brooks2024video] represents video as unified spacetime patches and uses a DiT to generate long, coherent sequences at scale. ForeDiff [zhang2025consistent] decouples conditioning from denoising by adding a deterministic predictive stream and employing a pretrained predictor to guide generation, improving accuracy and consistency. For domain-specific synthesis, AirScape [zhao2025airscape] introduces an aerial video–intention dataset, applies supervised fine-tuning for controllability, and leverages VLMs to impose spatiotemporal constraints; MarsGen [li2025martian] builds a multimodal Mars dataset from NASA’s sparse rover stereo imagery using geometric foundation models, then trains a controllable generator to produce visually realistic, geometry-consistent Martian videos. In clinical guidance, EchoWorld [yue2025echoworld] proposes a motion-aware world model for echocardiography probe control, pretraining on region- and motion-outcome prediction and fine-tuning attention to fuse visual and motion cues for precise guidance.

Spatial Latent Grid. Spatial-grid models forecast voxel grids in parallel and fuse multi-view visual features into a unified map, learning a general-purpose world model.

Recent work converges on unified scene understanding and future prediction. UniFuture [liang2025seeing] couples Dual Latent Sharing with multi-scale latent interaction to jointly model appearance and depth in future driving scenes, and HERMES [zhou2025hermes] integrates multiview BEV features into an LLM with world queries that link scene understanding to future prediction within a single framework. BEVWorld [zhang2024bevworld] maps images and LiDAR into a compact BEV latent space through a unified tokenizer and applies a latent BEV diffusion model for synchronized multimodal forecasting. Progress in grid and occupancy forecasting includes differentiable raycasting with a proxy reformulation for sensor agnostic motion learning by Khurana et al. [khurana2022differentiable, khurana2023point] and LiDAR to range images 3D spatiotemporal convolutions by Mersch et al. [mersch2022self]. Cam4DOcc [ma2024cam4docc] established the first vision-only benchmark with an E2E 3D CNN baseline, and Liu et al. [liu2025towards] enhanced cross-task transfer through high-ratio compression and latent flow matching.

On the generative front, tokenized 4D representations enable controllable scene synthesis. OccSora [wang2024occsora] uses a 4D tokenizer to derive compact representations for trajectory conditioned diffusion, and DynamicCity [bian2025dynamiccity] encodes 4D occupancy as HexPlanes representations with a VAE and employs a conditional DiT for high fidelity controllable dynamics. Fidelity and consistency improve through decoupling ego-motion with scene evolution in COME [shi2025come], physics-informed constraints in DrivePhysica [yang2024physical], cross-view point map alignment in Liu et al. [liu2025geometry], and photometric warping-based supervision in PosePilot [jin2025posepilot]. For controllable conditioning, DriveDreamer 2 [zhao2025drivedreamer] translates prompts into agent trajectories and HDMaps for customizable video generation, EOT-WM [zhu2025other] encodes ego and surrounding trajectories as trajectory videos for trajectory consistent synthesis, and ORV [yang2025orv] uses 4D semantic occupancy sequences to guide action conditioned video with S2R transfer. AETHER [team2025aether] unifies dynamic 4D reconstruction, action-conditioned video prediction, and vision-based planning under training on synthetic 4D data and achieves zero-shot generalization to real-world scenarios.

Decomposed Rendering Representation. This paradigm performs global prediction by combining explicit 3D structure with video generative priors.

A trend is combining video generation with Gaussian Splatting. DriveDreamer4D [zhao2025drivedreamer4d] exploits complex driving trajectories such as lane changes to guide video synthesis and optimize a 4DGS model, which enhances reconstruction fidelity and spatiotemporal coherence from novel viewpoints. ReconDreamer [ni2025recondreamer] introduces an online restoration module together with progressive data reuse to correct artifacts in Gaussian-rendered views and enables reliable reconstruction of large-scale trajectories. MagicDrive3D [gao2024magicdrive3d] generates multi-view street scenes conditioned on BEV maps, 3D boxes, and text, and further converts the outputs into full 3D environments through fault-tolerant GS. In contrast, implicit-field methods replace GS with continuous neural representations. UnO [agro2024uno] leverages future point clouds to learn a NeRF-style 4D occupancy field, which allows annotation-free prediction and achieves strong transfer performance beyond supervised baselines in point-cloud forecasting.

## IV Data Resources & Metrics

World models in embodied AI are required to address diverse tasks spanning manipulation, navigation, and autonomous driving, requiring heterogeneous resources and rigorous evaluation. Accordingly, we present Data Resources in §[IV-A](#S4.SS1) and Metrics in §[IV-B](#S4.SS2), focusing on the most widely adopted platforms and evaluation measures as a unified foundation for cross-domain assessment.

### IV-A Data Resources

To meet the diverse demands of embodied AI, we categorize data resources into four categories: Simulation Platforms, Interactive Benchmarks, Offline Datasets, and Real-world Robot Platforms, as detailed in the following subsections. Tab. [III](#S4.T3) provides a comprehensive overview of these resources.

*TABLE III: An overview of data resources for training and evaluating embodied world models.*

#### IV-A1 Simulation Platforms

Simulation platforms provide controllable and scalable virtual environments for training and evaluating world models.

-
•

MuJoCo [todorov2012mujoco] is a customizable physics engine widely adopted for its efficient robotic simulation of articulated systems and contact dynamics in robotics and control research.

-
•

NVIDIA Isaac is an E2E, GPU-accelerated simulation stack that comprises Isaac Sim, Isaac Gym [makoviychuk2021isaac], and Isaac Lab [mittal2023orbit]. It offers photorealistic rendering and large-scale RL capabilities.

-
•

CARLA [dosovitskiy2017carla] is an open-source simulator based on Unreal Engine for urban autonomous driving, providing realistic rendering, diverse sensors, and closed-loop evaluation protocols.

-
•

Habitat [savva2019habitat] is a high-performance simulator for embodied AI, specializing in photorealistic 3D indoor navigation.

#### IV-A2 Interactive Benchmarks

Interactive benchmarks offer standardized task suites and protocols for reproducible closed-loop evaluation of world models.

-
•

DeepMind Control (DMC) [tassa2018deepmind] is a standard MuJoCo-based suite for control tasks, offering a consistent basis for comparing agents that learn from state or pixel-based observations.

-
•

Atari [bellemare2013arcade] is a suite of pixel-based, discrete-action games for evaluating agent performance. The Atari100k [kaiser2020model] specifically assesses sample efficiency by limiting interaction to 100k steps.

-
•

Meta-World [yu2020meta] is a benchmark for multi-task and meta-RL, featuring $50$ diverse robotic manipulation tasks with a Sawyer arm in MuJoCo under standardized evaluation protocols.

-
•

RLBench [james2020rlbench] offers $100$ simulated tabletop manipulation tasks with sparse rewards and rich, multi-modal observations, designed to test complex skills and rapid adaptation.

-
•

LIBERO [liu2023libero] is a benchmark for lifelong robotic manipulation, providing $130$ procedurally generated tasks and human demonstrations to evaluate sample-efficient and continual learning.

-
•

nuPlan [caesar2021nuplan] is a planning benchmark for autonomous driving, using a lightweight closed-loop simulator and over $1\,500\text{\,}\mathrm{h}\text{/}$ of real-world driving logs to evaluate long-horizon performance.

#### IV-A3 Offline Datasets

Offline datasets are large-scale, pre-collected trajectories that eliminate interactive rollouts and provide a foundation for reproducible evaluation and data-efficient pretraining of world models.

-
•

RT-1 [brohan2022rt] is a real-world dataset for robot learning, collected over $17$ months with $13$ Everyday Robots mobile manipulators. It contains $130\,000$ demonstrations spanning more than $700$ tasks, pairing language instructions and image observations with discretized 11-DoF actions for the arm and mobile base.

-
•

Open X-Embodiment (OXE) [o2024open] is a corpus aggregating $60$ sources from $21$ institutions, spanning $22$ robot embodiments, $527$ skills, and over one million trajectories in a unified format for cross-embodiment training. Models trained on OXE demonstrate strong transfer beyond single-robot baselines, underscoring the effectiveness of cross-platform data sharing.

-
•

Habitat-Matterport 3D (HM3D) [ramakrishnan2021habitatmatterport] is a large-scale dataset of $1\,000$ indoor reconstructions with $112\,500\text{\,}{\mathrm{m}}^{2}\text{/}$ navigable area, substantially expanding the scope and diversity of embodied AI simulation. Released for the Habitat platform, it provides the necessary metadata and resources for seamless use.

-
•

nuScenes [caesar2020nuscenes] is a large-scale multimodal driving dataset with a 360-degree sensor suite comprising six cameras, five radars, one LiDAR, and GPS/IMU. It contains $1\,000$ twenty-second scenes collected in Boston and Singapore with dense 3D annotations for $23$ classes and HDMaps, providing a core benchmark for multimodal fusion and long-horizon prediction.

-
•

Waymo [sun2020scalability] is a multimodal autonomous driving benchmark with $1\,150$ twenty-second scenes at $10\text{\,}\mathrm{Hz}\text{/}$ from San Francisco, Phoenix, and Mountain View. It includes five LiDARs and five cameras, with about $12$ million 3D and 2D annotations, making it a large-scale resource for modeling traffic dynamics.

-
•

Occ3D [tian2023occ3d] defines 3D occupancy prediction from surround-view images, providing voxel labels that distinguish free, occupied, and unobserved states. Occ3D-nuScenes contains about $40\,000$ frames at $0.4\text{\,}\mathrm{m}\text{/}$ resolution, while Occ3D-Waymo offers about $200\,000$ frames at $0.05\text{\,}\mathrm{m}\text{/}$. This voxel-level supervision enables holistic scene understanding beyond bounding boxes.

-
•

Something-Something v2 (SSv2) [goyal2017something] is a video dataset for fine-grained action understanding. It contains $220\,847$ clips across $174$ categories, collected from crowd workers following textual prompts (*e.g.*, Putting something into something) with splits of $168\,913$ train, $24\,777$ val, and $27\,157$ test videos.

-
•

OpenDV [yang2024generalized] is the largest large-scale video-text dataset for autonomous driving, proposed by GenAD, which supports video prediction and world-model pretraining. It contains $2\,059$ hours and $65.1$ million frames from YouTube and seven public datasets, covering over $40$ countries and $244$ cities. The dataset provides command and context annotations to enable language- and action-conditioned prediction and planning.

-
•

VideoMix22M [assran2025v] is a large-scale dataset introduced with V-JEPA 2 for self-supervised pretraining. It scales from $2$ million to $22$ million samples, drawn from YT-Temporal-1B [zellers2022merlot], HowTo100M [miech2019howto100m], Kinetics [carreira2019short], SSv2, and ImageNet [deng2009imagenet]. The largest source, YT-Temporal-1B, is curated with retrieval-based filtering to suppress noise, while ImageNet images are converted into static video clips for consistency.

#### IV-A4 Real-world Robot Platforms

Real-world robot platforms provide physical embodiments for interaction, enabling closed-loop evaluation, high-fidelity data collection, and S2R validation under real-world constraints.

-
•

Franka Emika [haddadin2022franka] is a 7-DoF collaborative robot arm with joint torque sensors for precise force control. Through the control interface, it supports $1\text{\,}\mathrm{kHz}\text{/}$ torque control for contact-rich tasks, while its ROS integration makes it a versatile platform.

-
•

Unitree Go1 [unitree_go1_web] is a cost-effective and widely adopted quadrupedal robot equipped with a panoramic depth-sensing suite, onboard computing of 1.5 TFLOPS, and a maximum speed of $4.7\text{\,}\mathrm{m}\text{/}\mathrm{s}$, establishing it as a standard platform for locomotion and embodied-AI research.

-
•

Unitree G1 [unitree_g1_web] is a compact humanoid robot for research, offering up to $43$-DoF and knee torques of $120$ N$\cdot$m, with integrated 3D LiDAR and depth cameras. With multimodal sensing, onboard compute, ROS support, and swappable batteries, this low-cost platform provides a practical real-robot testbed for training and evaluating embodied world models.

### IV-B Metrics

Metrics evaluate the capability of world models to capture dynamics, generalize to unseen scenarios, and scale with additional resources. We organize them into three abstraction levels: §[IV-B1](#S4.SS2.SSS1) Pixel Prediction Quality, §[IV-B2](#S4.SS2.SSS2) State-level Understanding, and §[IV-B3](#S4.SS2.SSS3) Task Performance, representing a progression from low-level signal fidelity to high-level goal attainment.

#### IV-B1 Pixel Generation Quality

At the most fundamental level, world models are evaluated by their ability to reconstruct sensory inputs and generate realistic sequences. Metrics assess image fidelity, temporal consistency, and perceptual similarity, providing quantitative measures of the extent to which models capture raw environmental dynamics.

Fréchet Inception Distance (FID) [heusel2017gans]. FID is a metric for assessing the realism and diversity of generated images. It compares real and generated image distributions in the feature space of an ImageNet-pretrained Inception-v3 [szegedy2016rethinking], modeling embeddings as Gaussians with means $\bm{\mu}_{x},\bm{\mu}_{y}$ and covariances $\bm{\Sigma}_{x},\bm{\Sigma}_{y}$. Defined as

$$\operatorname{FID}(x,y)=\lVert\bm{\mu}_{x}-\bm{\mu}_{y}\rVert_{2}^{2}+\operatorname{Tr}\left(\bm{\Sigma}_{x}+\bm{\Sigma}_{y}-2(\bm{\Sigma}_{x}\bm{\Sigma}_{y})^{1/2}\right),$$ \tag{6}

a lower FID denotes a closer alignment between real and generated distributions. By comparing the first and second moments, it penalizes fidelity loss (mean shift) and mode collapse (covariance mismatch), offering a holistic measure of generative performance.

Fréchet Video Distance (FVD) [unterthiner2018towards]. FVD extends FID to videos, evaluating both per-frame quality and temporal consistency. It replaces the image-based Inception network with an I3D [carreira2017quo] pretrained on Kinetics-400 [kay2017kinetics]. Using the same Fréchet distance as Eq. ([6](#S4.E6)) on motion-aware features, FVD yields a holistic video quality score. A lower value indicates a closer alignment of distributions in appearance and dynamics while penalizing temporal artifacts like unnatural motion or flickering.

Structural Similarity Index Measure (SSIM) [wang2004image]. SSIM is a perceptual metric for image quality that compares luminance, contrast, and structure between a generated image and its reference. For two patches $x$ and $y$ with means $\bm{\mu}_{x}$, $\bm{\mu}_{y}$, variances $\bm{\Sigma}_{x}^{2},\bm{\Sigma}_{y}^{2}$, and covariance $\bm{\Sigma}_{xy}$, SSIM is defined as

$$\operatorname{SSIM}(x,y)=\frac{(2\bm{\mu}_{x}\bm{\mu}_{y}+C_{1})(2\bm{\Sigma}_{xy}+C_{2})}{(\bm{\mu}_{x}^{2}+\bm{\mu}_{y}^{2}+C_{1})(\bm{\Sigma}_{x}^{2}+\bm{\Sigma}_{y}^{2}+C_{2})}.$$ \tag{7}

The final score is obtained by averaging SSIM over sliding windows, and values closer to $1$ indicate higher similarity.

Peak Signal-to-Noise Ratio (PSNR) [hore2010image]. PSNR measures pixel-wise distortion between a reconstruction and its reference. Let the mean-squared error (MSE) over $N$ pixels be

$$\operatorname{MSE}=\frac{1}{N}\sum_{i=1}^{N}\left(x_{i}-y_{i}\right)^{2},$$ \tag{8}

and let $\operatorname{MAX}$ denote the maximum possible pixel value(*e.g.*, $255$ for RGB or $1$ for normalized images). Then

$$\operatorname{PSNR}(x,y)=10\cdot\log_{10}\left(\frac{\mathrm{MAX}^{2}}{\mathrm{MSE}}\right).$$ \tag{9}

Higher PSNR values indicate lower distortion and greater fidelity.

Learned Perceptual Image Patch Similarity (LPIPS) [zhang2018unreasonable]. LPIPS is a metric that correlates with human judgments by comparing features extracted from pretrained networks. Let $\hat{f}^{l}_{x}$ and $\hat{f}^{l}_{y}$ denote the unit-normalized activations at layer $l$ for inputs $x$ and $y$, and $w_{l}$ the channel-wise weights. LPIPS is defined as

$$\operatorname{LPIPS}(x,y)=\sum_{l}\frac{1}{H_{l}W_{l}}\sum_{h,w}\left\|w_{l}\odot\big(\hat{f}_{h,w,x}^{l}-\hat{f}_{h,w,y}^{l}\big)\right\|_{2}^{2}.$$ \tag{10}

Lower LPIPS values imply greater similarity, offering enhanced fidelity compared to pixel-based metrics and robustness against distortions.

VBench [huang2024vbench]. VBench is a comprehensive benchmark for video generation that assesses performance across 16 dimensions grouped into two categories: Video Quality (*e.g.*, subject consistency, motion smoothness) and Video-Condition Consistency (*e.g.*, object class, human action). It provides carefully curated prompt suites and large-scale human preference annotations to ensure strong perceptual alignment, thereby enabling fine-grained evaluation of model capabilities and limitations.

#### IV-B2 State-level Understanding

Beyond pixel fidelity, state-level understanding assesses whether models capture objects, layouts, and semantics, and can predict their evolution. Metrics span semantic, BEV, and 3D segmentation, detection, occupancy, geometry, and trajectory accuracy, emphasizing structural understanding beyond appearance.

mean Intersection over Union (mIoU). mIoU evaluates semantic segmentation by averaging the Intersection over Union (IoU) across classes. For class $c$,

$$\operatorname{IoU}=\frac{\operatorname{TP}}{\operatorname{TP}+\operatorname{FP}+\operatorname{FN}},$$ \tag{11}

where TP, FP, and FN denote the true positives, false positives, and false negatives. IoU quantifies overlap with the ground truth while penalizing segmentation errors. The dataset-level score is

$$\operatorname{mIoU}=\frac{1}{\left|C\right|}\sum_{c\in C}\operatorname{IoU}_{c}.$$ \tag{12}

A higher mIoU reflects more precise semantic scene understanding.

mean Average Precision (mAP). mAP evaluates detection and instance segmentation by averaging per-class Average Precision (AP). For a class $c$ at IoU threshold $\tau$, predictions are ranked by confidence and matched one-to-one with ground truths when $\mathrm{IoU}\geq\tau$ with unmatched predictions counted as FP and unmatched ground truths as FN. Precision and recall are

$$\operatorname{Precision}=\frac{\operatorname{TP}}{\operatorname{TP+FP}},\quad\operatorname{Recall}=\frac{\operatorname{TP}}{\operatorname{TP}+\operatorname{FN}}.$$ \tag{13}

Let $P_{c,\tau}(r)$ denote the precision-recall envelope obtained via monotonic interpolation. The AP for class $c$ at threshold $\tau$ is

$$\operatorname{AP}_{c,\tau}=\int_{0}^{1}P_{c,\tau}(r)\mathrm{d}r.$$ \tag{14}

mAP averages AP across classes and thresholds $T$:

$$\operatorname{mAP}=\frac{1}{\left|C\right|}\sum_{c\in C}\left(\frac{1}{\left|T\right|}\sum_{\tau\in T}\operatorname{AP}_{c,\tau}\right).$$ \tag{15}

A higher mAP indicates better instance recognition, more accurate localization, and more calibrated confidence estimates.

Displacement Error. Displacement error metrics assess state-level understanding by measuring spatial accuracy for keypoints, object centers, and trajectory waypoints. The L2 trajectory error computes the Euclidean distance between predicted and ground-truth waypoints. Common variants include Average Displacement Error (ADE), which calculates the average displacement, and Final Displacement Error (FDE), which measures the displacement at the final step. Lower values indicate more accurate localization.

Chamfer Distance (CD) [fan2017point]. CD quantifies geometric similarity between a prediction $S_{1}$ and ground truth $S_{2}$ by summing squared nearest-neighbor distances across the two sets:

$$\operatorname{CD}(S_{1},S_{2})=\sum_{x\in S_{1}}\min_{y\in S_{2}}\left\|x-y\right\|_{2}^{2}+\sum_{y\in S_{2}}\min_{x\in S_{1}}\left\|x-y\right\|_{2}^{2}.$$ \tag{16}

Unlike pixel-level metrics, CD captures surfaces, occupancy, BEV, and 3D structures, and its differentiability enables use as both a training loss and an evaluation metric that complements IoU.

#### IV-B3 Task Performance

Ultimately, the value of a world model lies in supporting effective decision-making, with task-level metrics assessing goal achievement under safety and efficiency constraints in embodied settings.

Success Rate (SR). SR quantifies performance as the fraction of evaluation episodes that satisfy a predefined success condition. In navigation and manipulation, the condition is typically binary, such as reaching a target or placing an object correctly. In autonomous driving, the requirement is stricter, demanding route completion without collisions or major violations. The final SR is reported as the average of binary outcomes across all test episodes.

Sample Efficiency (SE). SE quantifies the samples needed to reach target performance. It is evaluated by fixed-budget benchmarks (*e.g.*, Atari-100k), data–performance curves, or in robotics by the demonstrations required to achieve a given success rate.

Reward. In RL, the reward is a signal $r_{t}$ at timestep $t$. The goal is to maximize the discounted return $G_{t}=\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+1}$. Results are reported as cumulative reward or average return, often with normalization for cross-task comparison.

Collision. Safety is evaluated with collision-based metrics. The primary measure, collision rate, is the proportion of evaluation episodes with at least one collision and is common in indoor navigation. In autonomous driving, exposure-normalized variants are used, such as collisions per kilometer or collisions per hour.

## V Performance Comparison

Given the proliferation of world-model variants and heterogeneous metrics, we organize comparisons by task objectives and rely on standardized benchmarks, reporting concise tables that highlight each method’s strengths and limitations.

### V-A Pixel Generation

Generation on nuScenes. Driving video generation is treated as a world-modeling task that synthesizes plausible scene dynamics in fixed-length clips. Typical protocols produce short sequences and evaluate quality with *FID* for appearance fidelity and *FVD* for temporal consistency. For a fair comparison on the nuScenes validation split, recent approaches have achieved remarkable progress, as shown in Tab. [IV](#S5.T4). DrivePhysica delivers the best visual fidelity, while MiLA achieves the strongest temporal coherence, together establishing new state-of-the-art performance.

*TABLE IV: Performance comparison of video generation on the nuScenes.*

### V-B Scene Understanding

4D Occupancy Forecasting on Occ3D-nuScenes. 4D occupancy forecasting is treated as a representative world modeling task. Given $2\text{\,}\mathrm{s}\text{/}$ of past 3D occupancy, models predict the subsequent $3\text{\,}\mathrm{s}\text{/}$ of scene dynamics. Evaluation follows the Occ3D-nuScenes protocol and reports *mIoU* and per horizon *IoU*. As summarized in Tab. [V](#S5.T5), we compare methods by input modality, auxiliary supervision, and ego trajectory usage to reveal design choices for spatiotemporal forecasting. Methods using occupancy inputs outperform camera-only variants, and adding auxiliary supervision with a GT ego trajectory further mitigates performance decay at 2–3 s. Among all methods, COME (with GT ego) achieves the best average mIoU and per-horizon IoU.

*TABLE V: Performance comparison of 4D occupancy forecasting on the Occ3D-nuScenes benchmark1.*

### V-C Control Tasks

Evaluation on DMC. Most studies probe the capacity of world models to learn control-relevant dynamics, typically adopting a pixel-based setting with $64{\times}64{\times}3$ observations. The primary metric is *Episode Return*, defined as the cumulative reward over $1\,000$ steps, with a theoretical maximum of $1\,000$ given $r_{t}\in[0,1]$. For comparability, Tab. [VI](#S5.T6) reports the step budget and summarizes performance by task score and task count. The results indicate improved data efficiency, with recent models reaching strong performance in far fewer training steps. However, inconsistent evaluation protocols and task subsets impede a fair assessment of generalization, and building a broadly transferable model across tasks, modalities, and datasets remains an open challenge.

*TABLE VI: Performance comparison on the DMC benchmark.1.*

Evaluation on RLBench. RLBench evaluates manipulation with a 7-DoF simulated Franka arm and is widely used to assess whether world models capture task-relevant dynamics and support conditioned action generation. The primary metric is *Success Rate*, defined as the fraction of episodes that reach the goal within the step limit. As summarized in Tab. [VII](#S5.T7), implementations differ in episode budgets, resolution, and modalities, which complicates like-for-like comparison. Despite this heterogeneity, several trends are evident. Recent methods increasingly leverage multimodal inputs and adopt stronger backbones such as 3DGS and DiT. VidMan achieves a high average success rate on the broadest task, revealing IDM as a promising architectural direction.

*TABLE VII: Performance comparison for manipulation tasks on RLBench.*

Planning on nuScenes. Open-loop planning is treated as a world modeling task on the nuScenes validation split, where models predict ego motion from a limited history. Methods observe $2\text{\,}\mathrm{s}\text{/}$ of past trajectories and forecast the next $3\text{\,}\mathrm{s}\text{/}$ as 2D BEV waypoints. Evaluation reports *L2* error and *collision rate* at multiple horizons, and Tab. [VIII](#S5.T8) summarizes results by input modality, auxiliary supervision, and metric settings. Under this shared protocol, a clear tradeoff emerges. UniAD+DriveWorld achieves the lowest *L2* with extensive auxiliary supervision, whereas SSR attains the best collision rate with competitive *L2* without extra supervision. Camera-based methods now surpass models that use privileged occupancy, reflecting the growing maturity of E2E planning.

*TABLE VIII: Performance comparison for open-loop planning on the nuScenes validation split1.*

## VI Challenges and Trends

This section reviews the open challenges and emerging directions for world models in embodied AI. We discuss them across three dimensions: §[VI-A](#S6.SS1) Data & Evaluation, §[VI-B](#S6.SS2) Computational Efficiency, and §[VI-C](#S6.SS3) Modeling Strategies.

### VI-A Data & Evaluation

Challenges. From a data perspective, the central challenge lies in the scarcity and heterogeneity of existing corpora. Although embodied AI spans diverse domains such as navigation, manipulation, and autonomous driving, a unified large-scale dataset remains lacking. This fragmentation constrains the capacity of world models and substantially hinders their ability to generalize.

Evaluation practices face similar limitations. Metrics such as FID and FVD emphasize pixel fidelity while ignoring physical consistency, dynamics, and causality. Recent benchmarks, such as EWM-Bench [yue2025ewmbench], introduce new measures but remain task-specific and lack cross-domain standards.

Future Directions.
Recent initiatives such as OpenDV-2K [yang2024generalized] and VideoMix22M [assran2025v] highlight the growing focus on large-scale pretraining and broader modality coverage, yet resources remain fragmented and domain specific. Future work should prioritize constructing unified multimodal, cross-domain datasets to enable transferable pretraining, while advancing evaluation frameworks that move beyond perceptual realism to assess physical consistency, causal reasoning, and long-horizon dynamics.

### VI-B Computational Efficiency

Challenges. Embodied AI tasks encounter significant challenges in computational efficiency, particularly in real-time applications. Although models such as Transformers and Diffusion networks exhibit strong performance, their high inference costs conflict with the real-time control demands of robotic systems. Consequently, traditional methods like RNNs and global latent vectors remain widely employed, as they offer greater computational efficiency, despite limitations in capturing long-term dependencies.

Future Directions. To address this challenge, future research should focus on optimizing model architectures using techniques like quantization, pruning, and sparse computation to reduce inference latency without compromising performance. Additionally, exploring novel temporal methods, such as SSMs, could enhance long-range reasoning while maintaining real-time efficiency, offering a promising solution for robotic systems.

### VI-C Modeling Strategy

Challenges. Despite rapid progress, world models still struggle with long-horizon temporal dynamics and efficient spatial representations. The main difficulty lies in balancing recurrent simulation and global prediction: autoregressive designs are compact and sample-efficient but accumulate errors over time, whereas global prediction improves multi-step coherence at the cost of heavy computation and weaker closed-loop interactivity. On the spatial side, efficiency remains a bottleneck. Latent vectors, token sequences, and spatial grids each present trade-offs between efficiency and expressiveness, while decomposed rendering approaches (*e.g.*., NeRF and 3DGS) offer high fidelity yet scale poorly in dynamic scenes. Taken together, temporal and spatial modeling are still constrained by structural trade-offs that limit scalability and adaptability.

Future Directions. Several promising avenues have emerged to address current bottlenecks. SSMs (*e.g.*, Mamba), aligned with autoregressive modeling, offer linear-time scalability and strong potential for long-horizon reasoning. In contrast, masked approaches (*e.g.*, JEPA), closer to global prediction, improve representation learning and efficiency, though their integration into closed-loop control remains challenging. Furthermore, a promising approach lies in integrating the strengths of both autoregressive and global prediction methods. Explicit memory or hierarchical planning can enhance long-horizon prediction stability, while task decomposition inspired by CoT can improve temporal consistency through intermediate goal setting. Future frameworks should prioritize optimizing long-range reasoning, computational efficiency, and generative fidelity, while seamlessly integrating temporal and spatial modeling into unified architectures that strike an effective balance between efficiency, fidelity, and interactivity.

## VII Conclusion

This survey organizes world models in embodied AI using a novel three-part framework: functionality, temporal modeling, and spatial representation. Based on this, we review existing research, datasets, and metrics to establish a standard for comparison. However, significant challenges remain, including a lack of unified datasets and evaluation methods that overlook physical causality. A core modeling challenge involves reconciling the trade-off between efficient autoregressive approaches and robust global prediction paradigms. Future work should address these gaps by creating unified, physically-grounded benchmarks and exploring efficient architectures. Developing hybrid methods that balance fidelity, efficiency, and interactivity is key, as world models form the foundation for the next generation of embodied AI by unifying perception, prediction, and decision-making.

Generated on Sat Nov 29 05:29:15 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)