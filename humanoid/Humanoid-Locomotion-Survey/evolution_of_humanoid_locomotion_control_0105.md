Evolution of Humanoid Locomotion Control 

Yan Gu1,*⋆*, Guanya Shi2,*⋆*, Fan Shi3,*⋆*, I-Chia Chang1, Yen-Jen Wang4,   
Qilong Cheng5, Zachary Olkin6, Ivan Lopez-Sanchez5, Yunchu Feng5,   
Jian Zhang7, Aaron D. Ames6, Hao Su5,†, and Koushil Sreenath4,† 

1School of Mechanical Engineering, Purdue University, West Lafayette, IN 47907, USA. 2Robotics Institute, Carnegie Mellon University, Pittsburgh, PA 15213, USA. 

3Department of Electrical and Computer Engineering, National University of Singapore, Singapore. 4Department of Mechanical Engineering, University of California, Berkeley, Berkeley, CA 94720 USA. 5Tandon School of Engineering, New York University, New York, NY 11201 USA. 

6Department of Mechanical and Civil Engineering, California Institute of Technology, 7Meta Platforms Inc., Menlo Park, CA 94025, USA. 

†Corresponding authors. Email: hao.su@nyu.edu, koushils@berkeley.edu *⋆*These authors contributed equally to this work. 

Email: yangu@purdue.edu, guanyas@andrew.cmu.edu, fan.shi@nus.edu.sg 

Date: December 3, 2025 

Website: https://github.com/purdue-tracelab/Humanoid-Locomotion-Survey 

Abstract 

Humanoid robots stand at the forefront of robotics, aiming to capture the agility, robustness, and expressivity of human movement within an anthropomorphic form. The locomotion control of humanoids has evolved from classical model-based methods to reinforcement learning powered by large-scale simulation, and now to generative models that produce adaptive, whole-body behaviors, propelling humanoids toward operation in real-world environments. This survey positions humanoid control at a turning point, converging toward a unified paradigm of physics-guided generative intelli gence that integrates optimization, learning, and predictive reasoning. We identify three core princi ples linking these paradigms: physics-based modeling, constrained decision making, and adaptation to uncertainty. Building on these connections, we provide recommendations for researchers and outline open challenges in safety, accessibility, and human-level capability. These directions rep resent a transformation from engineered stability to intelligent autonomy, laying the groundwork for humanoid generalists capable of operating safely, collaborating naturally, and extending human capability in the open world. 

1 Introduction 

Humanoid robots represent one of the most ambitious frontiers in robotics, integrating human-like agility, robustness, and expressivity within an anthropomorphic body. Equipped with legs and arms, they can climb stairs, traverse cluttered environments, and collaborate safely with people. Unlike wheeled machines, humanoids can step over obstacles, adapt to discontinuous or moving terrain, and operate in spaces designed for humans. Their human-like morphology narrows the embodiment gap and en ables direct use of human motion data and control strategies, making humanoids a natural platform for connecting high-level reasoning with whole-body motor execution in existing human environments. 

The long-term vision is to realize humanoids that coexist safely and productively with people, robots that are physically capable, perceptually aware, and socially intelligible. Achieving this vision, however, 

1

Classical 

1960- 

Simplified and complex  dynamics models 

Reduced-order modeling Control theory 

Convex optimization 

Control 

Optimization 

Neuro/biomechanics   
Learning-based 2015- 

The controller considers 

Physics-based simulation Specific limited real-world data 

Tools 

GPU-parallel simulation 

Reinforcement learning 

Imitation learning 

Drawing inspiration from 

Machine learning 

Computer graphics 

Tasks 

Emerging 

2023- 

Rich real-world dynamics  Internet-scale data 

Foundation models 

World models 

Diffusion models 

Natural language processing Computer vision 

Stable locomotion on restrictive terrains Agile locomotion on various terrains Multi-task loco-manipulation in  the open world 

Figure 1: Evolution of humanoid locomotion control paradigms over the past several decades. Humanoid control has progressed from classical methods, including real-time feedback and predictive control based on simplified dynamics, control theory, and convex optimization, to learning-based ap proaches that leverage physics simulation, parallel computation, and reinforcement learning. It is now entering an emerging era driven by generative AI. Each paradigm expands the scope of reasoning from reduced-order models to simulated and real-world physics and increasingly integrates insights from mul tiple disciplines, including biomechanics, control theory, machine learning, computer graphics, natural language processing, and computer vision. This progression has enabled a transition from stable walk ing on constrained terrains to agile locomotion and, most recently, to multi-task loco-manipulation in unstructured environments. \[1, 2, 3\] 

remains a grand challenge. Coordinating dozens of joints and intermittent contacts under uncertain, non linear dynamics requires controllers that prioritize safety while balancing precision, energy efficiency, and adaptability in unpredictable environments. 

A central challenge to realizing this vision is control, the mechanism that makes hardware pro duce purposeful behavior. Control for bipedal humanoid locomotion has advanced rapidly over the 

2  
past several decades (Fig. 1). Early approaches relied on simplified physics-based dynamics models and handcrafted feedback laws to ensure stability. Advances in computation enabled nonlinear con troller synthesis through offline trajectory optimization using detailed dynamic models, while predictive control introduced constraint handling for online motion generation of robust whole-body behaviors. Recently, Reinforcement Learning (RL) has demonstrated highly agile and resilient locomotion through large-scale simulation, parallel training, and physics-guided reward design. Emerging generative meth ods now unify perception, high-level decision-making, and control to enable versatile, adaptive, and human-like motion. 

This survey examines the evolution of humanoid locomotion control paradigms from classical con trol to RL and generative models, driven by the need for greater robustness, adaptability, and gener alization in complex, real-world environments (Fig. 1, Tab. 1). It reveals that progress across these paradigms is unified by enduring physics-based principles that continue to guide the development of intelligent, physically adept humanoids. We also identify open challenges in safety, accessibility, and cognitive capability essential for deployment beyond the laboratory. Perception \[4, 5\], task planning \[6\], and state estimation \[7\] are discussed only when directly relevant to control design. 

Unlike prior surveys that focus on individual control paradigms or specific application domains \[8, 9, 10, 11, 12, 13, 14, 15, 16\], this paper bridges classical and learning-based approaches through their common physical principles and extends the discussion to recent advances in generative and foundation models. Rather than serving solely as a retrospective review, it offers a forward-looking perspective on how these paradigms are converging toward unified principles of humanoid control. Ha et al. \[16\] focus on RL for quadrupeds, Wensing et al. \[17\] review optimization-based control without addressing learn ing approaches, and Gu et al. \[18\] provide an overview of whole-body loco-manipulation emphasizing system integration rather than the conceptual evolution of control frameworks. 

Contributions and key takeaways. This paper frames the advancements of humanoid locomotion control from physics-based to learning-based methods with a view toward their fusion within emerging paradigms for next-generation humanoids. Historically, physics-based control provided stability, inter pretability, and guarantees, while learning-based methods have delivered agility and robustness. Our first contribution unifies these perspectives (Fig. 1) by identifying their respective strengths and limita tions: classical real-time feedback control is fast and principled but restricted to pre-planned behaviors; predictive control manages constraints online but often sacrifices agility; RL achieves agility and robust ness but remains task-specific and opaque; and generative methods offer versatility yet lack real-world reliability. 

The second contribution reveals the inherent connections among these paradigms through three shared enablers: physics-based modeling, constrained decision-making, and adaptation to uncertainty. Each method excels in a particular domain, and their integration promises new capabilities for humanoid systems. We posit that the field is converging toward controllers that unify classical approaches with learning, predictive reasoning, and adaptation, where physics increasingly guides learning through re ward design, system identification, and real-time interfaces. Lastly, we outline emerging directions in generative multi-modal loco-manipulation, test-time adaptation, and human-centered humanoid design. The future of humanoid locomotion control lies at this intersection, drawing from the rigor of the past, leveraging the developments of the present, and integrating the transformative methods now on the hori zon. 

2 Modeling and Classical Control for Humanoid Locomotion 

This section reviews how classical control for bipedal humanoid locomotion has advanced from real-time feedback to predictive methods. This shift has been powered by the demand for improving robustness, 

3  
agility, and explicit constraint handling in complex environments. 

Early real-time feedback controllers operated myopically, reacting directly to measured states with minimal computation. Since the 1980s, such controllers have enabled remarkable physical demonstra tions, including Raibert’s dynamic hopping and jumping bipeds \[19\], Honda’s quasi-static ASIMO \[20\], and later dynamic bipedal \[21, 22\] and humanoid \[23\] walking and running \[24\]. 

Predictive control extended these methods by reasoning about system dynamics and constraints over a finite horizon to plan motions proactively. Offline trajectory optimization, which generates desired full body motions, enabled major milestones such as the DARPA Robotics Challenge (2013–2015) \[25, 26, 27\] and Boston Dynamics’ ATLAS demonstrations of parkour behaviors \[28, 29\]. Subsequent advances in online predictive control for quadrupeds \[30\] inspired optimization-based humanoid controllers \[17\], further enhancing agility and robustness. 

Physics-based dynamics models underpin both classical and learning-based control, shaping achiev able performance and computational requirements. Classical control, grounded in these models, has enabled agile and robust behaviors but struggles with generalization, state estimation, and scalability to unstructured environments. Learning-based methods have begun to overcome these challenges, yet their success often relies more on physics-guided training design than on algorithmic novelty, as dis cussed in “Learning-based Control for Humanoid Locomotion.” Reward shaping and parameter ran domization embed the same physics-informed priors and constraint reasoning that characterize classical control \[31, 32\]. Ultimately, continued progress in both paradigms depends on accurate physics-based modeling, a central theme examined next. 

2.1 Physics-based Analytical Models 

Physics-based analytical models provide interpretable structure by describing how joint torques, contact forces, and environmental interactions generate motion, typically derived from first-principles dynam ics. A general formalization for a single-domain hybrid dynamical system with state *x* and control input *u* has dynamics: 

|  *x*˙ \= *f* (*t*, *x*, *u*), *x* ∈ *D* \\ *S  x*\+ \= *∆*(*x*−), *x*− ∈ *S*(1) *Σ* :=  |
| ----- |

Here *t* denotes time, and the continuous *f* and discrete *∆* dynamics can be derived using Lagrangian or Newton-Euler formulations \[36\] given the domain *D* and switching surface *S* (extensions exist for multi-domain systems \[49\]). This hybrid structure is essential for legged systems, which repeatedly make and break ground contact. Any useful model (Fig. 2), analytical, numerical, or learned, must be accurate enough to reflect physical reality and efficient enough to run at control rates, which is difficult for high-dimensional, nonlinear, hybrid, and contact-rich humanoid dynamics. 

Full-order models. Full-order models describe the complete rigid-body dynamics and environment interaction of the robot \[36\]. The choice of contact defines holonomic constraints that shape the hy brid system’s graph structure, making contact selection critical to modeling. While utilizing full-order models allows the controllers to achieve more dynamic motion, their high dimensionality increases com putational cost and control complexity. 

Reduced-order models. Reduced-order models compress behavior to a few task-relevant quantities such as center of mass (CoM) motion or centroidal momentum \[50\]. Examples include the linear in verted pendulum (LIP) \[33\], spring-loaded inverted pendulum (SLIP) \[19, 34\]), compass-gait biped \[51\], centroidal models \[52\], point mass model \[53\], and single rigid body models \[30, 35\]. These abstractions 

4  
Expressiveness   
Physics-based simulator 

Hybrid method 

Data-driven 

residual 

neural network 

World model   
���� ��{��+1:��+ℎ} ���� 

Latent dynamics   
����   
����+1 

Full-order model **Reduced-order model** 

\+ 

Physics-based model  and simulator   
����, ���� ����+1 

Learned dynamics ����, ���� ����+1 

Physics-based Data-driven 

Figure 2: Spectrum of modeling approaches for humanoid locomotion control, spanning from physics-based to data-driven methods and from low to high model expressiveness. Physics-based models include: reduced-order models \[33, 34, 35\], which approximate dynamics using simplifying assumptions; full-order models \[36\], which capture complete robot dynamics under specified contact conditions; and simulators \[37, 38\], which provide the highest fidelity through detailed modeling of contact, actuation, and sensing (reviewed in “Modeling and Classical Control for Humanoid Locomo tion”). Data-driven models comprise: learned dynamics \[39, 40\], which model state transitions di rectly from data; latent dynamics models \[41\], which operate in lower-dimensional space; and world models \[42, 43, 44\], which predict high-dimensional robot and environment behavior (see “Learning based Control for Humanoid Locomotion” and “Emerging Frontiers in Humanoid Robot Control”). Hy brid approaches integrate physics-based and learned components to exploit complementary strengths, such as modeling actuator dynamics \[45\], learning residual effects \[46, 47\], or improving simulator fi delity with real-world data \[48\]. Color shading indicates different modeling approaches: classical , 

learning-based , hybrid , and emerging . Symbols *ot*, *st*, *zt*, and *at*represent the observation, state, latent state, and action at time *t*, respectively, while *o*{*t*\+1:*t*\+*h*} denotes a prediction of observations from *t* \+ 1 to *t* \+ *h* where *h* is the time horizon. 

enable real-time balance control, foot placement, and disturbance rejection \[54, 55, 56, 57, 58, 59, 60\], while also revealing limb coordination principles in animal locomotion \[61\] and guided bio-inspired control strategies \[62\]. Nonetheless, their simplifying assumptions, such as flat terrain, rigid contact, or constant CoM height \[33\], limit performance in fast, contact-rich motions and complex environments. 

Physics-based simulators. Physics-based models (typically full-order) remain the invisible engine be hind modern learning methods: RL owes much of its progress to versatile and fast physics simulators \[37, 38, 63, 64\], whose fairly accurate dynamics, ease of incorporating terrains and objects, and mas sive parallelization make training feasible at scale. Importantly, simulators implicitly handle discrete transitions allowing one to forgo explicit hybrid system models of locomotion. 

5  
2.2 Classical Real-time Feedback Control of Humanoid Locomotion 

The real-time control layer for legged robots operates at high loop rates and without a prediction hori zon. Early successes in bipedal locomotion were achieved through such real-time feedback control, which stabilized motion using physics-based models and handcrafted feedback laws. These classical controllers enabled the first demonstrations of stable walking and balance recovery but were limited in robustness and adaptability. Still, these controllers remains indispensable today, serving as the low-level layer that tracks motion commands in all control architectures, including learning-based (see “Learning based Control for Humanoid Locomotion”) and emerging approaches (see “Emerging Frontiers in Hu manoid Robot Control”), as illustrated in Fig. 3. Classical feedback methods are broadly categorized into linear approaches for reduced-order models and nonlinear approaches for full-order models; see \[8\] for a comprehensive review of real-time locomotion control. 

Linear feedback control. Early biped control replied on simplifying assumptions to derive analytical models and controllers. The Zero Moment Point (ZMP) method \[65, 66\] generated CoM trajectories to ensure stable quasi-static walking for fully actuated humanoids, often based on the LIP model \[66\]. Raibert’s control method \[19\] enabled dynamic hopping and running behaviors for bipeds based on compliant reduced-order models and foot placement. The capture point approach \[67\] bridged CoM and foot placement control by identifying where to step to stop motion, typically using LIP for efficiency. Subsequent methods generalized feedback control on reduced-order models using optimal control such as linear quadratic regulators \[68\]. Lastly, the simplest linear method is the Proportional-Derivative (PD) control, often combined with inverse kinematics or dynamics for classical motion tracking \[55, 69, 70, 71\]. In learning-based systems, PD target tracking \[72\] serves as the most popular interface between RL and real-time control: the RL policy outputs desired joint states, and a PD controller tracks them in real time (Fig. 3B). Future work may replace PD control with more advanced controllers for improved performance. 

Nonlinear feedback control. The simplifying assumptions of reduced-order models limit controller applicability. Full-order models remove these assumptions but require handling hybrid, nonlinear robot dynamics \[36, 73\]. This is typically achieved by decomposing the system into actuated and passive (zero) dynamics and applying nonlinear control \[36\]. Nonlinear full-order models can also be addressed by optimization-based control, such as embedding control Lyapunov functions (CLF) in a Quadratic Program (QP) (i.e., CLF-QP \[74, 75\]) and whole-body QPs \[25, 70, 76\]. They can track desired motions in real time but remain “greedy” in that they do not use a prediction horizon. 

2.3 Predictive Control for Humanoid Locomotion 

Advances in computation have enabled predictive control, a paradigm that optimizes behavior over a time horizon to enhance robustness beyond instantaneous feedback. These methods are generally for mulated as optimization problems that minimize a cost function *L*(*t*, *x*, *u*), composed of a running cost *l* and terminal cost *V* at the terminal time *T*, subject to system dynamics (discussed in “Physics-based An alytical Models”) and hard constraints, solved either offline through trajectory optimization or online via Model Predictive Control (MPC). The optimization yielding control inputs *u* and states *x* is expressed as: 

6

| Z  minimize  *l*(*t*, *x*, *u*)*d t* \+ *V*(*x*(*T*)) (Objective) (2a)  *u*,*xL*(*t*, *x*, *u*) \=  subject to *x*˙ \= *f* (*t*, *x*, *u*) (Dynamics) (2b) *x* ∈ X , *u* ∈ U (Constraints) (2c) |
| ----- |

Here X and U denote the admissible state and input sets, encompassing joint limits, contact constraints associated with the discrete model component in (1), and actuator bounds. For a comprehensive review of optimization-based methods, see \[17\]. 

Offline trajectory optimization. Trajectory optimization was a common tool for long-horizon plan ning, that is, planning over extended time horizons or across multiple footsteps, used prominently in the DARPA Robotics Challenges \[26\]. One successful strategy during the event involved two stages: (1) determining footstep locations and (2) planning continuous motions consistent with those steps \[25, 27, 29\]. Powerful solvers such as differential dynamic programming, sequential QP, and mixed-integer convex optimization were employed to plan both footstep locations and motions based on reduced-order models \[25, 29\], while later approaches unified them into a single optimization \[77\]. Subsequent work has incorporated upper-limb coordination, allowing hand contacts to enhance whole-body stability on challenging terrain \[78\]. Hybrid Zero Dynamics (HZD) is another trajectory optimization approach that stabilizes dynamic bipedal locomotion by explicitly addressing the robot’s full-order, nonlinear, hybrid, and underactuated dynamics \[21, 23, 36, 49, 73, 79\]. HZD can also generate human-like gait by em bedding human movement patterns into the optimization \[79\], mirroring the use of human data in RL as reviewed in “Learning from Real-World Human and Robot Data”. 

Online model predictive control. Solving trajectory optimization problems online introduces strict computational and timing constraints, which tends to limit horizons to under one second \[80\]. To reduce the dimensionality of the problem, controllers often use reduced-order models such as single rigid body or centroidal approximations \[81, 82\]. When nonlinear dynamics are considered, solving only one QP within a sequential QP framework enables real-time execution \[80, 83\]. 

To enhance robustness, MPC can adapt foot-contact locations and timings online rather than only optimizing continuous actions. One example is contact-implicit MPC \[84, 85, 86, 87\], which jointly optimizes contacts and motions via gradient-based solvers. Alternatively, sampling-based MPC employs gradient-free optimization to determine system inputs and implicitly choose contacts \[88, 89, 90\]. Yet, most humanoid implementations are limited to simulation, and achieving computational tractability in real-world deployment remains an open challenge. 

2.4 Summary of Performance Boundaries of Classical Control 

Classical control methods face fundamental limitations in real-world deployment. First, they rely on accurate state estimation and lack robustness under sensor noise, partial observability, or perceptual delays. Second, a persistent modeling gap also constrains performance: reduced-order models are fast but miss key limb and contact dynamics during agile motion, while full-order models capture these effects but are sensitive to modeling errors and environmental uncertainty such as terrain compliance and friction variation. Third, the core trade-off between model accuracy and computational speed remains unresolved: controllers based on detailed models are often too slow for onboard deployment, while simplified models may prevent agility. Finally, classical approaches require extensive manual tuning and strong contact assumptions, limiting adaptability to new tasks or environments. These challenges 

7  
motivate learning-based approaches that can handle uncertainty and partial observability, generalize across conditions, and use data to close the gap between model assumptions and real-world complexity, while still drawing on principles from classical control as discussed next. 

**A**   
**learning/optimization happen?**   
**When does major**   
Offline   
**Trajectory optimization Reinforcement**  

**learning Motion mimic** 

**Hybrid**    
Online 

Minor   
**Convex model  predictive control** 

**Linear**  

**feedback control** 

**Non-convex model  predictive control** 

**Nonlinear**  

**feedback control**   
**methods Emerging frontiers**  (using generative AI, world model, …) 

Reduced-order  

model 

**deployment**    
Full-order  model   
Physics-based  

simulator without  

real-world data 

**Model/data complexity**   
Physics-based  simulator with  real-world data   
Internet-scale  multimodal data 

**B Classical** 

**Learning Emerging** 

**Pre** 

**Deployment phase**   
**phase** 

Task planning \~1Hz Motion generation \~100Hz Joint torque \~1000Hz 

**Trajectory optimization** 

**Reduced-order model \+ Inverse kinematics** 

**Model predictive control** 

**Nonlinear** 

**feedback control** 

**Linear**  

**feedback control** 

**Reinforcement learning  training** 

**Reinforcement learning  inference** 

**Pre-training** 

**Fine tuning** 

**Generative models** 

**Foundation model inference** 

**Diffusion**  

**inference** 

“**System 2**” “**System 1**”   
Standard signal flow Hybrid methods 

Figure 3: Unified view of humanoid locomotion control paradigms across model–data complex ity and computational hierarchy. (A) Categorization of control approaches by when optimization or learning occurs (minor, online, or offline) and by model or data complexity. (B) Organization of con trol paradigms by their role in pre-deployment versus deployment phase, with corresponding control loop rates. Both views reveal a consistent progression from classical to learning-based and emerging paradigms. Classical real-time feedback controllers, including linear and nonlinear methods, involve minimal optimization or learning and operate at high frequencies (∼1000 Hz) but offer limited foresight and adaptability. Predictive control methods, such as model predictive control, reason about future states online (∼100 Hz) using physics-based models. Learning-based controllers, such as RL, are trained of fline in simulation and deployed onboard at comparable rates. Emerging generative approaches leverage large-scale data and hybrid training, typically operating at lower deployment frequencies due to com putational demands. The diagram also highlights two levels of reasoning: “System 1,” governing fast, reactive motor execution, and “System 2,” capturing slower, deliberative reasoning. This organization suggests a future integration of rapid feedback and high-level decision-making in humanoid control. Color shading indicates control paradigms: classical , learning-based , hybrid , and emerging . 

8  
Table 1: Summary of humanoid locomotion control paradigms and their core principles. 

*Category Main methods Core ideas Year Classical*   
Linear feedback control 

Nonlinear   
feedback control 

Trajectory   
optimization 

Model   
predictive   
control 

*Learning-based* 

Reinforcement learning with pure simulation 

Reinforcement learning with real-world data 

Model-guided reinforcement learning 

*Emerging* 

LLM / VLM guidance 

Diffusion   
motion models 

World /   
foundation   
models   
ZMP \[65, 66\]; Raibert’s control \[19\]; capture point \[67\] 

Inverse dynamics \[69\]; CLF-QP \[74, 75\]; 

whole-body QP   
\[25, 70, 76, 91\] 

HZD \[36\]; methods using reduced-order models \[25, 26, 27, 29, 92\] 

Methods using   
reduced-order models \[81\]; contact-implicit MPC \[84\]; sampling-based \[88\] 

Policy gradient \[93, 94\]; curriculum learning \[95\]; teacher-student \[96, 97\] 

Motion imitation   
\[98, 99, 100\]; residual learning \[48\] 

Tracking trajectories \[101, 102, 103\]; controllers for reward shaping 

\[104, 105\] 

LLM / VLM-conditioned control \[106, 107, 108\] 

Generative gait and   
trajectory priors \[100, 109\] 

Latent predictive dynamics \[110, 111\]   
Using reduced-order models and employing linear systems theory to regulate locomotion behavior in real time. 

Using full-order models and applying nonlinear control theory to produce joint torques for whole-body motion tracking. 

Long-horizon planning that selects footsteps and optimizes CoM and whole-body trajectories based on analytical models. 

Short-horizon predictive control with online replanning; reduced-order methods for real-time control; contact-implicit MPC and sampling-based methods for contact selection. 

Utilizing physics-based simulation and massive-parallel training to learn a policy offline and generate motion plans online. 

Motion imitation: using human motion data as reference to reduce the need for manually designed reward functions. Residual learning: mitigating the sim-to-real gap caused by modeling errors by refining both the simulator and the learned policy using real-world data. 

Using model-based methods to guide the learning and reduce dependence on heuristics; embedding trajectories or controllers into the RL reward. 

Conditioning control on language and vision for goal grounding and semantic reasoning for high-level intent planning. 

Diffusion priors sample diverse and feasible humanoid motions used as tracking and regularization targets. 

Learning latent dynamics from data that predicts future observations, rewards, and contact events, enabling control via imagined rollouts in latent space.   
1980s– 2000s– 

2010s– 2010s– 

2016– 2018– 

2020– 

2024– 2025– 2025– 

3 Learning-based Control for Humanoid Locomotion 

Conventional RL algorithms, developed since the 1980s \[112, 113, 114, 115, 116\], struggled to scale in high-dimensional continuous spaces due to limited function approximation and costly data collec tion \[117, 118, 119, 120\]. Neural networks (NN) have been integrated into RL since around the 1990s (e.g., TD-Gammon \[121\] and early NN-based RL \[122, 123\]), but the 2013-2015 rise of deep learn ing \[124, 125\] triggered modern Deep RL (DRL). Breakthroughs such as Deep Q-Networks \[126, 127\], AlphaGo \[128\], and continuous control in MuJoCo \[93\] marked the shift toward scalable policy learning. 

Translating DRL to humanoid locomotion remained challenging due to high-dimensional continuous actions, contact-rich dynamics, and costly failure. Key obstacles include slow or inaccurate simulators, expensive and risky real-world data collection, and insufficient compute for large-scale training. Re cent advances have mitigated these barriers. Modern GPU simulators such as NVIDIA Isaac Sim \[38\] 9  
provide fairly realistic dynamics and thousands of parallel environments, cutting training time by or ders of magnitude. Combined with open-source DRL libraries \[129, 130, 131\], robust hardware, and improved algorithms, learning-based control now exceeds classical approaches in locomotion agility and robustness (Fig. 1). Notably, RL for quadrupeds \[45, 132, 133\] and bipeds \[134, 135, 136, 137\] paved the way for humanoids, addressing the challenge of sim-to-real transfer. Subsequent work incor porated human motion data during training \[98, 100, 138, 139, 140\], improving realism and efficiency. These advances have enabled extreme behaviors such as parkour, human motion imitation, and dynamic loco-manipulation (e.g., humanoid table tennis \[141, 142\]). 

This section reviews learning-based control for humanoid locomotion, focusing on simulation-driven RL, learning from real-world data, and hybrid approaches integrating RL with model-based control (Fig. 3). It also highlights the architectural and conceptual connections between learning-based and classical control paradigms (Tab. 2). 

3.1 Learning from Pure Simulation 

RL enables humanoid locomotion controllers to learn agile and robust behaviors directly from inter action data, and simulation provides a safe, efficient environment for exploring diverse terrains and disturbances before real-world deployment. RL-based humanoid locomotion control can be formu lated as a Partially Observable Markov Decision Process (POMDP) \[143\]. A POMDP is defined as M \= 〈S ,A , *p*,*r*,O ,*γ*〉, where S , A , and O are the admissible state, action, and observation sets. Here *p*(*s*′|*s*, *a*) denotes the state transition function capturing the system dynamics that governs the evo lution from state *s* and action *a* to state *s*′, *r*(*s*, *a*) is the reward function, and *γ* is the discount factor. The agent learns a policy *π*(*at*|*ot*) that maps observations *ot*to action *at* at timestep *t* to maximize the expectation of the discounted cumulative reward *J*(*π*), which encourages long-horizon performance under stochastic dynamics: 

| X∞    *γtr*(*st*, *at*)  maximize  (Objective) (3a)  *πJ*(*π*) \= E*τ*∼*pτ*(·|*π*)  *t*\=0  subject to *st*\+1 ∼ *p*(· | *st*, *at*), (Dynamics) (3b) *at* ∼ *π*(· | *ot*), (Action model) (3c) *ot* ∼ O (· | *st*), (Observation model) (3d) *st* ∈ S , *at* ∈ A , *ot* ∈ O , (Constraints) (3e) |
| ----- |

where *τ* \= (*s*0, *o*0, *a*0,*r*0,*s*1, . . .) denotes a trajectory sampled from the trajectory distribution *pτ*(· | *π*) induced by policy *π* and the environment dynamics *p*(*s*′|*s*, *a*). The observation model O (*ot*|*st*) reflects partial observability. For an in-depth discussion of POMDP and RL, see \[16, 116\]. 

To enable efficient training and real-world deployment, recent research has introduced a suite of complementary techniques. The rest of this subsection reviews advances in parallelized simulation plat forms, open-source learning frameworks, efficient policy training algorithms, and sim-to-real transfer strategies, which collectively enhance scalability, reproducibility, and robustness in RL-based humanoid locomotion control. 

Parallelized GPU-accelerated simulators. RL typically requires extensive trial-and-error interac tions, making data collection a major training bottleneck. To address this, the field has seen a significant shift in simulation frameworks, from traditional CPU-based engines, such as RaiSim \[63\] and Pybullet 

10  
\[64\], to massively parallel GPU-accelerated platforms, including NVIDIA Isaac series \[38\] and MuJoCo MJX \[144\]. These modern simulators can execute thousands of environment rollouts simultaneously on a single consumer-grade GPU, improving simulation throughput by several orders of magnitude and making rapid experimentation far more accessible. 

Policy gradient methods and open-source frameworks. Advances in policy gradient algorithms, such as Trust Region Policy Optimization \[93\] and Proximal Policy Optimization \[94\], have enabled stable and sample-efficient learning in legged locomotion, building on successes like AlphaGo \[128\]. Together with high-quality open-source libraries \[131, 144, 145, 146\], these advances have made RL re search more accessible, reproducible, and customizable. Shared pipelines and pretrained models reduce overhead for newcomers, accelerate experimentation, and promote fair comparisons across methods. 

Domain randomization. Domain randomization \[147\] improves sim-to-real transfer by training poli cies under randomized conditions such as sensor noise, actuator latency, external forces, and model pa rameters. By exposing the policy to broad variations in simulation, it becomes more robust to real-world uncertainties. Carefully designed randomization schemes have demonstrated significantly improved transfer success in legged locomotion \[132, 148\]. 

Curriculum learning. Learning locomotion in high-dimensional, dynamic environments remains chal lenging. Curriculum learning \[149\] addresses this by gradually increasing task complexity, allowing policies to acquire basic motor skills before adapting to complex scenarios. For instance, training can begin on flat terrain without disturbances and progress to uneven terrain or external pushes. Adaptive curriculum design further adjusts difficulty based on policy performance during training \[95, 137, 150\]. 

Teacher-student and asymmetric actor-critic frameworks. The teacher-student framework \[96, 151\] improves learning by using privileged information, such as terrain friction, contact states, or external forces available only during training. The teacher policy, with full information, learns robust behaviors, while a student policy restricted to onboard sensors is trained to imitate the teacher. To improve gen eralization, a reconstruction loss can help the student infer unobservable features, accelerating training and benefiting sim-to-real transfer \[97, 150\]. Similarly, the asymmetric actor-critic framework \[152\] as signs privileged information to the critic while limiting the actor (policy) to onboard observations. This asymmetry has recently gained attention for its simplicity and strong performance \[153, 154\]. 

3.2 Learning from Real-World Human and Robot Data 

Learning from real-world human data. Because humanoid locomotion involves high-dimensional actions and complex limb coordination, simple reward terms including velocity tracking or gait fre quency often yield unnatural behaviors, such as straight knees, foot scuffing, or asymmetric gait \[138, 155\]. To address this, researchers increasingly utilize human motion priors to guide humanoid skill learning. 

In computer graphics and animation, a variety of public motion capture datasets \[156, 157, 158, 159, 160, 161, 162, 163\] provide valuable priors for this purpose. Due to the embodiment gaps between humans and humanoids, a retargeting step \[139, 164, 165, 166, 167\] is typically required to convert human motion into kinematically feasible references. Once motion references are obtained, they can be incorporated into RL through two main strategies. Indirect (adversarial) methods, such as GAIL \[168\] and AMP \[138, 169\], learn discriminators to align robot motion distributions with those in reference data. Direct tracking methods, such as DeepMimic and its variants \[48, 98, 100\], H2O \[99, 139\], HumanPlus \[170\], ExBody \[171\], and general tracking methods \[172, 173, 174, 175\], define rewards 

11  
explicitly as errors between reference and robot motions. Tracking-based schemes also extend to loco manipulation, where RL agents jointly track robot, object, and contact references \[103, 166, 176, 177\]. 

System identification and actuator model learning from real-world data. While physics-based simulation reduces the gap between simulation and hardware, some parameters such as inertia are diffi cult to identify, and actuators or transmission dynamics remain hard to model due to nonlinearities and mechanical imperfections. Real-world data is therefore essential for improving simulation fidelity. 

Because locomotion exhibits non-smooth hybrid dynamics, sampling-based identification methods are often preferred \[134, 178\]. For actuators, NN models trained on hardware data can predict actual torque output, capturing actuator behavior more accurately and enhancing simulation realism (Fig. 2). Studies on the ANYmal quadruped show that such hybrid modeling improves simulation accuracy and control performance \[45, 131, 179\]. 

Learning robot dynamics from real-world data. System identification and actuator modeling remain limited by predefined structures. System identification requires prior knowledge of which parameters to estimated, while actuator models focus only on local uncertainties. A broader family of approaches in stead aims to learn full-body robot dynamics directly from real-world data (Fig. 2). Given the complexity of humanoid dynamics, residual learning is a common strategy, where a residual model is learned on top of nominal simulator dynamics models to account for unmodeled effects. For example, ASAP \[48\] and its variants \[180, 181\] learn residual action models that align simulation with real-world dynamics. Be yond residual learning, entire forward dynamics can also be learned from both simulated and real-world data to achieve safe navigation in complex environment \[39\]. 

3.3 Inherent Connections to Classical Control 

Building model-based systems requires expertise in dynamics, control, and optimization, whereas learning based pipelines are often more accessible. However, classical principles remain essential, especially in RL components like domain randomization, reward design, and curriculum learning, all of which rely on understanding underlying dynamics. This section explores the connections between classical and learning-based control, offering a unified framework for humanoid locomotion (Tab. 2, Fig. 3). 

An optimal control view point on RL-trained locomotion controller. As shown in (3) and Fig. 3, RL-trained locomotion controllers effectively solve a large number of optimization problems offline in simulation, and “encode” the solutions into NN policies. The optimization problems are solved across enormously diverse states and environments, producing policies that are adaptive, robust, and general izable. Meanwhile, classical predictive control methods, formulated as in (2), often solve the problem *online* for a given state using simplified models (e.g., reduced-order model) to ensure fast computation and responsive behavior. In general, the resulting behavior is often sub-optimal due to model inac curacies and the need for explicit state estimation. In contrast, RL policies can leverage large-scale GPU-accelerated offline training, sophisticated simulators, and frameworks such as teacher-student or asymmetric actor-critic learning to bypass explicit state estimation and improve generalization. How ever, as suggested by recent advances, leveraging both offline and online computation often leads to best results (details in “Emerging Frontiers in Humanoid Robot Control” and Fig. 3B). 

Architecture and environment design in RL-based pipelines inspired by classical control. The connections between RL training and classical control are detailed next (Tab. 2). (i) Domain random ization \[72, 147\], which adds noise into the training process, aligns with principles of robust control \[194, 200\], where controllers are designed to ensure robustness across a range of system parameters. (ii) 

12  
Table 2: Core challenges in humanoid locomotion control and inherent connections across different paradigms. Emerging topics primarily refer to generative approaches such as diffusion policies, vision language-action (VLA) models, and world models. 

*Key*   
*challenges of humanoid locomotion*   
*Key ideas to resolve the challenges shared by different approaches* Classical Learning-based Emerging 

Stabilizing   
Analytical dynamic 

learning-based controller \[131, 182\]World models for dynamics,   
unstable   
dynamics 

High   
dimensionality   
modeling \[36, 33\];   
stabilizing controllers \[54, 66\] 

Reduced-order models \[50, 33\]; hierarchical control frameworks \[55\]   
Physics simulators \[37, 38\]; 

Autoencoders, latent   
space \[43, 107\];   
latent dynamics \[184\] 

planning, and control \[41, 111, 183\] 

Diffusion policies \[100, 185\]; VLA models \[186, 187\] 

hybrid decision making \[84, 88\]Learning hybrid automata \[188\]Generative sequence hybrid dynamicsHybrid system formulations \[36\];   
Stabilizing 

models \[189, 190\]; world models \[111\] 

Constraints Constrained optimization \[91\] Reward shaping \[136, 191\]Conditional diffusion \[100, 192\]; VLA grounding \[108, 193\] 

Robustness Robust control \[194, 195\] Domain randomization \[147, 148\]Diffusion for diverse motion \[109\];   
large-scale VLA \[196\];   
world models \[197\] 

Adaptation Adaptive control \[59, 198\] Adaptation modules \[199\]In-context learning \[106\]; diffusion test-time adaptation \[100\] 

Adaptation module and teacher-student frameworks \[45, 199, 201\] reflect the idea of adaptive control \[59, 198, 202\], which estimates and compensates for uncertain system parameters online using historical state-action data. (iii) Finding the latent space, whether through modern methods such as autoencoders \[43, 184\] or classical techniques (e.g., principle component analysis \[203\], singular value decomposi tion \[204\], and using dynamics reduction through feedback \[55\]), follows the same goal of representing the high-dimensional robot dynamics using low-dimensional coordinates for efficient planning and con trol \[55, 43, 184\]. (iv) The hybrid nature of locomotion, marked by discrete contact switching, has long been modeled as a hybrid dynamical system \[36\], inspiring structural design of learning policy, e.g., hybrid automata learning \[188, 205\], that explicitly handle multiple contact modes and transitions. (v) Hierarchical or cascaded control structures \[55\] common in classical approaches continue to inform hierarchical learning frameworks \[141\], offering interpretable interfaces, modularity and reuse, and im proved sim-to-real transfer. 

Merging classical control and RL for performance, robustness, and safety. Tuning RL policies for humanoid locomotion is often time-consuming due to the need for carefully designed rewards, envi ronments, and curricula. To address this challenge, model-guided approaches leverage physics-based or control-theoretic insights to reduce heuristics while improving precision, robustness, and safety. Reduced-order models (e.g., SLIP \[101\] and single rigid body \[35\]) and full-order models \[102, 103\] have been used to generate reference motions for RL tracking. Furthermore, rewards grounded in con trol theory, such as those based on CLFs, enhance reference tracking \[105, 206\]. Other methods em bed controllers directly within RL training, for instance using LIP-based planners to provide desired foot placements \[104\]. Trajectory computation can also be performed online, where MPC is computed during both training and deployment to enable adaptive motion generation \[207\]. Finally, integrating control-theoretic safety notions with RL provides formal safety guarantees \[208, 209\]. 

13  
3.4 Summary of Performance Boundaries of Learning-based Control 

Despite advances improving the generalizability, robustness, and sample efficiency of learning-based humanoid locomotion control, several limitations persist. First, these methods remain algorithmically fragile: without motion references or curricula, they often converge to unnatural gaits \[98, 138\] and fail on challenging terrain \[150\]. Second, transferring a single policy across tasks or morphologies remains difficult, as embodiment differences and motion retargeting typically require retraining or hierarchical control \[166, 210\]. Third, sim-to-real fidelity is still limited. Domain randomization mitigates physics mismatches but cannot fully eliminate them, especially for loco-manipulation and interaction tasks. Ex isting correction methods are often task-specific or complicate the pipeline \[45, 48, 211\]. Fourth, many policies rely solely on proprioception, omitting perception and tactile sensing. Perception improves observability but introduces latency, brittleness, and new sim-to-real gaps \[212\], while added sensing increases calibration and compute demands. Fifth, even with GPU-parallel simulation, training still re quires high-end hardware, extensive environment engineering, and nontrivial real-world data collection for system identification and edge-case coverage \[131\]. Finally, ensuring safety is still a major chal lenge: learning-based controllers operate largely as black boxes, making failures unpredictable. Though constrained RL and control-theoretic filters can enhance safety, they remain rare and add modeling and computational overhead \[213, 214, 215, 216\]. 

4 Emerging Frontiers in Humanoid Robot Control 

Recent advances in RL have expanded humanoid control beyond its classical foundations, paving the way for generative approaches that promise greater adaptability and naturalness in motion (Fig. 1). These advances (Fig. 4) include a progression from *discriminative* to *generative* modeling, moving from direct, reactive input-output mappings toward models that learn underlying data distributions and generate or predict future trajectories; from *uni-modal* to *multi-modal* learning that unifies perception, language, and proprioception; from *single-task* to *multi-task* control with shared policies for diverse behaviors; and from *locomotion* to *loco-manipulation*, extending control from agile mobility to whole body, tool-using interaction. 

**Learning paradigm Discriminative** 

**Modality** 

Proprioceptive 

**Uni-modality** 

Language Tactile Perceptive Audio 

**Task scope Single-task** 

**Functionality Locomotion** 

**Computation** Policy Environment 

**Offline training** Policy Environment 

**Generative Test–time adaptation**   
**Multi-modality**   
**Multi-task**   
**Loco-manipulation** 

Figure 4: Emerging research shifts in humanoid control across five dimensions. Current trends reflect a transition from discriminative to generative models, uni-modal to multi-modal sensing, single task to multi-task learning, locomotion to loco-manipulation, and offline training to test-time adaptation. These shifts point toward unified generative approaches that integrate perception, reasoning, and control to enable humanoids to generalize across modalities, tasks, functionalities, and environments. 

14  
The driving force behind these shifts is the demand for general-purpose humanoids capable of human-level reasoning and physical performance in diverse, complex, novel environments. Break throughs in foundation-scale AI tools, such as Large Language Models (LLMs) \[106\], Vision-Language Models (VLMs) \[108\], Vision-Language-Action Models (VLAs) \[186\], diffusion models \[100\], and world models \[111\], make this goal increasingly attainable. The models offer expressive capacity for diverse motion styles, robustness to sparse or partial data, seamless integration of multi-modality, and enhanced transferability for zero-shot or few-shot adaptation across terrains and morphologies. Ex amples include next-token prediction for sequential motion generation \[189\], diffusion policies for terrain-adaptive walking \[100\], and multi-modal foundation models for high-level locomotion plan ning \[108, 217\]. 

Early progress in other domains, primarily tabletop manipulation \[218\], suggests that generative models could enable humanoids to perform agile whole-body manipulation, mobile tool use, and phys ically interactive collaboration. Yet, integrating these models into humanoid control is challenging. Key barriers include data scarcity, safety, and the need for high-frequency inference at control rates above hundreds of Hz. Additional barriers specific to humanoids stem from their underactuated, high dimensional dynamics, stringent stability requirements, and extreme agility demands. 

The remainder of this section surveys the emerging generative paradigms, their connections to clas sical methods, and the open challenges and opportunities shaping the next decade of humanoid control. 

4.1 Learning Paradigm Shift: From Discriminative to Generative Models 

Motivation and limitations. In learning-based locomotion, *discriminative* models learn a direct map ping from states (or observations) to actions, often producing a single deterministic action. While effective on hardware, such policies are limited to behaviors that are explicitly rewarded or demon strated \[45, 94, 219\], making multi-modal choices (e.g., stepping left or right) and adaptation to data shifts difficult without retraining. These limitations motivate *generative* policies that model a distri bution over feasible actions or trajectories, enabling uncertainty awareness and explicit conditioning at inference time \[98, 138, 192, 220, 221, 222, 223, 224, 225, 226\]. 

Unlike point estimates, a generative control policy *πφ* with parameters *φ* learns a distribution over feasible trajectories rather than a single deterministic control sequence. It is trained to reproduce ex pert demonstrations or real-world trajectories while remaining physically plausible, similar to diffusion policies conditioned on state or observation history. This problem can be expressed as: 

| *φJ*(*πφ*) \= E*τ*∼*pτ*(·|*πφ*) log *p*data(*τ*) − *λ*D*πφ*(*τ*) *p*phys(*τ*)   (Objective) (4a) maximize  subject to *st*\+1 ∼ *p*(· | *st*, *at*), (Dynamics) (4b) *at* ∼ *πφ*(· | *o*≤*t*), (Generative model) (4c) *ot* ∼ O (· | *st*), (Observation model) (4d) *st* ∈ S , *at* ∈ A , *ot* ∈ O , (*st*, *at*) ∈ Cphys. (Constraints) (4e) |
| :---- |

Here, *J*(*πφ*) is a data-driven objective that encourages the policy *πφ* to match the data distribution *p*data(*τ*) of experts or real-world trajectories while remaining consistent with physical feasibility via the KL regularization term D(*πφ*∥ *p*phys) weighted by *λ* ≥ 0. The term *p*phys(*τ*) represents a physics-aware prior or simulator-consistent process that enforces dynamic plausibility. The policy *πφ* models the con ditional action distribution given partial observations *ot*, typically parameterized as a denoising process 

15  
refining latent noise into valid actions. The probabilistic dynamics *st*\+1 ∼ *p*(· |*st*, *at*) generalize real or learned transition models, and the observation model *ot* ∼ O (· | *st*) captures partial observability. The constraint (*st*, *at*) ∈ Cphys enforces physical feasibility such as contact, friction, torque, and kinematic limits. This formulation reframes motion generation as sampling from a distribution of feasible fu tures, unifying classical predictive control and generative modeling within a common decision-making framework. 

Pathways from discriminative to generative policies. Bridging these paradigms, adversarial imita tion learning reveals motion priors and style rewards \[138, 220\], while transformer- and diffusion-based planners enable trajectory-level reasoning and conditional synthesis \[192, 222, 224, 225\]. In visuomotor tasks, diffusion policies perform online replanning within learned manifolds \[226\]. World models ex tend this idea by performing control through latent imagination \[41, 111, 197\]. In humanoid locomotion, generative motion models (e.g., adversarial or diffusion-based) sample diverse, physically consistent footstep and CoM trajectories under uncertainty, while discriminative controllers handle fast stabiliza tion \[109, 227\]. A hierarchical design, where a generative planner proposes motion distributions and a low-level controller executes them, combines reactivity with expressivity \[189, 224, 226, 228, 229\]. This integration transforms locomotion from deterministic mapping to generative reasoning, sampling from feasible futures for adaptive, multi-modal control. 

4.2 Modality Shift: From Uni-modal to Multi-modal Learning Empowered by Founda tion AI Models 

Motivation and limitations. Most humanoid locomotion controllers remain *uni-modal*, relying solely on proprioceptive feedback, such as joint encoders, inertial measurement units, and contact sensors, without exteroceptive perception. Rooted in early model-based control \[29, 67, 230, 231\], such “blind” controllers achieve real-time balance but lack look-ahead capability. Although recent DRL policies show strong agility and robustness \[45, 131, 132, 201, 232, 233\], their proprioceptive design limits performance in terrain-aware tasks such as stairs or gaps. This motivates a shift toward *multi-modal* learning that integrates vision, depth, or tactile sensing to enhance anticipation and adaptability. Em pirical studies show that vision-based locomotion policies can double success rates over blind base lines \[234, 235\]. Fusion architectures such as LocoTransformer \[236\], VB-Com \[237\], and attention based encoders \[96, 238\] further improve terrain adaptation and temporal consistency. Overall, uni modal controllers remain efficient but short-sighted, motivating perception-rich multi-modal frame works. 

From uni-modal control to multi-modal integration. Achieving robust multi-modal control requires architectures that preserve modality-specific cues while supporting cross-modal reasoning. Recent frameworks fuse proprioception, vision, and additional inputs (e.g., lidar and tactile) using structured fusion or Transformer-based cross-attention \[236, 239\], enhancing terrain affordance modeling and con textual adaptability \[218, 240\]. Large robot datasets \[187, 196\] and foundation AI models have accel erated this trend. Vision-language-action (VLA) systems such as RT-2 \[187\], *π*0\[241\], and Gemini Robotics \[242\] unify perception, reasoning, and control through internet-scale pre-training. Humanoid oriented variants like Helix \[228\] and GR00T N1 \[229\] embed vision, proprioception, and language conditioned goals into unified latent spaces. Such policies enable zero-shot generalization to unseen terrains and semantically grounded locomotion. This marks a shift from reactive stabilization to context aware, intelligent control for generalist humanoid agents. 

16  
4.3 Task Scope Shift: From Single-Task to Multi-Task Policy with Generalizability and Adaptability 

Motivation and limitations. Conventional humanoid locomotion controllers are optimized for *single tasks*, e.g., walking, running, or jumping, each trained with its own rewards and hyperparameters. While effective within domain, they lack scalability and fail to adapt to new goals or conditions without re training. This narrow specialization hinders the creation of versatile humanoids capable of composing or adapting skills. Recent work therefore pursues *multi-task* policies that share representations across tasks to learn reusable locomotion priors. Training one model over diverse tasks promotes shared structure, data efficiency, and adaptability beyond rigid single-task setups \[45, 131, 132, 201\]. However, multi task learning introduces reward conflicts, gradient interference, and catastrophic forgetting \[243, 244\], calling for structured training and scalable architectures. 

Toward generalizable multi-task policies. Recent approaches seek controllers that generalize across tasks. Multi-task RL uses shared parameters or modular architectures to co-train diverse skills \[245, 246\], while hierarchical schemes organize reusable motion primitives and high-level composition \[247, 248\]. Diffusion-based motion synthesis enables skill blending and interpolation at inference \[224, 225\]. Cross-embodiment generalization leverages heterogeneous datasets (e.g., BridgeData V2 \[249\], RT X \[196\]) and meta-learning or distillation \[250\] to share latent skill spaces \[251, 252\]. The latest trend, large-scale behavior foundation models \[187, 218, 228, 229, 239, 240\], unifies task, morphology, and sensing modalities within one latent policy, enabling zero-shot transfer and open-world locomotion. This marks the transition from handcrafted single-skill control to broadly adaptive agents. 

4.4 Functionality Shift: From Locomotion to Loco-Manipulation and Interaction 

Motivation and limitations. Traditional humanoid research has treated *locomotion* and *manipulation* separately: locomotion focuses on balance and terrain adaptation \[29, 45, 131\], while manipulation emphasizes dexterous grasping and contact-rich control \[253, 254, 255\]. This separation limits coordi nated whole-body behaviors. Locomotion-centric systems can traverse but not interact, while manip ulation frameworks often assume a static base and ignore body dynamics. To enable general-purpose humanoids capable of assistance, tool use, and human collaboration, locomotion must evolve into *loco manipulation*, a unified paradigm jointly reasoning over movement, contact, and force. 

Toward unified loco-manipulation and interaction. Recent studies pursue this unification through human motion mimicking, hierarchical control, multi-modal perception, and generative planning. For single tasks, co-tracking approaches jointly track robot, object, and contact references from retargeted human-scene interaction data \[103, 166, 176, 177\]. Diffusion-based models synthesize physically plau sible human-object interaction trajectories \[256\], while task-and-motion planning couples contact rea soning with long-horizon objectives \[257\]. Hierarchical systems integrate high-level vision planning with low-level RL \[139\]. Dual-agent RL frameworks such as FALCON \[153, 182\] synchronize upper and lower-body control, and hybrid visuomotor systems \[258\] combine visual and proprioceptive cues for tasks like carrying or door operation. Unified loco-manipulation controllers such as ULC \[259\] achieve end-to-end walking and handling coordination. At scale, behavior foundation models (e.g., Helix and GR00T N1 \[196, 228, 229\]) integrate locomotion, manipulation, and interaction trajectories into shared visuomotor latent spaces, enabling zero-shot transfer across tasks and morphologies. These advances represent the transition from locomotion-focused control to embodied humanoids that move, manipulate, and collaborate fluidly in complex environments. 

17  
4.5 Inherent Connections to Classical Control 

Despite their seemingly radical departure from traditional paradigms, generative and foundation-scale models share core principles with classical control (Tab. 2). LLMs function as dynamic feedback sys tems: their evolving context window acts as memory (state), updated by each new observation and token to form a closed feedback loop in latent space, as formalized in *Prompt a Robot to Walk* \[106\]. Diffusion models \[260\] mirror stochastic optimal control, where denoising solves a stochastic differen tial equation minimizing an implicit energy functional, analogous to optimal control under uncertainty via the Hamilton-Jacobi-Bellman framework. Representation learning in LLMs \[106\], VLMs \[108\], and VLAs \[186, 229\] likewise seeks latent manifolds where tasks become controllable and observable, echo ing classical concepts of state estimation and observability. The transformer’s self-attention mechanism can be viewed as adaptive gain scheduling\[261\], dynamically weighting tokens by contextual relevance to implement state-dependent feedback. In-context learning further parallels adaptive control\[59, 202\], where controllers refine online from input-output behavior without explicit parameter updates. Viewed through this lens, generative models extend rather than replace control-theoretic foundations, embodying feedback through autoregression, adaptation through context updates, gain scheduling through attention, and optimality through diffusion guidance. 

4.6 Toward the Next Generation of Humanoid Control 

Humanoid control is evolving from walking and manipulation toward safe, reliable, and useful operation around people. Solving this open challenge now depends on integrating control with safety, scalable hardware, perception, and human-compatible decision making to transform lab systems into deployable collaborators. 

Safer and more reliable humanoids. Falling remains the main barrier to humanoid deployment. Stan dardized tests for recovery, fall rate, and impact severity \[262\], beyond average performance, are needed to evaluate robustness, especially in rare failure cases \[213\]. Achieving reliability will rely on compli ant actuation, energy-dissipating design, and test-time adaptation (Fig. 4) that adjusts control online to unforeseen disturbances during deployment. 

Scalable and accessible platforms. The cost of humanoid platforms has dropped by over an order of magnitude through advances in actuators, lightweight design, and scalable manufacturing. Continued reductions via mass production could democratize access, but affordability alone is insufficient. To move from research prototypes to practical, economically viable systems, platforms must also be durable, maintainable, and capable of recovering from minor damage without expert repair. 

Perceptual and cognitive intelligence. Next-generation humanoids will integrate control with cogni tive and perceptual intelligence. Multi-modal sensing and generative planning will enable anticipatory, context-aware behavior, blurring the boundary between control and cognition so humanoids can act, un derstand, anticipate, and convey intent in human-compatible ways. This integration naturally suggests a dual-system hierarchy (Fig. 3B): a fast, reflexive “System 1” for stabilization and real-time interac tion, and a slower, deliberative “System 2” for predictive reasoning \[263, 264\]. Achieving effective coordination between these layers may be key to unified cognitive-motor control in future humanoids. 

Embodied intelligence ecosystems. Humanoid robots could become as transformative as personal computers. While computers enabled disembodied intelligence in software, humanoids will bring em bodied intelligence into the physical world. Their evolution may mirror the path of personal computers: 

18  
from isolated machines to networked, specialized embodiments (“an internet of humanoids”) across con texts. This shift could enable shared platforms, new industries, and research in embodied safety, ethical autonomy, and large-scale human-robot collaboration. 

5 Getting Started in Humanoid Locomotion Control 

This section offers concrete guidance on hardware choice, software infrastructure, RL workflow, and foundational references so new teams could build controllers that are physically grounded rather than opaque. 

Starting from first principles. Even with learning-based methods, understanding dynamics, feed back control, and optimization remains indispensable. These principles explain gait stability, controller failure, and hardware safety, making learned policies more interpretable and reliable. 

Hardware: build or buy. Teams typically choose between open-source and commercial humanoids. Open-source systems, such as the Berkeley Humanoid \[265\], Berkeley Humanoid Lite \[266\], Stanford ToddlerBot \[267\], NimbRo-OP2X \[268\], and Poppy Humanoid \[269\], offer public designs, actuators, and baseline controllers. They are relatively low-cost and modifiable but require careful assembly and maintenance. Commercial robots, including Unitree’s G1 and H1, Booster’s T1 and K1, EngineAI’s SE/PM, PNDbotics’ Adam, and U.S. platforms such as Agility’s Digit and Apptronik’s Apollo, pro vide safety-tested hardware, integrated software, and vendor support. They cost more but accelerate algorithm development and improve reproducibility. In short, open-source systems maximize flexibility, while commercial systems maximize productivity. 

Open-source software: control stacks, simulators, and data. Progress is increasingly enabled by shared infrastructure. For model-based control, MIT Cheetah \[270\] and OpenLoong \[271\] demonstrate whole-body and MPC-based locomotion. CasADi \[272\], Acados \[273\], OCS2 \[274\], and Judo \[275\] make nonlinear predictive control practical. For learning-based control, Isaac Lab \[276\], MuJoCo Play ground \[144\], and MJLab \[277\] support GPU-accelerated RL simulation, while Newton \[278\] and Gen esis \[279\] scale to large parallel computation with differentiable physics. Human motion datasets such as AMASS \[156\] and pipelines like BeyondMimic \[100, 166\] help encode human-like behaviors and enhance robust transfer to hardware. 

Getting started with RL. A practical progression includes: (a) Install Isaac Sim and Isaac Lab; (b) Execute a walking policy in simulation; (c) Train balancing and walking controllers; (d) Condition on commanded velocity; (e) Track human motion; (f) Learn perceptive locomotion on rough terrain; and deploy a learned policy from (c)-(f) on hardware. 

Core references. For deeper study, we recommend: *Feedback Control of Dynamic Bipedal Robot Locomotion* \[36\] and *Robot Modeling and Control* \[280\] for dynamics and classical feedback control; *Reinforcement Learning: An Introduction* \[116\] and *PPO* \[94\] for decision making; *DreamerV3* \[197\] for world-model based control; *Deep Learning* \[281\], *Introduction to Variational Autoencoders* \[282\], and *The Principles of Diffusion Models* \[283\] for generative foundations. 

6 Conclusion 

This paper has reviewed the evolution of humanoid locomotion control from classical real-time feedback and predictive control to reinforcement learning and emerging generative paradigms. We established a 

19  
unified perspective that connects these paradigms through shared principles of physics-based modeling, constrained decision-making, and adaptation to uncertainties. We clarified the distinct characteristics of each paradigm: classical control provides stability and interpretability, learning-based approaches enable agility and robustness, and emerging generative methods promise versatility and generalization across tasks and functionalities. Beyond this synthesis, we provided practical recommendations that lower the barrier to entry for new researchers and practitioners in humanoid control. Finally, we outlined future challenges and opportunities toward humanoids capable of intelligent, adaptive, and safe operation in real-world human environments. 

Acknowledgments 

This template is modified from the Max Simchowitz paper template and Meta paper template. 20  
References 

\[1\] S. Shigemi, “Asimo and humanoid robot research at honda,” in *Humanoid robotics: A reference*, pp. 1–36, Springer, 2017\. 

\[2\] “Unitree g1.” https://www.unitree.com/cn/g1. Accessed: 2025-11-21. 

\[3\] B. Dynamics, “Large behavior models and atlas find new footing.” https:// bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/2025\. Accessed: 2025-11-21. 

\[4\] A. Roychoudhury, S. Khorshidi, S. Agrawal, and M. Bennewitz, “Perception for humanoid robots,” *Current Robotics Reports*, vol. 4, no. 4, pp. 127–140, 2023\. 

\[5\] Y. Tong, H. Liu, and Z. Zhang, “Advancements in humanoid robots: A comprehensive review and future prospects,” *IEEE/CAA Journal of Automatica Sinica*, vol. 11, no. 2, pp. 301–328, 2024\. 

\[6\] Z. Zhao, S. Cheng, Y. Ding, Z. Zhou, S. Zhang, D. Xu, and Y. Zhao, “A survey of optimization based task and motion planning: From classical to learning approaches,” *IEEE/ASME Transac tions on Mechatronics*, vol. 30, p. 2799–2825, Aug. 2025\. 

\[7\] W. Talbot, J. Nubert, T. Tuna, C. Cadena, F. Dümbgen, J. Tordesillas, T. D. Barfoot, and M. Hut ter, “Continuous-time state estimation methods in robotics: A survey,” *IEEE Transactions on Robotics*, vol. 41, pp. 4975–4999, 2025\. 

\[8\] J. Reher and A. D. Ames, “Dynamic walking: Toward agile and efficient bipedal robots,” *Annual Review of Control, Robotics, and Autonomous Systems*, vol. 4, no. 1, pp. 535–572, 2021\. 

\[9\] G. Satheesh Kumar, S. Aravind, R. Subramanian, S. K. Bharadwaj, R. R. Muthuraman, R. Steve Mitchell, A. Bucha, M. Sriram, K. Shri Hari, and N. J. Robin, “Literature survey on four-legged robots,” *Trends in Mechanical and Biomedical Design*, pp. 691–702, 2021\. 

\[10\] J. Carpentier and P.-B. Wieber, “Recent progress in legged robots locomotion control,” *Current Robotics Reports*, vol. 2, no. 3, pp. 231–238, 2021\. 

\[11\] A. Torres-Pardo, D. Pinto-Fernández, M. Garabini, F. Angelini, D. Rodriguez-Cianca, S. Mas sardi, J. Tornero, J. C. Moreno, and D. Torricelli, “Legged locomotion over irregular terrains: State of the art of human and robot performance,” *Bioinspiration & Biomimetics*, vol. 17, no. 6, p. 061002, 2022\. 

\[12\] T. Mikołajczyk, E. Mikołajewska, H. F. N. Al-Shuka, T. Malinowski, A. Kłodowski, D. Y. Pi menov, T. P ˛aczkowski, F. Hu, K. Giasin, D. Mikołajewski, and M. Macko, “Recent advances in bipedal walking robots: Review of gait, drive, sensors and control systems,” *Sensors*, vol. 22, no. 12, p. 4440, 2022\. 

\[13\] G. Picardi, A. Astolfi, D. Chatzievangelou, J. Aguzzi, and M. Calisti, “Underwater legged robotics: Review and perspectives,” *Bioinspiration & Biomimetics*, vol. 18, no. 3, p. 031001, 2023\. 

\[14\] Y. Gong, G. Sun, A. Nair, A. Bidwai, R. CS, J. Grezmak, G. Sartoretti, and K. A. Daltorio, “Legged robots for object manipulation: A review,” *Frontiers in Mechanical Engineering*, vol. 9, p. 1142421, 2023\. 

\[15\] Y. Zhao, J. Wang, G. Cao, Y. Yuan, X. Yao, and L. Qi, “Intelligent control of multilegged robot smooth motion: a review,” *IEEE Access*, vol. 11, pp. 86645–86685, 2023\. 

21  
\[16\] S. Ha, J. Lee, M. van de Panne, Z. Xie, W. Yu, and M. Khadiv, “Learning-based legged locomo tion: State of the art and future perspectives,” *The International Journal of Robotics Research*, vol. 44, no. 8, pp. 1396–1427, 2025\. 

\[17\] P. M. Wensing, M. Posa, Y. Hu, A. Escande, N. Mansard, and A. Del Prete, “Optimization-based control for dynamic legged robots,” *IEEE Transactions on Robotics*, vol. 40, pp. 43–63, 2023\. 

\[18\] Z. Gu, J. Li, W. Shen, W. Yu, Z. Xie, S. McCrory, X. Cheng, A. Shamsah, R. Griffin, C. K. Liu, A. Kheddar, X. B. Peng, Y. Zhu, G. Shi, Q. Nguyen, G. Cheng, H. Gao, and Y. Zhao, “Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning,” *arXiv:2501.02116*, 2025\. 

\[19\] M. H. Raibert, *Legged robots that balance*. MIT press, 1986\. 

\[20\] Y. Sakagami, R. Watanabe, C. Aoyama, S. Matsunaga, N. Higaki, and K. Fujimura, “The intelli gent asimo: System overview and integration,” in *IEEE/RSJ international conference on intelli gent robots and systems*, vol. 3, pp. 2478–2483, 2002\. 

\[21\] K. Sreenath, H.-W. Park, and I. Poulakakis, “A compliant hybrid zero dynamics controller for sta ble, efficient and fast bipedal walking on mabel,” *The International Journal of Robotics Research*, vol. 30, no. 9, pp. 1170–1193, 2011\. 

\[22\] S. Rezazadeh, C. Hubicki, M. Jones, A. Peekema, J. Van Why, A. Abate, and J. Hurst, “Spring mass walking with atrias in 3d: Robust gait control spanning zero to 4.3 kph on a heavily underac tuated bipedal robot,” in *Dynamic Systems and Control Conference*, vol. 57243, p. V001T04A003, 2015\. 

\[23\] A. Hereid, C. M. Hubicki, E. A. Cousineau, and A. D. Ames, “Dynamic humanoid locomotion: A scalable formulation for hzd gait optimization,” *IEEE Transactions on Robotics*, vol. 34, no. 2, pp. 370–387, 2018\. 

\[24\] W.-L. Ma, S. Kolathaya, E. R. Ambrose, C. M. Hubicki, and A. D. Ames, “Bipedal robotic running with durus-2d: Bridging the gap between theory and experiment,” in *International con ference on hybrid systems: computation and control*, pp. 265–274, 2017\. 

\[25\] S. Feng, E. Whitman, X. Xinjilefu, and C. G. Atkeson, “Optimization-based full body control for the darpa robotics challenge,” *Journal of field robotics*, vol. 32, no. 2, pp. 293–312, 2015\. 

\[26\] E. Krotkov, D. Hackett, L. Jackel, M. Perschbacher, J. Pippine, J. Strauss, G. Pratt, and C. Or lowski, “The darpa robotics challenge finals: Results and perspectives,” in *The DARPA robotics challenge finals: Humanoid robots to the rescue*, pp. 1–26, 2018\. 

\[27\] M. Johnson, B. Shrewsbury, S. Bertrand, T. Wu, D. Duran, M. Floyd, P. Abeles, D. Stephen, N. Mertins, A. Lesman, J. Carff, W. Rifenburgh, P. Kaveti, W. Straatman, J. Smith, M. Griffioen, B. Layton, T. De Boer, T. Koolen, and J. Pratt, “Team IHMC’s lessons learned from the darpa robotics challenge trials,” *Journal of Field Robotics*, vol. 32, no. 2, pp. 192–208, 2015\. 

\[28\] R. Today, ““recent progress on atlas, the world’s most dynamic humanoid robot” \- scott kuinder sma,” June 2020\. 

\[29\] S. Kuindersma, R. Deits, M. Fallon, A. Valenzuela, H. Dai, F. Permenter, T. Koolen, P. Marion, and R. Tedrake, “Optimization-based locomotion planning, estimation, and control design for the atlas humanoid robot,” *Autonomous robots*, vol. 40, no. 3, pp. 429–455, 2016\. 

22  
\[30\] J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, “Dynamic locomotion in the mit cheetah 3 through convex model-predictive control,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 1–9, 2018\. 

\[31\] A. J. Ijspeert, “Biorobotics: Using robots to emulate and investigate agile locomotion,” *Science*, vol. 346, no. 6206, pp. 196–203, 2014\. 

\[32\] C. Li and F. Qian, “Swift progress for robots over complex terrain,” *Nature*, vol. 616, p. 252–253, Mar. 2023\. 

\[33\] S. Kajita, F. Kanehiro, K. Kaneko, K. Yokoi, and H. Hirukawa, “The 3d linear inverted pendulum mode: A simple modeling for a biped walking pattern generation,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 239–246, 2001\. 

\[34\] C. Hubicki, J. Grimes, M. Jones, D. Renjewski, A. Spröwitz, A. Abate, and J. Hurst, “Atrias: Design and validation of a tether-free 3d-capable spring-mass bipedal robot,” *The International Journal of Robotics Research*, vol. 35, no. 12, pp. 1497–1521, 2016\. 

\[35\] R. Batke, F. Yu, J. Dao, J. Hurst, R. L. Hatton, A. Fern, and K. Green, “Optimizing bipedal maneuvers of single rigid-body models for reinforcement learning,” in *IEEE-RAS International Conference on Humanoid Robots*, pp. 714–721, 2022\. 

\[36\] E. R. Westervelt, J. W. Grizzle, C. Chevallereau, J. H. Choi, and B. Morris, *Feedback Control of Dynamic Bipedal Robot Locomotion*. CRC Press, 2007\. 

\[37\] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 5026–5033, 2012\. 

\[38\] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, “Isaac gym: High performance gpu-based physics simulation for robot learning,” *arXiv:2108.10470*, 2021\. 

\[39\] P. Roth, J. Frey, C. Cadena, and M. Hutter, “Learned perceptive forward dynamics model for safe and platform-aware robotic navigation,” *arXiv:2504.19322*, 2025\. 

\[40\] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar, “Deep dynamics models for learning dex terous manipulation,” in *Conference on Robot Learning*, pp. 1101–1112, 2020\. 

\[41\] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” *arXiv:1912.01603*, 2019\. 

\[42\] Y. Matsuo, Y. LeCun, M. Sahani, D. Precup, D. Silver, M. Sugiyama, E. Uchibe, and J. Morimoto, “Deep learning, reinforcement learning, and world models,” *Neural Networks*, vol. 152, pp. 267– 275, 2022\. 

\[43\] P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg, “Daydreamer: World models for physical robot learning,” in *Conference on Robot Learning*, pp. 2226–2240, 2023\. 

\[44\] M. Q. Ali, A. Sridhar, S. Matiana, A. Wong, and M. Al-Sharman, “Humanoid world models: Open world foundation models for humanoid robotics,” *arXiv:2506.01182*, 2025\. 

\[45\] J. Hwangbo, J. Lee, A. Dosovitskiy, C. D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, “Learning agile and dynamic motor skills for legged robots,” *Science Robotics*, vol. 4, no. 26, p. eaau5872, 2019\. 

23  
\[46\] A. H. Chang, C. M. Hubicki, J. J. Aguilar, D. I. Goldman, A. D. Ames, and P. A. Vela, “Learning to jump in granular media: Unifying optimal control synthesis with gaussian process-based re gression,” in *IEEE International Conference on Robotics and Automation*, pp. 2154–2160, 2017\. 

\[47\] W.-S. Yang, W.-C. Lu, and P.-C. Lin, “Legged robot running using a physics-data hybrid motion template,” *IEEE Transactions on Robotics*, vol. 37, no. 5, pp. 1680–1695, 2021\. 

\[48\] T. He, J. Gao, W. Xiao, Y. Zhang, Z. Wang, J. Wang, Z. Luo, G. He, N. Sobanbab, C. Pan, Z. Yi, G. Qu, K. Kitani, J. Hodgins, L. J. Fan, Y. Zhu, C. Liu, and G. Shi, “Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills,” *arXiv:2502.01143*, 2025\. 

\[49\] J. P. Reher, A. Hereid, S. Kolathaya, C. M. Hubicki, and A. D. Ames, “Algorithmic foundations of realizing multi-contact locomotion on the humanoid robot durus,” in *Algorithmic Foundations of Robotics XII: Proceedings of the Twelfth Workshop on the Algorithmic Foundations of Robotics*, pp. 400–415, Springer, 2020\. 

\[50\] R. J. Full and D. E. Koditschek, “Templates and anchors: neuromechanical hypotheses of legged locomotion on land,” *Journal of experimental biology*, vol. 202, no. 23, pp. 3325–3332, 1999\. 

\[51\] T. McGeer, “Passive dynamic walking,” *The International Journal of Robotics Research*, vol. 9, no. 2, pp. 62–82, 1990\. 

\[52\] D. E. Orin, A. Goswami, and S.-H. Lee, “Centroidal dynamics of a humanoid robot,” *Autonomous robots*, vol. 35, no. 2, pp. 161–176, 2013\. 

\[53\] H. Audren and A. Kheddar, “3-d robust stability polyhedron in multicontact,” *IEEE Transactions on Robotics*, vol. 34, no. 2, pp. 388–403, 2018\. 

\[54\] X. Xiong and A. Ames, “3-D Underactuated Bipedal Walking via H-LIP Based Gait Synthesis and Stepping Stabilization,” *IEEE Transactions on Robotics*, vol. 38, no. 4, pp. 2405–2425, 2022\. 

\[55\] Y. Gong and J. W. Grizzle, “Zero dynamics, pendulum models, and angular momentum in feed back control of bipedal locomotion,” *Journal of Dynamic Systems, Measurement, and Control*, vol. 144, no. 12, p. 121006, 2022\. 

\[56\] Y. Gao, V. Paredes, Y. Gong, Z. He, A. Hereid, and Y. Gu, “Time-varying foot placement control for humanoid walking on swaying rigid surface,” *IEEE Transactions on Robotics*, 2025\. 

\[57\] G. Mesesan, R. Schuller, J. Englsberger, C. Ott, and A. Albu-Schäffer, “Unified motion planner for walking, running, and jumping using the three-dimensional divergent component of motion,” *IEEE Transactions on Robotics*, vol. 39, no. 6, pp. 4443–4463, 2023\. 

\[58\] O. Dosunmu-Ogunbi, A. Shrivastava, G. Gibson, and J. W. Grizzle, “Stair climbing using the angular momentum linear inverted pendulum model and model predictive control,” in *Proc. IEEE Int. Conf. Intel. Rob. Syst.*, pp. 8558–8565, 2023\. 

\[59\] J. Stewart, I.-C. Chang, Y. Gu, and P. A. Ioannou, “Adaptive ankle torque control for bipedal hu manoid walking on surfaces with unknown horizontal and vertical motion,” in *American Control Conference*, pp. 4647–4652, 2025\. 

\[60\] X. Xiong, A. D. Ames, and D. I. Goldman, “A stability region criterion for flat-footed bipedal walking on deformable granular terrain,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 4552–4559, 2017\. 

24  
\[61\] A. J. Ijspeert, “Central pattern generators for locomotion control in animals and robots: a review,” *Neural Networks*, vol. 21, no. 4, pp. 642–653, 2008\. 

\[62\] B. Han, H. Yi, Z. Xu, X. Yang, and X. Luo, “3d-slip model based dynamic stability strategy for legged robots with impact disturbance rejection,” *Scientific Reports*, vol. 12, no. 1, p. 5892, 2022\. 

\[63\] J. Hwangbo, J. Lee, and M. Hutter, “Per-contact iteration method for solving contact dynamics,” *IEEE Robotics and Automation Letters*, vol. 3, no. 2, pp. 895–902, 2018\. 

\[64\] E. Coumans and Y. Bai, “Pybullet, a python module for physics simulation for games, robotics and machine learning,” 2016\. 

\[65\] M. Vukobratovic and D. Stoki ´ c, “Dynamic control of unstable locomotion robots,” ´ *Mathematical Biosciences*, vol. 24, no. 1-2, pp. 129–157, 1975\. 

\[66\] S. Kajita, F. Kanehiro, K. Kaneko, K. Fujiwara, K. Harada, K. Yokoi, and H. Hirukawa, “Biped walking pattern generation by using preview control of the zero-moment point,” in *IEEE Interna tional Conference on Robotics and Automation*, vol. 2, pp. 1620–1626, 2003\. 

\[67\] J. Pratt, J. Carff, S. Drakunov, and A. Goswami, “Capture point: A step toward humanoid push recovery,” in *IEEE-RAS International Conference on Humanoid Robots*, pp. 200–207, 2006\. 

\[68\] X. Xiong and A. D. Ames, “Dynamic and versatile humanoid walking via embedding 3d actuated slip model with hybrid lip based stepping,” *IEEE Robotics and Automation Letters*, vol. 5, no. 4, pp. 6286–6293, 2020\. 

\[69\] M. Mistry, J. Buchli, and S. Schaal, “Inverse dynamics control of floating base systems us ing orthogonal decomposition,” in *IEEE international conference on robotics and automation*, pp. 3406–3412, 2010\. 

\[70\] J. Reher and A. D. Ames, “Inverse dynamics control of compliant hybrid zero dynamic walking,” in *IEEE International Conference on Robotics and Automation*, pp. 2040–2047, 2021\. 

\[71\] H. Sadeghian, C. Ott, G. Garofalo, and G. Cheng, “Passivity-based control of underactuated biped robots within hybrid zero dynamics approach,” in *IEEE International Conference on Robotics and Automation*, pp. 4096–4101, 2017\. 

\[72\] I. Radosavovic, T. Xiao, B. Zhang, T. Darrell, J. Malik, and K. Sreenath, “Real-world humanoid locomotion with reinforcement learning,” *Science Robotics*, vol. 9, no. 89, p. eadi9579, 2024\. 

\[73\] E. Westervelt, J. Grizzle, and D. Koditschek, “Hybrid zero dynamics of planar biped walkers,” *IEEE Transactions on Automatic Control*, vol. 48, pp. 42–56, Jan. 2003\. 

\[74\] A. D. Ames, K. Galloway, K. Sreenath, and J. W. Grizzle, “Rapidly exponentially stabilizing control lyapunov functions and hybrid zero dynamics,” *IEEE Transactions on Automatic Control*, vol. 59, no. 4, pp. 876–891, 2014\. 

\[75\] K. Galloway, K. Sreenath, A. D. Ames, and J. W. Grizzle, “Torque saturation in bipedal robotic walking through control lyapunov function-based quadratic programs,” *IEEE Access*, vol. 3, pp. 323–332, 2015\. 

\[76\] C. Dario Bellicoso, F. Jenelten, P. Fankhauser, C. Gehring, J. Hwangbo, and M. Hutter, “Dy namic locomotion and whole-body control for quadrupedal robots,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 3359–3365, Sept. 2017\. 

25  
\[77\] A. W. Winkler, C. D. Bellicoso, M. Hutter, and J. Buchli, “Gait and trajectory optimization for legged systems through phase-based end-effector parameterization,” *IEEE Robotics and Automa tion Letters*, vol. 3, no. 3, pp. 1560–1567, 2018\. 

\[78\] K. Bouyarmane, S. Caron, A. Escande, and A. Kheddar, “Multi-contact motion planning and control,” in *Humanoid Robotics: A Reference*, pp. 1763–1804, Springer, 2019\. 

\[79\] A. D. Ames, “Human-inspired control of bipedal walking robots,” *IEEE Transactions on Auto matic Control*, vol. 59, no. 5, pp. 1115–1130, 2014\. 

\[80\] C. Khazoom, S. Hong, M. Chignoli, E. Stanger-Jones, and S. Kim, “Tailoring solution accuracy for fast whole-body model predictive control of legged robots,” *IEEE Robotics and Automation Letters*, 2024\. 

\[81\] A. B. Ghansah, S. A. Esteban, and A. D. Ames, “Hierarchical reduced-order model predictive control for robust locomotion on humanoid robots,” in *International Conference on Humanoid Robots*, pp. 1–8, 2025\. 

\[82\] G. Romualdi, S. Dafarra, G. L’Erario, I. Sorrentino, S. Traversaro, and D. Pucci, “Online non linear centroidal mpc for humanoid robot locomotion with step adjustment,” in *International Conference on Robotics and Automation*, pp. 10412–10419, 2022\. 

\[83\] R. Grandia, F. Jenelten, S. Yang, F. Farshidian, and M. Hutter, “Perceptive locomotion through nonlinear model-predictive control,” *IEEE Transactions on Robotics*, vol. 39, no. 5, pp. 3402– 3421, 2023\. 

\[84\] G. Kim, D. Kang, J.-H. Kim, S. Hong, and H.-W. Park, “Contact-implicit model predictive con trol: Controlling diverse quadruped motions without pre-planned contact modes or trajectories,” *The International Journal of Robotics Research*, vol. 44, no. 3, pp. 486–510, 2025\. 

\[85\] S. Le Cleac’h, T. A. Howell, S. Yang, C.-Y. Lee, J. Zhang, A. Bishop, M. Schwager, and Z. Manchester, “Fast contact-implicit model-predictive control,” *IEEE Transactions on Robotics*, vol. 40, pp. 1617–1629, 2024\. 

\[86\] V. Kurtz, A. Castro, A. Ö. Önol, and H. Lin, “Inverse dynamics trajectory optimization for contact-implicit model predictive control,” *International Journal of Robotics Research*, vol. 44, pp. 320–348, Mar. 2025\. 

\[87\] N. J. Kong, C. Li, G. Council, and A. M. Johnson, “Hybrid iLQR model predictive control for contact implicit stabilization on legged robots,” *IEEE Transactions on Robotics*, vol. 39, no. 6, pp. 4712–4727, 2023\. 

\[88\] H. Xue, C. Pan, Z. Yi, G. Qu, and G. Shi, “Full-order sampling-based mpc for torque-level loco motion control via diffusion-style annealing,” *arXiv:2409.15610*, 2024\. 

\[89\] T. Howell, N. Gileadi, S. Tunyasuvunakool, K. Zakka, T. Erez, and Y. Tassa, “Predictive sam pling: Real-time behaviour synthesis with mujoco,” *arXiv:2212.00541*, 2022\. 

\[90\] J. Alvarez-Padilla, J. Z. Zhang, S. Kwok, J. M. Dolan, and Z. Manchester, “Real-time whole body control of legged robots with model-predictive path integral control,” in *IEEE International Conference on Robotics and Automation*, pp. 14721–14727, 2025\. 

\[91\] K. Bouyarmane, K. Chappellet, J. Vaillant, and A. Kheddar, “Quadratic programming for mul tirobot and task-space force control,” *IEEE Transactions on Robotics*, vol. 35, no. 1, pp. 64–77, 2018\. 

26  
\[92\] R. Deits and R. Tedrake, “Footstep planning on uneven terrain with mixed-integer convex opti mization,” in *IEEE-RAS international conference on humanoid robots*, pp. 279–286, 2014\. 

\[93\] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in *International Conference on Machine Learning*, 2015\. 

\[94\] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” *arXiv:1707.06347*, 2017\. 

\[95\] Z. Xie, H. Y. Ling, N. H. Kim, and M. Van De Panne, “Allsteps: curriculum-driven learning of stepping stone skills,” in *Computer Graphics Forum*, vol. 39, pp. 213–224, 2020\. 

\[96\] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning robust perceptive locomotion for quadrupedal robots in the wild,” *Science robotics*, vol. 7, no. 62, p. eabk2822, 2022\. 

\[97\] T. Haarnoja, B. Moran, G. Lever, S. H. Huang, D. Tirumala, J. Humplik, M. Wulfmeier, S. Tun yasuvunakool, N. Y. Siegel, R. Hafner, M. Bloesch, K. Hartikainen, A. Byravan, L. Hasenclever, Y. Tassa, F. Sadeghi, N. Batchelor, F. Casarini, S. Saliceti, C. Game, N. Sreendra, K. Patel, M. Gwira, A. Huber, N. Hurley, F. Nori, R. Hadsell, and N. Heess, “Learning agile soccer skills for a bipedal robot with deep reinforcement learning,” *Science Robotics*, vol. 9, no. 89, p. eadi8022, 2024\. 

\[98\] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne, “Deepmimic: Example-guided deep re inforcement learning of physics-based character skills,” *ACM Transactions on Graphics*, vol. 37, no. 4, pp. 143:1–143:14, 2018\. 

\[99\] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. M. Kitani, C. Liu, and G. Shi, “Om nih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning,” in *Conference on Robot Learning*, 2024\. 

\[100\] T. E. Truong, Q. Liao, X. Huang, G. Tevet, C. K. Liu, and K. Sreenath, “Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion,” *arXiv:2508.08241*, 2025\. 

\[101\] K. Green, Y. Godse, J. Dao, R. L. Hatton, A. Fern, and J. Hurst, “Learning spring mass loco motion: Guiding policies with a reduced-order model,” *IEEE Robotics and Automation Letters*, vol. 6, no. 2, pp. 3926–3932, 2021\. 

\[102\] Z. Li, X. Cheng, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Reinforcement Learning for Robust Parameterized Locomotion Control of Bipedal Robots,” in *IEEE Interna tional Conference on Robotics and Automation*, pp. 2811–2817, 2021\. 

\[103\] F. Liu, Z. Gu, Y. Cai, Z. Zhou, H. Jung, J. Jang, S. Zhao, S. Ha, Y. Chen, D. Xu, and Y. Zhao, “Opt2skill: Imitating dynamically-feasible whole-body trajectories for versatile humanoid loco manipulation,” *IEEE Robotics and Automation Letters*, 2025\. 

\[104\] H. J. Lee, S. Hong, and S. Kim, “Integrating model-based footstep planning with model-free reinforcement learning for dynamic legged locomotion,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 11248–11255, 2024\. 

\[105\] K. Li, Z. Olkin, Y. Yue, and A. D. Ames, “Clf-rl: Control lyapunov function guided reinforcement learning,” *arXiv:2508.09354*, 2025\. 

\[106\] Y.-J. Wang, B. Zhang, J. Chen, and K. Sreenath, “Prompt a robot to walk with large language models,” in *IEEE Conference on Decision and Control*, pp. 1531–1538, 2024\. 

27  
\[107\] Y. Shao, X. Huang, B. Zhang, Q. Liao, Y. Gao, Y. Chi, Z. Li, S. Shao, and K. Sreenath, “Langwbc: Language-directed humanoid whole-body control via end-to-end learning,” *arXiv:2504.21738*, 2025\. 

\[108\] Y. Guo, Y.-J. Wang, L. Zha, and J. Chen, “Doremi: Grounding language model by detecting and recovering from plan-execution misalignment,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 12124–12131, 2024\. 

\[109\] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano, “Human motion diffusion model,” in *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023\. 

\[110\] X. Gu, Y.-J. Wang, X. Zhu, C. Shi, Y. Guo, Y. Liu, and J. Chen, “Advancing humanoid locomo tion: Mastering challenging terrains with denoising world model learning,” in *Robotics: Science and Systems (RSS)*, 2024\. 

\[111\] H. Liu, Y. Gao, S. Teng, Y. Chi, Y. S. Shao, Z. Li, M. Ghaffari, and K. Sreenath, “Ego-vision world model for humanoid contact planning,” *arXiv:2510.11682*, 2025\. 

\[112\] A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike adaptive elements that can solve diffi cult learning control problems,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC 13, no. 5, pp. 834–846, 1983\. 

\[113\] C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8, no. 3–4, pp. 279–292, 1992\. 

\[114\] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” *Machine Learning*, vol. 8, no. 3–4, pp. 229–256, 1992\. 

\[115\] D. P. Bertsekas and J. N. Tsitsiklis, *Neuro-Dynamic Programming*. Athena Scientific, 1996\. 

\[116\] R. S. Sutton and A. G. Barto, *Reinforcement learning: An introduction*, vol. 1\. MIT press Cam bridge, 1998\. 

\[117\] J. N. Tsitsiklis and B. Van Roy, “An analysis of temporal-difference learning with function ap proximation,” *IEEE Transactions on Automatic Control*, vol. 42, no. 5, pp. 674–690, 1997\. 

\[118\] L. C. Baird, “Residual algorithms: Reinforcement learning with function approximation,” in *In ternational Conference on Machine Learning*, pp. 30–37, 1995\. 

\[119\] J. Kober, J. A. Bagnell, and J. Peters, “Reinforcement learning in robotics: A survey,” *The Inter national Journal of Robotics Research*, vol. 32, no. 11, pp. 1238–1274, 2013\. 

\[120\] M. P. Deisenroth, G. Neumann, and J. Peters, “A survey on policy search for robotics,” *Founda tions and Trends in Robotics*, vol. 2, no. 1–2, pp. 1–142, 2013\. 

\[121\] G. Tesauro, “Temporal difference learning and td-gammon,” *Communications of the ACM*, vol. 38, no. 3, pp. 58–68, 1995\. 

\[122\] L.-J. Lin, “Self-improving reactive agents based on reinforcement learning, planning and teach ing,” *Machine Learning*, vol. 8, no. 3–4, pp. 293–321, 1992\. 

\[123\] M. Riedmiller, “Neural fitted q iteration—first experiences with a data efficient neural reinforce ment learning method,” in *European Conference on Machine Learning*, pp. 317–328, 2005\. 

28  
\[124\] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in *Advances in Neural Information Processing Systems*, vol. 25, pp. 1097–1105, 2012\. 

\[125\] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, vol. 521, pp. 436–444, 2015\. 

\[126\] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried miller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep rein forcement learning,” *Nature*, vol. 518, no. 7540, pp. 529–533, 2015\. 

\[127\] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” *arXiv:1312.5602*, 2013\. 

\[128\] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbren ner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, “Mas tering the game of go with deep neural networks and tree search,” *Nature*, vol. 529, no. 7587, pp. 484–489, 2016\. 

\[129\] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021\. 

\[130\] E. Liang, R. Liaw, R. Nishihara, P. Moritz, R. Fox, K. Goldberg, J. Gonzalez, M. Jordan, and I. Stoica, “RLlib: Abstractions for distributed reinforcement learning,” in *International Confer ence on Machine Learning*, pp. 3053–3062, 2018\. 

\[131\] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes using massively parallel deep reinforcement learning,” in *Conference on Robot Learning*, pp. 91–100, 2022\. 

\[132\] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, and V. Vanhoucke, “Sim to-real: Learning agile locomotion for quadruped robots,” *Robotics: Science and Systems XIV*, 2018\. 

\[133\] X. B. Peng, E. Coumans, T. Zhang, T.-W. Lee, J. Tan, and S. Levine, “Learning agile robotic locomotion skills by imitating animals,” *arXiv:2004.00784*, 2020\. 

\[134\] W. Yu, V. C. Kumar, G. Turk, and C. K. Liu, “Sim-to-real transfer for biped locomotion,” in *international conference on intelligent robots and systems*, pp. 3503–3510, 2019\. 

\[135\] Z. Xie, P. Clary, J. Dao, P. Morais, J. Hurst, and M. van de Panne, “Learning locomotion skills for cassie: Iterative design and sim-to-real,” in *Conference on Robot Learning*, pp. 317–329, 2020\. 

\[136\] H. Duan, A. Malik, J. Dao, A. Saxena, K. Green, J. Siekmann, A. Fern, and J. Hurst, “Sim-to real learning of footstep-constrained bipedal dynamic walking,” in *International Conference on Robotics and Automation*, pp. 10428–10434, 2022\. 

\[137\] F. Shi, Y. Kojio, T. Makabe, T. Anzai, K. Kojima, K. Okada, and M. Inaba, “Reference-free learning bipedal motor skills via assistive force curricula,” in *The International Symposium of Robotics Research*, pp. 304–320, Springer, 2022\. 

\[138\] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “Amp: Adversarial motion priors for stylized physics-based character control,” *ACM Transactions on Graphics (ToG)*, vol. 40, no. 4, pp. 1–20, 2021\. 

29  
\[139\] T. He, Z. Luo, W. Xiao, C. Zhang, K. Kitani, C. Liu, and G. Shi, “Learning human-to-humanoid real-time whole-body teleoperation,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 8944–8951, 2024\. 

\[140\] M. Ji, X. Peng, F. Liu, J. Li, G. Yang, X. Cheng, and X. Wang, “Exbody2: Advanced expressive humanoid whole-body control,” *arXiv:2412.13196*, 2024\. 

\[141\] Z. Su, B. Zhang, N. Rahmanian, Y. Gao, Q. Liao, C. Regan, K. Sreenath, and S. S. Sastry, “Hitter: A humanoid table tennis robot via hierarchical planning and learning,” *arXiv:2508.21043*, 2025\. 

\[142\] M. Hu, W. Chen, W. Li, F. Mandali, Z. He, R. Zhang, P. Krisna, K. Christian, L. Benaharon, D. Ma, K. Ramani, and Y. Gu, “Towards versatile humanoid table tennis: Unified reinforcement learning with prediction augmentation,” *arXiv:2509.21690*, 2025\. 

\[143\] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” *Artificial intelligence*, vol. 101, no. 1-2, pp. 99–134, 1998\. 

\[144\] K. Zakka, B. Tabanpour, Q. Liao, M. Haiderbhai, S. Holt, J. Y. Luo, A. Allshire, E. Frey, K. Sreenath, L. A. Kahrs, C. Sferrazza, Y. Tassa, and P. Abbeel, “Mujoco playground,” *arXiv:2502.08844*, 2025\. 

\[145\] A. Hill, A. Raffin, M. Ernestus, A. Gleave, A. Kanervisto, R. Traore, P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu, “Stable base lines.” https://github.com/hill-a/stable-baselines, 2018\. 

\[146\] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” *arXiv:1606.01540*, 2016\. 

\[147\] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 23–30, 2017\. 

\[148\] Z. Xie, X. Da, M. Van de Panne, B. Babich, and A. Garg, “Dynamics randomization revisited: A case study for quadrupedal locomotion,” in *IEEE International Conference on Robotics and Automation*, pp. 4955–4961, 2021\. 

\[149\] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in *International Conference on Machine Learning*, pp. 41–48, 2009\. 

\[150\] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,” *Science robotics*, vol. 5, no. 47, p. eabc5986, 2020\. 

\[151\] D. Chen, B. Zhou, V. Koltun, and P. Krähenbühl, “Learning by cheating,” in *Conference on Robot Learning*, pp. 66–75, 2020\. 

\[152\] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, “Asymmetric actor critic for image-based robot learning,” in *Proceedings of Robotics: Science and Systems*, June 2018\. 

\[153\] Y. Li, Y. Zhang, W. Xiao, C. Pan, H. Weng, G. He, T. He, and G. Shi, “Hold my beer: Learn ing gentle humanoid locomotion and end-effector stabilization control,” in *Conference on Robot Learning*, 2025\. 

\[154\] Y. Xue, W. Dong, M. Liu^, W. Zhang, and J. Pang, “A unified and general humanoid whole-body controller for versatile locomotion,” *arXiv:2502.03206*, 2025\. 

30  
\[155\] H. Zhang, L. Zhang, Z. Chen, L. Chen, Y. Wang, and R. Xiong, “Natural humanoid robot loco motion with generative motion prior,” *arXiv:2503.09015*, 2025\. 

\[156\] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “Amass: Archive of motion capture as surface shapes,” in *IEEE/CVF international conference on computer vision*, pp. 5442–5451, 2019\. 

\[157\] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu, “Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments,” *IEEE transactions on pattern analysis and machine intelligence*, vol. 36, no. 7, pp. 1325–1339, 2013\. 

\[158\] Carnegie Mellon University Graphics Lab, “Cmu graphics lab motion capture database.” http: //mocap.cs.cmu.edu/. 

\[159\] C. Mandery, O. Terlemez, M. Do, N. Vahrenkamp, and T. Asfour, “Unifying representations and large–scale whole-body motion databases for studying human motion,” *IEEE Transactions on Robotics*, vol. 32, no. 4, pp. 796–809, 2016\. 

\[160\] S. Ghorbani, K. Mahdaviani, A. Thaler, K. Kording, D. J. Cook, G. Blohm, and N. F. Troje, “Movi: A large multipurpose motion and video dataset,” *arXiv:2003.01888*, 2020\. 

\[161\] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3d human motions from text,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5152–5161, June 2022\. 

\[162\] J. Li, J. Wu, and C. K. Liu, “Object motion guided human motion synthesis,” *ACM Transactions on Graphics (TOG)*, vol. 42, no. 6, pp. 1–11, 2023\. 

\[163\] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, “Motion-x: A large-scale 3d expressive whole-body human motion dataset,” *Advances in Neural Information Processing Systems*, vol. 36, pp. 25268–25280, 2023\. 

\[164\] Z. Luo, J. Cao, K. Kitani, and W. Xu, “Perpetual humanoid control for real-time simu lated avatars,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 10895–10904, 2023\. 

\[165\] L. Penco, B. Clément, V. Modugno, E. M. Hoffman, G. Nava, D. Pucci, N. G. Tsagarakis, J.- B. Mouret, and S. Ivaldi, “Robust real-time whole-body motion retargeting from human to hu manoid,” in *IEEE-RAS 18th International Conference on Humanoid Robots*, pp. 425–432, 2018\. 

\[166\] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi, “Omniretarget: Interaction-preserving data generation for humanoid whole-body loco manipulation and scene interaction,” *arXiv:2509.26633*, 2025\. 

\[167\] J. P. Araujo, Y. Ze, P. Xu, J. Wu, and C. K. Liu, “Retargeting matters: General motion retargeting for humanoid motion tracking,” *arXiv:2510.02252*, 2025\. 

\[168\] J. Merel, Y. Tassa, D. TB, S. Srinivasan, J. Lemmon, Z. Wang, G. Wayne, and N. Heess, “Learning human behaviors from motion capture by adversarial imitation,” *arXiv:1707.02201*, 2017\. 

\[169\] A. Tang, T. Hiraoka, N. Hiraoka, F. Shi, K. Kawaharazuka, K. Kojima, K. Okada, and M. Inaba, “Humanmimic: Learning natural locomotion and transitions for humanoid robot via wasserstein adversarial imitation,” in *IEEE International Conference on Robotics and Automation*, pp. 13107– 13114, 2024\. 

31  
\[170\] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “Humanplus: Humanoid shadowing and imitation from humans,” in *Conference on Robot Learning*, pp. 2828–2844, 2025\. 

\[171\] X. Cheng, Y. Ji, J. Chen, R. Yang, G. Yang, and X. Wang, “Expressive whole-body control for humanoid robots,” *arXiv:2402.16796*, 2024\. 

\[172\] Z. Chen, M. Ji, X. Cheng, X. Peng, X. B. Peng, and X. Wang, “Gmt: General motion tracking for humanoid whole-body control,” *arXiv:2506.14770*, 2025\. 

\[173\] Y. Ze, Z. Chen, J. P. AraÃšjo, Z.-a. Cao, X. B. Peng, J. Wu, and C. K. Liu, “Twist: Teleoperated whole-body imitation system,” *arXiv:2505.02833*, 2025\. 

\[174\] K. Yin, W. Zeng, K. Fan, M. Dai, Z. Wang, Q. Zhang, Z. Tian, J. Wang, J. Pang, and W. Zhang, “Unitracker: Learning universal whole-body motion tracker for humanoid robots,” *arXiv:2507.07356*, 2025\. 

\[175\] T. He, W. Xiao, T. Lin, Z. Luo, Z. Xu, Z. Jiang, J. Kautz, C. Liu, G. Shi, X. Wang, L. Fan, and Y. Zhu, “Hover: Versatile neural whole-body controller for humanoid robots,” in *IEEE Interna tional Conference on Robotics and Automation*, pp. 9989–9996, 2025\. 

\[176\] H. Weng, Y. Li, N. Sobanbabu, Z. Wang, Z. Luo, T. He, D. Ramanan, and G. Shi, “Hdmi: Learn ing interactive humanoid whole-body control from human videos,” *arXiv:2509.16757*, 2025\. 

\[177\] S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan, “Resmimic: From general motion tracking to humanoid whole-body loco-manipulation via residual learning,” *arXiv:2510.05070*, 2025\. 

\[178\] N. Sobanbabu, G. He, T. He, Y. Yang, and G. Shi, “Sampling-based system identification with active exploration for legged sim2real learning,” in *9th Annual Conference on Robot Learning*, 2025\. 

\[179\] D. Hoeller, N. Rudin, D. Sako, and M. Hutter, “Anymal parkour: Learning agile navigation for quadrupedal robots,” *Science Robotics*, vol. 9, no. 88, p. eadi7566, 2024\. 

\[180\] Y. Wang, M. Yang, W. Zeng, Y. Zhang, X. Xu, H. Jiang, Z. Ding, and Z. Lu, “From experts to a generalist: Toward general whole-body control for humanoid robots,” *arXiv:2506.12779*, 2025\. 

\[181\] N. Fey, G. B. Margolis, M. Peticco, and P. Agrawal, “Bridging the sim-to-real gap for athletic loco-manipulation,” *arXiv:2502.10894*, 2025\. 

\[182\] Y. Zhang, Y. Yuan, P. Gurunath, T. He, S. Omidshafiei, A. akbar Agha-mohammadi, M. Vazquez Chanlatte, L. Pedersen, and G. Shi, “Falcon: Learning force-adaptive humanoid loco manipulation,” *arXiv:2505.06776*, 2025\. 

\[183\] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in *International Conference on Machine Learning*, pp. 2555– 2565, 2019\. 

\[184\] G. A. Castillo, B. Weng, W. Zhang, and A. Hereid, “Data-driven latent space representation for robust bipedal locomotion learning,” in *IEEE International Conference on Robotics and Automa tion*, pp. 1172–1178, 2024\. 

\[185\] X. Huang, T. Truong, Y. Zhang, F. Yu, J. P. Sleiman, J. Hodgins, K. Sreenath, and F. Farshidian, “Diffuse-cloc: Guided diffusion for physics-based character look-ahead control,” *ACM Transac tions on Graphics*, vol. 44, no. 4, pp. 1–12, 2025\. 

32  
\[186\] H. Xue, X. Huang, D. Niu, Q. Liao, T. Kragerud, J. T. Gravdahl, X. B. Peng, G. Shi, T. Darrell, K. Sreenath, and S. Sastry, “Leverb: Humanoid whole-body control with latent vision-language instruction,” *arXiv:2506.13751*, 2025\. 

\[187\] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, Q. Vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, J. Singh, P. Sermanet, P. R. Sanketi, G. Salazar, M. S. Ryoo, K. Reymann, K. Rao, K. Pertsch, I. Mordatch, H. Michalewski, Y. Lu, S. Levine, L. Lee, T.-W. E. Lee, I. Leal, Y. Kuang, D. Kalashnikov, R. Julian, N. J. Joshi, A. Irpan, B. Ichter, J. Hsu, A. Herzog, K. Hausman, K. Gopalakrishnan, C. Fu, P. Florence, C. Finn, K. A. Dubey, D. Driess, T. Ding, K. M. Choromanski, X. Chen, Y. Chebotar, J. Carbajal, N. Brown, A. Brohan, M. G. Arenas, and K. Han, “Rt-2: Vision-language-action models transfer web knowl edge to robotic control,” in *Proceedings of The 7th Conference on Robot Learning*, vol. 229 of *Proceedings of Machine Learning Research*, pp. 2165–2183, 06–09 Nov 2023\. 

\[188\] H. Liu, S. Teng, B. Liu, W. Zhang, and M. Ghaffari, “Discrete-time hybrid automata learning: Legged locomotion meets skateboarding,” *arXiv:2503.01842*, 2025\. 

\[189\] I. Radosavovic, B. Zhang, B. Shi, J. Rajasegaran, S. Kamat, T. Darrell, K. Sreenath, and J. Malik, “Humanoid locomotion as next token prediction,” *Advances in neural information processing systems*, vol. 37, pp. 79307–79324, 2024\. 

\[190\] G. Feng, H. Zhang, Z. Li, X. B. Peng, B. Basireddy, L. Yue, Z. Song, L. Yang, Y. Liu, K. Sreenath, and S. Levine, “Genloco: Generalized locomotion controllers for quadrupedal robots,” in *Confer ence on Robot Learning*, pp. 1893–1903, 2023\. 

\[191\] E. Chane-Sane, P.-A. Leziart, T. Flayols, O. Stasse, P. Souères, and N. Mansard, “Cat: Constraints as terminations for legged locomotion reinforcement learning,” in *IEEE/RSJ International Con ference on Intelligent Robots and Systems*, pp. 13303–13310, 2024\. 

\[192\] Z. Wang, J. J. Hunt, and M. Zhou, “Diffusion policies as an expressive policy class for offline reinforcement learning,” *arXiv:2208.06193*, 2022\. 

\[193\] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakr ishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. J. Ruano, K. Jeffrey, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K.-H. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao, K. Rao, J. Rettinghouse, D. Reyes, P. Sermanet, N. Sievers, C. Tan, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, and A. Zeng, “Do as i can, not as i say: Grounding language in robotic affordances,” *arXiv:2204.01691*, 2022\. 

\[194\] K. Chen, X. Huang, X. Chen, and J. Yi, “Contingency model predictive control for bipedal loco motion on moving surfaces with a linear inverted pendulum model,” in *2024 American Control Conference (ACC)*, pp. 3166–3171, IEEE, 2024\. 

\[195\] Q. Nguyen and K. Sreenath, “Robust safety-critical control for dynamic robotics,” *IEEE Trans actions on Automatic Control*, vol. 67, no. 3, pp. 1073–1088, 2021\. 

\[196\] E. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary, A. Jain, A. Balakrishna, A. Wahid, B. Burgess Limerick, B. Kim, B. Schölkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn, C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak, D. Shah, D. Büchler, D. Jayaraman, D. Kalashnikov, D. Sadigh, E. Johns, 

33  
E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri, F. Stulp, G. Zhou, G. S. Sukhatme, G. Sal hotra, G. Yan, G. Feng, G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H.-S. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta, H. Bharadhwaj, H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider, J. Hsu, J. Vakil, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu, J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang, J. Malik, J. Silvério, J. Hejna, J. Booher, J. Tomp son, J. Yang, J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani, K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh, K.-H. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. Fei-Fei, L. Tan, L. J. Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero, M. Du, M. Ahn, M. C. Yip, M. Zhang, M. Ding, M. Heo, M. K. Srirama, M. Sharma, M. J. Kim, M. Z. Irshad, N. Kanazawa, N. Hansen, N. Heess, N. J. Joshi, N. Suenderhauf, N. Liu, N. D. Palo, N. M. M. Shafiullah, O. Mees, O. Kroemer, O. Bastani, P. R. Sanketi, P. T. Miller, P. Yin, P. Wohlhart, P. Xu, P. D. Fagan, P. Mitrano, P. Sermanet, P. Abbeel, P. Sundaresan, Q. Chen, Q. Vuong, R. Rafailov, R. Tian, R. Doshi, R. Martín-Martín, R. Baijal, R. Scalise, R. Hendrix, R. Lin, R. Qian, R. Zhang, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kir mani, S. Levine, S. Lin, S. Moore, S. Bahl, S. Dass, S. Sonawani, S. Tulsiani, S. Song, S. Xu, S. Haldar, S. Karamcheti, S. Adebola, S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Ra mamoorthy, S. Dasari, S. Belkhale, S. Park, S. Nair, S. Mirchandani, T. Osa, T. Gupta, T. Harada, T. Matsushima, T. Xiao, T. Kollar, T. Yu, T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Dar rell, T. Chung, V. Jain, V. Kumar, V. Vanhoucke, V. Guizilini, W. Zhan, W. Zhou, W. Burgard, X. Chen, X. Chen, X. Wang, X. Zhu, X. Geng, X. Liu, X. Liangwei, X. Li, Y. Pang, Y. Lu, Y. J. Ma, Y. Kim, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Wu, Y. Xu, Y. Wang, Y. Bisk, Y. Dou, Y. Cho, Y. Lee, Y. Cui, Y. Cao, Y.-H. Wu, Y. Tang, Y. Zhu, Y. Zhang, Y. Jiang, Y. Li, Y. Li, Y. Iwasawa, Y. Matsuo, Z. Ma, Z. Xu, Z. J. Cui, Z. Zhang, Z. Fu, and Z. Lin, “Open x-embodiment: Robotic learning datasets and rt-x models,” 2025\. 

\[197\] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world mod els,” *arXiv:2301.04104*, 2023\. 

\[198\] M. Dai, X. Xiong, J. Lee, and A. D. Ames, “Data-driven adaptation for robust bipedal locomotion with step-to-step dynamics,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 8574–8581, 2023\. 

\[199\] A. Kumar, Z. Li, J. Zeng, D. Pathak, K. Sreenath, and J. Malik, “Adapting rapid motor adaptation for bipedal robots,” in *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 1161–1168, 2022\. 

\[200\] K. Zhou and J. C. Doyle, *Essentials of robust control*, vol. 104\. Prentice hall Upper Saddle River, NJ, 1998\. 

\[201\] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,” in *Robotics: Science and Systems*, 2021\. 

\[202\] P. A. Ioannou and J. Sun, *Robust adaptive control*, vol. 1\. PTR Prentice-Hall Upper Saddle River, NJ, 1996\. 

\[203\] I. T. Jolliffe and J. Cadima, “Principal component analysis: A review and recent developments,” *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sci ences*, vol. 374, no. 2065, p. 20150202, 2016\. 

34  
\[204\] G. H. Golub and C. Reinsch, “Singular value decomposition and least squares solutions,” in *Linear algebra*, pp. 134–151, Springer, 1971\. 

\[205\] C. Zhang, W. Xiao, T. He, and G. Shi, “Wococo: Learning whole-body humanoid control with sequential contacts,” in *Conference on Robot Learning*, pp. 455–472, 2025\. 

\[206\] Z. Olkin, K. Li, W. D. Compton, and A. D. Ames, “Chasing stability: Humanoid running via control lyapunov function guided reinforcement learning,” *arXiv:2509.19573*, 2025\. 

\[207\] F. Jenelten, J. He, F. Farshidian, and M. Hutter, “Dtc: Deep tracking control,” *Science Robotics*, vol. 9, no. 86, p. eadh5401, 2024\. 

\[208\] T. He, C. Zhang, W. Xiao, G. He, C. Liu, and G. Shi, “Agile but safe: Learning collision-free high-speed legged locomotion,” *arXiv:2401.17583*, 2024\. 

\[209\] L. Yang, B. Werner, and M. d. S. A. D. Ames, “Cbf-rl: Safety filtering reinforcement learning in training with control barrier functions,” *arXiv:2510.14959*, 2025\. 

\[210\] C. Sferrazza, D.-M. Huang, X. Lin, Y. Lee, and P. Abbeel, “Humanoidbench: Simulated hu manoid benchmark for whole-body locomotion and manipulation,” *arXiv:2403.10506*, 2024\. 

\[211\] P. Huang, X. Zhang, Z. Cao, S. Liu, M. Xu, W. Ding, J. Francis, B. Chen, and D. Zhao, “What went wrong? closing the sim-to-real gap via differentiable causal discovery,” in *Conference on Robot Learning*, pp. 734–760, 2023\. 

\[212\] T. Miki, J. Lee, L. Wellhausen, E. Kaufmann, J. Hwangbo, C. D. Bellicoso, F. Tresoldi, M. Fässler, and M. Hutter, “Perceptive locomotion in rough terrain,” *Science Robotics*, vol. 7, no. 66, p. eabk2822, 2022\. 

\[213\] F. Shi, C. Zhang, T. Miki, J. Lee, M. Hutter, and S. Coros, “Rethinking robustness assessment: Adversarial attacks on learning-based quadrupedal locomotion controllers,” in *Robotics: Science and Systems*, 2024\. 

\[214\] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in *International conference on machine learning*, pp. 22–31, 2017\. 

\[215\] J. K. Wabersich and M. N. Zeilinger, “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems,” *Automatica*, vol. 129, p. 109597, 2021\. 

\[216\] A. D. Ames, S. Xu, J. W. Grizzle, and P. Tabuada, “Control barrier function based quadratic programs for safety critical systems,” *IEEE Transactions on Automatic Control*, vol. 62, no. 8, pp. 3861–3876, 2017\. 

\[217\] A.-C. Cheng, Y. Ji, Z. Yang, Z. Gongye, X. Zou, J. Kautz, E. Bıyık, H. Yin, S. Liu, and X. Wang, “Navila: Legged robot vision-language-action model for navigation,” *arXiv:2412.04453*, 2024\. 

\[218\] R. Firoozi, J. Tucker, S. Tian, A. Majumdar, J. Sun, W. Liu, Y. Zhu, S. Song, A. Kapoor, K. Haus man, B. Ichter, D. Driess, J. Wu, C. Lu, and M. Schwager, “Foundation models in robotics: Ap plications, challenges, and the future,” *The International Journal of Robotics Research*, vol. 44, no. 5, pp. 701–739, 2024\. 

\[219\] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in *International Conference on Machine Learning*, pp. 1861–1870, 2018\. 

35  
\[220\] J. Ho and S. Ermon, “Generative adversarial imitation learning,” in *Advances in Neural Informa tion Processing Systems*, 2016\. 

\[221\] Y. Li, J. Song, and S. Ermon, “Infogail: Interpretable imitation learning from visual demonstra tions,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017\. 

\[222\] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mor datch, “Decision transformer: Reinforcement learning via sequence modeling,” in *Advances in Neural Information Processing Systems*, vol. 34, pp. 15084–15097, 2021\. 

\[223\] M. Janner, Q. C. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” in *Advances in Neural Information Processing Systems*, vol. 34, pp. 1273–1286, 2021\. 

\[224\] M. Janner, G. Varun, A. Ajay, J. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” in *International Conference on Machine Learning*, 2022\. 

\[225\] A. Ajay, Y. Du, A. Gupta, J. Tenenbaum, T. Jaakkola, and P. Agrawal, “Is conditional generative modeling all you need for decision-making?,” *arXiv:2211.15657*, 2022\. 

\[226\] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” *The International Journal of Robotics Research*, vol. 44, no. 10-11, pp. 1684–1704, 2024\. 

\[227\] Y. Yuan, J. Song, U. Iqbal, A. Vahdat, and J. Kautz, “Physdiff: Physics-guided human motion diffusion model,” *arXiv:2212.02500*, 2022\. 

\[228\] F. AI, “Helix: A vision-language-action model for generalist humanoid control,” 2025\. 

\[229\] NVIDIA, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu, “Gr00t n1: An open foundation model for generalist humanoid robots,” *arXiv:2503.14734*, 2025\. 

\[230\] T. Koolen, T. De Boer, J. Rebula, A. Goswami, and J. Pratt, “Capturability-based analysis and control of legged locomotion, part 1: Theory and application to three simple gait models,” *The international journal of robotics research*, vol. 31, no. 9, pp. 1094–1113, 2012\. 

\[231\] A. Herzog, N. Rotella, S. Mason, F. Grimminger, S. Schaal, and L. Righetti, “Momentum con trol with hierarchical inverse dynamics on a torque-controlled humanoid,” *Autonomous Robots*, vol. 40, no. 3, pp. 473–491, 2016\. 

\[232\] Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control,” *The International Journal of Robotics Research*, vol. 44, no. 5, pp. 840–888, 2025\. 

\[233\] G. B. Margolis, G. Yang, K. Paigwar, T. Chen, and P. Agrawal, “Rapid locomotion via reinforce ment learning,” *The International Journal of Robotics Research*, vol. 43, no. 4, pp. 572–587, 2024\. 

\[234\] H. Duan, B. Pandit, M. S. Gadde, B. Van Marum, J. Dao, C. Kim, and A. Fern, “Learning vision based bipedal locomotion for challenging terrain,” in *IEEE International Conference on Robotics and Automation*, pp. 56–62, 2024\. 

36  
\[235\] A. Loquercio, A. Kumar, and J. Malik, “Learning visual locomotion with cross-modal supervi sion,” in *IEEE International Conference on Robotics and Automation*, pp. 7295–7302, 2023\. 

\[236\] R. Yang, M. Zhang, N. Hansen, H. Xu, and X. Wang, “Learning vision-guided quadrupedal locomotion end-to-end with cross-modal transformers,” in *International Conference on Learning Representations*, 2022\. 

\[237\] J. Ren, T. Huang, H. Wang, Z. Wang, Q. Ben, J. Long, Y. Yang, J. Pang, and P. Luo, “Vb-com: Learning vision-blind composite humanoid locomotion against deficient perception,” *arXiv:2502.14814*, 2025\. 

\[238\] J. He, C. Zhang, F. Jenelten, R. Grandia, M. Bächer, and M. Hutter, “Attention-based map encod ing for learning generalized legged locomotion,” *Science Robotics*, vol. 10, no. 105, p. eadv3604, 2025\. 

\[239\] X. Han, M. Li, J. Chen, K. Zhou, and H. Wu, “Multimodal fusion and vision-language models: A survey for robot vision,” *Information Fusion*, vol. 86, pp. 1–25, 2025\. 

\[240\] W. Zhao, K. Gangaraju, and F. Yuan, “Multimodal perception-driven decision-making for human robot interaction: A survey,” *Frontiers in Robotics and AI*, vol. 12, p. 101234, 2025\. 

\[241\] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky, “*π*0: A vision-language action flow model for general robot control,” *arXiv.2410.24164*, 2024\. 

\[242\] G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakr ishna, R. Baruch, M. Bauza, M. Blokzijl, S. Bohez, K. Bousmalis, A. Brohan, T. Buschmann, A. Byravan, S. Cabi, K. Caluwaerts, F. Casarini, O. Chang, J. E. Chen, X. Chen, H.-T. L. Chi ang, K. Choromanski, D. D’Ambrosio, S. Dasari, T. Davchev, C. Devin, N. D. Palo, T. Ding, A. Dostmohamed, D. Driess, Y. Du, D. Dwibedi, M. Elabd, C. Fantacci, C. Fong, E. Frey, C. Fu, M. Giustina, K. Gopalakrishnan, L. Graesser, L. Hasenclever, N. Heess, B. Hernaez, A. Herzog, R. A. Hofer, J. Humplik, A. Iscen, M. G. Jacob, D. Jain, R. Julian, D. Kalashnikov, M. E. Karago zler, S. Karp, C. Kew, J. Kirkland, S. Kirmani, Y. Kuang, T. Lampe, A. Laurens, I. Leal, A. X. Lee, T.-W. E. Lee, J. Liang, Y. Lin, S. Maddineni, A. Majumdar, A. H. Michaely, R. Moreno, M. Neunert, F. Nori, C. Parada, E. Parisotto, P. Pastor, A. Pooley, K. Rao, K. Reymann, D. Sadigh, S. Saliceti, P. Sanketi, P. Sermanet, D. Shah, M. Sharma, K. Shea, C. Shu, V. Sindhwani, S. Singh, R. Soricut, J. T. Springenberg, R. Sterneck, R. Surdulescu, J. Tan, J. Tompson, V. Vanhoucke, J. Varley, G. Vesom, G. Vezzani, O. Vinyals, A. Wahid, S. Welker, P. Wohlhart, F. Xia, T. Xiao, A. Xie, J. Xie, P. Xu, S. Xu, Y. Xu, Z. Xu, Y. Yang, R. Yao, S. Yaroshenko, W. Yu, W. Yuan, J. Zhang, T. Zhang, A. Zhou, and Y. Zhou, “Gemini robotics: Bringing ai into the physical world,” *arXiv:2503.20020*, 2025\. 

\[243\] E. Parisotto, F. Song, J. Rae, R. Pascanu, C. Gulcehre, S. Jayakumar, M. Jaderberg, R. L. Kauf man, A. Clark, S. Noury, M. Botvinick, N. Heess, and R. Hadsell, “Stabilizing transformers for reinforcement learning,” in *International conference on machine learning*, pp. 7487–7498, 2020\. 

\[244\] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pas canu, “Distral: Robust multi-task reinforcement learning,” in *Advances in Neural Information Processing Systems*, 2017\. 

\[245\] T. Yu, C. Finn, S. Dasari, A. Xie, T. Zhang, P. Abbeel, and S. Levine, “One-shot imitation from observing humans via domain-adaptive meta-learning,” in *Robotics: Science and Systems*, 2018\. 

37  
\[246\] D. Kalashnikov, J. Varley, Y. Chebotar, B. Swanson, R. Jonschkowski, C. Finn, S. Levine, and K. Hausman, “Mt-opt: Continuous multi-task robotic reinforcement learning at scale,” *arXiv:2104.08212*, 2021\. 

\[247\] J. Merel, L. Hasenclever, A. Galashov, A. Ahuja, V. Pham, G. Wayne, Y. Teh, and N. Heess, “Neural probabilistic motor primitives for humanoid control,” in *International Conference on Learning Representations*, 2019\. 

\[248\] L. Pan, Z. Yang, Z. Dou, W. Wang, B. Huang, B. Dai, T. Komura, and J. Wang, “Tokenhsi: Unified synthesis of physical human-scene interactions through task tokenization,” in *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 5379–5391, 2025\. 

\[249\] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, A. Lee, K. Fang, C. Finn, and S. Levine, “Bridgedata v2: A dataset for robot learning at scale,” in *Conference on Robot Learning*, pp. 1723–1736, 2023\. 

\[250\] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in *International Conference on Machine Learning*, pp. 1126–1135, 2017\. 

\[251\] K. Rakelly, A. Zhou, C. Finn, S. Levine, and D. Quillen, “Efficient off-policy meta-reinforcement learning via probabilistic context variables,” in *International conference on machine learning*, pp. 5331–5340, 2019\. 

\[252\] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A bench mark and evaluation for multi-task and meta reinforcement learning,” in *Conference on robot learning*, pp. 1094–1100, 2020\. 

\[253\] OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba, “Learning dexterous in-hand manipulation,” *The International Journal of Robotics Research*, vol. 39, no. 1, pp. 3–20, 2020\. 

\[254\] A. Rajeswaran, V. Kumar, A. Gupta, E. Todorov, and S. Levine, “Learning complex dexterous manipulation with deep reinforcement learning and demonstrations,” in *Robotics: Science and Systems*, 2018\. 

\[255\] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Haus man, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mor datch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “Rt-1: Robotics transformer for real-world control at scale,” *arXiv:2212.06817*, 2022\. 

\[256\] I. Taouil, H. Zhao, A. Dai, and M. Khadiv, “Physically consistent humanoid loco-manipulation using latent diffusion models,” *arXiv:2504.16843*, 2025\. 

\[257\] M. Ciebielski, V. Dhédin, and M. Khadiv, “Task and motion planning for humanoid loco manipulation,” in *IEEE-RAS 24th International Conference on Humanoid Robots*, pp. 1179– 1186, 2025\. 

\[258\] C. Wang, Q. Huang, X. Chen, Z. Zhang, and J. Shi, “Robust visuomotor control for humanoid loco-manipulation using hybrid reinforcement learning,” *Biomimetics*, vol. 10, no. 7, p. 469, 2025\. 

38  
\[259\] W. Sun, L. Feng, B. Cao, Y. Liu, Y. Jin, and Z. Xie, “Ulc: A unified and fine-grained controller for humanoid loco-manipulation,” *arXiv:2507.06905*, 2025\. 

\[260\] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020\. 

\[261\] W. J. Rugh and J. S. Shamma, “Research on gain scheduling,” *Automatica*, vol. 36, no. 10, pp. 1401–1425, 2000\. 

\[262\] B. Weng, G. A. Castillo, Y.-S. Kang, and A. Hereid, “Towards standardized disturbance rejection testing of legged robot locomotion with linear impactor: A preliminary study, observations, and implications,” in *IEEE International Conference on Robotics and Automation*, pp. 9946–9952, 2024\. 

\[263\] D. Kahneman, “Thinking, fast and slow,” *Farrar, Straus and Giroux*, 2011\. 

\[264\] M. Zhu, Y. Zhu, J. Li, J. Wen, Z. Xu, Z. Che, C. Shen, Y. Peng, D. Liu, F. Feng, and J. Tang, “Language-conditioned robotic manipulation with fast and slow thinking,” in *IEEE International Conference on Robotics and Automation*, pp. 4333–4339, 2024\. 

\[265\] Q. Liao, B. Zhang, X. Huang, X. Huang, Z. Li, and K. Sreenath, “Berkeley humanoid: A re search platform for learning-based control,” in *IEEE International Conference on Robotics and Automation*, pp. 2897–2904, 2025\. 

\[266\] Y. Chi, Q. Liao, J. Long, X. Huang, S. Shao, B. Nikolic, Z. Li, and K. Sreenath, “Demonstrat ing berkeley humanoid lite: An open-source, accessible, and customizable 3d-printed humanoid robot,” *arXiv:2504.17249*, 2025\. 

\[267\] H. Shi, W. Wang, S. Song, and C. K. Liu, “Toddlerbot: Open-source ml-compatible humanoid platform for loco-manipulation,” *arXiv:2502.00893*, 2025\. 

\[268\] G. Ficht, H. Farazi, A. Brandenburger, D. Rodriguez, D. Pavlichenko, P. Allgeuer, M. Hosseini, and S. Behnke, “Nimbro-op2x: Adult-sized open-source 3d printed humanoid robot,” in *IEEE RAS 18th International Conference on Humanoid Robots*, pp. 1–9, 2018\. 

\[269\] “Poppy project: Open-source 3d-printed humanoid platform.” https://www. poppy-project.org/. 

\[270\] “Mit cheetah software.” https://github.com/mit-biomimetics/ Cheetah-Software. Retrieved 2025\. 

\[271\] L. Humanoid Robot (Shanghai) Co., “OpenLoong-DynamicsControl: Motion control framework of humanoid robot based on MPC and WBC,” 2024\. 

\[272\] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, “CasADi – A software frame work for nonlinear optimization and optimal control,” *Mathematical Programming Computation*, vol. 11, no. 1, pp. 1–36, 2019\. 

\[273\] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. v. Duijkeren, A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, “acados—a modular open-source framework for fast em bedded optimal control,” *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022\. 

39  
\[274\] G. Neunert, T. Stäuble, M. Giftthaler, M. Stäuble, M. R. Tognon, M. Frigerio, and J. Buchli, “OCS2: An open source library for optimal control of switched systems,” *IEEE Transactions on Control Systems Technology*, vol. 28, no. 5, pp. 1313–1325, 2020\. 

\[275\] A. H. Li, B. Hung, A. D. Ames, J. Wang, S. L. Cleac’h, and P. Culbertson, “Judo: A user-friendly open-source package for sampling-based model predictive control,” *arXiv:2506.17184*, 2025\. 

\[276\] “Nvidia isaac lab.” https://developer.nvidia.com/isaac/lab. Retrieved 2025\. 

\[277\] mujocolab, “MJLab: Isaac lab api, powered by mujoco-warp,” 2025\. 

\[278\] D. R. Newton Physics Engine Contributors (NVIDIA, Google DeepMind, “Newton: Gpu accelerated physics simulation engine for robotics,” 2025\. 

\[279\] G. Authors, “Genesis: A generative and universal physics engine for robotics and beyond,” De cember 2024\. 

\[280\] M. W. Spong, S. Hutchinson, and M. Vidyasagar, *Robot Modeling and Control*. John Wiley & Sons, 2006\. 

\[281\] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT press, 2016\. 

\[282\] D. P. Kingma and M. Welling, *An Introduction to Variational Autoencoders*. Foundations and Trends in Machine Learning, 2019\. 

\[283\] C.-H. Lai, Y. Song, D. Kim, Y. Mitsufuji, and S. Ermon, “The principles of diffusion models,” *arXiv:2510.21890*, 2025\. 

40