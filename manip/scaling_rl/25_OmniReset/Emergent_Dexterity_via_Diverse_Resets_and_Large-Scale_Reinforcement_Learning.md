##### Report GitHub Issue

**×




Title:
*


Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.15789v2 [cs.RO] 24 Mar 2026

[# Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning Patrick Yin1,∗ Tyler Westenbroek1,∗ Zhengyu Zhang2 Joshua Tran1 Ignacio Dagnino1 &Eeshani Shilamkar1 Numfor Mbiziwo-Tiapo1 Simran Bagaria3 Xinlei Liu1 &Galen Mullins3 Andrey Kolobov3 Abhishek Gupta1 1University of Washington 2NVIDIA 3Microsoft Research ∗Equal contribution ###### Abstract Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning. However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, typical reinforcement learning methods can often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce OmniReset, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using fixed algorithm hyperparameters, no curricula, minimal reward engineering, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions that underlie dexterous manipulation. OmniReset programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains for dynamic policies. We show that OmniReset gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies demonstrating a variety of dynamic, contact-rich recovery behavior. Finally, we distill OmniReset into visuomotor policies that can be transferred to the real world zero-shot, displaying robust retrying behavior to accomplish complex, contact-rich tasks with non-trivial success rates. Project webpage: https://omnireset.github.io Figure 1: OmniReset is a scalable framework for dexterous manipulation which uses diverse simulator resets and large scale reinforcement learning to solve contact-rich, long horizon tasks beyond the capabilities of existing approaches. These policies are then distilled to RGB camera observations and robustly transferred to the real world zero-shot, where they are able to consistently solve challenging tasks over much wider ranges of initial conditions than baselines. ## 1 Introduction Reinforcement learning (RL) in massively parallelized simulation environments (Mittal et al., 2023](#bib.bib50); Todorov et al., [2012](#bib.bib51)) has driven recent successes in sim-to-real robotics [(Akkaya et al., [2019](#bib.bib52); Hwangbo et al., [2019](#bib.bib54))]. While very successful for locomotion and navigation problems, these methods have seen relatively less success in robotic manipulation problems. In principle, these algorithms can also be applied to robotic manipulation, automatically acquiring complex, contact-rich behaviors through repeated interaction with the simulated environment. Yet in practice, obtaining robust and performant policies still requires extensive per-task environment and reward engineering, limiting the scalability of this paradigm—particularly for long-horizon manipulation tasks. This sharply contrasts with domains such as language modeling, where simply scaling data and compute has yielded dramatic gains with similarly simple RL algorithms [Guo et al. ([2025](#bib.bib72))]. How can we achieve comparable scalability for robotic manipulation?

The central bottleneck in robotic RL is that standard exploration techniques [Schulman et al. ([2017](#bib.bib57)); Haarnoja et al. ([2018](#bib.bib64))] saturate as parallelism and compute are scaled, repeatedly sampling a narrow state–action distribution and becoming trapped in local minima [Singla et al. ([2024](#bib.bib73))]. While numerous advanced exploration methods have been proposed [Pathak et al. ([2017](#bib.bib23)); Burda et al. ([2018](#bib.bib22))], their added algorithmic complexity makes them difficult to scale in practice. As a result, practitioners often rely on human intuition to reduce the exploration burden, injecting structure through task-specific reward design [Westenbroek et al. ([2022](#bib.bib70)); [Ng and others](#bib.bib49); Handa et al. ([2023](#bib.bib58))], hand-designed curricula [Lee et al. ([2020a](#bib.bib68))], or user-provided demonstrations [Bauza et al. ([2025](#bib.bib35)); Peng et al. ([2018](#bib.bib60)); Nair et al. ([2018](#bib.bib36))]. Although effective in many settings, these approaches are fundamentally limited by the amount of human effort they require. A complementary strategy is to simplify the learning problem through additional system scaffolding—using RL only for contact-rich phases while relying on motion planning or trajectory optimization for the remainder [Lee et al. ([2020b](#bib.bib67)); Tang et al. ([2024](#bib.bib42)); Zhou et al. ([2024](#bib.bib3)); Lee et al. ([2020c](#bib.bib2))]. While this reduces exploration demands, it comes at the cost of increased system complexity and a lack of smooth, adaptive behavior for complex tasks. These approaches reflect a prevailing assumption in robotics: that dexterous manipulation is too complex to emerge from large-scale RL alone and must instead be scaffolded with additional task-specific structure.

We argue that with the right system design, this scaffolding is unnecessary. In this work, we instead propose systematically exposing RL to a superset of the interactions it will encounter when manipulating the scene*, and then allowing dexterous task-specific behaviors to emerge from large-scale compute and optimization. Although the space of possible behaviors for robust task performance is vast—flipping, screwing, insertion, and other contact-rich motions—successful policies reuse a relatively small set of recurring interaction modes, such as approaching objects, initiating contact, and forming stable grasps. These modes can be densely covered through generic resets that do not encode task-specific solutions. By sufficiently randomizing object poses and sampling these interaction states, we substantially reduce the exploration burden and ensure the agent encounters meaningful success signals. This coverage allows sparse rewards to propagate smoothly through the state space, enabling the agent to identify high-value regions and learn how to stitch together multiple distinct behaviors to reach these goals, without task-specific reward shaping or guidance.

Concretely, we introduce OmniReset, a scalable framework for robotic manipulation that automatically generates diverse initial-state distributions to densely cover the contact-rich interactions the robot may encounter. This coverage allows PPO [Schulman et al. ([2017](#bib.bib57))] to fully leverage large-scale compute, improving performance as the number of parallel environments increases. When scaled sufficiently, OmniReset learns the emergent behavior of combining multiple interaction modes—such as pushing, flipping, and insertion (Fig. [1](#S0.F1))—into coherent, multi-stage strategies without task-specific reward shaping, curricula, or demonstrations. Across diverse contact-rich tasks, this approach solves long-horizon problems that are considerably out of reach for existing methods, producing robust policies that succeed from a huge range of initial states rather than narrow distributions typical of many prior approaches. Finally, we demonstrate that OmniReset can be used to train visuomotor policies via student-teacher distillation which can robustly transfer zero-shot to the real-world, substantially outperforming alternatives such as imitation learning.

## 2 Related Work

Exploiting Resets in Reinforcement Learning: Exploiting simulator resets for RL is a natural idea which has been explored in many contexts. Prior theoretical works [(Kakade and Langford, [2002](#bib.bib66))] have suggested more uniform sampling over initial states, but do not provide practical algorithms. The primary focus of many works is to make learning more tractable by generating an explicit curriculum over resets [(Tang et al., [2023](#bib.bib19); Dennis et al., [2020](#bib.bib78); Bauza et al., [2025](#bib.bib35))], for instance through a “reverse-curriculum” of states going backwards from the goal [(Florensa et al., [2017](#bib.bib16))] or using a learned dynamics model to propose viable resets [(Edwards et al., [2018](#bib.bib18); Ivanovic et al., [2019](#bib.bib37))]. In contrast, a second category of methods leverage *demonstrations* (whether human or automatically generated) to generate feasible pathways to the goal [(Tao et al., [2024](#bib.bib39); Resnick et al., [2018](#bib.bib40); Salimans and Chen, [2018](#bib.bib41); Bauza et al., [2025](#bib.bib35); Tang et al., [2024](#bib.bib42))]. In contrast to these prior works, we show that neither human demonstrations, nor a curriculum is needed, but rather that the simple approach for generating diverse resets in OmniReset naturally scales to various long-horizon manipulation problems without added algorithmic complexity. More recently, staggered environment resets [(Bharthulwar et al., [2025](#bib.bib89))] have been shown to improve optimization by decorrelating rollouts, but leave the initial state distribution unchanged and thus do not alleviate exploration. In contrast, OmniReset directly addresses exploration by enumerating diverse, task-relevant initial states, enabling coverage of critical intermediate states that are otherwise rarely visited.

Exploration Strategies for Reinforcement Learning: RL practitioners have designed a variety of exploration strategies to effectively uncover goal-reaching paths for a *fixed set of initial conditions*, with uninformative rewards. A major line of work is bonus-based exploration, where agents receive intrinsic rewards for visiting novel or unpredictable states. Count-based methods reward visits to rarely seen states [(Ostrovski et al., [2017](#bib.bib24); Bellemare et al., [2016](#bib.bib25); Martin et al., [2017](#bib.bib26))], while curiosity-based methods provide bonuses based on prediction errors [(Burda et al., [2018](#bib.bib22); Sancaktar et al., [2022](#bib.bib29); Pathak et al., [2017](#bib.bib23))]. Other approaches [(Osband et al., [2016](#bib.bib30); [2019](#bib.bib11); Russo et al., [2018](#bib.bib10))] promote temporally correlated exploration by injecting stochasticity at the policy or value-function level. Finally, diversity-driven methods optimize for behavioral variety [(Eysenbach et al., [2019](#bib.bib34); Rajeswar et al., [2023](#bib.bib9))]. These approaches are complimentary to our work; our main contribution is demonstrating that large scale-scale parallelization and resetting schemes lead to the emergence of surprising levels of dexterity without the need for advanced exploration incentives.

Leveraging Demonstrations: An alternative is to increasingly rely on human demonstrations and imitation learning to overcome difficult long-horizon exploration. Approaches include adding auxiliary BC loss terms to RL objectives [(Nair et al., [2018](#bib.bib36); Hester et al., [2018](#bib.bib45); Rajeswaran et al., [2017](#bib.bib85))], simply adding demonstrations to the replay buffer [(Vecerik et al., [2017](#bib.bib46))], and introducing reward shaping terms which encourage RL agent to follow demonstrations [(Tang et al., [2024](#bib.bib42); Reddy et al., [2019](#bib.bib47); Koprulu et al., [2024](#bib.bib48); Peng et al., [2018](#bib.bib60))]. Other works have sought to squeeze more information out demonstrations by automatically translating existing demonstrations to new initial conditions and scenes [(Mandlekar et al., [2023](#bib.bib44))] or by robustifying BC policies by promoting recovery behavior [(Ke et al., [2023](#bib.bib21); Ankile et al., [2024](#bib.bib20))]. While our work complementarily pushes the limits of what behaviors can be learned entirely from scratch, we expect that demonstrations (when available) can also be incorporated into our framework to further accelerate learning.

## 3 Generating Diverse Resets for Learning Dexterous Manipulation

*Figure 2: Sim-to-Real Pipeline with OmniReset (1) After generating partial assemblies and grasps from the simulator, (2) we collect reset states: reaching, near object, grasped, and near goal. (3) We then train a state-based RL policy initialized from these reset states, which is used to (4) train student-teacher distillation to get a RGB policy. (5) By finetuning this RGB policy on a mix of simulation data and small set of real demonstrations, (6) we deploy the RGB-based policy in the real world.*

This section introduces OmniReset, a scalable framework for robotic manipulation that automatically constructs RL problems by generating diverse, manipulation-centric reset distributions in a task-agnostic way. Rather than relying on task-specific curricula, demonstrations, or carefully tuned rewards to guide exploration, OmniReset exposes standard RL algorithms to a broad superset of key interaction states that would be rarely encountered under naïve exploration. When paired with large-scale simulation and compute, this reset design continually exposes the algorithm to diverse interaction states, preventing convergence to narrow, suboptimal behaviors and enabling complex multi-step manipulation skills to emerge from large-scale optimization.

### 3.1 Problem Setting

Reinforcement Learning Problem: We formalize the RL problems synthesized by OmniReset as a Markov decision process (MDP) defined by the tuple $(\mathcal{S},\mathcal{A},P,r,\gamma,\rho)$, where $s\in\mathcal{S}$ denotes the state, $a\in\mathcal{A}$ denotes the robot’s action, $s^{\prime}\sim P(\cdot|s,a)$ denotes the next state sampled from the transition dynamics, $r$ is the reward function, $\gamma$ is the discount factor, and $s_{0}\sim\rho$ is the initial state distribution. We optimize for the discounted sum of rewards: $J(\pi)=\mathbb{E}_{s_{0}\sim\rho,a\sim\pi}\!\left[\sum_{t=0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right]$, where $a\sim\pi(\cdot|s)$ denotes actions sampled from the policy. We focus our RL training on compact state representations (i.e., Lagrangian states), only moving to vision-based distillation for transfer to the real world (Sec. [5](#S5)).

Task Scope and User Inputs: To effectively leverage task structure when automatically designing RL problems we make several practical assumptions. First, we focus on manipulating rigid bodies where the goal is to move a single user-specified object to a target configuration (potentially relative to other objects). For example, the assembly task in Figure [1](#S0.F1) requires picking up a table leg, moving it to the desired hole, and screwing the pieces together—a long-horizon task requiring diverse behaviors, yet framed as moving one object to a goal. Specifically, OmniReset requires the following input from the user:

A target object $s^{\text{tar}}\subset s$ to be manipulated.

A set of goal configurations $\mathcal{G}\subset\mathcal{S}$ for $s^{\text{tar}}$.

A workspace $\mathcal{W}\subset\mathcal{S}$ for the robot.

At a code level, this requires that the user identify $s^{tar}$ in the environment definition, provide a means for sampling goal states, and select and operating region for the robot (such as the area over a table top). These pieces of information place minimal burden on the user, yet nonetheless contain rich information about problem structure which OmniReset exploits to design easily solvable RL problems. In short, OmniReset automatically constructs a diverse initial state distribution $s_{0}\sim\rho$ that enables RL algorithms to solve all considered tasks using a single task-agnostic reward function, without any task-specific hyperparameter tuning or explicit curricula.

### 3.2 Automatically Generating RL Problems with Diverse Resets

Reinforcement learning fails to solve long-horizon manipulation tasks when training only encounters a narrow slice of potential object configurations or robot-object interactions. To prevent this collapse, OmniReset systematically expands coverage along two axes. First, we approximately cover the space of pathways along which the target object can be transported to the goal by densely resetting $s^{tar}$ on the tabletop, at random point in the air, and at and near the goal $\mathcal{G}$. This exposes the RL algorithm to difficult-to-discover success signals, and allows these signals to smoothly propagate throughout the state-space as the value function is updated. Next, we cover the different ways the robot can interact with $s^{tar}$ by resetting the arm in configurations where it is reaching towards $s^{tar}$, making contact with $s^{tar}$ from a wide variety of points, and with stable grasps on $s^{tar}$. This exposes the RL algorithm to the different ways the robot can interact with $s^{tar}$ to move it towards the goal. Altogether, this broad coverage enables the RL algorithm to discover high-value regions of the state space and the behaviors required to reach those states. In the parlance of motion planning, we are exposing the agent to states in “narrow passages”, allowing them to be easily traversed through RL-based optimization.

Specifically, we generate these resets as follows. First, we use the grasp sampler from [Mittal et al. ([2023](#bib.bib50))] to calculate a set of $1000$ feasible grasp points on the target object $s^{tar}$, as depicted in Figure [3(a)](#S3.F3.sf1). Next, we generate a set of feasible offsets for $s^{tar}$ relative to the goal $\mathcal{G}$. This is accomplished by spawning $s^{tar}$ at $\mathcal{G}$, and then applying small random forces to dislodge the target from the goal, as in [Tang et al. ([2023](#bib.bib19))]. For example, for the insertion task in Figure [1](#S0.F1), this process generates a continuum of relative configurations between the peg and the hole where the peg is only partially inserted. With these pre-computed quantities, we generate the following resets over the space of robot-object configurations:

![Figure](2603.15789v2/figs/Grasp_Sampling.png)

*(a) Grasp Sampling. We display the grasp poses ($S^{G}\subset S$) sampled for the table leg. From left-to-right, the grasp sampling ranges are broad, moderate, and narrow. Our method uses the broad range.*

1) Reaching Resets: $S^{R}\subset\mathcal{S}$ capture states where the robot is moving towards the target object. For our tabletop manipulation problems, this corresponds to resetting the target object at diverse poses on the table top, with the gripper spawned at random poses throughout the workspace $\mathcal{W}$.

2) Near-Object Resets: $S^{NO}$ also spawns $s^{tar}$ across the tabletop, but then resets the end effector to one of the pre-computed grasp points with a small random offset and randomly set the gripper to either be open or closed. This provides broad coverage over states where the robot is initiating contact with $s^{tar}$ from a wide distribution of directions, providing coverage over both non-prehensile interactions and the initiation of stable grasps.

3) Stable Grasp Resets: $S^{G}\subset S$ covers states where prehensile manipulation occurs and the robot has secured a stable grasp on $s^{tar}$. We spawn $s^{tar}$ randomly in the air throughout $\mathcal{W}$, then spawn the gripper at a feasible grasp point.

4) Near-Goal Resets: $S^{NG}\subset S$ provides dense coverage over near goal states where contact rich behavior such as insertion or twisting occur. We spawn $s^{tar}$ at one of the pre-computed offsets from the goal $G$, then spawn the gripper to be in contact with $s^{tar}$ as with the Near-Object resets.

Altogether, these different reset regions provide dense, approximate coverage over the space of pathways to the goal, without prescribing a priori* which behaviors are needed to solve the task. It is important to note two important things about this reset distribution- 1) we do not order or connect the states in any graph structure, the paths between them are completely determined by RL, 2) we do not inform any dynamic behavior between these states, these are completely emergent from the RL process. Indeed, as we see from our wide range of examples (Fig.[1](#S0.F1)), the RL algorithm is free to select completely different pathways through the state space when solving different tasks, utilizing the reset states that are useful while ignoring those that are not. For example, in the Drawer Insertion task, at convergence, the RL algorithm does not obtain stable grasps on the drawer, but instead repeatedly flips the drawer and then pushes it into the cabinet. In contrast, for the Leg Twisting tasks, the robot picks up the table leg, pushes it against the table to obtain a more favorable grasp, and then twists the leg into the hole.

Practical Implementation: In practice, it can be difficult to efficiently sample *feasible resets* that respect the contact constraints of the physics simulator. Sampling invalid initial conditions can lead to pathological and non-physical behavior that destabilizes learning. Moreover, reset states must be sampled with minimal overhead to maximize GPU-parallelism. Thus, we first sample feasible resets during an offline phase, sampling proposed resets from the four regions defined above and rejecting invalid samples using a combination of collision checking and stepping the simulator for a few steps to allow for stabilization. This yields four validated datasets corresponding to the four reset distributions described above: $D^{R}$, $D^{NO}$, $D^{G}$, and $D^{NG}$. During RL training, we sample from $\mathrm{Uniform}(D)$ where $D=D^{C}\cup D^{NO}\cup D^{G}\cup D^{NG}$. We cache these resets on-GPU to ensure efficient sampling during training.

### 3.3 Algorithmic Decisions for RL Training

Next, we discuss key algorithmic decisions that are required for an on-policy algorithm (PPO [Schulman et al. ([2017](#bib.bib57))]) to learn manipulation behavior by leveraging the diversity of reset states and scale to the complexity of tasks we consider in this work.

Task-Agnostic Reward Structure: Leveraging the design choices described above, we use a simple, common reward function shared across all tasks:

$$r(s_{t},a_{t})=r_{\text{success}}(s_{t})+r_{\text{dist}}(s_{t})+r_{\text{reach}}(s_{t})+r_{\text{smooth}}(s_{t},a_{t})+r_{\text{term}}(s_{t}).$$ \tag{1}

Here, $r_{\text{success}}$ is a sparse binary reward indicating task completion, $r_{\text{dist}}$ encourages minimizing the distance of the target object $s^{tar}$ to the goal, $r_{\text{reach}}$ encourages the gripper to be near $s^{tar}$, $r_{\text{smooth}}$ penalizes large or rapidly changing actions, and $r_{\text{term}}$ penalizes unsafe or physically invalid states that trigger terminations. Importantly, this reward does not encode task-specific strategies as all components and weights are kept fixed across experiments. We find that this generic structure is sufficient for stable training across diverse manipulation tasks, and performance is largely insensitive to the precise weighting of individual terms. See Appendix [A.1](#A1.SS1) for additional details.

Scaling Parallel Environments: When combined with increasing reset diversity, we found that scaling the number of parallel environments used by PPO stabilized and accelerated learning. This enables OmniReset to obviate curricula and task dependent rewards, significantly reducing the amount of tuning which is required to solve a new task. Intuitively, this large batch size prevents catastrophic forgetting in situations when many of the reset states result in unsuccessful policy behavior. This is necessary to ensure continued value propagation backwards from the target configuration.

Asymmetric Actor-Critic: Since we have access to privileged information in a simulator, we use an asymmetric actor-critic approach [(Pinto et al., [2017](#bib.bib75))] for our learning architecture. The actor observations include a history of the five previous time-steps for the state of the robot, the poses of all objects in the scene, and the previous actions taken by the policy. The critic takes in these observations as well as additional privileged parameters of the environment. We found that conditioning the actor on larger observation spaces led to less stable training and led us to only provide this information to the critic.

Generalized State-Dependent Exploration Noise: We employ the policy noise parameterization from gSDE [(Raffin et al., [2022](#bib.bib76))]. gSDE has a separate prediction head which determines the gaussian exploration noise at each time-step and is conditioned on the features of the final layer of the policy network. This approach enables the actor to learn different temporally-correlated exploration strategies in different regions of the state-space, crucial for solving heterogeneous multi-stage tasks.

## 4 Simulation Experiments

Let us first consider a study of data generation in simulation with OmniReset. We aim to address the following questions experimentally - (Q1) Does OmniReset outperform baselines in terms of asymptotic performance and sample complexity, enabling tasks beyond current methods? (Q2) How do the key design decisions laid out in Section [3.3](#S3.SS3) affect performance? (Q3) Can the learned RL policies be used to generate diverse data for sim-to-real transfer?

### 4.1 Task Descriptions

We use the following tasks to demonstrate the effectiveness of the OmniReset framework. The Leg Twisting tasks is a replication of the the square_table task from [(Heo et al., [2023](#bib.bib84))], and involves screwing in a single table leg. The Drawer Insertion task is based on the task by the same name from [(Heo et al., [2023](#bib.bib84))] and requires inserting a drawer into a dresser. The Peg Insertion task requires inserting a peg into a hole. The Cube Stacking task requires stacking one cube on top of another with a desired orientation. The Wall Slide task requires non-prehensile motion to push a block up against a wall into a desired orientation. Finally, the Cupcake Placement task requires placing a cupcake on a plate at a desired orientation. For each task, we consider both Hard and Easy variants. For the Hard variants, the target object $s^{tar}$ is distributed on the table with $x-y$ coordinates in $(x,y)\in[-0.2,0.2]\times[-0.15,0,15]$ and task-specific randomization over the goal position, whereas the Easy version of each task uses a highly restricted set of initial conditions with $(x,y)\in[0.1,0.12]\times[0.1,0.12]$ with a single fixed goal location. We refer the reader to our public code release for additional task specific parameters, but summarize each task below. Snapshots from the tasks are depicted in Figures [1](#S0.F1) and [4](#S4.F4)

Finally, to fully demonstrate the utility of OmniReset, we solve the Four Leg task depicted in Figure [4](#S4.F4) by $a)$ training independent OmniReset policies to screw in each of the four legs then $b)$ using a simple scripting policy to switch between the policies to complete the overall long-horizon task. The demonstrations highlights how OmniReset can be combined with high-level planning to push the horizon of tasks that can be solved to even greater extremes.

*Figure 4: Additional tasks. Visualizations of the new manipulation tasks solved with OmniReset. From left to right: Four-Leg Table Assembly, Cube Stacking, Cupcake on Plate, Block Reorientation on Wall*

### 4.2 Baseline Comparisons in Simulation

![Figure](2603.15789v2/x3.png)

*Figure 5: Success rates during RL training. We plot success rates over learning process for tasks described in Sec.[4](#S4). We see that OmniResetscales to task where baselines struggle to make meaningful progress, especially with the wide range of initial conditions specified in the Hard variants of tasks.*

We compare to the following baselines, which bootstrap learning with expert demonstrations, providing them with more prior information about how to solve the task than OmniReset. For each of the methods, we supply 100 successful demonstrations, effectively giving the baselines additional access to the optimal behavior, while OmniReset does not have this information. The initial conditions for these demonstrations are drawn from the reaching region $S^{R}$, which corresponds to the set of initial conditions the robot will encounter when solving the full task.

-

1) BC-PPO: We add a Behavior-Cloning (BC) loss to the PPO objective to construct a baseline emblematic of numerous works combining BC and RL objectives [(Hester et al., [2018](#bib.bib45); Rajeswaran et al., [2017](#bib.bib85))]. When training this algorithm, the environment is always reset from the reaching region $S^{R}$, which reflects the ‘standard’ reset distribution typically used to solve such tasks.

-

2) DeepMimic: To compare with methods for motion imitation, we use DeepMimic-style reward augmentation [(Peng et al., [2018](#bib.bib60))] on top of our generic rewards. During resets, a random demonstration is chosen, and the agent is reset from a random point along the demonstration and is given an auxiliary reward which provides bonuses for following the demonstration.

-

3) Demo Curriculum: This baseline is constructed in the spirit of method such as [(Bauza et al., [2025](#bib.bib35))] and uses a success-weighted autocurriculum to sample reset states from the demonstrations. We use PPO [(Schulman et al., [2017](#bib.bib57))] as the base RL algorithm to ensure a fair comparison.

Learning Curves and Success Rates: We plot learning curves for the Hard variants of each task and the Easy variants of the Peg Insertion, Leg Twisting, and Drawer Insertion tasks in Figure [5](#S4.F5). Success rates are reported for initial conditions sampled from the Reaching resets $S^{R}$, which correspond to states from which the robot must solve the full task. We see that OmniReset is able to consistently obtain high success rates on each of the tasks, substantially outperforming baseline methods. While the baselines make some progress on the Easy variants of the tasks, they consistently struggle to scale to the wider distributions of initial conditions that define the Hard variants. Appendix Figure [12](#A1.F12) provides additional fine-grained insight into the failure modes of the baselines. Here we plot success rates during training for initial conditions drawn from the demonstrations that fall into the Near-Goal and Reaching regions of the state-space. This plot demonstrates how the baselines are able to solve part of the task (i.e, when starting near the goal) but fail to scale to the full long-horizon task (i.e, starting from a reaching position). We refer readers to the supplementary website for a more detailed look at failure modes.

![Figure](2603.15789v2/figs/Perturbations.png)

*Figure 6: Success rate over perturbations. We plot the decline in success rate (measured by ratio of success rate between no perturbations and current level of perturbations). We find that OmniResetis robust to perturbations while performance of baselines drops significantly.*

Emergent Curricula: OmniReset does not rely on curricula to stabilize and accelerate learning. We visualize how learning for OmniReset progresses over time on our [website](https://omnireset.github.io/#learning-over-time). We observe that the diverse resets OmniReset provides enables PPO to naturally solve the problem backwards, first learning to succeed from near-goal states and eventually learning to succeed from the entire search space. This behavior is entirely emergent from our diverse resets and large number of parallel environments.

Robustness of Policies: We conduct a robustness analysis on the learned policies (Fig [6](#S4.F6)). We sample an initial condition from one of the demonstrations, perturb the initial condition with forces of different magnitudes and report policy success from these perturbed initial conditions. We find that baseline performance quickly degrades under small perturbations, while OmniReset is barely affected under large perturbations.

We also analyze the ability of policies to solve the task from a wide range of initial conditions in the scatter plots in Figure [7](#S4.F7). These plots show success rates over various initial conditions for both OmniResetand Demo Curriculum (the most successful baseline) on the Leg Twisting Easy task. For each plot we show the success rate over 1000 sampled initial conditions from the full task distribution. This plot demonstrates how the baseline struggles to achieve achieve consistent success across the distribution of initial conditions it was trained on, while OmniReset achieves is able to succeed across the entire workspace.

![Figure](2603.15789v2/x4.png)

*Figure 7: Successful initial states of RL policy. For the Lew Twisting, we plot the xy configurations from which RL polices trained with Demo Curriculum and OmniResetsucceed. We find that OmniResetsucceeds from a much broader range of initial conditions.*

### 4.3 Ablating Key Design Decisions

![Figure](2603.15789v2/x5.png)

*Figure 8: Ablation on number of environments. We plot the success rates over course of RL using different number of environments. We find that the number of environments significantly impacts training performance.*

We ablate $1)$ the number of parallel environments (and PPO batch size) and $2)$ the range of reset randomization used by OmniReseton the Leg Twisting Hard task. Figure [8](#S4.F8) ablates the number of parallel environments and shows success rates during training from the four different reset distributions used by OmniReset. The reset distributions in Figure [8](#S4.F8) are roughly ordered from left to right from the end of the task (Near-Goal) to the beginning of the task (Reaching). While runs with a smaller number of environments can make progress at solving the task from Near-Goal states, we observe that a large number of parallel environments are essential for scaling to the complexity of the full multi-stage task (i.e. from reaching states). Similarly, we see that increased diversity in the grasps used by OmniReset has a substantial effect on sample efficiency, highlighting how densely covering the different modes of robot-object interactions is crucial for efficient and scalable RL training.

![Figure](2603.15789v2/x6.png)

*Figure 9: Ablation on grasp sampling range. For this ablation on the Leg Twisting task, we find that training RL on narrower grasp sampling ranges leads to worse sample efficiency and lower success rate.*

## 5 Distillation and Real-World Transfer

We demonstrate the utility of our learned data-generation policies by distilling them into visuomotor policies deployable directly on hardware from RGB inputs. Experiments are conducted on a UR7e robot equipped with a Robotiq 2F-85 gripper, with control and policy inference running on a PC with an RTX 4090 GPU. The robot observes $224\times 224$ RGB images from three RealSense cameras: a D455 providing the front view, a D435 for the side view, and a D415 mounted on the wrist. Using the photorealistic rendering capabilities of IsaacLab [(Mittal et al., [2023](#bib.bib50))], we collect 80,000 expert rollouts with synchronized images and actions for standard student–teacher distillation [Chen et al. ([2021](#bib.bib6))]. The student policy operates at 10Hz and uses an ImageNet-pretrained ResNet-18 encoder and a Gaussian MLP head conditioned on the five most recent observations. See Appendix [A.3](#A1.SS3) for full implementation details for distillation and transfer.

Visual randomization. As shown in Fig. [11](#S5.F11), to mitigate the sim-to-real visual gap, we apply extensive domain randomization following DextrAH-G [(Singh et al., [2025](#bib.bib86))], varying lighting, backgrounds, object and robot appearance, and workspace textures. Camera extrinsics are calibrated to the real setup with additional pose and FOV jitter for robustness. We also apply standard image augmentations including color jitter, blur, grayscale, and noise.

![Figure](2603.15789v2/x7.png)

*Figure 10: Sim2Real Tasks. Simulation setup for pretraining (top) and real-world deployment (bottom) of peg insertion (left), leg twisting (middle), and drawer assembly (right).*

Dynamics randomization. To reduce the control gap, we calibrate kinematics to hardware and deploy the same task-space operational space controller [(Khatib, [1987](#bib.bib88))] in simulation and reality, with the policy predicting end-effector pose deltas that are converted to torques using identical Jacobian-based control. We perform system identification of key actuator parameters (friction, armature, and delay) to match real joint behavior following PACE [(Bjelonic et al., [2025](#bib.bib87))], then randomize controller gains and physical parameters around identified values during RL training. We additionally randomize object mass and friction to improve robustness to contact dynamics, and apply a curriculum that progressively reduces the action space to encourage smaller, smoother motions. We emphasize that this curricula was used to improve sim-to-real transfer, and is not necessary for learning high-performing policies in simulation. Indeed, the simulation only evaluations we report are trained without the curricula, maintaining our claim that OmniReset does not rely on complex training curricula for behavior generation.

![Figure](2603.15789v2/figs/paper_dr_grid_4x4.png)

*Figure 11: Visual randomizations. Examples of domain randomization applied during training. We vary lighting conditions, backgrounds, object and robot appearance, and workspace textures, along with camera pose and field-of-view jitter, to improve robustness to real-world visual variation.*

Real-World Transfer. We successfully deploy a distilled OmniReset policy on the Peg Insertion, Leg Twisting, and Drawer Insertion tasks, as shown in Figure [10](#S5.F10). The distilled RGB policies, trained on 80,000 simulation trajectories spanning a wide range of initial conditions, achieve zero-shot real-world success rates of 85.37% on Peg, 56.36% on Leg, and 15.38% on Drawer, as summarized in Table [1](#S5.T1). This substantially outperforms a behavior cloning Diffusion Policy baseline [Chi et al. ([2025](#bib.bib63))], which achieves $\sim$2% success across tasks when trained on 100 demonstrations.

We additionally report first-try success*, defined as completing the task with a single grasp and execution (i.e., no dropping and regrasping), as well as *throughput*, measured as successful completions per minute. These metrics highlight that there remains significant room for improvement in efficiency and reliability. Full real-world evaluation details are provided in Appendix [A.4](#A1.SS4).

Qualitatively, the OmniReset visuomotor policy exhibits robust retrying behavior, recovering from initial failures and successfully completing the task. For examples of these behaviors, see [our project website](https://omnireset.github.io/#evaluations). Overall, these results demonstrate that OmniReset provides a scalable foundation for training sim-to-real policies capable of handling substantially broader initial condition distributions than prior approaches.

*Table 1: Sim-to-real performance across tasks: We report state-based RL performance in simulation, distilled image policy performance in simulation and real-world deployment, and a behavior cloning baseline trained on only 100 real-world demos.*

## 6 Conclusion

In this work we presented OmniReset, a simple and scalable system for data generation in simulation for complex, dexterous tasks. The primary insight in OmniReset is showing that a diverse, minimally structured set of reset states paired with large-batch on-policy reinforcement learning in simulation can lead to the emergence of surprisingly complex dexterous behavior. We provide a general purpose recipe to instantiate data generators across a variety of manipulation tasks, and demonstrate both the efficacy of this paradigm in simulation and it’s ability to train robust policies which can be successfully transferred directly to the real world. However, the preset framework has several limitations which leave the door open for exciting future works. OmniReset is dependent on the quality of grasps obtained by the grasp sampler, which can fail to generate diverse grasps on complex, highly non-convex objects. Moreover, pre-computing stable grasps for tasks which require bimanual manipulation or dexterous hands will present additional challenges, and whether the techniques and principles behind OmniReset will scale to such settings remains an open question. Moreover, OmniReset uses relatively modest levels of dynamics randomization for RL training compared existing sim-to-real approaches, and we expect there to be additional challenges training policies which must adapt their behavior to solve the task over a wide range of potential operating conditions. For distillation and transfer, even in simulation our RGB policies achieve much lower success rates than the state-based experts. We expect that reaching higher success rates will require additional research into how to best combine techniques such as DAgger and RL directly from images. Moreover, we found that scaling the RGB dataset to be as large as possible continually improved performance up to the $80k$ trajectories we were able to train on with our compute budget, and we believe obtaining a deeper understanding of scaling laws for this setting will prove valuable. [Xu et al. ([2023](#bib.bib4))].

## 7 Reproducibility Statement

We have made efforts to ensure reproducibility of our results by describing the steps of our data generation and training pipeline (Sec. [3](#S3)), our distillation and transfer pipeline (Sec. [5](#S5)), and our experimental results (Sec. [4](#S4)). Additional ablation studies are provided in the Appendix.

## 8 Acknowledgements

We would like to thank Arhan Jain, Sriyash Poddar, Marius Memmel, and Emma Romig for their invaluable engineering help, and Mateo Guaman Castro for assistance with creative real-robot videos. We also thank Filip Bjelonic, Bingjie Tang, and Iretiayo Akinola for research discussions and advice. Finally, we thank Chuning Zhu, Prashanth Rajan, Arhan Jain, Mateo Guaman Castro, Entong Su, Marius Memmel, David Celis Garcia, and Brenda Potts for their gracious feedback on the website design. Patrick Yin is supported by the National Science Foundation Graduate Research Fellowship (NSF GRFP). Patrick Yin and Tyler Westenbroek have also been supported by National Science Foundation under Grant No. 2212310.

## References

-
I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. (2019)
Solving rubik’s cube with a robot hand.

arXiv preprint arXiv:1910.07113.

Cited by: [§1](#S1.p1.1).

-
L. Ankile, A. Simeonov, I. Shenfeld, and P. Agrawal (2024)
Juicer: data-efficient imitation learning for robotic assembly.

In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

pp. 5096–5103.

Cited by: [§2](#S2.p3.1).

-
M. Bauza, J. E. Chen, V. Dalibard, N. Gileadi, R. Hafner, M. F. Martins, J. Moore, R. Pevceviciute, A. Laurens, D. Rao, et al. (2025)
Demostart: demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots.

In 2025 IEEE International Conference on Robotics and Automation (ICRA),

pp. 6756–6763.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p1.1),
[1st item](#S4.I3.i1.p1.1).

-
M. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos (2016)
Unifying count-based exploration and intrinsic motivation.

Advances in neural information processing systems 29.

Cited by: [§2](#S2.p2.1).

-
S. Bharthulwar, S. Tao, and H. Su (2025)
Staggered environment resets improve massively parallel on-policy reinforcement learning.

External Links: 2511.21011,
[Link](https://arxiv.org/abs/2511.21011)

Cited by: [§2](#S2.p1.1).

-
F. Bjelonic, F. Tischhauser, and M. Hutter (2025)
Towards bridging the gap: systematic sim-to-real transfer for diverse legged robots.

arXiv preprint arXiv:2509.06342.

Cited by: [§A.3.2](#A1.SS3.SSS2.p3.1),
[§5](#S5.p3.1).

-
Y. Burda, H. Edwards, A. Storkey, and O. Klimov (2018)
Exploration by random network distillation.

arXiv preprint arXiv:1810.12894.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p2.1).

-
T. Chen, J. Xu, and P. Agrawal (2021)
A system for general in-hand object re-orientation.

In Conference on Robot Learning, 8-11 November 2021, London, UK, A. Faust, D. Hsu, and G. Neumann (Eds.),

Proceedings of Machine Learning Research, Vol. 164, pp. 297–307.

External Links: [Link](https://proceedings.mlr.press/v164/chen22a.html)

Cited by: [§5](#S5.p1.1).

-
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2025)
Diffusion policy: visuomotor policy learning via action diffusion.

The International Journal of Robotics Research 44 (10-11), pp. 1684–1704.

Cited by: [§5](#S5.p4.1).

-
M. Dennis, N. Jaques, E. Vinitsky, A. Bayen, S. Russell, A. Critch, and S. Levine (2020)
Emergent complexity and zero-shot transfer via unsupervised environment design.

Advances in neural information processing systems 33, pp. 13049–13061.

Cited by: [§2](#S2.p1.1).

-
A. D. Edwards, L. Downs, and J. C. Davidson (2018)
Forward-backward reinforcement learning.

arXiv preprint arXiv:1803.10227.

Cited by: [§2](#S2.p1.1).

-
B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine (2019)
Diversity is all you need: learning skills without a reward function.

In International Conference on Learning Representations (ICLR),

Cited by: [§2](#S2.p2.1).

-
C. Florensa, D. Held, M. Wulfmeier, M. Zhang, and P. Abbeel (2017)
Reverse curriculum generation for reinforcement learning.

In Conference on robot learning,

pp. 482–495.

Cited by: [§2](#S2.p1.1).

-
D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025)
Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948.

Cited by: [§1](#S1.p1.1).

-
T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine (2018)
Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor.

In International conference on machine learning,

pp. 1861–1870.

Cited by: [§1](#S1.p2.1).

-
A. Handa, A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, et al. (2023)
Dextreme: transfer of agile in-hand manipulation from simulation to reality.

In 2023 IEEE International Conference on Robotics and Automation (ICRA),

pp. 5977–5984.

Cited by: [§1](#S1.p2.1).

-
M. Heo, Y. Lee, D. Lee, and J. J. Lim (2023)
Furniturebench: reproducible real-world benchmark for long-horizon complex manipulation.

The International Journal of Robotics Research, pp. 02783649241304789.

Cited by: [§A.4](#A1.SS4.SSS0.Px3.p3.1),
[§4.1](#S4.SS1.p1.5).

-
T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, I. Osband, et al. (2018)
Deep q-learning from demonstrations.

In Proceedings of the AAAI conference on artificial intelligence,

Vol. 32.

Cited by: [§2](#S2.p3.1),
[1st item](#S4.I1.i1.p1.1).

-
J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter (2019)
Learning agile and dynamic motor skills for legged robots.

Science Robotics 4 (26), pp. eaau5872.

Cited by: [§1](#S1.p1.1).

-
B. Ivanovic, J. Harrison, A. Sharma, M. Chen, and M. Pavone (2019)
Barc: backward reachability curriculum for robotic reinforcement learning.

In 2019 International Conference on Robotics and Automation (ICRA),

pp. 15–21.

Cited by: [§2](#S2.p1.1).

-
S. Kakade and J. Langford (2002)
Approximately optimal approximate reinforcement learning.

In Proceedings of the nineteenth international conference on machine learning,

pp. 267–274.

Cited by: [§2](#S2.p1.1).

-
L. Ke, Y. Zhang, A. Deshpande, S. Srinivasa, and A. Gupta (2023)
Ccil: continuity-based data augmentation for corrective imitation learning.

arXiv preprint arXiv:2310.12972.

Cited by: [§2](#S2.p3.1).

-
O. Khatib (1987)
A unified approach for motion and force control of robot manipulators: the operational space formulation.

IEEE Journal of Robotics and Automation 3 (1), pp. 43–53.

External Links: [Document](https://dx.doi.org/10.1109/JRA.1987.1087068)

Cited by: [§A.3.5](#A1.SS3.SSS5.p1.1),
[§5](#S5.p3.1).

-
C. Koprulu, P. Li, T. Qiu, R. Zhao, T. Westenbroek, D. Fridovich-Keil, S. Chinchali, and U. Topcu (2024)
Dense dynamics-aware reward synthesis: integrating prior experience with demonstrations.

arXiv preprint arXiv:2412.01114.

Cited by: [§2](#S2.p3.1).

-
J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter (2020a)
Learning quadrupedal locomotion over challenging terrain.

Science robotics 5 (47), pp. eabc5986.

Cited by: [§1](#S1.p2.1).

-
M. A. Lee, C. Florensa, J. Tremblay, N. Ratliff, A. Garg, F. Ramos, and D. Fox (2020b)
Guided uncertainty-aware policy optimization: combining learning and model-based strategies for sample-efficient policy learning.

In 2020 IEEE international conference on robotics and automation (ICRA),

pp. 7505–7512.

Cited by: [§1](#S1.p2.1).

-
M. A. Lee, C. Florensa, J. Tremblay, N. D. Ratliff, A. Garg, F. Ramos, and D. Fox (2020c)
Guided uncertainty-aware policy optimization: combining learning and model-based strategies for sample-efficient policy learning.

In 2020 IEEE International Conference on Robotics and Automation, ICRA
2020, Paris, France, May 31 - August 31, 2020,

pp. 7505–7512.

External Links: [Link](https://doi.org/10.1109/ICRA40945.2020.9197125),
[Document](https://dx.doi.org/10.1109/ICRA40945.2020.9197125)

Cited by: [§1](#S1.p2.1).

-
A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox (2023)
Mimicgen: a data generation system for scalable robot learning using human demonstrations.

arXiv preprint arXiv:2310.17596.

Cited by: [§2](#S2.p3.1).

-
J. Martin, S. N. Sasikumar, T. Everitt, and M. Hutter (2017)
Count-based exploration in feature space for reinforcement learning.

arXiv preprint arXiv:1706.08090.

Cited by: [§2](#S2.p2.1).

-
M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg (2023)
Orbit: a unified simulation framework for interactive robot learning environments.

IEEE Robotics and Automation Letters 8 (6), pp. 3740–3747.

External Links: [Document](https://dx.doi.org/10.1109/LRA.2023.3270034)

Cited by: [§1](#S1.p1.1),
[§3.2](#S3.SS2.p2.6),
[§5](#S5.p1.1).

-
A. Nair, B. McGrew, M. Andrychowicz, W. Zaremba, and P. Abbeel (2018)
Overcoming exploration in reinforcement learning with demonstrations.

In 2018 IEEE international conference on robotics and automation (ICRA),

pp. 6292–6299.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p3.1).

-
[32]
A. Y. Ng et al.

Algorithms for inverse reinforcement learning..

Cited by: [§1](#S1.p2.1).

-
I. Osband, C. Blundell, A. Pritzel, and B. Van Roy (2016)
Deep exploration via bootstrapped DQN.

In Advances in Neural Information Processing Systems (NeurIPS),

Cited by: [§2](#S2.p2.1).

-
I. Osband, B. V. Roy, D. J. Russo, and Z. Wen (2019)
Deep exploration via randomized value functions.

J. Mach. Learn. Res. 20, pp. 124:1–124:62.

External Links: [Link](https://jmlr.org/papers/v20/18-339.html)

Cited by: [§2](#S2.p2.1).

-
G. Ostrovski, M. G. Bellemare, A. Oord, and R. Munos (2017)
Count-based exploration with neural density models.

In International conference on machine learning,

pp. 2721–2730.

Cited by: [§2](#S2.p2.1).

-
D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell (2017)
Curiosity-driven exploration by self-supervised prediction.

In International conference on machine learning,

pp. 2778–2787.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p2.1).

-
X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne (2018)
Deepmimic: example-guided deep reinforcement learning of physics-based character skills.

ACM Transactions On Graphics (TOG) 37 (4), pp. 1–14.

Cited by: [§1](#S1.p2.1),
[§2](#S2.p3.1),
[1st item](#S4.I2.i1.p1.1).

-
L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel (2017)
Asymmetric actor critic for image-based robot learning.

arXiv preprint arXiv:1710.06542.

Cited by: [§A.3.9](#A1.SS3.SSS9.p1.1),
[§3.3](#S3.SS3.p4.1).

-
A. Raffin, J. Kober, and F. Stulp (2022)
Smooth exploration for robotic reinforcement learning.

In Conference on robot learning,

pp. 1634–1644.

Cited by: [§3.3](#S3.SS3.p5.1).

-
S. Rajeswar, P. Mazzaglia, T. Verbelen, A. Piché, B. Dhoedt, A. C. Courville, and A. Lacoste (2023)
Mastering the unsupervised reinforcement learning benchmark from pixels.

In International Conference on Machine Learning, ICML 2023, 23-29 July
2023, Honolulu, Hawaii, USA, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.),

Proceedings of Machine Learning Research, Vol. 202, pp. 28598–28617.

External Links: [Link](https://proceedings.mlr.press/v202/rajeswar23a.html)

Cited by: [§2](#S2.p2.1).

-
A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine (2017)
Learning complex dexterous manipulation with deep reinforcement learning and demonstrations.

arXiv preprint arXiv:1709.10087.

Cited by: [§2](#S2.p3.1),
[1st item](#S4.I1.i1.p1.1).

-
S. Reddy, A. D. Dragan, and S. Levine (2019)
Sqil: imitation learning via reinforcement learning with sparse rewards.

arXiv preprint arXiv:1905.11108.

Cited by: [§2](#S2.p3.1).

-
C. Resnick, R. Raileanu, S. Kapoor, A. Peysakhovich, K. Cho, and J. Bruna (2018)
Backplay:” man muss immer umkehren”.

arXiv preprint arXiv:1807.06919.

Cited by: [§2](#S2.p1.1).

-
D. Russo, B. V. Roy, A. Kazerouni, I. Osband, and Z. Wen (2018)
A tutorial on thompson sampling.

Found. Trends Mach. Learn. 11 (1), pp. 1–96.

External Links: [Link](https://doi.org/10.1561/2200000070),
[Document](https://dx.doi.org/10.1561/2200000070)

Cited by: [§2](#S2.p2.1).

-
T. Salimans and R. Chen (2018)
Learning montezuma’s revenge from a single demonstration.

arXiv preprint arXiv:1812.03381.

Cited by: [§2](#S2.p1.1).

-
C. Sancaktar, S. Blaes, and G. Martius (2022)
Curious exploration via structured world models yields zero-shot object manipulation.

Advances in Neural Information Processing Systems 35, pp. 24170–24183.

Cited by: [§2](#S2.p2.1).

-
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)
Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347.

Cited by: [§1](#S1.p2.1),
[§1](#S1.p4.1),
[§3.3](#S3.SS3.p1.1),
[1st item](#S4.I3.i1.p1.1).

-
R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. V. Wyk (2025)
DextrAH-rgb: visuomotor policies to grasp anything with dexterous hands.

External Links: 2412.01791,
[Link](https://arxiv.org/abs/2412.01791)

Cited by: [§A.3.11](#A1.SS3.SSS11.p2.1),
[§A.3.7](#A1.SS3.SSS7.p3.1),
[§5](#S5.p2.1).

-
J. Singla, A. Agarwal, and D. Pathak (2024)
Sapg: split and aggregate policy gradients.

arXiv preprint arXiv:2407.20230.

Cited by: [§1](#S1.p2.1).

-
B. Tang, I. Akinola, J. Xu, B. Wen, A. Handa, K. Van Wyk, D. Fox, G. S. Sukhatme, F. Ramos, and Y. Narang (2024)
Automate: specialist and generalist assembly policies over diverse geometries.

arXiv preprint arXiv:2407.08028 1 (2).

Cited by: [§1](#S1.p2.1),
[§2](#S2.p1.1),
[§2](#S2.p3.1).

-
B. Tang, M. A. Lin, I. Akinola, A. Handa, G. S. Sukhatme, F. Ramos, D. Fox, and Y. Narang (2023)
Industreal: transferring contact-rich assembly tasks from simulation to reality.

arXiv preprint arXiv:2305.17110.

Cited by: [§2](#S2.p1.1),
[§3.2](#S3.SS2.p2.6).

-
S. Tao, A. Shukla, T. Chan, and H. Su (2024)
Reverse forward curriculum learning for extreme sample and demonstration efficiency in reinforcement learning.

arXiv preprint arXiv:2405.03379.

Cited by: [§2](#S2.p1.1).

-
E. Todorov, T. Erez, and Y. Tassa (2012)
MuJoCo: a physics engine for model-based control.

In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,

pp. 5026–5033.

Cited by: [§1](#S1.p1.1).

-
M. Vecerik, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Rothörl, T. Lampe, and M. Riedmiller (2017)
Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards.

arXiv preprint arXiv:1707.08817.

Cited by: [§2](#S2.p3.1).

-
T. Westenbroek, F. Castaneda, A. Agrawal, S. Sastry, and K. Sreenath (2022)
Lyapunov design for robust and efficient robotic reinforcement learning.

arXiv preprint arXiv:2208.06721.

Cited by: [§1](#S1.p2.1).

-
Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. (2023)
Unidexgrasp: universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 4737–4746.

Cited by: [§6](#S6.p1.1).

-
Z. Zhou, A. Garg, D. Fox, C. R. Garrett, and A. Mandlekar (2024)
SPIRE: synergistic planning, imitation, and reinforcement learning for long-horizon manipulation.

In Conference on Robot Learning, 6-9 November 2024, Munich, Germany, P. Agrawal, O. Kroemer, and W. Burgard (Eds.),

Proceedings of Machine Learning Research, Vol. 270, pp. 2347–2371.

External Links: [Link](https://proceedings.mlr.press/v270/zhou25a.html)

Cited by: [§1](#S1.p2.1).

## Appendix A Appendix

### A.1 Detailed Reward Specification

We provide the full specification of the task-agnostic reward introduced in Equation [1](#S3.E1). The reward consists of a weighted sum of safety and task-related terms that are shared across all environments, with identical weights and no task-specific tuning.

##### Safety and regularization terms.

To encourage stable and physically plausible behavior, we penalize large actions, rapid changes in actions, and excessive joint velocities:

$$
\begin{aligned}
r_{\text{smooth}}(s_{t},a_{t}) &=-\lambda_{\text{act}}\|a_{t}\|_{2}^{2}-\lambda_{\text{rate}}\|a_{t}-a_{t-1}\|_{2}^{2}-\lambda_{\text{vel}}\|\dot{q}_{t}\|_{2}^{2}. \tag{2}
\end{aligned}
$$

Additionally, we include a large penalty for invalid or unsafe robot states:

$$r_{\text{term}}(s_{t})=-\lambda_{\text{abnormal}}\cdot\mathbf{1}[\text{abnormal state}].$$ \tag{3}

In practice, we use weights $\lambda_{\text{act}}=10^{-4}$, $\lambda_{\text{rate}}=10^{-3}$, $\lambda_{\text{vel}}=10^{-2}$, and $\lambda_{\text{abnormal}}=100$.

##### Task-related terms.

We include several dense reward components that guide the agent toward successful completion of the task.

First, we encourage the end-effector to approach the target object:

$$r_{\text{reach}}(s_{t})=\lambda_{\text{reach}}\left(1-\tanh\left(\frac{\|p_{\text{ee}}-p_{\text{obj}}\|}{\sigma}\right)\right),$$ \tag{4}

where $p_{\text{ee}}$ and $p_{\text{obj}}$ denote the end-effector and target object positions, respectively.

Second, we provide a dense shaping signal based on the relative pose between the manipulated object and the goal:

$$r_{\text{dist}}(s_{t})=\lambda_{\text{dist}}\cdot\frac{1}{2}\left[\exp\left(-\frac{\|x_{\text{err}}\|}{\sigma}\right)+\exp\left(-\frac{\|\theta_{\text{err}}\|}{\sigma}\right)\right],$$ \tag{5}

where $x_{\text{err}}$ and $\theta_{\text{err}}$ denote position and orientation errors computed in the goal frame.

We also include a sparse success reward:

$$r_{\text{success}}(s_{t})=\lambda_{\text{success}}\cdot\mathbf{1}[\text{position and orientation thresholds satisfied}],$$ \tag{6}

which activates when both position and orientation errors fall below predefined thresholds.

##### Implementation details.

The full reward is implemented as a weighted sum of the above terms with fixed coefficients across all tasks. In particular, we use:

| | $$\lambda_{\text{reach}}=0.1,\quad\lambda_{\text{dist}}=0.1,\quad\lambda_{\text{success}}=1.0.$$ | |
|---|---|---|

Notably, all reward components and weights are shared across tasks without modification. We find that performance is largely insensitive to moderate changes in these weights, and that the primary driver of successful learning is the coverage of the initial state distribution rather than reward shaping.

### A.2 Additional Simulation Results

![Figure](2603.15789v2/x8.png)

*Figure 12: Success Rates on Different Stages of Task. We plot the success rates for the Leg Twisting task when starting from states that are in the Near-Goal region and also the Reaching Region of the state space. When evaluating these success rates, we sample resets from the demonstrations used by the baseline algorithms to ensure the resulting policies start from in-distribution states. We see that the baselines can achieve moderate success rates when starting close to the goal (Near Goal), but struggle to make meaningful progress on the full long-horizon task (captured by the reaching resets).*

### A.3 Distillation and Transfer Details

#### A.3.1 Robot Kinematics

Accurate modeling of robot kinematics is a prerequisite for sim-to-real transfer. We use a Universal Robots UR7e for all real-world experiments. However, each physical
UR7e deviates from its nominal model due to manufacturing tolerances in link lengths and joint offsets. As a result, using the default URDF leads to systematic pose errors that accumulate along the kinematic chain.

To address this, we regenerate the URDF using the factory-provided calibration file specific to our robot. This ensures that forward kinematics in simulation closely match the real system. Without this calibration step, we observe consistent end-effector pose misalignment, which significantly degrades performance on precision tasks such as insertion.

For safety, we align joint limits between simulation and the real system, with one exception for the wrist joints. Specifically, we reduce the wrist joint limits from $[-360^{\circ},360^{\circ}]$ to $[-180^{\circ},180^{\circ}]$ in simulation. This prevents the policy from exploiting extreme joint rotations during training, which could lead to the robot hitting joint limits during real-world execution and triggering safety stops.

In addition, we augment the wrist camera mount region with an enlarged, invisible collision geometry in simulation. This acts as a safety buffer to discourage behaviors that bring the wrist-mounted camera into contact with the environment. We find that these conservative modifications improve the reliability of deployment without negatively impacting task performance.

#### A.3.2 Robot Dynamics and System Identification

![Figure](2603.15789v2/figs/sysid_fit_zero.png)

*(a) Without system identification*

![Figure](2603.15789v2/figs/sysid_fit.png)

*(b) With system identification*

Figure 13: Effect of system identification.
Comparison of chirp trajectory fit with and without system identification.

Even with matched kinematics, sim-to-real transfer fails without accurate modeling of robot dynamics and actuation. A key requirement is that the interface between the policy and the robot, which maps low-frequency position outputs to high-frequency joint torques, is identical in simulation and on hardware.

We therefore re-implement the same controller with identical gains in both simulation and real-world execution, ensuring identical computation of torques from end-effector commands. Small implementation details are critical. For example, Jacobian computations must be consistent. We found that default simulator choices, such as computing Jacobians at link centers of mass, can differ from real implementations and lead to subtle but important discrepancies.

After aligning the controller, residual mismatch remains due to unmodeled actuator dynamics. In particular, the UR7e exhibits significant joint friction. We perform system identification of actuator parameters following PACE [(Bjelonic et al., [2025](#bib.bib87))] by executing open-loop trajectories and minimizing the error between simulated and real joint trajectories using CMA-ES.

We identify friction, armature, and motor delay parameters. To ensure sufficient excitation, we use chirp trajectories that span a wide range of velocities. After system identification, we achieve an under 2 degree RMSE in joint space. In contrast, if we use zero-defaults for these parameters, we have approximately a 7 degree RMSE in joint space (see Fig. [13](#A1.F13) for details).

This level of accuracy is sufficient for successful transfer in tasks such as Peg Insertion. However, more sensitive tasks, such as flipping a drawer in the Drawer Insertion task, highlight that even small residual control errors can lead to significant performance degradation and failure. Reducing this control gap between simulation and hardware remains an important direction for future work.

#### A.3.3 Gripper Modeling

We use a Robotiq 2F-85 gripper mounted on the UR7e. Accurate modeling of the gripper is important for transfer, particularly for contact-rich manipulation.

Due to limitations of the physics engine, since PhysX does not support closed kinematic chains, we approximate the gripper using mimic joints in simulation. If the gripper is instead modeled with unconstrained joints, the resulting system behaves as a compliant mechanism, similar to a spring-damper system, which does not match the real hardware. In this case, the policy can exploit unrealistic deformation modes that do not transfer to the real world.

To avoid this issue, we enforce rigid coupling via mimic joints. We use a simplified binary gripper action, open or close, rather than modeling the full internal control loop. While this introduces some mismatch relative to the real system, we find that it is not performance-limiting for our tasks. In practice, careful tuning of gripper force and speed parameters is required to approximately match the effective behavior in simulation.

#### A.3.4 Contact Modeling

We use signed distance fields (SDFs) for contact modeling of objects, which is critical for accurately representing contact geometry in assembly tasks such as threaded insertions. Compared to simpler collision approximations, SDF-based contacts provide more precise surface interactions, which we find important for successful sim-to-real transfer.

We also find that the simulation timestep plays a key role in contact fidelity. Using a simulation timestep of $\texttt{sim\_dt}=1/120$ provides a good balance between computational efficiency and accurate contact modeling for the tasks considered in this work.

Despite these improvements, and the use of extensive randomization (Section [A.3.7](#A1.SS3.SSS7)), certain contact-rich behaviors remain difficult to transfer reliably. For example, strategies that rely on exploiting fine contact interactions, such as reorienting a peg by pushing it against the back of the hole, often fail due to small discrepancies in contact dynamics. Improving contact modeling to better capture such interactions remains an important direction for future work, including exploring more accurate contact models or integrating multiple physics simulation backends.

#### A.3.5 Controller Design

Controller design plays a central role in both policy learning and sim-to-real transfer. We use a torque-level operational space controller [(Khatib, [1987](#bib.bib88))] that tracks a desired end-effector pose.

Given current joint positions $q$ and velocities $\dot{q}$, we compute the end-effector pose error $e\in\mathbb{R}^{6}$, consisting of position and orientation error (represented as axis-angle). The desired task-space force is:

$$F=K_{p}\,e-K_{d}\,\dot{x},$$ \tag{7}

where $\dot{x}=J(q)\dot{q}$ is the end-effector velocity, $J(q)$ is the Jacobian, and $K_{p},K_{d}$ are diagonal stiffness and damping matrices.

Joint torques are then computed via:

$$\tau=J(q)^{\top}F.$$ \tag{8}

We additionally clip the pose error to bound the maximum task-space force, and apply torque limits at execution time:

$$\tau\leftarrow\mathrm{clip}(\tau,-\tau_{\max},\tau_{\max}).$$ \tag{9}

Importantly, we do not apply torque clipping during simulation training, as we found that saturation adversely affects learning performance by distorting the policy’s action-to-dynamics mapping. Instead, torque limits are enforced only during real-world deployment for safety and hardware protection.

In practice, we use separate gains for translational and rotational control. These gains are tuned empirically via teleoperation to achieve smooth, stable, and responsive behavior. For damping, we follow a critical damping heuristic and set:

$$K_{d}=2\sqrt{K_{p}},$$ \tag{10}

which ensures fast convergence without oscillations under a second-order system approximation. We command joint torques to the robot at 500Hz with policy end effector pose commands updating at 10Hz.

In contrast, joint-space PD control or inverse kinematics combined with joint PD results in significantly worse performance. These controllers are more prone to jamming and unstable motion. Empirically, both RL training and distilled policy performance are substantially worse under joint-space control formulations. A deeper investigation into the interaction between controller design and policy learning remains an important direction for future work.

#### A.3.6 Action Space

We use a relative Cartesian action space, where the policy predicts incremental end-effector pose updates. We find that this choice is critical for stable RL training. In particular, using small action magnitudes significantly improves exploration and learning. In contrast, absolute Cartesian actions or large action scales lead to unstable behavior, where the robot exhibits large, wild motions and fails to learn.

During training, we use an action scale of $(0.02,0.02,0.02,0.02,0.02,0.2)$ for $(x,y,z,r_{x},r_{y},r_{z})$. We assign a larger scale to the final rotational dimension to encourage exploration of twisting motions, which are important for tasks such as leg twisting, while remaining beneficial for other tasks. More generally, we find that larger action magnitudes are better tolerated at the wrist, where
motions have more localized effects, whereas large motions at the base joints can lead to unstable global movements.

For sim-to-real transfer, we further reduce the action scale to
$(0.01,0.01,0.002,0.02,0.02,0.2)$. This reduction improves stability and mitigates aggressive motions that can degrade performance on hardware. In particular, we decrease the translational components, especially along the vertical ($z$) axis, to prevent the gripper from applying excessive forces during contact.

To avoid destabilizing the learned policy, we do not switch directly to the smaller action scale. Instead, we apply a curriculum that gradually transitions from the original training scale to the reduced scale, conditioned on task success. This allows the policy to adapt smoothly to the more conservative control regime while preserving previously learned behaviors. Note that this is not a curriculum over behaviors or state distributions, since the policy is already succeeding, but rather a parameter curriculum designed specifically to facilitate stable sim-to-real transfer.

We also note that simply clipping actions to enforce smaller motions is not effective, as it introduces discontinuities that degrade optimization and can cause RL training to collapse. Gradually adjusting the action scale through a curriculum provides a more stable alternative.

*Table 2: Domain randomization and state initialization used during training. Parameters are either sampled directly from distributions or applied as multiplicative scaling to nominal values (denoted by $\times$).*

#### A.3.7 Randomizations

We apply domain randomization across dynamics, actuation, state initialization, and visual observations, as summarized in Table [2](#A1.T2). We find that these randomizations significantly improve sim-to-real transfer and robustness.

For camera setup, we first calibrate extrinsics using ArUco markers in the real environment. We then manually refine alignment by adjusting camera pose and field-of-view (FOV) in simulation to match overlaid real and simulated images. To account for residual calibration error and potential perturbations such as camera movement, we randomize camera pose and focal length within small ranges around the calibrated values.

For visual observations, we follow extensive domain randomization similar to DextrAH-RGB [(Singh et al., [2025](#bib.bib86))], varying lighting, backgrounds, object and robot appearance, and workspace textures. We randomize between 920 high-dynamic range conditions (HDR) for lighting and 957 texture paths for object and gripper fingertip textures. In addition, we apply standard image augmentations including color jitter, blur, and grayscale.

We note that the ranges of randomization were not extensively tuned, and were chosen to provide broad coverage of plausible variations. Future work could study how to more systematically select these ranges, either by narrowing them to improve data
efficiency and alignment with the real system, or by expanding them further to increase diversity and potentially improve robustness.

#### A.3.8 Camera placement

We find that camera placement is critical for successful RGB-based distillation. In particular, using a wrist-mounted camera together with third-person cameras positioned close to the workspace along the $z$-axis provides significantly better performance than top-down viewpoints.

We use two third-person cameras: one capturing the entire tabletop region for grasping, and another focused on the insertion area for contact-rich interactions. This setup helps reduce partial observability by ensuring that both pre-contact and contact phases of the task are well observed.

In practice, we find that a single third-person camera can already achieve reasonable performance. We choose to use both third-person cameras to maximize performance.

#### A.3.9 RL Training and Finetuning

We train RL policies using an asymmetric actor-critic setup [(Pinto et al., [2017](#bib.bib75))], where the policy observes object and robot positions, while the critic has access to privileged information such as object mass and other physical parameters. The policy input consists of a stack of 5 frames at 10 Hz to provide short-term temporal context, whereas the critic does not use any history. We find that this design significantly improves training stability under large-scale domain randomization.

We find that directly training RL policies with system-identified dynamics, particularly high joint friction, leads to poor learning performance. In practice, the policy struggles to explore effectively and training often fails to converge.

To address this, we adopt a staged training procedure. We first train the policy to convergence under simplified dynamics, without system-identified parameters (with zero friction, armature, and motor delay), which enables efficient exploration and stable learning. We then apply a curriculum that gradually transitions the environment dynamics toward the system-identified parameters, conditioned on task success. In conjunction with this transition, we also increase controller gains to ensure that the robot remains responsive under higher friction. Note that this is not a curriculum over behaviors or state distributions, since the policy is already succeeding, but rather a parameter curriculum designed specifically to facilitate stable sim-to-real transfer.

Training the RL policy from scratch takes approximately 8 hours for
Peg Insertion, 16 hours for Leg Twisting, and 24 hours for Drawer Assembly using 4 L40S GPUs. Fine-tuning with the curriculum takes approximately 8 hours on 1 L40S GPU for Peg Insertion, 24 hours on 4 L40S GPUs for Leg Twisting, and 24 hours on 1 L40S GPU for Drawer Assembly.

#### A.3.10 Seed Selection

![Figure](2603.15789v2/x9.png)

*(a) Behavior which doesn’t transfer well. We show an example of a behavior which doesn’t transfer well, where the policy reorient the peg by dropping it above the hole and re-grasping it.*

We observe that policies trained with different random seeds can exhibit
substantially different behaviors, with some transferring more reliably to the real world than others. For example, in the peg insertion task, certain policies learn to reorient the peg by dropping it above the hole and re-grasping it as shown in Fig. [14(a)](#A1.F14.sf1). While effective in simulation, this strategy requires precise timing and accurate gripper behavior, making it sensitive to sim-to-real discrepancies and prone to failure on hardware.

To mitigate this, we train multiple policies with different random seeds and select among them using an offline proxy metric. Specifically, we evaluate each policy under injected action noise and measure its success rate. We then select the policy with the highest performance under noise. This metric serves as a proxy for robustness to control error and sim-to-real mismatch, and we find that it correlates well with real-world performance.

#### A.3.11 Student-Teacher Distillation

We collect 80K trajectories from a state-based expert under extensive randomization (Section [A.3.7](#A1.SS3.SSS7)) and use them to train a visuomotor policy. The policy consists of an ImageNet-pretrained ResNet-18 encoder followed by a 4-layer MLP with 512 hidden units, trained via supervised learning. The policy takes as input a stack of 5 consecutive frames and predicts the mean and standard deviation of a Gaussian action distribution.

Following DextrAH-RGB [(Singh et al., [2025](#bib.bib86))], we find that combining pose reconstruction with KL matching to the expert’s action distribution leads to the best distillation and transfer performance. In particular, our most performant policies are trained with a KL loss and deployed only using the mean action. Without pose reconstruction, the RGB policy sometimes fails to reliably localize and grasp objects. We also find that end-to-end training, with gradients propagated through the visual encoder, is critical; freezing the encoder significantly degrades performance.

We experiment with alternative representations and architectures, including DINO features and fine-tuning larger pretrained models such as Pi-0.5, but do not observe significant performance improvements. These approaches substantially increase training cost, suggesting that larger models may only provide benefits at larger data scales or across more diverse tasks.

We also evaluate action chunking and diffusion-based policies, but find that they underperform simple MLP policies. We hypothesize that this is due to the high-frequency and reactive nature of the expert RL policy, particularly in contact-rich assembly tasks where small corrective motions and jittering are important.

In terms of data scaling, we observe that using 10K–80K trajectories yields similar performance in simulation, but real-world transfer improves significantly with larger datasets. Training for a sufficient number of iterations is also important; we train for approximately 350K iterations (about 2 days on a single H200 GPU). Collecting 80K trajectories requires approximately 24 GPU hours on a 3090 GPU. Further study is needed to better understand the relationship between
dataset size, training time, and transfer performance.

Despite these efforts, RGB policy performance in simulation remains limited (approximately 50% success rate), indicating a gap between imitation learning and robust deployment. This suggests that incorporating online adaptation methods in simulation, such as image-based DAgger or RL fine-tuning, may be necessary to achieve higher performance. We leave this to future work.

#### A.3.12 Inference Speed

We deploy the policy at 10 Hz using a non-blocking control pipeline. Sensor data is collected asynchronously at their native frequencies. At each control step, the policy retrieves the most recent observation from each sensor that precedes the current timestep, performs inference, and sends the resulting action to the robot.

This design enables smooth and responsive motion, but introduces latency due to sensor delay, data processing, and policy inference. Our setup uses a RealSense D415 wrist camera, along with D435 and D455 third-person cameras, all streamed over USB 3.2. We measure approximately 30 ms of sensor-to-PC delay and an additional 5 ms for
processing. Policy inference using a ResNet18-MLP architecture takes approximately 5 ms on an NVIDIA RTX 4090 GPU, resulting in a total end-to-end latency of roughly 40 ms.

Despite not incorporating explicit latency randomization during training, we find that policies remain smooth and robust under this level of delay. However, fast inference speed is critical. When we artificially increase inference latency to 60 ms, policies exhibit significant jitter and performance degrades substantially.

These results highlight the importance of low-latency inference for stable control and suggest that robustness to delay remains an important direction for future work.

### A.4 Real World Evaluation Details

##### Evaluation Setup.

We deploy the policy at 10 Hz, matching the control frequency used during simulation training. At the start of evaluation, the robot joints are reset to a fixed initial configuration; no further resets are performed, and the policy continuously rolls out in the environment.

To improve robustness, we implement a simple stuck-detection mechanism. If the robot’s joint positions exhibit negligible movement over a 2-second window (maximum joint displacement below 0.002 radians across all joints), we override the policy by commanding the gripper to open for 1 second. This helps the robot recover from situations such as failed grasps or unfavorable contact configurations. Triggering this mechanism is not counted as a failure in our evaluation.

With this mechanism, the robot operates continuously without manual resets. We define a trajectory as a failure only if human intervention is required, such as repositioning the robot or the object. This typically occurs when the robot becomes stuck or takes excessively long to complete the task.

##### Initialization Distribution.

For evaluation, we sample initial object configurations by uniformly randomizing their positions on the table. We intentionally include challenging setups, such as objects in contact that require separation, stacked objects, or objects placed far from the workspace center, for example near the robot base, to stress test the policy’s capabilities.

##### Physical Setup.

Task success also depends on aspects of the physical setup. We use a compliant silicone mat on the workspace, which facilitates manipulation by allowing slight deformation during contact, particularly for reorientation and flipping behaviors. The mat is secured to the table using command strips to prevent motion.

For the Peg Insertion and Leg Twisting tasks, we additionally fix the receptive objects, such as the peg hole and tabletop, to the workspace using command strips. This is important because, in simulation, these objects are static, and the policy learns to reorient the insertive object relative to a fixed reference. Without securing them, the receptive objects may move during interaction, leading to failures. If the robot applies sufficient force to dislodge these fixtures during evaluation, we exclude the corresponding trajectory from success rate calculations.

For the Drawer Assembly task, directly fixing the drawer is insufficient due to larger interaction forces. Instead, we secure an L-shaped 3D-printed bracket behind the drawer box, following [(Heo et al., [2023](#bib.bib84))], which provides a more stable constraint. We find that the policy largely ignores this additional structure, likely due to domain randomization during training. Additionally, to address height misalignment caused by the drawer lip, we use layered mats with a cutout to align the drawer bottom with the workspace surface, which significantly improves insertion success. Reducing reliance on such environment modifications is an important direction for future work, for example through the use of bimanual manipulation.

For all tasks, we mount a stage light above the workspace to ensure consistent lighting conditions throughout the day.

##### Metrics and Evaluation Duration.

We continuously run the policy for approximately 25 minutes for the peg and drawer tasks, and 50 minutes for the leg task, collecting 41, 52, and 55 trajectories, respectively.

We define a first-try success as completing the task without re-grasping or re-inserting the object (e.g., dropping the object and attempting again counts as a retry). Throughput is computed as the total number of successful trajectories divided by the total evaluation time.



Experimental support, please
[view the build logs](./2603.15789v2/__stdout.txt)
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