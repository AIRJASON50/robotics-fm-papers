##### Report GitHub Issue

**×




Title:



Content selection saved. Describe the issue below:

Description:




Submit without GitHub
Submit in GitHub





[Back to arXiv](/)





[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)


arXiv:2603.12243v2 [cs.RO] 15 Mar 2026

[# HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies Amber Xie1, Haozhi Qi2, Dorsa Sadigh1 1Stanford University2Amazon FAR (Frontier AI & Robotics) ###### Abstract Mastering dexterous manipulation with multi-fingered hands has been a grand challenge in robotics for decades. Despite its potential, the difficulty of collecting high-quality data remains a primary bottleneck for high-precision tasks. While reinforcement learning and simulation-to-real-world (sim-to-real) transfer offer a promising alternative, the transferred policies often fail for tasks requiring millimeter-scale precision, such as bimanual piano playing. In this work, we introduce HandelBot, a framework that combines a simulation policy and rapid adaptation through a two-stage pipeline. Starting from a simulation-trained policy, we first apply a structured refinement stage to correct spatial alignments by adjusting lateral finger joints based on physical rollouts. Next, we use residual reinforcement learning to autonomously learn fine-grained corrective actions. Through extensive hardware experiments across five recognized songs, we demonstrate that HandelBot can successfully perform precise bimanual piano playing. Our system outperforms direct simulation deployment by a factor of 1.8$\times$ and requires only 30 minutes of physical interaction data. Code and more videos are available at https://amberxie88.github.io/handelbot](https://amberxie88.github.io/handelbot).

## I INTRODUCTION

Learning complex dexterous behaviors with multi-fingered hands remains a central challenge in robotics. Tasks such as piano playing represent a grand milestone in this domain, as they require precise, coordinated finger motions, fine contact timing, and long-horizon control. While recent advances have enabled impressive dexterous manipulation in simulation [[[60](#bib.bib1)]], real-world piano playing with robotic hands pushes far beyond what current systems can reliably demonstrate.

To tackle these complex dexterous tasks, imitation learning has emerged as a promising paradigm for achieving precise real-world control, but it relies heavily on large-scale, high-quality data [[[26](#bib.bib18), [4](#bib.bib19)]]. While teleoperation is a popular method for collecting this data, controlling high-DoF robot hands is cumbersome and lacks scalability. More importantly, it is often completely infeasible for tasks requiring rapid, independent finger motions, such as piano playing. Alternatively, learning directly from human data mitigates scalability issues but introduces a substantial embodiment gap, a problem heavily amplified by the spatial and temporal precision needed for this task. Reinforcement learning (RL) in simulation bypasses these issues entirely and has driven recent progress in simulated piano playing [[[60](#bib.bib1)]]. However, relying solely on simulation introduces a sim-to-real gap that becomes highly problematic when millimeter-scale errors inevitably lead to task failure.

To address these challenges, we propose Hand-elBot (inspired by Baroque composer George Frideric Handel), a reinforcement learning framework designed to bridge the gap between simulation and the real world. Our core insight is that while simulation struggles to model subtle contact dynamics, it is highly effective at providing a strong structural prior for finger coordination. Building on this, we decompose the learning process into a hybrid pipeline. We first utilize simulation to acquire a base policy of coarse motor movements. Next, we deploy this policy in the real world, refining it through structured lateral joint adjustments and residual reinforcement learning. By incorporating a small amount of high-quality physical interaction data, our policy efficiently adapts to the real world.

Specifically, we first train a policy in simulation. This achieves strong performance in the simulated environment but only gives limited success on real hardware due to the sim-to-real gap. To address this, we apply simple, human-defined heuristics that refine the physical trajectory to better match the expected performance. These heuristics exploit human priors of keyboard geometry and hand kinematics to correct consistent lateral biases and contact misalignments. By comparing the desired key presses to the actual keys pressed during a real-world rollout, we iteratively adjust the lateral joints of the fingers to align them horizontally with their targets. We subsequently refined policy via residual reinforcement learning, which learns corrective actions on top. This residual formulation enables safer exploration and effectively bridges the sim-to-real gap.

We extensively evaluate HandelBot on a bimanual robot setup across a diverse suite of five songs. Our experiments demonstrate that directly deploying simulation policies struggles to accurately execute the tasks. In contrast, HandelBot consistently achieves the highest F1 scores across all evaluated musical pieces. We find that the initial policy refinement effectively aligns finger presses with target keys. Furthermore, the subsequent residual reinforcement learning stage significantly boosts performance by addressing errors and adapting to the physical dynamics.

To summarize, our contributions are as follows:

-
1.

To our knowledge, we present the first learning-based system capable of real-world, two-handed piano playing. We comprehensively evaluate this system across a diverse suite of five recognized songs.

-
2.

We propose a novel two-stage method for bridging the sim-to-real gap. This approach first refines a simulation-trained policy using physical rollouts, and subsequently utilizes real-world residual reinforcement learning to learn fine-grained corrective actions.

-
3.

We demonstrate that HandelBot outperforms direct sim-to-real deployment by a factor of 1.8x, with as little as 30 minutes of real-world interaction data.

## II RELATED WORK

Dexterous Manipulation. Dexterous manipulation with multi-fingered robot hands offers the potential to achieve truly human-like motor skills. A promising approach to learning such skills is through imitation over teleoperated demonstrations [[[55](#bib.bib20), [64](#bib.bib21), [45](#bib.bib22), [32](#bib.bib23), [6](#bib.bib24), [3](#bib.bib25), [21](#bib.bib26), [43](#bib.bib27), [14](#bib.bib28), [7](#bib.bib29)]], similar to recent advances in robot learning for parallel jaw grippers [[[4](#bib.bib19), [67](#bib.bib17), [26](#bib.bib18), [28](#bib.bib57), [9](#bib.bib55), [10](#bib.bib56), [20](#bib.bib59), [38](#bib.bib58), [37](#bib.bib65), [53](#bib.bib60)]]. However, teleoperation is challenging for these high degree-of-freedom embodiments and lacks scalability. To tackle this, many works leverage direct human data [[[47](#bib.bib30), [13](#bib.bib31), [42](#bib.bib32), [23](#bib.bib33), [49](#bib.bib34), [58](#bib.bib54), [33](#bib.bib61)]]; yet, this introduces a substantial embodiment gap between human and robot morphology. An alternative paradigm trains policies entirely in simulation [[[39](#bib.bib37), [57](#bib.bib38), [40](#bib.bib35), [25](#bib.bib62), [5](#bib.bib63), [34](#bib.bib64)]], which reduces the data collection burden but inevitably introduces a sim-to-real gap.
When zero-shot sim-to-real does not lead to satisfactory task success, many methods leverage small amounts of real-world data for policy improvement [[[50](#bib.bib36), [15](#bib.bib53)]].

Robotic Piano Playing. Early robotic piano systems relied on specialized hardware and handcrafted controllers designed explicitly for keyboard actuation [[[24](#bib.bib41), [31](#bib.bib42), [48](#bib.bib50), [19](#bib.bib43), [2](#bib.bib44), [63](#bib.bib45), [62](#bib.bib47), [30](#bib.bib46), [52](#bib.bib67)]]. Recent learning-based approaches have shown dexterous piano playing using general-purpose, non-specialized robotic hardware. These approaches leverage data-driven methods, including large-scale human motion capture [[[51](#bib.bib48)]], video-based kinematic retargeting [[[41](#bib.bib39)]], and reinforcement learning in simulation [[[54](#bib.bib49), [60](#bib.bib1)]].

Despite these advances, sim-to-real transfer for physical piano playing remains largely underexplored. While recent work [[[61](#bib.bib16)]] has demonstrated hybrid transfer for unimanual performance, we introduce a bimanual framework that couples simulation pretraining with real-world residual reinforcement learning, enabling complex, two-handed piano playing on non-specialized hardware.

Real-World Reinforcement Learning. Real-world reinforcement learning holds the potential of autonomous learning under real-world dynamics and environments. However, it remains challenging due to real-world reward functions, environment resets, safety and more. Prior work has shown that real-world RL from scratch is possible with careful design choices [[[46](#bib.bib2), [17](#bib.bib3), [12](#bib.bib4), [35](#bib.bib15)]]. To address challenges with sample inefficiency and exploration in high-dimensional action spaces, prior works bootstrap with prior demonstrations [[[16](#bib.bib5), [65](#bib.bib6), [29](#bib.bib68)]], multi-task or offline data [[[18](#bib.bib7), [68](#bib.bib14)]], or large pretrained models [[[56](#bib.bib12), [36](#bib.bib13)]]. In our work, we use an open-loop rollout learned from RL in sim, providing a strong initialization and reducing the exploration space.

Our specific direction of real-world RL focuses on residual reinforcement learning [[[22](#bib.bib11)]], which mitigates policy collapse and instabilities by keeping a frozen base policy. Prior work has explored residual policies for adaptation and exploration [[[59](#bib.bib8), [1](#bib.bib9), [66](#bib.bib10)]], generally assuming a strong pretrained policy that should be preserved. Our work is similar, as we learn a residual policy over an open-loop policy or trajectory. However, our approach does not assume access to expert demonstrations, which are difficult to obtain for piano playing, or closed-loop base policies. Our method enables efficient real-world learning to bridge the sim-to-real gap for a highly precise dexterous task.

![Figure](2603.12243v2/x1.png)

*Figure 1: HandelBot Method (0) RL in Sim. We leverage fast, parallel simulators for reinforcement learning. This leads to a coarse base policy, $\pi_{sim}$, from which we extract an open-loop rollout, $\tau_{sim}$. (1) Policy Refinement. Second, we refine $\tau_{sim}$, yielding $\tau^{*}_{sim}$. We use real-world updates to iteratively update the lateral joints of the fingers, moving the finger horizontally in the direction of the keys it is intended to press. (2) Residual RL. We perform residual RL atop $\tau_{sim}$, using the keyboard’s MIDI output as a reward. This allows us to further update our policy for better piano playing.*

## III Real-World Piano Playing

To adapt a simulation-trained policy for real-world piano playing, HandelBot follows a two-phase process. First, we apply a structured policy refinement step to align the execution trajectory with the desired key presses, drawing inspiration from prior work [[[41](#bib.bib39)]]. This alone still fails to reach the level of precision needed for piano playing. Second, HandelBot uses residual reinforcement learning on top of the refined trajectory. The residual policy learns corrective action perturbations using real-world rewards. An overview of our approach is shown in Figure [1](#S2.F1).

### III-A Problem Statement

We model robot piano playing as a Markov Decision Process (MDP), specified by the tuple $(\mathcal{O},\mathcal{A},P,r,\gamma)$. At each discrete time step $t$, the agent observes a state $o_{t}\in\mathcal{O}$, selects an action $a_{t}\in\mathcal{A}$ according to a policy $\pi(a_{t}\mid o_{t})$, and transitions to a new state $o_{t+1}\sim P(\cdot\mid o_{t},a_{t})$ while receiving a scalar reward $r(o_{t})$. We describe how we model robot piano playing as this MDP below in [section III-B](#S3.SS2).

To learn our policy $\pi$, we use reinforcement learning, where $\pi$ is trained to maximize the expected discounted return:

| | $$\mathbb{E}_{\pi}\left[\sum_{t=0}^{T}\gamma^{t}r(o_{t},a_{t})\right],$$ | |
|---|---|---|

where $\gamma\in(0,1]$ is a discount factor and $T$ denotes the episode horizon.

### III-B Reinforcement Learning in Simulation

We first train a policy in a simulated environment to learn core piano-playing behaviors. We use RL, which leverages fast, parallel rollouts and dense reward signals, both of which are difficult to obtain in the real world.

##### Reward Design

The simulated reward function largely follows the design of RoboPianist [[[60](#bib.bib1)]], and consists of a key press reward that rewards playing the target notes; a dense fingering reward for being near the correct key; and an energy penalty.

##### Observations and Actions

Following prior work [[[60](#bib.bib1)]], the observation space includes robot proprioception, current piano activation, goal piano activations, and active fingers. The action space consists of delta joint positions, representing low-level control commands for the robotic hand. We script the end-effector based on physical location of keys to be pressed. We denote the resulting simulation policy as $\pi_{\text{sim}}$.

While $\pi_{\text{sim}}$ achieves strong performance in simulation, direct deployment in the real world leads to degraded performance due to mismatches in the controller and piano key-pressing dynamics.

### III-C Policy Refinement

Before running residual reinforcement learning, we first apply a lightweight policy refinement procedure in the real world. This is done by exploiting domain knowledge about keyboard geometry and hand kinematics to correct consistent lateral biases and contact misalignments. This can be viewed as a structured, system-identification-inspired trajectory refinement procedure. Empirically, this stage substantially improves key accuracy and provides a stronger initialization for subsequent real-world residual reinforcement learning.

The goal of this step is to iteratively correct systematic key-pressing errors in an open-loop rollout of $\pi_{\text{sim}}$ without requiring additional learning. We denote the initial open-loop trajectory from $\pi_{\text{sim}}$ as $\tau^{0}=(s_{0}^{0},...,s_{T}^{0})$, where $s$ are the target joint states for timesteps $0,...,T$. Concretely, we would like to produce $\tau^{*}=(s_{0}^{*},...,s_{T}^{*})$, where $\tau^{*}$ denotes our corrected, improved trajectory.

##### Lateral Joint Correction

In our embodiment, each finger is controlled by 4 joints, with a primary lateral joint responsible for horizontal motion and 3 others used for vertical actuation. After executing $\tau^{0}$ on the real piano, we compare desired keys presses to actual key presses, and use this information to adjust the lateral joint of the finger. For example, if the finger plays a lower-pitched note than the desired key, the finger should be adjusted to the right.

Specifically, we first execute the simulation-trained policy $\pi_{\text{sim}}$ in an open-loop manner on the real robot and record, at each time step $t$, (i) the target note and finger designated to press the key, and (ii) the set of keys actually pressed by each finger. Let $k_{t}^{\text{target}}$ denote the intended key index for a given finger, and let $\mathcal{K}_{t}^{\text{press}}$ denote the indices of keys that were physically activated.

For each finger, we identify the pressed key closest to the target. If the closest key $k_{t}^{\text{press}}$ differs from $k_{t}^{\text{target}}$, we compute a signed directional error:

| | $$\Delta_{t}=\begin{cases}+\delta&\text{if }k_{t}^{\text{press}}k_{t}^{\text{target}},\\ 0&\text{otherwise},\end{cases}$$ | |
|---|---|---|

where $\delta$ is a step size controlling the amount of adjustment to the lateral finger joint. Then, we apply an update function $f:(\tau_{0},\Delta_{t})\mapsto\tau_{1}$ to obtain our updated trajectory, which we describe below in Chunked Updates.

##### Iterative Updates

We apply this correction procedure iteratively, where we alternate between executing the current trajectory and updating it. During these iterations, we must determine how much to update the lateral joint, i.e. the value of $\delta$. We initialize $\delta$ to a large value, and after every iteration, we anneal $\delta$, to help avoid oscillation and allow for smoother convergence to an improved trajectory.

In practice, there are a few additional details that improve this process. First, we also introduce smaller corrective terms, $0.3\Delta_{t}$, to neighboring fingers, encouraging spatial separation between adjacent fingers and reducing self-collisions. Second, there may be many $k_{t}^{target}$ at each step, and therefore many $\Delta_{t}$ at each timestep $t$. For simplicity of notation, we previously defined $f$ under the assumption that there is only one key press and one potential lateral movement at each timestep. In reality, we calculate $\Delta_{t}$ for each active key press, and apply the correction to each lateral joint. Finally, when there are multiple key presses, we must also determine which key is pressed by which finger. In this case, we assume, based on finger and piano kinematics, that within $\mathcal{K}_{t}^{\text{press}}$, the active finger to the left is pressing the lower keys, and the active finger to the right is pressing the higher keys.

##### Chunked Updates

Previously, we described our update function $f:(\tau_{0},\Delta_{t})\mapsto\tau_{1}$ as operating over each state in the open-loop rollout. However, in practice, we perform updates over temporal chunks for motion smoothness. Intuitively, this allows us to incorporate the temporal context of a segment; for instance, if a finger must move rightward to strike a key, the corrective residual should initiate a preparatory lateral movement during the approach phase, rather than a discrete correction at the moment of contact.

Specifically, we divide the trajectory into sub-chunks of length $K$. Instead of computing $\Delta_{t}$ for each $s_{t}^{i}$ at iteration $i$, we instead compute $\Delta_{t}^{\text{chunk}}$ for each chunk $s_{t}^{i},...,s_{t+K}^{i}$. To do so, we calculate $\Delta_{t},...,\Delta_{t+K+L}$, with lookahead $L$. Notice here that for a chunk from $t$ to $t+K$, we consider fingertip errors all the way to $t+K+L$, facilitating anticipatory spatial adjustments. To extract our $\Delta_{t}^{\text{chunk}}$ which we will apply to our entire chunk, we use $\Delta_{t}^{\text{chunk}}=\frac{1}{K+L}\sum_{j=t}^{t+k+L}\Delta_{j}$.

At the end of this iterative process, we save the trajectory with the best F1 score as our refined trajectory.

### III-D Real-World Residual Reinforcement Learning

To address the sim-to-real gap, we adopt a residual reinforcement learning framework that fine-tunes the open-loop trajectory $s^{*}_{0},...,s^{*}_{T}$ from the policy refinement step.

##### Residual Policy Formulation

We introduce a residual policy $\pi_{\text{res}}$ that outputs an additive correction to the base action:

| | $$\hat{s}_{t+1}=\pi_{\text{res}}(o_{t})+s^{*}_{t+1},$$ | |
|---|---|---|

where $o_{t}$ denotes the real-world observation at time $t$, $s^{*}_{t+1}$ denotes the next state from the open-loop trajectory, and $\hat{s}_{t+1}$ incorporates the residual action. Note here that $\hat{s}_{t+1}$ is roughly equivalent to $a_{t}$: because we are commanding absolute joint positions of the hands, our actions are simply the next target joint states.

The residual action space is intentionally constrained to small perturbations, enabling safer exploration and faster learning compared to learning a full policy from scratch.

##### Residual RL Objective

In the real world, we rely exclusively a reward signal derived from the piano’s MIDI output. Our reward is simply the key press reward, identical to the one used in simulation.

The residual policy $\pi_{\text{res}}$ is trained using reinforcement learning to maximize the expected return under the real-world dynamics. By learning only corrective action deltas, residual reinforcement learning effectively compensates for simulation inaccuracies while maintaining the structured behavior learned in simulation. This approach enables efficient and stable adaptation, resulting in robust real-world piano-playing performance.

![Figure](2603.12243v2/x2.png)

*Figure 2: Hardware Setup. We use a MIDI keyboard, two Tesollo DG-5F hands, and two Franka arms for piano playing. We use the MIDI output from the piano, which tells us which notes are pressed, in order to calculate rewards. We emphasize that the robot hands are far larger than the average human hand, thus making piano playing difficult. Finally, for RL training, we include a collision checker which prevents fingers from pressing down beyond the keys.*

![Figure](2603.12243v2/x3.png)

*Figure 3: Main Results. We include F1 score, multiplied by 100, for 5 songs. HandelBot consistently achieves the strongest F1 score, showing the importance of effectively using real-world samples to accomplish precise, dexteorus piano-playing. Methods only using simulated data, such as $\pi_{sim}$ (CL) and $\pi_{sim}$, have weak performance due to the sim-to-real gap.*

##### Guided Noise

We choose TD3 [[[8](#bib.bib52)]] as our RL algorithm, which adds a noising term to the sampled action from the actor. Concretely, for each environment step, the actor computes $a=\mu_{\theta}(o)+\text{clip}(\epsilon,-0.5,0.5)$, where $\epsilon\sim N(0,1)$. Motivated by the lateral adjustment used in policy refinement, we also adjust the noise in the direction of the correct lateral movement. We thus apply $a=\mu_{\theta}(o)+\text{clip}(\hat{\epsilon},-0.5,0.5)$, where $\hat{\epsilon}$ is a modification of $\epsilon$ as follows:

First, for any keys pressed, we calculate the signed directional error $\Delta_{t}$ for the current timestep, as we did in policy refinement. With probability $\Pr(\text{guided noise})=0.5$, we change the sign of the noise at that lateral joint to be the same sign as $\Delta_{t}$. This produces $\hat{\epsilon}$, where $||\hat{\epsilon}||_{2}=||\epsilon||_{2}$. Note that our only modification is the sign of the noise term for the action indices corresponding to the lateral joints. Then, if guided noise is sampled, the final action our actor takes is $a=\mu_{\theta}(o)+\text{clip}(\hat{\epsilon},-0.5,0.5)$.

This is a lightweight heuristic that guides exploration to hitting the correct keys. In practice, we do not find this to be an important hyperparameter, as explored in [table II](#S4.T2).

## IV Experiments

### IV-A Experimental Setup

#### IV-A1 Hardware Platform

Our physical system consists of two Tesollo DG-5F hands mounted on a Franka Emika Panda arm and a FR3 arm. To better emulate human piano-playing posture, we design a custom 3D-printed mount so the the fingers are parallel to the keyboard surface. The arms follow a script end-effector pose trajectories, and the hands are controlled with joint control.

#### IV-A2 Simulation Environment

We use ManiSkill [[[11](#bib.bib51)]], a fast, parallelizable simulator, for reinforcement learning in simulation. For each trained policy, we select the trajectory achieving the highest validation F1 score and treat it as the nominal open-loop solution for real-world deployment. This selection protocol ensures that sim-to-real transfer begins from the strongest available simulation behavior.

#### IV-A3 Real-World Deployment and Safety

Direct execution of simulated trajectories on hardware introduces safety risks and control instability. We therefore augment deployment with a real-world safety layer. Given desired joint targets, we solve a constrained inverse kinematics problem using PyRoki [[[27](#bib.bib40)]] to produce feasible configurations while penalizing self-collisions and contact with the piano surface, which we approximate as a planar constraint. To improve motion smoothness, policy actions are produced at 10Hz and linearly interpolated to 80Hz before being sent to the hands. This upsampling reduces jerk and high-frequency oscillations that are particularly detrimental in piano playing. The arms are controlled using the Polymetis controller at 100Hz to ensure stable end-effector tracking.

![Figure](2603.12243v2/x4.png)

*Figure 4: Visualization of HandelBot Trajectories. Per each song, we visualize the notes pressed correctly, pressed incorrectly, and missed. The x axis is the timestep of the song, and the y axis are the different notes, with the top half representing keys for the right hand, and the bottom for the left hand. Across easier songs such as Twinkle Twinkle and Ode to Joy, we find that HandelBot makes few mistakes, with occasional timing errors or wrong presses. For harder songs such as Fur Elise, large jumps in the left hand notes (bottom section of each song plot) are challenging for the left hand.*

For residual reinforcement learning, we restrict the residual actions to only the three active fingers (index, middle, ring), with total dimensionality 9 per hand. Each hand is trained as an independent agent, which reduces action dimensionality and simplifies credit assignment during adaptation.

#### IV-A4 Musical Tasks

We evaluate on 5 widely recognized songs with varying song lengths: Twinkle Twinkle (160 timesteps; 16 seconds), Ode to Joy (330 timesteps; 33 sec), Hot Cross Buns (160 timesteps; 16 sec), Fur Elise by Beethoven (320 timesteps; 32 sec), Prelude in C by Bach (330 timesteps; 33 sec). Due to physical reach and lateral dexterity constraints for the thumb and pinky, we modify the fingerings for the song to three fingers per hand. To prevent inter-arm collisions, we arrange left and right hand parts to be separated by multiple octaves.

#### IV-A5 Evaluation Protocol

Following prior work [[[60](#bib.bib1)]], we measure policy performance using the F1 score. For all methods, we report the mean F1 score across 5 rollouts. For reinforcement learning methods, policies are evaluated with the mean of 5 rollouts, after every 20 trajectories during training, and we report the maximum validation F1 score achieved over the course of learning. RL methods are trained for 100 trajectories. This roughly corresponds to 30k environment interactions in 1 hour for Ode to Joy, Prelude in C, and Fur Elise; and roughly 16k environment interactions in 30 minutes for Twinkle Twinkle and Hot Cross Buns.

#### IV-A6 Baselines

We compare against a comprehensive set of baselines to isolate the contribution of each component in our method. We use TD3 for all RL methods [[[8](#bib.bib52)]].

-
•

$\bm{\pi_{sim}}$(CL): policy trained in simulation with domain randomization, with closed-loop inference.

-
•

RL from Scratch: a real-world policy from random initialization, using scripted end-effector motion but learning finger control entirely on hardware with real-world dynamics.

-
•

$\bm{\pi_{sim}}$: open-loop policy from sim.

-
•

$\bm{\pi_{sim}}$+ ResRL: residual RL over $\pi_{sim}$.

-
•

HandelBot w/o ResRL: policy refinement over $\pi_{sim}$.

-
•

HandelBot (Ours): policy refinement and residual reinforcement learning, yielding our full system.

### IV-B Can HandelBot Accomplish Real-World Piano Playing?

In [fig. 3](#S3.F3), we examine the performance across a suite of 5 songs. First, we find that real data consistently boosts performance of piano-playing. For example, real-world RL from scratch, which leverages no prior pretraining, is able to achieve strong performance for many songs, matching or outperforming $\pi_{sim}$. Applying residual RL over $\pi_{sim}$ also leads to consistent improvements. In contrast, methods using no real data, like $\pi_{sim}$ (CL) and $\pi_{sim}$, have the lowest F1 scores. Notably, $\pi_{sim}$ (CL) performs much worse than open-loop $\pi_{sim}$, which we hypothesize is because of the dynamics gap and compounding errors across the trajectory. Qualitatively, we find that $\pi_{sim}$ and $\pi_{sim}$ (CL) often overshoot past the target key, and often may press down forcefully and get stuck on a key, issues that using real data addresses.

*TABLE I: We report F1 scores x 100 for the closed-loop $\pi_{sim}$, which runs inference over real-world inputs, with hybrid execution $\pi_{sim}$. Hybrid execution runs a simulation in parallel with the real-world, and uses proprioception from the parallel sim instead of the real-world.*

*TABLE II: Ablations. By default, HandelBot uses $\gamma=0.8$ and $\Pr(\text{guided noise})=0.5$.*

Across all songs, HandelBot is the strongest performing method, leveraging both policy refinement and residual RL for enhanced performance. We find policy refinement to be effective, as it directly aligns finger presses with the correct target keys. However, policy refinement cannot address many issues, which is why residual RL is still necessary. First, we only adjust the lateral joint during refinement, meaning that missed presses cannot be adjusted. Second, refinement makes a series of assumptions, such as which finger is pressing which key, and this may not always be accurate. This is addressed by learning the end-to-end residual policy, which leads to a boost in performance across most songs.

### IV-C What Factors are Important for Real-World Residual RL?

Next, we examine the effectiveness of real-world RL. First, we evaluate how initializing RL with a pretrained trajectory impacts performance. We include 3 variants: RL-Scratch, which learns the entire hand policy without prior pretraining; $\pi_{sim}$ + ResRL, which learns residuals over $\pi_{sim}$; and HandelBot, which learns residuals over a refined trajectory. Across all songs, we find that residual RL over a strong trajectory ($\pi_{sim}$ with policy refinement $>$ $\pi_{sim}$ $>$ no initialization) leads to higher F1 scores. We hypothesize that a refined policy reduces the exploration space, leading to stabler and more efficient training.

Next, we ablate RL design decisions in [table II](#S4.T2). Following prior piano work [[[60](#bib.bib1)]], we try 2 values of $\gamma$, the RL discount factor, and find that a lower discount leads to lower F1 scores, and qualitatively, jerkier motions. We ablate the probability of sampling guided noise, and find our default hyperparameter is similar to not sampling guided noise. However, always sampling guided noise leads to degraded performance, which we hypothesize is because finger exploration is biased, which prevents learning from suboptimal data.

![Figure](2603.12243v2/x5.png)

*Figure 5: HandelBot Trajectories across Residual RL Training. We include 4 evaluation trajectories during HandelBot training, with the final, best-performing trajectory in [fig. 4](#S4.F4). Across these 4 trajectories, we see that HandelBot initially struggles with many keys in the left hand. However, with real-world interactions, the residual policy is able to adapt to real world and press the correct keys.*

### IV-D Ablation: Closed-Loop Sim-to-Real

In [table I](#S4.T1), we compare closed-loop inference and hybrid execution [[[61](#bib.bib16)]] for closed-loop sim-to-real policies. Hybrid execution, an alternative to direct sim-to-real transfer, runs a simulated environment in parallel with the real environment. Instead of taking real-world observations, hybrid execution mitigates the sim-to-real gap by using simulated observations. Similar to [[[61](#bib.bib16)]], we find an improvement when using hybrid execution, but performance is still far from HandelBot and other methods that utilize real-world data. We believe this is because hybrid execution only artificially reduces the gap between sim and real, and it cannot properly adjust to real-world dynamics.

## V CONCLUSIONS

In this paper, we introduced HandelBot, which leverages reinforcement learning in simulation, policy refinement, and residual reinforcement learning to tackle the extreme precision requirements of robotic piano playing. Our key insight is treating simulation as a foundation for global motor coordination and real-world interaction as the mechanism for fine-grained refinement. Our results demonstrate that our method can transform a brittle, imperfect simulated policy into a much more robust piano playing robot, requiring as little as 30 minutes of real-world data. We believe HandelBot represents a significant step toward deploying high-DoF dexterous hands in environments where spatial and temporal timing is the difference between success and failure.

Limitations. HandelBot relies on scripted end-effector movements with a fixed orientation, which leads to various limitations. First, this requires some amount of manual tuning each time. Residual RL over the end effector movements may reduce this problem. Second, this makes using the thumb and pinky more difficult. For this reason, we only evaluate on relatively simple songs. Future work may explore allowing rotations or learned movements in order to better utilize other fingers for more complex songs. Another limitation is the policy refinement step, which depends on human-guided heuristics. While this is sensible for piano playing and exploits obvious domain knowledge, this is not directly applicable to other tasks. However, policy refinement is possible for other tasks, either by human-guided heuristics or large models like vision-language models, which can help refine the policy at a coarse level before residual RL.

## ACKNOWLEDGMENT

This work was funded by ONR MURI N00014-25-1-2479, and ONR YIP N00014-22-1-2293, NSF #2218760 and NSF #1941722. We would like to thank members of ILIAD for their feedback and support. We thank Satvik Sharma and Jayson Meribe for hardware assistance and support, Marcel Torne Villasevil and Megha Srivastava for project brainstorming and simulator development, Hung-Chieh Fang for hardware and software advice, and Abrar Anwar and Hengyuan Hu for RL advice.

## References

-
[1]
L. Ankile, Z. Jiang, R. Duan, G. Shi, P. Abbeel, and A. Nagabandi (2025)

Residual off-policy rl for finetuning behavior cloning policies.

arXiv:2509.19301.

Cited by: [§II](#S2.p5.1).

-
[2]
R. Castro Ornelas (2022)

Robotic finger hardware and controls design for dynamic piano playing.

Ph.D. Thesis, Massachusetts Institute of Technology.

Cited by: [§II](#S2.p2.1).

-
[3]
X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang (2024)

Open-television: teleoperation with immersive active visual feedback.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[4]
O. X. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary, A. Jain, A. Balakrishna, A. Wahid, B. Burgess-Limerick, B. Kim, B. Schölkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn, C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak, D. Shah, D. Büchler, D. Jayaraman, D. Kalashnikov, D. Sadigh, E. Johns, E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Feng, G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta, H. Bharadhwaj, H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider, J. Hsu, J. Vakil, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu, J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang, J. Malik, J. Silvério, J. Hejna, J. Booher, J. Tompson, J. Yang, J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani, K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh, K. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. Fei-Fei, L. Tan, L. ”. Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero, M. Du, M. Ahn, M. C. Yip, M. Zhang, M. Ding, M. Heo, M. K. Srirama, M. Sharma, M. J. Kim, M. Z. Irshad, N. Kanazawa, N. Hansen, N. Heess, N. J. Joshi, N. Suenderhauf, N. Liu, N. D. Palo, N. M. M. Shafiullah, O. Mees, O. Kroemer, O. Bastani, P. R. Sanketi, P. ”. Miller, P. Yin, P. Wohlhart, P. Xu, P. D. Fagan, P. Mitrano, P. Sermanet, P. Abbeel, P. Sundaresan, Q. Chen, Q. Vuong, R. Rafailov, R. Tian, R. Doshi, R. Mart’in-Mart’in, R. Baijal, R. Scalise, R. Hendrix, R. Lin, R. Qian, R. Zhang, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kirmani, S. Levine, S. Lin, S. Moore, S. Bahl, S. Dass, S. Sonawani, S. Tulsiani, S. Song, S. Xu, S. Haldar, S. Karamcheti, S. Adebola, S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Ramamoorthy, S. Dasari, S. Belkhale, S. Park, S. Nair, S. Mirchandani, T. Osa, T. Gupta, T. Harada, T. Matsushima, T. Xiao, T. Kollar, T. Yu, T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Darrell, T. Chung, V. Jain, V. Kumar, V. Vanhoucke, V. Guizilini, W. Zhan, W. Zhou, W. Burgard, X. Chen, X. Chen, X. Wang, X. Zhu, X. Geng, X. Liu, X. Liangwei, X. Li, Y. Pang, Y. Lu, Y. J. Ma, Y. Kim, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Wu, Y. Xu, Y. Wang, Y. Bisk, Y. Dou, Y. Cho, Y. Lee, Y. Cui, Y. Cao, Y. Wu, Y. Tang, Y. Zhu, Y. Zhang, Y. Jiang, Y. Li, Y. Li, Y. Iwasawa, Y. Matsuo, Z. Ma, Z. Xu, Z. J. Cui, Z. Zhang, Z. Fu, and Z. Lin (2024)

Open X-Embodiment: robotic learning datasets and RT-X models.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[5]
V. de Bakker, J. Hejna, T. G. W. Lum, O. Celik, A. Taranovic, D. Blessing, G. Neumann, J. Bohg, and D. Sadigh (2026)

Scaffolding dexterous manipulation with vision-language models.

arXiv:2506.19212.

Cited by: [§II](#S2.p1.1).

-
[6]
R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang (2025)

Bunny-visionpro: real-time bimanual dexterous teleoperation for imitation learning.

In International Conference on Intelligent Robots and Systems (IROS),

Cited by: [§II](#S2.p1.1).

-
[7]
H. Fang, B. Romero, Y. Xie, A. Hu, B. Huang, J. Alvarez, M. Kim, G. Margolis, K. Anbarasu, M. Tomizuka, E. Adelson, and P. Agrawal (2025)

Dexop: a device for robotic transfer of dexterous human manipulation.

arXiv:2509.04441.

Cited by: [§II](#S2.p1.1).

-
[8]
S. Fujimoto, H. Hoof, and D. Meger (2018)

Addressing function approximation error in actor-critic methods.

In International conference on machine learning (ICML),

Cited by: [§III-D](#S3.SS4.SSS0.Px3.p1.5),
[§IV-A6](#S4.SS1.SSS6.p1.1).

-
[9]
J. Gao, S. Belkhale, S. Dasari, A. Balakrishna, D. Shah, and D. Sadigh (2026)

A taxonomy for evaluating generalist robot manipulation policies.

Robotics and Automation Letters (RA-L).

Cited by: [§II](#S2.p1.1).

-
[10]
J. Gao, A. Xie, T. Xiao, C. Finn, and D. Sadigh (2024)

Efficient data collection for robotic manipulation via compositional generalization.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[11]
J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. H. Huang, R. Chen, and H. Su (2023)

Maniskill2: a unified benchmark for generalizable manipulation skills.

In International Conference on Learning Representations (ICLR),

Cited by: [§IV-A2](#S4.SS1.SSS2.p1.1).

-
[12]
A. Gupta, J. Yu, T. Z. Zhao, V. Kumar, A. Rovinsky, K. Xu, T. Devlin, and S. Levine (2021)

Reset-free reinforcement learning via multi-task learning: learning dexterous manipulation behaviors without human intervention.

In International Conference on Information and Automation (ICRA),

Cited by: [§II](#S2.p4.1).

-
[13]
I. Guzey, H. Qi, J. Urain, C. Wang, J. Yin, K. Bodduluri, M. Lambeta, L. Pinto, A. Rai, J. Malik, T. Wu, A. Sharma, and H. Bharadhwaj (2026)

Dexterity from smart lenses: multi-fingered robot manipulation with in-the-wild human demonstrations.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p1.1).

-
[14]
A. Handa, K. Van Wyk, W. Yang, J. Liang, Y. Chao, Q. Wan, S. Birchfield, N. Ratliff, and D. Fox (2020)

Dexpilot: vision-based teleoperation of dexterous robotic hand-arm system.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p1.1).

-
[15]
E. Hsieh, W. Hsieh, Y. Wang, T. Lin, J. Malik, K. Sreenath, and H. Qi (2026)

Learning dexterous manipulation skills from imperfect simulations.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p1.1).

-
[16]
H. Hu, S. Mirchandani, and D. Sadigh (2024)

Imitation bootstrapped reinforcement learning.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p4.1).

-
[17]
K. Hu, H. Shi, Y. He, W. Wang, C. K. Liu, and S. Song (2025)

Robot trains robot: automatic real-world policy adaptation and learning for humanoids.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p4.1).

-
[18]
Z. Hu, A. Rovinsky, J. Luo, V. Kumar, A. Gupta, and S. Levine (2023)

REBOOT: reuse data for bootstrapping efficient real-world dexterous manipulation.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p4.1).

-
[19]
J. Hughes, P. Maiolino, and F. Iida (2018)

An anthropomorphic soft skeleton hand exploiting conditional models for piano playing.

Science Robotics.

Cited by: [§II](#S2.p2.1).

-
[20]
P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky (2025)

$\pi_{0.5}$: A vision-language-action model with open-world generalization.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[21]
A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto (2024)

Open teach: a versatile teleoperation system for robotic manipulation.

arXiv:2403.07870.

Cited by: [§II](#S2.p1.1).

-
[22]
T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, and S. Levine (2018)

Residual reinforcement learning for robot control.

arXiv:1812.03201.

Cited by: [§II](#S2.p5.1).

-
[23]
A. Kannan, K. Shaw, S. Bahl, P. Mannam, and D. Pathak (2023)

Deft: dexterous fine-tuning for real-world hand policies.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[24]
I. Kato, S. Ohteru, K. Shirai, T. Matsushima, S. Narita, S. Sugano, T. Kobayashi, and E. Fujisawa (1987)

The robot musician ‘wabot-2’(waseda robot-2).

Robotics.

Cited by: [§II](#S2.p2.1).

-
[25]
K. Kedia, T. G. W. Lum, J. Bohg, and C. K. Liu (2026)

SimToolReal: an object-centric policy for zero-shot dexterous tool manipulation.

arXiv:2602.16863.

Cited by: [§II](#S2.p1.1).

-
[26]
A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, J. Gao, V. Guizilini, D. A. Herrera, M. Heo, K. Hsu, J. Hu, M. Z. Irshad, D. Jackson, C. Le, Y. Li, K. Lin, R. Lin, Z. Ma, A. Maddukuri, S. Mirchandani, D. Morton, T. Nguyen, A. O’Neill, R. Scalise, D. Seale, V. Son, S. Tian, E. Tran, A. E. Wang, Y. Wu, A. Xie, J. Yang, P. Yin, Y. Zhang, O. Bastani, G. Berseth, J. Bohg, K. Goldberg, A. Gupta, A. Gupta, D. Jayaraman, J. J. Lim, J. Malik, R. Martín-Martín, S. Ramamoorthy, D. Sadigh, S. Song, J. Wu, M. C. Yip, Y. Zhu, T. Kollar, S. Levine, and C. Finn (2024)

DROID: a large-scale in-the-wild robot manipulation dataset.

In Robotics: Science and Systems (RSS),

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[27]
C. M. Kim, B. Yi, H. Choi, Y. Ma, K. Goldberg, and A. Kanazawa (2025)

PyRoki: a modular toolkit for robot kinematic optimization.

In International Conference on Intelligent Robots and Systems (IROS),

Cited by: [§IV-A3](#S4.SS1.SSS3.p1.1).

-
[28]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn (2025)

OpenVLA: an open-source vision-language-action model.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[29]
K. Lei, H. Li, D. Yu, Z. Wei, L. Guo, Z. Jiang, Z. Wang, S. Liang, and H. Xu (2026)

RL-100: performant robotic manipulation with real-world reinforcement learning.

External Links: 2510.14830,
[Link](https://arxiv.org/abs/2510.14830)

Cited by: [§II](#S2.p4.1).

-
[30]
Y. Li and L. Chuang (2013)

Controller design for music playing robot—applied to the anthropomorphic piano robot.

In International Conference on Power Electronics and Drive Systems (PEDS),

Cited by: [§II](#S2.p2.1).

-
[31]
J. Lin, H. Huang, Y. Li, J. Tai, and L. Liu (2010)

Electronic piano playing robot.

In International Symposium on Computer, Communication, Control and Automation (3CA),

Cited by: [§II](#S2.p2.1).

-
[32]
H. Liu, Z. Zhang, X. Xie, Y. Zhu, Y. Liu, Y. Wang, and S. Zhu (2019)

High-fidelity grasping in virtual reality using a glove-based system.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p1.1).

-
[33]
T. G. W. Lum, O. Y. Lee, C. K. Liu, and J. Bohg (2025)

Crossing the human-robot embodiment gap with sim-to-real rl using one human demonstration.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[34]
T. G. W. Lum, M. Matak, V. Makoviychuk, A. Handa, A. Allshire, T. Hermans, N. D. Ratliff, and K. V. Wyk (2024)

DextrAH-g: pixels-to-action dexterous arm-hand grasping with geometric fabrics.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[35]
J. Luo, Z. Hu, C. Xu, Y. L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine (2024)

SERL: a software suite for sample-efficient robotic reinforcement learning.

In International Conference on Information and Automation (ICRA),

Cited by: [§II](#S2.p4.1),
[§V-B](#Sx2.SS2.p1.4).

-
[36]
M. S. Mark, T. Gao, G. G. Sampaio, M. K. Srirama, A. Sharma, C. Finn, and A. Kumar (2024)

Policy agnostic rl: offline rl and online rl fine-tuning of any class and backbone.

arXiv:2412.06685.

Cited by: [§II](#S2.p4.1).

-
[37]
S. Mirchandani, M. Tang, J. Duan, J. I. Hamid, M. Cho, and D. Sadigh (2025)

RoboCade: gamifying robot data collection.

arXiv:2512.21235.

Cited by: [§II](#S2.p1.1).

-
[38]
S. Mirchandani, D. D. Yuan, K. Burns, M. S. Islam, T. Z. Zhao, C. Finn, and D. Sadigh (2025)

RoboCrowd: scaling robot data collection through crowdsourcing.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p1.1).

-
[39]
OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang (2019)

Solving rubik’s cube with a robot hand.

arXiv:1910.07113.

Cited by: [§II](#S2.p1.1).

-
[40]
H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik (2022)

In-hand object rotation via rapid motor adaptation.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[41]
C. Qian, J. Urain, K. Zakka, and J. Peters (2024)

PianoMime: learning a generalist, dexterous piano player from internet demonstrations.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p2.1),
[§III](#S3.p1.1).

-
[42]
Y. Qin, Y. Wu, S. Liu, H. Jiang, R. Yang, Y. Fu, and X. Wang (2022)

Dexmv: imitation learning for dexterous manipulation from human videos.

In European Conference on Computer Vision (ECCV),

Cited by: [§II](#S2.p1.1).

-
[43]
Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y. Chao, and D. Fox (2023)

Anyteleop: a general vision-based dexterous robot arm-hand teleoperation system.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[44]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)

Proximal policy optimization algorithms.

arXiv:1707.06347.

Cited by: [§V-A](#Sx2.SS1.p1.4).

-
[45]
K. Shaw, Y. Li, J. Yang, M. K. Srirama, R. Liu, H. Xiong, R. Mendonca, and D. Pathak (2024)

Bimanual dexterity for complex tasks.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[46]
L. Smith, I. Kostrikov, and S. Levine (2023)

A walk in the park: learning to walk in 20 minutes with model-free reinforcement learning.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p4.1).

-
[47]
T. Tao, M. K. Srirama, J. J. Liu, K. Shaw, and D. Pathak (2025)

DexWild: dexterous human interactions for in-the-wild robot policies.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[48]
A. Topper, T. Maloney, S. Barton, and X. Kong (2019)

Piano-playing robotic arm.

Worcester MA.

Cited by: [§II](#S2.p2.1).

-
[49]
C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu (2024)

DexCap: scalable and portable mocap data collection system for dexterous manipulation.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[50]
J. Wang, Y. Yuan, H. Che, H. Qi, Y. Ma, J. Malik, and X. Wang (2024)

Lessons from learning to spin “pens”.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[51]
R. Wang, P. Xu, H. Shi, E. Schumann, and C. K. Liu (2024)

Fürelise: capturing and physically synthesizing hand motion of piano performance.

In SIGGRAPH Asia,

Cited by: [§II](#S2.p2.1).

-
[52]
Z. K. Weng (2025)

BiDexHand: design and evaluation of an open-source 16-dof biomimetic dexterous hand.

External Links: 2504.14712,
[Link](https://arxiv.org/abs/2504.14712)

Cited by: [§II](#S2.p2.1).

-
[53]
P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel (2024)

Gello: a general, low-cost, and intuitive teleoperation framework for robot manipulators.

In International Conference on Intelligent Robots and Systems (IROS),

Cited by: [§II](#S2.p1.1).

-
[54]
H. Xu, Y. Luo, S. Wang, T. Darrell, and R. Calandra (2022)

Towards learning to play piano with dexterous hands and touch.

In International Conference on Intelligent Robots and Systems (IROS),

Cited by: [§II](#S2.p2.1).

-
[55]
M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song (2025)

DexUMI: using human hand as the universal manipulation interface for dexterous manipulation.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[56]
J. Yang, M. S. Mark, B. Vu, A. Sharma, J. Bohg, and C. Finn (2023)

Robot fine-tuning made easy: pre-training rewards and policies for autonomous real-world reinforcement learning.

arXiv:2310.15145.

Cited by: [§II](#S2.p4.1).

-
[57]
M. Yang, C. Lu, A. Church, Y. Lin, C. Ford, H. Li, E. Psomopoulou, D. A. W. Barton, and N. F. Lepora (2024)

AnyRotate: gravity-invariant in-hand object rotation with sim-to-real touch.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p1.1).

-
[58]
J. Yin, H. Qi, Y. Wi, S. Kundu, M. Lambeta, W. Yang, C. Wang, T. Wu, J. Malik, and T. Hellebrekers (2025)

OSMO: open-source tactile glove for human-to-robot skill transfer.

arXiv:2512.08920.

Cited by: [§II](#S2.p1.1).

-
[59]
X. Yuan, T. Mu, S. Tao, Y. Fang, M. Zhang, and H. Su (2025)

Policy decorator: model-agnostic online refinement for large policy model.

In International Conference on Learning Representations (ICLR),

Cited by: [§II](#S2.p5.1).

-
[60]
K. Zakka, P. Wu, L. Smith, N. Gileadi, T. Howell, X. B. Peng, S. Singh, Y. Tassa, P. Florence, A. Zeng, and P. Abbeel (2023)

RoboPianist: dexterous piano playing with deep reinforcement learning.

In Conference on Robot Learning (CoRL),

Cited by: [§I](#S1.p1.1),
[§I](#S1.p2.1),
[§II](#S2.p2.1),
[§III-B](#S3.SS2.SSS0.Px1.p1.1),
[§III-B](#S3.SS2.SSS0.Px2.p1.1),
[§IV-A5](#S4.SS1.SSS5.p1.1),
[§IV-C](#S4.SS3.p2.1),
[item 1](#Sx2.I1.i1.p1.1),
[§V-A](#Sx2.SS1.p1.4).

-
[61]
Y. Zeulner, S. Selvaraj, and R. Calandra (2025)

Learning to play piano in the real world.

arXiv preprint arXiv:2503.15481.

Cited by: [§II](#S2.p3.1),
[§IV-D](#S4.SS4.p1.1).

-
[62]
A. Zhang, M. Malhotra, and Y. Matsuoka (2011)

Musical piano performance by the act hand.

In International Conference on Robotics and Automation (ICRA),

Cited by: [§II](#S2.p2.1).

-
[63]
D. Zhang, J. Lei, B. Li, D. Lau, and C. Cameron (2009)

Design and analysis of a piano playing robot.

In International Conference on Information and Automation (ICRA),

Cited by: [§II](#S2.p2.1).

-
[64]
H. Zhang, S. Hu, Z. Yuan, and H. Xu (2025)

Doglove: dexterous manipulation with a low-cost open-source haptic force feedback glove.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[65]
J. Zhang, Y. Luo, A. Anwar, S. A. Sontakke, J. J. Lim, J. Thomason, E. Biyik, and J. Zhang (2025)

ReWiND: language-guided rewards teach robot policies without new demonstrations.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p4.1).

-
[66]
S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan (2025)

ResMimic: from general motion tracking to humanoid whole-body loco-manipulation via residual learning.

arXiv:2510.05070.

Cited by: [§II](#S2.p5.1).

-
[67]
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)

Learning fine-grained bimanual manipulation with low-cost hardware.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p1.1).

-
[68]
Z. Zhou, A. Peng, Q. Li, S. Levine, and A. Kumar (2024)

Efficient online reinforcement learning fine-tuning need not retain offline data.

arXiv preprint arXiv:2412.07762.

Cited by: [§II](#S2.p4.1).

## APPENDIX

We open-source our simulated and real-world implementations in [https://github.com/amberxie88/handelbot](https://github.com/amberxie88/handelbot) and show videos on our website [https://amberxie88.github.io/handelbot](https://amberxie88.github.io/handelbot).

### V-A Simulation Training

We train a PPO [[[44](#bib.bib66)]] policy per song. However, instead of a power penalty, we instead penalize an action L1 penalty on the delta finger joint actions. We also ablate the Key Press reward with a different Key On coefficient. The resulting key press reward, copying RoboPianist notation, is: $0.7\cdot\left(\frac{1}{K}\sum_{i}^{K}g(||k_{s}^{i}-1||_{2})\right)+0.3\cdot(1-\mathbf{1}_{\{\text{false positive\}}})$, where $K$ is the number of keys that need to be pressed at the current timestep, $k_{s}$ is the normalized joint position of the key between 0 and 1, and $\mathbf{1}_{\{\text{false positive\}}}$ is an indicator function that is 1 if any key that should not be pressed created a sound [[[60](#bib.bib1)]]. We find this necessary to encourage key presses at the expense of some wrong key presses. For this embodiment, wrong key presses, even in simulation, are almost unavoidable, hence this modification.

Furthermore, we fix the last joint angle of all fingers to be 1 radian. This reduces the action space to 3 joints per finger, reducing RL exploration and optimization. We found this helpful because the Tesollo DG-5F fingers are quite large, but the fingertips are slightly narrower. By fixing the last joint of each finger, the finger presses the key with only the tip of the finger, leading to a more human-like position and potentially a lower likelihood of pressing multiple keys.

To script the end effector position, we calculate the wrist trajectory based on the sheet music. The wrist trajectory is derived from the note sequence by computing the wrist position required to place the specified finger on each target key. For every note, the Y position of the key is obtained from the piano geometry, and a predefined finger–wrist offset is subtracted so that the corresponding finger aligns with the key when the wrist is at that position. The X position is set based on whether the key is white or black, since black keys are located slightly farther away.

When multiple notes occur at the same timestep, the wrist targets are aggregated by averaging the Y positions and selecting the minimum X position. These positions define anchor points in time, and a continuous wrist trajectory is then produced by linearly interpolating between them across all timesteps. Finally, the trajectory is expressed relative to the robot’s initial wrist pose.

Additional hyperparameters are described in [table III](#Sx2.T3).

*TABLE III: Hyperparameters for Simulation RL.*

### V-B Real-World Training

For real-world residual RL, we run an actor and learner process separately, as inspired by [[[35](#bib.bib15)]]. Our residual policy has chunk size 2, meaning that each residual action the policy outputs is repeated twice, effectively learning residuals at 5Hz over the 10Hz simulated trajectory. We find this leads to smoother learning. Next, we also sample correlated noise for our TD3 agent to reduce jitter. Concretely, our noise sampled at every step is $\hat{\epsilon}=\beta*\epsilon_{prev}+\sqrt{1-\beta^{2}}*\epsilon$, where our correlation coefficient $\beta=0.2$, $\epsilon\sim N(0,1)$, and $\hat{\epsilon}$ is the correlated noise we actually use. We also use a linear noising schedule for the first 10,000 gradient steps of policy training.

We include hyperparameters in [table IV](#Sx2.T4).

*TABLE IV: Hyperparameters for Real-World RL.*

### V-C Piano Songs

-
1.

Twinkle Twinkle. We adapt Twinkle Twinkle from the original RoboPianist [[[60](#bib.bib1)]] MIDI file, making minor modifications such as changing the octaves for each hand and reannotating fingerings.

-
2.

Ode to Joy. We use a popular YouTube version with minor modifications: [https://www.youtube.com/watch?v=i4bfhW7uLmc](https://www.youtube.com/watch?v=i4bfhW7uLmc)

-
3.

Hot Cross Buns. We compose a simple version of the popular nursery rhyme.

-
4.

Prelude in C. We change octaves and annotate fingerings for the original classical piece, Prelude and Fugue in C major, BWV 846 by Johann Sebastian Bach.

-
5.

Fur Elise. We change octaves for the original classical piece, Für Elise by Ludwig Van Beethoven.



Experimental support, please
[view the build logs](./2603.12243v2/__stdout.txt)
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