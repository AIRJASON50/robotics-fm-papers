[# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation Kushal Kedia∗1 Tyler Ga Wei Lum∗2 Jeannette Bohg†2 C. Karen Liu†2 1Cornell University 2Stanford University ∗Equal contribution †Equal advising simtoolreal.github.io](https://simtoolreal.github.io)

###### Abstract

The ability to manipulate tools significantly expands the set of
tasks a robot can perform. Yet, tool manipulation represents a
challenging class of dexterity, requiring grasping thin objects,
in-hand object rotations, and forceful interactions. Since
collecting teleoperation data for these behaviors is challenging,
sim-to-real reinforcement learning (RL) is a promising alternative.
However, prior approaches typically require substantial engineering
effort to model objects and tune reward functions for each task. In
this work, we propose SimToolReal, taking a step towards
generalizing sim-to-real RL policies for tool manipulation. Instead
of focusing on a single object and task, we procedurally generate a
large variety of tool-like object primitives in simulation and
train a single RL policy with the universal goal of manipulating
each object to random goal poses. This approach enables
SimToolReal to perform general dexterous tool manipulation at
test-time without any object or task-specific training. We
demonstrate that SimToolReal outperforms prior retargeting and
fixed-grasp methods by 37% while matching the performance of
specialist RL policies trained on specific target objects and
tasks. Finally, we show that SimToolReal generalizes across a
diverse set of everyday tools, achieving strong zero-shot
performance over 120 real-world rollouts spanning 24 tasks, 12
object instances, and 6 tool categories. Website: [simtoolreal.github.io](https://simtoolreal.github.io).

*

*Figure 1: SimToolReal is a framework for training a single general-purpose, object-centric RL policy in simulation and transferring it to real-world tool use. (Top) Zero-shot deployment on novel real tools and tasks, spanning thin markers to thick hammers. (Bottom) Tool use typically involves grasping objects from flat surfaces, reorienting them in-hand, and performing the task.*

## I Introduction

![Figure](x2.png)

*Figure 2: Overview of SimToolReal. (Top) Training in Simulation: We train a goal-conditioned RL policy in simulation that manipulates a wide variety of procedurally-generated objects to randomly sampled goal poses. (Bottom) Inference in Real: We deploy this policy zero-shot on real-world tools from DexToolBench, following tool trajectories from human videos.*

From household chores to industrial work, tool use expands a robot’s
capacity to act on its environment. However, tool use represents a
particularly challenging class of dexterous manipulation tasks.
Success requires grasping tools lying flat on a surface, reorienting
them into functional poses, and maintaining control during
potentially forceful interactions with the environment (see
Fig. [1](#S0.F1)). Consider hammering a nail: the robot
must grasp the hammer by its thin handle, rotate it in-hand into a
striking configuration, and deliver impact without losing the grasp.
Such interactions expose a fundamental limitation of parallel-jaw
grippers: with only two opposing contacts along a single grasp axis,
they provide limited resistance against externally induced torques,
making grasp stability primarily rely on friction and grip force.
This motivates multi-fingered dexterous hands for stable tool use.

Imitation learning from teleoperated demonstrations is a common
paradigm for learning manipulation policies [[[20](#bib.bib62), [90](#bib.bib33)]], but teleoperation is a poor fit for collecting
high-quality dexterous tool-use
data [[[13](#bib.bib95)]]. Human and
robot hands differ in kinematics and actuation, creating a
human-to-robot correspondence
gap [[[68](#bib.bib142), [7](#bib.bib143)]] that makes precise
dexterous control unintuitive for the operator [[[81](#bib.bib144)]]. At the
same time, operators receive limited to no force and tactile
feedback, preventing reliable regulation of contacts during the
task [[[53](#bib.bib31), [55](#bib.bib32), [51](#bib.bib145), [73](#bib.bib146)]].

Sim-to-real reinforcement learning (RL) is a promising alternative
that has successfully demonstrated agile, dynamic, and dexterous
behaviors in the real world [[[3](#bib.bib11), [14](#bib.bib139)]].
Although effective for specific skills, extending its success to
diverse tools and tasks is bottlenecked by substantial per-object
simulation setup and task-specific reward
engineering [[[56](#bib.bib28)]]. Consequently, existing
approaches are limited to sub-problems of dexterous tool-use, such as
grasping [[[46](#bib.bib77), [1](#bib.bib36), [18](#bib.bib30)]],
in-hand object reorientation [[[15](#bib.bib38), [54](#bib.bib37), [26](#bib.bib54)]], or
spinning objects [[[78](#bib.bib73), [40](#bib.bib72)]].

Our goal is to learn a dexterous manipulation policy that can be
applied zero-shot to novel tools and tasks at test-time. Our key
insight is to view dexterous tool use from an object-centric
lens, framing a task as moving a tool through a sequence of goal
poses. Thus, we focus on training a single goal-conditioned policy
that manipulates diverse objects to random goal poses. In simulation,
we procedurally generate a large set of primitive objects and train
one unified goal-conditioned policy. At test time, we compose this
policy for zero-shot tool use by sequentially conditioning it on a
goal pose trajectory extracted from a human video [[[21](#bib.bib14), [45](#bib.bib131)]]. This provides dense guidance
from the initial grasp, through in-hand reorientation into a
functional configuration, and into the ensuing tool-use motion. Our
approach avoids per-task reward shaping while allowing generalization
to novel tools and trajectories at test-time. Intuitively, mastering
the ability of manipulating objects to any random pose induces the
core skills required for tool use: establishing an initial grasp,
in-hand object reorientation, and maintaining stable contact.

We instantiate this approach in SimToolReal, a framework for training
a single general-purpose RL policy in simulation and deploying it for
real-world tool use (Fig. [2](#S1.F2)). To enable
generalization to novel objects, the policy is conditioned on the
tool’s current 6D pose and a coarse 3D bounding box over its
graspable region (e.g., the handle of a hammer). This abstraction
enables zero-shot transfer to the real-world by effectively bypassing
the sim-to-real visual gap. At deployment, we recover this
representation on real tools using a perception pipeline based on
vision foundation models, combining SAM 3D [[[16](#bib.bib27)]] for
segmentation and mesh extraction with
FoundationPose [[[80](#bib.bib58)]] for 6D pose tracking.

To evaluate generalization, we introduce DexToolBench, a
benchmark of daily tool-use behaviors, both in simulation and in the
real-world. Each task is paired with a human video demonstration, and
the robot is evaluated on its ability to follow the demonstrated tool
trajectory. Notably, SimToolReal is trained exclusively on
procedurally-generated primitives with randomly sampled goal pose
trajectories. Yet, it transfers zero-shot to the trajectory following
tasks in DexToolBench and matches or outperforms specialist
policies trained on a single object instance and task. Our contributions:

-
1.

We propose SimToolReal, a unified RL framework that frames
dexterous tool use as manipulating a tool through a sequence of
desired poses. Training a single policy under this objective
induces dexterous skills, such as stable grasping and in-hand
object reorientation. This policy is used zero-shot for tool-use
without task-specific engineering.

-
2.

We develop an object-centric perception pipeline that
estimates the tool’s 6D pose and a coarse 3D grasp bounding box
using vision foundation models, enabling zero-shot sim-to-real
deployment of the RL policy.

-
3.

We introduce DexToolBench, a benchmark of dexterous
tool-use tasks. We show that SimToolReal achieves strong
zero-shot performance on this benchmark over 120 real-world
rollouts spanning 24 tasks, 12 object instances, and 6 tool
categories. We outperform prior methods using fixed grasps and
motion retargeting by 37% in task progress while matching the
performance of specialist policies.

## II Related Work

Imitation Learning from Teleoperated Data. Imitation
learning (IL) is the predominant approach to train robot manipulation
policies [[[88](#bib.bib63), [41](#bib.bib116), [19](#bib.bib9)]]. However, IL relies on high-quality
demonstrations, typically collected via human teleoperation. Many
systems have been proposed to teleoperate multi-fingered dexterous
hands via motion capture gloves [[[67](#bib.bib113), [77](#bib.bib24)]], VR devices [[[22](#bib.bib115), [30](#bib.bib26), [42](#bib.bib23), [8](#bib.bib22)]], or even direct
camera input [[[63](#bib.bib34), [27](#bib.bib35), [72](#bib.bib25)]].
However, directly mapping hand motion to robot actions can lead to
imprecise, unintuitive
control [[[13](#bib.bib95), [68](#bib.bib142), [7](#bib.bib143)]]. Further, operators receive
limited haptic and tactile feedback during teleoperation, which
limits the collected demonstrations to simple manipulation
tasks [[[53](#bib.bib31), [55](#bib.bib32), [51](#bib.bib145), [73](#bib.bib146)]].

Recent works propose wearable exoskeleton systems that physically
couple the robot hand to the operator during data
collection [[[74](#bib.bib21), [82](#bib.bib118), [23](#bib.bib117)]]. Still,
these bulky exoskeleton systems carry a considerable burden on the
operator, constrain the range of human motion, and fail to
demonstrate complex skills, such as in-hand reorientations. Another
line of work treats the problem of teleoperation as shared autonomy.
These approaches learn structured action-spaces from human hand
motion data [[[52](#bib.bib109)]] or RL in
simulation [[[28](#bib.bib112), [44](#bib.bib111)]],
which are used by the human operator to control the robot. Recently,
DexterityGen [[[86](#bib.bib110)]]
trains RL policies conditioned on target grasp poses across diverse
objects. However, unlike our approach, it trains object-specific
policies and distills them into a model that only conditions on
hand-motion inputs. Largely, shared-autonomy methods are
object-blind at execution time and rely on skilled operators
to bridge the gap between perception and control. In contrast to all
these approaches, we focus on autonomous manipulation policies
without any human teleoperation effort.

Learning Dexterity from Human Video Demonstrations. Learning
dexterous manipulation directly from human video demonstrations is a
promising alternative to teleoperation. Kinematic motion retargeting
pipelines can translate human hand motions into robot actions by
reconstructing hand–object interaction [[[38](#bib.bib90), [35](#bib.bib19)]]. More recently, these pipelines have been
extended to convert demonstrations synthesized by video-generation
models into robot-executable trajectories [[[57](#bib.bib16), [37](#bib.bib17), [12](#bib.bib18)]]. However, because these methods
primarily rely on kinematic hand-motion references, they struggle to
produce actions that reliably grasp and maintain contact with tools.

To address this limitation, another line of work performs
functional retargeting, focusing on matching the
demonstrated task outcome rather than precisely matching human hand
motion. For instance, hand–object 3D motion-capture can be used to
train dexterous RL policies in simulation that reproduce the object
motion while discovering robot-specific contact
strategies [[[49](#bib.bib132), [17](#bib.bib78), [39](#bib.bib15)]]. More recently, functional
retargeting has been extended to RGB-D human video
demonstrations [[[45](#bib.bib131), [21](#bib.bib14)]] and even AI-generated videos [[[50](#bib.bib13)]]. Despite
their success, these approaches typically require time-intensive RL
training where the resulting policy is limited to the demonstrated
manipulation task and object. In contrast, our approach can imitate
object motion directly from a single RGB-D human demonstration
without any additional demonstration-specific training. We can use
the same policy across diverse tasks.

Sim-to-Real Learning. Policies learned in simulation can
acquire dexterous manipulation behaviors without relying on human
demonstrations, but transferring these skills to the real world often
requires substantial environment
shaping [[[56](#bib.bib28)]]. This includes closing the
sim-to-real gap, building accurate task-specific object models, and
carefully designing and tuning rewards for each behavior. Domain
randomization over physical and visual
parameters [[[26](#bib.bib54), [3](#bib.bib11), [6](#bib.bib10)]] and adaptation methods that infer
real-world dynamics online [[[34](#bib.bib8), [62](#bib.bib129), [75](#bib.bib5)]] can reduce the sim-to-real gap significantly. Still,
sim-to-real dexterous manipulation requires per-task engineering,
which has led many systems to focus on narrower skill subsets such as
grasping [[[1](#bib.bib36), [84](#bib.bib124), [46](#bib.bib77), [70](#bib.bib7), [69](#bib.bib6)]], in-hand object
reorientation [[[15](#bib.bib38), [44](#bib.bib111)]], or
dynamic object spinning [[[78](#bib.bib73), [40](#bib.bib72)]].
Recent research in locomotion has begun to unify RL policies across
embodiments by leveraging large-scale simulation with procedurally
generated robots and unified reward
functions [[[43](#bib.bib3), [2](#bib.bib4)]]. Inspired by this
direction, we train a single RL policy across procedurally generated
object primitives and the universal reward function of reaching
random goal poses. We show that this formulation induces highly
dexterous skills (e.g., in-hand reorientation and dynamic spinning)
and transfers to diverse real-world tools without task-specific
engineering effort.

![Figure](x3.png)

*Figure 3: Real-World Deployment. (Left) Human Video Processing: We collect an RGB-D human video and process it using vision foundation models. We use SAM 3D [[[16](#bib.bib27)]] to generate a metric-scale object mesh and segment a 3D grasp bounding box. Then, we use FoundationPose [[[80](#bib.bib58)]] to extract a sequence of 6D goal poses. (Right) Inference-Time Pipeline: Our LSTM policy takes in proprioception, object pose, grasp bounding box, and goal pose, and it outputs joint position targets for the 29-DoF dexterous robot (arm + hand).*

## III SimToolReal

We propose SimToolReal, a sim-to-real RL framework for zero-shot,
dexterous tool manipulation that generalizes to unseen tools and
manipulation sequences. Our key insight is to view dexterous tool-use
tasks through an object-centric lens: a wide range of tasks
can be specified as manipulating a tool through a sequence of goal
poses. This reduces the tool-use problem to learning a single
goal-reaching RL policy in simulation that manipulates procedurally
generated tool primitives toward random goal poses. This training
objective induces dexterous behaviors, such as stable grasping and
in-hand reorientation, that are essential for tool use. At test time,
the policy takes as input a sequence of object goal poses (e.g., from
a human video) and executes actions to track this sequence, enabling
diverse zero-shot tool-use without any real-world object modeling in
simulation or task-specific training.

### III-A Problem Formulation

We consider the problem of training dexterous policies that can
manipulate any tool to arbitrary goal poses. Let $\bm{o}_{t}\in SE(3)$
denote the current object pose at time $t$, $\bm{s}_{t}$ denote robot
proprioception, and $\bm{\phi}$ denote a coarse object descriptor
(e.g., approximate geometry or physical properties). Given a goal
pose $\bm{g}\in SE(3)$, we learn a policy $\bm{a}_{t}=\pi_{\theta}\!\left(\bm{s}_{t},\,\bm{o}_{t},\,\bm{\phi},\,\bm{g}\right)$
that outputs joint position targets for the robot’s arm and hand.
Task execution then reduces to repeatedly reaching the current goal
pose and advancing to the next goal. This abstraction lets a single
goal-reaching controller support diverse tool-use tasks, without
task-specific policy training or reward design.

To perform a tool-use task, we assume access to a sequence of goal object poses
$\{\bm{g}^{k}\}_{k=1}^{K}$ with $\bm{g}^{k}\in SE(3)$, extracted from a
human demonstration video. For example, in Fig. [2](#S1.F2),
we visualize the goal sequence of reorienting the hammer, aligning it
above the nail, and swinging. During execution, we condition the
policy on the current goal pose $\bm{g}^{k}$ and advance to the next
goal $\bm{g}^{k+1}$ when the object pose $\bm{o}_{t}$
is sufficiently close, i.e., when $d(\bm{o}_{t},\bm{g}^{k})Environment Setup. We design a general environment to model
tool-use tasks. At the start of each episode, we place a randomly
selected object on the table in a random pose and initialize the
robot in a randomized joint configuration. We then sample a sequence
of goal object poses
$\{\bm{g}^{k}\}_{k=1}^{K}$. The robot must first learn to grasp the
object from the flat table surface and subsequently manipulate it to
reach each goal. If the object is dropped, the episode terminates and
is reset. We sample the first goal randomly in the robot’s reachable
workspace to expose the robot to a wide range of object poses and
large reorientations. Subsequent goals are sampled close to the
previous goal to encourage smooth, trajectory-like motion.

Reward Function. Our primary objective is to train a
goal-pose reaching RL policy. Practically, we
optimize [[[59](#bib.bib133)]]:

$$r=r_{\textrm{smooth}}+r_{\textrm{grasp}}+\mathbb{I}_{\textrm{grasped}}r_{\textrm{goal}}.$$ \tag{1}

$r_{\textrm{smooth}}$ is a regularization term that encourages smooth
actions, $r_{\textrm{grasp}}$ is a shaping term that encourages
grasping the object at the start of the episode, and
$\mathbb{I}_{\textrm{grasped}}$ indicates whether the object has been
grasped. After the object is grasped, $r_{\textrm{goal}}$ is the
dominant reward term, driving goal-pose reaching. We define:

| | $$r_{\rm goal}=\textrm{max}(d^{*}-d(\bm{o}_{t},\bm{g}),0)+B_{\rm succ}\,\mathbb{I}\!\left[d(\bm{o}_{t},\bm{g})Procedural Generation of Tools in Simulation. We generate a
large diversity of tool-like primitive objects in simulation such
that they span the variation seen in real-world tools. Each tool
primitive is generated as a combination of a handle and a
head: we sample their geometries as cylinders and cuboids
with varying dimensions. While simple, this design captures the key
structure of many everyday tools, such as brushes, spatulas, markers,
and hammers. A random subset of our generated primitives is
visualized in Fig [2](#S1.F2). In addition to geometric
variation, we randomize mass distribution by assigning different
densities to the handle and head, representative of many real-world
tools (e.g., hammer heads are typically much denser than handles).
Additional generation details are provided in Appendix [A-A](#A1.SS1).

Object-Centric Policy Inputs. To deploy a single policy
across diverse real-world tools, we need an object representation
that makes learning efficient while also being feasible to extract in
the real-world. Real-world tools are diverse in their geometry and
physical properties, and accurately estimating detailed geometric or
physical parameters from visual sensing alone is impractical.
Instead, we provide the policy only the object inputs that are
reliably available at deployment: the current 6D tool pose and a
coarse 3D grasp-region bounding box. The bounding box encodes the
intended graspable region (center + extents in the object frame;
Fig. [3](#S2.F3)) and is held fixed during an episode. We use
an LSTM backbone so the policy can integrate interaction history and
implicitly infer latent physical and geometric properties that are
not directly observed. This design follows the spirit of prior work
such as DexFunc [[[1](#bib.bib36)]] and
RMA [[[62](#bib.bib129)]], which demonstrate dexterous manipulation
behaviors with limited object perception to reduce the sim-to-real
observation gap.

RL Training Details. We highlight key RL design choices
essential for policy learning and sim-to-real transfer.

a) SAPG [[[71](#bib.bib138)]]: We train our policy using
SAPG, a variant of PPO [[[66](#bib.bib60)]]. We find that standard PPO
encounters exploration bottlenecks in massively parallel simulations.
SAPG mitigates this by maintaining a population of policies to
promote exploration diversity and updating the leader policy using
their collective experience.

b) Domain Randomization: We apply targeted domain
randomization during training to aid sim-to-real transfer. This
includes observation delays, action-execution latency, noise and
delay in object-pose estimates, and perturbations to the grasp-region
bounding box. We additionally apply random force and torque
perturbations to the object, encouraging the policy to learn strong,
stable grasps. Randomization ranges and other details are provided in
Appendix [A-C](#A1.SS3).

c) Asymmetric
Critic [[[60](#bib.bib105)]]: While the actor receives only
the minimal object representation available at test time, we train
with an asymmetric actor–critic setup in which the critic has access
to privileged states in simulation. We also provide the critic with
clean (noise-free, undelayed) observations to improve value
estimation and stabilize training. Additional details are included in
Appendix [A-C](#A1.SS3).

![Figure](images/final_experiment_plots/verbs_generalization_instance_colors_v2.png)

*Figure 4: Generalization to Unseen DexToolBench Tools and Tasks in the Real World. We evaluate our policy in the real world on unseen tool-use tasks in DexToolBench. Our evaluations span 24 unique task trajectories across 6 different object categories and 12 object instances. Each bar corresponds to 1 task trajectory on 1 object instance. We report the average Task Progress across 5 rollouts. Despite not being trained on these objects or trajectories, our policy demonstrates strong generalization to diverse tools of varying masses and geometries.*

### III-C Real-World Deployment

In this section, we describe how we deploy our object-centric policy
in the real world to perform a wide range of tool-use tasks.
Fig. [3](#S2.F3) provides an overview. Given a third-person
RGB-D view of a human demonstration, we extract (i) a 3D metric-scale
object mesh and grasp region bounding box, and (ii) a sequence of
target object poses that serves as the goal trajectory. During
execution, we run the policy in a closed loop to reach the goal poses
sequentially.

In contrast to [Lum et al. [[45](#bib.bib131)]], our
approach does not require access to the real-world object or target
trajectory during training, enabling zero-shot generalization to
novel objects using demonstrations collected entirely after training.

Human Video Processing. From the first RGB-D frame, we
reconstruct a metric-scale 3D mesh of the object using SAM
3D [[[16](#bib.bib27)]]. We then segment the intended grasp region
using SAM 2 [[[65](#bib.bib85)]], and convert this region into a
coarse 3D bounding box that is provided as part of the policy input.
Finally, we run FoundationPose [[[80](#bib.bib58)]] on the
RGB-D video conditioned on the extracted mesh to obtain the 6D
object-pose trajectory. We process this trajectory by downsampling to
3 Hz. We use this processed trajectory as the goal sequence at test
time. See Appendix [A-D](#A1.SS4) for implementation details.

Inference-time Object Tracking. During inference, we
estimate the current 6D object pose with an update frequency of 30
Hz. For live pose tracking, we use
FoundationPose [[[80](#bib.bib58)]] with RGB-D observations
from a third-person camera view combined with the 3D mesh extracted
from the human video. At each control step, the policy is conditioned
on proprioception, the current object pose, the (fixed) grasp-region
bounding box, and the current goal pose from the demonstration
trajectory. We initialize the goal as the first pose in the sequence,
switching it to the next goal whenever the pose error
$d(\bm{o}_{t},\bm{g})$ falls within a tolerance $\epsilon$. The episode
ends when the final goal is reached or upon failure (e.g., loss of
grasp or tracking).

### III-D DexToolBench

Existing robot manipulation benchmarks primarily focus on robots with
parallel-jaw
grippers [[[47](#bib.bib153), [31](#bib.bib154), [9](#bib.bib155), [61](#bib.bib156), [32](#bib.bib157), [24](#bib.bib160), [36](#bib.bib161)]]
or dexterous
grasping [[[84](#bib.bib124), [79](#bib.bib158), [89](#bib.bib159), [83](#bib.bib74), [76](#bib.bib75)]].
We introduce DexToolBench: a real-world, dexterous
manipulation benchmark of challenging tool-use tasks paired with
digital-twin simulation environments. These tasks require grasping
tools from flat surfaces, in-hand rotations to functional
configurations, and environment interactions. This benchmark consists
of 24 daily tool-use tasks, 12 unique object instances across 6 categories.

Each task is defined by an RGB-D human video, captured via a ZED 1
stereo camera. We process each video with our perception pipeline
(Fig. [3](#S2.F3)) to extract a 3D object mesh and a 6D
object pose trajectory. To facilitate reproducibility, we provide the
raw RGB-D videos, processed data, simulation environments, and
purchase links for the real-world tools.

Each tool instance is visualized in
Fig [4](#S3.F4). We briefly describe the skills
required in each tool-use category in the benchmark:

a) Hammer: Grasp, 90∘ in-hand rotation,
swing down or side.

b) Marker: Grasp a thin object, write on a whiteboard.

c) Eraser: Grasp, wipe marker ink on whiteboard.

d) Brush: Grasp, 90∘ in-hand rotation,
sweep forward or right.

e) Spatula: Grasp, serve or 180∘ in-hand
rotation and flip.

f) Screwdriver: Grasp, 90∘ in-hand
rotation, spin in free space along vertical or horizontal axis.

See Appendix [A-E](#A1.SS5) for more details about
DexToolBench.

## IV Experiments

Experimental Setup: We evaluate SimToolReal on a robot
consisting of a 22-DoF Sharpa five-fingered left hand mounted on a
7-DoF KUKA iiwa 14 arm. We evaluate our policies on
DexToolBench (Sec.[III-D](#S3.SS4)). For each
task, we evaluate Task Progress, measuring the percentage of
demonstrated goal poses tracked successfully. We focus on trajectory
following of a fixed goal sequence rather than evaluating functional
task completion. We consider a goal pose to be reached if the
distance between the object pose and goal pose $d(\bm{o}_{t},\bm{g})$
is below our defined success tolerance $\epsilon=2cm$. We design
experiments to answer the following questions:

-
A.

What is the zero-shot transfer performance of SimToolReal
on unseen objects and trajectories in the real world?

-
B.

How does SimToolReal compare to prior methods that use
kinematic motion retargeting or fixed-grasps?

-
C.

How does SimToolReal compare to task-specific specialists
trained on a single object and trajectory?

-
D.

To what extent does our training objective correlate with
downstream performance on unseen tools and tasks?

-
E.

Which RL design decisions are important for maximizing
training performance?

![Figure](x4.png)

*Figure 5: Comparison against Baselines in the Real World. We compare SimToolReal against baselines on two variations of sweeping a table with a brush: with and without requiring tool rotation based on the initial states shown on the left. Average Task Progress is indicated in parentheses. SimToolReal succeeds on both variations, performing dexterous in-hand tool rotations in the harder variation. Fixed Grasp succeeds on the simpler variation of this task without tool rotation. However, when rotation is required, enforcing a fixed grasp causes the arm to collide with the table while tracking the target trajectory. Kinematic Retargeting fails to reason about contact forces, and is unable to grasp the brush in both variations.*

### IV-A Zero-Shot Real-World Tool-Use

Setup. We evaluate whether our single policy trained in
simulation transfers zero-shot to real-world tool use, without any
object- or task-specific fine-tuning. As shown in
Fig. [4](#S3.F4), our evaluation spans 24 task
trajectories across 6 tool categories and 12 object instances in
DexToolBench (Sec. [III-D](#S3.SS4)), all of
which were unseen during training. We report average Task
Progress over 5 trials, totaling 120 real-world rollouts.

Results. Overall, the policy demonstrates strong zero-shot
generalization across tools with diverse masses and geometries. We
observe the highest Task Progress on eraser
trajectories, which rely primarily on translation motion rather than
in-hand object rotation. Similarly, marker trajectories
don’t require in-hand rotations, but because the tool is much
thinner, the grasps are less reliable. Further, the marker’s small
size makes it prone to pose tracking loss during occlusion.

The remaining four categories (hammer, brush,
spatula, screwdriver) require substantial in-hand
object rotation to reach functional tool configurations. While the
policy still achieves strong Task Progress across these
tasks, performance degrades on thinner tools (e.g., performs better
on the $\sim$3cm thick spoon spatula than the $\sim$1cm
thick flat spatula) and heavier tools (e.g., performs better
on the 36g claw hammer than the 331g mallet
hammer). We also note that tool geometry impacts task difficulty.
For instance, the red brush’s sideways head makes
sweep forward easier, but the blue brush’s frontal head
makes sweep right easier. The screwdriver is the
most challenging category, as the task requires both functional
reorientation and continuous spinning.

Failure Analysis. The most common failure mode was pose
tracking loss (43.7% of failures), followed by object drops
(34.5%), failure to reach the goal pose due to incomplete in-hand
rotation (18.2%), and grasp failure (3.6%). See
Appendix [A-F](#A1.SS6) for additional quantitative
and qualitative analysis of our results. Notably, we observe strong
recovery behavior when the policy makes mistakes. For example, when
the object is dropped, the policy consistently attempts to re-grasp
the object, provided the object remains within the workspace and pose
tracking is not lost.

### IV-B Comparisons to Retargeting and Fixed Grasp Baselines

Setup. In this section, we compare SimToolReal with:

(i) Kinematic Retargeting: Following prior
works [[[38](#bib.bib90), [45](#bib.bib131), [25](#bib.bib147), [64](#bib.bib148), [58](#bib.bib59)]], we retarget hand motion from a human
video into dexterous robot actions. From an RGB-D video, we estimate
3D hand finger positions and then solve for arm and hand joint
positions using an Inverse Kinematics (IK) solver. Although this
method bypasses the need for RL, the resulting motion is purely
kinematic and fails to account for contact interactions. See
Appendix [A-G](#A1.SS7) for implementation details.

(ii) Fixed Grasp: Recent
works [[[1](#bib.bib36), [57](#bib.bib16), [29](#bib.bib152), [91](#bib.bib151), [5](#bib.bib149)]]
frame manipulation tasks as establishing a grasp, then subsequently
fixing the grasp while moving the object. To perform any object
rotations, the robot must strictly rely on arm motion. This baseline
isolates the necessity of in-hand manipulation; failure here
demonstrates that the arm lacks the kinematic workspace to rotate
tools arbitrarily without the assistance of in-hand rotation. See
Appendix [A-H](#A1.SS8) for implementation details.

To systematically evaluate the baselines, we set up two variations of
sweep forward with the red brush in the real-world (see
Fig. [5](#S4.F5)): (i) Variation #1 starts
with the brush top-down and does not require tool rotation, and (ii)
Variation #2 starts with the brush sideways, which
necessitates a 90∘ tool rotation. We evaluate average
Task Progress across 5 rollouts.

Results. Kinematic Retargeting fails to grasp the
brush in both variations as kinematic motion alone does not establish
stable contacts. Fixed Grasp succeeds on Variation
#1 as tool rotation is not required and arm motion alone is
sufficient to finish the task. Still, it has lower Task
Progress than SimToolReal since it is an open-loop method and
cannot react to small errors. On Variation #2, enforcing a
fixed grasp causes the robot arm to collide with the table while
tracking the target trajectory, as the optimizer is unable to find an
arm motion trajectory that satisfies the tool rotation while avoiding
table collision. In contrast, SimToolReal performs efficient
in-hand object rotations to complete the task.

![Figure](images/final_experiment_plots/new_specialist_comparison_3settings_revised.png)

*Figure 6: Comparison against Specialists. We compare SimToolReal in simulation against specialist policies trained on a single object (Obj A) and trajectory (Traj A). We train one specialist policy for each of the 6 object categories in DexToolBench and report the average Task Progress across these categories. While the specialists succeed on their training setup (Obj A / Traj A), performance degrades under deviation in the trajectory (Obj A / Traj B) or the object (Obj B / Traj A). SimToolReal has high zero-shot performance across all variants, despite not being trained on these objects or trajectories.*

### IV-C Comparisons to Specialist Baseline

Setup.
In simulation, we compare SimToolReal against task-specific
specialist RL baselines trained on a single tool and task.
Following [[[45](#bib.bib131)]], we train 6
specialist policies, one for each of the 6 tool-use categories in
DexToolBench. Each specialist policy trains on a single
object instance (Obj A) and task trajectory (Traj A). We evaluate
both methods in simulation on (i) the training setup for the
specialist: (Obj A / Traj A), (ii) the training object with a novel
trajectory (Obj A / Traj B), and (iii) a novel object instance with
the same trajectory (Obj B / Traj A). For each variation, we perform
10 rollouts for each of the 6 specialist policies, reporting the
average Task Progress across all trials.

Results. Fig. [6](#S4.F6) summarizes
these results. On the setup used for training the specialist (Obj A /
Traj A), SimToolReal matches the specialist without training on that
object or trajectory. However, the specialist’s performance drops
significantly under any variation from the training setup. When the
object stays the same, with only a change in trajectory (Obj A / Traj
B), the specialist can only track the first few goals that lift the
object. The performance is lowest when the object instance is changed
(e.g., from red brush to blue brush) even if the trajectory remains
the same (Obj B / Traj A). This indicates that the specialist
overfits to the training conditions. In contrast, SimToolReal shows
strong zero-shot Task Progress across both object and
trajectory variations.

### IV-D Training Objective Predicts Generalization Performance

Setup. We study how improving on our training objective:
reaching random goal poses across procedurally generated primitive
objects, relates to performance on our test objective: executing
human-demonstrated tool-use trajectories in DexToolBench.
Fig. [7](#S4.F7) reports (i) training rewards on
the goal-pose reaching objective on primitive objects and (ii)
average Task Progress on DexToolBench trajectory
following tasks in simulation, plotted against the number of environment steps.

![Figure](images/final_experiment_plots/sim_tp_and_reward_vs_env_steps.png)

*Figure 7: Training Objective Drives Generalization. In simulation, we evaluate (Left) the episode reward on procedurally-generated objects and (Right) the zero-shot Task Progress on unseen DexToolBench tools on different policy checkpoints throughout training. The strong correlation between the two curves validates our core hypothesis: improving random goal-pose reaching performance on diverse object primitives drives corresponding gains in generalization to unseen tool-use tasks.*

Results. As training proceeds, the policy’s reward on the
goal-pose reaching objective increases steadily. In parallel, the
policy’s average Task Progress on DexToolBench
trajectories also improves consistently. The correlation between
these curves indicates that optimizing our training objective is a
reliable predictor of downstream generalization to tool-use
behaviors, despite the mismatch between random goal-pose reaching on
primitive objects during training and trajectory following on real
tool objects at test time. This strong correlation suggests that our
randomized primitive training effectively covers the space of skills
required for real-world tool use.

### IV-E Ablations on RL Training

Setup. We investigate which RL design decisions are critical
for maximizing training performance. We focus on two key components
of our RL pipeline: the choice of optimization algorithm
(SAPG [[[71](#bib.bib138)]] vs. PPO [[[66](#bib.bib60)]]) and the use of privileged
information (Asymmetric Critic [[[60](#bib.bib105)]] vs.
standard critic). We train these variants with identical
hyperparameters and environment settings, averaging results across 5
random seeds.

![Figure](images/final_experiment_plots/training_reward_vs_env_steps.png)

*Figure 8: Ablation of RL Training Components. We compare the training reward across environment steps of SimToolReal against ablations averaged across 5 seeds. Replacing SAPG [[[71](#bib.bib138)]] with PPO [[[66](#bib.bib60)]] or not using Asymmetric Critic [[[60](#bib.bib105)]] results in a significant performance drop, highlighting their importance.*

Results. Fig. [8](#S4.F8) compares the training
reward curves of our full method against these ablations.
Importance of SAPG. We observe that replacing
SAPG [[[71](#bib.bib138)]] with PPO [[[66](#bib.bib60)]] leads to a significant drop
in performance. While PPO suffers from exploration saturation at
scale, SAPG mitigates this by training separate policies across
environment chunks to increase data diversity, and then fusing
gradients via importance sampling. Our results confirm that this is
essential for learning complex dexterous tool-use. Importance
of Asymmetric Critic. We observe that removing the Asymmetric
Critic [[[60](#bib.bib105)]] and forcing the critic to rely
solely on the same partial observations as the actor severely hinders
learning. By accessing privileged simulation states, the critic
learns a more accurate value function to guide policy learning. This
guidance is essential for overcoming the task’s partial observability.

## V Discussion and Limitations

We introduced SimToolReal, an RL framework for zero-shot dexterous
tool manipulation that generalizes to unseen tool-use tasks. Our key
insight is an object-centric reduction: tool-use can be specified as
manipulating a tool through a sequence of goal poses. This shifts the
problem from per-task reward design to a universal objective of
training goal-pose reaching policies. Training with this objective
over procedurally generated primitives induces dexterous skills, such
as stable grasping and in-hand reorientation, critical for tool-use.

Limitations. While our approach can track tool-use goal
sequences, it does not guarantee functional task completion,
especially for high-force interactions. Conditioning on object pose
goals alone is environment-blind, which can lead to collisions in
cluttered scenes. We also currently assume tools are rigid. Pose
alone can be insufficient to describe the state of non-rigid tools
(e.g., scissors). Finally, our high-level goal sequence is fixed and
is not replanned dynamically.

## Acknowledgements

This work is supported by Stanford Human-Centered Artificial
Intelligence (HAI), ONR Young Investigator Award, the National
Science Foundation (NSF) under Grant Numbers 2153854, 2327974,
2312956, 2327973, and 2342246, and the Natural Sciences and
Engineering Research Council of Canada (NSERC) under Award Number
526541680. We thank Sharpa for their research collaboration and for the technical support provided by their team, specifically Kaifeng Zhang, Wenjie Mei, Yi Zhou, Yunfang Yang, Jie Yin, and Jason Lee.

## References

-
[1]
A. Agarwal, S. Uppal, K. Shaw, and D. Pathak (2023)

Dexterous functional grasping.

External Links: 2312.02975,
[Link](https://arxiv.org/abs/2312.02975)

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1),
[§III-B](#S3.SS2.p4.1),
[§IV-B](#S4.SS2.p3.1).

-
[2]
B. Ai, L. Dai, N. Bohlinger, D. Li, T. Mu, Z. Wu, K. Fay, H. I. Christensen, J. Peters, and H. Su (2025)

Towards embodiment scaling laws in robot locomotion.

arXiv preprint arXiv:2505.05753.

Cited by: [§II](#S2.p5.1).

-
[3]
I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. (2019)

Solving rubik’s cube with a robot hand.

arXiv preprint arXiv:1910.07113.

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[4]
A. Allshire, M. MittaI, V. Lodaya, V. Makoviychuk, D. Makoviichuk, F. Widmaier, M. Wüthrich, S. Bauer, A. Handa, and A. Garg (2022)

Transferring dexterous manipulation from gpu simulation to a remote real-world trifinger.

In 2022 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS),

pp. 11802–11809.

Cited by: [Appendix A](#A1.p7.7),
[§III-B](#S3.SS2.p2.13).

-
[5]
S. H. Allu, J. J. P, N. Khargonkar, T. Summers, J. Yao, and Y. Xiang (2025)

HRT1: one-shot human-to-robot trajectory transfer for mobile manipulation.

External Links: 2510.21026,
[Link](https://arxiv.org/abs/2510.21026)

Cited by: [§IV-B](#S4.SS2.p3.1).

-
[6]
O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. (2020)

Learning dexterous in-hand manipulation.

The International Journal of Robotics Research 39 (1), pp. 3–20.

Cited by: [§II](#S2.p5.1).

-
[7]
M. Arduengo, A. Arduengo, A. Colomé, J. Lobo-Prat, and C. Torras (2021)

Human to robot whole-body motion transfer.

In 2020 IEEE-RAS 20th International Conference on Humanoid
Robots (Humanoids),

Vol. , pp. 299–305.

External Links: [Document](https://dx.doi.org/10.1109/HUMANOIDS47582.2021.9555769)

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[8]
S. P. Arunachalam, I. Güzey, S. Chintala, and L. Pinto (2022)

Holo-dex: teaching dexterity with immersive mixed reality.

arXiv preprint arXiv:2210.06463.

Cited by: [§II](#S2.p1.1).

-
[9]
P. Atreya, K. Pertsch, T. Lee, M. J. Kim, A. Jain, A. Kuramshin, C. Eppner, C. Neary, E. Hu, F. Ramos, et al. (2025)

RoboArena: distributed real-world evaluation of generalist robot policies.

In Proceedings of the Conference on Robot Learning (CoRL 2025),

Cited by: [§III-D](#S3.SS4.p1.1).

-
[10]
G. Authors (2024-12)

Genesis: a generative and universal physics engine for robotics and beyond.

External Links: [Link](https://github.com/Genesis-Embodied-AI/Genesis)

Cited by: [§A-B1](#A1.SS2.SSS1.p2.1).

-
[11]
S. R. Buss (2004)

Introduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods.

Unpublished manuscript

University of California, San Diego.

Note: Available at
[https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf](https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf)

Cited by: [1st item](#A1.I13.i1.p1.3),
[§A-G](#A1.SS7.p1.1).

-
[12]
B. Chen, T. Zhang, H. Geng, K. Song, C. Zhang, P. Li, W. T. Freeman, J. Malik, P. Abbeel, R. Tedrake, et al. (2025)

Large video planner enables generalizable robot control.

arXiv preprint arXiv:2512.15840.

Cited by: [§II](#S2.p3.1).

-
[13]
C. Chen, Z. Yu, H. Choi, M. Cutkosky, and J. Bohg (2025)

DexForce: extracting force-informed actions from kinesthetic demonstrations for dexterous manipulation.

IEEE Robotics and Automation Letters 10 (6), pp. 6416–6423.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[14]
T. Chen, E. Cousineau, N. Kuppuswamy, and P. Agrawal (2025)

Vegetable peeling: a case study in constrained dexterous manipulation.

In 2025 IEEE International Conference on Robotics and
Automation (ICRA),

Vol. , pp. 4542–4550.

External Links: [Document](https://dx.doi.org/10.1109/ICRA55743.2025.11127224)

Cited by: [§I](#S1.p3.1).

-
[15]
T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal (2023)

Visual dexterity: in-hand reorientation of novel and complex object shapes.

Science Robotics 8 (84).

External Links: ISSN 2470-9476,
[Link](http://dx.doi.org/10.1126/scirobotics.adc9244),
[Document](https://dx.doi.org/10.1126/scirobotics.adc9244)

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[16]
X. Chen, F. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, et al. (2025)

Sam 3d: 3dfy anything in images.

arXiv preprint arXiv:2511.16624.

Cited by: [Figure 12](#A1.F12),
[Figure 13](#A1.F13),
[item b)](#A1.I8.i2.p1.1),
[§A-D1](#A1.SS4.SSS1.p1.1),
[§A-E](#A1.SS5.p1.1),
[§I](#S1.p5.1),
[Figure 3](#S2.F3),
[§III-C](#S3.SS3.p3.1).

-
[17]
Y. Chen, C. Wang, Y. Yang, and K. Liu (2024)

Object-centric dexterous manipulation from human motion data.

In 8th Annual Conference on Robot Learning,

Cited by: [§II](#S2.p4.1).

-
[18]
Z. Chen, Q. Yan, Y. Chen, T. Wu, J. Zhang, Z. Ding, J. Li, Y. Yang, and H. Dong (2025)

ClutterDexGrasp: a sim-to-real system for general dexterous grasping in cluttered scenes.

arXiv preprint arXiv:2506.14317.

Cited by: [§I](#S1.p3.1).

-
[19]
B. Cheng, T. Liang, S. Huang, M. Shao, F. Zhang, B. Xu, Z. Xue, and H. Xu (2025)

MoE-dp: an moe-enhanced diffusion policy for robust long-horizon robotic manipulation with skill decomposition and failure recovery.

arXiv preprint arXiv:2511.05007.

Cited by: [§II](#S2.p1.1).

-
[20]
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2023)

Diffusion policy: visuomotor policy learning via action diffusion.

In Robotics: Science and Systems,

Cited by: [§I](#S1.p2.1).

-
[21]
P. Dan, K. Kedia, A. Chao, E. W. Duan, M. A. Pace, W. Ma, and S. Choudhury (2025)

X-sim: cross-embodiment learning via real-to-sim-to-real.

arXiv preprint arXiv:2505.07096.

Cited by: [§I](#S1.p4.1),
[§II](#S2.p4.1).

-
[22]
R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang (2024)

Bunny-visionpro: real-time bimanual dexterous teleoperation for imitation learning.

Cited by: [§II](#S2.p1.1).

-
[23]
H. Fang, B. Romero, Y. Xie, A. Hu, B. Huang, J. Alvarez, M. Kim, G. Margolis, K. Anbarasu, M. Tomizuka, E. Adelson, and P. Agrawal (2025)

DEXOP: a device for robotic transfer of dexterous human manipulation.

arXiv preprint arXiv:2509.04441.

Cited by: [§II](#S2.p2.1).

-
[24]
H. Fang, C. Wang, M. Gou, and C. Lu (2020)

GraspNet-1billion: a large-scale benchmark for general object grasping.

In 2020 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR),

Vol. , pp. 11441–11450.

External Links: [Document](https://dx.doi.org/10.1109/CVPR42600.2020.01146)

Cited by: [§III-D](#S3.SS4.p1.1).

-
[25]
I. Guzey, Y. Dai, G. Savva, R. Bhirangi, and L. Pinto (2024)

Bridging the human to robot dexterity gap through object-oriented rewards.

External Links: 2410.23289,
[Link](https://arxiv.org/abs/2410.23289)

Cited by: [§IV-B](#S4.SS2.p2.1).

-
[26]
A. Handa, A. Allshire, V. Makoviychuk, A. Petrenko, R. Singh, J. Liu, D. Makoviichuk, K. Van Wyk, A. Zhurkevich, B. Sundaralingam, and Y. Narang (2023)

DeXtreme: transfer of agile in-hand manipulation from simulation to reality.

In 2023 IEEE International Conference on Robotics and
Automation (ICRA),

Vol. , pp. 5977–5984.

External Links: [Document](https://dx.doi.org/10.1109/ICRA48891.2023.10160216)

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[27]
A. Handa, K. V. Wyk, W. Yang, J. Liang, Y. Chao, Q. Wan, S. Birchfield, N. Ratliff, and D. Fox (2019)

DexPilot: vision based teleoperation of dexterous robotic hand-arm system.

External Links: 1910.03135,
[Link](https://arxiv.org/abs/1910.03135)

Cited by: [§II](#S2.p1.1).

-
[28]
E. Hsieh, W. Hsieh, Y. Wang, T. Lin, J. Malik, K. Sreenath, and H. Qi (2025)

Learning dexterous manipulation skills from imperfect simulations.

arXiv:2512.02011.

Cited by: [§II](#S2.p2.1).

-
[29]
C. Hsu, B. Wen, J. Xu, Y. Narang, X. Wang, Y. Zhu, J. Biswas, and S. Birchfield (2025)

SPOT: se(3) pose trajectory diffusion for object-centric manipulation.

External Links: 2411.00965,
[Link](https://arxiv.org/abs/2411.00965)

Cited by: [§IV-B](#S4.SS2.p3.1).

-
[30]
A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto (2024)

Open teach: a versatile teleoperation system for robotic manipulation.

arXiv preprint arXiv:2403.07870.

Cited by: [§II](#S2.p1.1).

-
[31]
A. Jain, M. Zhang, K. Arora, W. Chen, M. Torne, M. Z. Irshad, S. Zakharov, Y. Wang, S. Levine, C. Finn, W. Ma, D. Shah, A. Gupta, and K. Pertsch (2025)

PolaRiS: scalable real-to-sim evaluations for generalist robot policies.

External Links: 2512.16881,
[Link](https://arxiv.org/abs/2512.16881)

Cited by: [§III-D](#S3.SS4.p1.1).

-
[32]
S. James, Z. Ma, D. Rovick Arrojo, and A. J. Davison (2020)

RLBench: the robot learning benchmark & learning environment.

IEEE Robotics and Automation Letters.

Cited by: [§III-D](#S3.SS4.p1.1).

-
[33]
C. M. Kim*, B. Yi*, H. Choi, Y. Ma, K. Goldberg, and A. Kanazawa (2025)

PyRoki: a modular toolkit for robot kinematic optimization.

In 2025 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS),

External Links: [Link](https://arxiv.org/abs/2505.03728)

Cited by: [2nd item](#A1.I13.i2.p1.2).

-
[34]
A. Kumar, Z. Fu, D. Pathak, and J. Malik (2021)

Rma: rapid motor adaptation for legged robots.

arXiv preprint arXiv:2107.04034.

Cited by: [§II](#S2.p5.1).

-
[35]
A. S. Lakshmipathy, J. K. Hodgins, and N. S. Pollard (2025)

Kinematic motion retargeting for contact-rich anthropomorphic manipulations.

ACM Transactions on Graphics 44 (2), pp. 1–20.

Cited by: [§II](#S2.p3.1).

-
[36]
C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, C. Wang, G. Levine, W. Ai, B. Martinez, H. Yin, M. Lingelbach, M. Hwang, A. Hiranaka, S. Garlanka, A. Aydin, S. Lee, J. Sun, M. Anvari, M. Sharma, D. Bansal, S. Hunter, K. Kim, A. Lou, C. R. Matthews, I. Villa-Renteria, J. H. Tang, C. Tang, F. Xia, Y. Li, S. Savarese, H. Gweon, C. K. Liu, J. Wu, and L. Fei-Fei (2024)

BEHAVIOR-1k: a human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation.

arXiv preprint arXiv:2403.09227.

Cited by: [§III-D](#S3.SS4.p1.1).

-
[37]
H. Li, L. Sun, Y. Hu, D. Ta, J. Barry, G. Konidaris, and J. Fu (2025)

Novaflow: zero-shot manipulation via actionable flow from generated videos.

arXiv preprint arXiv:2510.08568.

Cited by: [§II](#S2.p3.1).

-
[38]
J. Li, Y. Zhu, Y. Xie, Z. Jiang, M. Seo, G. Pavlakos, and Y. Zhu (2024)

OKAMI: teaching humanoid robots manipulation skills through single video imitation.

In 8th Annual Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p3.1),
[§IV-B](#S4.SS2.p2.1).

-
[39]
K. Li, P. Li, T. Liu, Y. Li, and S. Huang (2025)

Maniptrans: efficient dexterous bimanual manipulation transfer via residual learning.

In Proceedings of the Computer Vision and Pattern
Recognition Conference,

pp. 6991–7003.

Cited by: [§II](#S2.p4.1).

-
[40]
T. Lin, Z. Yin, H. Qi, P. Abbeel, and J. Malik (2024)

Twisting lids off with two hands.

arXiv:2403.02338.

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[41]
T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik (2024)

Learning visuotactile skills with two multifingered hands.

arXiv:2404.16823.

Cited by: [§II](#S2.p1.1).

-
[42]
T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik (2025)

Learning visuotactile skills with two multifingered hands.

In 2025 IEEE International Conference on Robotics and
Automation (ICRA),

pp. 5637–5643.

Cited by: [§II](#S2.p1.1).

-
[43]
M. Liu, D. Pathak, and A. Agarwal (2025)

LocoFormer: generalist locomotion via long-context adaptation.

arXiv preprint arXiv:2509.23745.

Cited by: [§II](#S2.p5.1).

-
[44]
X. Liu, H. Wang, and L. Yi (2025)

DexNDM: closing the reality gap for dexterous in-hand rotation via joint-wise neural dynamics model.

External Links: 2510.08556,
[Link](https://arxiv.org/abs/2510.08556)

Cited by: [§II](#S2.p2.1),
[§II](#S2.p5.1).

-
[45]
T. G. W. Lum, O. Y. Lee, C. K. Liu, and J. Bohg (2025)

Crossing the human-robot embodiment gap with sim-to-real rl using one human demonstration.

External Links: 2504.12609,
[Link](https://arxiv.org/abs/2504.12609)

Cited by: [§A-G](#A1.SS7.p1.1),
[§I](#S1.p4.1),
[§II](#S2.p4.1),
[§III-C](#S3.SS3.p2.1),
[§IV-B](#S4.SS2.p2.1),
[§IV-C](#S4.SS3.p1.1).

-
[46]
T. G. W. Lum, M. Matak, V. Makoviychuk, A. Handa, A. Allshire, T. Hermans, N. D. Ratliff, and K. V. Wyk (2024)

DextrAH-g: pixels-to-action dexterous arm-hand grasping with geometric fabrics.

In 8th Annual Conference on Robot Learning,

External Links: [Link](https://openreview.net/forum?id=S2Jwb0i7HN)

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[47]
J. Luo, C. Xu, F. Liu, L. Tan, Z. Lin, J. Wu, P. Abbeel, and S. Levine (2024)

FMB: a functional manipulation benchmark for generalizable robotic learning.

External Links: 2401.08553,
[Link](https://arxiv.org/abs/2401.08553)

Cited by: [§III-D](#S3.SS4.p1.1).

-
[48]
V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State (2021)

Isaac gym: high performance gpu-based physics simulation for robot learning.

External Links: 2108.10470,
[Link](https://arxiv.org/abs/2108.10470)

Cited by: [§A-B1](#A1.SS2.SSS1.p1.1).

-
[49]
Z. Mandi, Y. Hou, D. Fox, Y. Narang, A. Mandlekar, and S. Song (2025)

DexMachina: functional retargeting for bimanual dexterous manipulation.

External Links: 2505.24853,
[Link](https://arxiv.org/abs/2505.24853)

Cited by: [§II](#S2.p4.1).

-
[50]
J. Mao, S. He, H. Wu, Y. You, S. Sun, Z. Wang, Y. Bao, H. Chen, L. Guibas, V. Guizilini, et al. (2025)

Robot learning from a physical world model.

arXiv preprint arXiv:2511.07416.

Cited by: [§II](#S2.p4.1).

-
[51]
D. Morris, H. Tan, F. Barbagli, T. Chang, and K. Salisbury (2007)

Haptic feedback enhances force skill learning.

In Second Joint EuroHaptics Conference and Symposium on
Haptic Interfaces for Virtual Environment and Teleoperator Systems (WHC’07),

Vol. , pp. 21–26.

External Links: [Document](https://dx.doi.org/10.1109/WHC.2007.65)

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[52]
P. Naughton, J. Cui, K. Patel, and S. Iba (2024)

ResPilot: teleoperated finger gaiting via gaussian process residual learning.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p2.1).

-
[53]
A. M. Okamura (2009)

Haptic feedback in robot-assisted minimally invasive surgery.

Current opinion in urology 19 (1), pp. 102–107.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[54]
OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba (2019)

Learning dexterous in-hand manipulation.

External Links: 1808.00177,
[Link](https://arxiv.org/abs/1808.00177)

Cited by: [§I](#S1.p3.1).

-
[55]
C. Pacchierotti and D. Prattichizzo (2023)

Cutaneous/tactile haptic feedback in robotic teleoperation: motivation, survey, and perspectives.

IEEE Transactions on Robotics 40, pp. 978–998.

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[56]
Y. Park, G. B. Margolis, and P. Agrawal (2024)

Position: automatic environment shaping is the next frontier in rl.

In Forty-first International Conference on Machine Learning,

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[57]
S. Patel, S. Mohan, H. Mai, U. Jain, S. Lazebnik, and Y. Li (2025)

Robotic manipulation by imitating generated videos without physical demonstrations.

arXiv preprint arXiv:2507.00990.

Cited by: [§II](#S2.p3.1),
[§IV-B](#S4.SS2.p3.1).

-
[58]
G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik (2024)

Reconstructing hands in 3D with transformers.

In CVPR,

Cited by: [Figure 14](#A1.F14),
[§A-G](#A1.SS7.p1.1),
[§IV-B](#S4.SS2.p2.1).

-
[59]
A. Petrenko, 1. Allshire, 1. State, 1. Handa, and 1. Makoviychuk (2023)

DexPBT: scaling up dexterous manipulation for hand-arm systems with population based training.

CoRR abs/2305.12127.

External Links: [Link](https://doi.org/10.48550/arXiv.2305.12127),
2305.12127

Cited by: [§A-B1](#A1.SS2.SSS1.p1.1),
[Appendix A](#A1.p7.7),
[§III-B](#S3.SS2.p2.13),
[§III-B](#S3.SS2.p2.14).

-
[60]
L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel (2018)

Asymmetric actor critic for image-based robot learning.

In Robotics: Science and Systems XIV, Carnegie Mellon
University, Pittsburgh,
Pennsylvania, USA, June 26-30, 2018, H. Kress-Gazit, S. S. Srinivasa, T. Howard, and N. Atanasov (Eds.),

External Links: [Link](http://www.roboticsproceedings.org/rss14/p08.html),
[Document](https://dx.doi.org/10.15607/RSS.2018.XIV.008)

Cited by: [§III-B](#S3.SS2.p8.1),
[Figure 8](#S4.F8),
[§IV-E](#S4.SS5.p1.1),
[§IV-E](#S4.SS5.p2.1).

-
[61]
W. Pumacay, I. Singh, J. Duan, R. Krishna, J. Thomason, and D. Fox (2024)

THE colosseum: a benchmark for evaluating generalization for robotic manipulation.

Cited by: [§III-D](#S3.SS4.p1.1).

-
[62]
H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik (2022)

In-Hand Object Rotation via Rapid Motor Adaptation.

In Conference on Robot Learning (CoRL),

Cited by: [§II](#S2.p5.1),
[§III-B](#S3.SS2.p4.1).

-
[63]
Y. Qin, W. Yang, B. Huang, K. V. Wyk, H. Su, X. Wang, Y. Chao, and D. Fox (2024)

AnyTeleop: a general vision-based dexterous robot arm-hand teleoperation system.

External Links: 2307.04577,
[Link](https://arxiv.org/abs/2307.04577)

Cited by: [§II](#S2.p1.1).

-
[64]
R. Qiu, S. Yang, X. Cheng, C. Chawla, J. Li, T. He, G. Yan, D. J. Yoon, R. Hoque, L. Paulsen, G. Yang, J. Zhang, S. Yi, G. Shi, and X. Wang (2025)

Humanoid policy ˜ human policy.

arXiv preprint arXiv:2503.13441.

Cited by: [§IV-B](#S4.SS2.p2.1).

-
[65]
N. Ravi, V. Gabeur, Y. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer (2024)

SAM 2: segment anything in images and videos.

arXiv preprint arXiv:2408.00714.

External Links: [Link](https://arxiv.org/abs/2408.00714)

Cited by: [Figure 12](#A1.F12),
[Figure 14](#A1.F14),
[item a)](#A1.I8.i1.p1.1),
[item c)](#A1.I8.i3.p1.1),
[§A-G](#A1.SS7.p1.1),
[§III-C](#S3.SS3.p3.1).

-
[66]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347.

Cited by: [§III-B](#S3.SS2.p6.1),
[Figure 8](#S4.F8),
[§IV-E](#S4.SS5.p1.1),
[§IV-E](#S4.SS5.p2.1).

-
[67]
K. Shaw, Y. Li, J. Yang, M. K. Srirama, R. Liu, H. Xiong, R. Mendonca, and D. Pathak (2024)

Bimanual dexterity for complex tasks.

In 8th Annual Conference on Robot Learning,

Cited by: [§II](#S2.p1.1).

-
[68]
Z. Si, K. L. Zhang, Z. Temel, and O. Kroemer (2024-07)

Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand.

In Proceedings of Robotics: Science and Systems,

Delft, Netherlands.

External Links: [Document](https://dx.doi.org/10.15607/RSS.2024.XX.128)

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[69]
R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. Van Wyk (2024)

Dextrah-rgb: visuomotor policies to grasp anything with dexterous hands.

arXiv preprint arXiv:2412.01791.

Cited by: [§II](#S2.p5.1).

-
[70]
R. Singh, K. Van Wyk, P. Abbeel, J. Malik, N. Ratliff, and A. Handa (2025)

End-to-end rl improves dexterous grasping policies.

arXiv preprint arXiv:2509.16434.

Cited by: [§II](#S2.p5.1).

-
[71]
J. Singla, A. Agarwal, and D. Pathak (2024-07)

SAPG: split and aggregate policy gradients.

In Proceedings of the 41st International Conference on
Machine Learning (ICML 2024),

Proceedings of Machine Learning Research, Vienna, Austria.

Cited by: [§III-B](#S3.SS2.p6.1),
[Figure 8](#S4.F8),
[§IV-E](#S4.SS5.p1.1),
[§IV-E](#S4.SS5.p2.1).

-
[72]
A. Sivakumar, K. Shaw, and D. Pathak (2022)

Robotic telekinesis: learning a robotic hand imitator by watching humans on youtube.

arXiv preprint arXiv:2202.10448.

Cited by: [§II](#S2.p1.1).

-
[73]
P. Ström, L. Hedman, L. Särnå, A. Kjellin, T. Wredmark, and L. Felländer-Tsai (2006-10)

Early exposure to haptic feedback enhances performance in surgical simulator training: a prospective randomized crossover study in surgical residents.

Surgical endoscopy 20, pp. 1383–8.

External Links: [Document](https://dx.doi.org/10.1007/s00464-005-0545-3)

Cited by: [§I](#S1.p2.1),
[§II](#S2.p1.1).

-
[74]
T. Tao, M. K. Srirama, J. J. Liu, K. Shaw, and D. Pathak (2025)

Dexwild: dexterous human interactions for in-the-wild robot policies.

arXiv preprint arXiv:2505.07813.

Cited by: [§II](#S2.p2.1).

-
[75]
S. Uppal, A. Agarwal, H. Xiong, K. Shaw, and D. Pathak (2024)

Spin: simultaneous perception interaction and navigation.

In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition,

pp. 18133–18142.

Cited by: [§II](#S2.p5.1).

-
[76]
W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang (2023)

UniDexGrasp++: improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning.

arXiv preprint arXiv:2304.00464.

Cited by: [§III-D](#S3.SS4.p1.1).

-
[77]
C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu (2024)

Dexcap: scalable and portable mocap data collection system for dexterous manipulation.

arXiv preprint arXiv:2403.07788.

Cited by: [§II](#S2.p1.1).

-
[78]
J. Wang, Y. Yuan, H. Che, H. Qi, Y. Ma, J. Malik, and X. Wang (2024)

Lessons from learning to spin “pens”.

In CoRL,

Cited by: [§I](#S1.p3.1),
[§II](#S2.p5.1).

-
[79]
Y. Wang, J. Ye, C. Xiao, Y. Zhong, H. Tao, H. Yu, Y. Liu, J. Yu, and Y. Ma (2025)

DexH2R: a benchmark for dynamic dexterous grasping in human-to-robot handover.

External Links: 2506.23152,
[Link](https://arxiv.org/abs/2506.23152)

Cited by: [§III-D](#S3.SS4.p1.1).

-
[80]
B. Wen, W. Yang, J. Kautz, and S. Birchfield (2024-06)

FoundationPose: unified 6d pose estimation and tracking of novel objects.

In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR),

pp. 17868–17879.

Cited by: [§A-F](#A1.SS6.p2.1),
[§I](#S1.p5.1),
[Figure 3](#S2.F3),
[§III-C](#S3.SS3.p3.1),
[§III-C](#S3.SS3.p4.2).

-
[81]
P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel (2024)

GELLO: a general, low-cost, and intuitive teleoperation framework for robot manipulators.

In 2024 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS),

Vol. , pp. 12156–12163.

External Links: [Document](https://dx.doi.org/10.1109/IROS58592.2024.10801581)

Cited by: [§I](#S1.p2.1).

-
[82]
M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song (2025)

DexUMI: using human hand as the universal manipulation interface for dexterous manipulation.

arXiv preprint arXiv:2505.21864.

Cited by: [§II](#S2.p2.1).

-
[83]
Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. (2023)

UniDexGrasp: universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy.

arXiv preprint arXiv:2303.00938.

Cited by: [§III-D](#S3.SS4.p1.1).

-
[84]
J. Ye, K. Wang, C. Yuan, R. Yang, Y. Li, J. Zhu, Y. Qin, X. Zou, and X. Wang (2025)

Dex1B: learning with 1b demonstrations for dexterous manipulation.

In Robotics: Science and Systems (RSS),

Cited by: [§II](#S2.p5.1),
[§III-D](#S3.SS4.p1.1).

-
[85]
B. Yi, C. M. Kim, J. Kerr, G. Wu, R. Feng, A. Zhang, J. Kulhanek, H. Choi, Y. Ma, M. Tancik, and A. Kanazawa (2025)

Viser: imperative, web-based 3d visualization in python.

arXiv preprint arXiv:2507.22885.

Cited by: [1st item](#A1.I12.i1.p1.1).

-
[86]
Z. Yin, C. Wang, L. Pineda, F. Hogan, K. Bodduluri, A. Sharma, P. Lancaster, I. Prasad, M. Kalakrishnan, J. Malik, M. Lambeta, T. Wu, P. Abbeel, and M. Mukadam (2025)

DexterityGen: foundation controller for unprecedented dexterity.

External Links: 2502.04307,
[Link](https://arxiv.org/abs/2502.04307)

Cited by: [§II](#S2.p2.1).

-
[87]
K. Zakka, B. Tabanpour, Q. Liao, M. Haiderbhai, S. Holt, J. Y. Luo, A. Allshire, E. Frey, K. Sreenath, L. A. Kahrs, C. Sferrazza, Y. Tassa, and P. Abbeel (2025)

MuJoCo playground: an open-source framework for gpu-accelerated robot learning and sim-to-real transfer..

GitHub.

External Links: [Link](https://github.com/google-deepmind/mujoco_playground)

Cited by: [§A-B1](#A1.SS2.SSS1.p2.1).

-
[88]
Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu (2024)

3D diffusion policy: generalizable visuomotor policy learning via simple 3d representations.

In Robotics: Science and Systems,

Cited by: [§II](#S2.p1.1).

-
[89]
J. Zhang, H. Liu, D. Li, X. Yu, H. Geng, Y. Ding, J. Chen, and H. Wang

DexGraspNet 2.0: learning generative dexterous grasping in large-scale synthetic cluttered scenes.

In 8th Annual Conference on Robot Learning,

Cited by: [§III-D](#S3.SS4.p1.1).

-
[90]
T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)

Learning fine-grained bimanual manipulation with low-cost hardware.

arXiv preprint arXiv:2304.13705.

Cited by: [§I](#S1.p2.1).

-
[91]
Y. Zhu, A. Lim, P. Stone, and Y. Zhu (2024)

Vision-based manipulation from single human video with open-world object graphs.

arXiv preprint arXiv:2405.20321.

Cited by: [§IV-B](#S4.SS2.p3.1).

### -A Reward Function Details

![Figure](x5.png)

*Figure 9: Object Keypoints. (Left) Visualization of 4 pose keypoints in the local object frame. (Right) Visualization of the keypoint distances used to compute the distance $d(\bm{o}_{t},\bm{g})$.*

![Figure](x6.png)

*Figure 10: Representative Examples of DexToolBench Tasks. Visual breakdown of representative tasks in DexToolBench across the 6 tool categories, highlighting the diversity of objects and manipulation tasks.*

![Figure](x7.png)

*Figure 11: Visualization of Policy Observations during Real-World Deployment. (Top) Image frames from a real-world rollout of the brush manipulation task. (Bottom) The corresponding visualization of the policy observations at each timestep, consisting of the robot state, estimated object pose, and goal pose (green). When the distance between the current pose and goal pose is sufficiently small, the goal pose is updated to the next pose in the goal sequence. Note that this visualization is a rendering of the policy’s inputs, not a physics-based simulation.*

## Appendix A Reward Function Details

The reward function $r$ has three primary components:

$$r=r_{\text{smooth}}+r_{\text{grasp}}+\mathbb{I}_{\text{grasped}}r_{\text{goal}}$$ \tag{3}

where $\mathbb{I}_{\text{grasped}}$ is an indicator function that
triggers once the object has been successfully grasped. We describe
each reward term in further detail below:

Smoothness Reward ($r_{\text{smooth}}$). To promote
physically plausible control and reduce hardware wear, we penalize
the $L_{1}$ norm of joint velocities:

$$r_{\text{smooth}}=-\lambda_{\text{arm}}\|\dot{\mathbf{q}}_{\text{arm}}\|_{1}-\lambda_{\text{hand}}\|\dot{\mathbf{q}}_{\text{hand}}\|_{1}$$ \tag{4}

where $\dot{\mathbf{q}}^{\text{arm}}$ and
$\dot{\mathbf{q}}^{\text{hand}}$ correspond to the current velocities
of the 7-DOF Kuka arm and the 22-DoF Sharpa hand, respectively.

Grasp Reward ($r_{\text{grasp}}$). The term
$r_{\text{grasp}}$ facilitates the transition from a neutral pose to
a stable grasp:

$$r_{\text{grasp}}=r_{\text{approach}}+(1-\mathbb{I}_{\text{grasped}})r_{\text{lift}}$$ \tag{5}

$r_{\text{approach}}$ encourages the agent to reduce the distance
between the robot’s hand and the object. It rewards the agent for
approaching the object rather than simply maintaining a fixed close
distance. We define:

$$r_{\text{approach}}=\lambda_{\text{approach}}\max(\bar{d}^{*}_{\text{ft}}-\bar{d}_{\text{ft}},0)$$ \tag{6}

where $\bar{d}_{\text{ft}}$ is the current mean distance between the
fingertips and the object, and $\bar{d}^{*}_{\text{ft}}$ is a
stateful variable that stores the minimum mean distance achieved so
far in the episode. $r_{\text{lift}}$ encourages the agent to grasp
and lift the object. We define:

$$r_{\text{lift}}=\lambda_{\text{lift}}\max(z-z_{\text{init}},0)+\mathbb{I}[z\geq z_{\text{lifted}}]B_{\text{lifted}}$$ \tag{7}

where $z$ is the vertical position of the object, $z_{\text{init}}$
is the initial $z$ position of the object, $z_{\text{lifted}}$ is the
lifted threshold, and $B_{\text{lifted}}$ is a bonus that is awarded
at most once per episode when the object has been lifted.
$\mathbb{I}_{\text{grasped}}$ turns true once $z\geq z_{\text{lifted}}$. This ensures that this lifting reward only
applies at the start of the episode before the object has been
lifted, and then is 0 for the remainder of the episode. At this
point, $r_{\text{goal}}$ takes over.

Goal-Pose Reward ($r_{\text{goal}}$). Once
$\mathbb{I}_{\text{grasped}}=1$, the goal-reaching term
$r_{\text{goal}}$ becomes the dominant signal to train our RL policy.
As Section [III-B](#S3.SS2) describes, we define:

| | $$r_{\rm goal}=\textrm{max}(d^{*}-d(\bm{o}_{t},\bm{g}),0)+B_{\rm succ}\,\mathbb{I}\!\left[d(\bm{o}_{t},\bm{g})Keypoint Distance Formulation:
Following [[[4](#bib.bib2), [59](#bib.bib133)]], we
measure distance using $D{=}4$ object-frame keypoints $d(\bm{o}_{t},\bm{g})=\max_{i}\left\|\bm{o}_{t,i}-\bm{g}_{i}\right\|,$
where $\bm{o}_{t,i}$ and $\bm{g}_{i}$ are the world-frame positions of
the $i$-th keypoint under the current and goal poses, respectively.
We define the following scales $\mathbf{s}^{\text{rew}}=[s_{x}^{\text{rew}},s_{y}^{\text{rew}},s_{z}^{\text{rew}}]=[0.14,0.03,0.03]$ (in metres), and we define the keypoints
$\{\mathbf{k}_{i}\}_{i=1}^{4}$ at the following offsets:

$$\mathbf{k}\in\left\{\begin{bmatrix}s_{x}^{\text{rew}}/2\\ s_{y}^{\text{rew}}/2\\ s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}s_{x}^{\text{rew}}/2\\ s_{y}^{\text{rew}}/2\\ -s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}-s_{x}^{\text{rew}}/2\\ -s_{y}^{\text{rew}}/2\\ s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}-s_{x}^{\text{rew}}/2\\ -s_{y}^{\text{rew}}/2\\ -s_{z}^{\text{rew}}/2\end{bmatrix}\right\}$$ \tag{9}

Fig. [9](#A0.F9) visualizes our keypoint-based pose
representation. Defining pose distance in this space yields a single,
interpretable metric that jointly captures translation and rotation
error. Since many tools are elongated, we set $s_{x}^{\text{rew}}>s_{y}^{\text{rew}},s_{z}^{\text{rew}}$ so the reward is more sensitive
to pitch and yaw errors (rotations about the $y$ and $z$ axes) than
to roll about the long $x$-axis. This biases the policy toward
aligning the tool’s principal axis with the target pose, which is
typically the dominant requirement for tool use. Using fixed relative
scales across objects also ensures a consistent trade-off between
translational and rotational progress for all tasks.

### A-A Procedural Asset Generation Details

To promote robustness to a wide range of inertial properties, we
procedurally generate tool-like objects using a simple
handle–head* abstraction. This minimal design still spans a
broad family of real-world handheld tools (e.g., brushes, markers,
spatulas, screwdrivers, and hammers). The goal is to expose the
policy to large variations in geometry and physics without relying on
complex meshes that slow down simulation and training. Each tool is
composed of cuboids and cylinder primitives, which ensures stable
physics and fast simulation.

Tool parameterization.
Each asset consists of (i) a handle and (ii) a head rigidly attached
to one end of the handle. We randomize both geometry and density. For
each part, we sample its length $L$ and cross-sectional dimensions.
For cuboids we use width $W$ and height $H$; for cylindrical/capsule
variants, $W$ denotes the diameter and we set $H=W$.

Geometric randomizations.
Each tool is a rigid union of two parts: a *handle* and a
*head*. For each part, we randomly choose one of two primitive
shapes: a *cuboid* (box) or a *capsule* (cylinder with
rounded ends). We then sample the part dimensions uniformly from
ranges chosen to span common handheld tools (from small
brushes/markers to larger hammers). The head is attached to one end
of the handle, and the head’s long axis is rotated by $90^{\circ}$
relative to the handle’s long axis. For a *cuboid*, we sample a
length, width, and height. For a *capsule*, we sample a length
and a diameter:

*Handle dimensions:* length is sampled in $[5,30]$ cm;
width/height (or diameter) is sampled in $[1,4]$ cm.

*Head dimensions:* length is sampled in $[1,15]$ cm;
width/height (or diameter) is sampled in $[0.5,12]$ cm.

Physics randomizations.
To systematically induce physics randomizations, we assign different
densities to the handle and head. Handles are sampled with a lower density,
$\rho_{\text{low}}\sim\mathcal{U}[300,600]\ \text{kg/m}^{3}$,
approximating wood or plastics. Heads are sampled with a higher density range
$\rho_{\text{high}}\sim\mathcal{U}[300,2000]\ \text{kg/m}^{3}$,
approximating metals or dense rubber. Together with geometric
variation, this density variation results in diverse center-of-mass locations
(often shifted toward the head) and a broad range of rotational
inertias, encouraging the policy to learn to adapt under varying
physics in the real-world.

### A-B Simulation Training Details

#### A-B1 Simulation Details

We use
IsaacGym [[[48](#bib.bib57)]] for massively-parallel
GPU-accelerated simulation. We build from the DexPBT Kuka Allegro
Reorientation environment [[[59](#bib.bib133)]], and add
substantial changes to the environment to improve dexterous tool-use,
enable sim-to-real transfer, and improve generalization to unseen
objects and trajectories.

Notably, many other GPU-accelerated simulators, such as
Genesis [[[10](#bib.bib163)]] and MJWarp [[[87](#bib.bib164)]],
could not be used because they do not currently support simulation of
parallel environments with different object geometries per scene.

#### A-B2 Policy Action

The RL policy outputs an action $\bm{a}_{t}\in\mathbb{R}^{J}$, where
$J=29$ represents the total number of actuated joints. We first clip
the actions to the range $[-1,1]$. We partition this action into arm
components $\bm{a}^{\text{arm}}_{t}\in[-1,1]^{7}$ and hand
components $\bm{a}^{\text{hand}}_{t}\in[-1,1]^{22}$. These are
processed into joint position targets $\bm{q}^{\text{target}}_{t}$ as follows:

Arm Control (Delta). For the 7-DoF arm, we interpret the
action as a relative displacement (delta) from the previous target.
The intermediate target is computed as:

$$\hat{\bm{q}}^{\text{arm}}_{t}=\bm{q}^{\text{target, arm}}_{t-1}+k^{\text{arm}}\bm{a}^{\text{arm}}_{t}$$ \tag{10}

where $k^{\text{arm}}=0.025$ is a scaling factor derived from the
control frequency and speed limits. We clip
$\hat{\bm{q}}^{\text{arm}}_{t}$ to the joint limits, and then apply an
exponential moving average (EMA) filter with smoothing factor
$\alpha_{\text{arm}}$ to obtain the final $\bm{q}^{\text{target, arm}}_{t}$.

Hand Control (Absolute). For the 22-DoF hand, we interpret
the action as an absolute target within the joint limits. We map the
action range $[-1,1]$ to the physical limits
$[\bm{q}^{\text{hand}}_{\text{lower}},\bm{q}^{\text{hand}}_{\text{upper}}]$ via an affine transformation:

$$\hat{\bm{q}}^{\text{hand}}_{t}=\frac{\bm{a}^{\text{hand}}_{t}+1}{2}\odot(\bm{q}^{\text{hand}}_{\text{upper}}-\bm{q}^{\text{hand}}_{\text{lower}})+\bm{q}^{\text{hand}}_{\text{lower}}$$ \tag{11}

We then apply an EMA filter with smoothing factor
$\alpha_{\text{hand}}$ to $\hat{\bm{q}}^{\text{hand}}_{t}$, and finally
clip the result to the joint limits to ensure safety, yielding
$\bm{q}^{\text{target, hand}}_{t}$.

In our experiments, we use $\alpha_{\text{arm}}=\alpha_{\text{hand}}=0.1$, where $\alpha=1.0$ means no smoothing.

#### A-B3 Policy Observation

The policy observation space $\mathcal{O}$ comprises robot
proprioception $\bm{s}_{t}$, object state $\bm{o}_{t}$, goal information
$\bm{g}$, and the object descriptor $\bm{\phi}$. We typically provide
relative representations (e.g., keypoints expressed relative to the
palm) over absolute world positions to facilitate generalization
across tool geometries and improve invariance to absolute world
coordinates. The specific observations are:

Proprioception ($\bm{s}_{t}$).

-
•

Joint positions $\bm{q}_{t}\in\mathbb{R}^{29}$ and velocities
$\dot{\bm{q}}_{t}\in\mathbb{R}^{29}$.

-
•

Previous joint position targets $\bm{q}^{\text{target}}_{t-1}\in\mathbb{R}^{29}$.

-
•

Palm pose, consisting of world position $\bm{x}_{\text{palm}}\in\mathbb{R}^{3}$ and orientation quaternion
$\bm{r}_{\text{palm}}\in\mathbb{R}^{4}$.

-
•

Fingertip positions relative to the palm center:
$\{\bm{x}_{\text{tip},j}-\bm{x}_{\text{palm}}\}_{j=1}^{5}\in\mathbb{R}^{15}$.

Object Pose, Goal Pose, and Descriptor ($\bm{o}_{t},\bm{g},\bm{\phi}$).
Instead of directly inputting raw pose matrices, we represent the
object and goal states using $K=4$ keypoints. These observation
keypoints are computed using the object’s grasp bounding box $\bm{s}\in\mathbb{R}^{3}$ (length, width, height), effectively conditioning
the policy on the specific tool geometry.

-
•

Object orientation quaternion $\bm{r}_{\text{obj}}\in\mathbb{R}^{4}$.

-
•

Object keypoints relative to the palm: $\{\bm{o}_{t,i}-\bm{x}_{\text{palm}}\}_{i=1}^{4}\in\mathbb{R}^{12}$.

-
•

Keypoint errors to goal: $\{\bm{o}_{t,i}-\bm{g}_{i}\}_{i=1}^{4}\in\mathbb{R}^{12}$. Here, $\bm{o}_{t,i}$
and $\bm{g}_{i}$ correspond to the $i$-th keypoint of the current
and goal poses, respectively.

-
•

Object scales $\bm{s}\in\mathbb{R}^{3}$, representing the
dimensions of the tool’s graspable region (handle).

We note that while the reward function (Appendix [A](#A1))
uses fixed keypoint offsets to maintain a consistent success
metric across objects, the policy observations use the
instance-specific offsets defined by the object scales
$\bm{s}=[s_{x},s_{y},s_{z}]$:

$$\mathbf{k}_{\text{obs}}\in\left\{\begin{bmatrix}s_{x}/2\\ s_{y}/2\\ s_{z}/2\end{bmatrix},\begin{bmatrix}s_{x}/2\\ s_{y}/2\\ -s_{z}/2\end{bmatrix},\begin{bmatrix}-s_{x}/2\\ -s_{y}/2\\ s_{z}/2\end{bmatrix},\begin{bmatrix}-s_{x}/2\\ -s_{y}/2\\ -s_{z}/2\end{bmatrix}\right\}$$ \tag{12}

#### A-B4 Asymmetric Critic State

We utilize an asymmetric actor-critic setup where the critic is
trained on a privileged state space $\mathcal{S}$, while the policy
acts on the restricted observation space $\mathcal{O}$. The critic
state $\mathcal{S}$ contains the exact, noise-free, and instantaneous
ground-truth state of the system, whereas the policy observation
$\mathcal{O}$ is subjected to noise and delays to bridge the
sim-to-real gap. In addition to the standard observations, the critic
state includes:

-
•

Ground-truth Velocities: Exact linear and angular velocities
for the palm ($\bm{v}_{\text{palm}},\bm{\omega}_{\text{palm}}$)
and the object ($\bm{v}_{\text{obj}},\bm{\omega}_{\text{obj}}$).

-
•

Reward Signals: The instantaneous reward $r_{t}$ and the number
of successes achieved this episode, which help to improve value
function estimation.

-
•

Stateful Progress Features: Auxiliary features including the
minimum fingertip-to-object distance achieved since the last
reset, the minimum keypoint distance achieved since last reset,
the number of environment steps that have occurred since last
reset, and a binary signal $\mathbb{I}_{\text{grasped}}$
indicating if the object has been lifted.

-
•

Noise-Free and Undelayed Object Pose: Exact object pose
without noise or delays.

#### A-B5 Reset and Initialization

Next, we describe the
termination conditions and initialization of the environment.

Episode Termination Conditions. An environment is reset if
any of the following conditions are met:

-
•

Object Fallen Off Table: The object height drops below the
table surface.

-
•

Object Drop (Hysteresis): To penalize dropping the object
after grasping, we trigger a reset if the object returns to the
table surface ($zafter the grasped
flag $\mathbb{I}_{\text{grasped}}$ has been activated.

-
•

Hand Wander: The distance between the hand and the object
exceeds $1.5$ m.

-
•

Table Force Limits: The net force measured by the table
sensor exceeds $100$ N, preventing aggressive collisions.

-
•

Timeout: The episode duration exceeds the maximum step limit.

-
•

Max Consecutive Successes: The agent achieves the maximum
number of consecutive successes for an episode.

Scene Initialization.
At the start of each episode, the robot and object are reset to
randomized states to encourage robustness:

-
•

Robot: Joint positions are initialized to a default position
with additive uniform noise.

-
•

Object: The object is spawned above the table surface with a
randomized planar position $(x,y)$ and a random rotation
quaternion $\bm{r}\in SO(3)$. We also apply small random
perturbations to the table height to improve robustness to
geometric calibration errors.

Goal Sampling Strategy.

-
1.

Initial Goal ($\bm{g}_{0}$): The first goal is sampled
uniformly from a 3D workspace volume $V$ centered above the
table. The sampling ranges relative to the table center are $x\in[-0.35,0.35]$, $y\in[-0.1,0.2]$, and $z\in[0.15,0.52]$, where $z$ is up, $-y$ is forward (wrt the robot), and $x$
is left (wrt the robot).

-
2.

Subsequent Goals ($\bm{g}_{k+1}$): Upon successfully reaching
goal $\bm{g}_{k}$, we sample the next goal relative to the
previous goal. We apply a random perturbation of up to
$0.1$ m in position and $90^{\circ}$ in rotation.

*TABLE I: Simulation environment and SAPG training hyperparameters.*

#### A-B6 Hyperparameters

Table [I](#A1.T1) shows
the important hyperparameters used for the simulation environment and
SAPG training.

### A-C Sim-to-Real Details

-
•

Observation and action delay: We model system latency by
passing observations, actions, and object states through FIFO
queues and randomly sampling delayed values. To reflect the
uncertainty and inference latency of vision-based object pose
tracking, object pose estimates are assigned a significantly
larger delay than other signals.

-
•

Accurate robot modeling: We explicitly match the simulation’s
joint gains, armature, and damping to those of the physical robot.

-
•

Sensor noise: We inject zero-mean Gaussian noise into joint
velocities and object pose observations, as we found these
signals to exhibit the highest measurement noise in the real
world. The noise magnitudes are calibrated to match empirical
levels, robustifying the policy against realistic sensor noise.

-
•

Smooth actions: We apply exponential moving average (EMA)
smoothing to joint position targets to prevent high-frequency
control artifacts.

-
•

Table height randomization: We randomize the table height at
the start of each episode to improve robustness to geometric
variations in the physical workspace.

-
•

External force and torque perturbations: We apply random
external forces and torques to the object during training to help
the policy generalize to unmodeled contact dynamics and
unexpected disturbances.

-
•

Finger self-collisions: We restrict the range of the finger
abduction/adduction joints to a conservative subset of their
physical limits. This prevents the policy from reaching
configurations that could result in potentially damaging
self-collisions on the real hardware.

### A-D Human Video Processing Details

![Figure](x8.png)

*Figure 12: Metric-Scale Mesh and Grasp Bounding Box Acquisition. We present a semi-automated pipeline to extract a metric-scale mesh and grasp bounding box from a single RGB-D video scan. (1) We first segment the target object in the initial frame and reconstruct a 3D mesh using SAM 3D [[[16](#bib.bib27)]], injecting the captured depth map to ensure metric accuracy. (2) To establish a canonical coordinate frame, we virtually render the mesh from multiple views and use SAM 2 [[[65](#bib.bib85)]] to segment the geometry into “handle” and “head” regions based on user prompts. (3) The final grasp bounding box is derived from the handle’s geometry: it is centered on the handle, with the positive $x$-axis oriented toward the head, ensuring consistent alignment with policy training.*

#### A-D1 Metric-Scale Mesh and Grasp Bounding Box Acquisition

Fig. [12](#A1.F12) illustrates our pipeline, which leverages
SAM 3D [[[16](#bib.bib27)]] to extract a metric-scale mesh and grasp
bounding box from a single RGB-D video demonstration.

-
a)

Initial Object Segmentation: We first obtain a
segmentation mask of the object from the first frame of the video.
We use the first frame because it typically minimizes occlusion, as
this is prior to human interaction. Although segmentation could be
conditioned on a text prompt, we found that prompting SAM
2 [[[65](#bib.bib85)]] with two user-specified points on the object
provided superior reliability.

-
b)

Metric-Scale Mesh Extraction: Next, we generate a 3D
mesh using SAM 3D [[[16](#bib.bib27)]]. Although the default SAM 3D
pipeline relies on monocular depth estimation from RGB input, the
resulting mesh is not metrically accurate. Since we require
accurate scale for both pose tracking and grasp bounding box
estimation, we modify the pipeline to directly utilize the captured
depth image instead of the predicted depth.

-
c)

Part Segmentation (Handle vs. Head): To define the
object’s origin, we segment the mesh into a handle
(graspable region) and a head (non-graspable region). This
decomposition allows us to center the grasp bounding box on the
handle and align the object’s x-axis along the handle’s primary
axis (x-axis pointing from handle to head). We automate this by
rendering a video of the mesh from a virtual camera circling the
object. The user provides point prompts on the first frame to
define the handle and head, and we use SAM 2 [[[65](#bib.bib85)]] to
propagate these masks across the rendered sequence.

-
d)

Grasp Bounding Box Definition: Using the rendered
camera extrinsics and intrinsics, we back-project the masked depth
maps to 3D to obtain separate point clouds for the handle and head.
We crop the full mesh based on these point clouds. Finally, we
define the grasp bounding box using the handle’s geometry: the box
is centered at the handle’s centroid, with its x-axis oriented
along the vector pointing from the handle centroid toward the head centroid.

#### A-D2 Goal Pose Sequence

We use FoundationPose to extract the raw object pose trajectory from
the RGB-D human video demonstration, which is collected at the
camera’s native frequency of 30Hz. Directly tracking this raw
sequence is suboptimal for two reasons: (1) frame-to-frame pose
estimation jitter can make the goal pose sequence shaky, and (2) the
demonstration often includes a pre-grasp phase where the object
remains stationary on the table while the human approaches it. To
address these issues, we apply the following preprocessing steps.

Temporal Downsampling. We downsample the trajectory from
30Hz to 3Hz. This acts as a low-pass filter, removing high-frequency
perception noise and ensuring the policy tracks the underlying smooth
motion profile rather than attempting to replicate spurious artifacts
in the pose estimation.

Lift-off Truncation. We automatically trim the start of the
trajectory to remove the static phase where the object rests on the
table. We calculate the object’s vertical position $z_{t}$ relative to
the table surface at each timestep. The goal sequence is defined to
start at the first frame $t_{\text{start}}$ where the object’s height
exceeds a threshold $z_{\text{thresh}}=10\text{cm}$. This ensures
that the robot immediately attempts to lift and manipulate the tool,
rather than trying to maintain a static pose on the table surface.

*TABLE II: Detailed Real-World Evaluation Results. We report the Task Progress (%) for each of the 5 rollouts across all 24 object-task variations. The specific tools correspond to the instances shown in Fig. [4](#S3.F4).*

### A-E DexToolBench Details

Fig. [10](#A0.F10) visualizes tasks from
DexToolBench. Fig. [13](#A1.F13) visualizes the
real-world objects and SAM 3D [[[16](#bib.bib27)]] generated objects in
DexToolBench.

![Figure](x9.png)

*Figure 13: DexToolBench Objects. (Left): Real-world objects. (Right): SAM 3D [[[16](#bib.bib27)]] generated meshes of these objects.*

Objects. Here, we describe the 12 object instances across
the 6 object categories:

-
•

Hammer:

-
–

Claw Hammer: A 3D printed hammer consisting of a
thin black handle and a gray head.

-
–

Mallet Hammer: A rubber mallet consisting of a
wooden handle and a heavy, cylindrical black head.

-
•

Marker:

-
–

Sharpie Marker: A standard Sharpie permanent
marker with a black cap.

-
–

Staples Marker: A dry-erase Staples marker with
a white barrel. It is slightly thinner than the Sharpie.

-
•

Eraser:

-
–

Handle Eraser: An eraser with a thin handle and
yellow bristles.

-
–

Flat Eraser: An Expo eraser consisting of a rectangular foam block without a handle.

-
•

Brush:

-
–

Blue Brush: A blue brush with a thick black
handle and yellow bristles along the same direction as the handle.

-
–

Red Brush: A red brush with a long black handle
and black bristles at a 90∘ angle with respect to the
handle. It is lighter than the Blue brush.

-
•

Spatula:

-
–

Spoon Spatula: A spoonula with a long,
cylindrical black handle and a shallow oval spoon.

-
–

Flat Spatula: A black spatula with a flat
rectangular blade and thin, rectangular handle.

-
•

Screwdriver:

-
–

Long Screwdriver: A screwdriver with a long
shaft and wooden handle. Alternating yellow, black, and red
pieces of tape are added to the handle to reduce rotational symmetry.

-
–

Short Screwdriver: A screwdriver with a short
shaft and a bulbous handle with a red and black pattern.

Tasks. Here, we describe the 12 tool-use task trajectories
instances across the 6 object categories:

-
•

Hammer:

-
–

Swing Down: Grasp the hammer from flat on the
table, rotate it by 90∘ into a striking configuration,
swing down onto a nail 3 times.

-
–

Swing Side: Grasp the hammer from flat on the
table, rotate it by 90∘ into a striking configuration,
swing sideways onto a nail 3 times.

-
•

Marker:

-
–

Draw Smile: Grasp the marker from flat on the
table, move to the whiteboard, draw two dots and a smile.

-
–

Write C: Grasp the marker from flat on the
table, move to the whiteboard (different location from smile), draw a C.

-
•

Eraser:

-
–

Wipe Smile: Grasp the eraser from flat on the
table, move to the whiteboard, erase the smile.

-
–

Wipe C: Grasp the eraser from flat on the table,
move to the whiteboard (different location from smile), erase the C.

-
•

Brush:

-
–

Sweep Forward: Grasp the brush from flat on the
table, rotate it by 90∘ into a sweeping configuration,
sweep forward 3 times to sweep paper balls into trash.

-
–

Sweep Right: Grasp the brush from flat on the
table, rotate it by 90∘ into a sweeping configuration,
sweep rightwards 3 times to sweep paper balls into trash.

-
•

Spatula:

-
–

Serve Plate: Grasp the spatula from flat on the
plate. Move to the other bowl/plate, perform a scooping
motion, perform a serving motion onto the original plate.

-
–

Flip Over: Grasp the spatula upside-down with
the spatula head on the pan/bowl. Rotate it by 180∘,
perform scooping motion, perform flipping motion.

-
•

Screwdriver:

-
–

Spin Vertical: Grasp the screwdriver from flat
on the table, rotate it by 90∘ into a vertical
configuration, spin it by 360∘ along long axis.

-
–

Spin Horizontal: Grasp the screwdriver from flat
on the table, keep it at a horizontal configuration, spin it
by 360∘ along long axis.

Processed Data. Each task is defined by an RGB-D human
video, captured via a ZED 1 stereo camera. We process each video with
our perception pipeline, described in detail in
Appendix [A-D](#A1.SS4). The processed data includes:

-
•

Raw RGB-D video

-
•

Object segmentation masks

-
•

Metric-scale object meshes

-
•

6D object pose trajectories

Additionally, we provide:

-
•

Visualization scripts to visualize the data in 3D space using
Viser [[[85](#bib.bib165)]].

-
•

Simulation scripts to evaluate policies on these tasks. This
will support multiple robot arms and hands.

-
•

Links to purchase the real-world objects

Task Progress. For each task, we evaluate
Task Progress, defined as the percentage of object pose
waypoints in the demonstrated trajectory that the robot successfully
reached. We consider a goal pose to be reached if the distance
between the object pose and goal pose $d(\bm{o}_{t},\bm{g})$ is below
our defined success tolerance $\epsilon=2cm$.

This evaluation is performed closed-loop: the goal advances
to the next waypoint only after the current waypoint is reached. This
contrasts with open-loop playback, where goals advance at a
fixed frequency regardless of the object’s state. Our closed-loop
protocol decouples execution speed from spatial accuracy, allowing
the policy to utilize retry behaviors and execute as fast or slow as
it needs to without being penalized for failing to match the exact
timing profile of the demonstration.

![Figure](x10.png)

*Figure 14: Kinematic Retargeting Pipeline. From the RGB-D human video, we use SAM 2 [[[65](#bib.bib85)]] for hand masks and HaMeR [[[58](#bib.bib59)]] for hand pose prediction. Next, we use ICP registration to align the hand pose prediction with the segmented hand point cloud to obtain accurate 3D hand poses. Lastly, we perform IK-based retargeting of the arm and hand to match the human wrist pose and fingertip positions.*

### A-F Real-World Experiment Additional Analysis

Table [II](#A1.T2) shows the per-rollout Task
Progress of the results shown in
Fig. [4](#S3.F4).
Fig. [11](#A0.F11) visualizes the policy
observations at each timestep during real-world deployment.

Perception Failures. We observe that object pose estimation
is the most common failure mode of our system (43.7% of failures).
Specifically, FoundationPose [[[80](#bib.bib58)]] struggles
under three conditions common in dexterous manipulation: (1) heavy
object occlusion, which is exacerbated when the robot hand
manipulates small objects (e.g., the small screwdriver); (2) visual
ambiguity due to rotational symmetry (e.g., the cylindrical body of
the marker); and (3) low visual contrast between the object and the
environment (e.g., dark bristles on a black surface). While our
policy is trained to be robust to significant pose noise, it cannot
recover from catastrophic tracking failures where the estimator loses
the object entirely. Future work could address this by incorporating
additional camera views or leveraging temporal consistency in 2D
tracking to improve state estimation stability.

Manipulation Failures. Manipulation failures primarily stem
from two sources: object drops (34.5%) and failure to reach the goal
pose due to incomplete in-hand rotation (18.2%). Object drops were
most common on heavy objects (e.g., the 331g mallet hammer
and the 325g blue brush), typically occurring during
reorientation or environment contact. Failure to reach the goal pose
due to incomplete in-hand rotation was most common on thin objects
(e.g., the $\sim$1cm thick flat spatula), as the policy
would repeatedly attempt in-hand rotation but fail to manipulate the
thin geometry. Grasp failures were rare (3.6%) but occasionally
occurred with the marker due to its thin shape. In these instances,
the marker’s cylindrical shape often caused it to roll off of the
table, so the policy was unable to recover.

### A-G Kinematic Retargeting Baseline Details

Fig. [14](#A1.F14) visualizes the
kinematic retargeting pipeline. Following
[Lum et al. [[45](#bib.bib131)]], we perform hand pose
estimation on each frame of the human video and then retarget these
hand poses to a dexterous robot. Specifically, we first use
HaMeR [[[58](#bib.bib59)]] to predict initial hand
keypoints and mesh vertices from RGB images. Next, we refine this
prediction by using the corresponding depth image and a mask of the
hand (from SAM 2 [[[65](#bib.bib85)]]) to extract a 3D hand point
cloud, and then aligning the HaMeR mesh to this point cloud via
Iterative Closest Point (ICP) registration. Finally, we retarget the
robot to these poses in two stages: we first compute the arm
configuration using Damped Least Squares [[[11](#bib.bib166)]]
to place the palm, and subsequently optimize the hand finger joints
to reach the fingertip targets relative to the palm.

### A-H Fixed-Grasp Baseline Details

![Figure](x11.png)

*Figure 15: Fixed Grasp Baselines. We first use our SimToolReal policy to grasp and lift the object. We then attempt to follow the trajectory using a fixed grasp. Option 1 (Damped Least Squares): Tracks the target poses but causes severe collisions with the table. Option 2 (Collision-Free Trajectory Optimization): Avoids collisions but fails to reach the target poses.*

To run this baseline, we first use our SimToolReal policy to
grasp and lift the object to the initial goal pose. Once the first
goal pose is reached, we stop running the policy and begin planning a
fixed grasp trajectory to perform the object-trajectory following
task. For the hand, we maintain a fixed grasp by storing the current
hand joint position targets and continuing to apply these joint
position targets to the hand. For the arm, we plan a motion that
moves the object along the goal trajectory, assuming its relative
pose to the end-effector remains constant.

Problem Formulation. Specifically, we assume a rigid grasp
where the transformation $T_{EO}$ between the end-effector frame $E$
and the object frame $O$ remains constant throughout the trajectory.
Let $T_{BO}^{(i)}$ denote the $i$-th target object pose in the robot
base frame $B$ from a goal sequence of length $N$. The corresponding
target end-effector pose $T_{BE}^{(i)}$ is derived as:

$$T_{BE}^{(i)}=T_{BO}^{(i)}\left(T_{EO}\right)^{-1}$$ \tag{13}

We test two methods for computing the arm joint trajectory
$\bm{q}_{1:N}$ that reach these target end-effector poses.

-
•

Damped Least Squares.
For each waypoint, we compute the arm Jacobian $J(\bm{q})$ and
the pose error vector $\bm{e}$ (concatenating translation and
axis-angle rotation error). We compute the iterative joint update
$\Delta\bm{q}$ using the Damped Least Squares
method [[[11](#bib.bib166)]]:

$$\Delta\bm{q}=J^{T}(JJ^{T}+\lambda^{2}I)^{-1}\bm{e}$$ \tag{14}

where the damping factor $\lambda$ smooths the motion and
prevents discontinuities. While this method is computationally
efficient (solving in seconds), it is strictly local and does not
account for environmental obstacles (e.g., the table surface or
whiteboard), often leading to collisions.

-
•

Collision-Free Trajectory Optimization.
To address environment collisions, we utilize
PyRoki [[[33](#bib.bib162)]], which uses a JAX-based
Levenberg-Marquardt solver for trajectory optimization. We extend
its standard trajectory optimization formulation to optimize a
trajectory with $N$ waypoints instead of 1. This method incurs a
higher computation time ($\sim$40–60 seconds per trajectory) and
requires a time-consuming process of modeling the robot as a set
of collision spheres.

-
1.

Initialization: We warm-start the optimization
by sequentially solving collision-free Inverse Kinematics
(IK) for each waypoint, seeding each step with the solution
from the previous waypoint to encourage temporal consistency.

-
2.

Optimization: The solver minimizes a cost
function composed of target pose error and smoothness (joint
velocity) terms, subject to constraints: joint
position/velocity limits, self-collision avoidance, and
environment collision avoidance.

Fig. [15](#A1.F15) visualizes the trajectories
produced by both methods. We use the Collision-Free Trajectory
Optimization for all reported baseline experiments, as the simple
Damped Least Squares method frequently fails due to significant table
collisions during complex object rotations.

### A-I Specialist Details

![Figure](appendix/appendix_images/per_tool_figures/brush_comparison.png)
![Figure](appendix/appendix_images/per_tool_figures/eraser_comparison.png)
![Figure](appendix/appendix_images/per_tool_figures/hammer_comparison.png)
![Figure](appendix/appendix_images/per_tool_figures/marker_comparison.png)
![Figure](appendix/appendix_images/per_tool_figures/screwdriver_comparison.png)
![Figure](appendix/appendix_images/per_tool_figures/spatula_comparison.png)

*Figure 16: Detailed Comparison against Specialists. We provide a breakdown of the results in Fig. [6](#S4.F6). Each plot compares SimToolReal to a specialist policy trained on a single object and trajectory (Obj A / Traj A) for a single object category.*

Fig. [16](#A1.F16) presents a granular breakdown
of the aggregate results shown in
Fig. [6](#S4.F6). While the main text reports the
average performance across all categories, this figure details the
specific generalization capabilities of the specialist policy trained
for each of the six tool-use categories individually (e.g., Hammer,
Spatula, Brush). Each specialist policy uses the same architecture,
observation space, and action space as used by SimToolReal.
It uses the same reward function, termination criteria, and episode
initialization settings. The only changes are (i) Fixed
Object Geometry: instead of training on diverse, procedurally
generated primitive objects, the specialist is trained exclusively on
a single object instance (Obj A), and (ii) Fixed Task
Trajectory: instead of training on randomly sampled goal poses, the
specialist is trained on a single, fixed sequence of goal poses (Traj
A) extracted from a specific human demonstration.

Table [III](#A1.T3) details the specific object
instances and task trajectories that define Obj A, Traj A, Obj B, and
Traj B for each tool category.

*TABLE III: Specialist Objects and Trajectories. Objects and trajectories used for specialist policy training and evaluation.*

Generated on Tue Feb 24 17:16:39 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)