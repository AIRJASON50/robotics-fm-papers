# **Twisting Lids Off with Two Hands**

**Authors:** Toru Lin, Zhao-Heng Yin, Haozhi Qi, Pieter Abbeel, Jitendra Malik

**Affiliation:** University of California, Berkeley

**Correspondence:** {toru, zhaohengyin}@berkeley.edu

**Publication:** arXiv:2403.02338v2 \[cs.RO\] 14 Oct 2024 / 8th Conference on Robot Learning (CoRL 2024\)

**Figure 1:** We train two anthropomorphic robot hands to twist (off) lids of various articulated objects. The control policy is first trained in simulation with deep reinforcement learning, then zero-shot transferred to a real-world setup. We show that a single policy trained to manipulate simplistic, simulated bottle-like objects can generalize to real-world objects that have drastically different physical properties (e.g. shape, size, color, material, mass). The length, diameter (or diagonal length), and mass of each object are annotated at the bottom of individual subfigures. More results can be found in our video and project website.

## **Abstract**

Manipulating objects with two multi-fingered hands has been a long-standing challenge in robotics, due to the contact-rich nature of many manipulation tasks and the complexity inherent in coordinating a high-dimensional bimanual system. In this work, we share novel insights into physical modeling, real-time perception, and reward design that enable policies trained in simulation using deep reinforcement learning (RL) to be effectively and efficiently transferred to the real world. Specifically, we consider the problem of twisting lids of various bottle-like objects with two hands, demonstrating policies with generalization capabilities across a diverse set of unseen objects as well as dynamic and dexterous behaviors. To the best of our knowledge, this is the first sim-to-real RL system that enables such capabilities on bimanual multi-fingered hands.

**Keywords:** Bimanual Manipulation, Sim-to-Real, Reinforcement Learning

## **1\. Introduction**

Achieving dexterous bimanual manipulation with two anthropomorphic robot hands has been exceptionally challenging, due to the inherently contact-rich nature of many manipulation tasks and the complexity of coordinating a high-dimensional bimanual system. This work takes a step towards this grand goal, by demonstrating the feasibility of learning a highly dexterous and dynamic bimanual manipulation policy purely in simulation and zero-shot transferring it to the real world. Specifically, we study the task of twisting or removing lids with two multi-fingered robot hands. This task is both practically important and profoundly interesting: for one, the ability to twist or remove lids from containers is a crucial motor skill that toddlers acquire during their early developmental stages \[1, 2\]; for another, the manipulation skills required for this task, such as the coordination of fingers to manipulate a multi-part object, can be generally useful across a large collection of practical tasks.

Since collecting human expert demonstration data to solve contact-rich tasks via imitation learning is highly challenging and expensive \[3\], we aim at training a generalizable policy through sim-to-real reinforcement learning without using any expert data (Figure 1). Our method does not require precise modeling of any individual object, or hardcoding prior knowledge on object properties; instead, stable and natural bimanual finger behaviors emerge through large-scale reinforcement learning (RL) training. Below, we share the novel insights that enable us to develop such a system.

**Physical Modeling.** Our work features a novel class of objects for in-hand manipulation: articulated objects defined as two rigid bodies connected via a revolute joint with a threaded structure. Accurately modeling friction and contact with revolute joints and threaded structures has traditionally been a hard challenge in robotic simulation \[4\]. To address this, we introduce a brake-based design to model the interaction between the lid and body of bottle-like objects. This design is fast to simulate while maintaining high fidelity to real-world physical dynamics, enabling efficient policy learning and successful sim-to-real transfer.

**Perception.** We initially hypothesize that a fine-grained, contact-rich manipulation task like lid-twisting must require precise perceptual information on object states and shapes. To our surprise, a two-point sparse object representation, extracted from off-the-shelf object segmentation and tracking tools, is sufficient to solve the perception problem. With simple domain randomization techniques, we train policies that are robust against occlusion and camera noise. This discovery suggests that a minimal amount of perception information can be adequate for complicated bimanual manipulation tasks.

**Reward Design.** Previously performant reward designs for tasks like in-hand reorientation \[5, 6, 7\] cannot be straightforwardly applied to our task, since those tasks focus on manipulating single-part rigid bodies with one hand rather than multi-part articulated bodies with two hands. Solving this task is more challenging since it involves more complex and precise contact (e.g. using two hands to hold a lid). In addressing this challenge, we discover a simple keypoint-based contact reward that yields natural lid-twisting behavior on the robot fingers.

We conduct several controlled experiments in both simulation and the real world. Through empirical analysis, we verify that our simulation modeling, perception module, and reward design can reliably lead to the desired behavior of lid-twisting. Our final successful policy manifests natural behavior across test objects with various physical properties such as shapes, sizes, and masses in simulation. Moreover, the learned policy can be zero-shot transferred to a wide range of novel household objects whose lids can be removed (Figure 1), and it is robust against perturbations.

## **2\. Background**

For decades, bimanual manipulation has remained an unsolved challenge in robotics \[8, 9, 10, 11, 12, 13\]. While multi-fingered robot hands seem to be a natural choice for bimanual robot systems in theory, designing controllers for high-dimensional action spaces remains an open problem. Classical approach has made significant progress but usually assume known object or physics information \[14, 15\] and the generalizability remains unknown. In recent years, bimanual manipulation has been actively studied with learning-based methods, as a result of progress in learning algorithms and compute infrastructure. These learning-based approaches can be categorized into two types: 1\) learning from real-world data; 2\) learning in simulation, then transferring to the real world (sim-to-real).

**Learning from Real-World Data.** Rapid progress has been made in RL in the real world. Zhang et al. \[16\] learn to chain motor primitives for vegetable cutting, with relatively simple motion primitives; much of the task difficulty is bypassed via the use of specialized end-effectors \[17, 18\]. Chiu et al. \[19\] learn precise needle manipulation with two grippers by integrating RL with a sampling-based planner. While impressive, these works cannot easily scale to higher dimensional action space due to their sample inefficiency or the need to define heuristic-based action primitives.

Most recent successes in bimanual manipulation are achieved by learning from demonstrations \[20, 21, 22, 23, 24, 25\]. However, successes so far are largely limited to simple end-effectors like parallel jaw grippers due to the lack of high-quality demonstration data from multi-fingered robot hands \[26\]. Although several works aiming to improve demonstration data collection with two arms \[27, 28, 29\] or multi-fingered hands \[21, 30, 31, 32, 33\], their latency and retargeting errors limit their practical applicability and scalability. Lin et al. \[20\] proposes a scalable hands-arms teleoperation system that learns smooth bimanual policies, but the system compromises on dexterity. Similar systems that offer more dexterous control \[34, 35\], on the other hand, suffer from drawbacks ranging from jittery control to high costs. Our method uses RL in simulation and is thus not limited by the hardware and data collection infrastructure problems faced by learning from demonstration approaches.

**Sim-to-Real.** There has been growing interest in sim-to-real approaches for robotics \- i.e. learning policies in simulation and transferring them to the real world \- stimulated by several notable successes in recent years ranging from locomotion \[36, 37, 38\] to manipulation \[5, 7, 6, 39\]. Existing works in manipulation, however, are mostly done with either a single multi-fingered hand \[3, 40, 41, 42, 43, 44, 45, 46, 47\], or two arms with simpler end-effectors \[48, 49, 50\]. While Chen et al. \[51\] and Zakka et al. \[52\] feature bimanual tasks with dexterous hands, only simulation results are shown. The work most related to ours is Huang et al. \[53\], where the authors demonstrate throwing and catching objects using two dexterous hands. However, our task is significantly more contact-rich and requires substantially more challenging bimanual coordination to maintain object stability at all times. To our best knowledge, there is no learning-based method directly comparable to ours on the proposed task.

## **3\. Learning to Twist Lids with Two Hands**

**Figure 2:** Our bottle model and the used bottles in the simulation and the real world. **A:** Simulated bottle URDF. **B:** Training bottle objects in simulation. **C:** Custom-made bottles (in-distribution except for the rightmost square bottle). **D:** Household object bottles (out-of-distribution).

We focus on the challenging task of lid-twisting for container objects, since it is a complex in-hand manipulation process that requires dynamic dexterity of multiple fingers and precise coordination between two hands. The goal of this task is to twist the lid about the object's axis of rotation in one direction as much as possible; during this process, the object should always stay in hand. Achieving this involves a sequence of delicate movements: 1\) after initialization, the robot hand should firmly grasp and slightly rotate the bottle to a suitable pose; 2\) the hand that is closer to the object lid should place its finger around the lid to initiate twisting motion; 3\) the two hands should coordinate to avoid dropping the object while one hand twists the lid. Motor skills that arise from this task could serve as generic abstractions for skills necessary to manipulate many other household objects, especially those with revolute joints such as Rubik's cubes, light bulbs, and jars.

### **3.1 Object Simulation**

A central challenge in simulating the lid-twisting task is how to model friction between the bottle body and the lid properly, particularly static friction. Simulating this type of physical force has been a long-standing problem in robotics and graphics \[4\]. We design a simple modeling approximation that strikes a balance between fidelity and speed during physical simulation; our bottle-like object model is illustrated in Figure 2(A). Our design features a special Brake Link that constantly presses against the bottle lid via a prismatic joint. This artificially generates frictional forces between the bottle body (Base Link) and the lid (Lid Link), preventing relative rotation between them \- similar to a bottle with its lid screwed on. We replicate these bottles in the real world, as shown in Figure 2(B).

We note that naively tuning static friction properties between two revolute-joint-connected bodies is not realistic enough with our simulator. Such a brake-based is the only way we find that can simulate the static friction well.

### **3.2 Task Initialization**

To better benchmark the bimanual twisting capability, we consider a class of articulated bottles with lids that can be twisted infinitely (see Figure 2(A) and Section 3.1 for more details). Each object of interest consists of two rigid, near-cylindrical parts (a "body" and a "lid"); the two parts are connected via a continuous revolute joint, allowing them to rotate about each other. At the beginning of each episode, the two robotic hands are initialized in a static pose with upward-facing palms, and a bottle-like object is gently dropped or placed onto the fingers. The initial pose of the object is randomized both in translation and rotation to a fixed default pose; the initial joint positions of the hands are randomized about a canonical pose by adding Gaussian noise. Note that since we do not assume a stable grasp configuration at task initialization, the control policy needs to learn in-grasp reorientation to place the object in a stable location to perform successive manipulation.

### **3.3 Policy Learning**

Bimanual in-hand dexterous manipulation involves highly complex hand-object contacts, and remains challenging to solve with traditional methods. In this work, we address the control challenge through RL. We formulate our control problem as a partially observable Markov Decision Process.

**Observation Space.** At each time step ![][image1], the control policy observes the following information from the environment: the proprioceptive hand joint positions ![][image2], the estimated center-of-mass 3D positions of the bottle base and lid, and previously commanded target joint positions ![][image3].

**Action Space.** We use a PD controller to drive the robot hand. The control policy produces a relative target joint position as the action ![][image4], which is added to the current target joint position ![][image3] to produce the next target: ![][image5]. ![][image6] is a scaling factor. Note that we smooth the action with its exponential moving average (EMA) to produce smooth motion. The next target position is sent to the PD controller to generate torque on each joint.

**Figure 3:** **Left:** Real-time perception system. Top: overview. Bottom: we segment and track object parts from the RGB frames (left), take mask centers as object part centers (middle), and estimate 3D object keypoints using noisy depth information from the camera (right). **Right:** Illustration of reward design. Our task-specific reward contains finger contact reward (yellow arrows), twisting reward (white arrow), and pose reward (blue arrow). In particular, our keypoint-based finger contact reward is crucial for learning the desired behavior.

**Reward.** While one way to approach hard exploration problems is to add intrinsic rewards \[54, 55\], we introduce the following fine-grained reward terms to shape the hand behavior (Figure 3 right).

(1). **Twisting Reward.** We define the twisting reward as ![][image7] which is the rotation angle of the lid during one-step execution. This reward term encourages the hand to twist the lid.

(2). **Finger Contact Reward.** We find it crucial to use a set of reference contact points to guide effective contact between fingertips and the bottle. We define two set of points ![][image8] and ![][image9] attached on the bottle base and lid respectively. Then, we define the finger contact reward as

![][image10]where ![][image11] and ![][image12] are the position of left and right fingertips, ![][image13] is a scaling hyperparameter, and ![][image14] is a distance function defined as ![][image15]. Therefore, we require each fingertip to stay as close to one of the reference contact points as possible. As we will see later, this term is necessary for eliciting desired behavior and task success.

(3). **Pose Reward.** We also introduce a pose matching reward term to encourage the bottle main axis ![][image16] aligned with a predefined direction ![][image17]. This term is defined as ![][image18].

(4). **Regularizations.** Besides the three task-specific rewards, we also introduce another few regularization terms as in previous works \[44\], including work penalty and action penalty to penalize large, jerky motions. We leave the details of the definition to the appendix.

**Reset Strategy.** There exist many possible hand-object interaction modes. Among these, most modes lead to failures such as getting the object stuck between fingers, and exploring those modes rarely provides good learning signals. To circumvent the high dimensionality of our exploration problem, we introduce two early termination criteria. First, we reset an episode if the robot hands fail to rotate the bottle into a desired pose for bimanual twisting within a short time limit. Additionally, we reset when the bottle's z-position is below a certain threshold, as the fingertips of the two hands can pinch the bottle at a low position without being able to reposition it into the palm.

**Training.** We use PPO \[56\] with asymmetric critic observation \[57\] to train our policy, and introduce various domain randomizations to make the policy transferable to the real world. We apply both physical and non-physical randomizations. The detailed training setup can be found in the appendix.

### **3.4 Real-World Perception.**

Figure 3 shows an overview of our real-world perception pipeline. To make our RL policy more transferable, we use bottle and lid center points instead of pixels as vision input. We extract these keypoints from real-time images through object segmentation and tracking in the real world. Specifically, we utilize the Segment Anything model \[58\] to generate two separate masks for the bottle body and the lid on the first frame of each trajectory sequence, and XMem \[59\] to track the masks throughout all remaining frames. To approximate the 3D center-of-mass coordinates of the bottle body and lid, we calculate the center position of their masks in the image plane, then obtain noisy depth readings from a depth camera to recover a corresponding 3D position. The perception pipeline runs at 10 Hz to match the neural network policy's control frequency.

## **4\. Simulation Experiments**

To test how to enable the emergence of natural and robust manipulation behaviors, we first conduct several experiments in simulation. Specifically, we study the following questions: How important is the keypoint-based reward for eliciting desired twisting behavior in bimanual manipulation? How important is visual information for solving this task? Is a sparse keypoint representation enough for learning a generalizable policy?

### **4.1 Setup**

**Object Set.** In simulated experiments, we use a collection of simulated cylindrical bottles with varying aspect ratios for both training and evaluation. Some samples are visualized in Figure 2(B). We consider two setups in simulation: 1\) multi-object, in which all the objects are used, and 2\) single-object, in which a single medium-sized bottle that represents the mean of the dataset is used.

**Evaluation Metric.** We introduce the following metrics for evaluating the performance: 1\) Angular Displacement (AD) is the total number of degrees through which the lid has been twisted; 2\) Time-to-Fail (TTF) is the period measured from the moment the bottle is held to the point when it either slips from the hand or becomes lodged; 3\) Velocity (Vel) is AD divided by TTF, reflecting the speed of twisting motion.

**Baselines.** We compare our policy with the following baselines. 1\) *Policy without Vision.* This is a neural network policy without object state information. We use this to evaluate the importance of vision. 2\) *Policy with Reduced Contact Reward.* In training this policy, we reduce the intensity of our proposed finger contact reward. We use this to study the role of our contact reward in policy learning and shaping the policy's behavior. 3\) *Policy with Gait Constraint Reward.* In training this policy, we replace our contact reward with a gait constraint reward function similar to ones used for in-hand reorientation tasks \[39\]. This baseline is only used for qualitative analysis.

### **4.2 Results**

**Figure 4:** Training curves in different settings. **Top:** Single-object training results (evaluated on single-object setup). **Bottom:** Multi-object training results (evaluated on multi-object setup). **Left half:** Comparisons of different reward setups. **Right half:** Ablations on the use of vision. The results are averaged on 5 seeds. The shaded area shows the standard deviation. The AD score is averaged by the total execution steps.

**Reward Design.** We first compare our approach with the reduced finger reward baseline (Figure 4). After decreasing the scale of finger contact reward, learned policies fail to master the desired lid-twisting skill and have low performance in general. We hypothesize that this is because the motion of lid-twisting requires a very specific pose pattern for holding the object; without explicitly encouraging such a pose pattern (e.g., via its contact modes), RL exploration becomes so hard that it is unsolvable within the available training time. We also observe a positive correlation between the intensity of finger contact reward and both 1\) sample efficiency during learning and 2\) performance of learned policies (as reflected by Figure 4 and qualitative observations in Figure 5 (left)).

**Vision vs. No Vision.** We also study the importance of vision modality. Existing works show that certain rotation behaviors can be achieved through implicit tactile sensing (via proprioception) \[39\]. However, our empirical results show that, in both single and multi-object setups, the no-vision baseline performs substantially worse than our full method. This suggests that knowledge of the position of bottle keypoints is essential for successful lid-twisting.

**Single Object vs Multi Object.** We run RL training with two object settings: 1\) using a single bottle-like object; 2\) using multiple bottle-like objects with more variation in the ratio between the bottle base and lid. For results shown in Figure 4, all multi-object training runs are evaluated on multi-object setup and all single-object training runs are evaluated on single-object setup. The two settings pose a trade-off between specialization and generalization: in the single-object scenario, the policy might learn successful behaviors more easily but find it harder to generalize to unseen objects, and vice versa. To our surprise, we observe that multi-object training yields slightly better performance compared to single-object training. We hypothesize that multi-object makes exploring lid-twisting behavior an easier process by introducing an object curriculum that covers both easy and hard object instances during training.

## **5\. Real-world Experiments**

**Figure 5:** **Left:** Behavior of different reward functions. Top: Our full reward function achieves a stable grasp, as well as a smooth, natural, and human-like twisting motion. Middle: A naive gait constraint reward without any contact hints leads to erratic finger motion and unnatural grasps. Bottom: A reduced contact reward yields somewhat natural behavior, but the grasp is loose compared to the full contact reward case. **Right:** Perturbing a learned policy with random external force. Our policy is resilient to these external forces and able to recover.

### **5.1 Experiment Setup**

**Hardware Setup.** We use two 16-DoF Allegro Hands from Wonik Robotics for our experiments. Each Allegro Hand is mounted on a fixed UR5e arm. We employ a single RealSense D435 depth camera to provide visual information, from which we extract object state information. We send control commands to the robot at a frequency of 10 Hz via a Linux workstation.

**Object Set.** For quantitative evaluation, we evaluate the sim-to-real transfer capabilities of our policies on five different articulated bottle objects (Figure 2(C)). Among them, four are in-distribution (round-body bottles) and one is outside of the training distribution (square-body bottle).

**Evaluation Metric.** We measure both AD and TTF in 20 trials, with each trial lasting for a maximum of 30 seconds. For each evaluated method, we select the three best policies out of ten policies trained on ten different random seeds. We end a trial if the bottle falls off the palm.

**Baselines.** We compare our final policy with the following baselines to study the effect of several key design choices. 1\) *Open-loop Replay Policy (Replay).* We record successful trials of our learned policy in the simulation and randomly select a trajectory to replay on the real robot. This baseline is used to evaluate whether the task can be solved by a deterministic motion pattern. 2\) *Policy without Vision (No-Vis).* This baseline policy only takes proprioceptive state information as input, without information about the object state. 3\) *Policy without Asymmetric Training (No-Asym).* We compare with a baseline where policy is trained without asymmetric PPO, and evaluate whether introducing additional privileged information into the value network will affect the transfer performance. 4\) *Larger Policy Network Size (Large).* We increase the size of our actor-network and train a large-size policy. We use this to evaluate whether over-parameterization harms policy performance.

### **5.2 Twisting Lids in the Real World**

We show quantitative results comparing our policy with baseline policies in Table 1\. For both metrics, our policy outperforms all baselines across all evaluated objects. Our method can perform stable grasp on all the objects, and can rotate 3 out of 5 objects at a reasonable speed. In particular, for the blue bottle, one of the deployed policies can achieve 4 full turns (360 degrees) in 30 seconds on average. In contrast, almost all the baselines fail to achieve any effective rotations, either getting stuck or dropping the bottle to the ground. We find that the open-loop policy has the lowest TTF score. Replaying a successful trajectory will not lead to a stable grasp for most of the time, and the bottle will directly roll on the fingers and then drop off the palm. This suggests that the considered task involves very fine-grained contacts and requires the policy to act very precisely according to the object state. Another interesting observation is that the large policy does not transfer to the real world, although we confirm that it can achieve similar performance to our full policy in simulation. This suggests that some overfitting occurs, and controlling the size of the policy network is very important for the successful sim-to-real transfer of our considered contact-rich task.

**Table 1: Comparison with baselines on real setup.** For each method, we deploy 3 policies trained on 3 different seeds and average the results. Each deployment trial is conducted for 30 seconds.

| Method | Blue Bottle (AD/TTF/Vel) | Wood Bottle (AD/TTF/Vel) | Red Bottle (AD/TTF/Vel) | Gold Bottle (AD/TTF/Vel) | Square Bottle (AD/TTF/Vel) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Replay** | 128.33 ±217.96 / 7.67 ±4.03 / 11.68 ±19.80 | 2.67 ±4.62 / 7.67 ±5.86 / 0.22 ±0.38 | 15.00 ±25.19 / 4.67 ±4.62 / 1.59 ±2.60 | 7.67 ±4.04 / 28.33 ±2.99 / 0.27 ±0.14 | 43.04 ±29.67 / 10.00 ±8.62 / 2.97 ±0.00 |
| **No-Vis** | 1.33 ±2.31 / 21.67 ±14.43 / 0.04 ±0.08 | 1.07 ±1.85 / 14.67 ±13.61 / 0.27 ±0.46 | 8.33 ±6.11 / 1.90 ±3.29 / 0.27 ±0.47 | 16.33 ±1.15 / 0.67 ±13.05 / 0.04 ±0.08 | 0.18 ±0.20 / 20.33 ±11.24 / 5.00 ±16.24 |
| **No-Asym** | 0.62 ±0.96 / 18.67 ±14.43 / 30.00 ±0.00 | 28.94 ±15.14 / 19.33 ±15.13 / 0.67 ±0.28 | 1.15 ±0.48 / 0.03 ±4.3 / 0.04 ±3.67 | 13.00 ±7.51 / 8.33 ±3.79 / 0.54 ±0.94 | 0.00 ±0.00 / 2.33 ±1.53 / 0.00 ±0.00 |
| **Large** | 2.00 ±0.14 / 22.33 ±2.00 / 0.14 ±13.28 | 0.00 ±0.00 / 24.00 ±10.39 / 0.00 ±0.00 | 9.33 ±2.52 / 4.73 ±0.68 / 2.33 ±0.47 | 2.67 ±0.09 / 3.06 ±0.10 / 22.33 ±13.28 | 0.06 ±2.89 / 30.00 ±0.00 / 1.67 ±0.10 |
| **Ours** | **946.33** ±383.81 / 41.26 ±4.36 / **23.67 ±10.97** | **499.50 ±578.23 / 30.0 ±0.0 / 16.65 ±19.27** | **150.67 ±113.47 / 30.00 ±0.00 / 5.02 ±3.78** | **98.67 ±66.91 / 30.00 ±0.00 / 3.29 ±2.23** | **43.00 ±12.12 / 30.00 ±0.00 / 1.43 ±0.40** |

### **5.3 Robustness against Perturbation**

Finally, we also evaluate our policy's robustness against force perturbation. Specifically, we perturb the object during deployment at random times by poking or pushing it along random directions using a picker tool (see the right of Figure 5). We find that our policy can reorient and move the object back to a stable pose for continuous manipulation, indicating that it has some robustness against external forces and can adapt to these unexpected changes. Note that we use a marker-based object detection system in this experiment to disentangle the visual occlusion effect.

### **5.4 Exploration of Twisting Lids Off**

In the above section, we mainly study whether the twisting behavior can naturally emerge and be transferred to the real world. Next, we explore the limit of our approach by testing it on 10 novel household objects (Figure 2(D)). These objects differ substantially from our training objects in terms of shape, size, mass, material, color, and mechanical design. While the lids of the synthetic bottles that we use for both simulation training and real-world testing can be twisted infinitely, the lids of these household objects cannot. To evaluate our policy's ability to generalize the lid-twisting skill to these novel objects, we use the success rate on a novel yet adjacent lid-removal task as the criterion. We define lid-removal as the object's lid being completely detached from the object body (e.g., when the lids fall from the robot's hands in Figure 1\.

We find that our policy continues to achieve stable and natural twisting behaviors on these novel objects. Furthermore, while we only train the policy for lid-twisting, we also find our policy capable of removing lids. For HairMask and FiberGummies, our policy showcases lid-removal rates of more than 50%. For particularly challenging objects that require many turns to remove the lid, such as Peanut Butter and EmptyNutella, our policy only achieves 10% lid-removal rates; however, its twisting behavior is steady and robust to perturbation (see our video supplementary for visualization).

## **6\. Conclusion**

We present an RL-based sim-to-real system for twisting or removing lids of bottle-like objects with two hands. We propose several techniques to handle the challenges that arise: a novel reward design, a sparse object representation for real-time perception, and an efficient yet high-fidelity method to simulate twisting bottle caps. We conduct experiments in both simulation and real world to demonstrate the effectiveness of our approach. Our real-world results show generalization across a wide range of seen and unseen objects.

**Acknowledgments**

We thank Chen Wang and Yuzhe Qin for helpful discussions on hardware setup and simulation of the Allegro Hand. TL is supported by fellowships from the National Science Foundation and UC Berkeley. ZY is supported by funding from InnoHK Centre for Logistics Robotics and ONR MURI N00014-22-1-2773. HQ is supported by the DARPA Machine Common Sense and ONR MURI N00014-21-1-2801.

## **References**

\[1\] A. A. O. Pediatrics. Caring for Your Baby and Young Child: Birth to Age 5\. American Academy Of Pediatrics, 2019\.

\[2\] R. Watling. Peabody Developmental Motor Scales. Springer, 2013\.

\[3\] T. Chen, J. Xu, and P. Agrawal. A system for general in-hand object re-orientation. In CoRL, 2021\.

\[4\] Y. Narang et al. Factory: Fast contact for robotic assembly. In RSS, 2022\.

\[5\] T. Chen et al. Visual dexterity: In-hand reorientation of novel and complex object shapes. Science Robotics, 2023\.

\[6\] OpenAI et al. Solving rubik’s cube with a robot hand. arXiv: 1910.07113, 2019\.

\[7\] A. Handa et al. Dextreme: Transfer of agile in-hand manipulation from simulation to reality. In ICRA, 2023\.

\[8\] F. Krebs and T. Asfour. A bimanual manipulation taxonomy. RA-L, 2022\.

\[9\] C. Smith et al. Dual arm manipulation \- a survey. RA-L, 2012\.

\[10\] N. Vahrenkamp et al. Bimanual grasp planning. In Humanoids, 2011\.

\[11\] K. Chatzilygeroudis et al. Benchmark for bimanual robotic manipulation of semi-deformable objects. RA-L, 2020\.

\[12\] N. Sommer, M. Li, and A. Billard. Bimanual compliant tactile exploration for grasping unknown objects. In ICRA, 2014\.

\[13\] R. Platt, A. H. Fagg, and R. A. Grupen. Manipulation Gaits: Sequences of Grasp Control Tasks. In ICRA, 2004\.

\[14\] C. Ott et al. A humanoid two-arm system for dexterous manipulation. In Humanoids, 2006\.

\[15\] T. Wimbock, C. Ott, and G. Hirzinger. Impedance behaviors for two-handed manipulation: Design and experiments. In ICRA, 2007\.

\[16\] K. Zhang et al. Leveraging multimodal haptic sensory data for robust cutting. In Humanoids, 2019\.

\[17\] A. Amice, P. Werner, and R. Tedrake. Certifying bimanual rrt motion plans in a second. arXiv:2310.16603, 2023\.

\[18\] T. Cohn et al. Constrained bimanual planning with analytic inverse kinematics. arXiv:2309.08770, 2023\.

\[19\] Z.-Y. Chiu et al. Bimanual regrasping for suture needles using reinforcement learning for rapid motion planning. In ICRA, 2021\.

\[20\] T. Lin et al. Learning visuotactile skills with two multifingered hands. arXiv:2404.16823, 2024\.

\[21\] C. Wang et al. Dexcap: Scalable and portable mocap data collection system for dexterous manipulation. arXiv preprint arXiv:2403.07788, 2024\.

\[22\] C. Chi et al. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024\.

\[23\] L. X. Shi et al. Waypoint-based imitation learning for robotic manipulation. In CoRL, 2023\.

\[24\] S. Stepputtis et al. A system for imitation learning of contact-rich bimanual manipulation policies. In IROS, 2022\.

\[25\] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In RSS, 2023\.

\[26\] F. Krebs et al. The kit bimanual manipulation dataset. In Humanoids. 2021\.

\[27\] Z. Li et al. Asymmetric bimanual control of dual-arm exoskeletons for human-cooperative manipulations. Transactions on Robotics, 2017\.

\[28\] M. Laghi et al. Shared-autonomy control for intuitive bimanual tele-manipulation. In Humanoids, 2018\.

\[29\] H. Fang et al. Low-cost exoskeletons for learning whole-arm manipulation in the wild. arXiv: 2309.14975, 2023\.

\[30\] A. Handa et al. Dexpilot: Vision-based teleoperation of dexterous robotic hand-arm system. In ICRA, 2020\.

\[31\] S. P. Arunachalam et al. Holo-dex: Teaching dexterity with immersive mixed reality. In ICRA, 2023\.

\[32\] S. P. Arunachalam et al. Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation. In ICRA, 2023\.

\[33\] Y. Qin et al. Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system. In RSS, 2023\.

\[34\] X. Cheng et al. Open-television: Teleoperation with immersive active visual feedback. arXiv preprint arXiv:2407.01512, 2024\.

\[35\] Jun 2024\. URL https://www.shadowrobot.com/teleoperation/.

\[36\] J. Hwangbo et al. Learning agile and dynamic motor skills for legged robots. Science Robotics, 2019\.

\[37\] T. Miki et al. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 2022\.

\[38\] A. Kumar, Z. Fu, D. Pathak, and J. Malik. Rma: Rapid motor adaptation for legged robots. In RSS, 2021\.

\[39\] H. Qi et al. In-hand object rotation via rapid motor adaptation. In CoRL, 2022\.

\[40\] G. Khandate et al. Sampling-based exploration for reinforcement learning of dexterous manipulation. In RSS, 2023\.

\[41\] J. Pitz et al. Dextrous tactile in-hand manipulation using a modular reinforcement learning architecture. In ICRA, 2023\.

\[42\] H. Qi et al. General in-hand object rotation with vision and touch. In CoRL, 2023\.

\[43\] L. Rostel et al. Estimator-coupled reinforcement learning for robust purely tactile in-hand manipulation. In Humanoids, 2023\.

\[44\] Z.-H. Yin et al. Rotating without seeing: Towards in-hand dexterity through touch. In RSS, 2023\.

\[45\] Y. Yuan et al. Robot synesthesia: In-hand manipulation with visuotactile sensing. In ICRA, 2024\.

\[46\] Y. Chen et al. Sequential dexterity: Chaining dexterous policies for long-horizon manipulation. In CoRL, 2023\.

\[47\] S. Suresh et al. Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation. arXiv:2312.13469, 2023\.

\[48\] Y. Lin et al. Bi-touch: Bimanual tactile manipulation with sim-to-real deep reinforcement learning. RA-L, 2023\.

\[49\] Y. Li et al. Efficient bimanual handover and rearrangement via symmetry-aware actor-critic learning. In ICRA, 2023\.

\[50\] S. Kataoka et al. Bi-manual manipulation and attachment via sim-to-real reinforcement learning. arXiv:2203.08277, 2022\.

\[51\] Y. Chen et al. Towards human-level bimanual dexterous manipulation with reinforcement learning. In NeurIPS, 2022\.

\[52\] K. Zakka et al. Robopianist: Dexterous piano playing with deep reinforcement learning. In CORL, 2023\.

\[53\] B. Huang et al. Dynamic handover: Throw and catch with bimanual hands. In CoRL, 2023\.

\[54\] Y. Burda et al. Exploration by random network distillation. In ICLR, 2019\.

\[55\] T. Lin and A. Jabri. Mimex: Intrinsic rewards from masked input modeling. In NeurIPS, 2023\.

\[56\] J. Schulman et al. Proximal policy optimization algorithms. arXiv:1707.06347, 2017\.

\[57\] L. Pinto et al. Asymmetric actor critic for image-based robot learning. In RSS, 2018\.

\[58\] A. Kirillov et al. Segment anything. In ICCV, 2023\.

\[59\] H. K. Cheng and A. G. Schwing. Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022\.

\[60\] V. Makoviychuk et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv:2108.10470, 2021\.

\[61\] J. Schulman et al. High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438, 2015\.

\[62\] D.-A. Clevert et al. Fast and accurate deep network learning by exponential linear units. arXiv: 1511.07289, 2015\.

### **7\. Object Details**

**Simulated Bottles.** We use Isaac Gym \[60\] to model the simulated learning environments. For the multi-object environment, we use bottles whose bodies range from 82cm to 86cm in diameter and 55cm to 67cm in height, and whose caps range from 62cm to 70cm in diameter and 20cm to 33cm in height. For the single-object environment, we use a bottle whose body is 84cm in diameter and 60cm in height, and whose cap is 67cm in diameter and 26cm in height.

**Real-World Bottles.** We show details of our real-world bottle design in Figure 6\.

**Figure 6:** Real-world bottle design. Each bottle is consisted of three parts: the cap (top left), the pin (top right), and the body (bottom left). The parts can be 3D printed and assembled by inserting the pin into the cap and fixing the cap onto the body. With the pin holding the cap and body in place, the cap can be infinitely twisted about the body. Examples of printed bottles are shown in Figure 2(C).

### **8\. Real-World Experiment Details**

**Camera Calibration.** We use a novel marker-based approach to calibrate the extrinsics matrix of our camera. Specifically, we add the marker tag used in our real-world setup into the simulation environment, such that pair coordinates of the marker corners can be obtained easily in both the camera frame and the world frame (Figure 7). We then use the paired coordinates to solve for camera extrinsics. Doing so greatly reduces the manual labor required by other camera calibration approaches, such as capturing checkerboard images and solving for multiple extrinsic matrices.

**Figure 7:** Modeling real-world marker tag in simulation for easy camera calibration.

**Hardware Communication.** To keep our control loop running reliably at 10Hz, we use ZeroMQ to manage communication between robot hands, camera, and Linux workstation.

**Figure 8:** Real-world task initialization. At the beginning of each task sequence, we initialize the robot hands about a canonical position with upward-facing palms. Then, we lightly place an object onto the fingers.

**Task Initialization.** We illustrate details of how we initialize the task sequence in the real world in Figure 8\. The canonical joint positions of each finger is documented in Table 2\.

**Table 2: Initial joint positions of both robot hands.**

| Finger | Initial Joint Positions |
| :---- | :---- |
| Index | \[-0.0080, 0.9478, 0.6420, \-0.0330\] |
| Middle | \[0.0530, 0.7163, 0.9609, 0.0000\] |
| Ring | \[0.0000, 0.7811, 0.7868, 0.3454\] |
| Thumb | \[1.0670, 1.1670, 0.7500, 0.4500\] |

### **9\. Training Details**

**RL implementation.** We use the proximal policy optimization (PPO) algorithm to learn RL policies. We use an advantage clipping coefficient ![][image19]; a horizon length of 16, with ![][image20], and generalized advantage estimator (GAE) \[61\] coefficient ![][image21]. The policy network is a three-layer MLP with ELU \[62\] activation, whose hidden layer is \[256, 256, 128\]. The policy network outputs a Gaussian distribution with a learnable state-independent standard deviation. The value network is also an MLP with ELU activation, whose hidden layer is \[512, 512, 512\]. We use an adaptive learning rate with KL threshold of 0.016 \[36\]. During training, we normalize the state input, value, and advantage. The gradient norm is set to 1.0 and the minibatch size is set to 8192\. We use asymmetric observation \[57\] for the policy and value network, adding privileged information to the value network inputs. This privileged information is not accessible by the policy network.

**Asymmetric States.** In addition to the policy inputs, we provide the following privilege state inputs to the value network of asymmetric PPO: hand joint velocities, all fingertip positions, all contact keypoint positions, object orientation, object velocity, object angular velocity, random forces applied to object, object brake torque, object mass randomization scale, object friction randomization scale, and object shape randomization scale.

**Action Hyperparameters.** To generate action commands, we clip neural network policy output to \[-1,1\] range. We then apply an action scale of 0.1 and a moving average parameter of 0.75 to the actions.

**Reward Hyperparameters.** We use ![][image22], ![][image23], ![][image24], ![][image25], and ![][image26] as reward weights.

**Domain Randomization Setup.** We apply a wide range of domain randomizations to ensure zero-shot sim-to-real transfer, including both physical and non-physical randomizations. Physical randomizations include the randomization of object friction, mass, and scale. We also apply random forces to the object to simulate the physical effects that are not implemented by the simulator. Non-physical randomizations model the noise in observation (e.g. joint position measurement and detected object positions) and action. A summary of our randomization attributes and parameters is shown in Table 3\.

**Table 3: Domain Randomization Setup.**

| Parameter | Range/Value |
| :---- | :---- |
| Object: Mass (kg) | \[0.03, 0.1\] |
| Object: Friction | \[0.5, 1.5\] |
| Object: Shape | x U(0.95, 1.05) |
| Object: Initial Position (cm) | \+ U(-0.02, 0.02) |
| Object: Initial z-orientation | \+ U(-0.75, 0.75) |
| Hand: Friction | \[0.5, 1.5\] |
| PD Controller: P Gain | x U(0.8, 1.1) |
| PD Controller: D Gain | x U(0.7, 1.2) |
| Random Force: Scale | 2.0 |
| Random Force: Probability | 0.2 |
| Random Force: Decay Coeff. and Interval | 0.99 every 0.1s |
| Bottle Pos Observation: Noise | 0.02 \+ N(0, 0.4) |
| Joint Observation Noise | \+ N(0, 0.1) |
| Action Noise | \+ N(0, 0.1) |
| Frame Lag Probability | 0.1 |
| Action Lag Probability | 0.1 |

#### **9.1 Generalization Experiment Details**

In Table 4, we provide per-object quantitative results and further analysis on the lid-removal success rate mentioned in Section 5.4. These results show that the overall low lid-removal success rate mostly comes from "hard" objects, i.e. objects that require more turns and/or are more out-of-distribution in shape. We also note that, if we define "number of turns needed to remove lids" of in-distribution objects to be 1, the success rates are consistently 100% for the in-distribution objects.

**Table 4: Real-world objects differ greatly in the design of lids.** Each object requires a different number of turns to remove the lids, and the difficulty of manipulating different object shapes also varies. In this table, objects are sorted by number of turns needed to remove lids (high to low) and then lid-removal success rate (low to high). Note that lid-removal is a novel task rather than the training task.

| Object | No. of Turns | Lid-Removal % |
| :---- | :---- | :---- |
| Peanut Butter | 5 | 10 |
| EmptyNutella | 5 | 10 |
| Nutella | 5 | 40 |
| FiberGummies | 5 | 50 |
| Earplugs | 3 | 20 |
| OilCapsules | 3 | 40 |
| StressGummies | 1 | 40 |
| HairMask | 1 | 60 |
| **Overall** | \- | **33.75** |

#### **9.2 Generalization to a Vertical Task Setup**

To showcase the generalizability of our approach, we train policies with a novel vertical setup (i.e. the agent opens lids of bottles held vertically). Other than changing the initialization setup (see Figure 9 and Table 5\) and turning off the perception system, no change is made to the system components proposed in our work. We additionally note that the horizontal setup in our main text is more challenging than the vertical setup. The vertical setup prevents the most common failure case \- lack of stabilization and dropping objects by design.

**Figure 9:** A successful lid-twisting policy in a novel vertical setup. The leftmost image shows task initialization of the vertical setup. The remaining three images show action trajectory of a successful lid-twisting policy being deployed. Additional results can be found in the video supplementary materials.

**Table 5: Initial joint positions of robot hands for a vertical task setup.**

| Finger | Initial Joint Positions |
| :---- | :---- |
| Left: Index | \[-0.0080, 0.0772, 1.6655, 0.2697\] |
| Left: Middle | \[0.0530, 0.0031, 1.7090, 0.0000\] |
| Left: Ring | \[0.0000, \-0.0617, 1.5400, 0.3454\] |
| Left: Thumb | \[0.6670, 1.1670, 1.0000, 0.8800\] |
| Right: Index | \[-0.0080, 0.9478, 0.6420, \-0.0330\] |
| Right: Middle | \[0.0530, 0.7163, 0.9609, 0.0000\] |
| Right: Ring | \[0.0000, 0.7811, 0.7868, 0.3454\] |
| Right: Thumb | \[0.6670, 1.1670, 0.7500, 0.4500\] |

#### **9.3 Additional Details on Domain Randomization**

During the process of hyperparameter tuning, we note that the highest policy variances are introduced by the following parameters: Bottle Position Observation Noise, Joint Observation Noise, Action Noise. This suggests that noise parameters relevant to action space and observation space might be the most important for domain randomization for successful sim-to-real transfer.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAbCAYAAACwRpUzAAAApElEQVR4XmNgGMxAQUEhXV5efj+6OBgAJU/JyckdRBdnEBUV5QHq+g3EXehyIF0OQIn/QOwNFwRyeqGCKBioWAGuCGjXBqDgRbgAMgBKvAMqnokuDpLQhBqXhC4HkkwCSQKN1kKXA0nOAeIXML6srKwOkGKBSV4H4tUgtrS0tDCQvQ+mECR5H4j7QKqB9DJ5ZL8C7UoDCjwA4itAFyfCJUYBAwMAZ8kpjsSRb9cAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAABPUlEQVR4Xu2SP0sDQRDFD5EQtQoIB8f9w/K0CaSyEUSsLSwsLMRgbWcZAukEC7+CSCxsbUQRLQTBTvwAaRJSBgslNslvcHczFxW0vweP2/f2zezc3nlegQL/QRRFtTiOr5MkeYS3sArbcHU6mwNFB4TeeW4YvYsewWEQBPPTeQdO3JQgBQ3t4w3wHrT3DYSe4CfBivVYZ+bkY53NIU3TLRNqaR99CT/0yOgzmh65EMVrpnjfejIBugtvXNDzZtA9WHVOGIZzGH2anIqmcBn9Kg3tHbDXRD/DN2mI3nENpBu8x3wxIfk8I7huMzQ6RF+4ot9A6BwOaVZW3hWs69yPINSRSZQ1KyNz+pKsucRFtTeBeWcZ+cR6/Acr6K6smWZP9KQC+L6/kHzdphRadmQvy7IS6ztYp/l2rrDA3zEGTt1TcQECFuwAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABb0lEQVR4XuWSO0sDURCFgwR8FDaCyL6SzS4iiA+wsFUR0tnYWmlAwcrGwkI7tbDQv+ALUwYL0SgIYmUn2NkpBhEsBLES/UbuXiab/ADFA4c7c+6Z2ZndzWT+OaIo6s7lcifwPgiC9TAMh/L5/Dwsp70NoGhfiqSAuAo/4RN5X9rbgDiOO3XuOI7veV671n4h2HcEnrLnNTznhQ1yHrLzaNpbB0wl+AYnJKfJDPEX/KC4Le234LJojCtaN80utFaHQqHQK4U0ONM62qLRi0rOMtGkzXzfnxIT4oYyyTRlGZnP1ZFoshq+ijaNmZHnEs113S7yV1i1xszPezjCv2QF+Qkw1bjYNvkA+Z2ZZlU04mlpZB5yCbdsA7oNI1xx3nLewD1jHE95aoQttrAZMB3Ad37X1kRjimW0Xe1rCkwPMmZKO4azEjNFj76zMDvLvptaR3tG6zeftqTvBFnER7NrwpfkkqftULzGuSBeVffX8A1HpWHdK0nxQQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABIklEQVR4XmNgGAWjgNpAQUHBQl5efrecnNxBIL0OiKPR1eAEQM0ZQA0fFRUV7UB8oCEuQP5/dHVYAVCzB0gxUFM5TMzY2JiVaAOACk8D8Xd1dXVeJDErogwA2u4Ptb0eWRwotg2I7wHlFYBcRqgwI1DsGVDIAK4QyHGAGhCLJCYBchEQLwWyd4mLi3ODxIF8IyB+AjIIboCUlBQXUPAFEHdDFSkC8R4g/gnErUC8BCq+HojvAPF9eUhMKcENkZWVNQUKHgbiy0B8EWirO5BOAeIzQCwJUwfUtAEkDtdIImABav4OjGZ5dAmiANB2a6AB10FsZWVlMSDFhKYEPwAakAfEC0BsoBc7gRQLqgoCQEZGRhXogr1AQ+KAtCW6/CigAAAAvHZCPbqnt+EAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAA5ElEQVR4XmNgGAVUA/Ly8p5ycnIHFRQUTgGxBpC9ASi2DohPANntQCWMDFJSUlxAgdPi4uLcQPo6EN8AKQYZAKQjgPz/QMU+INOigQKFUA2/gPgOzCaguANIIZBOB3FAQACoyxkqWInkpBKQGBAbwsQYgArrodbYICncBsSv4IqggruB+BOQyQLia2lpsYH4QBsWwhWpqKiwAwV/AvFmJI1OUBvS4AqRHF2AJNYIVagFVwgUCALij0CfyyIpPAQUewpXhA1Aw/Qv0LQF6HIoAGiaP9TaWHQ5FABUuBKocC+6+LABAB1SPpatg0HbAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFwAAAAZCAYAAAC8ekmHAAAENklEQVR4Xu2YW2iURxTHN7Gt2tqqlTUm2d3Zza6G5EHUvFhbLIgUrUUFUduI9VJRwcuzd1GKIFqFGCiIhZLgpehDUYiIttKqbRQtFMSX4gVFQQQvD7aIiP5Ovplk9rjZrHGbSPj+8Oc7c86ZmTNnzzcz30YiIUKEKBDJZPJT+JMx5jn8BXlLVVXVYO3Xl5FIJGaz9ibYEIvFMtpedDDhUiZ7UFdX97a29XWw7hp4lRzU2vZ+7fMSpDJthTo+SKfTw61tl7LtyNG/GR7R+p5ARUXFu8x9Ev4DL1r5ITxnn/9KbLx543XfrkC1fkjfqUrdD90k14jH42mZk4TXSRv5WIdrHtDhNxMktEUW4dsI9nf0f8BZvt6iBP1t+q/Shp4EMdQTw3yRKysrY55+H5xmgqJa0dGjLe4W6QM3I280wY9U7vmU0m5m/TOkwXMA7UY4wfNpB3mLp1KpiVrfGeSXu2qCpLcFxnMSPC0TaWcH7OvhPcRSX88iJvvtfJA9nzG+lbngHXjFBAmSaj2uCyAXiHEuvotFjkajg5zevgH7RJa9Fr+1IqMrhzfkDJJ2WVnZe7T/oz3G9fWBrVkqXus1pDh5vKX1OYHzGhMk/C8JhOd1mNJ+PrD/Cg/7On5l4xZWCOh/hWRs6mqufLAH10sJz2QyH6BvElmqz70FAvQPIzY5xPsl7cuIJTqxjBHFtsGN74MtZTr6H2pra9+RNvJz+o/UfjlBQGV0eCadTFBpY7WPDwYeiM8TPxCC60/7oCkweeJHhSe0/lXBOLPgNyJXV1e/7/SMPYpkbrc+3yEP8fq0Fwo/xCG4GvtXruoFyCPw22nlmXCRswnosxvdBXczw/caj36+T17IxDbhjdrmg0kW4HPK+u6FO+Ax+t+E57V/Z6DPca3rDkyQ8KwKp50ywYG+0j6zziDWsMzJ2BqI+2e4zensdrQ14iWQ9kfyo7i2PVhlX5czYCescbYuIZu+6djH7/mV8n/BFDfhbRVuF34LnoGP4bxknnOoVyDbg1QmwS2B1yTptJdqv2KCJCTljTDB4dgZCzo0jVfhUiiM+6NskUyxRt5a7d/rINjDBLfcyusk4bBV+xUTJGQ0czRrfXdAUuf4Cbd7r4wtV7szHG6fqS69ilI5MFzDBvtUko7+Y9+xyJCr6N9a2R0Q81y4UGS3FYqO8VfYW8ZZk33H7j2Q1O+1juD22yo/oG3FRNLeIF4XuRIuMMGHj3x+12A/GlHfCz0GuZ+S6E8IpAXe13Z7v5SEP8FvqLYXC7I/k4gpWv+qIM56OE9kP17GHiJrtNfVz2FTj//nQ0Bf22S207eb4B6dZYeP/E/mYoKxT0qlE9cXyOM4wCsjBd5liWmY9DfBfymXrNwK9zgfxp3M+Bes7S78M1Lo12AfRYl9oxpsUtpJsk4UcksJESJEiBAhQoQIEeINxwuRyyrOQNqu+gAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAZCAYAAAAhd0APAAAEi0lEQVR4Xu2YbYiVRRTHd1d7f4Nq29qXZ+7e3djagtTtWy9QX0oTyxQ0zbJFyA9J34wio6KCghJ6IUMk6bZQKm6poCQWpVGhEX5IirSkSOmFWoW+FLH9/vc5c3d2urt7r9ybSz5/+DNnzjkzc2bmPDNzb0NDhgwZagzn3KwkST6jHIa74c5cLveR6WbG/qcKWIOFzP8N+DJyLrbXHAzUD4f6+vpO8zo2YQW6P0O/UwXM/Srmfqizs7NHdeR1sc+/QKMPLZM9f+np6TlPNuS1kW24TPv16DdGumfQfRPq6oWWlpZzGGunxoP7TP4aDsFj8DgskJDT4rbVoq2t7SL6uSVST6X/m3ylvb39cothuuqUm0dcxwGOH7t0kQfp5KzA1ITuC+t0FrvbEtiKQH+YwJb7OoG2o3sf32tCv3qDMRfBxSYXuru7mxEbkd/UwlHuprwvaNJI/QnNjaRZQvko9fc6OjquRX4EPg13lVmPgq+0traeTds1sC/wKUFHEbbrYn1Z6Cih8x9cuhH90lHOhNsRp0TuJTDASnwO+7ombn2UMqNS6Ouxr/Io/FIL4NLN34q5KfaPwXwXBLFvpH6mZIvpRdMvgw9IZrFbbd4/Me71LPaFih19l+w6Tqj/5U+FEDbvQj6fvyC2xdA8GiqIvwg6fkxBwA8sc/RJXxb7haDNDvi6r2vnbSJzQr+JQJubafowE3exrVKMtQldXV2XEOOrpr8NzvVtiPNq6r8hTsX/VuSvvI02D8F3tNA68rxewG8ArtIXFOoF+rkd2zp/R2o9Kp6XHSPDxm/hlbFPCMsinbfFI0CQrPb6BPlULw79xwK+edrsjfXVgn7mu5FN2OQ3AXmGEszkl3SEBG1W4Pe22Z6jvtrbqO/RgsLHFaPXU7/Ub4rGhHd7m+lW0/bT4F49QNEY+owLBtiiRaR8NraFUAa49KgYhpu16NIrQOR30a1VGbcrB3y3KyNjfbWgn3nBJhS/BHgF3ACXoxvIRZcqugJcZrLmU3pWIw8yh7fgg16nDUT/lK8LZPmN6O70dTvWXoGr4PP+lVQRaNDJgN+7dGF/9JlUbzDWtlh3IlBWEvNSyfS5SQuQpHfMH3Beb2/v6VGTyYXm5uZzCfRzJnEv5RFtBPJdsV+toUvQNl5ZOBYrupjjTVAS6UmJ/GQS3FuTFgQ6mLPnm0ufZjrXd8R+tQZjTmOc9bH+REBfC5REkv0mWN96x+/TsTG6xeRCEwHf7yu8BhKC/lsb4Sa4nGuAKYxxgGfkGbGhWjCHhSz6PZL9Jrj013w/53Qb5Z5KHwv/OQj2tTK64gWd2NOunmCcFxhnZayvFrYJxZeK3wTJ+hp07KGbgW7D6FYnEWTe+QR2g44cgvs5tjt7asIhP5l6we6j/bG+WtDHIh1JJg/4Z6Sy39m9onvDEmvCO6auINCltsAlhvYkfWKOssPfy/1yrBUYM5+kv5hnM9Z0HR8N4/xSD2GLrAv8oBv57+g7+In3Yc53JOk/u7L96tJfsRli2EKtsYUKqSfsyc3cDBkyZMiQIUOGDBn+3/gHwDVO2VFAEiUAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFgAAAAZCAYAAAC1ken9AAAEEklEQVR4Xu2YW4hVVRjHz4yXzASNnKa5nb3PzFAyYjaNaPMQXsDUQFJG0C5mIlhQWBQ+FIToQz0MaY2gD+UtnXFEx0qYl3wSLwWCIYKSBPWghCiKgvg4/v5z1hrXfHPG8Uxnjob7D3/WWt9lrW9/a+1v7XNSqQQJHntEUfQlvAYvw7Y4jhdam0cdxPxBOp3+1sqHQlVV1TP4bYdbmOMTqy8YSOy/LLLOyv8PqK2tfZ74rw8nwfi14/e16/eQ5NiY9MMojG7LcCgy6UrvlMlkXnSyqeFkxUJ5eflTrP8XPA2PEsdF2hvwJLwJb8G91dXV06wvKEH3g/Q2wYwXh2OhpqbmVeZ50o/x2wQ3u34PuZh+z3oQYPgy/JvuqFDOTk/Ubmki+LqX69VgfCm0LTZY/23ieNP12+vr68ucSgncBxfB34j/ncBNsb+vpKDbYROMbnZDQ8NYP9Yc/rTmAvouK8sJJj6gmmTlHkx0moWeDsZ3FGhow07XDfW6GIxWiWGubnjZncKjIgl4wRpbsNRyuEp9fH6iGe11jCsid8r8cylxyL6nW+psdqL7zvt4yJdYOuFaq7NQHnju16zcohTDq+FRVyFn8Tf8GP0Z33dj1Z4pRrY13P37AdsMPAG3EeAMq38Q3C/BvNJVyNqcbqmzn60Yo+wFLf6hGLTJ3s/Zf4Tt53BPWVnZhFCnUoH9bnQLnK3K5PbQZgAwapRhyu2sgNNKZJ/6sa0z6P4043m5TkMuuJN0Evs5VpcPiHEZc7ynfmQSrE1TklLZcrHVy0OgP2JLhJP3fhHRTsK3VYct0D2H7Dbtao1dgvuVoAHA4GOX4F64+nQh18mqrKys0Y6hP0+7gfZHeFz+cK61zwXsvohNeRkOmKclznGCkU1hfFCvOG0H7fzQz12QOsG6JMX1Xoftu6FtU1PTmHS2Bvdtnmzw+UbPrzUC89xQMC5BIW+kzIVXKDD3ITZqvJXnC3uCYWuU/XQ8FmXviCVKkHErOlR/ryupXqDyAH8OjQoJ1joXuctsMOo71fpZuAT3O8HIdunVpv0qyl5oDxcE85KSC68EshWMPwvtCgnmPmtlwwFxLvevtE8wG5Omv9PV+TOw2bgVF2lXfxWUl9GvIPhxoV0hwfyHYaOV5wsdhLS7YHyCXX+NTjaXUzX9EzrR/RyLCQLoUoIVrNWNFFhvKYn5xcrzxWAJduN9cRaz6Hf0ORUT/PJ5gsWvKMF6tax+JMGa+3n4jVaeD5jjLZUJ9Ul0Z/jW1dXVPes2sTTO/tor8boRB4E1wztKrOGH1nak4D6V9H+ATtoKPg1n6rhZu8HAV8jkKPgvAv4DT8FWb0O/Bf7u9N2h/2MDcvqKbnzaX10iehmbX4kJEiRIkCBBggQJEvxH3AUGNidsat6FUwAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAAAZCAYAAABaU4LDAAAEUklEQVR4Xu2YfWjVZRTH55qZJJramO7lPveuGTgWavsjJAVBCPxD9I9JMJAtC90QyjaKaU6jIKE/7A+dGmkaCuoUexMtjClIgi/TEAs1CmFEbyAUiFB/rM/Z73nmuefeuLvu3iH4+8KX55zznPO8nN/5Pb+XkpIYMWJ4OOcOwj/gADzleS2RSOwsLy+fZP0fVLDmefCctY8Apclk8m3Yy5530T5mHQoCFncLdijTI+i34UZle2AhiWGt/fCy7csFYtuI66uqqppOO0iiV1ofjTKc7ojjCNgcgqqrq2eJraampiHYKisrn8T2D1wVbMWEVBBzXYI/wlNs/IKf/1v4C/wXn6+wL7OxAvo24fMO7Nf22trap7UuYG81+M0OOrFN6CfkrpU8wLXaPytwehb+hFiq7Qw2Ffv7MhAJfSHY/ZW8pVzF9z1s72pbsZFKpZjS7RCZNbXD5aEPeR19r8AD9yKG++bCNbDVmUpG/wamgi4Fhd8ndXV1k7WfQCoY3+uSJ9uXgWR0tqy29gBZiJ4E/W9iXlJ6j1RN0PMBcUvhIbnItOedP+ORV1hfC27Xanx3i0zbQcxi3c8aD0uLfW+wIW/Bd6bItKtskj3KEtGdshV5nO0MSEZHzm+0622fRSmOf1IVc4KBhJbL5oNO/5Uge31QqijocqWxndU+uUCFTCRmDzwJl2Aqsz65wN1VGZLMet+0SabvU98O3WEcBVOQ98NuIf6fueho6Za+EIf9eWw9MjZyfbAHYH+LPb/u5dPwmvVJg4uesIMl6qhggBYGfzXo+gL4/qtax/emLFhkxpqv+/4PfhNvWHs+YB0zGGOPyFmSLMVzRAQu6DPKPgyZ35lKZsxFjPOaV2WMDZIj7YPej8/O+vr6R130AtCj+zMgA/okD4FJ5qLfkFb7CVhsFX3b4ABxm4I96Z+22D6g7dQx2UAFLsTvoLXnC5/kjEr2T32pVHlObE+PikBsu4vO37/g1lDJ2FuNa8jR8DntosLswfdD+jZnO6/TgPNRl/kmcbvEPAQLCRbXJRVj7flCJ5m2E34Mv4cX4a+wG58nbNxYQ24HebfVldzi/FlWLPhknHH3PmQySIU02TgLn+SPRKbtlErGth4+h76E9mpjY+N4GzemcP48hgPK1uzSPzIKDsY/SkJqrT1fyPElt6zIIcmIZdh6KyoqHk9Ebwg5j6+iIhm9S0qSh245gTyxOWMmaL9CQ+ZVD5f7hrzCZUmy7KEBfbvsA9vX6VFjDBZyTJLMQl+0fcUEc85k8zdHe15KkqVaRdZJDjrjL6Pap+V8MBULcpVZyO+SZPlstP3FBglog18kR/FzxX/xDb090K7VX6VgHLYv5c4k+Z+P9oLmBSaeD+9KcjVZRLv1LTaYtwP2wZdJxAL9QZAL+O9z6t8F/A7+AE+Ehx0JfspFbxo/w+vscYYd56EAm0+x+S7a4z5ZI/6sjhEjRowYMWLEiBGjQPgPm7Y52vOLvQoAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAA3UlEQVR4XmNgGAVDH0hLS8soKCjMlJeXPwrEWxUVFe3Q1cABUIEVED8FasgE8YF0JZD/S1ZW1hbEl5OTmw8UMwcrBjIUgJIvgYITYAYYGxuzAsW+A/EyFRUVPiB9FiYH0r0DKPAfqE8CLsgAtvUmFBcDcSlM0BuqeCKyYqjcYZCzgH7RhwsCTTcGaQDiGiS1DKKiojxAsYtA/BPIZUSWYwQKngPifSB3gwSAtnEA+UeB9CaQYdDQS1BWVhYD6wAxgBKTgHgzUGIXkN4DxJJQj88C4tNAvATdplEwhAAAER02bjVx23gAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAA+ElEQVR4XmNgGMJATk5uury8/H8o3ocujwIUFBQcQAqBmtrR5VAAUOEmqEJBdDlkwARU9B6Ij6NLMEhLSwsDJVqAJp0CmpIPdV8ziiKghA9Q8CWQjgFiYyD7N9RaZ7giWVlZU6juKTAxIHs/EP8Cms4BVwhyB0ghUIMyiK+iosIO5H8FmrYTWZEk1IqTSGJOIDGgaZXICsGCQDwNJgZU0AjVbA00nU9KSkoEpNASqrAYSfNRIH4NZLIA6SIgjoK55wnQlIVACUYlJSU5IP8DSDHII0D6iLq6Oi/YBEVFRT2gwD4gvgzEd0FWgsISyF4HZLvCbBoFlAMAb85GteESHIIAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,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>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAZCAYAAABdEVzWAAAB80lEQVR4Xu2UzytEURTHx6/82LDyY8zMe29mwgZNQymlWFghxYIlNlJCKZHyKwsWKLKQBU1RkhILvxcWsyIbC/kD1JRSFiws+Jy8l9vLwkiTeN/6ds/53nPPPXPmvOtyOXDgwMEfgc/nq9Y07RhGTU6j7bLuG4ZRbI9PCPx+fxFFHFJEgfi6rq9jD3q93krWGHsL9jMJAZfveTyeoOVT2BbMQO+Cr7BFjU8Y+KvKFTeJQu4U/3eAokLSJbv+E2A06sl9Zde/BJkttTBsIxwOp6kx3wXjUsiItNn1TxEIBHK5/FrmSnzt/ct8MbeTsc/4OLKVI4kBF9dKhyhswrTv4QNbqXyN8+j9ajzzWIa+RswsHIUX1h52BG6wP8S5Aewdt9vtDQaD6fjj8AitTs1H7BjaHAypuouDWXIRPJAng4vzCFoRn7VTjSVxPlrMSs7ag3YuNl31ycq5Vngrbx/7T8yVm5humCOFwQ4rHzE1Zp5FzvRZetwg6QxJopYvCeGk2NbfLW8e2pIVI+CcLoWjP0uBil4lzYCNanzcIPEJnFL8G+meZj7MpnbJhc1iS7cUfQRGZKb5CDKRUvAN/X2EHm1PVnwgSQlJllmH+ZW92JvSNQqogE1mR2XOVuG2reBTiSWmXXyKKzWLkpls+LjFgYN/jjcDp3ipwkQDCAAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAyklEQVR4XmNgGAVUA4qKiuLy8vLTgPgMEFeDxBQUFAyA7GtA2h+mjhnI2SQlJSULlCgF4v8gQTk5uVwQG4iXglUBBWKBCitAbKDgeiB+BTWAEcg+AMR9YJ4CBAgANQgCBb8D8WyoQpAhHUC5cBgfDIAC6SCrZGVl3WBiQP5ekAHI6kCCu0EmGhsbs4L4QAXaILejKAIBqA9PwfhAhROAoWGHrAYMQG4D4qcgE0FBAmS3oKsBA3V1dV6gKf1AvBGI24FCLOhqRgHlAACYOixw5QKgvAAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,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>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACXElEQVR4Xu2VO2gUURSGExOwsAgYcGFfM/tAFCSKjyJY2Cg2QgohIQERQdRIGiUqEZVgZyVYKYKFIiYgCpKAYNBGEEQQGy00RBCEJIWgWFiIfmc5dzlz2d0YwVmL+8PPzP3Pf86cM3ceHR0BAQEBq0AURf3wAXwGX8VxfMr3NAP+cfgdvoAnkbp8z38BhtpPg8sc98i6XC4XWb/1fY1AziTep+Rs5PgQ/ioWi9d9X9tBowPSHDzq6dvQTlvNh+RILt5ppzHkVa1303rbDhq7pI0dtLrsKnhsNR8MOKy5C07j/LJqt623hkKhsJPAXbgI7/vxfwm589IYQx2weqVS2WAHaIJOPCMMHDuB00c66Ljx1QKjiF85niiVSplEsAHwTWihP+USaWv8Og7E74nPHzSXy/Wif7baSmDDKuT8gPOZTGZdPUDxQb3IIeNPFVz/TqNB8/n8+mjlHU1Ad3OJgbckAogfzJ2vMWFIAe7jQZMDVpdHF+2l1VqBGmfgF+rtSAT0HZDhZhKBlMH1L2gfI1bnNUKKZq3WCngXyNnq6/JoVPUCt/xYK/zFOyrvWbdfx4Ed2Ke+Mavz+O1Cu2i1ZqDGXvra5NbklaTP2qJara5F+Abn6hntQRc9zEfef4/mj8GyW/O76WF9LpvNFqyPXeyTfKsx5JD8tuoChvPwJ+Juo22uG1KCNvuJBmNZ6ya8sR7W13TnnziNL3Oe9Uf4Hj6H7+Cy+Cg1bPNl+uMEXsvPWYolPsspQm4wvUzSxxTHG/CwjbM+ogNccRreszp8I263+QEBAQEBAQGrx2++rcF3laPD/wAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAZCAYAAABjNDOYAAADeUlEQVR4Xu2XS0iUURTHx7Q3vYspdeZzhiHLHhQTtWjT+0HQE3pgUUgQRUGLVkUE0aakFgqSEkGYvVZtksgKo0grwtKiRQ8Cg1qEYjuTsN+Z79zxdhnLFj4W3x8O33f+538f53junc9QKECAAAECpBCJRBYVFBTc8jzvAfYSO1FUVDTC1WUCY+eir2f8c57l+fn5k10NsYnYTeIfotHobWyOqxmSYMMrsO8kuUb8WCwW1gLVuVoXaBejayfxfbhZJH0Dv5b3YUaTm5s7Fe41VpFMJofzPIx9TiQS09ITDUWwyfVYN0kdsXkKlSs8VmrzNhizWjUvDEfCI5W7Iz6FGcN7l64xyRr7CK6tr905KGCDJzWZYpunE0YJz7PZ5m2Q4G4d22Dz+J1Yh7xTnIhqugsLC8dZmlotWLJnZCiUDXkc+yRBGWwH2cw9+JjN9SdY65IWYYcTyjJJOXwajFmnCT4znBTAjIvH4xPC4fBY2zc6/KfC0aEbDSfkZRW3Ym+wd9KKEmORVfjVabEDbdkOs1hfjAQq3Xls6B2RqTiy118Sc3kD9jOeeDv21XCSrFnbXLq814kvF7f4cmF7etRY92BqIC+bIB7zXGAmw/+IbdP3+wN9i7OXK38pTioBl7fBuLVo2tj3Hp4zPL8Tf+icBaIhFsd/i5XLHcPznOc3hmj2pibCOZ2Xl5fvTL4LvkoKJn9FOzYQYO0LusmdTsgcq3RX9AY6IaFFruI9T8f9JJRjNHrcSrEmtEt4PtR1l/XM5EAGIWhGeJ7iLHXj/Q3Pv/8kmRKbJ8nRyr+y+X/B87tHjtRdN2YDTQvWIb9mbuwPeH6L1bu8C23J/7pzsAp3HhsksVJ1x2yeyzMqPPGLNu8CTTFWw4mYIr65cxi33WjMB6Z8E4mv3z3yi1ZuNL0CUSODD7j8AEGOz3vP+SHA36pJprsZvwRbbuui/veKFHeDaq7LfCHrSOGfEQ05nlX/ENZJd84zml6BsEFufpcfKLD+bKyVbpmpVA7+E6zMaEhssxahK2p9zHn+ndWid4oU7wsdMt/EVbNFiqG/VvLH+OY531UZ4fln9L/OdX9ACkPSp9jLNQpR6XYy/izP/xVqws02vNxN0hHwNVgZ79OtYWlo4aqxRmyhG88IhPuxqy4fIJQqTg3VPuryAUL+vwsuFyBAgAABAgwafgN+KCVR1DpYRQAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAZCAYAAABjNDOYAAADCElEQVR4Xu2XTUhUURTH1WpRQR+0GHB03jgZBBEEtigpgqA2ZQR9URSF5MaEaBP0oRFWkPRFVIYGQpntKglaJIVQWia4KCja1DKoVUIJtajfac7J62mU2TRKvD8c3r3////eOfe8++57U1QUI0aMGDEE06IoOkr0EC+IB+l0epk3jYOSVCp1hjHviD5igzfAbcFTU1VVNUP6ZWVlS+Hued+UgyQsiZJ8d2lp6SzhaB+E+0Gs9X6HEjydxHOKOa+8vHw57a+014QmuMvEzyBGiNrQMyVBks+ITzn4a7IQWbTXDOi96tkccKe0ANUBJ8WZ+sVwkMdJdsgbL8Cd1oXv8JoB/a142GkbjcN/RMcdMg79Ur7FkYTOR9mtJRUeE0zU6gf8K7CAtC7kldfIo0lzOu41A9pT8fA4bTKOuc7quHPGSXFkPqID/RFaZ2Vl5RzTDcViQBzieoJrI9eLcrXgh1b7QQb0lfrDeQfz7/bzGPRgzFkc+GM6R7PXDFE2Z/HUG8fvDSjXFfhaiF4rCO0P+O6b/ht60F2gWSz9TCYz107wyQDFWTRBceTtJcVt8pqB8TPx3CX6ksnkAubZF2XfdlKcG+Zjjoys1fpozeIJd9xfYLK9niskSHq+LuS116Icu2IcFONpkALJLmVNJ3Vcozca8NWp55bX/gBTt+cKDFnYd+K9F1LZbxfZOdu8NhEYc1XG2fFAu5r4SLQHnl1anMejIwPINwXiN89PBO7KCp0072DMTj9PCDxPiOEcfJuM53FIec2gO6+L656AGwgfU/R6zeNlwB0QDu9148ZAnjcxeL7Q0HNCkl8c8nBDxEPrk+9CFnM4kUjMDrj1ehP6pY++RBddZx69oYMVFRUJ4+i3q2+dcWOA0IphxPOTAfK4Qj4d1qddA/dZDuzAM6iFaDEOPUl/mNivnn6Zy3QDXKcWbLr2vxBtzjYKxDvjbqvCQ/4G1EbZ76/bcuOITGiQXKU4XLc7fhX8TaKH9tZQM8gRIrsuyn5RSzQU6Rs7RowYMWLEiBHjf8cvdHwLwfujGiYAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAACvUlEQVR4Xu2XW4hNYRTHZ+RSyiUKnds+N06dRBxFkVCKcnuTlFBeUF5ciigGD8rQJLmVZMqDF8TLDBnJdcibJ8zTKE3R5MGT+K05ax/LmiPnpMYZ9r/+7f3911rf/tba6/vOPk1NESJEiBChJgRBsB8+zGQys1KpVDadTl9h/I37G97XA7828TX8Ard6v2GJbDY7g2R6E4nEJCOPQHshyVKoDUYfBHzOwNfwI3yZTCbneZ9hC5LfokW4ZfWg3E3SCXes7hGUO+ff6BQPirJei9BndbbULtXvWt0Dv9N1FSeXy00h6BRBj2Vy7ld7nwZCsxSI7ZCzImu+rMVptbqHFAceEn/m6cC/PZ/Pj/d+A8A4F/bAAzImaJM8RBag9j11VdqAh44htl8XXRN57nk/z+9AzDRiP8G+WCyW9HYLfE7ArrAg3PdUPchlIozv5bS3Otob+ETvX2GfaO2NBu2afq4Lvc1Dft3AhHBMXIu8FDpxjfUTx5tq+KlF0Tr1La7jesnaGg2scTNr/Ewui7ytFhC3TXKFVysik65QcVDyaNe0OGudvlK6DB62+q9QLBZHB3VuK3jWz1MNUgx8e+GcUJMOZ3zS+llg2y7PwO+50XaIxnznKo58QM3WxbRURFAqlUYFPzonbW0hfMxQg0SKrOEdy5tvdbRlaT2zCoXCOPz2oWVCO7YFjLvJfaqJuajFWR5qoeEpfCQHp4wJWsz4HryuxVlC0Cq23UwX99eKI4kF5TPxLexkfQ+C8kfdB/gV7hY/rq2SQ6BnZwjG7bKVuB2pY+nsC9ZnAPF4fLJOcptCdMh2kQNLbFyPo3fD+/IWbBzaETseSkjymnRVpvQzhPuNqrXZeH6ExuKzF71LuRO52fr8EZjwqNciKCjOMa9FUMiW89p/DzpmKTxIcZ7JlX/G071PhAgRIkSoD98BUwXk9sDucOoAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAZCAYAAACxZDnAAAAEhklEQVR4Xu2Ye4hVVRTGx8ykh9prGpvHPXfuTA0ORdQNCyPB1JTCJKlUygilh2B/Fb18pKgRkQ9qFPOBmGNYBBEolMOYBZOWEGEEauD8EfRCCv8IsSHq992z9nXd7YwY2ujA+WBx9v7Wt9Y9e52zH+dWVWXIkCFDhgz/A3K53EPYlGKxOCRw9fX1N8N97HXgoiRJXsE6sL3YzoaGhtsjjfI9Ih+2B+vK5/MzY01fILaI/gPiOrEFzc3NQ2PNgAUDasP+iew4g34qaPQQ4D5S8Wtray+zuHlYD9wEl+s57CDFyqvP5Q76v3NdHDR9Ad092G9oJ1l/P9YZ6wYsGMxb2OyY98DfpSL0wpceEjPgagr+utqNjY23eA2Fmy6e61TPe+C/TxpyPBs49CPFYSu8dsCCwa1mMHNi3mEw/h7sUOyAW2pFnJGkS4oKnXgN+SdawdZ43gPfQiv044FrbW29xOK+89oymFrXErAKwZfYZ6xjD8SaCwlWaA10EwXbxfUotq1QKIyQX8uADfjbOBZuvhVoEXZY7aampuu8hsKPtfgOz3vg22h5Ho74v8XTHOR5OW7DunUD6hM4S0I9cfM/D/dkRdAZQhsD8cfsps/UNsV5YqB5g3v6nPzD1efFqIXr1nqsvjZG5WIMByojSw/hZfudZdgvphvpNeS5S7x+w/Me+LZb7HTPw50QX7Ep8iY3QP6Eveu0SnIY+0ptfN/Qvsr7zze4n0J4ewMY8GIr4IMU+obTFPol80nf3Vuh4cZYrj43NmK2WGxc6OPiaV5cJvUGiGSqtJyUlpJoOuqHpqHZ7H0XKrjXOXbP7XoxrH3KWpnY0pGkJxCdEtS+3mvC0kEd3ve8B/6VpinNfMdr6fjVE432IwedLvg+NJ/OlPnA19XV1cNtTdLz5jgX0m+waf0ztsHzerPsnnfTHcT1L+yI1wjEvyad9LQ/UZsZ0Ow1+CZbrjbPeyTp+Vya8umHPJda7pMzySXbWCYNcO3mm+Z5buxTnvZoYqfiO0G/6P0xbBf+r2v0+jiPB7/9jHRcv474uT6e627sT68x3TvS6aRBe4nlutNrGNdj4rnO8rwHvvGmeSFwthRXjkFnRyOXl8mq9KCft6WDTabJ++COJLZpJulOv9L7+wN6uPzufu6/JuLXWdFKHw9cn7DxjfI6xepNVpt1/kb6PWifjjQrsD+qq6uvUL+lpWUYMS9iBSfTrPkBaw+EvYCq291OV0q4D+sKO6StTZ359JNST0tP7X4Cb/Jx6KvNXz6s9yf47a259CRU2nDo35qkM6didtJvYyxbXF8fGUdV4MDhn4EdCF+P+fSjQ1+G5c/wxNZjbG/gjB+F/RjyEfNF0tvZmzX3GkuyQ8sCwiXhqcEtT9LNYg/8lT4O7k3s+5qamss9319QUTRldW9mekv10CvPrul/HbNtjNuwtfEsFeDHYcvsBVufc5/o5n80SQv9tucFFRn9q/jeI35u7D8rkLQjnroZzjEo8hgug9XmCU6O3BnOAfT/wSGbQiVjGt4bizJkyJAhQ4YMGSrwL78ikI5rHdVIAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAZCAYAAABuKkPfAAAC8klEQVR4Xu2WTYiNYRTH763xmSK6g/v13q/S3AVxw4INzWBB8pUFERtmoSjZ0IyklI+RhfKRj5SyQZqxMDMbZBBF2AilLJAFG2lYjN+573ne+8zTyEhd9473X6fnPf9zzvOec56P941EQoQIEWIEoVAojPE87y3yBelNp9MrXB9BJpOZj/0q8hifPVANrk+9Ikpx3RQ2PZvNzmN8igzAbbOdKLoF/hOyWHyxP2PstH3qFhRyHum3OQq+LY1g3KH6cm1Mq/FJJpMF9WmvRNYpKOSkFONwXcIhu0Wn0DZtwnrjQxMmq09fJVLBlpqK8wmM92WbpVKpJa5PLSEWi00gzy02R+4fdJWbVT+nBa8xPvF4fLxynyuRgILnev4FU+6gbB+drHzR8HwQWT0o6A/APAv0xcMS3r/MneN3IG6VxvcajvdeUS5ogl6mwn0znOwAdO89cjYgI+VJPyI9PDYwPiexsba9lqCr+wZ5Qp6TDI9+yW1CsVgcrVxlJ6DcVDIbkD7fh3ylmxuR47atxhAlz2vIC1Y5Zhskb6mNca3hrOPwskxwDPJCyGcjiFQQeEttd5Bphk8kElPgzyAd8LvsmH8ByUMWLJ/PNw5h26c1bDac5i9NuFsm5Oxpp04FkQr4TnVe5PCX8T+kzxKbs+1DAZ+FOtewBP8Wd45fAd+HuVxuotGpaSncOrU1y3xwO40dvUnfc9QQs9Vpv3ESyOUhO0Bs0jnbBncA6dBnSbhk26sJOaruESCnI+S+UlU5Kq/xu2jZN0je3IWzDCfkIylYCldd/qzuef45G0DmyJmSn4wgqBJ73eWqBc2zH+lBHiCvPP/3WXJuMn7kXkR/J/nLpSg7B/2wPZeseizj/x906T2wl+2VxhRN+xeLNKk7Y30dZPVlIro505qqqiCHG1qwKz9KpdIo25c8Z+DfTg2nsW+1bX8NJvwuZ9DlRzTYUuPo6AW5UEXXzg/6v/gvQBM2UfgxxjaasR0q6vqECBEiRIg6xk9HXPa6cSP/dQAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGkAAAAZCAYAAAAyoAD7AAADzUlEQVR4Xu1YS2gTURSNxQ/+UYzRNs1M0mogG5XgohvFz6K6EBU/oIgfxE13olbB2kVx40IRf6VShYIiiHRRF2L9Y6uioii4UEsFCy5cCgq6qOdM7jS3r5M2AYtJ+w5c5t3z7pk37973XmYSCllYWFhYWFj8d1RWVi53HOcm7AHsteu6x8yYXBDtgC6VSk02Y6qrq8OxWKyZMbCWqqqq+WaMRiKRmG1y4xpI2mrYdyRxjfgLYW/MuCD4Wml7Otznjo6pqKiIgv8Aa0CxpuB6DdbHwuk49sXj8SXoOwz7pPvGNZCM9bB+WJ3msSNccCc1ZyJIKzpynpY7Au0/KNyNrDIUgv8RoU805wP8dt7D5MctkKwTktQdmseqngXupeZMBGlFR87T4rqKPmIvZZUe/5x8eXn5PM0TeRcJ2y6C4LMI7oLdg60zY8YCMK9WSeIWzafT6Ungf2rORJBWdCySp0UO99BnLrNKT/tQ4mo0T+RVJAyaRtAXBB+lj/Z+ueEm8Y/Ddg5W5Q+5VyHmjTsa4DHEMcwiAWXkDW4Qcmg9na/FtY7tgCJx4ZNfqXlixCJh+1Ui4BusVfPw+/Awj9GcgPb7cDg8Q/eXKjCXtoBEE5znL4MbhBxa6vp9rSMLPKBInRJX+E5C520RxzXPAoH7zQeCNeu+YgCe7ZZMfERDEvb6OszlDOeL6zZ9Pzm2ejVnIkirjjtPi+tm8c9lldnjLhqNLtI8MWyRMFiCnQh6Z/aB75DBuvhKafYT6LtgcsUOJ3N0c84DhSP4AoB8vNCciSCt/+Lga3mcSd6uZJWe1ntxCPoeGrZI6KyVAYbsFCfzoccHqjX7CK6IfIskD5234WNxg3mPfwXMda2Mc9DgE+YRZSJIqxa6p00mkzPh/4C1Z5VeDj7D3mrOx0hFWiqDNmletjC/qDn4At3nA30tsIsmXwLgbwgT1qZJJHsr/0nwfSYbXD3m76qwIVrqmCetBXcVXA+aZfT55ix5PuTHaAxbJMLJbMOn/Pql72a26yMnc+bzxjXgNmKbLlaafbBlTp47qdiAJKbw7F/V78NE+N1GjPf7E8B7WnE9HWN1jOymZ8jbbvpon+Zx6OfYhCuv7SY/AH5ccasiqAPXu7hZo796wJ2CvYLdR9w0clgVScQcYTtmfLCVEmQejZjbdVybYQd0P/xdTJxZAILaXDof4OfA6pkjxDZFIpHpZgz4dlgvxxHjX0mdWDxzzdhCwJVzOSTbuJSLNGaBFbQCRToPaxDrdowfYIsiAbbjVClSj1vAX/wWFhYWFhYWFhajiL8ylWggCmMz7AAAAABJRU5ErkJggg==>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFYAAAAZCAYAAACrWNlOAAAC0ElEQVR4Xu2Yv2sUQRTHT1AUQRBUlPs1e+fJgTaaQy3UwoASC5UEBVELCYpaBFH8kRgEkShEELET8h9YxEJFzHUBtRARjNiIYCNqp4KVRfy8u7d7s5MzXA6u2Nx84bEz3/edmdvvvp2dJJXy8PDw8PDoIIIgWJ3JZNa4/HwwxpwgnjN2mmu/m+9alEql5ZiyNZ/PX8WYz1yvuZr/gXEjjPmYTqdz0qf9m7js6roaGHKcmMWsYTfXDGjvip4HscXiBoUjem1tV6MNY6uq32Bx/cqN29oIhUJhPckHiF5zncrlcvtdzWJDG8Z+Ej3bwFqLO6AVO2lra8DE7SS+GN0rWOi8iCn5g9KnPUYMxEe1DubZpYu3FKzf587RCbDOMVmv1T0W7XfR28ZKAervfmZrpVLhzDdiwubp/yCqNJdyneFHrLDziwFm4RUrxRczloeyT42NV6w4rYmCw78i/jDwJHE/5HkQO+gfXegRpROQB99quGMFpmHsiJtrBrRvRc+pYp3FhVtBozAp44068fuIVGDeC81NO5v1JZ1I4rQ9JmkIjSWuu7lmkG+PehKEHP0B4fDrji3sU/JhRCrgn+iie22eMRfRn4Hvsfn5gH63ztVSyOvlztEJmIaxo26uGdDdUn107/hxVjiK9JAt3CYkyZsRmYoO0PJXxaz7ynPTF+RjZ3NJhWXsDTeXzWY3wV9hP10ZcmyDZbi/xGDISVHS/1qpVJaFXA2Qb8REMVP7vcRLYlIX7WHwERYq6URDcGOMGRY+NlnCIAbpPd5rkqvtp8Rth5eH8c7y6xdeHLY1NchGHNTPr091Xx0tFot5Ukvko2Xqxk8FeiqQ15T2TmnLwlTvntiECQC/+7HRL7wVM0S1XC6vUs2E8nP+F2DqxSfF9agj92/q57o5r5FHG8DIDzypU9r+KVuCI/FoB1L6mHmOGGdb2OzmPTw8PDw8PBKIfxi6+YnYqY7cAAAAAElFTkSuQmCC>