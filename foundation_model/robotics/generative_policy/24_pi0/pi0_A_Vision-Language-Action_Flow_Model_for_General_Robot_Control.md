# pi_0: A Vision-Language-Action Flow Model for General Robot Control

**Physical Intelligence**
Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky

https://physicalintelligence.company/blog/pi0

arXiv:2410.24164v4, 8 Jan 2026

[Figure 1: Our generalist robot policy uses a pre-trained vision-language model (VLM) backbone, as well as a diverse cross-embodiment dataset with a variety of dexterous manipulation tasks. The model is adapted to robot control by adding a separate action expert that produces continuous actions via flow matching, enabling precise and fluent manipulation skills. The model can then be used directly to perform tasks based on a prompt, or fine-tuned on high-quality data to enable complex multi-stage tasks, such as folding multiple articles of laundry or assembling a box.]

## Abstract

Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems, as well as to address some of the deepest questions in artificial intelligence. However, bringing robot learning to the level of generality required for effective real-world systems faces major obstacles in terms of data, generalization, and robustness. In this paper, we discuss how generalist robot policies (i.e., robot foundation models) can address these challenges, and how we can design effective generalist robot policies for complex and highly dexterous tasks. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We then discuss how this model can be trained on a large and diverse dataset from multiple dexterous robot platforms, including single-arm robots, dual-arm robots, and mobile manipulators. We evaluate our model in terms of its ability to perform tasks via direct prompting, follow language instructions from people and from a high-level VLM policy, and its ability to acquire new skills via fine-tuning. Our results cover a wide variety of tasks, such as laundry folding, table cleaning, and assembling boxes.

[Figure 2: pi_0 controls a mobile manipulator to fold laundry. Our model is pre-trained on diverse data from 7 distinct robot configurations and 68 tasks, and can then either be prompted directly or fine-tuned to complex downstream tasks, as in the case of this laundry folding policy, which fetches laundry from a dryer, packs it into a hamper, brings the hamper to a folding table, and then folds each article of clothing.]

## I. Introduction

Artificial intelligence systems come in all shapes and sizes, from highly specialized systems that solve complex problems inaccessible to the human mind, such as predicting the conformation of a protein, to systems that can produce lifelike high-resolution images or videos based on textual prompts. However, the axis along which human intelligence most outpaces machine intelligence is *versatility*: the ability to solve diverse tasks situated in varied physical environments, while responding intelligently to environmental constraints, language commands, and unexpected perturbations. Perhaps the most tangible progress toward this kind of versatility in AI can be seen in large language- and vision-language models: systems that are pre-trained on large and very diverse corpora of images and text from the web, and then fine-tuned ("aligned") using more carefully curated datasets meant to induce the desired pattern of behavior and responsiveness. While such models have been shown to exhibit broad instruction-following and problem-solving abilities, they are not truly *situated* in a physical world the way that people are, and their understanding of physical interaction is based entirely on abstract descriptions. If such methods are to make tangible progress toward AI systems that exhibit the kind of physically situated versatility that people have, we will need to train them on physically situated data -- that is, data from embodied robot agents.

Flexible and general-purpose models that can be tasked to perform a variety of robot behaviors have tremendous practical ramifications, but they may also offer solutions to some of the toughest challenges facing robot learning today, such as availability of data, generalization, and robustness. In natural language and computer vision, general-purpose foundation models that are pre-trained on diverse multi-task data tend to outperform narrowly tailored and specialized solutions. For example, if the goal is to recognize birds in photographs, it is likely more expedient to pre-train on many different image-language associations and then fine-tune or prompt for the bird recognition task, than it is to train on only bird recognition data. Similarly, we may find that for effective specialized robot systems, it is more effective to first pre-train on highly diverse robot data, and then fine-tune or prompt for the desired task. This can resolve the data scarcity challenge, because many more sources of data are available to a generalist model -- including data from other tasks, other robots, or even non-robot sources -- and it may resolve robustness and generalization challenges, because the diverse data exhibits a greater coverage of observations and actions, providing a variety of scenes, corrections, and recovery behaviors that might not be present in more narrow specialized data. Thus, adopting a large-scale pre-training approach to robot learning has the potential to address many of the field's challenges and make practical learning-enabled robots a reality, while at the same time furthering our understanding of the deepest problems in artificial intelligence.

However, developing such generalist robot policies -- i.e., robot foundation models -- involves a number of major challenges. First, any such research must be done at a very large scale, because the full benefits of large-scale pre-training are often not present at smaller scales. Second, it requires developing the right model architectures that can effectively make use of diverse data sources, while at the same time being able to represent the intricate and subtle behaviors necessary to interact with complex physical scenes. Third, it requires the right training *recipe*. This is perhaps the most important ingredient, as much of the recent progress with large models in NLP and computer vision has relied heavily on delicate strategies for curating pre-training and post-training data.

In this paper, we present a prototype model and learning framework, which we call pi_0, that illustrates how each of these three bottlenecks could be tackled. We illustrate our model and system in Figure 1. To incorporate diverse data sources, we begin by utilizing a pre-trained vision-language model (VLM) to import Internet-scale experience. By basing our model on a VLM, we inherit the general knowledge, semantic reasoning, and problem-solving abilities of language and vision-language models. We then further train our model to incorporate robot actions, turning it into a vision-language-action (VLA) model. In order to make it feasible to utilize a variety of diverse robot data sources, we employ *cross-embodiment training*, where data from many robot types is combined into the same model. These different robot types have different configuration spaces and action representations, including single and dual-arm systems, as well as mobile manipulators. Additionally, in order to make it possible to perform highly dexterous and intricate physical tasks, we use an action chunking architecture with flow matching (a variant of diffusion) to represent complex continuous action distributions. This enables our model to control robots at frequencies of up to 50 Hz and highly dexterous tasks such as laundry folding (see Figure 1). To combine flow matching with VLMs, we use a novel *action expert* that augments the standard VLM with flow-based outputs.

As with language models, the architecture of our model is only part of our method. In order to flexibly and robustly perform complex tasks, we need the right training recipe. Our recipe mirrors the pre-training/post-training separation commonly seen in exascale language- and image-language models, where the model is first pre-trained on a very large and diverse corpus, and then fine-tuned on more narrow and more carefully curated data to induce the desired pattern of behavior -- in our case, dexterity, efficiency, and robustness. Intuitively, training only on high-quality data does not teach the model how to recover from mistakes, since mistakes are rarely seen in such data. Training on only lower-quality pre-training data does not teach the model to act efficiently and robustly. Combining both provides the desired behavior: the model attempts insofar as possible to act in a manner similar to the high-quality data, but still has a repertoire of recoveries and corrections that it can deploy in the case of a mistake.

The contributions of our work consist of a novel generalist robot policy architecture based on VLM pre-training and flow matching, and an empirical investigation of pre-training/post-training recipes for such robot foundation models. We evaluate our model out of the box with language commands, with fine-tuning to downstream tasks, and in combination with a high-level semantic policy that outputs intermediate language commands to perform complex and temporally extended tasks. While our model and system make use of a variety of ideas presented in recent work, the combination of ingredients is novel, and the empirical evaluation demonstrates a level of dexterity and generality that goes significantly beyond previously demonstrated robot foundation models. We evaluate our approach by pre-training on over 10,000 hours of robot data, and fine-tuning to a variety of dexterous tasks, including laundry folding (see Figure 2), clearing a table, putting dishes in a microwave, stacking eggs into a carton, assembling a box, and bagging groceries.

## II. Related Work

Our work builds on recently proposed methods in large-scale robot learning, as well as multimodal language models. Our work is most closely related to recently proposed vision-language action (VLA) models, which use pre-trained VLMs that are fine-tuned for robot control. Such models employ autoregressive discretization to represent actions in a manner analogous to text tokens. In contrast, our model employs a novel design that fine-tunes a VLM to produce actions via flow matching, a variant of diffusion. This allows us to handle high-frequency action chunks (up to 50 Hz) and highly dexterous tasks, which we show pose a major challenge for prior autoregressive VLAs. This resembles a number of recent works on diffusion models for action generation. In contrast to these works, our model uses a pre-trained VLM backbone. Our contribution is also fundamentally integrative, focusing on a *framework* for robot foundation models, including not only the model architecture itself but also a pre-training recipe, pre-training and post-training phases, and a range of real-world experiments.

Outside of robot control, many models have been proposed that combine pre-trained language models with diffusion, including models that specifically hybridize diffusion and autoregressive large language models. Such models are typically concerned with image generation, but our action generation model builds on a number of previously proposed concepts. Like Zhou et al., we train our model via a diffusion-style (flow matching) loss applied on individual sequence elements, in lieu of the standard cross-entropy loss for decoder-only transformers. Like Liu et al., we use a separate set of weights for the tokens corresponding to diffusion. Incorporating these concepts into a VLA model, we introduce what is to our knowledge the first flow matching VLA that produces high-frequency action chunks for dexterous control.

Our work also builds on a rich history of prior works on large-scale robot learning. Early work in this area often utilized self-supervised or autonomous data collection, providing a tractable data source for simple tasks such as grasping or pushing, but without the complexity of more dexterous behaviors. More recently, a number of high-quality datasets have been collected for robot control that enable broad generalization, but typically for simpler tasks that consist of object relocation and rudimentary furniture manipulation (e.g., drawer opening). More dexterous tasks have been studied at a smaller scale, typically with 10s or 100s of training trajectories, equivalent to 10 or less hours. Since one of our aims is to study complex and dexterous behaviors, we utilize a much larger dataset, with about 10,000 hours of demonstrations, complemented by the open-source OXE dataset. To our knowledge, this represents by far the largest robot learning experiment in terms of the amount of robot data. At this scale, we show that a more sophisticated pre-training/post-training recipe is highly effective -- analogously to the recipes used for large language models, a pre-training phase endows our model with a broad base of knowledge, which is then refined in a post-training phase with higher-quality curated data to achieve the desired behavior.

The complexity of the tasks we illustrate goes significantly beyond prior work. While recent work has illustrated a number of more complex and dexterous behaviors, such as tying shoelaces or cooking shrimp, we show that our framework can learn very long tasks, sometimes tens of minutes in length, for behaviors that combine both physical dexterity and combinatorial complexity. For example, our laundry folding task requires the robot to manipulate a variety of clothing items that can start in any configuration, and fold multiple items in sequence. Our table bussing task requires discerning the class of novel objects (trash or dishes). We show that a single cross-embodiment model can be used as the base model for these tasks. To our knowledge, our work demonstrates the longest dexterous tasks in the end-to-end robot learning literature.

## III. Overview

We provide an outline of our model and training procedure in Figure 3. In our training framework, we first assemble a pre-training mixture consisting of a weighted combination of our own dexterous manipulation datasets (Section V-C), collected on 7 different robot configurations for 68 different tasks, and the entire OXE dataset, which contains data from 22 robots. The pre-training phase (Section V-A) also uses diverse language labels, combining *task names* and *segment annotations* (fine-grained labels for sub-trajectories, typically about 2 seconds in length). The purpose of the pre-training phase is to train a *base model* that exhibits broad capabilities and generalization, but is not necessarily specialized for high performance on any one task. This base model can follow language commands and perform a variety of tasks at rudimentary proficiency. For complex and dexterous tasks, we then employ a post-training procedure (Section V-A), which uses high-quality curated data to adapt the model to specific downstream tasks. We study both efficient post-training with small to moderate amounts of data, and high-quality post-training with larger datasets for complex tasks such as laundry folding and mobile manipulation.

Our model, which we describe in Section IV, is based on the PaliGemma vision-language model, which we then further train with our data mixture. To turn the base PaliGemma VLM into pi_0, we add action outputs that use flow matching to generate continuous action distributions. We describe this design in detail in the following section. Note that we use PaliGemma for convenience and because of its comparatively small size (which is useful for real-time control), but our framework is compatible with any base pre-trained VLM.

## IV. The pi_0 Model

The pi_0 model, illustrated in Figure 3, consists primarily of a language model transformer backbone. Following the standard late fusion VLM recipe, image encoders embed the robot's image observations into the same embedding space as language tokens. We further augment this backbone with robotics-specific inputs and outputs -- namely, proprioceptive state and robot actions. pi_0 uses conditional flow matching to model the continuous distribution of actions. Flow matching provides our model with high precision and multimodal modeling capability, making it especially well suited to high-frequency dexterous tasks. Our architecture is inspired by Transfusion, which trains a single transformer using multiple objectives, with tokens corresponding to continuous outputs supervised via a flow matching loss and tokens corresponding to discrete outputs supervised via a cross-entropy loss. Building on Transfusion, we additionally found that using a separate set of weights for the robotics-specific (action and state) tokens led to an improvement in performance. This design is analogous to a mixture of experts with two mixture elements, where the first element is used for image and text inputs, and the second is used for robotics-specific inputs and outputs. We refer to the second set of weights as the *action expert*.

[Figure 3: Overview of our framework. We start with a pre-training mixture, which consists of both our own dexterous manipulation datasets and open-source data. We use this mixture to train our flow matching VLA model, which consists of a larger VLM backbone and a smaller action expert for processing robot states and actions. The VLM backbone weights are initialized from PaliGemma, providing representations learned from large-scale Internet pre-training. The resulting pi_0 model can be used to control multiple robot embodiments with differing action spaces to accomplish a wide variety of tasks.]

Formally, we want to model the data distribution p(A_t|o_t), where A_t = [a_t, a_{t+1}, ..., a_{t+H-1}] corresponds to an *action chunk* of future actions (we use H = 50 for our tasks), and o_t is an observation. The observation consists of multiple RGB images, a language command, and the robot's proprioceptive state, such that o_t = [I_t^1, ..., I_t^n, l_t, q_t], where I_t^i is i-th image (with 2 or 3 images per robot), l_t is a sequence of language tokens, and q_t is a vector of joint angles. The images I_t^i are encoded via corresponding encoders and then projected via a linear projection layer into the same embedding space as the language tokens.

For each action a_{t'} in the action chunk A_t, we have a corresponding *action token* that we feed through the action expert. During training, we supervise these action tokens using a conditional flow matching loss:

$$L^\tau(\theta) = \mathbb{E}_{p(A_t|o_t), q(A_t^\tau|A_t)} ||\mathbf{v}_\theta(A_t^\tau, o_t) - \mathbf{u}(A_t^\tau|A_t)||^2$$

where subscripts denote robot timesteps and superscripts denote flow matching timesteps, with tau in [0, 1]. Recent work in high-resolution image and video synthesis has shown that flow matching can achieve strong empirical performance when combined with a simple linear-Gaussian (or optimal transport) probability path, given by q(A_t^tau|A_t) = N(tau * A_t, (1-tau) * I). In practice, the network is trained by sampling random noise epsilon ~ N(0, I), computing the "noisy actions" A_t^tau = tau * A_t + (1-tau) * epsilon, and then training the network outputs v_theta(A_t^tau, o_t) to match the denoising vector field u(A_t^tau|A_t) = A_t - epsilon. The action expert uses a full bidirectional attention mask, so that all action tokens attend to each other. During training, we sample the flow matching timestep tau from a beta distribution that emphasizes lower (noisier) timesteps. See Appendix B for more details.

At inference time, we generate actions by integrating the learned vector field from tau = 0 to tau = 1, starting with random noise A_t^0 ~ N(0, I). We use the forward Euler integration rule:

$$A_t^{\tau+\delta} = A_t^\tau + \delta \mathbf{v}_\theta(A_t^\tau, o_t)$$

where delta is the integration step size. We use 10 integration steps (corresponding to delta = 0.1) in our experiments. Note that inference can be implemented efficiently by caching the attention keys and values for the prefix o_t and only recomputing the suffix corresponding to the action tokens for each integration step. We provide more details regarding the inference procedure, including the inference time for each part of the model, in Appendix D.

While in principle our model can be initialized from scratch or fine-tuned from any VLM backbone, in practice we use PaliGemma as our base model. PaliGemma is an open-source 3 billion parameter VLM that offers a convenient trade-off between size and performance. We add 300M parameters for the action expert (which is initialized from scratch) for a total of 3.3 billion parameters. We provide a full description of the model architecture in Appendix B.

### Non-VLM baseline model

In addition to our main VLA model, we also trained a similar baseline model that did not use a VLM initialization for ablation experiments. This model, which we refer to as pi_0-small, has 470M parameters, does not use VLM initialization, and has a number of small differences that we found to be helpful for training on our data without VLM initialization, which are summarized in Appendix C. This model is used in our comparisons to evaluate the benefits of incorporating VLM pertaining.

## V. Data Collection and Training Recipe

Broadly capable robot foundation models require not only an expressive and powerful architecture, but also the right dataset and, more importantly, the right training *recipe*. In the same way that LLM training is typically divided into pre-training and post-training phases, we employ a multi-stage training procedure for our model. The goal of the pre-training phase is to expose the model to a diverse range of tasks so that it can acquire broadly applicable and general physical capabilities, while the goal of the post-training phase is to provide the model with the ability to skillfully and fluently execute the desired downstream task. Because of this, the requirements for the pre-training and post-training datasets are distinct: the pre-training dataset should cover as many tasks as possible, and within each of those tasks should cover a diversity of behaviors. The post-training dataset should instead cover behaviors that are conducive to effective task execution, which should exhibit a consistent and fluent strategy. Intuitively, the diverse (but lower quality) pre-training data allows the model to recover from mistakes and handle highly varied situations, which might not otherwise occur in the high-quality post-training data, while the post-training data teaches the model to perform the task well.

### A. Pre-training and post-training

[Figure 4: Overview of our dataset: The pre-training mixture consists of a subset of OXE and the pi dataset. We use a subset of OXE, which we refer to as OXE Magic Soup. The right figure illustrates the weight of the different datasets in the pre-training mixture. The left figure illustrates their relative sizes as measured by the number of steps.]

We provide an overview of our pre-training mixture in Figure 4. Since each training example corresponds to a timestep -- i.e., a tuple (o_t, A_t), -- we will quantify data in terms of timesteps in this discussion. 9.1% of the training mixture consists of open-source datasets, including OXE, Bridge v2, and DROID. The robots and tasks in these datasets typically have one or two cameras and use low-frequency control, between 2 and 10 Hz. However, these datasets cover a wide range of objects and environments. To learn dexterous and more complex tasks, we also use 903M timesteps of data from our own datasets, where 106M steps are from single-arm robots and 797M are from dual-arm robots. This data has 68 tasks, where each task is composed of complex behaviors -- e.g., the "bussing" task involves putting a wide range of different dishes, cups, and utensils into a bussing bin, and a wide array of trash items into the garbage. Note that this definition of task is significantly different from prior work, which typically uses any combination of noun and verb (e.g., "pick up the cup" vs. "pick up the plate") to constitute a distinct task. Therefore, the actual range of behaviors in our dataset is significantly broader than this number of "tasks" would imply. We discuss the specific robots and tasks in our dataset in more detail in Section V-C.

Since the datasets are somewhat imbalanced in size (e.g., the more difficult laundry folding tasks are overrepresented), we weight each task-robot combination by n^{0.43}, where n is the number of samples for that combination, such that over-represented combinations are down-weighted. The configuration vector q_t and action vectors a_t always have the dimensionality of the largest robot in the dataset (18 in our case, to accommodate two 6-DoF arms, 2 grippers, a mobile base, and a vertically actuated torso). For robots with lower-dimensional configuration and action spaces, we zero-pad the configuration and action vectors. For robots with fewer than three images, we also mask out the missing image slots.

In the post-training phase, we fine-tune our model with a smaller task-specific dataset to specialize it to particular downstream applications. As mentioned previously, our definition of "task" is fairly broad -- e.g., the "bussing" task requires manipulating a wide range of different objects. Different tasks require very different datasets, with the simplest of the tasks necessitating only 5 hours and the most complex tasks using 100 or more hours of data.

### B. Language and high-level policies

More complex tasks that require semantic reasoning and high-level strategy, such as table bussing, can also benefit from a high-level policy that decomposes high-level tasks (such as "bus the table") into more immediate subtasks (such as "pick up the napkin" or "throw the napkin into the trash"). Since our model is trained to process language inputs, we can use a high-level VLM to make these semantic inferences, a method that is analogous to LLM/VLM planning methods such as SayCan. We use such a high-level policy to assist our model with high-level strategy for several of our experimental tasks, as we will discuss in Section VI.

### C. Robot system details

Our dexterous manipulation datasets include 7 different robot configurations and 68 tasks. We summarize these platforms in Figure 5, and discuss them below:

[Figure 5: The robots used in our experiments. These include single and dual-arm manipulators with 6-DoF and 7-DoF arms, as well as holonomic and nonholonomic mobile manipulators. pi_0 is trained jointly on all of these platforms.]

**UR5e.** An arm with a parallel jaw gripper, with a wrist-mounted and over-the-shoulder camera, for a total of two camera images and a 7-dimensional configuration and action space.

**Bimanual UR5e.** Two UR5e setups, for a total of three camera images and a 14-dimensional configuration and action space.

**Franka.** The Franka setup has two cameras and an 8-dimensional configuration and action space.

**Bimanual Trossen.** This setup has two 6-DoF Trossen ViperX arms in a configuration based on the ALOHA setup, with two wrist cameras and a base camera, and a 14-dimensional configuration and action space.

**Bimanual ARX & bimanual AgileX.** This setup uses two 6-DoF arms, and supports either ARX or AgileX arms, with three cameras (two wrist and one base) and a 14-dimensional configuration and action space. This class encompasses two distinct platforms, but we categorize them together because of their similar kinematic properties.

**Mobile Trossen & mobile ARX.** This setup is based on the Mobile ALOHA platform, with two 6-DoF arms on a mobile base, which are either ARX arms or Trossen ViperX arms. The nonholonomic base adds two action dimensions, for a 14-dimensional configuration and 16-dimensional action space. There are two wrist cameras and a base camera. This class encompasses two distinct platforms, but we categorize them together because of their similar kinematic properties.

**Mobile Fibocom.** Two 6-DoF ARX arms on a holonomic base. The base adds three action dimensions (two for translation and one for orientation), for a 14-dimensional configuration and 17-dimensional action space.

We summarize the proportion of our dataset from each robot in Figure 4.

## VI. Experimental Evaluation

Our experimental evaluation consists of out-of-box evaluation experiments that compare our base (pre-trained) model to alternative model designs with direct prompting, as well as detailed fine-tuning experiments that evaluate our model on challenging downstream tasks, comparing it to other methods that have been proposed for dexterous manipulation. We study the following research questions:

**How well does pi_0 perform after pre-training on a variety of tasks that are present in the pre-training data?** We study this question by directly evaluating pi_0, with comparisons to other robot foundation models in the literature: both VLAs and smaller models that are trained from scratch on the same pre-training mixture. We evaluate on the following tasks, visualized in Figure 6, with each task commanded to the same base model via a language command.

**Shirt folding:** the robot must fold a t-shirt, which starts flattened.
**Bussing easy:** the robot must clean a table, putting trash in the trash bin and dishes into the dish bin. The score indicates the number of objects that were placed in the correct receptacle.
**Bussing hard:** a harder version of the bussing task, with more objects and more challenging configurations.
**Grocery bagging:** the robot must bag all grocery items, such as potato chips, marshmallows, and cat food.
**Toast out of toaster:** the robot removes toast from a toaster.

[Figure 6: Out-of-box evaluation tasks.]

We compare to OpenVLA, a 7B parameter VLA model that was originally trained on the OXE dataset. We train OpenVLA on our full mixture. We also compare to Octo, a smaller 93M parameter model. While Octo is not a VLA, it does use a diffusion process to generate actions, providing a valuable point of comparison for our flow matching VLA. We also train Octo on the same mixture as our model. We therefore also compare to a "compute parity" version of our model, which is trained for only 160k steps (as opposed to 700k steps for our main model). We also include a comparison to the pi_0-small model described in Section IV.

[Figure 7: Out-of-box evaluation results: pi_0 trained for the full 700k steps, a version trained for 160k steps, pi_0-small, and three baselines: OpenVLA and Octo trained on all of our data, and OpenVLA trained only on the UR5e data. Across all tasks and all comparisons, even the "parity" version of pi_0 outperforms all the baselines, and the full version achieves the best results by a large margin.]

The results, shown in Figure 7, show that pi_0 attains by far the best results across the board on all the out-of-box tasks, with near perfect success rates on shirt folding and the easier bussing tasks, and large improvements over all baselines. The "parity" version of pi_0, which is trained for only 160k steps, still outperforms all the baselines, and even pi_0-small outperforms OpenVLA and Octo. OpenVLA struggles on these tasks because its autoregressive discretization architecture does not support action chunks.

### A. Evaluating the base model

[Details of out-of-box evaluation comparing pi_0 to OpenVLA, Octo, and pi_0-small across shirt folding, bussing, grocery bagging, and toast tasks.]

### B. Following language commands

In the next set of experiments, we fine-tune the base pi_0 model to follow language commands in a set of evaluation domains. We compare this fine-tuned pi_0 model with the pi_0-small model. We evaluate with both human-provided commands (pi_0-human), flat commands (pi_0-flat), and with high-level model guidance (pi_0-HL).

[Figure 8: The tasks in our language evaluation: bussing, table setting, and grocery bagging.]

The results in Figure 9, averaging over 10 trials per task, show that the language following accuracy of pi_0 is significantly better than that of pi_0-small. This suggests a significant improvement from the larger pre-trained VLM initialization. This capability translates to an improvement in performance with expert human guidance (pi_0-human) and with high-level model guidance (pi_0-HL).

[Figure 9: Language evaluation. pi_0 significantly outperforms pi_0-small in language following and task performance.]

### C. Learning new dexterous tasks

In the next set of experiments, we evaluate our model on new tasks that differ significantly from the pre-training data, requiring entirely new behaviors. We fine-tune the model using various amounts of data for each new task. The tasks, shown in Figure 10, are:

**UR5e stack bowls.** Stacking bowls, with four bowls of different sizes.
**Towel folding.** Folding a towel.
**Tupperware in microwave.** Opening a microwave, putting a plastic container inside, and closing it.
**Paper towel replacement.** Removing an old paper towel tube and replacing it with a fresh roll.
**Franka items in drawer.** Opening a drawer, packing items into it, and closing it.

[Figure 10: Fine-tuning evaluation tasks.]

We compare to OpenVLA, Octo, ACT, and Diffusion Policy. Figure 11 shows the performance across all tasks, averaging over 10 trials per task, with different amounts of fine-tuning data on each task. The results show that pi_0 generally outperforms other methods.

[Figure 11: Fine-tuning with varying amounts of data. pi_0 can learn some easier tasks even with smaller amounts of data, and the pre-trained model often attains a larger improvement over the model trained from scratch.]

### D. Mastering complex multi-stage tasks

In our final set of experiments, we tackle a range of challenging multi-stage tasks via a combination of fine-tuning and language. The tasks in this evaluation, shown in Figure 12, are:

**Laundry folding:** Folding articles of clothing from a bin using a static bimanual system.
**Mobile laundry:** Folding laundry on a mobile robot.
**Dryer unloading:** Taking laundry from a dryer and placing it into a hamper.
**Table bussing:** Bussing a table with diverse novel objects in a clutter scene.
**Box building:** Assembling a cardboard box from a flattened state.
**To-go box:** Moving food items from a plate into a to-go box and closing it.
**Packing eggs:** Taking six eggs from a bowl and packing them into an egg carton.

[Figure 12: We evaluate a range of complex and temporally extended tasks including folding laundry, bussing a table, assembling a box, packing eggs, and packing food into a to-go box.]

The results are presented in Figure 13. The full pre-trained pi_0 model attains more than 50% of the maximum score across all of the tasks, and typically outperforms the ablations, with especially significant improvements on the hardest tasks.

[Figure 13: Post-training results on complex tasks in terms of average scores over 10 trials.]

## VII. Discussion, Limitations, and Future Work

We presented a framework for training a robot foundation model, which we refer to as pi_0, that consists of pre-training on highly diverse data, followed by either out-of-box evaluation or fine-tuning to complex downstream tasks. Our empirical evaluation studies tasks that combine dexterity, generalization, and temporally extended multi-stage behaviors. Our model incorporates Internet-scale vision-language model (VLM) pre-training with flow matching for representing complex high-frequency action chunks. Our pre-training mixture consists of 10,000 hours of dexterous manipulation data from 7 different robot configurations and 68 tasks, in addition to large amounts of previously collected robot manipulation data from OXE, DROID, and Bridge. To our knowledge, this represents the largest pre-training mixture ever used for a robot manipulation model. Our fine-tuning experiments include over 20 tasks, where we show that our model outperforms a variety of baselines, including prior VLA models and models designed specifically for dexterous manipulation. We also examine how our post-training recipe can enable highly complex tasks, such as folding multiple articles of clothing from arbitrary initial configurations or assembling boxes.

Our framework broadly resembles the training procedures employed for large language models, which typically consist of pre-training a base model on very large datasets scraped from the web, followed by a post-training procedure that aims to "align" the model to enable it to follow instructions and perform user commands. It is generally recognized that most of the "knowledge" in such models is acquired in the pre-training phase, while the post-training phase serves to tell the model how it should leverage that knowledge to fulfill user commands. Our experiments imply that an analogous phenomenon might take place with robot foundation models, where pre-trained models have some zero-shot capabilities, but complex tasks like laundry following require fine-tuning with high-quality data. Training on only this high-quality data results in a brittle model that does not reliably recover from mistakes, while running the pre-trained model in zero shot does not always exhibit the fluent strategies demonstrated in the post-training data.

We hope that our results will serve as a stepping stone toward general and broadly applicable robot foundation models. Our experiments suggest that such models may soon be a reality, but there are a number of limitations and ample room for future work. First, our experiments do not yet provide a comprehensive understanding of how the pre-training datasets should be composed: we combined all data available to us, but understanding what type of data is more helpful to add and how it should be weighted remains an open problem. Not all tasks in our evaluation work reliably, and it remains unclear how to predict how much and what kind of data is needed to attain near-perfect performance. Finally, it remains to be seen how much positive transfer there is in combining highly diverse data, particularly from different tasks and different robots: although our results suggest that universal pre-trained robot foundation models might become a reality, it is left for future work to understand whether this universality extends to much more distinct domains, such as autonomous driving, navigation, and legged locomotion.

## Appendix B. Model Architecture Details

The pi_0 model follows the PaliGemma VLM design, with the following differences: (1) additional input and output projections for the robotics-specific tokens, including the state vector q_t and action vectors A_t = [a_t, ..., a_{t+H-1}], (2) an additional MLP for incorporating the flow matching timestep information tau, and (3) a second, smaller set of weights for the action expert.

**Additional inputs and outputs.** The standard PaliGemma architecture takes in a sequence of images [I_t^1, ..., I_t^n] followed by a language prompt l_t. We add an input q_t for the robot's proprioceptive state, which is mapped to the transformer embedding dimension using a linear projection. The final set of input tokens correspond to the noisy action chunk A_t^tau = [a_t^tau, ..., a_{t+H-1}^tau], with the number of tokens equal to the action horizon (H = 50 for our tasks). We only use the transformer outputs corresponding to the H noisy actions, which are decoded into v_theta(A_t^tau, o_t) using a linear projection.

**Incorporating the flow matching timestep.** The noisy action chunk A_t^tau is mapped to the transformer's embedding dimension using an MLP that also incorporates the flow matching timestep tau. For each noisy action a_{t'}^tau, the expression for the corresponding embedding that is fed into the transformer is W_3 * swish(W_2 * concat(W_1 * a_{t'}^tau, phi(tau))), where phi: R -> R^w is a sinusoidal positional encoding function, W_1 in R^{w x d}, W_2 in R^{w x 2w}, W_3 in R^{w x w}, d is the action dimension, and w is the embedding dimension (or *width*) of the action expert.

**Attention mask.** pi_0 uses a blockwise causal attention mask with 3 blocks: [I_t^1, ..., I_t^n, l_t], [q_t], and [a_t^tau, ..., a_{t+H-1}^tau]. Within each block, there is full bidirectional attention, whereas each block cannot attend to the tokens in future blocks. The first block includes the input modalities from PaliGemma's VLM pre-training, which are prevented from attending to future blocks (which include new inputs) to minimize distribution shift from said pre-training. The robot state q_t is its own block because it does not change with each flow matching integration step; preventing it from attending to the final block allows its corresponding keys and values to be cached during sampling. The final block corresponds to the noisy actions A_t^tau, which can attend to the full input sequence.

**Action expert.** pi_0 is implemented as a single transformer with two sets of weights (also known as experts), where each token is routed to one of the experts; the weights interact only through the transformer's self-attention layers. The images and language prompt, [I_t^1, ..., I_t^n, l_t], are routed to the larger VLM backbone, which we initialize from PaliGemma. The inputs not seen during VLM pre-training, [q_t, A_t^tau], are routed to the action expert. PaliGemma is based on the Gemma 2B language model, which uses multi-query attention and a configuration of {width=2048, depth=18, mlp_dim=16,384, num_heads=18, num_kv_heads=1, head_dim=256}. Since the experts interact only in the self-attention layers, width and mlp_dim do not necessarily need to match between experts. To speed up inference (which requires multiple forward passes of the action expert), we downsize the action expert to {width=1024, mlp_dim=4096}, resulting in a parameter count of ~300M.

**Sampling the flow matching timestep.** We designed a timestep sampling distribution that emphasizes low timesteps (high noise levels); additionally, timesteps above a given threshold s are not sampled at all, since they are not needed so long as the integration step delta is greater than 1-s. The distribution is given by p(tau) = Beta((s-tau)/s; 1.5, 1) and is visualized in Figure 14. We use s = 0.999 in our experiments, which allows for delta > 1/1000, or up to 1,000 integration steps.

[Figure 14: Flow matching timestep sampling distribution. We sample tau from a shifted beta distribution that emphasizes lower timesteps (corresponding to noisier actions), and does not sample timesteps at all above a cutoff value s. We use s = 0.999.]

## Appendix D. Inference

Recall that our model takes an observation o_t = [I_t^1, ..., I_t^n, l_t, q_t] and the noisy actions A_t^tau and outputs the vector field that needs to be integrated to obtain the next flow matching step, v_t^tau. Each time we predict a new action chunk A_t, we must encode each of the images I_t^1, ..., I_t^n, run a forward pass on the tokens corresponding to o_t, and then run 10 steps of flow matching, where each step requires running a forward pass on the tokens corresponding to A_t^tau (the keys and values corresponding to o_t are cached). Table I summarizes the computation time for this operation with 3 camera images. The operations were timed on an NVIDIA GeForce RTX 4090 consumer-grade GPU. For the mobile robot, inference was done off-board over a Wi-Fi connection, adding a small amount of network latency.

Since the model generates an entire H-step action chunk at once, we can execute up to H actions before we need to run inference again. However, we may run inference more often than that, as well as combine actions from different inference calls using various aggregation strategies. We opted not to aggregate actions and instead execute action chunks open-loop. For the 20Hz UR5e and Franka robots, we run inference every 0.8 seconds (after executing 16 actions), and for all other robots, which run at 50Hz, we run inference every 0.5 seconds (after executing 25 actions).

| Model part | Inference time |
|-----------|---------------|
| Image encoders | 14 ms |
| Observation forward pass | 32 ms |
| x10 action forward pass (flow) | 27 ms |
| Network latency (if off-board) | 13 ms |
| Total on-board inference | 73 ms |
| Total off-board inference | 86 ms |

Table I: Inference time of our model on an NVIDIA GeForce RTX 4090 GPU.

## References

- [1] J. Achiam et al. Gpt-4 technical report. arXiv:2303.08774, 2023.
- [2] M. Ahn et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv:2204.01691, 2022.
- [3] J.-B. Alayrac et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 35, 2022.
- [4] J. Aldaco et al. Aloha 2: An enhanced low-cost hardware for bimanual teleoperation. arXiv:2405.02292, 2024.
- [5] L. Beyer et al. PaliGemma: A versatile 3b VLM for transfer. arXiv:2407.07726, 2024.
- [6] H. Bharadhwaj et al. RoboAgent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. ICRA, 2024.
- [7] A. Brohan et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818, 2023.
- [8] S. Cabi et al. Scaling data-driven robotics with reward sketching and batch reinforcement learning. arXiv:1909.12200, 2019.
- [9] C. Chi et al. Diffusion policy: Visuomotor policy learning via action diffusion. IJRR, 2023.
- [10] OX-Embodiment Collaboration et al. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv:2310.08864, 2023.
- [11] D. Driess et al. PaLM-E: An embodied multimodal language model. arXiv:2303.03378, 2023.
- [12] N. Du et al. GLaM: Efficient scaling of language models with mixture-of-experts. ICML, 2022.
- [13] F. Ebert et al. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv:2109.13396, 2021.
- [14] P. Esser et al. Scaling rectified flow transformers for high-resolution image synthesis. ICML, 2024.
- [15] H. Etukuru et al. Robot utility models: General policies for zero-shot deployment in new environments. arXiv:2409.05865, 2024.
- [16] W. Fedus et al. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR, 23, 2022.
- [17] Z. Fu et al. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. CoRL, 2024.
- [18] A. Gupta et al. Robot learning in homes: Improving generalization and reducing dataset bias. NeurIPS, 31, 2018.
- [19] W. He et al. Mars: Mixture of auto-regressive models for fine-grained text-to-image synthesis. arXiv:2407.07614, 2024.
- [20] J. Ho et al. Denoising diffusion probabilistic models. NeurIPS, 33, 2020.
- [21] J. Jumper et al. Highly accurate protein structure prediction with alphafold. Nature, 596, 2021.
- [22] D. Kalashnikov et al. Scalable deep reinforcement learning for vision-based robotic manipulation. CoRL, 2018.
- [23] A. Khazatsky et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv:2403.12945, 2024.
- [24] M. J. Kim et al. OpenVLA: An open-source vision-language-action model. arXiv:2406.09246, 2024.
- [25] D. Lepikhin et al. GShard: Scaling giant models with conditional computation and automatic sharding. arXiv:2006.16668, 2020.
- [26] S. Levine et al. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. IJRR, 37, 2018.
- [27] Y. Li et al. Competition-level code generation with alphacode. Science, 378, 2022.
- [28] Y. Lipman et al. Flow matching for generative modeling. arXiv:2210.02747, 2022.
- [29] B. Liu et al. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv:2409.10695, 2024.
- [30] H. Liu et al. Visual instruction tuning. NeurIPS, 36, 2024.
- [31] P. Liu et al. OK-robot: What really matters in integrating open-knowledge models for robotics. arXiv:2401.12202, 2024.
- [32] Q. Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv:2209.14577, 2022.
- [33] A. Mandlekar et al. RoboTurk: A crowdsourcing platform for robotic skill learning through imitation. CoRL, 2018.
- [34] A. Mandlekar et al. MimicGen: A data generation system for scalable robot learning using human demonstrations. arXiv:2310.17596, 2023.
- [35] L. Ouyang et al. Training language models to follow instructions with human feedback. NeurIPS, 35, 2022.
- [36] W. Peebles and S. Xie. Scalable diffusion models with transformers. CVPR, 2023.
- [37] L. Pinto and A. Gupta. Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours. ICRA, 2016.
- [38] A. Polyak et al. Movie gen: A cast of media foundation models. arXiv:2410.13720, 2024.
- [39] A. Radford et al. Learning transferable visual models from natural language supervision. ICML, 2021.
- [40] R. Rombach et al. High-resolution image synthesis with latent diffusion models. CVPR, 2022.
- [41] C. Saharia et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35, 2022.
- [42] V. Sanh. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv:1910.01108, 2019.
- [43] N. M. M. Shafiullah et al. On bringing robots home. arXiv:2311.16098, 2023.
- [44] N. Shazeer. Fast transformer decoding: One write-head is all you need. arXiv:1911.02150, 2019.
- [45] N. Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv:1701.06538, 2017.
- [46] J. Sohl-Dickstein et al. Deep unsupervised learning using nonequilibrium thermodynamics. ICML, 2015.
- [47] A. Steiner et al. How to train your ViT? Data, augmentation, and regularization in vision transformers. arXiv:2106.10270, 2021.
- [48] Gemini Team et al. Gemini: A family of highly capable multimodal models. arXiv:2312.11805, 2023.
- [49] Gemma Team et al. Gemma: Open models based on gemini research and technology. arXiv:2403.08295, 2024.
- [50] Octo Model Team et al. Octo: An open-source generalist robot policy. arXiv:2405.12213, 2024.
- [51] A. Vaswani et al. Attention is all you need. NeurIPS, 30, 2017.
- [52] H. R. Walke et al. BridgeData v2: A dataset for robot learning at scale. PMLR, 2023.
- [53] J. Wei et al. Finetuned language models are zero-shot learners. arXiv:2109.01652, 2021.
- [54] J. Wei et al. Emergent abilities of large language models. arXiv:2206.07682, 2022.
- [55] J. Wen et al. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation. arXiv:2409.12514, 2024.
- [56] K.-T. Yu et al. More than a million ways to be pushed. IROS, 2016.
- [57] T. Z. Zhao et al. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv:2304.13705, 2023.
- [58] T. Z. Zhao et al. Aloha unleashed: A simple recipe for robot dexterity. arXiv:2410.13126, 2024.
- [59] C. Zhou et al. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv:2408.11039, 2024.
- [60] M. Zhu et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. arXiv:2409.14411, 2024.
