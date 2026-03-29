HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: graphbox

- failed: stackengine

- failed: inconsolata

- failed: regexpatch

- failed: icomma

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: arXiv.org perpetual non-exclusive license

arXiv:2312.13469v1 [cs.RO] 20 Dec 2023

[# Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation Sudharshan Suresh,${}^{1,2\ast}$ Haozhi Qi,${}^{2,3}$ Tingfan Wu,${}^{2}$ Taosha Fan,${}^{2}$ Luis Pineda,${}^{2}$ Mike Lambeta,${}^{2}$ Jitendra Malik,${}^{2,3}$ Mrinal Kalakrishnan,${}^{2}$ Roberto Calandra,${}^{4,5}$ Michael Kaess,${}^{1}$ Joseph Ortiz,${}^{2}$ Mustafa Mukadam${}^{2}$ ${}^{1}$\hrefhttps://www.ri.cmu.edu/CMU, ${}^{2}$\hrefhttps://ai.meta.com/research/FAIR, ${}^{3}$\hrefhttps://www.berkeley.edu/UC Berkeley, ${}^{4}$\hrefhttps://tu-dresden.de/?set_language=enTU Dresden, ${}^{5}$\hrefhttps://ceti.one/CeTI ${}^{\ast}$To whom correspondence should be addressed; E-mail: \hrefmailto:suddhus@gmail.comsuddhus@gmail.com To achieve human-level dexterity, robots must infer spatial awareness from multimodal sensing to reason over contact interactions. During in-hand manipulation of novel objects, such spatial awareness involves estimating the object’s pose and shape. The status quo for in-hand perception primarily employs vision, and restricts to tracking a priori known objects. Moreover, visual occlusion of objects in-hand is imminent during manipulation, preventing current systems to push beyond tasks without occlusion. We combine vision and touch sensing on a multi-fingered hand to estimate an object’s pose and shape during in-hand manipulation. Our method, NeuralFeels encodes object geometry by learning a neural field online and jointly tracks it by optimizing a pose graph problem. We study multimodal in-hand perception in simulation and the real-world, interacting with different objects via a proprioception-driven policy. Our experiments show final reconstruction F-scores of $\mathbf{81}$% and average pose drifts of $\mathbf{4.7}\,\text{mm}$, further reduced to $\mathbf{2.3}\,\text{mm}$ with known CAD models. Additionally, we observe that under heavy visual occlusion we can achieve up to $\mathbf{94}$% improvements in tracking compared to vision-only methods. Our results demonstrate that touch, at the very least, refines and, at the very best, disambiguates visual estimates during in-hand manipulation. We release our evaluation dataset of 70 experiments, FeelSight, as a step towards benchmarking in this domain. Our neural representation driven by multimodal sensing can serve as a perception backbone towards advancing robot dexterity. Videos can be found on our project website: https://suddhu.github.io/neural-feels](https://suddhu.github.io/neural-feels).

## Summary

Neural perception with vision and touch yields robust tracking and reconstruction of novel objects for in-hand manipulation.

![Figure](extracted/5308748/images/cover.jpg)

*Figure 1: Visuo-tactile perception with NeuralFeels. Our method estimates pose and shape of novel objects (right) during in-hand manipulation, by learning neural field models online from a stream of vision, touch, and proprioception (left).*

## 1 Introduction

To perceive deeply is to have sensed fully. Humans effortlessly combine their senses for everyday interactions—we can rummage through our pockets in search of our keys, and deftly insert them to unlock our front door. Currently, robots lack the cognition to replicate even a fraction of the mundane tasks we perform, a trend summarized by Moravec’s Paradox [[[48](#bib.bib48)]]. For dexterity in unstructured environments, a robot must first understand its spatial relationship with respect to the manipuland. Indeed, as robots move out of instrumented labs and factories to cohabit our spaces, there is a need for generalizable spatial AI [[[12](#bib.bib12)]].

Specific to in-hand dexterity, knowledge of object pose and geometry is crucial to policy generalization [[[51](#bib.bib51), [50](#bib.bib50), [24](#bib.bib24), [57](#bib.bib57)]]. As opposed to end-to-end supervision [[[89](#bib.bib89), [23](#bib.bib23), [10](#bib.bib10)]], these methods require a persistent 3D representation of the object. However, the status quo for in-hand perception is currently restricted to the narrow scope of tracking known objects with vision as the dominant modality [[[24](#bib.bib24)]]. Further, it is common for practitioners to sidestep the perception problem entirely, retrofitting objects and environments with fiducials [[[51](#bib.bib51), [50](#bib.bib50)]]. To further progress towards general dexterity, it is clear that one of the missing pieces is general, robust perception.

With visual sensing, researchers tend to tolerate interaction rather than embrace it. This is at odds with contact-rich problems where self-occlusions is imminent, like rotating [[[56](#bib.bib56)]], re-orienting [[[24](#bib.bib24), [10](#bib.bib10)]], and sliding [[[63](#bib.bib63), [72](#bib.bib72)]]. Additionally, vision often fails in the real-world due to poor illumination, limited range, transparency, and specularity. Touch provides a direct window into these dynamic interactions, and human cognitive studies have reinforced the complementarity with vision [[[26](#bib.bib26)]].

Hardware advances have led to affordable vision-based touch sensors [[[93](#bib.bib93), [16](#bib.bib16), [81](#bib.bib81), [1](#bib.bib1), [40](#bib.bib40), [53](#bib.bib53), [79](#bib.bib79)]] like the GelSight and DIGIT. Progress in touch simulation [[[78](#bib.bib78)]] enables practitioners to learn tactile observation models that transfer to real-world interactions [[[79](#bib.bib79), [67](#bib.bib67), [73](#bib.bib73)]]. With a fingertip form-factor, their illuminated gel deforms on contact and the physical interaction is captured by an internal camera. When chained with robot kinematics, we obtain dense, situated contact that can be processed similar to natural camera images.

Now given multimodal sensing, how best to represent the spatial information? Coordinate-based learning, formalized as neural fields [[[87](#bib.bib87)]], has found great success in visual computing. With neural fields, practitioners can create high-quality 3D assets offline given noisy visual data and pose annotation [[[46](#bib.bib46), [49](#bib.bib49), [43](#bib.bib43)]]. They are continuous representations with higher fidelity than their discrete counterparts like point clouds and meshes. While they are specialized towards batch optimization, lightweight SDF models [[[52](#bib.bib52), [69](#bib.bib69), [97](#bib.bib97), [82](#bib.bib82)]] have made online perception possible.

Researchers have used this extensible architecture not just for continuous 3D quantities like signed distance fields (SDFs) and radiance [[[54](#bib.bib54), [46](#bib.bib46), [49](#bib.bib49)]], but also for pose estimation [[[88](#bib.bib88), [82](#bib.bib82)]], planning [[[22](#bib.bib22)]], and latent physics [[[41](#bib.bib41)]]. Moreover, the ease of imparting generative priors [[[90](#bib.bib90)]] and initializing with pre-trained models [[[54](#bib.bib54)]] future-proofs them. While neural fields have emerged little by little in robot manipulation [[[96](#bib.bib96), [32](#bib.bib32), [83](#bib.bib83), [22](#bib.bib22)]], the optimization of multimodal data remains an open question.

The domain of our work—an intersection of simultaneous localization and mapping (SLAM) and manipulation—has been studied for over two decades. A first exemplar is from Moll and Erdmann [[[47](#bib.bib47)]], who reconstruct the shape and motion of an object rolled between robot palms, later reproduced with specialized sensors [[[68](#bib.bib68), [42](#bib.bib42)]]. Tactile SLAM has been thoroughly investigated for planar pushing due to its well-understood mechanics [[[91](#bib.bib91), [71](#bib.bib71)]]. The combination of vision and touch has been explored for reconstructing fixed objects [[[80](#bib.bib80), [64](#bib.bib64), [73](#bib.bib73), [11](#bib.bib11)]] and tracking known objects [[[92](#bib.bib92), [39](#bib.bib39), [66](#bib.bib66)]]. Closest to our work is FingerSLAM [[[95](#bib.bib95)]], combining dense touch from a single finger with vision, however we consider the more challenging case of in-hand manipulation.

NeuralFeels presents an online solution to localize and reconstruct objects for in-hand manipulation with multimodal sensing. We unify vision, touch, and proprioception into a neural representation and demonstrate SLAM for apriori unknown objects, and robust tracking of known objects. In our experiments, we present our robot with a novel object, and it infers and tracks its geometry through just interaction. We use a dexterous hand [[[84](#bib.bib84)]] sensorized with commercial vision-based touch sensors [[[40](#bib.bib40)]] and a fixed RGB-D camera (Figure [1](#Sx1.F1)). With a proprioception-driven policy [[[56](#bib.bib56)]] we explore the object’s extents through in-hand rotation.

Through our experiments we study the role that vision and touch play in interactive perception, the effects of occlusion, and visual sensing noise. To evaluate our work, we collect a dataset of $70$ in-hand rotation trials in both the real-world and simulation, with ground-truth object meshes and tracking. Our results on novel objects show average reconstruction F-scores of $81\%$ with pose drifts of just $4.7\,\text{mm}$, further reduced to $2.3\,\text{mm}$ with known CAD models. Under heavy occlusion, we demonstrate up to $94$% improvements in pose tracking compared to vision-only methods. Our combination of rich sensing and spatial AI requires minimal hardware compared to complex sensing cages, and is easier to interpret than end-to-end perception methods. The output of the neural SLAM pipeline—pose and geometry—can drive further research in general dexterity, broadening the capabilities of home robots.

*Figure 2: A visuo-tactile perception stack amidst interaction. An online representation of object shape and pose is built from vision, touch, and proprioception during in-hand manipulation. Raw sensor data is first fed into the frontend, which extracts visuo-tactile depth with our pre-trained models. Following this, the backend samples from the depth to train a neural signed distance field (SDF), while the pose graph tracks the posed neural field.*

## 2 Results

Our multi-fingered robot hand is presented with a novel object, placed randomly between its fingertips. It rotates the object in-hand, through a proprioception-driven policy [[[56](#bib.bib56)]], which gives rise to a stream of visual and tactile signals. We combine the visual, tactile, and proprioceptive sensing into our online neural field, for a persistent, evolving 3D representation of the unknown object. The full pipeline of our NeuralFeels perception stack is illustrated in Figure [2](#S1.F2). We also summarize our experiments and findings in [our webpage](https://suddhu.github.io/neural-feels).

We evaluate NeuralFeels over simulated and real-world interactions, totaling up to $70$ experiments over different object classes. Details of the dataset can be found in Section [4.3](#S4.SS3). First, we demonstrate SLAM results for novel objects, and highlight some qualitative examples. Next, we demonstrate pose-tracking when we have a priori shape of the manipuland. Finally, we analyze the role touch plays in improving perception under occlusion and visual sensing noise.

### 2.1 Metrics and baseline

Pose and reconstruction metrics. We use the symmetric average Euclidean distance metric (ADD-S) to evaluate the pose tracking error over time [[[77](#bib.bib77)]]. The ADD metric is commonly used in manipulation [[[86](#bib.bib86), [6](#bib.bib6), [76](#bib.bib76), [77](#bib.bib77)]] as a geometrically-interpretable distance metric for pose error. It is computed by sub-sampling the ground-truth object mesh and averaging the Euclidean distance between the point-set in the estimated and ground-truth object pose frames. Rather than pairwise distance, ADD-S considers the closest point distance, which disambiguates symmetric objects.

For reconstruction, we compare how accurate (precision) and complete (recall) the neural SDF is in comparison to the ground-truth mesh. The F-score, an established metric in the multi-view reconstruction community [[[37](#bib.bib37), [75](#bib.bib75)]], combines these two criteria into an interpretable ${\small[0\!-\!1]}$ value. To compute this, we first sub-sample the ground-truth and reconstructed meshes, and transform both to the common object-centric reference frame. Given a distance threshold, in our case $\tau\!=\!5\,\text{mm}$, precision measures the percentage of reconstructed points within $\tau$ distance from the ground-truth points. Conversely, recall measures the percentage of ground-truth points within $\tau$ distance from the reconstructed points. The harmonic mean of these two quantities give us the F-score, which captures both surface reconstruction accuracy and shape completion. Broadly, a higher F-score with tighter $\tau$ bounds implies better object reconstructions. For brevity, we refer to ADD-S and F-score as the pose metric and shape metric respectively.

Ground-truth shape and pose. We evaluate these metrics against the ground-truth estimates of object shape and pose. For each object, the ground-truth shape is obtained from offline scans (Figure [S1](#Sx3.F1)). Ground-truth object pose is straightforward in simulation experiments, directly exposed by IsaacGym [[[45](#bib.bib45)]]. In the real-world, we estimate a pseudo ground-truth, via multi-camera pose tracking of the experiment. Instrumented solutions, such as 3D motion capture, are infeasible as it both visually and physically interferes with the experiments. We opt to install two additional cameras (Section [4.3](#S4.SS3)) and run NeuralFeels in pose tracking mode with the ground-truth object shape. This represents the best tracking estimates given known shape and occlusion-free vision. For further details, refer to Section [S1](#Sx3.SS1).

### 2.2 Neural SLAM: object pose and shape estimation

In this section, we evaluate NeuralFeels’ ability for embodied spatial reasoning from scratch. We present the robot with a novel object, and the robot is tasked with building an object model on-the-fly. This is typical where robots continually learn from interaction, such as when deployed in unstructured household environments. We make no assumptions about the object geometry, which is built from scratch, or manipulation actions, which are decided at deployment. We process visuo-tactile data sequentially with no access to future information or category-level priors. This formulation aligns with other dexterous manipulation work [[[24](#bib.bib24), [56](#bib.bib56), [57](#bib.bib57), [10](#bib.bib10)]], and is less restrictive than that of FingerSLAM [[[95](#bib.bib95)]], where the object is always in contact with a single tactile sensor and the camera is unobstructed.

We evaluate over a combined $70$ experiments in simulation and real-world across of $14$ different objects. The objects are placed in-hand, after which the policy collects $30$ seconds of vision, touch, and proprioception data. As each run is non-deterministic, we average our results across $5$ different seeds, resulting in a total of $350$ trials. The first frame of each sequence only presents limited visual knowledge: a single side of Rubik’s cube or large dice; the underside of the rubber duck. Through the course of any $30$ second sequence, in-hand rotation exposes previously unseen geometries to vision and touch fills in the rest of the occluded surfaces. In Figure [3](#S2.F3), we present the main set of results, where we compare the multimodal fusion schemes against ground-truth.

Object reconstructions. Figure [3](#S2.F3) (a) shows the final shape metric at the end of each sequence for a fixed threshold $\tau$. Here we pick $\tau\!=\!5\,\text{mm}$ for this evaluation, around $3$% of the maximum diagonal length of the objects. Greater the value of the shape metric, the closer the surface reconstructions are to ground-truth. We observe large gains when incorporating touch, with surface reconstructions on average $15.3$% better in simulation and $14.6$% better in the real-world. Our final reconstructions, as seen in Figure [3](#S2.F3) (e), have a median error of $2.1\text{mm}$ in simulation and $3.9\text{mm}$ in the real-world. Additionally, the second plot compares the final shape metrics against a range of $\tau$ thresholds. Here we observe that multimodal fusion leads to consistently better shape metrics across all $\tau$ values in simulation and the real-world.

![Figure](extracted/5308748/images/aggregate_stats.jpg)

*Figure 3: Summary of SLAM experiments. (a, b) We present aggregated statistics for SLAM over a combined 70 experiments (40 in simulation and 30 in the real-world), with each trial run over 5 different seeds. We compare across simulation and real-world to show low pose drift and high reconstruction accuracy. (c) Table 1 illustrates the number of trials that our method fails to track (and reconstruct) the object. (d) Representative examples of the final object pose and neural field renderings from the experiments. (e) The final 3D objects generated by marching cubes on our neural field. Here, we highlight the role tactile plays in both shape completion and shape refinement.*

Object pose drift. As SLAM is the exemplar of a chicken and egg problem, there is a strong correlation between a low shape metric and high pose metric. Empirically, we observe larger pose drift in the initial few seconds due to incomplete geometry, which levels off with further exploration. For fair comparisons we initialize the object’s canonical pose to the ground-truth, but this is not necessary otherwise. With this initialization, we ignore the pose metric over the first five seconds, as it is ill-defined.

Figure [3](#S2.F3) (b) plots the drift of the object’s estimated pose with respect to the ground-truth, lower being more accurate. We observe better tracking with respect to the vision-only baseline, with improvements of $21.3$% in simulation and $26.6$% in the real-world. Table 1 in Figure [3](#S2.F3) (c) reports the number of failures in vision-only tracking compared to NeuralFeels. Here, a failed experiment is defined as when the average pose drift exceeds an empirical threshold of $10\,\text{mm}$.

Qualitative results. Figure [3](#S2.F3) (d) visualizes the rendered normals of the posed neural field at the end of each experiment, with the 3D coordinate axes superimposed. The final 3D reconstructions, generated via marching cubes, are shown in Figure [3](#S2.F3) (e) alongside the ground-truth meshes. Below that, we highlight the gains with visuo-tactile integration, with examples of shape completion and refinements.

In Figure [4](#S2.F4) we show the incremental pose tracking and reconstructions of objects across different time slices of a few representative experiments. We present two results from the real-world, bell pepper and large dice, and two from simulation, rubber duck and peach. At each timestep, we highlight the input stream, frontend depth and output object model. The 3D visualizations are generated by marching-cubes, in addition to the rendered normals of the neural field projected onto the visual image. In each case, we partially reconstruct the object at the initial frame, and build the surfaces out progressively over time.

![Figure](extracted/5308748/images/representative_slam.jpg)

*Figure 4: Representative SLAM results. In both real-world and simulation, we build an evolving neural SDF that integrates vision and touch while simultaneously tracking the object. We illustrate the input stream of RGB-D and tactile images, paired with the posed reconstruction at that timestep.*

### 2.3 Neural tracking: object pose estimation given shape

As a special case of NeuralFeels, we demonstrate superior multimodal pose tracking when provided the CAD models of the objects at runtime. Tracking known geometries is an active area of research in visual SLAM [[[24](#bib.bib24), [38](#bib.bib38)]], with some work that incorporates touch as well [[[92](#bib.bib92), [39](#bib.bib39), [66](#bib.bib66), [72](#bib.bib72), [5](#bib.bib5)]]. This is applicable in environments like warehouses and manufacturing lines, where robots have intimate knowledge of the manipulands [[[5](#bib.bib5)]]. It is further useful in household scenarios, where the robot has already generated an object model through interaction.

In implementation, the object’s SDF is pre-computed from a given CAD model. During runtime, we freeze the weights of the neural field, and only perform visuo-tactile tracking with the frontend estimates. Similar to the SLAM experiments, we run each of the $70$ experiments over $5$ seeds, and report the pose metrics with respect to ground-truth.

Results from pose tracking. Figure [5](#S2.F5) (a) shows some qualitative examples of tracking the pose of the Rubik’s cube and potted meat can with vision and touch. For the given examples, the pose metric over the sequences are plotted in Figure [5](#S2.F5) (b). We observe low, bounded pose error even with imprecise visual segmentation and sparse touch signals. In Figure [5](#S2.F5) (c) we observe the role touch plays in reducing the average pose error over all experiments to the range of $2.3\,\text{mm}$. Given the CAD model, we observe that incorporating touch can refine our pose estimates, with a decrease in average pose error by $22.29$% in simulation and $3.9$% in the real-world. As addresed in Section [3](#S3), the less-pronounced contacts in the real-world can explain this disparity. In the following section, we highlight greater improvements with respect to the baseline when visual sensing is suboptimal.

![Figure](extracted/5308748/images/representative_pose.jpg)

*Figure 5: Neural pose tracking of known objects. (a) With known ground-truth shape, we can robustly track objects such as the Rubik’s cube and potted meat can. (b) We observe reliable tracking performance, with average pose errors of $2\,\text{mm}$ through the sequence. (c) With a known object model and good visibility, touch plays the role of pose refinement.*

![Figure](extracted/5308748/images/noise_ablation.jpg)

*Figure 6: Ablations on occlusions and sensing noise. (a) With occluded viewpoints, visuo-tactile fusion helps improve tracking performance with an unobstructed local perspective. We quantify these gains across a sphere of camera viewpoint to show improvements, particularly in occlusion-heavy points-of-view. (b) We observe that touch plays a larger role when vision is heavily occluded, and a refinement role when we there is negligible occlusion. (c) With larger noise in visual depth, tactile help curb large pose tracking errors.*

### 2.4 Perceiving under duress: occlusion and visual depth noise

In this section, we explore the broader benefits of fusing touch and vision through ablations on visual sensing properties. The previous results were achieved through the iterative co-design of perception and hardware, such that we have favorable camera positioning and precise stereo depth tuning. Indeed, this attention to detail is necessary for practitioners [[[24](#bib.bib24), [10](#bib.bib10)]], but can we also harness touch to improve over sub-optimal visual data? We consider two such scenarios in simulation, where we can freely control these parameters, and evaluate on the pose tracking problem from the previous section.

The effects of camera-robot occlusion. In an embodied problem, third-person and egocentric cameras are both susceptible to occlusion from robot motion and environment changes. For example, if we were to retrieve a cup off the top shelf in the kitchen, we rely primarily on tactile signals to complete the task. For the perception system, this translates to the object of interest disappearing from the field of view, while local touch sensing is still unaffected. To emulate this we consider tracking the pose of a known Rubik’s cube. We simulate $200$ different cameras in a sphere of radius $0.5\,\text{m}$, each facing towards the robot. As shown in Figure [6](#S2.F6) (a), each camera captures a unique vantage point of the same in-hand sequence, with varying levels of robot-object occlusion. This serves as proxy for occlusion faced by an egocentric or fixed camera when either the hand or environment occludes the object.

To simplify the experiment, we assume the upper-bound performance of the vison-only frontend by providing ground-truth object segmentation masks. We characterize the visibility in terms of an occlusion score by calculating the average segmentation mask area for each viewpoint, and normalizing them to $\left[0\!-\!1\right]$. For example, scores closer to 00 correspond to viewpoints beneath the hand (most occluded), while those closer to $1$ correspond to cameras placed atop (least occluded). We run pose tracking experiments for each of the $200$ cameras in two modes: vision-only and visuo-tactile and compare between them.

In Figure [6](#S2.F6) (a) we colormap each camera view based on the pose tracking improvements from incorporating touch. On average the improvement across all cameras is $21.2$%, and it peaks at $94.1$% at heavily occluded views. We inset frames from a few representative viewpoints and their corresponding relative improvement with visuo-tactile fusion. In Figure [6](#S2.F6) (b) the pose error for each modality is further plotted versus the $\left[0\!-\!1\right]$ occlusion score. This corroborates the idea that touch refines perception in low-occlusion regimes and robustifies it in high-occlusion regimes.

The effects of noisy visual depth. Depth from commodity RGB-D sensors are degraded as a function of camera-robot distance, environment lighting, and object specularity. Even in ideal scenarios, the RealSense depth algorithm has $35$ hyperparameters [[[34](#bib.bib34)]] that considerably affect the frontend input to NeuralFeels. To simulate this, we corrupt the depth maps progressively with a realistic RGB-D noise, and observe the tracking performance for a known geometry.

As implemented by Handa et al. [[[25](#bib.bib25)]], we simulate common sources of depth-map errors as a sequence of pixel shuffling, quantization, and high frequency noise. The depth noise factor $D$ determines the magnitude of these operations, with the depth-maps visualized in Figure [6](#S2.F6) (c). While all prior simulation experiments have been collected with $D\!=\!5$, here we vary the magnitude from $0\!-\!50$ in intervals of $10$. At each noise level, we run pose tracking across the $5$ Rubik’s cube experiments with $5$ unique seeds, resulting in a total of $150$ experiments. In Figure [6](#S2.F6) (c) we plot error against the noise factor $D$, showing an expected upward trend in error with noise. However, we see markedly better tracking when fusing touch, especially in high-noise regimes.

## 3 Discussion

NeuralFeels achieves robust object-centric SLAM through interaction. To the best of our knowledge, NeuralFeels is the first demonstration of full-SLAM for multimodal, multifinger manipulation. We are inspired by computer vision systems that achieve high-fidelity neural reconstructions without pose annotation [[[69](#bib.bib69), [97](#bib.bib97), [82](#bib.bib82)]] through online learning. They highlight the benefit of co-designed pose tracking and reconstruction, which has also shown promise in manipulation systems [[[71](#bib.bib71), [95](#bib.bib95)]]. More broadly, our stack relies on recent progress in somewhat disparate fields: SLAM, neural rendering, tactile sensing, and reinforcement-learning.

As shown in the Figure [3](#S2.F3) (a), we achieve average reconstruction F-scores of $81$% across simulation and real-world experiments on novel objects. Simultaneously, we stably track these objects amidst interaction with minimal drift, an average of $4.7\,\text{mm}$. While the vision-only baseline may suffice for some scenarios, the results validate the utility of rich, multimodal sensing for interactive tasks. This corroborates years of research in interactive perception from touch and vision [[[65](#bib.bib65), [73](#bib.bib73), [5](#bib.bib5)]], now applied on dexterous manipulation platforms.

Touch and proprioception ground embodied perception. Interactive perception is far from ideal, an embodiment can more often than not get in the way of sensing. As seen in Figure [4](#S2.F4), in-hand manipulation suffers from challenges such as frequent occlusions, limited field-of-view, noisy segmentation, and rapid object motion. To tackle this, proprioception helps focus the perception problem: we can accurately singulate the object of interest through embodied prompting (Section [4.6.1](#S4.SS6.SSS1)). When combined with touch, we robustify our visual estimates by giving us a window into local interactions. These are evident in simulated / real SLAM and pose tracking experiments, where multimodal fusion leads to improvements of $15.3$% / $14.6$% in reconstruction and $21.3$% / $26.6$% in pose tracking.

Qualitatively, we see touch performs two key functions: (i) disambiguating noisy frontend estimates and (ii) providing context in the presence of occlusion. The former alleviates the effect of noisy visual segmentation and depth with co-located local information for mapping and localization. The latter provides important context hidden from visual sensing, like the occluded face of the large dice or back of the rubber duck. The final reconstructions in Figure [3](#S2.F3) (e) support these findings, with improved shape completion and refinement. This is important in the few-shot interactions of everyday life, where the richer sensing can create better object models.

The largest gains from incorporating touch are in heavy-occlusion regimes (Figure [6](#S2.F6) (a)), where we can observe up to $94.1$% improvements at certain camera viewpoints. To our knowledge, this co-design of perception and hardware has not been explored by practitioners before. This doesn’t just demonstrate the complementary nature of the modalities, but further, the ideal configurations for occlusion-free manipulation. Finally, our results in tactile-only tracking (Figure [5](#S2.F5) (c)) support the analysis of Smith et al. [[[64](#bib.bib64)]] that learning exclusively from touch leads to poor performance as it lacks any global context.

Modularity marries pre-training with online learning. As opposed to an end-to-end perception, NeuralFeels is fully interpretable due to its modular construction. This allows us to combine foundational models trained on large-scale image and tactile data (frontend), with SLAM as online learning (backend). Furthermore, our backend is a combination of state-of-the-art neural models [[[49](#bib.bib49)]] with classical least-square optimization [[[55](#bib.bib55)]] that have found success in SLAM [[[8](#bib.bib8)]]. Chaining these systems together, we can achieve first-of-its-kind multimodal SLAM results without explicit training in the domain. This is crucial given the dearth of training data for in-hand tasks, and robot manipulation in general.

This modular design has benefits for future generalization of our system: (i) Other models of tactile sensors [[[93](#bib.bib93), [79](#bib.bib79), [1](#bib.bib1)]] can be easily integrated as long as they can be accurately simulated; (ii) alternate scene representations [[[4](#bib.bib4), [31](#bib.bib31)]] can supplant our neural field model, as required; (iii) additional state knowledge can be seamlessly integrated as factor graph cost functions, e.g. tactile odometry [[[95](#bib.bib95)]] and force-constraints [[[71](#bib.bib71)]]; (iv) any combination of tactile and visual sensors can be fused into our multimodal framework with appropriate calibration and kinematics.

Application towards perception-driven planning. NeuralFeels is relevant to manipulation researchers and practitioners who require spatial perception with a single camera and affordable tactile sensing. It can be extended to not just in-hand rotation, but many other object-centric manipulation tasks like in-hand reorientation [[[10](#bib.bib10)]], pick-and-place [[[5](#bib.bib5)]], insertion [[[42](#bib.bib42)]], nonprehensile sliding [[[33](#bib.bib33)]], and planar pushing [[[71](#bib.bib71)]]. In the future, we hope to generalize to these different tasks and varied robot morphologies. While not explored in this work, the direct benefit of an online SDF is the ability to seamlessly plan for dexterous interactions. Recent works demonstrate the benefit of apriori-known object point-clouds [[[57](#bib.bib57)]] and SDFs [[[18](#bib.bib18)]] for goal-conditioned planning, and running our perception stack in-the-loop is the next natural step.

System limitations. NeuralFeels shows the potential of a multimodal system for manipulation that leverages pre-training and online learning for high accuracy spatial understanding. We present some of the limitations and promising directions for future work:

-
•

Generic 3D priors for object reconstruction. For each experiment with a novel object, our method learns a 3D geometry from scratch to best explain the visuo-tactile sensor stream. The pose tracker has a higher chance of failure in the initial few seconds, when the neural SDF is a poor-approximation of the full object due to limited sensor coverage. We further note that our rotation policy might not completely explore the object in the real-world, resulting in a lower average final F-Score of $81$%. Out-of-scope in our work, but of great interest in the visual learning community [[[54](#bib.bib54), [85](#bib.bib85), [29](#bib.bib29)]], is leveraging pre-trained models for an initial object prior. Given an initial occluded view, careful integration of these large reconstruction models trained via category [[[54](#bib.bib54)]] or multi-view supervision [[[85](#bib.bib85), [29](#bib.bib29)]] may yield an initial-guess SDF that we refine over time with vision and touch. In manipulation, Wang et al. [[[80](#bib.bib80)]] have seen promising results in using shape priors for visuo-tactile reconstruction of fixed objects.

-
•

Sim-to-real adaptation. Our findings indicate that while multimodal fusion performs well both in simulation and the real-world, its benefits are less pronounced in real-world deployment. This is a common problem in sim-to-real applications, and we qualitatively identify several domain gaps that explain this: (i) the DIGIT elastomer is less sensitive in real-world deployment, leading to sparser contact predictions; (ii) our RL policy is less stable in the real-world (sometimes requiring human intervention) and causes rapid jumps in object motion; (iii) noise in proprioception is only indirectly modelled as uncertainty terms in estimation. To tackle these, we must leverage work in sim-to-real generalization for tactile simulation [[[27](#bib.bib27)]] and reinforcement-learning [[[57](#bib.bib57)]].

-
•

System design considerations. We identify viable engineering improvements that can be made towards a general-purpose system. We are currently restricted to a fixed-camera setup, with an online hand-eye calibration or egocentric vision, this can be relaxed. Depth uncertainty [[[15](#bib.bib15)]] is valuable information for our neural model to handle visually-adversarial objects like glass and metal. To achieve true real-time frequencies, efficiency gains can be made in the pose optimizer and frontend estimation. Finally, we can increase robustness by using the color information for feature-based tracking of objects [[[14](#bib.bib14)]].

-
•

Perceiving latent state. We consider geometry as just the starting point for neural models: interaction reveals latent properties like texture [[[33](#bib.bib33)]], friction [[[41](#bib.bib41)]], and object dynamics [[[70](#bib.bib70)]]. Neural fields can embed these latents as auxiliary optimization terms so as to benefit tasks that go beyond just geometry and pose. Applications can range from learning to manipulate inertially-significant objects (e.g. a hammer), to identifying a grasp point from local texture (e.g. a saucepan handle).

## 4 Materials and methods

NeuralFeels ingests multimodal information to build a persistent 3D object representation. Similar to classical SLAM frameworks, it first has a frontend, responsible for abstracting the vision (RGB-D) and touch (RGB) input stream into a format suitable for estimation (segmented depth). Thereafter, the backend fuses this data into an optimization structure that infers the object model: an evolving posed object SDF. An illustration of the entire pipeline is found in Figure [2](#S1.F2), which we refer the reader back to throughout this section.

### 4.1 Task definition

NeuralFeels incrementally builds an object model, simultaneously optimizing for the object SDF network’s weights $\theta$ and its corresponding pose $\mathbf{x}_{t}$ at the current timestep $t$. For object exploration, we use a proprioception-driven policy $\mathbf{\pi}_{t}$ that executes the optimal action to achieve stable rotation. The input stream of all sensors $\mathcal{S}$ consists of the following (left column of Figure [2](#S1.F2)):

-
•

RGB-D vision: image $I_{t}^{c}$ and depth $D_{t}^{c}$ from calibrated camera $c\in\mathcal{S}$

-
•

RGB touch: images $I_{t}^{s}$ from four DIGITs [[[40](#bib.bib40)]]; $s\in\{d_{\text{index}},d_{\text{middle}},d_{\text{ring}},d_{\text{thumb}}\}\in%
\mathcal{S}$

-
•

Proprioception: joint-angles $\mathbf{q}_{t}$ from robot encoders.

### 4.2 Robot hardware and simulation

The Allegro hand [[[84](#bib.bib84)]] is retrofit with four DIGIT vision-based tactile sensors [[[40](#bib.bib40)]], at each of the distal ends. The DIGIT produces a $240\!\times\!320$ RGB image of the physical interaction at $30\,\text{Hz}$. The Allegro publishes 16D joint-angles so as to situate the tactile sensors with respect to the base frame. The hand is rigidly mounted on a Franka Panda arm, with an Intel D435 RGB-D camera placed at approximately $35\,\text{cm}$ from it. The camera extrinsics are computed with respect to the base frame of the Allegro through ArUco [[[20](#bib.bib20)]] hand-eye calibration. For our vision pseudo-ground-truth we use three such cameras in the workspace (Figure [7](#S4.F7)), jointly calibrated via Kalibr [[[19](#bib.bib19)]], to achieve $\approx 1\,\text{px}$ reprojection error. Our simulator replicates the real-world setup: a combination of the IsaacGym physics simulator [[[45](#bib.bib45)]] with the TACTO touch renderer [[[78](#bib.bib78)]]. In this case, we can record and store the true ground-truth object pose directly from IsaacGym.

### 4.3 FeelSight: a visuo-tactile perception dataset

![Figure](extracted/5308748/images/robot_cell.jpg)

*Figure 7: Robot setup in the real-world and simulation. (a) We capture diverse visuo-tactile interactions across different object categories in the real-world and physics simulation. (b) The robot cell is made up of three realsense RGB-D cameras, an Allegro robot hand mounted on a Franka Panda, and four DIGIT tactile sensors. All real-world results use the primary camera and DIGIT sensing, while the additional cameras are fused for our ground-truth pose tracking. In simulation, we use an identical primary camera in IsaacGym with touch simulated in TACTO. The simulator provides ground-truth object pose, so multi-camera tracking is not necessary.*

Visuo-tactile perception lacks a standardized benchmark or dataset that has driven progress in adjacent fields like visual tracking [[[28](#bib.bib28)]], SLAM [[[21](#bib.bib21)]], and reinforcement learning [[[30](#bib.bib30)]]. Towards this, we introduce our FeelSight dataset for visuo-tactile manipulation. We use the in-hand rotation policy (Section [S4](#Sx3.SS4)) to collect vision, touch, and proprioception for $30$ seconds per trial.

When we encounter a novel object, we tend to twirl it in our hand to get a better look from different views, and regrasp it from different angles. The equivalent for a multi-fingered hand, in-hand rotation, is an ideal choice for the interactive perception problem. We adopt the method of Qi et al. [[[56](#bib.bib56)]] where they train a proprioception-based policy in simulation, and directly transfer it to the real-world. Recent work has further shown in-hand object rotation using touch and proprioceptive history [[[57](#bib.bib57), [89](#bib.bib89)]], however our simpler abstraction proves sufficient for this task. In our experiments, the rotation policy $\mathbf{\pi}_{t}$ sends commands to the robot hand at $20\,\text{Hz}$ via the ROS Allegro controller. This achieves stable rotation of novel objects and interesting visuo-tactile stimuli; for further details refer to Section [S4](#Sx3.SS4).

The dataset has $5$ in-hand rotation trials each of $6$ objects in the real-world and $8$ objects in simulation; a total $35$ minutes of interaction. As explained in Figure [7](#S4.F7), we record a pseudo-ground-truth in the real-world, and exact ground-truth poses in simulation. We ensure diversity in the class of objects: they vary in geometry and size from $6$-$18\,\text{cm}$ in diagonal length. Ground-truth meshes of each object are obtained with the Revopoint 3D scanner [[[59](#bib.bib59)]], which uses dual-camera infrared for $\approx 0.05\,\text{mm}$ scan accuracy. Additionally, the the simulated experiments have ground-truth meshes from the YCB [[[9](#bib.bib9)]] and ContactDB [[[7](#bib.bib7)]] datasets.

### 4.4 Method overview and key insights

Object model (Section [4.5](#S4.SS5)): We represent the object SDF as a neural network with weights $\theta$, whose output is transformed by the current object pose $\mathbf{x_{t}}$. This continuous function $F_{\mathbf{x}_{t}}^{\theta}(\mathbf{p}):\mathbb{R}^{3}\rightarrow\mathbb{R}$ maps a 3D coordinate $\mathbf{p}$ to a scalar signed-distance from the object’s closest surface. Online updates are decomposed into alternating steps between refining the weights of the neural SDF $\theta$, and optimizing the object pose $\mathbf{x_{t}}$. Our bespoke object model is a representation of both the pose and object geometry over time.

Frontend (Section [4.6.1](#S4.SS6.SSS1), [4.6.2](#S4.SS6.SSS2)): Given the RGB-D, RGB, and proprioception inputs, our frontend returns segmented depth measurements compatible with our backend optimizer. These modules are pre-trained with a large corpus of data.

Shape optimizer (Section [4.7.1](#S4.SS7.SSS1)): Takes in frontend output and optimizes for $\theta$ at fixed object pose $\mathbf{\bar{x}_{t}}$ via gradient descent [[[49](#bib.bib49)]]. Each shape iteration results in improved object SDF $F_{\mathbf{\bar{x}}_{t}}^{\theta}$.

Pose optimizer (Section [4.7.2](#S4.SS7.SSS2)): Builds and optimizes an object pose-graph [[[55](#bib.bib55)]] for $\mathbf{x}_{t}$ given fixed network weights $\bar{\theta}$. Every pose iteration spatially aligns the evolving object SDF with the current set of frontend output.

#### 4.4.1 Insight 1: NeuralFeels is a posed neural field

The object model $F_{\mathbf{x_{t}}}^{\theta}$ is estimated by a chicken-and-egg optimization of both the instant-NGP weights $\theta$, and the object pose $\mathbf{x}_{t}$. Prior work has estimated the pose of a sensor in fixed neural field, either by freezing the network weights [[[88](#bib.bib88), [44](#bib.bib44)]], or joint-optimization [[[69](#bib.bib69), [97](#bib.bib97), [60](#bib.bib60)]]. In our case, robot kinematics gives us the pose of the touch sensors, and extrinsics give us the pose of the camera. So, we instead flip this paradigm to estimate the pose of the neural field with respect to known-pose sensors.

#### 4.4.2 Insight 2: Touch is vision, albeit local

We extend neural fields to directly incorporate touch just as it would vision. Our key insight is that vision-based touch can be approximated as a perspective camera model in tactile simulators like TACTO [[[78](#bib.bib78)]]. There are, however, differences that must be accounted for in image formation (i) vision-based tactile sensor impose their own color and illumination to the scene, which makes it hard to get reliable visual cues, (ii) a tactile image stream has considerably smaller metric field-of-view and depth-range is usually in centimeters rather than meters, (iii) tactile images have depth discontinuities along all non-contact regions, while natural images only encounter them along occlusion boundaries. Our method adapts each of these by (i) consistently using depth rather than color for optimization, (ii) sampling at different scales (centimeter v.s. meter) based on sensing source, (iii) sampling only surface points for touch, but both free-space and surface points for vision. More details are described in Section [4.7.1](#S4.SS7.SSS1). After accounting for these differences, we can sample touch consistent with vision, giving us a rich perspective of the object.

### 4.5 Object model

Our object model is depicted in the right column of Figure [2](#S1.F2). In general, a neural SDF [[[52](#bib.bib52), [3](#bib.bib3), [49](#bib.bib49)]] represents 3D surfaces as the zero level-set of a learnable function $F(\mathbf{p}):\mathbb{R}^{3}\rightarrow\mathbb{R}$. The scalar field’s sign indicates if any query point $\mathbf{p}$ in the volume is inside (negative), outside (positive) or on ($\approx$ 0) the reconstructed surface. $\mathbf{p}$ is first positionally-encoded [[[74](#bib.bib74)]] into a higher-dimensional space, an important routine that helps networks better approximate high-frequency surfaces. This is followed by a multi-layer perceptron (MLP) that fits the encoding to a scalar field. Typically, this network is optimized with depth samples from a camera of known intrinsics, and annotated poses from structure-from-motion [[[61](#bib.bib61)]].

A neural SDF is more compact than the more popular neural radiance fields [[[46](#bib.bib46)]], as they do not model color and appearance properties of the scene. This is sufficient for manipulation, as we care more about estimating geometry than generating novel-views. Recently, instant-NGP [[[49](#bib.bib49)]] demonstrated a learnable multiresolution hash table as a positional encoding that greatly accelerates SDF optimization with small MLP backbones. This has been successfully leveraged for real-time SLAM in indoor scene [[[60](#bib.bib60)]].

In our work, $F_{\mathbf{x}_{t}}^{\theta}$ represents the neural SDF of the object at a given pose $\mathbf{x}_{t}$. While $\mathbf{x}_{t}$ is initialized to be between the robot fingers, $\theta$ is randomly initialized. Both shape and pose are estimated via alternating optimization, which emulating the paradigm of tracking and mapping that has found great success in robot vision [[[8](#bib.bib8)]]. The model is fully-differentiable, can be queried arbitrarily in 3D space, and easily extensible to color, latent physics, and other properties.

### 4.6 Frontend

The frontend processes are shown in the center column of Figure [2](#S1.F2). Its function is to robustly extract depth measurements from raw vision and touch sensing. Depth is available as-is in an RGB-D camera, but the challenge is to robustly segment out object depth pixels in heavily-occluded interactions. Towards this, we introduce a kinematics-aware segmentation strategy using powerful vision foundation models [[[36](#bib.bib36)]] (Section [4.6.1](#S4.SS6.SSS1)). Estimating depth from vision-based touch is an open research problem [[[6](#bib.bib6), [79](#bib.bib79), [67](#bib.bib67), [2](#bib.bib2), [73](#bib.bib73)]] where millimeter precision and generalization across sensors is important. Towards this, we present a transformer architecture that accurately predicts DIGIT contact patches from inputs images (Section [4.6.2](#S4.SS6.SSS2)). Unlike our backend that is optimized online, the frontend networks are pre-trained from a large corpus of data. The output of our frontend is a segmented depth image $\hat{D}_{t}^{s}$ for each sensor $s\in\mathcal{S}$.

![Figure](extracted/5308748/images/frontend_and_backend.jpg)

*Figure 8: Frontend and backend description. (a) Segment-Anything [[[36](#bib.bib36)]] combined with embodied prompts, gives us robust object segmentation. Through reasoning about finger occlusion and object pose with respect to the fingers, we can accurately prompt the segmentation network for robust output masks. (b) Representative examples of the sim-to-real performance of the tactile transformer. Each RGB image is fed through the network to output a predicted depth, along with a contact mask. (c) Our sliding window nonlinear least squares optimizer estimates the object pose $x_{t}$ from the outputs of the frontend. Each object pose $x_{t}$ is constrained by the SDF loss, frame-to-frame ICP, and pose regularization to ensure tracking remains stable.*

#### 4.6.1 Segmented visual depth

During in-hand manipulation, finger-object occlusion is inevitable and the foreground-background is ambiguous. Robust segmentation of the image stream $I_{t}^{c}$ via prompts has successfully been demonstrated by image foundation models, like the Segment Anything Model (SAM)
[[[36](#bib.bib36)]]. Trained with a vision transformer (ViT) in the data-rich natural image domain, SAM generalizes to novel scenes for state-of-the-art, zero-shot instance segmentation.

Even with SAM, in-hand object segmentation requires appropriate prompts to guide the pre-trained model. With an embodied agent, we can take advantage of robot kinematics to achieve this. Given our camera $c$ with known projection operation $\Pi^{c}$, we can obtain any 3D point $\mathbf{p}$ as a pixel $(u,v)=\Pi^{c}\left(\mathbf{p}\right)$ on the image $I_{t}^{c}$. Our insight is to use the 3D center of grasp and robot kinematics as prompts for SAM (refer to Section [S3](#Sx3.SS3)). This makes the reasonable assumption that the object exists between the robot’s fingers, which is almost always the case. In Figure [8](#S4.F8) (a) we visualize the segmentation on real-world images, alongside the SAM prompts. In our experiments we use the ViT-L model with $308\,\!\text{M}$ parameters. While this achieves a speed of around $4$Hz, in practice, we can use efficient segmentation models [[[94](#bib.bib94)]] for speeds up to $40$Hz.

#### 4.6.2 Tactile transformer

In contrast, vision-based touch images are out-of-distribution from images SAM is typically trained on, and does not directly provide depth either. The embedded camera perceives an illuminated gelpad, and contact depth is either obtained via photometric stereo [[[93](#bib.bib93)]], or supervised learning [[[6](#bib.bib6), [79](#bib.bib79), [67](#bib.bib67), [2](#bib.bib2), [73](#bib.bib73)]]. Existing touch-to-depth relies on convolution, however recent work has shown the benefit of a ViT for dense depth prediction [[[58](#bib.bib58)]] in natural images. We train a tactile transformer for predicting contact depth from vision-based touch to generalize across multiple real-world DIGIT sensors.

The architecture is trained entirely in tactile simulation, using weights initialized from a pre-trained image-to-depth model [[[58](#bib.bib58)]]. The tactile transformer represents the inverse sensor model $\mathbf{\Omega}:I_{t}^{s}\mapsto\hat{D}_{t}^{s}$ where $s\in\{d_{\text{index}},d_{\text{middle}},d_{\text{ring}},d_{\text{thumb}}\}\in%
\mathcal{S}$. This architecture is based on the dense vision transformer [[[58](#bib.bib58)]] and is lightweight (21.7M parameters) compared to its fully-convolution counterparts [[[72](#bib.bib72)]].

Similar to prior work [[[73](#bib.bib73), [72](#bib.bib72)]], we generate a large corpus of tactile images and paired ground-truth depthmaps in the optical touch simulator TACTO [[[78](#bib.bib78)]]. We collect 10K random tactile interactions each on the surface of 40 unique YCB objects [[[9](#bib.bib9)]]. For sim-to-real transfer we augment the data with randomization in sensor LED lighting, indentation depth, and pixel noise. In TACTO, image realism is achieved by compositing with template non-contact images from real-world DIGITs. For details on the training and data, refer to Section [S2](#Sx3.SS2).

These augmentations enable generalized performance across our multi-finger platform, where each sensor has differing image characteristics. Our tactile transformer is supervised on mean-square depth reconstruction loss against the ground-truth depthmaps from simulation. Based on the predicted depthmaps, the output is thresholded to mask out non-contact regions. The tactile transformer demonstrates an average prediction error of $0.042\,\text{mm}$ on simulated test set. Figure [8](#S4.F8) (b) shows sim-to-real performance of the tactile transformer on real-world interactions.

### 4.7 Backend: shape and pose optimizer

The backend (right column of Figure [2](#S1.F2)) is responsible for taking in depth and sensor poses from the frontend to build our object model online. This alternates between shape (Section [4.7.1](#S4.SS7.SSS1)) and pose optimization (Section [4.7.2](#S4.SS7.SSS2)) steps using samples from the visuo-tactile depth stream. Similar to other neural SLAM methods [[[52](#bib.bib52)]], the modules maintain a bank of keyframes over time to generate these samples. Additional implementation details for the backend are found in Section [S3](#Sx3.SS3).

#### 4.7.1 Shape optimizer

For online estimation it is intractable to optimize $F_{\mathbf{\bar{x}}_{t}}^{\theta}$ using all input frames as in neural radiance fields [[[46](#bib.bib46)]]. We opt for an online learning approach [[[69](#bib.bib69), [52](#bib.bib52)]], which builds a subset of keyframes $\mathcal{K}$ on-the-fly to optimize over. The backend must both (i) accept new keyframes based on a criteria, and (ii) replay old keyframes in the optimization to prevent catastrophic forgetting [[[69](#bib.bib69)]]. Each iteration of the shape optimizer replays a batch $k_{t}\in\mathcal{K}$ of size $10$ per sensor to optimize our network. This includes the latest two frames, and a weighted random sampling of past keyframes based on average rendering loss.

The initial visuo-tactile frame is automatically added as a keyframe $\mathcal{K}_{0}=\{\hat{D}_{0}^{s}\ |\ s\in\mathcal{S}\}$, and every subsequent keyframe $\mathcal{K}_{t}$ is accepted using an information gain metric [[[69](#bib.bib69)]]. For this, the average rendering loss is computed from the frozen network $F_{\mathbf{\bar{x}}_{t}}^{\theta}$ using the given keyframe pose and compared against a threshold $d_{\text{thresh}}=0.01\,\text{m}$. Finally, if we have not added a keyframe for an interval $t_{\text{max}}=0.2\,\text{secs}$, we force one to be added.

Sampling and SDF loss. At each iteration, we sample coordinates in the neural volume from $k_{t}$ to optimize the neural weights $\theta$. The first step is to sample a batch of pixels $\mathbf{u}_{k_{t}}$ from $k_{t}$—a mix of surface and free-space pixels. While surface pixels directly supervise the SDF zero level-set, free-space pixels carve out the neural volume. In our implementation, we sample $50\%$ of camera pixels in free-space, while we only sample surface pixels for touch. Through each pixel $u\in\mathbf{u}_{k_{t}}$ given their corresponding sensor pose, we project a ray into the neural volume. Similar to Ortiz et al. [[[52](#bib.bib52)]], we sample $P_{u}$ points per ray, a mix of stratified and surface points.

With these samples, we compute an SDF prediction $\mathbf{\hat{d}}_{u}$ for each $\hat{D}_{t}\in k_{t}$, as the batch distance bound [[[52](#bib.bib52)]]. For each ray, we split the samples into $P_{u}^{\text{f}}$ and $P_{u}^{\text{tr}}$ based on $\mathbf{\hat{d}}_{u}$ lies within the truncation distance $d_{\text{tr}}\!=\!5\,\text{mm}$ from the surface. Our shape loss $\mathcal{L}_{\text{shape}}=\mathcal{L}_{\text{f}}+w_{\text{tr}}\mathcal{L}_{%
\text{tr}}$, with $w_{\text{tr}}=10$, resembles the truncated SDF loss of Azinović et al. [[[3](#bib.bib3)]]:

| | $\displaystyle\mathcal{L}_{\text{f}}=\frac{1}{|\mathbf{u}_{k_{t}}|}\sum\limits_% {u\in\mathbf{u}_{k_{t}}}\frac{1}{|P_{u}^{\text{f}}|}|F_{\mathbf{\bar{x}}_{t}}^% {\theta}(P_{u}^{\text{f}})-d_{\text{tr}}|\qquad\text{and}\qquad\mathcal{L}_{% \text{tr}}=\frac{1}{|\mathbf{u}_{k_{t}}|}\sum\limits_{u\in\mathbf{u}_{k_{t}}}% \frac{1}{|P_{u}^{\text{tr}}|}|F_{\mathbf{\bar{x}}_{t}}^{\theta}(P_{u}^{\text{% tr}})-\mathbf{\hat{d}}_{u}|$ | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

#### 4.7.2 Pose optimizer

Before each shape iteration, we use a pose graph [[[13](#bib.bib13)]] to refine the object pose $\mathbf{x}_{t}$ with respect to the frozen neural field $F_{\mathbf{x}_{t}}^{\bar{\theta}}$. We achieve this by inverting the problem to instead optimize for the 6-DoF poses in a sliding window of size $n$. At timestep $t$, if we have accumulated $N$ keyframes, this is represents poses
$\mathcal{X}_{t}\!=\!{\left(\mathbf{x}_{i}\right)}_{N\!-\!n\leq i\leq N}$
and measurements
$\mathcal{M}_{t}\!=\!{\left(\hat{D}_{i}^{s}\ |\ s\!\in\!\mathcal{S}\right)}_{N%
\!-\!n\leq i\leq N}$.
Similar to pose updates in visual SLAM [[[88](#bib.bib88), [69](#bib.bib69), [97](#bib.bib97)]], the network weights $\bar{\theta}$ are frozen and we estimate the $\textit{SE}(3)$ poses $\mathcal{X}_{t}$ instead.

We formulate the problem as a nonlinear least squares optimization with custom measurement factors in Theseus [[[55](#bib.bib55)]]. While prior work uses gradient descent [[[88](#bib.bib88)]], we instead use a second-order Levenberg–Marquardt (LM) solver, which provides faster convergence [[[13](#bib.bib13)]]. The pose graph, illustrated in Figure [8](#S4.F8) (c), solves for:

| | $\displaystyle\hat{\mathcal{X}_{t}}=\underset{\mathcal{X}_{t}}{\text{argmin}}\ % \mathcal{L_{\text{pose}}}(\mathcal{X}_{t}\ |\ \mathcal{M}_{t},\bar{\theta})% \qquad\text{where}\qquad\mathcal{L_{\text{pose}}}=w_{\text{sdf}}\mathcal{L}_{% \text{sdf}}+w_{\text{reg}}\mathcal{L}_{\text{reg}}+w_{\text{icp}}\mathcal{L}_{% \text{icp}}$ | |
|---|---|---|---|

-
•

SDF loss $\mathcal{L}_{\text{sdf}}$. We use the shape loss $\mathcal{L}_{\text{shape}}$, modified such that we sample only about surface points of each ray. This works well for both visual and tactile sensing as we have higher confidence in SDFs about the surface of the object than in free-space. For each depth measurement in $\mathcal{M}_{t}$, we sample surface points over $M$ rays, and average the SDF loss along each ray. This results in an $M\!\times\!n$ SDF loss, which we use to update the se(3) lie algebra of $\mathcal{X}_{t}$. We implement a custom Jacobian for this cost function, which is up to 4$\times$ more efficient than PyTorch automatic differentiation.

-
•

Pose regularizer $\mathcal{L}_{\text{reg}}$. We apply a weak regularizer between consecutive keyframe poses in $\mathcal{X}_{t}$ to ensure the relative pose updates stay well-behaved. This is important for robustness to noisy frontend depth and incorrect segmentations.

-
•

ICP loss $\mathcal{L}_{\text{icp}}$. We further apply iterative closest point (ICP) between the current visuo-tactile pointcloud $\Pi^{-1}(\mathcal{M}_{t})$ and previous pointcloud $\Pi^{-1}(\mathcal{M}_{t-1})$. This gives us frame-to-frame constraints in addition to the frame-to-model $\mathcal{L}_{\text{sdf}}$.

## References

-
[1]

Alex Alspach, Kunimatsu Hashimoto, Naveen Kuppuswamy, and Russ Tedrake.

Soft-bubble: A highly compliant dense geometry tactile sensor for robot manipulation.

In Proc. IEEE Intl. Conf. on Soft Robotics (RoboSoft), pages 597–604. IEEE, 2019.

-
[2]

Rares Ambrus, Vitor Guizilini, Naveen Kuppuswamy, Andrew Beaulieu, Adrien Gaidon, and Alex Alspach.

Monocular depth estimation for soft visuotactile sensors.

In Proc. IEEE Intl. Conf. on Soft Robotics (RoboSoft), 2021.

-
[3]

Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies.

Neural rgb-d surface reconstruction.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6290–6301, 2022.

-
[4]

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.

Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864, 2021.

-
[5]

Maria Bauza, Antonia Bronars, Yifan Hou, Ian Taylor, Nikhil Chavan-Dafle, and Alberto Rodriguez.

simple: a visuotactile method learned in simulation to precisely pick, localize, regrasp, and place objects.

arXiv preprint arXiv:2307.13133, 2023.

-
[6]

Maria Bauza, Oleguer Canal, and Alberto Rodriguez.

Tactile mapping and localization from high-resolution tactile imprints.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), pages 3811–3817. IEEE, 2019.

-
[7]

Samarth Brahmbhatt, Ankur Handa, James Hays, and Dieter Fox.

ContactGrasp: Functional multi-finger grasp synthesis from contact.

In Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 2386–2393. IEEE, 2019.

-
[8]

Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza, José Neira, Ian Reid, and John J Leonard.

Past, present, and future of Simultaneous Localization and Mapping: Toward the robust-perception age.

IEEE Trans. on Robotics (TRO), 32(6):1309–1332, 2016.

-
[9]

Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M Dollar.

Yale-CMU-Berkeley dataset for robotic manipulation research.

Intl. J. of Robotics Research (IJRR), 36(3):261–268, 2017.

-
[10]

Tao Chen, Megha Tippur, Siyang Wu, Vikash Kumar, Edward Adelson, and Pulkit Agrawal.

Visual dexterity: In-hand dexterous manipulation from depth.

In ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems, 2023.

-
[11]

Yiting Chen, Ahmet Ercan Tekden, Marc Peter Deisenroth, and Yasemin Bekiroglu.

Sliding touch-based exploration for modeling unknown object shape with multi-fingered hands.

arXiv preprint arXiv:2308.00576, 2023.

-
[12]

Andrew J Davison.

FutureMapping: The computational structure of spatial AI systems.

arXiv preprint arXiv:1803.11288, 2018.

-
[13]

Frank Dellaert and Michael Kaess.

Factor graphs for robot perception.

Foundations and Trends in Robotics, 6(1-2):1–139, 2017.

-
[14]

Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich.

Superpoint: Self-supervised interest point detection and description.

In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224–236, 2018.

-
[15]

Eric Dexheimer and Andrew J Davison.

Learning a depth covariance function.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13122–13131, 2023.

-
[16]

Elliott Donlon, Siyuan Dong, Melody Liu, Jianhua Li, Edward Adelson, and Alberto Rodriguez.

GelSlim: A high-resolution, compact, robust, and calibrated tactile-sensing finger.

In Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 1927–1934. IEEE, 2018.

-
[17]

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.

An image is worth 16x16 words: Transformers for image recognition at scale.

arXiv preprint arXiv:2010.11929, 2020.

-
[18]

Danny Driess, Jung-Su Ha, Marc Toussaint, and Russ Tedrake.

Learning models as functionals of signed-distance fields for manipulation planning.

In Conference on Robot Learning, pages 245–255. PMLR, 2022.

-
[19]

Paul Furgale, Joern Rehder, and Roland Siegwart.

Unified temporal and spatial calibration for multi-sensor systems.

In 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 1280–1286. IEEE, 2013.

-
[20]

Sergio Garrido-Jurado, Rafael Muñoz-Salinas, Francisco José Madrid-Cuevas, and Manuel Jesús Marín-Jiménez.

Automatic generation and detection of highly reliable fiducial markers under occlusion.

Pattern Recognition, 47(6):2280–2292, 2014.

-
[21]

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun.

Vision meets robotics: The KITTI dataset.

The International Journal of Robotics Research, 32(11):1231–1237, 2013.

-
[22]

Phillip Grote, Joaquim Ortiz-Haro, Marc Toussaint, and Ozgur S Oguz.

Neural field representations of articulated objects for robotic manipulation planning.

arXiv preprint arXiv:2309.07620, 2023.

-
[23]

Irmak Guzey, Ben Evans, Soumith Chintala, and Lerrel Pinto.

Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play.

arXiv preprint arXiv:2303.12076, 2023.

-
[24]

Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, Yashraj Narang, Jean-Francois Lafleche, Dieter Fox, and Gavriel State.

DeXtreme: Transfer of agile in-hand manipulation from simulation to reality.

arXiv, 2022.

-
[25]

Ankur Handa, Thomas Whelan, John McDonald, and Andrew J Davison.

A benchmark for rgb-d visual odometry, 3d reconstruction and slam.

In 2014 IEEE international conference on Robotics and automation (ICRA), pages 1524–1531. IEEE, 2014.

-
[26]

Hannah B Helbig and Marc O Ernst.

Optimal integration of shape information from vision and touch.

Experimental brain research, 179(4):595–606, 2007.

-
[27]

Carolina Higuera, Byron Boots, and Mustafa Mukadam.

Learning to read braille: Bridging the tactile reality gap with diffusion models.

arXiv preprint arXiv:2304.01182, 2023.

-
[28]

Tomas Hodan, Frank Michel, Eric Brachmann, Wadim Kehl, Anders GlentBuch, Dirk Kraft, Bertram Drost, Joel Vidal, Stephan Ihrke, Xenophon Zabulis, et al.

BOP: Benchmark for 6D object pose estimation.

In Proceedings of the European conference on computer vision (ECCV), pages 19–34, 2018.

-
[29]

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan.

Lrm: Large reconstruction model for single image to 3d.

arXiv preprint arXiv:2311.04400, 2023.

-
[30]

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison.

RLbench: The robot learning benchmark and learning environment.

IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.

-
[31]

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.

3D Gaussian splatting for real-time radiance field rendering.

ACM Transactions on Graphics (ToG), 42(4):1–14, 2023.

-
[32]

Justin Kerr, Letian Fu, Huang Huang, Yahav Avigal, Matthew Tancik, Jeffrey Ichnowski, Angjoo Kanazawa, and Ken Goldberg.

Evo-NeRF: Evolving NeRF for sequential robot grasping of transparent objects.

In 6th Annual Conference on Robot Learning, 2022.

-
[33]

Justin Kerr, Huang Huang, Albert Wilcox, Ryan Hoque, Jeffrey Ichnowski, Roberto Calandra, and Ken Goldberg.

Learning self-supervised representations from vision and touch for active sliding perception of deformable surfaces.

arXiv preprint arXiv:2209.13042, 2022.

-
[34]

Leonid Keselman, Katherine Shih, Martial Hebert, and Aaron Steinfeld.

Optimizing algorithms from pairwise user preferences.

arXiv preprint arXiv:2308.04571, 2023.

-
[35]

Diederik P Kingma and Jimmy Ba.

Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980, 2014.

-
[36]

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.

Segment anything.

arXiv preprint arXiv:2304.02643, 2023.

-
[37]

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun.

Tanks and temples: Benchmarking large-scale scene reconstruction.

ACM Transactions on Graphics (ToG), 36(4):1–13, 2017.

-
[38]

Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, and Josef Sivic.

Megapose: 6d pose estimation of novel objects via render & compare.

arXiv preprint arXiv:2212.06870, 2022.

-
[39]

Alexander Sasha Lambert, Mustafa Mukadam, Balakumar Sundaralingam, Nathan Ratliff, Byron Boots, and Dieter Fox.

Joint inference of kinematic and force trajectories with visuo-tactile sensing.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), pages 3165–3171. IEEE, 2019.

-
[40]

Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, et al.

DIGIT: A novel design for a low-cost compact high-resolution tactile sensor with application to in-hand manipulation.

IEEE Robotics and Automation Letters (RA-L), 5(3):3838–3845, 2020.

-
[41]

Simon Le Cleac’h, Hong-Xing Yu, Michelle Guo, Taylor Howell, Ruohan Gao, Jiajun Wu, Zachary Manchester, and Mac Schwager.

Differentiable physics simulation of dynamics-augmented neural objects.

IEEE Robotics and Automation Letters, 8(5):2780–2787, 2023.

-
[42]

Marion Lepert, Chaoyi Pan, Shenli Yuan, Rika Antonova, and Jeannette Bohg.

In-hand manipulation of unknown objects with tactile sensing for insertion.

In Embracing Contacts-Workshop at ICRA 2023, 2023.

-
[43]

Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin.

Neuralangelo: High-fidelity neural surface reconstruction.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8456–8465, 2023.

-
[44]

Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey.

BARF: Bundle-adjusting neural radiance fields.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5741–5751, 2021.

-
[45]

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al.

Isaac Gym: High performance GPU-based physics simulation for robot learning.

arXiv preprint arXiv:2108.10470, 2021.

-
[46]

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.

NeRF: Representing scenes as neural radiance fields for view synthesis.

Communications of the ACM, 65(1):99–106, 2021.

-
[47]

Mark Moll and Michael A Erdmann.

Reconstructing the shape and motion of unknown objects with active tactile sensors.

In Algorithmic Foundations of Robotics V, pages 293–309. Springer, 2004.

-
[48]

Hans Moravec.

Mind children: The future of robot and human intelligence.

Harvard University Press, 1988.

-
[49]

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller.

Instant neural graphics primitives with a multiresolution hash encoding.

ACM Transactions on Graphics (ToG), 41(4):1–15, 2022.

-
[50]

OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al.

Solving Rubik’s Cube with a robot hand.

arXiv preprint arXiv:1910.07113, 2019.

-
[51]

OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafał Józefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, and Wojciech Zaremba.

Learning dexterous in-hand manipulation.

CoRR, 2018.

-
[52]

Joseph Ortiz, Alexander Clegg, Jing Dong, Edgar Sucar, David Novotny, Michael Zollhoefer, and Mustafa Mukadam.

iSDF: Real-time neural signed distance fields for robot perception.

arXiv preprint arXiv:2204.02296, 2022.

-
[53]

Akhil Padmanabha, Frederik Ebert, Stephen Tian, Roberto Calandra, Chelsea Finn, and Sergey Levine.

OmniTact: A multi-directional high-resolution touch sensor.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), pages 618–624. IEEE, 2020.

-
[54]

Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove.

DeepSDF: Learning continuous signed distance functions for shape representation.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 165–174, 2019.

-
[55]

Luis Pineda, Taosha Fan, Maurizio Monge, Shobha Venkataraman, Paloma Sodhi, Ricky TQ Chen, Joseph Ortiz, Daniel DeTone, Austin Wang, Stuart Anderson, Jing Dong, Brandon Amos, and Mustafa Mukadam.

Theseus: A Library for Differentiable Nonlinear Optimization.

Advances in Neural Information Processing Systems, 2022.

-
[56]

Haozhi Qi, Ashish Kumar, Roberto Calandra, Yi Ma, and Jitendra Malik.

In-hand object rotation via rapid motor adaptation.

In Conference on Robot Learning, pages 1722–1732. PMLR, 2022.

-
[57]

Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra Malik.

General in-hand object rotation with vision and touch.

In Conference on Robot Learning, pages 1722–1732. PMLR, 2023.

-
[58]

René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun.

Vision transformers for dense prediction.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

-
[59]

Revopoint.

Revopoint POP 3 3D Scanner, 2023.

-
[60]

Antoni Rosinol, John J Leonard, and Luca Carlone.

NeRF-SLAM: Real-time dense monocular SLAM with neural radiance fields.

arXiv preprint arXiv:2210.13641, 2022.

-
[61]

Johannes L Schonberger and Jan-Michael Frahm.

Structure-from-Motion revisited.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.

-
[62]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347, 2017.

-
[63]

Yu She, Shaoxiong Wang, Siyuan Dong, Neha Sunil, Alberto Rodriguez, and Edward Adelson.

Cable manipulation with a tactile-reactive gripper.

Intl. J. of Robotics Research (IJRR), 40(12-14):1385–1401, 2021.

-
[64]

Edward J Smith, Roberto Calandra, Adriana Romero, Georgia Gkioxari, David Meger, Jitendra Malik, and Michal Drozdzal.

3D shape reconstruction from vision and touch.

In Proc. Conf. on Neural Information Processing Systems (NeurIPS), 2020.

-
[65]

Edward J Smith, David Meger, Luis Pineda, Roberto Calandra, Jitendra Malik, Adriana Romero, and Michal Drozdzal.

Active 3D shape reconstruction from vision and touch.

In Proc. Conf. on Neural Information Processing Systems (NeurIPS), 2021.

-
[66]

Paloma Sodhi, Michael Kaess, Mustafa Mukadam, and Stuart Anderson.

Learning tactile models for factor graph-based estimation.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), pages 13686–13692. IEEE, 2021.

-
[67]

Paloma Sodhi, Michael Kaess, Mustafa Mukadam, and Stuart Anderson.

Patchgraph: In-hand tactile tracking with learned surface normals.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), 2022.

-
[68]

Claudius Strub, Florentin Wörgötter, Helge Ritter, and Yulia Sandamirskaya.

Correcting pose estimates during tactile exploration of object shape: a neuro-robotic study.

In 4th International Conference on Development and Learning and on Epigenetic Robotics, pages 26–33. IEEE, 2014.

-
[69]

Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison.

iMAP: Implicit mapping and positioning in real-time.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6229–6238, 2021.

-
[70]

Balakumar Sundaralingam and Tucker Hermans.

In-hand object-dynamics inference using tactile fingertips.

IEEE Transactions on Robotics, 37(4):1115–1126, 2021.

-
[71]

Sudharshan Suresh, Maria Bauza, Kuan-Ting Yu, Joshua G Mangelson, Alberto Rodriguez, and Michael Kaess.

Tactile SLAM: Real-time inference of shape and pose from planar pushing.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), May 2021.

-
[72]

Sudharshan Suresh, Zilin Si, Stuart Anderson, Michael Kaess, and Mustafa Mukadam.

Midastouch: Monte-carlo inference over distributions across sliding touch.

In 6th Annual Conference on Robot Learning, 2022.

-
[73]

Sudharshan Suresh, Zilin Si, Joshua G Mangelson, Wenzhen Yuan, and Michael Kaess.

ShapeMap 3-D: Efficient shape mapping through dense touch and vision.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), Philadelphia, PA, USA, May 2022.

-
[74]

Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng.

Fourier features let networks learn high frequency functions in low dimensional domains.

Advances in Neural Information Processing Systems, 33:7537–7547, 2020.

-
[75]

Maxim Tatarchenko, Stephan R Richter, René Ranftl, Zhuwen Li, Vladlen Koltun, and Thomas Brox.

What do single-view 3D reconstruction networks learn?

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3405–3414, 2019.

-
[76]

Jonathan Tremblay, Thang To, Balakumar Sundaralingam, Yu Xiang, Dieter Fox, and Stan Birchfield.

Deep object pose estimation for semantic robotic grasping of household objects.

arXiv preprint arXiv:1809.10790, 2018.

-
[77]

Jonathan Tremblay, Bowen Wen, Valts Blukis, Balakumar Sundaralingam, Stephen Tyree, and Stan Birchfield.

Diff-dope: Differentiable deep object pose estimation.

arXiv preprint arXiv:2310.00463, 2023.

-
[78]

Shaoxiong Wang, Mike Maroje Lambeta, Po-Wei Chou, and Roberto Calandra.

TACTO: A fast, flexible, and open-source simulator for high-resolution vision-based tactile sensors.

IEEE Robotics and Automation Letters (RA-L), 2022.

-
[79]

Shaoxiong Wang, Yu She, Branden Romero, and Edward Adelson.

GelSight Wedge: Measuring high-resolution 3D contact geometry with a compact robot finger.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA). IEEE, 2021.

-
[80]

Shaoxiong Wang, Jiajun Wu, Xingyuan Sun, Wenzhen Yuan, William T Freeman, Joshua B Tenenbaum, and Edward H Adelson.

3D shape perception from monocular vision, touch, and shape priors.

In Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 1606–1613. IEEE, 2018.

-
[81]

Benjamin Ward-Cherrier, Nicholas Pestell, Luke Cramphorn, Benjamin Winstone, Maria Elena Giannaccini, Jonathan Rossiter, and Nathan F Lepora.

The TacTip family: Soft optical tactile sensors with 3D-printed biomimetic morphologies.

Soft robotics, 5(2):216–227, 2018.

-
[82]

Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, and Stan Birchfield.

Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 606–617, 2023.

-
[83]

Youngsun Wi, Andy Zeng, Pete Florence, and Nima Fazeli.

Virdo++: Real-world, visuo-tactile dynamics and perception of deformable objects.

arXiv preprint arXiv:2210.03701, 2022.

-
[84]

Wonik Robotics.

Allegro Hand, 2023.

-
[85]

Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari.

Multiview compressive coding for 3D reconstruction.

arXiv:2301.08247, 2023.

-
[86]

Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox.

PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes.

arXiv preprint arXiv:1711.00199, 2017.

-
[87]

Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar.

Neural fields in visual computing and beyond.

In Computer Graphics Forum, volume 41, pages 641–676. Wiley Online Library, 2022.

-
[88]

Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin.

iNeRF: Inverting neural radiance fields for pose estimation.

In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1323–1330. IEEE, 2021.

-
[89]

Zhao-Heng Yin, Binghao Huang, Yuzhe Qin, Qifeng Chen, and Xiaolong Wang.

Rotating without seeing: Towards in-hand dexterity through touch.

arXiv preprint arXiv:2303.10880, 2023.

-
[90]

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.

PixelNeRF: Neural radiance fields from one or few images.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021.

-
[91]

Kuan-Ting Yu, John Leonard, and Alberto Rodriguez.

Shape and pose recovery from planar pushing.

In Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 1208–1215. IEEE, 2015.

-
[92]

Kuan-Ting Yu and Alberto Rodriguez.

Realtime state estimation with tactile and visual sensing: application to planar manipulation.

In Proc. IEEE Intl. Conf. on Robotics and Automation (ICRA), pages 7778–7785. IEEE, 2018.

-
[93]

Wenzhen Yuan, Siyuan Dong, and Edward H Adelson.

GelSight: High-resolution robot tactile sensors for estimating geometry and force.

Sensors, 17(12):2762, 2017.

-
[94]

Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong.

Faster segment anything: Towards lightweight sam for mobile applications.

arXiv preprint arXiv:2306.14289, 2023.

-
[95]

Jialiang Zhao, Maria Bauza, and Edward H Adelson.

FingerSLAM: Closed-loop unknown object localization and reconstruction from visuo-tactile feedback.

arXiv preprint arXiv:2303.07997, 2023.

-
[96]

Shaohong Zhong, Alessandro Albini, Oiwi Parker Jones, Perla Maiolino, and Ingmar Posner.

Touching a NeRF: Leveraging neural radiance fields for tactile sensory data generation.

In 6th Annual Conference on Robot Learning, 2022.

-
[97]

Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys.

NICE-SLAM: Neural implicit scalable encoding for SLAM.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786–12796, 2022.

## Acknowledgments

The authors thank Dhruv Batra, Theophile Gervet, Akshara Rai for feedback on the writing, and Wei Dong, Tess Hellebrekers, Carolina Higuera, Patrick Lancaster, Franziska Meier, Alberto Rodriguez, Akash Sharma, Jessica Yin for helpful discussions on the research.

Author contributions:

Sudharshan Suresh developed and implemented the core approach including tactile transformer, visual depth segmentation, neural SDF reconstruction, pose-graph optimization, performed full-stack tuning, worked on Allegro and DIGIT integration, TACTO and IsaacGym integration, camera and robot calibration, data collection, ground truth object scans, and live visualizations, conducted evaluations, made visuals, and wrote the paper.

Haozhi Qi designed and implemented in-hand object rotation policies and sim-to-real policy transfer, helped with Allegro and DIGIT integration, TACTO and IsaacGym integration, data collection, did code reviews and bug fixes, and helped edit the paper.

Tingfan Wu coordinated hardware and software systems integration, performed profiling of software stack, helped with Allegro and DIGIT integration, camera and robot calibration, ground truth object scans, and advised on evaluations.

Taosha Fan designed and implemented forward kinematics, helped implement visual depth segmentation, pose-graph cost functions and optimization, software systems integration, and advised on evaluations.

Luis Pineda implemented workflow for cluster deployment, streamlined development workflow, helped with modules that use Theseus, did code reviews and bug fixes, and advised on evaluations.

Mike Lambeta helped with Allegro and DIGIT integration, TACTO and IsaacGym integration, and hardware systems integrations.

Jitendra Malik advised on the project, gave feedback on approach, evaluations, and the paper.

Mrinal Kalakrishnan advised on the project, managed and supported researchers, gave feedback on approach, evaluations, and the paper.

Roberto Calandra advised on the project, helped with Allegro and DIGIT integration, TACTO and IsaacGym integration, gave feedback on approach, evaluations, and the paper.

Michael Kaess advised on the project, helped design pose-graph optimization, gave feedback on approach, evaluations, and the paper.

Joseph Ortiz advised on the project, co-developed the core approach, implemented volumetric ray sampling, SDF cost function, and 2D live visualizations, helped implement workflow for cluster deployment, streamlined development workflow, did code reviews and bug fixes, gave feedback on evaluations, designed visuals, and edited the paper.

Mustafa Mukadam set the vision and research direction, steered and aligned the team, provided guidance on all aspects of the project including core approach, systems, and evaluations, designed visuals, and edited the paper.

Funding: Sudharshan Suresh and Haozhi Qi acknowledge funding from Meta, and their work was partially conducted while at FAIR, Meta. Sudharshan Suresh was further partially supported by NSF grant IIS-2008279 while at CMU. Roberto Calandra acknowledge support from the German Research Foundation (DFG, Deutsche Forschungsgemeinschaft) as part of Germany’s Excellence Strategy – EXC 2050/1 – Project ID 390696704 – Cluster of Excellence “Centre for Tactile Internet with Human-in-the-Loop” (CeTI) of Technische Universität Dresden, and from Bundesministerium für Bildung und Forschung (BMBF) and German Academic Exchange Service (DAAD) in project 57616814 (School of Embedded and Composite AI, [SECAI](https://secai.org/)).

## Supplementary materials

-
•

Section [S1](#Sx3.SS1) to [S7](#Sx3.SS7)

-
•

Figure [S1](#Sx3.F1) to [S12](#Sx3.F12)

-
•

Multimedia on our webpage at [https://suddhu.github.io/neural-feels](https://suddhu.github.io/neural-feels)

### S1 Ground-truth shape and pose

![Figure](extracted/5308748/images/ground_truth.jpg)

*Figure S1: Object ground-truth with dual-camera infrared scanner. (a) Objects are placed on a turntable and scanned, followed by post-processing to ensure complete, accurate meshes. (b) Meshes visualized for the real and simulated FeelSight objects.*

Ground-truth object scans. Our results in Section [2](#S2) require ground-truth object shape to compare against. For this, we use a commercial dual-camera infrared scanner, the Revopoint POP 3 [[[59](#bib.bib59)]]. The hardware can scan objects from a close range with a minimum precision of $0.05\,\text{mm}$. Each real-world object is placed on a turntable and scanned while rotating about its axis (Figure [S1](#Sx3.F1) (a)). For object’s that lack texture, an artificial dot pattern is tracked by adding stickers. After generating the scans, we perform hole-filling for unseen regions of the object, like the bottom. Figure [S1](#Sx3.F1) (b) shows all the scanned meshes—a few meshes are directly sourced from the YCB [[[9](#bib.bib9)]] and ContactDB [[[7](#bib.bib7)]] datasets.

Pseudo ground-truth pose. In the real-world, we pass three RGB-D cameras as input into our pose tracking pipeline to use as a pseudo ground-truth estimate. This consists of three unique cameras (front left, back right, top down) with complementary but overlapping fields-of-view (Figure [S2](#Sx3.F2) and Figure [S3](#Sx3.F3) (b)). With this broad perspective of the scene, known shape from ground-truth scans, and the tracker running at $0.5$Hz, we can obtain an accurate estimation of object pose at each timestep.

![Figure](extracted/5308748/images/three_cam.jpg)

*Figure S2: Robot cell for pseudo-ground-truth tracking. Each of the three camera’s captures a different field-of-view of the interaction (left). For a pseudo-ground-truth, we pass the RGB-D stream from all three cameras into our pipeline, with known shape obtained from scanning. The output pose tracking (right) represents the ground-truth we compare to in the real-world results.*

![Figure](extracted/5308748/images/camera_viewpoint.jpg)

*Figure S3: (a) As a proof-of-concept, we assembled a minimal robot cell for live demonstrations of our method with one RGB-D camera and the robot policy deployed at $2$Hz. (b) The three different RGB-D camera viewpoints in our full robot cell used to collect FeelSight evaluation dataset. (c) Average pose error for known shape experiments based on camera viewpoint. We see that while the front and back cameras perform comparably, there are larger errors in the top-down camera as it is further away.*

### S2 Tactile transformer: data and training

![Figure](extracted/5308748/images/tactIle_transformer_training.jpg)

*Figure S4: Our tactile transformer is trained in simulation with real-world augmentation. (a) The tactile transformer is supervised from paired RGB-depth images rendered in TACTO [[[78](#bib.bib78)]]. (b) Each of these samples are generated from dense, random interactions with 40 different YCB objects. (c) In our training, we augment the data with background images collected from $25$ unique DIGIT sensors [[[40](#bib.bib40)]].*

Model architecture. Our model architecture is based on a monocular depth network, the dense prediction transformer (DPT) [[[58](#bib.bib58)]]. It comprises of a vision transformer (ViT) backbone that outputs bag-of-words features at different resolutions, finally combined into a dense prediction via a convolutional decoder. Compared to fully-convolutional methods, DPT has a global receptive field and the resulting embedding does not explicitly down-sample the image.

Training and loss metric.
Our image-to-depth training dataset comprises of 10K simulated tactile interactions each on the surface of $40$ YCB objects. We illustrate examples of the interactions in Figure [S4](#Sx3.F4) (b). We use the ADAM optimizer with momentum and a batch size of $100$, trained with mean-square depth reconstruction loss (Figure [S4](#Sx3.F4) (a)). We start with a pre-trained small ViT [[[17](#bib.bib17)]], with an embedding dimension of $384$ patch size of $16$. The dataloader splits the train, test, and validation data into 60%, 20%, and 20% respectively. To supplement our results in Section [4.6.2](#S4.SS6.SSS2), we visualize additional simulation results in Figure [S5](#Sx3.F5).

Data augmentation.
An important aspect of generalization and sim-to-real transfer is the augmentation applied during data collection and training. These include:

-
•

Real-world backgrounds. We compose simulated renderings with real-world background images, collected from $25$ different DIGIT sensors. These are shown in Figure [S2](#Sx3.SS2) (c).

-
•

Pose variations. Before rendering a sensor pose, we apply noise in rotation/translation and sensing normal direction. Additionally, we randomly vary the distance of penetration into the object surface.

-
•

Sensor lighting. We randomize position, direction and intensity of the three DIGIT LEDs.

-
•

Sensor pixel noise. We add Gaussian noise to RGB data, with a standard deviation of $7$px.

-
•

Standard transforms. Randomized horizontal flipping, cropping, and rotations of the tactile images.

![Figure](extracted/5308748/images/tactile_transformer_sim.jpg)

*Figure S5: Image to depth predictions by the tactile transformer on simulated contacts. Our tactile transformer shows good performance in simulated interactions, capturing both large contact patches, as well as smaller edge features. These objects are unseen during training—as highlighted in Section [4.6.2](#S4.SS6.SSS2), we demonstrate an average prediction error of $0.042\,\text{mm}$ on simulated test images.*

### S3 Additional implementation details

Segmented visual depth. As discussed in Section [4.6.1](#S4.SS6.SSS1), we use the 3D center of grasp, by computing the centroid of the end-effectors as a positive point prompt for SAM. However, in practice, this prompt alone doesn’t suffice. First, the robot fingers frequently appear in these segmentations, which is misleading to our shape optimizer. This is solved by adding negative point prompts to fingertip pixels that we obtain by projecting the forward kinematics results. We first verify if the fingertips are unoccluded by the object, which we do by comparing against the current rendered object model. Second, SAM tends to over segment objects with distinct parts (e.g. different faces of the Rubik’s cube). In case of these ambiguities, SAM outputs multiple masks, at different spatial scales. We apply a final pruning step to find the mask prediction closest to the average mask area we typically observe in simulation.

Shape optimizer. The neural field is optimized via
Adam [[[35](#bib.bib35)]] with learning rate of $2\text{e-}4$ and weight decay of $1\text{e-}6$. Instant-NGP uses a hash table of size $2^{19}$ for positional encoding, followed by a 3-layer MLP with $64$ dimensional width. We use a uniform random weights $\theta_{\text{init}}$ and initialize the SDF by running $500$ shape iterations using the first keyframe $\mathcal{K}_{0}$.

For evaluating the neural field we freeze the network and query a $200^{3}$ feature grid. The feature grid’s extents are defined as a bounding box of $15\,\text{cm}$ side, centered at the object’s initial pose $\mathbf{x}_{0}$. When training, we apply a series of bounding-box checks post hoc, to eliminate any ray samples $P_{u}$ found outside this bounding box. Mesh visualizations (Figure [4](#S2.F4)) are periodically generated via marching-cubes on the feature grid. We add color to the mesh by averaging the colored object pointcloud with a Gaussian kernel.

Pose optimizer. We use the vectorized $SE(3)$ pose graph optimizer in Theseus [[[55](#bib.bib55)]], with $20$ LM iterations of step size $1.0$. The keyframe window size $n\!=\!3$ and we run $2$ pose iterations for each shape iteration. The weighting factors for each loss are $w_{\text{sdf}}\!=\!0.01$, $w_{\text{reg}}\!=\!0.01$, and $w_{\text{icp}}\!=\!1.0$.

Compute and timings. All results in Section [2](#S2) are generated from playing-back the trials at a publishing rate of $1\,\text{Hz}$. Experimentally, however, we can run the pose optimizer at $10\,\text{Hz}$ and full backend at 5 Hz. Figure [S3](#Sx3.F3) (a) has a minimal robot setup of an online SLAM system with rotation policy in-the-loop. Experiments are run on an Nvidia GeForce RTX 4090, while the aggregate results are evaluated on a cluster with Nvidia Tesla V100s.

### S4 In-hand exploration policy

We first train a policy in simulation with access to an embedding of physical properties such as object position, size, mass, friction, and center-of-mass (denoted as $\mathbf{z}_{t}$). From the joint-angles $\mathbf{q}_{t}$ and this embedding $\mathbf{z}_{t}$, the policy outputs a PD controller target $\mathbf{a}_{t}\!\in\!\mathbb{R}^{16}$. The policy is trained in parallel simulated environments [[[45](#bib.bib45)]] using proximal policy optimization [[[62](#bib.bib62)]]. The reward function is a weighted combination of a rotational reward, joint-angle regularizer, torque penalty, and object velocity penalty. The resulting policy can adaptively rotate objects in-hand according to different physical properties.

During deployment, however, the policy does not have access to these physical properties. The estimator is instead trained to infer $\mathbf{z}_{t}$ from a history of proprioceptive states, which is in turn fed into the policy $\mathbf{\pi}_{t}$. A crucial change compared to Qi et al. [[[56](#bib.bib56)]] is that we train the policy to rotate objects with DIGIT sensors on the distal ends (Figure [1](#Sx1.F1)). This results in different gaits, as it (i) relies on finger-object friction instead of gravity, and (ii) learns to maintain contact with the DIGIT gelpads.

### S5 Additional results

![Figure](extracted/5308748/images/metrics_over_time.jpg)

*Figure S6: Shape and pose metrics over time for in-hand SLAM. Here, we plot the time-varying metrics for experiments visualized in Figure [4](#S2.F4). First, we note the gradual increase in F-score over time with further coverage. Additionally, we have bounded pose drift over time—for each experiment we omit the first five seconds as the metric is ill-defined then.*

Shape and pose metrics over time. In Figure [S6](#Sx3.F6), we plot these metrics for each of the experiments in Figure [4](#S2.F4), instead against $0\!-\!30\,\text{sec}$ timesteps. For shape, we observe gradual convergence to an asymptote close to $1.0$, indicating evolution of both shape completion and refinement over time. Also visualized here is the precision and recall metrics over time, whose harmonic mean represents the F-score. For pose, we observe stable drift over time, indicating the estimated object pose lies close to the ground-truth estimate.

Effect of camera viewpoint in the real-world. In Section [2.4](#S2.SS4), we establish the relationship between occlusion/sensing noise and pose error. Here, we run additional experiments, on a limited set of viewpoints in the real-world. Figure [S3](#Sx3.F3) (b) shows the RGB-D data from three cameras front left, back right, top down, at distances of $27\,\text{cm}$, $28\,\text{cm}$, and $49\,\text{cm}$ respectively from the robot. We run our vision-only pose tracker with known shape using each of three cameras over all $5$ Rubik’s cube rotation experiments and plot the average metrics in Figure [S3](#Sx3.F3) (c). We observe that the front left and back right viewpoints result in lowest average pose error due to their closer proximity. The top down camera gives less reliable depth measurements and segmentation output, leading to almost $2$x greater pose error.

![Figure](extracted/5308748/images/per_object_slam_error.jpg)

*Figure S7: Pose (left) and shape (right) metrics for each object class, sorted in best-to-worst performance.*

Class-specific metrics. In Figure [S7](#Sx3.F7), we present our metrics for the SLAM results in Section [2.2](#S2.SS2), dividing based on object class. This helps us make some assessments on how object geometry and scale can affect our results. Some observations include:

-
•

Object symmetry. Objects with symmetries about their rotation axis are challenging for our depth-based estimator. This leads to higher pose errors for the peach and pear, for example.

-
•

Object visibility. Partial visibility of the large objects, such as the pepper grinder, affect the completeness of the reconstructions. Touch in this case is not advantageous since the finger gait does not span the length of the object to provide coverage.

-
•

Object scale. Smaller-sized objects, such as the peach, may demonstrate better shape metrics as their scale is closer to the F-score threshold of $5\,\text{mm}$.

### S6 Additional visualizations

All experiments from the FeelSight dataset. In Figure [S8](#Sx3.F8) we illustrate all of the $70$ visuo-tactile experiments that comprise our dataset. While both simulation and real data collection use the proprioception-driven policy [[[56](#bib.bib56)]], the policy generalizes better in simulation across the class of objects. Some objects in the real-world require a human-in-the-loop to assist with in-hand rotation; e.g. supporting cube-shaped objects from the bottom to occasionally prevent falling out of hand.

*Figure S8: A collage depicting the entirety of the FeelSight dataset. We collect (i) 5 sequences each (row) in the real-world across 6 different objects (column), and (ii) 5 sequences each (row) in simulation across 8 different objects (column).*

Additional neural tracking visualizations. Figure [S9](#Sx3.F9) shows rendering results from the experiments in Section [2.3](#S2.SS3) along with the pose axes. We see good alignment of the renderings when overlaid on the RGB camera frame.

Further visual segmentation results. Figure [S10](#Sx3.F10) shows additional qualitative results of visual segmentation for (a) real-world and (b) simulated rotations sequences.

![Figure](extracted/5308748/images/pose_tracking_suppl.jpg)

*Figure S9: Further visualizations of neural tracking experiments. These qualitatively complement the results from Section [2.3](#S2.SS3) for both (a) simulated and (b) real-world experiments.*

![Figure](extracted/5308748/images/sam_suppl.jpg)

*Figure S10: Additional results on visual segmentation. Our segmentation module can accurately singulate the in-hand object in both (a) real-world and (b) simulated image sequences.*

### S7 Illustrating the role of touch

Sensor coverage visualized in SLAM. To illustrate the complementary nature of touch and vision, we color the reconstructed mesh regions based on their dominant sensing modality in Figure [S11](#Sx3.F11). After running the SLAM experiments in Section [2.2](#S2.SS2), we first run marching-cubes on the final neural SDF. In the resultant mesh, we assign each vertices color based on if vision or touch is the nearest pointcloud measurement to it. In the case where there is no vision or touch pointcloud within a $5\,\text{mm}$ radius, it is assigned as a hallucinated vertex. This is a demonstrable advantage of neural SDFs, where the network can extrapolate well based on information in the neighborhood of the query point. From the meshes in Figure [S11](#Sx3.F11) we see that while vision gets broad coverage of each object, there is considerable tactile signal from the interaction utilized for shape estimation.

![Figure](extracted/5308748/images/sensor_meshes.jpg)

*Figure S11: Sensor coverage illustrated in final mesh reconstructions of select objects—indicating vision, touch, and hallucinated regions.*

![Figure](extracted/5308748/images/tactile_features.jpg)

*Figure S12: Six examples of tactile images compared against the neural field. We see that our tactile pose optimizer matches the predicted local geometry with the neural surface rendering. Thus, patches and edges predicted by touch appear in the rendering as well.*

Touch aligns local geometries with predicted depth. As described in Section [4.7.2](#S4.SS7.SSS2), the pose optimizer inverts the neural field to back-propagate a loss in pose space [[[88](#bib.bib88), [69](#bib.bib69), [97](#bib.bib97)]]. This has been illustrated in work such as iNeRF [[[88](#bib.bib88)]], where the rendered neural field attempts to match the image measurements via updates to the se(3) Lie algebra of the camera pose. As our framework leverages the idea that vision-based touch is just another perspective camera, we show how the rendered neural field matches with tactile depth features in Figure [S12](#Sx3.F12).

Each RGB image is first passed through the tactile transformer (Section [4.6.2](#S4.SS6.SSS2)) to output a predicted tactile depthmap. Our pose optimizer aligns the neural rendering of the surface with the measured depthmap, based on 3D samples from the measured depthmap. Thus we can see that both in simulation (Figure [S12](#Sx3.F12) (a)) and the real-world (Figure [S12](#Sx3.F12) (b)), the edge and patch features predicted match well with the rendered object.

Generated on Wed Dec 20 22:11:05 2023 by [LATExml](http://dlmf.nist.gov/LaTeXML/)