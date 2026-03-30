[[1]\fnmRadu Tudor \surIonescu 1]\orgdivDepartment of Computer Science, \orgnameUniversity of Bucharest, \orgaddress\street14 Academiei, \cityBucharest, \postcode010014, \countryRomania 2]\orgdivApplied AI Team, \orgnameAmazon, \citySeattle, \countryUSA 3]\orgdivDepartment of Information Engineering and Computer Science, \orgnameUniversity of Trento, \orgaddress\street9 via Sommarive, \cityPovo-Trento, \postcode38123, \countryItaly # Masked Image Modeling: A Survey \fnmVlad \surHondru \fnmFlorinel Alin \surCroitoru \fnmShervin \surMinaee raducu.ionescu@gmail.com](mailto:raducu.ionescu@gmail.com)


\fnmNicu \surSebe

[

[

[

###### Abstract

In this work, we survey recent studies on masked image modeling (MIM), an approach that emerged as a powerful self-supervised learning technique in computer vision. The MIM task involves masking some information, e.g. pixels, patches, or even latent representations, and training a model, usually an autoencoder, to predicting the missing information by using the context available in the visible part of the input. We identify and formalize two categories of approaches on how to implement MIM as a pretext task, one based on reconstruction and one based on contrastive learning. Then, we construct a taxonomy and review the most prominent papers in recent years. We complement the manually constructed taxonomy with a dendrogram obtained by applying a hierarchical clustering algorithm. We further identify relevant clusters via manually inspecting the resulting dendrogram. Our review also includes datasets that are commonly used in MIM research. We aggregate the performance results of various masked image modeling methods on the most popular datasets, to facilitate the comparison of competing methods. Finally, we identify research gaps and propose several interesting directions of future work. We supplement our survey with the following public repository containing organized references: [https://github.com/vladhondru25/MIM-Survey](https://github.com/vladhondru25/MIM-Survey).

###### keywords:

masked image modeling, masked autoencoders, self-supervised learning.

## 1 Introduction

*Figure 1: A timeline with the most prominent works in Masked Image Modeling. This timeline illustrates the evolution of MIM methods and the primary research directions in this area. Early works adapted masked language modeling directly to the image domain, as in MST [[[1](#bib.bib1)]] and SimMIM [[[2](#bib.bib2)]]. Subsequent efforts [[[3](#bib.bib3), [4](#bib.bib4)]] tried to combine the MIM ideas with another successful self-supervised method applied in computer vision, namely contrastive learning. The most recent developments extend MIM to multimodal settings [[[5](#bib.bib5), [6](#bib.bib6)]] and tackle more complex tasks [[[7](#bib.bib7), [8](#bib.bib8)]].*

Data samples have always played a crucial role in training large deep neural networks. Procuring a curated labeled dataset not only involves a great effort and a laborious process from a team of human annotators, but it also represents a significant expense. As a result, various self-supervised learning strategies have been explored, where the model is pre-trained with a different objective, which does not require human labels. Self-supervised learning can help the model to learn a rich feature representation, and even surpass supervised alternatives [[[9](#bib.bib9)]]. Then, during the fine-tuning phase, the model is further optimized for a specific downstream task. For example, in most image classification tasks, a common approach is to train the network on the ImageNet dataset [[[10](#bib.bib10), [11](#bib.bib11)]], and then change the classification head for a new task and train the resulting model on a relevant dataset.

Self-supervised learning is a popular method to pre-train a deep learning model. It involves creating a supervised setting without annotating the data, but rather using the inherent structure of the data. Due to its potential, self-supervised pre-training algorithms have been rapidly adopted in computer vision in recent years.
The early works in this direction by [[[12](#bib.bib12)]] and [[[13](#bib.bib13)]] were inspired by the jigsaw puzzle idea, in which the authors proposed to split the image into patches and estimate the position of each patch. Another pre-training strategy was introduced by [[[14](#bib.bib14)]], where the objective was to have the distance between the initial and last frame in a video smaller than the distance between the initial frame and a random frame from another video. [[[15](#bib.bib15)]] created a segmentation map from the motion of objects in videos, and then applied a segmentation task for pre-taining, using the artificial segmentation map as ground-truth. Other prominent pretext tasks are colorizing a gray-scale image [[[16](#bib.bib16)]] or rotating the input image and estimating the angle it was rotated with [[[17](#bib.bib17)]].

The idea of pre-training a model by masking a part of the input and then predicting the masked information gained traction with the introduction of Bidirectional Encoder Representations from Transformers (BERT) [[[18](#bib.bib18)]], which brought significant advancements in the Natural Language Processing domain. The main advantage was that huge amounts of unlabeled and unstructured text data could be used. Notably, this pretext strategy was applied in vision problems earlier, by [[[19](#bib.bib19)]] and [[[20](#bib.bib20)]], where the original input signal was altered (corrupted or obscured) and then reconstructed using an architecture based on convolutional layers. The latter work even employed an encoder-decoder model to inpaint the missing regions from the input image. Nevertheless, as presented by [[[9](#bib.bib9)]], there are some differences from the recent Masked Image Modeling (MIM) literature, as the aforementioned papers framed the problem as a denoising task.

In Figure [1](#S1.F1), we present a timeline with the most prominent works in masked image modeling. At ICCV 2021, [[[21](#bib.bib21)]] presented a self-supervised learning method that applies a contrastive loss between a masked patch from an image and other regions. At NeurIPS 2021, [[[1](#bib.bib1)]] proposed a teacher-student framework that employs both a reconstruction and a contrastive objective. At CVPR 2022, [[[9](#bib.bib9)]] and [[[2](#bib.bib2)]] introduced a pre-training framework that involves masking a high portion of the input and reconstruct it using autoencoders based on the Vision Transformer (ViT) architecture [[[22](#bib.bib22)]]. Their concurrent studies represent the base for the research that followed. Following [[[1](#bib.bib1)]], [[[23](#bib.bib23)]] formally presented a pre-training method that involves both objectives, but follows the masked autoencoder (MAE) framework. Besides employing MAE for pre-training, [[[24](#bib.bib24)]] demonstrated how to use MAE at inference time, improving performance on downstream tasks. At ICLR 2023, [[[3](#bib.bib3)]] combined MIM with denoising contrastive learning for a better feature learning, while [[[25](#bib.bib25)]] analyzed the teacher-student MIM framework and showed the advantages of updating the teacher’s weights as an exponential moving average of the student’s. [[[4](#bib.bib4)]] is another important stepping stone in integrating both reconstruction and contrastive objectives in the MAE framework. At NeurIPS 2023, [[[26](#bib.bib26)]] presented a method that applies the masking pre-training method on multiple input vision modalities, as well as text. More recent contributions are focused on multimodal settings [[[5](#bib.bib5), [6](#bib.bib6)]] and more complex tasks [[[7](#bib.bib7), [8](#bib.bib8)]].

Within the context of more complex architectures of the latest neural networks and the large quantities of annotated data they require, pre-training such models has started to become a prerequisite. Masked Image Modeling represents a pretext task that consists of masking some information from the input (either the raw signal or some features obtained from it), and then estimating an output that should be the same as if the input was unaltered, or even predicting the original input. This pre-training strategy has quickly become popular, especially since the mechanism is easily implemented with the well-known transformer architecture, and thus it emerged in many domains and tasks. As a result, it is very difficult and time-consuming to study such a high number of research papers and find the necessary information. Our work aims to mitigate this challenge and facilitate further research or industrial endeavors. Firstly, we present a generic framework that all masked image modeling methods follow and identify two different categories of approaches: one involving input reconstruction and one employing a contrastive objective. Furthermore, we have carefully reviewed the most recent papers and extracted the main ideas and contributions from these studies. Furthermore, we manually organize the reviewed studies into a taxonomy based on multiple criteria, which is complemented by a dendrogram obtained via hierarchical clustering.

Given the increasing prominence of masked image modeling and the corresponding rise in research publications on this topic, several studies have been conducted with objectives similar to ours: to facilitate the literature review process. Among these, the survey on masked modeling by [[[27](#bib.bib27)]] stands out as a notable contribution, with which our paper shares many characteristics, such as the general frameworks employed during pre-training, some criteria of the manual taxonomy, or even some presented papers. Nevertheless, our survey is focused on vision and how MIM is applied in the most recent techniques. The work of [[[27](#bib.bib27)]] surveys a broad range of domains, while including a limited number of papers per domain. Furthermore, the more focused scope of our survey on the image domain allows us to review a greater number of vision papers for a more thorough study. Thus, we consider our work to be more comprehensive, particularly in the computer vision domain. In contrast to [[[27](#bib.bib27)]], we organize the papers via both manual and automatic clustering, providing complementary ways to categorize the surveyed papers. Other relevant surveys are the ones conducted by [[[28](#bib.bib28)]] and [[[29](#bib.bib29)]], where the authors focus specifically on masked autoencoders. Unlike these related surveys, we address the broader domain of masked image modeling, which is not necessarily coupled with autoencoders. Our survey is therefore more comprehensive, containing more than 100 additional references compared with both surveys on masked autoencoders [[[28](#bib.bib28), [29](#bib.bib29)]]. For example, we review works that integrate masking into basic operations, e.g. convolutions [[[30](#bib.bib30), [31](#bib.bib31)]]. As shown by the respective authors, masked convolutions can be integrated into any architecture.

To highlight the aim of our survey, we sum up our contributions as follows:

-
•

We highlight two categories of approaches on how to implement masked image modeling as a pretext task.

-
•

We review the most prominent papers in recent years, and construct a taxonomy that facilitates studying the related literature.

-
•

We apply a hierarchical clustering algorithm on the abstracts and identify relevant clusters via manually inspecting the resulting dendrogram.

-
•

We review commonly used datasets and aggregate the results of various masked image modeling methods in a single place, to facilitate the comparison of competing methods.

-
•

We identify research gaps and propose several interesting directions of future work in the area of masked image modeling.

## 2 Generic Framework

Masked Image Modeling is an unsupervised technique that is usually applied during the pre-training phase. It involves masking some information, either from the input or from the latent representation, and then estimating the original data, as if the data would not have been concealed. Although many masked image modeling techniques have been proposed, the research has been focused on two main schemes, either reconstructing the masked signal, or comparing two latent representations, one for the unaltered input signal and one for the masked input. On a few occasions, different approaches have been explored, but they are built on similar grounds. Therefore, in the following subsections, we aim to give a general formulation of the first two aforementioned schemes.

![Figure](x2.png)

*Figure 2: Reconstruction-based MIM pipeline. The input image is split into patches. Some of the resulting patches are masked, and the remaining patches are passed through an encoder. Next, latent vectors corresponding to masked and visible patches are passed through a decoder. Finally, a reconstruction loss is computed between the output patches and the original input patches. The whole purpose of this self-supervised pipeline is to generate a robust latent representation by learning to reconstruct masked patches. Best viewed in color.*

### 2.1 Reconstruction

The first scheme that we identified revolves around the idea of masking some piece of information at any stage during the forward pass of the model, and then employing a decoder to reconstruct the missing data. We illustrate the reconstruction framework in Figure [2](#S2.F2). Typically, the input tokens in this pipeline correspond to patches of raw pixels [[[9](#bib.bib9), [2](#bib.bib2)]]. However, there are scenarios in which these tokens reside in a latent space [[[32](#bib.bib32), [33](#bib.bib33)]]. In Figure [2](#S2.F2), we illustrate this optional latent space projection through an image tokenizer, which can vary in complexity. For instance, [[[32](#bib.bib32)]] employed a VQ-GAN [[[34](#bib.bib34)]] tokenizer to transform images into sequences of semantic visual tokens. We emphasize that all recent advances [[[35](#bib.bib35), [36](#bib.bib36), [37](#bib.bib37), [38](#bib.bib38)]] on image tokenizers can be easily integrated into our generic formulation.

The reconstruction-based pre-training framework was concurrently introduced by [[[9](#bib.bib9)]] and [[[2](#bib.bib2)]]. Both studies employ an encoder based on the ViT architecture [[[22](#bib.bib22)]], in which a significant portion of the input tokens is masked. The encoder processes only the visible tokens. After the encoding stage, the masked tokens are replaced with a special mask token, and a decoder reconstructs their original content. In the Masked Autoencoder (MAE) proposed by [[[9](#bib.bib9)]], the decoder is lighter than the encoder, which results in an asymmetric architecture. The loss function is applied only to the output corresponding to the masked tokens.

After pre-training, the resulting encoder can be repurposed for feature extraction in downstream tasks, thanks to its strong representational capacity. Moreover, the decoder can be employed for generation, as demonstrated by [[[32](#bib.bib32)]] and [[[33](#bib.bib33)]]. The generative process involves iterative model evaluations and begins with an image in which all tokens are initially masked. During each iteration, some of the masked tokens are replaced with predictions from the model. This process continues for a fixed number of steps, with the final iteration replacing any remaining masked tokens with the model’s predictions, ultimately producing a complete image.

*Algorithm 1 Reconstruction-based MIM*

We formally present the reconstruction-based training strategy in Algorithm [1](#alg1). The first step of the algorithm splits the input images into non-overlapping patches, resulting in a set $P$ that contains patches of the same size, namely of $h\times w$ pixels. The second step applies the tokenizer $T_{\omega}$, which can differ from one method to another. Notably, $T_{\omega}$ can be represented by the identity function when the MIM pre-training is directly applied on the raw pixels. The masking operation indicates which patches should be kept and which should be masked. The indexes of the visible and masked patches are stored in $I_{v}$ and $I_{m}$, respectively. In the next steps, the masked patches are usually dropped, and only the visible patches are processed by an encoder ($E_{\theta}$) that extracts a latent representation. Before the decoding step, the previously masked patches are replaced by a learnable representation ($M$), which resides in the latent space of the encoder. These transformations correspond to steps 3-8 in Algorithm [1](#alg1).

Using the sequence formed by concatenating the masked representations with those from the encoder, the decoder ($D_{\phi}$) reconstructs the patches as depicted in step 9 of Algorithm [1](#alg1). Finally, in steps 10-13 of Algorithm [1](#alg1), a distance function is optimized by updating the encoder, the decoder, the projection layer, and the learnable masked representation.

![Figure](x3.png)

*Figure 3: Contrastive-based MIM pipeline. Two versions of the input image are used in this framework, one that is unaltered (or weakly augmented) and one that is strongly augmented and masked. The images are processed by two encoders, a teacher encoder (left) and student encoder (right). The teacher encoder is either identical to the student encoder, or an exponential moving average (EMA) of the student encoder. The training is based on a contrastive loss applied on the latent representations of the patches. Gradients are propagated only through the student encoder. Best viewed in color.*

### 2.2 Contrastive

The second generic scheme is represented by comparing two different latent representations of the same input. One latent representation corresponds to an unaltered or weakly augmented input image, while the other corresponds to a masked and strongly augmented version of the same input image. The approach is based on a contrastive learning framework, as illustrated in Figure [3](#S2.F3). There are two common architectural configurations. One configuration uses an encoder with shared weights, as in [[[39](#bib.bib39), [40](#bib.bib40)]]. The other configuration uses an encoder for the masked input and an exponential moving average (EMA) version of the encoder for the original input, as in [[[25](#bib.bib25)]].

*Algorithm 2 Contrastive-based MIM*

The generic contrastive-based MIM framework is formalized in Algorithm [2](#alg2). The aim of this framework is to obtain a similar embedding, irrespective of the applied masking. Steps 1 and 2 of the algorithm generate two versions of the input image by augmenting them at different intensities. The version that undergoes masking is strongly augmented, while the other one is weakly augmented (or can even remain unaltered). In steps 3-4, each image is divided into non-overlapping patches, and in step 5, masking is applied solely to the strongly augmented image. The unmasked image undergoes processing by the projection layer and the encoder (step 7), whereas the masked image has its omitted patches replaced with a learnable vector before being encoded (steps 8-11). Notably, Algorithm [2](#alg2) processes both input images using the same encoder, which can be regarded as two encoders with shared weights. Gradient propagation is restricted to the processing of the masked image. An alternative approach [[[25](#bib.bib25)]] is to process the unmasked image with an EMA-based encoder in steps 9 and 11. Step 12 of the algorithm computes the contrastive loss, in which the negative patches originate from different positions within the same image. The most common loss function used in Step 12 is some variant of the InfoNCE (Noise-Contrastive Estimation) loss:

$$\mathcal{L}\left(\varphi,\!\theta,\!M\right)\!\leftarrow\!-\!\sum_{i=1}^{n}\!% \log{\frac{\exp(\langle\hat{H}[i],\tilde{H}[i]\rangle/\tau)}{\sum_{j=1}^{n}% \exp(\langle\hat{H}[j],\tilde{H}[i]\rangle/\tau)}},$$ \tag{1}

where $\tau$ is the temperature factor scaling the pairwise similarity. The goal of InfoNCE is to maximize the similarity between $\hat{H}[i]$ and $\tilde{H}[i]$, while minimizing the similarities $\hat{H}[j]$ and $\tilde{H}[i]$, $\forall j\neq i$. In practice, the negative patches denoted by $\hat{H}[j]$ can also be sourced from other images. Finally, the last steps (13-15) of the algorithm update the weights of the encoder, the projection layer, and the learnable representation $M$.
Although the contrastive learning method is technically different from the reconstruction-based scheme, [[[41](#bib.bib41)]] demonstrated that the contrastive approach strongly correlates with the reconstruction-based framework in terms of the learned latent representations.

![Figure](x4.png)

*Figure 4: A multi-level classification of Masked Image Modeling papers into various categories, based on the research directions studied in the respective papers. References are clickable links to papers. Best viewed in color.*

### 2.3 Relation to Other SSL methods

Contrastive learning. The standard contrastive learning paradigm [[[42](#bib.bib42), [43](#bib.bib43), [44](#bib.bib44)]] is to learn representations by pulling different augmented views of the same image closer in the embedding space, while pushing views of different images apart. In Section [2.2](#S2.SS2), we show that MIM and contrastive learning are not mutually exclusive. However, if we compare the standard contrastive learning with MIM, we can identify a few different properties.
First, MIM is naturally well-suited to transformer architectures, as it draws inspiration from masked language modeling, which was originally designed for such models. Second, unlike contrastive learning, MIM supports both recognition and generation tasks, as discussed in Section [2.1](#S2.SS1). In terms of representation learning, MIM encourages the model to focus on spatially meaningful features, because it reconstructs masked regions of the input. Lastly, contrastive learning relies on negative examples, which increases the batch size and often requires hard sample mining, a procedure that introduces additional computational overhead and complexity during training [[[45](#bib.bib45)]].

Non-contrastive methods. Early self-supervised learning approaches primarily focused on creating pretext tasks, e.g. solving jigsaw puzzles [[[13](#bib.bib13), [46](#bib.bib46)]], learning the arrow of time [[[47](#bib.bib47)]], or colorizing images [[[16](#bib.bib16)]]. While conceptually simple, these methods often exhibit limited generalization, because the objectives of the pretext tasks are not always aligned with those of downstream tasks. More recent studies [[[48](#bib.bib48), [49](#bib.bib49), [50](#bib.bib50)]] leverage semantic similarities between samples. The core idea is to employ a dual-network architecture, where both networks are trained to generate similar embeddings for different augmented views of the same image. Notably, this approach differs from contrastive learning, because it eliminates the need for negative examples.

Compared with MIM, the aforementioned self-supervised approaches differ in several key aspects. First, they can be less stable during training and often rely on additional techniques, such as momentum encoders, to ensure convergence. Second, unlike alternative methods, MIM inherently supports image generation. Lastly, while MIM focuses on learning spatial and contextual features through reconstruction, other self-supervised learning methods prioritize strong global representations by enforcing consistency across different views of the same image.

## 3 Taxonomy and Overview

In Figure [4](#S2.F4), we present a manually-generated taxonomy of the most promising MIM papers, organizing them according to their main contributions. In constructing the taxonomy, we consider six main research directions related to: the masking strategy, the type of masked signals, the neural architecture, the objective function, the type of downstream tasks, and theoretical results. Nonetheless, since many papers focused on more than one of the above aspects, we grouped the respective works into distinct categories representing composed contributions. We now continue by presenting the aforementioned papers, divided into sub-sections according to our taxonomy.

### 3.1 Masking Strategy

![Figure](x5.png)

*Figure 5: Types of masking strategies employed in MIM pipelines. The masking strategies can either rely on random patch selection or leverage some additional information to selectively mask the salient patches. Additional information can come from auxiliary neural networks or from features derived through classical computer vision algorithms.*

In Figure [5](#S3.F5), we categorize the main masking strategies employed in MIM pipelines. A common practice is to mask the content randomly [[[2](#bib.bib2), [9](#bib.bib9), [51](#bib.bib51), [52](#bib.bib52), [53](#bib.bib53), [54](#bib.bib54), [55](#bib.bib55)]]. Nevertheless, several recent works advocate for a more informed masking [[[56](#bib.bib56), [57](#bib.bib57), [58](#bib.bib58), [59](#bib.bib59), [60](#bib.bib60), [61](#bib.bib61), [62](#bib.bib62), [63](#bib.bib63), [64](#bib.bib64), [65](#bib.bib65), [66](#bib.bib66)]], guided by meaningful information, whether it stems from a neural network [[[57](#bib.bib57), [58](#bib.bib58), [67](#bib.bib67)]] or from classical features [[[56](#bib.bib56), [59](#bib.bib59), [60](#bib.bib60), [68](#bib.bib68)]]. By masking the salient patches or semantically relevant regions, these informed approaches enhance the capacity of the model to learn robust representations.

![Figure](extracted/6612948/fig_simMIM.jpg)

*Figure 6: SimMIM pipeline, courtesy of [[[2](#bib.bib2)]] (image licensed under CC BY 4.0). SimMIM is a reconstruction-based method (belonging to the generic pipeline in Figure [2](#S2.F2)), as it employs an ${L}^{1}$ loss between the masked patches and their counterparts from the original image.*

[Xie et al. [[2](#bib.bib2)]] introduced SimMIM, a self-supervised pre-training framework that reconstructs pixel values of randomly masked images. An overview of the method is depicted in Figure [6](#S3.F6). The model is a simple encoder (ViT) receiving as input an image with masked patches replaced by a learnable token. Additionally, the study motivates the choice of random masking and the raw pixels as target features through extensive ablation studies.

![Figure](x6.png)

*Figure 7: MAE pipeline, courtesy of [[[9](#bib.bib9)]] (image licensed under CC BY 4.0). MAE is a reconstruction-based method (belonging to the generic pipeline in Figure [2](#S2.F2)), as it employs an ${L}^{2}$ loss between the reconstructed patches and their counterparts from the original image.*

[[[9](#bib.bib9)]] presented the masked autoencoder (MAE) framework, where the main contribution is to exclude the masked tokens from the encoder’s input and use a very high masking ratio (75%). These changes imply a more efficient framework compared with other works, such as SimMIM [[[2](#bib.bib2)]]. However, as a negative effect, having an encoder that operates only on visible tokens makes the framework incompatible by default with the Hierarchical Vision Transformer [[[69](#bib.bib69)]] architecture, which usually performs better than a standard ViT. The MAE pipeline is presented in Figure [7](#S3.F7). Aiming to learn more robust representations,
OmniMAE, introduced by [[[54](#bib.bib54)]], is similar to the standard MAE framework, except that the model is trained with both video and image data. Another extension is MixMAE [[[55](#bib.bib55)]], a pre-training approach that leverages elements of both MAE [[[9](#bib.bib9)]] and SimMIM [[[2](#bib.bib2)]]. This method selects two distinct images from a training dataset and extracts random tokens from each. The tokens are then amalgamated to form a new, hybrid image. This composite image undergoes processing by an encoder. Further, a decoder is employed, aiming to reconstruct the original two images from which the mixed image was derived. During the decoding phase, the input provided to the decoder is unmixed and the patches corresponding to the missing tokens in the hybrid image are filled with masked tokens.

[[[56](#bib.bib56)]] adapted the masked pre-training strategy to 3D Point Clouds. The method groups the points into patches and masks some of them. Then, it embeds and encodes only the unmasked patches. Next, the visible patches get concatenated with the masked tokens and passed through the decoder, with the objective to reconstruct the latter. The authors state that passing the masked patches earlier leaks spatial information which simplifies the task. Additionally, [[[64](#bib.bib64)]] showed that masked image modeling is more effective on 3D scenes when certain points are excluded from masking, the so-called Informative Points*. Keeping these points unchanged will help to preserve the geometric structure of the scene after the masking process.

A number of studies used low-level information to improve the masking strategy.
[[[68](#bib.bib68)]] formulated the masking strategy as as curriculum learning problem by gradually increasing the masking ratio during training. Additionally, they introduced a patch selection strategy that emphasizes masking patches with higher gradient magnitudes. The masking strategy proposed by [[[70](#bib.bib70)]] is inspired by the color noise used in image processing. Specifically, their approach introduces four filters that operate on the uniform noise typically generated for random masking. This strategy avoids the extra computational overhead often required by other informed masking methods, while still outperforming standard random masking. Rather than imposing a masking policy, several studies tried to learn the optimal masking strategy. SemMAE [[[58](#bib.bib58)]] deployed an additional stage before masking. This stage, called *semantic part learning*, is responsible for learning attention maps that correspond to meaningful semantic components in the image. The training of this stage is performed by embedding the class token provided by a ViT encoder into part embeddings. The resulting embeddings together with the patch embeddings provided by the same encoder are used in an attention layer. Finally, the obtained attention maps are processed by a decoder to reconstruct the original image. In the second stage, the attention maps are used for part segmentation. The masking varies from masking patches in each part to masking entire parts randomly. In a similar fashion, [[[57](#bib.bib57)]] proposed a framework in which the masking strategy is learned. A teacher network, seeing the intact image, generates a mask that is used to pre-train the student. The objective is to reconstruct the feature representation of the teacher for the masked tokens, as well as the class token *[CLS]* that is used in generating the mask. The teacher’s parameters are updated using an exponential moving average of the student’s weights. The experiments demonstrated the superiority of the method over other masking strategies. Instead of relying on an additional teacher network, [[[63](#bib.bib63)]] developed a masking strategy based on the direct feedback provided by the model being pre-trained. The strategy computes the reconstruction loss for each patch and then selectively masks the patches that are more challenging to reconstruct, as indicated by their higher loss values. This approach ensures that the model focuses on the most difficult aspects of the data during training. Similarly, but leveraging attention maps, [[[71](#bib.bib71)]] proposed a guided masking strategy for MAE, rather than a random policy. Periodically, all images are passed through the encoder and their latent feature representation of the *[CLS]* token from the last attention layer is extracted in order to compute an importance map of all patches. Using this, the masked patches are sampled (higher chance for more salient patches) and then estimated via reconstruction, while a portion of the least important patches are completely put aside (i.e. not fed to the encoder).

[[[60](#bib.bib60)]] added masked image modeling in the training pipeline of GANs. The masking is based on two methods. The first one is called *shifted spatial masking* and constitutes random masking of the image. The second one is called *balanced spectral masking* and randomly masks some spectral bands of the image decomposed in the spectral space. The authors observed that these two strategies are orthogonal and help the adversarial training process to become more stable. The frequency domain is also leveraged by [[[72](#bib.bib72)]], who proposed to apply the masks in the frequency domain. The images are converted with fast Fourier transform, then either low or high frequencies are masked, and mapped back into the pixel space. Using an encoder-decoder model, the original image is estimated and transformed into the frequency domain. The objective is to identify the frequencies that were masked.

The work of [[[51](#bib.bib51)]] pioneered the application of Masked Autoencoders (VideoMAE) to video data, adapting the principles of MAE used in image processing to the temporal domain. In this approach, sequences of frames are masked with a masking ratio of approximately 90%. This notably high ratio is strategically chosen to effectively minimize the issue of information leakage that occurs between closely spaced frames. [[[73](#bib.bib73)]] extended this work and introduced VideoMAEv2, a method for scaling up the original VideoMAE. Other studies [[[74](#bib.bib74), [65](#bib.bib65), [75](#bib.bib75)]] focused on designing more suitable masking strategies for video data. [[[74](#bib.bib74)]] argued that, for the successful application of MAE on video, it is crucial to consistently mask video segments across time. Without this consistency, there is a risk of temporal information leakage, rendering the learning task overly simplistic. To address this challenge, they introduced a novel solution that leverages optical flow techniques to generate time-coherent masking volumes. These volumes are then utilized to selectively sample visible tokens, ensuring that the masking process maintains temporal integrity and effectively prevents information leakage. [[[65](#bib.bib65)]] proposed DropMAE, an adaptation of MAE for videos. The main observation of [[[65](#bib.bib65)]] is that it is not enough to mask and reconstruct video patches in order to learn spatio-temporal features. The proposed solution is to drop spatial-attention tokens to guide the model towards looking at the temporal information. Driven by the same goal, [[[75](#bib.bib75)]] extended the MAE framework to learn temporal feature representations between the frames of a videoclip. Two frames are randomly sampled, the earlier one being split into patches, while the future one is both split and masked. Both images are then separately encoded using Siamese encoders. The resulting embeddings are passed through a decoder with cross-attention (the queries consisting of the future frame’s visible token embeddings and mask tokens) in order to reconstruct the future frame. To learn video-language representations, [[[66](#bib.bib66)]] proposed SMAUG, which stands out as an efficient framework for video-language pre-training, surpassing previous methodologies. In its pre-training process, SMAUG employed a strategy of masking a significant portion of both frame patches and text tokens, enabling simultaneous masked video modeling and masked language modeling. Targeting video generation, [[[59](#bib.bib59)]] introduced Masked Conditional Video Diffusion (MCVD). MCVD leverages different frame masking strategies to train a video diffusion model for unconditional video synthesis and video interpolation. At training time, the method chooses randomly and independently to mask the future or past frames of a given video. This simple strategy allows the diffusion model to perform four different tasks during inference: future and past frame prediction, unconditional generation, and frame interpolation.

Starting from the idea that video and speech data are strongly related, [[[52](#bib.bib52)]] proposed a joint masked pre-training framework for both modalities. Firstly, each data type is encoded into an intermediate latent representation using ResNet for images and a linear layer for audio samples. At each timestep, frames from both modalities are masked by replacing them with an arbitrary sampled frame from the same sequence. These are then concatenated and the information is fused through a transformer-based encoder. Inspired by [[[76](#bib.bib76)]], the objective is to predict which cluster every frame belongs to, the clusters being repeatedly computed by applying a discrete latent variable algorithm to the features extracted from the audio sequence.

[[[62](#bib.bib62)]] introduced a method to fine-tune models, with the goal of achieving a more balanced trade-off between in-distribution (ID) and out-of-distribution (OOD) performance. The authors argue that fine-tuning models on a specific dataset tends to enhance their performance on ID data, but this diminishes their performance on OOD data. Therefore, they proposed an approach that involves masking certain patches within an image and replacing them with content from another image. This modified image is then used to train the model under the supervision of the pre-trained model to recognize the masked image features. Enhancing the robustness of models is also the primary objective of
[[[53](#bib.bib53)]], who observed that existing image denoising methods often overfit on the type of noise seen during training. Their approach focused on improving the generalization capabilities of this type of models by leveraging masked image modeling. To this end, the proposed method masks randomly chosen pixels from the input image and tokens in the self-attention layers of the transformer architecture. The use of token masking in self-attention layers effectively mimics the unreliability observed in tokens during inference, when the data is compromised by various types of noise. This simulation helps to prepare the model for real-world scenarios, where data quality may be inconsistent.

Multiple studies have proposed masking strategies that are more meaningful for medical data than random masking.
[[[77](#bib.bib77)]] introduced masked relation modeling to improve self-supervised pre-training on medical images. This masking strategy uses the self-attention mechanism to identify strong dependent regions in the input image and breaks such relations by masking the most important patches for a given patch. On the same note, the cross-attention is applied between images and genome features, aiming to capture the correspondence between these two modalities. A similar approach that leverages strong dependent regions is presented by [[[78](#bib.bib78)]]. They proposed a masking strategy that is applied to superpixels (contiguous groups of pixels with similar properties), demonstrating its capability on medical image segmentation of skin lesions. Nevertheless, after masking a proportion of superpixels, the same MIM methodology is followed: reconstructing the masked superpixels based on the visible ones. Initially, a base policy [[[79](#bib.bib79)]] is adopted to generate and mask superpixels, after which, the policy is optimized in a self-supervised manner, the model being further pre-trained with the new policy. Leveraging medical reports, [[[61](#bib.bib61)]] presented a masked autoencoder pre-training scheme for vision and language medical data. This model consists of two encoders: one based on ViT for images and one based on BERT for texts. The embeddings of the two modalities are jointly processed with a cross-attention module in order to fuse the information. Finally, each input is reconstructed with a separate decoder: a transformer model for images and a simple multi-layer feed-forward network for texts. While the language decoder receives the output of the last layer of the multimodal module, the vision decoder uses an intermediate latent representation. Another important aspect considered by [[[61](#bib.bib61)]] is that different masking ratios are used for different input types. Working on a similar task, [[[80](#bib.bib80)]] introduced MedIM, a method that guides the masking based on radiology reports. Firstly, both inputs (the medical image and the corresponding report) are encoded using separate encoders. Then, the text embeddings are split into two subsets, according to the original word categories: MeSH and Sentence tokens. Each subset of text embeddings, together with the visual embeddings, are used to generate a mask that hides different information. The loss adds the reconstruction errors between the two resulting decoded masked embeddings (with separate heads) and the original image. In a more specific use case of dental panoramic radiographs, [[[81](#bib.bib81)]] adopted a masked image modeling strategy (either SimMIM or UM-MAE) for pre-training a SwinViT in order to ameliorate its need for a large training dataset. Then, the encoder of the pre-trained SwinViT is taken and used as a backbone in the detection and image segmentation downstream tasks.

[[[82](#bib.bib82)]] extended the MAE approach to temporal skeleton sequences. Rather than opting for the straightforward method of reconstructing the skeleton sequence directly, this research reconstructed the temporal motion embedded within the sequence, using the masked skeleton sequences as input. In addition, the same motion is used to guide the masking of the skeletons.

[[[83](#bib.bib83)]] presented a self-supervised masking-based method for the multi-agent reinforcement learning (RL) setting. For a given timestamp, the observations of all agents are taken. Some of them are masked and replaced by the values of the previous timestamp, eventually being encoded in a latent space. Then, a reconstruction model is used to generate the original agents’ feature representations from the masked sequence. The authors assert that the resulting feature space is stimulated to learn more about the interaction between agents, while being more temporally aware.

*Figure 8: Types of target features employed in MIM pipelines. These features can be either low-level or high-level features. In most cases, low-level features are directly represented by the pipeline’s input. However, in certain cases, they are derived through a transformation of the input. The high-level features are extracted by neural networks that may be either frozen or updated during the pre-training process.*

To detect malicious network traffic, [[[84](#bib.bib84)]] introduced a transformer model, solving the scant data problem by adopting MAE. During pre-training, the raw traffic is processed into a compact 2D matrix consisting of 5 packet levels, dividing it into patches and masking some of them. Then, a transformer encodes the visible tokens and tries to reconstruct the matrix with a decoder. However, during the fine-tuning step, two different encoders are created from the earlier pre-trained one, each having distinctive attention layers (compared with the global attention from the previous stage). On the one hand, an encoder operates only on the tokens within the same packet level. On the other hand, the other encoder performs attention between the patches from all packet levels.

### 3.2 Target Features

In Figure [8](#S3.F8), we illustrate the main categories of target features employed across various MIM models. These features can be categorized in two primary groups, low-level and high-level features. The low-level category encompasses representations such as pixels [[[9](#bib.bib9), [2](#bib.bib2)]], discrete tokens [[[85](#bib.bib85), [86](#bib.bib86)]], 3D [[[87](#bib.bib87), [88](#bib.bib88), [89](#bib.bib89), [90](#bib.bib90)]], HOG [[[91](#bib.bib91)]], and spectral features [[[92](#bib.bib92)]]. Methods based on low-level features often employ high masking ratios to encourage the learning of sufficiently abstract representations that are necessary to enhance performance on downstream tasks. The methods that use high-level features as target features can be split into two groups, namely those that rely on a pre-trained teacher model [[[93](#bib.bib93), [94](#bib.bib94), [95](#bib.bib95), [96](#bib.bib96), [97](#bib.bib97)]], and those in which the teacher and student are trained simultaneously [[[98](#bib.bib98), [57](#bib.bib57), [23](#bib.bib23), [99](#bib.bib99), [100](#bib.bib100), [101](#bib.bib101)]]. In the remainder of this section, we briefly describe the MIM pipelines that primarily contribute through the types of target features they employ.

![Figure](x8.png)

*Figure 9: SdAE pipeline, courtesy of [[[23](#bib.bib23)]] (image licensed under CC BY 4.0). SdAE is classified as a contrastive-based approach (belonging to the generic pipeline in Figure [3](#S2.F3)), as it uses the cosine distance between the student and online teacher features, which are extracted from distinct masked variants of the same input image.*

[[[23](#bib.bib23)]] argued that reconstructing the image pixel space does not force the network to learn an ideal representation of the data. Thus, they introduced a self-distillation framework, shown in Figure [9](#S3.F9), in which the student branch follows the original MAE flow, while the teacher network (which is not updated by gradient descent, but rather from the weights of the student) only takes the masked patches. The objective is to match the high-level features between the two networks. A similar approach is proposed by [[[99](#bib.bib99)]]. The authors presented a self-supervised masking pre-training strategy that involves Siamese networks. Given a source image and applying transformations on it, two different views (anchor and target) are generated. Then, only the anchor’s tokenized patches are masked, and using the Siamese networks (which follow the ViT architecture), both views are encoded. The latent representations are compared with a set of learnable prototypes in order to generate a distribution, the final goal being that of obtaining matching distributions (predictions of anchor and target). While the anchor’s model is updated using gradient descent, the target’s network parameters are computed as an exponential moving average from the anchor network’s weights. The authors attest that the method greatly improves the performance in few-shot scenarios.

A handful of studies used 3D data as target features within their proposed MIM pipeline.
[[[87](#bib.bib87)]] proposed the masked image modeling pre-training for 3D meshes. Inspired by ViT, they begin by grouping faces together (each face being represented by a 10D vector) into a patch. Then, the authors adopt the transformer architecture, using the 3D location of the center of each patch to compute the positional embedding. The algorithm follows MAE: patches are randomly masked with a high masking ratio, the visible patches are passed through the encoder, then the resulting latent embeddings are concatenated with masked tokens (associated with the masked patches), and subsequently decoded. The objective is not only to reconstruct the masked patches (by predicting the coordinates of the vertices), but also the faces of the patch (the 10D representations). Aiming to learn more robust representations with 2D vision models, Mask3D [[[89](#bib.bib89)]] enhanced the 2D feature representations of the ViT backbone [[[22](#bib.bib22)]] by integrating 3D priors into the training pipeline. Mask3D utilized RGB-D data within a self-supervised framework, where both color images and corresponding dense depth maps are masked and processed through dual encoders. These encoders project the data into a higher-dimensional space, enabling a decoder to accurately reconstruct the dense depth map. This method enriches the capability of ViT to handle spatial depth alongside traditional 2D data. Focusing on point clouds, GeoMAE [[[90](#bib.bib90)]] is an adaptation of the MAE framework to point clouds. This method changed the usual reconstruction objective and replaced it with centroid prediction, normal estimation and curvature prediction. This change showed significant improvements in downstream tasks such as object detection, segmentation and multi-object tracking. In a semi-supervised setting, given a labeled (source) and an unlabeled (target) 3D point cloud dataset, the aim of [[[102](#bib.bib102)]] is to transfer the knowledge from the latter to the former by embedding information about common features in an encoder. This model is trained simultaneously on both datasets, but with different objectives. On the one hand, training on the source dataset is performed in a supervised setting on a specific task. On the other hand, points from the target dataset are randomly masked in arbitrary areas and the objective is to estimate their cardinality (number of points in the neighborhood), position and normal vectors. Therefore, the model benefits from unlabeled data by encapsulating information about the structure of objects.

To boost the performance of any knowledge distillation framework based on feature learning, [[[95](#bib.bib95)]] proposed an auxiliary task. A proportion of the latent feature maps of the student is masked, and a projection layer tries to recover the masked feature maps and match them with those of the teacher. This results in higher performance for a wide range of computer vision tasks. In a similar manner, but focusing on efficiency, [[[96](#bib.bib96)]] introduced a method for knowledge distillation using the MAE framework. Their approach reconstructs masked patches, while training the student model to replicate the early (low-level) feature maps of the teacher. The efficiency stems from utilizing the MAE framework and from the partial evaluation of the teacher network, as it only requires early feature maps for training the student. This significantly reduces computational cost and training time.

[[[91](#bib.bib91)]] used the HOG features of the masked regions as target values. The masked patches are replaced with learnable tokens and the architecture is based on a single encoder followed by a linear head for predicting the HOG features. Low-level target features were also used by [[[103](#bib.bib103)]]. They introduced a novel approach for boosting the performance of ViT. A number of patches from the original image are shuffled and have their positional encodings masked. Besides the original loss of the downstream task, another objective that tries to predict the masked positional encodings is employed.

A few studies have attempted to employ target features better suited for video data.
[[[88](#bib.bib88)]] proposed a video representation learning method based on masking, which reconstructs trajectory features rather than static information (like frames). These target features are carefully designed to capture long-term motion changes (object shape and position). [[[97](#bib.bib97)]] conducted an empirical analysis on optimal target features for video-language pre-training. They found that spatially-focused image features, extracted using a Swin-B transformer [[[69](#bib.bib69)]], yield the best results. Additionally, their research incorporates the usual tasks employed in video-language pre-training, such as masked video modeling, masked language modeling and video-text matching. For their video compression method, [[[104](#bib.bib104)]] employed a transformer-based entropy that is pre-trained using masked image modeling. Some tokens from the current frame are randomly masked, and the model tries to estimate their probability mass functions. More prior information about the last decoded frames is supplied as keys and values. At inference, the prediction is performed as an iterative process.

[[[100](#bib.bib100)]] combined image-text contrastive learning (CLIP) [[[105](#bib.bib105)]] and MIM. Their main contribution over the naive approach for combining the two methods consists in using the language space as target space for the reconstruction objective. This is motivated by the intuition that the language features serve as rich semantic representations of the visual signal. The CLIP representation space is also utilized by
EVA [[[93](#bib.bib93)]], a ViT model trained to reconstruct CLIP features conditioned on masked image tokens. [[[93](#bib.bib93)]] showed that this self-supervised task is suitable for large scale representation learning. The EVA model scales to one billion parameters using tens of millions of samples, showcasing its potential for handling extensive datasets and complex learning tasks.

[[[101](#bib.bib101)]] introduced a masked-based auxiliary objective to train a model for semantic segmentation of Laparoscopic images. Due to the scarcity of labeled data, the authors proposed to use a labeled proxy source dataset (with simulated images) and an unlabeled target dataset (with real images) to transfer the knowledge. The former dataset was used to compute the supervised loss for segmentation with a student model. Each image from the second dataset has its higher frequencies masked (by first applying the Fourier transform and then the inverse), and its segmentation map is predicted using the student model. The resulting output is compared with the prediction of the intact image given by a teacher model (i.e. the exponential moving average of the student), the objective being that of minimizing the distance between the two.

### 3.3 Objective Function

![Figure](x9.png)

*Figure 10: Types of objective functions employed in MIM pipelines. Reconstruction losses, particularly those based on $L^{1}$ and $L^{2}$ distances, constitute the most commonly employed objective functions in MIM pipelines. These are followed by the contrastive loss objectives, where the most used instance is the InfoNCE loss. Classification losses are less frequent, but they are present in pipelines that employ pseudo-labels or conduct masking within a discrete space.*

We next present the works that have as a main contribution the loss function employed in their MIM pipeline. In Figure [10](#S3.F10), we provide an overview of the main objective functions used across MIM approaches, distinguishing between reconstruction [[[9](#bib.bib9), [2](#bib.bib2), [106](#bib.bib106), [4](#bib.bib4)]], contrastive [[[107](#bib.bib107), [25](#bib.bib25), [108](#bib.bib108), [4](#bib.bib4)]], and classification losses [[[109](#bib.bib109), [110](#bib.bib110)]]. Reconstruction losses remain the most frequent option in MIM pipelines. In this section, we highlight methods that employed them in less typical ways. For instance [[[106](#bib.bib106)]] used a reconstruction loss at test time, while [[[4](#bib.bib4)]] combined reconstruction and contrastive losses. Although classification losses are less common, they appear in scenarios where pseudo-labels can be generated [[[109](#bib.bib109), [110](#bib.bib110)]].

Inspired by MAE, [[[109](#bib.bib109)]] proposed an analogous pre-training framework for 3D point clouds, but substituted the reconstruction task with discrimination. A small proportion of points are left unmasked, and subsequently encoded. Then, a subset of the masked ones is sampled (real), along with some random 3D points from the space (fake), and the decoder’s objective is to discriminate between the two. The encoded unmasked points are used in the cross-attention blocks of the decoder. Experiments are conducted for various downstream tasks, in which the method shows considerable performance improvements. A classification loss is also used by [[[110](#bib.bib110)]]. To boost the performance of models that have zero-shot classifying capabilities (such as CLIP), [[[110](#bib.bib110)]] proposed a framework composed of three tasks. Besides the reconstruction loss of the masked patches, the second objective is to minimize the distance between the resulting embeddings of the masked tokens and the embedding of the prepended [CLS]* token in a shared projected space. The third objective employs a classification loss in which the labels are provided by an EMA teacher applied on the unmasked image.

[[[107](#bib.bib107)]] studied the effectiveness of combining MAE and CLIP in a single framework. The conclusion is that the combination brings some benefit when training on a smaller dataset (tens of millions of samples), but this benefit is marginal when the experiments are carried out on a much larger dataset (1 billion samples).

[[[106](#bib.bib106)]] introduced an application of the MAE self-supervised learning framework during test phases, followed by executing predictions with the refined weights. This method was evaluated in the context of point cloud classification, demonstrating enhanced performance across a variety of standard perturbations affecting 3D point clouds. The findings suggest that updating model weights with MAE at test time significantly improves the robustness and accuracy of point cloud classification tasks.

*Figure 11: CMAE pipeline, courtesy of [[[4](#bib.bib4)]] (image licensed under CC BY 4.0). CMAE is a hybrid of the generic pipelines illustrated in Figures [2](#S2.F2) and [3](#S2.F3), combining a pixel-level reconstruction loss with a contrastive loss. The latter ensures alignment between features extracted from a masked image and those derived from a pixel-shifted version of the same image, extracted with a momentum encoder.*

[[[25](#bib.bib25)]] formally introduced the self-distillation masked autoencoder framework. An input image is divided into patches, some of which are randomly masked. Two encoder-decoder networks (teacher and student) are used to reconstruct the original image. A new objective is employed to minimize the distance between the predictions of the teacher and the student. While the student is trained using gradient descent, the teacher’s weights are computed as an exponential moving average of the student’s. [[[108](#bib.bib108)]] proposed a similar method based on Siamese networks. One network, which is fed with a highly masked input image, tries to predict the tokens from the other network, which is given the complete image. The work of [[[4](#bib.bib4)]] is built on the same idea, but combines the reconstruction objective with a contrastive loss. As presented in Figure [11](#S3.F11), the authors achieve this by using two branches, one for each strategy. The first one uses an encoder and a pixel decoder, which are updated at every step. The second one employs a distinct encoder, which is updated as an exponential moving average from the other encoder. It also uses a projection layer and a feature decoder that have the same output vector space. Two views that have a different shift are created from the input image, one being passed through the first branch (as well as being masked), while the other is fed into the second branch (unaltered). The first branch follows the original MAE framework, with the reconstruction objective. The second branch applies a contrastive loss between the projected embedded features (with the latter encoder) of the second view and the decoded embedded features (i.e. from the former encoder) of the first view. A similar hybrid approach is used by [[[111](#bib.bib111)]]. The authors apply a reconstruction loss to masked images and a contrastive loss to masked video frames, enabling their method to be effective in both image and video tasks.

[[[112](#bib.bib112)]] integrated a masked autoencoder model into a reinforcement learning setting in order to obtain a reward model for exploration. The autoencoder is trained by masking some states from a trajectory and then estimating them. Given a sampled trajectory, for each timestamp, a fixed number of previous states are kept and encoded. Some of the resulting embeddings are then masked, while the rest are passed through the decoder to predict the missing states. In the end, the exploration reward is given by the prediction error.

### 3.4 Downstream Task

![Figure](x11.png)

*Figure 12: Downstream tasks on which MIM pipelines are typically applied. The downstream tasks can be grouped by the input modality. Most MIM research focuses on natural images, evaluating the learned representations through downstream tasks such as classification, segmentation, generation, and object detection. Similar downstream tasks are used for medical images, multi-view images and point clouds. In the case of videos, the primary task is action classification, but there are also some studies that investigate the effectiveness of MIM in video generation and anomaly detection.*

In Figure [12](#S3.F12), we outline the downstream tasks where MIM pipelines have been successfully applied, grouping them by modality. In the case of images, pre-training strategies are typically evaluated on classification [[[9](#bib.bib9), [2](#bib.bib2), [24](#bib.bib24)]], object detection [[[113](#bib.bib113)]], segmentation [[[9](#bib.bib9)]], and generation [[[32](#bib.bib32), [114](#bib.bib114)]]. For medical images, segmentation [[[115](#bib.bib115)]] and classification [[[116](#bib.bib116)]] remain prevalent, with anomaly detection [[[117](#bib.bib117)]] also being investigated. When it comes to videos, MIM pipelines are employed for generation [[[118](#bib.bib118)]], action classification [[[119](#bib.bib119)]], and abnormal event detection [[[120](#bib.bib120)]]. In multi-view images, object detection [[[121](#bib.bib121)]] and segmentation [[[121](#bib.bib121)]] are common applications, while for point clouds, researchers have explored object detection [[[90](#bib.bib90)]], classification [[[122](#bib.bib122)]], and action classification [[[123](#bib.bib123)]].

MaskGIT [[[32](#bib.bib32)]] trains a generative transformer to reconstruct randomly masked image patches. At inference time, it starts by predicting all the patches and keeps the most confident ones for the next iteration, when the rest of the patches are again masked and regenerated. The process continues for a few iterations. Overall, the pipeline has actually two stages, the first one encodes the patches into visual tokens with a VQ-Encoder, and the second stage (decoder) receives masked tokens for reconstruction. Inspired by this, [[[7](#bib.bib7)]] adopted the token masking strategy in their work, but replaced the values with zero (rather than a special token) and the token embeddings are obtained with a single dense layer. Another work that studies the application of MIM in image synthesis is MDT [[[124](#bib.bib124)]]. In their work, [[[124](#bib.bib124)]] integrated latent masking in the training pipeline of a latent diffusion model.

[[[119](#bib.bib119)]] proposed to use MAE on videos. The video is split into equally-sized patches that do not overlap along any dimension (i.e. including time), as well as consisting only of two timesteps. As hypothesized and demonstrated by the authors, a high masking ratio (about 90% of the whole input video, unaware of any axis) is used due to redundant data when decoding. Positional (i.e. height and width) and temporal (i.e. time) embeddings are added to the input tokens. The architectures of both the encoder and the decoder are based on ViT. The method follows the same logic as MAE: encoding only the visible tokens, then decoding the complete set of tokens. Targeting video generation, MAGVIT [[[118](#bib.bib118)]] trained a transformer-based model for conditional video generation, handling tasks like frame prediction, interpolation, and central outpainting. The model processes videos by selecting a task, and creating conditional tokens from the preprocessed video. The conditional tokens, along with masked and original tokens, form the input used to train the model, which is optimized with three objectives: refining conditional tokens, predicting masked tokens, and reconstructing original tokens. Inference is conducted autoregressively, reducing the masking ratio progressively.

![Figure](x12.png)

*Figure 13: AMAE [[[117](#bib.bib117)]] belongs to the family of reconstruction-based methods (it falls under the generic pipeline presented in Figure [2](#S2.F2)), as it employs an ${L}^{2}$ loss for masked patch reconstruction, similar to MAE.*

[[[24](#bib.bib24)]] adapted MAE for test-time training. Employing a pre-trained MAE whose weights are frozen, a classification head, represented by a ViT, is attached to the encoder and fined-tuned on the supervised dataset. At inference time, each sample is initially used to train the network on the reconstruction objective in multiple steps, thus modifying the resulting encoded latent representation, while the head does not change, and then classifying the input. The encoder and the decoder are always reset with their original weights after each prediction. An increase in performance comes at the cost of inference speed.

[[[125](#bib.bib125)]] applied a pre-training masking strategy to the less studied task of panoramic depth completion. A pair consisting of an RGB image and the associated sparse depth map are jointly masked. Both are then passed through a guided convolutional network [[[126](#bib.bib126)]] to generate the sparse depth map, and the reconstruction loss is only computed for the masked depths. Carrying out experiments on the dense depth map prediction downstream task, the proposed method, called $M^{3}PT$, shows better qualitative and quantitative results than previous state of the art.

Several works have focused on various tasks in the medical domain.
Masked Autoencoder Guided Segmentation at Pixel Resolution (MAESTER) [[[127](#bib.bib127)]] is a masked image modeling approach for the segmentation of cellular images. MAESTER incorporates the masked autoencoder in the training pipeline of a visual transformer to learn token representations relevant to the sub-cellular structure segmentation task (i.e. texture) and performs the segmentation at inference time by clustering the learned representations.
[[[114](#bib.bib114)]] used masked image modeling to learn a latent space from fMRI inputs. After this stage, the authors used the learned latent representations to condition a diffusion model that is able to generate visualizations of the initial visual stimuli. [[[117](#bib.bib117)]] leveraged an MAE for detecting anomalies in chest X-rays. The first stage of their pipeline
consists of initially pre-training a masked autoencoder. A classification head is then attached to the encoder (whose weights are frozen) to classify normal and abnormal samples, the latter being artificially created. During the second stage, unlabeled data is classified using the model from the previous stage, and the examples that have high confidence are kept. The last step is to employ two different copies of the pre-trained autoencoder, one for each class, and train them separately on the masked reconstruction task. At inference, multiple reconstructions are generated from both autoencoders, an anomalous prediction being detected by a large difference between the mean reconstructed images of the two modules. The steps in the second stage are presented in Figure [13](#S3.F13). In an effort to boost the performance of computer vision models in the medical domain, [[[116](#bib.bib116)]] used visual transformers for multi-label classification of chest X-rays. The large quantity of data required for training a ViT was overcome by pre-training the model on the masked modeling task. Furthermore, the authors proved the superior performance of their method compared with CNN-based architectures.
Similar to [[[116](#bib.bib116)]], [[[115](#bib.bib115)]] demonstrated the benefits of masked image modeling on 3D medical images. They adopted two strategies (original MAE and SimMIM) with various masking hyperparameters, showcasing improved results on two different segmentation tasks. [[[128](#bib.bib128)]] adopted the MAE pre-training strategy for microscopy images. Their objective was to learn a rich feature representation of the cellular morphology of the data. The extensive experiments demonstrated excellent results for predicting biological relationships. Leveraging both images and text,
[[[129](#bib.bib129)]] introduced masked modeling as a pre-training strategy for medical radiography tasks by using the associated radiology reports. While the images are downsampled by half and their unmasked patches are encoded, the text reports are tokenized, randomly masked and then embedded through a look-up table. A global average pooling over the resulting visual embeddings is added to the unmasked text embeddings, and all of them are decoded to obtain the intact report. The image tokens are fed as well through another decoder in order to reconstruct the radiography at its original resolution.

[[[113](#bib.bib113)]] explored the effectiveness of a pre-trained Vision Transformer (ViT) on masked image modeling tasks within the context of object detection.
Geometry Enhanced Masked Image Modeling (GeoMIM) [[[121](#bib.bib121)]] tackled the problem of 3D detection using multi-view cameras. This method involves pre-training a student network by utilizing masked inputs from multiple camera views. The objective is for the student network to reconstruct the bird’s-eye-view features, leveraging the guidance of a pre-trained LiDAR-based model. This strategy bridges the gap between multi-view camera inputs and LiDAR precision, enhancing the student network’s ability to accurately interpret and reconstruct 3D environments.

In order to combine masked image modeling with contrastive learning during pre-training, [[[130](#bib.bib130)]] proposed to apply the former on the initial layers, while the latter is used on the last layers. This is achieved in an iterative manner, by firstly pre-training on the reconstruction task, freezing the respective layers, and then training on the contrastive task.

[[[131](#bib.bib131)]] integrated a secondary masked image modeling task in their pose-guided human image generation training pipeline. The authors opted for a transformer-based denoising architecture, rather than a U-Net. As a result, the masked tokens of the target image are reconstructed not only through the self-attention layers, but also using cross-attention modules with the aggregated tokens of the target pose and the source image.

### 3.5 Theoretical Analysis

Besides the applied contributions of MIM, some pieces of work take another approach: they theorize about various aspects of MIM and dive deeper into its fundamentals. The papers from this category address aspects such as overall understanding [[[132](#bib.bib132), [133](#bib.bib133), [134](#bib.bib134), [135](#bib.bib135), [136](#bib.bib136), [137](#bib.bib137)]], the connection between different MIM strategies [[[41](#bib.bib41)]], or present certain drawbacks and how to overcome them [[[138](#bib.bib138), [139](#bib.bib139)]].

[[[135](#bib.bib135)]] evaluated the performance of self-supervised learning (SSL) by determining whether the trained model represents enough information to obtain the data distribution, given additional information about the distribution family. The SSL task chosen for this assessment was the masked prediction task.

[[[136](#bib.bib136)]] explored the effectiveness of MIM on out-of-distribution (OOD) detection. Their results showed that MIM improves the performance in multiple settings, such as one-class, multi-class or near-distribution OOD.

![Figure](x13.png)

*Figure 14: SSMCTB architecture, courtesy of [[[31](#bib.bib31)]] (image available under the ArXiv perpetual license agreement). SSMCTB falls into the category of reconstruction-based methods depicted in Figure [2](#S2.F2), as it employs the Huber loss between the reconstructed output and the input.*

[[[132](#bib.bib132)]] formulated the underlying data generation process as a hierarchical latent variable process.
The authors discovered relationships between the latent variables of the data generation process and the masking parameters (masking ratio and patch size) of the MAE framework. These relationships allow MAEs to recover variables of different semantic levels.
The authors validated their theoretical discoveries with several experiments, where the main result showed that very large masking ratios have a similar effect as low ratios, namely that of learning low-level information.

[[[139](#bib.bib139)]] investigated the data scaling capabilities of masked image modeling. Their experiments use datasets of various sizes, ranging from 100.000 samples to 14 million samples. The authors verified their observations against two masked image modeling approaches, SimMIM [[[2](#bib.bib2)]] and MAE [[[9](#bib.bib9)]]. The conclusions of their study state that MIM still necessitates data scaling to effectively facilitate model scaling. The study also noted that, in non-overfitting scenarios, simply increasing the number of unique samples does not necessarily enhance performance.

[[[134](#bib.bib134)]] explored the differences in representations learned by deep models through MIM versus traditional supervised training. They observed that MIM encourages models to focus on local patterns across all layers, whereas supervised training emphasizes these patterns only in the initial layers. Additionally, MIM results in a greater diversity among the attention heads compared with supervised methods, suggesting a more nuanced feature recognition within the model. In addition, [[[138](#bib.bib138)]] studied the adversarial robustness of the transformers pre-trained with MIM. Their first observation is that MAE, in particular, has a lower robustness compared with other methods. Moreover, they found that the robustness is related to the reconstruction target. For example, a model trained to reproduce the pixels of an image is prone to adversarial attacks because its focus is on medium and high-frequency features. To ameliorate this issue, the authors proposed a test-time solution based on visual prompts optimized on the frequency domain. These prompts are then included in the input images through prototype-based prompt selection.

[[[133](#bib.bib133)]] theorized about the mechanisms behind reconstructing the masked input, the benefits of this pre-training strategy and why it learns valuable feature representations. The main finding is that discriminative features are learned during pre-training, and thus, when applied to a downstream task, these are further enhanced, which has a great advantage over randomly initialized weights.

![Figure](x14.png)

*Figure 15: MCMAE [[[140](#bib.bib140)]] falls into the category of reconstruction-based methods illustrated in Figure [2](#S2.F2), as it leverages an ${L}^{2}$ loss between the masked patches and the corresponding patches from the input image.*

### 3.6 Model Architecture

While the architecture employed throughout MIM research was consistent (a transformer-based encoder and a shallow decoder), there have been some important contributions that further enhance the performance of the pre-training task through architectural modifications [[[141](#bib.bib141)]]. Some of these advances increase the benefit of MIM pre-training on compact transformers [[[142](#bib.bib142), [141](#bib.bib141)]], while others integrate multiple modalities within the architecture [[[143](#bib.bib143)]]. Furthermore, a few attempts have tried to distinguish themselves from the usual ViT-based models, either by using CNNs [[[144](#bib.bib144), [145](#bib.bib145), [140](#bib.bib140)]] or by integrating MIM into the convolution operation [[[31](#bib.bib31), [30](#bib.bib30)]].

[[[30](#bib.bib30)]] presented the self-supervised predictive convolutional
attentive block (SSPCAB), a novel block comprising a masked convolutional layer and a Squeeze-and-Excitation (SE) module. The filters of the masked convolutional layer contain learnable parameters in the corner regions of the receptive field and the masked region is located in the center. This novel block is trained using a self-supervised reconstruction loss, being integrated in anomaly detection networks. Later, [[[31](#bib.bib31)]] introduced the self-supervised masked convolutional transformer block (SSMCTB) for anomaly detection. SSMCTB is an extension of SSPCAB, being trained via a self-supervised reconstruction loss and comprising a masked convolutional layer. In contrast to [[[30](#bib.bib30)]], [[[31](#bib.bib31)]] employed a channel-wise transformer block instead of the SE module, as shown in Figure [14](#S3.F14).

Motivated by leveraging the masked pre-training strategy of autoencoders with convolutional layers, [[[140](#bib.bib140)]] presented an architecture that combines them, as illustrated in Figure [15](#S3.F15). The encoder consists of three stages, where the first two are convolutional and the last one is transformer-based. First, a mask is sampled to determine which tokens are visible. The mask is then upsampled at the resolutions of the first two stages, in order to be used by masked convolutional blocks. Information from the first two stages is added to the resulting tokens of the third stage, and then, they are linearly projected. Finally, all tokens (predicted and masked) are decoded into the original pixel space. The authors state that the advantage of this method is represented by the multi-scale features learned by the encoder. Having the same goal of adopting convolutional layers in MIM pipelines, [[[144](#bib.bib144)]] presented a MIM method that leverages CNNs. As in the MAE framework, the input is split into non-overlapping patches and some of them are masked. The encoder is composed of sparse convolutional layers, thus preserving the mask pattern intact along the feature maps. The decoder, used for reconstructing the original image, is composed of upsampling blocks that receive the previous layer, as well as the encoder’s features on the same level. The masking regions are replaced with a mask embedding.

SparseMAE [[[142](#bib.bib142)]] offered a novel solution to the problem that small transformers face in not benefiting significantly from MAE pre-training. It did this by concurrently training a full-scale transformer alongside a smaller sparse network, which resides inside the full transformer. This smaller network is tasked with reconstructing the masked patches of input images. Uniquely, SparseMAE independently manages two sets of weights, one for the sparse network and another for the encompassing larger transformer. Despite this separation, both networks aim to achieve the same objective: the accurate reconstruction of the masked image patches. After pre-training, only the sparse network is used for fine-tuning. A similar approach based on smaller networks is introduced by [[[141](#bib.bib141)]]. Inspired by TokenMoE [[[146](#bib.bib146)]], [[[141](#bib.bib141)]] proposed a pre-training method that is more robust for all downstream tasks. The first step is to obtain the features representations of a pre-trained MAE and cluster them (obtaining the centroids). The architecture of the model is adapted to contain multiple experts (i.e. groups of heads in the transformer layers), each expert being associated with a cluster. The tokens are routed through the expert which the image was assigned to. When applied to a downstream task, the method selects only the most used experts by the dataset for fine-tuning.

[[[147](#bib.bib147)]] proposed to integrate RevCol [[[148](#bib.bib148)]] in the MAE framework. Besides the bottom-up columns, RevCol is extended to include top-down columns as well, thus resembling an encoder-decoder. The network contains reversible connections, helping the model to learn disentangled representations. In this way, the decoder is no longer dropped during fine-tuning, as it contains salient information.

[[[143](#bib.bib143)]] proposed to combine audio and visual modalities at the earliest stage in a multimodal architecture. Aside from separate modality transformer blocks, another block is used to process both the audio and visual tokens, further fusing the resulting information into the other blocks. The experiments demonstrated strong performance on a wide range of downstream tasks, such as visually-guided sound source separation, audio-visual segmentation or classification.

### 3.7 Features and Objective

We next discuss studies that make contributions to both the target features and the objective function. While most approaches use a deep encoder, either frozen [[[149](#bib.bib149), [150](#bib.bib150)]] or updated online [[[21](#bib.bib21), [151](#bib.bib151), [152](#bib.bib152), [153](#bib.bib153), [154](#bib.bib154)]] during training, to extract the target features, there are also methods that extract their features directly from the raw input signal [[[155](#bib.bib155)]], eliminating the need for a neural network. In terms of objective functions, most of the works focus on integrating a contrastive loss component [[[21](#bib.bib21), [152](#bib.bib152), [153](#bib.bib153)]], but there are also a few methods that leverage a multi-scale reconstruction loss [[[155](#bib.bib155)]]. Another important direction is closely following the strategy in BERT [[[18](#bib.bib18)]], using discrete tokens and predicting the id from a learned vocabulary [[[85](#bib.bib85), [86](#bib.bib86)]]. These tokens are obtained using a variational autoencoder (VAE), the pixels being mapped to a discrete latent space [[[156](#bib.bib156)]]. With a similar goal in mind, other works map each token into the nearest entry from a codebook of learned embeddings [[[157](#bib.bib157), [34](#bib.bib34)]].

MaskCo [[[21](#bib.bib21)]] is a region-level contrastive learning framework. It begins by augmenting images to create two distinct views of the same image, with one view partially masked. The model is then trained using contrastive learning to align the features of the masked regions with those of the corresponding regions in the unmasked view.
[[[153](#bib.bib153)]] reinterpreted MIM via the lens of contrastive learning. They found a formulation showing that the classic MIM approach is equivalent to a setting with Siamese networks, where one network reconstructs the masked tokens and its counterpart focuses on the unmasked tokens. The goal is to closely align the outputs of these models.
MaskCLIP [[[152](#bib.bib152)]] combines masked image modeling and the CLIP contrastive loss between images and text in a single framework. Compared with vanilla CLIP, MaskCLIP has an additional loss for reconstructing a masked image and the two losses share the same visual encoder.

[[[151](#bib.bib151)]] proposed to integrate an additional masked contrastive objective into an RL pipeline dedicated to video games. Sequences from the video clips are sampled and then each frame is encoded using a CNN-based network. Then, besides the main RL policy network, an auxiliary branch is added that employs the student and teacher (exponential moving average of the student) framework. While the former receives masked input features, the latter is fed with the intact latent representation. The objective is to maximize the similarity between the two resulting embeddings.

Aside from studying the contrastive objective, additional components added to the loss function were sometimes explored. Interestingly, such studies used multimodal data.
To learn better visual representations during MIM pre-training within the medical domain, [[[154](#bib.bib154)]] leveraged an additional modality (text). Two different encoders are used (one for each modality), however, the text encoder is fed with the average global vision embedding as well. The method consists of two reconstruction objectives, one for the image and one for the text, each with its own decoder. Furthermore, the authors employed a third contrastive objective between the fully visible encoded text and a decoded representation (using an additional decoder) of all the image embeddings (masked and unmasked tokens).
[[[158](#bib.bib158)]] improved the MAE pre-training strategy on multimodal data by optimizing the latent encoding space. Some neighboring tokens of the unmasked tokens are sampled as well, in order to compute the reconstruction loss and use the resulting gradient to produce a more explicit latent representation. This is further used to compute the final reconstruction loss. Moreover, an additional contrastive loss is employed to maximize the similarity of the two resulting latent representations of a task, while minimizing it for different tasks.

[[[155](#bib.bib155)]] showed that, in the previous masked image modeling approaches, it is difficult for the lower layers to learn inter-patch relations. To alleviate this issue, the authors introduced LocalMIM. They addressed the problem by using a loss function based on the weighted sum of the local patches losses. Their pipeline also includes a multi-scale reconstruction task, in which the model is supervised with different feature descriptors.

[[[149](#bib.bib149)]] presented a two stage distillation framework. In the first stage, the student network distills the task-agnostic knowledge of the teacher, and the authors chose MIM as a proxy task for this goal. Thus, the student learns to align its visible and masked patches representations with those of the teacher. In the second stage, the classic task-oriented knowledge distillation takes place.

[[[150](#bib.bib150)]] presented a modified MAE framework for Whole Slide Images (WSI). After removing the background, the images are split into patches (some of which are masked), and for each patch, a feature vector is computed using DINO [[[48](#bib.bib48)]]. Following [[[159](#bib.bib159)]], trainable anchor vectors are employed, which are then used to calculate the distance and polar angle between the features and the anchor vectors. All these representations computed only for the visible patches are encoded, appended with mask tokens, and finally decoded to reconstruct all WSI representations. The cross-attention units between the patches and anchors from both the encoder and the decoder are bidirectional.

[[[85](#bib.bib85)]] adapted the pre-training strategy from BERT to images. First, the image patches are mapped to discrete visual tokens using a discrete Variational Autoencoder (dVAE) [[[156](#bib.bib156)]]. Then, some tokenized patches are randomly masked and replaced with a special mask token. These are then passed through the encoder together with the visible tokens. The final objective is to predict the id of the masked tokens from the learned vocabulary of visual tokens. Instead of a decoder, this method uses only a classification head, which is dropped during fine-tuning. [[[160](#bib.bib160)]] extended BEiT by employing multiple tokens for each patch. Inspired by the aforementioned work, [[[86](#bib.bib86)]] proposed to learn both the tokenizer and the encoder. Their method adopts the teacher-student framework, such that the student tries to reconstruct the corresponding masked tokenized patches from the teacher and to match the [CLS]* token.

### 3.8 Architecture and Objective

A number of papers contributed to both the model architecture and the objective function. An important subset of these studies made their architectural changes as a result of integrating the new objective [[[161](#bib.bib161), [94](#bib.bib94), [162](#bib.bib162), [163](#bib.bib163)]]. For example, the introduction of adversarial loss by [[[161](#bib.bib161)]] requires an additional discriminator, and the multi-scale reconstruction loss used by [[[94](#bib.bib94)]] implies additional adapters to ensure compatibility between features. Another direction of research that requires both architectural and objective function modifications is the integration of multiple modalities [[[164](#bib.bib164), [6](#bib.bib6)]]. In such scenarios, each modality typically requires its own dedicated modules, leading to architectural adjustments, while an additional loss term is introduced to ensure alignment between the features of the two modalities.

Rather than following the standard approach in MIM with a reconstruction objective, [[[94](#bib.bib94)]] proposed a unique loss function that aligns features extracted from visible tokens with those extracted by a teacher model across various architectural levels. To facilitate this alignment, this study introduced a novel module named Dynamic Alignment. This module is specifically designed to ensure compatibility between the two sets of features, enabling more effective feature alignment.

For an improved objective function, some methods were augmented with additional loss components.
[[[163](#bib.bib163)]] demonstrated that an additional auxiliary supervised classification task helps the MAE pre-training framework. Besides the main reconstruction loss, the authors integrated another branch that takes only a subset of the visible encoded tokens as input. Further, the branch applies a global average pooling operation on the visible tokens, and finally predicts the class through a multi-layer perceptron. During the fine-tuning stage, all tokens are used.
[[[161](#bib.bib161)]] enhanced the standard MAE pipeline by integrating a discriminator for adversarial training. Notably, this discriminator, which shares parameters with the MAE’s encoder, is trained to distinguish between synthesized and real patches. This enhancement is an addition to the existing pipeline, with the typical reconstruction loss still being present in the training process.

Inspired by masked autoencoders that process both visual and textual modalities, [[[165](#bib.bib165)]] adapted the masking pre-training framework for optical and radar inputs. Both modalities, after being aligned, tokenized and randomly masked, are encoded using an individual encoder, and then, they are jointly processed by a multimodal encoder. The resulting embedding is decoded to reconstruct both input images. A contrastive loss between the mean embeddings of the individual encoders is adopted to match sensor measurements from the same timestamp, while maximizing the difference between those at different timestamps.

[[[162](#bib.bib162)]] demonstrated how MAE can boost the performance of 3D medical image segmentation. The input volume (a 3D scan) is split into equal subvolumes and these are randomly masked. Then, different views (frontal, horizontal, or longitudinal) are obtained, and a further arbitrary rotation is applied. During pre-training, besides the main reconstruction objective for each view, three more losses are utilized: the rotation angle estimation, a contrastive loss, as well as an additional MSE loss between two different reconstructed views (after being normalized). The architecture of the encoder is based on Swin transformer. A cross-attention module, which attends to the features between two views, is integrated before the first level of the decoder.

In the work of [[[164](#bib.bib164)]], the audio spectrogram and the images are tokenized. The unmasked tokens are encoded with separate encoders, while also adding a corresponding modality embedding. Then, three separate forward passes are performed through a common encoder: one for each modality embedding, as well as the concatenation of the two. The concatenated tokens, together with the masked tokens, are decoded and the reconstruction loss is applied. Furthermore, a contrastive loss is applied between the averaged-pooled encoding of each modality. Rather than relying on global information for learning audio-visual features, [[[6](#bib.bib6)]] took a different approach by focusing on a finer-grained level, as well as improving the linkage between the two modalities. Their pipeline involves two more objectives besides the contrastive loss between the embeddings of the two modalities. One additional objective aims to reconstruct the original signal from the unmasked tokens of both input sources, while the second additional objective aims to reconstruct the embeddings using the counterpart tokens and some learnable queries.

The contribution of [[[166](#bib.bib166)]] is twofold. Firstly, they integrated MAE pre-training for videos by applying a consistency loss between two successive frames that are masked in the same manner. The masked frames are then encoded and decoded with two different networks (one is an exponential moving average of the other). Secondly, they proposed an analogous architecture using sparse convolutional layers instead of ViT, which results in lower computational costs.

### 3.9 Masking Strategy and Features

Several studies have impacted both the masking strategy and the target features. Generally, these studies adjust their target features to reflect changes in masking strategy or in the type of input features that they use [[[167](#bib.bib167), [168](#bib.bib168), [169](#bib.bib169), [170](#bib.bib170), [171](#bib.bib171), [172](#bib.bib172)]]. For example, a common scenario is when the model processes pairs of images [[[167](#bib.bib167), [173](#bib.bib173)]]. In such a scenario, the model employs shared decoders and the target image for reconstruction for at least one of the decoders is altered to reflect the relationship between the images comprising the input pair. Additionally, other studies [[[168](#bib.bib168)]] use the feedback from an auxiliary model to identify the relevant regions for masking and to define the space in which the reconstruction loss is applied.

A couple of papers in this section were applied to object tracking.
[[[167](#bib.bib167)]] presented a method for learning representations useful for object tracking. The method is based on MAE, but the authors used two inputs, one is the search region and the other one is the template. The MAE is trained to reconstruct the search region as it is, and to recreate the template in the position found in the search region.
Inspired by the MAE framework, [[[173](#bib.bib173)]] proposed to boost the performance of a ViT model used for object tracking (given an object in a template image, find the same object in the search image) by applying MIM as an additional concurrent task. Both input images are concatenated and passed through the encoder. Besides the main task head, two more decoders are employed. After a high portion of the embeddings is masked, the two sets (each corresponding to an image) are separately fed through one decoder to reconstruct the two frames. The other decoder only receives as input the token embeddings of the search image and reconstructs the template image.

Other works focused on implementing MIM for point clouds. On the one hand,
[[[168](#bib.bib168)]] presented I2P-MAE, a method designed to learn better 3D features by reconstructing masked point clouds. The approach leverages 2D pre-trained models to keep the important point tokens visible while masking. Moreover, the 2D models are used to get the target representations for a semantic reconstruction loss that is applied on the visible tokens, after the decoder.
On the other hand, some works harnessed 2D images in their point-cloud methods.
[[[174](#bib.bib174)]] proposed a pre-training method for 3D point clouds by leveraging the corresponding 2D visual representation. The 3D points and their 2D projections are jointly encoded. The resulting encoding is randomly masked and passed through a two-stage decoder. First, there is a shared decoder, then each representation continues through a separate module to reconstruct both input modalities.
[[[171](#bib.bib171)]] utilized the mask reconstruction strategy in 3D segmentation due to the lack of supervised data, as well as the domain difference between training and testing data. During training, given a pair composed of a 2D image and a 3D point cloud, patches from one modality are masked and the model tries to estimate them using the other modality. A CNN backbone with a lower masking ratio is employed. Having two different datasets (source-labeled and target-unlabeled), the MIM is performed on both datasets, while the supervised task is performed only on the former.

The teacher-student framework was utilized in a couple of scenarios.
[[[169](#bib.bib169)]] combined self-supervised knowledge distillation and masked image modeling into a single framework. In the proposed pipeline, the teacher network processes an image from the same class as the student network. The student network processes a masked image, being trained to maximize the similarity between its class token and the teacher’s class token. In addition, the student is trained to distill the knowledge of the most similar tokens of the teacher.
The study of [[[170](#bib.bib170)]] integrated the MAE pipeline into a teacher-student setting for domain adaptive object detection. In this setup, the student network has a dual focus: it learns the detection task using labels generated by the teacher network, and concurrently, it undertakes the reconstruction of missing multi-scale features from the target images. This reconstruction aspect is pivotal, especially when the availability of pseudo-labels from the teacher is limited, as it significantly improves the model’s adaptability to the target domain, ensuring more robust and accurate object detection performance.

[[[175](#bib.bib175)]] evaluated the efficacy of using image generation as a self-supervised pre-training task, finding that it yields only marginal improvements in downstream recognition tasks when applied within a diffusion model framework. In response to this observation, the authors presented a novel strategy that merges MAE with diffusion models for self-supervised pre-training, focusing specifically on an inpainting task. This approach demonstrated competitive performance, aligning closely with state-of-the-art methods in image recognition, thus offering a compelling alternative to enhance pre-training effectiveness.

Inspired by MAE, [[[172](#bib.bib172)]] modified the MAE framework in order to boost performance when applied to downstream classification tasks in few-shot scenarios. Rather than reconstructing the original pixel space, the authors proposed to operate in the latent representation space of a frozen backbone. Two subsets of images, called support and query, are firstly embedded, the latter’s embeddings being masked. Then, the support embeddings are encoded, concatenated with mask tokens, and decoded to estimate the corresponding query embeddings, MSE being the computed reconstruction loss. After being pre-trained with a large labeled dataset, the method is further trained on a smaller dataset (containing few examples per class) in order to learn global information about each class.

### 3.10 Masking Strategy and Architecture

A number of studies have influenced both the masking strategy and the employed architecture. A notable proportion of these studies have multimodal inputs [[[26](#bib.bib26), [176](#bib.bib176), [177](#bib.bib177), [178](#bib.bib178), [179](#bib.bib179), [180](#bib.bib180)]], which typically requires distinct modules for each modality, coupled with carefully designed masking strategies to prevent information leakage between modalities [[[178](#bib.bib178), [180](#bib.bib180)]]. Additionally, some studies have adapted the MAE framework to be compatible with other popular architectures in computer vision, such as hierarchical transformers [[[181](#bib.bib181)]] and CNNs [[[145](#bib.bib145)]]. Furthermore, there are works that specifically develop architectural modules to directly influence the masking strategy employed during pre-training [[[67](#bib.bib67)]].

While ViT is the main architecture employed in MIM, convolutional layers were still found to be beneficial. [[[145](#bib.bib145)]] implemented the MAE framework for convolutional networks. One of the changes was to create the masks based on the deep feature maps of the encoder and resize them to the resolution of the input images. The second change was also in the encoder, where the authors used sparse convolutional layers to preserve the speed improvements brought by the masking.

[[[181](#bib.bib181)]] introduced a set of changes required by the Hierarchical Vision Transformer architectures in order to be compatible with the MAE framework, where the masked tokens are ignored from the input sequence. There are two problems when applied directly, one is the window attention with non-overlapping windows, and the other is the usage of convolutional and pooling layers. For the first issue, the authors’ solution is to group the tokens from uneven windows with a novel Optimal Grouping algorithm and then apply the mask attention. For the second issue, they opted for using sparse convolutions.
[[[180](#bib.bib180)]] made several contributions to adapt the MAE pre-training strategy to their scenario. Firstly, the linear projection layer of the ViT architecture is substituted with CNN layers. However, positional, viewpoint and timestep embeddings are still added. Secondly, the method operates on sequences of images that have multiple viewpoints, requiring a novel masking strategy, on the one hand, to fully mask one viewpoint per video frame, and on the other hand, to mask the latent feature maps of the intact frames. In order to facilitate the reconstruction task, the encoder is fed with tokens from different viewpoints, as well as from adjacent frames. This pre-training framework allows the authors to train a world model for visual robotic manipulation.

*Figure 16: The 4M framework for multiple modalities, courtesy of [[[26](#bib.bib26)]] (image available under the ArXiv perpetual license agreement). 4M belongs to the category of reconstruction-based methods shown in Figure [2](#S2.F2), as it tries to recover masked discrete tokens using the cross-entropy loss.*

Harnessing multiple sources of information during MIM pre-training was demonstrated to bring great benefits, especially for multimodal downstream tasks. [[[26](#bib.bib26)]] proposed an MAE framework that can be used with multiple modalities. Given a set of modalities, each is tokenized into a common representation form. Rather than using all tokens, only two subsets from each modality are sampled: one that is encoded and the other one that is masked and then reconstructed. A joint encoder is employed for all input types. Still, a modality embedding is added to each token to indicate its source. In the cross-attention layers of the decoder, the embeddings corresponding to one modality are masked from the rest, while using all resulting tokens from the encoder. Besides demonstrating good results on downstream tasks, the method shows promising cross-modal generative capabilities.
To leverage the information from multiple input data types and learn richer feature representations, [[[179](#bib.bib179)]] proposed to pre-train a network with multiple modalities. Their self-supervised method involved three types of data: an RGB image, a depth map and a semantic segmentation map. Using the ViT architecture, the modalities are split into patches and projected into tokens (with separate layers for each modality). Then, a great proportion of tokens from each modality is masked, while the rest (unmasked) are encoded (using a joint encoder) and concatenated with the masked tokens. A separate decoder for each data type is used to reconstruct the corresponding input. In the first stage of the decoder composed of a cross-attention layer, the tokens associated with the respective modality are given as queries, while the given keys and values are from all tokens. Experiments carried out on downstream tasks for all three data types indicate that the proposed strategy shows competitive results.
PiMAE [[[178](#bib.bib178)]] is a self-supervised framework based on MAE, which learns representations that capture interactions between point clouds and images. Overall, the approach is based on the usual reconstruction objective for each modality. However, in contrast to MAE, the masking strategy in this case is designed to be complementary between the two modalities. In terms of architectural changes, the encoder and decoder include some common blocks between the two modalities, but they also have modality specific layers.
[[[176](#bib.bib176)]] combined two sources of information (Hematoxylin and Eosin and Immunohistochemical staining images) to detect breast cancer, adopting MAE as the base for their method. Both images are split into patches, some of which are randomly masked, while the remaining visible patches are fed together through a ViT-based model. The resulting embeddings, together with the learnable mask embeddings, are further processed by two self-attention modules (one specific for each modality), as well as a cross-attention module (that processes all embeddings). Finally, two separate decoders reconstruct the original images, each receiving the modality-specific embeddings, as well as the inter-modal ones. More recently, [[[182](#bib.bib182)]] employed the MAE framework to learn robust cross-modal representations. Their primary contribution is the MultiModal 3D Interaction module, which takes as input the concatenation of masked 3D volume features from both video frames and LiDAR sensors. Built upon self-attention layers and feed-forward networks, this module outputs a representation that is then split along the channel dimension and fed into modality-specific decoders to reconstruct the original inputs.

[[[177](#bib.bib177)]] proposed a novel two-stage pre-training framework for video foundation models. The initial stage focuses on training the model to align the features extracted from masked frames with those derived from an image foundation model on unmasked frames. In the subsequent stage, the authors introduced a text encoder and a cross-modality decoder to further train the model for video-text matching and masked language modeling, while maintaining the training objective employed in the first stage.

Scale-MAE [[[183](#bib.bib183)]] introduced a pre-training method suitable for scale-dependent domains, specifically testing it on remote sensing data. The method is similar to the MAE framework, but it has some key changes. First, the positional embeddings that are added to the token embeddings depend on the Earth area covered in the image. Second, the authors changed the decoder to use a three-stage architecture. The first stage is a transformer-based block. The second stage performs upsampling with deconvolutional layers. The last stage employs Laplacian blocks to reconstruct the high and low frequency features.

[[[67](#bib.bib67)]] introduced a learnable masking module that facilitates curriculum learning within the MAE framework. Initially, the new module creates masks that are easy to reconstruct. The training objective is changed over time to adopt an adversarial role, progressively creating more challenging masks for the MAE to reconstruct. This dynamic adjustment of the training masks enhances the MAE’s ability to handle increasingly complex reconstruction tasks, thereby improving its learning efficiency and robustness.

### 3.11 Masking Strategy and Objective

A body of works have contributed to both the masking strategy and the objective function. These works either introduced a novel masking strategy that excels in hiding the salient information [[[1](#bib.bib1), [184](#bib.bib184), [40](#bib.bib40)]], or adapted the masking strategy to a different input type [[[122](#bib.bib122), [185](#bib.bib185)]]. Further, to maximize the benefit of using a new masking strategy, the objective function is altered to be more suitable for recovering the original input source [[[122](#bib.bib122), [39](#bib.bib39), [40](#bib.bib40), [186](#bib.bib186)]]. For example, [[[122](#bib.bib122)]] performed the masking in the discrete space of a dVAE and, due to this change, the reconstruction loss is the cross-entropy. Another example is the work of [[[185](#bib.bib185)]], in which the contrastive loss is employed to align the token representations of two different masked versions of the same image.

![Figure](x16.png)

*Figure 17: MST pipeline, courtesy of [[[1](#bib.bib1)]] (image licensed under CC BY 4.0). MST leverages an ${L}^{1}$ loss to reconstruct the patches that are masked according to the attention maps provided by the teacher, thus falling into the category of reconstruction-based methods depicted in Figure [2](#S2.F2). The cross-entropy loss is only used to synchronize the teacher and the student.*

A consistent number of papers in this section employed an objective function composed of multiple losses.
Masked Scene Contrast [[[39](#bib.bib39)]] is a framework for 3D representation learning that utilizes contrastive learning. This framework generates input pairs through a series of data augmentation techniques and applies complementary masking. The contrastive learning objective is then employed between the features of the unmasked and masked patches. Additionally, the framework incorporates a reconstruction loss to enhance learning efficacy.
[[[185](#bib.bib185)]] integrated MAE in their method for self-supervised video hashing. A video clip is first downsampled using a CNN, and then, a token is extracted for each frame. Two different subsets of token frames are sampled, each being then fed into an encoder and hashed. After some of the hashed embeddings are masked, a decoder is used to reconstruct the original frame tokens. A contrastive loss is added to maximize the similarity of the mean hash embeddings between the two subsets.

[[[186](#bib.bib186)]] slightly modified the MAE model by adding Gaussian noise to all pixels in the input image. With this change, besides reconstructing the masked regions, the objective is further extended to decode all denoised patches. The experiments demonstrated the superiority of this method over the original MAE.
[[[40](#bib.bib40)]] introduced some improvements to the teacher-student MIM based on contrastive pre-training. The first addition is a ranking component that divides the patches into two subsets: the ones that contain salient information and the meaningless ones. The former subset is passed to the student, while the latter is given to the teacher, both being partially masked. The first objective is to reconstruct the masked patches of the student, as they are harder to predict due to containing more salient information. A second loss is used to align the globally encoded representations of the two subsets by maximizing the similarity between the embeddings of the [CLS]* token. While the same encoder is used for both models, the gradients do not flow through the teacher.

Aside from multi-component losses, informed masking strategies were also employed in some studies.
The Masked Self-Supervised Transformer (MST) [[[1](#bib.bib1)]] selects patches for masking based on low responses as determined by attention maps from a teacher network, which is an EMA of the student network. The selected patches are replaced with a special token. The training of MST includes a dual-objective approach: a reconstruction loss to rebuild the masked inputs, and a cross-entropy loss designed to synchronize the teacher and student networks, particularly in class differentiation. A detailed overview of the pipeline is depicted in Figure [17](#S3.F17).
[[[184](#bib.bib184)]] tailored the MIM pre-training strategy for human perception tasks. They began by detecting the human parts and masking the patches corresponding to these parts, the objective being that of reconstructing the masked tokens from the visible ones. The authors also generated another masked view of the input image (sampling other human parts), aiming to align the global representations of both views (by applying a contrastive loss on the *[CLS]* tokens).

[[[5](#bib.bib5)]] introduced a masking strategy that selects important patches according to their similarity with a given text, computed in the CLIP embedding space. They further employed a contrastive loss to align text features with the features extracted from the unmasked patches.

[[[122](#bib.bib122)]] presented a method inspired by BERT to pre-train transformers from point cloud data. First, the points are partitioned into sub-clouds, and a PointNet is applied to extract point embeddings. With the resulting embeddings, the authors trained a dVAE, where the encoder is a tokenizer, mapping the continuous embeddings into discrete tokens. After this stage, the transformer pre-training is performed, in which the model receives a masked sequence of point embeddings and learns to output the discrete tokens. The masked tokens are replaced by a learnable token.

A couple of papers opted for a contrastive objective between two different views of an image obtained by distinct augmentations.
Taking a different masking SSL approach, [[[187](#bib.bib187)]] proposed a method whose objective is to be more robust for downstream tasks. On the input image, two different sets of augmentations are applied. The augmented images are encoded, and a projection layer transforms the encoded representations into multivariate Gaussian distributions. Finally, a series of learnable masks are applied. The difference between both masked probabilistic embeddings is computed, the objective being that of minimizing this difference.
[[[3](#bib.bib3)]] presented a pre-training framework that combines MIM and contrastive learning. Different subsets (a more aggressive one and a lighter one) of augmentations are applied to the input image to create two views, the latter having a portion of its tokens masked. Then, two vision transformer models (where the one for the unmasked image is an EMA of the other model) are used to encode the tokenized patches. The contrastive loss is applied only between the corresponding corrupted patches.

### 3.12 Masking Strategy and Task

The research highlighted in this subsection focuses on developing MIM pre-training methods that are effective on specific downstream tasks. These studies accomplish this by creating custom masking strategies [[[188](#bib.bib188), [189](#bib.bib189), [123](#bib.bib123), [190](#bib.bib190)]] or by transforming the input signal into a specialized feature space [[[191](#bib.bib191), [192](#bib.bib192), [33](#bib.bib33), [193](#bib.bib193), [194](#bib.bib194)]]. The custom masking strategies are created such that the features of interest are masked in the pre-training phase. For example, [[[188](#bib.bib188)]] learn representations that are meaningful for facial recognition by masking the regions that contain the eyes, nose and mouth, during pre-training.

To improve the masking strategy, some papers proposed an informed masking policy.
[[[188](#bib.bib188)]] used masked autoencoders to learn rich, generic, transferable and robust facial representations from face videos. The masking prioritizes specific tokens (those containing eyes, nose, mouth and hair). The learned representations are then tested on downstream tasks, such as facial attribute recognition, facial expression recognition, DeepFake detection, and lip synchronization.
Aiming to improve the performance on WSI classification, [[[195](#bib.bib195)]] studied several masking strategies to create a hard mining method useful for multiple instance learning. They observed that the best candidates for masking are the salient patches. To identify this type of patches during training, the authors proposed a pipeline in which the attention scores provided by a teacher network serve as indicators of patch saliency.
[[[190](#bib.bib190)]] addressed the shortcomings of gallbladder cancer detection in static images, proposing the use of video sequences instead. They adopted MAE as a pretext task, but presented an improved masking strategy that is able to hide the malignant regions more consistently, and thus learn a better representation of the disease. The masking strategy involved a Region Selection Network that generates a probability for each token, which is then used to sample the visible tokens.

A few papers adopted MIM to increase the quality of images.
The Saturated Mask AutoEncoder (SMAE), introduced by [[[193](#bib.bib193)]], is a two-stage approach for few-shot HDR imaging. The first stage focuses on representation learning, which is performed via masked image modeling. In this stage, the method creates two additional images from the original frame using exposure adjustment. Next, all three images are randomly masked with a high masking ratio before passing them to the model.
LEMaRT [[[192](#bib.bib192)]] is an effective pre-training framework when applied on image harmonization as a downstream task. In this approach, the masked patches are replaced with the patches taken from a perturbed version of the original image. The authors also investigated what is the best strategy for creating the masks, concluding that random masking works best for image harmonization.
Magnetic Resonance scans are generated in k-space and then transformed in the image domain with the inverse Fourier transform. To obtain an image with high quality and fidelity, the k-space needs to be fully sampled, but this is not realistic in most scenarios. To this end, [[[194](#bib.bib194)]] proposed to leverage the MAE framework by masking data in the k-space, represented in 3D: height, width and time, along the first dimension. By reconstructing the missing tokens via the $L^{1}$ loss, the adopted ViT models learn a rich feature representation that is able to estimate the unsampled data from k-space at inference. The estimated k-space is further refined using three sequential transformer-based decoders (one for each pair of dimensions), and employing the High-Dynamic Range loss after each decoder.

[[[33](#bib.bib33)]] presented a framework for representation learning and image generation. The applied pre-training method is similar to MAE [[[9](#bib.bib9)]], but the tokens are given by a VQ-GAN tokenizer and the masking ratio is variable.

[[[196](#bib.bib196)]] demonstrated that MAEs can be used in class incremental learning as a rehearsal-based method. The efficiency of MAEs, which require only a few patches, allows for the storage of more examples from previous tasks. In addition, the authors designed a two-branch MAE architecture to ensure higher quality reconstructions, one branch being responsible with inserting details into the image.

[[[189](#bib.bib189)]] introduced the TVC (text-guided video completion) task: the model needs to generate videos based on a subset of frames, while respecting some text instructions. Depending on the subset of frames, the task requires either prediction (future), rewinding (past) or infilling (between two moments). The authors proposed a training strategy that is based on masking frames, which addresses all three possible TVC subtasks.

[[[123](#bib.bib123)]] introduced a novel self-supervised pre-training technique designed for point cloud videos, employing a unique approach that involves masking point tubes. This method focused on training the model to accurately reconstruct these masked tubes. Simultaneously, the model is engaged in training on a temporal cardinality difference task. This dual training strategy enhances the model’s ability to understand both the spatial structure and temporal dynamics inherent in point cloud videos.

[[[191](#bib.bib191)]] argued that MIM alone is not sufficient for downstream tasks such as geometric matching. Therefore, they proposed paired MIM, which reconstructs pairs of masked images instead of a single masked image. Their study demonstrated that this pre-training task is more effective for geometric matching, because such tasks require the correlation between two images.

### 3.13 Features and Task

Several studies aiming to develop MIM methods optimized for specific downstream tasks achieved their goal by exploring various types of target features for masking. Most of these works are in the medical domain [[[197](#bib.bib197), [198](#bib.bib198), [199](#bib.bib199)]]. The remaining two are unrelated, MAGVLT [[[200](#bib.bib200)]] being a visual-language transformer, and MVD [[[201](#bib.bib201)]] being a distillation framework.

The adoption of MIM in the medical domain is highly motivated by the scarcity of annotated datasets.
[[[197](#bib.bib197)]] proposed a method to pre-train a model using MIM, which is able to process both 2D and 3D ophthalmic images. The authors developed a new module, called Unified Patch Embedding, consisting of two branches, one for each data type. The module divides the inputs into equal patches and then masks most of them. Then, a common encoder computes the latent representations of the visible patches. Finally, two decoders are employed: one that reconstructs the patches, and another one that estimates their gradient maps (composed of horizontal and vertical edge maps). The experiments showed that the method yields state-of-the-art performance in ophthalmic image classification.
To boost the performance gains of an MAE in the ultrasound imaging domain, [[[198](#bib.bib198)]] introduced an additional task during the pre-training stage. Due to the high noise-to-signal ratios in such images, they are initially blurred, so that masked patches are reconstructed, while the visible patches are deblurred. Thus, in contrast to the original MAE, all patches are passed through the encoder. The experiments on the downstream task of thyroid ultrasound image classification demonstrated leading results.
Different from the previous methods, [[[199](#bib.bib199)]] introduced a novel unsupervised domain adaptation framework based on MAE, in which the authors employed a convolutional architecture. The masked reconstruction task is applied to two input signals that are created from two volumetric scans: a local sub-volume and a global downsampled scan. When applied to the segmentation downstream task, the method uses a teacher-student framework. The teacher (an EMA of the student) generates a pseudo-label segmentation mask of a target domain image, and then, the student is trained using the resulting pair along with a sample from the source domain.

MAGVLT [[[200](#bib.bib200)]] is a non-autoregressive generative visual-language transformer trained jointly for image-to-text, text-to-image and image-text-to-image-text. The training objectives are three mask prediction losses, one for each task.

![Figure](x17.png)

*Figure 18: The MVD knowledge distillation pipeline, courtesy of [[[201](#bib.bib201)]] (image available under the ArXiv perpetual license agreement). MVD is classified as a reconstruction-based approach (belongs to the generic pipeline illustrated in Figure [2](#S2.F2)), as it uses an $L^{2}$ loss between the output features of its two decoders and the features provided by two teacher networks, one for video and one for images.*

[[[201](#bib.bib201)]] introduced Masked Video Distillation (MVD), a new method for self-supervised video representation learning, depicted in Figure [18](#S3.F18).
This approach has two stages. The first stage is to train masked image models and masked video models as teachers.
The second stage is to train a student with the representations learned by the teachers as target vectors. The masking is also used in this latter stage.

### 3.14 Architecture and Features

Some studies contributed in terms of both model architecture and target features. The modifications proposed by these works are tightly coupled. On the one hand, some works proposed new target features and consequently implemented architectural adjustments to support these features [[[202](#bib.bib202), [203](#bib.bib203)]]. On the other hand, other works extended the existing MIM frameworks [[[204](#bib.bib204), [205](#bib.bib205)]], or even introduced new pre-training frameworks based on MIM [[[206](#bib.bib206), [207](#bib.bib207)]].

Rather than using a single decoder, some works introduced multiple decoders in their pipeline. [[[204](#bib.bib204)]] proposed a couple of modifications to the original MAE pre-training. They used two decoders: one for image reconstruction (the original task) and one for feature representation estimation. The latter one tries to predict the feature representation of the masked patches, the ground-truth coming from an EMA replica of the MAE encoder. For both decoders, some information about the visible patches from the encoder is injected with the cross-attention layers: the former receives low-level context, while the latter is given high-level features.
Observing that masked pre-training negatively affects the final layers of a deep ViT model, [[[203](#bib.bib203)]] proposed Masked Image Residual Learning (MIRL). The framework consists of dividing a ViT along the depth into an even number of stages. A decoder is added for each stage, which is fed with the corresponding intermediate embeddings, as well as the masked tokens. While the decoders in the first half reconstruct the main components of the input image, the rest estimate the residuals, i.e. the differences between the target and the prediction.

[[[206](#bib.bib206)]] introduced a complex self-supervised self-distilled pre-training framework, demonstrating its capability on the 3D medical image segmentation downstream task. The first step is to generate two views of the same 3D image, split them into equally-sized patches, and then randomly mask them. Two encoder networks, teacher and student, are used, the former being updated as an exponential moving average with momentum of the latter. While the student processes only the masked patches of one view, the teacher fully encodes the same uncorrupted view, as well as the masked patches of the other view. Three independent single linear layers are then used to densely reconstruct the image from each embedded patch, estimate the masked patches and generate a global embedding of the image. Consequently, three losses are computed: the reconstruction loss between the predicted masked tokens of the student and the corresponding tokens encoded by the teacher, the cross-entropy between the global teacher’s and student’s embeddings of the two masked views, but also the reconstruction loss of the view predicted by the student.

Masked Shape Prediction (MSP) [[[202](#bib.bib202)]] uses geometric shape as prediction target for masked points. This target includes explicit shape context and deep shape features. Additionally, the architecture of MSP modifies cross-attention and self-attention layers to avoid possible masked shape leakage caused by including the masked part positions in the token embeddings. These modifications are designed to restrict the interaction among masked points.

[[[205](#bib.bib205)]] presented a knowledge distillation technique suitable for object detection, based on the MAE framework. The student network receives a masked image and learns to recover the missing multi-scale features provided by the teacher network. The student and the teacher networks are based on convolutional layers. Thus, in the design of the student, the authors used masked convolutions to avoid information leakage.

[[[207](#bib.bib207)]] introduced a masking pre-training framework that tackles the difference in training and test data distribution, the main idea being that of reconstructing images from other domains. The framework consists of one encoder and multiple decoders (one for each domain). First, the style-mixed image is obtained from the input image. The patches of the style-mixed image are masked, the visible tokens being passed through the encoder and all decoders to estimated the input image in all styles. The reconstruction loss is first employed for the predicted style corresponding to the input. The second stage of the framework involves taking the other estimated images, randomly masking their patches, and passing them through the autoencoder (with the decoder of the input style) to estimate the input image. The second objective is the reconstruction of the input style from all other inputs.

[[[208](#bib.bib208)]] proposed to integrate masked modeling for pre-training video-language tasks. A common cross-modal transformer is employed for the tokens of both modalities, with the objective of reconstructing them. While the text tokens follow the approach in BERT, the video frames are tokenized into discrete values with a dVAE, the masking being achieved at the pixel level by replacing image patches with zero. Their experiments attest the benefits of reconstructing discrete visual tokens.

### 3.15 Objective, Task and Theoretical Analysis

Two studies had an impact on the objective function and the downstream task, while also conducting a theoretical analysis. These works [[[41](#bib.bib41), [209](#bib.bib209)]] begin by conducting a deep analysis about different MIM aspects, and then, they provide solutions in order to improve them [[[41](#bib.bib41)]] or exploit the properties induced by MIM to implement a particular downstream task [[[209](#bib.bib209)]].

[[[41](#bib.bib41)]] unveiled several theoretical insights of the MAE paradigm. Initially, they uncovered a link between the MAE framework and contrastive learning principles, revealing that the reconstruction loss in MAE is analogous to the alignment loss found in contrastive learning. Further exploration provided theoretical assurances regarding the downstream efficacy of MAE models.
The connection with contrastive learning also implies the presence of feature collapse, a common challenge in contrastive learning, where aligning solely positive samples diminishes model effectiveness.
The researchers introduced Uniformity-enhanced MAE as a solution for the feature collapse problem. This adaptation modifies the loss objective to integrate a novel loss function, specifically designed to reduce feature similarity across unmasked views, thereby preserving feature diversity and enhancing model robustness.

MaskSketch [[[209](#bib.bib209)]] is a method to generate images from sketches using a masked generative transformer. In general, a masked generative transformer synthesizes new examples by accepting new tokens in consecutive iterations, and the accepted tokens are the ones above a certain threshold. In this case, the authors modified this threshold to depend on a distance computed between the self-attention maps of a sketch and the image that is being generated. Hence, the main observation is that the self-attention maps provided by a masked transformer are domain-invariant, preserving a similar structure for both sketches and natural images.

### 3.16 Masking Strategy, Architecture and Objective

A handful of works contributed to advancements in the masking strategy, the model architecture and the objective function. Some of these studies leveraged their three-fold contributions to tackle unique pre-training scenarios [[[210](#bib.bib210), [211](#bib.bib211)]]. For example, [[[210](#bib.bib210)]] presented a custom solution for images that contain text. Additionally, other studies discuss about the suboptimal choices made in previous MIM pipelines [[[212](#bib.bib212), [213](#bib.bib213)]] and try to improve them. For example, [[[213](#bib.bib213)]] imposed certain properties on the masked patches to avoid the patch correlation issue.

Adaptive Masking (AdaMAE) [[[212](#bib.bib212)]] employs an adaptive masking strategy for MAE performed by an additional neural network that assigns greater masking probabilities to the patches containing spatio-temporal information (a.k.a. foreground). The new neural network gives a vector of probabilities from which the sampling is performed. Thus, it cannot be trained with the reconstruction loss. The solution for this problem is to use an additional loss function based on the reconstruction one, where the terms are weighted with the probability vectors given by the adaptive network.

[[[210](#bib.bib210)]] presented a masking pre-training method specially designed for images with text. Random patches that contain text are masked, then passed through a CNN-based model to extract a feature representation. The feature maps are tokenized and then fed into a transformer. Following the Feature Pyramid Network (FPN) architecture [[[214](#bib.bib214)]], the resulting embedding is upsampled and combined with feature maps from multiple levels. The first objective is to predict the masked words, while the second goal is to reconstruct the corrupted pixels (with the help of the predicted word tokens). A ROI-alignment unit is used to associate the feature maps with the masked regions.

In order to pre-train a ViT for videos, [[[211](#bib.bib211)]] proposed to first transform the input video into the latent space of a VQ-VAE. Then, each frame is transformed into a set of tokens, and a varying high ratio of the tokens from the whole video sequence is masked. The attention units of the model are modified such that every token has access either to only the surrounding tokens in the same frame, or to a small neighboring region of tokens along all dimensions (i.e. including time). The objective is to estimate the masked tokens by minimizing their negative log-likelihood.

In their work, [[[8](#bib.bib8)]] introduced a general framework to learn various computer vision tasks with transformers. They framed the input and output of each task (e.g. detection and segmentation) as a sequence, and used an encoder-decoder transformer with bidirectional attention masks. To capture a rich context for each task, they employed MAE pre-training by reconstructing the sequence of tokens.

[[[213](#bib.bib213)]] analyzed the drawbacks in previous latent-space MIM applications [[[3](#bib.bib3)]], showing how a conventional reconstruction loss restricts diverse latent learning. To counter this, the authors introduced a patch discrimination objective to increase similarity between the predicted and the corresponding target latents. Additionally, [[[213](#bib.bib213)]] tackled the issue of patch correlation by changing the masking strategy. They used a high masking ratio (90%), a gap between adjacent patches, and a similarity constraint on visible and target patch sets. Further, the last contribution was an improved decoder architecture, suitable for latent representation prediction and incorporating self-attention, cross-attention layers, and visual cues from visible patches.

### 3.17 Masking Strategy, Features and Objective

Some papers had an impact on the masking strategy, the target features, as well as the objective function. The contributions to the masking strategies are highly varied: integrating new target features [[[215](#bib.bib215), [92](#bib.bib92)]], improving the masking policy based on guidance [[[216](#bib.bib216)]], and even leveraging multimodal data [[[217](#bib.bib217)]]. The objective function of these works is adapted as a result of the employed features.

Within the context of few-shot learning, [[[215](#bib.bib215)]] employed a masked autoencoder for reconstructing the latent embeddings rather than the original image. Additionally, each instance acts as a patch, and thus, the input consists of more images from the same class. After encoding the input, a high proportion of the input is masked and the decoder tries to reconstruct the masked embedding representation (given some identification variables). In this way, the backbone, i.e. the encoder, learns more discriminative features and has a better few-shot performance.

In their work, [[[92](#bib.bib92)]] introduced a pre-training method that is applicable to standard convolutional neural networks. The authors begin by removing patches from the image and substituting them with the mean value of the pixels. Different from previous methods, the masking tokens, corresponding to the previously erased patches, are introduced in the intermediate layers of the network. Aside from the reconstruction objective, another loss is added, which takes into account the difference between the discrete Fourier transforms of the original and reconstructed images. The role of the additional loss is to enhance the representations learned due to the interactions between patches at an intermediate level rather than at the lower levels.

[[[216](#bib.bib216)]] proposed to use MAE for reconstructing a normal estimation of the masked input image and compare it with the original, in order to detect anomalies. The pre-training of the model follows the original MAE, only optimizing the masking strategy for contiguous blocks of patches. As the aim is to reconstruct the abnormal regions, during inference, a proposal masking unit is employed, estimating the likely locations of the image patches that contain anomalies in order to mask them. The unit is composed of a feature extraction model that obtains the latent representations of the input image patches and some prototype normal images (one example from each normal class), as well as a normalizing flow model for computing the likelihood.

[[[217](#bib.bib217)]] focused on implementing a masking pre-training strategy for both visual and textual data. The main idea is to mask one modality and reconstruct the missing data using the other input type. First, each modality is encoded with its own encoder, and further processed by its specific cross-attention encoder (which is also fed with the other modality embedding). Finally, the masked tokens are decoded using a transformer for images, or a linear classification layer for text. Besides the reconstruction objective, two more losses are employed to align the embeddings of the modalities.

### 3.18 Masking Strategy, Architecture, Task and Objective

Few papers have contributed in four directions, having an extensive scope: the masking strategy, the model architecture, the downstream task and the objective function. Among these, one study stands out for its integration within a reinforcement learning pipeline [[[218](#bib.bib218)]], illustrating how MIM principles can be extended beyond conventional computer vision tasks. The remaining works delve into multimodal data [[[98](#bib.bib98), [120](#bib.bib120), [219](#bib.bib219)]], proposing innovative ways of combining diverse input sources, while simultaneously refining the pre-training methodology and the architectural design.

[[[218](#bib.bib218)]] proposed a Token-Critic algorithm to guide the synthesis of a non-autoregressive image generation model by predicting which tokens need to be sampled or not. To train the model, the following procedure is employed. First, a tokenized image (through a Vector-Quantized Autoencoder) is randomly masked. Then, using the transformer-based generative model, the image is reconstructed, while the critic must distinguish between the sampled and the original tokens. In this way, a completely masked tokenized image is gradually unmasked, using the Token-Critic to select which tokens to sample, eventually generating a new synthesized image.

![Figure](x18.png)

*Figure 19: The dendrogram generated through hierarchical clustering (Ward linkage) applied on TF-IDF vectors derived from the titles and abstracts of the papers. Zoom in supported for the electronic version. Best viewed in color.*

[[[98](#bib.bib98)]] harnessed MAE to pre-train an audio-video model by integrating all prominent objectives of MIM into a two-stage framework. The first stage consists of reconstructing the original inputs, while the second stage adopts the teacher-student distillation scheme, where the objective of the student is to reconstruct the predictions of the teacher. The teacher receives the full visible inputs, while the student is fed with the masked modalities. During both training stages, two different masked views of the same modality are generated, encoded, and two contrastive losses are computed: between embeddings of the same modality, as well as across modalities. Then, a joint encoder fuses the two modality embeddings, in the end being decoded with separate decoders. [[[219](#bib.bib219)]] also harnessed audiovisual data to enhance self-supervised representation learning. They proposed various pre-training architectures and objectives within a masked autoencoding framework to improve performance on audiovisual downstream classification tasks. The framework also supports multiple unimodal downstream tasks, using a single audiovisual pre-trained model.

[[[120](#bib.bib120)]] developed a lightweight MAE tailored for video anomaly detection. This MAE model leveraged weights derived from motion gradients to emphasize foreground objects in the reconstruction loss. Additionally, the authors enhanced the training procedure by introducing synthetic anomalies, while using normal frames as the target for the reconstruction loss. Later stages of training involve a student decoder, which learns to mimic the output of the main (teacher) decoder, further refining the detection process.

## 4 Automatic Clustering

To complement the manual taxonomy, we generate a dendrogram by executing a hierarchical clustering algorithm on TF-IDF vectors, which are computed by concatenating the titles and abstracts of the surveyed papers. We employ TF-IDF vectors to diminish the influence of stop words and increase the importance of content words. The hierarchical clustering is based on the Ward linkage, which aims to minimize the total within-cluster variance. Each TF-IDF vector starts as its own individual cluster. At each step, the two clusters that result in the smallest increase in total within-cluster variance when merged are combined. The process of merging is repeated until all TF-IDF vectors are combined into a single cluster, thus generating a dendrogram. We opted for Ward linkage in detriment of other alternatives to ensure that the resulting clusters are more homogeneous. We illustrate the resulting dendrogram in Figure [19](#S3.F19).

By analyzing the dendrogram, we identify several relevant clusters, which are annotated in Figure [19](#S3.F19). The first observed category of clusters is related to the input data type: temporal data, 3D data or 3D point clouds, video, audio, and even multimodal. Other identified clusters are represented by the domain in which the MAE framework was used, or by the downstream task it was applied on: medical imaging, anomaly detection, image classification in few-shot scenarios, object detection, object tracking and semantic segmentation. The clustering algorithm was also able to capture more complex concepts: training at test time, multi-view masked reconstruction, and domain or out-of-distribution adaptation. Probably one of the most notable clusters is formed by the papers that adopted the teacher-student MAE framework based on a contrastive objective. Another category consists of methods that employed diffusion models. Finally, two clusters related to the input masking strategy and theoretical analysis overlap with our manual taxonomy.

![Figure](extracted/6612948/fig_cifar100.png)

*Figure 20: Sample images from CIFAR-100 dataset.*

When compared with the manual taxonomy, we consider that the automatically generated clustering provides a distinct yet equally-useful categorization of the papers.

## 5 Datasets

Various datasets have been used by different masked image modeling frameworks. Some of the most representative datasets are: CIFAR-100, ImageNet-1K, MS COCO, UCF101, ShapeNet, CC3M, FFHQ, LAION-400M, and Visual Genome. A brief description of these datasets are provided in the following part.

![Figure](extracted/6612948/fig_imagenet1k.jpg)

*Figure 21: Sample images from ImageNet-1k dataset. Courtesy of [[[10](#bib.bib10)]].*

![Figure](extracted/6612948/fig_mscoco.png)

*Figure 22: Sample images from MS COCO dataset. Courtesy of [[[220](#bib.bib220)]].*

![Figure](extracted/6612948/fig_ffhq.jpg)

*Figure 23: Sample images from FFHQ dataset. Courtesy of [[[221](#bib.bib221)]].*

![Figure](extracted/6612948/fig_laion400m.png)

*Figure 24: Sample images with the the queries “blue cat” or “cat with blue eyes” from LAION-400M dataset. Courtesy of [[[222](#bib.bib222)]].*

CIFAR-100 [[[223](#bib.bib223)]] contains low-resolution images (32$\times$32 pixels), with 600 images for each of the 100 available classes. The classes are grouped into 20 categories, called super-classes. The labels range from animals to humans and objects. CIFAR-100 is typically used to demonstrate MIM methods on downstream tasks. Some sample images from this dataset are shown in Figure [20](#S4.F20).

ImageNet-1K [[[10](#bib.bib10)]] is a subset of one of the most popular datasets, namely ImageNet-21K. It consists of approximately 1.45 million images divided into 1000 object classes. ImageNet-1K provides a diverse range of high-resolution images organized according to the WordNet hierarchy. ImageNet-1K is generally used in the pre-training stage. Some examples from this dataset are shown in Figure [21](#S5.F21).

MS COCO [[[220](#bib.bib220)]] contains around 330k images, in which over 1.5 million object instances are annotated across 80 object categories. The dataset includes annotations for a wide variety of tasks: object detection, segmentation, and captioning.
Some sample images from this dataset are shown in Figure [22](#S5.F22).

UCF101 [[[224](#bib.bib224)]] is a dataset of 13,320 videos spanning 101 action categories. The videos cover a wide range of activities including human actions, sports, and daily activities. This dataset is typically used by MIM frameworks in the video domain.

Kinetics 400 [[[225](#bib.bib225)]] contains 400 human action classes, with at least 400 video clips for each action. Each clip lasts around 10 seconds and is sampled from a different YouTube video. The actions are focused on humans, covering a broad range of classes based on human-object interactions, such as playing instruments, as well as human-human interactions, such as shaking hands. Kinetics 400 is often employed in the pre-training stage of video-based MIM.

ShapeNet [[[226](#bib.bib226)]] is a dataset that contains 3D shapes, covering over 55 categories with about 51,300 unique 3D models. It includes both geometric and semantic annotations.

CC3M [[[227](#bib.bib227)]] consists of approximately 3.3 million images and their associated captions, describing the visual content of the images.

FFHQ [[[221](#bib.bib221)]] is a dataset composed of 70,000 high-quality (1024$\times$1024) images of diverse human faces, varying in age, ethnicity, and background. Some of the sample images from this dataset are shown in Figure [23](#S5.F23).

LAION-400M [[[222](#bib.bib222)]] is one of the primary datasets used for generative text-to-image models, consisting of 400 million image-text pairs. Some of the sample images from this dataset are shown in Figure [24](#S5.F24).

Visual Genome [[[228](#bib.bib228)]] contains over 100,000 images annotated with objects, attributes, and relationships. It includes region descriptions, object metadata, and dense annotations to facilitate scene understanding. It is mainly used within the visual question-answering task.

*Table 1: Statistics of the datasets that are commonly used in MIM literature.*

In Table [1](#S5.T1), we classify the datasets used for MIM based on their application, distinguishing between those employed for self-supervised pre-training and those used for downstream performance evaluation. Additionally, we provide the number of training and test samples for each dataset.

## 6 Performance Overview

We next provide an in-depth analysis of the performance achieved by MIM pre-training methods, when applied on leading computer vision architectures and popular benchmarks. We focus our analysis on three widely used datasets in MIM research: ImageNet-1K, MS COCO, and Kinetics-400.

In Table [2](#S6.T2), we present the results of different frameworks on ImageNet-1K.
Most of these works obtained accuracy improvements by introducing novel masking strategies [[[63](#bib.bib63), [9](#bib.bib9), [2](#bib.bib2)]] or target features [[[91](#bib.bib91), [63](#bib.bib63)]]. In general, the new masking strategies are based on the intuition that randomly masking patches, as proposed in MAE [[[9](#bib.bib9)]] and SimMIM [[[2](#bib.bib2)]], is suboptimal. Thus, recent methods [[[63](#bib.bib63), [58](#bib.bib58), [67](#bib.bib67)]] aim to identify semantically important patches, usually implementing masking strategies in an easy-to-hard pipeline, starting by masking the least important patches and progressively targeting the more semantically significant ones. Other works [[[54](#bib.bib54), [100](#bib.bib100)]] studied approaches to scale up the architectures and the number of samples used during pre-training. In contrast to these studies, some works [[[74](#bib.bib74)]] tried to limit the computational requirements of the networks, while preserving the performance of the previous works. Another significant observation that stems out from Table [2](#S6.T2) is that integrating MIM with generative models results in substantial performance enhancements, as presented in some recent studies [[[161](#bib.bib161), [175](#bib.bib175)]]. This finding supports the widely known view that generative tasks yield robust world representations. Moreover, it demonstrates the orthogonality between MIM and generative models, suggesting complementary benefits when combined.

*Table 2: Performance on ImageNet-1K (IN1K) of different MIM pre-training schemes.*

In Table [3](#S6.T3), we present a performance overview of different MIM frameworks on the MS COCO dataset. For object detection, we report the mean Average Precision for bounding boxes (mAPbox), and for segmentation tasks, we provide the mean Average Precision for masks (mAPmask). EVA [[[93](#bib.bib93)]] stands out as the most effective method, concentrating on the expansion of training examples and network parameters. According to Table [3](#S6.T3), its pre-training dataset integrates four distinct datasets, collectively summing over 29 million images. The closest result to EVA is obtained by SimMIM [[[2](#bib.bib2)]], when applied to a similar size architecture (SwinV2-G), but with much fewer images in the pre-training phase.
As demonstrated in Table [3](#S6.T3), the MAE strategy exhibits suboptimal performance when compared with alternative approaches applied to ViT-B and ViT-L models. Notably, among the methods outperforming MAE on ViT-B, two incorporate convolutional layers into their frameworks [[[113](#bib.bib113), [140](#bib.bib140)]]. This observation suggests that employing a hybrid architecture that combines both transformer and convolutional layers may offer significant improvements in object detection and segmentation.

In terms of training objectives, based on the results reported in Tables [2](#S6.T2) and [3](#S6.T3), the reconstruction loss remains the primary driver for high performance in MIM methods across both ImageNet-1K and MS COCO. At the same time, the contrastive loss often serves as a complementary component. For example, one of the top-performing methods, CMAE [[[4](#bib.bib4)]], leverages both reconstruction and contrastive losses.

*Table 3: Performance on MS COCO of different MIM pre-training schemes.*

Results on video recognition are included in Table [4](#S6.T4).
We select the Kinetics-400 dataset for this analysis because it is the most frequently used video dataset across the reviewed studies. A common practice among the studies listed in Table [4](#S6.T4) is their incorporation of large-scale image datasets during the pre-training phase, alongside a video dataset. Additionally, an analysis of the performance outcomes from EVA [[[93](#bib.bib93)]] and VideoMAEv2 [[[73](#bib.bib73)]] on the ViT-G architecture reveals that the size of the pre-training dataset plays an important role in determining the final performance of the model. Larger datasets tend to provide more effective and generalized capabilities in complex video processing tasks.

Consistent with what we observed in image classification, the integration of diffusion models and MIM proves beneficial in the video domain as well. Notably, the DiffMAE model [[[175](#bib.bib175)]] achieves results that are competitive with those of EVA [[[93](#bib.bib93)]], although DiffMAE is operating with a substantially smaller model size. This finding underscores again the effectiveness of combining generative diffusion models with MIM techniques. However, a significant factor contributing to this performance enhancement is the utilization of a large dataset, WIT400M, during the pre-training phase. When the model is pre-trained solely with the smaller Kinetics-400 dataset, its results fall short of those achieved by VideoMAE [[[51](#bib.bib51)]]. This underscores the critical importance of employing large-scale datasets in the pre-training phase to maximize model effectiveness.

*Table 4: Performance on Kinetics-400 (K400) of different MIM pre-training schemes.*

Other significant observations, based on the results presented in Table [4](#S6.T4), refer to the masking strategy and the types of features that are reconstructed. First, in videos, the temporal correlation between frames [[[51](#bib.bib51)]] can introduce a bias in the model, allowing it to reconstruct masked patches by leveraging information from adjacent frames. To alleviate this issue, OmniMAE [[[54](#bib.bib54)]] randomly masks patches from videos, similar to MAE [[[9](#bib.bib9)]], but uses a higher masking ratio, $95\%$ instead of $75\%$. However, this masking strategy performs poorly compared with those used in VideoMAE [[[51](#bib.bib51)]] and VideoMAEv2 [[[73](#bib.bib73)]], which are designed to mask temporal tubes instead of spatial patches. The latter approach is more effective in preventing information leakage from adjacent frames. Second, the top performing methods, such as MVD [[[201](#bib.bib201)]] and EVA [[[93](#bib.bib93)]], benefit from the use of high-level features as target features for reconstruction. These features help address the tendency of pixel-level MIM models to focus on spatial and structural information, leading to improved performance on downstream tasks.

## 7 Closing Remarks and Future Directions

In this paper, we highlighted two strategies for applying masked image modeling, one based on reconstruction and one based on contrastive learning. Moreover, we presented how both methods are great pre-training approaches for feature learning. Although their objectives are different, the theoretical analysis shows that they are equivalent. Furthermore, we provided a review of the most recent research advancements in masked image modeling, and explained how this pre-training strategy was implemented for various tasks.

Through our work, we aimed to give a better overview of masked image modeling, simplifying the effort needed by the research community and the industry to analyze the literature. We believe that both the manual taxonomy and the hierarchical clustering dendrogram are great resources for all individuals interested in learning more about masked image modeling, or how to apply this technique to their specific use case.

As previously mentioned, masked pre-training started from natural language processing, and it was later adopted in vision. Over time, masked image modeling was integrated into multiple downstream tasks and this is perhaps the main research direction that will continue, especially in domains or tasks with low quantities of annotated data, such as the medical domain.

While masked image modeling (MIM) has achieved remarkable progress in self-supervised visual representation learning, several promising directions remain open for future exploration. One important area is the extension of MIM to multi-modal and cross-modal learning scenarios, such as integrating vision with language or audio signals, which could further enhance contextual understanding. Each modality holds a different type of information, and these can be combined to learn richer feature representations. Distilling the knowledge of each modality into a single network could represent a stepping stone in artificial intelligence.

Another avenue is the development of more efficient masking strategies and reconstruction objectives that better align with downstream tasks beyond classification, such as segmentation, detection, or 3D understanding. While a random strategy may perform well, it has been demonstrated that an informed masking policy that focuses on hiding the salient information of the input is superior [[[67](#bib.bib67)]]. Thus, future endeavors may attempt to formulate various guided masking strategies, some of which can be specific to the downstream task. Additionally, there is growing interest in making MIM more resource-efficient.

Another future direction is the integration of physical priors or domain-specific laws into the MIM framework. For domains where data is governed by known physical constraints (e.g. medical imaging, climate science, robotics), embedding known physical constraints or conservation laws into the masking or reconstruction process could guide the model toward more meaningful and generalizable representations.

Finally, understanding the theoretical underpinnings of why and how MIM works so effectively remains an open challenge. Insights on this path could drive the design of next-generation self-supervised learning algorithms.

## Declarations

Funding. This research is supported by the project “Romanian Hub for Artificial Intelligence - HRIA”, Smart Growth, Digitization and Financial Instruments Program, 2021-2027, MySMIS no. 334906. This work was also supported by a grant of the Ministry of Research, Innovation and Digitization, CCCDI - UEFISCDI, project number PN-IV-P7-7.1-PED-2024-1856, within PNCDI IV.

Conflict of interest. The authors have no conflicts of interest to declare that are relevant to the content of this article.

Availability of data and materials. This paper is a survey paper, hence it does not report new experiments on any dataset.

## References

-
\bibcommenthead

-
Li et al. [2021]

Li Z, Chen Z, Yang F, Li W, Zhu Y, Zhao C, et al.

MST: Masked Self-Supervised Transformer for Visual Representation.

In: Proceedings of NeurIPS. vol. 34; 2021. p. 13165–13176.

-
Xie et al. [2022]

Xie Z, Zhang Z, Cao Y, Lin Y, Bao J, Yao Z, et al.

SimMIM: A Simple Framework for Masked Image Modeling.

In: Proceedings of CVPR; 2022. p. 9653–9663.

-
Yi et al. [2023]

Yi K, Ge Y, Li X, Yang S, Li D, Wu J, et al.

Masked image modeling with denoising contrast.

In: Proceedings of ICLR; 2023. .

-
Huang et al. [2023]

Huang Z, Jin X, Lu C, Hou Q, Cheng MM, Fu D, et al.

Contrastive masked autoencoders are stronger vision learners.

IEEE Transactions on Pattern Analysis and Machine Intelligence.
2023;46(4):2506–2517.

-
Fan et al. [2024]

Fan D, Wang J, Liao S, Zhang Z, Bhat V, Li X.

Text-Guided Video Masked Autoencoder.

In: Proceedings of ECCV; 2024. p. 282–298.

-
Guo et al. [2024]

Guo Y, Sun S, Ma S, Zheng K, Bao X, Ma S, et al.

CrossMAE: Cross-Modality Masked Autoencoders for Region-Aware
Audio-Visual Pre-Training.

In: Proceedings of CVPR; 2024. p. 26721–26731.

-
Tschannen et al. [2024]

Tschannen M, Eastwood C, Mentzer F.

GIVT: Generative Infinite-Vocabulary Transformers.

In: Proceedings of ECCV; 2024. p. 292–309.

-
Qiu et al. [2024]

Qiu H, Huang J, Gao P, Lu L, Zhang X, Lu S.

Masked AutoDecoder is Effective Multi-Task Vision Generalist.

In: Proceedings of CVPR; 2024. p. 14152–14161.

-
He et al. [2022]

He K, Chen X, Xie S, Li Y, Dollár P, Girshick R.

Masked autoencoders are scalable vision learners.

In: Proceedings of CVPR; 2022. p. 16000–16009.

-
Deng et al. [2009]

Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L.

ImageNet: A large-scale hierarchical image database.

In: Proceedings of CVPR; 2009. p. 248–255.

-
Russakovsky
et al. [2015]

Russakovsky O, Deng J, Su H, Krause J, Satheesh S, Ma S, et al.

ImageNet Large Scale Visual Recognition Challenge.

International Journal of Computer Vision. 2015;115:211–252.

-
Doersch et al. [2015]

Doersch C, Gupta A, Efros AA.

Unsupervised visual representation learning by context prediction.

In: Proceedings of ICCV; 2015. p. 1422–1430.

-
Noroozi and Favaro [2016]

Noroozi M, Favaro P.

Unsupervised learning of visual representations by solving jigsaw
puzzles.

In: Proceedings of ECCV; 2016. p. 69–84.

-
Wang and Gupta [2015]

Wang X, Gupta A.

Unsupervised learning of visual representations using videos.

In: Proceedings of ICCV; 2015. p. 2794–2802.

-
Pathak et al. [2017]

Pathak D, Girshick R, Dollár P, Darrell T, Hariharan B.

Learning features by watching objects move.

In: Proceedings of CVPR; 2017. p. 2701–2710.

-
Zhang et al. [2016]

Zhang R, Isola P, Efros AA.

Colorful image colorization.

In: Proceedings of ECCV; 2016. p. 649–666.

-
Gidaris et al. [2018]

Gidaris S, Singh P, Komodakis N.

Unsupervised representation learning by predicting image rotations.

In: Proceedings of ICLR; 2018. .

-
Devlin et al. [2019]

Devlin J, Chang MW, Lee K, Toutanova K.

BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding.

In: Proceedings of NAACL; 2019. p. 4171–4186.

-
Vincent et al. [2010]

Vincent P, Larochelle H, Lajoie I, Bengio Y, Manzagol PA, Bottou L.

Stacked denoising autoencoders: Learning useful representations in a
deep network with a local denoising criterion.

Journal of Machine Learning Research. 2010;11(12):3371–3408.

-
Pathak et al. [2016]

Pathak D, Krahenbuhl P, Donahue J, Darrell T, Efros AA.

Context encoders: Feature learning by inpainting.

In: Proceedings of CVPR; 2016. p. 2536–2544.

-
Zhao et al. [2021]

Zhao Y, Wang G, Luo C, Zeng W, Zha ZJ.

Self-supervised visual representations learning by contrastive mask
prediction.

In: Proceedings of ICCV; 2021. p. 10160–10169.

-
Dosovitskiy
et al. [2021]

Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T,
et al.

An image is worth 16x16 words: Transformers for image recognition at
scale.

In: Proceedings of ICLR; 2021. .

-
Chen et al. [2022]

Chen Y, Liu Y, Jiang D, Zhang X, Dai W, Xiong H, et al.

SdAE: Self-distillated Masked Autoencoder.

In: Proceedings of ECCV; 2022. p. 108–124.

-
Gandelsman
et al. [2022]

Gandelsman Y, Sun Y, Chen X, Efros A.

Test-Time Training with Masked Autoencoders.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 29374–29385.

-
Lee et al. [2023]

Lee Y, Willette JR, Kim J, Lee J, Hwang SJ.

Exploring the role of mean teachers in self-supervised masked
auto-encoders.

In: Proceedings of ICLR; 2023. .

-
Mizrahi et al. [2023]

Mizrahi D, Bachmann R, Kar O, Yeo T, Gao M, Dehghan A, et al.

4M: Massively Multimodal Masked Modeling.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 58363–58408.

-
Li et al. [2023]

Li S, Zhang L, Wang Z, Wu D, Wu L, Liu Z, et al.

Masked Modeling for Self-supervised Representation Learning on
Vision and Beyond.

arXiv preprint arXiv:240100897. 2023;.

-
Zhang et al. [2023]

Zhang C, Zhang C, Song J, Yi JSK, Kweon IS.

A survey on masked autoencoder for visual self-supervised learning.

In: Proceedings of IJCAI; 2023. p. 6805–6813.

-
Zhou and Liu [2023]

Zhou Z, Liu X.

Masked Autoencoders in Computer Vision: A Comprehensive Survey.

IEEE Access. 2023;11:113560–113579.

-
Ristea et al. [2022]

Ristea NC, Madan N, Ionescu RT, Nasrollahi K, Khan FS, Moeslund TB, et al.

Self-Supervised Predictive Convolutional Attentive Block for Anomaly
Detection.

In: Proceedings of CVPR; 2022. p. 13576–13586.

-
Madan et al. [2024]

Madan N, Ristea NC, Ionescu RT, Nasrollahi K, Khan FS, Moeslund TB, et al.

Self-Supervised Masked Convolutional Transformer Block for Anomaly
Detection.

IEEE Transactions on Pattern Analysis and Machine Intelligence.
2024;46(1):525–542.

-
Chang et al. [2022]

Chang H, Zhang H, Jiang L, Liu C, Freeman WT.

MaskGIT: Masked Generative Image Transformer.

In: Proceedings of CVPR; 2022. p. 11315–11325.

-
Li et al. [2023]

Li T, Chang H, Mishra S, Zhang H, Katabi D, Krishnan D.

MAGE: MAsked Generative Encoder to Unify Representation Learning and
Image Synthesis.

In: Proceedings of CVPR; 2023. p. 2142–2152.

-
Esser et al. [2021]

Esser P, Rombach R, Ommer B.

Taming Transformers for High-Resolution Image Synthesis.

In: Proceedings of CVPR; 2021. p. 12873–12883.

-
Zhu et al. [2024]

Zhu Y, Li B, Zhang H, Li X, Xu L, Bing L.

Stabilize the Latent Space for Image Autoregressive Modeling: A
Unified Perspective.

In: Proceedings of NeurIPS; 2024. p. 28636–28661.

-
Chen et al. [2023]

Chen Y, Yuan J, Tian Y, Geng S, Li X, Zhou D, et al.

Revisiting Multimodal Representation in Contrastive Learning: From
Patch and Token Embeddings to Finite Discrete Tokens.

In: Proceedings of CVPR; 2023. p. 15095–15104.

-
Wang et al. [2024]

Wang L, Zhao Y, Zhang Z, Feng J, Liu S, Kang B.

Image Understanding Makes for A Good Tokenizer for Image
Generation.

In: Proceedings of NeurIPS; 2024. p. 31015–31035.

-
Li et al. [2025]

Li S, Zhang L, Wang Z, Tian J, Tan C, Liu Z, et al.

MergeVQ: A Unified Framework for Visual Generation and
Representation with Disentangled Token Merging and Quantization.

In: Proceedings of CVPR; 2025. p. 19713–19723.

-
Wu et al. [2023]

Wu X, Wen X, Liu X, Zhao H.

Masked Scene Contrast: A Scalable Framework for Unsupervised 3D
Representation Learning.

In: Proceedings of CVPR; 2023. p. 9415–9424.

-
Zhang et al. [2023]

Zhang S, Zhu F, Zhao R, Yan J.

Contextual image masking modeling via synergized contrasting without
view augmentation for faster and better visual pretraining.

In: Proceedings of ICLR; 2023. .

-
Zhang et al. [2022]

Zhang Q, Wang Y, Wang Y.

How Mask Matters: Towards Theoretical Understandings of Masked
Autoencoders.

In: Proceesdings of NeurIPS; 2022. p. 27127–27139.

-
Chen et al. [2020]

Chen T, Kornblith S, Norouzi M, Hinton G.

A simple framework for contrastive learning of visual
representations.

In: Proceedings of ICML; 2020. p. 1597–1607.

-
Grill et al. [2020]

Grill JB, Strub F, Altché F, Tallec C, Richemond PH, Buchatskaya E, et al.

Bootstrap your own latent a new approach to self-supervised learning.

In: Proceedings of NeurIPS; 2020. p. 21271–21284.

-
He et al. [2020]

He K, Fan H, Wu Y, Xie S, Girshick R.

Momentum Contrast for Unsupervised Visual Representation Learning.

In: Proceedings of CVPR; 2020. p. 9726–9735.

-
Ghiţă and
Ionescu [2024]

Ghiţă A, Ionescu RT.

A New Loss for Image Retrieval: Class Anchor Margin.

In: Proceedings of PAKDD; 2024. p. 43–54.

-
Chen et al. [2021]

Chen P, Liu S, Jia J.

Jigsaw Clustering for Unsupervised Visual Representation Learning.

In: Proceedings of CVPR; 2021. p. 11526–11535.

-
Wei et al. [2018]

Wei D, Lim J, Zisserman A, Freeman WT.

Learning and Using the Arrow of Time.

In: Proceedings of CVPR; 2018. p. 8052–8060.

-
Caron et al. [2021]

Caron M, Touvron H, Misra I, Jégou H, Mairal J, Bojanowski P, et al.

Emerging properties in self-supervised vision transformers.

In: Proceedings of ICCV; 2021. p. 9650–9660.

-
Caron et al. [2020]

Caron M, Misra I, Mairal J, Goyal P, Bojanowski P, Joulin A.

Unsupervised learning of visual features by contrasting cluster
assignments.

In: Proceedings of NeurIPS; 2020. p. 9912–9924.

-
Bardes et al. [2022]

Bardes A, Ponce J, LeCun Y.

VICReg: Variance-Invariance-Covariance Regularization For
Self-Supervised Learning.

In: Proceedings of ICLR; 2022. .

-
Tong et al. [2022]

Tong Z, Song Y, Wang J, Wang L.

VideoMAE: Masked Autoencoders are Data-Efficient Learners for
Self-Supervised Video Pre-Training.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 10078–10093.

-
Shi et al. [2022]

Shi B, Hsu WN, Lakhotia K, Mohamed A.

Learning audio-visual speech representation by masked multimodal
cluster prediction.

In: Proceedings of ICLR; 2022. .

-
Chen et al. [2023]

Chen H, Gu J, Liu Y, Magid SA, Dong C, Wang Q, et al.

Masked image training for generalizable deep image denoising.

In: Proceedings of CVPR; 2023. p. 1692–1703.

-
Girdhar et al. [2023]

Girdhar R, El-Nouby A, Singh M, Alwala KV, Joulin A, Misra I.

OmniMAE: Single Model Masked Pretraining on Images and Videos.

In: Proceedings of CVPR; 2023. p. 10406–10417.

-
Liu et al. [2023]

Liu J, Huang X, Zheng J, Liu Y, Li H.

MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of
Hierarchical Vision Transformers.

In: Proceedings of CVPR; 2023. p. 6252–6261.

-
Pang et al. [2022]

Pang Y, Wang W, Tay FE, Liu W, Tian Y, Yuan L.

Masked autoencoders for point cloud self-supervised learning.

In: Proceedings of ECCV; 2022. p. 604–621.

-
Kakogeorgiou
et al. [2022]

Kakogeorgiou I, Gidaris S, Psomas B, Avrithis Y, Bursuc A, Karantzalos K,
et al.

What to hide from your students: Attention-guided masked image
modeling.

In: Proceedings of ECCV; 2022. p. 300–318.

-
Li et al. [2022]

Li G, Zheng H, Liu D, Wang C, Su B, Zheng C.

SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 14290–14302.

-
Voleti et al. [2022]

Voleti V, Jolicoeur-Martineau A, Pal C.

MCVD: Masked Conditional Video Diffusion for Prediction, Generation,
and Interpolation.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 23371–23385.

-
Huang et al. [2022]

Huang J, Cui K, Guan D, Xiao A, Zhan F, Lu S, et al.

Masked generative adversarial networks are data-efficient generation
learners.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 2154–2167.

-
Chen et al. [2022]

Chen Z, Du Y, Hu J, Liu Y, Li G, Wan X, et al.

Multi-Modal Masked Autoencoders for Medical Vision-and-Language
Pre-Training.

In: Proceedings of MICCAI; 2022. p. 679–689.

-
Xiao et al. [2023]

Xiao Y, Tang Z, Wei P, Liu C, Lin L.

Masked images are counterfactual samples for robust fine-tuning.

In: Proceedings of CVPR; 2023. p. 20301–20310.

-
Wang et al. [2023]

Wang H, Song K, Fan J, Wang Y, Xie J, Zhang Z.

Hard patches mining for masked image modeling.

In: Proceedings of CVPR; 2023. p. 10375–10385.

-
Xu et al. [2023]

Xu M, Xu M, He T, Ouyang W, Wang Y, Han X, et al.

MM-3DScene: 3D Scene Understanding by Customizing Masked Modeling
with Informative-Preserved Reconstruction and Self-Distilled Consistency.

In: Proceedings of CVPR; 2023. p. 4380–4390.

-
Wu et al. [2023]

Wu Q, Yang T, Liu Z, Wu B, Shan Y, Chan AB.

DropMAE: Masked Autoencoders with Spatial-Attention Dropout for
Tracking Tasks.

In: Proceedings of CVPR; 2023. p. 14561–14571.

-
Lin et al. [2023]

Lin Y, Wei C, Wang H, Yuille A, Xie C.

SMAUG: Sparse Masked Autoencoder for Efficient Video-Language
Pre-training.

In: Proceedings of ICCV; 2023. p. 2459–2469.

-
Madan et al. [2024]

Madan N, Ristea NC, Nasrollahi K, Moeslund TB, Ionescu RT.

CL-MAE: Curriculum-Learned Masked Autoencoders.

In: Proceedings of WACV; 2024. p. 2492–2502.

-
Jarca et al. [2024]

Jarca A, Croitoru FA, Ionescu RT.

CBM: Curriculum by Masking.

In: Proceedings of ECAI; 2024. p. 314–321.

-
Liu et al. [2021]

Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, et al.

Swin Transformer: Hierarchical Vision Transformer using Shifted
Windows.

In: Proceedings of ICCV; 2021. p. 9992–10002.

-
Hinojosa et al. [2024]

Hinojosa C, Liu S, Ghanem B.

ColorMAE: Exploring Data-Independent Masking Strategies in Masked
AutoEncoders.

In: Proceedings of ECCV; 2024. p. 432–449.

-
Liu et al. [2023]

Liu Z, Gui J, Luo H.

Good helper is around you: Attention-driven masked image modeling.

In: Proceedings of AAAI; 2023. p. 1799–1807.

-
Xie et al. [2023]

Xie J, Li W, Zhan X, Liu Z, Ong YS, Loy CC.

Masked frequency modeling for self-supervised visual pre-training.

In: Proceedings of ICLR; 2023. .

-
Wang et al. [2023]

Wang L, Huang B, Zhao Z, Tong Z, He Y, Wang Y, et al.

VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking.

In: Proceedings of CVPR; 2023. p. 14549–14560.

-
Huang et al. [2023]

Huang B, Zhao Z, Zhang G, Qiao Y, Wang L.

MGMAE: Motion Guided Masking for Video Masked Autoencoding.

In: Proceedings of ICCV; 2023. p. 13493–13504.

-
Gupta et al. [2023]

Gupta A, Wu J, Deng J, Li FF.

Siamese masked autoencoders.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 40676–40693.

-
Hsu et al. [2021]

Hsu WN, Sriram A, Baevski A, Likhomanenko T, Xu Q, Pratap V, et al.

Robust wav2vec 2.0: Analyzing domain shift in self-supervised
pre-training.

In: Proceedings of INTERSPEECH; 2021. p. 721–725.

-
Yang et al. [2023]

Yang Q, Li W, Li B, Yuan Y.

MRM: Masked Relation Modeling for Medical Image Pre-Training with
Genetics.

In: Proceedings of ICCV; 2023. p. 21452–21462.

-
Wang et al. [2023]

Wang Z, Lyu J, Tang X.

autoSMIM: Automatic Superpixel-Based Masked Image Modeling for Skin
Lesion Segmentation.

IEEE Transactions on Medical Imaging. 2023;42(12):3501–3511.

-
Achanta et al. [2012]

Achanta R, Shaji A, Smith K, Lucchi A, Fua P, Süsstrunk S.

SLIC superpixels compared to state-of-the-art superpixel methods.

IEEE Transactions on Pattern Analysis and Machine Intelligence.
2012;34(11):2274–2282.

-
Xie et al. [2023]

Xie Y, Gu L, Harada T, Zhang J, Xia Y, Wu Q.

MedIM: Boost Medical Image Representation via Radiology
Report-guided Masking.

In: Proceedings of MICCAI; 2023. p. 13–23.

-
Almalki and Latecki [2023]

Almalki A, Latecki LJ.

Self-supervised learning with masked image modeling for teeth
numbering, detection of dental restorations, and instance segmentation in
dental panoramic radiographs.

In: Proceedings of WACV; 2023. p. 5594–5603.

-
Mao et al. [2023]

Mao Y, Deng J, Zhou W, Fang Y, Ouyang W, Li H.

Masked motion predictors are strong 3D action representation
learners.

In: Proceedings of ICCV; 2023. p. 10181–10191.

-
Song et al. [2023]

Song H, Feng M, Zhou W, Li H.

MA2CL: Masked attentive contrastive learning for multi-agent
reinforcement learning.

In: Proceedings of IJCAI; 2023. p. 4226–4234.

-
Zhao et al. [2023]

Zhao R, Zhan M, Deng X, Wang Y, Wang Y, Gui G, et al.

Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic
Transformer with Multi-Level Flow Representation.

In: Proceedings of AAAI; 2023. p. 5420–5427.

-
Bao et al. [2022]

Bao H, Dong L, Piao S, Wei F.

BEiT: BERT Pre-Training of Image Transformers.

In: Proceedings of ICLR; 2022. .

-
Zhou et al. [2022]

Zhou J, Wei C, Wang H, Shen W, Xie C, Yuille A, et al.

iBOT: Image BERT Pre-Training with Online Tokenizer.

In: Proceedings of ICLR; 2022. .

-
Liang et al. [2022]

Liang Y, Zhao S, Yu B, Zhang J, He F.

MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis.

In: Proceedings of ECCV; 2022. p. 37–54.

-
Sun et al. [2023]

Sun X, Chen P, Chen L, Li C, Li TH, Tan M, et al.

Masked motion encoding for self-supervised video representation
learning.

In: Proceedings of CVPR; 2023. p. 2235–2245.

-
Hou et al. [2023]

Hou J, Dai X, He Z, Dai A, Nießner M.

Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D
Priors.

In: Proceedings of CVPR; 2023. p. 13510–13519.

-
Tian et al. [2023]

Tian X, Ran H, Wang Y, Zhao H.

GeoMAE: Masked Geometric Target Prediction for Self-supervised Point
Cloud Pre-Training.

In: Proceedings of CVPR; 2023. p. 13570–13580.

-
Wei et al. [2022]

Wei C, Fan H, Xie S, Wu CY, Yuille A, Feichtenhofer C.

Masked feature prediction for self-supervised visual pre-training.

In: Proceedings of CVPR; 2022. p. 14668–14678.

-
Li et al. [2022]

Li S, Wu D, Wu F, Zang Z, Li S, et al.

Architecture-Agnostic Masked Image Modeling–From ViT back to CNN.

In: Proceedings of ICML; 2022. p. 20149–20167.

-
Fang et al. [2023]

Fang Y, Wang W, Xie B, Sun Q, Wu L, Wang X, et al.

EVA: Exploring the Limits of Masked Visual Representation Learning
at Scale.

In: Proceedings of CVPR; 2023. p. 19358–19369.

-
Xue et al. [2023]

Xue H, Gao P, Li H, Qiao Y, Sun H, Li H, et al.

Stare at What You See: Masked Image Modeling without
Reconstruction.

In: Proceedings of CVPR; 2023. p. 22732–22741.

-
Yang et al. [2022]

Yang Z, Li Z, Shao M, Shi D, Yuan Z, Yuan C.

Masked generative distillation.

In: Proceedings of ECCV; 2022. p. 53–69.

-
Bai et al. [2023]

Bai Y, Wang Z, Xiao J, Wei C, Wang H, Yuille AL, et al.

Masked Autoencoders Enable Efficient Knowledge Distillers.

In: Proceedings of CVPR; 2023. p. 24256–24265.

-
Fu et al. [2023]

Fu TJ, Li L, Gan Z, Lin K, Wang WY, Wang L, et al.

An empirical study of end-to-end video-language transformers with
masked visual modeling.

In: Proceedings of CVPR; 2023. p. 22898–22909.

-
Huang et al. [2023]

Huang PY, Sharma V, Xu H, Ryali C, Li Y, Li SW, et al.

MAViL: Masked Audio-Video Learners.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 20371–20393.

-
Assran et al. [2022]

Assran M, Caron M, Misra I, Bojanowski P, Bordes F, Vincent P, et al.

Masked Siamese networks for label-efficient learning.

In: Proceedings of ECCV; 2022. p. 456–473.

-
Yang et al. [2023]

Yang S, Ge Y, Yi K, Li D, Shan Y, Qie X, et al.

RILS: Masked Visual Reconstruction in Language Semantic Space.

In: Proceedings of CVPR; 2023. p. 23304–23314.

-
Zhao et al. [2023]

Zhao X, Hayashi Y, Oda M, Kitasaka T, Mori K.

Masked Frequency Consistency for Domain-Adaptive Semantic
Segmentation of Laparoscopic Images.

In: Proceedings of MICCAI; 2023. p. 663–673.

-
Liang et al. [2022]

Liang H, Fan H, Fan Z, Wang Y, Chen T, Cheng Y, et al.

Point cloud domain adaptation via masked local 3D structure
prediction.

In: Proceedings of ECCV; 2022. p. 156–172.

-
Ren et al. [2023]

Ren B, Liu Y, Song Y, Bi W, Cucchiara R, Sebe N, et al.

Masked Jigsaw Puzzle: A Versatile Position Embedding for Vision
Transformers.

In: Proceedings of CVPR; 2023. p. 20382–20391.

-
Xiang et al. [2023]

Xiang J, Tian K, Zhang J.

MIMT: Masked Image Modeling Transformer for Video Compression.

In: Proceedings of ICLR; 2023. .

-
Radford et al. [2021]

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, et al.

Learning Transferable Visual Models From Natural Language
Supervision.

In: Proceedings of ICML; 2021. p. 8748–8763.

-
Mirza et al. [2023]

Mirza MJ, Shin I, Lin W, Schriebl A, Sun K, Choe J, et al.

MATE: Masked Autoencoders are Online 3D Test-Time Learners.

In: Proceedings of ICCV; 2023. p. 16709–16718.

-
Weers et al. [2023]

Weers F, Shankar V, Katharopoulos A, Yang Y, Gunter T.

Masked autoencoding does not help natural language supervision at
scale.

In: Proceedings of CVPR; 2023. p. 23432–23444.

-
Wu et al. [2023]

Wu Z, Lai Z, Sun X, Lin S.

Extreme Masking for Learning Instance and Distributed Visual
Representations.

Transactions on Machine Learning Research. 2023;.

-
Liu et al. [2022]

Liu H, Cai M, Lee YJ.

Masked Discrimination for Self-Supervised Learning on Point Clouds.

In: Proceedings of ECCV; 2022. p. 657–675.

-
Li et al. [2023]

Li J, Savarese S, Hoi SC.

Masked unsupervised self-training for label-free image
classification.

In: Proceedings of ICLR; 2023. .

-
Hernandez et al. [2024]

Hernandez J, Villegas R, Ordonez V.

ViC-MAE: Self-supervised Representation Learning from Images and
Video with Contrastive Masked Autoencoders.

In: Proceedings of ECCV; 2024. p. 444–463.

-
Lin and Jabri [2023]

Lin T, Jabri A.

MIMEx: Intrinsic Rewards from Masked Input Modeling.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 35592–35605.

-
Fang et al. [2023]

Fang Y, Yang S, Wang S, Ge Y, Shan Y, Wang X.

Unleashing vanilla vision transformer with masked image modeling for
object detection.

In: Proceedings of ICCV; 2023. p. 6244–6253.

-
Chen et al. [2023]

Chen Z, Qing J, Xiang T, Yue WL, Zhou JH.

Seeing beyond the brain: Conditional diffusion model with sparse
masked modeling for vision decoding.

In: Proceedings of CVPR; 2023. p. 22710–22720.

-
Chen et al. [2023]

Chen Z, Agarwal D, Aggarwal K, Safta W, Balan MM, Brown K.

Masked Image Modeling Advances 3D Medical Image Analysis.

In: Proceedings of WACV; 2023. p. 1970–1980.

-
Xiao et al. [2023]

Xiao J, Bai Y, Yuille A, Zhou Z.

Delving into masked autoencoders for multi-label thorax disease
classification.

In: Proceedings of WACV; 2023. p. 3588–3600.

-
Bozorgtabar
et al. [2023]

Bozorgtabar B, Mahapatra D, Thiran JP.

AMAE: Adaptation of Pre-Trained Masked Autoencoder for
Dual-Distribution Anomaly Detection in Chest X-Rays.

In: Proceedings of MICCAI; 2023. p. 195–205.

-
Yu et al. [2023]

Yu L, Cheng Y, Sohn K, Lezama J, Zhang H, Chang H, et al.

MAGVIT: Masked Generative Video Transformer.

In: Proceedings of CVPR; 2023. p. 10459–10469.

-
Feichtenhofer
et al. [2022]

Feichtenhofer C, Fan H, Li Y, He K.

Masked Autoencoders As Spatiotemporal Learners.

In: Proceedings of NeurIPS; 2022. p. 35946–35958.

-
Ristea et al. [2024]

Ristea NC, Croitoru FA, Ionescu RT, Popescu M, Khan FS, Shah M.

Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly
Detectors.

In: Proceedings of CVPR; 2024. p. 15984–15995.

-
Liu et al. [2023]

Liu J, Wang T, Liu B, Zhang Q, Liu Y, Li H.

GeoMIM: Towards Better 3D Knowledge Transfer via Masked Image
Modeling for Multi-view 3D Understanding.

In: Proceedings of ICCV; 2023. p. 17793–17803.

-
Yu et al. [2022]

Yu X, Tang L, Rao Y, Huang T, Zhou J, Lu J.

Point-BERT: Pre-training 3D Point Cloud Transformers with Masked
Point Modeling.

In: Proceedings of CVPR; 2022. p. 19313–19322.

-
Shen et al. [2023]

Shen Z, Sheng X, Fan H, Wang L, Guo Y, Liu Q, et al.

Masked Spatio-Temporal Structure Prediction for Self-Supervised
Learning on Point Cloud Videos.

In: Proceedings of ICCV; 2023. p. 16580–16589.

-
Gao et al. [2023]

Gao S, Zhou P, Cheng MM, Yan S.

Masked Diffusion Transformer is a Strong Image Synthesizer.

In: Proceedings of ICCV; 2023. p. 23107–23116.

-
Yan et al. [2022]

Yan Z, Li X, Wang K, Zhang Z, Li J, Yang J.

Multi-modal masked pre-training for monocular panoramic depth
completion.

In: Proceedings of ECCV; 2022. p. 378–395.

-
Tang et al. [2020]

Tang J, Tian FP, Feng W, Li J, Tan P.

Learning guided convolutional network for depth completion.

IEEE Transactions on Image Processing. 2020;30:1116–1129.

-
Xie et al. [2023]

Xie R, Pang K, Bader GD, Wang B.

MAESTER: Masked Autoencoder Guided Segmentation at Pixel Resolution
for Accurate, Self-Supervised Subcellular Structure Recognition.

In: Proceedings of CVPR; 2023. p. 3292–3301.

-
Kraus et al. [2024]

Kraus O, Kenyon-Dean K, Saberian S, Fallah M, McLean P, Leung J, et al.

Masked Autoencoders for Microscopy are Scalable Learners of Cellular
Biology.

In: Proceedings of CVPR; 2024. p. 11757–11768.

-
Zhou et al. [2023]

Zhou HY, Lian C, Wang L, Yu Y.

Advancing Radiograph Representation Learning with Masked Record
Modeling.

In: Proceedings of ICLR; 2023. .

-
Jiang et al. [2023]

Jiang Z, Chen Y, Liu M, Chen D, Dai X, Yuan L, et al.

Layer Grafted Pre-training: Bridging Contrastive Learning And Masked
Image Modeling For Label-Efficient Representations.

In: Proceedings of ICLR; 2023. .

-
Pham et al. [2024]

Pham T, Zhang K, Yoo C.

Cross-view Masked Diffusion Transformers for Person Image Synthesis.

In: Proceedings of ICML; 2024. .

-
Kong et al. [2023]

Kong L, Ma MQ, Chen G, Xing EP, Chi Y, Morency LP, et al.

Understanding Masked Autoencoders via Hierarchical Latent Variable
Models.

In: Proceedings of CVPR; 2023. p. 7918–7928.

-
Pan et al. [2023]

Pan J, Zhou P, Yan S.

Towards Understanding Why Mask Reconstruction Pretraining Helps in
Downstream Tasks.

In: Proceedings of ICLR; 2023. .

-
Xie et al. [2023]

Xie Z, Geng Z, Hu J, Zhang Z, Hu H, Cao Y.

Revealing the dark secrets of masked image modeling.

In: Proceedings of CVPR; 2023. p. 14475–14485.

-
Liu et al. [2022]

Liu B, Hsu DJ, Ravikumar P, Risteski A.

Masked prediction: A parameter identifiability view.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 21241–21254.

-
Li et al. [2023]

Li J, Chen P, He Z, Yu S, Liu S, Jia J.

Rethinking Out-of-distribution (OOD) Detection: Masked Image
Modeling is All You Need.

In: Proceedings of CVPR; 2023. p. 11578–11589.

-
Moreno-Muñoz
et al. [2023]

Moreno-Muñoz P, Garcia Recasens P, Hauberg S.

On masked pre-training and the marginal likelihood.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 79781–79791.

-
Huang et al. [2023]

Huang Q, Dong X, Chen D, Chen Y, Yuan L, Hua G, et al.

Improving Adversarial Robustness of Masked Autoencoders via
Test-time Frequency-domain Prompting.

In: Proceedings of ICCV; 2023. p. 1600–1610.

-
Xie et al. [2023]

Xie Z, Zhang Z, Cao Y, Lin Y, Wei Y, Dai Q, et al.

On Data Scaling in Masked Image Modeling.

In: Proceedings of CVPR; 2023. p. 10365–10374.

-
Gao et al. [2022]

Gao P, Ma T, Li H, Lin Z, Dai J, Qiao Y.

MCMAE: Masked Convolution Meets Masked Autoencoders.

In: Proceedings of NeurIPS; 2022. p. 35632–35644.

-
Liu et al. [2023]

Liu Z, Chen K, Han J, Hong L, Xu H, Li Z, et al.

Task-Customized Masked Autoencoder via Mixture of
Cluster-Conditional Experts.

In: Proceedings of ICLR; 2023. .

-
Zhou et al. [2023]

Zhou A, Li Y, Qin Z, Liu J, Pan J, Zhang R, et al.

SparseMAE: Sparse Training Meets Masked Autoencoders.

In: Proceedings of ICCV; 2023. p. 16176–16186.

-
Mo and Morgado [2024]

Mo S, Morgado P.

Unveiling the Power of Audio-Visual Early Fusion Transformers with
Dense Interactions through Masked Modeling.

In: Proceedings of CVPR; 2024. p. 27186–27196.

-
Tian et al. [2023]

Tian K, Jiang Y, Diao Q, Lin C, Wang L, Yuan Z.

Designing BERT for Convolutional Networks: Sparse and Hierarchical
Masked Modeling.

In: Proceedings of ICLR; 2023. .

-
Woo et al. [2023]

Woo S, Debnath S, Hu R, Chen X, Liu Z, Kweon IS, et al.

ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders.

In: Proceedings of CVPR; 2023. p. 16133–16142.

-
Riquelme et al. [2021]

Riquelme C, Puigcerver J, Mustafa B, Neumann M, Jenatton R, Susano Pinto A,
et al.

Scaling vision with sparse mixture of experts.

In: Proceedings of NeurIPS. vol. 34; 2021. p. 8583–8595.

-
Han et al. [2023]

Han Q, Cai Y, Zhang X.

RevColV2: Exploring disentangled representations in masked image
modeling.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 29273–29291.

-
Cai et al. [2023]

Cai Y, Zhou Y, Han Q, Sun J, Kong X, Li J, et al.

Reversible Column Networks.

In: Proceedings of ICLR; 2023. .

-
Huang et al. [2023]

Huang W, Peng Z, Dong L, Wei F, Jiao J, Ye Q.

Generic-to-specific distillation of masked autoencoders.

In: Proceedings of CVPR; 2023. p. 15996–16005.

-
Wu et al. [2023]

Wu K, Zheng Y, Shi J, Xie F, Jiang Z.

Position-Aware Masked Autoencoder for Histopathology WSI
Representation Learning.

In: Proceedings of MICCAI; 2023. p. 714–724.

-
Zhu et al. [2022]

Zhu J, Xia Y, Wu L, Deng J, Zhou W, Qin T, et al.

Masked contrastive representation learning for reinforcement
learning.

IEEE Transactions on Pattern Analysis and Machine Intelligence.
2022;45(3):3421–3433.

-
Dong et al. [2023]

Dong X, Bao J, Zheng Y, Zhang T, Chen D, Yang H, et al.

MaskCLIP: Masked Self-Distillation Advances Contrastive
Language-Image Pretraining.

In: Proceedings of CVPR; 2023. p. 10995–11005.

-
Kong and Zhang [2023]

Kong X, Zhang X.

Understanding masked image modeling via learning occlusion invariant
feature.

In: Proceedings of CVPR; 2023. p. 6241–6251.

-
Chen et al. [2023]

Chen C, Zhong A, Wu D, Luo J, Li Q.

Contrastive Masked Image-Text Modeling for Medical Visual
Representation Learning.

In: Proceedings of MICCAI; 2023. p. 493–503.

-
Wang et al. [2023]

Wang H, Tang Y, Wang Y, Guo J, Deng ZH, Han K.

Masked image modeling with local multi-scale reconstruction.

In: Proceedings of CVPR; 2023. p. 2122–2131.

-
Ramesh et al. [2021]

Ramesh A, Pavlov M, Goh G, Gray S, Voss C, Radford A, et al.

Zero-Shot Text-to-Image Generation.

In: Proceedings of ICML; 2021. p. 8821–8831.

-
van den Oord et al. [2017]

van den Oord A, Vinyals O, Kavukcuoglu K.

Neural discrete representation learning.

In: Proceedings of NeurIPS; 2017. p. 6309–6318.

-
Jang et al. [2023]

Jang H, Tack J, Choi D, Jeong J, Shin J.

Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked
Auto-Encoder.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 73879–73897.

-
Zheng et al. [2022]

Zheng Y, Li J, Shi J, Xie F, Jiang Z.

Kernel Attention Transformer (KAT) for Histopathology Whole Slide
Image Classification.

In: Proceedings of MICCAI; 2022. p. 283–292.

-
Li et al. [2022]

Li X, Ge Y, Yi K, Hu Z, Shan Y, Duan LY.

mc-BEiT: Multi-choice Discretization for Image BERT Pre-training.

In: Proceedings of ECCV; 2022. p. 231–246.

-
Fei et al. [2023]

Fei Z, Fan M, Zhu L, Huang J, Wei X, Wei X.

Masked auto-encoders meet generative adversarial networks and beyond.

In: Proceedings of CVPR; 2023. p. 24449–24459.

-
Wang et al. [2023]

Wang Y, Li Z, Mei J, Wei Z, Liu L, Wang C, et al.

SwinMM: Masked Multi-view with Swin Transformers for 3D Medical
Image Segmentation.

In: Proceedings of MICCAI; 2023. p. 486–496.

-
Liang et al. [2024]

Liang F, Li Y, Marculescu D.

SupMAE: Supervised Masked Autoencoders Are Efficient Vision
Learners.

In: Proceedings of EIW; 2024. .

-
Gong et al. [2023]

Gong Y, Rouditchenko A, Liu AH, Harwath D, Karlinsky L, Kuehne H, et al.

Contrastive audio-visual masked autoencoder.

In: Proceedings of ICLR; 2023. .

-
Fuller et al. [2023]

Fuller A, Millard K, Green J.

CROMA: Remote Sensing Representations with Contrastive Radar-Optical
Masked Autoencoders.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 5506–5538.

-
Pei et al. [2024]

Pei G, Chen T, Jiang X, Liu H, Sun Z, Yao Y.

VideoMAC: Video Masked Autoencoders Meet ConvNets.

In: Proceedings of CVPR; 2024. p. 22733–22743.

-
Zhao et al. [2023]

Zhao H, Wang D, Lu H.

Representation learning for visual object tracking by masked
appearance transfer.

In: Proceedings of CVPR; 2023. p. 18696–18705.

-
Zhang et al. [2023]

Zhang R, Wang L, Qiao Y, Gao P, Li H.

Learning 3D representations from 2D pre-trained models via
image-to-point masked autoencoders.

In: Proceedings of CVPR; 2023. p. 21769–21780.

-
Lin et al. [2023]

Lin H, Han G, Ma J, Huang S, Lin X, Chang SF.

Supervised masked knowledge distillation for few-shot transformers.

In: Proceedings of CVPR; 2023. p. 19649–19659.

-
Zhao et al. [2023]

Zhao Z, Wei S, Chen Q, Li D, Yang Y, Peng Y, et al.

Masked retraining teacher-student framework for domain adaptive
object detection.

In: Proceedings of ICCV; 2023. p. 19039–19049.

-
Zhang et al. [2023]

Zhang B, Wang Z, Ling Y, Guan Y, Zhang S, Li W.

Mx2M: masked cross-modality modeling in domain adaptation for 3D
semantic segmentation.

In: Proceedings of AAAI; 2023. p. 3401–3409.

-
Walsh et al. [2023]

Walsh R, Osman I, Shehata MS.

Masked Embedding Modeling With Rapid Domain Adjustment for Few-Shot
Image Classification.

IEEE Transactions on Image Processing. 2023;32:4907–4920.

-
Song et al. [2023]

Song Z, Luo R, Yu J, Chen YPP, Yang W.

Compact transformer tracker with correlative masked modeling.

In: Proceedings of AAAI; 2023. p. 2321–2329.

-
Guo et al. [2023]

Guo Z, Zhang R, Qiu L, Li X, Heng PA.

Joint-MAE: 2D-3D joint masked autoencoders for 3D point cloud
pre-training.

In: Proceedings of IJCAI; 2023. p. 791–799.

-
Wei et al. [2023]

Wei C, Mangalam K, Huang PY, Li Y, Fan H, Xu H, et al.

Diffusion models as masked autoencoders.

In: Proceedings of ICCV; 2023. p. 16284–16294.

-
Lu et al. [2023]

Lu M, Wang T, Xia Y.

Multi-modal Pathological Pre-training via Masked Autoencoders for
Breast Cancer Diagnosis.

In: Proceedings of MICCAI; 2023. p. 457–466.

-
Li et al. [2023]

Li K, Wang Y, Li Y, Wang Y, He Y, Wang L, et al.

Unmasked Teacher: Towards Training-Efficient Video Foundation
Models.

In: Proceedings of ICCV; 2023. p. 19948–19960.

-
Chen et al. [2023]

Chen A, Zhang K, Zhang R, Wang Z, Lu Y, Guo Y, et al.

PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D
Object Detection.

In: Proceedings of CVPR; 2023. p. 5291–5301.

-
Bachmann et al. [2022]

Bachmann R, Mizrahi D, Atanov A, Zamir A.

MultiMAE: Multi-modal Multi-task Masked Autoencoders.

In: Proceedings of ECCV; 2022. p. 348–367.

-
Seo et al. [2023]

Seo Y, Kim J, James S, Lee K, Shin J, Abbeel P.

Multi-view masked world models for visual robotic manipulation.

In: Proceedings of ICML; 2023. p. 30613–30632.

-
Huang et al. [2022]

Huang L, You S, Zheng M, Wang F, Qian C, Yamasaki T.

Green hierarchical vision transformer for masked image modeling.

In: Proceedings of NeurIPS. vol. 35; 2022. p. 19997–20010.

-
Zou et al. [2024]

Zou J, Huang T, Yang G, Guo Z, Luo T, Feng CM, et al.

UniM2AE: Multi-modal Masked Autoencoders with Unified 3D
Representation for 3D Perception in Autonomous Driving.

In: Proceedings of ECCV; 2024. p. 296–313.

-
Reed et al. [2023]

Reed CJ, Gupta R, Li S, Brockman S, Funk C, Clipp B, et al.

Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale
Geospatial Representation Learning.

In: Proceedings of ICCV; 2023. p. 4088–4099.

-
Yuan et al. [2023]

Yuan J, Zhang X, Zhou H, Wang J, Qiu Z, Shao Z, et al.

HAP: Structure-Aware Masked Image Modeling for Human-Centric
Perception.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 50597–50616.

-
Wang et al. [2023]

Wang Y, Wang J, Chen B, Zeng Z, Xia ST.

Contrastive masked autoencoders for self-supervised video hashing.

In: Proceedings of AAAI; 2023. p. 2733–2741.

-
Wu et al. [2023]

Wu Q, Ye H, Gu Y, Zhang H, Wang L, He D.

Denoising Masked AutoEncoders Help Robust Classification.

In: Proceedings of ICLR; 2023. .

-
Huang et al. [2023]

Huang C, Goh H, Gu J, Susskind J.

MAST: Masked Augmentation Subspace Training for Generalizable
Self-Supervised Priors.

In: Proceedings of ICLR; 2023. .

-
Cai et al. [2023]

Cai Z, Ghosh S, Stefanov K, Dhall A, Cai J, Rezatofighi H, et al.

MARLIN: Masked Autoencoder for facial video Representation
LearnINg.

In: Proceedings of CVPR; 2023. p. 1493–1504.

-
Fu et al. [2023]

Fu TJ, Yu L, Zhang N, Fu CY, Su JC, Wang WY, et al.

Tell me what happened: Unifying text-guided video completion via
multimodal masked video generation.

In: Proceedings of CVPR; 2023. p. 10681–10692.

-
Basu et al. [2024]

Basu S, Gupta M, Madan C, Gupta P, Arora C.

FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with
Focused Masked Autoencoders.

In: Proceedings of CVPR; 2024. p. 11715–11725.

-
Zhu and Liu [2023]

Zhu S, Liu X.

PMatch: Paired Masked Image Modeling for Dense Geometric Matching.

In: Proceedings of CVPR; 2023. p. 21909–21918.

-
Liu et al. [2023]

Liu S, Huynh CP, Chen C, Arap M, Hamid R.

LEMaRT: Label-Efficient Masked Region Transform for Image
Harmonization.

In: Proceedings of CVPR; 2023. p. 18290–18299.

-
Yan et al. [2023]

Yan Q, Zhang S, Chen W, Tang H, Zhu Y, Sun J, et al.

SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware
Masked Autoencoders.

In: Proceedings of CVPR; 2023. p. 5775–5784.

-
Pan et al. [2023]

Pan J, Shit S, Turgut Ö, Huang W, Li HB, Stolt-Ansó N, et al.

Global k-Space Interpolation for Dynamic MRI Reconstruction Using
Masked Image Modeling.

In: Proceedings of MICCAI; 2023. p. 228–238.

-
Tang et al. [2023]

Tang W, Huang S, Zhang X, Zhou F, Zhang Y, Liu B.

Multiple instance learning framework with masked hard instance mining
for whole slide image classification.

In: Proceedings of ICCV; 2023. p. 4078–4087.

-
Zhai et al. [2023]

Zhai JT, Liu X, Bagdanov AD, Li K, Cheng MM.

Masked Autoencoders are Efficient Class Incremental Learners.

In: Proceedings of ICCV; 2023. p. 19047–19056.

-
Cai et al. [2022]

Cai Z, Lin L, He H, Tang X.

Uni4Eye: Unified 2D and 3D Self-supervised Pre-training via Masked
Image Modeling Transformer for Ophthalmic Image Classification.

In: Proceedings of MICCAI; 2022. p. 88–98.

-
Kang et al. [2023]

Kang Q, Gao J, Li K, Lao Q.

Deblurring masked autoencoder is better recipe for ultrasound image
recognition.

In: Proceedings of MICCAI; 2023. p. 352–362.

-
Zhang et al. [2024]

Zhang X, Wu Y, Angelini E, Li A, Guo J, Rasmussen JM, et al.

MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous
Medical Image Segmentation Based on 3D Masked Autoencoding and
Pseudo-Labeling.

In: Proceedings of CVPR; 2024. p. 5851–5862.

-
Kim et al. [2023]

Kim S, Jo D, Lee D, Kim J.

MAGVLT: Masked Generative Vision-and-Language Transformer.

In: Proceedings of CVPR; 2023. p. 23338–23348.

-
Wang et al. [2023]

Wang R, Chen D, Wu Z, Chen Y, Dai X, Liu M, et al.

Masked Video Distillation: Rethinking Masked Feature Modeling for
Self-Supervised Video Representation Learning.

In: Proceedings of CVPR; 2023. p. 6312–6322.

-
Jiang et al. [2023]

Jiang L, Yang Z, Shi S, Golyanik V, Dai D, Schiele B.

Self-supervised Pre-training with Masked Shape Prediction for 3D
Scene Understanding.

In: Proceedings of CVPR; 2023. p. 1168–1178.

-
Huang et al. [2023]

Huang G, Fu H, Bors AG.

Masked Image Residual Learning for Scaling Deeper Vision
Transformers.

In: Proceedings of NeurIPS. vol. 36; 2023. p. 57570–57582.

-
Dong et al. [2022]

Dong X, Bao J, Zhang T, Chen D, Zhang W, Yuan L, et al.

Bootstrapped Masked Autoencoders for Vision BERT Pretraining.

In: Proceedings of ECCV; 2022. p. 247–264.

-
Lao et al. [2023]

Lao S, Song G, Liu B, Liu Y, Yang Y.

Masked Autoencoders Are Stronger Knowledge Distillers.

In: Proceedings of ICCV; 2023. p. 6384–6393.

-
Jiang et al. [2022]

Jiang J, Tyagi N, Tringale K, Crane C, Veeraraghavan H.

Self-supervised 3D anatomy segmentation using self-distilled masked
image transformer (SMIT).

In: Proceedings of MICCAI; 2022. p. 556–566.

-
Yang et al. [2023]

Yang H, Li X, Tang S, Zhu F, Wang Y, Chen M, et al.

Cycle-consistent masked autoencoder for unsupervised domain
generalization.

In: Proceedings of ICLR; 2023. .

-
Fu et al. [2021]

Fu TJ, Li L, Gan Z, Lin K, Wang WY, Wang L, et al.

VIOLET: End-to-End Video-Language Transformers with Masked
Visual-token Modeling.

arXiv preprint arXiv:211112681. 2021;.

-
Bashkirova
et al. [2023]

Bashkirova D, Lezama J, Sohn K, Saenko K, Essa I.

MaskSketch: Unpaired Structure-guided Masked Image Generation.

In: Proceedings of CVPR; 2023. p. 1879–1889.

-
Yu et al. [2023]

Yu Y, Li Y, Zhang C, Zhang X, Guo Z, Qin X, et al.

StrucTexTv2: Masked Visual-Textual Prediction for Document Image
Pre-training.

In: Proceedings of ICLR; 2023. .

-
Gupta et al. [2023]

Gupta A, Tian S, Zhang Y, Wu J, Martín-Martín R, Fei-Fei L.

MaskViT: Masked Visual Pre-Training for Video Prediction.

In: Proceedings of ICLR; 2023. .

-
Bandara et al. [2023]

Bandara WGC, Patel N, Gholami A, Nikkhah M, Agrawal M, Patel VM.

AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning with
Masked Autoencoders.

In: Proceedings of CVPR; 2023. p. 14507–14517.

-
Wei et al. [2024]

Wei Y, Gupta A, Morgado P.

Towards Latent Masked Image Modeling for Self-Supervised Visual
Representation Learning.

In: Proceedings of ECCV; 2024. p. 1–17.

-
Lin et al. [2017]

Lin TY, Dollár P, Girshick R, He K, Hariharan B, Belongie S.

Feature Pyramid Networks for Object Detection.

In: Proceedings of CVPR; 2017. p. 2117–2125.

-
Yu et al. [2022]

Yu Y, Zhang D, Ji Z.

Masked Feature Generation Network for Few-Shot Learning.

In: Proceedings of IJCAI; 2022. p. 3695–3701.

-
Yao et al. [2023]

Yao X, Zhang C, Li R, Sun J, Liu Z.

One-for-All: Proposal Masked Cross-Class Anomaly Detection.

In: Proceedings of AAAI; 2023. p. 4792–4800.

-
Kwon et al. [2023]

Kwon G, Cai Z, Ravichandran A, Bas E, Bhotika R, Soatto S.

Masked vision and language modeling for multi-modal representation
learning.

In: Proceedings of ICLR; 2023. .

-
Lezama et al. [2022]

Lezama J, Chang H, Jiang L, Essa I.

Improved masked image generation with token-critic.

In: Proceedings of ECCV; 2022. p. 70–86.

-
Georgescu et al. [2023]

Georgescu MI, Fonseca E, Ionescu RT, Lucic M, Schmid C, Arnab A.

Audiovisual masked autoencoders.

In: Proceedings of ICCV; 2023. p. 16144–16154.

-
Lin et al. [2014]

Lin TY, Maire M, Belongie S, Hays J, Perona P, Ramanan D, et al.

Microsoft COCO: Common objects in context.

In: Proceedings of ECCV; 2014. p. 740–755.

-
Karras et al. [2019]

Karras T, Laine S, Aila T.

A style-based generator architecture for generative adversarial
networks.

In: Proceedings of CVPR; 2019. p. 4401–4410.

-
Schuhmann et al. [2021]

Schuhmann C, Vencu R, Beaumont R, Kaczmarczyk R, Mullis C, Katta A, et al.

LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text
Pairs.

In: Proceedings of DCAI; 2021. .

-
Krizhevsky [2009]

Krizhevsky A.

Learning multiple layers of features from tiny images.

Tech Report: University of Toronto; 2009.

-
Soomro et al. [2012]

Soomro K, Zamir A, Shah M.

UCF101: A Dataset of 101 Human Actions Classes From Videos in The
Wild.

arXiv preprint arXiv:12120402. 2012;.

-
Kay et al. [2017]

Kay W, Carreira J, Simonyan K, Zhang B, Hillier C, Vijayanarasimhan S, et al.

The Kinetics Human Action Video Dataset.

arXiv preprint arXiv:170506950. 2017;.

-
Chang et al. [2015]

Chang AX, Funkhouser T, Guibas L, Hanrahan P, Huang Q, Li Z, et al.

ShapeNet: An Information-Rich 3D Model Repository.

arXiv preprint arXiv:151203012. 2015;.

-
Sharma et al. [2018]

Sharma P, Ding N, Goodman S, Soricut R.

Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset
For Automatic Image Captioning.

In: Proceedings of ACL; 2018. p. 2556–2565.

-
Krishna et al. [2017]

Krishna R, Zhu Y, Groth O, Johnson J, Hata K, Kravitz J, et al.

Visual Genome: Connecting Language and Vision Using Crowdsourced
Dense Image Annotations.

International Journal of Computer Vision. 2017;123:32–73.

-
Ridnik et al. [2021]

Ridnik T, Ben-Baruch E, Noy A, Zelnik L.

ImageNet-21K Pretraining for the Masses.

In: Proceedings of NeurIPS; 2021. .

-
Bossard et al. [2014]

Bossard L, Guillaumin M, Van Gool L.

Food-101 – Mining Discriminative Components with Random Forests.

In: Proceedings of ECCV; 2014. p. 446–461.

-
Zhou et al. [2017]

Zhou B, Zhao H, Puig X, Barriuso SFA, Torralba A.

Scene Parsing through ADE20K Dataset.

In: Proceedings of CVPR; 2017. p. 5122–5130.

-
Changpinyo
et al. [2021]

Changpinyo S, Sharma P, Ding N, Soricut R.

Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To
Recognize Long-Tail Visual Concepts.

In: Proceedings of CVPR; 2021. p. 3557–3567.

-
Ordonez et al. [2011]

Ordonez V, Kulkarni G, Berg TL.

Im2Text: Describing Images Using 1 Million Captioned Photographs.

In: Proceedings of NeurIPS; 2011. p. 1143–1151.

-
Goyal et al. [2017]

Goyal R, Kahou SE, Michalski V, Materzynska J, Westphal S, Kim H, et al.

The “Something Something” Video Database for Learning and
Evaluating Visual Common Sense.

In: Proceedings of ICCV; 2017. p. 5843–5851.

-
Carreira et al. [2018]

Carreira J, Noland E, Banki-Horvath A, Hillier C, Zisserman A.

A Short Note about Kinetics-600.

arXiv preprint arXiv:180801340. 2018;.

-
Carreira et al. [2022]

Carreira J, Noland E, Hillier C, Zisserman A.

A Short Note on the Kinetics-700 Human Action Dataset.

arXiv preprint arXiv:190706987. 2022;.

-
Li et al. [2023]

Li K, Wang Y, He Y, Li Y, Wang Y, Wang L, et al.

UniFormerV2: Unlocking the Potential of Image ViTs for Video
Understanding.

In: Proceedings of ICCV; 2023. p. 1632–1643.

-
Bain et al. [2021]

Bain M, Nagrani A, Varol G, Zisserman A.

Frozen in Time: A Joint Video and Image Encoder for End-to-End
Retrieval.

In: Proceedings of ICCV; 2021. p. 1708–1718.

-
Li et al. [2020]

Li A, Thotakuri M, Ross DA, Carreira J, Vostrikov A, Zisserman A.

The AVA-Kinetics Localized Human Actions Video Dataset.

arXiv preprint arXiv:200500214. 2020;.

-
Dai et al. [2017]

Dai A, Chang AX, Savva M, Halber M, Funkhouser T, Nießner M.

ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes.

In: Proceedings of CVPR; 2017. p. 2432–2443.

-
Caesar et al. [2020]

Caesar H, Bankiti V, Lang AH, Vora S, Liong VE, Xu Q, et al.

nuScenes: A multimodal dataset for autonomous driving.

In: Proceedings of CVPR; 2020. p. 11618–11628.

-
Wu et al. [2015]

Wu Z, Song S, Khosla A, F Yu LZ, Tang X, Xiao J.

3D ShapeNets: A deep representation for volumetric shapes.

In: Proceedings of CVPR; 2015. p. 1912–1920.

-
Uy et al. [2019]

Uy MA, Pham QH, Hua BS, Nguyen DT, Yeung SK.

Revisiting Point Cloud Classification: A New Benchmark Dataset and
Classification Model on Real-World Data.

In: Proceedings of ICCV; 2019. p. 1588–1597.

-
Yi et al. [2016]

Yi L, Kim VG, Ceylan D, Shen IC, Yan M, Su H, et al.

A scalable active framework for region annotation in 3D shape
collections.

ACM Transactions on Graphics. 2016;35.

-
Spitzer et al. [2018]

Spitzer H, Kiwitz K, Amunts K, Harmeling S, Dickscheid T.

Improving Cytoarchitectonic Segmentation of Human Brain Areas with
Self-Supervised Siamese Networks.

In: Proceedings of MICCAI; 2018. p. 663–671.

-
Pelka et al. [2018]

Pelka O, Koitka S, Rückert J, Nensa F, Friedrich CM.

Radiology Objects in COntext (ROCO): A Multimodal Image Dataset.

In: Proceedings of CVII; 2018. p. 180–189.

-
Subramanian
et al. [2020]

Subramanian S, Wang LL, Bogin B, Mehta S, van Zuylen M, Parasa S, et al.

MedICaT: A Dataset of Medical Images, Captions, and Textual
References.

In: Proceedings of EMNLP; 2020. p. 2112–2120.

-
Harmon et al. [2020]

Harmon S, Sanford T, Xu S, Turkbey E, Roth H, Xu Z, et al.

Artificial intelligence for the detection of COVID-19 pneumonia on
chest CT using multinational datasets.

Nature Communications. 2020;11:4080.

-
Simpson et al. [2019]

Simpson AL, Antonelli M, Bakas S, Bilello M, Farahani K, van Ginneken B, et al.

A large annotated medical image dataset for the development and
evaluation of segmentation algorithms.

arXiv preprint arXiv:190209063. 2019;.

-
Landman et al. [2015]

Landman B, Xu Z, Iglesias JE, Styner M, Langerak TR, Klein A.

Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and
Challenge.

Synapse. 2015;.

-
Kaggle [2019]

Kaggle.

APTOS 2019 Blindness Detection.

Kaggle. 2019;.

-
Panchal et al. [2023]

Panchal S, Naik A, Kokare M, Pachade S, Naigaonkar R, Phadnis P, et al.

Retinal Fundus Multi-Disease Image Dataset (RFMiD) 2.0: A Dataset of
Frequently and Rarely Identified Diseases.

Data. 2023;8.

-
Lau et al. [2018]

Lau J, Gayen S, Ben Abacha A, Demner-Fushman D.

A dataset of clinically generated visual questions and answers about
radiology images.

Scientific Data. 2018;5:180251.

-
Ben Abacha et al. [2019]

Ben Abacha A, Hasan SA, Datla VV, Liu J, Demner-Fushman D, Müller H.

VQA-Med: Overview of the Medical Visual Question Answering Task at
ImageCLEF 2019.

In: Proceedings of CLEF. vol. 2380; 2019. .

-
Lin et al. [2025]

Lin J, Wu CE, Li H, Zhang J, Hu YH, Morgado P.

From Prototypes to General Distributions: An Efficient Curriculum
for Masked Image Modeling.

In: Proceedings of CVPR; 2025. p. 20028–20038.

-
Liu and Yi [2025]

Liu Y, Yi L.

MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone’s Potential
with Masked Autoregressive Pretraining.

In: Proceedings of CVPR; 2025. p. 9676–9685.

-
Thoker et al. [2025]

Thoker FM, Jiang L, Zhao C, Ghanem B.

SMILE: Infusing Spatial and Motion Semantics in Masked Video
Learning.

In: Proceedings of CVPR; 2025. p. 8438–8449.

Generated on Thu Jul 10 16:12:15 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)