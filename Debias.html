 
Motivations:
Motivated by the token distribution of PLTM, which are separated clusters occupying a narrow space. And the main idea in causal inference is to split the data into subgroups, which are dominated by a specific confounder.

Limitations of existing work:
Existing works in debiased methods either focus on a specific attribute, e.g., gender or address the bias in a global view, without considering the difference in subgroups. For example, the bias in the picture of “cat in the grass” is the green background, while the bias in the picture of “cats in tiger-like stripe” is the cat’s skin pattern. Therefore, we argue that the bias/dominant directions in subgroups are various and debased methods should be performed within subgroups.

Baseline
The basic model can be this
“Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure”, 
the main algorithm is as follows:
	
The improvements/contributions can be:
we need to do clustering first, before calculating the dominant samples/attributes. And one possible solution can be “VQ-VAE”, each code in the codebook E is the centroid of each cluster.
Different from step5 in the Algorithm, which selects the biased samples based on their overall latent variables (The multiplier of each z_{i}). However, we need to reweight the specific latent dimension, rather than the whole sample.
Following step2, we are going to adjust the sampling process in creating training batches, we are targeting to adjusting the latent spaces(modify the latent space, or rather, similar to our SoftDecay, adjust the singular value, but in each subgroup.)





Model Overview:


Key Components:
How to split subgroup?
Unsupervised clustering may suffer from issues, e.g., KNN may tend to generate more even clusters.
We want to follow the idea of “stratification” in causal inference. 
How to derive latent variables hierarchically (c->E->z)?
How to reweight confounder within subgroup?

Mixture-Gaussian distribution can be directly applied to the issue?

Related Work (Causal Inference/NonParametricVAE/Hierachical VAE … ):

Section1: More General VAEs:
Generating Diverse High-Fidelity Images with VQ-VAE-2
(**) lead to paper Discriminator Rejection Sampling. Without details of bottom-top structure, or just un-sampling and shortcut
Deep AutoRegressive Networks
(****) The first paper I know to use NN to learn a new prior for better generation, so it is done in two stages (1) train encoder-decoder and (2) train NN from latent code (reconstruct) to data point.
Taming Transformers for High-Resolution Image Synthesis. [code] 2020 citation 500+
Based on the two-stage VAE, they replace the pixelCNN with Transformer and replace reconstruction loss with perception loss+patch-level GAN loss.
Code can be used, consisting VQ-VAE and learning of rich prior.
Nonparametric Variational Auto-Encoders for Hierarchical Representation Learning
Multi-manifold clustering: A graph-constrained deep nonparametric method
Learning Hierarchical Priors in VAEs
(**)Continous hierarchical VAE (integral to aggregate multiple single-prior ), using Lagrangian constrained optimizer to rewrite the ELBO.
Taming VAEs (****) A work prior to Learning Hierarchical Priors in VAEs, while using the weighted sum to derive a rich prior.






	Section2: Specific Structure
Learning to Induce Causal Structure Arxiv,2022
Multi-Facet Clustering Variational Autoencoders, Neurips21.
(**)mainly talk about how to optimize the VAE, while the generative model (single-facet) has been proposed in Ruishu’s Blog
Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders (GMVAE) [Ruishublog]
(*****) Seems can be applied to our settings.
CausalVAE: Structured Causal Disentanglement in Variational Autoencoder
(****)  Good work with code of how to implement (latent variable + mask) . The mask is actually an adjacency/transiency matrix guiding z_{i} -->z_{j}.
While, my mask is supposed to mask the confounder part in the whole latent variable, the mask varies from different clustering.
 Motivation: latent variable is not independent, they have causal relations (dependent).
Limitation relies on a fine-grained label u. The endogenous variable z relies on  the fine-grained label u.
Right for the Right Latent Factors: Debiasing Generative Models via Disentanglement
(Disentangle+debais) Adding regularization terms to make the attribute-related latent 	
End-to-End Balancing for Causal Continuous Treatment-Effect Estimation, 2021 submitted to ICLR [pdf]
Hard Negative Mixing for Contrastive Learning, Neurips20[pdf]
Debiased Contrastive Learning of Unsupervised Sentence Representations, ACL22
(*****) Seems to have lots of to be improved


Section3: Implementation or observations
A Closer Look at How Fine-tuning Changes BERT
	




Kun Zhang
Section A

Conditional VAE
Learning Structured Output Representation using Deep Conditional Generative Models[pdf]


V.s. CausalVAE
Cons: needs fine-grained label
Pros: model dependency between latent variables, rather than rely on independent assumption.

Our work:
Modelling causal relation between latent variables/attributes
Create a counterfactual world, data sharing balances attributes, so can benefit
Domain adaptation
Not misled by frequent patterns in training data.

Section B
Conditional contrastive learning/ Attribute-level contrastive learning

Paper list:
Conditional Contrastive Learning With Kernel, ICLR22 [pdf]
(*****)
Contrastive Learning With Hard Negative Samples, ICLR2021 [pdf]
Hard Negative Mixing for Contrastive Learning, Neurips2020 [pdf]















Section C 
Debiasing in NLP applications. 

Tutorial of Robust Learning and Inference for IE [pdf]. USC, Muhao Chen.
Counterfactual Inference for Text Classification Debiasing, ACL21 [pdf]
Summerize the bias in text classification and existing methods can be divided into  (a) Data-level resampling (create new data to make the data distribution balanced) (b) re-weight (adjusting the weight of different training samples)

Should We Rely on Entity Mentions for Relation Extraction? Debiasing Relation Extraction with Counterfactual Analysis, NAACL22 [pdf]




Section D
General Debiasing&Robust learning. In ML conference
Just Train Twice: Improving Group Robustness without Training Group Information. [pdf] ICML2021, cite50+






Section E
Keywords: Concept, Interpretability

We need to reweight/resample the concept rather than attributes/single samples rely on human understanding. That is to expand the human’s boundary to machines.
 Kim Been [Slides] [BLOG]

Paper List:
Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV), ICML18 [pdf]
The concept vector is orthogonal to the decision boundary of labeled data and randomly selected data.
Concept Bottleneck Models, ICML20
Dissect: Disentangled Simultaneous Explanations Via Concept Traversals, ICLR22
Regularization to disentangle the latent features

Questions:
How to evaluate “concept” in NLP?
Clustering the data and showing the keywords to see its coherence, etc.[wordcloud]





First, determine the application situation

Unbalanced training data is very common in practical settings, so the model tends to link a certain particular pattern to a label, aka spurious correlation between frequent data pattern and label.

Spurious pattern and reweight.

How to interpret the clusters in PLMs, can we further split/revise the cluster and make them interpretable and more powerful for downstream tasks?
Any bias even if they are already locally isotropic?
Appropriate Isotropy(negative item in CL) is helpful to all the NLP tasks? Even better than the whole CL?
Classification problem and embedding space and STS tasks.
Sentence-level classification
Fine-tuning Pre-trained Language Models for Few-shot Intent Detection: Supervised Pre-training and Isotropization. NAACL22
A closer look at how fine-tuning changes BERT. NAACL22
Clustering 
Embedding Space
How Does Fine-tuning Affect the Geometry of Embedding Space: A Case Study on Isotropy



Bias in text classification:

Measuring and Mitigating Unintended Bias in Text Classification
Issue: A specific demographical group is more likely to be labeled toxicity.
Solution: Introduce extra non-toxicity data with the selected demographical group.
Metrics: 
Limitations:
Don’t know what are the sensitive words, but they are given/can be easily detected in the corpus, as the phrase bias.
Not words but bi/tri/gram or even is a concept, rather than literature words.


Counterfactual Inference for Text Classification Debiasing, ACL21 [pdf]
Solutions: Remove the influence of System bias/using all-mask input and influence of context/using content-mask input via summarization tool selecting content.
Limitations:
Definition of content words is relatively simple/straightforward

Demographics Should Not Be the Reason of Toxicity: Mitigating Discrimination in Text Classifications with Instance Weighting.
Limitations: 
need explicit bias labels.


