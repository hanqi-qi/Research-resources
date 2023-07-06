 
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


Keywords

Token Uniformity
Uncertainty–OOD 
model calibration
Robustness
Contrastive Learning


Dan Hendrycks https://people.eecs.berkeley.edu/~hendrycks/

Paper List

Using Pre-Training Can Improve Model Robustness and Uncertainty
Create adversarial samples and do experiments on them
DIRECTPROBE: Studying Representations without Classifiers
Uncertainty-Aware Reliable Text Classification
add regularization terms for in-distribution and OOD data, specifically, decrease the uncertainty of in-distribution data and increase for the OOD data.
Calibration of pre-trained transformers
and its citations


LEARNING BETTER STRUCTURED REPRESENTATIONS USING LOW-RANK ADAPTIVE LABEL SMOOTHING 



Except for accuracy, we also care about the metrics, including
model robustness to label corruption
class imbalance
 adversarial attacks
 uncertainty estimates for out-of-distribution detection
 calibration
…













Contrastive Learning in CV (0509 reads)
Understanding Self-Supervised Learning Dynamics without Contrastive Pairs
UNDERSTANDING DIMENSIONAL COLLAPSE IN CONTRASTIVE SELF-SUPERVISED LEARNING
A Simple Framework for Contrastive Learning of Visual Representations
Decorrelated Batch Normalization
On Feature Decorrelation in Self-Supervised Learning


DECOUPLED CONTRASTIVE LEARNING. 
Debiased Contrastive Learning
adjust the weight between the number of positive and negative samples


Contrastive Learning in NLP (0510 todo)
COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining
Self-Guided Contrastive Learning for BERT Sentence Representations
CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding
create positive and negative samples

Contrastive Self-supervised Sequential Recommendation with Robust Augmentation
SimCSE
Why uniformity can be learned automatically? do we need positive samples?
Debiased Contrastive Learning






Contrastive Learning and Generative model
Contrastive Learning Inverts the Data Generating Process
The process of contrastive learning is similar to the generative model, e.g., VAE, they are encouraged to get the similar z. which inspires us to get better CL loss.
We can connect token unifotmity->contrastive learning->VAE->topic model->generation diversity. 





The idea of Contrastive Learning


Balance the number of positive and negative samples?
DECOUPLED CONTRASTIVE LEARNING
Debiased Contrastive Learning
CONTRASTIVE LEARNING WITH HARD NEGATIVE SAMPLES
0608之前: 能否证明出 那个调节因子其实是跟pronpensity 相关的东西，或者增加causal inference的东西到cl中，也就是如何保证negative是 t=0, 而不是confounder。在无监督的场景下。或者是着重于正例。
0608: 对每一个正例-负例 加一个特定的权重，这个方法的特例其实是对 某些假负例乘以一个权重0。最简单的方法，这个权重应该是它到分类边界的距离，距离越远，权重越小。
Domain Generalization using Causal Matching. ICML21, cite50+
CV： 同一个object的不同domain分类。
NLP: 同一个表达（token/phrase）在不同语境/context中的情感不同（sentiment analysis）。对于sentence similarity, 同一个主体/主干加不同修饰词含义一样 (Sts dataset)。
contrastive learning的下游任务: 一般都是一种而已，要么是STS要么是sentiment analysis. 因为需要构造的是不同的正/负例子。其实也就是说 overall好的representation不太有？
能不能统一一下，把需要的各个属性投射到不同维度之后，只需要对对应维度聚类/正负，就可以直接针对各个任务做contrastive learning了。
那么，怎么找到BERT output feature, 不同维度代表的含义呢？
为什么gender bias可以直接找到bias对应的维度？need to collect representative words and then detect the most predominant singular value direction.
Bert encoded sentence embedding? 不同句子【cls】最显著的方向一致么？经过linear classifier（e.g. 情感）之后，又是为什么能够（假设能够生成出positive/negative两个显著方向。）
首先对bert encoded sent svd分解，再只取前n个给分类器看效果。如果效果跟用全部的相差不大，就用这n个继续下面的实验, 如果效果相差很大，或者能找到某些情况效果相差很大（different domain, directClassifier）。对于同一个类别positive或negative，bert encoded sentence rep, 如果特征方向基本一致，而不同类别则特征方向不一致。则说明linear layer帮助它保留了这个特征。也就是分类之后最显著的两个方向，跟bert sent rep两个方向有对应关系（只取topN, 积极是正北方向长，消极是正东）。如果，同类别的encoded rep相似性甚至低于不同类别（只取topN, 这四个方向的分布基本一致），分类器又是如何知道的哪两个方向是情感呢？
是多个维度线性组合的结果DirectProb， 使用SVM。
能找到是哪些组成了情感的？哪些组成了语义的？
How to generate/define positive/negative samples?
Large-Margin Contrastive Learning with Distance Polarization Regularizer, ICML21
Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations (rejected by ICLR22)
It takes two to tango: mixup for deep metric learning, ICLR22





Add contrastive learning objectives in specific tasks.
	

PTLM Probes:
A Closer Look at How Fine-tuning Changes BERT
It seems that the token uniformity does not mean less discriminative.


Isotropy in PTLM (June13, papers citing[1])
[1] Isotropy in the Contextual Embedding Space: Clusters and Manifolds. ICLR21 (****)
How Does Fine-tuning Affect the Geometry of Embedding Space: A Case Study on Isotropy. EMNLP21(***)
A closer look at how fine-tuning changes BERT
Fine-tuning Pre-trained Language Models for Few-shot Intent Detection: Supervised Pre-training and Isotropization NAACL2022 (*)
Measuring the Mixing of Contextual Information in the Transformer (****)
Out-of-manifold Regularization in Contextual Embedding Space for Text Classification(**)
All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality（**）
Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders (** Similar to SimCSE)
On Isotropy Calibration of Transformers (* just a comparison of existing isotropy-addressing issues)
 Isotropy calibration on transformers and find that they do not provide consistent improvements across models and tasks
Putting Words in BERT’s Mouth: Navigating Contextualized Vector Spaces with Pseudowords(*** TO read)
Latent Topology Induction for Understanding Contextualized Representations, Under Neurips22 (*****)
Totally unsupervised method of using crf-vae as supervised probe will hurt the original topology in PTLM.
Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings. ACL2022 (*****Addressing token uniformity via gradient)


Contrastive and spurious correlation/causal inference/robust representation:
[1] Correct-n-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations
worst-group performance is negatively correlated with alignment loss, thus introducing contrastive loss in training. Step1: clustering the hidden states before classifier as group labels; Step2: introduce the contrastive loss by “pulling the samples in the same group together”,
[2] Distributionally robust neural networks for group shifts: on the importance of regularization for worst-case generalization [Prior work for Paper 1.1]
[3] Domain Generalization using Causal Matching
(1) Using the causal representation (a subset of the original rep) for contrastive learning; (2) iteratively update the positive-negative group assignment if there is no 






Hypothesis
Token uniformity with training time, longer→severer
Token representations are related to domains(datasets) or tasks? （Domain adaptation）
In the same domain, the words tend to have very similar contexts. so, maybe we can adversarial change the context to make it more diverse
So, we can alternatively 
Token uniformity varies from different layers: 
Token Uniformity can somehow address the debias problem, model calibration, making two terms different
regularization in token uniformity paper, equation (6) L
regularization in debiased paper, equation (3) J_{a}
REPRESENTATION DEGENERATION PROBLEM IN TRAINING NATURAL LANGUAGE GENERATION MODELS [link]

	



It’s much easier to compare any two-word vectors in an embedding matrix E, but how can we do it without the E, like a sentence. Similar to contrastive learning, which considers the positive pairs as well.

	5）relevant to word orders in a sentence?



对现有考虑feature本身的东西 换成考虑梯度的。
选择某些样本进行优化









Future work about UAI22 (June10th): 
Naïve methods to do SVD, so slow
Have a little impact on model weight. à just greatly changing the last layer singular value distribution
may be good as keep the most parts of the PTLMs unchanged, but does not improve the PTLM itself.
punish greatly changed gradients
Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings

Connections to Contrastive Learning (Seems very promising)
The selection of negative pairs
The interactions between positive and negative pairs (closer to the decision boundary):
The weight between the two parts or
The weight of each 1-N pair
Causal Inference
The propensity score is used to balance training/test distribution, and domain generalization; maybe our alpha could be an interpretable factor, to balance the Pretrained Rep Space& DownstreamTasks
We only need to focus on the causal part of the representation, rather than the whole rep.
Token Uniformity and Discriminative
The token uniformity is sensitive to distance metrics, cosine similarity suffers the most. 
The Euclidean distance/Manhattan distance is worse than cosine similarity, as they are proportional to cosine similarity. 
The tokens occupy a narrow cone, but they are separate clusterings, not continuous.

Thus, global isotropy is not desirable. Analysis of predominant direction:
A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space. [It removes the first predominant directions in different clusters, and provides an explanation]
structual&syntactic： 去除前后？在period,comma附近，计算属于一个structual&syntactic的句子比例，这个比例下降，说明这个方向含有stuctual&syntactic的信息。那么其它的信息是不是都会下降呢？这个方向可能是杂糅了很多信息。那就是谁下降的多，主要信息是谁。
Avaliable dataset: Unsupervised Distillation of Syntactic Information from Contextualized Word Representations. [The dataset consists of groups in which sentences are structurally and syntactically similar but have no semantic similarity]
verb sense v.s. tense. 去除前后，衡量不同verb之间的距离。原来距离很大，现在距离变小了，说明这个信息主要是分布在这个方向上【看的是差值吧】。


Hierarchical VAE/NoParametric VAE.



Motivated by the token distribution of PLTM, which are separated clusters occupying a narrow space. And the main idea in causal inference is to split the data into subgroups, which are dominated by a specific confounder.

Existing works in debiased methods either focus on a specific attribute, e.g., gender or address the bias in a global view, without considering the difference in subgroups. For example, the bias in the pictures of “cat in the grass” is the green background, while the bias in the picture of “cats in tiger-like stripe” is the cat’s skin pattern. Therefore, we argue that the bias/dominant directions in subgroups are various and debased methods should be performed within subgroups.

The basic model can be this
“Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure”, 
the main algorithm is as follows:
	
The improvements/contributions can be:
we need to do clustering first, before calculating the dominant samples/attributes. And one possible solution can be “VQ-VAE”, each code in the codebook is the centroid of each cluster.
Different from step5 in the Algorithm, which selects the biased samples based on their overall latent variables (The multiplier of each z_{i}). However, we need to reweight the specific latent dimension, rather than the whole sample.
Following step2, we are going to adjust the sampling process in creating training batches, we are targeting to adjusting the latent spaces(modify the latent space, or rather, similar to our SoftDecay, adjust the singular value, but in each subgroup.)



我有一个想法，就是从为什么BERT/PLMs 的效果好出发。假设只搞sentence classification，用的是【CLS】. BERT-original representation，经过fine-tuning在很多数据集上都表现很好。有论文说 fine-tuning不怎么改变original representation/保持原有的cluster，只是把不同cluster的距离变得更远。那么，如果是同一个数据，但是有不同label, 假设每个label对应的这个task 理想的cluster分布是不同的。我们把original representation fix住，只训练不同的分类器去做不同的task, 是不是也会在各个tasks上得到很好的效果呢（比我用其它encoder(LSTM/TextCNN)+分类器（可训练）要好）？
（A）如果要好，是不是可以从以下以下方面探究好的原因：1. 维度大 2. original representation的cluster多/少，更利于下游classifier 合并/聚拢 cluster. 3. isotropy/特征值方向分布较均匀（集中）？在cluster内部？更易于分裂合并？
(B) 如果不好，那就加一下fine-tuning再看（按常理会更好）。然后可以对比的就是 BERTinit_Emb+(hierarchical LSTM/CNN)+fine-tuning. Fine-tuning就是这些token representation的自身可以训练。在这种情况下，就可以探究一下，是不是BERT本身的结构比LSTM/CNN更好，导致相同的token embedding也可以被学成不一样的分布。
我个人希望是第一种情况，这样就可以专注于研究bert-original representation到底有什么好的性质了。其实我现在猜测的就是这个维度很重要，因为我看有文章说在token-level的数据集上，用PCA之后只取第一个特征向量的结果都比static wordembedding好。那如果是维度的话，我们要是有能力训出一个768维的向量，是不是就也很厉害了【好吧，只有大模型才有这能力，这就是bert的厉害之处，感觉我回到了原点】。
fine-tuning会改变isotropy嘛？不过反正都是满秩的。
fine-tuning会改变最大的特征值么？我感觉是会的，因为每个任务需要的最好的特征值应该是不太相同的。如果不会的话，那应该那些方向就是基本的语言学特征。




Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning. [code]

non-degeneration 的设定下，G 来自PLMs和LSTM其它更简单的encoder好像没有区别。


是不是可以从实际情况出发，探究M究竟是如果对X聚类的。会是一个解读Machine concept的工作。并且跟一般的Encoder M进行比较。










An Explanation of In-context Learning as Implicit Bayesian Inference. [code]
选择的生成模型过于简单，不符合真实语言的情况。跟现实数据关系不大，没有挖掘自然语言的特点（除了长度），而像句法和主题等重要元素的影响没有研究。
仿真GINC数据的过程需要修改
理论证明的方向不一样了，是考虑pretraining data和 fine-tuning data 建模以后的match程度。
一个公共的latent variable给 observed variable 也就是input sequence, 然后让这个prompt distribution也尽可能共享这个M(比如就是加一些约束，让这两个分布接近。这里的M可以理解成topic 啊之类的). 这样一来，我们其实对prompt 微调了，那么那个prepend的prefix就没那么影响结果，手工调整的部分就不太需要了。
问题是要训练这个M 需要数据，如果只用few-shot的那几个数据怕是训练不好。




	先要有一个猜想/basic work否则很难下手。
探究一下fine-tuning/+prompt时，究竟怎么设计或者矫正这个distribution mismatch, 能够保证泛化性。从而解释什么样的fine-tuning/prompt更适合这个模型。
用已有公式推导出y-{xi} 变化规律，从而找到更好的M.
在PLMs 和 downstream task/prompts之间，加一个aligment M, 使得overlap更大，也就是尽量消除这个mismatch, 使得prompt的adjustion不再需要。
Synetic dataset x, generated by HMM models.
我们能否反推或者根据现有数据生成一个模型Model(fuyao 用了CRF-VAE，Xiang lisa Li 用了Posterior Control of Blackbox Generation)？不同之处，我们要用独立z和公共M.
假设我们的模型Model特别好, 也可以对x得到z(zero-shot), 也可以对x, y得到z(few-shot). 是否有一个指标，衡量这些z之间的match程度，从而推出更合适的训练方式。
问题是，需要预先有一个猜想的方向。比如说，z的什么活跃系数（就是哪些维度是有有效值的)。 z最大值index。z特征值之类的。
假设我们的转移矩阵定义的是句子 syntax, M是topic 之类的。那我们要生成仿真数据，是语法相似的和主题相似的。
Prototypical Calibration for Few-shot Learning of Language Models. 
using GMM trained on EM-annotated data as clustering tool, rather than GPT-2.
applying manually designed prompts
Prompt Distribution Learning.
SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer.


