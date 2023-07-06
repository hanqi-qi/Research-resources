
### Honesty AI defined by Anthropic

#### Question 1: how to express uncertainty/confidence

> Solution 1: Trained with confidence annotations.

Collect human annotated confidence and correctness dataset, and train a calibrator/classifier on it. (Simply using the match or Bert model can predict the human annotation, which makes it possible to train a calibrator to predict the scores) Training Models to Predict Whether They Can Answer Questions Correctly

* Training data: (Question, IK/IDK)
* Add a value head to predict the probability of I know. 

Related Papers:

* Paper1: Language Models (Mostly) Know What They Know
* Paper2: Teaching models to express their uncertainty in words

> solution 2:  Incorporate the generated logits into language expression

[Reducing Conversational Agents’ Overconfidence Through Linguistic Calibration](link)
TL;DR: Align the model’s expressed confidence with its actual correctness/confidence.

Key Point: How to derive the model confidence?
In the classification task, we have the probability of different labels, but:
In the generation task, we don’t have a probability for an output sentence (actually, we have, we can p(w1)*p(w2)...p(wL), probably it is not good as a confidence score)
We don’t provide a probability for IDK, the model is forced to give an answer.

#### Question 2. Language Models (Mostly) Know What They Know [link]

PLMs is well-calibrated in classification tasks.
PLMs can do self-evaluation: judge the open-ended prediction, i.e., the generated results by calculating P(True) (correctness)

PLM can answer P(IK) without reference. (uncertainty) about its own output
From Calibration to Knowing What You Know
Answers are from real-world.
Different question/prompt formats lead to different results
Replace an option with “None of the Above”. (Worse)


#### Question 3: LMs know **where** they induce this prediction.

> solution 1: decompose the reasoning process

[1] Chain of thoughts. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.  
* TL;DR Only 8 manually designed COT for few-shot learning achieve the SOTA  

[2] Self-consistency  
* Follow-up work, rather than using the single/beam-search result, they sample several outputs and use a majority vote for the final results. (self-consistency)

> Solution 2:  
Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning

LEARNING WHERE AND WHEN TO REASON IN NEURO-SYMBOLIC INFERENCE

> Solution 3: Retrieval-based methods

[Rethinking with Retrieval: Faithful Large Language Model Inference](https://arxiv.org/abs/2301.00303)
* Incorporate external knowledge to answer questions. The final prediction is the most likely reasoning path in all the selected knowledge paths.

### Take-Aways:

PLMs can judge the fact about the external world better in a True/False form.

PLMs can predict their own uncertainty after supervised learning with human annotation. 
Human annotated different levels of uncertainty
Use the hard label to approximate the soft label of P(True)
RL is the best, but most complex to implement.

### Future work:
Design a method to better calculate the uncertainty, using the existing method to find a metric, which can mimic people’s evaluation results (High correlation).  
No training on human-annotated, but only prompt-based.  
Adversarial training/attack, contrastive learning.  
Uncertainty degree is different in different tasks, across different domains.  
Locate/identify where the model is uncertain, to find the error.  
What if we Correct their errors in CoT.  

### Reading materials (survey)

Language Models (Mostly) Know What They Know. OpenAI(Anthropic)  
Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond  
Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models
