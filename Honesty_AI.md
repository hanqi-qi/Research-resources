Source Materials
Rethinking with Retrieval: Faithful Large Language Model Inference.
Incorporate external knowledge to answer questions. The final prediction is the most likely reasoning path in all the selected knowledge paths.

Self-Knowledge: Reasoning Process
CoT: Chain-of-thought prompting
Self-consistency improves chain of thought reasoning in language models.

Rethinking with Retrieval: Faithful Large Language Model Inference. 


Self-critical: evaluate their own prediction
 Showing Many T = 1 Samples Improves Self-Evaluation
Predict Test performance:


Language Models (Mostly) Know What They Know. OpenAI(Anthropic)




Evaluating Verifiability in Generative Search Engines
We’re Afraid Language Models Aren’t Modeling Ambiguity
Language models can explain neurons in language models

Honesty AI defined by Anthropic

Question 1: LMs can give more complete/up-to-date/correct answers by retrieving from external knowledge.
Rethinking with Retrieval: Faithful Large Language Model Inference. Dec2022
Retrieve the knowledge from KBs, that are entailed with the decomposed ChainOfThoughts.
Question 2: LMs know when they don’t know.
Prediction with uncertainty or even IDK.
1.Reducing Conversational Agents’ Overconfidence Through Linguistic Calibration[link]
TL;DR: Align the model’s expressed confidence with its actual correctness/confidence.

Key Point: How to derive the model confidence?
In the classification task, we have the probability of different labels, but:
In the generation task, we don’t have a probability for an output sentence (actually, we have, we can p(w1)*p(w2)...p(wL), probably it is not good as a confidence score)
We don’t provide a probability for IDK, the model is forced to give an answer.

How to define the confidence score, by training or just use the model intermediate output?
See Paper in “Teaching models to express their uncertainty in words[slides]”, which is similar to P(IK) described by OpenAI Antropic, a supervised fine-tuning, simpler to implement than RL. 


Solution: Collect human annotated confidence and correctness dataset, and train a calibrator/classifier on it. (Simply using the match or Bert model can predict the human annotation, which makes it possible to train a calibrator to predict the scores)

But things are not easy like that, in ChatGPT, etc, human evaluation standard is hard to predict so they employ RL to learn the reward from Human Feedback:
Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.
Learning to summarize from human feedback.

2. Language Models (Mostly) Know What They Know [link]

PLMs is well-calibrated in classification tasks.
PLMs can do self-evaluation: judge the open-ended prediction, i.e., the generated results by calculating P(True) (correctness)

PLM can answer P(IK) without reference. (uncertainty) about its own output
From Calibration to Knowing What You Know
Answers are from real-world.
Different question/prompt formats lead to different results
Replace an option with “None of the Above”. (Worse)

Simply ask models if a given answer is true or false. (Better)

Ask the AI: Is your proposed answer True or False P(True)
Without fine-tuning, just prompt.
Answers are sampled from its own decoding results.
Only One sample for evaluation, zero-shot, poorly calibrated.
Many samples improve the self-evaluation, especially in short-form answer tasks

Training Models to Predict Whether They Can Answer Questions Correctly
Training data: (Question, IK/IDK)
Add a value head to predict the probability of I know. 

Question 3: LMs know where they induce this prediction.
Chain of thoughts. 
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
TL;DR Only 8 manually designed COT for few-shot learning achieve the SOTA 

Follow-up work, rather than using the single/beam-search result, they sample several outputs and use a majority vote for the final results. (self-consistency)


Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning

LEARNING WHERE AND WHEN TO REASON IN NEURO-SYMBOLIC INFERENCE
Take-Aways:

PLMs can judge the fact about the external world better in a True/False form.

PLMs can predict their own uncertainty after supervised learning with human annotation. 
Human annotated different levels of uncertainty
Use the hard label to approximate the soft label of P(True)
RL is the best, but most complex to implement.

Future work:
Design a method to better calculate the uncertainty, using the existing method to find a metric, which can mimic people’s evaluation results (High correlation). 
No training on human-annotated, but only prompt-based.
Adversarial training/attack, contrastive learning. See Hehe ()
Uncertainty degree is different in different tasks, across different domains.
Locate/identify where the model is uncertain, to find the error.
What if we Correct their errors in CoT,


General Survey

Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond

Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models
