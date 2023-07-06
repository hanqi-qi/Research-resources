

## Q1: The details of the Transformer structure:
Word2vec, contextualized vector.
Word2Vec: CBOW&Skip-gram, relies on conditional probabilities predicting some words using some of their surrounding words in corpora.
Skip-gram: The surrounding words are conditioned on the centred word

CBOW: The surrounding words are conditioned on the centred word

Issues in optimization: (the summation of all the items in Vocab)

Negative sampling and hierarchical softmax to accelerate training
Position embedding.


BPE
BPE与Wordpiece都是首先初始化一个小词表，再根据一定准则将不同的子词合并。词表由小变大. BPE与Wordpiece的最大区别在于，如何选择两个子词进行合并：BPE选择频数
mask策略和改进
Bert模型中激活函数GELU
部分激活函数的公式及求导

## GPT variants and BERT.
GPT使用的是Transformer模型，而BERT使用的是双向Transformer模型。
GPT的预训练数据来源是大量的网络文本数据，而BERT的预训练数据来源是两个大型语料库，包括Wikipedia和BooksCorpus。
GPT预训练过程中，采用了语言模型的方法，即通过预测下一个词来学习语言模型，而BERT预训练过程中采用了双向预测的方法，即通过预测句子中丢失的词来学习语言模型。
GPT微调时，需要指定输入输出的语言模型任务，而BERT微调时，可以应用在多种任务上，例如文本分类、命名实体识别等。
Bert——Masking Input：(non-autoregressive) Only Encoder
从某方面而言，更像是利用深度学习对文本进行特征表示的过程。
Fine-tuning.
GPT-Predict next token: （autoregressive) Only Decoder
Prompting.

## GPT家族：

#### GPT1:
无监督预训练+有监督微调

#### GPT2（15亿）: 
统一各个任务的训练模式 Language Models are Unsupervised Multitask Learners
zero-shot

#### GPT3（1750亿）: 
Language Models are Few-Shot Learners (特指ICL, 参数不发生变化）
Innovation in Model Structure: sparse attention
Training corpus: （570G）

#### InstructGPT
GPT-3 中的 few-shot 对于同一个下游任务，通常采用固定的任务描述方式，而且需要人去探索哪一种任务表述方式更好.
SFT (supervised fine tuning)
InstructGPT 在 SFT 中标注的数据，正是为了消除这种模型预测与用户表达习惯之间的 gap。在标注过程中，他们从 GPT-3 的用户真实请求中采样大量下游任务的描述，然后让标注人员对任务描述进行续写，从而得到该问题的高质量回答。这里用户真实请求又被称为某个任务的指令，即 InstructGPT 的核心思想“基于人类反馈的指令微调”。

RLHF [BLOG]
Reward model, input: text, output: score/reward, r
Training objective: $$r-\lambda*KL(LM0|LM_{current})$$

### Other Encoder-Decoder Model Structures

BART&T5: Encoder-Decoder 

Large Language Model without Instruction Fine-tune

T5: encoder-decoder, GPT: decoder; 

从 GPT-3 开始才是真正意义的大模型
GPT-3 将模型参数规模扩大至 175B， 是 GPT-2 的 100 倍


