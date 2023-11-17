# Attention Mechanisms and Transformers

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/118)

<!-- One of the most important problems in the NLP domain is **machine translation**, an essential task that underlies tools such as Google Translate. In this section, we will focus on machine translation, or, more generally, on any *sequence-to-sequence* task (which is also called **sentence transduction**). -->

NLP 领域最重要的问题之一是**machine translation**机器翻译，这是 Google 翻译等工具的基础任务。在本节中，我们将重点关注机器翻译，或者更一般地说，关注任何*sequence-to-sequence*序列到序列任务（也称为**sentence transduction**句子转导）。

<!-- With RNNs, sequence-to-sequence is implemented by two recurrent networks, where one network, the **encoder**, collapses an input sequence into a hidden state, while another network, the **decoder**, unrolls this hidden state into a translated result. There are a couple of problems with this approach: -->
对于 RNN，序列到序列是由两个循环网络实现的，其中一个网络（**encoder**编码器）将输入序列折叠为隐藏状态，而另一个网络（**decoder**解码器）将该隐藏状态展开为翻译结果。这种方法有几个问题：

<!-- * The final state of the encoder network has a hard time remembering the beginning of a sentence, thus causing poor quality of the model for long sentences -->
<!-- * All words in a sequence have the same impact on the result. In reality, however, specific words in the input sequence often have more impact on sequential outputs than others. -->
* 编码器网络的最终状态很难记住句子的开头，从而导致长句子的模型质量较差
* 序列中的所有单词对结果都有相同的影响。然而，实际上，输入序列中的特定单词通常比其他单词对顺序输出产生更大的影响。

<!-- **Attention Mechanisms** provide a means of weighting the contextual impact of each input vector on each output prediction of the RNN. The way it is implemented is by creating shortcuts between intermediate states of the input RNN and the output RNN. In this manner, when generating output symbol y<sub>t</sub>, we will take into account all input hidden states h<sub>i</sub>, with different weight coefficients &alpha;<sub>t,i</sub>. -->

**Attention Mechanisms**注意力机制提供了一种加权每个输入向量对 RNN 每个输出预测的上下文影响的方法。它的实现方式是在输入 RNN 和输出 RNN 的中间状态之间创建快捷方式。以这种方式，当生成输出符号y<sub>t</sub>时，我们将考虑具有不同权重系数 &alpha;<sub>t,i</sub>, i的所有输入隐藏状态 h<sub>i</sub>。

![Image showing an encoder/decoder model with an additive attention layer](./images/encoder-decoder-attention.png)

> The encoder-decoder model with additive attention mechanism in [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf), cited from [this blog post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

<!-- The attention matrix {&alpha;<sub>i,j</sub>} would represent the degree that certain input words play in the generation of a given word in the output sequence. Below is an example of such a matrix: -->
注意力矩阵 {&alpha;<sub>i,j</sub>} 将表示某些输入单词在输出序列中给定单词的生成中发挥的程度。以下是此类矩阵的示例：

![Image showing a sample alignment found by RNNsearch-50, taken from Bahdanau - arviz.org](./images/bahdanau-fig3.png)

> Figure from [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf) (Fig.3)

<!-- Attention mechanisms are responsible for much of the current or near current state of the art in NLP. Adding attention however greatly increases the number of model parameters which led to scaling issues with RNNs. A key constraint of scaling RNNs is that the recurrent nature of the models makes it challenging to batch and parallelize training. In an RNN each element of a sequence needs to be processed in sequential order which means it cannot be easily parallelized. -->

注意力机制是 NLP 领域当前或接近当前技术水平的大部分原因。然而，增加注意力会大大增加模型参数的数量，从而导致 RNN 出现扩展问题。缩放 RNN 的一个关键限制是模型的循环性质使得批量和并行训练变得具有挑战性。在 RNN 中，序列的每个元素都需要按顺序处理，这意味着它不能轻易并行化。

![Encoder Decoder with Attention](images/EncDecAttention.gif)

> Figure from [Google's Blog](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html)

<!-- The adoption of attention mechanisms combined with this constraint led to the creation of the now State of the Art Transformer Models that we know and use today such as BERT to Open-GPT3. -->
注意力机制的采用与这一约束相结合，催生了我们今天所知并使用的最先进的 Transformer 模型，例如 BERT 到 Open-GPT3。

## Transformer models

<!-- One of the main ideas behind transformers is to avoid sequential nature of RNNs and to create a model that is parallelizable during training. This is achieved by implementing two ideas: -->
Transformer 背后的主要思想之一是避免 RNN 的顺序性质并创建一个在训练期间可并行化的模型。这是通过实施两个想法来实现的：

<!-- * positional encoding -->
<!-- * using self-attention mechanism to capture patterns instead of RNNs (or CNNs) (that is why the paper that introduces transformers is called *[Attention is all you need](https://arxiv.org/abs/1706.03762)* -->

* positional encoding位置编码
* using self-attention mechanism 使用自注意力机制来捕获模式而不是 RNN（或 CNN）（这就是为什么介绍 Transformer 的论文被称为*[Attention is all you need](https://arxiv.org/abs/1706.03762)*

### Positional Encoding/Embedding

<!-- The idea of positional encoding is the following.  -->
位置编码的思想如下。

<!-- 1. When using RNNs, the relative position of the tokens is represented by the number of steps, and thus does not need to be explicitly represented. 
2. However, once we switch to attention, we need to know the relative positions of tokens within a sequence. 
3. To get positional encoding, we augment our sequence of tokens with a sequence of token positions in the sequence (i.e., a sequence of numbers 0,1, ...).
4. We then mix the token position with a token embedding vector. To transform the position (integer) into a vector, we can use different approaches: -->

1. 当使用 RNN 时，标记的相对位置由步数表示，因此不需要显式表示。
2. 然而，一旦我们转向注意力，我们就需要知道序列中标记的相对位置。
3. 为了获得位置编码，我们用序列中的标记位置序列（即数字 0,1, ...）序列来扩充标记序列。
4. 然后，我们将标记位置与标记嵌入向量混合。要将位置（整数）转换为向量，我们可以使用不同的方法：

<!-- * Trainable embedding, similar to token embedding. This is the approach we consider here. We apply embedding layers on top of both tokens and their positions, resulting in embedding vectors of the same dimensions, which we then add together. -->
<!-- * Fixed position encoding function, as proposed in the original paper. -->

* Trainable embedding 可训练的嵌入，类似于令牌嵌入。这就是我们在这里考虑的方法。我们在两个标记及其位置之上应用嵌入层，从而产生相同维度的嵌入向量，然后将它们相加。
* Fixed position encoding function 固定位置编码函数，如原论文中提出的。

<img src="images/pos-embedding.png" width="50%"/>

> Image by the author

<!-- The result that we get with positional embedding embeds both the original token and its position within a sequence. -->
我们通过位置嵌入得到的结果嵌入了原始标记及其在序列中的位置。

### Multi-Head Self-Attention

<!-- Next, we need to capture some patterns within our sequence. To do this, transformers use a **self-attention** mechanism, which is essentially attention applied to the same sequence as the input and output. Applying self-attention allows us to take into account **context** within the sentence, and see which words are inter-related. For example, it allows us to see which words are referred to by coreferences, such as *it*, and also take the context into account: -->

接下来，我们需要捕获序列中的一些模式。为此，变压器使用**self-attention** 自注意力机制，本质上是将注意力应用于与输入和输出相同的序列。应用自注意力使我们能够考虑句子中的**context**上下文，并查看哪些单词是相互关联的。例如，它允许我们查看哪些单词是通过共指引用的，例如it，并且还考虑了上下文：

![](images/CoreferenceResolution.png)

> Image from the [Google Blog](https://research.googleblog.com/2017/08/transformer-novel-neural-network.html)

<!-- In transformers, we use **Multi-Head Attention** in order to give the network the power to capture several different types of dependencies, eg. long-term vs. short-term word relations, co-reference vs. something else, etc. -->
在 Transformer 中，我们使用多头注意力（**Multi-Head Attention**）来赋予网络捕获几种不同类型的依赖关系的能力，例如。长期与短期词关系、共指与其他事物等。

[TensorFlow Notebook](TransformersTF.ipynb) contains more detains on the implementation of transformer layers.

### Encoder-Decoder Attention

<!-- In transformers, attention is used in two places: -->
在 Transformer 中，attention 用在两个地方：

<!-- * To capture patterns within the input text using self-attention -->
<!-- * To perform sequence translation - it is the attention layer between encoder and decoder. -->
* 使用自注意力捕获输入文本中的模式
* 执行序列翻译 - 它是编码器和解码器之间的注意层。

<!-- Encoder-decoder attention is very similar to the attention mechanism used in RNNs, as described in the beginning of this section. This animated diagram explains the role of encoder-decoder attention. -->
编码器-解码器注意力与 RNN 中使用的注意力机制非常相似，如本节开头所述。该动画图解释了编码器-解码器注意力的作用。

![Animated GIF showing how the evaluations are performed in transformer models.](./images/transformer-animated-explanation.gif)

<!-- Since each input position is mapped independently to each output position, transformers can parallelize better than RNNs, which enables much larger and more expressive language models. Each attention head can be used to learn different relationships between words that improves downstream Natural Language Processing tasks. -->

由于每个输入位置都独立映射到每个输出位置，因此 Transformer 可以比 RNN 更好地并行化，从而实现更大、更具表现力的语言模型。每个注意力头可用于学习单词之间的不同关系，从而改进下游自然语言处理任务。

## BERT

<!-- **BERT** (Bidirectional Encoder Representations from Transformers) is a very large multi layer transformer network with 12 layers for *BERT-base*, and 24 for *BERT-large*. The model is first pre-trained on a large corpus of text data (WikiPedia + books) using unsupervised training (predicting masked words in a sentence). During pre-training the model absorbs significant levels of language understanding which can then be leveraged with other datasets using fine tuning. This process is called **transfer learning**. -->
**BERT**（来自 Transformers 的双向编码器表示）是一个非常大的多层 Transformer 网络，BERT-base有 12 层，BERT-large有 24 层。该模型首先使用无监督训练（预测句子中的屏蔽词）对大量文本数据（维基百科+书籍）进行预训练。在预训练期间，模型吸收了大量的语言理解能力，然后可以通过微调将其与其他数据集结合使用。这个过程称为**transfer learning**迁移学习。


![picture from http://jalammar.github.io/illustrated-bert/](images/jalammarBERT-language-modeling-masked-lm.png)

> Image [source](http://jalammar.github.io/illustrated-bert/)

## ✍️ Exercises: Transformers

Continue your learning in the following notebooks:

* [Transformers in PyTorch](TransformersPyTorch.ipynb)
* [Transformers in TensorFlow](TransformersTF.ipynb)

## Conclusion

<!-- In this lesson you learned about Transformers and Attention Mechanisms, all essential tools in the NLP toolbox. There are many variations of Transformer architectures including BERT, DistilBERT. BigBird, OpenGPT3 and more that can be fine tuned. The [HuggingFace package](https://github.com/huggingface/) provides repository for training many of these architectures with both PyTorch and TensorFlow. -->

在本课程中，您了解了 Transformers 和 Attention Mechanisms，它们是 NLP 工具箱中的所有重要工具。Transformer 架构有很多变体，包括 BERT、DistilBERT。BigBird、OpenGPT3 等可以进行微调。[HuggingFace package](https://github.com/huggingface/)提供了使用 PyTorch 和 TensorFlow 训练许多此类架构的存储库。

## 🚀 Challenge

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/218)

## Review & Self Study

* [Blog post](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/), explaining the classical [Attention is all you need](https://arxiv.org/abs/1706.03762) paper on transformers.
* [A series of blog posts](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452) on transformers, explaining the architecture in detail.

## [Assignment](assignment.md)