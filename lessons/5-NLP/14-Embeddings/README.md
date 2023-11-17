# Embeddings

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/114)

<!-- When training classifiers based on BoW or TF/IDF, we operated on high-dimensional bag-of-words vectors with length `vocab_size`, and we were explicitly converting from low-dimensional positional representation vectors into sparse one-hot representation. This one-hot representation, however, is not memory-efficient. In addition, each word is treated independently from each other, i.e. one-hot encoded vectors do not express any semantic similarity between words. -->
当训练基于 BoW 或 TF/IDF 的分类器时，我们对长度为 的高维词袋向量进行操作`vocab_size`，并且我们显式地从低维位置表示向量转换为稀疏独热表示。然而，这种独热表示的内存效率不高。此外，每个单词都被相互独立地处理，即，one-hot 编码向量不表达单词之间的任何语义相似性。

<!-- The idea of **embedding** is to represent words by lower-dimensional dense vectors, which somehow reflect the semantic meaning of a word. We will later discuss how to build meaningful word embeddings, but for now let's just think of embeddings as a way to lower dimensionality of a word vector. -->
**embedding**的思想是用低维稠密向量来表示单词，这在某种程度上反映了单词的语义。稍后我们将讨论如何构建有意义的词嵌入，但现在我们只将嵌入视为降低词向量维度的一种方法。

<!-- So, the embedding layer would take a word as an input, and produce an output vector of specified `embedding_size`. In a sense, it is very similar to a `Linear` layer, but instead of taking a one-hot encoded vector, it will be able to take a word number as an input, allowing us to avoid creating large one-hot-encoded vectors. -->
因此，嵌入层会将一个单词作为输入，并生成指定的输出向量`embedding_size`。从某种意义上说，它与`Linear`层非常相似，但它不是采用 one-hot 编码向量，而是能够采用单词编号作为输入，从而使我们能够避免创建大型 one-hot 编码向量。

<!-- By using an embedding layer as a first layer in our classifier network, we can switch from a bag-of-words to **embedding bag** model, where we first convert each word in our text into corresponding embedding, and then compute some aggregate function over all those embeddings, such as `sum`, `average` or `max`.   -->
通过使用嵌入层作为分类器网络中的第一层，我们可以从词袋模型切换到**embedding bag**模型，其中我们首先将文本中的每个单词转换为相应的嵌入，然后计算所有单词的聚合函数这些嵌入，例如`sum`, `average` or `max`。


![Image showing an embedding classifier for five sequence words.](images/embedding-classifier-example.png)

> Image by the author

## ✍️ Exercises: Embeddings

Continue your learning in the following notebooks:
* [Embeddings with PyTorch](EmbeddingsPyTorch.ipynb)
* [Embeddings TensorFlow](EmbeddingsTF.ipynb)

## Semantic Embeddings: Word2Vec

<!-- While the embedding layer learned to map words to vector representation, however, this representation did not necessarily have much semantical meaning. It would be nice to learn a vector representation such that similar words or synonyms correspond to vectors that are close to each other in terms of some vector distance (eg. Euclidean distance). -->
然而，虽然嵌入层学会了将单词映射到向量表示，但这种表示并不一定具有太多语义意义。学习一种向量表示，使得相似的单词或同义词对应于在某些向量距离（例如欧几里德距离）方面彼此接近的向量，这将是很好的。

<!-- To do that, we need to pre-train our embedding model on a large collection of text in a specific way. One way to train semantic embeddings is called [Word2Vec](https://en.wikipedia.org/wiki/Word2vec). It is based on two main architectures that are used to produce a distributed representation of words: -->
为此，我们需要以特定方式在大量文本上预训练嵌入模型。训练语义嵌入的一种方法称为[Word2Vec](https://en.wikipedia.org/wiki/Word2vec)。它基于两种主要架构，用于生成单词的分布式表示：

 <!-- - **Continuous bag-of-words** (CBoW) — in this architecture, we train the model to predict a word from surrounding context. Given the ngram $(W_{-2},W_{-1},W_0,W_1,W_2)$, the goal of the model is to predict $W_0$ from $(W_{-2},W_{-1},W_1,W_2)$.
 - **Continuous skip-gram** is opposite to CBoW. The model uses surrounding window of context words to predict the current word. -->

 - **Continuous bag-of-words** (CBoW)——在这个架构中，我们训练模型从周围的上下文中预测单词。给定 ngram $(W_{-2},W_{-1},W_0,W_1,W_2)$，模型的目标是从 $W_0$ from $(W_{-2},W_{-1},W_1,W_2)$。
 - **Continuous skip-gram**与 CBoW 相反。该模型使用上下文单词的周围窗口来预测当前单词。

<!-- CBoW is faster, while skip-gram is slower, but does a better job of representing infrequent words. -->
CBoW 更快，而skip-gram 更慢，但在表示不常见单词方面做得更好。

![Image showing both CBoW and Skip-Gram algorithms to convert words to vectors.](./images/example-algorithms-for-converting-words-to-vectors.png)

> Image from [this paper](https://arxiv.org/pdf/1301.3781.pdf)

<!-- Word2Vec pre-trained embeddings (as well as other similar models, such as GloVe) can also be used in place of embedding layer in neural networks. However, we need to deal with vocabularies, because the vocabulary used to pre-train Word2Vec/GloVe is likely to differ from the vocabulary in our text corpus. Have a look into the above Notebooks to see how this problem can be resolved. -->
Word2Vec 预训练嵌入（以及其他类似模型，例如 GloVe）也可以用来代替神经网络中的嵌入层。然而，我们需要处理词汇，因为用于预训练 Word2Vec/GloVe 的词汇可能与我们文本语料库中的词汇不同。查看上面的笔记本，看看如何解决这个问题。

## Contextual Embeddings

<!-- One key limitation of traditional pretrained embedding representations such as Word2Vec is the problem of word sense disambiguation. While pretrained embeddings can capture some of the meaning of words in context, every possible meaning of a word is encoded into the same embedding. This can cause problems in downstream models, since many words such as the word 'play' have different meanings depending on the context they are used in. -->

传统预训练嵌入表示（例如 Word2Vec）的一个关键限制是词义消歧问题。虽然预训练的嵌入可以捕获上下文中单词的一些含义，但单词的每个可能的含义都被编码到相同的嵌入中。这可能会导致下游模型出现问题，因为许多单词（例如“play”）根据其使用的上下文而具有不同的含义。

<!-- For example word 'play' in those two different sentences have quite different meaning: -->
例如，这两个不同句子中的“play”一词具有完全不同的含义：

- I went to a **play** at the theatre.
- John wants to **play** with his friends.

<!-- The pretrained embeddings above represent both of these meanings of the word 'play' in the same embedding. To overcome this limitation, we need to build embeddings based on the **language model**, which is trained on a large corpus of text, and *knows* how words can be put together in different contexts. Discussing contextual embeddings is out of scope for this tutorial, but we will come back to them when talking about language models later in the course. -->
上面的预训练嵌入在同一嵌入中代表了“play”一词的这两种含义。为了克服这个限制，我们需要基于**语言模型**构建嵌入，该模型在大型文本语料库上进行训练，并且知道如何在不同的上下文中将单词组合在一起。讨论上下文嵌入超出了本教程的范围，但我们将在课程稍后讨论语言模型时再次讨论它们。

## Conclusion

<!-- In this lesson, you discovered how to build and use embedding layers in TensorFlow and Pytorch to better reflect the semantic meanings of words. -->
在本课程中，您了解了如何在 TensorFlow 和 Pytorch 中构建和使用嵌入层以更好地反映单词的语义。

## 🚀 Challenge

<!-- Word2Vec has been used for some interesting applications, including generating song lyrics and poetry. Take a look at [this article](https://www.politetype.com/blog/word2vec-color-poems) which walks through how the author used Word2Vec to generate poetry. Watch [this video by Dan Shiffmann](https://www.youtube.com/watch?v=LSS_bos_TPI&ab_channel=TheCodingTrain) as well to discover a different explanation of this technique. Then try to apply these techniques to your own text corpus, perhaps sourced from Kaggle. -->

Word2Vec 已用于一些有趣的应用，包括生成歌词和诗歌。看看[这篇文章](https://www.politetype.com/blog/word2vec-color-poems)，它介绍了作者如何使用 Word2Vec 生成诗歌。观看Dan Shiffmann 的[视频](https://www.youtube.com/watch?v=LSS_bos_TPI&ab_channel=TheCodingTrain)，了解该技术的不同解释。然后尝试将这些技术应用到您自己的文本语料库（可能来自 Kaggle）.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/214)

## Review & Self Study

Read through this paper on Word2Vec: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

## [Assignment: Notebooks](assignment.md)
