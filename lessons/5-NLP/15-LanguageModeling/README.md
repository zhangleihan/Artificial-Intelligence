# Language Modeling

<!-- Semantic embeddings, such as Word2Vec and GloVe, are in fact a first step towards **language modeling** - creating models that somehow *understand* (or *represent*) the nature of the language. -->
语义嵌入，例如 Word2Vec 和 GloVe，实际上是**语言建模**的第一步- 创建以某种方式*理解*（或*表示*）语言本质的模型。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/115)

<!-- The main idea behind language modeling is training them on unlabeled datasets in an unsupervised manner. This is important because we have huge amounts of unlabeled text available, while the amount of labeled text would always be limited by the amount of effort we can spend on labeling. Most often, we can build language models that can **predict missing words** in the text, because it is easy to mask out a random word in text and use it as a training sample. -->
语言建模背后的主要思想是以无监督的方式在未标记的数据集上训练它们。这很重要，因为我们有大量未标记的文本可用，而标记文本的数量始终受到我们在标记上花费的精力的限制。大多数情况下，我们可以构建可以**预测文本中缺失单词**的语言模型，因为很容易屏蔽文本中的随机单词并将其用作训练样本。

## Training Embeddings

<!-- In our previous examples, we used pre-trained semantic embeddings, but it is interesting to see how those embeddings can be trained. There are several possible ideas the can be used: -->
在之前的示例中，我们使用了预先训练的语义嵌入，但了解如何训练这些嵌入很有趣。有几种可能的想法可以使用：

<!-- * **N-Gram** language modeling, when we predict a token by looking at N previous tokens (N-gram)
* **Continuous Bag-of-Words** (CBoW), when we predict the middle token $W_0$ in a token sequence $W_{-N}$, ..., $W_N$.
* **Skip-gram**, where we predict a set of neighboring tokens {$W_{-N},\dots, W_{-1}, W_1,\dots, W_N$} from the middle token $W_0$. -->

* **N-Gram** N-Gram语言建模，当我们通过查看 N 个先前的标记来预测标记时 (N-gram)
* **Continuous Bag-of-Words** (CBoW)连续词袋（CBoW），当我们预测标记序列 $W_{-N}$, ..., $W_N$ 中的中间标记 $W_0$ 时。
* **Skip-gram**，我们从中间标记 $W_0$ 预测一组相邻标记 {$W_{-N},\dots, W_{-1}, W_1,\dots, W_N$}。

![image from paper on converting words to vectors](../14-Embeddings/images/example-algorithms-for-converting-words-to-vectors.png)

> Image from [this paper](https://arxiv.org/pdf/1301.3781.pdf)

## ✍️ Example Notebooks: Training CBoW model

Continue your learning in the following notebooks:

* [Training CBoW Word2Vec with TensorFlow](CBoW-TF.ipynb)
* [Training CBoW Word2Vec with PyTorch](CBoW-PyTorch.ipynb)


## Conclusion

<!-- In the previous lesson we have seen that words embeddings work like magic! Now we know that training word embeddings is not a very complex task, and we should be able to train our own word embeddings for domain specific text if needed.  -->
在上一课中，我们已经看到单词嵌入就像魔法一样！现在我们知道训练词嵌入并不是一项非常复杂的任务，如果需要，我们应该能够针对特定领域的文本训练我们自己的词嵌入。

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/215)

## Review & Self Study

* [Official PyTorch tutorial on Language Modeling](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html).
* [Official TensorFlow tutorial on training Word2Vec model](https://www.TensorFlow.org/tutorials/text/word2vec).
* Using the **gensim** framework to train most commonly used embeddings in a few lines of code is described [in this documentation](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html).

## 🚀 [Assignment: Train Skip-Gram Model](lab/README.md)

In the lab, we challenge you to modify the code from this lesson to train skip-gram model instead of CBoW. [Read the details](lab/README.md)
