# Representing Text as Tensors

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/113)

## Text Classification

<!-- Throughout the first part of this section, we will focus on **text classification** task. We will use the [AG News](https://www.kaggle.com/amananandrai/ag-news-classification-dataset) Dataset, which contains news articles like the following: -->

在本节的第一部分中，我们将重点关注文本分类任务。我们将使用[AG News](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)数据集，其中包含如下新闻文章：

<!-- * Category: Sci/Tech
* Title: Ky. Company Wins Grant to Study Peptides (AP)
* Body: AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop... -->

* 类别：科学/技术
* 标题：Ky. Company 获得肽研究资助 (美联社)
* 正文：美联社 - 路易斯维尔大学化学研究员创立的一家公司获得了开发……的资助

<!-- Our goal will be to classify the news item into one of the categories based on text. -->
我们的目标是根据文本将新闻项目分类为某一类别。

## Representing text

<!-- If we want to solve Natural Language Processing (NLP) tasks with neural networks, we need some way to represent text as tensors. Computers already represent textual characters as numbers that map to fonts on your screen using encodings such as ASCII or UTF-8. -->
如果我们想用神经网络解决自然语言处理（NLP）任务，我们需要某种方法将文本表示为张量。计算机已经将文本字符表示为数字，并使用 ASCII 或 UTF-8 等编码映射到屏幕上的字体。

<img alt="Image showing diagram mapping a character to an ASCII and binary representation" src="images/ascii-character-map.png" width="50%"/>

> [Image source](https://www.seobility.net/en/wiki/ASCII)

<!-- As humans, we understand what each letter **represents**, and how all characters come together to form the words of a sentence. However, computers by themselves do not have such an understanding, and neural network has to learn the meaning during training. -->
作为人类，我们了解每个字母代表什么，以及所有字符如何组合在一起形成句子的单词。然而，计算机本身并没有这样的理解能力，神经网络必须在训练过程中学习其含义。

<!-- Therefore, we can use different approaches when representing text: -->
因此，我们在表示文本时可以使用不同的方法：

<!-- * **Character-level representation**, when we represent text by treating each character as a number. Given that we have *C* different characters in our text corpus, the word *Hello* would be represented by 5x*C* tensor. Each letter would correspond to a tensor column in one-hot encoding. -->
* **Character-level representation字符级表示**，当我们通过将每个字符视为数字来表示文本时。鉴于我们的文本语料库中有C 个不同的字符，单词Hello将由 5x*C*张量表示。每个字母都对应于 one-hot 编码中的一个张量列。
<!-- * **Word-level representation**, in which we create a **vocabulary** of all words in our text, and then represent words using one-hot encoding. This approach is somehow better, because each letter by itself does not have much meaning, and thus by using higher-level semantic concepts - words - we simplify the task for the neural network. However, given the large dictionary size, we need to deal with high-dimensional sparse tensors. -->
* **Word-level representation单词级表示**，其中我们创建文本中所有单词的**词汇表**，然后使用 one-hot 编码来表示单词。这种方法在某种程度上更好，因为每个字母本身没有多大意义，因此通过使用更高级别的语义概念（单词），我们简化了神经网络的任务。然而，考虑到字典大小很大，我们需要处理高维稀疏张量。

<!-- Regardless of the representation, we first need to convert the text into a sequence of **tokens**, one token being either a character, a word, or sometimes even part of a word. Then, we convert the token into a number, typically using **vocabulary**, and this number can be fed into a neural network using one-hot encoding. -->

无论采用哪种表示形式，我们首先需要将文本转换为一系列**tokens标记**，一个标记可以是一个字符、一个单词，有时甚至是单词的一部分。然后，我们通常使用词汇将标记转换为数字，并且可以使用 one-hot 编码将该数字输入到神经网络中。

## N-Grams

<!-- In natural language, precise meaning of words can only be determined in context. For example, meanings of *neural network* and *fishing network* are completely different. One of the ways to take this into account is to build our model on pairs of words, and considering word pairs as separate vocabulary tokens. In this way, the sentence *I like to go fishing* will be represented by the following sequence of tokens: *I like*, *like to*, *to go*, *go fishing*. The problem with this approach is that the dictionary size grows significantly, and combinations like *go fishing* and *go shopping* are presented by different tokens, which do not share any semantic similarity despite the same verb.  -->
在自然语言中，单词的精确含义只能在上下文中确定。例如，*neural network*和*fishing network*的含义完全不同。考虑到这一点的方法之一是在单词对上构建我们的模型，并将单词对视为单独的词汇标记。这样，句子*I like to go fishing*将由以下标记序列表示：*I like*, *like to*, *to go*, *go fishing*。这种方法的问题在于，字典大小显着增加，并且像*go fishing*和*go shopping*这样的组合由不同的标记表示，尽管动词相同，但它们不具有任何语义相似性。 

<!-- In some cases, we may consider using tri-grams -- combinations of three words -- as well. Thus the approach is such is often called **n-grams**. Also, it makes sense to use n-grams with character-level representation, in which case n-grams will roughly correspond to different syllabi. -->
在某些情况下，我们也可以考虑使用三元组（三个单词的组合）。因此这种方法通常被称为**n-grams**。此外，使用具有字符级表示的 n-gram 是有意义的，在这种情况下，n-gram 将大致对应于不同的教学大纲。

## Bag-of-Words and TF/IDF

<!-- When solving tasks like text classification, we need to be able to represent text by one fixed-size vector, which we will use as an input to final dense classifier. One of the simplest ways to do that is to combine all individual word representations, eg. by adding them. If we add one-hot encodings of each word, we will end up with a vector of frequencies, showing how many times each word appears inside the text. Such representation of text is called **bag of words** (BoW). -->
在解决文本分类等任务时，我们需要能够用一个固定大小的向量表示文本，我们将使用该向量作为最终密集分类器的输入。最简单的方法之一是将所有单独的单词表示组合起来，例如。通过添加它们。如果我们添加每个单词的 one-hot 编码，我们最终会得到一个频率向量，显示每个单词在文本中出现的次数。这种文本表示称为词袋**bag of words** (BoW)。

<img src="images/bow.png" width="90%"/>

> Image by the author

<!-- A BoW essentially represents which words appear in text and in which quantities, which can indeed be a good indication of what the text is about. For example, news article on politics is likely to contains words such as *president* and *country*, while scientific publication would have something like *collider*, *discovered*, etc. Thus, word frequencies can in many cases be a good indicator of text content. -->
BoW 本质上表示文本中出现了哪些单词以及出现的数量，这确实可以很好地指示文本的内容。例如，有关政治的新闻文章可能包含*president*和*country*等词语，而科学出版物可能包含*collider*、*discovered*等词语。因此，词频在许多情况下可以很好地指示文本内容。

<!-- The problem with BoW is that certain common words, such as *and*, *is*, etc. appear in most of the texts, and they have highest frequencies, masking out the words that are really important. We may lower the importance of those words by taking into account the frequency at which words occur in the whole document collection. This is the main idea behind TF/IDF approach, which is covered in more detail in the notebooks attached to this lesson. -->
BoW 的问题在于，某些常见单词，例如*and*、*is*等出现在大多数文本中，并且它们的频率最高，从而掩盖了真正重要的单词。我们可以通过考虑单词在整个文档集合中出现的频率来降低​​这些单词的重要性。这是 TF/IDF 方法背后的主要思想，本课程附带的笔记本中对此进行了更详细的介绍。

<!-- However, none of those approaches can fully take into account the **semantics** of text. We need more powerful neural networks models to do this, which we will discuss later in this section. -->
然而，这些方法都不能充分考虑文本的语义**semantics**。我们需要更强大的神经网络模型来做到这一点，我们将在本节后面讨论。

## ✍️ Exercises: Text Representation

Continue your learning in the following notebooks:

* [Text Representation with PyTorch](TextRepresentationPyTorch.ipynb)
* [Text Representation with TensorFlow](TextRepresentationTF.ipynb)

## Conclusion

<!-- So far, we have studied techniques that can add frequency weight to different words. They are, however, unable to represent meaning or order. As the famous linguist J. R. Firth said in 1935, "The complete meaning of a word is always contextual, and no study of meaning apart from context can be taken seriously." We will learn later in the course how to capture contextual information from text using language modeling. -->
到目前为止，我们已经研究了可以为不同单词添加频率权重的技术。然而，它们无法代表意义或顺序。正如著名语言学家 JR Firth 在 1935 年所说：“一个词的完整含义总是与上下文有关，任何脱离上下文的意义研究都不能被认真对待。” 我们将在课程稍后学习如何使用语言建模从文本中捕获上下文信息。

## 🚀 Challenge

<!-- Try some other exercises using bag-of-words and different data models. You might be inspired by this [competition on Kaggle](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words) -->

使用词袋和不同的数据模型尝试其他一些练习。您可能会受到Kaggle 上的[这场比赛](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words)的启发

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/213)

## Review & Self Study

Practice your skills with text embeddings and bag-of-words techniques on [Microsoft Learn](https://docs.microsoft.com/learn/modules/intro-natural-language-processing-pytorch/?WT.mc_id=academic-77998-cacaste)

## [Assignment: Notebooks](assignment.md)
