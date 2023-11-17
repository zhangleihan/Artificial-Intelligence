# Named Entity Recognition

<!-- Up to now, we have mostly been concentrating on one NLP task - classification. However, there are also other NLP tasks that can be accomplished with neural networks. One of those tasks is **[Named Entity Recognition](https://wikipedia.org/wiki/Named-entity_recognition)** (NER), which deals with recognizing specific entities within text, such as places, person names, date-time intervals, chemical formulae and so on. -->
到目前为止，我们主要关注一项 NLP 任务——分类。然而，还有其他 NLP 任务可以通过神经网络来完成。其中一项任务是命名实体识别**[Named Entity Recognition](https://wikipedia.org/wiki/Named-entity_recognition)** (NER)，它处理识别文本中的特定实体，例如地点、人名、日期时间间隔、化学式等。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/119)

## Example of Using NER

<!-- Suppose you want to develop a natural language chat bot, similar to Amazon Alexa or Google Assistant. The way intelligent chat bots work is to *understand* what the user wants by doing text classification on the input sentence. The result of this classification is so-called **intent**, which determines what a chat bot should do. -->

假设您想开发一个自然语言聊天机器人，类似于 Amazon Alexa 或 Google Assistant。智能聊天机器人的工作方式是通过对输入句子进行文本分类来了解用户想要什么。这种分类的结果就是所谓的**intent**意图，它决定了聊天机器人应该做什么。

<img alt="Bot NER" src="images/bot-ner.png" width="50%"/>

> Image by the author

<!-- However, a user may provide some parameters as part of the phrase. For example, when asking for the weather, she may specify a location or date. A bot should be able to understand those entities, and fill in the parameter slots accordingly before performing the action. This is exactly where NER comes in. -->
然而，用户可以提供一些参数作为短语的一部分。例如，当询问天气时，她可能会指定地点或日期。机器人应该能够理解这些实体，并在执行操作之前相应地填充参数槽。这正是 NER 发挥作用的地方。

<!-- > ✅ Another example would be [analyzing scientific medical papers](https://soshnikov.com/science/analyzing-medical-papers-with-azure-and-text-analytics-for-health/). One of the main things we need to look for are specific medical terms, such as diseases and medical substances. While a small number of diseases can probably be extracted using substring search, more complex entities, such as chemical compounds and medication names, need a more complex approach. -->

> ✅ 另一个例子是分析科学医学论文[analyzing scientific medical papers](https://soshnikov.com/science/analyzing-medical-papers-with-azure-and-text-analytics-for-health/)。我们需要寻找的主要内容之一是特定的医学术语，例如疾病和药物。虽然可以使用子字符串搜索来提取少量疾病，但更复杂的实体（例如化合物和药物名称）需要更复杂的方法。

## NER as Token Classification

<!-- NER models are essentially **token classification models**, because for each of the input tokens we need to decide whether it belongs to an entity or not, and if it does - to which entity class. -->
NER 模型本质上是**token classification models**token 分类模型，因为对于每个输入 token，我们需要决定它是否属于一个实体，如果属于，则属于哪个实体类。

<!-- Consider the following paper title: -->
考虑以下论文标题：

**Tricuspid valve regurgitation** and **lithium carbonate** **toxicity** in a newborn infant.
新生儿**三尖瓣反流**和**碳酸锂** **毒性**。

Entities here are:

* Tricuspid valve regurgitation is a disease (`DIS`)三尖瓣关闭不全是一种疾病（DIS）
* Lithium carbonate is a chemical substance (`CHEM`)碳酸锂是一种化学物质( CHEM)
* Toxicity is also a disease (`DIS`)中毒也是一种病（DIS）

<!-- Notice that one entity can span several tokens. And, as in this case, we need to distinguish between two consecutive entities. Thus, it is common to use two classes for each entity - one specifying the first token of the entity (often the `B-` prefix is used, for **b**eginning), and another - the continuation of an entity (`I-`, for **i**nner token). We also use `O` as a class to represent all **o**ther tokens. Such token tagging is called [BIO tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) (or IOB). When tagged, our title will look like this: -->

请注意，一个实体可以跨越多个令牌。并且，在本例中，我们需要区分两个连续的实体。因此，通常为每个实体使用两个类 - 一个指定实体的第一个标记（通常`B-`使用前缀，用于开始），另一个 - 实体的延续（`I-`，表示内部标记）。我们还使用`O`一个类来表示所有其他标记。这种令牌标记称为[BIO tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)标记（或 IOB）。标记后，我们的标题将如下所示：

Token | Tag
------|-----
Tricuspid | B-DIS
valve | I-DIS
regurgitation | I-DIS
and | O
lithium | B-CHEM
carbonate | I-CHEM
toxicity | B-DIS
in | O
a | O
newborn | O
infant | O
. | O

<!-- Since we need to build a one-to-one correspondence between tokens and classes, we can train a rightmost **many-to-many** neural network model from this picture: -->

由于我们需要在标记和类别之间建立一对一的对应关系，因此我们可以从这张图中训练最右边的多对多神经网络模型：

![Image showing common recurrent neural network patterns.](../17-GenerativeNetworks/images/unreasonable-effectiveness-of-rnn.jpg)

> *Image from [this blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by [Andrej Karpathy](http://karpathy.github.io/). NER token classification models correspond to the right-most network architecture on this picture.*

## Training NER models

<!-- Since a NER model is essentially a token classification model, we can use RNNs that we are already familiar with for this task. In this case, each block of recurrent network will return the token ID. The following example notebook shows how to train LSTM for token classification. -->
由于 NER 模型本质上是一个 token 分类模型，因此我们可以使用我们已经熟悉的 RNN 来完成此任务。在这种情况下，循环网络的每个块都会返回代币 ID。以下示例笔记本展示了如何训练 LSTM 进行标记分类。

## ✍️ Example Notebooks: NER

Continue your learning in the following notebook:

* [NER with TensorFlow](NER-TF.ipynb)

## Conclusion

<!-- A NER model is a **token classification model**, which means that it can be used to perform token classification. This is a very common task in NLP, helping to recognize specific entities within text including places, names, dates, and more. -->
NER模型是一种token分类模型，这意味着它可以用来进行token分类。这是 NLP 中非常常见的任务，有助于识别文本中的特定实体，包括地点、名称、日期等。

## 🚀 Challenge

Complete the assignment linked below to train a named entity recognition model for medical terms, then try it on a different dataset.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/219)

## Review & Self Study

Read through the blog [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and follow along with the Further Reading section in that article to deepen your knowledge.

## [Assignment](lab/README.md)

<!-- In the assignment for this lesson, you will have to train a medical entity recognition model. You can start with training an LSTM model as described in this lesson, and proceed with using the BERT transformer model. Read [the instructions](lab/README.md) to get all the details. -->
在本课程的作业中，您将必须训练医疗实体识别模型。您可以按照本课程中的描述开始训练 LSTM 模型，然后继续使用 BERT 转换器模型。阅读[the instructions](lab/README.md)以获取所有详细信息。
