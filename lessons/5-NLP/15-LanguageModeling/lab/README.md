# Training Skip-Gram Model

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

## Task

<!-- In this lab, you we challenge you to train Word2Vec model using Skip-Gram technique. Train a network with embedding to predict neighboring words in $N$-tokens-wide Skip-Gram window. You can use the [code from this lesson](../CBoW-TF.ipynb), and slightly modify it. -->

在本实验中，我们将挑战您使用 Skip-Gram 技术训练 Word2Vec 模型。使用嵌入训练网络来预测 $N$ 令牌范围的 Skip-Gram 窗口中的相邻单词。您可以使用本课程中的[代码](../CBoW-TF.ipynb)，并稍加修改。

## The Dataset

You are welcome to use any book. You can find a lot of free texts at [Project Gutenberg](https://www.gutenberg.org/), for example, here is a direct link to [Alice's Adventures in Wonderland](https://www.gutenberg.org/files/11/11-0.txt)) by Lewis Carroll. Or, you can use Shakespeare's plays, which you can get using the following code:

欢迎您使用任何书籍。您可以在[古腾堡计划](https://www.gutenberg.org/)中找到大量免费文本，例如，这里是刘易斯·卡罗尔所著的[《爱丽丝梦游仙境》](https://www.gutenberg.org/files/11/11-0.txt)的直接链接。或者，您可以使用莎士比亚的戏剧，您可以使用以下代码获得：

```python
path_to_file = tf.keras.utils.get_file(
   'shakespeare.txt', 
   'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
```

## Explore!

<!-- If you have time and want to get deeper into the subject, try to explore several things: -->
如果您有时间并且想要更深入地了解该主题，请尝试探索以下几件事：

<!-- * How does embedding size affects the results? -->
<!-- * How does different text styles affect the result? -->
<!-- * Take several very different types of words and their synonyms, obtain their vector representations, apply PCA to reduce dimensions to 2, and plot them in 2D space. Do you see any patterns? -->

* embedding的维度大小如何影响结果？
* 不同的文本样式如何影响结果？
* 采用几种截然不同类型的单词及其同义词，获取它们的向量表示，应用 PCA 将维度减少到 2，并将它们绘制在 2D 空间中。你看到什么模式了吗？
 