# Generative networks

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/117)

<!-- Recurrent Neural Networks (RNNs) and their gated cell variants such as Long Short Term Memory Cells (LSTMs) and Gated Recurrent Units (GRUs) provided a mechanism for language modeling in that they can learn word ordering and provide predictions for the next word in a sequence. This allows us to use RNNs for **generative tasks**, such as ordinary text generation, machine translation, and even image captioning. -->

循环神经网络 (RNN) 及其门控单元变体，例如长短期记忆单元 (LSTM) 和门控循环单元 (GRU) 提供了一种语言建模机制，因为它们可以学习单词排序并提供对单词中下一个单词的预测。顺序。这使得我们能够使用 RNN 来执行**生成任务**，例如普通文本生成、机器翻译，甚至图像字幕。

<!-- > ✅ Think about all the times you've benefited from generative tasks such as text completion as you type. Do some research into your favorite applications to see if they leveraged RNNs. -->

> ✅ 回想一下您从生成性任务（例如键入时完成文本）中受益的所有时间。对您最喜欢的应用程序进行一些研究，看看它们是否利用了 RNN。

<!-- In RNN architecture we discussed in the previous unit, each RNN unit produced the next hidden state as an output. However, we can also add another output to each recurrent unit, which would allow us to output a **sequence** (which is equal in length to the original sequence). Moreover, we can use RNN units that do not accept an input at each step, and just take some initial state vector, and then produce a sequence of outputs. -->

在我们在前面的单元中讨论的 RNN 架构中，每个 RNN 单元都会产生下一个隐藏状态作为输出。但是，我们还可以向每个循环单元添加另一个输出，这将允许我们输出一个**序列**（其长度与原始序列相同）。此外，我们可以使用在每一步不接受输入的 RNN 单元，而只采用一些初始状态向量，然后产生一系列输出。

<!-- This allows for different neural architectures that are shown in the picture below: -->
这允许使用不同的神经架构，如下图所示：

![Image showing common recurrent neural network patterns.](images/unreasonable-effectiveness-of-rnn.jpg)

> Image from blog post [Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by [Andrej Karpaty](http://karpathy.github.io/)

<!-- * **One-to-one** is a traditional neural network with one input and one output
* **One-to-many** is a generative architecture that accepts one input value, and generates a sequence of output values. For example, if we want to train an **image captioning** network that would produce a textual description of a picture, we can a picture as input, pass it through a CNN to obtain its hidden state, and then have a recurrent chain generate caption word-by-word
* **Many-to-one** corresponds to the RNN architectures we described in the previous unit, such as text classification
* **Many-to-many**, or **sequence-to-sequence** corresponds to tasks such as **machine translation**, where we have first RNN collect all information from the input sequence into the hidden state, and another RNN chain unrolls this state into the output sequence. -->

* **One-to-one** 一对一是传统的神经网络，一个输入一个输出
* **One-to-many** 一对多是一种生成式架构，它接受一个输入值，并生成一系列输出值。例如，如果我们想要训练一个**图像字幕**网络来生成图片的文本描述，我们可以将图片作为输入，将其传递给 CNN 以获得其隐藏状态，然后让循环链逐字生成字幕词
* **Many-to-one** 多对一对应于我们在上一单元中描述的RNN架构，例如文本分类
* **Many-to-many** 多对多或序列对序列对应于诸如机器翻译之类的任务，其中我们首先让 RNN 将输入序列中的所有信息收集到隐藏状态中，然后另一个 RNN 链将此状态展开到输出序列中。

<!-- In this unit, we will focus on simple generative models that help us generate text. For simplicity, we will use character-level tokenization. -->

在本单元中，我们将重点关注帮助我们生成文本的简单生成模型。为简单起见，我们将使用字符级标记化。

<!-- We will train this RNN to generate text step by step. On each step, we will take a sequence of characters of length `nchars`, and ask the network to generate the next output character for each input character: -->

我们将逐步训练这个 RNN 来生成文本。在每一步中，我们将采用长度为`nchars`的字符序列，并要求网络为每个输入字符生成下一个输出字符：

![Image showing an example RNN generation of the word 'HELLO'.](images/rnn-generate.png)

<!-- When generating text (during inference), we start with some **prompt**, which is passed through RNN cells to generate its intermediate state, and then from this state the generation starts. We generate one character at a time, and pass the state and the generated character to another RNN cell to generate the next one, until we generate enough characters. -->
当生成文本时（在推理过程中），我们从一些提示**开始**，该提示通过 RNN 单元生成其中间状态，然后从该状态开始生成。我们一次生成一个字符，并将状态和生成的字符传递给另一个 RNN 单元以生成下一个字符，直到生成足够的字符。

<img src="images/rnn-generate-inf.png" width="60%"/>

> Image by the author

## ✍️ Exercises: Generative Networks

Continue your learning in the following notebooks:

* [Generative Networks with PyTorch](GenerativePyTorch.ipynb)
* [Generative Networks with TensorFlow](GenerativeTF.ipynb)

## Soft text generation and temperature

<!-- The output of each RNN cell is a probability distribution of characters. If we always take the character with the highest probability as the next character in generated text, the text often can become "cycled" between the same character sequences again and again, like in this example: -->
每个 RNN 单元的输出是字符的概率分布。如果我们始终将概率最高的字符作为生成文本中的下一个字符，则文本通常会在相同的字符序列之间一次又一次地“循环”，如下例所示：

```
today of the second the company and a second the company ...
```

<!-- However, if we look at the probability distribution for the next character, it could be that the difference between a few highest probabilities is not huge, e.g. one character can have probability 0.2, another - 0.19, etc. For example, when looking for the next character in the sequence '*play*', next character can equally well be either space, or **e** (as in the word *player*). -->

然而，如果我们查看下一个字符的概率分布，可能几个最高概率之间的差异并不大，例如，一个字符的概率为 0.2，另一个字符的概率为 0.19，等等。例如，当查找序列“*play*”中的下一个字符，下一个字符同样可以是空格或**e**（如单词*player*中）。

<!-- This leads us to the conclusion that it is not always "fair" to select the character with a higher probability, because choosing the second highest might still lead us to meaningful text. It is more wise to **sample** characters from the probability distribution given by the network output. We can also use a parameter, **temperature**, that will flatten out the probability distribution, in case we want to add more randomness, or make it more steep, if we want to stick more to the highest-probability characters. -->

这使我们得出这样的结论：选择概率较高的字符并不总是“公平”的，因为选择第二高的字符仍然可能会引导我们得到有意义的文本。从网络输出给出的概率分布中采样字符是更明智的做法。我们还可以使用一个参数，**温度**，它将使概率分布变平，以防我们想要添加更多随机性，或者如果我们想要更多地关注最高概率的字符，则使其更加陡峭。

<!-- Explore how this soft text generation is implemented in the notebooks linked above. -->

探索如何在上面链接的notebook中实现这种软文本生成。

## Conclusion

<!-- While text generation may be useful in its own right, the major benefits come from the ability to generate text using RNNs from some initial feature vector. For example, text generation is used as part of machine translation (sequence-to-sequence, in this case state vector from *encoder* is used to generate or *decode* translated message), or generating textual description of an image (in which case the feature vector would come from CNN extractor). -->
虽然文本生成本身可能很有用，但主要好处来自于使用 RNN 从某些初始特征向量生成文本的能力。例如，文本生成用作机器翻译的一部分（序列到序列，在这种情况下，来自*编码器*的状态向量用于生成或解码翻译后的消息），或生成图像的文本描述（在这种情况下，特征向量将来自 CNN 提取器）。

## 🚀 Challenge

Take some lessons on Microsoft Learn on this topic

* Text Generation with [PyTorch](https://docs.microsoft.com/learn/modules/intro-natural-language-processing-pytorch/6-generative-networks/?WT.mc_id=academic-77998-cacaste)/[TensorFlow](https://docs.microsoft.com/learn/modules/intro-natural-language-processing-tensorflow/5-generative-networks/?WT.mc_id=academic-77998-cacaste)

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/217)

## Review & Self Study

Here are some articles to expand your knowledge

* Different approaches to text generation with Markov Chain, LSTM and GPT-2: [blog post](https://towardsdatascience.com/text-generation-gpt-2-lstm-markov-chain-9ea371820e1e)
* Text generation sample in [Keras documentation](https://keras.io/examples/generative/lstm_character_level_text_generation/)

## [Assignment](lab/README.md)

<!-- We have seen how to generate text character-by-character. In the lab, you will explore word-level text generation. -->
我们已经了解了如何逐个字符生成文本。在实验室中，您将探索单词级文本生成。
