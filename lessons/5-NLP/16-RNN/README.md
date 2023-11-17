# Recurrent Neural Networks

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/116)

<!-- In previous sections, we have been using rich semantic representations of text and a simple linear classifier on top of the embeddings. What this architecture does is to capture the aggregated meaning of words in a sentence, but it does not take into account the **order** of words, because the aggregation operation on top of embeddings removed this information from the original text. Because these models are unable to model word ordering, they cannot solve more complex or ambiguous tasks such as text generation or question answering. -->
在前面的部分中，我们一直在使用丰富的文本语义表示和嵌入之上的简单线性分类器。该架构的作用是捕获句子中单词的聚合含义，但它没有考虑单词的顺序，因为嵌入之上的聚合操作从原始文本中删除了这些信息。由于这些模型无法对词序进行建模，因此它们无法解决更复杂或模糊的任务，例如文本生成或问答。

<!-- To capture the meaning of text sequence, we need to use another neural network architecture, which is called a **recurrent neural network**, or RNN. In RNN, we pass our sentence through the network one symbol at a time, and the network produces some **state**, which we then pass to the network again with the next symbol. -->
为了捕获文本序列的含义，我们需要使用另一种神经网络架构，称为**循环神经网络（RNN）**。在 RNN 中，我们一次将一个符号通过网络传递句子，网络会产生一些**状态**，然后我们将其与下一个符号再次传递到网络。

![RNN](./images/rnn.png)

> Image by the author

<!-- Given the input sequence of tokens X<sub>0</sub>,...,X<sub>n</sub>, RNN creates a sequence of neural network blocks, and trains this sequence end-to-end using backpropagation. Each network block takes a pair (X<sub>i</sub>,S<sub>i</sub>) as an input, and produces S<sub>i+1</sub> as a result. The final state S<sub>n</sub> or (output Y<sub>n</sub>) goes into a linear classifier to produce the result. All the network blocks share the same weights, and are trained end-to-end using one backpropagation pass. -->
给定标记X<sub>0</sub>,...,X<sub>n</sub>的输入序列，RNN 创建神经网络块序列，并使用反向传播端到端地训练该序列。每个网络块采用一对 (X<sub>i</sub>,S<sub>i</sub>) 作为输入，并产生S<sub>i+1</sub>作为结果。最终状态S<sub>n</sub>或（输出Y<sub>n</sub>）进入线性分类器以产生结果。所有网络块共享相同的权重，并使用一次反向传播进行端到端训练。

<!-- Because state vectors S<sub>0</sub>,...,S<sub>n</sub> are passed through the network, it is able to learn the sequential dependencies between words. For example, when the word *not* appears somewhere in the sequence, it can learn to negate certain elements within the state vector, resulting in negation. -->
由于状态向量 S<sub>0</sub>,...,S<sub>n</sub>通过网络传递，因此能够学习单词之间的顺序依赖关系。例如，当单词*not*出现在序列中的某个位置时，它可以学习对状态向量中的某些元素求反，从而导致否定。

<!-- > ✅ Since the weights of all RNN blocks on the picture above are shared, the same picture can be represented as one block (on the right) with a recurrent feedback loop, which passes the output state of the network back to the input. -->

> ✅ 由于上图中所有 RNN 块的权重都是共享的，因此同一张图片可以表示为一个具有循环反馈循环的块（右侧），该循环将网络的输出状态传递回输入。

## Anatomy of an RNN Cell

<!-- Let's see how a simple RNN cell is organized. It accepts the previous state S<sub>i-1</sub> and current symbol X<sub>i</sub> as inputs, and has to produce the output state S<sub>i</sub> (and, sometimes, we are also interested in some other output Y<sub>i</sub>, as in the case with generative networks). -->
让我们看看一个简单的 RNN 单元是如何组织的。它接受先前状态 S<sub>i-1</sub>和当前符号X<sub>i</sub>作为输入，并且必须产生输出状态 S<sub>i</sub>（有时，我们也对其他输出 Y<sub>i</sub>感兴趣，就像生成网络的情况一样）。

<!-- A simple RNN cell has two weight matrices inside: one transforms an input symbol (let's call it W), and another one transforms an input state (H). In this case the output of the network is calculated as &sigma;(W&times;X<sub>i</sub>+H&times;S<sub>i-1</sub>+b), where &sigma; is the activation function and b is additional bias. -->
一个简单的 RNN 单元内部有两个权重矩阵：一个变换输入符号（我们称之为 W），另一个变换输入状态 (H)。在这种情况下，网络的输出计算为 &sigma;(W&times;X<sub>i</sub>+H&times;S<sub>i-1</sub>+b)，其中&sigma;是激活函数，b 是附加偏差。

<img alt="RNN Cell Anatomy" src="images/rnn-anatomy.png" width="50%"/>

> Image by the author

In many cases, input tokens are passed through the embedding layer before entering the RNN to lower the dimensionality. In this case, if the dimension of the input vectors is *emb_size*, and state vector is *hid_size* - the size of W is *emb_size*&times;*hid_size*, and the size of H is *hid_size*&times;*hid_size*.
在许多情况下，输入标记在进入 RNN 之前会经过嵌入层以降低维度。在这种情况下，如果输入向量的维度为emb_size，状态向量为hid_size - W 的大小为emb_size × hid_size，H 的大小为hid_size × hid_size。

## Long Short Term Memory (LSTM)

<!-- One of the main problems of classical RNNs is the so-called **vanishing gradients** problem. Because RNNs are trained end-to-end in one backpropagation pass, it has difficulty propagating error to the first layers of the network, and thus the network cannot learn relationships between distant tokens. One of the ways to avoid this problem is to introduce **explicit state management** by using so called **gates**. There are two well-known architectures of this kind: **Long Short Term Memory** (LSTM) and **Gated Relay Unit** (GRU). -->

经典 RNN 的主要问题之一是所谓的**梯度消失**问题。由于 RNN 在一次反向传播过程中进行端到端训练，因此很难将误差传播到网络的第一层，因此网络无法学习远处标记之间的关系。避免此问题的方法之一是通过使用所谓的门来引入**显式状态管理**。这种类型有两种著名的架构：**长短期记忆（LSTM）**和**门控中继单元（GRU）**。

![Image showing an example long short term memory cell](./images/long-short-term-memory-cell.svg)

> Image source TBD

<!-- The LSTM Network is organized in a manner similar to RNN, but there are two states that are being passed from layer to layer: the actual state C, and the hidden vector H. At each unit, the hidden vector H<sub>i</sub> is concatenated with input X<sub>i</sub>, and they control what happens to the state C via **gates**. Each gate is a neural network with sigmoid activation (output in the range [0,1]), which can be thought of as a bitwise mask when multiplied by the state vector. There are the following gates (from left to right on the picture above): -->
LSTM 网络的组织方式与 RNN 类似，但有两个状态在层与层之间传递：实际状态 C 和隐藏向量 H。在每个单元，隐藏向量H<sub>i</sub>与输入连接 X<sub>i</sub>，它们通过**gates**控制状态 C 发生的情况。每个门都是一个具有 sigmoid 激活的神经网络（输出在 [0,1] 范围内），当乘以状态向量时，可以将其视为按位掩码。有以下门（上图从左到右）：

<!-- * The **forget gate** takes a hidden vector and determines which components of the vector C we need to forget, and which to pass through.
* The **input gate** takes some information from the input and hidden vectors and inserts it into state.
* The **output gate** transforms state via a linear layer with *tanh* activation, then selects some of its components using a hidden vector H<sub>i</sub> to produce a new state C<sub>i+1</sub>. -->

* **forget gate**采用一个隐藏向量并确定我们需要忘记向量 C 的哪些分量以及要通过哪些分量。
* **input gate**从输入和隐藏向量中获取一些信息并将其插入到状态中。
* **output gate**通过具有tanh激活的线性层转换状态，然后使用隐藏向量H<sub>i</sub>选择其一些组件以产生新状态C<sub>i+1</sub>。

<!-- Components of the state C can be thought of as some flags that can be switched on and off. For example, when we encounter a name *Alice* in the sequence, we may want to assume that it refers to a female character, and raise the flag in the state that we have a female noun in the sentence. When we further encounter phrases *and Tom*, we will raise the flag that we have a plural noun. Thus by manipulating state we can supposedly keep track of the grammatical properties of sentence parts. -->
状态C的组成部分可以被认为是一些可以打开和关闭的标志。例如，当我们在序列中遇到一个名字*Alice*时，我们可能想假设它指的是一个女性角色，并在句子中有一个女性名词的情况下举起标志。当我们进一步遇到短语*and Tom*时，我们将举起我们有复数名词的标志。因此，通过操纵状态，我们可以跟踪句子部分的语法属性。

> ✅ An excellent resource for understanding the internals of LSTM is this great article [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

## Bidirectional and Multilayer RNNs

<!-- We have discussed recurrent networks that operate in one direction, from beginning of a sequence to the end. It looks natural, because it resembles the way we read and listen to speech. However, since in many practical cases we have random access to the input sequence, it might make sense to run recurrent computation in both directions. Such networks are call **bidirectional** RNNs. When dealing with bidirectional network, we would need two hidden state vectors, one for each direction. -->

我们已经讨论了从序列的开始到结束以一个方向运行的循环网络。它看起来很自然，因为它类似于我们阅读和听演讲的方式。然而，由于在许多实际情况下我们可以随机访问输入序列，因此在两个方向上运行循环计算可能是有意义的。这种网络称为**双向RNNs**。在处理双向网络时，我们需要两个隐藏状态向量，每个方向一个。

<!-- A Recurrent network, either one-directional or bidirectional, captures certain patterns within a sequence, and can store them into a state vector or pass into output. As with convolutional networks, we can build another recurrent layer on top of the first one to capture higher level patterns and build from low-level patterns extracted by the first layer. This leads us to the notion of a **multi-layer RNN** which consists of two or more recurrent networks, where the output of the previous layer is passed to the next layer as input. -->

单向或双向的循环网络捕获序列中的某些模式，并可以将它们存储到状态向量中或传递到输出中。与卷积网络一样，我们可以在第一个循环层之上构建另一个循环层，以捕获更高级别的模式并根据第一层提取的低级模式进行构建。这引出了**多层 RNN**的概念，它由两个或多个循环网络组成，其中前一层的输出作为输入传递到下一层。

![Image showing a Multilayer long-short-term-memory- RNN](./images/multi-layer-lstm.jpg)

*Picture from [this wonderful post](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3) by Fernando López*

## ✍️ Exercises: Embeddings

Continue your learning in the following notebooks:

* [RNNs with PyTorch](RNNPyTorch.ipynb)
* [RNNs with TensorFlow](RNNTF.ipynb)

## Conclusion

<!-- In this unit, we have seen that RNNs can be used for sequence classification, but in fact, they can handle many more tasks, such as text generation, machine translation, and more. We will consider those tasks in the next unit. -->
在本单元中，我们已经看到 RNN 可用于序列分类，但实际上，它们可以处理更多任务，例如文本生成、机器翻译等。我们将在下一个单元中考虑这些任务。

## 🚀 Challenge

<!-- Read through some literature about LSTMs and consider their applications: -->
阅读一些有关 LSTM 的文献并考虑它们的应用：

- [Grid Long Short-Term Memory](https://arxiv.org/pdf/1507.01526v1.pdf)
- [Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention](https://arxiv.org/pdf/1502.03044v2.pdf)

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/216)

## Review & Self Study

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

## [Assignment: Notebooks](assignment.md)
