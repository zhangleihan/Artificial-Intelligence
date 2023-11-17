# Deep Learning Training Tricks

As neural networks become deeper, the process of their training becomes more and more challenging. One major problem is so-called [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) or [exploding gradients](https://deepai.org/machine-learning-glossary-and-terms/exploding-gradient-problem#:~:text=Exploding%20gradients%20are%20a%20problem,updates%20are%20small%20and%20controlled.). [This post](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11) gives a good introduction into those problems.

To make training deep networks more efficient, there are a few techniques that can be used.

## Keeping values in reasonable interval

<!-- To make numerical computations more stable, we want to make sure that all values within our neural network are within reasonable scale, typically [-1..1] or [0..1]. It is not a very strict requirement, but the nature of floating point computations is such that values of different magnitudes cannot be accurately manipulated together. For example, if we add 10<sup>-10</sup> and 10<sup>10</sup>, we are likely to get 10<sup>10</sup>, because smaller value would be "converted" to the same order as the larger one, and thus mantissa would be lost. -->

为了使数值计算更加稳定，我们希望确保神经网络中的所有值都在合理的范围内，通常为 [-1..1] 或 [0..1]。这不是一个非常严格的要求，但浮点计算的本质是不同量值的值无法精确地一起操作。例如，如果我们将10<sup>-10</sup>和10<sup>10</sup>加起来，我们很可能会得到 10<sup>10</sup>，因为较小的值将被“转换”为与较大的值相同的顺序，因此尾数将丢失。

<!-- Most activation functions have non-linearities around [-1..1], and thus it makes sense to scale all input data to [-1..1] or [0..1] interval. -->
大多数激活函数在 [-1..1] 附近具有非线性，因此将所有输入数据缩放到 [-1..1] 或 [0..1] 区间是有意义的。

## Initial Weight Initialization

<!-- Ideally, we want the values to be in the same range after passing through network layers. Thus it is important to initialize weights in such a way as to preserve the distribution of values. -->
理想情况下，我们希望这些值在经过网络层后处于相同的范围内。因此，以保持值分布的方式初始化权重非常重要。

<!-- Normal distribution **N(0,1)** is not a good idea, because if we have *n* inputs, the standard deviation of output would be *n*, and values are likely to jump out of [0..1] interval. -->
正态分布**N(0,1)**不是一个好主意，因为如果我们有n 个输入，输出的标准差将为n，并且值可能会跳出 [0..1] 区间。


<!-- The following initializations are often used: -->
经常使用以下初始化：

 * Uniform distribution -- `uniform`
 * **N(0,1/n)** -- `gaussian`
 * **N(0,1/&radic;n_in)** guarantees that for inputs with zero mean and standard deviation of 1 the same mean/standard deviation would remain
 * **N(0,&radic;2/(n_in+n_out))** -- so-called **Xavier initialization** (`glorot`), it helps to keep the signals in range during both forward and backward propagation

## Batch Normalization

<!-- Even with proper weight initialization, weights can get arbitrary big or small during the training, and they will bring signals out of proper range. We can bring signals back by using one of **normalization** techniques. While there are several of them (Weight normalization, Layer Normalization), the most often used is Batch Normalization. -->
即使进行了适当的权重初始化，权重在训练过程中也可能变得任意大或小，并且它们将使信号超出适当的范围。我们可以通过使用一种归一化技术**normalization**来恢复信号。虽然有几种（权重归一化、层归一化），但最常用的是批量归一化。


<!-- The idea of **batch normalization** is to take into account all values across the minibatch, and perform normalization (i.e. subtract mean and divide by standard deviation) based on those values. It is implemented as a network layer that does this normalization after applying the weights, but before activation function. As a result, we are likely to see higher final accuracy and faster training. -->
批量归一化**batch normalization**的想法是考虑小批量中的所有值，并根据这些值执行归一化（即减去平均值并除以标准差）。它被实现为一个网络层，在应用权重之后但在激活函数之前进行归一化。因此，我们可能会看到更高的最终准确性和更快的训练。

Here is the [original paper](https://arxiv.org/pdf/1502.03167.pdf) on batch normalization, the [explanation on Wikipedia](https://en.wikipedia.org/wiki/Batch_normalization), and [a good introductory blog post](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338) (and the one [in Russian](https://habrahabr.ru/post/309302/)).

## Dropout

<!-- **Dropout** is an interesting technique that removes a certain percentage of random neurons during training. It is also implemented as a layer with one parameter (percentage of neurons to remove, typically 10%-50%), and during training it zeroes random elements of the input vector, before passing it to the next layer. -->
**Dropout** 是一种有趣的技术，它在训练过程中删除一定比例的随机神经元。它还被实现为具有一个参数的层（要删除的神经元的百分比，通常为 10%-50%），并且在训练期间，它在将输入向量传递到下一层之前将其随机元素清零。

<!-- While this may sound like a strange idea, you can see the effect of dropout on training MNIST digit classifier in [`Dropout.ipynb`](Dropout.ipynb) notebook. It speeds up training and allows us to achieve higher accuracy in less training epochs. -->
虽然这听起来像是一个奇怪的想法，但您可以看到 dropout 对笔记本中训练 MNIST 数字分类器的影响[`Dropout.ipynb`](Dropout.ipynb)。它加快了训练速度，使我们能够在更少的训练周期内获得更高的准确率。

<!-- This effect can be explained in several ways: -->
这种效应可以用多种方式来解释：
 <!-- * It can be considered to be a random shocking factor to the model, which takes optimiation out of local minimum -->
 <!-- * It can be considered as *implicit model averaging*, because we can say that during dropout we are training slightly different model -->

 * 它可以被认为是模型的随机冲击因素，它从局部最小值中进行优化
 * 它可以被认为是隐式模型平均，因为我们可以说在dropout期间我们正在训练稍微不同的模型

<!-- > *Some people say that when a drunk person tries to learn something, he will remember this better next morning, comparing to a sober person, because a brain with some malfunctioning neurons tries to adapt better to gasp the meaning. We never tested ourselves if this is true of not* -->

> *有人说，与清醒的人相比，当醉酒的人试图学习某些东西时，他第二天早上会记得更好，因为大脑中的一些神经元出现故障，试图更好地适应并理解其含义。我们从未测试过自己这是真的还是假的*

## Preventing overfitting

<!-- One of the very important aspect of deep learning is too be able to prevent [overfitting](../../3-NeuralNetworks/05-Frameworks/Overfitting.md). While it might be tempting to use very powerful neural network model, we should always balance the number of model parameters with the number of training samples. -->
深度学习非常重要的方面之一就是能够防止过度拟合[overfitting](../../3-NeuralNetworks/05-Frameworks/Overfitting.md)。虽然使用非常强大的神经网络模型可能很诱人，但我们应该始终平衡模型参数的数量与训练样本的数量。

> Make sure you understand the concept of [overfitting](../../3-NeuralNetworks/05-Frameworks/Overfitting.md) we have introduced earlier!

There are several ways to prevent overfitting:

 * Early stopping -- continuously monitor error on validation set and stopping training when validation error starts to increase.
 * Explicit Weight Decay / Regularization -- adding an extra penalty to the loss function for high absolute values of weights, which prevents the model of getting very unstable results
 * Model Averaging -- training several models and then averaging the result. This helps to minimize the variance.
 * Dropout (Implicit Model Averaging)

 * 早停——持续监控验证集上的错误，并在验证错误开始增加时停止训练。
 * 显式权重衰减/正则化——为权重绝对值较高的损失函数添加额外的惩罚，这可以防止模型获得非常不稳定的结果
 * 模型平均——训练多个模型，然后对结果进行平均。这有助于最小化方差。
 * Dropout（隐式模型平均）

## Optimizers / Training Algorithms

<!-- Another important aspect of training is to chose good training algorithm. While classical **gradient descent** is a reasonable choice, it can sometimes be too slow, or result in other problems. -->
训练的另一个重要方面是选择好的训练算法。虽然经典梯度下降**gradient descent**是一个合理的选择，但它有时可能太慢，或导致其他问题。

<!-- In deep learning, we use **Stochastic Gradient Descent** (SGD), which is a gradient descent applied to minibatches, randomly selected from the training set. Weights are adjusted using this formula: -->
在深度学习中，我们使用**随机梯度下降（SGD）**，这是一种应用于小批量的梯度下降，从训练集中随机选择。使用以下公式调整权重：

w<sup>t+1</sup> = w<sup>t</sup> - &eta;&nabla;&lagran;

### Momentum

<!-- In **momentum SGD**, we are keeping a portion of a gradient from previous steps. It is similar to when we are moving somewhere with inertia, and we receive a punch in a different direction, our trajectory does not change immediately, but keeps some part of the original movement. Here we introduce another vector v to represent the *speed*: -->
在**动量 SGD**中，我们保留了之前步骤的一部分梯度。这类似于当我们以惯性运动到某处时，受到不同方向的一拳，我们的轨迹不会立即改变，而是保留原来运动的某些部分。这里我们引入另一个向量v来表示速度：

* v<sup>t+1</sup> = &gamma; v<sup>t</sup> - &eta;&nabla;&lagran;
* w<sup>t+1</sup> = w<sup>t</sup>+v<sup>t+1</sup>

<!-- Here parameter &gamma; indicates the extent to which we take inertia into account: &gamma;=0 corresponds to classical SGD; &gamma;=1 is a pure motion equation. -->

这里参数&gamma;表示我们考虑惯性的程度：&gamma;=0对应于经典SGD；&gamma;=1是纯运动方程。

### Adam, Adagrad, etc.

<!-- Since in each layer we multiply signals by some matrix W<sub>i</sub>, depending on ||W<sub>i</sub>||, the gradient can either diminish and be close to 0, or rise indefinitely. It is the essence of Exploding/Vanishing Gradients problem. -->
由于在每一层中，我们将信号乘以某个矩阵W<sub>i</sub>，具体取决于||W<sub>i</sub>||，梯度可以减小并接近 0，也可以无限上升。这是梯度爆炸/消失问题的本质。


<!-- One of the solutions to this problem is to use only direction of the gradient in the equation, and ignore the absolute value, i.e. -->
解决这个问题的方法之一是在方程中只使用梯度的方向，而忽略绝对值，即

w<sup>t+1</sup> = w<sup>t</sup> - &eta;(&nabla;&lagran;/||&nabla;&lagran;||), where ||&nabla;&lagran;|| = &radic;&sum;(&nabla;&lagran;)<sup>2</sup>

<!-- This algorithm is called **Adagrad**. Another algorithms that use the same idea: **RMSProp**, **Adam** -->
该算法称为**Adagrad**。另一种使用相同思想的算法：**RMSProp**、**Ada**

<!-- > **Adam** is considered to be a very efficient algorithm for many applications, so if you are not sure which one to use - use Adam. -->
> 对于许多应用程序来说， **Adam** 被认为是一种非常高效的算法，因此，如果您不确定使用哪一种算法，请使用 Adam。

### Gradient clipping

<!-- Gradient clipping is an extension the idea above. When the ||&nabla;&lagran;|| &le; &theta;, we consider the original gradient in the weight optimization, and when ||&nabla;&lagran;|| > &theta; - we divide the gradient by it's norm. Here &theta; is a parameter, in most cases we can take &theta;=1 or &theta;=10. -->

渐变裁剪是上述想法的延伸。当||&nabla;&lagran;|| &le; &theta;，权重优化时考虑原始梯度，当||&nabla;&lagran;|| > &theta; - 我们将梯度除以它的范数。这里&theta;是一个参数，大多数情况下我们可以取&theta;=1 or &theta;=10。

### Learning rate decay

<!-- Training success often depends on the learning rate parameter &eta;. It is logical to assume that larger values of &eta; result in faster training, which is something we typically want in the beginning of the training, and then smaller value of &eta; allow us to fine-tune the network. Thus, in most of the cases we want to decrease &eta; in the process of the training. -->
训练的成功通常取决于学习率参数&eta;。假设较大的&eta; 值会导致更快的训练（这是我们在训练开始时通常希望的）是合乎逻辑的，然后较小的&eta; 值允许我们微调网络。因此，在大多数情况下，我们希望在训练过程中减小&eta;。

<!-- This can be done by multiplying &eta; by some number (eg. 0.98) after each epoch of the training, or by using more complicated **learning rate schedule**. -->
这可以通过在训练的每个时期后将 &eta乘以某个数字（例如 0.98）或使用更复杂的学习率规划**learning rate schedule**来完成。

## Different Network Architectures

<!-- Selecting right network architecture for your problem can be tricky. Normally, we would take an architecture that has proven to work for our specific task (or similar one). Here is a [good overview](https://www.topbots.com/a-brief-history-of-neural-network-architectures/) or neural network architectures for computer vision. -->
为您的问题选择正确的网络架构可能很棘手。通常，我们会采用一种已被证明适用于我们的特定任务（或类似任务）的架构。这是计算机视觉神经网络架构的一个很好的[概述](https://www.topbots.com/a-brief-history-of-neural-network-architectures/)。


> It is important to select an architecture that will be powerful enough for the number of training samples that we have. Selecting too powerful model can result in [overfitting](../../3-NeuralNetworks/05-Frameworks/Overfitting.md)

> 选择一个对于我们拥有的训练样本数量来说足够强大的架构非常重要。选择太强大的模型可能会导致过度拟合[overfitting](../../3-NeuralNetworks/05-Frameworks/Overfitting.md)。

Another good way would be to use and architecture that will automatically adjust to the required complexity. To some extent, **ResNet** architecture and **Inception** are self-adjusting. [More on computer vision architectures](../07-ConvNets/CNN_Architectures.md)
另一个好方法是使用能够自动适应所需复杂性的架构。在某种程度上，**ResNet**架构和**Inception**是自我调整的。有关计算机视觉架构的[更多信息](../07-ConvNets/CNN_Architectures.md)
