# Neural Network Frameworks （神经网络框架）

<!-- As we have learned already, to be able to train neural networks efficiently we need to do two things: -->
正如我们已经了解到的，为了能够有效地训练神经网络，我们需要做两件事：

<!-- * To operate on tensors, eg. to multiply, add, and compute some functions such as sigmoid or softmax
* To compute gradients of all expressions, in order to perform gradient descent optimization -->

* 对张量进行操作，例如。乘法、加法和计算一些函数，例如 sigmoid 或 softmax
* 计算所有表达式的梯度，以执行梯度下降优化

<!-- ## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/105) -->

<!-- While the `numpy` library can do the first part, we need some mechanism to compute gradients. In [our framework](../04-OwnFramework/OwnFramework.ipynb) that we have developed in the previous section we had to manually program all derivative functions inside the `backward` method, which does backpropagation. Ideally, a framework should give us the opportunity to compute gradients of *any expression* that we can define. -->

虽然该numpy库可以完成第一部分，但我们需要一些机制来计算梯度。在上一节开发的框架[backward](../04-OwnFramework/OwnFramework.ipynb)中，我们必须手动编写方法内的所有导数函数，该方法会进行反向传播。理想情况下，框架应该让我们有机会计算我们可以定义的任何表达式的梯度。

<!-- Another important thing is to be able to perform computations on GPU, or any other specialized compute units, such as [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). Deep neural network training requires *a lot* of computations, and to be able to parallelize those computations on GPUs is very important. -->

另一件重要的事情是能够在 GPU 或任何其他专用计算单元（例如[TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)）上执行计算。深度神经网络训练需要大量计算，能够在 GPU 上并行化这些计算非常重要。

<!-- > ✅ The term 'parallelize' means to distribute the computations over multiple devices. -->
> 术语“并行化”意味着将计算分布在多个设备上。

<!-- Currently, the two most popular neural frameworks are: [TensorFlow](http://TensorFlow.org) and [PyTorch](https://pytorch.org/). Both provide a low-level API to operate with tensors on both CPU and GPU. On top of the low-level API, there is also higher-level API, called [Keras](https://keras.io/) and [PyTorch Lightning](https://pytorchlightning.ai/) correspondingly. -->

目前，最流行的两个神经框架是：[TensorFlow](http://TensorFlow.org)和[PyTorch](https://pytorch.org/)。两者都提供低级 API 来在 CPU 和 GPU 上操作张量。在低级 API 之上，还有更高级别的 API，分别称为[Keras](https://keras.io/)和[PyTorch Lightning](https://pytorchlightning.ai/)。

Low-Level API | [TensorFlow](http://TensorFlow.org) | [PyTorch](https://pytorch.org/)
--------------|-------------------------------------|--------------------------------
High-level API| [Keras](https://keras.io/) | [PyTorch Lightning](https://pytorchlightning.ai/)

<!-- **Low-level APIs** in both frameworks allow you to build so-called **computational graphs**. This graph defines how to compute the output (usually the loss function) with given input parameters, and can be pushed for computation on GPU, if it is available. There are functions to differentiate this computational graph and compute gradients, which can then be used for optimizing model parameters. -->

两个框架中的**低级 API**都允许构建所谓的**计算图**。该图定义了如何使用给定的输入参数计算输出（通常是损失函数），并且可以推送到 GPU 上进行计算（如果可用）。有一些函数可以区分该计算图并计算梯度，然后可以将其用于优化模型参数。

<!-- **High-level APIs** pretty much consider neural networks as a **sequence of layers**, and make constructing most of the neural networks much easier. Training the model usually requires preparing the data and then calling a `fit` function to do the job. -->

**高级 API**几乎将神经网络视为**神经网络层序列**，并且使构建大多数神经网络变得更加容易。训练模型通常需要准备数据，然后调用函数fit来完成这项工作。

<!-- The high-level API allows you to construct typical neural networks very quickly without worrying about lots of details. At the same time, low-level API offer much more control over the training process, and thus they are used a lot in research, when you are dealing with new neural network architectures. -->
高级 API 允许我们非常快速地构建典型的神经网络，而无需担心大量细节。同时，低级 API 提供了对训练过程的更多控制，因此当处理新的神经网络架构时，它们在研究中被大量使用。


<!-- It is also important to understand that you can use both APIs together, eg. you can develop your own network layer architecture using low-level API, and then use it inside the larger network constructed and trained with the high-level API. Or you can define a network using the high-level API as a sequence of layers, and then use your own low-level training loop to perform optimization. Both APIs use the same basic underlying concepts, and they are designed to work well together. -->
同样重要的是要了解我们可以同时使用这两个 API，例如。您可以使用低级 API 开发自己的网络层架构，然后在使用高级 API 构建和训练的更大网络中使用它。或者，我们可以使用高级 API 将网络定义为层序列，然后使用我们自己的低级训练循环来执行优化。这两个 API 使用相同的基本概念，并且它们被设计为可以很好地协同工作。

## Learning

<!-- In this course, we offer most of the content both for PyTorch and TensorFlow. You can choose your preferred framework and only go through the corresponding notebooks. If you are not sure which framework to choose, read some discussions on the internet regarding **PyTorch vs. TensorFlow**. You can also have a look at both frameworks to get better understanding. -->

在本课程中，我们介绍 PyTorch的用法。

<!-- Where possible, we will use High-Level APIs for simplicity. However, we believe it is important to understand how neural networks work from the ground up, thus in the beginning we start by working with low-level API and tensors. However, if you want to get going fast and do not want to spend a lot of time on learning these details, you can skip those and go straight into high-level API notebooks. -->

为了简单起见，我们将尽可能使用高级 API。然而，我们认为从头开始理解神经网络如何工作非常重要，因此一开始我们从使用低级 API 和张量开始。但是，如果想快速入门并且不想花费大量时间来学习这些细节，则可以跳过这些内容并直接进入高级 API 笔记本。

## ✍️ Exercises: Frameworks

Continue your learning in the following notebooks:

Low-Level API | [TensorFlow+Keras Notebook](IntroKerasTF.ipynb) | [PyTorch](IntroPyTorch.ipynb)
--------------|-------------------------------------|--------------------------------
High-level API| [Keras](IntroKeras.ipynb) | *PyTorch Lightning*

After mastering the frameworks, let's recap the notion of overfitting.

# Overfitting

<!-- Overfitting is an extremely important concept in machine learning, and it is very important to get it right! -->
过拟合是机器学习中极其重要的概念，正确理解它非常重要！

<!-- Consider the following problem of approximating 5 dots (represented by `x` on the graphs below): -->
考虑以下近似 5 个点的问题（x在下图中用表示）：

![linear](../images/overfit1.jpg) | ![overfit](../images/overfit2.jpg)
-------------------------|--------------------------
**Linear model, 2 parameters** | **Non-linear model, 7 parameters**
Training error = 5.3 | Training error = 0
Validation error = 5.1 | Validation error = 20

<!-- * On the left, we see a good straight line approximation. Because the number of parameters is adequate, the model gets the idea behind point distribution right. -->
<!-- * On the right, the model is too powerful. Because we only have 5 points and the model has 7 parameters, it can adjust in such a way as to pass through all points, making training the error to be 0. However, this prevents the model from understanding the correct pattern behind data, thus the validation error is very high. -->

* 在左边，我们看到了一个很好的直线近似。由于参数数量充足，该模型正确理解了点分布背后的想法。
* 右边的模型太强大了。因为我们只有 5 个点，而模型有 7 个参数，所以它可以调整为通过所有点，使训练误差为 0。但这会妨碍模型理解数据背后的正确模式，从而验证错误非常高。

<!-- It is very important to strike a correct balance between the richness of the model (number of parameters) and the number of training samples. -->
在模型的丰富度（参数数量）和训练样本数量之间取得正确的平衡非常重要。

## Why overfitting occurs

  * Not enough training data
  * Too powerful model
  * Too much noise in input data

## How to detect overfitting

<!-- As you can see from the graph above, overfitting can be detected by a very low training error, and a high validation error. Normally during training we will see both training and validation errors starting to decrease, and then at some point validation error might stop decreasing and start rising. This will be a sign of overfitting, and the indicator that we should probably stop training at this point (or at least make a snapshot of the model). -->
从上图中可以看出，过拟合可以通过非常低的训练误差和高验证误差来检测。通常在训练过程中，我们会看到训练误差和验证误差都开始减少，然后在某个时刻验证误差可能会停止减少并开始上升。这将是过度拟合的迹象，并且表明我们可能应该在此时停止训练（或者至少制作模型的快照）。

![overfitting](../images/Overfitting.png)

## How to prevent overfitting

<!-- If you can see that overfitting occurs, you can do one of the following: -->
如果您发现发生了过度拟合，您可以执行以下操作之一：

 * Increase the amount of training data
 * Decrease the complexity of the model
 * Use some [regularization technique](../../4-ComputerVision/08-TransferLearning/TrainingTricks.md), such as [Dropout](../../4-ComputerVision/08-TransferLearning/TrainingTricks.md#Dropout), which we will consider later.

## Overfitting and Bias-Variance Tradeoff

<!-- Overfitting is actually a case of a more generic problem in statistics called [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). If we consider the possible sources of error in our model, we can see two types of errors: -->
过拟合实际上是统计学中一个更常见的问题，称为“[偏差-方差权衡](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)”。如果我们考虑模型中可能的错误来源，我们可以看到两种类型的错误：

<!-- * **Bias errors** are caused by our algorithm not being able to capture the relationship between training data correctly. It can result from the fact that our model is not powerful enough (**underfitting**). -->
<!-- * **Variance errors**, which are caused by the model approximating noise in the input data instead of meaningful relationship (**overfitting**). -->

**偏差错误**是由于我们的算法无法正确捕获训练数据之间的关系而引起的。这可能是因为我们的模型不够强大（欠拟合）。
**方差误差**，这是由模型近似输入数据中的噪声而不是有意义的关系（过度拟合）引起的。

<!-- During training, bias error decreases (as our model learns to approximate the data), and variance error increases. It is important to stop training - either manually (when we detect overfitting) or automatically (by introducing regularization) - to prevent overfitting. -->
在训练过程中，偏差误差会减小（当我们的模型学习近似数据时），而方差误差会增加。重要的是停止训练——无论是手动（当我们检测到过度拟合时）还是自动（通过引入正则化）——以防止过度拟合。

## Conclusion

<!-- In this lesson, you learned about the differences between the various APIs for the two most popular AI frameworks, TensorFlow and PyTorch. In addition, you learned about a very important topic, overfitting. -->
在本课程中，您了解了两种最流行的 AI 框架 TensorFlow 和 PyTorch 的各种 API 之间的差异。另外，你还了解了一个非常重要的话题，过度拟合。

<!-- ## 🚀 Challenge

In the accompanying notebooks, you will find 'tasks' at the bottom; work through the notebooks and complete the tasks.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/205) -->


## [Assignment](lab/README.md)

<!-- In this lab, you are asked to solve two classification problems using single- and multi-layered fully-connected networks using PyTorch or TensorFlow. -->
使用pytorch实现单层和多层全连接神经网络实现二分类任务。

* [Instructions](lab/README.md)
* [Notebook](lab/LabFrameworks.ipynb)

