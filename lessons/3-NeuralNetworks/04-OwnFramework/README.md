<!-- # Introduction to Neural Networks. Multi-Layered Perceptron -->
# 神经网络，多层感知机
<!-- In the previous section, you learned about the simplest neural network model - one-layered perceptron, a linear two-class classification model. -->

在上一节中，我们了解了最简单的神经网络模型 - 单层感知机，一种线性二类分类模型。

在本节中，我们将此模型扩展为更灵活的框架，使我们能够：

* 除了二类分类之外还执行多类分类
* 除了分类之外还解决回归问题
* 分离不可线性分离的类

我们还将用 Python 开发我们自己的模块化框架，这将使我们能够构建不同的神经网络架构。
<!-- In this section we will extend this model into a more flexible framework, allowing us to: -->

<!-- * perform **multi-class classification** in addition to two-class -->
<!-- * solve **regression problems** in addition to classification -->
<!-- * separate classes that are not linearly separable -->

<!-- We will also develop our own modular framework in Python that will allow us to construct different neural network architectures. -->

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/104)

## Formalization of Machine Learning

<!-- Let's start with formalizing the Machine Learning problem. Suppose we have a training dataset **X** with labels **Y**, and we need to build a model *f* that will make most accurate predictions. The quality of predictions is measured by **Loss function** &lagran;. The following loss functions are often used: -->
让我们从形式化机器学习问题开始。假设我们有一个带有标签Y 的训练数据集X，并且我们需要构建一个模型f来做出最准确的预测。预测的质量通过损失函数ℒ 来衡量。经常使用以下损失函数：

* For regression problem, when we need to predict a number, we can use **absolute error** &sum;<sub>i</sub>|f(x<sup>(i)</sup>)-y<sup>(i)</sup>|, or **squared error** &sum;<sub>i</sub>(f(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
* For classification, we use **0-1 loss** (which is essentially the same as **accuracy** of the model), or **logistic loss**.

<!-- For one-level perceptron, function *f* was defined as a linear function *f(x)=wx+b* (here *w* is the weight matrix, *x* is the vector of input features, and *b* is bias vector). For different neural network architectures, this function can take more complex form. -->

对于一层感知机，函数f被定义为线性函数f(x)=wx+b（这里w是权重矩阵，x是输入特征向量，b是偏差向量）。对于不同的神经网络架构，该函数可以采用更复杂的形式。

<!-- > In the case of classification, it is often desirable to get probabilities of corresponding classes as network output. To convert arbitrary numbers to probabilities (eg. to normalize the output), we often use **softmax** function &sigma;, and the function *f* becomes *f(x)=&sigma;(wx+b)* -->

> 在分类的情况下，通常希望获得相应类别的概率作为网络输出。为了将任意数字转换为概率（例如，标准化输出），我们经常使用softmax函数σ，函数f变为f(x)=σ(wx+b)

<!-- In the definition of *f* above, *w* and *b* are called **parameters** &theta;=⟨*w,b*⟩. Given the dataset ⟨**X**,**Y**⟩, we can compute an overall error on the whole dataset as a function of parameters &theta;. -->

在上面f的定义中，w和b称为参数θ=⟨w ,b⟩。给定数据集 ⟨ X , Y ⟩，我们可以计算整个数据集上的总体误差作为参数 θ 的函数。

<!-- > ✅ **The goal of neural network training is to minimize the error by varying parameters &theta;** -->

> ✅ 神经网络训练的目标是通过改变参数 θ 来最小化误差

## Gradient Descent Optimization（梯度下降优化）

<!-- There is a well-known method of function optimization called **gradient descent**. The idea is that we can compute a derivative (in multi-dimensional case call **gradient**) of loss function with respect to parameters, and vary parameters in such a way that the error would decrease. This can be formalized as follows: -->

有一种众所周知的函数优化方法，称为梯度下降。这个想法是，我们可以计算损失函数相对于参数的导数（在多维情况下称为梯度），并以减少误差的方式改变参数。这可以形式化如下：


* Initialize parameters by some random values w<sup>(0)</sup>, b<sup>(0)</sup>
* Repeat the following step many times:
    - w<sup>(i+1)</sup> = w<sup>(i)</sup>-&eta;&part;&lagran;/&part;w
    - b<sup>(i+1)</sup> = b<sup>(i)</sup>-&eta;&part;&lagran;/&part;b

<!-- During training, the optimization steps are supposed to be calculated considering the whole dataset (remember that loss is calculated as a sum through all training samples). However, in real life we take small portions of the dataset called **minibatches**, and calculate gradients based on a subset of data. Because subset is taken randomly each time, such method is called **stochastic gradient descent** (SGD). -->

在训练期间，应该考虑整个数据集来计算优化步骤（请记住，损失是通过所有训练样本计算的总和）。然而，在现实生活中，我们采用称为小批量的数据集的一小部分，并根据数据子集计算梯度。由于每次子集都是随机选取的，因此这种方法称为随机梯度下降（SGD）。

## Multi-Layered Perceptrons and Backpropagation（多层感知机和反向传播）

<!-- One-layer network, as we have seen above, is capable of classifying linearly separable classes. To build a richer model, we can combine several layers of the network. Mathematically it would mean that the function *f* would have a more complex form, and will be computed in several steps: -->
正如我们上面所看到的，一层网络能够对线性可分类进行分类。为了构建更丰富的模型，我们可以组合多个网络层。从数学上来说，这意味着函数f将具有更复杂的形式，并且将分几个步骤进行计算：

* z<sub>1</sub>=w<sub>1</sub>x+b<sub>1</sub>
* z<sub>2</sub>=w<sub>2</sub>&alpha;(z<sub>1</sub>)+b<sub>2</sub>
* f = &sigma;(z<sub>2</sub>)

Here, &alpha; is a **non-linear activation function**, &sigma; is a softmax function, and parameters &theta;=<*w<sub>1</sub>,b<sub>1</sub>,w<sub>2</sub>,b<sub>2</sub>*>.

<!-- The gradient descent algorithm would remain the same, but it would be more difficult to calculate gradients. Given the chain differentiation rule, we can calculate derivatives as: -->
梯度下降算法将保持不变，但计算梯度会更加困难。根据链式求导规则，我们可以将导数计算为：

* &part;&lagran;/&part;w<sub>2</sub> = (&part;&lagran;/&part;&sigma;)(&part;&sigma;/&part;z<sub>2</sub>)(&part;z<sub>2</sub>/&part;w<sub>2</sub>)
* &part;&lagran;/&part;w<sub>1</sub> = (&part;&lagran;/&part;&sigma;)(&part;&sigma;/&part;z<sub>2</sub>)(&part;z<sub>2</sub>/&part;&alpha;)(&part;&alpha;/&part;z<sub>1</sub>)(&part;z<sub>1</sub>/&part;w<sub>1</sub>)

<!-- > ✅ The chain differentiation rule is used to calculate derivatives of the loss function with respect to parameters. -->
> ✅ 链式求导法则用于计算损失函数对于参数的导数。

<!-- Note that the left-most part of all those expressions is the same, and thus we can effectively calculate derivatives starting from the loss function and going "backwards" through the computational graph. Thus the method of training a multi-layered perceptron is called **backpropagation**, or 'backprop'. -->
请注意，所有这些表达式的最左边部分都是相同的，因此我们可以有效地从损失函数开始计算导数，并通过计算图“向后”进行计算。因此，训练多层感知器的方法称为反向传播。

<img alt="compute graph" src="images/ComputeGraphGrad.png"/>

> TODO: image citation

> ✅ We will cover backprop in much more detail in our notebook example.  

## 总结

<!-- In this lesson, we have built our own neural network library, and we have used it for a simple two-dimensional classification task. -->
本节课中，我们实现了自己的神经网络模型，并将其用于二分类任务。

<!-- ## 🚀 Challenge -->
## 挑战

<!-- In the accompanying notebook, you will implement your own framework for building and training multi-layered perceptrons. You will be able to see in detail how modern neural networks operate. -->
在本节课的ipynb代码[OwnFramework](OwnFramework.ipynb) 中，你需要设计并训练多层感知机模型，理解当前神经网络的学习过程。

<!-- Proceed to the [OwnFramework](OwnFramework.ipynb) notebook and work through it. -->

<!-- ## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/204) -->

<!-- ## Review & Self Study -->
## 复习与自学
反向传播是人工智能和机器学习中常用的优化算法，需要理解和掌握。
<!-- Backpropagation is a common algorithm used in AI and ML, worth studying [in more detail](https://wikipedia.org/wiki/Backpropagation) -->

<!-- ## [Assignment](lab/README.md) -->
## 作业

<!-- In this lab, you are asked to use the framework you constructed in this lesson to solve MNIST handwritten digit classification. -->
使用自己设计的多层感知机，实现手写字的分类任务。

* [Instructions](lab/README.md)
* [Notebook](lab/MyFW_MNIST.ipynb)
