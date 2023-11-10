<!-- # Introduction to Neural Networks: Perceptron -->
# 神经网络模型基础：感知机

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/103)

<!-- One of the first attempts to implement something similar to a modern neural network was done by Frank Rosenblatt from Cornell Aeronautical Laboratory in 1957. It was a hardware implementation called "Mark-1", designed to recognize primitive geometric figures, such as triangles, squares and circles. -->

康奈尔航空实验室的弗兰克·罗森布拉特 (Frank Rosenblatt) 于 1957 年首次尝试实现类似于现代神经网络的东西。这是一种名为“Mark-1”的硬件实现，旨在识别原始几何图形，例如三角形、正方形和圆圈。

|      |      |
|--------------|-----------|
|<img src='images/Rosenblatt-wikipedia.jpg' alt='Frank Rosenblatt'/> | <img src='images/Mark_I_perceptron_wikipedia.jpg' alt='The Mark 1 Perceptron' />|

> Images [from Wikipedia](https://en.wikipedia.org/wiki/Perceptron)

<!-- An input image was represented by 20x20 photocell array, so the neural network had 400 inputs and one binary output. A simple network contained one neuron, also called a **threshold logic unit**. Neural network weights acted like potentiometers that required manual adjustment during the training phase. -->

输入图像由 20x20 光电管阵列表示，因此神经网络有 400 个输入和一个二进制输出。一个简单的网络包含一个神经元，也称为阈值逻辑单元。神经网络权重就像电位器一样，需要在训练阶段手动调整。

<!-- > ✅ A potentiometer is a device that allows the user to adjust the resistance of a circuit. -->
> ✅ 电位器是一种允许用户调节电路电阻的装置。

<!-- > The New York Times wrote about perceptron at that time: *the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence.* -->

>《纽约时报》当时对感知机进行了报道：[海军]期望电子计算机的胚胎能够行走、说话、看、写、自我复制并意识到自己的存在。

## Perceptron Model

<!-- Suppose we have N features in our model, in which case the input vector would be a vector of size N. A perceptron is a **binary classification** model, i.e. it can distinguish between two classes of input data. We will assume that for each input vector x the output of our perceptron would be either +1 or -1, depending on the class. The output will be computed using the formula: -->

假设我们的模型中有 N 个特征，在这种情况下，输入向量将是大小为 N 的向量。感知机是二元分类模型，即它可以区分两类输入数据。我们假设对于每个输入向量 x，感知机的输出将为 +1 或 -1，具体取决于类别。将使用以下公式计算输出：

y(x) = f(w<sup>T</sup>x)

where f is a step activation function

<!-- img src="http://www.sciweavers.org/tex2img.php?eq=f%28x%29%20%3D%20%5Cbegin%7Bcases%7D%0A%20%20%20%20%20%20%20%20%20%2B1%20%26%20x%20%5Cgeq%200%20%5C%5C%0A%20%20%20%20%20%20%20%20%20-1%20%26%20x%20%3C%200%0A%20%20%20%20%20%20%20%5Cend%7Bcases%7D%20%5C%5C%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="f(x) = \begin{cases} +1 & x \geq 0 \\ -1 & x < 0 \end{cases} \\" width="154" height="50" / -->
<img src="images/activation-func.png"/>

## Training the Perceptron

<!-- To train a perceptron we need to find a weights vector w that classifies most of the values correctly, i.e. results in the smallest **error**. This error is defined by **perceptron criterion** in the following manner: -->

为了训练感知机，我们需要找到一个权重向量 w 来正确分类大多数值，即产生最小的误差。该误差由感知机标准按以下方式定义：

E(w) = -&sum;w<sup>T</sup>x<sub>i</sub>t<sub>i</sub>

where:

<!-- * the sum is taken on those training data points i that result in the wrong classification -->
<!-- * x<sub>i</sub> is the input data, and t<sub>i</sub> is either -1 or +1 for negative and positive examples accordingly. -->

* 对那些导致错误分类的训练数据点 i 求和
* x<sub>i</sub>是输入数据，对于负例和正例，t<sub>i</sub>相应地为-1或+1。

<!-- This criteria is considered as a function of weights w, and we need to minimize it. Often, a method called **gradient descent** is used, in which we start with some initial weights w<sup>(0)</sup>, and then at each step update the weights according to the formula: -->

该标准被视为权重 w 的函数，我们需要将其最小化。通常，使用称为梯度下降的方法，其中我们从一些初始权重w<sup>(0)</sup>开始，然后在每一步根据以下公式更新权重：

w<sup>(t+1)</sup> = w<sup>(t)</sup> - &eta;&nabla;E(w)

<!-- Here &eta; is the so-called **learning rate**, and &nabla;E(w) denotes the **gradient** of E. After we calculate the gradient, we end up with -->

这里 η 就是所谓的学习率，∇E(w) 表示E 的梯度。计算梯度后，我们得到

w<sup>(t+1)</sup> = w<sup>(t)</sup> + &sum;&eta;x<sub>i</sub>t<sub>i</sub>

The algorithm in Python looks like this:

```python
def train(positive_examples, negative_examples, num_iterations = 100, eta = 1):

    weights = [0,0,0] # Initialize weights (almost randomly :)
        
    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights) # compute perceptron output
        if z < 0: # positive example classified as negative
            weights = weights + eta*weights.shape

        z  = np.dot(neg, weights)
        if z >= 0: # negative example classified as positive
            weights = weights - eta*weights.shape

    return weights
```

## Conclusion

<!-- In this lesson, you learned about a perceptron, which is a binary classification model, and how to train it by using a weights vector. -->

在本课程中，您了解了感知机（一种二元分类模型），以及如何使用权重向量来训练它。

## 🚀 Challenge

If you'd like to try to build your own perceptron, try [this lab on Microsoft Learn](https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/two-class-averaged-perceptron?WT.mc_id=academic-77998-cacaste) which uses the [Azure ML designer](https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer?WT.mc_id=academic-77998-cacaste).

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/203)

## Review & Self Study

<!-- To see how we can use perceptron to solve a toy problem as well as real-life problems, and to continue learning - go to [Perceptron](Perceptron.ipynb) notebook. -->

<!-- Here's an interesting [article about perceptrons](https://towardsdatascience.com/what-is-a-perceptron-basics-of-neural-networks-c4cfea20c590  as well.-->

要了解如何使用感知机来解决玩具问题和现实生活中的问题，并继续学习 - 请访问[感知机notebook](Perceptron.ipynb)。

这里还有一篇关于感知机的[有趣文章](https://towardsdatascience.com/what-is-a-perceptron-basics-of-neural-networks-c4cfea20c590
) 。

## [Assignment](lab/README.md)

<!-- In this lesson, we have implemented a perceptron for binary classification task, and we have used it to classify between two handwritten digits. In this lab, you are asked to solve the problem of digit classification entirely, i.e. determine which digit is most likely to correspond to a given image. -->

在本课中，我们实现了用于二元分类任务的感知机，并用它来对两个手写数字进行分类。在本实验中，您需要完全解决数字分类问题，即确定哪个数字最有可能对应于给定图像。

* [Instructions](lab/README.md)
* [Notebook](PerceptronMultiClass.ipynb)
