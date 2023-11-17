# Convolutional Neural Networks

<!-- We have seen before that neural networks are quite good at dealing with images, and even one-layer perceptron is able to recognize handwritten digits from MNIST dataset with reasonable accuracy. However, the MNIST dataset is very special, and all digits are centered inside the image, which makes the task simpler. -->
我们之前已经看到，神经网络非常擅长处理图像，甚至一层感知器也能够以合理的精度识别 MNIST 数据集中的手写数字。然而，MNIST 数据集非常特殊，所有数字都集中在图像内部，这使得任务更简单

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/107)

<!-- In real life, we want to be able to recognize objects on a picture regardless of their exact location in the image. Computer vision is different from generic classification, because when we are trying to find a certain object in the picture, we are scanning the image looking for some specific **patterns** and their combinations. For example, when looking for a cat, we first may look for horizontal lines, which can form whiskers, and then certain a combination of whiskers can tell us that it is actually a picture of a cat. Relative position and presence of certain patterns is important, and not their exact position on the image. -->

在现实生活中，我们希望能够识别图片上的物体，无论它们在图像中的确切位置如何。计算机视觉与一般分类不同，因为当我们试图在图片中找到某个对象时，我们正在扫描图像以寻找一些特定的**模式**及其组合。例如，当寻找一只猫时，我们首先可能会寻找可以形成胡须的水平线，然后某些胡须的组合可以告诉我们这实际上是一只猫的图片。某些图案的相对位置和存在很重要，而不是它们在图像上的确切位置。

<!-- To extract patterns, we will use the notion of **convolutional filters**. As you know, an image is represented by a 2D-matrix, or a 3D-tensor with color depth. Applying a filter means that we take relatively small **filter kernel** matrix, and for each pixel in the original image we compute the weighted average with neighboring points. We can view this like a small window sliding over the whole image, and averaging out all pixels according to the weights in the filter kernel matrix. -->
为了提取模式，我们将使用**卷积滤波器**的概念。如您所知，图像由 2D 矩阵或具有颜色深度的 3D 张量表示。应用滤波器意味着我们采用相对较小的**过滤/卷积内核**矩阵，并且对于原始图像中的每个像素，我们计算相邻点的加权平均值。我们可以将其视为在整个图像上滑动的小窗口，并根据滤波器内核矩阵中的权重对所有像素进行平均。

![Vertical Edge Filter](images/filter-vert.png) | ![Horizontal Edge Filter](images/filter-horiz.png)
----|----

> Image by Dmitry Soshnikov

<!-- For example, if we apply 3x3 vertical edge and horizontal edge filters to the MNIST digits, we can get highlights (e.g. high values) where there are vertical and horizontal edges in our original image. Thus those two filters can be used to "look for" edges. Similarly, we can design different filters to look for other low-level patterns: -->
例如，如果我们将 3x3 垂直边缘和水平边缘过滤器应用于 MNIST 数字，我们可以在原始图像中存在垂直和水平边缘的地方获得高光（例如高值）。因此，这两个过滤器可用于“寻找”边缘。同样，我们可以设计不同的过滤器来寻找其他低级模式：

<img src="images/lmfilters.jpg" width="500" align="center"/>


> Image of [Leung-Malik Filter Bank](https://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)

<!-- However, while we can design the filters to extract some patterns manually, we can also design the network in such a way that it will learn the patterns automatically. It is one of the main ideas behind the CNN. -->
然而，虽然我们可以设计过滤器来手动提取一些模式，但我们也可以设计网络使其自动学习模式。这是 CNN 背后的主要想法之一。

## Main ideas behind CNN

<!-- The way CNNs work is based on the following important ideas: -->
CNN 的工作方式基于以下重要思想：

<!-- * Convolutional filters can extract patterns
* We can design the network in such a way that filters are trained automatically
* We can use the same approach to find patterns in high-level features, not only in the original image. Thus CNN feature extraction work on a hierarchy of features, starting from low-level pixel combinations, up to higher level combination of picture parts. -->

* 卷积滤波器可以提取模式
* 我们可以以自动训练过滤器的方式设计网络
* 我们可以使用相同的方法来查找高级特征中的模式，而不仅仅是在原始图像中。因此，CNN 特征提取在特征层次上进行，从低级像素组合开始，一直到更高级别的图片部分组合。

![Hierarchical Feature Extraction](images/FeatureExtractionCNN.png)

> Image from [a paper by Hislop-Lynch](https://www.semanticscholar.org/paper/Computer-vision-based-pedestrian-trajectory-Hislop-Lynch/26e6f74853fc9bbb7487b06dc2cf095d36c9021d), based on [their research](https://dl.acm.org/doi/abs/10.1145/1553374.1553453)

## ✍️ Exercises: Convolutional Neural Networks

<!-- Let's continue exploring how convolutional neural networks work, and how we can achieve trainable filters, by working through the corresponding notebooks: -->
让我们通过相应的笔记本继续探索卷积神经网络的工作原理，以及如何实现可训练的滤波器：
* [Convolutional Neural Networks - PyTorch](ConvNetsPyTorch.ipynb)
* [Convolutional Neural Networks - TensorFlow](ConvNetsTF.ipynb)

## Pyramid Architecture

<!-- Most of the CNNs used for image processing follow a so-called pyramid architecture. The first convolutional layer applied to the original images typically has a relatively low number of filters (8-16), which correspond to different pixel combinations, such as horizontal/vertical lines of strokes. At the next level, we reduce the spatial dimension of the network, and increase the number of filters, which corresponds to more possible combinations of simple features. With each layer, as we move towards the final classifier, spatial dimensions of the image decrease, and the number of filters grow. -->
大多数用于图像处理的 CNN 都遵循所谓的金字塔架构。应用于原始图像的第一个卷积层通常具有相对较少数量的滤波器（8-16），它们对应于不同的像素组合，例如笔画的水平/垂直线。在下一个级别，我们减少网络的空间维度，并增加滤波器的数量，这对应于更多可能的简单特征组合。对于每一层，当我们走向最终分类器时，图像的空间维度会减少，而过滤器的数量会增加。

<!-- As an example, let's look at the architecture of VGG-16, a network that achieved 92.7% accuracy in ImageNet's top-5 classification in 2014: -->
作为一个例子，让我们看一下 VGG-16 的架构，该网络在 2014 年 ImageNet 的 top-5 分类中达到了 92.7% 的准确率：

![ImageNet Layers](images/vgg-16-arch1.jpg)

![ImageNet Pyramid](images/vgg-16-arch.jpg)

> Image from [Researchgate](https://www.researchgate.net/figure/Vgg16-model-structure-To-get-the-VGG-NIN-model-we-replace-the-2-nd-4-th-6-th-7-th_fig2_335194493)

## Best-Known CNN Architectures

[Continue your study about the best-known CNN architectures](CNN_Architectures.md)