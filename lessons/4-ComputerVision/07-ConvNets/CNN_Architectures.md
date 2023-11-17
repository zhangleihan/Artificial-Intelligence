# Well-Known CNN Architectures

### VGG-16

<!-- VGG-16 is a network that achieved 92.7% accuracy in ImageNet top-5 classification in 2014. It has the following layer structure: -->
VGG-16是2014年在ImageNet top-5分类中达到92.7%准确率的网络。它具有以下层结构：

![ImageNet Layers](images/vgg-16-arch1.jpg)

<!-- As you can see, VGG follows a traditional pyramid architecture, which is a sequence of convolution-pooling layers. -->
正如您所看到的，VGG 遵循传统的金字塔架构，它是一系列卷积-池化层。

![ImageNet Pyramid](images/vgg-16-arch.jpg)

> Image from [Researchgate](https://www.researchgate.net/figure/Vgg16-model-structure-To-get-the-VGG-NIN-model-we-replace-the-2-nd-4-th-6-th-7-th_fig2_335194493)

### ResNet

<!-- ResNet is a family of models proposed by Microsoft Research in 2015. The main idea of ResNet is to use **residual blocks**: -->
ResNet是微软研究院在2015年提出的一系列模型。ResNet的主要思想是使用残差块：

<img src="images/resnet-block.png" width="300"/>

> Image from [this paper](https://arxiv.org/pdf/1512.03385.pdf)

<!-- The reason for using identity pass-through is to have our layer predict **the difference** between the result of a previous layer and the output of the residual block - hence the name *residual*. Those blocks are much easier to train, and one can construct networks with several hundreds of those blocks (most common variants are ResNet-52, ResNet-101 and ResNet-152). -->
使用身份传递的原因是让我们的层预测前一层的结果与残差块的输出之间的**差异**- 因此称为**残差**。这些块更容易训练，并且可以构建具有数百个此类块的网络（最常见的变体是 ResNet-52、ResNet-101 和 ResNet-152）。

<!-- You can also think of this network as being able to adjust its complexity to the dataset. Initially, when you are starting to train the network, the weights values are small, and most of the signal goes through passthrough identity layers. As training progresses and weights become larger, the significance of network parameters grow, and the networks adjusts to accommodate required expressive power to correctly classify training images. -->
您还可以认为该网络能够根据数据集调整其复杂性。最初，当您开始训练网络时，权重值很小，并且大部分信号都会通过直通身份层。随着训练的进行和权重变大，网络参数的重要性随之增加，并且网络会进行调整以适应正确分类训练图像所需的表达能力。

### Google Inception

<!-- Google Inception architecture takes this idea one step further, and builds each network layer as a combination of several different paths: -->
Google Inception 架构将这一想法更进一步，并将每个网络层构建为几种不同路径的组合：

<img src="images/inception.png" width="400"/>

> Image from [Researchgate](https://www.researchgate.net/figure/Inception-module-with-dimension-reductions-left-and-schema-for-Inception-ResNet-v1_fig2_355547454)

<!-- Here, we need to emphasize the role of 1x1 convolutions, because at first they do not make sense. Why would we need to run through the image with 1x1 filter? However, you need to remember that convolution filters also work with several depth channels (originally - RGB colors, in subsequent layers - channels for different filters), and 1x1 convolution is used to mix those input channels together using different trainable weights. It can be also viewed as downsampling (pooling) over channel dimension. -->

在这里，我们需要强调 1x1 卷积的作用，因为一开始它们没有意义。为什么我们需要使用 1x1 滤镜来遍历图像？但是，您需要记住，卷积滤波器还可以使用多个深度通道（最初是 RGB 颜色，在后续层中是不同滤波器的通道），并且 1x1 卷积用于使用不同的可训练权重将这些输入通道混合在一起。它也可以被视为在通道维度上的下采样（池化）。

Here is [a good blog post](https://medium.com/analytics-vidhya/talented-mr-1x1-comprehensive-look-at-1x1-convolution-in-deep-learning-f6b355825578) on the subject, and [the original paper](https://arxiv.org/pdf/1312.4400.pdf).

### MobileNet

<!-- MobileNet is a family of models with reduced size, suitable for mobile devices. Use them if you are short in resources, and can sacrifice a little bit of accuracy. The main idea behind them is so-called **depthwise separable convolution**, which allows representing convolution filters by a composition of spatial convolutions and 1x1 convolution over depth channels. This significantly reduces the number of parameters, making the network smaller in size, and also easier to train with less data. -->
MobileNet 是一系列尺寸缩小的模型，适用于移动设备。如果您资源短缺并且可能会牺牲一点准确性，请使用它们。它们背后的主要思想是所谓的**深度可分离卷积（depthwise separable convolution）**，它允许通过空间卷积和深度通道上的 1x1 卷积的组合来表示卷积滤波器。这显着减少了参数数量，使网络尺寸更小，并且更容易用更少的数据进行训练。

Here is [a good blog post on MobileNet](https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470).

## Conclusion

<!-- In this unit, you have learned the main concept behind computer vision neural networks - convolutional networks. Real-life architectures that power image classification, object detection, and even image generation networks are all based on CNNs, just with more layers and some additional training tricks. -->
在本单元中，您学习了计算机视觉神经网络背后的主要概念 - 卷积网络。支持图像分类、对象检测甚至图像生成网络的现实架构均基于 CNN，只是具有更多层和一些额外的训练技巧。



## 🚀 Challenge

In the accompanying notebooks, there are notes at the bottom about how to obtain greater accuracy. Do some experiments to see if you can achieve higher accuracy.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/207)

## Review & Self Study

<!-- While CNNs are most often used for Computer Vision tasks, they are generally good for extracting fixed-sized patterns. For example, if we are dealing with sounds, we may also want to use CNNs to look for some specific patterns in audio signal - in which case filters would be 1-dimensional (and this CNN would be called 1D-CNN). Also, sometimes 3D-CNN is used to extract features in multi-dimensional space, such as certain events occurring on video - CNN can capture certain patterns of feature changing over time. Do some review and self-study about other tasks that can be done with CNNs. -->
虽然 CNN 最常用于计算机视觉任务，但它们通常适合提取固定大小的模式。例如，如果我们正在处理声音，我们可能还想使用 CNN 来查找音频信号中的某些特定模式 - 在这种情况下，滤波器将是一维的（该 CNN 将称为 1D-CNN）。此外，有时 3D-CNN 用于提取多维空间中的特征，例如视频上发生的某些事件 - CNN 可以捕获特征随时间变化的某些模式。对可以使用 CNN 完成的其他任务进行一些回顾和自学。

## [Assignment](lab/README.md)

<!-- In this lab, you are tasked with classifying different cat and dog breeds. These images are more complex than the MNIST dataset and of higher dimensions, and there are more than 10 classes. -->
在本实验室中，您的任务是对不同的猫和狗品种进行分类。这些图像比 MNIST 数据集更复杂，维度更高，有 10 多个类。