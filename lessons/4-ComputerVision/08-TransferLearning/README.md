# Pre-trained Networks and Transfer Learning

<!-- Training CNNs can take a lot of time, and a lot of data is required for that task. However, much of the time is spent learning the best low-level filters that a network can use to extract patterns from images. A natural question arises - can we use a neural network trained on one dataset and adapt it to classify different images without requiring a full training process? -->
训练 CNN 可能需要大量时间，并且该任务需要大量数据。然而，大部分时间都花在学习网络可用于从图像中提取模式的最佳低级滤波器上。一个自然的问题出现了——我们是否可以使用在一个数据集上训练的神经网络，并使其适应不同图像的分类，而不需要完整的训练过程？

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/108)

<!-- This approach is called **transfer learning**, because we transfer some knowledge from one neural network model to another. In transfer learning, we typically start with a pre-trained model, which has been trained on some large image dataset, such as **ImageNet**. Those models can already do a good job extracting different features from generic images, and in many cases just building a classifier on top of those extracted features can yield a good result. -->
这种方法称为**迁移学习**，因为我们将一些知识从一个神经网络模型迁移到另一个神经网络模型。在迁移学习中，我们通常从预训练的模型开始，该模型已经在一些大型图像数据集（例如**ImageNet**）上进行了训练。这些模型已经可以很好地从通用图像中提取不同的特征，并且在许多情况下，只需在这些提取的特征之上构建分类器就可以产生良好的结果。

<!-- > ✅ Transfer Learning is a term you find in other academic fields, such as Education. It refers to the process of taking knowledge from one domain and applying it to another. -->
> ✅ 迁移学习是您在其他学术领域（例如教育）中找到的术语。它是指从一个领域获取知识并将其应用到另一个领域的过程。

## Pre-Trained Models as Feature Extractors

<!-- The convolutional networks that we have talked about in the previous section contained a number of layers, each of which is supposed to extract some features from the image, starting from low-level pixel combinations (such as horizontal/vertical line or stroke), up to higher level combinations of features, corresponding to things like an eye of a flame. If we train CNN on sufficiently large dataset of generic and diverse images, the network should learn to extract those common features. -->
我们在上一节中讨论的卷积网络包含多个层，每个层都应该从图像中提取一些特征，从低级像素组合（例如水平/垂直线或笔划）开始，向上到更高级别的特征组合，对应于火焰之眼之类的东西。如果我们在足够大的通用和多样化图像数据集上训练 CNN，网络应该学会提取这些共同特征。

<!-- Both Keras and PyTorch contain functions to easily load pre-trained neural network weights for some common architectures, most of which were trained on ImageNet images. The most often used ones are described on the [CNN Architectures](../07-ConvNets/CNN_Architectures.md) page from the prior lesson. In particular, you may want to consider using one of the following: -->
Keras 和 PyTorch 都包含可以轻松加载某些常见架构的预训练神经网络权重的函数，其中大多数架构都是在 ImageNet 图像上进行训练的。最常用的在上一课的[CNN Architectures](../07-ConvNets/CNN_Architectures.md)页面中进行了描述。特别是，您可能需要考虑使用以下其中一项：

<!-- * **VGG-16/VGG-19** which are relatively simple models that still give good accuracy. Often using VGG as a first attempt is a good choice to see how transfer learning is working. -->
* **VGG-16/VGG-19**是相对简单的模型，但仍然具有良好的精度。通常使用 VGG 作为第一次尝试是了解迁移学习如何发挥作用的不错选择。

<!-- * **ResNet** is a family of models proposed by Microsoft Research in 2015. They have more layers, and thus take more resources. -->
* **ResNet**是微软研究院在2015年提出的一系列模型。它们有更多的层，因此占用更多的资源。

<!-- * **MobileNet** is a family of models with reduced size, suitable for mobile devices. Use them if you are short in resources and can sacrifice a little bit of accuracy. -->
* **MobileNet**是一系列尺寸缩小的模型，适用于移动设备。如果您资源短缺并且可能会牺牲一点准确性，请使用它们。

<!-- Here are sample features extracted from a picture of a cat by VGG-16 network: -->
以下是 VGG-16 网络从猫的图片中提取的示例特征：

![Features extracted by VGG-16](images/features.png)

## Cats vs. Dogs Dataset

In this example, we will use a dataset of [Cats and Dogs](https://www.microsoft.com/download/details.aspx?id=54765&WT.mc_id=academic-77998-cacaste), which is very close to a real-life image classification scenario.

## ✍️ Exercise: Transfer Learning

Let's see transfer learning in action in corresponding notebooks:

* [Transfer Learning - PyTorch](TransferLearningPyTorch.ipynb)
* [Transfer Learning - TensorFlow](TransferLearningTF.ipynb)

## Visualizing Adversarial Cat

<!-- Pre-trained neural network contains different patterns inside it's *brain*, including notions of **ideal cat** (as well as ideal dog, ideal zebra, etc.). It would be interesting to somehow **visualize this image**. However, it is not simple, because patterns are spread all over the network weights, and also organized in a hierarchical structure. -->
预先训练的神经网络在其大脑中包含不同的模式，包括**理想猫**的概念（以及理想狗、理想斑马等）。以某种方式**可视化这个图像**会很有趣。然而，它并不简单，因为模式分布在整个网络权重中，并且还以层次结构组织。

<!-- One approach we can take is to start with a random image, and then try to use **gradient descent optimization** technique to adjust that image in such a way, that the network starts thinking that it's a cat. -->
我们可以采取的一种方法是从随机图像开始，然后尝试使用梯度下降优化**gradient descent optimization**技术来调整该图像，使网络开始认为它是一只猫。


![Image Optimization Loop](images/ideal-cat-loop.png)

<!-- However, if we do this, we will receive something very similar to a random noise. This is because *there are many ways to make network think the input image is a cat*, including some that do not make sense visually. While those images contain a lot of patterns typical for a cat, there is nothing to constrain them to be visually distinctive. -->
然而，如果我们这样做，我们将收到与随机噪声非常相似的东西。这是因为 *有很多方法可以让网络认为输入图像是一只猫* ，包括一些在视觉上没有意义的方法。虽然这些图像包含许多猫的典型图案，但没有什么可以限制它们在视觉上的独特性。

<!-- To improve the result, we can add another term into the loss function, which is called **variation loss**. It is a metric that shows how similar neighboring pixels of the image are. Minimizing variation loss makes image smoother, and gets rid of noise - thus revealing more visually appealing patterns. Here is an example of such "ideal" images, that are classified as cat and as zebra with high probability: -->
为了改善结果，我们可以在损失函数中添加另一项，称为**变异损失**。它是一个显示图像相邻像素相似程度的指标。最大限度地减少变化损失使图像更平滑，并消除噪音 - 从而揭示更具视觉吸引力的图案。以下是此类“理想”图像的示例，它们很有可能被分类为猫和斑马：

![Ideal Cat](images/ideal-cat.png) | ![Ideal Zebra](images/ideal-zebra.png)
-----|-----
 *Ideal Cat* | *Ideal Zebra*

<!-- Similar approach can be used to perform so-called **adversarial attacks** on a neural network. Suppose we want to fool a neural network and make a dog look like a cat. If we take dog's image, which is recognized by a network as a dog, we can then tweak it a little but using gradient descent optimization, until the network starts classifying it as a cat: -->
类似的方法可用于对神经网络执行所谓的对抗性攻击**adversarial attacks**。假设我们想欺骗一个神经网络，让一只狗看起来像一只猫。如果我们采用狗的图像，网络将其识别为狗，然后我们可以使用梯度下降优化对其进行一些调整，直到网络开始将其分类为猫：

![Picture of a Dog](images/original-dog.png) | ![Picture of a dog classified as a cat](images/adversarial-dog.png)
-----|-----
*Original picture of a dog* | *Picture of a dog classified as a cat*

See the code to reproduce the results above in the following notebook:

* [Ideal and Adversarial Cat - TensorFlow](AdversarialCat_TF.ipynb)
## Conclusion

<!-- Using transfer learning, you are able to quickly put together a classifier for a custom object classification task and achieve high accuracy. You can see that more complex tasks that we are solving now require higher computational power, and cannot be easily solved on the CPU. In the next unit, we will try to use a more lightweight implementation to train the same model using lower compute resources, which results in just slightly lower accuracy. -->
使用迁移学习，您可以快速组合用于自定义对象分类任务的分类器并实现高精度。你可以看到，我们现在解决的更复杂的任务需要更高的计算能力，并且无法在CPU上轻松解决。在下一个单元中，我们将尝试使用更轻量级的实现来使用较低的计算资源来训练相同的模型，这会导致准确性稍低。

## 🚀 Challenge

In the accompanying notebooks, there are notes at the bottom about how transfer knowledge works best with somewhat similar training data (a new type of animal, perhaps). Do some experimentation with completely new types of images to see how well or poorly your transfer knowledge models perform.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/208)

## Review & Self Study

Read through [TrainingTricks.md](TrainingTricks.md) to deepen your knowledge of some other way to train your models.

## [Assignment](lab/README.md)

In this lab, we will use real-life [Oxford-IIIT](https://www.robots.ox.ac.uk/~vgg/data/pets/) pets dataset with 35 breeds of cats and dogs, and we will build a transfer learning classifier.
