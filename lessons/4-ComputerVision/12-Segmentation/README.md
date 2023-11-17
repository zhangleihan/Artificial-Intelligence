# Segmentation

<!-- We have previously learned about Object Detection, which allows us to locate objects in the image by predicting their *bounding boxes*. However, for some tasks we do not only need bounding boxes, but also more precise object localization. This task is called  **segmentation**. -->
我们之前学习过对象检测，它允许我们通过预测对象的边界框来定位图像中的对象。然而，对于某些任务，我们不仅需要*bounding boxes边界框*，还需要更精确的对象定位。这个任务称为**segmentation分割**。


## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/112)

<!-- Segmentation can be viewed as **pixel classification**, whereas for **each** pixel of image we must predict its class (*background* being one of the classes). There are two main segmentation algorithms: -->
分割可以被视为像素分类，而对于图像的**每个**像素，我们必须预测其类别（**背景**是类别之一）。主要有两种分割算法：

<!-- * **Semantic segmentation** only tells the pixel class, and does not make a distinction between different objects of the same class
* **Instance segmentation** divides classes into different instances. -->

* **Semantic segmentation语义分割**只告诉像素类别，并不区分同一类别的不同对象
* **Instance segmentation实例分割**将类划分为不同的实例。

<!-- For instance segmentation, these sheep are different objects, but for semantic segmentation all sheep are represented by one class. -->
对于实例分割，这些羊是不同的对象，但对于语义分割，所有羊都由一个类表示。

<img src="images/instance_vs_semantic.jpeg" width="50%">

> Image from [this blog post](https://nirmalamurali.medium.com/image-classification-vs-semantic-segmentation-vs-instance-segmentation-625c33a08d50)

<!-- There are different neural architectures for segmentation, but they all have the same structure. In a way, it is similar to the autoencoder you learned about previously, but instead of deconstructing the original image, our goal is to deconstruct a **mask**. Thus, a segmentation network has the following parts: -->
用于分割的神经架构有不同，但它们都具有相同的结构。在某种程度上，它类似于您之前了解的自动编码器，但我们的目标不是解构原始图像，而是解构**mask**。因此，分割网络具有以下部分：

<!-- * **Encoder** extracts features from input image
* **Decoder** transforms those features into the **mask image**, with the same size and number of channels corresponding to the number of classes. -->

* **Encoder编码器**从输入图像中提取特征
* **Decoder解码器**将这些特征转换为**掩模图像**，其大小和通道数与类别数相对应。

<img src="images/segm.png" width="80%">

> Image from [this publication](https://arxiv.org/pdf/2001.05566.pdf)

<!-- We should especially mention the loss function that is used for segmentation. When using classical autoencoders, we need to measure the similarity between two images, and we can use mean square error (MSE) to do that. In segmentation, each pixel in the target mask image represents the class number (one-hot-encoded along the third dimension), so we need to use loss functions specific for classification - cross-entropy loss, averaged over all pixels. If the mask is binary - **binary cross-entropy loss** (BCE) is used. -->

我们应该特别提到用于分割的损失函数。当使用经典自动编码器时，我们需要测量两个图像之间的相似度，我们可以使用均方误差（MSE）来做到这一点。在分割中，目标掩模图像中的每个像素代表类别号（沿第三维进行单热编码），因此我们需要使用特定于分类的损失函数——交叉熵损失，对所有像素进行平均。如果掩码是二进制的 -使用二进制交叉熵损失(BCE)**binary cross-entropy loss** (BCE)。

<!-- > ✅ One-hot encoding is a way to encode a class label into a vector of length equal to the number of classes. Take a look at [this article](https://datagy.io/sklearn-one-hot-encode/) on this technique. -->

> ✅ One-hot 编码是一种将类标签编码为长度等于类数的向量的方法。看看这篇关于这项技术的[文章](https://datagy.io/sklearn-one-hot-encode/)。

## Segmentation for Medical Imaging

<!-- In this lesson, we will see the segmentation in action by training the network to recognize human nevi (also known as moles) on medical images. We will be using <a href="https://www.fc.up.pt/addi/ph2%20database.html">PH<sup>2</sup> Database</a> of dermoscopy images as the image source. This dataset contains 200 images of three classes: typical nevus, atypical nevus, and melanoma. All images also contain a corresponding **mask** that outlines the nevus. -->
在本课程中，我们将通过训练网络识别医学图像上的人类痣（也称为痣）来了解分割的实际效果。我们将使用皮肤镜图像的<a href="https://www.fc.up.pt/addi/ph2%20database.html">PH<sup>2</sup> Database</a>数据库作为图像源。该数据集包含三类的 200 张图像：典型痣、非典型痣和黑色素瘤。所有图像还包含勾勒出痣轮廓的相应掩模。

<!-- > ✅ This technique is particularly appropriate for this type of medical imaging, but what other real-world applications could you envision? -->
> ✅ 这项技术特别适合这种类型的医学成像，但您还能想到哪些其他实际应用？

<img alt="navi" src="images/navi.png"/>

> Image from the PH<sup>2</sup> Database

<!-- We will train a model to segment any nevus from its background. -->
我们将训练一个模型来从背景中分割任何痣。

## ✍️ Exercises: Semantic Segmentation

<!-- Open the notebooks below to learn more about different semantic segmentation architectures, practice working with them, and see them in action. -->
打开下面的笔记本，了解有关不同语义分割架构的更多信息，练习使用它们，并查看它们的实际效果。

* [Semantic Segmentation Pytorch](SemanticSegmentationPytorch.ipynb)
* [Semantic Segmentation TensorFlow](SemanticSegmentationTF.ipynb)

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/212)

## Conclusion

<!-- Segmentation is a very powerful technique for image classification, moving beyond bounding boxes to pixel-level classification. It is a technique used in medical imaging, among other applications. -->
分割是一种非常强大的图像分类技术，超越了边界框到像素级分类。它是一种用于医学成像等应用的技术。

## 🚀 Challenge

<!-- Body segmentation is just one of the common tasks that we can do with images of people. Another important tasks include **skeleton detection** and **pose detection**. Try out [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) library to see how pose detection can be used. -->
身体分割只是我们可以对人物图像执行的常见任务之一。另一个重要的任务包括**skeleton detection骨骼检测**和**pose detection姿势检测**。尝试[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 库，了解如何使用姿势检测。

## Review & Self Study

<!-- This [wikipedia article](https://wikipedia.org/wiki/Image_segmentation) offers a good overview of the various applications of this technique. Learn more on your own about the subdomains of Instance segmentation and Panoptic segmentation in this field of inquiry. -->

这篇[维基百科文章](https://wikipedia.org/wiki/Image_segmentation)很好地概述了该技术的各种应用。自行了解有关该研究领域中的实例分割和全景分割子域的更多信息。

## [Assignment](lab/README.md)

In this lab, try **human body segmentation** using [Segmentation Full Body MADS Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset) from Kaggle.

