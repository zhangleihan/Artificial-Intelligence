# Autoencoders

<!-- When training CNNs, one of the problems is that we need a lot of labeled data. In the case of image classification, we need to separate images into different classes, which is a manual effort. -->
训练 CNN 时，问题之一是我们需要大量标记数据。在图像分类的情况下，我们需要将图像分成不同的类，这是一个手动工作。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/109)

<!-- However, we might want to use raw (unlabeled) data for training CNN feature extractors, which is called **self-supervised learning**. Instead of labels, we will use training images as both network input and output. The main idea of **autoencoder** is that we will have an **encoder network** that converts input image into some **latent space** (normally it is just a vector of some smaller size), then the **decoder network**, whose goal would be to reconstruct the original image. -->
然而，我们可能希望使用原始（未标记）数据来训练 CNN 特征提取器，这称为**自监督学习self-supervised learning**。我们将使用训练图像作为网络输入和输出，而不是标签。**autoencoder自编码器**的主要思想是，我们将有一个**encoder network编码器网络**，将输入图像转换为某个潜在空间（通常它只是一个较小尺寸的向量），然后是**decoder network解码器网络**，其目标是重建原始图像。



<!-- > ✅ An [autoencoder](https://wikipedia.org/wiki/Autoencoder) is "a type of artificial neural network used to learn efficient codings of unlabeled data." -->

> ✅自动编码器[autoencoder](https://wikipedia.org/wiki/Autoencoder)是“一种人工神经网络，用于学习未标记数据的有效编码。”

<!-- Since we are training an autoencoder to capture as much of the information from the original image as possible for accurate reconstruction, the network tries to find the best **embedding** of input images to capture the meaning.л. -->
由于我们正在训练自动编码器以从原始图像中捕获尽可能多的信息以进行准确重建，因此网络尝试找到输入图像的最佳**embedding**以捕获含义。

![AutoEncoder Diagram](images/autoencoder_schema.jpg)

> Image from [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html)

## Scenarios for using Autoencoders

<!-- While reconstructing original images does not seem useful in its own right, there are a few scenarios where autoencoders are especially useful: -->
虽然重建原始图像本身似乎没什么用，但在某些情况下自动编码器特别有用：

<!-- * **Lowering the dimension of images for visualization** or **training image embeddings**. Usually autoencoders give better results than PCA, because it takes into account spatial nature of images and hierarchical features.
* **Denoising**, i.e. removing noise from the image. Because noise carries out a lot of useless information, autoencoder cannot fit it all into relatively small latent space, and thus it captures only important part of the image. When training denoisers, we start with original images, and use images with artificially added noise as input for autoencoder.
* **Super-resolution**, increasing image resolution. We start with high-resolution images, and use the image with lower resolution as the autoencoder input.
* **Generative models**. Once we train the autoencoder, the decoder part can be used to create new objects starting from random latent vectors. -->

* **Lowering the dimension of images for visualization** or **training image embeddings**.。通常自动编码器给出的结果比 PCA 更好，因为它考虑了图像的空间性质和层次特征。
* **Denoising去噪**,，即去除图像中的噪声。由于噪声携带了大量无用信息，自动编码器无法将其全部放入相对较小的潜在空间中，因此它仅捕获图像的重要部分。在训练降噪器时，我们从原始图像开始，并使用人工添加噪声的图像作为自动编码器的输入。
* **Super-resolution**，提高图像分辨率。我们从高分辨率图像开始，并使用较低分辨率的图像作为自动编码器输入。
* **Generative models**。一旦我们训练了自动编码器，解码器部分就可以用于从随机潜在向量开始创建新对象。

## Variational Autoencoders (VAE)

<!-- Traditional autoencoders reduce the dimension of the input data somehow, figuring out the important features of input images. However, latent vectors ofter do not make much sense. In other words, taking MNIST dataset as an example, figuring out which digits correspond to different latent vectors is not an easy task, because close latent vectors would not necessarily correspond to the same digits. -->
传统的自动编码器以某种方式减少输入数据的维度，找出输入图像的重要特征。然而，潜在向量通常没有多大意义。换句话说，以 MNIST 数据集为例，找出哪些数字对应于不同的潜在向量并不是一件容易的任务，因为接近的潜在向量不一定对应于相同的数字。

<!-- On the other hand, to train *generative* models it is better to have some understanding of the latent space. This idea leads us to **variational auto-encoder** (VAE). -->
另一方面，为了训练生成模型，最好对潜在空间有一些了解。这个想法让我们想到了变分自动编码器**variational auto-encoder** (VAE)。

<!-- VAE is the autoencoder that learns to predict *statistical distribution* of the latent parameters, so-called **latent distribution**. For example, we may want latent vectors to be distributed normally with some mean z<sub>mean</sub> and standard deviation z<sub>sigma</sub> (both mean and standard deviation are vectors of some dimensionality d). Encoder in VAE learns to predict those parameters, and then decoder takes a random vector from this distribution to reconstruct the object. -->
VAE 是学习预测潜在参数的统计分布（即所谓的潜在分布**latent distribution**）的自动编码器。例如，我们可能希望潜在向量呈正态分布，具有某个平均值 z<sub>mean</sub>和标准差 z<sub>sigma</sub>（平均值和标准差都是某个维度 d 的向量）。VAE 中的编码器学习预测这些参数，然后解码器从该分布中获取随机向量来重建对象。

To summarize:

 * From input vector, we predict `z_mean` and `z_log_sigma` (instead of predicting the standard deviation itself, we predict its logarithm)
 * We sample a vector `sample` from the distribution N(z<sub>mean</sub>,exp(z<sub>log\_sigma</sub>))
 * The decoder tries to decode the original image using `sample` as an input vector

 <img src="images/vae.png" width="50%">

> Image from [this blog post](https://ijdykeman.github.io/ml/2016/12/21/cvae.html) by Isaak Dykeman

<!-- Variational auto-encoders use a complex loss function that consists of two parts: -->
变分自动编码器使用复杂的损失函数，该函数由两部分组成：

<!-- * **Reconstruction loss** is the loss function that shows how close a reconstructed image is to the target (it can be Mean Squared Error, or MSE). It is the same loss function as in normal autoencoders.
* **KL loss**, which ensures that latent variable distributions stays close to normal distribution. It is based on the notion of [Kullback-Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) - a metric to estimate how similar two statistical distributions are. -->

* **Reconstruction loss重建损失**是显示重建图像与目标的接近程度的损失函数（可以是均方误差，或 MSE）。它与普通自动编码器中的损失函数相同。
* **KL loss，KL 损失**，确保潜在变量分布保持接近正态分布。它基于[Kullback-Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)散度的概念- 一种估计两个统计分布相似程度的指标。

<!-- One important advantage of VAEs is that they allow us to generate new images relatively easily, because we know which distribution from which to sample latent vectors. For example, if we train VAE with 2D latent vector on MNIST, we can then vary components of the latent vector to get different digits: -->
VAE 的一个重要优点是它们使我们能够相对轻松地生成新图像，因为我们知道从哪个分布中对潜在向量进行采样。例如，如果我们在 MNIST 上使用 2D 潜在向量训练 VAE，那么我们可以改变潜在向量的组成部分以获得不同的数字：

<img alt="vaemnist" src="images/vaemnist.png" width="50%"/>

> Image by [Dmitry Soshnikov](http://soshnikov.com)

<!-- Observe how images blend into each other, as we start getting latent vectors from the different portions of the latent parameter space. We can also visualize this space in 2D: -->
当我们开始从潜在参数空间的不同部分获取潜在向量时，观察图像如何相互混合。我们还可以以二维方式可视化这个空间：



<img alt="vaemnist cluster" src="images/vaemnist-diag.png" width="50%"/> 

> Image by [Dmitry Soshnikov](http://soshnikov.com)

## ✍️ Exercises: Autoencoders

Learn more about autoencoders in these corresponding notebooks:

* [Autoencoders in TensorFlow](AutoencodersTF.ipynb)
* [Autoencoders in PyTorch](AutoEncodersPyTorch.ipynb)

## Properties of Autoencoders

<!-- * **Data Specific** - they only work well with the type of images they have been trained on. For example, if we train a super-resolution network on flowers, it will not work well on portraits. This is because the network can produce higher resolution image by taking fine details from features learned from the training dataset.
* **Lossy** - the reconstructed image is not the same as the original image. The nature of loss is defined by the *loss function* used during training
* Works on **unlabeled data** -->

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/209)

* **Data Specific数据特定**- 它们仅适用于经过训练的图像类型。例如，如果我们在花朵上训练超分辨率网络，则它在肖像上效果不佳。这是因为网络可以通过从训练数据集中学习到的特征获取精细细节来生成更高分辨率的图像。
* **Lossy有损**- 重建图像与原始图像不同。损失的性质由训练期间使用的损失函数定义
* 适用于**unlabeled data未标记的数据**

## Conclusion

<!-- In this lesson, you learned about the various types of autoencoders available to the AI scientist. You learned how to build them, and how to use them to reconstruct images. You also learned about the VAE and how to use it to generate new images. -->
在本课程中，您了解了人工智能科学家可用的各种类型的自动编码器。您学习了如何构建它们，以及如何使用它们来重建图像。您还了解了 VAE 以及如何使用它生成新图像。

## 🚀 Challenge

In this lesson, you learned about using autoencoders for images. But they can also be used for music! Check out the Magenta project's [MusicVAE](https://magenta.tensorflow.org/music-vae) project, which uses autoencoders to learn to reconstruct music. Do some [experiments](https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/Multitrack_MusicVAE.ipynb) with this library to see what you can create.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/208)

## Review & Self Study

For reference, read more about autoencoders in these resources:

* [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
* [Blog post on NeuroHive](https://neurohive.io/ru/osnovy-data-science/variacionnyj-avtojenkoder-vae/)
* [Variational Autoencoders Explained](https://kvfrans.com/variational-autoencoders-explained/)
* [Conditional Variational Autoencoders](https://ijdykeman.github.io/ml/2016/12/21/cvae.html)

## Assignment

At the end of [this notebook using TensorFlow](AutoencodersTF.ipynb), you will find a 'task' - use this as your assignment.