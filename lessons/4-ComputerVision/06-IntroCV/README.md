# Introduction to Computer Vision

<!-- [Computer Vision](https://wikipedia.org/wiki/Computer_vision) is a discipline whose aim is to allow computers to gain high-level understanding of digital images. This is quite a broad definition, because *understanding* can mean many different things, including finding an object on a picture (**object detection**), understanding what is happening (**event detection**), describing a picture in text, or reconstructing a scene in 3D. There are also special tasks related to human images: age and emotion estimation, face detection and identification, and 3D pose estimation, to name a few. -->

[计算机视觉](https://wikipedia.org/wiki/Computer_vision)是一门学科，其目标是让计算机获得对数字图像的高级理解。这是一个相当广泛的定义，因为理解可以意味着许多不同的事情，包括在图片上查找对象（**目标检测**）、理解正在发生的事情（**事件检测**）、用文本描述图片或以 3D 形式重建场景。还有与人类图像相关的特殊任务：年龄和情绪估计、人脸检测和识别以及 3D 姿势估计等。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/106)

<!-- One of the simplest tasks of computer vision is **image classification**. -->
计算机视觉最简单的任务之一是**图像分类**。

<!-- Computer vision is often considered to be a branch of AI. Nowadays, most of computer vision tasks are solved using neural networks. We will learn more about the special type of neural networks used for computer vision, [convolutional neural networks](../07-ConvNets/README.md), throughout this section. -->
计算机视觉通常被认为是人工智能的一个分支。如今，大多数计算机视觉任务都是使用神经网络来解决的。我们将在本节中详细了解用于计算机视觉的特殊类型的神经网络，即[卷积神经网络](../07-ConvNets/README.md)。

<!-- However, before you pass the image to a neural network, in many cases it makes sense to use some algorithmic techniques to enhance the image. -->
然而，在将图像传递给神经网络之前，在许多情况下，使用一些算法技术来增强图像是有意义的。

<!-- There are several Python libraries available for image processing: -->
有几个可用于图像处理的 Python 库：

<!-- * **[imageio](https://imageio.readthedocs.io/en/stable/)** can be used for reading/writing different image formats. It also support ffmpeg, a useful tool to convert video frames to images.
* **[Pillow](https://pillow.readthedocs.io/en/stable/index.html)** (also known as PIL) is a bit more powerful, and also supports some image manipulation such as morphing, palette adjustments, and more.
* **[OpenCV](https://opencv.org/)** is a powerful image processing library written in C++, which has become the *de facto* standard for image processing. It has a convenient Python interface.
* **[dlib](http://dlib.net/)** is a C++ library that implements many machine learning algorithms, including some of the Computer Vision algorithms. It also has a Python interface, and can be used for challenging tasks such as face and facial landmark detection. -->

* **[imageio](https://imageio.readthedocs.io/en/stable/)**可用于读取/写入不同的图像格式。它还支持 ffmpeg，这是一个将视频帧转换为图像的有用工具。
* **[Pillow](https://pillow.readthedocs.io/en/stable/index.html)**（也称为 PIL）功能更强大一些，还支持一些图像操作，例如变形、调色板调整等。
* **[OpenCV](https://opencv.org/)**是一个用C++编写的强大的图像处理库，它已经成为图像处理事实上的标准。它有一个方便的Python 界面。
* **[dlib](http://dlib.net/)**是一个 C++ 库，它实现了许多机器学习算法，包括一些计算机视觉算法。它还具有 Python 界面，可用于具有挑战性的任务，例如面部和面部标志检测。


## OpenCV

<!-- [OpenCV](https://opencv.org/) is considered to be the *de facto* standard for image processing. It contains a lot of useful algorithms, implemented in C++. You can call OpenCV from Python as well. -->
[OpenCV](https://opencv.org/)被认为是图像处理事实上的标准。它包含许多有用的算法，用 C++ 实现。您也可以从 Python 调用 OpenCV。

<!-- A good place to learn OpenCV is [this Learn OpenCV course](https://learnopencv.com/getting-started-with-opencv/). In our curriculum, our goal is not to learn OpenCV, but to show you some examples when it can be used, and how. -->
学习 OpenCV 的好地方是学习 [OpenCV 课程](https://learnopencv.com/getting-started-with-opencv/)。在我们的课程中，我们的目标不是学习 OpenCV，而是向您展示一些何时可以使用它以及如何使用它的示例。

### Loading Images

<!-- Images in Python can be conveniently represented by NumPy arrays. For example, grayscale images with the size of 320x200 pixels would be stored in a 200x320 array, and color images of the same dimension would have shape of 200x320x3 (for 3 color channels). To load an image, you can use the following code: -->
Python 中的图像可以方便地用 NumPy 数组表示。例如，尺寸为 320x200 像素的灰度图像将存储在 200x320 数组中，相同尺寸的彩色图像将具有 200x320x3 的形状（对于 3 个颜色通道）。要加载图像，您可以使用以下代码：

```python
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('image.jpeg')
plt.imshow(im)
```

<!-- Traditionally, OpenCV uses BGR (Blue-Green-Red) encoding for color images, while the rest of Python tools use the more traditional RGB (Red-Green-Blue). For the image to look right, you need to convert it to the RGB color space, either by swapping dimensions in the NumPy array, or by calling an OpenCV function: -->
传统上，OpenCV 对彩色图像使用 BGR（蓝-绿-红）编码，而其他 Python 工具则使用更传统的 RGB（红-绿-蓝）。为了使图像看起来正确，您需要通过交换 NumPy 数组中的维度或调用 OpenCV 函数将其转换为 RGB 颜色空间：

```python
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
```

<!-- The same `cvtColor` function can be used to perform other color space transformations such as converting an image to grayscale or to the HSV (Hue-Saturation-Value) color space. -->
相同的`cvtColor`函数可用于执行其他颜色空间转换，例如将图像转换为灰度或 HSV（色调-饱和度-值）颜色空间。

<!-- You can also use OpenCV to load video frame-by-frame - an example is given in the exercise [OpenCV Notebook](OpenCV.ipynb). -->
您还可以使用 OpenCV 逐帧加载视频 - [OpenCV Notebook](OpenCV.ipynb)练习中给出了一个示例。

### Image Processing

<!-- Before feeding an image to a neural network, you may want to apply several pre-processing steps. OpenCV can do many things, including: -->
在将图像输入神经网络之前，您可能需要应用几个预处理步骤。OpenCV 可以做很多事情，包括：


* **Resizing** the image using `im = cv2.resize(im, (320,200),interpolation=cv2.INTER_LANCZOS)`
* **Blurring** the image using `im = cv2.medianBlur(im,3)` or `im = cv2.GaussianBlur(im, (3,3), 0)`
* Changing the **brightness and contrast** of the image can be done by NumPy array manipulations, as described [in this Stackoverflow note](https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv).
* Using [thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) by calling `cv2.threshold`/`cv2.adaptiveThreshold` functions, which is often preferable to adjusting brightness or contrast.

* Applying different [transformations](https://docs.opencv.org/4.5.5/da/d6e/tutorial_py_geometric_transformations.html) to the image:
    - **[Affine transformations](https://docs.opencv.org/4.5.5/d4/d61/tutorial_warp_affine.html)** can be useful if you need to combine rotation, resizing and skewing to the image and you know the source and destination location of three points in the image. Affine transformations keep parallel lines parallel.
    - **[Perspective transformations](https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143)** can be useful when you know the source and destination positions of 4 points in the image. For example, if you take a picture of a rectangular document via a smartphone camera from some angle, and you want to make a rectangular image of the document itself.
* Understanding movement inside the image by using **[optical flow](https://docs.opencv.org/4.5.5/d4/dee/tutorial_optical_flow.html)**.

对图像应用不同的[transformations](https://docs.opencv.org/4.5.5/da/d6e/tutorial_py_geometric_transformations.html)：
- 如果您需要将图像的旋转、调整大小和倾斜结合起来，并且您知道图像中三个点的源位置和目标位置，则仿射变换会很有用- **[Affine transformations](https://docs.opencv.org/4.5.5/d4/d61/tutorial_warp_affine.html)**。仿射变换使平行线保持平行。
- 当您知道图像中 4 个点的源位置和目标位置时，透视变换 **[Perspective transformations](https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143)** 会很有用。例如，如果您通过智能手机相机从某个角度拍摄矩形文档的照片，并且您想要制作文档本身的矩形图像。
- 使用 **[optical flow](https://docs.opencv.org/4.5.5/d4/dee/tutorial_optical_flow.html)** 了解图像内部的运动。

## Examples of using Computer Vision

<!-- In our [OpenCV Notebook](OpenCV.ipynb), we give some examples of when computer vision can be used to perform specific tasks: -->
在我们的[OpenCV Notebook](OpenCV.ipynb)中，我们给出了一些计算机视觉何时可用于执行特定任务的示例：

<!-- * **Pre-processing a photograph of a Braille book**. We focus on how we can use thresholding, feature detection, perspective transformation and NumPy manipulations to separate individual Braille symbols for further classification by a neural network. -->
* **预处理盲文书籍的照片**。我们重点关注如何使用阈值处理、特征检测、透视变换和 NumPy 操作来分离各个盲文符号，以便通过神经网络进行进一步分类。

![Braille Image](data/braille.jpeg) | ![Braille Image Pre-processed](images/braille-result.png) | ![Braille Symbols](images/braille-symbols.png)
----|-----|-----

> Image from [OpenCV.ipynb](OpenCV.ipynb)

<!-- * **Detecting motion in video using frame difference**. If the camera is fixed, then frames from the camera feed should be pretty similar to each other. Since frames are represented as arrays, just by subtracting those arrays for two subsequent frames we will get the pixel difference, which should be low for static frames, and become higher once there is substantial motion in the image. -->

* **使用帧差异检测视频中的运动**。如果相机是固定的，那么相机输入的帧应该非常相似。由于帧被表示为数组，只需减去两个后续帧的这些数组，我们就会得到像素差，对于静态帧来说像素差应该很低，一旦图像中有大量运动，像素差就会变得更高。

![Image of video frames and frame differences](images/frame-difference.png)

> Image from [OpenCV.ipynb](OpenCV.ipynb)

* **Detecting motion using Optical Flow**. [Optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) allows us to understand how individual pixels on video frames move. There are two types of optical flow:

* **使用光流检测运动**。[Optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)使我们能够了解视频帧上的各个像素如何移动。光流有两种类型：

   <!-- - **Dense Optical Flow** computes the vector field that shows for each pixel where is it moving -->
   <!-- - **Sparse Optical Flow** is based on taking some distinctive features in the image (eg. edges), and building their trajectory from frame to frame. -->

   - **Dense Optical Flow(密集光流)** 计算矢量场，显示每个像素的移动位置
   - **Sparse Optical Flow(稀疏光流)**基于获取图像中的一些独特特征（例如边缘），并在帧与帧之间建立它们的轨迹。

![Image of Optical Flow](images/optical.png)

> Image from [OpenCV.ipynb](OpenCV.ipynb)

## ✍️ Example Notebooks: OpenCV [try OpenCV in Action](OpenCV.ipynb)

Let's do some experiments with OpenCV by exploring [OpenCV Notebook](OpenCV.ipynb)

## Conclusion

<!-- Sometimes, relatively complex tasks such as movement detection or fingertip detection can be solved purely by computer vision. Thus, it is very helpful to know the basic techniques of computer vision, and what libraries like OpenCV can do. -->

有时，相对复杂的任务（例如运动检测或指尖检测）可以纯粹通过计算机视觉来解决。因此，了解计算机视觉的基本技术以及 OpenCV 等库的功能非常有帮助。

## 🚀 Challenge

Watch [this video](https://docs.microsoft.com/shows/ai-show/ai-show--2021-opencv-ai-competition--grand-prize-winners--cortic-tigers--episode-32?WT.mc_id=academic-77998-cacaste) from the AI show to learn about the Cortic Tigers project and how they built a block-based solution to democratize computer vision tasks via a robot. Do some research on other projects like this that help onboard new learners into the field.

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/206)

## Review & Self Study

Read more on optical flow [in this great tutorial](https://learnopencv.com/optical-flow-in-opencv/).

## [Assignment](lab/README.md)

In this lab, you will take a video with simple gestures, and your goal is to extract up/down/left/right movements using optical flow.

<img src="images/palm-movement.png" width="30%" alt="Palm Movement Frame"/>