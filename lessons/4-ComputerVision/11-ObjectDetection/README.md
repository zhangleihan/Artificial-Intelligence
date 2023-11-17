# Object Detection

<!-- The image classification models we have dealt with so far took an image and produced a categorical result, such as the class 'number' in a MNIST problem. However, in many cases we do not want just to know that a picture portrays objects - we want to be able to determine their precise location. This is exactly the point of **object detection**. -->
到目前为止，我们处理的图像分类模型获取图像并生成分类结果，例如 MNIST 问题中的“数字”类。然而，在许多情况下，我们不仅仅想知道图片描绘了物体 - 我们希望能够确定它们的精确位置。这正是**目标检测**的重点。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/111)

![Object Detection](images/Screen_Shot_2016-11-17_at_11.14.54_AM.png)

> Image from [YOLO v2 web site](https://pjreddie.com/darknet/yolov2/)

## A Naive Approach to Object Detection

<!-- Assuming we wanted to find a cat on a picture, a very naive approach to object detection would be the following: -->

<!-- 1. Break the picture down to a number of tiles -->
<!-- 2. Run image classification on each tile. -->
<!-- 3. Those tiles that result in sufficiently high activation can be considered to contain the object in question. -->

假设我们想在图片上找到一只猫，一种非常简单的对象检测方法如下：

1. 将图片分解为多个图块
2. 在每个图块上运行图像分类。
3. 那些导致足够高激活的图块可以被认为包含所讨论的对象。

![Naive Object Detection](images/naive-detection.png)

> *Image from [Exercise Notebook](ObjectDetection-TF.ipynb)*

<!-- However, this approach is far from ideal, because it only allows the algorithm to locate the object's bounding box very imprecisely. For more precise location, we need to run some sort of **regression** to predict the coordinates of bounding boxes - and for that, we need specific datasets. -->
然而，这种方法远非理想，因为它只能让算法非常不精确地定位对象的边界框。为了更精确的定位，我们需要运行某种**回归**来预测边界框的坐标 - 为此，我们需要特定的数据集。

## Regression for Object Detection

<!-- [This blog post](https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491) has a great gentle introduction to detecting shapes. -->
这篇[博文](https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491)对形状检测进行了非常温和的介绍。

## Datasets for Object Detection

<!-- You might run across the following datasets for this task: -->
您可能会为此任务运行以下数据集：

* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - 20 classes
* [COCO](http://cocodataset.org/#home) - Common Objects in Context. 80 classes, bounding boxes and segmentation masks

![COCO](images/coco-examples.jpg)

## Object Detection Metrics

### Intersection over Union

<!-- While for image classification it is easy to measure how well the algorithm performs, for object detection we need to measure both the correctness of the class, as well as the precision of the inferred bounding box location. For the latter, we use the so-called **Intersection over Union** (IoU), which measures how well two boxes (or two arbitrary areas) overlap. -->

对于图像分类，很容易测量算法的性能，而对于对象检测，我们需要测量类别的正确性以及推断的边界框位置的精度。对于后者，我们使用所谓的**交并比（IoU）**，它测量两个框（或两个任意区域）的重叠程度。


![IoU](images/iou_equation.png)

> *Figure 2 from [this excellent blog post on IoU](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)*

<!-- The idea is simple - we divide the area of intersection between two figures by the area of their union. For two identical areas, IoU would be 1, while for completely disjointed areas it will be 0. Otherwise it will vary from 0 to 1. We typically only consider those bounding boxes for which IoU is over a certain value. -->
这个想法很简单——我们用两个图形之间的交集面积除以它们的并集面积。对于两个相同的区域，IoU 将为 1，而对于完全不相交的区域，它将为 0。否则它将从 0 到 1 变化。我们通常只考虑 IoU 超过某个值的边界框。

### Average Precision

<!-- Suppose we want to measure how well a given class of objects $C$ is recognized. To measure it, we use **Average Precision** metrics, which is calculated as follows: -->
假设我们想要测量给定类别的对象 $C$ 的识别程度。为了衡量它，我们使用**平均精度**指标，其计算如下：

<!-- 1. Consider Precision-Recall curve shows the accuracy depending on a detection threshold value (from 0 to 1).
2. Depending on the threshold, we will get more or less objects detected in the image, and different values of precision and recall.
3. The curve will look like this: -->

1. 考虑精确率-召回率曲线显示的准确度取决于检测阈值（从 0 到 1）。
2. 根据阈值，我们将在图像中检测到更多或更少的对象，以及不同的精度和召回值。
3. 曲线将如下所示：

<img src="https://github.com/shwars/NeuroWorkshop/raw/master/images/ObjDetectionPrecisionRecall.png"/>

> *Image from [NeuroWorkshop](http://github.com/shwars/NeuroWorkshop)*

<!-- The average Precision for a given class $C$ is the area under this curve. More precisely, Recall axis is typically divided into 10 parts, and Precision is averaged over all those points: -->

给定类别 $C$ 的平均精度是该曲线下的面积。更准确地说，Recall 轴通常分为 10 个部分，Precision 是所有这些点的平均值：

$$
AP = {1\over11}\sum_{i=0}^{10}\mbox{Precision}(\mbox{Recall}={i\over10})
$$

### AP and IoU

<!-- We shall consider only those detections, for which IoU is above a certain value. For example, in PASCAL VOC dataset typically $\mbox{IoU Threshold} = 0.5$ is assumed, while in COCO AP is measured for different values of $\mbox{IoU Threshold}$. -->

我们将只考虑那些 IoU 高于某个值的检测。例如，在 PASCAL VOC 数据集中，通常假设 $\mbox{IoU Threshold} = 0.5$，而在 COCO AP 中，则针对 $\mbox{IoU Threshold}$ 的不同值进行测量。

<img src="https://github.com/shwars/NeuroWorkshop/raw/master/images/ObjDetectionPrecisionRecallIoU.png"/>

> *Image from [NeuroWorkshop](http://github.com/shwars/NeuroWorkshop)*

### Mean Average Precision - mAP

<!-- The main metric for Object Detection is called **Mean Average Precision**, or **mAP**. It is the value of Average Precision, average across all object classes, and sometimes also over $\mbox{IoU Threshold}$. In more detail, the process of calculating **mAP** is described -->
<!-- [in this blog post](https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3)), and also [here with code samples](https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734). -->

对象检测的主要指标称为**平均精度 (Mean Average Precision)或mAP**。它是平均精度的值，是所有对象类的平均值，有时也超过 $\mbox{IoU Threshold}$。更详细地，计算**mAP**的过程在这篇[博文](https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3)中有描述，这里还提供了[代码示例](https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734)。

## Different Object Detection Approaches

<!-- There are two broad classes of object detection algorithms: -->
目标检测算法有两大类：

<!-- * **Region Proposal Networks** (R-CNN, Fast R-CNN, Faster R-CNN). The main idea is to generate **Regions of Interests** (ROI) and run CNN over them, looking for maximum activation. It is a bit similar to the naive approach, with the exception that ROIs are generated in a more clever way. One of the majors drawbacks of such methods is that they are slow, because we need many passes of the CNN classifier over the image. -->
* **Region Proposal Networks** 区域提议网络（R-CNN、Fast R-CNN、Faster R-CNN）。主要思想是生成**感兴趣区域(ROI)** 并在其上运行 CNN，寻找最大激活值。它有点类似于简单的方法，不同之处在于 ROI 是以更聪明的方式生成的。这种方法的主要缺点之一是速度慢，因为我们需要对图像进行多次 CNN 分类器的传递。

<!-- * **One-pass** (YOLO, SSD, RetinaNet) methods. In those architectures we design the network to predict both classes and ROIs in one pass. -->
* **One-pass** 一次性（YOLO、SSD、RetinaNet）方法。在这些架构中，我们设计的网络可以一次性预测类别和 ROI。

### R-CNN: Region-Based CNN

<!-- [R-CNN](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf) uses [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) to generate hierarchical structure of ROI regions, which are then passed through CNN feature extractors and SVM-classifiers to determine the object class, and linear regression to determine *bounding box* coordinates. [Official Paper](https://arxiv.org/pdf/1506.01497v1.pdf) -->

[R-CNN](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)使用选择性搜索[Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)生成 ROI 区域的层次结构，然后通过 CNN 特征提取器和 SVM 分类器来确定对象类别，并通过线性回归来确定边界框坐标。[Official Paper](https://arxiv.org/pdf/1506.01497v1.pdf)

![RCNN](images/rcnn1.png)

> *Image from van de Sande et al. ICCV’11*

![RCNN-1](images/rcnn2.png)

> *Images from [this blog](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)

### F-RCNN - Fast R-CNN

<!-- This approach is similar to R-CNN, but regions are defined after convolution layers have been applied. -->
这种方法与 R-CNN 类似，但区域是在应用卷积层之后定义的。

![FRCNN](images/f-rcnn.png)

> Image from [the Official Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf), [arXiv](https://arxiv.org/pdf/1504.08083.pdf), 2015

### Faster R-CNN

<!-- The main idea of this approach is to use neural network to predict ROIs - so-called *Region Proposal Network*. [Paper](https://arxiv.org/pdf/1506.01497.pdf), 2016 -->

这种方法的主要思想是使用神经网络来预测ROI——所谓的Region Proposal Network,[论文](https://arxiv.org/pdf/1506.01497.pdf), 2016

![FasterRCNN](images/faster-rcnn.png)

> Image from [the official paper](https://arxiv.org/pdf/1506.01497.pdf)

### R-FCN: Region-Based Fully Convolutional Network

<!-- This algorithm is even faster than Faster R-CNN. The main idea is the following: -->
该算法甚至比 Faster R-CNN 还要快。主要思想如下：

<!-- 1. We extract features using ResNet-101
1. Features are processed by **Position-Sensitive Score Map**. Each object from $C$ classes is divided by $k\times k$ regions, and we are training to predict parts of objects.
1. For each part from $k\times k$ regions all networks vote for object classes, and the object class with maximum vote is selected. -->

1. 我们使用 ResNet-101 提取特征
2. 特征由**Position-Sensitive Score Map**处理。$C$ 类中的每个对象都被 $k\times k$ 个区域划分，我们正在训练预测对象的各个部分。
3. 对于 $k\times k$ 区域中的每个部分，所有网络都对对象类别进行投票，并选择投票数最多的对象类别。

![r-fcn image](images/r-fcn.png)

> Image from [official paper](https://arxiv.org/abs/1605.06409)

### YOLO - You Only Look Once

<!-- YOLO is a realtime one-pass algorithm. The main idea is the following: -->
YOLO 是一种实时单遍算法。主要思想如下：

 <!-- * Image is divided into $S\times S$ regions -->
 <!-- * For each region, **CNN** predicts $n$ possible objects, *bounding box* coordinates and *confidence*=*probability* * IoU. -->

 * 图像被划分为 $S\times S$ 个区域
 * 对于每个区域，**CNN**预测 $n$ 个可能的对象、**边框**坐标和*置信度*=*概率* IoU。

 ![YOLO](images/yolo.png)

> Image from [official paper](https://arxiv.org/abs/1506.02640)

### Other Algorithms

* RetinaNet: [official paper](https://arxiv.org/abs/1708.02002)
   - [PyTorch Implementation in Torchvision](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html)
   - [Keras Implementation](https://github.com/fizyr/keras-retinanet)
   - [Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet/) in Keras Samples
* SSD (Single Shot Detector): [official paper](https://arxiv.org/abs/1512.02325)

## ✍️ Exercises: Object Detection

Continue your learning in the following notebook:

[ObjectDetection.ipynb](ObjectDetection.ipynb)

## Conclusion

In this lesson you took a whirlwind tour of all the various ways that object detection can be accomplished!

## 🚀 Challenge

Read through these articles and notebooks about YOLO and try them for yourself

* [Good blog post](https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/) describing YOLO
 * [Official site](https://pjreddie.com/darknet/yolo/)
 * Yolo: [Keras implementation](https://github.com/experiencor/keras-yolo2), [step-by-step notebook](https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb)
 * Yolo v2: [Keras implementation](https://github.com/experiencor/keras-yolo2), [step-by-step notebook](https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb)

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/211)

## Review & Self Study

* [Object Detection](https://tjmachinelearning.com/lectures/1718/obj/) by Nikhil Sardana
* [A good comparison of object detection algorithms](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)
* [Review of Deep Learning Algorithms for Object Detection](https://medium.com/comet-app/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
* [A Step-by-Step Introduction to the Basic Object Detection Algorithms](https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/)
* [Implementation of Faster R-CNN in Python for Object Detection](https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/)

## [Assignment: Object Detection](lab/README.md)
