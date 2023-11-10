# Multi-Class Classification with Perceptron

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

## Task

<!-- Using the code we have developed in this lesson for binary classification of MNIST handwritten digits, create a multi-class classified that would be able to recognize any digit. Compute the classification accuracy on the train and test dataset, and print out the confusion matrix. -->
使用我们在本课程中开发的用于 MNIST 手写数字二进制分类的代码，创建一个能够识别任何数字的多类分类。计算训练和测试数据集的分类精度，并打印出混淆矩阵。

## Hints

<!-- 1. For each digit, create a dataset for binary classifier of "this digit vs. all other digits" -->
<!-- 1. Train 10 different perceptrons for binary classification (one for each digit) -->
<!-- 1. Define a function that will classify an input digit -->

1. 对于每个数字，为​​“该数字与所有其他数字”的二元分类器创建一个数据集
2. 训练 10 个不同的感知机进行二元分类（每个数字一个）
3. 定义一个对输入数字进行分类的函数

<!-- > **Hint**: If we combine weights of all 10 perceptrons into one matrix, we should be able to apply all 10 perceptrons to the input digits by one matrix multiplication. Most probable digit can then be found just by applying `argmax` operation on the output. -->

> 提示：如果我们将所有 10 个感知机的权重合并到一个矩阵中，我们应该能够通过一次矩阵乘法将所有 10 个感知机应用于输入数字。argmax然后只需对输出进行运算即可找到最可能的数字。

## Stating Notebook

Start the lab by opening [PerceptronMultiClass.ipynb](PerceptronMultiClass.ipynb)
