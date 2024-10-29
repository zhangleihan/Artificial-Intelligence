<!-- # Introduction to AI -->
# 人工智能简介

![Summary of Introduction of AI content in a doodle](../sketchnotes/ai-intro.png)

> Sketchnote by [Tomomi Imura](https://twitter.com/girlie_mac)

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/101)

<!-- **Artificial Intelligence** is an exciting scientific discipline that studies how we can make computers exhibit intelligent behavior, e.g. do those things that human beings are good at doing. -->

人工智能是一门令人兴奋的科学学科，研究如何让计算机表现出智能行为，例如做人类擅长做的事情。



Originally, computers were invented by [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) to operate on numbers following a well-defined procedure - an algorithm. Modern computers, even though significantly more advanced than the original model proposed in the 19th century, still follow the same idea of controlled computations. Thus it is possible to program a computer to do something if we know the exact sequence of steps that we need to do in order to achieve the goal.
最初，计算机是由查尔斯·巴贝奇[Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage)发明的，用于按照明确定义的程序（即算法）对数字进行运算。现代计算机虽然比 19 世纪提出的原始模型先进得多，但仍然遵循受控计算的相同理念。因此，如果我们知道实现目标所需执行的确切步骤顺序，就可以对计算机进行编程以执行某些操作。

![Photo of a person](images/dsh_age.png)

> Photo by [Vickie Soshnikova](http://twitter.com/vickievalerie)

<!-- > ✅ Defining the age of a person from his or her photograph is a task that cannot be explicitly programmed, because we do not know how we come up with a number inside our head when we do it. -->
> ✅ 从一个人的相片中确定他的年龄是一项无法明确编程的任务，因为我们不知道在做这件事时我们的脑子里是如何得出一个数字的。

---

<!-- There are some tasks, however, that we do not explicitly know how to solve. Consider determining the age of a person from his/her photograph. We somehow learn to do it, because we have seen many examples of people of different age, but we cannot explicitly explain how we do it, nor can we program the computer to do it. This is exactly the kind of task that are of interest to **Artificial Intelligence** (AI for short). -->

然而，有些任务我们并不明确知道如何解决。比如，从一个人的照片中确定他的年龄。我们以某种方式学会了如何做到这一点，因为我们已经见过许多不同年龄的人的例子，但我们无法明确解释我们如何做到这一点，也无法编程让计算机做到这一点。这正是人工智能（简称 AI）感兴趣的任务。

<!-- ✅ Think of some tasks that you could offload to a computer that would benefit from AI. Consider the fields of finance, medicine, and the arts - how are these fields benefiting today from AI? -->

✅ 想一想你可以将哪些任务交给计算机来完成，从而受益于人工智能。考虑一下金融、医学和艺术领域——这些领域如今如何从人工智能中受益？

<!-- ## Weak AI vs. Strong AI -->

## 弱人工智能与强人工智能

<!-- The task of solving a specific human-like problem, such as determining a person's age from a photo, can be called **Weak AI**, because we are creating a system for only one task, and not a system that can solve many tasks, such as can be done by a human being. Of course, developing a generally intelligent computer system is also extremely interesting from many points of view, including for students of the philosophy of consciousness. Such system would be called **Strong AI**, or **[Artificial General Intelligence](https://en.wikipedia.org/wiki/Artificial_general_intelligence)** (AGI). -->

解决特定类人问题（例如根据照片确定一个人的年龄）的任务可以称为弱人工智能，因为我们创建的系统只用于一项任务，而不是可以解决多项任务（例如人类可以完成的任务）的系统。当然，从许多角度来看，开发通用智能计算机系统也非常有趣，包括对于意识哲学的学生来说。这样的系统被称为强人工智能或通用人工智能(**[Artificial General Intelligence](https://en.wikipedia.org/wiki/Artificial_general_intelligence)** (AGI))。

<!-- ## The Definition of Intelligence and the Turing Test -->
## 智能的定义与图灵测试
<!-- One of the problems when dealing with the term **[Intelligence](https://en.wikipedia.org/wiki/Intelligence)** is that there is no clear definition of this term. One can argue that intelligence is connected to **abstract thinking**, or to **self-awareness**, but we cannot properly define it. -->

在处理“智力”这一术语时，一个问题是它没有明确的定义。人们可以争辩说，智力与抽象思维或自我意识有关，但我们无法正确定义它。

![Photo of a Cat](images/photo-cat.jpg)

> [Photo](https://unsplash.com/photos/75715CVEJhI) by [Amber Kipp](https://unsplash.com/@sadmax) from Unsplash

<!-- To see the ambiguity of a term *intelligence*, try answering a question: "Is a cat intelligent?". Different people tend to give different answers to this question, as there is no universally accepted test to prove the assertion is true or not. And if you think there is - try running your cat through an IQ test... -->

要了解术语“智力”的歧义，请尝试回答一个问题：“猫聪明吗？”。不同的人往往会对这个问题给出不同的答案，因为没有普遍接受的测试来证明该断言是否正确。如果您认为有 - 尝试让您的猫接受智商测试...

<!-- ✅ Think for a minute about how you define intelligence. Is a crow who can solve a maze and get at some food intelligent? Is a child intelligent? -->
✅ 想一想你是如何定义智力的。一只能走出迷宫并找到食物的乌鸦是聪明的吗？一个孩子是聪明的吗？

---

<!-- When speaking about AGI we need to have some way to tell if we have created a truly intelligent system. [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) proposed a way called a **[Turing Test](https://en.wikipedia.org/wiki/Turing_test)**, which also acts like a definition of intelligence. The test compares a given system to something inherently intelligent - a real human being, and because any automatic comparison can be bypassed by a computer program, we use a human interrogator. So, if a human being is unable to distinguish between a real person and a computer system in text-based dialogue - the system is considered intelligent. -->
说到 AGI，我们需要某种方法来判断我们是否创建了一个真正智能的系统。艾伦·图灵提出了一种称为图灵测试的方法，它也可以作为智能的定义。该测试将给定系统与某种天生智能的东西（真实的人类）进行比较，由于计算机程序可以绕过任何自动比较，因此我们使用人类询问者。因此，如果人类无法在基于文本的对话中区分真人和计算机系统，则该系统被视为智能。

<!-- > A chat-bot called [Eugene Goostman](https://en.wikipedia.org/wiki/Eugene_Goostman), developed in St.Petersburg, came close to passing the Turing test in 2014 by using a clever personality trick. It announced up front that it was a 13-year old Ukrainian boy, which would explain the lack of knowledge and some discrepancies in the text. The bot convinced 30% of the judges that it was human after a 5 minute dialogue, a metric that Turing believed a machine would be able to pass by 2000. However, one should understand that this does not indicate that we have created an intelligent system, or that a computer system has fooled the human interrogator - the system didn't fool the humans, but rather the bot creators did! -->
2014 年，圣彼得堡开发了一款名为Eugene Goostman的聊天机器人，它通过巧妙的个性技巧，几乎通过了图灵测试。它一开始就宣布自己是一名 13 岁的乌克兰男孩，这可以解释为什么它缺乏知识，以及文本中存在一些差异。经过 5 分钟的对话，该机器人让 30% 的评委​​相信它是人类，图灵预言到 2000 年，机器就能通过这一标准。然而，我们应该明白，这并不表示我们已经创建了一个智能系统，也不表示计算机系统已经欺骗了人类询问者——欺骗人类的不是系统，而是机器人的创造者！

<!-- ✅ Have you ever been fooled by a chat bot into thinking that you are speaking to a human? How did it convince you? -->

✅ 您是否曾被聊天机器人欺骗，以为您正在与人类交谈？它是如何说服您的？

<!-- ## Different Approaches to AI -->
## 人工智能的不同方法
<!-- If we want a computer to behave like a human, we need somehow to model inside a computer our way of thinking. Consequently, we need to try to understand what makes a human being intelligent. -->
如果我们希望计算机的行为像人类，我们需要以某种方式在计算机内部模拟我们的思维方式。因此，我们需要尝试理解是什么让人类变得聪明。

<!-- > To be able to program intelligence into a machine, we need to understand how our own processes of making decisions work. If you do a little self-introspection, you will realize that there are some processes that happen subconsciously – eg. we can distinguish a cat from a dog without thinking about it - while some others involve reasoning. -->
> 为了能够将智能编程到机器中，我们需要了解我们自己的决策过程是如何进行的。如果你进行一些自我反省，你就会意识到有些过程是潜意识中发生的 - 例如，我们可以不假思索地区分猫和狗 - 而其他一些过程则需要推理。

<!-- There are two possible approaches to this problem: -->
有两种方法可以解决此问题：
Top-down Approach (Symbolic Reasoning) | Bottom-up Approach (Neural Networks)
---------------------------------------|-------------------------------------
<!-- A top-down approach models the way a person reasons to solve a problem. It involves extracting **knowledge** from a human being, and representing it in a computer-readable form. We also need to develop a way to model **reasoning** inside a computer. | A bottom-up approach models the structure of a human brain, consisting of huge number of simple units called **neurons**. Each neuron acts like a weighted average of its inputs, and we can train a network of neurons to solve useful problems by providing **training data**. -->

自上而下的方法（符号推理）| 自下而上的方法（神经网络）
自上而下的方法模拟了人类解决问题的推理方式。它涉及从人类那里提取知识，并将其表示为计算机可读的形式。我们还需要开发一种在计算机内部模拟推理的方法。 |  自下而上的方法模拟了人类大脑的结构，大脑由大量称为神经元的简单单元组成。每个神经元就像其输入的加权平均值，我们可以通过提供训练数据来训练神经元网络来解决有用的问题。

还有一些其他可能的智能方法：
涌现法**Emergent**、协同法**Synergetic**或多智能体方法**multi-agent approach**基于这样一个事实：通过大量简单智能体的相互作用可以获得复杂的智能行为。根据进化控制论[evolutionary cybernetics](https://en.wikipedia.org/wiki/Global_brain#Evolutionary_cybernetics)，智能可以从元系统转换过程中更简单、更被动的行为中涌现出来。

<!-- There are also some other possible approaches to intelligence: -->

<!-- * An **Emergent**, **Synergetic** or **multi-agent approach** are based on the fact that complex intelligent behaviour can be obtained by an interaction of a large number of simple agents. According to [evolutionary cybernetics](https://en.wikipedia.org/wiki/Global_brain#Evolutionary_cybernetics), intelligence can *emerge* from more simple, reactive behaviour in the process of *metasystem transition*. -->

<!-- * An **Evolutionary approach**, or **genetic algorithm** is an optimization process based on the principles of evolution. -->
进化方法**Evolutionary approach**或遗传算法**genetic algorithm**是基于进化原理的优化过程。

<!-- We will consider those approaches later in the course, but right now we will focus on two main directions: top-down and bottom-up. -->
我们将在课程的后面考虑这些方法，但现在我们将重点关注两个主要方向：自上而下和自下而上。

<!-- ### The Top-Down Approach -->
### 自上而下的方法

<!-- In a **top-down approach**, we try to model our reasoning.  Because we can follow our thoughts when we reason, we can try to formalize this process and program it inside the computer. This is called **symbolic reasoning**.

People tend to have some rules in their head that guide their decision making processes. For example, when a doctor is diagnosing a patient, he or she may realize that a person has a fever, and thus there might be some inflammation going on inside the body. By applying a large set of rules to a specific problem a doctor may be able to come up with the final diagnosis.

This approach relies heavily on **knowledge representation** and **reasoning**. Extracting knowledge from a human expert might be the most difficult part, because a doctor in many cases would not know exactly why he or she is coming up with a particular diagnosis. Sometimes the solution just comes up in his or her head without explicit thinking. Some tasks, such as determining the age of a person from a photograph, cannot be at all reduced to manipulating knowledge. -->

在自上而下的方法中，我们尝试对推理进行建模。由于我们在推理时可以遵循自己的想法，因此我们可以尝试将这个过程形式化，并在计算机内部进行编程。这称为符号推理。

人们的头脑中往往有一些规则来指导他们的决策过程。例如，当医生在诊断病人时，他或她可能意识到一个人发烧了，因此身体内部可能有炎症。通过将大量规则应用于特定问题，医生可能能够得出最终诊断。

这种方法在很大程度上依赖于知识表示和推理。从人类专家那里提取知识可能是最困难的部分，因为在许多情况下，医生并不知道他或她为什么会得出特定的诊断。有时解决方案只是在他或她的脑海中浮现，而无需明确的思考。有些任务，例如从照片中确定一个人的年龄，根本不能归结为操纵知识。

<!-- ### Bottom-Up Approach -->
### 自下而上的方法

<!-- Alternately, we can try to model the simplest elements inside our brain – a neuron. We can construct a so-called **artificial neural network** inside a computer, and then try to teach it to solve problems by giving it examples. This process is similar to how a newborn child learns about his or her surroundings by making observations. -->
或者，我们可以尝试模拟大脑中最简单的元素——神经元。我们可以在计算机内部构建所谓的人工神经网络，然后尝试通过给它提供示例来教它解决问题。这个过程类似于新生儿通过观察来了解周围环境的方式。

<!-- ✅ Do a little research on how babies learn. What are the basic elements of a baby's brain? -->
✅ 对婴儿的学习方式做一点研究。婴儿大脑的基本元素是什么？

<!-- > | What about ML?         |      |
> |--------------|-----------|
> | Part of Artificial Intelligence that is based on computer learning to solve a problem based on some data is called **Machine Learning**. We will not consider classical machine learning in this course - we refer you to a separate [Machine Learning for Beginners](http://aka.ms/ml-beginners) curriculum. |   ![ML for Beginners](images/ml-for-beginners.png)    | -->

<!-- > 那么 ML 怎么样？
> 人工智能的一部分是基于计算机学习，根据一些数据解决问题，这被称为机器学习。我们不会在本课程中考虑经典机器学习 - 我们会将您引荐到单独的初学者机器学习课程。 -->

<!-- ## A Brief History of AI -->
## 人工智能简史

<!-- Artificial Intelligence was started as a field in the middle of the twentieth century. Initially, symbolic reasoning was a prevalent approach, and it led to a number of important successes, such as expert systems – computer programs that were able to act as an expert in some limited problem domains. However, it soon became clear that such approach does not scale well. Extracting the knowledge from an expert, representing it in a computer, and keeping that knowledgebase accurate turns out to be a very complex task, and too expensive to be practical in many cases. This led to so-called [AI Winter](https://en.wikipedia.org/wiki/AI_winter) in the 1970s. -->
人工智能始于二十世纪中叶。最初，符号推理是一种流行的方法，并取得了许多重大成功，例如专家系统——能够在某些有限问题领域充当专家的计算机程序。然而，很快人们就发现这种方法的可扩展性不佳。从专家那里提取知识，将其表示在计算机中，并保持该知识库的准确性是一项非常复杂的任务，而且在许多情况下成本太高，不切实际。这导致了20 世纪 70 年代所谓的人工智能寒冬。

<img alt="Brief History of AI" src="images/history-of-ai.png" width="70%"/>

> Image by [Dmitry Soshnikov](http://soshnikov.com)

<!-- As time passed, computing resources became cheaper, and more data has become available, so neural network approaches started demonstrating great performance in competing with human beings in many areas, such as computer vision or speech understanding. In the last decade, the term Artificial Intelligence has been mostly used as a synonym for Neural Networks, because most of the AI successes that we hear about are based on them. -->
随着时间的推移，计算资源变得越来越便宜，可用的数据也越来越多，因此神经网络方法开始在计算机视觉或语音理解等许多领域与人类竞争，表现出色。在过去十年中，人工智能一词主要用作神经网络的同义词，因为我们听到的大多数人工智能成功都是基于神经网络的。

<!-- We can observe how the approaches changed, for example, in creating a chess playing computer program: -->

我们可以观察到方法是如何变化的，例如在创建一个下棋的计算机程序时：

* 早期的国际象棋程序基于搜索——程序明确地尝试估计对手在给定的下一步棋数下可能采取的行动，并根据在几步棋内可以达到的最佳位置选择最佳行动。这导致了所谓的alpha-beta 剪枝搜索算法的发展。
* 搜索策略在游戏结束时效果很好，此时搜索空间受限于少数可能的动作。然而，在游戏开始时，搜索空间非常大，可以通过从人类玩家之间的现有比赛中学习来改进算法。后续实验采用了所谓的基于案例的推理，其中程序在知识库中寻找与游戏中当前位置非常相似的案例。
* 战胜人类玩家的现代程序基于神经网络和强化学习，程序通过长时间与自己对弈并从自己的错误中学习来学习下棋——就像人类学习下棋一样。然而，计算机程序可以在更短的时间内玩更多的游戏，因此学习速度更快。
<!-- 
* Early chess programs were based on search – a program explicitly tried to estimate possible moves of an opponent for a given number of next moves, and selected an optimal move based on the optimal position that can be achieved in a few moves. It led to the development of the so-called [alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) search algorithm.
* Search strategies work well toward the end of the game, where the search space is limited by a small number of possible moves. However, at the beginning of the game, the search space is huge, and the algorithm can be improved by learning from existing matches between human players. Subsequent experiments employed so-called [case-based reasoning](https://en.wikipedia.org/wiki/Case-based_reasoning), where the program looked for cases in the knowledge base very similar to the current position in the game.
* Modern programs that win over human players are based on neural networks and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), where the programs learn to play solely by playing a long time against themselves and learning from their own mistakes – much like human beings do when learning to play chess. However, a computer program can play many more games in much less time, and thus can learn much faster. -->

<!-- ✅ Do a little research on other games that have been played by AI. -->
✅ 对 AI 玩过的其他游戏做一点研究。

<!-- Similarly, we can see how the approach towards creating “talking programs” (that might pass the Turing test) changed: -->
类似地，我们可以看到创建“说话的程序”（可能通过图灵测试）的方法是如何改变的：

<!-- * Early programs of this kind such as [Eliza](https://en.wikipedia.org/wiki/ELIZA), were based on very simple grammatical rules and the re-formulation of the input sentence into a question.
* Modern assistants, such as Cortana, Siri or Google Assistant are all hybrid systems that use Neural networks to convert speech into text and recognize our intent, and then employ some reasoning or explicit algorithms to perform required actions.
* In the future, we may expect a complete neural-based model to handle dialogue by itself. The recent GPT and [Turing-NLG](https://turing.microsoft.com/) family of neural networks show great success in this. -->

早期的此类程序，例如Eliza，基于非常简单的语法规则以及将输入句子重新表述为问题。
现代助手，例如 Cortana、Siri 或 Google Assistant 都是混合系统，它们使用神经网络将语音转换为文本并识别我们的意图，然后采用一些推理或明确的算法来执行所需的操作。
未来，我们或许会期待一个完整的基于神经的模型能够自行处理对话。最近的 GPT 和Turing-NLG神经网络系列在这方面取得了巨大成功。

<img alt="the Turing test's evolution" src="images/turing-test-evol.png" width="70%"/>

> Image by Dmitry Soshnikov, [photo](https://unsplash.com/photos/r8LmVbUKgns) by [Marina Abrosimova](https://unsplash.com/@abrosimova_marina_foto), Unsplash

<!-- ## Recent AI Research -->
## 人工智能研究进展

<!-- The huge recent growth in neural network research started around 2010, when large public datasets started to become available. A huge collection of images called [ImageNet](https://en.wikipedia.org/wiki/ImageNet), which contains around 14 million annotated images, gave birth to the [ImageNet Large Scale Visual Recognition Challenge](https://image-net.org/challenges/LSVRC/). -->

神经网络研究的近期快速发展始于 2010 年左右，当时大量公共数据集开始出现。一个名为ImageNet的庞大图像集合包含约 1400 万张带注释的图像，由此催生了ImageNet 大规模视觉识别挑战赛。

![ILSVRC Accuracy](images/ilsvrc.gif)

> Image by [Dmitry Soshnikov](http://soshnikov.com)

<!-- In 2012, [Convolutional Neural Networks](../4-ComputerVision/07-ConvNets/README.md) were first used in image classification, which led to a significant drop in classification errors (from almost 30% to 16.4%). In 2015, ResNet architecture from Microsoft Research [achieved human-level accuracy](https://doi.org/10.1109/ICCV.2015.123).

Since then, Neural Networks demonstrated very successful behaviour in many tasks: -->

2012 年，卷积神经网络首次应用于图像分类，分类错误率大幅下降（从近 30% 降至 16.4%）。2015 年，微软研究院的 ResNet 架构实现了与人类水平相当的准确率。

从那时起，神经网络在许多任务中表现出非常成功的行为：

---

Year | Human Parity achieved
-----|--------
2015 | [Image Classification](https://doi.org/10.1109/ICCV.2015.123)
2016 | [Conversational Speech Recognition](https://arxiv.org/abs/1610.05256)
2018 | [Automatic Machine Translation](https://arxiv.org/abs/1803.05567) (Chinese-to-English)
2020 | [Image Captioning](https://arxiv.org/abs/2009.13682)

<!-- Over the past few years we have witnessed huge successes with large language models, such as BERT and GPT-3. This happened mostly due to the fact that there is a lot of general text data available that allows us to train models to capture the structure and meaning of texts, pre-train them on general text collections, and then specialize those models for more specific tasks. We will learn more about [Natural Language Processing](../5-NLP/README.md) later in this course. -->

过去几年，我们见证了大型语言模型（如 BERT 和 GPT-3）的巨大成功。这主要是因为有大量可用的通用文本数据，使我们能够训练模型以捕获文本的结构和含义，在通用文本集合上对它们进行预训练，然后将这些模型专门用于更具体的任务。我们将在本课程的后面部分了解有关自然语言处理的更多信息。

## 🚀 Challenge


<!-- Do a tour of the internet to determine where, in your opinion, AI is most effectively used. Is it in a Mapping app, or some speech-to-text service or a video game? Research how the system was built. -->
浏览一下互联网，看看你认为人工智能最有效的应用领域在哪里。是地图应用、语音转文本服务还是视频游戏？研究一下这个系统是如何构建的。

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/201)

## Review & Self Study

<!-- Review the history of AI and ML by reading through [this lesson](https://github.com/microsoft/ML-For-Beginners/tree/main/1-Introduction/2-history-of-ML). Take an element from the sketchnote at the top of that lesson or this one and research it in more depth to understand the cultural context informing its evolution. -->

通过阅读本课，回顾 AI 和 ML 的历史。从该课或本课顶部的速写笔记中选取一个元素，并对其进行更深入的研究，以了解其演变的文化背景。

**Assignment**: [Game Jam](assignment.md)
