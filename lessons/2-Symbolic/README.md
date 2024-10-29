# Knowledge Representation and Expert Systems

![Summary of Symbolic AI content](../sketchnotes/ai-symbolic.png)

> Sketchnote by [Tomomi Imura](https://twitter.com/girlie_mac)

<!-- The quest for artificial intelligence is based on a search for knowledge, to make sense of the world similar to how humans do. But how can you go about doing this? -->
对人工智能的追求是基于对知识的探索，以类似于人类的方式来理解世界。但你怎样才能做到这一点呢？

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/102)

<!-- In the early days of AI, the top-down approach to creating intelligent systems (discussed in the previous lesson) was popular. The idea was to extract the knowledge from people into some machine-readable form, and then use it to automatically solve problems. This approach was based on two big ideas: -->

在人工智能的早期，创建智能系统的自上而下的方法（在上一课中讨论）很流行。这个想法是将人们的知识提取成某种机器可读的形式，然后用它来自动解决问题。这种方法基于两个重要想法：

<!-- * Knowledge Representation
* Reasoning -->

*知识表示
*推理

## Knowledge Representation

<!-- One of the important concepts in Symbolic AI is **knowledge**. It is important to differentiate knowledge from *information* or *data*. For example, one can say that books contain knowledge, because one can study books and become an expert. However, what books contain is actually called *data*, and by reading books and integrating this data into our world model we convert this data to knowledge. -->

符号人工智能中的重要概念之一是知识。区分知识与信息或数据非常重要。例如，我们可以说书籍包含知识，因为我们可以学习书籍并成为专家。然而，书籍所包含的内容实际上称为数据，通过阅读书籍并将这些数据集成到我们的世界模型中，我们将这些数据转换为知识。

<!-- > ✅ **Knowledge** is something which is contained in our head and represents our understanding of the world. It is obtained by an active **learning** process, which integrates pieces of information that we receive into our active model of the world. -->

> ✅**知识是包含**在我们头脑中的东西，代表我们对世界的理解。它是通过**主动学习**过程获得的，该过程将我们收到的信息整合到我们对于世界的描述模型中。

<!-- Most often, we do not strictly define knowledge, but we align it with other related concepts using [DIKW Pyramid](https://en.wikipedia.org/wiki/DIKW_pyramid). It contains the following concepts:

* **Data** is something represented in physical media, such as written text or spoken words. Data exists independently of human beings and can be passed between people.
* **Information** is how we interpret data in our head. For example, when we hear the word *computer*, we have some understanding of what it is.
* **Knowledge** is information being integrated into our world model. For example, once we learn what a computer is, we start having some ideas about how it works, how much it costs, and what it can be used for. This network of interrelated concepts forms our knowledge.
* **Wisdom** is yet one more level of our understanding of the world, and it represents *meta-knowledge*, eg. some notion on how and when the knowledge should be used. -->

大多数情况下，我们并不严格定义知识，而是使用[DIKW Pyramid](https://en.wikipedia.org/wiki/DIKW_pyramid)将其与其他相关概念对齐。它包含以下概念：

* **数据**是用物理媒体表示的东西，例如书面文本或口头语言。数据独立于人类而存在，可以在人与人之间传递。
* **信息**是我们在头脑中解释数据的方式。例如，当我们听到计算机这个词时，我们对它是什么有一定的了解。
* **知识**是被整合到我们的世界模型中的信息。例如，一旦我们了解了计算机是什么，我们就开始对它的工作原理、它的成本以及它的用途有一些想法。这个相互关联的概念网络构成了我们的知识。
* **智慧**是我们对世界的理解的又一个层次，它代表元知识，例如。关于如何以及何时使用知识的一些概念。

<img src="images/DIKW_Pyramid.png" width="30%"/>

*Image [from Wikipedia](https://commons.wikimedia.org/w/index.php?curid=37705247), By Longlivetheux - Own work, CC BY-SA 4.0*

<!-- Thus, the problem of **knowledge representation** is to find some effective way to represent knowledge inside a computer in the form of data, to make it automatically usable. This can be seen as a spectrum: -->

因此，知识表示的问题就是找到某种有效的方法，将计算机内部的知识以数据的形式表示出来，使其自动可用。这可以看作是一个频谱：

![Knowledge representation spectrum](images/knowledge-spectrum.png)

> Image by [Dmitry Soshnikov](http://soshnikov.com)

<!-- * On the left, there are very simple types of knowledge representations that can be effectively used by computers. The simplest one is algorithmic, when knowledge is represented by a computer program. This, however, is not the best way to represent knowledge, because it is not flexible. Knowledge inside our head is often non-algorithmic.
* On the right, there are representations such as natural text. It is the most powerful, but cannot be used for automatic reasoning. -->

* 左边是计算机可以有效使用的非常简单的知识表示类型。最简单的一种是算法，即知识由计算机程序表示。然而，这并不是表示知识的最佳方式，因为它不灵活。我们头脑中的知识通常是非算法的。
* 右侧有自然文本等表示。它是最强大的，但不能用于自动推理。


<!-- > ✅ Think for a minute about how you represent knowledge in your head and convert it to notes. Is there a particular format that works well for you to aid in retention? -->

> ✅ 想一想如何在头脑中表达知识并将其转换为笔记。有没有一种特别适合您的格式来帮助您保留？

## Classifying Computer Knowledge Representations

<!-- We can classify different computer knowledge representation methods in the following categories: -->

我们可以将不同的计算机知识表示方法分为以下几类：

<!-- * **Network representations** are based on the fact that we have a network of interrelated concepts inside our head. We can try to reproduce the same networks as a graph inside a computer - a so-called **semantic network**. -->

* **网络表征**基于这样一个事实：我们的头脑中有一个由相互关联的概念组成的网络。我们可以尝试在计算机内重现与图形相同的网络 - 所谓的**语义网络**。


<!-- 1. **Object-Attribute-Value triplets** or **attribute-value pairs**. Since a graph can be represented inside a computer as a list of nodes and edges, we can represent a semantic network by a list of triplets, containing objects, attributes, and values. For example, we build the following triplets about programming languages: -->

1. 对象-属性-值三元组或属性-值对。由于图可以在计算机内部表示为节点和边的列表，因此我们可以通过包含对象、属性和值的三元组列表来表示语义网络。例如，我们构建以下关于编程语言的三元组：

Object | Attribute | Value
-------|-----------|------
Python | is | Untyped-Language
Python | invented-by | Guido van Rossum
Python | block-syntax | indentation
Untyped-Language | doesn't have | type definitions

<!-- > ✅ Think how triplets can be used to represent other types of knowledge. -->

> ✅ 思考如何使用三元组来表示其他类型的知识。

<!-- 2. **Hierarchical representations** emphasize the fact that we often create a hierarchy of objects inside our head. For example, we know that canary is a bird, and all birds have wings. We also have some idea about what colour canary usually is, and what is their flight speed. -->

2. **层次表示**强调了这样一个事实：我们经常在头脑中创建对象的层次结构。例如，我们知道金丝雀是一种鸟，所有的鸟都有翅膀。我们还了解金丝雀通常是什么颜色以及它们的飞行速度是多少。

   <!-- - **Frame representation** is based on representing each object or class of objects as a **frame** which contains **slots**. Slots have possible default values, value restrictions, or stored procedures that can be called to obtain the value of a slot. All frames form a hierarchy similar to an object hierarchy in object-oriented programming languages.
   - **Scenarios** are special kind of frames that represent complex situations that can unfold in time. -->

   - **帧（框架）表示**基于将每个对象或对象类表示为包含槽的帧。槽具有可能的默认值、值限制或可调用以获取槽的值的存储过程。所有帧形成类似于面向对象编程语言中的对象层次结构的层次结构。
   - **场景**是一种特殊的帧（框架），代表可以及时展开的复杂情况。

**Python**

Slot | Value | Default value | Interval |
-----|-------|---------------|----------|
Name | Python | | |
Is-A | Untyped-Language | | |
Variable Case | | CamelCase | |
Program Length | | | 5-5000 lines |
Block Syntax | Indent | | |

<!-- 3. **Procedural representations** are based on representing knowledge by a list of actions that can be executed when a certain condition occurs.
   - Production rules are if-then statements that allow us to draw conclusions. For example, a doctor can have a rule saying that **IF** a patient has high fever **OR** high level of C-reactive protein in blood test **THEN** he has an inflammation. Once we encounter one of the conditions, we can make a conclusion about inflammation, and then use it in further reasoning.
   - Algorithms can be considered another form of procedural representation, although they are almost never used directly in knowledge-based systems. -->

3. **过程表示**基于通过在发生特定条件时可以执行的一系列动作来表示知识。

   - 产生式规则是让我们得出结论的 if-then 语句。例如，医生可以制定一条规则，规定 **IF** 患者发高烧 **OR** 血液检查中C反应蛋白水平较高，**then**说明他患有炎症。一旦我们遇到其中一种情况，我们就可以得出关于炎症的结论，然后用它来进一步推理。
   - 算法可以被认为是过程表示的另一种形式，尽管它们几乎从未直接在基于知识的系统中使用。

<!-- 4. **Logic** was originally proposed by Aristotle as a way to represent universal human knowledge.
   - Predicate Logic as a mathematical theory is too rich to be computable, therefore some subset of it is normally used, such as Horn clauses used in Prolog.
   - Descriptive Logic is a family of logical systems used to represent and reason about hierarchies of objects distributed knowledge representations such as *semantic web*. -->

4. 逻辑最初由亚里士多德提出，作为表示人类普遍知识的一种方式。

   - 谓词逻辑作为一种数学理论过于丰富而难以计算，因此通常使用它的一些子集，例如Prolog中使用的Horn子句。
   - 描述逻辑是一系列逻辑系统，用于表示和推理对象分布式知识表示（例如**语义网**）的层次结构。

## Expert Systems

<!-- One of the early successes of symbolic AI were so-called **expert systems** - computer systems that were designed to act as an expert in some limited problem domain. They were based on a **knowledge base** extracted from one or more human experts, and they contained an **inference engine** that performed some reasoning on top of it. -->

符号人工智能的早期成功之一是所谓的专家系统——旨在充当某些有限问题领域的专家的计算机系统。它们基于从一名或多名人类专家提取的知识库，并且包含一个推理引擎，可以在此基础上执行一些推理。

![Human Architecture](images/arch-human.png) | ![Knowledge-Based System](images/arch-kbs.png)
---------------------------------------------|------------------------------------------------
Simplified structure of a human neural system | Architecture of a knowledge-based system

<!-- Expert systems are built like the human reasoning system, which contains **short-term memory** and **long-term memory**. Similarly, in knowledge-based systems we distinguish the following components:

* **Problem memory**: contains the knowledge about the problem being currently solved, i.e. the temperature or blood pressure of a patient, whether he has inflammation or not, etc. This knowledge is also called **static knowledge**, because it contains a snapshot of what we currently know about the problem - the so-called *problem state*.
* **Knowledge base**: represents long-term knowledge about a problem domain. It is extracted manually from human experts, and does not change from consultation to consultation. Because it allows us to navigate from one problem state to another, it is also called **dynamic knowledge**.
* **Inference engine**: orchestrates the whole process of searching in the problem state space, asking questions of the user when necessary. It is also responsible for finding the right rules to be applied to each state. -->

专家系统的构建就像人类推理系统一样，包含短期记忆和长期记忆。同样，在基于知识的系统中，我们区分以下组件：

**问题记忆**：包含当前正在解决的问题的知识，即患者的体温或血压、是否有炎症等。这种知识也称为静态知识，因为它包含了我们当前所知道的快照关于问题——所谓问题状态。
**知识库**：代表有关问题领域的长期知识。它是从人类专家那里手动提取的，并且不会因咨询而改变。因为它允许我们从一种问题状态导航到另一种问题状态，所以也称为动态知识。
**推理引擎**：协调在问题状态空间中搜索的整个过程，并在必要时向用户提出问题。它还负责寻找适用于每个州的正确规则。

<!-- As an example, let's consider the following expert system of determining an animal based on its physical characteristics: -->
作为一个例子，我们考虑以下根据动物的物理特征来确定动物的专家系统：

![AND-OR Tree](images/AND-OR-Tree.png)

> Image by [Dmitry Soshnikov](http://soshnikov.com)

<!-- This diagram is called an **AND-OR tree**, and it is a graphical representation of a set of production rules. Drawing a tree is useful at the beginning of extracting knowledge from the expert. To represent the knowledge inside the computer it is more convenient to use rules: -->

该图称为**AND-OR树**，它是一组产生式规则的图形表示。在开始从专家那里提取知识时，绘制树很有用。为了表示计算机内部的知识，使用规则更方便：

```
IF the animal eats meat
OR (animal has sharp teeth
    AND animal has claws
    AND animal has forward-looking eyes
) 
THEN the animal is a carnivore
```

<!-- You can notice that each condition on the left-hand-side of the rule and the action are essentially object-attribute-value (OAV) triplets. **Working memory** contains the set of OAV triplets that correspond to the problem currently being solved. A **rules engine** looks for rules for which a condition is satisfied and applies them, adding another triplet to the working memory. -->

您可以注意到，规则左侧的每个条件和操作本质上都是对象-属性-值 (OAV) 三元组。**工作记忆**包含与当前正在解决的问题相对应的 OAV 三元组集合。**规则引擎**查找满足条件的规则并应用它们，将另一个三元组添加到工作内存中。

> ✅ Write your own AND-OR tree on a topic you like!

### Forward vs. Backward Inference

<!-- The process described above is called **forward inference**. It starts with some initial data about the problem available in the working memory, and then executes the following reasoning loop: -->
上述过程称为前向推理。它从工作内存中可用的有关问题的一些初始数据开始，然后执行以下推理循环：

1. 如果目标属性存在于工作记忆中 - 停止并给出结果
2. 查找当前满足条件的所有规则 - 获取冲突规则集。
3. 执行冲突解决- 选择将在此步骤中执行的一个规则。可能有不同的冲突解决策略：
   - 选择知识库中第一个适用的规则
   - 选择随机规则
   - 选择更具体的规则，即满足“左侧”(LHS) 中最多条件的规则
4. 应用选定的规则并将新的知识插入到问题状态中
5. 从步骤 1 开始重复。

<!-- 1. If the target attribute is present in the working memory - stop and give the result
2. Look for all the rules whose condition is currently satisfied - obtain **conflict set** of rules.
3. Perform **conflict resolution** - select one rule that will be executed on this step. There could be different conflict resolution strategies:
   - Select the first applicable rule in the knowledge base
   - Select a random rule
   - Select a *more specific* rule, i.e. the one meeting the most conditions in the "left-hand-side" (LHS)
4. Apply selected rule and insert new piece of knowledge into the problem state
5. Repeat from step 1. -->

<!-- However, in some cases we might want to start with an empty knowledge about the problem, and ask questions that will help us arrive to the conclusion. For example, when doing medical diagnosis, we usually do not perform all medical analyses in advance before starting diagnosing the patient. We rather want to perform analyses when a decision needs to be made. -->

然而，在某些情况下，我们可能想从对问题的空洞知识开始，并提出有助于我们得出结论的问题。例如，在进行医学诊断时，我们通常不会在开始诊断患者之前提前进行所有医学分析。我们宁愿在需要做出决定时进行分析。

<!-- This process can be modeled using **backward inference**. It is driven by the **goal** - the attribute value that we are looking to find:

1. Select all rules that can give us the value of a goal (i.e. with the goal on the RHS ("right-hand-side")) - a conflict set
1. If there are no rules for this attribute, or there is a rule saying that we should ask the value from the user - ask for it, otherwise:
1. Use conflict resolution strategy to select one rule that we will use as *hypothesis* - we will try to prove it
1. Recurrently repeat the process for all attributes in the LHS of the rule, trying to prove them as goals
1. If at any point the process fails - use another rule at step 3. -->

这个过程可以使用后向推理来建模。它是由目标驱动的- 我们正在寻找的属性值：

1. 选择可以为我们提供目标值的所有规则（即目标位于 RHS（“右侧”））- 冲突集
2. 如果这个属性没有规则，或者有规则说我们应该向用户询问值 - 询问它，否则：
3. 使用冲突解决策略选择一个我们将用作假设的规则- 我们将尝试证明它
4. 对规则 LHS 中的所有属性反复重复该过程，尝试将它们证明为目标
5. 如果该过程在任何时候失败 - 在步骤 3 中使用另一个规则。

<!-- > ✅ In which situations is forward inference more appropriate? How about backward inference? -->

> ✅ 在什么情况下前向推理更合适？向后推理怎么样？

### Implementing Expert Systems

<!-- Expert systems can be implemented using different tools:

* Programming them directly in some high level programming language. This is not the best idea, because the main advantage of a knowledge-based system is that knowledge is separated from inference, and potentially a problem domain expert should be able to write rules without understanding the details of the inference process
* Using **expert systems shell**, i.e. a system specifically designed to be populated by knowledge using some knowledge representation language. -->

专家系统可以使用不同的工具来实现：

* 直接用某种高级编程语言对它们进行编程。这不是最好的想法，因为基于知识的系统的主要优点是知识与推理分离，并且潜在的问题领域专家应该能够在不了解推理过程细节的情况下编写规则
* 使用专家系统外壳，即专门设计为使用某种知识表示语言填充知识的系统。


## ✍️ Exercise: Animal Inference

<!-- See [Animals.ipynb](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/Animals.ipynb) for an example of implementing forward and backward inference expert system. -->

有关实现前向和后向推理专家系统的示例，请参考[Animals.ipynb](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/Animals.ipynb)。

<!-- > **Note**: This example is rather simple, and only gives the idea of how an expert system looks like. Once you start creating such a system, you will only notice some *intelligent* behaviour from it once you reach certain number of rules, around 200+. At some point, rules become too complex to keep all of them in mind, and at this point you may start wondering why a system makes certain decisions. However, the important characteristics of knowledge-based systems is that you can always *explain* exactly how any of the decisions were made. -->

> **注意**：这个例子相当简单，仅给出了专家系统的思路。一旦开始创建这样的系统，只有达到一定数量的规则（大约 200 条以上）时，你才会注意到其中的一些智能行为。在某些时候，规则变得太复杂而无法记住所有规则，此时你可能会开始想知道为什么系统会做出某些决策。然而，基于知识的系统的重要特征是你始终可以准确地解释任何决策是如何做出的。

## Ontologies and the Semantic Web

<!-- At the end of 20th century there was an initiative to use knowledge representation to annotate Internet resources, so that it would be possible to find resources that correspond to very specific queries. This motion was called **Semantic Web**, and it relied on several concepts:

- A special knowledge representation based on **[description logics](https://en.wikipedia.org/wiki/Description_logic)** (DL). It is similar to frame knowledge representation, because it builds a hierarchy of objects with properties, but it has formal logical semantics and inference. There is a whole family of DLs which balance between expressiveness and algorithmic complexity of inference.
- Distributed knowledge representation, where all concepts are represented by a global URI identifier, making it possible to create knowledge hierarchies that span the internet.
- A family of XML-based languages for knowledge description: RDF (Resource Description Framework), RDFS (RDF Schema), OWL (Ontology Web Language). -->

20世纪末出现了使用知识表示来注释互联网资源的倡议，以便可以找到与非常具体的查询相对应的资源。该动议被称为语义网，它依赖于几个概念：

- 基于描述逻辑（DL）的特殊知识表示。它类似于框架知识表示，因为它构建了具有属性的对象的层次结构，但它具有形式的逻辑语义和推理。有一个完整的深度学习家族可以在推理的表达性和算法复杂性之间取得平衡。
- 分布式知识表示，其中所有概念都由全局 URI 标识符表示，从而可以创建跨越互联网的知识层次结构。
- 基于 XML 的知识描述语言家族：RDF（资源描述框架）、RDFS（RDF 模式）、OWL（本体 Web 语言）。


<!-- A core concept in the Semantic Web is a concept of **Ontology**. It refers to a explicit specification of a problem domain using some formal knowledge representation. The simplest ontology can be just a hierarchy of objects in a problem domain, but more complex ontologies will include rules that can be used for inference.

In the semantic web, all representations are based on triplets. Each object and each relation are uniquely identified by the URI. For example, if we want to state the fact that this AI Curriculum has been developed by Dmitry Soshnikov on Jan 1st, 2022 - here are the triplets we can use: -->

语义网中的一个核心概念是本体（Ontology）的概念。它是指使用某种形式的知识表示对问题域进行显式规范。最简单的本体可以只是问题域中对象的层次结构，但更复杂的本体将包括可用于推理的规则。

在语义网中，所有表示都基于三元组。每个对象和每个关系都由 URI 唯一标识。例如，如果我们想要声明这个 AI 课程是由 Dmitry Soshnikov 于 2022 年 1 月 1 日开发的事实 - 以下是我们可以使用的三元组：

<img src="images/triplet.png" width="30%"/>

```
http://github.com/microsoft/ai-for-beginners http://www.example.com/terms/creation-date “Jan 13, 2007”
http://github.com/microsoft/ai-for-beginners http://purl.org/dc/elements/1.1/creator http://soshnikov.com
```

<!-- > ✅ Here `http://www.example.com/terms/creation-date` and `http://purl.org/dc/elements/1.1/creator` are some well-known and universally accepted URIs to express the concepts of *creator* and *creation date*. -->

> ✅`http://www.example.com/terms/creation-date` and `http://purl.org/dc/elements/1.1/creator`是一些众所周知且普遍接受的 URI，用于表达创建者和创建日期的概念。

<!-- In a more complex case, if we want to define a list of creators, we can use some data structures defined in RDF. -->
在更复杂的情况下，如果我们想要定义创建者列表，我们可以使用 RDF 中定义的一些数据结构。

<img src="images/triplet-complex.png" width="40%"/>

> Diagrams above by [Dmitry Soshnikov](http://soshnikov.com)

<!-- The progress of building the Semantic Web was somehow slowed down by the success of search engines and natural language processing techniques, which allow extracting structured data from text. However, in some areas there are still significant efforts to maintain ontologies and knowledge bases. A few projects worth noting: -->

搜索引擎和自然语言处理技术的成功在某种程度上减缓了语义网的建设进程，这些技术允许从文本中提取结构化数据。然而，在某些领域，仍然需要付出巨大努力来维护本体和知识库。几个值得注意的项目：

<!-- * [WikiData](https://wikidata.org/) is a collection of machine readable knowledge bases associated with Wikipedia. Most of the data is mined from Wikipedia *InfoBoxes*, pieces of structured content inside Wikipedia pages. You can [query](https://query.wikidata.org/) wikidata in SPARQL, a special query language for Semantic Web. Here is a sample query that displays most popular eye colors among humans: -->

* [WikiData](https://wikidata.org/)是与维基百科相关的机器可读知识库的集合。大多数数据是从维基百科信息框（维基百科页面内的结构化内容）中挖掘的。您可以使用 SPARQL（一种语义 Web 的特殊查询语言）[查询](https://query.wikidata.org/)wiki 数据。以下是显示人类中最流行的眼睛颜色的示例查询：

```sparql
#defaultView:BubbleChart
SELECT ?eyeColorLabel (COUNT(?human) AS ?count)
WHERE
{
  ?human wdt:P31 wd:Q5.       # human instance-of homo sapiens
  ?human wdt:P1340 ?eyeColor. # human eye-color ?eyeColor
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?eyeColorLabel
```

* [DBpedia](https://www.dbpedia.org/) is another effort similar to WikiData.

* [DBpedia](https://www.dbpedia.org/)是另一个类似于 WikiData 的项目。

<!-- > ✅ If you want to experiment with building your own ontologies, or opening existing ones, there is a great visual ontology editor called [Protégé](https://protege.stanford.edu/). Download it, or use it online. -->

>✅ 如果您想尝试构建自己的本体，或打开现有的本体，有一个很棒的可视化本体编辑器，名为[Protégé](https://protege.stanford.edu/)。下载它，或在线使用。

<img src="images/protege.png" width="70%"/>

*Web Protégé editor open with the Romanov Family ontology. Screenshot by Dmitry Soshnikov*


## ✍️ Exercise: A Family Ontology


<!-- See [FamilyOntology.ipynb](https://github.com/Ezana135/AI-For-Beginners/blob/main/lessons/2-Symbolic/FamilyOntology.ipynb) for an example of using Semantic Web techniques to reason about family relationships. We will take a family tree represented in common GEDCOM format and an ontology of family relationships and build a graph of all family relationships for given set of individuals. -->

有关使用语义 Web 技术来推理家庭关系的示例，请参阅[FamilyOntology.ipynb](https://github.com/Ezana135/AI-For-Beginners/blob/main/lessons/2-Symbolic/FamilyOntology.ipynb)。我们将采用通用 GEDCOM 格式表示的家谱和家庭关系本体，并为给定的一组个人构建所有家庭关系的图表。

## Microsoft Concept Graph

<!-- In most of the cases, ontologies are carefully created by hand. However, it is also possible to **mine** ontologies from unstructured data, for example, from natural language texts.

One such attempt was done by Microsoft Research, and resulted in [Microsoft Concept Graph](https://blogs.microsoft.com/ai/microsoft-researchers-release-graph-that-helps-machines-conceptualize/?WT.mc_id=academic-77998-cacaste).

It is a large collection of entities grouped together using `is-a` inheritance relationship. It allows answering questions like "What is Microsoft?" - the answer being something like "a company with probability 0.87, and a brand with probability 0.75".

The Graph is available either as REST API, or as a large downloadable text file that lists all entity pairs. -->

在大多数情况下，本体是手工精心创建的。然而，也可以从非结构化数据（例如自然语言文本）中挖掘本体。

微软研究院就进行了一项这样的尝试，并产生了[Microsoft Concept Graph](https://blogs.microsoft.com/ai/microsoft-researchers-release-graph-that-helps-machines-conceptualize/?WT.mc_id=academic-77998-cacaste)。

它是使用继承关系分组在一起的实体的大集合is-a。它可以回答诸如“微软是什么？”之类的问题。- 答案类似于“概率为 0.87 的公司，概率为 0.75 的品牌”。
该图可以作为 REST API 提供，也可以作为列出所有实体对的大型可下载文本文件提供。

## ✍️ Exercise: A Concept Graph

<!-- Try the [MSConceptGraph.ipynb](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/MSConceptGraph.ipynb) notebook to see how we can use Microsoft Concept Graph to group news articles into several categories. -->

尝试[MSConceptGraph.ipynb](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/MSConceptGraph.ipynb)笔记本，了解如何使用 Microsoft Concept Graph 将新闻文章分为几个类别。

## Conclusion

<!-- Nowadays, AI is often considered to be a synonym for *Machine Learning* or *Neural Networks*. However, a human being also exhibits explicit reasoning, which is something currently not being handled by neural networks. In real world projects, explicit reasoning is still used to perform tasks that require explanations, or being able to modify the behavior of the system in a controlled way. -->

如今，人工智能通常被认为是机器学习或神经网络的同义词。然而，人类也表现出明确的推理能力，这是神经网络目前无法处理的。在现实世界的项目中，显式推理仍然用于执行需要解释的任务，或者能够以受控的方式修改系统的行为。

## 🚀 Challenge

<!-- In the Family Ontology notebook associated to this lesson, there is an opportunity to experiment with other family relations. Try to discover new connections between people in the family tree. -->

在与本课程相关的家庭本体笔记本中，有机会尝试其他家庭关系。尝试发现家谱中的人之间的新联系。



## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/202)

## Review & Self Study

<!-- Do some research on the internet to discover areas where humans have tried to quantify and codify knowledge. Take a look at Bloom's Taxonomy, and go back in history to learn how humans tried to make sense of their world. Explore the work of Linnaeus to create a taxonomy of organisms, and observe the way Dmitri Mendeleev created a way for chemical elements to be described and grouped. What other interesting examples can you find? -->

在互联网上进行一些研究，以发现人类试图量化和整理知识的领域。看看布鲁姆的分类法，回顾历史，了解人类如何试图理解他们的世界。探索林奈创建生物分类学的工作，并观察德米特里·门捷列夫创建化学元素描述和分组方式的方式。您还能找到哪些其他有趣的例子？

**Assignment**: [Build an Ontology](assignment.md)
