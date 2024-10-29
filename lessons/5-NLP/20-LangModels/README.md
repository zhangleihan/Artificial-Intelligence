# Pre-Trained Large Language Models

<!-- In all of our previous tasks, we were training a neural network to perform a certain task using labeled dataset. With large transformer models, such as BERT, we use language modelling in self-supervised fashion to build a language model, which is then specialized for specific downstream task with further domain-specific training. However, it has been demonstrated that large language models can also solve many tasks without ANY domain-specific training. A family of models capable of doing that is called **GPT**: Generative Pre-Trained Transformer. -->

在我们之前的所有任务中，我们正在训练神经网络以使用标记数据集执行特定任务。对于大型 Transformer 模型，例如 BERT，我们以自监督的方式使用语言建模来构建语言模型，然后通过进一步的特定领域训练专门用于特定的下游任务。然而，事实证明，大型语言模型也可以解决许多任务，而无需任何特定领域的训练。能够做到这一点的一系列模型称为**GPT**：生成式预训练变压器。

## [Pre-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/120)

## Text Generation and Perplexity

<!-- The idea of a neural network being able to do general tasks without downstream training is presented in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) paper. The main idea is the many other tasks can be modeled using **text generation**, because understanding text essentially means being able to produce it. Because the model is trained on a huge amount of text that encompasses human knowledge, it also becomes knowledgeable about wide variety of subjects. -->

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)论文中提出了神经网络无需下游训练即可完成一般任务的想法。主要思想是许多其他任务可以使用文本生成来建模，因为理解文本本质上意味着能够生成文本。由于该模型是根据包含人类知识的大量文本进行训练的，因此它也变得了解各种主题。

<!-- > Understanding and being able to produce text also entails knowing something about the world around us. People  also learn by reading to the large extent, and GPT network is similar in this respect. -->

> 理解并能够生成文本还需要了解我们周围的世界。人们很大程度上也是通过阅读来学习的，GPT网络在这方面也是类似的。

<!-- Text generation networks work by predicting probability of the next word $$P(w_N)$$ However, unconditional probability of the next word equals to the frequency of the this word in the text corpus. GPT is able to give us **conditional probability** of the next word, given the previous ones: $$P(w_N | w_{n-1}, ..., w_0)$$ -->

文本生成网络通过预测下一个单词 $$P(w_N)$$ 的概率来工作，但是，下一个单词的无条件概率等于该单词在文本语料库中的频率。GPT 能够给我们下一个单词的**conditional probability**条件概率，给定前面的单词：$$P(w_N | w_{n-1}, ..., w_0)$$

> You can read more about probabilities in our [Data Science for Beginers Curriculum](https://github.com/microsoft/Data-Science-For-Beginners/tree/main/1-Introduction/04-stats-and-probability)

<!-- Quality of language generating model can be defined using **perplexity**. It is intrinsic metric that allows us to measure the model quality without any task-specific dataset. It is based on the notion of *probability of a sentence* - the model assigns high probability to a sentence that is likely to be real (i.e. the model is not **perplexed** by it), and low probability to sentences that make less sense (eg. *Can it does what?*). When we give our model sentences from real text corpus, we would expect them to have high probability, and low **perplexity**. Mathematically, it is defined as normalized inverse probability of the test set: -->

语言生成模型的质量可以使用**perplexity**困惑度来定义。它是内在的度量，允许我们在没有任何特定于任务的数据集的情况下测量模型质量。它基于句子概率的概念- 模型将高概率分配给可能是真实的句子（即模型不会被它困惑），并将低概率分配给不太有意义的句子（例如，可以它有什么作用？）。当我们从真实文本语料库中给出模型句子时，我们期望它们具有高概率和低困惑度。在数学上，它被定义为测试集的归一化逆概率： 
<!-- $$ \mathrm{Perplexity}(W) = \sqrt[N]{1\over P(W_1,...,W_N)} $$ -->

$$
\mathrm{Perplexity}(W) = \sqrt[N]{1\over P(W_1,...,W_N)}
$$ 

<!-- **You can experiment with text generation using [GPT-powered text editor from Hugging Face](https://transformer.huggingface.co/doc/gpt2-large)**. In this editor, you start writing your text, and pressing **[TAB]** will offer you several completion options. If they are too short, or you are not satisfied with them - press [TAB] again, and you will have more options, including longer pieces of text. -->

你可以使用[GPT-powered text editor from Hugging Face](https://transformer.huggingface.co/doc/gpt2-large)**尝试文本生成。在此编辑器中，开始编写文本，然后按[TAB]将为您提供几个完成选项。如果它们太短，或者对它们不满意 - 再次按[TAB]，将有更多选项，包括较长的文本片段。

## GPT is a Family

<!-- GPT is not a single model, but rather a collection of models developed and trained by [OpenAI](https://openai.com).  -->

GPT 不是单一模型，而是由[OpenAI](https://openai.com)开发和训练的模型集合。

Under the GPT models, we have:

| [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2#openai-gpt2) | [GPT 3](https://openai.com/research/language-models-are-few-shot-learners) | [GPT-4](https://openai.com/gpt-4) |
| -- | -- | -- |
|Language model with upto 1.5 billion parameters. | Language model with up to 175 billion parameters | 100T parameters and accepts both image and text inputs and outputs text. |


The GPT-3 and GPT-4 models are available [as a cognitive service from Microsoft Azure](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/#overview?WT.mc_id=academic-77998-cacaste), and as [OpenAI API](https://openai.com/api/).

## Prompt Engineering

<!-- Because GPT has been trained on a vast volumes of data to understand language and code, they provide outputs in response to inputs (prompts). Prompts are GPT inputs or queries whereby one provides instructions to models on tasks they next completed. To elicit a desired outcome, you need the most effective prompt which involves selecting the right words, formats, phrases or even symbols. This approach is [Prompt Engineering](https://learn.microsoft.com/en-us/shows/ai-show/the-basics-of-prompt-engineering-with-azure-openai-service?WT.mc_id=academic-77998-bethanycheum) -->

由于 GPT 经过大量数据训练来理解语言和代码，因此它们会根据输入（提示）提供输出。提示是 GPT 输入或查询，可以向模型提供有关他们接下来完成的任务的指令。为了获得期望的结果，您需要最有效的提示，其中包括选择正确的单词、格式、短语甚至符号。这种方法是[Prompt Engineering](https://learn.microsoft.com/en-us/shows/ai-show/the-basics-of-prompt-engineering-with-azure-openai-service?WT.mc_id=academic-77998-bethanycheum)

[This documentation](https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/?WT.mc_id=academic-77998-bethanycheum) provides you with more information on prompt engineering.

## ✍️ Example Notebook: [Playing with OpenAI-GPT](GPT-PyTorch.ipynb)

Continue your learning in the following notebooks:

* [Generating text with OpenAI-GPT and Hugging Face Transformers](GPT-PyTorch.ipynb)

## Conclusion

<!-- New general pre-trained language models do not only model language structure, but also contain vast amount of natural language. Thus, they can be effectively used to solve some NLP tasks in zero-shop or few-shot settings. -->
新的通用预训练语言模型不仅对语言结构进行建模，还包含大量自然语言。因此，它们可以有效地用于解决零车间或少样本设置中的一些 NLP 任务。

## [Post-lecture quiz](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/220)
