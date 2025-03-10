---
slug: 'cs-trends-h1-2024-selected-topics'
title: 'CS Trends H1 2024: Selected Topics'
subtitle: 'Opinion piece on selected topics'
category: 'CS Yearly Trends'
date: '2025-03-03'
cover_image: '/blog_images/cs-trends-h1-2024/wordcloud.png'
cover_image_prompt: 'Wordcloud of most used words in AI research papers'
---

# CS Trends H1 2024: Selected Topics

I originally planned to write this article right after publishing the [previous one](https://digital-madness.in/writing/cs-trends-h1-2024/), but as always, life had other plans. Nevertheless, it's time to dive back into the latest trends shaping computer science research.

The word cloud in the banner image—and introduced in the [previous article](https://digital-madness.in/writing/cs-trends-h1-2024/)—highlights the most common terms extracted from arXiv papers. Below, you’ll find the full list of extracted words. If you're interested in exploring the complete dataset, simply click ‘All Words’ to expand it:

<details>
<summary>All words</summary>

All words extracted using the [wordcloud](https://pypi.org/project/wordcloud/) Python package from papers published since 2007:

> neural network, language model, machine learning, deep learning, Experimental results, large language, reinforcement learning, Extensive experiment, natural language, learning model, training data, deep neural, models LLM, convolutional neural, ie, existing methods, results demonstrate, computer vision, code available, learning algorithm, diffusion model, publicly available, generative model, stateoftheart methods, object detection, loss function, time serie, well, point cloud, particular, artificial intelligence, benchmark dataset, addition, context, language processing, demonstrate effectiveness, data set, Moreover, address issue, experiments demonstrate, downstream task, recent year, learning methods, classification task, representation learning, data augmentation, widely used, proposed model, previous work, et al, image classification, wide range, outperforms stateoftheart, one, transfer learning, federated learning, pretrained model, address challenge, given, semantic segmentation, known, Furthermore, form, lower bound, achieves stateoftheart, learning framework, medical image, include, network architecture, graph neural, datasets demonstrate, ii, state art, concept, resulting, significant improvement, consider, output, question answering, describe, contrastive learning, along, latent space, adversarial attack, knowledge graph, effectiveness proposed, second, recent work, models trained, attention mechanism, proposed algorithm, general, significantly outperform, find, significantly improve, demonstrate proposed, computational cost, thu, large number, domain adaptation, gradient descent, across different, source code, anomaly detection, Additionally, ground truth, across various, generative adversarial, synthetic data, combined, utilize, proposed framework, future research, corresponding, case, instance, adversarial network, considered, prior work, enable, employ, Although, upper bound, contrast, combination, variant, learning RL, realworld dataset, model trained, make, called, social media, target domain, operation, machine translation, stateoftheart results, real world, instead, benefit, apply, challenging task, allows us, example, existing approaches, classifier, eg, data distribution, realworld application, labeled data, learning techniques, adversarial example, observation, foundation model, utilizing, outperforms existing, supervised learning, rather, end, implementation, autonomous driving, introduced, represent, sequence, measure, either, recent advance, change, training set, leverage, cluster, feature extraction, compared stateoftheart, provided, perspective, networks CNN, hand, computational complexity, learning task, best knowledge, associated, found, results proposed, allow, characteristic, considering, incorporate, behavior, support, recurrent neural, individual, commonly used, image segmentation, namely, Simulation results, respectively, call, part, construct, common, presented

</details>

We'll go over the most relevant terms, filtering out common or redundant words. As [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) famously said:

> If you can't explain something in simple words, you don't understand it well enough

<details>
<summary>Ignored Words</summary>

These words are excluded from our analysis as they are either too common or redundant:

> Extensive experiment, learning model, deep neural, models LLM, ie, existing methods, results demonstrate, code available, well, particular, addition, context, language processing, demonstrate effectiveness, Moreover, address issue, experiments demonstrate, downstream task, recent year, learning methods, widely used, proposed model, previous work, et al, image classification, wide range, outperforms stateoftheart, one, address challenge, given, known, Furthermore, form, achieves stateoftheart, learning framework, include, datasets demonstrate, ii, state art, concept, resulting, significant improvement, consider, output, describe, along, effectiveness proposed, second, recent work, proposed algorithm, general, significantly outperform, find, significantly improve, demonstrate proposed, thu, large number, across different, source code, Additionally, across various, combined, utilize, proposed framework, future research, corresponding, case, instance, considered, prior work, enable, employ, Although, upper bound, contrast, combination, variant, model trained, make, called, operation, stateoftheart results, real world, instead, benefit, apply, challenging task, allows us, example, existing approaches, eg, realworld application, learning techniques, adversarial example, observation, utilizing, outperforms existing, rather, end, implementation, introduced, represent, sequence, measure, either, recent advance, change, training set, leverage, compared stateoftheart, provided, perspective, networks CNN, hand, computational complexity, learning task, best knowledge, associated, found, results proposed, allow, characteristic, considering, incorporate, behavior, support, individual, commonly used, namely, Simulation results, respectively, call, part, construct, common, presented, publicly available

</details>

## Brief Explanations

Now, let’s explore the key terms extracted from nearly two decades of computer science research. I’ll arrange them in a logical order so that one concept naturally leads to the next.

### Artificial Intelligence

Artificial Intelligence (AI) is a branch of computer science dedicated to creating machine-based systems that can perceive, reason, plan, learn, and make decisions—much like humans do. Broadly, any automated system can be viewed as a form of intelligence, with varying levels of complexity and problem-solving abilities. At the most advanced end lies _Artificial **General** Intelligence_ (AGI)—a machine capable of tackling any task or problem it encounters. Pursuing AGI is a major goal for governments, corporations, and research institutions alike, as it promises a versatile solution to virtually every conceivable challenge.

### Machine Learning

When trying to build Artificial Intelligence (AI), there are generally two ways to give machines “intelligence”:

1. **Hand-coded Rules**: Write down every single rule or step the machine should follow. For example, you could program a chess computer by teaching it every valid chess move and telling it exactly what to do in each situation. This is time-consuming, but precise. It often relies on logic and principles which we can
2. **Learning from Data**: Provide the machine with lots of examples and let it find patterns on its own, without explicitly programming every rule.

The second approach is known as _Machine Learning (ML)_. In ML, a computer uses a _learning algorithm_ to discover patterns in large amounts of data—rather than relying on rules that programmers have written by hand. This ability to “learn” from data makes ML the most popular way to build modern AI systems. The output of a learning algorithm is a *trained machine learning model*, often referred to simply as an **ML model**.

### Types of Learning

When we talk about a machine learning from data, we're essentially deciding *how* it acquires knowledge. Just as humans learn from parents and teachers telling us what's right or wrong, by interacting with different objects around us, or by introspecting—so too can a machine. These approaches are commonly grouped into three main categories (though the third, sadly, didn’t show up among the most common words from the research, it remains one of the most important). Understanding these categories helps us appreciate why certain algorithms excel at specific tasks and how different learning methods can be combined to build more versatile AI systems.

- In **Supervised Learning**, the machine is trained on a **labeled dataset**—meaning each example comes with the “correct answer.” This correct answer is often referred to as the **ground truth**. Think of it as having a personal tutor who checks your homework: you try to solve math problems (predict something), and the tutor immediately tells you if you’re right or wrong. Over time, you learn to get more problems right on your own. This is the most common way of *training* machines.

- With **Reinforcement Learning (RL)**, machines learn by trial and error in an environment that rewards or penalizes their actions. This is somewhat like training a pet: when the pet does something good (e.g., fetch a ball), it gets a treat. When it messes up (e.g., chew on your slippers, poop on your sofa), it gets scolded. Over many attempts, the dog and the same with a machine, figures out which actions yield the best outcomes in different scenarios.

- In **Unsupervised Learning**, there are no labeled examples—no right or wrong answers. The machine’s job is to uncover hidden patterns or structures in the data. Unsupervised methods are especially valuable when labels are scarce or non-existent, and they often serve as a foundation for more advanced tasks in both supervised and reinforcement settings. This is the most common way humans learn, finding patterns in everyday observations and self-discovery.

### Datasets

A dataset is simply a collection of examples used for machine learning. Depending on the learning approach, these examples may be labeled, partially labeled, or entirely unlabeled. The size of a dataset can range from just a handful of examples—often referred to as a few-shot learning scenario, where models strive to learn effectively with minimal data—to billions or even trillions of examples.

The dataset used to train a model is called the **training dataset**. In a typical machine learning project, many design choices—such as the learning algorithm or pace of learning—must be made. A validation dataset helps in making these decisions by helping us tweak these levers. Finally, a test dataset is used to assess how well the model performs on previously unseen data. Together, these three sets are often referred to as a **benchmark dataset**, providing a consistent basis for comparing different learning algorithms.

#### Data Distribution

In a dataset, we often need a formal way to describe where the data comes from, the range in which it lies, and the variability we can expect. This method of describing the data is called its **data distribution**. If a data point does not fit within the dataset’s usual range, it is said to be out-of-distribution or out-of-domain. One of the more prized capabilities of a machine learning algorithm is to handle such out-of-distribution data effectively—a challenge often tackled by **domain adaptation**.

**Real-world data** frequently comes from sensors such as satellites, cameras, traffic lights, or everyday devices like smartphones. Additional data can also be created through a technique called data augmentation—for example, rotating an image of a cat still yields a valid example for training a cat-recognition model. Data can be generated in many other ways as well, such as creating multiplication tables to build a dataset of math problems. Both data augmentation and generating data that isn’t directly observed are forms of **synthetic data**, which is heavily used in machine learning.

### Large Language Models

Have you heard of ChatGPT? If not, it's time to catch up with one of the fastest-growing tech products. Launched in November 2022, ChatGPT [amassed one million users in just five days](https://explodingtopics.com/blog/chatgpt-users). The name ChatGPT combines "chat" and "GPT," which stands for Generative Pretrained Transformer, indicating that you are interacting with a sophisticated machine learning model. While we will delve deeper into the concepts of **GPT** in another technical blog, for now, we will refer to this sophisticated and VERY _LARGE_ machine learning _model_ to imitate human _lanaguage_, commonly known as a **LLM**.

This ML model is trained on a vast corpus of digitized knowledge to generate text by [continuously determining the next word based on preceding words](https://en.wikipedia.org/wiki/Autoregressive_model) until a designated stop word is reached. There are [numerous models](https://huggin
gface.co/spaces/lmsys/chatbot-arena-leaderboard) available, each with its strengths and weaknesses. The capabilities of this ML model are remarkable, including generating realistic [images](https://www.midjourney.com/home) and [videos](https://lumalabs.ai/dream-machine), composing [poetry](https://lil.law.harvard.edu/blog/2022/12/20/chatgpt-poems-and-secrets/), creating [music](https://suno.com/), writing [code](https://github.com/features/copilot), diagnosing [diseases](https://arxiv.org/pdf/2312.00164), predicting [protein structures](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10410766/), and even assisting in the development of targeted cancer treatments, all with a simple [prompt](https://en.wikipedia.org/wiki/Prompt_engineering) given in human language. This variant of artificial intelligenc is supported and funded by investment in Big Technology Firms and venture capitalists from . As one of the most transformative technologies of our era, it influences various aspects of life, from macro-level geopolitical effects [^2] to micro-level interactions with our beloved devices, like smartphones [^3].

[^2]: https://www.lazard.com/research-insights/the-geopolitics-of-artificial-intelligence/
[^3]: https://www.apple.com/apple-intelligence/

In the near future, it is likely that any text, image, or video you encounter on the internet or elsewhere will be partly or wholly generated by a machine learning model. Indeed, portions of this and other blogs on this website have been enhanced by such a chatbot. Despite the current excitement around Artificial Intelligence and its transformative potential, there are concerns about it leading to a scientific monoculture [^4] and the degradation of scientific inquiry [^5]. Although AI tools can objectively boost productivity, they may also result in reduced learning and understanding. If you always have an expert by your side, would you ever strive to become an expert yourself? This could lead to the proliferation of misinformation and job losses [^6] due to the risks associated with Artificial General Intelligence. Additionally, it may provide better tools for scammers to commit identity theft or impersonate loved ones over the phone, enable politicians to create targeted political propaganda, or allow terrorists to engineer dangerous biological materials. Another significant issue is the substantial energy consumption of this technology, which accelerates climate change—a major looming threat to humanity [^7]. Consequently, governments worldwide are racing to create long-term governance frameworks to research, develop, and commercialize this technology [^8].

[^4]: [Artificial intelligence and illusions of understanding in scientific research](https://www.nature.com/articles/s41586-024-07146-0)
[^5]: [The exponential enshitification of science](https://garymarcus.substack.com/p/the-exponential-enshittification)
[^6]: https://fortune.com/2024/05/14/ai-tsunami-imf-chief-labor-market-job-loss/
[^7]: [AI data centers wreaks havoc on power grids](https://www.bloomberg.com/graphics/2024-ai-data-centers-power-grids/), [AI Energy consumption](https://spectrum.ieee.org/ai-energy-consumption)
[^8]: [President Biden Issues Executive Order on Safe, Secure, and Trustworthy Artificial Intelligence](https://www.whitehouse.gov/briefing-room/statements-releases/2023/10/30/fact-sheet-president-biden-issues-executive-order-on-safe-secure-and-trustworthy-artificial-intelligence/)

As with any great technology it has its pros and cons. As long as the pros largely outweigh the cons, this technology should overall improve and uplift our lives, isn't it? For example, even though LLM take immense amount of energy to train, once trained they are super useful and can help in all sorts of tasks. It's much better to use a LLM to write faster and overall save time and energy [^9]. It's not AI that is going to take your job, but someone who knows how to use AI might [^10]. At the same time, LLMs bring tools of creativity accessible to each and everyone in the world as distribution and access to these tools becomes wider. If people accept this as the most useful tool[^11] created since the birth of a computer and internet or hate it[^12], only time will tell, but, the genie is surely out of the bottle and there's no putting it back.

[^9]: [Comparing the energy footprint of a human an a LLM](https://cacm.acm.org/blogcacm/the-energy-footprint-of-humans-and-large-language-models/)
[^10]: [Human + AI](https://www.businessinsider.com/ai-wont-take-your-job-someone-who-uses-it-might-2023-5)
[^11]: [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/pdf/2303.12712)
[^12]: [Hollywood's strike against AI](https://apnews.com/article/hollywood-ai-strike-wga-artificial-intelligence-39ab72582c3a15f77510c9c30a45ffc8)
