---
slug: 'thoughts-the-internet-has-changed-forever'
title: 'The internet has changed forever'
subtitle: 'The world with AI'
category: 'Thoughts'
date: '2025-05-25'
cover_image: '/blog_images/thoughts-the-internet-has-changed-forever/brain_transition.png'
cover_image_prompt: 'Three brains: Brain without AI used, Brain with AI in conjunction, complete AI brain'
ai_assisted: false
---

What's the first thing you do when you connect to the internet? If you're like me, you'll go to Google.com. Each browser comes with a default search engine, and for most of us, it's **Google**. That gateway earlier had two main buttons, `Google Search` and `I'm Feeling Lucky`. `I'm feeling lucky` simply opened the first link that google ranked the highest for the query.

In 2022 chatGPT was released by OpenAI, and it was a game changer. It was able to answer questions better than any other search engine although didn't have the ability to be up-to-date with the latest information. It was able to write code, and even write essays. As the traffic on chatGPT grew, Google had to act fast before it loses in its own game. After many iterations, like the LamDa, Palm, gemini studio, notebookLM, etc., new `AI Mode` was unveiled in March 2025 in the US. They took a lot more time to bring this to the forefront compared to other AI companies even though the seminal paper on <dict word="transformer">transformers</dict> was published by google researchers; mostly because they couldn't forego their reputation with a _half-baked product_ and practically AI based search _breaks the main business model of Google's_ <dict word="Adsense">Adsense</dict>.
![Google before 2025 and after 2025](/blog_images/thoughts-the-internet-has-changed-forever/google_transition.png)

> And just like that, the gateway to the internet (and practically the internet itself) has changed forever.

As I work in the industry related to AI and its applications, I've been thinking a lot about the future of AI and the internet. I've been thinking about the impact of AI on the society, economy, job-market, environment, privacy, impact on the human mind, how it changes creativity itself, and more. In this post, I'll be discussing what I think AI is, the affordability of AI, some of its dark-sides, and finally some of my own "predictions" for the future of AI and the internet.

## Artificial Intelligence is a _Tool_

Like with most human inventions, AI is a tool. It's multi-disciplinary knowledge put into application to make **almost all** human knowledge available to you instantly via a digital interface. It's the evolution of written word and the internet. I understand, it feels like a magical tool. See all the different logos from different companies, they all look like a digital _all-encompassing oracle_. These companies might want you to think it can never be wrong, it can do anything, it's the ultimate future, but, it's not. It's limited by the knowledge it's trained on which is not even close to the entire human knowledge, the modalities it can work with (can it smell? feel? taste?), the compuatational power and time it's provided to solve a query. Even if we assume it has all the human knowledge in the world, human knowledge itself is not complete and not even "perfect", i.e., there can be more than one truth sometimes (or most of the time).

![Logos of AI tools from various technology companies](/blog_images/thoughts-the-internet-has-changed-forever/ai_logos.png)

Because it's a tool, we should be aware of both its use and abuse. Given it's immense potential of helping you solve problems faster, not using it would surely leave you behind in the race you're running. Whatever race you're running. If you're into making films, writing novels, producing music, working on electronics, or farming, etc. You should figure out the tool which integrates AI in your workflow, and learn to use it. Don't blindly follow it, understand it's capabilities and its shortcomings.

## Is it affordable for all?

> Future is now and here, but, it's unevenly distributed

Access to technology is not the same for all. And in general, access boils down to "availability" and "affordability". There are different aspects which govern availability which we'll skip for the time being. But, talking about affordability, we can to get a sense of the problem of affordability to these tools with some income based data available about the world.

As of this writing, the latest and the greatest models from big-technology companies costs in the range of approximately \$20 per month per user which are still rate limited. How many people do you think can afford that? The lower-end of the best AI model devices cost more than \$350 and then there's application fees. Each "good" AI application (for eg: cursorAI) you want to use might have its own cost, might bring it's total cost to a multiple of it. Free tier is generally highly rate-limited or has many other restrictions such that to effectively use the tool, you need at-least the first paid subscription.

Let's say, the AI cost usage for an application heavy user to be $B + nA$ where, $B$ is the cost of base LLM you generally want to use, $n$ the cost of applications you use and $A$ the average cost of each AI application you purchase. Assuming, a base LLM cost of \$20 and \$20 for each application as well; 1-2 (i.e., 1.5) for a light user, 2-3 for a medium user (i.e., 2.5) and 5-10 (i.e., 7.5); the cost of using AI in your daily "productivity" life would range from \$50 to \$170 per month. Also, once you become a heavy user, your base LLM cost can rise much above \$200 totaling the cost upwards of \$400-\$500 per month. To put perspective to this number, that currently is like buying 4 to 6 iPhones in a year. 😱

<iframe src="https://ourworldindata.org/grapher/world-bank-income-groups?time=2024&tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>

Looking at the [income groups of the world in the below infographic from "Our world in Data"](https://ourworldindata.org/grapher/world-bank-income-groups?time=2024), shows that most of the population of the world can't afford these tools if they also want to afford basic housing, utilities, food, etc. This brings back to the need of small LLMs which run on the edge and this affordability gap continue to reduce. The cost keeps coming down for access to these technology in some form or the other. This is a magical tool and hopefully power of intelligence is in the hands of everyone so that they can make better decisions, can help with whatever they're doing and certainly uplift society to the next level. _"Intelligence on demand"_, it's the evolution of computing and integration of computing with human action and perception.

The access definitely is democratized via the company HuggingFace ecosystem of models. But you need expertise in using those models, plus a GPU or atleast a decent laptop. They'll never be as powerfull as you see in the applications created by <dict word="bigtech">BigTech</dict>. Locally, most models won't have the right scaffolding and machinery required to give a "good" answer and will generally run with reduced thinking due to local inference limitations. BigTech has the ability to run the larger-models with more efficiency, at-scale and has the expertise to keep these models up-to-date with latest information. I think that's why almost all governments are trying to achieve AI sovereignty, data-pipelines custom built for the country with the ability to satisfy the intelligence demands of the population.

## The dark side 🌗

I think with anything great, comes it dark sides. Nuclear technology deemed to be the best energy source has also lead to nuclear bombs and some of the worst mishaps of humanity. Same is with AI. The LLM we all love and adore also has childhood issues.

![nuclear energy next to nuclear bomb exploding showing the stark difference of usage of the same technology](/blog_images/thoughts-the-internet-has-changed-forever/nuclear_difference.png)

-   **Errors, compounding errors**: LLMs errors compound with each token. Once the model is on the wrong path of token generation, everything downstream will also be bad. Look at examples of <dict word="slop">slop</dict>.
-   **Energy requirements**: As these models run on highly-specialized chips which are hungry for power, running world population scale intelligence on demand requires immense amount of electricity.
-   **Increased inequality**: Technology is an _enabler_. Let's say you get a car, you can travel further and get anywhere faster. Same with human cognition. These tools will help you to reach your mental-goal faster. This gives you an edge over who's simply walking towards that goal.
-   **It's a crutch**: Using a car to get everywhere can make you lethargic and unable to walk when needed. Similarly, always relying on AI (your brain-car) can cause your brain to atrophy.
-   **Mass surveillance**: As this tool helps "read" any digital information, it also opens the gate for mass surveillance which is already happening around the world.
-   **Eroding Trust**: Internet is filled with bots and deepfake content. Even customer service is just filled with bots. Your email is filled with auto-generated emails. It's easier for anyone to scam you with deepfake voice/video. When everything you see, you have to see with a suspicious eye; it indicates erosion of trust. In 2024, bots comprise more than 50% of the internet traffic. The thing you did, is probably now only consumed by another algorithm.

## Is <dict word="agi">AGI</dict> real or that mythical land of abundance?

These are some raw thoughts for you! Any chatter about AGI is purely philosophical at this point. Most experts disagree, on the timeline when we'll achieve AGI.

Isn't the dream of AGI similar to the dream of time-travel? As soon as computational methods reach a certain AI goal, we just move the goal-post further and then solve those goal-posts again. I think as humans, we always look for that infinite resource or a perpetual machine per se. Something (AGI) that will solve all our problems. And I think, AGI is contextually is marketed in the same format by BigTech. Artificial intelligence is quite different than the intelligence humans possess. One is designed to mimic the other through a digital interface, but, there are subtle but crucial differences. Although, not sure if we have any new turing test or if those "are you human tests" still work on the internet because you are seeing a human through its digital image.

I think, I have a theory; AGI is be more like a god, than a human for everyone? Will there be many AIs or just one? If you have played board games like _"raja mantri chor sipahi"_, Mafia, Werewolves, you would know what I mean by "god" here. The one that writes the rules or upholds them. Algorithms that runs your life 24x7. You're being interviewed by an AI. Your meeting notes are coming from another AI assistant. Your research is being done with an AI assistant. What you watch on the media is being run by an AI. Your healthcare claim will be accepted or denied by an AI. Who you make friends with is decided by an AI. Your next date is decided by an AI.

The philosophy of AGI blends into Godels incompleteness theorem and I think the existence of AGI contradicts with some fundamental laws, not sure which? But there definitely can be an intelligence which runs your life, it doesn't need to be an AGI 😊.

# Predictions for the next 5-10 years

To finish of the blog, I wanted to think hard on what are some of the high-likely things to happen or what problems are important to solve to improve AI. I am not the best expert, so, take my predictions with a grain of salt. But you're reading my blog, so, might as well finish this last part.

-   Error rate of AI systems will remain, but we'll make reliable engineered systems on top to make it generally useful across many industries. It will bleed into <dict word="agentic">Agentic AI</dict>, that is, AI working autonomously without any human intervention.
-   AI platform companies will solve commerce and ads on AI, such that economy can be powered by these systems. Companies will fight to be the brand be recommended by your AI.
-   In most cases, purpose built foundational models of many different flavors will exist for each industry and problem type. AI will be a hybrid of Small and Large models deployed locally and through cloud respectively.
-   No, you're not going to lose your job to AI anytime soon. But, you will lose your job to someone who knows more about AI than you. How to use it. Maybe more people will go into entrepreneurship, as AI democratizes knowledge.
-   Only bots will visit your website. When the trust on AI builds up, people will stop looking at the citation, and directly ingest AIs output without question. I don't know how much frameworks like [RSL](https://kumrayush.medium.com/really-simple-licensing-rsl-a-new-standard-protecting-writers-from-ai-companies-a9f5905f5c72) will work in the long-term.
-   We won't achieve AGI, but we will continue to solve harder and harder problems with the help of AI. We just have to keep asking the right question (🤧 prompt).

If you're really curious about this stuff, go on and read: https://ai-2027.com/research. They have some wild predictions.

# Blog Updates

In the spirit of this blog, I'm adding a small icon to differentiate between different articles on my website:

1. Brain Logo 1: Have they been entirely written by me
2. Brain Logo 2: with the help of AI
3. Brain Logo 3: fully automated.

I think most articles will be me+AI (except for this one), as I have no intention on going back of writing everything on my own in this world of hyper-speed. This is a magical tool and I want to learn how to best use it. The only way to do so, is use it often.

![The three brains](/blog_images/thoughts-the-internet-has-changed-forever/brain_transition.png)
