# 🐅Chinese-Tiger-LoRA
### <font color=#871F78>record [2023/07/03]</font>
最近面试了安远AI的对齐与安全项目，面试中交流了关于ai是否已经“主动意识” 或者“理解能力”和“逻辑能力”，我的观点倾向于（对于当前的基于Autoregressive LM的大模型）没有，  
相反的，有很多观点认为，在许多所谓推理数据集上，大模型能够展现出逐步分析的推理过程。基于此认为 大模型有一定的理解能力。
实际上，区分 模型到底是有现实世界概念间的理解能力，还是“背书”（rote learning), 那么我们需要设计一个新的 “逻辑规则” （保证这个规则足够奇葩、新颖，历史上没出现过，训练数据里没见过），看模型是否能按照这个逻辑，去进行推理。  

结果是让人悲观的，即使是最简单的逻辑规则，模型的回答结果也是一塌糊涂。  
如下，是我在测试自己的模型时，曾经使用的两个自创的例子，即使强如 chat-gpt/gpt-4 也不能回答正确，并且出现幻觉分析过程。  

![chatGPT_1](https://raw.githubusercontent.com/D1026/Chinese-Tiger-LoRA/main/picture/chatGPT_1.jpeg)

个人观点，目前的模型很可能并没有人们期望的“自主能力”、“推理能力”，“幻觉”现象是一个很好的佐证。  
相比于“<font color=red>理解力</font>”，它更像依然停留在“归纳”、“总结”、“模仿”上，这可能与我们期望的 “理解” 依然有距离。  

### <font color=#871F78>record [2023/04/06]</font>
By testing the check-points during the model training, we found that model not always towards to better.

In the first thousands training steps(batch_size=128, 1 epoch ~= 20,000 steps), model's response was going better in terms of
"helpful", "honest" and "harmless". 

As the training continues, there are some question responses oscillated between getting nicer and getting worse. 
In some extreme occasion, model's response even degraded into outputting repetitive words.(This is certainly unacceptable!)

Without any constraint, 
Supervised FineTuning(SFT) on LLM for small instruct dataset shows improvement.
But with more and more samples for pure SFT, model can't be continuously optimized by user expectations.
I guess this is one reason of why chatGPT(Instruct-GPT) only use 13k SFT dataset, meanwhile Reward set and PPO dataset have 30+k.

Besides, chatGPT's Reward model learned by rank-relation instead of a scalar score, this is another important point. 
Learn to rank, not only reducing the annotation costs, 
but more influentially is, ranking is more stable and easy to learn. 
Because it is the essential way of the real world exist, 
while scalar measurement is only a means for humans to describe the world and exchange the knowledge.

I'm now working on find out a new way for LLM SFT constraint. 
by the way, I'm not consider using KL divergence penalty, as chatGPT/instructGPT did during reinforcement learning training.
If you have any idea, contact me by {ivan_duan@126.com or open an Issues, I will reply soon}.

result-record:
It appears various constraint can't help for training a better SFT model. In contrast, designing a more diverse and effective SFT dataset is even more useful.

尝试了各种SFT training 阶段约束，效果上看起来，增加“约束”条件 并不如 设计更好、更多样性的 SFT 数据集；

### project Information
1. This code based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
2. We collected all chinese instruction dataset before 2023/04/01 from belows:

    https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json
    
    https://github.com/hikariming/alpaca_chinese_dataset
    
    https://github.com/LianjiaTech/BELLE
    
    totally about 2.5 M samples. We trained [Bloomz](https://huggingface.co/bigscience/bloomz-3b) model on these data.
    