# 🐅Chinese-Tiger-LoRA
### record [2023/04/06]
By testing the check-points during the model training, we found that model not always towards to better.

In the first thousands training steps(batch_size=128, 1 epoch ~= 20,000 steps), model's response was going better in terms of
"helpful", "honest" and "harmless". 

As the training continues, there are some question responses oscillated between getting nicer and getting worse. 
In some extreme occasion, model's response even degraded into outputting repetitive words.(This is certainly unacceptable!)

Without any constraint, 
Supervised FineTuning(SFT) on LLM for small instruct dataset shows improvement.
But with more and more samples for pure SFT, model can't be continuously optimized by user expectations.
I guess that's why chatGPT(Instruct-GPT) only use 13k SFT dataset, meanwhile Reward set and PPO dataset have 30+k.

Besides, chatGPT's Reward model learned by rank-relation instead of a scalar score, this is another important point. 
Learn to rank, not only reducing the annotation costs, 
but more influentially is, ranking is more stable and easy to learn. 
Because it is the essential way of the real world exist, 
while scalar measurement is only a means for humans to describe the world and exchange the knowledge.

I'm now working on find out a new way for LLM SFT constraint. 
by the way, I'm not consider using KL divergence penalty, as chatGPT/instructGPT did during reinforcement learning training.
If you have any idea, contact me by {ivan_duan@126.com or open an Issues, I will reply soon}.

### project Information
1. This code based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
2. We collected all chinese instruction dataset before 2023/04/01 from belows:

    https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json
    
    https://github.com/hikariming/alpaca_chinese_dataset
    
    https://github.com/LianjiaTech/BELLE
    
    totally about 2.5 M samples. We trained [Bloomz](https://huggingface.co/bigscience/bloomz-3b) model on these data.
    