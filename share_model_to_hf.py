import torch
from transformers import GenerationConfig, AutoTokenizer, BloomForCausalLM

hf_ckpt = "/data/Chinese-Tiger-LoRA/alpaca-lora/hf_ckpt"

# tokenizer = AutoTokenizer.from_pretrained(hf_ckpt)
# tokenizer.push_to_hub("Tiger-3b")

model = BloomForCausalLM.from_pretrained('/data/Chinese-Tiger-LoRA/alpaca-lora/hf_ckpt', torch_dtype=torch.float16)
model.push_to_hub("Tiger-3b")