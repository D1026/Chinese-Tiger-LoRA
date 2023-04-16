import os

import torch
import transformers
from peft import PeftModel
# from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from transformers import AutoTokenizer, BloomForCausalLM

# BASE_MODEL = os.environ.get("BASE_MODEL", None)
BASE_MODEL = "/data/models/bloomz-3b"
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = BloomForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

# first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight = base_model.transformer.h[0].self_attention.query_key_value.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    # "tloen/alpaca-lora-7b",
    "/data/Chinese-Tiger-LoRA/alpaca-lora/bloom-lora",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# lora_weight = lora_model.base_model.model.model.layers[
#     0
# ].self_attn.q_proj.weight
lora_weight = lora_model.base_model.transformer.h[0].self_attention.query_key_value

assert torch.allclose(first_weight_old, first_weight)

# merge weights
# for layer in lora_model.base_model.model.model.layers:
#     layer.self_attn.q_proj.merge_weights = True
#     layer.self_attn.v_proj.merge_weights = True
for layer in lora_model.base_model.transformer.h:
    layer.self_attention.query_key_value.merge_weights = True


lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()  # key like:base_model.model.transformer.h.15.self_attention.dense.weight
# deloreanized_sd = {
#     k.replace("base_model.model.", ""): v
#     for k, v in lora_model_sd.items()
#     if "lora" not in k
# }

deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

BloomForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="10GB"
)
tokenizer.save_pretrained("./hf_ckpt")