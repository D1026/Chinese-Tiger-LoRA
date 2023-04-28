# from transformers import BloomConfig, BloomModel
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
#
# configuration = BloomConfig()
# configuration.vocab_size = 128
# print(configuration)
#
# model_name_or_path = "bigscience/mt0-large"
# tokenizer_name_or_path = "bigscience/mt0-large"
#
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, target_modules=['query_key_value'],
#     inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1
# )
#
# model = BloomModel(configuration)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# print(model.h[0].self_attention.query_key_value.lora_A)
# print(model.h[0].self_attention.query_key_value.lora_B)
# print(model.h[0].self_attention.query_key_value.lora_A['default'].weight)
# print(model.h[0].self_attention.query_key_value.lora_B['default'].weight)
# print(len(model.h))


# --- same function name in super-class and sub-class ---
class A():
    def __init__(self):
        self.x = 1

    def forward(self, y):
        print("coordinate: ({}, {})".format(self.x, y))

class B():
    def __init__(self, a):
        self.forward = a.forward

a = A()
b = B(a)
b.forward(3)

class C(B):
    def __init__(self, a):
        super().__init__(a)
        self.forward = self.forward_

    def forward_(self, z):
        print("coordinate: ({}, {}, {})".format(a.x, z, a.x+z))

c = C(a)
c.forward(3)


