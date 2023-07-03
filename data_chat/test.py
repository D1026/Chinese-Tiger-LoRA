import json
import copy
import os

#  === open assistant ===

# with open('./oasst_chat_train.jsonl', "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     # print(lines)
#     x = [json.loads(_) for _ in lines]
#
# with open('./oasst_chat_train.json', "w", encoding="utf-8") as f:
#     json.dump(x, f)
#
# print(len(x))  # 25943
# print(x[0])
# print(x[-1]['round'])
#
# with open('./oasst_chat_val.jsonl', "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     # print(lines)
#     x = [json.loads(_) for _ in lines]
#
# with open('./oasst_chat_val.json', "w", encoding="utf-8") as f:
#     json.dump(x, f)
#
#
#  === alpaca_gpt4: en, zh ===
# all = []
# # with open("./alpaca_gpt4_data.json", "r") as f:
# with open("./alpaca_gpt4_data_zh.json", "r") as f:
#     data = json.load(f)
#
# print(len(data))
# print(data[0].keys())
#
# for item in data:
#     res = {}
#     round = item['instruction'] + '\n'
#     if item['input']:
#         round += "input: " + item['input'] + '\n'
#
#     output = item['output']
#
#     res['round'] = [round]
#     res['output'] = output
#
#     all.append(copy.deepcopy(res))
#
# with open("./alpaca_gpt4_chat_zh.json", "w", encoding="utf-8") as f:
#     # f.write(json.dumps(all)
#     # )
#     json.dump(all, f)

# all = []
# with open("./alpaca_gpt4_data.json", "r") as f:
#     data = json.load(f)
#
# print(len(data))
# print(data[0].keys())
#
# for item in data:
#     res = {}
#     round = item['instruction'] + '\n'
#     if item['input']:
#         round += "input: " + item['input'] + '\n'
#
#     output = item['output']
#
#     res['round'] = [round]
#     res['output'] = output
#
#     all.append(copy.deepcopy(res))
#
# with open("./alpaca_gpt4_chat.json", "w", encoding="utf-8") as f:
#     # f.write(json.dumps(all)
#     # )
#     json.dump(all, f)
#
#
#  ===  databricks-dolly  ===
# with open("./databricks-dolly-15k.jsonl", "r") as f:
#     lines = f.readlines()
#
# data = [json.loads(_) for _ in lines]
# print(data[0])
# print(data[0].keys())
#
# all = []
# for item in data:
#     res = {}
#     round = item['instruction'] + '\n'
#     if item['context']:
#         round += "context: " + item['context'] + '\n'
#
#     output = item['response']
#
#     res['round'] = [round]
#     res['output'] = output
#
#     all.append(copy.deepcopy(res))
#
# with open("./databricks-dolly-chat.json", "w", encoding="utf-8") as f:
#     # f.write(json.dumps(all)
#     # )
#     json.dump(all, f)


# dataset
# def is_chinese(string):
#     for ch in string:
#         if u'\u4e00' <= ch <= u'\u9fff':
#             return True
#
#     return False
#
#
# from datasets import load_dataset
# from utils.prompter import Prompter
# from transformers import AutoTokenizer
#
# fpath = "/Users/duanyiwen/PycharmProjects/Chinese-Tiger-LoRA/data_chat2"
# fs = os.listdir(fpath)
# data = load_dataset("json", data_dir=fpath, data_files=fs)
# # data = load_dataset(fpath)
# print(data["train"][0])
# print(data["train"][-1])
# for i in range(len(data["train"])):
#     x = data["train"][i]["output"]
#     # if is_chinese(x):
#         # print(x)
"""
{'round': ["Could you retrieve the top 5 current most watched pornhub.com video's?\n"], 'output': "I'm sorry, but I cannot retrieve or provide information on adult content as it is not appropriate for this platform. chat-Abell policies prohibit the discussion of sexually explicit or violent material, as well as other inappropriate topics. If you have any other questions, I would be happy to help with those."}
{'round': ['How to split a video into images using ffmpeg?\n', '1. To split a video into images using ffmpeg, open a command prompt and navigate to the folder containing the video.\n\n2. Then, type the following command and press Enter:\n\nffmpeg -i <name-of-video-file> -r <frame-rate> <name-of-image-file-%d.jpg>\n\nReplace <name-of-video-file> with the name of the video file, <frame-rate> with the desired frame rate, and <name-of-image-file> with the desired name for the image files.\n\n3. ffmpeg will then split the video into images and save them in the current directory.\n', 'How do I install ffmpeg and add it to my environment variables?\n'], 'output': 'Follow these steps:\nType system variables into the search bar and click the Edit the system environment variables option.\nUnder the User variables section, select Path and click the Edit button.\nChoose New from the side menu.\nAdd C:\\ffmpeg\\bin to the empty field and confirm changes with OK.'}

"""

# fpath = "/Users/duanyiwen/PycharmProjects/Chinese-Tiger-LoRA/data_chat2/alpaca_gpt4_chat_zh.json"  # 48818
# # fpath = "/Users/duanyiwen/PycharmProjects/Chinese-Tiger-LoRA/data_chat2/oasst_chat_train.json"  # 25943
# data = load_dataset("json", data_files=fpath)
# print(data["train"][0])
# print(data["train"][-1])

# print(data)
#
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
#
# tokenizer.pad_token_id = (
#         0  # unk. we want this to be different from the eos token
#     )
# tokenizer.padding_side = "left"  # Allow batched inference
#
# cutoff_len: int = 512
# train_on_inputs = True
#
#
# def tokenize(prompt, add_eos_token=True):
#     # there's probably a way to do this with the tokenizer settings
#     # but again, gotta move fast
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=cutoff_len,
#         padding=False,
#         return_tensors=None,
#     )
#     if (
#             result["input_ids"][-1] != tokenizer.eos_token_id
#             and len(result["input_ids"]) < cutoff_len
#             and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)
#
#     result["labels"] = result["input_ids"].copy()
#
#     return result
#
# prompter = Prompter("chat-Abell")
# def generate_and_tokenize_prompt(data_point):
#     full_prompt = prompter.generate_prompt_chat(
#         data_point["round"],  # ["How big is the earth?\n"], ["How big is the earth?\n",
#         # "6,371km.\n",
#         # "so, what..?\n"]
#         data_point["output"],
#     )
#
#     # print("full_prompt:", full_prompt)
#     # print("-.-.-.-.-")
#
#     tokenized_full_prompt = tokenize(full_prompt)
#     if not train_on_inputs:
#         user_prompt = prompter.generate_prompt_chat(
#             data_point["round"]
#         )
#         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
#         user_prompt_len = len(tokenized_user_prompt["input_ids"])
#
#         tokenized_full_prompt["labels"] = [
#                                               -100
#                                           ] * user_prompt_len + tokenized_full_prompt["labels"][
#                                                                 user_prompt_len:
#                                                                 ]  # could be sped up, probably
#     return tokenized_full_prompt
#
# train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)


# ---
# import json
#
# fn = "./cot_fsopt.jsonl"
# fn = "./cot_zsopt.jsonl"
#
# with open(fn, "r") as f:
#     lines = f.readlines()
#
# data = [json.loads(_) for _ in lines]
# print(data[0].keys())
#
# chat_cot = []
#
# for item in data:
#     cot = {}
#     cot["round"] = [item["inputs"] + "\n"]
#     cot["output"] = item["targets"]
#     chat_cot.append(cot)
#
# with open("./cot_zsopt_chat.json", "w", encoding="utf-8") as f:
#     json.dump(chat_cot, f, indent=4)


# ----
import json

with open("./school_math_0.25M.json", "r") as f:
    lines = f.readlines()
data = [json.loads(_) for _ in lines]

print(data[0])
print(data[0].keys())

chat_all = []
for item in data:
    math = {}
    round = item['instruction'] + '\n'
    if item['input']:
        round += "input: " + item['input'] + '\n'

    output = item['output']

    math['round'] = [round]
    math['output'] = output

    chat_all.append(copy.deepcopy(math))

with open("./Belle_school_math_0.25M.json", "w", encoding="utf-8") as f:
    json.dump(chat_all, f, indent=4)
