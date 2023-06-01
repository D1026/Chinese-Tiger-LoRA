import json

with open('./oasst_chat_train.jsonl', "r", encoding="utf-8") as f:
    lines = f.readlines()
    # print(lines)
    x = [json.loads(_) for _ in lines]

print(len(x))  # 25943
print(x[0])
print(x[-1]['round'])