import json
f_path = "/mnt/user/E-liyunshui.lys-385738/LLMs/code/DAMO-ConvAI/deep-thinking/datasets/dumped/"
# 读取 JSON 文件
with open(f'{f_path}/alpaca_data.json', 'r') as f:
    data = json.load(f)

# add ids
for i in range(len(data)):
    data[i]["id"] = i
# 定义排序键函数，根据文本长度排序
def sort_key(item):
    return len(item['instruction'])+len(item['input'])+len(item['output'])

# 对 JSON 对象进行排序
sorted_data = sorted(data, key=sort_key)
for i in range(len(sorted_data)):
    sorted_data[i]["cur_id"] = i
# 将排序后的 JSON 对象转换回 JSON 字符串
sorted_json = json.dumps(sorted_data, indent=4)

# 将排序后的 JSON 字符串写入新的 JSON 文件
with open(f'{f_path}/alpaca_data_sorted.json', 'w') as f:
    f.write(sorted_json)
