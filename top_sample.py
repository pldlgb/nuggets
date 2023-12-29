import json 

score_path = "save/alpaca_gpt4/score"
f_path = "datasets/alpaca_gpt4"
origin_name = "alpaca_gpt4_data.json"
save_name = "alpaca_gpt4_sorted_score_sample_100_kmeans.json"

step = 5000
count = 52002
big_table = []

for i in range(11):
    start = str(i * step)
    end = str(min(i * step + step, count))

    with open(f"{score_path}/{start}_{end}_score.json", "r") as f:
        data = json.load(f)

        for k, v in data.items():
            big_table.append(v[0])

with open(f"{f_path}/{origin_name}", "r") as f:
    sample_data = json.load(f)

for i in range(len(sample_data)):
    sample_data[i]["score"] = float('%.3f' % big_table[i])

sorted_indices = sorted(range(len(big_table)), key=lambda x: -big_table[x])
big_table.sort(reverse=True)

numsbiger = 0
for s in [0.5,0.6,0.7,0.8,0.85,0.9]:
    for i in big_table:
        if i > s:
            numsbiger += 1
    print(s, numsbiger)
    numsbiger = 0


sorted_sample = [sample_data[i] for i in sorted_indices]

sorted_sample = json.dumps(sorted_sample, indent=4)
with open(f'{f_path}/{save_name}', 'w') as f:
    f.write(sorted_sample)

