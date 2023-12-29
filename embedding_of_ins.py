import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModel
import json
from tqdm import tqdm
import numpy as np
import os
import argparse

def embed_texts_batched(texts, batch_size=30):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        all_embeddings.extend(embeddings)
    return all_embeddings

def dataset_preprocess(raw_data):
    data = []
    for e in raw_data:
        data.append({"instruction": e["instruction"].strip(), "input": e["input"].strip(), "output": e["output"].strip()})
    return data

def multiple_gen_promptify(instruction, input, output):
    if input != "":
        with_query = f"Instruction:\n{instruction}\nInput:\n{input}\nResponse:\n"
    else:
        with_query = f"Instruction:\n{instruction}\nResponse:\n"

    with_query_and_choice = f"{with_query}{output}"

    return with_query, with_query_and_choice

def load_sample(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
        data = dataset_preprocess(data)
        print(f"Data loaded: {file_name}.")
    
    ex_list = [[e["instruction"], e["input"], e["output"]] for e in data]
    ex_prompted = []
    for instruction, input, output in ex_list:
        _, line = multiple_gen_promptify(instruction, input, output)  # query, <query_with_answer>
        ex_prompted.append(line)
    return ex_prompted


# 初始化模型和分词器 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="checkpoint/LLaMA/convert_llama_7b")
    parser.add_argument('--instruction_path', type=str, help="datasets/alpaca_gpt4/alpaca_gpt4_data.json")
    parser.add_argument('--save_embedding_path', type=str, help="save/alpaca_gpt4/embeddings")
    args = parser.parse_args()

    MODEL_PATH = args.model_path 
    INSTRUCTION_PATH = args.instruction_path
    SAVE_EMBEDDING_PATH = args.save_embedding_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            # load_in_8bit=in_8bit,
        )
    model.eval()

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # load sample
    sample = load_sample(INSTRUCTION_PATH)
    print("START EMBEDDING ..."*3)
    embeddings = embed_texts_batched(sample)
    print(len(embeddings))
    np.save(f'{SAVE_EMBEDDING_PATH}/{len(sample)}.npy', embeddings)