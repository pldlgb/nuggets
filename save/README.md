### Build Predefined Task Set
#### 1. Download instruction dataset from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) or [Alpaca-GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)
```bash
mkdir datasets
# Alpaca
mkdir datasets/alpaca/
wget -P datasets/alpaca/ https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json 
# Alpaca-GPT4
mkdir datasets/alpaca_gpt4
wget datasets/alpaca_gpt4/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json  
```
#### 2. instruction embedding
```bash
python embedding_of_ins.py --model_path $MODEL_PATH --save_embedding_path $SAVE_EMBEDDING_PATH
```
#### 3. KMeans Sampling
```bash
python kmeans_sample.py
```