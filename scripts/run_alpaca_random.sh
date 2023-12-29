gpu=$1
start=$2
pace=$3
batch_size=$4

python instruction_main.py \
      --dataset alpaca --prompt_version default \
      --exemplar_method stratified --num_k_shots 1 \
      --model_type local --model_size 7b \
      --model_path "checkpoint/LLaMA/convert_llama_7b" \
      --prompt_path "datasets/alpaca_gpt4/alpaca_gpt4_data.json" \
      --test_path "datasets/alpaca_gpt4/alpaca_gpt4_kmeans_100.json" \
      --save_path "save/alpaca_gpt4/score" \
      --kv_iter 1 \
      --step_size 0.01 --momentum 0.9 \
      --batch_size $batch_size \
      --gpus $gpu \
      --start $start \
      --pace $pace
