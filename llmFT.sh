set -e
# set -x
source /app/bin/proxy.sh
root=$(dirname $(dirname "$PWD"))
model_name=llama3.1-8b
model_path="$root/modelscope/$model_name"

cd "$root/llmR/tail_score"
pip install peft
pip install trl

python -m pdb llmFT.py \
    --train_data_file data/ft_train_data.jsonl \
    --val_data_file data/ft_val_data.jsonl \
    --model_path $model_path \
    --output_path FToutput \
    --eval_freq 5 \
    --save_freq 5 \
    --log_freq 5 \
    --train_bsz 1 \
    --eval_bsz 1 \
    --FTlr 1e-4 \
    --gradient_accumulation_steps 1 \
    --max_steps 15 \
    --gpus 0,1,2,3 \
