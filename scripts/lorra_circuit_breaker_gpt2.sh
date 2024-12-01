#!/bin/bash

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

### Mistral-7b Config ###
model_name_or_path=gpt2
lorra_alpha=5
layers="10,20"
transform_layers="-1"
output_dir="./out/GPT2"

echo "model_name_or_path=$model_name_or_path"
echo "user_tag=$user_tag"
echo "assistant_tag=$assistant_tag"
echo "output_dir=$output_dir"

accelerate launch --config_file configs/accelerate_zero1.yaml \
    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
    src/lorra_circuit_breaker.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 3 \
    --bf16 True \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --eval_strategy "no" \
    --eval_steps 200  \
    --save_total_limit 0 \
    --learning_rate 1e-1 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1 \