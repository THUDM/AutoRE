#!/bin/bash

declare -A task_params

task_params["vicuna_relation"]="dataset=vicuna_relation_train eval_path=vicuna_relation_test cache_path=autore/vicuna/relation/train eval_cache_path=autore/vicuna/relation/test output_dir=ckpt/vicuna/relation learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6 max_steps=1200"
task_params["vicuna_subject"]="dataset=vicuna_subject_train eval_path=vicuna_subject_test cache_path=autore/vicuna/subject/train eval_cache_path=autore/vicuna/subject/test output_dir=ckpt/vicuna/subject learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6 max_steps=5300"
task_params["vicuna_fact"]="dataset=vicuna_fact_train eval_path=vicuna_fact_test cache_path=autore/vicuna/fact/train eval_cache_path=autore/vicuna/fact/test output_dir=ckpt/vicuna/fact learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6 max_steps=4430"

for task_name in "${!task_params[@]}"; do
  export WANDB_PROJECT_NAME="$task_name"

  declare -A params
  for param in ${task_params[$task_name]}; do
    key=$(echo $param | cut -f1 -d=)
    value=$(echo $param | cut -f2 -d=)
    params[$key]=$value
  done

  log_dir="${params[output_dir]}"
  parent_dir=$(dirname "${params[output_dir]}")
  log_dir="$parent_dir/log"
  if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
  fi

  CUDA_VISIBLE_DEVICES=6 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path /workspace/xll/checkpoints/lmsys/vicuna-7b-v1.5 \
    --dataset "${params[dataset]}" \
    --eval_path "${params[eval_path]}" \
    --cache_path "${params[cache_path]}" \
    --eval_cache_path "${params[eval_cache_path]}" \
    --template vicuna \
    --output_dir "${params[output_dir]}" \
    --finetuning_type lora \
    --cutoff_len 2048 \
    --lora_target q_proj,v_proj \
    --save_total_limit 3 \
    --lora_r 100 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps "${params[save_steps]}" \
    --eval_steps "${params[eval_steps]}" \
    --learning_rate "${params[learning_rate]}" \
    --num_train_epochs "${params[num_train_epochs]}" \
    --max_steps "${params[max_steps]}" \
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/$task_name.log"
done
