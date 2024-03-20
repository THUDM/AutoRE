#!/bin/bash

# 初始化任务字典
declare -A task_params

# 为每个任务定义参数
task_params["chatglm3_subject"]="dataset=subject_train eval_path=subject_test cache_path=autore/chatglm3/subject/train eval_cache_path=autore/chatglm3/subject/test output_dir=ckpt/chatglm3/subject learning_rate=2e-4 save_steps=100 eval_steps=100 num_train_epochs=6"

# 循环遍历每个任务
for task_name in "${!task_params[@]}"; do
  # 设置环境变量

  # 解析任务参数
  declare -A params
  for param in ${task_params[$task_name]}; do
    key=$(echo $param | cut -f1 -d=)
    value=$(echo $param | cut -f2 -d=)
    params[$key]=$value
  done

  # 设置日志目录
  log_dir="${params[output_dir]}"
  parent_dir=$(dirname "${params[output_dir]}")
  log_dir="$parent_dir/log"
  if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
  fi

  export WANDB_PROJECT_NAME="${task_name}_${params[learning_rate]}_gpu"
  # 执行训练命令
  CUDA_VISIBLE_DEVICES=4 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path /workspace/xll/checkpoints/THUDM6B3 \
    --dataset "${params[dataset]}" \
    --eval_path "${params[eval_path]}" \
    --cache_path "${params[cache_path]}" \
    --eval_cache_path "${params[eval_cache_path]}" \
    --template chatglm3 \
    --output_dir "${params[output_dir]}_${params[learning_rate]}" \
    --finetuning_type lora \
    --cutoff_len 2048 \
    --lora_target query_key_value \
    --save_total_limit 5 \
    --lora_r 100 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps "${params[save_steps]}" \
    --eval_steps "${params[eval_steps]}" \
    --learning_rate "${params[learning_rate]}" \
    --num_train_epochs "${params[num_train_epochs]}" \
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/$task_name.log"
done

#    --max_steps "${params[max_steps]}" \
