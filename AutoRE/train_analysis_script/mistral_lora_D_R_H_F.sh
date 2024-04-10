#!/bin/bash
source /workspace/xll/Anaconda3/bin/activate chatglm
mode="D_R_H_F"

declare -A tasks
tasks["mistral"]="dataset=train_$mode eval_path=test_$mode cache_path=autore/mistral/$mode/train eval_cache_path=autore/mistral/$mode/test output_dir_base=ckpt/mistral/$mode model_name_or_path=checkpoints/Mistral-7B-Instruct-v0.2 template=mistral learning_rate=2e-4 num_train_epochs=6.0"

for task_name in "${!tasks[@]}"; do
  task_config=${tasks[$task_name]}

  for kv in $task_config; do
    key=${kv%%=*}
    value=${kv#*=}
    declare $key=$value
  done

  output_dir="${output_dir_base}_lr${learning_rate}_deepspeed"
  export WANDB_PROJECT_NAME="${task_name}_${mode}_${learning_rate}"

  log_dir="$output_dir"
  parent_dir=$(dirname "$output_dir")
  log_dir="$parent_dir/log"
  if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
  fi

  CUDA_VISIBLE_DEVICES=0 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --eval_path "$eval_path" \
    --cache_path "$cache_path" \
    --eval_cache_path "$eval_cache_path" \
    --template "$template" \
    --output_dir "$output_dir" \
    --finetuning_type lora \
    --cutoff_len 2048 \
    --lora_target q_proj,v_proj \
    --save_total_limit 5 \
    --lora_r 300 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/${task_name}_${mode}_${learning_rate}.log"
done
