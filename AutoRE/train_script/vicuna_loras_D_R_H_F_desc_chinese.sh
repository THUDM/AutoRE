#!/bin/bash
source /workspace/xll/Anaconda3/bin/activate auto
declare -A task_params

task_params["vicuna_relation_chinese"]="dataset=relation_chinese_train eval_path=relation_chinese_test cache_path=autore/vicuna/relation_chinese/train eval_cache_path=autore/vicuna/relation_chinese/test output_dir=ckpt/vicuna/relation_chinese learning_rate=5e-5 save_steps=10 eval_steps=10 num_train_epochs=6"
task_params["vicuna_subject_chinese"]="dataset=subject_chinese_train eval_path=subject_chinese_test cache_path=autore/vicuna/subject_chinese/train eval_cache_path=autore/vicuna/subject_chinese/test output_dir=ckpt/vicuna/subject_chinese learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
task_params["vicuna_fact_chinese"]="dataset=fact_chinese_train eval_path=fact_chinese_test cache_path=autore/vicuna/fact_chinese/train eval_cache_path=autore/vicuna/fact_chinese/test output_dir=ckpt/vicuna/fact_chinese learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"

for task_name in "${!task_params[@]}"; do

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
  export WANDB_PROJECT_NAME="$task_name"
#  _${params[learning_rate]}

  # 执行训练命令
#  CUDA_VISIBLE_DEVICES=6 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
  deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config/stage2.json \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path /workspace/xll/AutoRE_GitHub/AutoRE/checkpoints/lmsys/vicuna-7b-v1.5 \
    --dataset "${params[dataset]}" \
    --eval_path "${params[eval_path]}" \
    --cache_path "${params[cache_path]}" \
    --eval_cache_path "${params[eval_cache_path]}" \
    --template vicuna \
    --output_dir "${params[output_dir]}_${params[learning_rate]}" \
    --finetuning_type lora \
    --cutoff_len 2048 \
    --lora_target q_proj,v_proj \
    --save_total_limit 5 \
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
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/$task_name.log"
done
#    --max_steps "${params[max_steps]}" \

