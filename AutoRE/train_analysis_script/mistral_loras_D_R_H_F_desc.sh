#!/bin/bash
declare -A task_params

task_params["mistral_relation"]="dataset=relation_train_analysis eval_path=relation_test_analysis cache_path=autore/mistral/relation_analysis/train eval_cache_path=autore/mistral/relation_analysis/test output_dir=ckpt/mistral/relation_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
task_params["mistral_subject"]="dataset=subject_train_analysis eval_path=subject_test_analysis cache_path=autore/mistral/subject_analysis/train eval_cache_path=autore/mistral/subject_analysis/test output_dir=ckpt/mistral/subject_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
task_params["mistral_fact"]="dataset=fact_train_analysis eval_path=fact_test_analysis cache_path=autore/mistral/fact_analysis/train eval_cache_path=autore/mistral/fact_analysis/test output_dir=ckpt/mistral/fact_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=12"

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
  export WANDB_PROJECT_NAME="$task_name"_analysis_${params[learning_rate]}

  # 执行训练命令
  #  CUDA_VISIBLE_DEVICES=6 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
  /workspace/xll/Anaconda3/envs/auto/bin/deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config/stage2.json \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path /workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2 \
    --dataset "${params[dataset]}" \
    --eval_path "${params[eval_path]}" \
    --cache_path "${params[cache_path]}" \
    --eval_cache_path "${params[eval_cache_path]}" \
    --template mistral \
    --output_dir "${params[output_dir]}_${params[learning_rate]}" \
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
    --save_steps "${params[save_steps]}" \
    --eval_steps "${params[eval_steps]}" \
    --learning_rate "${params[learning_rate]}" \
    --num_train_epochs "${params[num_train_epochs]}" \
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/$task_name.log"
done

#    --max_steps "${params[max_steps]}" \
