##!/bin/bash
#source /workspace/xll/Anaconda3/bin/activate chatglm
#declare -A task_params
#
#
#task_params["mistral_relation"]="dataset=relation_train_analysis eval_path=relation_test_analysis cache_path=autore/vicuna/relation_analysis/train eval_cache_path=autore/vicuna/relation_analysis/test output_dir=ckpt/vicuna/relation_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
#task_params["mistral_subject"]="dataset=subject_train_analysis eval_path=subject_test_analysis cache_path=autore/vicuna/subject_analysis/train eval_cache_path=autore/vicuna/subject_analysis/test output_dir=ckpt/vicuna/subject_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
#task_params["mistral_fact"]="dataset=fact_train_analysis eval_path=fact_test_analysis cache_path=autore/vicuna/fact_analysis/train eval_cache_path=autore/vicuna/fact_analysis/test output_dir=ckpt/vicuna/fact_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
#
#for task_name in "${!task_params[@]}"; do
#  export WANDB_PROJECT_NAME="$task_name"_${params[learning_rate]}
#
#  declare -A params
#  for param in ${task_params[$task_name]}; do
#    key=$(echo $param | cut -f1 -d=)
#    value=$(echo $param | cut -f2 -d=)
#    params[$key]=$value
#  done
#
#  log_dir="${params[output_dir]}"
#  parent_dir=$(dirname "${params[output_dir]}")
#  log_dir="$parent_dir/log"
#  if [ ! -d "$log_dir" ]; then
#    mkdir -p "$log_dir"
#  fi
#
#  # 执行训练命令
#  #  CUDA_VISIBLE_DEVICES=6 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/train_bash.py \
#  /workspace/xll/Anaconda3/envs/auto/bin/deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
#    --deepspeed ds_config/stage2.json \
#    --stage sft \
#    --do_train \
#    --evaluation_strategy "steps" \
#    --model_name_or_path /workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2 \
#    --dataset "${params[dataset]}" \
#    --eval_path "${params[eval_path]}" \
#    --cache_path "${params[cache_path]}" \
#    --eval_cache_path "${params[eval_cache_path]}" \
#    --template mistral \
#    --output_dir "${params[output_dir]}_${params[learning_rate]}" \
#    --finetuning_type lora \
#    --cutoff_len 2048 \
#    --lora_target q_proj,v_proj \
#    --save_total_limit 5 \
#    --lora_r 100 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --quantization_bit 4 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_strategy "steps" \
#    --save_steps "${params[save_steps]}" \
#    --eval_steps "${params[eval_steps]}" \
#    --learning_rate "${params[learning_rate]}" \
#    --num_train_epochs "${params[num_train_epochs]}" \
#    --max_steps "${params[max_steps]}" \
#    --plot_loss \
#    --fp16 2>&1 | tee -a "$log_dir/$task_name.log"
#done

# 这个脚本用于同时启动任务，分配到8个卡上
#!/bin/bash
source /workspace/xll/Anaconda3/bin/activate chatglm
declare -A task_params

task_params["mistral_relation"]="dataset=relation_train_analysis eval_path=relation_test_analysis cache_path=autore/vicuna/relation_analysis/train eval_cache_path=autore/vicuna/relation_analysis/test output_dir=ckpt/vicuna/relation_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
task_params["mistral_subject"]="dataset=subject_train_analysis eval_path=subject_test_analysis cache_path=autore/vicuna/subject_analysis/train eval_cache_path=autore/vicuna/subject_analysis/test output_dir=ckpt/vicuna/subject_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"
task_params["mistral_fact"]="dataset=fact_train_analysis eval_path=fact_test_analysis cache_path=autore/vicuna/fact_analysis/train eval_cache_path=autore/vicuna/fact_analysis/test output_dir=ckpt/vicuna/fact_analysis learning_rate=5e-5 save_steps=100 eval_steps=100 num_train_epochs=6"

# GPU assignment for each task
declare -A gpu_assignments
gpu_assignments["mistral_relation"]="0,1,2"
gpu_assignments["mistral_subject"]="3,4,5"
gpu_assignments["mistral_fact"]="6,7"

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
  export WANDB_PROJECT_NAME="$task_name"_${params[learning_rate]}_analysis
  random_port=$((49152 + RANDOM % (65535 - 49152 + 1)))
  # Set the GPUs for the current task
  CUDA_VISIBLE_DEVICES=${gpu_assignments[$task_name]} /workspace/xll/Anaconda3/envs/auto/bin/deepspeed --num_gpus 3 --master_port=${random_port} src/train_bash.py \
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
    --fp16 2>&1 | tee -a "$log_dir/$task_name.log" &
done

wait
#    --max_steps "${params[max_steps]}" \
