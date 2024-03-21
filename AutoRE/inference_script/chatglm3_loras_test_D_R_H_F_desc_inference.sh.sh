#!/bin/bash
MODEL_NAME="chatglm3"
DATA_PATH="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_test.json"
version="v5"
mode="2e-4"
run_task() {
  STEP=$1
  TASK_TYPE=$2
  model_path="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/ckpt/${MODEL_NAME}/${TASK_TYPE}_${mode}/checkpoint-$STEP"
  EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/${MODEL_NAME}/${TASK_TYPE}_${mode}/redocred_$STEP"
  mkdir -p ${EVAL_SAVE_PATH}
  #    CUDA_VISIBLE_DEVICES=2 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/inference.py \
  /workspace/xll/Anaconda3/envs/chatglm/bin/deepspeed --master_port 12347 --include localhost:0,1,2,3,4,5,6,7 src/inference.py \
    --model_name_or_path /workspace/xll/checkpoints/THUDM6B3 \
    --adapter_name_or_path ${model_path} \
    --template chatglm3 \
    --version ${version} \
    --finetuning_type lora \
    --do_sample false \
    --temperature 0.95 \
    --top_p 0.6 \
    --data_path ${DATA_PATH} \
    --lora_test lora_${TASK_TYPE} \
    --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
}

# 按顺序执行任务
#run_task "3300" "relation"
#run_task "3350" "relation"
#run_task "3400" "relation"
run_task "5000" "subject"
#run_task "7500" "fact"
