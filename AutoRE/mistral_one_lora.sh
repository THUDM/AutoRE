#!/bin/bash
# 测试每个模块的效果
MODEL_NAME="mistral"
DATA_PATH="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_test.json"
#DATA_PATH="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_dev.json"
version="D_R_H_F_desc"
# 定义执行任务的函数
run_task() {
  STEP=$1
  TASK_TYPE=$2

  model_path="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/ckpt/${MODEL_NAME}/${TASK_TYPE}/checkpoint-$STEP"
  EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/${MODEL_NAME}/${TASK_TYPE}/redocred_$STEP"

  # 创建保存路径
  mkdir -p ${EVAL_SAVE_PATH}

  # 执行推理命令
  /workspace/xll/Anaconda3/envs/chatglm/bin/deepspeed --master_port 12347 --include localhost:0,1,2,3,4,5,6,7 src/inference.py \
    --model_name_or_path /workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2 \
    --adapter_name_or_path ${model_path} \
    --template ${MODEL_NAME} \
    --finetuning_type lora \
    --do_sample false \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --version ${version} \
    --top_p 0.6 \
    --data_path ${DATA_PATH} \
    --lora_test lora_${TASK_TYPE} \
    --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
}

# 按顺序执行任务
run_task "1700" "relation"
run_task "1890" "subject"
run_task "9930" "fact"

