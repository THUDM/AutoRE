#!/bin/bash

MODEL_NAME="vicuna-7b-v1.5"
DATA_PATH="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_test.json"

# 定义执行任务的函数
run_task() {
    STEP=$1
    TASK_TYPE=$2

    model_path="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/ckpt/vicuna/${TASK_TYPE}/checkpoint-$STEP"
    EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/vicuna/${TASK_TYPE}/redocred_$STEP"

    # 创建保存路径
    mkdir -p ${EVAL_SAVE_PATH}

    # 执行推理命令
    CUDA_VISIBLE_DEVICES=1 /workspace/xll/Anaconda3/envs/chatglm/bin/python src/inference.py \
        --model_name_or_path /workspace/xll/checkpoints/lmsys/vicuna-7b-v1.5 \
        --adapter_name_or_path ${model_path} \
        --template vicuna \
        --finetuning_type lora \
        --do_sample false \
        --temperature 0.95 \
        --top_p 0.6 \
        --data_path ${DATA_PATH} \
        --lora_test lora_${TASK_TYPE} \
        --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
}

# 按顺序执行任务
run_task "1020" "relation"
run_task "5390" "subject"
run_task "4430" "fact"
