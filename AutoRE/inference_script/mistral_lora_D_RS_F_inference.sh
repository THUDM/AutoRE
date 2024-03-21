#!/bin/bash

MODEL_NAME="mistral"
declare -A datasets
datasets["redocred_dev"]="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_dev.json"
datasets["redocred_test"]="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_test.json"
version="v1"
step="450"
paradigms="sentence_relations_fact"
mode="lr2e-4_deepspeed"
#mode="lr5e-5_gpu"

model_path="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/ckpt/${MODEL_NAME}/${paradigms}_${mode}/checkpoint-${step}"

for dataset_name in "${!datasets[@]}"; do
  DATA_PATH="${datasets[$dataset_name]}"
  EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/${MODEL_NAME}/${paradigms}_${mode}/${dataset_name}_${step}/"
  mkdir -p ${EVAL_SAVE_PATH}
  # 执行推理命令
  /workspace/xll/Anaconda3/envs/chatglm/bin/deepspeed --master_port 12347 --include localhost:0,1,2,3,4,5,6,7 src/inference.py \
    --model_name_or_path /workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2 \
    --adapter_name_or_path ${model_path} \
    --template ${MODEL_NAME} \
    --finetuning_type lora \
    --max_new_tokens 1024 \
    --do_sample true \
    --temperature 0.95 \
    --version ${version} \
    --top_p 0.6 \
    --data_path ${DATA_PATH} \
    --lora_test lora_${paradigms} \
    --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
done
