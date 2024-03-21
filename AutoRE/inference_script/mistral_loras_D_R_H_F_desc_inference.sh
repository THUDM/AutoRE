#!/bin/bash

model=mistral
BASE_MODEL=/workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2
model_path="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/ckpt/${model}/"
relation_step="3400"
subject_step="1890"
fact_step="9930"
version="v5"
declare -A datasets
datasets["redocred_test"]="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_test.json"
datasets["redocred_dev"]="/workspace/xll/analysis_kg/public_data/augment/RE/redocred/redocred_dev.json"

# 循环遍历数据集
for dataset_name in "${!datasets[@]}"; do
  DATA_PATH="${datasets[$dataset_name]}"
  EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/${model}/loras/${dataset_name}/"
  mkdir -p ${EVAL_SAVE_PATH}
  /workspace/xll/Anaconda3/envs/chatglm/bin/deepspeed --master_port 12347 --include localhost:6 src/inference.py \
    --model_name_or_path ${BASE_MODEL} \
    --adapter_name_or_path ${model_path} \
    --template ${model} \
    --finetuning_type lora \
    --version ${version} \
    --do_sample false \
    --temperature 0.95 \
    --top_p 0.6 \
    --data_path ${DATA_PATH} \
    --lora_test lora_relation_subject_fact \
    --relation_step ${relation_step} \
    --subject_step ${subject_step} \
    --fact_step ${fact_step} \
    --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
done
