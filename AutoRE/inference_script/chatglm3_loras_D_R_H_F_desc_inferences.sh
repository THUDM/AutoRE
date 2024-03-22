#!/bin/bash

model=chatglm3
BASE_MODEL=/workspace/xll/checkpoints/THUDM6B3
model_path="ckpt/${model}/"
relation_step="3390"
subject_step="5000"
fact_step="7500"
version="v5"

declare -A datasets
datasets["redocred_test"]="data/redocred/redocred_test.json"
datasets["redocred_dev"]="data/redocred/redocred_dev.json"

# 循环遍历数据集
for dataset_name in "${!datasets[@]}"; do
  DATA_PATH="${datasets[$dataset_name]}"
  EVAL_SAVE_PATH="result/${model}/loras/${dataset_name}_3390/"
  mkdir -p ${EVAL_SAVE_PATH}

  deepspeed --master_port 12347 --include localhost:0,1,2,3,4,5,6,7 src/inference.py \
    --model_name_or_path ${BASE_MODEL} \
    --adapter_name_or_path ${model_path} \
    --template ${model} \
    --finetuning_type lora \
    --do_sample false \
    --version ${version} \
    --temperature 0.95 \
    --top_p 0.6 \
    --data_path ${DATA_PATH} \
    --lora_test lora_relation_subject_fact \
    --relation_step ${relation_step} \
    --subject_step ${subject_step} \
    --fact_step ${fact_step} \
    --save_path ${EVAL_SAVE_PATH} | tee -a ${EVAL_SAVE_PATH}/log.log
done
