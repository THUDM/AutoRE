#!/bin/bash

declare -A model_paths
model_paths["mistral"]="/workspace/xll/checkpoints/Mistral-7B-Instruct-v0.2"
model_paths["vicuna"]="/workspace/xll/checkpoints/lmsys/vicuna-7b-v1.5"
model_paths["chatglm"]="/workspace/xll/checkpoints/THUDM6B3/"
DATA_PATH="../../data/redocred/redocred_test.json"
modes=("joint_sentence_fact")
model_order=("chatglm" "vicuna" "mistral")
for model in "${model_order[@]}"; do
  model_path="${model_paths[$model]}"
  for mode in "${modes[@]}"; do
    EVAL_SAVE_PATH="/workspace/xll/analysis_kg/public_code/inference/baseline/${model}/no_desc_no_given/${mode}"
    mkdir -p "${EVAL_SAVE_PATH}"
    ARGS="base_models_for_DocRE.py \
          --model_path ${model_path} \
          --test_model_type ${model} \
          --mode ${mode} \
          --data_path ${DATA_PATH} \
          --save_path ${EVAL_SAVE_PATH}"
    run_cmd="deepspeed --master_port 12347 --include localhost:3 ${ARGS}"
    eval ${run_cmd} 2>&1 | tee -a "${EVAL_SAVE_PATH}/log.log"
  done
done
