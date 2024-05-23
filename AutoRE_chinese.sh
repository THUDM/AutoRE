#!/bin/bash

# choice = [vicuna,chatglm3,mistral]
model=vicuna
# Base model, replace with the base model corresponding to model
#BASE_MODEL=AutoRE/checkpoints/Mistral-7B-Instruct-v0.2
#BASE_MODEL=AutoRE/checkpoints/THUDM6B3
BASE_MODEL=AutoRE/checkpoints/lmsys/vicuna-7b-v1.5
# Path where the QLoRA module is located
model_path="AutoRE/ckpt/${model}/"

# Set the extraction paradigm for AutoRE, refer to the paper for detailed meaning
# version choice = [D_F,D_RS_F,D_R_F,D_R_H_F,D_R_H_F_desc]
version="D_R_H_F_desc_chinese"

# lora_test corresponds to different relation extraction paradigms and the number of lora modules used, lora for a single, loras for 3 (the parameter volume of a single lora = the parameter volume of 3 loras)
# lora_test choice=[lora_D_F,lora_D_RS_F,lora_D_R_F,lora_D_R_H_F,lora_D_R_H_F_desc,lora_relation,lora_subject,lora_facts,loras_D_R_H_F_desc]
lora_test="loras_D_R_H_F_desc"

# If loras_D_R_H_F_desc or lora_relation (subject, facts) is specified, then the following 3 steps need to be further set, refer to the specific steps in the ckpt folder.
relation_step="490"
subject_step="1500"
fact_step="2600"

# When --inference, the DATA_PATH setting does not work, and the script performs inference on the user's data
# To test on the redocred dataset, remove the --inference below
#DATA_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/data/redocred/redocred_test.json"
DATA_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/data/other_source/hacred/test.json"
#DATA_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/data/redocred/redocred_dev.json"

# set the save_path to save test_result for redocred
#EVAL_SAVE_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/result/${model}/loras/redocred_test/"
EVAL_SAVE_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/result/${model}/loras_chinese/hacred_test/"
#EVAL_SAVE_PATH="/workspace/xll/AutoRE_GitHub/AutoRE/result/${model}/loras/redocred_dev/"
# Test user input
/workspace/xll/Anaconda3/envs/auto/bin/deepspeed --master_port 12347 --include localhost:5 inference_chinese.py \
  --model_name_or_path ${BASE_MODEL} \
  --adapter_name_or_path ${model_path} \
  --template ${model} \
  --data_path ${DATA_PATH} \
  --version ${version} \
  --do_sample true \
  --max_new_tokens 1024 \
  --temperature 0.95 \
  --top_p 0.6 \
  --lora_test ${lora_test} \
  --relation_step ${relation_step} \
  --subject_step ${subject_step} \
  --fact_step ${fact_step} \
  --save_path ${EVAL_SAVE_PATH}
# | tee -a log.log，可以将结果输出，但是会导致terminal的颜色无法显示
#  --inference \



