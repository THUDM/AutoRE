#!/bin/bash
# this shell is for analysis for autore
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
version="D_R_H_F_desc_analysis"

# lora_test corresponds to different relation extraction paradigms and the number of lora modules used, lora for a single, loras for 3 (the parameter volume of a single lora = the parameter volume of 3 loras)
# lora_test choice=[lora_D_F,lora_D_RS_F,lora_D_R_F,lora_D_R_H_F,lora_D_R_H_F_desc,lora_relation,lora_subject,lora_facts,loras_D_R_H_F_desc]
lora_test="loras_D_R_H_F_desc"

# If loras_D_R_H_F_desc or lora_relation (subject, facts) is specified, then the following 3 steps need to be further set, refer to the specific steps in the ckpt folder.
relation_step="1200"
subject_step="5390"
fact_step="4430"

# When inference is true, the DATA_PATH setting does not work, and the script performs inference on the user's data
# To test on the redocred dataset, remove the --inference below
DATA_PATH="data/redocred/redocred_test.json"
#DATA_PATH="data/redocred/redocred_dev.json"

# Test user input
deepspeed --master_port 12347 --include localhost:1 inference_analysis.py \
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
  --inference true \
  --fact_step ${fact_step} | tee -a log.log
