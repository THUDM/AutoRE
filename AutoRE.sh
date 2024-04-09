#!/bin/bash

# choice = [vicuna,chatglm3,mistral]
model=vicuna
# 基础模型，替换为model对应的基础模型
#BASE_MODEL=AutoRE/checkpoints/Mistral-7B-Instruct-v0.2
#BASE_MODEL=AutoRE/checkpoints/THUDM6B3
BASE_MODEL=AutoRE/checkpoints/lmsys/vicuna-7b-v1.5
# 加载QLoRA模块所在的地方
model_path="AutoRE/ckpt/${model}/"

# 切换模型后，step需要相应改变，具体参考ckpt文件夹内的具体step。
relation_step="1200"
subject_step="5390"
fact_step="4430"

# 该模式为RHF抽取范式，即AutoRE的抽取范式
version="D_R_H_F_desc"

deepspeed --master_port 12347 --include localhost:1 inference.py \
  --model_name_or_path ${BASE_MODEL} \
  --adapter_name_or_path ${model_path} \
  --template ${model} \
  --finetuning_type lora \
  --version ${version} \
  --do_sample false \
  --temperature 0.95 \
  --top_p 0.6 \
  --lora_test lora_relation_subject_fact \
  --relation_step ${relation_step} \
  --subject_step ${subject_step} \
  --fact_step ${fact_step} | tee -a log.log
