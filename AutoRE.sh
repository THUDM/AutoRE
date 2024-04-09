#!/bin/bash

# choice = [vicuna,chatglm3,mistral]
model=vicuna
# 基础模型，替换为model对应的基础模型
#BASE_MODEL=AutoRE/checkpoints/Mistral-7B-Instruct-v0.2
#BASE_MODEL=AutoRE/checkpoints/THUDM6B3
BASE_MODEL=AutoRE/checkpoints/lmsys/vicuna-7b-v1.5
# 加载QLoRA模块所在的路径
model_path="AutoRE/ckpt/${model}/"

# 设置AutoRE的抽取范式, 详细意义参考论文
# version choic = [D_F,D_RS_F,D_R_F,D_R_H_F,D_R_H_F_desc]
version="D_R_H_F_desc"

# lora_test 对应着不同的关系抽取范式，以及使用的lora模块数量，lora为单个，loras代表3个（单个lora参数量=3个loras参数量）
# lora_test choice=[lora_D_F,lora_D_RS_F,lora_D_R_F,lora_D_R_H_F,lora_D_R_H_F_desc,lora_relation,lora_subject,lora_facts,loras_D_R_H_F_desc]
lora_test="loras_D_R_H_F_desc"

# 如果指定了loras_D_R_H_F_desc或者lora_relation(subject,facts)，则需要进一步设置以下3个step，具体参考ckpt文件夹内的具体step。
relation_step="1200"
subject_step="5390"
fact_step="4430"

# 当inference 为true时，DATA_PATH设置不起作用，脚本进行用户的数据inference
# 如果需要对redocred数据集进行测试，将下面--inference去掉
DATA_PATH="data/redocred/redocred_test.json"
#DATA_PATH="data/redocred/redocred_dev.json"

# 测试用户输入
deepspeed --master_port 12347 --include localhost:1 inference.py \
  --model_name_or_path ${BASE_MODEL} \
  --adapter_name_or_path ${model_path} \
  --template ${model} \
  --finetuning_type lora \
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
