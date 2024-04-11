"""
Description: 
Author: dante
Created on: 2023/10/18
"""
import itertools
import argparse
import multiprocessing
import os
import torch
from filelock import FileLock
from fuzzywuzzy import fuzz
import csv
import json

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
pid_name = json.load(open(os.path.join(current_dir, "../data/relations_desc/relation_pid_name.json")))
relations_description = json.load(open(os.path.join(current_dir, "../data/relations_desc/relation_description_redocred.json")))
inverse_relation_mapping = {
    "has part": "part of",
    "part of": "has part",
    "participant": "participant in",
    "participant in": "participant",
    "follows": "followed by",
    "followed by": "follows",
    "capital": "capital of",
    "capital of": "capital",
    "replaced by": "replaces",
    "replaces": "replaced by",
    "father": "child",
    "mother": "child",
    "spouse": "spouse",
    "members": "member of",
    "member of": "members",
    "ethnic group": "ethnicity of people",
    "ethnicity of people": "ethnic group",
    "sports team of location": "sports team location of teams",
    "sports team location of teams": "sports team of location"
}


def get_params():
    """
        获取参数
    :return:
    """
    parser = argparse.ArgumentParser(description="Run inference script.")
    parser.add_argument("--version", type=str, required=False, default="", help="D_F,D_RS_F,D_R_F,D_R_H_F,D_R_H_F_desc, to determine the template in template.py")
    parser.add_argument("--node", type=int, required=False, help="node_num", default=0)
    parser.add_argument("--worker_num", type=int, required=False, help="worker_num", default=1)
    parser.add_argument("--local_rank", type=int, required=False, help="CUDA ID.", default="0")
    parser.add_argument("--lora_test", type=str, required=False,
                        help="test different re paradigms,lora_D_F,lora_D_RS_F,lora_D_R_F,lora_D_R_H_F,lora_D_R_H_F_desc,lora_relation,lora_subject,lora_facts,loras_D_R_H_F_desc",
                        default="relation")
    parser.add_argument("--relation_step", type=str, required=False, default="700")
    parser.add_argument("--subject_step", type=str, required=False, default="1950")
    parser.add_argument("--fact_step", type=str, required=False, default="1550")
    parser.add_argument("--data_path", type=str, required=False, help="re_docred data path.")
    parser.add_argument("--model_name_or_path", type=str, required=False, help="base model for lora")
    parser.add_argument("--save_path", type=str, required=False, help="lora model save path.")
    parser.add_argument("--adapter_name_or_path", type=str, required=False, help="lora ckpt path, for inference or test")
    parser.add_argument("--do_sample", type=str, required=False)
    parser.add_argument("--temperature", type=str, required=False)
    parser.add_argument("--top_p", type=str, required=False)
    parser.add_argument("--top_k", type=str, required=False)
    parser.add_argument("--template", type=str, required=True, help="vicuna or chatglm3 or mistral, use for llmtuner")
    parser.add_argument("--max_new_tokens", type=int, required=False)
    parser.add_argument("--inference", required=False, action="store_true", help="inference=false means test re_docred data, otherwise test your input data.")
    args = parser.parse_args()
    return args


def get_test_data(args):
    """
        获取test的data
    :param cuda_id:
    :param data_path:
    :param node:
    :param save_path:
    :param worker_num:
    :return:
    """
    cuda_id, data_path, node, save_path, worker_num = args.local_rank, args.data_path, args.node, args.save_path, args.worker_num
    processed_data = []
    seen_passages = set()
    lock_path = f"{save_path}/predict.json.lock"
    with FileLock(lock_path):
        try:
            with open(f"{save_path}/predict.json", "r") as file:
                for line in file.readlines():
                    try:
                        data = json.loads(line)
                        if data['sentence'] not in seen_passages:
                            processed_data.append(data)
                            seen_passages.add(data['sentence'])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        with open(f"{save_path}/predict.json", "w") as file:
            for item in processed_data:
                file.write(json.dumps(item) + "\n")

    processed_ids = set(item['data_from'] for item in processed_data)
    data = json.load(open(data_path))
    to_process = [sample for sample in data if sample['data_from'] not in processed_ids]
    print(f"total: {len(data)}, remain {len(to_process)} to process")
    if to_process:
        data = split_data_by_cuda_id(to_process, cuda_id, node, worker_num)
        return data
    else:
        return []


def get_wikidata_desc():
    csv_file_path = '../../data/relations_desc/wikidata-properties.csv'
    data_dict = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['label'] not in relations_description:
                continue
            data_dict[row['label']] = row['description']
    with open('wikidata_desc.json', mode='w', encoding='utf-8') as jsonfile:
        json.dump(data_dict, jsonfile, ensure_ascii=False, indent=4)


def split_dict_into_parts(dictionary, num_elements_per_part):
    """
    Split the dict type data into parts, with each part having num_elements_per_part elements.
    """
    keys = list(dictionary.keys())
    num_keys = len(keys)
    parts = []
    start_idx = 0
    while start_idx < num_keys:
        end_idx = min(start_idx + num_elements_per_part, num_keys)
        part_keys = keys[start_idx:end_idx]
        part = {key: dictionary[key] for key in part_keys}
        parts.append(part)
        start_idx = end_idx
    return parts


def sliding_window_fuzzy_match(entity, text):
    """
    :param entity:
    :param text:
    :return:
    """
    target_list = entity.split()
    max_similarity = 0
    best_match = ""
    target_combinations = list(itertools.combinations(target_list, len(target_list)))
    words = text.split()
    for window_size in range(len(target_list), len(words) + 1):
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            for target_combination in target_combinations:
                target_string = " ".join(target_combination)
                similarity_ratio = fuzz.ratio(window, target_string)
                if similarity_ratio > max_similarity:
                    if window.startswith(",") or window.startswith("."):
                        continue
                    if window.endswith(",") or window.endswith("("):
                        window = window[:-1]
                    max_similarity = similarity_ratio
                    best_match = window
    if max_similarity > 60:
        return best_match.strip()
    else:
        return ""


def split_data_by_cuda_id(evaluation_data, cuda_id, node=0, worker_num=1):
    """
        根据node和cuda_id对数据进行划分
    :param evaluation_data:
    :param cuda_id:
    :param node:
    :param worker_num:
    :return:
    """
    gpus_count = torch.cuda.device_count()
    total_cards = worker_num * gpus_count
    global_cuda_id = node * gpus_count + cuda_id
    data_size = len(evaluation_data)
    data_per_card, remainder = divmod(data_size, total_cards)
    start_idx = global_cuda_id * data_per_card
    if global_cuda_id < remainder:
        start_idx += global_cuda_id
        data_per_card += 1
    else:
        start_idx += remainder
    end_idx = start_idx + data_per_card
    cuda_data = evaluation_data[start_idx:end_idx]
    print(len(cuda_data))
    return cuda_data


def get_fixed_relation(ori_relations_list):
    """
        对relation进行修正
    :param ori_relations_list:
    :return:
    """
    relations = []
    for r in ori_relations_list.split("\n"):
        r = r.strip()
        if r not in relations_description:
            continue
        else:
            relations.append(r)
            if r in inverse_relation_mapping:
                relations.append(inverse_relation_mapping.get(r))
    relations = list(set(relations))
    return relations


def get_fixed_entity(ori_entities, sentence):
    """
        将entity进行修正
    :param ori_entities:
    :param sentence:
    :return:
    """
    entities = []
    for entity in ori_entities:
        if entity and entity.strip() in sentence:
            entities.append(entity.strip())
        else:
            fixed_entity = sliding_window_fuzzy_match(entity.strip(), sentence).strip()
            if fixed_entity:
                if fixed_entity.strip() in sentence:
                    entities.append(fixed_entity.strip())
            else:
                continue
    entities = list(set(entities))
    return entities


def get_fixed_facts(ori_fact_list, sentences, subject=None):
    """
        修正entity
    :param ori_fact_list:
    :param sentences:
    :param subject:
    :return:
    """
    facts = []
    for fact in ori_fact_list.split("\n"):
        fact = fact.strip()
        try:
            fact = eval(fact)
            if type(fact) != list or not all(isinstance(item, str) for item in fact):
                continue
            if len(fact) != 3 or not fact[0] or not fact[2] or fact[0] == fact[2]:
                continue
            if fact not in facts:
                facts.append(fact)
        except:
            print("error fact: ", fact)
            continue
    fixed_facts = []

    for fact in facts:
        if fact[1] not in relations_description:
            continue
        fact[0] = fact[0].strip()
        fact[2] = fact[2].strip()
        if subject:
            fact[0] = subject
            check_index = [2]
        else:
            check_index = [0, 2]
        for i in check_index:
            if fact[i] not in sentences:
                fixed_entity = sliding_window_fuzzy_match(fact[i], sentences)
                if fixed_entity and fixed_entity in sentences:
                    fact[i] = fixed_entity
                else:
                    break
        if fact[0] in sentences and fact[2] in sentences and fact not in fixed_facts:
            fixed_facts.append(fact)
    facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
    return facts


def make_redocred_data_parallel(source_file, save_path, func, num_processes=None):
    """
        chatgpt 并行请求
    :param source_file:
    :param save_file:
    :param func:
    :param num_processes:
    :return:
    """
    processed_data = []
    seen_passages = set()
    lock_path = f"{save_path}/predict.json.lock"
    with FileLock(lock_path):
        try:
            with open(f"{save_path}/predict.json", "r") as file:
                for line in file.readlines():
                    try:
                        data = json.loads(line)
                        if data['sentence'] not in seen_passages:
                            processed_data.append(data)
                            seen_passages.add(data['sentence'])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        with open(f"{save_path}/predict.json", "w") as file:
            for item in processed_data:
                file.write(json.dumps(item) + "\n")
    data = json.load(open(source_file))
    processed_ids = set(item['data_from'] for item in processed_data)
    to_process_data = [sample for sample in data if sample['data_from'] not in processed_ids]
    if not num_processes:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    def update_progress(_):
        pbar.update()

    print(f"model: {func}, total: {len(data)}, left: {len(to_process_data)}")
    with tqdm(total=len(to_process_data)) as pbar:
        for sample in to_process_data:
            pool.apply_async(func, (sample, save_path), callback=update_progress)
        pool.close()
        pool.join()
