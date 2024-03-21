"""
Description: 
Author: dante
Created on: 2023/11/23
"""
import time
from typing import List, Dict
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from .template import *
from .basic import *


def get_split_model_vicuna():
    """
        将一个模型分割到不同的卡上
    :return:
    """

    def auto_configure_device_map(gpus: List[int]) -> Dict[str, int]:
        num_trans_layers = 32
        device_map = {'model.embed_tokens': gpus[1], "model.norm": gpus[1], "lm_head": gpus[2]}
        gpu_target_index = 0
        for i in range(num_trans_layers):
            device_map[f'model.layers.{i}'] = gpus[gpu_target_index]
            gpu_target_index += 1
            if gpu_target_index >= len(gpus):
                gpu_target_index = 0
        return device_map

    model_path = "/home/xll/nell162/FastChat/output_auto_kg/checkpoint-3300"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True,
    )
    from accelerate import dispatch_model
    device_map = auto_configure_device_map([0, 1, 2])
    model = dispatch_model(model, device_map=device_map)
    model.eval()
    return model, tokenizer


def get_vicuna_model(cuda_id, model_path):
    """
        加载模型，分配到指定的cuda_id
    :param cuda_id:
    :param model_path:
    :return:
    """
    start = time.time()
    cuda_device = torch.device(f"cuda:{cuda_id}") if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True,
    )
    model.to(cuda_device)
    model.eval()
    print(f"model load done, consume time {time.time() - start}")
    return model, tokenizer


def get_vicuna_lora_model(cuda_id, lora_path, base_model_path):
    """
        加载lora模型
    :param cuda_id:
    :param lora_path:
    :param base_model_path:
    :return:
    """
    start = time.time()
    cuda_device = torch.device(f"cuda:{cuda_id}") if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16
    )
    model = lora_model.merge_and_unload()
    print(f"LoRA model load done, consume time {time.time() - start}")
    model.to(cuda_device)
    model.eval()
    return model, tokenizer


def load_relation_subject_fact_lora(args, cuda_device, model_name):
    """
        这个用来测试三个LoRA联合起来的效果
    :param args:
    :param cuda_device:
    :param model_name:
    :return:
    """
    start = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    r_model = PeftModel.from_pretrained(
        base,
        os.path.join(args.model_path, f"relation/{model_name}/checkpoint-{args.relation_step}"),
        torch_dtype=torch.float16,
        adapter_name="relation_lora"
    )
    r_model = r_model.merge_and_unload()
    r_model.to(cuda_device)
    r_model.eval()
    print(f"relation model load done, consume time {time.time() - start}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    s_model = PeftModel.from_pretrained(
        base,
        os.path.join(args.model_path, f"subject/{model_name}/checkpoint-{args.subject_step}"),
        torch_dtype=torch.float16,
        adapter_name="subject_lora"
    )
    s_model = s_model.merge_and_unload()
    s_model.to(cuda_device)
    s_model.eval()
    print(f"subject model load done, consume time {time.time() - start}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    f_model = PeftModel.from_pretrained(
        base,
        os.path.join(args.model_path, f"facts/{model_name}/checkpoint-{args.fact_step}"),
        torch_dtype=torch.float16,
        adapter_name="fact_lora"
    )
    f_model = f_model.merge_and_unload()
    f_model.to(cuda_device)
    f_model.eval()
    print(f"fact model load done, consume time {time.time() - start}")
    return f_model, r_model, s_model


def vicuna_inference(model, tokenizer, text, max_new_tokens=1024, temperature=0.95, top_p=0.6, do_sample=False):
    """
        根据vicuna原来模型设置的推理
    :param max_new_tokens:
    :param model:
    :param tokenizer:
    :param text:
    :param temperature:
    :param top_p:
    :param do_sample:
    :return:
    """
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Human: {" \
             "text} ASSISTANT:"
    prompt = prompt.format(text=text)
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(model.device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs


def mistral_inference(model, tokenizer, text, max_new_tokens=1024, temperature=0.95, top_p=0.6, do_sample=False):
    """
        这个是为mistral 准备的inference
    :param model:
    :param tokenizer:
    :param text:
    :param max_new_tokens:
    :param temperature:
    :param top_p:
    :param do_sample:
    :return:
    """
    prompt = "<s>[INST] {message} [/INST]".format(message=text)
    pad_token_id = tokenizer.eos_token_id
    input_tensor = tokenizer([prompt])
    attention_mask = input_tensor.attention_mask
    input_ids = input_tensor.input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(model.device),
        attention_mask=torch.as_tensor(attention_mask).to(model.device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=pad_token_id,
        top_p=top_p,
        do_sample=do_sample
        ,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs


def llama_factory_inference(chat_model, query):
    response = ""
    for new_text in chat_model.stream_chat({"role": "user", "content": query}):
        response += new_text
    return response


def sentence_facts(args):
    """
        测试直接获取facts的效果
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence)
        ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
        print("ori_fact_list: ", ori_fact_list)
        facts = get_fixed_facts(ori_fact_list, sentence)
        for fact in facts:
            flag = 0
            for index, true_fact in enumerate(sample['same_fact_list']):
                if fact in true_fact:
                    flag = 1
                    if index not in fact_index:
                        right.append(fact)
                        fact_index.append(index)
            if not flag:
                wrong.append(fact)
        save = {"data_from": sample['data_from'], "sentence": sentence, "ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def sentence_relation_fact(args):
    """
        先抽取relation，再生成fact
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = []
        for r in ori_relations_list.split("\n"):
            r = r.lower().strip()
            if r not in relations_description:
                continue
            else:
                relations.append(r)
        relations = list(set(relations))
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "ori_relations": ori_relations_list,
            "relations": relations
        }
        for relation in relations:
            if with_relation_desc:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation))
            else:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation)
            ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
            print("relation: ", relation, " ori_fact_list: ", ori_fact_list)
            facts = get_fixed_facts(ori_fact_list, sentence)
            for fact in facts:
                flag = 0
                for index, true_fact in enumerate(sample['same_fact_list']):
                    if fact in true_fact:
                        flag = 1
                        if index not in fact_index:
                            right.append(fact)
                            fact_index.append(index)
                if not flag:
                    wrong.append(fact)
            save[relation] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def sentence_relation_subject_fact(args):
    """
        先抽取relation，再抽取subject，再抽取fact
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "ori_relations": ori_relations_list,
            "relations": relations
        }
        for relation in relations:
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            save[relation] = {"ori_subject_list": ori_subjects, "subject_list": entities}
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = get_fixed_facts(ori_fact_list, sentence, subject=subject)
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts, "data_from": sample['data_from']}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_relation(args):
    """
        对vicuna模型进行lora测试relation
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc
    for sample in tqdm(args.data):
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "ori_relations": ori_relations_list,
            "relations": relations,
            "right_relations": [r for r in relations if r in sample['relations']],
            "wrong_relations": [r for r in relations if r not in sample['relations']],
            "miss_relations": [r for r in sample['relations'] if r not in relations]
        }
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_ana_relation(args):
    """
        对vicuna模型进行lora测试relation
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc
    for sample in tqdm(args.data):
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_analysis_prompt = templates[template_version]['relation_template'].format(sentences=sentence)
        print(relations_analysis_prompt)
        relations_analysis = vicuna_inference(model, tokenizer, relations_analysis_prompt)
        print("relation analysis: ", relations_analysis)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence, relation_analysis=relations_analysis)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "relation_analysis": relations_analysis,
            "ori_relations": ori_relations_list,
            "relations": relations,
            "right_relations": [r for r in relations if r in sample['relations']],
            "wrong_relations": [r for r in relations if r not in sample['relations']],
            "miss_relations": [r for r in sample['relations'] if r not in relations]
        }
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_subject(args):
    """
        对vicuna模型进行lora测试subject
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data, desc=f"cuda id :{cuda_id}"):
        sentence = sample['passage']
        print(f"{cuda_id} sentence: ", sentence)
        right = {}
        wrong = {}
        for relation in sample['relations']:
            right[relation] = []
            wrong[relation] = []
            entity_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_entity_list = vicuna_inference(model, tokenizer, entity_prompt)
            print("entities: ", ori_entity_list)
            ori_entities = list(set(ori_entity_list.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            fact_index = []
            for entity in entities:
                flag = 0
                subject_facts = [facts for facts in sample['same_fact_list'] if facts[0][1] == relation]
                for index, true_fact in enumerate(subject_facts):
                    true_entities = list(set([fact[0] for fact in true_fact]))
                    if entity in true_entities:
                        flag = 1
                        if index not in fact_index:
                            right[relation].append(entity)
                            fact_index.append(index)
                if not flag:
                    wrong[relation].append(entity)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "relations": sample['relations'],
            "right_entities": right,
            "wrong_entities": wrong,
            "same_fact_list": sample['same_fact_list']
        }
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_ana_subject(args):
    """
        对vicuna模型进行lora测试subject
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data, desc=f"cuda id :{cuda_id}"):
        sentence = sample['passage']
        print(f"{cuda_id} sentence: ", sentence)
        right = {}
        wrong = {}
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "relations": sample['relations'],
        }
        for relation in sample['relations']:
            right[relation] = []
            wrong[relation] = []
            subject_analysis_prompt = templates[template_version]["entity_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation))
            subjects_analysis = vicuna_inference(model, tokenizer, subject_analysis_prompt)
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation),
                                                                                             subjects_analysis=subjects_analysis)
            ori_entity_list = vicuna_inference(model, tokenizer, subject_list_prompt)
            print("relation: ", relation, " subjects_analysis: ", subjects_analysis, "entities: ", ori_entity_list)
            ori_entities = list(set(ori_entity_list.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            save[relation] = {"subjects_analysis": subjects_analysis, "entity": entities}
            fact_index = []
            for entity in entities:
                flag = 0
                subject_facts = [facts for facts in sample['same_fact_list'] if facts[0][1] == relation]
                for index, true_fact in enumerate(subject_facts):
                    true_entities = list(set([fact[0] for fact in true_fact]))
                    if entity in true_entities:
                        flag = 1
                        if index not in fact_index:
                            right[relation].append(entity)
                            fact_index.append(index)
                if not flag:
                    wrong[relation].append(entity)
        save = {
            "right_entities": right,
            "wrong_entities": wrong,
            "same_fact_list": sample['same_fact_list']
        }
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_facts(args):
    """
        对vicuna模型进行lora测试subject
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        for relation in sample['relations']:
            save[relation] = {}
            entities = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = []
                for fact in ori_fact_list.split("\n"):
                    try:
                        fact = eval(fact)
                        if len(fact) != 3:
                            continue
                        facts.append(fact)
                    except:
                        continue
                fixed_facts = []
                for fact in facts:
                    fact[0] = subject
                    fact[2] = fact[2].strip()
                    if fact[2] not in sentence:
                        fixed_entity = sliding_window_fuzzy_match(fact[2], sentence)
                        if fixed_entity:
                            if fixed_entity not in sentence:
                                continue
                            else:
                                fact[2] = fixed_entity
                                fixed_facts.append(fact)
                    else:
                        fixed_facts.append(fact)
                facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_relation_facts(args):
    """
        对vicuna模型进行lora测试subject
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        for relation in sample['relations']:
            save[relation] = {}
            fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation,
                                                                                        description=relations_description.get(relation))
            ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
            print("relation: ", relation, " ori_fact_list: ", ori_fact_list)
            facts = get_fixed_facts(ori_fact_list, sentence)
            for fact in facts:
                flag = 0
                for index, true_fact in enumerate(sample['same_fact_list']):
                    if fact in true_fact:
                        flag = 1
                        if index not in fact_index:
                            right.append(fact)
                            fact_index.append(index)
                if not flag:
                    wrong.append(fact)
            save[relation] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_ana_facts(args):
    """
        对vicuna模型进行lora测试subject
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        for relation in sample['relations']:
            save[relation] = {}
            entities = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(model, tokenizer, fact_prompt)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " facts_analysis: ", facts_analysis, " ori_fact_list: ", ori_fact_list)
                facts = []
                for fact in ori_fact_list.split("\n"):
                    try:
                        fact = eval(fact)
                        if len(fact) != 3:
                            continue
                        facts.append(fact)
                    except:
                        continue
                fixed_facts = []
                for fact in facts:
                    fact[0] = subject
                    fact[2] = fact[2].strip()
                    if fact[2] not in sentence:
                        fixed_entity = sliding_window_fuzzy_match(fact[2], sentence)
                        if fixed_entity:
                            if fixed_entity not in sentence:
                                continue
                            else:
                                fact[2] = fixed_entity
                                fixed_facts.append(fact)
                    else:
                        fixed_facts.append(fact)
                facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"fact_analysis": facts_analysis, "ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def relation_subject_facts(args):
    """
        对vicuna模型进行lora测试整个pipeline
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    model, tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.model, args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc
    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = get_fixed_facts(ori_fact_list, sentence)
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_relation_subject_facts(args):
    """
        对vicuna模型进行lora测试整个pipeline
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    cuda_id, node, worker_num, data_path, save_path, template_version = args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    cuda_device = torch.device(f"cuda:{cuda_id}") if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model_name = "vicuna-7b-v1.5"
    f_model, r_model, s_model = load_relation_subject_fact_lora(args, cuda_device, model_name)
    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(r_model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(s_model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = vicuna_inference(f_model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = get_fixed_facts(ori_fact_list, sentence)
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_relation_subject_facts_test(args):
    """
        对vicuna模型进行lora测试整个pipeline
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    cuda_id, node, worker_num, data_path, save_path, template_version = args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model_name = "vicuna-7b-v1.5"
    cuda_device = "cuda:2"
    f_model, r_model, s_model = load_relation_subject_fact_lora(args, cuda_device, model_name)
    for line in open("./test_kgc.txt").readlines():
        sentence = line
        # while True:
        #     sentence = input("input a sentence:\n")
        #     if sentence == "quit":
        #         break
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(r_model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(s_model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            print("ori_entity: ", ori_entities)
            entities = get_fixed_entity(ori_entities, sentence)
            print(" entity: ", entities)
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = vicuna_inference(f_model, tokenizer, fact_list_prompt)
                facts = get_fixed_facts(ori_fact_list, sentence)
                print("     relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list, "fact_list: ", facts)
                print("=" * 50)


def lora_ana_relation_subject_facts_test(args):
    """
        对vicuna模型进行测试的代码，进行了数据分割
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    cuda_id, node, worker_num, data_path, save_path, template_version = args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model_name = "vicuna-7b-v1.5"
    cuda_device = "cuda:1"
    f_model, r_model, s_model = load_relation_subject_fact_lora(args, cuda_device, model_name)

    for line in open("./test_kgc.txt").readlines():
        sentence = line
        # while True:
        #     sentence = input("input a sentence:\n")
        #     if sentence == "quit":
        #         break
        print("sentence: ", sentence)
        relations_analysis_prompt = templates[template_version]['relation_template'].format(sentences=sentence)
        print(relations_analysis_prompt)
        relations_analysis = vicuna_inference(r_model, tokenizer, relations_analysis_prompt)
        print("relation analysis: ", relations_analysis)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence, relation_analysis=relations_analysis)
        ori_relations_list = vicuna_inference(r_model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = []
        for r in ori_relations_list.split("\n"):
            r = r.lower().strip()
            if r not in relations_description:
                continue
            else:
                relations.append(r)
        relations = list(set(relations))
        for relation in relations:
            subject_analysis_prompt = templates[template_version]["entity_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation))
            subjects_analysis = vicuna_inference(s_model, tokenizer, subject_analysis_prompt)
            print(" relation: ", relation)
            print(" subjects_analysis: ", subjects_analysis)
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation),
                                                                                             subjects_analysis=subjects_analysis)
            ori_subjects = vicuna_inference(s_model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(f_model, tokenizer, fact_prompt)
                print("     subject", subject)
                print("     facts_analysis", facts_analysis)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(f_model, tokenizer, fact_list_prompt)
                print("     ori_fact_list: ", ori_fact_list)
                facts = get_fixed_facts(ori_fact_list, sentence, subject)
                print("     fact_list: ", facts)
                print("=" * 100)


def lora_relation_subject_ana_facts(args):
    """
        对vicuna模型进行lora测试整个pipeline
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    tokenizer, cuda_id, node, worker_num, data_path, save_path, template_version, with_relation_desc = args.tokenizer, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version, args.with_relation_desc

    cuda_device = torch.device(f"cuda:{cuda_id}") if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model_name = "vicuna-7b-v1.5"
    f_model, r_model, s_model = load_relation_subject_fact_lora(args, cuda_device, model_name)

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(r_model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(s_model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = []
            for entity in ori_entities:
                if entity in sentence:
                    entities.append(entity.strip())
                else:
                    fixed_entity = sliding_window_fuzzy_match(entity, sentence).strip()
                    if fixed_entity:
                        if fixed_entity in sentence:
                            entities.append(fixed_entity)
                    else:
                        continue
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)

            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(f_model, tokenizer, fact_prompt)
                print("facts_analysis: ", facts_analysis)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(f_model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = get_fixed_facts(ori_fact_list, sentence)
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"fact_analysis": facts_analysis, "ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def vicuna_model_test_relation_subject_ana_fact(model, tokenizer, cuda_id, node, worker_num, data_path, save_path=None, template_version="version1"):
    """
        对vicuna模型进行测试的代码，进行了数据分割
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    try:
        processed_data = [json.loads(line) for line in open(f"{save_path}/predict.json").readlines()]
    except FileNotFoundError:
        processed_data = []
    processed_ids = set(item['data_from'] for item in processed_data)
    data = json.load(open(data_path))
    to_process = [sample for sample in data if sample['data_from'] not in processed_ids]
    data = split_data_by_cuda_id(to_process, cuda_id, node, worker_num)
    for sample in tqdm(data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = []
        for r in ori_relations_list.split("\n"):
            r = r.lower().strip()
            if r not in relations_description:
                continue
            else:
                relations.append(r)
        relations = list(set(relations))
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "ori_relations": ori_relations_list,
            "relations": relations
        }
        for relation in relations:
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation))
            ori_subjects = vicuna_inference(model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = []
            for entity in ori_entities:
                if entity in sentence:
                    entities.append(entity.strip())
                else:
                    fixed_entity = sliding_window_fuzzy_match(entity, sentence).strip()
                    if fixed_entity:
                        if fixed_entity in sentence:
                            entities.append(fixed_entity)
                    else:
                        continue
            save[relation] = {"ori_subject_list": ori_subjects, "subject_list": entities}
            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(model, tokenizer, fact_prompt)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = []
                for fact in ori_fact_list.split("\n"):
                    try:
                        fact = eval(fact)
                        if len(fact) != 3:
                            continue
                        facts.append(fact)
                    except:
                        continue
                fixed_facts = []
                for fact in facts:
                    fact[0] = subject
                    fact[2] = fact[2].strip()
                    if fact[2] not in sentence:
                        fixed_entity = sliding_window_fuzzy_match(fact[2], sentence)
                        if fixed_entity:
                            if fixed_entity not in sentence:
                                continue
                            else:
                                fact[2] = fixed_entity
                                fixed_facts.append(fact)
                    else:
                        fixed_facts.append(fact)
                facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"fact_analysis": facts_analysis, "ori_facts": ori_fact_list, "facts": facts, "data_from": sample['data_from']}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def vicuna_model_test_ana_relation_subject_fact(model, tokenizer, cuda_id, node, worker_num, data_path, save_path=None, template_version="version1"):
    """
        对vicuna模型进行测试的代码，进行了数据分割
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    try:
        processed_data = [json.loads(line) for line in open(f"{save_path}/predict.json").readlines()]
    except FileNotFoundError:
        processed_data = []
    processed_ids = set(item['data_from'] for item in processed_data)
    data = json.load(open(data_path))
    to_process = [sample for sample in data if sample['data_from'] not in processed_ids]
    data = split_data_by_cuda_id(to_process, cuda_id, node, worker_num)
    for sample in tqdm(data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_analysis_prompt = templates[template_version]['relation_template'].format(sentences=sentence)
        print(relations_analysis_prompt)
        relations_analysis = vicuna_inference(model, tokenizer, relations_analysis_prompt)
        print("relation analysis: ", relations_analysis)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence, relation_analysis=relations_analysis)
        ori_relations_list = vicuna_inference(model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = []
        for r in ori_relations_list.split("\n"):
            r = r.lower().strip()
            if r not in relations_description:
                continue
            else:
                relations.append(r)
        relations = list(set(relations))
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "relation_analysis": relations_analysis,
            "ori_relations": ori_relations_list,
            "relations": relations
        }
        for relation in relations:
            subject_analysis_prompt = templates[template_version]["entity_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation))
            subjects_analysis = vicuna_inference(model, tokenizer, subject_analysis_prompt)
            print("relation: ", relation, " subjects_analysis: ", subjects_analysis)
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation),
                                                                                             subjects_analysis=subjects_analysis)
            ori_subjects = vicuna_inference(model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = []
            for entity in ori_entities:
                if entity in sentence:
                    entities.append(entity.strip())
                else:
                    fixed_entity = sliding_window_fuzzy_match(entity, sentence).strip()
                    if fixed_entity:
                        if fixed_entity in sentence:
                            entities.append(fixed_entity)
                    else:
                        continue
            save[relation] = {"subject_analysis": subjects_analysis, "ori_subject_list": ori_subjects, "subject_list": entities}
            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(model, tokenizer, fact_prompt)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = []
                for fact in ori_fact_list.split("\n"):
                    try:
                        fact = eval(fact)
                        if len(fact) != 3:
                            continue
                        facts.append(fact)
                    except:
                        continue
                fixed_facts = []
                for fact in facts:
                    fact[0] = subject
                    fact[2] = fact[2].strip()
                    if fact[2] not in sentence:
                        fixed_entity = sliding_window_fuzzy_match(fact[2], sentence)
                        if fixed_entity:
                            if fixed_entity not in sentence:
                                continue
                            else:
                                fact[2] = fixed_entity
                                fixed_facts.append(fact)
                    else:
                        fixed_facts.append(fact)
                facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"fact_analysis": facts_analysis, "ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def vicuna_model_test_lora_relation_subject_facts_true_entity(args, template_version):
    """
        对vicuna模型进行lora测试subject，没啥用
    :param model:
    :param tokenizer:
    :param cuda_id:
    :param node:
    :param worker_num:
    :param data_path:
    :param save_path:
    :param template_version:
    :return:
    """
    cuda_device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model_name = "vicuna-7b-v1.5"
    f_model, r_model, s_model = load_relation_subject_fact_lora(args, cuda_device, model_name)
    for sample in tqdm(args.data):
        t_entities = list(set(entity for fact in sample['fact_list'] for entity in [fact[0], fact[2]]))
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
        }
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = vicuna_inference(r_model, tokenizer, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = []
        for r in ori_relations_list.split("\n"):
            r = r.lower().strip()
            if r not in relations_description:
                continue
            else:
                relations.append(r)
        relations = list(set(relations))
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = vicuna_inference(s_model, tokenizer, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = []
            for entity in ori_entities:
                if entity in sentence and entity in t_entities:
                    entities.append(entity.strip())
                else:
                    fixed_entity = sliding_window_fuzzy_match(entity, sentence).strip()
                    if fixed_entity:
                        if fixed_entity in sentence and entity in t_entities:
                            entities.append(fixed_entity)
                    else:
                        continue
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)

            for subject in entities:
                fact_prompt = templates[template_version]["fact_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                  description=relations_description.get(relation))
                facts_analysis = vicuna_inference(f_model, tokenizer, fact_prompt)
                print("facts_analysis: ", facts_analysis)
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation),
                                                                                            facts_analysis=facts_analysis)
                ori_fact_list = vicuna_inference(f_model, tokenizer, fact_list_prompt)
                print("relation: ", relation, " subject: ", subject, " ori_fact_list: ", ori_fact_list)
                facts = []
                for fact in ori_fact_list.split("\n"):
                    try:
                        fact = eval(fact)
                        if len(fact) != 3:
                            continue
                        facts.append(fact)
                    except:
                        continue
                fixed_facts = []
                for fact in facts:
                    fact[0] = subject
                    fact[2] = fact[2].strip()
                    if fact[2] not in sentence:
                        fixed_entity = sliding_window_fuzzy_match(fact[2], sentence)
                        if fixed_entity:
                            if fixed_entity not in sentence:
                                continue
                            else:
                                if fixed_entity in t_entities:
                                    fact[2] = fixed_entity
                                    fixed_facts.append(fact)
                    else:
                        if fact[2] in t_entities:
                            fixed_facts.append(fact)
                facts = [list(x) for x in set(tuple(x) for x in fixed_facts)]
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                right.append(fact)
                                fact_index.append(index)
                    if not flag:
                        wrong.append(fact)
                save[relation][subject] = {"fact_analysis": facts_analysis, "ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')
