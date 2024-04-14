"""
Description: 
Author: dante
Created on: 2024/1/5
"""

"""
Description: 
Author: dante
Created on: 2023/8/31
"""

from ..utils.chatgpt_query import *
from ..utils.report_result import *
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)


def get_model(test_model_type, cudaid, model_path):
    start = time.time()
    cuda_device = torch.device(f"cuda:{cudaid}") if torch.cuda.is_available() else "cpu"
    if test_model_type == "chatglm":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device=cuda_device)
    else:
        # vicuna or mistral
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True,
        )
    model.to(cuda_device)
    model.eval()
    print(f"model load done, consume time {time.time() - start}")
    return model, tokenizer


def inference(test_model_type, model, tokenizer, text):
    if test_model_type == "chatglm":
        outputs, history = model.chat(tokenizer, text, max_new_tokens=1024, do_sample=True, temperature=0.8, history=[])
    elif test_model_type == "vicuna":
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " \
                 "USER: {text} ASSISTANT:"
        prompt = prompt.format(text=text)
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).to(model.device),
            max_new_tokens=512,
            temperature=0.95,
            do_sample=True,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    elif test_model_type == "mistral":
        prompt = "<s>[INST] {message} [/INST]".format(message=text)
        pad_token_id = tokenizer.eos_token_id
        input_tensor = tokenizer([prompt])
        attention_mask = input_tensor.attention_mask
        input_ids = input_tensor.input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).to(model.device),
            attention_mask=torch.as_tensor(attention_mask).to(model.device),
            max_new_tokens=512,
            temperature=0.9,
            pad_token_id=pad_token_id,
            top_p=0.6,
            do_sample=True
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs


def joint_sentence_fact(args):
    """
        抽取出对应的fact
    :param args:
    :return:
    """
    test_model_type, model, tokenizer, data_path, save_path = args.test_model_type, args.model, args.tokenizer, args.data_path, args.save_path
    for sample in tqdm(args.data, desc=test_model_type + "_" + args.mode):
        fact_index = []
        wrong = []
        right = []
        sentences = sample['passage']
        print("sentence: ", sentences)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentences,
            "relations": sample['relations']
        }
        relations_list = list(relations_description.keys())
        fact_list_prompt = f"The relation list is : {relations_list}.\n" \
                           f"Given a the passage : \"{sentences}\".\n" \
                           f"Derive all the triplet facts from the passage according to the given relations. \n" \
                           f"Your output format is as following:\n" \
                           f"[\"subject\",\"relation\",\"object\"]\n" \
                           f"[\"subject\",\"relation\",\"object\"]\n" \
                           f"...\n" \
                           f"The subject and object should be entity from the passage and the relation must be in {relations_list}.\n"
        ori_fact_list = inference(test_model_type, model, tokenizer, fact_list_prompt)
        fact_list = get_fixed_facts(ori_fact_list, sentences)
        save['facts'] = fact_list
        for fact in fact_list:
            flag = 0
            for index, true_fact in enumerate(sample['same_fact_list']):
                if fact in true_fact:
                    flag = 1
                    if index not in fact_index:
                        right.append(fact)
                        fact_index.append(index)
            if not flag:
                wrong.append(fact)
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def joint_sentence_relation_fact(args):
    """
        抽取出relation，再抽取出fact
    :param args:
    :return:
    """
    test_model_type, model, tokenizer, data_path, save_path = args.test_model_type, args.model, args.tokenizer, args.data_path, args.save_path
    for sample in tqdm(args.data, desc=test_model_type + "_" + args.mode):
        fact_index = []
        wrong = []
        right = []
        sentences = sample['passage']
        args.sentences = sentences
        relations_list = get_relations(args)
        relations = {r: relations_description.get(r) for r in relations_list}
        save = {
            "data_from": sample['data_from'],
            "sentence": sentences,
            "relations": sample['relations'],
            "predict_relations": relations_list
        }
        if relations_list:
            fact_list = []
            for desc in split_dict_into_parts(relations, args.relation_part):
                fact_list_prompt = f"Given the relations: {list(desc.keys())}.\n" \
                                   f"Now the passage is: \"{sentences}\".\n" \
                                   f"Derive all the triplet facts from the passage. \n" \
                                   f"Your output format is as following:\n" \
                                   f"[\"subject\",\"relation\",\"object\"]\n" \
                                   f"[\"subject\",\"relation\",\"object\"]\n" \
                                   f"...\n" \
                                   f"The subject and object should be entity from the passage and the relation must be in {list(desc.keys())}.\n"
                ori_fact_list = inference(test_model_type, model, tokenizer, fact_list_prompt)
                facts = get_fixed_facts(ori_fact_list, sentences)
                fact_list.extend(facts)
            save['facts'] = fact_list
            for fact in fact_list:
                flag = 0
                for index, true_fact in enumerate(sample['same_fact_list']):
                    if fact in true_fact:
                        flag = 1
                        if index not in fact_index:
                            right.append(fact)
                            fact_index.append(index)
                if not flag:
                    wrong.append(fact)
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def joint_sentence_one_relation_fact(args):
    """
       抽取出relation，在逐个输入relation，抽取出fact
    :param args:
    :return:
    """
    test_model_type, model, tokenizer, data_path, save_path = args.test_model_type, args.model, args.tokenizer, args.data_path, args.save_path
    for sample in tqdm(args.data, desc=test_model_type + "_" + args.mode):
        fact_index = []
        wrong = []
        right = []
        sentences = sample['passage']
        args.sentences = sentences
        relations_list = get_relations(args)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentences,
            "relations": sample['relations'],
            "predict_relations": relations_list
        }
        if relations_list:
            for relation in relations_list:
                fact_list_prompt = f"Given the relation: {relation}.\n" \
                                   f"Now the passage is: \"{sentences}\".\n" \
                                   f"Derive all the triplet facts from the passage. \n" \
                                   f"Your output format is as following:" \
                                   f"[\"subject\",\"{relation}\",\"object\"]\n" \
                                   f"[\"subject\",\"{relation}\",\"object\"]\n" \
                                   f"...\n" \
                                   f"The subject and object should be entity from the passage.\n"
                ori_fact_list = inference(test_model_type, model, tokenizer, fact_list_prompt)
                save['ori_fact'] = ori_fact_list
                facts = get_fixed_facts(ori_fact_list, sentences)
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
                save[relation] = facts
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def joint_sentence_one_relation_subject_fact(args):
    """
        给relation的描述，抽取出relation，再逐个输入，抽取出subject，在抽取出fact
    :param args:
    :return:
    """
    test_model_type, model, tokenizer, data_path, save_path = args.test_model_type, args.model, args.tokenizer, args.data_path, args.save_path
    for sample in tqdm(args.data, desc=test_model_type + "_" + args.mode):
        fact_index = []
        wrong = []
        right = []
        sentences = sample['passage']
        args.sentences = sentences
        relations_list = get_relations(args)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentences,
            "relations": sample['relations'],
            "predict_relations": relations_list
        }
        if relations_list:
            for relation in relations_list:
                subject_list_prompt = f"Given the relation: {relation}.\n" \
                                      f"Now the passage is: \"{sentences}\".\n" \
                                      f"Derive all the entity from the passage that can serve as the subject of the {relation}. \n" \
                                      f"Your output format is as following:" \
                                      f"entity1\n" \
                                      f"entity2" \
                                      f"...\n" \
                                      f"The entities should all be from the passage.\n"
                ori_entities_list = inference(test_model_type, model, tokenizer, subject_list_prompt)
                ori_entities = list(set(ori_entities_list.split("\n")))
                entities = []
                for entity in ori_entities:
                    if entity and entity in sentences:
                        entities.append(entity.strip())
                    else:
                        fixed_entity = sliding_window_fuzzy_match(entity, sentences).strip()
                        if fixed_entity:
                            if fixed_entity in sentences:
                                entities.append(fixed_entity)
                        else:
                            continue
                save[relation] = {"ori_subject_list": ori_entities_list, "subject_list": entities}
                for subject in entities:
                    fact_list_prompt = f"Given the relation: {relation}.\n" \
                                       f"Now the passage is: \"{sentences}\".\n" \
                                       f"Derive all the triplet facts from the passage that take {subject} as subject. \n" \
                                       f"Your output format is as following:" \
                                       f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                       f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                       f"...\n" \
                                       f"The object should be entity from the passage.\n"
                    ori_fact_list = inference(test_model_type, model, tokenizer, fact_list_prompt)
                    facts = get_fixed_facts(ori_fact_list, sentences, subject=subject)
                    save[relation][subject] = {'ori_fact': ori_fact_list, "fact": facts}
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
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def get_relations(args):
    test_model_type, model, sentences, tokenizer = args.test_model_type, args.model, args.sentences, args.tokenizer
    relations_list = []
    for desc in split_dict_into_parts(relations_description, args.relation_part):
        prompt = f"Given passage: {sentences}, and relation list: {list(desc.keys())}." \
                 f"Check the passage, and find which relations can be derived from the passage.\n" \
                 f"Your output format is as following:\n" \
                 "relation1\n" \
                 "relation2\n" \
                 "...\n" \
                 "one example like:\n" \
                 f"country of citizenship\n" \
                 f"father\n" \
                 "The relations must be in the relation list.\n" \
                 "If no relation in the sentence, you should  only output:\n" \
                 "no relation"
        print("relation prompt: ", prompt)
        relations = inference(test_model_type, model, tokenizer, prompt)
        print("ori_relations: ", relations)
        relations = list(set(relations.split("\n")))
        print("predict_relations: ", relations)
        for r in relations:
            if r and r in relations_description:
                relations_list.extend(relations)
    return relations_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference script.")
    parser.add_argument("--local_rank", type=int, required=False, help="CUDA ID.")
    parser.add_argument("--test_model_type", type=str, required=False, help="the model to test")
    parser.add_argument("--model_path", type=str, required=False, help="Model path.")
    parser.add_argument("--data_path", type=str, required=False, help="data path.")
    parser.add_argument("--save_path", type=str, required=False, help="save path.")
    parser.add_argument("--mode", type=str, required=False, help="fact or relation fact or relation subject fact")
    parser.add_argument("--node", type=int, required=False, help="node_num", default=0)
    parser.add_argument("--worker_num", type=int, required=False, help="worker_num", default=1)
    parser.add_argument("--relation_part", type=int, required=False, help="dived relation", default=100)
    args = parser.parse_args()
    args.data = get_test_data(args)
    if args.data:
        args.model, args.tokenizer = get_model(args.test_model_type, args.local_rank, args.model_path)
        if args.mode == "joint_sentence_fact":
            joint_sentence_fact(args)
        elif args.mode == "joint_sentence_relation_fact":
            joint_sentence_relation_fact(args)
        elif args.mode == "joint_sentence_one_relation_fact":
            joint_sentence_one_relation_fact(args)
        elif args.mode == "joint_sentence_one_relation_subject_fact":
            joint_sentence_one_relation_subject_fact(args)
        cal_result_lora_facts(file_path=args.save_path)
