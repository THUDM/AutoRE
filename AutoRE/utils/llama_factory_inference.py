"""
Description: 
Author: dante
Created on: 2023/11/23
"""
from template import *
from basic import *
from tqdm import tqdm
from termcolor import colored


def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


def llama_factory_inference(chat_model, query):
    response = chat_model.chat([{"role": "user", "content": query}])[0].response_text
    return response


def lora_D_F(args):
    """
        测试D_F的效果
    :return:
    """
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence)
        ori_fact_list = llama_factory_inference(model, fact_list_prompt)
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


def lora_D_F_for_test(args):
    """
        测试D_F的效果
    :return:
    """
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    clear()
    while True:
        print("AutoRE Loaded Done")
        sentence = input("input a document:")
        print(colored(f"Document: {sentence}\n", 'yellow'))
        fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence)
        print(colored(f'fact_list_prompt:\n{fact_list_prompt}\n', 'green'))
        ori_fact_list = llama_factory_inference(model, fact_list_prompt)
        facts = get_fixed_facts(ori_fact_list, sentence)
        print(colored(f'    Extracted Facts:{facts}\n', 'blue'))


def lora_D_R_F(args):
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = llama_factory_inference(model, relations_prompt)
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
            fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation)
            ori_fact_list = llama_factory_inference(model, fact_list_prompt)
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


def lora_D_R_F_for_test(args):
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    clear()
    while True:
        print("AutoRE Loaded Done")
        sentence = input("input a document:")
        print(colored(f"Document: {sentence}\n", 'yellow'))
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        print(colored(f'relations_prompt:\n{relations_prompt}\n', 'green'))
        ori_relations_list = llama_factory_inference(model, relations_prompt)
        relations = get_fixed_relation(ori_relations_list)
        print(colored(f"Extracted Relations:{relations}\n", 'blue'))
        for relation in relations:
            print(colored(f'    Relation: {relation}\n', 'green'))
            fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation))
            print(colored(f'         fact_list_prompt:{fact_list_prompt}\n', 'blue'))
            ori_fact_list = llama_factory_inference(model, fact_list_prompt)
            facts = get_fixed_facts(ori_fact_list, sentence)
            print(colored(f'         Extracted Facts:{facts}\n', 'blue'))


def lora_D_RS_F(args):
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version

    for sample in tqdm(args.data):
        wrong = []
        right = []
        fact_index = []
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = llama_factory_inference(model, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        save = {
            "data_from": sample['data_from'],
            "sentence": sentence,
            "ori_relations": ori_relations_list,
            "relations": relations
        }
        fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relations=relations)
        ori_fact_list = llama_factory_inference(model, fact_list_prompt)
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
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_D_RS_F_for_test(args):
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path
    clear()
    while True:
        print("AutoRE Loaded Done")
        sentence = input("input a document:")
        print(colored(f"Document: {sentence}\n", 'yellow'))
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        print(colored(f'relations_prompt:\n{relations_prompt}\n', 'green'))
        ori_relations_list = llama_factory_inference(model, relations_prompt)
        relations = get_fixed_relation(ori_relations_list)
        print(colored(f"Extracted Relations:{relations}\n", 'blue'))
        fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence)
        print(colored(f'         fact_list_prompt:{fact_list_prompt}\n', 'blue'))
        ori_fact_list = llama_factory_inference(model, fact_list_prompt)
        facts = get_fixed_facts(ori_fact_list, sentence)
        print(colored(f'         Extracted Facts:{facts}\n', 'blue'))


def lora_D_R_H_F(args):
    """
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
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
        ori_relations_list = llama_factory_inference(model, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = llama_factory_inference(model, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = llama_factory_inference(model, fact_list_prompt)
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
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def lora_D_R_H_F_for_test(args):
    """
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    clear()
    while True:
        print("AutoRE Loaded Done")
        sentence = input("input a document:")
        print(colored(f"Document: {sentence}\n", 'yellow'))
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        print(colored(f'relations_prompt:\n{relations_prompt}\n', 'green'))
        ori_relations_list = llama_factory_inference(model, relations_prompt)
        relations = get_fixed_relation(ori_relations_list)
        print(colored(f"Extracted Relations:{relations}\n", 'blue'))
        for relation in relations:
            print(colored(f'    Relation: {relation}\n', 'green'))
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            print(colored(f'        subject_list_prompt:{subject_list_prompt}\n', 'cyan'))
            ori_subjects = llama_factory_inference(model, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print(colored(f'        Extracted Entities: {entities}\n', 'cyan'))
            for subject in entities:
                print(colored(f'         Entity: {subject}\n', 'magenta'))
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                print(colored(f'         fact_list_prompt:{fact_list_prompt}\n', 'blue'))
                ori_fact_list = llama_factory_inference(model, fact_list_prompt)
                facts = get_fixed_facts(ori_fact_list, sentence)
                print(colored(f'         Extracted Facts:{facts}\n', 'blue'))

        clear()


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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version
    for sample in tqdm(args.data):
        sentence = sample['passage']
        print("sentence: ", sentence)
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        ori_relations_list = llama_factory_inference(model, relations_prompt)
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version

    for sample in tqdm(args.data, desc=f"cuda id :{cuda_id}"):
        sentence = sample['passage']
        print(f"{cuda_id} sentence: ", sentence)
        right = {}
        wrong = {}
        for relation in sample['relations']:
            right[relation] = []
            wrong[relation] = []
            entity_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_entity_list = llama_factory_inference(model, entity_prompt)
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


def lora_facts(args):
    """
        对模型进行lora测试subject
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
    model, cuda_id, node, worker_num, data_path, save_path, template_version = args.model, args.local_rank, args.node, args.worker_num, args.data_path, args.save_path, args.template_version

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
                ori_fact_list = llama_factory_inference(model, fact_list_prompt)
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
                save[relation][subject] = {"ori_facts": ori_fact_list, "facts": facts}
        miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        save["right_fact_list"] = right
        save["wrong_fact_list"] = wrong
        save["miss_fact_list"] = miss
        save["true_fact_list"] = sample['same_fact_list']
        with open(f"{save_path}/predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')


def loras_RHF_desc(args):
    """
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
    f_model, r_model, s_model = args.f_model, args.r_model, args.s_model

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
        ori_relations_list = llama_factory_inference(r_model, relations_prompt)
        print("relations: ", ori_relations_list)
        relations = get_fixed_relation(ori_relations_list)
        for relation in relations:
            save[relation] = {}
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            ori_subjects = llama_factory_inference(s_model, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print("ori_entity: ", ori_entities)
            print("entity: ", entities)
            for subject in entities:
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                ori_fact_list = llama_factory_inference(f_model, fact_list_prompt)
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


def loras_RHF_desc_for_test(args):
    """
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
    f_model, r_model, s_model = args.f_model, args.r_model, args.s_model
    clear()
    while True:
        print("AutoRE Loaded Done")
        sentence = input("input a document:")
        print(colored(f"Document: {sentence}\n", 'yellow'))
        relations_prompt = templates[template_version]["relation_list_template"].format(sentences=sentence)
        print(colored(f'relations_prompt:\n{relations_prompt}\n', 'green'))
        ori_relations_list = llama_factory_inference(r_model, relations_prompt)
        relations = get_fixed_relation(ori_relations_list)
        print(colored(f"Extracted Relations:{relations}\n", 'blue'))
        for relation in relations:
            print(colored(f'    Relation: {relation}\n', 'green'))
            subject_list_prompt = templates[template_version]["entity_list_template"].format(sentences=sentence, description=relations_description.get(relation), relation=relation)
            print(colored(f'        subject_list_prompt:{subject_list_prompt}\n', 'cyan'))
            ori_subjects = llama_factory_inference(s_model, subject_list_prompt)
            ori_entities = list(set(ori_subjects.split("\n")))
            entities = get_fixed_entity(ori_entities, sentence)
            print(colored(f'        Extracted Entities: {entities}\n', 'cyan'))
            for subject in entities:
                print(colored(f'         Entity: {subject}\n', 'magenta'))
                fact_list_prompt = templates[template_version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                            description=relations_description.get(relation))
                print(colored(f'         fact_list_prompt:{fact_list_prompt}\n', 'blue'))
                ori_fact_list = llama_factory_inference(f_model, fact_list_prompt)
                facts = get_fixed_facts(ori_fact_list, sentence)
                print(colored(f'         Extracted Facts:{facts}\n', 'blue'))

        clear()
