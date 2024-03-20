import sys

sys.path.append("../../../../../analysis_kg")
from public_code.utils.report_result import *
from public_code.utils.basic import *
from public_code.utils.chatgpt_query import *


def joint_sentence_fact(sample, save_path, relation_part=100):
    """
        给定真实的relation，然后获取fact
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    save = {
        "data_from": sample['data_from'],
        "sentence": sentences,
        "relations": sample['relations']
    }
    relations_list = sample['relations']
    relations = {r: relations_description.get(r) for r in relations_list}
    fact_list = []
    for desc in split_dict_into_parts(relations, relation_part):
        fact_list_prompt = f"The relation list is : {list(desc.keys())}.\n" \
                           f"Given a the passage : \"{sentences}\".\n" \
                           f"Derive all the triplet facts from the passage according to the given relations. \n" \
                           f"Your output format is as following:\n" \
                           f"[\"subject\",\"relation\",\"object\"]\n" \
                           f"[\"subject\",\"relation\",\"object\"]\n" \
                           f"...\n" \
                           f"The subject and object should be entity from the passage and the relation must be in {list(desc.keys())}.\n"
        ori_fact_list = make_chat_request(fact_list_prompt)
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


def joint_sentence_one_relation_fact(sample, save_path):
    """
        给定relation，然后逐个对真实的relation，获取fact
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    relations_list = sample['relations']
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
            ori_fact_list = make_chat_request(fact_list_prompt)
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


def joint_sentence_one_relation_subject_fact(sample, save_path):
    """
        给定relation，然后逐个对真实的relation，抽取出subject，再抽取出fact
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    relations_list = sample['relations']
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
                                  f"Derive all the entity from the passage that can serve as the subject of the {relation}.\n" \
                                  f"Your output format is as following:" \
                                  f"entity1\n" \
                                  f"entity2" \
                                  f"...\n" \
                                  f"The entities should all be from the passage.\n"
            ori_entities_list = make_chat_request(subject_list_prompt)
            ori_entities = list(set(ori_entities_list.split("\n")))
            entities = get_fixed_entity(ori_entities, sentences)
            save[relation] = {"ori_subject_list": ori_entities_list, "subject_list": entities}
            for subject in entities:
                fact_list_prompt = f"Given the relation: {relation} and specific description: {relations_description.get(relation)}.\n" \
                                   f"Now the passage is: \"{sentences}\".\n" \
                                   f"Derive all the triplet facts from the passage that take {subject} as subject. \n" \
                                   f"Your output format is as following:" \
                                   f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                   f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                   f"...\n" \
                                   f"The object should be entity from the passage.\n"
                ori_fact_list = make_chat_request(fact_list_prompt)
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


if __name__ == '__main__':
    test_file_path = "../../../../public_data/augment/RE/redocred/redocred_test.json"
    save_path = "./result_no_desc_given/joint_sentence_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_fact)
    cal_relations_result_lora_facts(file_path=save_path)

    save_path = "./result_no_desc_given/joint_sentence_one_relation_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_fact)
    cal_relations_result_lora_facts(file_path=save_path)

    save_path = "./result_no_desc_given/joint_sentence_one_relation_subject_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_subject_fact)
    cal_relations_result_lora_facts(file_path=save_path)
