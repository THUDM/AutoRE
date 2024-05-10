from utils.report_result import *
from utils.chatgpt_query import *


def joint_sentence_fact(sample, save_path, given_relation=False, relation_part=100):
    """
        没有给定relation，直接将所有的relation输入到模板中，没有关系的描述
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
    relations_list = sample['relations'] if given_relation else list(relations_description.keys())
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


def joint_sentence_relation_fact(sample, save_path, given_relation=False, relation_part=20):
    """
        没有给定relation，需要先抽取出relation，并且将所有的relation输入到模板中，没有关系的描述
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    relations_list = sample['relations'] if given_relation else get_relations(sentences, relation_part)
    relations = {r: relations_description.get(r) for r in relations_list}
    save = {
        "data_from": sample['data_from'],
        "sentence": sentences,
        "relations": sample['relations'],
        "predict_relations": relations_list
    }
    if relations_list:
        fact_list = []
        for desc in split_dict_into_parts(relations, relation_part):
            fact_list_prompt = f"Given the relations: {list(desc.keys())}.\n" \
                               f"Now the passage is: \"{sentences}\".\n" \
                               f"Derive all the triplet facts from the passage. \n" \
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


def joint_sentence_one_relation_fact(sample, save_path, given_relation=False, relation_part=20):
    """
        抽取出relation，逐个输入relation，抽取出fact
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    relations_list = sample['relations'] if given_relation else get_relations(sentences, relation_part)
    save = {
        "data_from": sample['data_from'],
        "sentence": sentences,
        "relations": sample['relations'],
        "predict_relations": relations_list
    }
    if relations_list:
        for relation in relations_list:
            if not desc_type:
                fact_list_prompt = f"Given the relation: {relation}.\n" \
                                   f"Now the passage is: \"{sentences}\".\n" \
                                   f"Derive all the triplet facts from the passage. \n" \
                                   f"Your output format is as following:" \
                                   f"[\"subject\",\"{relation}\",\"object\"]\n" \
                                   f"[\"subject\",\"{relation}\",\"object\"]\n" \
                                   f"...\n" \
                                   f"The subject and object should be entity from the passage.\n"
            else:
                fact_list_prompt = f"Given the relation: {relation} and specific description: {relations_description.get(relation)}.\n" \
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


def joint_sentence_one_relation_subject_fact(sample, save_path, given_relation=False, relation_part=20):
    """
        抽取出relation，逐个输入relation，抽取出subject，抽取出fact
    :param args:
    :return:
    """
    fact_index = []
    wrong = []
    right = []
    sentences = sample['passage']
    relations_list = sample['relations'] if given_relation else get_relations(sentences, relation_part)
    save = {
        "data_from": sample['data_from'],
        "sentence": sentences,
        "relations": sample['relations'],
        "predict_relations": relations_list
    }
    if relations_list:
        for relation in relations_list:
            if not desc_type:
                subject_list_prompt = f"Given the relation: {relation}.\n" \
                                      f"Now the passage is: \"{sentences}\".\n" \
                                      f"Derive all the entity from the passage that can serve as the subject of the {relation}.\n" \
                                      f"Your output format is as following:" \
                                      f"entity1\n" \
                                      f"entity2" \
                                      f"...\n" \
                                      f"The entities should all be from the passage.\n"
            else:
                subject_list_prompt = f"Given the relation: {relation} and specific description: {relations_description.get(relation)}.\n" \
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
                if not desc_type:
                    fact_list_prompt = f"Given the relation: {relation} and specific description: {relations_description.get(relation)}.\n" \
                                       f"Now the passage is: \"{sentences}\".\n" \
                                       f"Derive all the triplet facts from the passage that take {subject} as subject. \n" \
                                       f"Your output format is as following:" \
                                       f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                       f"[\"{subject}\",\"{relation}\",\"object\"]\n" \
                                       f"...\n" \
                                       f"The object should be entity from the passage.\n"
                else:
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


def get_relations(sentences, relation_part):
    relations_list = []
    for desc in split_dict_into_parts(relations_description, relation_part):
        prompt = f"Given passage: {sentences}, and relation list: {list(desc.keys())}\n" \
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
        relations = make_chat_request(prompt)
        relations = list(set(relations.split("\n")))
        for r in relations:
            if r and r in relations_description:
                relations_list.extend(relations)
    return relations_list


if __name__ == '__main__':
    test_file_path = "../../data/redocred/redocred_test.json"
    desc_type = ""
    # 测试base_models_for_DocRE
    save_path = "./chatgpt/result_no_desc_no_given_few_shot/joint_sentence_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_fact)
    cal_result_lora_facts(file_path=save_path)

    # 测试不同的抽取范式的效果
    save_path = "./chatgpt/result_no_desc_no_given/joint_sentence_relations_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_relation_fact)
    cal_result_lora_facts(file_path=save_path)

    save_path = "./chatgpt/result_no_desc_no_given/joint_sentence_one_relation_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_fact)
    cal_result_lora_facts(file_path=save_path)

    save_path = "./chatgpt/result_no_desc_no_given/joint_sentence_one_relation_subject_fact"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_subject_fact)
    cal_result_lora_facts(file_path=save_path)

    # 测试加入wiki_desc和new_desc的效果

    save_path = "./chatgpt/result_desc_ori_given/joint_sentence_one_relation_fact"
    relations_description = json.load(open("../../data/relations_desc/wikidata_desc.json"))
    os.makedirs(save_path, exist_ok=True)
    desc_type = "wiki"
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_fact)
    cal_result_lora_facts(file_path=save_path)

    save_path = "./chatgpt/result_desc_new_given/joint_sentence_one_relation_subject_fact"
    relations_description = json.load(open("../../data/relations_desc/relation_description_redocred.json"))
    desc_type = "new"
    os.makedirs(save_path, exist_ok=True)
    make_redocred_data_parallel(source_file=test_file_path, save_path=save_path, func=joint_sentence_one_relation_subject_fact)
    cal_result_lora_facts(file_path=save_path)
