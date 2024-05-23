"""
Description: 
Author: dante
Created on: 2023/10/11
"""
import os.path
from collections import defaultdict
from .basic import *

current_dir = os.path.dirname(os.path.abspath(__file__))


def cal_result_lora_relation(file_path):
    """
        Calculate the results of each relation
    :param file_path:
    :return:
    """
    datas = []
    for file in os.listdir(file_path):
        if "predict" in file:
            datas += [json.loads(line) for line in open(os.path.join(file_path, file)).readlines()]
    json.dump(datas, open(os.path.join(file_path, "all.json"), "w"), indent=4)
    relations_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
    for data in tqdm(datas):
        for relation in data["right_relations"] + data["miss_relations"]:
            relations_result[relation]['all'] += 1
        for relation in data["right_relations"]:
            relations_result[relation]['tp'] += 1
        for relation in data["wrong_relations"]:
            relations_result[relation]['fp'] += 1
        for relation in data["miss_relations"]:
            relations_result[relation]['miss'] += 1
    for relation in relations_description:
        if relation not in relations_result:
            continue
        right = relations_result[relation]['tp']
        wrong = relations_result[relation]['fp']
        recall = right / relations_result[relation]['all']
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        relations_result[relation]['recall'] = recall
        relations_result[relation]['precision'] = precision
        relations_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
    json.dump(relations_result, open(os.path.join(file_path, "relation_result.json"), "w"), indent=4)


def cal_result_lora_subject(file_path):
    """
        Calculate the results of each subject
    :param file_path:
    :return:
    """
    datas = []
    for file in os.listdir(file_path):
        if "predict.json" in file:
            datas += [json.loads(line) for line in open(os.path.join(file_path, file)).readlines()]
    json.dump(datas, open(os.path.join(file_path, "all.json"), "w"), indent=4)
    entity_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
    for data in tqdm(datas):
        for relation in data["right_entities"]:
            entity_result[relation]['tp'] += len(data["right_entities"][relation])
        for relation in data["wrong_entities"]:
            entity_result[relation]['fp'] += len(data["wrong_entities"][relation])
        for relation in data['relations']:
            entity_result[relation]['all'] += len([facts for facts in data['same_fact_list'] if facts[0][1] == relation])
    for relation in relations_description:
        if relation not in entity_result:
            continue
        right = entity_result[relation]['tp']
        wrong = entity_result[relation]['fp']
        recall = right / entity_result[relation]['all']
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        entity_result[relation]['recall'] = recall
        entity_result[relation]['precision'] = precision
        entity_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
    json.dump(entity_result, open(os.path.join(file_path, "relation_result.json"), "w"), indent=4)


def cal_result_lora_facts(file_path):
    """
        Calculate the results of facts
    :param file_path:
    :return:
    """
    if "dev" in file_path:
        true_relation_count = json.load(open(os.path.join(current_dir, "../data/redocred/redocred_dev_fact_count.json")))
    else:
        # true_relation_count = json.load(open(os.path.join(current_dir, "../data/redocred/redocred_test_fact_count.json")))
        true_relation_count = json.load(open(os.path.join(current_dir, "../data/other_source/hacred/test_fact_count.json")))
    datas = [json.loads(line) for line in open(os.path.join(file_path, "predict.json"), encoding='utf-8').readlines()]
    json.dump(datas, open(os.path.join(file_path, "all.json"), "w"), indent=4)
    relations_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0, "precision": 0, "f1": 0})
    for data in tqdm(datas):
        if "right_fact_list" in data:
            for fact in data["right_fact_list"]:
                relations_result[fact[1]]['tp'] += 1
        if "wrong_fact_list" in data:
            for fact in data["wrong_fact_list"]:
                relations_result[fact[1]]['fp'] += 1
        if "miss_fact_list" in data:
            for fact in data["miss_fact_list"]:
                relations_result[fact[0][1]]['miss'] += 1
    for relation in relations_description:
        if relation not in relations_result or relation not in true_relation_count:
            continue
        true_relation_count[relation] = relations_result[relation]['tp'] + relations_result[relation]['miss']
        relations_result[relation]['all'] = true_relation_count[relation]
        right = relations_result[relation]['tp']
        wrong = relations_result[relation]['fp']
        recall = right / true_relation_count[relation]
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        relations_result[relation]['recall'] = recall
        relations_result[relation]['precision'] = precision
        relations_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
    json.dump(relations_result, open(os.path.join(file_path, "relation_result.json"), "w"), ensure_ascii=False, indent=4)


def report_relations_result(file_path):
    """
        print the result
    :param file_path:
    :return:
    """
    data = json.load(open(os.path.join(file_path, "relation_result.json")))
    for key in ["all", "tp", "fp"]:
        for relation in relations_description:
            if relation not in data:
                print(0)
            else:
                print(data[relation][key])
        print("=" * 100)
