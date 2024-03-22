import os
from collections import defaultdict, OrderedDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
from template import *
from basic import *
import json


def refine_redocred_data():
    """
        对redocred data进行矫正，使得labels中的head/tail的name名称能够和句子中的匹配
    :return:
    """
    for type in ['train', 'dev', 'test']:
        data = json.load(open(f"../data/redocred/ori_redocred/{type}_revised.json"))
        new = []
        for index, sample in enumerate(data):
            if sample['labels']:
                sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents'])]])
                for fact in sample['labels']:
                    for h in sample['vertexSet'][fact['h']]:
                        if h['name'] not in sentence:
                            h['name'] = " ".join(sample['sents'][h['sent_id']][h['pos'][0]:h['pos'][1]])
                    for t in sample['vertexSet'][fact['t']]:
                        if t['name'] not in sentence:
                            t['name'] = " ".join(sample['sents'][t['sent_id']][t['pos'][0]:t['pos'][1]])
                new.append(sample)
        json.dump(new, open(f"../data/redocred/ori_redocred/{type}_revised_refined.json", "w"), indent=4)

def relation_count(source_file, save_file):
    """
        统计数据的relation分布情况
    :return:
    """
    data = json.load(open(source_file))
    relations_dict = defaultdict(int)
    for sample in data:
        for relation in sample['relations']:
            relations_dict[relation] += 1
    relations_dict = OrderedDict(sorted(relations_dict.items(), key=lambda x: x[1], reverse=True))
    json.dump(relations_dict, open(save_file, "w"), indent=4)

def fact_count(source_file, save_file):
    """
        统计数据的fact分布情况
    :return:
    """
    data = json.load(open(source_file))
    relations_dict = defaultdict(int)
    for sample in data:
        relations = [fact[0][1] for fact in sample['same_fact_list']]
        for relation in relations:
            relations_dict[relation] += 1
    relations_dict = OrderedDict(sorted(relations_dict.items(), key=lambda x: x[1], reverse=True))
    json.dump(relations_dict, open(save_file, "w"), indent=4)

def make_redocred_data(data_types, source_path, save_path):
    """
        将redocred数据处理成指定的格式
    :param data_types: 处理的数据类型
    :param source_path: redocred 数据所在路径文件夹
    :param save_path: 保存的文件夹
    :return:
    """
    refine_redocred_data()
    final_save = []
    for index, data_type in enumerate(data_types):
        data = json.load(open(os.path.join(source_path, f"{data_type}_revised_refined.json")))
        for page_id, sample in enumerate(data):
            fact_list = []
            same_fact_list = []
            relations = set()
            sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents'])]])
            for fact in sample['labels']:
                head_name = list(set([h['name'] for h in sample['vertexSet'][fact['h']]]))
                tail_name = list(set([t['name'] for t in sample['vertexSet'][fact['t']]]))
                relation = pid_name[fact['r']]
                same_fact = []
                for head in head_name:
                    for tail in tail_name:
                        relations.add(relation)
                        if (head, relation, tail) not in same_fact:
                            same_fact.append([head, relation, tail])
                        if (head, relation, tail) not in fact_list:
                            fact_list.append(
                                [head, relation, tail],
                            )
                same_fact_list.append(same_fact)
            save = {
                "index": index,
                "page_id": page_id,
                "passage": sentence,
                "relations": list(relations),
                "fact_list": fact_list,
                "same_fact_list": same_fact_list,
                "data_from": f"redocred_{data_type}"
            }
            final_save.append(save)
        with open(os.path.join(save_path, f"redocred_{data_type}.json"), "w") as f:
            json.dump(final_save, f, indent=4)


def fact(source_file, save_file):
    """
        D_F
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["fact_list_template"].format(sentences=sentence),
            "input": "",
            "output": str("\n".join([str(fact) for fact in sample['fact_list']])),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def relations_fact(source_file, save_file):
    """
        D_RS_F
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
            "input": "",
            "output": str(str("\n".join(sample['relations']))),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1

        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["fact_list_template"].format(sentences=sentence, relations=sample['relations']),
            "input": "",
            "output": str("\n".join([str(fact) for fact in sample['fact_list']])),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def relation_fact(source_file, save_file):
    """
        D_R_F
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
            "input": "",
            "output": str(str("\n".join(sample['relations']))),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
        for relation in sample['relations']:
            fact_list = [list(fact_tuple) for fact_tuple in set(tuple(fact) for fact in sample['fact_list'] if fact[1] == relation)]
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["fact_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation)),
                "input": "",
                "output": str("\n".join([str(fact) for fact in fact_list])),
                "history": []
            }
            train_data.append(block_dict)
            global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def relation_subject_fact(source_file, save_file):
    """
        D_R_H_F
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    try:
        data = [json.loads(line) for line in open(source_file).readlines()]
        json.dump(data, open(source_file, "w"), indent=4)
    except:
        data = json.load(open(source_file))
    if "test" in source_file:
        data = random.sample(data, int(len(data) * 0.1))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
            "input": "",
            "output": str(str("\n".join(sample['relations']))),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
        for relation in sample['relations']:
            entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["entity_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation)),
                "input": "",
                "output": str("\n".join(entity_list)),
                "history": []
            }
            train_data.append(block_dict)
            global_id += 1
            for subject in entity_list:
                fact_list = [list(fact_tuple) for fact_tuple in set(tuple(fact) for fact in sample['fact_list'] if fact[1] == relation)]
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["fact_list_template"].format(sentences=sentence, relation=relation, subject=subject,
                                                                                   description=relations_description.get(relation)),
                    "input": "",
                    "output": str("\n".join([str(fact) for fact in fact_list])),
                    "history": []
                }
                train_data.append(block_dict)
                global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def lora_relation(source_file, save_file):
    """
        先抽取relation
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in tqdm(data):
        sentence = sample['passage']
        if any(relation not in relations_description for relation in sample['relations']):
            continue
        ori_relations = sample['relations'].copy()
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
            "input": "",
            "output": str("\n".join(sample['relations'])),
            "history": []

        }
        train_data.append(block_dict)
        global_id += 1
        if len(sample['relations']) > 1 and "train" in source_file:
            while True:
                random.shuffle(sample['relations'])
                if sample['relations'] != ori_relations:
                    ori_relations2 = sample['relations'].copy()
                    break
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
                "input": "",
                "output": str("\n".join(sample['relations'])),
                "history": []
            }
            train_data.append(block_dict)
            global_id += 1
        if len(sample['relations']) > 2 and "train" in source_file:
            while True:
                random.shuffle(sample['relations'])
                if sample['relations'] != ori_relations and sample['relations'] != ori_relations2:
                    break
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
                "input": "",
                "output": str("\n".join(sample['relations'])),
                "history": []
            }
            train_data.append(block_dict)
            global_id += 1

    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def lora_subject(source_file, save_file):
    """
        接着抽取subject
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in tqdm(data):
        sentence = sample['passage']
        for relation in sample['relations']:
            entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            ori_entity_list = entity_list.copy()
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["entity_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation)),
                "input": "",
                "output": str("\n".join(entity_list)),
                "history": [],
            }
            train_data.append(block_dict)
            global_id += 1
            if len(entity_list) > 1 and "train" in source_file:
                while True:
                    random.shuffle(entity_list)
                    if entity_list != ori_entity_list:
                        break
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["entity_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation)),
                    "input": "",
                    "output": str("\n".join(entity_list)),
                    "history": [],
                }
                train_data.append(block_dict)
                global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def lora_fact(source_file, save_file):
    """
        最后抽取fact
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    if "test" in source_file:
        data = random.sample(data, int(len(data) * 0.5))
    global_id = 0
    for sample in tqdm(data):
        sentence = sample['passage']
        for relation in sample['relations']:
            entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            for subject in entity_list:
                fact_list = [fact for fact in sample['fact_list'] if fact[1] == relation and fact[0] == subject]
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["fact_list_template"].format(sentences=sentence, description=relations_description.get(relation), subject=subject,
                                                                                   relation=relation),
                    "input": "",
                    "output": str("\n".join([str(fact) for fact in fact_list])),
                    "history": []
                }
                train_data.append(block_dict)
                global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


if __name__ == '__main__':
    # 对redocred数据进行预处理
    make_redocred_data(data_types=['train', 'dev', 'test'], source_path="../data/redocred/ori_redocred", save_path="../data/redocred")
    source_train = "../data/redocred/redocred_train.json"
    source_test = "../data/redocred/redocred_test.json"
    relation_count(source_file=source_test, save_file="../data/redocred/redocred_train_relation_count.json")
    fact_count(source_file=source_test, save_file="../data/redocred/redocred_train_fact_count.json")
    # 制作各个抽取范式的训练和测试数据集
    version = "D_F"
    fact(source_file=source_train, save_file=f"../data/train/{version}/train.json")
    fact(source_file=source_test, save_file=f"../data/train/{version}/test.json")

    version = "D_RS_F"
    relations_fact(source_file=source_train, save_file=f"../data/train/{version}/train.json")
    relations_fact(source_file=source_test, save_file=f"../data/train/{version}/test.json")

    version = "D_R_F"
    relation_fact(source_file=source_train, save_file=f"../data/train/{version}/train.json")
    relation_fact(source_file=source_test, save_file=f"../data/train/{version}/test.json")

    version = "D_R_H_F"
    relation_subject_fact(source_file=source_train, save_file=f"../data/train/{version}/train.json")
    relation_subject_fact(source_file=source_test, save_file=f"../data/train/{version}/test.json")

    version = "D_R_H_F_desc"
    relation_subject_fact(source_file=source_train, save_file=f"../data/train/{version}/train.json")
    relation_subject_fact(source_file=source_test, save_file=f"../data/train/{version}/test.json")

    lora_relation(source_file=source_train, save_file=f"../data/loras/relation/train.json")
    lora_relation(source_file=source_train, save_file=f"../data/loras/relation/test.json")
    lora_subject(source_file=source_train, save_file=f"../data/loras/subject/train.json")
    lora_subject(source_file=source_test, save_file=f"../data/loras/subject/test.json")
    lora_fact(source_file=source_train, save_file=f"../data/loras/fact/train.json")
    lora_fact(source_file=source_test, save_file=f"../data/loras/fact/test.json")
