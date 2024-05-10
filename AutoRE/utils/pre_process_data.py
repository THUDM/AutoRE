import os
from collections import defaultdict, OrderedDict
from template import *
from chatgpt_query import *

filter_entity = ['it', 'he', 'she', 'they', 'its', ""]


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
    for index, data_type in enumerate(data_types):
        final_save = []
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


def lora_relation_analysis(source_file, save_file):
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
        if "test" not in source_file:
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["relation_template"].format(sentences=sentence),
                "input": "",
                "output": sample['relation_analysis'],
                "history": []

            }
            train_data.append(block_dict)
            global_id += 1

        ori_relations = sample['relations'].copy()
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence, relation_analysis=sample['relation_analysis']),
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
                "instruction": templates[version]["relation_list_template"].format(sentences=sentence, relation_analysis=sample['relation_analysis']),
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
                "instruction": templates[version]["relation_list_template"].format(sentences=sentence, relation_analysis=sample['relation_analysis']),
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


def lora_subject_analysis(source_file, save_file):
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
            if "test" not in source_file:
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["entity_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation)),
                    "input": "",
                    "output": sample['entity_analysis'][relation],
                    "history": [],
                }
                train_data.append(block_dict)
                global_id += 1

            ori_entity_list = entity_list.copy()
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["entity_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation),
                                                                                 subjects_analysis=sample['entity_analysis'][relation]),
                "input": "",
                "output": str("\n".join(entity_list)),
                "history": [],
            }
            train_data.append(block_dict)
            global_id += 1
            if "test" not in source_file:
                if len(entity_list) > 1 and "train" in source_file:
                    while True:
                        random.shuffle(entity_list)
                        if entity_list != ori_entity_list:
                            break
                    block_dict = {
                        "id": f"identity_{global_id}",
                        "instruction": templates[version]["entity_list_template"].format(sentences=sentence, relation=relation, description=relations_description.get(relation),
                                                                                         subjects_analysis=sample['entity_analysis'][relation]),
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


def lora_fact_analysis(source_file, save_file):
    """
        最后抽取fact
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    if "test" in source_file:
        data = random.sample(data, int(len(data) * 0.8))
    global_id = 0
    for sample in tqdm(data):
        sentence = sample['passage']
        for relation in sample['relations']:
            entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            for subject in entity_list:
                if "test" not in source_file:
                    block_dict = {
                        "id": f"identity_{global_id}",
                        "instruction": templates[version]["fact_template"].format(sentences=sentence, description=relations_description.get(relation), subject=subject,
                                                                                  relation=relation),
                        "input": "",
                        "output": sample['fact_analysis'][relation][subject],
                        "history": []
                    }
                    train_data.append(block_dict)
                    global_id += 1
                fact_list = [fact for fact in sample['fact_list'] if fact[1] == relation and fact[0] == subject]
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["fact_list_template"].format(sentences=sentence, description=relations_description.get(relation), subject=subject,
                                                                                   relation=relation, facts_analysis=sample['fact_analysis'][relation][subject]),
                    "input": "",
                    "output": str("\n".join([str(fact) for fact in fact_list])),
                    "history": []
                }
                train_data.append(block_dict)
                global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def gen_unknown_analysis(source_dir, save_dir):
    """
        生成为什么是unknown的解释
    :param sample:
    :param save_file:
    :return:
    """
    for relation in relations_description:
        data = json.load(open(os.path.join(source_dir, relation + ".json")))
        save_file = json.load(open(os.path.join(save_dir, relation + ".json")))
        for sample in data:
            relations_desc = relations_description.get(sample['relations'])
            prompt = f"Given the passage: {sample['passage']}, and relation description: {relations_desc}, " \
                     f"explain why the object of triple fact: ({sample['entity'][0]},{sample['relations'][0]}, object) is \"unknown\".\n" \
                     f"The unknown means we can not get the object of triple fact: ({sample['entity'][0]},{sample['relations']}) based on the given passage."
            analysis = relations_description(prompt)
            sample['unknown_analysis'] = analysis
            with open(save_file, "a") as f:
                print(f"{sample['index']} write to {save_file}")
                f.write(json.dumps(sample) + "\n")
                update_keys_file()


def lora_unknown_fact_analysis(source_file, save_file):
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
            entity_list = sample['entity']
            for subject in entity_list:
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["fact_template"].format(sentences=sentence, description=relations_description.get(relation), subject=subject,
                                                                              relation=relation),
                    "input": "",
                    "output": sample['unknown_analysis'],
                    "history": []
                }
                train_data.append(block_dict)
                global_id += 1
                relation = sample['relations']
                subject = sample['entity'][0]
                fact_list = [[subject, relation, "unknown"]]
                block_dict = {
                    "id": f"identity_{global_id}",
                    "instruction": templates[version]["fact_list_template"].format(sentences=sentence, description=relations_description.get(relation), subject=subject,
                                                                                   relation=relation, facts_analysis=sample['unknown_analysis']),
                    "input": "",
                    "output": str("\n".join([str(fact) for fact in fact_list])),
                    "history": []
                }
                train_data.append(block_dict)
                global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


def gen_analysis(sample, save_file):
    # 生成抽取过程的解释
    print(f"index: {sample['data_from']}")
    if "relation_analysis" not in sample or not sample['relation_analysis']:
        relations_desc = [relations_description.get(relation) for relation in sample['relations']]
        relations = ", ".join(sample['relations'])
        prompt = f"You are a relation analyzer.\n" \
                 f"Given passage: {sample['passage']}, the true relations that can be derived from the passage are: {relations}, every specific relation description " \
                 f"are as follows: {relations_desc}. Now give me an analysis why the relations can be derived from the passage base on the context. " \
                 f"It is required to be integrated into a paragraph, without line breaks, without numbers before each relation. " \
                 f"You need to give me the analysis based on the content of the passage. Be as short as possible, but explain clearly. " \
                 f"You can start like: According to the passage, the relations are: {relations}, the reasons are: ...\n" \
                 f"the relations names must all be lower case. And you should give every relation an explain. Do not leave out any relations.\n" \
                 f"Again you should not use any numbers before any relation. Do not use line breaks, give me just one paragraph." \
                 f"The most important is all relations can be derived from the passage, so do not say that no evidence support the relation, you should always analysis carefully. "
        analysis = make_chat_request(prompt)
        sample['relation_analysis'] = analysis.replace("\n\n", " ").replace("\n", " ")
    if "entity_analysis" not in sample:
        sample["entity_analysis"] = {}
    if "fact_analysis" not in sample:
        sample["fact_analysis"] = {}
    for relation in sample['relations']:
        if relation not in sample["entity_analysis"] or not sample["entity_analysis"][relation]:
            entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
            if not entity_list:
                continue
            entity_prompt = f"You are an expert in entity analysis.\n" \
                            f"Given a passage:\"{sample['passage']}\" and a relation: \"{relation}\", the description of this relation is: \"{relations_description.get(relation)}\". \n" \
                            f"Based on these, {entity_list} can sever as the subject of a triple fact of \"{relation}\": " \
                            f"Now give me an analysis why these entities can be considered as the subject of a triple fact of \"{relation}\". " \
                            f"It is required to be integrated into a paragraph, without line breaks, without numbers before any entity. " \
                            f"Be as short as possible, but explain clearly. " \
                            f"You can start like: Given the passage, the entities are: {', '.join(entity_list)}, the reasons are: ...\n" \
                            f"Again you should not use any numbers before any entity. Do not use line breaks, give me just one paragraph." \
                            f"The most important is all entities can serve as the subject, you should always analysis carefully. "
            sample["entity_analysis"][relation] = make_chat_request(entity_prompt).replace("\n\n", " ").replace("\n", " ")
        if relation not in sample["fact_analysis"] or not sample["fact_analysis"][relation]:
            sample["fact_analysis"][relation] = {}
        entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
        for entity in entity_list:
            if entity not in sample["fact_analysis"][relation] or not sample["fact_analysis"][relation][entity]:
                fact_list = [fact for fact in sample['fact_list'] if fact[1] == relation and fact[0] == entity]
            else:
                continue
            fact_prompt = f"You are a fact analysis expert.\n" \
                          f"Given a passage: \"{sample['passage']}\" and a relation description : \"{relations_description.get(relation)}\".\n" \
                          f"We extracted the facts as : {fact_list}. Now give me the analysis why these facts can be derived from the passage." \
                          f"Be as short as possible, but explain clearly. " \
                          f"You can start like: According to the passage, the facts are: {fact_list}, the reasons are: ...\n" \
                          f"Again you should not use any numbers before any fact. Do not use line breaks, give me just one paragraph.\n" \
                          f"The most important is all facts are right, you should always analysis carefully. "
            sample["fact_analysis"][relation][entity] = make_chat_request(fact_prompt).replace("\n\n", " ").replace("\n", " ")
    with open(save_file, "a") as f:
        print(f"{sample['index']} write to {save_file}")
        f.write(json.dumps(sample) + "\n")
        update_keys_file()


def clean_data(passages, data_type):
    """
        对passage进行去重和合并
    :param passages:
    :return:
    """
    passages.sort(key=lambda x: len(x['passage']))
    passage_dict = {}
    for passage in tqdm(passages, desc=f"clean_{data_type}"):
        p = passage['passage']
        if p not in passage_dict:
            passage_dict[p] = passage
        else:
            passage_dict[p]['fact_list'].extend(passage['fact_list'])
            passage_dict[p]['fact_list'] = list(set(tuple(fact) for fact in passage_dict[p]['fact_list']))
            for relation in passage['relations']:
                if relation not in passage_dict[p]['relations']:
                    passage_dict[p]['relations'].append(relation)
    filter_passage = list(passage_dict.values())
    return filter_passage


def process_fewrel(data_type, data_folder):
    """
    :return:
    """
    data = open(os.path.join(data_folder, "fewrel/fewrel_train.txt")).readlines() + open(os.path.join(data_folder, "fewrel/fewrel_val.txt")).readlines()
    save = []
    for index, sample in enumerate(tqdm(data, desc=data_type)):
        sample = json.loads(sample)
        sentence = " ".join(sample['token']).strip()
        head = " ".join(sample['token'][sample['h']["pos"][0]:sample['h']["pos"][1]])
        tail = " ".join(sample['token'][sample['t']["pos"][0]:sample['t']["pos"][1]])
        if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail:
            continue
        relation = pid_name.get(sample['relation'])
        if relation in relation_mapping[data_type]:
            relation = relation_mapping[data_type][relation]
        relations = [relation]
        fact_list = [[head, relation, tail]]
        for fact in fact_list:
            relation = fact[1]
            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in relations:
                    relations.append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in fact_list:
                    fact_list.append(new_fact)
        save.append({
            "index": index,
            "passage": sentence,
            "relations": relations,
            "fact_list": fact_list,
            "data_from": f"fewrel_{index}"
        })
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "fewrel/fewrel.json"), "w"), indent=4)
    relation_count(os.path.join(data_folder, "fewrel/fewrel.json"), os.path.join(data_folder, "fewrel/fewrel_relation_count.json"))


def process_nyt10(data_type, data_folder):
    """
    :return:
    """
    data = open(os.path.join(data_folder, "nyt10/nyt10_train.txt")).readlines() + open(os.path.join(data_folder, "nyt10/nyt10_test.txt")).readlines()
    save = []
    for index, sample in enumerate(tqdm(data, desc=data_type)):
        sample = json.loads(sample)
        sentence = sample['text'].strip()
        head = sample['h']['name']
        tail = sample['t']['name']
        if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail:
            continue
        relation = sample['relation'].split("/")[-1].replace("_", " ") if sample['relation'] != "NA" else sample['relation'].replace("_", " ")
        if relation in relation_mapping[data_type]:
            relation = relation_mapping[data_type][relation]
        relations = [relation]
        fact_list = [[head, relation, tail]]
        for fact in fact_list:
            relation = fact[1]
            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in relations:
                    relations.append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in fact_list:
                    fact_list.append(new_fact)
        save.append({
            "index": index,
            "passage": sentence,
            "relations": relations,
            "fact_list": fact_list,
            "data_from": f"nyt10_{index}"
        })
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "nyt10/nyt10.json"), "w"), indent=4)
    relation_count(os.path.join(data_folder, "nyt10/nyt10.json"), os.path.join(data_folder, "nyt10/nyt10_relation_count.json"))


def process_semeval(data_type, data_folder):
    '''
        数据不好用，relation很奇怪
    :return:
    '''
    data = open(os.path.join(data_folder, "semeval/semeval_train.txt")).readlines() + open(os.path.join(data_folder, "semeval/semeval_test.txt")).readlines() + open(
        os.path.join(data_folder, "semeval/semeval_val.txt")).readlines()
    save = []
    for index, sample in enumerate(tqdm(data, desc=data_type)):
        sample = json.loads(sample)
        sentence = " ".join(sample['token']).strip()
        head = sample['h']['name'] if "(e1,e2)" in sample['relation'] else sample['t']['name']
        tail = sample['t']['name'] if "(e1,e2)" in sample['relation'] else sample['h']['name']
        if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail:
            continue
        relation = sample['relation'].split("(")[0]
        if relation in relation_mapping[data_type]:
            relation = relation_mapping[data_type][relation]
        relations = [relation]
        fact_list = [[head, relation, tail]]
        for fact in fact_list:
            relation = fact[1]
            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in relations:
                    relations.append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in fact_list:
                    fact_list.append(new_fact)
        save.append({
            "index": index,
            "passage": sentence,
            'relations': relations,
            "fact_list": fact_list,
            "data_from": f"semeval_{index}"
        })
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "semeval/semeval.json"), "w"), indent=4)
    relation_count(os.path.join(data_folder, "semeval/semeval.json"), os.path.join(data_folder, "semeval/semeval_relation_count.json"))


def process_wiki(data_type, data_folder):
    """
    :return:
    """
    data = open(os.path.join(data_folder, "wiki/wiki20m/wiki20m_train.txt")).readlines() + open(os.path.join(data_folder, "wiki/wiki20m/wiki20m_test.txt")).readlines() + open(
        os.path.join(data_folder, "wiki/wiki80/wiki80_train.txt")).readlines() + open(os.path.join(data_folder, "wiki/wiki80/wiki80_test.txt")).readlines()
    save = []
    for index, sample in enumerate(tqdm(data, desc=data_type)):
        sample = json.loads(sample)
        sentence = " ".join(sample['token']).strip()
        head = " ".join(sample['token'][sample['h']["pos"][0]:sample['h']["pos"][1]])
        tail = " ".join(sample['token'][sample['t']["pos"][0]:sample['t']["pos"][1]])
        if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail:
            continue
        relation = sample['relation']
        if relation in relation_mapping[data_type]:
            relation = relation_mapping[data_type][relation]
        relations = [relation]
        fact_list = [[head, relation, tail]]
        for fact in fact_list:
            relation = fact[1]
            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in relations:
                    relations.append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in fact_list:
                    fact_list.append(new_fact)
        save.append({
            'index': index,
            "passage": sentence,
            "relations": relations,
            "fact_list": fact_list,
            "data_from": f"wiki_{index}"
        })
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "wiki/wiki.json"), "w"), indent=4)
    relation_count(os.path.join(data_folder, "wiki/wiki.json"), os.path.join(data_folder, "wiki/wiki_relation_count.json"))


def process_instruct(data_type, data_folder):
    """
    :return:
    """
    folder = os.path.join(data_folder, "instruct/RE")
    data = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "dev" in file or "train" in file:
                file_path = os.path.join(root, file)
                data += json.load(open(file_path))
    save = []
    for index, sample in enumerate(tqdm(data, desc="instruct")):
        sentence = sample['sentence'].strip()
        facts = sample['relations']
        fact_list = []
        relations = []
        skip_sample = False
        for fact in facts:
            head = fact['head']['name']
            tail = fact['tail']['name']
            relation = fact['type']
            if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail or relation == "NA":
                skip_sample = True
                break
            if relation in relation_mapping[data_type]:
                relation = relation_mapping[data_type][relation]
            relations.append(relation)
            fact_list.append([head, relation, tail])
            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in relations:
                    relations.append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in fact_list:
                    fact_list.append(new_fact)
        if skip_sample:
            continue
        if fact_list and relations:
            save.append({
                "index": index,
                "passage": sentence,
                "relations": list(set(relations)),
                "fact_list": fact_list,
                "data_from": f"instruct_{index}"
            })
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "instruct/instruct.json"), "w"), indent=4)
    relation_count(os.path.join(data_folder, "instruct/instruct.json"), os.path.join(data_folder, "instruct/instruct_relation_count.json"))


def process_trex(data_type, data_folder):
    """
        将原始对trex整理成需要的格式
    :return:
    """

    def process_trex_chunk(files, data_index):
        save = []
        for file in files:
            data = json.load(open(file))
            for index, sample in enumerate(data):
                sentence = sample['text']
                fact_list = []
                relations = []
                for fact in sample['triples']:
                    head = fact['subject']['surfaceform']
                    tail = fact['object']['surfaceform']
                    if head == tail:
                        continue
                    if head not in sentence or tail not in sentence:
                        continue
                    relation = fact['predicate']['uri'].split("/")[-1]
                    try:
                        relation = pid_name.get(relation)
                    except:
                        pass
                    if relation not in relations:
                        relations.append(relation)
                    if [head, relation, tail] not in fact_list:
                        fact_list.append([head, relation, tail])

                if fact_list and relations:
                    save.append({
                        'index': index,
                        "passage": sentence,
                        "relations": list(set(relations)),
                        "fact_list": fact_list,
                        "data_from": f"trex_{index}"
                    })
        os.makedirs(os.path.join(data_folder, "trex/TREx/tmp/"), exist_ok=True)
        json.dump(save, open(os.path.join(data_folder, f"trex/TREx/tmp/data_{data_index}.json"), "w"), indent=4)

    folder_path = os.path.join(data_folder, "trex/TREx")
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    final_data = []
    chunk_size = 5
    for i in tqdm(range(0, len(files), chunk_size), desc=data_type):
        chunk_files = files[i:i + chunk_size]
        process_trex_chunk(chunk_files, f"data_{i // chunk_size}")
    for i in tqdm(range(0, len(files), chunk_size)):
        chunk_file = os.path.join(data_folder, f"trex/TREx/tmp/data_data_{i // chunk_size}.json")
        final_data += json.load(open(chunk_file))
    for file in files:
        final_data += json.load(open(file))
    save = []
    for sample in final_data:
        skip_sample = False
        sentence = sample['passage']
        for fact in sample["fact_list"]:
            head = fact[0]
            relation = fact[1]
            tail = fact[2]
            if head.lower() in filter_entity or tail.lower() in filter_entity or head not in sentence or tail not in sentence or head == tail:
                skip_sample = True
                break
            if relation in relation_mapping[data_type]:
                sample['relations'].remove(relation)
                relation = relation_mapping[data_type][relation]
                sample['relations'].append(relation)
                sample['fact_list'].remove(fact)
                sample['fact_list'].append([head, relation, tail])

            if relation in inverse_relation_mapping:
                if inverse_relation_mapping[relation] not in sample['relations']:
                    sample['relations'].append(inverse_relation_mapping[relation])
                new_fact = [tail, inverse_relation_mapping[relation], head]
                if new_fact not in sample['fact_list']:
                    sample['fact_list'].append(new_fact)
        if skip_sample:
            continue
        save.append(sample)
    save = clean_data(save, data_type)
    json.dump(save, open(os.path.join(data_folder, "trex/trex.json", "w")), indent=4)
    # shutil.rmtree("../../public_data/augment/RE/trex/TREx/tmp/")


if __name__ == '__main__':
    # preprocess for redocred
    make_redocred_data(data_types=['train', 'dev', 'test'], source_path="../data/redocred/ori_redocred", save_path="../data/redocred")
    source_train = "../data/redocred/redocred_train.json"
    source_test = "../data/redocred/redocred_test.json"
    relation_count(source_file=source_test, save_file="../data/redocred/redocred_test_relation_count.json")
    fact_count(source_file=source_test, save_file="../data/redocred/redocred_test_fact_count.json")
    # make data for train_set and test_set for 1 lora
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

    # make train_set and test_set for 3 loras
    lora_relation(source_file=source_train, save_file=f"../data/loras/relation/train.json")
    lora_relation(source_file=source_train, save_file=f"../data/loras/relation/test.json")
    lora_subject(source_file=source_train, save_file=f"../data/loras/subject/train.json")
    lora_subject(source_file=source_test, save_file=f"../data/loras/subject/test.json")
    lora_fact(source_file=source_train, save_file=f"../data/loras/fact/train.json")
    lora_fact(source_file=source_test, save_file=f"../data/loras/fact/test.json")

    # 以上是论文中的内容，接下来是新的尝试，如果不需要，请使用本代码时候全部注释掉

    # 以下是尝试用analysis的方法

    # base_path = "../data/redocred/analysis_redocred"
    # files = ["redocred_train.json", "redocred_dev.json", "redocred_test.json"]
    # for file in files:
    #     save_path = f"{base_path}/{file.replace('.json', '_analysis.json')}"
    #     source_file = f"../data/redocred/{file}"
    #     make_redocred_data_parallel(save_path=save_path, source_file=source_file, func=gen_analysis, num_processes=100)

    # 我用手动的方式构建了unknown数据集，现在也为unknown生成解释
    # gen_unknown_analysis(source_dir="../data/redocred/unknown/unknown_handcraft", save_dir="../data/redocred/unknown/unknown_relations_analysis")
    # 再将结果整合起来
    input_directory = "../data/redocred/unknown/unknown_relations_analysis"
    output_file = os.path.join(input_directory, '..', 'relations_unknown_analysis.json')
    data = [json.loads(line) for fname in os.listdir(input_directory) if fname.endswith('.json') for line in open(os.path.join(input_directory, fname))]
    json.dump(data, open(output_file, 'w'), indent=4)

    # make analysis_set for 3 loras, test_set remains unchanged
    source_train = "../data/redocred/analysis_redocred/redocred_train_analysis.json"
    source_test = "../data/redocred/analysis_redocred/redocred_test_analysis.json"
    version = "D_R_H_F_desc_analysis"
    lora_relation_analysis(source_file=source_train, save_file=f"../data/loras_analysis/relation/train.json")
    lora_relation_analysis(source_file=source_test, save_file=f"../data/loras_analysis/relation/test.json")

    lora_subject_analysis(source_file=source_train, save_file=f"../data/loras_analysis/subject/train.json")
    lora_subject_analysis(source_file=source_test, save_file=f"../data/loras_analysis/subject/test.json")

    lora_unknown_fact_analysis(source_file="../data/redocred/unknown/relations_unknown_analysis.json", save_file=f"../data/loras_analysis/fact/train_unknown.json")
    lora_fact_analysis(source_file=source_train, save_file=f"../data/loras_analysis/fact/train.json")
    # 这里需要将两个train文件进行合并，如果不使用unknown，则不需要。
    data = [item for f in ["../data/loras_analysis/fact/train_unknown.json", "../data/loras_analysis/fact/train.json"] for item in json.load(open(f))]
    # 此时，train.json是两个文件的合并
    json.dump(data, open("../data/loras_analysis/fact/train.json", 'w'), indent=4)
    lora_fact_analysis(source_file=source_test, save_file=f"../data/loras_analysis/fact/test.json")

    # 收集数据集合，进行增强
    datasets = ["instruct", "nyt10", "fewrel", "wiki", "semeval", "trex"]
    data_folder = "../data/other_source"
    for dataset in datasets:
        if dataset == "trex":
            process_trex(dataset, data_folder)
        else:
            globals()[f"process_{dataset}"](dataset, data_folder)
