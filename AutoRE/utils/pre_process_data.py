import os
from collections import defaultdict, OrderedDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from template import *
from chatgpt_query import *


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
        if "test" not in source_file:
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

    # 以下是尝试用analysis的方法
    # base_path = "../data/redocred/analysis_redocred"
    # files = ["redocred_train.json", "redocred_dev.json", "redocred_test.json"]
    # for file in files:
    #     save_path = f"{base_path}/{file.replace('.json', '_analysis.json')}"
    #     source_file = f"../data/redocred/{file}"
    #     make_redocred_data_parallel(save_path=save_path, source_file=source_file, func=gen_analysis, num_processes=100)
    # 我用手动的方式构建了unknown数据集，现在为unknown生成解释
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
