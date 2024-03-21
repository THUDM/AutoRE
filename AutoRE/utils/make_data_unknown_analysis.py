import re
import os
import multiprocessing as mp
import multiprocessing
from tqdm import tqdm

from chatgpt_query import *

entity_type = ["PER", "NUM", "MISC", "LOC", "TIME", "ORG"]
relation_descript = json.load(open("../../relations_desc/relation_description.json"))




def gen_entity_type(sample, save_file):
    entities = [e['name'] for es in sample['vertexSet'] for e in es]
    sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents'])]])
    prompt = f"Given entity types: {entity_type}, and entities: {entities}, the sentence is {sentence}, you should decide which type the entity is.\n" \
             f"Your output must be:\n" \
             "{\n" \
             "\"entity\": entity type.\n" \
             "\"entity\": entity type.\n" \
             "..." \
             "}"
    for _ in range(5):
        try:
            result = eval(make_chat_request(prompt))
            if any(t not in entity_type for t in result.values()):
                continue
            for entity_list in sample['vertexSet']:
                for entity in entity_list:
                    entity['type'] = result[entity['name']]
            with open(save_file, "a") as f:
                print(f"{sample['index']} write to {save_file}")
                f.write(json.dumps(sample) + "\n")
                update_keys_file()
            break
        except:
            continue


def check_result():
    def are_same_fact_lists_equal(same_fact_list1, same_fact_list2):
        if len(same_fact_list1) != len(same_fact_list2):
            return False
        for sublist1 in same_fact_list1:
            if sublist1 not in same_fact_list2:
                return False
        for sublist2 in same_fact_list2:
            if sublist2 not in same_fact_list1:
                return False
        return True

    save = []
    ori_data = json.load(open("./redocred_train.json"))
    analysis_data = [json.loads(line) for line in open("./redocred_train_analysis.json").readlines()]
    for sample in analysis_data:
        for ori_sample in ori_data:
            if ori_sample['passage'] == sample['passage']:
                sample['index'] = ori_sample['index']
                sample['page_id'] = ori_data.index(ori_sample)
                if ori_sample['fact_list'] == sample['fact_list'] and are_same_fact_lists_equal(ori_sample['same_fact_list'], sample['same_fact_list']):
                    save.append(sample)
    with open("./redocred_train_analysis.json", "w") as f:
        for sample in save:
            f.write(json.dumps(sample) + "\n")


def make_redocred():
    """
    Turns data into Redocred-testable format
    :return:
    """
    source_file = "../../../public_data/augment/RE/unknown/unknown_analysis/"
    data = []
    for file in os.listdir(source_file):
        data += [json.loads(line) for line in open(os.path.join(source_file, file))][:50]

    for index, sample in enumerate(data):
        sample['index'] = index
    json.dump(data, open("./unknown/relations_unknown_analysis.json", 'w'), indent=4)
    data = []
    # 这里只取出后50个，前50个用于训练了
    for file in os.listdir(source_file):
        data += [json.loads(line) for line in open(os.path.join(source_file, file))][50:]
    output = []
    count = 0
    for index, sample in enumerate(tqdm(data)):
        passage = sample["passage"]
        entities = sample["entity"]
        # 这里是为了去除掉 新添加entity的引号
        for ent in entities:
            if passage.find("'" + ent + "'") != -1:
                passage = passage.replace("'" + ent + "'", ent)
        sample["passage"] = passage
        my_output = {"passage": sample["passage"]}
        my_output["title"] = sample["passage"].split()[0] + f"_{index}"
        fact_list = sample["fact_list"]
        entities = []
        for fact in fact_list:
            entities.extend([fact[0], fact[2]])
        if type(sample['entity']) == str:
            sample['entity'] = [sample['entity']]
        entities.append(sample['entity'][0])
        entities = list(set(entities))
        sentences = [re.findall(r"\b[\w-]+\b|[,.]", sample["passage"])]
        my_output["vertexSet"] = []
        my_output["sents"] = sentences
        for entity in entities:
            entity_tokens = re.findall(r"\b[\w-]+\b|[,.]", entity)
            # print(entities, entity, entity_tokens)
            target_len = len(entity_tokens)
            v = []
            for i in range(len(sentences[0])):
                if sentences[0][i:i + target_len] == entity_tokens:
                    v.append({
                        "type": "",
                        "pos": [i, i + target_len],
                        "name": entity,
                        "sent_id": 0
                    })
            if not v:
                entity_tokens = re.findall(r"\b\w+\b|[,.]", entity + "s")
                target_len = len(entity_tokens)
                v = []
                for i in range(len(sentences[0])):
                    if sentences[0][i:i + target_len] == entity_tokens:
                        v.append({
                            "type": "",
                            "pos": [i, i + target_len],
                            "name": entity,
                            "sent_id": 0
                        })
                if not v:
                    # print(sample)
                    # print(sentences[0], entity)
                    count += 1
                    # exit()
            my_output["vertexSet"].append(v)

        my_output['labels'] = []
        for fact in sample['fact_list']:
            for idx, v_list in enumerate(my_output["vertexSet"]):
                for v in v_list:
                    if v['name'] == fact[0]:
                        subject_index = idx
                    if v['name'] == fact[2]:
                        object_index = idx
            my_output['labels'].append({"h": subject_index, "r": fact[1], "t": object_index})
        my_output['index'] = index
        output.append(my_output)
    print(count)
    json.dump(output, open("./unknown/redocred_unknown_test.json", "w"), indent=4)


def make_redocred_formate(index, sample):
    """
    Turns data into Redocred-testable format
    :return:
    """
    my_output = {}
    my_output["title"] = sample["passage"].split()[0] + f"_{index}"
    fact_list = sample["fact_list"]
    entities = []
    for fact in fact_list:
        entities.extend([fact[0], fact[2]])
    entities.append(sample['entity'][0])
    entities = list(set(entities))
    sentences = [re.findall(r"\b[\w-]+\b|[,.]", sample["passage"])]
    my_output["vertexSet"] = []
    my_output["sents"] = sentences
    for entity in entities:
        entity_tokens = re.findall(r"\b[\w-]+\b|[,.]", entity)
        target_len = len(entity_tokens)
        v = []
        for i in range(len(sentences[0])):
            if sentences[0][i:i + target_len] == entity_tokens:
                v.append({
                    "type": "",
                    "pos": [i, i + target_len],
                    "name": entity,
                    "sent_id": 0
                })
        if not v:
            entity_tokens = re.findall(r"\b\w+\b|[,.]", entity + "s")
            target_len = len(entity_tokens)
            v = []
            for i in range(len(sentences[0])):
                if sentences[0][i:i + target_len] == entity_tokens:
                    v.append({
                        "type": "",
                        "pos": [i, i + target_len],
                        "name": entity + "s",
                        "sent_id": 0
                    })
            if not v:
                print(sample)
                print(sentences[0], entity)
                exit()
            else:
                if entity in sample['entity']:
                    sample['entity'] = entity + "s"
                else:
                    for fact in sample['fact_list']:
                        if entity == fact[0]:
                            fact[0] = fact[0] + "s"
                        if entity == fact[2]:
                            fact[2] = fact[2] + "s"

        my_output["vertexSet"].append(v)

    my_output['labels'] = []
    for fact in sample['fact_list']:
        for idx, v_list in enumerate(my_output["vertexSet"]):
            for v in v_list:
                if v['name'] == fact[0]:
                    subject_index = idx
                if v['name'] == fact[2]:
                    object_index = idx
        my_output['labels'].append({"h": subject_index, "r": fact[1], "t": object_index})
    sample['redocred_formate'] = my_output
    return sample


def process_file(file):
    data = json.load(open(os.path.join(target_file, file)))
    for sample in tqdm(data, desc=file.split(".")[0] + "-" + str(files.index(file))):
        if not sample['fact_list'] or not sample['redocred_formate']['labels']:
            exit()
        passage = sample["passage"]
        vertex_set = sample["redocred_formate"]["vertexSet"]
        relations_desc = relation_descript.get(sample['relations'])
        if "unknown_analysis" not in sample:
            prompt = f"Given the passage: {sample['passage']}, and relation description: {relations_desc}, " \
                     f"explain why the object of triple fact: ({sample['entity'][0]},{sample['relations']}, object) is \"unknown\".\n" \
                     f"The unknown means we can not get the object of triple fact: ({sample['entity'][0]},{sample['relations']}) based on the given passage."
            analysis = make_chat_request(prompt)
            sample["unknown_analysis"] = analysis
        all_types_non_empty = all(vertex["type"] != "" for vertices in vertex_set for vertex in vertices)
        if not all_types_non_empty:
            entities = [e['name'] for es in sample['redocred_formate']['vertexSet'] for e in es]
            prompt = f"Given entity types: {entity_type}, and entities: {entities}, the sentence is {passage}, you should decide which type the entity is.\n" \
                     f"Your output must be:\n" \
                     "{\n" \
                     "\"entity\": entity type.\n" \
                     "\"entity\": entity type.\n" \
                     "..." \
                     "}"
            for _ in range(10):
                try:
                    result = eval(make_chat_request(prompt))
                    if any(t not in entity_type for t in result.values()):
                        continue
                    for entity_list in sample['redocred_formate']['vertexSet']:
                        for entity in entity_list:
                            entity['type'] = result[entity['name']]
                    update_keys_file()
                    break
                except:
                    continue
        json.dump(data, open(os.path.join(target_file, file), "w"), indent=4)


def multiprocess_files(target_file):
    files = os.listdir(target_file)
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_file, files)


if __name__ == '__main__':
    data = []
    for file in os.listdir("../../../public_data/augment/RE/unknown/unknown_handcraft_new"):
        data += json.load(open(os.path.join("../../../public_data/augment/RE/unknown/unknown_handcraft_new", file)))[50:]
    json.dump(data, open("./unknown/redocred_unknown_test.json", "w"), indent=4)

    # source_file = "../../../public_data/augment/RE/unknown/unknown_handcraft/"
    # analysis_file = "../../../public_data/augment/RE/unknown/unknown_analysis/"
    # target_file = "../../../public_data/augment/RE/unknown/unknown_handcraft_new/"
    # files = os.listdir(target_file)
    # for file in files:
    #     print(file)
    #     process_file(file)

    # with mp.Pool(mp.cpu_count()) as pool:
    #     pool.map(process_file, files)
    # for file in tqdm(os.listdir(source_file)):
    #     data = json.load(open(os.path.join(source_file, file)))
    #     for sample in data:
    #         passage = sample["passage"]
    #         entities = sample["entity"]
    #         for ent in entities:
    #             if passage.find("'" + ent + "'") != -1:
    #                 passage = passage.replace("'" + ent + "'", ent)
    #         sample["passage"] = passage
    #         if type(sample['entity']) == str:
    #             sample['entity'] = [sample['entity']]
    #         sample = make_redocred_formate(sample['index'], sample)
    #     json.dump(data, open(os.path.join(target_file, file), "w"), indent=4)
    # multiprocess_files(target_file)
