"""
Description: 
Author: dante
Created on: 2023/10/11
"""
import os.path
from collections import defaultdict
from basic import *

current_dir = os.path.dirname(os.path.abspath(__file__))


def cal_relations_result_lora_relation(file_path):
    """
        统计计算的lora每个relation的结果
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


def cal_relations_result_lora_subject(file_path):
    """
        统计计算的lora每个subject的结果
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


def cal_relations_result_lora_facts(file_path):
    """
        统计计算的lora每个relation的结果
    :param file_path:
    :return:
    """
    if "dev" in file_path:
        true_relation_count = json.load(open(os.path.join(current_dir, "../data/redocred/redocred_dev_fact_count.json")))
    else:
        true_relation_count = json.load(open(os.path.join(current_dir, "../data/redocred/redocred_test_fact_count.json")))

    datas = [json.loads(line) for line in open(os.path.join(file_path, "predict.json")).readlines()]

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
        relations_result[relation]['all'] = true_relation_count[relation]
        right = relations_result[relation]['tp']
        wrong = relations_result[relation]['fp']
        recall = right / true_relation_count[relation]
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        relations_result[relation]['recall'] = recall
        relations_result[relation]['precision'] = precision
        relations_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
    json.dump(relations_result, open(os.path.join(file_path, "relation_result.json"), "w"), indent=4)


def report_relations_result(file_path):
    """
        打印所有relation的表现结果
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


if __name__ == '__main__':
    # model = "vicuna"
    # test_mode = ["joint_sentence_fact", "joint_sentence_relation_fact", "joint_sentence_one_relation_fact", "joint_sentence_one_relation_subject_fact"]
    # for mode in test_mode[0:1]:
    #     print(mode)
    #     file_path = f"../../public_code/inference/baseline/{model}/no_desc_no_given/{mode}"
    #     cal_relations_result_lora_facts(file_path=file_path)
    #     report_relations_result(file_path=file_path)

    # model = "mistral"
    # mode = "no_desc_no_given"
    # file_path = f"/workspace/xll/analysis_kg/public_code/inference/baseline/{model}/{mode}/joint_sentence_fact/"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # check_subject()
    # version = "vicuna-7b-v1.5-relations"
    # print_relations_result(file_path=f"../versions/v8/result/{version}/redocred_3200/")
    # v = "v0"
    # step = 120
    # version = f"vicuna-7b-v1.5"

    # v = "v0"
    # step = 170
    # version = f"vicuna-13b-v1.5"

    # v = "v1"
    # step = 4300
    # version = f"vicuna-7b-v1.5"

    # v = "v2"
    # step = 4300
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/{version}/redocred_{step}/"
    # cal_relations_result(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v0_1"
    # step = 4400
    # version = f"vicuna-13b-v1.5"
    # v = "v3"
    # step = 4080
    # version = f"vicuna-7b-v1.5"
    #
    # file_path = f"../../versions/{v}/result/{version}/redocred_{step}"
    # cal_relations_result(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v7"
    # step = 23600
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/{version}/redocred_{step}/subject"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # v = "v7"
    # step = 23600
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/{version}/redocred_{step}/fact"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v8"
    # step = 2380
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/relation/{version}/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v8"
    # step = 2740
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/subject/{version}/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v8"
    # step = 7400
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v8"
    # relation_step = 2380
    # subject_step = 2750
    # fact_step = 7300
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/docred_relation_subject_fact_{relation_step}_{subject_step}_{fact_step}/{version}/"
    # cal_relations_result_lora_facts_docred(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # v = "v8"
    # relation_step = 2380
    # subject_step = 2740
    # fact_step = 7000
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/redocred_relation_subject_fact_{relation_step}_{subject_step}_{fact_step}/{version}/"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # v = "v9_2_1"
    # version = "vicuna-7b-v1.5"
    # data_type="dev"
    # step = 390
    # file_path = f"../../versions/{v}/result/{version}/{data_type}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path, type=data_type)
    # report_relations_result(file_path=file_path)

    # v = "v9_3"
    # version = "vicuna-7b-v1.5"
    # data_type="test"
    # step = 1490
    # file_path = f"../../versions/{v}/result/{version}/{data_type}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path, type=data_type)
    # report_relations_result(file_path=file_path)

    # v = "v9_4"
    # step = 700
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/relation/{version}/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_4"
    # step = 1050
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/subject/{version}/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_4"
    # step = 1000
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_4"
    # relation_step = 450
    # subject_step = 1000
    # fact_step = 1000
    # type = "dev"
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/redocred_{type}_relation_{relation_step}_subject_{subject_step}_fact_{fact_step}/{version}"
    # cal_relations_result_lora_facts(file_path=file_path,type=type)
    # report_relations_result(file_path=file_path)

    # v = "v9_5"
    # step = 1020
    # file_path = f"../../public_code/train/LLaMA-Factory-main/result/vicuna/relation/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_5"
    # step = 5390
    # file_path = f"../../public_code/train/LLaMA-Factory-main/result/vicuna/subject/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_5"
    # step = 1950
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/subject/{version}/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_5"
    # step = 1650
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_5"
    # relation_step = 700
    # subject_step = 1950
    # fact_step = 1550
    # type = "dev"
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/redocred_{type}_relation_{relation_step}_subject_{subject_step}_fact_{fact_step}/{version}"
    # cal_relations_result_lora_facts(file_path=file_path, type=type)
    # report_relations_result(file_path=file_path)

    # v = "v9_6"
    # step = 2450
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/subject/{version}/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_6"
    # step = 4550
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # v = "v9_7"
    # step = 850
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_8"
    # step = 1050
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/relation/{version}/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_9"
    # step = 1100
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/relation/{version}/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "v9_9"
    # step = 1450
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/subject/{version}/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    #
    # v = "v9_9"
    # step = 5050
    # version = f"vicuna-7b-v1.5"
    # file_path = f"../../versions/{v}/result/facts/{version}/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "vicuna"
    # step = 1020
    # file_path = f"../../public_code/train/LLaMA-Factory-main/result/{v}/relation/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # v = "vicuna"
    # step = 5390
    # file_path = f"../../public_code/train/LLaMA-Factory-main/result/{v}/subject/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # step = 4430
    # model = "vicuna"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/fact/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "vicuna"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/loras/redocred_dev"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "chatglm3"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/loras/redocred_test"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "vicuna"
    # step = 1020
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/relation/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # model = "mistral"
    # step = 1700
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/relation/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "chatglm3"
    # step = 3400
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/relation_2e-4/redocred_{step}"
    # cal_relations_result_lora_relation(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "chatglm3"
    # step = 5000
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/subject_2e-4/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "chatglm3"
    # step = 7500
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/fact_2e-4/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    model = "mistral"
    step = 14200
    file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/relation_lr5e-5_gpu/redocred_test_{step}"
    cal_relations_result_lora_relation(file_path=file_path)
    report_relations_result(file_path=file_path)
    #

    file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/subject_lr5e-5_gpu/redocred_test_{step}"
    cal_relations_result_lora_subject(file_path=file_path)
    report_relations_result(file_path=file_path)
    #

    file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/fact_lr5e-5_gpu/redocred_test_{step}"
    cal_relations_result_lora_facts(file_path=file_path)
    report_relations_result(file_path=file_path)

    # model = "vicuna"
    # step = 1890
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/subject/redocred_{step}"
    # cal_relations_result_lora_subject(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "mistral"
    # step = 7530
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/fact/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)
    #
    # model = "vicuna"
    # step = 1890
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/fact/redocred_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "vicuna"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/loras/redocred_dev"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "mistral"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/loras/redocred_dev"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "chatglm3"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/loras/redocred_dev"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # # baseline
    # model = "mistral"
    # step = 1000
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/sentence_fact_lr5e-5/redocred_test_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "mistral"
    # step = 450
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/sentence_relations_fact_lr2e-4_deepspeed/redocred_dev_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "mistral"
    # step = 300
    # # mode = "lr5e-5"
    # mode = "lr2e-4_deepspeed"
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/sentence_relation_fact_{mode}/redocred_dev_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)

    # model = "mistral"
    # step = 13700
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/sentence_relation_subject_fact_lr5e-5/redocred_test_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)
    # #
    # model = "mistral"
    # step = 14200
    # file_path = f"/workspace/xll/analysis_kg/public_code/train/LLaMA-Factory-main/result/{model}/sentence_relation_subject_fact_desc_lr5e-5_gpu/redocred_dev_{step}"
    # cal_relations_result_lora_facts(file_path=file_path)
    # report_relations_result(file_path=file_path)
