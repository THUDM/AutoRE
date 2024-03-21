from AutoRE.src.llmtuner import ChatModel
from AutoRE.utils.llama_factory_inference import *
from AutoRE.utils.basic import *
from termcolor import colored
import os
import sys


def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


def lora_relation_subject_facts_for_test(args):
    """
        对vicuna模型进行lora测试整个pipeline
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


if __name__ == '__main__':
    args = get_params()
    args.node = 0
    args.template_version = args.version
    lora_test = args.lora_test
    args_to_filter = ['--data_path', '--save_path', '--lora_test', '--version']
    sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank=")]
    if lora_test == "lora_relation_subject_fact":
        base = args.adapter_name_or_path
        r_step = args.relation_step
        s_step = args.subject_step
        f_step = args.fact_step
        args_to_filter = ['--relation_step', '--subject_step', '--fact_step']
        sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
        argv_index = 4
        sys.argv[argv_index] = base + f"relation/checkpoint-{r_step}"
        args.r_model = ChatModel()
        sys.argv[argv_index] = base + f"subject/checkpoint-{s_step}"
        args.s_model = ChatModel()
        sys.argv[argv_index] = base + f"fact/checkpoint-{f_step}"
        args.f_model = ChatModel()
        lora_relation_subject_facts_for_test(args)
