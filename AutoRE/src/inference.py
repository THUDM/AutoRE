import sys

from llmtuner import ChatModel
from ..utils.report_result import *
from ..utils.llama_factory_inference import *

if __name__ == '__main__':
    args = get_params()
    args.node = 0
    args.template_version = args.version
    args.data = get_test_data(args)
    lora_test = args.lora_test
    args_to_filter = ['--data_path', '--save_path', '--lora_test', '--version']
    sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank=")]
    if args.data:
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
            loras_RHF_desc_for_test(args)
            cal_result_lora_facts(file_path=args.save_path)
        else:
            args.model = ChatModel()
            if lora_test == "lora_relation":
                lora_relation(args)
                cal_result_lora_relation(file_path=args.save_path)
            elif lora_test == "lora_subject":
                lora_subject(args)
                cal_result_lora_subject(file_path=args.save_path)
            elif lora_test == "lora_fact":
                lora_facts(args)
                cal_result_lora_facts(file_path=args.save_path)
            elif lora_test == "lora_sentence_fact":
                lora_D_F(args)
                cal_result_lora_facts(file_path=args.save_path)
            elif lora_test == "lora_sentence_relations_fact":
                lora_D_RS_F(args)
                cal_result_lora_facts(file_path=args.save_path)
            elif lora_test == "lora_sentence_relation_fact":
                lora_D_R_F(args)
                cal_result_lora_facts(file_path=args.save_path)
            elif lora_test == "lora_sentence_relation_subject_fact" or lora_test == "lora_sentence_relation_subject_fact_desc":
                lora_D_R_H_F(args)
                cal_result_lora_facts(file_path=args.save_path)
