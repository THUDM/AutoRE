# from AutoRE.src.llmtuner import ChatModel
# from AutoRE.utils.llama_factory_inference import *
from AutoRE.utils.report_result import *
import sys

if __name__ == '__main__':
    cal_result_lora_facts(file_path="/workspace/xll/AutoRE_GitHub/AutoRE/result/vicuna/loras_chinese/hacred_test/")
    report_relations_result(file_path="/workspace/xll/AutoRE_GitHub/AutoRE/result/vicuna/loras_chinese/hacred_test/")
    # args = get_params()
    # args.node = 0
    # args.template_version = args.version
    # lora_test = args.lora_test
    # do_inference = args.inference
    # args_to_filter = ['--data_path', '--save_path', '--lora_test', '--version', '--inference']
    # sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    # sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank=")]
    # base = args.adapter_name_or_path
    # if lora_test == "loras_D_R_H_F_desc":
    #     r_step = args.relation_step
    #     s_step = args.subject_step
    #     f_step = args.fact_step
    #     # This adjustment was made to avoid conflicts with the Llama_factory codebase.
    #     args_to_filter = ['--relation_step', '--subject_step', '--fact_step']
    #     sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    #     argv_index = 4
    #     sys.argv[argv_index] = base + f"relation_chinese_5e-5/checkpoint-{r_step}"
    #     args.r_model = ChatModel()
    #     sys.argv[argv_index] = base + f"subject_chinese_5e-5/checkpoint-{s_step}"
    #     args.s_model = ChatModel()
    #     sys.argv[argv_index] = base + f"fact_chinese_5e-5/checkpoint-{f_step}"
    #     args.f_model = ChatModel()
    #     if not do_inference:
    #         args.data = get_test_data(args)
    #         loras_RHF_desc(args)
    #         cal_result_lora_facts(file_path=args.save_path)
    #     else:
    #         loras_RHF_desc_for_test(args)
    # else:
    #     args.model = ChatModel()
    #     # Define a dictionary to map the values of lora_test to their respective functions
    #     function_mapping = {
    #         "lora_D_F": (lora_D_F, lora_D_F_for_test, cal_result_lora_facts),
    #         "lora_D_RS_F": (lora_D_RS_F, lora_D_RS_F_for_test, cal_result_lora_facts),
    #         "lora_D_R_F": (lora_D_R_F, lora_D_R_F_for_test, cal_result_lora_facts),
    #         "lora_D_R_H_F": (lora_D_R_H_F, lora_D_R_H_F_for_test, cal_result_lora_facts),
    #         "lora_D_R_H_F_desc": (lora_D_R_H_F, lora_D_R_H_F_for_test, cal_result_lora_facts),  # Notice the different test function
    #         "relation": (lora_relation, lora_relation, cal_result_lora_relation),  # Assuming no special function for inference
    #         "subject": (lora_subject, lora_subject, cal_result_lora_subject),  # Assuming no special function for inference
    #         "facts": (lora_facts, lora_facts, cal_result_lora_facts)  # Assuming no special function for inference
    #     }
    #     # Check if lora_test is in the mapping
    #     if lora_test in function_mapping:
    #         # Select the appropriate function based on whether args.inference is True or False
    #         func, func_for_test, func_cal_result = function_mapping[lora_test]
    #         if not do_inference:
    #             args.data = get_test_data(args)
    #             func(args)
    #         else:
    #             func_for_test(args)
    #             func_cal_result(file_path=args.save_path)
    #     else:
    #         print("Invalid lora_test value")
