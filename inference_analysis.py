from AutoRE.src.llmtuner import ChatModel
from AutoRE.utils.llama_factory_inference_analysis import *
from AutoRE.utils.report_result import *
import sys
if __name__ == '__main__':
    args = get_params()
    args.node = 0
    args.template_version = args.version
    lora_test = args.lora_test
    do_inference = args.inference
    r_step = args.relation_step
    s_step = args.subject_step
    f_step = args.fact_step
    args_to_filter = ['--data_path', '--save_path', '--lora_test', '--version','--inference']
    sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank=")]
    base = args.adapter_name_or_path
    # This adjustment was made to avoid conflicts with the Llama_factory codebase.
    args_to_filter = ['--relation_step', '--subject_step', '--fact_step']
    sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    argv_index = 4
    sys.argv[argv_index] = base + f"relation_analysis/checkpoint-{r_step}"
    args.r_model = ChatModel()
    sys.argv[argv_index] = base + f"subject_analysis/checkpoint-{s_step}"
    args.s_model = ChatModel()
    sys.argv[argv_index] = base + f"fact_analysis/checkpoint-{f_step}"
    args.f_model = ChatModel()
    if not do_inference:
        args.data = get_test_data(args)
        loras_RHF_desc_analysis(args)
        cal_result_lora_facts(file_path=args.save_path)
    else:
        loras_RHF_desc_for_inference(args)
