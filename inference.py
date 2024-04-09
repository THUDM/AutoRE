from AutoRE.src.llmtuner import ChatModel
from AutoRE.utils.llama_factory_inference import *
from AutoRE.utils.basic import *
import sys

if __name__ == '__main__':
    args = get_params()
    args.node = 0
    args.template_version = args.version
    lora_test = args.lora_test
    args_to_filter = ['--data_path', '--save_path', '--lora_test', '--version']
    sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank=")]
    base = args.adapter_name_or_path
    if lora_test == "loras_D_R_H_F_desc":
        r_step = args.relation_step
        s_step = args.subject_step
        f_step = args.fact_step
        # 这里是为了和Llama_factory代码库不产生冲突做的调整
        args_to_filter = ['--relation_step', '--subject_step', '--fact_step']
        sys.argv = [arg for i, arg in enumerate(sys.argv) if all(arg != filter_arg and (i == 0 or sys.argv[i - 1] != filter_arg) for filter_arg in args_to_filter)]
        argv_index = 4
        sys.argv[argv_index] = base + f"relation/checkpoint-{r_step}"
        args.r_model = ChatModel()
        sys.argv[argv_index] = base + f"subject/checkpoint-{s_step}"
        args.s_model = ChatModel()
        sys.argv[argv_index] = base + f"fact/checkpoint-{f_step}"
        args.f_model = ChatModel()
        if args.inference:
            loras_RHF_desc_for_test(args)
        else:
            loras_RHF_desc(args)
    else:
        args.model = ChatModel()
        if lora_test == "lora_D_F":
            if args.inference:
                lora_D_F(args)
        elif lora_test == "lora_D_RS_F":
            if args.inference:
                lora_D_RS_F(args)
        elif lora_test == "lora_D_R_F":
            if args.inference:
                lora_D_R_F(args)
        elif lora_test == "lora_D_R_H_F":
            if args.inference:
                lora_D_R_H_F(args)
        elif lora_test == "lora_D_R_H_F_desc":
            if args.inference:
                lora_D_R_H_F(args)
        elif lora_test == "relation":
            lora_relation(args)
        elif lora_test == "subject":
            lora_subject(args)
        elif lora_test == "facts":
            lora_facts(args)
