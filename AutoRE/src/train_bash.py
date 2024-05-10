from llmtuner import run_exp
import os
import wandb

project_name = os.environ.get('WANDB_PROJECT_NAME', 'autokg')
# api_key = os.environ.get('WANDB_API_KEY', "7c6f4e133de2c05cecc67d156885e5d58a9031a1")
api_key = os.environ.get('WANDB_API_KEY', "f41e21d9f3f01b024c832d1b678d2592b7625855")
if api_key:
    wandb.login(key=api_key)
wandb.init(project=project_name)

def main():
    run_exp()


# 这是为多进程或多线程环境设计的函数，index参数通常表示进程或线程的索引
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
