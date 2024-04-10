from llmtuner import run_exp
import os
import wandb

project_name = os.environ.get('WANDB_PROJECT_NAME', 'autokg')

api_key = os.environ.get('WANDB_API_KEY', "input your id")
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
