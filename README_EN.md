[read this in Chinese](https://github.com/bigdante/AutoRE/blob/main/README.md)

# AutoRE
This repository is based on the code from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and implements a document-level relation extraction system named AutoRE based on large language models. The extraction paradigm used is RHF ([paper link](https://arxiv.org/abs/2403.14888v1)).
Currently, experiments are conducted on the [Re-DocRED](https://github.com/tonytan48/Re-DocRED) dataset, and it is capable of extracting triples of 96 relations from document-level text.

## Usage
Download the model from [ckpt](https://cloud.tsinghua.edu.cn/d/4d12cf0620164caca82c/), which corresponds to the AutoRE models fine-tuned on Mistral-7B, Vicuna-7B, and ChatGLM3-6B.

### 0. Environment prepare
```shell
    cd AutoRE/
    pip install -r requirement.txt
```
I use wandb，so make sure insert your API key in [train_bash.py](https://github.com/bigdante/AutoRE/blob/main/AutoRE/src/train_bash.py) firstly.
```shell
api_key = os.environ.get('WANDB_API_KEY', "your api key")
```
### 1. Inference

```shell
# Modify according to the prompts in AutoRE.sh
bash AutoRE.sh
# Enter the corresponding document to automatically extract
````

### 2.model training

#### 1) data prepare
```shell
cd AutoRE/utils/
python pre_process_data.py
```

#### 2） model finetuning

```shell
cd AutoRE/
# Modify according to the prompts in AutoRE.sh and choose the RE paradigms you need
bash train_script/mistral_loras_D_R_H_F_desc.sh
```

### 3.model test

```shell
cd AutoRE/
# Choose the corresponding model for testing, the dataset is Re-DocRED, remove --inference, and set the specific model and ckpt
bash AutoRE.sh
```

## Citation

If you find our work helpful, please consider citing the paper.

```
@article{lilong2024autore,
  title={AutoRE: Document-Level Relation Extraction with Large Language Models},
  author={Lilong, Xue and Dan, Zhang and Yuxiao, Dong and Jie, Tang},
  journal={arXiv preprint arXiv:2403.14888},
  year={2024}
}
```




