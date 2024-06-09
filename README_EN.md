[read this in Chinese](https://github.com/bigdante/AutoRE/blob/main/README.md)

# AutoRE
This repository is based on the code from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and implements a document-level relation extraction system named AutoRE based on large language models. The extraction paradigm used is RHF ([paper link](https://arxiv.org/abs/2403.14888v1)).
Currently, experiments are conducted on the [Re-DocRED](https://github.com/tonytan48/Re-DocRED) dataset, and it is capable of extracting triples of 96 relations from document-level text.

## Usage
Method 1.Download the model from [ckpt](https://cloud.tsinghua.edu.cn/d/4d12cf0620164caca82c/), which corresponds to the AutoRE models fine-tuned on Mistral-7B, Vicuna-7B, and ChatGLM3-6B.

Method 2.Download from huggingface[dante123/AutoRE](https://huggingface.co/dante123/AutoRE/tree/main)。
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

#### 1）data prepare
```shell
cd AutoRE/utils/
python pre_process_data.py
```

#### 2）model finetuning

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

## AutoRE_analysis
This verifies whether the analysis process is helpful for extraction. The overall thought process aligns with the AutoRE framework, but includes an analysis step before each extraction phase.
For specific examples, please see [redocred_train_analysis.json](https://github.com/bigdante/AutoRE/blob/main/AutoRE/data/redocred/analysis_redocred/redocred_train_analysis.json)
The data and code have been shared, hoping to provide some inspiration to everyone.

Additionally, in order for AutoRE to perform more types of relation extraction, other open-source datasets, including English datasets such as FewRel and NYT, as well as Chinese datasets like HaCred, should be incorporated. If the focus is solely on the work of this paper, you only need to comment out the other parts of the data processing in the `pre_process_data.py` file, retaining only the part that processes RedoCred (there are many comments in the code that should help you).



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




