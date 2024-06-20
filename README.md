[Read this in English.](https://github.com/bigdante/AutoRE/blob/main/README_EN.md)
# AutoRE
本仓库基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)代码，实现了基于大语言模型的文档级关系抽取系统AutoRE。使用的抽取范式为RHF（[论文链接](https://arxiv.org/abs/2403.14888v1)）。
目前基于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集进行实验，能够抽取文档级文本中的96个关系的三元组事实。

## 使用方法
方法1:下载模型地址[ckpt](https://cloud.tsinghua.edu.cn/d/4d12cf0620164caca82c/)，其中对应着微调Mistral-7B，Vicuna-7B和ChatGLM3-6B后的AutoRE模型。

方法2:从 huggingface上下载[dante123/AutoRE](https://huggingface.co/dante123/AutoRE/tree/main)。
### 0.环境准备
```shell
cd AutoRE/
pip install -r requirement.txt
```
因为使用了wandb，所以需要先将[train_bash.py](https://github.com/bigdante/AutoRE/blob/main/AutoRE/src/train_bash.py)中的key进行设置
```shell
api_key = os.environ.get('WANDB_API_KEY', "your api key")
```
### 1.推理

```shell
# 根据AutoRE.sh内的提示进行修改
bash AutoRE.sh
# 输入对应文档即可自动抽取
```

### 2.模型训练

#### 1）数据准备
```shell
cd AutoRE/utils/
python pre_process_data.py
```

#### 2）微调模型
我们的代码参考自[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，并进行了适当修改。

```shell
cd AutoRE/
# 选择对应的模型进行微调
# 可以指定单卡或者多卡
bash train_script/mistral_loras_D_R_H_F_desc.sh
```

### 3.模型测试

```shell
cd AutoRE/
# 选择对应的模型进行测试，数据集为Re-DocRED，将--inference去除，并且设置具体的模型和ckpt
bash AutoRE.sh
```

## AutoRE_analysis
验证analysis过程是否对抽取是有帮助的。整个过程思路与AutoRE的框架一致，只是在每一步抽取前加入了analysis。
具体可以看[redocred_train_analysis.json](https://github.com/bigdante/AutoRE/blob/main/AutoRE/data/redocred/analysis_redocred/redocred_train_analysis.json)中的例子。
数据和代码已经分享，希望对大家能有些许启发～

另外，为了使AutoRE能够做更多类的关系抽取，加入其他的开源数据，包括英文的fewrel，nyt等，以及中文的hacred等。如果只关注本论文的工作，只需要将数据处理中pre_process_data.py的其他代码注释掉，只保留处理redocred的处理部分（代码中给了很多的注释，希望能帮到你们～）

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用论文。

```
@article{lilong2024autore,
  title={AutoRE: Document-Level Relation Extraction with Large Language Models},
  author={Lilong, Xue and Dan, Zhang and Yuxiao, Dong and Jie, Tang},
  journal={arXiv preprint arXiv:2403.14888},
  year={2024}
}
```







