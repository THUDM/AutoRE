read in English[README_EN](https://github.com/bigdante/AutoRE/README_EN.md)
# AutoRE
本仓库基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)代码，实现了基于大语言模型的文档级关系抽取系统AutoRE。使用的抽取范式为RHF（[论文链接](https://arxiv.org/submit/5482782/view)）。
目前基于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集进行实验，能够抽取文档级文本中的96个关系的三元组事实。

## 使用方法
下载模型地址[ckpt](https://cloud.tsinghua.edu.cn/d/4d12cf0620164caca82c/)，其中对应着微调Mistral-7B，Vicuna-7B和ChatGLM3-6B后的AutoRE模型。
### 1.推理

```shell
# 根据AutoRE.sh内的提示进行修改
bash AutoRE.sh
# 输入对应文档即可自动抽取
```

### 2.模型训练

#### 1) 数据准备
```shell
cd AutoRE/utils/
python pre_process_data.py
```

#### 2） 微调模型
我们的代码参考自[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，并进行了适当修改。

```shell
cd AutoRE/
# 选择对应的模型进行微调
# 可以指定单卡或者多卡
bash train_script/mistral_loras_D_R_H_F_desc.sh
```

#### 3） 微调测试

```shell
cd AutoRE/
# 选择对应的模型进行测试，数据集为Re-DocRED，将--inference去除，并且设置具体的模型和ckpt
bash AutoRE.sh
```





