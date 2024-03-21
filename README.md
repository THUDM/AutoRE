*Read this in [English](README_EN.md)*
# AutoRE
本仓库基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)代码，实现了基于大语言模型的文档级关系抽取系统AutoRE。使用的抽取范式为RHF（[论文链接(https://arxiv.org/submit/5482782/view)])。
目前基于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集进行实验，能够文档级文本中的96个关系的三元组事实。

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

### 微调模型
我们的代码参考自[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

```shell
cd AutoRE/
bash train_script/mistral_lora
```

### 推理
#### 运行[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集，检查模型对每个relation的表现效果。
首先下载[ckpt](https://cloud.tsinghua.edu.cn/d/ead42cf68f484c73af22/)。
```shell
# 在shell中指定ckpt路径，结果保存的路径, cuda_id。
cd code/model_inference
bash run_13b_vicuna_v0.sh
```
#### 命令行测试模式
命令行模式，用于手动逐个输入句子，体验效果。
<img width="1397" alt="image" src="https://github.com/bigdante/Analysis_KG/assets/39661185/efd07341-9e87-4508-be4a-cff1cd1ef346">

```shell
# 需要在shell中指定预训练好的ckpt路径，cuda_id。
cd code/model_inference
bash mode_test.sh
```

#### 自己数据集测试模式
```shell
# 需要在shell中指定预训练好的ckpt路径，cuda_id。并且需要指定待验证数据集的路径，以及结果保存的路径
cd code/model_inference
bash mode_test.sh
```

#### ChatGPT 验证
我们实现了简易的基于ChatGPT turbo3.5的fact verification。用于校验生成的triple fact是否是正确的。
在命令行测试模式下，添加参数，`--chatgpt_check`， 即可开启。ChatGPT 将会只根据上下文以及relation的定义，判断fact是否是正确的。【输出True或者False】
同理，需要先设置API-key，同上述的训练数据准备阶段。
```shell
# 需要在shell中指定预训练好的ckpt路径，cuda_id。
cd code/model_inference
bash mode_test.sh
```

## 引用

```

```




