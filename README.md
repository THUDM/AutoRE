*Read this in [English](README_EN.md)*
# Auto-KG
本仓库基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)代码，实现了基于大语言模型的文档级关系抽取系统AutoRE。使用的抽取范式为RHF（[论文链接(https://arxiv.org/submit/5482782/view)])。
目前基于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集进行实验，能够文档级文本中的96个关系的三元组事实。

## 使用方法

### 1.模型训练

#### 1) 数据准备
当前的知识抽取数据集，大多是一个句子中只有一个relation，relation对应1个或者多个的fact。而现实场景中，一个句子其实会包含多个relation。为了得到含有多个relation并且标注良好的语料，我们通过对[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据的train_devised和dev_devised进行预处理。
具体如下：
##### 清晰定义relation descripiton
[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集总共96个关系。但是存在的问题是：
###### 1.relation的description不够清晰
多个relation的表述不够清晰，例如`author`的关系中，不同的人，可能理解不同，可以将此理解为 `somebody is the author of somebook`，也可以理解为`somebook the author is somebody`, 如果没有清晰的定义，则会造成主体和客体混乱。
###### 2.relation互相包含或者相反
例如`member of`和`member of political party`，`member of sports team`其实可以统一为`member of`。
另外例如 `participant`和`participant of`语义相反，保留其中一个即可。

针对以上的问题，我们重新整理改造了relation，梳理出共64个relations，并且赋予了清晰明确的定义，更符合语言模型的理解。具体参见：
[relation_map.json](https://github.com/bigdante/Analysis_KG/blob/main/data/relations_desc/relation_map.json)
##### 2) analysis process
在对基础数据预处理后，我们通过ChatGPT和人工，使用prompt engineering，生成relation、subjects，以及fact的分析过程。并且为了后续的方便，我们将数据整理如下。
```python
# one sample
[{
        "index": 0,
        "passage": "Niklas Bergqvist ( born 6 October 1962 in Stockholm ) , is a Swedish songwriter , producer and musician . After the band split - up in 1987 , Bergqvist formed and played in several other bands until he decided to focus more on songwriting , resulting in several releases ever since .",
        "relations": [
            "date of birth",
            "place of birth"
        ],
        "fact_list": [
            {
                "fact": [
                    "Niklas Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            },
            {
                "fact": [
                    "Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            },
            {
                "fact": [
                    "Niklas Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            },
            {
                "fact": [
                    "Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            }
        ],
        "same_fact_list": [
            [
                [
                    "Niklas Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ],
                [
                    "Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            ],
            [
                [
                    "Niklas Bergqvist",
                    "place of birth",
                    "Stockholm"
                ],
                [
                    "Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            ]
        ],
        "relation_analysis": "According to the passage, the relations identified are \"date of birth\" and \"place of birth.\" The reason for this conclusion is that the passage explicitly states that Niklas Bergqvist was born on 6 October 1962 in Stockholm, which supports the relation \"date of birth.\" Additionally, the passage mentions that Bergqvist was born in Stockholm, providing evidence for the relation \"place of birth.\" Thus, these specific details mentioned in the passage lead to the identification of these two relations.",
        "entity_analysis": {
            "date of birth": "In the given passage, the entities \"Bergqvist\" and \"Niklas Bergqvist\" can be considered as the subjects of the fact related to \"date of birth\" because both refer to the same person. The passage mentions that Niklas Bergqvist was born on 6 October 1962 in Stockholm. \"Bergqvist\" is likely being referred to as a shorthand or a last name reference to Niklas Bergqvist himself. Thus, both entities represent the individual who was born on the specific date mentioned in the passage.",
            "place of birth": "The entities \"Bergqvist\" and \"Niklas Bergqvist\" can be considered as the subjects of the fact related to \"place of birth\" because the passage explicitly states that Niklas Bergqvist was born in Stockholm. This aligns with the relation's description, which states that \"place of birth\" refers to the specific location where a person was born. As Niklas Bergqvist is mentioned as the individual being discussed, it is reasonable to identify him and his last name \"Bergqvist\" as the subjects associated with the fact of being born in Stockholm."
        },
        "fact_analysis": {
            "date of birth": {
                "Bergqvist": "According to the subject \"Bergqvist\" and the relation \"date of birth,\" the fact is that Niklas Bergqvist was born on 6 October 1962. This information is based on the passage which states that Bergqvist was born in Stockholm. The passage also mentions his involvement in various bands and his transition to focusing more on songwriting after the band split-up. However, these details are not relevant to the fact extraction regarding his date of birth.",
                "Niklas Bergqvist": "According to the subject \"Niklas Bergqvist\" and the relation \"date of birth,\" the fact is that Niklas Bergqvist was born on 6 October 1962. This information is derived from the mention of his birthdate in the passage. The reason for this fact being true is that the passage explicitly states that he was born on this specific date. Therefore, based on the given information, we can conclude that Niklas Bergqvist's date of birth is 6 October 1962."
            },
            "place of birth": {
                "Bergqvist": "According to the subject (Bergqvist) and the relation (place of birth), the fact is that Niklas Bergqvist was born in Stockholm. This is based on the specific information provided in the passage, which states that he was born in Stockholm on October 6, 1962. The reason for this conclusion is the clear mention of his birthplace in the passage, indicating that Stockholm is the most specific known location of his birth.",
                "Niklas Bergqvist": "According to the subject \"Niklas Bergqvist\" and the relation \"place of birth,\" the fact is that Niklas Bergqvist was born in Stockholm. This is evident from the passage which explicitly states, \"Niklas Bergqvist (born 6 October 1962 in Stockholm).\" The mention of a specific date and location of birth reinforces the accuracy of this fact."
            }
        }
    },
    ...
]
```
通过运行脚本，即可完成vicuna训练数据准备。
在此之前，需要下载[Re-DocRED](https://github.com/tonytan48/Re-DocRED)到data/redocred文件夹下。
在data/chatgpt_count下的key.json文件中，按照所示的格式，添加可用的API keys（keys的数量越多，数据处理效率越高）。
并在shell中指定训练数据保存的路径。【中间生成的数据将会保存在data/redocred文件夹下】
```shell
cd code/data_process/
bash data_process.sh
```

### 微调vicuna-13b-v1.5模型
我们的代码参考自[FastChat](https://github.com/lm-sys/FastChat/tree/main)。
在运行脚本前，需要指定脚本中的训练集路径以及checkpoint保存路径。
```shell
cd code/model_train/vicuna_train
bash train.sh
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




