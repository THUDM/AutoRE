*Read this in [English](README_EN.md)*
# Analysis-KG
本仓库基于[Vicuna 13b-v1.5](https://github.com/lm-sys/FastChat/tree/main)模型，实现了可解释的英文语料知识抽取系统。
## 介绍
Analysis-KG 是基于[Vicuna 13b-v1.5](https://github.com/lm-sys/FastChat/tree/main)模型进行微调，实现了可解释的英文语料知识抽取系统。
微调的训练数据集来源于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)。
我们实现了从句子中，一次性分析抽取多个relations，并且能够抽取出relation对应的多个triple facts，例如：
```python
>>>Drossinis Museum is in the center of Kifisia , a northern suburb of Athens , and it is housed in \u201c Amaryllis \u201d villa , where Georgios Drossinis lived in his last years and which is named after a central character of one of his earliest and most popular works . The museum was founded in 1997 with the aim to preserve and promote Drossinis\u2019 ( 1859 - 1951 ) multidimensional work . Drossinis , along with Costis Palamas and Nikos Kampas co - founded the New Athenian School , the Greek literary \u2018 1880s Generation\u2019 , a movement which renewed Greek literature and fought for the establishment of modern Greek language ( \" Demotic \" ) The museum includes three halls , which are on the first floor , while the ground floor houses the Municipal Library of Kifisia , according to Drossinis \u2019s wish .

According to the passage, the relations identified are "country of citizenship," "date of death," "notable work," "date of birth," "place of birth," "location in," and "languages spoken, written or signed." The reasons for these relations are as follows:
1. "Country of citizenship": The passage mentions that the museum is located in Greece, indicating that the country recognizes it as its property.
2. "Date of death": The passage states that Georgios Drossinis lived in the museum for the last 1951, implying that he is no longer alive.
3. "Notable work": The passage mentions that the museum is named after a central character of Drossinis's earliest and most popular works, highlighting its significance.
4. "Date of birth": The passage provides the specific years of Drossinis's birth and death, indicating the time of his birth.
5. "Place of birth": The passage mentions that Drossinis lived in a northern suburb of Athens, indicating that he was born in that specific location.
6. "Location in": The passage states that the museum is located in Kifisia, a suburb of Athens, indicating its association with that specific location.
7. "Languages spoken, written or signed": The passage does not provide evidence to support this relation.
In conclusion, the relations identified are supported by the explicit information provided in the passage, such as the names, dates, and the association of the museum with Drossinis's works and the specific location of Kifisia.

['date of birth', 'location in', 'country of citizenship', 'notable work', 'date of death',  'place of birth', 'residence', 'languages spoken, written or signed',]
=================================date of birth=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "date of birth" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for 1859-1951. The passage also states that Georgios Drossinis is considered an early and popular work, further emphasizing his significance. Additionally, the passage mentions the establishment of the museum in 1997, indicating that it was created to preserve and promote Georgios Drossinis's legacy. Therefore, based on the information provided, it is reasonable to infer that the entities "Drossinis" and "Georgios Drossinis" are the subjects of the fact related to "date of birth."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject "Drossinis" and the relation "date of birth," the fact is that Georgios Drossinis was born in 1859. This information is derived from the passage which states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. The passage also mentions that Drossinis lived in the Drossinis Museum for a significant period of time, indicating his connection to the institution. Therefore, based on the given information, it can be concluded that Georgios Drossinis's date of birth is 1859.
['Drossinis', 'date of birth', '1859']

fact_analysis:  According to the subject "Georgios Drossinis" and the relation "date of birth," the fact is that Georgios Drossinis was born in 1859. This information is derived from the passage which states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. The passage also mentions that Drossinis lived in his last years and is named after a central character of his works. Therefore, it can be concluded that Georgios Drossinis's date of birth is 1859.
['Georgios Drossinis', 'date of birth', '1859']
=================================location in=================================
subjects_analysis:  The entities 'Drossinis', 'Greek', and 'Kifisia' can be considered as the subjects of the fact related to "location in" because they are all associated with specific locations or geographic entities. 
1. 'Drossinis' is mentioned as the name of the museum, indicating a physical location. The fact that it is mentioned in the passage suggests that it is associated with a specific location.
2. 'Greek' is mentioned in relation to the name of the museum, indicating a language associated with a specific location. The fact that the name is mentioned in the passage implies that the language is associated with a specific location.
3. 'Kifisia' is mentioned as the specific location where the museum is situated. The fact that the museum is located in a specific location implies that it is associated with a specific geographic entity.
In summary, all these entities can be considered as the subjects of the fact related to "location in" because they are associated with specific locations or geographic entities, as stated in the passage.
['Drossinis', 'Greek', 'Kifisia']

fact_analysis:  According to the subject and relation, the fact is that "Drossinis" is associated with the location "Greece". The reason for this association is that the passage mentions the Drossinis Museum, which is located in Greece. The passage states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works, indicating his affiliation with Greece. Therefore, based on the information provided, it can be concluded that "Drossinis" is linked to the location of Greece.
['Drossinis', 'location in', 'Greek']

fact_analysis:  According to the subject "Greek" and the relation "location in," the fact is that the Greek language is associated with Greece. This is evident from the passage which states that the Greek literary work, "Democratic," was fought for the establishment of modern Greek language. The passage also mentions the Greek literary work, the 1880s Generation, which is included in the museum. Therefore, it can be concluded that the Greek language is closely tied to Greece based on this information.
['Greek', 'location in', 'Greece']

fact_analysis:  According to the subject "Kifisia" and the relation "location in," the fact is that Kifisia is located in Greece. This is evident from the passage which states that the Drossinis Museum is in the center of Kifisia, a northern suburb of Athens, and is named after a central character of Drossinis's earliest and most popular works. The passage also mentions that the museum was founded in 1997 with the aim to preserve and promote Drossinis's work, which indicates his affiliation with Greece. Therefore, based on this information, it can be concluded that Kifisia is indeed located in Greece.
['Kifisia', 'location in', 'Greek']
=================================country of citizenship=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "country of citizenship" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for a significant period of time. The passage also states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works. Additionally, it is mentioned that the municipality where the museum is located is in Greece, which further supports the inference that the individuals associated with the museum, namely Drossinis and Georgios Drossinis, are likely to be citizens of Greece. Therefore, based on the information provided, it is reasonable to consider Drossinis and Georgios Drossinis as the subjects of the fact related to "country of citizenship."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the fact is that Drossinis's country of citizenship is Greece. This is evident from the passage which states that Drossinis lived in Greece for a significant period of time, and the Drossinis Museum is located in Greece. Therefore, it can be concluded that Drossinis is recognized as a citizen of Greece.
['Drossinis', 'country of citizenship', 'Greek']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis' country of citizenship is Greece. This is evident from the passage which states that Drossinis lived in his last years in Greece and is recognized as a central character of his works. The passage also mentions the founding of the Drossinis Museum in Greece, further indicating his connection to the country. Therefore, based on this information, it can be concluded that Georgios Drossinis is a citizen of Greece.
['Georgios Drossinis', 'country of citizenship', 'Greek']
=================================notable work=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "notable work" because the passage explicitly states that the Drossinis Museum is named after a central character of one of Drossinis's earliest and most popular works. This indicates that Drossinis's literary creations have had a significant impact and are recognized as noteworthy works. Additionally, the passage mentions that the museum was founded with the aim of preserving and promoting Drossinis's work, further emphasizing his significance as a literary creator. Therefore, based on the information provided, it is evident that both "Drossinis" and "Georgios Drossinis" are associated with notable works, making them suitable subjects for the fact related to "notable work".
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the notable work associated with Drossinis is the "Democratic" or "Democratic Language." This is evident from the passage which states that the Drossinis Museum includes three halls, one of which is the "Democratic" and the other is the "Democratic Language." The reason for considering this work as notable is that it is mentioned in the passage as being included in the museum, which is a significant creation in the field of Greek literature. Therefore, it can be concluded that "Democratic" and "Democratic Language" are noteworthy works associated with Drossinis.
['Drossinis', 'notable work', 'Democratic']
['Drossinis', 'notable work', 'Democratic Language']

fact_analysis:  According to the subject and relation, the notable work associated with Georgios Drossinis is the "Democratic" or "Democratic Language." This is evident from the passage which states that the Drossinis Museum is named after a central character of Drossinis's earliest and most popular works, and it includes three halls, one of which is the "Democratic" or "Democratic Language." Therefore, it can be concluded that "Democratic" is a significant creation associated with Georgios Drossinis.
['Georgios Drossinis', 'notable work', 'Democratic']
['Georgios Drossinis', 'notable work', 'Democratic Language']
=================================date of death=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "date of death" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for the last 1951. The passage also states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works. Since the passage mentions the specific names and the timeframe of his life and legacy, it is reasonable to infer that the entities "Drossinis" and "Georgios Drossinis" are the subjects of the fact related to "date of death."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the fact is that "Drossinis" died in 1951. The reason for this conclusion is the mention in the passage that the Drossinis Museum was founded in 1997 and is named after a central character of Drossinis's earliest and most popular works. Additionally, the passage states that the museum includes three halls, which were on the first floor, while the ground floor is described as the "Melal Library of Kifisia" according to Drossinis's wishes. This implies that Drossinis is no longer alive, as the passage does not provide any information about his current status or any events after 1951.
['Drossinis', 'date of death', '1951']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis died in 1951. This is evident from the passage which states that the Drossinis Museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. It further mentions that Drossinis lived in his last years and the museum is named after him. Therefore, based on this information, it can be concluded that Georgios Drossinis's date of death was in 1951.
['Georgios Drossinis', 'date of death', '1951']
# 省略剩下的结果
```
## 依赖
### 软件依赖
```
#运行微调和推理需要安装以下依赖
pip3 install fschat
```
### 硬件依赖
```
A100 40GB，单卡即可运行
```

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




