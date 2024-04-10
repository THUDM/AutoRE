"""
Description: 
Author: dante
Created on: 2024/4/10
"""




import json

# 加载JSON文件
with open('redocred_train_analysis.json', 'r') as file:
    data = json.load(file)

# 先去除每个字典中的'fact_list'和'same_fact_list'键
for item in data:
    item.pop('fact_list', None)  # 如果键不存在，pop方法不会抛出异常
    item.pop('same_fact_list', None)

# 按照每个字典中的'index'键进行排序
sorted_list = sorted(data, key=lambda x: x.get('index', 0))  # 使用get避免键不存在时出错

# 保存修改后的数据回原文件
with open('redocred_train_analysis.json', 'w') as file:
    json.dump(sorted_list, file, indent=4)
