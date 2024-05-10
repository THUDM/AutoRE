# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 示例数据
# # methods = ['TAG', 'D-F', 'D-RS-F', 'D-R-F', 'D-R-H-F', 'D-R-H-F-desc', 'Mis-A-R-H-F', 'Vic-A-R-H-F', 'GLM-A-R-H-F']
# # f1_scores_test = [49.38, 39.65, 40.33, 41.48, 42.90, 44.11, 51.91, 53.84, 51.11]  # test数据
# # f1_scores_dev = [49.34, 39.07, 40.30, 42.52, 44.12, 45.29, 53.01, 54.29, 49.86]  # dev数据
#
# methods = ['TAG',  'Mis-AutoRE', 'Vic-AutoRE', 'GLM-AutoRE']
# f1_scores_test = [49.38,  51.91, 53.84, 51.11]  # test数据
# f1_scores_dev = [49.34,  53.01, 54.29, 49.86]  # dev数据
#
# # 创建x轴的位置
# x = np.arange(len(methods))
#
# # 设置Seaborn样式
# sns.set_theme()
#
# # 绘制直方图
# fig, ax = plt.subplots()
# bar_width = 0.35  # 条形图的宽度
# opacity = 0.8
#
# # 创建直方图
# rects1 = ax.bar(x - bar_width / 2, f1_scores_dev, bar_width, alpha=opacity, label='Dev')
# rects2 = ax.bar(x + bar_width / 2, f1_scores_test, bar_width, alpha=opacity, label='Test')
#
# # 移除右边和上边的线条
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# # 在每个直方图上显示值（堆叠文本）
# for i, rect in enumerate(rects1 + rects2):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}',
#             ha='center', va='bottom', fontsize=15, color='black' if height > 1 else 'white')
#
# # 将图例放在右上角，并确保不与图像重叠
# ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)  # 调整图例的字体大小
#
# # 移除其他网格线
# ax.yaxis.grid(False)
#
# # 添加水平线
# ax.axhline(y=49.34, color='red', linestyle='--', linewidth=1, label='Horizontal Line 1')
# ax.axhline(y=49.38, color='blue', linestyle='--', linewidth=1, label='Horizontal Line 2')
#
# # 设置y轴的范围为40到70
# ax.set_ylim(38, 57)
#
# # 其他设置
# # ax.set_xlabel('不同的范式和模型', fontsize=18)
# ax.set_ylabel('F1 Score', fontsize=18)
# ax.set_xticks(x)
# ax.set_xticklabels(methods, rotation=0, ha="left", fontsize=10)
#
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
methods = ['TAG',  'Mis-AutoRE', 'Vic-AutoRE', 'GLM-AutoRE']
f1_scores_test = [49.38,  51.91, 53.84, 51.11]  # test数据
f1_scores_dev = [49.34,  53.01, 54.29, 49.86]  # dev数据

# 创建x轴的位置
x = np.arange(len(methods))

# 设置Seaborn样式
sns.set_theme()

# 绘制直方图
fig, ax = plt.subplots()
bar_width = 0.35  # 条形图的宽度
opacity = 0.8

# 创建直方图
rects1 = ax.bar(x - bar_width / 2, f1_scores_dev, bar_width, alpha=opacity, label='Dev')
rects2 = ax.bar(x + bar_width / 2, f1_scores_test, bar_width, alpha=opacity, label='Test')

# 移除右边和上边的线条
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 在每个直方图上显示值（堆叠文本）
for i, rect in enumerate(rects1 + rects2):
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points", ha='center', va='bottom', fontsize=12)

# 将图例放在右上角，并确保不与图像重叠
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)  # 调整图例的字体大小

# 移除其他网格线
ax.yaxis.grid(False)
ax.set_ylabel('F1', fontsize=18)
# 添加水平线
ax.axhline(y=49.34, color='red', linestyle='--', linewidth=1, label='Baseline Dev')
ax.axhline(y=49.38, color='blue', linestyle='--', linewidth=1, label='Baseline Test')

# 设置y轴的范围为40到70
ax.set_ylim(38, 57)

# 修改x轴标签的位置，让标签位于两个柱状图的中间位置
ax.set_xticks(x)
ax.set_xticklabels(methods,  fontsize=10)  # 修改标签的旋转角度和对齐方式

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
methods = ["ChatGPT", 'TAG', 'Mistral', 'Vicuna', 'GLM']
f1_scores_test = [0, 49.38, 51.91, 53.84, 51.11]  # test数据
f1_scores_dev = [6.68, 0, 6.59, 4.29, 4.25]

# 将值为0的数据转换为NaN
f1_scores_test = [np.nan if score == 0 else score for score in f1_scores_test]
f1_scores_dev = [np.nan if score == 0 else score for score in f1_scores_dev]

# 创建x轴的位置
x = np.arange(len(methods))

# 设置Seaborn样式
sns.set_theme()

# 绘制直方图
fig, ax = plt.subplots()
bar_width = 0.5  # 条形图的宽度
opacity = 0.8
hatches = ['/', '|', '+', 'x', 'o']
# 创建直方图
rects1 = ax.bar(x - bar_width / 2, f1_scores_dev, bar_width, alpha=opacity, label='NF', hatch=hatches[0])

rects2 = ax.bar(x + bar_width / 2, f1_scores_test, bar_width, alpha=opacity, label='F', hatch=hatches[0])

# 移除右边和上边的线条
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 在每个直方图上显示值（堆叠文本）
for i, rect in enumerate(rects1 + rects2):
    height = rect.get_height()
    if not np.isnan(height):  # 仅在高度不是NaN时显示文本
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=17, color='black' if height > 1 else 'white')

# 将图例放在右上角，并确保不与图像重叠
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)  # 调整图例的字体大小

# 移除其他网格线
ax.yaxis.grid(False)

# 添加水平线
ax.axhline(y=49.34, color='red', linestyle='--', linewidth=1, label='Horizontal Line 1')
ax.axhline(y=49.38, color='blue', linestyle='--', linewidth=1, label='Horizontal Line 2')

# 设置y轴的范围为40到70
ax.set_ylim(0, 65)
ax.tick_params(axis='y', labelsize=20)
# 其他设置
# ax.set_xlabel('不同的范式和模型', fontsize=18)
ax.set_ylabel('F1', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=15)

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 示例数据
# methods = ["ChatGPT", 'TAG', 'Mistral', 'Vicuna', 'GLM']
# f1_scores_test = [0, 49.38, 51.91, 53.84, 51.11]
# f1_scores_dev = [6.68, 0, 6.59, 4.29, 4.25]
#
# # 将值为0的数据转换为NaN
# f1_scores_test = [np.nan if score == 0 else score for score in f1_scores_test]
# f1_scores_dev = [np.nan if score == 0 else score for score in f1_scores_dev]
#
# # 创建x轴的位置
# x = np.arange(len(methods))
#
# # 设置Seaborn样式
# sns.set_theme()
#
# # 绘制直方图
# fig, ax = plt.subplots()
# bar_width = 0.5  # 条形图的宽度
# opacity = 0.8
# hatches = ['/', '|', '+', 'x', 'o']
#
# # 创建直方图
# rects1 = [ax.bar(x_val - bar_width / 2, score, bar_width, alpha=opacity, hatch=hatches[i], label='NF') for i, (x_val, score) in enumerate(zip(x, f1_scores_dev))]
# rects2 = [ax.bar(x_val + bar_width / 2, score, bar_width, alpha=opacity, hatch=hatches[i], label='F') for i, (x_val, score) in enumerate(zip(x, f1_scores_test))]
#
# # 移除右边和上边的线条
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# # 在每个直方图上显示值（堆叠文本）
# for i, rect in enumerate(rects1 + rects2):
#     height = rect[0].get_height()
#     if not np.isnan(height):  # 仅在高度不是NaN时显示文本
#         ax.text(rect[0].get_x() + rect[0].get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=17, color='black' if height > 1 else 'white')
#
# # 将图例放在右上角，并确保不与图像重叠
# ax.legend(('NF', 'F'), loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)  # 调整图例的字体大小
#
# # 移除其他网格线
# ax.yaxis.grid(False)
#
# # 添加水平线
# ax.axhline(y=49.34, color='red', linestyle='--', linewidth=1, label='Horizontal Line 1')
# ax.axhline(y=49.38, color='blue', linestyle='--', linewidth=1, label='Horizontal Line 2')
#
# # 设置y轴的范围为40到70
# ax.set_ylim(0, 65)
# ax.tick_params(axis='y', labelsize=20)
# # 其他设置
# # ax.set_xlabel('不同的范式和模型', fontsize=18)
# ax.set_ylabel('F1', fontsize=20)
# ax.set_xticks(x)
# ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=15)
#
# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 示例数据
# methods = ["ChatGPT", 'TAG', 'Mistral', 'Vicuna', 'GLM']
# hatches = ['/', '|', '+', 'x', 'o']
# f1_scores_test = [0, 49.38, 51.91, 53.84, 51.11]
# f1_scores_dev = [6.68, 0, 6.59, 4.29, 4.25]
#
# # 将值为0的数据转换为NaN
# f1_scores_test = [np.nan if score == 0 else score for score in f1_scores_test]
# f1_scores_dev = [np.nan if score == 0 else score for score in f1_scores_dev]
#
# # 创建x轴的位置
# x = np.arange(len(methods))
#
# # 设置Seaborn样式
# sns.set_theme()
#
# # 绘制直方图
# fig, ax = plt.subplots()
# bar_width = 0.5  # 条形图的宽度
# opacity = 0.8
#
# # 创建直方图并设置填充样式
# rects1 = [ax.bar(pos - bar_width / 2, score, bar_width,
#                  alpha=opacity, edgecolor='gray', hatch=hatches[i])
#           for i, (pos, score) in enumerate(zip(x, f1_scores_dev))]
#
# rects2 = [ax.bar(pos + bar_width / 2, score, bar_width,
#                  alpha=opacity, edgecolor='gray', hatch=hatches[i])
#           for i, (pos, score) in enumerate(zip(x, f1_scores_test))]
#
# # 在每个直方图上显示值（堆叠文本）
# for i, rect in enumerate(rects1 + rects2):
#     height = rect[0].get_height()
#     if not np.isnan(height):  # 仅在高度不是NaN时显示文本
#         ax.text(
#             rect[0].get_x() + rect[0].get_width() / 2, height, f'{height:.2f}',
#             ha='center', va='bottom', fontsize=17,
#             color='black' if height > 1 else 'white'
#         )
#
# # 将图例放在右上角，并确保不与图像重叠
# rects1[0].set_label('NF') # Adds label to bar
# rects2[0].set_label('F')  # Adds label to bar
# ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12) # Adjust the font size of the legend
#
# # 添加水平线
# ax.axhline(y=49.34, color='red', linestyle='--', linewidth=1, label='Horizontal Line 1')
# ax.axhline(y=49.38, color='blue', linestyle='--', linewidth=1, label='Horizontal Line 2')
#
# # 设置y轴的范围为40到70
# ax.set_ylim(0, 65)
# ax.tick_params(axis='y', labelsize=20)
#
# # 设置标签和标题
# ax.set_ylabel('F1', fontsize=20)
# ax.set_xticks(x)
# ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=15)
#
# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
methods = ["ChatGPT", 'TAG', 'Mistral-A', 'Vicuna-A', 'GLM-A']
f1_scores_test = [0, 49.38, 51.91, 53.84, 51.11]  # test数据
f1_scores_dev = [6.68, 0, 6.59, 4.29, 4.25]

# 将值为0的数据转换为NaN
f1_scores_test = [np.nan if score == 0 else score for score in f1_scores_test]
f1_scores_dev = [np.nan if score == 0 else score for score in f1_scores_dev]

# 创建x轴的位置
x = np.arange(len(methods))

# 设置Seaborn样式
sns.set_theme()

# 绘制直方图
fig, ax = plt.subplots()
fig.patch.set_facecolor('none')    # set figure background to none
ax.set_facecolor('none')    # set axes background to none

bar_width = 0.45  # 条形图的宽度
opacity = 0.8
hatches = ['', '|', '+', 'x', 'o']

# 创建直方图
rects1 = ax.bar(x - bar_width / 2, f1_scores_dev, bar_width, alpha=opacity, label='Not-Finetuned')
rects2 = ax.bar(x + bar_width / 2, f1_scores_test, bar_width, alpha=opacity, label='Finetuned')

for i in range(len(rects1)):
    rects1[i].set_hatch(hatches[i])
    rects2[i].set_hatch(hatches[i])

# 在每个直方图上显示值（堆叠文本）
for i, rect in enumerate(rects1 + rects2):
    height = rect.get_height()
    if not np.isnan(height):  # 仅在高度不是NaN时显示文本
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=17, color='black' if height > 1 else 'white')

# 将图例放在右上角，并确保不与图像重叠
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)  # 调整图例的字体大小

# 移除其他网格线
ax.yaxis.grid(False)

# 添加水平线
ax.axhline(y=49.34,  linestyle='--', linewidth=1, label='Horizontal Line 1')
ax.axhline(y=49.38, linestyle='--', linewidth=1, label='Horizontal Line 2')

# 设置y轴的范围为40到70
ax.set_ylim(0, 65)
ax.tick_params(axis='y', labelsize=20)

# ax.set_xlabel('不同的范式和模型', fontsize=18)
ax.set_ylabel('F1', fontsize=20)
ax.set_xticks(x)
import matplotlib.font_manager as font_manager

# Other code

font = font_manager.FontProperties(family='Arial',
                                   size=15,
                                   style='normal',
                                   weight='bold')
ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=15,fontproperties=font)

# 添加坐标轴
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('gray')
plt.tight_layout()
plt.show()