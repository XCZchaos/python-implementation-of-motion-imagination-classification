import pandas as pd
import matplotlib.pyplot as plt

# 定义数据
data = {
    'methods': ['FBCSP', 'ConvNet', 'EEGNet', 'C2CM', 'FBCNet', 'DRDA', 'Conformer'],
    'S01': [76.00, 76.39, 85.76, 87.50, 85.42, 83.19, 88.19],
    'S02': [56.50, 55.21, 61.46, 65.28, 60.42, 55.14, 61.46],
    'S03': [81.25, 89.24, 88.54, 90.28, 90.63, 87.43, 93.40],
    'S04': [61.00, 74.65, 67.01, 66.67, 76.39, 75.28, 78.13],
    'S05': [55.00, 56.94, 55.90, 62.50, 74.31, 62.29, 52.08],
    'S06': [45.25, 54.17, 52.08, 45.49, 53.82, 57.15, 65.28],
    'S07': [82.75, 92.71, 89.58, 89.58, 84.38, 86.18, 92.36],
    'S08': [81.25, 77.08, 83.33, 83.33, 79.51, 83.61, 88.19],
    'S09': [70.75, 76.39, 86.81, 79.51, 80.90, 82.00, 88.89],
    'average': [67.75, 72.53, 74.50, 74.46, 76.20, 74.74, 78.66],
    'kappa': [0.5700, 0.6337, 0.6600, 0.6595, 0.6827, 0.6632, 0.7155]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建表格并保存为图片
fig, ax = plt.subplots(figsize=(12, 5))  # 设置图像大小
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# 设置表格样式
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)

# 设置三线表的样式
for k, cell in the_table.get_celld().items():
    cell.set_linewidth(0)

# 添加三条线
ax.plot([0, 1], [1, 1], color='black', lw=2, transform=ax.transAxes, clip_on=False)
ax.plot([0, 1], [0, 0], color='black', lw=2, transform=ax.transAxes, clip_on=False)
ax.plot([0, 1], [0.5, 0.5], color='black', lw=2, transform=ax.transAxes, clip_on=False)

# plt.savefig("/mnt/data/table_image.png", bbox_inches='tight', dpi=300)

# 显示表格
plt.show()
