# =============================================================================
# 7.2.1 数据可视化基础.py  by 王宇韬&肖金鑫
# =============================================================================

# 如果是用pycharm运行，要把生成的图片关掉，才能弹出下一张图片

# 导入模块
import numpy as np
from matplotlib import pyplot as plt

# 1 折线图
# 1.1 折线图1
x = [1, 2, 3]
y = [2, 4, 6]
# color设置颜色，linewidth设置线宽，单位像素，linestyle默认为实线，“--”表示虚线
plt.plot(x, y, color='red', linewidth=3, linestyle='--')

# 绘制并展示图形
plt.show()

# 1.1 折线图2-多条线
# 第一条线：y = x*2
x1 = np.array([1, 2, 3])
y1 = x1*2
# color设置颜色，linewidth设置线宽，单位像素，linestyle默认为实线，“--”表示虚线
plt.plot(x1, y1, color='red', linewidth=3, linestyle='--')

# 第二条线：y = x + 1
y2 = x1 + 1
plt.plot(x1, y2) # 使用默认参数画图

plt.show()

# 2 柱状图
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
plt.bar(x, y)
plt.show()

# 3 添加文字说明
x = [1, 2, 3]
y = [2, 4, 6]
plt.plot(x, y)
plt.title('TITLE')  # 添加标题
plt.xlabel('X')  # 添加X轴
plt.ylabel('Y')  # 添加Y轴
plt.show()  # 显示图片

# 4 添加图例
# 第一条线, 设定标签lable为y = x*2
x1 = np.array([1, 2, 3])
y1 = x1*2
plt.plot(x1, y1, color='red', linestyle='--', label='y = x*2')
# 第二条线, 设定标签lable为y = x + 1
y2 = x1 + 1
plt.plot(x1,y2,label='y = x + 1')

plt.legend(loc='upper left')# 图例位置设置为左上角
plt.show()

# 5 设置双坐标轴
# 第一条线, 设定标签lable为y = x
x1 = np.array([10, 20, 30])
y1 = x1
plt.plot(x1, y1, color='red', linestyle='--', label='y = x')
plt.legend(loc='upper left')  # 该图图例设置在左上角

plt.twinx()  # 设置双坐标轴

# 第二条线, 设定标签lable为y = x^2
y2 = x1*x1
plt.plot(x1, y2, label='y = x^2')
plt.legend(loc='upper right') # 改图图例设置在右上角

plt.show()

