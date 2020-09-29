# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
import MyPackage
__mypath__=MyPackage.MyClass_Path.MyClass_Path("\\利用Python进行数据分析") #路径类
myfile=MyPackage.MyClass_File.MyClass_File()            #文件操作类
myplt=MyPackage.MyClass_Plot.MyClass_Plot()             #直接绘图类(单个图窗)
myfig=MyPackage.MyClass_Plot.MyClass_Figure()           #对象式绘图类(可多个图窗)
myfigpro=MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #高级对象式绘图类
mynp=MyPackage.MyClass_Array.MyClass_NumPy()            #多维数组类(整合Numpy)
mypd=MyPackage.MyClass_Array.MyClass_Pandas()           #矩阵数组类(整合Pandas)
mypdpro=MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime=MyPackage.MyClass_Time.MyClass_Time()            #时间类
#---------------------------------------------------------
path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\利用Python进行数据分析(第二版)代码\\examples\\"
path1="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\Python数据科学手册\\notebooks\\data\\"
myfig.set_axes_3d2d()
myfig.reset_figure_axes()
# myfig.AxesList[0].
# myfig.fig.
myfig.show()
plt.show()
print(123)
# --------

start=1
count=0
while start>0:
    if np.random.rand() <0.5:
        start+=-1
    elif np.random.rand() >= 0.5:
        start+=1
    count+=1
    print(count,start)








# --------

start=1
count=0
while start>0:
    if np.random.rand() <0.5:
        start+=-1
    elif np.random.rand() >= 0.5:
        start+=1
    count+=1
    print(count,start)











