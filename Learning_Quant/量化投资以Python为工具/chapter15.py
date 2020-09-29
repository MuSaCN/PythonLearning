# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
from scipy import stats
import MyPackage

__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

#1
au=np.array((6.683,6.678,6.767,6.692,6.672,6.678))
pa=np.array((6.661,6.664,6.668,6.666,6.665))
myDA.interval_estimation(au,0.90)
myDA.interval_estimation(pa,0.90)

#5.
import math
Bwages = pd.read_csv(Path2+'/Part2/002/Bwages.csv')
stats.ttest_1samp(Bwages.wage,0)
Bwages.wage.hist(normed=True)
plt.show()

mu = Bwages.wage.mean()
std = Bwages.wage.std()
low = mu - stats.t.ppf(0.975,len(Bwages)-1) * std / math.sqrt(len(Bwages))
high = mu + stats.t.ppf(0.975,len(Bwages)-1) * std / math.sqrt(len(Bwages))
low
high

#6.
Bwages.wage.hist(bins=100,normed=True)
bins=np.linspace(0,50,200)
plt.plot(bins,stats.norm.pdf(bins,mu,std))
plt.show()

#7
MT=(0.225,0.262,0.217,0.24,0.23,0.229,0.235,0.217)
Sn=(0.209,0.205,0.196,0.21,0.202,0.207,0.224,0.223)
stats.ttest_ind(MT,Sn)
myDA.ttest_2samp(MT,Sn,True)

#9
Bwages = pd.read_csv(Path2+'/Part2/002/Bwages.csv')
Bwages["wage"].mean()
myDA.ttest_1samp(Bwages["wage"],11)

#10
history = pd.read_csv(Path2+'/Part2/001/history.csv', index_col = 'Date')
stats.ttest_ind(history['Emerging.Markets'], history['Global.Macro'])

#11.
stats.ttest_rel(history['Emerging.Markets'],history['Global.Macro'])


