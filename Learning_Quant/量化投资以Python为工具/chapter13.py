# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
import MyPackage

__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #高级对象式绘图类
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#------------------------------------------------------------------------------------------

Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答\\Part2\\001"
File2=Path2+"\\history.csv"
data2=myfile.read_pd(File2,",",index=["Date"],parse_dates=True)
data2.head()
myDA.describe(data2,False)
myDA.describe(data2['Emerging.Markets'],True)

#1.
history = pd.read_csv(File2,index_col = 'Date')
history.index = pd.to_datetime(history.index,format='%Y-%m-%d')
history.head()
history['Emerging.Markets'].mean()
history['Emerging.Markets'].median()
history['Emerging.Markets'].mode()
history['Emerging.Markets'].quantile([0.1,0.9])

#2.
myDA.describe(data2['Event.Driven'],True)
history['Event.Driven'].max() - history['Event.Driven'].min()
history['Event.Driven'].mad()
history['Event.Driven'].var()
history['Event.Driven'].std()

#3.
history[['Relative.Value','Fixed.Income.Arbitrage']].plot()
plt.show()
history['Relative.Value'].mean()
history['Fixed.Income.Arbitrage'].mean()
history['Relative.Value'].std()
history['Fixed.Income.Arbitrage'].std()

#4.
history.describe()




