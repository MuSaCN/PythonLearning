# Author:Zhang Yuan
import MyPackage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
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
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\022"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"


#2.
Yen=pd.read_csv(Path2+'/Part4/001/Yen.csv',index_col='date')
Yen.index=pd.to_datetime(Yen.index,format='%Y%m%d')

Yen=myfile.read_pd(Path2+'/Part4/001/Yen.csv',index_col="date",parse_dates=True)

#3.
Yen.s.plot()
plt.show()

#4.
pfyh=pd.read_csv(Path2+'/Part4/001/pfyh.csv',index_col='Date')
pfyh.index=pd.to_datetime(pfyh.index,format='%Y-%m-%d')
returns=(pfyh.Close-pfyh.Close.shift(1))/pfyh.Close.shift(1)
returns=returns[1:]
returns.plot()
plt.show()

