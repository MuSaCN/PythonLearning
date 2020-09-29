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
myfile = MyPackage.MyClass_File.MyClass_File()                #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()                 #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()               #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()        #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()      #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()                #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()               #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()      #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()                #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------

Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\018"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"



#9.
import pandas_datareader.data as web

import datetime as dt

import numpy as np

start = dt.datetime(2014,1,1)

end = dt.datetime(2014,12,31)

baidu = web.DataReader('BIDU','yahoo',start,end)

returns = (baidu.Close - baidu.Close.shift(1))/baidu.Close.shift(1)
R=myDA.to_returns(baidu.Close)


comp_returns = np.log(baidu.Close/baidu.Close.shift(1))
LR=myDA.to_log_returns(baidu.Close)


comp_returns.sum()
LR.sum()
np.exp(LR.cumsum()).plot()
LR.cumsum().plot()
returns.plot()

plt.show()
comp_returns.plot()












