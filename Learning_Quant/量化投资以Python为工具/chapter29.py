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
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()          #回测类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\029"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

#strategy
BOCM=pd.read_csv(Path+'\\BOCM.csv')
BOCM.index=BOCM.iloc[:,1]
BOCM.index=pd.to_datetime(BOCM.index, format='%Y-%m-%d')
BOCMclp=BOCM.Close

rsi6=myDA.rsi(BOCMclp,6)
rsi24=myDA.rsi(BOCMclp,24)

# rsi6捕捉买卖点
Signal1=pd.Series(0,index=rsi6.index)
for i in rsi6.index:
    if rsi6[i]>80:
        Signal1[i]= -1
    elif rsi6[i]<20:
        Signal1[i]= 1
    else:
        Signal1[i]= 0

# 交叉信号
Signal2=pd.Series(0,index=rsi24.index)
lagrsi6= rsi6.shift(1)
lagrsi24= rsi24.shift(1)
for i in rsi24.index:
    if (rsi6[i]>rsi24[i]) & (lagrsi6[i]<lagrsi24[i]):
        Signal2[i]=1
    elif (rsi6[i]<rsi24[i]) & (lagrsi6[i]>lagrsi24[i]):
        Signal2[i]=-1

# 信号合并
signal=Signal1+Signal2
signal[signal>=1]=1
signal[signal<=-1]=-1
signal=signal.dropna()

myBT.SignalQuality(signal,price_Series=BOCM.Close,holding=1,lag_trade=3,plotRet=True,plotStrat=True)



