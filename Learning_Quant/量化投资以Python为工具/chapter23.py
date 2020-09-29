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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\023"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

#5.
CRSPday=pd.read_csv(Path2+'/Part4/002/CRSPday.csv')
ibm=CRSPday.ibm
ibm.plot()
plt.show()
myDA.TSA_acf(ibm,nlags=20,plot=True)
myDA.TSA_acf(ibm,nlags=20,qstat=True)

from statsmodels.graphics.tsaplots import *
plot_acf(ibm,lags=20)

from statsmodels.tsa import stattools
LjungBox=stattools.q_stat(stattools.acf(ibm)[1:13],len(ibm))
LjungBox[1][-1]

#6.
ge=CRSPday.iloc[:,3]
ge.plot()
plt.show()
plot_acf(ge,lags=20)

LjungBox=stattools.q_stat(stattools.acf(ge)[1:2],len(ge))
LjungBox[1][-1]

myDA.TSA_acf(ge,nlags=9,qstat=True)

LjungBox=stattools.q_stat(stattools.acf(ge)[1:9],len(ge))
LjungBox[1][-1]

#7.
SP500=pd.read_csv(Path2+'/Part4/002/SP500.csv')
r500=SP500.r500
r500.plot()

plot_acf(r500,lags=20)
plot_pacf(r500,lags=20)

from arch.unitroot import ADF
adf=ADF(r500,lags=3)
print(adf.summary().as_text())
plt.show()
myDA.tsa_ADF(r500)




