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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\024"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

#读取数据
CPI=pd.read_csv(Path+'/CPI.csv',index_col='time')
CPI.index=pd.to_datetime(CPI.index)
CPItrain=CPI[3:]
CPItrain=CPItrain.dropna().CPI
timeseries = CPItrain
# ---
myDA.tsa_auto_test(timeseries)
myDA.tsa_auto_ARIMA(timeseries)

#上证指数的平稳时间序列建模
Datang=pd.read_csv(Path+'/Datang.csv',index_col='time')
Datang.index=pd.to_datetime(Datang.index)
returns=Datang.datang['2014-01-01':'2016-01-01']
timeseries1 = returns
# ---
myDA.tsa_auto_test(timeseries1)
myDA.tsa_auto_ARIMA(timeseries1,method="BIC")

# ------------------------------------------------------------------------
import statsmodels.tsa.arima_process as sm
from statsmodels.graphics.tsaplots import *

#4.
arma=sm.ArmaProcess([-1,-0.6],[1])
sample=arma.generate_sample(200)
plot_acf(sample,lags=20)
plot_pacf(sample,lags=20)
plt.show()

#5.
arma=sm.ArmaProcess([-1],[1,0.4])
sample=arma.generate_sample(200)
plot_acf(sample,lags=20)
plot_pacf(sample,lags=20)
plt.show()

#6.
import statsmodels.tsa.arima_process as sm
from statsmodels.graphics.tsaplots import *
import numpy as np
import pandas as pd
numbers=np.random.normal(size=100)
numbers=pd.Series(numbers)

numbers.plot()
plt.show()
plot_acf(numbers,lags=20)

from statsmodels.tsa import stattools
stattools.arma_order_select_ic(numbers.values,max_ma=4)

#7.
zgsy=pd.read_csv('Data/Part4/003/zgsy.csv')
clprice=zgsy.iloc[:,4]
clprice.plot()
plot_acf(clprice,lags=20)
from arch.unitroot import ADF
adf=ADF(clprice,lags=6)
print(adf.summary().as_text())

logReturn=pd.Series((np.log(clprice))).diff().dropna()
logReturn.plot()

adf=ADF(logReturn,lags=6)
print(adf.summary().as_text())

plot_acf(logReturn,lags=20)
plot_pacf(logReturn,lags=20)

from statsmodels.tsa import arima_model
model1=arima_model.ARIMA(logReturn.values,order=(0,0,1)).fit()
model1.summary()

model2=arima_model.ARIMA(logReturn.values,order=(1,0,0)).fit()
model2.summary()

#8.
baiyun=zgsy=pd.read_csv('Data/Part4/003/baiyun.csv',index_col='Date')
baiyun.index=pd.to_datetime(baiyun.index)
clprice=baiyun.Close

logReturn=pd.Series((np.log(clprice))).diff().dropna()
logReturn.plot()

adf=ADF(logReturn,lags=6)
print(adf.summary().as_text())

plot_acf(logReturn,lags=20)
plot_pacf(logReturn,lags=20)
model1=arima_model.ARIMA(logReturn.values,order=(0,0,2)).fit()
model2=arima_model.ARIMA(logReturn.values,order=(2,0,0)).fit()
model1.aic
model2.aic

import math
stdresid=model2.resid/math.sqrt(model2.sigma2)
stdresid.plot()
plot_acf(stdresid,lags=20)
LjungBox=stattools.q_stat(stattools.acf(stdresid)[1:13],len(stdresid))
LjungBox[1][-1]

pd.Series(model2.forecast(10)[0]).plot()



