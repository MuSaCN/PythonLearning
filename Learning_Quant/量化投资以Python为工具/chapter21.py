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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\021"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import statsmodels.formula.api as smf
from statsmodels.api import add_constant
import statsmodels.api as sm
# 1.
wanke = web.DataReader('000002.SZ', 'yahoo', dt.datetime(2015, 1, 1), dt.datetime(2015, 12, 31))
gldc = web.DataReader('600185.SS', 'yahoo', dt.datetime(2015, 1, 1), dt.datetime(2015, 12, 31))
price = pd.concat([wanke.Close, gldc.Close], 1)
price.columns = ['wanke', 'geli']
ret = (price - price.shift(1)) / price.shift(1)
model = smf.ols('wanke~geli', data=ret).fit()
print(model.summary())

# 5.
zyhy = pd.read_table(Path2+'/Part3/004/problem21.txt',sep='\t', usecols=['zyhy', 'Date'], index_col='Date')
zyhy.index = pd.to_datetime(zyhy.index)

ret = (zyhy - zyhy.shift(1)) / zyhy.shift(1)

ThreeFactors = pd.read_table(Path2+'/Part3/004/ThreeFactors.txt',sep='\t', index_col='TradingDate')
ThreeFactors.index = pd.to_datetime(ThreeFactors.index)
ThrFac = ThreeFactors['2014']
ThrFac.columns
ThrFac = ThrFac.iloc[:, [2, 4, 6]]
dat = pd.concat([ret, ThrFac], 1).dropna()
dat = pd.concat([ret, ThrFac], 1)
model = smf.ols('zyhy~RiskPremium2+SMB2+HML2', data=dat).fit()
print(model.summary())
myDA.three_factors(dat.zyhy,dat.RiskPremium2,dat.SMB2,dat.HML2,False)

# 6.
zhongxin = pd.read_table(Path2+'/Part3/004/problem21.txt',sep='\t', usecols=['zhongxin', 'Date'], index_col='Date')
zhongxin.index = pd.to_datetime(zhongxin.index)
ret = (zhongxin - zhongxin.shift(1)) / zhongxin.shift(1)

ThreeFactors = pd.read_table(Path2+'/Part3/004/ThreeFactors.txt',sep='\t', index_col='TradingDate')
ThreeFactors.index = pd.to_datetime(ThreeFactors.index)
ThrFac = ThreeFactors['2014']
ThrFac = ThrFac.iloc[:, [2, 4, 6]]
dat = pd.concat([ret, ThrFac], 1)

model = smf.ols('zhongxin~RiskPremium2', data=dat).fit()
print(model.summary())

model2 = smf.ols('zhongxin~RiskPremium2+SMB2+HML2', data=dat).fit()
print(model2.summary())

ThrFac = ThreeFactors['2015-01']
preCAPM = model.predict(add_constant(ThrFac.RiskPremium2),transform=False)
preFactors = model2.predict(add_constant(ThrFac[['RiskPremium2', 'SMB2', 'HML2']]),transform=False)
preCAPM.plot()
preFactors.plot()
plt.show()

# 7.
codes = pd.read_csv(Path2+'/Part3/004/codes.csv', header=None, dtype=str)

ThreeFactors = pd.read_table(Path2+'/Part3/004/ThreeFactors.txt', sep='\t', index_col='TradingDate')

ThreeFactors.index = pd.to_datetime(ThreeFactors.index)

ThrFac = ThreeFactors['2014']

ThrFac = ThrFac.iloc[:, [2, 4, 6]]


def create_func(model):
    def cal_alpha(code, model_name=model):
        price = web.DataReader(code, 'yahoo', dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31)).Close
        ret = (price - price.shift(1)) / price.shift(1)
        ret.name = 'ret'
        dat = pd.concat([ret, ThrFac], 1)
        if model_name == 'CAPM':
            model = smf.ols('ret~RiskPremium2', data=dat).fit()
        elif model_name == 'factors':
            model = smf.ols('ret~RiskPremium2+SMB2+HML2', data=dat).fit()
        return (model.params[0])
    return (cal_alpha)


alpha_CAPM = list(map(create_func('CAPM'), codes[0].values))
alpha_CAPM2 = pd.Series(alpha_CAPM).sort(ascending=False, inplace=False)
alpha_CAPM2[:3]

alpha_factors = list(map(create_func('factors'), codes[0]))
alpha_factors2 = pd.Series(alpha_factors).sort(ascending=False, inplace=False)
alpha_factors2[:3]










