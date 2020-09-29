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
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\034"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"



TsingTao=pd.read_csv(Path+'\\TsingTao.csv',index_col='Date')
TsingTao=TsingTao.dropna()
TsingTao.index = pd.to_datetime(TsingTao.index)
a=myDA.obv_indi(TsingTao)



import ffn
def trade(obv,price):
    signal=(2*(obv.diff()>0)-1)[1:]
    ret=ffn.to_returns(price)[1:]
    ret.name='ret'
    tradeRet=ret*signal.shift(1)
    tradeRet.name='tradeRet'
    Returns=pd.merge(pd.DataFrame(ret),\
                     pd.DataFrame(tradeRet),\
                    left_index=True,right_index=True).dropna()
    return(Returns)

OBVtrade=trade(OBV,close)
OBVtrade.head()

ret=OBVtrade.ret
tradeRet=OBVtrade.tradeRet
ret.name='BuyAndHold'
tradeRet.name='OBVTrade'
(1+ret).cumprod().plot(label='ret',linestyle='dashed')
(1+tradeRet).cumprod().plot(label='tradeRet')
plt.title('累积OBV交易策略绩效表现')
plt.legend()

def backtest(ret,tradeRet):
    def performance(x):
        winpct=len(x[x>0])/len(x[x!=0])
        annRet=(1+x).cumprod()[-1]**(245/len(x))-1
        sharpe=ffn.calc_risk_return_ratio(x)
        maxDD=ffn.calc_max_drawdown((1+x).cumprod())
        perfo=pd.Series([winpct,annRet,sharpe,maxDD],index=['win rate',
                        'annualized return',\
                        'sharpe ratio','maximum drawdown'])
        return(perfo)
    BuyAndHold=performance(ret)
    OBVTrade=performance(tradeRet)
    return(pd.DataFrame({ret.name:BuyAndHold,\
    tradeRet.name:OBVTrade}))

OBVtest=backtest(ret,tradeRet)
OBVtest

smOBVtrade=trade(smOBV,close)
smOBVtrade.head(n=3)

ret=smOBVtrade.ret
ret.name='BuyAndHold'
smtradeRet=smOBVtrade.tradeRet
smtradeRet.name='smOBVTrade'
(1+ret).cumprod().plot(label='ret',linestyle='dashed')
(1+tradeRet).cumprod().plot(label='tradeRet')
plt.title('简单OBV交易策略绩效表现')
plt.legend()

test=backtest(ret,smtradeRet)
test
