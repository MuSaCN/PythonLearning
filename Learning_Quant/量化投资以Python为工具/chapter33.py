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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

CJSecurities=pd.read_csv(Path+'\\CJSecurities.csv',index_col='Date')
CJSecurities=CJSecurities.iloc[:,1:]
CJSecurities.index=pd.to_datetime(CJSecurities.index)


myDA.myfigpro.reset_figure_axes()
myDA.candle_ohlcV(CJSecurities,False)
myDA.candle_ohlcV(CJSecurities,True)


volume=CJSecurities.Volume
VolSMA5=pd.rolling_apply(volume,5,np.mean).dropna()
VolSMA10=pd.rolling_apply(volume,10,np.mean).dropna()
VolSMA=((VolSMA5+VolSMA10)/2).dropna()
VolSMA.head(n=3)

VolSignal=(volume[-len(VolSMA):]>VolSMA)*1
VolSignal[VolSignal==0]=-1
VolSignal.head()

close=CJSecurities.Close
PrcSMA5=pd.rolling_apply(close,5,np.mean).dropna()
PrcSMA20=pd.rolling_apply(close,20,np.mean).dropna()

def upbreak(Line,RefLine):
    signal=np.all([Line>RefLine,Line.shift(1)<RefLine.shift(1)],axis=0)
    return(pd.Series(signal[1:],index=Line.index[1:]))
def downbreak(Line,RefLine):
    signal=np.all([Line<RefLine,Line.shift(1)>RefLine.shift(1)],axis=0)
    return(pd.Series(signal[1:],index=Line.index[1:]))

UpSMA=upbreak(PrcSMA5[-len(PrcSMA20):],PrcSMA20)*1
DownSMA=downbreak(PrcSMA5[-len(PrcSMA20):],PrcSMA20)*1
SMAsignal=UpSMA-DownSMA
VolSignal=VolSignal[-len(SMAsignal):]
signal=VolSignal+SMAsignal
signal.describe()

trade=signal.replace([2,-2,1,-1,0],[1,-1,0,0,0])
trade=trade.shift(1)[1:]
trade.head()

ret=((close-close.shift(1))/close.shift(1))['2014-01-31':]
ret.name='stockRet'
tradeRet=trade*ret
tradeRet.name='tradeRet'
winRate=len(tradeRet[tradeRet>0])/len(tradeRet[tradeRet!=0])
winRate

(1+ret).cumprod().plot(label='stockRet')
(1+tradeRet).cumprod().plot(label='tradeRet')
plt.legend()


def Hold(signal):
    hold=np.zeros(len(signal))
    for index in range(1,len(hold)):
        if hold[index-1]==0 and signal[index]==1:
            hold[index]=1
        elif hold[index-1]==1 and signal[index]==1:
            hold[index]=1
        elif hold[index-1]==1 and signal[index]==0:
            hold[index]=1
    return(pd.Series(hold,index=signal.index))

hold=Hold(trade)

def TradeSim(price,hold):
    position=pd.Series(np.zeros(len(price)),index=price.index)
    position[hold.index]=hold.values
    cash=20000*np.ones(len(price))
    for t in range(1,len(price)):
        if position[t-1]==0 and position[t]>0:
            cash[t]=cash[t-1]-price[t]*1000
        if position[t-1]>=1 and position[t]==0:
            cash[t]=cash[t-1]+price[t]*1000
        if position[t-1]==position[t]:
            cash[t]=cash[t-1]
    asset=cash+price*position*1000
    asset.name='asset'
    account=pd.DataFrame({'asset':asset,'cash':cash,'position':position})
    return(account)

TradeAccount=TradeSim(close,hold)
TradeAccount.tail()
TradeAccount.plot(subplots=True,\
        title='成交量与均线策略交易账户')








