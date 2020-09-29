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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\032"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

GSPC=pd.read_csv(Path+'\\GSPC.csv',index_col='Date')
GSPC=GSPC.iloc[:,1:]
GSPC.index=pd.to_datetime(GSPC.index)
close=GSPC.Close
high=GSPC.High
low=GSPC.Low
RSV = myDA.rsv_indi(GSPC,9)
KDJ = myDA.kdj_indi(GSPC,9)

KSignal=KDJ.KValue.apply(lambda x:-1 if x>85 else 1 if x<20 else 0)
DSignal=KDJ.DValue.apply(lambda x:-1 if x>80 else 1 if x<20 else 0)
KDSignal=KSignal+DSignal
KDSignal.name='KDSignal'
myBT.SignalQuality(KDSignal,price_Series=close,plotStrat=True,plotRet=True)


KDSignal[KDSignal>=1]==1
KDSignal[KDSignal<=-1]==-1
KDSignal.head(n=3)
KDSignal[KDSignal==1].head(n=3)

def trade(signal,price):
    ret=((price-price.shift(1))/price.shift\
         (1))[1:]
    ret.name='ret'
    signal=signal.shift(1)[1:]
    tradeRet=ret*signal+0
    tradeRet.name='tradeRet'
    Returns=pd.merge(pd.DataFrame(ret),\
                     pd.DataFrame(tradeRet),
                     left_index=True,\
                     right_index=True).dropna()
    return(Returns)
myBT.SignalQuality(KDSignal,price_Series=close,holding=1,plotRet=False,plotStrat=False)

KDtrade=trade(KDSignal,close)
KDtrade.rename(columns={'ret':'Ret','tradeRet':'KDtradeRet'},inplace=True)

import ffn
def backtest(ret,tradeRet):
    def performance(x):
        winpct=len(x[x>0])/len(x[x!=0])
        annRet=(1+x).cumprod()[-1]**(245/len(x))-1
        sharpe=ffn.calc_risk_return_ratio(x)
        maxDD=ffn.calc_max_drawdown((1+x).cumprod())
        perfo=pd.Series([winpct,annRet,sharpe,maxDD],\
        index=['win rate','annualized return',\
        'sharpe ratio','maximum drawdown'])
        return(perfo)
    BuyAndHold=performance(ret)
    Trade=performance(tradeRet)
    return(pd.DataFrame({ret.name:BuyAndHold,\
           tradeRet.name:Trade}))

backtest(KDtrade.Ret,KDtrade.KDtradeRet)

cumRets1=(1+KDtrade).cumprod()
plt.plot(cumRets1.Ret,label='Ret')
plt.plot(cumRets1.KDtradeRet,'--',\
          label='KDtradeRet')
plt.title('KD指标交易策略绩效表现')
plt.legend()
plt.show()
myBT.SignalQuality(KDSignal[:'2014-10-10'],price_Series=close[:'2014-10-10'],plotRet=False,plotStrat=False)

backtest(KDtrade.Ret[:'2014-10-10'],\
          KDtrade.KDtradeRet[:'2014-10-10'])

cumRets2=(1+KDtrade[:'2014-10-10']).cumprod()
plt.plot(cumRets2.Ret,\
          label='''Ret[:'2014-10-10']''')
plt.plot(cumRets2.KDtradeRet,'--',\
          label='''KDtradeRet[:'2014-10-10']''')
plt.title('KD指标交易策略10月10日之前绩效表现')
plt.legend(loc='upper left')
plt.show()

JSignal=JValue.apply(lambda x:\
         -1 if x>100 else 1 if x<0 else 0)


KDJSignal=KSignal+DSignal+JSignal
KDJSignal=KDJSignal.apply(lambda x:\
          1 if x>=2 else -1 if x<=-2 else 0)

KDJtrade=trade(KDJSignal,close)
KDJtrade.rename(columns={'ret':'Ret',\
             'tradeRet':'KDJtradeRet'},\
             inplace=True)
backtest(KDJtrade.Ret,KDJtrade.KDJtradeRet)

KDJCumRet=(1+KDJtrade).cumprod()
plt.plot(KDJCumRet.Ret,label='Ret')
plt.plot(KDJCumRet.KDJtradeRet,'--',\
          label='KDJtradeRet')
plt.title('KDJ指标交易策略绩效表现')
plt.legend(loc='upper left')

backtest(KDJtrade.Ret[:'2014-10-10'],\
             KDJtrade.KDJtradeRet[:'2014-10-10'])

def upbreak(Line,RefLine):
    signal=np.all([Line>RefLine,\
                   Line.shift(1)<RefLine.shift(1)],\
                   axis=0)
    return(pd.Series(signal[1:],\
                     index=Line.index[1:]))

KDupbreak=upbreak(KValue,DValue)*1
KDupbreak[KDupbreak==1].head()

def downbreak(Line,RefLine):
    signal=np.all([Line<RefLine,\
                   Line.shift(1)>RefLine.shift(1)],\
                   axis=0)
    return(pd.Series(signal[1:],\
           index=Line.index[1:]))

KDdownbreak=downbreak(KValue,DValue)*1
KDdownbreak[KDdownbreak==1].head()

close=close['2014-01-14':]
difclose=close.diff()

prctrend=2*(difclose[1:]>=0)-1
prctrend.head()

KDupSig=(KDupbreak[1:]+prctrend)==2
KDupSig.head(n=3)

KDdownSig=pd.Series(np.all([KDdownbreak[1:]==1,prctrend==-1],\
                    axis=0),\
                  index=prctrend.index)

breakSig=KDupSig*1+KDdownSig*-1
breakSig.name='breakSig'
breakSig.head()

KDbreak=trade(breakSig,close)
KDbreak.rename(columns={'ret':'Ret',\
              'tradeRet':'KDbreakRet'},\
              inplace=True)
KDbreak.head()

backtest(KDbreak.Ret,KDbreak.KDbreakRet)

KDbreakRet=(1+KDbreak).cumprod()
plt.plot(KDbreakRet.Ret,label='Ret')
plt.plot(KDbreakRet.KDbreakRet,'--',\
          label='KDbreakRet')
plt.title('KD"金叉"与"死叉"交易策略绩效表现')
plt.legend(loc='upper left')






