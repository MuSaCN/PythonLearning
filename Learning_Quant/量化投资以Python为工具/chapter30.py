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
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\030"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

# ChinaBank
ChinaBank = pd.read_csv(Path+'\\ChinaBank.csv')
ChinaBank.index = ChinaBank.iloc[:, 1]
ChinaBank.index = pd.to_datetime(ChinaBank.index, format='%Y-%m-%d')
ChinaBank = ChinaBank.iloc[:, 2:]
CBClose = ChinaBank.Close

# short  and  long
Ssma5 = myDA.sma_Indi(CBClose, 5);
Lsma30 = myDA.sma_Indi(CBClose, 30);
SLSignal = pd.Series(0, index=Lsma30.index)
for i in range(1, len(Lsma30)):
    if all([Ssma5[i] > Lsma30[i], Ssma5[i - 1] < Lsma30[i - 1]]):
        SLSignal[i] = 1
    elif all([Ssma5[i] < Lsma30[i], Ssma5[i - 1] > Lsma30[i - 1]]):
        SLSignal[i] = -1
# myBT.SignalQuality(SLSignal,price_Series=CBClose)


# MACD
a=myDA.macd_indi(CBClose,plot=False)
myDA.ema_indi(CBClose)
myDA.bias_indi(CBClose,38)

DIF = a[0]
DEA = a[1]
MACD = a[2]

macddata = pd.DataFrame()
macddata['DIF'] = DIF['2015']
macddata['DEA'] = DEA['2015']
macddata['MACD'] = MACD['2015']


myDA.CandlePlot_ohlc(ChinaBank['2015'],candleTitle='中国银行2015年日K线图',splitFigures=True, Data=macddata,ylabel='MACD')



macdSignal = pd.Series(0, index=DIF.index)
for i in range(1, len(DIF)):
    if all([DIF[i] > DEA[i] > 0.0, DIF[i - 1] < DEA[i - 1]]):
        macdSignal[i] = 1
    elif all([DIF[i] < DEA[i] < 0.0, DIF[i - 1] > DEA[i - 1]]):
        macdSignal[i] = -1
macdSignal.tail()

myBT.SignalQuality(macdSignal,price_Series=CBClose)

macdTrade = macdSignal.shift(1)

CBRet = CBClose / CBClose.shift(1) - 1
macdRet = (CBRet * macdTrade).dropna()
macdRet[macdRet == -0] = 0
macdWinRate = len(macdRet[macdRet > 0]) / len(macdRet[macdRet != 0])
macdWinRate

AllSignal =  SLSignal + macdSignal
for i in AllSignal.index:
    if AllSignal[i] > 1:
        AllSignal[i] = 1
    elif AllSignal[i] < -1:
        AllSignal[i] = -1
    else:
        AllSignal[i] = 0

AllSignal[AllSignal == 1]
AllSignal[AllSignal == -1]

tradSig = AllSignal.shift(1).dropna()

CBClose = CBClose[-len(tradSig):]
asset = pd.Series(0.0, index=CBClose.index)
cash = pd.Series(0.0, index=CBClose.index)
share = pd.Series(0, index=CBClose.index)

# 当价格连续两天上升且交易信号没有显示卖出时，
# 第一次开账户持有股票
entry = 3
cash[:entry] = 20000
while entry < len(CBClose):
    cash[entry] = cash[entry - 1]
    if all([CBClose[entry - 1] >= CBClose[entry - 2], \
            CBClose[entry - 2] >= CBClose[entry - 3], \
            AllSignal[entry - 1] != -1]):
        share[entry] = 1000
        cash[entry] = cash[entry] - 1000 * CBClose[entry]
        break
    entry += 1

# 根据sigal买卖股票
i = entry + 1
while i < len(tradSig):
    cash[i] = cash[i - 1]
    share[i] = share[i - 1]
    flag = 1
    if tradSig[i] == 1:
        share[i] = share[i] + 3000
        cash[i] = cash[i] - 3000 * CBClose[i]

    if all([tradSig[i] == -1, share[i] > 0]):
        share[i] = share[i] - 1000
        cash[i] = cash[i] + 1000 * CBClose[i]
    i += 1

asset = cash + share * CBClose

plt.subplot(411)
plt.title("2014-2015年上:中国银行均线交易账户")
plt.plot(CBClose, color='b')
plt.ylabel("Pricce")
plt.subplot(412)
plt.plot(share, color='b')
plt.ylabel("Share")
plt.ylim(0, max(share) + 1000)

plt.subplot(413)
plt.plot(asset, label="asset", color='r')
plt.ylabel("Asset")
plt.ylim(min(asset) - 5000, max(asset) + 5000)

plt.subplot(414)
plt.plot(cash, label="cash", color='g')
plt.ylabel("Cash")
plt.ylim(0, max(cash) + 5000)

TradeReturn = (asset[-1] - 20000) / 20000
TradeReturn

plt.show()


