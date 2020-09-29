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
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\031"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

ChinaUnicom=pd.read_csv(Path+'\\ChinaUnicom.csv')
ChinaUnicom.index=ChinaUnicom.iloc[:,1]
ChinaUnicom.index=pd.to_datetime(ChinaUnicom.index, format='%Y-%m-%d')
ChinaUnicom=ChinaUnicom.iloc[:,2:]

Close=ChinaUnicom.Close
High=ChinaUnicom.High
Low=ChinaUnicom.Low

bound = myDA.highlow_indi(ChinaUnicom,20)

# ---plot
myDA.CandlePlot_ohlc(ChinaUnicom["2011"],axesindex=0,show=False)
myDA.myfigpro.myfig.plot_line(bound.upboundDC["2011"],axesindex=0,show=False)
myDA.myfigpro.myfig.plot_line(bound.downboundDC["2011"],axesindex=0,show=False)
myDA.myfigpro.myfig.show()
myDA.myfigpro.reset_figure_axes()


def upbreak(tsLine,tsRefLine):
    n=min(len(tsLine),len(tsRefLine))
    tsLine=tsLine[-n:]
    tsRefLine=tsRefLine[-n:]
    signal=pd.Series(0,index=tsLine.index)
    for i in range(1,len(tsLine)):
        if all([tsLine[i]>tsRefLine[i],tsLine[i-1]<tsRefLine[i-1]]):
            signal[i]=1
    return(signal)

def downbreak(tsLine,tsRefLine):
    n=min(len(tsLine),len(tsRefLine))
    tsLine=tsLine[-n:]
    tsRefLine=tsRefLine[-n:]
    signal=pd.Series(0,index=tsLine.index)
    for i in range(1,len(tsLine)):
        if all([tsLine[i]<tsRefLine[i],tsLine[i-1]>tsRefLine[i-1]]):
            signal[i]=1
    return(signal)

#DC Strategy
UpBreak=upbreak(Close[bound.upboundDC.index[0]:],bound.upboundDC)
DownBreak=downbreak(Close[bound.downboundDC.index[0]:],bound.downboundDC)
BreakSig=UpBreak-DownBreak
myBT.SignalQuality(BreakSig,price_Series=Close,holding=1)

#BBands
UnicomBBands=myDA.bbands_Indi(Close,20,2)

upDownBB=UnicomBBands[['downBBand','upBBand']]
upDownBB13=upDownBB['2013-01-01':'2013-06-28']


def CalBollRisk(tsPrice,multiplier):
    k=len(multiplier)
    overUp=[]
    belowDown=[]
    BollRisk=[]
    for i in range(k):
        BBands=myDA.bbands_Indi(tsPrice,20,multiplier[i])
        a=0
        b=0
        for j in range(len(BBands)):
            tsPrice=tsPrice[-(len(BBands)):]
            if tsPrice[j]>BBands.upBBand[j]:
                a+=1
            elif tsPrice[j]<BBands.downBBand[j]:
                b+=1
        overUp.append(a)
        belowDown.append(b)
        BollRisk.append(100*(a+b)/len(tsPrice))
    return(BollRisk)

multiplier=[1,1.65,1.96,2,2.58]
price2010=Close['2010-01-04':'2010-12-31']
CalBollRisk(price2010,multiplier)

price2011=Close['2011-01-04':'2011-12-31']
CalBollRisk(price2011,multiplier)

price2012=Close['2012-01-04':'2012-12-31']
CalBollRisk(price2012,multiplier)

price2013=Close['2013-01-04':'2013-12-31']
CalBollRisk(price2013,multiplier)


#strategy
BBands=myDA.bbands_Indi(Close,20,2)

upbreakBB1=upbreak(Close,BBands.upBBand)
downbreakBB1=downbreak(Close,BBands.downBBand)

tradSignal = -upbreakBB1 + downbreakBB1
tradSignal[tradSignal==-0]=0
myBT.SignalQuality(tradSignal,price_Series=Close,lag_trade=2)

upBBSig1=-upbreakBB1.shift(2)
downBBSig1=downbreakBB1.shift(2)

tradSignal1=upBBSig1+downBBSig1
tradSignal1[tradSignal1==0]=0

def perform(tsPrice,tsTradSig):
    ret=tsPrice/tsPrice.shift(1)-1
    tradRet=(ret*tsTradSig).dropna()
    ret=ret[-len(tradRet):].dropna()
    winRate=[len(ret[ret>0])/len(ret[ret!=0]),\
             len(tradRet[tradRet>0])/len(tradRet[tradRet!=0])]
    meanWin=[np.mean(ret[ret>0]),\
             np.mean(tradRet[tradRet>0])]
    meanLoss=[np.mean(ret[ret<0]),\
             np.mean(tradRet[tradRet<0])]
    Performance=pd.DataFrame({'winRate':winRate,'meanWin':meanWin,\
                             'meanLoss':meanLoss})
    Performance.index=['Stock','Trade']
    return(Performance)

Performance1= perform(Close,tradSignal1)
Performance1

upbreakBB2=upbreak(Close,BBands.downBBand)
downbreakBB2=downbreak(Close,BBands.upBBand)
tradSignal = upbreakBB2 - downbreakBB2
myBT.SignalQuality(tradSignal,price_Series=Close,lag_trade=2)

upBBSig2=upbreakBB2.shift(2)
downBBSig2=-downbreakBB2.shift(2)
tradSignal2=upBBSig2+downBBSig2
tradSignal2[tradSignal2==0]=0

Performance2= perform(Close,tradSignal2)
Performance2


#2.
PB = myDA.bbandsDerive_Indi(Close,mode="PB")
PB.plot()
plt.show()

#3.
BW = myDA.bbandsDerive_Indi(Close,mode="BW")
BW.plot()
plt.show()

#4.
b = myDA.bbandsDerive_Indi(Close,mode="BBIB")
b.iloc[:,0:3].plot()
plt.show()
