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
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\test")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure(AddFigure=False)  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro(AddFigure=False)  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  # 事件驱动型回测
myBTV = MyPackage.MyClass_BackTestVector.MyClass_BackTestVector()  # 向量化回测
myWebQD = MyPackage.MyClass_WebCrawler.MyClass_WebQuotesDownload()  #金融行情下载类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities["2015"]

# ---基础设置
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myBT.ValueCash(2000)
myBT.addsizer(10)
myBT.setcommission(0.001)
myBT.AddBarsData(data0,fromdate=None,todate=None)

# ---策略开始
@myBT.OnInit
def __init__():
    print("init检测无仓位 = ", not myBT.position())
    myBT.Indi_MovingAverageSimple(0,5,None,True)

# ---策略递归，next()执行完就进入下一个bar
order = []; barscount = [0]
@myBT.OnNext
def next():
    print("next()", myBT.bars_executed, myBT.datetime(0))
    if not myBT.position():
        if myBT.close(0) > myBT.SMA[0]:
            order.append(myBT.buy())
            print("next()发出buy信号", myBT.bars_executed, myBT.datetime(0))
    else:
        if myBT.bars_executed >= barscount[0]+5:
            order.append(myBT.sell())
            print("next()发出sell信号", myBT.bars_executed)

# ---策略订单通知，已经进入下一个bar，且在next()之前执行
@myBT.OnNotify_Order
def notify_order():
    print("notify_order", myBT.bars_executed)
    if myBT.OrderStatusCheck(myBT.order_noti,feedback=True) == False:
        return
    else:
        # 必须记录在这里，因为执行在这里
        barscount[0] = myBT.bars_executed

# ---策略交易通知，已经进入下一个bar，且在notify_order()之后，next()之前执行
@myBT.OnNotify_Trade
def notify_trade():
    print("notify_trade", myBT.bars_executed)
    myBT.TradeStatus(myBT.trade_noti,isclosed=True)

myBT.addstrategy()
# ---运行
myBT.run(plot = True)




import backtrader as bt
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)
    def next(self):
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.sell()
    def stop(self):
        print(self.params.maperiod, self.broker.getvalue())

cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)

# cerebro.optstrategy(TestStrategy,maperiod=range(5, 10)) # *************

Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data0 = CJSecurities["2015"]
if "openinterest" not in data0.columns:
    data0['openinterest'] = 0
data = bt.feeds.PandasData(dataname=data0)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.addsizer(bt.sizers.FixedSize, stake=10)
cerebro.broker.setcommission(commission=0.0)
cerebro.run()
cerebro.plot(iplot=False)


