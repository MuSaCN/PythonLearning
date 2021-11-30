# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
myDefault = MyDefault.MyClass_Default_Matplotlib()  # 画图恢复默认设置类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

#%%
import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=False)
mypd.__init__(maxrows=None, maxcolumns=None)
data0 = eurusd[:20]


class ABCStrategy(myBT.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (("Para0", 5),)

    # ---只开头执行一次
    def __init__(self):
        print("__init__ :", len(self)) # __init__ : 0
        self.barscount = 0
        self.sma = myBT.indi.add_indi_SMA(self.datas[0].close, timeperiod=5)
        # open索引
        self.open = self.datas[0].open
        # high索引
        self.high = self.datas[0].high
        # low索引
        self.low = self.datas[0].low
        # close索引
        self.close = self.datas[0].close
        # datetime.date索引，用()索引
        self.time = self.datas[0].datetime.date

    # ---策略激活的时候被调用，类似__init__，此时len(self) = 0.
    def start(self):
        print("start , ",len(self)) # start ,  0

    # ---技术指标预载时自动调用，假如需要n个的数据才能产生指标，它执行n-1次.
    def prenext(self):
        # prenext,  1
        # prenext,  2
        # prenext,  3
        # prenext,  4
        print("prenext, ", len(self))

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar，执行完 len(self) 数值+1。交易单发出信号，在下一个bar成交。
    def next(self):
        self.getposition().size # 获取仓位大小，>0多仓，<0空仓
        if not self.position:
            if len(self) == 7:
                self.sell()
                print("next: sell ",self.time(0), self.close[0]) # next: sell  2000-01-11 1.0336
        else:
            if len(self) >= self.barscount: # 想当初持有1期
                self.buy()
                print("next: buy ",self.time(0), self.close[0]) # next: buy  2000-01-12 1.0308

    # ---策略每笔订单通知函数(订单执行前会多次执行)。已经进入下一个bar，且在下一个next()之前执行。交易单执行成功会在 next() 下一个 bar 的开盘价执行。
    def notify_order(self, order):
        ""
        '''
        notify_order: 订单通知前：len= 8 2000-01-12
        Buy/Sell order submitted to broker / accepted by broker - waiting...
        notify_order: 订单通知前：len= 8 2000-01-12
        Buy/Sell order submitted to broker / accepted by broker - waiting...
        notify_order: 订单通知前：len= 8 2000-01-12
        SELL 执行成功，Price: 1.0337, Cost: -1.0337, Commission 0.0000
        '''
        '''
        notify_order: 订单通知前：len= 9 2000-01-13
        Buy/Sell order submitted to broker / accepted by broker - waiting...
        notify_order: 订单通知前：len= 9 2000-01-13
        Buy/Sell order submitted to broker / accepted by broker - waiting...
        notify_order: 订单通知前：len= 9 2000-01-13
        BUY 执行成功，Price: 1.0309, Cost: -1.0337, Commission 0.0000
        '''
        print("notify_order: 订单通知前：len=", len(self), self.time(0))
        if myBT.strat.order_status_check(order, True) == True:
            self.barscount = len(self)
            # notify_order: 订单通知后：len= 8 2000-01-12
            # notify_order: 订单通知后：len= 9 2000-01-13
            print("notify_order: 订单通知后：len=", len(self), self.time(0))

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        # notify_trade: 交易通知前：len= 8 2000-01-12
        # notify_trade: 交易通知前：len= 9 2000-01-13
        print("notify_trade: 交易通知前：len=", len(self), self.time(0))
        # 每单交易利润, GROSS = 0.00, NET = 0.00
        # 每单交易利润, GROSS = 0.0028, NET = 0.0028
        myBT.strat.trade_status(trade, isclosed=False)
        '''
        ------ trade属性展示 ------
        # ref: 唯一id = 10821
        # size(int): trade的当前头寸 = -1
        # price(float): trade资产的当前价格 = 1.0337
        # value(float): trade的当前价值 = -1.0337
        # commission(float): trade的累计手续费 = 0.0
        # pnl(float): trade的当前pnl = 0.0
        # pnlcomm(float): trade的当前pnl减去手续费 = 0.0
        # isclosed(bool): 当前时刻trade头寸是否归零 = False
        # isopen(bool): 新的交易更新了trade = True
        # justopened(bool): 新开头寸 = True
        # dtopen (float): trade open的datetime = 730131.0
        # dtclose (float): trade close的datetime = 0.0

        ------ trade属性展示 ------
        # ref: 唯一id = 10821
        # size(int): trade的当前头寸 = 0
        # price(float): trade资产的当前价格 = 1.0337
        # value(float): trade的当前价值 = 0.0
        # commission(float): trade的累计手续费 = 0.0
        # pnl(float): trade的当前pnl = 0.0028000000000001357
        # pnlcomm(float): trade的当前pnl减去手续费 = 0.0028000000000001357
        # isclosed(bool): 当前时刻trade头寸是否归零 = True
        # isopen(bool): 新的交易更新了trade = False
        # justopened(bool): 新开头寸 = False
        # dtopen (float): trade open的datetime = 730131.0
        # dtclose (float): trade close的datetime = 730132.0
        '''
        myBT.strat.trade_show(trade)
        # notify_trade: 交易通知后：len= 8 2000-01-12
        # notify_trade: 交易通知后：len= 9 2000-01-13
        print("notify_trade: 交易通知后：len=", len(self), self.time(0))

    # ---策略加载完会触发此语句
    def stop(self):
        # stop():  5 99999.9924 99999.9924
        print("stop(): ", self.params.Para0 , self.broker.getvalue(), self.broker.get_cash())
        print("stop(): ", len(self), self.time(0)) # stop():  20 2000-01-28


myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(100000)
myBT.setcommission(0.000)
myBT.addsizer(1)
myBT.adddata(data0, fromdate=None, todate=None)

#%%
myBT.addanalyzer_all()  #(多核时能用，但有的analyzer不支持多核)
myBT.addstrategy(ABCStrategy)
myBT.run(maxcpus=1, plot=True, backend="pycharm")

myDefault.set_backend_default("pycharm")
myBT.plot_value(data0)

all_analyzer = myBT.get_analysis_all()
print(len(all_analyzer))
for key in all_analyzer[0]:
    print("--- ",key," :")
    print(all_analyzer[0][key])


