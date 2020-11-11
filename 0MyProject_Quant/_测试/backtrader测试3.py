# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
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
# ------------------------------------------------------------

# %%
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", [2018, 1, 1, 0, 0, 0], [2020, 1, 1, 0, 0, 0], index_time=True, col_capitalize=False)
data0 = eurusd

# %%
a = None
class TestStrategy(myBT.bt.Strategy):
    # 定义MA均线策略的周期参数变量，默认值是15
    # 增加类一个log打印开关变量： fgPrint，默认自是关闭
    params = (
        ('maperiod', 15),
        ('fgPrint', False),
    )

    def log(self, txt, dt=None, fgPrint=False):
        # 增强型log记录函数，带fgPrint打印开关变量
        if self.params.fgPrint or fgPrint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    # ---只开头执行一次
    def __init__(self):
        # 把策略传到外面，方便查看属性和方法
        global a
        a = self
        print("init", len(self))
        # 检查完成，没有交易中订单（pending order）
        self.order = None
        self.dataclose = self.datas[0].close
        # 增加一个均线指标：indicator
        self.sma = myBT.indi.add_indi_SMA(self.datas[0].close, timeperiod=self.params.maperiod)

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar
    def next(self):
        # 调用log函数，输出BT回溯过程当中，工作节点数据包BAR，对应的close收盘价
        self.log('当前收盘价Close, %.2f' % self.dataclose[0])
        # 这句的意思是，控制未成功的订单只能有一个
        if self.order:
            return
        # 检查当前股票的仓位position
        if not self.position:
            # 如果该股票仓位为0 ，可以进行BUY买入操作，
            # 使用最简单的MA均线策略
            if self.dataclose[0] < self.sma[0]:
                self.log('设置买单 BUY CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
                self.order = self.buy()
        else:
            # 如果该股票仓位>0 ，可以进行SELL卖出操作，
            if self.dataclose[0] > self.sma[0]:
                self.log('SELL CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
                self.order = self.sell()

     # ---策略每笔订单通知函数。已经进入下一个bar，且在next()之前执行
    def notify_order(self, order):
        if myBT.strat.order_status_check(order, feedback=False):
            self.order = None

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        # myBT.strat.trade_status(trade, isclosed=False)
        pass

    # ---策略加载完会触发此语句
    def stop(self):
        self.log('(MA均线周期变量Period= %2d) ，最终资产总值： %.2f' % (self.params.maperiod, self.broker.getvalue()), fgPrint=True)


# %%
# ---基础设置
myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(100000)
myBT.setcommission(0.000)
myBT.addsizer(size=1)
myBT.adddata(data0, fromdate=None, todate=None, filter_mode=None)
# myBT.addanalyzer_all() # 包括 PyFolio 到分析器
# 添加 PyFolio 到分析器
myBT.addanalyzer_pyfolio()

# %%
myBT.addstrategy(TestStrategy)
# 画的价格图 style 默认是"line"，style = "line" 线条图, "candle" 蜡烛图, style="bar"美式k线图, style="ohlc"美式k线图
myBT.run(maxcpus=1, plot=False, backend="tkagg", style="candle", volume=True, voloverlay=True, numfigs=1) # 设置了 plot为False
# myBT.cerebro.plot(iplot=False)

a


#%%
# analy_list = myBT.get_analysis_all()
# analy_list[0]["SQN"]
# analy_list[0]["SharpeRatio"]
# analy_list[0]["DrawDown"]["max"]
# analy_list[0]["TradeAnalyzer"]
# analy_list[0]["PyFolio"]

#%%
# 访问 PyFolio 分析器 (不推荐的)
ana_pyfolio = myBT.get_analysis("PyFolio")
type(ana_pyfolio)
for i in ana_pyfolio.keys():
    print(ana_pyfolio[i])
ana_pyfolio["returns"]
ana_pyfolio["positions"]
ana_pyfolio["transactions"]
ana_pyfolio["gross_lev"]

#%%
# 访问 PyFolio 分析器 (推荐的)，由于 pyfolio 默认使用的是美国东部时间，所以需要转换成utc时区。
returns, positions, transactions, gross_lev = myBT.get_pyfolio_analysis(utc=False)
type(returns)
returns.index

#%%
import pyfolio as pf
# ?不知道为什么不行！！！！！！
pf.create_full_tear_sheet(returns , positions=positions , transactions=transactions , benchmark_rets=returns, round_trips=True)
pf.create_returns_tear_sheet(returns, benchmark_rets=benchmark_rets, live_start_date=live_start_date)

plt.show()
myBT.cerebro.plot(iplot=False)


