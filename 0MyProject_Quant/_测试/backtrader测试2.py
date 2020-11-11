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
#------------------------------------------------------------

#%%
# ---获取数据
import warnings
warnings.filterwarnings('ignore')
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2019,7,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=False)
data0 = eurusd

#%%
class TestStrategy(myBT.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (('Para0', 250),)

    # ---只开头执行一次
    def __init__(self):
        print("init", len(self))
        # 检查完成，没有交易中订单（pending order）
        self.order = None

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar
    def next(self):

        # 这句的意思是，控制未成功的订单只能有一个。
        if self.order:
            return

        # 检查订单执行情况，默认每次只能执行一张order订单交易，可以修改相关参数，进行调整
        if len(self) == 15 or len(self) == 25:
            print("next: len(self) = ",len(self), "; self.order = ", self.order)
        if len(self) == 10:
            print("next: len(self) = ", len(self))
            self.order = self.buy() # 这句的意思是记录下订单的状态，包括很多信息
            print("next: self.order = ", self.order) # next: self.order =  Ref: 1 ...
        if len(self) == 20:
            print("next: len(self) = ", len(self))
            self.order = self.sell() # 这句的意思是记录下订单的状态，包括很多信息
            print("next: self.order = ", self.order) # next: self.order =  Ref: 2 ...


    # ---策略每笔订单通知函数。已经进入下一个bar，且在next()之前执行
    def notify_order(self, order):
        if myBT.strat.order_status_check(order,feedback=False):
            self.order = None

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        # myBT.strat.trade_status(trade, isclosed=False)
        pass

    # ---策略加载完会触发此语句
    def stop(self):
        print("stop(): ", self.params.Para0 , self.broker.getvalue(), self.broker.get_cash())

#%%
# ---基础设置
myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(100000)
myBT.setcommission(0.000)
# myBT.adddata(data0, fromdate=None, todate=None, filter_mode=None)
# 数据过滤，转成平均K线
myBT.adddata(data0, fromdate=None, todate=None, filter_mode="HeikinAshi", name="EURUSD")
# myBT.addanalyzer_all()
myBT.addanalyzer(myBT.bt.analyzers.SharpeRatio, _name = 'SharpeRatio')
#不同时间周期 bt.TimeFrame.Days, bt.TimeFrame.Weeks, bt.TimeFrame.Months, bt.TimeFrame.Years)
myBT.addanalyzer(myBT.bt.analyzers.TimeReturn, _name='TimeReturn')
myBT.addanalyzer(myBT.bt.analyzers.TimeReturn, _name='TimeReturn_Days', timeframe=myBT.bt.TimeFrame.Days)
myBT.addanalyzer(myBT.bt.analyzers.TimeReturn, _name='TimeReturn_Weeks', timeframe=myBT.bt.TimeFrame.Weeks)
myBT.addanalyzer(myBT.bt.analyzers.TimeReturn, _name='TimeReturn_Months', timeframe=myBT.bt.TimeFrame.Months)
myBT.addanalyzer(myBT.bt.analyzers.TimeReturn, _name='TimeReturn_Years', timeframe=myBT.bt.TimeFrame.Years)


#%%
myBT.addstrategy(TestStrategy)
# 画的价格图 style 默认是"line"，style = "line" 线条图, "candle" 蜡烛图, style="bar"美式k线图, style="ohlc"美式k线图
myBT.run(maxcpus=1 ,plot = True, backend="tkagg", style="candle", volume=True, voloverlay=True, numfigs = 1)
# 15 99999.94443999976 99998.82211999976
myBT.every_cash_value()

# analy_list = myBT.get_analysis_all()
# analy_list[0]["SQN"]
myBT.get_analysis("SharpeRatio")
myBT.get_analysis("TimeReturn")
myBT.get_analysis("TimeReturn_Days")
myBT.get_analysis("TimeReturn_Weeks")
myBT.get_analysis("TimeReturn_Months")
myBT.get_analysis("TimeReturn_Years")


