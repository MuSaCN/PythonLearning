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
data0 = eurusd

class MomentumStrategy(myBT.bt.Strategy):
    # ---设定参数，必须写params，以self.params.Para0索引，可用于优化，内部必须要有逗号
    params = (("Para0", 100),)

    # ---只开头执行一次
    def __init__(self):
        # print(self.datas) # 返回list
        self.barscount = 0
        # open索引
        self.open = self.datas[0].open
        # high索引
        self.high = self.datas[0].high
        # low索引
        self.low = self.datas[0].low
        # close索引
        self.close = self.datas[0].close
        # datetime.date索引，时间必须用()索引，其他用[]
        self.time = self.datas[0].datetime.date

    # ---每一个Bar迭代执行一次。next()执行完就进入下一个bar
    def next(self):
        # ---有多仓就卖，由于到下一bar的开盘才成交，所以回到next()至少经历1期。
        # if self.getposition().size > 0:
        #     self.sell()
        # ---表示有多仓时，且持有了指定根K线，且在 barscount 被赋值时
        # if self.getposition().size > 0 and len(self) >= self.barscount + 5 and self.barscount != 0:
        #     self.sell()
        #     self.barscount = 0
        if self.getposition().size > 0 and self.close[0] < self.close[-self.params.Para0]:
            self.sell()
            self.barscount = 0
        # 仅发出信号，在下一个bar的开盘价成交
        if self.barscount == 0 and self.close[0] > self.close[-self.params.Para0]:
            self.buy()
            self.barscount = len(self)

    # ---策略每笔订单通知函数。已经进入下一个bar，且在next()之前执行
    def notify_order(self, order):
        # lastdirect = myBT.strat.order_buy_sell(order)
        # if myBT.strat.order_status_check(order, False) == True:
        #     self.barscount = len(self) # 注意这里的 len(self) 比 next() 中的大1。
        pass

    # ---策略每笔交易通知函数。已经进入下一个bar，且在notify_order()之后，next()之前执行。
    def notify_trade(self, trade):
        # myBT.strat.trade_status(trade, isclosed=False)
        # myBT.strat.trade_show(trade)
        pass

    # ---策略加载完会触发此语句
    def stop(self):
        print("stop(): ", self.params.Para0, self.broker.get_cash(), self.broker.getvalue())

myBT = MyBackTest.MyClass_BackTestEvent()  # 回测类
myBT.setcash(5000)
myBT.setcommission(0.000)
myBT.addsizer(1)
myBT.adddata(data0, fromdate=None, todate=None)

#%%
myBT.addanalyzer_all()  #(多核时能用，但有的analyzer不支持多核)
myBT.addstrategy(MomentumStrategy)
result = myBT.run(plot=True, backend="pycharm")
result[0].analyzers.CashValue.vals


# 简介的资金曲线
myDefault.set_backend_default("pycharm")
myBT.plot_value(data0, cash_value=None, train_x0=pd.Timestamp('2000-01-01 00:00:00'), train_x1=pd.Timestamp('2014-12-31 00:00:00'))

#%%
# 画转成MT5的收益曲线(backtrader固定仓位才有效，且mt5点差大于50才有效)(用于测试策略信号是否相同)(可做点差压力测试)
pnl_detail = myBT.every_finish_ticket_pnlpnlcom()
myBT.plot_mt5_cumNET(data0, pnl_detail=None, mt5_spread=50, train_x0=pd.Timestamp('2000-01-01 00:00:00'), train_x1=pd.Timestamp('2014-12-31 00:00:00'))


#%%
all_analyzer = myBT.get_analysis_all()
print(len(all_analyzer))
for key in all_analyzer[0]:
    print("--- ",key," :")
    print(all_analyzer[0][key])


#%%
# 多核优化时运行
if __name__ == '__main__':  # 这句必须要有
    # ---
    myBT.addanalyzer_all()  #(多核时能用，但有的analyzer不支持多核)
    myBT.optstrategy(MomentumStrategy, Para0=range(20,150))
    results = myBT.run(maxcpus=None, plot=False)
    # ---优化结果
    paras = [x[0].params.Para0 for x in results]
    cashs = [x[0].analyzers.CashValue.vals[0] for x in results]
    values = [x[0].analyzers.CashValue.vals[1] for x in results]
    df = pd.DataFrame({"para":paras,"cash":cashs,"value":values})
    df = df.sort_values("value",ascending=False)
    print(df)

    # strats = [x[0] for x in results]  # flatten the result
    # for i, strat in enumerate(strats):
    #     para = strat.params.Para0
    #     cash = strat.analyzers.CashValue.vals[0]
    #     value = strat.analyzers.CashValue.vals[1]
    #     print(para, "CashValue: ", cash, value)


    # all_analyzer = myBT.get_analysis_all()
    # print("len(all_analyzer) = ", len(all_analyzer)) # len(all_analyzer) =  5
    #
    # print("\n")
    # for key in all_analyzer[0]:
    #     print("--- ",key," :")
    #     print(all_analyzer[0][key])





