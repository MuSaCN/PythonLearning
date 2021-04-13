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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

###
'''
# "不同时间权重的信号质量分析" 是不同于 "相同时间权重的信号质量分析"，后者每个信号的持有时间是相同的，所以是相同的时间权重。前者是每个信号的持有时间可能是不同的，所以既有入场信号、也有出场信号。
# 根据交易哲学，我认为市场在多空方向上机制不同，所以这里的分析是把做多做空平仓的方式分开来看的。并不是多头必须要平仓才可以持有空仓。显然同一模式下入场信号的多空存在互斥，但是可能有在持有多仓未平仓的情况下，也可以出现做空信号而持有空仓。
# 重叠的信号遵循先平后入。即做多信号与平多信号假如是同一个bar，则平多信号是平上一个做多的，做多信号是平仓后再次入场的，所以此bar依然有持仓。
'''


#%%
### 重要测试：信号去重复 不等于 信号过程化
signal = pd.Series([0,0,1,1,1,1,0,1,1,1,0,0,1,0,1,0])
adj_signal = myBTV.__signal_no_repeat_hold__(signal, holding=2)
#             前[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
#             后[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
# 由于持有2期 = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1] ≠ process

signal_close = signal.shift(2)
signal_close[signal_close == 1] = 2
signal_process = myBTV.__get_signalprocess__(signal,signal_close,1,2)
# signal  [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
# close   [n, n, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 0]
# process [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1] ≠ 持有2期

#%%
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close
# 单独测试不需要把数据集区分训练集、测试集，仅画区间就可以了
train_x0 = pd.Timestamp('2000-01-01 00:00:00')
train_x1 = pd.Timestamp('2014-12-31 00:00:00')

#%% 仅做多分析
holding = 1
k = 100
lag_trade = 1

# ---获取训练集的信号数据
signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
signaldata = signaldata[0:50]
signal_in = signaldata["All"]


#%%
# 信号分析，用于对比，只有holding=1才可以对比
outStrat0, outSignal0 = myBTV.signal_quality_NoRepeatHold(signal_in, price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1)
outStrat1, outSignal1 = myBTV.signal_quality(signal_in, price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1)


#%%
signalbuy = signaldata["BuyOnly"]
signalsell = signaldata["SellOnly"]
signal_closebuy = signalbuy.shift(1)
signal_closebuy[signal_closebuy == 1] = 2
signal_closesell = signalsell.shift(1)
signal_closesell[signal_closesell == -1] = -2

# 信号分析
outStrat, outSignal = myBTV.signal_quality_NoEqualTimeWeight(signal_in, signal_closebuy, signal_closesell, price_DataFrame=eurusd, price_Series=price, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1, savefig=None, ax1=None, ax2=None, show=True, return_Ret=False)










