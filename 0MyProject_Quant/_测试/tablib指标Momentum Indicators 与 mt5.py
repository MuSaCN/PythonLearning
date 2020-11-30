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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator() # MT5指标Python版
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

#%%
import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[1990,1,1,0,0,0],[2020,11,24,0,0,0],index_time=True, col_capitalize=False)

#%%
### Momentum Indicators
# adx 与 MT5 不同
real =  myBTV.indi.get_oscillator_indicator(eurusd, "adx",["high","low","close"],timeperiod=14)
real.plot();plt.show()

# adxr
real =  myBTV.indi.get_oscillator_indicator(eurusd, "adxr",["high","low","close"],timeperiod=14)
real.plot();plt.show()

# apo
real = myBTV.indi.get_oscillator_indicator(eurusd,"apo",["close"],fastperiod=12, slowperiod=26, matype=0)
real.plot();plt.show()

# aroon
aroondown, aroonup = myBTV.indi.get_oscillator_indicator(eurusd,"aroon",["high","low"],timeperiod=14)
aroondown.plot();plt.show()
aroonup.plot();plt.show()

# aroonosc
real = myBTV.indi.get_oscillator_indicator(eurusd,"aroonosc",["high","low"],timeperiod=14)
real.plot();plt.show()

# bop
real = myBTV.indi.get_oscillator_indicator(eurusd,"bop",["open", "high", "low", "close"])
real.plot();plt.show()

# cci 与 MT5中 typical price 相同
real = myBTV.indi.get_oscillator_indicator(eurusd,"cci",["high", "low", "close"],timeperiod=14)
real.plot();plt.show()

# cmo
real = myBTV.indi.get_oscillator_indicator(eurusd,"cmo",["close"],timeperiod=14)
real.plot();plt.show()

# dx
real = myBTV.indi.get_oscillator_indicator(eurusd,"dx",["high", "low", "close"],timeperiod=14)
real.plot();plt.show()

# macd 中 macd 与 MT5相同，macdsignal与MT5不同
macd, macdsignal, macdhist = myBTV.indi.get_oscillator_indicator(eurusd,"macd",["close"],fastperiod=12, slowperiod=26, signalperiod=9)
macd.plot();macdsignal.plot();macdhist.plot();plt.show()

# macdext
macd, macdsignal, macdhist = myBTV.indi.get_oscillator_indicator(eurusd,"macdext",["close"],fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
macd.plot();macdsignal.plot();macdhist.plot();plt.show()

# macdfix
macd, macdsignal, macdhist = myBTV.indi.get_oscillator_indicator(eurusd,"macdfix",["close"], signalperiod=9)
macd.plot();macdsignal.plot();macdhist.plot();plt.show()

# mfi
real = myBTV.indi.get_oscillator_indicator(eurusd,"mfi",["high", "low", "close", "volume"],timeperiod=14)
real.plot();plt.show()

# minus_di
real = myBTV.indi.get_oscillator_indicator(eurusd,"minus_di",["high", "low", "close"],timeperiod=14)
real.plot();plt.show()

# minus_dm
real = myBTV.indi.get_oscillator_indicator(eurusd,"minus_dm",["high", "low"],timeperiod=14)
real.plot();plt.show()

# mom
real = myBTV.indi.get_oscillator_indicator(eurusd,"mom",["close"],timeperiod=10)
real.plot();plt.show()

# plus_di
real = myBTV.indi.get_oscillator_indicator(eurusd,"plus_di",["high", "low", "close"],timeperiod=14)
real.plot();plt.show()

# plus_dm
real = myBTV.indi.get_oscillator_indicator(eurusd,"plus_dm",["high", "low"],timeperiod=14)
real.plot();plt.show()

# ppo
real = myBTV.indi.get_oscillator_indicator(eurusd,"ppo",["close"],fastperiod=12, slowperiod=26, matype=0)
real.plot();plt.show()

# roc 与 MT5 不同
real = myBTV.indi.get_oscillator_indicator(eurusd, "roc", ["close"], timeperiod=10)
real.plot();plt.show()

# rocp
real = myBTV.indi.get_oscillator_indicator(eurusd, "rocp", ["close"], timeperiod=10)
real.plot();plt.show()

# rocr
real = myBTV.indi.get_oscillator_indicator(eurusd, "rocr", ["close"], timeperiod=10)
real.plot();plt.show()

# rocr100
real = myBTV.indi.get_oscillator_indicator(eurusd, "rocr100", ["close"], timeperiod=10)
real.plot();plt.show()

# rsi 与 MT5 相同
real = myBTV.indi.get_oscillator_indicator(eurusd, "rsi", ["close"], timeperiod=10)
real.plot();plt.show()

# stoch 与 MT5 不相同
slowk, slowd = myBTV.indi.get_oscillator_indicator(eurusd, "stoch", ["high", "low", "close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
slowk.plot();slowd.plot();plt.show()

# stochf
fastk, fastd = myBTV.indi.get_oscillator_indicator(eurusd, "stochf", ["high", "low", "close"], fastk_period=5, fastd_period=3, fastd_matype=0)
fastk.plot();fastd.plot();plt.show()

# stochrsi
fastk, fastd = myBTV.indi.get_oscillator_indicator(eurusd, "stochrsi", ["close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
fastk.plot();fastd.plot();plt.show()

# trix
real = myBTV.indi.get_oscillator_indicator(eurusd, "trix", ["close"], timeperiod=30)
real.plot();plt.show()

# ultosc
real = myBTV.indi.get_oscillator_indicator(eurusd, "ultosc", ["high", "low", "close"],  timeperiod1=7, timeperiod2=14, timeperiod3=28)
real.plot();plt.show()

# willr 与 MT5 相同
real = myBTV.indi.get_oscillator_indicator(eurusd, "willr", ["high", "low", "close"], timeperiod=14)
real.plot();plt.show()



