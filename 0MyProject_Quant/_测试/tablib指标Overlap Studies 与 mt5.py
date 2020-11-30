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
### Overlap Studies
# bbands 与 MT5 结果相同
upperband, middleband, lowerband = myBTV.indi.get_trend_indicator(eurusd,"bbands",["close"],timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
upperband.plot(); middleband.plot(); lowerband.plot(); plt.show()

# dema 与 MT5 结果相同
real = myBTV.indi.get_trend_indicator(eurusd,"dema",["close"],timeperiod=30)
real.plot(); plt.show()

# ema
real = myBTV.indi.get_trend_indicator(eurusd,"ema",["close"],timeperiod=30)
real.plot(); plt.show()

# ht_trendline
real = myBTV.indi.get_trend_indicator(eurusd,"ht_trendline",["close"])
real.plot(); plt.show()

# kama
real = myBTV.indi.get_trend_indicator(eurusd,"kama",["close"],timeperiod=30)
real.plot(); plt.show()

# ma
real = myBTV.indi.get_trend_indicator(eurusd,"ma",["close"],timeperiod=30, matype=0)
real.plot(); plt.show()

# mama ****************
mama, fama = myBTV.indi.get_trend_indicator(eurusd,"mama",["close"],fastlimit=0, slowlimit=0)
mama.plot(); fama.plot(); plt.show()

# mavp
real = myBTV.indi.get_trend_indicator(eurusd,"mavp",["close","periods"],minperiod=2, maxperiod=30, matype=0)
real.plot(); plt.show()

# midpoint
real = myBTV.indi.get_trend_indicator(eurusd,"midpoint",["close"],timeperiod=14)
real.plot(); plt.show()

# midprice
real = myBTV.indi.get_trend_indicator(eurusd,"midprice",["high", "low"],timeperiod=14)
real.plot(); plt.show()

# sar 与 MT5 相同
real = myBTV.indi.get_trend_indicator(eurusd,"sar",["high", "low"],acceleration=0, maximum=0)
real.plot(); plt.show()

# sarext
real = myBTV.indi.get_trend_indicator(eurusd,"sarext",["high", "low"],startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
real.plot(); plt.show()

# sma
real = myBTV.indi.get_trend_indicator(eurusd,"sma",["close"],timeperiod=30)
real.plot(); plt.show()

# t3
real = myBTV.indi.get_trend_indicator(eurusd,"t3",["close"],timeperiod=5, vfactor=0)
real.plot(); plt.show()

# tema 与 MT5 结果相同
real = myBTV.indi.get_trend_indicator(eurusd,"tema",["close"],timeperiod=30)
real.plot(); plt.show()

# trima
real = myBTV.indi.get_trend_indicator(eurusd,"trima",["close"],timeperiod=30)
real.plot(); plt.show()

# wma
real = myBTV.indi.get_trend_indicator(eurusd,"wma",["close"],timeperiod=30)
real.plot(); plt.show()





