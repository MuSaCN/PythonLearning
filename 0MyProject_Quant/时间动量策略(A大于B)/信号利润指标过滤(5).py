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

'''
说明：从累计利润角度进行过滤。所用的思想为“求积分(累积和)来进行噪音过滤”。
# 为了让策略有更好的表现，我们希望过滤掉一些不好的信号。那么如何过滤信号才是科学的呢？
# 比如我们用rsi(60)来过滤，当然也可以用其他的比如rsi(50)。具体哪个可以根据测试。
# 一个指定参数的策略，每次信号时，我们都可以计算出相应的 rsi(60)。显然每次信号都有一个利润，那么就可以得到两个序列，一个为 rsi(60) 的序列，一个为对应的利润的序列。
# 得到两个序列后，把 rsi(60) 按从小到大排列，利润也会相应的排列，然后得到利润的累计和序列。
# 这样 rsi(60)序列 与 利润累积和序列 就可以画出两者的关系。可以分析出 rsi(60)值 的哪些区间对于累计利润是正的贡献、哪些区间是负的贡献。
'''

#%% ###################################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
# 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
eurusd_train = eurusd.loc[:"2014-12-31"]
eurusd_test = eurusd.loc["2015-01-01":]
price = eurusd.Close
price_train = eurusd_train.Close
price_test = eurusd_test.Close

# 获取非共线性的技术指标
import talib
timeperiod = [5, 6+1] # 指标参数的范围
rsi = [talib.RSI(price,timeperiod=i) for i in range(timeperiod[0], timeperiod[1])]


#%% 仅做多分析
holding = 1
k = 100
lag_trade = 1

# ---仅做多分析，获取训练集的信号数据
signaldata = myBTV.stra.momentum(price_train, k=k, stra_mode="Continue")
signal=signaldata["buysignal"]

# ---信号过滤，根据信号的利润，运用其他指标来过滤。
for i in range(timeperiod[0], timeperiod[1]):
    indicator = rsi[i-timeperiod[0]]
    savefig = __mypath__.get_desktop_path() + "\\__动量指标过滤(Buy)__\\rsi(%s).png"%i
    myBTV.rfilter.signal_range_filter(signal,indicator=indicator,price_DataFrame=eurusd,holding=holding,lag_trade=lag_trade,noRepeatHold=True,indi_name="rsi(%s)"%i,savefig = savefig)


#%% 仅做空分析
holding = 1
k = 100
lag_trade = 1

# ---仅做空分析，获取训练集的信号数据
signaldata = myBTV.stra.momentum(price_train, k=k, stra_mode="Continue")
signal=signaldata["sellsignal"]


# ---信号过滤，根据信号的利润，运用其他指标来过滤。
for i in range(timeperiod[0], timeperiod[1]):
    indicator = rsi[i-timeperiod[0]]
    savefig = __mypath__.get_desktop_path() + "\\__动量指标过滤(Sell)__\\rsi(%s).png"%i
    myBTV.rfilter.signal_range_filter(signal,indicator=indicator,price_DataFrame=eurusd,holding=holding,lag_trade=lag_trade,noRepeatHold=True,indi_name="rsi(%s)"%i,savefig = savefig)



#%% 做多空分析
holding = 1
k = 100
lag_trade = 1

# ---做多空分析，获取训练集的信号数据
signaldata = myBTV.stra.momentum(price_train, k=k, stra_mode="Continue")
signal=signaldata["All"]

# ---信号过滤，根据信号的利润，运用其他指标来过滤。
for i in range(timeperiod[0], timeperiod[1]):
    indicator = rsi[i-timeperiod[0]]
    savefig = __mypath__.get_desktop_path() + "\\__动量指标过滤(All)__\\rsi(%s).png"%i
    myBTV.rfilter.signal_range_filter(signal,indicator=indicator,price_DataFrame=eurusd,holding=holding,lag_trade=lag_trade,noRepeatHold=True,indi_name="rsi(%s)"%i,savefig = savefig)




