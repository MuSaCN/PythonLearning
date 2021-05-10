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

'''
# 信号分类分析，由于每个类别的信号都要分析，所以不用过滤一词。
# 前面的范围过滤、方向过滤都可以泛化为分类分析，只不过范围过滤、方向过滤是更具体的概念。
# 我们引入的信号分析分类，是指一段市场的信号，根据某种方式，被划分了几个类别(分组)。有可能有的类别信号质量会很好，有的不好，本质上也是类别过滤的思想。
# 对于市场的研究，我们不能受制于市场理论，我们不能想当然的认为市场在某些类别中更好。比如动量策略在均线以上做多，均线以下不做多，我们为什么不能研究下动量策略在均线以下做多的情况呢？
'''

#%%
import warnings
warnings.filterwarnings('ignore')
para_name = ["k", "holding", "lag_trade"]

#%%
symbol = "EURUSD"
timeframe = "TIMEFRAME_D1"
direct = "BuyOnly"
[k, holding, lag_trade] = [101, 1, 1]
indi_name = "DEMA"
indi_para = [14,0,"PRICE_CLOSE"]
para_str = myBTV.string_strat_para(para_name, [k, holding, lag_trade])

# ---获取数据
date_from, date_to = myMT5Pro.get_date_range(timeframe, to_Timestamp=True)
data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
# 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
data_train, data_test = myMT5Pro.get_train_test(data=data_total, train_scale=0.8)
# 测试不需要把数据集区分训练集、测试集，仅画区间就可以了
train_x0 = data_train.index[0]
train_x1 = data_train.index[-1]
# # 把训练集的时间进行左右扩展
bound_left, bound_right = myMT5Pro.extend_train_time(train_t0=train_x0, train_t1=train_x1, extend_scale=0.3)
# 再次重新加载下全部的数据
data_total = myMT5Pro.getsymboldata(symbol, timeframe, bound_left, bound_right, index_time=True, col_capitalize=True)

#%%
# ---获取训练集和全集的信号
# 获取训练集的信号 ******(修改这里)******
signaldata_train = myBTV.stra.momentum(data_train.Close, k=k, stra_mode="Reverse")
signal_train = signaldata_train[direct]
signaldata_all = myBTV.stra.momentum(data_total.Close, k=k, stra_mode="Reverse")
signal_all = signaldata_all[direct]

# ---(核心，在库中添加)获取指标 ******(修改这里)******
indicator = myBTV.indiMT5.get_indicator_firstbuffer(data_total, indi_name, *indi_para)

# ---信号分类(分组)
df = pd.concat((signal_train, signal_all, indicator, data_total.Close), axis=1)
df.columns = ["signal_train", "signal_all", "indicator", "price"]
signal1_train = df["signal_train"][df["price"] > df["indicator"]].dropna()
signal1_train.name = signal_train.name
signal2_train = df["signal_train"][df["price"] < df["indicator"]].dropna()
signal2_train.name = signal_train.name
signal1_all =  df["signal_all"][df["price"] > df["indicator"]].dropna()
signal1_all.name = signal_train.name
signal2_all =  df["signal_all"][df["price"] < df["indicator"]].dropna()
signal2_all.name = signal_train.name


#%%
signal_train_list = [signal1_train, signal2_train]
signal_all_list = [signal1_all, signal2_all]
myBTV.sigclassify.plot_signal_classify(signal_train_list=signal_train_list,signal_all_list=signal_all_list, accum=True, figsize=[1920, 1080], noRepeatHold=True,price_DataFrame=data_total,price_Series=data_total.Close,holding=holding,lag_trade=lag_trade,train_x0=train_x0,train_x1=train_x1,para_str=para_str,show=True)

