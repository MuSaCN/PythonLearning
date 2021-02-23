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

myplt.set_backend("agg")  # 后台输出图片，不占pycharm内存

#%% ###################################
import warnings
warnings.filterwarnings('ignore')

# 获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)

# 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
total_data = eurusd
total_data_train = total_data.loc[:"2014-12-31"]
total_data_test = total_data.loc["2015-01-01":]
price = total_data.Close
price_train = total_data_train.Close
price_test = total_data_test.Close

# 测试不需要把数据集区分训练集、测试集，仅画区间就可以了
train_x0 = pd.Timestamp('2000-01-01 00:00:00')
train_x1 = pd.Timestamp('2014-12-31 00:00:00')

# 获取非共线性的技术指标
indi_name="rsi"
indi_params = [("Close",i) for i in range(5,100+1)]


#%% 分析
holding = 1
k = 100
lag_trade = 1
trade_direct = "All" # "BuyOnly","SellOnly","All"

# ---获取训练集的信号
signaldata_train = myBTV.stra.momentum(price_train, k=k, stra_mode="Continue")
signal_train = signaldata_train[trade_direct]

# ---计算整个样本的信号
signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
signal = signaldata[trade_direct]

#%%
# ---过滤前策略
folder = __mypath__.get_desktop_path() + "\\__动量指标过滤(%s)__"%trade_direct
savefig_initial = folder + "\\过滤前策略.png"
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signal, price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1, savefig=savefig_initial)

#%%
# ---信号过滤，根据信号的利润，运用其他指标来过滤。
import timeit
t0 = timeit.default_timer()
for i_para in indi_params:
    # 指定指标和图片的保存位置
    indicator = myBTV.indi.talib_momentum_indicator(indi_name, total_data[i_para[0]], i_para[1])
    # 每个参数存放的文件名不同
    savefig = folder + "\\%s(%s).png" % (indi_name, i_para[1])
    # 信号利润过滤及测试
    myBTV.rfilter.plot_signal_range_filter_and_quality(signal_train=signal_train, signal_all=signal, indicator=indicator, train_x0=train_x0, train_x1=train_x1, price_DataFrame=total_data, price_Series=price, holding=holding, lag_trade=lag_trade, noRepeatHold=True, indi_name="%s(%s)"%(indi_name, i_para[1]), savefig=savefig)

t1 = timeit.default_timer()
print("\n", 'signal_indicator_filter_and_quality 耗时为：', t1 - t0)







