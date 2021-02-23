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
说明：
# 单独测试时，在所有的数据集上，不需要专门的设定训练集、测试集，只需要指定画图时训练集的区间即可。
# 显然，测试集可以在训练集之前，也可以在训练集之后。通过图示能更好的观察收益情况。
# 对于特定的策略参数，通过分析不同的持有期 holding，我们可以分析其“订单可管理性”。我们认为在持有的前3-5期，夏普比要上升，胜率上升或变动不大才行。
'''

#%%
########## 单次测试部分 #################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
symbol = "AUDUSD"
timeframe = "TIMEFRAME_M3"

date_from, date_to = myMT5Pro.get_date_range(timeframe)
data_total = myMT5Pro.getsymboldata(symbol,timeframe,date_from,date_to,index_time=True, col_capitalize=True)
data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)

# 单独测试对全数据进行测试，训练集、测试集仅画区间就可以了
train_x0 = data_train.index[0]
train_x1 = data_train.index[-1]


#%%
# ---仅做多分析
direct = "BuyOnly"
k_range = [k for k in range(127, 127+1)]
holding_range = [holding for holding in range(1, 20+1)]
lag_trade_range = [lag_trade for lag_trade in range(1, 1+1)]

# ---策略结果分析
out_list1 = []
out_list2 = []
# 选择夏普比和胜率来分析，下面的信号质量计算是否重复持仓都要分析。重复持仓主要看胜率。
label1 = "sharpe"
label2 = "winRate"
for k in k_range:
    for holding in holding_range:
        for lag_trade in lag_trade_range:
            # ---获取信号数据
            signaldata_buy = myBTV.stra.momentum(data_total.Close, k=k, stra_mode="Continue")
            # 信号分析: signal_quality_NoRepeatHold / signal_quality
            outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signal=signaldata_buy[direct], price_DataFrame=data_total, holding=holding, lag_trade=lag_trade, plotStrat=False, train_x0=train_x0, train_x1=train_x1, savefig=None)
            out_list1.append(outStrat[direct][label1])
            out_list2.append(outStrat[direct][label2])
# ---
myplt.plot_twoline_twinx(out_list1, out_list2, label1, label2)
myBTV.signal_quality_explain()


#%%
# ---仅做空分析
direct = "SellOnly"
k_range = [k for k in range(275, 275+1)]
holding_range = [holding for holding in range(1, 20+1)]
lag_trade_range = [lag_trade for lag_trade in range(1, 1+1)]

# ---策略结果分析
out_list1 = []
out_list2 = []
# 选择夏普比和胜率来分析，下面的信号质量计算是否重复持仓都要分析。重复持仓主要看胜率。
label1 = "sharpe"
label2 = "winRate"
for k in k_range:
    for holding in holding_range:
        for lag_trade in lag_trade_range:
            # ---获取信号数据
            signaldata_sell = myBTV.stra.momentum(data_total.Close, k=k, stra_mode="Continue")
            # 信号分析: signal_quality_NoRepeatHold / signal_quality
            outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_sell[direct], price_DataFrame=data_total, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False, train_x0=train_x0, train_x1=train_x1, savefig=None)
            out_list1.append(outStrat[direct][label1])
            out_list2.append(outStrat[direct][label2])
# ---
myplt.plot_twoline_twinx(out_list1, out_list2, label1, label2)
myBTV.signal_quality_explain()


# %%
# ---多空都做分析，相同参数
direct = "All"
k_range = [k for k in range(139, 139 + 1)]
holding_range = [holding for holding in range(1, 1 + 1)]
lag_trade_range = [lag_trade for lag_trade in range(1, 1 + 1)]

# ---策略结果分析
out_list1 = []
out_list2 = []
# 选择夏普比和胜率来分析，下面的信号质量计算是否重复持仓都要分析。重复持仓主要看胜率。
label1 = "sharpe"
label2 = "winRate"
for k in k_range:
    for holding in holding_range:
        for lag_trade in lag_trade_range:
            # ---获取信号数据
            signaldata_all = myBTV.stra.momentum(data_total.Close, k=k, stra_mode="Continue")
            # 信号分析，不重复持仓
            outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_all[direct], price_DataFrame=data_total, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1, savefig=None)
            out_list1.append(outStrat[direct][label1])
            out_list2.append(outStrat[direct][label2])
# ---
myplt.plot_twoline_twinx(out_list1, out_list2, label1, label2)
myBTV.signal_quality_explain()


#%%
# ---多空不同参数合并分析
k_buy = 139
k_sell = 139
holding = 1
lag_trade = 1

signaldata_buy = myBTV.stra.momentum(data_total.Close, k=k_buy, stra_mode="Continue")
signaldata_sell = myBTV.stra.momentum(data_total.Close, k=k_sell, stra_mode="Continue")
signal_add = signaldata_buy["BuyOnly"] + signaldata_sell["SellOnly"]

# 信号分析，不重复持仓
outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signal_add, price_DataFrame=data_total, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=True, train_x0=train_x0, train_x1=train_x1, savefig=None)

myBTV.signal_quality_explain()


