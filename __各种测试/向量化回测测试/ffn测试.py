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
myDefault = MyDefault.MyClass_Default_Matplotlib() # 画图恢复默认设置类
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
import ffn
# download price data from Yahoo! Finance. By default,
# the Adj. Close will be used.
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)
data = eurusd[["open","high","low","close"]]
prices = data
type(prices)  # pandas.core.frame.DataFrame
# let's compare the relative performance of each stock
# we will rebase here to get a common starting point for both securities
ax = prices.rebase().plot()
plt.show()
# now what do the return distributions look like?
returns = prices.to_returns().dropna()
ax = returns.hist()
plt.show()
# ok now what about some performance metrics?
stats = prices.calc_stats()
stats.display()
# what about the drawdowns?
prices.to_drawdown_series()
myDA.fin.to_drawdown_series(prices=prices).min()
myDA.fin.calc_max_drawdown(prices, datamode="p")
r = prices.to_returns()
p = myDA.fin.r_to_price(r,"r")
p.to_drawdown_series()
ax = stats.prices.to_drawdown_series().plot()
plt.show()

#%%
import ffn
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)
data = eurusd[["open","high","low","close"]]
type(data)
print(data.head())

returns = data.to_log_returns().dropna()
print (returns.head())
ax = returns.hist(figsize=(12, 5))
plt.show()

returns.corr().as_format('.2f')

returns.plot_corr_heatmap()
plt.show()

ax = data.rebase().plot(figsize=(12,5))
plt.show()

perf = data.calc_stats()
perf.plot()
plt.show()
print (perf.display())

# we can also use perf[2] in this case
perf[2].stats

returns.calc_mean_var_weights().as_format('.2%')

groupstats = ffn.GroupStats(data)
groupstats.display()
groupstats.display_lookback_returns()
groupstats.plot()
plt.show()

#%%
import ffn
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)
data = eurusd[["open","high","low","close"]]

prices = data["close"]
prices.plot()
plt.show()

r = data["close"].to_returns().dropna()

myDA.fin.calc_calmar_ratio(returns=r)
myDA.fin.calc_max_drawdown(r,datamode="r")

myDA.fin.calc_information_ratio(r, 0)
myDA.fin.calc_sortino_ratio(r)

# 输入回撤数据
drawdown = myDA.fin.to_drawdown_series(returns=r)
ffn.drawdown_details(drawdown)

myDA.fin.to_ulcer_index(prices=prices)
ffn.to_ulcer_index(prices)
myDA.fin.to_ulcer_index(returns=r)






























