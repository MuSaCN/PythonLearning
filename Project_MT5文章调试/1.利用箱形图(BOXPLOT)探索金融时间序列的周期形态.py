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
myini = MyFile.MyClass_INI()  # ini文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
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
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

'''
根据MT5文章：利用箱形图（BOXPLOT）探索金融时间序列的季节性形态
https://www.mql5.com/zh/articles/7038
'''
#%%
myMT5Pro.__init__(connect=True)

rates_original = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", [2010,1,1], [2020,1,1], index_time=False)
# leave only 'time' and 'close' columns
rates = rates_original[["time","close"]]

# get percent change (price returns)
returns = pd.DataFrame(rates['close'].pct_change(1))
returns = returns.set_index(rates['time'])
returns = returns[1:]
returns.head(5)

# 把数据按照 年-月 group
indexyear = returns.index.year.rename('year')
indexmonth = returns.index.month.rename('month')
Monthly_Returns = returns.groupby([indexyear, indexmonth]).mean()
# 按月来画箱型图，每个箱型图是统计每年的指定月数据
Monthly_Returns.boxplot(column='close', by='month', figsize=(15, 8))
plt.show()
'''
我们可以看到，第五个月（五月份）的增幅中位数偏移到零轴下方，且其异常值显见高于零轴。 通常，从十年的统计数据里可以看出，五月份的市场相对于三月份有所下跌。 只有一个年份，五月份市场上涨。 这是一个有趣的思路，很符合交易者的格言“在五月份卖掉，并离开！”。
我们看一下五月份之后的六月份。 相对于五月份，六月份市场几乎总是（排除一年以外）在增长，这种情况每年都在重复。 六月份的波动范围很小，没有异常值（与五月份不同），这表明良好的季节性稳定。
请注意第 11 个月（十一月份）。 在此期间市场下跌的概率很高。 之后，在十二月份，市场通常会再度上行。 一月份（第一个月）的波动性很高，且相对于十二月份有所下跌。
所获得的数据可为交易决策提供很有用的基础条件概览。 而且，概率可以集成到交易系统当中。 例如，可以在某些月份执行更多的买卖操作。
'''

# 我们利用相同的 10 年度观察一周中每个单独交易日的价格增幅分布：
indexweek = returns.index.week.rename('week')
indexday = returns.index.dayofweek.rename('day')
Daily_Returns = returns.groupby([indexweek, indexday]).mean()
# 按 星期几 来统计
Daily_Returns.boxplot(column='close', by='day', figsize=(15, 8))
plt.show()
'''
此处零对应于星期一，四则对应于星期五。 根据价格范围，按日波动率几乎保持不变。 然而不能据此得出结论，即在一周中的某个特定日期交易更为密集。 平均来说，市场在星期一和星期五时更倾向于下行非上行。
'''
# 也许在一些单独的月份中，按日分布的样子会有所不同。
# leave only one month "returns.index[~returns.index.month.isin([1])"
for i in range(1,13): # i=1
    returns_solo = returns.drop(returns.index[~returns.index.month.isin([i])])
    indexweek_solo = returns_solo.index.week.rename('week')
    indexday_solo = returns_solo.index.dayofweek.rename('day')
    Daily_Returns_solo = returns_solo.groupby([indexweek_solo, indexday_solo]).mean()
    # 按 星期几 来统计
    Daily_Returns_solo.boxplot(column='close', by='day', figsize=(15, 8))
    plt.suptitle("Boxplot grouped by day month_%s"%i)
    plt.show()

#%%
### 分析日内形态
# 在创建交易系统时，通常要考虑日内的分布，例如，除了日线和月线的分布，还要用到小时线数据。 这轻易就可做到。
rates_original = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_M15", [2010,1,1], [2019, 11, 25], index_time=False)
# leave only 'time' and 'close' columns
rates = rates_original[["time","close"]]
# get percent change (price returns)
returns = pd.DataFrame(rates['close'].pct_change(1))
returns = returns.set_index(rates['time'])
returns = returns[1:]

# 将数据按日和小时分组
indexday = returns.index.day.rename('day')
indexhour = returns.index.hour.rename('hour')
Hourly_Returns = returns.groupby([indexday, indexhour]).median()
Hourly_Returns.boxplot(column='close', by='hour', figsize=(10, 5))
plt.show()

#%%
### 按均线（MA）搜索去趋势化形态
# 正确检测趋势分量非常棘手。 有时，时间序列可能会太平滑。 在这种情况下，交易信号极少。 如果缩短平滑周期，那么高频成交可能无法负担点差和佣金。 我们编辑代码，以便利用移动平均线进行去趋势化：
rates_original = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_M15", [2010,1,1], [2019, 11, 25], index_time=False)
# leave only 'time' and 'close' columns
rates = rates_original[["time","close"]]
rates = rates.set_index('time')
# set the moving average period
window = 25
# detrend tome series by MA
ratesM = rates.rolling(window).mean()
plt.figure(figsize=(10, 5))
plt.plot(rates)
plt.plot(ratesM)
plt.show()
# 我们得到了小时收盘价与 15 分钟移动平均线之间的平均偏差
ratesD = rates[window:] - ratesM[window:]
# 从收盘价中减去移动平均线值，得到一个去趋势化的时间序列
plt.plot(ratesD)
plt.show()

# 现在，我们得到每个交易小时的偏差分布小时统计信息：
Hourly_Returns = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).median()
Hourly_Returns.boxplot(column='close', by='hour', figsize=(15, 8))
plt.show()
# 下一个合乎逻辑的步骤是仔细检查分布矩，以便获得更准确的统计评估。 例如，以箱形图的形式计算得到的去趋势化序列的标准差：
Hourly_std = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).std()
Hourly_std.boxplot(column='close', by='hour', figsize=(15, 8))
plt.show()
# 另一个有趣的点是偏度。 我们计算一下：
Hourly_skew = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).skew()
Hourly_skew.boxplot(column='close', by='hour', figsize=(15, 8))
plt.show()
# 峰度的统计数据显示类似的结果：
Hourly_kurt = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).apply(pd.DataFrame.kurt)
Hourly_kurt.boxplot(column='close', by='hour', figsize=(15, 8))
plt.show()

#%%
### 搜索特定月份或一周中某天的周期形态，按均线去趋势化
rates_original = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_M15", [2010,1,1], [2019, 11, 25], index_time=False)
# leave only 'time' and 'close' columns
rates = rates_original[["time","close"]]
rates = rates.set_index('time')
# set the moving average period
window = 25
# detrend tome series by MA
ratesM = rates.rolling(window).mean()
ratesD = rates[window:] - ratesM[window:]
# 仅分析指定月份
ratesD = ratesD.drop(ratesD.index[~ratesD.index.month.isin([11])])
Hourly_Returns = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).median()
Hourly_Returns.boxplot(column='close', by='hour', figsize=(11, 6))
plt.show()

# 仅分析指定周几
ratesD = ratesD.drop(ratesD.index[~ratesD.index.dayofweek.isin([0])])
Hourly_Returns = ratesD.groupby([ratesD.index.day.rename('day'), ratesD.index.hour.rename('hour')]).median()
Hourly_Returns.boxplot(column='close', by='hour', figsize=(11, 6))
plt.show()







