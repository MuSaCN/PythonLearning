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
myMT5Report = MyMql.MyClass_StratTestReport()  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

'''
根据MT5文章：寻找市场形态的计量经济学方法：自相关，热点图和散点图
https://www.mql5.com/zh/articles/5451
'''

#%%
myMT5Pro.__init__(connect=True)

# 此函数将 H1 收盘价转换为指定周期的相关差额（所用为滞后 50），并显示自相关图表。
def standard_autocorrelation(symbol, lag):
    rates = myMT5Pro.getsymboldata(symbol, "TIMEFRAME_H1", [2015,1,1], [2020,1,1], index_time=False)
    rates = rates[["time","close"]].set_index('time')
    rates = rates.diff(lag).dropna()
    from pandas.plotting import autocorrelation_plot
    plt.figure(figsize=(10, 5))
    autocorrelation_plot(rates)
standard_autocorrelation(symbol='EURUSD', lag=50)
plt.show()

# 排除了时段的价格增量相关性图表
def seasonal_autocorrelation(symbol, lag=1, hour1=1, hour2=1):
    rates = myMT5Pro.getsymboldata(symbol, "TIMEFRAME_H1", [2019, 1, 1], [2021, 1, 1], index_time=False)
    rates = rates[["time", "close"]].set_index('time')
    rates.index = pd.to_datetime(rates.index, unit='s')
    rates = rates.drop(rates.index[~rates.index.hour.isin([hour1, hour2])]).diff(lag).dropna()
    from pandas.plotting import autocorrelation_plot
    plt.figure(figsize=(10, 5))
    autocorrelation_plot(rates)
seasonal_autocorrelation("EURUSD", 25, 1, 1)  # （仅剩每天的第一小时）
'''这意味着当天的第一小时与lag天之前的第一小时的差值增量紧密相关，依此类推。'''
'''注意：此处思路有问题'''

# 现在，我们查看相邻时段之间是否存在相关性。
seasonal_autocorrelation('EURUSD', 50, 1, 2)
seasonal_autocorrelation("EURUSD", 50, 12, 13)
plt.show()

#%%
### 所有时段的季节相关性热点图
# calculate correlation heatmap between all hours
def correlation_heatmap(symbol, lag, corrthresh):
    out = pd.DataFrame()
    rates = myMT5Pro.getsymboldata(symbol, "TIMEFRAME_H1", [2015, 1, 1], [2020, 1, 1], index_time=False)
    rates = rates[["time", "close"]].set_index('time')
    for i in range(24): # i=2
        ratesH = rates.drop(rates.index[~rates.index.hour.isin([i])]).diff(lag).dropna()
        out[str(i)] = ratesH['close'].reset_index(drop=True)
    plt.figure(figsize=(10, 10))
    corr = out.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr[corr >= corrthresh], mask=mask)
    return out
out = correlation_heatmap(symbol='EURUSD', lag=50, corrthresh=0.9)
out = correlation_heatmap(symbol='EURUSD', lag=25, corrthresh=0.9)
out = correlation_heatmap(symbol='EURUSD', lag=5, corrthresh=0.9)
plt.show()
out[['10','11','12','13','14']].describe()
'''由于趋势分量的出现，增量滞后的递增会导致时段间的相关性更大，而滞后的递减会导致较低的值。 不过，集簇的相对排列几乎未变。 '''
# 例如，对于单个滞后，时段 12、13 和 14 的增量仍然强烈相关：
plt.figure(figsize=(10,5))
plt.plot(out[['12','13','14']])
plt.legend(out[['12','13','14']])
plt.show()


#%%
### 连接图和计算收益预测的公式
def hourly_signals_statistics(symbol, lag, hour, hour2, rfilter):
    rates = myMT5Pro.getsymboldata(symbol, "TIMEFRAME_H1", [2015, 1, 1], [2020, 1, 1], index_time=False)
    rates = rates[["time", "close"]].set_index('time')
    # price differences for every hour series
    H = rates.drop(rates.index[~rates.index.hour.isin([hour])]).reset_index(drop=True).diff(lag).dropna() # H --> hour小时lag价格的序列
    H2 = rates.drop(rates.index[~rates.index.hour.isin([hour2])]).reset_index(drop=True).diff(lag).dropna() # H2 --> hour2小时lag价格的序列
    # 为了分析过去和未来，每小时延迟为 1 的序列
    # 当前的 current returns for both hours
    HF = H[1:].reset_index(drop=True)
    HL = H2[1:].reset_index(drop=True)
    # 前一期的 previous returns for both hours
    HF2 = H[:-1].reset_index(drop=True)
    HL2 = H2[:-1].reset_index(drop=True)

    # 根据前一个增量预测下一个增量。 为此，此处的一个简单公式，可以预测下一个值：
    # Basic equation:  ret[-1] = ret[0] - (ret[lag] - ret[lag-1])
    # or Close[-1] = (Close[0]-Close[lag]) - ((Close[lag]-Close[lag*2]) - (Close[lag-1]-Close[lag*2-1]))
    predicted = HF - (HF2 - HL2) # 当前13小时的 - (前一期13小时 - 前一期14小时)
    real = HL # 当前14小时的

    # 第一个和第二个小时的增量构建一个散点图 correlation joinplot between two series
    outcorr = pd.DataFrame()
    outcorr['Hour ' + str(hour)] = H['close']
    outcorr['Hour ' + str(hour2)] = H2['close']
    # 实际和预测增量的散点图 real VS predicted prices
    out = pd.DataFrame()
    out['real'] = real['close']
    out['predicted'] = predicted['close']
    out = out.loc[((out['predicted'] >= rfilter) | (out['predicted'] <= - rfilter))]
    # plptting results
    sns.jointplot(x='Hour ' + str(hour), y='Hour ' + str(hour2), data=outcorr, kind="reg", height=7, ratio=6)
    plt.show()
    sns.jointplot(x='real', y='predicted', data=out, kind="reg", height=7, ratio=6)
    plt.show()

hourly_signals_statistics(symbol='EURUSD', lag=25, hour=13, hour2=14, rfilter=0.00)
# 零附近的预测并不是很有趣：如果下一个价格增量的预测等于当前值，则您无法从中获利。 可以利用 rfilter 参数过滤预测。
hourly_signals_statistics(symbol='EURUSD', lag=25, hour=13, hour2=14, rfilter=0.03)

#%%
myDefault.set_backend_default("tkagg")
### 3D连接图的实际和预测回报
def hourly_signals_statistics3D(symbol, lag, hour, hour2, rfilter):
    rates = myMT5Pro.getsymboldata(symbol, "TIMEFRAME_H1", [2015, 1, 1], [2020, 1, 1], index_time=False)
    rates = rates[["time", "close"]].set_index('time')
    rates = pd.DataFrame(rates['close'].diff(lag)).dropna()

    out = pd.DataFrame();
    for i in range(hour, hour2):
        H = None;
        H2 = None;
        HF = None;
        HL = None;
        HF2 = None;
        HL2 = None;
        predicted = None;
        real = None;
        H = rates.drop(rates.index[~rates.index.hour.isin([hour])]).reset_index(drop=True)
        H2 = rates.drop(rates.index[~rates.index.hour.isin([i + 1])]).reset_index(drop=True)
        HF = H[1:].reset_index(drop=True);
        HL = H2[1:].reset_index(drop=True);  # current hours
        HF2 = H[:-1].reset_index(drop=True);
        HL2 = H2[:-1].reset_index(drop=True)  # last day hours

        predicted = HF - (HF2 - HL2)
        real = HL

        out3D = pd.DataFrame()
        out3D['real'] = real['close']
        out3D['predicted'] = predicted['close']
        out3D['predictedABS'] = predicted['close'].abs()
        out3D['hour'] = i
        out3D = out3D.loc[((out3D['predicted'] >= rfilter) | (out3D['predicted'] <= - rfilter))]

        out = out.append(out3D)

    import plotly.express as px
    fig = px.scatter_3d(out, x='hour', y='predicted', z='real', size='predictedABS', color='hour', height=1000, width=1000)
    fig.show()

hourly_signals_statistics3D('EURUSD', lag=24, hour=10, hour2=23, rfilter=0.000)







