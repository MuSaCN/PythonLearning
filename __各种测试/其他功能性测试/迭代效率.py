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
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%% 算加法
sr = pd.Series([i+10 for i in range(10000)])

def func0(sr):
    temp = 0
    for i in sr:
        temp += i
    return temp
# %timeit out0 = func0(sr) # 1.32 ms ± 105 µs per loop
out0 = func0(sr)

def func1(sr):
    temp=0
    for item in sr.iteritems():
        temp += item[1]
    return temp
# %timeit out1 = func1(sr) # 1.98 ms ± 8.11 µs
out1 = func1(sr)

temp = [0]
def add(x):
    temp[0] += x
# %timeit x = sr.map(add) # 2.76 ms ± 7.1 µs
# %timeit x = sr.apply(add) # 3.13 ms ± 353 µs
x = sr.map(add)
x = sr.apply(add)

# %timeit out = sr.sum() # 41 µs ± 2.31 µs
out = sr.sum()

#%% 仅访问
sr = pd.Series([i+10 for i in range(10000)])

def call0(sr):
    for i in sr:
        pass
def call1(sr):
    for item in sr.iteritems():
        pass

# %timeit call0(sr) # 943 µs ± 24.3 µs
# %timeit call1(sr) # 1.47 ms ± 33.7 µs

def onlycall(x):
    pass
# %timeit a = sr.apply(onlycall) # 1.9 ms ± 35.7 µs
# %timeit a = sr.map(onlycall) # 1.88 ms ± 26.8 µs


#%%
data = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_H1",[2000,1,1], [2020,1,1],index_time=True, col_capitalize=True)
# 以迭代进行纯访问测试
def test1():
    for i in range(len(data)): # i=0
        data.iloc[i]["Time"]
#%timeit test1()
def test2():
    for index, row in data.iterrows():
        row["Time"]
#%timeit test2()
def test3():
    def func(x):
        x["Time"]
    data.apply(func,axis=1)
#%timeit test3()
import timeit
start = timeit.default_timer()
test1()
print("test1() Time used:", (timeit.default_timer() - start)) # 23.7911667999997
start = timeit.default_timer()
test2()
print("test2() Time used:", (timeit.default_timer() - start)) # 8.885708599999816
start = timeit.default_timer()
test3()
print("test3() Time used:", (timeit.default_timer() - start)) # 1.5251791999999114

