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
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

'''
切片指的是：把数据按每小时进行拆分，如数据的小时为1作为1组。
'''

#%%
### 数据切片的分布研究
symbol_list =['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDJPY','USDCAD','USDCHF','XAUUSD','XAGUSD'] # myMT5Pro.get_mainusd_symbol_name_list()
timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                  "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                  "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                  "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                  "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]
symbol = "AUDUSD"
timeframe = "TIMEFRAME_H1"
date_from, date_to = myMT5Pro.get_date_range(timeframe)
data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
data_total["C-O"] = data_total["Close"] - data_total["Open"]
data_total["C-O"] = data_total["C-O"].abs()

# 切片数据的分布统计，把原数据按照每小时进行拆分
def slice_statistic(affix = "Range"):
    df_out = pd.DataFrame()
    for limited_i in range(24):
        # ---以指定的时间词缀进行数据切片。# mode = "minute"/"hour"/"day"/"day_of_week"/"days_in_month"/"month"/"quarter"# limited 选择限制于里面的元素。
        data_choose = myMT5Pro.slice_by_timeaffix(data_total, mode="hour", limited=[limited_i])
        data_vola = data_choose[affix].dropna()
        # myplt.hist(data_vola,bins=100, objectname="%s%s"%(affix, limited_i), show=True)
        out = myDA.describe(data_vola, modeshow=True, return_out=True)
        out.columns = ["%s%s" % (affix, limited_i)]
        df_out = pd.concat((df_out, out), axis=1)
    return df_out

# 画统计图
def plot_statistic(df_out, affix = "Range"):
    myfig.__init__(1,1, figsize=[1280,720], AddFigure=True)
    mean = df_out.loc["mean"].reset_index(drop=True)
    std = df_out.loc["std"].reset_index(drop=True)
    ax = myfig.axeslist[0]
    myfig.plot_line(mean, axesindex=0, objectname = "mean", show=False)
    ax.legend(loc="upper right")
    myfig.plot_line(std, axesindex=0, objectname = "std", show=False, color="red",twinXY="X")
    ax.legend(loc="upper left")
    myfig.suptitle(affix+ ": mean+std")
    myfig.show()
    #---
    myfig.__init__(1, 1, figsize=[1280, 720], sharex=True, AddFigure=True)
    skew = df_out.loc["skew偏度"].reset_index(drop=True)
    kurt = df_out.loc["kurt峰度"].reset_index(drop=True)
    ax = myfig.axeslist[0]
    myfig.plot_line(skew, axesindex=0, objectname = "skew", show=False)
    ax.legend(loc="upper right")
    myfig.plot_line(kurt, axesindex=0, objectname = "kurt", show=False, color="red", twinXY="X")
    ax.legend(loc="upper left")
    myfig.suptitle(affix+ ": skew+kurt")
    myfig.show()



#%%
df_out = slice_statistic(affix = "Range")
df_out
plot_statistic(df_out, affix = "Range")

#%%
df_out = slice_statistic(affix = "Rate")
df_out
plot_statistic(df_out, affix = "Rate")

#%%
df_out = slice_statistic(affix = "RateInt")
df_out
plot_statistic(df_out, affix = "RateInt")

#%%
df_out = slice_statistic(affix = "LogRate")
df_out
plot_statistic(df_out, affix = "LogRate")

#%%
df_out = slice_statistic(affix = "C-O")
df_out
plot_statistic(df_out, affix = "C-O")



