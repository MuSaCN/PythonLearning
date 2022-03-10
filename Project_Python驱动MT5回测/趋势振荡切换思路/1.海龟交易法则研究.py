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

#%%
'''
注意信号是 Cross 还是 Momentum，若要修改需要到 EA 中修改。
'''

#%% ###### 通用参数 ######
experfolder = "My_Experts\\Strategy走势分类研究\海龟交易法则趋势振荡分类"
expertfile = "1.海龟交易法则研究.Trend.ex5"
expertname = experfolder + "\\" + expertfile
fromdate = "2010.01.01"
todate = "2017.01.01"
symbol = "EURUSD"
timeframe = "TIMEFRAME_M15"
reportfolder = r"F:\工作(同步)\工作---MT5策略研究\海龟交易法则_趋势振荡分类研究"



#%% ###### Step1.0 优化信号参数和固定持仓 ######
reportfile = reportfolder + "\\{}.{}\\{}\\1.opt信号固定持仓.xml".format(symbol, timeframe, expertfile.rsplit(sep=".", maxsplit=1)[0])
optimization = 1 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)

myMT5run.input_set("Inp_ChannelPeriod", "24||5||1||100||Y")
# ======(通用)用于分析======
myMT5run.input_set("CustomMode", "0") # 设置自定义的回测结果 0-TB, 42-最大连亏
myMT5run.input_set("backtestmode", "0") # 0-FitnessPerformance
# ------1.固定持仓------
myMT5run.input_set("FixedHolding", "0||1||1||10||Y") # 0表示不是固定持仓模式，>0表示固定周期持仓。
# ------2.信号过滤------
myMT5run.input_set("FilterMode", "0||0||0||4||N") # 0-NoFilter, 1-Range, 2-TwoSide
# ------3.止盈止损------
myMT5run.input_set("InitTPPoint", "0||200||100||2000||N")
myMT5run.input_set("SL_Init", "false") # false表示没有初期止损，true表示有初期止损。
# ---止损设置 SL_Init=true 启用---
myMT5run.input_set("atr_Period", "5||3||1||36||N") # 止损ATR周期.
myMT5run.input_set("atr_N", "1.0||0.5||0.2||3.0||N") # ATR倍数.
# ------4.重复入场------
myMT5run.input_set("Is_ReSignal", "true") # true允许信号重复入场，false不允许信号重复入场。

# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()

#%% ###### Step1.1 找寻随着持仓周期增加策略表现递增的信号参数 ######
opt = myMT5Report.read_opt_xml(reportfile)
# ---
myDefault.set_backend_default("tkagg")
# ---固定指定, 排除 "FixedHolding" 后剩下的
para0,para1,para2 = "Result","FixedHolding",opt.columns[-2:].drop(para1)[0]
x, y, z = opt[para1], opt[para2], opt[para0]

# ---
myfig.__init__(1,1,figsize=[1024,768])
myfig.set_axes_3d2d()
myfig.plot3Ddf_trisurf(xs=x,ys=y,zs=z, PlotLabel=[para0,para1,para2])
# ---
y[x == 1]
z[x == 1]

#%%





#%% 单独一次回测
reportfile = reportfolder + "\\{}.{}\\{}\\1.best_test.xml".format(symbol, timeframe, expertfile.rsplit(sep=".", maxsplit=1)[0])
optimization = 0 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)

myMT5run.input_set("filter_mode", "1||0||0||3||Y")
myMT5run.input_set("close_mode", "3||0||0||5||N")
myMT5run.input_set("init_sl_FixedBars", "1900||1900||1||19000||N")
myMT5run.input_set("holding", "12||12||1||120||N")
myMT5run.input_set("init_sl_FixedTrailing", "0||0||1||10||N")
myMT5run.input_set("fixed_trailing", "1950||1950||1||19500||N")
myMT5run.input_set("init_sl_defend", "1900||1900||1||19000||N")
myMT5run.input_set("fixed_trailing_defend", "1650||1650||1||16500||N")
myMT5run.input_set("ATR_Period", "29||29||1||290||N")
myMT5run.input_set("Multiple", "2.6||2.6||0.260000||26.000000||N")
myMT5run.input_set("SAR_Step", "0.016||0.016||0.001600||0.160000||N")
myMT5run.input_set("SAR_Max", "0.22||0.22||0.022000||2.200000||N")
myMT5run.input_set("n_count", "38||38||1||380||N")
myMT5run.input_set("adjust_point", "120||120||1||1200||N")

# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.myini.items("TesterInputs")
myMT5run.run_MT5()

