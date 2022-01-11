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
# 本项目是通过 Python 驱动 MT5
# 为通用的模式"各过滤各平仓"，对各个过滤模式下、不同的止损模式，作不同的优化。
# 测试时间根据timeframe自动获取，进行了时间左移。
'''


#%% ******参数优化******
expertname = "My_Experts\简单动量策略(AB比较)_Test\EURUSD\EURUSD.H8.B.各过滤各平仓.ex5"
symbol = "EURUSD"
timeframe = "TIMEFRAME_H8"
direct = "BuyOnly"

reportfolder = r"F:\工作(同步)\工作---MT5策略研究\1.简单的动量策略\{}.{}\{}".format(symbol, myMT5code.timeframe_to_affix(timeframe), direct)


#%%
from MyPackage.MyProjects.止损与移动止损.各过滤各平仓 import MT5_filter_and_close
fnc = MT5_filter_and_close()
fnc.expertname = expertname
fnc.symbol = symbol
fnc.timeframe = timeframe


#%% FixedBars固定K线周期 (基准)
fnc.reportfile = reportfolder+r"\Opt_FixedBars.xml"
fnc.cache_affix = "FixedBars"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "0||0||0||5||N") # 指定为 FixedBars固定K线周期
fnc.input_set("init_sl_FixedBars", "0||0||100||3000||Y") # 下订单时的固定止损点优化
fnc.input_set("holding", "12||1||1||30||Y") # 固定周期平仓优化
fnc.run()

#%% FixedTrailing固定移动止损
fnc.reportfile = reportfolder+r"\Opt_FixedTrailing.xml"
fnc.cache_affix = "FixedTrailing"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "1||0||0||5||N") # 指定为 FixedTrailing固定移动止损
fnc.input_set("init_sl_FixedTrailing", "0||0||100||3000||Y") # 下订单时的固定止损点
fnc.input_set("fixed_trailing", "0||0||100||3000||Y") # 固定的移动止损点
fnc.run()

#%% FixedTrailing_Defend防御性移动止损(MT5界面模式)
fnc.reportfile = reportfolder+r"\Opt_FixedTrailing_Defend.xml"
fnc.cache_affix = "FixedTrailing_Defend"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "2||0||0||5||N") # 指定为 FixedTrailing_Defend防御性移动止损
fnc.input_set("init_sl_defend", "0||0||100||3000||Y") # 下订单时的固定止损点
fnc.input_set("fixed_trailing_defend", "0||0||100||3000||Y") # 防御性移动止损点
fnc.run()

#%% ATR_Trailing为ATR移动止损
fnc.reportfile = reportfolder+r"\Opt_ATR_Trailing.xml"
fnc.cache_affix = "ATR_Trailing"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "3||0||0||5||N") # 指定为 ATR_Trailing为ATR移动止损
fnc.input_set("ATR_Period", "1||1||1||60||Y") # ATR周期
fnc.input_set("Multiple", "1||0.5||0.1||3||Y") # ATR值的倍数
fnc.run()

#%% SAR_Trailing为SAT移动止损
fnc.reportfile = reportfolder+r"\Opt_SAR_Trailing.xml"
fnc.cache_affix = "SAR_Trailing"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "4||0||0||5||N") # 指定为 SAR_Trailing为SAT移动止损
fnc.input_set("SAR_Step", "0.02||0.01||0.001||0.1||Y") # SAR步长
fnc.input_set("SAR_Max", "0.2||0.14||0.02||0.26||Y") # SAR最大值
fnc.run()

#%% NBar_Trailing从shift=1开始上n根K线(包括shift位置的K线)的极值做移动止损
fnc.reportfile = reportfolder+r"\Opt_NBar_Trailing.xml"
fnc.cache_affix = "NBar_Trailing"
fnc.first_default()
fnc.input_set("filter_mode", "0||0||0||3||Y") # 各过滤优化
fnc.input_set("close_mode", "5||0||0||5||N") # 指定为 SAR_Trailing为SAT移动止损
fnc.input_set("n_count", "1||1||1||60||Y") # 从shift=1开始上n根K线(包括shift位置)的极值
fnc.input_set("adjust_point", "0||0||10||300||Y") # 极值调节点数
fnc.run()

