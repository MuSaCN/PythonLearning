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
myini = MyFile.MyClass_INI() # ini文件操作类
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
myMT5run = MyMql.MyClass_RunningMT5() # Python运行MT5
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%% 参数优化
expertname = "My_Experts\简单动量策略(AB比较)_Test\EURUSD.D1\各过滤各平仓.ex5"
fromdate = "2000.01.01"
todate = "2020.01.01"
reportfile = r"F:\工作---MT5策略研究\1.简单的动量策略\EURUSD.D1\ABC\opt.xml"
myMT5run.__init__()
myMT5run.config_Tester(expertname, "EURUSD", "TIMEFRAME_D1", fromdate=fromdate, todate=todate,
                       delays=100, optimization=1, reportfile=reportfile)


#%% 设置参数
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
myMT5run.run_MT5()


#%% 单独一次回测
expertname = "My_Experts\简单动量策略(AB比较)_Test\EURUSD.D1\各过滤各平仓.ex5"
fromdate = "2000.01.01"
todate = "2020.01.01"
reportfile = r"F:\工作---MT5策略研究\1.简单的动量策略\EURUSD.D1\best_test.xml"
myMT5run.__init__()
myMT5run.config_Tester(expertname, "EURUSD", "TIMEFRAME_D1", fromdate=fromdate, todate=todate,
                       delays=100, optimization=0, reportfile=reportfile)

#%% 设置参数
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


