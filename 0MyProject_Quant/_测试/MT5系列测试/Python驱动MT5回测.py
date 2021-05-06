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
myMT5Report = MyMql.MyClass_StratTestReport()  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5() # Python运行MT5
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%%

out_config = __mypath__.get_desktop_path() + "\\MT5_common1.ini"

config_common = __mypath__.get_user_path() + r"\AppData\Roaming\MetaQuotes\Terminal\6E8A5B613BD795EE57C550F7EF90598D\config\common.ini"

# ---读取默认设置，编码 utf-16 为 mt5 的格式
myini.__init__(configfile=config_common, encoding="utf-16")

# ---启动智能交易系统测试或优化
Tester = "Tester"
myini.add_section(Tester)
# EA 位于 平台_数据_目录\MQL5\Experts\...
myini.set(Tester,'Expert','My_Experts\简单动量策略(AB比较)_Test\EURUSD.D1\各过滤各平仓.ex5')
# 初始存款、杠杆、品种
myini.set(Tester,'Deposit','5000')
myini.set(Tester,'Leverage','100')
myini.set(Tester,'Symbol','EURUSD')
# 用于测试/优化的时间帧，注意输入不是 TIMEFRAME_D1，而是 Daily
myini.set(Tester,'Period', myMT5run.timeframe_to_affix("TIMEFRAME_D1"))
# 优化(0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种")。
myini.set(Tester,'Optimization','0')
# "分时"模式 (0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点")
myini.set(Tester,'Model','1')
# 测试范围的起始和结束日期
myini.set(Tester,'FromDate','2000.01.01')
myini.set(Tester,'ToDate','2020.01.01')
# 向前测试的自定义模式(0--关闭, 1--1/2 测试周期, 2--1/3 测试周期, 3--1/4 测试周期, 4--使用 ForwardDate=2011.03.01 参数自定义指定间隔)。
myini.set(Tester,'ForwardMode','0')
# 随机延迟交易订单执行(0 - 正常，-1 - 执行交易订单时的随机延迟，&gt;0 - 以毫秒为单位的交易执行延迟，不可超过600000)
myini.set(Tester,'ExecutionMode','-1')
# 优化准则(0 -- Balance max, 1 -- Profit Factor max, 2 -- Expected Payoff max, 3 -- Drawdown min, 4 -- Recovery Factor max, 5 -- Sharpe Ratio max, 6 -- Custom max, 7 -- Complex Criterion max)
myini.set(Tester,'OptimizationCriterion','6')
# 报告文件将保存在文件夹 平台_安装_目录 # ABC\out_test.xml  ABC\out_test1.xlsx
myini.set(Tester,'Report','ABC\out_test1.xlsx')
# 如果指定报告已经存在, 它将被覆盖 (0 — 禁用, 1 — 启用)。
myini.set(Tester,'ReplaceReport','1')
# 设置测试/优化完成后, 平台随即自动关闭 (0 — 禁用, 1 — 启用)。
myini.set(Tester,'ShutdownTerminal','1')


# ---回测的参数设置
Inputs = "TesterInputs"
myini.add_section(Inputs)
# 参数设置
myini.set(Inputs,'filter_mode','1||0||0||3||N')
myini.set(Inputs,'close_mode','3||0||0||5||N')
myini.set(Inputs,'init_sl_FixedBars','1900||1900||1||19000||N')
myini.set(Inputs,'holding','12||12||1||120||N')
myini.set(Inputs,'init_sl_FixedTrailing','0||0||1||10||N')
myini.set(Inputs,'fixed_trailing','1950||1950||1||19500||N')
myini.set(Inputs,'init_sl_defend','1900||1900||1||19000||N')
myini.set(Inputs,'fixed_trailing_defend','1650||1650||1||16500||N')
myini.set(Inputs,'ATR_Period','29||29||1||290||N')
myini.set(Inputs,'Multiple','2.6||2.6||0.260000||26.000000||N')
myini.set(Inputs,'SAR_Step','0.016||0.016||0.001600||0.160000||N')
myini.set(Inputs,'SAR_Max','0.22||0.22||0.022000||2.200000||N')
myini.set(Inputs,'n_count','38||38||1||380||N')
myini.set(Inputs,'adjust_point','120||120||1||1200||N')

# ---输出结果
myini.write(out_config)

#%%
command = r"C:\Users\i2011\Desktop\MQL5系列\M3-ALL-MT5.lnk /config:C:\Users\i2011\Desktop\MT5_common1.ini"

import os
# 以阻塞方式运行
os.system(command)




