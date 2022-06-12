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
myplthtml = MyPlot.MyClass_PlotHTML()  # 画可以交互的html格式的图
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
一个策略要优化两个，一个是货币所在市场的时间，一个是货币在全部的市场时间。
'''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()


#%% ###### 外部参数 ######
experfolder = "My_Experts\\Strategy深度研究\\某市场形成盒子之后再突破策略"
fromdate = "2014.01.01"
todate = "2022.05.11"
reportfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\4.某市场形成盒子之后再突破策略"
symbol = "NZDUSD" # ************
timeframe = "TIMEFRAME_M30"

# ====== 策略参数 ======
# ------通用分析套件参数------
def common_set():
    myMT5run.input_set("FrameMode", "1") # 0-FRAME_None 1-FRAME_Result 2-FRAME_GUI
    # ; ======(通用)0.用于分析======
    myMT5run.input_set("Inp_Signal_Shift", "1")
    myMT5run.input_set("Inp_CustomMode", "0") # 0-TB
    myMT5run.input_set("Inp_IsBackTestCSV", "false")
    # ; ------1.入场限制------
    # ------1.1 同方向重复入场------
    myMT5run.input_set("Inp_Is_ReSignal", "false")
    myMT5run.input_set("Inp_ReSignalLimit", "0||0||1||10||N")
    # ------1.2 时间范围内同方向限定交易次数------
    myMT5run.input_set("Inp_Is_TimeRangeLimit", "true")
    myMT5run.input_set("Inp_TimeRangeLimit", "1||1||1||10||N")
    myMT5run.input_set("Inp_TimeRangeTF", "16408||0||0||49153||N") # 16408-1D
    myMT5run.input_set("Inp_TimeRangeTFShift", "0||0||1||10||N")
    # ; ------2.出场模式------
    myMT5run.input_set("Inp_Is_SigToCloseInver", "false")
    myMT5run.input_set("Inp_Is_PendToCloseInver", "false")
    myMT5run.input_set("Inp_FixedHolding", "0||4||1||24||Y")
    myMT5run.input_set("Inp_FixedHoldTF", "0")
    # ; ------3.信号过滤(范围和方向)------
    # ------3.1 范围过滤------
    myMT5run.input_set("Inp_FilterMode", "0||0||0||4||N") # 0-NoFilter, 1-Range, 2-TwoSide
    myMT5run.input_set("Inp_FilterIndiName", "") # 过滤指标名称
    myMT5run.input_set("Inp_FilterIndiTF", "_Period") # 过滤指标时间框字符串
    myMT5run.input_set("Inp_FilterIndiPara0", "0") # 过滤指标首个参数
    myMT5run.input_set("Inp_FilterLeftValue", "0") # 过滤指标左侧的值
    myMT5run.input_set("Inp_FilterRightValue", "0") # 过滤指标右侧的值
    # ------3.2 方向过滤------
    myMT5run.input_set("Inp_DirectMode", "3||2||0||4||Y") # 2-TwoSide, 3-Direct1, 4-Direct2
    myMT5run.input_set("Inp_DirectIndiName", "EMA")
    myMT5run.input_set("Inp_DirectIndiTF", "_Period")
    myMT5run.input_set("Inp_DirectIndiPara0", "100||0||0.000000||0.000000||N")
    myMT5run.input_set("Inp_DirectCompareCloseTF", "0")
    # ; ------4.1 初始止损设置------
    myMT5run.input_set("Inp_Init_SLMode", "2") # 2-SLMode_SpecifyDist
    myMT5run.input_set("Inp_SL_Point", "100||100||1||1000||N")
    myMT5run.input_set("Inp_SL_PreBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_SL_ATR_Period", "7||7||1||70||N")
    myMT5run.input_set("Inp_SL_ATR_N", "3||3||0.300000||30.000000||N")
    myMT5run.input_set("Inp_SL_SAR_Step", "0.02||0.02||0.002000||0.200000||N")
    myMT5run.input_set("Inp_SL_SAR_Max", "0.2||0.2||0.020000||2.000000||N")
    myMT5run.input_set("Inp_SL_RangeBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_SL_RangeN", "1.5||1.5||0.150000||15.000000||N")
    myMT5run.input_set("Inp_SL_Adjust", "0||0||1||10||N")
    # ; ------4.2 初始止盈设置------
    myMT5run.input_set("Inp_Init_TPMode", "3") # 3-TPMode_POINT
    myMT5run.input_set("Inp_TP_Point", "800||500||100||1200||Y")
    myMT5run.input_set("Inp_TP_SLMultiple", "1.5||1.5||0.150000||15.000000||N")
    myMT5run.input_set("Inp_TP_PreBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_TP_atr_Period", "7||7||1||70||N")
    myMT5run.input_set("Inp_TP_atr_N", "3||3||0.300000||30.000000||N")
    myMT5run.input_set("Inp_TP_SAR_Step", "0.02||0.02||0.002000||0.200000||N")
    myMT5run.input_set("Inp_TP_SAR_Max", "0.2||0.2||0.020000||2.000000||N")
    myMT5run.input_set("Inp_TP_RangeBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_TP_RangeN", "1.5||1.5||0.150000||15.000000||N")
    myMT5run.input_set("Inp_TP_Adjust", "0||0||1||10||N")
    # ; ------5.移动止损------
    myMT5run.input_set("Inp_Trailing_Mode", "0")
    myMT5run.input_set("Inp_Trail_StartProfit", "0||0||1||10||N")
    myMT5run.input_set("Inp_Trail_Point", "100||100||1||1000||N")
    myMT5run.input_set("Inp_Trail_PreBar", "3||3||1||30||N")
    myMT5run.input_set("Inp_Trail_PreBarTF", "0||0||0||49153||N")
    myMT5run.input_set("Inp_Trail_Atr_Period", "7||7||1||70||N")
    myMT5run.input_set("Inp_Trail_Atr_N", "3||3||0.300000||30.000000||N")
    myMT5run.input_set("Inp_Trail_SAR_Step", "0.02||0.02||0.002000||0.200000||N")
    myMT5run.input_set("Inp_Trail_SAR_Max", "0.2||0.2||0.020000||2.000000||N")
    myMT5run.input_set("Inp_Trail_Adjust", "10||10||1||100||N")
    # ; ------6.盈亏平衡------
    myMT5run.input_set("Inp_BreakEven_Mode", "1") # BreakEven_POINT
    myMT5run.input_set("Inp_BreakEven_Point", "150||100||50||600||Y")
    # ; ------7.挂单交易------
    myMT5run.input_set("Inp_PendingMode", "0")
    myMT5run.input_set("Inp_Is_PendDeal_SetSLTP", "true")
    myMT5run.input_set("Inp_Pending_PreBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_Pending_Atr_Period", "7||7||1||70||N")
    myMT5run.input_set("Inp_Pending_Atr_N", "0.33||0.33||0.033000||3.300000||N")
    myMT5run.input_set("Inp_Pending_RangeBar", "1||1||1||10||N")
    myMT5run.input_set("Inp_Pending_RangeN", "1||1||0.100000||10.000000||N")
    myMT5run.input_set("Inp_Pending_Adjust", "0||0||1||10||N")
    myMT5run.input_set("Inp_Pending_ExpireTF", "0||0||0||49153||N")
    myMT5run.input_set("Inp_Pending_ExpireBar", "3||3||1||30||N")
    # ; ------8.星期过滤------
    myMT5run.input_set("Inp_IsIn_MONDAY", "true")
    myMT5run.input_set("Inp_IsIn_TUESDAY", "true")
    myMT5run.input_set("Inp_IsIn_WEDNESDAY", "true")
    myMT5run.input_set("Inp_IsIn_THURSDAY", "true")
    myMT5run.input_set("Inp_IsIn_FRIDAY", "true")
    # ; ------9.资金管理------
    myMT5run.input_set("Inp_MM_Mode", "0||0||0||5||N")
    myMT5run.input_set("Inp_Lots_Fixed", "0.01||0.01||0.001000||0.100000||N")
    myMT5run.input_set("Inp_Lots_IncreDelta", "100||100||10.000000||1000.000000||N")
    myMT5run.input_set("Inp_Lots_IncreInitLots", "1||1||0.100000||10.000000||N")
    myMT5run.input_set("Inp_Lots_SLRiskPercent", "0.05||0.05||0.005000||0.500000||N")
    myMT5run.input_set("Inp_Lots_ATRPeriod", "14||14||1||140||N")
    myMT5run.input_set("Inp_Lots_ATRMultiple", "1||1||0.100000||10.000000||N")
    myMT5run.input_set("Inp_Is_Adjust_ATRRatio", "false||false||0||true||N")
    myMT5run.input_set("Inp_Lots_ATRRatio1", "5||5||1||50||N")
    myMT5run.input_set("Inp_Lots_ATRRatio2", "60||60||1||600||N")

# ------策略参数------
# myMT5run.input_set("StartHour", "13||0||1||2||Y") # 时间修改：**盘
# myMT5run.input_set("StartMinute", "0||0||30||30||Y")
# myMT5run.input_set("Box", "1||1||1||3||Y")
# myMT5run.input_set("AfterMarket", "1||1||1||3||Y")
# myMT5run.input_set("MaxSLPoint", "400||300||50||600||Y")

# ---延续策略信号过滤参数
def para_conti():
    myMT5run.input_set("IsFilter0", "false||false||0||true||Y")
    myMT5run.input_set("F0_BoxSize", "250||150||50||600||Y")
    myMT5run.input_set("IsFilter1", "false||false||0||true||Y")
    myMT5run.input_set("F1_ContiBar", "5||5||1||50||N")
    myMT5run.input_set("IsFilter2", "false||false||0||true||Y")
    myMT5run.input_set("F2_AroonPeriod", "50||50||1||500||N")
    myMT5run.input_set("F2_AroonLimit", "95||95||9.500000||950.000000||N")
    myMT5run.input_set("IsFilter3", "true||false||0||true||Y")
    myMT5run.input_set("F3_TradeTrend", "false||false||0||true||Y")
    myMT5run.input_set("F3_AMADiffLimit", "30||30||1||300||N")
    myMT5run.input_set("F3_AMADiffTF", "16408||0||0||49153||N")

# ---逆转策略信号过滤参数(信号过滤只有过滤3优化)
def para_reverse():
    myMT5run.input_set("IsFilter0", "false||false||0||true||N")
    myMT5run.input_set("F0_BoxSize", "250||150||50||600||N")
    myMT5run.input_set("IsFilter1", "false||false||0||true||N")
    myMT5run.input_set("F1_ContiBar", "5||5||1||50||N")
    myMT5run.input_set("IsFilter2", "false||false||0||true||N")
    myMT5run.input_set("F2_AroonPeriod", "50||50||1||500||N")
    myMT5run.input_set("F2_AroonLimit", "95||95||9.500000||950.000000||N")
    myMT5run.input_set("IsFilter3", "true||false||0||true||Y")
    myMT5run.input_set("F3_TradeTrend", "false||false||0||true||Y")
    myMT5run.input_set("F3_AMADiffLimit", "30||30||1||300||N")
    myMT5run.input_set("F3_AMADiffTF", "16408||0||0||49153||N")


#%% ###### d1.新西兰元澳洲盘突破延续策略 澳洲盘 策略优化 ######
expertfile = "d1.新西兰元澳洲盘突破延续策略.ex5" # ************
expertname = experfolder + "\\" + expertfile
reportfile = reportfolder + "\\{}.{}.{}.xml".format(expertfile.rsplit(sep=".", maxsplit=1)[0],symbol,timeframe)
optimization = 2 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)
common_set() # 通用分析套件参数
myMT5run.input_set("StartHour", "0||0||1||2||Y") # ******时间修改：澳洲盘******
myMT5run.input_set("StartMinute", "0||0||30||30||Y")
myMT5run.input_set("Box", "1||1||1||3||Y")
myMT5run.input_set("AfterMarket", "1||1||1||3||Y")
myMT5run.input_set("MaxSLPoint", "400||300||50||600||Y")
para_conti() # 延续策略信号过滤参数
# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()


#%% ###### d1.新西兰元澳洲盘突破延续策略 xx盘 策略优化 ######
expertfile = "d1.新西兰元澳洲盘突破延续策略.ex5"
expertname = experfolder + "\\" + expertfile
reportfile = reportfolder + "\\{}.{}.{}.xml".format(expertfile.rsplit(sep=".", maxsplit=1)[0],symbol,timeframe)
optimization = 2 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)
common_set() # 通用分析套件参数
myMT5run.input_set("StartHour", "0||0||1||20||Y") # ******时间修改：全部盘******
myMT5run.input_set("StartMinute", "0||0||30||30||Y")
myMT5run.input_set("Box", "1||1||1||3||Y")
myMT5run.input_set("AfterMarket", "1||1||1||3||Y")
myMT5run.input_set("MaxSLPoint", "400||300||50||600||Y")
para_conti() # 延续策略信号过滤参数
# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()


#%% ###### d2.新西兰元澳洲盘突破逆转策略 澳洲盘 策略优化 ######
expertfile = "d2.新西兰元澳洲盘突破逆转策略.ex5" # ************
expertname = experfolder + "\\" + expertfile
reportfile = reportfolder + "\\{}.{}.{}.xml".format(expertfile.rsplit(sep=".", maxsplit=1)[0],symbol,timeframe)
optimization = 2 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)
common_set() # 通用分析套件参数
myMT5run.input_set("StartHour", "0||0||1||2||Y") # ******时间修改：澳洲盘******
myMT5run.input_set("StartMinute", "0||0||30||30||Y")
myMT5run.input_set("Box", "1||1||1||3||Y")
myMT5run.input_set("AfterMarket", "1||1||1||3||Y")
myMT5run.input_set("MaxSLPoint", "400||300||50||600||Y")
para_reverse() # 逆转策略信号过滤参数
# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()


#%% ###### d2.新西兰元澳洲盘突破逆转策略 xx盘 策略优化 ######
expertfile = "d2.新西兰元澳洲盘突破逆转策略.ex5"
expertname = experfolder + "\\" + expertfile
reportfile = reportfolder + "\\{}.{}.{}.xml".format(expertfile.rsplit(sep=".", maxsplit=1)[0],symbol,timeframe)
optimization = 2 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)
common_set() # 通用分析套件参数
myMT5run.input_set("StartHour", "0||0||1||20||Y") # ******时间修改：全部盘******
myMT5run.input_set("StartMinute", "0||0||30||30||Y")
myMT5run.input_set("Box", "1||1||1||3||Y")
myMT5run.input_set("AfterMarket", "1||1||1||3||Y")
myMT5run.input_set("MaxSLPoint", "400||300||50||600||Y")
para_reverse() # 逆转策略信号过滤参数
# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()

