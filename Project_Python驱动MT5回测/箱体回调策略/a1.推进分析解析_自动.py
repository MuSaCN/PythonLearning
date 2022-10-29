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
myMT5Analy = MyMT5Analysis.MyClass_ForwardAnalysis()  # MT5分析类
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


# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')


# %%
''' # 输出内容保存到"工作---MT5策略研究"目录，以及MT5的Common目录。 '''
import warnings
warnings.filterwarnings('ignore')


# (***)基础EA(***)。用于优化分析的，注意不同于下面推进回测的EA，后者要阶段更新参数。
expertfile = "a1.箱体回调策略.ex5"
contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\7.箱体回调策略" # 输出的总目录******
# (***)根据基础EA源码的Input变量的顺序来整理下面参数名(***)
ea_inputparalist = ["MaxBoxPeriod", "OsciBoxPeriod", "K_TrendBuyU", "K_TrendBuyD", "TrendGap",
                    "K_OsciBuyLevel", "OsciGap", "CloseBuyLevel", "PriceGap", "MaxSpread",
                    "SL_Min", "SL_Max", "AvgLotsToPPoint_L", "AvgLotsToPPoint_R"]


symbollist = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "XAUUSD", "XAGUSD", "AUDJPY","CHFJPY","EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","GBPAUD","GBPCAD","GBPCHF","GBPJPY","NZDJPY"] # 策略的品种列表******
# symbollist = ["EURUSD"]

timeframe = "TIMEFRAME_M15" # 策略的时间框******


# (***)推进分析的相关参数(***)
forward_starttime = "2015.01.01" # 推进分析数据的开始时间******
forward_endtime = "2022.07.01" # 推进分析数据的结束时间(最后一个格子只做优化，不做推进)******
length_year = 2 # 1,2 # 样本总时间包括训练集和测试集，单位年(允许小数)******
step_months = 6 # 3,6 # 推进步长，单位月(允许大于12)******


# (***)优化词缀(***): -1 Complete, 0 Balance max, 6 Custom max, 7 Complex Criterion max.
optcriterionaffix = myMT5run.get_optcriterion_affix(optcriterion=0) # ******


# (***)推进回测EA的目录(后面不能带\\)和文件名(***)
bt_experfolder = "My_Experts\\Strategy深度研究\\箱体回调策略".format(optcriterionaffix)
bt_expertfile = "a1.推进交易.{}.{}.ex5".format("_Symbol", myMT5run.timeframe_to_ini_affix(timeframe))
# (***)推进回测的时间起始(***)
bt_starttime = "2015.07.01"  # 手动指定******
bt_endtime = "2022.07.1"  # 手动指定******
# 推进回测保存的总目录
bt_folder = contentfolder + r"\1.推进回测.{}.{}.{}".format(optcriterionaffix,
    myMT5run.change_timestr_format(bt_starttime), myMT5run.change_timestr_format(bt_endtime))


#%% ###### 主要函数 ######
# ------通用分析套件参数(版本2022.10.22)------
# 使用时要修改，请标注 *******
def common_set():
    myMT5run.input_set("FrameMode", "2") # 0-FRAME_None 1-BTMoreResult 2-OptResult
    # ; ======(通用)0.用于分析======
    myMT5run.input_set("Inp_Signal_Shift", "1") # >=1为信号确认，且每bar运行一次，=0为实时。
    myMT5run.input_set("Inp_CustomMode", "0") # 0-TB
    myMT5run.input_set("Inp_IsBackTestCSV", "false") # 单次回测是否输出csv结果
    # ; ------1.入场限制------
    # ------1.1 同方向重复入场------
    myMT5run.input_set("Inp_Is_ReSignal", "false") # true允许信号同向重复入场，false不允许。
    myMT5run.input_set("Inp_ReSignalLimit", "0||0||1||10||N") # 当true时，允许重复入场的同方向订单限制，0不限制。
    # ------1.2 时间范围内同方向限定交易次数------
    myMT5run.input_set("Inp_Is_TimeRangeLimit", "false") # true时间范围内限定交易次数(包括历史)，false不限定。
    myMT5run.input_set("Inp_TimeRangeLimit", "1||1||1||10||N") # 当true时，时间范围内同方向限定交易次数。
    myMT5run.input_set("Inp_TimeRangeTF", "16408||0||0||49153||N") # 当true时，时间范围的时间框. # 16408-1D
    myMT5run.input_set("Inp_TimeRangeTFShift", "0||0||1||10||N") # 当true时，Shift=0为从当前bar至今，1为从上个bar至今。
    # ------1.3 节假日刚开盘触发入场信号------
    myMT5run.input_set("Inp_Is_AfHoliOpLimit", "false") # true限制节假日刚开盘的入场信号，false不限制.
    # ; ------2.出场模式------
    myMT5run.input_set("Inp_Is_SigToCloseInver", "true") # true信号平反向仓，false则不是。sig=4不适合.
    myMT5run.input_set("Inp_Is_PendToCloseInver", "true") # true挂单成交平反向仓，false则不是。
    myMT5run.input_set("Inp_FixedHolding", "0||0||1||10||N") # 0表示不是固定持仓模式，>0表示固定周期持仓。
    myMT5run.input_set("Inp_FixedHoldTF", "0") # FixedHolding的时间框
    myMT5run.input_set("Inp_AvgLotsToProfit_L", "0||0.0||0.000000||0.000000||N") # 1.AvgLotsToProfit_L:一局单子平均1仓位净利润达到指定额度平仓。LR都为0不启用。
    myMT5run.input_set("Inp_AvgLotsToProfit_R", "0||0.0||0.000000||0.000000||N") # 1.AvgLotsToProfit_R:一局单子平均1仓位净利润达到指定额度平仓。LR都为0不启用。
    myMT5run.input_set("Inp_AvgLotsToPPoint_L", "0||0||1||10||N") # 2.AvgLotsToPPoint_L:一局单子平均1仓位净利润达到指定点数平仓。LR都为0不启用。
    myMT5run.input_set("Inp_AvgLotsToPPoint_R", "0||0||1||10||N") # 2.AvgLotsToPPoint_R:一局单子平均1仓位净利润达到指定点数平仓。LR都为0不启用。
    myMT5run.input_set("Inp_CumLotsToProfit_L", "0||0.0||0.000000||0.000000||N") # 3.CumLotsToProfit_L:一局单子总净利润达到指定额度平仓。LR都为0不启用。
    myMT5run.input_set("Inp_CumLotsToProfit_R", "0||0.0||0.000000||0.000000||N") # 3.CumLotsToProfit_R:一局单子总净利润达到指定额度平仓。LR都为0不启用。
    myMT5run.input_set("Inp_CumLotsToPercBalance_L", "0||0.0||0.000000||0.000000||N") # 4.CumLotsToPercBalance_L:一局单子总净利润达到Balance指定百分比平仓。LR都为0不启用。
    myMT5run.input_set("Inp_CumLotsToPercBalance_R", "0||0.0||0.000000||0.000000||N") # 4.CumLotsToPercBalance_R:一局单子总净利润达到Balance指定百分比平仓。LR都为0不启用。
    # ; ------3.信号过滤(范围和方向)------
    # ------3.1 范围过滤------
    myMT5run.input_set("Inp_FilterMode", "0||0||0||2||N") # 0-NoFilter, 1-Range, 2-TwoSide
    myMT5run.input_set("Inp_FilterIndiName", "") # 过滤指标名称
    myMT5run.input_set("Inp_FilterIndiTF", "_Period") # 过滤指标时间框字符串
    myMT5run.input_set("Inp_FilterIndiPara0", "0") # 过滤指标首个参数
    myMT5run.input_set("Inp_FilterLeftValue", "0") # 过滤指标左侧的值
    myMT5run.input_set("Inp_FilterRightValue", "0") # 过滤指标右侧的值
    # ------3.2 方向过滤------
    myMT5run.input_set("Inp_DirectMode", "0||2||0||4||N") # 2-TwoSide, 3-Direct1, 4-Direct2
    myMT5run.input_set("Inp_DirectIndiName", "") # 方向指标名称
    myMT5run.input_set("Inp_DirectIndiTF", "_Period") # 方向指标时间框字符串
    myMT5run.input_set("Inp_DirectIndiPara0", "0||0||0.000000||0.000000||N") # 方向指标首个参数
    myMT5run.input_set("Inp_DirectCompareCloseTF", "0") # 与方向指标作比较的close时间框
    # ; ------4.1 初始止损设置------
    myMT5run.input_set("Inp_Init_SLMode", "0") # 0-SLMode_NONE, 2-SLMode_SpecifyDist, 3-SLMode_POINT
    myMT5run.input_set("Inp_SL_Point", "100||100||1||1000||N") # SLMode_POINT模式：指定止损点
    myMT5run.input_set("Inp_SL_PreBar", "1||1||1||10||N") # SLMode_BAR模式：信号前的bar数量
    myMT5run.input_set("Inp_SL_ATR_Period", "7||7||1||70||N") # SLMode_ATR模式(shift=1)：止损ATR周期.
    myMT5run.input_set("Inp_SL_ATR_N", "3||3||0.300000||30.000000||N") # SLMode_ATR模式(shift=1)：ATR倍数.
    myMT5run.input_set("Inp_SL_SAR_Step", "0.02||0.02||0.002000||0.200000||N") # SLMode_SAR模式(shift=0)：SAR_Step.
    myMT5run.input_set("Inp_SL_SAR_Max", "0.2||0.2||0.020000||2.000000||N") # SLMode_SAR模式(shift=0)：SAR_Max.
    myMT5run.input_set("Inp_SL_RangeBar", "1||1||1||10||N") # SLMode_Range：计算range的bar数量
    myMT5run.input_set("Inp_SL_RangeN", "1.5||1.5||0.150000||15.000000||N") # SLMode_Range：range的倍数
    myMT5run.input_set("Inp_SL_Adjust", "0||0||1||10||N") # SLMode_*模式：调节点数.
    myMT5run.input_set("Inp_SL_Min", "0||0||1||10||N")
    myMT5run.input_set("Inp_SL_Max", "0||0||1||10||N")
    # ; ------4.2 初始止盈设置------
    myMT5run.input_set("Inp_Init_TPMode", "0") # 0-TPMode_NONE, 3-TPMode_POINT
    myMT5run.input_set("Inp_TP_Point", "0||0||1||10||N") # TPMode_POINT模式：0表示没有。
    myMT5run.input_set("Inp_TP_SLMultiple", "1.5||1.5||0.150000||15.000000||N") # TPMode_PnLRatio模式：止盈盈亏比。
    myMT5run.input_set("Inp_TP_PreBar", "1||1||1||10||N") # TPMode_BAR模式：信号前的bar数量
    myMT5run.input_set("Inp_TP_atr_Period", "7||7||1||70||N") # TPMode_ATR模式(shift=1)：止盈ATR周期.
    myMT5run.input_set("Inp_TP_atr_N", "3.0||3.0||0.300000||30.000000||N") # TPMode_ATR模式(shift=1)：ATR倍数.
    myMT5run.input_set("Inp_TP_SAR_Step", "0.02||0.02||0.002000||0.200000||N") # TPMode_SAR模式(shift=0)：SAR_Step.
    myMT5run.input_set("Inp_TP_SAR_Max", "0.2||0.2||0.020000||2.000000||N") # TPMode_SAR模式(shift=0)：SAR_Max.
    myMT5run.input_set("Inp_TP_RangeBar", "1||1||1||10||N") # TPMode_Range：计算range的bar数量
    myMT5run.input_set("Inp_TP_RangeN", "1.5||1.5||0.150000||15.000000||N") # TPMode_Range：range的倍数
    myMT5run.input_set("Inp_TP_Adjust", "0||0||1||10||N") # TPMode_*模式：调节点数.
    # ; ------5.移动止损------
    myMT5run.input_set("Inp_Trailing_Mode", "0")
    myMT5run.input_set("Inp_Trail_StartProfit", "0||0||1||10||N") # 移动止损启动的利润
    myMT5run.input_set("Inp_Trail_Point", "100||100||1||1000||N") # TrailMode_POINT模式：固定点移动止损.
    myMT5run.input_set("Inp_Trail_PreBar", "3||3||1||30||N") # TrailMode_BAR模式(shift=1)：信号前的bar数量
    myMT5run.input_set("Inp_Trail_PreBarTF", "0||0||0||49153||N") # TrailMode_BAR模式：信号前的bar时间框
    myMT5run.input_set("Inp_Trail_Atr_Period", "7||7||1||70||N") # TrailMode_ATR模式(shift=1)：移动止损ATR周期.
    myMT5run.input_set("Inp_Trail_Atr_N", "3||3||0.300000||30.000000||N") # TrailMode_ATR模式(shift=1)：ATR倍数.
    myMT5run.input_set("Inp_Trail_SAR_Step", "0.02||0.02||0.002000||0.200000||N") # TrailMode_SAR模式(shift=0)：SAR_Step
    myMT5run.input_set("Inp_Trail_SAR_Max", "0.2||0.2||0.020000||2.000000||N") # TrailMode_SAR模式(shift=0)：SAR_Max
    myMT5run.input_set("Inp_Trail_Adjust", "10||10||1||100||N") # TrailMode_*模式：调节点数(Fixed不适用).
    # ; ------6.盈亏平衡------
    myMT5run.input_set("Inp_BreakEven_Mode", "0") # 1-BreakEven_POINT
    myMT5run.input_set("Inp_BreakEven_Point", "200||100||50||1000||N") # BreakEven_POINT模式: 达到多少点利润进行盈亏平衡
    myMT5run.input_set("Inp_BreakEven_CostPoint", "0") # 盈亏平衡的成本点，比如Commission占用的点数。
    # ; ------7.1 挂单交易------
    myMT5run.input_set("Inp_PendingMode", "0") # 0-PENDMode_NONE直接交易
    myMT5run.input_set("Inp_Is_PendDeal_SetSLTP", "true") # true挂单成交后再设置止损止盈；false直接设置再挂单。
    myMT5run.input_set("Inp_Pending_PreBar", "1||1||1||10||N") # STOP_BAR / LIMIT_BAR:在之前的N根极值处挂单(有挂单价格无效)。
    myMT5run.input_set("Inp_Pending_Atr_Period", "7||7||1||70||N") # STOP_ATR / LIMIT_ATR：挂单ATR周期.
    myMT5run.input_set("Inp_Pending_Atr_N", "0.33||0.33||0.033000||3.300000||N") # STOP_ATR / LIMIT_ATR：ATR倍数.
    myMT5run.input_set("Inp_Pending_RangeBar", "1||1||1||10||N") # STOP_RANGE / LIMIT_RANGE：用之前的N根bar(tf默认当前)的range
    myMT5run.input_set("Inp_Pending_RangeN", "1||1||0.100000||10.000000||N") # STOP_RANGE / LIMIT_RANGE：Range的多少倍
    myMT5run.input_set("Inp_Pending_Adjust", "0||0||1||10||N") # 挂单：点数修正挂单位置(有挂单价格无效)。
    myMT5run.input_set("Inp_Pending_ExpireTF", "0||0||0||49153||N") # 挂单：挂单有效的时间框
    myMT5run.input_set("Inp_Pending_ExpireBar", "3||3||1||30||N") # 挂单：挂单有效的Bar个数
    # ; ------7.2 挂单移动修改(吸附性原理) ------
    myMT5run.input_set("Inp_MovePendMode", "0")  # 移动挂单的模式
    myMT5run.input_set("Inp_MovePendPrice", "-1||-1||1||-10||N")  # MovePend_POINT: 挂单移动的价格点数
    myMT5run.input_set("Inp_MovePendSL", "-1||-1||1||-10||N")  # MovePend_POINT: 挂单移动的止损点数
    myMT5run.input_set("Inp_MovePendTP", "-1||-1||1||-10||N")  # MovePend_POINT: 挂单移动的止盈点数
    # ; ------8.时间过滤------
    myMT5run.input_set("Inp_IsIn_MONDAY", "true") # 允许星期一入场
    myMT5run.input_set("Inp_IsIn_TUESDAY", "true") # 允许星期二入场
    myMT5run.input_set("Inp_IsIn_WEDNESDAY", "true") # 允许星期三入场
    myMT5run.input_set("Inp_IsIn_THURSDAY", "true") # 允许星期四入场
    myMT5run.input_set("Inp_IsIn_FRIDAY", "true") # 允许星期五入场
    myMT5run.input_set("Inp_StartEndTime", "00:00-23:59") # 允许入场的开始小时
    # ; ------9.初始仓单资金管理------
    myMT5run.input_set("Inp_MM_Mode", "0||0||0||7||N") # 0-MM_Minimum
    myMT5run.input_set("Inp_Lots_Fixed", "0.01||0.01||0.001000||0.100000||N") # MM_Fixed模式：固定仓位
    myMT5run.input_set("Inp_Lots_IncreDelta", "100||100||10.000000||1000.000000||N") # MM_FixedIncrement模式：原书建议delta值可以设置为"基仓回测系统"中：历史最大回撤数值的一半 或者 最大亏损额的倍数。
    myMT5run.input_set("Inp_Lots_IncreInitLots", "0.1||1||0.100000||10.000000||N") # MM_FixedIncrement模式：初始仓位，可以调节大。
    myMT5run.input_set("Inp_Lots_SLRiskPercent", "0.05||0.05||0.005000||0.500000||N") # MM_SL模式, MM_SL_ATR模式：所用资金比例
    myMT5run.input_set("Inp_Lots_ATRPeriod", "14||14||1||140||N") # MM_SL_ATR模式：ATR周期
    myMT5run.input_set("Inp_Lots_ATRMultiple", "1||1||0.100000||10.000000||N") # MM_SL_ATR模式：ATR倍数
    myMT5run.input_set("Inp_Lots_BasicEveryLot", "5000||5000.0||500.000000||50000.000000||N")  # MM_EqutiyRatio模式：每手匹配的净值数量
    myMT5run.input_set("Inp_Is_Adjust_ATRRatio", "false||false||0||true||N") # Is_Adjust_ATRRatio=true：用lots=lots/ATR_Ratio来修正仓位.
    myMT5run.input_set("Inp_Lots_ATRRatio1", "5||5||1||50||N") # Is_Adjust_ATRRatio=true，FastATR周期.
    myMT5run.input_set("Inp_Lots_ATRRatio2", "60||60||1||600||N") # Is_Adjust_ATRRatio=true，SlowATR周期.
    # ; ------10.加仓管理------
    # ------10.1加仓基础设置------
    myMT5run.input_set("Inp_Is_StartAddIn", "false") # true开启加仓管理，必须设置Is_ReSignal=true.
    myMT5run.input_set("Inp_TargetCommentAffix", "Affix") # 标的单注释词缀，总格式=MainComment.Affix
    myMT5run.input_set("Inp_AddIn_Profit", "true") # true盈利加仓，false亏损加仓。
    myMT5run.input_set("Inp_PnL_PointLeft", "0||0||50||400||N") # 盈亏加仓左侧的点数，亏损加仓要设为负值。
    myMT5run.input_set("Inp_Pnl_PointRight", "9999||9999||1||99990||N") # 盈亏加仓右侧的点数，亏损加仓要设为负值。
    myMT5run.input_set("Inp_AddIn_IntervalTF", "0") # 加仓的时间间隔timeframe.
    myMT5run.input_set("Inp_AddIn_IntervalBar", "0||0||1||10||N") # 加仓的时间间隔bar，0表示没有。
    # ------10.2加仓止盈损、仓位大小设置------
    myMT5run.input_set("Inp_TIB_SLMode", "3") # TIB_SLMode加仓单基于标的单的止损模式(标的单若盈亏平衡会冲突).
    myMT5run.input_set("Inp_TIB_TPMode", "2") # TIB_TPMode加仓单基于标的单的止盈模式.
    myMT5run.input_set("Inp_TIB_LotsMode", "0") # TIB_LotsMode加仓单基于标的单的仓位模式.
    myMT5run.input_set("Inp_TIB_RatioLots", "0.5||0.5||0.050000||5.000000||N") # 当TIB_LotsMode=TIB_Lots_Ratio启用，以标的单持仓大小的比例加仓.
    # ------10.3加仓策略方法------
    myMT5run.input_set("Inp_TIB_Method", "0") # TIB_Method加仓策略方法.
    myMT5run.input_set("Inp_TIB_MaxAddCount", "0||1||1||10||N") # TIB_MaxAddCount最大的加仓次数(通用)，0不限次数。
    myMT5run.input_set("Inp_TIB_AddInPoint", "100||100||1||1000||N") # TIB_Method_Point模式：价格每次移动点数进行加仓.
    myMT5run.input_set("Inp_TIB_ATRPeriod", "14||14||1||140||N") # TIB_Method_ATR模式：ATR周期.
    myMT5run.input_set("Inp_TIB_ATRMultiple", "0.5||0.5||0.050000||5.000000||N") # TIB_Method_ATR模式：ATR倍数.

# ---(***)推进回测策略参数(***)---
def strategy_set():
    myMT5run.input_set("PriceGap", "999||999||1||9990||N")
    myMT5run.input_set("MaxSpread", "999||999||1||9990||N")
    myMT5run.input_set("SL_Min", "0||0||1||10||N")
    myMT5run.input_set("SL_Max", "0||0||1||10||N")
    myMT5run.input_set("AvgLotsToPPoint_L", "0||0||1||10||N")
    myMT5run.input_set("AvgLotsToPPoint_R", "0||0||1||10||N")
    myMT5run.input_set("OptCriterion", optcriterionaffix)


# ---获取 timedf, matchlist, violent
def get_timedf_matchlist_and_violent():
    # 推进测试的起止时间
    timedf = myMT5Analy.get_everystep_time(starttime, endtime, step_months=step_months,
                                           length_year=length_year)
    # 保存推进时间
    myfile.makedirs(forwardparapath, True)
    timedf.to_csv(forwardparapath + "\\推进时间.{}.{}.{}.{}.length={}.step={}.csv".
                  format(symbol,myMT5Analy.timeframe_to_ini_affix(timeframe),
                         timeaffix0,timeaffix1, length, step),sep=",")  # 逗号的csv可直接被excel解析。
    # ---批量读取推进优化的报告(csv比xlsx速度快)，保存到matchlist中 [[0,1],[0,1]]--- 0 trainmatch, 1 testmatch.
    matchlist = []  # [[0,1]]
    for i, row in timedf.iterrows():
        # 时间参数必须转成"%Y.%m.%d"字符串
        fromdate = row["from"]
        forwarddate = row["forward"]
        todate = row["to"]
        # ---xlsx格式优化报告
        islast = pd.Timestamp(forwarddate) == pd.Timestamp(endtime)
        tf_affix = myMT5run.timeframe_to_ini_affix(timeframe)
        t0 = myMT5run.change_timestr_format(fromdate)
        t1 = myMT5run.change_timestr_format(forwarddate) if islast is False else None
        t2 = myMT5run.change_timestr_format(todate) if islast is False \
            else myMT5run.change_timestr_format(forwarddate)
        # ---注意最后一个不是推进优化
        csvfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.csv".\
            format(expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2)
        print("读取 csvfile=", csvfile)
        trainmatch, testmatch = myMT5Analy.read_forward_opt_csv(filepath=csvfile)
        if (len(testmatch) == 0):
            print("  注意：此 csvfile 不是向前优化！")
        matchlist.append([trainmatch, testmatch])

    # ---把表示负面意义的数据改成负数。
    negetivelist = ["%最大相对回撤比", "最大相对回撤比占额", "最大绝对回撤值", "%最大绝对回撤值占比",
                    "LRStandardError", "亏损交易数量", "(int)最长亏损序列","(int)最大的连亏序列数",
                    "平均连亏序列"]
    for i in range(len(matchlist)):  # i=0
        trainmatch = matchlist[i][0]  # 这里不需要copy()
        testmatch = matchlist[i][1]  # 这里不需要copy()
        for nege in negetivelist:
            trainmatch[nege] = -1 * trainmatch[nege]
            testmatch[nege] = -1 * testmatch[nege]

    # ---设置自定义准则
    mycriterion = "myCriterion"
    for i in range(len(matchlist)):
        trainmatch = matchlist[i][0]  # 这里不需要copy()
        testmatch = matchlist[i][1]  # 这里不需要copy()
        try:
            trainmatch.drop(labels=mycriterion, axis=1, inplace=True)
            testmatch.drop(labels=mycriterion, axis=1, inplace=True)
        except:
            pass
        #
        trainmatch.insert(loc=2, column=mycriterion, value=None)
        trainmatch[mycriterion] = np.power(trainmatch["总交易"], 0.5) * trainmatch["平均盈利"] *\
                                  np.power(trainmatch["盈利总和"], 0.5) /\
                                  np.power(np.abs(trainmatch["亏损总和"]), 0.5) \
                                  * np.power(trainmatch["盈利交易数量"], 0.5)
        #
        testmatch.insert(loc=2, column=mycriterion, value=None)
        testmatch[mycriterion] = np.power(testmatch["总交易"], 0.5) * testmatch["平均盈利"] *\
                                 np.power(testmatch["盈利总和"],0.5) / \
                                 np.power(np.abs(testmatch["亏损总和"]), 0.5) * \
                                 np.power(testmatch["盈利交易数量"], 0.5)

    ### 展示相关性 ###
    print("1: 获取 matchlist，len(matchlist)=", len(matchlist))
    # 获取训练集测试集相关性的界限计数，比如某个相关性的绝对值>0.5，分数加1。
    totalcorr = myMT5Analy.traintest_corr_score(matchlist=matchlist, corrlimit=[0.5, 0.6, 0.7, 0.8, 0.9])

    ### 暴力测试下怎么筛选结果较好(循环比多线程好，多进程不方便) ###
    ### 第一次筛选
    if __mypath__.path_exists(choosefilename):
        # violent1 = violent # 用于研究超参数
        violent = myfile.read_pd(choosefilename, index_col=0)
    else:
        # 筛选第一步：根据violent选择一个占优势的排序方式
        sortbylist = trainmatch.loc[:, "净利润":"亏损交易中的最大值"].columns  # ["平均盈利"]
        choosebylist = ["myCriterion", "TB", "Sharpe_MT5", "SQN_MT5_No", "Sharpe_Balance",
                        "SQN_Balance","SQN_Balance_No", "Sharpe_Price", "SQN_Price",
                        "SQN_Price_No", "平均盈利", "盈亏比", "利润因子", "恢复因子",
                        "期望利润", "Kelly占用仓位杠杆", "Kelly止损仓位比率", "Vince止损仓位比率",
                        "回归系数", "LRCorrelation", "盈利总和"]  # ["TB"]
        resultlist = ["TB", "净利润"]  # ***非循环迭代***
        # 筛选第二步。选出每个分组的第一个，即sortby排序第一个
        func = lambda x: x.iloc[0]
        count = 0.5  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
        n = 5
        import timeit
        t0 = timeit.default_timer()
        violent = myMT5Analy.violenttest_howtochoose(timedf=timedf, matchlist=matchlist, func=func,
                                                     sortbylist=sortbylist, choosebylist=choosebylist,
                                                     resultlist=resultlist, count=count, n=n,
                                                     dropmaxchooseby=True)
        t1 = timeit.default_timer()
        print("\n", '简单循环 multi processing 耗时为：', t1 - t0)  # 17
        # violent 在SciView中查看
        # 保存到xlsx，研究超参数时不要写入
        # 保存后下次分析可以直接从 F:\BaiduNetdiskWorkspace\工作---MT5策略研究\中读取
        violent.to_excel(choosefilename)
        violent = myfile.read_pd(choosefilename, index_col=0)
    return timedf, matchlist, violent

# ---获取[sortby, chooseby, resultlist]的模式收集，返回 modecollection
def get_modecollection(violent):
    modecollection = [] # [sortby, chooseby, resultlist]的收集
    for key in ["mean0.5","mean0.4","mean0.3","mean0.2","mean0.1"]:
        # key="mean0.2"
        # 根据每个键获取最高的模式
        tempviolent = violent.sort_values(by=key, ascending=False)
        tempmax = tempviolent[key].max()
        tempseeds = tempviolent[tempviolent[key]==tempmax]
        tempindexlist = tempseeds.index.tolist()
        print("\n")
        print("2:",key,"排序最前的模式为",tempindexlist)
        # 解析每个模式的参数
        for tempindex in tempindexlist: # tempindex=tempindexlist[0]
            [sortby, chooseby, resultlist] = tempindex.split(".")
            # resultlist = eval(resultlist) if type(resultlist)==str else resultlist
            modecollection.append([sortby, chooseby, resultlist])
    modecollection = pd.DataFrame(modecollection)
    modecollection.drop_duplicates(inplace=True) # 去除重复的
    # 列2字符串要转为list
    modecollection[2] = modecollection[2].apply(lambda x: eval(x) if type(x)==str else x)
    modecollection = modecollection.values.tolist()
    return modecollection

# ---生成EA的参数
def get_EA_parainput(sortby, chooseby, resultlist, count=0.5, n=5):
    # ---训练集根据sortby降序排序后，从中选择count个行，再根据chooseby选择前n个最大值，再根据resultby表示结果.
    sortby = sortby  # "Kelly占用仓位杠杆" "myCriterion" "盈亏比" "平均盈利" "盈利总和" "盈利交易数量"
    count = count  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
    chooseby = chooseby  # "TB"
    n = n
    resultlist = resultlist
    ### 第一次筛选 ###
    totaldf = myMT5Analy.analysis_forward(timedf=timedf, matchlist=matchlist, sortby=sortby, count=count,
                                          chooseby=chooseby, n=n, resultlist=resultlist,
                                          dropmaxchooseby=True, show=False)
    print("3: len(totaldf)=", len(totaldf))

    ### 第二次筛选：根据某种方法选出一个占优的结果 ###
    group = totaldf.groupby(by="tag", axis=0, as_index=False)  # tag为各个分组的标签
    # mypd.groupby_print(group)

    # ---根据训练集选择，测试集反馈。
    lastchoose = group.apply(lambda x: x.iloc[0])  # 选出每个分组的第一个，即sortby排序第一个

    ### 根据out整理出策略每个阶段的外置参数
    parainput = pd.DataFrame([])
    for i in range(len(lastchoose)):
        tag = lastchoose["tag"][i]
        ipass = lastchoose["Pass"][i]
        trainmatch = matchlist[tag][0]  # 这里不需要copy()
        # 下面参数名要根据EA源码的输入变量来整理，trainmatch中策略参数顺序不是对应的。
        trainmatch = trainmatch[["Pass"]+ea_inputparalist]
        trainrow = trainmatch[trainmatch["Pass"] == ipass]
        trainrow["tag"] = tag
        parainput = parainput.append(trainrow, ignore_index=True)
    # ---
    parainput.drop(labels="Pass", axis=1, inplace=True)
    parainput.sort_values(by="tag", inplace=True, ignore_index=True)
    parainput.set_index(keys="tag", drop=True, inplace=True)
    parainput.to_csv(forwardparapath + "\\推进参数.{}.{}.{}.{}.length={}.step={}.csv".
                     format(symbol,myMT5Analy.timeframe_to_ini_affix(timeframe), timeaffix0,
                            timeaffix1, length, step),sep=",")  # 逗号的csv可直接被excel解析。
    print("3: 已保存到", forwardparapath + "\\推进参数.{}.{}.{}.{}.length={}.step={}.csv".
          format(symbol,myMT5Analy.timeframe_to_ini_affix(timeframe), timeaffix0,
                 timeaffix1, length, step))


#%%
for symbol in symbollist:
    # symbol = "EURUSD"
    print("1: symbol=",symbol)
    if symbol in []:
        continue
    # ---
    #%% ###### 获得 violent ######
    length = "%sY"%length_year
    step = "%sM"%step_months # "6M","3M"
    timeaffix0 = myMT5run.change_timestr_format(forward_starttime)
    timeaffix1 = myMT5run.change_timestr_format(forward_endtime)
    starttime = pd.Timestamp(forward_starttime)
    endtime = pd.Timestamp(forward_endtime)

    # 报告目录
    reportfolder = contentfolder + r"\推进分析.{}\推进.{}.{}.{}.{}.length={}.step={}".format(optcriterionaffix, symbol,myMT5Analy.timeframe_to_ini_affix(timeframe), timeaffix0,timeaffix1, length, step)
    # 筛选汇总文件
    choosefilename = contentfolder + r"\推进分析.{}\筛选汇总.{}.{}.{}.{}.length={}.step={}.xlsx".format(optcriterionaffix, symbol, myMT5Analy.timeframe_to_ini_affix(timeframe),timeaffix0, timeaffix1, length, step)
    # 推进分析参数输出目录
    forwardparapath = __mypath__.get_mt5_commonfile_path() + r"\推进分析参数.{}.{}".format(optcriterionaffix, expertfile.rsplit(".",1)[0])

    ### ---获取 timedf, matchlist, violent
    timedf, matchlist, violent = get_timedf_matchlist_and_violent()
    print("1: get_timedf_matchlist_and_violent()完成，准备解析 violent.")

    #%% ###### 解析下violent ######
    # ---获取[sortby, chooseby, resultlist]的模式收集
    modecollection = get_modecollection(violent=violent)

    # ---每个模式都进行推进回测
    for sortby, chooseby, resultlist in modecollection:
        print("3: 当前模式的参数为：sortby={}, chooseby={}, resultlist={}".format(sortby,chooseby,resultlist))
        #%% ### 生成EA的参数 ###
        get_EA_parainput(sortby, chooseby, resultlist, count=0.5, n=5)

        ### 回测 ###
        # symbol='EURCHF'; timeframe="TIMEFRAME_M30"
        # sortby,chooseby,resultlist = ('亏损总和', '盈利总和', ['TB', '净利润'])
        # tf_affix="M30";starttime="2015.07.01"
        tf_affix = myMT5run.timeframe_to_ini_affix(timeframe)  # 时间框词缀

        # EA的位置
        bt_expertname = bt_experfolder + "\\" + bt_expertfile
        print("3. EA=",bt_expertname)

        # xml格式优化报告的目录
        bt_reportfolder = bt_folder + r"\{}.{}".format(symbol, tf_affix)
        myfile.makedirs(bt_reportfolder, True)

        # 输出文档不能有%符号
        if "%" in sortby:
            sortby = sortby.replace("%","")
        if "%" in chooseby:
            chooseby = chooseby.replace("%","")
        bt_reportfile = bt_reportfolder + "\\{}.{}.{}.xml".format(sortby,chooseby,resultlist)
        print("3. reportfile=", bt_reportfile)

        # 回测设置
        bt_forwardmode = 0  # 向前检测 (0 "No", 1 "1/2", 2 "1/3", 3 "1/4", 4 "Custom")
        bt_model = 1  # 0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点"
        bt_optimization = 0  # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
        print("3: 开始MT5回测EA：sortby={}, chooseby={}, resultlist={}".format(sortby,chooseby,resultlist))

        #%% # ---
        myMT5run.__init__()
        myMT5run.config_Tester(bt_expertname, symbol, timeframe, fromdate=bt_starttime,
                               todate=bt_endtime,forwardmode=bt_forwardmode, forwarddate=None,
                               delays=0, model=bt_model, optimization=bt_optimization,
                               optcriterion=6, reportfile=bt_reportfile)
        common_set()
        strategy_set()
        # ---检查参数输入是否匹配优化的模式，且写出配置结果。
        myMT5run.check_inputs_and_write()
        myMT5run.run_MT5()



