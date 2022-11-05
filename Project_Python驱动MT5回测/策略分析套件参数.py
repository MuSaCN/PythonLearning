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

# %%
# ------通用分析套件参数(版本2022.11.05)------
# 使用时要修改，请标注 *******
def common_set():
    myMT5run.input_set("FrameMode", "2") # 0-None 1-BTMoreResult 2-OptResult 3-ToDesk 4-GUI
    # ; ======(通用)0.用于分析======
    myMT5run.input_set("Inp_Signal_Shift", "1") # >=1为信号确认，且每bar运行一次，=0为实时。
    myMT5run.input_set("Inp_CustomMode", "0") # 0-TB
    myMT5run.input_set("Inp_IsBackTestCSV", "false") # 单次回测是否输出csv结果
    myMT5run.input_set("Inp_CommissionMode", "0") # CommissionMode佣金模式
    myMT5run.input_set("Inp_CommissionValue", "0.0") # CommissionValue佣金模式对应的值
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
    # ------1.4 交易成本限制------
    myMT5run.input_set("Inp_AvgCostTickCount", "0") # 最近n个tick平均点差+佣金点数
    myMT5run.input_set("Inp_Is_CurCostLimit", "false") # 是否启用当前点差成本
    myMT5run.input_set("Inp_CostPointLimit", "30") # 限制成本点最大的点数
    # ------1.5 交易时间间隔(自然时间)限制------
    myMT5run.input_set("Inp_IntervalMultiply", "0.0||0.0||0.000000||0.000000||N")
    myMT5run.input_set("Inp_IntervalBarTF", "0||0||0||49153||N")
    # ; ------2.出场模式------
    myMT5run.input_set("Inp_Is_CloseAttachPend", "true")
    # ------2.1 常规出场模式------
    myMT5run.input_set("Inp_Is_SigToCloseInver", "true") # true信号平反向仓，false则不是。sig=4不适合.
    myMT5run.input_set("Inp_Is_PendToCloseInver", "true") # true挂单成交平反向仓，false则不是。
    # ------2.2 订单固定Bar出场模式------
    myMT5run.input_set("Inp_FixedHolding", "0||0||1||10||N") # 0表示不是固定持仓模式，>0表示固定周期持仓。
    myMT5run.input_set("Inp_FixedHoldTF", "0") # FixedHolding的时间框
    myMT5run.input_set("Inp_FixedHoldPPoint", "0||0||1||10||N")  #
    # ------2.3 资金出场模式------
    myMT5run.input_set("Inp_MoneyToCloseMode", "0")
    myMT5run.input_set("Inp_AvgLotsToProfit", "0.0||0.0||0.000000||0.000000||N")
    myMT5run.input_set("Inp_AvgLotsToPPoint", "0||0||1||10||N")
    myMT5run.input_set("Inp_CumLotsToProfit", "0.0||0.0||0.000000||0.000000||N")
    myMT5run.input_set("Inp_CumLotsToPercBalance", "0.0||0.0||0.000000||0.000000||N")
    # ------2.4 资金移动止损------
    myMT5run.input_set("Inp_MoneyTrailMode", "0")
    myMT5run.input_set("Inp_MoneyTrailStart", "0.0||0.0||0.000000||0.000000||N")
    myMT5run.input_set("Inp_MoneyTrailFall", "0.0||0.0||0.000000||0.000000||N")
    # ------2.5 极端净利润平极端单------
    myMT5run.input_set("Inp_ExtNetProfitMode", "0")
    myMT5run.input_set("Inp_ExtNetProfitN", "1||0||1||10||N")
    myMT5run.input_set("Inp_ExtNetProfitM", "2||0||1||10||N")
    myMT5run.input_set("Inp_ExtNetProfit", "0.0||0.0||0.000000||0.000000||N")
    # ; ------3.信号过滤(范围和方向)------
    # ------3.1 范围过滤------
    myMT5run.input_set("Inp_FilterMode", "0||0||0||2||N") # 0-NoFilter, 1-Range, 2-TwoSide
    myMT5run.input_set("Inp_FilterIndiName", "") # 过滤指标名称
    myMT5run.input_set("Inp_FilterIndiTF", "_Period") # 过滤指标时间框字符串
    myMT5run.input_set("Inp_FilterIndiPara0", "0.0") # 过滤指标首个参数
    myMT5run.input_set("Inp_FilterLeftValue", "0.0") # 过滤指标左侧的值
    myMT5run.input_set("Inp_FilterRightValue", "0.0") # 过滤指标右侧的值
    # ------3.2 方向过滤------
    myMT5run.input_set("Inp_DirectMode", "0||2||0||4||N") # 2-TwoSide, 3-Direct1, 4-Direct2
    myMT5run.input_set("Inp_DirectIndiName", "") # 方向指标名称
    myMT5run.input_set("Inp_DirectIndiTF", "_Period") # 方向指标时间框字符串
    myMT5run.input_set("Inp_DirectIndiPara0", "0.0||0.0||0.000000||0.000000||N") # 方向指标首个参数
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
    myMT5run.input_set("Inp_BreakEven_ExtraCostPoint", "0") # 盈亏平衡的成本点，比如Commission占用的点数。
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
    myMT5run.input_set("Inp_HourFilter", "") # 不允许交易，过滤的小时
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
    myMT5run.input_set("Inp_PnL_Point", "0||0||1||10||N") # 盈亏加仓的点数，0关闭，>0盈加，<0亏加。
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


