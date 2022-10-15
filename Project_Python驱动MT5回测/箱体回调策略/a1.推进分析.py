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


#%%
'''
策略优化结果使用遗传算法！
'''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()



#%% ###### 外部参数 ######


# ====== 策略参数 ======
# ------通用分析套件参数------
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

# ------策略参数------
def strategy_set():
    myMT5run.input_set("MaxBoxPeriod", "53||50||1||70||Y")  # 大箱体周期************
    myMT5run.input_set("OsciBoxPeriod", "9||5||1||15||Y") # 小箱体周期************
    myMT5run.input_set("K_TrendBuyU", "0.70||1.0||-0.02||0.7||Y") # 做多：大箱体K值上界************
    myMT5run.input_set("K_TrendBuyD", "0.54||0.4||0.02||0.6||Y") # 做多：大箱体K值下界************
    myMT5run.input_set("TrendGap", "500||0||100||1000||Y") # 做多：close[1]高于趋势箱体的最低价TrendGap点************
    myMT5run.input_set("K_OsciBuyLevel", "0.15||0.05||0.05||0.2||Y")  # 做多：价格要在低位置才可以，K < buylevel****
    myMT5run.input_set("OsciGap", "90||80||10||150||Y")  # 做多：振荡箱体2的最高价与close[1]相差超过OSCGap点*****
    myMT5run.input_set("CloseBuyLevel", "0.90||0.8||0.02||0.9||Y") # 多平：K大于规定的DirectCloseLevel*****
    myMT5run.input_set("PriceGap", "999||200||1||2000||N") # 1.做多：图表上bid价格<上根bar收盘价+20点；
    myMT5run.input_set("MaxSpread", "999||200||1||2000||N") # 2.做多：当前点差不超过设置的最大点差Spread 30；
    myMT5run.input_set("SL_Min", "0||0||1||10||N") # 止损的限定范围 250
    myMT5run.input_set("SL_Max", "1500||500||100||2000||N") # 止损的限定范围 15000
    myMT5run.input_set("AvgLotsToPPoint_L", "0||0||1||10||N") # 平均1手盈利达到此平仓 200
    myMT5run.input_set("AvgLotsToPPoint_R", "0||0||1||10||N") # 平均1手盈利达到此平仓 99999



#%% ###### a1.三均线顺势拉回策略 策略优化 ######
# 推进测试的起止时间
starttime = "2015.01.01" # ************
endtime = "2022.07.1" # ************
step_months = 6 # 6, 3 # 推进步长，单位月 # ************
length_year = 2 # 2, 1 # 样本总时间包括训练集和测试集 # ************

timeaffix0 = myMT5run.change_timestr_format(starttime)
timeaffix1 = myMT5run.change_timestr_format(endtime)
starttime = pd.Timestamp(starttime)
endtime = pd.Timestamp(endtime)

timedf = myMT5run.get_everystep_time(starttime, endtime, step_months=step_months, length_year=length_year)

symbollist = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "XAUUSD", "XAGUSD", "AUDJPY","CHFJPY","EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","GBPAUD","GBPCAD","GBPCHF","GBPJPY","NZDJPY"]
symbollist = ["EURUSD"]

#---测试下哪个优化标准更能找到好策略
# 0 -- Balance max, 1 -- Profit Factor max, 2 -- Expected Payoff max, 3 -- Drawdown min, 4 -- Recovery Factor max, 5 -- Sharpe Ratio max, 6 -- Custom max, 7 -- Complex Criterion max
def run(criterionindex=0):
    for symbol in symbollist:
        if symbol in []: # symbol = "EURUSD"
            continue

        timeframe = "TIMEFRAME_M15" # ************
        length = "%sY"%length_year
        step = "%sM"%step_months

        experfolder = "My_Experts\\Strategy深度研究\\箱体回调策略"

        optcriterionaffix=myMT5run.get_optcriterion_affix(optcriterion=criterionindex)
        reportfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\7.箱体回调策略\推进分析.{}\推进.{}.{}.{}.{}.length={}.step={}".format(optcriterionaffix, symbol,myMT5run.timeframe_to_ini_affix(timeframe),timeaffix0,timeaffix1,length,step) # 以 "推进.EURUSD.M30.2015-01-01.2022-07-01.length=2Y.step=6M" 格式

        expertfile = "a1.箱体回调策略.ex5" # ************
        expertname = experfolder + "\\" + expertfile

        forwardmode = 4 # 向前检测 (0 "No", 1 "1/2", 2 "1/3", 3 "1/4", 4 "Custom")
        model = 1 # 0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点"
        optimization = 2 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
        optcriterion = criterionindex # 0 -- Balance max, 1 -- Profit Factor max, 2 -- Expected Payoff max, 3 -- Drawdown min, 4 -- Recovery Factor max, 5 -- Sharpe Ratio max, 6 -- Custom max, 7 -- Complex Criterion max


        for i, row in timedf.iterrows():
            # 时间参数必须转成"%Y.%m.%d"字符串
            fromdate = row["from"]
            forwarddate = row["forward"]
            todate = row["to"]
            print("======开始测试：fromdate={}, forwarddate={}, todate={}".format(fromdate,forwarddate,todate))

            # ---xml格式优化报告的目录
            tf_affix = myMT5run.timeframe_to_ini_affix(timeframe)
            t0 = myMT5run.change_timestr_format(fromdate)
            t1 = myMT5run.change_timestr_format(forwarddate)
            t2 = myMT5run.change_timestr_format(todate)
            reportfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.Crit={}.xml".format(expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2, optcriterion)
            print("reportfile=",reportfile)

        #%%
            myMT5run.__init__()
            myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                                   forwardmode=forwardmode, forwarddate=forwarddate,
                                   delays=0, model=model, optimization=optimization,
                                   optcriterion=optcriterion, reportfile=reportfile)
            common_set()
            strategy_set()
            # ---检查参数输入是否匹配优化的模式，且写出配置结果。
            myMT5run.check_inputs_and_write()
            myMT5run.run_MT5()


#%% 测试下哪个优化标准更能找到好策略
run(0)
run(6)
run(7)



