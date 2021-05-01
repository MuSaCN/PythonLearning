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
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

# 说明：
'''
仓位管理逻辑：
模式1：lots_risk_percent() (保证金止损仓位)固定比例仓位。
    ·对于止损点，分别考虑开仓的止损点 "StopLossPoint"、基仓回测下最大亏损 "worst_point" 对应的止损点。
    ·由于涉及固定比例，所以最优仓位理论作为关键比例进行考虑。
    ·考虑破产概率。
    ·比例优化的范围可以外部指定
模式2：lots_FixedIncrement_*() 固定增长法计算仓位。
    ·分别考虑 "分割资金SplitFund" "拆分公式SplitFormula" 两种不同的方式。
    ·涉及初期开仓，所以最优仓位理论仅能用于初期开仓。
    ·比例不固定，无法考虑破产概率。
    ·delta资金的优化针对不同的品种而不同，内部自动判定，不做外部指定。
    ·关键的delta值为"基仓回测系统"中：历史最大回撤数值的一半 或者 最大亏损额的倍数。
模式3：ATR止损的 lots_risk_percent() (保证金止损仓位)固定比例仓位。
    ·可以优化的变量有：ATR的周期、ATR倍数(默认设为1，不优化)、资金百分比(考虑几个特殊值)
    ·ATR的周期优化范围可以外部指定

所有的模式都有：
    ·以 收益率/最大回撤 ret_maxDD 的1次卡尔曼过滤作为标的，进行极值判定。结果作为关键比例。
    ·不同的方法，极值判定的order不一样。
    ·对于关键的结果，进行蒙特卡罗模拟测试 最大回撤分布、收益率分布、盈亏比分布。


'''

#%%
import warnings
warnings.filterwarnings('ignore')
# 通用参数
file = __mypath__.get_desktop_path() + "\\ATR_test.xlsx" # ATR_test test
init_deposit = 5000
simucount = 100 # 模拟次数
direct = "BuyOnly" # 考虑的交易方向 "BuyOnly" "SellOnly"
pic_folder = __mypath__.get_desktop_path() + "\\资金管理\\"

# ---仓位百分比法专用参数
used_percent_list = [(i + 1) / 100 for i in range(100)]  # 仓位百分比0.001精度
order_lots_risk_percent = 100 # 用于仓位百分比法判断极值

# ---固定增长量法专用参数
init_percent = 0.1 # 0.1, "f_kelly", "f_twr", 利用多核来执行多个
order_fixed_increment = 50  # 用于固定增长量判断极值

# ---ATR变动持仓
used_percent_atr = "f_twr" # 0.1, "f_kelly", "f_twr", 利用多核来执行多个
order_atr = 100  # 用于判断极值
atr_multiple = 1.0 # ATR点数的倍数
atr_period_list = [i for i in range(1, 150, 1)]


#%% 以 lots_risk_percent() 的 "StopLossPoint" 分析
from MyPackage.MyProjects.资金管理分析.Lots_Risk_Percent import Mode_Lots_Rist_Percent
mode_lots_rist_percent0 = Mode_Lots_Rist_Percent()
mode_lots_rist_percent0.file = file
mode_lots_rist_percent0.init_deposit = init_deposit
mode_lots_rist_percent0.stoplosspoint = "StopLossPoint"
mode_lots_rist_percent0.used_percent_list = used_percent_list
mode_lots_rist_percent0.order = order_lots_risk_percent
mode_lots_rist_percent0.simucount = simucount
mode_lots_rist_percent0.direct = direct
mode_lots_rist_percent0.pic_folder = pic_folder
mode_lots_rist_percent0.run()


#%% 以 lots_risk_percent() 的 "worst_point" 分析
mode_lots_rist_percent1 = Mode_Lots_Rist_Percent()
mode_lots_rist_percent1.file = file
mode_lots_rist_percent1.init_deposit = init_deposit
mode_lots_rist_percent1.stoplosspoint = "worst_point"
mode_lots_rist_percent1.used_percent_list = used_percent_list
mode_lots_rist_percent1.order = order_lots_risk_percent
mode_lots_rist_percent1.simucount = simucount
mode_lots_rist_percent1.direct = direct
mode_lots_rist_percent1.pic_folder = pic_folder
mode_lots_rist_percent1.run()


#%% 以 lots_FixedIncrement_SplitFund() 分析
from MyPackage.MyProjects.资金管理分析.Fixed_Increment import Mode_Fixed_Increment
mode_fixed_increment0 = Mode_Fixed_Increment()
mode_fixed_increment0.file = file
mode_fixed_increment0.init_deposit = init_deposit
mode_fixed_increment0.init_percent = init_percent # 0.1, f_kelly, f_twr, 利用多核来执行多个
mode_fixed_increment0.order = order_fixed_increment  # 用于判断极值
mode_fixed_increment0.simucount = simucount  # 模拟次数
mode_fixed_increment0.funcmode = "SplitFund" # "SplitFund"拆分资金法 / "SplitFormula"拆分公式法
mode_fixed_increment0.direct = direct
mode_fixed_increment0.pic_folder = pic_folder
mode_fixed_increment0.run()


#%% 以 lots_FixedIncrement_SplitFormula() 分析
mode_fixed_increment1 = Mode_Fixed_Increment()
mode_fixed_increment1.file = file
mode_fixed_increment1.init_deposit = init_deposit
mode_fixed_increment1.init_percent = init_percent # 0.1, f_kelly, f_twr, 利用多核来执行多个
mode_fixed_increment1.order = order_fixed_increment  # 用于判断极值
mode_fixed_increment1.simucount = simucount  # 模拟次数
mode_fixed_increment1.funcmode = "SplitFormula" # "SplitFund"拆分资金法 / "SplitFormula"拆分公式法
mode_fixed_increment1.direct = direct
mode_fixed_increment1.pic_folder = pic_folder
mode_fixed_increment1.run()


#%% 以 ATR止损点的 lots_risk_percent() 分析
from MyPackage.MyProjects.资金管理分析.ATR_Lots import Mode_ATR_Lots
mode_atr_lots = Mode_ATR_Lots()
mode_atr_lots.file = file
mode_atr_lots.init_deposit = init_deposit
mode_atr_lots.used_percent = used_percent_atr # best_f.f_kelly best_f.f_twr
mode_atr_lots.order = order_atr  # 用于判断极值
mode_atr_lots.simucount = simucount  # 模拟次数
mode_atr_lots.multiple = atr_multiple # ATR点数的倍数
mode_atr_lots.atr_period_list = atr_period_list
mode_atr_lots.direct = direct
mode_atr_lots.pic_folder = pic_folder
mode_atr_lots.run()


#%%
# mode_lots_rist_percent0.run()
# mode_lots_rist_percent1.run()
# mode_fixed_increment0.run()
# mode_fixed_increment1.run()
# mode_atr_lots.run()
