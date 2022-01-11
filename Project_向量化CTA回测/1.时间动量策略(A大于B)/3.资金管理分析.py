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
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

# 把MT5单独回测的结果都放在一个目录里，会批量依次执行资金管理分析。仓位管理逻辑说明：
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

输出的表格：
    ·进行了原顺序的最大回撤过滤，要求 maxDD > -0.5。

'''

#%%
import warnings
warnings.filterwarnings('ignore')

# ---MT5回测文档所在的文件夹，用于批量分析
folder = r"F:\工作(同步)\工作---资金管理\1.简单的动量策略\EURUSD.D1"

# ---通用参数
init_deposit = 5000
simucount = 5000  # 5000 模拟次数
direct = "BuyOnly"  # 考虑的交易方向 "BuyOnly" "SellOnly"

# ---仓位百分比法专用参数
used_percent_list = [(i + 1) / 1000 for i in range(1000)]  # 1000 仓位百分比0.001精度
order_lots_risk_percent = 100  # 用于仓位百分比法判断极值

# ---固定增长量法专用参数
# init_percent = 0.1 # 0.1, "f_kelly", "f_twr", 利用多核来执行多个
# funcmode = "SplitFund"拆分资金法 / "SplitFormula"拆分公式法, 利用多核来执行多个
order_fixed_increment = 50  # 用于固定增长量判断极值

# ---ATR变动持仓专用参数
# used_percent_atr = "f_twr" # 0.1, "f_kelly", "f_twr", 利用多核来执行多个
order_atr = 100  # 用于判断极值
atr_multiple = 1.0  # ATR点数的倍数
atr_period_list = [i for i in range(1, 150, 1)] # [i for i in range(1, 150, 1)]

#%% 加载批量资金管理分析类
from MyPackage.MyProjects.资金管理分析.Batch_Analysis import Lots_Batch_Analysis
listdir = __mypath__.listdir(folder)
listdir = [i for i in listdir if ".xlsx" in i] # 排除非 xlxs 文件
batch_analysis_list = [] # 存放 批量分析类
for name in listdir: # name = listdir[0]
    file = folder + "\\" + name
    batch_analysis = Lots_Batch_Analysis(file=file)
    # ---外部参数赋值
    # 通用参数
    batch_analysis.file = file
    batch_analysis.init_deposit = init_deposit
    batch_analysis.simucount = simucount
    batch_analysis.direct = direct
    # 仓位百分比法专用参数
    batch_analysis.used_percent_list = used_percent_list
    batch_analysis.order_lots_risk_percent = order_lots_risk_percent
    # 固定增长量法专用参数
    batch_analysis.order_fixed_increment = order_fixed_increment
    # ATR变动持仓专用参数
    batch_analysis.order_atr = order_atr
    batch_analysis.atr_multiple = atr_multiple
    batch_analysis.atr_period_list = atr_period_list
    # ---配置各个分析类(必须要执行)
    batch_analysis.config_analysis()
    # 存储到list中
    batch_analysis_list.append(batch_analysis)

#%%
# 后台输出，不占pycharm内存
myDefault.set_backend_default("agg")
if __name__ == '__main__':
    # ---生成多核参数para
    # 外部指定的参数，但总参数不限于此，内部会自动分析添加新参数。
    sl_point_list = ["StopLossPoint", "worst_point"]  # --> risk_percent
    funcmode_list = ["SplitFund", "SplitFormula"]  # --> FixedIncrement
    init_percent_list = [0.1, "f_kelly", "f_twr"]  # --> FixedIncrement
    risk_percent_list = [0.1, "f_kelly", "f_twr"]  # --> ATR_risk_percent
    para0 = [("risk_percent", sl) for sl in sl_point_list]
    para1 = [("FixedIncrement", func, init) for func in funcmode_list for init in init_percent_list]
    para2 = [("ATR_risk_percent", used) for used in risk_percent_list]
    para_list = para0 + para1 + para2
    # 每个策略文档，依次进行多核执行 # batch_analysis # i=0
    # ---进度
    finished_basename = []  # 完成的文件名称，不是完整路径。
    want_basename = [] # 需要进行的文件名称，与上方逻辑相反，不用要清空
    for i in range(len(batch_analysis_list)):
        basename = __mypath__.basename(batch_analysis_list[i].file)
        if basename in finished_basename:
            print("finished: ", basename)
            continue
        if len(want_basename) > 0 and basename not in want_basename:
            print("not want: ", basename)
            continue
        # ---
        batch_analysis_list[i].multi_process(para_list)
        finished_basename.append(basename)
        print("finished_basename: ", finished_basename)
    # ---
    print("finished all!!!")
    __mypath__.open_folder(folder)




