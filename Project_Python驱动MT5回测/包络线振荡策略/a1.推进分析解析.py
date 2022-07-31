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
myMT5Analy = MyMT5Analysis.MyClass_ForwardAnalysis() # MT5分析类
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
import warnings
warnings.filterwarnings('ignore')

symbol = "EURUSD"
timeframe = "TIMEFRAME_M30"
length = "2Y"
step = "6M"

reportfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\6.包络线振荡策略\推进.{}.{}.length={}.step={}".format(symbol,myMT5Analy.timeframe_to_ini_affix(timeframe),length,step)
expertfile = "a1.包络线振荡策略(1).ex5"

# 推进测试的起止时间
starttime = pd.Timestamp("2015.01.01") # ************
endtime = pd.Timestamp("2022.07.01") # ************
step_months = 6 # 推进步长，单位月 # ************
length_year = 2 # 样本总时间包括训练集和测试集 # ************
timedf = myMT5Analy.get_everystep_time(starttime, endtime, step_months=step_months, length_year=length_year)

# ---批量读取推进优化的报告(csv比xlsx速度快)，保存到matchlist中 [[0,1],[0,1]]--- 0 trainmatch, 1 testmatch.
matchlist = [] # [[0,1]]
for i, row in timedf.iterrows():
    # 时间参数必须转成"%Y.%m.%d"字符串
    fromdate = row["from"]
    forwarddate = row["forward"]
    todate = row["to"]
    # ---xlsx格式优化报告
    tf_affix = myMT5Analy.timeframe_to_ini_affix(timeframe)
    t0 = myMT5Analy.change_timestr_format(fromdate)
    t1 = myMT5Analy.change_timestr_format(forwarddate)
    t2 = myMT5Analy.change_timestr_format(todate)
    csvfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.csv".format(expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2)
    print("读取 csvfile=", csvfile)
    trainmatch, testmatch = myMT5Analy.read_forward_opt_csv(filepath=csvfile)
    matchlist.append([trainmatch, testmatch])

#%% ### 展示相关性 ###
for i in range(len(matchlist)):  # i=0
    trainmatch = matchlist[i][0].copy()
    testmatch = matchlist[i][1].copy()
    # 显示训练集测试集的 spearman pearson 相关性.
    myMT5Analy.show_traintest_spearcorr(trainmatch, testmatch)

#%% ### 一次筛选 ###
# "净利润"	"总交易" "多头交易" "空头交易" "%总胜率" "%多胜率" "%空胜率" "TB" "Sharpe_MT5"
# "SQN_MT5_No" "Sharpe_Balance"	"SQN_Balance" "SQN_Balance_No" "Sharpe_Price" "SQN_Price" "SQN_Price_No"
# "平均盈利" "平均亏损" "盈亏比" "利润因子" "恢复因子" "期望利润" "Kelly占用仓位杠杆" "Kelly止损仓位比率"
# "Vince止损仓位比率" "最小净值" "%最大相对回撤比" "最大相对回撤比占额" "%最小保证金" "最大绝对回撤值"
# "%最大绝对回撤值占比" "回归系数" "回归截距" "LRCorrelation" "LRStandardError" "盈利总和" "亏损总和"
# "AHPR" "GHPR" "%无仓GHPR_Profit" "%无仓GHPR_Loss" "盈利交易数量" "亏损交易数量" "(int)最长获利序列"
# "最长获利序列额($)" "(int)最长亏损序列" "最长亏损序列额($)" "最大的连利($)" "(int)最大的连利序列数"
# "最大的连亏($)" "(int)最大的连亏序列数" "平均连胜序列" "平均连亏序列" "获利交易中的最大值"
# "亏损交易中的最大值"

# ---训练集根据sortby降序排序后，从中选择count个行，再根据chooseby选择前n个最大值，再根据resultby表示结果。
sortby = "Vince止损仓位比率" # "myCriterion" "盈亏比" "平均盈利" "盈利总和" "盈利交易数量"
count = 0.5  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
chooseby = "TB"
n = 5
resultby = "净利润"

totaldf = myMT5Analy.analysis_forward(timedf=timedf, matchlist=matchlist, sortby=sortby, count=count, chooseby=chooseby, n=n, resultby=resultby, dropmaxchooseby=True, show=False)

#%% ### 二次筛选：是否存在某种方法选出一个占优的结果 ###
group = totaldf.groupby(by="tag", axis=0, as_index=False) # tag为各个分组的标签
# mypd.groupby_print(group)

group.apply(lambda x: x.iloc[0]) # 选出每个分组的第一个，即sortby排序第一个
group.apply(lambda x: x.iloc[x[chooseby].argmax()]) # 选出每个分组chooseby最大的一个
group.apply(lambda x: x.iloc[x[resultby].argmax()]) # 选出每个分组resultby最大的一个




#%% ### 暴力测试下怎么筛选结果较好 ###
sortbylist = trainmatch.loc[:, "净利润":"亏损交易中的最大值"].columns
choosebylist = ["TB"]
resultbylist = ["净利润"]
func = lambda x: x.iloc[0] # 二次筛选的模式。选出每个分组的第一个，即sortby排序第一个
count = 0.5  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
n = 5

violent =  myMT5Analy.violenttest_howtochoose(timedf=timedf, matchlist=matchlist, func=func,
                                              sortbylist=sortbylist, choosebylist=choosebylist,
                                              resultbylist=resultbylist,count=count, n=n,
                                              dropmaxchooseby=True)
violent

