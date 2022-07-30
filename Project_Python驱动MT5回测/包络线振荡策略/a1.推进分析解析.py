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
import warnings
warnings.filterwarnings('ignore')


symbol = "EURUSD"
timeframe = "TIMEFRAME_M30"
length = "2Y"
step = "6M"

reportfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\6.包络线振荡策略\推进.{}.{}.length={}.step={}".format(symbol,myMT5run.timeframe_to_ini_affix(timeframe),length,step)
expertfile = "a1.包络线振荡策略(1).ex5"

# 推进测试的起止时间
starttime = pd.Timestamp("2015.01.01") # ************
endtime = pd.Timestamp("2022.07.1") # ************
step_months = 6 # 推进步长，单位月 # ************
length_year = 2 # 样本总时间包括训练集和测试集 # ************
timedf = myMT5run.get_everystep_time(starttime, endtime, step_months=step_months, length_year=length_year)

# ---批量自动读取(csv比xlsx速度快)
match = []
for i, row in timedf.iterrows():
    # 时间参数必须转成"%Y.%m.%d"字符串
    fromdate = row["from"]
    forwarddate = row["forward"]
    todate = row["to"]

    # ---xlsx格式优化报告
    tf_affix = myMT5run.timeframe_to_ini_affix(timeframe)
    t0 = myMT5run.change_timestr_format(fromdate)
    t1 = myMT5run.change_timestr_format(forwarddate)
    t2 = myMT5run.change_timestr_format(todate)
    csvfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.csv".format(expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2)
    print("读取 csvfile=",csvfile)
    trainmatch, testmatch = myMT5Report.read_forward_opt_csv(filepath=csvfile)
    match.append([trainmatch, testmatch])


#%%
# ---写按某种模式下推进快速分析！！！
for i in range(len(match)): # i=0
    trainmatch = match[i][0]
    testmatch = match[i][1]

    # ---设置自定义准则
    mycriterion = "myCriterion"
    trainmatch.insert(loc=2, column=mycriterion, value=None)
    trainmatch[mycriterion] = np.power(trainmatch["总交易"],0.5)*trainmatch["盈亏比"]*trainmatch["%总胜率"]*np.power(trainmatch["盈利总和"],0.5)/np.power(np.abs(trainmatch["亏损总和"]),0.5) * np.power(trainmatch["盈利交易数量"], 0.5)
    testmatch.insert(loc=2, column=mycriterion, value=None)
    testmatch[mycriterion] = np.power(testmatch["总交易"],0.5)*testmatch["盈亏比"]*testmatch["%总胜率"]*np.power(testmatch["盈利总和"],0.5)/np.power(np.abs(testmatch["亏损总和"]),0.5) * np.power(testmatch["盈利交易数量"], 0.5)


    #%% 自动选择
    # 训练集根据sortby降序排序后，从中选择count个行，再根据chooseby选择前5个最大值，返回 trainchoose。
    sortby = mycriterion # "mycriterion" "盈亏比" "平均盈利" "盈利总和" "盈利交易数量"
    count = 0.5  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
    chooseby = "TB"
    n = 5
    resultby = "净利润"

    # 选择的结果不一定是5个中最大的tb，要看看最大的tb是否为全局最大的tb。然后再判断。根据自己的标准可以考虑第一个。
    trainmatch[chooseby].max()

    # 简单的选择的结果
    trainchoose = myMT5Report.choose_opttrain_by2index(trainmatch=trainmatch, testmatch=None, sortby = sortby, count=count, chooseby = chooseby, n = n, resultby=resultby)
    trainchoose

    # 选择的结果在测试集中所占的百分比位置
    trainchoose = myMT5Report.choose_opttrain_by2index(trainmatch=trainmatch, testmatch=testmatch, sortby = sortby, count=count, chooseby = chooseby, n = n, resultby=resultby)
    trainchoose

    # 看看选择的结果中是否有最大的tb。
    trainchoose[trainchoose["TB"] == trainmatch[chooseby].max()]

    # 显示训练集测试集的 spearman pearson 相关性.
    myMT5Report.show_traintest_spearcorr(trainmatch, testmatch)



#%% 记录下高相关性的词缀
'''
a1.包络线振荡策略(1).EURUSD.M30.2015-01-01.2016-07-01.2017-01-01.xlsx
总交易  spearcorr = 0.9559208932274254  pearcorr = 0.9682589002033708
多头交易  spearcorr = 0.940742610290166  pearcorr = 0.9487822612339877
空头交易  spearcorr = 0.9312374081558316  pearcorr = 0.9595628548607197
平均盈利  spearcorr = 0.7214713311882196  pearcorr = 0.6807107044955332
盈利总和  spearcorr = 0.7840437427212836  pearcorr = 0.8734806686006242
亏损总和  spearcorr = 0.7850453261050849  pearcorr = 0.9158680612828269
%无仓GHPR_Profit  spearcorr = 0.7149091160180842  pearcorr = 0.6735018105166403
盈利交易数量  spearcorr = 0.8945392183992829  pearcorr = 0.917201846947379
亏损交易数量  spearcorr = 0.8637352551832493  pearcorr = 0.9472228325804275

a1.包络线振荡策略(1).EURUSD.M30.2015-07-01.2017-01-01.2017-07-01.xlsx
总交易  spearcorr = 0.7724663833720277  pearcorr = 0.9024321547785981
多头交易  spearcorr = 0.7157690648761454  pearcorr = 0.8168961857740761
空头交易  spearcorr = 0.6595577585017396  pearcorr = 0.8137632737221938
Sharpe_Balance  spearcorr = 0.594905009488733  pearcorr = 0.5758782990346901
Sharpe_Price  spearcorr = 0.5787045425595557  pearcorr = 0.5628518026045123
利润因子  spearcorr = 0.5838191977935663  pearcorr = 0.5678291381124563
期望利润  spearcorr = 0.6953525316135941  pearcorr = 0.6772716600094094
Kelly止损仓位比率  spearcorr = 0.5727524311865076  pearcorr = 0.5762404909525187
%最大相对回撤比  spearcorr = 0.5482841536282168  pearcorr = 0.3767667118415165
最大相对回撤比占额  spearcorr = 0.5362210598253685  pearcorr = 0.3755968036604421
最大绝对回撤值  spearcorr = 0.5362210598253685  pearcorr = 0.3755968036604421
%最大绝对回撤值占比  spearcorr = 0.5482841536282168  pearcorr = 0.3767667118415165
回归系数  spearcorr = 0.7363220951315584  pearcorr = 0.7093854543334555
亏损总和  spearcorr = 0.6662482070989251  pearcorr = 0.7639717223076998
盈利交易数量  spearcorr = 0.6601800364893541  pearcorr = 0.8562213166180398
亏损交易数量  spearcorr = 0.7333172658836594  pearcorr = 0.8715237763860191

a1.包络线振荡策略(1).EURUSD.M30.2016-01-01.2017-07-01.2018-01-01.xlsx
总交易  spearcorr = 0.9288072833133255  pearcorr = 0.9506152789312625
多头交易  spearcorr = 0.7975026200428488  pearcorr = 0.8574021091857378
空头交易  spearcorr = 0.8706077849190784  pearcorr = 0.9133565368528833
盈利总和  spearcorr = 0.7343052820266709  pearcorr = 0.8024419146326569
亏损总和  spearcorr = 0.8096650359821721  pearcorr = 0.7860023230723364
盈利交易数量  spearcorr = 0.8683669476200266  pearcorr = 0.9101292841040685
亏损交易数量  spearcorr = 0.7963955057479855  pearcorr = 0.8257605886808501




'''

