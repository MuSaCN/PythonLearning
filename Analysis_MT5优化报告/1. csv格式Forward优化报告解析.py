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

filepath = __mypath__.get_desktop_path() + "\\" + "a1.包络线振荡策略(7).EURUSD.M30.csv"

# 匹配后的训练集和测试集(csv为完整优化或遗传算法).
trainmatch, testmatch = myMT5Report.read_forward_opt_csv(filepath=filepath)

# 设置自定义准则
mycriterion = "myCriterion"
trainmatch.insert(loc=2, column=mycriterion, value=None)
trainmatch[mycriterion] = np.power(trainmatch["总交易"],0.5)*trainmatch["盈亏比"]*trainmatch["%总胜率"]*np.power(trainmatch["盈利总和"],0.5)/np.power(np.abs(trainmatch["亏损总和"]),0.5) * np.power(trainmatch["盈利交易数量"], 0.5)
testmatch.insert(loc=2, column=mycriterion, value=None)
testmatch[mycriterion] = np.power(testmatch["总交易"],0.5)*testmatch["盈亏比"]*testmatch["%总胜率"]*np.power(testmatch["盈利总和"],0.5)/np.power(np.abs(testmatch["亏损总和"]),0.5) * np.power(testmatch["盈利交易数量"], 0.5)


# 显示训练集测试集的 spearman pearson 相关性.
myMT5Report.show_traintest_spearcorr(trainmatch, testmatch)

# 手工根据秩相关性从数据面板中研究 trainmatch, testmatch


#%% 自动选择
# 训练集根据sortby降序排序后，从中选择count个行，再根据chooseby选择前n个最大值，返回 trainchoose。
count = 0.5 # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好。
count = -1
sortby = mycriterion # insertname "盈亏比" "平均盈利"
chooseby = "TB"
trainchoose = myMT5Report.choose_opttrain_by2index(trainmatch=trainmatch, count=count, sortby=sortby, chooseby=chooseby, n=5)
trainchoose

# 选择的结果在测试集中所占的百分比位置
trainchoose = myMT5Report.choose_opttrain_by2index(trainmatch=trainmatch, testmatch=testmatch, count=count, sortby=sortby, chooseby=chooseby, n=5)
trainchoose

# 选择的结果不一定是5个中最大的tb，要看看最大的tb是否为全局最大的tb。然后再判断。根据自己的标准可以考虑第一个。
choosedmax = trainmatch[chooseby].max()
# 看看选择的结果中是否有最大的tb。
trainchoose[trainchoose["TB"] == choosedmax]













