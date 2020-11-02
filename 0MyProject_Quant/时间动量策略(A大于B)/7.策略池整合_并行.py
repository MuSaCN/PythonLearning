# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
mytime = MyTime.MyClass_Time()  # 时间类
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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 说明：
# 我们的思想是，不同组的策略参数可以看成不同的策略进行叠加。但是过滤的指标参数只能选择一个。
# 前面已经对一个品种、一个时间框、一个方向、一组参数进行了指标范围过滤和指标方向过滤。
# 这一步把这些结果整合到一起，形成策略池。
# 过滤后的结果选择 filter1 中的 sharpe_filter 最大值，即选择思想为过滤后的最大值。
# 由于前面对某些品种可能设置了条件，整合时注意要先判断对应的参数目录是否存在。
'''

#%%
# 策略参数名称，用于文档中解析参数 ******修改这里******
strategy_para_name = ["k", "holding", "lag_trade"]

#%%
symbol = "EURUSD"
# ---定位策略参数自动选择文档，获取各组参数 ******修改这里******
total_folder = __mypath__.get_desktop_path() + "\\_动量研究"
strat_file = total_folder + "\\策略参数自动选择\\{}\\{}.total.{}.xlsx".format(symbol, symbol, "filter1")   # 固定只分析 filter1
strat_filecontent = pd.read_excel(strat_file)
# ---解析，显然没有内容则直接跳过
for i in range(len(strat_filecontent)):  # i=0
    # ---解析文档
    # 获取各参数
    timeframe = strat_filecontent.iloc[i]["timeframe"]
    direct = strat_filecontent.iloc[i]["direct"]
    # 策略参数 ******修改这里******
    k = strat_filecontent.iloc[i][strategy_para_name[0]]
    holding = strat_filecontent.iloc[i][strategy_para_name[1]]
    lag_trade = strat_filecontent.iloc[i][strategy_para_name[2]]
    strat_para = [k, holding, lag_trade]
    # 输出的文档路径
    suffix = myBTV.string_strat_para(strategy_para_name, strat_para)

    # ---定位范围指标参数自动选择文档
    range_folder = total_folder + "\\范围指标参数自动选择\\{}.{}\\{}.{}".format(symbol,timeframe,direct,suffix)
    range_file = range_folder + "\\{}.filter1.xlsx".format(suffix) # 固定只分析 filter1
    # 检测文件是否存在，不存在则跳过
    if __mypath__.path_exists(range_file) == False:
        continue
    # 读取范围文档
    range_filecontent = pd.read_excel(range_file)
    range_filecontent.sort_values(by="sharpe_filter", ascending=False, inplace=True)

    range_filecontent.iloc[0]
    strat_filecontent.iloc[i]




    # ---定位方向指标参数自动选择文档
    direct_folder = total_folder + "\\方向指标参数自动选择\\{}.{}\\{}.{}".format(symbol,timeframe,direct,suffix)
    direct_file = direct_folder + "\\{}.filter1.xlsx".format(suffix)  # 固定只分析 filter1
    # 检测文件是否存在，不存在则跳过
    if __mypath__.path_exists(direct_file) == False:
        continue
    # 读取范围文档
    direct_filecontent = pd.read_excel(direct_file)




