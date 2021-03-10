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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
说明：
# 由于优化结果被保存在硬盘，所以读取后解析参数和策略结果就可以进行分析。
# 在多个参数的情况下，为了分析需要把一些参数取固定值、另一些参数不取固定值。需要通过字典传递。
# 在分析最佳参数时，需要进行 单独测试 来观察图示。
# 注意使用的表格是基于训练集样本，与后面的分析用全集样本时的sharpe不一样。
'''


#%% 根据 非策略参数 定位文件 ###########################
import warnings
warnings.filterwarnings('ignore')

# direct_para = ["BuyOnly","SellOnly","All"]
symbol_list = ["AUDUSD"]
timeframe_list = ["TIMEFRAME_D1"]


#%% 根据 策略参数 分析 ############################
# ---画参数图1D
# k 动量向左参数；holding 必须小于 k
symbol = symbol_list[0]
timeframe = timeframe_list[0]
direct = "BuyOnly"  # 0-"BuyOnly", 1-"SellOnly", 2-"All"

# para_fixed = {"k":100, "holding":1, "lag_trade":None}
# para_fixed = {"k":None, "holding":1, "lag_trade":1}
# para_fixed = {"k":[0,400], "holding":1, "lag_trade":1}
para_fixed = {"k":112, "holding":[1,10], "lag_trade":1}

folder = "F:\\工作---策略研究\\简单的动量反转" + "\\_动量研究\\{}.{}".format(symbol, timeframe)
filepath = folder + "\\动量_{}.xlsx".format(direct)  # 选择训练集文件
filecontent = pd.read_excel(filepath)

y_name = ["sharpe"] # ["sharpe", "calmar_ratio", "cumRet"]

myBTV.plot_strat_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, output=False)

myBTV.auto_strat_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=30, filterlevel=0, plot=True, savefolder=None, batch=False)



#%%
# ---画参数图2D热力图
myDefault.set_backend_default("pycharm")
# k 动量向左参数；holding 必须小于 k
symbol = symbol_list[0]
timeframe = timeframe_list[0]
direct = "BuyOnly"  # 0-"BuyOnly", 1-"SellOnly", 2-"All"

para_fixed = {"k":None, "holding":None, "lag_trade":1}
para_fixed = {"k":[50,150], "holding":None, "lag_trade":1}
para_fixed = {"k":[50,150], "holding":[1,5], "lag_trade":1}
para_fixed = {"k":[18,50], "holding":1, "lag_trade":None}

folder = "F:\\工作---策略研究\\简单的动量反转" + "\\_动量研究\\{}.{}".format(symbol, timeframe)
filepath = folder + "\\动量_{}.xlsx".format(direct)  # 选择训练集文件
filecontent = pd.read_excel(filepath)

y_name = ["sharpe", "calmar_ratio", "cumRet"]
myBTV.plot_strat_para_2D_heatmap(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, output=False, annot=False)


#%%
# ---画参数图3D热力图
myDefault.set_backend_default("tkagg")
# k 动量向左参数；holding 必须小于 k
symbol = symbol_list[0]
timeframe = timeframe_list[0]
direct = "BuyOnly"  # 0-"BuyOnly", 1-"SellOnly", 2-"All"

para_fixed = {"k":None, "holding":None, "lag_trade":1}
para_fixed = {"k":[50,150], "holding":None, "lag_trade":1}
para_fixed = {"k":[50,150], "holding":[1,5], "lag_trade":1}
para_fixed = {"k":None, "holding":1, "lag_trade":None}

folder = "F:\\工作---策略研究\\简单的动量反转" + "\\_动量研究\\{}.{}".format(symbol, timeframe)
filepath = folder + "\\动量_{}.xlsx".format(direct)  # 选择训练集文件
filecontent = pd.read_excel(filepath)

y_name = ["sharpe", "calmar_ratio", "cumRet"]
myBTV.plot_strat_para_3D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, output=False)















