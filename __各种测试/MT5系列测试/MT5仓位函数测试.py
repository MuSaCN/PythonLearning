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
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False) # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False) # 浮动杠杆仓位类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------


#%%
mypd.__init__(None)
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=True,symbol="EURUSD")
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=True,symbol="EURUSD",broker="FXTM",sets="FX Majors")

myMT5Lots_Fix.sysleverage
myMT5Lots_Dy.sysleverage

myMT5Lots_Fix.cumvalue_array
myMT5Lots_Fix.lever_array
myMT5Lots_Fix.stepvalue_array
myMT5Lots_Fix.margin_arr
myMT5Lots_Fix.cummargin

myMT5Lots_Dy.cumvalue_array
myMT5Lots_Dy.lever_array
myMT5Lots_Dy.stepvalue_array
myMT5Lots_Dy.margin_arr
myMT5Lots_Dy.cummargin


symbol = "GBPUSD"
myMT5Lots_Fix.check_symbol(symbol)
myMT5Lots_Fix.symbol_df[symbol].loc["pip_value"]

myMT5Lots_Dy.check_symbol(symbol)
myMT5Lots_Dy.symbol_df[symbol].loc["pip_value"]



#%%
myMT5Lots_Fix.lots_open_marginP(16631.21, "EURUSD", "ORDER_TYPE_BUY", 0.01, None, False)
myMT5Lots_Dy.lots_open_marginP(16631.21, "EURUSD", "ORDER_TYPE_BUY", 0.01, None, False)

myMT5Lots_Fix.lots_optlever(16631.21, "EURUSD", "ORDER_TYPE_BUY", 5, None, False)
myMT5Lots_Dy.lots_optlever(16631.21, "EURUSD", "ORDER_TYPE_BUY", 5, None, False)

myMT5Lots_Fix.lots_risk_percent(16631.21, "EURUSD", 0.1, 100, -1, False)
myMT5Lots_Dy.lots_risk_percent(16631.21, "EURUSD", 0.1, 100, -1, False)

myMT5Lots_Fix.diff_point("EURUSD",1,1200,-1)
myMT5Lots_Dy.diff_point("EURUSD",1,1200,-1)

myMT5Lots_Fix.diff_point_percent(16631.21, "EURUSD", "ORDER_TYPE_BUY", 0.01, None, True, -0.2, -1)
myMT5Lots_Dy.diff_point_percent(16631.21, "EURUSD", "ORDER_TYPE_BUY", 0.01, None, True, -0.2, -1)


#%%
myMT5Lots_Fix.lots_normalize(symbol, 1.256)
myMT5Lots_Dy.lots_normalize(symbol, 1.256)

# "EURUSD" "USDJPY" "AUDCAD" "CADCHF" "Brent"
myMT5Lots_Fix.check_symbol("USDJPY")
myMT5Lots_Fix.get_contract_size(action="ORDER_TYPE_BUY", symbol="USDJPY", volume=1, price=None)

myMT5Lots_Fix.check_symbol("AUDCAD")
myMT5Lots_Fix.get_contract_size(action="ORDER_TYPE_BUY", symbol="AUDCAD", volume=1, price=None)

myMT5Lots_Fix.get_spread(symbol,1,spread=-1)

myMT5Lots_Fix.standard_margin(0,"ORDER_TYPE_BUY","EURUSD",price=None)
myMT5Lots_Fix.standard_margin(0,"ORDER_TYPE_BUY","EURUSD",price=2)

myMT5Lots_Fix.__margin_open_fix__("ORDER_TYPE_BUY","EURUSD",1,None)

myMT5Lots_Dy.__get_every_stdmargin__("ORDER_TYPE_BUY","EURUSD",price=None)
myMT5Lots_Dy.__get_every_lots__("ORDER_TYPE_BUY","EURUSD",price=None)
myMT5Lots_Dy.__get_every_stdmargin_lots__("ORDER_TYPE_BUY","EURUSD",price=None)
stdmargin_arr, lots_arr = myMT5Lots_Dy.__get_every_stdmargin_lots__("ORDER_TYPE_BUY","EURUSD",price=None)

#%%
# "EURUSD" "USDJPY" "AUDCAD" "CADCHF" "Brent"
symbol = "EURUSD"
price = None

myMT5Lots_Dy.check_symbol(symbol)
myMT5Lots_Dy.__margin_open_dy__("ORDER_TYPE_BUY",symbol,50,price=price)
myMT5Lots_Dy.__margin_open_dy1__("ORDER_TYPE_BUY",symbol,50,price=price)
myMT5Lots_Dy.margin_open("ORDER_TYPE_BUY",symbol,50,price=price)


myMT5Lots_Dy.__lots_open_dy__("EURUSD", "ORDER_TYPE_BUY", 2000, None, True)
myMT5Lots_Dy.__lots_open_fix__("EURUSD", "ORDER_TYPE_BUY", 2000, None, True)
myMT5Lots_Dy.lots_open("EURUSD", "ORDER_TYPE_BUY", 2000, None, True)




