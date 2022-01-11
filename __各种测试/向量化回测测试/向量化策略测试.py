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
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,12,4,0,0,0],[2020,12,5,0,0,0],index_time=True, col_capitalize=True)

direct = "BuyOnly" # "SellOnly" "All"
direct = "SellOnly"
direct = "All"

#%%
# ---DailyRange交叉动量策略。当价格与上轨金叉，做多；当价格与下轨死叉，做空。基于DailyRange指标，其中 DailyRange 以当日open为标的，+- 上个日线的 range*n 波动。(PS: 注意该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。)
result1 = myBTV.stra.dailyrange_cross_momentum(eurusd, 1.0)
a = result1[direct][result1[direct]!=0]

# ---DailyRange交叉反转策略。当价格与上轨金叉，做空；当价格与下轨死叉，做多。(PS: 注意该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。)
result2 = myBTV.stra.dailyrange_cross_reverse(eurusd, 1.0)
b = result2[direct][result2[direct]!=0]

a[a != -b]

# ---DailyRange交叉反转策略(先突破再交叉)。当价格与下轨金叉，做多；当价格与上轨死叉，做空。(PS: 注意该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。)
result3 = myBTV.stra.dailyrange_break_cross_reverse(eurusd, 0.1)
c = result3[direct][result3[direct]!=0]




#%%
# ---海龟交叉动量策略。当价格与上轨金叉，做多；当价格与下轨死叉，做空。(PS: 注意该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。)
result = myBTV.stra.turtle_cross_momentum(eurusd,20,price_arug=["High", "Low", "Close"])
a = result[direct][result[direct]!=0]

# ---海龟交叉反转策略。当价格与上轨金叉，做空；当价格与下轨死叉，做多。(注意：因为唐奇安通道的性质，无法用策略：当价格与上轨死叉，做空；当价格与下轨金叉，做多。)(PS: 注意该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。)
result = myBTV.stra.turtle_cross_reverse(eurusd,20,price_arug=["High", "Low", "Close"])
b = result[direct][result[direct]!=0]

a[a != -b]

#%%



