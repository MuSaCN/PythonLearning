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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

#%%
import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[1990,1,1,0,0,0],[2020,11,24,0,0,0],index_time=True, col_capitalize=False)
eurusd1 = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2019,1,1,0,0,0],[2020,11,24,0,0,0],index_time=True, col_capitalize=False)

#%%
# ATR 平均真实波动指标(Oscillators类-幅图) Average True Range，返回Series。





#%%
# AMA 适应移动平均指标(Trend类-主图)，Adaptive Moving Average，返回Series。(！算法有迭代，必须一定数据后才相同)
price_arug = ["open","high","low","close"]
ama = myMT5Indi.AMA(eurusd,price_arug,10,2,30,0,"PRICE_OPEN")
ama1 = myMT5Indi.AMA(eurusd1,price_arug,10,2,30,0,"PRICE_OPEN")

# Alligator 鳄鱼指标(Bill Williams类-主图)，返回df：Jaws(13) Teeth(8) Lips(5)
price_arug = ["open","high","low","close"]
df = myMT5Indi.Alligator(eurusd,price_arug,13,8,8,5,5,3,InpMAMethod = "MODE_SMMA",InpAppliedPrice = "PRICE_MEDIAN")
df1 = myMT5Indi.Alligator(eurusd1,price_arug,13,8,8,5,5,3,InpMAMethod = "MODE_SMMA",InpAppliedPrice = "PRICE_MEDIAN")

# ADXW 韦尔达平均定向移动指数(Trend类-幅图), ADX Wilder, 返回df：ADX Wilder, +DI, -DI。(！算法有 SmoothedMA，必须一定数据后才相同)
price_arug = ["high","low","close"] # 顺序不能搞错
timeperiod=14
df = myMT5Indi.ADXW(eurusd,price_arug=price_arug,timeperiod=timeperiod)
df1 = myMT5Indi.ADXW(eurusd1,price_arug=price_arug,timeperiod=timeperiod)

# ADX 平均趋向指数(Trend类-幅图), Average Directional Movement Index, 返回df: ADX, +DI, -DI。(！算法有 ExponentialMA，必须一定数据后才相同)
price_arug = ["high","low","close"] # 顺序不能搞错
timeperiod=14
df = myMT5Indi.ADX(eurusd, price_arug=price_arug, timeperiod=timeperiod)
df1 = myMT5Indi.ADX(eurusd1, price_arug=price_arug, timeperiod=timeperiod)

# Accelerator 加速振荡指标(Bill Williams类-幅图)，Accelerator Oscillator，返回series.
price_arug = ["high","low"]
ac = myMT5Indi.Accelerator(eurusd, price_arug=price_arug)
ac1 = myMT5Indi.Accelerator(eurusd1, price_arug=price_arug)

# AD 累积/分配指标(Volumes类-幅图)，Accumulation/Distribution，返回series.(！算法有累加，必须用全数据段才相同)
price_arug = ["high","low","close","tick_volume"] # 顺序不能搞错
ad = myMT5Indi.AD(eurusd, price_arug=price_arug)
ad1 = myMT5Indi.AD(eurusd1, price_arug=price_arug)










