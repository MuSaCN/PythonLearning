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

mypd.__init__(None,None)
#%%
import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2022,12,5,0,0,0],index_time=True, col_capitalize=True)
eurusd1 = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2019,1,1,0,0,0],[2022,11,27,0,0,0],index_time=True, col_capitalize=True)


#%%
# 测试：有价格参数版本/无价格参数版本
# price_arug = ["Open", "High", "Low", "Close", "Tick_volume"]
# price_arug = ["Open", "High", "Low", "Close"]
# data = eurusd[price_arug]
# price = myMT5Indi.applied_price(eurusd, price_arug=price_arug, mode=InpAppliedPrice)
# func = myMT5Indi.ma_method_func(mode=InpMAMethod)

#%%
# Custom类

# Donachian_Channel 唐奇安通道，返回series.
price_arug = ["High", "Low"]
df = myMT5Indi.Donachian_Channel(eurusd, price_arug=price_arug, timeperiod=20)


#%% 基础运算测试 %timeit
lwma =  myMT5Indi.LinearWeightedMA(series=eurusd.Close, timeperiod=20) # 298 ms ± 8.34 ms
# 1.196566 1.180928
smooma = myMT5Indi.SmoothedMA(eurusd.Close, timeperiod=20)
ema = myMT5Indi.ExponentialMA(eurusd.Close, timeperiod=20)
sma = myMT5Indi.SimpleMA(eurusd.Close, timeperiod=20)

#%%
# Bill Williams类
# Accelerator 加速振荡指标(Bill Williams类-幅图)，Accelerator Oscillator，返回series.
price_arug = ["High","Low"]
ac = myMT5Indi.Accelerator(eurusd, price_arug=price_arug)
ac1 = myMT5Indi.Accelerator(eurusd1, price_arug=price_arug)
myMT5Indi.get_bill_williams(eurusd,"Accelerator")

# AO 动量震荡指标(Bill Williams类-幅图) Awesome Oscillator，返回Series。
price_arug=["High","Low"]
ao = myMT5Indi.Awesome_Oscillator(eurusd,price_arug)
ao1 = myMT5Indi.Awesome_Oscillator(eurusd1,price_arug)
myMT5Indi.get_bill_williams(eurusd,"Awesome_Oscillator")

# Alligator 鳄鱼指标(Bill Williams类-主图)，返回df：Jaws(13) Teeth(8) Lips(5)
price_arug = ["Open","High","Low","Close"]
df = myMT5Indi.Alligator(eurusd,price_arug,13,8,8,5,5,3,InpMAMethod = "MODE_SMMA",InpAppliedPrice = "PRICE_MEDIAN")
df1 = myMT5Indi.Alligator(eurusd1,price_arug,13,8,8,5,5,3,InpMAMethod = "MODE_SMMA",InpAppliedPrice = "PRICE_MEDIAN")
myMT5Indi.get_bill_williams(eurusd,"Alligator",13,8,8,5,5,3,"MODE_SMMA","PRICE_MEDIAN")

# Fractals 比尔威廉姆分形指标(Bill Williams类-主图)，Fractals，返回df：Up, Down。
price_arug = ["High", "Low"]
df = myMT5Indi.Fractals(eurusd,price_arug)
df1 = myMT5Indi.Fractals(eurusd1,price_arug)
myMT5Indi.get_bill_williams(eurusd,"Fractals")

# Gator 鳄鱼振荡器指标(Bill Williams类-幅图)，Gator Oscillator，返回df：Up, Down
price_arug = ["Open", "High", "Low", "Close"]
df = myMT5Indi.Gator(eurusd,price_arug,13,8,8,5,5,3,"MODE_SMMA","PRICE_MEDIAN")
df1 = myMT5Indi.Gator(eurusd1,price_arug,13,8,8,5,5,3,"MODE_SMMA","PRICE_MEDIAN")
myMT5Indi.get_bill_williams(eurusd,"Gator",13,8,8,5,5,3,"MODE_SMMA","PRICE_MEDIAN")

# MarketFacilitationIndex 市场便利指标(Bill Williams类-幅图)，Market Facilitation Index，返回Series.
price_arug = ["High", "Low", "Tick_volume"]
market_facilitation_index = myMT5Indi.MarketFacilitationIndex(eurusd,price_arug,0.00001)
market_facilitation_index1 = myMT5Indi.MarketFacilitationIndex(eurusd1,price_arug,0.00001)
myMT5Indi.get_bill_williams(eurusd,"MarketFacilitationIndex",0.00001)


#%%
# Oscillators类
# ATR 平均真实波动指标(Oscillators类-幅图) Average True Range，返回Series。(！算法有迭代，必须一定数据后才相同)
price_arug = ["High", "Low", "Close"]
atr = myMT5Indi.ATR(eurusd,price_arug,14)
atr1 = myMT5Indi.ATR(eurusd1,price_arug,14)
myMT5Indi.get_oscillators(eurusd,"ATR",14)

# Bears 熊市指标(Oscillators类-幅图) Bears Power，返回Series.(！算法有 ExponentialMA，必须一定数据后才相同)
price_arug = ["Low", "Close"]
bears = myMT5Indi.Bears(eurusd,price_arug,13)
bears1 = myMT5Indi.Bears(eurusd1,price_arug,13)
myMT5Indi.get_oscillators(eurusd,"Bears",13)

# Bulls 牛市指标(Oscillators类-幅图) Bulls Power，返回Series.(！算法有 ExponentialMA，必须一定数据后才相同)
price_arug = ["High", "Close"]
bulls = myMT5Indi.Bulls(eurusd,price_arug,13)
bulls1 = myMT5Indi.Bulls(eurusd1,price_arug,13)
myMT5Indi.get_oscillators(eurusd,"Bulls",13)

# CCI 顺势指标(Oscillators类-幅图)，Commodity Channel Index，返回Series。
price_arug = ["Open","High","Low","Close"]
cci = myMT5Indi.CCI(eurusd,price_arug,14,"PRICE_TYPICAL")
cci1 = myMT5Indi.CCI(eurusd1,price_arug,14,"PRICE_TYPICAL")
myMT5Indi.get_oscillators(eurusd,"CCI",14,"PRICE_TYPICAL")

# CHO 蔡金摆动指标(Oscillators类-幅图)，Chaikin Oscillator，返回Series。(！算法有迭代，必须一定数据后才相同)
price_arug=["High","Low","Close","Tick_volume"]
cho = myMT5Indi.CHO(eurusd,price_arug,3,10,"MODE_EMA")
cho1 = myMT5Indi.CHO(eurusd1,price_arug,3,10,"MODE_EMA")
myMT5Indi.get_oscillators(eurusd,"CHO",3,10,"MODE_EMA")

# DeMarker DeMarker指标(Oscillators类-幅图)，DeMarker，返回Series。
price_arug = ["High", "Low"]
demarker = myMT5Indi.DeMarker(eurusd,price_arug,14)
demarker1 = myMT5Indi.DeMarker(eurusd1,price_arug,14)
myMT5Indi.get_oscillators(eurusd,"DeMarker",14)

# Force_Index 强力指数指标(Oscillators类-幅图)，Force Index，返回Series。
price_arug = ["Open", "High", "Low", "Close", "Tick_volume"]
force = myMT5Indi.Force_Index(eurusd,price_arug,13,"MODE_SMA","PRICE_CLOSE")
force1 = myMT5Indi.Force_Index(eurusd1,price_arug,13,"MODE_SMA","PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"Force_Index",13,"MODE_SMA","PRICE_CLOSE")

# MACD 移动平均数聚/散指标(有@)(Oscillators类-幅图)(有EMA算法，需要一定数据)，MACD，返回df：MACD, Signal.
price_arug = ["Open", "High", "Low", "Close"]
df = myMT5Indi.MACD(eurusd,price_arug,12,26,9,"PRICE_CLOSE")
df1 = myMT5Indi.MACD(eurusd1,price_arug,12,26,9,"PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"MACD",12,26,9,"PRICE_CLOSE")

# Momentum 动量指标(Oscillators类-幅图)，Momentum，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
momentum = myMT5Indi.Momentum(eurusd,price_arug,14,"PRICE_CLOSE")
momentum1 = myMT5Indi.Momentum(eurusd1,price_arug,14,"PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"Momentum",14,"PRICE_CLOSE")

# OsMA 移动平均震荡指标(Oscillators类-幅图)(有@OsMA)(有EMA算法，需要一定数据)，Moving Average of Oscillator，返回Series.
price_arug = ["Open", "High", "Low", "Close"]
osma = myMT5Indi.OsMA(eurusd,price_arug,12,26,9,"PRICE_CLOSE")
osma1 = myMT5Indi.OsMA(eurusd1,price_arug,12,26,9,"PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"OsMA",12,26,9,"PRICE_CLOSE")

# RSI 相对强弱指数(Oscillators类-幅图)(有@RSI)(有迭代，需要一定数据)，Relative Strength Index，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
rsi = myMT5Indi.RSI(eurusd,price_arug,14,"PRICE_CLOSE")
rsi1 = myMT5Indi.RSI(eurusd1,price_arug,14,"PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"RSI",14,"PRICE_CLOSE")

# RVI 相对活力指数(Oscillators类-幅图)，Relative Vigor Index，返回df：RVI, Signal.
price_arug = ["Open", "High", "Low", "Close"]
rvi=myMT5Indi.RVI(eurusd,price_arug,10)
rvi1=myMT5Indi.RVI(eurusd1,price_arug,10)
myMT5Indi.get_oscillators(eurusd,"RVI",10)

# Stochastic 随机摆动指标(Oscillators类-幅图)，Stochastic Oscillator，返回df：Main, Signal.
price_arug = ["High", "Low", "Close"]
stochastic = myMT5Indi.Stochastic(eurusd,price_arug,5,3,3)
stochastic1 = myMT5Indi.Stochastic(eurusd1,price_arug,5,3,3)
myMT5Indi.get_oscillators(eurusd,"Stochastic",5,3,3)

# TRIX 三倍指数移动平均数振荡指标(Oscillators类-幅图)(有@TRIX)(有EMA算法，需要一定数据)，Triple Exponential Average，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
trix = myMT5Indi.TRIX(eurusd,price_arug,14,"PRICE_CLOSE")
trix1 = myMT5Indi.TRIX(eurusd1,price_arug,14,"PRICE_CLOSE")
myMT5Indi.get_oscillators(eurusd,"TRIX",14,"PRICE_CLOSE")

# WPR 威廉指数指标(Oscillators类-幅图)，William's Percent Range，返回Series。
price_arug = ["High", "Low", "Close"]
wpr = myMT5Indi.WPR(eurusd,price_arug,14)
wpr1 = myMT5Indi.WPR(eurusd1,price_arug,14)
myMT5Indi.get_oscillators(eurusd,"WPR",14)

#%% %timeit
# Trend
# ADX 平均趋向指数(Trend类-幅图), Average Directional Movement Index, 返回df: ADX, +DI, -DI。(！算法有 ExponentialMA，必须一定数据后才相同)(注意，ADX，ADXW 在参数大时，与MT5结果不一样，因为无理数ema精度不一样)
price_arug = ["High","Low","Close"] # 顺序不能搞错
df = myMT5Indi.ADX(dataframe=eurusd, price_arug=price_arug, timeperiod=14)
df1 = myMT5Indi.ADX(eurusd1, price_arug=price_arug, timeperiod=14)
myMT5Indi.get_trend(eurusd,"ADX",99)

# AMA 适应移动平均指标(Trend类-主图)，Adaptive Moving Average，返回Series(效率不高)。(！算法有迭代，必须一定数据后才相同)
price_arug = ["Open","High","Low","Close"]
# %timeit ama = myMT5Indi.AMA(eurusd,price_arug,10,2,30,0,"PRICE_OPEN") # 445 ms ± 8.37 ms
# 2000-12-15 0.892700   2020-11-19 1.178909   2020-12-04    1.198273  # 129 ms ± 2.29 ms
ama = myMT5Indi.AMA(eurusd,price_arug,10,2,30,0,"PRICE_OPEN") # 698 ms ± 11.4 ms
ama1 = myMT5Indi.AMA(eurusd1,price_arug,10,2,30,0,"PRICE_OPEN")
myMT5Indi.get_trend(eurusd,"AMA",10,2,30,0,"PRICE_OPEN")


# ADXW 韦尔达平均定向移动指数(Trend类-幅图), ADX Wilder, 返回df：ADX Wilder, +DI, -DI。(！算法有 SmoothedMA，必须一定数据后才相同)(注意，ADX，ADXW 在参数大时，与MT5结果不一样，因为无理数ema精度不一样)
price_arug = ["High","Low","Close"] # 顺序不能搞错
df = myMT5Indi.ADXW(eurusd,price_arug=price_arug,timeperiod=14)
df1 = myMT5Indi.ADXW(eurusd1,price_arug=price_arug,timeperiod=14)
myMT5Indi.get_trend(eurusd,"ADXW",14)

# BB 布林带指标(Trend类-主图) Bollinger Bands，返回df：Middle, Upper, Lower
price_arug = ["Open","High","Low","Close"]
df = myMT5Indi.BB(eurusd, price_arug, 20, 2, "PRICE_CLOSE")
df1 = myMT5Indi.BB(eurusd1, price_arug, 20, 2, "PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"BB",20,2,"PRICE_CLOSE")

# MA 移动平均数指标(Trend类-主图)，Moving Average，返回Series。
price_arug = ["Open","High","Low","Close"]
ma = myMT5Indi.MA(eurusd,price_arug,13,"PRICE_CLOSE","MODE_SMMA")
ma1 = myMT5Indi.MA(eurusd1,price_arug,13,"PRICE_CLOSE","MODE_SMMA")
myMT5Indi.get_trend(eurusd,"MA",13,"PRICE_CLOSE","MODE_EMA")

# DEMA 双指数移动平均线指标(Trend类-主图)(算法有双EMA，要一定数据后才相同，有@)，Double Exponential Moving Average，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
dema = myMT5Indi.DEMA(eurusd,price_arug,14,0,"PRICE_CLOSE")
dema1 = myMT5Indi.DEMA(eurusd1,price_arug,14,0,"PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"DEMA",14,0,"PRICE_CLOSE")

# Envelopes 轨道线指标(Trend类-主图)，Envelopes，返回df：Upper, Lower。
price_arug = ["Open", "High", "Low", "Close"]
df = myMT5Indi.Envelopes(eurusd,price_arug,14,0,0.1,"PRICE_CLOSE","MODE_SMA")
df1 = myMT5Indi.Envelopes(eurusd1,price_arug,14,0,0.1,"PRICE_CLOSE","MODE_SMA")
myMT5Indi.get_trend(eurusd,"Envelopes",14,0,0.1,"PRICE_CLOSE","MODE_SMA")

# FrAMA 分形学适应移动平均指标(Trend类-主图)(效率不高)，Fractal Adaptive Moving Average，返回Series。(！算法有迭代，必须一定数据后才相同)
price_arug = ["Open", "High", "Low", "Close"]
# %timeit frama = myMT5Indi.FrAMA(eurusd,price_arug,14,0,"PRICE_CLOSE") # 451 ms ± 12 ms
# 2000-12-04 0.884500   2020-11-19 1.179224   2020-12-04 1.202853 # 176 ms ± 4.14 ms
frama = myMT5Indi.FrAMA(eurusd,price_arug,14,0,"PRICE_CLOSE") # 687 ms ± 5.94 ms
frama1 = myMT5Indi.FrAMA(eurusd1,price_arug,14,0,"PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"FrAMA",14,0,"PRICE_CLOSE")

# Ichimoku 一目均衡图指标(Trend类-主图)，Ichimoku Kinko Hyo，返回df：TenKan-Sen(9), Kijun-sen(26), Senkou Span A, Senkou Span B(52), Chikou Span
price_arug = ["High", "Low", "Close"]
df = myMT5Indi.Ichimoku(eurusd,price_arug,9,26,53)
df1 = myMT5Indi.Ichimoku(eurusd1,price_arug,9,26,53)
myMT5Indi.get_trend(eurusd,"Ichimoku",9,26,52)

# ParabolicSAR 抛物转向系统指标(Trend类-主图)，Parabolic SAR，返回Series。
price_arug = ["High", "Low"]
sar = myMT5Indi.ParabolicSAR(eurusd,price_arug,0.02,0.2)
sar1 = myMT5Indi.ParabolicSAR(eurusd1,price_arug,0.02,0.2)
myMT5Indi.get_trend(eurusd,"ParabolicSAR",0.02,0.2)

# StdDev 标准偏差指标(Trend类-幅图)(除"MODE_SMA"外，其他效率不高)，Standard Deviation，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
stdde = myMT5Indi.StdDev(eurusd,price_arug,20,0,"MODE_SMA", "PRICE_CLOSE")
stdde1 = myMT5Indi.StdDev(eurusd1,price_arug,20,0,"MODE_SMA", "PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"StdDev",20,0,"MODE_SMA","PRICE_CLOSE")

# TEMA 三倍指数移动平均指标(Trend类-主图)(有@，需要前置数据)，Triple Exponential Moving Average，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
tema = myMT5Indi.TEMA(eurusd,price_arug,14,0,"PRICE_CLOSE")
tema1 = myMT5Indi.TEMA(eurusd1,price_arug,14,0,"PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"TEMA",14,0,"PRICE_CLOSE")

# VIDYA 变量指数动态平均数指标(Trend类-主图)(有@，需要前置数据)(效率不高)，Variable Index Dynamic Average，返回Series。
price_arug = ["Open", "High", "Low", "Close"]
# 2020-12-04 1.192259   2020-11-19 1.176575   2000-12-19 0.895600   2000-12-04 0.884500
# %timeit vidya = myMT5Indi.VIDYA(eurusd,price_arug,9,12,0,"PRICE_CLOSE") # 746 ms ± 59.4 ms
vidya = myMT5Indi.VIDYA(eurusd,price_arug,9,12,0,"PRICE_CLOSE") # 176 ms ± 2.93 ms
vidya1 = myMT5Indi.VIDYA(eurusd1,price_arug,9,12,0,"PRICE_CLOSE")
myMT5Indi.get_trend(eurusd,"VIDYA",9,12,0,"PRICE_CLOSE")


#%%
# Volumes
# AD 累积/分配指标(Volumes类-幅图)，Accumulation/Distribution，返回series.(！算法有累加，必须用全数据段才相同)
price_arug = ["High","Low","Close","Tick_volume"] # 顺序不能搞错
ad = myMT5Indi.AD(eurusd, price_arug=price_arug)
ad1 = myMT5Indi.AD(eurusd1, price_arug=price_arug)
myMT5Indi.get_volumes(eurusd,"AD")

# MFI 资金流向指标(Volumes类-幅图)，Money Flow Index，返回Series。
price_arug = ["High", "Low", "Close", "Tick_volume"]
mfi = myMT5Indi.MFI(eurusd,price_arug,14)
mfi1 = myMT5Indi.MFI(eurusd1,price_arug,14)
myMT5Indi.get_volumes(eurusd,"MFI",14)

# OBV 平衡交易量指标(Volumes类-幅图)，On Balance Volume，返回Series。(！必须用全数据段才相同)
price_arug = ["Close", "Tick_volume"]
obv = myMT5Indi.OBV(eurusd,price_arug)
obv1 = myMT5Indi.OBV(eurusd1,price_arug)
myMT5Indi.get_volumes(eurusd,"OBV")



