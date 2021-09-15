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
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
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
#------------------------------------------------------------

'''
单核算法本质是显示地循环，已被并行算法取代。
'''

#%%
########## 单次测试部分 #################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close   # 设定价格为考虑收盘价

#%%
holding = 1
k = 109
# 获取信号数据
signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
# 信号分析
outStrat, outSignal = myBTV.signal_quality(signaldata["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=1, plotRet=True, plotStrat=True)
myBTV.signal_quality_explain()


#%%
######### 优化部分 ##############
import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close   # 设定价格为考虑收盘价

#%%
# ---计算信号，仅分析做多信号
# 外部参数
k_end = 300
holding_end = 1

# 原始计算
import timeit
t0 = timeit.default_timer()

temp = 0 # 用于打印进度
result = pd.DataFrame() # 要放到外面
for k in range(1, k_end + 1):
    for holding in range(1, holding_end + 1):
        # 打印进度
        temp += 1
        print("\r", "{}/{}".format(temp, k_end * holding_end), end="", flush=True)
        # 退出条件
        if holding > k: continue
        # 获取信号数据
        signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
        # 信号分析
        outStrat, outSignal = myBTV.signal_quality(signaldata["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=1, plotRet=False, plotStrat=False)
        # 设置信号统计
        out = outStrat["BuyOnly"]
        winRate = out["winRate"]
        cumRet = out["cumRet"]
        sharpe = out["sharpe"]
        maxDD = out["maxDD"]
        count = out["TradeCount"]
        out["k"] = k
        out["holding"] = holding
        # ---
        if  cumRet > 0 and sharpe > 0 and sharpe < 0.5:
            result = result.append(out, ignore_index=True)

t1 = timeit.default_timer()
print("\n","耗时为：",t1 - t0) # 耗时为： 363.2092888
result


#%%
###########读取文档，用于分析训练集和测试集
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
price = eurusd.Close   # 设定价格为考虑收盘价

#%%
folder = __mypath__.get_desktop_path() + "\\__动量研究__"
filename = "动量_Buy.xlsx"
trainpath = folder + "\\" + filename
paranames = ["k","holding","lag_trade"] # 顺序不能搞错了

# ---解析参数
paralist = myBTV.parse_opt_xlsx(trainpath, paranames)

#%%
temp = 0
result = pd.DataFrame()  # 要放到里面
for para in paralist:
    k = para[0]
    holding = para[1]
    lag_trade = para[2]
    trade_direct = para[3]  # "BuyOnly","SellOnly","All"
    # 不同交易方向下，数据字符串索引
    if trade_direct == "BuyOnly":
        sig_mode, signalname, tradename = "BuyOnly", "buysignal", "BuyOnly"
    elif trade_direct == "SellOnly":
        sig_mode, signalname, tradename = "SellOnly", "sellsignal", "SellOnly"
    elif trade_direct == "All":
        sig_mode, signalname, tradename = "All", "allsignal", "AllTrade"
    # 打印进度
    temp += 1
    print("\r", "{}/{}".format(temp, len(paralist)), end="", flush=True)
    # 退出条件
    if holding > k: continue
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata[signalname], price_DataFrame=eurusd, holding=holding, lag_trade=lag_trade, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat[tradename]
    cumRet = out["cumRet"]
    sharpe = out["sharpe"]
    maxDD = out["maxDD"]
    out["k"] = k
    out["holding"] = holding
    out["lag_trade"] = lag_trade
    # ---
    if cumRet > 0 and sharpe > 0 and maxDD < 0.5:
        result = result.append(out, ignore_index=True)
result

filename_test = filename.rsplit(".")[0]+"_测试集"+"."+filename.split(".")[1]
result.to_excel(folder+"\\"+filename_test)

#%%
# ---合并两个数据
testpath = folder + "\\" + filename_test
myBTV.concat_opt_xlsx(trainpath, testpath, paranames)




