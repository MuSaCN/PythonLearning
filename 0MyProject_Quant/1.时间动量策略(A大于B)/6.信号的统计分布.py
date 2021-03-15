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
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%%
strategy_para_name = ["k", "holding", "lag_trade"]
symbol_list = myMT5Pro.get_main_symbol_name_list()
total_folder = "F:\\工作---策略研究\\1.简单的动量反转\\_动量研究"
readfile_suffix = ".better"
holding_testcount = 10

para = ("EURUSD",)
symbol = para[0]  # symbol = "EURUSD"
print("%s 开始进行信号统计分布..." % symbol)
# ---定位策略参数自动选择文档，获取各组参数
in_file = total_folder + "\\策略参数自动选择\\{}\\{}.total.{}{}.xlsx".format(symbol, symbol, "filter1", readfile_suffix)  # 固定只分析 filter1
filecontent = pd.read_excel(in_file)

#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Continue"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)

#%%
# ---解析，显然没有内容则直接跳过
for i in range(len(filecontent)):  # i=0
    # ---解析文档
    # 获取各参数
    timeframe = filecontent.iloc[i]["timeframe"]
    direct = filecontent.iloc[i]["direct"]
    # 策略参数
    strat_para = [filecontent.iloc[i][strategy_para_name[j]] for j in
                  range(len(strategy_para_name))]
    # 满足3个标的指标都是递增才输出路径
    suffix = myBTV.string_strat_para(strategy_para_name, strat_para)
    # ---准备数据
    date_from, date_to = myMT5Pro.get_date_range(timeframe)
    data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
    # 展开holding参数
    holding_test = [holding for holding in range(1, holding_testcount + 1)]

    # ---
    for holding in holding_test:  # holding=1
        # 策略参数更换 -2位置的holding参数
        para_list = strat_para[0:-2] + [holding] + [strat_para[-1]]
        # 获取信号数据
        signal = stratgy_signal(data_train, para_list=para_list)
        # ---
        signal=signal["All"]
        price_DataFrame=data_train
        price_Series = data_train.Close
        holding=holding
        lag_trade=strat_para[-1]



        trade_signal = signal.shift(lag_trade)  # shift移动，以满足信号出现后下 lag_trade 期交易
        # ---获得持有期价格波动(持有期收益率)
        # if holding < 1:
        #     raise "SignalQuality(): holding参数不能小于1"

        # ---计算所有时间的最大的 做多正负波动 和 做空正负波动
        df_fluc = pd.DataFrame([], index=price_DataFrame.index) if price_DataFrame is not None else pd.DataFrame([], index=price_Series.index)
        if price_DataFrame is None and price_Series is not None :
            df_fluc["price"] = price_Series
            # 考虑未来的holding期波动，这里所以必须以holding+1滚动，且反向平移holding
            df_fluc["holding_Highest"] = price_Series.rolling(holding+1).max().shift(-holding)
            df_fluc["holding_Lowest"] = price_Series.rolling(holding + 1).min().shift(-holding)
            df_fluc["BuyPositive"] = (df_fluc["holding_Highest"] - df_fluc["price"])/df_fluc["price"]
            df_fluc["BuyNegative"] = (df_fluc["holding_Lowest"] - df_fluc["price"])/df_fluc["price"]
            df_fluc["SellPositive"] = -df_fluc["BuyNegative"]
            df_fluc["SellNegative"] = -df_fluc["BuyPositive"]
        elif price_DataFrame is not None:
            df_fluc[["Open","High","Low","Close"]] = price_DataFrame[["Open","High","Low","Close"]]
            # 考虑未来的holding期波动，这里是以开盘价作为入场点，本身算一期，所以以holding滚动，且反向平移holding-1
            df_fluc["holding_Highest"] = df_fluc["High"].rolling(holding).max().shift(-holding+1)
            df_fluc["holding_Lowest"] = df_fluc["Low"].rolling(holding).min().shift(-holding + 1)
            df_fluc["BuyPositive"] = (df_fluc["holding_Highest"] - df_fluc["Open"]) / df_fluc["Open"]
            df_fluc["BuyNegative"] = (df_fluc["holding_Lowest"] - df_fluc["Open"]) / df_fluc["Open"]
            df_fluc["SellPositive"] = -df_fluc["BuyNegative"]
            df_fluc["SellNegative"] = -df_fluc["BuyPositive"]

        # ---对做多做空积极消极做统计
        buy_positive = df_fluc["BuyPositive"][trade_signal==1]
        buy_negative = df_fluc["BuyNegative"][trade_signal==-1]
        sell_positive = df_fluc["SellPositive"][trade_signal==-1]
        sell_negative = df_fluc["SellNegative"][trade_signal==1]
        all_positive = buy_positive.combine_first(sell_positive)
        all_negative = buy_negative.combine_first(sell_negative)


























