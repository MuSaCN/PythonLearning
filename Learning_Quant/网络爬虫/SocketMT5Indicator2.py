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
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------


from MyPackage.Include.trade import Trade
from MyPackage.Include.tick import Tick
from MyPackage.Include.rates import Rates
from MyPackage.Include.indicator_connector import Indicator
import MetaTrader5 as Mt5

# You need this MQL5 service to use indicator:
# https://www.mql5.com/en/market/product/57574
indicator = Indicator()

trade = Trade('Example',  # Expert name
              0.1,  # Expert Version
              'EURUSD',  # symbol
              567,  # Magic number
              100.0,  # lot
              10,  # stop loss - 10 cents
              30,  # emergency stop loss - 30 cents
              10,  # take profit - 10 cents
              30,  # emergency take profit - 30 cents
              '9:15',  # It is allowed to trade after that hour. Do not use zeros, like: 09
              '17:30',  # It is not allowed to trade after that hour but let open all the position already opened.
              '17:50',  # It closes all the position opened. Do not use zeros, like: 09
              0.5,  # average fee
              )

time = 0
while True:

    # You need this MQL5 service to use indicator:
    # https://www.mql5.com/en/market/product/57574

    # Example of calling the same indicator with different parameters.
    stochastic_now = indicator.stochastic(symbol=trade.symbol, time_frame=Mt5.TIMEFRAME_M1)
    stochastic_past3 = indicator.stochastic(symbol=trade.symbol, time_frame=Mt5.TIMEFRAME_M1, start_position=3)

    moving_average = indicator.moving_average(symbol=trade.symbol, period=50)

    tick = Tick(trade.symbol)
    rates = Rates(trade.symbol, 1, 0, 1)

    # It uses "try" and catch because sometimes it returns None.
    try:

        # When in doubt how to handle the indicator, print it, it returns a Dictionary.
        # print(moving_average)
        # It prints:
        # {'symbol': 'PETR4', 'time_frame': 1, 'period': 50, 'start_position': 0, 'method': 0,
        # 'applied_price': 0, 'moving_average_result': 23.103}

        k_now = stochastic_now['k_result']
        d_now = stochastic_now['d_result']

        k_past3 = stochastic_now['k_result']
        d_past3 = stochastic_now['d_result']

        if tick.time_msc != time:
            # It is trading of the time frame of one minute.
            #
            # Stochastic logic:
            # To do the buy it checks if the K value at present is higher than the D value and
            # if the K at 3 candles before now was lower than the D value.
            # For the selling logic, it is the opposite of the buy logic.
            #
            # Moving Average Logic:
            # If the last price is higher than the Moving Average it allows to open a buy position.
            # If the last price is lower than the Moving Average it allows to open a sell position.
            #
            # To open a position this expert combines the Stochastic logic and Moving Average.
            # When Stochastic logic and Moving Average logic are true, it open position to the determined direction.

            # It is the buy logic.
            buy = (

                # Stochastic
                (
                        k_now > d_now
                        and
                        k_past3 < d_past3
                )

                and

                # Moving Average
                (
                        tick.last > moving_average['moving_average_result']
                )

            )  # End of buy logic.

            # -------------------------------------------------------------------- #

            # It is the sell logic.
            sell = (
                # Stochastic
                (
                        k_now < d_now
                        and
                        k_past3 > d_past3
                )

                and

                # Moving Average
                (
                        tick.last < moving_average['moving_average_result']
                )
            )  # End of sell logic.

            # -------------------------------------------------------------------- #

            # When buy or sell are true, it open a position.
            trade.open_position(buy, sell, 'Example Advisor Comment, the comment here can be seen in MetaTrader5')

    except TypeError:
        pass

    time = tick.time_msc

    if trade.days_end():
        trade.close_position('End of the trading day reached.')
        break

print('Finishing the program.')
print('Program finished.')





