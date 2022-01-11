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
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

mypd.__init__(0)

#%%
myMT5.__init__(connect=True)
myMT5.version
myMT5.account_info
myMT5.terminal_info
myMT5.last_error()

#%%
myMT5.symbols_total()
myMT5.symbols_get("EURUSD")
myMT5.symbol_info("EURUSD")["volume_min"]
myMT5.symbol_info_tick("EURUSD")

#%%
myMT5.symbol_select("USDCNH", False)

myMT5.market_book_add("EURUSD")
myMT5.market_book_get("EURUSD")
myMT5.market_book_release("EURUSD")

#%%
# 获取活动挂单的数量
myMT5.orders_total()
myMT5.orders_get(symbol="EURUSD")
myMT5.orders_get(group="*USD*")
myMT5.orders_get(ticket=2248047902)

# 获取未结持仓的数量。
myMT5.positions_total()
myMT5.positions_get(symbol="EURUSD")
myMT5.positions_get(group="*USD*")
myMT5.positions_get(ticket=2248047631)

# 返回预付款（用账户货币表示）来执行指定的交易操作。
myMT5.order_calc_margin("ORDER_TYPE_BUY","EURUSD",0.01)
myMT5.order_calc_margin("ORDER_TYPE_SELL","EURUSD",0.01)

# 返回指定交易操作的盈利（用账户货币表示）。
myMT5.order_calc_profit("ORDER_TYPE_BUY","EURUSD", 1, 1, 2.0)
myMT5.order_calc_profit("ORDER_TYPE_BUY","EURUSD", 1, 1, None, 100)
myMT5.order_calc_profit("ORDER_TYPE_BUY","EURUSD", 1, None, 1.18)
myMT5.order_calc_profit("ORDER_TYPE_BUY","EURUSD", 1, 1.17, 1.18)
myMT5.order_calc_profit("ORDER_TYPE_SELL","EURUSD", 1,  None, None, 100)


#%% 未完
myMT5.mt5.history_deals_total()

from datetime import datetime
# 获取历史中的交易数量
from_date = datetime(2020, 1, 1)
to_date = datetime.now()
deals = myMT5.mt5.history_deals_total(from_date, to_date)
if deals > 0:
    print("Total deals=", deals)
print("Deals not found in history")
deals = myMT5.mt5.history_deals_get(from_date, to_date)
