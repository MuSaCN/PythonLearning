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
myMT5Report = MyMql.MyClass_StratTestReport() # MT5策略报告类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%%
file = __mypath__.get_desktop_path() + "\\test.xlsx"
# 读取报告
strat_setting, strat_result, order_content, deal_content = myMT5Report.read_report_xlsx(filepath=file)

# ---把 deal_content 的内容修正为基准仓位
# 根据deal的仓位计算倍数
multi = deal_content["Volume"][1:-1].astype(float) / 0.01
# 根据倍数修正下数据
deal_content["Volume"][1:-1] = deal_content["Volume"][1:-1].astype(float) / multi
deal_content["Commission"][1:-1] = deal_content["Commission"][1:-1] / multi
deal_content["Commission"].iloc[-1] /= 2
deal_content["Swap"][1:-1] = deal_content["Swap"][1:-1] / multi
deal_content["Swap"].iloc[-1] /= 2
deal_content["Profit"][1:-1] = deal_content["Profit"][1:-1] / multi
deal_content["Profit"].iloc[-1] /= 2
cum = deal_content["Commission"] + deal_content["Swap"] + deal_content["Profit"]
cum = cum[1:-1].cumsum()
deal_content["Balance"][1:-1] = deal_content["Balance"].iloc[0] + cum
deal_content["Balance"].iloc[-1] = deal_content["Balance"].iloc[-2]


# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)
# 获取数据
data = myMT5Pro.getsymboldata(symbol,timeframe,timefrom, timeto,index_time=True, col_capitalize=True)

# 分析 orders、deals，先拆分为 BuyOnly、SellOnly，要分开分析。
order_buyonly, order_sellonly, deal_buyonly, deal_sellonly = myMT5Report.order_deal_split_buyonly_sellonly(order_content, deal_content)


#%%
# 分析 deal_buyonly, deal_sellonly。从deal中获取交易单元(即 in 的累计Volume = out 的累计Volume)，生成 订单号和累计利润df.
# %timeit unit_buyonly = myMT5Report.get_deal_unit_order(deal_buyonly) # 497 ms ± 15.2 ms
unit_buyonly = myMT5Report.get_deal_unit_order(deal_direct=deal_buyonly)
unit_sellonly = myMT5Report.get_deal_unit_order(deal_direct=deal_sellonly)

# 拆分内容为 in 和 out 两部分，并整理成df输出。
df_buyonly = myMT5Report.to_in_out(unit_buyonly, deal_buyonly, order_buyonly)
df_sellonly = myMT5Report.to_in_out(unit_sellonly, deal_sellonly, order_sellonly)

# 把报告中的 时间df 解析成 总数据 中的时间，因为报告中的时间太详细，我们定位到总数据中的时间框架中。
newtime_buyonly = myMT5Report.parse_unit_to_timenorm(unit_buyonly, deal_buyonly, data)
newtime_sellonly = myMT5Report.parse_unit_to_timenorm(unit_sellonly, deal_sellonly, data)


#%%
# 计算下各方向下的各种指标：收益、回撤、...
# myBTV.__returns_result__()
# myBTV.__strat__()














