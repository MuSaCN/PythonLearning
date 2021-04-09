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
import warnings
warnings.filterwarnings('ignore')

file = __mypath__.get_desktop_path() + "\\ATR_test.xlsx" # ATR_test
# 读取报告。注意部分平仓不适合deal_standard = True修正。
strat_setting, strat_result, order_content, deal_content = myMT5Report.read_report_xlsx(filepath=file, deal_standard=False)

# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)
# 获取数据
data = myMT5Pro.getsymboldata(symbol,timeframe,timefrom, timeto,index_time=True, col_capitalize=True)

# 分析 orders、deals，先拆分为 BuyOnly、SellOnly，要分开分析。
order_buyonly, order_sellonly, deal_buyonly, deal_sellonly = myMT5Report.order_deal_split_buyonly_sellonly(order_content=order_content, deal_content=deal_content)


#%%
# 分析 deal_buyonly, deal_sellonly。从deal中获取交易单元(即对应 out 的 in)，生成 订单号和累计利润df.
# %timeit myMT5Report.get_unit_order1(deal_buyonly,order_buyonly) # 2.96 s ± 71.4 ms
# %timeit myMT5Report.get_unit_order(deal_buyonly,order_buyonly) # 2.23 s ± 37.5 ms

# 'Profit_Base' 表示基准仓位时的 利润
# 'NetProfit_Base' 表示基准仓位时的 净利润(除去了佣金和隔夜费)
# 'Balance_Base' 表示基准仓位时的 余额(净利润的累积和)
# 'Diff' 表示 out 的价格 - in 的价格差
# 'Diff_Point' 表示 out 的价格 - in 的价格差的点数
# 'Rate' 表示 价格收益率
unit_buyonly = myMT5Report.get_unit_order(deal_direct=deal_buyonly, order_direct=order_buyonly)
# unit_buyonly.set_index(keys="Time0", drop=False, inplace=True)
unit_sellonly = myMT5Report.get_unit_order(deal_direct=deal_sellonly, order_direct=order_sellonly)

# 符合MT5实际的资金曲线计算。
unit_buyonly["Balance_Base"].plot()
plt.show()
deal_content["Balance"][1:-1].plot()
plt.show()

# 回测框架以1为基准单位，算收益率
myDA.fin.r_to_price(unit_buyonly["Rate"]).plot()
plt.show()
unit_buyonly["Profit_Base"].cumsum().plot()
plt.show()



#%% #############################
# 根据 unit_order 把报告中的时间解析成 总数据 中的时间。因为报告中的时间太详细，我们定位到总数据中的时间框架中。
newtime_buyonly = myMT5Report.parse_unit_to_timenorm(unit_order=unit_buyonly, data=data)
newtime_sellonly = myMT5Report.parse_unit_to_timenorm(unit_sellonly, data)

#%%
# 计算下各方向下的各种指标。注意这里与回测中的计算有所不同，回测是以1单位算累计收益率。
# myBTV.__returns_result__()
# myBTV.__strat__()

Deposit = 10000
# p = myDA.fin.r_to_price(unit_buyonly["Rate"])
p = unit_buyonly["Balance_Base"] + Deposit
p.index = unit_buyonly["Time0"]
returns = unit_buyonly["Rate"]
returns.index = unit_buyonly["Time0"]

# 最大回撤与起初资金有关
maxDD = myDA.fin.calc_max_drawdown(p)
# 累计净利润
cumReturn = unit_buyonly["Balance_Base"].iloc[-1]
# CAGR复合年增长率/收益率(用这个替代年化收益率)
annRet = myDA.fin.calc_cagr(prices = p) if len(p) >= 2 else np.nan
# Calmar比率(年收益率/最大回撤)
calmar_ratio = myDA.fin.calc_calmar_ratio(prices = p) if len(p) >= 2 else np.nan


#%%
Deposit = 5000
alpha = 0.9
np.random.seed(0)
maxDD_list = []
for i in range(1000):
    balance = unit_buyonly["NetProfit_Base"].sample(frac=1).cumsum() + Deposit
    # balance.reset_index(drop=True,inplace=True)
    # balance.plot()
    # plt.show()
    maxDD = myDA.fin.calc_max_drawdown(balance)
    maxDD_list.append(maxDD)
#
maxDD_data = pd.Series(maxDD_list)
leftq = np.around(maxDD_data.quantile(q=(1-alpha)/2),4)
rightq = np.around(maxDD_data.quantile(q=alpha + (1-alpha)/2),4)
#
maxDD_data.hist(bins=20)
plt.axvline(x=leftq, color="red")
plt.annotate(s="{:.2f}%".format(leftq*100), xy=[leftq,0], xytext=[leftq,0])
plt.axvline(x=rightq, color="red")
plt.annotate(s="{:.2f}%".format(rightq*100), xy=[rightq,0], xytext=[rightq,0])
plt.show()








