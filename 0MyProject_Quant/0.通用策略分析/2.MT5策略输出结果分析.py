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
mylogging = MyDefault.MyClass_Default_Logging(activate=False) # 日志记录类，需要放在上面才行
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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator() # MT5指标Python版
myMT5Report = MyMql.MyClass_StratTestReport() # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False) # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False) # 浮动杠杆仓位类
myMoneyM = MyTrade.MyClass_MoneyManage() # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------


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
# unit_buyonly["Balance_Base"].plot()
# plt.show()
# deal_content["Balance"][1:-1].plot()
# plt.show()

# 回测框架以单位1为基准单位，算收益率
# myDA.fin.r_to_price(unit_buyonly["Rate"]).plot()
# plt.show()
# unit_buyonly["Profit_Base"].cumsum().plot()
# plt.show()

#%%
mypd.__init__(0)
strat_result

# 以基准仓位算各项结果
# 胜率要以净利润算。不要字符串解析，win_rate = strat_result.loc["Profit Trades (% of total):"]
win_rate = (unit_buyonly["NetProfit_Base"] > 0).sum() / 104
# 平均利润 strat_result.loc["Average profit trade:"]
average_profit = unit_buyonly["NetProfit_Base"][unit_buyonly["NetProfit_Base"] > 0].mean()
# 平均亏损 strat_result.loc["Average loss trade:"]
average_loss = unit_buyonly["NetProfit_Base"][unit_buyonly["NetProfit_Base"] <= 0].mean()
# 几何平均利润率
rate_profit = unit_buyonly["Rate"][unit_buyonly["Rate"] > 0]
rate_profit = rate_profit + 1
grate_profit = stats.gmean(rate_profit) - 1 # 计算=0.0222 表格=0.0221
# 几何平均亏损率
rate_loss = unit_buyonly["Rate"][unit_buyonly["Rate"] <= 0]
rate_loss = rate_loss + 1
grate_loss = 1 - stats.gmean(rate_loss) # 计算=0.0109 表格= 0.011
# 凯利公式结果
f_kelly = myMoneyM.kelly_losslot_percent(win_rate, average_profit, average_loss) # 计算=0.129 表格=0.13
lever = myMoneyM.kelly_occupylot_lever(win_rate, grate_profit, grate_loss) # 计算=11.504 表格=11.1

# 用历史回报法求 使"最终财富比值TWR"最大的"资金百分比f"("保证金止损仓位")
profit_series = unit_buyonly["NetProfit_Base"] # NetProfit_Base Profit_Base Rate
f_twr, max_twr = myMoneyM.terminal_wealth_relative(profit_series, bounds=(0,1)) # f = 0.19377305347158474

# 假设报酬率限定为2时，且 胜率 > 1/3 时，破产概率为：
win_rate
# 实际报酬率，盈亏比
np.abs(average_profit / average_loss)
# 破产风险，error=None：f为资金百分比；reward_rate报酬率(盈亏比) = 2或1 (不能为其他值)；报酬率为1时，win_rate要大于0.5，报酬率为2时，win_rate要大于 1/3 ；
myMoneyM.bankrupt_risk(win_rate, f_kelly, reward_rate=2) # f_kelly, f_twr
# 限定破产风险为指定值，得出最大的仓位比例f，error=None。
myMoneyM.f_limit_bankrupt(win_rate, bankrupt_risk=0.1, reward_rate=2)




#%% #############################
# 根据 unit_order 把报告中的时间解析成 总数据 中的时间。因为报告中的时间太详细，我们定位到总数据中的时间框架中。
newtime_buyonly = myMT5Report.parse_unit_to_timenorm(unit_order=unit_buyonly, data=data)
newtime_sellonly = myMT5Report.parse_unit_to_timenorm(unit_sellonly, data)

#%% 计算下各方向下的各种指标。注意这里与向量化回测中的计算有所不同，向量化回测是以1单位算累计收益率。
# myBTV.__returns_result__()
# myBTV.__strat__()

Deposit = 10000
# p = myDA.fin.r_to_price(unit_buyonly["Rate"])
p = unit_buyonly["Balance_Base"] + Deposit
p.index = unit_buyonly["Time0"]
returns = unit_buyonly["Rate"]
returns.index = unit_buyonly["Time0"]

# MT5上夏普比，它反应的是持仓时间的算术平均盈利与其标准方差的比率。无风险比率, 此处也考虑从相应的银行存款资金获得的利润。经过源码的阅读，MT5夏普比也是通过价格转换收益率计算，但是Python上计算的结果与实际MT5结果有差别。
# 无杠杆单位1占用资金曲线的夏普比
myDA.fin.calc_risk_return_ratio(returns) # 0.18956866861486457，MT5是 0.129751
returns.mean() / returns.std()
# 下面结果更接近MT5夏普比
profit = unit_buyonly["NetProfit_Base"] # NetProfit_Base Profit_Base Profit Diff Diff_Point
profit.mean() / profit.std() # 0.1269115668443189
#
# temp = deal_content[1:-1]
# profit = temp["Profit"]+temp["Commission"]+temp["Swap"] # Balance
# profit.mean() / profit.std(ddof=0)

# 最大回撤与起初资金有关(最大回撤以真实情况来计算，非单位1全额交易.)
maxDD = myDA.fin.calc_max_drawdown(p)
# 累计净利润
cumReturn = unit_buyonly["Balance_Base"].iloc[-1]
# CAGR复合年增长率/收益率(用这个替代年化收益率)
annRet = myDA.fin.calc_cagr(prices = p) if len(p) >= 2 else np.nan
# Calmar比率(年收益率/最大回撤)
calmar_ratio = myDA.fin.calc_calmar_ratio(prices = p) if len(p) >= 2 else np.nan


#%% 无仓位管理，打乱收益，模拟最大回撤分布。
# 最大回撤以真实情况来计算，非单位1全额交易。
Deposit = 5000
alpha = 0.9
np.random.seed(0)
maxDD_list = []
for i in range(1000):
    # 最大回撤以真实情况来计算，非单位1全额交易。
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








