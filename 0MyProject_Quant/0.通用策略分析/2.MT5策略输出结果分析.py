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
myparallel = MyTools.MyClass_ParallelCal() # 并行运算类
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

file = __mypath__.get_desktop_path() + "\\test.xlsx" # ATR_test test
# 读取报告。注意部分平仓不适合deal_standard = True修正。
strat_setting, strat_result, order_content, deal_content = myMT5Report.read_report_xlsx(filepath=file, deal_standard=False)

# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)
# 获取数据
data = myMT5Pro.getsymboldata(symbol,timeframe,timefrom, timeto,index_time=True, col_capitalize=True)

# 分析 orders、deals，先拆分为 BuyOnly、SellOnly，要分开分析。
order_buyonly, order_sellonly, deal_buyonly, deal_sellonly = myMT5Report.order_deal_split_buyonly_sellonly(order_content=order_content, deal_content=deal_content)


# ---从 deal_direct, order_direct 中获取交易单元(根据out获取in)(整体算法)，生成交易in和out匹配单元信息df.

# %timeit myMT5Report.get_unit_order1(deal_buyonly,order_buyonly) # 2.96 s ± 71.4 ms
# %timeit myMT5Report.get_unit_order(deal_buyonly,order_buyonly) # 2.23 s ± 37.5 ms
unit_buyonly = myMT5Report.get_unit_order(deal_direct=deal_buyonly, order_direct=order_buyonly)
# unit_buyonly.set_index(keys="Time0", drop=False, inplace=True)
unit_sellonly = myMT5Report.get_unit_order(deal_direct=deal_sellonly, order_direct=order_sellonly)

# ---符合MT5实际的资金曲线计算。
# unit_buyonly["Balance_Base"].plot()
# plt.show()
# deal_content["Balance"][1:-1].plot()
# plt.show()

# ---回测框架以单位1为基准单位，算收益率
# myDA.fin.r_to_price(unit_buyonly["Rate"]).plot()
# plt.show()
# unit_buyonly["Profit_Base"].cumsum().plot()
# plt.show()

#%% # 不考虑仓位管理时的信息，以 收益率 或 基准仓位 算各项结果 以及 最佳仓位 f

# ---各项结果以及最佳仓位f
# 胜率；单位1满仓时的最大回撤；单位1满仓时的总收益率；基仓盈亏比；
# 凯利公式"保证金止损仓位"百分比；凯利公式"保证金占用仓位"杠杆；用历史回报法资金百分比；
win_rate, maxDD_nolots, return_nolots, pnl_ratio_base, f_kelly, f_lever, f_twr = myMT5Report.cal_result_no_money_manage(unit_buyonly)

text_base = "胜率={:.5f}\n信号总收益率={:.5f}\n信号最大回撤={:.5f}\n基仓盈亏比={:.5f}".format(win_rate, return_nolots, maxDD_nolots, pnl_ratio_base)
print(text_base)

# ---破产风险分析
# 假设盈亏比限定为2时，且 胜率 > 1/3 时，破产概率为：
# 破产风险，error=None：f为资金百分比；reward_rate报酬率(盈亏比) = 2或1 (不能为其他值)；报酬率为1时，win_rate要大于0.5，报酬率为2时，win_rate要大于 1/3 ；
myMoneyM.bankrupt_risk(win_rate, f_kelly, reward_rate=2) # f_kelly, f_twr
# 限定破产风险为指定值，得出最大的仓位比例f，error=None。
f_limit_bankrupt = myMoneyM.f_limit_bankrupt(win_rate, bankrupt_risk=0.1, reward_rate=2)


#%% ############
volume_min = myMT5.symbol_info(symbol)["volume_min"]
tick_value = myMT5.symbol_info(symbol)["trade_tick_value_profit"]
digits = myMT5.symbol_info(symbol)["digits"]
point = myMT5.symbol_info(symbol)["point"]



# 最差的一单
worst = unit_buyonly["NetProfit_Base"].min()
worst_point = np.abs(np.around(worst / volume_min / tick_value, 0))

# 以浮动杠杆来分析。
myMT5Lots_Dy.__init__(connect=True,symbol=symbol,broker="FXTM",sets="FX Majors")
init_deposit = 5000
used_percent = 0.2# 0.12
backtest_data = unit_buyonly[["NetProfit_Base","StopLossPoint"]].copy()

# ---原始
current_deposit = init_deposit
result_netprofit = []  # 记录每次模拟的净利润数组
for i, row in backtest_data.iterrows():
    # break
    used_equity = current_deposit * used_percent
    cur_lots = myMT5Lots_Dy.lots_risk_percent(fund=init_deposit, symbol=symbol, riskpercent=used_percent, stoplosspoint=row['StopLossPoint'], spread=0, adjust=True)
    cur_netprofit = row["NetProfit_Base"] * (cur_lots / volume_min)
    result_netprofit.append(cur_netprofit)
    current_deposit = current_deposit + cur_netprofit

ret, maxDD, pnl_ratio = myMT5Report.process_netprofit(result_netprofit, init_deposit=init_deposit,plot=True,show=True,ax=None,text_base=text_base)






#%%
# ---模拟
np.random.seed(0)
simulate_return = []
simulate_maxDD = []
simulate_pl_ratio = []
for i in range(1000):
    # ---
    simulate_data = backtest_data.sample(frac=1)
    current_deposit = init_deposit
    simulate_netprofit = [] # 记录每次模拟的净利润数组
    for i, row in simulate_data.iterrows():
        # break
        used_equity = current_deposit * used_percent
        cur_lots = myMT5Lots_Dy.lots_risk_percent(fund=used_equity,symbol=symbol,riskpercent=used_percent, stoplosspoint=worst_point, spread=0, adjust=True)
        cur_netprofit = row["NetProfit_Base"] * (cur_lots / volume_min)
        simulate_netprofit.append(cur_netprofit)
        current_deposit = current_deposit + cur_netprofit
    # ---
    # 资金曲线
    simulate_netprofit =  pd.Series(simulate_netprofit)
    simulate_balance = simulate_netprofit.cumsum() + init_deposit
    # 平均利润 strat_result.loc["Average profit trade:"]
    average_profit = result_netprofit[result_netprofit > 0].mean()
    # 平均亏损 strat_result.loc["Average loss trade:"]
    average_loss = result_netprofit[result_netprofit <= 0].mean()
    # 当前的收益
    simulate_return.append(simulate_balance.iloc[-1] / init_deposit)
    # 当前的最大回撤
    simulate_maxDD.append(myDA.fin.calc_max_drawdown(simulate_balance))
    # 盈亏比
    simulate_pl_ratio.append(np.abs(average_profit / average_loss))

# ---
simulate_return = pd.Series(simulate_return)
simulate_maxDD = pd.Series(simulate_maxDD)
simulate_pl_ratio = pd.Series(simulate_pl_ratio)


# ---画图
alpha=0.9 # 分布的分位概率

myfig.__init__(nrows=2, ncols=2, figsize=[1920,1080], AddFigure=True)
# 画原始顺序的走势图



# 画模拟结果的散点图
myplt.scatter(simulate_return,simulate_maxDD)



# 画模拟结果的最大回撤分布图
myplt.hist(simulate_maxDD, bins=50, objectname="simulate_maxDD", show=False)
maxDD_leftq = np.around(simulate_maxDD.quantile(q=(1 - alpha) / 2), 4)
maxDD_rightq = np.around(simulate_maxDD.quantile(q=alpha + (1 - alpha) / 2), 4)

plt.axvline(x=maxDD_leftq, color="red")
plt.annotate(s="{:.2f}%".format(maxDD_leftq * 100), xy=[maxDD_leftq, 0], xytext=[maxDD_leftq, 0], color="red")
plt.axvline(x=maxDD_rightq, color="red")
plt.annotate(s="{:.2f}%".format(maxDD_rightq * 100), xy=[maxDD_rightq, 0], xytext=[maxDD_rightq, 0], color="red")
plt.show()

# 画模拟结果的总收益分布图
myplt.hist(simulate_return, bins=50, objectname="simulate_return", show=False)
ret_leftq = np.around(simulate_return.quantile(q=(1 - alpha) / 2), 4)
ret_rightq = np.around(simulate_return.quantile(q=alpha + (1 - alpha) / 2), 4)
plt.axvline(x=ret_leftq, color="red")
plt.annotate(s="{:.2f}%".format(ret_leftq * 100), xy=[ret_leftq, 0], xytext=[ret_leftq, 0], color="red")
plt.axvline(x=ret_rightq, color="red")
plt.annotate(s="{:.2f}%".format(ret_rightq * 100), xy=[ret_rightq, 0], xytext=[ret_rightq, 0], color="red")
plt.show()



# return maxDD_leftq, maxDD_rightq, ret_leftq, ret_rightq






#%% #############################
# 根据 unit_order 把报告中的时间解析成 总数据 中的时间。因为报告中的时间太详细，我们定位到总数据中的时间框架中。
newtime_buyonly = myMT5Report.parse_unit_to_timenorm(unit_order=unit_buyonly, data=data)
newtime_sellonly = myMT5Report.parse_unit_to_timenorm(unit_sellonly, data)

#%% 计算下各方向下的各种指标。注意这里与向量化回测中的计算有所不同，向量化回测是以1单位算累计收益率。
# myBTV.__returns_result__()

Deposit = 10000
# p = myDA.fin.r_to_price(unit_buyonly["Rate"])
p = unit_buyonly["Balance_Base"] + Deposit
p.index = unit_buyonly["Time0"]
returns = unit_buyonly["Rate"]
returns.index = unit_buyonly["Time0"]

myBTV.__returns_result__(returns,benchmark_returns=0)



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
# # 累计净利润
cumReturn = unit_buyonly["Balance_Base"].iloc[-1]
# CAGR复合年增长率/收益率(用这个替代年化收益率)
annRet = myDA.fin.calc_cagr(prices = p) if len(p) >= 2 else np.nan
# Calmar比率(年收益率/最大回撤)
calmar_ratio = myDA.fin.calc_calmar_ratio(prices = p) if len(p) >= 2 else np.nan


#%% 无仓位管理，打乱收益，模拟最大回撤分布。
# 最大回撤以真实情况来计算，非单位1全额交易。
Deposit = 5000
alpha = 0.9
seed = 0
random_count = 1000
net_profit_series = unit_buyonly["NetProfit_Base"]

leftq, rightq = myMT5Report.maxDD_distribution(net_profit_series=net_profit_series, deposit=Deposit, alpha=alpha, seed=seed, random_count=random_count)




