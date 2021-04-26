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
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
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
myMT5Report = MyMql.MyClass_StratTestReport()  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%% 单独测试
mypd.__init__(None) # None 0
# 以浮动杠杆和固定杠杆来分析。
myMoneyM.lots_FixedIncrement_SplitFund(6000,5000,delta=100,init_lots=0.03,min_lots=0.01)
myMoneyM.lots_FixedIncrement_SplitFormula(6000,5000,delta=100,init_lots=0.03,min_lots=0.01)

a = myMoneyM.lots_FixedIncrement_SplitFormula(6000,5000,delta=100,init_lots=0.03,min_lots=0.01)

init=5000
delta = 100
init_lots = 0.01
out = []
for n in range(0,50):
    a = myMoneyM.lots_FixedIncrement_SplitFund(init+n*delta, init, delta=delta, init_lots=init_lots, min_lots=0.01)
    out.append(a)
out=pd.Series(out)
out.plot()
plt.show()


#%%
import warnings
warnings.filterwarnings('ignore')

file = __mypath__.get_desktop_path() + "\\ATR_test.xlsx" # ATR_test test
# 读取报告，加载品种信息到 self.symbol_df。注意部分平仓不适合deal_standard = True修正。
strat_setting, strat_result, order_content, deal_content = myMT5Report.read_report_xlsx(filepath=file, deal_standard=False)

# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)
# 获取数据
data = myMT5Pro.getsymboldata(symbol,timeframe,timefrom, timeto,index_time=True, col_capitalize=True)

# 分析 orders、deals，先拆分为 BuyOnly、SellOnly，要分开分析。
order_buyonly, order_sellonly, deal_buyonly, deal_sellonly = myMT5Report.order_deal_split_buyonly_sellonly(order_content=order_content, deal_content=deal_content)

# ---从 deal_direct, order_direct 中获取交易单元(根据out获取in)(整体算法)，生成交易in和out匹配单元信息df.
unit_buyonly = myMT5Report.get_unit_order(deal_direct=deal_buyonly, order_direct=order_buyonly)
# unit_buyonly.set_index(keys="Time0", drop=False, inplace=True)
unit_sellonly = myMT5Report.get_unit_order(deal_direct=deal_sellonly, order_direct=order_sellonly)

#%% # 不考虑仓位管理时的信息，以 收益率 或 基准仓位 算各项结果 以及 最佳仓位 f
# ---各项结果以及最佳仓位f
# 胜率；单位1满仓时的最大回撤；单位1满仓时的总收益率；基仓盈亏比；
# 凯利公式"保证金止损仓位"百分比；凯利公式"保证金占用仓位"杠杆；用历史回报法资金百分比；
result_base, best_f = myMT5Report.cal_result_no_money_manage(unit_buyonly)
text_base = result_base.to_string()
print(text_base)

#%% 测试仓位比例
volume_min = myMT5Report.symbol_df[symbol]["volume_min"]
tick_value = myMT5Report.symbol_df[symbol]["trade_tick_value_profit"]
digits = myMT5Report.symbol_df[symbol]["digits"]
point = myMT5Report.symbol_df[symbol]["point"]

# 以浮动杠杆和固定杠杆来分析。
myMT5Lots_Dy.__init__(connect=True,symbol=symbol,broker="FXTM",sets="FX Majors")
myMT5Lots_Fix.__init__(connect=True,symbol=symbol)

# ---
init_deposit = 10000

# --- # myMT5Report
worst = unit_buyonly["NetProfit_Base"].min()
worst_point = myMT5Report.worst_point(unit_buyonly)
maxDDr = myMT5Report.basic_max_down_range(unit_buyonly)

# 初始化的仓位
init_lots = myMT5Lots_Dy.lots_risk_percent(fund=init_deposit, symbol=symbol, riskpercent=0.1, stoplosspoint=worst_point, spread=0, adjust=True)
funcmode = "SplitFund" # "SplitFund" / "SplitFormula"
delta_list = [i for i  in range(10,200,5)]
out = pd.DataFrame()
for delta in delta_list:
    temp_out = myMT5Report.backtest_with_lots_FixedIncrement(myMT5Lots_Dy, unit_order=unit_buyonly, backtest_data=None, init_deposit=init_deposit, delta=delta, init_lots=init_lots, funcmode=funcmode, plot=False, show=False, ax=None)
    out = out.append([temp_out])

out.index = delta_list

# 除去无法交易的和爆仓的，很重要
out = out[out["count"]==len(unit_buyonly)]
out.drop("count",axis=1).plot() # out["deposit_sharpe"].plot()
plt.show()



# ---单独调试
# 设置固定增长的delta
delta = maxDDr/2 # maxDDr/2 np.abs(worst)*2
# funcmode = "SplitFund" / "SplitFormula"
myMT5Report.backtest_with_lots_FixedIncrement(myMT5Lots_Dy, unit_order=unit_buyonly, backtest_data=None, init_deposit=init_deposit, delta=delta, init_lots=init_lots, funcmode="SplitFund", plot=True, show=True, ax=None)
myMT5Report.backtest_with_lots_FixedIncrement(myMT5Lots_Dy, unit_order=unit_buyonly, backtest_data=None, init_deposit=init_deposit, delta=delta, init_lots=init_lots, funcmode="SplitFormula", plot=True, show=True, ax=None)





#%% 蒙特卡罗模拟 # 按顺序并不能说明太多内容，所以打乱净利润再重新回测。




# maxDD_leftq = np.around(simulate_maxDD.quantile(q=(1 - alpha) / 2), 4)
# maxDD_rightq = np.around(simulate_maxDD.quantile(q=alpha + (1 - alpha) / 2), 4)
# ret_leftq = np.around(simulate_return.quantile(q=(1 - alpha) / 2), 4)
# ret_rightq = np.around(simulate_return.quantile(q=alpha + (1 - alpha) / 2), 4)
# plr_leftq = np.around(simulate_pl_ratio.quantile(q=(1 - alpha) / 2), 4)
# plr_rightq = np.around(simulate_pl_ratio.quantile(q=alpha + (1 - alpha) / 2), 4)
