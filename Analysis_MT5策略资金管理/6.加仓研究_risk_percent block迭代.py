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
myini = MyFile.MyClass_INI()  # ini文件操作类
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
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------


#%%
import warnings
warnings.filterwarnings('ignore')

file = r"F:\工作(同步)\工作---资金管理\1.简单的动量策略\EURUSD.D1\filter=1 atr=1 mul=1.1.xlsx" # ATR_test test
# 读取报告，加载品种信息到 self.symbol_df。注意部分平仓不适合deal_standard = True修正。
strat_setting, strat_result, dict_order_content, dict_deal_content = myMT5Report.read_report_xlsx(filepath=file, result_vert=True, deal_standard=False, onlytestsymbol=False)

# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)

# 设置为指定品种的内容
order_content = dict_order_content[symbol]
deal_content = dict_deal_content[symbol]

# 把 order_content 和 deal_content 解析成 unit_order。返回 unit_buyonly, unit_sellonly。
unit_buyonly, unit_sellonly = myMT5Report.content_to_direct_unit_order(order_content, deal_content)

# 不考虑仓位管理时的信息，以 收益率 或 基准仓位 算各项结果 以及 最佳仓位 f
# ---各项结果以及最佳仓位f
# 胜率；单位1满仓时的最大回撤；单位1满仓时的总收益率；基仓盈亏比；
# 凯利公式"保证金止损仓位"百分比；凯利公式"保证金占用仓位"杠杆；用历史回报法资金百分比；
base = myMT5Report.cal_result_no_money_manage(unit_order=unit_buyonly)
result_base = base[0]
best_f = base[1]
best_delta = base[2]
text_base = result_base.to_string(float_format="%0.4f")
print(text_base)


#%% #############################
# 获取数据，当前时间框或降低级别
timebar_timeframe =  "TIMEFRAME_H4"
data = myMT5Pro.getsymboldata(symbol,timebar_timeframe,timefrom, timeto,index_time=True, col_capitalize=True)

# 把 unit_order 订单按 data 的时间框拆分为多个子订单块。用于对原订单进行分阶段仓位管理，比如加减仓。
'''
# 未拆分的一单基仓利润 = (block["DiffProfit_Base"] + block["JumpProfit_Base"]).sum()
# 拆单无法整合进 Profit_Base。连续持仓情况下 Profit_Base = DiffProfit_Base + JumpProfit_Base。但是要注意：新仓要减去所在的跳空利润
# MT5手续费 Commission_Base 在一个单子的开仓和平仓都收。且计算一次就行了。拆单无法整合进手续费
# 隔夜仓费 Swap_Base 要单独算，因为时间跳会有不同结果。拆单无法整合进隔夜仓费。
# 思考：
    # 拆分情况下，可以同时存在多个单。算仓位百分比时是否需要利润兑现，才能考虑？
    # 加仓后，一直保持状态。还是加仓后，条件外再减仓，条件内再重新加仓？
'''
all_block_buyonly, newtime_buy = myMT5Report.parse_unit_to_ticket_block(unit_order=unit_buyonly, data=data)
all_block_sellonly,newtime_sell = myMT5Report.parse_unit_to_ticket_block(unit_sellonly, data)

block = all_block_buyonly[all_block_buyonly["SplitOrder0"] == 2]

def print_order(df, order):
    if df["SplitOrder0"].iloc[0] == order:
        print(df)
# groupby_buy.apply(print_order, order=2)

# 注意3倍隔夜费的星期几的数量，若3倍隔夜费是周三，表示周三到周四隔夜仓会收取3倍隔夜费。
i=1
t0 = unit_buyonly.iloc[i]["Time0"]
t1 = unit_buyonly.iloc[i]["Time1"]
unit_buyonly.iloc[i]["Swap_Base"]
myMT5Report.swap_base(t0,t1,symbol,long_or_short="long")



#%% (淘汰)子订单以block迭代模式：以时间结构加仓，注意原策略是移动止损，所以一定存在尾部回撤。加仓结果并不好。
### 可读性好点，但是groupby.apply 速度要慢于 apply
myMT5Lots_Dy.__init__(connect=True,symbol=symbol,broker="FXTM",sets="FX Majors")
init_deposit = 5000
init_percent = 0.1
add_percent = 0
add_index = 20
stoplosspoint = "StopLossPoint" # "StopLossPoint" "worst_point"
volume_min = myMT5Lots_Dy.symbol_df[symbol]["volume_min"] # 注意别忘记要除以它
commission_base = unit_buyonly["Commission_Base"][0] # 开仓和平仓时才收取，这里以block考虑，初始计算一次就行。

result_netprofit = []  # 记录每次模拟的净利润数组
result_deposit_rate = []  # 记录资金波动率
current_deposit = [init_deposit] # 用于apply传输数据
# unit_buyonly["StopLossPoint"]
# ---以block迭代，暂不考虑交易成本，用于 groupby.apply()
def block_profit(block): # block = all_block_buyonly[all_block_buyonly["SplitOrder0"]==4]
    slpoint = block.iloc[0]["StopLossPoint"] # 每个block的 "StopLossPoint" 都一样
    # ---初始化仓位，占全部 unit
    init_lots = myMT5Lots_Dy.lots_risk_percent(fund=current_deposit[0], symbol=symbol, riskpercent=init_percent, stoplosspoint=slpoint, spread=0, adjust=True) # 0.4
    # print(init_lots, "; slpoint=",slpoint)
    # ---以时间结构加仓，从 add_index 开始
    add_lots = myMT5Lots_Dy.lots_risk_percent(fund=current_deposit[0], symbol=symbol, riskpercent=add_percent,stoplosspoint=slpoint, spread=0, adjust=True)
    cur_netprofit = [0]
    init_profit = [0]
    add_profit = [0]
    # apply()函数
    def cal_cur_netprofit(row):
        index = row["index"]
        # ---初始化仓位
        if index == 0: # 开仓 不考虑跳空利润
            init_profit[0] = init_lots * (row["DiffProfit_Base"] + commission_base)/ volume_min
            cur_netprofit[0] += init_profit[0]
        elif index > 0:
            init_profit[0] = init_lots * (row["DiffProfit_Base"] + row["JumpProfit_Base"]) / volume_min
            cur_netprofit[0] += init_profit[0]
        # ---加仓
        # if index == add_index: # 开仓 不考虑跳空利润
        #     add_profit[0] = add_lots * (row["DiffProfit_Base"]+commission_base) / volume_min
        #     cur_netprofit[0] += add_profit[0]
        # elif index > add_index:
        #     add_profit[0] = add_lots * (row["DiffProfit_Base"] + row["JumpProfit_Base"]) / volume_min
        #     cur_netprofit[0] += add_profit[0]
    _ = block.apply(cal_cur_netprofit, axis=1)
    # 添加block的swap隔夜费
    swap = myMT5Report.swap_base(block["SplitTime0"].iloc[0],block["SplitTime1"].iloc[len(block)-1],symbol,long_or_short="long") * init_lots / volume_min
    # print(block["SplitTime0"].iloc[0], block["SplitTime1"].iloc[len(block)-1], init_lots, swap)
    # if add_index <= len(block)-1:
    #     swap_add = myMT5Report.swap_base(block["SplitTime0"].iloc[add_index],block["SplitTime1"].iloc[len(block)-1],symbol,long_or_short="long") * add_lots / volume_min
    #     swap += swap_add
    cur_netprofit[0] = cur_netprofit[0] + swap
    # ---添加和更新结果
    result_netprofit.append(cur_netprofit[0])
    deposit_rate = cur_netprofit[0] / current_deposit[0]  # current_deposit
    result_deposit_rate.append(deposit_rate)
    current_deposit[0] = current_deposit[0] + cur_netprofit[0]
# block = all_block_buyonly[all_block_buyonly["SplitOrder0"]==2]
# block_profit(block)

groupby_buy = all_block_buyonly.groupby(by="SplitOrder0",axis=0)
_ = groupby_buy.apply(block_profit) # 776 ms ± 24.2 ms # 512 ms ± 11.8 ms

# ---处理净利润结果 # myMT5Report
out = myMT5Report.__process_result__(result_netprofit=result_netprofit, result_deposit_rate=result_deposit_rate, init_deposit=init_deposit, plot=True, show=True, ax=None, text_base=text_base)
# result_netprofit[0] = 1140.8000000000038

result_netprofit[:10]

#%%
# ---用于比较
result_out = myMT5Report.backtest_with_lots_risk_percent(lots_class_case=myMT5Lots_Dy, unit_order=unit_buyonly, backtest_data=None,init_deposit=init_deposit,used_percent=0.1,stoplosspoint=stoplosspoint, plot=True, show=True, ax=None)
result_out = myMT5Report.backtest_with_lots_risk_percent(lots_class_case=myMT5Lots_Dy, unit_order=unit_buyonly, backtest_data=None,init_deposit=init_deposit,used_percent=0.2,stoplosspoint=stoplosspoint, plot=True, show=True, ax=None)


